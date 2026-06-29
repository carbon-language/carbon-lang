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

fn TestIterate[R:! Core.Iterate where .ElementType = i32](r: R) {
  var sum: i32 = 0;

  //@dump-sem-ir-begin
  for (i: i32 in r) {
    sum += i;
  }
  //@dump-sem-ir-end
}

fn TestIterateDriver(var m: Cpp.MutableRange, c: Cpp.ConstRange) {
  TestIterate(m);
  TestIterate(c);
}

fn TestRangeFor(var m: Cpp.MutableRange, c: Cpp.ConstRange) {
  var sum: i32 = 0;

  //@dump-sem-ir-begin
  for (i: i32 in m) {
    sum += i;
  }
  //@dump-sem-ir-end

  //@dump-sem-ir-begin
  for (i: i32 in c) {
    sum += i;
  }
  //@dump-sem-ir-end
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

fn TestIterate[R:! Core.Iterate where .ElementType = Cpp.ValueType](var r: R) {
  var sum: Cpp.ValueType = Cpp.ValueType.ValueType();

  //@dump-sem-ir-begin
  for (i: Cpp.ValueType in r) {
    sum += i;
  }
  //@dump-sem-ir-end
}

fn TestIterateDriver(var m: Cpp.MutableRange, c: Cpp.ConstRange) {
  TestIterate(m);
  TestIterate(c);
}

fn TestRangeFor(var m: Cpp.MutableRange, c: Cpp.ConstRange) {
  var sum: Cpp.ValueType = Cpp.ValueType.ValueType();

  //@dump-sem-ir-begin
  for (i: Cpp.ValueType in m) {
    sum += i;
  }

  for (i: Cpp.ValueType in c) {
    sum += i;
  }
  //@dump-sem-ir-end
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

fn TestIterate[R:! Core.Iterate where .ElementType = f64](r: R) {
  var sum: f64 = 0.0;

  //@dump-sem-ir-begin
  for (i: f64 in r) {
    sum += i;
  }
  //@dump-sem-ir-end
}

fn TestIterateDriver(c: Cpp.N.ConstRange) {
  TestIterate(c);
}

fn TestRangeFor(c: Cpp.N.ConstRange) {
  var sum: f64 = 0;

  //@dump-sem-ir-begin
  for (i: f64 in c) {
    sum += i;
  }
  //@dump-sem-ir-end
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

fn TestIterate[R:! Core.Iterate where .ElementType = f64](r: R) {
  var sum: f64 = 0.0;

  //@dump-sem-ir-begin
  for (i: f64 in r) {
    sum += i;
  }
  //@dump-sem-ir-end
}

fn TestIterateDriver(var m: Cpp.N.MutableRange) {
  // CHECK:STDERR: fail_todo_adl_returns_same_types_mutable.carbon:[[@LINE+28]]:3: error: facet type `Core.CppRangeForIterate where .(Core.CppRangeForIterate.Iterator) impls Core.Destroy and .(Core.CppRangeForIterate.Iterator) impls Core.Copy and .(Core.CppRangeForIterate.Iterator) impls Core.CppUnsafeDeref and .(Core.CppRangeForIterate.Iterator) impls Core.EqWith(.(Core.CppRangeForIterate.Sentinel)) and .(Core.CppRangeForIterate.Iterator) impls Core.Inc and .(Core.CppRangeForIterate.Iterator).(Core.CppUnsafeDeref.Result) impls Core.Destroy and .(Core.CppRangeForIterate.Iterator).(Core.CppUnsafeDeref.Result) impls Core.Copy and .(Core.CppRangeForIterate.Sentinel) impls Core.Destroy and .(Core.CppRangeForIterate.Sentinel) impls Core.Copy` can not be identified [ImplLookupInUnidentifiedFacetType]
  // CHECK:STDERR:   TestIterate(m);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_adl_returns_same_types_mutable.carbon: note: value expression passed to reference parameter [ValueForRefParam]
  // CHECK:STDERR: fail_todo_adl_returns_same_types_mutable.carbon:[[@LINE-21]]:34: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR:   friend auto begin(MutableRange&) -> Iterator;
  // CHECK:STDERR:                                  ^
  // CHECK:STDERR: {{.*}}/prelude/iterate.carbon:39:3: note: while building thunk to match the signature of this function [ThunkSignature]
  // CHECK:STDERR:   fn Begin(self) -> Iterator;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_adl_returns_same_types_mutable.carbon:[[@LINE-21]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn TestIterate[R:! Core.Iterate where .ElementType = f64](r: R) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_todo_adl_returns_same_types_mutable.carbon:[[@LINE+14]]:3: error: facet type `Core.CppRangeForIterate where .(Core.CppRangeForIterate.Iterator) impls Core.Destroy and .(Core.CppRangeForIterate.Iterator) impls Core.Copy and .(Core.CppRangeForIterate.Iterator) impls Core.CppUnsafeDeref and .(Core.CppRangeForIterate.Iterator) impls Core.EqWith(.(Core.CppRangeForIterate.Sentinel)) and .(Core.CppRangeForIterate.Iterator) impls Core.Inc and .(Core.CppRangeForIterate.Iterator).(Core.CppUnsafeDeref.Result) impls Core.Destroy and .(Core.CppRangeForIterate.Iterator).(Core.CppUnsafeDeref.Result) impls Core.Copy and .(Core.CppRangeForIterate.Sentinel) impls Core.Destroy and .(Core.CppRangeForIterate.Sentinel) impls Core.Copy` can not be identified [ImplLookupInUnidentifiedFacetType]
  // CHECK:STDERR:   TestIterate(m);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_adl_returns_same_types_mutable.carbon: note: value expression passed to reference parameter [ValueForRefParam]
  // CHECK:STDERR: fail_todo_adl_returns_same_types_mutable.carbon:[[@LINE-34]]:32: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR:   friend auto end(MutableRange&) -> Iterator;
  // CHECK:STDERR:                                ^
  // CHECK:STDERR: {{.*}}/prelude/iterate.carbon:40:3: note: while building thunk to match the signature of this function [ThunkSignature]
  // CHECK:STDERR:   fn End(self) -> Sentinel;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_adl_returns_same_types_mutable.carbon:[[@LINE-35]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn TestIterate[R:! Core.Iterate where .ElementType = f64](r: R) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  TestIterate(m);
}

fn TestRangeFor(var m: Cpp.N.MutableRange) {
  var sum: f64 = 0;

  //@dump-sem-ir-begin
  for (i: f64 in m) {
    sum += i;
  }
  //@dump-sem-ir-end
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

fn TestIterate[R:! Core.Iterate where .ElementType = Cpp.N.ValueType](r: R) {
  var sum: Cpp.N.ValueType = Cpp.N.ValueType.ValueType();

  //@dump-sem-ir-begin
  for (i: Cpp.N.ValueType in r) {
    sum += i;
  }
  //@dump-sem-ir-end
}

fn TestIterateDriver(c: Cpp.N.ConstRange) {
  TestIterate(c);
}

fn TestRangeFor(c: Cpp.N.ConstRange) {
  var sum: Cpp.N.ValueType = Cpp.N.ValueType.ValueType();

  //@dump-sem-ir-begin
  for (i: Cpp.N.ValueType in c) {
    sum += i;
  }
  //@dump-sem-ir-end
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

fn TestIterate[R:! Core.Iterate where .ElementType = Cpp.N.ValueType](r: R) {
  var sum: Cpp.N.ValueType = Cpp.N.ValueType.ValueType();

  //@dump-sem-ir-begin
  for (i: Cpp.N.ValueType in r) {
    sum += i;
  }
  //@dump-sem-ir-end
}

fn TestIterateDriver(var m: Cpp.N.MutableRange) {
  // CHECK:STDERR: fail_todo_adl_returns_different_types_mutable.carbon:[[@LINE+42]]:3: error: facet type `Core.CppRangeForIterate where .(Core.CppRangeForIterate.Iterator) impls Core.Destroy and .(Core.CppRangeForIterate.Iterator) impls Core.Copy and .(Core.CppRangeForIterate.Iterator) impls Core.CppUnsafeDeref and .(Core.CppRangeForIterate.Iterator) impls Core.EqWith(.(Core.CppRangeForIterate.Sentinel)) and .(Core.CppRangeForIterate.Iterator) impls Core.Inc and .(Core.CppRangeForIterate.Iterator).(Core.CppUnsafeDeref.Result) impls Core.Destroy and .(Core.CppRangeForIterate.Iterator).(Core.CppUnsafeDeref.Result) impls Core.Copy and .(Core.CppRangeForIterate.Sentinel) impls Core.Destroy and .(Core.CppRangeForIterate.Sentinel) impls Core.Copy` can not be identified [ImplLookupInUnidentifiedFacetType]
  // CHECK:STDERR:   TestIterate(m);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_adl_returns_different_types_mutable.carbon: note: value expression passed to reference parameter [ValueForRefParam]
  // CHECK:STDERR: fail_todo_adl_returns_different_types_mutable.carbon:[[@LINE-21]]:34: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR:   friend auto begin(MutableRange&) -> Iterator;
  // CHECK:STDERR:                                  ^
  // CHECK:STDERR: {{.*}}/prelude/iterate.carbon:39:3: note: while building thunk to match the signature of this function [ThunkSignature]
  // CHECK:STDERR:   fn Begin(self) -> Iterator;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_adl_returns_different_types_mutable.carbon:[[@LINE-21]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn TestIterate[R:! Core.Iterate where .ElementType = Cpp.N.ValueType](r: R) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_todo_adl_returns_different_types_mutable.carbon:[[@LINE+28]]:3: error: facet type `Core.CppRangeForIterate where .(Core.CppRangeForIterate.Iterator) impls Core.Destroy and .(Core.CppRangeForIterate.Iterator) impls Core.Copy and .(Core.CppRangeForIterate.Iterator) impls Core.CppUnsafeDeref and .(Core.CppRangeForIterate.Iterator) impls Core.EqWith(.(Core.CppRangeForIterate.Sentinel)) and .(Core.CppRangeForIterate.Iterator) impls Core.Inc and .(Core.CppRangeForIterate.Iterator).(Core.CppUnsafeDeref.Result) impls Core.Destroy and .(Core.CppRangeForIterate.Iterator).(Core.CppUnsafeDeref.Result) impls Core.Copy and .(Core.CppRangeForIterate.Sentinel) impls Core.Destroy and .(Core.CppRangeForIterate.Sentinel) impls Core.Copy` can not be identified [ImplLookupInUnidentifiedFacetType]
  // CHECK:STDERR:   TestIterate(m);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_adl_returns_different_types_mutable.carbon: note: value expression passed to reference parameter [ValueForRefParam]
  // CHECK:STDERR: fail_todo_adl_returns_different_types_mutable.carbon:[[@LINE-34]]:32: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR:   friend auto end(MutableRange&) -> Sentinel;
  // CHECK:STDERR:                                ^
  // CHECK:STDERR: {{.*}}/prelude/iterate.carbon:40:3: note: while building thunk to match the signature of this function [ThunkSignature]
  // CHECK:STDERR:   fn End(self) -> Sentinel;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_adl_returns_different_types_mutable.carbon:[[@LINE-35]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn TestIterate[R:! Core.Iterate where .ElementType = Cpp.N.ValueType](r: R) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_todo_adl_returns_different_types_mutable.carbon:[[@LINE+14]]:3: error: semantics TODO: `Rewriting operator== using operator== is not supported` [SemanticsTodo]
  // CHECK:STDERR:   TestIterate(m);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_adl_returns_different_types_mutable.carbon:[[@LINE-42]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn TestIterate[R:! Core.Iterate where .ElementType = Cpp.N.ValueType](r: R) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_todo_adl_returns_different_types_mutable.carbon:[[@LINE+7]]:3: error: cannot convert type `Cpp.N.MutableRange` into type implementing `Core.Iterate where .(Core.Iterate.ElementType) = Cpp.N.ValueType as Core.Destroy & Core.Copy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   TestIterate(m);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_adl_returns_different_types_mutable.carbon:[[@LINE-49]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn TestIterate[R:! Core.Iterate where .ElementType = Cpp.N.ValueType](r: R) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  TestIterate(m);
}

fn TestRangeFor(var m: Cpp.N.MutableRange) {
  var sum: Cpp.N.ValueType = Cpp.N.ValueType.ValueType();

  //@dump-sem-ir-begin
  // CHECK:STDERR: fail_todo_adl_returns_different_types_mutable.carbon:[[@LINE+8]]:7: error: cannot access member of interface `Core.Iterate` in type `Cpp.N.MutableRange` that does not implement that interface [MissingImplInMemberAccess]
  // CHECK:STDERR:   for (i: Cpp.N.ValueType in m) {
  // CHECK:STDERR:       ^~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_todo_adl_returns_different_types_mutable.carbon:[[@LINE+4]]:7: error: cannot access member of interface `Core.Iterate` in type `Cpp.N.MutableRange` that does not implement that interface [MissingImplInMemberAccess]
  // CHECK:STDERR:   for (i: Cpp.N.ValueType in m) {
  // CHECK:STDERR:       ^~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  for (i: Cpp.N.ValueType in m) {
    sum += i;
  }
  //@dump-sem-ir-end
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

fn TestIterate[R:! Core.Iterate where .ElementType = Cpp.N.ValueType](r: R) {
  var sum: Cpp.N.ValueType = Cpp.N.ValueType.ValueType();

  //@dump-sem-ir-begin
  for (i: Cpp.N.ValueType in r) {
    sum += i;
  }
  //@dump-sem-ir-end
}

fn TestIterateDriver(var mc: Cpp.N.MutableAndConstRange) {
  // CHECK:STDERR: fail_todo_adl_overloads.carbon:[[@LINE+28]]:3: error: facet type `Core.CppRangeForIterate where .(Core.CppRangeForIterate.Iterator) impls Core.Destroy and .(Core.CppRangeForIterate.Iterator) impls Core.Copy and .(Core.CppRangeForIterate.Iterator) impls Core.CppUnsafeDeref and .(Core.CppRangeForIterate.Iterator) impls Core.EqWith(.(Core.CppRangeForIterate.Sentinel)) and .(Core.CppRangeForIterate.Iterator) impls Core.Inc and .(Core.CppRangeForIterate.Iterator).(Core.CppUnsafeDeref.Result) impls Core.Destroy and .(Core.CppRangeForIterate.Iterator).(Core.CppUnsafeDeref.Result) impls Core.Copy and .(Core.CppRangeForIterate.Sentinel) impls Core.Destroy and .(Core.CppRangeForIterate.Sentinel) impls Core.Copy` can not be identified [ImplLookupInUnidentifiedFacetType]
  // CHECK:STDERR:   TestIterate(mc);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_adl_overloads.carbon: note: value expression passed to reference parameter [ValueForRefParam]
  // CHECK:STDERR: fail_todo_adl_overloads.carbon:[[@LINE-24]]:44: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR:     friend auto begin(MutableAndConstRange&) -> Iterator;
  // CHECK:STDERR:                                            ^
  // CHECK:STDERR: {{.*}}/prelude/iterate.carbon:39:3: note: while building thunk to match the signature of this function [ThunkSignature]
  // CHECK:STDERR:   fn Begin(self) -> Iterator;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_adl_overloads.carbon:[[@LINE-21]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn TestIterate[R:! Core.Iterate where .ElementType = Cpp.N.ValueType](r: R) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_todo_adl_overloads.carbon:[[@LINE+14]]:3: error: facet type `Core.CppRangeForIterate where .(Core.CppRangeForIterate.Iterator) impls Core.Destroy and .(Core.CppRangeForIterate.Iterator) impls Core.Copy and .(Core.CppRangeForIterate.Iterator) impls Core.CppUnsafeDeref and .(Core.CppRangeForIterate.Iterator) impls Core.EqWith(.(Core.CppRangeForIterate.Sentinel)) and .(Core.CppRangeForIterate.Iterator) impls Core.Inc and .(Core.CppRangeForIterate.Iterator).(Core.CppUnsafeDeref.Result) impls Core.Destroy and .(Core.CppRangeForIterate.Iterator).(Core.CppUnsafeDeref.Result) impls Core.Copy and .(Core.CppRangeForIterate.Sentinel) impls Core.Destroy and .(Core.CppRangeForIterate.Sentinel) impls Core.Copy` can not be identified [ImplLookupInUnidentifiedFacetType]
  // CHECK:STDERR:   TestIterate(mc);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_adl_overloads.carbon: note: value expression passed to reference parameter [ValueForRefParam]
  // CHECK:STDERR: fail_todo_adl_overloads.carbon:[[@LINE-37]]:42: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR:     friend auto end(MutableAndConstRange&) -> Iterator;
  // CHECK:STDERR:                                          ^
  // CHECK:STDERR: {{.*}}/prelude/iterate.carbon:40:3: note: while building thunk to match the signature of this function [ThunkSignature]
  // CHECK:STDERR:   fn End(self) -> Sentinel;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_adl_overloads.carbon:[[@LINE-35]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn TestIterate[R:! Core.Iterate where .ElementType = Cpp.N.ValueType](r: R) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  TestIterate(mc);
}

fn TestRangeFor(var mc: Cpp.N.MutableAndConstRange) {
  var sum: Cpp.N.ValueType = Cpp.N.ValueType.ValueType();

  //@dump-sem-ir-begin
  for (i: Cpp.N.ValueType in mc) {
    sum += i;
  }
  //@dump-sem-ir-end
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
// CHECK:STDOUT:   %facet_type.f48: type = facet_type <@Destroy & @Copy> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %T.220: %OptionalStorage.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Optional.Get.type.bb9: type = fn_type @Optional.Get, @Optional(%T.220) [symbolic]
// CHECK:STDOUT:   %Optional.Get.fa1: %Optional.Get.type.bb9 = struct_value () [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.type.e9e: type = fn_type @Optional.HasValue, @Optional(%T.220) [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.6cd: %Optional.HasValue.type.e9e = struct_value () [symbolic]
// CHECK:STDOUT:   %T.283: %facet_type.f48 = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.d50: type = class_type @DefaultOptionalStorage, @DefaultOptionalStorage(%T.283) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Get.type.029: type = fn_type @T.as_type.as.OptionalStorage.impl.Get, @T.as_type.as.OptionalStorage.impl(%T.283) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Get.cb5: %T.as_type.as.OptionalStorage.impl.Get.type.029 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Has.type.4bc: type = fn_type @T.as_type.as.OptionalStorage.impl.Has, @T.as_type.as.OptionalStorage.impl(%T.283) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Has.dc2: %T.as_type.as.OptionalStorage.impl.Has.type.4bc = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Some.type.800: type = fn_type @T.as_type.as.OptionalStorage.impl.Some, @T.as_type.as.OptionalStorage.impl(%T.283) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Some.e83: %T.as_type.as.OptionalStorage.impl.Some.type.800 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.None.type.300: type = fn_type @T.as_type.as.OptionalStorage.impl.None, @T.as_type.as.OptionalStorage.impl(%T.283) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.None.f54: %T.as_type.as.OptionalStorage.impl.None.type.300 = struct_value () [symbolic]
// CHECK:STDOUT:   %.Self.35d: %Iterate.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %Iterate.lookup_impl_witness.b37: <witness> = lookup_impl_witness %.Self.35d, @Iterate [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.903: %facet_type.f48 = impl_witness_access %Iterate.lookup_impl_witness.b37, element0 [symbolic_self]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %N.fe9: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %Int.67af24.1: type = class_type @Int, @Int(%N.fe9) [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.6b6: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.2: type = fn_type @Destroy.Op.loc30_54.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.2: %Destroy.Op.type.1d8f74.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.df9cc1.2: <witness> = custom_witness (%Destroy.Op.1a2547.2), @Destroy [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.ac8: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N.fe9) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.5e0: %Int.as.Copy.impl.Op.type.ac8 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.0e0: <witness> = impl_witness imports.%Copy.impl_witness_table.367, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %facet_value.ba2: %facet_type.f48 = facet_value %i32, (%custom_witness.df9cc1.2, %Copy.impl_witness.0e0) [concrete]
// CHECK:STDOUT:   %Iterate_where.type.8bc: type = facet_type <@Iterate where %impl.elem0.903 = %facet_value.ba2> [concrete]
// CHECK:STDOUT:   %pattern_type.e3e: type = pattern_type %Iterate_where.type.8bc [concrete]
// CHECK:STDOUT:   %R.patt.860: %pattern_type.e3e = symbolic_binding_pattern R, 0 [symbolic]
// CHECK:STDOUT:   %R.891: %Iterate_where.type.8bc = symbolic_binding R, 0 [symbolic]
// CHECK:STDOUT:   %R.as_type.67d: type = facet_access_type %R.891 [symbolic]
// CHECK:STDOUT:   %pattern_type.e21: type = pattern_type %R.as_type.67d [symbolic]
// CHECK:STDOUT:   %r.param_patt.fdd: %pattern_type.e21 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %r.patt.939: %pattern_type.e21 = at_binding_pattern r, %r.param_patt.fdd [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.914: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Int.67af24.2: type = class_type @Int, @Int(%To) [symbolic]
// CHECK:STDOUT:   %facet_type.037: type = facet_type <@IntFitsIn, @IntFitsIn(%Int.67af24.2) & @AnyInt> [symbolic]
// CHECK:STDOUT:   %From.6a4: %facet_type.037 = symbolic_binding From, 1 [symbolic]
// CHECK:STDOUT:   %From.as_type.as.ImplicitAs.impl.Convert.type.7fa: type = fn_type @From.as_type.as.ImplicitAs.impl.Convert, @From.as_type.as.ImplicitAs.impl(%To, %From.6a4) [symbolic]
// CHECK:STDOUT:   %From.as_type.as.ImplicitAs.impl.Convert.643: %From.as_type.as.ImplicitAs.impl.Convert.type.7fa = struct_value () [symbolic]
// CHECK:STDOUT:   %i.patt: %pattern_type.6b6 = value_binding_pattern i [concrete]
// CHECK:STDOUT:   %CppRangeForIterate.type: type = facet_type <@CppRangeForIterate> [concrete]
// CHECK:STDOUT:   %.Self.6bd: %CppRangeForIterate.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %CppRangeForIterate.lookup_impl_witness.143: <witness> = lookup_impl_witness %.Self.6bd, @CppRangeForIterate [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.109: type = impl_witness_access %CppRangeForIterate.lookup_impl_witness.143, element0 [symbolic_self]
// CHECK:STDOUT:   %CppUnsafeDeref.lookup_impl_witness.712: <witness> = lookup_impl_witness %impl.elem0.109, @CppUnsafeDeref [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.56a: type = impl_witness_access %CppUnsafeDeref.lookup_impl_witness.712, element0 [symbolic_self]
// CHECK:STDOUT:   %impl.elem1.222: type = impl_witness_access %CppRangeForIterate.lookup_impl_witness.143, element1 [symbolic_self]
// CHECK:STDOUT:   %CppRangeForIterate_where.type: type = facet_type <@CppRangeForIterate where %impl.elem0.109 impls @Destroy and %impl.elem0.109 impls @Copy and %impl.elem0.109 impls @CppUnsafeDeref and %impl.elem0.109 impls @EqWith, @EqWith(%impl.elem1.222) and %impl.elem0.109 impls @Inc and %impl.elem0.56a impls @Destroy and %impl.elem0.56a impls @Copy and %impl.elem1.222 impls @Destroy and %impl.elem1.222 impls @Copy> [concrete]
// CHECK:STDOUT:   %R.39b: %CppRangeForIterate_where.type = symbolic_binding R, 0 [symbolic]
// CHECK:STDOUT:   %CppRangeForIterate.lookup_impl_witness.7e2712.2: <witness> = lookup_impl_witness %R.39b, @CppRangeForIterate [symbolic]
// CHECK:STDOUT:   %impl.elem1.7a0bff.2: type = impl_witness_access %CppRangeForIterate.lookup_impl_witness.7e2712.2, element1 [symbolic]
// CHECK:STDOUT:   %impl.elem0.73eb7f.2: type = impl_witness_access %CppRangeForIterate.lookup_impl_witness.7e2712.2, element0 [symbolic]
// CHECK:STDOUT:   %tuple.type.3cf: type = tuple_type (%impl.elem0.73eb7f.2, %impl.elem1.7a0bff.2) [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.69f: <witness> = lookup_impl_witness %tuple.type.3cf, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.593: %Destroy.type = facet_value %tuple.type.3cf, (%Destroy.lookup_impl_witness.69f) [symbolic]
// CHECK:STDOUT:   %CppUnsafeDeref.lookup_impl_witness.aa7: <witness> = lookup_impl_witness %impl.elem0.73eb7f.2, @CppUnsafeDeref [symbolic]
// CHECK:STDOUT:   %impl.elem0.af7: type = impl_witness_access %CppUnsafeDeref.lookup_impl_witness.aa7, element0 [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.ec3: <witness> = lookup_impl_witness %impl.elem0.af7, @Destroy [symbolic]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.a7a: <witness> = lookup_impl_witness %impl.elem0.af7, @Copy [symbolic]
// CHECK:STDOUT:   %facet_value.194: %facet_type.f48 = facet_value %impl.elem0.af7, (%Destroy.lookup_impl_witness.ec3, %Copy.lookup_impl_witness.a7a) [symbolic]
// CHECK:STDOUT:   %R.as_type.as.Iterate.impl.Next.type.1f7: type = fn_type @R.as_type.as.Iterate.impl.Next, @R.as_type.as.Iterate.impl(%R.39b) [symbolic]
// CHECK:STDOUT:   %R.as_type.as.Iterate.impl.Next.0d6: %R.as_type.as.Iterate.impl.Next.type.1f7 = struct_value () [symbolic]
// CHECK:STDOUT:   %R.as_type.as.Iterate.impl.NewCursor.type.36f: type = fn_type @R.as_type.as.Iterate.impl.NewCursor, @R.as_type.as.Iterate.impl(%R.39b) [symbolic]
// CHECK:STDOUT:   %R.as_type.as.Iterate.impl.NewCursor.29b: %R.as_type.as.Iterate.impl.NewCursor.type.36f = struct_value () [symbolic]
// CHECK:STDOUT:   %Iterate.lookup_impl_witness.c86: <witness> = lookup_impl_witness %R.891, @Iterate [symbolic]
// CHECK:STDOUT:   %Iterate.facet.bbb: %Iterate.type = facet_value %R.as_type.67d, (%Iterate.lookup_impl_witness.c86) [symbolic]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.d2b: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.bbb) [symbolic]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.28b: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.bbb) [symbolic]
// CHECK:STDOUT:   %.3e2: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.d2b, %Iterate.facet.bbb [symbolic]
// CHECK:STDOUT:   %impl.elem2.f53: %.3e2 = impl_witness_access %Iterate.lookup_impl_witness.c86, element2 [symbolic]
// CHECK:STDOUT:   %impl.elem1.d31: %Destroy.type = impl_witness_access %Iterate.lookup_impl_witness.c86, element1 [symbolic]
// CHECK:STDOUT:   %as_type.ca2: type = facet_access_type %impl.elem1.d31 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.87a: <specific function> = specific_impl_function %impl.elem2.f53, @Iterate.WithSelf.NewCursor(%Iterate.facet.bbb) [symbolic]
// CHECK:STDOUT:   %ptr.3b4: type = ptr_type %as_type.ca2 [symbolic]
// CHECK:STDOUT:   %.a8b: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.28b, %Iterate.facet.bbb [symbolic]
// CHECK:STDOUT:   %impl.elem3.9c3: %.a8b = impl_witness_access %Iterate.lookup_impl_witness.c86, element3 [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.970: type = class_type @DefaultOptionalStorage, @DefaultOptionalStorage(%facet_value.ba2) [concrete]
// CHECK:STDOUT:   %OptionalStorage.impl_witness.a49: <witness> = impl_witness imports.%OptionalStorage.impl_witness_table.655, @T.as_type.as.OptionalStorage.impl(%facet_value.ba2) [concrete]
// CHECK:STDOUT:   %OptionalStorage.facet.172: %OptionalStorage.type = facet_value %i32, (%OptionalStorage.impl_witness.a49) [concrete]
// CHECK:STDOUT:   %Optional.e28: type = class_type @Optional, @Optional(%OptionalStorage.facet.172) [concrete]
// CHECK:STDOUT:   %specific_impl_fn.ab2: <specific function> = specific_impl_function %impl.elem3.9c3, @Iterate.WithSelf.Next(%Iterate.facet.bbb) [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.type.c6d: type = fn_type @Optional.HasValue, @Optional(%OptionalStorage.facet.172) [concrete]
// CHECK:STDOUT:   %Optional.HasValue.89b: %Optional.HasValue.type.c6d = struct_value () [concrete]
// CHECK:STDOUT:   %Optional.Get.type.7e6: type = fn_type @Optional.Get, @Optional(%OptionalStorage.facet.172) [concrete]
// CHECK:STDOUT:   %Optional.Get.722: %Optional.Get.type.7e6 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.facet.d07318.2: %Destroy.type = facet_value %i32, (%custom_witness.df9cc1.2) [concrete]
// CHECK:STDOUT:   %MaybeUnformed.023: type = class_type @MaybeUnformed, @MaybeUnformed(%Destroy.facet.d07318.2) [concrete]
// CHECK:STDOUT:   %.1d6a7: type = maybe_unformed_type %i32 [concrete]
// CHECK:STDOUT:   %struct_type.value.has_value.ad0: type = struct_type {.value: %MaybeUnformed.023, .has_value: bool} [concrete]
// CHECK:STDOUT:   %Optional.HasValue.specific_fn: <specific function> = specific_function %Optional.HasValue.89b, @Optional.HasValue(%OptionalStorage.facet.172) [concrete]
// CHECK:STDOUT:   %Optional.Get.specific_fn: <specific function> = specific_function %Optional.Get.722, @Optional.Get(%OptionalStorage.facet.172) [concrete]
// CHECK:STDOUT:   %AddAssignWith.type.941: type = facet_type <@AddAssignWith, @AddAssignWith(%i32)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.f7de31.2: type = facet_type <@ImplicitAs, @ImplicitAs(%Int.67af24.1)> [symbolic]
// CHECK:STDOUT:   %U.4e6: %ImplicitAs.type.f7de31.2 = symbolic_binding U, 1 [symbolic]
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.type.a58fbe.1: type = fn_type @Int.as.AddAssignWith.impl.Op.1, @Int.as.AddAssignWith.impl(%N.fe9, %U.4e6) [symbolic]
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.45b360.1: %Int.as.AddAssignWith.impl.Op.type.a58fbe.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.type.a58fbe.2: type = fn_type @Int.as.AddAssignWith.impl.Op.2, @Int.as.AddAssignWith.impl(%N.fe9, %U.4e6) [symbolic]
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.45b360.2: %Int.as.AddAssignWith.impl.Op.type.a58fbe.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %facet_type.138: type = facet_type <@IntFitsIn, @IntFitsIn(%i32) & @AnyInt> [concrete]
// CHECK:STDOUT:   %custom_witness.36a: <witness> = custom_witness (), @IntFitsIn, @IntFitsIn(%i32) [concrete]
// CHECK:STDOUT:   %Int.as.AnyInt.impl.AsInt.type.324: type = fn_type @Int.as.AnyInt.impl.AsInt, @Int.as.AnyInt.impl(%N.fe9) [symbolic]
// CHECK:STDOUT:   %Int.as.AnyInt.impl.AsInt.ec6: %Int.as.AnyInt.impl.AsInt.type.324 = struct_value () [symbolic]
// CHECK:STDOUT:   %AnyInt.impl_witness.b8b: <witness> = impl_witness imports.%AnyInt.impl_witness_table, @Int.as.AnyInt.impl(%int_32) [concrete]
// CHECK:STDOUT:   %facet_value.0ec: %facet_type.138 = facet_value %i32, (%custom_witness.36a, %AnyInt.impl_witness.b8b) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.577: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.3b9, @From.as_type.as.ImplicitAs.impl(%int_32, %facet_value.0ec) [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.eaa: %ImplicitAs.type.914 = facet_value %i32, (%ImplicitAs.impl_witness.577) [concrete]
// CHECK:STDOUT:   %AddAssignWith.impl_witness.22e: <witness> = impl_witness imports.%AddAssignWith.impl_witness_table, @Int.as.AddAssignWith.impl(%int_32, %ImplicitAs.facet.eaa) [concrete]
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.type.73460a.1: type = fn_type @Int.as.AddAssignWith.impl.Op.2, @Int.as.AddAssignWith.impl(%int_32, %ImplicitAs.facet.eaa) [concrete]
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.00e760.1: %Int.as.AddAssignWith.impl.Op.type.73460a.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.type.73460a.2: type = fn_type @Int.as.AddAssignWith.impl.Op.1, @Int.as.AddAssignWith.impl(%int_32, %ImplicitAs.facet.eaa) [concrete]
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.00e760.2: %Int.as.AddAssignWith.impl.Op.type.73460a.2 = struct_value () [concrete]
// CHECK:STDOUT:   %AddAssignWith.facet.d42: %AddAssignWith.type.941 = facet_value %i32, (%AddAssignWith.impl_witness.22e) [concrete]
// CHECK:STDOUT:   %AddAssignWith.WithSelf.Op.type.f66: type = fn_type @AddAssignWith.WithSelf.Op, @AddAssignWith.WithSelf(%i32, %AddAssignWith.facet.d42) [concrete]
// CHECK:STDOUT:   %.b28: type = fn_type_with_self_type %AddAssignWith.WithSelf.Op.type.f66, %AddAssignWith.facet.d42 [concrete]
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.specific_fn.b14d9d.1: <specific function> = specific_function %Int.as.AddAssignWith.impl.Op.00e760.2, @Int.as.AddAssignWith.impl.Op.1(%int_32, %ImplicitAs.facet.eaa) [concrete]
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.specific_fn.b14d9d.2: <specific function> = specific_function %Int.as.AddAssignWith.impl.Op.00e760.1, @Int.as.AddAssignWith.impl.Op.2(%int_32, %ImplicitAs.facet.eaa) [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.9: type = fn_type @Destroy.Op.loc34_19.6 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.9: %Destroy.Op.type.1d8f74.9 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.5d5: <witness> = lookup_impl_witness %impl.elem1.d31, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.515: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%impl.elem1.d31) [symbolic]
// CHECK:STDOUT:   %.94f: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.515, %impl.elem1.d31 [symbolic]
// CHECK:STDOUT:   %impl.elem0.e52: %.94f = impl_witness_access %Destroy.lookup_impl_witness.5d5, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.12e: <specific function> = specific_impl_function %impl.elem0.e52, @Destroy.WithSelf.Op(%impl.elem1.d31) [symbolic]
// CHECK:STDOUT:   %MutableRange: type = class_type @MutableRange [concrete]
// CHECK:STDOUT:   %pattern_type.e635: type = pattern_type %MutableRange [concrete]
// CHECK:STDOUT:   %ConstRange: type = class_type @ConstRange [concrete]
// CHECK:STDOUT:   %pattern_type.0ba: type = pattern_type %ConstRange [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %Iterator.3ee: type = class_type @Iterator.1 [concrete]
// CHECK:STDOUT:   %MutableRange.Begin.type: type = fn_type @MutableRange.Begin [concrete]
// CHECK:STDOUT:   %MutableRange.Begin: %MutableRange.Begin.type = struct_value () [concrete]
// CHECK:STDOUT:   %MutableRange.End.type: type = fn_type @MutableRange.End [concrete]
// CHECK:STDOUT:   %MutableRange.End: %MutableRange.End.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.ed3: <witness> = custom_witness (%Iterator.3ee, %Iterator.3ee, %MutableRange.Begin, %MutableRange.End), @CppRangeForIterate [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.0aed5d.1: type = fn_type @Iterator.Op.1 [concrete]
// CHECK:STDOUT:   %Iterator.Op.a2a866.1: %Iterator.Op.type.0aed5d.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.ef8: <witness> = custom_witness (%i32, %Iterator.Op.a2a866.1), @CppUnsafeDeref [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.0aed5d.2: type = fn_type @Iterator.Op.2 [concrete]
// CHECK:STDOUT:   %Iterator.Op.a2a866.2: %Iterator.Op.type.0aed5d.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.955: <witness> = custom_witness (%Iterator.Op.a2a866.2), @Destroy [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.0aed5d.3: type = fn_type @Iterator.Op.3 [concrete]
// CHECK:STDOUT:   %Iterator.Op.a2a866.3: %Iterator.Op.type.0aed5d.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.1f5: <witness> = custom_witness (%Iterator.Op.a2a866.3), @Copy [concrete]
// CHECK:STDOUT:   %Equal.type.ca8475.1: type = fn_type @Equal.1 [concrete]
// CHECK:STDOUT:   %Equal.285b15.1: %Equal.type.ca8475.1 = struct_value () [concrete]
// CHECK:STDOUT:   %NotEqual.type.53d9ee.1: type = fn_type @NotEqual.1 [concrete]
// CHECK:STDOUT:   %NotEqual.119263.1: %NotEqual.type.53d9ee.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.981: <witness> = custom_witness (%Equal.285b15.1, %NotEqual.119263.1), @EqWith, @EqWith(%Iterator.3ee) [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.0aed5d.4: type = fn_type @Iterator.Op.4 [concrete]
// CHECK:STDOUT:   %Iterator.Op.a2a866.4: %Iterator.Op.type.0aed5d.4 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.444: <witness> = custom_witness (%Iterator.Op.a2a866.4), @Inc [concrete]
// CHECK:STDOUT:   %facet_value.cd6: %CppRangeForIterate_where.type = facet_value %MutableRange, (%custom_witness.df9cc1.2, %Copy.impl_witness.0e0, %custom_witness.ed3, %custom_witness.955, %custom_witness.1f5, %custom_witness.ef8, %custom_witness.981, %custom_witness.444) [concrete]
// CHECK:STDOUT:   %tuple.type.508: type = tuple_type (%Iterator.3ee, %Iterator.3ee) [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.10: type = fn_type @Destroy.Op.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.10: %Destroy.Op.type.1d8f74.10 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.df9cc1.10: <witness> = custom_witness (%Destroy.Op.1a2547.10), @Destroy [concrete]
// CHECK:STDOUT:   %Destroy.facet.30f: %Destroy.type = facet_value %tuple.type.508, (%custom_witness.df9cc1.10) [concrete]
// CHECK:STDOUT:   %Iterate.impl_witness.cea: <witness> = impl_witness imports.%Iterate.impl_witness_table.344, @R.as_type.as.Iterate.impl(%facet_value.cd6) [concrete]
// CHECK:STDOUT:   %R.as_type.as.Iterate.impl.NewCursor.type.f1f: type = fn_type @R.as_type.as.Iterate.impl.NewCursor, @R.as_type.as.Iterate.impl(%facet_value.cd6) [concrete]
// CHECK:STDOUT:   %R.as_type.as.Iterate.impl.NewCursor.8d2: %R.as_type.as.Iterate.impl.NewCursor.type.f1f = struct_value () [concrete]
// CHECK:STDOUT:   %R.as_type.as.Iterate.impl.Next.type.dba: type = fn_type @R.as_type.as.Iterate.impl.Next, @R.as_type.as.Iterate.impl(%facet_value.cd6) [concrete]
// CHECK:STDOUT:   %R.as_type.as.Iterate.impl.Next.34d: %R.as_type.as.Iterate.impl.Next.type.dba = struct_value () [concrete]
// CHECK:STDOUT:   %Iterate.facet.ffb: %Iterate.type = facet_value %MutableRange, (%Iterate.impl_witness.cea) [concrete]
// CHECK:STDOUT:   %facet_value.879: %Iterate_where.type.8bc = facet_value %MutableRange, (%Iterate.impl_witness.cea) [concrete]
// CHECK:STDOUT:   %r.param_patt.c79: %pattern_type.e635 = value_param_pattern [concrete]
// CHECK:STDOUT:   %r.patt.16f: %pattern_type.e635 = at_binding_pattern r, %r.param_patt.c79 [concrete]
// CHECK:STDOUT:   %Iterator.260: type = class_type @Iterator.2 [concrete]
// CHECK:STDOUT:   %ConstRange.Begin.type: type = fn_type @ConstRange.Begin [concrete]
// CHECK:STDOUT:   %ConstRange.Begin: %ConstRange.Begin.type = struct_value () [concrete]
// CHECK:STDOUT:   %ConstRange.End.type: type = fn_type @ConstRange.End [concrete]
// CHECK:STDOUT:   %ConstRange.End: %ConstRange.End.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.23e: <witness> = custom_witness (%Iterator.260, %Iterator.260, %ConstRange.Begin, %ConstRange.End), @CppRangeForIterate [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.0f34b8.1: type = fn_type @Iterator.Op.5 [concrete]
// CHECK:STDOUT:   %Iterator.Op.1c3e4a.1: %Iterator.Op.type.0f34b8.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.adb: <witness> = custom_witness (%i32, %Iterator.Op.1c3e4a.1), @CppUnsafeDeref [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.0f34b8.2: type = fn_type @Iterator.Op.6 [concrete]
// CHECK:STDOUT:   %Iterator.Op.1c3e4a.2: %Iterator.Op.type.0f34b8.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.6e8: <witness> = custom_witness (%Iterator.Op.1c3e4a.2), @Destroy [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.0f34b8.3: type = fn_type @Iterator.Op.7 [concrete]
// CHECK:STDOUT:   %Iterator.Op.1c3e4a.3: %Iterator.Op.type.0f34b8.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.150: <witness> = custom_witness (%Iterator.Op.1c3e4a.3), @Copy [concrete]
// CHECK:STDOUT:   %Equal.type.ca8475.2: type = fn_type @Equal.2 [concrete]
// CHECK:STDOUT:   %Equal.285b15.2: %Equal.type.ca8475.2 = struct_value () [concrete]
// CHECK:STDOUT:   %NotEqual.type.53d9ee.2: type = fn_type @NotEqual.2 [concrete]
// CHECK:STDOUT:   %NotEqual.119263.2: %NotEqual.type.53d9ee.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.941: <witness> = custom_witness (%Equal.285b15.2, %NotEqual.119263.2), @EqWith, @EqWith(%Iterator.260) [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.0f34b8.4: type = fn_type @Iterator.Op.8 [concrete]
// CHECK:STDOUT:   %Iterator.Op.1c3e4a.4: %Iterator.Op.type.0f34b8.4 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.746: <witness> = custom_witness (%Iterator.Op.1c3e4a.4), @Inc [concrete]
// CHECK:STDOUT:   %facet_value.e88: %CppRangeForIterate_where.type = facet_value %ConstRange, (%custom_witness.df9cc1.2, %Copy.impl_witness.0e0, %custom_witness.23e, %custom_witness.6e8, %custom_witness.150, %custom_witness.adb, %custom_witness.941, %custom_witness.746) [concrete]
// CHECK:STDOUT:   %tuple.type.69f: type = tuple_type (%Iterator.260, %Iterator.260) [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.11: type = fn_type @Destroy.Op.3 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.11: %Destroy.Op.type.1d8f74.11 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.df9cc1.11: <witness> = custom_witness (%Destroy.Op.1a2547.11), @Destroy [concrete]
// CHECK:STDOUT:   %Destroy.facet.e4c: %Destroy.type = facet_value %tuple.type.69f, (%custom_witness.df9cc1.11) [concrete]
// CHECK:STDOUT:   %Iterate.impl_witness.12a: <witness> = impl_witness imports.%Iterate.impl_witness_table.344, @R.as_type.as.Iterate.impl(%facet_value.e88) [concrete]
// CHECK:STDOUT:   %R.as_type.as.Iterate.impl.NewCursor.type.fe1: type = fn_type @R.as_type.as.Iterate.impl.NewCursor, @R.as_type.as.Iterate.impl(%facet_value.e88) [concrete]
// CHECK:STDOUT:   %R.as_type.as.Iterate.impl.NewCursor.2f6: %R.as_type.as.Iterate.impl.NewCursor.type.fe1 = struct_value () [concrete]
// CHECK:STDOUT:   %R.as_type.as.Iterate.impl.Next.type.f9d: type = fn_type @R.as_type.as.Iterate.impl.Next, @R.as_type.as.Iterate.impl(%facet_value.e88) [concrete]
// CHECK:STDOUT:   %R.as_type.as.Iterate.impl.Next.393: %R.as_type.as.Iterate.impl.Next.type.f9d = struct_value () [concrete]
// CHECK:STDOUT:   %Iterate.facet.5eb: %Iterate.type = facet_value %ConstRange, (%Iterate.impl_witness.12a) [concrete]
// CHECK:STDOUT:   %facet_value.740: %Iterate_where.type.8bc = facet_value %ConstRange, (%Iterate.impl_witness.12a) [concrete]
// CHECK:STDOUT:   %r.param_patt.83e: %pattern_type.0ba = value_param_pattern [concrete]
// CHECK:STDOUT:   %r.patt.116: %pattern_type.0ba = at_binding_pattern r, %r.param_patt.83e [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.dcb: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.ffb) [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.30e: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.ffb) [concrete]
// CHECK:STDOUT:   %.d6a: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.dcb, %Iterate.facet.ffb [concrete]
// CHECK:STDOUT:   %R.as_type.as.Iterate.impl.NewCursor.specific_fn.bf1: <specific function> = specific_function %R.as_type.as.Iterate.impl.NewCursor.8d2, @R.as_type.as.Iterate.impl.NewCursor(%facet_value.cd6) [concrete]
// CHECK:STDOUT:   %ptr.45c: type = ptr_type %tuple.type.508 [concrete]
// CHECK:STDOUT:   %.d23: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.30e, %Iterate.facet.ffb [concrete]
// CHECK:STDOUT:   %R.as_type.as.Iterate.impl.Next.specific_fn.dd5: <specific function> = specific_function %R.as_type.as.Iterate.impl.Next.34d, @R.as_type.as.Iterate.impl.Next(%facet_value.cd6) [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.ab5: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.5eb) [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.e25: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.5eb) [concrete]
// CHECK:STDOUT:   %.744: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.ab5, %Iterate.facet.5eb [concrete]
// CHECK:STDOUT:   %R.as_type.as.Iterate.impl.NewCursor.specific_fn.a05: <specific function> = specific_function %R.as_type.as.Iterate.impl.NewCursor.2f6, @R.as_type.as.Iterate.impl.NewCursor(%facet_value.e88) [concrete]
// CHECK:STDOUT:   %ptr.c5c: type = ptr_type %tuple.type.69f [concrete]
// CHECK:STDOUT:   %.03b: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.e25, %Iterate.facet.5eb [concrete]
// CHECK:STDOUT:   %R.as_type.as.Iterate.impl.Next.specific_fn.4cc: <specific function> = specific_function %R.as_type.as.Iterate.impl.Next.393, @R.as_type.as.Iterate.impl.Next(%facet_value.e88) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.7ec: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.e4c) [concrete]
// CHECK:STDOUT:   %.0e7: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.7ec, %Destroy.facet.e4c [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.c6c: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.30f) [concrete]
// CHECK:STDOUT:   %.0a4: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.c6c, %Destroy.facet.30f [concrete]
// CHECK:STDOUT:   %complete_type.c28: <witness> = complete_type_witness %tuple.type.508 [concrete]
// CHECK:STDOUT:   %complete_type.8d5: <witness> = complete_type_witness %tuple.type.69f [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.bb6: @Optional.%Optional.HasValue.type (%Optional.HasValue.type.e9e) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.HasValue (constants.%Optional.HasValue.6cd)]
// CHECK:STDOUT:   %Core.import_ref.daba: @Optional.%Optional.Get.type (%Optional.Get.type.bb9) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.Get (constants.%Optional.Get.fa1)]
// CHECK:STDOUT:   %Core.import_ref.0c6: type = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%DefaultOptionalStorage (constants.%DefaultOptionalStorage.d50)]
// CHECK:STDOUT:   %Core.import_ref.f4f: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.None.type (%T.as_type.as.OptionalStorage.impl.None.type.300) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.None (constants.%T.as_type.as.OptionalStorage.impl.None.f54)]
// CHECK:STDOUT:   %Core.import_ref.18f: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Some.type (%T.as_type.as.OptionalStorage.impl.Some.type.800) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Some (constants.%T.as_type.as.OptionalStorage.impl.Some.e83)]
// CHECK:STDOUT:   %Core.import_ref.c68: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Has.type (%T.as_type.as.OptionalStorage.impl.Has.type.4bc) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Has (constants.%T.as_type.as.OptionalStorage.impl.Has.dc2)]
// CHECK:STDOUT:   %Core.import_ref.a83: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Get.type (%T.as_type.as.OptionalStorage.impl.Get.type.029) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Get (constants.%T.as_type.as.OptionalStorage.impl.Get.cb5)]
// CHECK:STDOUT:   %Core.import_ref.aaa = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], unloaded
// CHECK:STDOUT:   %OptionalStorage.impl_witness_table.655 = impl_witness_table (%Core.import_ref.0c6, %Core.import_ref.f4f, %Core.import_ref.18f, %Core.import_ref.c68, %Core.import_ref.a83, %Core.import_ref.aaa), @T.as_type.as.OptionalStorage.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.cd6: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.ac8) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.5e0)]
// CHECK:STDOUT:   %Copy.impl_witness_table.367 = impl_witness_table (%Core.import_ref.cd6), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.2a4: @From.as_type.as.ImplicitAs.impl.%From.as_type.as.ImplicitAs.impl.Convert.type (%From.as_type.as.ImplicitAs.impl.Convert.type.7fa) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @From.as_type.as.ImplicitAs.impl.%From.as_type.as.ImplicitAs.impl.Convert (constants.%From.as_type.as.ImplicitAs.impl.Convert.643)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.3b9 = impl_witness_table (%Core.import_ref.2a4), @From.as_type.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.301: %facet_type.f48 = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @R.as_type.as.Iterate.impl.%facet_value (constants.%facet_value.194)]
// CHECK:STDOUT:   %Core.import_ref.89f: %Destroy.type = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @R.as_type.as.Iterate.impl.%Destroy.facet (constants.%Destroy.facet.593)]
// CHECK:STDOUT:   %Core.import_ref.3e6: @R.as_type.as.Iterate.impl.%R.as_type.as.Iterate.impl.NewCursor.type (%R.as_type.as.Iterate.impl.NewCursor.type.36f) = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @R.as_type.as.Iterate.impl.%R.as_type.as.Iterate.impl.NewCursor (constants.%R.as_type.as.Iterate.impl.NewCursor.29b)]
// CHECK:STDOUT:   %Core.import_ref.85d: @R.as_type.as.Iterate.impl.%R.as_type.as.Iterate.impl.Next.type (%R.as_type.as.Iterate.impl.Next.type.1f7) = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @R.as_type.as.Iterate.impl.%R.as_type.as.Iterate.impl.Next (constants.%R.as_type.as.Iterate.impl.Next.0d6)]
// CHECK:STDOUT:   %Iterate.impl_witness_table.344 = impl_witness_table (%Core.import_ref.301, %Core.import_ref.89f, %Core.import_ref.3e6, %Core.import_ref.85d), @R.as_type.as.Iterate.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.098: @Int.as.AddAssignWith.impl.%Int.as.AddAssignWith.impl.Op.type.2 (%Int.as.AddAssignWith.impl.Op.type.a58fbe.1) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.AddAssignWith.impl.%Int.as.AddAssignWith.impl.Op.2 (constants.%Int.as.AddAssignWith.impl.Op.45b360.1)]
// CHECK:STDOUT:   %AddAssignWith.impl_witness_table = impl_witness_table (%Core.import_ref.098), @Int.as.AddAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.1b1: @Int.as.AddAssignWith.impl.%Int.as.AddAssignWith.impl.Op.type.1 (%Int.as.AddAssignWith.impl.Op.type.a58fbe.2) = import_ref Core//prelude/types/int, Op, loaded [symbolic = @Int.as.AddAssignWith.impl.%Int.as.AddAssignWith.impl.Op.1 (constants.%Int.as.AddAssignWith.impl.Op.45b360.2)]
// CHECK:STDOUT:   %Core.import_ref.47b: Core.IntLiteral = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.AnyInt.impl.%N (constants.%N.fe9)]
// CHECK:STDOUT:   %Core.import_ref.a64: @Int.as.AnyInt.impl.%Int.as.AnyInt.impl.AsInt.type (%Int.as.AnyInt.impl.AsInt.type.324) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.AnyInt.impl.%Int.as.AnyInt.impl.AsInt (constants.%Int.as.AnyInt.impl.AsInt.ec6)]
// CHECK:STDOUT:   %AnyInt.impl_witness_table = impl_witness_table (%Core.import_ref.47b, %Core.import_ref.a64), @Int.as.AnyInt.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @TestIterate(%R.loc30_17.2: %Iterate_where.type.8bc) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Iterate.lookup_impl_witness: <witness> = lookup_impl_witness %R.loc30_17.1, @Iterate [symbolic = %Iterate.lookup_impl_witness (constants.%Iterate.lookup_impl_witness.c86)]
// CHECK:STDOUT:   %Iterate.facet.loc34_19.5: %Iterate.type = facet_value %R.as_type.loc30_62.1, (%Iterate.lookup_impl_witness) [symbolic = %Iterate.facet.loc34_19.5 (constants.%Iterate.facet.bbb)]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.loc34_19.5) [symbolic = %Iterate.WithSelf.NewCursor.type (constants.%Iterate.WithSelf.NewCursor.type.d2b)]
// CHECK:STDOUT:   %.loc34_19.19: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type, %Iterate.facet.loc34_19.5 [symbolic = %.loc34_19.19 (constants.%.3e2)]
// CHECK:STDOUT:   %impl.elem2.loc34_19.2: @TestIterate.%.loc34_19.19 (%.3e2) = impl_witness_access %Iterate.lookup_impl_witness, element2 [symbolic = %impl.elem2.loc34_19.2 (constants.%impl.elem2.f53)]
// CHECK:STDOUT:   %specific_impl_fn.loc34_19.4: <specific function> = specific_impl_function %impl.elem2.loc34_19.2, @Iterate.WithSelf.NewCursor(%Iterate.facet.loc34_19.5) [symbolic = %specific_impl_fn.loc34_19.4 (constants.%specific_impl_fn.87a)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %ptr: type = ptr_type %as_type [symbolic = %ptr (constants.%ptr.3b4)]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.loc34_19.5) [symbolic = %Iterate.WithSelf.Next.type (constants.%Iterate.WithSelf.Next.type.28b)]
// CHECK:STDOUT:   %.loc34_19.20: type = fn_type_with_self_type %Iterate.WithSelf.Next.type, %Iterate.facet.loc34_19.5 [symbolic = %.loc34_19.20 (constants.%.a8b)]
// CHECK:STDOUT:   %impl.elem3.loc34_19.2: @TestIterate.%.loc34_19.20 (%.a8b) = impl_witness_access %Iterate.lookup_impl_witness, element3 [symbolic = %impl.elem3.loc34_19.2 (constants.%impl.elem3.9c3)]
// CHECK:STDOUT:   %specific_impl_fn.loc34_19.5: <specific function> = specific_impl_function %impl.elem3.loc34_19.2, @Iterate.WithSelf.Next(%Iterate.facet.loc34_19.5) [symbolic = %specific_impl_fn.loc34_19.5 (constants.%specific_impl_fn.ab2)]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%impl.elem1) [symbolic = %Destroy.WithSelf.Op.type (constants.%Destroy.WithSelf.Op.type.515)]
// CHECK:STDOUT:   %.loc34_19.21: type = fn_type_with_self_type %Destroy.WithSelf.Op.type, %impl.elem1 [symbolic = %.loc34_19.21 (constants.%.94f)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %impl.elem1, @Destroy [symbolic = %Destroy.lookup_impl_witness (constants.%Destroy.lookup_impl_witness.5d5)]
// CHECK:STDOUT:   %impl.elem0.loc34_19.2: @TestIterate.%.loc34_19.21 (%.94f) = impl_witness_access %Destroy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc34_19.2 (constants.%impl.elem0.e52)]
// CHECK:STDOUT:   %specific_impl_fn.loc34_19.6: <specific function> = specific_impl_function %impl.elem0.loc34_19.2, @Destroy.WithSelf.Op(%impl.elem1) [symbolic = %specific_impl_fn.loc34_19.6 (constants.%specific_impl_fn.12e)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%r.param: @TestIterate.%R.as_type.loc30_62.1 (%R.as_type.67d)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %i.patt: %pattern_type.6b6 = value_binding_pattern i [concrete = constants.%i.patt]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %r.ref: @TestIterate.%R.as_type.loc30_62.1 (%R.as_type.67d) = name_ref r, %r
// CHECK:STDOUT:     %impl.elem2.loc34_19.1: @TestIterate.%.loc34_19.19 (%.3e2) = impl_witness_access constants.%Iterate.lookup_impl_witness.c86, element2 [symbolic = %impl.elem2.loc34_19.2 (constants.%impl.elem2.f53)]
// CHECK:STDOUT:     %bound_method.loc34_19.1: <bound method> = bound_method %r.ref, %impl.elem2.loc34_19.1
// CHECK:STDOUT:     %Iterate.facet.loc34_19.1: %Iterate.type = facet_value constants.%R.as_type.67d, (constants.%Iterate.lookup_impl_witness.c86) [symbolic = %Iterate.facet.loc34_19.5 (constants.%Iterate.facet.bbb)]
// CHECK:STDOUT:     %.loc34_19.1: %Iterate.type = converted constants.%R.as_type.67d, %Iterate.facet.loc34_19.1 [symbolic = %Iterate.facet.loc34_19.5 (constants.%Iterate.facet.bbb)]
// CHECK:STDOUT:     %Iterate.facet.loc34_19.2: %Iterate.type = facet_value constants.%R.as_type.67d, (constants.%Iterate.lookup_impl_witness.c86) [symbolic = %Iterate.facet.loc34_19.5 (constants.%Iterate.facet.bbb)]
// CHECK:STDOUT:     %.loc34_19.2: %Iterate.type = converted constants.%R.as_type.67d, %Iterate.facet.loc34_19.2 [symbolic = %Iterate.facet.loc34_19.5 (constants.%Iterate.facet.bbb)]
// CHECK:STDOUT:     %specific_impl_fn.loc34_19.1: <specific function> = specific_impl_function %impl.elem2.loc34_19.1, @Iterate.WithSelf.NewCursor(constants.%Iterate.facet.bbb) [symbolic = %specific_impl_fn.loc34_19.4 (constants.%specific_impl_fn.87a)]
// CHECK:STDOUT:     %bound_method.loc34_19.2: <bound method> = bound_method %r.ref, %specific_impl_fn.loc34_19.1
// CHECK:STDOUT:     %var: ref @TestIterate.%as_type (%as_type.ca2) = var invalid
// CHECK:STDOUT:     %Iterate.WithSelf.NewCursor.call: init @TestIterate.%as_type (%as_type.ca2) to %var = call %bound_method.loc34_19.2(%r.ref)
// CHECK:STDOUT:     assign %var, %Iterate.WithSelf.NewCursor.call
// CHECK:STDOUT:     br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.next:
// CHECK:STDOUT:     %addr: @TestIterate.%ptr (%ptr.3b4) = addr_of %var
// CHECK:STDOUT:     %impl.elem3.loc34_19.1: @TestIterate.%.loc34_19.20 (%.a8b) = impl_witness_access constants.%Iterate.lookup_impl_witness.c86, element3 [symbolic = %impl.elem3.loc34_19.2 (constants.%impl.elem3.9c3)]
// CHECK:STDOUT:     %bound_method.loc34_19.3: <bound method> = bound_method %r.ref, %impl.elem3.loc34_19.1
// CHECK:STDOUT:     %Iterate.facet.loc34_19.3: %Iterate.type = facet_value constants.%R.as_type.67d, (constants.%Iterate.lookup_impl_witness.c86) [symbolic = %Iterate.facet.loc34_19.5 (constants.%Iterate.facet.bbb)]
// CHECK:STDOUT:     %.loc34_19.3: %Iterate.type = converted constants.%R.as_type.67d, %Iterate.facet.loc34_19.3 [symbolic = %Iterate.facet.loc34_19.5 (constants.%Iterate.facet.bbb)]
// CHECK:STDOUT:     %Iterate.facet.loc34_19.4: %Iterate.type = facet_value constants.%R.as_type.67d, (constants.%Iterate.lookup_impl_witness.c86) [symbolic = %Iterate.facet.loc34_19.5 (constants.%Iterate.facet.bbb)]
// CHECK:STDOUT:     %.loc34_19.4: %Iterate.type = converted constants.%R.as_type.67d, %Iterate.facet.loc34_19.4 [symbolic = %Iterate.facet.loc34_19.5 (constants.%Iterate.facet.bbb)]
// CHECK:STDOUT:     %.loc34_19.5: %Destroy.type = converted constants.%as_type.ca2, constants.%impl.elem1.d31 [symbolic = %impl.elem1 (constants.%impl.elem1.d31)]
// CHECK:STDOUT:     %.loc34_19.6: %Destroy.type = converted constants.%as_type.ca2, constants.%impl.elem1.d31 [symbolic = %impl.elem1 (constants.%impl.elem1.d31)]
// CHECK:STDOUT:     %specific_impl_fn.loc34_19.2: <specific function> = specific_impl_function %impl.elem3.loc34_19.1, @Iterate.WithSelf.Next(constants.%Iterate.facet.bbb) [symbolic = %specific_impl_fn.loc34_19.5 (constants.%specific_impl_fn.ab2)]
// CHECK:STDOUT:     %bound_method.loc34_19.4: <bound method> = bound_method %r.ref, %specific_impl_fn.loc34_19.2
// CHECK:STDOUT:     %.loc34_19.7: ref %Optional.e28 = temporary_storage
// CHECK:STDOUT:     %Iterate.WithSelf.Next.call: init %Optional.e28 to %.loc34_19.7 = call %bound_method.loc34_19.4(%r.ref, %addr)
// CHECK:STDOUT:     %.loc34_19.8: ref %Optional.e28 = temporary %.loc34_19.7, %Iterate.WithSelf.Next.call
// CHECK:STDOUT:     %.loc34_19.9: %Optional.HasValue.type.c6d = specific_constant imports.%Core.import_ref.bb6, @Optional(constants.%OptionalStorage.facet.172) [concrete = constants.%Optional.HasValue.89b]
// CHECK:STDOUT:     %HasValue.ref: %Optional.HasValue.type.c6d = name_ref HasValue, %.loc34_19.9 [concrete = constants.%Optional.HasValue.89b]
// CHECK:STDOUT:     %Optional.HasValue.bound: <bound method> = bound_method %.loc34_19.8, %HasValue.ref
// CHECK:STDOUT:     %Optional.HasValue.specific_fn: <specific function> = specific_function %HasValue.ref, @Optional.HasValue(constants.%OptionalStorage.facet.172) [concrete = constants.%Optional.HasValue.specific_fn]
// CHECK:STDOUT:     %bound_method.loc34_19.5: <bound method> = bound_method %.loc34_19.8, %Optional.HasValue.specific_fn
// CHECK:STDOUT:     %.loc34_19.10: %Optional.e28 = acquire_value %.loc34_19.8
// CHECK:STDOUT:     %Optional.HasValue.call: init bool = call %bound_method.loc34_19.5(%.loc34_19.10)
// CHECK:STDOUT:     %.loc34_19.11: bool = value_of_initializer %Optional.HasValue.call
// CHECK:STDOUT:     %.loc34_19.12: bool = converted %Optional.HasValue.call, %.loc34_19.11
// CHECK:STDOUT:     if %.loc34_19.12 br !for.body else br !for.done
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.body:
// CHECK:STDOUT:     %.loc34_19.13: %Optional.Get.type.7e6 = specific_constant imports.%Core.import_ref.daba, @Optional(constants.%OptionalStorage.facet.172) [concrete = constants.%Optional.Get.722]
// CHECK:STDOUT:     %Get.ref: %Optional.Get.type.7e6 = name_ref Get, %.loc34_19.13 [concrete = constants.%Optional.Get.722]
// CHECK:STDOUT:     %Optional.Get.bound: <bound method> = bound_method %.loc34_19.8, %Get.ref
// CHECK:STDOUT:     %Optional.Get.specific_fn: <specific function> = specific_function %Get.ref, @Optional.Get(constants.%OptionalStorage.facet.172) [concrete = constants.%Optional.Get.specific_fn]
// CHECK:STDOUT:     %bound_method.loc34_19.6: <bound method> = bound_method %.loc34_19.8, %Optional.Get.specific_fn
// CHECK:STDOUT:     %.loc34_19.14: %Optional.e28 = acquire_value %.loc34_19.8
// CHECK:STDOUT:     %Optional.Get.call: init %i32 = call %bound_method.loc34_19.6(%.loc34_19.14)
// CHECK:STDOUT:     %.loc34_19.15: %i32 = value_of_initializer %Optional.Get.call
// CHECK:STDOUT:     %.loc34_19.16: %i32 = converted %Optional.Get.call, %.loc34_19.15
// CHECK:STDOUT:     %i32.loc34: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %i: %i32 = wrapper_binding i, %.loc34_19.16
// CHECK:STDOUT:     %sum.ref: ref %i32 = name_ref sum, %sum
// CHECK:STDOUT:     %i.ref: %i32 = name_ref i, %i
// CHECK:STDOUT:     %impl.elem0.loc35: %.b28 = impl_witness_access constants.%AddAssignWith.impl_witness.22e, element0 [concrete = constants.%Int.as.AddAssignWith.impl.Op.00e760.2]
// CHECK:STDOUT:     %bound_method.loc35_9.1: <bound method> = bound_method %sum.ref, %impl.elem0.loc35
// CHECK:STDOUT:     %specific_fn.loc35: <specific function> = specific_function %impl.elem0.loc35, @Int.as.AddAssignWith.impl.Op.1(constants.%int_32, constants.%ImplicitAs.facet.eaa) [concrete = constants.%Int.as.AddAssignWith.impl.Op.specific_fn.b14d9d.1]
// CHECK:STDOUT:     %bound_method.loc35_9.2: <bound method> = bound_method %sum.ref, %specific_fn.loc35
// CHECK:STDOUT:     %.loc35: %Int.as.AddAssignWith.impl.Op.type.73460a.1 = specific_constant imports.%Core.Op.1b1, @Int.as.AddAssignWith.impl(constants.%int_32, constants.%ImplicitAs.facet.eaa) [concrete = constants.%Int.as.AddAssignWith.impl.Op.00e760.1]
// CHECK:STDOUT:     %Op.ref: %Int.as.AddAssignWith.impl.Op.type.73460a.1 = name_ref Op, %.loc35 [concrete = constants.%Int.as.AddAssignWith.impl.Op.00e760.1]
// CHECK:STDOUT:     %Int.as.AddAssignWith.impl.Op.bound: <bound method> = bound_method %sum.ref, %Op.ref
// CHECK:STDOUT:     %Int.as.AddAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref, @Int.as.AddAssignWith.impl.Op.2(constants.%int_32, constants.%ImplicitAs.facet.eaa) [concrete = constants.%Int.as.AddAssignWith.impl.Op.specific_fn.b14d9d.2]
// CHECK:STDOUT:     %bound_method.loc35_9.3: <bound method> = bound_method %sum.ref, %Int.as.AddAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:     %Int.as.AddAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc35_9.3(%sum.ref, %i.ref)
// CHECK:STDOUT:     br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.done:
// CHECK:STDOUT:     %Destroy.Op.bound.loc34: <bound method> = bound_method %.loc34_19.8, constants.%Destroy.Op.1a2547.9
// CHECK:STDOUT:     %Destroy.Op.call.loc34: init %empty_tuple.type = call %Destroy.Op.bound.loc34(%.loc34_19.8)
// CHECK:STDOUT:     %impl.elem0.loc34_19.1: @TestIterate.%.loc34_19.21 (%.94f) = impl_witness_access constants.%Destroy.lookup_impl_witness.5d5, element0 [symbolic = %impl.elem0.loc34_19.2 (constants.%impl.elem0.e52)]
// CHECK:STDOUT:     %bound_method.loc34_19.7: <bound method> = bound_method %var, %impl.elem0.loc34_19.1
// CHECK:STDOUT:     %.loc34_19.17: %Destroy.type = converted constants.%as_type.ca2, constants.%impl.elem1.d31 [symbolic = %impl.elem1 (constants.%impl.elem1.d31)]
// CHECK:STDOUT:     %.loc34_19.18: %Destroy.type = converted constants.%as_type.ca2, constants.%impl.elem1.d31 [symbolic = %impl.elem1 (constants.%impl.elem1.d31)]
// CHECK:STDOUT:     %specific_impl_fn.loc34_19.3: <specific function> = specific_impl_function %impl.elem0.loc34_19.1, @Destroy.WithSelf.Op(constants.%impl.elem1.d31) [symbolic = %specific_impl_fn.loc34_19.6 (constants.%specific_impl_fn.12e)]
// CHECK:STDOUT:     %bound_method.loc34_19.8: <bound method> = bound_method %var, %specific_impl_fn.loc34_19.3
// CHECK:STDOUT:     %Destroy.WithSelf.Op.call: init %empty_tuple.type = call %bound_method.loc34_19.8(%var)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc34_19.1(%self.param: ref %.1d6a7) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc34_19.2(%self.param: ref %MaybeUnformed.023) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc34_19.3(%self.param: ref bool) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc34_19.4(%self.param: ref %struct_type.value.has_value.ad0) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc34_19.5(%self.param: ref %DefaultOptionalStorage.970) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc34_19.6(%self.param: ref %Optional.e28) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestRangeFor(%m.param: ref %MutableRange, %c.param: %ConstRange) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %i.patt.loc49: %pattern_type.6b6 = value_binding_pattern i [concrete = constants.%i.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %m.ref: ref %MutableRange = name_ref m, %m
// CHECK:STDOUT:   %impl.elem2.loc49: %.d6a = impl_witness_access constants.%Iterate.impl_witness.cea, element2 [concrete = constants.%R.as_type.as.Iterate.impl.NewCursor.8d2]
// CHECK:STDOUT:   %bound_method.loc49_19.1: <bound method> = bound_method %m.ref, %impl.elem2.loc49
// CHECK:STDOUT:   %facet_value.loc49_19.1: %CppRangeForIterate_where.type = facet_value constants.%MutableRange, (constants.%custom_witness.df9cc1.2, constants.%Copy.impl_witness.0e0, constants.%custom_witness.ed3, constants.%custom_witness.955, constants.%custom_witness.1f5, constants.%custom_witness.ef8, constants.%custom_witness.981, constants.%custom_witness.444) [concrete = constants.%facet_value.cd6]
// CHECK:STDOUT:   %.loc49_19.1: %CppRangeForIterate_where.type = converted constants.%MutableRange, %facet_value.loc49_19.1 [concrete = constants.%facet_value.cd6]
// CHECK:STDOUT:   %facet_value.loc49_19.2: %CppRangeForIterate_where.type = facet_value constants.%MutableRange, (constants.%custom_witness.df9cc1.2, constants.%Copy.impl_witness.0e0, constants.%custom_witness.ed3, constants.%custom_witness.955, constants.%custom_witness.1f5, constants.%custom_witness.ef8, constants.%custom_witness.981, constants.%custom_witness.444) [concrete = constants.%facet_value.cd6]
// CHECK:STDOUT:   %.loc49_19.2: %CppRangeForIterate_where.type = converted constants.%MutableRange, %facet_value.loc49_19.2 [concrete = constants.%facet_value.cd6]
// CHECK:STDOUT:   %specific_fn.loc49_19.1: <specific function> = specific_function %impl.elem2.loc49, @R.as_type.as.Iterate.impl.NewCursor(constants.%facet_value.cd6) [concrete = constants.%R.as_type.as.Iterate.impl.NewCursor.specific_fn.bf1]
// CHECK:STDOUT:   %bound_method.loc49_19.2: <bound method> = bound_method %m.ref, %specific_fn.loc49_19.1
// CHECK:STDOUT:   %var.loc49: ref %tuple.type.508 = var invalid
// CHECK:STDOUT:   %.loc49_18.1: %MutableRange = acquire_value %m.ref
// CHECK:STDOUT:   %R.as_type.as.Iterate.impl.NewCursor.call.loc49: init %tuple.type.508 to %var.loc49 = call %bound_method.loc49_19.2(%.loc49_18.1)
// CHECK:STDOUT:   assign %var.loc49, %R.as_type.as.Iterate.impl.NewCursor.call.loc49
// CHECK:STDOUT:   br !for.next.loc49
// CHECK:STDOUT:
// CHECK:STDOUT: !for.next.loc49:
// CHECK:STDOUT:   %addr.loc49: %ptr.45c = addr_of %var.loc49
// CHECK:STDOUT:   %impl.elem3.loc49: %.d23 = impl_witness_access constants.%Iterate.impl_witness.cea, element3 [concrete = constants.%R.as_type.as.Iterate.impl.Next.34d]
// CHECK:STDOUT:   %bound_method.loc49_19.3: <bound method> = bound_method %m.ref, %impl.elem3.loc49
// CHECK:STDOUT:   %facet_value.loc49_19.3: %CppRangeForIterate_where.type = facet_value constants.%MutableRange, (constants.%custom_witness.df9cc1.2, constants.%Copy.impl_witness.0e0, constants.%custom_witness.ed3, constants.%custom_witness.955, constants.%custom_witness.1f5, constants.%custom_witness.ef8, constants.%custom_witness.981, constants.%custom_witness.444) [concrete = constants.%facet_value.cd6]
// CHECK:STDOUT:   %.loc49_19.3: %CppRangeForIterate_where.type = converted constants.%MutableRange, %facet_value.loc49_19.3 [concrete = constants.%facet_value.cd6]
// CHECK:STDOUT:   %facet_value.loc49_19.4: %CppRangeForIterate_where.type = facet_value constants.%MutableRange, (constants.%custom_witness.df9cc1.2, constants.%Copy.impl_witness.0e0, constants.%custom_witness.ed3, constants.%custom_witness.955, constants.%custom_witness.1f5, constants.%custom_witness.ef8, constants.%custom_witness.981, constants.%custom_witness.444) [concrete = constants.%facet_value.cd6]
// CHECK:STDOUT:   %.loc49_19.4: %CppRangeForIterate_where.type = converted constants.%MutableRange, %facet_value.loc49_19.4 [concrete = constants.%facet_value.cd6]
// CHECK:STDOUT:   %specific_fn.loc49_19.2: <specific function> = specific_function %impl.elem3.loc49, @R.as_type.as.Iterate.impl.Next(constants.%facet_value.cd6) [concrete = constants.%R.as_type.as.Iterate.impl.Next.specific_fn.dd5]
// CHECK:STDOUT:   %bound_method.loc49_19.4: <bound method> = bound_method %m.ref, %specific_fn.loc49_19.2
// CHECK:STDOUT:   %.loc49_19.5: ref %Optional.e28 = temporary_storage
// CHECK:STDOUT:   %.loc49_18.2: %MutableRange = acquire_value %m.ref
// CHECK:STDOUT:   %R.as_type.as.Iterate.impl.Next.call.loc49: init %Optional.e28 to %.loc49_19.5 = call %bound_method.loc49_19.4(%.loc49_18.2, %addr.loc49)
// CHECK:STDOUT:   %.loc49_19.6: ref %Optional.e28 = temporary %.loc49_19.5, %R.as_type.as.Iterate.impl.Next.call.loc49
// CHECK:STDOUT:   %.loc49_19.7: %Optional.HasValue.type.c6d = specific_constant imports.%Core.import_ref.bb6, @Optional(constants.%OptionalStorage.facet.172) [concrete = constants.%Optional.HasValue.89b]
// CHECK:STDOUT:   %HasValue.ref.loc49: %Optional.HasValue.type.c6d = name_ref HasValue, %.loc49_19.7 [concrete = constants.%Optional.HasValue.89b]
// CHECK:STDOUT:   %Optional.HasValue.bound.loc49: <bound method> = bound_method %.loc49_19.6, %HasValue.ref.loc49
// CHECK:STDOUT:   %Optional.HasValue.specific_fn.loc49: <specific function> = specific_function %HasValue.ref.loc49, @Optional.HasValue(constants.%OptionalStorage.facet.172) [concrete = constants.%Optional.HasValue.specific_fn]
// CHECK:STDOUT:   %bound_method.loc49_19.5: <bound method> = bound_method %.loc49_19.6, %Optional.HasValue.specific_fn.loc49
// CHECK:STDOUT:   %.loc49_19.8: %Optional.e28 = acquire_value %.loc49_19.6
// CHECK:STDOUT:   %Optional.HasValue.call.loc49: init bool = call %bound_method.loc49_19.5(%.loc49_19.8)
// CHECK:STDOUT:   %.loc49_19.9: bool = value_of_initializer %Optional.HasValue.call.loc49
// CHECK:STDOUT:   %.loc49_19.10: bool = converted %Optional.HasValue.call.loc49, %.loc49_19.9
// CHECK:STDOUT:   if %.loc49_19.10 br !for.body.loc49 else br !for.done.loc49
// CHECK:STDOUT:
// CHECK:STDOUT: !for.body.loc49:
// CHECK:STDOUT:   %.loc49_19.11: %Optional.Get.type.7e6 = specific_constant imports.%Core.import_ref.daba, @Optional(constants.%OptionalStorage.facet.172) [concrete = constants.%Optional.Get.722]
// CHECK:STDOUT:   %Get.ref.loc49: %Optional.Get.type.7e6 = name_ref Get, %.loc49_19.11 [concrete = constants.%Optional.Get.722]
// CHECK:STDOUT:   %Optional.Get.bound.loc49: <bound method> = bound_method %.loc49_19.6, %Get.ref.loc49
// CHECK:STDOUT:   %Optional.Get.specific_fn.loc49: <specific function> = specific_function %Get.ref.loc49, @Optional.Get(constants.%OptionalStorage.facet.172) [concrete = constants.%Optional.Get.specific_fn]
// CHECK:STDOUT:   %bound_method.loc49_19.6: <bound method> = bound_method %.loc49_19.6, %Optional.Get.specific_fn.loc49
// CHECK:STDOUT:   %.loc49_19.12: %Optional.e28 = acquire_value %.loc49_19.6
// CHECK:STDOUT:   %Optional.Get.call.loc49: init %i32 = call %bound_method.loc49_19.6(%.loc49_19.12)
// CHECK:STDOUT:   %.loc49_19.13: %i32 = value_of_initializer %Optional.Get.call.loc49
// CHECK:STDOUT:   %.loc49_19.14: %i32 = converted %Optional.Get.call.loc49, %.loc49_19.13
// CHECK:STDOUT:   %i32.loc49: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %i.loc49: %i32 = wrapper_binding i, %.loc49_19.14
// CHECK:STDOUT:   %sum.ref.loc50: ref %i32 = name_ref sum, %sum
// CHECK:STDOUT:   %i.ref.loc50: %i32 = name_ref i, %i.loc49
// CHECK:STDOUT:   %impl.elem0.loc50: %.b28 = impl_witness_access constants.%AddAssignWith.impl_witness.22e, element0 [concrete = constants.%Int.as.AddAssignWith.impl.Op.00e760.2]
// CHECK:STDOUT:   %bound_method.loc50_9.1: <bound method> = bound_method %sum.ref.loc50, %impl.elem0.loc50
// CHECK:STDOUT:   %specific_fn.loc50: <specific function> = specific_function %impl.elem0.loc50, @Int.as.AddAssignWith.impl.Op.1(constants.%int_32, constants.%ImplicitAs.facet.eaa) [concrete = constants.%Int.as.AddAssignWith.impl.Op.specific_fn.b14d9d.1]
// CHECK:STDOUT:   %bound_method.loc50_9.2: <bound method> = bound_method %sum.ref.loc50, %specific_fn.loc50
// CHECK:STDOUT:   %.loc50: %Int.as.AddAssignWith.impl.Op.type.73460a.1 = specific_constant imports.%Core.Op.1b1, @Int.as.AddAssignWith.impl(constants.%int_32, constants.%ImplicitAs.facet.eaa) [concrete = constants.%Int.as.AddAssignWith.impl.Op.00e760.1]
// CHECK:STDOUT:   %Op.ref.loc50: %Int.as.AddAssignWith.impl.Op.type.73460a.1 = name_ref Op, %.loc50 [concrete = constants.%Int.as.AddAssignWith.impl.Op.00e760.1]
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.bound.loc50: <bound method> = bound_method %sum.ref.loc50, %Op.ref.loc50
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.specific_fn.loc50: <specific function> = specific_function %Op.ref.loc50, @Int.as.AddAssignWith.impl.Op.2(constants.%int_32, constants.%ImplicitAs.facet.eaa) [concrete = constants.%Int.as.AddAssignWith.impl.Op.specific_fn.b14d9d.2]
// CHECK:STDOUT:   %bound_method.loc50_9.3: <bound method> = bound_method %sum.ref.loc50, %Int.as.AddAssignWith.impl.Op.specific_fn.loc50
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.call.loc50: init %empty_tuple.type = call %bound_method.loc50_9.3(%sum.ref.loc50, %i.ref.loc50)
// CHECK:STDOUT:   br !for.next.loc49
// CHECK:STDOUT:
// CHECK:STDOUT: !for.done.loc49:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %i.patt.loc55: %pattern_type.6b6 = value_binding_pattern i [concrete = constants.%i.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c.ref: %ConstRange = name_ref c, %c
// CHECK:STDOUT:   %impl.elem2.loc55: %.744 = impl_witness_access constants.%Iterate.impl_witness.12a, element2 [concrete = constants.%R.as_type.as.Iterate.impl.NewCursor.2f6]
// CHECK:STDOUT:   %bound_method.loc55_19.1: <bound method> = bound_method %c.ref, %impl.elem2.loc55
// CHECK:STDOUT:   %facet_value.loc55_19.1: %CppRangeForIterate_where.type = facet_value constants.%ConstRange, (constants.%custom_witness.df9cc1.2, constants.%Copy.impl_witness.0e0, constants.%custom_witness.23e, constants.%custom_witness.6e8, constants.%custom_witness.150, constants.%custom_witness.adb, constants.%custom_witness.941, constants.%custom_witness.746) [concrete = constants.%facet_value.e88]
// CHECK:STDOUT:   %.loc55_19.1: %CppRangeForIterate_where.type = converted constants.%ConstRange, %facet_value.loc55_19.1 [concrete = constants.%facet_value.e88]
// CHECK:STDOUT:   %facet_value.loc55_19.2: %CppRangeForIterate_where.type = facet_value constants.%ConstRange, (constants.%custom_witness.df9cc1.2, constants.%Copy.impl_witness.0e0, constants.%custom_witness.23e, constants.%custom_witness.6e8, constants.%custom_witness.150, constants.%custom_witness.adb, constants.%custom_witness.941, constants.%custom_witness.746) [concrete = constants.%facet_value.e88]
// CHECK:STDOUT:   %.loc55_19.2: %CppRangeForIterate_where.type = converted constants.%ConstRange, %facet_value.loc55_19.2 [concrete = constants.%facet_value.e88]
// CHECK:STDOUT:   %specific_fn.loc55_19.1: <specific function> = specific_function %impl.elem2.loc55, @R.as_type.as.Iterate.impl.NewCursor(constants.%facet_value.e88) [concrete = constants.%R.as_type.as.Iterate.impl.NewCursor.specific_fn.a05]
// CHECK:STDOUT:   %bound_method.loc55_19.2: <bound method> = bound_method %c.ref, %specific_fn.loc55_19.1
// CHECK:STDOUT:   %var.loc55: ref %tuple.type.69f = var invalid
// CHECK:STDOUT:   %R.as_type.as.Iterate.impl.NewCursor.call.loc55: init %tuple.type.69f to %var.loc55 = call %bound_method.loc55_19.2(%c.ref)
// CHECK:STDOUT:   assign %var.loc55, %R.as_type.as.Iterate.impl.NewCursor.call.loc55
// CHECK:STDOUT:   br !for.next.loc55
// CHECK:STDOUT:
// CHECK:STDOUT: !for.next.loc55:
// CHECK:STDOUT:   %addr.loc55: %ptr.c5c = addr_of %var.loc55
// CHECK:STDOUT:   %impl.elem3.loc55: %.03b = impl_witness_access constants.%Iterate.impl_witness.12a, element3 [concrete = constants.%R.as_type.as.Iterate.impl.Next.393]
// CHECK:STDOUT:   %bound_method.loc55_19.3: <bound method> = bound_method %c.ref, %impl.elem3.loc55
// CHECK:STDOUT:   %facet_value.loc55_19.3: %CppRangeForIterate_where.type = facet_value constants.%ConstRange, (constants.%custom_witness.df9cc1.2, constants.%Copy.impl_witness.0e0, constants.%custom_witness.23e, constants.%custom_witness.6e8, constants.%custom_witness.150, constants.%custom_witness.adb, constants.%custom_witness.941, constants.%custom_witness.746) [concrete = constants.%facet_value.e88]
// CHECK:STDOUT:   %.loc55_19.3: %CppRangeForIterate_where.type = converted constants.%ConstRange, %facet_value.loc55_19.3 [concrete = constants.%facet_value.e88]
// CHECK:STDOUT:   %facet_value.loc55_19.4: %CppRangeForIterate_where.type = facet_value constants.%ConstRange, (constants.%custom_witness.df9cc1.2, constants.%Copy.impl_witness.0e0, constants.%custom_witness.23e, constants.%custom_witness.6e8, constants.%custom_witness.150, constants.%custom_witness.adb, constants.%custom_witness.941, constants.%custom_witness.746) [concrete = constants.%facet_value.e88]
// CHECK:STDOUT:   %.loc55_19.4: %CppRangeForIterate_where.type = converted constants.%ConstRange, %facet_value.loc55_19.4 [concrete = constants.%facet_value.e88]
// CHECK:STDOUT:   %specific_fn.loc55_19.2: <specific function> = specific_function %impl.elem3.loc55, @R.as_type.as.Iterate.impl.Next(constants.%facet_value.e88) [concrete = constants.%R.as_type.as.Iterate.impl.Next.specific_fn.4cc]
// CHECK:STDOUT:   %bound_method.loc55_19.4: <bound method> = bound_method %c.ref, %specific_fn.loc55_19.2
// CHECK:STDOUT:   %.loc55_19.5: ref %Optional.e28 = temporary_storage
// CHECK:STDOUT:   %R.as_type.as.Iterate.impl.Next.call.loc55: init %Optional.e28 to %.loc55_19.5 = call %bound_method.loc55_19.4(%c.ref, %addr.loc55)
// CHECK:STDOUT:   %.loc55_19.6: ref %Optional.e28 = temporary %.loc55_19.5, %R.as_type.as.Iterate.impl.Next.call.loc55
// CHECK:STDOUT:   %.loc55_19.7: %Optional.HasValue.type.c6d = specific_constant imports.%Core.import_ref.bb6, @Optional(constants.%OptionalStorage.facet.172) [concrete = constants.%Optional.HasValue.89b]
// CHECK:STDOUT:   %HasValue.ref.loc55: %Optional.HasValue.type.c6d = name_ref HasValue, %.loc55_19.7 [concrete = constants.%Optional.HasValue.89b]
// CHECK:STDOUT:   %Optional.HasValue.bound.loc55: <bound method> = bound_method %.loc55_19.6, %HasValue.ref.loc55
// CHECK:STDOUT:   %Optional.HasValue.specific_fn.loc55: <specific function> = specific_function %HasValue.ref.loc55, @Optional.HasValue(constants.%OptionalStorage.facet.172) [concrete = constants.%Optional.HasValue.specific_fn]
// CHECK:STDOUT:   %bound_method.loc55_19.5: <bound method> = bound_method %.loc55_19.6, %Optional.HasValue.specific_fn.loc55
// CHECK:STDOUT:   %.loc55_19.8: %Optional.e28 = acquire_value %.loc55_19.6
// CHECK:STDOUT:   %Optional.HasValue.call.loc55: init bool = call %bound_method.loc55_19.5(%.loc55_19.8)
// CHECK:STDOUT:   %.loc55_19.9: bool = value_of_initializer %Optional.HasValue.call.loc55
// CHECK:STDOUT:   %.loc55_19.10: bool = converted %Optional.HasValue.call.loc55, %.loc55_19.9
// CHECK:STDOUT:   if %.loc55_19.10 br !for.body.loc55 else br !for.done.loc55
// CHECK:STDOUT:
// CHECK:STDOUT: !for.body.loc55:
// CHECK:STDOUT:   %.loc55_19.11: %Optional.Get.type.7e6 = specific_constant imports.%Core.import_ref.daba, @Optional(constants.%OptionalStorage.facet.172) [concrete = constants.%Optional.Get.722]
// CHECK:STDOUT:   %Get.ref.loc55: %Optional.Get.type.7e6 = name_ref Get, %.loc55_19.11 [concrete = constants.%Optional.Get.722]
// CHECK:STDOUT:   %Optional.Get.bound.loc55: <bound method> = bound_method %.loc55_19.6, %Get.ref.loc55
// CHECK:STDOUT:   %Optional.Get.specific_fn.loc55: <specific function> = specific_function %Get.ref.loc55, @Optional.Get(constants.%OptionalStorage.facet.172) [concrete = constants.%Optional.Get.specific_fn]
// CHECK:STDOUT:   %bound_method.loc55_19.6: <bound method> = bound_method %.loc55_19.6, %Optional.Get.specific_fn.loc55
// CHECK:STDOUT:   %.loc55_19.12: %Optional.e28 = acquire_value %.loc55_19.6
// CHECK:STDOUT:   %Optional.Get.call.loc55: init %i32 = call %bound_method.loc55_19.6(%.loc55_19.12)
// CHECK:STDOUT:   %.loc55_19.13: %i32 = value_of_initializer %Optional.Get.call.loc55
// CHECK:STDOUT:   %.loc55_19.14: %i32 = converted %Optional.Get.call.loc55, %.loc55_19.13
// CHECK:STDOUT:   %i32.loc55: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %i.loc55: %i32 = wrapper_binding i, %.loc55_19.14
// CHECK:STDOUT:   %sum.ref.loc56: ref %i32 = name_ref sum, %sum
// CHECK:STDOUT:   %i.ref.loc56: %i32 = name_ref i, %i.loc55
// CHECK:STDOUT:   %impl.elem0.loc56: %.b28 = impl_witness_access constants.%AddAssignWith.impl_witness.22e, element0 [concrete = constants.%Int.as.AddAssignWith.impl.Op.00e760.2]
// CHECK:STDOUT:   %bound_method.loc56_9.1: <bound method> = bound_method %sum.ref.loc56, %impl.elem0.loc56
// CHECK:STDOUT:   %specific_fn.loc56: <specific function> = specific_function %impl.elem0.loc56, @Int.as.AddAssignWith.impl.Op.1(constants.%int_32, constants.%ImplicitAs.facet.eaa) [concrete = constants.%Int.as.AddAssignWith.impl.Op.specific_fn.b14d9d.1]
// CHECK:STDOUT:   %bound_method.loc56_9.2: <bound method> = bound_method %sum.ref.loc56, %specific_fn.loc56
// CHECK:STDOUT:   %.loc56: %Int.as.AddAssignWith.impl.Op.type.73460a.1 = specific_constant imports.%Core.Op.1b1, @Int.as.AddAssignWith.impl(constants.%int_32, constants.%ImplicitAs.facet.eaa) [concrete = constants.%Int.as.AddAssignWith.impl.Op.00e760.1]
// CHECK:STDOUT:   %Op.ref.loc56: %Int.as.AddAssignWith.impl.Op.type.73460a.1 = name_ref Op, %.loc56 [concrete = constants.%Int.as.AddAssignWith.impl.Op.00e760.1]
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.bound.loc56: <bound method> = bound_method %sum.ref.loc56, %Op.ref.loc56
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.specific_fn.loc56: <specific function> = specific_function %Op.ref.loc56, @Int.as.AddAssignWith.impl.Op.2(constants.%int_32, constants.%ImplicitAs.facet.eaa) [concrete = constants.%Int.as.AddAssignWith.impl.Op.specific_fn.b14d9d.2]
// CHECK:STDOUT:   %bound_method.loc56_9.3: <bound method> = bound_method %sum.ref.loc56, %Int.as.AddAssignWith.impl.Op.specific_fn.loc56
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.call.loc56: init %empty_tuple.type = call %bound_method.loc56_9.3(%sum.ref.loc56, %i.ref.loc56)
// CHECK:STDOUT:   br !for.next.loc55
// CHECK:STDOUT:
// CHECK:STDOUT: !for.done.loc55:
// CHECK:STDOUT:   %Destroy.Op.bound.loc55_19.1: <bound method> = bound_method %.loc55_19.6, constants.%Destroy.Op.1a2547.9
// CHECK:STDOUT:   %Destroy.Op.call.loc55_19.1: init %empty_tuple.type = call %Destroy.Op.bound.loc55_19.1(%.loc55_19.6)
// CHECK:STDOUT:   %Destroy.Op.bound.loc55_19.2: <bound method> = bound_method %var.loc55, constants.%Destroy.Op.1a2547.11
// CHECK:STDOUT:   %Destroy.Op.call.loc55_19.2: init %empty_tuple.type = call %Destroy.Op.bound.loc55_19.2(%var.loc55)
// CHECK:STDOUT:   %Destroy.Op.bound.loc49_19.1: <bound method> = bound_method %.loc49_19.6, constants.%Destroy.Op.1a2547.9
// CHECK:STDOUT:   %Destroy.Op.call.loc49_19.1: init %empty_tuple.type = call %Destroy.Op.bound.loc49_19.1(%.loc49_19.6)
// CHECK:STDOUT:   %Destroy.Op.bound.loc49_19.2: <bound method> = bound_method %var.loc49, constants.%Destroy.Op.1a2547.10
// CHECK:STDOUT:   %Destroy.Op.call.loc49_19.2: init %empty_tuple.type = call %Destroy.Op.bound.loc49_19.2(%var.loc49)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestIterate(constants.%R.891) {
// CHECK:STDOUT:   %R.patt.loc30_17.2 => constants.%R.patt.860
// CHECK:STDOUT:   %R.loc30_17.1 => constants.%R.891
// CHECK:STDOUT:   %R.as_type.loc30_62.1 => constants.%R.as_type.67d
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.e21
// CHECK:STDOUT:   %r.param_patt.loc30_60.2 => constants.%r.param_patt.fdd
// CHECK:STDOUT:   %r.patt.loc30_60.2 => constants.%r.patt.939
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestIterate(constants.%facet_value.879) {
// CHECK:STDOUT:   %R.patt.loc30_17.2 => constants.%R.patt.860
// CHECK:STDOUT:   %R.loc30_17.1 => constants.%facet_value.879
// CHECK:STDOUT:   %R.as_type.loc30_62.1 => constants.%MutableRange
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.e635
// CHECK:STDOUT:   %r.param_patt.loc30_60.2 => constants.%r.param_patt.c79
// CHECK:STDOUT:   %r.patt.loc30_60.2 => constants.%r.patt.16f
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc30 => constants.%complete_type.357
// CHECK:STDOUT:   %Iterate.lookup_impl_witness => constants.%Iterate.impl_witness.cea
// CHECK:STDOUT:   %Iterate.facet.loc34_19.5 => constants.%Iterate.facet.ffb
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type => constants.%Iterate.WithSelf.NewCursor.type.dcb
// CHECK:STDOUT:   %.loc34_19.19 => constants.%.d6a
// CHECK:STDOUT:   %impl.elem2.loc34_19.2 => constants.%R.as_type.as.Iterate.impl.NewCursor.8d2
// CHECK:STDOUT:   %specific_impl_fn.loc34_19.4 => constants.%R.as_type.as.Iterate.impl.NewCursor.specific_fn.bf1
// CHECK:STDOUT:   %impl.elem1 => constants.%Destroy.facet.30f
// CHECK:STDOUT:   %as_type => constants.%tuple.type.508
// CHECK:STDOUT:   %require_complete.1 => constants.%complete_type.c28
// CHECK:STDOUT:   %ptr => constants.%ptr.45c
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type => constants.%Iterate.WithSelf.Next.type.30e
// CHECK:STDOUT:   %.loc34_19.20 => constants.%.d23
// CHECK:STDOUT:   %impl.elem3.loc34_19.2 => constants.%R.as_type.as.Iterate.impl.Next.34d
// CHECK:STDOUT:   %specific_impl_fn.loc34_19.5 => constants.%R.as_type.as.Iterate.impl.Next.specific_fn.dd5
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.c6c
// CHECK:STDOUT:   %.loc34_19.21 => constants.%.0a4
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.df9cc1.10
// CHECK:STDOUT:   %impl.elem0.loc34_19.2 => constants.%Destroy.Op.1a2547.10
// CHECK:STDOUT:   %specific_impl_fn.loc34_19.6 => constants.%Destroy.Op.1a2547.10
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestIterate(constants.%facet_value.740) {
// CHECK:STDOUT:   %R.patt.loc30_17.2 => constants.%R.patt.860
// CHECK:STDOUT:   %R.loc30_17.1 => constants.%facet_value.740
// CHECK:STDOUT:   %R.as_type.loc30_62.1 => constants.%ConstRange
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.0ba
// CHECK:STDOUT:   %r.param_patt.loc30_60.2 => constants.%r.param_patt.83e
// CHECK:STDOUT:   %r.patt.loc30_60.2 => constants.%r.patt.116
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc30 => constants.%complete_type.357
// CHECK:STDOUT:   %Iterate.lookup_impl_witness => constants.%Iterate.impl_witness.12a
// CHECK:STDOUT:   %Iterate.facet.loc34_19.5 => constants.%Iterate.facet.5eb
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type => constants.%Iterate.WithSelf.NewCursor.type.ab5
// CHECK:STDOUT:   %.loc34_19.19 => constants.%.744
// CHECK:STDOUT:   %impl.elem2.loc34_19.2 => constants.%R.as_type.as.Iterate.impl.NewCursor.2f6
// CHECK:STDOUT:   %specific_impl_fn.loc34_19.4 => constants.%R.as_type.as.Iterate.impl.NewCursor.specific_fn.a05
// CHECK:STDOUT:   %impl.elem1 => constants.%Destroy.facet.e4c
// CHECK:STDOUT:   %as_type => constants.%tuple.type.69f
// CHECK:STDOUT:   %require_complete.1 => constants.%complete_type.8d5
// CHECK:STDOUT:   %ptr => constants.%ptr.c5c
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type => constants.%Iterate.WithSelf.Next.type.e25
// CHECK:STDOUT:   %.loc34_19.20 => constants.%.03b
// CHECK:STDOUT:   %impl.elem3.loc34_19.2 => constants.%R.as_type.as.Iterate.impl.Next.393
// CHECK:STDOUT:   %specific_impl_fn.loc34_19.5 => constants.%R.as_type.as.Iterate.impl.Next.specific_fn.4cc
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.7ec
// CHECK:STDOUT:   %.loc34_19.21 => constants.%.0e7
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.df9cc1.11
// CHECK:STDOUT:   %impl.elem0.loc34_19.2 => constants.%Destroy.Op.1a2547.11
// CHECK:STDOUT:   %specific_impl_fn.loc34_19.6 => constants.%Destroy.Op.1a2547.11
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
// CHECK:STDOUT:   %T.220: %OptionalStorage.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Optional.Get.type.bb9: type = fn_type @Optional.Get, @Optional(%T.220) [symbolic]
// CHECK:STDOUT:   %Optional.Get.fa1: %Optional.Get.type.bb9 = struct_value () [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.type.e9e: type = fn_type @Optional.HasValue, @Optional(%T.220) [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.6cd: %Optional.HasValue.type.e9e = struct_value () [symbolic]
// CHECK:STDOUT:   %T.283: %facet_type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.d50: type = class_type @DefaultOptionalStorage, @DefaultOptionalStorage(%T.283) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Get.type.029: type = fn_type @T.as_type.as.OptionalStorage.impl.Get, @T.as_type.as.OptionalStorage.impl(%T.283) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Get.cb5: %T.as_type.as.OptionalStorage.impl.Get.type.029 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Has.type.4bc: type = fn_type @T.as_type.as.OptionalStorage.impl.Has, @T.as_type.as.OptionalStorage.impl(%T.283) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Has.dc2: %T.as_type.as.OptionalStorage.impl.Has.type.4bc = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Some.type.800: type = fn_type @T.as_type.as.OptionalStorage.impl.Some, @T.as_type.as.OptionalStorage.impl(%T.283) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Some.e83: %T.as_type.as.OptionalStorage.impl.Some.type.800 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.None.type.300: type = fn_type @T.as_type.as.OptionalStorage.impl.None, @T.as_type.as.OptionalStorage.impl(%T.283) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.None.f54: %T.as_type.as.OptionalStorage.impl.None.type.300 = struct_value () [symbolic]
// CHECK:STDOUT:   %.Self.35d: %Iterate.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %Iterate.lookup_impl_witness.b37: <witness> = lookup_impl_witness %.Self.35d, @Iterate [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.903: %facet_type = impl_witness_access %Iterate.lookup_impl_witness.b37, element0 [symbolic_self]
// CHECK:STDOUT:   %ValueType: type = class_type @ValueType [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %pattern_type.e9f: type = pattern_type %ValueType [concrete]
// CHECK:STDOUT:   %ValueType.cpp_destructor.type: type = fn_type @ValueType.cpp_destructor [concrete]
// CHECK:STDOUT:   %ValueType.cpp_destructor: %ValueType.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT:   %ValueType.Op.type.83af6e.1: type = fn_type @ValueType.Op.1 [concrete]
// CHECK:STDOUT:   %ValueType.Op.4dca31.1: %ValueType.Op.type.83af6e.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.e7a31c.1: <witness> = custom_witness (%ValueType.Op.4dca31.1), @Destroy [concrete]
// CHECK:STDOUT:   %const.243: type = const_type %ValueType [concrete]
// CHECK:STDOUT:   %ptr.6bf: type = ptr_type %const.243 [concrete]
// CHECK:STDOUT:   %ptr.a3d: type = ptr_type %ValueType [concrete]
// CHECK:STDOUT:   %ValueType.Op.type.83af6e.2: type = fn_type @ValueType.Op.2 [concrete]
// CHECK:STDOUT:   %ValueType.Op.4dca31.2: %ValueType.Op.type.83af6e.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.d0da43.1: <witness> = custom_witness (%ValueType.Op.4dca31.2), @Copy [concrete]
// CHECK:STDOUT:   %facet_value.bf1: %facet_type = facet_value %ValueType, (%custom_witness.e7a31c.1, %custom_witness.d0da43.1) [concrete]
// CHECK:STDOUT:   %Iterate_where.type.dfc: type = facet_type <@Iterate where %impl.elem0.903 = %facet_value.bf1> [concrete]
// CHECK:STDOUT:   %pattern_type.161: type = pattern_type %Iterate_where.type.dfc [concrete]
// CHECK:STDOUT:   %R.patt.f38: %pattern_type.161 = symbolic_binding_pattern R, 0 [symbolic]
// CHECK:STDOUT:   %R.de1: %Iterate_where.type.dfc = symbolic_binding R, 0 [symbolic]
// CHECK:STDOUT:   %R.as_type.f62: type = facet_access_type %R.de1 [symbolic]
// CHECK:STDOUT:   %pattern_type.669: type = pattern_type %R.as_type.f62 [symbolic]
// CHECK:STDOUT:   %r.patt.c0f: %pattern_type.669 = ref_binding_pattern r [symbolic]
// CHECK:STDOUT:   %r.param_patt.bb2: %pattern_type.669 = var_param_pattern %r.patt.c0f [symbolic]
// CHECK:STDOUT:   %i.patt: %pattern_type.e9f = value_binding_pattern i [concrete]
// CHECK:STDOUT:   %CppRangeForIterate.type: type = facet_type <@CppRangeForIterate> [concrete]
// CHECK:STDOUT:   %.Self.6bd: %CppRangeForIterate.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %CppRangeForIterate.lookup_impl_witness.143: <witness> = lookup_impl_witness %.Self.6bd, @CppRangeForIterate [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.109: type = impl_witness_access %CppRangeForIterate.lookup_impl_witness.143, element0 [symbolic_self]
// CHECK:STDOUT:   %CppUnsafeDeref.lookup_impl_witness.712: <witness> = lookup_impl_witness %impl.elem0.109, @CppUnsafeDeref [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.56a: type = impl_witness_access %CppUnsafeDeref.lookup_impl_witness.712, element0 [symbolic_self]
// CHECK:STDOUT:   %impl.elem1.222: type = impl_witness_access %CppRangeForIterate.lookup_impl_witness.143, element1 [symbolic_self]
// CHECK:STDOUT:   %CppRangeForIterate_where.type: type = facet_type <@CppRangeForIterate where %impl.elem0.109 impls @Destroy and %impl.elem0.109 impls @Copy and %impl.elem0.109 impls @CppUnsafeDeref and %impl.elem0.109 impls @EqWith, @EqWith(%impl.elem1.222) and %impl.elem0.109 impls @Inc and %impl.elem0.56a impls @Destroy and %impl.elem0.56a impls @Copy and %impl.elem1.222 impls @Destroy and %impl.elem1.222 impls @Copy> [concrete]
// CHECK:STDOUT:   %R.39b: %CppRangeForIterate_where.type = symbolic_binding R, 0 [symbolic]
// CHECK:STDOUT:   %CppRangeForIterate.lookup_impl_witness.7e2712.2: <witness> = lookup_impl_witness %R.39b, @CppRangeForIterate [symbolic]
// CHECK:STDOUT:   %impl.elem1.7a0bff.2: type = impl_witness_access %CppRangeForIterate.lookup_impl_witness.7e2712.2, element1 [symbolic]
// CHECK:STDOUT:   %impl.elem0.73eb7f.2: type = impl_witness_access %CppRangeForIterate.lookup_impl_witness.7e2712.2, element0 [symbolic]
// CHECK:STDOUT:   %tuple.type.3cf: type = tuple_type (%impl.elem0.73eb7f.2, %impl.elem1.7a0bff.2) [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.69f: <witness> = lookup_impl_witness %tuple.type.3cf, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.593: %Destroy.type = facet_value %tuple.type.3cf, (%Destroy.lookup_impl_witness.69f) [symbolic]
// CHECK:STDOUT:   %CppUnsafeDeref.lookup_impl_witness.aa7: <witness> = lookup_impl_witness %impl.elem0.73eb7f.2, @CppUnsafeDeref [symbolic]
// CHECK:STDOUT:   %impl.elem0.af7: type = impl_witness_access %CppUnsafeDeref.lookup_impl_witness.aa7, element0 [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.ec3: <witness> = lookup_impl_witness %impl.elem0.af7, @Destroy [symbolic]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.a7a: <witness> = lookup_impl_witness %impl.elem0.af7, @Copy [symbolic]
// CHECK:STDOUT:   %facet_value.194: %facet_type = facet_value %impl.elem0.af7, (%Destroy.lookup_impl_witness.ec3, %Copy.lookup_impl_witness.a7a) [symbolic]
// CHECK:STDOUT:   %R.as_type.as.Iterate.impl.Next.type.1f7: type = fn_type @R.as_type.as.Iterate.impl.Next, @R.as_type.as.Iterate.impl(%R.39b) [symbolic]
// CHECK:STDOUT:   %R.as_type.as.Iterate.impl.Next.0d6: %R.as_type.as.Iterate.impl.Next.type.1f7 = struct_value () [symbolic]
// CHECK:STDOUT:   %R.as_type.as.Iterate.impl.NewCursor.type.36f: type = fn_type @R.as_type.as.Iterate.impl.NewCursor, @R.as_type.as.Iterate.impl(%R.39b) [symbolic]
// CHECK:STDOUT:   %R.as_type.as.Iterate.impl.NewCursor.29b: %R.as_type.as.Iterate.impl.NewCursor.type.36f = struct_value () [symbolic]
// CHECK:STDOUT:   %Iterate.lookup_impl_witness.acc: <witness> = lookup_impl_witness %R.de1, @Iterate [symbolic]
// CHECK:STDOUT:   %Iterate.facet.b95: %Iterate.type = facet_value %R.as_type.f62, (%Iterate.lookup_impl_witness.acc) [symbolic]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.d69: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.b95) [symbolic]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.d6f: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.b95) [symbolic]
// CHECK:STDOUT:   %.f8a: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.d69, %Iterate.facet.b95 [symbolic]
// CHECK:STDOUT:   %impl.elem2.398: %.f8a = impl_witness_access %Iterate.lookup_impl_witness.acc, element2 [symbolic]
// CHECK:STDOUT:   %impl.elem1.411: %Destroy.type = impl_witness_access %Iterate.lookup_impl_witness.acc, element1 [symbolic]
// CHECK:STDOUT:   %as_type.bc6: type = facet_access_type %impl.elem1.411 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.544: <specific function> = specific_impl_function %impl.elem2.398, @Iterate.WithSelf.NewCursor(%Iterate.facet.b95) [symbolic]
// CHECK:STDOUT:   %ptr.bdd: type = ptr_type %as_type.bc6 [symbolic]
// CHECK:STDOUT:   %.52a: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.d6f, %Iterate.facet.b95 [symbolic]
// CHECK:STDOUT:   %impl.elem3.510: %.52a = impl_witness_access %Iterate.lookup_impl_witness.acc, element3 [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.84c: type = class_type @DefaultOptionalStorage, @DefaultOptionalStorage(%facet_value.bf1) [concrete]
// CHECK:STDOUT:   %OptionalStorage.impl_witness.af1: <witness> = impl_witness imports.%OptionalStorage.impl_witness_table.655, @T.as_type.as.OptionalStorage.impl(%facet_value.bf1) [concrete]
// CHECK:STDOUT:   %OptionalStorage.facet.705: %OptionalStorage.type = facet_value %ValueType, (%OptionalStorage.impl_witness.af1) [concrete]
// CHECK:STDOUT:   %Optional.95f: type = class_type @Optional, @Optional(%OptionalStorage.facet.705) [concrete]
// CHECK:STDOUT:   %specific_impl_fn.a00: <specific function> = specific_impl_function %impl.elem3.510, @Iterate.WithSelf.Next(%Iterate.facet.b95) [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.type.006: type = fn_type @Optional.HasValue, @Optional(%OptionalStorage.facet.705) [concrete]
// CHECK:STDOUT:   %Optional.HasValue.988: %Optional.HasValue.type.006 = struct_value () [concrete]
// CHECK:STDOUT:   %Optional.Get.type.30f: type = fn_type @Optional.Get, @Optional(%OptionalStorage.facet.705) [concrete]
// CHECK:STDOUT:   %Optional.Get.59c: %Optional.Get.type.30f = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.facet.4e5: %Destroy.type = facet_value %ValueType, (%custom_witness.e7a31c.1) [concrete]
// CHECK:STDOUT:   %MaybeUnformed.02e: type = class_type @MaybeUnformed, @MaybeUnformed(%Destroy.facet.4e5) [concrete]
// CHECK:STDOUT:   %.98c: type = maybe_unformed_type %ValueType [concrete]
// CHECK:STDOUT:   %struct_type.value.has_value.761: type = struct_type {.value: %MaybeUnformed.02e, .has_value: bool} [concrete]
// CHECK:STDOUT:   %Optional.HasValue.specific_fn: <specific function> = specific_function %Optional.HasValue.988, @Optional.HasValue(%OptionalStorage.facet.705) [concrete]
// CHECK:STDOUT:   %Optional.Get.specific_fn: <specific function> = specific_function %Optional.Get.59c, @Optional.Get(%OptionalStorage.facet.705) [concrete]
// CHECK:STDOUT:   %operator_PlusEqual__carbon_thunk.type: type = fn_type @operator_PlusEqual__carbon_thunk [concrete]
// CHECK:STDOUT:   %operator_PlusEqual__carbon_thunk: %operator_PlusEqual__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.7: type = fn_type @Destroy.Op.loc44_29.6 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.7: %Destroy.Op.type.1d8f74.7 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.ff8: <witness> = lookup_impl_witness %impl.elem1.411, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.d67: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%impl.elem1.411) [symbolic]
// CHECK:STDOUT:   %.4b9: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.d67, %impl.elem1.411 [symbolic]
// CHECK:STDOUT:   %impl.elem0.fbe: %.4b9 = impl_witness_access %Destroy.lookup_impl_witness.ff8, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.517: <specific function> = specific_impl_function %impl.elem0.fbe, @Destroy.WithSelf.Op(%impl.elem1.411) [symbolic]
// CHECK:STDOUT:   %MutableRange: type = class_type @MutableRange [concrete]
// CHECK:STDOUT:   %pattern_type.e635: type = pattern_type %MutableRange [concrete]
// CHECK:STDOUT:   %ConstRange: type = class_type @ConstRange [concrete]
// CHECK:STDOUT:   %pattern_type.0ba: type = pattern_type %ConstRange [concrete]
// CHECK:STDOUT:   %Iterator.3ee: type = class_type @Iterator.1 [concrete]
// CHECK:STDOUT:   %Sentinel.9b2: type = class_type @Sentinel.1 [concrete]
// CHECK:STDOUT:   %MutableRange.Begin.type: type = fn_type @MutableRange.Begin [concrete]
// CHECK:STDOUT:   %MutableRange.Begin: %MutableRange.Begin.type = struct_value () [concrete]
// CHECK:STDOUT:   %MutableRange.End.type: type = fn_type @MutableRange.End [concrete]
// CHECK:STDOUT:   %MutableRange.End: %MutableRange.End.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.6b6: <witness> = custom_witness (%Iterator.3ee, %Sentinel.9b2, %MutableRange.Begin, %MutableRange.End), @CppRangeForIterate [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.0aed5d.1: type = fn_type @Iterator.Op.1 [concrete]
// CHECK:STDOUT:   %Iterator.Op.a2a866.1: %Iterator.Op.type.0aed5d.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.90a: <witness> = custom_witness (%ValueType, %Iterator.Op.a2a866.1), @CppUnsafeDeref [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.0aed5d.2: type = fn_type @Iterator.Op.2 [concrete]
// CHECK:STDOUT:   %Iterator.Op.a2a866.2: %Iterator.Op.type.0aed5d.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.95581e.1: <witness> = custom_witness (%Iterator.Op.a2a866.2), @Destroy [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.0aed5d.3: type = fn_type @Iterator.Op.3 [concrete]
// CHECK:STDOUT:   %Iterator.Op.a2a866.3: %Iterator.Op.type.0aed5d.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.1f52e9.1: <witness> = custom_witness (%Iterator.Op.a2a866.3), @Copy [concrete]
// CHECK:STDOUT:   %Equal.type.ca8475.1: type = fn_type @Equal.1 [concrete]
// CHECK:STDOUT:   %Equal.285b15.1: %Equal.type.ca8475.1 = struct_value () [concrete]
// CHECK:STDOUT:   %NotEqual.type.53d9ee.1: type = fn_type @NotEqual.1 [concrete]
// CHECK:STDOUT:   %NotEqual.119263.1: %NotEqual.type.53d9ee.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.3e8: <witness> = custom_witness (%Equal.285b15.1, %NotEqual.119263.1), @EqWith, @EqWith(%Sentinel.9b2) [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.0aed5d.4: type = fn_type @Iterator.Op.4 [concrete]
// CHECK:STDOUT:   %Iterator.Op.a2a866.4: %Iterator.Op.type.0aed5d.4 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.444: <witness> = custom_witness (%Iterator.Op.a2a866.4), @Inc [concrete]
// CHECK:STDOUT:   %Sentinel.Op.type.0aed5d.1: type = fn_type @Sentinel.Op.1 [concrete]
// CHECK:STDOUT:   %Sentinel.Op.a2a866.1: %Sentinel.Op.type.0aed5d.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.95581e.2: <witness> = custom_witness (%Sentinel.Op.a2a866.1), @Destroy [concrete]
// CHECK:STDOUT:   %Sentinel.Op.type.0aed5d.2: type = fn_type @Sentinel.Op.2 [concrete]
// CHECK:STDOUT:   %Sentinel.Op.a2a866.2: %Sentinel.Op.type.0aed5d.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.1f52e9.2: <witness> = custom_witness (%Sentinel.Op.a2a866.2), @Copy [concrete]
// CHECK:STDOUT:   %facet_value.55a: %CppRangeForIterate_where.type = facet_value %MutableRange, (%custom_witness.e7a31c.1, %custom_witness.d0da43.1, %custom_witness.6b6, %custom_witness.95581e.1, %custom_witness.1f52e9.1, %custom_witness.90a, %custom_witness.3e8, %custom_witness.444, %custom_witness.95581e.2, %custom_witness.1f52e9.2) [concrete]
// CHECK:STDOUT:   %tuple.type.829: type = tuple_type (%Iterator.3ee, %Sentinel.9b2) [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.8: type = fn_type @Destroy.Op.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.8: %Destroy.Op.type.1d8f74.8 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.df9cc1.8: <witness> = custom_witness (%Destroy.Op.1a2547.8), @Destroy [concrete]
// CHECK:STDOUT:   %Destroy.facet.305: %Destroy.type = facet_value %tuple.type.829, (%custom_witness.df9cc1.8) [concrete]
// CHECK:STDOUT:   %Iterate.impl_witness.096: <witness> = impl_witness imports.%Iterate.impl_witness_table.344, @R.as_type.as.Iterate.impl(%facet_value.55a) [concrete]
// CHECK:STDOUT:   %R.as_type.as.Iterate.impl.NewCursor.type.c00: type = fn_type @R.as_type.as.Iterate.impl.NewCursor, @R.as_type.as.Iterate.impl(%facet_value.55a) [concrete]
// CHECK:STDOUT:   %R.as_type.as.Iterate.impl.NewCursor.498: %R.as_type.as.Iterate.impl.NewCursor.type.c00 = struct_value () [concrete]
// CHECK:STDOUT:   %R.as_type.as.Iterate.impl.Next.type.d64: type = fn_type @R.as_type.as.Iterate.impl.Next, @R.as_type.as.Iterate.impl(%facet_value.55a) [concrete]
// CHECK:STDOUT:   %R.as_type.as.Iterate.impl.Next.055: %R.as_type.as.Iterate.impl.Next.type.d64 = struct_value () [concrete]
// CHECK:STDOUT:   %Iterate.facet.723: %Iterate.type = facet_value %MutableRange, (%Iterate.impl_witness.096) [concrete]
// CHECK:STDOUT:   %facet_value.150: %Iterate_where.type.dfc = facet_value %MutableRange, (%Iterate.impl_witness.096) [concrete]
// CHECK:STDOUT:   %r.patt.f69: %pattern_type.e635 = ref_binding_pattern r [concrete]
// CHECK:STDOUT:   %r.param_patt.91e: %pattern_type.e635 = var_param_pattern %r.patt.f69 [concrete]
// CHECK:STDOUT:   %Iterator.260: type = class_type @Iterator.2 [concrete]
// CHECK:STDOUT:   %Sentinel.4c4: type = class_type @Sentinel.2 [concrete]
// CHECK:STDOUT:   %ConstRange.Begin.type: type = fn_type @ConstRange.Begin [concrete]
// CHECK:STDOUT:   %ConstRange.Begin: %ConstRange.Begin.type = struct_value () [concrete]
// CHECK:STDOUT:   %ConstRange.End.type: type = fn_type @ConstRange.End [concrete]
// CHECK:STDOUT:   %ConstRange.End: %ConstRange.End.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.a03: <witness> = custom_witness (%Iterator.260, %Sentinel.4c4, %ConstRange.Begin, %ConstRange.End), @CppRangeForIterate [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.0f34b8.1: type = fn_type @Iterator.Op.5 [concrete]
// CHECK:STDOUT:   %Iterator.Op.1c3e4a.1: %Iterator.Op.type.0f34b8.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.78b: <witness> = custom_witness (%ValueType, %Iterator.Op.1c3e4a.1), @CppUnsafeDeref [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.0f34b8.2: type = fn_type @Iterator.Op.6 [concrete]
// CHECK:STDOUT:   %Iterator.Op.1c3e4a.2: %Iterator.Op.type.0f34b8.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.6e8a2e.1: <witness> = custom_witness (%Iterator.Op.1c3e4a.2), @Destroy [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.0f34b8.3: type = fn_type @Iterator.Op.7 [concrete]
// CHECK:STDOUT:   %Iterator.Op.1c3e4a.3: %Iterator.Op.type.0f34b8.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.15091a.1: <witness> = custom_witness (%Iterator.Op.1c3e4a.3), @Copy [concrete]
// CHECK:STDOUT:   %Equal.type.ca8475.2: type = fn_type @Equal.2 [concrete]
// CHECK:STDOUT:   %Equal.285b15.2: %Equal.type.ca8475.2 = struct_value () [concrete]
// CHECK:STDOUT:   %NotEqual.type.53d9ee.2: type = fn_type @NotEqual.2 [concrete]
// CHECK:STDOUT:   %NotEqual.119263.2: %NotEqual.type.53d9ee.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.c55: <witness> = custom_witness (%Equal.285b15.2, %NotEqual.119263.2), @EqWith, @EqWith(%Sentinel.4c4) [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.0f34b8.4: type = fn_type @Iterator.Op.8 [concrete]
// CHECK:STDOUT:   %Iterator.Op.1c3e4a.4: %Iterator.Op.type.0f34b8.4 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.746: <witness> = custom_witness (%Iterator.Op.1c3e4a.4), @Inc [concrete]
// CHECK:STDOUT:   %Sentinel.Op.type.0f34b8.1: type = fn_type @Sentinel.Op.3 [concrete]
// CHECK:STDOUT:   %Sentinel.Op.1c3e4a.1: %Sentinel.Op.type.0f34b8.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.6e8a2e.2: <witness> = custom_witness (%Sentinel.Op.1c3e4a.1), @Destroy [concrete]
// CHECK:STDOUT:   %Sentinel.Op.type.0f34b8.2: type = fn_type @Sentinel.Op.4 [concrete]
// CHECK:STDOUT:   %Sentinel.Op.1c3e4a.2: %Sentinel.Op.type.0f34b8.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.15091a.2: <witness> = custom_witness (%Sentinel.Op.1c3e4a.2), @Copy [concrete]
// CHECK:STDOUT:   %facet_value.445: %CppRangeForIterate_where.type = facet_value %ConstRange, (%custom_witness.e7a31c.1, %custom_witness.d0da43.1, %custom_witness.a03, %custom_witness.6e8a2e.1, %custom_witness.15091a.1, %custom_witness.78b, %custom_witness.c55, %custom_witness.746, %custom_witness.6e8a2e.2, %custom_witness.15091a.2) [concrete]
// CHECK:STDOUT:   %tuple.type.99f: type = tuple_type (%Iterator.260, %Sentinel.4c4) [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.9: type = fn_type @Destroy.Op.3 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.9: %Destroy.Op.type.1d8f74.9 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.df9cc1.9: <witness> = custom_witness (%Destroy.Op.1a2547.9), @Destroy [concrete]
// CHECK:STDOUT:   %Destroy.facet.e08: %Destroy.type = facet_value %tuple.type.99f, (%custom_witness.df9cc1.9) [concrete]
// CHECK:STDOUT:   %Iterate.impl_witness.40c: <witness> = impl_witness imports.%Iterate.impl_witness_table.344, @R.as_type.as.Iterate.impl(%facet_value.445) [concrete]
// CHECK:STDOUT:   %R.as_type.as.Iterate.impl.NewCursor.type.8c3: type = fn_type @R.as_type.as.Iterate.impl.NewCursor, @R.as_type.as.Iterate.impl(%facet_value.445) [concrete]
// CHECK:STDOUT:   %R.as_type.as.Iterate.impl.NewCursor.bd2: %R.as_type.as.Iterate.impl.NewCursor.type.8c3 = struct_value () [concrete]
// CHECK:STDOUT:   %R.as_type.as.Iterate.impl.Next.type.b0c: type = fn_type @R.as_type.as.Iterate.impl.Next, @R.as_type.as.Iterate.impl(%facet_value.445) [concrete]
// CHECK:STDOUT:   %R.as_type.as.Iterate.impl.Next.c50: %R.as_type.as.Iterate.impl.Next.type.b0c = struct_value () [concrete]
// CHECK:STDOUT:   %Iterate.facet.c46: %Iterate.type = facet_value %ConstRange, (%Iterate.impl_witness.40c) [concrete]
// CHECK:STDOUT:   %facet_value.e78: %Iterate_where.type.dfc = facet_value %ConstRange, (%Iterate.impl_witness.40c) [concrete]
// CHECK:STDOUT:   %r.patt.36a: %pattern_type.0ba = ref_binding_pattern r [concrete]
// CHECK:STDOUT:   %r.param_patt.501: %pattern_type.0ba = var_param_pattern %r.patt.36a [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.47d: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.723) [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.6f1: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.723) [concrete]
// CHECK:STDOUT:   %.47d: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.47d, %Iterate.facet.723 [concrete]
// CHECK:STDOUT:   %R.as_type.as.Iterate.impl.NewCursor.specific_fn.532: <specific function> = specific_function %R.as_type.as.Iterate.impl.NewCursor.498, @R.as_type.as.Iterate.impl.NewCursor(%facet_value.55a) [concrete]
// CHECK:STDOUT:   %ptr.d6b: type = ptr_type %tuple.type.829 [concrete]
// CHECK:STDOUT:   %.fd2: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.6f1, %Iterate.facet.723 [concrete]
// CHECK:STDOUT:   %R.as_type.as.Iterate.impl.Next.specific_fn.5cb: <specific function> = specific_function %R.as_type.as.Iterate.impl.Next.055, @R.as_type.as.Iterate.impl.Next(%facet_value.55a) [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.cd5: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.c46) [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.285: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.c46) [concrete]
// CHECK:STDOUT:   %.763: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.cd5, %Iterate.facet.c46 [concrete]
// CHECK:STDOUT:   %R.as_type.as.Iterate.impl.NewCursor.specific_fn.7c7: <specific function> = specific_function %R.as_type.as.Iterate.impl.NewCursor.bd2, @R.as_type.as.Iterate.impl.NewCursor(%facet_value.445) [concrete]
// CHECK:STDOUT:   %ptr.2f7: type = ptr_type %tuple.type.99f [concrete]
// CHECK:STDOUT:   %.bca: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.285, %Iterate.facet.c46 [concrete]
// CHECK:STDOUT:   %R.as_type.as.Iterate.impl.Next.specific_fn.a9a: <specific function> = specific_function %R.as_type.as.Iterate.impl.Next.c50, @R.as_type.as.Iterate.impl.Next(%facet_value.445) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.e30: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.e08) [concrete]
// CHECK:STDOUT:   %.987: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.e30, %Destroy.facet.e08 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.c89: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.305) [concrete]
// CHECK:STDOUT:   %.e8a: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.c89, %Destroy.facet.305 [concrete]
// CHECK:STDOUT:   %complete_type.94f: <witness> = complete_type_witness %tuple.type.829 [concrete]
// CHECK:STDOUT:   %complete_type.dea: <witness> = complete_type_witness %tuple.type.99f [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .ValueType = %ValueType.decl
// CHECK:STDOUT:     .MutableRange = %MutableRange.decl
// CHECK:STDOUT:     .ConstRange = %ConstRange.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.bb6: @Optional.%Optional.HasValue.type (%Optional.HasValue.type.e9e) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.HasValue (constants.%Optional.HasValue.6cd)]
// CHECK:STDOUT:   %Core.import_ref.daba: @Optional.%Optional.Get.type (%Optional.Get.type.bb9) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.Get (constants.%Optional.Get.fa1)]
// CHECK:STDOUT:   %Core.import_ref.0c6: type = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%DefaultOptionalStorage (constants.%DefaultOptionalStorage.d50)]
// CHECK:STDOUT:   %Core.import_ref.f4f: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.None.type (%T.as_type.as.OptionalStorage.impl.None.type.300) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.None (constants.%T.as_type.as.OptionalStorage.impl.None.f54)]
// CHECK:STDOUT:   %Core.import_ref.18f: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Some.type (%T.as_type.as.OptionalStorage.impl.Some.type.800) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Some (constants.%T.as_type.as.OptionalStorage.impl.Some.e83)]
// CHECK:STDOUT:   %Core.import_ref.c68: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Has.type (%T.as_type.as.OptionalStorage.impl.Has.type.4bc) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Has (constants.%T.as_type.as.OptionalStorage.impl.Has.dc2)]
// CHECK:STDOUT:   %Core.import_ref.a83: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Get.type (%T.as_type.as.OptionalStorage.impl.Get.type.029) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Get (constants.%T.as_type.as.OptionalStorage.impl.Get.cb5)]
// CHECK:STDOUT:   %Core.import_ref.aaa = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], unloaded
// CHECK:STDOUT:   %OptionalStorage.impl_witness_table.655 = impl_witness_table (%Core.import_ref.0c6, %Core.import_ref.f4f, %Core.import_ref.18f, %Core.import_ref.c68, %Core.import_ref.a83, %Core.import_ref.aaa), @T.as_type.as.OptionalStorage.impl [concrete]
// CHECK:STDOUT:   %ValueType.decl: type = class_decl @ValueType [concrete = constants.%ValueType] {} {}
// CHECK:STDOUT:   %ValueType.cpp_destructor.decl: %ValueType.cpp_destructor.type = fn_decl @ValueType.cpp_destructor [concrete = constants.%ValueType.cpp_destructor] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.301: %facet_type = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @R.as_type.as.Iterate.impl.%facet_value (constants.%facet_value.194)]
// CHECK:STDOUT:   %Core.import_ref.89f: %Destroy.type = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @R.as_type.as.Iterate.impl.%Destroy.facet (constants.%Destroy.facet.593)]
// CHECK:STDOUT:   %Core.import_ref.3e6: @R.as_type.as.Iterate.impl.%R.as_type.as.Iterate.impl.NewCursor.type (%R.as_type.as.Iterate.impl.NewCursor.type.36f) = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @R.as_type.as.Iterate.impl.%R.as_type.as.Iterate.impl.NewCursor (constants.%R.as_type.as.Iterate.impl.NewCursor.29b)]
// CHECK:STDOUT:   %Core.import_ref.85d: @R.as_type.as.Iterate.impl.%R.as_type.as.Iterate.impl.Next.type (%R.as_type.as.Iterate.impl.Next.type.1f7) = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @R.as_type.as.Iterate.impl.%R.as_type.as.Iterate.impl.Next (constants.%R.as_type.as.Iterate.impl.Next.0d6)]
// CHECK:STDOUT:   %Iterate.impl_witness_table.344 = impl_witness_table (%Core.import_ref.301, %Core.import_ref.89f, %Core.import_ref.3e6, %Core.import_ref.85d), @R.as_type.as.Iterate.impl [concrete]
// CHECK:STDOUT:   %operator_PlusEqual__carbon_thunk.decl: %operator_PlusEqual__carbon_thunk.type = fn_decl @operator_PlusEqual__carbon_thunk [concrete = constants.%operator_PlusEqual__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %MutableRange.decl: type = class_decl @MutableRange [concrete = constants.%MutableRange] {} {}
// CHECK:STDOUT:   %ConstRange.decl: type = class_decl @ConstRange [concrete = constants.%ConstRange] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @TestIterate(%R.loc40_17.2: %Iterate_where.type.dfc) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Iterate.lookup_impl_witness: <witness> = lookup_impl_witness %R.loc40_17.1, @Iterate [symbolic = %Iterate.lookup_impl_witness (constants.%Iterate.lookup_impl_witness.acc)]
// CHECK:STDOUT:   %Iterate.facet.loc44_29.5: %Iterate.type = facet_value %R.as_type.loc40_76.1, (%Iterate.lookup_impl_witness) [symbolic = %Iterate.facet.loc44_29.5 (constants.%Iterate.facet.b95)]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.loc44_29.5) [symbolic = %Iterate.WithSelf.NewCursor.type (constants.%Iterate.WithSelf.NewCursor.type.d69)]
// CHECK:STDOUT:   %.loc44_29.20: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type, %Iterate.facet.loc44_29.5 [symbolic = %.loc44_29.20 (constants.%.f8a)]
// CHECK:STDOUT:   %impl.elem2.loc44_29.2: @TestIterate.%.loc44_29.20 (%.f8a) = impl_witness_access %Iterate.lookup_impl_witness, element2 [symbolic = %impl.elem2.loc44_29.2 (constants.%impl.elem2.398)]
// CHECK:STDOUT:   %specific_impl_fn.loc44_29.4: <specific function> = specific_impl_function %impl.elem2.loc44_29.2, @Iterate.WithSelf.NewCursor(%Iterate.facet.loc44_29.5) [symbolic = %specific_impl_fn.loc44_29.4 (constants.%specific_impl_fn.544)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %ptr: type = ptr_type %as_type [symbolic = %ptr (constants.%ptr.bdd)]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.loc44_29.5) [symbolic = %Iterate.WithSelf.Next.type (constants.%Iterate.WithSelf.Next.type.d6f)]
// CHECK:STDOUT:   %.loc44_29.21: type = fn_type_with_self_type %Iterate.WithSelf.Next.type, %Iterate.facet.loc44_29.5 [symbolic = %.loc44_29.21 (constants.%.52a)]
// CHECK:STDOUT:   %impl.elem3.loc44_29.2: @TestIterate.%.loc44_29.21 (%.52a) = impl_witness_access %Iterate.lookup_impl_witness, element3 [symbolic = %impl.elem3.loc44_29.2 (constants.%impl.elem3.510)]
// CHECK:STDOUT:   %specific_impl_fn.loc44_29.5: <specific function> = specific_impl_function %impl.elem3.loc44_29.2, @Iterate.WithSelf.Next(%Iterate.facet.loc44_29.5) [symbolic = %specific_impl_fn.loc44_29.5 (constants.%specific_impl_fn.a00)]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%impl.elem1) [symbolic = %Destroy.WithSelf.Op.type (constants.%Destroy.WithSelf.Op.type.d67)]
// CHECK:STDOUT:   %.loc44_29.22: type = fn_type_with_self_type %Destroy.WithSelf.Op.type, %impl.elem1 [symbolic = %.loc44_29.22 (constants.%.4b9)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %impl.elem1, @Destroy [symbolic = %Destroy.lookup_impl_witness (constants.%Destroy.lookup_impl_witness.ff8)]
// CHECK:STDOUT:   %impl.elem0.loc44_29.2: @TestIterate.%.loc44_29.22 (%.4b9) = impl_witness_access %Destroy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc44_29.2 (constants.%impl.elem0.fbe)]
// CHECK:STDOUT:   %specific_impl_fn.loc44_29.6: <specific function> = specific_impl_function %impl.elem0.loc44_29.2, @Destroy.WithSelf.Op(%impl.elem1) [symbolic = %specific_impl_fn.loc44_29.6 (constants.%specific_impl_fn.517)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%r.param: ref @TestIterate.%R.as_type.loc40_76.1 (%R.as_type.f62)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %i.patt: %pattern_type.e9f = value_binding_pattern i [concrete = constants.%i.patt]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %r.ref: ref @TestIterate.%R.as_type.loc40_76.1 (%R.as_type.f62) = name_ref r, %r
// CHECK:STDOUT:     %impl.elem2.loc44_29.1: @TestIterate.%.loc44_29.20 (%.f8a) = impl_witness_access constants.%Iterate.lookup_impl_witness.acc, element2 [symbolic = %impl.elem2.loc44_29.2 (constants.%impl.elem2.398)]
// CHECK:STDOUT:     %bound_method.loc44_29.1: <bound method> = bound_method %r.ref, %impl.elem2.loc44_29.1
// CHECK:STDOUT:     %Iterate.facet.loc44_29.1: %Iterate.type = facet_value constants.%R.as_type.f62, (constants.%Iterate.lookup_impl_witness.acc) [symbolic = %Iterate.facet.loc44_29.5 (constants.%Iterate.facet.b95)]
// CHECK:STDOUT:     %.loc44_29.1: %Iterate.type = converted constants.%R.as_type.f62, %Iterate.facet.loc44_29.1 [symbolic = %Iterate.facet.loc44_29.5 (constants.%Iterate.facet.b95)]
// CHECK:STDOUT:     %Iterate.facet.loc44_29.2: %Iterate.type = facet_value constants.%R.as_type.f62, (constants.%Iterate.lookup_impl_witness.acc) [symbolic = %Iterate.facet.loc44_29.5 (constants.%Iterate.facet.b95)]
// CHECK:STDOUT:     %.loc44_29.2: %Iterate.type = converted constants.%R.as_type.f62, %Iterate.facet.loc44_29.2 [symbolic = %Iterate.facet.loc44_29.5 (constants.%Iterate.facet.b95)]
// CHECK:STDOUT:     %specific_impl_fn.loc44_29.1: <specific function> = specific_impl_function %impl.elem2.loc44_29.1, @Iterate.WithSelf.NewCursor(constants.%Iterate.facet.b95) [symbolic = %specific_impl_fn.loc44_29.4 (constants.%specific_impl_fn.544)]
// CHECK:STDOUT:     %bound_method.loc44_29.2: <bound method> = bound_method %r.ref, %specific_impl_fn.loc44_29.1
// CHECK:STDOUT:     %var: ref @TestIterate.%as_type (%as_type.bc6) = var invalid
// CHECK:STDOUT:     %.loc44_28.1: @TestIterate.%R.as_type.loc40_76.1 (%R.as_type.f62) = acquire_value %r.ref
// CHECK:STDOUT:     %Iterate.WithSelf.NewCursor.call: init @TestIterate.%as_type (%as_type.bc6) to %var = call %bound_method.loc44_29.2(%.loc44_28.1)
// CHECK:STDOUT:     assign %var, %Iterate.WithSelf.NewCursor.call
// CHECK:STDOUT:     br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.next:
// CHECK:STDOUT:     %addr.loc44: @TestIterate.%ptr (%ptr.bdd) = addr_of %var
// CHECK:STDOUT:     %impl.elem3.loc44_29.1: @TestIterate.%.loc44_29.21 (%.52a) = impl_witness_access constants.%Iterate.lookup_impl_witness.acc, element3 [symbolic = %impl.elem3.loc44_29.2 (constants.%impl.elem3.510)]
// CHECK:STDOUT:     %bound_method.loc44_29.3: <bound method> = bound_method %r.ref, %impl.elem3.loc44_29.1
// CHECK:STDOUT:     %Iterate.facet.loc44_29.3: %Iterate.type = facet_value constants.%R.as_type.f62, (constants.%Iterate.lookup_impl_witness.acc) [symbolic = %Iterate.facet.loc44_29.5 (constants.%Iterate.facet.b95)]
// CHECK:STDOUT:     %.loc44_29.3: %Iterate.type = converted constants.%R.as_type.f62, %Iterate.facet.loc44_29.3 [symbolic = %Iterate.facet.loc44_29.5 (constants.%Iterate.facet.b95)]
// CHECK:STDOUT:     %Iterate.facet.loc44_29.4: %Iterate.type = facet_value constants.%R.as_type.f62, (constants.%Iterate.lookup_impl_witness.acc) [symbolic = %Iterate.facet.loc44_29.5 (constants.%Iterate.facet.b95)]
// CHECK:STDOUT:     %.loc44_29.4: %Iterate.type = converted constants.%R.as_type.f62, %Iterate.facet.loc44_29.4 [symbolic = %Iterate.facet.loc44_29.5 (constants.%Iterate.facet.b95)]
// CHECK:STDOUT:     %.loc44_29.5: %Destroy.type = converted constants.%as_type.bc6, constants.%impl.elem1.411 [symbolic = %impl.elem1 (constants.%impl.elem1.411)]
// CHECK:STDOUT:     %.loc44_29.6: %Destroy.type = converted constants.%as_type.bc6, constants.%impl.elem1.411 [symbolic = %impl.elem1 (constants.%impl.elem1.411)]
// CHECK:STDOUT:     %specific_impl_fn.loc44_29.2: <specific function> = specific_impl_function %impl.elem3.loc44_29.1, @Iterate.WithSelf.Next(constants.%Iterate.facet.b95) [symbolic = %specific_impl_fn.loc44_29.5 (constants.%specific_impl_fn.a00)]
// CHECK:STDOUT:     %bound_method.loc44_29.4: <bound method> = bound_method %r.ref, %specific_impl_fn.loc44_29.2
// CHECK:STDOUT:     %.loc44_29.7: ref %Optional.95f = temporary_storage
// CHECK:STDOUT:     %.loc44_28.2: @TestIterate.%R.as_type.loc40_76.1 (%R.as_type.f62) = acquire_value %r.ref
// CHECK:STDOUT:     %Iterate.WithSelf.Next.call: init %Optional.95f to %.loc44_29.7 = call %bound_method.loc44_29.4(%.loc44_28.2, %addr.loc44)
// CHECK:STDOUT:     %.loc44_29.8: ref %Optional.95f = temporary %.loc44_29.7, %Iterate.WithSelf.Next.call
// CHECK:STDOUT:     %.loc44_29.9: %Optional.HasValue.type.006 = specific_constant imports.%Core.import_ref.bb6, @Optional(constants.%OptionalStorage.facet.705) [concrete = constants.%Optional.HasValue.988]
// CHECK:STDOUT:     %HasValue.ref: %Optional.HasValue.type.006 = name_ref HasValue, %.loc44_29.9 [concrete = constants.%Optional.HasValue.988]
// CHECK:STDOUT:     %Optional.HasValue.bound: <bound method> = bound_method %.loc44_29.8, %HasValue.ref
// CHECK:STDOUT:     %Optional.HasValue.specific_fn: <specific function> = specific_function %HasValue.ref, @Optional.HasValue(constants.%OptionalStorage.facet.705) [concrete = constants.%Optional.HasValue.specific_fn]
// CHECK:STDOUT:     %bound_method.loc44_29.5: <bound method> = bound_method %.loc44_29.8, %Optional.HasValue.specific_fn
// CHECK:STDOUT:     %.loc44_29.10: %Optional.95f = acquire_value %.loc44_29.8
// CHECK:STDOUT:     %Optional.HasValue.call: init bool = call %bound_method.loc44_29.5(%.loc44_29.10)
// CHECK:STDOUT:     %.loc44_29.11: bool = value_of_initializer %Optional.HasValue.call
// CHECK:STDOUT:     %.loc44_29.12: bool = converted %Optional.HasValue.call, %.loc44_29.11
// CHECK:STDOUT:     if %.loc44_29.12 br !for.body else br !for.done
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.body:
// CHECK:STDOUT:     %.loc44_29.13: %Optional.Get.type.30f = specific_constant imports.%Core.import_ref.daba, @Optional(constants.%OptionalStorage.facet.705) [concrete = constants.%Optional.Get.59c]
// CHECK:STDOUT:     %Get.ref: %Optional.Get.type.30f = name_ref Get, %.loc44_29.13 [concrete = constants.%Optional.Get.59c]
// CHECK:STDOUT:     %Optional.Get.bound: <bound method> = bound_method %.loc44_29.8, %Get.ref
// CHECK:STDOUT:     %Optional.Get.specific_fn: <specific function> = specific_function %Get.ref, @Optional.Get(constants.%OptionalStorage.facet.705) [concrete = constants.%Optional.Get.specific_fn]
// CHECK:STDOUT:     %bound_method.loc44_29.6: <bound method> = bound_method %.loc44_29.8, %Optional.Get.specific_fn
// CHECK:STDOUT:     %.loc44_29.14: ref %ValueType = temporary_storage
// CHECK:STDOUT:     %.loc44_29.15: %Optional.95f = acquire_value %.loc44_29.8
// CHECK:STDOUT:     %Optional.Get.call: init %ValueType to %.loc44_29.14 = call %bound_method.loc44_29.6(%.loc44_29.15)
// CHECK:STDOUT:     %.loc44_29.16: ref %ValueType = temporary %.loc44_29.14, %Optional.Get.call
// CHECK:STDOUT:     %.loc44_29.17: %ValueType = acquire_value %.loc44_29.16
// CHECK:STDOUT:     %.loc44_14: type = splice_block %ValueType.ref.loc44 [concrete = constants.%ValueType] {
// CHECK:STDOUT:       %Cpp.ref.loc44: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:       %ValueType.ref.loc44: type = name_ref ValueType, imports.%ValueType.decl [concrete = constants.%ValueType]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %i: %ValueType = wrapper_binding i, %.loc44_29.17
// CHECK:STDOUT:     %sum.ref: ref %ValueType = name_ref sum, %sum
// CHECK:STDOUT:     %i.ref: %ValueType = name_ref i, %i
// CHECK:STDOUT:     %.loc45_12: ref %ValueType = value_as_ref %i.ref
// CHECK:STDOUT:     %addr.loc45: %ptr.a3d = addr_of %.loc45_12
// CHECK:STDOUT:     %.loc45_9.1: %ptr.6bf = as_compatible %addr.loc45
// CHECK:STDOUT:     %.loc45_9.2: %ptr.6bf = converted %addr.loc45, %.loc45_9.1
// CHECK:STDOUT:     %operator_PlusEqual__carbon_thunk.call: ref %ValueType = call imports.%operator_PlusEqual__carbon_thunk.decl(%sum.ref, %.loc45_9.2)
// CHECK:STDOUT:     br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.done:
// CHECK:STDOUT:     %ValueType.Op.bound.loc44: <bound method> = bound_method %.loc44_29.16, constants.%ValueType.Op.4dca31.1
// CHECK:STDOUT:     %Op.ref.loc44: %ValueType.cpp_destructor.type = name_ref Op, imports.%ValueType.cpp_destructor.decl [concrete = constants.%ValueType.cpp_destructor]
// CHECK:STDOUT:     %ValueType.cpp_destructor.bound.loc44: <bound method> = bound_method %.loc44_29.16, %Op.ref.loc44
// CHECK:STDOUT:     %ValueType.cpp_destructor.call.loc44: init %empty_tuple.type = call %ValueType.cpp_destructor.bound.loc44(%.loc44_29.16)
// CHECK:STDOUT:     %Destroy.Op.bound: <bound method> = bound_method %.loc44_29.8, constants.%Destroy.Op.1a2547.7
// CHECK:STDOUT:     %Destroy.Op.call: init %empty_tuple.type = call %Destroy.Op.bound(%.loc44_29.8)
// CHECK:STDOUT:     %impl.elem0.loc44_29.1: @TestIterate.%.loc44_29.22 (%.4b9) = impl_witness_access constants.%Destroy.lookup_impl_witness.ff8, element0 [symbolic = %impl.elem0.loc44_29.2 (constants.%impl.elem0.fbe)]
// CHECK:STDOUT:     %bound_method.loc44_29.7: <bound method> = bound_method %var, %impl.elem0.loc44_29.1
// CHECK:STDOUT:     %.loc44_29.18: %Destroy.type = converted constants.%as_type.bc6, constants.%impl.elem1.411 [symbolic = %impl.elem1 (constants.%impl.elem1.411)]
// CHECK:STDOUT:     %.loc44_29.19: %Destroy.type = converted constants.%as_type.bc6, constants.%impl.elem1.411 [symbolic = %impl.elem1 (constants.%impl.elem1.411)]
// CHECK:STDOUT:     %specific_impl_fn.loc44_29.3: <specific function> = specific_impl_function %impl.elem0.loc44_29.1, @Destroy.WithSelf.Op(constants.%impl.elem1.411) [symbolic = %specific_impl_fn.loc44_29.6 (constants.%specific_impl_fn.517)]
// CHECK:STDOUT:     %bound_method.loc44_29.8: <bound method> = bound_method %var, %specific_impl_fn.loc44_29.3
// CHECK:STDOUT:     %Destroy.WithSelf.Op.call: init %empty_tuple.type = call %bound_method.loc44_29.8(%var)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc44_29.1(%self.param: ref %.98c) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc44_29.2(%self.param: ref %MaybeUnformed.02e) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc44_29.3(%self.param: ref bool) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc44_29.4(%self.param: ref %struct_type.value.has_value.761) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc44_29.5(%self.param: ref %DefaultOptionalStorage.84c) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc44_29.6(%self.param: ref %Optional.95f) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestRangeFor(%m.param: ref %MutableRange, %c.param: %ConstRange) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %i.patt.loc59: %pattern_type.e9f = value_binding_pattern i [concrete = constants.%i.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %m.ref: ref %MutableRange = name_ref m, %m
// CHECK:STDOUT:   %impl.elem2.loc59: %.47d = impl_witness_access constants.%Iterate.impl_witness.096, element2 [concrete = constants.%R.as_type.as.Iterate.impl.NewCursor.498]
// CHECK:STDOUT:   %bound_method.loc59_29.1: <bound method> = bound_method %m.ref, %impl.elem2.loc59
// CHECK:STDOUT:   %facet_value.loc59_29.1: %CppRangeForIterate_where.type = facet_value constants.%MutableRange, (constants.%custom_witness.e7a31c.1, constants.%custom_witness.d0da43.1, constants.%custom_witness.6b6, constants.%custom_witness.95581e.1, constants.%custom_witness.1f52e9.1, constants.%custom_witness.90a, constants.%custom_witness.3e8, constants.%custom_witness.444, constants.%custom_witness.95581e.2, constants.%custom_witness.1f52e9.2) [concrete = constants.%facet_value.55a]
// CHECK:STDOUT:   %.loc59_29.1: %CppRangeForIterate_where.type = converted constants.%MutableRange, %facet_value.loc59_29.1 [concrete = constants.%facet_value.55a]
// CHECK:STDOUT:   %facet_value.loc59_29.2: %CppRangeForIterate_where.type = facet_value constants.%MutableRange, (constants.%custom_witness.e7a31c.1, constants.%custom_witness.d0da43.1, constants.%custom_witness.6b6, constants.%custom_witness.95581e.1, constants.%custom_witness.1f52e9.1, constants.%custom_witness.90a, constants.%custom_witness.3e8, constants.%custom_witness.444, constants.%custom_witness.95581e.2, constants.%custom_witness.1f52e9.2) [concrete = constants.%facet_value.55a]
// CHECK:STDOUT:   %.loc59_29.2: %CppRangeForIterate_where.type = converted constants.%MutableRange, %facet_value.loc59_29.2 [concrete = constants.%facet_value.55a]
// CHECK:STDOUT:   %specific_fn.loc59_29.1: <specific function> = specific_function %impl.elem2.loc59, @R.as_type.as.Iterate.impl.NewCursor(constants.%facet_value.55a) [concrete = constants.%R.as_type.as.Iterate.impl.NewCursor.specific_fn.532]
// CHECK:STDOUT:   %bound_method.loc59_29.2: <bound method> = bound_method %m.ref, %specific_fn.loc59_29.1
// CHECK:STDOUT:   %var.loc59: ref %tuple.type.829 = var invalid
// CHECK:STDOUT:   %.loc59_28.1: %MutableRange = acquire_value %m.ref
// CHECK:STDOUT:   %R.as_type.as.Iterate.impl.NewCursor.call.loc59: init %tuple.type.829 to %var.loc59 = call %bound_method.loc59_29.2(%.loc59_28.1)
// CHECK:STDOUT:   assign %var.loc59, %R.as_type.as.Iterate.impl.NewCursor.call.loc59
// CHECK:STDOUT:   br !for.next.loc59
// CHECK:STDOUT:
// CHECK:STDOUT: !for.next.loc59:
// CHECK:STDOUT:   %addr.loc59: %ptr.d6b = addr_of %var.loc59
// CHECK:STDOUT:   %impl.elem3.loc59: %.fd2 = impl_witness_access constants.%Iterate.impl_witness.096, element3 [concrete = constants.%R.as_type.as.Iterate.impl.Next.055]
// CHECK:STDOUT:   %bound_method.loc59_29.3: <bound method> = bound_method %m.ref, %impl.elem3.loc59
// CHECK:STDOUT:   %facet_value.loc59_29.3: %CppRangeForIterate_where.type = facet_value constants.%MutableRange, (constants.%custom_witness.e7a31c.1, constants.%custom_witness.d0da43.1, constants.%custom_witness.6b6, constants.%custom_witness.95581e.1, constants.%custom_witness.1f52e9.1, constants.%custom_witness.90a, constants.%custom_witness.3e8, constants.%custom_witness.444, constants.%custom_witness.95581e.2, constants.%custom_witness.1f52e9.2) [concrete = constants.%facet_value.55a]
// CHECK:STDOUT:   %.loc59_29.3: %CppRangeForIterate_where.type = converted constants.%MutableRange, %facet_value.loc59_29.3 [concrete = constants.%facet_value.55a]
// CHECK:STDOUT:   %facet_value.loc59_29.4: %CppRangeForIterate_where.type = facet_value constants.%MutableRange, (constants.%custom_witness.e7a31c.1, constants.%custom_witness.d0da43.1, constants.%custom_witness.6b6, constants.%custom_witness.95581e.1, constants.%custom_witness.1f52e9.1, constants.%custom_witness.90a, constants.%custom_witness.3e8, constants.%custom_witness.444, constants.%custom_witness.95581e.2, constants.%custom_witness.1f52e9.2) [concrete = constants.%facet_value.55a]
// CHECK:STDOUT:   %.loc59_29.4: %CppRangeForIterate_where.type = converted constants.%MutableRange, %facet_value.loc59_29.4 [concrete = constants.%facet_value.55a]
// CHECK:STDOUT:   %specific_fn.loc59_29.2: <specific function> = specific_function %impl.elem3.loc59, @R.as_type.as.Iterate.impl.Next(constants.%facet_value.55a) [concrete = constants.%R.as_type.as.Iterate.impl.Next.specific_fn.5cb]
// CHECK:STDOUT:   %bound_method.loc59_29.4: <bound method> = bound_method %m.ref, %specific_fn.loc59_29.2
// CHECK:STDOUT:   %.loc59_29.5: ref %Optional.95f = temporary_storage
// CHECK:STDOUT:   %.loc59_28.2: %MutableRange = acquire_value %m.ref
// CHECK:STDOUT:   %R.as_type.as.Iterate.impl.Next.call.loc59: init %Optional.95f to %.loc59_29.5 = call %bound_method.loc59_29.4(%.loc59_28.2, %addr.loc59)
// CHECK:STDOUT:   %.loc59_29.6: ref %Optional.95f = temporary %.loc59_29.5, %R.as_type.as.Iterate.impl.Next.call.loc59
// CHECK:STDOUT:   %.loc59_29.7: %Optional.HasValue.type.006 = specific_constant imports.%Core.import_ref.bb6, @Optional(constants.%OptionalStorage.facet.705) [concrete = constants.%Optional.HasValue.988]
// CHECK:STDOUT:   %HasValue.ref.loc59: %Optional.HasValue.type.006 = name_ref HasValue, %.loc59_29.7 [concrete = constants.%Optional.HasValue.988]
// CHECK:STDOUT:   %Optional.HasValue.bound.loc59: <bound method> = bound_method %.loc59_29.6, %HasValue.ref.loc59
// CHECK:STDOUT:   %Optional.HasValue.specific_fn.loc59: <specific function> = specific_function %HasValue.ref.loc59, @Optional.HasValue(constants.%OptionalStorage.facet.705) [concrete = constants.%Optional.HasValue.specific_fn]
// CHECK:STDOUT:   %bound_method.loc59_29.5: <bound method> = bound_method %.loc59_29.6, %Optional.HasValue.specific_fn.loc59
// CHECK:STDOUT:   %.loc59_29.8: %Optional.95f = acquire_value %.loc59_29.6
// CHECK:STDOUT:   %Optional.HasValue.call.loc59: init bool = call %bound_method.loc59_29.5(%.loc59_29.8)
// CHECK:STDOUT:   %.loc59_29.9: bool = value_of_initializer %Optional.HasValue.call.loc59
// CHECK:STDOUT:   %.loc59_29.10: bool = converted %Optional.HasValue.call.loc59, %.loc59_29.9
// CHECK:STDOUT:   if %.loc59_29.10 br !for.body.loc59 else br !for.done.loc59
// CHECK:STDOUT:
// CHECK:STDOUT: !for.body.loc59:
// CHECK:STDOUT:   %.loc59_29.11: %Optional.Get.type.30f = specific_constant imports.%Core.import_ref.daba, @Optional(constants.%OptionalStorage.facet.705) [concrete = constants.%Optional.Get.59c]
// CHECK:STDOUT:   %Get.ref.loc59: %Optional.Get.type.30f = name_ref Get, %.loc59_29.11 [concrete = constants.%Optional.Get.59c]
// CHECK:STDOUT:   %Optional.Get.bound.loc59: <bound method> = bound_method %.loc59_29.6, %Get.ref.loc59
// CHECK:STDOUT:   %Optional.Get.specific_fn.loc59: <specific function> = specific_function %Get.ref.loc59, @Optional.Get(constants.%OptionalStorage.facet.705) [concrete = constants.%Optional.Get.specific_fn]
// CHECK:STDOUT:   %bound_method.loc59_29.6: <bound method> = bound_method %.loc59_29.6, %Optional.Get.specific_fn.loc59
// CHECK:STDOUT:   %.loc59_29.12: ref %ValueType = temporary_storage
// CHECK:STDOUT:   %.loc59_29.13: %Optional.95f = acquire_value %.loc59_29.6
// CHECK:STDOUT:   %Optional.Get.call.loc59: init %ValueType to %.loc59_29.12 = call %bound_method.loc59_29.6(%.loc59_29.13)
// CHECK:STDOUT:   %.loc59_29.14: ref %ValueType = temporary %.loc59_29.12, %Optional.Get.call.loc59
// CHECK:STDOUT:   %.loc59_29.15: %ValueType = acquire_value %.loc59_29.14
// CHECK:STDOUT:   %.loc59_14: type = splice_block %ValueType.ref.loc59 [concrete = constants.%ValueType] {
// CHECK:STDOUT:     %Cpp.ref.loc59: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %ValueType.ref.loc59: type = name_ref ValueType, imports.%ValueType.decl [concrete = constants.%ValueType]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %i.loc59: %ValueType = wrapper_binding i, %.loc59_29.15
// CHECK:STDOUT:   %sum.ref.loc60: ref %ValueType = name_ref sum, %sum
// CHECK:STDOUT:   %i.ref.loc60: %ValueType = name_ref i, %i.loc59
// CHECK:STDOUT:   %.loc60_12: ref %ValueType = value_as_ref %i.ref.loc60
// CHECK:STDOUT:   %addr.loc60: %ptr.a3d = addr_of %.loc60_12
// CHECK:STDOUT:   %.loc60_9.1: %ptr.6bf = as_compatible %addr.loc60
// CHECK:STDOUT:   %.loc60_9.2: %ptr.6bf = converted %addr.loc60, %.loc60_9.1
// CHECK:STDOUT:   %operator_PlusEqual__carbon_thunk.call.loc60: ref %ValueType = call imports.%operator_PlusEqual__carbon_thunk.decl(%sum.ref.loc60, %.loc60_9.2)
// CHECK:STDOUT:   br !for.next.loc59
// CHECK:STDOUT:
// CHECK:STDOUT: !for.done.loc59:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %i.patt.loc63: %pattern_type.e9f = value_binding_pattern i [concrete = constants.%i.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c.ref: %ConstRange = name_ref c, %c
// CHECK:STDOUT:   %impl.elem2.loc63: %.763 = impl_witness_access constants.%Iterate.impl_witness.40c, element2 [concrete = constants.%R.as_type.as.Iterate.impl.NewCursor.bd2]
// CHECK:STDOUT:   %bound_method.loc63_29.1: <bound method> = bound_method %c.ref, %impl.elem2.loc63
// CHECK:STDOUT:   %facet_value.loc63_29.1: %CppRangeForIterate_where.type = facet_value constants.%ConstRange, (constants.%custom_witness.e7a31c.1, constants.%custom_witness.d0da43.1, constants.%custom_witness.a03, constants.%custom_witness.6e8a2e.1, constants.%custom_witness.15091a.1, constants.%custom_witness.78b, constants.%custom_witness.c55, constants.%custom_witness.746, constants.%custom_witness.6e8a2e.2, constants.%custom_witness.15091a.2) [concrete = constants.%facet_value.445]
// CHECK:STDOUT:   %.loc63_29.1: %CppRangeForIterate_where.type = converted constants.%ConstRange, %facet_value.loc63_29.1 [concrete = constants.%facet_value.445]
// CHECK:STDOUT:   %facet_value.loc63_29.2: %CppRangeForIterate_where.type = facet_value constants.%ConstRange, (constants.%custom_witness.e7a31c.1, constants.%custom_witness.d0da43.1, constants.%custom_witness.a03, constants.%custom_witness.6e8a2e.1, constants.%custom_witness.15091a.1, constants.%custom_witness.78b, constants.%custom_witness.c55, constants.%custom_witness.746, constants.%custom_witness.6e8a2e.2, constants.%custom_witness.15091a.2) [concrete = constants.%facet_value.445]
// CHECK:STDOUT:   %.loc63_29.2: %CppRangeForIterate_where.type = converted constants.%ConstRange, %facet_value.loc63_29.2 [concrete = constants.%facet_value.445]
// CHECK:STDOUT:   %specific_fn.loc63_29.1: <specific function> = specific_function %impl.elem2.loc63, @R.as_type.as.Iterate.impl.NewCursor(constants.%facet_value.445) [concrete = constants.%R.as_type.as.Iterate.impl.NewCursor.specific_fn.7c7]
// CHECK:STDOUT:   %bound_method.loc63_29.2: <bound method> = bound_method %c.ref, %specific_fn.loc63_29.1
// CHECK:STDOUT:   %var.loc63: ref %tuple.type.99f = var invalid
// CHECK:STDOUT:   %R.as_type.as.Iterate.impl.NewCursor.call.loc63: init %tuple.type.99f to %var.loc63 = call %bound_method.loc63_29.2(%c.ref)
// CHECK:STDOUT:   assign %var.loc63, %R.as_type.as.Iterate.impl.NewCursor.call.loc63
// CHECK:STDOUT:   br !for.next.loc63
// CHECK:STDOUT:
// CHECK:STDOUT: !for.next.loc63:
// CHECK:STDOUT:   %addr.loc63: %ptr.2f7 = addr_of %var.loc63
// CHECK:STDOUT:   %impl.elem3.loc63: %.bca = impl_witness_access constants.%Iterate.impl_witness.40c, element3 [concrete = constants.%R.as_type.as.Iterate.impl.Next.c50]
// CHECK:STDOUT:   %bound_method.loc63_29.3: <bound method> = bound_method %c.ref, %impl.elem3.loc63
// CHECK:STDOUT:   %facet_value.loc63_29.3: %CppRangeForIterate_where.type = facet_value constants.%ConstRange, (constants.%custom_witness.e7a31c.1, constants.%custom_witness.d0da43.1, constants.%custom_witness.a03, constants.%custom_witness.6e8a2e.1, constants.%custom_witness.15091a.1, constants.%custom_witness.78b, constants.%custom_witness.c55, constants.%custom_witness.746, constants.%custom_witness.6e8a2e.2, constants.%custom_witness.15091a.2) [concrete = constants.%facet_value.445]
// CHECK:STDOUT:   %.loc63_29.3: %CppRangeForIterate_where.type = converted constants.%ConstRange, %facet_value.loc63_29.3 [concrete = constants.%facet_value.445]
// CHECK:STDOUT:   %facet_value.loc63_29.4: %CppRangeForIterate_where.type = facet_value constants.%ConstRange, (constants.%custom_witness.e7a31c.1, constants.%custom_witness.d0da43.1, constants.%custom_witness.a03, constants.%custom_witness.6e8a2e.1, constants.%custom_witness.15091a.1, constants.%custom_witness.78b, constants.%custom_witness.c55, constants.%custom_witness.746, constants.%custom_witness.6e8a2e.2, constants.%custom_witness.15091a.2) [concrete = constants.%facet_value.445]
// CHECK:STDOUT:   %.loc63_29.4: %CppRangeForIterate_where.type = converted constants.%ConstRange, %facet_value.loc63_29.4 [concrete = constants.%facet_value.445]
// CHECK:STDOUT:   %specific_fn.loc63_29.2: <specific function> = specific_function %impl.elem3.loc63, @R.as_type.as.Iterate.impl.Next(constants.%facet_value.445) [concrete = constants.%R.as_type.as.Iterate.impl.Next.specific_fn.a9a]
// CHECK:STDOUT:   %bound_method.loc63_29.4: <bound method> = bound_method %c.ref, %specific_fn.loc63_29.2
// CHECK:STDOUT:   %.loc63_29.5: ref %Optional.95f = temporary_storage
// CHECK:STDOUT:   %R.as_type.as.Iterate.impl.Next.call.loc63: init %Optional.95f to %.loc63_29.5 = call %bound_method.loc63_29.4(%c.ref, %addr.loc63)
// CHECK:STDOUT:   %.loc63_29.6: ref %Optional.95f = temporary %.loc63_29.5, %R.as_type.as.Iterate.impl.Next.call.loc63
// CHECK:STDOUT:   %.loc63_29.7: %Optional.HasValue.type.006 = specific_constant imports.%Core.import_ref.bb6, @Optional(constants.%OptionalStorage.facet.705) [concrete = constants.%Optional.HasValue.988]
// CHECK:STDOUT:   %HasValue.ref.loc63: %Optional.HasValue.type.006 = name_ref HasValue, %.loc63_29.7 [concrete = constants.%Optional.HasValue.988]
// CHECK:STDOUT:   %Optional.HasValue.bound.loc63: <bound method> = bound_method %.loc63_29.6, %HasValue.ref.loc63
// CHECK:STDOUT:   %Optional.HasValue.specific_fn.loc63: <specific function> = specific_function %HasValue.ref.loc63, @Optional.HasValue(constants.%OptionalStorage.facet.705) [concrete = constants.%Optional.HasValue.specific_fn]
// CHECK:STDOUT:   %bound_method.loc63_29.5: <bound method> = bound_method %.loc63_29.6, %Optional.HasValue.specific_fn.loc63
// CHECK:STDOUT:   %.loc63_29.8: %Optional.95f = acquire_value %.loc63_29.6
// CHECK:STDOUT:   %Optional.HasValue.call.loc63: init bool = call %bound_method.loc63_29.5(%.loc63_29.8)
// CHECK:STDOUT:   %.loc63_29.9: bool = value_of_initializer %Optional.HasValue.call.loc63
// CHECK:STDOUT:   %.loc63_29.10: bool = converted %Optional.HasValue.call.loc63, %.loc63_29.9
// CHECK:STDOUT:   if %.loc63_29.10 br !for.body.loc63 else br !for.done.loc63
// CHECK:STDOUT:
// CHECK:STDOUT: !for.body.loc63:
// CHECK:STDOUT:   %.loc63_29.11: %Optional.Get.type.30f = specific_constant imports.%Core.import_ref.daba, @Optional(constants.%OptionalStorage.facet.705) [concrete = constants.%Optional.Get.59c]
// CHECK:STDOUT:   %Get.ref.loc63: %Optional.Get.type.30f = name_ref Get, %.loc63_29.11 [concrete = constants.%Optional.Get.59c]
// CHECK:STDOUT:   %Optional.Get.bound.loc63: <bound method> = bound_method %.loc63_29.6, %Get.ref.loc63
// CHECK:STDOUT:   %Optional.Get.specific_fn.loc63: <specific function> = specific_function %Get.ref.loc63, @Optional.Get(constants.%OptionalStorage.facet.705) [concrete = constants.%Optional.Get.specific_fn]
// CHECK:STDOUT:   %bound_method.loc63_29.6: <bound method> = bound_method %.loc63_29.6, %Optional.Get.specific_fn.loc63
// CHECK:STDOUT:   %.loc63_29.12: ref %ValueType = temporary_storage
// CHECK:STDOUT:   %.loc63_29.13: %Optional.95f = acquire_value %.loc63_29.6
// CHECK:STDOUT:   %Optional.Get.call.loc63: init %ValueType to %.loc63_29.12 = call %bound_method.loc63_29.6(%.loc63_29.13)
// CHECK:STDOUT:   %.loc63_29.14: ref %ValueType = temporary %.loc63_29.12, %Optional.Get.call.loc63
// CHECK:STDOUT:   %.loc63_29.15: %ValueType = acquire_value %.loc63_29.14
// CHECK:STDOUT:   %.loc63_14: type = splice_block %ValueType.ref.loc63 [concrete = constants.%ValueType] {
// CHECK:STDOUT:     %Cpp.ref.loc63: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %ValueType.ref.loc63: type = name_ref ValueType, imports.%ValueType.decl [concrete = constants.%ValueType]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %i.loc63: %ValueType = wrapper_binding i, %.loc63_29.15
// CHECK:STDOUT:   %sum.ref.loc64: ref %ValueType = name_ref sum, %sum
// CHECK:STDOUT:   %i.ref.loc64: %ValueType = name_ref i, %i.loc63
// CHECK:STDOUT:   %.loc64_12: ref %ValueType = value_as_ref %i.ref.loc64
// CHECK:STDOUT:   %addr.loc64: %ptr.a3d = addr_of %.loc64_12
// CHECK:STDOUT:   %.loc64_9.1: %ptr.6bf = as_compatible %addr.loc64
// CHECK:STDOUT:   %.loc64_9.2: %ptr.6bf = converted %addr.loc64, %.loc64_9.1
// CHECK:STDOUT:   %operator_PlusEqual__carbon_thunk.call.loc64: ref %ValueType = call imports.%operator_PlusEqual__carbon_thunk.decl(%sum.ref.loc64, %.loc64_9.2)
// CHECK:STDOUT:   br !for.next.loc63
// CHECK:STDOUT:
// CHECK:STDOUT: !for.done.loc63:
// CHECK:STDOUT:   %ValueType.Op.bound.loc63: <bound method> = bound_method %.loc63_29.14, constants.%ValueType.Op.4dca31.1
// CHECK:STDOUT:   %Op.ref.loc63: %ValueType.cpp_destructor.type = name_ref Op, imports.%ValueType.cpp_destructor.decl [concrete = constants.%ValueType.cpp_destructor]
// CHECK:STDOUT:   %ValueType.cpp_destructor.bound.loc63: <bound method> = bound_method %.loc63_29.14, %Op.ref.loc63
// CHECK:STDOUT:   %ValueType.cpp_destructor.call.loc63: init %empty_tuple.type = call %ValueType.cpp_destructor.bound.loc63(%.loc63_29.14)
// CHECK:STDOUT:   %Destroy.Op.bound.loc63_29.1: <bound method> = bound_method %.loc63_29.6, constants.%Destroy.Op.1a2547.7
// CHECK:STDOUT:   %Destroy.Op.call.loc63_29.1: init %empty_tuple.type = call %Destroy.Op.bound.loc63_29.1(%.loc63_29.6)
// CHECK:STDOUT:   %Destroy.Op.bound.loc63_29.2: <bound method> = bound_method %var.loc63, constants.%Destroy.Op.1a2547.9
// CHECK:STDOUT:   %Destroy.Op.call.loc63_29.2: init %empty_tuple.type = call %Destroy.Op.bound.loc63_29.2(%var.loc63)
// CHECK:STDOUT:   %ValueType.Op.bound.loc59: <bound method> = bound_method %.loc59_29.14, constants.%ValueType.Op.4dca31.1
// CHECK:STDOUT:   %Op.ref.loc59: %ValueType.cpp_destructor.type = name_ref Op, imports.%ValueType.cpp_destructor.decl [concrete = constants.%ValueType.cpp_destructor]
// CHECK:STDOUT:   %ValueType.cpp_destructor.bound.loc59: <bound method> = bound_method %.loc59_29.14, %Op.ref.loc59
// CHECK:STDOUT:   %ValueType.cpp_destructor.call.loc59: init %empty_tuple.type = call %ValueType.cpp_destructor.bound.loc59(%.loc59_29.14)
// CHECK:STDOUT:   %Destroy.Op.bound.loc59_29.1: <bound method> = bound_method %.loc59_29.6, constants.%Destroy.Op.1a2547.7
// CHECK:STDOUT:   %Destroy.Op.call.loc59_29.1: init %empty_tuple.type = call %Destroy.Op.bound.loc59_29.1(%.loc59_29.6)
// CHECK:STDOUT:   %Destroy.Op.bound.loc59_29.2: <bound method> = bound_method %var.loc59, constants.%Destroy.Op.1a2547.8
// CHECK:STDOUT:   %Destroy.Op.call.loc59_29.2: init %empty_tuple.type = call %Destroy.Op.bound.loc59_29.2(%var.loc59)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestIterate(constants.%R.de1) {
// CHECK:STDOUT:   %R.patt.loc40_17.2 => constants.%R.patt.f38
// CHECK:STDOUT:   %R.loc40_17.1 => constants.%R.de1
// CHECK:STDOUT:   %R.as_type.loc40_76.1 => constants.%R.as_type.f62
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.669
// CHECK:STDOUT:   %r.patt.loc40_74.2 => constants.%r.patt.c0f
// CHECK:STDOUT:   %r.param_patt.loc40_69.2 => constants.%r.param_patt.bb2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestIterate(constants.%facet_value.150) {
// CHECK:STDOUT:   %R.patt.loc40_17.2 => constants.%R.patt.f38
// CHECK:STDOUT:   %R.loc40_17.1 => constants.%facet_value.150
// CHECK:STDOUT:   %R.as_type.loc40_76.1 => constants.%MutableRange
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.e635
// CHECK:STDOUT:   %r.patt.loc40_74.2 => constants.%r.patt.f69
// CHECK:STDOUT:   %r.param_patt.loc40_69.2 => constants.%r.param_patt.91e
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc40 => constants.%complete_type.357
// CHECK:STDOUT:   %Iterate.lookup_impl_witness => constants.%Iterate.impl_witness.096
// CHECK:STDOUT:   %Iterate.facet.loc44_29.5 => constants.%Iterate.facet.723
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type => constants.%Iterate.WithSelf.NewCursor.type.47d
// CHECK:STDOUT:   %.loc44_29.20 => constants.%.47d
// CHECK:STDOUT:   %impl.elem2.loc44_29.2 => constants.%R.as_type.as.Iterate.impl.NewCursor.498
// CHECK:STDOUT:   %specific_impl_fn.loc44_29.4 => constants.%R.as_type.as.Iterate.impl.NewCursor.specific_fn.532
// CHECK:STDOUT:   %impl.elem1 => constants.%Destroy.facet.305
// CHECK:STDOUT:   %as_type => constants.%tuple.type.829
// CHECK:STDOUT:   %require_complete.1 => constants.%complete_type.94f
// CHECK:STDOUT:   %ptr => constants.%ptr.d6b
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type => constants.%Iterate.WithSelf.Next.type.6f1
// CHECK:STDOUT:   %.loc44_29.21 => constants.%.fd2
// CHECK:STDOUT:   %impl.elem3.loc44_29.2 => constants.%R.as_type.as.Iterate.impl.Next.055
// CHECK:STDOUT:   %specific_impl_fn.loc44_29.5 => constants.%R.as_type.as.Iterate.impl.Next.specific_fn.5cb
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.c89
// CHECK:STDOUT:   %.loc44_29.22 => constants.%.e8a
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.df9cc1.8
// CHECK:STDOUT:   %impl.elem0.loc44_29.2 => constants.%Destroy.Op.1a2547.8
// CHECK:STDOUT:   %specific_impl_fn.loc44_29.6 => constants.%Destroy.Op.1a2547.8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestIterate(constants.%facet_value.e78) {
// CHECK:STDOUT:   %R.patt.loc40_17.2 => constants.%R.patt.f38
// CHECK:STDOUT:   %R.loc40_17.1 => constants.%facet_value.e78
// CHECK:STDOUT:   %R.as_type.loc40_76.1 => constants.%ConstRange
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.0ba
// CHECK:STDOUT:   %r.patt.loc40_74.2 => constants.%r.patt.36a
// CHECK:STDOUT:   %r.param_patt.loc40_69.2 => constants.%r.param_patt.501
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc40 => constants.%complete_type.357
// CHECK:STDOUT:   %Iterate.lookup_impl_witness => constants.%Iterate.impl_witness.40c
// CHECK:STDOUT:   %Iterate.facet.loc44_29.5 => constants.%Iterate.facet.c46
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type => constants.%Iterate.WithSelf.NewCursor.type.cd5
// CHECK:STDOUT:   %.loc44_29.20 => constants.%.763
// CHECK:STDOUT:   %impl.elem2.loc44_29.2 => constants.%R.as_type.as.Iterate.impl.NewCursor.bd2
// CHECK:STDOUT:   %specific_impl_fn.loc44_29.4 => constants.%R.as_type.as.Iterate.impl.NewCursor.specific_fn.7c7
// CHECK:STDOUT:   %impl.elem1 => constants.%Destroy.facet.e08
// CHECK:STDOUT:   %as_type => constants.%tuple.type.99f
// CHECK:STDOUT:   %require_complete.1 => constants.%complete_type.dea
// CHECK:STDOUT:   %ptr => constants.%ptr.2f7
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type => constants.%Iterate.WithSelf.Next.type.285
// CHECK:STDOUT:   %.loc44_29.21 => constants.%.bca
// CHECK:STDOUT:   %impl.elem3.loc44_29.2 => constants.%R.as_type.as.Iterate.impl.Next.c50
// CHECK:STDOUT:   %specific_impl_fn.loc44_29.5 => constants.%R.as_type.as.Iterate.impl.Next.specific_fn.a9a
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.e30
// CHECK:STDOUT:   %.loc44_29.22 => constants.%.987
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.df9cc1.9
// CHECK:STDOUT:   %impl.elem0.loc44_29.2 => constants.%Destroy.Op.1a2547.9
// CHECK:STDOUT:   %specific_impl_fn.loc44_29.6 => constants.%Destroy.Op.1a2547.9
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
// CHECK:STDOUT:   %T.220: %OptionalStorage.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Optional.Get.type.bb9: type = fn_type @Optional.Get, @Optional(%T.220) [symbolic]
// CHECK:STDOUT:   %Optional.Get.fa1: %Optional.Get.type.bb9 = struct_value () [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.type.e9e: type = fn_type @Optional.HasValue, @Optional(%T.220) [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.6cd: %Optional.HasValue.type.e9e = struct_value () [symbolic]
// CHECK:STDOUT:   %T.283: %facet_type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.d50: type = class_type @DefaultOptionalStorage, @DefaultOptionalStorage(%T.283) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Get.type.029: type = fn_type @T.as_type.as.OptionalStorage.impl.Get, @T.as_type.as.OptionalStorage.impl(%T.283) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Get.cb5: %T.as_type.as.OptionalStorage.impl.Get.type.029 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Has.type.4bc: type = fn_type @T.as_type.as.OptionalStorage.impl.Has, @T.as_type.as.OptionalStorage.impl(%T.283) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Has.dc2: %T.as_type.as.OptionalStorage.impl.Has.type.4bc = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Some.type.800: type = fn_type @T.as_type.as.OptionalStorage.impl.Some, @T.as_type.as.OptionalStorage.impl(%T.283) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Some.e83: %T.as_type.as.OptionalStorage.impl.Some.type.800 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.None.type.300: type = fn_type @T.as_type.as.OptionalStorage.impl.None, @T.as_type.as.OptionalStorage.impl(%T.283) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.None.f54: %T.as_type.as.OptionalStorage.impl.None.type.300 = struct_value () [symbolic]
// CHECK:STDOUT:   %.Self.35d: %Iterate.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %Iterate.lookup_impl_witness.b37: <witness> = lookup_impl_witness %.Self.35d, @Iterate [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.903: %facet_type = impl_witness_access %Iterate.lookup_impl_witness.b37, element0 [symbolic_self]
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete]
// CHECK:STDOUT:   %N.fe9: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %f64.dc1: type = class_type @Float, @Float(%int_64) [concrete]
// CHECK:STDOUT:   %pattern_type.fb7: type = pattern_type %f64.dc1 [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.2: type = fn_type @Destroy.Op.loc20_54.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.2: %Destroy.Op.type.1d8f74.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.df9cc1.2: <witness> = custom_witness (%Destroy.Op.1a2547.2), @Destroy [concrete]
// CHECK:STDOUT:   %Float.as.Copy.impl.Op.type.ff3: type = fn_type @Float.as.Copy.impl.Op, @Float.as.Copy.impl(%N.fe9) [symbolic]
// CHECK:STDOUT:   %Float.as.Copy.impl.Op.d93: %Float.as.Copy.impl.Op.type.ff3 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.873: <witness> = impl_witness imports.%Copy.impl_witness_table.4e8, @Float.as.Copy.impl(%int_64) [concrete]
// CHECK:STDOUT:   %facet_value.209: %facet_type = facet_value %f64.dc1, (%custom_witness.df9cc1.2, %Copy.impl_witness.873) [concrete]
// CHECK:STDOUT:   %Iterate_where.type.83b: type = facet_type <@Iterate where %impl.elem0.903 = %facet_value.209> [concrete]
// CHECK:STDOUT:   %pattern_type.b62: type = pattern_type %Iterate_where.type.83b [concrete]
// CHECK:STDOUT:   %R.patt.053: %pattern_type.b62 = symbolic_binding_pattern R, 0 [symbolic]
// CHECK:STDOUT:   %R.7b2: %Iterate_where.type.83b = symbolic_binding R, 0 [symbolic]
// CHECK:STDOUT:   %R.as_type.657: type = facet_access_type %R.7b2 [symbolic]
// CHECK:STDOUT:   %pattern_type.afe: type = pattern_type %R.as_type.657 [symbolic]
// CHECK:STDOUT:   %r.param_patt.184: %pattern_type.afe = value_param_pattern [symbolic]
// CHECK:STDOUT:   %r.patt.727: %pattern_type.afe = at_binding_pattern r, %r.param_patt.184 [symbolic]
// CHECK:STDOUT:   %i.patt: %pattern_type.fb7 = value_binding_pattern i [concrete]
// CHECK:STDOUT:   %CppRangeForIterate.type: type = facet_type <@CppRangeForIterate> [concrete]
// CHECK:STDOUT:   %.Self.6bd: %CppRangeForIterate.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %CppRangeForIterate.lookup_impl_witness.143: <witness> = lookup_impl_witness %.Self.6bd, @CppRangeForIterate [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.109: type = impl_witness_access %CppRangeForIterate.lookup_impl_witness.143, element0 [symbolic_self]
// CHECK:STDOUT:   %CppUnsafeDeref.lookup_impl_witness.712: <witness> = lookup_impl_witness %impl.elem0.109, @CppUnsafeDeref [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.56a: type = impl_witness_access %CppUnsafeDeref.lookup_impl_witness.712, element0 [symbolic_self]
// CHECK:STDOUT:   %impl.elem1.222: type = impl_witness_access %CppRangeForIterate.lookup_impl_witness.143, element1 [symbolic_self]
// CHECK:STDOUT:   %CppRangeForIterate_where.type: type = facet_type <@CppRangeForIterate where %impl.elem0.109 impls @Destroy and %impl.elem0.109 impls @Copy and %impl.elem0.109 impls @CppUnsafeDeref and %impl.elem0.109 impls @EqWith, @EqWith(%impl.elem1.222) and %impl.elem0.109 impls @Inc and %impl.elem0.56a impls @Destroy and %impl.elem0.56a impls @Copy and %impl.elem1.222 impls @Destroy and %impl.elem1.222 impls @Copy> [concrete]
// CHECK:STDOUT:   %R.39b: %CppRangeForIterate_where.type = symbolic_binding R, 0 [symbolic]
// CHECK:STDOUT:   %CppRangeForIterate.lookup_impl_witness.7e2712.2: <witness> = lookup_impl_witness %R.39b, @CppRangeForIterate [symbolic]
// CHECK:STDOUT:   %impl.elem1.7a0bff.2: type = impl_witness_access %CppRangeForIterate.lookup_impl_witness.7e2712.2, element1 [symbolic]
// CHECK:STDOUT:   %impl.elem0.73eb7f.2: type = impl_witness_access %CppRangeForIterate.lookup_impl_witness.7e2712.2, element0 [symbolic]
// CHECK:STDOUT:   %tuple.type.3cf: type = tuple_type (%impl.elem0.73eb7f.2, %impl.elem1.7a0bff.2) [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.69f: <witness> = lookup_impl_witness %tuple.type.3cf, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.593: %Destroy.type = facet_value %tuple.type.3cf, (%Destroy.lookup_impl_witness.69f) [symbolic]
// CHECK:STDOUT:   %CppUnsafeDeref.lookup_impl_witness.aa7: <witness> = lookup_impl_witness %impl.elem0.73eb7f.2, @CppUnsafeDeref [symbolic]
// CHECK:STDOUT:   %impl.elem0.af7: type = impl_witness_access %CppUnsafeDeref.lookup_impl_witness.aa7, element0 [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.ec3: <witness> = lookup_impl_witness %impl.elem0.af7, @Destroy [symbolic]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.a7a: <witness> = lookup_impl_witness %impl.elem0.af7, @Copy [symbolic]
// CHECK:STDOUT:   %facet_value.194: %facet_type = facet_value %impl.elem0.af7, (%Destroy.lookup_impl_witness.ec3, %Copy.lookup_impl_witness.a7a) [symbolic]
// CHECK:STDOUT:   %R.as_type.as.Iterate.impl.Next.type.1f7: type = fn_type @R.as_type.as.Iterate.impl.Next, @R.as_type.as.Iterate.impl(%R.39b) [symbolic]
// CHECK:STDOUT:   %R.as_type.as.Iterate.impl.Next.0d6: %R.as_type.as.Iterate.impl.Next.type.1f7 = struct_value () [symbolic]
// CHECK:STDOUT:   %R.as_type.as.Iterate.impl.NewCursor.type.36f: type = fn_type @R.as_type.as.Iterate.impl.NewCursor, @R.as_type.as.Iterate.impl(%R.39b) [symbolic]
// CHECK:STDOUT:   %R.as_type.as.Iterate.impl.NewCursor.29b: %R.as_type.as.Iterate.impl.NewCursor.type.36f = struct_value () [symbolic]
// CHECK:STDOUT:   %Iterate.lookup_impl_witness.171: <witness> = lookup_impl_witness %R.7b2, @Iterate [symbolic]
// CHECK:STDOUT:   %Iterate.facet.523: %Iterate.type = facet_value %R.as_type.657, (%Iterate.lookup_impl_witness.171) [symbolic]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.51e: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.523) [symbolic]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.e1d: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.523) [symbolic]
// CHECK:STDOUT:   %.250: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.51e, %Iterate.facet.523 [symbolic]
// CHECK:STDOUT:   %impl.elem2.9ef: %.250 = impl_witness_access %Iterate.lookup_impl_witness.171, element2 [symbolic]
// CHECK:STDOUT:   %impl.elem1.f12: %Destroy.type = impl_witness_access %Iterate.lookup_impl_witness.171, element1 [symbolic]
// CHECK:STDOUT:   %as_type.6f2: type = facet_access_type %impl.elem1.f12 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.5ff: <specific function> = specific_impl_function %impl.elem2.9ef, @Iterate.WithSelf.NewCursor(%Iterate.facet.523) [symbolic]
// CHECK:STDOUT:   %ptr.b94: type = ptr_type %as_type.6f2 [symbolic]
// CHECK:STDOUT:   %.f81: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.e1d, %Iterate.facet.523 [symbolic]
// CHECK:STDOUT:   %impl.elem3.04c: %.f81 = impl_witness_access %Iterate.lookup_impl_witness.171, element3 [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.541: type = class_type @DefaultOptionalStorage, @DefaultOptionalStorage(%facet_value.209) [concrete]
// CHECK:STDOUT:   %OptionalStorage.impl_witness.fd3: <witness> = impl_witness imports.%OptionalStorage.impl_witness_table.655, @T.as_type.as.OptionalStorage.impl(%facet_value.209) [concrete]
// CHECK:STDOUT:   %OptionalStorage.facet.cc1: %OptionalStorage.type = facet_value %f64.dc1, (%OptionalStorage.impl_witness.fd3) [concrete]
// CHECK:STDOUT:   %Optional.f48: type = class_type @Optional, @Optional(%OptionalStorage.facet.cc1) [concrete]
// CHECK:STDOUT:   %specific_impl_fn.c5b: <specific function> = specific_impl_function %impl.elem3.04c, @Iterate.WithSelf.Next(%Iterate.facet.523) [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.type.d4e: type = fn_type @Optional.HasValue, @Optional(%OptionalStorage.facet.cc1) [concrete]
// CHECK:STDOUT:   %Optional.HasValue.e79: %Optional.HasValue.type.d4e = struct_value () [concrete]
// CHECK:STDOUT:   %Optional.Get.type.822: type = fn_type @Optional.Get, @Optional(%OptionalStorage.facet.cc1) [concrete]
// CHECK:STDOUT:   %Optional.Get.f05: %Optional.Get.type.822 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.facet.339: %Destroy.type = facet_value %f64.dc1, (%custom_witness.df9cc1.2) [concrete]
// CHECK:STDOUT:   %MaybeUnformed.b7e: type = class_type @MaybeUnformed, @MaybeUnformed(%Destroy.facet.339) [concrete]
// CHECK:STDOUT:   %.cc0: type = maybe_unformed_type %f64.dc1 [concrete]
// CHECK:STDOUT:   %struct_type.value.has_value.ad9: type = struct_type {.value: %MaybeUnformed.b7e, .has_value: bool} [concrete]
// CHECK:STDOUT:   %Optional.HasValue.specific_fn: <specific function> = specific_function %Optional.HasValue.e79, @Optional.HasValue(%OptionalStorage.facet.cc1) [concrete]
// CHECK:STDOUT:   %Optional.Get.specific_fn: <specific function> = specific_function %Optional.Get.f05, @Optional.Get(%OptionalStorage.facet.cc1) [concrete]
// CHECK:STDOUT:   %AddAssignWith.type.ff9: type = facet_type <@AddAssignWith, @AddAssignWith(%f64.dc1)> [concrete]
// CHECK:STDOUT:   %Float.as.AddAssignWith.impl.Op.type.fdd: type = fn_type @Float.as.AddAssignWith.impl.Op, @Float.as.AddAssignWith.impl(%N.fe9) [symbolic]
// CHECK:STDOUT:   %Float.as.AddAssignWith.impl.Op.430: %Float.as.AddAssignWith.impl.Op.type.fdd = struct_value () [symbolic]
// CHECK:STDOUT:   %AddAssignWith.impl_witness.e85: <witness> = impl_witness imports.%AddAssignWith.impl_witness_table, @Float.as.AddAssignWith.impl(%int_64) [concrete]
// CHECK:STDOUT:   %Float.as.AddAssignWith.impl.Op.type.a92: type = fn_type @Float.as.AddAssignWith.impl.Op, @Float.as.AddAssignWith.impl(%int_64) [concrete]
// CHECK:STDOUT:   %Float.as.AddAssignWith.impl.Op.ac1: %Float.as.AddAssignWith.impl.Op.type.a92 = struct_value () [concrete]
// CHECK:STDOUT:   %AddAssignWith.facet: %AddAssignWith.type.ff9 = facet_value %f64.dc1, (%AddAssignWith.impl_witness.e85) [concrete]
// CHECK:STDOUT:   %AddAssignWith.WithSelf.Op.type.f8a: type = fn_type @AddAssignWith.WithSelf.Op, @AddAssignWith.WithSelf(%f64.dc1, %AddAssignWith.facet) [concrete]
// CHECK:STDOUT:   %.746: type = fn_type_with_self_type %AddAssignWith.WithSelf.Op.type.f8a, %AddAssignWith.facet [concrete]
// CHECK:STDOUT:   %Float.as.AddAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Float.as.AddAssignWith.impl.Op.ac1, @Float.as.AddAssignWith.impl.Op(%int_64) [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.9: type = fn_type @Destroy.Op.loc24_19.6 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.9: %Destroy.Op.type.1d8f74.9 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.f8f: <witness> = lookup_impl_witness %impl.elem1.f12, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.b9c: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%impl.elem1.f12) [symbolic]
// CHECK:STDOUT:   %.75c: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.b9c, %impl.elem1.f12 [symbolic]
// CHECK:STDOUT:   %impl.elem0.688: %.75c = impl_witness_access %Destroy.lookup_impl_witness.f8f, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.1e0: <specific function> = specific_impl_function %impl.elem0.688, @Destroy.WithSelf.Op(%impl.elem1.f12) [symbolic]
// CHECK:STDOUT:   %ConstRange: type = class_type @ConstRange [concrete]
// CHECK:STDOUT:   %pattern_type.025: type = pattern_type %ConstRange [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %Iterator: type = class_type @Iterator.1 [concrete]
// CHECK:STDOUT:   %Begin.type: type = fn_type @Begin [concrete]
// CHECK:STDOUT:   %Begin: %Begin.type = struct_value () [concrete]
// CHECK:STDOUT:   %End.type: type = fn_type @End [concrete]
// CHECK:STDOUT:   %End: %End.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.5d5: <witness> = custom_witness (%Iterator, %Iterator, %Begin, %End), @CppRangeForIterate [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.ac7e09.1: type = fn_type @Iterator.Op.1 [concrete]
// CHECK:STDOUT:   %Iterator.Op.583fda.1: %Iterator.Op.type.ac7e09.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.994: <witness> = custom_witness (%f64.dc1, %Iterator.Op.583fda.1), @CppUnsafeDeref [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.ac7e09.2: type = fn_type @Iterator.Op.2 [concrete]
// CHECK:STDOUT:   %Iterator.Op.583fda.2: %Iterator.Op.type.ac7e09.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.213: <witness> = custom_witness (%Iterator.Op.583fda.2), @Destroy [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.ac7e09.3: type = fn_type @Iterator.Op.3 [concrete]
// CHECK:STDOUT:   %Iterator.Op.583fda.3: %Iterator.Op.type.ac7e09.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.a31: <witness> = custom_witness (%Iterator.Op.583fda.3), @Copy [concrete]
// CHECK:STDOUT:   %Equal.type: type = fn_type @Equal [concrete]
// CHECK:STDOUT:   %Equal: %Equal.type = struct_value () [concrete]
// CHECK:STDOUT:   %NotEqual.type: type = fn_type @NotEqual [concrete]
// CHECK:STDOUT:   %NotEqual: %NotEqual.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.821: <witness> = custom_witness (%Equal, %NotEqual), @EqWith, @EqWith(%Iterator) [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.ac7e09.4: type = fn_type @Iterator.Op.4 [concrete]
// CHECK:STDOUT:   %Iterator.Op.583fda.4: %Iterator.Op.type.ac7e09.4 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.393: <witness> = custom_witness (%Iterator.Op.583fda.4), @Inc [concrete]
// CHECK:STDOUT:   %facet_value.c05: %CppRangeForIterate_where.type = facet_value %ConstRange, (%custom_witness.df9cc1.2, %Copy.impl_witness.873, %custom_witness.5d5, %custom_witness.213, %custom_witness.a31, %custom_witness.994, %custom_witness.821, %custom_witness.393) [concrete]
// CHECK:STDOUT:   %tuple.type.005: type = tuple_type (%Iterator, %Iterator) [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.10: type = fn_type @Destroy.Op.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.10: %Destroy.Op.type.1d8f74.10 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.df9cc1.10: <witness> = custom_witness (%Destroy.Op.1a2547.10), @Destroy [concrete]
// CHECK:STDOUT:   %Destroy.facet.40b: %Destroy.type = facet_value %tuple.type.005, (%custom_witness.df9cc1.10) [concrete]
// CHECK:STDOUT:   %Iterate.impl_witness.1f0: <witness> = impl_witness imports.%Iterate.impl_witness_table.344, @R.as_type.as.Iterate.impl(%facet_value.c05) [concrete]
// CHECK:STDOUT:   %R.as_type.as.Iterate.impl.NewCursor.type.315: type = fn_type @R.as_type.as.Iterate.impl.NewCursor, @R.as_type.as.Iterate.impl(%facet_value.c05) [concrete]
// CHECK:STDOUT:   %R.as_type.as.Iterate.impl.NewCursor.d2e: %R.as_type.as.Iterate.impl.NewCursor.type.315 = struct_value () [concrete]
// CHECK:STDOUT:   %R.as_type.as.Iterate.impl.Next.type.c69: type = fn_type @R.as_type.as.Iterate.impl.Next, @R.as_type.as.Iterate.impl(%facet_value.c05) [concrete]
// CHECK:STDOUT:   %R.as_type.as.Iterate.impl.Next.0b7: %R.as_type.as.Iterate.impl.Next.type.c69 = struct_value () [concrete]
// CHECK:STDOUT:   %Iterate.facet.9f4: %Iterate.type = facet_value %ConstRange, (%Iterate.impl_witness.1f0) [concrete]
// CHECK:STDOUT:   %facet_value.9d0: %Iterate_where.type.83b = facet_value %ConstRange, (%Iterate.impl_witness.1f0) [concrete]
// CHECK:STDOUT:   %r.param_patt.a73: %pattern_type.025 = value_param_pattern [concrete]
// CHECK:STDOUT:   %r.patt.0e2: %pattern_type.025 = at_binding_pattern r, %r.param_patt.a73 [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.d89: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.9f4) [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.64f: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.9f4) [concrete]
// CHECK:STDOUT:   %.e0f: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.d89, %Iterate.facet.9f4 [concrete]
// CHECK:STDOUT:   %R.as_type.as.Iterate.impl.NewCursor.specific_fn: <specific function> = specific_function %R.as_type.as.Iterate.impl.NewCursor.d2e, @R.as_type.as.Iterate.impl.NewCursor(%facet_value.c05) [concrete]
// CHECK:STDOUT:   %ptr.d1a: type = ptr_type %tuple.type.005 [concrete]
// CHECK:STDOUT:   %.50c: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.64f, %Iterate.facet.9f4 [concrete]
// CHECK:STDOUT:   %R.as_type.as.Iterate.impl.Next.specific_fn: <specific function> = specific_function %R.as_type.as.Iterate.impl.Next.0b7, @R.as_type.as.Iterate.impl.Next(%facet_value.c05) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.117: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.40b) [concrete]
// CHECK:STDOUT:   %.5a4: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.117, %Destroy.facet.40b [concrete]
// CHECK:STDOUT:   %complete_type.320: <witness> = complete_type_witness %tuple.type.005 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.bb6: @Optional.%Optional.HasValue.type (%Optional.HasValue.type.e9e) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.HasValue (constants.%Optional.HasValue.6cd)]
// CHECK:STDOUT:   %Core.import_ref.daba: @Optional.%Optional.Get.type (%Optional.Get.type.bb9) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.Get (constants.%Optional.Get.fa1)]
// CHECK:STDOUT:   %Core.import_ref.0c6: type = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%DefaultOptionalStorage (constants.%DefaultOptionalStorage.d50)]
// CHECK:STDOUT:   %Core.import_ref.f4f: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.None.type (%T.as_type.as.OptionalStorage.impl.None.type.300) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.None (constants.%T.as_type.as.OptionalStorage.impl.None.f54)]
// CHECK:STDOUT:   %Core.import_ref.18f: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Some.type (%T.as_type.as.OptionalStorage.impl.Some.type.800) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Some (constants.%T.as_type.as.OptionalStorage.impl.Some.e83)]
// CHECK:STDOUT:   %Core.import_ref.c688: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Has.type (%T.as_type.as.OptionalStorage.impl.Has.type.4bc) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Has (constants.%T.as_type.as.OptionalStorage.impl.Has.dc2)]
// CHECK:STDOUT:   %Core.import_ref.a83: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Get.type (%T.as_type.as.OptionalStorage.impl.Get.type.029) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Get (constants.%T.as_type.as.OptionalStorage.impl.Get.cb5)]
// CHECK:STDOUT:   %Core.import_ref.aaa3 = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], unloaded
// CHECK:STDOUT:   %OptionalStorage.impl_witness_table.655 = impl_witness_table (%Core.import_ref.0c6, %Core.import_ref.f4f, %Core.import_ref.18f, %Core.import_ref.c688, %Core.import_ref.a83, %Core.import_ref.aaa3), @T.as_type.as.OptionalStorage.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.e69: @Float.as.Copy.impl.%Float.as.Copy.impl.Op.type (%Float.as.Copy.impl.Op.type.ff3) = import_ref Core//prelude/types/float, loc{{\d+_\d+}}, loaded [symbolic = @Float.as.Copy.impl.%Float.as.Copy.impl.Op (constants.%Float.as.Copy.impl.Op.d93)]
// CHECK:STDOUT:   %Copy.impl_witness_table.4e8 = impl_witness_table (%Core.import_ref.e69), @Float.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.301: %facet_type = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @R.as_type.as.Iterate.impl.%facet_value (constants.%facet_value.194)]
// CHECK:STDOUT:   %Core.import_ref.89f: %Destroy.type = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @R.as_type.as.Iterate.impl.%Destroy.facet (constants.%Destroy.facet.593)]
// CHECK:STDOUT:   %Core.import_ref.3e6: @R.as_type.as.Iterate.impl.%R.as_type.as.Iterate.impl.NewCursor.type (%R.as_type.as.Iterate.impl.NewCursor.type.36f) = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @R.as_type.as.Iterate.impl.%R.as_type.as.Iterate.impl.NewCursor (constants.%R.as_type.as.Iterate.impl.NewCursor.29b)]
// CHECK:STDOUT:   %Core.import_ref.85d: @R.as_type.as.Iterate.impl.%R.as_type.as.Iterate.impl.Next.type (%R.as_type.as.Iterate.impl.Next.type.1f7) = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @R.as_type.as.Iterate.impl.%R.as_type.as.Iterate.impl.Next (constants.%R.as_type.as.Iterate.impl.Next.0d6)]
// CHECK:STDOUT:   %Iterate.impl_witness_table.344 = impl_witness_table (%Core.import_ref.301, %Core.import_ref.89f, %Core.import_ref.3e6, %Core.import_ref.85d), @R.as_type.as.Iterate.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.cb3: @Float.as.AddAssignWith.impl.%Float.as.AddAssignWith.impl.Op.type (%Float.as.AddAssignWith.impl.Op.type.fdd) = import_ref Core//prelude/types/float, loc{{\d+_\d+}}, loaded [symbolic = @Float.as.AddAssignWith.impl.%Float.as.AddAssignWith.impl.Op (constants.%Float.as.AddAssignWith.impl.Op.430)]
// CHECK:STDOUT:   %AddAssignWith.impl_witness_table = impl_witness_table (%Core.import_ref.cb3), @Float.as.AddAssignWith.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @TestIterate(%R.loc20_17.2: %Iterate_where.type.83b) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Iterate.lookup_impl_witness: <witness> = lookup_impl_witness %R.loc20_17.1, @Iterate [symbolic = %Iterate.lookup_impl_witness (constants.%Iterate.lookup_impl_witness.171)]
// CHECK:STDOUT:   %Iterate.facet.loc24_19.5: %Iterate.type = facet_value %R.as_type.loc20_62.1, (%Iterate.lookup_impl_witness) [symbolic = %Iterate.facet.loc24_19.5 (constants.%Iterate.facet.523)]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.loc24_19.5) [symbolic = %Iterate.WithSelf.NewCursor.type (constants.%Iterate.WithSelf.NewCursor.type.51e)]
// CHECK:STDOUT:   %.loc24_19.19: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type, %Iterate.facet.loc24_19.5 [symbolic = %.loc24_19.19 (constants.%.250)]
// CHECK:STDOUT:   %impl.elem2.loc24_19.2: @TestIterate.%.loc24_19.19 (%.250) = impl_witness_access %Iterate.lookup_impl_witness, element2 [symbolic = %impl.elem2.loc24_19.2 (constants.%impl.elem2.9ef)]
// CHECK:STDOUT:   %specific_impl_fn.loc24_19.4: <specific function> = specific_impl_function %impl.elem2.loc24_19.2, @Iterate.WithSelf.NewCursor(%Iterate.facet.loc24_19.5) [symbolic = %specific_impl_fn.loc24_19.4 (constants.%specific_impl_fn.5ff)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %ptr: type = ptr_type %as_type [symbolic = %ptr (constants.%ptr.b94)]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.loc24_19.5) [symbolic = %Iterate.WithSelf.Next.type (constants.%Iterate.WithSelf.Next.type.e1d)]
// CHECK:STDOUT:   %.loc24_19.20: type = fn_type_with_self_type %Iterate.WithSelf.Next.type, %Iterate.facet.loc24_19.5 [symbolic = %.loc24_19.20 (constants.%.f81)]
// CHECK:STDOUT:   %impl.elem3.loc24_19.2: @TestIterate.%.loc24_19.20 (%.f81) = impl_witness_access %Iterate.lookup_impl_witness, element3 [symbolic = %impl.elem3.loc24_19.2 (constants.%impl.elem3.04c)]
// CHECK:STDOUT:   %specific_impl_fn.loc24_19.5: <specific function> = specific_impl_function %impl.elem3.loc24_19.2, @Iterate.WithSelf.Next(%Iterate.facet.loc24_19.5) [symbolic = %specific_impl_fn.loc24_19.5 (constants.%specific_impl_fn.c5b)]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%impl.elem1) [symbolic = %Destroy.WithSelf.Op.type (constants.%Destroy.WithSelf.Op.type.b9c)]
// CHECK:STDOUT:   %.loc24_19.21: type = fn_type_with_self_type %Destroy.WithSelf.Op.type, %impl.elem1 [symbolic = %.loc24_19.21 (constants.%.75c)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %impl.elem1, @Destroy [symbolic = %Destroy.lookup_impl_witness (constants.%Destroy.lookup_impl_witness.f8f)]
// CHECK:STDOUT:   %impl.elem0.loc24_19.2: @TestIterate.%.loc24_19.21 (%.75c) = impl_witness_access %Destroy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc24_19.2 (constants.%impl.elem0.688)]
// CHECK:STDOUT:   %specific_impl_fn.loc24_19.6: <specific function> = specific_impl_function %impl.elem0.loc24_19.2, @Destroy.WithSelf.Op(%impl.elem1) [symbolic = %specific_impl_fn.loc24_19.6 (constants.%specific_impl_fn.1e0)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%r.param: @TestIterate.%R.as_type.loc20_62.1 (%R.as_type.657)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %i.patt: %pattern_type.fb7 = value_binding_pattern i [concrete = constants.%i.patt]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %r.ref: @TestIterate.%R.as_type.loc20_62.1 (%R.as_type.657) = name_ref r, %r
// CHECK:STDOUT:     %impl.elem2.loc24_19.1: @TestIterate.%.loc24_19.19 (%.250) = impl_witness_access constants.%Iterate.lookup_impl_witness.171, element2 [symbolic = %impl.elem2.loc24_19.2 (constants.%impl.elem2.9ef)]
// CHECK:STDOUT:     %bound_method.loc24_19.1: <bound method> = bound_method %r.ref, %impl.elem2.loc24_19.1
// CHECK:STDOUT:     %Iterate.facet.loc24_19.1: %Iterate.type = facet_value constants.%R.as_type.657, (constants.%Iterate.lookup_impl_witness.171) [symbolic = %Iterate.facet.loc24_19.5 (constants.%Iterate.facet.523)]
// CHECK:STDOUT:     %.loc24_19.1: %Iterate.type = converted constants.%R.as_type.657, %Iterate.facet.loc24_19.1 [symbolic = %Iterate.facet.loc24_19.5 (constants.%Iterate.facet.523)]
// CHECK:STDOUT:     %Iterate.facet.loc24_19.2: %Iterate.type = facet_value constants.%R.as_type.657, (constants.%Iterate.lookup_impl_witness.171) [symbolic = %Iterate.facet.loc24_19.5 (constants.%Iterate.facet.523)]
// CHECK:STDOUT:     %.loc24_19.2: %Iterate.type = converted constants.%R.as_type.657, %Iterate.facet.loc24_19.2 [symbolic = %Iterate.facet.loc24_19.5 (constants.%Iterate.facet.523)]
// CHECK:STDOUT:     %specific_impl_fn.loc24_19.1: <specific function> = specific_impl_function %impl.elem2.loc24_19.1, @Iterate.WithSelf.NewCursor(constants.%Iterate.facet.523) [symbolic = %specific_impl_fn.loc24_19.4 (constants.%specific_impl_fn.5ff)]
// CHECK:STDOUT:     %bound_method.loc24_19.2: <bound method> = bound_method %r.ref, %specific_impl_fn.loc24_19.1
// CHECK:STDOUT:     %var: ref @TestIterate.%as_type (%as_type.6f2) = var invalid
// CHECK:STDOUT:     %Iterate.WithSelf.NewCursor.call: init @TestIterate.%as_type (%as_type.6f2) to %var = call %bound_method.loc24_19.2(%r.ref)
// CHECK:STDOUT:     assign %var, %Iterate.WithSelf.NewCursor.call
// CHECK:STDOUT:     br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.next:
// CHECK:STDOUT:     %addr: @TestIterate.%ptr (%ptr.b94) = addr_of %var
// CHECK:STDOUT:     %impl.elem3.loc24_19.1: @TestIterate.%.loc24_19.20 (%.f81) = impl_witness_access constants.%Iterate.lookup_impl_witness.171, element3 [symbolic = %impl.elem3.loc24_19.2 (constants.%impl.elem3.04c)]
// CHECK:STDOUT:     %bound_method.loc24_19.3: <bound method> = bound_method %r.ref, %impl.elem3.loc24_19.1
// CHECK:STDOUT:     %Iterate.facet.loc24_19.3: %Iterate.type = facet_value constants.%R.as_type.657, (constants.%Iterate.lookup_impl_witness.171) [symbolic = %Iterate.facet.loc24_19.5 (constants.%Iterate.facet.523)]
// CHECK:STDOUT:     %.loc24_19.3: %Iterate.type = converted constants.%R.as_type.657, %Iterate.facet.loc24_19.3 [symbolic = %Iterate.facet.loc24_19.5 (constants.%Iterate.facet.523)]
// CHECK:STDOUT:     %Iterate.facet.loc24_19.4: %Iterate.type = facet_value constants.%R.as_type.657, (constants.%Iterate.lookup_impl_witness.171) [symbolic = %Iterate.facet.loc24_19.5 (constants.%Iterate.facet.523)]
// CHECK:STDOUT:     %.loc24_19.4: %Iterate.type = converted constants.%R.as_type.657, %Iterate.facet.loc24_19.4 [symbolic = %Iterate.facet.loc24_19.5 (constants.%Iterate.facet.523)]
// CHECK:STDOUT:     %.loc24_19.5: %Destroy.type = converted constants.%as_type.6f2, constants.%impl.elem1.f12 [symbolic = %impl.elem1 (constants.%impl.elem1.f12)]
// CHECK:STDOUT:     %.loc24_19.6: %Destroy.type = converted constants.%as_type.6f2, constants.%impl.elem1.f12 [symbolic = %impl.elem1 (constants.%impl.elem1.f12)]
// CHECK:STDOUT:     %specific_impl_fn.loc24_19.2: <specific function> = specific_impl_function %impl.elem3.loc24_19.1, @Iterate.WithSelf.Next(constants.%Iterate.facet.523) [symbolic = %specific_impl_fn.loc24_19.5 (constants.%specific_impl_fn.c5b)]
// CHECK:STDOUT:     %bound_method.loc24_19.4: <bound method> = bound_method %r.ref, %specific_impl_fn.loc24_19.2
// CHECK:STDOUT:     %.loc24_19.7: ref %Optional.f48 = temporary_storage
// CHECK:STDOUT:     %Iterate.WithSelf.Next.call: init %Optional.f48 to %.loc24_19.7 = call %bound_method.loc24_19.4(%r.ref, %addr)
// CHECK:STDOUT:     %.loc24_19.8: ref %Optional.f48 = temporary %.loc24_19.7, %Iterate.WithSelf.Next.call
// CHECK:STDOUT:     %.loc24_19.9: %Optional.HasValue.type.d4e = specific_constant imports.%Core.import_ref.bb6, @Optional(constants.%OptionalStorage.facet.cc1) [concrete = constants.%Optional.HasValue.e79]
// CHECK:STDOUT:     %HasValue.ref: %Optional.HasValue.type.d4e = name_ref HasValue, %.loc24_19.9 [concrete = constants.%Optional.HasValue.e79]
// CHECK:STDOUT:     %Optional.HasValue.bound: <bound method> = bound_method %.loc24_19.8, %HasValue.ref
// CHECK:STDOUT:     %Optional.HasValue.specific_fn: <specific function> = specific_function %HasValue.ref, @Optional.HasValue(constants.%OptionalStorage.facet.cc1) [concrete = constants.%Optional.HasValue.specific_fn]
// CHECK:STDOUT:     %bound_method.loc24_19.5: <bound method> = bound_method %.loc24_19.8, %Optional.HasValue.specific_fn
// CHECK:STDOUT:     %.loc24_19.10: %Optional.f48 = acquire_value %.loc24_19.8
// CHECK:STDOUT:     %Optional.HasValue.call: init bool = call %bound_method.loc24_19.5(%.loc24_19.10)
// CHECK:STDOUT:     %.loc24_19.11: bool = value_of_initializer %Optional.HasValue.call
// CHECK:STDOUT:     %.loc24_19.12: bool = converted %Optional.HasValue.call, %.loc24_19.11
// CHECK:STDOUT:     if %.loc24_19.12 br !for.body else br !for.done
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.body:
// CHECK:STDOUT:     %.loc24_19.13: %Optional.Get.type.822 = specific_constant imports.%Core.import_ref.daba, @Optional(constants.%OptionalStorage.facet.cc1) [concrete = constants.%Optional.Get.f05]
// CHECK:STDOUT:     %Get.ref: %Optional.Get.type.822 = name_ref Get, %.loc24_19.13 [concrete = constants.%Optional.Get.f05]
// CHECK:STDOUT:     %Optional.Get.bound: <bound method> = bound_method %.loc24_19.8, %Get.ref
// CHECK:STDOUT:     %Optional.Get.specific_fn: <specific function> = specific_function %Get.ref, @Optional.Get(constants.%OptionalStorage.facet.cc1) [concrete = constants.%Optional.Get.specific_fn]
// CHECK:STDOUT:     %bound_method.loc24_19.6: <bound method> = bound_method %.loc24_19.8, %Optional.Get.specific_fn
// CHECK:STDOUT:     %.loc24_19.14: %Optional.f48 = acquire_value %.loc24_19.8
// CHECK:STDOUT:     %Optional.Get.call: init %f64.dc1 = call %bound_method.loc24_19.6(%.loc24_19.14)
// CHECK:STDOUT:     %.loc24_19.15: %f64.dc1 = value_of_initializer %Optional.Get.call
// CHECK:STDOUT:     %.loc24_19.16: %f64.dc1 = converted %Optional.Get.call, %.loc24_19.15
// CHECK:STDOUT:     %f64.loc24: type = type_literal constants.%f64.dc1 [concrete = constants.%f64.dc1]
// CHECK:STDOUT:     %i: %f64.dc1 = wrapper_binding i, %.loc24_19.16
// CHECK:STDOUT:     %sum.ref: ref %f64.dc1 = name_ref sum, %sum
// CHECK:STDOUT:     %i.ref: %f64.dc1 = name_ref i, %i
// CHECK:STDOUT:     %impl.elem0.loc25: %.746 = impl_witness_access constants.%AddAssignWith.impl_witness.e85, element0 [concrete = constants.%Float.as.AddAssignWith.impl.Op.ac1]
// CHECK:STDOUT:     %bound_method.loc25_9.1: <bound method> = bound_method %sum.ref, %impl.elem0.loc25
// CHECK:STDOUT:     %specific_fn.loc25: <specific function> = specific_function %impl.elem0.loc25, @Float.as.AddAssignWith.impl.Op(constants.%int_64) [concrete = constants.%Float.as.AddAssignWith.impl.Op.specific_fn]
// CHECK:STDOUT:     %bound_method.loc25_9.2: <bound method> = bound_method %sum.ref, %specific_fn.loc25
// CHECK:STDOUT:     %Float.as.AddAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc25_9.2(%sum.ref, %i.ref)
// CHECK:STDOUT:     br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.done:
// CHECK:STDOUT:     %Destroy.Op.bound.loc24: <bound method> = bound_method %.loc24_19.8, constants.%Destroy.Op.1a2547.9
// CHECK:STDOUT:     %Destroy.Op.call.loc24: init %empty_tuple.type = call %Destroy.Op.bound.loc24(%.loc24_19.8)
// CHECK:STDOUT:     %impl.elem0.loc24_19.1: @TestIterate.%.loc24_19.21 (%.75c) = impl_witness_access constants.%Destroy.lookup_impl_witness.f8f, element0 [symbolic = %impl.elem0.loc24_19.2 (constants.%impl.elem0.688)]
// CHECK:STDOUT:     %bound_method.loc24_19.7: <bound method> = bound_method %var, %impl.elem0.loc24_19.1
// CHECK:STDOUT:     %.loc24_19.17: %Destroy.type = converted constants.%as_type.6f2, constants.%impl.elem1.f12 [symbolic = %impl.elem1 (constants.%impl.elem1.f12)]
// CHECK:STDOUT:     %.loc24_19.18: %Destroy.type = converted constants.%as_type.6f2, constants.%impl.elem1.f12 [symbolic = %impl.elem1 (constants.%impl.elem1.f12)]
// CHECK:STDOUT:     %specific_impl_fn.loc24_19.3: <specific function> = specific_impl_function %impl.elem0.loc24_19.1, @Destroy.WithSelf.Op(constants.%impl.elem1.f12) [symbolic = %specific_impl_fn.loc24_19.6 (constants.%specific_impl_fn.1e0)]
// CHECK:STDOUT:     %bound_method.loc24_19.8: <bound method> = bound_method %var, %specific_impl_fn.loc24_19.3
// CHECK:STDOUT:     %Destroy.WithSelf.Op.call: init %empty_tuple.type = call %bound_method.loc24_19.8(%var)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc24_19.1(%self.param: ref %.cc0) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc24_19.2(%self.param: ref %MaybeUnformed.b7e) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc24_19.3(%self.param: ref bool) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc24_19.4(%self.param: ref %struct_type.value.has_value.ad9) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc24_19.5(%self.param: ref %DefaultOptionalStorage.541) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc24_19.6(%self.param: ref %Optional.f48) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestRangeFor(%c.param: %ConstRange) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %i.patt: %pattern_type.fb7 = value_binding_pattern i [concrete = constants.%i.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c.ref: %ConstRange = name_ref c, %c
// CHECK:STDOUT:   %impl.elem2: %.e0f = impl_witness_access constants.%Iterate.impl_witness.1f0, element2 [concrete = constants.%R.as_type.as.Iterate.impl.NewCursor.d2e]
// CHECK:STDOUT:   %bound_method.loc38_19.1: <bound method> = bound_method %c.ref, %impl.elem2
// CHECK:STDOUT:   %facet_value.loc38_19.1: %CppRangeForIterate_where.type = facet_value constants.%ConstRange, (constants.%custom_witness.df9cc1.2, constants.%Copy.impl_witness.873, constants.%custom_witness.5d5, constants.%custom_witness.213, constants.%custom_witness.a31, constants.%custom_witness.994, constants.%custom_witness.821, constants.%custom_witness.393) [concrete = constants.%facet_value.c05]
// CHECK:STDOUT:   %.loc38_19.1: %CppRangeForIterate_where.type = converted constants.%ConstRange, %facet_value.loc38_19.1 [concrete = constants.%facet_value.c05]
// CHECK:STDOUT:   %facet_value.loc38_19.2: %CppRangeForIterate_where.type = facet_value constants.%ConstRange, (constants.%custom_witness.df9cc1.2, constants.%Copy.impl_witness.873, constants.%custom_witness.5d5, constants.%custom_witness.213, constants.%custom_witness.a31, constants.%custom_witness.994, constants.%custom_witness.821, constants.%custom_witness.393) [concrete = constants.%facet_value.c05]
// CHECK:STDOUT:   %.loc38_19.2: %CppRangeForIterate_where.type = converted constants.%ConstRange, %facet_value.loc38_19.2 [concrete = constants.%facet_value.c05]
// CHECK:STDOUT:   %specific_fn.loc38_19.1: <specific function> = specific_function %impl.elem2, @R.as_type.as.Iterate.impl.NewCursor(constants.%facet_value.c05) [concrete = constants.%R.as_type.as.Iterate.impl.NewCursor.specific_fn]
// CHECK:STDOUT:   %bound_method.loc38_19.2: <bound method> = bound_method %c.ref, %specific_fn.loc38_19.1
// CHECK:STDOUT:   %var: ref %tuple.type.005 = var invalid
// CHECK:STDOUT:   %R.as_type.as.Iterate.impl.NewCursor.call: init %tuple.type.005 to %var = call %bound_method.loc38_19.2(%c.ref)
// CHECK:STDOUT:   assign %var, %R.as_type.as.Iterate.impl.NewCursor.call
// CHECK:STDOUT:   br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT: !for.next:
// CHECK:STDOUT:   %addr: %ptr.d1a = addr_of %var
// CHECK:STDOUT:   %impl.elem3: %.50c = impl_witness_access constants.%Iterate.impl_witness.1f0, element3 [concrete = constants.%R.as_type.as.Iterate.impl.Next.0b7]
// CHECK:STDOUT:   %bound_method.loc38_19.3: <bound method> = bound_method %c.ref, %impl.elem3
// CHECK:STDOUT:   %facet_value.loc38_19.3: %CppRangeForIterate_where.type = facet_value constants.%ConstRange, (constants.%custom_witness.df9cc1.2, constants.%Copy.impl_witness.873, constants.%custom_witness.5d5, constants.%custom_witness.213, constants.%custom_witness.a31, constants.%custom_witness.994, constants.%custom_witness.821, constants.%custom_witness.393) [concrete = constants.%facet_value.c05]
// CHECK:STDOUT:   %.loc38_19.3: %CppRangeForIterate_where.type = converted constants.%ConstRange, %facet_value.loc38_19.3 [concrete = constants.%facet_value.c05]
// CHECK:STDOUT:   %facet_value.loc38_19.4: %CppRangeForIterate_where.type = facet_value constants.%ConstRange, (constants.%custom_witness.df9cc1.2, constants.%Copy.impl_witness.873, constants.%custom_witness.5d5, constants.%custom_witness.213, constants.%custom_witness.a31, constants.%custom_witness.994, constants.%custom_witness.821, constants.%custom_witness.393) [concrete = constants.%facet_value.c05]
// CHECK:STDOUT:   %.loc38_19.4: %CppRangeForIterate_where.type = converted constants.%ConstRange, %facet_value.loc38_19.4 [concrete = constants.%facet_value.c05]
// CHECK:STDOUT:   %specific_fn.loc38_19.2: <specific function> = specific_function %impl.elem3, @R.as_type.as.Iterate.impl.Next(constants.%facet_value.c05) [concrete = constants.%R.as_type.as.Iterate.impl.Next.specific_fn]
// CHECK:STDOUT:   %bound_method.loc38_19.4: <bound method> = bound_method %c.ref, %specific_fn.loc38_19.2
// CHECK:STDOUT:   %.loc38_19.5: ref %Optional.f48 = temporary_storage
// CHECK:STDOUT:   %R.as_type.as.Iterate.impl.Next.call: init %Optional.f48 to %.loc38_19.5 = call %bound_method.loc38_19.4(%c.ref, %addr)
// CHECK:STDOUT:   %.loc38_19.6: ref %Optional.f48 = temporary %.loc38_19.5, %R.as_type.as.Iterate.impl.Next.call
// CHECK:STDOUT:   %.loc38_19.7: %Optional.HasValue.type.d4e = specific_constant imports.%Core.import_ref.bb6, @Optional(constants.%OptionalStorage.facet.cc1) [concrete = constants.%Optional.HasValue.e79]
// CHECK:STDOUT:   %HasValue.ref: %Optional.HasValue.type.d4e = name_ref HasValue, %.loc38_19.7 [concrete = constants.%Optional.HasValue.e79]
// CHECK:STDOUT:   %Optional.HasValue.bound: <bound method> = bound_method %.loc38_19.6, %HasValue.ref
// CHECK:STDOUT:   %Optional.HasValue.specific_fn: <specific function> = specific_function %HasValue.ref, @Optional.HasValue(constants.%OptionalStorage.facet.cc1) [concrete = constants.%Optional.HasValue.specific_fn]
// CHECK:STDOUT:   %bound_method.loc38_19.5: <bound method> = bound_method %.loc38_19.6, %Optional.HasValue.specific_fn
// CHECK:STDOUT:   %.loc38_19.8: %Optional.f48 = acquire_value %.loc38_19.6
// CHECK:STDOUT:   %Optional.HasValue.call: init bool = call %bound_method.loc38_19.5(%.loc38_19.8)
// CHECK:STDOUT:   %.loc38_19.9: bool = value_of_initializer %Optional.HasValue.call
// CHECK:STDOUT:   %.loc38_19.10: bool = converted %Optional.HasValue.call, %.loc38_19.9
// CHECK:STDOUT:   if %.loc38_19.10 br !for.body else br !for.done
// CHECK:STDOUT:
// CHECK:STDOUT: !for.body:
// CHECK:STDOUT:   %.loc38_19.11: %Optional.Get.type.822 = specific_constant imports.%Core.import_ref.daba, @Optional(constants.%OptionalStorage.facet.cc1) [concrete = constants.%Optional.Get.f05]
// CHECK:STDOUT:   %Get.ref: %Optional.Get.type.822 = name_ref Get, %.loc38_19.11 [concrete = constants.%Optional.Get.f05]
// CHECK:STDOUT:   %Optional.Get.bound: <bound method> = bound_method %.loc38_19.6, %Get.ref
// CHECK:STDOUT:   %Optional.Get.specific_fn: <specific function> = specific_function %Get.ref, @Optional.Get(constants.%OptionalStorage.facet.cc1) [concrete = constants.%Optional.Get.specific_fn]
// CHECK:STDOUT:   %bound_method.loc38_19.6: <bound method> = bound_method %.loc38_19.6, %Optional.Get.specific_fn
// CHECK:STDOUT:   %.loc38_19.12: %Optional.f48 = acquire_value %.loc38_19.6
// CHECK:STDOUT:   %Optional.Get.call: init %f64.dc1 = call %bound_method.loc38_19.6(%.loc38_19.12)
// CHECK:STDOUT:   %.loc38_19.13: %f64.dc1 = value_of_initializer %Optional.Get.call
// CHECK:STDOUT:   %.loc38_19.14: %f64.dc1 = converted %Optional.Get.call, %.loc38_19.13
// CHECK:STDOUT:   %f64.loc38: type = type_literal constants.%f64.dc1 [concrete = constants.%f64.dc1]
// CHECK:STDOUT:   %i: %f64.dc1 = wrapper_binding i, %.loc38_19.14
// CHECK:STDOUT:   %sum.ref: ref %f64.dc1 = name_ref sum, %sum
// CHECK:STDOUT:   %i.ref: %f64.dc1 = name_ref i, %i
// CHECK:STDOUT:   %impl.elem0.loc39: %.746 = impl_witness_access constants.%AddAssignWith.impl_witness.e85, element0 [concrete = constants.%Float.as.AddAssignWith.impl.Op.ac1]
// CHECK:STDOUT:   %bound_method.loc39_9.1: <bound method> = bound_method %sum.ref, %impl.elem0.loc39
// CHECK:STDOUT:   %specific_fn.loc39: <specific function> = specific_function %impl.elem0.loc39, @Float.as.AddAssignWith.impl.Op(constants.%int_64) [concrete = constants.%Float.as.AddAssignWith.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc39_9.2: <bound method> = bound_method %sum.ref, %specific_fn.loc39
// CHECK:STDOUT:   %Float.as.AddAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc39_9.2(%sum.ref, %i.ref)
// CHECK:STDOUT:   br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT: !for.done:
// CHECK:STDOUT:   %Destroy.Op.bound.loc38_19.1: <bound method> = bound_method %.loc38_19.6, constants.%Destroy.Op.1a2547.9
// CHECK:STDOUT:   %Destroy.Op.call.loc38_19.1: init %empty_tuple.type = call %Destroy.Op.bound.loc38_19.1(%.loc38_19.6)
// CHECK:STDOUT:   %Destroy.Op.bound.loc38_19.2: <bound method> = bound_method %var, constants.%Destroy.Op.1a2547.10
// CHECK:STDOUT:   %Destroy.Op.call.loc38_19.2: init %empty_tuple.type = call %Destroy.Op.bound.loc38_19.2(%var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestIterate(constants.%R.7b2) {
// CHECK:STDOUT:   %R.patt.loc20_17.2 => constants.%R.patt.053
// CHECK:STDOUT:   %R.loc20_17.1 => constants.%R.7b2
// CHECK:STDOUT:   %R.as_type.loc20_62.1 => constants.%R.as_type.657
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.afe
// CHECK:STDOUT:   %r.param_patt.loc20_60.2 => constants.%r.param_patt.184
// CHECK:STDOUT:   %r.patt.loc20_60.2 => constants.%r.patt.727
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestIterate(constants.%facet_value.9d0) {
// CHECK:STDOUT:   %R.patt.loc20_17.2 => constants.%R.patt.053
// CHECK:STDOUT:   %R.loc20_17.1 => constants.%facet_value.9d0
// CHECK:STDOUT:   %R.as_type.loc20_62.1 => constants.%ConstRange
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.025
// CHECK:STDOUT:   %r.param_patt.loc20_60.2 => constants.%r.param_patt.a73
// CHECK:STDOUT:   %r.patt.loc20_60.2 => constants.%r.patt.0e2
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc20 => constants.%complete_type.357
// CHECK:STDOUT:   %Iterate.lookup_impl_witness => constants.%Iterate.impl_witness.1f0
// CHECK:STDOUT:   %Iterate.facet.loc24_19.5 => constants.%Iterate.facet.9f4
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type => constants.%Iterate.WithSelf.NewCursor.type.d89
// CHECK:STDOUT:   %.loc24_19.19 => constants.%.e0f
// CHECK:STDOUT:   %impl.elem2.loc24_19.2 => constants.%R.as_type.as.Iterate.impl.NewCursor.d2e
// CHECK:STDOUT:   %specific_impl_fn.loc24_19.4 => constants.%R.as_type.as.Iterate.impl.NewCursor.specific_fn
// CHECK:STDOUT:   %impl.elem1 => constants.%Destroy.facet.40b
// CHECK:STDOUT:   %as_type => constants.%tuple.type.005
// CHECK:STDOUT:   %require_complete.1 => constants.%complete_type.320
// CHECK:STDOUT:   %ptr => constants.%ptr.d1a
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type => constants.%Iterate.WithSelf.Next.type.64f
// CHECK:STDOUT:   %.loc24_19.20 => constants.%.50c
// CHECK:STDOUT:   %impl.elem3.loc24_19.2 => constants.%R.as_type.as.Iterate.impl.Next.0b7
// CHECK:STDOUT:   %specific_impl_fn.loc24_19.5 => constants.%R.as_type.as.Iterate.impl.Next.specific_fn
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.117
// CHECK:STDOUT:   %.loc24_19.21 => constants.%.5a4
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.df9cc1.10
// CHECK:STDOUT:   %impl.elem0.loc24_19.2 => constants.%Destroy.Op.1a2547.10
// CHECK:STDOUT:   %specific_impl_fn.loc24_19.6 => constants.%Destroy.Op.1a2547.10
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
// CHECK:STDOUT:   %T.220: %OptionalStorage.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Optional.Get.type.bb9: type = fn_type @Optional.Get, @Optional(%T.220) [symbolic]
// CHECK:STDOUT:   %Optional.Get.fa1: %Optional.Get.type.bb9 = struct_value () [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.type.e9e: type = fn_type @Optional.HasValue, @Optional(%T.220) [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.6cd: %Optional.HasValue.type.e9e = struct_value () [symbolic]
// CHECK:STDOUT:   %T.283: %facet_type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.d50: type = class_type @DefaultOptionalStorage, @DefaultOptionalStorage(%T.283) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Get.type.029: type = fn_type @T.as_type.as.OptionalStorage.impl.Get, @T.as_type.as.OptionalStorage.impl(%T.283) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Get.cb5: %T.as_type.as.OptionalStorage.impl.Get.type.029 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Has.type.4bc: type = fn_type @T.as_type.as.OptionalStorage.impl.Has, @T.as_type.as.OptionalStorage.impl(%T.283) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Has.dc2: %T.as_type.as.OptionalStorage.impl.Has.type.4bc = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Some.type.800: type = fn_type @T.as_type.as.OptionalStorage.impl.Some, @T.as_type.as.OptionalStorage.impl(%T.283) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Some.e83: %T.as_type.as.OptionalStorage.impl.Some.type.800 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.None.type.300: type = fn_type @T.as_type.as.OptionalStorage.impl.None, @T.as_type.as.OptionalStorage.impl(%T.283) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.None.f54: %T.as_type.as.OptionalStorage.impl.None.type.300 = struct_value () [symbolic]
// CHECK:STDOUT:   %.Self.35d: %Iterate.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %Iterate.lookup_impl_witness.b37: <witness> = lookup_impl_witness %.Self.35d, @Iterate [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.903: %facet_type = impl_witness_access %Iterate.lookup_impl_witness.b37, element0 [symbolic_self]
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete]
// CHECK:STDOUT:   %N.fe9: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %f64.dc1: type = class_type @Float, @Float(%int_64) [concrete]
// CHECK:STDOUT:   %pattern_type.fb7: type = pattern_type %f64.dc1 [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.2: type = fn_type @Destroy.Op.loc22_54.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.2: %Destroy.Op.type.1d8f74.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.df9cc1.2: <witness> = custom_witness (%Destroy.Op.1a2547.2), @Destroy [concrete]
// CHECK:STDOUT:   %Float.as.Copy.impl.Op.type.ff3: type = fn_type @Float.as.Copy.impl.Op, @Float.as.Copy.impl(%N.fe9) [symbolic]
// CHECK:STDOUT:   %Float.as.Copy.impl.Op.d93: %Float.as.Copy.impl.Op.type.ff3 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.873: <witness> = impl_witness imports.%Copy.impl_witness_table.4e8, @Float.as.Copy.impl(%int_64) [concrete]
// CHECK:STDOUT:   %facet_value.209: %facet_type = facet_value %f64.dc1, (%custom_witness.df9cc1.2, %Copy.impl_witness.873) [concrete]
// CHECK:STDOUT:   %Iterate_where.type.83b: type = facet_type <@Iterate where %impl.elem0.903 = %facet_value.209> [concrete]
// CHECK:STDOUT:   %pattern_type.b62: type = pattern_type %Iterate_where.type.83b [concrete]
// CHECK:STDOUT:   %R.patt.053: %pattern_type.b62 = symbolic_binding_pattern R, 0 [symbolic]
// CHECK:STDOUT:   %R.7b2: %Iterate_where.type.83b = symbolic_binding R, 0 [symbolic]
// CHECK:STDOUT:   %R.as_type.657: type = facet_access_type %R.7b2 [symbolic]
// CHECK:STDOUT:   %pattern_type.afe: type = pattern_type %R.as_type.657 [symbolic]
// CHECK:STDOUT:   %r.param_patt.184: %pattern_type.afe = value_param_pattern [symbolic]
// CHECK:STDOUT:   %r.patt.727: %pattern_type.afe = at_binding_pattern r, %r.param_patt.184 [symbolic]
// CHECK:STDOUT:   %i.patt: %pattern_type.fb7 = value_binding_pattern i [concrete]
// CHECK:STDOUT:   %CppRangeForIterate.type: type = facet_type <@CppRangeForIterate> [concrete]
// CHECK:STDOUT:   %.Self.6bd: %CppRangeForIterate.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %CppRangeForIterate.lookup_impl_witness.143: <witness> = lookup_impl_witness %.Self.6bd, @CppRangeForIterate [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.109: type = impl_witness_access %CppRangeForIterate.lookup_impl_witness.143, element0 [symbolic_self]
// CHECK:STDOUT:   %CppUnsafeDeref.lookup_impl_witness.712: <witness> = lookup_impl_witness %impl.elem0.109, @CppUnsafeDeref [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.56a: type = impl_witness_access %CppUnsafeDeref.lookup_impl_witness.712, element0 [symbolic_self]
// CHECK:STDOUT:   %impl.elem1.222: type = impl_witness_access %CppRangeForIterate.lookup_impl_witness.143, element1 [symbolic_self]
// CHECK:STDOUT:   %CppRangeForIterate_where.type: type = facet_type <@CppRangeForIterate where %impl.elem0.109 impls @Destroy and %impl.elem0.109 impls @Copy and %impl.elem0.109 impls @CppUnsafeDeref and %impl.elem0.109 impls @EqWith, @EqWith(%impl.elem1.222) and %impl.elem0.109 impls @Inc and %impl.elem0.56a impls @Destroy and %impl.elem0.56a impls @Copy and %impl.elem1.222 impls @Destroy and %impl.elem1.222 impls @Copy> [concrete]
// CHECK:STDOUT:   %R.39b: %CppRangeForIterate_where.type = symbolic_binding R, 0 [symbolic]
// CHECK:STDOUT:   %CppRangeForIterate.lookup_impl_witness.7e2712.2: <witness> = lookup_impl_witness %R.39b, @CppRangeForIterate [symbolic]
// CHECK:STDOUT:   %impl.elem1.7a0bff.2: type = impl_witness_access %CppRangeForIterate.lookup_impl_witness.7e2712.2, element1 [symbolic]
// CHECK:STDOUT:   %impl.elem0.73eb7f.2: type = impl_witness_access %CppRangeForIterate.lookup_impl_witness.7e2712.2, element0 [symbolic]
// CHECK:STDOUT:   %tuple.type.3cf: type = tuple_type (%impl.elem0.73eb7f.2, %impl.elem1.7a0bff.2) [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.69f: <witness> = lookup_impl_witness %tuple.type.3cf, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.593: %Destroy.type = facet_value %tuple.type.3cf, (%Destroy.lookup_impl_witness.69f) [symbolic]
// CHECK:STDOUT:   %CppUnsafeDeref.lookup_impl_witness.aa7: <witness> = lookup_impl_witness %impl.elem0.73eb7f.2, @CppUnsafeDeref [symbolic]
// CHECK:STDOUT:   %impl.elem0.af7: type = impl_witness_access %CppUnsafeDeref.lookup_impl_witness.aa7, element0 [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.ec3: <witness> = lookup_impl_witness %impl.elem0.af7, @Destroy [symbolic]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.a7a: <witness> = lookup_impl_witness %impl.elem0.af7, @Copy [symbolic]
// CHECK:STDOUT:   %facet_value.194: %facet_type = facet_value %impl.elem0.af7, (%Destroy.lookup_impl_witness.ec3, %Copy.lookup_impl_witness.a7a) [symbolic]
// CHECK:STDOUT:   %R.as_type.as.Iterate.impl.Next.type.1f7: type = fn_type @R.as_type.as.Iterate.impl.Next, @R.as_type.as.Iterate.impl(%R.39b) [symbolic]
// CHECK:STDOUT:   %R.as_type.as.Iterate.impl.Next.0d6: %R.as_type.as.Iterate.impl.Next.type.1f7 = struct_value () [symbolic]
// CHECK:STDOUT:   %R.as_type.as.Iterate.impl.NewCursor.type.36f: type = fn_type @R.as_type.as.Iterate.impl.NewCursor, @R.as_type.as.Iterate.impl(%R.39b) [symbolic]
// CHECK:STDOUT:   %R.as_type.as.Iterate.impl.NewCursor.29b: %R.as_type.as.Iterate.impl.NewCursor.type.36f = struct_value () [symbolic]
// CHECK:STDOUT:   %Iterate.lookup_impl_witness.171: <witness> = lookup_impl_witness %R.7b2, @Iterate [symbolic]
// CHECK:STDOUT:   %Iterate.facet.523: %Iterate.type = facet_value %R.as_type.657, (%Iterate.lookup_impl_witness.171) [symbolic]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.51e: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.523) [symbolic]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.e1d: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.523) [symbolic]
// CHECK:STDOUT:   %.250: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.51e, %Iterate.facet.523 [symbolic]
// CHECK:STDOUT:   %impl.elem2.9ef: %.250 = impl_witness_access %Iterate.lookup_impl_witness.171, element2 [symbolic]
// CHECK:STDOUT:   %impl.elem1.f12: %Destroy.type = impl_witness_access %Iterate.lookup_impl_witness.171, element1 [symbolic]
// CHECK:STDOUT:   %as_type.6f2: type = facet_access_type %impl.elem1.f12 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.5ff: <specific function> = specific_impl_function %impl.elem2.9ef, @Iterate.WithSelf.NewCursor(%Iterate.facet.523) [symbolic]
// CHECK:STDOUT:   %ptr.b94: type = ptr_type %as_type.6f2 [symbolic]
// CHECK:STDOUT:   %.f81: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.e1d, %Iterate.facet.523 [symbolic]
// CHECK:STDOUT:   %impl.elem3.04c: %.f81 = impl_witness_access %Iterate.lookup_impl_witness.171, element3 [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.541: type = class_type @DefaultOptionalStorage, @DefaultOptionalStorage(%facet_value.209) [concrete]
// CHECK:STDOUT:   %OptionalStorage.impl_witness.fd3: <witness> = impl_witness imports.%OptionalStorage.impl_witness_table.655, @T.as_type.as.OptionalStorage.impl(%facet_value.209) [concrete]
// CHECK:STDOUT:   %OptionalStorage.facet.cc1: %OptionalStorage.type = facet_value %f64.dc1, (%OptionalStorage.impl_witness.fd3) [concrete]
// CHECK:STDOUT:   %Optional.f48: type = class_type @Optional, @Optional(%OptionalStorage.facet.cc1) [concrete]
// CHECK:STDOUT:   %specific_impl_fn.c5b: <specific function> = specific_impl_function %impl.elem3.04c, @Iterate.WithSelf.Next(%Iterate.facet.523) [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.type.d4e: type = fn_type @Optional.HasValue, @Optional(%OptionalStorage.facet.cc1) [concrete]
// CHECK:STDOUT:   %Optional.HasValue.e79: %Optional.HasValue.type.d4e = struct_value () [concrete]
// CHECK:STDOUT:   %Optional.Get.type.822: type = fn_type @Optional.Get, @Optional(%OptionalStorage.facet.cc1) [concrete]
// CHECK:STDOUT:   %Optional.Get.f05: %Optional.Get.type.822 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.facet.339: %Destroy.type = facet_value %f64.dc1, (%custom_witness.df9cc1.2) [concrete]
// CHECK:STDOUT:   %MaybeUnformed.b7e: type = class_type @MaybeUnformed, @MaybeUnformed(%Destroy.facet.339) [concrete]
// CHECK:STDOUT:   %.cc0: type = maybe_unformed_type %f64.dc1 [concrete]
// CHECK:STDOUT:   %struct_type.value.has_value.ad9: type = struct_type {.value: %MaybeUnformed.b7e, .has_value: bool} [concrete]
// CHECK:STDOUT:   %Optional.HasValue.specific_fn: <specific function> = specific_function %Optional.HasValue.e79, @Optional.HasValue(%OptionalStorage.facet.cc1) [concrete]
// CHECK:STDOUT:   %Optional.Get.specific_fn: <specific function> = specific_function %Optional.Get.f05, @Optional.Get(%OptionalStorage.facet.cc1) [concrete]
// CHECK:STDOUT:   %AddAssignWith.type.ff9: type = facet_type <@AddAssignWith, @AddAssignWith(%f64.dc1)> [concrete]
// CHECK:STDOUT:   %Float.as.AddAssignWith.impl.Op.type.fdd: type = fn_type @Float.as.AddAssignWith.impl.Op, @Float.as.AddAssignWith.impl(%N.fe9) [symbolic]
// CHECK:STDOUT:   %Float.as.AddAssignWith.impl.Op.430: %Float.as.AddAssignWith.impl.Op.type.fdd = struct_value () [symbolic]
// CHECK:STDOUT:   %AddAssignWith.impl_witness.e85: <witness> = impl_witness imports.%AddAssignWith.impl_witness_table, @Float.as.AddAssignWith.impl(%int_64) [concrete]
// CHECK:STDOUT:   %Float.as.AddAssignWith.impl.Op.type.a92: type = fn_type @Float.as.AddAssignWith.impl.Op, @Float.as.AddAssignWith.impl(%int_64) [concrete]
// CHECK:STDOUT:   %Float.as.AddAssignWith.impl.Op.ac1: %Float.as.AddAssignWith.impl.Op.type.a92 = struct_value () [concrete]
// CHECK:STDOUT:   %AddAssignWith.facet: %AddAssignWith.type.ff9 = facet_value %f64.dc1, (%AddAssignWith.impl_witness.e85) [concrete]
// CHECK:STDOUT:   %AddAssignWith.WithSelf.Op.type.f8a: type = fn_type @AddAssignWith.WithSelf.Op, @AddAssignWith.WithSelf(%f64.dc1, %AddAssignWith.facet) [concrete]
// CHECK:STDOUT:   %.746: type = fn_type_with_self_type %AddAssignWith.WithSelf.Op.type.f8a, %AddAssignWith.facet [concrete]
// CHECK:STDOUT:   %Float.as.AddAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Float.as.AddAssignWith.impl.Op.ac1, @Float.as.AddAssignWith.impl.Op(%int_64) [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.9: type = fn_type @Destroy.Op.loc26_19.6 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.9: %Destroy.Op.type.1d8f74.9 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.f8f: <witness> = lookup_impl_witness %impl.elem1.f12, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.b9c: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%impl.elem1.f12) [symbolic]
// CHECK:STDOUT:   %.75c: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.b9c, %impl.elem1.f12 [symbolic]
// CHECK:STDOUT:   %impl.elem0.688: %.75c = impl_witness_access %Destroy.lookup_impl_witness.f8f, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.1e0: <specific function> = specific_impl_function %impl.elem0.688, @Destroy.WithSelf.Op(%impl.elem1.f12) [symbolic]
// CHECK:STDOUT:   %MutableRange: type = class_type @MutableRange [concrete]
// CHECK:STDOUT:   %pattern_type.992: type = pattern_type %MutableRange [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %Iterator: type = class_type @Iterator.1 [concrete]
// CHECK:STDOUT:   %Begin.type: type = fn_type @Begin [concrete]
// CHECK:STDOUT:   %Begin: %Begin.type = struct_value () [concrete]
// CHECK:STDOUT:   %End.type: type = fn_type @End [concrete]
// CHECK:STDOUT:   %End: %End.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.902: <witness> = custom_witness (%Iterator, %Iterator, %Begin, %End), @CppRangeForIterate [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.b9731e.1: type = fn_type @Iterator.Op.1 [concrete]
// CHECK:STDOUT:   %Iterator.Op.ac6aad.1: %Iterator.Op.type.b9731e.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.05b: <witness> = custom_witness (%f64.dc1, %Iterator.Op.ac6aad.1), @CppUnsafeDeref [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.b9731e.2: type = fn_type @Iterator.Op.2 [concrete]
// CHECK:STDOUT:   %Iterator.Op.ac6aad.2: %Iterator.Op.type.b9731e.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.c08: <witness> = custom_witness (%Iterator.Op.ac6aad.2), @Destroy [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.b9731e.3: type = fn_type @Iterator.Op.3 [concrete]
// CHECK:STDOUT:   %Iterator.Op.ac6aad.3: %Iterator.Op.type.b9731e.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.29f: <witness> = custom_witness (%Iterator.Op.ac6aad.3), @Copy [concrete]
// CHECK:STDOUT:   %Equal.type: type = fn_type @Equal [concrete]
// CHECK:STDOUT:   %Equal: %Equal.type = struct_value () [concrete]
// CHECK:STDOUT:   %NotEqual.type: type = fn_type @NotEqual [concrete]
// CHECK:STDOUT:   %NotEqual: %NotEqual.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.ed8: <witness> = custom_witness (%Equal, %NotEqual), @EqWith, @EqWith(%Iterator) [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.b9731e.4: type = fn_type @Iterator.Op.4 [concrete]
// CHECK:STDOUT:   %Iterator.Op.ac6aad.4: %Iterator.Op.type.b9731e.4 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.491: <witness> = custom_witness (%Iterator.Op.ac6aad.4), @Inc [concrete]
// CHECK:STDOUT:   %facet_value.85f: %CppRangeForIterate_where.type = facet_value %MutableRange, (%custom_witness.df9cc1.2, %Copy.impl_witness.873, %custom_witness.902, %custom_witness.c08, %custom_witness.29f, %custom_witness.05b, %custom_witness.ed8, %custom_witness.491) [concrete]
// CHECK:STDOUT:   %tuple.type.dd0: type = tuple_type (%Iterator, %Iterator) [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.10: type = fn_type @Destroy.Op.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.10: %Destroy.Op.type.1d8f74.10 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.df9cc1.10: <witness> = custom_witness (%Destroy.Op.1a2547.10), @Destroy [concrete]
// CHECK:STDOUT:   %Destroy.facet.e1e: %Destroy.type = facet_value %tuple.type.dd0, (%custom_witness.df9cc1.10) [concrete]
// CHECK:STDOUT:   %Iterate.impl_witness.5f7: <witness> = impl_witness imports.%Iterate.impl_witness_table.344, @R.as_type.as.Iterate.impl(%facet_value.85f) [concrete]
// CHECK:STDOUT:   %R.as_type.as.Iterate.impl.NewCursor.type.a95: type = fn_type @R.as_type.as.Iterate.impl.NewCursor, @R.as_type.as.Iterate.impl(%facet_value.85f) [concrete]
// CHECK:STDOUT:   %R.as_type.as.Iterate.impl.NewCursor.41a: %R.as_type.as.Iterate.impl.NewCursor.type.a95 = struct_value () [concrete]
// CHECK:STDOUT:   %R.as_type.as.Iterate.impl.Next.type.5c1: type = fn_type @R.as_type.as.Iterate.impl.Next, @R.as_type.as.Iterate.impl(%facet_value.85f) [concrete]
// CHECK:STDOUT:   %R.as_type.as.Iterate.impl.Next.928: %R.as_type.as.Iterate.impl.Next.type.5c1 = struct_value () [concrete]
// CHECK:STDOUT:   %Iterate.facet.c9f: %Iterate.type = facet_value %MutableRange, (%Iterate.impl_witness.5f7) [concrete]
// CHECK:STDOUT:   %facet_value.7dc: %Iterate_where.type.83b = facet_value %MutableRange, (%Iterate.impl_witness.5f7) [concrete]
// CHECK:STDOUT:   %r.param_patt.aeb: %pattern_type.992 = value_param_pattern [concrete]
// CHECK:STDOUT:   %r.patt.fe7: %pattern_type.992 = at_binding_pattern r, %r.param_patt.aeb [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.bde: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.c9f) [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.454: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.c9f) [concrete]
// CHECK:STDOUT:   %.c29a: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.bde, %Iterate.facet.c9f [concrete]
// CHECK:STDOUT:   %R.as_type.as.Iterate.impl.NewCursor.specific_fn: <specific function> = specific_function %R.as_type.as.Iterate.impl.NewCursor.41a, @R.as_type.as.Iterate.impl.NewCursor(%facet_value.85f) [concrete]
// CHECK:STDOUT:   %ptr.a4f: type = ptr_type %tuple.type.dd0 [concrete]
// CHECK:STDOUT:   %.b4c: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.454, %Iterate.facet.c9f [concrete]
// CHECK:STDOUT:   %R.as_type.as.Iterate.impl.Next.specific_fn: <specific function> = specific_function %R.as_type.as.Iterate.impl.Next.928, @R.as_type.as.Iterate.impl.Next(%facet_value.85f) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.452: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.e1e) [concrete]
// CHECK:STDOUT:   %.899: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.452, %Destroy.facet.e1e [concrete]
// CHECK:STDOUT:   %complete_type.843: <witness> = complete_type_witness %tuple.type.dd0 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.bb6: @Optional.%Optional.HasValue.type (%Optional.HasValue.type.e9e) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.HasValue (constants.%Optional.HasValue.6cd)]
// CHECK:STDOUT:   %Core.import_ref.daba: @Optional.%Optional.Get.type (%Optional.Get.type.bb9) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.Get (constants.%Optional.Get.fa1)]
// CHECK:STDOUT:   %Core.import_ref.0c6: type = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%DefaultOptionalStorage (constants.%DefaultOptionalStorage.d50)]
// CHECK:STDOUT:   %Core.import_ref.f4f: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.None.type (%T.as_type.as.OptionalStorage.impl.None.type.300) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.None (constants.%T.as_type.as.OptionalStorage.impl.None.f54)]
// CHECK:STDOUT:   %Core.import_ref.18f: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Some.type (%T.as_type.as.OptionalStorage.impl.Some.type.800) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Some (constants.%T.as_type.as.OptionalStorage.impl.Some.e83)]
// CHECK:STDOUT:   %Core.import_ref.c688: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Has.type (%T.as_type.as.OptionalStorage.impl.Has.type.4bc) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Has (constants.%T.as_type.as.OptionalStorage.impl.Has.dc2)]
// CHECK:STDOUT:   %Core.import_ref.a83: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Get.type (%T.as_type.as.OptionalStorage.impl.Get.type.029) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Get (constants.%T.as_type.as.OptionalStorage.impl.Get.cb5)]
// CHECK:STDOUT:   %Core.import_ref.aaa3 = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], unloaded
// CHECK:STDOUT:   %OptionalStorage.impl_witness_table.655 = impl_witness_table (%Core.import_ref.0c6, %Core.import_ref.f4f, %Core.import_ref.18f, %Core.import_ref.c688, %Core.import_ref.a83, %Core.import_ref.aaa3), @T.as_type.as.OptionalStorage.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.e69: @Float.as.Copy.impl.%Float.as.Copy.impl.Op.type (%Float.as.Copy.impl.Op.type.ff3) = import_ref Core//prelude/types/float, loc{{\d+_\d+}}, loaded [symbolic = @Float.as.Copy.impl.%Float.as.Copy.impl.Op (constants.%Float.as.Copy.impl.Op.d93)]
// CHECK:STDOUT:   %Copy.impl_witness_table.4e8 = impl_witness_table (%Core.import_ref.e69), @Float.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.301: %facet_type = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @R.as_type.as.Iterate.impl.%facet_value (constants.%facet_value.194)]
// CHECK:STDOUT:   %Core.import_ref.89f: %Destroy.type = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @R.as_type.as.Iterate.impl.%Destroy.facet (constants.%Destroy.facet.593)]
// CHECK:STDOUT:   %Core.import_ref.3e6: @R.as_type.as.Iterate.impl.%R.as_type.as.Iterate.impl.NewCursor.type (%R.as_type.as.Iterate.impl.NewCursor.type.36f) = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @R.as_type.as.Iterate.impl.%R.as_type.as.Iterate.impl.NewCursor (constants.%R.as_type.as.Iterate.impl.NewCursor.29b)]
// CHECK:STDOUT:   %Core.import_ref.85d: @R.as_type.as.Iterate.impl.%R.as_type.as.Iterate.impl.Next.type (%R.as_type.as.Iterate.impl.Next.type.1f7) = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @R.as_type.as.Iterate.impl.%R.as_type.as.Iterate.impl.Next (constants.%R.as_type.as.Iterate.impl.Next.0d6)]
// CHECK:STDOUT:   %Iterate.impl_witness_table.344 = impl_witness_table (%Core.import_ref.301, %Core.import_ref.89f, %Core.import_ref.3e6, %Core.import_ref.85d), @R.as_type.as.Iterate.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.cb3: @Float.as.AddAssignWith.impl.%Float.as.AddAssignWith.impl.Op.type (%Float.as.AddAssignWith.impl.Op.type.fdd) = import_ref Core//prelude/types/float, loc{{\d+_\d+}}, loaded [symbolic = @Float.as.AddAssignWith.impl.%Float.as.AddAssignWith.impl.Op (constants.%Float.as.AddAssignWith.impl.Op.430)]
// CHECK:STDOUT:   %AddAssignWith.impl_witness_table = impl_witness_table (%Core.import_ref.cb3), @Float.as.AddAssignWith.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @TestIterate(%R.loc22_17.2: %Iterate_where.type.83b) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Iterate.lookup_impl_witness: <witness> = lookup_impl_witness %R.loc22_17.1, @Iterate [symbolic = %Iterate.lookup_impl_witness (constants.%Iterate.lookup_impl_witness.171)]
// CHECK:STDOUT:   %Iterate.facet.loc26_19.5: %Iterate.type = facet_value %R.as_type.loc22_62.1, (%Iterate.lookup_impl_witness) [symbolic = %Iterate.facet.loc26_19.5 (constants.%Iterate.facet.523)]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.loc26_19.5) [symbolic = %Iterate.WithSelf.NewCursor.type (constants.%Iterate.WithSelf.NewCursor.type.51e)]
// CHECK:STDOUT:   %.loc26_19.19: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type, %Iterate.facet.loc26_19.5 [symbolic = %.loc26_19.19 (constants.%.250)]
// CHECK:STDOUT:   %impl.elem2.loc26_19.2: @TestIterate.%.loc26_19.19 (%.250) = impl_witness_access %Iterate.lookup_impl_witness, element2 [symbolic = %impl.elem2.loc26_19.2 (constants.%impl.elem2.9ef)]
// CHECK:STDOUT:   %specific_impl_fn.loc26_19.4: <specific function> = specific_impl_function %impl.elem2.loc26_19.2, @Iterate.WithSelf.NewCursor(%Iterate.facet.loc26_19.5) [symbolic = %specific_impl_fn.loc26_19.4 (constants.%specific_impl_fn.5ff)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %ptr: type = ptr_type %as_type [symbolic = %ptr (constants.%ptr.b94)]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.loc26_19.5) [symbolic = %Iterate.WithSelf.Next.type (constants.%Iterate.WithSelf.Next.type.e1d)]
// CHECK:STDOUT:   %.loc26_19.20: type = fn_type_with_self_type %Iterate.WithSelf.Next.type, %Iterate.facet.loc26_19.5 [symbolic = %.loc26_19.20 (constants.%.f81)]
// CHECK:STDOUT:   %impl.elem3.loc26_19.2: @TestIterate.%.loc26_19.20 (%.f81) = impl_witness_access %Iterate.lookup_impl_witness, element3 [symbolic = %impl.elem3.loc26_19.2 (constants.%impl.elem3.04c)]
// CHECK:STDOUT:   %specific_impl_fn.loc26_19.5: <specific function> = specific_impl_function %impl.elem3.loc26_19.2, @Iterate.WithSelf.Next(%Iterate.facet.loc26_19.5) [symbolic = %specific_impl_fn.loc26_19.5 (constants.%specific_impl_fn.c5b)]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%impl.elem1) [symbolic = %Destroy.WithSelf.Op.type (constants.%Destroy.WithSelf.Op.type.b9c)]
// CHECK:STDOUT:   %.loc26_19.21: type = fn_type_with_self_type %Destroy.WithSelf.Op.type, %impl.elem1 [symbolic = %.loc26_19.21 (constants.%.75c)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %impl.elem1, @Destroy [symbolic = %Destroy.lookup_impl_witness (constants.%Destroy.lookup_impl_witness.f8f)]
// CHECK:STDOUT:   %impl.elem0.loc26_19.2: @TestIterate.%.loc26_19.21 (%.75c) = impl_witness_access %Destroy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc26_19.2 (constants.%impl.elem0.688)]
// CHECK:STDOUT:   %specific_impl_fn.loc26_19.6: <specific function> = specific_impl_function %impl.elem0.loc26_19.2, @Destroy.WithSelf.Op(%impl.elem1) [symbolic = %specific_impl_fn.loc26_19.6 (constants.%specific_impl_fn.1e0)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%r.param: @TestIterate.%R.as_type.loc22_62.1 (%R.as_type.657)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %i.patt: %pattern_type.fb7 = value_binding_pattern i [concrete = constants.%i.patt]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %r.ref: @TestIterate.%R.as_type.loc22_62.1 (%R.as_type.657) = name_ref r, %r
// CHECK:STDOUT:     %impl.elem2.loc26_19.1: @TestIterate.%.loc26_19.19 (%.250) = impl_witness_access constants.%Iterate.lookup_impl_witness.171, element2 [symbolic = %impl.elem2.loc26_19.2 (constants.%impl.elem2.9ef)]
// CHECK:STDOUT:     %bound_method.loc26_19.1: <bound method> = bound_method %r.ref, %impl.elem2.loc26_19.1
// CHECK:STDOUT:     %Iterate.facet.loc26_19.1: %Iterate.type = facet_value constants.%R.as_type.657, (constants.%Iterate.lookup_impl_witness.171) [symbolic = %Iterate.facet.loc26_19.5 (constants.%Iterate.facet.523)]
// CHECK:STDOUT:     %.loc26_19.1: %Iterate.type = converted constants.%R.as_type.657, %Iterate.facet.loc26_19.1 [symbolic = %Iterate.facet.loc26_19.5 (constants.%Iterate.facet.523)]
// CHECK:STDOUT:     %Iterate.facet.loc26_19.2: %Iterate.type = facet_value constants.%R.as_type.657, (constants.%Iterate.lookup_impl_witness.171) [symbolic = %Iterate.facet.loc26_19.5 (constants.%Iterate.facet.523)]
// CHECK:STDOUT:     %.loc26_19.2: %Iterate.type = converted constants.%R.as_type.657, %Iterate.facet.loc26_19.2 [symbolic = %Iterate.facet.loc26_19.5 (constants.%Iterate.facet.523)]
// CHECK:STDOUT:     %specific_impl_fn.loc26_19.1: <specific function> = specific_impl_function %impl.elem2.loc26_19.1, @Iterate.WithSelf.NewCursor(constants.%Iterate.facet.523) [symbolic = %specific_impl_fn.loc26_19.4 (constants.%specific_impl_fn.5ff)]
// CHECK:STDOUT:     %bound_method.loc26_19.2: <bound method> = bound_method %r.ref, %specific_impl_fn.loc26_19.1
// CHECK:STDOUT:     %var: ref @TestIterate.%as_type (%as_type.6f2) = var invalid
// CHECK:STDOUT:     %Iterate.WithSelf.NewCursor.call: init @TestIterate.%as_type (%as_type.6f2) to %var = call %bound_method.loc26_19.2(%r.ref)
// CHECK:STDOUT:     assign %var, %Iterate.WithSelf.NewCursor.call
// CHECK:STDOUT:     br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.next:
// CHECK:STDOUT:     %addr: @TestIterate.%ptr (%ptr.b94) = addr_of %var
// CHECK:STDOUT:     %impl.elem3.loc26_19.1: @TestIterate.%.loc26_19.20 (%.f81) = impl_witness_access constants.%Iterate.lookup_impl_witness.171, element3 [symbolic = %impl.elem3.loc26_19.2 (constants.%impl.elem3.04c)]
// CHECK:STDOUT:     %bound_method.loc26_19.3: <bound method> = bound_method %r.ref, %impl.elem3.loc26_19.1
// CHECK:STDOUT:     %Iterate.facet.loc26_19.3: %Iterate.type = facet_value constants.%R.as_type.657, (constants.%Iterate.lookup_impl_witness.171) [symbolic = %Iterate.facet.loc26_19.5 (constants.%Iterate.facet.523)]
// CHECK:STDOUT:     %.loc26_19.3: %Iterate.type = converted constants.%R.as_type.657, %Iterate.facet.loc26_19.3 [symbolic = %Iterate.facet.loc26_19.5 (constants.%Iterate.facet.523)]
// CHECK:STDOUT:     %Iterate.facet.loc26_19.4: %Iterate.type = facet_value constants.%R.as_type.657, (constants.%Iterate.lookup_impl_witness.171) [symbolic = %Iterate.facet.loc26_19.5 (constants.%Iterate.facet.523)]
// CHECK:STDOUT:     %.loc26_19.4: %Iterate.type = converted constants.%R.as_type.657, %Iterate.facet.loc26_19.4 [symbolic = %Iterate.facet.loc26_19.5 (constants.%Iterate.facet.523)]
// CHECK:STDOUT:     %.loc26_19.5: %Destroy.type = converted constants.%as_type.6f2, constants.%impl.elem1.f12 [symbolic = %impl.elem1 (constants.%impl.elem1.f12)]
// CHECK:STDOUT:     %.loc26_19.6: %Destroy.type = converted constants.%as_type.6f2, constants.%impl.elem1.f12 [symbolic = %impl.elem1 (constants.%impl.elem1.f12)]
// CHECK:STDOUT:     %specific_impl_fn.loc26_19.2: <specific function> = specific_impl_function %impl.elem3.loc26_19.1, @Iterate.WithSelf.Next(constants.%Iterate.facet.523) [symbolic = %specific_impl_fn.loc26_19.5 (constants.%specific_impl_fn.c5b)]
// CHECK:STDOUT:     %bound_method.loc26_19.4: <bound method> = bound_method %r.ref, %specific_impl_fn.loc26_19.2
// CHECK:STDOUT:     %.loc26_19.7: ref %Optional.f48 = temporary_storage
// CHECK:STDOUT:     %Iterate.WithSelf.Next.call: init %Optional.f48 to %.loc26_19.7 = call %bound_method.loc26_19.4(%r.ref, %addr)
// CHECK:STDOUT:     %.loc26_19.8: ref %Optional.f48 = temporary %.loc26_19.7, %Iterate.WithSelf.Next.call
// CHECK:STDOUT:     %.loc26_19.9: %Optional.HasValue.type.d4e = specific_constant imports.%Core.import_ref.bb6, @Optional(constants.%OptionalStorage.facet.cc1) [concrete = constants.%Optional.HasValue.e79]
// CHECK:STDOUT:     %HasValue.ref: %Optional.HasValue.type.d4e = name_ref HasValue, %.loc26_19.9 [concrete = constants.%Optional.HasValue.e79]
// CHECK:STDOUT:     %Optional.HasValue.bound: <bound method> = bound_method %.loc26_19.8, %HasValue.ref
// CHECK:STDOUT:     %Optional.HasValue.specific_fn: <specific function> = specific_function %HasValue.ref, @Optional.HasValue(constants.%OptionalStorage.facet.cc1) [concrete = constants.%Optional.HasValue.specific_fn]
// CHECK:STDOUT:     %bound_method.loc26_19.5: <bound method> = bound_method %.loc26_19.8, %Optional.HasValue.specific_fn
// CHECK:STDOUT:     %.loc26_19.10: %Optional.f48 = acquire_value %.loc26_19.8
// CHECK:STDOUT:     %Optional.HasValue.call: init bool = call %bound_method.loc26_19.5(%.loc26_19.10)
// CHECK:STDOUT:     %.loc26_19.11: bool = value_of_initializer %Optional.HasValue.call
// CHECK:STDOUT:     %.loc26_19.12: bool = converted %Optional.HasValue.call, %.loc26_19.11
// CHECK:STDOUT:     if %.loc26_19.12 br !for.body else br !for.done
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.body:
// CHECK:STDOUT:     %.loc26_19.13: %Optional.Get.type.822 = specific_constant imports.%Core.import_ref.daba, @Optional(constants.%OptionalStorage.facet.cc1) [concrete = constants.%Optional.Get.f05]
// CHECK:STDOUT:     %Get.ref: %Optional.Get.type.822 = name_ref Get, %.loc26_19.13 [concrete = constants.%Optional.Get.f05]
// CHECK:STDOUT:     %Optional.Get.bound: <bound method> = bound_method %.loc26_19.8, %Get.ref
// CHECK:STDOUT:     %Optional.Get.specific_fn: <specific function> = specific_function %Get.ref, @Optional.Get(constants.%OptionalStorage.facet.cc1) [concrete = constants.%Optional.Get.specific_fn]
// CHECK:STDOUT:     %bound_method.loc26_19.6: <bound method> = bound_method %.loc26_19.8, %Optional.Get.specific_fn
// CHECK:STDOUT:     %.loc26_19.14: %Optional.f48 = acquire_value %.loc26_19.8
// CHECK:STDOUT:     %Optional.Get.call: init %f64.dc1 = call %bound_method.loc26_19.6(%.loc26_19.14)
// CHECK:STDOUT:     %.loc26_19.15: %f64.dc1 = value_of_initializer %Optional.Get.call
// CHECK:STDOUT:     %.loc26_19.16: %f64.dc1 = converted %Optional.Get.call, %.loc26_19.15
// CHECK:STDOUT:     %f64.loc26: type = type_literal constants.%f64.dc1 [concrete = constants.%f64.dc1]
// CHECK:STDOUT:     %i: %f64.dc1 = wrapper_binding i, %.loc26_19.16
// CHECK:STDOUT:     %sum.ref: ref %f64.dc1 = name_ref sum, %sum
// CHECK:STDOUT:     %i.ref: %f64.dc1 = name_ref i, %i
// CHECK:STDOUT:     %impl.elem0.loc27: %.746 = impl_witness_access constants.%AddAssignWith.impl_witness.e85, element0 [concrete = constants.%Float.as.AddAssignWith.impl.Op.ac1]
// CHECK:STDOUT:     %bound_method.loc27_9.1: <bound method> = bound_method %sum.ref, %impl.elem0.loc27
// CHECK:STDOUT:     %specific_fn.loc27: <specific function> = specific_function %impl.elem0.loc27, @Float.as.AddAssignWith.impl.Op(constants.%int_64) [concrete = constants.%Float.as.AddAssignWith.impl.Op.specific_fn]
// CHECK:STDOUT:     %bound_method.loc27_9.2: <bound method> = bound_method %sum.ref, %specific_fn.loc27
// CHECK:STDOUT:     %Float.as.AddAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc27_9.2(%sum.ref, %i.ref)
// CHECK:STDOUT:     br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.done:
// CHECK:STDOUT:     %Destroy.Op.bound.loc26: <bound method> = bound_method %.loc26_19.8, constants.%Destroy.Op.1a2547.9
// CHECK:STDOUT:     %Destroy.Op.call.loc26: init %empty_tuple.type = call %Destroy.Op.bound.loc26(%.loc26_19.8)
// CHECK:STDOUT:     %impl.elem0.loc26_19.1: @TestIterate.%.loc26_19.21 (%.75c) = impl_witness_access constants.%Destroy.lookup_impl_witness.f8f, element0 [symbolic = %impl.elem0.loc26_19.2 (constants.%impl.elem0.688)]
// CHECK:STDOUT:     %bound_method.loc26_19.7: <bound method> = bound_method %var, %impl.elem0.loc26_19.1
// CHECK:STDOUT:     %.loc26_19.17: %Destroy.type = converted constants.%as_type.6f2, constants.%impl.elem1.f12 [symbolic = %impl.elem1 (constants.%impl.elem1.f12)]
// CHECK:STDOUT:     %.loc26_19.18: %Destroy.type = converted constants.%as_type.6f2, constants.%impl.elem1.f12 [symbolic = %impl.elem1 (constants.%impl.elem1.f12)]
// CHECK:STDOUT:     %specific_impl_fn.loc26_19.3: <specific function> = specific_impl_function %impl.elem0.loc26_19.1, @Destroy.WithSelf.Op(constants.%impl.elem1.f12) [symbolic = %specific_impl_fn.loc26_19.6 (constants.%specific_impl_fn.1e0)]
// CHECK:STDOUT:     %bound_method.loc26_19.8: <bound method> = bound_method %var, %specific_impl_fn.loc26_19.3
// CHECK:STDOUT:     %Destroy.WithSelf.Op.call: init %empty_tuple.type = call %bound_method.loc26_19.8(%var)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc26_19.1(%self.param: ref %.cc0) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc26_19.2(%self.param: ref %MaybeUnformed.b7e) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc26_19.3(%self.param: ref bool) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc26_19.4(%self.param: ref %struct_type.value.has_value.ad9) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc26_19.5(%self.param: ref %DefaultOptionalStorage.541) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc26_19.6(%self.param: ref %Optional.f48) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestRangeFor(%m.param: ref %MutableRange) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %i.patt: %pattern_type.fb7 = value_binding_pattern i [concrete = constants.%i.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %m.ref: ref %MutableRange = name_ref m, %m
// CHECK:STDOUT:   %impl.elem2: %.c29a = impl_witness_access constants.%Iterate.impl_witness.5f7, element2 [concrete = constants.%R.as_type.as.Iterate.impl.NewCursor.41a]
// CHECK:STDOUT:   %bound_method.loc68_19.1: <bound method> = bound_method %m.ref, %impl.elem2
// CHECK:STDOUT:   %facet_value.loc68_19.1: %CppRangeForIterate_where.type = facet_value constants.%MutableRange, (constants.%custom_witness.df9cc1.2, constants.%Copy.impl_witness.873, constants.%custom_witness.902, constants.%custom_witness.c08, constants.%custom_witness.29f, constants.%custom_witness.05b, constants.%custom_witness.ed8, constants.%custom_witness.491) [concrete = constants.%facet_value.85f]
// CHECK:STDOUT:   %.loc68_19.1: %CppRangeForIterate_where.type = converted constants.%MutableRange, %facet_value.loc68_19.1 [concrete = constants.%facet_value.85f]
// CHECK:STDOUT:   %facet_value.loc68_19.2: %CppRangeForIterate_where.type = facet_value constants.%MutableRange, (constants.%custom_witness.df9cc1.2, constants.%Copy.impl_witness.873, constants.%custom_witness.902, constants.%custom_witness.c08, constants.%custom_witness.29f, constants.%custom_witness.05b, constants.%custom_witness.ed8, constants.%custom_witness.491) [concrete = constants.%facet_value.85f]
// CHECK:STDOUT:   %.loc68_19.2: %CppRangeForIterate_where.type = converted constants.%MutableRange, %facet_value.loc68_19.2 [concrete = constants.%facet_value.85f]
// CHECK:STDOUT:   %specific_fn.loc68_19.1: <specific function> = specific_function %impl.elem2, @R.as_type.as.Iterate.impl.NewCursor(constants.%facet_value.85f) [concrete = constants.%R.as_type.as.Iterate.impl.NewCursor.specific_fn]
// CHECK:STDOUT:   %bound_method.loc68_19.2: <bound method> = bound_method %m.ref, %specific_fn.loc68_19.1
// CHECK:STDOUT:   %var: ref %tuple.type.dd0 = var invalid
// CHECK:STDOUT:   %.loc68_18.1: %MutableRange = acquire_value %m.ref
// CHECK:STDOUT:   %R.as_type.as.Iterate.impl.NewCursor.call: init %tuple.type.dd0 to %var = call %bound_method.loc68_19.2(%.loc68_18.1)
// CHECK:STDOUT:   assign %var, %R.as_type.as.Iterate.impl.NewCursor.call
// CHECK:STDOUT:   br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT: !for.next:
// CHECK:STDOUT:   %addr: %ptr.a4f = addr_of %var
// CHECK:STDOUT:   %impl.elem3: %.b4c = impl_witness_access constants.%Iterate.impl_witness.5f7, element3 [concrete = constants.%R.as_type.as.Iterate.impl.Next.928]
// CHECK:STDOUT:   %bound_method.loc68_19.3: <bound method> = bound_method %m.ref, %impl.elem3
// CHECK:STDOUT:   %facet_value.loc68_19.3: %CppRangeForIterate_where.type = facet_value constants.%MutableRange, (constants.%custom_witness.df9cc1.2, constants.%Copy.impl_witness.873, constants.%custom_witness.902, constants.%custom_witness.c08, constants.%custom_witness.29f, constants.%custom_witness.05b, constants.%custom_witness.ed8, constants.%custom_witness.491) [concrete = constants.%facet_value.85f]
// CHECK:STDOUT:   %.loc68_19.3: %CppRangeForIterate_where.type = converted constants.%MutableRange, %facet_value.loc68_19.3 [concrete = constants.%facet_value.85f]
// CHECK:STDOUT:   %facet_value.loc68_19.4: %CppRangeForIterate_where.type = facet_value constants.%MutableRange, (constants.%custom_witness.df9cc1.2, constants.%Copy.impl_witness.873, constants.%custom_witness.902, constants.%custom_witness.c08, constants.%custom_witness.29f, constants.%custom_witness.05b, constants.%custom_witness.ed8, constants.%custom_witness.491) [concrete = constants.%facet_value.85f]
// CHECK:STDOUT:   %.loc68_19.4: %CppRangeForIterate_where.type = converted constants.%MutableRange, %facet_value.loc68_19.4 [concrete = constants.%facet_value.85f]
// CHECK:STDOUT:   %specific_fn.loc68_19.2: <specific function> = specific_function %impl.elem3, @R.as_type.as.Iterate.impl.Next(constants.%facet_value.85f) [concrete = constants.%R.as_type.as.Iterate.impl.Next.specific_fn]
// CHECK:STDOUT:   %bound_method.loc68_19.4: <bound method> = bound_method %m.ref, %specific_fn.loc68_19.2
// CHECK:STDOUT:   %.loc68_19.5: ref %Optional.f48 = temporary_storage
// CHECK:STDOUT:   %.loc68_18.2: %MutableRange = acquire_value %m.ref
// CHECK:STDOUT:   %R.as_type.as.Iterate.impl.Next.call: init %Optional.f48 to %.loc68_19.5 = call %bound_method.loc68_19.4(%.loc68_18.2, %addr)
// CHECK:STDOUT:   %.loc68_19.6: ref %Optional.f48 = temporary %.loc68_19.5, %R.as_type.as.Iterate.impl.Next.call
// CHECK:STDOUT:   %.loc68_19.7: %Optional.HasValue.type.d4e = specific_constant imports.%Core.import_ref.bb6, @Optional(constants.%OptionalStorage.facet.cc1) [concrete = constants.%Optional.HasValue.e79]
// CHECK:STDOUT:   %HasValue.ref: %Optional.HasValue.type.d4e = name_ref HasValue, %.loc68_19.7 [concrete = constants.%Optional.HasValue.e79]
// CHECK:STDOUT:   %Optional.HasValue.bound: <bound method> = bound_method %.loc68_19.6, %HasValue.ref
// CHECK:STDOUT:   %Optional.HasValue.specific_fn: <specific function> = specific_function %HasValue.ref, @Optional.HasValue(constants.%OptionalStorage.facet.cc1) [concrete = constants.%Optional.HasValue.specific_fn]
// CHECK:STDOUT:   %bound_method.loc68_19.5: <bound method> = bound_method %.loc68_19.6, %Optional.HasValue.specific_fn
// CHECK:STDOUT:   %.loc68_19.8: %Optional.f48 = acquire_value %.loc68_19.6
// CHECK:STDOUT:   %Optional.HasValue.call: init bool = call %bound_method.loc68_19.5(%.loc68_19.8)
// CHECK:STDOUT:   %.loc68_19.9: bool = value_of_initializer %Optional.HasValue.call
// CHECK:STDOUT:   %.loc68_19.10: bool = converted %Optional.HasValue.call, %.loc68_19.9
// CHECK:STDOUT:   if %.loc68_19.10 br !for.body else br !for.done
// CHECK:STDOUT:
// CHECK:STDOUT: !for.body:
// CHECK:STDOUT:   %.loc68_19.11: %Optional.Get.type.822 = specific_constant imports.%Core.import_ref.daba, @Optional(constants.%OptionalStorage.facet.cc1) [concrete = constants.%Optional.Get.f05]
// CHECK:STDOUT:   %Get.ref: %Optional.Get.type.822 = name_ref Get, %.loc68_19.11 [concrete = constants.%Optional.Get.f05]
// CHECK:STDOUT:   %Optional.Get.bound: <bound method> = bound_method %.loc68_19.6, %Get.ref
// CHECK:STDOUT:   %Optional.Get.specific_fn: <specific function> = specific_function %Get.ref, @Optional.Get(constants.%OptionalStorage.facet.cc1) [concrete = constants.%Optional.Get.specific_fn]
// CHECK:STDOUT:   %bound_method.loc68_19.6: <bound method> = bound_method %.loc68_19.6, %Optional.Get.specific_fn
// CHECK:STDOUT:   %.loc68_19.12: %Optional.f48 = acquire_value %.loc68_19.6
// CHECK:STDOUT:   %Optional.Get.call: init %f64.dc1 = call %bound_method.loc68_19.6(%.loc68_19.12)
// CHECK:STDOUT:   %.loc68_19.13: %f64.dc1 = value_of_initializer %Optional.Get.call
// CHECK:STDOUT:   %.loc68_19.14: %f64.dc1 = converted %Optional.Get.call, %.loc68_19.13
// CHECK:STDOUT:   %f64.loc68: type = type_literal constants.%f64.dc1 [concrete = constants.%f64.dc1]
// CHECK:STDOUT:   %i: %f64.dc1 = wrapper_binding i, %.loc68_19.14
// CHECK:STDOUT:   %sum.ref: ref %f64.dc1 = name_ref sum, %sum
// CHECK:STDOUT:   %i.ref: %f64.dc1 = name_ref i, %i
// CHECK:STDOUT:   %impl.elem0.loc69: %.746 = impl_witness_access constants.%AddAssignWith.impl_witness.e85, element0 [concrete = constants.%Float.as.AddAssignWith.impl.Op.ac1]
// CHECK:STDOUT:   %bound_method.loc69_9.1: <bound method> = bound_method %sum.ref, %impl.elem0.loc69
// CHECK:STDOUT:   %specific_fn.loc69: <specific function> = specific_function %impl.elem0.loc69, @Float.as.AddAssignWith.impl.Op(constants.%int_64) [concrete = constants.%Float.as.AddAssignWith.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc69_9.2: <bound method> = bound_method %sum.ref, %specific_fn.loc69
// CHECK:STDOUT:   %Float.as.AddAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc69_9.2(%sum.ref, %i.ref)
// CHECK:STDOUT:   br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT: !for.done:
// CHECK:STDOUT:   %Destroy.Op.bound.loc68_19.1: <bound method> = bound_method %.loc68_19.6, constants.%Destroy.Op.1a2547.9
// CHECK:STDOUT:   %Destroy.Op.call.loc68_19.1: init %empty_tuple.type = call %Destroy.Op.bound.loc68_19.1(%.loc68_19.6)
// CHECK:STDOUT:   %Destroy.Op.bound.loc68_19.2: <bound method> = bound_method %var, constants.%Destroy.Op.1a2547.10
// CHECK:STDOUT:   %Destroy.Op.call.loc68_19.2: init %empty_tuple.type = call %Destroy.Op.bound.loc68_19.2(%var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestIterate(constants.%R.7b2) {
// CHECK:STDOUT:   %R.patt.loc22_17.2 => constants.%R.patt.053
// CHECK:STDOUT:   %R.loc22_17.1 => constants.%R.7b2
// CHECK:STDOUT:   %R.as_type.loc22_62.1 => constants.%R.as_type.657
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.afe
// CHECK:STDOUT:   %r.param_patt.loc22_60.2 => constants.%r.param_patt.184
// CHECK:STDOUT:   %r.patt.loc22_60.2 => constants.%r.patt.727
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestIterate(constants.%facet_value.7dc) {
// CHECK:STDOUT:   %R.patt.loc22_17.2 => constants.%R.patt.053
// CHECK:STDOUT:   %R.loc22_17.1 => constants.%facet_value.7dc
// CHECK:STDOUT:   %R.as_type.loc22_62.1 => constants.%MutableRange
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.992
// CHECK:STDOUT:   %r.param_patt.loc22_60.2 => constants.%r.param_patt.aeb
// CHECK:STDOUT:   %r.patt.loc22_60.2 => constants.%r.patt.fe7
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc22 => constants.%complete_type.357
// CHECK:STDOUT:   %Iterate.lookup_impl_witness => constants.%Iterate.impl_witness.5f7
// CHECK:STDOUT:   %Iterate.facet.loc26_19.5 => constants.%Iterate.facet.c9f
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type => constants.%Iterate.WithSelf.NewCursor.type.bde
// CHECK:STDOUT:   %.loc26_19.19 => constants.%.c29a
// CHECK:STDOUT:   %impl.elem2.loc26_19.2 => constants.%R.as_type.as.Iterate.impl.NewCursor.41a
// CHECK:STDOUT:   %specific_impl_fn.loc26_19.4 => constants.%R.as_type.as.Iterate.impl.NewCursor.specific_fn
// CHECK:STDOUT:   %impl.elem1 => constants.%Destroy.facet.e1e
// CHECK:STDOUT:   %as_type => constants.%tuple.type.dd0
// CHECK:STDOUT:   %require_complete.1 => constants.%complete_type.843
// CHECK:STDOUT:   %ptr => constants.%ptr.a4f
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type => constants.%Iterate.WithSelf.Next.type.454
// CHECK:STDOUT:   %.loc26_19.20 => constants.%.b4c
// CHECK:STDOUT:   %impl.elem3.loc26_19.2 => constants.%R.as_type.as.Iterate.impl.Next.928
// CHECK:STDOUT:   %specific_impl_fn.loc26_19.5 => constants.%R.as_type.as.Iterate.impl.Next.specific_fn
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.452
// CHECK:STDOUT:   %.loc26_19.21 => constants.%.899
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.df9cc1.10
// CHECK:STDOUT:   %impl.elem0.loc26_19.2 => constants.%Destroy.Op.1a2547.10
// CHECK:STDOUT:   %specific_impl_fn.loc26_19.6 => constants.%Destroy.Op.1a2547.10
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
// CHECK:STDOUT:   %T.220: %OptionalStorage.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Optional.Get.type.bb9: type = fn_type @Optional.Get, @Optional(%T.220) [symbolic]
// CHECK:STDOUT:   %Optional.Get.fa1: %Optional.Get.type.bb9 = struct_value () [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.type.e9e: type = fn_type @Optional.HasValue, @Optional(%T.220) [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.6cd: %Optional.HasValue.type.e9e = struct_value () [symbolic]
// CHECK:STDOUT:   %T.283: %facet_type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.d50: type = class_type @DefaultOptionalStorage, @DefaultOptionalStorage(%T.283) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Get.type.029: type = fn_type @T.as_type.as.OptionalStorage.impl.Get, @T.as_type.as.OptionalStorage.impl(%T.283) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Get.cb5: %T.as_type.as.OptionalStorage.impl.Get.type.029 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Has.type.4bc: type = fn_type @T.as_type.as.OptionalStorage.impl.Has, @T.as_type.as.OptionalStorage.impl(%T.283) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Has.dc2: %T.as_type.as.OptionalStorage.impl.Has.type.4bc = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Some.type.800: type = fn_type @T.as_type.as.OptionalStorage.impl.Some, @T.as_type.as.OptionalStorage.impl(%T.283) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Some.e83: %T.as_type.as.OptionalStorage.impl.Some.type.800 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.None.type.300: type = fn_type @T.as_type.as.OptionalStorage.impl.None, @T.as_type.as.OptionalStorage.impl(%T.283) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.None.f54: %T.as_type.as.OptionalStorage.impl.None.type.300 = struct_value () [symbolic]
// CHECK:STDOUT:   %.Self.35d: %Iterate.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %Iterate.lookup_impl_witness.b37: <witness> = lookup_impl_witness %.Self.35d, @Iterate [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.903: %facet_type = impl_witness_access %Iterate.lookup_impl_witness.b37, element0 [symbolic_self]
// CHECK:STDOUT:   %ValueType: type = class_type @ValueType [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %pattern_type.e41: type = pattern_type %ValueType [concrete]
// CHECK:STDOUT:   %ValueType.cpp_destructor.type: type = fn_type @ValueType.cpp_destructor [concrete]
// CHECK:STDOUT:   %ValueType.cpp_destructor: %ValueType.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT:   %ValueType.Op.type.79266b.1: type = fn_type @ValueType.Op.1 [concrete]
// CHECK:STDOUT:   %ValueType.Op.4c94b9.1: %ValueType.Op.type.79266b.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.fef: <witness> = custom_witness (%ValueType.Op.4c94b9.1), @Destroy [concrete]
// CHECK:STDOUT:   %const.773: type = const_type %ValueType [concrete]
// CHECK:STDOUT:   %ptr.32e: type = ptr_type %const.773 [concrete]
// CHECK:STDOUT:   %ptr.9d3: type = ptr_type %ValueType [concrete]
// CHECK:STDOUT:   %ValueType.Op.type.79266b.2: type = fn_type @ValueType.Op.2 [concrete]
// CHECK:STDOUT:   %ValueType.Op.4c94b9.2: %ValueType.Op.type.79266b.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.c19: <witness> = custom_witness (%ValueType.Op.4c94b9.2), @Copy [concrete]
// CHECK:STDOUT:   %facet_value.ccb: %facet_type = facet_value %ValueType, (%custom_witness.fef, %custom_witness.c19) [concrete]
// CHECK:STDOUT:   %Iterate_where.type.99d: type = facet_type <@Iterate where %impl.elem0.903 = %facet_value.ccb> [concrete]
// CHECK:STDOUT:   %pattern_type.3f6: type = pattern_type %Iterate_where.type.99d [concrete]
// CHECK:STDOUT:   %R.patt.324: %pattern_type.3f6 = symbolic_binding_pattern R, 0 [symbolic]
// CHECK:STDOUT:   %R.623: %Iterate_where.type.99d = symbolic_binding R, 0 [symbolic]
// CHECK:STDOUT:   %R.as_type.a62: type = facet_access_type %R.623 [symbolic]
// CHECK:STDOUT:   %pattern_type.725: type = pattern_type %R.as_type.a62 [symbolic]
// CHECK:STDOUT:   %r.param_patt.162: %pattern_type.725 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %r.patt.15b: %pattern_type.725 = at_binding_pattern r, %r.param_patt.162 [symbolic]
// CHECK:STDOUT:   %i.patt: %pattern_type.e41 = value_binding_pattern i [concrete]
// CHECK:STDOUT:   %CppRangeForIterate.type: type = facet_type <@CppRangeForIterate> [concrete]
// CHECK:STDOUT:   %.Self.6bd: %CppRangeForIterate.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %CppRangeForIterate.lookup_impl_witness.143: <witness> = lookup_impl_witness %.Self.6bd, @CppRangeForIterate [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.109: type = impl_witness_access %CppRangeForIterate.lookup_impl_witness.143, element0 [symbolic_self]
// CHECK:STDOUT:   %CppUnsafeDeref.lookup_impl_witness.712: <witness> = lookup_impl_witness %impl.elem0.109, @CppUnsafeDeref [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.56a: type = impl_witness_access %CppUnsafeDeref.lookup_impl_witness.712, element0 [symbolic_self]
// CHECK:STDOUT:   %impl.elem1.222: type = impl_witness_access %CppRangeForIterate.lookup_impl_witness.143, element1 [symbolic_self]
// CHECK:STDOUT:   %CppRangeForIterate_where.type: type = facet_type <@CppRangeForIterate where %impl.elem0.109 impls @Destroy and %impl.elem0.109 impls @Copy and %impl.elem0.109 impls @CppUnsafeDeref and %impl.elem0.109 impls @EqWith, @EqWith(%impl.elem1.222) and %impl.elem0.109 impls @Inc and %impl.elem0.56a impls @Destroy and %impl.elem0.56a impls @Copy and %impl.elem1.222 impls @Destroy and %impl.elem1.222 impls @Copy> [concrete]
// CHECK:STDOUT:   %R.39b: %CppRangeForIterate_where.type = symbolic_binding R, 0 [symbolic]
// CHECK:STDOUT:   %CppRangeForIterate.lookup_impl_witness.7e2712.2: <witness> = lookup_impl_witness %R.39b, @CppRangeForIterate [symbolic]
// CHECK:STDOUT:   %impl.elem1.7a0bff.2: type = impl_witness_access %CppRangeForIterate.lookup_impl_witness.7e2712.2, element1 [symbolic]
// CHECK:STDOUT:   %impl.elem0.73eb7f.2: type = impl_witness_access %CppRangeForIterate.lookup_impl_witness.7e2712.2, element0 [symbolic]
// CHECK:STDOUT:   %tuple.type.3cf: type = tuple_type (%impl.elem0.73eb7f.2, %impl.elem1.7a0bff.2) [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.69f: <witness> = lookup_impl_witness %tuple.type.3cf, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.593: %Destroy.type = facet_value %tuple.type.3cf, (%Destroy.lookup_impl_witness.69f) [symbolic]
// CHECK:STDOUT:   %CppUnsafeDeref.lookup_impl_witness.aa7: <witness> = lookup_impl_witness %impl.elem0.73eb7f.2, @CppUnsafeDeref [symbolic]
// CHECK:STDOUT:   %impl.elem0.af7: type = impl_witness_access %CppUnsafeDeref.lookup_impl_witness.aa7, element0 [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.ec3: <witness> = lookup_impl_witness %impl.elem0.af7, @Destroy [symbolic]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.a7a: <witness> = lookup_impl_witness %impl.elem0.af7, @Copy [symbolic]
// CHECK:STDOUT:   %facet_value.194: %facet_type = facet_value %impl.elem0.af7, (%Destroy.lookup_impl_witness.ec3, %Copy.lookup_impl_witness.a7a) [symbolic]
// CHECK:STDOUT:   %R.as_type.as.Iterate.impl.Next.type.1f7: type = fn_type @R.as_type.as.Iterate.impl.Next, @R.as_type.as.Iterate.impl(%R.39b) [symbolic]
// CHECK:STDOUT:   %R.as_type.as.Iterate.impl.Next.0d6: %R.as_type.as.Iterate.impl.Next.type.1f7 = struct_value () [symbolic]
// CHECK:STDOUT:   %R.as_type.as.Iterate.impl.NewCursor.type.36f: type = fn_type @R.as_type.as.Iterate.impl.NewCursor, @R.as_type.as.Iterate.impl(%R.39b) [symbolic]
// CHECK:STDOUT:   %R.as_type.as.Iterate.impl.NewCursor.29b: %R.as_type.as.Iterate.impl.NewCursor.type.36f = struct_value () [symbolic]
// CHECK:STDOUT:   %Iterate.lookup_impl_witness.8b4: <witness> = lookup_impl_witness %R.623, @Iterate [symbolic]
// CHECK:STDOUT:   %Iterate.facet.b19: %Iterate.type = facet_value %R.as_type.a62, (%Iterate.lookup_impl_witness.8b4) [symbolic]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.217: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.b19) [symbolic]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.10a: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.b19) [symbolic]
// CHECK:STDOUT:   %.a27: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.217, %Iterate.facet.b19 [symbolic]
// CHECK:STDOUT:   %impl.elem2.7c0: %.a27 = impl_witness_access %Iterate.lookup_impl_witness.8b4, element2 [symbolic]
// CHECK:STDOUT:   %impl.elem1.46a: %Destroy.type = impl_witness_access %Iterate.lookup_impl_witness.8b4, element1 [symbolic]
// CHECK:STDOUT:   %as_type.862: type = facet_access_type %impl.elem1.46a [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.e23: <specific function> = specific_impl_function %impl.elem2.7c0, @Iterate.WithSelf.NewCursor(%Iterate.facet.b19) [symbolic]
// CHECK:STDOUT:   %ptr.530: type = ptr_type %as_type.862 [symbolic]
// CHECK:STDOUT:   %.df57: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.10a, %Iterate.facet.b19 [symbolic]
// CHECK:STDOUT:   %impl.elem3.199: %.df57 = impl_witness_access %Iterate.lookup_impl_witness.8b4, element3 [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.257: type = class_type @DefaultOptionalStorage, @DefaultOptionalStorage(%facet_value.ccb) [concrete]
// CHECK:STDOUT:   %OptionalStorage.impl_witness.be5: <witness> = impl_witness imports.%OptionalStorage.impl_witness_table.655, @T.as_type.as.OptionalStorage.impl(%facet_value.ccb) [concrete]
// CHECK:STDOUT:   %OptionalStorage.facet.7ce: %OptionalStorage.type = facet_value %ValueType, (%OptionalStorage.impl_witness.be5) [concrete]
// CHECK:STDOUT:   %Optional.552: type = class_type @Optional, @Optional(%OptionalStorage.facet.7ce) [concrete]
// CHECK:STDOUT:   %specific_impl_fn.169: <specific function> = specific_impl_function %impl.elem3.199, @Iterate.WithSelf.Next(%Iterate.facet.b19) [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.type.109: type = fn_type @Optional.HasValue, @Optional(%OptionalStorage.facet.7ce) [concrete]
// CHECK:STDOUT:   %Optional.HasValue.605: %Optional.HasValue.type.109 = struct_value () [concrete]
// CHECK:STDOUT:   %Optional.Get.type.135: type = fn_type @Optional.Get, @Optional(%OptionalStorage.facet.7ce) [concrete]
// CHECK:STDOUT:   %Optional.Get.5aa: %Optional.Get.type.135 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.facet.306: %Destroy.type = facet_value %ValueType, (%custom_witness.fef) [concrete]
// CHECK:STDOUT:   %MaybeUnformed.669: type = class_type @MaybeUnformed, @MaybeUnformed(%Destroy.facet.306) [concrete]
// CHECK:STDOUT:   %.b36: type = maybe_unformed_type %ValueType [concrete]
// CHECK:STDOUT:   %struct_type.value.has_value.5a1: type = struct_type {.value: %MaybeUnformed.669, .has_value: bool} [concrete]
// CHECK:STDOUT:   %Optional.HasValue.specific_fn: <specific function> = specific_function %Optional.HasValue.605, @Optional.HasValue(%OptionalStorage.facet.7ce) [concrete]
// CHECK:STDOUT:   %Optional.Get.specific_fn: <specific function> = specific_function %Optional.Get.5aa, @Optional.Get(%OptionalStorage.facet.7ce) [concrete]
// CHECK:STDOUT:   %operator_PlusEqual__carbon_thunk.type: type = fn_type @operator_PlusEqual__carbon_thunk [concrete]
// CHECK:STDOUT:   %operator_PlusEqual__carbon_thunk: %operator_PlusEqual__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.7: type = fn_type @Destroy.Op.loc31_31.6 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.7: %Destroy.Op.type.1d8f74.7 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.f37: <witness> = lookup_impl_witness %impl.elem1.46a, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.e7e: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%impl.elem1.46a) [symbolic]
// CHECK:STDOUT:   %.bc1: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.e7e, %impl.elem1.46a [symbolic]
// CHECK:STDOUT:   %impl.elem0.c7c: %.bc1 = impl_witness_access %Destroy.lookup_impl_witness.f37, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.cd3: <specific function> = specific_impl_function %impl.elem0.c7c, @Destroy.WithSelf.Op(%impl.elem1.46a) [symbolic]
// CHECK:STDOUT:   %ConstRange: type = class_type @ConstRange [concrete]
// CHECK:STDOUT:   %pattern_type.025: type = pattern_type %ConstRange [concrete]
// CHECK:STDOUT:   %Iterator: type = class_type @Iterator.1 [concrete]
// CHECK:STDOUT:   %Sentinel: type = class_type @Sentinel.1 [concrete]
// CHECK:STDOUT:   %Begin.type: type = fn_type @Begin [concrete]
// CHECK:STDOUT:   %Begin: %Begin.type = struct_value () [concrete]
// CHECK:STDOUT:   %End.type: type = fn_type @End [concrete]
// CHECK:STDOUT:   %End: %End.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.c2f: <witness> = custom_witness (%Iterator, %Sentinel, %Begin, %End), @CppRangeForIterate [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.ac7e09.1: type = fn_type @Iterator.Op.1 [concrete]
// CHECK:STDOUT:   %Iterator.Op.583fda.1: %Iterator.Op.type.ac7e09.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.1d3: <witness> = custom_witness (%ValueType, %Iterator.Op.583fda.1), @CppUnsafeDeref [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.ac7e09.2: type = fn_type @Iterator.Op.2 [concrete]
// CHECK:STDOUT:   %Iterator.Op.583fda.2: %Iterator.Op.type.ac7e09.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.21358e.1: <witness> = custom_witness (%Iterator.Op.583fda.2), @Destroy [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.ac7e09.3: type = fn_type @Iterator.Op.3 [concrete]
// CHECK:STDOUT:   %Iterator.Op.583fda.3: %Iterator.Op.type.ac7e09.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.a3150e.1: <witness> = custom_witness (%Iterator.Op.583fda.3), @Copy [concrete]
// CHECK:STDOUT:   %Equal.type: type = fn_type @Equal [concrete]
// CHECK:STDOUT:   %Equal: %Equal.type = struct_value () [concrete]
// CHECK:STDOUT:   %NotEqual.type: type = fn_type @NotEqual [concrete]
// CHECK:STDOUT:   %NotEqual: %NotEqual.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.aa4: <witness> = custom_witness (%Equal, %NotEqual), @EqWith, @EqWith(%Sentinel) [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.ac7e09.4: type = fn_type @Iterator.Op.4 [concrete]
// CHECK:STDOUT:   %Iterator.Op.583fda.4: %Iterator.Op.type.ac7e09.4 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.393: <witness> = custom_witness (%Iterator.Op.583fda.4), @Inc [concrete]
// CHECK:STDOUT:   %Sentinel.Op.type.ac7e09.1: type = fn_type @Sentinel.Op.1 [concrete]
// CHECK:STDOUT:   %Sentinel.Op.583fda.1: %Sentinel.Op.type.ac7e09.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.21358e.2: <witness> = custom_witness (%Sentinel.Op.583fda.1), @Destroy [concrete]
// CHECK:STDOUT:   %Sentinel.Op.type.ac7e09.2: type = fn_type @Sentinel.Op.2 [concrete]
// CHECK:STDOUT:   %Sentinel.Op.583fda.2: %Sentinel.Op.type.ac7e09.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.a3150e.2: <witness> = custom_witness (%Sentinel.Op.583fda.2), @Copy [concrete]
// CHECK:STDOUT:   %facet_value.b1e: %CppRangeForIterate_where.type = facet_value %ConstRange, (%custom_witness.fef, %custom_witness.c19, %custom_witness.c2f, %custom_witness.21358e.1, %custom_witness.a3150e.1, %custom_witness.1d3, %custom_witness.aa4, %custom_witness.393, %custom_witness.21358e.2, %custom_witness.a3150e.2) [concrete]
// CHECK:STDOUT:   %tuple.type.df8: type = tuple_type (%Iterator, %Sentinel) [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.8: type = fn_type @Destroy.Op.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.8: %Destroy.Op.type.1d8f74.8 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.df9cc1.8: <witness> = custom_witness (%Destroy.Op.1a2547.8), @Destroy [concrete]
// CHECK:STDOUT:   %Destroy.facet.777: %Destroy.type = facet_value %tuple.type.df8, (%custom_witness.df9cc1.8) [concrete]
// CHECK:STDOUT:   %Iterate.impl_witness.466: <witness> = impl_witness imports.%Iterate.impl_witness_table.344, @R.as_type.as.Iterate.impl(%facet_value.b1e) [concrete]
// CHECK:STDOUT:   %R.as_type.as.Iterate.impl.NewCursor.type.0b3: type = fn_type @R.as_type.as.Iterate.impl.NewCursor, @R.as_type.as.Iterate.impl(%facet_value.b1e) [concrete]
// CHECK:STDOUT:   %R.as_type.as.Iterate.impl.NewCursor.a27: %R.as_type.as.Iterate.impl.NewCursor.type.0b3 = struct_value () [concrete]
// CHECK:STDOUT:   %R.as_type.as.Iterate.impl.Next.type.f2b: type = fn_type @R.as_type.as.Iterate.impl.Next, @R.as_type.as.Iterate.impl(%facet_value.b1e) [concrete]
// CHECK:STDOUT:   %R.as_type.as.Iterate.impl.Next.bf3: %R.as_type.as.Iterate.impl.Next.type.f2b = struct_value () [concrete]
// CHECK:STDOUT:   %Iterate.facet.099: %Iterate.type = facet_value %ConstRange, (%Iterate.impl_witness.466) [concrete]
// CHECK:STDOUT:   %facet_value.00d: %Iterate_where.type.99d = facet_value %ConstRange, (%Iterate.impl_witness.466) [concrete]
// CHECK:STDOUT:   %r.param_patt.a73: %pattern_type.025 = value_param_pattern [concrete]
// CHECK:STDOUT:   %r.patt.0e2: %pattern_type.025 = at_binding_pattern r, %r.param_patt.a73 [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.e43: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.099) [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.be4: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.099) [concrete]
// CHECK:STDOUT:   %.da4e: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.e43, %Iterate.facet.099 [concrete]
// CHECK:STDOUT:   %R.as_type.as.Iterate.impl.NewCursor.specific_fn: <specific function> = specific_function %R.as_type.as.Iterate.impl.NewCursor.a27, @R.as_type.as.Iterate.impl.NewCursor(%facet_value.b1e) [concrete]
// CHECK:STDOUT:   %ptr.fc0: type = ptr_type %tuple.type.df8 [concrete]
// CHECK:STDOUT:   %.4b8: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.be4, %Iterate.facet.099 [concrete]
// CHECK:STDOUT:   %R.as_type.as.Iterate.impl.Next.specific_fn: <specific function> = specific_function %R.as_type.as.Iterate.impl.Next.bf3, @R.as_type.as.Iterate.impl.Next(%facet_value.b1e) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.589: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.777) [concrete]
// CHECK:STDOUT:   %.4e1: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.589, %Destroy.facet.777 [concrete]
// CHECK:STDOUT:   %complete_type.e35: <witness> = complete_type_witness %tuple.type.df8 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .N = %N
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.bb6: @Optional.%Optional.HasValue.type (%Optional.HasValue.type.e9e) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.HasValue (constants.%Optional.HasValue.6cd)]
// CHECK:STDOUT:   %Core.import_ref.daba: @Optional.%Optional.Get.type (%Optional.Get.type.bb9) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.Get (constants.%Optional.Get.fa1)]
// CHECK:STDOUT:   %Core.import_ref.0c6: type = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%DefaultOptionalStorage (constants.%DefaultOptionalStorage.d50)]
// CHECK:STDOUT:   %Core.import_ref.f4f: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.None.type (%T.as_type.as.OptionalStorage.impl.None.type.300) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.None (constants.%T.as_type.as.OptionalStorage.impl.None.f54)]
// CHECK:STDOUT:   %Core.import_ref.18f: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Some.type (%T.as_type.as.OptionalStorage.impl.Some.type.800) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Some (constants.%T.as_type.as.OptionalStorage.impl.Some.e83)]
// CHECK:STDOUT:   %Core.import_ref.c68: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Has.type (%T.as_type.as.OptionalStorage.impl.Has.type.4bc) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Has (constants.%T.as_type.as.OptionalStorage.impl.Has.dc2)]
// CHECK:STDOUT:   %Core.import_ref.a83: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Get.type (%T.as_type.as.OptionalStorage.impl.Get.type.029) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Get (constants.%T.as_type.as.OptionalStorage.impl.Get.cb5)]
// CHECK:STDOUT:   %Core.import_ref.aaa = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], unloaded
// CHECK:STDOUT:   %OptionalStorage.impl_witness_table.655 = impl_witness_table (%Core.import_ref.0c6, %Core.import_ref.f4f, %Core.import_ref.18f, %Core.import_ref.c68, %Core.import_ref.a83, %Core.import_ref.aaa), @T.as_type.as.OptionalStorage.impl [concrete]
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
// CHECK:STDOUT:   %Core.import_ref.301: %facet_type = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @R.as_type.as.Iterate.impl.%facet_value (constants.%facet_value.194)]
// CHECK:STDOUT:   %Core.import_ref.89f: %Destroy.type = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @R.as_type.as.Iterate.impl.%Destroy.facet (constants.%Destroy.facet.593)]
// CHECK:STDOUT:   %Core.import_ref.3e6: @R.as_type.as.Iterate.impl.%R.as_type.as.Iterate.impl.NewCursor.type (%R.as_type.as.Iterate.impl.NewCursor.type.36f) = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @R.as_type.as.Iterate.impl.%R.as_type.as.Iterate.impl.NewCursor (constants.%R.as_type.as.Iterate.impl.NewCursor.29b)]
// CHECK:STDOUT:   %Core.import_ref.85d: @R.as_type.as.Iterate.impl.%R.as_type.as.Iterate.impl.Next.type (%R.as_type.as.Iterate.impl.Next.type.1f7) = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @R.as_type.as.Iterate.impl.%R.as_type.as.Iterate.impl.Next (constants.%R.as_type.as.Iterate.impl.Next.0d6)]
// CHECK:STDOUT:   %Iterate.impl_witness_table.344 = impl_witness_table (%Core.import_ref.301, %Core.import_ref.89f, %Core.import_ref.3e6, %Core.import_ref.85d), @R.as_type.as.Iterate.impl [concrete]
// CHECK:STDOUT:   %operator_PlusEqual__carbon_thunk.decl: %operator_PlusEqual__carbon_thunk.type = fn_decl @operator_PlusEqual__carbon_thunk [concrete = constants.%operator_PlusEqual__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ConstRange.decl: type = class_decl @ConstRange [concrete = constants.%ConstRange] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @TestIterate(%R.loc27_17.2: %Iterate_where.type.99d) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Iterate.lookup_impl_witness: <witness> = lookup_impl_witness %R.loc27_17.1, @Iterate [symbolic = %Iterate.lookup_impl_witness (constants.%Iterate.lookup_impl_witness.8b4)]
// CHECK:STDOUT:   %Iterate.facet.loc31_31.5: %Iterate.type = facet_value %R.as_type.loc27_74.1, (%Iterate.lookup_impl_witness) [symbolic = %Iterate.facet.loc31_31.5 (constants.%Iterate.facet.b19)]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.loc31_31.5) [symbolic = %Iterate.WithSelf.NewCursor.type (constants.%Iterate.WithSelf.NewCursor.type.217)]
// CHECK:STDOUT:   %.loc31_31.20: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type, %Iterate.facet.loc31_31.5 [symbolic = %.loc31_31.20 (constants.%.a27)]
// CHECK:STDOUT:   %impl.elem2.loc31_31.2: @TestIterate.%.loc31_31.20 (%.a27) = impl_witness_access %Iterate.lookup_impl_witness, element2 [symbolic = %impl.elem2.loc31_31.2 (constants.%impl.elem2.7c0)]
// CHECK:STDOUT:   %specific_impl_fn.loc31_31.4: <specific function> = specific_impl_function %impl.elem2.loc31_31.2, @Iterate.WithSelf.NewCursor(%Iterate.facet.loc31_31.5) [symbolic = %specific_impl_fn.loc31_31.4 (constants.%specific_impl_fn.e23)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %ptr: type = ptr_type %as_type [symbolic = %ptr (constants.%ptr.530)]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.loc31_31.5) [symbolic = %Iterate.WithSelf.Next.type (constants.%Iterate.WithSelf.Next.type.10a)]
// CHECK:STDOUT:   %.loc31_31.21: type = fn_type_with_self_type %Iterate.WithSelf.Next.type, %Iterate.facet.loc31_31.5 [symbolic = %.loc31_31.21 (constants.%.df57)]
// CHECK:STDOUT:   %impl.elem3.loc31_31.2: @TestIterate.%.loc31_31.21 (%.df57) = impl_witness_access %Iterate.lookup_impl_witness, element3 [symbolic = %impl.elem3.loc31_31.2 (constants.%impl.elem3.199)]
// CHECK:STDOUT:   %specific_impl_fn.loc31_31.5: <specific function> = specific_impl_function %impl.elem3.loc31_31.2, @Iterate.WithSelf.Next(%Iterate.facet.loc31_31.5) [symbolic = %specific_impl_fn.loc31_31.5 (constants.%specific_impl_fn.169)]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%impl.elem1) [symbolic = %Destroy.WithSelf.Op.type (constants.%Destroy.WithSelf.Op.type.e7e)]
// CHECK:STDOUT:   %.loc31_31.22: type = fn_type_with_self_type %Destroy.WithSelf.Op.type, %impl.elem1 [symbolic = %.loc31_31.22 (constants.%.bc1)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %impl.elem1, @Destroy [symbolic = %Destroy.lookup_impl_witness (constants.%Destroy.lookup_impl_witness.f37)]
// CHECK:STDOUT:   %impl.elem0.loc31_31.2: @TestIterate.%.loc31_31.22 (%.bc1) = impl_witness_access %Destroy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc31_31.2 (constants.%impl.elem0.c7c)]
// CHECK:STDOUT:   %specific_impl_fn.loc31_31.6: <specific function> = specific_impl_function %impl.elem0.loc31_31.2, @Destroy.WithSelf.Op(%impl.elem1) [symbolic = %specific_impl_fn.loc31_31.6 (constants.%specific_impl_fn.cd3)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%r.param: @TestIterate.%R.as_type.loc27_74.1 (%R.as_type.a62)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %i.patt: %pattern_type.e41 = value_binding_pattern i [concrete = constants.%i.patt]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %r.ref: @TestIterate.%R.as_type.loc27_74.1 (%R.as_type.a62) = name_ref r, %r
// CHECK:STDOUT:     %impl.elem2.loc31_31.1: @TestIterate.%.loc31_31.20 (%.a27) = impl_witness_access constants.%Iterate.lookup_impl_witness.8b4, element2 [symbolic = %impl.elem2.loc31_31.2 (constants.%impl.elem2.7c0)]
// CHECK:STDOUT:     %bound_method.loc31_31.1: <bound method> = bound_method %r.ref, %impl.elem2.loc31_31.1
// CHECK:STDOUT:     %Iterate.facet.loc31_31.1: %Iterate.type = facet_value constants.%R.as_type.a62, (constants.%Iterate.lookup_impl_witness.8b4) [symbolic = %Iterate.facet.loc31_31.5 (constants.%Iterate.facet.b19)]
// CHECK:STDOUT:     %.loc31_31.1: %Iterate.type = converted constants.%R.as_type.a62, %Iterate.facet.loc31_31.1 [symbolic = %Iterate.facet.loc31_31.5 (constants.%Iterate.facet.b19)]
// CHECK:STDOUT:     %Iterate.facet.loc31_31.2: %Iterate.type = facet_value constants.%R.as_type.a62, (constants.%Iterate.lookup_impl_witness.8b4) [symbolic = %Iterate.facet.loc31_31.5 (constants.%Iterate.facet.b19)]
// CHECK:STDOUT:     %.loc31_31.2: %Iterate.type = converted constants.%R.as_type.a62, %Iterate.facet.loc31_31.2 [symbolic = %Iterate.facet.loc31_31.5 (constants.%Iterate.facet.b19)]
// CHECK:STDOUT:     %specific_impl_fn.loc31_31.1: <specific function> = specific_impl_function %impl.elem2.loc31_31.1, @Iterate.WithSelf.NewCursor(constants.%Iterate.facet.b19) [symbolic = %specific_impl_fn.loc31_31.4 (constants.%specific_impl_fn.e23)]
// CHECK:STDOUT:     %bound_method.loc31_31.2: <bound method> = bound_method %r.ref, %specific_impl_fn.loc31_31.1
// CHECK:STDOUT:     %var: ref @TestIterate.%as_type (%as_type.862) = var invalid
// CHECK:STDOUT:     %Iterate.WithSelf.NewCursor.call: init @TestIterate.%as_type (%as_type.862) to %var = call %bound_method.loc31_31.2(%r.ref)
// CHECK:STDOUT:     assign %var, %Iterate.WithSelf.NewCursor.call
// CHECK:STDOUT:     br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.next:
// CHECK:STDOUT:     %addr.loc31: @TestIterate.%ptr (%ptr.530) = addr_of %var
// CHECK:STDOUT:     %impl.elem3.loc31_31.1: @TestIterate.%.loc31_31.21 (%.df57) = impl_witness_access constants.%Iterate.lookup_impl_witness.8b4, element3 [symbolic = %impl.elem3.loc31_31.2 (constants.%impl.elem3.199)]
// CHECK:STDOUT:     %bound_method.loc31_31.3: <bound method> = bound_method %r.ref, %impl.elem3.loc31_31.1
// CHECK:STDOUT:     %Iterate.facet.loc31_31.3: %Iterate.type = facet_value constants.%R.as_type.a62, (constants.%Iterate.lookup_impl_witness.8b4) [symbolic = %Iterate.facet.loc31_31.5 (constants.%Iterate.facet.b19)]
// CHECK:STDOUT:     %.loc31_31.3: %Iterate.type = converted constants.%R.as_type.a62, %Iterate.facet.loc31_31.3 [symbolic = %Iterate.facet.loc31_31.5 (constants.%Iterate.facet.b19)]
// CHECK:STDOUT:     %Iterate.facet.loc31_31.4: %Iterate.type = facet_value constants.%R.as_type.a62, (constants.%Iterate.lookup_impl_witness.8b4) [symbolic = %Iterate.facet.loc31_31.5 (constants.%Iterate.facet.b19)]
// CHECK:STDOUT:     %.loc31_31.4: %Iterate.type = converted constants.%R.as_type.a62, %Iterate.facet.loc31_31.4 [symbolic = %Iterate.facet.loc31_31.5 (constants.%Iterate.facet.b19)]
// CHECK:STDOUT:     %.loc31_31.5: %Destroy.type = converted constants.%as_type.862, constants.%impl.elem1.46a [symbolic = %impl.elem1 (constants.%impl.elem1.46a)]
// CHECK:STDOUT:     %.loc31_31.6: %Destroy.type = converted constants.%as_type.862, constants.%impl.elem1.46a [symbolic = %impl.elem1 (constants.%impl.elem1.46a)]
// CHECK:STDOUT:     %specific_impl_fn.loc31_31.2: <specific function> = specific_impl_function %impl.elem3.loc31_31.1, @Iterate.WithSelf.Next(constants.%Iterate.facet.b19) [symbolic = %specific_impl_fn.loc31_31.5 (constants.%specific_impl_fn.169)]
// CHECK:STDOUT:     %bound_method.loc31_31.4: <bound method> = bound_method %r.ref, %specific_impl_fn.loc31_31.2
// CHECK:STDOUT:     %.loc31_31.7: ref %Optional.552 = temporary_storage
// CHECK:STDOUT:     %Iterate.WithSelf.Next.call: init %Optional.552 to %.loc31_31.7 = call %bound_method.loc31_31.4(%r.ref, %addr.loc31)
// CHECK:STDOUT:     %.loc31_31.8: ref %Optional.552 = temporary %.loc31_31.7, %Iterate.WithSelf.Next.call
// CHECK:STDOUT:     %.loc31_31.9: %Optional.HasValue.type.109 = specific_constant imports.%Core.import_ref.bb6, @Optional(constants.%OptionalStorage.facet.7ce) [concrete = constants.%Optional.HasValue.605]
// CHECK:STDOUT:     %HasValue.ref: %Optional.HasValue.type.109 = name_ref HasValue, %.loc31_31.9 [concrete = constants.%Optional.HasValue.605]
// CHECK:STDOUT:     %Optional.HasValue.bound: <bound method> = bound_method %.loc31_31.8, %HasValue.ref
// CHECK:STDOUT:     %Optional.HasValue.specific_fn: <specific function> = specific_function %HasValue.ref, @Optional.HasValue(constants.%OptionalStorage.facet.7ce) [concrete = constants.%Optional.HasValue.specific_fn]
// CHECK:STDOUT:     %bound_method.loc31_31.5: <bound method> = bound_method %.loc31_31.8, %Optional.HasValue.specific_fn
// CHECK:STDOUT:     %.loc31_31.10: %Optional.552 = acquire_value %.loc31_31.8
// CHECK:STDOUT:     %Optional.HasValue.call: init bool = call %bound_method.loc31_31.5(%.loc31_31.10)
// CHECK:STDOUT:     %.loc31_31.11: bool = value_of_initializer %Optional.HasValue.call
// CHECK:STDOUT:     %.loc31_31.12: bool = converted %Optional.HasValue.call, %.loc31_31.11
// CHECK:STDOUT:     if %.loc31_31.12 br !for.body else br !for.done
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.body:
// CHECK:STDOUT:     %.loc31_31.13: %Optional.Get.type.135 = specific_constant imports.%Core.import_ref.daba, @Optional(constants.%OptionalStorage.facet.7ce) [concrete = constants.%Optional.Get.5aa]
// CHECK:STDOUT:     %Get.ref: %Optional.Get.type.135 = name_ref Get, %.loc31_31.13 [concrete = constants.%Optional.Get.5aa]
// CHECK:STDOUT:     %Optional.Get.bound: <bound method> = bound_method %.loc31_31.8, %Get.ref
// CHECK:STDOUT:     %Optional.Get.specific_fn: <specific function> = specific_function %Get.ref, @Optional.Get(constants.%OptionalStorage.facet.7ce) [concrete = constants.%Optional.Get.specific_fn]
// CHECK:STDOUT:     %bound_method.loc31_31.6: <bound method> = bound_method %.loc31_31.8, %Optional.Get.specific_fn
// CHECK:STDOUT:     %.loc31_31.14: ref %ValueType = temporary_storage
// CHECK:STDOUT:     %.loc31_31.15: %Optional.552 = acquire_value %.loc31_31.8
// CHECK:STDOUT:     %Optional.Get.call: init %ValueType to %.loc31_31.14 = call %bound_method.loc31_31.6(%.loc31_31.15)
// CHECK:STDOUT:     %.loc31_31.16: ref %ValueType = temporary %.loc31_31.14, %Optional.Get.call
// CHECK:STDOUT:     %.loc31_31.17: %ValueType = acquire_value %.loc31_31.16
// CHECK:STDOUT:     %.loc31_16: type = splice_block %ValueType.ref.loc31 [concrete = constants.%ValueType] {
// CHECK:STDOUT:       %Cpp.ref.loc31: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:       %N.ref.loc31: <namespace> = name_ref N, imports.%N [concrete = imports.%N]
// CHECK:STDOUT:       %ValueType.ref.loc31: type = name_ref ValueType, imports.%ValueType.decl [concrete = constants.%ValueType]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %i: %ValueType = wrapper_binding i, %.loc31_31.17
// CHECK:STDOUT:     %sum.ref: ref %ValueType = name_ref sum, %sum
// CHECK:STDOUT:     %i.ref: %ValueType = name_ref i, %i
// CHECK:STDOUT:     %.loc32_12: ref %ValueType = value_as_ref %i.ref
// CHECK:STDOUT:     %addr.loc32: %ptr.9d3 = addr_of %.loc32_12
// CHECK:STDOUT:     %.loc32_9.1: %ptr.32e = as_compatible %addr.loc32
// CHECK:STDOUT:     %.loc32_9.2: %ptr.32e = converted %addr.loc32, %.loc32_9.1
// CHECK:STDOUT:     %operator_PlusEqual__carbon_thunk.call: ref %ValueType = call imports.%operator_PlusEqual__carbon_thunk.decl(%sum.ref, %.loc32_9.2)
// CHECK:STDOUT:     br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.done:
// CHECK:STDOUT:     %ValueType.Op.bound.loc31: <bound method> = bound_method %.loc31_31.16, constants.%ValueType.Op.4c94b9.1
// CHECK:STDOUT:     %Op.ref.loc31: %ValueType.cpp_destructor.type = name_ref Op, imports.%ValueType.cpp_destructor.decl [concrete = constants.%ValueType.cpp_destructor]
// CHECK:STDOUT:     %ValueType.cpp_destructor.bound.loc31: <bound method> = bound_method %.loc31_31.16, %Op.ref.loc31
// CHECK:STDOUT:     %ValueType.cpp_destructor.call.loc31: init %empty_tuple.type = call %ValueType.cpp_destructor.bound.loc31(%.loc31_31.16)
// CHECK:STDOUT:     %Destroy.Op.bound: <bound method> = bound_method %.loc31_31.8, constants.%Destroy.Op.1a2547.7
// CHECK:STDOUT:     %Destroy.Op.call: init %empty_tuple.type = call %Destroy.Op.bound(%.loc31_31.8)
// CHECK:STDOUT:     %impl.elem0.loc31_31.1: @TestIterate.%.loc31_31.22 (%.bc1) = impl_witness_access constants.%Destroy.lookup_impl_witness.f37, element0 [symbolic = %impl.elem0.loc31_31.2 (constants.%impl.elem0.c7c)]
// CHECK:STDOUT:     %bound_method.loc31_31.7: <bound method> = bound_method %var, %impl.elem0.loc31_31.1
// CHECK:STDOUT:     %.loc31_31.18: %Destroy.type = converted constants.%as_type.862, constants.%impl.elem1.46a [symbolic = %impl.elem1 (constants.%impl.elem1.46a)]
// CHECK:STDOUT:     %.loc31_31.19: %Destroy.type = converted constants.%as_type.862, constants.%impl.elem1.46a [symbolic = %impl.elem1 (constants.%impl.elem1.46a)]
// CHECK:STDOUT:     %specific_impl_fn.loc31_31.3: <specific function> = specific_impl_function %impl.elem0.loc31_31.1, @Destroy.WithSelf.Op(constants.%impl.elem1.46a) [symbolic = %specific_impl_fn.loc31_31.6 (constants.%specific_impl_fn.cd3)]
// CHECK:STDOUT:     %bound_method.loc31_31.8: <bound method> = bound_method %var, %specific_impl_fn.loc31_31.3
// CHECK:STDOUT:     %Destroy.WithSelf.Op.call: init %empty_tuple.type = call %bound_method.loc31_31.8(%var)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc31_31.1(%self.param: ref %.b36) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc31_31.2(%self.param: ref %MaybeUnformed.669) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc31_31.3(%self.param: ref bool) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc31_31.4(%self.param: ref %struct_type.value.has_value.5a1) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc31_31.5(%self.param: ref %DefaultOptionalStorage.257) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc31_31.6(%self.param: ref %Optional.552) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestRangeFor(%c.param: %ConstRange) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %i.patt: %pattern_type.e41 = value_binding_pattern i [concrete = constants.%i.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c.ref: %ConstRange = name_ref c, %c
// CHECK:STDOUT:   %impl.elem2: %.da4e = impl_witness_access constants.%Iterate.impl_witness.466, element2 [concrete = constants.%R.as_type.as.Iterate.impl.NewCursor.a27]
// CHECK:STDOUT:   %bound_method.loc45_31.1: <bound method> = bound_method %c.ref, %impl.elem2
// CHECK:STDOUT:   %facet_value.loc45_31.1: %CppRangeForIterate_where.type = facet_value constants.%ConstRange, (constants.%custom_witness.fef, constants.%custom_witness.c19, constants.%custom_witness.c2f, constants.%custom_witness.21358e.1, constants.%custom_witness.a3150e.1, constants.%custom_witness.1d3, constants.%custom_witness.aa4, constants.%custom_witness.393, constants.%custom_witness.21358e.2, constants.%custom_witness.a3150e.2) [concrete = constants.%facet_value.b1e]
// CHECK:STDOUT:   %.loc45_31.1: %CppRangeForIterate_where.type = converted constants.%ConstRange, %facet_value.loc45_31.1 [concrete = constants.%facet_value.b1e]
// CHECK:STDOUT:   %facet_value.loc45_31.2: %CppRangeForIterate_where.type = facet_value constants.%ConstRange, (constants.%custom_witness.fef, constants.%custom_witness.c19, constants.%custom_witness.c2f, constants.%custom_witness.21358e.1, constants.%custom_witness.a3150e.1, constants.%custom_witness.1d3, constants.%custom_witness.aa4, constants.%custom_witness.393, constants.%custom_witness.21358e.2, constants.%custom_witness.a3150e.2) [concrete = constants.%facet_value.b1e]
// CHECK:STDOUT:   %.loc45_31.2: %CppRangeForIterate_where.type = converted constants.%ConstRange, %facet_value.loc45_31.2 [concrete = constants.%facet_value.b1e]
// CHECK:STDOUT:   %specific_fn.loc45_31.1: <specific function> = specific_function %impl.elem2, @R.as_type.as.Iterate.impl.NewCursor(constants.%facet_value.b1e) [concrete = constants.%R.as_type.as.Iterate.impl.NewCursor.specific_fn]
// CHECK:STDOUT:   %bound_method.loc45_31.2: <bound method> = bound_method %c.ref, %specific_fn.loc45_31.1
// CHECK:STDOUT:   %var: ref %tuple.type.df8 = var invalid
// CHECK:STDOUT:   %R.as_type.as.Iterate.impl.NewCursor.call: init %tuple.type.df8 to %var = call %bound_method.loc45_31.2(%c.ref)
// CHECK:STDOUT:   assign %var, %R.as_type.as.Iterate.impl.NewCursor.call
// CHECK:STDOUT:   br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT: !for.next:
// CHECK:STDOUT:   %addr.loc45: %ptr.fc0 = addr_of %var
// CHECK:STDOUT:   %impl.elem3: %.4b8 = impl_witness_access constants.%Iterate.impl_witness.466, element3 [concrete = constants.%R.as_type.as.Iterate.impl.Next.bf3]
// CHECK:STDOUT:   %bound_method.loc45_31.3: <bound method> = bound_method %c.ref, %impl.elem3
// CHECK:STDOUT:   %facet_value.loc45_31.3: %CppRangeForIterate_where.type = facet_value constants.%ConstRange, (constants.%custom_witness.fef, constants.%custom_witness.c19, constants.%custom_witness.c2f, constants.%custom_witness.21358e.1, constants.%custom_witness.a3150e.1, constants.%custom_witness.1d3, constants.%custom_witness.aa4, constants.%custom_witness.393, constants.%custom_witness.21358e.2, constants.%custom_witness.a3150e.2) [concrete = constants.%facet_value.b1e]
// CHECK:STDOUT:   %.loc45_31.3: %CppRangeForIterate_where.type = converted constants.%ConstRange, %facet_value.loc45_31.3 [concrete = constants.%facet_value.b1e]
// CHECK:STDOUT:   %facet_value.loc45_31.4: %CppRangeForIterate_where.type = facet_value constants.%ConstRange, (constants.%custom_witness.fef, constants.%custom_witness.c19, constants.%custom_witness.c2f, constants.%custom_witness.21358e.1, constants.%custom_witness.a3150e.1, constants.%custom_witness.1d3, constants.%custom_witness.aa4, constants.%custom_witness.393, constants.%custom_witness.21358e.2, constants.%custom_witness.a3150e.2) [concrete = constants.%facet_value.b1e]
// CHECK:STDOUT:   %.loc45_31.4: %CppRangeForIterate_where.type = converted constants.%ConstRange, %facet_value.loc45_31.4 [concrete = constants.%facet_value.b1e]
// CHECK:STDOUT:   %specific_fn.loc45_31.2: <specific function> = specific_function %impl.elem3, @R.as_type.as.Iterate.impl.Next(constants.%facet_value.b1e) [concrete = constants.%R.as_type.as.Iterate.impl.Next.specific_fn]
// CHECK:STDOUT:   %bound_method.loc45_31.4: <bound method> = bound_method %c.ref, %specific_fn.loc45_31.2
// CHECK:STDOUT:   %.loc45_31.5: ref %Optional.552 = temporary_storage
// CHECK:STDOUT:   %R.as_type.as.Iterate.impl.Next.call: init %Optional.552 to %.loc45_31.5 = call %bound_method.loc45_31.4(%c.ref, %addr.loc45)
// CHECK:STDOUT:   %.loc45_31.6: ref %Optional.552 = temporary %.loc45_31.5, %R.as_type.as.Iterate.impl.Next.call
// CHECK:STDOUT:   %.loc45_31.7: %Optional.HasValue.type.109 = specific_constant imports.%Core.import_ref.bb6, @Optional(constants.%OptionalStorage.facet.7ce) [concrete = constants.%Optional.HasValue.605]
// CHECK:STDOUT:   %HasValue.ref: %Optional.HasValue.type.109 = name_ref HasValue, %.loc45_31.7 [concrete = constants.%Optional.HasValue.605]
// CHECK:STDOUT:   %Optional.HasValue.bound: <bound method> = bound_method %.loc45_31.6, %HasValue.ref
// CHECK:STDOUT:   %Optional.HasValue.specific_fn: <specific function> = specific_function %HasValue.ref, @Optional.HasValue(constants.%OptionalStorage.facet.7ce) [concrete = constants.%Optional.HasValue.specific_fn]
// CHECK:STDOUT:   %bound_method.loc45_31.5: <bound method> = bound_method %.loc45_31.6, %Optional.HasValue.specific_fn
// CHECK:STDOUT:   %.loc45_31.8: %Optional.552 = acquire_value %.loc45_31.6
// CHECK:STDOUT:   %Optional.HasValue.call: init bool = call %bound_method.loc45_31.5(%.loc45_31.8)
// CHECK:STDOUT:   %.loc45_31.9: bool = value_of_initializer %Optional.HasValue.call
// CHECK:STDOUT:   %.loc45_31.10: bool = converted %Optional.HasValue.call, %.loc45_31.9
// CHECK:STDOUT:   if %.loc45_31.10 br !for.body else br !for.done
// CHECK:STDOUT:
// CHECK:STDOUT: !for.body:
// CHECK:STDOUT:   %.loc45_31.11: %Optional.Get.type.135 = specific_constant imports.%Core.import_ref.daba, @Optional(constants.%OptionalStorage.facet.7ce) [concrete = constants.%Optional.Get.5aa]
// CHECK:STDOUT:   %Get.ref: %Optional.Get.type.135 = name_ref Get, %.loc45_31.11 [concrete = constants.%Optional.Get.5aa]
// CHECK:STDOUT:   %Optional.Get.bound: <bound method> = bound_method %.loc45_31.6, %Get.ref
// CHECK:STDOUT:   %Optional.Get.specific_fn: <specific function> = specific_function %Get.ref, @Optional.Get(constants.%OptionalStorage.facet.7ce) [concrete = constants.%Optional.Get.specific_fn]
// CHECK:STDOUT:   %bound_method.loc45_31.6: <bound method> = bound_method %.loc45_31.6, %Optional.Get.specific_fn
// CHECK:STDOUT:   %.loc45_31.12: ref %ValueType = temporary_storage
// CHECK:STDOUT:   %.loc45_31.13: %Optional.552 = acquire_value %.loc45_31.6
// CHECK:STDOUT:   %Optional.Get.call: init %ValueType to %.loc45_31.12 = call %bound_method.loc45_31.6(%.loc45_31.13)
// CHECK:STDOUT:   %.loc45_31.14: ref %ValueType = temporary %.loc45_31.12, %Optional.Get.call
// CHECK:STDOUT:   %.loc45_31.15: %ValueType = acquire_value %.loc45_31.14
// CHECK:STDOUT:   %.loc45_16: type = splice_block %ValueType.ref.loc45 [concrete = constants.%ValueType] {
// CHECK:STDOUT:     %Cpp.ref.loc45: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %N.ref.loc45: <namespace> = name_ref N, imports.%N [concrete = imports.%N]
// CHECK:STDOUT:     %ValueType.ref.loc45: type = name_ref ValueType, imports.%ValueType.decl [concrete = constants.%ValueType]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %i: %ValueType = wrapper_binding i, %.loc45_31.15
// CHECK:STDOUT:   %sum.ref: ref %ValueType = name_ref sum, %sum
// CHECK:STDOUT:   %i.ref: %ValueType = name_ref i, %i
// CHECK:STDOUT:   %.loc46_12: ref %ValueType = value_as_ref %i.ref
// CHECK:STDOUT:   %addr.loc46: %ptr.9d3 = addr_of %.loc46_12
// CHECK:STDOUT:   %.loc46_9.1: %ptr.32e = as_compatible %addr.loc46
// CHECK:STDOUT:   %.loc46_9.2: %ptr.32e = converted %addr.loc46, %.loc46_9.1
// CHECK:STDOUT:   %operator_PlusEqual__carbon_thunk.call: ref %ValueType = call imports.%operator_PlusEqual__carbon_thunk.decl(%sum.ref, %.loc46_9.2)
// CHECK:STDOUT:   br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT: !for.done:
// CHECK:STDOUT:   %ValueType.Op.bound.loc45: <bound method> = bound_method %.loc45_31.14, constants.%ValueType.Op.4c94b9.1
// CHECK:STDOUT:   %Op.ref.loc45: %ValueType.cpp_destructor.type = name_ref Op, imports.%ValueType.cpp_destructor.decl [concrete = constants.%ValueType.cpp_destructor]
// CHECK:STDOUT:   %ValueType.cpp_destructor.bound.loc45: <bound method> = bound_method %.loc45_31.14, %Op.ref.loc45
// CHECK:STDOUT:   %ValueType.cpp_destructor.call.loc45: init %empty_tuple.type = call %ValueType.cpp_destructor.bound.loc45(%.loc45_31.14)
// CHECK:STDOUT:   %Destroy.Op.bound.loc45_31.1: <bound method> = bound_method %.loc45_31.6, constants.%Destroy.Op.1a2547.7
// CHECK:STDOUT:   %Destroy.Op.call.loc45_31.1: init %empty_tuple.type = call %Destroy.Op.bound.loc45_31.1(%.loc45_31.6)
// CHECK:STDOUT:   %Destroy.Op.bound.loc45_31.2: <bound method> = bound_method %var, constants.%Destroy.Op.1a2547.8
// CHECK:STDOUT:   %Destroy.Op.call.loc45_31.2: init %empty_tuple.type = call %Destroy.Op.bound.loc45_31.2(%var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestIterate(constants.%R.623) {
// CHECK:STDOUT:   %R.patt.loc27_17.2 => constants.%R.patt.324
// CHECK:STDOUT:   %R.loc27_17.1 => constants.%R.623
// CHECK:STDOUT:   %R.as_type.loc27_74.1 => constants.%R.as_type.a62
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.725
// CHECK:STDOUT:   %r.param_patt.loc27_72.2 => constants.%r.param_patt.162
// CHECK:STDOUT:   %r.patt.loc27_72.2 => constants.%r.patt.15b
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestIterate(constants.%facet_value.00d) {
// CHECK:STDOUT:   %R.patt.loc27_17.2 => constants.%R.patt.324
// CHECK:STDOUT:   %R.loc27_17.1 => constants.%facet_value.00d
// CHECK:STDOUT:   %R.as_type.loc27_74.1 => constants.%ConstRange
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.025
// CHECK:STDOUT:   %r.param_patt.loc27_72.2 => constants.%r.param_patt.a73
// CHECK:STDOUT:   %r.patt.loc27_72.2 => constants.%r.patt.0e2
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc27 => constants.%complete_type.357
// CHECK:STDOUT:   %Iterate.lookup_impl_witness => constants.%Iterate.impl_witness.466
// CHECK:STDOUT:   %Iterate.facet.loc31_31.5 => constants.%Iterate.facet.099
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type => constants.%Iterate.WithSelf.NewCursor.type.e43
// CHECK:STDOUT:   %.loc31_31.20 => constants.%.da4e
// CHECK:STDOUT:   %impl.elem2.loc31_31.2 => constants.%R.as_type.as.Iterate.impl.NewCursor.a27
// CHECK:STDOUT:   %specific_impl_fn.loc31_31.4 => constants.%R.as_type.as.Iterate.impl.NewCursor.specific_fn
// CHECK:STDOUT:   %impl.elem1 => constants.%Destroy.facet.777
// CHECK:STDOUT:   %as_type => constants.%tuple.type.df8
// CHECK:STDOUT:   %require_complete.1 => constants.%complete_type.e35
// CHECK:STDOUT:   %ptr => constants.%ptr.fc0
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type => constants.%Iterate.WithSelf.Next.type.be4
// CHECK:STDOUT:   %.loc31_31.21 => constants.%.4b8
// CHECK:STDOUT:   %impl.elem3.loc31_31.2 => constants.%R.as_type.as.Iterate.impl.Next.bf3
// CHECK:STDOUT:   %specific_impl_fn.loc31_31.5 => constants.%R.as_type.as.Iterate.impl.Next.specific_fn
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.589
// CHECK:STDOUT:   %.loc31_31.22 => constants.%.4e1
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.df9cc1.8
// CHECK:STDOUT:   %impl.elem0.loc31_31.2 => constants.%Destroy.Op.1a2547.8
// CHECK:STDOUT:   %specific_impl_fn.loc31_31.6 => constants.%Destroy.Op.1a2547.8
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
// CHECK:STDOUT:   %T.220: %OptionalStorage.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Optional.Get.type.bb9: type = fn_type @Optional.Get, @Optional(%T.220) [symbolic]
// CHECK:STDOUT:   %Optional.Get.fa1: %Optional.Get.type.bb9 = struct_value () [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.type.e9e: type = fn_type @Optional.HasValue, @Optional(%T.220) [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.6cd: %Optional.HasValue.type.e9e = struct_value () [symbolic]
// CHECK:STDOUT:   %T.283: %facet_type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.d50: type = class_type @DefaultOptionalStorage, @DefaultOptionalStorage(%T.283) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Get.type.029: type = fn_type @T.as_type.as.OptionalStorage.impl.Get, @T.as_type.as.OptionalStorage.impl(%T.283) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Get.cb5: %T.as_type.as.OptionalStorage.impl.Get.type.029 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Has.type.4bc: type = fn_type @T.as_type.as.OptionalStorage.impl.Has, @T.as_type.as.OptionalStorage.impl(%T.283) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Has.dc2: %T.as_type.as.OptionalStorage.impl.Has.type.4bc = struct_value () [symbolic]
// CHECK:STDOUT:   %.Self.35d: %Iterate.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %Iterate.lookup_impl_witness.b37: <witness> = lookup_impl_witness %.Self.35d, @Iterate [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.903: %facet_type = impl_witness_access %Iterate.lookup_impl_witness.b37, element0 [symbolic_self]
// CHECK:STDOUT:   %ValueType: type = class_type @ValueType [concrete]
// CHECK:STDOUT:   %pattern_type.e41: type = pattern_type %ValueType [concrete]
// CHECK:STDOUT:   %ValueType.cpp_destructor.type: type = fn_type @ValueType.cpp_destructor [concrete]
// CHECK:STDOUT:   %ValueType.cpp_destructor: %ValueType.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT:   %ValueType.Op.type.79266b.1: type = fn_type @ValueType.Op.1 [concrete]
// CHECK:STDOUT:   %ValueType.Op.4c94b9.1: %ValueType.Op.type.79266b.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.fef: <witness> = custom_witness (%ValueType.Op.4c94b9.1), @Destroy [concrete]
// CHECK:STDOUT:   %const.773: type = const_type %ValueType [concrete]
// CHECK:STDOUT:   %ptr.32e: type = ptr_type %const.773 [concrete]
// CHECK:STDOUT:   %ptr.9d3: type = ptr_type %ValueType [concrete]
// CHECK:STDOUT:   %ValueType.Op.type.79266b.2: type = fn_type @ValueType.Op.2 [concrete]
// CHECK:STDOUT:   %ValueType.Op.4c94b9.2: %ValueType.Op.type.79266b.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.c19: <witness> = custom_witness (%ValueType.Op.4c94b9.2), @Copy [concrete]
// CHECK:STDOUT:   %facet_value.ccb: %facet_type = facet_value %ValueType, (%custom_witness.fef, %custom_witness.c19) [concrete]
// CHECK:STDOUT:   %Iterate_where.type.99d: type = facet_type <@Iterate where %impl.elem0.903 = %facet_value.ccb> [concrete]
// CHECK:STDOUT:   %pattern_type.3f6: type = pattern_type %Iterate_where.type.99d [concrete]
// CHECK:STDOUT:   %R.patt.324: %pattern_type.3f6 = symbolic_binding_pattern R, 0 [symbolic]
// CHECK:STDOUT:   %R.623: %Iterate_where.type.99d = symbolic_binding R, 0 [symbolic]
// CHECK:STDOUT:   %R.as_type.a62: type = facet_access_type %R.623 [symbolic]
// CHECK:STDOUT:   %pattern_type.725: type = pattern_type %R.as_type.a62 [symbolic]
// CHECK:STDOUT:   %r.param_patt: %pattern_type.725 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %r.patt: %pattern_type.725 = at_binding_pattern r, %r.param_patt [symbolic]
// CHECK:STDOUT:   %i.patt: %pattern_type.e41 = value_binding_pattern i [concrete]
// CHECK:STDOUT:   %Iterate.lookup_impl_witness.8b4: <witness> = lookup_impl_witness %R.623, @Iterate [symbolic]
// CHECK:STDOUT:   %Iterate.facet: %Iterate.type = facet_value %R.as_type.a62, (%Iterate.lookup_impl_witness.8b4) [symbolic]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.217: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet) [symbolic]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.10a: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet) [symbolic]
// CHECK:STDOUT:   %.a27: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.217, %Iterate.facet [symbolic]
// CHECK:STDOUT:   %impl.elem2.7c0: %.a27 = impl_witness_access %Iterate.lookup_impl_witness.8b4, element2 [symbolic]
// CHECK:STDOUT:   %impl.elem1.46a: %Destroy.type = impl_witness_access %Iterate.lookup_impl_witness.8b4, element1 [symbolic]
// CHECK:STDOUT:   %as_type.862: type = facet_access_type %impl.elem1.46a [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.e23: <specific function> = specific_impl_function %impl.elem2.7c0, @Iterate.WithSelf.NewCursor(%Iterate.facet) [symbolic]
// CHECK:STDOUT:   %ptr.530: type = ptr_type %as_type.862 [symbolic]
// CHECK:STDOUT:   %.df57: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.10a, %Iterate.facet [symbolic]
// CHECK:STDOUT:   %impl.elem3.199: %.df57 = impl_witness_access %Iterate.lookup_impl_witness.8b4, element3 [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.257: type = class_type @DefaultOptionalStorage, @DefaultOptionalStorage(%facet_value.ccb) [concrete]
// CHECK:STDOUT:   %OptionalStorage.impl_witness.3d3: <witness> = impl_witness imports.%OptionalStorage.impl_witness_table.6db, @T.as_type.as.OptionalStorage.impl(%facet_value.ccb) [concrete]
// CHECK:STDOUT:   %OptionalStorage.facet.642: %OptionalStorage.type = facet_value %ValueType, (%OptionalStorage.impl_witness.3d3) [concrete]
// CHECK:STDOUT:   %Optional.be6: type = class_type @Optional, @Optional(%OptionalStorage.facet.642) [concrete]
// CHECK:STDOUT:   %specific_impl_fn.169: <specific function> = specific_impl_function %impl.elem3.199, @Iterate.WithSelf.Next(%Iterate.facet) [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.type.2bf: type = fn_type @Optional.HasValue, @Optional(%OptionalStorage.facet.642) [concrete]
// CHECK:STDOUT:   %Optional.HasValue.cdf: %Optional.HasValue.type.2bf = struct_value () [concrete]
// CHECK:STDOUT:   %Optional.Get.type.112: type = fn_type @Optional.Get, @Optional(%OptionalStorage.facet.642) [concrete]
// CHECK:STDOUT:   %Optional.Get.6bf: %Optional.Get.type.112 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.facet.306: %Destroy.type = facet_value %ValueType, (%custom_witness.fef) [concrete]
// CHECK:STDOUT:   %MaybeUnformed.669: type = class_type @MaybeUnformed, @MaybeUnformed(%Destroy.facet.306) [concrete]
// CHECK:STDOUT:   %.b36: type = maybe_unformed_type %ValueType [concrete]
// CHECK:STDOUT:   %struct_type.value.has_value.5a1: type = struct_type {.value: %MaybeUnformed.669, .has_value: bool} [concrete]
// CHECK:STDOUT:   %Optional.HasValue.specific_fn: <specific function> = specific_function %Optional.HasValue.cdf, @Optional.HasValue(%OptionalStorage.facet.642) [concrete]
// CHECK:STDOUT:   %Optional.Get.specific_fn: <specific function> = specific_function %Optional.Get.6bf, @Optional.Get(%OptionalStorage.facet.642) [concrete]
// CHECK:STDOUT:   %operator_PlusEqual__carbon_thunk.type: type = fn_type @operator_PlusEqual__carbon_thunk [concrete]
// CHECK:STDOUT:   %operator_PlusEqual__carbon_thunk: %operator_PlusEqual__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.7: type = fn_type @Destroy.Op.loc31_31.6 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.7: %Destroy.Op.type.1d8f74.7 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.f37: <witness> = lookup_impl_witness %impl.elem1.46a, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.e7e: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%impl.elem1.46a) [symbolic]
// CHECK:STDOUT:   %.bc1: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.e7e, %impl.elem1.46a [symbolic]
// CHECK:STDOUT:   %impl.elem0.c7c: %.bc1 = impl_witness_access %Destroy.lookup_impl_witness.f37, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.cd3: <specific function> = specific_impl_function %impl.elem0.c7c, @Destroy.WithSelf.Op(%impl.elem1.46a) [symbolic]
// CHECK:STDOUT:   %MutableRange: type = class_type @MutableRange [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .N = %N
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.bb6: @Optional.%Optional.HasValue.type (%Optional.HasValue.type.e9e) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.HasValue (constants.%Optional.HasValue.6cd)]
// CHECK:STDOUT:   %Core.import_ref.daba: @Optional.%Optional.Get.type (%Optional.Get.type.bb9) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.Get (constants.%Optional.Get.fa1)]
// CHECK:STDOUT:   %Core.import_ref.0c6: type = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%DefaultOptionalStorage (constants.%DefaultOptionalStorage.d50)]
// CHECK:STDOUT:   %Core.import_ref.c86 = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], unloaded
// CHECK:STDOUT:   %Core.import_ref.91a = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], unloaded
// CHECK:STDOUT:   %Core.import_ref.c68: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Has.type (%T.as_type.as.OptionalStorage.impl.Has.type.4bc) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Has (constants.%T.as_type.as.OptionalStorage.impl.Has.dc2)]
// CHECK:STDOUT:   %Core.import_ref.a83: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Get.type (%T.as_type.as.OptionalStorage.impl.Get.type.029) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Get (constants.%T.as_type.as.OptionalStorage.impl.Get.cb5)]
// CHECK:STDOUT:   %Core.import_ref.aaa = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], unloaded
// CHECK:STDOUT:   %OptionalStorage.impl_witness_table.6db = impl_witness_table (%Core.import_ref.0c6, %Core.import_ref.c86, %Core.import_ref.91a, %Core.import_ref.c68, %Core.import_ref.a83, %Core.import_ref.aaa), @T.as_type.as.OptionalStorage.impl [concrete]
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
// CHECK:STDOUT:   %operator_PlusEqual__carbon_thunk.decl: %operator_PlusEqual__carbon_thunk.type = fn_decl @operator_PlusEqual__carbon_thunk [concrete = constants.%operator_PlusEqual__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %MutableRange.decl: type = class_decl @MutableRange [concrete = constants.%MutableRange] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @TestIterate(%R.loc27_17.2: %Iterate_where.type.99d) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Iterate.lookup_impl_witness: <witness> = lookup_impl_witness %R.loc27_17.1, @Iterate [symbolic = %Iterate.lookup_impl_witness (constants.%Iterate.lookup_impl_witness.8b4)]
// CHECK:STDOUT:   %Iterate.facet.loc31_31.5: %Iterate.type = facet_value %R.as_type.loc27_74.1, (%Iterate.lookup_impl_witness) [symbolic = %Iterate.facet.loc31_31.5 (constants.%Iterate.facet)]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.loc31_31.5) [symbolic = %Iterate.WithSelf.NewCursor.type (constants.%Iterate.WithSelf.NewCursor.type.217)]
// CHECK:STDOUT:   %.loc31_31.20: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type, %Iterate.facet.loc31_31.5 [symbolic = %.loc31_31.20 (constants.%.a27)]
// CHECK:STDOUT:   %impl.elem2.loc31_31.2: @TestIterate.%.loc31_31.20 (%.a27) = impl_witness_access %Iterate.lookup_impl_witness, element2 [symbolic = %impl.elem2.loc31_31.2 (constants.%impl.elem2.7c0)]
// CHECK:STDOUT:   %specific_impl_fn.loc31_31.4: <specific function> = specific_impl_function %impl.elem2.loc31_31.2, @Iterate.WithSelf.NewCursor(%Iterate.facet.loc31_31.5) [symbolic = %specific_impl_fn.loc31_31.4 (constants.%specific_impl_fn.e23)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %ptr: type = ptr_type %as_type [symbolic = %ptr (constants.%ptr.530)]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.loc31_31.5) [symbolic = %Iterate.WithSelf.Next.type (constants.%Iterate.WithSelf.Next.type.10a)]
// CHECK:STDOUT:   %.loc31_31.21: type = fn_type_with_self_type %Iterate.WithSelf.Next.type, %Iterate.facet.loc31_31.5 [symbolic = %.loc31_31.21 (constants.%.df57)]
// CHECK:STDOUT:   %impl.elem3.loc31_31.2: @TestIterate.%.loc31_31.21 (%.df57) = impl_witness_access %Iterate.lookup_impl_witness, element3 [symbolic = %impl.elem3.loc31_31.2 (constants.%impl.elem3.199)]
// CHECK:STDOUT:   %specific_impl_fn.loc31_31.5: <specific function> = specific_impl_function %impl.elem3.loc31_31.2, @Iterate.WithSelf.Next(%Iterate.facet.loc31_31.5) [symbolic = %specific_impl_fn.loc31_31.5 (constants.%specific_impl_fn.169)]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%impl.elem1) [symbolic = %Destroy.WithSelf.Op.type (constants.%Destroy.WithSelf.Op.type.e7e)]
// CHECK:STDOUT:   %.loc31_31.22: type = fn_type_with_self_type %Destroy.WithSelf.Op.type, %impl.elem1 [symbolic = %.loc31_31.22 (constants.%.bc1)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %impl.elem1, @Destroy [symbolic = %Destroy.lookup_impl_witness (constants.%Destroy.lookup_impl_witness.f37)]
// CHECK:STDOUT:   %impl.elem0.loc31_31.2: @TestIterate.%.loc31_31.22 (%.bc1) = impl_witness_access %Destroy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc31_31.2 (constants.%impl.elem0.c7c)]
// CHECK:STDOUT:   %specific_impl_fn.loc31_31.6: <specific function> = specific_impl_function %impl.elem0.loc31_31.2, @Destroy.WithSelf.Op(%impl.elem1) [symbolic = %specific_impl_fn.loc31_31.6 (constants.%specific_impl_fn.cd3)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%r.param: @TestIterate.%R.as_type.loc27_74.1 (%R.as_type.a62)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %i.patt: %pattern_type.e41 = value_binding_pattern i [concrete = constants.%i.patt]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %r.ref: @TestIterate.%R.as_type.loc27_74.1 (%R.as_type.a62) = name_ref r, %r
// CHECK:STDOUT:     %impl.elem2.loc31_31.1: @TestIterate.%.loc31_31.20 (%.a27) = impl_witness_access constants.%Iterate.lookup_impl_witness.8b4, element2 [symbolic = %impl.elem2.loc31_31.2 (constants.%impl.elem2.7c0)]
// CHECK:STDOUT:     %bound_method.loc31_31.1: <bound method> = bound_method %r.ref, %impl.elem2.loc31_31.1
// CHECK:STDOUT:     %Iterate.facet.loc31_31.1: %Iterate.type = facet_value constants.%R.as_type.a62, (constants.%Iterate.lookup_impl_witness.8b4) [symbolic = %Iterate.facet.loc31_31.5 (constants.%Iterate.facet)]
// CHECK:STDOUT:     %.loc31_31.1: %Iterate.type = converted constants.%R.as_type.a62, %Iterate.facet.loc31_31.1 [symbolic = %Iterate.facet.loc31_31.5 (constants.%Iterate.facet)]
// CHECK:STDOUT:     %Iterate.facet.loc31_31.2: %Iterate.type = facet_value constants.%R.as_type.a62, (constants.%Iterate.lookup_impl_witness.8b4) [symbolic = %Iterate.facet.loc31_31.5 (constants.%Iterate.facet)]
// CHECK:STDOUT:     %.loc31_31.2: %Iterate.type = converted constants.%R.as_type.a62, %Iterate.facet.loc31_31.2 [symbolic = %Iterate.facet.loc31_31.5 (constants.%Iterate.facet)]
// CHECK:STDOUT:     %specific_impl_fn.loc31_31.1: <specific function> = specific_impl_function %impl.elem2.loc31_31.1, @Iterate.WithSelf.NewCursor(constants.%Iterate.facet) [symbolic = %specific_impl_fn.loc31_31.4 (constants.%specific_impl_fn.e23)]
// CHECK:STDOUT:     %bound_method.loc31_31.2: <bound method> = bound_method %r.ref, %specific_impl_fn.loc31_31.1
// CHECK:STDOUT:     %var: ref @TestIterate.%as_type (%as_type.862) = var invalid
// CHECK:STDOUT:     %Iterate.WithSelf.NewCursor.call: init @TestIterate.%as_type (%as_type.862) to %var = call %bound_method.loc31_31.2(%r.ref)
// CHECK:STDOUT:     assign %var, %Iterate.WithSelf.NewCursor.call
// CHECK:STDOUT:     br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.next:
// CHECK:STDOUT:     %addr.loc31: @TestIterate.%ptr (%ptr.530) = addr_of %var
// CHECK:STDOUT:     %impl.elem3.loc31_31.1: @TestIterate.%.loc31_31.21 (%.df57) = impl_witness_access constants.%Iterate.lookup_impl_witness.8b4, element3 [symbolic = %impl.elem3.loc31_31.2 (constants.%impl.elem3.199)]
// CHECK:STDOUT:     %bound_method.loc31_31.3: <bound method> = bound_method %r.ref, %impl.elem3.loc31_31.1
// CHECK:STDOUT:     %Iterate.facet.loc31_31.3: %Iterate.type = facet_value constants.%R.as_type.a62, (constants.%Iterate.lookup_impl_witness.8b4) [symbolic = %Iterate.facet.loc31_31.5 (constants.%Iterate.facet)]
// CHECK:STDOUT:     %.loc31_31.3: %Iterate.type = converted constants.%R.as_type.a62, %Iterate.facet.loc31_31.3 [symbolic = %Iterate.facet.loc31_31.5 (constants.%Iterate.facet)]
// CHECK:STDOUT:     %Iterate.facet.loc31_31.4: %Iterate.type = facet_value constants.%R.as_type.a62, (constants.%Iterate.lookup_impl_witness.8b4) [symbolic = %Iterate.facet.loc31_31.5 (constants.%Iterate.facet)]
// CHECK:STDOUT:     %.loc31_31.4: %Iterate.type = converted constants.%R.as_type.a62, %Iterate.facet.loc31_31.4 [symbolic = %Iterate.facet.loc31_31.5 (constants.%Iterate.facet)]
// CHECK:STDOUT:     %.loc31_31.5: %Destroy.type = converted constants.%as_type.862, constants.%impl.elem1.46a [symbolic = %impl.elem1 (constants.%impl.elem1.46a)]
// CHECK:STDOUT:     %.loc31_31.6: %Destroy.type = converted constants.%as_type.862, constants.%impl.elem1.46a [symbolic = %impl.elem1 (constants.%impl.elem1.46a)]
// CHECK:STDOUT:     %specific_impl_fn.loc31_31.2: <specific function> = specific_impl_function %impl.elem3.loc31_31.1, @Iterate.WithSelf.Next(constants.%Iterate.facet) [symbolic = %specific_impl_fn.loc31_31.5 (constants.%specific_impl_fn.169)]
// CHECK:STDOUT:     %bound_method.loc31_31.4: <bound method> = bound_method %r.ref, %specific_impl_fn.loc31_31.2
// CHECK:STDOUT:     %.loc31_31.7: ref %Optional.be6 = temporary_storage
// CHECK:STDOUT:     %Iterate.WithSelf.Next.call: init %Optional.be6 to %.loc31_31.7 = call %bound_method.loc31_31.4(%r.ref, %addr.loc31)
// CHECK:STDOUT:     %.loc31_31.8: ref %Optional.be6 = temporary %.loc31_31.7, %Iterate.WithSelf.Next.call
// CHECK:STDOUT:     %.loc31_31.9: %Optional.HasValue.type.2bf = specific_constant imports.%Core.import_ref.bb6, @Optional(constants.%OptionalStorage.facet.642) [concrete = constants.%Optional.HasValue.cdf]
// CHECK:STDOUT:     %HasValue.ref: %Optional.HasValue.type.2bf = name_ref HasValue, %.loc31_31.9 [concrete = constants.%Optional.HasValue.cdf]
// CHECK:STDOUT:     %Optional.HasValue.bound: <bound method> = bound_method %.loc31_31.8, %HasValue.ref
// CHECK:STDOUT:     %Optional.HasValue.specific_fn: <specific function> = specific_function %HasValue.ref, @Optional.HasValue(constants.%OptionalStorage.facet.642) [concrete = constants.%Optional.HasValue.specific_fn]
// CHECK:STDOUT:     %bound_method.loc31_31.5: <bound method> = bound_method %.loc31_31.8, %Optional.HasValue.specific_fn
// CHECK:STDOUT:     %.loc31_31.10: %Optional.be6 = acquire_value %.loc31_31.8
// CHECK:STDOUT:     %Optional.HasValue.call: init bool = call %bound_method.loc31_31.5(%.loc31_31.10)
// CHECK:STDOUT:     %.loc31_31.11: bool = value_of_initializer %Optional.HasValue.call
// CHECK:STDOUT:     %.loc31_31.12: bool = converted %Optional.HasValue.call, %.loc31_31.11
// CHECK:STDOUT:     if %.loc31_31.12 br !for.body else br !for.done
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.body:
// CHECK:STDOUT:     %.loc31_31.13: %Optional.Get.type.112 = specific_constant imports.%Core.import_ref.daba, @Optional(constants.%OptionalStorage.facet.642) [concrete = constants.%Optional.Get.6bf]
// CHECK:STDOUT:     %Get.ref: %Optional.Get.type.112 = name_ref Get, %.loc31_31.13 [concrete = constants.%Optional.Get.6bf]
// CHECK:STDOUT:     %Optional.Get.bound: <bound method> = bound_method %.loc31_31.8, %Get.ref
// CHECK:STDOUT:     %Optional.Get.specific_fn: <specific function> = specific_function %Get.ref, @Optional.Get(constants.%OptionalStorage.facet.642) [concrete = constants.%Optional.Get.specific_fn]
// CHECK:STDOUT:     %bound_method.loc31_31.6: <bound method> = bound_method %.loc31_31.8, %Optional.Get.specific_fn
// CHECK:STDOUT:     %.loc31_31.14: ref %ValueType = temporary_storage
// CHECK:STDOUT:     %.loc31_31.15: %Optional.be6 = acquire_value %.loc31_31.8
// CHECK:STDOUT:     %Optional.Get.call: init %ValueType to %.loc31_31.14 = call %bound_method.loc31_31.6(%.loc31_31.15)
// CHECK:STDOUT:     %.loc31_31.16: ref %ValueType = temporary %.loc31_31.14, %Optional.Get.call
// CHECK:STDOUT:     %.loc31_31.17: %ValueType = acquire_value %.loc31_31.16
// CHECK:STDOUT:     %.loc31_16: type = splice_block %ValueType.ref.loc31 [concrete = constants.%ValueType] {
// CHECK:STDOUT:       %Cpp.ref.loc31: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:       %N.ref.loc31: <namespace> = name_ref N, imports.%N [concrete = imports.%N]
// CHECK:STDOUT:       %ValueType.ref.loc31: type = name_ref ValueType, imports.%ValueType.decl [concrete = constants.%ValueType]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %i: %ValueType = wrapper_binding i, %.loc31_31.17
// CHECK:STDOUT:     %sum.ref: ref %ValueType = name_ref sum, %sum
// CHECK:STDOUT:     %i.ref: %ValueType = name_ref i, %i
// CHECK:STDOUT:     %.loc32_12: ref %ValueType = value_as_ref %i.ref
// CHECK:STDOUT:     %addr.loc32: %ptr.9d3 = addr_of %.loc32_12
// CHECK:STDOUT:     %.loc32_9.1: %ptr.32e = as_compatible %addr.loc32
// CHECK:STDOUT:     %.loc32_9.2: %ptr.32e = converted %addr.loc32, %.loc32_9.1
// CHECK:STDOUT:     %operator_PlusEqual__carbon_thunk.call: ref %ValueType = call imports.%operator_PlusEqual__carbon_thunk.decl(%sum.ref, %.loc32_9.2)
// CHECK:STDOUT:     br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.done:
// CHECK:STDOUT:     %ValueType.Op.bound.loc31: <bound method> = bound_method %.loc31_31.16, constants.%ValueType.Op.4c94b9.1
// CHECK:STDOUT:     %Op.ref.loc31: %ValueType.cpp_destructor.type = name_ref Op, imports.%ValueType.cpp_destructor.decl [concrete = constants.%ValueType.cpp_destructor]
// CHECK:STDOUT:     %ValueType.cpp_destructor.bound.loc31: <bound method> = bound_method %.loc31_31.16, %Op.ref.loc31
// CHECK:STDOUT:     %ValueType.cpp_destructor.call.loc31: init %empty_tuple.type = call %ValueType.cpp_destructor.bound.loc31(%.loc31_31.16)
// CHECK:STDOUT:     %Destroy.Op.bound: <bound method> = bound_method %.loc31_31.8, constants.%Destroy.Op.1a2547.7
// CHECK:STDOUT:     %Destroy.Op.call: init %empty_tuple.type = call %Destroy.Op.bound(%.loc31_31.8)
// CHECK:STDOUT:     %impl.elem0.loc31_31.1: @TestIterate.%.loc31_31.22 (%.bc1) = impl_witness_access constants.%Destroy.lookup_impl_witness.f37, element0 [symbolic = %impl.elem0.loc31_31.2 (constants.%impl.elem0.c7c)]
// CHECK:STDOUT:     %bound_method.loc31_31.7: <bound method> = bound_method %var, %impl.elem0.loc31_31.1
// CHECK:STDOUT:     %.loc31_31.18: %Destroy.type = converted constants.%as_type.862, constants.%impl.elem1.46a [symbolic = %impl.elem1 (constants.%impl.elem1.46a)]
// CHECK:STDOUT:     %.loc31_31.19: %Destroy.type = converted constants.%as_type.862, constants.%impl.elem1.46a [symbolic = %impl.elem1 (constants.%impl.elem1.46a)]
// CHECK:STDOUT:     %specific_impl_fn.loc31_31.3: <specific function> = specific_impl_function %impl.elem0.loc31_31.1, @Destroy.WithSelf.Op(constants.%impl.elem1.46a) [symbolic = %specific_impl_fn.loc31_31.6 (constants.%specific_impl_fn.cd3)]
// CHECK:STDOUT:     %bound_method.loc31_31.8: <bound method> = bound_method %var, %specific_impl_fn.loc31_31.3
// CHECK:STDOUT:     %Destroy.WithSelf.Op.call: init %empty_tuple.type = call %bound_method.loc31_31.8(%var)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc31_31.1(%self.param: ref %.b36) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc31_31.2(%self.param: ref %MaybeUnformed.669) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc31_31.3(%self.param: ref bool) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc31_31.4(%self.param: ref %struct_type.value.has_value.5a1) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc31_31.5(%self.param: ref %DefaultOptionalStorage.257) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc31_31.6(%self.param: ref %Optional.be6) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestRangeFor(%m.param: ref %MutableRange) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %i.patt: %pattern_type.e41 = value_binding_pattern i [concrete = constants.%i.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %m.ref: ref %MutableRange = name_ref m, %m
// CHECK:STDOUT:   %var: ref <error> = var invalid [concrete = <error>]
// CHECK:STDOUT:   assign <error>, <error>
// CHECK:STDOUT:   br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT: !for.next:
// CHECK:STDOUT:   %addr.loc95: <error> = addr_of <error> [concrete = <error>]
// CHECK:STDOUT:   %HasValue.ref: <error> = name_ref HasValue, <error> [concrete = <error>]
// CHECK:STDOUT:   if <error> br !for.body else br !for.done
// CHECK:STDOUT:
// CHECK:STDOUT: !for.body:
// CHECK:STDOUT:   %Get.ref: <error> = name_ref Get, <error> [concrete = <error>]
// CHECK:STDOUT:   %.loc95: type = splice_block %ValueType.ref.loc95 [concrete = constants.%ValueType] {
// CHECK:STDOUT:     %Cpp.ref.loc95: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %N.ref.loc95: <namespace> = name_ref N, imports.%N [concrete = imports.%N]
// CHECK:STDOUT:     %ValueType.ref.loc95: type = name_ref ValueType, imports.%ValueType.decl [concrete = constants.%ValueType]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %i: %ValueType = wrapper_binding i, <error> [concrete = <error>]
// CHECK:STDOUT:   %sum.ref: ref %ValueType = name_ref sum, %sum
// CHECK:STDOUT:   %i.ref: %ValueType = name_ref i, %i [concrete = <error>]
// CHECK:STDOUT:   br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT: !for.done:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestIterate(constants.%R.623) {
// CHECK:STDOUT:   %R.patt.loc27_17.2 => constants.%R.patt.324
// CHECK:STDOUT:   %R.loc27_17.1 => constants.%R.623
// CHECK:STDOUT:   %R.as_type.loc27_74.1 => constants.%R.as_type.a62
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.725
// CHECK:STDOUT:   %r.param_patt.loc27_72.2 => constants.%r.param_patt
// CHECK:STDOUT:   %r.patt.loc27_72.2 => constants.%r.patt
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
// CHECK:STDOUT:   %T.220: %OptionalStorage.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Optional.Get.type.bb9: type = fn_type @Optional.Get, @Optional(%T.220) [symbolic]
// CHECK:STDOUT:   %Optional.Get.fa1: %Optional.Get.type.bb9 = struct_value () [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.type.e9e: type = fn_type @Optional.HasValue, @Optional(%T.220) [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.6cd: %Optional.HasValue.type.e9e = struct_value () [symbolic]
// CHECK:STDOUT:   %T.283: %facet_type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.d50: type = class_type @DefaultOptionalStorage, @DefaultOptionalStorage(%T.283) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Get.type.029: type = fn_type @T.as_type.as.OptionalStorage.impl.Get, @T.as_type.as.OptionalStorage.impl(%T.283) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Get.cb5: %T.as_type.as.OptionalStorage.impl.Get.type.029 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Has.type.4bc: type = fn_type @T.as_type.as.OptionalStorage.impl.Has, @T.as_type.as.OptionalStorage.impl(%T.283) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Has.dc2: %T.as_type.as.OptionalStorage.impl.Has.type.4bc = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Some.type.800: type = fn_type @T.as_type.as.OptionalStorage.impl.Some, @T.as_type.as.OptionalStorage.impl(%T.283) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Some.e83: %T.as_type.as.OptionalStorage.impl.Some.type.800 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.None.type.300: type = fn_type @T.as_type.as.OptionalStorage.impl.None, @T.as_type.as.OptionalStorage.impl(%T.283) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.None.f54: %T.as_type.as.OptionalStorage.impl.None.type.300 = struct_value () [symbolic]
// CHECK:STDOUT:   %.Self.35d: %Iterate.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %Iterate.lookup_impl_witness.b37: <witness> = lookup_impl_witness %.Self.35d, @Iterate [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.903: %facet_type = impl_witness_access %Iterate.lookup_impl_witness.b37, element0 [symbolic_self]
// CHECK:STDOUT:   %ValueType: type = class_type @ValueType [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %pattern_type.e41: type = pattern_type %ValueType [concrete]
// CHECK:STDOUT:   %ValueType.cpp_destructor.type: type = fn_type @ValueType.cpp_destructor [concrete]
// CHECK:STDOUT:   %ValueType.cpp_destructor: %ValueType.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT:   %ValueType.Op.type.79266b.1: type = fn_type @ValueType.Op.1 [concrete]
// CHECK:STDOUT:   %ValueType.Op.4c94b9.1: %ValueType.Op.type.79266b.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.fef: <witness> = custom_witness (%ValueType.Op.4c94b9.1), @Destroy [concrete]
// CHECK:STDOUT:   %const.773: type = const_type %ValueType [concrete]
// CHECK:STDOUT:   %ptr.32e: type = ptr_type %const.773 [concrete]
// CHECK:STDOUT:   %ptr.9d3: type = ptr_type %ValueType [concrete]
// CHECK:STDOUT:   %ValueType.Op.type.79266b.2: type = fn_type @ValueType.Op.2 [concrete]
// CHECK:STDOUT:   %ValueType.Op.4c94b9.2: %ValueType.Op.type.79266b.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.c19: <witness> = custom_witness (%ValueType.Op.4c94b9.2), @Copy [concrete]
// CHECK:STDOUT:   %facet_value.ccb: %facet_type = facet_value %ValueType, (%custom_witness.fef, %custom_witness.c19) [concrete]
// CHECK:STDOUT:   %Iterate_where.type.99d: type = facet_type <@Iterate where %impl.elem0.903 = %facet_value.ccb> [concrete]
// CHECK:STDOUT:   %pattern_type.3f6: type = pattern_type %Iterate_where.type.99d [concrete]
// CHECK:STDOUT:   %R.patt.324: %pattern_type.3f6 = symbolic_binding_pattern R, 0 [symbolic]
// CHECK:STDOUT:   %R.623: %Iterate_where.type.99d = symbolic_binding R, 0 [symbolic]
// CHECK:STDOUT:   %R.as_type.a62: type = facet_access_type %R.623 [symbolic]
// CHECK:STDOUT:   %pattern_type.725: type = pattern_type %R.as_type.a62 [symbolic]
// CHECK:STDOUT:   %r.param_patt.162: %pattern_type.725 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %r.patt.15b: %pattern_type.725 = at_binding_pattern r, %r.param_patt.162 [symbolic]
// CHECK:STDOUT:   %i.patt: %pattern_type.e41 = value_binding_pattern i [concrete]
// CHECK:STDOUT:   %CppRangeForIterate.type: type = facet_type <@CppRangeForIterate> [concrete]
// CHECK:STDOUT:   %.Self.6bd: %CppRangeForIterate.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %CppRangeForIterate.lookup_impl_witness.143: <witness> = lookup_impl_witness %.Self.6bd, @CppRangeForIterate [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.109: type = impl_witness_access %CppRangeForIterate.lookup_impl_witness.143, element0 [symbolic_self]
// CHECK:STDOUT:   %CppUnsafeDeref.lookup_impl_witness.712: <witness> = lookup_impl_witness %impl.elem0.109, @CppUnsafeDeref [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.56a: type = impl_witness_access %CppUnsafeDeref.lookup_impl_witness.712, element0 [symbolic_self]
// CHECK:STDOUT:   %impl.elem1.222: type = impl_witness_access %CppRangeForIterate.lookup_impl_witness.143, element1 [symbolic_self]
// CHECK:STDOUT:   %CppRangeForIterate_where.type: type = facet_type <@CppRangeForIterate where %impl.elem0.109 impls @Destroy and %impl.elem0.109 impls @Copy and %impl.elem0.109 impls @CppUnsafeDeref and %impl.elem0.109 impls @EqWith, @EqWith(%impl.elem1.222) and %impl.elem0.109 impls @Inc and %impl.elem0.56a impls @Destroy and %impl.elem0.56a impls @Copy and %impl.elem1.222 impls @Destroy and %impl.elem1.222 impls @Copy> [concrete]
// CHECK:STDOUT:   %R.39b: %CppRangeForIterate_where.type = symbolic_binding R, 0 [symbolic]
// CHECK:STDOUT:   %CppRangeForIterate.lookup_impl_witness.7e2712.2: <witness> = lookup_impl_witness %R.39b, @CppRangeForIterate [symbolic]
// CHECK:STDOUT:   %impl.elem1.7a0bff.2: type = impl_witness_access %CppRangeForIterate.lookup_impl_witness.7e2712.2, element1 [symbolic]
// CHECK:STDOUT:   %impl.elem0.73eb7f.2: type = impl_witness_access %CppRangeForIterate.lookup_impl_witness.7e2712.2, element0 [symbolic]
// CHECK:STDOUT:   %tuple.type.3cf: type = tuple_type (%impl.elem0.73eb7f.2, %impl.elem1.7a0bff.2) [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.69f: <witness> = lookup_impl_witness %tuple.type.3cf, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.593: %Destroy.type = facet_value %tuple.type.3cf, (%Destroy.lookup_impl_witness.69f) [symbolic]
// CHECK:STDOUT:   %CppUnsafeDeref.lookup_impl_witness.aa7: <witness> = lookup_impl_witness %impl.elem0.73eb7f.2, @CppUnsafeDeref [symbolic]
// CHECK:STDOUT:   %impl.elem0.af7: type = impl_witness_access %CppUnsafeDeref.lookup_impl_witness.aa7, element0 [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.ec3: <witness> = lookup_impl_witness %impl.elem0.af7, @Destroy [symbolic]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.a7a: <witness> = lookup_impl_witness %impl.elem0.af7, @Copy [symbolic]
// CHECK:STDOUT:   %facet_value.194: %facet_type = facet_value %impl.elem0.af7, (%Destroy.lookup_impl_witness.ec3, %Copy.lookup_impl_witness.a7a) [symbolic]
// CHECK:STDOUT:   %R.as_type.as.Iterate.impl.Next.type.1f7: type = fn_type @R.as_type.as.Iterate.impl.Next, @R.as_type.as.Iterate.impl(%R.39b) [symbolic]
// CHECK:STDOUT:   %R.as_type.as.Iterate.impl.Next.0d6: %R.as_type.as.Iterate.impl.Next.type.1f7 = struct_value () [symbolic]
// CHECK:STDOUT:   %R.as_type.as.Iterate.impl.NewCursor.type.36f: type = fn_type @R.as_type.as.Iterate.impl.NewCursor, @R.as_type.as.Iterate.impl(%R.39b) [symbolic]
// CHECK:STDOUT:   %R.as_type.as.Iterate.impl.NewCursor.29b: %R.as_type.as.Iterate.impl.NewCursor.type.36f = struct_value () [symbolic]
// CHECK:STDOUT:   %Iterate.lookup_impl_witness.8b4: <witness> = lookup_impl_witness %R.623, @Iterate [symbolic]
// CHECK:STDOUT:   %Iterate.facet.b19: %Iterate.type = facet_value %R.as_type.a62, (%Iterate.lookup_impl_witness.8b4) [symbolic]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.217: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.b19) [symbolic]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.10a: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.b19) [symbolic]
// CHECK:STDOUT:   %.a27: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.217, %Iterate.facet.b19 [symbolic]
// CHECK:STDOUT:   %impl.elem2.7c0: %.a27 = impl_witness_access %Iterate.lookup_impl_witness.8b4, element2 [symbolic]
// CHECK:STDOUT:   %impl.elem1.46a: %Destroy.type = impl_witness_access %Iterate.lookup_impl_witness.8b4, element1 [symbolic]
// CHECK:STDOUT:   %as_type.862: type = facet_access_type %impl.elem1.46a [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.e23: <specific function> = specific_impl_function %impl.elem2.7c0, @Iterate.WithSelf.NewCursor(%Iterate.facet.b19) [symbolic]
// CHECK:STDOUT:   %ptr.530: type = ptr_type %as_type.862 [symbolic]
// CHECK:STDOUT:   %.df57: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.10a, %Iterate.facet.b19 [symbolic]
// CHECK:STDOUT:   %impl.elem3.199: %.df57 = impl_witness_access %Iterate.lookup_impl_witness.8b4, element3 [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.257: type = class_type @DefaultOptionalStorage, @DefaultOptionalStorage(%facet_value.ccb) [concrete]
// CHECK:STDOUT:   %OptionalStorage.impl_witness.be5: <witness> = impl_witness imports.%OptionalStorage.impl_witness_table.655, @T.as_type.as.OptionalStorage.impl(%facet_value.ccb) [concrete]
// CHECK:STDOUT:   %OptionalStorage.facet.7ce: %OptionalStorage.type = facet_value %ValueType, (%OptionalStorage.impl_witness.be5) [concrete]
// CHECK:STDOUT:   %Optional.552: type = class_type @Optional, @Optional(%OptionalStorage.facet.7ce) [concrete]
// CHECK:STDOUT:   %specific_impl_fn.169: <specific function> = specific_impl_function %impl.elem3.199, @Iterate.WithSelf.Next(%Iterate.facet.b19) [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.type.109: type = fn_type @Optional.HasValue, @Optional(%OptionalStorage.facet.7ce) [concrete]
// CHECK:STDOUT:   %Optional.HasValue.605: %Optional.HasValue.type.109 = struct_value () [concrete]
// CHECK:STDOUT:   %Optional.Get.type.135: type = fn_type @Optional.Get, @Optional(%OptionalStorage.facet.7ce) [concrete]
// CHECK:STDOUT:   %Optional.Get.5aa: %Optional.Get.type.135 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.facet.306: %Destroy.type = facet_value %ValueType, (%custom_witness.fef) [concrete]
// CHECK:STDOUT:   %MaybeUnformed.669: type = class_type @MaybeUnformed, @MaybeUnformed(%Destroy.facet.306) [concrete]
// CHECK:STDOUT:   %.b36: type = maybe_unformed_type %ValueType [concrete]
// CHECK:STDOUT:   %struct_type.value.has_value.5a1: type = struct_type {.value: %MaybeUnformed.669, .has_value: bool} [concrete]
// CHECK:STDOUT:   %Optional.HasValue.specific_fn: <specific function> = specific_function %Optional.HasValue.605, @Optional.HasValue(%OptionalStorage.facet.7ce) [concrete]
// CHECK:STDOUT:   %Optional.Get.specific_fn: <specific function> = specific_function %Optional.Get.5aa, @Optional.Get(%OptionalStorage.facet.7ce) [concrete]
// CHECK:STDOUT:   %operator_PlusEqual__carbon_thunk.type: type = fn_type @operator_PlusEqual__carbon_thunk [concrete]
// CHECK:STDOUT:   %operator_PlusEqual__carbon_thunk: %operator_PlusEqual__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.7: type = fn_type @Destroy.Op.loc38_31.6 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.7: %Destroy.Op.type.1d8f74.7 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.f37: <witness> = lookup_impl_witness %impl.elem1.46a, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.e7e: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%impl.elem1.46a) [symbolic]
// CHECK:STDOUT:   %.bc1: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.e7e, %impl.elem1.46a [symbolic]
// CHECK:STDOUT:   %impl.elem0.c7c: %.bc1 = impl_witness_access %Destroy.lookup_impl_witness.f37, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.cd3: <specific function> = specific_impl_function %impl.elem0.c7c, @Destroy.WithSelf.Op(%impl.elem1.46a) [symbolic]
// CHECK:STDOUT:   %MutableAndConstRange: type = class_type @MutableAndConstRange [concrete]
// CHECK:STDOUT:   %pattern_type.394: type = pattern_type %MutableAndConstRange [concrete]
// CHECK:STDOUT:   %Iterator: type = class_type @Iterator.1 [concrete]
// CHECK:STDOUT:   %Begin.type: type = fn_type @Begin [concrete]
// CHECK:STDOUT:   %Begin: %Begin.type = struct_value () [concrete]
// CHECK:STDOUT:   %End.type: type = fn_type @End [concrete]
// CHECK:STDOUT:   %End: %End.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.a40: <witness> = custom_witness (%Iterator, %Iterator, %Begin, %End), @CppRangeForIterate [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.fa7609.1: type = fn_type @Iterator.Op.1 [concrete]
// CHECK:STDOUT:   %Iterator.Op.549055.1: %Iterator.Op.type.fa7609.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.dd5: <witness> = custom_witness (%ValueType, %Iterator.Op.549055.1), @CppUnsafeDeref [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.fa7609.2: type = fn_type @Iterator.Op.2 [concrete]
// CHECK:STDOUT:   %Iterator.Op.549055.2: %Iterator.Op.type.fa7609.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.04a: <witness> = custom_witness (%Iterator.Op.549055.2), @Destroy [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.fa7609.3: type = fn_type @Iterator.Op.3 [concrete]
// CHECK:STDOUT:   %Iterator.Op.549055.3: %Iterator.Op.type.fa7609.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.cfc: <witness> = custom_witness (%Iterator.Op.549055.3), @Copy [concrete]
// CHECK:STDOUT:   %Equal.type: type = fn_type @Equal [concrete]
// CHECK:STDOUT:   %Equal: %Equal.type = struct_value () [concrete]
// CHECK:STDOUT:   %NotEqual.type: type = fn_type @NotEqual [concrete]
// CHECK:STDOUT:   %NotEqual: %NotEqual.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.c66: <witness> = custom_witness (%Equal, %NotEqual), @EqWith, @EqWith(%Iterator) [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.fa7609.4: type = fn_type @Iterator.Op.4 [concrete]
// CHECK:STDOUT:   %Iterator.Op.549055.4: %Iterator.Op.type.fa7609.4 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.01a: <witness> = custom_witness (%Iterator.Op.549055.4), @Inc [concrete]
// CHECK:STDOUT:   %facet_value.88d: %CppRangeForIterate_where.type = facet_value %MutableAndConstRange, (%custom_witness.fef, %custom_witness.c19, %custom_witness.a40, %custom_witness.04a, %custom_witness.cfc, %custom_witness.dd5, %custom_witness.c66, %custom_witness.01a) [concrete]
// CHECK:STDOUT:   %tuple.type.56e: type = tuple_type (%Iterator, %Iterator) [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.8: type = fn_type @Destroy.Op.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.8: %Destroy.Op.type.1d8f74.8 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.df9cc1.8: <witness> = custom_witness (%Destroy.Op.1a2547.8), @Destroy [concrete]
// CHECK:STDOUT:   %Destroy.facet.7aa: %Destroy.type = facet_value %tuple.type.56e, (%custom_witness.df9cc1.8) [concrete]
// CHECK:STDOUT:   %Iterate.impl_witness.a6e: <witness> = impl_witness imports.%Iterate.impl_witness_table.344, @R.as_type.as.Iterate.impl(%facet_value.88d) [concrete]
// CHECK:STDOUT:   %R.as_type.as.Iterate.impl.NewCursor.type.11b: type = fn_type @R.as_type.as.Iterate.impl.NewCursor, @R.as_type.as.Iterate.impl(%facet_value.88d) [concrete]
// CHECK:STDOUT:   %R.as_type.as.Iterate.impl.NewCursor.96c: %R.as_type.as.Iterate.impl.NewCursor.type.11b = struct_value () [concrete]
// CHECK:STDOUT:   %R.as_type.as.Iterate.impl.Next.type.b29: type = fn_type @R.as_type.as.Iterate.impl.Next, @R.as_type.as.Iterate.impl(%facet_value.88d) [concrete]
// CHECK:STDOUT:   %R.as_type.as.Iterate.impl.Next.683: %R.as_type.as.Iterate.impl.Next.type.b29 = struct_value () [concrete]
// CHECK:STDOUT:   %Iterate.facet.5cc: %Iterate.type = facet_value %MutableAndConstRange, (%Iterate.impl_witness.a6e) [concrete]
// CHECK:STDOUT:   %facet_value.0ec: %Iterate_where.type.99d = facet_value %MutableAndConstRange, (%Iterate.impl_witness.a6e) [concrete]
// CHECK:STDOUT:   %r.param_patt.8bb: %pattern_type.394 = value_param_pattern [concrete]
// CHECK:STDOUT:   %r.patt.ac4: %pattern_type.394 = at_binding_pattern r, %r.param_patt.8bb [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.875: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.5cc) [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.b41: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.5cc) [concrete]
// CHECK:STDOUT:   %.7e3: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.875, %Iterate.facet.5cc [concrete]
// CHECK:STDOUT:   %R.as_type.as.Iterate.impl.NewCursor.specific_fn: <specific function> = specific_function %R.as_type.as.Iterate.impl.NewCursor.96c, @R.as_type.as.Iterate.impl.NewCursor(%facet_value.88d) [concrete]
// CHECK:STDOUT:   %ptr.484: type = ptr_type %tuple.type.56e [concrete]
// CHECK:STDOUT:   %.fef: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.b41, %Iterate.facet.5cc [concrete]
// CHECK:STDOUT:   %R.as_type.as.Iterate.impl.Next.specific_fn: <specific function> = specific_function %R.as_type.as.Iterate.impl.Next.683, @R.as_type.as.Iterate.impl.Next(%facet_value.88d) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.b7b: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.7aa) [concrete]
// CHECK:STDOUT:   %.1ad: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.b7b, %Destroy.facet.7aa [concrete]
// CHECK:STDOUT:   %complete_type.1b5: <witness> = complete_type_witness %tuple.type.56e [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .N = %N
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.bb6: @Optional.%Optional.HasValue.type (%Optional.HasValue.type.e9e) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.HasValue (constants.%Optional.HasValue.6cd)]
// CHECK:STDOUT:   %Core.import_ref.daba: @Optional.%Optional.Get.type (%Optional.Get.type.bb9) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.Get (constants.%Optional.Get.fa1)]
// CHECK:STDOUT:   %Core.import_ref.0c6: type = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%DefaultOptionalStorage (constants.%DefaultOptionalStorage.d50)]
// CHECK:STDOUT:   %Core.import_ref.f4f: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.None.type (%T.as_type.as.OptionalStorage.impl.None.type.300) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.None (constants.%T.as_type.as.OptionalStorage.impl.None.f54)]
// CHECK:STDOUT:   %Core.import_ref.18f: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Some.type (%T.as_type.as.OptionalStorage.impl.Some.type.800) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Some (constants.%T.as_type.as.OptionalStorage.impl.Some.e83)]
// CHECK:STDOUT:   %Core.import_ref.c68: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Has.type (%T.as_type.as.OptionalStorage.impl.Has.type.4bc) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Has (constants.%T.as_type.as.OptionalStorage.impl.Has.dc2)]
// CHECK:STDOUT:   %Core.import_ref.a83: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Get.type (%T.as_type.as.OptionalStorage.impl.Get.type.029) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Get (constants.%T.as_type.as.OptionalStorage.impl.Get.cb5)]
// CHECK:STDOUT:   %Core.import_ref.aaa = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], unloaded
// CHECK:STDOUT:   %OptionalStorage.impl_witness_table.655 = impl_witness_table (%Core.import_ref.0c6, %Core.import_ref.f4f, %Core.import_ref.18f, %Core.import_ref.c68, %Core.import_ref.a83, %Core.import_ref.aaa), @T.as_type.as.OptionalStorage.impl [concrete]
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
// CHECK:STDOUT:   %Core.import_ref.301: %facet_type = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @R.as_type.as.Iterate.impl.%facet_value (constants.%facet_value.194)]
// CHECK:STDOUT:   %Core.import_ref.89f: %Destroy.type = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @R.as_type.as.Iterate.impl.%Destroy.facet (constants.%Destroy.facet.593)]
// CHECK:STDOUT:   %Core.import_ref.3e6: @R.as_type.as.Iterate.impl.%R.as_type.as.Iterate.impl.NewCursor.type (%R.as_type.as.Iterate.impl.NewCursor.type.36f) = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @R.as_type.as.Iterate.impl.%R.as_type.as.Iterate.impl.NewCursor (constants.%R.as_type.as.Iterate.impl.NewCursor.29b)]
// CHECK:STDOUT:   %Core.import_ref.85d: @R.as_type.as.Iterate.impl.%R.as_type.as.Iterate.impl.Next.type (%R.as_type.as.Iterate.impl.Next.type.1f7) = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @R.as_type.as.Iterate.impl.%R.as_type.as.Iterate.impl.Next (constants.%R.as_type.as.Iterate.impl.Next.0d6)]
// CHECK:STDOUT:   %Iterate.impl_witness_table.344 = impl_witness_table (%Core.import_ref.301, %Core.import_ref.89f, %Core.import_ref.3e6, %Core.import_ref.85d), @R.as_type.as.Iterate.impl [concrete]
// CHECK:STDOUT:   %operator_PlusEqual__carbon_thunk.decl: %operator_PlusEqual__carbon_thunk.type = fn_decl @operator_PlusEqual__carbon_thunk [concrete = constants.%operator_PlusEqual__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %MutableAndConstRange.decl: type = class_decl @MutableAndConstRange [concrete = constants.%MutableAndConstRange] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @TestIterate(%R.loc34_17.2: %Iterate_where.type.99d) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Iterate.lookup_impl_witness: <witness> = lookup_impl_witness %R.loc34_17.1, @Iterate [symbolic = %Iterate.lookup_impl_witness (constants.%Iterate.lookup_impl_witness.8b4)]
// CHECK:STDOUT:   %Iterate.facet.loc38_31.5: %Iterate.type = facet_value %R.as_type.loc34_74.1, (%Iterate.lookup_impl_witness) [symbolic = %Iterate.facet.loc38_31.5 (constants.%Iterate.facet.b19)]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.loc38_31.5) [symbolic = %Iterate.WithSelf.NewCursor.type (constants.%Iterate.WithSelf.NewCursor.type.217)]
// CHECK:STDOUT:   %.loc38_31.20: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type, %Iterate.facet.loc38_31.5 [symbolic = %.loc38_31.20 (constants.%.a27)]
// CHECK:STDOUT:   %impl.elem2.loc38_31.2: @TestIterate.%.loc38_31.20 (%.a27) = impl_witness_access %Iterate.lookup_impl_witness, element2 [symbolic = %impl.elem2.loc38_31.2 (constants.%impl.elem2.7c0)]
// CHECK:STDOUT:   %specific_impl_fn.loc38_31.4: <specific function> = specific_impl_function %impl.elem2.loc38_31.2, @Iterate.WithSelf.NewCursor(%Iterate.facet.loc38_31.5) [symbolic = %specific_impl_fn.loc38_31.4 (constants.%specific_impl_fn.e23)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %ptr: type = ptr_type %as_type [symbolic = %ptr (constants.%ptr.530)]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.loc38_31.5) [symbolic = %Iterate.WithSelf.Next.type (constants.%Iterate.WithSelf.Next.type.10a)]
// CHECK:STDOUT:   %.loc38_31.21: type = fn_type_with_self_type %Iterate.WithSelf.Next.type, %Iterate.facet.loc38_31.5 [symbolic = %.loc38_31.21 (constants.%.df57)]
// CHECK:STDOUT:   %impl.elem3.loc38_31.2: @TestIterate.%.loc38_31.21 (%.df57) = impl_witness_access %Iterate.lookup_impl_witness, element3 [symbolic = %impl.elem3.loc38_31.2 (constants.%impl.elem3.199)]
// CHECK:STDOUT:   %specific_impl_fn.loc38_31.5: <specific function> = specific_impl_function %impl.elem3.loc38_31.2, @Iterate.WithSelf.Next(%Iterate.facet.loc38_31.5) [symbolic = %specific_impl_fn.loc38_31.5 (constants.%specific_impl_fn.169)]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%impl.elem1) [symbolic = %Destroy.WithSelf.Op.type (constants.%Destroy.WithSelf.Op.type.e7e)]
// CHECK:STDOUT:   %.loc38_31.22: type = fn_type_with_self_type %Destroy.WithSelf.Op.type, %impl.elem1 [symbolic = %.loc38_31.22 (constants.%.bc1)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %impl.elem1, @Destroy [symbolic = %Destroy.lookup_impl_witness (constants.%Destroy.lookup_impl_witness.f37)]
// CHECK:STDOUT:   %impl.elem0.loc38_31.2: @TestIterate.%.loc38_31.22 (%.bc1) = impl_witness_access %Destroy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc38_31.2 (constants.%impl.elem0.c7c)]
// CHECK:STDOUT:   %specific_impl_fn.loc38_31.6: <specific function> = specific_impl_function %impl.elem0.loc38_31.2, @Destroy.WithSelf.Op(%impl.elem1) [symbolic = %specific_impl_fn.loc38_31.6 (constants.%specific_impl_fn.cd3)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%r.param: @TestIterate.%R.as_type.loc34_74.1 (%R.as_type.a62)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %i.patt: %pattern_type.e41 = value_binding_pattern i [concrete = constants.%i.patt]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %r.ref: @TestIterate.%R.as_type.loc34_74.1 (%R.as_type.a62) = name_ref r, %r
// CHECK:STDOUT:     %impl.elem2.loc38_31.1: @TestIterate.%.loc38_31.20 (%.a27) = impl_witness_access constants.%Iterate.lookup_impl_witness.8b4, element2 [symbolic = %impl.elem2.loc38_31.2 (constants.%impl.elem2.7c0)]
// CHECK:STDOUT:     %bound_method.loc38_31.1: <bound method> = bound_method %r.ref, %impl.elem2.loc38_31.1
// CHECK:STDOUT:     %Iterate.facet.loc38_31.1: %Iterate.type = facet_value constants.%R.as_type.a62, (constants.%Iterate.lookup_impl_witness.8b4) [symbolic = %Iterate.facet.loc38_31.5 (constants.%Iterate.facet.b19)]
// CHECK:STDOUT:     %.loc38_31.1: %Iterate.type = converted constants.%R.as_type.a62, %Iterate.facet.loc38_31.1 [symbolic = %Iterate.facet.loc38_31.5 (constants.%Iterate.facet.b19)]
// CHECK:STDOUT:     %Iterate.facet.loc38_31.2: %Iterate.type = facet_value constants.%R.as_type.a62, (constants.%Iterate.lookup_impl_witness.8b4) [symbolic = %Iterate.facet.loc38_31.5 (constants.%Iterate.facet.b19)]
// CHECK:STDOUT:     %.loc38_31.2: %Iterate.type = converted constants.%R.as_type.a62, %Iterate.facet.loc38_31.2 [symbolic = %Iterate.facet.loc38_31.5 (constants.%Iterate.facet.b19)]
// CHECK:STDOUT:     %specific_impl_fn.loc38_31.1: <specific function> = specific_impl_function %impl.elem2.loc38_31.1, @Iterate.WithSelf.NewCursor(constants.%Iterate.facet.b19) [symbolic = %specific_impl_fn.loc38_31.4 (constants.%specific_impl_fn.e23)]
// CHECK:STDOUT:     %bound_method.loc38_31.2: <bound method> = bound_method %r.ref, %specific_impl_fn.loc38_31.1
// CHECK:STDOUT:     %var: ref @TestIterate.%as_type (%as_type.862) = var invalid
// CHECK:STDOUT:     %Iterate.WithSelf.NewCursor.call: init @TestIterate.%as_type (%as_type.862) to %var = call %bound_method.loc38_31.2(%r.ref)
// CHECK:STDOUT:     assign %var, %Iterate.WithSelf.NewCursor.call
// CHECK:STDOUT:     br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.next:
// CHECK:STDOUT:     %addr.loc38: @TestIterate.%ptr (%ptr.530) = addr_of %var
// CHECK:STDOUT:     %impl.elem3.loc38_31.1: @TestIterate.%.loc38_31.21 (%.df57) = impl_witness_access constants.%Iterate.lookup_impl_witness.8b4, element3 [symbolic = %impl.elem3.loc38_31.2 (constants.%impl.elem3.199)]
// CHECK:STDOUT:     %bound_method.loc38_31.3: <bound method> = bound_method %r.ref, %impl.elem3.loc38_31.1
// CHECK:STDOUT:     %Iterate.facet.loc38_31.3: %Iterate.type = facet_value constants.%R.as_type.a62, (constants.%Iterate.lookup_impl_witness.8b4) [symbolic = %Iterate.facet.loc38_31.5 (constants.%Iterate.facet.b19)]
// CHECK:STDOUT:     %.loc38_31.3: %Iterate.type = converted constants.%R.as_type.a62, %Iterate.facet.loc38_31.3 [symbolic = %Iterate.facet.loc38_31.5 (constants.%Iterate.facet.b19)]
// CHECK:STDOUT:     %Iterate.facet.loc38_31.4: %Iterate.type = facet_value constants.%R.as_type.a62, (constants.%Iterate.lookup_impl_witness.8b4) [symbolic = %Iterate.facet.loc38_31.5 (constants.%Iterate.facet.b19)]
// CHECK:STDOUT:     %.loc38_31.4: %Iterate.type = converted constants.%R.as_type.a62, %Iterate.facet.loc38_31.4 [symbolic = %Iterate.facet.loc38_31.5 (constants.%Iterate.facet.b19)]
// CHECK:STDOUT:     %.loc38_31.5: %Destroy.type = converted constants.%as_type.862, constants.%impl.elem1.46a [symbolic = %impl.elem1 (constants.%impl.elem1.46a)]
// CHECK:STDOUT:     %.loc38_31.6: %Destroy.type = converted constants.%as_type.862, constants.%impl.elem1.46a [symbolic = %impl.elem1 (constants.%impl.elem1.46a)]
// CHECK:STDOUT:     %specific_impl_fn.loc38_31.2: <specific function> = specific_impl_function %impl.elem3.loc38_31.1, @Iterate.WithSelf.Next(constants.%Iterate.facet.b19) [symbolic = %specific_impl_fn.loc38_31.5 (constants.%specific_impl_fn.169)]
// CHECK:STDOUT:     %bound_method.loc38_31.4: <bound method> = bound_method %r.ref, %specific_impl_fn.loc38_31.2
// CHECK:STDOUT:     %.loc38_31.7: ref %Optional.552 = temporary_storage
// CHECK:STDOUT:     %Iterate.WithSelf.Next.call: init %Optional.552 to %.loc38_31.7 = call %bound_method.loc38_31.4(%r.ref, %addr.loc38)
// CHECK:STDOUT:     %.loc38_31.8: ref %Optional.552 = temporary %.loc38_31.7, %Iterate.WithSelf.Next.call
// CHECK:STDOUT:     %.loc38_31.9: %Optional.HasValue.type.109 = specific_constant imports.%Core.import_ref.bb6, @Optional(constants.%OptionalStorage.facet.7ce) [concrete = constants.%Optional.HasValue.605]
// CHECK:STDOUT:     %HasValue.ref: %Optional.HasValue.type.109 = name_ref HasValue, %.loc38_31.9 [concrete = constants.%Optional.HasValue.605]
// CHECK:STDOUT:     %Optional.HasValue.bound: <bound method> = bound_method %.loc38_31.8, %HasValue.ref
// CHECK:STDOUT:     %Optional.HasValue.specific_fn: <specific function> = specific_function %HasValue.ref, @Optional.HasValue(constants.%OptionalStorage.facet.7ce) [concrete = constants.%Optional.HasValue.specific_fn]
// CHECK:STDOUT:     %bound_method.loc38_31.5: <bound method> = bound_method %.loc38_31.8, %Optional.HasValue.specific_fn
// CHECK:STDOUT:     %.loc38_31.10: %Optional.552 = acquire_value %.loc38_31.8
// CHECK:STDOUT:     %Optional.HasValue.call: init bool = call %bound_method.loc38_31.5(%.loc38_31.10)
// CHECK:STDOUT:     %.loc38_31.11: bool = value_of_initializer %Optional.HasValue.call
// CHECK:STDOUT:     %.loc38_31.12: bool = converted %Optional.HasValue.call, %.loc38_31.11
// CHECK:STDOUT:     if %.loc38_31.12 br !for.body else br !for.done
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.body:
// CHECK:STDOUT:     %.loc38_31.13: %Optional.Get.type.135 = specific_constant imports.%Core.import_ref.daba, @Optional(constants.%OptionalStorage.facet.7ce) [concrete = constants.%Optional.Get.5aa]
// CHECK:STDOUT:     %Get.ref: %Optional.Get.type.135 = name_ref Get, %.loc38_31.13 [concrete = constants.%Optional.Get.5aa]
// CHECK:STDOUT:     %Optional.Get.bound: <bound method> = bound_method %.loc38_31.8, %Get.ref
// CHECK:STDOUT:     %Optional.Get.specific_fn: <specific function> = specific_function %Get.ref, @Optional.Get(constants.%OptionalStorage.facet.7ce) [concrete = constants.%Optional.Get.specific_fn]
// CHECK:STDOUT:     %bound_method.loc38_31.6: <bound method> = bound_method %.loc38_31.8, %Optional.Get.specific_fn
// CHECK:STDOUT:     %.loc38_31.14: ref %ValueType = temporary_storage
// CHECK:STDOUT:     %.loc38_31.15: %Optional.552 = acquire_value %.loc38_31.8
// CHECK:STDOUT:     %Optional.Get.call: init %ValueType to %.loc38_31.14 = call %bound_method.loc38_31.6(%.loc38_31.15)
// CHECK:STDOUT:     %.loc38_31.16: ref %ValueType = temporary %.loc38_31.14, %Optional.Get.call
// CHECK:STDOUT:     %.loc38_31.17: %ValueType = acquire_value %.loc38_31.16
// CHECK:STDOUT:     %.loc38_16: type = splice_block %ValueType.ref.loc38 [concrete = constants.%ValueType] {
// CHECK:STDOUT:       %Cpp.ref.loc38: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:       %N.ref.loc38: <namespace> = name_ref N, imports.%N [concrete = imports.%N]
// CHECK:STDOUT:       %ValueType.ref.loc38: type = name_ref ValueType, imports.%ValueType.decl [concrete = constants.%ValueType]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %i: %ValueType = wrapper_binding i, %.loc38_31.17
// CHECK:STDOUT:     %sum.ref: ref %ValueType = name_ref sum, %sum
// CHECK:STDOUT:     %i.ref: %ValueType = name_ref i, %i
// CHECK:STDOUT:     %.loc39_12: ref %ValueType = value_as_ref %i.ref
// CHECK:STDOUT:     %addr.loc39: %ptr.9d3 = addr_of %.loc39_12
// CHECK:STDOUT:     %.loc39_9.1: %ptr.32e = as_compatible %addr.loc39
// CHECK:STDOUT:     %.loc39_9.2: %ptr.32e = converted %addr.loc39, %.loc39_9.1
// CHECK:STDOUT:     %operator_PlusEqual__carbon_thunk.call: ref %ValueType = call imports.%operator_PlusEqual__carbon_thunk.decl(%sum.ref, %.loc39_9.2)
// CHECK:STDOUT:     br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.done:
// CHECK:STDOUT:     %ValueType.Op.bound.loc38: <bound method> = bound_method %.loc38_31.16, constants.%ValueType.Op.4c94b9.1
// CHECK:STDOUT:     %Op.ref.loc38: %ValueType.cpp_destructor.type = name_ref Op, imports.%ValueType.cpp_destructor.decl [concrete = constants.%ValueType.cpp_destructor]
// CHECK:STDOUT:     %ValueType.cpp_destructor.bound.loc38: <bound method> = bound_method %.loc38_31.16, %Op.ref.loc38
// CHECK:STDOUT:     %ValueType.cpp_destructor.call.loc38: init %empty_tuple.type = call %ValueType.cpp_destructor.bound.loc38(%.loc38_31.16)
// CHECK:STDOUT:     %Destroy.Op.bound: <bound method> = bound_method %.loc38_31.8, constants.%Destroy.Op.1a2547.7
// CHECK:STDOUT:     %Destroy.Op.call: init %empty_tuple.type = call %Destroy.Op.bound(%.loc38_31.8)
// CHECK:STDOUT:     %impl.elem0.loc38_31.1: @TestIterate.%.loc38_31.22 (%.bc1) = impl_witness_access constants.%Destroy.lookup_impl_witness.f37, element0 [symbolic = %impl.elem0.loc38_31.2 (constants.%impl.elem0.c7c)]
// CHECK:STDOUT:     %bound_method.loc38_31.7: <bound method> = bound_method %var, %impl.elem0.loc38_31.1
// CHECK:STDOUT:     %.loc38_31.18: %Destroy.type = converted constants.%as_type.862, constants.%impl.elem1.46a [symbolic = %impl.elem1 (constants.%impl.elem1.46a)]
// CHECK:STDOUT:     %.loc38_31.19: %Destroy.type = converted constants.%as_type.862, constants.%impl.elem1.46a [symbolic = %impl.elem1 (constants.%impl.elem1.46a)]
// CHECK:STDOUT:     %specific_impl_fn.loc38_31.3: <specific function> = specific_impl_function %impl.elem0.loc38_31.1, @Destroy.WithSelf.Op(constants.%impl.elem1.46a) [symbolic = %specific_impl_fn.loc38_31.6 (constants.%specific_impl_fn.cd3)]
// CHECK:STDOUT:     %bound_method.loc38_31.8: <bound method> = bound_method %var, %specific_impl_fn.loc38_31.3
// CHECK:STDOUT:     %Destroy.WithSelf.Op.call: init %empty_tuple.type = call %bound_method.loc38_31.8(%var)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc38_31.1(%self.param: ref %.b36) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc38_31.2(%self.param: ref %MaybeUnformed.669) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc38_31.3(%self.param: ref bool) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc38_31.4(%self.param: ref %struct_type.value.has_value.5a1) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc38_31.5(%self.param: ref %DefaultOptionalStorage.257) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc38_31.6(%self.param: ref %Optional.552) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestRangeFor(%mc.param: ref %MutableAndConstRange) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %i.patt: %pattern_type.e41 = value_binding_pattern i [concrete = constants.%i.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %mc.ref: ref %MutableAndConstRange = name_ref mc, %mc
// CHECK:STDOUT:   %impl.elem2: %.7e3 = impl_witness_access constants.%Iterate.impl_witness.a6e, element2 [concrete = constants.%R.as_type.as.Iterate.impl.NewCursor.96c]
// CHECK:STDOUT:   %bound_method.loc80_32.1: <bound method> = bound_method %mc.ref, %impl.elem2
// CHECK:STDOUT:   %facet_value.loc80_32.1: %CppRangeForIterate_where.type = facet_value constants.%MutableAndConstRange, (constants.%custom_witness.fef, constants.%custom_witness.c19, constants.%custom_witness.a40, constants.%custom_witness.04a, constants.%custom_witness.cfc, constants.%custom_witness.dd5, constants.%custom_witness.c66, constants.%custom_witness.01a) [concrete = constants.%facet_value.88d]
// CHECK:STDOUT:   %.loc80_32.1: %CppRangeForIterate_where.type = converted constants.%MutableAndConstRange, %facet_value.loc80_32.1 [concrete = constants.%facet_value.88d]
// CHECK:STDOUT:   %facet_value.loc80_32.2: %CppRangeForIterate_where.type = facet_value constants.%MutableAndConstRange, (constants.%custom_witness.fef, constants.%custom_witness.c19, constants.%custom_witness.a40, constants.%custom_witness.04a, constants.%custom_witness.cfc, constants.%custom_witness.dd5, constants.%custom_witness.c66, constants.%custom_witness.01a) [concrete = constants.%facet_value.88d]
// CHECK:STDOUT:   %.loc80_32.2: %CppRangeForIterate_where.type = converted constants.%MutableAndConstRange, %facet_value.loc80_32.2 [concrete = constants.%facet_value.88d]
// CHECK:STDOUT:   %specific_fn.loc80_32.1: <specific function> = specific_function %impl.elem2, @R.as_type.as.Iterate.impl.NewCursor(constants.%facet_value.88d) [concrete = constants.%R.as_type.as.Iterate.impl.NewCursor.specific_fn]
// CHECK:STDOUT:   %bound_method.loc80_32.2: <bound method> = bound_method %mc.ref, %specific_fn.loc80_32.1
// CHECK:STDOUT:   %var: ref %tuple.type.56e = var invalid
// CHECK:STDOUT:   %.loc80_30.1: %MutableAndConstRange = acquire_value %mc.ref
// CHECK:STDOUT:   %R.as_type.as.Iterate.impl.NewCursor.call: init %tuple.type.56e to %var = call %bound_method.loc80_32.2(%.loc80_30.1)
// CHECK:STDOUT:   assign %var, %R.as_type.as.Iterate.impl.NewCursor.call
// CHECK:STDOUT:   br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT: !for.next:
// CHECK:STDOUT:   %addr.loc80: %ptr.484 = addr_of %var
// CHECK:STDOUT:   %impl.elem3: %.fef = impl_witness_access constants.%Iterate.impl_witness.a6e, element3 [concrete = constants.%R.as_type.as.Iterate.impl.Next.683]
// CHECK:STDOUT:   %bound_method.loc80_32.3: <bound method> = bound_method %mc.ref, %impl.elem3
// CHECK:STDOUT:   %facet_value.loc80_32.3: %CppRangeForIterate_where.type = facet_value constants.%MutableAndConstRange, (constants.%custom_witness.fef, constants.%custom_witness.c19, constants.%custom_witness.a40, constants.%custom_witness.04a, constants.%custom_witness.cfc, constants.%custom_witness.dd5, constants.%custom_witness.c66, constants.%custom_witness.01a) [concrete = constants.%facet_value.88d]
// CHECK:STDOUT:   %.loc80_32.3: %CppRangeForIterate_where.type = converted constants.%MutableAndConstRange, %facet_value.loc80_32.3 [concrete = constants.%facet_value.88d]
// CHECK:STDOUT:   %facet_value.loc80_32.4: %CppRangeForIterate_where.type = facet_value constants.%MutableAndConstRange, (constants.%custom_witness.fef, constants.%custom_witness.c19, constants.%custom_witness.a40, constants.%custom_witness.04a, constants.%custom_witness.cfc, constants.%custom_witness.dd5, constants.%custom_witness.c66, constants.%custom_witness.01a) [concrete = constants.%facet_value.88d]
// CHECK:STDOUT:   %.loc80_32.4: %CppRangeForIterate_where.type = converted constants.%MutableAndConstRange, %facet_value.loc80_32.4 [concrete = constants.%facet_value.88d]
// CHECK:STDOUT:   %specific_fn.loc80_32.2: <specific function> = specific_function %impl.elem3, @R.as_type.as.Iterate.impl.Next(constants.%facet_value.88d) [concrete = constants.%R.as_type.as.Iterate.impl.Next.specific_fn]
// CHECK:STDOUT:   %bound_method.loc80_32.4: <bound method> = bound_method %mc.ref, %specific_fn.loc80_32.2
// CHECK:STDOUT:   %.loc80_32.5: ref %Optional.552 = temporary_storage
// CHECK:STDOUT:   %.loc80_30.2: %MutableAndConstRange = acquire_value %mc.ref
// CHECK:STDOUT:   %R.as_type.as.Iterate.impl.Next.call: init %Optional.552 to %.loc80_32.5 = call %bound_method.loc80_32.4(%.loc80_30.2, %addr.loc80)
// CHECK:STDOUT:   %.loc80_32.6: ref %Optional.552 = temporary %.loc80_32.5, %R.as_type.as.Iterate.impl.Next.call
// CHECK:STDOUT:   %.loc80_32.7: %Optional.HasValue.type.109 = specific_constant imports.%Core.import_ref.bb6, @Optional(constants.%OptionalStorage.facet.7ce) [concrete = constants.%Optional.HasValue.605]
// CHECK:STDOUT:   %HasValue.ref: %Optional.HasValue.type.109 = name_ref HasValue, %.loc80_32.7 [concrete = constants.%Optional.HasValue.605]
// CHECK:STDOUT:   %Optional.HasValue.bound: <bound method> = bound_method %.loc80_32.6, %HasValue.ref
// CHECK:STDOUT:   %Optional.HasValue.specific_fn: <specific function> = specific_function %HasValue.ref, @Optional.HasValue(constants.%OptionalStorage.facet.7ce) [concrete = constants.%Optional.HasValue.specific_fn]
// CHECK:STDOUT:   %bound_method.loc80_32.5: <bound method> = bound_method %.loc80_32.6, %Optional.HasValue.specific_fn
// CHECK:STDOUT:   %.loc80_32.8: %Optional.552 = acquire_value %.loc80_32.6
// CHECK:STDOUT:   %Optional.HasValue.call: init bool = call %bound_method.loc80_32.5(%.loc80_32.8)
// CHECK:STDOUT:   %.loc80_32.9: bool = value_of_initializer %Optional.HasValue.call
// CHECK:STDOUT:   %.loc80_32.10: bool = converted %Optional.HasValue.call, %.loc80_32.9
// CHECK:STDOUT:   if %.loc80_32.10 br !for.body else br !for.done
// CHECK:STDOUT:
// CHECK:STDOUT: !for.body:
// CHECK:STDOUT:   %.loc80_32.11: %Optional.Get.type.135 = specific_constant imports.%Core.import_ref.daba, @Optional(constants.%OptionalStorage.facet.7ce) [concrete = constants.%Optional.Get.5aa]
// CHECK:STDOUT:   %Get.ref: %Optional.Get.type.135 = name_ref Get, %.loc80_32.11 [concrete = constants.%Optional.Get.5aa]
// CHECK:STDOUT:   %Optional.Get.bound: <bound method> = bound_method %.loc80_32.6, %Get.ref
// CHECK:STDOUT:   %Optional.Get.specific_fn: <specific function> = specific_function %Get.ref, @Optional.Get(constants.%OptionalStorage.facet.7ce) [concrete = constants.%Optional.Get.specific_fn]
// CHECK:STDOUT:   %bound_method.loc80_32.6: <bound method> = bound_method %.loc80_32.6, %Optional.Get.specific_fn
// CHECK:STDOUT:   %.loc80_32.12: ref %ValueType = temporary_storage
// CHECK:STDOUT:   %.loc80_32.13: %Optional.552 = acquire_value %.loc80_32.6
// CHECK:STDOUT:   %Optional.Get.call: init %ValueType to %.loc80_32.12 = call %bound_method.loc80_32.6(%.loc80_32.13)
// CHECK:STDOUT:   %.loc80_32.14: ref %ValueType = temporary %.loc80_32.12, %Optional.Get.call
// CHECK:STDOUT:   %.loc80_32.15: %ValueType = acquire_value %.loc80_32.14
// CHECK:STDOUT:   %.loc80_16: type = splice_block %ValueType.ref.loc80 [concrete = constants.%ValueType] {
// CHECK:STDOUT:     %Cpp.ref.loc80: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %N.ref.loc80: <namespace> = name_ref N, imports.%N [concrete = imports.%N]
// CHECK:STDOUT:     %ValueType.ref.loc80: type = name_ref ValueType, imports.%ValueType.decl [concrete = constants.%ValueType]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %i: %ValueType = wrapper_binding i, %.loc80_32.15
// CHECK:STDOUT:   %sum.ref: ref %ValueType = name_ref sum, %sum
// CHECK:STDOUT:   %i.ref: %ValueType = name_ref i, %i
// CHECK:STDOUT:   %.loc81_12: ref %ValueType = value_as_ref %i.ref
// CHECK:STDOUT:   %addr.loc81: %ptr.9d3 = addr_of %.loc81_12
// CHECK:STDOUT:   %.loc81_9.1: %ptr.32e = as_compatible %addr.loc81
// CHECK:STDOUT:   %.loc81_9.2: %ptr.32e = converted %addr.loc81, %.loc81_9.1
// CHECK:STDOUT:   %operator_PlusEqual__carbon_thunk.call: ref %ValueType = call imports.%operator_PlusEqual__carbon_thunk.decl(%sum.ref, %.loc81_9.2)
// CHECK:STDOUT:   br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT: !for.done:
// CHECK:STDOUT:   %ValueType.Op.bound.loc80: <bound method> = bound_method %.loc80_32.14, constants.%ValueType.Op.4c94b9.1
// CHECK:STDOUT:   %Op.ref.loc80: %ValueType.cpp_destructor.type = name_ref Op, imports.%ValueType.cpp_destructor.decl [concrete = constants.%ValueType.cpp_destructor]
// CHECK:STDOUT:   %ValueType.cpp_destructor.bound.loc80: <bound method> = bound_method %.loc80_32.14, %Op.ref.loc80
// CHECK:STDOUT:   %ValueType.cpp_destructor.call.loc80: init %empty_tuple.type = call %ValueType.cpp_destructor.bound.loc80(%.loc80_32.14)
// CHECK:STDOUT:   %Destroy.Op.bound.loc80_32.1: <bound method> = bound_method %.loc80_32.6, constants.%Destroy.Op.1a2547.7
// CHECK:STDOUT:   %Destroy.Op.call.loc80_32.1: init %empty_tuple.type = call %Destroy.Op.bound.loc80_32.1(%.loc80_32.6)
// CHECK:STDOUT:   %Destroy.Op.bound.loc80_32.2: <bound method> = bound_method %var, constants.%Destroy.Op.1a2547.8
// CHECK:STDOUT:   %Destroy.Op.call.loc80_32.2: init %empty_tuple.type = call %Destroy.Op.bound.loc80_32.2(%var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestIterate(constants.%R.623) {
// CHECK:STDOUT:   %R.patt.loc34_17.2 => constants.%R.patt.324
// CHECK:STDOUT:   %R.loc34_17.1 => constants.%R.623
// CHECK:STDOUT:   %R.as_type.loc34_74.1 => constants.%R.as_type.a62
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.725
// CHECK:STDOUT:   %r.param_patt.loc34_72.2 => constants.%r.param_patt.162
// CHECK:STDOUT:   %r.patt.loc34_72.2 => constants.%r.patt.15b
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestIterate(constants.%facet_value.0ec) {
// CHECK:STDOUT:   %R.patt.loc34_17.2 => constants.%R.patt.324
// CHECK:STDOUT:   %R.loc34_17.1 => constants.%facet_value.0ec
// CHECK:STDOUT:   %R.as_type.loc34_74.1 => constants.%MutableAndConstRange
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.394
// CHECK:STDOUT:   %r.param_patt.loc34_72.2 => constants.%r.param_patt.8bb
// CHECK:STDOUT:   %r.patt.loc34_72.2 => constants.%r.patt.ac4
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc34 => constants.%complete_type.357
// CHECK:STDOUT:   %Iterate.lookup_impl_witness => constants.%Iterate.impl_witness.a6e
// CHECK:STDOUT:   %Iterate.facet.loc38_31.5 => constants.%Iterate.facet.5cc
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type => constants.%Iterate.WithSelf.NewCursor.type.875
// CHECK:STDOUT:   %.loc38_31.20 => constants.%.7e3
// CHECK:STDOUT:   %impl.elem2.loc38_31.2 => constants.%R.as_type.as.Iterate.impl.NewCursor.96c
// CHECK:STDOUT:   %specific_impl_fn.loc38_31.4 => constants.%R.as_type.as.Iterate.impl.NewCursor.specific_fn
// CHECK:STDOUT:   %impl.elem1 => constants.%Destroy.facet.7aa
// CHECK:STDOUT:   %as_type => constants.%tuple.type.56e
// CHECK:STDOUT:   %require_complete.1 => constants.%complete_type.1b5
// CHECK:STDOUT:   %ptr => constants.%ptr.484
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type => constants.%Iterate.WithSelf.Next.type.b41
// CHECK:STDOUT:   %.loc38_31.21 => constants.%.fef
// CHECK:STDOUT:   %impl.elem3.loc38_31.2 => constants.%R.as_type.as.Iterate.impl.Next.683
// CHECK:STDOUT:   %specific_impl_fn.loc38_31.5 => constants.%R.as_type.as.Iterate.impl.Next.specific_fn
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.b7b
// CHECK:STDOUT:   %.loc38_31.22 => constants.%.1ad
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.df9cc1.8
// CHECK:STDOUT:   %impl.elem0.loc38_31.2 => constants.%Destroy.Op.1a2547.8
// CHECK:STDOUT:   %specific_impl_fn.loc38_31.6 => constants.%Destroy.Op.1a2547.8
// CHECK:STDOUT: }
// CHECK:STDOUT:
