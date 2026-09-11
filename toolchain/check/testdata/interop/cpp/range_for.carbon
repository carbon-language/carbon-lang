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

fn TestIterate[R: Core.Iterate where .ElementType = i32](r: R) {
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
struct DiffValueType {
  auto operator+=(const DiffValueType&) -> DiffValueType&;
};

class DiffMutableRange {
  struct Iterator {
    auto operator*() const -> DiffValueType&;
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

class DiffConstRange {
  struct Iterator {
    auto operator*() const -> DiffValueType;
    auto operator++() -> Iterator&;
  };

  struct Sentinel {
    friend auto operator==(Iterator, Sentinel) -> bool;
    friend auto operator!=(Iterator, Sentinel) -> bool;
  };
public:
  auto begin() const -> Iterator;
  auto end() -> Sentinel;
};
''';

fn TestIterate[R: Core.Iterate where .ElementType = Cpp.DiffValueType](var r: R) {
  var sum: Cpp.DiffValueType = Cpp.DiffValueType.DiffValueType();

  //@dump-sem-ir-begin
  for (i: Cpp.DiffValueType in r) {
    sum += i;
  }
  //@dump-sem-ir-end
}

fn TestIterateDriver(var m: Cpp.DiffMutableRange, c: Cpp.DiffConstRange) {
  TestIterate(m);
  TestIterate(c);
}

fn TestRangeFor(var m: Cpp.DiffMutableRange, c: Cpp.DiffConstRange) {
  var sum: Cpp.DiffValueType = Cpp.DiffValueType.DiffValueType();

  //@dump-sem-ir-begin
  for (i: Cpp.DiffValueType in m) {
    sum += i;
  }

  for (i: Cpp.DiffValueType in c) {
    sum += i;
  }
  //@dump-sem-ir-end
}

// --- method_overloads.carbon

library "[[@TEST_NAME]]";

import Cpp inline '''c++
struct OverloadValueType {};

class Range {
  template<typename T>
  struct Iterator {
    auto operator*() const -> T&;
    auto operator++() -> Iterator&;
    friend auto operator==(Iterator, Iterator) -> bool;
    friend auto operator!=(Iterator, Iterator) -> bool;
  };

public:
  auto begin() const -> Iterator<const OverloadValueType>;
  auto end() const -> Iterator<const OverloadValueType>;

  auto begin() -> Iterator<OverloadValueType>;
  auto end() -> Iterator<OverloadValueType>;
};
''';

fn TestIterate[R: Core.Iterate where .ElementType = Cpp.OverloadValueType](var r: R) {
  //@dump-sem-ir-begin
  for (_: Cpp.OverloadValueType in r) {}
  //@dump-sem-ir-end
}

fn TestIterateDriver(r: Cpp.Range) {
  TestIterate(r);
}

fn TestRangeFor(var r: Cpp.Range) {
  //@dump-sem-ir-begin
  for (_: Cpp.OverloadValueType in r) {}
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

fn TestIterate[R: Core.Iterate where .ElementType = f64](r: R) {
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

fn TestIterate[R: Core.Iterate where .ElementType = f64](r: R) {
  var sum: f64 = 0.0;

  //@dump-sem-ir-begin
  for (i: f64 in r) {
    sum += i;
  }
  //@dump-sem-ir-end
}

fn TestIterateDriver(var m: Cpp.N.MutableRange) {
  // CHECK:STDERR: fail_todo_adl_returns_same_types_mutable.carbon:[[@LINE+28]]:3: error: facet type `Core.CppRange where .(Core.CppRangeForIterate.Iterator).(Core.CppUnsafeDeref.Result) impls Core.Copy` can not be identified [ImplLookupInUnidentifiedFacetType]
  // CHECK:STDERR:   TestIterate(m);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_adl_returns_same_types_mutable.carbon: note: value expression passed to reference parameter [ValueForRefParam]
  // CHECK:STDERR: fail_todo_adl_returns_same_types_mutable.carbon:[[@LINE-21]]:34: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR:   friend auto begin(MutableRange&) -> Iterator;
  // CHECK:STDERR:                                  ^
  // CHECK:STDERR: {{.*}}/prelude/iterate.carbon:43:3: note: while building thunk to match the signature of this function [ThunkSignature]
  // CHECK:STDERR:   fn Begin(self) -> Iterator;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_adl_returns_same_types_mutable.carbon:[[@LINE-21]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn TestIterate[R: Core.Iterate where .ElementType = f64](r: R) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_todo_adl_returns_same_types_mutable.carbon:[[@LINE+14]]:3: error: facet type `Core.CppRange where .(Core.CppRangeForIterate.Iterator).(Core.CppUnsafeDeref.Result) impls Core.Copy` can not be identified [ImplLookupInUnidentifiedFacetType]
  // CHECK:STDERR:   TestIterate(m);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_adl_returns_same_types_mutable.carbon: note: value expression passed to reference parameter [ValueForRefParam]
  // CHECK:STDERR: fail_todo_adl_returns_same_types_mutable.carbon:[[@LINE-34]]:32: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR:   friend auto end(MutableRange&) -> Iterator;
  // CHECK:STDERR:                                ^
  // CHECK:STDERR: {{.*}}/prelude/iterate.carbon:44:3: note: while building thunk to match the signature of this function [ThunkSignature]
  // CHECK:STDERR:   fn End(self) -> Sentinel;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_adl_returns_same_types_mutable.carbon:[[@LINE-35]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn TestIterate[R: Core.Iterate where .ElementType = f64](r: R) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
namespace DiffADL {
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
} // namespace DiffADL
''';

fn TestIterate[R: Core.Iterate where .ElementType = Cpp.DiffADL.ValueType](r: R) {
  var sum: Cpp.DiffADL.ValueType = Cpp.DiffADL.ValueType.ValueType();

  //@dump-sem-ir-begin
  for (i: Cpp.DiffADL.ValueType in r) {
    sum += i;
  }
  //@dump-sem-ir-end
}

fn TestIterateDriver(c: Cpp.DiffADL.ConstRange) {
  TestIterate(c);
}

fn TestRangeFor(c: Cpp.DiffADL.ConstRange) {
  var sum: Cpp.DiffADL.ValueType = Cpp.DiffADL.ValueType.ValueType();

  //@dump-sem-ir-begin
  for (i: Cpp.DiffADL.ValueType in c) {
    sum += i;
  }
  //@dump-sem-ir-end
}

// --- fail_todo_adl_returns_different_types_mutable.carbon

library "[[@TEST_NAME]]";

import Cpp inline '''c++
namespace DiffMutableADL {
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

fn TestIterate[R: Core.Iterate where .ElementType = Cpp.DiffMutableADL.ValueType](r: R) {
  var sum: Cpp.DiffMutableADL.ValueType = Cpp.DiffMutableADL.ValueType.ValueType();

  //@dump-sem-ir-begin
  for (i: Cpp.DiffMutableADL.ValueType in r) {
    sum += i;
  }
  //@dump-sem-ir-end
}

fn TestIterateDriver(var m: Cpp.DiffMutableADL.MutableRange) {
  // CHECK:STDERR: fail_todo_adl_returns_different_types_mutable.carbon:[[@LINE+45]]:3: error: facet type `Core.CppRange where .(Core.CppRangeForIterate.Iterator).(Core.CppUnsafeDeref.Result) impls Core.Copy` can not be identified [ImplLookupInUnidentifiedFacetType]
  // CHECK:STDERR:   TestIterate(m);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_adl_returns_different_types_mutable.carbon: note: value expression passed to reference parameter [ValueForRefParam]
  // CHECK:STDERR: fail_todo_adl_returns_different_types_mutable.carbon:[[@LINE-21]]:34: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR:   friend auto begin(MutableRange&) -> Iterator;
  // CHECK:STDERR:                                  ^
  // CHECK:STDERR: {{.*}}/prelude/iterate.carbon:43:3: note: while building thunk to match the signature of this function [ThunkSignature]
  // CHECK:STDERR:   fn Begin(self) -> Iterator;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_adl_returns_different_types_mutable.carbon:[[@LINE-21]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn TestIterate[R: Core.Iterate where .ElementType = Cpp.DiffMutableADL.ValueType](r: R) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_todo_adl_returns_different_types_mutable.carbon:[[@LINE+31]]:3: error: facet type `Core.CppRange where .(Core.CppRangeForIterate.Iterator).(Core.CppUnsafeDeref.Result) impls Core.Copy` can not be identified [ImplLookupInUnidentifiedFacetType]
  // CHECK:STDERR:   TestIterate(m);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_adl_returns_different_types_mutable.carbon: note: value expression passed to reference parameter [ValueForRefParam]
  // CHECK:STDERR: fail_todo_adl_returns_different_types_mutable.carbon:[[@LINE-34]]:32: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR:   friend auto end(MutableRange&) -> Sentinel;
  // CHECK:STDERR:                                ^
  // CHECK:STDERR: {{.*}}/prelude/iterate.carbon:44:3: note: while building thunk to match the signature of this function [ThunkSignature]
  // CHECK:STDERR:   fn End(self) -> Sentinel;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_adl_returns_different_types_mutable.carbon:[[@LINE-35]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn TestIterate[R: Core.Iterate where .ElementType = Cpp.DiffMutableADL.ValueType](r: R) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_todo_adl_returns_different_types_mutable.carbon:[[@LINE+17]]:3: error: facet type `Core.CppRange where .(Core.CppRangeForIterate.Iterator).(Core.CppUnsafeDeref.Result) impls Core.Copy` can not be identified [ImplLookupInUnidentifiedFacetType]
  // CHECK:STDERR:   TestIterate(m);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_adl_returns_different_types_mutable.carbon:[[@LINE+14]]:3: note: semantics TODO: `Rewriting operator== using operator== is not supported` [SemanticsTodo]
  // CHECK:STDERR:   TestIterate(m);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_adl_returns_different_types_mutable.carbon:[[@LINE-45]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn TestIterate[R: Core.Iterate where .ElementType = Cpp.DiffMutableADL.ValueType](r: R) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_todo_adl_returns_different_types_mutable.carbon:[[@LINE+7]]:3: error: cannot convert type `Cpp.DiffMutableADL.MutableRange` into type implementing `Core.Iterate where .(Core.Iterate.ElementType) = Cpp.DiffMutableADL.ValueType as Core.Destroy & Core.Copy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   TestIterate(m);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_adl_returns_different_types_mutable.carbon:[[@LINE-52]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn TestIterate[R: Core.Iterate where .ElementType = Cpp.DiffMutableADL.ValueType](r: R) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  TestIterate(m);
}

fn TestRangeFor(var m: Cpp.DiffMutableADL.MutableRange) {
  var sum: Cpp.DiffMutableADL.ValueType = Cpp.DiffMutableADL.ValueType.ValueType();

  //@dump-sem-ir-begin
  // CHECK:STDERR: fail_todo_adl_returns_different_types_mutable.carbon:[[@LINE+4]]:7: error: cannot access member of interface `Core.Iterate` in type `Cpp.DiffMutableADL.MutableRange` that does not implement that interface [MissingImplInMemberAccess]
  // CHECK:STDERR:   for (i: Cpp.DiffMutableADL.ValueType in m) {
  // CHECK:STDERR:       ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  for (i: Cpp.DiffMutableADL.ValueType in m) {
    sum += i;
  }
  //@dump-sem-ir-end
}

// --- fail_todo_adl_overloads.carbon

import Cpp inline '''c++
namespace OverloadADL {
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

fn TestIterate[R: Core.Iterate where .ElementType = Cpp.OverloadADL.ValueType](r: R) {
  var sum: Cpp.OverloadADL.ValueType = Cpp.OverloadADL.ValueType.ValueType();

  //@dump-sem-ir-begin
  for (i: Cpp.OverloadADL.ValueType in r) {
    sum += i;
  }
  //@dump-sem-ir-end
}

fn TestIterateDriver(var mc: Cpp.OverloadADL.MutableAndConstRange) {
  // CHECK:STDERR: fail_todo_adl_overloads.carbon:[[@LINE+28]]:3: error: facet type `Core.CppRange where .(Core.CppRangeForIterate.Iterator).(Core.CppUnsafeDeref.Result) impls Core.Copy` can not be identified [ImplLookupInUnidentifiedFacetType]
  // CHECK:STDERR:   TestIterate(mc);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_adl_overloads.carbon: note: value expression passed to reference parameter [ValueForRefParam]
  // CHECK:STDERR: fail_todo_adl_overloads.carbon:[[@LINE-24]]:44: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR:     friend auto begin(MutableAndConstRange&) -> Iterator;
  // CHECK:STDERR:                                            ^
  // CHECK:STDERR: {{.*}}/prelude/iterate.carbon:43:3: note: while building thunk to match the signature of this function [ThunkSignature]
  // CHECK:STDERR:   fn Begin(self) -> Iterator;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_adl_overloads.carbon:[[@LINE-21]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn TestIterate[R: Core.Iterate where .ElementType = Cpp.OverloadADL.ValueType](r: R) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_todo_adl_overloads.carbon:[[@LINE+14]]:3: error: facet type `Core.CppRange where .(Core.CppRangeForIterate.Iterator).(Core.CppUnsafeDeref.Result) impls Core.Copy` can not be identified [ImplLookupInUnidentifiedFacetType]
  // CHECK:STDERR:   TestIterate(mc);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_adl_overloads.carbon: note: value expression passed to reference parameter [ValueForRefParam]
  // CHECK:STDERR: fail_todo_adl_overloads.carbon:[[@LINE-37]]:42: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR:     friend auto end(MutableAndConstRange&) -> Iterator;
  // CHECK:STDERR:                                          ^
  // CHECK:STDERR: {{.*}}/prelude/iterate.carbon:44:3: note: while building thunk to match the signature of this function [ThunkSignature]
  // CHECK:STDERR:   fn End(self) -> Sentinel;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_adl_overloads.carbon:[[@LINE-35]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn TestIterate[R: Core.Iterate where .ElementType = Cpp.OverloadADL.ValueType](r: R) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  TestIterate(mc);
}

fn TestRangeFor(var mc: Cpp.OverloadADL.MutableAndConstRange) {
  var sum: Cpp.OverloadADL.ValueType = Cpp.OverloadADL.ValueType.ValueType();

  //@dump-sem-ir-begin
  for (i: Cpp.OverloadADL.ValueType in mc) {
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

fn Test[R: Core.Iterate](unused r: R) {}

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
  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE+7]]:3: error: cannot convert type `Cpp.NoBeginEnd` into type implementing `Core.Iterate` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   Test(no_begin_end);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE-19]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn Test[R: Core.Iterate](unused r: R) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Test(no_begin_end);

  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE+7]]:3: error: cannot convert type `Cpp.NoBeginMethod` into type implementing `Core.Iterate` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   Test(no_begin_method);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE-28]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn Test[R: Core.Iterate](unused r: R) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Test(no_begin_method);

  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE+7]]:3: error: cannot convert type `Cpp.NoEndMethod` into type implementing `Core.Iterate` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   Test(no_end_method);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE-37]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn Test[R: Core.Iterate](unused r: R) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Test(no_end_method);

  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE+7]]:3: error: cannot convert type `Cpp.NoBeginADL` into type implementing `Core.Iterate` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   Test(no_begin_adl);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE-46]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn Test[R: Core.Iterate](unused r: R) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Test(no_begin_adl);

  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE+7]]:3: error: cannot convert type `Cpp.NoEndADL` into type implementing `Core.Iterate` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   Test(no_end_adl);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE-55]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn Test[R: Core.Iterate](unused r: R) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Test(no_end_adl);

  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE+7]]:3: error: cannot convert type `Cpp.BeginMethodEndADL` into type implementing `Core.Iterate` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   Test(begin_method_end_adl);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE-64]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn Test[R: Core.Iterate](unused r: R) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Test(begin_method_end_adl);

  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE+7]]:3: error: cannot convert type `Cpp.BeginADLEndMethod` into type implementing `Core.Iterate` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   Test(begin_adl_end_method);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE-73]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn Test[R: Core.Iterate](unused r: R) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Test(begin_adl_end_method);

  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE+7]]:3: error: cannot convert type `Cpp.BeginFieldEndMethod` into type implementing `Core.Iterate` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   Test(begin_field_end_method);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE-82]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn Test[R: Core.Iterate](unused r: R) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Test(begin_field_end_method);

  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE+7]]:3: error: cannot convert type `Cpp.BeginMethodEndField` into type implementing `Core.Iterate` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   Test(begin_method_end_field);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE-91]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn Test[R: Core.Iterate](unused r: R) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Test(begin_method_end_field);

  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE+20]]:3: error: facet type `Core.CppRange where .(Core.CppRangeForIterate.Iterator).(Core.CppUnsafeDeref.Result) impls Core.Copy` can not be identified [ImplLookupInUnidentifiedFacetType]
  // CHECK:STDERR:   Test(begin_method_deleted);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: {{.*}}/prelude/iterate.carbon:65:9: note: attempt to use a deleted function [CppInteropParseError]
  // CHECK:STDERR:    63 |   alias ValueType = Self.(Iterator).(CppIterator.ValueType);
  // CHECK:STDERR:       |         ^
  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE-123]]:8: note: 'begin' has been explicitly marked deleted here [CppInteropParseNote]
  // CHECK:STDERR:    44 |   auto begin() = delete;
  // CHECK:STDERR:       |        ^
  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE-106]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn Test[R: Core.Iterate](unused r: R) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE+7]]:3: error: cannot convert type `Cpp.BeginMethodDeleted` into type implementing `Core.Iterate` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   Test(begin_method_deleted);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE-113]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn Test[R: Core.Iterate](unused r: R) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Test(begin_method_deleted);

  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE+20]]:3: error: facet type `Core.CppRange where .(Core.CppRangeForIterate.Iterator).(Core.CppUnsafeDeref.Result) impls Core.Copy` can not be identified [ImplLookupInUnidentifiedFacetType]
  // CHECK:STDERR:   Test(end_method_deleted);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: {{.*}}/prelude/iterate.carbon:65:9: note: attempt to use a deleted function [CppInteropParseError]
  // CHECK:STDERR:    63 |   alias ValueType = Self.(Iterator).(CppIterator.ValueType);
  // CHECK:STDERR:       |         ^
  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE-139]]:8: note: 'end' has been explicitly marked deleted here [CppInteropParseNote]
  // CHECK:STDERR:    50 |   auto end() = delete;
  // CHECK:STDERR:       |        ^
  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE-128]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn Test[R: Core.Iterate](unused r: R) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE+7]]:3: error: cannot convert type `Cpp.EndMethodDeleted` into type implementing `Core.Iterate` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   Test(end_method_deleted);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE-135]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn Test[R: Core.Iterate](unused r: R) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Test(end_method_deleted);

  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE+20]]:3: error: facet type `Core.CppRange where .(Core.CppRangeForIterate.Iterator).(Core.CppUnsafeDeref.Result) impls Core.Copy` can not be identified [ImplLookupInUnidentifiedFacetType]
  // CHECK:STDERR:   Test(begin_function_deleted);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: {{.*}}/prelude/iterate.carbon:65:9: note: call to deleted function 'begin' [CppInteropParseError]
  // CHECK:STDERR:    63 |   alias ValueType = Self.(Iterator).(CppIterator.ValueType);
  // CHECK:STDERR:       |         ^~~~~~~~~
  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE-157]]:15: note: candidate function has been explicitly deleted [CppInteropParseNote]
  // CHECK:STDERR:    54 |   friend auto begin(const BeginFunctionDeleted&) = delete;
  // CHECK:STDERR:       |               ^
  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE-150]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn Test[R: Core.Iterate](unused r: R) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE+7]]:3: error: cannot convert type `Cpp.BeginFunctionDeleted` into type implementing `Core.Iterate` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   Test(begin_function_deleted);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE-157]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn Test[R: Core.Iterate](unused r: R) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Test(begin_function_deleted);

  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE+20]]:3: error: facet type `Core.CppRange where .(Core.CppRangeForIterate.Iterator).(Core.CppUnsafeDeref.Result) impls Core.Copy` can not be identified [ImplLookupInUnidentifiedFacetType]
  // CHECK:STDERR:   Test(end_function_deleted);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: {{.*}}/prelude/iterate.carbon:65:9: note: call to deleted function 'end' [CppInteropParseError]
  // CHECK:STDERR:    63 |   alias ValueType = Self.(Iterator).(CppIterator.ValueType);
  // CHECK:STDERR:       |         ^~~~~~~~~
  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE-173]]:15: note: candidate function has been explicitly deleted [CppInteropParseNote]
  // CHECK:STDERR:    60 |   friend auto end(const EndFunctionDeleted&) = delete;
  // CHECK:STDERR:       |               ^
  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE-172]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn Test[R: Core.Iterate](unused r: R) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE+7]]:3: error: cannot convert type `Cpp.EndFunctionDeleted` into type implementing `Core.Iterate` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   Test(end_function_deleted);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE-179]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn Test[R: Core.Iterate](unused r: R) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Get.type.a12: type = fn_type @T.as_type.as.OptionalStorage.impl.Get, @T.as_type.as.OptionalStorage.impl(%T.283) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Get.d32: %T.as_type.as.OptionalStorage.impl.Get.type.a12 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Has.type.e5a: type = fn_type @T.as_type.as.OptionalStorage.impl.Has, @T.as_type.as.OptionalStorage.impl(%T.283) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Has.4ee: %T.as_type.as.OptionalStorage.impl.Has.type.e5a = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Some.type.492: type = fn_type @T.as_type.as.OptionalStorage.impl.Some, @T.as_type.as.OptionalStorage.impl(%T.283) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Some.89a: %T.as_type.as.OptionalStorage.impl.Some.type.492 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.None.type.0a1: type = fn_type @T.as_type.as.OptionalStorage.impl.None, @T.as_type.as.OptionalStorage.impl(%T.283) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.None.e06: %T.as_type.as.OptionalStorage.impl.None.type.0a1 = struct_value () [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %N.fe9: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %Int.67af24.1: type = class_type @Int, @Int(%N.fe9) [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.6b6: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %.Self.500: %Iterate.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Int.67af24.2: type = class_type @Int, @Int(%To) [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.914: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %CppRange.type: type = facet_type <@CppRange> [concrete]
// CHECK:STDOUT:   %.Self.e0b: %CppRange.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %CppRangeForIterate.lookup_impl_witness.50b: <witness> = lookup_impl_witness %.Self.e0b, @CppRangeForIterate [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.32d: type = impl_witness_access %CppRangeForIterate.lookup_impl_witness.50b, element0 [symbolic_self]
// CHECK:STDOUT:   %CppUnsafeDeref.lookup_impl_witness.851: <witness> = lookup_impl_witness %impl.elem0.32d, @CppUnsafeDeref [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.10d: type = impl_witness_access %CppUnsafeDeref.lookup_impl_witness.851, element0 [symbolic_self]
// CHECK:STDOUT:   %CppRange_where.type.d682d6.1: type = facet_type <@CppRange where %impl.elem0.10d impls @Copy> [concrete]
// CHECK:STDOUT:   %T.f45530.2: %CppRange_where.type.d682d6.1 = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.type.b36: type = fn_type @T.as_type.as.Iterate.impl.Next, @T.as_type.as.Iterate.impl(%T.f45530.2) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.d6b: %T.as_type.as.Iterate.impl.Next.type.b36 = struct_value () [symbolic]
// CHECK:STDOUT:   %CppRangeForIterate.lookup_impl_witness.b581a8.2: <witness> = lookup_impl_witness %T.f45530.2, @CppRangeForIterate [symbolic]
// CHECK:STDOUT:   %impl.elem0.5f971c.2: type = impl_witness_access %CppRangeForIterate.lookup_impl_witness.b581a8.2, element0 [symbolic]
// CHECK:STDOUT:   %CppUnsafeDeref.lookup_impl_witness.87e543.2: <witness> = lookup_impl_witness %impl.elem0.5f971c.2, @CppUnsafeDeref [symbolic]
// CHECK:STDOUT:   %impl.elem0.493765.2: type = impl_witness_access %CppUnsafeDeref.lookup_impl_witness.87e543.2, element0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.type.953: type = fn_type @T.as_type.as.Iterate.impl.NewCursor, @T.as_type.as.Iterate.impl(%T.f45530.2) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.544: %T.as_type.as.Iterate.impl.NewCursor.type.953 = struct_value () [symbolic]
// CHECK:STDOUT:   %impl.elem1.c2dc15.2: type = impl_witness_access %CppRangeForIterate.lookup_impl_witness.b581a8.2, element1 [symbolic]
// CHECK:STDOUT:   %tuple.type.1ba: type = tuple_type (%impl.elem0.5f971c.2, %impl.elem1.c2dc15.2) [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.812: <witness> = lookup_impl_witness %tuple.type.1ba, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.9cd: %Destroy.type = facet_value %tuple.type.1ba, (%Destroy.lookup_impl_witness.812) [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.35816c.2: <witness> = lookup_impl_witness %impl.elem0.493765.2, @Destroy [symbolic]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.167: <witness> = lookup_impl_witness %impl.elem0.493765.2, @Copy [symbolic]
// CHECK:STDOUT:   %facet_value.f93: %facet_type.f48 = facet_value %impl.elem0.493765.2, (%Destroy.lookup_impl_witness.35816c.2, %Copy.lookup_impl_witness.167) [symbolic]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.3: type = fn_type @Destroy.Op.loc30_53.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.3: %Destroy.Op.type.1d8f74.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.df9cc1.3: <witness> = custom_witness (%Destroy.Op.1a2547.3), @Destroy [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.ac8: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N.fe9) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.5e0: %Int.as.Copy.impl.Op.type.ac8 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.b51: <witness> = impl_witness imports.%Copy.impl_witness_table.8d2, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %facet_value.53d: %facet_type.f48 = facet_value %i32, (%custom_witness.df9cc1.3, %Copy.impl_witness.b51) [concrete]
// CHECK:STDOUT:   %Iterate.lookup_impl_witness.1d8: <witness> = lookup_impl_witness %.Self.500, @Iterate [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.294: %facet_type.f48 = impl_witness_access %Iterate.lookup_impl_witness.1d8, element0 [symbolic_self]
// CHECK:STDOUT:   %Iterate_where.type.01c: type = facet_type <@Iterate where %impl.elem0.294 = %facet_value.53d> [concrete]
// CHECK:STDOUT:   %pattern_type.024: type = pattern_type %Iterate_where.type.01c [concrete]
// CHECK:STDOUT:   %R.patt: %pattern_type.024 = symbolic_binding_pattern R, 0 [symbolic]
// CHECK:STDOUT:   %R: %Iterate_where.type.01c = symbolic_binding R, 0 [symbolic]
// CHECK:STDOUT:   %R.as_type: type = facet_access_type %R [symbolic]
// CHECK:STDOUT:   %pattern_type.073: type = pattern_type %R.as_type [symbolic]
// CHECK:STDOUT:   %r.param_patt.d39: %pattern_type.073 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %r.patt.5af: %pattern_type.073 = wrapper_binding_pattern r, %r.param_patt.d39 [symbolic]
// CHECK:STDOUT:   %facet_type.037: type = facet_type <@IntFitsIn, @IntFitsIn(%Int.67af24.2) & @AnyInt> [symbolic]
// CHECK:STDOUT:   %From.6a4: %facet_type.037 = symbolic_binding From, 1 [symbolic]
// CHECK:STDOUT:   %From.as_type.as.ImplicitAs.impl.Convert.type.7fa: type = fn_type @From.as_type.as.ImplicitAs.impl.Convert, @From.as_type.as.ImplicitAs.impl(%To, %From.6a4) [symbolic]
// CHECK:STDOUT:   %From.as_type.as.ImplicitAs.impl.Convert.643: %From.as_type.as.ImplicitAs.impl.Convert.type.7fa = struct_value () [symbolic]
// CHECK:STDOUT:   %i.patt: %pattern_type.6b6 = value_binding_pattern i [concrete]
// CHECK:STDOUT:   %Iterate.lookup_impl_witness.403: <witness> = lookup_impl_witness %R, @Iterate [symbolic]
// CHECK:STDOUT:   %Iterate.facet.a7e: %Iterate.type = facet_value %R.as_type, (%Iterate.lookup_impl_witness.403) [symbolic]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.d86: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.a7e) [symbolic]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.7ad: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.a7e) [symbolic]
// CHECK:STDOUT:   %.7f7: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.d86, %Iterate.facet.a7e [symbolic]
// CHECK:STDOUT:   %impl.elem2.680: %.7f7 = impl_witness_access %Iterate.lookup_impl_witness.403, element2 [symbolic]
// CHECK:STDOUT:   %impl.elem1.595: %Destroy.type = impl_witness_access %Iterate.lookup_impl_witness.403, element1 [symbolic]
// CHECK:STDOUT:   %as_type.038: type = facet_access_type %impl.elem1.595 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.740: <specific function> = specific_impl_function %impl.elem2.680, @Iterate.WithSelf.NewCursor(%Iterate.facet.a7e) [symbolic]
// CHECK:STDOUT:   %ptr.d7f: type = ptr_type %as_type.038 [symbolic]
// CHECK:STDOUT:   %.339: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.7ad, %Iterate.facet.a7e [symbolic]
// CHECK:STDOUT:   %impl.elem3.ffb: %.339 = impl_witness_access %Iterate.lookup_impl_witness.403, element3 [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.bfa: type = class_type @DefaultOptionalStorage, @DefaultOptionalStorage(%facet_value.53d) [concrete]
// CHECK:STDOUT:   %OptionalStorage.impl_witness.c8c: <witness> = impl_witness imports.%OptionalStorage.impl_witness_table.460, @T.as_type.as.OptionalStorage.impl(%facet_value.53d) [concrete]
// CHECK:STDOUT:   %OptionalStorage.facet.097: %OptionalStorage.type = facet_value %i32, (%OptionalStorage.impl_witness.c8c) [concrete]
// CHECK:STDOUT:   %Optional.755: type = class_type @Optional, @Optional(%OptionalStorage.facet.097) [concrete]
// CHECK:STDOUT:   %specific_impl_fn.c09: <specific function> = specific_impl_function %impl.elem3.ffb, @Iterate.WithSelf.Next(%Iterate.facet.a7e) [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.type.af3: type = fn_type @Optional.HasValue, @Optional(%OptionalStorage.facet.097) [concrete]
// CHECK:STDOUT:   %Optional.HasValue.4f6: %Optional.HasValue.type.af3 = struct_value () [concrete]
// CHECK:STDOUT:   %Optional.Get.type.336: type = fn_type @Optional.Get, @Optional(%OptionalStorage.facet.097) [concrete]
// CHECK:STDOUT:   %Optional.Get.0a8: %Optional.Get.type.336 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.facet.d07318.2: %Destroy.type = facet_value %i32, (%custom_witness.df9cc1.3) [concrete]
// CHECK:STDOUT:   %MaybeUnformed.023: type = class_type @MaybeUnformed, @MaybeUnformed(%Destroy.facet.d07318.2) [concrete]
// CHECK:STDOUT:   %.1d6a7: type = maybe_unformed_type %i32 [concrete]
// CHECK:STDOUT:   %struct_type.value.has_value.ad0: type = struct_type {.value: %MaybeUnformed.023, .has_value: bool} [concrete]
// CHECK:STDOUT:   %Optional.HasValue.specific_fn.b8c: <specific function> = specific_function %Optional.HasValue.4f6, @Optional.HasValue(%OptionalStorage.facet.097) [concrete]
// CHECK:STDOUT:   %Optional.Get.specific_fn.fc5: <specific function> = specific_function %Optional.Get.0a8, @Optional.Get(%OptionalStorage.facet.097) [concrete]
// CHECK:STDOUT:   %AddAssignWith.type.941: type = facet_type <@AddAssignWith, @AddAssignWith(%i32)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.f7de31.2: type = facet_type <@ImplicitAs, @ImplicitAs(%Int.67af24.1)> [symbolic]
// CHECK:STDOUT:   %U.4e6: %ImplicitAs.type.f7de31.2 = symbolic_binding U, 1 [symbolic]
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.type.a58fbe.1: type = fn_type @Int.as.AddAssignWith.impl.Op.1, @Int.as.AddAssignWith.impl(%N.fe9, %U.4e6) [symbolic]
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.45b360.1: %Int.as.AddAssignWith.impl.Op.type.a58fbe.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.type.a58fbe.2: type = fn_type @Int.as.AddAssignWith.impl.Op.2, @Int.as.AddAssignWith.impl(%N.fe9, %U.4e6) [symbolic]
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.45b360.2: %Int.as.AddAssignWith.impl.Op.type.a58fbe.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %facet_type.138: type = facet_type <@IntFitsIn, @IntFitsIn(%i32) & @AnyInt> [concrete]
// CHECK:STDOUT:   %custom_witness.36a: <witness> = custom_witness (), @IntFitsIn, @IntFitsIn(%i32) [concrete]
// CHECK:STDOUT:   %Int.as.AnyInt.impl.AsInt.type.f15: type = fn_type @Int.as.AnyInt.impl.AsInt, @Int.as.AnyInt.impl(%N.fe9) [symbolic]
// CHECK:STDOUT:   %Int.as.AnyInt.impl.AsInt.d10: %Int.as.AnyInt.impl.AsInt.type.f15 = struct_value () [symbolic]
// CHECK:STDOUT:   %AnyInt.impl_witness.48c: <witness> = impl_witness imports.%AnyInt.impl_witness_table, @Int.as.AnyInt.impl(%int_32) [concrete]
// CHECK:STDOUT:   %facet_value.cd1: %facet_type.138 = facet_value %i32, (%custom_witness.36a, %AnyInt.impl_witness.48c) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.1d8: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.992, @From.as_type.as.ImplicitAs.impl(%int_32, %facet_value.cd1) [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.e88: %ImplicitAs.type.914 = facet_value %i32, (%ImplicitAs.impl_witness.1d8) [concrete]
// CHECK:STDOUT:   %AddAssignWith.impl_witness.e9d: <witness> = impl_witness imports.%AddAssignWith.impl_witness_table, @Int.as.AddAssignWith.impl(%int_32, %ImplicitAs.facet.e88) [concrete]
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.type.0b80aa.1: type = fn_type @Int.as.AddAssignWith.impl.Op.2, @Int.as.AddAssignWith.impl(%int_32, %ImplicitAs.facet.e88) [concrete]
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.d34738.1: %Int.as.AddAssignWith.impl.Op.type.0b80aa.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.type.0b80aa.2: type = fn_type @Int.as.AddAssignWith.impl.Op.1, @Int.as.AddAssignWith.impl(%int_32, %ImplicitAs.facet.e88) [concrete]
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.d34738.2: %Int.as.AddAssignWith.impl.Op.type.0b80aa.2 = struct_value () [concrete]
// CHECK:STDOUT:   %AddAssignWith.facet.e5e: %AddAssignWith.type.941 = facet_value %i32, (%AddAssignWith.impl_witness.e9d) [concrete]
// CHECK:STDOUT:   %AddAssignWith.WithSelf.Op.type.640: type = fn_type @AddAssignWith.WithSelf.Op, @AddAssignWith.WithSelf(%i32, %AddAssignWith.facet.e5e) [concrete]
// CHECK:STDOUT:   %.004: type = fn_type_with_self_type %AddAssignWith.WithSelf.Op.type.640, %AddAssignWith.facet.e5e [concrete]
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.specific_fn.d76216.1: <specific function> = specific_function %Int.as.AddAssignWith.impl.Op.d34738.2, @Int.as.AddAssignWith.impl.Op.1(%int_32, %ImplicitAs.facet.e88) [concrete]
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.specific_fn.d76216.2: <specific function> = specific_function %Int.as.AddAssignWith.impl.Op.d34738.1, @Int.as.AddAssignWith.impl.Op.2(%int_32, %ImplicitAs.facet.e88) [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.9: type = fn_type @Destroy.Op.loc34_19.6 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.9: %Destroy.Op.type.1d8f74.9 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.3c4: <witness> = lookup_impl_witness %impl.elem1.595, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.76f: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%impl.elem1.595) [symbolic]
// CHECK:STDOUT:   %.2f2: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.76f, %impl.elem1.595 [symbolic]
// CHECK:STDOUT:   %impl.elem0.528: %.2f2 = impl_witness_access %Destroy.lookup_impl_witness.3c4, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.dfd: <specific function> = specific_impl_function %impl.elem0.528, @Destroy.WithSelf.Op(%impl.elem1.595) [symbolic]
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
// CHECK:STDOUT:   %custom_witness.444: <witness> = custom_witness (%Iterator.Op.a2a866.3), @Inc [concrete]
// CHECK:STDOUT:   %Equal.type.ca8475.1: type = fn_type @Equal.1 [concrete]
// CHECK:STDOUT:   %Equal.285b15.1: %Equal.type.ca8475.1 = struct_value () [concrete]
// CHECK:STDOUT:   %NotEqual.type.53d9ee.1: type = fn_type @NotEqual.1 [concrete]
// CHECK:STDOUT:   %NotEqual.119263.1: %NotEqual.type.53d9ee.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.981: <witness> = custom_witness (%Equal.285b15.1, %NotEqual.119263.1), @EqWith, @EqWith(%Iterator.3ee) [concrete]
// CHECK:STDOUT:   %facet_value.c6e: %CppRange_where.type.d682d6.1 = facet_value %MutableRange, (%custom_witness.df9cc1.3, %Copy.impl_witness.b51, %custom_witness.ed3, %custom_witness.955, %custom_witness.ef8, %custom_witness.444, %custom_witness.981) [concrete]
// CHECK:STDOUT:   %tuple.type.508: type = tuple_type (%Iterator.3ee, %Iterator.3ee) [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.10: type = fn_type @Destroy.Op.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.10: %Destroy.Op.type.1d8f74.10 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.df9cc1.10: <witness> = custom_witness (%Destroy.Op.1a2547.10), @Destroy [concrete]
// CHECK:STDOUT:   %Destroy.facet.30f: %Destroy.type = facet_value %tuple.type.508, (%custom_witness.df9cc1.10) [concrete]
// CHECK:STDOUT:   %Iterate.impl_witness.0b0: <witness> = impl_witness imports.%Iterate.impl_witness_table.121, @T.as_type.as.Iterate.impl(%facet_value.c6e) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.type.540: type = fn_type @T.as_type.as.Iterate.impl.NewCursor, @T.as_type.as.Iterate.impl(%facet_value.c6e) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.45a: %T.as_type.as.Iterate.impl.NewCursor.type.540 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.type.9d1: type = fn_type @T.as_type.as.Iterate.impl.Next, @T.as_type.as.Iterate.impl(%facet_value.c6e) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.a5b: %T.as_type.as.Iterate.impl.Next.type.9d1 = struct_value () [concrete]
// CHECK:STDOUT:   %Iterate.facet.fdf: %Iterate.type = facet_value %MutableRange, (%Iterate.impl_witness.0b0) [concrete]
// CHECK:STDOUT:   %facet_value.8a3: %Iterate_where.type.01c = facet_value %MutableRange, (%Iterate.impl_witness.0b0) [concrete]
// CHECK:STDOUT:   %r.param_patt.c79: %pattern_type.e635 = value_param_pattern [concrete]
// CHECK:STDOUT:   %r.patt.94f: %pattern_type.e635 = wrapper_binding_pattern r, %r.param_patt.c79 [concrete]
// CHECK:STDOUT:   %Iterator.260: type = class_type @Iterator.2 [concrete]
// CHECK:STDOUT:   %ConstRange.Begin.type: type = fn_type @ConstRange.Begin [concrete]
// CHECK:STDOUT:   %ConstRange.Begin: %ConstRange.Begin.type = struct_value () [concrete]
// CHECK:STDOUT:   %ConstRange.End.type: type = fn_type @ConstRange.End [concrete]
// CHECK:STDOUT:   %ConstRange.End: %ConstRange.End.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.23e: <witness> = custom_witness (%Iterator.260, %Iterator.260, %ConstRange.Begin, %ConstRange.End), @CppRangeForIterate [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.0f34b8.1: type = fn_type @Iterator.Op.4 [concrete]
// CHECK:STDOUT:   %Iterator.Op.1c3e4a.1: %Iterator.Op.type.0f34b8.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.adb: <witness> = custom_witness (%i32, %Iterator.Op.1c3e4a.1), @CppUnsafeDeref [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.0f34b8.2: type = fn_type @Iterator.Op.5 [concrete]
// CHECK:STDOUT:   %Iterator.Op.1c3e4a.2: %Iterator.Op.type.0f34b8.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.6e8: <witness> = custom_witness (%Iterator.Op.1c3e4a.2), @Destroy [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.0f34b8.3: type = fn_type @Iterator.Op.6 [concrete]
// CHECK:STDOUT:   %Iterator.Op.1c3e4a.3: %Iterator.Op.type.0f34b8.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.746: <witness> = custom_witness (%Iterator.Op.1c3e4a.3), @Inc [concrete]
// CHECK:STDOUT:   %Equal.type.ca8475.2: type = fn_type @Equal.2 [concrete]
// CHECK:STDOUT:   %Equal.285b15.2: %Equal.type.ca8475.2 = struct_value () [concrete]
// CHECK:STDOUT:   %NotEqual.type.53d9ee.2: type = fn_type @NotEqual.2 [concrete]
// CHECK:STDOUT:   %NotEqual.119263.2: %NotEqual.type.53d9ee.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.941: <witness> = custom_witness (%Equal.285b15.2, %NotEqual.119263.2), @EqWith, @EqWith(%Iterator.260) [concrete]
// CHECK:STDOUT:   %facet_value.96f: %CppRange_where.type.d682d6.1 = facet_value %ConstRange, (%custom_witness.df9cc1.3, %Copy.impl_witness.b51, %custom_witness.23e, %custom_witness.6e8, %custom_witness.adb, %custom_witness.746, %custom_witness.941) [concrete]
// CHECK:STDOUT:   %tuple.type.69f: type = tuple_type (%Iterator.260, %Iterator.260) [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.11: type = fn_type @Destroy.Op.3 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.11: %Destroy.Op.type.1d8f74.11 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.df9cc1.11: <witness> = custom_witness (%Destroy.Op.1a2547.11), @Destroy [concrete]
// CHECK:STDOUT:   %Destroy.facet.e4c: %Destroy.type = facet_value %tuple.type.69f, (%custom_witness.df9cc1.11) [concrete]
// CHECK:STDOUT:   %Iterate.impl_witness.bb2: <witness> = impl_witness imports.%Iterate.impl_witness_table.121, @T.as_type.as.Iterate.impl(%facet_value.96f) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.type.356: type = fn_type @T.as_type.as.Iterate.impl.NewCursor, @T.as_type.as.Iterate.impl(%facet_value.96f) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.ab8: %T.as_type.as.Iterate.impl.NewCursor.type.356 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.type.64d: type = fn_type @T.as_type.as.Iterate.impl.Next, @T.as_type.as.Iterate.impl(%facet_value.96f) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.a02: %T.as_type.as.Iterate.impl.Next.type.64d = struct_value () [concrete]
// CHECK:STDOUT:   %Iterate.facet.4c8: %Iterate.type = facet_value %ConstRange, (%Iterate.impl_witness.bb2) [concrete]
// CHECK:STDOUT:   %facet_value.9af: %Iterate_where.type.01c = facet_value %ConstRange, (%Iterate.impl_witness.bb2) [concrete]
// CHECK:STDOUT:   %r.param_patt.83e: %pattern_type.0ba = value_param_pattern [concrete]
// CHECK:STDOUT:   %r.patt.c21: %pattern_type.0ba = wrapper_binding_pattern r, %r.param_patt.83e [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.5e7: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.fdf) [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.08a: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.fdf) [concrete]
// CHECK:STDOUT:   %.641: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.5e7, %Iterate.facet.fdf [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.specific_fn.b38: <specific function> = specific_function %T.as_type.as.Iterate.impl.NewCursor.45a, @T.as_type.as.Iterate.impl.NewCursor(%facet_value.c6e) [concrete]
// CHECK:STDOUT:   %ptr.45c: type = ptr_type %tuple.type.508 [concrete]
// CHECK:STDOUT:   %.d1b: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.08a, %Iterate.facet.fdf [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.specific_fn.b5a: <specific function> = specific_function %T.as_type.as.Iterate.impl.Next.a5b, @T.as_type.as.Iterate.impl.Next(%facet_value.c6e) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.c6c: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.30f) [concrete]
// CHECK:STDOUT:   %.0a4: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.c6c, %Destroy.facet.30f [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.9d7: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.4c8) [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.f73: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.4c8) [concrete]
// CHECK:STDOUT:   %.c93: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.9d7, %Iterate.facet.4c8 [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.specific_fn.31a: <specific function> = specific_function %T.as_type.as.Iterate.impl.NewCursor.ab8, @T.as_type.as.Iterate.impl.NewCursor(%facet_value.96f) [concrete]
// CHECK:STDOUT:   %ptr.c5c: type = ptr_type %tuple.type.69f [concrete]
// CHECK:STDOUT:   %.fe6: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.f73, %Iterate.facet.4c8 [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.specific_fn.480: <specific function> = specific_function %T.as_type.as.Iterate.impl.Next.a02, @T.as_type.as.Iterate.impl.Next(%facet_value.96f) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.7ec: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.e4c) [concrete]
// CHECK:STDOUT:   %.0e7: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.7ec, %Destroy.facet.e4c [concrete]
// CHECK:STDOUT:   %complete_type.c28: <witness> = complete_type_witness %tuple.type.508 [concrete]
// CHECK:STDOUT:   %complete_type.8d5: <witness> = complete_type_witness %tuple.type.69f [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.bb6: @Optional.%Optional.HasValue.type (%Optional.HasValue.type.e9e) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.HasValue (constants.%Optional.HasValue.6cd)]
// CHECK:STDOUT:   %Core.import_ref.dab: @Optional.%Optional.Get.type (%Optional.Get.type.bb9) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.Get (constants.%Optional.Get.fa1)]
// CHECK:STDOUT:   %Core.import_ref.0c6: type = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%DefaultOptionalStorage (constants.%DefaultOptionalStorage.d50)]
// CHECK:STDOUT:   %Core.import_ref.dc3: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.None.type (%T.as_type.as.OptionalStorage.impl.None.type.0a1) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.None (constants.%T.as_type.as.OptionalStorage.impl.None.e06)]
// CHECK:STDOUT:   %Core.import_ref.c53: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Some.type (%T.as_type.as.OptionalStorage.impl.Some.type.492) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Some (constants.%T.as_type.as.OptionalStorage.impl.Some.89a)]
// CHECK:STDOUT:   %Core.import_ref.ed4: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Has.type (%T.as_type.as.OptionalStorage.impl.Has.type.e5a) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Has (constants.%T.as_type.as.OptionalStorage.impl.Has.4ee)]
// CHECK:STDOUT:   %Core.import_ref.e9c: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Get.type (%T.as_type.as.OptionalStorage.impl.Get.type.a12) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Get (constants.%T.as_type.as.OptionalStorage.impl.Get.d32)]
// CHECK:STDOUT:   %Core.import_ref.0b2 = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], unloaded
// CHECK:STDOUT:   %OptionalStorage.impl_witness_table.460 = impl_witness_table (%Core.import_ref.0c6, %Core.import_ref.dc3, %Core.import_ref.c53, %Core.import_ref.ed4, %Core.import_ref.e9c, %Core.import_ref.0b2), @T.as_type.as.OptionalStorage.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.eef: %facet_type.f48 = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.Iterate.impl.%facet_value (constants.%facet_value.f93)]
// CHECK:STDOUT:   %Core.import_ref.fcc: %Destroy.type = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.Iterate.impl.%Destroy.facet (constants.%Destroy.facet.9cd)]
// CHECK:STDOUT:   %Core.import_ref.fd1: @T.as_type.as.Iterate.impl.%T.as_type.as.Iterate.impl.NewCursor.type (%T.as_type.as.Iterate.impl.NewCursor.type.953) = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.Iterate.impl.%T.as_type.as.Iterate.impl.NewCursor (constants.%T.as_type.as.Iterate.impl.NewCursor.544)]
// CHECK:STDOUT:   %Core.import_ref.273: @T.as_type.as.Iterate.impl.%T.as_type.as.Iterate.impl.Next.type (%T.as_type.as.Iterate.impl.Next.type.b36) = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.Iterate.impl.%T.as_type.as.Iterate.impl.Next (constants.%T.as_type.as.Iterate.impl.Next.d6b)]
// CHECK:STDOUT:   %Iterate.impl_witness_table.121 = impl_witness_table (%Core.import_ref.eef, %Core.import_ref.fcc, %Core.import_ref.fd1, %Core.import_ref.273), @T.as_type.as.Iterate.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.cd6: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.ac8) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.5e0)]
// CHECK:STDOUT:   %Copy.impl_witness_table.8d2 = impl_witness_table (%Core.import_ref.cd6), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.2a4: @From.as_type.as.ImplicitAs.impl.%From.as_type.as.ImplicitAs.impl.Convert.type (%From.as_type.as.ImplicitAs.impl.Convert.type.7fa) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @From.as_type.as.ImplicitAs.impl.%From.as_type.as.ImplicitAs.impl.Convert (constants.%From.as_type.as.ImplicitAs.impl.Convert.643)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.992 = impl_witness_table (%Core.import_ref.2a4), @From.as_type.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.098: @Int.as.AddAssignWith.impl.%Int.as.AddAssignWith.impl.Op.type.2 (%Int.as.AddAssignWith.impl.Op.type.a58fbe.1) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.AddAssignWith.impl.%Int.as.AddAssignWith.impl.Op.2 (constants.%Int.as.AddAssignWith.impl.Op.45b360.1)]
// CHECK:STDOUT:   %AddAssignWith.impl_witness_table = impl_witness_table (%Core.import_ref.098), @Int.as.AddAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.1b1: @Int.as.AddAssignWith.impl.%Int.as.AddAssignWith.impl.Op.type.1 (%Int.as.AddAssignWith.impl.Op.type.a58fbe.2) = import_ref Core//prelude/types/int, Op, loaded [symbolic = @Int.as.AddAssignWith.impl.%Int.as.AddAssignWith.impl.Op.1 (constants.%Int.as.AddAssignWith.impl.Op.45b360.2)]
// CHECK:STDOUT:   %Core.import_ref.47b: Core.IntLiteral = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.AnyInt.impl.%N (constants.%N.fe9)]
// CHECK:STDOUT:   %Core.import_ref.49b: @Int.as.AnyInt.impl.%Int.as.AnyInt.impl.AsInt.type (%Int.as.AnyInt.impl.AsInt.type.f15) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.AnyInt.impl.%Int.as.AnyInt.impl.AsInt (constants.%Int.as.AnyInt.impl.AsInt.d10)]
// CHECK:STDOUT:   %AnyInt.impl_witness_table = impl_witness_table (%Core.import_ref.47b, %Core.import_ref.49b), @Int.as.AnyInt.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @TestIterate(%R.loc30_17.2: %Iterate_where.type.01c) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Iterate.lookup_impl_witness: <witness> = lookup_impl_witness %R.loc30_17.1, @Iterate [symbolic = %Iterate.lookup_impl_witness (constants.%Iterate.lookup_impl_witness.403)]
// CHECK:STDOUT:   %Iterate.facet.loc34_19.5: %Iterate.type = facet_value %R.as_type.loc30_61.1, (%Iterate.lookup_impl_witness) [symbolic = %Iterate.facet.loc34_19.5 (constants.%Iterate.facet.a7e)]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.loc34_19.5) [symbolic = %Iterate.WithSelf.NewCursor.type (constants.%Iterate.WithSelf.NewCursor.type.d86)]
// CHECK:STDOUT:   %.loc34_19.19: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type, %Iterate.facet.loc34_19.5 [symbolic = %.loc34_19.19 (constants.%.7f7)]
// CHECK:STDOUT:   %impl.elem2.loc34_19.2: @TestIterate.%.loc34_19.19 (%.7f7) = impl_witness_access %Iterate.lookup_impl_witness, element2 [symbolic = %impl.elem2.loc34_19.2 (constants.%impl.elem2.680)]
// CHECK:STDOUT:   %specific_impl_fn.loc34_19.4: <specific function> = specific_impl_function %impl.elem2.loc34_19.2, @Iterate.WithSelf.NewCursor(%Iterate.facet.loc34_19.5) [symbolic = %specific_impl_fn.loc34_19.4 (constants.%specific_impl_fn.740)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %ptr: type = ptr_type %as_type [symbolic = %ptr (constants.%ptr.d7f)]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.loc34_19.5) [symbolic = %Iterate.WithSelf.Next.type (constants.%Iterate.WithSelf.Next.type.7ad)]
// CHECK:STDOUT:   %.loc34_19.20: type = fn_type_with_self_type %Iterate.WithSelf.Next.type, %Iterate.facet.loc34_19.5 [symbolic = %.loc34_19.20 (constants.%.339)]
// CHECK:STDOUT:   %impl.elem3.loc34_19.2: @TestIterate.%.loc34_19.20 (%.339) = impl_witness_access %Iterate.lookup_impl_witness, element3 [symbolic = %impl.elem3.loc34_19.2 (constants.%impl.elem3.ffb)]
// CHECK:STDOUT:   %specific_impl_fn.loc34_19.5: <specific function> = specific_impl_function %impl.elem3.loc34_19.2, @Iterate.WithSelf.Next(%Iterate.facet.loc34_19.5) [symbolic = %specific_impl_fn.loc34_19.5 (constants.%specific_impl_fn.c09)]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%impl.elem1) [symbolic = %Destroy.WithSelf.Op.type (constants.%Destroy.WithSelf.Op.type.76f)]
// CHECK:STDOUT:   %.loc34_19.21: type = fn_type_with_self_type %Destroy.WithSelf.Op.type, %impl.elem1 [symbolic = %.loc34_19.21 (constants.%.2f2)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %impl.elem1, @Destroy [symbolic = %Destroy.lookup_impl_witness (constants.%Destroy.lookup_impl_witness.3c4)]
// CHECK:STDOUT:   %impl.elem0.loc34_19.2: @TestIterate.%.loc34_19.21 (%.2f2) = impl_witness_access %Destroy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc34_19.2 (constants.%impl.elem0.528)]
// CHECK:STDOUT:   %specific_impl_fn.loc34_19.6: <specific function> = specific_impl_function %impl.elem0.loc34_19.2, @Destroy.WithSelf.Op(%impl.elem1) [symbolic = %specific_impl_fn.loc34_19.6 (constants.%specific_impl_fn.dfd)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%r.param: @TestIterate.%R.as_type.loc30_61.1 (%R.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %i.patt: %pattern_type.6b6 = value_binding_pattern i [concrete = constants.%i.patt]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %r.ref: @TestIterate.%R.as_type.loc30_61.1 (%R.as_type) = name_ref r, %r
// CHECK:STDOUT:     %impl.elem2.loc34_19.1: @TestIterate.%.loc34_19.19 (%.7f7) = impl_witness_access constants.%Iterate.lookup_impl_witness.403, element2 [symbolic = %impl.elem2.loc34_19.2 (constants.%impl.elem2.680)]
// CHECK:STDOUT:     %bound_method.loc34_19.1: <bound method> = bound_method %r.ref, %impl.elem2.loc34_19.1
// CHECK:STDOUT:     %Iterate.facet.loc34_19.1: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.403) [symbolic = %Iterate.facet.loc34_19.5 (constants.%Iterate.facet.a7e)]
// CHECK:STDOUT:     %.loc34_19.1: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc34_19.1 [symbolic = %Iterate.facet.loc34_19.5 (constants.%Iterate.facet.a7e)]
// CHECK:STDOUT:     %Iterate.facet.loc34_19.2: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.403) [symbolic = %Iterate.facet.loc34_19.5 (constants.%Iterate.facet.a7e)]
// CHECK:STDOUT:     %.loc34_19.2: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc34_19.2 [symbolic = %Iterate.facet.loc34_19.5 (constants.%Iterate.facet.a7e)]
// CHECK:STDOUT:     %specific_impl_fn.loc34_19.1: <specific function> = specific_impl_function %impl.elem2.loc34_19.1, @Iterate.WithSelf.NewCursor(constants.%Iterate.facet.a7e) [symbolic = %specific_impl_fn.loc34_19.4 (constants.%specific_impl_fn.740)]
// CHECK:STDOUT:     %bound_method.loc34_19.2: <bound method> = bound_method %r.ref, %specific_impl_fn.loc34_19.1
// CHECK:STDOUT:     %var: ref @TestIterate.%as_type (%as_type.038) = var_storage invalid
// CHECK:STDOUT:     %Iterate.WithSelf.NewCursor.call: init @TestIterate.%as_type (%as_type.038) to %var = call %bound_method.loc34_19.2(%r.ref)
// CHECK:STDOUT:     assign %var, %Iterate.WithSelf.NewCursor.call
// CHECK:STDOUT:     br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.next:
// CHECK:STDOUT:     %addr: @TestIterate.%ptr (%ptr.d7f) = addr_of %var
// CHECK:STDOUT:     %impl.elem3.loc34_19.1: @TestIterate.%.loc34_19.20 (%.339) = impl_witness_access constants.%Iterate.lookup_impl_witness.403, element3 [symbolic = %impl.elem3.loc34_19.2 (constants.%impl.elem3.ffb)]
// CHECK:STDOUT:     %bound_method.loc34_19.3: <bound method> = bound_method %r.ref, %impl.elem3.loc34_19.1
// CHECK:STDOUT:     %Iterate.facet.loc34_19.3: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.403) [symbolic = %Iterate.facet.loc34_19.5 (constants.%Iterate.facet.a7e)]
// CHECK:STDOUT:     %.loc34_19.3: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc34_19.3 [symbolic = %Iterate.facet.loc34_19.5 (constants.%Iterate.facet.a7e)]
// CHECK:STDOUT:     %Iterate.facet.loc34_19.4: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.403) [symbolic = %Iterate.facet.loc34_19.5 (constants.%Iterate.facet.a7e)]
// CHECK:STDOUT:     %.loc34_19.4: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc34_19.4 [symbolic = %Iterate.facet.loc34_19.5 (constants.%Iterate.facet.a7e)]
// CHECK:STDOUT:     %.loc34_19.5: %Destroy.type = converted constants.%as_type.038, constants.%impl.elem1.595 [symbolic = %impl.elem1 (constants.%impl.elem1.595)]
// CHECK:STDOUT:     %.loc34_19.6: %Destroy.type = converted constants.%as_type.038, constants.%impl.elem1.595 [symbolic = %impl.elem1 (constants.%impl.elem1.595)]
// CHECK:STDOUT:     %specific_impl_fn.loc34_19.2: <specific function> = specific_impl_function %impl.elem3.loc34_19.1, @Iterate.WithSelf.Next(constants.%Iterate.facet.a7e) [symbolic = %specific_impl_fn.loc34_19.5 (constants.%specific_impl_fn.c09)]
// CHECK:STDOUT:     %bound_method.loc34_19.4: <bound method> = bound_method %r.ref, %specific_impl_fn.loc34_19.2
// CHECK:STDOUT:     %.loc34_19.7: ref %Optional.755 = temporary_storage
// CHECK:STDOUT:     %Iterate.WithSelf.Next.call: init %Optional.755 to %.loc34_19.7 = call %bound_method.loc34_19.4(%r.ref, %addr)
// CHECK:STDOUT:     %.loc34_19.8: ref %Optional.755 = temporary %.loc34_19.7, %Iterate.WithSelf.Next.call
// CHECK:STDOUT:     %.loc34_19.9: %Optional.HasValue.type.af3 = specific_constant imports.%Core.import_ref.bb6, @Optional(constants.%OptionalStorage.facet.097) [concrete = constants.%Optional.HasValue.4f6]
// CHECK:STDOUT:     %HasValue.ref: %Optional.HasValue.type.af3 = name_ref HasValue, %.loc34_19.9 [concrete = constants.%Optional.HasValue.4f6]
// CHECK:STDOUT:     %Optional.HasValue.bound: <bound method> = bound_method %.loc34_19.8, %HasValue.ref
// CHECK:STDOUT:     %Optional.HasValue.specific_fn: <specific function> = specific_function %HasValue.ref, @Optional.HasValue(constants.%OptionalStorage.facet.097) [concrete = constants.%Optional.HasValue.specific_fn.b8c]
// CHECK:STDOUT:     %bound_method.loc34_19.5: <bound method> = bound_method %.loc34_19.8, %Optional.HasValue.specific_fn
// CHECK:STDOUT:     %.loc34_19.10: %Optional.755 = acquire_value %.loc34_19.8
// CHECK:STDOUT:     %Optional.HasValue.call: init bool = call %bound_method.loc34_19.5(%.loc34_19.10)
// CHECK:STDOUT:     %.loc34_19.11: bool = value_of_initializer %Optional.HasValue.call
// CHECK:STDOUT:     %.loc34_19.12: bool = converted %Optional.HasValue.call, %.loc34_19.11
// CHECK:STDOUT:     if %.loc34_19.12 br !for.body else br !for.done
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.body:
// CHECK:STDOUT:     %.loc34_19.13: %Optional.Get.type.336 = specific_constant imports.%Core.import_ref.dab, @Optional(constants.%OptionalStorage.facet.097) [concrete = constants.%Optional.Get.0a8]
// CHECK:STDOUT:     %Get.ref: %Optional.Get.type.336 = name_ref Get, %.loc34_19.13 [concrete = constants.%Optional.Get.0a8]
// CHECK:STDOUT:     %Optional.Get.bound: <bound method> = bound_method %.loc34_19.8, %Get.ref
// CHECK:STDOUT:     %Optional.Get.specific_fn: <specific function> = specific_function %Get.ref, @Optional.Get(constants.%OptionalStorage.facet.097) [concrete = constants.%Optional.Get.specific_fn.fc5]
// CHECK:STDOUT:     %bound_method.loc34_19.6: <bound method> = bound_method %.loc34_19.8, %Optional.Get.specific_fn
// CHECK:STDOUT:     %.loc34_19.14: %Optional.755 = acquire_value %.loc34_19.8
// CHECK:STDOUT:     %Optional.Get.call: init %i32 = call %bound_method.loc34_19.6(%.loc34_19.14)
// CHECK:STDOUT:     %.loc34_19.15: %i32 = value_of_initializer %Optional.Get.call
// CHECK:STDOUT:     %.loc34_19.16: %i32 = converted %Optional.Get.call, %.loc34_19.15
// CHECK:STDOUT:     %i32.loc34: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %i: %i32 = wrapper_binding i, %.loc34_19.16
// CHECK:STDOUT:     %sum.ref: ref %i32 = name_ref sum, %sum
// CHECK:STDOUT:     %i.ref: %i32 = name_ref i, %i
// CHECK:STDOUT:     %impl.elem0.loc35: %.004 = impl_witness_access constants.%AddAssignWith.impl_witness.e9d, element0 [concrete = constants.%Int.as.AddAssignWith.impl.Op.d34738.2]
// CHECK:STDOUT:     %bound_method.loc35_9.1: <bound method> = bound_method %sum.ref, %impl.elem0.loc35
// CHECK:STDOUT:     %specific_fn.loc35: <specific function> = specific_function %impl.elem0.loc35, @Int.as.AddAssignWith.impl.Op.1(constants.%int_32, constants.%ImplicitAs.facet.e88) [concrete = constants.%Int.as.AddAssignWith.impl.Op.specific_fn.d76216.1]
// CHECK:STDOUT:     %bound_method.loc35_9.2: <bound method> = bound_method %sum.ref, %specific_fn.loc35
// CHECK:STDOUT:     %.loc35: %Int.as.AddAssignWith.impl.Op.type.0b80aa.1 = specific_constant imports.%Core.Op.1b1, @Int.as.AddAssignWith.impl(constants.%int_32, constants.%ImplicitAs.facet.e88) [concrete = constants.%Int.as.AddAssignWith.impl.Op.d34738.1]
// CHECK:STDOUT:     %Op.ref: %Int.as.AddAssignWith.impl.Op.type.0b80aa.1 = name_ref Op, %.loc35 [concrete = constants.%Int.as.AddAssignWith.impl.Op.d34738.1]
// CHECK:STDOUT:     %Int.as.AddAssignWith.impl.Op.bound: <bound method> = bound_method %sum.ref, %Op.ref
// CHECK:STDOUT:     %Int.as.AddAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref, @Int.as.AddAssignWith.impl.Op.2(constants.%int_32, constants.%ImplicitAs.facet.e88) [concrete = constants.%Int.as.AddAssignWith.impl.Op.specific_fn.d76216.2]
// CHECK:STDOUT:     %bound_method.loc35_9.3: <bound method> = bound_method %sum.ref, %Int.as.AddAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:     %Int.as.AddAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc35_9.3(%sum.ref, %i.ref)
// CHECK:STDOUT:     %Destroy.Op.bound.loc34_19.1: <bound method> = bound_method %.loc34_19.8, constants.%Destroy.Op.1a2547.9
// CHECK:STDOUT:     %Destroy.Op.call.loc34_19.1: init %empty_tuple.type = call %Destroy.Op.bound.loc34_19.1(%.loc34_19.8)
// CHECK:STDOUT:     br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.done:
// CHECK:STDOUT:     %Destroy.Op.bound.loc34_19.2: <bound method> = bound_method %.loc34_19.8, constants.%Destroy.Op.1a2547.9
// CHECK:STDOUT:     %Destroy.Op.call.loc34_19.2: init %empty_tuple.type = call %Destroy.Op.bound.loc34_19.2(%.loc34_19.8)
// CHECK:STDOUT:     %impl.elem0.loc34_19.1: @TestIterate.%.loc34_19.21 (%.2f2) = impl_witness_access constants.%Destroy.lookup_impl_witness.3c4, element0 [symbolic = %impl.elem0.loc34_19.2 (constants.%impl.elem0.528)]
// CHECK:STDOUT:     %bound_method.loc34_19.7: <bound method> = bound_method %var, %impl.elem0.loc34_19.1
// CHECK:STDOUT:     %.loc34_19.17: %Destroy.type = converted constants.%as_type.038, constants.%impl.elem1.595 [symbolic = %impl.elem1 (constants.%impl.elem1.595)]
// CHECK:STDOUT:     %.loc34_19.18: %Destroy.type = converted constants.%as_type.038, constants.%impl.elem1.595 [symbolic = %impl.elem1 (constants.%impl.elem1.595)]
// CHECK:STDOUT:     %specific_impl_fn.loc34_19.3: <specific function> = specific_impl_function %impl.elem0.loc34_19.1, @Destroy.WithSelf.Op(constants.%impl.elem1.595) [symbolic = %specific_impl_fn.loc34_19.6 (constants.%specific_impl_fn.dfd)]
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
// CHECK:STDOUT: fn @Destroy.Op.loc34_19.5(%self.param: ref %DefaultOptionalStorage.bfa) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc34_19.6(%self.param: ref %Optional.755) {
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
// CHECK:STDOUT:   %impl.elem2.loc49: %.641 = impl_witness_access constants.%Iterate.impl_witness.0b0, element2 [concrete = constants.%T.as_type.as.Iterate.impl.NewCursor.45a]
// CHECK:STDOUT:   %bound_method.loc49_19.1: <bound method> = bound_method %m.ref, %impl.elem2.loc49
// CHECK:STDOUT:   %facet_value.loc49_19.1: %CppRange_where.type.d682d6.1 = facet_value constants.%MutableRange, (constants.%custom_witness.df9cc1.3, constants.%Copy.impl_witness.b51, constants.%custom_witness.ed3, constants.%custom_witness.955, constants.%custom_witness.ef8, constants.%custom_witness.444, constants.%custom_witness.981) [concrete = constants.%facet_value.c6e]
// CHECK:STDOUT:   %.loc49_19.1: %CppRange_where.type.d682d6.1 = converted constants.%MutableRange, %facet_value.loc49_19.1 [concrete = constants.%facet_value.c6e]
// CHECK:STDOUT:   %facet_value.loc49_19.2: %CppRange_where.type.d682d6.1 = facet_value constants.%MutableRange, (constants.%custom_witness.df9cc1.3, constants.%Copy.impl_witness.b51, constants.%custom_witness.ed3, constants.%custom_witness.955, constants.%custom_witness.ef8, constants.%custom_witness.444, constants.%custom_witness.981) [concrete = constants.%facet_value.c6e]
// CHECK:STDOUT:   %.loc49_19.2: %CppRange_where.type.d682d6.1 = converted constants.%MutableRange, %facet_value.loc49_19.2 [concrete = constants.%facet_value.c6e]
// CHECK:STDOUT:   %specific_fn.loc49_19.1: <specific function> = specific_function %impl.elem2.loc49, @T.as_type.as.Iterate.impl.NewCursor(constants.%facet_value.c6e) [concrete = constants.%T.as_type.as.Iterate.impl.NewCursor.specific_fn.b38]
// CHECK:STDOUT:   %bound_method.loc49_19.2: <bound method> = bound_method %m.ref, %specific_fn.loc49_19.1
// CHECK:STDOUT:   %var.loc49: ref %tuple.type.508 = var_storage invalid
// CHECK:STDOUT:   %.loc49_18.1: %MutableRange = acquire_value %m.ref
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.call.loc49: init %tuple.type.508 to %var.loc49 = call %bound_method.loc49_19.2(%.loc49_18.1)
// CHECK:STDOUT:   assign %var.loc49, %T.as_type.as.Iterate.impl.NewCursor.call.loc49
// CHECK:STDOUT:   br !for.next.loc49
// CHECK:STDOUT:
// CHECK:STDOUT: !for.next.loc49:
// CHECK:STDOUT:   %addr.loc49: %ptr.45c = addr_of %var.loc49
// CHECK:STDOUT:   %impl.elem3.loc49: %.d1b = impl_witness_access constants.%Iterate.impl_witness.0b0, element3 [concrete = constants.%T.as_type.as.Iterate.impl.Next.a5b]
// CHECK:STDOUT:   %bound_method.loc49_19.3: <bound method> = bound_method %m.ref, %impl.elem3.loc49
// CHECK:STDOUT:   %facet_value.loc49_19.3: %CppRange_where.type.d682d6.1 = facet_value constants.%MutableRange, (constants.%custom_witness.df9cc1.3, constants.%Copy.impl_witness.b51, constants.%custom_witness.ed3, constants.%custom_witness.955, constants.%custom_witness.ef8, constants.%custom_witness.444, constants.%custom_witness.981) [concrete = constants.%facet_value.c6e]
// CHECK:STDOUT:   %.loc49_19.3: %CppRange_where.type.d682d6.1 = converted constants.%MutableRange, %facet_value.loc49_19.3 [concrete = constants.%facet_value.c6e]
// CHECK:STDOUT:   %facet_value.loc49_19.4: %CppRange_where.type.d682d6.1 = facet_value constants.%MutableRange, (constants.%custom_witness.df9cc1.3, constants.%Copy.impl_witness.b51, constants.%custom_witness.ed3, constants.%custom_witness.955, constants.%custom_witness.ef8, constants.%custom_witness.444, constants.%custom_witness.981) [concrete = constants.%facet_value.c6e]
// CHECK:STDOUT:   %.loc49_19.4: %CppRange_where.type.d682d6.1 = converted constants.%MutableRange, %facet_value.loc49_19.4 [concrete = constants.%facet_value.c6e]
// CHECK:STDOUT:   %specific_fn.loc49_19.2: <specific function> = specific_function %impl.elem3.loc49, @T.as_type.as.Iterate.impl.Next(constants.%facet_value.c6e) [concrete = constants.%T.as_type.as.Iterate.impl.Next.specific_fn.b5a]
// CHECK:STDOUT:   %bound_method.loc49_19.4: <bound method> = bound_method %m.ref, %specific_fn.loc49_19.2
// CHECK:STDOUT:   %.loc49_19.5: ref %Optional.755 = temporary_storage
// CHECK:STDOUT:   %.loc49_18.2: %MutableRange = acquire_value %m.ref
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.call.loc49: init %Optional.755 to %.loc49_19.5 = call %bound_method.loc49_19.4(%.loc49_18.2, %addr.loc49)
// CHECK:STDOUT:   %.loc49_19.6: ref %Optional.755 = temporary %.loc49_19.5, %T.as_type.as.Iterate.impl.Next.call.loc49
// CHECK:STDOUT:   %.loc49_19.7: %Optional.HasValue.type.af3 = specific_constant imports.%Core.import_ref.bb6, @Optional(constants.%OptionalStorage.facet.097) [concrete = constants.%Optional.HasValue.4f6]
// CHECK:STDOUT:   %HasValue.ref.loc49: %Optional.HasValue.type.af3 = name_ref HasValue, %.loc49_19.7 [concrete = constants.%Optional.HasValue.4f6]
// CHECK:STDOUT:   %Optional.HasValue.bound.loc49: <bound method> = bound_method %.loc49_19.6, %HasValue.ref.loc49
// CHECK:STDOUT:   %Optional.HasValue.specific_fn.loc49: <specific function> = specific_function %HasValue.ref.loc49, @Optional.HasValue(constants.%OptionalStorage.facet.097) [concrete = constants.%Optional.HasValue.specific_fn.b8c]
// CHECK:STDOUT:   %bound_method.loc49_19.5: <bound method> = bound_method %.loc49_19.6, %Optional.HasValue.specific_fn.loc49
// CHECK:STDOUT:   %.loc49_19.8: %Optional.755 = acquire_value %.loc49_19.6
// CHECK:STDOUT:   %Optional.HasValue.call.loc49: init bool = call %bound_method.loc49_19.5(%.loc49_19.8)
// CHECK:STDOUT:   %.loc49_19.9: bool = value_of_initializer %Optional.HasValue.call.loc49
// CHECK:STDOUT:   %.loc49_19.10: bool = converted %Optional.HasValue.call.loc49, %.loc49_19.9
// CHECK:STDOUT:   if %.loc49_19.10 br !for.body.loc49 else br !for.done.loc49
// CHECK:STDOUT:
// CHECK:STDOUT: !for.body.loc49:
// CHECK:STDOUT:   %.loc49_19.11: %Optional.Get.type.336 = specific_constant imports.%Core.import_ref.dab, @Optional(constants.%OptionalStorage.facet.097) [concrete = constants.%Optional.Get.0a8]
// CHECK:STDOUT:   %Get.ref.loc49: %Optional.Get.type.336 = name_ref Get, %.loc49_19.11 [concrete = constants.%Optional.Get.0a8]
// CHECK:STDOUT:   %Optional.Get.bound.loc49: <bound method> = bound_method %.loc49_19.6, %Get.ref.loc49
// CHECK:STDOUT:   %Optional.Get.specific_fn.loc49: <specific function> = specific_function %Get.ref.loc49, @Optional.Get(constants.%OptionalStorage.facet.097) [concrete = constants.%Optional.Get.specific_fn.fc5]
// CHECK:STDOUT:   %bound_method.loc49_19.6: <bound method> = bound_method %.loc49_19.6, %Optional.Get.specific_fn.loc49
// CHECK:STDOUT:   %.loc49_19.12: %Optional.755 = acquire_value %.loc49_19.6
// CHECK:STDOUT:   %Optional.Get.call.loc49: init %i32 = call %bound_method.loc49_19.6(%.loc49_19.12)
// CHECK:STDOUT:   %.loc49_19.13: %i32 = value_of_initializer %Optional.Get.call.loc49
// CHECK:STDOUT:   %.loc49_19.14: %i32 = converted %Optional.Get.call.loc49, %.loc49_19.13
// CHECK:STDOUT:   %i32.loc49: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %i.loc49: %i32 = wrapper_binding i, %.loc49_19.14
// CHECK:STDOUT:   %sum.ref.loc50: ref %i32 = name_ref sum, %sum
// CHECK:STDOUT:   %i.ref.loc50: %i32 = name_ref i, %i.loc49
// CHECK:STDOUT:   %impl.elem0.loc50: %.004 = impl_witness_access constants.%AddAssignWith.impl_witness.e9d, element0 [concrete = constants.%Int.as.AddAssignWith.impl.Op.d34738.2]
// CHECK:STDOUT:   %bound_method.loc50_9.1: <bound method> = bound_method %sum.ref.loc50, %impl.elem0.loc50
// CHECK:STDOUT:   %specific_fn.loc50: <specific function> = specific_function %impl.elem0.loc50, @Int.as.AddAssignWith.impl.Op.1(constants.%int_32, constants.%ImplicitAs.facet.e88) [concrete = constants.%Int.as.AddAssignWith.impl.Op.specific_fn.d76216.1]
// CHECK:STDOUT:   %bound_method.loc50_9.2: <bound method> = bound_method %sum.ref.loc50, %specific_fn.loc50
// CHECK:STDOUT:   %.loc50: %Int.as.AddAssignWith.impl.Op.type.0b80aa.1 = specific_constant imports.%Core.Op.1b1, @Int.as.AddAssignWith.impl(constants.%int_32, constants.%ImplicitAs.facet.e88) [concrete = constants.%Int.as.AddAssignWith.impl.Op.d34738.1]
// CHECK:STDOUT:   %Op.ref.loc50: %Int.as.AddAssignWith.impl.Op.type.0b80aa.1 = name_ref Op, %.loc50 [concrete = constants.%Int.as.AddAssignWith.impl.Op.d34738.1]
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.bound.loc50: <bound method> = bound_method %sum.ref.loc50, %Op.ref.loc50
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.specific_fn.loc50: <specific function> = specific_function %Op.ref.loc50, @Int.as.AddAssignWith.impl.Op.2(constants.%int_32, constants.%ImplicitAs.facet.e88) [concrete = constants.%Int.as.AddAssignWith.impl.Op.specific_fn.d76216.2]
// CHECK:STDOUT:   %bound_method.loc50_9.3: <bound method> = bound_method %sum.ref.loc50, %Int.as.AddAssignWith.impl.Op.specific_fn.loc50
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.call.loc50: init %empty_tuple.type = call %bound_method.loc50_9.3(%sum.ref.loc50, %i.ref.loc50)
// CHECK:STDOUT:   %Destroy.Op.bound.loc49_19.1: <bound method> = bound_method %.loc49_19.6, constants.%Destroy.Op.1a2547.9
// CHECK:STDOUT:   %Destroy.Op.call.loc49_19.1: init %empty_tuple.type = call %Destroy.Op.bound.loc49_19.1(%.loc49_19.6)
// CHECK:STDOUT:   br !for.next.loc49
// CHECK:STDOUT:
// CHECK:STDOUT: !for.done.loc49:
// CHECK:STDOUT:   %Destroy.Op.bound.loc49_19.2: <bound method> = bound_method %.loc49_19.6, constants.%Destroy.Op.1a2547.9
// CHECK:STDOUT:   %Destroy.Op.call.loc49_19.2: init %empty_tuple.type = call %Destroy.Op.bound.loc49_19.2(%.loc49_19.6)
// CHECK:STDOUT:   %Destroy.Op.bound.loc49_19.3: <bound method> = bound_method %var.loc49, constants.%Destroy.Op.1a2547.10
// CHECK:STDOUT:   %Destroy.Op.call.loc49_19.3: init %empty_tuple.type = call %Destroy.Op.bound.loc49_19.3(%var.loc49)
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %i.patt.loc55: %pattern_type.6b6 = value_binding_pattern i [concrete = constants.%i.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c.ref: %ConstRange = name_ref c, %c
// CHECK:STDOUT:   %impl.elem2.loc55: %.c93 = impl_witness_access constants.%Iterate.impl_witness.bb2, element2 [concrete = constants.%T.as_type.as.Iterate.impl.NewCursor.ab8]
// CHECK:STDOUT:   %bound_method.loc55_19.1: <bound method> = bound_method %c.ref, %impl.elem2.loc55
// CHECK:STDOUT:   %facet_value.loc55_19.1: %CppRange_where.type.d682d6.1 = facet_value constants.%ConstRange, (constants.%custom_witness.df9cc1.3, constants.%Copy.impl_witness.b51, constants.%custom_witness.23e, constants.%custom_witness.6e8, constants.%custom_witness.adb, constants.%custom_witness.746, constants.%custom_witness.941) [concrete = constants.%facet_value.96f]
// CHECK:STDOUT:   %.loc55_19.1: %CppRange_where.type.d682d6.1 = converted constants.%ConstRange, %facet_value.loc55_19.1 [concrete = constants.%facet_value.96f]
// CHECK:STDOUT:   %facet_value.loc55_19.2: %CppRange_where.type.d682d6.1 = facet_value constants.%ConstRange, (constants.%custom_witness.df9cc1.3, constants.%Copy.impl_witness.b51, constants.%custom_witness.23e, constants.%custom_witness.6e8, constants.%custom_witness.adb, constants.%custom_witness.746, constants.%custom_witness.941) [concrete = constants.%facet_value.96f]
// CHECK:STDOUT:   %.loc55_19.2: %CppRange_where.type.d682d6.1 = converted constants.%ConstRange, %facet_value.loc55_19.2 [concrete = constants.%facet_value.96f]
// CHECK:STDOUT:   %specific_fn.loc55_19.1: <specific function> = specific_function %impl.elem2.loc55, @T.as_type.as.Iterate.impl.NewCursor(constants.%facet_value.96f) [concrete = constants.%T.as_type.as.Iterate.impl.NewCursor.specific_fn.31a]
// CHECK:STDOUT:   %bound_method.loc55_19.2: <bound method> = bound_method %c.ref, %specific_fn.loc55_19.1
// CHECK:STDOUT:   %var.loc55: ref %tuple.type.69f = var_storage invalid
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.call.loc55: init %tuple.type.69f to %var.loc55 = call %bound_method.loc55_19.2(%c.ref)
// CHECK:STDOUT:   assign %var.loc55, %T.as_type.as.Iterate.impl.NewCursor.call.loc55
// CHECK:STDOUT:   br !for.next.loc55
// CHECK:STDOUT:
// CHECK:STDOUT: !for.next.loc55:
// CHECK:STDOUT:   %addr.loc55: %ptr.c5c = addr_of %var.loc55
// CHECK:STDOUT:   %impl.elem3.loc55: %.fe6 = impl_witness_access constants.%Iterate.impl_witness.bb2, element3 [concrete = constants.%T.as_type.as.Iterate.impl.Next.a02]
// CHECK:STDOUT:   %bound_method.loc55_19.3: <bound method> = bound_method %c.ref, %impl.elem3.loc55
// CHECK:STDOUT:   %facet_value.loc55_19.3: %CppRange_where.type.d682d6.1 = facet_value constants.%ConstRange, (constants.%custom_witness.df9cc1.3, constants.%Copy.impl_witness.b51, constants.%custom_witness.23e, constants.%custom_witness.6e8, constants.%custom_witness.adb, constants.%custom_witness.746, constants.%custom_witness.941) [concrete = constants.%facet_value.96f]
// CHECK:STDOUT:   %.loc55_19.3: %CppRange_where.type.d682d6.1 = converted constants.%ConstRange, %facet_value.loc55_19.3 [concrete = constants.%facet_value.96f]
// CHECK:STDOUT:   %facet_value.loc55_19.4: %CppRange_where.type.d682d6.1 = facet_value constants.%ConstRange, (constants.%custom_witness.df9cc1.3, constants.%Copy.impl_witness.b51, constants.%custom_witness.23e, constants.%custom_witness.6e8, constants.%custom_witness.adb, constants.%custom_witness.746, constants.%custom_witness.941) [concrete = constants.%facet_value.96f]
// CHECK:STDOUT:   %.loc55_19.4: %CppRange_where.type.d682d6.1 = converted constants.%ConstRange, %facet_value.loc55_19.4 [concrete = constants.%facet_value.96f]
// CHECK:STDOUT:   %specific_fn.loc55_19.2: <specific function> = specific_function %impl.elem3.loc55, @T.as_type.as.Iterate.impl.Next(constants.%facet_value.96f) [concrete = constants.%T.as_type.as.Iterate.impl.Next.specific_fn.480]
// CHECK:STDOUT:   %bound_method.loc55_19.4: <bound method> = bound_method %c.ref, %specific_fn.loc55_19.2
// CHECK:STDOUT:   %.loc55_19.5: ref %Optional.755 = temporary_storage
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.call.loc55: init %Optional.755 to %.loc55_19.5 = call %bound_method.loc55_19.4(%c.ref, %addr.loc55)
// CHECK:STDOUT:   %.loc55_19.6: ref %Optional.755 = temporary %.loc55_19.5, %T.as_type.as.Iterate.impl.Next.call.loc55
// CHECK:STDOUT:   %.loc55_19.7: %Optional.HasValue.type.af3 = specific_constant imports.%Core.import_ref.bb6, @Optional(constants.%OptionalStorage.facet.097) [concrete = constants.%Optional.HasValue.4f6]
// CHECK:STDOUT:   %HasValue.ref.loc55: %Optional.HasValue.type.af3 = name_ref HasValue, %.loc55_19.7 [concrete = constants.%Optional.HasValue.4f6]
// CHECK:STDOUT:   %Optional.HasValue.bound.loc55: <bound method> = bound_method %.loc55_19.6, %HasValue.ref.loc55
// CHECK:STDOUT:   %Optional.HasValue.specific_fn.loc55: <specific function> = specific_function %HasValue.ref.loc55, @Optional.HasValue(constants.%OptionalStorage.facet.097) [concrete = constants.%Optional.HasValue.specific_fn.b8c]
// CHECK:STDOUT:   %bound_method.loc55_19.5: <bound method> = bound_method %.loc55_19.6, %Optional.HasValue.specific_fn.loc55
// CHECK:STDOUT:   %.loc55_19.8: %Optional.755 = acquire_value %.loc55_19.6
// CHECK:STDOUT:   %Optional.HasValue.call.loc55: init bool = call %bound_method.loc55_19.5(%.loc55_19.8)
// CHECK:STDOUT:   %.loc55_19.9: bool = value_of_initializer %Optional.HasValue.call.loc55
// CHECK:STDOUT:   %.loc55_19.10: bool = converted %Optional.HasValue.call.loc55, %.loc55_19.9
// CHECK:STDOUT:   if %.loc55_19.10 br !for.body.loc55 else br !for.done.loc55
// CHECK:STDOUT:
// CHECK:STDOUT: !for.body.loc55:
// CHECK:STDOUT:   %.loc55_19.11: %Optional.Get.type.336 = specific_constant imports.%Core.import_ref.dab, @Optional(constants.%OptionalStorage.facet.097) [concrete = constants.%Optional.Get.0a8]
// CHECK:STDOUT:   %Get.ref.loc55: %Optional.Get.type.336 = name_ref Get, %.loc55_19.11 [concrete = constants.%Optional.Get.0a8]
// CHECK:STDOUT:   %Optional.Get.bound.loc55: <bound method> = bound_method %.loc55_19.6, %Get.ref.loc55
// CHECK:STDOUT:   %Optional.Get.specific_fn.loc55: <specific function> = specific_function %Get.ref.loc55, @Optional.Get(constants.%OptionalStorage.facet.097) [concrete = constants.%Optional.Get.specific_fn.fc5]
// CHECK:STDOUT:   %bound_method.loc55_19.6: <bound method> = bound_method %.loc55_19.6, %Optional.Get.specific_fn.loc55
// CHECK:STDOUT:   %.loc55_19.12: %Optional.755 = acquire_value %.loc55_19.6
// CHECK:STDOUT:   %Optional.Get.call.loc55: init %i32 = call %bound_method.loc55_19.6(%.loc55_19.12)
// CHECK:STDOUT:   %.loc55_19.13: %i32 = value_of_initializer %Optional.Get.call.loc55
// CHECK:STDOUT:   %.loc55_19.14: %i32 = converted %Optional.Get.call.loc55, %.loc55_19.13
// CHECK:STDOUT:   %i32.loc55: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %i.loc55: %i32 = wrapper_binding i, %.loc55_19.14
// CHECK:STDOUT:   %sum.ref.loc56: ref %i32 = name_ref sum, %sum
// CHECK:STDOUT:   %i.ref.loc56: %i32 = name_ref i, %i.loc55
// CHECK:STDOUT:   %impl.elem0.loc56: %.004 = impl_witness_access constants.%AddAssignWith.impl_witness.e9d, element0 [concrete = constants.%Int.as.AddAssignWith.impl.Op.d34738.2]
// CHECK:STDOUT:   %bound_method.loc56_9.1: <bound method> = bound_method %sum.ref.loc56, %impl.elem0.loc56
// CHECK:STDOUT:   %specific_fn.loc56: <specific function> = specific_function %impl.elem0.loc56, @Int.as.AddAssignWith.impl.Op.1(constants.%int_32, constants.%ImplicitAs.facet.e88) [concrete = constants.%Int.as.AddAssignWith.impl.Op.specific_fn.d76216.1]
// CHECK:STDOUT:   %bound_method.loc56_9.2: <bound method> = bound_method %sum.ref.loc56, %specific_fn.loc56
// CHECK:STDOUT:   %.loc56: %Int.as.AddAssignWith.impl.Op.type.0b80aa.1 = specific_constant imports.%Core.Op.1b1, @Int.as.AddAssignWith.impl(constants.%int_32, constants.%ImplicitAs.facet.e88) [concrete = constants.%Int.as.AddAssignWith.impl.Op.d34738.1]
// CHECK:STDOUT:   %Op.ref.loc56: %Int.as.AddAssignWith.impl.Op.type.0b80aa.1 = name_ref Op, %.loc56 [concrete = constants.%Int.as.AddAssignWith.impl.Op.d34738.1]
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.bound.loc56: <bound method> = bound_method %sum.ref.loc56, %Op.ref.loc56
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.specific_fn.loc56: <specific function> = specific_function %Op.ref.loc56, @Int.as.AddAssignWith.impl.Op.2(constants.%int_32, constants.%ImplicitAs.facet.e88) [concrete = constants.%Int.as.AddAssignWith.impl.Op.specific_fn.d76216.2]
// CHECK:STDOUT:   %bound_method.loc56_9.3: <bound method> = bound_method %sum.ref.loc56, %Int.as.AddAssignWith.impl.Op.specific_fn.loc56
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.call.loc56: init %empty_tuple.type = call %bound_method.loc56_9.3(%sum.ref.loc56, %i.ref.loc56)
// CHECK:STDOUT:   %Destroy.Op.bound.loc55_19.1: <bound method> = bound_method %.loc55_19.6, constants.%Destroy.Op.1a2547.9
// CHECK:STDOUT:   %Destroy.Op.call.loc55_19.1: init %empty_tuple.type = call %Destroy.Op.bound.loc55_19.1(%.loc55_19.6)
// CHECK:STDOUT:   br !for.next.loc55
// CHECK:STDOUT:
// CHECK:STDOUT: !for.done.loc55:
// CHECK:STDOUT:   %Destroy.Op.bound.loc55_19.2: <bound method> = bound_method %.loc55_19.6, constants.%Destroy.Op.1a2547.9
// CHECK:STDOUT:   %Destroy.Op.call.loc55_19.2: init %empty_tuple.type = call %Destroy.Op.bound.loc55_19.2(%.loc55_19.6)
// CHECK:STDOUT:   %Destroy.Op.bound.loc55_19.3: <bound method> = bound_method %var.loc55, constants.%Destroy.Op.1a2547.11
// CHECK:STDOUT:   %Destroy.Op.call.loc55_19.3: init %empty_tuple.type = call %Destroy.Op.bound.loc55_19.3(%var.loc55)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestIterate(constants.%R) {
// CHECK:STDOUT:   %R.patt.loc30_17.2 => constants.%R.patt
// CHECK:STDOUT:   %R.loc30_17.1 => constants.%R
// CHECK:STDOUT:   %R.as_type.loc30_61.1 => constants.%R.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.073
// CHECK:STDOUT:   %r.param_patt.loc30_59.2 => constants.%r.param_patt.d39
// CHECK:STDOUT:   %r.patt.loc30_59.2 => constants.%r.patt.5af
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestIterate(constants.%facet_value.8a3) {
// CHECK:STDOUT:   %R.patt.loc30_17.2 => constants.%R.patt
// CHECK:STDOUT:   %R.loc30_17.1 => constants.%facet_value.8a3
// CHECK:STDOUT:   %R.as_type.loc30_61.1 => constants.%MutableRange
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.e635
// CHECK:STDOUT:   %r.param_patt.loc30_59.2 => constants.%r.param_patt.c79
// CHECK:STDOUT:   %r.patt.loc30_59.2 => constants.%r.patt.94f
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc30 => constants.%complete_type.357
// CHECK:STDOUT:   %Iterate.lookup_impl_witness => constants.%Iterate.impl_witness.0b0
// CHECK:STDOUT:   %Iterate.facet.loc34_19.5 => constants.%Iterate.facet.fdf
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type => constants.%Iterate.WithSelf.NewCursor.type.5e7
// CHECK:STDOUT:   %.loc34_19.19 => constants.%.641
// CHECK:STDOUT:   %impl.elem2.loc34_19.2 => constants.%T.as_type.as.Iterate.impl.NewCursor.45a
// CHECK:STDOUT:   %specific_impl_fn.loc34_19.4 => constants.%T.as_type.as.Iterate.impl.NewCursor.specific_fn.b38
// CHECK:STDOUT:   %impl.elem1 => constants.%Destroy.facet.30f
// CHECK:STDOUT:   %as_type => constants.%tuple.type.508
// CHECK:STDOUT:   %require_complete.1 => constants.%complete_type.c28
// CHECK:STDOUT:   %ptr => constants.%ptr.45c
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type => constants.%Iterate.WithSelf.Next.type.08a
// CHECK:STDOUT:   %.loc34_19.20 => constants.%.d1b
// CHECK:STDOUT:   %impl.elem3.loc34_19.2 => constants.%T.as_type.as.Iterate.impl.Next.a5b
// CHECK:STDOUT:   %specific_impl_fn.loc34_19.5 => constants.%T.as_type.as.Iterate.impl.Next.specific_fn.b5a
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.c6c
// CHECK:STDOUT:   %.loc34_19.21 => constants.%.0a4
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.df9cc1.10
// CHECK:STDOUT:   %impl.elem0.loc34_19.2 => constants.%Destroy.Op.1a2547.10
// CHECK:STDOUT:   %specific_impl_fn.loc34_19.6 => constants.%Destroy.Op.1a2547.10
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestIterate(constants.%facet_value.9af) {
// CHECK:STDOUT:   %R.patt.loc30_17.2 => constants.%R.patt
// CHECK:STDOUT:   %R.loc30_17.1 => constants.%facet_value.9af
// CHECK:STDOUT:   %R.as_type.loc30_61.1 => constants.%ConstRange
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.0ba
// CHECK:STDOUT:   %r.param_patt.loc30_59.2 => constants.%r.param_patt.83e
// CHECK:STDOUT:   %r.patt.loc30_59.2 => constants.%r.patt.c21
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc30 => constants.%complete_type.357
// CHECK:STDOUT:   %Iterate.lookup_impl_witness => constants.%Iterate.impl_witness.bb2
// CHECK:STDOUT:   %Iterate.facet.loc34_19.5 => constants.%Iterate.facet.4c8
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type => constants.%Iterate.WithSelf.NewCursor.type.9d7
// CHECK:STDOUT:   %.loc34_19.19 => constants.%.c93
// CHECK:STDOUT:   %impl.elem2.loc34_19.2 => constants.%T.as_type.as.Iterate.impl.NewCursor.ab8
// CHECK:STDOUT:   %specific_impl_fn.loc34_19.4 => constants.%T.as_type.as.Iterate.impl.NewCursor.specific_fn.31a
// CHECK:STDOUT:   %impl.elem1 => constants.%Destroy.facet.e4c
// CHECK:STDOUT:   %as_type => constants.%tuple.type.69f
// CHECK:STDOUT:   %require_complete.1 => constants.%complete_type.8d5
// CHECK:STDOUT:   %ptr => constants.%ptr.c5c
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type => constants.%Iterate.WithSelf.Next.type.f73
// CHECK:STDOUT:   %.loc34_19.20 => constants.%.fe6
// CHECK:STDOUT:   %impl.elem3.loc34_19.2 => constants.%T.as_type.as.Iterate.impl.Next.a02
// CHECK:STDOUT:   %specific_impl_fn.loc34_19.5 => constants.%T.as_type.as.Iterate.impl.Next.specific_fn.480
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
// CHECK:STDOUT:   %facet_type.f48: type = facet_type <@Destroy & @Copy> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %T.220: %OptionalStorage.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Optional.Get.type.bb9: type = fn_type @Optional.Get, @Optional(%T.220) [symbolic]
// CHECK:STDOUT:   %Optional.Get.fa1: %Optional.Get.type.bb9 = struct_value () [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.type.e9e: type = fn_type @Optional.HasValue, @Optional(%T.220) [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.6cd: %Optional.HasValue.type.e9e = struct_value () [symbolic]
// CHECK:STDOUT:   %T.283: %facet_type.f48 = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.d50: type = class_type @DefaultOptionalStorage, @DefaultOptionalStorage(%T.283) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Get.type.a12: type = fn_type @T.as_type.as.OptionalStorage.impl.Get, @T.as_type.as.OptionalStorage.impl(%T.283) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Get.d32: %T.as_type.as.OptionalStorage.impl.Get.type.a12 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Has.type.e5a: type = fn_type @T.as_type.as.OptionalStorage.impl.Has, @T.as_type.as.OptionalStorage.impl(%T.283) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Has.4ee: %T.as_type.as.OptionalStorage.impl.Has.type.e5a = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Some.type.492: type = fn_type @T.as_type.as.OptionalStorage.impl.Some, @T.as_type.as.OptionalStorage.impl(%T.283) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Some.89a: %T.as_type.as.OptionalStorage.impl.Some.type.492 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.None.type.0a1: type = fn_type @T.as_type.as.OptionalStorage.impl.None, @T.as_type.as.OptionalStorage.impl(%T.283) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.None.e06: %T.as_type.as.OptionalStorage.impl.None.type.0a1 = struct_value () [symbolic]
// CHECK:STDOUT:   %.Self.500: %Iterate.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %CppRange.type: type = facet_type <@CppRange> [concrete]
// CHECK:STDOUT:   %.Self.e0b: %CppRange.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %CppRangeForIterate.lookup_impl_witness.50b: <witness> = lookup_impl_witness %.Self.e0b, @CppRangeForIterate [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.32d: type = impl_witness_access %CppRangeForIterate.lookup_impl_witness.50b, element0 [symbolic_self]
// CHECK:STDOUT:   %CppUnsafeDeref.lookup_impl_witness.851: <witness> = lookup_impl_witness %impl.elem0.32d, @CppUnsafeDeref [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.10d: type = impl_witness_access %CppUnsafeDeref.lookup_impl_witness.851, element0 [symbolic_self]
// CHECK:STDOUT:   %CppRange_where.type.d682d6.1: type = facet_type <@CppRange where %impl.elem0.10d impls @Copy> [concrete]
// CHECK:STDOUT:   %T.f45530.2: %CppRange_where.type.d682d6.1 = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.type.b36: type = fn_type @T.as_type.as.Iterate.impl.Next, @T.as_type.as.Iterate.impl(%T.f45530.2) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.d6b: %T.as_type.as.Iterate.impl.Next.type.b36 = struct_value () [symbolic]
// CHECK:STDOUT:   %CppRangeForIterate.lookup_impl_witness.b581a8.2: <witness> = lookup_impl_witness %T.f45530.2, @CppRangeForIterate [symbolic]
// CHECK:STDOUT:   %impl.elem0.5f971c.2: type = impl_witness_access %CppRangeForIterate.lookup_impl_witness.b581a8.2, element0 [symbolic]
// CHECK:STDOUT:   %CppUnsafeDeref.lookup_impl_witness.87e543.2: <witness> = lookup_impl_witness %impl.elem0.5f971c.2, @CppUnsafeDeref [symbolic]
// CHECK:STDOUT:   %impl.elem0.493765.2: type = impl_witness_access %CppUnsafeDeref.lookup_impl_witness.87e543.2, element0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.type.953: type = fn_type @T.as_type.as.Iterate.impl.NewCursor, @T.as_type.as.Iterate.impl(%T.f45530.2) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.544: %T.as_type.as.Iterate.impl.NewCursor.type.953 = struct_value () [symbolic]
// CHECK:STDOUT:   %impl.elem1.c2dc15.2: type = impl_witness_access %CppRangeForIterate.lookup_impl_witness.b581a8.2, element1 [symbolic]
// CHECK:STDOUT:   %tuple.type.1ba: type = tuple_type (%impl.elem0.5f971c.2, %impl.elem1.c2dc15.2) [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.812: <witness> = lookup_impl_witness %tuple.type.1ba, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.9cd: %Destroy.type = facet_value %tuple.type.1ba, (%Destroy.lookup_impl_witness.812) [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.35816c.2: <witness> = lookup_impl_witness %impl.elem0.493765.2, @Destroy [symbolic]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.167: <witness> = lookup_impl_witness %impl.elem0.493765.2, @Copy [symbolic]
// CHECK:STDOUT:   %facet_value.f93: %facet_type.f48 = facet_value %impl.elem0.493765.2, (%Destroy.lookup_impl_witness.35816c.2, %Copy.lookup_impl_witness.167) [symbolic]
// CHECK:STDOUT:   %DiffValueType: type = class_type @DiffValueType [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %pattern_type.149: type = pattern_type %DiffValueType [concrete]
// CHECK:STDOUT:   %DiffValueType.cpp_destructor.type: type = fn_type @DiffValueType.cpp_destructor [concrete]
// CHECK:STDOUT:   %DiffValueType.cpp_destructor: %DiffValueType.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT:   %DiffValueType.Op.type.83af6e.1: type = fn_type @DiffValueType.Op.1 [concrete]
// CHECK:STDOUT:   %DiffValueType.Op.4dca31.1: %DiffValueType.Op.type.83af6e.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.e7a31c.1: <witness> = custom_witness (%DiffValueType.Op.4dca31.1), @Destroy [concrete]
// CHECK:STDOUT:   %const.879: type = const_type %DiffValueType [concrete]
// CHECK:STDOUT:   %ptr.816: type = ptr_type %const.879 [concrete]
// CHECK:STDOUT:   %ptr.cd0: type = ptr_type %DiffValueType [concrete]
// CHECK:STDOUT:   %DiffValueType.Op.type.83af6e.2: type = fn_type @DiffValueType.Op.2 [concrete]
// CHECK:STDOUT:   %DiffValueType.Op.4dca31.2: %DiffValueType.Op.type.83af6e.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.d0da43.1: <witness> = custom_witness (%DiffValueType.Op.4dca31.2), @Copy [concrete]
// CHECK:STDOUT:   %facet_value.824: %facet_type.f48 = facet_value %DiffValueType, (%custom_witness.e7a31c.1, %custom_witness.d0da43.1) [concrete]
// CHECK:STDOUT:   %Iterate.lookup_impl_witness.1d8: <witness> = lookup_impl_witness %.Self.500, @Iterate [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.294: %facet_type.f48 = impl_witness_access %Iterate.lookup_impl_witness.1d8, element0 [symbolic_self]
// CHECK:STDOUT:   %Iterate_where.type.3fe: type = facet_type <@Iterate where %impl.elem0.294 = %facet_value.824> [concrete]
// CHECK:STDOUT:   %pattern_type.d93: type = pattern_type %Iterate_where.type.3fe [concrete]
// CHECK:STDOUT:   %R.patt: %pattern_type.d93 = symbolic_binding_pattern R, 0 [symbolic]
// CHECK:STDOUT:   %R: %Iterate_where.type.3fe = symbolic_binding R, 0 [symbolic]
// CHECK:STDOUT:   %R.as_type: type = facet_access_type %R [symbolic]
// CHECK:STDOUT:   %pattern_type.20b: type = pattern_type %R.as_type [symbolic]
// CHECK:STDOUT:   %r.patt.9f3: %pattern_type.20b = ref_binding_pattern r [symbolic]
// CHECK:STDOUT:   %r.param_patt.b12: %pattern_type.20b = var_param_pattern %r.patt.9f3 [symbolic]
// CHECK:STDOUT:   %i.patt: %pattern_type.149 = value_binding_pattern i [concrete]
// CHECK:STDOUT:   %Iterate.lookup_impl_witness.b8b: <witness> = lookup_impl_witness %R, @Iterate [symbolic]
// CHECK:STDOUT:   %Iterate.facet.6ec: %Iterate.type = facet_value %R.as_type, (%Iterate.lookup_impl_witness.b8b) [symbolic]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.309: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.6ec) [symbolic]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.75a: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.6ec) [symbolic]
// CHECK:STDOUT:   %.799: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.309, %Iterate.facet.6ec [symbolic]
// CHECK:STDOUT:   %impl.elem2.114: %.799 = impl_witness_access %Iterate.lookup_impl_witness.b8b, element2 [symbolic]
// CHECK:STDOUT:   %impl.elem1.697: %Destroy.type = impl_witness_access %Iterate.lookup_impl_witness.b8b, element1 [symbolic]
// CHECK:STDOUT:   %as_type.59f: type = facet_access_type %impl.elem1.697 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.c74: <specific function> = specific_impl_function %impl.elem2.114, @Iterate.WithSelf.NewCursor(%Iterate.facet.6ec) [symbolic]
// CHECK:STDOUT:   %ptr.38a: type = ptr_type %as_type.59f [symbolic]
// CHECK:STDOUT:   %.f06: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.75a, %Iterate.facet.6ec [symbolic]
// CHECK:STDOUT:   %impl.elem3.463: %.f06 = impl_witness_access %Iterate.lookup_impl_witness.b8b, element3 [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.5e4: type = class_type @DefaultOptionalStorage, @DefaultOptionalStorage(%facet_value.824) [concrete]
// CHECK:STDOUT:   %OptionalStorage.impl_witness.313: <witness> = impl_witness imports.%OptionalStorage.impl_witness_table.460, @T.as_type.as.OptionalStorage.impl(%facet_value.824) [concrete]
// CHECK:STDOUT:   %OptionalStorage.facet.c3f: %OptionalStorage.type = facet_value %DiffValueType, (%OptionalStorage.impl_witness.313) [concrete]
// CHECK:STDOUT:   %Optional.15a: type = class_type @Optional, @Optional(%OptionalStorage.facet.c3f) [concrete]
// CHECK:STDOUT:   %specific_impl_fn.dcf: <specific function> = specific_impl_function %impl.elem3.463, @Iterate.WithSelf.Next(%Iterate.facet.6ec) [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.type.6f3: type = fn_type @Optional.HasValue, @Optional(%OptionalStorage.facet.c3f) [concrete]
// CHECK:STDOUT:   %Optional.HasValue.038: %Optional.HasValue.type.6f3 = struct_value () [concrete]
// CHECK:STDOUT:   %Optional.Get.type.d3e: type = fn_type @Optional.Get, @Optional(%OptionalStorage.facet.c3f) [concrete]
// CHECK:STDOUT:   %Optional.Get.57f: %Optional.Get.type.d3e = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.facet.82e: %Destroy.type = facet_value %DiffValueType, (%custom_witness.e7a31c.1) [concrete]
// CHECK:STDOUT:   %MaybeUnformed.923: type = class_type @MaybeUnformed, @MaybeUnformed(%Destroy.facet.82e) [concrete]
// CHECK:STDOUT:   %.a18: type = maybe_unformed_type %DiffValueType [concrete]
// CHECK:STDOUT:   %struct_type.value.has_value.3cf: type = struct_type {.value: %MaybeUnformed.923, .has_value: bool} [concrete]
// CHECK:STDOUT:   %Optional.HasValue.specific_fn.bc4: <specific function> = specific_function %Optional.HasValue.038, @Optional.HasValue(%OptionalStorage.facet.c3f) [concrete]
// CHECK:STDOUT:   %Optional.Get.specific_fn.c49: <specific function> = specific_function %Optional.Get.57f, @Optional.Get(%OptionalStorage.facet.c3f) [concrete]
// CHECK:STDOUT:   %operator_PlusEqual__carbon_thunk.type: type = fn_type @operator_PlusEqual__carbon_thunk [concrete]
// CHECK:STDOUT:   %operator_PlusEqual__carbon_thunk: %operator_PlusEqual__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.7: type = fn_type @Destroy.Op.loc44_33.6 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.7: %Destroy.Op.type.1d8f74.7 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.38d: <witness> = lookup_impl_witness %impl.elem1.697, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.dbc: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%impl.elem1.697) [symbolic]
// CHECK:STDOUT:   %.763: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.dbc, %impl.elem1.697 [symbolic]
// CHECK:STDOUT:   %impl.elem0.891: %.763 = impl_witness_access %Destroy.lookup_impl_witness.38d, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.b52: <specific function> = specific_impl_function %impl.elem0.891, @Destroy.WithSelf.Op(%impl.elem1.697) [symbolic]
// CHECK:STDOUT:   %DiffMutableRange: type = class_type @DiffMutableRange [concrete]
// CHECK:STDOUT:   %pattern_type.611: type = pattern_type %DiffMutableRange [concrete]
// CHECK:STDOUT:   %DiffConstRange: type = class_type @DiffConstRange [concrete]
// CHECK:STDOUT:   %pattern_type.7dd: type = pattern_type %DiffConstRange [concrete]
// CHECK:STDOUT:   %Iterator.f25: type = class_type @Iterator.1 [concrete]
// CHECK:STDOUT:   %Sentinel.b4c: type = class_type @Sentinel.1 [concrete]
// CHECK:STDOUT:   %DiffMutableRange.Begin.type: type = fn_type @DiffMutableRange.Begin [concrete]
// CHECK:STDOUT:   %DiffMutableRange.Begin: %DiffMutableRange.Begin.type = struct_value () [concrete]
// CHECK:STDOUT:   %DiffMutableRange.End.type: type = fn_type @DiffMutableRange.End [concrete]
// CHECK:STDOUT:   %DiffMutableRange.End: %DiffMutableRange.End.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.aa8: <witness> = custom_witness (%Iterator.f25, %Sentinel.b4c, %DiffMutableRange.Begin, %DiffMutableRange.End), @CppRangeForIterate [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.41bbea.1: type = fn_type @Iterator.Op.1 [concrete]
// CHECK:STDOUT:   %Iterator.Op.1caa36.1: %Iterator.Op.type.41bbea.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.392: <witness> = custom_witness (%DiffValueType, %Iterator.Op.1caa36.1), @CppUnsafeDeref [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.41bbea.2: type = fn_type @Iterator.Op.2 [concrete]
// CHECK:STDOUT:   %Iterator.Op.1caa36.2: %Iterator.Op.type.41bbea.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.9bb15e.1: <witness> = custom_witness (%Iterator.Op.1caa36.2), @Destroy [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.41bbea.3: type = fn_type @Iterator.Op.3 [concrete]
// CHECK:STDOUT:   %Iterator.Op.1caa36.3: %Iterator.Op.type.41bbea.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.1c9: <witness> = custom_witness (%Iterator.Op.1caa36.3), @Inc [concrete]
// CHECK:STDOUT:   %Equal.type.ca8475.1: type = fn_type @Equal.1 [concrete]
// CHECK:STDOUT:   %Equal.285b15.1: %Equal.type.ca8475.1 = struct_value () [concrete]
// CHECK:STDOUT:   %NotEqual.type.53d9ee.1: type = fn_type @NotEqual.1 [concrete]
// CHECK:STDOUT:   %NotEqual.119263.1: %NotEqual.type.53d9ee.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.723: <witness> = custom_witness (%Equal.285b15.1, %NotEqual.119263.1), @EqWith, @EqWith(%Sentinel.b4c) [concrete]
// CHECK:STDOUT:   %Sentinel.Op.type.41b: type = fn_type @Sentinel.Op.1 [concrete]
// CHECK:STDOUT:   %Sentinel.Op.1ca: %Sentinel.Op.type.41b = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.9bb15e.2: <witness> = custom_witness (%Sentinel.Op.1ca), @Destroy [concrete]
// CHECK:STDOUT:   %facet_value.0a7: %CppRange_where.type.d682d6.1 = facet_value %DiffMutableRange, (%custom_witness.e7a31c.1, %custom_witness.d0da43.1, %custom_witness.aa8, %custom_witness.9bb15e.1, %custom_witness.392, %custom_witness.1c9, %custom_witness.723, %custom_witness.9bb15e.2) [concrete]
// CHECK:STDOUT:   %tuple.type.b18: type = tuple_type (%Iterator.f25, %Sentinel.b4c) [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.8: type = fn_type @Destroy.Op.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.8: %Destroy.Op.type.1d8f74.8 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.df9cc1.8: <witness> = custom_witness (%Destroy.Op.1a2547.8), @Destroy [concrete]
// CHECK:STDOUT:   %Destroy.facet.2c1: %Destroy.type = facet_value %tuple.type.b18, (%custom_witness.df9cc1.8) [concrete]
// CHECK:STDOUT:   %Iterate.impl_witness.747: <witness> = impl_witness imports.%Iterate.impl_witness_table.121, @T.as_type.as.Iterate.impl(%facet_value.0a7) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.type.096: type = fn_type @T.as_type.as.Iterate.impl.NewCursor, @T.as_type.as.Iterate.impl(%facet_value.0a7) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.136: %T.as_type.as.Iterate.impl.NewCursor.type.096 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.type.185: type = fn_type @T.as_type.as.Iterate.impl.Next, @T.as_type.as.Iterate.impl(%facet_value.0a7) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.31b: %T.as_type.as.Iterate.impl.Next.type.185 = struct_value () [concrete]
// CHECK:STDOUT:   %Iterate.facet.68a: %Iterate.type = facet_value %DiffMutableRange, (%Iterate.impl_witness.747) [concrete]
// CHECK:STDOUT:   %facet_value.753: %Iterate_where.type.3fe = facet_value %DiffMutableRange, (%Iterate.impl_witness.747) [concrete]
// CHECK:STDOUT:   %r.patt.5fe: %pattern_type.611 = ref_binding_pattern r [concrete]
// CHECK:STDOUT:   %r.param_patt.a4b: %pattern_type.611 = var_param_pattern %r.patt.5fe [concrete]
// CHECK:STDOUT:   %Iterator.a59: type = class_type @Iterator.2 [concrete]
// CHECK:STDOUT:   %Sentinel.b33: type = class_type @Sentinel.2 [concrete]
// CHECK:STDOUT:   %DiffConstRange.Begin.type: type = fn_type @DiffConstRange.Begin [concrete]
// CHECK:STDOUT:   %DiffConstRange.Begin: %DiffConstRange.Begin.type = struct_value () [concrete]
// CHECK:STDOUT:   %DiffConstRange.End.type: type = fn_type @DiffConstRange.End [concrete]
// CHECK:STDOUT:   %DiffConstRange.End: %DiffConstRange.End.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.7b4: <witness> = custom_witness (%Iterator.a59, %Sentinel.b33, %DiffConstRange.Begin, %DiffConstRange.End), @CppRangeForIterate [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.b7ae80.1: type = fn_type @Iterator.Op.4 [concrete]
// CHECK:STDOUT:   %Iterator.Op.690fee.1: %Iterator.Op.type.b7ae80.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.0c0: <witness> = custom_witness (%DiffValueType, %Iterator.Op.690fee.1), @CppUnsafeDeref [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.b7ae80.2: type = fn_type @Iterator.Op.5 [concrete]
// CHECK:STDOUT:   %Iterator.Op.690fee.2: %Iterator.Op.type.b7ae80.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.f7d4ff.1: <witness> = custom_witness (%Iterator.Op.690fee.2), @Destroy [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.b7ae80.3: type = fn_type @Iterator.Op.6 [concrete]
// CHECK:STDOUT:   %Iterator.Op.690fee.3: %Iterator.Op.type.b7ae80.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.ffb: <witness> = custom_witness (%Iterator.Op.690fee.3), @Inc [concrete]
// CHECK:STDOUT:   %Equal.type.ca8475.2: type = fn_type @Equal.2 [concrete]
// CHECK:STDOUT:   %Equal.285b15.2: %Equal.type.ca8475.2 = struct_value () [concrete]
// CHECK:STDOUT:   %NotEqual.type.53d9ee.2: type = fn_type @NotEqual.2 [concrete]
// CHECK:STDOUT:   %NotEqual.119263.2: %NotEqual.type.53d9ee.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.a26: <witness> = custom_witness (%Equal.285b15.2, %NotEqual.119263.2), @EqWith, @EqWith(%Sentinel.b33) [concrete]
// CHECK:STDOUT:   %Sentinel.Op.type.b7a: type = fn_type @Sentinel.Op.2 [concrete]
// CHECK:STDOUT:   %Sentinel.Op.690: %Sentinel.Op.type.b7a = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.f7d4ff.2: <witness> = custom_witness (%Sentinel.Op.690), @Destroy [concrete]
// CHECK:STDOUT:   %facet_value.733: %CppRange_where.type.d682d6.1 = facet_value %DiffConstRange, (%custom_witness.e7a31c.1, %custom_witness.d0da43.1, %custom_witness.7b4, %custom_witness.f7d4ff.1, %custom_witness.0c0, %custom_witness.ffb, %custom_witness.a26, %custom_witness.f7d4ff.2) [concrete]
// CHECK:STDOUT:   %tuple.type.df5: type = tuple_type (%Iterator.a59, %Sentinel.b33) [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.9: type = fn_type @Destroy.Op.3 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.9: %Destroy.Op.type.1d8f74.9 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.df9cc1.9: <witness> = custom_witness (%Destroy.Op.1a2547.9), @Destroy [concrete]
// CHECK:STDOUT:   %Destroy.facet.1ed: %Destroy.type = facet_value %tuple.type.df5, (%custom_witness.df9cc1.9) [concrete]
// CHECK:STDOUT:   %Iterate.impl_witness.f7f: <witness> = impl_witness imports.%Iterate.impl_witness_table.121, @T.as_type.as.Iterate.impl(%facet_value.733) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.type.2bd: type = fn_type @T.as_type.as.Iterate.impl.NewCursor, @T.as_type.as.Iterate.impl(%facet_value.733) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.1a9: %T.as_type.as.Iterate.impl.NewCursor.type.2bd = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.type.e26: type = fn_type @T.as_type.as.Iterate.impl.Next, @T.as_type.as.Iterate.impl(%facet_value.733) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.59a: %T.as_type.as.Iterate.impl.Next.type.e26 = struct_value () [concrete]
// CHECK:STDOUT:   %Iterate.facet.608: %Iterate.type = facet_value %DiffConstRange, (%Iterate.impl_witness.f7f) [concrete]
// CHECK:STDOUT:   %facet_value.1e4: %Iterate_where.type.3fe = facet_value %DiffConstRange, (%Iterate.impl_witness.f7f) [concrete]
// CHECK:STDOUT:   %r.patt.9c0: %pattern_type.7dd = ref_binding_pattern r [concrete]
// CHECK:STDOUT:   %r.param_patt.992: %pattern_type.7dd = var_param_pattern %r.patt.9c0 [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.e9c: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.68a) [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.a94: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.68a) [concrete]
// CHECK:STDOUT:   %.cbf: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.e9c, %Iterate.facet.68a [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.specific_fn.7b5: <specific function> = specific_function %T.as_type.as.Iterate.impl.NewCursor.136, @T.as_type.as.Iterate.impl.NewCursor(%facet_value.0a7) [concrete]
// CHECK:STDOUT:   %ptr.3ce: type = ptr_type %tuple.type.b18 [concrete]
// CHECK:STDOUT:   %.922: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.a94, %Iterate.facet.68a [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.specific_fn.a81: <specific function> = specific_function %T.as_type.as.Iterate.impl.Next.31b, @T.as_type.as.Iterate.impl.Next(%facet_value.0a7) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.188: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.2c1) [concrete]
// CHECK:STDOUT:   %.082: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.188, %Destroy.facet.2c1 [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.185: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.608) [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.f38: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.608) [concrete]
// CHECK:STDOUT:   %.f70: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.185, %Iterate.facet.608 [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.specific_fn.f9d: <specific function> = specific_function %T.as_type.as.Iterate.impl.NewCursor.1a9, @T.as_type.as.Iterate.impl.NewCursor(%facet_value.733) [concrete]
// CHECK:STDOUT:   %ptr.310: type = ptr_type %tuple.type.df5 [concrete]
// CHECK:STDOUT:   %.802: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.f38, %Iterate.facet.608 [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.specific_fn.2ac: <specific function> = specific_function %T.as_type.as.Iterate.impl.Next.59a, @T.as_type.as.Iterate.impl.Next(%facet_value.733) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.a81: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.1ed) [concrete]
// CHECK:STDOUT:   %.471: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.a81, %Destroy.facet.1ed [concrete]
// CHECK:STDOUT:   %complete_type.bfe: <witness> = complete_type_witness %tuple.type.b18 [concrete]
// CHECK:STDOUT:   %complete_type.ad6: <witness> = complete_type_witness %tuple.type.df5 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .DiffValueType = %DiffValueType.decl
// CHECK:STDOUT:     .DiffMutableRange = %DiffMutableRange.decl
// CHECK:STDOUT:     .DiffConstRange = %DiffConstRange.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.bb6: @Optional.%Optional.HasValue.type (%Optional.HasValue.type.e9e) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.HasValue (constants.%Optional.HasValue.6cd)]
// CHECK:STDOUT:   %Core.import_ref.dab: @Optional.%Optional.Get.type (%Optional.Get.type.bb9) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.Get (constants.%Optional.Get.fa1)]
// CHECK:STDOUT:   %Core.import_ref.0c6: type = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%DefaultOptionalStorage (constants.%DefaultOptionalStorage.d50)]
// CHECK:STDOUT:   %Core.import_ref.dc3: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.None.type (%T.as_type.as.OptionalStorage.impl.None.type.0a1) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.None (constants.%T.as_type.as.OptionalStorage.impl.None.e06)]
// CHECK:STDOUT:   %Core.import_ref.c53: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Some.type (%T.as_type.as.OptionalStorage.impl.Some.type.492) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Some (constants.%T.as_type.as.OptionalStorage.impl.Some.89a)]
// CHECK:STDOUT:   %Core.import_ref.ed4: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Has.type (%T.as_type.as.OptionalStorage.impl.Has.type.e5a) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Has (constants.%T.as_type.as.OptionalStorage.impl.Has.4ee)]
// CHECK:STDOUT:   %Core.import_ref.e9c: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Get.type (%T.as_type.as.OptionalStorage.impl.Get.type.a12) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Get (constants.%T.as_type.as.OptionalStorage.impl.Get.d32)]
// CHECK:STDOUT:   %Core.import_ref.0b2 = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], unloaded
// CHECK:STDOUT:   %OptionalStorage.impl_witness_table.460 = impl_witness_table (%Core.import_ref.0c6, %Core.import_ref.dc3, %Core.import_ref.c53, %Core.import_ref.ed4, %Core.import_ref.e9c, %Core.import_ref.0b2), @T.as_type.as.OptionalStorage.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.eef: %facet_type.f48 = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.Iterate.impl.%facet_value (constants.%facet_value.f93)]
// CHECK:STDOUT:   %Core.import_ref.fcc: %Destroy.type = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.Iterate.impl.%Destroy.facet (constants.%Destroy.facet.9cd)]
// CHECK:STDOUT:   %Core.import_ref.fd1: @T.as_type.as.Iterate.impl.%T.as_type.as.Iterate.impl.NewCursor.type (%T.as_type.as.Iterate.impl.NewCursor.type.953) = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.Iterate.impl.%T.as_type.as.Iterate.impl.NewCursor (constants.%T.as_type.as.Iterate.impl.NewCursor.544)]
// CHECK:STDOUT:   %Core.import_ref.273: @T.as_type.as.Iterate.impl.%T.as_type.as.Iterate.impl.Next.type (%T.as_type.as.Iterate.impl.Next.type.b36) = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.Iterate.impl.%T.as_type.as.Iterate.impl.Next (constants.%T.as_type.as.Iterate.impl.Next.d6b)]
// CHECK:STDOUT:   %Iterate.impl_witness_table.121 = impl_witness_table (%Core.import_ref.eef, %Core.import_ref.fcc, %Core.import_ref.fd1, %Core.import_ref.273), @T.as_type.as.Iterate.impl [concrete]
// CHECK:STDOUT:   %DiffValueType.decl: type = class_decl @DiffValueType [concrete = constants.%DiffValueType] {} {}
// CHECK:STDOUT:   %DiffValueType.cpp_destructor.decl: %DiffValueType.cpp_destructor.type = fn_decl @DiffValueType.cpp_destructor [concrete = constants.%DiffValueType.cpp_destructor] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %operator_PlusEqual__carbon_thunk.decl: %operator_PlusEqual__carbon_thunk.type = fn_decl @operator_PlusEqual__carbon_thunk [concrete = constants.%operator_PlusEqual__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %DiffMutableRange.decl: type = class_decl @DiffMutableRange [concrete = constants.%DiffMutableRange] {} {}
// CHECK:STDOUT:   %DiffConstRange.decl: type = class_decl @DiffConstRange [concrete = constants.%DiffConstRange] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @TestIterate(%R.loc40_17.2: %Iterate_where.type.3fe) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Iterate.lookup_impl_witness: <witness> = lookup_impl_witness %R.loc40_17.1, @Iterate [symbolic = %Iterate.lookup_impl_witness (constants.%Iterate.lookup_impl_witness.b8b)]
// CHECK:STDOUT:   %Iterate.facet.loc44_33.5: %Iterate.type = facet_value %R.as_type.loc40_79.1, (%Iterate.lookup_impl_witness) [symbolic = %Iterate.facet.loc44_33.5 (constants.%Iterate.facet.6ec)]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.loc44_33.5) [symbolic = %Iterate.WithSelf.NewCursor.type (constants.%Iterate.WithSelf.NewCursor.type.309)]
// CHECK:STDOUT:   %.loc44_33.20: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type, %Iterate.facet.loc44_33.5 [symbolic = %.loc44_33.20 (constants.%.799)]
// CHECK:STDOUT:   %impl.elem2.loc44_33.2: @TestIterate.%.loc44_33.20 (%.799) = impl_witness_access %Iterate.lookup_impl_witness, element2 [symbolic = %impl.elem2.loc44_33.2 (constants.%impl.elem2.114)]
// CHECK:STDOUT:   %specific_impl_fn.loc44_33.4: <specific function> = specific_impl_function %impl.elem2.loc44_33.2, @Iterate.WithSelf.NewCursor(%Iterate.facet.loc44_33.5) [symbolic = %specific_impl_fn.loc44_33.4 (constants.%specific_impl_fn.c74)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %ptr: type = ptr_type %as_type [symbolic = %ptr (constants.%ptr.38a)]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.loc44_33.5) [symbolic = %Iterate.WithSelf.Next.type (constants.%Iterate.WithSelf.Next.type.75a)]
// CHECK:STDOUT:   %.loc44_33.21: type = fn_type_with_self_type %Iterate.WithSelf.Next.type, %Iterate.facet.loc44_33.5 [symbolic = %.loc44_33.21 (constants.%.f06)]
// CHECK:STDOUT:   %impl.elem3.loc44_33.2: @TestIterate.%.loc44_33.21 (%.f06) = impl_witness_access %Iterate.lookup_impl_witness, element3 [symbolic = %impl.elem3.loc44_33.2 (constants.%impl.elem3.463)]
// CHECK:STDOUT:   %specific_impl_fn.loc44_33.5: <specific function> = specific_impl_function %impl.elem3.loc44_33.2, @Iterate.WithSelf.Next(%Iterate.facet.loc44_33.5) [symbolic = %specific_impl_fn.loc44_33.5 (constants.%specific_impl_fn.dcf)]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%impl.elem1) [symbolic = %Destroy.WithSelf.Op.type (constants.%Destroy.WithSelf.Op.type.dbc)]
// CHECK:STDOUT:   %.loc44_33.22: type = fn_type_with_self_type %Destroy.WithSelf.Op.type, %impl.elem1 [symbolic = %.loc44_33.22 (constants.%.763)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %impl.elem1, @Destroy [symbolic = %Destroy.lookup_impl_witness (constants.%Destroy.lookup_impl_witness.38d)]
// CHECK:STDOUT:   %impl.elem0.loc44_33.2: @TestIterate.%.loc44_33.22 (%.763) = impl_witness_access %Destroy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc44_33.2 (constants.%impl.elem0.891)]
// CHECK:STDOUT:   %specific_impl_fn.loc44_33.6: <specific function> = specific_impl_function %impl.elem0.loc44_33.2, @Destroy.WithSelf.Op(%impl.elem1) [symbolic = %specific_impl_fn.loc44_33.6 (constants.%specific_impl_fn.b52)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%r.param: ref @TestIterate.%R.as_type.loc40_79.1 (%R.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %i.patt: %pattern_type.149 = value_binding_pattern i [concrete = constants.%i.patt]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %r.ref: ref @TestIterate.%R.as_type.loc40_79.1 (%R.as_type) = name_ref r, %r
// CHECK:STDOUT:     %impl.elem2.loc44_33.1: @TestIterate.%.loc44_33.20 (%.799) = impl_witness_access constants.%Iterate.lookup_impl_witness.b8b, element2 [symbolic = %impl.elem2.loc44_33.2 (constants.%impl.elem2.114)]
// CHECK:STDOUT:     %bound_method.loc44_33.1: <bound method> = bound_method %r.ref, %impl.elem2.loc44_33.1
// CHECK:STDOUT:     %Iterate.facet.loc44_33.1: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.b8b) [symbolic = %Iterate.facet.loc44_33.5 (constants.%Iterate.facet.6ec)]
// CHECK:STDOUT:     %.loc44_33.1: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc44_33.1 [symbolic = %Iterate.facet.loc44_33.5 (constants.%Iterate.facet.6ec)]
// CHECK:STDOUT:     %Iterate.facet.loc44_33.2: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.b8b) [symbolic = %Iterate.facet.loc44_33.5 (constants.%Iterate.facet.6ec)]
// CHECK:STDOUT:     %.loc44_33.2: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc44_33.2 [symbolic = %Iterate.facet.loc44_33.5 (constants.%Iterate.facet.6ec)]
// CHECK:STDOUT:     %specific_impl_fn.loc44_33.1: <specific function> = specific_impl_function %impl.elem2.loc44_33.1, @Iterate.WithSelf.NewCursor(constants.%Iterate.facet.6ec) [symbolic = %specific_impl_fn.loc44_33.4 (constants.%specific_impl_fn.c74)]
// CHECK:STDOUT:     %bound_method.loc44_33.2: <bound method> = bound_method %r.ref, %specific_impl_fn.loc44_33.1
// CHECK:STDOUT:     %var: ref @TestIterate.%as_type (%as_type.59f) = var_storage invalid
// CHECK:STDOUT:     %.loc44_32.1: @TestIterate.%R.as_type.loc40_79.1 (%R.as_type) = acquire_value %r.ref
// CHECK:STDOUT:     %Iterate.WithSelf.NewCursor.call: init @TestIterate.%as_type (%as_type.59f) to %var = call %bound_method.loc44_33.2(%.loc44_32.1)
// CHECK:STDOUT:     assign %var, %Iterate.WithSelf.NewCursor.call
// CHECK:STDOUT:     br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.next:
// CHECK:STDOUT:     %addr.loc44: @TestIterate.%ptr (%ptr.38a) = addr_of %var
// CHECK:STDOUT:     %impl.elem3.loc44_33.1: @TestIterate.%.loc44_33.21 (%.f06) = impl_witness_access constants.%Iterate.lookup_impl_witness.b8b, element3 [symbolic = %impl.elem3.loc44_33.2 (constants.%impl.elem3.463)]
// CHECK:STDOUT:     %bound_method.loc44_33.3: <bound method> = bound_method %r.ref, %impl.elem3.loc44_33.1
// CHECK:STDOUT:     %Iterate.facet.loc44_33.3: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.b8b) [symbolic = %Iterate.facet.loc44_33.5 (constants.%Iterate.facet.6ec)]
// CHECK:STDOUT:     %.loc44_33.3: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc44_33.3 [symbolic = %Iterate.facet.loc44_33.5 (constants.%Iterate.facet.6ec)]
// CHECK:STDOUT:     %Iterate.facet.loc44_33.4: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.b8b) [symbolic = %Iterate.facet.loc44_33.5 (constants.%Iterate.facet.6ec)]
// CHECK:STDOUT:     %.loc44_33.4: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc44_33.4 [symbolic = %Iterate.facet.loc44_33.5 (constants.%Iterate.facet.6ec)]
// CHECK:STDOUT:     %.loc44_33.5: %Destroy.type = converted constants.%as_type.59f, constants.%impl.elem1.697 [symbolic = %impl.elem1 (constants.%impl.elem1.697)]
// CHECK:STDOUT:     %.loc44_33.6: %Destroy.type = converted constants.%as_type.59f, constants.%impl.elem1.697 [symbolic = %impl.elem1 (constants.%impl.elem1.697)]
// CHECK:STDOUT:     %specific_impl_fn.loc44_33.2: <specific function> = specific_impl_function %impl.elem3.loc44_33.1, @Iterate.WithSelf.Next(constants.%Iterate.facet.6ec) [symbolic = %specific_impl_fn.loc44_33.5 (constants.%specific_impl_fn.dcf)]
// CHECK:STDOUT:     %bound_method.loc44_33.4: <bound method> = bound_method %r.ref, %specific_impl_fn.loc44_33.2
// CHECK:STDOUT:     %.loc44_33.7: ref %Optional.15a = temporary_storage
// CHECK:STDOUT:     %.loc44_32.2: @TestIterate.%R.as_type.loc40_79.1 (%R.as_type) = acquire_value %r.ref
// CHECK:STDOUT:     %Iterate.WithSelf.Next.call: init %Optional.15a to %.loc44_33.7 = call %bound_method.loc44_33.4(%.loc44_32.2, %addr.loc44)
// CHECK:STDOUT:     %.loc44_33.8: ref %Optional.15a = temporary %.loc44_33.7, %Iterate.WithSelf.Next.call
// CHECK:STDOUT:     %.loc44_33.9: %Optional.HasValue.type.6f3 = specific_constant imports.%Core.import_ref.bb6, @Optional(constants.%OptionalStorage.facet.c3f) [concrete = constants.%Optional.HasValue.038]
// CHECK:STDOUT:     %HasValue.ref: %Optional.HasValue.type.6f3 = name_ref HasValue, %.loc44_33.9 [concrete = constants.%Optional.HasValue.038]
// CHECK:STDOUT:     %Optional.HasValue.bound: <bound method> = bound_method %.loc44_33.8, %HasValue.ref
// CHECK:STDOUT:     %Optional.HasValue.specific_fn: <specific function> = specific_function %HasValue.ref, @Optional.HasValue(constants.%OptionalStorage.facet.c3f) [concrete = constants.%Optional.HasValue.specific_fn.bc4]
// CHECK:STDOUT:     %bound_method.loc44_33.5: <bound method> = bound_method %.loc44_33.8, %Optional.HasValue.specific_fn
// CHECK:STDOUT:     %.loc44_33.10: %Optional.15a = acquire_value %.loc44_33.8
// CHECK:STDOUT:     %Optional.HasValue.call: init bool = call %bound_method.loc44_33.5(%.loc44_33.10)
// CHECK:STDOUT:     %.loc44_33.11: bool = value_of_initializer %Optional.HasValue.call
// CHECK:STDOUT:     %.loc44_33.12: bool = converted %Optional.HasValue.call, %.loc44_33.11
// CHECK:STDOUT:     if %.loc44_33.12 br !for.body else br !for.done
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.body:
// CHECK:STDOUT:     %.loc44_33.13: %Optional.Get.type.d3e = specific_constant imports.%Core.import_ref.dab, @Optional(constants.%OptionalStorage.facet.c3f) [concrete = constants.%Optional.Get.57f]
// CHECK:STDOUT:     %Get.ref: %Optional.Get.type.d3e = name_ref Get, %.loc44_33.13 [concrete = constants.%Optional.Get.57f]
// CHECK:STDOUT:     %Optional.Get.bound: <bound method> = bound_method %.loc44_33.8, %Get.ref
// CHECK:STDOUT:     %Optional.Get.specific_fn: <specific function> = specific_function %Get.ref, @Optional.Get(constants.%OptionalStorage.facet.c3f) [concrete = constants.%Optional.Get.specific_fn.c49]
// CHECK:STDOUT:     %bound_method.loc44_33.6: <bound method> = bound_method %.loc44_33.8, %Optional.Get.specific_fn
// CHECK:STDOUT:     %.loc44_33.14: ref %DiffValueType = temporary_storage
// CHECK:STDOUT:     %.loc44_33.15: %Optional.15a = acquire_value %.loc44_33.8
// CHECK:STDOUT:     %Optional.Get.call: init %DiffValueType to %.loc44_33.14 = call %bound_method.loc44_33.6(%.loc44_33.15)
// CHECK:STDOUT:     %.loc44_33.16: ref %DiffValueType = temporary %.loc44_33.14, %Optional.Get.call
// CHECK:STDOUT:     %.loc44_33.17: %DiffValueType = acquire_value %.loc44_33.16
// CHECK:STDOUT:     %.loc44_14: type = splice_block %DiffValueType.ref.loc44 [concrete = constants.%DiffValueType] {
// CHECK:STDOUT:       %Cpp.ref.loc44: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:       %DiffValueType.ref.loc44: type = name_ref DiffValueType, imports.%DiffValueType.decl [concrete = constants.%DiffValueType]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %i: %DiffValueType = wrapper_binding i, %.loc44_33.17
// CHECK:STDOUT:     %sum.ref: ref %DiffValueType = name_ref sum, %sum
// CHECK:STDOUT:     %i.ref: %DiffValueType = name_ref i, %i
// CHECK:STDOUT:     %.loc45_12: ref %DiffValueType = value_as_ref %i.ref
// CHECK:STDOUT:     %addr.loc45: %ptr.cd0 = addr_of %.loc45_12
// CHECK:STDOUT:     %.loc45_9.1: %ptr.816 = as_compatible %addr.loc45
// CHECK:STDOUT:     %.loc45_9.2: %ptr.816 = converted %addr.loc45, %.loc45_9.1
// CHECK:STDOUT:     %operator_PlusEqual__carbon_thunk.call: ref %DiffValueType = call imports.%operator_PlusEqual__carbon_thunk.decl(%sum.ref, %.loc45_9.2)
// CHECK:STDOUT:     %DiffValueType.Op.bound.loc44: <bound method> = bound_method %.loc44_33.16, constants.%DiffValueType.Op.4dca31.1
// CHECK:STDOUT:     %Op.ref.loc44: %DiffValueType.cpp_destructor.type = name_ref Op, imports.%DiffValueType.cpp_destructor.decl [concrete = constants.%DiffValueType.cpp_destructor]
// CHECK:STDOUT:     %DiffValueType.cpp_destructor.bound.loc44: <bound method> = bound_method %.loc44_33.16, %Op.ref.loc44
// CHECK:STDOUT:     %DiffValueType.cpp_destructor.call.loc44: init %empty_tuple.type = call %DiffValueType.cpp_destructor.bound.loc44(%.loc44_33.16)
// CHECK:STDOUT:     %Destroy.Op.bound.loc44_33.1: <bound method> = bound_method %.loc44_33.8, constants.%Destroy.Op.1a2547.7
// CHECK:STDOUT:     %Destroy.Op.call.loc44_33.1: init %empty_tuple.type = call %Destroy.Op.bound.loc44_33.1(%.loc44_33.8)
// CHECK:STDOUT:     br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.done:
// CHECK:STDOUT:     %Destroy.Op.bound.loc44_33.2: <bound method> = bound_method %.loc44_33.8, constants.%Destroy.Op.1a2547.7
// CHECK:STDOUT:     %Destroy.Op.call.loc44_33.2: init %empty_tuple.type = call %Destroy.Op.bound.loc44_33.2(%.loc44_33.8)
// CHECK:STDOUT:     %impl.elem0.loc44_33.1: @TestIterate.%.loc44_33.22 (%.763) = impl_witness_access constants.%Destroy.lookup_impl_witness.38d, element0 [symbolic = %impl.elem0.loc44_33.2 (constants.%impl.elem0.891)]
// CHECK:STDOUT:     %bound_method.loc44_33.7: <bound method> = bound_method %var, %impl.elem0.loc44_33.1
// CHECK:STDOUT:     %.loc44_33.18: %Destroy.type = converted constants.%as_type.59f, constants.%impl.elem1.697 [symbolic = %impl.elem1 (constants.%impl.elem1.697)]
// CHECK:STDOUT:     %.loc44_33.19: %Destroy.type = converted constants.%as_type.59f, constants.%impl.elem1.697 [symbolic = %impl.elem1 (constants.%impl.elem1.697)]
// CHECK:STDOUT:     %specific_impl_fn.loc44_33.3: <specific function> = specific_impl_function %impl.elem0.loc44_33.1, @Destroy.WithSelf.Op(constants.%impl.elem1.697) [symbolic = %specific_impl_fn.loc44_33.6 (constants.%specific_impl_fn.b52)]
// CHECK:STDOUT:     %bound_method.loc44_33.8: <bound method> = bound_method %var, %specific_impl_fn.loc44_33.3
// CHECK:STDOUT:     %Destroy.WithSelf.Op.call: init %empty_tuple.type = call %bound_method.loc44_33.8(%var)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc44_33.1(%self.param: ref %.a18) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc44_33.2(%self.param: ref %MaybeUnformed.923) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc44_33.3(%self.param: ref bool) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc44_33.4(%self.param: ref %struct_type.value.has_value.3cf) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc44_33.5(%self.param: ref %DefaultOptionalStorage.5e4) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc44_33.6(%self.param: ref %Optional.15a) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestRangeFor(%m.param: ref %DiffMutableRange, %c.param: %DiffConstRange) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %i.patt.loc59: %pattern_type.149 = value_binding_pattern i [concrete = constants.%i.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %m.ref: ref %DiffMutableRange = name_ref m, %m
// CHECK:STDOUT:   %impl.elem2.loc59: %.cbf = impl_witness_access constants.%Iterate.impl_witness.747, element2 [concrete = constants.%T.as_type.as.Iterate.impl.NewCursor.136]
// CHECK:STDOUT:   %bound_method.loc59_33.1: <bound method> = bound_method %m.ref, %impl.elem2.loc59
// CHECK:STDOUT:   %facet_value.loc59_33.1: %CppRange_where.type.d682d6.1 = facet_value constants.%DiffMutableRange, (constants.%custom_witness.e7a31c.1, constants.%custom_witness.d0da43.1, constants.%custom_witness.aa8, constants.%custom_witness.9bb15e.1, constants.%custom_witness.392, constants.%custom_witness.1c9, constants.%custom_witness.723, constants.%custom_witness.9bb15e.2) [concrete = constants.%facet_value.0a7]
// CHECK:STDOUT:   %.loc59_33.1: %CppRange_where.type.d682d6.1 = converted constants.%DiffMutableRange, %facet_value.loc59_33.1 [concrete = constants.%facet_value.0a7]
// CHECK:STDOUT:   %facet_value.loc59_33.2: %CppRange_where.type.d682d6.1 = facet_value constants.%DiffMutableRange, (constants.%custom_witness.e7a31c.1, constants.%custom_witness.d0da43.1, constants.%custom_witness.aa8, constants.%custom_witness.9bb15e.1, constants.%custom_witness.392, constants.%custom_witness.1c9, constants.%custom_witness.723, constants.%custom_witness.9bb15e.2) [concrete = constants.%facet_value.0a7]
// CHECK:STDOUT:   %.loc59_33.2: %CppRange_where.type.d682d6.1 = converted constants.%DiffMutableRange, %facet_value.loc59_33.2 [concrete = constants.%facet_value.0a7]
// CHECK:STDOUT:   %specific_fn.loc59_33.1: <specific function> = specific_function %impl.elem2.loc59, @T.as_type.as.Iterate.impl.NewCursor(constants.%facet_value.0a7) [concrete = constants.%T.as_type.as.Iterate.impl.NewCursor.specific_fn.7b5]
// CHECK:STDOUT:   %bound_method.loc59_33.2: <bound method> = bound_method %m.ref, %specific_fn.loc59_33.1
// CHECK:STDOUT:   %var.loc59: ref %tuple.type.b18 = var_storage invalid
// CHECK:STDOUT:   %.loc59_32.1: %DiffMutableRange = acquire_value %m.ref
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.call.loc59: init %tuple.type.b18 to %var.loc59 = call %bound_method.loc59_33.2(%.loc59_32.1)
// CHECK:STDOUT:   assign %var.loc59, %T.as_type.as.Iterate.impl.NewCursor.call.loc59
// CHECK:STDOUT:   br !for.next.loc59
// CHECK:STDOUT:
// CHECK:STDOUT: !for.next.loc59:
// CHECK:STDOUT:   %addr.loc59: %ptr.3ce = addr_of %var.loc59
// CHECK:STDOUT:   %impl.elem3.loc59: %.922 = impl_witness_access constants.%Iterate.impl_witness.747, element3 [concrete = constants.%T.as_type.as.Iterate.impl.Next.31b]
// CHECK:STDOUT:   %bound_method.loc59_33.3: <bound method> = bound_method %m.ref, %impl.elem3.loc59
// CHECK:STDOUT:   %facet_value.loc59_33.3: %CppRange_where.type.d682d6.1 = facet_value constants.%DiffMutableRange, (constants.%custom_witness.e7a31c.1, constants.%custom_witness.d0da43.1, constants.%custom_witness.aa8, constants.%custom_witness.9bb15e.1, constants.%custom_witness.392, constants.%custom_witness.1c9, constants.%custom_witness.723, constants.%custom_witness.9bb15e.2) [concrete = constants.%facet_value.0a7]
// CHECK:STDOUT:   %.loc59_33.3: %CppRange_where.type.d682d6.1 = converted constants.%DiffMutableRange, %facet_value.loc59_33.3 [concrete = constants.%facet_value.0a7]
// CHECK:STDOUT:   %facet_value.loc59_33.4: %CppRange_where.type.d682d6.1 = facet_value constants.%DiffMutableRange, (constants.%custom_witness.e7a31c.1, constants.%custom_witness.d0da43.1, constants.%custom_witness.aa8, constants.%custom_witness.9bb15e.1, constants.%custom_witness.392, constants.%custom_witness.1c9, constants.%custom_witness.723, constants.%custom_witness.9bb15e.2) [concrete = constants.%facet_value.0a7]
// CHECK:STDOUT:   %.loc59_33.4: %CppRange_where.type.d682d6.1 = converted constants.%DiffMutableRange, %facet_value.loc59_33.4 [concrete = constants.%facet_value.0a7]
// CHECK:STDOUT:   %specific_fn.loc59_33.2: <specific function> = specific_function %impl.elem3.loc59, @T.as_type.as.Iterate.impl.Next(constants.%facet_value.0a7) [concrete = constants.%T.as_type.as.Iterate.impl.Next.specific_fn.a81]
// CHECK:STDOUT:   %bound_method.loc59_33.4: <bound method> = bound_method %m.ref, %specific_fn.loc59_33.2
// CHECK:STDOUT:   %.loc59_33.5: ref %Optional.15a = temporary_storage
// CHECK:STDOUT:   %.loc59_32.2: %DiffMutableRange = acquire_value %m.ref
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.call.loc59: init %Optional.15a to %.loc59_33.5 = call %bound_method.loc59_33.4(%.loc59_32.2, %addr.loc59)
// CHECK:STDOUT:   %.loc59_33.6: ref %Optional.15a = temporary %.loc59_33.5, %T.as_type.as.Iterate.impl.Next.call.loc59
// CHECK:STDOUT:   %.loc59_33.7: %Optional.HasValue.type.6f3 = specific_constant imports.%Core.import_ref.bb6, @Optional(constants.%OptionalStorage.facet.c3f) [concrete = constants.%Optional.HasValue.038]
// CHECK:STDOUT:   %HasValue.ref.loc59: %Optional.HasValue.type.6f3 = name_ref HasValue, %.loc59_33.7 [concrete = constants.%Optional.HasValue.038]
// CHECK:STDOUT:   %Optional.HasValue.bound.loc59: <bound method> = bound_method %.loc59_33.6, %HasValue.ref.loc59
// CHECK:STDOUT:   %Optional.HasValue.specific_fn.loc59: <specific function> = specific_function %HasValue.ref.loc59, @Optional.HasValue(constants.%OptionalStorage.facet.c3f) [concrete = constants.%Optional.HasValue.specific_fn.bc4]
// CHECK:STDOUT:   %bound_method.loc59_33.5: <bound method> = bound_method %.loc59_33.6, %Optional.HasValue.specific_fn.loc59
// CHECK:STDOUT:   %.loc59_33.8: %Optional.15a = acquire_value %.loc59_33.6
// CHECK:STDOUT:   %Optional.HasValue.call.loc59: init bool = call %bound_method.loc59_33.5(%.loc59_33.8)
// CHECK:STDOUT:   %.loc59_33.9: bool = value_of_initializer %Optional.HasValue.call.loc59
// CHECK:STDOUT:   %.loc59_33.10: bool = converted %Optional.HasValue.call.loc59, %.loc59_33.9
// CHECK:STDOUT:   if %.loc59_33.10 br !for.body.loc59 else br !for.done.loc59
// CHECK:STDOUT:
// CHECK:STDOUT: !for.body.loc59:
// CHECK:STDOUT:   %.loc59_33.11: %Optional.Get.type.d3e = specific_constant imports.%Core.import_ref.dab, @Optional(constants.%OptionalStorage.facet.c3f) [concrete = constants.%Optional.Get.57f]
// CHECK:STDOUT:   %Get.ref.loc59: %Optional.Get.type.d3e = name_ref Get, %.loc59_33.11 [concrete = constants.%Optional.Get.57f]
// CHECK:STDOUT:   %Optional.Get.bound.loc59: <bound method> = bound_method %.loc59_33.6, %Get.ref.loc59
// CHECK:STDOUT:   %Optional.Get.specific_fn.loc59: <specific function> = specific_function %Get.ref.loc59, @Optional.Get(constants.%OptionalStorage.facet.c3f) [concrete = constants.%Optional.Get.specific_fn.c49]
// CHECK:STDOUT:   %bound_method.loc59_33.6: <bound method> = bound_method %.loc59_33.6, %Optional.Get.specific_fn.loc59
// CHECK:STDOUT:   %.loc59_33.12: ref %DiffValueType = temporary_storage
// CHECK:STDOUT:   %.loc59_33.13: %Optional.15a = acquire_value %.loc59_33.6
// CHECK:STDOUT:   %Optional.Get.call.loc59: init %DiffValueType to %.loc59_33.12 = call %bound_method.loc59_33.6(%.loc59_33.13)
// CHECK:STDOUT:   %.loc59_33.14: ref %DiffValueType = temporary %.loc59_33.12, %Optional.Get.call.loc59
// CHECK:STDOUT:   %.loc59_33.15: %DiffValueType = acquire_value %.loc59_33.14
// CHECK:STDOUT:   %.loc59_14: type = splice_block %DiffValueType.ref.loc59 [concrete = constants.%DiffValueType] {
// CHECK:STDOUT:     %Cpp.ref.loc59: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %DiffValueType.ref.loc59: type = name_ref DiffValueType, imports.%DiffValueType.decl [concrete = constants.%DiffValueType]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %i.loc59: %DiffValueType = wrapper_binding i, %.loc59_33.15
// CHECK:STDOUT:   %sum.ref.loc60: ref %DiffValueType = name_ref sum, %sum
// CHECK:STDOUT:   %i.ref.loc60: %DiffValueType = name_ref i, %i.loc59
// CHECK:STDOUT:   %.loc60_12: ref %DiffValueType = value_as_ref %i.ref.loc60
// CHECK:STDOUT:   %addr.loc60: %ptr.cd0 = addr_of %.loc60_12
// CHECK:STDOUT:   %.loc60_9.1: %ptr.816 = as_compatible %addr.loc60
// CHECK:STDOUT:   %.loc60_9.2: %ptr.816 = converted %addr.loc60, %.loc60_9.1
// CHECK:STDOUT:   %operator_PlusEqual__carbon_thunk.call.loc60: ref %DiffValueType = call imports.%operator_PlusEqual__carbon_thunk.decl(%sum.ref.loc60, %.loc60_9.2)
// CHECK:STDOUT:   %DiffValueType.Op.bound.loc59: <bound method> = bound_method %.loc59_33.14, constants.%DiffValueType.Op.4dca31.1
// CHECK:STDOUT:   %Op.ref.loc59: %DiffValueType.cpp_destructor.type = name_ref Op, imports.%DiffValueType.cpp_destructor.decl [concrete = constants.%DiffValueType.cpp_destructor]
// CHECK:STDOUT:   %DiffValueType.cpp_destructor.bound.loc59: <bound method> = bound_method %.loc59_33.14, %Op.ref.loc59
// CHECK:STDOUT:   %DiffValueType.cpp_destructor.call.loc59: init %empty_tuple.type = call %DiffValueType.cpp_destructor.bound.loc59(%.loc59_33.14)
// CHECK:STDOUT:   %Destroy.Op.bound.loc59_33.1: <bound method> = bound_method %.loc59_33.6, constants.%Destroy.Op.1a2547.7
// CHECK:STDOUT:   %Destroy.Op.call.loc59_33.1: init %empty_tuple.type = call %Destroy.Op.bound.loc59_33.1(%.loc59_33.6)
// CHECK:STDOUT:   br !for.next.loc59
// CHECK:STDOUT:
// CHECK:STDOUT: !for.done.loc59:
// CHECK:STDOUT:   %Destroy.Op.bound.loc59_33.2: <bound method> = bound_method %.loc59_33.6, constants.%Destroy.Op.1a2547.7
// CHECK:STDOUT:   %Destroy.Op.call.loc59_33.2: init %empty_tuple.type = call %Destroy.Op.bound.loc59_33.2(%.loc59_33.6)
// CHECK:STDOUT:   %Destroy.Op.bound.loc59_33.3: <bound method> = bound_method %var.loc59, constants.%Destroy.Op.1a2547.8
// CHECK:STDOUT:   %Destroy.Op.call.loc59_33.3: init %empty_tuple.type = call %Destroy.Op.bound.loc59_33.3(%var.loc59)
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %i.patt.loc63: %pattern_type.149 = value_binding_pattern i [concrete = constants.%i.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c.ref: %DiffConstRange = name_ref c, %c
// CHECK:STDOUT:   %impl.elem2.loc63: %.f70 = impl_witness_access constants.%Iterate.impl_witness.f7f, element2 [concrete = constants.%T.as_type.as.Iterate.impl.NewCursor.1a9]
// CHECK:STDOUT:   %bound_method.loc63_33.1: <bound method> = bound_method %c.ref, %impl.elem2.loc63
// CHECK:STDOUT:   %facet_value.loc63_33.1: %CppRange_where.type.d682d6.1 = facet_value constants.%DiffConstRange, (constants.%custom_witness.e7a31c.1, constants.%custom_witness.d0da43.1, constants.%custom_witness.7b4, constants.%custom_witness.f7d4ff.1, constants.%custom_witness.0c0, constants.%custom_witness.ffb, constants.%custom_witness.a26, constants.%custom_witness.f7d4ff.2) [concrete = constants.%facet_value.733]
// CHECK:STDOUT:   %.loc63_33.1: %CppRange_where.type.d682d6.1 = converted constants.%DiffConstRange, %facet_value.loc63_33.1 [concrete = constants.%facet_value.733]
// CHECK:STDOUT:   %facet_value.loc63_33.2: %CppRange_where.type.d682d6.1 = facet_value constants.%DiffConstRange, (constants.%custom_witness.e7a31c.1, constants.%custom_witness.d0da43.1, constants.%custom_witness.7b4, constants.%custom_witness.f7d4ff.1, constants.%custom_witness.0c0, constants.%custom_witness.ffb, constants.%custom_witness.a26, constants.%custom_witness.f7d4ff.2) [concrete = constants.%facet_value.733]
// CHECK:STDOUT:   %.loc63_33.2: %CppRange_where.type.d682d6.1 = converted constants.%DiffConstRange, %facet_value.loc63_33.2 [concrete = constants.%facet_value.733]
// CHECK:STDOUT:   %specific_fn.loc63_33.1: <specific function> = specific_function %impl.elem2.loc63, @T.as_type.as.Iterate.impl.NewCursor(constants.%facet_value.733) [concrete = constants.%T.as_type.as.Iterate.impl.NewCursor.specific_fn.f9d]
// CHECK:STDOUT:   %bound_method.loc63_33.2: <bound method> = bound_method %c.ref, %specific_fn.loc63_33.1
// CHECK:STDOUT:   %var.loc63: ref %tuple.type.df5 = var_storage invalid
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.call.loc63: init %tuple.type.df5 to %var.loc63 = call %bound_method.loc63_33.2(%c.ref)
// CHECK:STDOUT:   assign %var.loc63, %T.as_type.as.Iterate.impl.NewCursor.call.loc63
// CHECK:STDOUT:   br !for.next.loc63
// CHECK:STDOUT:
// CHECK:STDOUT: !for.next.loc63:
// CHECK:STDOUT:   %addr.loc63: %ptr.310 = addr_of %var.loc63
// CHECK:STDOUT:   %impl.elem3.loc63: %.802 = impl_witness_access constants.%Iterate.impl_witness.f7f, element3 [concrete = constants.%T.as_type.as.Iterate.impl.Next.59a]
// CHECK:STDOUT:   %bound_method.loc63_33.3: <bound method> = bound_method %c.ref, %impl.elem3.loc63
// CHECK:STDOUT:   %facet_value.loc63_33.3: %CppRange_where.type.d682d6.1 = facet_value constants.%DiffConstRange, (constants.%custom_witness.e7a31c.1, constants.%custom_witness.d0da43.1, constants.%custom_witness.7b4, constants.%custom_witness.f7d4ff.1, constants.%custom_witness.0c0, constants.%custom_witness.ffb, constants.%custom_witness.a26, constants.%custom_witness.f7d4ff.2) [concrete = constants.%facet_value.733]
// CHECK:STDOUT:   %.loc63_33.3: %CppRange_where.type.d682d6.1 = converted constants.%DiffConstRange, %facet_value.loc63_33.3 [concrete = constants.%facet_value.733]
// CHECK:STDOUT:   %facet_value.loc63_33.4: %CppRange_where.type.d682d6.1 = facet_value constants.%DiffConstRange, (constants.%custom_witness.e7a31c.1, constants.%custom_witness.d0da43.1, constants.%custom_witness.7b4, constants.%custom_witness.f7d4ff.1, constants.%custom_witness.0c0, constants.%custom_witness.ffb, constants.%custom_witness.a26, constants.%custom_witness.f7d4ff.2) [concrete = constants.%facet_value.733]
// CHECK:STDOUT:   %.loc63_33.4: %CppRange_where.type.d682d6.1 = converted constants.%DiffConstRange, %facet_value.loc63_33.4 [concrete = constants.%facet_value.733]
// CHECK:STDOUT:   %specific_fn.loc63_33.2: <specific function> = specific_function %impl.elem3.loc63, @T.as_type.as.Iterate.impl.Next(constants.%facet_value.733) [concrete = constants.%T.as_type.as.Iterate.impl.Next.specific_fn.2ac]
// CHECK:STDOUT:   %bound_method.loc63_33.4: <bound method> = bound_method %c.ref, %specific_fn.loc63_33.2
// CHECK:STDOUT:   %.loc63_33.5: ref %Optional.15a = temporary_storage
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.call.loc63: init %Optional.15a to %.loc63_33.5 = call %bound_method.loc63_33.4(%c.ref, %addr.loc63)
// CHECK:STDOUT:   %.loc63_33.6: ref %Optional.15a = temporary %.loc63_33.5, %T.as_type.as.Iterate.impl.Next.call.loc63
// CHECK:STDOUT:   %.loc63_33.7: %Optional.HasValue.type.6f3 = specific_constant imports.%Core.import_ref.bb6, @Optional(constants.%OptionalStorage.facet.c3f) [concrete = constants.%Optional.HasValue.038]
// CHECK:STDOUT:   %HasValue.ref.loc63: %Optional.HasValue.type.6f3 = name_ref HasValue, %.loc63_33.7 [concrete = constants.%Optional.HasValue.038]
// CHECK:STDOUT:   %Optional.HasValue.bound.loc63: <bound method> = bound_method %.loc63_33.6, %HasValue.ref.loc63
// CHECK:STDOUT:   %Optional.HasValue.specific_fn.loc63: <specific function> = specific_function %HasValue.ref.loc63, @Optional.HasValue(constants.%OptionalStorage.facet.c3f) [concrete = constants.%Optional.HasValue.specific_fn.bc4]
// CHECK:STDOUT:   %bound_method.loc63_33.5: <bound method> = bound_method %.loc63_33.6, %Optional.HasValue.specific_fn.loc63
// CHECK:STDOUT:   %.loc63_33.8: %Optional.15a = acquire_value %.loc63_33.6
// CHECK:STDOUT:   %Optional.HasValue.call.loc63: init bool = call %bound_method.loc63_33.5(%.loc63_33.8)
// CHECK:STDOUT:   %.loc63_33.9: bool = value_of_initializer %Optional.HasValue.call.loc63
// CHECK:STDOUT:   %.loc63_33.10: bool = converted %Optional.HasValue.call.loc63, %.loc63_33.9
// CHECK:STDOUT:   if %.loc63_33.10 br !for.body.loc63 else br !for.done.loc63
// CHECK:STDOUT:
// CHECK:STDOUT: !for.body.loc63:
// CHECK:STDOUT:   %.loc63_33.11: %Optional.Get.type.d3e = specific_constant imports.%Core.import_ref.dab, @Optional(constants.%OptionalStorage.facet.c3f) [concrete = constants.%Optional.Get.57f]
// CHECK:STDOUT:   %Get.ref.loc63: %Optional.Get.type.d3e = name_ref Get, %.loc63_33.11 [concrete = constants.%Optional.Get.57f]
// CHECK:STDOUT:   %Optional.Get.bound.loc63: <bound method> = bound_method %.loc63_33.6, %Get.ref.loc63
// CHECK:STDOUT:   %Optional.Get.specific_fn.loc63: <specific function> = specific_function %Get.ref.loc63, @Optional.Get(constants.%OptionalStorage.facet.c3f) [concrete = constants.%Optional.Get.specific_fn.c49]
// CHECK:STDOUT:   %bound_method.loc63_33.6: <bound method> = bound_method %.loc63_33.6, %Optional.Get.specific_fn.loc63
// CHECK:STDOUT:   %.loc63_33.12: ref %DiffValueType = temporary_storage
// CHECK:STDOUT:   %.loc63_33.13: %Optional.15a = acquire_value %.loc63_33.6
// CHECK:STDOUT:   %Optional.Get.call.loc63: init %DiffValueType to %.loc63_33.12 = call %bound_method.loc63_33.6(%.loc63_33.13)
// CHECK:STDOUT:   %.loc63_33.14: ref %DiffValueType = temporary %.loc63_33.12, %Optional.Get.call.loc63
// CHECK:STDOUT:   %.loc63_33.15: %DiffValueType = acquire_value %.loc63_33.14
// CHECK:STDOUT:   %.loc63_14: type = splice_block %DiffValueType.ref.loc63 [concrete = constants.%DiffValueType] {
// CHECK:STDOUT:     %Cpp.ref.loc63: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %DiffValueType.ref.loc63: type = name_ref DiffValueType, imports.%DiffValueType.decl [concrete = constants.%DiffValueType]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %i.loc63: %DiffValueType = wrapper_binding i, %.loc63_33.15
// CHECK:STDOUT:   %sum.ref.loc64: ref %DiffValueType = name_ref sum, %sum
// CHECK:STDOUT:   %i.ref.loc64: %DiffValueType = name_ref i, %i.loc63
// CHECK:STDOUT:   %.loc64_12: ref %DiffValueType = value_as_ref %i.ref.loc64
// CHECK:STDOUT:   %addr.loc64: %ptr.cd0 = addr_of %.loc64_12
// CHECK:STDOUT:   %.loc64_9.1: %ptr.816 = as_compatible %addr.loc64
// CHECK:STDOUT:   %.loc64_9.2: %ptr.816 = converted %addr.loc64, %.loc64_9.1
// CHECK:STDOUT:   %operator_PlusEqual__carbon_thunk.call.loc64: ref %DiffValueType = call imports.%operator_PlusEqual__carbon_thunk.decl(%sum.ref.loc64, %.loc64_9.2)
// CHECK:STDOUT:   %DiffValueType.Op.bound.loc63: <bound method> = bound_method %.loc63_33.14, constants.%DiffValueType.Op.4dca31.1
// CHECK:STDOUT:   %Op.ref.loc63: %DiffValueType.cpp_destructor.type = name_ref Op, imports.%DiffValueType.cpp_destructor.decl [concrete = constants.%DiffValueType.cpp_destructor]
// CHECK:STDOUT:   %DiffValueType.cpp_destructor.bound.loc63: <bound method> = bound_method %.loc63_33.14, %Op.ref.loc63
// CHECK:STDOUT:   %DiffValueType.cpp_destructor.call.loc63: init %empty_tuple.type = call %DiffValueType.cpp_destructor.bound.loc63(%.loc63_33.14)
// CHECK:STDOUT:   %Destroy.Op.bound.loc63_33.1: <bound method> = bound_method %.loc63_33.6, constants.%Destroy.Op.1a2547.7
// CHECK:STDOUT:   %Destroy.Op.call.loc63_33.1: init %empty_tuple.type = call %Destroy.Op.bound.loc63_33.1(%.loc63_33.6)
// CHECK:STDOUT:   br !for.next.loc63
// CHECK:STDOUT:
// CHECK:STDOUT: !for.done.loc63:
// CHECK:STDOUT:   %Destroy.Op.bound.loc63_33.2: <bound method> = bound_method %.loc63_33.6, constants.%Destroy.Op.1a2547.7
// CHECK:STDOUT:   %Destroy.Op.call.loc63_33.2: init %empty_tuple.type = call %Destroy.Op.bound.loc63_33.2(%.loc63_33.6)
// CHECK:STDOUT:   %Destroy.Op.bound.loc63_33.3: <bound method> = bound_method %var.loc63, constants.%Destroy.Op.1a2547.9
// CHECK:STDOUT:   %Destroy.Op.call.loc63_33.3: init %empty_tuple.type = call %Destroy.Op.bound.loc63_33.3(%var.loc63)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestIterate(constants.%R) {
// CHECK:STDOUT:   %R.patt.loc40_17.2 => constants.%R.patt
// CHECK:STDOUT:   %R.loc40_17.1 => constants.%R
// CHECK:STDOUT:   %R.as_type.loc40_79.1 => constants.%R.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.20b
// CHECK:STDOUT:   %r.patt.loc40_77.2 => constants.%r.patt.9f3
// CHECK:STDOUT:   %r.param_patt.loc40_72.2 => constants.%r.param_patt.b12
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestIterate(constants.%facet_value.753) {
// CHECK:STDOUT:   %R.patt.loc40_17.2 => constants.%R.patt
// CHECK:STDOUT:   %R.loc40_17.1 => constants.%facet_value.753
// CHECK:STDOUT:   %R.as_type.loc40_79.1 => constants.%DiffMutableRange
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.611
// CHECK:STDOUT:   %r.patt.loc40_77.2 => constants.%r.patt.5fe
// CHECK:STDOUT:   %r.param_patt.loc40_72.2 => constants.%r.param_patt.a4b
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc40 => constants.%complete_type.357
// CHECK:STDOUT:   %Iterate.lookup_impl_witness => constants.%Iterate.impl_witness.747
// CHECK:STDOUT:   %Iterate.facet.loc44_33.5 => constants.%Iterate.facet.68a
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type => constants.%Iterate.WithSelf.NewCursor.type.e9c
// CHECK:STDOUT:   %.loc44_33.20 => constants.%.cbf
// CHECK:STDOUT:   %impl.elem2.loc44_33.2 => constants.%T.as_type.as.Iterate.impl.NewCursor.136
// CHECK:STDOUT:   %specific_impl_fn.loc44_33.4 => constants.%T.as_type.as.Iterate.impl.NewCursor.specific_fn.7b5
// CHECK:STDOUT:   %impl.elem1 => constants.%Destroy.facet.2c1
// CHECK:STDOUT:   %as_type => constants.%tuple.type.b18
// CHECK:STDOUT:   %require_complete.1 => constants.%complete_type.bfe
// CHECK:STDOUT:   %ptr => constants.%ptr.3ce
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type => constants.%Iterate.WithSelf.Next.type.a94
// CHECK:STDOUT:   %.loc44_33.21 => constants.%.922
// CHECK:STDOUT:   %impl.elem3.loc44_33.2 => constants.%T.as_type.as.Iterate.impl.Next.31b
// CHECK:STDOUT:   %specific_impl_fn.loc44_33.5 => constants.%T.as_type.as.Iterate.impl.Next.specific_fn.a81
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.188
// CHECK:STDOUT:   %.loc44_33.22 => constants.%.082
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.df9cc1.8
// CHECK:STDOUT:   %impl.elem0.loc44_33.2 => constants.%Destroy.Op.1a2547.8
// CHECK:STDOUT:   %specific_impl_fn.loc44_33.6 => constants.%Destroy.Op.1a2547.8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestIterate(constants.%facet_value.1e4) {
// CHECK:STDOUT:   %R.patt.loc40_17.2 => constants.%R.patt
// CHECK:STDOUT:   %R.loc40_17.1 => constants.%facet_value.1e4
// CHECK:STDOUT:   %R.as_type.loc40_79.1 => constants.%DiffConstRange
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.7dd
// CHECK:STDOUT:   %r.patt.loc40_77.2 => constants.%r.patt.9c0
// CHECK:STDOUT:   %r.param_patt.loc40_72.2 => constants.%r.param_patt.992
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc40 => constants.%complete_type.357
// CHECK:STDOUT:   %Iterate.lookup_impl_witness => constants.%Iterate.impl_witness.f7f
// CHECK:STDOUT:   %Iterate.facet.loc44_33.5 => constants.%Iterate.facet.608
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type => constants.%Iterate.WithSelf.NewCursor.type.185
// CHECK:STDOUT:   %.loc44_33.20 => constants.%.f70
// CHECK:STDOUT:   %impl.elem2.loc44_33.2 => constants.%T.as_type.as.Iterate.impl.NewCursor.1a9
// CHECK:STDOUT:   %specific_impl_fn.loc44_33.4 => constants.%T.as_type.as.Iterate.impl.NewCursor.specific_fn.f9d
// CHECK:STDOUT:   %impl.elem1 => constants.%Destroy.facet.1ed
// CHECK:STDOUT:   %as_type => constants.%tuple.type.df5
// CHECK:STDOUT:   %require_complete.1 => constants.%complete_type.ad6
// CHECK:STDOUT:   %ptr => constants.%ptr.310
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type => constants.%Iterate.WithSelf.Next.type.f38
// CHECK:STDOUT:   %.loc44_33.21 => constants.%.802
// CHECK:STDOUT:   %impl.elem3.loc44_33.2 => constants.%T.as_type.as.Iterate.impl.Next.59a
// CHECK:STDOUT:   %specific_impl_fn.loc44_33.5 => constants.%T.as_type.as.Iterate.impl.Next.specific_fn.2ac
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.a81
// CHECK:STDOUT:   %.loc44_33.22 => constants.%.471
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.df9cc1.9
// CHECK:STDOUT:   %impl.elem0.loc44_33.2 => constants.%Destroy.Op.1a2547.9
// CHECK:STDOUT:   %specific_impl_fn.loc44_33.6 => constants.%Destroy.Op.1a2547.9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- method_overloads.carbon
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
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Get.type.a12: type = fn_type @T.as_type.as.OptionalStorage.impl.Get, @T.as_type.as.OptionalStorage.impl(%T.283) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Get.d32: %T.as_type.as.OptionalStorage.impl.Get.type.a12 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Has.type.e5a: type = fn_type @T.as_type.as.OptionalStorage.impl.Has, @T.as_type.as.OptionalStorage.impl(%T.283) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Has.4ee: %T.as_type.as.OptionalStorage.impl.Has.type.e5a = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Some.type.492: type = fn_type @T.as_type.as.OptionalStorage.impl.Some, @T.as_type.as.OptionalStorage.impl(%T.283) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Some.89a: %T.as_type.as.OptionalStorage.impl.Some.type.492 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.None.type.0a1: type = fn_type @T.as_type.as.OptionalStorage.impl.None, @T.as_type.as.OptionalStorage.impl(%T.283) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.None.e06: %T.as_type.as.OptionalStorage.impl.None.type.0a1 = struct_value () [symbolic]
// CHECK:STDOUT:   %.Self.500: %Iterate.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %CppRange.type: type = facet_type <@CppRange> [concrete]
// CHECK:STDOUT:   %.Self.e0b: %CppRange.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %CppRangeForIterate.lookup_impl_witness.50b: <witness> = lookup_impl_witness %.Self.e0b, @CppRangeForIterate [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.32d: type = impl_witness_access %CppRangeForIterate.lookup_impl_witness.50b, element0 [symbolic_self]
// CHECK:STDOUT:   %CppUnsafeDeref.lookup_impl_witness.851: <witness> = lookup_impl_witness %impl.elem0.32d, @CppUnsafeDeref [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.10d: type = impl_witness_access %CppUnsafeDeref.lookup_impl_witness.851, element0 [symbolic_self]
// CHECK:STDOUT:   %CppRange_where.type.d682d6.1: type = facet_type <@CppRange where %impl.elem0.10d impls @Copy> [concrete]
// CHECK:STDOUT:   %T.f45530.2: %CppRange_where.type.d682d6.1 = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.type.b36: type = fn_type @T.as_type.as.Iterate.impl.Next, @T.as_type.as.Iterate.impl(%T.f45530.2) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.d6b: %T.as_type.as.Iterate.impl.Next.type.b36 = struct_value () [symbolic]
// CHECK:STDOUT:   %CppRangeForIterate.lookup_impl_witness.b581a8.2: <witness> = lookup_impl_witness %T.f45530.2, @CppRangeForIterate [symbolic]
// CHECK:STDOUT:   %impl.elem0.5f971c.2: type = impl_witness_access %CppRangeForIterate.lookup_impl_witness.b581a8.2, element0 [symbolic]
// CHECK:STDOUT:   %CppUnsafeDeref.lookup_impl_witness.87e543.2: <witness> = lookup_impl_witness %impl.elem0.5f971c.2, @CppUnsafeDeref [symbolic]
// CHECK:STDOUT:   %impl.elem0.493765.2: type = impl_witness_access %CppUnsafeDeref.lookup_impl_witness.87e543.2, element0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.type.953: type = fn_type @T.as_type.as.Iterate.impl.NewCursor, @T.as_type.as.Iterate.impl(%T.f45530.2) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.544: %T.as_type.as.Iterate.impl.NewCursor.type.953 = struct_value () [symbolic]
// CHECK:STDOUT:   %impl.elem1.c2dc15.2: type = impl_witness_access %CppRangeForIterate.lookup_impl_witness.b581a8.2, element1 [symbolic]
// CHECK:STDOUT:   %tuple.type.1ba: type = tuple_type (%impl.elem0.5f971c.2, %impl.elem1.c2dc15.2) [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.812: <witness> = lookup_impl_witness %tuple.type.1ba, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.9cd: %Destroy.type = facet_value %tuple.type.1ba, (%Destroy.lookup_impl_witness.812) [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.35816c.2: <witness> = lookup_impl_witness %impl.elem0.493765.2, @Destroy [symbolic]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.167: <witness> = lookup_impl_witness %impl.elem0.493765.2, @Copy [symbolic]
// CHECK:STDOUT:   %facet_value.f93: %facet_type.f48 = facet_value %impl.elem0.493765.2, (%Destroy.lookup_impl_witness.35816c.2, %Copy.lookup_impl_witness.167) [symbolic]
// CHECK:STDOUT:   %OverloadValueType: type = class_type @OverloadValueType [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %pattern_type.804: type = pattern_type %OverloadValueType [concrete]
// CHECK:STDOUT:   %OverloadValueType.cpp_destructor.type: type = fn_type @OverloadValueType.cpp_destructor [concrete]
// CHECK:STDOUT:   %OverloadValueType.cpp_destructor: %OverloadValueType.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT:   %OverloadValueType.Op.type.83af6e.1: type = fn_type @OverloadValueType.Op.1 [concrete]
// CHECK:STDOUT:   %OverloadValueType.Op.4dca31.1: %OverloadValueType.Op.type.83af6e.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.e7a31c.1: <witness> = custom_witness (%OverloadValueType.Op.4dca31.1), @Destroy [concrete]
// CHECK:STDOUT:   %OverloadValueType.Op.type.83af6e.2: type = fn_type @OverloadValueType.Op.2 [concrete]
// CHECK:STDOUT:   %OverloadValueType.Op.4dca31.2: %OverloadValueType.Op.type.83af6e.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.d0da43.1: <witness> = custom_witness (%OverloadValueType.Op.4dca31.2), @Copy [concrete]
// CHECK:STDOUT:   %facet_value.437: %facet_type.f48 = facet_value %OverloadValueType, (%custom_witness.e7a31c.1, %custom_witness.d0da43.1) [concrete]
// CHECK:STDOUT:   %Iterate.lookup_impl_witness.1d8: <witness> = lookup_impl_witness %.Self.500, @Iterate [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.294: %facet_type.f48 = impl_witness_access %Iterate.lookup_impl_witness.1d8, element0 [symbolic_self]
// CHECK:STDOUT:   %Iterate_where.type.494: type = facet_type <@Iterate where %impl.elem0.294 = %facet_value.437> [concrete]
// CHECK:STDOUT:   %pattern_type.aac: type = pattern_type %Iterate_where.type.494 [concrete]
// CHECK:STDOUT:   %R.patt: %pattern_type.aac = symbolic_binding_pattern R, 0 [symbolic]
// CHECK:STDOUT:   %R: %Iterate_where.type.494 = symbolic_binding R, 0 [symbolic]
// CHECK:STDOUT:   %R.as_type: type = facet_access_type %R [symbolic]
// CHECK:STDOUT:   %pattern_type.def: type = pattern_type %R.as_type [symbolic]
// CHECK:STDOUT:   %r.patt.bb3: %pattern_type.def = ref_binding_pattern r [symbolic]
// CHECK:STDOUT:   %r.param_patt.9b2: %pattern_type.def = var_param_pattern %r.patt.bb3 [symbolic]
// CHECK:STDOUT:   %_.patt.a10: %pattern_type.804 = value_binding_pattern _ [concrete]
// CHECK:STDOUT:   %Iterate.lookup_impl_witness.eca: <witness> = lookup_impl_witness %R, @Iterate [symbolic]
// CHECK:STDOUT:   %Iterate.facet.bb8: %Iterate.type = facet_value %R.as_type, (%Iterate.lookup_impl_witness.eca) [symbolic]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.c6e: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.bb8) [symbolic]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.76f: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.bb8) [symbolic]
// CHECK:STDOUT:   %.426: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.c6e, %Iterate.facet.bb8 [symbolic]
// CHECK:STDOUT:   %impl.elem2.82f: %.426 = impl_witness_access %Iterate.lookup_impl_witness.eca, element2 [symbolic]
// CHECK:STDOUT:   %impl.elem1.424: %Destroy.type = impl_witness_access %Iterate.lookup_impl_witness.eca, element1 [symbolic]
// CHECK:STDOUT:   %as_type.810: type = facet_access_type %impl.elem1.424 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.5f9: <specific function> = specific_impl_function %impl.elem2.82f, @Iterate.WithSelf.NewCursor(%Iterate.facet.bb8) [symbolic]
// CHECK:STDOUT:   %ptr.a41: type = ptr_type %as_type.810 [symbolic]
// CHECK:STDOUT:   %.474: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.76f, %Iterate.facet.bb8 [symbolic]
// CHECK:STDOUT:   %impl.elem3.316: %.474 = impl_witness_access %Iterate.lookup_impl_witness.eca, element3 [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.799: type = class_type @DefaultOptionalStorage, @DefaultOptionalStorage(%facet_value.437) [concrete]
// CHECK:STDOUT:   %OptionalStorage.impl_witness.1e9: <witness> = impl_witness imports.%OptionalStorage.impl_witness_table.460, @T.as_type.as.OptionalStorage.impl(%facet_value.437) [concrete]
// CHECK:STDOUT:   %OptionalStorage.facet.496: %OptionalStorage.type = facet_value %OverloadValueType, (%OptionalStorage.impl_witness.1e9) [concrete]
// CHECK:STDOUT:   %Optional.4e9: type = class_type @Optional, @Optional(%OptionalStorage.facet.496) [concrete]
// CHECK:STDOUT:   %specific_impl_fn.754: <specific function> = specific_impl_function %impl.elem3.316, @Iterate.WithSelf.Next(%Iterate.facet.bb8) [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.type.59f: type = fn_type @Optional.HasValue, @Optional(%OptionalStorage.facet.496) [concrete]
// CHECK:STDOUT:   %Optional.HasValue.91a: %Optional.HasValue.type.59f = struct_value () [concrete]
// CHECK:STDOUT:   %Optional.Get.type.cab: type = fn_type @Optional.Get, @Optional(%OptionalStorage.facet.496) [concrete]
// CHECK:STDOUT:   %Optional.Get.2d3: %Optional.Get.type.cab = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.facet.7ac: %Destroy.type = facet_value %OverloadValueType, (%custom_witness.e7a31c.1) [concrete]
// CHECK:STDOUT:   %MaybeUnformed.59c: type = class_type @MaybeUnformed, @MaybeUnformed(%Destroy.facet.7ac) [concrete]
// CHECK:STDOUT:   %.8a2: type = maybe_unformed_type %OverloadValueType [concrete]
// CHECK:STDOUT:   %struct_type.value.has_value.3e8: type = struct_type {.value: %MaybeUnformed.59c, .has_value: bool} [concrete]
// CHECK:STDOUT:   %Optional.HasValue.specific_fn.bf1: <specific function> = specific_function %Optional.HasValue.91a, @Optional.HasValue(%OptionalStorage.facet.496) [concrete]
// CHECK:STDOUT:   %Optional.Get.specific_fn.548: <specific function> = specific_function %Optional.Get.2d3, @Optional.Get(%OptionalStorage.facet.496) [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.7: type = fn_type @Destroy.Op.loc27_37.6 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.7: %Destroy.Op.type.1d8f74.7 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.d2d: <witness> = lookup_impl_witness %impl.elem1.424, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.9de: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%impl.elem1.424) [symbolic]
// CHECK:STDOUT:   %.696: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.9de, %impl.elem1.424 [symbolic]
// CHECK:STDOUT:   %impl.elem0.18a: %.696 = impl_witness_access %Destroy.lookup_impl_witness.d2d, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.b04: <specific function> = specific_impl_function %impl.elem0.18a, @Destroy.WithSelf.Op(%impl.elem1.424) [symbolic]
// CHECK:STDOUT:   %Range: type = class_type @Range [concrete]
// CHECK:STDOUT:   %pattern_type.23f: type = pattern_type %Range [concrete]
// CHECK:STDOUT:   %Iterator: type = class_type @Iterator.1 [concrete]
// CHECK:STDOUT:   %Range.Begin.type: type = fn_type @Range.Begin [concrete]
// CHECK:STDOUT:   %Range.Begin: %Range.Begin.type = struct_value () [concrete]
// CHECK:STDOUT:   %Range.End.type: type = fn_type @Range.End [concrete]
// CHECK:STDOUT:   %Range.End: %Range.End.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.090: <witness> = custom_witness (%Iterator, %Iterator, %Range.Begin, %Range.End), @CppRangeForIterate [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.4232e0.1: type = fn_type @Iterator.Op.1 [concrete]
// CHECK:STDOUT:   %Iterator.Op.e8caec.1: %Iterator.Op.type.4232e0.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.6bd: <witness> = custom_witness (%OverloadValueType, %Iterator.Op.e8caec.1), @CppUnsafeDeref [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.4232e0.2: type = fn_type @Iterator.Op.2 [concrete]
// CHECK:STDOUT:   %Iterator.Op.e8caec.2: %Iterator.Op.type.4232e0.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.f4a: <witness> = custom_witness (%Iterator.Op.e8caec.2), @Destroy [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.4232e0.3: type = fn_type @Iterator.Op.3 [concrete]
// CHECK:STDOUT:   %Iterator.Op.e8caec.3: %Iterator.Op.type.4232e0.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.716: <witness> = custom_witness (%Iterator.Op.e8caec.3), @Inc [concrete]
// CHECK:STDOUT:   %Equal.type: type = fn_type @Equal [concrete]
// CHECK:STDOUT:   %Equal: %Equal.type = struct_value () [concrete]
// CHECK:STDOUT:   %NotEqual.type: type = fn_type @NotEqual [concrete]
// CHECK:STDOUT:   %NotEqual: %NotEqual.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.617: <witness> = custom_witness (%Equal, %NotEqual), @EqWith, @EqWith(%Iterator) [concrete]
// CHECK:STDOUT:   %facet_value.cbe: %CppRange_where.type.d682d6.1 = facet_value %Range, (%custom_witness.e7a31c.1, %custom_witness.d0da43.1, %custom_witness.090, %custom_witness.f4a, %custom_witness.6bd, %custom_witness.716, %custom_witness.617) [concrete]
// CHECK:STDOUT:   %tuple.type.9ef: type = tuple_type (%Iterator, %Iterator) [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.8: type = fn_type @Destroy.Op.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.8: %Destroy.Op.type.1d8f74.8 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.df9cc1.8: <witness> = custom_witness (%Destroy.Op.1a2547.8), @Destroy [concrete]
// CHECK:STDOUT:   %Destroy.facet.15b: %Destroy.type = facet_value %tuple.type.9ef, (%custom_witness.df9cc1.8) [concrete]
// CHECK:STDOUT:   %Iterate.impl_witness.175: <witness> = impl_witness imports.%Iterate.impl_witness_table.121, @T.as_type.as.Iterate.impl(%facet_value.cbe) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.type.72c: type = fn_type @T.as_type.as.Iterate.impl.NewCursor, @T.as_type.as.Iterate.impl(%facet_value.cbe) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.036: %T.as_type.as.Iterate.impl.NewCursor.type.72c = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.type.fb2: type = fn_type @T.as_type.as.Iterate.impl.Next, @T.as_type.as.Iterate.impl(%facet_value.cbe) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.751: %T.as_type.as.Iterate.impl.Next.type.fb2 = struct_value () [concrete]
// CHECK:STDOUT:   %Iterate.facet.cc4: %Iterate.type = facet_value %Range, (%Iterate.impl_witness.175) [concrete]
// CHECK:STDOUT:   %facet_value.3af: %Iterate_where.type.494 = facet_value %Range, (%Iterate.impl_witness.175) [concrete]
// CHECK:STDOUT:   %r.patt.b9f: %pattern_type.23f = ref_binding_pattern r [concrete]
// CHECK:STDOUT:   %r.param_patt.466: %pattern_type.23f = var_param_pattern %r.patt.b9f [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.a44: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.cc4) [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.1f9: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.cc4) [concrete]
// CHECK:STDOUT:   %.687: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.a44, %Iterate.facet.cc4 [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.specific_fn: <specific function> = specific_function %T.as_type.as.Iterate.impl.NewCursor.036, @T.as_type.as.Iterate.impl.NewCursor(%facet_value.cbe) [concrete]
// CHECK:STDOUT:   %ptr.b18: type = ptr_type %tuple.type.9ef [concrete]
// CHECK:STDOUT:   %.79c: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.1f9, %Iterate.facet.cc4 [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.specific_fn: <specific function> = specific_function %T.as_type.as.Iterate.impl.Next.751, @T.as_type.as.Iterate.impl.Next(%facet_value.cbe) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.fd5: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.15b) [concrete]
// CHECK:STDOUT:   %.fb6: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.fd5, %Destroy.facet.15b [concrete]
// CHECK:STDOUT:   %complete_type.e35: <witness> = complete_type_witness %tuple.type.9ef [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .OverloadValueType = %OverloadValueType.decl
// CHECK:STDOUT:     .Range = %Range.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.bb6: @Optional.%Optional.HasValue.type (%Optional.HasValue.type.e9e) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.HasValue (constants.%Optional.HasValue.6cd)]
// CHECK:STDOUT:   %Core.import_ref.dab: @Optional.%Optional.Get.type (%Optional.Get.type.bb9) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.Get (constants.%Optional.Get.fa1)]
// CHECK:STDOUT:   %Core.import_ref.0c6: type = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%DefaultOptionalStorage (constants.%DefaultOptionalStorage.d50)]
// CHECK:STDOUT:   %Core.import_ref.dc3: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.None.type (%T.as_type.as.OptionalStorage.impl.None.type.0a1) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.None (constants.%T.as_type.as.OptionalStorage.impl.None.e06)]
// CHECK:STDOUT:   %Core.import_ref.c53: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Some.type (%T.as_type.as.OptionalStorage.impl.Some.type.492) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Some (constants.%T.as_type.as.OptionalStorage.impl.Some.89a)]
// CHECK:STDOUT:   %Core.import_ref.ed4: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Has.type (%T.as_type.as.OptionalStorage.impl.Has.type.e5a) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Has (constants.%T.as_type.as.OptionalStorage.impl.Has.4ee)]
// CHECK:STDOUT:   %Core.import_ref.e9c: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Get.type (%T.as_type.as.OptionalStorage.impl.Get.type.a12) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Get (constants.%T.as_type.as.OptionalStorage.impl.Get.d32)]
// CHECK:STDOUT:   %Core.import_ref.0b2 = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], unloaded
// CHECK:STDOUT:   %OptionalStorage.impl_witness_table.460 = impl_witness_table (%Core.import_ref.0c6, %Core.import_ref.dc3, %Core.import_ref.c53, %Core.import_ref.ed4, %Core.import_ref.e9c, %Core.import_ref.0b2), @T.as_type.as.OptionalStorage.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.eef: %facet_type.f48 = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.Iterate.impl.%facet_value (constants.%facet_value.f93)]
// CHECK:STDOUT:   %Core.import_ref.fcc: %Destroy.type = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.Iterate.impl.%Destroy.facet (constants.%Destroy.facet.9cd)]
// CHECK:STDOUT:   %Core.import_ref.fd1: @T.as_type.as.Iterate.impl.%T.as_type.as.Iterate.impl.NewCursor.type (%T.as_type.as.Iterate.impl.NewCursor.type.953) = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.Iterate.impl.%T.as_type.as.Iterate.impl.NewCursor (constants.%T.as_type.as.Iterate.impl.NewCursor.544)]
// CHECK:STDOUT:   %Core.import_ref.273: @T.as_type.as.Iterate.impl.%T.as_type.as.Iterate.impl.Next.type (%T.as_type.as.Iterate.impl.Next.type.b36) = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.Iterate.impl.%T.as_type.as.Iterate.impl.Next (constants.%T.as_type.as.Iterate.impl.Next.d6b)]
// CHECK:STDOUT:   %Iterate.impl_witness_table.121 = impl_witness_table (%Core.import_ref.eef, %Core.import_ref.fcc, %Core.import_ref.fd1, %Core.import_ref.273), @T.as_type.as.Iterate.impl [concrete]
// CHECK:STDOUT:   %OverloadValueType.decl: type = class_decl @OverloadValueType [concrete = constants.%OverloadValueType] {} {}
// CHECK:STDOUT:   %OverloadValueType.cpp_destructor.decl: %OverloadValueType.cpp_destructor.type = fn_decl @OverloadValueType.cpp_destructor [concrete = constants.%OverloadValueType.cpp_destructor] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Range.decl: type = class_decl @Range [concrete = constants.%Range] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @TestIterate(%R.loc25_17.2: %Iterate_where.type.494) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Iterate.lookup_impl_witness: <witness> = lookup_impl_witness %R.loc25_17.1, @Iterate [symbolic = %Iterate.lookup_impl_witness (constants.%Iterate.lookup_impl_witness.eca)]
// CHECK:STDOUT:   %Iterate.facet.loc27_37.5: %Iterate.type = facet_value %R.as_type.loc25_83.1, (%Iterate.lookup_impl_witness) [symbolic = %Iterate.facet.loc27_37.5 (constants.%Iterate.facet.bb8)]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.loc27_37.5) [symbolic = %Iterate.WithSelf.NewCursor.type (constants.%Iterate.WithSelf.NewCursor.type.c6e)]
// CHECK:STDOUT:   %.loc27_37.20: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type, %Iterate.facet.loc27_37.5 [symbolic = %.loc27_37.20 (constants.%.426)]
// CHECK:STDOUT:   %impl.elem2.loc27_37.2: @TestIterate.%.loc27_37.20 (%.426) = impl_witness_access %Iterate.lookup_impl_witness, element2 [symbolic = %impl.elem2.loc27_37.2 (constants.%impl.elem2.82f)]
// CHECK:STDOUT:   %specific_impl_fn.loc27_37.4: <specific function> = specific_impl_function %impl.elem2.loc27_37.2, @Iterate.WithSelf.NewCursor(%Iterate.facet.loc27_37.5) [symbolic = %specific_impl_fn.loc27_37.4 (constants.%specific_impl_fn.5f9)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %ptr: type = ptr_type %as_type [symbolic = %ptr (constants.%ptr.a41)]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.loc27_37.5) [symbolic = %Iterate.WithSelf.Next.type (constants.%Iterate.WithSelf.Next.type.76f)]
// CHECK:STDOUT:   %.loc27_37.21: type = fn_type_with_self_type %Iterate.WithSelf.Next.type, %Iterate.facet.loc27_37.5 [symbolic = %.loc27_37.21 (constants.%.474)]
// CHECK:STDOUT:   %impl.elem3.loc27_37.2: @TestIterate.%.loc27_37.21 (%.474) = impl_witness_access %Iterate.lookup_impl_witness, element3 [symbolic = %impl.elem3.loc27_37.2 (constants.%impl.elem3.316)]
// CHECK:STDOUT:   %specific_impl_fn.loc27_37.5: <specific function> = specific_impl_function %impl.elem3.loc27_37.2, @Iterate.WithSelf.Next(%Iterate.facet.loc27_37.5) [symbolic = %specific_impl_fn.loc27_37.5 (constants.%specific_impl_fn.754)]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%impl.elem1) [symbolic = %Destroy.WithSelf.Op.type (constants.%Destroy.WithSelf.Op.type.9de)]
// CHECK:STDOUT:   %.loc27_37.22: type = fn_type_with_self_type %Destroy.WithSelf.Op.type, %impl.elem1 [symbolic = %.loc27_37.22 (constants.%.696)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %impl.elem1, @Destroy [symbolic = %Destroy.lookup_impl_witness (constants.%Destroy.lookup_impl_witness.d2d)]
// CHECK:STDOUT:   %impl.elem0.loc27_37.2: @TestIterate.%.loc27_37.22 (%.696) = impl_witness_access %Destroy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc27_37.2 (constants.%impl.elem0.18a)]
// CHECK:STDOUT:   %specific_impl_fn.loc27_37.6: <specific function> = specific_impl_function %impl.elem0.loc27_37.2, @Destroy.WithSelf.Op(%impl.elem1) [symbolic = %specific_impl_fn.loc27_37.6 (constants.%specific_impl_fn.b04)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%r.param: ref @TestIterate.%R.as_type.loc25_83.1 (%R.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %_.patt: %pattern_type.804 = value_binding_pattern _ [concrete = constants.%_.patt.a10]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %r.ref: ref @TestIterate.%R.as_type.loc25_83.1 (%R.as_type) = name_ref r, %r
// CHECK:STDOUT:     %impl.elem2.loc27_37.1: @TestIterate.%.loc27_37.20 (%.426) = impl_witness_access constants.%Iterate.lookup_impl_witness.eca, element2 [symbolic = %impl.elem2.loc27_37.2 (constants.%impl.elem2.82f)]
// CHECK:STDOUT:     %bound_method.loc27_37.1: <bound method> = bound_method %r.ref, %impl.elem2.loc27_37.1
// CHECK:STDOUT:     %Iterate.facet.loc27_37.1: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.eca) [symbolic = %Iterate.facet.loc27_37.5 (constants.%Iterate.facet.bb8)]
// CHECK:STDOUT:     %.loc27_37.1: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc27_37.1 [symbolic = %Iterate.facet.loc27_37.5 (constants.%Iterate.facet.bb8)]
// CHECK:STDOUT:     %Iterate.facet.loc27_37.2: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.eca) [symbolic = %Iterate.facet.loc27_37.5 (constants.%Iterate.facet.bb8)]
// CHECK:STDOUT:     %.loc27_37.2: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc27_37.2 [symbolic = %Iterate.facet.loc27_37.5 (constants.%Iterate.facet.bb8)]
// CHECK:STDOUT:     %specific_impl_fn.loc27_37.1: <specific function> = specific_impl_function %impl.elem2.loc27_37.1, @Iterate.WithSelf.NewCursor(constants.%Iterate.facet.bb8) [symbolic = %specific_impl_fn.loc27_37.4 (constants.%specific_impl_fn.5f9)]
// CHECK:STDOUT:     %bound_method.loc27_37.2: <bound method> = bound_method %r.ref, %specific_impl_fn.loc27_37.1
// CHECK:STDOUT:     %var: ref @TestIterate.%as_type (%as_type.810) = var_storage invalid
// CHECK:STDOUT:     %.loc27_36.1: @TestIterate.%R.as_type.loc25_83.1 (%R.as_type) = acquire_value %r.ref
// CHECK:STDOUT:     %Iterate.WithSelf.NewCursor.call: init @TestIterate.%as_type (%as_type.810) to %var = call %bound_method.loc27_37.2(%.loc27_36.1)
// CHECK:STDOUT:     assign %var, %Iterate.WithSelf.NewCursor.call
// CHECK:STDOUT:     br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.next:
// CHECK:STDOUT:     %addr: @TestIterate.%ptr (%ptr.a41) = addr_of %var
// CHECK:STDOUT:     %impl.elem3.loc27_37.1: @TestIterate.%.loc27_37.21 (%.474) = impl_witness_access constants.%Iterate.lookup_impl_witness.eca, element3 [symbolic = %impl.elem3.loc27_37.2 (constants.%impl.elem3.316)]
// CHECK:STDOUT:     %bound_method.loc27_37.3: <bound method> = bound_method %r.ref, %impl.elem3.loc27_37.1
// CHECK:STDOUT:     %Iterate.facet.loc27_37.3: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.eca) [symbolic = %Iterate.facet.loc27_37.5 (constants.%Iterate.facet.bb8)]
// CHECK:STDOUT:     %.loc27_37.3: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc27_37.3 [symbolic = %Iterate.facet.loc27_37.5 (constants.%Iterate.facet.bb8)]
// CHECK:STDOUT:     %Iterate.facet.loc27_37.4: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.eca) [symbolic = %Iterate.facet.loc27_37.5 (constants.%Iterate.facet.bb8)]
// CHECK:STDOUT:     %.loc27_37.4: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc27_37.4 [symbolic = %Iterate.facet.loc27_37.5 (constants.%Iterate.facet.bb8)]
// CHECK:STDOUT:     %.loc27_37.5: %Destroy.type = converted constants.%as_type.810, constants.%impl.elem1.424 [symbolic = %impl.elem1 (constants.%impl.elem1.424)]
// CHECK:STDOUT:     %.loc27_37.6: %Destroy.type = converted constants.%as_type.810, constants.%impl.elem1.424 [symbolic = %impl.elem1 (constants.%impl.elem1.424)]
// CHECK:STDOUT:     %specific_impl_fn.loc27_37.2: <specific function> = specific_impl_function %impl.elem3.loc27_37.1, @Iterate.WithSelf.Next(constants.%Iterate.facet.bb8) [symbolic = %specific_impl_fn.loc27_37.5 (constants.%specific_impl_fn.754)]
// CHECK:STDOUT:     %bound_method.loc27_37.4: <bound method> = bound_method %r.ref, %specific_impl_fn.loc27_37.2
// CHECK:STDOUT:     %.loc27_37.7: ref %Optional.4e9 = temporary_storage
// CHECK:STDOUT:     %.loc27_36.2: @TestIterate.%R.as_type.loc25_83.1 (%R.as_type) = acquire_value %r.ref
// CHECK:STDOUT:     %Iterate.WithSelf.Next.call: init %Optional.4e9 to %.loc27_37.7 = call %bound_method.loc27_37.4(%.loc27_36.2, %addr)
// CHECK:STDOUT:     %.loc27_37.8: ref %Optional.4e9 = temporary %.loc27_37.7, %Iterate.WithSelf.Next.call
// CHECK:STDOUT:     %.loc27_37.9: %Optional.HasValue.type.59f = specific_constant imports.%Core.import_ref.bb6, @Optional(constants.%OptionalStorage.facet.496) [concrete = constants.%Optional.HasValue.91a]
// CHECK:STDOUT:     %HasValue.ref: %Optional.HasValue.type.59f = name_ref HasValue, %.loc27_37.9 [concrete = constants.%Optional.HasValue.91a]
// CHECK:STDOUT:     %Optional.HasValue.bound: <bound method> = bound_method %.loc27_37.8, %HasValue.ref
// CHECK:STDOUT:     %Optional.HasValue.specific_fn: <specific function> = specific_function %HasValue.ref, @Optional.HasValue(constants.%OptionalStorage.facet.496) [concrete = constants.%Optional.HasValue.specific_fn.bf1]
// CHECK:STDOUT:     %bound_method.loc27_37.5: <bound method> = bound_method %.loc27_37.8, %Optional.HasValue.specific_fn
// CHECK:STDOUT:     %.loc27_37.10: %Optional.4e9 = acquire_value %.loc27_37.8
// CHECK:STDOUT:     %Optional.HasValue.call: init bool = call %bound_method.loc27_37.5(%.loc27_37.10)
// CHECK:STDOUT:     %.loc27_37.11: bool = value_of_initializer %Optional.HasValue.call
// CHECK:STDOUT:     %.loc27_37.12: bool = converted %Optional.HasValue.call, %.loc27_37.11
// CHECK:STDOUT:     if %.loc27_37.12 br !for.body else br !for.done
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.body:
// CHECK:STDOUT:     %.loc27_37.13: %Optional.Get.type.cab = specific_constant imports.%Core.import_ref.dab, @Optional(constants.%OptionalStorage.facet.496) [concrete = constants.%Optional.Get.2d3]
// CHECK:STDOUT:     %Get.ref: %Optional.Get.type.cab = name_ref Get, %.loc27_37.13 [concrete = constants.%Optional.Get.2d3]
// CHECK:STDOUT:     %Optional.Get.bound: <bound method> = bound_method %.loc27_37.8, %Get.ref
// CHECK:STDOUT:     %Optional.Get.specific_fn: <specific function> = specific_function %Get.ref, @Optional.Get(constants.%OptionalStorage.facet.496) [concrete = constants.%Optional.Get.specific_fn.548]
// CHECK:STDOUT:     %bound_method.loc27_37.6: <bound method> = bound_method %.loc27_37.8, %Optional.Get.specific_fn
// CHECK:STDOUT:     %.loc27_37.14: ref %OverloadValueType = temporary_storage
// CHECK:STDOUT:     %.loc27_37.15: %Optional.4e9 = acquire_value %.loc27_37.8
// CHECK:STDOUT:     %Optional.Get.call: init %OverloadValueType to %.loc27_37.14 = call %bound_method.loc27_37.6(%.loc27_37.15)
// CHECK:STDOUT:     %.loc27_37.16: ref %OverloadValueType = temporary %.loc27_37.14, %Optional.Get.call
// CHECK:STDOUT:     %.loc27_37.17: %OverloadValueType = acquire_value %.loc27_37.16
// CHECK:STDOUT:     %.loc27_14: type = splice_block %OverloadValueType.ref.loc27 [concrete = constants.%OverloadValueType] {
// CHECK:STDOUT:       %Cpp.ref.loc27: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:       %OverloadValueType.ref.loc27: type = name_ref OverloadValueType, imports.%OverloadValueType.decl [concrete = constants.%OverloadValueType]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %_: %OverloadValueType = wrapper_binding _, %.loc27_37.17
// CHECK:STDOUT:     %OverloadValueType.Op.bound: <bound method> = bound_method %.loc27_37.16, constants.%OverloadValueType.Op.4dca31.1
// CHECK:STDOUT:     %Op.ref: %OverloadValueType.cpp_destructor.type = name_ref Op, imports.%OverloadValueType.cpp_destructor.decl [concrete = constants.%OverloadValueType.cpp_destructor]
// CHECK:STDOUT:     %OverloadValueType.cpp_destructor.bound: <bound method> = bound_method %.loc27_37.16, %Op.ref
// CHECK:STDOUT:     %OverloadValueType.cpp_destructor.call: init %empty_tuple.type = call %OverloadValueType.cpp_destructor.bound(%.loc27_37.16)
// CHECK:STDOUT:     %Destroy.Op.bound.loc27_37.1: <bound method> = bound_method %.loc27_37.8, constants.%Destroy.Op.1a2547.7
// CHECK:STDOUT:     %Destroy.Op.call.loc27_37.1: init %empty_tuple.type = call %Destroy.Op.bound.loc27_37.1(%.loc27_37.8)
// CHECK:STDOUT:     br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.done:
// CHECK:STDOUT:     %Destroy.Op.bound.loc27_37.2: <bound method> = bound_method %.loc27_37.8, constants.%Destroy.Op.1a2547.7
// CHECK:STDOUT:     %Destroy.Op.call.loc27_37.2: init %empty_tuple.type = call %Destroy.Op.bound.loc27_37.2(%.loc27_37.8)
// CHECK:STDOUT:     %impl.elem0.loc27_37.1: @TestIterate.%.loc27_37.22 (%.696) = impl_witness_access constants.%Destroy.lookup_impl_witness.d2d, element0 [symbolic = %impl.elem0.loc27_37.2 (constants.%impl.elem0.18a)]
// CHECK:STDOUT:     %bound_method.loc27_37.7: <bound method> = bound_method %var, %impl.elem0.loc27_37.1
// CHECK:STDOUT:     %.loc27_37.18: %Destroy.type = converted constants.%as_type.810, constants.%impl.elem1.424 [symbolic = %impl.elem1 (constants.%impl.elem1.424)]
// CHECK:STDOUT:     %.loc27_37.19: %Destroy.type = converted constants.%as_type.810, constants.%impl.elem1.424 [symbolic = %impl.elem1 (constants.%impl.elem1.424)]
// CHECK:STDOUT:     %specific_impl_fn.loc27_37.3: <specific function> = specific_impl_function %impl.elem0.loc27_37.1, @Destroy.WithSelf.Op(constants.%impl.elem1.424) [symbolic = %specific_impl_fn.loc27_37.6 (constants.%specific_impl_fn.b04)]
// CHECK:STDOUT:     %bound_method.loc27_37.8: <bound method> = bound_method %var, %specific_impl_fn.loc27_37.3
// CHECK:STDOUT:     %Destroy.WithSelf.Op.call: init %empty_tuple.type = call %bound_method.loc27_37.8(%var)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc27_37.1(%self.param: ref %.8a2) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc27_37.2(%self.param: ref %MaybeUnformed.59c) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc27_37.3(%self.param: ref bool) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc27_37.4(%self.param: ref %struct_type.value.has_value.3e8) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc27_37.5(%self.param: ref %DefaultOptionalStorage.799) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc27_37.6(%self.param: ref %Optional.4e9) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestRangeFor(%r.param: ref %Range) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %_.patt: %pattern_type.804 = value_binding_pattern _ [concrete = constants.%_.patt.a10]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %r.ref: ref %Range = name_ref r, %r
// CHECK:STDOUT:   %impl.elem2: %.687 = impl_witness_access constants.%Iterate.impl_witness.175, element2 [concrete = constants.%T.as_type.as.Iterate.impl.NewCursor.036]
// CHECK:STDOUT:   %bound_method.loc37_37.1: <bound method> = bound_method %r.ref, %impl.elem2
// CHECK:STDOUT:   %facet_value.loc37_37.1: %CppRange_where.type.d682d6.1 = facet_value constants.%Range, (constants.%custom_witness.e7a31c.1, constants.%custom_witness.d0da43.1, constants.%custom_witness.090, constants.%custom_witness.f4a, constants.%custom_witness.6bd, constants.%custom_witness.716, constants.%custom_witness.617) [concrete = constants.%facet_value.cbe]
// CHECK:STDOUT:   %.loc37_37.1: %CppRange_where.type.d682d6.1 = converted constants.%Range, %facet_value.loc37_37.1 [concrete = constants.%facet_value.cbe]
// CHECK:STDOUT:   %facet_value.loc37_37.2: %CppRange_where.type.d682d6.1 = facet_value constants.%Range, (constants.%custom_witness.e7a31c.1, constants.%custom_witness.d0da43.1, constants.%custom_witness.090, constants.%custom_witness.f4a, constants.%custom_witness.6bd, constants.%custom_witness.716, constants.%custom_witness.617) [concrete = constants.%facet_value.cbe]
// CHECK:STDOUT:   %.loc37_37.2: %CppRange_where.type.d682d6.1 = converted constants.%Range, %facet_value.loc37_37.2 [concrete = constants.%facet_value.cbe]
// CHECK:STDOUT:   %specific_fn.loc37_37.1: <specific function> = specific_function %impl.elem2, @T.as_type.as.Iterate.impl.NewCursor(constants.%facet_value.cbe) [concrete = constants.%T.as_type.as.Iterate.impl.NewCursor.specific_fn]
// CHECK:STDOUT:   %bound_method.loc37_37.2: <bound method> = bound_method %r.ref, %specific_fn.loc37_37.1
// CHECK:STDOUT:   %var: ref %tuple.type.9ef = var_storage invalid
// CHECK:STDOUT:   %.loc37_36.1: %Range = acquire_value %r.ref
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.call: init %tuple.type.9ef to %var = call %bound_method.loc37_37.2(%.loc37_36.1)
// CHECK:STDOUT:   assign %var, %T.as_type.as.Iterate.impl.NewCursor.call
// CHECK:STDOUT:   br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT: !for.next:
// CHECK:STDOUT:   %addr: %ptr.b18 = addr_of %var
// CHECK:STDOUT:   %impl.elem3: %.79c = impl_witness_access constants.%Iterate.impl_witness.175, element3 [concrete = constants.%T.as_type.as.Iterate.impl.Next.751]
// CHECK:STDOUT:   %bound_method.loc37_37.3: <bound method> = bound_method %r.ref, %impl.elem3
// CHECK:STDOUT:   %facet_value.loc37_37.3: %CppRange_where.type.d682d6.1 = facet_value constants.%Range, (constants.%custom_witness.e7a31c.1, constants.%custom_witness.d0da43.1, constants.%custom_witness.090, constants.%custom_witness.f4a, constants.%custom_witness.6bd, constants.%custom_witness.716, constants.%custom_witness.617) [concrete = constants.%facet_value.cbe]
// CHECK:STDOUT:   %.loc37_37.3: %CppRange_where.type.d682d6.1 = converted constants.%Range, %facet_value.loc37_37.3 [concrete = constants.%facet_value.cbe]
// CHECK:STDOUT:   %facet_value.loc37_37.4: %CppRange_where.type.d682d6.1 = facet_value constants.%Range, (constants.%custom_witness.e7a31c.1, constants.%custom_witness.d0da43.1, constants.%custom_witness.090, constants.%custom_witness.f4a, constants.%custom_witness.6bd, constants.%custom_witness.716, constants.%custom_witness.617) [concrete = constants.%facet_value.cbe]
// CHECK:STDOUT:   %.loc37_37.4: %CppRange_where.type.d682d6.1 = converted constants.%Range, %facet_value.loc37_37.4 [concrete = constants.%facet_value.cbe]
// CHECK:STDOUT:   %specific_fn.loc37_37.2: <specific function> = specific_function %impl.elem3, @T.as_type.as.Iterate.impl.Next(constants.%facet_value.cbe) [concrete = constants.%T.as_type.as.Iterate.impl.Next.specific_fn]
// CHECK:STDOUT:   %bound_method.loc37_37.4: <bound method> = bound_method %r.ref, %specific_fn.loc37_37.2
// CHECK:STDOUT:   %.loc37_37.5: ref %Optional.4e9 = temporary_storage
// CHECK:STDOUT:   %.loc37_36.2: %Range = acquire_value %r.ref
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.call: init %Optional.4e9 to %.loc37_37.5 = call %bound_method.loc37_37.4(%.loc37_36.2, %addr)
// CHECK:STDOUT:   %.loc37_37.6: ref %Optional.4e9 = temporary %.loc37_37.5, %T.as_type.as.Iterate.impl.Next.call
// CHECK:STDOUT:   %.loc37_37.7: %Optional.HasValue.type.59f = specific_constant imports.%Core.import_ref.bb6, @Optional(constants.%OptionalStorage.facet.496) [concrete = constants.%Optional.HasValue.91a]
// CHECK:STDOUT:   %HasValue.ref: %Optional.HasValue.type.59f = name_ref HasValue, %.loc37_37.7 [concrete = constants.%Optional.HasValue.91a]
// CHECK:STDOUT:   %Optional.HasValue.bound: <bound method> = bound_method %.loc37_37.6, %HasValue.ref
// CHECK:STDOUT:   %Optional.HasValue.specific_fn: <specific function> = specific_function %HasValue.ref, @Optional.HasValue(constants.%OptionalStorage.facet.496) [concrete = constants.%Optional.HasValue.specific_fn.bf1]
// CHECK:STDOUT:   %bound_method.loc37_37.5: <bound method> = bound_method %.loc37_37.6, %Optional.HasValue.specific_fn
// CHECK:STDOUT:   %.loc37_37.8: %Optional.4e9 = acquire_value %.loc37_37.6
// CHECK:STDOUT:   %Optional.HasValue.call: init bool = call %bound_method.loc37_37.5(%.loc37_37.8)
// CHECK:STDOUT:   %.loc37_37.9: bool = value_of_initializer %Optional.HasValue.call
// CHECK:STDOUT:   %.loc37_37.10: bool = converted %Optional.HasValue.call, %.loc37_37.9
// CHECK:STDOUT:   if %.loc37_37.10 br !for.body else br !for.done
// CHECK:STDOUT:
// CHECK:STDOUT: !for.body:
// CHECK:STDOUT:   %.loc37_37.11: %Optional.Get.type.cab = specific_constant imports.%Core.import_ref.dab, @Optional(constants.%OptionalStorage.facet.496) [concrete = constants.%Optional.Get.2d3]
// CHECK:STDOUT:   %Get.ref: %Optional.Get.type.cab = name_ref Get, %.loc37_37.11 [concrete = constants.%Optional.Get.2d3]
// CHECK:STDOUT:   %Optional.Get.bound: <bound method> = bound_method %.loc37_37.6, %Get.ref
// CHECK:STDOUT:   %Optional.Get.specific_fn: <specific function> = specific_function %Get.ref, @Optional.Get(constants.%OptionalStorage.facet.496) [concrete = constants.%Optional.Get.specific_fn.548]
// CHECK:STDOUT:   %bound_method.loc37_37.6: <bound method> = bound_method %.loc37_37.6, %Optional.Get.specific_fn
// CHECK:STDOUT:   %.loc37_37.12: ref %OverloadValueType = temporary_storage
// CHECK:STDOUT:   %.loc37_37.13: %Optional.4e9 = acquire_value %.loc37_37.6
// CHECK:STDOUT:   %Optional.Get.call: init %OverloadValueType to %.loc37_37.12 = call %bound_method.loc37_37.6(%.loc37_37.13)
// CHECK:STDOUT:   %.loc37_37.14: ref %OverloadValueType = temporary %.loc37_37.12, %Optional.Get.call
// CHECK:STDOUT:   %.loc37_37.15: %OverloadValueType = acquire_value %.loc37_37.14
// CHECK:STDOUT:   %.loc37_14: type = splice_block %OverloadValueType.ref [concrete = constants.%OverloadValueType] {
// CHECK:STDOUT:     %Cpp.ref.loc37: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %OverloadValueType.ref: type = name_ref OverloadValueType, imports.%OverloadValueType.decl [concrete = constants.%OverloadValueType]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %_: %OverloadValueType = wrapper_binding _, %.loc37_37.15
// CHECK:STDOUT:   %OverloadValueType.Op.bound: <bound method> = bound_method %.loc37_37.14, constants.%OverloadValueType.Op.4dca31.1
// CHECK:STDOUT:   %Op.ref: %OverloadValueType.cpp_destructor.type = name_ref Op, imports.%OverloadValueType.cpp_destructor.decl [concrete = constants.%OverloadValueType.cpp_destructor]
// CHECK:STDOUT:   %OverloadValueType.cpp_destructor.bound: <bound method> = bound_method %.loc37_37.14, %Op.ref
// CHECK:STDOUT:   %OverloadValueType.cpp_destructor.call: init %empty_tuple.type = call %OverloadValueType.cpp_destructor.bound(%.loc37_37.14)
// CHECK:STDOUT:   %Destroy.Op.bound.loc37_37.1: <bound method> = bound_method %.loc37_37.6, constants.%Destroy.Op.1a2547.7
// CHECK:STDOUT:   %Destroy.Op.call.loc37_37.1: init %empty_tuple.type = call %Destroy.Op.bound.loc37_37.1(%.loc37_37.6)
// CHECK:STDOUT:   br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT: !for.done:
// CHECK:STDOUT:   %Destroy.Op.bound.loc37_37.2: <bound method> = bound_method %.loc37_37.6, constants.%Destroy.Op.1a2547.7
// CHECK:STDOUT:   %Destroy.Op.call.loc37_37.2: init %empty_tuple.type = call %Destroy.Op.bound.loc37_37.2(%.loc37_37.6)
// CHECK:STDOUT:   %Destroy.Op.bound.loc37_37.3: <bound method> = bound_method %var, constants.%Destroy.Op.1a2547.8
// CHECK:STDOUT:   %Destroy.Op.call.loc37_37.3: init %empty_tuple.type = call %Destroy.Op.bound.loc37_37.3(%var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestIterate(constants.%R) {
// CHECK:STDOUT:   %R.patt.loc25_17.2 => constants.%R.patt
// CHECK:STDOUT:   %R.loc25_17.1 => constants.%R
// CHECK:STDOUT:   %R.as_type.loc25_83.1 => constants.%R.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.def
// CHECK:STDOUT:   %r.patt.loc25_81.2 => constants.%r.patt.bb3
// CHECK:STDOUT:   %r.param_patt.loc25_76.2 => constants.%r.param_patt.9b2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestIterate(constants.%facet_value.3af) {
// CHECK:STDOUT:   %R.patt.loc25_17.2 => constants.%R.patt
// CHECK:STDOUT:   %R.loc25_17.1 => constants.%facet_value.3af
// CHECK:STDOUT:   %R.as_type.loc25_83.1 => constants.%Range
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.23f
// CHECK:STDOUT:   %r.patt.loc25_81.2 => constants.%r.patt.b9f
// CHECK:STDOUT:   %r.param_patt.loc25_76.2 => constants.%r.param_patt.466
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc25 => constants.%complete_type.357
// CHECK:STDOUT:   %Iterate.lookup_impl_witness => constants.%Iterate.impl_witness.175
// CHECK:STDOUT:   %Iterate.facet.loc27_37.5 => constants.%Iterate.facet.cc4
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type => constants.%Iterate.WithSelf.NewCursor.type.a44
// CHECK:STDOUT:   %.loc27_37.20 => constants.%.687
// CHECK:STDOUT:   %impl.elem2.loc27_37.2 => constants.%T.as_type.as.Iterate.impl.NewCursor.036
// CHECK:STDOUT:   %specific_impl_fn.loc27_37.4 => constants.%T.as_type.as.Iterate.impl.NewCursor.specific_fn
// CHECK:STDOUT:   %impl.elem1 => constants.%Destroy.facet.15b
// CHECK:STDOUT:   %as_type => constants.%tuple.type.9ef
// CHECK:STDOUT:   %require_complete.1 => constants.%complete_type.e35
// CHECK:STDOUT:   %ptr => constants.%ptr.b18
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type => constants.%Iterate.WithSelf.Next.type.1f9
// CHECK:STDOUT:   %.loc27_37.21 => constants.%.79c
// CHECK:STDOUT:   %impl.elem3.loc27_37.2 => constants.%T.as_type.as.Iterate.impl.Next.751
// CHECK:STDOUT:   %specific_impl_fn.loc27_37.5 => constants.%T.as_type.as.Iterate.impl.Next.specific_fn
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.fd5
// CHECK:STDOUT:   %.loc27_37.22 => constants.%.fb6
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.df9cc1.8
// CHECK:STDOUT:   %impl.elem0.loc27_37.2 => constants.%Destroy.Op.1a2547.8
// CHECK:STDOUT:   %specific_impl_fn.loc27_37.6 => constants.%Destroy.Op.1a2547.8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- adl_return_same_types.carbon
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
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Get.type.a12: type = fn_type @T.as_type.as.OptionalStorage.impl.Get, @T.as_type.as.OptionalStorage.impl(%T.283) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Get.d32: %T.as_type.as.OptionalStorage.impl.Get.type.a12 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Has.type.e5a: type = fn_type @T.as_type.as.OptionalStorage.impl.Has, @T.as_type.as.OptionalStorage.impl(%T.283) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Has.4ee: %T.as_type.as.OptionalStorage.impl.Has.type.e5a = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Some.type.492: type = fn_type @T.as_type.as.OptionalStorage.impl.Some, @T.as_type.as.OptionalStorage.impl(%T.283) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Some.89a: %T.as_type.as.OptionalStorage.impl.Some.type.492 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.None.type.0a1: type = fn_type @T.as_type.as.OptionalStorage.impl.None, @T.as_type.as.OptionalStorage.impl(%T.283) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.None.e06: %T.as_type.as.OptionalStorage.impl.None.type.0a1 = struct_value () [symbolic]
// CHECK:STDOUT:   %N.fe9: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %.Self.500: %Iterate.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %CppRange.type: type = facet_type <@CppRange> [concrete]
// CHECK:STDOUT:   %.Self.e0b: %CppRange.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %CppRangeForIterate.lookup_impl_witness.50b: <witness> = lookup_impl_witness %.Self.e0b, @CppRangeForIterate [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.32d: type = impl_witness_access %CppRangeForIterate.lookup_impl_witness.50b, element0 [symbolic_self]
// CHECK:STDOUT:   %CppUnsafeDeref.lookup_impl_witness.851: <witness> = lookup_impl_witness %impl.elem0.32d, @CppUnsafeDeref [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.10d: type = impl_witness_access %CppUnsafeDeref.lookup_impl_witness.851, element0 [symbolic_self]
// CHECK:STDOUT:   %CppRange_where.type.d682d6.1: type = facet_type <@CppRange where %impl.elem0.10d impls @Copy> [concrete]
// CHECK:STDOUT:   %T.f45530.2: %CppRange_where.type.d682d6.1 = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.type.b36: type = fn_type @T.as_type.as.Iterate.impl.Next, @T.as_type.as.Iterate.impl(%T.f45530.2) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.d6b: %T.as_type.as.Iterate.impl.Next.type.b36 = struct_value () [symbolic]
// CHECK:STDOUT:   %CppRangeForIterate.lookup_impl_witness.b581a8.2: <witness> = lookup_impl_witness %T.f45530.2, @CppRangeForIterate [symbolic]
// CHECK:STDOUT:   %impl.elem0.5f971c.2: type = impl_witness_access %CppRangeForIterate.lookup_impl_witness.b581a8.2, element0 [symbolic]
// CHECK:STDOUT:   %CppUnsafeDeref.lookup_impl_witness.87e543.2: <witness> = lookup_impl_witness %impl.elem0.5f971c.2, @CppUnsafeDeref [symbolic]
// CHECK:STDOUT:   %impl.elem0.493765.2: type = impl_witness_access %CppUnsafeDeref.lookup_impl_witness.87e543.2, element0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.type.953: type = fn_type @T.as_type.as.Iterate.impl.NewCursor, @T.as_type.as.Iterate.impl(%T.f45530.2) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.544: %T.as_type.as.Iterate.impl.NewCursor.type.953 = struct_value () [symbolic]
// CHECK:STDOUT:   %impl.elem1.c2dc15.2: type = impl_witness_access %CppRangeForIterate.lookup_impl_witness.b581a8.2, element1 [symbolic]
// CHECK:STDOUT:   %tuple.type.1ba: type = tuple_type (%impl.elem0.5f971c.2, %impl.elem1.c2dc15.2) [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.812: <witness> = lookup_impl_witness %tuple.type.1ba, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.9cd: %Destroy.type = facet_value %tuple.type.1ba, (%Destroy.lookup_impl_witness.812) [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.35816c.2: <witness> = lookup_impl_witness %impl.elem0.493765.2, @Destroy [symbolic]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.167: <witness> = lookup_impl_witness %impl.elem0.493765.2, @Copy [symbolic]
// CHECK:STDOUT:   %facet_value.f93: %facet_type.f48 = facet_value %impl.elem0.493765.2, (%Destroy.lookup_impl_witness.35816c.2, %Copy.lookup_impl_witness.167) [symbolic]
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete]
// CHECK:STDOUT:   %f64.dc1: type = class_type @Float, @Float(%int_64) [concrete]
// CHECK:STDOUT:   %pattern_type.fb7: type = pattern_type %f64.dc1 [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.3: type = fn_type @Destroy.Op.loc20_53.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.3: %Destroy.Op.type.1d8f74.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.df9cc1.3: <witness> = custom_witness (%Destroy.Op.1a2547.3), @Destroy [concrete]
// CHECK:STDOUT:   %Float.as.Copy.impl.Op.type.ff3: type = fn_type @Float.as.Copy.impl.Op, @Float.as.Copy.impl(%N.fe9) [symbolic]
// CHECK:STDOUT:   %Float.as.Copy.impl.Op.d93: %Float.as.Copy.impl.Op.type.ff3 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.0d6: <witness> = impl_witness imports.%Copy.impl_witness_table.8dd, @Float.as.Copy.impl(%int_64) [concrete]
// CHECK:STDOUT:   %facet_value.20d: %facet_type.f48 = facet_value %f64.dc1, (%custom_witness.df9cc1.3, %Copy.impl_witness.0d6) [concrete]
// CHECK:STDOUT:   %Iterate.lookup_impl_witness.1d8: <witness> = lookup_impl_witness %.Self.500, @Iterate [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.294: %facet_type.f48 = impl_witness_access %Iterate.lookup_impl_witness.1d8, element0 [symbolic_self]
// CHECK:STDOUT:   %Iterate_where.type.e6d: type = facet_type <@Iterate where %impl.elem0.294 = %facet_value.20d> [concrete]
// CHECK:STDOUT:   %pattern_type.dc5: type = pattern_type %Iterate_where.type.e6d [concrete]
// CHECK:STDOUT:   %R.patt: %pattern_type.dc5 = symbolic_binding_pattern R, 0 [symbolic]
// CHECK:STDOUT:   %R: %Iterate_where.type.e6d = symbolic_binding R, 0 [symbolic]
// CHECK:STDOUT:   %R.as_type: type = facet_access_type %R [symbolic]
// CHECK:STDOUT:   %pattern_type.d35: type = pattern_type %R.as_type [symbolic]
// CHECK:STDOUT:   %r.param_patt.bc2: %pattern_type.d35 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %r.patt.47e: %pattern_type.d35 = wrapper_binding_pattern r, %r.param_patt.bc2 [symbolic]
// CHECK:STDOUT:   %i.patt: %pattern_type.fb7 = value_binding_pattern i [concrete]
// CHECK:STDOUT:   %Iterate.lookup_impl_witness.757: <witness> = lookup_impl_witness %R, @Iterate [symbolic]
// CHECK:STDOUT:   %Iterate.facet.737: %Iterate.type = facet_value %R.as_type, (%Iterate.lookup_impl_witness.757) [symbolic]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.346: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.737) [symbolic]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.f8a: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.737) [symbolic]
// CHECK:STDOUT:   %.41f: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.346, %Iterate.facet.737 [symbolic]
// CHECK:STDOUT:   %impl.elem2.804: %.41f = impl_witness_access %Iterate.lookup_impl_witness.757, element2 [symbolic]
// CHECK:STDOUT:   %impl.elem1.e8e: %Destroy.type = impl_witness_access %Iterate.lookup_impl_witness.757, element1 [symbolic]
// CHECK:STDOUT:   %as_type.b98: type = facet_access_type %impl.elem1.e8e [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.a60: <specific function> = specific_impl_function %impl.elem2.804, @Iterate.WithSelf.NewCursor(%Iterate.facet.737) [symbolic]
// CHECK:STDOUT:   %ptr.f03: type = ptr_type %as_type.b98 [symbolic]
// CHECK:STDOUT:   %.ecb: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.f8a, %Iterate.facet.737 [symbolic]
// CHECK:STDOUT:   %impl.elem3.d3c: %.ecb = impl_witness_access %Iterate.lookup_impl_witness.757, element3 [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.8c0: type = class_type @DefaultOptionalStorage, @DefaultOptionalStorage(%facet_value.20d) [concrete]
// CHECK:STDOUT:   %OptionalStorage.impl_witness.872: <witness> = impl_witness imports.%OptionalStorage.impl_witness_table.460, @T.as_type.as.OptionalStorage.impl(%facet_value.20d) [concrete]
// CHECK:STDOUT:   %OptionalStorage.facet.4d5: %OptionalStorage.type = facet_value %f64.dc1, (%OptionalStorage.impl_witness.872) [concrete]
// CHECK:STDOUT:   %Optional.e07: type = class_type @Optional, @Optional(%OptionalStorage.facet.4d5) [concrete]
// CHECK:STDOUT:   %specific_impl_fn.b90: <specific function> = specific_impl_function %impl.elem3.d3c, @Iterate.WithSelf.Next(%Iterate.facet.737) [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.type.43a: type = fn_type @Optional.HasValue, @Optional(%OptionalStorage.facet.4d5) [concrete]
// CHECK:STDOUT:   %Optional.HasValue.1c0: %Optional.HasValue.type.43a = struct_value () [concrete]
// CHECK:STDOUT:   %Optional.Get.type.a84: type = fn_type @Optional.Get, @Optional(%OptionalStorage.facet.4d5) [concrete]
// CHECK:STDOUT:   %Optional.Get.4e0: %Optional.Get.type.a84 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.facet.339: %Destroy.type = facet_value %f64.dc1, (%custom_witness.df9cc1.3) [concrete]
// CHECK:STDOUT:   %MaybeUnformed.b7e: type = class_type @MaybeUnformed, @MaybeUnformed(%Destroy.facet.339) [concrete]
// CHECK:STDOUT:   %.cc0: type = maybe_unformed_type %f64.dc1 [concrete]
// CHECK:STDOUT:   %struct_type.value.has_value.ad9: type = struct_type {.value: %MaybeUnformed.b7e, .has_value: bool} [concrete]
// CHECK:STDOUT:   %Optional.HasValue.specific_fn.0a6: <specific function> = specific_function %Optional.HasValue.1c0, @Optional.HasValue(%OptionalStorage.facet.4d5) [concrete]
// CHECK:STDOUT:   %Optional.Get.specific_fn.3a1: <specific function> = specific_function %Optional.Get.4e0, @Optional.Get(%OptionalStorage.facet.4d5) [concrete]
// CHECK:STDOUT:   %AddAssignWith.type.ff9: type = facet_type <@AddAssignWith, @AddAssignWith(%f64.dc1)> [concrete]
// CHECK:STDOUT:   %Float.as.AddAssignWith.impl.Op.type.fdd: type = fn_type @Float.as.AddAssignWith.impl.Op, @Float.as.AddAssignWith.impl(%N.fe9) [symbolic]
// CHECK:STDOUT:   %Float.as.AddAssignWith.impl.Op.430: %Float.as.AddAssignWith.impl.Op.type.fdd = struct_value () [symbolic]
// CHECK:STDOUT:   %AddAssignWith.impl_witness.402: <witness> = impl_witness imports.%AddAssignWith.impl_witness_table, @Float.as.AddAssignWith.impl(%int_64) [concrete]
// CHECK:STDOUT:   %Float.as.AddAssignWith.impl.Op.type.a92: type = fn_type @Float.as.AddAssignWith.impl.Op, @Float.as.AddAssignWith.impl(%int_64) [concrete]
// CHECK:STDOUT:   %Float.as.AddAssignWith.impl.Op.ac1: %Float.as.AddAssignWith.impl.Op.type.a92 = struct_value () [concrete]
// CHECK:STDOUT:   %AddAssignWith.facet: %AddAssignWith.type.ff9 = facet_value %f64.dc1, (%AddAssignWith.impl_witness.402) [concrete]
// CHECK:STDOUT:   %AddAssignWith.WithSelf.Op.type.d04: type = fn_type @AddAssignWith.WithSelf.Op, @AddAssignWith.WithSelf(%f64.dc1, %AddAssignWith.facet) [concrete]
// CHECK:STDOUT:   %.918: type = fn_type_with_self_type %AddAssignWith.WithSelf.Op.type.d04, %AddAssignWith.facet [concrete]
// CHECK:STDOUT:   %Float.as.AddAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Float.as.AddAssignWith.impl.Op.ac1, @Float.as.AddAssignWith.impl.Op(%int_64) [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.9: type = fn_type @Destroy.Op.loc24_19.6 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.9: %Destroy.Op.type.1d8f74.9 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.02f: <witness> = lookup_impl_witness %impl.elem1.e8e, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.f26: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%impl.elem1.e8e) [symbolic]
// CHECK:STDOUT:   %.f38: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.f26, %impl.elem1.e8e [symbolic]
// CHECK:STDOUT:   %impl.elem0.54b: %.f38 = impl_witness_access %Destroy.lookup_impl_witness.02f, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.b70: <specific function> = specific_impl_function %impl.elem0.54b, @Destroy.WithSelf.Op(%impl.elem1.e8e) [symbolic]
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
// CHECK:STDOUT:   %custom_witness.393: <witness> = custom_witness (%Iterator.Op.583fda.3), @Inc [concrete]
// CHECK:STDOUT:   %Equal.type: type = fn_type @Equal [concrete]
// CHECK:STDOUT:   %Equal: %Equal.type = struct_value () [concrete]
// CHECK:STDOUT:   %NotEqual.type: type = fn_type @NotEqual [concrete]
// CHECK:STDOUT:   %NotEqual: %NotEqual.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.821: <witness> = custom_witness (%Equal, %NotEqual), @EqWith, @EqWith(%Iterator) [concrete]
// CHECK:STDOUT:   %facet_value.bbe: %CppRange_where.type.d682d6.1 = facet_value %ConstRange, (%custom_witness.df9cc1.3, %Copy.impl_witness.0d6, %custom_witness.5d5, %custom_witness.213, %custom_witness.994, %custom_witness.393, %custom_witness.821) [concrete]
// CHECK:STDOUT:   %tuple.type.005: type = tuple_type (%Iterator, %Iterator) [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.10: type = fn_type @Destroy.Op.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.10: %Destroy.Op.type.1d8f74.10 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.df9cc1.10: <witness> = custom_witness (%Destroy.Op.1a2547.10), @Destroy [concrete]
// CHECK:STDOUT:   %Destroy.facet.40b: %Destroy.type = facet_value %tuple.type.005, (%custom_witness.df9cc1.10) [concrete]
// CHECK:STDOUT:   %Iterate.impl_witness.526: <witness> = impl_witness imports.%Iterate.impl_witness_table.121, @T.as_type.as.Iterate.impl(%facet_value.bbe) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.type.fa7: type = fn_type @T.as_type.as.Iterate.impl.NewCursor, @T.as_type.as.Iterate.impl(%facet_value.bbe) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.5bc: %T.as_type.as.Iterate.impl.NewCursor.type.fa7 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.type.9a0: type = fn_type @T.as_type.as.Iterate.impl.Next, @T.as_type.as.Iterate.impl(%facet_value.bbe) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.a4d: %T.as_type.as.Iterate.impl.Next.type.9a0 = struct_value () [concrete]
// CHECK:STDOUT:   %Iterate.facet.212: %Iterate.type = facet_value %ConstRange, (%Iterate.impl_witness.526) [concrete]
// CHECK:STDOUT:   %facet_value.eb1: %Iterate_where.type.e6d = facet_value %ConstRange, (%Iterate.impl_witness.526) [concrete]
// CHECK:STDOUT:   %r.param_patt.a73: %pattern_type.025 = value_param_pattern [concrete]
// CHECK:STDOUT:   %r.patt.0b5: %pattern_type.025 = wrapper_binding_pattern r, %r.param_patt.a73 [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.8e0: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.212) [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.2d3: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.212) [concrete]
// CHECK:STDOUT:   %.835: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.8e0, %Iterate.facet.212 [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.specific_fn: <specific function> = specific_function %T.as_type.as.Iterate.impl.NewCursor.5bc, @T.as_type.as.Iterate.impl.NewCursor(%facet_value.bbe) [concrete]
// CHECK:STDOUT:   %ptr.d1a: type = ptr_type %tuple.type.005 [concrete]
// CHECK:STDOUT:   %.7d6: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.2d3, %Iterate.facet.212 [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.specific_fn: <specific function> = specific_function %T.as_type.as.Iterate.impl.Next.a4d, @T.as_type.as.Iterate.impl.Next(%facet_value.bbe) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.117: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.40b) [concrete]
// CHECK:STDOUT:   %.5a4: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.117, %Destroy.facet.40b [concrete]
// CHECK:STDOUT:   %complete_type.320: <witness> = complete_type_witness %tuple.type.005 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.bb6: @Optional.%Optional.HasValue.type (%Optional.HasValue.type.e9e) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.HasValue (constants.%Optional.HasValue.6cd)]
// CHECK:STDOUT:   %Core.import_ref.dab: @Optional.%Optional.Get.type (%Optional.Get.type.bb9) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.Get (constants.%Optional.Get.fa1)]
// CHECK:STDOUT:   %Core.import_ref.0c6: type = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%DefaultOptionalStorage (constants.%DefaultOptionalStorage.d50)]
// CHECK:STDOUT:   %Core.import_ref.dc3: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.None.type (%T.as_type.as.OptionalStorage.impl.None.type.0a1) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.None (constants.%T.as_type.as.OptionalStorage.impl.None.e06)]
// CHECK:STDOUT:   %Core.import_ref.c530: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Some.type (%T.as_type.as.OptionalStorage.impl.Some.type.492) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Some (constants.%T.as_type.as.OptionalStorage.impl.Some.89a)]
// CHECK:STDOUT:   %Core.import_ref.ed4: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Has.type (%T.as_type.as.OptionalStorage.impl.Has.type.e5a) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Has (constants.%T.as_type.as.OptionalStorage.impl.Has.4ee)]
// CHECK:STDOUT:   %Core.import_ref.e9c: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Get.type (%T.as_type.as.OptionalStorage.impl.Get.type.a12) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Get (constants.%T.as_type.as.OptionalStorage.impl.Get.d32)]
// CHECK:STDOUT:   %Core.import_ref.0b2 = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], unloaded
// CHECK:STDOUT:   %OptionalStorage.impl_witness_table.460 = impl_witness_table (%Core.import_ref.0c6, %Core.import_ref.dc3, %Core.import_ref.c530, %Core.import_ref.ed4, %Core.import_ref.e9c, %Core.import_ref.0b2), @T.as_type.as.OptionalStorage.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.eef: %facet_type.f48 = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.Iterate.impl.%facet_value (constants.%facet_value.f93)]
// CHECK:STDOUT:   %Core.import_ref.fcc: %Destroy.type = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.Iterate.impl.%Destroy.facet (constants.%Destroy.facet.9cd)]
// CHECK:STDOUT:   %Core.import_ref.fd1a: @T.as_type.as.Iterate.impl.%T.as_type.as.Iterate.impl.NewCursor.type (%T.as_type.as.Iterate.impl.NewCursor.type.953) = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.Iterate.impl.%T.as_type.as.Iterate.impl.NewCursor (constants.%T.as_type.as.Iterate.impl.NewCursor.544)]
// CHECK:STDOUT:   %Core.import_ref.273: @T.as_type.as.Iterate.impl.%T.as_type.as.Iterate.impl.Next.type (%T.as_type.as.Iterate.impl.Next.type.b36) = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.Iterate.impl.%T.as_type.as.Iterate.impl.Next (constants.%T.as_type.as.Iterate.impl.Next.d6b)]
// CHECK:STDOUT:   %Iterate.impl_witness_table.121 = impl_witness_table (%Core.import_ref.eef, %Core.import_ref.fcc, %Core.import_ref.fd1a, %Core.import_ref.273), @T.as_type.as.Iterate.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.e69: @Float.as.Copy.impl.%Float.as.Copy.impl.Op.type (%Float.as.Copy.impl.Op.type.ff3) = import_ref Core//prelude/types/float, loc{{\d+_\d+}}, loaded [symbolic = @Float.as.Copy.impl.%Float.as.Copy.impl.Op (constants.%Float.as.Copy.impl.Op.d93)]
// CHECK:STDOUT:   %Copy.impl_witness_table.8dd = impl_witness_table (%Core.import_ref.e69), @Float.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.cb3: @Float.as.AddAssignWith.impl.%Float.as.AddAssignWith.impl.Op.type (%Float.as.AddAssignWith.impl.Op.type.fdd) = import_ref Core//prelude/types/float, loc{{\d+_\d+}}, loaded [symbolic = @Float.as.AddAssignWith.impl.%Float.as.AddAssignWith.impl.Op (constants.%Float.as.AddAssignWith.impl.Op.430)]
// CHECK:STDOUT:   %AddAssignWith.impl_witness_table = impl_witness_table (%Core.import_ref.cb3), @Float.as.AddAssignWith.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @TestIterate(%R.loc20_17.2: %Iterate_where.type.e6d) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Iterate.lookup_impl_witness: <witness> = lookup_impl_witness %R.loc20_17.1, @Iterate [symbolic = %Iterate.lookup_impl_witness (constants.%Iterate.lookup_impl_witness.757)]
// CHECK:STDOUT:   %Iterate.facet.loc24_19.5: %Iterate.type = facet_value %R.as_type.loc20_61.1, (%Iterate.lookup_impl_witness) [symbolic = %Iterate.facet.loc24_19.5 (constants.%Iterate.facet.737)]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.loc24_19.5) [symbolic = %Iterate.WithSelf.NewCursor.type (constants.%Iterate.WithSelf.NewCursor.type.346)]
// CHECK:STDOUT:   %.loc24_19.19: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type, %Iterate.facet.loc24_19.5 [symbolic = %.loc24_19.19 (constants.%.41f)]
// CHECK:STDOUT:   %impl.elem2.loc24_19.2: @TestIterate.%.loc24_19.19 (%.41f) = impl_witness_access %Iterate.lookup_impl_witness, element2 [symbolic = %impl.elem2.loc24_19.2 (constants.%impl.elem2.804)]
// CHECK:STDOUT:   %specific_impl_fn.loc24_19.4: <specific function> = specific_impl_function %impl.elem2.loc24_19.2, @Iterate.WithSelf.NewCursor(%Iterate.facet.loc24_19.5) [symbolic = %specific_impl_fn.loc24_19.4 (constants.%specific_impl_fn.a60)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %ptr: type = ptr_type %as_type [symbolic = %ptr (constants.%ptr.f03)]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.loc24_19.5) [symbolic = %Iterate.WithSelf.Next.type (constants.%Iterate.WithSelf.Next.type.f8a)]
// CHECK:STDOUT:   %.loc24_19.20: type = fn_type_with_self_type %Iterate.WithSelf.Next.type, %Iterate.facet.loc24_19.5 [symbolic = %.loc24_19.20 (constants.%.ecb)]
// CHECK:STDOUT:   %impl.elem3.loc24_19.2: @TestIterate.%.loc24_19.20 (%.ecb) = impl_witness_access %Iterate.lookup_impl_witness, element3 [symbolic = %impl.elem3.loc24_19.2 (constants.%impl.elem3.d3c)]
// CHECK:STDOUT:   %specific_impl_fn.loc24_19.5: <specific function> = specific_impl_function %impl.elem3.loc24_19.2, @Iterate.WithSelf.Next(%Iterate.facet.loc24_19.5) [symbolic = %specific_impl_fn.loc24_19.5 (constants.%specific_impl_fn.b90)]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%impl.elem1) [symbolic = %Destroy.WithSelf.Op.type (constants.%Destroy.WithSelf.Op.type.f26)]
// CHECK:STDOUT:   %.loc24_19.21: type = fn_type_with_self_type %Destroy.WithSelf.Op.type, %impl.elem1 [symbolic = %.loc24_19.21 (constants.%.f38)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %impl.elem1, @Destroy [symbolic = %Destroy.lookup_impl_witness (constants.%Destroy.lookup_impl_witness.02f)]
// CHECK:STDOUT:   %impl.elem0.loc24_19.2: @TestIterate.%.loc24_19.21 (%.f38) = impl_witness_access %Destroy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc24_19.2 (constants.%impl.elem0.54b)]
// CHECK:STDOUT:   %specific_impl_fn.loc24_19.6: <specific function> = specific_impl_function %impl.elem0.loc24_19.2, @Destroy.WithSelf.Op(%impl.elem1) [symbolic = %specific_impl_fn.loc24_19.6 (constants.%specific_impl_fn.b70)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%r.param: @TestIterate.%R.as_type.loc20_61.1 (%R.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %i.patt: %pattern_type.fb7 = value_binding_pattern i [concrete = constants.%i.patt]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %r.ref: @TestIterate.%R.as_type.loc20_61.1 (%R.as_type) = name_ref r, %r
// CHECK:STDOUT:     %impl.elem2.loc24_19.1: @TestIterate.%.loc24_19.19 (%.41f) = impl_witness_access constants.%Iterate.lookup_impl_witness.757, element2 [symbolic = %impl.elem2.loc24_19.2 (constants.%impl.elem2.804)]
// CHECK:STDOUT:     %bound_method.loc24_19.1: <bound method> = bound_method %r.ref, %impl.elem2.loc24_19.1
// CHECK:STDOUT:     %Iterate.facet.loc24_19.1: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.757) [symbolic = %Iterate.facet.loc24_19.5 (constants.%Iterate.facet.737)]
// CHECK:STDOUT:     %.loc24_19.1: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc24_19.1 [symbolic = %Iterate.facet.loc24_19.5 (constants.%Iterate.facet.737)]
// CHECK:STDOUT:     %Iterate.facet.loc24_19.2: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.757) [symbolic = %Iterate.facet.loc24_19.5 (constants.%Iterate.facet.737)]
// CHECK:STDOUT:     %.loc24_19.2: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc24_19.2 [symbolic = %Iterate.facet.loc24_19.5 (constants.%Iterate.facet.737)]
// CHECK:STDOUT:     %specific_impl_fn.loc24_19.1: <specific function> = specific_impl_function %impl.elem2.loc24_19.1, @Iterate.WithSelf.NewCursor(constants.%Iterate.facet.737) [symbolic = %specific_impl_fn.loc24_19.4 (constants.%specific_impl_fn.a60)]
// CHECK:STDOUT:     %bound_method.loc24_19.2: <bound method> = bound_method %r.ref, %specific_impl_fn.loc24_19.1
// CHECK:STDOUT:     %var: ref @TestIterate.%as_type (%as_type.b98) = var_storage invalid
// CHECK:STDOUT:     %Iterate.WithSelf.NewCursor.call: init @TestIterate.%as_type (%as_type.b98) to %var = call %bound_method.loc24_19.2(%r.ref)
// CHECK:STDOUT:     assign %var, %Iterate.WithSelf.NewCursor.call
// CHECK:STDOUT:     br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.next:
// CHECK:STDOUT:     %addr: @TestIterate.%ptr (%ptr.f03) = addr_of %var
// CHECK:STDOUT:     %impl.elem3.loc24_19.1: @TestIterate.%.loc24_19.20 (%.ecb) = impl_witness_access constants.%Iterate.lookup_impl_witness.757, element3 [symbolic = %impl.elem3.loc24_19.2 (constants.%impl.elem3.d3c)]
// CHECK:STDOUT:     %bound_method.loc24_19.3: <bound method> = bound_method %r.ref, %impl.elem3.loc24_19.1
// CHECK:STDOUT:     %Iterate.facet.loc24_19.3: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.757) [symbolic = %Iterate.facet.loc24_19.5 (constants.%Iterate.facet.737)]
// CHECK:STDOUT:     %.loc24_19.3: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc24_19.3 [symbolic = %Iterate.facet.loc24_19.5 (constants.%Iterate.facet.737)]
// CHECK:STDOUT:     %Iterate.facet.loc24_19.4: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.757) [symbolic = %Iterate.facet.loc24_19.5 (constants.%Iterate.facet.737)]
// CHECK:STDOUT:     %.loc24_19.4: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc24_19.4 [symbolic = %Iterate.facet.loc24_19.5 (constants.%Iterate.facet.737)]
// CHECK:STDOUT:     %.loc24_19.5: %Destroy.type = converted constants.%as_type.b98, constants.%impl.elem1.e8e [symbolic = %impl.elem1 (constants.%impl.elem1.e8e)]
// CHECK:STDOUT:     %.loc24_19.6: %Destroy.type = converted constants.%as_type.b98, constants.%impl.elem1.e8e [symbolic = %impl.elem1 (constants.%impl.elem1.e8e)]
// CHECK:STDOUT:     %specific_impl_fn.loc24_19.2: <specific function> = specific_impl_function %impl.elem3.loc24_19.1, @Iterate.WithSelf.Next(constants.%Iterate.facet.737) [symbolic = %specific_impl_fn.loc24_19.5 (constants.%specific_impl_fn.b90)]
// CHECK:STDOUT:     %bound_method.loc24_19.4: <bound method> = bound_method %r.ref, %specific_impl_fn.loc24_19.2
// CHECK:STDOUT:     %.loc24_19.7: ref %Optional.e07 = temporary_storage
// CHECK:STDOUT:     %Iterate.WithSelf.Next.call: init %Optional.e07 to %.loc24_19.7 = call %bound_method.loc24_19.4(%r.ref, %addr)
// CHECK:STDOUT:     %.loc24_19.8: ref %Optional.e07 = temporary %.loc24_19.7, %Iterate.WithSelf.Next.call
// CHECK:STDOUT:     %.loc24_19.9: %Optional.HasValue.type.43a = specific_constant imports.%Core.import_ref.bb6, @Optional(constants.%OptionalStorage.facet.4d5) [concrete = constants.%Optional.HasValue.1c0]
// CHECK:STDOUT:     %HasValue.ref: %Optional.HasValue.type.43a = name_ref HasValue, %.loc24_19.9 [concrete = constants.%Optional.HasValue.1c0]
// CHECK:STDOUT:     %Optional.HasValue.bound: <bound method> = bound_method %.loc24_19.8, %HasValue.ref
// CHECK:STDOUT:     %Optional.HasValue.specific_fn: <specific function> = specific_function %HasValue.ref, @Optional.HasValue(constants.%OptionalStorage.facet.4d5) [concrete = constants.%Optional.HasValue.specific_fn.0a6]
// CHECK:STDOUT:     %bound_method.loc24_19.5: <bound method> = bound_method %.loc24_19.8, %Optional.HasValue.specific_fn
// CHECK:STDOUT:     %.loc24_19.10: %Optional.e07 = acquire_value %.loc24_19.8
// CHECK:STDOUT:     %Optional.HasValue.call: init bool = call %bound_method.loc24_19.5(%.loc24_19.10)
// CHECK:STDOUT:     %.loc24_19.11: bool = value_of_initializer %Optional.HasValue.call
// CHECK:STDOUT:     %.loc24_19.12: bool = converted %Optional.HasValue.call, %.loc24_19.11
// CHECK:STDOUT:     if %.loc24_19.12 br !for.body else br !for.done
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.body:
// CHECK:STDOUT:     %.loc24_19.13: %Optional.Get.type.a84 = specific_constant imports.%Core.import_ref.dab, @Optional(constants.%OptionalStorage.facet.4d5) [concrete = constants.%Optional.Get.4e0]
// CHECK:STDOUT:     %Get.ref: %Optional.Get.type.a84 = name_ref Get, %.loc24_19.13 [concrete = constants.%Optional.Get.4e0]
// CHECK:STDOUT:     %Optional.Get.bound: <bound method> = bound_method %.loc24_19.8, %Get.ref
// CHECK:STDOUT:     %Optional.Get.specific_fn: <specific function> = specific_function %Get.ref, @Optional.Get(constants.%OptionalStorage.facet.4d5) [concrete = constants.%Optional.Get.specific_fn.3a1]
// CHECK:STDOUT:     %bound_method.loc24_19.6: <bound method> = bound_method %.loc24_19.8, %Optional.Get.specific_fn
// CHECK:STDOUT:     %.loc24_19.14: %Optional.e07 = acquire_value %.loc24_19.8
// CHECK:STDOUT:     %Optional.Get.call: init %f64.dc1 = call %bound_method.loc24_19.6(%.loc24_19.14)
// CHECK:STDOUT:     %.loc24_19.15: %f64.dc1 = value_of_initializer %Optional.Get.call
// CHECK:STDOUT:     %.loc24_19.16: %f64.dc1 = converted %Optional.Get.call, %.loc24_19.15
// CHECK:STDOUT:     %f64.loc24: type = type_literal constants.%f64.dc1 [concrete = constants.%f64.dc1]
// CHECK:STDOUT:     %i: %f64.dc1 = wrapper_binding i, %.loc24_19.16
// CHECK:STDOUT:     %sum.ref: ref %f64.dc1 = name_ref sum, %sum
// CHECK:STDOUT:     %i.ref: %f64.dc1 = name_ref i, %i
// CHECK:STDOUT:     %impl.elem0.loc25: %.918 = impl_witness_access constants.%AddAssignWith.impl_witness.402, element0 [concrete = constants.%Float.as.AddAssignWith.impl.Op.ac1]
// CHECK:STDOUT:     %bound_method.loc25_9.1: <bound method> = bound_method %sum.ref, %impl.elem0.loc25
// CHECK:STDOUT:     %specific_fn.loc25: <specific function> = specific_function %impl.elem0.loc25, @Float.as.AddAssignWith.impl.Op(constants.%int_64) [concrete = constants.%Float.as.AddAssignWith.impl.Op.specific_fn]
// CHECK:STDOUT:     %bound_method.loc25_9.2: <bound method> = bound_method %sum.ref, %specific_fn.loc25
// CHECK:STDOUT:     %Float.as.AddAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc25_9.2(%sum.ref, %i.ref)
// CHECK:STDOUT:     %Destroy.Op.bound.loc24_19.1: <bound method> = bound_method %.loc24_19.8, constants.%Destroy.Op.1a2547.9
// CHECK:STDOUT:     %Destroy.Op.call.loc24_19.1: init %empty_tuple.type = call %Destroy.Op.bound.loc24_19.1(%.loc24_19.8)
// CHECK:STDOUT:     br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.done:
// CHECK:STDOUT:     %Destroy.Op.bound.loc24_19.2: <bound method> = bound_method %.loc24_19.8, constants.%Destroy.Op.1a2547.9
// CHECK:STDOUT:     %Destroy.Op.call.loc24_19.2: init %empty_tuple.type = call %Destroy.Op.bound.loc24_19.2(%.loc24_19.8)
// CHECK:STDOUT:     %impl.elem0.loc24_19.1: @TestIterate.%.loc24_19.21 (%.f38) = impl_witness_access constants.%Destroy.lookup_impl_witness.02f, element0 [symbolic = %impl.elem0.loc24_19.2 (constants.%impl.elem0.54b)]
// CHECK:STDOUT:     %bound_method.loc24_19.7: <bound method> = bound_method %var, %impl.elem0.loc24_19.1
// CHECK:STDOUT:     %.loc24_19.17: %Destroy.type = converted constants.%as_type.b98, constants.%impl.elem1.e8e [symbolic = %impl.elem1 (constants.%impl.elem1.e8e)]
// CHECK:STDOUT:     %.loc24_19.18: %Destroy.type = converted constants.%as_type.b98, constants.%impl.elem1.e8e [symbolic = %impl.elem1 (constants.%impl.elem1.e8e)]
// CHECK:STDOUT:     %specific_impl_fn.loc24_19.3: <specific function> = specific_impl_function %impl.elem0.loc24_19.1, @Destroy.WithSelf.Op(constants.%impl.elem1.e8e) [symbolic = %specific_impl_fn.loc24_19.6 (constants.%specific_impl_fn.b70)]
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
// CHECK:STDOUT: fn @Destroy.Op.loc24_19.5(%self.param: ref %DefaultOptionalStorage.8c0) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc24_19.6(%self.param: ref %Optional.e07) {
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
// CHECK:STDOUT:   %impl.elem2: %.835 = impl_witness_access constants.%Iterate.impl_witness.526, element2 [concrete = constants.%T.as_type.as.Iterate.impl.NewCursor.5bc]
// CHECK:STDOUT:   %bound_method.loc38_19.1: <bound method> = bound_method %c.ref, %impl.elem2
// CHECK:STDOUT:   %facet_value.loc38_19.1: %CppRange_where.type.d682d6.1 = facet_value constants.%ConstRange, (constants.%custom_witness.df9cc1.3, constants.%Copy.impl_witness.0d6, constants.%custom_witness.5d5, constants.%custom_witness.213, constants.%custom_witness.994, constants.%custom_witness.393, constants.%custom_witness.821) [concrete = constants.%facet_value.bbe]
// CHECK:STDOUT:   %.loc38_19.1: %CppRange_where.type.d682d6.1 = converted constants.%ConstRange, %facet_value.loc38_19.1 [concrete = constants.%facet_value.bbe]
// CHECK:STDOUT:   %facet_value.loc38_19.2: %CppRange_where.type.d682d6.1 = facet_value constants.%ConstRange, (constants.%custom_witness.df9cc1.3, constants.%Copy.impl_witness.0d6, constants.%custom_witness.5d5, constants.%custom_witness.213, constants.%custom_witness.994, constants.%custom_witness.393, constants.%custom_witness.821) [concrete = constants.%facet_value.bbe]
// CHECK:STDOUT:   %.loc38_19.2: %CppRange_where.type.d682d6.1 = converted constants.%ConstRange, %facet_value.loc38_19.2 [concrete = constants.%facet_value.bbe]
// CHECK:STDOUT:   %specific_fn.loc38_19.1: <specific function> = specific_function %impl.elem2, @T.as_type.as.Iterate.impl.NewCursor(constants.%facet_value.bbe) [concrete = constants.%T.as_type.as.Iterate.impl.NewCursor.specific_fn]
// CHECK:STDOUT:   %bound_method.loc38_19.2: <bound method> = bound_method %c.ref, %specific_fn.loc38_19.1
// CHECK:STDOUT:   %var: ref %tuple.type.005 = var_storage invalid
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.call: init %tuple.type.005 to %var = call %bound_method.loc38_19.2(%c.ref)
// CHECK:STDOUT:   assign %var, %T.as_type.as.Iterate.impl.NewCursor.call
// CHECK:STDOUT:   br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT: !for.next:
// CHECK:STDOUT:   %addr: %ptr.d1a = addr_of %var
// CHECK:STDOUT:   %impl.elem3: %.7d6 = impl_witness_access constants.%Iterate.impl_witness.526, element3 [concrete = constants.%T.as_type.as.Iterate.impl.Next.a4d]
// CHECK:STDOUT:   %bound_method.loc38_19.3: <bound method> = bound_method %c.ref, %impl.elem3
// CHECK:STDOUT:   %facet_value.loc38_19.3: %CppRange_where.type.d682d6.1 = facet_value constants.%ConstRange, (constants.%custom_witness.df9cc1.3, constants.%Copy.impl_witness.0d6, constants.%custom_witness.5d5, constants.%custom_witness.213, constants.%custom_witness.994, constants.%custom_witness.393, constants.%custom_witness.821) [concrete = constants.%facet_value.bbe]
// CHECK:STDOUT:   %.loc38_19.3: %CppRange_where.type.d682d6.1 = converted constants.%ConstRange, %facet_value.loc38_19.3 [concrete = constants.%facet_value.bbe]
// CHECK:STDOUT:   %facet_value.loc38_19.4: %CppRange_where.type.d682d6.1 = facet_value constants.%ConstRange, (constants.%custom_witness.df9cc1.3, constants.%Copy.impl_witness.0d6, constants.%custom_witness.5d5, constants.%custom_witness.213, constants.%custom_witness.994, constants.%custom_witness.393, constants.%custom_witness.821) [concrete = constants.%facet_value.bbe]
// CHECK:STDOUT:   %.loc38_19.4: %CppRange_where.type.d682d6.1 = converted constants.%ConstRange, %facet_value.loc38_19.4 [concrete = constants.%facet_value.bbe]
// CHECK:STDOUT:   %specific_fn.loc38_19.2: <specific function> = specific_function %impl.elem3, @T.as_type.as.Iterate.impl.Next(constants.%facet_value.bbe) [concrete = constants.%T.as_type.as.Iterate.impl.Next.specific_fn]
// CHECK:STDOUT:   %bound_method.loc38_19.4: <bound method> = bound_method %c.ref, %specific_fn.loc38_19.2
// CHECK:STDOUT:   %.loc38_19.5: ref %Optional.e07 = temporary_storage
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.call: init %Optional.e07 to %.loc38_19.5 = call %bound_method.loc38_19.4(%c.ref, %addr)
// CHECK:STDOUT:   %.loc38_19.6: ref %Optional.e07 = temporary %.loc38_19.5, %T.as_type.as.Iterate.impl.Next.call
// CHECK:STDOUT:   %.loc38_19.7: %Optional.HasValue.type.43a = specific_constant imports.%Core.import_ref.bb6, @Optional(constants.%OptionalStorage.facet.4d5) [concrete = constants.%Optional.HasValue.1c0]
// CHECK:STDOUT:   %HasValue.ref: %Optional.HasValue.type.43a = name_ref HasValue, %.loc38_19.7 [concrete = constants.%Optional.HasValue.1c0]
// CHECK:STDOUT:   %Optional.HasValue.bound: <bound method> = bound_method %.loc38_19.6, %HasValue.ref
// CHECK:STDOUT:   %Optional.HasValue.specific_fn: <specific function> = specific_function %HasValue.ref, @Optional.HasValue(constants.%OptionalStorage.facet.4d5) [concrete = constants.%Optional.HasValue.specific_fn.0a6]
// CHECK:STDOUT:   %bound_method.loc38_19.5: <bound method> = bound_method %.loc38_19.6, %Optional.HasValue.specific_fn
// CHECK:STDOUT:   %.loc38_19.8: %Optional.e07 = acquire_value %.loc38_19.6
// CHECK:STDOUT:   %Optional.HasValue.call: init bool = call %bound_method.loc38_19.5(%.loc38_19.8)
// CHECK:STDOUT:   %.loc38_19.9: bool = value_of_initializer %Optional.HasValue.call
// CHECK:STDOUT:   %.loc38_19.10: bool = converted %Optional.HasValue.call, %.loc38_19.9
// CHECK:STDOUT:   if %.loc38_19.10 br !for.body else br !for.done
// CHECK:STDOUT:
// CHECK:STDOUT: !for.body:
// CHECK:STDOUT:   %.loc38_19.11: %Optional.Get.type.a84 = specific_constant imports.%Core.import_ref.dab, @Optional(constants.%OptionalStorage.facet.4d5) [concrete = constants.%Optional.Get.4e0]
// CHECK:STDOUT:   %Get.ref: %Optional.Get.type.a84 = name_ref Get, %.loc38_19.11 [concrete = constants.%Optional.Get.4e0]
// CHECK:STDOUT:   %Optional.Get.bound: <bound method> = bound_method %.loc38_19.6, %Get.ref
// CHECK:STDOUT:   %Optional.Get.specific_fn: <specific function> = specific_function %Get.ref, @Optional.Get(constants.%OptionalStorage.facet.4d5) [concrete = constants.%Optional.Get.specific_fn.3a1]
// CHECK:STDOUT:   %bound_method.loc38_19.6: <bound method> = bound_method %.loc38_19.6, %Optional.Get.specific_fn
// CHECK:STDOUT:   %.loc38_19.12: %Optional.e07 = acquire_value %.loc38_19.6
// CHECK:STDOUT:   %Optional.Get.call: init %f64.dc1 = call %bound_method.loc38_19.6(%.loc38_19.12)
// CHECK:STDOUT:   %.loc38_19.13: %f64.dc1 = value_of_initializer %Optional.Get.call
// CHECK:STDOUT:   %.loc38_19.14: %f64.dc1 = converted %Optional.Get.call, %.loc38_19.13
// CHECK:STDOUT:   %f64.loc38: type = type_literal constants.%f64.dc1 [concrete = constants.%f64.dc1]
// CHECK:STDOUT:   %i: %f64.dc1 = wrapper_binding i, %.loc38_19.14
// CHECK:STDOUT:   %sum.ref: ref %f64.dc1 = name_ref sum, %sum
// CHECK:STDOUT:   %i.ref: %f64.dc1 = name_ref i, %i
// CHECK:STDOUT:   %impl.elem0.loc39: %.918 = impl_witness_access constants.%AddAssignWith.impl_witness.402, element0 [concrete = constants.%Float.as.AddAssignWith.impl.Op.ac1]
// CHECK:STDOUT:   %bound_method.loc39_9.1: <bound method> = bound_method %sum.ref, %impl.elem0.loc39
// CHECK:STDOUT:   %specific_fn.loc39: <specific function> = specific_function %impl.elem0.loc39, @Float.as.AddAssignWith.impl.Op(constants.%int_64) [concrete = constants.%Float.as.AddAssignWith.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc39_9.2: <bound method> = bound_method %sum.ref, %specific_fn.loc39
// CHECK:STDOUT:   %Float.as.AddAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc39_9.2(%sum.ref, %i.ref)
// CHECK:STDOUT:   %Destroy.Op.bound.loc38_19.1: <bound method> = bound_method %.loc38_19.6, constants.%Destroy.Op.1a2547.9
// CHECK:STDOUT:   %Destroy.Op.call.loc38_19.1: init %empty_tuple.type = call %Destroy.Op.bound.loc38_19.1(%.loc38_19.6)
// CHECK:STDOUT:   br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT: !for.done:
// CHECK:STDOUT:   %Destroy.Op.bound.loc38_19.2: <bound method> = bound_method %.loc38_19.6, constants.%Destroy.Op.1a2547.9
// CHECK:STDOUT:   %Destroy.Op.call.loc38_19.2: init %empty_tuple.type = call %Destroy.Op.bound.loc38_19.2(%.loc38_19.6)
// CHECK:STDOUT:   %Destroy.Op.bound.loc38_19.3: <bound method> = bound_method %var, constants.%Destroy.Op.1a2547.10
// CHECK:STDOUT:   %Destroy.Op.call.loc38_19.3: init %empty_tuple.type = call %Destroy.Op.bound.loc38_19.3(%var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestIterate(constants.%R) {
// CHECK:STDOUT:   %R.patt.loc20_17.2 => constants.%R.patt
// CHECK:STDOUT:   %R.loc20_17.1 => constants.%R
// CHECK:STDOUT:   %R.as_type.loc20_61.1 => constants.%R.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.d35
// CHECK:STDOUT:   %r.param_patt.loc20_59.2 => constants.%r.param_patt.bc2
// CHECK:STDOUT:   %r.patt.loc20_59.2 => constants.%r.patt.47e
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestIterate(constants.%facet_value.eb1) {
// CHECK:STDOUT:   %R.patt.loc20_17.2 => constants.%R.patt
// CHECK:STDOUT:   %R.loc20_17.1 => constants.%facet_value.eb1
// CHECK:STDOUT:   %R.as_type.loc20_61.1 => constants.%ConstRange
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.025
// CHECK:STDOUT:   %r.param_patt.loc20_59.2 => constants.%r.param_patt.a73
// CHECK:STDOUT:   %r.patt.loc20_59.2 => constants.%r.patt.0b5
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc20 => constants.%complete_type.357
// CHECK:STDOUT:   %Iterate.lookup_impl_witness => constants.%Iterate.impl_witness.526
// CHECK:STDOUT:   %Iterate.facet.loc24_19.5 => constants.%Iterate.facet.212
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type => constants.%Iterate.WithSelf.NewCursor.type.8e0
// CHECK:STDOUT:   %.loc24_19.19 => constants.%.835
// CHECK:STDOUT:   %impl.elem2.loc24_19.2 => constants.%T.as_type.as.Iterate.impl.NewCursor.5bc
// CHECK:STDOUT:   %specific_impl_fn.loc24_19.4 => constants.%T.as_type.as.Iterate.impl.NewCursor.specific_fn
// CHECK:STDOUT:   %impl.elem1 => constants.%Destroy.facet.40b
// CHECK:STDOUT:   %as_type => constants.%tuple.type.005
// CHECK:STDOUT:   %require_complete.1 => constants.%complete_type.320
// CHECK:STDOUT:   %ptr => constants.%ptr.d1a
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type => constants.%Iterate.WithSelf.Next.type.2d3
// CHECK:STDOUT:   %.loc24_19.20 => constants.%.7d6
// CHECK:STDOUT:   %impl.elem3.loc24_19.2 => constants.%T.as_type.as.Iterate.impl.Next.a4d
// CHECK:STDOUT:   %specific_impl_fn.loc24_19.5 => constants.%T.as_type.as.Iterate.impl.Next.specific_fn
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
// CHECK:STDOUT:   %facet_type.f48: type = facet_type <@Destroy & @Copy> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %T.220: %OptionalStorage.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Optional.Get.type.bb9: type = fn_type @Optional.Get, @Optional(%T.220) [symbolic]
// CHECK:STDOUT:   %Optional.Get.fa1: %Optional.Get.type.bb9 = struct_value () [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.type.e9e: type = fn_type @Optional.HasValue, @Optional(%T.220) [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.6cd: %Optional.HasValue.type.e9e = struct_value () [symbolic]
// CHECK:STDOUT:   %T.283: %facet_type.f48 = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.d50: type = class_type @DefaultOptionalStorage, @DefaultOptionalStorage(%T.283) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Get.type.a12: type = fn_type @T.as_type.as.OptionalStorage.impl.Get, @T.as_type.as.OptionalStorage.impl(%T.283) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Get.d32: %T.as_type.as.OptionalStorage.impl.Get.type.a12 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Has.type.e5a: type = fn_type @T.as_type.as.OptionalStorage.impl.Has, @T.as_type.as.OptionalStorage.impl(%T.283) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Has.4ee: %T.as_type.as.OptionalStorage.impl.Has.type.e5a = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Some.type.492: type = fn_type @T.as_type.as.OptionalStorage.impl.Some, @T.as_type.as.OptionalStorage.impl(%T.283) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Some.89a: %T.as_type.as.OptionalStorage.impl.Some.type.492 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.None.type.0a1: type = fn_type @T.as_type.as.OptionalStorage.impl.None, @T.as_type.as.OptionalStorage.impl(%T.283) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.None.e06: %T.as_type.as.OptionalStorage.impl.None.type.0a1 = struct_value () [symbolic]
// CHECK:STDOUT:   %N.fe9: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %.Self.500: %Iterate.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %CppRange.type: type = facet_type <@CppRange> [concrete]
// CHECK:STDOUT:   %.Self.e0b: %CppRange.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %CppRangeForIterate.lookup_impl_witness.50b: <witness> = lookup_impl_witness %.Self.e0b, @CppRangeForIterate [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.32d: type = impl_witness_access %CppRangeForIterate.lookup_impl_witness.50b, element0 [symbolic_self]
// CHECK:STDOUT:   %CppUnsafeDeref.lookup_impl_witness.851: <witness> = lookup_impl_witness %impl.elem0.32d, @CppUnsafeDeref [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.10d: type = impl_witness_access %CppUnsafeDeref.lookup_impl_witness.851, element0 [symbolic_self]
// CHECK:STDOUT:   %CppRange_where.type.d682d6.1: type = facet_type <@CppRange where %impl.elem0.10d impls @Copy> [concrete]
// CHECK:STDOUT:   %T.f45530.2: %CppRange_where.type.d682d6.1 = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.type.b36: type = fn_type @T.as_type.as.Iterate.impl.Next, @T.as_type.as.Iterate.impl(%T.f45530.2) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.d6b: %T.as_type.as.Iterate.impl.Next.type.b36 = struct_value () [symbolic]
// CHECK:STDOUT:   %CppRangeForIterate.lookup_impl_witness.b581a8.2: <witness> = lookup_impl_witness %T.f45530.2, @CppRangeForIterate [symbolic]
// CHECK:STDOUT:   %impl.elem0.5f971c.2: type = impl_witness_access %CppRangeForIterate.lookup_impl_witness.b581a8.2, element0 [symbolic]
// CHECK:STDOUT:   %CppUnsafeDeref.lookup_impl_witness.87e543.2: <witness> = lookup_impl_witness %impl.elem0.5f971c.2, @CppUnsafeDeref [symbolic]
// CHECK:STDOUT:   %impl.elem0.493765.2: type = impl_witness_access %CppUnsafeDeref.lookup_impl_witness.87e543.2, element0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.type.953: type = fn_type @T.as_type.as.Iterate.impl.NewCursor, @T.as_type.as.Iterate.impl(%T.f45530.2) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.544: %T.as_type.as.Iterate.impl.NewCursor.type.953 = struct_value () [symbolic]
// CHECK:STDOUT:   %impl.elem1.c2dc15.2: type = impl_witness_access %CppRangeForIterate.lookup_impl_witness.b581a8.2, element1 [symbolic]
// CHECK:STDOUT:   %tuple.type.1ba: type = tuple_type (%impl.elem0.5f971c.2, %impl.elem1.c2dc15.2) [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.812: <witness> = lookup_impl_witness %tuple.type.1ba, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.9cd: %Destroy.type = facet_value %tuple.type.1ba, (%Destroy.lookup_impl_witness.812) [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.35816c.2: <witness> = lookup_impl_witness %impl.elem0.493765.2, @Destroy [symbolic]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.167: <witness> = lookup_impl_witness %impl.elem0.493765.2, @Copy [symbolic]
// CHECK:STDOUT:   %facet_value.f93: %facet_type.f48 = facet_value %impl.elem0.493765.2, (%Destroy.lookup_impl_witness.35816c.2, %Copy.lookup_impl_witness.167) [symbolic]
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete]
// CHECK:STDOUT:   %f64.dc1: type = class_type @Float, @Float(%int_64) [concrete]
// CHECK:STDOUT:   %pattern_type.fb7: type = pattern_type %f64.dc1 [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.3: type = fn_type @Destroy.Op.loc22_53.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.3: %Destroy.Op.type.1d8f74.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.df9cc1.3: <witness> = custom_witness (%Destroy.Op.1a2547.3), @Destroy [concrete]
// CHECK:STDOUT:   %Float.as.Copy.impl.Op.type.ff3: type = fn_type @Float.as.Copy.impl.Op, @Float.as.Copy.impl(%N.fe9) [symbolic]
// CHECK:STDOUT:   %Float.as.Copy.impl.Op.d93: %Float.as.Copy.impl.Op.type.ff3 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.0d6: <witness> = impl_witness imports.%Copy.impl_witness_table.8dd, @Float.as.Copy.impl(%int_64) [concrete]
// CHECK:STDOUT:   %facet_value.20d: %facet_type.f48 = facet_value %f64.dc1, (%custom_witness.df9cc1.3, %Copy.impl_witness.0d6) [concrete]
// CHECK:STDOUT:   %Iterate.lookup_impl_witness.1d8: <witness> = lookup_impl_witness %.Self.500, @Iterate [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.294: %facet_type.f48 = impl_witness_access %Iterate.lookup_impl_witness.1d8, element0 [symbolic_self]
// CHECK:STDOUT:   %Iterate_where.type.e6d: type = facet_type <@Iterate where %impl.elem0.294 = %facet_value.20d> [concrete]
// CHECK:STDOUT:   %pattern_type.dc5: type = pattern_type %Iterate_where.type.e6d [concrete]
// CHECK:STDOUT:   %R.patt: %pattern_type.dc5 = symbolic_binding_pattern R, 0 [symbolic]
// CHECK:STDOUT:   %R: %Iterate_where.type.e6d = symbolic_binding R, 0 [symbolic]
// CHECK:STDOUT:   %R.as_type: type = facet_access_type %R [symbolic]
// CHECK:STDOUT:   %pattern_type.d35: type = pattern_type %R.as_type [symbolic]
// CHECK:STDOUT:   %r.param_patt.bc2: %pattern_type.d35 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %r.patt.47e: %pattern_type.d35 = wrapper_binding_pattern r, %r.param_patt.bc2 [symbolic]
// CHECK:STDOUT:   %i.patt: %pattern_type.fb7 = value_binding_pattern i [concrete]
// CHECK:STDOUT:   %Iterate.lookup_impl_witness.757: <witness> = lookup_impl_witness %R, @Iterate [symbolic]
// CHECK:STDOUT:   %Iterate.facet.737: %Iterate.type = facet_value %R.as_type, (%Iterate.lookup_impl_witness.757) [symbolic]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.346: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.737) [symbolic]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.f8a: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.737) [symbolic]
// CHECK:STDOUT:   %.41f: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.346, %Iterate.facet.737 [symbolic]
// CHECK:STDOUT:   %impl.elem2.804: %.41f = impl_witness_access %Iterate.lookup_impl_witness.757, element2 [symbolic]
// CHECK:STDOUT:   %impl.elem1.e8e: %Destroy.type = impl_witness_access %Iterate.lookup_impl_witness.757, element1 [symbolic]
// CHECK:STDOUT:   %as_type.b98: type = facet_access_type %impl.elem1.e8e [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.a60: <specific function> = specific_impl_function %impl.elem2.804, @Iterate.WithSelf.NewCursor(%Iterate.facet.737) [symbolic]
// CHECK:STDOUT:   %ptr.f03: type = ptr_type %as_type.b98 [symbolic]
// CHECK:STDOUT:   %.ecb: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.f8a, %Iterate.facet.737 [symbolic]
// CHECK:STDOUT:   %impl.elem3.d3c: %.ecb = impl_witness_access %Iterate.lookup_impl_witness.757, element3 [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.8c0: type = class_type @DefaultOptionalStorage, @DefaultOptionalStorage(%facet_value.20d) [concrete]
// CHECK:STDOUT:   %OptionalStorage.impl_witness.872: <witness> = impl_witness imports.%OptionalStorage.impl_witness_table.460, @T.as_type.as.OptionalStorage.impl(%facet_value.20d) [concrete]
// CHECK:STDOUT:   %OptionalStorage.facet.4d5: %OptionalStorage.type = facet_value %f64.dc1, (%OptionalStorage.impl_witness.872) [concrete]
// CHECK:STDOUT:   %Optional.e07: type = class_type @Optional, @Optional(%OptionalStorage.facet.4d5) [concrete]
// CHECK:STDOUT:   %specific_impl_fn.b90: <specific function> = specific_impl_function %impl.elem3.d3c, @Iterate.WithSelf.Next(%Iterate.facet.737) [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.type.43a: type = fn_type @Optional.HasValue, @Optional(%OptionalStorage.facet.4d5) [concrete]
// CHECK:STDOUT:   %Optional.HasValue.1c0: %Optional.HasValue.type.43a = struct_value () [concrete]
// CHECK:STDOUT:   %Optional.Get.type.a84: type = fn_type @Optional.Get, @Optional(%OptionalStorage.facet.4d5) [concrete]
// CHECK:STDOUT:   %Optional.Get.4e0: %Optional.Get.type.a84 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.facet.339: %Destroy.type = facet_value %f64.dc1, (%custom_witness.df9cc1.3) [concrete]
// CHECK:STDOUT:   %MaybeUnformed.b7e: type = class_type @MaybeUnformed, @MaybeUnformed(%Destroy.facet.339) [concrete]
// CHECK:STDOUT:   %.cc0: type = maybe_unformed_type %f64.dc1 [concrete]
// CHECK:STDOUT:   %struct_type.value.has_value.ad9: type = struct_type {.value: %MaybeUnformed.b7e, .has_value: bool} [concrete]
// CHECK:STDOUT:   %Optional.HasValue.specific_fn.0a6: <specific function> = specific_function %Optional.HasValue.1c0, @Optional.HasValue(%OptionalStorage.facet.4d5) [concrete]
// CHECK:STDOUT:   %Optional.Get.specific_fn.3a1: <specific function> = specific_function %Optional.Get.4e0, @Optional.Get(%OptionalStorage.facet.4d5) [concrete]
// CHECK:STDOUT:   %AddAssignWith.type.ff9: type = facet_type <@AddAssignWith, @AddAssignWith(%f64.dc1)> [concrete]
// CHECK:STDOUT:   %Float.as.AddAssignWith.impl.Op.type.fdd: type = fn_type @Float.as.AddAssignWith.impl.Op, @Float.as.AddAssignWith.impl(%N.fe9) [symbolic]
// CHECK:STDOUT:   %Float.as.AddAssignWith.impl.Op.430: %Float.as.AddAssignWith.impl.Op.type.fdd = struct_value () [symbolic]
// CHECK:STDOUT:   %AddAssignWith.impl_witness.402: <witness> = impl_witness imports.%AddAssignWith.impl_witness_table, @Float.as.AddAssignWith.impl(%int_64) [concrete]
// CHECK:STDOUT:   %Float.as.AddAssignWith.impl.Op.type.a92: type = fn_type @Float.as.AddAssignWith.impl.Op, @Float.as.AddAssignWith.impl(%int_64) [concrete]
// CHECK:STDOUT:   %Float.as.AddAssignWith.impl.Op.ac1: %Float.as.AddAssignWith.impl.Op.type.a92 = struct_value () [concrete]
// CHECK:STDOUT:   %AddAssignWith.facet: %AddAssignWith.type.ff9 = facet_value %f64.dc1, (%AddAssignWith.impl_witness.402) [concrete]
// CHECK:STDOUT:   %AddAssignWith.WithSelf.Op.type.d04: type = fn_type @AddAssignWith.WithSelf.Op, @AddAssignWith.WithSelf(%f64.dc1, %AddAssignWith.facet) [concrete]
// CHECK:STDOUT:   %.918: type = fn_type_with_self_type %AddAssignWith.WithSelf.Op.type.d04, %AddAssignWith.facet [concrete]
// CHECK:STDOUT:   %Float.as.AddAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Float.as.AddAssignWith.impl.Op.ac1, @Float.as.AddAssignWith.impl.Op(%int_64) [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.9: type = fn_type @Destroy.Op.loc26_19.6 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.9: %Destroy.Op.type.1d8f74.9 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.02f: <witness> = lookup_impl_witness %impl.elem1.e8e, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.f26: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%impl.elem1.e8e) [symbolic]
// CHECK:STDOUT:   %.f38: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.f26, %impl.elem1.e8e [symbolic]
// CHECK:STDOUT:   %impl.elem0.54b: %.f38 = impl_witness_access %Destroy.lookup_impl_witness.02f, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.b70: <specific function> = specific_impl_function %impl.elem0.54b, @Destroy.WithSelf.Op(%impl.elem1.e8e) [symbolic]
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
// CHECK:STDOUT:   %custom_witness.491: <witness> = custom_witness (%Iterator.Op.ac6aad.3), @Inc [concrete]
// CHECK:STDOUT:   %Equal.type: type = fn_type @Equal [concrete]
// CHECK:STDOUT:   %Equal: %Equal.type = struct_value () [concrete]
// CHECK:STDOUT:   %NotEqual.type: type = fn_type @NotEqual [concrete]
// CHECK:STDOUT:   %NotEqual: %NotEqual.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.ed8: <witness> = custom_witness (%Equal, %NotEqual), @EqWith, @EqWith(%Iterator) [concrete]
// CHECK:STDOUT:   %facet_value.bed: %CppRange_where.type.d682d6.1 = facet_value %MutableRange, (%custom_witness.df9cc1.3, %Copy.impl_witness.0d6, %custom_witness.902, %custom_witness.c08, %custom_witness.05b, %custom_witness.491, %custom_witness.ed8) [concrete]
// CHECK:STDOUT:   %tuple.type.dd0: type = tuple_type (%Iterator, %Iterator) [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.10: type = fn_type @Destroy.Op.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.10: %Destroy.Op.type.1d8f74.10 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.df9cc1.10: <witness> = custom_witness (%Destroy.Op.1a2547.10), @Destroy [concrete]
// CHECK:STDOUT:   %Destroy.facet.e1e: %Destroy.type = facet_value %tuple.type.dd0, (%custom_witness.df9cc1.10) [concrete]
// CHECK:STDOUT:   %Iterate.impl_witness.15e: <witness> = impl_witness imports.%Iterate.impl_witness_table.121, @T.as_type.as.Iterate.impl(%facet_value.bed) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.type.ed2: type = fn_type @T.as_type.as.Iterate.impl.NewCursor, @T.as_type.as.Iterate.impl(%facet_value.bed) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.f6c: %T.as_type.as.Iterate.impl.NewCursor.type.ed2 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.type.ea9: type = fn_type @T.as_type.as.Iterate.impl.Next, @T.as_type.as.Iterate.impl(%facet_value.bed) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.630: %T.as_type.as.Iterate.impl.Next.type.ea9 = struct_value () [concrete]
// CHECK:STDOUT:   %Iterate.facet.24a: %Iterate.type = facet_value %MutableRange, (%Iterate.impl_witness.15e) [concrete]
// CHECK:STDOUT:   %facet_value.6bc: %Iterate_where.type.e6d = facet_value %MutableRange, (%Iterate.impl_witness.15e) [concrete]
// CHECK:STDOUT:   %r.param_patt.aeb: %pattern_type.992 = value_param_pattern [concrete]
// CHECK:STDOUT:   %r.patt.892: %pattern_type.992 = wrapper_binding_pattern r, %r.param_patt.aeb [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.4f1: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.24a) [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.e92: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.24a) [concrete]
// CHECK:STDOUT:   %.534: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.4f1, %Iterate.facet.24a [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.specific_fn: <specific function> = specific_function %T.as_type.as.Iterate.impl.NewCursor.f6c, @T.as_type.as.Iterate.impl.NewCursor(%facet_value.bed) [concrete]
// CHECK:STDOUT:   %ptr.a4f: type = ptr_type %tuple.type.dd0 [concrete]
// CHECK:STDOUT:   %.8b9: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.e92, %Iterate.facet.24a [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.specific_fn: <specific function> = specific_function %T.as_type.as.Iterate.impl.Next.630, @T.as_type.as.Iterate.impl.Next(%facet_value.bed) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.452: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.e1e) [concrete]
// CHECK:STDOUT:   %.899: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.452, %Destroy.facet.e1e [concrete]
// CHECK:STDOUT:   %complete_type.843: <witness> = complete_type_witness %tuple.type.dd0 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.bb6: @Optional.%Optional.HasValue.type (%Optional.HasValue.type.e9e) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.HasValue (constants.%Optional.HasValue.6cd)]
// CHECK:STDOUT:   %Core.import_ref.dab: @Optional.%Optional.Get.type (%Optional.Get.type.bb9) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.Get (constants.%Optional.Get.fa1)]
// CHECK:STDOUT:   %Core.import_ref.0c6: type = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%DefaultOptionalStorage (constants.%DefaultOptionalStorage.d50)]
// CHECK:STDOUT:   %Core.import_ref.dc3: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.None.type (%T.as_type.as.OptionalStorage.impl.None.type.0a1) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.None (constants.%T.as_type.as.OptionalStorage.impl.None.e06)]
// CHECK:STDOUT:   %Core.import_ref.c530: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Some.type (%T.as_type.as.OptionalStorage.impl.Some.type.492) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Some (constants.%T.as_type.as.OptionalStorage.impl.Some.89a)]
// CHECK:STDOUT:   %Core.import_ref.ed4: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Has.type (%T.as_type.as.OptionalStorage.impl.Has.type.e5a) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Has (constants.%T.as_type.as.OptionalStorage.impl.Has.4ee)]
// CHECK:STDOUT:   %Core.import_ref.e9c: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Get.type (%T.as_type.as.OptionalStorage.impl.Get.type.a12) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Get (constants.%T.as_type.as.OptionalStorage.impl.Get.d32)]
// CHECK:STDOUT:   %Core.import_ref.0b2 = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], unloaded
// CHECK:STDOUT:   %OptionalStorage.impl_witness_table.460 = impl_witness_table (%Core.import_ref.0c6, %Core.import_ref.dc3, %Core.import_ref.c530, %Core.import_ref.ed4, %Core.import_ref.e9c, %Core.import_ref.0b2), @T.as_type.as.OptionalStorage.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.eef: %facet_type.f48 = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.Iterate.impl.%facet_value (constants.%facet_value.f93)]
// CHECK:STDOUT:   %Core.import_ref.fcc: %Destroy.type = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.Iterate.impl.%Destroy.facet (constants.%Destroy.facet.9cd)]
// CHECK:STDOUT:   %Core.import_ref.fd1a: @T.as_type.as.Iterate.impl.%T.as_type.as.Iterate.impl.NewCursor.type (%T.as_type.as.Iterate.impl.NewCursor.type.953) = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.Iterate.impl.%T.as_type.as.Iterate.impl.NewCursor (constants.%T.as_type.as.Iterate.impl.NewCursor.544)]
// CHECK:STDOUT:   %Core.import_ref.273: @T.as_type.as.Iterate.impl.%T.as_type.as.Iterate.impl.Next.type (%T.as_type.as.Iterate.impl.Next.type.b36) = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.Iterate.impl.%T.as_type.as.Iterate.impl.Next (constants.%T.as_type.as.Iterate.impl.Next.d6b)]
// CHECK:STDOUT:   %Iterate.impl_witness_table.121 = impl_witness_table (%Core.import_ref.eef, %Core.import_ref.fcc, %Core.import_ref.fd1a, %Core.import_ref.273), @T.as_type.as.Iterate.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.e69: @Float.as.Copy.impl.%Float.as.Copy.impl.Op.type (%Float.as.Copy.impl.Op.type.ff3) = import_ref Core//prelude/types/float, loc{{\d+_\d+}}, loaded [symbolic = @Float.as.Copy.impl.%Float.as.Copy.impl.Op (constants.%Float.as.Copy.impl.Op.d93)]
// CHECK:STDOUT:   %Copy.impl_witness_table.8dd = impl_witness_table (%Core.import_ref.e69), @Float.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.cb3: @Float.as.AddAssignWith.impl.%Float.as.AddAssignWith.impl.Op.type (%Float.as.AddAssignWith.impl.Op.type.fdd) = import_ref Core//prelude/types/float, loc{{\d+_\d+}}, loaded [symbolic = @Float.as.AddAssignWith.impl.%Float.as.AddAssignWith.impl.Op (constants.%Float.as.AddAssignWith.impl.Op.430)]
// CHECK:STDOUT:   %AddAssignWith.impl_witness_table = impl_witness_table (%Core.import_ref.cb3), @Float.as.AddAssignWith.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @TestIterate(%R.loc22_17.2: %Iterate_where.type.e6d) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Iterate.lookup_impl_witness: <witness> = lookup_impl_witness %R.loc22_17.1, @Iterate [symbolic = %Iterate.lookup_impl_witness (constants.%Iterate.lookup_impl_witness.757)]
// CHECK:STDOUT:   %Iterate.facet.loc26_19.5: %Iterate.type = facet_value %R.as_type.loc22_61.1, (%Iterate.lookup_impl_witness) [symbolic = %Iterate.facet.loc26_19.5 (constants.%Iterate.facet.737)]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.loc26_19.5) [symbolic = %Iterate.WithSelf.NewCursor.type (constants.%Iterate.WithSelf.NewCursor.type.346)]
// CHECK:STDOUT:   %.loc26_19.19: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type, %Iterate.facet.loc26_19.5 [symbolic = %.loc26_19.19 (constants.%.41f)]
// CHECK:STDOUT:   %impl.elem2.loc26_19.2: @TestIterate.%.loc26_19.19 (%.41f) = impl_witness_access %Iterate.lookup_impl_witness, element2 [symbolic = %impl.elem2.loc26_19.2 (constants.%impl.elem2.804)]
// CHECK:STDOUT:   %specific_impl_fn.loc26_19.4: <specific function> = specific_impl_function %impl.elem2.loc26_19.2, @Iterate.WithSelf.NewCursor(%Iterate.facet.loc26_19.5) [symbolic = %specific_impl_fn.loc26_19.4 (constants.%specific_impl_fn.a60)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %ptr: type = ptr_type %as_type [symbolic = %ptr (constants.%ptr.f03)]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.loc26_19.5) [symbolic = %Iterate.WithSelf.Next.type (constants.%Iterate.WithSelf.Next.type.f8a)]
// CHECK:STDOUT:   %.loc26_19.20: type = fn_type_with_self_type %Iterate.WithSelf.Next.type, %Iterate.facet.loc26_19.5 [symbolic = %.loc26_19.20 (constants.%.ecb)]
// CHECK:STDOUT:   %impl.elem3.loc26_19.2: @TestIterate.%.loc26_19.20 (%.ecb) = impl_witness_access %Iterate.lookup_impl_witness, element3 [symbolic = %impl.elem3.loc26_19.2 (constants.%impl.elem3.d3c)]
// CHECK:STDOUT:   %specific_impl_fn.loc26_19.5: <specific function> = specific_impl_function %impl.elem3.loc26_19.2, @Iterate.WithSelf.Next(%Iterate.facet.loc26_19.5) [symbolic = %specific_impl_fn.loc26_19.5 (constants.%specific_impl_fn.b90)]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%impl.elem1) [symbolic = %Destroy.WithSelf.Op.type (constants.%Destroy.WithSelf.Op.type.f26)]
// CHECK:STDOUT:   %.loc26_19.21: type = fn_type_with_self_type %Destroy.WithSelf.Op.type, %impl.elem1 [symbolic = %.loc26_19.21 (constants.%.f38)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %impl.elem1, @Destroy [symbolic = %Destroy.lookup_impl_witness (constants.%Destroy.lookup_impl_witness.02f)]
// CHECK:STDOUT:   %impl.elem0.loc26_19.2: @TestIterate.%.loc26_19.21 (%.f38) = impl_witness_access %Destroy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc26_19.2 (constants.%impl.elem0.54b)]
// CHECK:STDOUT:   %specific_impl_fn.loc26_19.6: <specific function> = specific_impl_function %impl.elem0.loc26_19.2, @Destroy.WithSelf.Op(%impl.elem1) [symbolic = %specific_impl_fn.loc26_19.6 (constants.%specific_impl_fn.b70)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%r.param: @TestIterate.%R.as_type.loc22_61.1 (%R.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %i.patt: %pattern_type.fb7 = value_binding_pattern i [concrete = constants.%i.patt]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %r.ref: @TestIterate.%R.as_type.loc22_61.1 (%R.as_type) = name_ref r, %r
// CHECK:STDOUT:     %impl.elem2.loc26_19.1: @TestIterate.%.loc26_19.19 (%.41f) = impl_witness_access constants.%Iterate.lookup_impl_witness.757, element2 [symbolic = %impl.elem2.loc26_19.2 (constants.%impl.elem2.804)]
// CHECK:STDOUT:     %bound_method.loc26_19.1: <bound method> = bound_method %r.ref, %impl.elem2.loc26_19.1
// CHECK:STDOUT:     %Iterate.facet.loc26_19.1: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.757) [symbolic = %Iterate.facet.loc26_19.5 (constants.%Iterate.facet.737)]
// CHECK:STDOUT:     %.loc26_19.1: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc26_19.1 [symbolic = %Iterate.facet.loc26_19.5 (constants.%Iterate.facet.737)]
// CHECK:STDOUT:     %Iterate.facet.loc26_19.2: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.757) [symbolic = %Iterate.facet.loc26_19.5 (constants.%Iterate.facet.737)]
// CHECK:STDOUT:     %.loc26_19.2: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc26_19.2 [symbolic = %Iterate.facet.loc26_19.5 (constants.%Iterate.facet.737)]
// CHECK:STDOUT:     %specific_impl_fn.loc26_19.1: <specific function> = specific_impl_function %impl.elem2.loc26_19.1, @Iterate.WithSelf.NewCursor(constants.%Iterate.facet.737) [symbolic = %specific_impl_fn.loc26_19.4 (constants.%specific_impl_fn.a60)]
// CHECK:STDOUT:     %bound_method.loc26_19.2: <bound method> = bound_method %r.ref, %specific_impl_fn.loc26_19.1
// CHECK:STDOUT:     %var: ref @TestIterate.%as_type (%as_type.b98) = var_storage invalid
// CHECK:STDOUT:     %Iterate.WithSelf.NewCursor.call: init @TestIterate.%as_type (%as_type.b98) to %var = call %bound_method.loc26_19.2(%r.ref)
// CHECK:STDOUT:     assign %var, %Iterate.WithSelf.NewCursor.call
// CHECK:STDOUT:     br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.next:
// CHECK:STDOUT:     %addr: @TestIterate.%ptr (%ptr.f03) = addr_of %var
// CHECK:STDOUT:     %impl.elem3.loc26_19.1: @TestIterate.%.loc26_19.20 (%.ecb) = impl_witness_access constants.%Iterate.lookup_impl_witness.757, element3 [symbolic = %impl.elem3.loc26_19.2 (constants.%impl.elem3.d3c)]
// CHECK:STDOUT:     %bound_method.loc26_19.3: <bound method> = bound_method %r.ref, %impl.elem3.loc26_19.1
// CHECK:STDOUT:     %Iterate.facet.loc26_19.3: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.757) [symbolic = %Iterate.facet.loc26_19.5 (constants.%Iterate.facet.737)]
// CHECK:STDOUT:     %.loc26_19.3: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc26_19.3 [symbolic = %Iterate.facet.loc26_19.5 (constants.%Iterate.facet.737)]
// CHECK:STDOUT:     %Iterate.facet.loc26_19.4: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.757) [symbolic = %Iterate.facet.loc26_19.5 (constants.%Iterate.facet.737)]
// CHECK:STDOUT:     %.loc26_19.4: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc26_19.4 [symbolic = %Iterate.facet.loc26_19.5 (constants.%Iterate.facet.737)]
// CHECK:STDOUT:     %.loc26_19.5: %Destroy.type = converted constants.%as_type.b98, constants.%impl.elem1.e8e [symbolic = %impl.elem1 (constants.%impl.elem1.e8e)]
// CHECK:STDOUT:     %.loc26_19.6: %Destroy.type = converted constants.%as_type.b98, constants.%impl.elem1.e8e [symbolic = %impl.elem1 (constants.%impl.elem1.e8e)]
// CHECK:STDOUT:     %specific_impl_fn.loc26_19.2: <specific function> = specific_impl_function %impl.elem3.loc26_19.1, @Iterate.WithSelf.Next(constants.%Iterate.facet.737) [symbolic = %specific_impl_fn.loc26_19.5 (constants.%specific_impl_fn.b90)]
// CHECK:STDOUT:     %bound_method.loc26_19.4: <bound method> = bound_method %r.ref, %specific_impl_fn.loc26_19.2
// CHECK:STDOUT:     %.loc26_19.7: ref %Optional.e07 = temporary_storage
// CHECK:STDOUT:     %Iterate.WithSelf.Next.call: init %Optional.e07 to %.loc26_19.7 = call %bound_method.loc26_19.4(%r.ref, %addr)
// CHECK:STDOUT:     %.loc26_19.8: ref %Optional.e07 = temporary %.loc26_19.7, %Iterate.WithSelf.Next.call
// CHECK:STDOUT:     %.loc26_19.9: %Optional.HasValue.type.43a = specific_constant imports.%Core.import_ref.bb6, @Optional(constants.%OptionalStorage.facet.4d5) [concrete = constants.%Optional.HasValue.1c0]
// CHECK:STDOUT:     %HasValue.ref: %Optional.HasValue.type.43a = name_ref HasValue, %.loc26_19.9 [concrete = constants.%Optional.HasValue.1c0]
// CHECK:STDOUT:     %Optional.HasValue.bound: <bound method> = bound_method %.loc26_19.8, %HasValue.ref
// CHECK:STDOUT:     %Optional.HasValue.specific_fn: <specific function> = specific_function %HasValue.ref, @Optional.HasValue(constants.%OptionalStorage.facet.4d5) [concrete = constants.%Optional.HasValue.specific_fn.0a6]
// CHECK:STDOUT:     %bound_method.loc26_19.5: <bound method> = bound_method %.loc26_19.8, %Optional.HasValue.specific_fn
// CHECK:STDOUT:     %.loc26_19.10: %Optional.e07 = acquire_value %.loc26_19.8
// CHECK:STDOUT:     %Optional.HasValue.call: init bool = call %bound_method.loc26_19.5(%.loc26_19.10)
// CHECK:STDOUT:     %.loc26_19.11: bool = value_of_initializer %Optional.HasValue.call
// CHECK:STDOUT:     %.loc26_19.12: bool = converted %Optional.HasValue.call, %.loc26_19.11
// CHECK:STDOUT:     if %.loc26_19.12 br !for.body else br !for.done
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.body:
// CHECK:STDOUT:     %.loc26_19.13: %Optional.Get.type.a84 = specific_constant imports.%Core.import_ref.dab, @Optional(constants.%OptionalStorage.facet.4d5) [concrete = constants.%Optional.Get.4e0]
// CHECK:STDOUT:     %Get.ref: %Optional.Get.type.a84 = name_ref Get, %.loc26_19.13 [concrete = constants.%Optional.Get.4e0]
// CHECK:STDOUT:     %Optional.Get.bound: <bound method> = bound_method %.loc26_19.8, %Get.ref
// CHECK:STDOUT:     %Optional.Get.specific_fn: <specific function> = specific_function %Get.ref, @Optional.Get(constants.%OptionalStorage.facet.4d5) [concrete = constants.%Optional.Get.specific_fn.3a1]
// CHECK:STDOUT:     %bound_method.loc26_19.6: <bound method> = bound_method %.loc26_19.8, %Optional.Get.specific_fn
// CHECK:STDOUT:     %.loc26_19.14: %Optional.e07 = acquire_value %.loc26_19.8
// CHECK:STDOUT:     %Optional.Get.call: init %f64.dc1 = call %bound_method.loc26_19.6(%.loc26_19.14)
// CHECK:STDOUT:     %.loc26_19.15: %f64.dc1 = value_of_initializer %Optional.Get.call
// CHECK:STDOUT:     %.loc26_19.16: %f64.dc1 = converted %Optional.Get.call, %.loc26_19.15
// CHECK:STDOUT:     %f64.loc26: type = type_literal constants.%f64.dc1 [concrete = constants.%f64.dc1]
// CHECK:STDOUT:     %i: %f64.dc1 = wrapper_binding i, %.loc26_19.16
// CHECK:STDOUT:     %sum.ref: ref %f64.dc1 = name_ref sum, %sum
// CHECK:STDOUT:     %i.ref: %f64.dc1 = name_ref i, %i
// CHECK:STDOUT:     %impl.elem0.loc27: %.918 = impl_witness_access constants.%AddAssignWith.impl_witness.402, element0 [concrete = constants.%Float.as.AddAssignWith.impl.Op.ac1]
// CHECK:STDOUT:     %bound_method.loc27_9.1: <bound method> = bound_method %sum.ref, %impl.elem0.loc27
// CHECK:STDOUT:     %specific_fn.loc27: <specific function> = specific_function %impl.elem0.loc27, @Float.as.AddAssignWith.impl.Op(constants.%int_64) [concrete = constants.%Float.as.AddAssignWith.impl.Op.specific_fn]
// CHECK:STDOUT:     %bound_method.loc27_9.2: <bound method> = bound_method %sum.ref, %specific_fn.loc27
// CHECK:STDOUT:     %Float.as.AddAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc27_9.2(%sum.ref, %i.ref)
// CHECK:STDOUT:     %Destroy.Op.bound.loc26_19.1: <bound method> = bound_method %.loc26_19.8, constants.%Destroy.Op.1a2547.9
// CHECK:STDOUT:     %Destroy.Op.call.loc26_19.1: init %empty_tuple.type = call %Destroy.Op.bound.loc26_19.1(%.loc26_19.8)
// CHECK:STDOUT:     br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.done:
// CHECK:STDOUT:     %Destroy.Op.bound.loc26_19.2: <bound method> = bound_method %.loc26_19.8, constants.%Destroy.Op.1a2547.9
// CHECK:STDOUT:     %Destroy.Op.call.loc26_19.2: init %empty_tuple.type = call %Destroy.Op.bound.loc26_19.2(%.loc26_19.8)
// CHECK:STDOUT:     %impl.elem0.loc26_19.1: @TestIterate.%.loc26_19.21 (%.f38) = impl_witness_access constants.%Destroy.lookup_impl_witness.02f, element0 [symbolic = %impl.elem0.loc26_19.2 (constants.%impl.elem0.54b)]
// CHECK:STDOUT:     %bound_method.loc26_19.7: <bound method> = bound_method %var, %impl.elem0.loc26_19.1
// CHECK:STDOUT:     %.loc26_19.17: %Destroy.type = converted constants.%as_type.b98, constants.%impl.elem1.e8e [symbolic = %impl.elem1 (constants.%impl.elem1.e8e)]
// CHECK:STDOUT:     %.loc26_19.18: %Destroy.type = converted constants.%as_type.b98, constants.%impl.elem1.e8e [symbolic = %impl.elem1 (constants.%impl.elem1.e8e)]
// CHECK:STDOUT:     %specific_impl_fn.loc26_19.3: <specific function> = specific_impl_function %impl.elem0.loc26_19.1, @Destroy.WithSelf.Op(constants.%impl.elem1.e8e) [symbolic = %specific_impl_fn.loc26_19.6 (constants.%specific_impl_fn.b70)]
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
// CHECK:STDOUT: fn @Destroy.Op.loc26_19.5(%self.param: ref %DefaultOptionalStorage.8c0) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc26_19.6(%self.param: ref %Optional.e07) {
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
// CHECK:STDOUT:   %impl.elem2: %.534 = impl_witness_access constants.%Iterate.impl_witness.15e, element2 [concrete = constants.%T.as_type.as.Iterate.impl.NewCursor.f6c]
// CHECK:STDOUT:   %bound_method.loc68_19.1: <bound method> = bound_method %m.ref, %impl.elem2
// CHECK:STDOUT:   %facet_value.loc68_19.1: %CppRange_where.type.d682d6.1 = facet_value constants.%MutableRange, (constants.%custom_witness.df9cc1.3, constants.%Copy.impl_witness.0d6, constants.%custom_witness.902, constants.%custom_witness.c08, constants.%custom_witness.05b, constants.%custom_witness.491, constants.%custom_witness.ed8) [concrete = constants.%facet_value.bed]
// CHECK:STDOUT:   %.loc68_19.1: %CppRange_where.type.d682d6.1 = converted constants.%MutableRange, %facet_value.loc68_19.1 [concrete = constants.%facet_value.bed]
// CHECK:STDOUT:   %facet_value.loc68_19.2: %CppRange_where.type.d682d6.1 = facet_value constants.%MutableRange, (constants.%custom_witness.df9cc1.3, constants.%Copy.impl_witness.0d6, constants.%custom_witness.902, constants.%custom_witness.c08, constants.%custom_witness.05b, constants.%custom_witness.491, constants.%custom_witness.ed8) [concrete = constants.%facet_value.bed]
// CHECK:STDOUT:   %.loc68_19.2: %CppRange_where.type.d682d6.1 = converted constants.%MutableRange, %facet_value.loc68_19.2 [concrete = constants.%facet_value.bed]
// CHECK:STDOUT:   %specific_fn.loc68_19.1: <specific function> = specific_function %impl.elem2, @T.as_type.as.Iterate.impl.NewCursor(constants.%facet_value.bed) [concrete = constants.%T.as_type.as.Iterate.impl.NewCursor.specific_fn]
// CHECK:STDOUT:   %bound_method.loc68_19.2: <bound method> = bound_method %m.ref, %specific_fn.loc68_19.1
// CHECK:STDOUT:   %var: ref %tuple.type.dd0 = var_storage invalid
// CHECK:STDOUT:   %.loc68_18.1: %MutableRange = acquire_value %m.ref
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.call: init %tuple.type.dd0 to %var = call %bound_method.loc68_19.2(%.loc68_18.1)
// CHECK:STDOUT:   assign %var, %T.as_type.as.Iterate.impl.NewCursor.call
// CHECK:STDOUT:   br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT: !for.next:
// CHECK:STDOUT:   %addr: %ptr.a4f = addr_of %var
// CHECK:STDOUT:   %impl.elem3: %.8b9 = impl_witness_access constants.%Iterate.impl_witness.15e, element3 [concrete = constants.%T.as_type.as.Iterate.impl.Next.630]
// CHECK:STDOUT:   %bound_method.loc68_19.3: <bound method> = bound_method %m.ref, %impl.elem3
// CHECK:STDOUT:   %facet_value.loc68_19.3: %CppRange_where.type.d682d6.1 = facet_value constants.%MutableRange, (constants.%custom_witness.df9cc1.3, constants.%Copy.impl_witness.0d6, constants.%custom_witness.902, constants.%custom_witness.c08, constants.%custom_witness.05b, constants.%custom_witness.491, constants.%custom_witness.ed8) [concrete = constants.%facet_value.bed]
// CHECK:STDOUT:   %.loc68_19.3: %CppRange_where.type.d682d6.1 = converted constants.%MutableRange, %facet_value.loc68_19.3 [concrete = constants.%facet_value.bed]
// CHECK:STDOUT:   %facet_value.loc68_19.4: %CppRange_where.type.d682d6.1 = facet_value constants.%MutableRange, (constants.%custom_witness.df9cc1.3, constants.%Copy.impl_witness.0d6, constants.%custom_witness.902, constants.%custom_witness.c08, constants.%custom_witness.05b, constants.%custom_witness.491, constants.%custom_witness.ed8) [concrete = constants.%facet_value.bed]
// CHECK:STDOUT:   %.loc68_19.4: %CppRange_where.type.d682d6.1 = converted constants.%MutableRange, %facet_value.loc68_19.4 [concrete = constants.%facet_value.bed]
// CHECK:STDOUT:   %specific_fn.loc68_19.2: <specific function> = specific_function %impl.elem3, @T.as_type.as.Iterate.impl.Next(constants.%facet_value.bed) [concrete = constants.%T.as_type.as.Iterate.impl.Next.specific_fn]
// CHECK:STDOUT:   %bound_method.loc68_19.4: <bound method> = bound_method %m.ref, %specific_fn.loc68_19.2
// CHECK:STDOUT:   %.loc68_19.5: ref %Optional.e07 = temporary_storage
// CHECK:STDOUT:   %.loc68_18.2: %MutableRange = acquire_value %m.ref
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.call: init %Optional.e07 to %.loc68_19.5 = call %bound_method.loc68_19.4(%.loc68_18.2, %addr)
// CHECK:STDOUT:   %.loc68_19.6: ref %Optional.e07 = temporary %.loc68_19.5, %T.as_type.as.Iterate.impl.Next.call
// CHECK:STDOUT:   %.loc68_19.7: %Optional.HasValue.type.43a = specific_constant imports.%Core.import_ref.bb6, @Optional(constants.%OptionalStorage.facet.4d5) [concrete = constants.%Optional.HasValue.1c0]
// CHECK:STDOUT:   %HasValue.ref: %Optional.HasValue.type.43a = name_ref HasValue, %.loc68_19.7 [concrete = constants.%Optional.HasValue.1c0]
// CHECK:STDOUT:   %Optional.HasValue.bound: <bound method> = bound_method %.loc68_19.6, %HasValue.ref
// CHECK:STDOUT:   %Optional.HasValue.specific_fn: <specific function> = specific_function %HasValue.ref, @Optional.HasValue(constants.%OptionalStorage.facet.4d5) [concrete = constants.%Optional.HasValue.specific_fn.0a6]
// CHECK:STDOUT:   %bound_method.loc68_19.5: <bound method> = bound_method %.loc68_19.6, %Optional.HasValue.specific_fn
// CHECK:STDOUT:   %.loc68_19.8: %Optional.e07 = acquire_value %.loc68_19.6
// CHECK:STDOUT:   %Optional.HasValue.call: init bool = call %bound_method.loc68_19.5(%.loc68_19.8)
// CHECK:STDOUT:   %.loc68_19.9: bool = value_of_initializer %Optional.HasValue.call
// CHECK:STDOUT:   %.loc68_19.10: bool = converted %Optional.HasValue.call, %.loc68_19.9
// CHECK:STDOUT:   if %.loc68_19.10 br !for.body else br !for.done
// CHECK:STDOUT:
// CHECK:STDOUT: !for.body:
// CHECK:STDOUT:   %.loc68_19.11: %Optional.Get.type.a84 = specific_constant imports.%Core.import_ref.dab, @Optional(constants.%OptionalStorage.facet.4d5) [concrete = constants.%Optional.Get.4e0]
// CHECK:STDOUT:   %Get.ref: %Optional.Get.type.a84 = name_ref Get, %.loc68_19.11 [concrete = constants.%Optional.Get.4e0]
// CHECK:STDOUT:   %Optional.Get.bound: <bound method> = bound_method %.loc68_19.6, %Get.ref
// CHECK:STDOUT:   %Optional.Get.specific_fn: <specific function> = specific_function %Get.ref, @Optional.Get(constants.%OptionalStorage.facet.4d5) [concrete = constants.%Optional.Get.specific_fn.3a1]
// CHECK:STDOUT:   %bound_method.loc68_19.6: <bound method> = bound_method %.loc68_19.6, %Optional.Get.specific_fn
// CHECK:STDOUT:   %.loc68_19.12: %Optional.e07 = acquire_value %.loc68_19.6
// CHECK:STDOUT:   %Optional.Get.call: init %f64.dc1 = call %bound_method.loc68_19.6(%.loc68_19.12)
// CHECK:STDOUT:   %.loc68_19.13: %f64.dc1 = value_of_initializer %Optional.Get.call
// CHECK:STDOUT:   %.loc68_19.14: %f64.dc1 = converted %Optional.Get.call, %.loc68_19.13
// CHECK:STDOUT:   %f64.loc68: type = type_literal constants.%f64.dc1 [concrete = constants.%f64.dc1]
// CHECK:STDOUT:   %i: %f64.dc1 = wrapper_binding i, %.loc68_19.14
// CHECK:STDOUT:   %sum.ref: ref %f64.dc1 = name_ref sum, %sum
// CHECK:STDOUT:   %i.ref: %f64.dc1 = name_ref i, %i
// CHECK:STDOUT:   %impl.elem0.loc69: %.918 = impl_witness_access constants.%AddAssignWith.impl_witness.402, element0 [concrete = constants.%Float.as.AddAssignWith.impl.Op.ac1]
// CHECK:STDOUT:   %bound_method.loc69_9.1: <bound method> = bound_method %sum.ref, %impl.elem0.loc69
// CHECK:STDOUT:   %specific_fn.loc69: <specific function> = specific_function %impl.elem0.loc69, @Float.as.AddAssignWith.impl.Op(constants.%int_64) [concrete = constants.%Float.as.AddAssignWith.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc69_9.2: <bound method> = bound_method %sum.ref, %specific_fn.loc69
// CHECK:STDOUT:   %Float.as.AddAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc69_9.2(%sum.ref, %i.ref)
// CHECK:STDOUT:   %Destroy.Op.bound.loc68_19.1: <bound method> = bound_method %.loc68_19.6, constants.%Destroy.Op.1a2547.9
// CHECK:STDOUT:   %Destroy.Op.call.loc68_19.1: init %empty_tuple.type = call %Destroy.Op.bound.loc68_19.1(%.loc68_19.6)
// CHECK:STDOUT:   br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT: !for.done:
// CHECK:STDOUT:   %Destroy.Op.bound.loc68_19.2: <bound method> = bound_method %.loc68_19.6, constants.%Destroy.Op.1a2547.9
// CHECK:STDOUT:   %Destroy.Op.call.loc68_19.2: init %empty_tuple.type = call %Destroy.Op.bound.loc68_19.2(%.loc68_19.6)
// CHECK:STDOUT:   %Destroy.Op.bound.loc68_19.3: <bound method> = bound_method %var, constants.%Destroy.Op.1a2547.10
// CHECK:STDOUT:   %Destroy.Op.call.loc68_19.3: init %empty_tuple.type = call %Destroy.Op.bound.loc68_19.3(%var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestIterate(constants.%R) {
// CHECK:STDOUT:   %R.patt.loc22_17.2 => constants.%R.patt
// CHECK:STDOUT:   %R.loc22_17.1 => constants.%R
// CHECK:STDOUT:   %R.as_type.loc22_61.1 => constants.%R.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.d35
// CHECK:STDOUT:   %r.param_patt.loc22_59.2 => constants.%r.param_patt.bc2
// CHECK:STDOUT:   %r.patt.loc22_59.2 => constants.%r.patt.47e
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestIterate(constants.%facet_value.6bc) {
// CHECK:STDOUT:   %R.patt.loc22_17.2 => constants.%R.patt
// CHECK:STDOUT:   %R.loc22_17.1 => constants.%facet_value.6bc
// CHECK:STDOUT:   %R.as_type.loc22_61.1 => constants.%MutableRange
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.992
// CHECK:STDOUT:   %r.param_patt.loc22_59.2 => constants.%r.param_patt.aeb
// CHECK:STDOUT:   %r.patt.loc22_59.2 => constants.%r.patt.892
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc22 => constants.%complete_type.357
// CHECK:STDOUT:   %Iterate.lookup_impl_witness => constants.%Iterate.impl_witness.15e
// CHECK:STDOUT:   %Iterate.facet.loc26_19.5 => constants.%Iterate.facet.24a
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type => constants.%Iterate.WithSelf.NewCursor.type.4f1
// CHECK:STDOUT:   %.loc26_19.19 => constants.%.534
// CHECK:STDOUT:   %impl.elem2.loc26_19.2 => constants.%T.as_type.as.Iterate.impl.NewCursor.f6c
// CHECK:STDOUT:   %specific_impl_fn.loc26_19.4 => constants.%T.as_type.as.Iterate.impl.NewCursor.specific_fn
// CHECK:STDOUT:   %impl.elem1 => constants.%Destroy.facet.e1e
// CHECK:STDOUT:   %as_type => constants.%tuple.type.dd0
// CHECK:STDOUT:   %require_complete.1 => constants.%complete_type.843
// CHECK:STDOUT:   %ptr => constants.%ptr.a4f
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type => constants.%Iterate.WithSelf.Next.type.e92
// CHECK:STDOUT:   %.loc26_19.20 => constants.%.8b9
// CHECK:STDOUT:   %impl.elem3.loc26_19.2 => constants.%T.as_type.as.Iterate.impl.Next.630
// CHECK:STDOUT:   %specific_impl_fn.loc26_19.5 => constants.%T.as_type.as.Iterate.impl.Next.specific_fn
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
// CHECK:STDOUT:   %facet_type.f48: type = facet_type <@Destroy & @Copy> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %T.220: %OptionalStorage.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Optional.Get.type.bb9: type = fn_type @Optional.Get, @Optional(%T.220) [symbolic]
// CHECK:STDOUT:   %Optional.Get.fa1: %Optional.Get.type.bb9 = struct_value () [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.type.e9e: type = fn_type @Optional.HasValue, @Optional(%T.220) [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.6cd: %Optional.HasValue.type.e9e = struct_value () [symbolic]
// CHECK:STDOUT:   %T.283: %facet_type.f48 = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.d50: type = class_type @DefaultOptionalStorage, @DefaultOptionalStorage(%T.283) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Get.type.a12: type = fn_type @T.as_type.as.OptionalStorage.impl.Get, @T.as_type.as.OptionalStorage.impl(%T.283) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Get.d32: %T.as_type.as.OptionalStorage.impl.Get.type.a12 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Has.type.e5a: type = fn_type @T.as_type.as.OptionalStorage.impl.Has, @T.as_type.as.OptionalStorage.impl(%T.283) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Has.4ee: %T.as_type.as.OptionalStorage.impl.Has.type.e5a = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Some.type.492: type = fn_type @T.as_type.as.OptionalStorage.impl.Some, @T.as_type.as.OptionalStorage.impl(%T.283) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Some.89a: %T.as_type.as.OptionalStorage.impl.Some.type.492 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.None.type.0a1: type = fn_type @T.as_type.as.OptionalStorage.impl.None, @T.as_type.as.OptionalStorage.impl(%T.283) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.None.e06: %T.as_type.as.OptionalStorage.impl.None.type.0a1 = struct_value () [symbolic]
// CHECK:STDOUT:   %.Self.500: %Iterate.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %CppRange.type: type = facet_type <@CppRange> [concrete]
// CHECK:STDOUT:   %.Self.e0b: %CppRange.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %CppRangeForIterate.lookup_impl_witness.50b: <witness> = lookup_impl_witness %.Self.e0b, @CppRangeForIterate [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.32d: type = impl_witness_access %CppRangeForIterate.lookup_impl_witness.50b, element0 [symbolic_self]
// CHECK:STDOUT:   %CppUnsafeDeref.lookup_impl_witness.851: <witness> = lookup_impl_witness %impl.elem0.32d, @CppUnsafeDeref [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.10d: type = impl_witness_access %CppUnsafeDeref.lookup_impl_witness.851, element0 [symbolic_self]
// CHECK:STDOUT:   %CppRange_where.type.d682d6.1: type = facet_type <@CppRange where %impl.elem0.10d impls @Copy> [concrete]
// CHECK:STDOUT:   %T.f45530.2: %CppRange_where.type.d682d6.1 = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.type.b36: type = fn_type @T.as_type.as.Iterate.impl.Next, @T.as_type.as.Iterate.impl(%T.f45530.2) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.d6b: %T.as_type.as.Iterate.impl.Next.type.b36 = struct_value () [symbolic]
// CHECK:STDOUT:   %CppRangeForIterate.lookup_impl_witness.b581a8.2: <witness> = lookup_impl_witness %T.f45530.2, @CppRangeForIterate [symbolic]
// CHECK:STDOUT:   %impl.elem0.5f971c.2: type = impl_witness_access %CppRangeForIterate.lookup_impl_witness.b581a8.2, element0 [symbolic]
// CHECK:STDOUT:   %CppUnsafeDeref.lookup_impl_witness.87e543.2: <witness> = lookup_impl_witness %impl.elem0.5f971c.2, @CppUnsafeDeref [symbolic]
// CHECK:STDOUT:   %impl.elem0.493765.2: type = impl_witness_access %CppUnsafeDeref.lookup_impl_witness.87e543.2, element0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.type.953: type = fn_type @T.as_type.as.Iterate.impl.NewCursor, @T.as_type.as.Iterate.impl(%T.f45530.2) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.544: %T.as_type.as.Iterate.impl.NewCursor.type.953 = struct_value () [symbolic]
// CHECK:STDOUT:   %impl.elem1.c2dc15.2: type = impl_witness_access %CppRangeForIterate.lookup_impl_witness.b581a8.2, element1 [symbolic]
// CHECK:STDOUT:   %tuple.type.1ba: type = tuple_type (%impl.elem0.5f971c.2, %impl.elem1.c2dc15.2) [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.812: <witness> = lookup_impl_witness %tuple.type.1ba, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.9cd: %Destroy.type = facet_value %tuple.type.1ba, (%Destroy.lookup_impl_witness.812) [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.35816c.2: <witness> = lookup_impl_witness %impl.elem0.493765.2, @Destroy [symbolic]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.167: <witness> = lookup_impl_witness %impl.elem0.493765.2, @Copy [symbolic]
// CHECK:STDOUT:   %facet_value.f93: %facet_type.f48 = facet_value %impl.elem0.493765.2, (%Destroy.lookup_impl_witness.35816c.2, %Copy.lookup_impl_witness.167) [symbolic]
// CHECK:STDOUT:   %ValueType: type = class_type @ValueType [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %pattern_type.c43: type = pattern_type %ValueType [concrete]
// CHECK:STDOUT:   %ValueType.cpp_destructor.type: type = fn_type @ValueType.cpp_destructor [concrete]
// CHECK:STDOUT:   %ValueType.cpp_destructor: %ValueType.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT:   %ValueType.Op.type.4c0699.1: type = fn_type @ValueType.Op.1 [concrete]
// CHECK:STDOUT:   %ValueType.Op.4ec7f2.1: %ValueType.Op.type.4c0699.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.c4c: <witness> = custom_witness (%ValueType.Op.4ec7f2.1), @Destroy [concrete]
// CHECK:STDOUT:   %const.dd6: type = const_type %ValueType [concrete]
// CHECK:STDOUT:   %ptr.69c: type = ptr_type %const.dd6 [concrete]
// CHECK:STDOUT:   %ptr.233: type = ptr_type %ValueType [concrete]
// CHECK:STDOUT:   %ValueType.Op.type.4c0699.2: type = fn_type @ValueType.Op.2 [concrete]
// CHECK:STDOUT:   %ValueType.Op.4ec7f2.2: %ValueType.Op.type.4c0699.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.202: <witness> = custom_witness (%ValueType.Op.4ec7f2.2), @Copy [concrete]
// CHECK:STDOUT:   %facet_value.9f0: %facet_type.f48 = facet_value %ValueType, (%custom_witness.c4c, %custom_witness.202) [concrete]
// CHECK:STDOUT:   %Iterate.lookup_impl_witness.1d8: <witness> = lookup_impl_witness %.Self.500, @Iterate [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.294: %facet_type.f48 = impl_witness_access %Iterate.lookup_impl_witness.1d8, element0 [symbolic_self]
// CHECK:STDOUT:   %Iterate_where.type.081: type = facet_type <@Iterate where %impl.elem0.294 = %facet_value.9f0> [concrete]
// CHECK:STDOUT:   %pattern_type.d57: type = pattern_type %Iterate_where.type.081 [concrete]
// CHECK:STDOUT:   %R.patt: %pattern_type.d57 = symbolic_binding_pattern R, 0 [symbolic]
// CHECK:STDOUT:   %R: %Iterate_where.type.081 = symbolic_binding R, 0 [symbolic]
// CHECK:STDOUT:   %R.as_type: type = facet_access_type %R [symbolic]
// CHECK:STDOUT:   %pattern_type.a97: type = pattern_type %R.as_type [symbolic]
// CHECK:STDOUT:   %r.param_patt.8bc: %pattern_type.a97 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %r.patt.440: %pattern_type.a97 = wrapper_binding_pattern r, %r.param_patt.8bc [symbolic]
// CHECK:STDOUT:   %i.patt: %pattern_type.c43 = value_binding_pattern i [concrete]
// CHECK:STDOUT:   %Iterate.lookup_impl_witness.bda: <witness> = lookup_impl_witness %R, @Iterate [symbolic]
// CHECK:STDOUT:   %Iterate.facet.5c3: %Iterate.type = facet_value %R.as_type, (%Iterate.lookup_impl_witness.bda) [symbolic]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.825: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.5c3) [symbolic]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.37e: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.5c3) [symbolic]
// CHECK:STDOUT:   %.ca5: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.825, %Iterate.facet.5c3 [symbolic]
// CHECK:STDOUT:   %impl.elem2.60f: %.ca5 = impl_witness_access %Iterate.lookup_impl_witness.bda, element2 [symbolic]
// CHECK:STDOUT:   %impl.elem1.75b: %Destroy.type = impl_witness_access %Iterate.lookup_impl_witness.bda, element1 [symbolic]
// CHECK:STDOUT:   %as_type.aee: type = facet_access_type %impl.elem1.75b [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.979: <specific function> = specific_impl_function %impl.elem2.60f, @Iterate.WithSelf.NewCursor(%Iterate.facet.5c3) [symbolic]
// CHECK:STDOUT:   %ptr.779: type = ptr_type %as_type.aee [symbolic]
// CHECK:STDOUT:   %.493: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.37e, %Iterate.facet.5c3 [symbolic]
// CHECK:STDOUT:   %impl.elem3.e03: %.493 = impl_witness_access %Iterate.lookup_impl_witness.bda, element3 [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.8b2: type = class_type @DefaultOptionalStorage, @DefaultOptionalStorage(%facet_value.9f0) [concrete]
// CHECK:STDOUT:   %OptionalStorage.impl_witness.986: <witness> = impl_witness imports.%OptionalStorage.impl_witness_table.460, @T.as_type.as.OptionalStorage.impl(%facet_value.9f0) [concrete]
// CHECK:STDOUT:   %OptionalStorage.facet.b3f: %OptionalStorage.type = facet_value %ValueType, (%OptionalStorage.impl_witness.986) [concrete]
// CHECK:STDOUT:   %Optional.a40: type = class_type @Optional, @Optional(%OptionalStorage.facet.b3f) [concrete]
// CHECK:STDOUT:   %specific_impl_fn.80c: <specific function> = specific_impl_function %impl.elem3.e03, @Iterate.WithSelf.Next(%Iterate.facet.5c3) [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.type.fe3: type = fn_type @Optional.HasValue, @Optional(%OptionalStorage.facet.b3f) [concrete]
// CHECK:STDOUT:   %Optional.HasValue.cb1: %Optional.HasValue.type.fe3 = struct_value () [concrete]
// CHECK:STDOUT:   %Optional.Get.type.316: type = fn_type @Optional.Get, @Optional(%OptionalStorage.facet.b3f) [concrete]
// CHECK:STDOUT:   %Optional.Get.4e3: %Optional.Get.type.316 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.facet.c02: %Destroy.type = facet_value %ValueType, (%custom_witness.c4c) [concrete]
// CHECK:STDOUT:   %MaybeUnformed.e16: type = class_type @MaybeUnformed, @MaybeUnformed(%Destroy.facet.c02) [concrete]
// CHECK:STDOUT:   %.be1: type = maybe_unformed_type %ValueType [concrete]
// CHECK:STDOUT:   %struct_type.value.has_value.baa: type = struct_type {.value: %MaybeUnformed.e16, .has_value: bool} [concrete]
// CHECK:STDOUT:   %Optional.HasValue.specific_fn.048: <specific function> = specific_function %Optional.HasValue.cb1, @Optional.HasValue(%OptionalStorage.facet.b3f) [concrete]
// CHECK:STDOUT:   %Optional.Get.specific_fn.5ce: <specific function> = specific_function %Optional.Get.4e3, @Optional.Get(%OptionalStorage.facet.b3f) [concrete]
// CHECK:STDOUT:   %operator_PlusEqual__carbon_thunk.type: type = fn_type @operator_PlusEqual__carbon_thunk [concrete]
// CHECK:STDOUT:   %operator_PlusEqual__carbon_thunk: %operator_PlusEqual__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.7: type = fn_type @Destroy.Op.loc31_37.6 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.7: %Destroy.Op.type.1d8f74.7 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.3ea: <witness> = lookup_impl_witness %impl.elem1.75b, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.696: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%impl.elem1.75b) [symbolic]
// CHECK:STDOUT:   %.fbd: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.696, %impl.elem1.75b [symbolic]
// CHECK:STDOUT:   %impl.elem0.bb2: %.fbd = impl_witness_access %Destroy.lookup_impl_witness.3ea, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.fa3: <specific function> = specific_impl_function %impl.elem0.bb2, @Destroy.WithSelf.Op(%impl.elem1.75b) [symbolic]
// CHECK:STDOUT:   %ConstRange: type = class_type @ConstRange [concrete]
// CHECK:STDOUT:   %pattern_type.c1d: type = pattern_type %ConstRange [concrete]
// CHECK:STDOUT:   %Iterator: type = class_type @Iterator.1 [concrete]
// CHECK:STDOUT:   %Sentinel: type = class_type @Sentinel.1 [concrete]
// CHECK:STDOUT:   %Begin.type: type = fn_type @Begin [concrete]
// CHECK:STDOUT:   %Begin: %Begin.type = struct_value () [concrete]
// CHECK:STDOUT:   %End.type: type = fn_type @End [concrete]
// CHECK:STDOUT:   %End: %End.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.116: <witness> = custom_witness (%Iterator, %Sentinel, %Begin, %End), @CppRangeForIterate [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.3f6dc2.1: type = fn_type @Iterator.Op.1 [concrete]
// CHECK:STDOUT:   %Iterator.Op.240e50.1: %Iterator.Op.type.3f6dc2.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.5be: <witness> = custom_witness (%ValueType, %Iterator.Op.240e50.1), @CppUnsafeDeref [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.3f6dc2.2: type = fn_type @Iterator.Op.2 [concrete]
// CHECK:STDOUT:   %Iterator.Op.240e50.2: %Iterator.Op.type.3f6dc2.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.a147d4.1: <witness> = custom_witness (%Iterator.Op.240e50.2), @Destroy [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.3f6dc2.3: type = fn_type @Iterator.Op.3 [concrete]
// CHECK:STDOUT:   %Iterator.Op.240e50.3: %Iterator.Op.type.3f6dc2.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.c31: <witness> = custom_witness (%Iterator.Op.240e50.3), @Inc [concrete]
// CHECK:STDOUT:   %Equal.type: type = fn_type @Equal [concrete]
// CHECK:STDOUT:   %Equal: %Equal.type = struct_value () [concrete]
// CHECK:STDOUT:   %NotEqual.type: type = fn_type @NotEqual [concrete]
// CHECK:STDOUT:   %NotEqual: %NotEqual.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.e21: <witness> = custom_witness (%Equal, %NotEqual), @EqWith, @EqWith(%Sentinel) [concrete]
// CHECK:STDOUT:   %Sentinel.Op.type: type = fn_type @Sentinel.Op [concrete]
// CHECK:STDOUT:   %Sentinel.Op: %Sentinel.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.a147d4.2: <witness> = custom_witness (%Sentinel.Op), @Destroy [concrete]
// CHECK:STDOUT:   %facet_value.289: %CppRange_where.type.d682d6.1 = facet_value %ConstRange, (%custom_witness.c4c, %custom_witness.202, %custom_witness.116, %custom_witness.a147d4.1, %custom_witness.5be, %custom_witness.c31, %custom_witness.e21, %custom_witness.a147d4.2) [concrete]
// CHECK:STDOUT:   %tuple.type.7a6: type = tuple_type (%Iterator, %Sentinel) [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.8: type = fn_type @Destroy.Op.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.8: %Destroy.Op.type.1d8f74.8 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.df9cc1.8: <witness> = custom_witness (%Destroy.Op.1a2547.8), @Destroy [concrete]
// CHECK:STDOUT:   %Destroy.facet.2fb: %Destroy.type = facet_value %tuple.type.7a6, (%custom_witness.df9cc1.8) [concrete]
// CHECK:STDOUT:   %Iterate.impl_witness.6f6: <witness> = impl_witness imports.%Iterate.impl_witness_table.121, @T.as_type.as.Iterate.impl(%facet_value.289) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.type.0fb: type = fn_type @T.as_type.as.Iterate.impl.NewCursor, @T.as_type.as.Iterate.impl(%facet_value.289) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.126: %T.as_type.as.Iterate.impl.NewCursor.type.0fb = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.type.846: type = fn_type @T.as_type.as.Iterate.impl.Next, @T.as_type.as.Iterate.impl(%facet_value.289) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.c7f: %T.as_type.as.Iterate.impl.Next.type.846 = struct_value () [concrete]
// CHECK:STDOUT:   %Iterate.facet.8d2: %Iterate.type = facet_value %ConstRange, (%Iterate.impl_witness.6f6) [concrete]
// CHECK:STDOUT:   %facet_value.142: %Iterate_where.type.081 = facet_value %ConstRange, (%Iterate.impl_witness.6f6) [concrete]
// CHECK:STDOUT:   %r.param_patt.ac5: %pattern_type.c1d = value_param_pattern [concrete]
// CHECK:STDOUT:   %r.patt.97f: %pattern_type.c1d = wrapper_binding_pattern r, %r.param_patt.ac5 [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.c36: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.8d2) [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.01e: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.8d2) [concrete]
// CHECK:STDOUT:   %.a1c: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.c36, %Iterate.facet.8d2 [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.specific_fn: <specific function> = specific_function %T.as_type.as.Iterate.impl.NewCursor.126, @T.as_type.as.Iterate.impl.NewCursor(%facet_value.289) [concrete]
// CHECK:STDOUT:   %ptr.d33: type = ptr_type %tuple.type.7a6 [concrete]
// CHECK:STDOUT:   %.eab: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.01e, %Iterate.facet.8d2 [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.specific_fn: <specific function> = specific_function %T.as_type.as.Iterate.impl.Next.c7f, @T.as_type.as.Iterate.impl.Next(%facet_value.289) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.2d5: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.2fb) [concrete]
// CHECK:STDOUT:   %.b07: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.2d5, %Destroy.facet.2fb [concrete]
// CHECK:STDOUT:   %complete_type.cf0: <witness> = complete_type_witness %tuple.type.7a6 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .DiffADL = %DiffADL
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.bb6: @Optional.%Optional.HasValue.type (%Optional.HasValue.type.e9e) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.HasValue (constants.%Optional.HasValue.6cd)]
// CHECK:STDOUT:   %Core.import_ref.dab: @Optional.%Optional.Get.type (%Optional.Get.type.bb9) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.Get (constants.%Optional.Get.fa1)]
// CHECK:STDOUT:   %Core.import_ref.0c6: type = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%DefaultOptionalStorage (constants.%DefaultOptionalStorage.d50)]
// CHECK:STDOUT:   %Core.import_ref.dc3: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.None.type (%T.as_type.as.OptionalStorage.impl.None.type.0a1) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.None (constants.%T.as_type.as.OptionalStorage.impl.None.e06)]
// CHECK:STDOUT:   %Core.import_ref.c53: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Some.type (%T.as_type.as.OptionalStorage.impl.Some.type.492) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Some (constants.%T.as_type.as.OptionalStorage.impl.Some.89a)]
// CHECK:STDOUT:   %Core.import_ref.ed4: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Has.type (%T.as_type.as.OptionalStorage.impl.Has.type.e5a) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Has (constants.%T.as_type.as.OptionalStorage.impl.Has.4ee)]
// CHECK:STDOUT:   %Core.import_ref.e9c: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Get.type (%T.as_type.as.OptionalStorage.impl.Get.type.a12) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Get (constants.%T.as_type.as.OptionalStorage.impl.Get.d32)]
// CHECK:STDOUT:   %Core.import_ref.0b2 = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], unloaded
// CHECK:STDOUT:   %OptionalStorage.impl_witness_table.460 = impl_witness_table (%Core.import_ref.0c6, %Core.import_ref.dc3, %Core.import_ref.c53, %Core.import_ref.ed4, %Core.import_ref.e9c, %Core.import_ref.0b2), @T.as_type.as.OptionalStorage.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.eef: %facet_type.f48 = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.Iterate.impl.%facet_value (constants.%facet_value.f93)]
// CHECK:STDOUT:   %Core.import_ref.fcc: %Destroy.type = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.Iterate.impl.%Destroy.facet (constants.%Destroy.facet.9cd)]
// CHECK:STDOUT:   %Core.import_ref.fd1: @T.as_type.as.Iterate.impl.%T.as_type.as.Iterate.impl.NewCursor.type (%T.as_type.as.Iterate.impl.NewCursor.type.953) = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.Iterate.impl.%T.as_type.as.Iterate.impl.NewCursor (constants.%T.as_type.as.Iterate.impl.NewCursor.544)]
// CHECK:STDOUT:   %Core.import_ref.273: @T.as_type.as.Iterate.impl.%T.as_type.as.Iterate.impl.Next.type (%T.as_type.as.Iterate.impl.Next.type.b36) = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.Iterate.impl.%T.as_type.as.Iterate.impl.Next (constants.%T.as_type.as.Iterate.impl.Next.d6b)]
// CHECK:STDOUT:   %Iterate.impl_witness_table.121 = impl_witness_table (%Core.import_ref.eef, %Core.import_ref.fcc, %Core.import_ref.fd1, %Core.import_ref.273), @T.as_type.as.Iterate.impl [concrete]
// CHECK:STDOUT:   %DiffADL: <namespace> = namespace [concrete] {
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
// CHECK:STDOUT:   %operator_PlusEqual__carbon_thunk.decl: %operator_PlusEqual__carbon_thunk.type = fn_decl @operator_PlusEqual__carbon_thunk [concrete = constants.%operator_PlusEqual__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ConstRange.decl: type = class_decl @ConstRange [concrete = constants.%ConstRange] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @TestIterate(%R.loc27_17.2: %Iterate_where.type.081) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Iterate.lookup_impl_witness: <witness> = lookup_impl_witness %R.loc27_17.1, @Iterate [symbolic = %Iterate.lookup_impl_witness (constants.%Iterate.lookup_impl_witness.bda)]
// CHECK:STDOUT:   %Iterate.facet.loc31_37.5: %Iterate.type = facet_value %R.as_type.loc27_79.1, (%Iterate.lookup_impl_witness) [symbolic = %Iterate.facet.loc31_37.5 (constants.%Iterate.facet.5c3)]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.loc31_37.5) [symbolic = %Iterate.WithSelf.NewCursor.type (constants.%Iterate.WithSelf.NewCursor.type.825)]
// CHECK:STDOUT:   %.loc31_37.20: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type, %Iterate.facet.loc31_37.5 [symbolic = %.loc31_37.20 (constants.%.ca5)]
// CHECK:STDOUT:   %impl.elem2.loc31_37.2: @TestIterate.%.loc31_37.20 (%.ca5) = impl_witness_access %Iterate.lookup_impl_witness, element2 [symbolic = %impl.elem2.loc31_37.2 (constants.%impl.elem2.60f)]
// CHECK:STDOUT:   %specific_impl_fn.loc31_37.4: <specific function> = specific_impl_function %impl.elem2.loc31_37.2, @Iterate.WithSelf.NewCursor(%Iterate.facet.loc31_37.5) [symbolic = %specific_impl_fn.loc31_37.4 (constants.%specific_impl_fn.979)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %ptr: type = ptr_type %as_type [symbolic = %ptr (constants.%ptr.779)]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.loc31_37.5) [symbolic = %Iterate.WithSelf.Next.type (constants.%Iterate.WithSelf.Next.type.37e)]
// CHECK:STDOUT:   %.loc31_37.21: type = fn_type_with_self_type %Iterate.WithSelf.Next.type, %Iterate.facet.loc31_37.5 [symbolic = %.loc31_37.21 (constants.%.493)]
// CHECK:STDOUT:   %impl.elem3.loc31_37.2: @TestIterate.%.loc31_37.21 (%.493) = impl_witness_access %Iterate.lookup_impl_witness, element3 [symbolic = %impl.elem3.loc31_37.2 (constants.%impl.elem3.e03)]
// CHECK:STDOUT:   %specific_impl_fn.loc31_37.5: <specific function> = specific_impl_function %impl.elem3.loc31_37.2, @Iterate.WithSelf.Next(%Iterate.facet.loc31_37.5) [symbolic = %specific_impl_fn.loc31_37.5 (constants.%specific_impl_fn.80c)]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%impl.elem1) [symbolic = %Destroy.WithSelf.Op.type (constants.%Destroy.WithSelf.Op.type.696)]
// CHECK:STDOUT:   %.loc31_37.22: type = fn_type_with_self_type %Destroy.WithSelf.Op.type, %impl.elem1 [symbolic = %.loc31_37.22 (constants.%.fbd)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %impl.elem1, @Destroy [symbolic = %Destroy.lookup_impl_witness (constants.%Destroy.lookup_impl_witness.3ea)]
// CHECK:STDOUT:   %impl.elem0.loc31_37.2: @TestIterate.%.loc31_37.22 (%.fbd) = impl_witness_access %Destroy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc31_37.2 (constants.%impl.elem0.bb2)]
// CHECK:STDOUT:   %specific_impl_fn.loc31_37.6: <specific function> = specific_impl_function %impl.elem0.loc31_37.2, @Destroy.WithSelf.Op(%impl.elem1) [symbolic = %specific_impl_fn.loc31_37.6 (constants.%specific_impl_fn.fa3)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%r.param: @TestIterate.%R.as_type.loc27_79.1 (%R.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %i.patt: %pattern_type.c43 = value_binding_pattern i [concrete = constants.%i.patt]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %r.ref: @TestIterate.%R.as_type.loc27_79.1 (%R.as_type) = name_ref r, %r
// CHECK:STDOUT:     %impl.elem2.loc31_37.1: @TestIterate.%.loc31_37.20 (%.ca5) = impl_witness_access constants.%Iterate.lookup_impl_witness.bda, element2 [symbolic = %impl.elem2.loc31_37.2 (constants.%impl.elem2.60f)]
// CHECK:STDOUT:     %bound_method.loc31_37.1: <bound method> = bound_method %r.ref, %impl.elem2.loc31_37.1
// CHECK:STDOUT:     %Iterate.facet.loc31_37.1: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.bda) [symbolic = %Iterate.facet.loc31_37.5 (constants.%Iterate.facet.5c3)]
// CHECK:STDOUT:     %.loc31_37.1: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc31_37.1 [symbolic = %Iterate.facet.loc31_37.5 (constants.%Iterate.facet.5c3)]
// CHECK:STDOUT:     %Iterate.facet.loc31_37.2: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.bda) [symbolic = %Iterate.facet.loc31_37.5 (constants.%Iterate.facet.5c3)]
// CHECK:STDOUT:     %.loc31_37.2: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc31_37.2 [symbolic = %Iterate.facet.loc31_37.5 (constants.%Iterate.facet.5c3)]
// CHECK:STDOUT:     %specific_impl_fn.loc31_37.1: <specific function> = specific_impl_function %impl.elem2.loc31_37.1, @Iterate.WithSelf.NewCursor(constants.%Iterate.facet.5c3) [symbolic = %specific_impl_fn.loc31_37.4 (constants.%specific_impl_fn.979)]
// CHECK:STDOUT:     %bound_method.loc31_37.2: <bound method> = bound_method %r.ref, %specific_impl_fn.loc31_37.1
// CHECK:STDOUT:     %var: ref @TestIterate.%as_type (%as_type.aee) = var_storage invalid
// CHECK:STDOUT:     %Iterate.WithSelf.NewCursor.call: init @TestIterate.%as_type (%as_type.aee) to %var = call %bound_method.loc31_37.2(%r.ref)
// CHECK:STDOUT:     assign %var, %Iterate.WithSelf.NewCursor.call
// CHECK:STDOUT:     br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.next:
// CHECK:STDOUT:     %addr.loc31: @TestIterate.%ptr (%ptr.779) = addr_of %var
// CHECK:STDOUT:     %impl.elem3.loc31_37.1: @TestIterate.%.loc31_37.21 (%.493) = impl_witness_access constants.%Iterate.lookup_impl_witness.bda, element3 [symbolic = %impl.elem3.loc31_37.2 (constants.%impl.elem3.e03)]
// CHECK:STDOUT:     %bound_method.loc31_37.3: <bound method> = bound_method %r.ref, %impl.elem3.loc31_37.1
// CHECK:STDOUT:     %Iterate.facet.loc31_37.3: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.bda) [symbolic = %Iterate.facet.loc31_37.5 (constants.%Iterate.facet.5c3)]
// CHECK:STDOUT:     %.loc31_37.3: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc31_37.3 [symbolic = %Iterate.facet.loc31_37.5 (constants.%Iterate.facet.5c3)]
// CHECK:STDOUT:     %Iterate.facet.loc31_37.4: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.bda) [symbolic = %Iterate.facet.loc31_37.5 (constants.%Iterate.facet.5c3)]
// CHECK:STDOUT:     %.loc31_37.4: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc31_37.4 [symbolic = %Iterate.facet.loc31_37.5 (constants.%Iterate.facet.5c3)]
// CHECK:STDOUT:     %.loc31_37.5: %Destroy.type = converted constants.%as_type.aee, constants.%impl.elem1.75b [symbolic = %impl.elem1 (constants.%impl.elem1.75b)]
// CHECK:STDOUT:     %.loc31_37.6: %Destroy.type = converted constants.%as_type.aee, constants.%impl.elem1.75b [symbolic = %impl.elem1 (constants.%impl.elem1.75b)]
// CHECK:STDOUT:     %specific_impl_fn.loc31_37.2: <specific function> = specific_impl_function %impl.elem3.loc31_37.1, @Iterate.WithSelf.Next(constants.%Iterate.facet.5c3) [symbolic = %specific_impl_fn.loc31_37.5 (constants.%specific_impl_fn.80c)]
// CHECK:STDOUT:     %bound_method.loc31_37.4: <bound method> = bound_method %r.ref, %specific_impl_fn.loc31_37.2
// CHECK:STDOUT:     %.loc31_37.7: ref %Optional.a40 = temporary_storage
// CHECK:STDOUT:     %Iterate.WithSelf.Next.call: init %Optional.a40 to %.loc31_37.7 = call %bound_method.loc31_37.4(%r.ref, %addr.loc31)
// CHECK:STDOUT:     %.loc31_37.8: ref %Optional.a40 = temporary %.loc31_37.7, %Iterate.WithSelf.Next.call
// CHECK:STDOUT:     %.loc31_37.9: %Optional.HasValue.type.fe3 = specific_constant imports.%Core.import_ref.bb6, @Optional(constants.%OptionalStorage.facet.b3f) [concrete = constants.%Optional.HasValue.cb1]
// CHECK:STDOUT:     %HasValue.ref: %Optional.HasValue.type.fe3 = name_ref HasValue, %.loc31_37.9 [concrete = constants.%Optional.HasValue.cb1]
// CHECK:STDOUT:     %Optional.HasValue.bound: <bound method> = bound_method %.loc31_37.8, %HasValue.ref
// CHECK:STDOUT:     %Optional.HasValue.specific_fn: <specific function> = specific_function %HasValue.ref, @Optional.HasValue(constants.%OptionalStorage.facet.b3f) [concrete = constants.%Optional.HasValue.specific_fn.048]
// CHECK:STDOUT:     %bound_method.loc31_37.5: <bound method> = bound_method %.loc31_37.8, %Optional.HasValue.specific_fn
// CHECK:STDOUT:     %.loc31_37.10: %Optional.a40 = acquire_value %.loc31_37.8
// CHECK:STDOUT:     %Optional.HasValue.call: init bool = call %bound_method.loc31_37.5(%.loc31_37.10)
// CHECK:STDOUT:     %.loc31_37.11: bool = value_of_initializer %Optional.HasValue.call
// CHECK:STDOUT:     %.loc31_37.12: bool = converted %Optional.HasValue.call, %.loc31_37.11
// CHECK:STDOUT:     if %.loc31_37.12 br !for.body else br !for.done
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.body:
// CHECK:STDOUT:     %.loc31_37.13: %Optional.Get.type.316 = specific_constant imports.%Core.import_ref.dab, @Optional(constants.%OptionalStorage.facet.b3f) [concrete = constants.%Optional.Get.4e3]
// CHECK:STDOUT:     %Get.ref: %Optional.Get.type.316 = name_ref Get, %.loc31_37.13 [concrete = constants.%Optional.Get.4e3]
// CHECK:STDOUT:     %Optional.Get.bound: <bound method> = bound_method %.loc31_37.8, %Get.ref
// CHECK:STDOUT:     %Optional.Get.specific_fn: <specific function> = specific_function %Get.ref, @Optional.Get(constants.%OptionalStorage.facet.b3f) [concrete = constants.%Optional.Get.specific_fn.5ce]
// CHECK:STDOUT:     %bound_method.loc31_37.6: <bound method> = bound_method %.loc31_37.8, %Optional.Get.specific_fn
// CHECK:STDOUT:     %.loc31_37.14: ref %ValueType = temporary_storage
// CHECK:STDOUT:     %.loc31_37.15: %Optional.a40 = acquire_value %.loc31_37.8
// CHECK:STDOUT:     %Optional.Get.call: init %ValueType to %.loc31_37.14 = call %bound_method.loc31_37.6(%.loc31_37.15)
// CHECK:STDOUT:     %.loc31_37.16: ref %ValueType = temporary %.loc31_37.14, %Optional.Get.call
// CHECK:STDOUT:     %.loc31_37.17: %ValueType = acquire_value %.loc31_37.16
// CHECK:STDOUT:     %.loc31_22: type = splice_block %ValueType.ref.loc31 [concrete = constants.%ValueType] {
// CHECK:STDOUT:       %Cpp.ref.loc31: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:       %DiffADL.ref.loc31: <namespace> = name_ref DiffADL, imports.%DiffADL [concrete = imports.%DiffADL]
// CHECK:STDOUT:       %ValueType.ref.loc31: type = name_ref ValueType, imports.%ValueType.decl [concrete = constants.%ValueType]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %i: %ValueType = wrapper_binding i, %.loc31_37.17
// CHECK:STDOUT:     %sum.ref: ref %ValueType = name_ref sum, %sum
// CHECK:STDOUT:     %i.ref: %ValueType = name_ref i, %i
// CHECK:STDOUT:     %.loc32_12: ref %ValueType = value_as_ref %i.ref
// CHECK:STDOUT:     %addr.loc32: %ptr.233 = addr_of %.loc32_12
// CHECK:STDOUT:     %.loc32_9.1: %ptr.69c = as_compatible %addr.loc32
// CHECK:STDOUT:     %.loc32_9.2: %ptr.69c = converted %addr.loc32, %.loc32_9.1
// CHECK:STDOUT:     %operator_PlusEqual__carbon_thunk.call: ref %ValueType = call imports.%operator_PlusEqual__carbon_thunk.decl(%sum.ref, %.loc32_9.2)
// CHECK:STDOUT:     %ValueType.Op.bound.loc31: <bound method> = bound_method %.loc31_37.16, constants.%ValueType.Op.4ec7f2.1
// CHECK:STDOUT:     %Op.ref.loc31: %ValueType.cpp_destructor.type = name_ref Op, imports.%ValueType.cpp_destructor.decl [concrete = constants.%ValueType.cpp_destructor]
// CHECK:STDOUT:     %ValueType.cpp_destructor.bound.loc31: <bound method> = bound_method %.loc31_37.16, %Op.ref.loc31
// CHECK:STDOUT:     %ValueType.cpp_destructor.call.loc31: init %empty_tuple.type = call %ValueType.cpp_destructor.bound.loc31(%.loc31_37.16)
// CHECK:STDOUT:     %Destroy.Op.bound.loc31_37.1: <bound method> = bound_method %.loc31_37.8, constants.%Destroy.Op.1a2547.7
// CHECK:STDOUT:     %Destroy.Op.call.loc31_37.1: init %empty_tuple.type = call %Destroy.Op.bound.loc31_37.1(%.loc31_37.8)
// CHECK:STDOUT:     br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.done:
// CHECK:STDOUT:     %Destroy.Op.bound.loc31_37.2: <bound method> = bound_method %.loc31_37.8, constants.%Destroy.Op.1a2547.7
// CHECK:STDOUT:     %Destroy.Op.call.loc31_37.2: init %empty_tuple.type = call %Destroy.Op.bound.loc31_37.2(%.loc31_37.8)
// CHECK:STDOUT:     %impl.elem0.loc31_37.1: @TestIterate.%.loc31_37.22 (%.fbd) = impl_witness_access constants.%Destroy.lookup_impl_witness.3ea, element0 [symbolic = %impl.elem0.loc31_37.2 (constants.%impl.elem0.bb2)]
// CHECK:STDOUT:     %bound_method.loc31_37.7: <bound method> = bound_method %var, %impl.elem0.loc31_37.1
// CHECK:STDOUT:     %.loc31_37.18: %Destroy.type = converted constants.%as_type.aee, constants.%impl.elem1.75b [symbolic = %impl.elem1 (constants.%impl.elem1.75b)]
// CHECK:STDOUT:     %.loc31_37.19: %Destroy.type = converted constants.%as_type.aee, constants.%impl.elem1.75b [symbolic = %impl.elem1 (constants.%impl.elem1.75b)]
// CHECK:STDOUT:     %specific_impl_fn.loc31_37.3: <specific function> = specific_impl_function %impl.elem0.loc31_37.1, @Destroy.WithSelf.Op(constants.%impl.elem1.75b) [symbolic = %specific_impl_fn.loc31_37.6 (constants.%specific_impl_fn.fa3)]
// CHECK:STDOUT:     %bound_method.loc31_37.8: <bound method> = bound_method %var, %specific_impl_fn.loc31_37.3
// CHECK:STDOUT:     %Destroy.WithSelf.Op.call: init %empty_tuple.type = call %bound_method.loc31_37.8(%var)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc31_37.1(%self.param: ref %.be1) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc31_37.2(%self.param: ref %MaybeUnformed.e16) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc31_37.3(%self.param: ref bool) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc31_37.4(%self.param: ref %struct_type.value.has_value.baa) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc31_37.5(%self.param: ref %DefaultOptionalStorage.8b2) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc31_37.6(%self.param: ref %Optional.a40) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestRangeFor(%c.param: %ConstRange) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %i.patt: %pattern_type.c43 = value_binding_pattern i [concrete = constants.%i.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c.ref: %ConstRange = name_ref c, %c
// CHECK:STDOUT:   %impl.elem2: %.a1c = impl_witness_access constants.%Iterate.impl_witness.6f6, element2 [concrete = constants.%T.as_type.as.Iterate.impl.NewCursor.126]
// CHECK:STDOUT:   %bound_method.loc45_37.1: <bound method> = bound_method %c.ref, %impl.elem2
// CHECK:STDOUT:   %facet_value.loc45_37.1: %CppRange_where.type.d682d6.1 = facet_value constants.%ConstRange, (constants.%custom_witness.c4c, constants.%custom_witness.202, constants.%custom_witness.116, constants.%custom_witness.a147d4.1, constants.%custom_witness.5be, constants.%custom_witness.c31, constants.%custom_witness.e21, constants.%custom_witness.a147d4.2) [concrete = constants.%facet_value.289]
// CHECK:STDOUT:   %.loc45_37.1: %CppRange_where.type.d682d6.1 = converted constants.%ConstRange, %facet_value.loc45_37.1 [concrete = constants.%facet_value.289]
// CHECK:STDOUT:   %facet_value.loc45_37.2: %CppRange_where.type.d682d6.1 = facet_value constants.%ConstRange, (constants.%custom_witness.c4c, constants.%custom_witness.202, constants.%custom_witness.116, constants.%custom_witness.a147d4.1, constants.%custom_witness.5be, constants.%custom_witness.c31, constants.%custom_witness.e21, constants.%custom_witness.a147d4.2) [concrete = constants.%facet_value.289]
// CHECK:STDOUT:   %.loc45_37.2: %CppRange_where.type.d682d6.1 = converted constants.%ConstRange, %facet_value.loc45_37.2 [concrete = constants.%facet_value.289]
// CHECK:STDOUT:   %specific_fn.loc45_37.1: <specific function> = specific_function %impl.elem2, @T.as_type.as.Iterate.impl.NewCursor(constants.%facet_value.289) [concrete = constants.%T.as_type.as.Iterate.impl.NewCursor.specific_fn]
// CHECK:STDOUT:   %bound_method.loc45_37.2: <bound method> = bound_method %c.ref, %specific_fn.loc45_37.1
// CHECK:STDOUT:   %var: ref %tuple.type.7a6 = var_storage invalid
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.call: init %tuple.type.7a6 to %var = call %bound_method.loc45_37.2(%c.ref)
// CHECK:STDOUT:   assign %var, %T.as_type.as.Iterate.impl.NewCursor.call
// CHECK:STDOUT:   br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT: !for.next:
// CHECK:STDOUT:   %addr.loc45: %ptr.d33 = addr_of %var
// CHECK:STDOUT:   %impl.elem3: %.eab = impl_witness_access constants.%Iterate.impl_witness.6f6, element3 [concrete = constants.%T.as_type.as.Iterate.impl.Next.c7f]
// CHECK:STDOUT:   %bound_method.loc45_37.3: <bound method> = bound_method %c.ref, %impl.elem3
// CHECK:STDOUT:   %facet_value.loc45_37.3: %CppRange_where.type.d682d6.1 = facet_value constants.%ConstRange, (constants.%custom_witness.c4c, constants.%custom_witness.202, constants.%custom_witness.116, constants.%custom_witness.a147d4.1, constants.%custom_witness.5be, constants.%custom_witness.c31, constants.%custom_witness.e21, constants.%custom_witness.a147d4.2) [concrete = constants.%facet_value.289]
// CHECK:STDOUT:   %.loc45_37.3: %CppRange_where.type.d682d6.1 = converted constants.%ConstRange, %facet_value.loc45_37.3 [concrete = constants.%facet_value.289]
// CHECK:STDOUT:   %facet_value.loc45_37.4: %CppRange_where.type.d682d6.1 = facet_value constants.%ConstRange, (constants.%custom_witness.c4c, constants.%custom_witness.202, constants.%custom_witness.116, constants.%custom_witness.a147d4.1, constants.%custom_witness.5be, constants.%custom_witness.c31, constants.%custom_witness.e21, constants.%custom_witness.a147d4.2) [concrete = constants.%facet_value.289]
// CHECK:STDOUT:   %.loc45_37.4: %CppRange_where.type.d682d6.1 = converted constants.%ConstRange, %facet_value.loc45_37.4 [concrete = constants.%facet_value.289]
// CHECK:STDOUT:   %specific_fn.loc45_37.2: <specific function> = specific_function %impl.elem3, @T.as_type.as.Iterate.impl.Next(constants.%facet_value.289) [concrete = constants.%T.as_type.as.Iterate.impl.Next.specific_fn]
// CHECK:STDOUT:   %bound_method.loc45_37.4: <bound method> = bound_method %c.ref, %specific_fn.loc45_37.2
// CHECK:STDOUT:   %.loc45_37.5: ref %Optional.a40 = temporary_storage
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.call: init %Optional.a40 to %.loc45_37.5 = call %bound_method.loc45_37.4(%c.ref, %addr.loc45)
// CHECK:STDOUT:   %.loc45_37.6: ref %Optional.a40 = temporary %.loc45_37.5, %T.as_type.as.Iterate.impl.Next.call
// CHECK:STDOUT:   %.loc45_37.7: %Optional.HasValue.type.fe3 = specific_constant imports.%Core.import_ref.bb6, @Optional(constants.%OptionalStorage.facet.b3f) [concrete = constants.%Optional.HasValue.cb1]
// CHECK:STDOUT:   %HasValue.ref: %Optional.HasValue.type.fe3 = name_ref HasValue, %.loc45_37.7 [concrete = constants.%Optional.HasValue.cb1]
// CHECK:STDOUT:   %Optional.HasValue.bound: <bound method> = bound_method %.loc45_37.6, %HasValue.ref
// CHECK:STDOUT:   %Optional.HasValue.specific_fn: <specific function> = specific_function %HasValue.ref, @Optional.HasValue(constants.%OptionalStorage.facet.b3f) [concrete = constants.%Optional.HasValue.specific_fn.048]
// CHECK:STDOUT:   %bound_method.loc45_37.5: <bound method> = bound_method %.loc45_37.6, %Optional.HasValue.specific_fn
// CHECK:STDOUT:   %.loc45_37.8: %Optional.a40 = acquire_value %.loc45_37.6
// CHECK:STDOUT:   %Optional.HasValue.call: init bool = call %bound_method.loc45_37.5(%.loc45_37.8)
// CHECK:STDOUT:   %.loc45_37.9: bool = value_of_initializer %Optional.HasValue.call
// CHECK:STDOUT:   %.loc45_37.10: bool = converted %Optional.HasValue.call, %.loc45_37.9
// CHECK:STDOUT:   if %.loc45_37.10 br !for.body else br !for.done
// CHECK:STDOUT:
// CHECK:STDOUT: !for.body:
// CHECK:STDOUT:   %.loc45_37.11: %Optional.Get.type.316 = specific_constant imports.%Core.import_ref.dab, @Optional(constants.%OptionalStorage.facet.b3f) [concrete = constants.%Optional.Get.4e3]
// CHECK:STDOUT:   %Get.ref: %Optional.Get.type.316 = name_ref Get, %.loc45_37.11 [concrete = constants.%Optional.Get.4e3]
// CHECK:STDOUT:   %Optional.Get.bound: <bound method> = bound_method %.loc45_37.6, %Get.ref
// CHECK:STDOUT:   %Optional.Get.specific_fn: <specific function> = specific_function %Get.ref, @Optional.Get(constants.%OptionalStorage.facet.b3f) [concrete = constants.%Optional.Get.specific_fn.5ce]
// CHECK:STDOUT:   %bound_method.loc45_37.6: <bound method> = bound_method %.loc45_37.6, %Optional.Get.specific_fn
// CHECK:STDOUT:   %.loc45_37.12: ref %ValueType = temporary_storage
// CHECK:STDOUT:   %.loc45_37.13: %Optional.a40 = acquire_value %.loc45_37.6
// CHECK:STDOUT:   %Optional.Get.call: init %ValueType to %.loc45_37.12 = call %bound_method.loc45_37.6(%.loc45_37.13)
// CHECK:STDOUT:   %.loc45_37.14: ref %ValueType = temporary %.loc45_37.12, %Optional.Get.call
// CHECK:STDOUT:   %.loc45_37.15: %ValueType = acquire_value %.loc45_37.14
// CHECK:STDOUT:   %.loc45_22: type = splice_block %ValueType.ref.loc45 [concrete = constants.%ValueType] {
// CHECK:STDOUT:     %Cpp.ref.loc45: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %DiffADL.ref.loc45: <namespace> = name_ref DiffADL, imports.%DiffADL [concrete = imports.%DiffADL]
// CHECK:STDOUT:     %ValueType.ref.loc45: type = name_ref ValueType, imports.%ValueType.decl [concrete = constants.%ValueType]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %i: %ValueType = wrapper_binding i, %.loc45_37.15
// CHECK:STDOUT:   %sum.ref: ref %ValueType = name_ref sum, %sum
// CHECK:STDOUT:   %i.ref: %ValueType = name_ref i, %i
// CHECK:STDOUT:   %.loc46_12: ref %ValueType = value_as_ref %i.ref
// CHECK:STDOUT:   %addr.loc46: %ptr.233 = addr_of %.loc46_12
// CHECK:STDOUT:   %.loc46_9.1: %ptr.69c = as_compatible %addr.loc46
// CHECK:STDOUT:   %.loc46_9.2: %ptr.69c = converted %addr.loc46, %.loc46_9.1
// CHECK:STDOUT:   %operator_PlusEqual__carbon_thunk.call: ref %ValueType = call imports.%operator_PlusEqual__carbon_thunk.decl(%sum.ref, %.loc46_9.2)
// CHECK:STDOUT:   %ValueType.Op.bound.loc45: <bound method> = bound_method %.loc45_37.14, constants.%ValueType.Op.4ec7f2.1
// CHECK:STDOUT:   %Op.ref.loc45: %ValueType.cpp_destructor.type = name_ref Op, imports.%ValueType.cpp_destructor.decl [concrete = constants.%ValueType.cpp_destructor]
// CHECK:STDOUT:   %ValueType.cpp_destructor.bound.loc45: <bound method> = bound_method %.loc45_37.14, %Op.ref.loc45
// CHECK:STDOUT:   %ValueType.cpp_destructor.call.loc45: init %empty_tuple.type = call %ValueType.cpp_destructor.bound.loc45(%.loc45_37.14)
// CHECK:STDOUT:   %Destroy.Op.bound.loc45_37.1: <bound method> = bound_method %.loc45_37.6, constants.%Destroy.Op.1a2547.7
// CHECK:STDOUT:   %Destroy.Op.call.loc45_37.1: init %empty_tuple.type = call %Destroy.Op.bound.loc45_37.1(%.loc45_37.6)
// CHECK:STDOUT:   br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT: !for.done:
// CHECK:STDOUT:   %Destroy.Op.bound.loc45_37.2: <bound method> = bound_method %.loc45_37.6, constants.%Destroy.Op.1a2547.7
// CHECK:STDOUT:   %Destroy.Op.call.loc45_37.2: init %empty_tuple.type = call %Destroy.Op.bound.loc45_37.2(%.loc45_37.6)
// CHECK:STDOUT:   %Destroy.Op.bound.loc45_37.3: <bound method> = bound_method %var, constants.%Destroy.Op.1a2547.8
// CHECK:STDOUT:   %Destroy.Op.call.loc45_37.3: init %empty_tuple.type = call %Destroy.Op.bound.loc45_37.3(%var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestIterate(constants.%R) {
// CHECK:STDOUT:   %R.patt.loc27_17.2 => constants.%R.patt
// CHECK:STDOUT:   %R.loc27_17.1 => constants.%R
// CHECK:STDOUT:   %R.as_type.loc27_79.1 => constants.%R.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.a97
// CHECK:STDOUT:   %r.param_patt.loc27_77.2 => constants.%r.param_patt.8bc
// CHECK:STDOUT:   %r.patt.loc27_77.2 => constants.%r.patt.440
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestIterate(constants.%facet_value.142) {
// CHECK:STDOUT:   %R.patt.loc27_17.2 => constants.%R.patt
// CHECK:STDOUT:   %R.loc27_17.1 => constants.%facet_value.142
// CHECK:STDOUT:   %R.as_type.loc27_79.1 => constants.%ConstRange
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.c1d
// CHECK:STDOUT:   %r.param_patt.loc27_77.2 => constants.%r.param_patt.ac5
// CHECK:STDOUT:   %r.patt.loc27_77.2 => constants.%r.patt.97f
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc27 => constants.%complete_type.357
// CHECK:STDOUT:   %Iterate.lookup_impl_witness => constants.%Iterate.impl_witness.6f6
// CHECK:STDOUT:   %Iterate.facet.loc31_37.5 => constants.%Iterate.facet.8d2
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type => constants.%Iterate.WithSelf.NewCursor.type.c36
// CHECK:STDOUT:   %.loc31_37.20 => constants.%.a1c
// CHECK:STDOUT:   %impl.elem2.loc31_37.2 => constants.%T.as_type.as.Iterate.impl.NewCursor.126
// CHECK:STDOUT:   %specific_impl_fn.loc31_37.4 => constants.%T.as_type.as.Iterate.impl.NewCursor.specific_fn
// CHECK:STDOUT:   %impl.elem1 => constants.%Destroy.facet.2fb
// CHECK:STDOUT:   %as_type => constants.%tuple.type.7a6
// CHECK:STDOUT:   %require_complete.1 => constants.%complete_type.cf0
// CHECK:STDOUT:   %ptr => constants.%ptr.d33
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type => constants.%Iterate.WithSelf.Next.type.01e
// CHECK:STDOUT:   %.loc31_37.21 => constants.%.eab
// CHECK:STDOUT:   %impl.elem3.loc31_37.2 => constants.%T.as_type.as.Iterate.impl.Next.c7f
// CHECK:STDOUT:   %specific_impl_fn.loc31_37.5 => constants.%T.as_type.as.Iterate.impl.Next.specific_fn
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.2d5
// CHECK:STDOUT:   %.loc31_37.22 => constants.%.b07
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.df9cc1.8
// CHECK:STDOUT:   %impl.elem0.loc31_37.2 => constants.%Destroy.Op.1a2547.8
// CHECK:STDOUT:   %specific_impl_fn.loc31_37.6 => constants.%Destroy.Op.1a2547.8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_adl_returns_different_types_mutable.carbon
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
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Get.type.a12: type = fn_type @T.as_type.as.OptionalStorage.impl.Get, @T.as_type.as.OptionalStorage.impl(%T.283) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Get.d32: %T.as_type.as.OptionalStorage.impl.Get.type.a12 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Has.type.e5a: type = fn_type @T.as_type.as.OptionalStorage.impl.Has, @T.as_type.as.OptionalStorage.impl(%T.283) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Has.4ee: %T.as_type.as.OptionalStorage.impl.Has.type.e5a = struct_value () [symbolic]
// CHECK:STDOUT:   %.Self.500: %Iterate.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %ValueType: type = class_type @ValueType [concrete]
// CHECK:STDOUT:   %pattern_type.d2d: type = pattern_type %ValueType [concrete]
// CHECK:STDOUT:   %ValueType.cpp_destructor.type: type = fn_type @ValueType.cpp_destructor [concrete]
// CHECK:STDOUT:   %ValueType.cpp_destructor: %ValueType.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT:   %ValueType.Op.type.db6aa9.1: type = fn_type @ValueType.Op.1 [concrete]
// CHECK:STDOUT:   %ValueType.Op.4590ed.1: %ValueType.Op.type.db6aa9.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.d86: <witness> = custom_witness (%ValueType.Op.4590ed.1), @Destroy [concrete]
// CHECK:STDOUT:   %const.91f: type = const_type %ValueType [concrete]
// CHECK:STDOUT:   %ptr.5f4: type = ptr_type %const.91f [concrete]
// CHECK:STDOUT:   %ptr.9c6: type = ptr_type %ValueType [concrete]
// CHECK:STDOUT:   %ValueType.Op.type.db6aa9.2: type = fn_type @ValueType.Op.2 [concrete]
// CHECK:STDOUT:   %ValueType.Op.4590ed.2: %ValueType.Op.type.db6aa9.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.587: <witness> = custom_witness (%ValueType.Op.4590ed.2), @Copy [concrete]
// CHECK:STDOUT:   %facet_value.56f: %facet_type.f48 = facet_value %ValueType, (%custom_witness.d86, %custom_witness.587) [concrete]
// CHECK:STDOUT:   %Iterate.lookup_impl_witness.1d8: <witness> = lookup_impl_witness %.Self.500, @Iterate [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.294: %facet_type.f48 = impl_witness_access %Iterate.lookup_impl_witness.1d8, element0 [symbolic_self]
// CHECK:STDOUT:   %Iterate_where.type.f03: type = facet_type <@Iterate where %impl.elem0.294 = %facet_value.56f> [concrete]
// CHECK:STDOUT:   %pattern_type.dde: type = pattern_type %Iterate_where.type.f03 [concrete]
// CHECK:STDOUT:   %R.patt: %pattern_type.dde = symbolic_binding_pattern R, 0 [symbolic]
// CHECK:STDOUT:   %R: %Iterate_where.type.f03 = symbolic_binding R, 0 [symbolic]
// CHECK:STDOUT:   %R.as_type: type = facet_access_type %R [symbolic]
// CHECK:STDOUT:   %pattern_type.acd: type = pattern_type %R.as_type [symbolic]
// CHECK:STDOUT:   %r.param_patt: %pattern_type.acd = value_param_pattern [symbolic]
// CHECK:STDOUT:   %r.patt: %pattern_type.acd = wrapper_binding_pattern r, %r.param_patt [symbolic]
// CHECK:STDOUT:   %i.patt: %pattern_type.d2d = value_binding_pattern i [concrete]
// CHECK:STDOUT:   %Iterate.lookup_impl_witness.698: <witness> = lookup_impl_witness %R, @Iterate [symbolic]
// CHECK:STDOUT:   %Iterate.facet.122: %Iterate.type = facet_value %R.as_type, (%Iterate.lookup_impl_witness.698) [symbolic]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.d16: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.122) [symbolic]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.2f3: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.122) [symbolic]
// CHECK:STDOUT:   %.26d: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.d16, %Iterate.facet.122 [symbolic]
// CHECK:STDOUT:   %impl.elem2.d05: %.26d = impl_witness_access %Iterate.lookup_impl_witness.698, element2 [symbolic]
// CHECK:STDOUT:   %impl.elem1.127: %Destroy.type = impl_witness_access %Iterate.lookup_impl_witness.698, element1 [symbolic]
// CHECK:STDOUT:   %as_type.d08: type = facet_access_type %impl.elem1.127 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.342: <specific function> = specific_impl_function %impl.elem2.d05, @Iterate.WithSelf.NewCursor(%Iterate.facet.122) [symbolic]
// CHECK:STDOUT:   %ptr.2f2: type = ptr_type %as_type.d08 [symbolic]
// CHECK:STDOUT:   %.25b: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.2f3, %Iterate.facet.122 [symbolic]
// CHECK:STDOUT:   %impl.elem3.9e9: %.25b = impl_witness_access %Iterate.lookup_impl_witness.698, element3 [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.d67: type = class_type @DefaultOptionalStorage, @DefaultOptionalStorage(%facet_value.56f) [concrete]
// CHECK:STDOUT:   %OptionalStorage.impl_witness.2c9: <witness> = impl_witness imports.%OptionalStorage.impl_witness_table.460, @T.as_type.as.OptionalStorage.impl(%facet_value.56f) [concrete]
// CHECK:STDOUT:   %OptionalStorage.facet.19b: %OptionalStorage.type = facet_value %ValueType, (%OptionalStorage.impl_witness.2c9) [concrete]
// CHECK:STDOUT:   %Optional.40d: type = class_type @Optional, @Optional(%OptionalStorage.facet.19b) [concrete]
// CHECK:STDOUT:   %specific_impl_fn.ee9: <specific function> = specific_impl_function %impl.elem3.9e9, @Iterate.WithSelf.Next(%Iterate.facet.122) [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.type.5e1: type = fn_type @Optional.HasValue, @Optional(%OptionalStorage.facet.19b) [concrete]
// CHECK:STDOUT:   %Optional.HasValue.d7c: %Optional.HasValue.type.5e1 = struct_value () [concrete]
// CHECK:STDOUT:   %Optional.Get.type.1fe: type = fn_type @Optional.Get, @Optional(%OptionalStorage.facet.19b) [concrete]
// CHECK:STDOUT:   %Optional.Get.6fd: %Optional.Get.type.1fe = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.facet.10d: %Destroy.type = facet_value %ValueType, (%custom_witness.d86) [concrete]
// CHECK:STDOUT:   %MaybeUnformed.09f: type = class_type @MaybeUnformed, @MaybeUnformed(%Destroy.facet.10d) [concrete]
// CHECK:STDOUT:   %.dd1: type = maybe_unformed_type %ValueType [concrete]
// CHECK:STDOUT:   %struct_type.value.has_value.d6e: type = struct_type {.value: %MaybeUnformed.09f, .has_value: bool} [concrete]
// CHECK:STDOUT:   %Optional.HasValue.specific_fn: <specific function> = specific_function %Optional.HasValue.d7c, @Optional.HasValue(%OptionalStorage.facet.19b) [concrete]
// CHECK:STDOUT:   %Optional.Get.specific_fn: <specific function> = specific_function %Optional.Get.6fd, @Optional.Get(%OptionalStorage.facet.19b) [concrete]
// CHECK:STDOUT:   %operator_PlusEqual__carbon_thunk.type: type = fn_type @operator_PlusEqual__carbon_thunk [concrete]
// CHECK:STDOUT:   %operator_PlusEqual__carbon_thunk: %operator_PlusEqual__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.7: type = fn_type @Destroy.Op.loc31_44.6 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.7: %Destroy.Op.type.1d8f74.7 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.80c: <witness> = lookup_impl_witness %impl.elem1.127, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.678: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%impl.elem1.127) [symbolic]
// CHECK:STDOUT:   %.195: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.678, %impl.elem1.127 [symbolic]
// CHECK:STDOUT:   %impl.elem0.d51: %.195 = impl_witness_access %Destroy.lookup_impl_witness.80c, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.9d5: <specific function> = specific_impl_function %impl.elem0.d51, @Destroy.WithSelf.Op(%impl.elem1.127) [symbolic]
// CHECK:STDOUT:   %MutableRange: type = class_type @MutableRange [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .DiffMutableADL = %DiffMutableADL
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.bb6: @Optional.%Optional.HasValue.type (%Optional.HasValue.type.e9e) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.HasValue (constants.%Optional.HasValue.6cd)]
// CHECK:STDOUT:   %Core.import_ref.dab: @Optional.%Optional.Get.type (%Optional.Get.type.bb9) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.Get (constants.%Optional.Get.fa1)]
// CHECK:STDOUT:   %Core.import_ref.0c6: type = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%DefaultOptionalStorage (constants.%DefaultOptionalStorage.d50)]
// CHECK:STDOUT:   %Core.import_ref.daf = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], unloaded
// CHECK:STDOUT:   %Core.import_ref.bd4 = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], unloaded
// CHECK:STDOUT:   %Core.import_ref.ed4: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Has.type (%T.as_type.as.OptionalStorage.impl.Has.type.e5a) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Has (constants.%T.as_type.as.OptionalStorage.impl.Has.4ee)]
// CHECK:STDOUT:   %Core.import_ref.e9c: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Get.type (%T.as_type.as.OptionalStorage.impl.Get.type.a12) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Get (constants.%T.as_type.as.OptionalStorage.impl.Get.d32)]
// CHECK:STDOUT:   %Core.import_ref.0b2 = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], unloaded
// CHECK:STDOUT:   %OptionalStorage.impl_witness_table.460 = impl_witness_table (%Core.import_ref.0c6, %Core.import_ref.daf, %Core.import_ref.bd4, %Core.import_ref.ed4, %Core.import_ref.e9c, %Core.import_ref.0b2), @T.as_type.as.OptionalStorage.impl [concrete]
// CHECK:STDOUT:   %DiffMutableADL: <namespace> = namespace [concrete] {
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
// CHECK:STDOUT: generic fn @TestIterate(%R.loc27_17.2: %Iterate_where.type.f03) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Iterate.lookup_impl_witness: <witness> = lookup_impl_witness %R.loc27_17.1, @Iterate [symbolic = %Iterate.lookup_impl_witness (constants.%Iterate.lookup_impl_witness.698)]
// CHECK:STDOUT:   %Iterate.facet.loc31_44.5: %Iterate.type = facet_value %R.as_type.loc27_86.1, (%Iterate.lookup_impl_witness) [symbolic = %Iterate.facet.loc31_44.5 (constants.%Iterate.facet.122)]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.loc31_44.5) [symbolic = %Iterate.WithSelf.NewCursor.type (constants.%Iterate.WithSelf.NewCursor.type.d16)]
// CHECK:STDOUT:   %.loc31_44.20: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type, %Iterate.facet.loc31_44.5 [symbolic = %.loc31_44.20 (constants.%.26d)]
// CHECK:STDOUT:   %impl.elem2.loc31_44.2: @TestIterate.%.loc31_44.20 (%.26d) = impl_witness_access %Iterate.lookup_impl_witness, element2 [symbolic = %impl.elem2.loc31_44.2 (constants.%impl.elem2.d05)]
// CHECK:STDOUT:   %specific_impl_fn.loc31_44.4: <specific function> = specific_impl_function %impl.elem2.loc31_44.2, @Iterate.WithSelf.NewCursor(%Iterate.facet.loc31_44.5) [symbolic = %specific_impl_fn.loc31_44.4 (constants.%specific_impl_fn.342)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %ptr: type = ptr_type %as_type [symbolic = %ptr (constants.%ptr.2f2)]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.loc31_44.5) [symbolic = %Iterate.WithSelf.Next.type (constants.%Iterate.WithSelf.Next.type.2f3)]
// CHECK:STDOUT:   %.loc31_44.21: type = fn_type_with_self_type %Iterate.WithSelf.Next.type, %Iterate.facet.loc31_44.5 [symbolic = %.loc31_44.21 (constants.%.25b)]
// CHECK:STDOUT:   %impl.elem3.loc31_44.2: @TestIterate.%.loc31_44.21 (%.25b) = impl_witness_access %Iterate.lookup_impl_witness, element3 [symbolic = %impl.elem3.loc31_44.2 (constants.%impl.elem3.9e9)]
// CHECK:STDOUT:   %specific_impl_fn.loc31_44.5: <specific function> = specific_impl_function %impl.elem3.loc31_44.2, @Iterate.WithSelf.Next(%Iterate.facet.loc31_44.5) [symbolic = %specific_impl_fn.loc31_44.5 (constants.%specific_impl_fn.ee9)]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%impl.elem1) [symbolic = %Destroy.WithSelf.Op.type (constants.%Destroy.WithSelf.Op.type.678)]
// CHECK:STDOUT:   %.loc31_44.22: type = fn_type_with_self_type %Destroy.WithSelf.Op.type, %impl.elem1 [symbolic = %.loc31_44.22 (constants.%.195)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %impl.elem1, @Destroy [symbolic = %Destroy.lookup_impl_witness (constants.%Destroy.lookup_impl_witness.80c)]
// CHECK:STDOUT:   %impl.elem0.loc31_44.2: @TestIterate.%.loc31_44.22 (%.195) = impl_witness_access %Destroy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc31_44.2 (constants.%impl.elem0.d51)]
// CHECK:STDOUT:   %specific_impl_fn.loc31_44.6: <specific function> = specific_impl_function %impl.elem0.loc31_44.2, @Destroy.WithSelf.Op(%impl.elem1) [symbolic = %specific_impl_fn.loc31_44.6 (constants.%specific_impl_fn.9d5)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%r.param: @TestIterate.%R.as_type.loc27_86.1 (%R.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %i.patt: %pattern_type.d2d = value_binding_pattern i [concrete = constants.%i.patt]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %r.ref: @TestIterate.%R.as_type.loc27_86.1 (%R.as_type) = name_ref r, %r
// CHECK:STDOUT:     %impl.elem2.loc31_44.1: @TestIterate.%.loc31_44.20 (%.26d) = impl_witness_access constants.%Iterate.lookup_impl_witness.698, element2 [symbolic = %impl.elem2.loc31_44.2 (constants.%impl.elem2.d05)]
// CHECK:STDOUT:     %bound_method.loc31_44.1: <bound method> = bound_method %r.ref, %impl.elem2.loc31_44.1
// CHECK:STDOUT:     %Iterate.facet.loc31_44.1: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.698) [symbolic = %Iterate.facet.loc31_44.5 (constants.%Iterate.facet.122)]
// CHECK:STDOUT:     %.loc31_44.1: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc31_44.1 [symbolic = %Iterate.facet.loc31_44.5 (constants.%Iterate.facet.122)]
// CHECK:STDOUT:     %Iterate.facet.loc31_44.2: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.698) [symbolic = %Iterate.facet.loc31_44.5 (constants.%Iterate.facet.122)]
// CHECK:STDOUT:     %.loc31_44.2: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc31_44.2 [symbolic = %Iterate.facet.loc31_44.5 (constants.%Iterate.facet.122)]
// CHECK:STDOUT:     %specific_impl_fn.loc31_44.1: <specific function> = specific_impl_function %impl.elem2.loc31_44.1, @Iterate.WithSelf.NewCursor(constants.%Iterate.facet.122) [symbolic = %specific_impl_fn.loc31_44.4 (constants.%specific_impl_fn.342)]
// CHECK:STDOUT:     %bound_method.loc31_44.2: <bound method> = bound_method %r.ref, %specific_impl_fn.loc31_44.1
// CHECK:STDOUT:     %var: ref @TestIterate.%as_type (%as_type.d08) = var_storage invalid
// CHECK:STDOUT:     %Iterate.WithSelf.NewCursor.call: init @TestIterate.%as_type (%as_type.d08) to %var = call %bound_method.loc31_44.2(%r.ref)
// CHECK:STDOUT:     assign %var, %Iterate.WithSelf.NewCursor.call
// CHECK:STDOUT:     br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.next:
// CHECK:STDOUT:     %addr.loc31: @TestIterate.%ptr (%ptr.2f2) = addr_of %var
// CHECK:STDOUT:     %impl.elem3.loc31_44.1: @TestIterate.%.loc31_44.21 (%.25b) = impl_witness_access constants.%Iterate.lookup_impl_witness.698, element3 [symbolic = %impl.elem3.loc31_44.2 (constants.%impl.elem3.9e9)]
// CHECK:STDOUT:     %bound_method.loc31_44.3: <bound method> = bound_method %r.ref, %impl.elem3.loc31_44.1
// CHECK:STDOUT:     %Iterate.facet.loc31_44.3: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.698) [symbolic = %Iterate.facet.loc31_44.5 (constants.%Iterate.facet.122)]
// CHECK:STDOUT:     %.loc31_44.3: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc31_44.3 [symbolic = %Iterate.facet.loc31_44.5 (constants.%Iterate.facet.122)]
// CHECK:STDOUT:     %Iterate.facet.loc31_44.4: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.698) [symbolic = %Iterate.facet.loc31_44.5 (constants.%Iterate.facet.122)]
// CHECK:STDOUT:     %.loc31_44.4: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc31_44.4 [symbolic = %Iterate.facet.loc31_44.5 (constants.%Iterate.facet.122)]
// CHECK:STDOUT:     %.loc31_44.5: %Destroy.type = converted constants.%as_type.d08, constants.%impl.elem1.127 [symbolic = %impl.elem1 (constants.%impl.elem1.127)]
// CHECK:STDOUT:     %.loc31_44.6: %Destroy.type = converted constants.%as_type.d08, constants.%impl.elem1.127 [symbolic = %impl.elem1 (constants.%impl.elem1.127)]
// CHECK:STDOUT:     %specific_impl_fn.loc31_44.2: <specific function> = specific_impl_function %impl.elem3.loc31_44.1, @Iterate.WithSelf.Next(constants.%Iterate.facet.122) [symbolic = %specific_impl_fn.loc31_44.5 (constants.%specific_impl_fn.ee9)]
// CHECK:STDOUT:     %bound_method.loc31_44.4: <bound method> = bound_method %r.ref, %specific_impl_fn.loc31_44.2
// CHECK:STDOUT:     %.loc31_44.7: ref %Optional.40d = temporary_storage
// CHECK:STDOUT:     %Iterate.WithSelf.Next.call: init %Optional.40d to %.loc31_44.7 = call %bound_method.loc31_44.4(%r.ref, %addr.loc31)
// CHECK:STDOUT:     %.loc31_44.8: ref %Optional.40d = temporary %.loc31_44.7, %Iterate.WithSelf.Next.call
// CHECK:STDOUT:     %.loc31_44.9: %Optional.HasValue.type.5e1 = specific_constant imports.%Core.import_ref.bb6, @Optional(constants.%OptionalStorage.facet.19b) [concrete = constants.%Optional.HasValue.d7c]
// CHECK:STDOUT:     %HasValue.ref: %Optional.HasValue.type.5e1 = name_ref HasValue, %.loc31_44.9 [concrete = constants.%Optional.HasValue.d7c]
// CHECK:STDOUT:     %Optional.HasValue.bound: <bound method> = bound_method %.loc31_44.8, %HasValue.ref
// CHECK:STDOUT:     %Optional.HasValue.specific_fn: <specific function> = specific_function %HasValue.ref, @Optional.HasValue(constants.%OptionalStorage.facet.19b) [concrete = constants.%Optional.HasValue.specific_fn]
// CHECK:STDOUT:     %bound_method.loc31_44.5: <bound method> = bound_method %.loc31_44.8, %Optional.HasValue.specific_fn
// CHECK:STDOUT:     %.loc31_44.10: %Optional.40d = acquire_value %.loc31_44.8
// CHECK:STDOUT:     %Optional.HasValue.call: init bool = call %bound_method.loc31_44.5(%.loc31_44.10)
// CHECK:STDOUT:     %.loc31_44.11: bool = value_of_initializer %Optional.HasValue.call
// CHECK:STDOUT:     %.loc31_44.12: bool = converted %Optional.HasValue.call, %.loc31_44.11
// CHECK:STDOUT:     if %.loc31_44.12 br !for.body else br !for.done
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.body:
// CHECK:STDOUT:     %.loc31_44.13: %Optional.Get.type.1fe = specific_constant imports.%Core.import_ref.dab, @Optional(constants.%OptionalStorage.facet.19b) [concrete = constants.%Optional.Get.6fd]
// CHECK:STDOUT:     %Get.ref: %Optional.Get.type.1fe = name_ref Get, %.loc31_44.13 [concrete = constants.%Optional.Get.6fd]
// CHECK:STDOUT:     %Optional.Get.bound: <bound method> = bound_method %.loc31_44.8, %Get.ref
// CHECK:STDOUT:     %Optional.Get.specific_fn: <specific function> = specific_function %Get.ref, @Optional.Get(constants.%OptionalStorage.facet.19b) [concrete = constants.%Optional.Get.specific_fn]
// CHECK:STDOUT:     %bound_method.loc31_44.6: <bound method> = bound_method %.loc31_44.8, %Optional.Get.specific_fn
// CHECK:STDOUT:     %.loc31_44.14: ref %ValueType = temporary_storage
// CHECK:STDOUT:     %.loc31_44.15: %Optional.40d = acquire_value %.loc31_44.8
// CHECK:STDOUT:     %Optional.Get.call: init %ValueType to %.loc31_44.14 = call %bound_method.loc31_44.6(%.loc31_44.15)
// CHECK:STDOUT:     %.loc31_44.16: ref %ValueType = temporary %.loc31_44.14, %Optional.Get.call
// CHECK:STDOUT:     %.loc31_44.17: %ValueType = acquire_value %.loc31_44.16
// CHECK:STDOUT:     %.loc31_29: type = splice_block %ValueType.ref.loc31 [concrete = constants.%ValueType] {
// CHECK:STDOUT:       %Cpp.ref.loc31: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:       %DiffMutableADL.ref.loc31: <namespace> = name_ref DiffMutableADL, imports.%DiffMutableADL [concrete = imports.%DiffMutableADL]
// CHECK:STDOUT:       %ValueType.ref.loc31: type = name_ref ValueType, imports.%ValueType.decl [concrete = constants.%ValueType]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %i: %ValueType = wrapper_binding i, %.loc31_44.17
// CHECK:STDOUT:     %sum.ref: ref %ValueType = name_ref sum, %sum
// CHECK:STDOUT:     %i.ref: %ValueType = name_ref i, %i
// CHECK:STDOUT:     %.loc32_12: ref %ValueType = value_as_ref %i.ref
// CHECK:STDOUT:     %addr.loc32: %ptr.9c6 = addr_of %.loc32_12
// CHECK:STDOUT:     %.loc32_9.1: %ptr.5f4 = as_compatible %addr.loc32
// CHECK:STDOUT:     %.loc32_9.2: %ptr.5f4 = converted %addr.loc32, %.loc32_9.1
// CHECK:STDOUT:     %operator_PlusEqual__carbon_thunk.call: ref %ValueType = call imports.%operator_PlusEqual__carbon_thunk.decl(%sum.ref, %.loc32_9.2)
// CHECK:STDOUT:     %ValueType.Op.bound.loc31: <bound method> = bound_method %.loc31_44.16, constants.%ValueType.Op.4590ed.1
// CHECK:STDOUT:     %Op.ref.loc31: %ValueType.cpp_destructor.type = name_ref Op, imports.%ValueType.cpp_destructor.decl [concrete = constants.%ValueType.cpp_destructor]
// CHECK:STDOUT:     %ValueType.cpp_destructor.bound.loc31: <bound method> = bound_method %.loc31_44.16, %Op.ref.loc31
// CHECK:STDOUT:     %ValueType.cpp_destructor.call.loc31: init %empty_tuple.type = call %ValueType.cpp_destructor.bound.loc31(%.loc31_44.16)
// CHECK:STDOUT:     %Destroy.Op.bound.loc31_44.1: <bound method> = bound_method %.loc31_44.8, constants.%Destroy.Op.1a2547.7
// CHECK:STDOUT:     %Destroy.Op.call.loc31_44.1: init %empty_tuple.type = call %Destroy.Op.bound.loc31_44.1(%.loc31_44.8)
// CHECK:STDOUT:     br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.done:
// CHECK:STDOUT:     %Destroy.Op.bound.loc31_44.2: <bound method> = bound_method %.loc31_44.8, constants.%Destroy.Op.1a2547.7
// CHECK:STDOUT:     %Destroy.Op.call.loc31_44.2: init %empty_tuple.type = call %Destroy.Op.bound.loc31_44.2(%.loc31_44.8)
// CHECK:STDOUT:     %impl.elem0.loc31_44.1: @TestIterate.%.loc31_44.22 (%.195) = impl_witness_access constants.%Destroy.lookup_impl_witness.80c, element0 [symbolic = %impl.elem0.loc31_44.2 (constants.%impl.elem0.d51)]
// CHECK:STDOUT:     %bound_method.loc31_44.7: <bound method> = bound_method %var, %impl.elem0.loc31_44.1
// CHECK:STDOUT:     %.loc31_44.18: %Destroy.type = converted constants.%as_type.d08, constants.%impl.elem1.127 [symbolic = %impl.elem1 (constants.%impl.elem1.127)]
// CHECK:STDOUT:     %.loc31_44.19: %Destroy.type = converted constants.%as_type.d08, constants.%impl.elem1.127 [symbolic = %impl.elem1 (constants.%impl.elem1.127)]
// CHECK:STDOUT:     %specific_impl_fn.loc31_44.3: <specific function> = specific_impl_function %impl.elem0.loc31_44.1, @Destroy.WithSelf.Op(constants.%impl.elem1.127) [symbolic = %specific_impl_fn.loc31_44.6 (constants.%specific_impl_fn.9d5)]
// CHECK:STDOUT:     %bound_method.loc31_44.8: <bound method> = bound_method %var, %specific_impl_fn.loc31_44.3
// CHECK:STDOUT:     %Destroy.WithSelf.Op.call: init %empty_tuple.type = call %bound_method.loc31_44.8(%var)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc31_44.1(%self.param: ref %.dd1) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc31_44.2(%self.param: ref %MaybeUnformed.09f) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc31_44.3(%self.param: ref bool) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc31_44.4(%self.param: ref %struct_type.value.has_value.d6e) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc31_44.5(%self.param: ref %DefaultOptionalStorage.d67) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc31_44.6(%self.param: ref %Optional.40d) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestRangeFor(%m.param: ref %MutableRange) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %i.patt: %pattern_type.d2d = value_binding_pattern i [concrete = constants.%i.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %m.ref: ref %MutableRange = name_ref m, %m
// CHECK:STDOUT:   %var: ref <error> = var_storage invalid [concrete = <error>]
// CHECK:STDOUT:   assign <error>, <error>
// CHECK:STDOUT:   br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT: !for.next:
// CHECK:STDOUT:   %addr.loc94: <error> = addr_of <error> [concrete = <error>]
// CHECK:STDOUT:   %HasValue.ref: <error> = name_ref HasValue, <error> [concrete = <error>]
// CHECK:STDOUT:   if <error> br !for.body else br !for.done
// CHECK:STDOUT:
// CHECK:STDOUT: !for.body:
// CHECK:STDOUT:   %Get.ref: <error> = name_ref Get, <error> [concrete = <error>]
// CHECK:STDOUT:   %.loc94: type = splice_block %ValueType.ref.loc94 [concrete = constants.%ValueType] {
// CHECK:STDOUT:     %Cpp.ref.loc94: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %DiffMutableADL.ref.loc94: <namespace> = name_ref DiffMutableADL, imports.%DiffMutableADL [concrete = imports.%DiffMutableADL]
// CHECK:STDOUT:     %ValueType.ref.loc94: type = name_ref ValueType, imports.%ValueType.decl [concrete = constants.%ValueType]
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
// CHECK:STDOUT: specific @TestIterate(constants.%R) {
// CHECK:STDOUT:   %R.patt.loc27_17.2 => constants.%R.patt
// CHECK:STDOUT:   %R.loc27_17.1 => constants.%R
// CHECK:STDOUT:   %R.as_type.loc27_86.1 => constants.%R.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.acd
// CHECK:STDOUT:   %r.param_patt.loc27_84.2 => constants.%r.param_patt
// CHECK:STDOUT:   %r.patt.loc27_84.2 => constants.%r.patt
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_adl_overloads.carbon
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
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Get.type.a12: type = fn_type @T.as_type.as.OptionalStorage.impl.Get, @T.as_type.as.OptionalStorage.impl(%T.283) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Get.d32: %T.as_type.as.OptionalStorage.impl.Get.type.a12 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Has.type.e5a: type = fn_type @T.as_type.as.OptionalStorage.impl.Has, @T.as_type.as.OptionalStorage.impl(%T.283) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Has.4ee: %T.as_type.as.OptionalStorage.impl.Has.type.e5a = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Some.type.492: type = fn_type @T.as_type.as.OptionalStorage.impl.Some, @T.as_type.as.OptionalStorage.impl(%T.283) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Some.89a: %T.as_type.as.OptionalStorage.impl.Some.type.492 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.None.type.0a1: type = fn_type @T.as_type.as.OptionalStorage.impl.None, @T.as_type.as.OptionalStorage.impl(%T.283) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.None.e06: %T.as_type.as.OptionalStorage.impl.None.type.0a1 = struct_value () [symbolic]
// CHECK:STDOUT:   %.Self.500: %Iterate.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %CppRange.type: type = facet_type <@CppRange> [concrete]
// CHECK:STDOUT:   %.Self.e0b: %CppRange.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %CppRangeForIterate.lookup_impl_witness.50b: <witness> = lookup_impl_witness %.Self.e0b, @CppRangeForIterate [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.32d: type = impl_witness_access %CppRangeForIterate.lookup_impl_witness.50b, element0 [symbolic_self]
// CHECK:STDOUT:   %CppUnsafeDeref.lookup_impl_witness.851: <witness> = lookup_impl_witness %impl.elem0.32d, @CppUnsafeDeref [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.10d: type = impl_witness_access %CppUnsafeDeref.lookup_impl_witness.851, element0 [symbolic_self]
// CHECK:STDOUT:   %CppRange_where.type.d682d6.1: type = facet_type <@CppRange where %impl.elem0.10d impls @Copy> [concrete]
// CHECK:STDOUT:   %T.f45530.2: %CppRange_where.type.d682d6.1 = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.type.b36: type = fn_type @T.as_type.as.Iterate.impl.Next, @T.as_type.as.Iterate.impl(%T.f45530.2) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.d6b: %T.as_type.as.Iterate.impl.Next.type.b36 = struct_value () [symbolic]
// CHECK:STDOUT:   %CppRangeForIterate.lookup_impl_witness.b581a8.2: <witness> = lookup_impl_witness %T.f45530.2, @CppRangeForIterate [symbolic]
// CHECK:STDOUT:   %impl.elem0.5f971c.2: type = impl_witness_access %CppRangeForIterate.lookup_impl_witness.b581a8.2, element0 [symbolic]
// CHECK:STDOUT:   %CppUnsafeDeref.lookup_impl_witness.87e543.2: <witness> = lookup_impl_witness %impl.elem0.5f971c.2, @CppUnsafeDeref [symbolic]
// CHECK:STDOUT:   %impl.elem0.493765.2: type = impl_witness_access %CppUnsafeDeref.lookup_impl_witness.87e543.2, element0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.type.953: type = fn_type @T.as_type.as.Iterate.impl.NewCursor, @T.as_type.as.Iterate.impl(%T.f45530.2) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.544: %T.as_type.as.Iterate.impl.NewCursor.type.953 = struct_value () [symbolic]
// CHECK:STDOUT:   %impl.elem1.c2dc15.2: type = impl_witness_access %CppRangeForIterate.lookup_impl_witness.b581a8.2, element1 [symbolic]
// CHECK:STDOUT:   %tuple.type.1ba: type = tuple_type (%impl.elem0.5f971c.2, %impl.elem1.c2dc15.2) [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.812: <witness> = lookup_impl_witness %tuple.type.1ba, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.9cd: %Destroy.type = facet_value %tuple.type.1ba, (%Destroy.lookup_impl_witness.812) [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.35816c.2: <witness> = lookup_impl_witness %impl.elem0.493765.2, @Destroy [symbolic]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.167: <witness> = lookup_impl_witness %impl.elem0.493765.2, @Copy [symbolic]
// CHECK:STDOUT:   %facet_value.f93: %facet_type.f48 = facet_value %impl.elem0.493765.2, (%Destroy.lookup_impl_witness.35816c.2, %Copy.lookup_impl_witness.167) [symbolic]
// CHECK:STDOUT:   %ValueType: type = class_type @ValueType [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %pattern_type.e09: type = pattern_type %ValueType [concrete]
// CHECK:STDOUT:   %ValueType.cpp_destructor.type: type = fn_type @ValueType.cpp_destructor [concrete]
// CHECK:STDOUT:   %ValueType.cpp_destructor: %ValueType.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT:   %ValueType.Op.type.9ce105.1: type = fn_type @ValueType.Op.1 [concrete]
// CHECK:STDOUT:   %ValueType.Op.038ea9.1: %ValueType.Op.type.9ce105.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.ca2: <witness> = custom_witness (%ValueType.Op.038ea9.1), @Destroy [concrete]
// CHECK:STDOUT:   %const.924: type = const_type %ValueType [concrete]
// CHECK:STDOUT:   %ptr.403: type = ptr_type %const.924 [concrete]
// CHECK:STDOUT:   %ptr.6ed: type = ptr_type %ValueType [concrete]
// CHECK:STDOUT:   %ValueType.Op.type.9ce105.2: type = fn_type @ValueType.Op.2 [concrete]
// CHECK:STDOUT:   %ValueType.Op.038ea9.2: %ValueType.Op.type.9ce105.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.1c5: <witness> = custom_witness (%ValueType.Op.038ea9.2), @Copy [concrete]
// CHECK:STDOUT:   %facet_value.851: %facet_type.f48 = facet_value %ValueType, (%custom_witness.ca2, %custom_witness.1c5) [concrete]
// CHECK:STDOUT:   %Iterate.lookup_impl_witness.1d8: <witness> = lookup_impl_witness %.Self.500, @Iterate [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.294: %facet_type.f48 = impl_witness_access %Iterate.lookup_impl_witness.1d8, element0 [symbolic_self]
// CHECK:STDOUT:   %Iterate_where.type.290: type = facet_type <@Iterate where %impl.elem0.294 = %facet_value.851> [concrete]
// CHECK:STDOUT:   %pattern_type.9fd: type = pattern_type %Iterate_where.type.290 [concrete]
// CHECK:STDOUT:   %R.patt: %pattern_type.9fd = symbolic_binding_pattern R, 0 [symbolic]
// CHECK:STDOUT:   %R: %Iterate_where.type.290 = symbolic_binding R, 0 [symbolic]
// CHECK:STDOUT:   %R.as_type: type = facet_access_type %R [symbolic]
// CHECK:STDOUT:   %pattern_type.447: type = pattern_type %R.as_type [symbolic]
// CHECK:STDOUT:   %r.param_patt.207: %pattern_type.447 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %r.patt.e8b: %pattern_type.447 = wrapper_binding_pattern r, %r.param_patt.207 [symbolic]
// CHECK:STDOUT:   %i.patt: %pattern_type.e09 = value_binding_pattern i [concrete]
// CHECK:STDOUT:   %Iterate.lookup_impl_witness.116: <witness> = lookup_impl_witness %R, @Iterate [symbolic]
// CHECK:STDOUT:   %Iterate.facet.401: %Iterate.type = facet_value %R.as_type, (%Iterate.lookup_impl_witness.116) [symbolic]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.e10: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.401) [symbolic]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.7bd: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.401) [symbolic]
// CHECK:STDOUT:   %.4ab: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.e10, %Iterate.facet.401 [symbolic]
// CHECK:STDOUT:   %impl.elem2.739: %.4ab = impl_witness_access %Iterate.lookup_impl_witness.116, element2 [symbolic]
// CHECK:STDOUT:   %impl.elem1.9b3: %Destroy.type = impl_witness_access %Iterate.lookup_impl_witness.116, element1 [symbolic]
// CHECK:STDOUT:   %as_type.04b: type = facet_access_type %impl.elem1.9b3 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.2b2: <specific function> = specific_impl_function %impl.elem2.739, @Iterate.WithSelf.NewCursor(%Iterate.facet.401) [symbolic]
// CHECK:STDOUT:   %ptr.494: type = ptr_type %as_type.04b [symbolic]
// CHECK:STDOUT:   %.8ee: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.7bd, %Iterate.facet.401 [symbolic]
// CHECK:STDOUT:   %impl.elem3.65d: %.8ee = impl_witness_access %Iterate.lookup_impl_witness.116, element3 [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.338: type = class_type @DefaultOptionalStorage, @DefaultOptionalStorage(%facet_value.851) [concrete]
// CHECK:STDOUT:   %OptionalStorage.impl_witness.26a: <witness> = impl_witness imports.%OptionalStorage.impl_witness_table.460, @T.as_type.as.OptionalStorage.impl(%facet_value.851) [concrete]
// CHECK:STDOUT:   %OptionalStorage.facet.33d: %OptionalStorage.type = facet_value %ValueType, (%OptionalStorage.impl_witness.26a) [concrete]
// CHECK:STDOUT:   %Optional.3cc: type = class_type @Optional, @Optional(%OptionalStorage.facet.33d) [concrete]
// CHECK:STDOUT:   %specific_impl_fn.30e: <specific function> = specific_impl_function %impl.elem3.65d, @Iterate.WithSelf.Next(%Iterate.facet.401) [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.type.f78: type = fn_type @Optional.HasValue, @Optional(%OptionalStorage.facet.33d) [concrete]
// CHECK:STDOUT:   %Optional.HasValue.519: %Optional.HasValue.type.f78 = struct_value () [concrete]
// CHECK:STDOUT:   %Optional.Get.type.9c5: type = fn_type @Optional.Get, @Optional(%OptionalStorage.facet.33d) [concrete]
// CHECK:STDOUT:   %Optional.Get.323: %Optional.Get.type.9c5 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.facet.b42: %Destroy.type = facet_value %ValueType, (%custom_witness.ca2) [concrete]
// CHECK:STDOUT:   %MaybeUnformed.5d7: type = class_type @MaybeUnformed, @MaybeUnformed(%Destroy.facet.b42) [concrete]
// CHECK:STDOUT:   %.8f3: type = maybe_unformed_type %ValueType [concrete]
// CHECK:STDOUT:   %struct_type.value.has_value.bd1: type = struct_type {.value: %MaybeUnformed.5d7, .has_value: bool} [concrete]
// CHECK:STDOUT:   %Optional.HasValue.specific_fn.b3a: <specific function> = specific_function %Optional.HasValue.519, @Optional.HasValue(%OptionalStorage.facet.33d) [concrete]
// CHECK:STDOUT:   %Optional.Get.specific_fn.d62: <specific function> = specific_function %Optional.Get.323, @Optional.Get(%OptionalStorage.facet.33d) [concrete]
// CHECK:STDOUT:   %operator_PlusEqual__carbon_thunk.type: type = fn_type @operator_PlusEqual__carbon_thunk [concrete]
// CHECK:STDOUT:   %operator_PlusEqual__carbon_thunk: %operator_PlusEqual__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.7: type = fn_type @Destroy.Op.loc38_41.6 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.7: %Destroy.Op.type.1d8f74.7 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.95e: <witness> = lookup_impl_witness %impl.elem1.9b3, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.223: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%impl.elem1.9b3) [symbolic]
// CHECK:STDOUT:   %.32e: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.223, %impl.elem1.9b3 [symbolic]
// CHECK:STDOUT:   %impl.elem0.87c: %.32e = impl_witness_access %Destroy.lookup_impl_witness.95e, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.65a: <specific function> = specific_impl_function %impl.elem0.87c, @Destroy.WithSelf.Op(%impl.elem1.9b3) [symbolic]
// CHECK:STDOUT:   %MutableAndConstRange: type = class_type @MutableAndConstRange [concrete]
// CHECK:STDOUT:   %pattern_type.fdb: type = pattern_type %MutableAndConstRange [concrete]
// CHECK:STDOUT:   %Iterator: type = class_type @Iterator.1 [concrete]
// CHECK:STDOUT:   %Begin.type: type = fn_type @Begin [concrete]
// CHECK:STDOUT:   %Begin: %Begin.type = struct_value () [concrete]
// CHECK:STDOUT:   %End.type: type = fn_type @End [concrete]
// CHECK:STDOUT:   %End: %End.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.660: <witness> = custom_witness (%Iterator, %Iterator, %Begin, %End), @CppRangeForIterate [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.c95ff6.1: type = fn_type @Iterator.Op.1 [concrete]
// CHECK:STDOUT:   %Iterator.Op.c90ac2.1: %Iterator.Op.type.c95ff6.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.d56: <witness> = custom_witness (%ValueType, %Iterator.Op.c90ac2.1), @CppUnsafeDeref [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.c95ff6.2: type = fn_type @Iterator.Op.2 [concrete]
// CHECK:STDOUT:   %Iterator.Op.c90ac2.2: %Iterator.Op.type.c95ff6.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.a79: <witness> = custom_witness (%Iterator.Op.c90ac2.2), @Destroy [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.c95ff6.3: type = fn_type @Iterator.Op.3 [concrete]
// CHECK:STDOUT:   %Iterator.Op.c90ac2.3: %Iterator.Op.type.c95ff6.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.68a: <witness> = custom_witness (%Iterator.Op.c90ac2.3), @Inc [concrete]
// CHECK:STDOUT:   %Equal.type: type = fn_type @Equal [concrete]
// CHECK:STDOUT:   %Equal: %Equal.type = struct_value () [concrete]
// CHECK:STDOUT:   %NotEqual.type: type = fn_type @NotEqual [concrete]
// CHECK:STDOUT:   %NotEqual: %NotEqual.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.2ef: <witness> = custom_witness (%Equal, %NotEqual), @EqWith, @EqWith(%Iterator) [concrete]
// CHECK:STDOUT:   %facet_value.94c: %CppRange_where.type.d682d6.1 = facet_value %MutableAndConstRange, (%custom_witness.ca2, %custom_witness.1c5, %custom_witness.660, %custom_witness.a79, %custom_witness.d56, %custom_witness.68a, %custom_witness.2ef) [concrete]
// CHECK:STDOUT:   %tuple.type.de6: type = tuple_type (%Iterator, %Iterator) [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.8: type = fn_type @Destroy.Op.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.8: %Destroy.Op.type.1d8f74.8 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.df9cc1.8: <witness> = custom_witness (%Destroy.Op.1a2547.8), @Destroy [concrete]
// CHECK:STDOUT:   %Destroy.facet.84a: %Destroy.type = facet_value %tuple.type.de6, (%custom_witness.df9cc1.8) [concrete]
// CHECK:STDOUT:   %Iterate.impl_witness.e93: <witness> = impl_witness imports.%Iterate.impl_witness_table.121, @T.as_type.as.Iterate.impl(%facet_value.94c) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.type.c35: type = fn_type @T.as_type.as.Iterate.impl.NewCursor, @T.as_type.as.Iterate.impl(%facet_value.94c) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.237: %T.as_type.as.Iterate.impl.NewCursor.type.c35 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.type.50f: type = fn_type @T.as_type.as.Iterate.impl.Next, @T.as_type.as.Iterate.impl(%facet_value.94c) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.1b2: %T.as_type.as.Iterate.impl.Next.type.50f = struct_value () [concrete]
// CHECK:STDOUT:   %Iterate.facet.c00: %Iterate.type = facet_value %MutableAndConstRange, (%Iterate.impl_witness.e93) [concrete]
// CHECK:STDOUT:   %facet_value.266: %Iterate_where.type.290 = facet_value %MutableAndConstRange, (%Iterate.impl_witness.e93) [concrete]
// CHECK:STDOUT:   %r.param_patt.3a6: %pattern_type.fdb = value_param_pattern [concrete]
// CHECK:STDOUT:   %r.patt.403: %pattern_type.fdb = wrapper_binding_pattern r, %r.param_patt.3a6 [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.d86: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.c00) [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.4a6: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.c00) [concrete]
// CHECK:STDOUT:   %.e94: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.d86, %Iterate.facet.c00 [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.specific_fn: <specific function> = specific_function %T.as_type.as.Iterate.impl.NewCursor.237, @T.as_type.as.Iterate.impl.NewCursor(%facet_value.94c) [concrete]
// CHECK:STDOUT:   %ptr.b95: type = ptr_type %tuple.type.de6 [concrete]
// CHECK:STDOUT:   %.d53: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.4a6, %Iterate.facet.c00 [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.specific_fn: <specific function> = specific_function %T.as_type.as.Iterate.impl.Next.1b2, @T.as_type.as.Iterate.impl.Next(%facet_value.94c) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.544: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.84a) [concrete]
// CHECK:STDOUT:   %.a9c: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.544, %Destroy.facet.84a [concrete]
// CHECK:STDOUT:   %complete_type.95d: <witness> = complete_type_witness %tuple.type.de6 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .OverloadADL = %OverloadADL
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.bb6: @Optional.%Optional.HasValue.type (%Optional.HasValue.type.e9e) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.HasValue (constants.%Optional.HasValue.6cd)]
// CHECK:STDOUT:   %Core.import_ref.dab: @Optional.%Optional.Get.type (%Optional.Get.type.bb9) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.Get (constants.%Optional.Get.fa1)]
// CHECK:STDOUT:   %Core.import_ref.0c6: type = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%DefaultOptionalStorage (constants.%DefaultOptionalStorage.d50)]
// CHECK:STDOUT:   %Core.import_ref.dc3: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.None.type (%T.as_type.as.OptionalStorage.impl.None.type.0a1) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.None (constants.%T.as_type.as.OptionalStorage.impl.None.e06)]
// CHECK:STDOUT:   %Core.import_ref.c53: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Some.type (%T.as_type.as.OptionalStorage.impl.Some.type.492) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Some (constants.%T.as_type.as.OptionalStorage.impl.Some.89a)]
// CHECK:STDOUT:   %Core.import_ref.ed4: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Has.type (%T.as_type.as.OptionalStorage.impl.Has.type.e5a) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Has (constants.%T.as_type.as.OptionalStorage.impl.Has.4ee)]
// CHECK:STDOUT:   %Core.import_ref.e9c: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Get.type (%T.as_type.as.OptionalStorage.impl.Get.type.a12) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Get (constants.%T.as_type.as.OptionalStorage.impl.Get.d32)]
// CHECK:STDOUT:   %Core.import_ref.0b2 = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], unloaded
// CHECK:STDOUT:   %OptionalStorage.impl_witness_table.460 = impl_witness_table (%Core.import_ref.0c6, %Core.import_ref.dc3, %Core.import_ref.c53, %Core.import_ref.ed4, %Core.import_ref.e9c, %Core.import_ref.0b2), @T.as_type.as.OptionalStorage.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.eef: %facet_type.f48 = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.Iterate.impl.%facet_value (constants.%facet_value.f93)]
// CHECK:STDOUT:   %Core.import_ref.fcc: %Destroy.type = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.Iterate.impl.%Destroy.facet (constants.%Destroy.facet.9cd)]
// CHECK:STDOUT:   %Core.import_ref.fd1: @T.as_type.as.Iterate.impl.%T.as_type.as.Iterate.impl.NewCursor.type (%T.as_type.as.Iterate.impl.NewCursor.type.953) = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.Iterate.impl.%T.as_type.as.Iterate.impl.NewCursor (constants.%T.as_type.as.Iterate.impl.NewCursor.544)]
// CHECK:STDOUT:   %Core.import_ref.273: @T.as_type.as.Iterate.impl.%T.as_type.as.Iterate.impl.Next.type (%T.as_type.as.Iterate.impl.Next.type.b36) = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.Iterate.impl.%T.as_type.as.Iterate.impl.Next (constants.%T.as_type.as.Iterate.impl.Next.d6b)]
// CHECK:STDOUT:   %Iterate.impl_witness_table.121 = impl_witness_table (%Core.import_ref.eef, %Core.import_ref.fcc, %Core.import_ref.fd1, %Core.import_ref.273), @T.as_type.as.Iterate.impl [concrete]
// CHECK:STDOUT:   %OverloadADL: <namespace> = namespace [concrete] {
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
// CHECK:STDOUT:   %operator_PlusEqual__carbon_thunk.decl: %operator_PlusEqual__carbon_thunk.type = fn_decl @operator_PlusEqual__carbon_thunk [concrete = constants.%operator_PlusEqual__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %MutableAndConstRange.decl: type = class_decl @MutableAndConstRange [concrete = constants.%MutableAndConstRange] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @TestIterate(%R.loc34_17.2: %Iterate_where.type.290) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Iterate.lookup_impl_witness: <witness> = lookup_impl_witness %R.loc34_17.1, @Iterate [symbolic = %Iterate.lookup_impl_witness (constants.%Iterate.lookup_impl_witness.116)]
// CHECK:STDOUT:   %Iterate.facet.loc38_41.5: %Iterate.type = facet_value %R.as_type.loc34_83.1, (%Iterate.lookup_impl_witness) [symbolic = %Iterate.facet.loc38_41.5 (constants.%Iterate.facet.401)]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.loc38_41.5) [symbolic = %Iterate.WithSelf.NewCursor.type (constants.%Iterate.WithSelf.NewCursor.type.e10)]
// CHECK:STDOUT:   %.loc38_41.20: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type, %Iterate.facet.loc38_41.5 [symbolic = %.loc38_41.20 (constants.%.4ab)]
// CHECK:STDOUT:   %impl.elem2.loc38_41.2: @TestIterate.%.loc38_41.20 (%.4ab) = impl_witness_access %Iterate.lookup_impl_witness, element2 [symbolic = %impl.elem2.loc38_41.2 (constants.%impl.elem2.739)]
// CHECK:STDOUT:   %specific_impl_fn.loc38_41.4: <specific function> = specific_impl_function %impl.elem2.loc38_41.2, @Iterate.WithSelf.NewCursor(%Iterate.facet.loc38_41.5) [symbolic = %specific_impl_fn.loc38_41.4 (constants.%specific_impl_fn.2b2)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %ptr: type = ptr_type %as_type [symbolic = %ptr (constants.%ptr.494)]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.loc38_41.5) [symbolic = %Iterate.WithSelf.Next.type (constants.%Iterate.WithSelf.Next.type.7bd)]
// CHECK:STDOUT:   %.loc38_41.21: type = fn_type_with_self_type %Iterate.WithSelf.Next.type, %Iterate.facet.loc38_41.5 [symbolic = %.loc38_41.21 (constants.%.8ee)]
// CHECK:STDOUT:   %impl.elem3.loc38_41.2: @TestIterate.%.loc38_41.21 (%.8ee) = impl_witness_access %Iterate.lookup_impl_witness, element3 [symbolic = %impl.elem3.loc38_41.2 (constants.%impl.elem3.65d)]
// CHECK:STDOUT:   %specific_impl_fn.loc38_41.5: <specific function> = specific_impl_function %impl.elem3.loc38_41.2, @Iterate.WithSelf.Next(%Iterate.facet.loc38_41.5) [symbolic = %specific_impl_fn.loc38_41.5 (constants.%specific_impl_fn.30e)]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%impl.elem1) [symbolic = %Destroy.WithSelf.Op.type (constants.%Destroy.WithSelf.Op.type.223)]
// CHECK:STDOUT:   %.loc38_41.22: type = fn_type_with_self_type %Destroy.WithSelf.Op.type, %impl.elem1 [symbolic = %.loc38_41.22 (constants.%.32e)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %impl.elem1, @Destroy [symbolic = %Destroy.lookup_impl_witness (constants.%Destroy.lookup_impl_witness.95e)]
// CHECK:STDOUT:   %impl.elem0.loc38_41.2: @TestIterate.%.loc38_41.22 (%.32e) = impl_witness_access %Destroy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc38_41.2 (constants.%impl.elem0.87c)]
// CHECK:STDOUT:   %specific_impl_fn.loc38_41.6: <specific function> = specific_impl_function %impl.elem0.loc38_41.2, @Destroy.WithSelf.Op(%impl.elem1) [symbolic = %specific_impl_fn.loc38_41.6 (constants.%specific_impl_fn.65a)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%r.param: @TestIterate.%R.as_type.loc34_83.1 (%R.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %i.patt: %pattern_type.e09 = value_binding_pattern i [concrete = constants.%i.patt]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %r.ref: @TestIterate.%R.as_type.loc34_83.1 (%R.as_type) = name_ref r, %r
// CHECK:STDOUT:     %impl.elem2.loc38_41.1: @TestIterate.%.loc38_41.20 (%.4ab) = impl_witness_access constants.%Iterate.lookup_impl_witness.116, element2 [symbolic = %impl.elem2.loc38_41.2 (constants.%impl.elem2.739)]
// CHECK:STDOUT:     %bound_method.loc38_41.1: <bound method> = bound_method %r.ref, %impl.elem2.loc38_41.1
// CHECK:STDOUT:     %Iterate.facet.loc38_41.1: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.116) [symbolic = %Iterate.facet.loc38_41.5 (constants.%Iterate.facet.401)]
// CHECK:STDOUT:     %.loc38_41.1: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc38_41.1 [symbolic = %Iterate.facet.loc38_41.5 (constants.%Iterate.facet.401)]
// CHECK:STDOUT:     %Iterate.facet.loc38_41.2: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.116) [symbolic = %Iterate.facet.loc38_41.5 (constants.%Iterate.facet.401)]
// CHECK:STDOUT:     %.loc38_41.2: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc38_41.2 [symbolic = %Iterate.facet.loc38_41.5 (constants.%Iterate.facet.401)]
// CHECK:STDOUT:     %specific_impl_fn.loc38_41.1: <specific function> = specific_impl_function %impl.elem2.loc38_41.1, @Iterate.WithSelf.NewCursor(constants.%Iterate.facet.401) [symbolic = %specific_impl_fn.loc38_41.4 (constants.%specific_impl_fn.2b2)]
// CHECK:STDOUT:     %bound_method.loc38_41.2: <bound method> = bound_method %r.ref, %specific_impl_fn.loc38_41.1
// CHECK:STDOUT:     %var: ref @TestIterate.%as_type (%as_type.04b) = var_storage invalid
// CHECK:STDOUT:     %Iterate.WithSelf.NewCursor.call: init @TestIterate.%as_type (%as_type.04b) to %var = call %bound_method.loc38_41.2(%r.ref)
// CHECK:STDOUT:     assign %var, %Iterate.WithSelf.NewCursor.call
// CHECK:STDOUT:     br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.next:
// CHECK:STDOUT:     %addr.loc38: @TestIterate.%ptr (%ptr.494) = addr_of %var
// CHECK:STDOUT:     %impl.elem3.loc38_41.1: @TestIterate.%.loc38_41.21 (%.8ee) = impl_witness_access constants.%Iterate.lookup_impl_witness.116, element3 [symbolic = %impl.elem3.loc38_41.2 (constants.%impl.elem3.65d)]
// CHECK:STDOUT:     %bound_method.loc38_41.3: <bound method> = bound_method %r.ref, %impl.elem3.loc38_41.1
// CHECK:STDOUT:     %Iterate.facet.loc38_41.3: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.116) [symbolic = %Iterate.facet.loc38_41.5 (constants.%Iterate.facet.401)]
// CHECK:STDOUT:     %.loc38_41.3: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc38_41.3 [symbolic = %Iterate.facet.loc38_41.5 (constants.%Iterate.facet.401)]
// CHECK:STDOUT:     %Iterate.facet.loc38_41.4: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.116) [symbolic = %Iterate.facet.loc38_41.5 (constants.%Iterate.facet.401)]
// CHECK:STDOUT:     %.loc38_41.4: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc38_41.4 [symbolic = %Iterate.facet.loc38_41.5 (constants.%Iterate.facet.401)]
// CHECK:STDOUT:     %.loc38_41.5: %Destroy.type = converted constants.%as_type.04b, constants.%impl.elem1.9b3 [symbolic = %impl.elem1 (constants.%impl.elem1.9b3)]
// CHECK:STDOUT:     %.loc38_41.6: %Destroy.type = converted constants.%as_type.04b, constants.%impl.elem1.9b3 [symbolic = %impl.elem1 (constants.%impl.elem1.9b3)]
// CHECK:STDOUT:     %specific_impl_fn.loc38_41.2: <specific function> = specific_impl_function %impl.elem3.loc38_41.1, @Iterate.WithSelf.Next(constants.%Iterate.facet.401) [symbolic = %specific_impl_fn.loc38_41.5 (constants.%specific_impl_fn.30e)]
// CHECK:STDOUT:     %bound_method.loc38_41.4: <bound method> = bound_method %r.ref, %specific_impl_fn.loc38_41.2
// CHECK:STDOUT:     %.loc38_41.7: ref %Optional.3cc = temporary_storage
// CHECK:STDOUT:     %Iterate.WithSelf.Next.call: init %Optional.3cc to %.loc38_41.7 = call %bound_method.loc38_41.4(%r.ref, %addr.loc38)
// CHECK:STDOUT:     %.loc38_41.8: ref %Optional.3cc = temporary %.loc38_41.7, %Iterate.WithSelf.Next.call
// CHECK:STDOUT:     %.loc38_41.9: %Optional.HasValue.type.f78 = specific_constant imports.%Core.import_ref.bb6, @Optional(constants.%OptionalStorage.facet.33d) [concrete = constants.%Optional.HasValue.519]
// CHECK:STDOUT:     %HasValue.ref: %Optional.HasValue.type.f78 = name_ref HasValue, %.loc38_41.9 [concrete = constants.%Optional.HasValue.519]
// CHECK:STDOUT:     %Optional.HasValue.bound: <bound method> = bound_method %.loc38_41.8, %HasValue.ref
// CHECK:STDOUT:     %Optional.HasValue.specific_fn: <specific function> = specific_function %HasValue.ref, @Optional.HasValue(constants.%OptionalStorage.facet.33d) [concrete = constants.%Optional.HasValue.specific_fn.b3a]
// CHECK:STDOUT:     %bound_method.loc38_41.5: <bound method> = bound_method %.loc38_41.8, %Optional.HasValue.specific_fn
// CHECK:STDOUT:     %.loc38_41.10: %Optional.3cc = acquire_value %.loc38_41.8
// CHECK:STDOUT:     %Optional.HasValue.call: init bool = call %bound_method.loc38_41.5(%.loc38_41.10)
// CHECK:STDOUT:     %.loc38_41.11: bool = value_of_initializer %Optional.HasValue.call
// CHECK:STDOUT:     %.loc38_41.12: bool = converted %Optional.HasValue.call, %.loc38_41.11
// CHECK:STDOUT:     if %.loc38_41.12 br !for.body else br !for.done
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.body:
// CHECK:STDOUT:     %.loc38_41.13: %Optional.Get.type.9c5 = specific_constant imports.%Core.import_ref.dab, @Optional(constants.%OptionalStorage.facet.33d) [concrete = constants.%Optional.Get.323]
// CHECK:STDOUT:     %Get.ref: %Optional.Get.type.9c5 = name_ref Get, %.loc38_41.13 [concrete = constants.%Optional.Get.323]
// CHECK:STDOUT:     %Optional.Get.bound: <bound method> = bound_method %.loc38_41.8, %Get.ref
// CHECK:STDOUT:     %Optional.Get.specific_fn: <specific function> = specific_function %Get.ref, @Optional.Get(constants.%OptionalStorage.facet.33d) [concrete = constants.%Optional.Get.specific_fn.d62]
// CHECK:STDOUT:     %bound_method.loc38_41.6: <bound method> = bound_method %.loc38_41.8, %Optional.Get.specific_fn
// CHECK:STDOUT:     %.loc38_41.14: ref %ValueType = temporary_storage
// CHECK:STDOUT:     %.loc38_41.15: %Optional.3cc = acquire_value %.loc38_41.8
// CHECK:STDOUT:     %Optional.Get.call: init %ValueType to %.loc38_41.14 = call %bound_method.loc38_41.6(%.loc38_41.15)
// CHECK:STDOUT:     %.loc38_41.16: ref %ValueType = temporary %.loc38_41.14, %Optional.Get.call
// CHECK:STDOUT:     %.loc38_41.17: %ValueType = acquire_value %.loc38_41.16
// CHECK:STDOUT:     %.loc38_26: type = splice_block %ValueType.ref.loc38 [concrete = constants.%ValueType] {
// CHECK:STDOUT:       %Cpp.ref.loc38: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:       %OverloadADL.ref.loc38: <namespace> = name_ref OverloadADL, imports.%OverloadADL [concrete = imports.%OverloadADL]
// CHECK:STDOUT:       %ValueType.ref.loc38: type = name_ref ValueType, imports.%ValueType.decl [concrete = constants.%ValueType]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %i: %ValueType = wrapper_binding i, %.loc38_41.17
// CHECK:STDOUT:     %sum.ref: ref %ValueType = name_ref sum, %sum
// CHECK:STDOUT:     %i.ref: %ValueType = name_ref i, %i
// CHECK:STDOUT:     %.loc39_12: ref %ValueType = value_as_ref %i.ref
// CHECK:STDOUT:     %addr.loc39: %ptr.6ed = addr_of %.loc39_12
// CHECK:STDOUT:     %.loc39_9.1: %ptr.403 = as_compatible %addr.loc39
// CHECK:STDOUT:     %.loc39_9.2: %ptr.403 = converted %addr.loc39, %.loc39_9.1
// CHECK:STDOUT:     %operator_PlusEqual__carbon_thunk.call: ref %ValueType = call imports.%operator_PlusEqual__carbon_thunk.decl(%sum.ref, %.loc39_9.2)
// CHECK:STDOUT:     %ValueType.Op.bound.loc38: <bound method> = bound_method %.loc38_41.16, constants.%ValueType.Op.038ea9.1
// CHECK:STDOUT:     %Op.ref.loc38: %ValueType.cpp_destructor.type = name_ref Op, imports.%ValueType.cpp_destructor.decl [concrete = constants.%ValueType.cpp_destructor]
// CHECK:STDOUT:     %ValueType.cpp_destructor.bound.loc38: <bound method> = bound_method %.loc38_41.16, %Op.ref.loc38
// CHECK:STDOUT:     %ValueType.cpp_destructor.call.loc38: init %empty_tuple.type = call %ValueType.cpp_destructor.bound.loc38(%.loc38_41.16)
// CHECK:STDOUT:     %Destroy.Op.bound.loc38_41.1: <bound method> = bound_method %.loc38_41.8, constants.%Destroy.Op.1a2547.7
// CHECK:STDOUT:     %Destroy.Op.call.loc38_41.1: init %empty_tuple.type = call %Destroy.Op.bound.loc38_41.1(%.loc38_41.8)
// CHECK:STDOUT:     br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.done:
// CHECK:STDOUT:     %Destroy.Op.bound.loc38_41.2: <bound method> = bound_method %.loc38_41.8, constants.%Destroy.Op.1a2547.7
// CHECK:STDOUT:     %Destroy.Op.call.loc38_41.2: init %empty_tuple.type = call %Destroy.Op.bound.loc38_41.2(%.loc38_41.8)
// CHECK:STDOUT:     %impl.elem0.loc38_41.1: @TestIterate.%.loc38_41.22 (%.32e) = impl_witness_access constants.%Destroy.lookup_impl_witness.95e, element0 [symbolic = %impl.elem0.loc38_41.2 (constants.%impl.elem0.87c)]
// CHECK:STDOUT:     %bound_method.loc38_41.7: <bound method> = bound_method %var, %impl.elem0.loc38_41.1
// CHECK:STDOUT:     %.loc38_41.18: %Destroy.type = converted constants.%as_type.04b, constants.%impl.elem1.9b3 [symbolic = %impl.elem1 (constants.%impl.elem1.9b3)]
// CHECK:STDOUT:     %.loc38_41.19: %Destroy.type = converted constants.%as_type.04b, constants.%impl.elem1.9b3 [symbolic = %impl.elem1 (constants.%impl.elem1.9b3)]
// CHECK:STDOUT:     %specific_impl_fn.loc38_41.3: <specific function> = specific_impl_function %impl.elem0.loc38_41.1, @Destroy.WithSelf.Op(constants.%impl.elem1.9b3) [symbolic = %specific_impl_fn.loc38_41.6 (constants.%specific_impl_fn.65a)]
// CHECK:STDOUT:     %bound_method.loc38_41.8: <bound method> = bound_method %var, %specific_impl_fn.loc38_41.3
// CHECK:STDOUT:     %Destroy.WithSelf.Op.call: init %empty_tuple.type = call %bound_method.loc38_41.8(%var)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc38_41.1(%self.param: ref %.8f3) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc38_41.2(%self.param: ref %MaybeUnformed.5d7) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc38_41.3(%self.param: ref bool) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc38_41.4(%self.param: ref %struct_type.value.has_value.bd1) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc38_41.5(%self.param: ref %DefaultOptionalStorage.338) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc38_41.6(%self.param: ref %Optional.3cc) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestRangeFor(%mc.param: ref %MutableAndConstRange) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %i.patt: %pattern_type.e09 = value_binding_pattern i [concrete = constants.%i.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %mc.ref: ref %MutableAndConstRange = name_ref mc, %mc
// CHECK:STDOUT:   %impl.elem2: %.e94 = impl_witness_access constants.%Iterate.impl_witness.e93, element2 [concrete = constants.%T.as_type.as.Iterate.impl.NewCursor.237]
// CHECK:STDOUT:   %bound_method.loc80_42.1: <bound method> = bound_method %mc.ref, %impl.elem2
// CHECK:STDOUT:   %facet_value.loc80_42.1: %CppRange_where.type.d682d6.1 = facet_value constants.%MutableAndConstRange, (constants.%custom_witness.ca2, constants.%custom_witness.1c5, constants.%custom_witness.660, constants.%custom_witness.a79, constants.%custom_witness.d56, constants.%custom_witness.68a, constants.%custom_witness.2ef) [concrete = constants.%facet_value.94c]
// CHECK:STDOUT:   %.loc80_42.1: %CppRange_where.type.d682d6.1 = converted constants.%MutableAndConstRange, %facet_value.loc80_42.1 [concrete = constants.%facet_value.94c]
// CHECK:STDOUT:   %facet_value.loc80_42.2: %CppRange_where.type.d682d6.1 = facet_value constants.%MutableAndConstRange, (constants.%custom_witness.ca2, constants.%custom_witness.1c5, constants.%custom_witness.660, constants.%custom_witness.a79, constants.%custom_witness.d56, constants.%custom_witness.68a, constants.%custom_witness.2ef) [concrete = constants.%facet_value.94c]
// CHECK:STDOUT:   %.loc80_42.2: %CppRange_where.type.d682d6.1 = converted constants.%MutableAndConstRange, %facet_value.loc80_42.2 [concrete = constants.%facet_value.94c]
// CHECK:STDOUT:   %specific_fn.loc80_42.1: <specific function> = specific_function %impl.elem2, @T.as_type.as.Iterate.impl.NewCursor(constants.%facet_value.94c) [concrete = constants.%T.as_type.as.Iterate.impl.NewCursor.specific_fn]
// CHECK:STDOUT:   %bound_method.loc80_42.2: <bound method> = bound_method %mc.ref, %specific_fn.loc80_42.1
// CHECK:STDOUT:   %var: ref %tuple.type.de6 = var_storage invalid
// CHECK:STDOUT:   %.loc80_40.1: %MutableAndConstRange = acquire_value %mc.ref
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.call: init %tuple.type.de6 to %var = call %bound_method.loc80_42.2(%.loc80_40.1)
// CHECK:STDOUT:   assign %var, %T.as_type.as.Iterate.impl.NewCursor.call
// CHECK:STDOUT:   br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT: !for.next:
// CHECK:STDOUT:   %addr.loc80: %ptr.b95 = addr_of %var
// CHECK:STDOUT:   %impl.elem3: %.d53 = impl_witness_access constants.%Iterate.impl_witness.e93, element3 [concrete = constants.%T.as_type.as.Iterate.impl.Next.1b2]
// CHECK:STDOUT:   %bound_method.loc80_42.3: <bound method> = bound_method %mc.ref, %impl.elem3
// CHECK:STDOUT:   %facet_value.loc80_42.3: %CppRange_where.type.d682d6.1 = facet_value constants.%MutableAndConstRange, (constants.%custom_witness.ca2, constants.%custom_witness.1c5, constants.%custom_witness.660, constants.%custom_witness.a79, constants.%custom_witness.d56, constants.%custom_witness.68a, constants.%custom_witness.2ef) [concrete = constants.%facet_value.94c]
// CHECK:STDOUT:   %.loc80_42.3: %CppRange_where.type.d682d6.1 = converted constants.%MutableAndConstRange, %facet_value.loc80_42.3 [concrete = constants.%facet_value.94c]
// CHECK:STDOUT:   %facet_value.loc80_42.4: %CppRange_where.type.d682d6.1 = facet_value constants.%MutableAndConstRange, (constants.%custom_witness.ca2, constants.%custom_witness.1c5, constants.%custom_witness.660, constants.%custom_witness.a79, constants.%custom_witness.d56, constants.%custom_witness.68a, constants.%custom_witness.2ef) [concrete = constants.%facet_value.94c]
// CHECK:STDOUT:   %.loc80_42.4: %CppRange_where.type.d682d6.1 = converted constants.%MutableAndConstRange, %facet_value.loc80_42.4 [concrete = constants.%facet_value.94c]
// CHECK:STDOUT:   %specific_fn.loc80_42.2: <specific function> = specific_function %impl.elem3, @T.as_type.as.Iterate.impl.Next(constants.%facet_value.94c) [concrete = constants.%T.as_type.as.Iterate.impl.Next.specific_fn]
// CHECK:STDOUT:   %bound_method.loc80_42.4: <bound method> = bound_method %mc.ref, %specific_fn.loc80_42.2
// CHECK:STDOUT:   %.loc80_42.5: ref %Optional.3cc = temporary_storage
// CHECK:STDOUT:   %.loc80_40.2: %MutableAndConstRange = acquire_value %mc.ref
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.call: init %Optional.3cc to %.loc80_42.5 = call %bound_method.loc80_42.4(%.loc80_40.2, %addr.loc80)
// CHECK:STDOUT:   %.loc80_42.6: ref %Optional.3cc = temporary %.loc80_42.5, %T.as_type.as.Iterate.impl.Next.call
// CHECK:STDOUT:   %.loc80_42.7: %Optional.HasValue.type.f78 = specific_constant imports.%Core.import_ref.bb6, @Optional(constants.%OptionalStorage.facet.33d) [concrete = constants.%Optional.HasValue.519]
// CHECK:STDOUT:   %HasValue.ref: %Optional.HasValue.type.f78 = name_ref HasValue, %.loc80_42.7 [concrete = constants.%Optional.HasValue.519]
// CHECK:STDOUT:   %Optional.HasValue.bound: <bound method> = bound_method %.loc80_42.6, %HasValue.ref
// CHECK:STDOUT:   %Optional.HasValue.specific_fn: <specific function> = specific_function %HasValue.ref, @Optional.HasValue(constants.%OptionalStorage.facet.33d) [concrete = constants.%Optional.HasValue.specific_fn.b3a]
// CHECK:STDOUT:   %bound_method.loc80_42.5: <bound method> = bound_method %.loc80_42.6, %Optional.HasValue.specific_fn
// CHECK:STDOUT:   %.loc80_42.8: %Optional.3cc = acquire_value %.loc80_42.6
// CHECK:STDOUT:   %Optional.HasValue.call: init bool = call %bound_method.loc80_42.5(%.loc80_42.8)
// CHECK:STDOUT:   %.loc80_42.9: bool = value_of_initializer %Optional.HasValue.call
// CHECK:STDOUT:   %.loc80_42.10: bool = converted %Optional.HasValue.call, %.loc80_42.9
// CHECK:STDOUT:   if %.loc80_42.10 br !for.body else br !for.done
// CHECK:STDOUT:
// CHECK:STDOUT: !for.body:
// CHECK:STDOUT:   %.loc80_42.11: %Optional.Get.type.9c5 = specific_constant imports.%Core.import_ref.dab, @Optional(constants.%OptionalStorage.facet.33d) [concrete = constants.%Optional.Get.323]
// CHECK:STDOUT:   %Get.ref: %Optional.Get.type.9c5 = name_ref Get, %.loc80_42.11 [concrete = constants.%Optional.Get.323]
// CHECK:STDOUT:   %Optional.Get.bound: <bound method> = bound_method %.loc80_42.6, %Get.ref
// CHECK:STDOUT:   %Optional.Get.specific_fn: <specific function> = specific_function %Get.ref, @Optional.Get(constants.%OptionalStorage.facet.33d) [concrete = constants.%Optional.Get.specific_fn.d62]
// CHECK:STDOUT:   %bound_method.loc80_42.6: <bound method> = bound_method %.loc80_42.6, %Optional.Get.specific_fn
// CHECK:STDOUT:   %.loc80_42.12: ref %ValueType = temporary_storage
// CHECK:STDOUT:   %.loc80_42.13: %Optional.3cc = acquire_value %.loc80_42.6
// CHECK:STDOUT:   %Optional.Get.call: init %ValueType to %.loc80_42.12 = call %bound_method.loc80_42.6(%.loc80_42.13)
// CHECK:STDOUT:   %.loc80_42.14: ref %ValueType = temporary %.loc80_42.12, %Optional.Get.call
// CHECK:STDOUT:   %.loc80_42.15: %ValueType = acquire_value %.loc80_42.14
// CHECK:STDOUT:   %.loc80_26: type = splice_block %ValueType.ref.loc80 [concrete = constants.%ValueType] {
// CHECK:STDOUT:     %Cpp.ref.loc80: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %OverloadADL.ref.loc80: <namespace> = name_ref OverloadADL, imports.%OverloadADL [concrete = imports.%OverloadADL]
// CHECK:STDOUT:     %ValueType.ref.loc80: type = name_ref ValueType, imports.%ValueType.decl [concrete = constants.%ValueType]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %i: %ValueType = wrapper_binding i, %.loc80_42.15
// CHECK:STDOUT:   %sum.ref: ref %ValueType = name_ref sum, %sum
// CHECK:STDOUT:   %i.ref: %ValueType = name_ref i, %i
// CHECK:STDOUT:   %.loc81_12: ref %ValueType = value_as_ref %i.ref
// CHECK:STDOUT:   %addr.loc81: %ptr.6ed = addr_of %.loc81_12
// CHECK:STDOUT:   %.loc81_9.1: %ptr.403 = as_compatible %addr.loc81
// CHECK:STDOUT:   %.loc81_9.2: %ptr.403 = converted %addr.loc81, %.loc81_9.1
// CHECK:STDOUT:   %operator_PlusEqual__carbon_thunk.call: ref %ValueType = call imports.%operator_PlusEqual__carbon_thunk.decl(%sum.ref, %.loc81_9.2)
// CHECK:STDOUT:   %ValueType.Op.bound.loc80: <bound method> = bound_method %.loc80_42.14, constants.%ValueType.Op.038ea9.1
// CHECK:STDOUT:   %Op.ref.loc80: %ValueType.cpp_destructor.type = name_ref Op, imports.%ValueType.cpp_destructor.decl [concrete = constants.%ValueType.cpp_destructor]
// CHECK:STDOUT:   %ValueType.cpp_destructor.bound.loc80: <bound method> = bound_method %.loc80_42.14, %Op.ref.loc80
// CHECK:STDOUT:   %ValueType.cpp_destructor.call.loc80: init %empty_tuple.type = call %ValueType.cpp_destructor.bound.loc80(%.loc80_42.14)
// CHECK:STDOUT:   %Destroy.Op.bound.loc80_42.1: <bound method> = bound_method %.loc80_42.6, constants.%Destroy.Op.1a2547.7
// CHECK:STDOUT:   %Destroy.Op.call.loc80_42.1: init %empty_tuple.type = call %Destroy.Op.bound.loc80_42.1(%.loc80_42.6)
// CHECK:STDOUT:   br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT: !for.done:
// CHECK:STDOUT:   %Destroy.Op.bound.loc80_42.2: <bound method> = bound_method %.loc80_42.6, constants.%Destroy.Op.1a2547.7
// CHECK:STDOUT:   %Destroy.Op.call.loc80_42.2: init %empty_tuple.type = call %Destroy.Op.bound.loc80_42.2(%.loc80_42.6)
// CHECK:STDOUT:   %Destroy.Op.bound.loc80_42.3: <bound method> = bound_method %var, constants.%Destroy.Op.1a2547.8
// CHECK:STDOUT:   %Destroy.Op.call.loc80_42.3: init %empty_tuple.type = call %Destroy.Op.bound.loc80_42.3(%var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestIterate(constants.%R) {
// CHECK:STDOUT:   %R.patt.loc34_17.2 => constants.%R.patt
// CHECK:STDOUT:   %R.loc34_17.1 => constants.%R
// CHECK:STDOUT:   %R.as_type.loc34_83.1 => constants.%R.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.447
// CHECK:STDOUT:   %r.param_patt.loc34_81.2 => constants.%r.param_patt.207
// CHECK:STDOUT:   %r.patt.loc34_81.2 => constants.%r.patt.e8b
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestIterate(constants.%facet_value.266) {
// CHECK:STDOUT:   %R.patt.loc34_17.2 => constants.%R.patt
// CHECK:STDOUT:   %R.loc34_17.1 => constants.%facet_value.266
// CHECK:STDOUT:   %R.as_type.loc34_83.1 => constants.%MutableAndConstRange
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.fdb
// CHECK:STDOUT:   %r.param_patt.loc34_81.2 => constants.%r.param_patt.3a6
// CHECK:STDOUT:   %r.patt.loc34_81.2 => constants.%r.patt.403
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc34 => constants.%complete_type.357
// CHECK:STDOUT:   %Iterate.lookup_impl_witness => constants.%Iterate.impl_witness.e93
// CHECK:STDOUT:   %Iterate.facet.loc38_41.5 => constants.%Iterate.facet.c00
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type => constants.%Iterate.WithSelf.NewCursor.type.d86
// CHECK:STDOUT:   %.loc38_41.20 => constants.%.e94
// CHECK:STDOUT:   %impl.elem2.loc38_41.2 => constants.%T.as_type.as.Iterate.impl.NewCursor.237
// CHECK:STDOUT:   %specific_impl_fn.loc38_41.4 => constants.%T.as_type.as.Iterate.impl.NewCursor.specific_fn
// CHECK:STDOUT:   %impl.elem1 => constants.%Destroy.facet.84a
// CHECK:STDOUT:   %as_type => constants.%tuple.type.de6
// CHECK:STDOUT:   %require_complete.1 => constants.%complete_type.95d
// CHECK:STDOUT:   %ptr => constants.%ptr.b95
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type => constants.%Iterate.WithSelf.Next.type.4a6
// CHECK:STDOUT:   %.loc38_41.21 => constants.%.d53
// CHECK:STDOUT:   %impl.elem3.loc38_41.2 => constants.%T.as_type.as.Iterate.impl.Next.1b2
// CHECK:STDOUT:   %specific_impl_fn.loc38_41.5 => constants.%T.as_type.as.Iterate.impl.Next.specific_fn
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.544
// CHECK:STDOUT:   %.loc38_41.22 => constants.%.a9c
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.df9cc1.8
// CHECK:STDOUT:   %impl.elem0.loc38_41.2 => constants.%Destroy.Op.1a2547.8
// CHECK:STDOUT:   %specific_impl_fn.loc38_41.6 => constants.%Destroy.Op.1a2547.8
// CHECK:STDOUT: }
// CHECK:STDOUT:
