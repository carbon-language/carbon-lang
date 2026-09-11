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
  // CHECK:STDERR: {{.*}}/prelude/iterate.carbon:41:3: note: while building thunk to match the signature of this function [ThunkSignature]
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
  // CHECK:STDERR: {{.*}}/prelude/iterate.carbon:42:3: note: while building thunk to match the signature of this function [ThunkSignature]
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
  // CHECK:STDERR: {{.*}}/prelude/iterate.carbon:41:3: note: while building thunk to match the signature of this function [ThunkSignature]
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
  // CHECK:STDERR: {{.*}}/prelude/iterate.carbon:42:3: note: while building thunk to match the signature of this function [ThunkSignature]
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
  // CHECK:STDERR: {{.*}}/prelude/iterate.carbon:41:3: note: while building thunk to match the signature of this function [ThunkSignature]
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
  // CHECK:STDERR: {{.*}}/prelude/iterate.carbon:42:3: note: while building thunk to match the signature of this function [ThunkSignature]
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
  // CHECK:STDERR: {{.*}}/prelude/iterate.carbon:63:9: note: attempt to use a deleted function [CppInteropParseError]
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
  // CHECK:STDERR: {{.*}}/prelude/iterate.carbon:63:9: note: attempt to use a deleted function [CppInteropParseError]
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
  // CHECK:STDERR: {{.*}}/prelude/iterate.carbon:63:9: note: call to deleted function 'begin' [CppInteropParseError]
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
  // CHECK:STDERR: {{.*}}/prelude/iterate.carbon:63:9: note: call to deleted function 'end' [CppInteropParseError]
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
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
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
// CHECK:STDOUT:   %Int.type.facet: %type = facet_value %i32, () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.4ab: type = fn_type @Destroy.WithSelf.SelfDestruct, @Destroy.WithSelf(%Int.type.facet) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.228: %Destroy.WithSelf.SelfDestruct.type.4ab = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.fa6: type = fn_type @Destroy.WithSelf.SubobjectDestroy, @Destroy.WithSelf(%Int.type.facet) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.00a: %Destroy.WithSelf.SubobjectDestroy.type.fa6 = struct_value () [concrete]
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
// CHECK:STDOUT:   %custom_witness.1ca4d6.2: <witness> = custom_witness (%Destroy.Op.1a2547.3, %Destroy.WithSelf.SubobjectDestroy.00a, %Destroy.WithSelf.SelfDestruct.228), @Destroy [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.ac8: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N.fe9) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.5e0: %Int.as.Copy.impl.Op.type.ac8 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.b51: <witness> = impl_witness imports.%Copy.impl_witness_table.8d2, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %facet_value.b6b: %facet_type.f48 = facet_value %i32, (%custom_witness.1ca4d6.2, %Copy.impl_witness.b51) [concrete]
// CHECK:STDOUT:   %Iterate.lookup_impl_witness.1d8: <witness> = lookup_impl_witness %.Self.500, @Iterate [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.294: %facet_type.f48 = impl_witness_access %Iterate.lookup_impl_witness.1d8, element0 [symbolic_self]
// CHECK:STDOUT:   %Iterate_where.type.713: type = facet_type <@Iterate where %impl.elem0.294 = %facet_value.b6b> [concrete]
// CHECK:STDOUT:   %pattern_type.14a: type = pattern_type %Iterate_where.type.713 [concrete]
// CHECK:STDOUT:   %R.patt: %pattern_type.14a = symbolic_binding_pattern R, 0 [symbolic]
// CHECK:STDOUT:   %R: %Iterate_where.type.713 = symbolic_binding R, 0 [symbolic]
// CHECK:STDOUT:   %R.as_type: type = facet_access_type %R [symbolic]
// CHECK:STDOUT:   %pattern_type.6a1: type = pattern_type %R.as_type [symbolic]
// CHECK:STDOUT:   %r.param_patt.3b6: %pattern_type.6a1 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %r.patt.6d2: %pattern_type.6a1 = wrapper_binding_pattern r, %r.param_patt.3b6 [symbolic]
// CHECK:STDOUT:   %facet_type.037: type = facet_type <@IntFitsIn, @IntFitsIn(%Int.67af24.2) & @AnyInt> [symbolic]
// CHECK:STDOUT:   %From.6a4: %facet_type.037 = symbolic_binding From, 1 [symbolic]
// CHECK:STDOUT:   %From.as_type.as.ImplicitAs.impl.Convert.type.7fa: type = fn_type @From.as_type.as.ImplicitAs.impl.Convert, @From.as_type.as.ImplicitAs.impl(%To, %From.6a4) [symbolic]
// CHECK:STDOUT:   %From.as_type.as.ImplicitAs.impl.Convert.643: %From.as_type.as.ImplicitAs.impl.Convert.type.7fa = struct_value () [symbolic]
// CHECK:STDOUT:   %i.patt: %pattern_type.6b6 = value_binding_pattern i [concrete]
// CHECK:STDOUT:   %Iterate.lookup_impl_witness.f6e: <witness> = lookup_impl_witness %R, @Iterate [symbolic]
// CHECK:STDOUT:   %Iterate.facet.1af: %Iterate.type = facet_value %R.as_type, (%Iterate.lookup_impl_witness.f6e) [symbolic]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.0cb: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.1af) [symbolic]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.2d3: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.1af) [symbolic]
// CHECK:STDOUT:   %.3c2: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.0cb, %Iterate.facet.1af [symbolic]
// CHECK:STDOUT:   %impl.elem2.963: %.3c2 = impl_witness_access %Iterate.lookup_impl_witness.f6e, element2 [symbolic]
// CHECK:STDOUT:   %impl.elem1.4f1: %Destroy.type = impl_witness_access %Iterate.lookup_impl_witness.f6e, element1 [symbolic]
// CHECK:STDOUT:   %as_type.1d8: type = facet_access_type %impl.elem1.4f1 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.df8: <specific function> = specific_impl_function %impl.elem2.963, @Iterate.WithSelf.NewCursor(%Iterate.facet.1af) [symbolic]
// CHECK:STDOUT:   %ptr.93e: type = ptr_type %as_type.1d8 [symbolic]
// CHECK:STDOUT:   %.379: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.2d3, %Iterate.facet.1af [symbolic]
// CHECK:STDOUT:   %impl.elem3.606: %.379 = impl_witness_access %Iterate.lookup_impl_witness.f6e, element3 [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.5cb: type = class_type @DefaultOptionalStorage, @DefaultOptionalStorage(%facet_value.b6b) [concrete]
// CHECK:STDOUT:   %OptionalStorage.impl_witness.38d: <witness> = impl_witness imports.%OptionalStorage.impl_witness_table.460, @T.as_type.as.OptionalStorage.impl(%facet_value.b6b) [concrete]
// CHECK:STDOUT:   %OptionalStorage.facet.53e: %OptionalStorage.type = facet_value %i32, (%OptionalStorage.impl_witness.38d) [concrete]
// CHECK:STDOUT:   %Optional.f49: type = class_type @Optional, @Optional(%OptionalStorage.facet.53e) [concrete]
// CHECK:STDOUT:   %specific_impl_fn.104: <specific function> = specific_impl_function %impl.elem3.606, @Iterate.WithSelf.Next(%Iterate.facet.1af) [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.type.91a: type = fn_type @Optional.HasValue, @Optional(%OptionalStorage.facet.53e) [concrete]
// CHECK:STDOUT:   %Optional.HasValue.fe5: %Optional.HasValue.type.91a = struct_value () [concrete]
// CHECK:STDOUT:   %Optional.Get.type.6f2: type = fn_type @Optional.Get, @Optional(%OptionalStorage.facet.53e) [concrete]
// CHECK:STDOUT:   %Optional.Get.1f4: %Optional.Get.type.6f2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.facet.81c918.2: %Destroy.type = facet_value %i32, (%custom_witness.1ca4d6.2) [concrete]
// CHECK:STDOUT:   %MaybeUnformed.0a3: type = class_type @MaybeUnformed, @MaybeUnformed(%Destroy.facet.81c918.2) [concrete]
// CHECK:STDOUT:   %.1d6a7: type = maybe_unformed_type %i32 [concrete]
// CHECK:STDOUT:   %struct_type.value.has_value.fd1: type = struct_type {.value: %MaybeUnformed.0a3, .has_value: bool} [concrete]
// CHECK:STDOUT:   %Optional.HasValue.specific_fn.fb4: <specific function> = specific_function %Optional.HasValue.fe5, @Optional.HasValue(%OptionalStorage.facet.53e) [concrete]
// CHECK:STDOUT:   %Optional.Get.specific_fn.9da: <specific function> = specific_function %Optional.Get.1f4, @Optional.Get(%OptionalStorage.facet.53e) [concrete]
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
// CHECK:STDOUT:   %Optional.type.facet: %type = facet_value %Optional.f49, () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.4e4: type = fn_type @Destroy.WithSelf.SubobjectDestroy, @Destroy.WithSelf(%Optional.type.facet) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.67e: %Destroy.WithSelf.SubobjectDestroy.type.4e4 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.e29: type = fn_type @Destroy.WithSelf.SelfDestruct, @Destroy.WithSelf(%Optional.type.facet) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bd2: %Destroy.WithSelf.SelfDestruct.type.e29 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.5eb: <witness> = custom_witness (%Destroy.Op.1a2547.9, %Destroy.WithSelf.SubobjectDestroy.67e, %Destroy.WithSelf.SelfDestruct.bd2), @Destroy [concrete]
// CHECK:STDOUT:   %Destroy.facet.608: %Destroy.type = facet_value %Optional.f49, (%custom_witness.5eb) [concrete]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.2c4: <witness> = lookup_impl_witness %impl.elem1.4f1, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.9e0: type = fn_type @Destroy.WithSelf.SelfDestruct, @Destroy.WithSelf(%impl.elem1.4f1) [symbolic]
// CHECK:STDOUT:   %.97b: type = fn_type_with_self_type %Destroy.WithSelf.SelfDestruct.type.9e0, %impl.elem1.4f1 [symbolic]
// CHECK:STDOUT:   %impl.elem2.218: %.97b = impl_witness_access %Destroy.lookup_impl_witness.2c4, element2 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.057: <specific function> = specific_impl_function %impl.elem2.218, @Destroy.WithSelf.SelfDestruct(%impl.elem1.4f1) [symbolic]
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
// CHECK:STDOUT:   %Iterator.type.facet.919: %type = facet_value %Iterator.3ee, () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.5ef: type = fn_type @Destroy.WithSelf.SubobjectDestroy, @Destroy.WithSelf(%Iterator.type.facet.919) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.fc5: %Destroy.WithSelf.SubobjectDestroy.type.5ef = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.a71: type = fn_type @Destroy.WithSelf.SelfDestruct, @Destroy.WithSelf(%Iterator.type.facet.919) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.441: %Destroy.WithSelf.SelfDestruct.type.a71 = struct_value () [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.0aed5d.2: type = fn_type @Iterator.Op.2 [concrete]
// CHECK:STDOUT:   %Iterator.Op.a2a866.2: %Iterator.Op.type.0aed5d.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.f19: <witness> = custom_witness (%Iterator.Op.a2a866.2, %Destroy.WithSelf.SubobjectDestroy.fc5, %Destroy.WithSelf.SelfDestruct.441), @Destroy [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.0aed5d.3: type = fn_type @Iterator.Op.3 [concrete]
// CHECK:STDOUT:   %Iterator.Op.a2a866.3: %Iterator.Op.type.0aed5d.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.444: <witness> = custom_witness (%Iterator.Op.a2a866.3), @Inc [concrete]
// CHECK:STDOUT:   %Equal.type.ca8475.1: type = fn_type @Equal.1 [concrete]
// CHECK:STDOUT:   %Equal.285b15.1: %Equal.type.ca8475.1 = struct_value () [concrete]
// CHECK:STDOUT:   %NotEqual.type.53d9ee.1: type = fn_type @NotEqual.1 [concrete]
// CHECK:STDOUT:   %NotEqual.119263.1: %NotEqual.type.53d9ee.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.981: <witness> = custom_witness (%Equal.285b15.1, %NotEqual.119263.1), @EqWith, @EqWith(%Iterator.3ee) [concrete]
// CHECK:STDOUT:   %facet_value.736: %CppRange_where.type.d682d6.1 = facet_value %MutableRange, (%custom_witness.1ca4d6.2, %Copy.impl_witness.b51, %custom_witness.ed3, %custom_witness.f19, %custom_witness.ef8, %custom_witness.444, %custom_witness.981) [concrete]
// CHECK:STDOUT:   %tuple.type.508: type = tuple_type (%Iterator.3ee, %Iterator.3ee) [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.10: type = fn_type @Destroy.Op.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.10: %Destroy.Op.type.1d8f74.10 = struct_value () [concrete]
// CHECK:STDOUT:   %tuple.type.facet.f51: %type = facet_value %tuple.type.508, () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.44e: type = fn_type @Destroy.WithSelf.SubobjectDestroy, @Destroy.WithSelf(%tuple.type.facet.f51) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.74d: %Destroy.WithSelf.SubobjectDestroy.type.44e = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.1fc: type = fn_type @Destroy.WithSelf.SelfDestruct, @Destroy.WithSelf(%tuple.type.facet.f51) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.36e: %Destroy.WithSelf.SelfDestruct.type.1fc = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.c43: <witness> = custom_witness (%Destroy.Op.1a2547.10, %Destroy.WithSelf.SubobjectDestroy.74d, %Destroy.WithSelf.SelfDestruct.36e), @Destroy [concrete]
// CHECK:STDOUT:   %Destroy.facet.f27: %Destroy.type = facet_value %tuple.type.508, (%custom_witness.c43) [concrete]
// CHECK:STDOUT:   %Iterate.impl_witness.647: <witness> = impl_witness imports.%Iterate.impl_witness_table.121, @T.as_type.as.Iterate.impl(%facet_value.736) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.type.2bb: type = fn_type @T.as_type.as.Iterate.impl.NewCursor, @T.as_type.as.Iterate.impl(%facet_value.736) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.45c: %T.as_type.as.Iterate.impl.NewCursor.type.2bb = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.type.2ba: type = fn_type @T.as_type.as.Iterate.impl.Next, @T.as_type.as.Iterate.impl(%facet_value.736) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.23e: %T.as_type.as.Iterate.impl.Next.type.2ba = struct_value () [concrete]
// CHECK:STDOUT:   %Iterate.facet.0d0: %Iterate.type = facet_value %MutableRange, (%Iterate.impl_witness.647) [concrete]
// CHECK:STDOUT:   %facet_value.8bb: %Iterate_where.type.713 = facet_value %MutableRange, (%Iterate.impl_witness.647) [concrete]
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
// CHECK:STDOUT:   %Iterator.type.facet.b3d: %type = facet_value %Iterator.260, () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.651: type = fn_type @Destroy.WithSelf.SubobjectDestroy, @Destroy.WithSelf(%Iterator.type.facet.b3d) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.571: %Destroy.WithSelf.SubobjectDestroy.type.651 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.1ce: type = fn_type @Destroy.WithSelf.SelfDestruct, @Destroy.WithSelf(%Iterator.type.facet.b3d) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.3e8: %Destroy.WithSelf.SelfDestruct.type.1ce = struct_value () [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.0f34b8.2: type = fn_type @Iterator.Op.5 [concrete]
// CHECK:STDOUT:   %Iterator.Op.1c3e4a.2: %Iterator.Op.type.0f34b8.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.952: <witness> = custom_witness (%Iterator.Op.1c3e4a.2, %Destroy.WithSelf.SubobjectDestroy.571, %Destroy.WithSelf.SelfDestruct.3e8), @Destroy [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.0f34b8.3: type = fn_type @Iterator.Op.6 [concrete]
// CHECK:STDOUT:   %Iterator.Op.1c3e4a.3: %Iterator.Op.type.0f34b8.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.746: <witness> = custom_witness (%Iterator.Op.1c3e4a.3), @Inc [concrete]
// CHECK:STDOUT:   %Equal.type.ca8475.2: type = fn_type @Equal.2 [concrete]
// CHECK:STDOUT:   %Equal.285b15.2: %Equal.type.ca8475.2 = struct_value () [concrete]
// CHECK:STDOUT:   %NotEqual.type.53d9ee.2: type = fn_type @NotEqual.2 [concrete]
// CHECK:STDOUT:   %NotEqual.119263.2: %NotEqual.type.53d9ee.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.941: <witness> = custom_witness (%Equal.285b15.2, %NotEqual.119263.2), @EqWith, @EqWith(%Iterator.260) [concrete]
// CHECK:STDOUT:   %facet_value.2c1: %CppRange_where.type.d682d6.1 = facet_value %ConstRange, (%custom_witness.1ca4d6.2, %Copy.impl_witness.b51, %custom_witness.23e, %custom_witness.952, %custom_witness.adb, %custom_witness.746, %custom_witness.941) [concrete]
// CHECK:STDOUT:   %tuple.type.69f: type = tuple_type (%Iterator.260, %Iterator.260) [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.11: type = fn_type @Destroy.Op.3 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.11: %Destroy.Op.type.1d8f74.11 = struct_value () [concrete]
// CHECK:STDOUT:   %tuple.type.facet.d54: %type = facet_value %tuple.type.69f, () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.98d: type = fn_type @Destroy.WithSelf.SubobjectDestroy, @Destroy.WithSelf(%tuple.type.facet.d54) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.4fe: %Destroy.WithSelf.SubobjectDestroy.type.98d = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.692: type = fn_type @Destroy.WithSelf.SelfDestruct, @Destroy.WithSelf(%tuple.type.facet.d54) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.05a: %Destroy.WithSelf.SelfDestruct.type.692 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.621: <witness> = custom_witness (%Destroy.Op.1a2547.11, %Destroy.WithSelf.SubobjectDestroy.4fe, %Destroy.WithSelf.SelfDestruct.05a), @Destroy [concrete]
// CHECK:STDOUT:   %Destroy.facet.c06: %Destroy.type = facet_value %tuple.type.69f, (%custom_witness.621) [concrete]
// CHECK:STDOUT:   %Iterate.impl_witness.eb6: <witness> = impl_witness imports.%Iterate.impl_witness_table.121, @T.as_type.as.Iterate.impl(%facet_value.2c1) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.type.d32: type = fn_type @T.as_type.as.Iterate.impl.NewCursor, @T.as_type.as.Iterate.impl(%facet_value.2c1) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.930: %T.as_type.as.Iterate.impl.NewCursor.type.d32 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.type.fb0: type = fn_type @T.as_type.as.Iterate.impl.Next, @T.as_type.as.Iterate.impl(%facet_value.2c1) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.305: %T.as_type.as.Iterate.impl.Next.type.fb0 = struct_value () [concrete]
// CHECK:STDOUT:   %Iterate.facet.7f1: %Iterate.type = facet_value %ConstRange, (%Iterate.impl_witness.eb6) [concrete]
// CHECK:STDOUT:   %facet_value.d9f: %Iterate_where.type.713 = facet_value %ConstRange, (%Iterate.impl_witness.eb6) [concrete]
// CHECK:STDOUT:   %r.param_patt.83e: %pattern_type.0ba = value_param_pattern [concrete]
// CHECK:STDOUT:   %r.patt.c21: %pattern_type.0ba = wrapper_binding_pattern r, %r.param_patt.83e [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.969: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.0d0) [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.1bb: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.0d0) [concrete]
// CHECK:STDOUT:   %.ccb: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.969, %Iterate.facet.0d0 [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.specific_fn.440: <specific function> = specific_function %T.as_type.as.Iterate.impl.NewCursor.45c, @T.as_type.as.Iterate.impl.NewCursor(%facet_value.736) [concrete]
// CHECK:STDOUT:   %ptr.45c: type = ptr_type %tuple.type.508 [concrete]
// CHECK:STDOUT:   %.343: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.1bb, %Iterate.facet.0d0 [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.specific_fn.e33: <specific function> = specific_function %T.as_type.as.Iterate.impl.Next.23e, @T.as_type.as.Iterate.impl.Next(%facet_value.736) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.c02: type = fn_type @Destroy.WithSelf.SelfDestruct, @Destroy.WithSelf(%Destroy.facet.f27) [concrete]
// CHECK:STDOUT:   %.d19: type = fn_type_with_self_type %Destroy.WithSelf.SelfDestruct.type.c02, %Destroy.facet.f27 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.specific_fn.079: <specific function> = specific_function %Destroy.WithSelf.SelfDestruct.36e, @Destroy.WithSelf.SelfDestruct(%tuple.type.facet.f51) [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.165: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.7f1) [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.d6c: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.7f1) [concrete]
// CHECK:STDOUT:   %.81c: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.165, %Iterate.facet.7f1 [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.specific_fn.617: <specific function> = specific_function %T.as_type.as.Iterate.impl.NewCursor.930, @T.as_type.as.Iterate.impl.NewCursor(%facet_value.2c1) [concrete]
// CHECK:STDOUT:   %ptr.c5c: type = ptr_type %tuple.type.69f [concrete]
// CHECK:STDOUT:   %.4e2: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.d6c, %Iterate.facet.7f1 [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.specific_fn.52b: <specific function> = specific_function %T.as_type.as.Iterate.impl.Next.305, @T.as_type.as.Iterate.impl.Next(%facet_value.2c1) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.d17: type = fn_type @Destroy.WithSelf.SelfDestruct, @Destroy.WithSelf(%Destroy.facet.c06) [concrete]
// CHECK:STDOUT:   %.2c0: type = fn_type_with_self_type %Destroy.WithSelf.SelfDestruct.type.d17, %Destroy.facet.c06 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.specific_fn.199: <specific function> = specific_function %Destroy.WithSelf.SelfDestruct.05a, @Destroy.WithSelf.SelfDestruct(%tuple.type.facet.d54) [concrete]
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
// CHECK:STDOUT: generic fn @TestIterate(%R.loc30_17.2: %Iterate_where.type.713) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Iterate.lookup_impl_witness: <witness> = lookup_impl_witness %R.loc30_17.1, @Iterate [symbolic = %Iterate.lookup_impl_witness (constants.%Iterate.lookup_impl_witness.f6e)]
// CHECK:STDOUT:   %Iterate.facet.loc34_19.5: %Iterate.type = facet_value %R.as_type.loc30_61.1, (%Iterate.lookup_impl_witness) [symbolic = %Iterate.facet.loc34_19.5 (constants.%Iterate.facet.1af)]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.loc34_19.5) [symbolic = %Iterate.WithSelf.NewCursor.type (constants.%Iterate.WithSelf.NewCursor.type.0cb)]
// CHECK:STDOUT:   %.loc34_19.23: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type, %Iterate.facet.loc34_19.5 [symbolic = %.loc34_19.23 (constants.%.3c2)]
// CHECK:STDOUT:   %impl.elem2.loc34_19.3: @TestIterate.%.loc34_19.23 (%.3c2) = impl_witness_access %Iterate.lookup_impl_witness, element2 [symbolic = %impl.elem2.loc34_19.3 (constants.%impl.elem2.963)]
// CHECK:STDOUT:   %specific_impl_fn.loc34_19.4: <specific function> = specific_impl_function %impl.elem2.loc34_19.3, @Iterate.WithSelf.NewCursor(%Iterate.facet.loc34_19.5) [symbolic = %specific_impl_fn.loc34_19.4 (constants.%specific_impl_fn.df8)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %ptr: type = ptr_type %as_type [symbolic = %ptr (constants.%ptr.93e)]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.loc34_19.5) [symbolic = %Iterate.WithSelf.Next.type (constants.%Iterate.WithSelf.Next.type.2d3)]
// CHECK:STDOUT:   %.loc34_19.24: type = fn_type_with_self_type %Iterate.WithSelf.Next.type, %Iterate.facet.loc34_19.5 [symbolic = %.loc34_19.24 (constants.%.379)]
// CHECK:STDOUT:   %impl.elem3.loc34_19.2: @TestIterate.%.loc34_19.24 (%.379) = impl_witness_access %Iterate.lookup_impl_witness, element3 [symbolic = %impl.elem3.loc34_19.2 (constants.%impl.elem3.606)]
// CHECK:STDOUT:   %specific_impl_fn.loc34_19.5: <specific function> = specific_impl_function %impl.elem3.loc34_19.2, @Iterate.WithSelf.Next(%Iterate.facet.loc34_19.5) [symbolic = %specific_impl_fn.loc34_19.5 (constants.%specific_impl_fn.104)]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type: type = fn_type @Destroy.WithSelf.SelfDestruct, @Destroy.WithSelf(%impl.elem1) [symbolic = %Destroy.WithSelf.SelfDestruct.type (constants.%Destroy.WithSelf.SelfDestruct.type.9e0)]
// CHECK:STDOUT:   %.loc34_19.25: type = fn_type_with_self_type %Destroy.WithSelf.SelfDestruct.type, %impl.elem1 [symbolic = %.loc34_19.25 (constants.%.97b)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %impl.elem1, @Destroy [symbolic = %Destroy.lookup_impl_witness (constants.%Destroy.lookup_impl_witness.2c4)]
// CHECK:STDOUT:   %impl.elem2.loc34_19.4: @TestIterate.%.loc34_19.25 (%.97b) = impl_witness_access %Destroy.lookup_impl_witness, element2 [symbolic = %impl.elem2.loc34_19.4 (constants.%impl.elem2.218)]
// CHECK:STDOUT:   %specific_impl_fn.loc34_19.6: <specific function> = specific_impl_function %impl.elem2.loc34_19.4, @Destroy.WithSelf.SelfDestruct(%impl.elem1) [symbolic = %specific_impl_fn.loc34_19.6 (constants.%specific_impl_fn.057)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%r.param: @TestIterate.%R.as_type.loc30_61.1 (%R.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %i.patt: %pattern_type.6b6 = value_binding_pattern i [concrete = constants.%i.patt]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %r.ref: @TestIterate.%R.as_type.loc30_61.1 (%R.as_type) = name_ref r, %r
// CHECK:STDOUT:     %impl.elem2.loc34_19.1: @TestIterate.%.loc34_19.23 (%.3c2) = impl_witness_access constants.%Iterate.lookup_impl_witness.f6e, element2 [symbolic = %impl.elem2.loc34_19.3 (constants.%impl.elem2.963)]
// CHECK:STDOUT:     %bound_method.loc34_19.1: <bound method> = bound_method %r.ref, %impl.elem2.loc34_19.1
// CHECK:STDOUT:     %Iterate.facet.loc34_19.1: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.f6e) [symbolic = %Iterate.facet.loc34_19.5 (constants.%Iterate.facet.1af)]
// CHECK:STDOUT:     %.loc34_19.1: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc34_19.1 [symbolic = %Iterate.facet.loc34_19.5 (constants.%Iterate.facet.1af)]
// CHECK:STDOUT:     %Iterate.facet.loc34_19.2: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.f6e) [symbolic = %Iterate.facet.loc34_19.5 (constants.%Iterate.facet.1af)]
// CHECK:STDOUT:     %.loc34_19.2: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc34_19.2 [symbolic = %Iterate.facet.loc34_19.5 (constants.%Iterate.facet.1af)]
// CHECK:STDOUT:     %specific_impl_fn.loc34_19.1: <specific function> = specific_impl_function %impl.elem2.loc34_19.1, @Iterate.WithSelf.NewCursor(constants.%Iterate.facet.1af) [symbolic = %specific_impl_fn.loc34_19.4 (constants.%specific_impl_fn.df8)]
// CHECK:STDOUT:     %bound_method.loc34_19.2: <bound method> = bound_method %r.ref, %specific_impl_fn.loc34_19.1
// CHECK:STDOUT:     %var: ref @TestIterate.%as_type (%as_type.1d8) = var_storage invalid
// CHECK:STDOUT:     %Iterate.WithSelf.NewCursor.call: init @TestIterate.%as_type (%as_type.1d8) to %var = call %bound_method.loc34_19.2(%r.ref)
// CHECK:STDOUT:     assign %var, %Iterate.WithSelf.NewCursor.call
// CHECK:STDOUT:     br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.next:
// CHECK:STDOUT:     %addr: @TestIterate.%ptr (%ptr.93e) = addr_of %var
// CHECK:STDOUT:     %impl.elem3.loc34_19.1: @TestIterate.%.loc34_19.24 (%.379) = impl_witness_access constants.%Iterate.lookup_impl_witness.f6e, element3 [symbolic = %impl.elem3.loc34_19.2 (constants.%impl.elem3.606)]
// CHECK:STDOUT:     %bound_method.loc34_19.3: <bound method> = bound_method %r.ref, %impl.elem3.loc34_19.1
// CHECK:STDOUT:     %Iterate.facet.loc34_19.3: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.f6e) [symbolic = %Iterate.facet.loc34_19.5 (constants.%Iterate.facet.1af)]
// CHECK:STDOUT:     %.loc34_19.3: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc34_19.3 [symbolic = %Iterate.facet.loc34_19.5 (constants.%Iterate.facet.1af)]
// CHECK:STDOUT:     %Iterate.facet.loc34_19.4: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.f6e) [symbolic = %Iterate.facet.loc34_19.5 (constants.%Iterate.facet.1af)]
// CHECK:STDOUT:     %.loc34_19.4: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc34_19.4 [symbolic = %Iterate.facet.loc34_19.5 (constants.%Iterate.facet.1af)]
// CHECK:STDOUT:     %.loc34_19.5: %Destroy.type = converted constants.%as_type.1d8, constants.%impl.elem1.4f1 [symbolic = %impl.elem1 (constants.%impl.elem1.4f1)]
// CHECK:STDOUT:     %.loc34_19.6: %Destroy.type = converted constants.%as_type.1d8, constants.%impl.elem1.4f1 [symbolic = %impl.elem1 (constants.%impl.elem1.4f1)]
// CHECK:STDOUT:     %specific_impl_fn.loc34_19.2: <specific function> = specific_impl_function %impl.elem3.loc34_19.1, @Iterate.WithSelf.Next(constants.%Iterate.facet.1af) [symbolic = %specific_impl_fn.loc34_19.5 (constants.%specific_impl_fn.104)]
// CHECK:STDOUT:     %bound_method.loc34_19.4: <bound method> = bound_method %r.ref, %specific_impl_fn.loc34_19.2
// CHECK:STDOUT:     %.loc34_19.7: ref %Optional.f49 = temporary_storage
// CHECK:STDOUT:     %Iterate.WithSelf.Next.call: init %Optional.f49 to %.loc34_19.7 = call %bound_method.loc34_19.4(%r.ref, %addr)
// CHECK:STDOUT:     %.loc34_19.8: ref %Optional.f49 = temporary %.loc34_19.7, %Iterate.WithSelf.Next.call
// CHECK:STDOUT:     %.loc34_19.9: %Optional.HasValue.type.91a = specific_constant imports.%Core.import_ref.bb6, @Optional(constants.%OptionalStorage.facet.53e) [concrete = constants.%Optional.HasValue.fe5]
// CHECK:STDOUT:     %HasValue.ref: %Optional.HasValue.type.91a = name_ref HasValue, %.loc34_19.9 [concrete = constants.%Optional.HasValue.fe5]
// CHECK:STDOUT:     %Optional.HasValue.bound: <bound method> = bound_method %.loc34_19.8, %HasValue.ref
// CHECK:STDOUT:     %Optional.HasValue.specific_fn: <specific function> = specific_function %HasValue.ref, @Optional.HasValue(constants.%OptionalStorage.facet.53e) [concrete = constants.%Optional.HasValue.specific_fn.fb4]
// CHECK:STDOUT:     %bound_method.loc34_19.5: <bound method> = bound_method %.loc34_19.8, %Optional.HasValue.specific_fn
// CHECK:STDOUT:     %.loc34_19.10: %Optional.f49 = acquire_value %.loc34_19.8
// CHECK:STDOUT:     %Optional.HasValue.call: init bool = call %bound_method.loc34_19.5(%.loc34_19.10)
// CHECK:STDOUT:     %.loc34_19.11: bool = value_of_initializer %Optional.HasValue.call
// CHECK:STDOUT:     %.loc34_19.12: bool = converted %Optional.HasValue.call, %.loc34_19.11
// CHECK:STDOUT:     if %.loc34_19.12 br !for.body else br !for.done
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.body:
// CHECK:STDOUT:     %.loc34_19.13: %Optional.Get.type.6f2 = specific_constant imports.%Core.import_ref.dab, @Optional(constants.%OptionalStorage.facet.53e) [concrete = constants.%Optional.Get.1f4]
// CHECK:STDOUT:     %Get.ref: %Optional.Get.type.6f2 = name_ref Get, %.loc34_19.13 [concrete = constants.%Optional.Get.1f4]
// CHECK:STDOUT:     %Optional.Get.bound: <bound method> = bound_method %.loc34_19.8, %Get.ref
// CHECK:STDOUT:     %Optional.Get.specific_fn: <specific function> = specific_function %Get.ref, @Optional.Get(constants.%OptionalStorage.facet.53e) [concrete = constants.%Optional.Get.specific_fn.9da]
// CHECK:STDOUT:     %bound_method.loc34_19.6: <bound method> = bound_method %.loc34_19.8, %Optional.Get.specific_fn
// CHECK:STDOUT:     %.loc34_19.14: %Optional.f49 = acquire_value %.loc34_19.8
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
// CHECK:STDOUT:     %Destroy.WithSelf.SelfDestruct.bound.loc34_19.1: <bound method> = bound_method %.loc34_19.8, constants.%Destroy.WithSelf.SelfDestruct.bd2
// CHECK:STDOUT:     %Destroy.facet.loc34_19.1: %Destroy.type = facet_value constants.%Optional.f49, (constants.%custom_witness.5eb) [concrete = constants.%Destroy.facet.608]
// CHECK:STDOUT:     %.loc34_19.17: %Destroy.type = converted constants.%Optional.f49, %Destroy.facet.loc34_19.1 [concrete = constants.%Destroy.facet.608]
// CHECK:STDOUT:     %Destroy.facet.loc34_19.2: %Destroy.type = facet_value constants.%Optional.f49, (constants.%custom_witness.5eb) [concrete = constants.%Destroy.facet.608]
// CHECK:STDOUT:     %.loc34_19.18: %Destroy.type = converted constants.%Optional.f49, %Destroy.facet.loc34_19.2 [concrete = constants.%Destroy.facet.608]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %bound_method.loc34_19.7: <bound method> = bound_method %.loc34_19.8, %Destroy.WithSelf.SelfDestruct.specific_fn.1
// CHECK:STDOUT:     %Destroy.WithSelf.SelfDestruct.call.loc34_19.1: init %empty_tuple.type = call %bound_method.loc34_19.7(%.loc34_19.8)
// CHECK:STDOUT:     br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.done:
// CHECK:STDOUT:     %Destroy.WithSelf.SelfDestruct.bound.loc34_19.2: <bound method> = bound_method %.loc34_19.8, constants.%Destroy.WithSelf.SelfDestruct.bd2
// CHECK:STDOUT:     %Destroy.facet.loc34_19.3: %Destroy.type = facet_value constants.%Optional.f49, (constants.%custom_witness.5eb) [concrete = constants.%Destroy.facet.608]
// CHECK:STDOUT:     %.loc34_19.19: %Destroy.type = converted constants.%Optional.f49, %Destroy.facet.loc34_19.3 [concrete = constants.%Destroy.facet.608]
// CHECK:STDOUT:     %Destroy.facet.loc34_19.4: %Destroy.type = facet_value constants.%Optional.f49, (constants.%custom_witness.5eb) [concrete = constants.%Destroy.facet.608]
// CHECK:STDOUT:     %.loc34_19.20: %Destroy.type = converted constants.%Optional.f49, %Destroy.facet.loc34_19.4 [concrete = constants.%Destroy.facet.608]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %bound_method.loc34_19.8: <bound method> = bound_method %.loc34_19.8, %Destroy.WithSelf.SelfDestruct.specific_fn.2
// CHECK:STDOUT:     %Destroy.WithSelf.SelfDestruct.call.loc34_19.2: init %empty_tuple.type = call %bound_method.loc34_19.8(%.loc34_19.8)
// CHECK:STDOUT:     %impl.elem2.loc34_19.2: @TestIterate.%.loc34_19.25 (%.97b) = impl_witness_access constants.%Destroy.lookup_impl_witness.2c4, element2 [symbolic = %impl.elem2.loc34_19.4 (constants.%impl.elem2.218)]
// CHECK:STDOUT:     %bound_method.loc34_19.9: <bound method> = bound_method %var, %impl.elem2.loc34_19.2
// CHECK:STDOUT:     %.loc34_19.21: %Destroy.type = converted constants.%as_type.1d8, constants.%impl.elem1.4f1 [symbolic = %impl.elem1 (constants.%impl.elem1.4f1)]
// CHECK:STDOUT:     %.loc34_19.22: %Destroy.type = converted constants.%as_type.1d8, constants.%impl.elem1.4f1 [symbolic = %impl.elem1 (constants.%impl.elem1.4f1)]
// CHECK:STDOUT:     %specific_impl_fn.loc34_19.3: <specific function> = specific_impl_function %impl.elem2.loc34_19.2, @Destroy.WithSelf.SelfDestruct(constants.%impl.elem1.4f1) [symbolic = %specific_impl_fn.loc34_19.6 (constants.%specific_impl_fn.057)]
// CHECK:STDOUT:     %bound_method.loc34_19.10: <bound method> = bound_method %var, %specific_impl_fn.loc34_19.3
// CHECK:STDOUT:     %Destroy.WithSelf.SelfDestruct.call.loc34_19.3: init %empty_tuple.type = call %bound_method.loc34_19.10(%var)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc34_19.1(%self.param: ref %.1d6a7) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc34_19.2(%self.param: ref %MaybeUnformed.0a3) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc34_19.3(%self.param: ref bool) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc34_19.4(%self.param: ref %struct_type.value.has_value.fd1) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc34_19.5(%self.param: ref %DefaultOptionalStorage.5cb) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc34_19.6(%self.param: ref %Optional.f49) {
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
// CHECK:STDOUT:   %impl.elem2.loc49: %.ccb = impl_witness_access constants.%Iterate.impl_witness.647, element2 [concrete = constants.%T.as_type.as.Iterate.impl.NewCursor.45c]
// CHECK:STDOUT:   %bound_method.loc49_19.1: <bound method> = bound_method %m.ref, %impl.elem2.loc49
// CHECK:STDOUT:   %facet_value.loc49_19.1: %CppRange_where.type.d682d6.1 = facet_value constants.%MutableRange, (constants.%custom_witness.1ca4d6.2, constants.%Copy.impl_witness.b51, constants.%custom_witness.ed3, constants.%custom_witness.f19, constants.%custom_witness.ef8, constants.%custom_witness.444, constants.%custom_witness.981) [concrete = constants.%facet_value.736]
// CHECK:STDOUT:   %.loc49_19.1: %CppRange_where.type.d682d6.1 = converted constants.%MutableRange, %facet_value.loc49_19.1 [concrete = constants.%facet_value.736]
// CHECK:STDOUT:   %facet_value.loc49_19.2: %CppRange_where.type.d682d6.1 = facet_value constants.%MutableRange, (constants.%custom_witness.1ca4d6.2, constants.%Copy.impl_witness.b51, constants.%custom_witness.ed3, constants.%custom_witness.f19, constants.%custom_witness.ef8, constants.%custom_witness.444, constants.%custom_witness.981) [concrete = constants.%facet_value.736]
// CHECK:STDOUT:   %.loc49_19.2: %CppRange_where.type.d682d6.1 = converted constants.%MutableRange, %facet_value.loc49_19.2 [concrete = constants.%facet_value.736]
// CHECK:STDOUT:   %specific_fn.loc49_19.1: <specific function> = specific_function %impl.elem2.loc49, @T.as_type.as.Iterate.impl.NewCursor(constants.%facet_value.736) [concrete = constants.%T.as_type.as.Iterate.impl.NewCursor.specific_fn.440]
// CHECK:STDOUT:   %bound_method.loc49_19.2: <bound method> = bound_method %m.ref, %specific_fn.loc49_19.1
// CHECK:STDOUT:   %var.loc49: ref %tuple.type.508 = var_storage invalid
// CHECK:STDOUT:   %.loc49_18.1: %MutableRange = acquire_value %m.ref
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.call.loc49: init %tuple.type.508 to %var.loc49 = call %bound_method.loc49_19.2(%.loc49_18.1)
// CHECK:STDOUT:   assign %var.loc49, %T.as_type.as.Iterate.impl.NewCursor.call.loc49
// CHECK:STDOUT:   br !for.next.loc49
// CHECK:STDOUT:
// CHECK:STDOUT: !for.next.loc49:
// CHECK:STDOUT:   %addr.loc49: %ptr.45c = addr_of %var.loc49
// CHECK:STDOUT:   %impl.elem3.loc49: %.343 = impl_witness_access constants.%Iterate.impl_witness.647, element3 [concrete = constants.%T.as_type.as.Iterate.impl.Next.23e]
// CHECK:STDOUT:   %bound_method.loc49_19.3: <bound method> = bound_method %m.ref, %impl.elem3.loc49
// CHECK:STDOUT:   %facet_value.loc49_19.3: %CppRange_where.type.d682d6.1 = facet_value constants.%MutableRange, (constants.%custom_witness.1ca4d6.2, constants.%Copy.impl_witness.b51, constants.%custom_witness.ed3, constants.%custom_witness.f19, constants.%custom_witness.ef8, constants.%custom_witness.444, constants.%custom_witness.981) [concrete = constants.%facet_value.736]
// CHECK:STDOUT:   %.loc49_19.3: %CppRange_where.type.d682d6.1 = converted constants.%MutableRange, %facet_value.loc49_19.3 [concrete = constants.%facet_value.736]
// CHECK:STDOUT:   %facet_value.loc49_19.4: %CppRange_where.type.d682d6.1 = facet_value constants.%MutableRange, (constants.%custom_witness.1ca4d6.2, constants.%Copy.impl_witness.b51, constants.%custom_witness.ed3, constants.%custom_witness.f19, constants.%custom_witness.ef8, constants.%custom_witness.444, constants.%custom_witness.981) [concrete = constants.%facet_value.736]
// CHECK:STDOUT:   %.loc49_19.4: %CppRange_where.type.d682d6.1 = converted constants.%MutableRange, %facet_value.loc49_19.4 [concrete = constants.%facet_value.736]
// CHECK:STDOUT:   %specific_fn.loc49_19.2: <specific function> = specific_function %impl.elem3.loc49, @T.as_type.as.Iterate.impl.Next(constants.%facet_value.736) [concrete = constants.%T.as_type.as.Iterate.impl.Next.specific_fn.e33]
// CHECK:STDOUT:   %bound_method.loc49_19.4: <bound method> = bound_method %m.ref, %specific_fn.loc49_19.2
// CHECK:STDOUT:   %.loc49_19.5: ref %Optional.f49 = temporary_storage
// CHECK:STDOUT:   %.loc49_18.2: %MutableRange = acquire_value %m.ref
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.call.loc49: init %Optional.f49 to %.loc49_19.5 = call %bound_method.loc49_19.4(%.loc49_18.2, %addr.loc49)
// CHECK:STDOUT:   %.loc49_19.6: ref %Optional.f49 = temporary %.loc49_19.5, %T.as_type.as.Iterate.impl.Next.call.loc49
// CHECK:STDOUT:   %.loc49_19.7: %Optional.HasValue.type.91a = specific_constant imports.%Core.import_ref.bb6, @Optional(constants.%OptionalStorage.facet.53e) [concrete = constants.%Optional.HasValue.fe5]
// CHECK:STDOUT:   %HasValue.ref.loc49: %Optional.HasValue.type.91a = name_ref HasValue, %.loc49_19.7 [concrete = constants.%Optional.HasValue.fe5]
// CHECK:STDOUT:   %Optional.HasValue.bound.loc49: <bound method> = bound_method %.loc49_19.6, %HasValue.ref.loc49
// CHECK:STDOUT:   %Optional.HasValue.specific_fn.loc49: <specific function> = specific_function %HasValue.ref.loc49, @Optional.HasValue(constants.%OptionalStorage.facet.53e) [concrete = constants.%Optional.HasValue.specific_fn.fb4]
// CHECK:STDOUT:   %bound_method.loc49_19.5: <bound method> = bound_method %.loc49_19.6, %Optional.HasValue.specific_fn.loc49
// CHECK:STDOUT:   %.loc49_19.8: %Optional.f49 = acquire_value %.loc49_19.6
// CHECK:STDOUT:   %Optional.HasValue.call.loc49: init bool = call %bound_method.loc49_19.5(%.loc49_19.8)
// CHECK:STDOUT:   %.loc49_19.9: bool = value_of_initializer %Optional.HasValue.call.loc49
// CHECK:STDOUT:   %.loc49_19.10: bool = converted %Optional.HasValue.call.loc49, %.loc49_19.9
// CHECK:STDOUT:   if %.loc49_19.10 br !for.body.loc49 else br !for.done.loc49
// CHECK:STDOUT:
// CHECK:STDOUT: !for.body.loc49:
// CHECK:STDOUT:   %.loc49_19.11: %Optional.Get.type.6f2 = specific_constant imports.%Core.import_ref.dab, @Optional(constants.%OptionalStorage.facet.53e) [concrete = constants.%Optional.Get.1f4]
// CHECK:STDOUT:   %Get.ref.loc49: %Optional.Get.type.6f2 = name_ref Get, %.loc49_19.11 [concrete = constants.%Optional.Get.1f4]
// CHECK:STDOUT:   %Optional.Get.bound.loc49: <bound method> = bound_method %.loc49_19.6, %Get.ref.loc49
// CHECK:STDOUT:   %Optional.Get.specific_fn.loc49: <specific function> = specific_function %Get.ref.loc49, @Optional.Get(constants.%OptionalStorage.facet.53e) [concrete = constants.%Optional.Get.specific_fn.9da]
// CHECK:STDOUT:   %bound_method.loc49_19.6: <bound method> = bound_method %.loc49_19.6, %Optional.Get.specific_fn.loc49
// CHECK:STDOUT:   %.loc49_19.12: %Optional.f49 = acquire_value %.loc49_19.6
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
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc49_19.1: <bound method> = bound_method %.loc49_19.6, constants.%Destroy.WithSelf.SelfDestruct.bd2
// CHECK:STDOUT:   %Destroy.facet.loc49_19.1: %Destroy.type = facet_value constants.%Optional.f49, (constants.%custom_witness.5eb) [concrete = constants.%Destroy.facet.608]
// CHECK:STDOUT:   %.loc49_19.15: %Destroy.type = converted constants.%Optional.f49, %Destroy.facet.loc49_19.1 [concrete = constants.%Destroy.facet.608]
// CHECK:STDOUT:   %Destroy.facet.loc49_19.2: %Destroy.type = facet_value constants.%Optional.f49, (constants.%custom_witness.5eb) [concrete = constants.%Destroy.facet.608]
// CHECK:STDOUT:   %.loc49_19.16: %Destroy.type = converted constants.%Optional.f49, %Destroy.facet.loc49_19.2 [concrete = constants.%Destroy.facet.608]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc49_19.7: <bound method> = bound_method %.loc49_19.6, %Destroy.WithSelf.SelfDestruct.specific_fn.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc49_19.1: init %empty_tuple.type = call %bound_method.loc49_19.7(%.loc49_19.6)
// CHECK:STDOUT:   br !for.next.loc49
// CHECK:STDOUT:
// CHECK:STDOUT: !for.done.loc49:
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc49_19.2: <bound method> = bound_method %.loc49_19.6, constants.%Destroy.WithSelf.SelfDestruct.bd2
// CHECK:STDOUT:   %Destroy.facet.loc49_19.3: %Destroy.type = facet_value constants.%Optional.f49, (constants.%custom_witness.5eb) [concrete = constants.%Destroy.facet.608]
// CHECK:STDOUT:   %.loc49_19.17: %Destroy.type = converted constants.%Optional.f49, %Destroy.facet.loc49_19.3 [concrete = constants.%Destroy.facet.608]
// CHECK:STDOUT:   %Destroy.facet.loc49_19.4: %Destroy.type = facet_value constants.%Optional.f49, (constants.%custom_witness.5eb) [concrete = constants.%Destroy.facet.608]
// CHECK:STDOUT:   %.loc49_19.18: %Destroy.type = converted constants.%Optional.f49, %Destroy.facet.loc49_19.4 [concrete = constants.%Destroy.facet.608]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc49_19.8: <bound method> = bound_method %.loc49_19.6, %Destroy.WithSelf.SelfDestruct.specific_fn.2
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc49_19.2: init %empty_tuple.type = call %bound_method.loc49_19.8(%.loc49_19.6)
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc49_19.3: <bound method> = bound_method %var.loc49, constants.%Destroy.WithSelf.SelfDestruct.36e
// CHECK:STDOUT:   %Destroy.facet.loc49_19.5: %Destroy.type = facet_value constants.%tuple.type.508, (constants.%custom_witness.c43) [concrete = constants.%Destroy.facet.f27]
// CHECK:STDOUT:   %.loc49_19.19: %Destroy.type = converted constants.%tuple.type.508, %Destroy.facet.loc49_19.5 [concrete = constants.%Destroy.facet.f27]
// CHECK:STDOUT:   %Destroy.facet.loc49_19.6: %Destroy.type = facet_value constants.%tuple.type.508, (constants.%custom_witness.c43) [concrete = constants.%Destroy.facet.f27]
// CHECK:STDOUT:   %.loc49_19.20: %Destroy.type = converted constants.%tuple.type.508, %Destroy.facet.loc49_19.6 [concrete = constants.%Destroy.facet.f27]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc49_19.9: <bound method> = bound_method %var.loc49, %Destroy.WithSelf.SelfDestruct.specific_fn.3
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc49_19.3: init %empty_tuple.type = call %bound_method.loc49_19.9(%var.loc49)
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %i.patt.loc55: %pattern_type.6b6 = value_binding_pattern i [concrete = constants.%i.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c.ref: %ConstRange = name_ref c, %c
// CHECK:STDOUT:   %impl.elem2.loc55: %.81c = impl_witness_access constants.%Iterate.impl_witness.eb6, element2 [concrete = constants.%T.as_type.as.Iterate.impl.NewCursor.930]
// CHECK:STDOUT:   %bound_method.loc55_19.1: <bound method> = bound_method %c.ref, %impl.elem2.loc55
// CHECK:STDOUT:   %facet_value.loc55_19.1: %CppRange_where.type.d682d6.1 = facet_value constants.%ConstRange, (constants.%custom_witness.1ca4d6.2, constants.%Copy.impl_witness.b51, constants.%custom_witness.23e, constants.%custom_witness.952, constants.%custom_witness.adb, constants.%custom_witness.746, constants.%custom_witness.941) [concrete = constants.%facet_value.2c1]
// CHECK:STDOUT:   %.loc55_19.1: %CppRange_where.type.d682d6.1 = converted constants.%ConstRange, %facet_value.loc55_19.1 [concrete = constants.%facet_value.2c1]
// CHECK:STDOUT:   %facet_value.loc55_19.2: %CppRange_where.type.d682d6.1 = facet_value constants.%ConstRange, (constants.%custom_witness.1ca4d6.2, constants.%Copy.impl_witness.b51, constants.%custom_witness.23e, constants.%custom_witness.952, constants.%custom_witness.adb, constants.%custom_witness.746, constants.%custom_witness.941) [concrete = constants.%facet_value.2c1]
// CHECK:STDOUT:   %.loc55_19.2: %CppRange_where.type.d682d6.1 = converted constants.%ConstRange, %facet_value.loc55_19.2 [concrete = constants.%facet_value.2c1]
// CHECK:STDOUT:   %specific_fn.loc55_19.1: <specific function> = specific_function %impl.elem2.loc55, @T.as_type.as.Iterate.impl.NewCursor(constants.%facet_value.2c1) [concrete = constants.%T.as_type.as.Iterate.impl.NewCursor.specific_fn.617]
// CHECK:STDOUT:   %bound_method.loc55_19.2: <bound method> = bound_method %c.ref, %specific_fn.loc55_19.1
// CHECK:STDOUT:   %var.loc55: ref %tuple.type.69f = var_storage invalid
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.call.loc55: init %tuple.type.69f to %var.loc55 = call %bound_method.loc55_19.2(%c.ref)
// CHECK:STDOUT:   assign %var.loc55, %T.as_type.as.Iterate.impl.NewCursor.call.loc55
// CHECK:STDOUT:   br !for.next.loc55
// CHECK:STDOUT:
// CHECK:STDOUT: !for.next.loc55:
// CHECK:STDOUT:   %addr.loc55: %ptr.c5c = addr_of %var.loc55
// CHECK:STDOUT:   %impl.elem3.loc55: %.4e2 = impl_witness_access constants.%Iterate.impl_witness.eb6, element3 [concrete = constants.%T.as_type.as.Iterate.impl.Next.305]
// CHECK:STDOUT:   %bound_method.loc55_19.3: <bound method> = bound_method %c.ref, %impl.elem3.loc55
// CHECK:STDOUT:   %facet_value.loc55_19.3: %CppRange_where.type.d682d6.1 = facet_value constants.%ConstRange, (constants.%custom_witness.1ca4d6.2, constants.%Copy.impl_witness.b51, constants.%custom_witness.23e, constants.%custom_witness.952, constants.%custom_witness.adb, constants.%custom_witness.746, constants.%custom_witness.941) [concrete = constants.%facet_value.2c1]
// CHECK:STDOUT:   %.loc55_19.3: %CppRange_where.type.d682d6.1 = converted constants.%ConstRange, %facet_value.loc55_19.3 [concrete = constants.%facet_value.2c1]
// CHECK:STDOUT:   %facet_value.loc55_19.4: %CppRange_where.type.d682d6.1 = facet_value constants.%ConstRange, (constants.%custom_witness.1ca4d6.2, constants.%Copy.impl_witness.b51, constants.%custom_witness.23e, constants.%custom_witness.952, constants.%custom_witness.adb, constants.%custom_witness.746, constants.%custom_witness.941) [concrete = constants.%facet_value.2c1]
// CHECK:STDOUT:   %.loc55_19.4: %CppRange_where.type.d682d6.1 = converted constants.%ConstRange, %facet_value.loc55_19.4 [concrete = constants.%facet_value.2c1]
// CHECK:STDOUT:   %specific_fn.loc55_19.2: <specific function> = specific_function %impl.elem3.loc55, @T.as_type.as.Iterate.impl.Next(constants.%facet_value.2c1) [concrete = constants.%T.as_type.as.Iterate.impl.Next.specific_fn.52b]
// CHECK:STDOUT:   %bound_method.loc55_19.4: <bound method> = bound_method %c.ref, %specific_fn.loc55_19.2
// CHECK:STDOUT:   %.loc55_19.5: ref %Optional.f49 = temporary_storage
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.call.loc55: init %Optional.f49 to %.loc55_19.5 = call %bound_method.loc55_19.4(%c.ref, %addr.loc55)
// CHECK:STDOUT:   %.loc55_19.6: ref %Optional.f49 = temporary %.loc55_19.5, %T.as_type.as.Iterate.impl.Next.call.loc55
// CHECK:STDOUT:   %.loc55_19.7: %Optional.HasValue.type.91a = specific_constant imports.%Core.import_ref.bb6, @Optional(constants.%OptionalStorage.facet.53e) [concrete = constants.%Optional.HasValue.fe5]
// CHECK:STDOUT:   %HasValue.ref.loc55: %Optional.HasValue.type.91a = name_ref HasValue, %.loc55_19.7 [concrete = constants.%Optional.HasValue.fe5]
// CHECK:STDOUT:   %Optional.HasValue.bound.loc55: <bound method> = bound_method %.loc55_19.6, %HasValue.ref.loc55
// CHECK:STDOUT:   %Optional.HasValue.specific_fn.loc55: <specific function> = specific_function %HasValue.ref.loc55, @Optional.HasValue(constants.%OptionalStorage.facet.53e) [concrete = constants.%Optional.HasValue.specific_fn.fb4]
// CHECK:STDOUT:   %bound_method.loc55_19.5: <bound method> = bound_method %.loc55_19.6, %Optional.HasValue.specific_fn.loc55
// CHECK:STDOUT:   %.loc55_19.8: %Optional.f49 = acquire_value %.loc55_19.6
// CHECK:STDOUT:   %Optional.HasValue.call.loc55: init bool = call %bound_method.loc55_19.5(%.loc55_19.8)
// CHECK:STDOUT:   %.loc55_19.9: bool = value_of_initializer %Optional.HasValue.call.loc55
// CHECK:STDOUT:   %.loc55_19.10: bool = converted %Optional.HasValue.call.loc55, %.loc55_19.9
// CHECK:STDOUT:   if %.loc55_19.10 br !for.body.loc55 else br !for.done.loc55
// CHECK:STDOUT:
// CHECK:STDOUT: !for.body.loc55:
// CHECK:STDOUT:   %.loc55_19.11: %Optional.Get.type.6f2 = specific_constant imports.%Core.import_ref.dab, @Optional(constants.%OptionalStorage.facet.53e) [concrete = constants.%Optional.Get.1f4]
// CHECK:STDOUT:   %Get.ref.loc55: %Optional.Get.type.6f2 = name_ref Get, %.loc55_19.11 [concrete = constants.%Optional.Get.1f4]
// CHECK:STDOUT:   %Optional.Get.bound.loc55: <bound method> = bound_method %.loc55_19.6, %Get.ref.loc55
// CHECK:STDOUT:   %Optional.Get.specific_fn.loc55: <specific function> = specific_function %Get.ref.loc55, @Optional.Get(constants.%OptionalStorage.facet.53e) [concrete = constants.%Optional.Get.specific_fn.9da]
// CHECK:STDOUT:   %bound_method.loc55_19.6: <bound method> = bound_method %.loc55_19.6, %Optional.Get.specific_fn.loc55
// CHECK:STDOUT:   %.loc55_19.12: %Optional.f49 = acquire_value %.loc55_19.6
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
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc55_19.1: <bound method> = bound_method %.loc55_19.6, constants.%Destroy.WithSelf.SelfDestruct.bd2
// CHECK:STDOUT:   %Destroy.facet.loc55_19.1: %Destroy.type = facet_value constants.%Optional.f49, (constants.%custom_witness.5eb) [concrete = constants.%Destroy.facet.608]
// CHECK:STDOUT:   %.loc55_19.15: %Destroy.type = converted constants.%Optional.f49, %Destroy.facet.loc55_19.1 [concrete = constants.%Destroy.facet.608]
// CHECK:STDOUT:   %Destroy.facet.loc55_19.2: %Destroy.type = facet_value constants.%Optional.f49, (constants.%custom_witness.5eb) [concrete = constants.%Destroy.facet.608]
// CHECK:STDOUT:   %.loc55_19.16: %Destroy.type = converted constants.%Optional.f49, %Destroy.facet.loc55_19.2 [concrete = constants.%Destroy.facet.608]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc55_19.7: <bound method> = bound_method %.loc55_19.6, %Destroy.WithSelf.SelfDestruct.specific_fn.4
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc55_19.1: init %empty_tuple.type = call %bound_method.loc55_19.7(%.loc55_19.6)
// CHECK:STDOUT:   br !for.next.loc55
// CHECK:STDOUT:
// CHECK:STDOUT: !for.done.loc55:
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc55_19.2: <bound method> = bound_method %.loc55_19.6, constants.%Destroy.WithSelf.SelfDestruct.bd2
// CHECK:STDOUT:   %Destroy.facet.loc55_19.3: %Destroy.type = facet_value constants.%Optional.f49, (constants.%custom_witness.5eb) [concrete = constants.%Destroy.facet.608]
// CHECK:STDOUT:   %.loc55_19.17: %Destroy.type = converted constants.%Optional.f49, %Destroy.facet.loc55_19.3 [concrete = constants.%Destroy.facet.608]
// CHECK:STDOUT:   %Destroy.facet.loc55_19.4: %Destroy.type = facet_value constants.%Optional.f49, (constants.%custom_witness.5eb) [concrete = constants.%Destroy.facet.608]
// CHECK:STDOUT:   %.loc55_19.18: %Destroy.type = converted constants.%Optional.f49, %Destroy.facet.loc55_19.4 [concrete = constants.%Destroy.facet.608]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc55_19.8: <bound method> = bound_method %.loc55_19.6, %Destroy.WithSelf.SelfDestruct.specific_fn.5
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc55_19.2: init %empty_tuple.type = call %bound_method.loc55_19.8(%.loc55_19.6)
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc55_19.3: <bound method> = bound_method %var.loc55, constants.%Destroy.WithSelf.SelfDestruct.05a
// CHECK:STDOUT:   %Destroy.facet.loc55_19.5: %Destroy.type = facet_value constants.%tuple.type.69f, (constants.%custom_witness.621) [concrete = constants.%Destroy.facet.c06]
// CHECK:STDOUT:   %.loc55_19.19: %Destroy.type = converted constants.%tuple.type.69f, %Destroy.facet.loc55_19.5 [concrete = constants.%Destroy.facet.c06]
// CHECK:STDOUT:   %Destroy.facet.loc55_19.6: %Destroy.type = facet_value constants.%tuple.type.69f, (constants.%custom_witness.621) [concrete = constants.%Destroy.facet.c06]
// CHECK:STDOUT:   %.loc55_19.20: %Destroy.type = converted constants.%tuple.type.69f, %Destroy.facet.loc55_19.6 [concrete = constants.%Destroy.facet.c06]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc55_19.9: <bound method> = bound_method %var.loc55, %Destroy.WithSelf.SelfDestruct.specific_fn.6
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc55_19.3: init %empty_tuple.type = call %bound_method.loc55_19.9(%var.loc55)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestIterate(constants.%R) {
// CHECK:STDOUT:   %R.patt.loc30_17.2 => constants.%R.patt
// CHECK:STDOUT:   %R.loc30_17.1 => constants.%R
// CHECK:STDOUT:   %R.as_type.loc30_61.1 => constants.%R.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.6a1
// CHECK:STDOUT:   %r.param_patt.loc30_59.2 => constants.%r.param_patt.3b6
// CHECK:STDOUT:   %r.patt.loc30_59.2 => constants.%r.patt.6d2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestIterate(constants.%facet_value.8bb) {
// CHECK:STDOUT:   %R.patt.loc30_17.2 => constants.%R.patt
// CHECK:STDOUT:   %R.loc30_17.1 => constants.%facet_value.8bb
// CHECK:STDOUT:   %R.as_type.loc30_61.1 => constants.%MutableRange
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.e635
// CHECK:STDOUT:   %r.param_patt.loc30_59.2 => constants.%r.param_patt.c79
// CHECK:STDOUT:   %r.patt.loc30_59.2 => constants.%r.patt.94f
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc30 => constants.%complete_type.357
// CHECK:STDOUT:   %Iterate.lookup_impl_witness => constants.%Iterate.impl_witness.647
// CHECK:STDOUT:   %Iterate.facet.loc34_19.5 => constants.%Iterate.facet.0d0
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type => constants.%Iterate.WithSelf.NewCursor.type.969
// CHECK:STDOUT:   %.loc34_19.23 => constants.%.ccb
// CHECK:STDOUT:   %impl.elem2.loc34_19.3 => constants.%T.as_type.as.Iterate.impl.NewCursor.45c
// CHECK:STDOUT:   %specific_impl_fn.loc34_19.4 => constants.%T.as_type.as.Iterate.impl.NewCursor.specific_fn.440
// CHECK:STDOUT:   %impl.elem1 => constants.%Destroy.facet.f27
// CHECK:STDOUT:   %as_type => constants.%tuple.type.508
// CHECK:STDOUT:   %require_complete.1 => constants.%complete_type.c28
// CHECK:STDOUT:   %ptr => constants.%ptr.45c
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type => constants.%Iterate.WithSelf.Next.type.1bb
// CHECK:STDOUT:   %.loc34_19.24 => constants.%.343
// CHECK:STDOUT:   %impl.elem3.loc34_19.2 => constants.%T.as_type.as.Iterate.impl.Next.23e
// CHECK:STDOUT:   %specific_impl_fn.loc34_19.5 => constants.%T.as_type.as.Iterate.impl.Next.specific_fn.e33
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type => constants.%Destroy.WithSelf.SelfDestruct.type.c02
// CHECK:STDOUT:   %.loc34_19.25 => constants.%.d19
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.c43
// CHECK:STDOUT:   %impl.elem2.loc34_19.4 => constants.%Destroy.WithSelf.SelfDestruct.36e
// CHECK:STDOUT:   %specific_impl_fn.loc34_19.6 => constants.%Destroy.WithSelf.SelfDestruct.specific_fn.079
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestIterate(constants.%facet_value.d9f) {
// CHECK:STDOUT:   %R.patt.loc30_17.2 => constants.%R.patt
// CHECK:STDOUT:   %R.loc30_17.1 => constants.%facet_value.d9f
// CHECK:STDOUT:   %R.as_type.loc30_61.1 => constants.%ConstRange
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.0ba
// CHECK:STDOUT:   %r.param_patt.loc30_59.2 => constants.%r.param_patt.83e
// CHECK:STDOUT:   %r.patt.loc30_59.2 => constants.%r.patt.c21
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc30 => constants.%complete_type.357
// CHECK:STDOUT:   %Iterate.lookup_impl_witness => constants.%Iterate.impl_witness.eb6
// CHECK:STDOUT:   %Iterate.facet.loc34_19.5 => constants.%Iterate.facet.7f1
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type => constants.%Iterate.WithSelf.NewCursor.type.165
// CHECK:STDOUT:   %.loc34_19.23 => constants.%.81c
// CHECK:STDOUT:   %impl.elem2.loc34_19.3 => constants.%T.as_type.as.Iterate.impl.NewCursor.930
// CHECK:STDOUT:   %specific_impl_fn.loc34_19.4 => constants.%T.as_type.as.Iterate.impl.NewCursor.specific_fn.617
// CHECK:STDOUT:   %impl.elem1 => constants.%Destroy.facet.c06
// CHECK:STDOUT:   %as_type => constants.%tuple.type.69f
// CHECK:STDOUT:   %require_complete.1 => constants.%complete_type.8d5
// CHECK:STDOUT:   %ptr => constants.%ptr.c5c
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type => constants.%Iterate.WithSelf.Next.type.d6c
// CHECK:STDOUT:   %.loc34_19.24 => constants.%.4e2
// CHECK:STDOUT:   %impl.elem3.loc34_19.2 => constants.%T.as_type.as.Iterate.impl.Next.305
// CHECK:STDOUT:   %specific_impl_fn.loc34_19.5 => constants.%T.as_type.as.Iterate.impl.Next.specific_fn.52b
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type => constants.%Destroy.WithSelf.SelfDestruct.type.d17
// CHECK:STDOUT:   %.loc34_19.25 => constants.%.2c0
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.621
// CHECK:STDOUT:   %impl.elem2.loc34_19.4 => constants.%Destroy.WithSelf.SelfDestruct.05a
// CHECK:STDOUT:   %specific_impl_fn.loc34_19.6 => constants.%Destroy.WithSelf.SelfDestruct.specific_fn.199
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- methods_return_different_types.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
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
// CHECK:STDOUT:   %DiffValueType.type.facet: %type = facet_value %DiffValueType, () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.ac8: type = fn_type @Destroy.WithSelf.SubobjectDestroy, @Destroy.WithSelf(%DiffValueType.type.facet) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.db9: %Destroy.WithSelf.SubobjectDestroy.type.ac8 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.2f4: type = fn_type @Destroy.WithSelf.SelfDestruct, @Destroy.WithSelf(%DiffValueType.type.facet) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.d38: %Destroy.WithSelf.SelfDestruct.type.2f4 = struct_value () [concrete]
// CHECK:STDOUT:   %DiffValueType.Op.type.83af6e.1: type = fn_type @DiffValueType.Op.1 [concrete]
// CHECK:STDOUT:   %DiffValueType.Op.4dca31.1: %DiffValueType.Op.type.83af6e.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.c8d: <witness> = custom_witness (%DiffValueType.Op.4dca31.1, %Destroy.WithSelf.SubobjectDestroy.db9, %Destroy.WithSelf.SelfDestruct.d38), @Destroy [concrete]
// CHECK:STDOUT:   %const.879: type = const_type %DiffValueType [concrete]
// CHECK:STDOUT:   %ptr.816: type = ptr_type %const.879 [concrete]
// CHECK:STDOUT:   %ptr.cd0: type = ptr_type %DiffValueType [concrete]
// CHECK:STDOUT:   %DiffValueType.Op.type.83af6e.2: type = fn_type @DiffValueType.Op.2 [concrete]
// CHECK:STDOUT:   %DiffValueType.Op.4dca31.2: %DiffValueType.Op.type.83af6e.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.d0da43.1: <witness> = custom_witness (%DiffValueType.Op.4dca31.2), @Copy [concrete]
// CHECK:STDOUT:   %facet_value.07c: %facet_type.f48 = facet_value %DiffValueType, (%custom_witness.c8d, %custom_witness.d0da43.1) [concrete]
// CHECK:STDOUT:   %Iterate.lookup_impl_witness.1d8: <witness> = lookup_impl_witness %.Self.500, @Iterate [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.294: %facet_type.f48 = impl_witness_access %Iterate.lookup_impl_witness.1d8, element0 [symbolic_self]
// CHECK:STDOUT:   %Iterate_where.type.cb1: type = facet_type <@Iterate where %impl.elem0.294 = %facet_value.07c> [concrete]
// CHECK:STDOUT:   %pattern_type.f25: type = pattern_type %Iterate_where.type.cb1 [concrete]
// CHECK:STDOUT:   %R.patt: %pattern_type.f25 = symbolic_binding_pattern R, 0 [symbolic]
// CHECK:STDOUT:   %R: %Iterate_where.type.cb1 = symbolic_binding R, 0 [symbolic]
// CHECK:STDOUT:   %R.as_type: type = facet_access_type %R [symbolic]
// CHECK:STDOUT:   %pattern_type.345: type = pattern_type %R.as_type [symbolic]
// CHECK:STDOUT:   %r.patt.48b: %pattern_type.345 = ref_binding_pattern r [symbolic]
// CHECK:STDOUT:   %r.param_patt.217: %pattern_type.345 = var_param_pattern %r.patt.48b [symbolic]
// CHECK:STDOUT:   %i.patt: %pattern_type.149 = value_binding_pattern i [concrete]
// CHECK:STDOUT:   %Iterate.lookup_impl_witness.1fb: <witness> = lookup_impl_witness %R, @Iterate [symbolic]
// CHECK:STDOUT:   %Iterate.facet.0eb: %Iterate.type = facet_value %R.as_type, (%Iterate.lookup_impl_witness.1fb) [symbolic]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.8e7: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.0eb) [symbolic]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.5dd: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.0eb) [symbolic]
// CHECK:STDOUT:   %.3d5: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.8e7, %Iterate.facet.0eb [symbolic]
// CHECK:STDOUT:   %impl.elem2.27e: %.3d5 = impl_witness_access %Iterate.lookup_impl_witness.1fb, element2 [symbolic]
// CHECK:STDOUT:   %impl.elem1.132: %Destroy.type = impl_witness_access %Iterate.lookup_impl_witness.1fb, element1 [symbolic]
// CHECK:STDOUT:   %as_type.e58: type = facet_access_type %impl.elem1.132 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.645: <specific function> = specific_impl_function %impl.elem2.27e, @Iterate.WithSelf.NewCursor(%Iterate.facet.0eb) [symbolic]
// CHECK:STDOUT:   %ptr.704: type = ptr_type %as_type.e58 [symbolic]
// CHECK:STDOUT:   %.c42: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.5dd, %Iterate.facet.0eb [symbolic]
// CHECK:STDOUT:   %impl.elem3.9bb: %.c42 = impl_witness_access %Iterate.lookup_impl_witness.1fb, element3 [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.ee5: type = class_type @DefaultOptionalStorage, @DefaultOptionalStorage(%facet_value.07c) [concrete]
// CHECK:STDOUT:   %OptionalStorage.impl_witness.8cf: <witness> = impl_witness imports.%OptionalStorage.impl_witness_table.460, @T.as_type.as.OptionalStorage.impl(%facet_value.07c) [concrete]
// CHECK:STDOUT:   %OptionalStorage.facet.d9a: %OptionalStorage.type = facet_value %DiffValueType, (%OptionalStorage.impl_witness.8cf) [concrete]
// CHECK:STDOUT:   %Optional.6a3: type = class_type @Optional, @Optional(%OptionalStorage.facet.d9a) [concrete]
// CHECK:STDOUT:   %specific_impl_fn.7a4: <specific function> = specific_impl_function %impl.elem3.9bb, @Iterate.WithSelf.Next(%Iterate.facet.0eb) [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.type.11b: type = fn_type @Optional.HasValue, @Optional(%OptionalStorage.facet.d9a) [concrete]
// CHECK:STDOUT:   %Optional.HasValue.453: %Optional.HasValue.type.11b = struct_value () [concrete]
// CHECK:STDOUT:   %Optional.Get.type.d6e: type = fn_type @Optional.Get, @Optional(%OptionalStorage.facet.d9a) [concrete]
// CHECK:STDOUT:   %Optional.Get.902: %Optional.Get.type.d6e = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.facet.f3c: %Destroy.type = facet_value %DiffValueType, (%custom_witness.c8d) [concrete]
// CHECK:STDOUT:   %MaybeUnformed.84b: type = class_type @MaybeUnformed, @MaybeUnformed(%Destroy.facet.f3c) [concrete]
// CHECK:STDOUT:   %.a18: type = maybe_unformed_type %DiffValueType [concrete]
// CHECK:STDOUT:   %struct_type.value.has_value.bd3: type = struct_type {.value: %MaybeUnformed.84b, .has_value: bool} [concrete]
// CHECK:STDOUT:   %Optional.HasValue.specific_fn.b77: <specific function> = specific_function %Optional.HasValue.453, @Optional.HasValue(%OptionalStorage.facet.d9a) [concrete]
// CHECK:STDOUT:   %Optional.Get.specific_fn.f40: <specific function> = specific_function %Optional.Get.902, @Optional.Get(%OptionalStorage.facet.d9a) [concrete]
// CHECK:STDOUT:   %operator_PlusEqual__carbon_thunk.type: type = fn_type @operator_PlusEqual__carbon_thunk [concrete]
// CHECK:STDOUT:   %operator_PlusEqual__carbon_thunk: %operator_PlusEqual__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.7: type = fn_type @Destroy.Op.loc44_33.6 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.7: %Destroy.Op.type.1d8f74.7 = struct_value () [concrete]
// CHECK:STDOUT:   %Optional.type.facet: %type = facet_value %Optional.6a3, () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.5d9: type = fn_type @Destroy.WithSelf.SubobjectDestroy, @Destroy.WithSelf(%Optional.type.facet) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.1fb: %Destroy.WithSelf.SubobjectDestroy.type.5d9 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.b69: type = fn_type @Destroy.WithSelf.SelfDestruct, @Destroy.WithSelf(%Optional.type.facet) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.4f4: %Destroy.WithSelf.SelfDestruct.type.b69 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.159: <witness> = custom_witness (%Destroy.Op.1a2547.7, %Destroy.WithSelf.SubobjectDestroy.1fb, %Destroy.WithSelf.SelfDestruct.4f4), @Destroy [concrete]
// CHECK:STDOUT:   %Destroy.facet.2bb: %Destroy.type = facet_value %Optional.6a3, (%custom_witness.159) [concrete]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.2cf: <witness> = lookup_impl_witness %impl.elem1.132, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.871: type = fn_type @Destroy.WithSelf.SelfDestruct, @Destroy.WithSelf(%impl.elem1.132) [symbolic]
// CHECK:STDOUT:   %.da7: type = fn_type_with_self_type %Destroy.WithSelf.SelfDestruct.type.871, %impl.elem1.132 [symbolic]
// CHECK:STDOUT:   %impl.elem2.fe5: %.da7 = impl_witness_access %Destroy.lookup_impl_witness.2cf, element2 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.d19: <specific function> = specific_impl_function %impl.elem2.fe5, @Destroy.WithSelf.SelfDestruct(%impl.elem1.132) [symbolic]
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
// CHECK:STDOUT:   %Iterator.type.facet.600: %type = facet_value %Iterator.f25, () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.b83: type = fn_type @Destroy.WithSelf.SubobjectDestroy, @Destroy.WithSelf(%Iterator.type.facet.600) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.36b: %Destroy.WithSelf.SubobjectDestroy.type.b83 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.346: type = fn_type @Destroy.WithSelf.SelfDestruct, @Destroy.WithSelf(%Iterator.type.facet.600) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.44a: %Destroy.WithSelf.SelfDestruct.type.346 = struct_value () [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.41bbea.2: type = fn_type @Iterator.Op.2 [concrete]
// CHECK:STDOUT:   %Iterator.Op.1caa36.2: %Iterator.Op.type.41bbea.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.9b3: <witness> = custom_witness (%Iterator.Op.1caa36.2, %Destroy.WithSelf.SubobjectDestroy.36b, %Destroy.WithSelf.SelfDestruct.44a), @Destroy [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.41bbea.3: type = fn_type @Iterator.Op.3 [concrete]
// CHECK:STDOUT:   %Iterator.Op.1caa36.3: %Iterator.Op.type.41bbea.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.1c9: <witness> = custom_witness (%Iterator.Op.1caa36.3), @Inc [concrete]
// CHECK:STDOUT:   %Sentinel.type.facet.95a: %type = facet_value %Sentinel.b4c, () [concrete]
// CHECK:STDOUT:   %Equal.type.ca8475.1: type = fn_type @Equal.1 [concrete]
// CHECK:STDOUT:   %Equal.285b15.1: %Equal.type.ca8475.1 = struct_value () [concrete]
// CHECK:STDOUT:   %NotEqual.type.53d9ee.1: type = fn_type @NotEqual.1 [concrete]
// CHECK:STDOUT:   %NotEqual.119263.1: %NotEqual.type.53d9ee.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.723: <witness> = custom_witness (%Equal.285b15.1, %NotEqual.119263.1), @EqWith, @EqWith(%Sentinel.b4c) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.c3c: type = fn_type @Destroy.WithSelf.SubobjectDestroy, @Destroy.WithSelf(%Sentinel.type.facet.95a) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.97f: %Destroy.WithSelf.SubobjectDestroy.type.c3c = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.aa5: type = fn_type @Destroy.WithSelf.SelfDestruct, @Destroy.WithSelf(%Sentinel.type.facet.95a) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.886: %Destroy.WithSelf.SelfDestruct.type.aa5 = struct_value () [concrete]
// CHECK:STDOUT:   %Sentinel.Op.type.41b: type = fn_type @Sentinel.Op.1 [concrete]
// CHECK:STDOUT:   %Sentinel.Op.1ca: %Sentinel.Op.type.41b = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.161: <witness> = custom_witness (%Sentinel.Op.1ca, %Destroy.WithSelf.SubobjectDestroy.97f, %Destroy.WithSelf.SelfDestruct.886), @Destroy [concrete]
// CHECK:STDOUT:   %facet_value.179: %CppRange_where.type.d682d6.1 = facet_value %DiffMutableRange, (%custom_witness.c8d, %custom_witness.d0da43.1, %custom_witness.aa8, %custom_witness.9b3, %custom_witness.392, %custom_witness.1c9, %custom_witness.723, %custom_witness.161) [concrete]
// CHECK:STDOUT:   %tuple.type.b18: type = tuple_type (%Iterator.f25, %Sentinel.b4c) [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.8: type = fn_type @Destroy.Op.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.8: %Destroy.Op.type.1d8f74.8 = struct_value () [concrete]
// CHECK:STDOUT:   %tuple.type.facet.238: %type = facet_value %tuple.type.b18, () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.76e: type = fn_type @Destroy.WithSelf.SubobjectDestroy, @Destroy.WithSelf(%tuple.type.facet.238) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.ceb: %Destroy.WithSelf.SubobjectDestroy.type.76e = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.f4e: type = fn_type @Destroy.WithSelf.SelfDestruct, @Destroy.WithSelf(%tuple.type.facet.238) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.aca: %Destroy.WithSelf.SelfDestruct.type.f4e = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.3fe: <witness> = custom_witness (%Destroy.Op.1a2547.8, %Destroy.WithSelf.SubobjectDestroy.ceb, %Destroy.WithSelf.SelfDestruct.aca), @Destroy [concrete]
// CHECK:STDOUT:   %Destroy.facet.fb2: %Destroy.type = facet_value %tuple.type.b18, (%custom_witness.3fe) [concrete]
// CHECK:STDOUT:   %Iterate.impl_witness.de0: <witness> = impl_witness imports.%Iterate.impl_witness_table.121, @T.as_type.as.Iterate.impl(%facet_value.179) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.type.994: type = fn_type @T.as_type.as.Iterate.impl.NewCursor, @T.as_type.as.Iterate.impl(%facet_value.179) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.d94: %T.as_type.as.Iterate.impl.NewCursor.type.994 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.type.e8b: type = fn_type @T.as_type.as.Iterate.impl.Next, @T.as_type.as.Iterate.impl(%facet_value.179) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.10b: %T.as_type.as.Iterate.impl.Next.type.e8b = struct_value () [concrete]
// CHECK:STDOUT:   %Iterate.facet.b21: %Iterate.type = facet_value %DiffMutableRange, (%Iterate.impl_witness.de0) [concrete]
// CHECK:STDOUT:   %facet_value.b23: %Iterate_where.type.cb1 = facet_value %DiffMutableRange, (%Iterate.impl_witness.de0) [concrete]
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
// CHECK:STDOUT:   %Iterator.type.facet.e85: %type = facet_value %Iterator.a59, () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.964: type = fn_type @Destroy.WithSelf.SubobjectDestroy, @Destroy.WithSelf(%Iterator.type.facet.e85) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.952: %Destroy.WithSelf.SubobjectDestroy.type.964 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.f1c: type = fn_type @Destroy.WithSelf.SelfDestruct, @Destroy.WithSelf(%Iterator.type.facet.e85) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.8c7a: %Destroy.WithSelf.SelfDestruct.type.f1c = struct_value () [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.b7ae80.2: type = fn_type @Iterator.Op.5 [concrete]
// CHECK:STDOUT:   %Iterator.Op.690fee.2: %Iterator.Op.type.b7ae80.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.a60: <witness> = custom_witness (%Iterator.Op.690fee.2, %Destroy.WithSelf.SubobjectDestroy.952, %Destroy.WithSelf.SelfDestruct.8c7a), @Destroy [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.b7ae80.3: type = fn_type @Iterator.Op.6 [concrete]
// CHECK:STDOUT:   %Iterator.Op.690fee.3: %Iterator.Op.type.b7ae80.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.ffb: <witness> = custom_witness (%Iterator.Op.690fee.3), @Inc [concrete]
// CHECK:STDOUT:   %Sentinel.type.facet.b0b: %type = facet_value %Sentinel.b33, () [concrete]
// CHECK:STDOUT:   %Equal.type.ca8475.2: type = fn_type @Equal.2 [concrete]
// CHECK:STDOUT:   %Equal.285b15.2: %Equal.type.ca8475.2 = struct_value () [concrete]
// CHECK:STDOUT:   %NotEqual.type.53d9ee.2: type = fn_type @NotEqual.2 [concrete]
// CHECK:STDOUT:   %NotEqual.119263.2: %NotEqual.type.53d9ee.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.a26: <witness> = custom_witness (%Equal.285b15.2, %NotEqual.119263.2), @EqWith, @EqWith(%Sentinel.b33) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.a34: type = fn_type @Destroy.WithSelf.SubobjectDestroy, @Destroy.WithSelf(%Sentinel.type.facet.b0b) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.c40: %Destroy.WithSelf.SubobjectDestroy.type.a34 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.133: type = fn_type @Destroy.WithSelf.SelfDestruct, @Destroy.WithSelf(%Sentinel.type.facet.b0b) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.8dd: %Destroy.WithSelf.SelfDestruct.type.133 = struct_value () [concrete]
// CHECK:STDOUT:   %Sentinel.Op.type.b7a: type = fn_type @Sentinel.Op.2 [concrete]
// CHECK:STDOUT:   %Sentinel.Op.690: %Sentinel.Op.type.b7a = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.dd2: <witness> = custom_witness (%Sentinel.Op.690, %Destroy.WithSelf.SubobjectDestroy.c40, %Destroy.WithSelf.SelfDestruct.8dd), @Destroy [concrete]
// CHECK:STDOUT:   %facet_value.af9: %CppRange_where.type.d682d6.1 = facet_value %DiffConstRange, (%custom_witness.c8d, %custom_witness.d0da43.1, %custom_witness.7b4, %custom_witness.a60, %custom_witness.0c0, %custom_witness.ffb, %custom_witness.a26, %custom_witness.dd2) [concrete]
// CHECK:STDOUT:   %tuple.type.df5: type = tuple_type (%Iterator.a59, %Sentinel.b33) [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.9: type = fn_type @Destroy.Op.3 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.9: %Destroy.Op.type.1d8f74.9 = struct_value () [concrete]
// CHECK:STDOUT:   %tuple.type.facet.5f1: %type = facet_value %tuple.type.df5, () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.a33: type = fn_type @Destroy.WithSelf.SubobjectDestroy, @Destroy.WithSelf(%tuple.type.facet.5f1) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.455: %Destroy.WithSelf.SubobjectDestroy.type.a33 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.463: type = fn_type @Destroy.WithSelf.SelfDestruct, @Destroy.WithSelf(%tuple.type.facet.5f1) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.6d7: %Destroy.WithSelf.SelfDestruct.type.463 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.34c: <witness> = custom_witness (%Destroy.Op.1a2547.9, %Destroy.WithSelf.SubobjectDestroy.455, %Destroy.WithSelf.SelfDestruct.6d7), @Destroy [concrete]
// CHECK:STDOUT:   %Destroy.facet.bca: %Destroy.type = facet_value %tuple.type.df5, (%custom_witness.34c) [concrete]
// CHECK:STDOUT:   %Iterate.impl_witness.165: <witness> = impl_witness imports.%Iterate.impl_witness_table.121, @T.as_type.as.Iterate.impl(%facet_value.af9) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.type.d10: type = fn_type @T.as_type.as.Iterate.impl.NewCursor, @T.as_type.as.Iterate.impl(%facet_value.af9) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.1d0: %T.as_type.as.Iterate.impl.NewCursor.type.d10 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.type.055: type = fn_type @T.as_type.as.Iterate.impl.Next, @T.as_type.as.Iterate.impl(%facet_value.af9) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.dce: %T.as_type.as.Iterate.impl.Next.type.055 = struct_value () [concrete]
// CHECK:STDOUT:   %Iterate.facet.ad6: %Iterate.type = facet_value %DiffConstRange, (%Iterate.impl_witness.165) [concrete]
// CHECK:STDOUT:   %facet_value.4d4: %Iterate_where.type.cb1 = facet_value %DiffConstRange, (%Iterate.impl_witness.165) [concrete]
// CHECK:STDOUT:   %r.patt.9c0: %pattern_type.7dd = ref_binding_pattern r [concrete]
// CHECK:STDOUT:   %r.param_patt.992: %pattern_type.7dd = var_param_pattern %r.patt.9c0 [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.0d5: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.b21) [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.8a9: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.b21) [concrete]
// CHECK:STDOUT:   %.78a: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.0d5, %Iterate.facet.b21 [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.specific_fn.375: <specific function> = specific_function %T.as_type.as.Iterate.impl.NewCursor.d94, @T.as_type.as.Iterate.impl.NewCursor(%facet_value.179) [concrete]
// CHECK:STDOUT:   %ptr.3ce: type = ptr_type %tuple.type.b18 [concrete]
// CHECK:STDOUT:   %.4e1: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.8a9, %Iterate.facet.b21 [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.specific_fn.dcf: <specific function> = specific_function %T.as_type.as.Iterate.impl.Next.10b, @T.as_type.as.Iterate.impl.Next(%facet_value.179) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.e0d: type = fn_type @Destroy.WithSelf.SelfDestruct, @Destroy.WithSelf(%Destroy.facet.fb2) [concrete]
// CHECK:STDOUT:   %.433: type = fn_type_with_self_type %Destroy.WithSelf.SelfDestruct.type.e0d, %Destroy.facet.fb2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.specific_fn.d41: <specific function> = specific_function %Destroy.WithSelf.SelfDestruct.aca, @Destroy.WithSelf.SelfDestruct(%tuple.type.facet.238) [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.5bf: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.ad6) [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.a66: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.ad6) [concrete]
// CHECK:STDOUT:   %.d61: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.5bf, %Iterate.facet.ad6 [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.specific_fn.080: <specific function> = specific_function %T.as_type.as.Iterate.impl.NewCursor.1d0, @T.as_type.as.Iterate.impl.NewCursor(%facet_value.af9) [concrete]
// CHECK:STDOUT:   %ptr.310: type = ptr_type %tuple.type.df5 [concrete]
// CHECK:STDOUT:   %.77a: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.a66, %Iterate.facet.ad6 [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.specific_fn.5bd: <specific function> = specific_function %T.as_type.as.Iterate.impl.Next.dce, @T.as_type.as.Iterate.impl.Next(%facet_value.af9) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.b92: type = fn_type @Destroy.WithSelf.SelfDestruct, @Destroy.WithSelf(%Destroy.facet.bca) [concrete]
// CHECK:STDOUT:   %.5913: type = fn_type_with_self_type %Destroy.WithSelf.SelfDestruct.type.b92, %Destroy.facet.bca [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.specific_fn.33f: <specific function> = specific_function %Destroy.WithSelf.SelfDestruct.6d7, @Destroy.WithSelf.SelfDestruct(%tuple.type.facet.5f1) [concrete]
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
// CHECK:STDOUT:   %operator_PlusEqual__carbon_thunk.decl: %operator_PlusEqual__carbon_thunk.type = fn_decl @operator_PlusEqual__carbon_thunk [concrete = constants.%operator_PlusEqual__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %DiffMutableRange.decl: type = class_decl @DiffMutableRange [concrete = constants.%DiffMutableRange] {} {}
// CHECK:STDOUT:   %DiffConstRange.decl: type = class_decl @DiffConstRange [concrete = constants.%DiffConstRange] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @TestIterate(%R.loc40_17.2: %Iterate_where.type.cb1) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Iterate.lookup_impl_witness: <witness> = lookup_impl_witness %R.loc40_17.1, @Iterate [symbolic = %Iterate.lookup_impl_witness (constants.%Iterate.lookup_impl_witness.1fb)]
// CHECK:STDOUT:   %Iterate.facet.loc44_33.5: %Iterate.type = facet_value %R.as_type.loc40_79.1, (%Iterate.lookup_impl_witness) [symbolic = %Iterate.facet.loc44_33.5 (constants.%Iterate.facet.0eb)]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.loc44_33.5) [symbolic = %Iterate.WithSelf.NewCursor.type (constants.%Iterate.WithSelf.NewCursor.type.8e7)]
// CHECK:STDOUT:   %.loc44_33.26: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type, %Iterate.facet.loc44_33.5 [symbolic = %.loc44_33.26 (constants.%.3d5)]
// CHECK:STDOUT:   %impl.elem2.loc44_33.3: @TestIterate.%.loc44_33.26 (%.3d5) = impl_witness_access %Iterate.lookup_impl_witness, element2 [symbolic = %impl.elem2.loc44_33.3 (constants.%impl.elem2.27e)]
// CHECK:STDOUT:   %specific_impl_fn.loc44_33.4: <specific function> = specific_impl_function %impl.elem2.loc44_33.3, @Iterate.WithSelf.NewCursor(%Iterate.facet.loc44_33.5) [symbolic = %specific_impl_fn.loc44_33.4 (constants.%specific_impl_fn.645)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %ptr: type = ptr_type %as_type [symbolic = %ptr (constants.%ptr.704)]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.loc44_33.5) [symbolic = %Iterate.WithSelf.Next.type (constants.%Iterate.WithSelf.Next.type.5dd)]
// CHECK:STDOUT:   %.loc44_33.27: type = fn_type_with_self_type %Iterate.WithSelf.Next.type, %Iterate.facet.loc44_33.5 [symbolic = %.loc44_33.27 (constants.%.c42)]
// CHECK:STDOUT:   %impl.elem3.loc44_33.2: @TestIterate.%.loc44_33.27 (%.c42) = impl_witness_access %Iterate.lookup_impl_witness, element3 [symbolic = %impl.elem3.loc44_33.2 (constants.%impl.elem3.9bb)]
// CHECK:STDOUT:   %specific_impl_fn.loc44_33.5: <specific function> = specific_impl_function %impl.elem3.loc44_33.2, @Iterate.WithSelf.Next(%Iterate.facet.loc44_33.5) [symbolic = %specific_impl_fn.loc44_33.5 (constants.%specific_impl_fn.7a4)]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type: type = fn_type @Destroy.WithSelf.SelfDestruct, @Destroy.WithSelf(%impl.elem1) [symbolic = %Destroy.WithSelf.SelfDestruct.type (constants.%Destroy.WithSelf.SelfDestruct.type.871)]
// CHECK:STDOUT:   %.loc44_33.28: type = fn_type_with_self_type %Destroy.WithSelf.SelfDestruct.type, %impl.elem1 [symbolic = %.loc44_33.28 (constants.%.da7)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %impl.elem1, @Destroy [symbolic = %Destroy.lookup_impl_witness (constants.%Destroy.lookup_impl_witness.2cf)]
// CHECK:STDOUT:   %impl.elem2.loc44_33.4: @TestIterate.%.loc44_33.28 (%.da7) = impl_witness_access %Destroy.lookup_impl_witness, element2 [symbolic = %impl.elem2.loc44_33.4 (constants.%impl.elem2.fe5)]
// CHECK:STDOUT:   %specific_impl_fn.loc44_33.6: <specific function> = specific_impl_function %impl.elem2.loc44_33.4, @Destroy.WithSelf.SelfDestruct(%impl.elem1) [symbolic = %specific_impl_fn.loc44_33.6 (constants.%specific_impl_fn.d19)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%r.param: ref @TestIterate.%R.as_type.loc40_79.1 (%R.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %i.patt: %pattern_type.149 = value_binding_pattern i [concrete = constants.%i.patt]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %r.ref: ref @TestIterate.%R.as_type.loc40_79.1 (%R.as_type) = name_ref r, %r
// CHECK:STDOUT:     %impl.elem2.loc44_33.1: @TestIterate.%.loc44_33.26 (%.3d5) = impl_witness_access constants.%Iterate.lookup_impl_witness.1fb, element2 [symbolic = %impl.elem2.loc44_33.3 (constants.%impl.elem2.27e)]
// CHECK:STDOUT:     %bound_method.loc44_33.1: <bound method> = bound_method %r.ref, %impl.elem2.loc44_33.1
// CHECK:STDOUT:     %Iterate.facet.loc44_33.1: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.1fb) [symbolic = %Iterate.facet.loc44_33.5 (constants.%Iterate.facet.0eb)]
// CHECK:STDOUT:     %.loc44_33.1: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc44_33.1 [symbolic = %Iterate.facet.loc44_33.5 (constants.%Iterate.facet.0eb)]
// CHECK:STDOUT:     %Iterate.facet.loc44_33.2: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.1fb) [symbolic = %Iterate.facet.loc44_33.5 (constants.%Iterate.facet.0eb)]
// CHECK:STDOUT:     %.loc44_33.2: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc44_33.2 [symbolic = %Iterate.facet.loc44_33.5 (constants.%Iterate.facet.0eb)]
// CHECK:STDOUT:     %specific_impl_fn.loc44_33.1: <specific function> = specific_impl_function %impl.elem2.loc44_33.1, @Iterate.WithSelf.NewCursor(constants.%Iterate.facet.0eb) [symbolic = %specific_impl_fn.loc44_33.4 (constants.%specific_impl_fn.645)]
// CHECK:STDOUT:     %bound_method.loc44_33.2: <bound method> = bound_method %r.ref, %specific_impl_fn.loc44_33.1
// CHECK:STDOUT:     %var: ref @TestIterate.%as_type (%as_type.e58) = var_storage invalid
// CHECK:STDOUT:     %.loc44_32.1: @TestIterate.%R.as_type.loc40_79.1 (%R.as_type) = acquire_value %r.ref
// CHECK:STDOUT:     %Iterate.WithSelf.NewCursor.call: init @TestIterate.%as_type (%as_type.e58) to %var = call %bound_method.loc44_33.2(%.loc44_32.1)
// CHECK:STDOUT:     assign %var, %Iterate.WithSelf.NewCursor.call
// CHECK:STDOUT:     br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.next:
// CHECK:STDOUT:     %addr.loc44: @TestIterate.%ptr (%ptr.704) = addr_of %var
// CHECK:STDOUT:     %impl.elem3.loc44_33.1: @TestIterate.%.loc44_33.27 (%.c42) = impl_witness_access constants.%Iterate.lookup_impl_witness.1fb, element3 [symbolic = %impl.elem3.loc44_33.2 (constants.%impl.elem3.9bb)]
// CHECK:STDOUT:     %bound_method.loc44_33.3: <bound method> = bound_method %r.ref, %impl.elem3.loc44_33.1
// CHECK:STDOUT:     %Iterate.facet.loc44_33.3: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.1fb) [symbolic = %Iterate.facet.loc44_33.5 (constants.%Iterate.facet.0eb)]
// CHECK:STDOUT:     %.loc44_33.3: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc44_33.3 [symbolic = %Iterate.facet.loc44_33.5 (constants.%Iterate.facet.0eb)]
// CHECK:STDOUT:     %Iterate.facet.loc44_33.4: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.1fb) [symbolic = %Iterate.facet.loc44_33.5 (constants.%Iterate.facet.0eb)]
// CHECK:STDOUT:     %.loc44_33.4: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc44_33.4 [symbolic = %Iterate.facet.loc44_33.5 (constants.%Iterate.facet.0eb)]
// CHECK:STDOUT:     %.loc44_33.5: %Destroy.type = converted constants.%as_type.e58, constants.%impl.elem1.132 [symbolic = %impl.elem1 (constants.%impl.elem1.132)]
// CHECK:STDOUT:     %.loc44_33.6: %Destroy.type = converted constants.%as_type.e58, constants.%impl.elem1.132 [symbolic = %impl.elem1 (constants.%impl.elem1.132)]
// CHECK:STDOUT:     %specific_impl_fn.loc44_33.2: <specific function> = specific_impl_function %impl.elem3.loc44_33.1, @Iterate.WithSelf.Next(constants.%Iterate.facet.0eb) [symbolic = %specific_impl_fn.loc44_33.5 (constants.%specific_impl_fn.7a4)]
// CHECK:STDOUT:     %bound_method.loc44_33.4: <bound method> = bound_method %r.ref, %specific_impl_fn.loc44_33.2
// CHECK:STDOUT:     %.loc44_33.7: ref %Optional.6a3 = temporary_storage
// CHECK:STDOUT:     %.loc44_32.2: @TestIterate.%R.as_type.loc40_79.1 (%R.as_type) = acquire_value %r.ref
// CHECK:STDOUT:     %Iterate.WithSelf.Next.call: init %Optional.6a3 to %.loc44_33.7 = call %bound_method.loc44_33.4(%.loc44_32.2, %addr.loc44)
// CHECK:STDOUT:     %.loc44_33.8: ref %Optional.6a3 = temporary %.loc44_33.7, %Iterate.WithSelf.Next.call
// CHECK:STDOUT:     %.loc44_33.9: %Optional.HasValue.type.11b = specific_constant imports.%Core.import_ref.bb6, @Optional(constants.%OptionalStorage.facet.d9a) [concrete = constants.%Optional.HasValue.453]
// CHECK:STDOUT:     %HasValue.ref: %Optional.HasValue.type.11b = name_ref HasValue, %.loc44_33.9 [concrete = constants.%Optional.HasValue.453]
// CHECK:STDOUT:     %Optional.HasValue.bound: <bound method> = bound_method %.loc44_33.8, %HasValue.ref
// CHECK:STDOUT:     %Optional.HasValue.specific_fn: <specific function> = specific_function %HasValue.ref, @Optional.HasValue(constants.%OptionalStorage.facet.d9a) [concrete = constants.%Optional.HasValue.specific_fn.b77]
// CHECK:STDOUT:     %bound_method.loc44_33.5: <bound method> = bound_method %.loc44_33.8, %Optional.HasValue.specific_fn
// CHECK:STDOUT:     %.loc44_33.10: %Optional.6a3 = acquire_value %.loc44_33.8
// CHECK:STDOUT:     %Optional.HasValue.call: init bool = call %bound_method.loc44_33.5(%.loc44_33.10)
// CHECK:STDOUT:     %.loc44_33.11: bool = value_of_initializer %Optional.HasValue.call
// CHECK:STDOUT:     %.loc44_33.12: bool = converted %Optional.HasValue.call, %.loc44_33.11
// CHECK:STDOUT:     if %.loc44_33.12 br !for.body else br !for.done
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.body:
// CHECK:STDOUT:     %.loc44_33.13: %Optional.Get.type.d6e = specific_constant imports.%Core.import_ref.dab, @Optional(constants.%OptionalStorage.facet.d9a) [concrete = constants.%Optional.Get.902]
// CHECK:STDOUT:     %Get.ref: %Optional.Get.type.d6e = name_ref Get, %.loc44_33.13 [concrete = constants.%Optional.Get.902]
// CHECK:STDOUT:     %Optional.Get.bound: <bound method> = bound_method %.loc44_33.8, %Get.ref
// CHECK:STDOUT:     %Optional.Get.specific_fn: <specific function> = specific_function %Get.ref, @Optional.Get(constants.%OptionalStorage.facet.d9a) [concrete = constants.%Optional.Get.specific_fn.f40]
// CHECK:STDOUT:     %bound_method.loc44_33.6: <bound method> = bound_method %.loc44_33.8, %Optional.Get.specific_fn
// CHECK:STDOUT:     %.loc44_33.14: ref %DiffValueType = temporary_storage
// CHECK:STDOUT:     %.loc44_33.15: %Optional.6a3 = acquire_value %.loc44_33.8
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
// CHECK:STDOUT:     %Destroy.WithSelf.SelfDestruct.bound.loc44_33.1: <bound method> = bound_method %.loc44_33.16, constants.%Destroy.WithSelf.SelfDestruct.d38
// CHECK:STDOUT:     %Destroy.facet.loc44_33.1: %Destroy.type = facet_value constants.%DiffValueType, (constants.%custom_witness.c8d) [concrete = constants.%Destroy.facet.f3c]
// CHECK:STDOUT:     %.loc44_33.18: %Destroy.type = converted constants.%DiffValueType, %Destroy.facet.loc44_33.1 [concrete = constants.%Destroy.facet.f3c]
// CHECK:STDOUT:     %Destroy.facet.loc44_33.2: %Destroy.type = facet_value constants.%DiffValueType, (constants.%custom_witness.c8d) [concrete = constants.%Destroy.facet.f3c]
// CHECK:STDOUT:     %.loc44_33.19: %Destroy.type = converted constants.%DiffValueType, %Destroy.facet.loc44_33.2 [concrete = constants.%Destroy.facet.f3c]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %bound_method.loc44_33.7: <bound method> = bound_method %.loc44_33.16, %Destroy.WithSelf.SelfDestruct.specific_fn.1
// CHECK:STDOUT:     %Destroy.WithSelf.SelfDestruct.call.loc44_33.1: init %empty_tuple.type = call %bound_method.loc44_33.7(%.loc44_33.16)
// CHECK:STDOUT:     %Destroy.WithSelf.SelfDestruct.bound.loc44_33.2: <bound method> = bound_method %.loc44_33.8, constants.%Destroy.WithSelf.SelfDestruct.4f4
// CHECK:STDOUT:     %Destroy.facet.loc44_33.3: %Destroy.type = facet_value constants.%Optional.6a3, (constants.%custom_witness.159) [concrete = constants.%Destroy.facet.2bb]
// CHECK:STDOUT:     %.loc44_33.20: %Destroy.type = converted constants.%Optional.6a3, %Destroy.facet.loc44_33.3 [concrete = constants.%Destroy.facet.2bb]
// CHECK:STDOUT:     %Destroy.facet.loc44_33.4: %Destroy.type = facet_value constants.%Optional.6a3, (constants.%custom_witness.159) [concrete = constants.%Destroy.facet.2bb]
// CHECK:STDOUT:     %.loc44_33.21: %Destroy.type = converted constants.%Optional.6a3, %Destroy.facet.loc44_33.4 [concrete = constants.%Destroy.facet.2bb]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %bound_method.loc44_33.8: <bound method> = bound_method %.loc44_33.8, %Destroy.WithSelf.SelfDestruct.specific_fn.2
// CHECK:STDOUT:     %Destroy.WithSelf.SelfDestruct.call.loc44_33.2: init %empty_tuple.type = call %bound_method.loc44_33.8(%.loc44_33.8)
// CHECK:STDOUT:     br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.done:
// CHECK:STDOUT:     %Destroy.WithSelf.SelfDestruct.bound.loc44_33.3: <bound method> = bound_method %.loc44_33.8, constants.%Destroy.WithSelf.SelfDestruct.4f4
// CHECK:STDOUT:     %Destroy.facet.loc44_33.5: %Destroy.type = facet_value constants.%Optional.6a3, (constants.%custom_witness.159) [concrete = constants.%Destroy.facet.2bb]
// CHECK:STDOUT:     %.loc44_33.22: %Destroy.type = converted constants.%Optional.6a3, %Destroy.facet.loc44_33.5 [concrete = constants.%Destroy.facet.2bb]
// CHECK:STDOUT:     %Destroy.facet.loc44_33.6: %Destroy.type = facet_value constants.%Optional.6a3, (constants.%custom_witness.159) [concrete = constants.%Destroy.facet.2bb]
// CHECK:STDOUT:     %.loc44_33.23: %Destroy.type = converted constants.%Optional.6a3, %Destroy.facet.loc44_33.6 [concrete = constants.%Destroy.facet.2bb]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %bound_method.loc44_33.9: <bound method> = bound_method %.loc44_33.8, %Destroy.WithSelf.SelfDestruct.specific_fn.3
// CHECK:STDOUT:     %Destroy.WithSelf.SelfDestruct.call.loc44_33.3: init %empty_tuple.type = call %bound_method.loc44_33.9(%.loc44_33.8)
// CHECK:STDOUT:     %impl.elem2.loc44_33.2: @TestIterate.%.loc44_33.28 (%.da7) = impl_witness_access constants.%Destroy.lookup_impl_witness.2cf, element2 [symbolic = %impl.elem2.loc44_33.4 (constants.%impl.elem2.fe5)]
// CHECK:STDOUT:     %bound_method.loc44_33.10: <bound method> = bound_method %var, %impl.elem2.loc44_33.2
// CHECK:STDOUT:     %.loc44_33.24: %Destroy.type = converted constants.%as_type.e58, constants.%impl.elem1.132 [symbolic = %impl.elem1 (constants.%impl.elem1.132)]
// CHECK:STDOUT:     %.loc44_33.25: %Destroy.type = converted constants.%as_type.e58, constants.%impl.elem1.132 [symbolic = %impl.elem1 (constants.%impl.elem1.132)]
// CHECK:STDOUT:     %specific_impl_fn.loc44_33.3: <specific function> = specific_impl_function %impl.elem2.loc44_33.2, @Destroy.WithSelf.SelfDestruct(constants.%impl.elem1.132) [symbolic = %specific_impl_fn.loc44_33.6 (constants.%specific_impl_fn.d19)]
// CHECK:STDOUT:     %bound_method.loc44_33.11: <bound method> = bound_method %var, %specific_impl_fn.loc44_33.3
// CHECK:STDOUT:     %Destroy.WithSelf.SelfDestruct.call.loc44_33.4: init %empty_tuple.type = call %bound_method.loc44_33.11(%var)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc44_33.1(%self.param: ref %.a18) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc44_33.2(%self.param: ref %MaybeUnformed.84b) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc44_33.3(%self.param: ref bool) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc44_33.4(%self.param: ref %struct_type.value.has_value.bd3) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc44_33.5(%self.param: ref %DefaultOptionalStorage.ee5) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc44_33.6(%self.param: ref %Optional.6a3) {
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
// CHECK:STDOUT:   %impl.elem2.loc59: %.78a = impl_witness_access constants.%Iterate.impl_witness.de0, element2 [concrete = constants.%T.as_type.as.Iterate.impl.NewCursor.d94]
// CHECK:STDOUT:   %bound_method.loc59_33.1: <bound method> = bound_method %m.ref, %impl.elem2.loc59
// CHECK:STDOUT:   %facet_value.loc59_33.1: %CppRange_where.type.d682d6.1 = facet_value constants.%DiffMutableRange, (constants.%custom_witness.c8d, constants.%custom_witness.d0da43.1, constants.%custom_witness.aa8, constants.%custom_witness.9b3, constants.%custom_witness.392, constants.%custom_witness.1c9, constants.%custom_witness.723, constants.%custom_witness.161) [concrete = constants.%facet_value.179]
// CHECK:STDOUT:   %.loc59_33.1: %CppRange_where.type.d682d6.1 = converted constants.%DiffMutableRange, %facet_value.loc59_33.1 [concrete = constants.%facet_value.179]
// CHECK:STDOUT:   %facet_value.loc59_33.2: %CppRange_where.type.d682d6.1 = facet_value constants.%DiffMutableRange, (constants.%custom_witness.c8d, constants.%custom_witness.d0da43.1, constants.%custom_witness.aa8, constants.%custom_witness.9b3, constants.%custom_witness.392, constants.%custom_witness.1c9, constants.%custom_witness.723, constants.%custom_witness.161) [concrete = constants.%facet_value.179]
// CHECK:STDOUT:   %.loc59_33.2: %CppRange_where.type.d682d6.1 = converted constants.%DiffMutableRange, %facet_value.loc59_33.2 [concrete = constants.%facet_value.179]
// CHECK:STDOUT:   %specific_fn.loc59_33.1: <specific function> = specific_function %impl.elem2.loc59, @T.as_type.as.Iterate.impl.NewCursor(constants.%facet_value.179) [concrete = constants.%T.as_type.as.Iterate.impl.NewCursor.specific_fn.375]
// CHECK:STDOUT:   %bound_method.loc59_33.2: <bound method> = bound_method %m.ref, %specific_fn.loc59_33.1
// CHECK:STDOUT:   %var.loc59: ref %tuple.type.b18 = var_storage invalid
// CHECK:STDOUT:   %.loc59_32.1: %DiffMutableRange = acquire_value %m.ref
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.call.loc59: init %tuple.type.b18 to %var.loc59 = call %bound_method.loc59_33.2(%.loc59_32.1)
// CHECK:STDOUT:   assign %var.loc59, %T.as_type.as.Iterate.impl.NewCursor.call.loc59
// CHECK:STDOUT:   br !for.next.loc59
// CHECK:STDOUT:
// CHECK:STDOUT: !for.next.loc59:
// CHECK:STDOUT:   %addr.loc59: %ptr.3ce = addr_of %var.loc59
// CHECK:STDOUT:   %impl.elem3.loc59: %.4e1 = impl_witness_access constants.%Iterate.impl_witness.de0, element3 [concrete = constants.%T.as_type.as.Iterate.impl.Next.10b]
// CHECK:STDOUT:   %bound_method.loc59_33.3: <bound method> = bound_method %m.ref, %impl.elem3.loc59
// CHECK:STDOUT:   %facet_value.loc59_33.3: %CppRange_where.type.d682d6.1 = facet_value constants.%DiffMutableRange, (constants.%custom_witness.c8d, constants.%custom_witness.d0da43.1, constants.%custom_witness.aa8, constants.%custom_witness.9b3, constants.%custom_witness.392, constants.%custom_witness.1c9, constants.%custom_witness.723, constants.%custom_witness.161) [concrete = constants.%facet_value.179]
// CHECK:STDOUT:   %.loc59_33.3: %CppRange_where.type.d682d6.1 = converted constants.%DiffMutableRange, %facet_value.loc59_33.3 [concrete = constants.%facet_value.179]
// CHECK:STDOUT:   %facet_value.loc59_33.4: %CppRange_where.type.d682d6.1 = facet_value constants.%DiffMutableRange, (constants.%custom_witness.c8d, constants.%custom_witness.d0da43.1, constants.%custom_witness.aa8, constants.%custom_witness.9b3, constants.%custom_witness.392, constants.%custom_witness.1c9, constants.%custom_witness.723, constants.%custom_witness.161) [concrete = constants.%facet_value.179]
// CHECK:STDOUT:   %.loc59_33.4: %CppRange_where.type.d682d6.1 = converted constants.%DiffMutableRange, %facet_value.loc59_33.4 [concrete = constants.%facet_value.179]
// CHECK:STDOUT:   %specific_fn.loc59_33.2: <specific function> = specific_function %impl.elem3.loc59, @T.as_type.as.Iterate.impl.Next(constants.%facet_value.179) [concrete = constants.%T.as_type.as.Iterate.impl.Next.specific_fn.dcf]
// CHECK:STDOUT:   %bound_method.loc59_33.4: <bound method> = bound_method %m.ref, %specific_fn.loc59_33.2
// CHECK:STDOUT:   %.loc59_33.5: ref %Optional.6a3 = temporary_storage
// CHECK:STDOUT:   %.loc59_32.2: %DiffMutableRange = acquire_value %m.ref
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.call.loc59: init %Optional.6a3 to %.loc59_33.5 = call %bound_method.loc59_33.4(%.loc59_32.2, %addr.loc59)
// CHECK:STDOUT:   %.loc59_33.6: ref %Optional.6a3 = temporary %.loc59_33.5, %T.as_type.as.Iterate.impl.Next.call.loc59
// CHECK:STDOUT:   %.loc59_33.7: %Optional.HasValue.type.11b = specific_constant imports.%Core.import_ref.bb6, @Optional(constants.%OptionalStorage.facet.d9a) [concrete = constants.%Optional.HasValue.453]
// CHECK:STDOUT:   %HasValue.ref.loc59: %Optional.HasValue.type.11b = name_ref HasValue, %.loc59_33.7 [concrete = constants.%Optional.HasValue.453]
// CHECK:STDOUT:   %Optional.HasValue.bound.loc59: <bound method> = bound_method %.loc59_33.6, %HasValue.ref.loc59
// CHECK:STDOUT:   %Optional.HasValue.specific_fn.loc59: <specific function> = specific_function %HasValue.ref.loc59, @Optional.HasValue(constants.%OptionalStorage.facet.d9a) [concrete = constants.%Optional.HasValue.specific_fn.b77]
// CHECK:STDOUT:   %bound_method.loc59_33.5: <bound method> = bound_method %.loc59_33.6, %Optional.HasValue.specific_fn.loc59
// CHECK:STDOUT:   %.loc59_33.8: %Optional.6a3 = acquire_value %.loc59_33.6
// CHECK:STDOUT:   %Optional.HasValue.call.loc59: init bool = call %bound_method.loc59_33.5(%.loc59_33.8)
// CHECK:STDOUT:   %.loc59_33.9: bool = value_of_initializer %Optional.HasValue.call.loc59
// CHECK:STDOUT:   %.loc59_33.10: bool = converted %Optional.HasValue.call.loc59, %.loc59_33.9
// CHECK:STDOUT:   if %.loc59_33.10 br !for.body.loc59 else br !for.done.loc59
// CHECK:STDOUT:
// CHECK:STDOUT: !for.body.loc59:
// CHECK:STDOUT:   %.loc59_33.11: %Optional.Get.type.d6e = specific_constant imports.%Core.import_ref.dab, @Optional(constants.%OptionalStorage.facet.d9a) [concrete = constants.%Optional.Get.902]
// CHECK:STDOUT:   %Get.ref.loc59: %Optional.Get.type.d6e = name_ref Get, %.loc59_33.11 [concrete = constants.%Optional.Get.902]
// CHECK:STDOUT:   %Optional.Get.bound.loc59: <bound method> = bound_method %.loc59_33.6, %Get.ref.loc59
// CHECK:STDOUT:   %Optional.Get.specific_fn.loc59: <specific function> = specific_function %Get.ref.loc59, @Optional.Get(constants.%OptionalStorage.facet.d9a) [concrete = constants.%Optional.Get.specific_fn.f40]
// CHECK:STDOUT:   %bound_method.loc59_33.6: <bound method> = bound_method %.loc59_33.6, %Optional.Get.specific_fn.loc59
// CHECK:STDOUT:   %.loc59_33.12: ref %DiffValueType = temporary_storage
// CHECK:STDOUT:   %.loc59_33.13: %Optional.6a3 = acquire_value %.loc59_33.6
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
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc59_33.1: <bound method> = bound_method %.loc59_33.14, constants.%Destroy.WithSelf.SelfDestruct.d38
// CHECK:STDOUT:   %Destroy.facet.loc59_33.1: %Destroy.type = facet_value constants.%DiffValueType, (constants.%custom_witness.c8d) [concrete = constants.%Destroy.facet.f3c]
// CHECK:STDOUT:   %.loc59_33.16: %Destroy.type = converted constants.%DiffValueType, %Destroy.facet.loc59_33.1 [concrete = constants.%Destroy.facet.f3c]
// CHECK:STDOUT:   %Destroy.facet.loc59_33.2: %Destroy.type = facet_value constants.%DiffValueType, (constants.%custom_witness.c8d) [concrete = constants.%Destroy.facet.f3c]
// CHECK:STDOUT:   %.loc59_33.17: %Destroy.type = converted constants.%DiffValueType, %Destroy.facet.loc59_33.2 [concrete = constants.%Destroy.facet.f3c]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc59_33.7: <bound method> = bound_method %.loc59_33.14, %Destroy.WithSelf.SelfDestruct.specific_fn.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc59_33.1: init %empty_tuple.type = call %bound_method.loc59_33.7(%.loc59_33.14)
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc59_33.2: <bound method> = bound_method %.loc59_33.6, constants.%Destroy.WithSelf.SelfDestruct.4f4
// CHECK:STDOUT:   %Destroy.facet.loc59_33.3: %Destroy.type = facet_value constants.%Optional.6a3, (constants.%custom_witness.159) [concrete = constants.%Destroy.facet.2bb]
// CHECK:STDOUT:   %.loc59_33.18: %Destroy.type = converted constants.%Optional.6a3, %Destroy.facet.loc59_33.3 [concrete = constants.%Destroy.facet.2bb]
// CHECK:STDOUT:   %Destroy.facet.loc59_33.4: %Destroy.type = facet_value constants.%Optional.6a3, (constants.%custom_witness.159) [concrete = constants.%Destroy.facet.2bb]
// CHECK:STDOUT:   %.loc59_33.19: %Destroy.type = converted constants.%Optional.6a3, %Destroy.facet.loc59_33.4 [concrete = constants.%Destroy.facet.2bb]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc59_33.8: <bound method> = bound_method %.loc59_33.6, %Destroy.WithSelf.SelfDestruct.specific_fn.2
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc59_33.2: init %empty_tuple.type = call %bound_method.loc59_33.8(%.loc59_33.6)
// CHECK:STDOUT:   br !for.next.loc59
// CHECK:STDOUT:
// CHECK:STDOUT: !for.done.loc59:
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc59_33.3: <bound method> = bound_method %.loc59_33.6, constants.%Destroy.WithSelf.SelfDestruct.4f4
// CHECK:STDOUT:   %Destroy.facet.loc59_33.5: %Destroy.type = facet_value constants.%Optional.6a3, (constants.%custom_witness.159) [concrete = constants.%Destroy.facet.2bb]
// CHECK:STDOUT:   %.loc59_33.20: %Destroy.type = converted constants.%Optional.6a3, %Destroy.facet.loc59_33.5 [concrete = constants.%Destroy.facet.2bb]
// CHECK:STDOUT:   %Destroy.facet.loc59_33.6: %Destroy.type = facet_value constants.%Optional.6a3, (constants.%custom_witness.159) [concrete = constants.%Destroy.facet.2bb]
// CHECK:STDOUT:   %.loc59_33.21: %Destroy.type = converted constants.%Optional.6a3, %Destroy.facet.loc59_33.6 [concrete = constants.%Destroy.facet.2bb]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc59_33.9: <bound method> = bound_method %.loc59_33.6, %Destroy.WithSelf.SelfDestruct.specific_fn.3
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc59_33.3: init %empty_tuple.type = call %bound_method.loc59_33.9(%.loc59_33.6)
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc59_33.4: <bound method> = bound_method %var.loc59, constants.%Destroy.WithSelf.SelfDestruct.aca
// CHECK:STDOUT:   %Destroy.facet.loc59_33.7: %Destroy.type = facet_value constants.%tuple.type.b18, (constants.%custom_witness.3fe) [concrete = constants.%Destroy.facet.fb2]
// CHECK:STDOUT:   %.loc59_33.22: %Destroy.type = converted constants.%tuple.type.b18, %Destroy.facet.loc59_33.7 [concrete = constants.%Destroy.facet.fb2]
// CHECK:STDOUT:   %Destroy.facet.loc59_33.8: %Destroy.type = facet_value constants.%tuple.type.b18, (constants.%custom_witness.3fe) [concrete = constants.%Destroy.facet.fb2]
// CHECK:STDOUT:   %.loc59_33.23: %Destroy.type = converted constants.%tuple.type.b18, %Destroy.facet.loc59_33.8 [concrete = constants.%Destroy.facet.fb2]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc59_33.10: <bound method> = bound_method %var.loc59, %Destroy.WithSelf.SelfDestruct.specific_fn.4
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc59_33.4: init %empty_tuple.type = call %bound_method.loc59_33.10(%var.loc59)
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %i.patt.loc63: %pattern_type.149 = value_binding_pattern i [concrete = constants.%i.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c.ref: %DiffConstRange = name_ref c, %c
// CHECK:STDOUT:   %impl.elem2.loc63: %.d61 = impl_witness_access constants.%Iterate.impl_witness.165, element2 [concrete = constants.%T.as_type.as.Iterate.impl.NewCursor.1d0]
// CHECK:STDOUT:   %bound_method.loc63_33.1: <bound method> = bound_method %c.ref, %impl.elem2.loc63
// CHECK:STDOUT:   %facet_value.loc63_33.1: %CppRange_where.type.d682d6.1 = facet_value constants.%DiffConstRange, (constants.%custom_witness.c8d, constants.%custom_witness.d0da43.1, constants.%custom_witness.7b4, constants.%custom_witness.a60, constants.%custom_witness.0c0, constants.%custom_witness.ffb, constants.%custom_witness.a26, constants.%custom_witness.dd2) [concrete = constants.%facet_value.af9]
// CHECK:STDOUT:   %.loc63_33.1: %CppRange_where.type.d682d6.1 = converted constants.%DiffConstRange, %facet_value.loc63_33.1 [concrete = constants.%facet_value.af9]
// CHECK:STDOUT:   %facet_value.loc63_33.2: %CppRange_where.type.d682d6.1 = facet_value constants.%DiffConstRange, (constants.%custom_witness.c8d, constants.%custom_witness.d0da43.1, constants.%custom_witness.7b4, constants.%custom_witness.a60, constants.%custom_witness.0c0, constants.%custom_witness.ffb, constants.%custom_witness.a26, constants.%custom_witness.dd2) [concrete = constants.%facet_value.af9]
// CHECK:STDOUT:   %.loc63_33.2: %CppRange_where.type.d682d6.1 = converted constants.%DiffConstRange, %facet_value.loc63_33.2 [concrete = constants.%facet_value.af9]
// CHECK:STDOUT:   %specific_fn.loc63_33.1: <specific function> = specific_function %impl.elem2.loc63, @T.as_type.as.Iterate.impl.NewCursor(constants.%facet_value.af9) [concrete = constants.%T.as_type.as.Iterate.impl.NewCursor.specific_fn.080]
// CHECK:STDOUT:   %bound_method.loc63_33.2: <bound method> = bound_method %c.ref, %specific_fn.loc63_33.1
// CHECK:STDOUT:   %var.loc63: ref %tuple.type.df5 = var_storage invalid
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.call.loc63: init %tuple.type.df5 to %var.loc63 = call %bound_method.loc63_33.2(%c.ref)
// CHECK:STDOUT:   assign %var.loc63, %T.as_type.as.Iterate.impl.NewCursor.call.loc63
// CHECK:STDOUT:   br !for.next.loc63
// CHECK:STDOUT:
// CHECK:STDOUT: !for.next.loc63:
// CHECK:STDOUT:   %addr.loc63: %ptr.310 = addr_of %var.loc63
// CHECK:STDOUT:   %impl.elem3.loc63: %.77a = impl_witness_access constants.%Iterate.impl_witness.165, element3 [concrete = constants.%T.as_type.as.Iterate.impl.Next.dce]
// CHECK:STDOUT:   %bound_method.loc63_33.3: <bound method> = bound_method %c.ref, %impl.elem3.loc63
// CHECK:STDOUT:   %facet_value.loc63_33.3: %CppRange_where.type.d682d6.1 = facet_value constants.%DiffConstRange, (constants.%custom_witness.c8d, constants.%custom_witness.d0da43.1, constants.%custom_witness.7b4, constants.%custom_witness.a60, constants.%custom_witness.0c0, constants.%custom_witness.ffb, constants.%custom_witness.a26, constants.%custom_witness.dd2) [concrete = constants.%facet_value.af9]
// CHECK:STDOUT:   %.loc63_33.3: %CppRange_where.type.d682d6.1 = converted constants.%DiffConstRange, %facet_value.loc63_33.3 [concrete = constants.%facet_value.af9]
// CHECK:STDOUT:   %facet_value.loc63_33.4: %CppRange_where.type.d682d6.1 = facet_value constants.%DiffConstRange, (constants.%custom_witness.c8d, constants.%custom_witness.d0da43.1, constants.%custom_witness.7b4, constants.%custom_witness.a60, constants.%custom_witness.0c0, constants.%custom_witness.ffb, constants.%custom_witness.a26, constants.%custom_witness.dd2) [concrete = constants.%facet_value.af9]
// CHECK:STDOUT:   %.loc63_33.4: %CppRange_where.type.d682d6.1 = converted constants.%DiffConstRange, %facet_value.loc63_33.4 [concrete = constants.%facet_value.af9]
// CHECK:STDOUT:   %specific_fn.loc63_33.2: <specific function> = specific_function %impl.elem3.loc63, @T.as_type.as.Iterate.impl.Next(constants.%facet_value.af9) [concrete = constants.%T.as_type.as.Iterate.impl.Next.specific_fn.5bd]
// CHECK:STDOUT:   %bound_method.loc63_33.4: <bound method> = bound_method %c.ref, %specific_fn.loc63_33.2
// CHECK:STDOUT:   %.loc63_33.5: ref %Optional.6a3 = temporary_storage
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.call.loc63: init %Optional.6a3 to %.loc63_33.5 = call %bound_method.loc63_33.4(%c.ref, %addr.loc63)
// CHECK:STDOUT:   %.loc63_33.6: ref %Optional.6a3 = temporary %.loc63_33.5, %T.as_type.as.Iterate.impl.Next.call.loc63
// CHECK:STDOUT:   %.loc63_33.7: %Optional.HasValue.type.11b = specific_constant imports.%Core.import_ref.bb6, @Optional(constants.%OptionalStorage.facet.d9a) [concrete = constants.%Optional.HasValue.453]
// CHECK:STDOUT:   %HasValue.ref.loc63: %Optional.HasValue.type.11b = name_ref HasValue, %.loc63_33.7 [concrete = constants.%Optional.HasValue.453]
// CHECK:STDOUT:   %Optional.HasValue.bound.loc63: <bound method> = bound_method %.loc63_33.6, %HasValue.ref.loc63
// CHECK:STDOUT:   %Optional.HasValue.specific_fn.loc63: <specific function> = specific_function %HasValue.ref.loc63, @Optional.HasValue(constants.%OptionalStorage.facet.d9a) [concrete = constants.%Optional.HasValue.specific_fn.b77]
// CHECK:STDOUT:   %bound_method.loc63_33.5: <bound method> = bound_method %.loc63_33.6, %Optional.HasValue.specific_fn.loc63
// CHECK:STDOUT:   %.loc63_33.8: %Optional.6a3 = acquire_value %.loc63_33.6
// CHECK:STDOUT:   %Optional.HasValue.call.loc63: init bool = call %bound_method.loc63_33.5(%.loc63_33.8)
// CHECK:STDOUT:   %.loc63_33.9: bool = value_of_initializer %Optional.HasValue.call.loc63
// CHECK:STDOUT:   %.loc63_33.10: bool = converted %Optional.HasValue.call.loc63, %.loc63_33.9
// CHECK:STDOUT:   if %.loc63_33.10 br !for.body.loc63 else br !for.done.loc63
// CHECK:STDOUT:
// CHECK:STDOUT: !for.body.loc63:
// CHECK:STDOUT:   %.loc63_33.11: %Optional.Get.type.d6e = specific_constant imports.%Core.import_ref.dab, @Optional(constants.%OptionalStorage.facet.d9a) [concrete = constants.%Optional.Get.902]
// CHECK:STDOUT:   %Get.ref.loc63: %Optional.Get.type.d6e = name_ref Get, %.loc63_33.11 [concrete = constants.%Optional.Get.902]
// CHECK:STDOUT:   %Optional.Get.bound.loc63: <bound method> = bound_method %.loc63_33.6, %Get.ref.loc63
// CHECK:STDOUT:   %Optional.Get.specific_fn.loc63: <specific function> = specific_function %Get.ref.loc63, @Optional.Get(constants.%OptionalStorage.facet.d9a) [concrete = constants.%Optional.Get.specific_fn.f40]
// CHECK:STDOUT:   %bound_method.loc63_33.6: <bound method> = bound_method %.loc63_33.6, %Optional.Get.specific_fn.loc63
// CHECK:STDOUT:   %.loc63_33.12: ref %DiffValueType = temporary_storage
// CHECK:STDOUT:   %.loc63_33.13: %Optional.6a3 = acquire_value %.loc63_33.6
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
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc63_33.1: <bound method> = bound_method %.loc63_33.14, constants.%Destroy.WithSelf.SelfDestruct.d38
// CHECK:STDOUT:   %Destroy.facet.loc63_33.1: %Destroy.type = facet_value constants.%DiffValueType, (constants.%custom_witness.c8d) [concrete = constants.%Destroy.facet.f3c]
// CHECK:STDOUT:   %.loc63_33.16: %Destroy.type = converted constants.%DiffValueType, %Destroy.facet.loc63_33.1 [concrete = constants.%Destroy.facet.f3c]
// CHECK:STDOUT:   %Destroy.facet.loc63_33.2: %Destroy.type = facet_value constants.%DiffValueType, (constants.%custom_witness.c8d) [concrete = constants.%Destroy.facet.f3c]
// CHECK:STDOUT:   %.loc63_33.17: %Destroy.type = converted constants.%DiffValueType, %Destroy.facet.loc63_33.2 [concrete = constants.%Destroy.facet.f3c]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc63_33.7: <bound method> = bound_method %.loc63_33.14, %Destroy.WithSelf.SelfDestruct.specific_fn.5
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc63_33.1: init %empty_tuple.type = call %bound_method.loc63_33.7(%.loc63_33.14)
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc63_33.2: <bound method> = bound_method %.loc63_33.6, constants.%Destroy.WithSelf.SelfDestruct.4f4
// CHECK:STDOUT:   %Destroy.facet.loc63_33.3: %Destroy.type = facet_value constants.%Optional.6a3, (constants.%custom_witness.159) [concrete = constants.%Destroy.facet.2bb]
// CHECK:STDOUT:   %.loc63_33.18: %Destroy.type = converted constants.%Optional.6a3, %Destroy.facet.loc63_33.3 [concrete = constants.%Destroy.facet.2bb]
// CHECK:STDOUT:   %Destroy.facet.loc63_33.4: %Destroy.type = facet_value constants.%Optional.6a3, (constants.%custom_witness.159) [concrete = constants.%Destroy.facet.2bb]
// CHECK:STDOUT:   %.loc63_33.19: %Destroy.type = converted constants.%Optional.6a3, %Destroy.facet.loc63_33.4 [concrete = constants.%Destroy.facet.2bb]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc63_33.8: <bound method> = bound_method %.loc63_33.6, %Destroy.WithSelf.SelfDestruct.specific_fn.6
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc63_33.2: init %empty_tuple.type = call %bound_method.loc63_33.8(%.loc63_33.6)
// CHECK:STDOUT:   br !for.next.loc63
// CHECK:STDOUT:
// CHECK:STDOUT: !for.done.loc63:
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc63_33.3: <bound method> = bound_method %.loc63_33.6, constants.%Destroy.WithSelf.SelfDestruct.4f4
// CHECK:STDOUT:   %Destroy.facet.loc63_33.5: %Destroy.type = facet_value constants.%Optional.6a3, (constants.%custom_witness.159) [concrete = constants.%Destroy.facet.2bb]
// CHECK:STDOUT:   %.loc63_33.20: %Destroy.type = converted constants.%Optional.6a3, %Destroy.facet.loc63_33.5 [concrete = constants.%Destroy.facet.2bb]
// CHECK:STDOUT:   %Destroy.facet.loc63_33.6: %Destroy.type = facet_value constants.%Optional.6a3, (constants.%custom_witness.159) [concrete = constants.%Destroy.facet.2bb]
// CHECK:STDOUT:   %.loc63_33.21: %Destroy.type = converted constants.%Optional.6a3, %Destroy.facet.loc63_33.6 [concrete = constants.%Destroy.facet.2bb]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc63_33.9: <bound method> = bound_method %.loc63_33.6, %Destroy.WithSelf.SelfDestruct.specific_fn.7
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc63_33.3: init %empty_tuple.type = call %bound_method.loc63_33.9(%.loc63_33.6)
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc63_33.4: <bound method> = bound_method %var.loc63, constants.%Destroy.WithSelf.SelfDestruct.6d7
// CHECK:STDOUT:   %Destroy.facet.loc63_33.7: %Destroy.type = facet_value constants.%tuple.type.df5, (constants.%custom_witness.34c) [concrete = constants.%Destroy.facet.bca]
// CHECK:STDOUT:   %.loc63_33.22: %Destroy.type = converted constants.%tuple.type.df5, %Destroy.facet.loc63_33.7 [concrete = constants.%Destroy.facet.bca]
// CHECK:STDOUT:   %Destroy.facet.loc63_33.8: %Destroy.type = facet_value constants.%tuple.type.df5, (constants.%custom_witness.34c) [concrete = constants.%Destroy.facet.bca]
// CHECK:STDOUT:   %.loc63_33.23: %Destroy.type = converted constants.%tuple.type.df5, %Destroy.facet.loc63_33.8 [concrete = constants.%Destroy.facet.bca]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc63_33.10: <bound method> = bound_method %var.loc63, %Destroy.WithSelf.SelfDestruct.specific_fn.8
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc63_33.4: init %empty_tuple.type = call %bound_method.loc63_33.10(%var.loc63)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestIterate(constants.%R) {
// CHECK:STDOUT:   %R.patt.loc40_17.2 => constants.%R.patt
// CHECK:STDOUT:   %R.loc40_17.1 => constants.%R
// CHECK:STDOUT:   %R.as_type.loc40_79.1 => constants.%R.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.345
// CHECK:STDOUT:   %r.patt.loc40_77.2 => constants.%r.patt.48b
// CHECK:STDOUT:   %r.param_patt.loc40_72.2 => constants.%r.param_patt.217
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestIterate(constants.%facet_value.b23) {
// CHECK:STDOUT:   %R.patt.loc40_17.2 => constants.%R.patt
// CHECK:STDOUT:   %R.loc40_17.1 => constants.%facet_value.b23
// CHECK:STDOUT:   %R.as_type.loc40_79.1 => constants.%DiffMutableRange
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.611
// CHECK:STDOUT:   %r.patt.loc40_77.2 => constants.%r.patt.5fe
// CHECK:STDOUT:   %r.param_patt.loc40_72.2 => constants.%r.param_patt.a4b
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc40 => constants.%complete_type.357
// CHECK:STDOUT:   %Iterate.lookup_impl_witness => constants.%Iterate.impl_witness.de0
// CHECK:STDOUT:   %Iterate.facet.loc44_33.5 => constants.%Iterate.facet.b21
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type => constants.%Iterate.WithSelf.NewCursor.type.0d5
// CHECK:STDOUT:   %.loc44_33.26 => constants.%.78a
// CHECK:STDOUT:   %impl.elem2.loc44_33.3 => constants.%T.as_type.as.Iterate.impl.NewCursor.d94
// CHECK:STDOUT:   %specific_impl_fn.loc44_33.4 => constants.%T.as_type.as.Iterate.impl.NewCursor.specific_fn.375
// CHECK:STDOUT:   %impl.elem1 => constants.%Destroy.facet.fb2
// CHECK:STDOUT:   %as_type => constants.%tuple.type.b18
// CHECK:STDOUT:   %require_complete.1 => constants.%complete_type.bfe
// CHECK:STDOUT:   %ptr => constants.%ptr.3ce
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type => constants.%Iterate.WithSelf.Next.type.8a9
// CHECK:STDOUT:   %.loc44_33.27 => constants.%.4e1
// CHECK:STDOUT:   %impl.elem3.loc44_33.2 => constants.%T.as_type.as.Iterate.impl.Next.10b
// CHECK:STDOUT:   %specific_impl_fn.loc44_33.5 => constants.%T.as_type.as.Iterate.impl.Next.specific_fn.dcf
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type => constants.%Destroy.WithSelf.SelfDestruct.type.e0d
// CHECK:STDOUT:   %.loc44_33.28 => constants.%.433
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.3fe
// CHECK:STDOUT:   %impl.elem2.loc44_33.4 => constants.%Destroy.WithSelf.SelfDestruct.aca
// CHECK:STDOUT:   %specific_impl_fn.loc44_33.6 => constants.%Destroy.WithSelf.SelfDestruct.specific_fn.d41
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestIterate(constants.%facet_value.4d4) {
// CHECK:STDOUT:   %R.patt.loc40_17.2 => constants.%R.patt
// CHECK:STDOUT:   %R.loc40_17.1 => constants.%facet_value.4d4
// CHECK:STDOUT:   %R.as_type.loc40_79.1 => constants.%DiffConstRange
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.7dd
// CHECK:STDOUT:   %r.patt.loc40_77.2 => constants.%r.patt.9c0
// CHECK:STDOUT:   %r.param_patt.loc40_72.2 => constants.%r.param_patt.992
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc40 => constants.%complete_type.357
// CHECK:STDOUT:   %Iterate.lookup_impl_witness => constants.%Iterate.impl_witness.165
// CHECK:STDOUT:   %Iterate.facet.loc44_33.5 => constants.%Iterate.facet.ad6
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type => constants.%Iterate.WithSelf.NewCursor.type.5bf
// CHECK:STDOUT:   %.loc44_33.26 => constants.%.d61
// CHECK:STDOUT:   %impl.elem2.loc44_33.3 => constants.%T.as_type.as.Iterate.impl.NewCursor.1d0
// CHECK:STDOUT:   %specific_impl_fn.loc44_33.4 => constants.%T.as_type.as.Iterate.impl.NewCursor.specific_fn.080
// CHECK:STDOUT:   %impl.elem1 => constants.%Destroy.facet.bca
// CHECK:STDOUT:   %as_type => constants.%tuple.type.df5
// CHECK:STDOUT:   %require_complete.1 => constants.%complete_type.ad6
// CHECK:STDOUT:   %ptr => constants.%ptr.310
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type => constants.%Iterate.WithSelf.Next.type.a66
// CHECK:STDOUT:   %.loc44_33.27 => constants.%.77a
// CHECK:STDOUT:   %impl.elem3.loc44_33.2 => constants.%T.as_type.as.Iterate.impl.Next.dce
// CHECK:STDOUT:   %specific_impl_fn.loc44_33.5 => constants.%T.as_type.as.Iterate.impl.Next.specific_fn.5bd
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type => constants.%Destroy.WithSelf.SelfDestruct.type.b92
// CHECK:STDOUT:   %.loc44_33.28 => constants.%.5913
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.34c
// CHECK:STDOUT:   %impl.elem2.loc44_33.4 => constants.%Destroy.WithSelf.SelfDestruct.6d7
// CHECK:STDOUT:   %specific_impl_fn.loc44_33.6 => constants.%Destroy.WithSelf.SelfDestruct.specific_fn.33f
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- method_overloads.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
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
// CHECK:STDOUT:   %OverloadValueType.type.facet: %type = facet_value %OverloadValueType, () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.881: type = fn_type @Destroy.WithSelf.SubobjectDestroy, @Destroy.WithSelf(%OverloadValueType.type.facet) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d5c: %Destroy.WithSelf.SubobjectDestroy.type.881 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.ba1: type = fn_type @Destroy.WithSelf.SelfDestruct, @Destroy.WithSelf(%OverloadValueType.type.facet) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.d92: %Destroy.WithSelf.SelfDestruct.type.ba1 = struct_value () [concrete]
// CHECK:STDOUT:   %OverloadValueType.Op.type.83af6e.1: type = fn_type @OverloadValueType.Op.1 [concrete]
// CHECK:STDOUT:   %OverloadValueType.Op.4dca31.1: %OverloadValueType.Op.type.83af6e.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.ae2: <witness> = custom_witness (%OverloadValueType.Op.4dca31.1, %Destroy.WithSelf.SubobjectDestroy.d5c, %Destroy.WithSelf.SelfDestruct.d92), @Destroy [concrete]
// CHECK:STDOUT:   %OverloadValueType.Op.type.83af6e.2: type = fn_type @OverloadValueType.Op.2 [concrete]
// CHECK:STDOUT:   %OverloadValueType.Op.4dca31.2: %OverloadValueType.Op.type.83af6e.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.d0da43.1: <witness> = custom_witness (%OverloadValueType.Op.4dca31.2), @Copy [concrete]
// CHECK:STDOUT:   %facet_value.d74: %facet_type.f48 = facet_value %OverloadValueType, (%custom_witness.ae2, %custom_witness.d0da43.1) [concrete]
// CHECK:STDOUT:   %Iterate.lookup_impl_witness.1d8: <witness> = lookup_impl_witness %.Self.500, @Iterate [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.294: %facet_type.f48 = impl_witness_access %Iterate.lookup_impl_witness.1d8, element0 [symbolic_self]
// CHECK:STDOUT:   %Iterate_where.type.0a6: type = facet_type <@Iterate where %impl.elem0.294 = %facet_value.d74> [concrete]
// CHECK:STDOUT:   %pattern_type.a53: type = pattern_type %Iterate_where.type.0a6 [concrete]
// CHECK:STDOUT:   %R.patt: %pattern_type.a53 = symbolic_binding_pattern R, 0 [symbolic]
// CHECK:STDOUT:   %R: %Iterate_where.type.0a6 = symbolic_binding R, 0 [symbolic]
// CHECK:STDOUT:   %R.as_type: type = facet_access_type %R [symbolic]
// CHECK:STDOUT:   %pattern_type.2af: type = pattern_type %R.as_type [symbolic]
// CHECK:STDOUT:   %r.patt.b15: %pattern_type.2af = ref_binding_pattern r [symbolic]
// CHECK:STDOUT:   %r.param_patt.206: %pattern_type.2af = var_param_pattern %r.patt.b15 [symbolic]
// CHECK:STDOUT:   %_.patt.a10: %pattern_type.804 = value_binding_pattern _ [concrete]
// CHECK:STDOUT:   %Iterate.lookup_impl_witness.9e6: <witness> = lookup_impl_witness %R, @Iterate [symbolic]
// CHECK:STDOUT:   %Iterate.facet.7a0: %Iterate.type = facet_value %R.as_type, (%Iterate.lookup_impl_witness.9e6) [symbolic]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.a6a5: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.7a0) [symbolic]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.a4a: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.7a0) [symbolic]
// CHECK:STDOUT:   %.635: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.a6a5, %Iterate.facet.7a0 [symbolic]
// CHECK:STDOUT:   %impl.elem2.086: %.635 = impl_witness_access %Iterate.lookup_impl_witness.9e6, element2 [symbolic]
// CHECK:STDOUT:   %impl.elem1.35e: %Destroy.type = impl_witness_access %Iterate.lookup_impl_witness.9e6, element1 [symbolic]
// CHECK:STDOUT:   %as_type.f7d: type = facet_access_type %impl.elem1.35e [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.8ac: <specific function> = specific_impl_function %impl.elem2.086, @Iterate.WithSelf.NewCursor(%Iterate.facet.7a0) [symbolic]
// CHECK:STDOUT:   %ptr.6f8: type = ptr_type %as_type.f7d [symbolic]
// CHECK:STDOUT:   %.b08: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.a4a, %Iterate.facet.7a0 [symbolic]
// CHECK:STDOUT:   %impl.elem3.1c7: %.b08 = impl_witness_access %Iterate.lookup_impl_witness.9e6, element3 [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.4a5: type = class_type @DefaultOptionalStorage, @DefaultOptionalStorage(%facet_value.d74) [concrete]
// CHECK:STDOUT:   %OptionalStorage.impl_witness.8d7: <witness> = impl_witness imports.%OptionalStorage.impl_witness_table.460, @T.as_type.as.OptionalStorage.impl(%facet_value.d74) [concrete]
// CHECK:STDOUT:   %OptionalStorage.facet.0f7: %OptionalStorage.type = facet_value %OverloadValueType, (%OptionalStorage.impl_witness.8d7) [concrete]
// CHECK:STDOUT:   %Optional.f35: type = class_type @Optional, @Optional(%OptionalStorage.facet.0f7) [concrete]
// CHECK:STDOUT:   %specific_impl_fn.02d: <specific function> = specific_impl_function %impl.elem3.1c7, @Iterate.WithSelf.Next(%Iterate.facet.7a0) [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.type.6d2: type = fn_type @Optional.HasValue, @Optional(%OptionalStorage.facet.0f7) [concrete]
// CHECK:STDOUT:   %Optional.HasValue.304: %Optional.HasValue.type.6d2 = struct_value () [concrete]
// CHECK:STDOUT:   %Optional.Get.type.a45: type = fn_type @Optional.Get, @Optional(%OptionalStorage.facet.0f7) [concrete]
// CHECK:STDOUT:   %Optional.Get.b4d: %Optional.Get.type.a45 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.facet.909: %Destroy.type = facet_value %OverloadValueType, (%custom_witness.ae2) [concrete]
// CHECK:STDOUT:   %MaybeUnformed.275: type = class_type @MaybeUnformed, @MaybeUnformed(%Destroy.facet.909) [concrete]
// CHECK:STDOUT:   %.8a2: type = maybe_unformed_type %OverloadValueType [concrete]
// CHECK:STDOUT:   %struct_type.value.has_value.827: type = struct_type {.value: %MaybeUnformed.275, .has_value: bool} [concrete]
// CHECK:STDOUT:   %Optional.HasValue.specific_fn.855: <specific function> = specific_function %Optional.HasValue.304, @Optional.HasValue(%OptionalStorage.facet.0f7) [concrete]
// CHECK:STDOUT:   %Optional.Get.specific_fn.7d0: <specific function> = specific_function %Optional.Get.b4d, @Optional.Get(%OptionalStorage.facet.0f7) [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.7: type = fn_type @Destroy.Op.loc27_37.6 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.7: %Destroy.Op.type.1d8f74.7 = struct_value () [concrete]
// CHECK:STDOUT:   %Optional.type.facet: %type = facet_value %Optional.f35, () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.02e: type = fn_type @Destroy.WithSelf.SubobjectDestroy, @Destroy.WithSelf(%Optional.type.facet) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.0cf: %Destroy.WithSelf.SubobjectDestroy.type.02e = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.32d: type = fn_type @Destroy.WithSelf.SelfDestruct, @Destroy.WithSelf(%Optional.type.facet) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.e40: %Destroy.WithSelf.SelfDestruct.type.32d = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.5b3: <witness> = custom_witness (%Destroy.Op.1a2547.7, %Destroy.WithSelf.SubobjectDestroy.0cf, %Destroy.WithSelf.SelfDestruct.e40), @Destroy [concrete]
// CHECK:STDOUT:   %Destroy.facet.a53: %Destroy.type = facet_value %Optional.f35, (%custom_witness.5b3) [concrete]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.1e0: <witness> = lookup_impl_witness %impl.elem1.35e, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.0c5: type = fn_type @Destroy.WithSelf.SelfDestruct, @Destroy.WithSelf(%impl.elem1.35e) [symbolic]
// CHECK:STDOUT:   %.79a: type = fn_type_with_self_type %Destroy.WithSelf.SelfDestruct.type.0c5, %impl.elem1.35e [symbolic]
// CHECK:STDOUT:   %impl.elem2.084: %.79a = impl_witness_access %Destroy.lookup_impl_witness.1e0, element2 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.1a4: <specific function> = specific_impl_function %impl.elem2.084, @Destroy.WithSelf.SelfDestruct(%impl.elem1.35e) [symbolic]
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
// CHECK:STDOUT:   %Iterator.type.facet: %type = facet_value %Iterator, () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.8ff: type = fn_type @Destroy.WithSelf.SubobjectDestroy, @Destroy.WithSelf(%Iterator.type.facet) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.1e4: %Destroy.WithSelf.SubobjectDestroy.type.8ff = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.ada: type = fn_type @Destroy.WithSelf.SelfDestruct, @Destroy.WithSelf(%Iterator.type.facet) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.fdd: %Destroy.WithSelf.SelfDestruct.type.ada = struct_value () [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.4232e0.2: type = fn_type @Iterator.Op.2 [concrete]
// CHECK:STDOUT:   %Iterator.Op.e8caec.2: %Iterator.Op.type.4232e0.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.e0e: <witness> = custom_witness (%Iterator.Op.e8caec.2, %Destroy.WithSelf.SubobjectDestroy.1e4, %Destroy.WithSelf.SelfDestruct.fdd), @Destroy [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.4232e0.3: type = fn_type @Iterator.Op.3 [concrete]
// CHECK:STDOUT:   %Iterator.Op.e8caec.3: %Iterator.Op.type.4232e0.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.716: <witness> = custom_witness (%Iterator.Op.e8caec.3), @Inc [concrete]
// CHECK:STDOUT:   %Equal.type: type = fn_type @Equal [concrete]
// CHECK:STDOUT:   %Equal: %Equal.type = struct_value () [concrete]
// CHECK:STDOUT:   %NotEqual.type: type = fn_type @NotEqual [concrete]
// CHECK:STDOUT:   %NotEqual: %NotEqual.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.617: <witness> = custom_witness (%Equal, %NotEqual), @EqWith, @EqWith(%Iterator) [concrete]
// CHECK:STDOUT:   %facet_value.8c5: %CppRange_where.type.d682d6.1 = facet_value %Range, (%custom_witness.ae2, %custom_witness.d0da43.1, %custom_witness.090, %custom_witness.e0e, %custom_witness.6bd, %custom_witness.716, %custom_witness.617) [concrete]
// CHECK:STDOUT:   %tuple.type.9ef: type = tuple_type (%Iterator, %Iterator) [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.8: type = fn_type @Destroy.Op.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.8: %Destroy.Op.type.1d8f74.8 = struct_value () [concrete]
// CHECK:STDOUT:   %tuple.type.facet: %type = facet_value %tuple.type.9ef, () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dcc: type = fn_type @Destroy.WithSelf.SubobjectDestroy, @Destroy.WithSelf(%tuple.type.facet) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.45c: %Destroy.WithSelf.SubobjectDestroy.type.dcc = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.a9b: type = fn_type @Destroy.WithSelf.SelfDestruct, @Destroy.WithSelf(%tuple.type.facet) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.6f1: %Destroy.WithSelf.SelfDestruct.type.a9b = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.90f: <witness> = custom_witness (%Destroy.Op.1a2547.8, %Destroy.WithSelf.SubobjectDestroy.45c, %Destroy.WithSelf.SelfDestruct.6f1), @Destroy [concrete]
// CHECK:STDOUT:   %Destroy.facet.30e: %Destroy.type = facet_value %tuple.type.9ef, (%custom_witness.90f) [concrete]
// CHECK:STDOUT:   %Iterate.impl_witness.05a: <witness> = impl_witness imports.%Iterate.impl_witness_table.121, @T.as_type.as.Iterate.impl(%facet_value.8c5) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.type.3ef: type = fn_type @T.as_type.as.Iterate.impl.NewCursor, @T.as_type.as.Iterate.impl(%facet_value.8c5) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.a0c: %T.as_type.as.Iterate.impl.NewCursor.type.3ef = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.type.fc7: type = fn_type @T.as_type.as.Iterate.impl.Next, @T.as_type.as.Iterate.impl(%facet_value.8c5) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.d69: %T.as_type.as.Iterate.impl.Next.type.fc7 = struct_value () [concrete]
// CHECK:STDOUT:   %Iterate.facet.4bb: %Iterate.type = facet_value %Range, (%Iterate.impl_witness.05a) [concrete]
// CHECK:STDOUT:   %facet_value.e16: %Iterate_where.type.0a6 = facet_value %Range, (%Iterate.impl_witness.05a) [concrete]
// CHECK:STDOUT:   %r.patt.b9f: %pattern_type.23f = ref_binding_pattern r [concrete]
// CHECK:STDOUT:   %r.param_patt.466: %pattern_type.23f = var_param_pattern %r.patt.b9f [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.a6aa: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.4bb) [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.291: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.4bb) [concrete]
// CHECK:STDOUT:   %.3df: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.a6aa, %Iterate.facet.4bb [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.specific_fn: <specific function> = specific_function %T.as_type.as.Iterate.impl.NewCursor.a0c, @T.as_type.as.Iterate.impl.NewCursor(%facet_value.8c5) [concrete]
// CHECK:STDOUT:   %ptr.b18: type = ptr_type %tuple.type.9ef [concrete]
// CHECK:STDOUT:   %.d81: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.291, %Iterate.facet.4bb [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.specific_fn: <specific function> = specific_function %T.as_type.as.Iterate.impl.Next.d69, @T.as_type.as.Iterate.impl.Next(%facet_value.8c5) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.7c3: type = fn_type @Destroy.WithSelf.SelfDestruct, @Destroy.WithSelf(%Destroy.facet.30e) [concrete]
// CHECK:STDOUT:   %.c40: type = fn_type_with_self_type %Destroy.WithSelf.SelfDestruct.type.7c3, %Destroy.facet.30e [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.specific_fn.69a: <specific function> = specific_function %Destroy.WithSelf.SelfDestruct.6f1, @Destroy.WithSelf.SelfDestruct(%tuple.type.facet) [concrete]
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
// CHECK:STDOUT:   %Range.decl: type = class_decl @Range [concrete = constants.%Range] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @TestIterate(%R.loc25_17.2: %Iterate_where.type.0a6) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Iterate.lookup_impl_witness: <witness> = lookup_impl_witness %R.loc25_17.1, @Iterate [symbolic = %Iterate.lookup_impl_witness (constants.%Iterate.lookup_impl_witness.9e6)]
// CHECK:STDOUT:   %Iterate.facet.loc27_37.5: %Iterate.type = facet_value %R.as_type.loc25_83.1, (%Iterate.lookup_impl_witness) [symbolic = %Iterate.facet.loc27_37.5 (constants.%Iterate.facet.7a0)]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.loc27_37.5) [symbolic = %Iterate.WithSelf.NewCursor.type (constants.%Iterate.WithSelf.NewCursor.type.a6a5)]
// CHECK:STDOUT:   %.loc27_37.26: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type, %Iterate.facet.loc27_37.5 [symbolic = %.loc27_37.26 (constants.%.635)]
// CHECK:STDOUT:   %impl.elem2.loc27_37.3: @TestIterate.%.loc27_37.26 (%.635) = impl_witness_access %Iterate.lookup_impl_witness, element2 [symbolic = %impl.elem2.loc27_37.3 (constants.%impl.elem2.086)]
// CHECK:STDOUT:   %specific_impl_fn.loc27_37.4: <specific function> = specific_impl_function %impl.elem2.loc27_37.3, @Iterate.WithSelf.NewCursor(%Iterate.facet.loc27_37.5) [symbolic = %specific_impl_fn.loc27_37.4 (constants.%specific_impl_fn.8ac)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %ptr: type = ptr_type %as_type [symbolic = %ptr (constants.%ptr.6f8)]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.loc27_37.5) [symbolic = %Iterate.WithSelf.Next.type (constants.%Iterate.WithSelf.Next.type.a4a)]
// CHECK:STDOUT:   %.loc27_37.27: type = fn_type_with_self_type %Iterate.WithSelf.Next.type, %Iterate.facet.loc27_37.5 [symbolic = %.loc27_37.27 (constants.%.b08)]
// CHECK:STDOUT:   %impl.elem3.loc27_37.2: @TestIterate.%.loc27_37.27 (%.b08) = impl_witness_access %Iterate.lookup_impl_witness, element3 [symbolic = %impl.elem3.loc27_37.2 (constants.%impl.elem3.1c7)]
// CHECK:STDOUT:   %specific_impl_fn.loc27_37.5: <specific function> = specific_impl_function %impl.elem3.loc27_37.2, @Iterate.WithSelf.Next(%Iterate.facet.loc27_37.5) [symbolic = %specific_impl_fn.loc27_37.5 (constants.%specific_impl_fn.02d)]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type: type = fn_type @Destroy.WithSelf.SelfDestruct, @Destroy.WithSelf(%impl.elem1) [symbolic = %Destroy.WithSelf.SelfDestruct.type (constants.%Destroy.WithSelf.SelfDestruct.type.0c5)]
// CHECK:STDOUT:   %.loc27_37.28: type = fn_type_with_self_type %Destroy.WithSelf.SelfDestruct.type, %impl.elem1 [symbolic = %.loc27_37.28 (constants.%.79a)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %impl.elem1, @Destroy [symbolic = %Destroy.lookup_impl_witness (constants.%Destroy.lookup_impl_witness.1e0)]
// CHECK:STDOUT:   %impl.elem2.loc27_37.4: @TestIterate.%.loc27_37.28 (%.79a) = impl_witness_access %Destroy.lookup_impl_witness, element2 [symbolic = %impl.elem2.loc27_37.4 (constants.%impl.elem2.084)]
// CHECK:STDOUT:   %specific_impl_fn.loc27_37.6: <specific function> = specific_impl_function %impl.elem2.loc27_37.4, @Destroy.WithSelf.SelfDestruct(%impl.elem1) [symbolic = %specific_impl_fn.loc27_37.6 (constants.%specific_impl_fn.1a4)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%r.param: ref @TestIterate.%R.as_type.loc25_83.1 (%R.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %_.patt: %pattern_type.804 = value_binding_pattern _ [concrete = constants.%_.patt.a10]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %r.ref: ref @TestIterate.%R.as_type.loc25_83.1 (%R.as_type) = name_ref r, %r
// CHECK:STDOUT:     %impl.elem2.loc27_37.1: @TestIterate.%.loc27_37.26 (%.635) = impl_witness_access constants.%Iterate.lookup_impl_witness.9e6, element2 [symbolic = %impl.elem2.loc27_37.3 (constants.%impl.elem2.086)]
// CHECK:STDOUT:     %bound_method.loc27_37.1: <bound method> = bound_method %r.ref, %impl.elem2.loc27_37.1
// CHECK:STDOUT:     %Iterate.facet.loc27_37.1: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.9e6) [symbolic = %Iterate.facet.loc27_37.5 (constants.%Iterate.facet.7a0)]
// CHECK:STDOUT:     %.loc27_37.1: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc27_37.1 [symbolic = %Iterate.facet.loc27_37.5 (constants.%Iterate.facet.7a0)]
// CHECK:STDOUT:     %Iterate.facet.loc27_37.2: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.9e6) [symbolic = %Iterate.facet.loc27_37.5 (constants.%Iterate.facet.7a0)]
// CHECK:STDOUT:     %.loc27_37.2: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc27_37.2 [symbolic = %Iterate.facet.loc27_37.5 (constants.%Iterate.facet.7a0)]
// CHECK:STDOUT:     %specific_impl_fn.loc27_37.1: <specific function> = specific_impl_function %impl.elem2.loc27_37.1, @Iterate.WithSelf.NewCursor(constants.%Iterate.facet.7a0) [symbolic = %specific_impl_fn.loc27_37.4 (constants.%specific_impl_fn.8ac)]
// CHECK:STDOUT:     %bound_method.loc27_37.2: <bound method> = bound_method %r.ref, %specific_impl_fn.loc27_37.1
// CHECK:STDOUT:     %var: ref @TestIterate.%as_type (%as_type.f7d) = var_storage invalid
// CHECK:STDOUT:     %.loc27_36.1: @TestIterate.%R.as_type.loc25_83.1 (%R.as_type) = acquire_value %r.ref
// CHECK:STDOUT:     %Iterate.WithSelf.NewCursor.call: init @TestIterate.%as_type (%as_type.f7d) to %var = call %bound_method.loc27_37.2(%.loc27_36.1)
// CHECK:STDOUT:     assign %var, %Iterate.WithSelf.NewCursor.call
// CHECK:STDOUT:     br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.next:
// CHECK:STDOUT:     %addr: @TestIterate.%ptr (%ptr.6f8) = addr_of %var
// CHECK:STDOUT:     %impl.elem3.loc27_37.1: @TestIterate.%.loc27_37.27 (%.b08) = impl_witness_access constants.%Iterate.lookup_impl_witness.9e6, element3 [symbolic = %impl.elem3.loc27_37.2 (constants.%impl.elem3.1c7)]
// CHECK:STDOUT:     %bound_method.loc27_37.3: <bound method> = bound_method %r.ref, %impl.elem3.loc27_37.1
// CHECK:STDOUT:     %Iterate.facet.loc27_37.3: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.9e6) [symbolic = %Iterate.facet.loc27_37.5 (constants.%Iterate.facet.7a0)]
// CHECK:STDOUT:     %.loc27_37.3: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc27_37.3 [symbolic = %Iterate.facet.loc27_37.5 (constants.%Iterate.facet.7a0)]
// CHECK:STDOUT:     %Iterate.facet.loc27_37.4: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.9e6) [symbolic = %Iterate.facet.loc27_37.5 (constants.%Iterate.facet.7a0)]
// CHECK:STDOUT:     %.loc27_37.4: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc27_37.4 [symbolic = %Iterate.facet.loc27_37.5 (constants.%Iterate.facet.7a0)]
// CHECK:STDOUT:     %.loc27_37.5: %Destroy.type = converted constants.%as_type.f7d, constants.%impl.elem1.35e [symbolic = %impl.elem1 (constants.%impl.elem1.35e)]
// CHECK:STDOUT:     %.loc27_37.6: %Destroy.type = converted constants.%as_type.f7d, constants.%impl.elem1.35e [symbolic = %impl.elem1 (constants.%impl.elem1.35e)]
// CHECK:STDOUT:     %specific_impl_fn.loc27_37.2: <specific function> = specific_impl_function %impl.elem3.loc27_37.1, @Iterate.WithSelf.Next(constants.%Iterate.facet.7a0) [symbolic = %specific_impl_fn.loc27_37.5 (constants.%specific_impl_fn.02d)]
// CHECK:STDOUT:     %bound_method.loc27_37.4: <bound method> = bound_method %r.ref, %specific_impl_fn.loc27_37.2
// CHECK:STDOUT:     %.loc27_37.7: ref %Optional.f35 = temporary_storage
// CHECK:STDOUT:     %.loc27_36.2: @TestIterate.%R.as_type.loc25_83.1 (%R.as_type) = acquire_value %r.ref
// CHECK:STDOUT:     %Iterate.WithSelf.Next.call: init %Optional.f35 to %.loc27_37.7 = call %bound_method.loc27_37.4(%.loc27_36.2, %addr)
// CHECK:STDOUT:     %.loc27_37.8: ref %Optional.f35 = temporary %.loc27_37.7, %Iterate.WithSelf.Next.call
// CHECK:STDOUT:     %.loc27_37.9: %Optional.HasValue.type.6d2 = specific_constant imports.%Core.import_ref.bb6, @Optional(constants.%OptionalStorage.facet.0f7) [concrete = constants.%Optional.HasValue.304]
// CHECK:STDOUT:     %HasValue.ref: %Optional.HasValue.type.6d2 = name_ref HasValue, %.loc27_37.9 [concrete = constants.%Optional.HasValue.304]
// CHECK:STDOUT:     %Optional.HasValue.bound: <bound method> = bound_method %.loc27_37.8, %HasValue.ref
// CHECK:STDOUT:     %Optional.HasValue.specific_fn: <specific function> = specific_function %HasValue.ref, @Optional.HasValue(constants.%OptionalStorage.facet.0f7) [concrete = constants.%Optional.HasValue.specific_fn.855]
// CHECK:STDOUT:     %bound_method.loc27_37.5: <bound method> = bound_method %.loc27_37.8, %Optional.HasValue.specific_fn
// CHECK:STDOUT:     %.loc27_37.10: %Optional.f35 = acquire_value %.loc27_37.8
// CHECK:STDOUT:     %Optional.HasValue.call: init bool = call %bound_method.loc27_37.5(%.loc27_37.10)
// CHECK:STDOUT:     %.loc27_37.11: bool = value_of_initializer %Optional.HasValue.call
// CHECK:STDOUT:     %.loc27_37.12: bool = converted %Optional.HasValue.call, %.loc27_37.11
// CHECK:STDOUT:     if %.loc27_37.12 br !for.body else br !for.done
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.body:
// CHECK:STDOUT:     %.loc27_37.13: %Optional.Get.type.a45 = specific_constant imports.%Core.import_ref.dab, @Optional(constants.%OptionalStorage.facet.0f7) [concrete = constants.%Optional.Get.b4d]
// CHECK:STDOUT:     %Get.ref: %Optional.Get.type.a45 = name_ref Get, %.loc27_37.13 [concrete = constants.%Optional.Get.b4d]
// CHECK:STDOUT:     %Optional.Get.bound: <bound method> = bound_method %.loc27_37.8, %Get.ref
// CHECK:STDOUT:     %Optional.Get.specific_fn: <specific function> = specific_function %Get.ref, @Optional.Get(constants.%OptionalStorage.facet.0f7) [concrete = constants.%Optional.Get.specific_fn.7d0]
// CHECK:STDOUT:     %bound_method.loc27_37.6: <bound method> = bound_method %.loc27_37.8, %Optional.Get.specific_fn
// CHECK:STDOUT:     %.loc27_37.14: ref %OverloadValueType = temporary_storage
// CHECK:STDOUT:     %.loc27_37.15: %Optional.f35 = acquire_value %.loc27_37.8
// CHECK:STDOUT:     %Optional.Get.call: init %OverloadValueType to %.loc27_37.14 = call %bound_method.loc27_37.6(%.loc27_37.15)
// CHECK:STDOUT:     %.loc27_37.16: ref %OverloadValueType = temporary %.loc27_37.14, %Optional.Get.call
// CHECK:STDOUT:     %.loc27_37.17: %OverloadValueType = acquire_value %.loc27_37.16
// CHECK:STDOUT:     %.loc27_14: type = splice_block %OverloadValueType.ref.loc27 [concrete = constants.%OverloadValueType] {
// CHECK:STDOUT:       %Cpp.ref.loc27: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:       %OverloadValueType.ref.loc27: type = name_ref OverloadValueType, imports.%OverloadValueType.decl [concrete = constants.%OverloadValueType]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %_: %OverloadValueType = wrapper_binding _, %.loc27_37.17
// CHECK:STDOUT:     %Destroy.WithSelf.SelfDestruct.bound.loc27_37.1: <bound method> = bound_method %.loc27_37.16, constants.%Destroy.WithSelf.SelfDestruct.d92
// CHECK:STDOUT:     %Destroy.facet.loc27_37.1: %Destroy.type = facet_value constants.%OverloadValueType, (constants.%custom_witness.ae2) [concrete = constants.%Destroy.facet.909]
// CHECK:STDOUT:     %.loc27_37.18: %Destroy.type = converted constants.%OverloadValueType, %Destroy.facet.loc27_37.1 [concrete = constants.%Destroy.facet.909]
// CHECK:STDOUT:     %Destroy.facet.loc27_37.2: %Destroy.type = facet_value constants.%OverloadValueType, (constants.%custom_witness.ae2) [concrete = constants.%Destroy.facet.909]
// CHECK:STDOUT:     %.loc27_37.19: %Destroy.type = converted constants.%OverloadValueType, %Destroy.facet.loc27_37.2 [concrete = constants.%Destroy.facet.909]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %bound_method.loc27_37.7: <bound method> = bound_method %.loc27_37.16, %Destroy.WithSelf.SelfDestruct.specific_fn.1
// CHECK:STDOUT:     %Destroy.WithSelf.SelfDestruct.call.loc27_37.1: init %empty_tuple.type = call %bound_method.loc27_37.7(%.loc27_37.16)
// CHECK:STDOUT:     %Destroy.WithSelf.SelfDestruct.bound.loc27_37.2: <bound method> = bound_method %.loc27_37.8, constants.%Destroy.WithSelf.SelfDestruct.e40
// CHECK:STDOUT:     %Destroy.facet.loc27_37.3: %Destroy.type = facet_value constants.%Optional.f35, (constants.%custom_witness.5b3) [concrete = constants.%Destroy.facet.a53]
// CHECK:STDOUT:     %.loc27_37.20: %Destroy.type = converted constants.%Optional.f35, %Destroy.facet.loc27_37.3 [concrete = constants.%Destroy.facet.a53]
// CHECK:STDOUT:     %Destroy.facet.loc27_37.4: %Destroy.type = facet_value constants.%Optional.f35, (constants.%custom_witness.5b3) [concrete = constants.%Destroy.facet.a53]
// CHECK:STDOUT:     %.loc27_37.21: %Destroy.type = converted constants.%Optional.f35, %Destroy.facet.loc27_37.4 [concrete = constants.%Destroy.facet.a53]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %bound_method.loc27_37.8: <bound method> = bound_method %.loc27_37.8, %Destroy.WithSelf.SelfDestruct.specific_fn.2
// CHECK:STDOUT:     %Destroy.WithSelf.SelfDestruct.call.loc27_37.2: init %empty_tuple.type = call %bound_method.loc27_37.8(%.loc27_37.8)
// CHECK:STDOUT:     br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.done:
// CHECK:STDOUT:     %Destroy.WithSelf.SelfDestruct.bound.loc27_37.3: <bound method> = bound_method %.loc27_37.8, constants.%Destroy.WithSelf.SelfDestruct.e40
// CHECK:STDOUT:     %Destroy.facet.loc27_37.5: %Destroy.type = facet_value constants.%Optional.f35, (constants.%custom_witness.5b3) [concrete = constants.%Destroy.facet.a53]
// CHECK:STDOUT:     %.loc27_37.22: %Destroy.type = converted constants.%Optional.f35, %Destroy.facet.loc27_37.5 [concrete = constants.%Destroy.facet.a53]
// CHECK:STDOUT:     %Destroy.facet.loc27_37.6: %Destroy.type = facet_value constants.%Optional.f35, (constants.%custom_witness.5b3) [concrete = constants.%Destroy.facet.a53]
// CHECK:STDOUT:     %.loc27_37.23: %Destroy.type = converted constants.%Optional.f35, %Destroy.facet.loc27_37.6 [concrete = constants.%Destroy.facet.a53]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %bound_method.loc27_37.9: <bound method> = bound_method %.loc27_37.8, %Destroy.WithSelf.SelfDestruct.specific_fn.3
// CHECK:STDOUT:     %Destroy.WithSelf.SelfDestruct.call.loc27_37.3: init %empty_tuple.type = call %bound_method.loc27_37.9(%.loc27_37.8)
// CHECK:STDOUT:     %impl.elem2.loc27_37.2: @TestIterate.%.loc27_37.28 (%.79a) = impl_witness_access constants.%Destroy.lookup_impl_witness.1e0, element2 [symbolic = %impl.elem2.loc27_37.4 (constants.%impl.elem2.084)]
// CHECK:STDOUT:     %bound_method.loc27_37.10: <bound method> = bound_method %var, %impl.elem2.loc27_37.2
// CHECK:STDOUT:     %.loc27_37.24: %Destroy.type = converted constants.%as_type.f7d, constants.%impl.elem1.35e [symbolic = %impl.elem1 (constants.%impl.elem1.35e)]
// CHECK:STDOUT:     %.loc27_37.25: %Destroy.type = converted constants.%as_type.f7d, constants.%impl.elem1.35e [symbolic = %impl.elem1 (constants.%impl.elem1.35e)]
// CHECK:STDOUT:     %specific_impl_fn.loc27_37.3: <specific function> = specific_impl_function %impl.elem2.loc27_37.2, @Destroy.WithSelf.SelfDestruct(constants.%impl.elem1.35e) [symbolic = %specific_impl_fn.loc27_37.6 (constants.%specific_impl_fn.1a4)]
// CHECK:STDOUT:     %bound_method.loc27_37.11: <bound method> = bound_method %var, %specific_impl_fn.loc27_37.3
// CHECK:STDOUT:     %Destroy.WithSelf.SelfDestruct.call.loc27_37.4: init %empty_tuple.type = call %bound_method.loc27_37.11(%var)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc27_37.1(%self.param: ref %.8a2) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc27_37.2(%self.param: ref %MaybeUnformed.275) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc27_37.3(%self.param: ref bool) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc27_37.4(%self.param: ref %struct_type.value.has_value.827) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc27_37.5(%self.param: ref %DefaultOptionalStorage.4a5) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc27_37.6(%self.param: ref %Optional.f35) {
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
// CHECK:STDOUT:   %impl.elem2: %.3df = impl_witness_access constants.%Iterate.impl_witness.05a, element2 [concrete = constants.%T.as_type.as.Iterate.impl.NewCursor.a0c]
// CHECK:STDOUT:   %bound_method.loc37_37.1: <bound method> = bound_method %r.ref, %impl.elem2
// CHECK:STDOUT:   %facet_value.loc37_37.1: %CppRange_where.type.d682d6.1 = facet_value constants.%Range, (constants.%custom_witness.ae2, constants.%custom_witness.d0da43.1, constants.%custom_witness.090, constants.%custom_witness.e0e, constants.%custom_witness.6bd, constants.%custom_witness.716, constants.%custom_witness.617) [concrete = constants.%facet_value.8c5]
// CHECK:STDOUT:   %.loc37_37.1: %CppRange_where.type.d682d6.1 = converted constants.%Range, %facet_value.loc37_37.1 [concrete = constants.%facet_value.8c5]
// CHECK:STDOUT:   %facet_value.loc37_37.2: %CppRange_where.type.d682d6.1 = facet_value constants.%Range, (constants.%custom_witness.ae2, constants.%custom_witness.d0da43.1, constants.%custom_witness.090, constants.%custom_witness.e0e, constants.%custom_witness.6bd, constants.%custom_witness.716, constants.%custom_witness.617) [concrete = constants.%facet_value.8c5]
// CHECK:STDOUT:   %.loc37_37.2: %CppRange_where.type.d682d6.1 = converted constants.%Range, %facet_value.loc37_37.2 [concrete = constants.%facet_value.8c5]
// CHECK:STDOUT:   %specific_fn.loc37_37.1: <specific function> = specific_function %impl.elem2, @T.as_type.as.Iterate.impl.NewCursor(constants.%facet_value.8c5) [concrete = constants.%T.as_type.as.Iterate.impl.NewCursor.specific_fn]
// CHECK:STDOUT:   %bound_method.loc37_37.2: <bound method> = bound_method %r.ref, %specific_fn.loc37_37.1
// CHECK:STDOUT:   %var: ref %tuple.type.9ef = var_storage invalid
// CHECK:STDOUT:   %.loc37_36.1: %Range = acquire_value %r.ref
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.call: init %tuple.type.9ef to %var = call %bound_method.loc37_37.2(%.loc37_36.1)
// CHECK:STDOUT:   assign %var, %T.as_type.as.Iterate.impl.NewCursor.call
// CHECK:STDOUT:   br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT: !for.next:
// CHECK:STDOUT:   %addr: %ptr.b18 = addr_of %var
// CHECK:STDOUT:   %impl.elem3: %.d81 = impl_witness_access constants.%Iterate.impl_witness.05a, element3 [concrete = constants.%T.as_type.as.Iterate.impl.Next.d69]
// CHECK:STDOUT:   %bound_method.loc37_37.3: <bound method> = bound_method %r.ref, %impl.elem3
// CHECK:STDOUT:   %facet_value.loc37_37.3: %CppRange_where.type.d682d6.1 = facet_value constants.%Range, (constants.%custom_witness.ae2, constants.%custom_witness.d0da43.1, constants.%custom_witness.090, constants.%custom_witness.e0e, constants.%custom_witness.6bd, constants.%custom_witness.716, constants.%custom_witness.617) [concrete = constants.%facet_value.8c5]
// CHECK:STDOUT:   %.loc37_37.3: %CppRange_where.type.d682d6.1 = converted constants.%Range, %facet_value.loc37_37.3 [concrete = constants.%facet_value.8c5]
// CHECK:STDOUT:   %facet_value.loc37_37.4: %CppRange_where.type.d682d6.1 = facet_value constants.%Range, (constants.%custom_witness.ae2, constants.%custom_witness.d0da43.1, constants.%custom_witness.090, constants.%custom_witness.e0e, constants.%custom_witness.6bd, constants.%custom_witness.716, constants.%custom_witness.617) [concrete = constants.%facet_value.8c5]
// CHECK:STDOUT:   %.loc37_37.4: %CppRange_where.type.d682d6.1 = converted constants.%Range, %facet_value.loc37_37.4 [concrete = constants.%facet_value.8c5]
// CHECK:STDOUT:   %specific_fn.loc37_37.2: <specific function> = specific_function %impl.elem3, @T.as_type.as.Iterate.impl.Next(constants.%facet_value.8c5) [concrete = constants.%T.as_type.as.Iterate.impl.Next.specific_fn]
// CHECK:STDOUT:   %bound_method.loc37_37.4: <bound method> = bound_method %r.ref, %specific_fn.loc37_37.2
// CHECK:STDOUT:   %.loc37_37.5: ref %Optional.f35 = temporary_storage
// CHECK:STDOUT:   %.loc37_36.2: %Range = acquire_value %r.ref
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.call: init %Optional.f35 to %.loc37_37.5 = call %bound_method.loc37_37.4(%.loc37_36.2, %addr)
// CHECK:STDOUT:   %.loc37_37.6: ref %Optional.f35 = temporary %.loc37_37.5, %T.as_type.as.Iterate.impl.Next.call
// CHECK:STDOUT:   %.loc37_37.7: %Optional.HasValue.type.6d2 = specific_constant imports.%Core.import_ref.bb6, @Optional(constants.%OptionalStorage.facet.0f7) [concrete = constants.%Optional.HasValue.304]
// CHECK:STDOUT:   %HasValue.ref: %Optional.HasValue.type.6d2 = name_ref HasValue, %.loc37_37.7 [concrete = constants.%Optional.HasValue.304]
// CHECK:STDOUT:   %Optional.HasValue.bound: <bound method> = bound_method %.loc37_37.6, %HasValue.ref
// CHECK:STDOUT:   %Optional.HasValue.specific_fn: <specific function> = specific_function %HasValue.ref, @Optional.HasValue(constants.%OptionalStorage.facet.0f7) [concrete = constants.%Optional.HasValue.specific_fn.855]
// CHECK:STDOUT:   %bound_method.loc37_37.5: <bound method> = bound_method %.loc37_37.6, %Optional.HasValue.specific_fn
// CHECK:STDOUT:   %.loc37_37.8: %Optional.f35 = acquire_value %.loc37_37.6
// CHECK:STDOUT:   %Optional.HasValue.call: init bool = call %bound_method.loc37_37.5(%.loc37_37.8)
// CHECK:STDOUT:   %.loc37_37.9: bool = value_of_initializer %Optional.HasValue.call
// CHECK:STDOUT:   %.loc37_37.10: bool = converted %Optional.HasValue.call, %.loc37_37.9
// CHECK:STDOUT:   if %.loc37_37.10 br !for.body else br !for.done
// CHECK:STDOUT:
// CHECK:STDOUT: !for.body:
// CHECK:STDOUT:   %.loc37_37.11: %Optional.Get.type.a45 = specific_constant imports.%Core.import_ref.dab, @Optional(constants.%OptionalStorage.facet.0f7) [concrete = constants.%Optional.Get.b4d]
// CHECK:STDOUT:   %Get.ref: %Optional.Get.type.a45 = name_ref Get, %.loc37_37.11 [concrete = constants.%Optional.Get.b4d]
// CHECK:STDOUT:   %Optional.Get.bound: <bound method> = bound_method %.loc37_37.6, %Get.ref
// CHECK:STDOUT:   %Optional.Get.specific_fn: <specific function> = specific_function %Get.ref, @Optional.Get(constants.%OptionalStorage.facet.0f7) [concrete = constants.%Optional.Get.specific_fn.7d0]
// CHECK:STDOUT:   %bound_method.loc37_37.6: <bound method> = bound_method %.loc37_37.6, %Optional.Get.specific_fn
// CHECK:STDOUT:   %.loc37_37.12: ref %OverloadValueType = temporary_storage
// CHECK:STDOUT:   %.loc37_37.13: %Optional.f35 = acquire_value %.loc37_37.6
// CHECK:STDOUT:   %Optional.Get.call: init %OverloadValueType to %.loc37_37.12 = call %bound_method.loc37_37.6(%.loc37_37.13)
// CHECK:STDOUT:   %.loc37_37.14: ref %OverloadValueType = temporary %.loc37_37.12, %Optional.Get.call
// CHECK:STDOUT:   %.loc37_37.15: %OverloadValueType = acquire_value %.loc37_37.14
// CHECK:STDOUT:   %.loc37_14: type = splice_block %OverloadValueType.ref [concrete = constants.%OverloadValueType] {
// CHECK:STDOUT:     %Cpp.ref.loc37: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %OverloadValueType.ref: type = name_ref OverloadValueType, imports.%OverloadValueType.decl [concrete = constants.%OverloadValueType]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %_: %OverloadValueType = wrapper_binding _, %.loc37_37.15
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc37_37.1: <bound method> = bound_method %.loc37_37.14, constants.%Destroy.WithSelf.SelfDestruct.d92
// CHECK:STDOUT:   %Destroy.facet.loc37_37.1: %Destroy.type = facet_value constants.%OverloadValueType, (constants.%custom_witness.ae2) [concrete = constants.%Destroy.facet.909]
// CHECK:STDOUT:   %.loc37_37.16: %Destroy.type = converted constants.%OverloadValueType, %Destroy.facet.loc37_37.1 [concrete = constants.%Destroy.facet.909]
// CHECK:STDOUT:   %Destroy.facet.loc37_37.2: %Destroy.type = facet_value constants.%OverloadValueType, (constants.%custom_witness.ae2) [concrete = constants.%Destroy.facet.909]
// CHECK:STDOUT:   %.loc37_37.17: %Destroy.type = converted constants.%OverloadValueType, %Destroy.facet.loc37_37.2 [concrete = constants.%Destroy.facet.909]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc37_37.7: <bound method> = bound_method %.loc37_37.14, %Destroy.WithSelf.SelfDestruct.specific_fn.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc37_37.1: init %empty_tuple.type = call %bound_method.loc37_37.7(%.loc37_37.14)
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc37_37.2: <bound method> = bound_method %.loc37_37.6, constants.%Destroy.WithSelf.SelfDestruct.e40
// CHECK:STDOUT:   %Destroy.facet.loc37_37.3: %Destroy.type = facet_value constants.%Optional.f35, (constants.%custom_witness.5b3) [concrete = constants.%Destroy.facet.a53]
// CHECK:STDOUT:   %.loc37_37.18: %Destroy.type = converted constants.%Optional.f35, %Destroy.facet.loc37_37.3 [concrete = constants.%Destroy.facet.a53]
// CHECK:STDOUT:   %Destroy.facet.loc37_37.4: %Destroy.type = facet_value constants.%Optional.f35, (constants.%custom_witness.5b3) [concrete = constants.%Destroy.facet.a53]
// CHECK:STDOUT:   %.loc37_37.19: %Destroy.type = converted constants.%Optional.f35, %Destroy.facet.loc37_37.4 [concrete = constants.%Destroy.facet.a53]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc37_37.8: <bound method> = bound_method %.loc37_37.6, %Destroy.WithSelf.SelfDestruct.specific_fn.2
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc37_37.2: init %empty_tuple.type = call %bound_method.loc37_37.8(%.loc37_37.6)
// CHECK:STDOUT:   br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT: !for.done:
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc37_37.3: <bound method> = bound_method %.loc37_37.6, constants.%Destroy.WithSelf.SelfDestruct.e40
// CHECK:STDOUT:   %Destroy.facet.loc37_37.5: %Destroy.type = facet_value constants.%Optional.f35, (constants.%custom_witness.5b3) [concrete = constants.%Destroy.facet.a53]
// CHECK:STDOUT:   %.loc37_37.20: %Destroy.type = converted constants.%Optional.f35, %Destroy.facet.loc37_37.5 [concrete = constants.%Destroy.facet.a53]
// CHECK:STDOUT:   %Destroy.facet.loc37_37.6: %Destroy.type = facet_value constants.%Optional.f35, (constants.%custom_witness.5b3) [concrete = constants.%Destroy.facet.a53]
// CHECK:STDOUT:   %.loc37_37.21: %Destroy.type = converted constants.%Optional.f35, %Destroy.facet.loc37_37.6 [concrete = constants.%Destroy.facet.a53]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc37_37.9: <bound method> = bound_method %.loc37_37.6, %Destroy.WithSelf.SelfDestruct.specific_fn.3
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc37_37.3: init %empty_tuple.type = call %bound_method.loc37_37.9(%.loc37_37.6)
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc37_37.4: <bound method> = bound_method %var, constants.%Destroy.WithSelf.SelfDestruct.6f1
// CHECK:STDOUT:   %Destroy.facet.loc37_37.7: %Destroy.type = facet_value constants.%tuple.type.9ef, (constants.%custom_witness.90f) [concrete = constants.%Destroy.facet.30e]
// CHECK:STDOUT:   %.loc37_37.22: %Destroy.type = converted constants.%tuple.type.9ef, %Destroy.facet.loc37_37.7 [concrete = constants.%Destroy.facet.30e]
// CHECK:STDOUT:   %Destroy.facet.loc37_37.8: %Destroy.type = facet_value constants.%tuple.type.9ef, (constants.%custom_witness.90f) [concrete = constants.%Destroy.facet.30e]
// CHECK:STDOUT:   %.loc37_37.23: %Destroy.type = converted constants.%tuple.type.9ef, %Destroy.facet.loc37_37.8 [concrete = constants.%Destroy.facet.30e]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc37_37.10: <bound method> = bound_method %var, %Destroy.WithSelf.SelfDestruct.specific_fn.4
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc37_37.4: init %empty_tuple.type = call %bound_method.loc37_37.10(%var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestIterate(constants.%R) {
// CHECK:STDOUT:   %R.patt.loc25_17.2 => constants.%R.patt
// CHECK:STDOUT:   %R.loc25_17.1 => constants.%R
// CHECK:STDOUT:   %R.as_type.loc25_83.1 => constants.%R.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.2af
// CHECK:STDOUT:   %r.patt.loc25_81.2 => constants.%r.patt.b15
// CHECK:STDOUT:   %r.param_patt.loc25_76.2 => constants.%r.param_patt.206
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestIterate(constants.%facet_value.e16) {
// CHECK:STDOUT:   %R.patt.loc25_17.2 => constants.%R.patt
// CHECK:STDOUT:   %R.loc25_17.1 => constants.%facet_value.e16
// CHECK:STDOUT:   %R.as_type.loc25_83.1 => constants.%Range
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.23f
// CHECK:STDOUT:   %r.patt.loc25_81.2 => constants.%r.patt.b9f
// CHECK:STDOUT:   %r.param_patt.loc25_76.2 => constants.%r.param_patt.466
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc25 => constants.%complete_type.357
// CHECK:STDOUT:   %Iterate.lookup_impl_witness => constants.%Iterate.impl_witness.05a
// CHECK:STDOUT:   %Iterate.facet.loc27_37.5 => constants.%Iterate.facet.4bb
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type => constants.%Iterate.WithSelf.NewCursor.type.a6aa
// CHECK:STDOUT:   %.loc27_37.26 => constants.%.3df
// CHECK:STDOUT:   %impl.elem2.loc27_37.3 => constants.%T.as_type.as.Iterate.impl.NewCursor.a0c
// CHECK:STDOUT:   %specific_impl_fn.loc27_37.4 => constants.%T.as_type.as.Iterate.impl.NewCursor.specific_fn
// CHECK:STDOUT:   %impl.elem1 => constants.%Destroy.facet.30e
// CHECK:STDOUT:   %as_type => constants.%tuple.type.9ef
// CHECK:STDOUT:   %require_complete.1 => constants.%complete_type.e35
// CHECK:STDOUT:   %ptr => constants.%ptr.b18
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type => constants.%Iterate.WithSelf.Next.type.291
// CHECK:STDOUT:   %.loc27_37.27 => constants.%.d81
// CHECK:STDOUT:   %impl.elem3.loc27_37.2 => constants.%T.as_type.as.Iterate.impl.Next.d69
// CHECK:STDOUT:   %specific_impl_fn.loc27_37.5 => constants.%T.as_type.as.Iterate.impl.Next.specific_fn
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type => constants.%Destroy.WithSelf.SelfDestruct.type.7c3
// CHECK:STDOUT:   %.loc27_37.28 => constants.%.c40
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.90f
// CHECK:STDOUT:   %impl.elem2.loc27_37.4 => constants.%Destroy.WithSelf.SelfDestruct.6f1
// CHECK:STDOUT:   %specific_impl_fn.loc27_37.6 => constants.%Destroy.WithSelf.SelfDestruct.specific_fn.69a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- adl_return_same_types.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
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
// CHECK:STDOUT:   %Float.type.facet: %type = facet_value %f64.dc1, () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.107: type = fn_type @Destroy.WithSelf.SubobjectDestroy, @Destroy.WithSelf(%Float.type.facet) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.2d5: %Destroy.WithSelf.SubobjectDestroy.type.107 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.c4e: type = fn_type @Destroy.WithSelf.SelfDestruct, @Destroy.WithSelf(%Float.type.facet) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.a03: %Destroy.WithSelf.SelfDestruct.type.c4e = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.d89: <witness> = custom_witness (%Destroy.Op.1a2547.3, %Destroy.WithSelf.SubobjectDestroy.2d5, %Destroy.WithSelf.SelfDestruct.a03), @Destroy [concrete]
// CHECK:STDOUT:   %Float.as.Copy.impl.Op.type.ff3: type = fn_type @Float.as.Copy.impl.Op, @Float.as.Copy.impl(%N.fe9) [symbolic]
// CHECK:STDOUT:   %Float.as.Copy.impl.Op.d93: %Float.as.Copy.impl.Op.type.ff3 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.0d6: <witness> = impl_witness imports.%Copy.impl_witness_table.8dd, @Float.as.Copy.impl(%int_64) [concrete]
// CHECK:STDOUT:   %facet_value.915: %facet_type.f48 = facet_value %f64.dc1, (%custom_witness.d89, %Copy.impl_witness.0d6) [concrete]
// CHECK:STDOUT:   %Iterate.lookup_impl_witness.1d8: <witness> = lookup_impl_witness %.Self.500, @Iterate [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.294: %facet_type.f48 = impl_witness_access %Iterate.lookup_impl_witness.1d8, element0 [symbolic_self]
// CHECK:STDOUT:   %Iterate_where.type.42d: type = facet_type <@Iterate where %impl.elem0.294 = %facet_value.915> [concrete]
// CHECK:STDOUT:   %pattern_type.879: type = pattern_type %Iterate_where.type.42d [concrete]
// CHECK:STDOUT:   %R.patt: %pattern_type.879 = symbolic_binding_pattern R, 0 [symbolic]
// CHECK:STDOUT:   %R: %Iterate_where.type.42d = symbolic_binding R, 0 [symbolic]
// CHECK:STDOUT:   %R.as_type: type = facet_access_type %R [symbolic]
// CHECK:STDOUT:   %pattern_type.609: type = pattern_type %R.as_type [symbolic]
// CHECK:STDOUT:   %r.param_patt.09b: %pattern_type.609 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %r.patt.772: %pattern_type.609 = wrapper_binding_pattern r, %r.param_patt.09b [symbolic]
// CHECK:STDOUT:   %i.patt: %pattern_type.fb7 = value_binding_pattern i [concrete]
// CHECK:STDOUT:   %Iterate.lookup_impl_witness.a65: <witness> = lookup_impl_witness %R, @Iterate [symbolic]
// CHECK:STDOUT:   %Iterate.facet.d84: %Iterate.type = facet_value %R.as_type, (%Iterate.lookup_impl_witness.a65) [symbolic]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.906: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.d84) [symbolic]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.e19: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.d84) [symbolic]
// CHECK:STDOUT:   %.d8f: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.906, %Iterate.facet.d84 [symbolic]
// CHECK:STDOUT:   %impl.elem2.f1e: %.d8f = impl_witness_access %Iterate.lookup_impl_witness.a65, element2 [symbolic]
// CHECK:STDOUT:   %impl.elem1.d78: %Destroy.type = impl_witness_access %Iterate.lookup_impl_witness.a65, element1 [symbolic]
// CHECK:STDOUT:   %as_type.74e: type = facet_access_type %impl.elem1.d78 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.e73: <specific function> = specific_impl_function %impl.elem2.f1e, @Iterate.WithSelf.NewCursor(%Iterate.facet.d84) [symbolic]
// CHECK:STDOUT:   %ptr.e00: type = ptr_type %as_type.74e [symbolic]
// CHECK:STDOUT:   %.633: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.e19, %Iterate.facet.d84 [symbolic]
// CHECK:STDOUT:   %impl.elem3.50c: %.633 = impl_witness_access %Iterate.lookup_impl_witness.a65, element3 [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.83b: type = class_type @DefaultOptionalStorage, @DefaultOptionalStorage(%facet_value.915) [concrete]
// CHECK:STDOUT:   %OptionalStorage.impl_witness.3c1: <witness> = impl_witness imports.%OptionalStorage.impl_witness_table.460, @T.as_type.as.OptionalStorage.impl(%facet_value.915) [concrete]
// CHECK:STDOUT:   %OptionalStorage.facet.357: %OptionalStorage.type = facet_value %f64.dc1, (%OptionalStorage.impl_witness.3c1) [concrete]
// CHECK:STDOUT:   %Optional.226: type = class_type @Optional, @Optional(%OptionalStorage.facet.357) [concrete]
// CHECK:STDOUT:   %specific_impl_fn.d94: <specific function> = specific_impl_function %impl.elem3.50c, @Iterate.WithSelf.Next(%Iterate.facet.d84) [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.type.f0d: type = fn_type @Optional.HasValue, @Optional(%OptionalStorage.facet.357) [concrete]
// CHECK:STDOUT:   %Optional.HasValue.eb2: %Optional.HasValue.type.f0d = struct_value () [concrete]
// CHECK:STDOUT:   %Optional.Get.type.d30: type = fn_type @Optional.Get, @Optional(%OptionalStorage.facet.357) [concrete]
// CHECK:STDOUT:   %Optional.Get.617: %Optional.Get.type.d30 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.facet.445: %Destroy.type = facet_value %f64.dc1, (%custom_witness.d89) [concrete]
// CHECK:STDOUT:   %MaybeUnformed.ea6: type = class_type @MaybeUnformed, @MaybeUnformed(%Destroy.facet.445) [concrete]
// CHECK:STDOUT:   %.cc0: type = maybe_unformed_type %f64.dc1 [concrete]
// CHECK:STDOUT:   %struct_type.value.has_value.0fe: type = struct_type {.value: %MaybeUnformed.ea6, .has_value: bool} [concrete]
// CHECK:STDOUT:   %Optional.HasValue.specific_fn.0ad: <specific function> = specific_function %Optional.HasValue.eb2, @Optional.HasValue(%OptionalStorage.facet.357) [concrete]
// CHECK:STDOUT:   %Optional.Get.specific_fn.cde: <specific function> = specific_function %Optional.Get.617, @Optional.Get(%OptionalStorage.facet.357) [concrete]
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
// CHECK:STDOUT:   %Optional.type.facet: %type = facet_value %Optional.226, () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.21d: type = fn_type @Destroy.WithSelf.SubobjectDestroy, @Destroy.WithSelf(%Optional.type.facet) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.f50: %Destroy.WithSelf.SubobjectDestroy.type.21d = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.e02: type = fn_type @Destroy.WithSelf.SelfDestruct, @Destroy.WithSelf(%Optional.type.facet) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.f02: %Destroy.WithSelf.SelfDestruct.type.e02 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.372: <witness> = custom_witness (%Destroy.Op.1a2547.9, %Destroy.WithSelf.SubobjectDestroy.f50, %Destroy.WithSelf.SelfDestruct.f02), @Destroy [concrete]
// CHECK:STDOUT:   %Destroy.facet.5cd: %Destroy.type = facet_value %Optional.226, (%custom_witness.372) [concrete]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.646: <witness> = lookup_impl_witness %impl.elem1.d78, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.e6a: type = fn_type @Destroy.WithSelf.SelfDestruct, @Destroy.WithSelf(%impl.elem1.d78) [symbolic]
// CHECK:STDOUT:   %.9ad: type = fn_type_with_self_type %Destroy.WithSelf.SelfDestruct.type.e6a, %impl.elem1.d78 [symbolic]
// CHECK:STDOUT:   %impl.elem2.206: %.9ad = impl_witness_access %Destroy.lookup_impl_witness.646, element2 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.908: <specific function> = specific_impl_function %impl.elem2.206, @Destroy.WithSelf.SelfDestruct(%impl.elem1.d78) [symbolic]
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
// CHECK:STDOUT:   %Iterator.type.facet: %type = facet_value %Iterator, () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.e6c: type = fn_type @Destroy.WithSelf.SubobjectDestroy, @Destroy.WithSelf(%Iterator.type.facet) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.5eb: %Destroy.WithSelf.SubobjectDestroy.type.e6c = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.dcf: type = fn_type @Destroy.WithSelf.SelfDestruct, @Destroy.WithSelf(%Iterator.type.facet) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.2c2: %Destroy.WithSelf.SelfDestruct.type.dcf = struct_value () [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.ac7e09.2: type = fn_type @Iterator.Op.2 [concrete]
// CHECK:STDOUT:   %Iterator.Op.583fda.2: %Iterator.Op.type.ac7e09.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.6fe: <witness> = custom_witness (%Iterator.Op.583fda.2, %Destroy.WithSelf.SubobjectDestroy.5eb, %Destroy.WithSelf.SelfDestruct.2c2), @Destroy [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.ac7e09.3: type = fn_type @Iterator.Op.3 [concrete]
// CHECK:STDOUT:   %Iterator.Op.583fda.3: %Iterator.Op.type.ac7e09.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.393: <witness> = custom_witness (%Iterator.Op.583fda.3), @Inc [concrete]
// CHECK:STDOUT:   %Equal.type: type = fn_type @Equal [concrete]
// CHECK:STDOUT:   %Equal: %Equal.type = struct_value () [concrete]
// CHECK:STDOUT:   %NotEqual.type: type = fn_type @NotEqual [concrete]
// CHECK:STDOUT:   %NotEqual: %NotEqual.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.821: <witness> = custom_witness (%Equal, %NotEqual), @EqWith, @EqWith(%Iterator) [concrete]
// CHECK:STDOUT:   %facet_value.e04: %CppRange_where.type.d682d6.1 = facet_value %ConstRange, (%custom_witness.d89, %Copy.impl_witness.0d6, %custom_witness.5d5, %custom_witness.6fe, %custom_witness.994, %custom_witness.393, %custom_witness.821) [concrete]
// CHECK:STDOUT:   %tuple.type.005: type = tuple_type (%Iterator, %Iterator) [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.10: type = fn_type @Destroy.Op.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.10: %Destroy.Op.type.1d8f74.10 = struct_value () [concrete]
// CHECK:STDOUT:   %tuple.type.facet: %type = facet_value %tuple.type.005, () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.70b: type = fn_type @Destroy.WithSelf.SubobjectDestroy, @Destroy.WithSelf(%tuple.type.facet) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.ef1: %Destroy.WithSelf.SubobjectDestroy.type.70b = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.637: type = fn_type @Destroy.WithSelf.SelfDestruct, @Destroy.WithSelf(%tuple.type.facet) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.e9a: %Destroy.WithSelf.SelfDestruct.type.637 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.a5d: <witness> = custom_witness (%Destroy.Op.1a2547.10, %Destroy.WithSelf.SubobjectDestroy.ef1, %Destroy.WithSelf.SelfDestruct.e9a), @Destroy [concrete]
// CHECK:STDOUT:   %Destroy.facet.8b0: %Destroy.type = facet_value %tuple.type.005, (%custom_witness.a5d) [concrete]
// CHECK:STDOUT:   %Iterate.impl_witness.1da: <witness> = impl_witness imports.%Iterate.impl_witness_table.121, @T.as_type.as.Iterate.impl(%facet_value.e04) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.type.6e5: type = fn_type @T.as_type.as.Iterate.impl.NewCursor, @T.as_type.as.Iterate.impl(%facet_value.e04) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.78d: %T.as_type.as.Iterate.impl.NewCursor.type.6e5 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.type.055: type = fn_type @T.as_type.as.Iterate.impl.Next, @T.as_type.as.Iterate.impl(%facet_value.e04) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.c03: %T.as_type.as.Iterate.impl.Next.type.055 = struct_value () [concrete]
// CHECK:STDOUT:   %Iterate.facet.4b0: %Iterate.type = facet_value %ConstRange, (%Iterate.impl_witness.1da) [concrete]
// CHECK:STDOUT:   %facet_value.ce4: %Iterate_where.type.42d = facet_value %ConstRange, (%Iterate.impl_witness.1da) [concrete]
// CHECK:STDOUT:   %r.param_patt.a73: %pattern_type.025 = value_param_pattern [concrete]
// CHECK:STDOUT:   %r.patt.0b5: %pattern_type.025 = wrapper_binding_pattern r, %r.param_patt.a73 [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.4ed: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.4b0) [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.3e6: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.4b0) [concrete]
// CHECK:STDOUT:   %.ffa: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.4ed, %Iterate.facet.4b0 [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.specific_fn: <specific function> = specific_function %T.as_type.as.Iterate.impl.NewCursor.78d, @T.as_type.as.Iterate.impl.NewCursor(%facet_value.e04) [concrete]
// CHECK:STDOUT:   %ptr.d1a: type = ptr_type %tuple.type.005 [concrete]
// CHECK:STDOUT:   %.4826: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.3e6, %Iterate.facet.4b0 [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.specific_fn: <specific function> = specific_function %T.as_type.as.Iterate.impl.Next.c03, @T.as_type.as.Iterate.impl.Next(%facet_value.e04) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.f80: type = fn_type @Destroy.WithSelf.SelfDestruct, @Destroy.WithSelf(%Destroy.facet.8b0) [concrete]
// CHECK:STDOUT:   %.bb4: type = fn_type_with_self_type %Destroy.WithSelf.SelfDestruct.type.f80, %Destroy.facet.8b0 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.specific_fn.3e5: <specific function> = specific_function %Destroy.WithSelf.SelfDestruct.e9a, @Destroy.WithSelf.SelfDestruct(%tuple.type.facet) [concrete]
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
// CHECK:STDOUT: generic fn @TestIterate(%R.loc20_17.2: %Iterate_where.type.42d) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Iterate.lookup_impl_witness: <witness> = lookup_impl_witness %R.loc20_17.1, @Iterate [symbolic = %Iterate.lookup_impl_witness (constants.%Iterate.lookup_impl_witness.a65)]
// CHECK:STDOUT:   %Iterate.facet.loc24_19.5: %Iterate.type = facet_value %R.as_type.loc20_61.1, (%Iterate.lookup_impl_witness) [symbolic = %Iterate.facet.loc24_19.5 (constants.%Iterate.facet.d84)]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.loc24_19.5) [symbolic = %Iterate.WithSelf.NewCursor.type (constants.%Iterate.WithSelf.NewCursor.type.906)]
// CHECK:STDOUT:   %.loc24_19.23: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type, %Iterate.facet.loc24_19.5 [symbolic = %.loc24_19.23 (constants.%.d8f)]
// CHECK:STDOUT:   %impl.elem2.loc24_19.3: @TestIterate.%.loc24_19.23 (%.d8f) = impl_witness_access %Iterate.lookup_impl_witness, element2 [symbolic = %impl.elem2.loc24_19.3 (constants.%impl.elem2.f1e)]
// CHECK:STDOUT:   %specific_impl_fn.loc24_19.4: <specific function> = specific_impl_function %impl.elem2.loc24_19.3, @Iterate.WithSelf.NewCursor(%Iterate.facet.loc24_19.5) [symbolic = %specific_impl_fn.loc24_19.4 (constants.%specific_impl_fn.e73)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %ptr: type = ptr_type %as_type [symbolic = %ptr (constants.%ptr.e00)]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.loc24_19.5) [symbolic = %Iterate.WithSelf.Next.type (constants.%Iterate.WithSelf.Next.type.e19)]
// CHECK:STDOUT:   %.loc24_19.24: type = fn_type_with_self_type %Iterate.WithSelf.Next.type, %Iterate.facet.loc24_19.5 [symbolic = %.loc24_19.24 (constants.%.633)]
// CHECK:STDOUT:   %impl.elem3.loc24_19.2: @TestIterate.%.loc24_19.24 (%.633) = impl_witness_access %Iterate.lookup_impl_witness, element3 [symbolic = %impl.elem3.loc24_19.2 (constants.%impl.elem3.50c)]
// CHECK:STDOUT:   %specific_impl_fn.loc24_19.5: <specific function> = specific_impl_function %impl.elem3.loc24_19.2, @Iterate.WithSelf.Next(%Iterate.facet.loc24_19.5) [symbolic = %specific_impl_fn.loc24_19.5 (constants.%specific_impl_fn.d94)]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type: type = fn_type @Destroy.WithSelf.SelfDestruct, @Destroy.WithSelf(%impl.elem1) [symbolic = %Destroy.WithSelf.SelfDestruct.type (constants.%Destroy.WithSelf.SelfDestruct.type.e6a)]
// CHECK:STDOUT:   %.loc24_19.25: type = fn_type_with_self_type %Destroy.WithSelf.SelfDestruct.type, %impl.elem1 [symbolic = %.loc24_19.25 (constants.%.9ad)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %impl.elem1, @Destroy [symbolic = %Destroy.lookup_impl_witness (constants.%Destroy.lookup_impl_witness.646)]
// CHECK:STDOUT:   %impl.elem2.loc24_19.4: @TestIterate.%.loc24_19.25 (%.9ad) = impl_witness_access %Destroy.lookup_impl_witness, element2 [symbolic = %impl.elem2.loc24_19.4 (constants.%impl.elem2.206)]
// CHECK:STDOUT:   %specific_impl_fn.loc24_19.6: <specific function> = specific_impl_function %impl.elem2.loc24_19.4, @Destroy.WithSelf.SelfDestruct(%impl.elem1) [symbolic = %specific_impl_fn.loc24_19.6 (constants.%specific_impl_fn.908)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%r.param: @TestIterate.%R.as_type.loc20_61.1 (%R.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %i.patt: %pattern_type.fb7 = value_binding_pattern i [concrete = constants.%i.patt]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %r.ref: @TestIterate.%R.as_type.loc20_61.1 (%R.as_type) = name_ref r, %r
// CHECK:STDOUT:     %impl.elem2.loc24_19.1: @TestIterate.%.loc24_19.23 (%.d8f) = impl_witness_access constants.%Iterate.lookup_impl_witness.a65, element2 [symbolic = %impl.elem2.loc24_19.3 (constants.%impl.elem2.f1e)]
// CHECK:STDOUT:     %bound_method.loc24_19.1: <bound method> = bound_method %r.ref, %impl.elem2.loc24_19.1
// CHECK:STDOUT:     %Iterate.facet.loc24_19.1: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.a65) [symbolic = %Iterate.facet.loc24_19.5 (constants.%Iterate.facet.d84)]
// CHECK:STDOUT:     %.loc24_19.1: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc24_19.1 [symbolic = %Iterate.facet.loc24_19.5 (constants.%Iterate.facet.d84)]
// CHECK:STDOUT:     %Iterate.facet.loc24_19.2: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.a65) [symbolic = %Iterate.facet.loc24_19.5 (constants.%Iterate.facet.d84)]
// CHECK:STDOUT:     %.loc24_19.2: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc24_19.2 [symbolic = %Iterate.facet.loc24_19.5 (constants.%Iterate.facet.d84)]
// CHECK:STDOUT:     %specific_impl_fn.loc24_19.1: <specific function> = specific_impl_function %impl.elem2.loc24_19.1, @Iterate.WithSelf.NewCursor(constants.%Iterate.facet.d84) [symbolic = %specific_impl_fn.loc24_19.4 (constants.%specific_impl_fn.e73)]
// CHECK:STDOUT:     %bound_method.loc24_19.2: <bound method> = bound_method %r.ref, %specific_impl_fn.loc24_19.1
// CHECK:STDOUT:     %var: ref @TestIterate.%as_type (%as_type.74e) = var_storage invalid
// CHECK:STDOUT:     %Iterate.WithSelf.NewCursor.call: init @TestIterate.%as_type (%as_type.74e) to %var = call %bound_method.loc24_19.2(%r.ref)
// CHECK:STDOUT:     assign %var, %Iterate.WithSelf.NewCursor.call
// CHECK:STDOUT:     br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.next:
// CHECK:STDOUT:     %addr: @TestIterate.%ptr (%ptr.e00) = addr_of %var
// CHECK:STDOUT:     %impl.elem3.loc24_19.1: @TestIterate.%.loc24_19.24 (%.633) = impl_witness_access constants.%Iterate.lookup_impl_witness.a65, element3 [symbolic = %impl.elem3.loc24_19.2 (constants.%impl.elem3.50c)]
// CHECK:STDOUT:     %bound_method.loc24_19.3: <bound method> = bound_method %r.ref, %impl.elem3.loc24_19.1
// CHECK:STDOUT:     %Iterate.facet.loc24_19.3: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.a65) [symbolic = %Iterate.facet.loc24_19.5 (constants.%Iterate.facet.d84)]
// CHECK:STDOUT:     %.loc24_19.3: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc24_19.3 [symbolic = %Iterate.facet.loc24_19.5 (constants.%Iterate.facet.d84)]
// CHECK:STDOUT:     %Iterate.facet.loc24_19.4: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.a65) [symbolic = %Iterate.facet.loc24_19.5 (constants.%Iterate.facet.d84)]
// CHECK:STDOUT:     %.loc24_19.4: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc24_19.4 [symbolic = %Iterate.facet.loc24_19.5 (constants.%Iterate.facet.d84)]
// CHECK:STDOUT:     %.loc24_19.5: %Destroy.type = converted constants.%as_type.74e, constants.%impl.elem1.d78 [symbolic = %impl.elem1 (constants.%impl.elem1.d78)]
// CHECK:STDOUT:     %.loc24_19.6: %Destroy.type = converted constants.%as_type.74e, constants.%impl.elem1.d78 [symbolic = %impl.elem1 (constants.%impl.elem1.d78)]
// CHECK:STDOUT:     %specific_impl_fn.loc24_19.2: <specific function> = specific_impl_function %impl.elem3.loc24_19.1, @Iterate.WithSelf.Next(constants.%Iterate.facet.d84) [symbolic = %specific_impl_fn.loc24_19.5 (constants.%specific_impl_fn.d94)]
// CHECK:STDOUT:     %bound_method.loc24_19.4: <bound method> = bound_method %r.ref, %specific_impl_fn.loc24_19.2
// CHECK:STDOUT:     %.loc24_19.7: ref %Optional.226 = temporary_storage
// CHECK:STDOUT:     %Iterate.WithSelf.Next.call: init %Optional.226 to %.loc24_19.7 = call %bound_method.loc24_19.4(%r.ref, %addr)
// CHECK:STDOUT:     %.loc24_19.8: ref %Optional.226 = temporary %.loc24_19.7, %Iterate.WithSelf.Next.call
// CHECK:STDOUT:     %.loc24_19.9: %Optional.HasValue.type.f0d = specific_constant imports.%Core.import_ref.bb6, @Optional(constants.%OptionalStorage.facet.357) [concrete = constants.%Optional.HasValue.eb2]
// CHECK:STDOUT:     %HasValue.ref: %Optional.HasValue.type.f0d = name_ref HasValue, %.loc24_19.9 [concrete = constants.%Optional.HasValue.eb2]
// CHECK:STDOUT:     %Optional.HasValue.bound: <bound method> = bound_method %.loc24_19.8, %HasValue.ref
// CHECK:STDOUT:     %Optional.HasValue.specific_fn: <specific function> = specific_function %HasValue.ref, @Optional.HasValue(constants.%OptionalStorage.facet.357) [concrete = constants.%Optional.HasValue.specific_fn.0ad]
// CHECK:STDOUT:     %bound_method.loc24_19.5: <bound method> = bound_method %.loc24_19.8, %Optional.HasValue.specific_fn
// CHECK:STDOUT:     %.loc24_19.10: %Optional.226 = acquire_value %.loc24_19.8
// CHECK:STDOUT:     %Optional.HasValue.call: init bool = call %bound_method.loc24_19.5(%.loc24_19.10)
// CHECK:STDOUT:     %.loc24_19.11: bool = value_of_initializer %Optional.HasValue.call
// CHECK:STDOUT:     %.loc24_19.12: bool = converted %Optional.HasValue.call, %.loc24_19.11
// CHECK:STDOUT:     if %.loc24_19.12 br !for.body else br !for.done
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.body:
// CHECK:STDOUT:     %.loc24_19.13: %Optional.Get.type.d30 = specific_constant imports.%Core.import_ref.dab, @Optional(constants.%OptionalStorage.facet.357) [concrete = constants.%Optional.Get.617]
// CHECK:STDOUT:     %Get.ref: %Optional.Get.type.d30 = name_ref Get, %.loc24_19.13 [concrete = constants.%Optional.Get.617]
// CHECK:STDOUT:     %Optional.Get.bound: <bound method> = bound_method %.loc24_19.8, %Get.ref
// CHECK:STDOUT:     %Optional.Get.specific_fn: <specific function> = specific_function %Get.ref, @Optional.Get(constants.%OptionalStorage.facet.357) [concrete = constants.%Optional.Get.specific_fn.cde]
// CHECK:STDOUT:     %bound_method.loc24_19.6: <bound method> = bound_method %.loc24_19.8, %Optional.Get.specific_fn
// CHECK:STDOUT:     %.loc24_19.14: %Optional.226 = acquire_value %.loc24_19.8
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
// CHECK:STDOUT:     %Destroy.WithSelf.SelfDestruct.bound.loc24_19.1: <bound method> = bound_method %.loc24_19.8, constants.%Destroy.WithSelf.SelfDestruct.f02
// CHECK:STDOUT:     %Destroy.facet.loc24_19.1: %Destroy.type = facet_value constants.%Optional.226, (constants.%custom_witness.372) [concrete = constants.%Destroy.facet.5cd]
// CHECK:STDOUT:     %.loc24_19.17: %Destroy.type = converted constants.%Optional.226, %Destroy.facet.loc24_19.1 [concrete = constants.%Destroy.facet.5cd]
// CHECK:STDOUT:     %Destroy.facet.loc24_19.2: %Destroy.type = facet_value constants.%Optional.226, (constants.%custom_witness.372) [concrete = constants.%Destroy.facet.5cd]
// CHECK:STDOUT:     %.loc24_19.18: %Destroy.type = converted constants.%Optional.226, %Destroy.facet.loc24_19.2 [concrete = constants.%Destroy.facet.5cd]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %bound_method.loc24_19.7: <bound method> = bound_method %.loc24_19.8, %Destroy.WithSelf.SelfDestruct.specific_fn.1
// CHECK:STDOUT:     %Destroy.WithSelf.SelfDestruct.call.loc24_19.1: init %empty_tuple.type = call %bound_method.loc24_19.7(%.loc24_19.8)
// CHECK:STDOUT:     br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.done:
// CHECK:STDOUT:     %Destroy.WithSelf.SelfDestruct.bound.loc24_19.2: <bound method> = bound_method %.loc24_19.8, constants.%Destroy.WithSelf.SelfDestruct.f02
// CHECK:STDOUT:     %Destroy.facet.loc24_19.3: %Destroy.type = facet_value constants.%Optional.226, (constants.%custom_witness.372) [concrete = constants.%Destroy.facet.5cd]
// CHECK:STDOUT:     %.loc24_19.19: %Destroy.type = converted constants.%Optional.226, %Destroy.facet.loc24_19.3 [concrete = constants.%Destroy.facet.5cd]
// CHECK:STDOUT:     %Destroy.facet.loc24_19.4: %Destroy.type = facet_value constants.%Optional.226, (constants.%custom_witness.372) [concrete = constants.%Destroy.facet.5cd]
// CHECK:STDOUT:     %.loc24_19.20: %Destroy.type = converted constants.%Optional.226, %Destroy.facet.loc24_19.4 [concrete = constants.%Destroy.facet.5cd]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %bound_method.loc24_19.8: <bound method> = bound_method %.loc24_19.8, %Destroy.WithSelf.SelfDestruct.specific_fn.2
// CHECK:STDOUT:     %Destroy.WithSelf.SelfDestruct.call.loc24_19.2: init %empty_tuple.type = call %bound_method.loc24_19.8(%.loc24_19.8)
// CHECK:STDOUT:     %impl.elem2.loc24_19.2: @TestIterate.%.loc24_19.25 (%.9ad) = impl_witness_access constants.%Destroy.lookup_impl_witness.646, element2 [symbolic = %impl.elem2.loc24_19.4 (constants.%impl.elem2.206)]
// CHECK:STDOUT:     %bound_method.loc24_19.9: <bound method> = bound_method %var, %impl.elem2.loc24_19.2
// CHECK:STDOUT:     %.loc24_19.21: %Destroy.type = converted constants.%as_type.74e, constants.%impl.elem1.d78 [symbolic = %impl.elem1 (constants.%impl.elem1.d78)]
// CHECK:STDOUT:     %.loc24_19.22: %Destroy.type = converted constants.%as_type.74e, constants.%impl.elem1.d78 [symbolic = %impl.elem1 (constants.%impl.elem1.d78)]
// CHECK:STDOUT:     %specific_impl_fn.loc24_19.3: <specific function> = specific_impl_function %impl.elem2.loc24_19.2, @Destroy.WithSelf.SelfDestruct(constants.%impl.elem1.d78) [symbolic = %specific_impl_fn.loc24_19.6 (constants.%specific_impl_fn.908)]
// CHECK:STDOUT:     %bound_method.loc24_19.10: <bound method> = bound_method %var, %specific_impl_fn.loc24_19.3
// CHECK:STDOUT:     %Destroy.WithSelf.SelfDestruct.call.loc24_19.3: init %empty_tuple.type = call %bound_method.loc24_19.10(%var)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc24_19.1(%self.param: ref %.cc0) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc24_19.2(%self.param: ref %MaybeUnformed.ea6) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc24_19.3(%self.param: ref bool) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc24_19.4(%self.param: ref %struct_type.value.has_value.0fe) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc24_19.5(%self.param: ref %DefaultOptionalStorage.83b) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc24_19.6(%self.param: ref %Optional.226) {
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
// CHECK:STDOUT:   %impl.elem2: %.ffa = impl_witness_access constants.%Iterate.impl_witness.1da, element2 [concrete = constants.%T.as_type.as.Iterate.impl.NewCursor.78d]
// CHECK:STDOUT:   %bound_method.loc38_19.1: <bound method> = bound_method %c.ref, %impl.elem2
// CHECK:STDOUT:   %facet_value.loc38_19.1: %CppRange_where.type.d682d6.1 = facet_value constants.%ConstRange, (constants.%custom_witness.d89, constants.%Copy.impl_witness.0d6, constants.%custom_witness.5d5, constants.%custom_witness.6fe, constants.%custom_witness.994, constants.%custom_witness.393, constants.%custom_witness.821) [concrete = constants.%facet_value.e04]
// CHECK:STDOUT:   %.loc38_19.1: %CppRange_where.type.d682d6.1 = converted constants.%ConstRange, %facet_value.loc38_19.1 [concrete = constants.%facet_value.e04]
// CHECK:STDOUT:   %facet_value.loc38_19.2: %CppRange_where.type.d682d6.1 = facet_value constants.%ConstRange, (constants.%custom_witness.d89, constants.%Copy.impl_witness.0d6, constants.%custom_witness.5d5, constants.%custom_witness.6fe, constants.%custom_witness.994, constants.%custom_witness.393, constants.%custom_witness.821) [concrete = constants.%facet_value.e04]
// CHECK:STDOUT:   %.loc38_19.2: %CppRange_where.type.d682d6.1 = converted constants.%ConstRange, %facet_value.loc38_19.2 [concrete = constants.%facet_value.e04]
// CHECK:STDOUT:   %specific_fn.loc38_19.1: <specific function> = specific_function %impl.elem2, @T.as_type.as.Iterate.impl.NewCursor(constants.%facet_value.e04) [concrete = constants.%T.as_type.as.Iterate.impl.NewCursor.specific_fn]
// CHECK:STDOUT:   %bound_method.loc38_19.2: <bound method> = bound_method %c.ref, %specific_fn.loc38_19.1
// CHECK:STDOUT:   %var: ref %tuple.type.005 = var_storage invalid
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.call: init %tuple.type.005 to %var = call %bound_method.loc38_19.2(%c.ref)
// CHECK:STDOUT:   assign %var, %T.as_type.as.Iterate.impl.NewCursor.call
// CHECK:STDOUT:   br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT: !for.next:
// CHECK:STDOUT:   %addr: %ptr.d1a = addr_of %var
// CHECK:STDOUT:   %impl.elem3: %.4826 = impl_witness_access constants.%Iterate.impl_witness.1da, element3 [concrete = constants.%T.as_type.as.Iterate.impl.Next.c03]
// CHECK:STDOUT:   %bound_method.loc38_19.3: <bound method> = bound_method %c.ref, %impl.elem3
// CHECK:STDOUT:   %facet_value.loc38_19.3: %CppRange_where.type.d682d6.1 = facet_value constants.%ConstRange, (constants.%custom_witness.d89, constants.%Copy.impl_witness.0d6, constants.%custom_witness.5d5, constants.%custom_witness.6fe, constants.%custom_witness.994, constants.%custom_witness.393, constants.%custom_witness.821) [concrete = constants.%facet_value.e04]
// CHECK:STDOUT:   %.loc38_19.3: %CppRange_where.type.d682d6.1 = converted constants.%ConstRange, %facet_value.loc38_19.3 [concrete = constants.%facet_value.e04]
// CHECK:STDOUT:   %facet_value.loc38_19.4: %CppRange_where.type.d682d6.1 = facet_value constants.%ConstRange, (constants.%custom_witness.d89, constants.%Copy.impl_witness.0d6, constants.%custom_witness.5d5, constants.%custom_witness.6fe, constants.%custom_witness.994, constants.%custom_witness.393, constants.%custom_witness.821) [concrete = constants.%facet_value.e04]
// CHECK:STDOUT:   %.loc38_19.4: %CppRange_where.type.d682d6.1 = converted constants.%ConstRange, %facet_value.loc38_19.4 [concrete = constants.%facet_value.e04]
// CHECK:STDOUT:   %specific_fn.loc38_19.2: <specific function> = specific_function %impl.elem3, @T.as_type.as.Iterate.impl.Next(constants.%facet_value.e04) [concrete = constants.%T.as_type.as.Iterate.impl.Next.specific_fn]
// CHECK:STDOUT:   %bound_method.loc38_19.4: <bound method> = bound_method %c.ref, %specific_fn.loc38_19.2
// CHECK:STDOUT:   %.loc38_19.5: ref %Optional.226 = temporary_storage
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.call: init %Optional.226 to %.loc38_19.5 = call %bound_method.loc38_19.4(%c.ref, %addr)
// CHECK:STDOUT:   %.loc38_19.6: ref %Optional.226 = temporary %.loc38_19.5, %T.as_type.as.Iterate.impl.Next.call
// CHECK:STDOUT:   %.loc38_19.7: %Optional.HasValue.type.f0d = specific_constant imports.%Core.import_ref.bb6, @Optional(constants.%OptionalStorage.facet.357) [concrete = constants.%Optional.HasValue.eb2]
// CHECK:STDOUT:   %HasValue.ref: %Optional.HasValue.type.f0d = name_ref HasValue, %.loc38_19.7 [concrete = constants.%Optional.HasValue.eb2]
// CHECK:STDOUT:   %Optional.HasValue.bound: <bound method> = bound_method %.loc38_19.6, %HasValue.ref
// CHECK:STDOUT:   %Optional.HasValue.specific_fn: <specific function> = specific_function %HasValue.ref, @Optional.HasValue(constants.%OptionalStorage.facet.357) [concrete = constants.%Optional.HasValue.specific_fn.0ad]
// CHECK:STDOUT:   %bound_method.loc38_19.5: <bound method> = bound_method %.loc38_19.6, %Optional.HasValue.specific_fn
// CHECK:STDOUT:   %.loc38_19.8: %Optional.226 = acquire_value %.loc38_19.6
// CHECK:STDOUT:   %Optional.HasValue.call: init bool = call %bound_method.loc38_19.5(%.loc38_19.8)
// CHECK:STDOUT:   %.loc38_19.9: bool = value_of_initializer %Optional.HasValue.call
// CHECK:STDOUT:   %.loc38_19.10: bool = converted %Optional.HasValue.call, %.loc38_19.9
// CHECK:STDOUT:   if %.loc38_19.10 br !for.body else br !for.done
// CHECK:STDOUT:
// CHECK:STDOUT: !for.body:
// CHECK:STDOUT:   %.loc38_19.11: %Optional.Get.type.d30 = specific_constant imports.%Core.import_ref.dab, @Optional(constants.%OptionalStorage.facet.357) [concrete = constants.%Optional.Get.617]
// CHECK:STDOUT:   %Get.ref: %Optional.Get.type.d30 = name_ref Get, %.loc38_19.11 [concrete = constants.%Optional.Get.617]
// CHECK:STDOUT:   %Optional.Get.bound: <bound method> = bound_method %.loc38_19.6, %Get.ref
// CHECK:STDOUT:   %Optional.Get.specific_fn: <specific function> = specific_function %Get.ref, @Optional.Get(constants.%OptionalStorage.facet.357) [concrete = constants.%Optional.Get.specific_fn.cde]
// CHECK:STDOUT:   %bound_method.loc38_19.6: <bound method> = bound_method %.loc38_19.6, %Optional.Get.specific_fn
// CHECK:STDOUT:   %.loc38_19.12: %Optional.226 = acquire_value %.loc38_19.6
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
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc38_19.1: <bound method> = bound_method %.loc38_19.6, constants.%Destroy.WithSelf.SelfDestruct.f02
// CHECK:STDOUT:   %Destroy.facet.loc38_19.1: %Destroy.type = facet_value constants.%Optional.226, (constants.%custom_witness.372) [concrete = constants.%Destroy.facet.5cd]
// CHECK:STDOUT:   %.loc38_19.15: %Destroy.type = converted constants.%Optional.226, %Destroy.facet.loc38_19.1 [concrete = constants.%Destroy.facet.5cd]
// CHECK:STDOUT:   %Destroy.facet.loc38_19.2: %Destroy.type = facet_value constants.%Optional.226, (constants.%custom_witness.372) [concrete = constants.%Destroy.facet.5cd]
// CHECK:STDOUT:   %.loc38_19.16: %Destroy.type = converted constants.%Optional.226, %Destroy.facet.loc38_19.2 [concrete = constants.%Destroy.facet.5cd]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc38_19.7: <bound method> = bound_method %.loc38_19.6, %Destroy.WithSelf.SelfDestruct.specific_fn.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc38_19.1: init %empty_tuple.type = call %bound_method.loc38_19.7(%.loc38_19.6)
// CHECK:STDOUT:   br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT: !for.done:
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc38_19.2: <bound method> = bound_method %.loc38_19.6, constants.%Destroy.WithSelf.SelfDestruct.f02
// CHECK:STDOUT:   %Destroy.facet.loc38_19.3: %Destroy.type = facet_value constants.%Optional.226, (constants.%custom_witness.372) [concrete = constants.%Destroy.facet.5cd]
// CHECK:STDOUT:   %.loc38_19.17: %Destroy.type = converted constants.%Optional.226, %Destroy.facet.loc38_19.3 [concrete = constants.%Destroy.facet.5cd]
// CHECK:STDOUT:   %Destroy.facet.loc38_19.4: %Destroy.type = facet_value constants.%Optional.226, (constants.%custom_witness.372) [concrete = constants.%Destroy.facet.5cd]
// CHECK:STDOUT:   %.loc38_19.18: %Destroy.type = converted constants.%Optional.226, %Destroy.facet.loc38_19.4 [concrete = constants.%Destroy.facet.5cd]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc38_19.8: <bound method> = bound_method %.loc38_19.6, %Destroy.WithSelf.SelfDestruct.specific_fn.2
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc38_19.2: init %empty_tuple.type = call %bound_method.loc38_19.8(%.loc38_19.6)
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc38_19.3: <bound method> = bound_method %var, constants.%Destroy.WithSelf.SelfDestruct.e9a
// CHECK:STDOUT:   %Destroy.facet.loc38_19.5: %Destroy.type = facet_value constants.%tuple.type.005, (constants.%custom_witness.a5d) [concrete = constants.%Destroy.facet.8b0]
// CHECK:STDOUT:   %.loc38_19.19: %Destroy.type = converted constants.%tuple.type.005, %Destroy.facet.loc38_19.5 [concrete = constants.%Destroy.facet.8b0]
// CHECK:STDOUT:   %Destroy.facet.loc38_19.6: %Destroy.type = facet_value constants.%tuple.type.005, (constants.%custom_witness.a5d) [concrete = constants.%Destroy.facet.8b0]
// CHECK:STDOUT:   %.loc38_19.20: %Destroy.type = converted constants.%tuple.type.005, %Destroy.facet.loc38_19.6 [concrete = constants.%Destroy.facet.8b0]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc38_19.9: <bound method> = bound_method %var, %Destroy.WithSelf.SelfDestruct.specific_fn.3
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc38_19.3: init %empty_tuple.type = call %bound_method.loc38_19.9(%var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestIterate(constants.%R) {
// CHECK:STDOUT:   %R.patt.loc20_17.2 => constants.%R.patt
// CHECK:STDOUT:   %R.loc20_17.1 => constants.%R
// CHECK:STDOUT:   %R.as_type.loc20_61.1 => constants.%R.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.609
// CHECK:STDOUT:   %r.param_patt.loc20_59.2 => constants.%r.param_patt.09b
// CHECK:STDOUT:   %r.patt.loc20_59.2 => constants.%r.patt.772
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestIterate(constants.%facet_value.ce4) {
// CHECK:STDOUT:   %R.patt.loc20_17.2 => constants.%R.patt
// CHECK:STDOUT:   %R.loc20_17.1 => constants.%facet_value.ce4
// CHECK:STDOUT:   %R.as_type.loc20_61.1 => constants.%ConstRange
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.025
// CHECK:STDOUT:   %r.param_patt.loc20_59.2 => constants.%r.param_patt.a73
// CHECK:STDOUT:   %r.patt.loc20_59.2 => constants.%r.patt.0b5
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc20 => constants.%complete_type.357
// CHECK:STDOUT:   %Iterate.lookup_impl_witness => constants.%Iterate.impl_witness.1da
// CHECK:STDOUT:   %Iterate.facet.loc24_19.5 => constants.%Iterate.facet.4b0
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type => constants.%Iterate.WithSelf.NewCursor.type.4ed
// CHECK:STDOUT:   %.loc24_19.23 => constants.%.ffa
// CHECK:STDOUT:   %impl.elem2.loc24_19.3 => constants.%T.as_type.as.Iterate.impl.NewCursor.78d
// CHECK:STDOUT:   %specific_impl_fn.loc24_19.4 => constants.%T.as_type.as.Iterate.impl.NewCursor.specific_fn
// CHECK:STDOUT:   %impl.elem1 => constants.%Destroy.facet.8b0
// CHECK:STDOUT:   %as_type => constants.%tuple.type.005
// CHECK:STDOUT:   %require_complete.1 => constants.%complete_type.320
// CHECK:STDOUT:   %ptr => constants.%ptr.d1a
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type => constants.%Iterate.WithSelf.Next.type.3e6
// CHECK:STDOUT:   %.loc24_19.24 => constants.%.4826
// CHECK:STDOUT:   %impl.elem3.loc24_19.2 => constants.%T.as_type.as.Iterate.impl.Next.c03
// CHECK:STDOUT:   %specific_impl_fn.loc24_19.5 => constants.%T.as_type.as.Iterate.impl.Next.specific_fn
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type => constants.%Destroy.WithSelf.SelfDestruct.type.f80
// CHECK:STDOUT:   %.loc24_19.25 => constants.%.bb4
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.a5d
// CHECK:STDOUT:   %impl.elem2.loc24_19.4 => constants.%Destroy.WithSelf.SelfDestruct.e9a
// CHECK:STDOUT:   %specific_impl_fn.loc24_19.6 => constants.%Destroy.WithSelf.SelfDestruct.specific_fn.3e5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_adl_returns_same_types_mutable.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
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
// CHECK:STDOUT:   %Float.type.facet: %type = facet_value %f64.dc1, () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.107: type = fn_type @Destroy.WithSelf.SubobjectDestroy, @Destroy.WithSelf(%Float.type.facet) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.2d5: %Destroy.WithSelf.SubobjectDestroy.type.107 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.c4e: type = fn_type @Destroy.WithSelf.SelfDestruct, @Destroy.WithSelf(%Float.type.facet) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.a03: %Destroy.WithSelf.SelfDestruct.type.c4e = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.d89: <witness> = custom_witness (%Destroy.Op.1a2547.3, %Destroy.WithSelf.SubobjectDestroy.2d5, %Destroy.WithSelf.SelfDestruct.a03), @Destroy [concrete]
// CHECK:STDOUT:   %Float.as.Copy.impl.Op.type.ff3: type = fn_type @Float.as.Copy.impl.Op, @Float.as.Copy.impl(%N.fe9) [symbolic]
// CHECK:STDOUT:   %Float.as.Copy.impl.Op.d93: %Float.as.Copy.impl.Op.type.ff3 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.0d6: <witness> = impl_witness imports.%Copy.impl_witness_table.8dd, @Float.as.Copy.impl(%int_64) [concrete]
// CHECK:STDOUT:   %facet_value.915: %facet_type.f48 = facet_value %f64.dc1, (%custom_witness.d89, %Copy.impl_witness.0d6) [concrete]
// CHECK:STDOUT:   %Iterate.lookup_impl_witness.1d8: <witness> = lookup_impl_witness %.Self.500, @Iterate [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.294: %facet_type.f48 = impl_witness_access %Iterate.lookup_impl_witness.1d8, element0 [symbolic_self]
// CHECK:STDOUT:   %Iterate_where.type.42d: type = facet_type <@Iterate where %impl.elem0.294 = %facet_value.915> [concrete]
// CHECK:STDOUT:   %pattern_type.879: type = pattern_type %Iterate_where.type.42d [concrete]
// CHECK:STDOUT:   %R.patt: %pattern_type.879 = symbolic_binding_pattern R, 0 [symbolic]
// CHECK:STDOUT:   %R: %Iterate_where.type.42d = symbolic_binding R, 0 [symbolic]
// CHECK:STDOUT:   %R.as_type: type = facet_access_type %R [symbolic]
// CHECK:STDOUT:   %pattern_type.609: type = pattern_type %R.as_type [symbolic]
// CHECK:STDOUT:   %r.param_patt.09b: %pattern_type.609 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %r.patt.772: %pattern_type.609 = wrapper_binding_pattern r, %r.param_patt.09b [symbolic]
// CHECK:STDOUT:   %i.patt: %pattern_type.fb7 = value_binding_pattern i [concrete]
// CHECK:STDOUT:   %Iterate.lookup_impl_witness.a65: <witness> = lookup_impl_witness %R, @Iterate [symbolic]
// CHECK:STDOUT:   %Iterate.facet.d84: %Iterate.type = facet_value %R.as_type, (%Iterate.lookup_impl_witness.a65) [symbolic]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.906: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.d84) [symbolic]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.e19: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.d84) [symbolic]
// CHECK:STDOUT:   %.d8f: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.906, %Iterate.facet.d84 [symbolic]
// CHECK:STDOUT:   %impl.elem2.f1e: %.d8f = impl_witness_access %Iterate.lookup_impl_witness.a65, element2 [symbolic]
// CHECK:STDOUT:   %impl.elem1.d78: %Destroy.type = impl_witness_access %Iterate.lookup_impl_witness.a65, element1 [symbolic]
// CHECK:STDOUT:   %as_type.74e: type = facet_access_type %impl.elem1.d78 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.e73: <specific function> = specific_impl_function %impl.elem2.f1e, @Iterate.WithSelf.NewCursor(%Iterate.facet.d84) [symbolic]
// CHECK:STDOUT:   %ptr.e00: type = ptr_type %as_type.74e [symbolic]
// CHECK:STDOUT:   %.633: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.e19, %Iterate.facet.d84 [symbolic]
// CHECK:STDOUT:   %impl.elem3.50c: %.633 = impl_witness_access %Iterate.lookup_impl_witness.a65, element3 [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.83b: type = class_type @DefaultOptionalStorage, @DefaultOptionalStorage(%facet_value.915) [concrete]
// CHECK:STDOUT:   %OptionalStorage.impl_witness.3c1: <witness> = impl_witness imports.%OptionalStorage.impl_witness_table.460, @T.as_type.as.OptionalStorage.impl(%facet_value.915) [concrete]
// CHECK:STDOUT:   %OptionalStorage.facet.357: %OptionalStorage.type = facet_value %f64.dc1, (%OptionalStorage.impl_witness.3c1) [concrete]
// CHECK:STDOUT:   %Optional.226: type = class_type @Optional, @Optional(%OptionalStorage.facet.357) [concrete]
// CHECK:STDOUT:   %specific_impl_fn.d94: <specific function> = specific_impl_function %impl.elem3.50c, @Iterate.WithSelf.Next(%Iterate.facet.d84) [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.type.f0d: type = fn_type @Optional.HasValue, @Optional(%OptionalStorage.facet.357) [concrete]
// CHECK:STDOUT:   %Optional.HasValue.eb2: %Optional.HasValue.type.f0d = struct_value () [concrete]
// CHECK:STDOUT:   %Optional.Get.type.d30: type = fn_type @Optional.Get, @Optional(%OptionalStorage.facet.357) [concrete]
// CHECK:STDOUT:   %Optional.Get.617: %Optional.Get.type.d30 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.facet.445: %Destroy.type = facet_value %f64.dc1, (%custom_witness.d89) [concrete]
// CHECK:STDOUT:   %MaybeUnformed.ea6: type = class_type @MaybeUnformed, @MaybeUnformed(%Destroy.facet.445) [concrete]
// CHECK:STDOUT:   %.cc0: type = maybe_unformed_type %f64.dc1 [concrete]
// CHECK:STDOUT:   %struct_type.value.has_value.0fe: type = struct_type {.value: %MaybeUnformed.ea6, .has_value: bool} [concrete]
// CHECK:STDOUT:   %Optional.HasValue.specific_fn.0ad: <specific function> = specific_function %Optional.HasValue.eb2, @Optional.HasValue(%OptionalStorage.facet.357) [concrete]
// CHECK:STDOUT:   %Optional.Get.specific_fn.cde: <specific function> = specific_function %Optional.Get.617, @Optional.Get(%OptionalStorage.facet.357) [concrete]
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
// CHECK:STDOUT:   %Optional.type.facet: %type = facet_value %Optional.226, () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.21d: type = fn_type @Destroy.WithSelf.SubobjectDestroy, @Destroy.WithSelf(%Optional.type.facet) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.f50: %Destroy.WithSelf.SubobjectDestroy.type.21d = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.e02: type = fn_type @Destroy.WithSelf.SelfDestruct, @Destroy.WithSelf(%Optional.type.facet) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.f02: %Destroy.WithSelf.SelfDestruct.type.e02 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.372: <witness> = custom_witness (%Destroy.Op.1a2547.9, %Destroy.WithSelf.SubobjectDestroy.f50, %Destroy.WithSelf.SelfDestruct.f02), @Destroy [concrete]
// CHECK:STDOUT:   %Destroy.facet.5cd: %Destroy.type = facet_value %Optional.226, (%custom_witness.372) [concrete]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.646: <witness> = lookup_impl_witness %impl.elem1.d78, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.e6a: type = fn_type @Destroy.WithSelf.SelfDestruct, @Destroy.WithSelf(%impl.elem1.d78) [symbolic]
// CHECK:STDOUT:   %.9ad: type = fn_type_with_self_type %Destroy.WithSelf.SelfDestruct.type.e6a, %impl.elem1.d78 [symbolic]
// CHECK:STDOUT:   %impl.elem2.206: %.9ad = impl_witness_access %Destroy.lookup_impl_witness.646, element2 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.908: <specific function> = specific_impl_function %impl.elem2.206, @Destroy.WithSelf.SelfDestruct(%impl.elem1.d78) [symbolic]
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
// CHECK:STDOUT:   %Iterator.type.facet: %type = facet_value %Iterator, () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.7a6: type = fn_type @Destroy.WithSelf.SubobjectDestroy, @Destroy.WithSelf(%Iterator.type.facet) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.30f: %Destroy.WithSelf.SubobjectDestroy.type.7a6 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.ef6: type = fn_type @Destroy.WithSelf.SelfDestruct, @Destroy.WithSelf(%Iterator.type.facet) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.3d0: %Destroy.WithSelf.SelfDestruct.type.ef6 = struct_value () [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.b9731e.2: type = fn_type @Iterator.Op.2 [concrete]
// CHECK:STDOUT:   %Iterator.Op.ac6aad.2: %Iterator.Op.type.b9731e.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.5ef: <witness> = custom_witness (%Iterator.Op.ac6aad.2, %Destroy.WithSelf.SubobjectDestroy.30f, %Destroy.WithSelf.SelfDestruct.3d0), @Destroy [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.b9731e.3: type = fn_type @Iterator.Op.3 [concrete]
// CHECK:STDOUT:   %Iterator.Op.ac6aad.3: %Iterator.Op.type.b9731e.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.491: <witness> = custom_witness (%Iterator.Op.ac6aad.3), @Inc [concrete]
// CHECK:STDOUT:   %Equal.type: type = fn_type @Equal [concrete]
// CHECK:STDOUT:   %Equal: %Equal.type = struct_value () [concrete]
// CHECK:STDOUT:   %NotEqual.type: type = fn_type @NotEqual [concrete]
// CHECK:STDOUT:   %NotEqual: %NotEqual.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.ed8: <witness> = custom_witness (%Equal, %NotEqual), @EqWith, @EqWith(%Iterator) [concrete]
// CHECK:STDOUT:   %facet_value.0a2: %CppRange_where.type.d682d6.1 = facet_value %MutableRange, (%custom_witness.d89, %Copy.impl_witness.0d6, %custom_witness.902, %custom_witness.5ef, %custom_witness.05b, %custom_witness.491, %custom_witness.ed8) [concrete]
// CHECK:STDOUT:   %tuple.type.dd0: type = tuple_type (%Iterator, %Iterator) [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.10: type = fn_type @Destroy.Op.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.10: %Destroy.Op.type.1d8f74.10 = struct_value () [concrete]
// CHECK:STDOUT:   %tuple.type.facet: %type = facet_value %tuple.type.dd0, () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.3d2: type = fn_type @Destroy.WithSelf.SubobjectDestroy, @Destroy.WithSelf(%tuple.type.facet) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.586: %Destroy.WithSelf.SubobjectDestroy.type.3d2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.1e3: type = fn_type @Destroy.WithSelf.SelfDestruct, @Destroy.WithSelf(%tuple.type.facet) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.cbb: %Destroy.WithSelf.SelfDestruct.type.1e3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.ed6: <witness> = custom_witness (%Destroy.Op.1a2547.10, %Destroy.WithSelf.SubobjectDestroy.586, %Destroy.WithSelf.SelfDestruct.cbb), @Destroy [concrete]
// CHECK:STDOUT:   %Destroy.facet.027: %Destroy.type = facet_value %tuple.type.dd0, (%custom_witness.ed6) [concrete]
// CHECK:STDOUT:   %Iterate.impl_witness.58c: <witness> = impl_witness imports.%Iterate.impl_witness_table.121, @T.as_type.as.Iterate.impl(%facet_value.0a2) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.type.4e3: type = fn_type @T.as_type.as.Iterate.impl.NewCursor, @T.as_type.as.Iterate.impl(%facet_value.0a2) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.f24: %T.as_type.as.Iterate.impl.NewCursor.type.4e3 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.type.095: type = fn_type @T.as_type.as.Iterate.impl.Next, @T.as_type.as.Iterate.impl(%facet_value.0a2) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.843: %T.as_type.as.Iterate.impl.Next.type.095 = struct_value () [concrete]
// CHECK:STDOUT:   %Iterate.facet.1cb: %Iterate.type = facet_value %MutableRange, (%Iterate.impl_witness.58c) [concrete]
// CHECK:STDOUT:   %facet_value.9e2: %Iterate_where.type.42d = facet_value %MutableRange, (%Iterate.impl_witness.58c) [concrete]
// CHECK:STDOUT:   %r.param_patt.aeb: %pattern_type.992 = value_param_pattern [concrete]
// CHECK:STDOUT:   %r.patt.892: %pattern_type.992 = wrapper_binding_pattern r, %r.param_patt.aeb [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.b44: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.1cb) [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.a23: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.1cb) [concrete]
// CHECK:STDOUT:   %.de7: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.b44, %Iterate.facet.1cb [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.specific_fn: <specific function> = specific_function %T.as_type.as.Iterate.impl.NewCursor.f24, @T.as_type.as.Iterate.impl.NewCursor(%facet_value.0a2) [concrete]
// CHECK:STDOUT:   %ptr.a4f: type = ptr_type %tuple.type.dd0 [concrete]
// CHECK:STDOUT:   %.802: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.a23, %Iterate.facet.1cb [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.specific_fn: <specific function> = specific_function %T.as_type.as.Iterate.impl.Next.843, @T.as_type.as.Iterate.impl.Next(%facet_value.0a2) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.956: type = fn_type @Destroy.WithSelf.SelfDestruct, @Destroy.WithSelf(%Destroy.facet.027) [concrete]
// CHECK:STDOUT:   %.549: type = fn_type_with_self_type %Destroy.WithSelf.SelfDestruct.type.956, %Destroy.facet.027 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.specific_fn.383: <specific function> = specific_function %Destroy.WithSelf.SelfDestruct.cbb, @Destroy.WithSelf.SelfDestruct(%tuple.type.facet) [concrete]
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
// CHECK:STDOUT: generic fn @TestIterate(%R.loc22_17.2: %Iterate_where.type.42d) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Iterate.lookup_impl_witness: <witness> = lookup_impl_witness %R.loc22_17.1, @Iterate [symbolic = %Iterate.lookup_impl_witness (constants.%Iterate.lookup_impl_witness.a65)]
// CHECK:STDOUT:   %Iterate.facet.loc26_19.5: %Iterate.type = facet_value %R.as_type.loc22_61.1, (%Iterate.lookup_impl_witness) [symbolic = %Iterate.facet.loc26_19.5 (constants.%Iterate.facet.d84)]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.loc26_19.5) [symbolic = %Iterate.WithSelf.NewCursor.type (constants.%Iterate.WithSelf.NewCursor.type.906)]
// CHECK:STDOUT:   %.loc26_19.23: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type, %Iterate.facet.loc26_19.5 [symbolic = %.loc26_19.23 (constants.%.d8f)]
// CHECK:STDOUT:   %impl.elem2.loc26_19.3: @TestIterate.%.loc26_19.23 (%.d8f) = impl_witness_access %Iterate.lookup_impl_witness, element2 [symbolic = %impl.elem2.loc26_19.3 (constants.%impl.elem2.f1e)]
// CHECK:STDOUT:   %specific_impl_fn.loc26_19.4: <specific function> = specific_impl_function %impl.elem2.loc26_19.3, @Iterate.WithSelf.NewCursor(%Iterate.facet.loc26_19.5) [symbolic = %specific_impl_fn.loc26_19.4 (constants.%specific_impl_fn.e73)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %ptr: type = ptr_type %as_type [symbolic = %ptr (constants.%ptr.e00)]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.loc26_19.5) [symbolic = %Iterate.WithSelf.Next.type (constants.%Iterate.WithSelf.Next.type.e19)]
// CHECK:STDOUT:   %.loc26_19.24: type = fn_type_with_self_type %Iterate.WithSelf.Next.type, %Iterate.facet.loc26_19.5 [symbolic = %.loc26_19.24 (constants.%.633)]
// CHECK:STDOUT:   %impl.elem3.loc26_19.2: @TestIterate.%.loc26_19.24 (%.633) = impl_witness_access %Iterate.lookup_impl_witness, element3 [symbolic = %impl.elem3.loc26_19.2 (constants.%impl.elem3.50c)]
// CHECK:STDOUT:   %specific_impl_fn.loc26_19.5: <specific function> = specific_impl_function %impl.elem3.loc26_19.2, @Iterate.WithSelf.Next(%Iterate.facet.loc26_19.5) [symbolic = %specific_impl_fn.loc26_19.5 (constants.%specific_impl_fn.d94)]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type: type = fn_type @Destroy.WithSelf.SelfDestruct, @Destroy.WithSelf(%impl.elem1) [symbolic = %Destroy.WithSelf.SelfDestruct.type (constants.%Destroy.WithSelf.SelfDestruct.type.e6a)]
// CHECK:STDOUT:   %.loc26_19.25: type = fn_type_with_self_type %Destroy.WithSelf.SelfDestruct.type, %impl.elem1 [symbolic = %.loc26_19.25 (constants.%.9ad)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %impl.elem1, @Destroy [symbolic = %Destroy.lookup_impl_witness (constants.%Destroy.lookup_impl_witness.646)]
// CHECK:STDOUT:   %impl.elem2.loc26_19.4: @TestIterate.%.loc26_19.25 (%.9ad) = impl_witness_access %Destroy.lookup_impl_witness, element2 [symbolic = %impl.elem2.loc26_19.4 (constants.%impl.elem2.206)]
// CHECK:STDOUT:   %specific_impl_fn.loc26_19.6: <specific function> = specific_impl_function %impl.elem2.loc26_19.4, @Destroy.WithSelf.SelfDestruct(%impl.elem1) [symbolic = %specific_impl_fn.loc26_19.6 (constants.%specific_impl_fn.908)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%r.param: @TestIterate.%R.as_type.loc22_61.1 (%R.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %i.patt: %pattern_type.fb7 = value_binding_pattern i [concrete = constants.%i.patt]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %r.ref: @TestIterate.%R.as_type.loc22_61.1 (%R.as_type) = name_ref r, %r
// CHECK:STDOUT:     %impl.elem2.loc26_19.1: @TestIterate.%.loc26_19.23 (%.d8f) = impl_witness_access constants.%Iterate.lookup_impl_witness.a65, element2 [symbolic = %impl.elem2.loc26_19.3 (constants.%impl.elem2.f1e)]
// CHECK:STDOUT:     %bound_method.loc26_19.1: <bound method> = bound_method %r.ref, %impl.elem2.loc26_19.1
// CHECK:STDOUT:     %Iterate.facet.loc26_19.1: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.a65) [symbolic = %Iterate.facet.loc26_19.5 (constants.%Iterate.facet.d84)]
// CHECK:STDOUT:     %.loc26_19.1: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc26_19.1 [symbolic = %Iterate.facet.loc26_19.5 (constants.%Iterate.facet.d84)]
// CHECK:STDOUT:     %Iterate.facet.loc26_19.2: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.a65) [symbolic = %Iterate.facet.loc26_19.5 (constants.%Iterate.facet.d84)]
// CHECK:STDOUT:     %.loc26_19.2: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc26_19.2 [symbolic = %Iterate.facet.loc26_19.5 (constants.%Iterate.facet.d84)]
// CHECK:STDOUT:     %specific_impl_fn.loc26_19.1: <specific function> = specific_impl_function %impl.elem2.loc26_19.1, @Iterate.WithSelf.NewCursor(constants.%Iterate.facet.d84) [symbolic = %specific_impl_fn.loc26_19.4 (constants.%specific_impl_fn.e73)]
// CHECK:STDOUT:     %bound_method.loc26_19.2: <bound method> = bound_method %r.ref, %specific_impl_fn.loc26_19.1
// CHECK:STDOUT:     %var: ref @TestIterate.%as_type (%as_type.74e) = var_storage invalid
// CHECK:STDOUT:     %Iterate.WithSelf.NewCursor.call: init @TestIterate.%as_type (%as_type.74e) to %var = call %bound_method.loc26_19.2(%r.ref)
// CHECK:STDOUT:     assign %var, %Iterate.WithSelf.NewCursor.call
// CHECK:STDOUT:     br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.next:
// CHECK:STDOUT:     %addr: @TestIterate.%ptr (%ptr.e00) = addr_of %var
// CHECK:STDOUT:     %impl.elem3.loc26_19.1: @TestIterate.%.loc26_19.24 (%.633) = impl_witness_access constants.%Iterate.lookup_impl_witness.a65, element3 [symbolic = %impl.elem3.loc26_19.2 (constants.%impl.elem3.50c)]
// CHECK:STDOUT:     %bound_method.loc26_19.3: <bound method> = bound_method %r.ref, %impl.elem3.loc26_19.1
// CHECK:STDOUT:     %Iterate.facet.loc26_19.3: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.a65) [symbolic = %Iterate.facet.loc26_19.5 (constants.%Iterate.facet.d84)]
// CHECK:STDOUT:     %.loc26_19.3: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc26_19.3 [symbolic = %Iterate.facet.loc26_19.5 (constants.%Iterate.facet.d84)]
// CHECK:STDOUT:     %Iterate.facet.loc26_19.4: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.a65) [symbolic = %Iterate.facet.loc26_19.5 (constants.%Iterate.facet.d84)]
// CHECK:STDOUT:     %.loc26_19.4: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc26_19.4 [symbolic = %Iterate.facet.loc26_19.5 (constants.%Iterate.facet.d84)]
// CHECK:STDOUT:     %.loc26_19.5: %Destroy.type = converted constants.%as_type.74e, constants.%impl.elem1.d78 [symbolic = %impl.elem1 (constants.%impl.elem1.d78)]
// CHECK:STDOUT:     %.loc26_19.6: %Destroy.type = converted constants.%as_type.74e, constants.%impl.elem1.d78 [symbolic = %impl.elem1 (constants.%impl.elem1.d78)]
// CHECK:STDOUT:     %specific_impl_fn.loc26_19.2: <specific function> = specific_impl_function %impl.elem3.loc26_19.1, @Iterate.WithSelf.Next(constants.%Iterate.facet.d84) [symbolic = %specific_impl_fn.loc26_19.5 (constants.%specific_impl_fn.d94)]
// CHECK:STDOUT:     %bound_method.loc26_19.4: <bound method> = bound_method %r.ref, %specific_impl_fn.loc26_19.2
// CHECK:STDOUT:     %.loc26_19.7: ref %Optional.226 = temporary_storage
// CHECK:STDOUT:     %Iterate.WithSelf.Next.call: init %Optional.226 to %.loc26_19.7 = call %bound_method.loc26_19.4(%r.ref, %addr)
// CHECK:STDOUT:     %.loc26_19.8: ref %Optional.226 = temporary %.loc26_19.7, %Iterate.WithSelf.Next.call
// CHECK:STDOUT:     %.loc26_19.9: %Optional.HasValue.type.f0d = specific_constant imports.%Core.import_ref.bb6, @Optional(constants.%OptionalStorage.facet.357) [concrete = constants.%Optional.HasValue.eb2]
// CHECK:STDOUT:     %HasValue.ref: %Optional.HasValue.type.f0d = name_ref HasValue, %.loc26_19.9 [concrete = constants.%Optional.HasValue.eb2]
// CHECK:STDOUT:     %Optional.HasValue.bound: <bound method> = bound_method %.loc26_19.8, %HasValue.ref
// CHECK:STDOUT:     %Optional.HasValue.specific_fn: <specific function> = specific_function %HasValue.ref, @Optional.HasValue(constants.%OptionalStorage.facet.357) [concrete = constants.%Optional.HasValue.specific_fn.0ad]
// CHECK:STDOUT:     %bound_method.loc26_19.5: <bound method> = bound_method %.loc26_19.8, %Optional.HasValue.specific_fn
// CHECK:STDOUT:     %.loc26_19.10: %Optional.226 = acquire_value %.loc26_19.8
// CHECK:STDOUT:     %Optional.HasValue.call: init bool = call %bound_method.loc26_19.5(%.loc26_19.10)
// CHECK:STDOUT:     %.loc26_19.11: bool = value_of_initializer %Optional.HasValue.call
// CHECK:STDOUT:     %.loc26_19.12: bool = converted %Optional.HasValue.call, %.loc26_19.11
// CHECK:STDOUT:     if %.loc26_19.12 br !for.body else br !for.done
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.body:
// CHECK:STDOUT:     %.loc26_19.13: %Optional.Get.type.d30 = specific_constant imports.%Core.import_ref.dab, @Optional(constants.%OptionalStorage.facet.357) [concrete = constants.%Optional.Get.617]
// CHECK:STDOUT:     %Get.ref: %Optional.Get.type.d30 = name_ref Get, %.loc26_19.13 [concrete = constants.%Optional.Get.617]
// CHECK:STDOUT:     %Optional.Get.bound: <bound method> = bound_method %.loc26_19.8, %Get.ref
// CHECK:STDOUT:     %Optional.Get.specific_fn: <specific function> = specific_function %Get.ref, @Optional.Get(constants.%OptionalStorage.facet.357) [concrete = constants.%Optional.Get.specific_fn.cde]
// CHECK:STDOUT:     %bound_method.loc26_19.6: <bound method> = bound_method %.loc26_19.8, %Optional.Get.specific_fn
// CHECK:STDOUT:     %.loc26_19.14: %Optional.226 = acquire_value %.loc26_19.8
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
// CHECK:STDOUT:     %Destroy.WithSelf.SelfDestruct.bound.loc26_19.1: <bound method> = bound_method %.loc26_19.8, constants.%Destroy.WithSelf.SelfDestruct.f02
// CHECK:STDOUT:     %Destroy.facet.loc26_19.1: %Destroy.type = facet_value constants.%Optional.226, (constants.%custom_witness.372) [concrete = constants.%Destroy.facet.5cd]
// CHECK:STDOUT:     %.loc26_19.17: %Destroy.type = converted constants.%Optional.226, %Destroy.facet.loc26_19.1 [concrete = constants.%Destroy.facet.5cd]
// CHECK:STDOUT:     %Destroy.facet.loc26_19.2: %Destroy.type = facet_value constants.%Optional.226, (constants.%custom_witness.372) [concrete = constants.%Destroy.facet.5cd]
// CHECK:STDOUT:     %.loc26_19.18: %Destroy.type = converted constants.%Optional.226, %Destroy.facet.loc26_19.2 [concrete = constants.%Destroy.facet.5cd]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %bound_method.loc26_19.7: <bound method> = bound_method %.loc26_19.8, %Destroy.WithSelf.SelfDestruct.specific_fn.1
// CHECK:STDOUT:     %Destroy.WithSelf.SelfDestruct.call.loc26_19.1: init %empty_tuple.type = call %bound_method.loc26_19.7(%.loc26_19.8)
// CHECK:STDOUT:     br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.done:
// CHECK:STDOUT:     %Destroy.WithSelf.SelfDestruct.bound.loc26_19.2: <bound method> = bound_method %.loc26_19.8, constants.%Destroy.WithSelf.SelfDestruct.f02
// CHECK:STDOUT:     %Destroy.facet.loc26_19.3: %Destroy.type = facet_value constants.%Optional.226, (constants.%custom_witness.372) [concrete = constants.%Destroy.facet.5cd]
// CHECK:STDOUT:     %.loc26_19.19: %Destroy.type = converted constants.%Optional.226, %Destroy.facet.loc26_19.3 [concrete = constants.%Destroy.facet.5cd]
// CHECK:STDOUT:     %Destroy.facet.loc26_19.4: %Destroy.type = facet_value constants.%Optional.226, (constants.%custom_witness.372) [concrete = constants.%Destroy.facet.5cd]
// CHECK:STDOUT:     %.loc26_19.20: %Destroy.type = converted constants.%Optional.226, %Destroy.facet.loc26_19.4 [concrete = constants.%Destroy.facet.5cd]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %bound_method.loc26_19.8: <bound method> = bound_method %.loc26_19.8, %Destroy.WithSelf.SelfDestruct.specific_fn.2
// CHECK:STDOUT:     %Destroy.WithSelf.SelfDestruct.call.loc26_19.2: init %empty_tuple.type = call %bound_method.loc26_19.8(%.loc26_19.8)
// CHECK:STDOUT:     %impl.elem2.loc26_19.2: @TestIterate.%.loc26_19.25 (%.9ad) = impl_witness_access constants.%Destroy.lookup_impl_witness.646, element2 [symbolic = %impl.elem2.loc26_19.4 (constants.%impl.elem2.206)]
// CHECK:STDOUT:     %bound_method.loc26_19.9: <bound method> = bound_method %var, %impl.elem2.loc26_19.2
// CHECK:STDOUT:     %.loc26_19.21: %Destroy.type = converted constants.%as_type.74e, constants.%impl.elem1.d78 [symbolic = %impl.elem1 (constants.%impl.elem1.d78)]
// CHECK:STDOUT:     %.loc26_19.22: %Destroy.type = converted constants.%as_type.74e, constants.%impl.elem1.d78 [symbolic = %impl.elem1 (constants.%impl.elem1.d78)]
// CHECK:STDOUT:     %specific_impl_fn.loc26_19.3: <specific function> = specific_impl_function %impl.elem2.loc26_19.2, @Destroy.WithSelf.SelfDestruct(constants.%impl.elem1.d78) [symbolic = %specific_impl_fn.loc26_19.6 (constants.%specific_impl_fn.908)]
// CHECK:STDOUT:     %bound_method.loc26_19.10: <bound method> = bound_method %var, %specific_impl_fn.loc26_19.3
// CHECK:STDOUT:     %Destroy.WithSelf.SelfDestruct.call.loc26_19.3: init %empty_tuple.type = call %bound_method.loc26_19.10(%var)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc26_19.1(%self.param: ref %.cc0) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc26_19.2(%self.param: ref %MaybeUnformed.ea6) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc26_19.3(%self.param: ref bool) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc26_19.4(%self.param: ref %struct_type.value.has_value.0fe) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc26_19.5(%self.param: ref %DefaultOptionalStorage.83b) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc26_19.6(%self.param: ref %Optional.226) {
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
// CHECK:STDOUT:   %impl.elem2: %.de7 = impl_witness_access constants.%Iterate.impl_witness.58c, element2 [concrete = constants.%T.as_type.as.Iterate.impl.NewCursor.f24]
// CHECK:STDOUT:   %bound_method.loc68_19.1: <bound method> = bound_method %m.ref, %impl.elem2
// CHECK:STDOUT:   %facet_value.loc68_19.1: %CppRange_where.type.d682d6.1 = facet_value constants.%MutableRange, (constants.%custom_witness.d89, constants.%Copy.impl_witness.0d6, constants.%custom_witness.902, constants.%custom_witness.5ef, constants.%custom_witness.05b, constants.%custom_witness.491, constants.%custom_witness.ed8) [concrete = constants.%facet_value.0a2]
// CHECK:STDOUT:   %.loc68_19.1: %CppRange_where.type.d682d6.1 = converted constants.%MutableRange, %facet_value.loc68_19.1 [concrete = constants.%facet_value.0a2]
// CHECK:STDOUT:   %facet_value.loc68_19.2: %CppRange_where.type.d682d6.1 = facet_value constants.%MutableRange, (constants.%custom_witness.d89, constants.%Copy.impl_witness.0d6, constants.%custom_witness.902, constants.%custom_witness.5ef, constants.%custom_witness.05b, constants.%custom_witness.491, constants.%custom_witness.ed8) [concrete = constants.%facet_value.0a2]
// CHECK:STDOUT:   %.loc68_19.2: %CppRange_where.type.d682d6.1 = converted constants.%MutableRange, %facet_value.loc68_19.2 [concrete = constants.%facet_value.0a2]
// CHECK:STDOUT:   %specific_fn.loc68_19.1: <specific function> = specific_function %impl.elem2, @T.as_type.as.Iterate.impl.NewCursor(constants.%facet_value.0a2) [concrete = constants.%T.as_type.as.Iterate.impl.NewCursor.specific_fn]
// CHECK:STDOUT:   %bound_method.loc68_19.2: <bound method> = bound_method %m.ref, %specific_fn.loc68_19.1
// CHECK:STDOUT:   %var: ref %tuple.type.dd0 = var_storage invalid
// CHECK:STDOUT:   %.loc68_18.1: %MutableRange = acquire_value %m.ref
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.call: init %tuple.type.dd0 to %var = call %bound_method.loc68_19.2(%.loc68_18.1)
// CHECK:STDOUT:   assign %var, %T.as_type.as.Iterate.impl.NewCursor.call
// CHECK:STDOUT:   br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT: !for.next:
// CHECK:STDOUT:   %addr: %ptr.a4f = addr_of %var
// CHECK:STDOUT:   %impl.elem3: %.802 = impl_witness_access constants.%Iterate.impl_witness.58c, element3 [concrete = constants.%T.as_type.as.Iterate.impl.Next.843]
// CHECK:STDOUT:   %bound_method.loc68_19.3: <bound method> = bound_method %m.ref, %impl.elem3
// CHECK:STDOUT:   %facet_value.loc68_19.3: %CppRange_where.type.d682d6.1 = facet_value constants.%MutableRange, (constants.%custom_witness.d89, constants.%Copy.impl_witness.0d6, constants.%custom_witness.902, constants.%custom_witness.5ef, constants.%custom_witness.05b, constants.%custom_witness.491, constants.%custom_witness.ed8) [concrete = constants.%facet_value.0a2]
// CHECK:STDOUT:   %.loc68_19.3: %CppRange_where.type.d682d6.1 = converted constants.%MutableRange, %facet_value.loc68_19.3 [concrete = constants.%facet_value.0a2]
// CHECK:STDOUT:   %facet_value.loc68_19.4: %CppRange_where.type.d682d6.1 = facet_value constants.%MutableRange, (constants.%custom_witness.d89, constants.%Copy.impl_witness.0d6, constants.%custom_witness.902, constants.%custom_witness.5ef, constants.%custom_witness.05b, constants.%custom_witness.491, constants.%custom_witness.ed8) [concrete = constants.%facet_value.0a2]
// CHECK:STDOUT:   %.loc68_19.4: %CppRange_where.type.d682d6.1 = converted constants.%MutableRange, %facet_value.loc68_19.4 [concrete = constants.%facet_value.0a2]
// CHECK:STDOUT:   %specific_fn.loc68_19.2: <specific function> = specific_function %impl.elem3, @T.as_type.as.Iterate.impl.Next(constants.%facet_value.0a2) [concrete = constants.%T.as_type.as.Iterate.impl.Next.specific_fn]
// CHECK:STDOUT:   %bound_method.loc68_19.4: <bound method> = bound_method %m.ref, %specific_fn.loc68_19.2
// CHECK:STDOUT:   %.loc68_19.5: ref %Optional.226 = temporary_storage
// CHECK:STDOUT:   %.loc68_18.2: %MutableRange = acquire_value %m.ref
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.call: init %Optional.226 to %.loc68_19.5 = call %bound_method.loc68_19.4(%.loc68_18.2, %addr)
// CHECK:STDOUT:   %.loc68_19.6: ref %Optional.226 = temporary %.loc68_19.5, %T.as_type.as.Iterate.impl.Next.call
// CHECK:STDOUT:   %.loc68_19.7: %Optional.HasValue.type.f0d = specific_constant imports.%Core.import_ref.bb6, @Optional(constants.%OptionalStorage.facet.357) [concrete = constants.%Optional.HasValue.eb2]
// CHECK:STDOUT:   %HasValue.ref: %Optional.HasValue.type.f0d = name_ref HasValue, %.loc68_19.7 [concrete = constants.%Optional.HasValue.eb2]
// CHECK:STDOUT:   %Optional.HasValue.bound: <bound method> = bound_method %.loc68_19.6, %HasValue.ref
// CHECK:STDOUT:   %Optional.HasValue.specific_fn: <specific function> = specific_function %HasValue.ref, @Optional.HasValue(constants.%OptionalStorage.facet.357) [concrete = constants.%Optional.HasValue.specific_fn.0ad]
// CHECK:STDOUT:   %bound_method.loc68_19.5: <bound method> = bound_method %.loc68_19.6, %Optional.HasValue.specific_fn
// CHECK:STDOUT:   %.loc68_19.8: %Optional.226 = acquire_value %.loc68_19.6
// CHECK:STDOUT:   %Optional.HasValue.call: init bool = call %bound_method.loc68_19.5(%.loc68_19.8)
// CHECK:STDOUT:   %.loc68_19.9: bool = value_of_initializer %Optional.HasValue.call
// CHECK:STDOUT:   %.loc68_19.10: bool = converted %Optional.HasValue.call, %.loc68_19.9
// CHECK:STDOUT:   if %.loc68_19.10 br !for.body else br !for.done
// CHECK:STDOUT:
// CHECK:STDOUT: !for.body:
// CHECK:STDOUT:   %.loc68_19.11: %Optional.Get.type.d30 = specific_constant imports.%Core.import_ref.dab, @Optional(constants.%OptionalStorage.facet.357) [concrete = constants.%Optional.Get.617]
// CHECK:STDOUT:   %Get.ref: %Optional.Get.type.d30 = name_ref Get, %.loc68_19.11 [concrete = constants.%Optional.Get.617]
// CHECK:STDOUT:   %Optional.Get.bound: <bound method> = bound_method %.loc68_19.6, %Get.ref
// CHECK:STDOUT:   %Optional.Get.specific_fn: <specific function> = specific_function %Get.ref, @Optional.Get(constants.%OptionalStorage.facet.357) [concrete = constants.%Optional.Get.specific_fn.cde]
// CHECK:STDOUT:   %bound_method.loc68_19.6: <bound method> = bound_method %.loc68_19.6, %Optional.Get.specific_fn
// CHECK:STDOUT:   %.loc68_19.12: %Optional.226 = acquire_value %.loc68_19.6
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
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc68_19.1: <bound method> = bound_method %.loc68_19.6, constants.%Destroy.WithSelf.SelfDestruct.f02
// CHECK:STDOUT:   %Destroy.facet.loc68_19.1: %Destroy.type = facet_value constants.%Optional.226, (constants.%custom_witness.372) [concrete = constants.%Destroy.facet.5cd]
// CHECK:STDOUT:   %.loc68_19.15: %Destroy.type = converted constants.%Optional.226, %Destroy.facet.loc68_19.1 [concrete = constants.%Destroy.facet.5cd]
// CHECK:STDOUT:   %Destroy.facet.loc68_19.2: %Destroy.type = facet_value constants.%Optional.226, (constants.%custom_witness.372) [concrete = constants.%Destroy.facet.5cd]
// CHECK:STDOUT:   %.loc68_19.16: %Destroy.type = converted constants.%Optional.226, %Destroy.facet.loc68_19.2 [concrete = constants.%Destroy.facet.5cd]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc68_19.7: <bound method> = bound_method %.loc68_19.6, %Destroy.WithSelf.SelfDestruct.specific_fn.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc68_19.1: init %empty_tuple.type = call %bound_method.loc68_19.7(%.loc68_19.6)
// CHECK:STDOUT:   br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT: !for.done:
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc68_19.2: <bound method> = bound_method %.loc68_19.6, constants.%Destroy.WithSelf.SelfDestruct.f02
// CHECK:STDOUT:   %Destroy.facet.loc68_19.3: %Destroy.type = facet_value constants.%Optional.226, (constants.%custom_witness.372) [concrete = constants.%Destroy.facet.5cd]
// CHECK:STDOUT:   %.loc68_19.17: %Destroy.type = converted constants.%Optional.226, %Destroy.facet.loc68_19.3 [concrete = constants.%Destroy.facet.5cd]
// CHECK:STDOUT:   %Destroy.facet.loc68_19.4: %Destroy.type = facet_value constants.%Optional.226, (constants.%custom_witness.372) [concrete = constants.%Destroy.facet.5cd]
// CHECK:STDOUT:   %.loc68_19.18: %Destroy.type = converted constants.%Optional.226, %Destroy.facet.loc68_19.4 [concrete = constants.%Destroy.facet.5cd]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc68_19.8: <bound method> = bound_method %.loc68_19.6, %Destroy.WithSelf.SelfDestruct.specific_fn.2
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc68_19.2: init %empty_tuple.type = call %bound_method.loc68_19.8(%.loc68_19.6)
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc68_19.3: <bound method> = bound_method %var, constants.%Destroy.WithSelf.SelfDestruct.cbb
// CHECK:STDOUT:   %Destroy.facet.loc68_19.5: %Destroy.type = facet_value constants.%tuple.type.dd0, (constants.%custom_witness.ed6) [concrete = constants.%Destroy.facet.027]
// CHECK:STDOUT:   %.loc68_19.19: %Destroy.type = converted constants.%tuple.type.dd0, %Destroy.facet.loc68_19.5 [concrete = constants.%Destroy.facet.027]
// CHECK:STDOUT:   %Destroy.facet.loc68_19.6: %Destroy.type = facet_value constants.%tuple.type.dd0, (constants.%custom_witness.ed6) [concrete = constants.%Destroy.facet.027]
// CHECK:STDOUT:   %.loc68_19.20: %Destroy.type = converted constants.%tuple.type.dd0, %Destroy.facet.loc68_19.6 [concrete = constants.%Destroy.facet.027]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc68_19.9: <bound method> = bound_method %var, %Destroy.WithSelf.SelfDestruct.specific_fn.3
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc68_19.3: init %empty_tuple.type = call %bound_method.loc68_19.9(%var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestIterate(constants.%R) {
// CHECK:STDOUT:   %R.patt.loc22_17.2 => constants.%R.patt
// CHECK:STDOUT:   %R.loc22_17.1 => constants.%R
// CHECK:STDOUT:   %R.as_type.loc22_61.1 => constants.%R.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.609
// CHECK:STDOUT:   %r.param_patt.loc22_59.2 => constants.%r.param_patt.09b
// CHECK:STDOUT:   %r.patt.loc22_59.2 => constants.%r.patt.772
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestIterate(constants.%facet_value.9e2) {
// CHECK:STDOUT:   %R.patt.loc22_17.2 => constants.%R.patt
// CHECK:STDOUT:   %R.loc22_17.1 => constants.%facet_value.9e2
// CHECK:STDOUT:   %R.as_type.loc22_61.1 => constants.%MutableRange
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.992
// CHECK:STDOUT:   %r.param_patt.loc22_59.2 => constants.%r.param_patt.aeb
// CHECK:STDOUT:   %r.patt.loc22_59.2 => constants.%r.patt.892
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc22 => constants.%complete_type.357
// CHECK:STDOUT:   %Iterate.lookup_impl_witness => constants.%Iterate.impl_witness.58c
// CHECK:STDOUT:   %Iterate.facet.loc26_19.5 => constants.%Iterate.facet.1cb
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type => constants.%Iterate.WithSelf.NewCursor.type.b44
// CHECK:STDOUT:   %.loc26_19.23 => constants.%.de7
// CHECK:STDOUT:   %impl.elem2.loc26_19.3 => constants.%T.as_type.as.Iterate.impl.NewCursor.f24
// CHECK:STDOUT:   %specific_impl_fn.loc26_19.4 => constants.%T.as_type.as.Iterate.impl.NewCursor.specific_fn
// CHECK:STDOUT:   %impl.elem1 => constants.%Destroy.facet.027
// CHECK:STDOUT:   %as_type => constants.%tuple.type.dd0
// CHECK:STDOUT:   %require_complete.1 => constants.%complete_type.843
// CHECK:STDOUT:   %ptr => constants.%ptr.a4f
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type => constants.%Iterate.WithSelf.Next.type.a23
// CHECK:STDOUT:   %.loc26_19.24 => constants.%.802
// CHECK:STDOUT:   %impl.elem3.loc26_19.2 => constants.%T.as_type.as.Iterate.impl.Next.843
// CHECK:STDOUT:   %specific_impl_fn.loc26_19.5 => constants.%T.as_type.as.Iterate.impl.Next.specific_fn
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type => constants.%Destroy.WithSelf.SelfDestruct.type.956
// CHECK:STDOUT:   %.loc26_19.25 => constants.%.549
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.ed6
// CHECK:STDOUT:   %impl.elem2.loc26_19.4 => constants.%Destroy.WithSelf.SelfDestruct.cbb
// CHECK:STDOUT:   %specific_impl_fn.loc26_19.6 => constants.%Destroy.WithSelf.SelfDestruct.specific_fn.383
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- adl_returns_different_types.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
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
// CHECK:STDOUT:   %ValueType.type.facet: %type = facet_value %ValueType, () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.088: type = fn_type @Destroy.WithSelf.SubobjectDestroy, @Destroy.WithSelf(%ValueType.type.facet) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.527: %Destroy.WithSelf.SubobjectDestroy.type.088 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.73f: type = fn_type @Destroy.WithSelf.SelfDestruct, @Destroy.WithSelf(%ValueType.type.facet) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.299: %Destroy.WithSelf.SelfDestruct.type.73f = struct_value () [concrete]
// CHECK:STDOUT:   %ValueType.Op.type.4c0699.1: type = fn_type @ValueType.Op.1 [concrete]
// CHECK:STDOUT:   %ValueType.Op.4ec7f2.1: %ValueType.Op.type.4c0699.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.c48: <witness> = custom_witness (%ValueType.Op.4ec7f2.1, %Destroy.WithSelf.SubobjectDestroy.527, %Destroy.WithSelf.SelfDestruct.299), @Destroy [concrete]
// CHECK:STDOUT:   %const.dd6: type = const_type %ValueType [concrete]
// CHECK:STDOUT:   %ptr.69c: type = ptr_type %const.dd6 [concrete]
// CHECK:STDOUT:   %ptr.233: type = ptr_type %ValueType [concrete]
// CHECK:STDOUT:   %ValueType.Op.type.4c0699.2: type = fn_type @ValueType.Op.2 [concrete]
// CHECK:STDOUT:   %ValueType.Op.4ec7f2.2: %ValueType.Op.type.4c0699.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.202: <witness> = custom_witness (%ValueType.Op.4ec7f2.2), @Copy [concrete]
// CHECK:STDOUT:   %facet_value.d47: %facet_type.f48 = facet_value %ValueType, (%custom_witness.c48, %custom_witness.202) [concrete]
// CHECK:STDOUT:   %Iterate.lookup_impl_witness.1d8: <witness> = lookup_impl_witness %.Self.500, @Iterate [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.294: %facet_type.f48 = impl_witness_access %Iterate.lookup_impl_witness.1d8, element0 [symbolic_self]
// CHECK:STDOUT:   %Iterate_where.type.ba5: type = facet_type <@Iterate where %impl.elem0.294 = %facet_value.d47> [concrete]
// CHECK:STDOUT:   %pattern_type.80b: type = pattern_type %Iterate_where.type.ba5 [concrete]
// CHECK:STDOUT:   %R.patt: %pattern_type.80b = symbolic_binding_pattern R, 0 [symbolic]
// CHECK:STDOUT:   %R: %Iterate_where.type.ba5 = symbolic_binding R, 0 [symbolic]
// CHECK:STDOUT:   %R.as_type: type = facet_access_type %R [symbolic]
// CHECK:STDOUT:   %pattern_type.c6b: type = pattern_type %R.as_type [symbolic]
// CHECK:STDOUT:   %r.param_patt.5e7: %pattern_type.c6b = value_param_pattern [symbolic]
// CHECK:STDOUT:   %r.patt.3de: %pattern_type.c6b = wrapper_binding_pattern r, %r.param_patt.5e7 [symbolic]
// CHECK:STDOUT:   %i.patt: %pattern_type.c43 = value_binding_pattern i [concrete]
// CHECK:STDOUT:   %Iterate.lookup_impl_witness.5c5: <witness> = lookup_impl_witness %R, @Iterate [symbolic]
// CHECK:STDOUT:   %Iterate.facet.ac2: %Iterate.type = facet_value %R.as_type, (%Iterate.lookup_impl_witness.5c5) [symbolic]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.a61: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.ac2) [symbolic]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.0bf: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.ac2) [symbolic]
// CHECK:STDOUT:   %.c36: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.a61, %Iterate.facet.ac2 [symbolic]
// CHECK:STDOUT:   %impl.elem2.571: %.c36 = impl_witness_access %Iterate.lookup_impl_witness.5c5, element2 [symbolic]
// CHECK:STDOUT:   %impl.elem1.709: %Destroy.type = impl_witness_access %Iterate.lookup_impl_witness.5c5, element1 [symbolic]
// CHECK:STDOUT:   %as_type.1b1: type = facet_access_type %impl.elem1.709 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.a70: <specific function> = specific_impl_function %impl.elem2.571, @Iterate.WithSelf.NewCursor(%Iterate.facet.ac2) [symbolic]
// CHECK:STDOUT:   %ptr.8eb: type = ptr_type %as_type.1b1 [symbolic]
// CHECK:STDOUT:   %.1cdc: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.0bf, %Iterate.facet.ac2 [symbolic]
// CHECK:STDOUT:   %impl.elem3.4cb: %.1cdc = impl_witness_access %Iterate.lookup_impl_witness.5c5, element3 [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.888: type = class_type @DefaultOptionalStorage, @DefaultOptionalStorage(%facet_value.d47) [concrete]
// CHECK:STDOUT:   %OptionalStorage.impl_witness.c99: <witness> = impl_witness imports.%OptionalStorage.impl_witness_table.460, @T.as_type.as.OptionalStorage.impl(%facet_value.d47) [concrete]
// CHECK:STDOUT:   %OptionalStorage.facet.f89: %OptionalStorage.type = facet_value %ValueType, (%OptionalStorage.impl_witness.c99) [concrete]
// CHECK:STDOUT:   %Optional.6b7: type = class_type @Optional, @Optional(%OptionalStorage.facet.f89) [concrete]
// CHECK:STDOUT:   %specific_impl_fn.968: <specific function> = specific_impl_function %impl.elem3.4cb, @Iterate.WithSelf.Next(%Iterate.facet.ac2) [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.type.399: type = fn_type @Optional.HasValue, @Optional(%OptionalStorage.facet.f89) [concrete]
// CHECK:STDOUT:   %Optional.HasValue.408: %Optional.HasValue.type.399 = struct_value () [concrete]
// CHECK:STDOUT:   %Optional.Get.type.988: type = fn_type @Optional.Get, @Optional(%OptionalStorage.facet.f89) [concrete]
// CHECK:STDOUT:   %Optional.Get.148: %Optional.Get.type.988 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.facet.932: %Destroy.type = facet_value %ValueType, (%custom_witness.c48) [concrete]
// CHECK:STDOUT:   %MaybeUnformed.af1: type = class_type @MaybeUnformed, @MaybeUnformed(%Destroy.facet.932) [concrete]
// CHECK:STDOUT:   %.be1: type = maybe_unformed_type %ValueType [concrete]
// CHECK:STDOUT:   %struct_type.value.has_value.243: type = struct_type {.value: %MaybeUnformed.af1, .has_value: bool} [concrete]
// CHECK:STDOUT:   %Optional.HasValue.specific_fn.c7e: <specific function> = specific_function %Optional.HasValue.408, @Optional.HasValue(%OptionalStorage.facet.f89) [concrete]
// CHECK:STDOUT:   %Optional.Get.specific_fn.65b: <specific function> = specific_function %Optional.Get.148, @Optional.Get(%OptionalStorage.facet.f89) [concrete]
// CHECK:STDOUT:   %operator_PlusEqual__carbon_thunk.type: type = fn_type @operator_PlusEqual__carbon_thunk [concrete]
// CHECK:STDOUT:   %operator_PlusEqual__carbon_thunk: %operator_PlusEqual__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.7: type = fn_type @Destroy.Op.loc31_37.6 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.7: %Destroy.Op.type.1d8f74.7 = struct_value () [concrete]
// CHECK:STDOUT:   %Optional.type.facet: %type = facet_value %Optional.6b7, () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.559: type = fn_type @Destroy.WithSelf.SubobjectDestroy, @Destroy.WithSelf(%Optional.type.facet) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.9a9: %Destroy.WithSelf.SubobjectDestroy.type.559 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.1a3: type = fn_type @Destroy.WithSelf.SelfDestruct, @Destroy.WithSelf(%Optional.type.facet) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.e2c: %Destroy.WithSelf.SelfDestruct.type.1a3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.946: <witness> = custom_witness (%Destroy.Op.1a2547.7, %Destroy.WithSelf.SubobjectDestroy.9a9, %Destroy.WithSelf.SelfDestruct.e2c), @Destroy [concrete]
// CHECK:STDOUT:   %Destroy.facet.7a8: %Destroy.type = facet_value %Optional.6b7, (%custom_witness.946) [concrete]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.0e3: <witness> = lookup_impl_witness %impl.elem1.709, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.d94: type = fn_type @Destroy.WithSelf.SelfDestruct, @Destroy.WithSelf(%impl.elem1.709) [symbolic]
// CHECK:STDOUT:   %.bdd4: type = fn_type_with_self_type %Destroy.WithSelf.SelfDestruct.type.d94, %impl.elem1.709 [symbolic]
// CHECK:STDOUT:   %impl.elem2.5b9: %.bdd4 = impl_witness_access %Destroy.lookup_impl_witness.0e3, element2 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.745: <specific function> = specific_impl_function %impl.elem2.5b9, @Destroy.WithSelf.SelfDestruct(%impl.elem1.709) [symbolic]
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
// CHECK:STDOUT:   %Iterator.type.facet: %type = facet_value %Iterator, () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.64f: type = fn_type @Destroy.WithSelf.SubobjectDestroy, @Destroy.WithSelf(%Iterator.type.facet) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.988: %Destroy.WithSelf.SubobjectDestroy.type.64f = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.88b: type = fn_type @Destroy.WithSelf.SelfDestruct, @Destroy.WithSelf(%Iterator.type.facet) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.b41: %Destroy.WithSelf.SelfDestruct.type.88b = struct_value () [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.3f6dc2.2: type = fn_type @Iterator.Op.2 [concrete]
// CHECK:STDOUT:   %Iterator.Op.240e50.2: %Iterator.Op.type.3f6dc2.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.027: <witness> = custom_witness (%Iterator.Op.240e50.2, %Destroy.WithSelf.SubobjectDestroy.988, %Destroy.WithSelf.SelfDestruct.b41), @Destroy [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.3f6dc2.3: type = fn_type @Iterator.Op.3 [concrete]
// CHECK:STDOUT:   %Iterator.Op.240e50.3: %Iterator.Op.type.3f6dc2.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.c31: <witness> = custom_witness (%Iterator.Op.240e50.3), @Inc [concrete]
// CHECK:STDOUT:   %Sentinel.type.facet: %type = facet_value %Sentinel, () [concrete]
// CHECK:STDOUT:   %Equal.type: type = fn_type @Equal [concrete]
// CHECK:STDOUT:   %Equal: %Equal.type = struct_value () [concrete]
// CHECK:STDOUT:   %NotEqual.type: type = fn_type @NotEqual [concrete]
// CHECK:STDOUT:   %NotEqual: %NotEqual.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.e21: <witness> = custom_witness (%Equal, %NotEqual), @EqWith, @EqWith(%Sentinel) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.a9d: type = fn_type @Destroy.WithSelf.SubobjectDestroy, @Destroy.WithSelf(%Sentinel.type.facet) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.4e1: %Destroy.WithSelf.SubobjectDestroy.type.a9d = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.c52: type = fn_type @Destroy.WithSelf.SelfDestruct, @Destroy.WithSelf(%Sentinel.type.facet) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.e62: %Destroy.WithSelf.SelfDestruct.type.c52 = struct_value () [concrete]
// CHECK:STDOUT:   %Sentinel.Op.type: type = fn_type @Sentinel.Op [concrete]
// CHECK:STDOUT:   %Sentinel.Op: %Sentinel.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.876: <witness> = custom_witness (%Sentinel.Op, %Destroy.WithSelf.SubobjectDestroy.4e1, %Destroy.WithSelf.SelfDestruct.e62), @Destroy [concrete]
// CHECK:STDOUT:   %facet_value.26f: %CppRange_where.type.d682d6.1 = facet_value %ConstRange, (%custom_witness.c48, %custom_witness.202, %custom_witness.116, %custom_witness.027, %custom_witness.5be, %custom_witness.c31, %custom_witness.e21, %custom_witness.876) [concrete]
// CHECK:STDOUT:   %tuple.type.7a6: type = tuple_type (%Iterator, %Sentinel) [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.8: type = fn_type @Destroy.Op.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.8: %Destroy.Op.type.1d8f74.8 = struct_value () [concrete]
// CHECK:STDOUT:   %tuple.type.facet: %type = facet_value %tuple.type.7a6, () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.45f: type = fn_type @Destroy.WithSelf.SubobjectDestroy, @Destroy.WithSelf(%tuple.type.facet) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.faf: %Destroy.WithSelf.SubobjectDestroy.type.45f = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.7cb: type = fn_type @Destroy.WithSelf.SelfDestruct, @Destroy.WithSelf(%tuple.type.facet) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.c8b: %Destroy.WithSelf.SelfDestruct.type.7cb = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.c4a: <witness> = custom_witness (%Destroy.Op.1a2547.8, %Destroy.WithSelf.SubobjectDestroy.faf, %Destroy.WithSelf.SelfDestruct.c8b), @Destroy [concrete]
// CHECK:STDOUT:   %Destroy.facet.f63: %Destroy.type = facet_value %tuple.type.7a6, (%custom_witness.c4a) [concrete]
// CHECK:STDOUT:   %Iterate.impl_witness.b78: <witness> = impl_witness imports.%Iterate.impl_witness_table.121, @T.as_type.as.Iterate.impl(%facet_value.26f) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.type.e37: type = fn_type @T.as_type.as.Iterate.impl.NewCursor, @T.as_type.as.Iterate.impl(%facet_value.26f) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.81c: %T.as_type.as.Iterate.impl.NewCursor.type.e37 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.type.f1f: type = fn_type @T.as_type.as.Iterate.impl.Next, @T.as_type.as.Iterate.impl(%facet_value.26f) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.6fe: %T.as_type.as.Iterate.impl.Next.type.f1f = struct_value () [concrete]
// CHECK:STDOUT:   %Iterate.facet.726: %Iterate.type = facet_value %ConstRange, (%Iterate.impl_witness.b78) [concrete]
// CHECK:STDOUT:   %facet_value.205: %Iterate_where.type.ba5 = facet_value %ConstRange, (%Iterate.impl_witness.b78) [concrete]
// CHECK:STDOUT:   %r.param_patt.ac5: %pattern_type.c1d = value_param_pattern [concrete]
// CHECK:STDOUT:   %r.patt.97f: %pattern_type.c1d = wrapper_binding_pattern r, %r.param_patt.ac5 [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.02c: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.726) [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.815: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.726) [concrete]
// CHECK:STDOUT:   %.1e4: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.02c, %Iterate.facet.726 [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.specific_fn: <specific function> = specific_function %T.as_type.as.Iterate.impl.NewCursor.81c, @T.as_type.as.Iterate.impl.NewCursor(%facet_value.26f) [concrete]
// CHECK:STDOUT:   %ptr.d33: type = ptr_type %tuple.type.7a6 [concrete]
// CHECK:STDOUT:   %.2c4: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.815, %Iterate.facet.726 [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.specific_fn: <specific function> = specific_function %T.as_type.as.Iterate.impl.Next.6fe, @T.as_type.as.Iterate.impl.Next(%facet_value.26f) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.7cf: type = fn_type @Destroy.WithSelf.SelfDestruct, @Destroy.WithSelf(%Destroy.facet.f63) [concrete]
// CHECK:STDOUT:   %.842: type = fn_type_with_self_type %Destroy.WithSelf.SelfDestruct.type.7cf, %Destroy.facet.f63 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.specific_fn.1b1: <specific function> = specific_function %Destroy.WithSelf.SelfDestruct.c8b, @Destroy.WithSelf.SelfDestruct(%tuple.type.facet) [concrete]
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
// CHECK:STDOUT:   %operator_PlusEqual__carbon_thunk.decl: %operator_PlusEqual__carbon_thunk.type = fn_decl @operator_PlusEqual__carbon_thunk [concrete = constants.%operator_PlusEqual__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ConstRange.decl: type = class_decl @ConstRange [concrete = constants.%ConstRange] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @TestIterate(%R.loc27_17.2: %Iterate_where.type.ba5) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Iterate.lookup_impl_witness: <witness> = lookup_impl_witness %R.loc27_17.1, @Iterate [symbolic = %Iterate.lookup_impl_witness (constants.%Iterate.lookup_impl_witness.5c5)]
// CHECK:STDOUT:   %Iterate.facet.loc31_37.5: %Iterate.type = facet_value %R.as_type.loc27_79.1, (%Iterate.lookup_impl_witness) [symbolic = %Iterate.facet.loc31_37.5 (constants.%Iterate.facet.ac2)]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.loc31_37.5) [symbolic = %Iterate.WithSelf.NewCursor.type (constants.%Iterate.WithSelf.NewCursor.type.a61)]
// CHECK:STDOUT:   %.loc31_37.26: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type, %Iterate.facet.loc31_37.5 [symbolic = %.loc31_37.26 (constants.%.c36)]
// CHECK:STDOUT:   %impl.elem2.loc31_37.3: @TestIterate.%.loc31_37.26 (%.c36) = impl_witness_access %Iterate.lookup_impl_witness, element2 [symbolic = %impl.elem2.loc31_37.3 (constants.%impl.elem2.571)]
// CHECK:STDOUT:   %specific_impl_fn.loc31_37.4: <specific function> = specific_impl_function %impl.elem2.loc31_37.3, @Iterate.WithSelf.NewCursor(%Iterate.facet.loc31_37.5) [symbolic = %specific_impl_fn.loc31_37.4 (constants.%specific_impl_fn.a70)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %ptr: type = ptr_type %as_type [symbolic = %ptr (constants.%ptr.8eb)]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.loc31_37.5) [symbolic = %Iterate.WithSelf.Next.type (constants.%Iterate.WithSelf.Next.type.0bf)]
// CHECK:STDOUT:   %.loc31_37.27: type = fn_type_with_self_type %Iterate.WithSelf.Next.type, %Iterate.facet.loc31_37.5 [symbolic = %.loc31_37.27 (constants.%.1cdc)]
// CHECK:STDOUT:   %impl.elem3.loc31_37.2: @TestIterate.%.loc31_37.27 (%.1cdc) = impl_witness_access %Iterate.lookup_impl_witness, element3 [symbolic = %impl.elem3.loc31_37.2 (constants.%impl.elem3.4cb)]
// CHECK:STDOUT:   %specific_impl_fn.loc31_37.5: <specific function> = specific_impl_function %impl.elem3.loc31_37.2, @Iterate.WithSelf.Next(%Iterate.facet.loc31_37.5) [symbolic = %specific_impl_fn.loc31_37.5 (constants.%specific_impl_fn.968)]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type: type = fn_type @Destroy.WithSelf.SelfDestruct, @Destroy.WithSelf(%impl.elem1) [symbolic = %Destroy.WithSelf.SelfDestruct.type (constants.%Destroy.WithSelf.SelfDestruct.type.d94)]
// CHECK:STDOUT:   %.loc31_37.28: type = fn_type_with_self_type %Destroy.WithSelf.SelfDestruct.type, %impl.elem1 [symbolic = %.loc31_37.28 (constants.%.bdd4)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %impl.elem1, @Destroy [symbolic = %Destroy.lookup_impl_witness (constants.%Destroy.lookup_impl_witness.0e3)]
// CHECK:STDOUT:   %impl.elem2.loc31_37.4: @TestIterate.%.loc31_37.28 (%.bdd4) = impl_witness_access %Destroy.lookup_impl_witness, element2 [symbolic = %impl.elem2.loc31_37.4 (constants.%impl.elem2.5b9)]
// CHECK:STDOUT:   %specific_impl_fn.loc31_37.6: <specific function> = specific_impl_function %impl.elem2.loc31_37.4, @Destroy.WithSelf.SelfDestruct(%impl.elem1) [symbolic = %specific_impl_fn.loc31_37.6 (constants.%specific_impl_fn.745)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%r.param: @TestIterate.%R.as_type.loc27_79.1 (%R.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %i.patt: %pattern_type.c43 = value_binding_pattern i [concrete = constants.%i.patt]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %r.ref: @TestIterate.%R.as_type.loc27_79.1 (%R.as_type) = name_ref r, %r
// CHECK:STDOUT:     %impl.elem2.loc31_37.1: @TestIterate.%.loc31_37.26 (%.c36) = impl_witness_access constants.%Iterate.lookup_impl_witness.5c5, element2 [symbolic = %impl.elem2.loc31_37.3 (constants.%impl.elem2.571)]
// CHECK:STDOUT:     %bound_method.loc31_37.1: <bound method> = bound_method %r.ref, %impl.elem2.loc31_37.1
// CHECK:STDOUT:     %Iterate.facet.loc31_37.1: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.5c5) [symbolic = %Iterate.facet.loc31_37.5 (constants.%Iterate.facet.ac2)]
// CHECK:STDOUT:     %.loc31_37.1: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc31_37.1 [symbolic = %Iterate.facet.loc31_37.5 (constants.%Iterate.facet.ac2)]
// CHECK:STDOUT:     %Iterate.facet.loc31_37.2: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.5c5) [symbolic = %Iterate.facet.loc31_37.5 (constants.%Iterate.facet.ac2)]
// CHECK:STDOUT:     %.loc31_37.2: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc31_37.2 [symbolic = %Iterate.facet.loc31_37.5 (constants.%Iterate.facet.ac2)]
// CHECK:STDOUT:     %specific_impl_fn.loc31_37.1: <specific function> = specific_impl_function %impl.elem2.loc31_37.1, @Iterate.WithSelf.NewCursor(constants.%Iterate.facet.ac2) [symbolic = %specific_impl_fn.loc31_37.4 (constants.%specific_impl_fn.a70)]
// CHECK:STDOUT:     %bound_method.loc31_37.2: <bound method> = bound_method %r.ref, %specific_impl_fn.loc31_37.1
// CHECK:STDOUT:     %var: ref @TestIterate.%as_type (%as_type.1b1) = var_storage invalid
// CHECK:STDOUT:     %Iterate.WithSelf.NewCursor.call: init @TestIterate.%as_type (%as_type.1b1) to %var = call %bound_method.loc31_37.2(%r.ref)
// CHECK:STDOUT:     assign %var, %Iterate.WithSelf.NewCursor.call
// CHECK:STDOUT:     br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.next:
// CHECK:STDOUT:     %addr.loc31: @TestIterate.%ptr (%ptr.8eb) = addr_of %var
// CHECK:STDOUT:     %impl.elem3.loc31_37.1: @TestIterate.%.loc31_37.27 (%.1cdc) = impl_witness_access constants.%Iterate.lookup_impl_witness.5c5, element3 [symbolic = %impl.elem3.loc31_37.2 (constants.%impl.elem3.4cb)]
// CHECK:STDOUT:     %bound_method.loc31_37.3: <bound method> = bound_method %r.ref, %impl.elem3.loc31_37.1
// CHECK:STDOUT:     %Iterate.facet.loc31_37.3: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.5c5) [symbolic = %Iterate.facet.loc31_37.5 (constants.%Iterate.facet.ac2)]
// CHECK:STDOUT:     %.loc31_37.3: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc31_37.3 [symbolic = %Iterate.facet.loc31_37.5 (constants.%Iterate.facet.ac2)]
// CHECK:STDOUT:     %Iterate.facet.loc31_37.4: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.5c5) [symbolic = %Iterate.facet.loc31_37.5 (constants.%Iterate.facet.ac2)]
// CHECK:STDOUT:     %.loc31_37.4: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc31_37.4 [symbolic = %Iterate.facet.loc31_37.5 (constants.%Iterate.facet.ac2)]
// CHECK:STDOUT:     %.loc31_37.5: %Destroy.type = converted constants.%as_type.1b1, constants.%impl.elem1.709 [symbolic = %impl.elem1 (constants.%impl.elem1.709)]
// CHECK:STDOUT:     %.loc31_37.6: %Destroy.type = converted constants.%as_type.1b1, constants.%impl.elem1.709 [symbolic = %impl.elem1 (constants.%impl.elem1.709)]
// CHECK:STDOUT:     %specific_impl_fn.loc31_37.2: <specific function> = specific_impl_function %impl.elem3.loc31_37.1, @Iterate.WithSelf.Next(constants.%Iterate.facet.ac2) [symbolic = %specific_impl_fn.loc31_37.5 (constants.%specific_impl_fn.968)]
// CHECK:STDOUT:     %bound_method.loc31_37.4: <bound method> = bound_method %r.ref, %specific_impl_fn.loc31_37.2
// CHECK:STDOUT:     %.loc31_37.7: ref %Optional.6b7 = temporary_storage
// CHECK:STDOUT:     %Iterate.WithSelf.Next.call: init %Optional.6b7 to %.loc31_37.7 = call %bound_method.loc31_37.4(%r.ref, %addr.loc31)
// CHECK:STDOUT:     %.loc31_37.8: ref %Optional.6b7 = temporary %.loc31_37.7, %Iterate.WithSelf.Next.call
// CHECK:STDOUT:     %.loc31_37.9: %Optional.HasValue.type.399 = specific_constant imports.%Core.import_ref.bb6, @Optional(constants.%OptionalStorage.facet.f89) [concrete = constants.%Optional.HasValue.408]
// CHECK:STDOUT:     %HasValue.ref: %Optional.HasValue.type.399 = name_ref HasValue, %.loc31_37.9 [concrete = constants.%Optional.HasValue.408]
// CHECK:STDOUT:     %Optional.HasValue.bound: <bound method> = bound_method %.loc31_37.8, %HasValue.ref
// CHECK:STDOUT:     %Optional.HasValue.specific_fn: <specific function> = specific_function %HasValue.ref, @Optional.HasValue(constants.%OptionalStorage.facet.f89) [concrete = constants.%Optional.HasValue.specific_fn.c7e]
// CHECK:STDOUT:     %bound_method.loc31_37.5: <bound method> = bound_method %.loc31_37.8, %Optional.HasValue.specific_fn
// CHECK:STDOUT:     %.loc31_37.10: %Optional.6b7 = acquire_value %.loc31_37.8
// CHECK:STDOUT:     %Optional.HasValue.call: init bool = call %bound_method.loc31_37.5(%.loc31_37.10)
// CHECK:STDOUT:     %.loc31_37.11: bool = value_of_initializer %Optional.HasValue.call
// CHECK:STDOUT:     %.loc31_37.12: bool = converted %Optional.HasValue.call, %.loc31_37.11
// CHECK:STDOUT:     if %.loc31_37.12 br !for.body else br !for.done
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.body:
// CHECK:STDOUT:     %.loc31_37.13: %Optional.Get.type.988 = specific_constant imports.%Core.import_ref.dab, @Optional(constants.%OptionalStorage.facet.f89) [concrete = constants.%Optional.Get.148]
// CHECK:STDOUT:     %Get.ref: %Optional.Get.type.988 = name_ref Get, %.loc31_37.13 [concrete = constants.%Optional.Get.148]
// CHECK:STDOUT:     %Optional.Get.bound: <bound method> = bound_method %.loc31_37.8, %Get.ref
// CHECK:STDOUT:     %Optional.Get.specific_fn: <specific function> = specific_function %Get.ref, @Optional.Get(constants.%OptionalStorage.facet.f89) [concrete = constants.%Optional.Get.specific_fn.65b]
// CHECK:STDOUT:     %bound_method.loc31_37.6: <bound method> = bound_method %.loc31_37.8, %Optional.Get.specific_fn
// CHECK:STDOUT:     %.loc31_37.14: ref %ValueType = temporary_storage
// CHECK:STDOUT:     %.loc31_37.15: %Optional.6b7 = acquire_value %.loc31_37.8
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
// CHECK:STDOUT:     %Destroy.WithSelf.SelfDestruct.bound.loc31_37.1: <bound method> = bound_method %.loc31_37.16, constants.%Destroy.WithSelf.SelfDestruct.299
// CHECK:STDOUT:     %Destroy.facet.loc31_37.1: %Destroy.type = facet_value constants.%ValueType, (constants.%custom_witness.c48) [concrete = constants.%Destroy.facet.932]
// CHECK:STDOUT:     %.loc31_37.18: %Destroy.type = converted constants.%ValueType, %Destroy.facet.loc31_37.1 [concrete = constants.%Destroy.facet.932]
// CHECK:STDOUT:     %Destroy.facet.loc31_37.2: %Destroy.type = facet_value constants.%ValueType, (constants.%custom_witness.c48) [concrete = constants.%Destroy.facet.932]
// CHECK:STDOUT:     %.loc31_37.19: %Destroy.type = converted constants.%ValueType, %Destroy.facet.loc31_37.2 [concrete = constants.%Destroy.facet.932]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %bound_method.loc31_37.7: <bound method> = bound_method %.loc31_37.16, %Destroy.WithSelf.SelfDestruct.specific_fn.1
// CHECK:STDOUT:     %Destroy.WithSelf.SelfDestruct.call.loc31_37.1: init %empty_tuple.type = call %bound_method.loc31_37.7(%.loc31_37.16)
// CHECK:STDOUT:     %Destroy.WithSelf.SelfDestruct.bound.loc31_37.2: <bound method> = bound_method %.loc31_37.8, constants.%Destroy.WithSelf.SelfDestruct.e2c
// CHECK:STDOUT:     %Destroy.facet.loc31_37.3: %Destroy.type = facet_value constants.%Optional.6b7, (constants.%custom_witness.946) [concrete = constants.%Destroy.facet.7a8]
// CHECK:STDOUT:     %.loc31_37.20: %Destroy.type = converted constants.%Optional.6b7, %Destroy.facet.loc31_37.3 [concrete = constants.%Destroy.facet.7a8]
// CHECK:STDOUT:     %Destroy.facet.loc31_37.4: %Destroy.type = facet_value constants.%Optional.6b7, (constants.%custom_witness.946) [concrete = constants.%Destroy.facet.7a8]
// CHECK:STDOUT:     %.loc31_37.21: %Destroy.type = converted constants.%Optional.6b7, %Destroy.facet.loc31_37.4 [concrete = constants.%Destroy.facet.7a8]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %bound_method.loc31_37.8: <bound method> = bound_method %.loc31_37.8, %Destroy.WithSelf.SelfDestruct.specific_fn.2
// CHECK:STDOUT:     %Destroy.WithSelf.SelfDestruct.call.loc31_37.2: init %empty_tuple.type = call %bound_method.loc31_37.8(%.loc31_37.8)
// CHECK:STDOUT:     br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.done:
// CHECK:STDOUT:     %Destroy.WithSelf.SelfDestruct.bound.loc31_37.3: <bound method> = bound_method %.loc31_37.8, constants.%Destroy.WithSelf.SelfDestruct.e2c
// CHECK:STDOUT:     %Destroy.facet.loc31_37.5: %Destroy.type = facet_value constants.%Optional.6b7, (constants.%custom_witness.946) [concrete = constants.%Destroy.facet.7a8]
// CHECK:STDOUT:     %.loc31_37.22: %Destroy.type = converted constants.%Optional.6b7, %Destroy.facet.loc31_37.5 [concrete = constants.%Destroy.facet.7a8]
// CHECK:STDOUT:     %Destroy.facet.loc31_37.6: %Destroy.type = facet_value constants.%Optional.6b7, (constants.%custom_witness.946) [concrete = constants.%Destroy.facet.7a8]
// CHECK:STDOUT:     %.loc31_37.23: %Destroy.type = converted constants.%Optional.6b7, %Destroy.facet.loc31_37.6 [concrete = constants.%Destroy.facet.7a8]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %bound_method.loc31_37.9: <bound method> = bound_method %.loc31_37.8, %Destroy.WithSelf.SelfDestruct.specific_fn.3
// CHECK:STDOUT:     %Destroy.WithSelf.SelfDestruct.call.loc31_37.3: init %empty_tuple.type = call %bound_method.loc31_37.9(%.loc31_37.8)
// CHECK:STDOUT:     %impl.elem2.loc31_37.2: @TestIterate.%.loc31_37.28 (%.bdd4) = impl_witness_access constants.%Destroy.lookup_impl_witness.0e3, element2 [symbolic = %impl.elem2.loc31_37.4 (constants.%impl.elem2.5b9)]
// CHECK:STDOUT:     %bound_method.loc31_37.10: <bound method> = bound_method %var, %impl.elem2.loc31_37.2
// CHECK:STDOUT:     %.loc31_37.24: %Destroy.type = converted constants.%as_type.1b1, constants.%impl.elem1.709 [symbolic = %impl.elem1 (constants.%impl.elem1.709)]
// CHECK:STDOUT:     %.loc31_37.25: %Destroy.type = converted constants.%as_type.1b1, constants.%impl.elem1.709 [symbolic = %impl.elem1 (constants.%impl.elem1.709)]
// CHECK:STDOUT:     %specific_impl_fn.loc31_37.3: <specific function> = specific_impl_function %impl.elem2.loc31_37.2, @Destroy.WithSelf.SelfDestruct(constants.%impl.elem1.709) [symbolic = %specific_impl_fn.loc31_37.6 (constants.%specific_impl_fn.745)]
// CHECK:STDOUT:     %bound_method.loc31_37.11: <bound method> = bound_method %var, %specific_impl_fn.loc31_37.3
// CHECK:STDOUT:     %Destroy.WithSelf.SelfDestruct.call.loc31_37.4: init %empty_tuple.type = call %bound_method.loc31_37.11(%var)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc31_37.1(%self.param: ref %.be1) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc31_37.2(%self.param: ref %MaybeUnformed.af1) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc31_37.3(%self.param: ref bool) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc31_37.4(%self.param: ref %struct_type.value.has_value.243) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc31_37.5(%self.param: ref %DefaultOptionalStorage.888) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc31_37.6(%self.param: ref %Optional.6b7) {
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
// CHECK:STDOUT:   %impl.elem2: %.1e4 = impl_witness_access constants.%Iterate.impl_witness.b78, element2 [concrete = constants.%T.as_type.as.Iterate.impl.NewCursor.81c]
// CHECK:STDOUT:   %bound_method.loc45_37.1: <bound method> = bound_method %c.ref, %impl.elem2
// CHECK:STDOUT:   %facet_value.loc45_37.1: %CppRange_where.type.d682d6.1 = facet_value constants.%ConstRange, (constants.%custom_witness.c48, constants.%custom_witness.202, constants.%custom_witness.116, constants.%custom_witness.027, constants.%custom_witness.5be, constants.%custom_witness.c31, constants.%custom_witness.e21, constants.%custom_witness.876) [concrete = constants.%facet_value.26f]
// CHECK:STDOUT:   %.loc45_37.1: %CppRange_where.type.d682d6.1 = converted constants.%ConstRange, %facet_value.loc45_37.1 [concrete = constants.%facet_value.26f]
// CHECK:STDOUT:   %facet_value.loc45_37.2: %CppRange_where.type.d682d6.1 = facet_value constants.%ConstRange, (constants.%custom_witness.c48, constants.%custom_witness.202, constants.%custom_witness.116, constants.%custom_witness.027, constants.%custom_witness.5be, constants.%custom_witness.c31, constants.%custom_witness.e21, constants.%custom_witness.876) [concrete = constants.%facet_value.26f]
// CHECK:STDOUT:   %.loc45_37.2: %CppRange_where.type.d682d6.1 = converted constants.%ConstRange, %facet_value.loc45_37.2 [concrete = constants.%facet_value.26f]
// CHECK:STDOUT:   %specific_fn.loc45_37.1: <specific function> = specific_function %impl.elem2, @T.as_type.as.Iterate.impl.NewCursor(constants.%facet_value.26f) [concrete = constants.%T.as_type.as.Iterate.impl.NewCursor.specific_fn]
// CHECK:STDOUT:   %bound_method.loc45_37.2: <bound method> = bound_method %c.ref, %specific_fn.loc45_37.1
// CHECK:STDOUT:   %var: ref %tuple.type.7a6 = var_storage invalid
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.call: init %tuple.type.7a6 to %var = call %bound_method.loc45_37.2(%c.ref)
// CHECK:STDOUT:   assign %var, %T.as_type.as.Iterate.impl.NewCursor.call
// CHECK:STDOUT:   br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT: !for.next:
// CHECK:STDOUT:   %addr.loc45: %ptr.d33 = addr_of %var
// CHECK:STDOUT:   %impl.elem3: %.2c4 = impl_witness_access constants.%Iterate.impl_witness.b78, element3 [concrete = constants.%T.as_type.as.Iterate.impl.Next.6fe]
// CHECK:STDOUT:   %bound_method.loc45_37.3: <bound method> = bound_method %c.ref, %impl.elem3
// CHECK:STDOUT:   %facet_value.loc45_37.3: %CppRange_where.type.d682d6.1 = facet_value constants.%ConstRange, (constants.%custom_witness.c48, constants.%custom_witness.202, constants.%custom_witness.116, constants.%custom_witness.027, constants.%custom_witness.5be, constants.%custom_witness.c31, constants.%custom_witness.e21, constants.%custom_witness.876) [concrete = constants.%facet_value.26f]
// CHECK:STDOUT:   %.loc45_37.3: %CppRange_where.type.d682d6.1 = converted constants.%ConstRange, %facet_value.loc45_37.3 [concrete = constants.%facet_value.26f]
// CHECK:STDOUT:   %facet_value.loc45_37.4: %CppRange_where.type.d682d6.1 = facet_value constants.%ConstRange, (constants.%custom_witness.c48, constants.%custom_witness.202, constants.%custom_witness.116, constants.%custom_witness.027, constants.%custom_witness.5be, constants.%custom_witness.c31, constants.%custom_witness.e21, constants.%custom_witness.876) [concrete = constants.%facet_value.26f]
// CHECK:STDOUT:   %.loc45_37.4: %CppRange_where.type.d682d6.1 = converted constants.%ConstRange, %facet_value.loc45_37.4 [concrete = constants.%facet_value.26f]
// CHECK:STDOUT:   %specific_fn.loc45_37.2: <specific function> = specific_function %impl.elem3, @T.as_type.as.Iterate.impl.Next(constants.%facet_value.26f) [concrete = constants.%T.as_type.as.Iterate.impl.Next.specific_fn]
// CHECK:STDOUT:   %bound_method.loc45_37.4: <bound method> = bound_method %c.ref, %specific_fn.loc45_37.2
// CHECK:STDOUT:   %.loc45_37.5: ref %Optional.6b7 = temporary_storage
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.call: init %Optional.6b7 to %.loc45_37.5 = call %bound_method.loc45_37.4(%c.ref, %addr.loc45)
// CHECK:STDOUT:   %.loc45_37.6: ref %Optional.6b7 = temporary %.loc45_37.5, %T.as_type.as.Iterate.impl.Next.call
// CHECK:STDOUT:   %.loc45_37.7: %Optional.HasValue.type.399 = specific_constant imports.%Core.import_ref.bb6, @Optional(constants.%OptionalStorage.facet.f89) [concrete = constants.%Optional.HasValue.408]
// CHECK:STDOUT:   %HasValue.ref: %Optional.HasValue.type.399 = name_ref HasValue, %.loc45_37.7 [concrete = constants.%Optional.HasValue.408]
// CHECK:STDOUT:   %Optional.HasValue.bound: <bound method> = bound_method %.loc45_37.6, %HasValue.ref
// CHECK:STDOUT:   %Optional.HasValue.specific_fn: <specific function> = specific_function %HasValue.ref, @Optional.HasValue(constants.%OptionalStorage.facet.f89) [concrete = constants.%Optional.HasValue.specific_fn.c7e]
// CHECK:STDOUT:   %bound_method.loc45_37.5: <bound method> = bound_method %.loc45_37.6, %Optional.HasValue.specific_fn
// CHECK:STDOUT:   %.loc45_37.8: %Optional.6b7 = acquire_value %.loc45_37.6
// CHECK:STDOUT:   %Optional.HasValue.call: init bool = call %bound_method.loc45_37.5(%.loc45_37.8)
// CHECK:STDOUT:   %.loc45_37.9: bool = value_of_initializer %Optional.HasValue.call
// CHECK:STDOUT:   %.loc45_37.10: bool = converted %Optional.HasValue.call, %.loc45_37.9
// CHECK:STDOUT:   if %.loc45_37.10 br !for.body else br !for.done
// CHECK:STDOUT:
// CHECK:STDOUT: !for.body:
// CHECK:STDOUT:   %.loc45_37.11: %Optional.Get.type.988 = specific_constant imports.%Core.import_ref.dab, @Optional(constants.%OptionalStorage.facet.f89) [concrete = constants.%Optional.Get.148]
// CHECK:STDOUT:   %Get.ref: %Optional.Get.type.988 = name_ref Get, %.loc45_37.11 [concrete = constants.%Optional.Get.148]
// CHECK:STDOUT:   %Optional.Get.bound: <bound method> = bound_method %.loc45_37.6, %Get.ref
// CHECK:STDOUT:   %Optional.Get.specific_fn: <specific function> = specific_function %Get.ref, @Optional.Get(constants.%OptionalStorage.facet.f89) [concrete = constants.%Optional.Get.specific_fn.65b]
// CHECK:STDOUT:   %bound_method.loc45_37.6: <bound method> = bound_method %.loc45_37.6, %Optional.Get.specific_fn
// CHECK:STDOUT:   %.loc45_37.12: ref %ValueType = temporary_storage
// CHECK:STDOUT:   %.loc45_37.13: %Optional.6b7 = acquire_value %.loc45_37.6
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
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc45_37.1: <bound method> = bound_method %.loc45_37.14, constants.%Destroy.WithSelf.SelfDestruct.299
// CHECK:STDOUT:   %Destroy.facet.loc45_37.1: %Destroy.type = facet_value constants.%ValueType, (constants.%custom_witness.c48) [concrete = constants.%Destroy.facet.932]
// CHECK:STDOUT:   %.loc45_37.16: %Destroy.type = converted constants.%ValueType, %Destroy.facet.loc45_37.1 [concrete = constants.%Destroy.facet.932]
// CHECK:STDOUT:   %Destroy.facet.loc45_37.2: %Destroy.type = facet_value constants.%ValueType, (constants.%custom_witness.c48) [concrete = constants.%Destroy.facet.932]
// CHECK:STDOUT:   %.loc45_37.17: %Destroy.type = converted constants.%ValueType, %Destroy.facet.loc45_37.2 [concrete = constants.%Destroy.facet.932]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc45_37.7: <bound method> = bound_method %.loc45_37.14, %Destroy.WithSelf.SelfDestruct.specific_fn.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc45_37.1: init %empty_tuple.type = call %bound_method.loc45_37.7(%.loc45_37.14)
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc45_37.2: <bound method> = bound_method %.loc45_37.6, constants.%Destroy.WithSelf.SelfDestruct.e2c
// CHECK:STDOUT:   %Destroy.facet.loc45_37.3: %Destroy.type = facet_value constants.%Optional.6b7, (constants.%custom_witness.946) [concrete = constants.%Destroy.facet.7a8]
// CHECK:STDOUT:   %.loc45_37.18: %Destroy.type = converted constants.%Optional.6b7, %Destroy.facet.loc45_37.3 [concrete = constants.%Destroy.facet.7a8]
// CHECK:STDOUT:   %Destroy.facet.loc45_37.4: %Destroy.type = facet_value constants.%Optional.6b7, (constants.%custom_witness.946) [concrete = constants.%Destroy.facet.7a8]
// CHECK:STDOUT:   %.loc45_37.19: %Destroy.type = converted constants.%Optional.6b7, %Destroy.facet.loc45_37.4 [concrete = constants.%Destroy.facet.7a8]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc45_37.8: <bound method> = bound_method %.loc45_37.6, %Destroy.WithSelf.SelfDestruct.specific_fn.2
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc45_37.2: init %empty_tuple.type = call %bound_method.loc45_37.8(%.loc45_37.6)
// CHECK:STDOUT:   br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT: !for.done:
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc45_37.3: <bound method> = bound_method %.loc45_37.6, constants.%Destroy.WithSelf.SelfDestruct.e2c
// CHECK:STDOUT:   %Destroy.facet.loc45_37.5: %Destroy.type = facet_value constants.%Optional.6b7, (constants.%custom_witness.946) [concrete = constants.%Destroy.facet.7a8]
// CHECK:STDOUT:   %.loc45_37.20: %Destroy.type = converted constants.%Optional.6b7, %Destroy.facet.loc45_37.5 [concrete = constants.%Destroy.facet.7a8]
// CHECK:STDOUT:   %Destroy.facet.loc45_37.6: %Destroy.type = facet_value constants.%Optional.6b7, (constants.%custom_witness.946) [concrete = constants.%Destroy.facet.7a8]
// CHECK:STDOUT:   %.loc45_37.21: %Destroy.type = converted constants.%Optional.6b7, %Destroy.facet.loc45_37.6 [concrete = constants.%Destroy.facet.7a8]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc45_37.9: <bound method> = bound_method %.loc45_37.6, %Destroy.WithSelf.SelfDestruct.specific_fn.3
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc45_37.3: init %empty_tuple.type = call %bound_method.loc45_37.9(%.loc45_37.6)
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc45_37.4: <bound method> = bound_method %var, constants.%Destroy.WithSelf.SelfDestruct.c8b
// CHECK:STDOUT:   %Destroy.facet.loc45_37.7: %Destroy.type = facet_value constants.%tuple.type.7a6, (constants.%custom_witness.c4a) [concrete = constants.%Destroy.facet.f63]
// CHECK:STDOUT:   %.loc45_37.22: %Destroy.type = converted constants.%tuple.type.7a6, %Destroy.facet.loc45_37.7 [concrete = constants.%Destroy.facet.f63]
// CHECK:STDOUT:   %Destroy.facet.loc45_37.8: %Destroy.type = facet_value constants.%tuple.type.7a6, (constants.%custom_witness.c4a) [concrete = constants.%Destroy.facet.f63]
// CHECK:STDOUT:   %.loc45_37.23: %Destroy.type = converted constants.%tuple.type.7a6, %Destroy.facet.loc45_37.8 [concrete = constants.%Destroy.facet.f63]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc45_37.10: <bound method> = bound_method %var, %Destroy.WithSelf.SelfDestruct.specific_fn.4
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc45_37.4: init %empty_tuple.type = call %bound_method.loc45_37.10(%var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestIterate(constants.%R) {
// CHECK:STDOUT:   %R.patt.loc27_17.2 => constants.%R.patt
// CHECK:STDOUT:   %R.loc27_17.1 => constants.%R
// CHECK:STDOUT:   %R.as_type.loc27_79.1 => constants.%R.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.c6b
// CHECK:STDOUT:   %r.param_patt.loc27_77.2 => constants.%r.param_patt.5e7
// CHECK:STDOUT:   %r.patt.loc27_77.2 => constants.%r.patt.3de
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestIterate(constants.%facet_value.205) {
// CHECK:STDOUT:   %R.patt.loc27_17.2 => constants.%R.patt
// CHECK:STDOUT:   %R.loc27_17.1 => constants.%facet_value.205
// CHECK:STDOUT:   %R.as_type.loc27_79.1 => constants.%ConstRange
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.c1d
// CHECK:STDOUT:   %r.param_patt.loc27_77.2 => constants.%r.param_patt.ac5
// CHECK:STDOUT:   %r.patt.loc27_77.2 => constants.%r.patt.97f
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc27 => constants.%complete_type.357
// CHECK:STDOUT:   %Iterate.lookup_impl_witness => constants.%Iterate.impl_witness.b78
// CHECK:STDOUT:   %Iterate.facet.loc31_37.5 => constants.%Iterate.facet.726
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type => constants.%Iterate.WithSelf.NewCursor.type.02c
// CHECK:STDOUT:   %.loc31_37.26 => constants.%.1e4
// CHECK:STDOUT:   %impl.elem2.loc31_37.3 => constants.%T.as_type.as.Iterate.impl.NewCursor.81c
// CHECK:STDOUT:   %specific_impl_fn.loc31_37.4 => constants.%T.as_type.as.Iterate.impl.NewCursor.specific_fn
// CHECK:STDOUT:   %impl.elem1 => constants.%Destroy.facet.f63
// CHECK:STDOUT:   %as_type => constants.%tuple.type.7a6
// CHECK:STDOUT:   %require_complete.1 => constants.%complete_type.cf0
// CHECK:STDOUT:   %ptr => constants.%ptr.d33
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type => constants.%Iterate.WithSelf.Next.type.815
// CHECK:STDOUT:   %.loc31_37.27 => constants.%.2c4
// CHECK:STDOUT:   %impl.elem3.loc31_37.2 => constants.%T.as_type.as.Iterate.impl.Next.6fe
// CHECK:STDOUT:   %specific_impl_fn.loc31_37.5 => constants.%T.as_type.as.Iterate.impl.Next.specific_fn
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type => constants.%Destroy.WithSelf.SelfDestruct.type.7cf
// CHECK:STDOUT:   %.loc31_37.28 => constants.%.842
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.c4a
// CHECK:STDOUT:   %impl.elem2.loc31_37.4 => constants.%Destroy.WithSelf.SelfDestruct.c8b
// CHECK:STDOUT:   %specific_impl_fn.loc31_37.6 => constants.%Destroy.WithSelf.SelfDestruct.specific_fn.1b1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_adl_returns_different_types_mutable.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
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
// CHECK:STDOUT:   %ValueType.type.facet: %type = facet_value %ValueType, () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.07f: type = fn_type @Destroy.WithSelf.SubobjectDestroy, @Destroy.WithSelf(%ValueType.type.facet) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.861: %Destroy.WithSelf.SubobjectDestroy.type.07f = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.ba6: type = fn_type @Destroy.WithSelf.SelfDestruct, @Destroy.WithSelf(%ValueType.type.facet) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.000: %Destroy.WithSelf.SelfDestruct.type.ba6 = struct_value () [concrete]
// CHECK:STDOUT:   %ValueType.Op.type.db6aa9.1: type = fn_type @ValueType.Op.1 [concrete]
// CHECK:STDOUT:   %ValueType.Op.4590ed.1: %ValueType.Op.type.db6aa9.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.563: <witness> = custom_witness (%ValueType.Op.4590ed.1, %Destroy.WithSelf.SubobjectDestroy.861, %Destroy.WithSelf.SelfDestruct.000), @Destroy [concrete]
// CHECK:STDOUT:   %const.91f: type = const_type %ValueType [concrete]
// CHECK:STDOUT:   %ptr.5f4: type = ptr_type %const.91f [concrete]
// CHECK:STDOUT:   %ptr.9c6: type = ptr_type %ValueType [concrete]
// CHECK:STDOUT:   %ValueType.Op.type.db6aa9.2: type = fn_type @ValueType.Op.2 [concrete]
// CHECK:STDOUT:   %ValueType.Op.4590ed.2: %ValueType.Op.type.db6aa9.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.587: <witness> = custom_witness (%ValueType.Op.4590ed.2), @Copy [concrete]
// CHECK:STDOUT:   %facet_value.ddf: %facet_type.f48 = facet_value %ValueType, (%custom_witness.563, %custom_witness.587) [concrete]
// CHECK:STDOUT:   %Iterate.lookup_impl_witness.1d8: <witness> = lookup_impl_witness %.Self.500, @Iterate [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.294: %facet_type.f48 = impl_witness_access %Iterate.lookup_impl_witness.1d8, element0 [symbolic_self]
// CHECK:STDOUT:   %Iterate_where.type.a2e: type = facet_type <@Iterate where %impl.elem0.294 = %facet_value.ddf> [concrete]
// CHECK:STDOUT:   %pattern_type.748: type = pattern_type %Iterate_where.type.a2e [concrete]
// CHECK:STDOUT:   %R.patt: %pattern_type.748 = symbolic_binding_pattern R, 0 [symbolic]
// CHECK:STDOUT:   %R: %Iterate_where.type.a2e = symbolic_binding R, 0 [symbolic]
// CHECK:STDOUT:   %R.as_type: type = facet_access_type %R [symbolic]
// CHECK:STDOUT:   %pattern_type.33d: type = pattern_type %R.as_type [symbolic]
// CHECK:STDOUT:   %r.param_patt: %pattern_type.33d = value_param_pattern [symbolic]
// CHECK:STDOUT:   %r.patt: %pattern_type.33d = wrapper_binding_pattern r, %r.param_patt [symbolic]
// CHECK:STDOUT:   %i.patt: %pattern_type.d2d = value_binding_pattern i [concrete]
// CHECK:STDOUT:   %Iterate.lookup_impl_witness.bab: <witness> = lookup_impl_witness %R, @Iterate [symbolic]
// CHECK:STDOUT:   %Iterate.facet.323: %Iterate.type = facet_value %R.as_type, (%Iterate.lookup_impl_witness.bab) [symbolic]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.c53: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.323) [symbolic]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.094: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.323) [symbolic]
// CHECK:STDOUT:   %.28f: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.c53, %Iterate.facet.323 [symbolic]
// CHECK:STDOUT:   %impl.elem2.ffb: %.28f = impl_witness_access %Iterate.lookup_impl_witness.bab, element2 [symbolic]
// CHECK:STDOUT:   %impl.elem1.5ca: %Destroy.type = impl_witness_access %Iterate.lookup_impl_witness.bab, element1 [symbolic]
// CHECK:STDOUT:   %as_type.621: type = facet_access_type %impl.elem1.5ca [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.79e: <specific function> = specific_impl_function %impl.elem2.ffb, @Iterate.WithSelf.NewCursor(%Iterate.facet.323) [symbolic]
// CHECK:STDOUT:   %ptr.dcb: type = ptr_type %as_type.621 [symbolic]
// CHECK:STDOUT:   %.4f3: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.094, %Iterate.facet.323 [symbolic]
// CHECK:STDOUT:   %impl.elem3.a3f: %.4f3 = impl_witness_access %Iterate.lookup_impl_witness.bab, element3 [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.7fe: type = class_type @DefaultOptionalStorage, @DefaultOptionalStorage(%facet_value.ddf) [concrete]
// CHECK:STDOUT:   %OptionalStorage.impl_witness.814: <witness> = impl_witness imports.%OptionalStorage.impl_witness_table.460, @T.as_type.as.OptionalStorage.impl(%facet_value.ddf) [concrete]
// CHECK:STDOUT:   %OptionalStorage.facet.8cc: %OptionalStorage.type = facet_value %ValueType, (%OptionalStorage.impl_witness.814) [concrete]
// CHECK:STDOUT:   %Optional.69a: type = class_type @Optional, @Optional(%OptionalStorage.facet.8cc) [concrete]
// CHECK:STDOUT:   %specific_impl_fn.6d9: <specific function> = specific_impl_function %impl.elem3.a3f, @Iterate.WithSelf.Next(%Iterate.facet.323) [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.type.cfd: type = fn_type @Optional.HasValue, @Optional(%OptionalStorage.facet.8cc) [concrete]
// CHECK:STDOUT:   %Optional.HasValue.b01: %Optional.HasValue.type.cfd = struct_value () [concrete]
// CHECK:STDOUT:   %Optional.Get.type.a2e: type = fn_type @Optional.Get, @Optional(%OptionalStorage.facet.8cc) [concrete]
// CHECK:STDOUT:   %Optional.Get.180: %Optional.Get.type.a2e = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.facet.5a8: %Destroy.type = facet_value %ValueType, (%custom_witness.563) [concrete]
// CHECK:STDOUT:   %MaybeUnformed.428: type = class_type @MaybeUnformed, @MaybeUnformed(%Destroy.facet.5a8) [concrete]
// CHECK:STDOUT:   %.dd1: type = maybe_unformed_type %ValueType [concrete]
// CHECK:STDOUT:   %struct_type.value.has_value.446: type = struct_type {.value: %MaybeUnformed.428, .has_value: bool} [concrete]
// CHECK:STDOUT:   %Optional.HasValue.specific_fn: <specific function> = specific_function %Optional.HasValue.b01, @Optional.HasValue(%OptionalStorage.facet.8cc) [concrete]
// CHECK:STDOUT:   %Optional.Get.specific_fn: <specific function> = specific_function %Optional.Get.180, @Optional.Get(%OptionalStorage.facet.8cc) [concrete]
// CHECK:STDOUT:   %operator_PlusEqual__carbon_thunk.type: type = fn_type @operator_PlusEqual__carbon_thunk [concrete]
// CHECK:STDOUT:   %operator_PlusEqual__carbon_thunk: %operator_PlusEqual__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.7: type = fn_type @Destroy.Op.loc31_44.6 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.7: %Destroy.Op.type.1d8f74.7 = struct_value () [concrete]
// CHECK:STDOUT:   %Optional.type.facet: %type = facet_value %Optional.69a, () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.81a: type = fn_type @Destroy.WithSelf.SubobjectDestroy, @Destroy.WithSelf(%Optional.type.facet) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.3cf: %Destroy.WithSelf.SubobjectDestroy.type.81a = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.744: type = fn_type @Destroy.WithSelf.SelfDestruct, @Destroy.WithSelf(%Optional.type.facet) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.a3b: %Destroy.WithSelf.SelfDestruct.type.744 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.713: <witness> = custom_witness (%Destroy.Op.1a2547.7, %Destroy.WithSelf.SubobjectDestroy.3cf, %Destroy.WithSelf.SelfDestruct.a3b), @Destroy [concrete]
// CHECK:STDOUT:   %Destroy.facet.a9c: %Destroy.type = facet_value %Optional.69a, (%custom_witness.713) [concrete]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.54a: <witness> = lookup_impl_witness %impl.elem1.5ca, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.3b9: type = fn_type @Destroy.WithSelf.SelfDestruct, @Destroy.WithSelf(%impl.elem1.5ca) [symbolic]
// CHECK:STDOUT:   %.c06: type = fn_type_with_self_type %Destroy.WithSelf.SelfDestruct.type.3b9, %impl.elem1.5ca [symbolic]
// CHECK:STDOUT:   %impl.elem2.ea4: %.c06 = impl_witness_access %Destroy.lookup_impl_witness.54a, element2 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.6af: <specific function> = specific_impl_function %impl.elem2.ea4, @Destroy.WithSelf.SelfDestruct(%impl.elem1.5ca) [symbolic]
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
// CHECK:STDOUT:   %operator_PlusEqual__carbon_thunk.decl: %operator_PlusEqual__carbon_thunk.type = fn_decl @operator_PlusEqual__carbon_thunk [concrete = constants.%operator_PlusEqual__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %MutableRange.decl: type = class_decl @MutableRange [concrete = constants.%MutableRange] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @TestIterate(%R.loc27_17.2: %Iterate_where.type.a2e) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Iterate.lookup_impl_witness: <witness> = lookup_impl_witness %R.loc27_17.1, @Iterate [symbolic = %Iterate.lookup_impl_witness (constants.%Iterate.lookup_impl_witness.bab)]
// CHECK:STDOUT:   %Iterate.facet.loc31_44.5: %Iterate.type = facet_value %R.as_type.loc27_86.1, (%Iterate.lookup_impl_witness) [symbolic = %Iterate.facet.loc31_44.5 (constants.%Iterate.facet.323)]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.loc31_44.5) [symbolic = %Iterate.WithSelf.NewCursor.type (constants.%Iterate.WithSelf.NewCursor.type.c53)]
// CHECK:STDOUT:   %.loc31_44.26: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type, %Iterate.facet.loc31_44.5 [symbolic = %.loc31_44.26 (constants.%.28f)]
// CHECK:STDOUT:   %impl.elem2.loc31_44.3: @TestIterate.%.loc31_44.26 (%.28f) = impl_witness_access %Iterate.lookup_impl_witness, element2 [symbolic = %impl.elem2.loc31_44.3 (constants.%impl.elem2.ffb)]
// CHECK:STDOUT:   %specific_impl_fn.loc31_44.4: <specific function> = specific_impl_function %impl.elem2.loc31_44.3, @Iterate.WithSelf.NewCursor(%Iterate.facet.loc31_44.5) [symbolic = %specific_impl_fn.loc31_44.4 (constants.%specific_impl_fn.79e)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %ptr: type = ptr_type %as_type [symbolic = %ptr (constants.%ptr.dcb)]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.loc31_44.5) [symbolic = %Iterate.WithSelf.Next.type (constants.%Iterate.WithSelf.Next.type.094)]
// CHECK:STDOUT:   %.loc31_44.27: type = fn_type_with_self_type %Iterate.WithSelf.Next.type, %Iterate.facet.loc31_44.5 [symbolic = %.loc31_44.27 (constants.%.4f3)]
// CHECK:STDOUT:   %impl.elem3.loc31_44.2: @TestIterate.%.loc31_44.27 (%.4f3) = impl_witness_access %Iterate.lookup_impl_witness, element3 [symbolic = %impl.elem3.loc31_44.2 (constants.%impl.elem3.a3f)]
// CHECK:STDOUT:   %specific_impl_fn.loc31_44.5: <specific function> = specific_impl_function %impl.elem3.loc31_44.2, @Iterate.WithSelf.Next(%Iterate.facet.loc31_44.5) [symbolic = %specific_impl_fn.loc31_44.5 (constants.%specific_impl_fn.6d9)]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type: type = fn_type @Destroy.WithSelf.SelfDestruct, @Destroy.WithSelf(%impl.elem1) [symbolic = %Destroy.WithSelf.SelfDestruct.type (constants.%Destroy.WithSelf.SelfDestruct.type.3b9)]
// CHECK:STDOUT:   %.loc31_44.28: type = fn_type_with_self_type %Destroy.WithSelf.SelfDestruct.type, %impl.elem1 [symbolic = %.loc31_44.28 (constants.%.c06)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %impl.elem1, @Destroy [symbolic = %Destroy.lookup_impl_witness (constants.%Destroy.lookup_impl_witness.54a)]
// CHECK:STDOUT:   %impl.elem2.loc31_44.4: @TestIterate.%.loc31_44.28 (%.c06) = impl_witness_access %Destroy.lookup_impl_witness, element2 [symbolic = %impl.elem2.loc31_44.4 (constants.%impl.elem2.ea4)]
// CHECK:STDOUT:   %specific_impl_fn.loc31_44.6: <specific function> = specific_impl_function %impl.elem2.loc31_44.4, @Destroy.WithSelf.SelfDestruct(%impl.elem1) [symbolic = %specific_impl_fn.loc31_44.6 (constants.%specific_impl_fn.6af)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%r.param: @TestIterate.%R.as_type.loc27_86.1 (%R.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %i.patt: %pattern_type.d2d = value_binding_pattern i [concrete = constants.%i.patt]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %r.ref: @TestIterate.%R.as_type.loc27_86.1 (%R.as_type) = name_ref r, %r
// CHECK:STDOUT:     %impl.elem2.loc31_44.1: @TestIterate.%.loc31_44.26 (%.28f) = impl_witness_access constants.%Iterate.lookup_impl_witness.bab, element2 [symbolic = %impl.elem2.loc31_44.3 (constants.%impl.elem2.ffb)]
// CHECK:STDOUT:     %bound_method.loc31_44.1: <bound method> = bound_method %r.ref, %impl.elem2.loc31_44.1
// CHECK:STDOUT:     %Iterate.facet.loc31_44.1: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.bab) [symbolic = %Iterate.facet.loc31_44.5 (constants.%Iterate.facet.323)]
// CHECK:STDOUT:     %.loc31_44.1: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc31_44.1 [symbolic = %Iterate.facet.loc31_44.5 (constants.%Iterate.facet.323)]
// CHECK:STDOUT:     %Iterate.facet.loc31_44.2: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.bab) [symbolic = %Iterate.facet.loc31_44.5 (constants.%Iterate.facet.323)]
// CHECK:STDOUT:     %.loc31_44.2: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc31_44.2 [symbolic = %Iterate.facet.loc31_44.5 (constants.%Iterate.facet.323)]
// CHECK:STDOUT:     %specific_impl_fn.loc31_44.1: <specific function> = specific_impl_function %impl.elem2.loc31_44.1, @Iterate.WithSelf.NewCursor(constants.%Iterate.facet.323) [symbolic = %specific_impl_fn.loc31_44.4 (constants.%specific_impl_fn.79e)]
// CHECK:STDOUT:     %bound_method.loc31_44.2: <bound method> = bound_method %r.ref, %specific_impl_fn.loc31_44.1
// CHECK:STDOUT:     %var: ref @TestIterate.%as_type (%as_type.621) = var_storage invalid
// CHECK:STDOUT:     %Iterate.WithSelf.NewCursor.call: init @TestIterate.%as_type (%as_type.621) to %var = call %bound_method.loc31_44.2(%r.ref)
// CHECK:STDOUT:     assign %var, %Iterate.WithSelf.NewCursor.call
// CHECK:STDOUT:     br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.next:
// CHECK:STDOUT:     %addr.loc31: @TestIterate.%ptr (%ptr.dcb) = addr_of %var
// CHECK:STDOUT:     %impl.elem3.loc31_44.1: @TestIterate.%.loc31_44.27 (%.4f3) = impl_witness_access constants.%Iterate.lookup_impl_witness.bab, element3 [symbolic = %impl.elem3.loc31_44.2 (constants.%impl.elem3.a3f)]
// CHECK:STDOUT:     %bound_method.loc31_44.3: <bound method> = bound_method %r.ref, %impl.elem3.loc31_44.1
// CHECK:STDOUT:     %Iterate.facet.loc31_44.3: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.bab) [symbolic = %Iterate.facet.loc31_44.5 (constants.%Iterate.facet.323)]
// CHECK:STDOUT:     %.loc31_44.3: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc31_44.3 [symbolic = %Iterate.facet.loc31_44.5 (constants.%Iterate.facet.323)]
// CHECK:STDOUT:     %Iterate.facet.loc31_44.4: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.bab) [symbolic = %Iterate.facet.loc31_44.5 (constants.%Iterate.facet.323)]
// CHECK:STDOUT:     %.loc31_44.4: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc31_44.4 [symbolic = %Iterate.facet.loc31_44.5 (constants.%Iterate.facet.323)]
// CHECK:STDOUT:     %.loc31_44.5: %Destroy.type = converted constants.%as_type.621, constants.%impl.elem1.5ca [symbolic = %impl.elem1 (constants.%impl.elem1.5ca)]
// CHECK:STDOUT:     %.loc31_44.6: %Destroy.type = converted constants.%as_type.621, constants.%impl.elem1.5ca [symbolic = %impl.elem1 (constants.%impl.elem1.5ca)]
// CHECK:STDOUT:     %specific_impl_fn.loc31_44.2: <specific function> = specific_impl_function %impl.elem3.loc31_44.1, @Iterate.WithSelf.Next(constants.%Iterate.facet.323) [symbolic = %specific_impl_fn.loc31_44.5 (constants.%specific_impl_fn.6d9)]
// CHECK:STDOUT:     %bound_method.loc31_44.4: <bound method> = bound_method %r.ref, %specific_impl_fn.loc31_44.2
// CHECK:STDOUT:     %.loc31_44.7: ref %Optional.69a = temporary_storage
// CHECK:STDOUT:     %Iterate.WithSelf.Next.call: init %Optional.69a to %.loc31_44.7 = call %bound_method.loc31_44.4(%r.ref, %addr.loc31)
// CHECK:STDOUT:     %.loc31_44.8: ref %Optional.69a = temporary %.loc31_44.7, %Iterate.WithSelf.Next.call
// CHECK:STDOUT:     %.loc31_44.9: %Optional.HasValue.type.cfd = specific_constant imports.%Core.import_ref.bb6, @Optional(constants.%OptionalStorage.facet.8cc) [concrete = constants.%Optional.HasValue.b01]
// CHECK:STDOUT:     %HasValue.ref: %Optional.HasValue.type.cfd = name_ref HasValue, %.loc31_44.9 [concrete = constants.%Optional.HasValue.b01]
// CHECK:STDOUT:     %Optional.HasValue.bound: <bound method> = bound_method %.loc31_44.8, %HasValue.ref
// CHECK:STDOUT:     %Optional.HasValue.specific_fn: <specific function> = specific_function %HasValue.ref, @Optional.HasValue(constants.%OptionalStorage.facet.8cc) [concrete = constants.%Optional.HasValue.specific_fn]
// CHECK:STDOUT:     %bound_method.loc31_44.5: <bound method> = bound_method %.loc31_44.8, %Optional.HasValue.specific_fn
// CHECK:STDOUT:     %.loc31_44.10: %Optional.69a = acquire_value %.loc31_44.8
// CHECK:STDOUT:     %Optional.HasValue.call: init bool = call %bound_method.loc31_44.5(%.loc31_44.10)
// CHECK:STDOUT:     %.loc31_44.11: bool = value_of_initializer %Optional.HasValue.call
// CHECK:STDOUT:     %.loc31_44.12: bool = converted %Optional.HasValue.call, %.loc31_44.11
// CHECK:STDOUT:     if %.loc31_44.12 br !for.body else br !for.done
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.body:
// CHECK:STDOUT:     %.loc31_44.13: %Optional.Get.type.a2e = specific_constant imports.%Core.import_ref.dab, @Optional(constants.%OptionalStorage.facet.8cc) [concrete = constants.%Optional.Get.180]
// CHECK:STDOUT:     %Get.ref: %Optional.Get.type.a2e = name_ref Get, %.loc31_44.13 [concrete = constants.%Optional.Get.180]
// CHECK:STDOUT:     %Optional.Get.bound: <bound method> = bound_method %.loc31_44.8, %Get.ref
// CHECK:STDOUT:     %Optional.Get.specific_fn: <specific function> = specific_function %Get.ref, @Optional.Get(constants.%OptionalStorage.facet.8cc) [concrete = constants.%Optional.Get.specific_fn]
// CHECK:STDOUT:     %bound_method.loc31_44.6: <bound method> = bound_method %.loc31_44.8, %Optional.Get.specific_fn
// CHECK:STDOUT:     %.loc31_44.14: ref %ValueType = temporary_storage
// CHECK:STDOUT:     %.loc31_44.15: %Optional.69a = acquire_value %.loc31_44.8
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
// CHECK:STDOUT:     %Destroy.WithSelf.SelfDestruct.bound.loc31_44.1: <bound method> = bound_method %.loc31_44.16, constants.%Destroy.WithSelf.SelfDestruct.000
// CHECK:STDOUT:     %Destroy.facet.loc31_44.1: %Destroy.type = facet_value constants.%ValueType, (constants.%custom_witness.563) [concrete = constants.%Destroy.facet.5a8]
// CHECK:STDOUT:     %.loc31_44.18: %Destroy.type = converted constants.%ValueType, %Destroy.facet.loc31_44.1 [concrete = constants.%Destroy.facet.5a8]
// CHECK:STDOUT:     %Destroy.facet.loc31_44.2: %Destroy.type = facet_value constants.%ValueType, (constants.%custom_witness.563) [concrete = constants.%Destroy.facet.5a8]
// CHECK:STDOUT:     %.loc31_44.19: %Destroy.type = converted constants.%ValueType, %Destroy.facet.loc31_44.2 [concrete = constants.%Destroy.facet.5a8]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %bound_method.loc31_44.7: <bound method> = bound_method %.loc31_44.16, %Destroy.WithSelf.SelfDestruct.specific_fn.1
// CHECK:STDOUT:     %Destroy.WithSelf.SelfDestruct.call.loc31_44.1: init %empty_tuple.type = call %bound_method.loc31_44.7(%.loc31_44.16)
// CHECK:STDOUT:     %Destroy.WithSelf.SelfDestruct.bound.loc31_44.2: <bound method> = bound_method %.loc31_44.8, constants.%Destroy.WithSelf.SelfDestruct.a3b
// CHECK:STDOUT:     %Destroy.facet.loc31_44.3: %Destroy.type = facet_value constants.%Optional.69a, (constants.%custom_witness.713) [concrete = constants.%Destroy.facet.a9c]
// CHECK:STDOUT:     %.loc31_44.20: %Destroy.type = converted constants.%Optional.69a, %Destroy.facet.loc31_44.3 [concrete = constants.%Destroy.facet.a9c]
// CHECK:STDOUT:     %Destroy.facet.loc31_44.4: %Destroy.type = facet_value constants.%Optional.69a, (constants.%custom_witness.713) [concrete = constants.%Destroy.facet.a9c]
// CHECK:STDOUT:     %.loc31_44.21: %Destroy.type = converted constants.%Optional.69a, %Destroy.facet.loc31_44.4 [concrete = constants.%Destroy.facet.a9c]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %bound_method.loc31_44.8: <bound method> = bound_method %.loc31_44.8, %Destroy.WithSelf.SelfDestruct.specific_fn.2
// CHECK:STDOUT:     %Destroy.WithSelf.SelfDestruct.call.loc31_44.2: init %empty_tuple.type = call %bound_method.loc31_44.8(%.loc31_44.8)
// CHECK:STDOUT:     br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.done:
// CHECK:STDOUT:     %Destroy.WithSelf.SelfDestruct.bound.loc31_44.3: <bound method> = bound_method %.loc31_44.8, constants.%Destroy.WithSelf.SelfDestruct.a3b
// CHECK:STDOUT:     %Destroy.facet.loc31_44.5: %Destroy.type = facet_value constants.%Optional.69a, (constants.%custom_witness.713) [concrete = constants.%Destroy.facet.a9c]
// CHECK:STDOUT:     %.loc31_44.22: %Destroy.type = converted constants.%Optional.69a, %Destroy.facet.loc31_44.5 [concrete = constants.%Destroy.facet.a9c]
// CHECK:STDOUT:     %Destroy.facet.loc31_44.6: %Destroy.type = facet_value constants.%Optional.69a, (constants.%custom_witness.713) [concrete = constants.%Destroy.facet.a9c]
// CHECK:STDOUT:     %.loc31_44.23: %Destroy.type = converted constants.%Optional.69a, %Destroy.facet.loc31_44.6 [concrete = constants.%Destroy.facet.a9c]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %bound_method.loc31_44.9: <bound method> = bound_method %.loc31_44.8, %Destroy.WithSelf.SelfDestruct.specific_fn.3
// CHECK:STDOUT:     %Destroy.WithSelf.SelfDestruct.call.loc31_44.3: init %empty_tuple.type = call %bound_method.loc31_44.9(%.loc31_44.8)
// CHECK:STDOUT:     %impl.elem2.loc31_44.2: @TestIterate.%.loc31_44.28 (%.c06) = impl_witness_access constants.%Destroy.lookup_impl_witness.54a, element2 [symbolic = %impl.elem2.loc31_44.4 (constants.%impl.elem2.ea4)]
// CHECK:STDOUT:     %bound_method.loc31_44.10: <bound method> = bound_method %var, %impl.elem2.loc31_44.2
// CHECK:STDOUT:     %.loc31_44.24: %Destroy.type = converted constants.%as_type.621, constants.%impl.elem1.5ca [symbolic = %impl.elem1 (constants.%impl.elem1.5ca)]
// CHECK:STDOUT:     %.loc31_44.25: %Destroy.type = converted constants.%as_type.621, constants.%impl.elem1.5ca [symbolic = %impl.elem1 (constants.%impl.elem1.5ca)]
// CHECK:STDOUT:     %specific_impl_fn.loc31_44.3: <specific function> = specific_impl_function %impl.elem2.loc31_44.2, @Destroy.WithSelf.SelfDestruct(constants.%impl.elem1.5ca) [symbolic = %specific_impl_fn.loc31_44.6 (constants.%specific_impl_fn.6af)]
// CHECK:STDOUT:     %bound_method.loc31_44.11: <bound method> = bound_method %var, %specific_impl_fn.loc31_44.3
// CHECK:STDOUT:     %Destroy.WithSelf.SelfDestruct.call.loc31_44.4: init %empty_tuple.type = call %bound_method.loc31_44.11(%var)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc31_44.1(%self.param: ref %.dd1) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc31_44.2(%self.param: ref %MaybeUnformed.428) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc31_44.3(%self.param: ref bool) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc31_44.4(%self.param: ref %struct_type.value.has_value.446) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc31_44.5(%self.param: ref %DefaultOptionalStorage.7fe) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc31_44.6(%self.param: ref %Optional.69a) {
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
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.33d
// CHECK:STDOUT:   %r.param_patt.loc27_84.2 => constants.%r.param_patt
// CHECK:STDOUT:   %r.patt.loc27_84.2 => constants.%r.patt
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_adl_overloads.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
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
// CHECK:STDOUT:   %ValueType.type.facet: %type = facet_value %ValueType, () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.b61: type = fn_type @Destroy.WithSelf.SubobjectDestroy, @Destroy.WithSelf(%ValueType.type.facet) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.b46: %Destroy.WithSelf.SubobjectDestroy.type.b61 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.ffd: type = fn_type @Destroy.WithSelf.SelfDestruct, @Destroy.WithSelf(%ValueType.type.facet) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.2f2: %Destroy.WithSelf.SelfDestruct.type.ffd = struct_value () [concrete]
// CHECK:STDOUT:   %ValueType.Op.type.9ce105.1: type = fn_type @ValueType.Op.1 [concrete]
// CHECK:STDOUT:   %ValueType.Op.038ea9.1: %ValueType.Op.type.9ce105.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.0aa: <witness> = custom_witness (%ValueType.Op.038ea9.1, %Destroy.WithSelf.SubobjectDestroy.b46, %Destroy.WithSelf.SelfDestruct.2f2), @Destroy [concrete]
// CHECK:STDOUT:   %const.924: type = const_type %ValueType [concrete]
// CHECK:STDOUT:   %ptr.403: type = ptr_type %const.924 [concrete]
// CHECK:STDOUT:   %ptr.6ed: type = ptr_type %ValueType [concrete]
// CHECK:STDOUT:   %ValueType.Op.type.9ce105.2: type = fn_type @ValueType.Op.2 [concrete]
// CHECK:STDOUT:   %ValueType.Op.038ea9.2: %ValueType.Op.type.9ce105.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.1c5: <witness> = custom_witness (%ValueType.Op.038ea9.2), @Copy [concrete]
// CHECK:STDOUT:   %facet_value.fb7: %facet_type.f48 = facet_value %ValueType, (%custom_witness.0aa, %custom_witness.1c5) [concrete]
// CHECK:STDOUT:   %Iterate.lookup_impl_witness.1d8: <witness> = lookup_impl_witness %.Self.500, @Iterate [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.294: %facet_type.f48 = impl_witness_access %Iterate.lookup_impl_witness.1d8, element0 [symbolic_self]
// CHECK:STDOUT:   %Iterate_where.type.eec: type = facet_type <@Iterate where %impl.elem0.294 = %facet_value.fb7> [concrete]
// CHECK:STDOUT:   %pattern_type.200: type = pattern_type %Iterate_where.type.eec [concrete]
// CHECK:STDOUT:   %R.patt: %pattern_type.200 = symbolic_binding_pattern R, 0 [symbolic]
// CHECK:STDOUT:   %R: %Iterate_where.type.eec = symbolic_binding R, 0 [symbolic]
// CHECK:STDOUT:   %R.as_type: type = facet_access_type %R [symbolic]
// CHECK:STDOUT:   %pattern_type.dac: type = pattern_type %R.as_type [symbolic]
// CHECK:STDOUT:   %r.param_patt.769: %pattern_type.dac = value_param_pattern [symbolic]
// CHECK:STDOUT:   %r.patt.6b2: %pattern_type.dac = wrapper_binding_pattern r, %r.param_patt.769 [symbolic]
// CHECK:STDOUT:   %i.patt: %pattern_type.e09 = value_binding_pattern i [concrete]
// CHECK:STDOUT:   %Iterate.lookup_impl_witness.aa1: <witness> = lookup_impl_witness %R, @Iterate [symbolic]
// CHECK:STDOUT:   %Iterate.facet.357: %Iterate.type = facet_value %R.as_type, (%Iterate.lookup_impl_witness.aa1) [symbolic]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.9d7: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.357) [symbolic]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.971: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.357) [symbolic]
// CHECK:STDOUT:   %.48c: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.9d7, %Iterate.facet.357 [symbolic]
// CHECK:STDOUT:   %impl.elem2.da4: %.48c = impl_witness_access %Iterate.lookup_impl_witness.aa1, element2 [symbolic]
// CHECK:STDOUT:   %impl.elem1.a2e: %Destroy.type = impl_witness_access %Iterate.lookup_impl_witness.aa1, element1 [symbolic]
// CHECK:STDOUT:   %as_type.de5: type = facet_access_type %impl.elem1.a2e [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.b64: <specific function> = specific_impl_function %impl.elem2.da4, @Iterate.WithSelf.NewCursor(%Iterate.facet.357) [symbolic]
// CHECK:STDOUT:   %ptr.c62: type = ptr_type %as_type.de5 [symbolic]
// CHECK:STDOUT:   %.539: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.971, %Iterate.facet.357 [symbolic]
// CHECK:STDOUT:   %impl.elem3.b6d: %.539 = impl_witness_access %Iterate.lookup_impl_witness.aa1, element3 [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.1da: type = class_type @DefaultOptionalStorage, @DefaultOptionalStorage(%facet_value.fb7) [concrete]
// CHECK:STDOUT:   %OptionalStorage.impl_witness.a0c: <witness> = impl_witness imports.%OptionalStorage.impl_witness_table.460, @T.as_type.as.OptionalStorage.impl(%facet_value.fb7) [concrete]
// CHECK:STDOUT:   %OptionalStorage.facet.193: %OptionalStorage.type = facet_value %ValueType, (%OptionalStorage.impl_witness.a0c) [concrete]
// CHECK:STDOUT:   %Optional.164: type = class_type @Optional, @Optional(%OptionalStorage.facet.193) [concrete]
// CHECK:STDOUT:   %specific_impl_fn.264: <specific function> = specific_impl_function %impl.elem3.b6d, @Iterate.WithSelf.Next(%Iterate.facet.357) [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.type.c93: type = fn_type @Optional.HasValue, @Optional(%OptionalStorage.facet.193) [concrete]
// CHECK:STDOUT:   %Optional.HasValue.32a: %Optional.HasValue.type.c93 = struct_value () [concrete]
// CHECK:STDOUT:   %Optional.Get.type.7cd: type = fn_type @Optional.Get, @Optional(%OptionalStorage.facet.193) [concrete]
// CHECK:STDOUT:   %Optional.Get.4de: %Optional.Get.type.7cd = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.facet.0bd: %Destroy.type = facet_value %ValueType, (%custom_witness.0aa) [concrete]
// CHECK:STDOUT:   %MaybeUnformed.b96: type = class_type @MaybeUnformed, @MaybeUnformed(%Destroy.facet.0bd) [concrete]
// CHECK:STDOUT:   %.8f3: type = maybe_unformed_type %ValueType [concrete]
// CHECK:STDOUT:   %struct_type.value.has_value.9db: type = struct_type {.value: %MaybeUnformed.b96, .has_value: bool} [concrete]
// CHECK:STDOUT:   %Optional.HasValue.specific_fn.faa: <specific function> = specific_function %Optional.HasValue.32a, @Optional.HasValue(%OptionalStorage.facet.193) [concrete]
// CHECK:STDOUT:   %Optional.Get.specific_fn.7ee: <specific function> = specific_function %Optional.Get.4de, @Optional.Get(%OptionalStorage.facet.193) [concrete]
// CHECK:STDOUT:   %operator_PlusEqual__carbon_thunk.type: type = fn_type @operator_PlusEqual__carbon_thunk [concrete]
// CHECK:STDOUT:   %operator_PlusEqual__carbon_thunk: %operator_PlusEqual__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.7: type = fn_type @Destroy.Op.loc38_41.6 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.7: %Destroy.Op.type.1d8f74.7 = struct_value () [concrete]
// CHECK:STDOUT:   %Optional.type.facet: %type = facet_value %Optional.164, () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.860: type = fn_type @Destroy.WithSelf.SubobjectDestroy, @Destroy.WithSelf(%Optional.type.facet) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.da6: %Destroy.WithSelf.SubobjectDestroy.type.860 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.5fa: type = fn_type @Destroy.WithSelf.SelfDestruct, @Destroy.WithSelf(%Optional.type.facet) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.3ed: %Destroy.WithSelf.SelfDestruct.type.5fa = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.eb5: <witness> = custom_witness (%Destroy.Op.1a2547.7, %Destroy.WithSelf.SubobjectDestroy.da6, %Destroy.WithSelf.SelfDestruct.3ed), @Destroy [concrete]
// CHECK:STDOUT:   %Destroy.facet.8c1: %Destroy.type = facet_value %Optional.164, (%custom_witness.eb5) [concrete]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.b9f: <witness> = lookup_impl_witness %impl.elem1.a2e, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.2e8: type = fn_type @Destroy.WithSelf.SelfDestruct, @Destroy.WithSelf(%impl.elem1.a2e) [symbolic]
// CHECK:STDOUT:   %.aad: type = fn_type_with_self_type %Destroy.WithSelf.SelfDestruct.type.2e8, %impl.elem1.a2e [symbolic]
// CHECK:STDOUT:   %impl.elem2.ec1: %.aad = impl_witness_access %Destroy.lookup_impl_witness.b9f, element2 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.f62: <specific function> = specific_impl_function %impl.elem2.ec1, @Destroy.WithSelf.SelfDestruct(%impl.elem1.a2e) [symbolic]
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
// CHECK:STDOUT:   %Iterator.type.facet: %type = facet_value %Iterator, () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.072: type = fn_type @Destroy.WithSelf.SubobjectDestroy, @Destroy.WithSelf(%Iterator.type.facet) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.849: %Destroy.WithSelf.SubobjectDestroy.type.072 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.0b9: type = fn_type @Destroy.WithSelf.SelfDestruct, @Destroy.WithSelf(%Iterator.type.facet) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.acf: %Destroy.WithSelf.SelfDestruct.type.0b9 = struct_value () [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.c95ff6.2: type = fn_type @Iterator.Op.2 [concrete]
// CHECK:STDOUT:   %Iterator.Op.c90ac2.2: %Iterator.Op.type.c95ff6.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.955: <witness> = custom_witness (%Iterator.Op.c90ac2.2, %Destroy.WithSelf.SubobjectDestroy.849, %Destroy.WithSelf.SelfDestruct.acf), @Destroy [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.c95ff6.3: type = fn_type @Iterator.Op.3 [concrete]
// CHECK:STDOUT:   %Iterator.Op.c90ac2.3: %Iterator.Op.type.c95ff6.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.68a: <witness> = custom_witness (%Iterator.Op.c90ac2.3), @Inc [concrete]
// CHECK:STDOUT:   %Equal.type: type = fn_type @Equal [concrete]
// CHECK:STDOUT:   %Equal: %Equal.type = struct_value () [concrete]
// CHECK:STDOUT:   %NotEqual.type: type = fn_type @NotEqual [concrete]
// CHECK:STDOUT:   %NotEqual: %NotEqual.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.2ef: <witness> = custom_witness (%Equal, %NotEqual), @EqWith, @EqWith(%Iterator) [concrete]
// CHECK:STDOUT:   %facet_value.b69: %CppRange_where.type.d682d6.1 = facet_value %MutableAndConstRange, (%custom_witness.0aa, %custom_witness.1c5, %custom_witness.660, %custom_witness.955, %custom_witness.d56, %custom_witness.68a, %custom_witness.2ef) [concrete]
// CHECK:STDOUT:   %tuple.type.de6: type = tuple_type (%Iterator, %Iterator) [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.8: type = fn_type @Destroy.Op.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.8: %Destroy.Op.type.1d8f74.8 = struct_value () [concrete]
// CHECK:STDOUT:   %tuple.type.facet: %type = facet_value %tuple.type.de6, () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.cd1: type = fn_type @Destroy.WithSelf.SubobjectDestroy, @Destroy.WithSelf(%tuple.type.facet) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.76a: %Destroy.WithSelf.SubobjectDestroy.type.cd1 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fd0: type = fn_type @Destroy.WithSelf.SelfDestruct, @Destroy.WithSelf(%tuple.type.facet) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.f74: %Destroy.WithSelf.SelfDestruct.type.fd0 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.a91: <witness> = custom_witness (%Destroy.Op.1a2547.8, %Destroy.WithSelf.SubobjectDestroy.76a, %Destroy.WithSelf.SelfDestruct.f74), @Destroy [concrete]
// CHECK:STDOUT:   %Destroy.facet.992: %Destroy.type = facet_value %tuple.type.de6, (%custom_witness.a91) [concrete]
// CHECK:STDOUT:   %Iterate.impl_witness.afa: <witness> = impl_witness imports.%Iterate.impl_witness_table.121, @T.as_type.as.Iterate.impl(%facet_value.b69) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.type.183: type = fn_type @T.as_type.as.Iterate.impl.NewCursor, @T.as_type.as.Iterate.impl(%facet_value.b69) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.415: %T.as_type.as.Iterate.impl.NewCursor.type.183 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.type.52d: type = fn_type @T.as_type.as.Iterate.impl.Next, @T.as_type.as.Iterate.impl(%facet_value.b69) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.872: %T.as_type.as.Iterate.impl.Next.type.52d = struct_value () [concrete]
// CHECK:STDOUT:   %Iterate.facet.3f8: %Iterate.type = facet_value %MutableAndConstRange, (%Iterate.impl_witness.afa) [concrete]
// CHECK:STDOUT:   %facet_value.f2e: %Iterate_where.type.eec = facet_value %MutableAndConstRange, (%Iterate.impl_witness.afa) [concrete]
// CHECK:STDOUT:   %r.param_patt.3a6: %pattern_type.fdb = value_param_pattern [concrete]
// CHECK:STDOUT:   %r.patt.403: %pattern_type.fdb = wrapper_binding_pattern r, %r.param_patt.3a6 [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.735: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.3f8) [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.9c9: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.3f8) [concrete]
// CHECK:STDOUT:   %.bd7: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.735, %Iterate.facet.3f8 [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.specific_fn: <specific function> = specific_function %T.as_type.as.Iterate.impl.NewCursor.415, @T.as_type.as.Iterate.impl.NewCursor(%facet_value.b69) [concrete]
// CHECK:STDOUT:   %ptr.b95: type = ptr_type %tuple.type.de6 [concrete]
// CHECK:STDOUT:   %.2c8: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.9c9, %Iterate.facet.3f8 [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.specific_fn: <specific function> = specific_function %T.as_type.as.Iterate.impl.Next.872, @T.as_type.as.Iterate.impl.Next(%facet_value.b69) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.42d: type = fn_type @Destroy.WithSelf.SelfDestruct, @Destroy.WithSelf(%Destroy.facet.992) [concrete]
// CHECK:STDOUT:   %.5c5: type = fn_type_with_self_type %Destroy.WithSelf.SelfDestruct.type.42d, %Destroy.facet.992 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.specific_fn.13b: <specific function> = specific_function %Destroy.WithSelf.SelfDestruct.f74, @Destroy.WithSelf.SelfDestruct(%tuple.type.facet) [concrete]
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
// CHECK:STDOUT:   %operator_PlusEqual__carbon_thunk.decl: %operator_PlusEqual__carbon_thunk.type = fn_decl @operator_PlusEqual__carbon_thunk [concrete = constants.%operator_PlusEqual__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %MutableAndConstRange.decl: type = class_decl @MutableAndConstRange [concrete = constants.%MutableAndConstRange] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @TestIterate(%R.loc34_17.2: %Iterate_where.type.eec) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Iterate.lookup_impl_witness: <witness> = lookup_impl_witness %R.loc34_17.1, @Iterate [symbolic = %Iterate.lookup_impl_witness (constants.%Iterate.lookup_impl_witness.aa1)]
// CHECK:STDOUT:   %Iterate.facet.loc38_41.5: %Iterate.type = facet_value %R.as_type.loc34_83.1, (%Iterate.lookup_impl_witness) [symbolic = %Iterate.facet.loc38_41.5 (constants.%Iterate.facet.357)]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.loc38_41.5) [symbolic = %Iterate.WithSelf.NewCursor.type (constants.%Iterate.WithSelf.NewCursor.type.9d7)]
// CHECK:STDOUT:   %.loc38_41.26: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type, %Iterate.facet.loc38_41.5 [symbolic = %.loc38_41.26 (constants.%.48c)]
// CHECK:STDOUT:   %impl.elem2.loc38_41.3: @TestIterate.%.loc38_41.26 (%.48c) = impl_witness_access %Iterate.lookup_impl_witness, element2 [symbolic = %impl.elem2.loc38_41.3 (constants.%impl.elem2.da4)]
// CHECK:STDOUT:   %specific_impl_fn.loc38_41.4: <specific function> = specific_impl_function %impl.elem2.loc38_41.3, @Iterate.WithSelf.NewCursor(%Iterate.facet.loc38_41.5) [symbolic = %specific_impl_fn.loc38_41.4 (constants.%specific_impl_fn.b64)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %ptr: type = ptr_type %as_type [symbolic = %ptr (constants.%ptr.c62)]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.loc38_41.5) [symbolic = %Iterate.WithSelf.Next.type (constants.%Iterate.WithSelf.Next.type.971)]
// CHECK:STDOUT:   %.loc38_41.27: type = fn_type_with_self_type %Iterate.WithSelf.Next.type, %Iterate.facet.loc38_41.5 [symbolic = %.loc38_41.27 (constants.%.539)]
// CHECK:STDOUT:   %impl.elem3.loc38_41.2: @TestIterate.%.loc38_41.27 (%.539) = impl_witness_access %Iterate.lookup_impl_witness, element3 [symbolic = %impl.elem3.loc38_41.2 (constants.%impl.elem3.b6d)]
// CHECK:STDOUT:   %specific_impl_fn.loc38_41.5: <specific function> = specific_impl_function %impl.elem3.loc38_41.2, @Iterate.WithSelf.Next(%Iterate.facet.loc38_41.5) [symbolic = %specific_impl_fn.loc38_41.5 (constants.%specific_impl_fn.264)]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type: type = fn_type @Destroy.WithSelf.SelfDestruct, @Destroy.WithSelf(%impl.elem1) [symbolic = %Destroy.WithSelf.SelfDestruct.type (constants.%Destroy.WithSelf.SelfDestruct.type.2e8)]
// CHECK:STDOUT:   %.loc38_41.28: type = fn_type_with_self_type %Destroy.WithSelf.SelfDestruct.type, %impl.elem1 [symbolic = %.loc38_41.28 (constants.%.aad)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %impl.elem1, @Destroy [symbolic = %Destroy.lookup_impl_witness (constants.%Destroy.lookup_impl_witness.b9f)]
// CHECK:STDOUT:   %impl.elem2.loc38_41.4: @TestIterate.%.loc38_41.28 (%.aad) = impl_witness_access %Destroy.lookup_impl_witness, element2 [symbolic = %impl.elem2.loc38_41.4 (constants.%impl.elem2.ec1)]
// CHECK:STDOUT:   %specific_impl_fn.loc38_41.6: <specific function> = specific_impl_function %impl.elem2.loc38_41.4, @Destroy.WithSelf.SelfDestruct(%impl.elem1) [symbolic = %specific_impl_fn.loc38_41.6 (constants.%specific_impl_fn.f62)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%r.param: @TestIterate.%R.as_type.loc34_83.1 (%R.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %i.patt: %pattern_type.e09 = value_binding_pattern i [concrete = constants.%i.patt]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %r.ref: @TestIterate.%R.as_type.loc34_83.1 (%R.as_type) = name_ref r, %r
// CHECK:STDOUT:     %impl.elem2.loc38_41.1: @TestIterate.%.loc38_41.26 (%.48c) = impl_witness_access constants.%Iterate.lookup_impl_witness.aa1, element2 [symbolic = %impl.elem2.loc38_41.3 (constants.%impl.elem2.da4)]
// CHECK:STDOUT:     %bound_method.loc38_41.1: <bound method> = bound_method %r.ref, %impl.elem2.loc38_41.1
// CHECK:STDOUT:     %Iterate.facet.loc38_41.1: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.aa1) [symbolic = %Iterate.facet.loc38_41.5 (constants.%Iterate.facet.357)]
// CHECK:STDOUT:     %.loc38_41.1: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc38_41.1 [symbolic = %Iterate.facet.loc38_41.5 (constants.%Iterate.facet.357)]
// CHECK:STDOUT:     %Iterate.facet.loc38_41.2: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.aa1) [symbolic = %Iterate.facet.loc38_41.5 (constants.%Iterate.facet.357)]
// CHECK:STDOUT:     %.loc38_41.2: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc38_41.2 [symbolic = %Iterate.facet.loc38_41.5 (constants.%Iterate.facet.357)]
// CHECK:STDOUT:     %specific_impl_fn.loc38_41.1: <specific function> = specific_impl_function %impl.elem2.loc38_41.1, @Iterate.WithSelf.NewCursor(constants.%Iterate.facet.357) [symbolic = %specific_impl_fn.loc38_41.4 (constants.%specific_impl_fn.b64)]
// CHECK:STDOUT:     %bound_method.loc38_41.2: <bound method> = bound_method %r.ref, %specific_impl_fn.loc38_41.1
// CHECK:STDOUT:     %var: ref @TestIterate.%as_type (%as_type.de5) = var_storage invalid
// CHECK:STDOUT:     %Iterate.WithSelf.NewCursor.call: init @TestIterate.%as_type (%as_type.de5) to %var = call %bound_method.loc38_41.2(%r.ref)
// CHECK:STDOUT:     assign %var, %Iterate.WithSelf.NewCursor.call
// CHECK:STDOUT:     br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.next:
// CHECK:STDOUT:     %addr.loc38: @TestIterate.%ptr (%ptr.c62) = addr_of %var
// CHECK:STDOUT:     %impl.elem3.loc38_41.1: @TestIterate.%.loc38_41.27 (%.539) = impl_witness_access constants.%Iterate.lookup_impl_witness.aa1, element3 [symbolic = %impl.elem3.loc38_41.2 (constants.%impl.elem3.b6d)]
// CHECK:STDOUT:     %bound_method.loc38_41.3: <bound method> = bound_method %r.ref, %impl.elem3.loc38_41.1
// CHECK:STDOUT:     %Iterate.facet.loc38_41.3: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.aa1) [symbolic = %Iterate.facet.loc38_41.5 (constants.%Iterate.facet.357)]
// CHECK:STDOUT:     %.loc38_41.3: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc38_41.3 [symbolic = %Iterate.facet.loc38_41.5 (constants.%Iterate.facet.357)]
// CHECK:STDOUT:     %Iterate.facet.loc38_41.4: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.aa1) [symbolic = %Iterate.facet.loc38_41.5 (constants.%Iterate.facet.357)]
// CHECK:STDOUT:     %.loc38_41.4: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc38_41.4 [symbolic = %Iterate.facet.loc38_41.5 (constants.%Iterate.facet.357)]
// CHECK:STDOUT:     %.loc38_41.5: %Destroy.type = converted constants.%as_type.de5, constants.%impl.elem1.a2e [symbolic = %impl.elem1 (constants.%impl.elem1.a2e)]
// CHECK:STDOUT:     %.loc38_41.6: %Destroy.type = converted constants.%as_type.de5, constants.%impl.elem1.a2e [symbolic = %impl.elem1 (constants.%impl.elem1.a2e)]
// CHECK:STDOUT:     %specific_impl_fn.loc38_41.2: <specific function> = specific_impl_function %impl.elem3.loc38_41.1, @Iterate.WithSelf.Next(constants.%Iterate.facet.357) [symbolic = %specific_impl_fn.loc38_41.5 (constants.%specific_impl_fn.264)]
// CHECK:STDOUT:     %bound_method.loc38_41.4: <bound method> = bound_method %r.ref, %specific_impl_fn.loc38_41.2
// CHECK:STDOUT:     %.loc38_41.7: ref %Optional.164 = temporary_storage
// CHECK:STDOUT:     %Iterate.WithSelf.Next.call: init %Optional.164 to %.loc38_41.7 = call %bound_method.loc38_41.4(%r.ref, %addr.loc38)
// CHECK:STDOUT:     %.loc38_41.8: ref %Optional.164 = temporary %.loc38_41.7, %Iterate.WithSelf.Next.call
// CHECK:STDOUT:     %.loc38_41.9: %Optional.HasValue.type.c93 = specific_constant imports.%Core.import_ref.bb6, @Optional(constants.%OptionalStorage.facet.193) [concrete = constants.%Optional.HasValue.32a]
// CHECK:STDOUT:     %HasValue.ref: %Optional.HasValue.type.c93 = name_ref HasValue, %.loc38_41.9 [concrete = constants.%Optional.HasValue.32a]
// CHECK:STDOUT:     %Optional.HasValue.bound: <bound method> = bound_method %.loc38_41.8, %HasValue.ref
// CHECK:STDOUT:     %Optional.HasValue.specific_fn: <specific function> = specific_function %HasValue.ref, @Optional.HasValue(constants.%OptionalStorage.facet.193) [concrete = constants.%Optional.HasValue.specific_fn.faa]
// CHECK:STDOUT:     %bound_method.loc38_41.5: <bound method> = bound_method %.loc38_41.8, %Optional.HasValue.specific_fn
// CHECK:STDOUT:     %.loc38_41.10: %Optional.164 = acquire_value %.loc38_41.8
// CHECK:STDOUT:     %Optional.HasValue.call: init bool = call %bound_method.loc38_41.5(%.loc38_41.10)
// CHECK:STDOUT:     %.loc38_41.11: bool = value_of_initializer %Optional.HasValue.call
// CHECK:STDOUT:     %.loc38_41.12: bool = converted %Optional.HasValue.call, %.loc38_41.11
// CHECK:STDOUT:     if %.loc38_41.12 br !for.body else br !for.done
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.body:
// CHECK:STDOUT:     %.loc38_41.13: %Optional.Get.type.7cd = specific_constant imports.%Core.import_ref.dab, @Optional(constants.%OptionalStorage.facet.193) [concrete = constants.%Optional.Get.4de]
// CHECK:STDOUT:     %Get.ref: %Optional.Get.type.7cd = name_ref Get, %.loc38_41.13 [concrete = constants.%Optional.Get.4de]
// CHECK:STDOUT:     %Optional.Get.bound: <bound method> = bound_method %.loc38_41.8, %Get.ref
// CHECK:STDOUT:     %Optional.Get.specific_fn: <specific function> = specific_function %Get.ref, @Optional.Get(constants.%OptionalStorage.facet.193) [concrete = constants.%Optional.Get.specific_fn.7ee]
// CHECK:STDOUT:     %bound_method.loc38_41.6: <bound method> = bound_method %.loc38_41.8, %Optional.Get.specific_fn
// CHECK:STDOUT:     %.loc38_41.14: ref %ValueType = temporary_storage
// CHECK:STDOUT:     %.loc38_41.15: %Optional.164 = acquire_value %.loc38_41.8
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
// CHECK:STDOUT:     %Destroy.WithSelf.SelfDestruct.bound.loc38_41.1: <bound method> = bound_method %.loc38_41.16, constants.%Destroy.WithSelf.SelfDestruct.2f2
// CHECK:STDOUT:     %Destroy.facet.loc38_41.1: %Destroy.type = facet_value constants.%ValueType, (constants.%custom_witness.0aa) [concrete = constants.%Destroy.facet.0bd]
// CHECK:STDOUT:     %.loc38_41.18: %Destroy.type = converted constants.%ValueType, %Destroy.facet.loc38_41.1 [concrete = constants.%Destroy.facet.0bd]
// CHECK:STDOUT:     %Destroy.facet.loc38_41.2: %Destroy.type = facet_value constants.%ValueType, (constants.%custom_witness.0aa) [concrete = constants.%Destroy.facet.0bd]
// CHECK:STDOUT:     %.loc38_41.19: %Destroy.type = converted constants.%ValueType, %Destroy.facet.loc38_41.2 [concrete = constants.%Destroy.facet.0bd]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %bound_method.loc38_41.7: <bound method> = bound_method %.loc38_41.16, %Destroy.WithSelf.SelfDestruct.specific_fn.1
// CHECK:STDOUT:     %Destroy.WithSelf.SelfDestruct.call.loc38_41.1: init %empty_tuple.type = call %bound_method.loc38_41.7(%.loc38_41.16)
// CHECK:STDOUT:     %Destroy.WithSelf.SelfDestruct.bound.loc38_41.2: <bound method> = bound_method %.loc38_41.8, constants.%Destroy.WithSelf.SelfDestruct.3ed
// CHECK:STDOUT:     %Destroy.facet.loc38_41.3: %Destroy.type = facet_value constants.%Optional.164, (constants.%custom_witness.eb5) [concrete = constants.%Destroy.facet.8c1]
// CHECK:STDOUT:     %.loc38_41.20: %Destroy.type = converted constants.%Optional.164, %Destroy.facet.loc38_41.3 [concrete = constants.%Destroy.facet.8c1]
// CHECK:STDOUT:     %Destroy.facet.loc38_41.4: %Destroy.type = facet_value constants.%Optional.164, (constants.%custom_witness.eb5) [concrete = constants.%Destroy.facet.8c1]
// CHECK:STDOUT:     %.loc38_41.21: %Destroy.type = converted constants.%Optional.164, %Destroy.facet.loc38_41.4 [concrete = constants.%Destroy.facet.8c1]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %bound_method.loc38_41.8: <bound method> = bound_method %.loc38_41.8, %Destroy.WithSelf.SelfDestruct.specific_fn.2
// CHECK:STDOUT:     %Destroy.WithSelf.SelfDestruct.call.loc38_41.2: init %empty_tuple.type = call %bound_method.loc38_41.8(%.loc38_41.8)
// CHECK:STDOUT:     br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.done:
// CHECK:STDOUT:     %Destroy.WithSelf.SelfDestruct.bound.loc38_41.3: <bound method> = bound_method %.loc38_41.8, constants.%Destroy.WithSelf.SelfDestruct.3ed
// CHECK:STDOUT:     %Destroy.facet.loc38_41.5: %Destroy.type = facet_value constants.%Optional.164, (constants.%custom_witness.eb5) [concrete = constants.%Destroy.facet.8c1]
// CHECK:STDOUT:     %.loc38_41.22: %Destroy.type = converted constants.%Optional.164, %Destroy.facet.loc38_41.5 [concrete = constants.%Destroy.facet.8c1]
// CHECK:STDOUT:     %Destroy.facet.loc38_41.6: %Destroy.type = facet_value constants.%Optional.164, (constants.%custom_witness.eb5) [concrete = constants.%Destroy.facet.8c1]
// CHECK:STDOUT:     %.loc38_41.23: %Destroy.type = converted constants.%Optional.164, %Destroy.facet.loc38_41.6 [concrete = constants.%Destroy.facet.8c1]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %bound_method.loc38_41.9: <bound method> = bound_method %.loc38_41.8, %Destroy.WithSelf.SelfDestruct.specific_fn.3
// CHECK:STDOUT:     %Destroy.WithSelf.SelfDestruct.call.loc38_41.3: init %empty_tuple.type = call %bound_method.loc38_41.9(%.loc38_41.8)
// CHECK:STDOUT:     %impl.elem2.loc38_41.2: @TestIterate.%.loc38_41.28 (%.aad) = impl_witness_access constants.%Destroy.lookup_impl_witness.b9f, element2 [symbolic = %impl.elem2.loc38_41.4 (constants.%impl.elem2.ec1)]
// CHECK:STDOUT:     %bound_method.loc38_41.10: <bound method> = bound_method %var, %impl.elem2.loc38_41.2
// CHECK:STDOUT:     %.loc38_41.24: %Destroy.type = converted constants.%as_type.de5, constants.%impl.elem1.a2e [symbolic = %impl.elem1 (constants.%impl.elem1.a2e)]
// CHECK:STDOUT:     %.loc38_41.25: %Destroy.type = converted constants.%as_type.de5, constants.%impl.elem1.a2e [symbolic = %impl.elem1 (constants.%impl.elem1.a2e)]
// CHECK:STDOUT:     %specific_impl_fn.loc38_41.3: <specific function> = specific_impl_function %impl.elem2.loc38_41.2, @Destroy.WithSelf.SelfDestruct(constants.%impl.elem1.a2e) [symbolic = %specific_impl_fn.loc38_41.6 (constants.%specific_impl_fn.f62)]
// CHECK:STDOUT:     %bound_method.loc38_41.11: <bound method> = bound_method %var, %specific_impl_fn.loc38_41.3
// CHECK:STDOUT:     %Destroy.WithSelf.SelfDestruct.call.loc38_41.4: init %empty_tuple.type = call %bound_method.loc38_41.11(%var)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc38_41.1(%self.param: ref %.8f3) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc38_41.2(%self.param: ref %MaybeUnformed.b96) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc38_41.3(%self.param: ref bool) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc38_41.4(%self.param: ref %struct_type.value.has_value.9db) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc38_41.5(%self.param: ref %DefaultOptionalStorage.1da) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc38_41.6(%self.param: ref %Optional.164) {
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
// CHECK:STDOUT:   %impl.elem2: %.bd7 = impl_witness_access constants.%Iterate.impl_witness.afa, element2 [concrete = constants.%T.as_type.as.Iterate.impl.NewCursor.415]
// CHECK:STDOUT:   %bound_method.loc80_42.1: <bound method> = bound_method %mc.ref, %impl.elem2
// CHECK:STDOUT:   %facet_value.loc80_42.1: %CppRange_where.type.d682d6.1 = facet_value constants.%MutableAndConstRange, (constants.%custom_witness.0aa, constants.%custom_witness.1c5, constants.%custom_witness.660, constants.%custom_witness.955, constants.%custom_witness.d56, constants.%custom_witness.68a, constants.%custom_witness.2ef) [concrete = constants.%facet_value.b69]
// CHECK:STDOUT:   %.loc80_42.1: %CppRange_where.type.d682d6.1 = converted constants.%MutableAndConstRange, %facet_value.loc80_42.1 [concrete = constants.%facet_value.b69]
// CHECK:STDOUT:   %facet_value.loc80_42.2: %CppRange_where.type.d682d6.1 = facet_value constants.%MutableAndConstRange, (constants.%custom_witness.0aa, constants.%custom_witness.1c5, constants.%custom_witness.660, constants.%custom_witness.955, constants.%custom_witness.d56, constants.%custom_witness.68a, constants.%custom_witness.2ef) [concrete = constants.%facet_value.b69]
// CHECK:STDOUT:   %.loc80_42.2: %CppRange_where.type.d682d6.1 = converted constants.%MutableAndConstRange, %facet_value.loc80_42.2 [concrete = constants.%facet_value.b69]
// CHECK:STDOUT:   %specific_fn.loc80_42.1: <specific function> = specific_function %impl.elem2, @T.as_type.as.Iterate.impl.NewCursor(constants.%facet_value.b69) [concrete = constants.%T.as_type.as.Iterate.impl.NewCursor.specific_fn]
// CHECK:STDOUT:   %bound_method.loc80_42.2: <bound method> = bound_method %mc.ref, %specific_fn.loc80_42.1
// CHECK:STDOUT:   %var: ref %tuple.type.de6 = var_storage invalid
// CHECK:STDOUT:   %.loc80_40.1: %MutableAndConstRange = acquire_value %mc.ref
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.call: init %tuple.type.de6 to %var = call %bound_method.loc80_42.2(%.loc80_40.1)
// CHECK:STDOUT:   assign %var, %T.as_type.as.Iterate.impl.NewCursor.call
// CHECK:STDOUT:   br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT: !for.next:
// CHECK:STDOUT:   %addr.loc80: %ptr.b95 = addr_of %var
// CHECK:STDOUT:   %impl.elem3: %.2c8 = impl_witness_access constants.%Iterate.impl_witness.afa, element3 [concrete = constants.%T.as_type.as.Iterate.impl.Next.872]
// CHECK:STDOUT:   %bound_method.loc80_42.3: <bound method> = bound_method %mc.ref, %impl.elem3
// CHECK:STDOUT:   %facet_value.loc80_42.3: %CppRange_where.type.d682d6.1 = facet_value constants.%MutableAndConstRange, (constants.%custom_witness.0aa, constants.%custom_witness.1c5, constants.%custom_witness.660, constants.%custom_witness.955, constants.%custom_witness.d56, constants.%custom_witness.68a, constants.%custom_witness.2ef) [concrete = constants.%facet_value.b69]
// CHECK:STDOUT:   %.loc80_42.3: %CppRange_where.type.d682d6.1 = converted constants.%MutableAndConstRange, %facet_value.loc80_42.3 [concrete = constants.%facet_value.b69]
// CHECK:STDOUT:   %facet_value.loc80_42.4: %CppRange_where.type.d682d6.1 = facet_value constants.%MutableAndConstRange, (constants.%custom_witness.0aa, constants.%custom_witness.1c5, constants.%custom_witness.660, constants.%custom_witness.955, constants.%custom_witness.d56, constants.%custom_witness.68a, constants.%custom_witness.2ef) [concrete = constants.%facet_value.b69]
// CHECK:STDOUT:   %.loc80_42.4: %CppRange_where.type.d682d6.1 = converted constants.%MutableAndConstRange, %facet_value.loc80_42.4 [concrete = constants.%facet_value.b69]
// CHECK:STDOUT:   %specific_fn.loc80_42.2: <specific function> = specific_function %impl.elem3, @T.as_type.as.Iterate.impl.Next(constants.%facet_value.b69) [concrete = constants.%T.as_type.as.Iterate.impl.Next.specific_fn]
// CHECK:STDOUT:   %bound_method.loc80_42.4: <bound method> = bound_method %mc.ref, %specific_fn.loc80_42.2
// CHECK:STDOUT:   %.loc80_42.5: ref %Optional.164 = temporary_storage
// CHECK:STDOUT:   %.loc80_40.2: %MutableAndConstRange = acquire_value %mc.ref
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.call: init %Optional.164 to %.loc80_42.5 = call %bound_method.loc80_42.4(%.loc80_40.2, %addr.loc80)
// CHECK:STDOUT:   %.loc80_42.6: ref %Optional.164 = temporary %.loc80_42.5, %T.as_type.as.Iterate.impl.Next.call
// CHECK:STDOUT:   %.loc80_42.7: %Optional.HasValue.type.c93 = specific_constant imports.%Core.import_ref.bb6, @Optional(constants.%OptionalStorage.facet.193) [concrete = constants.%Optional.HasValue.32a]
// CHECK:STDOUT:   %HasValue.ref: %Optional.HasValue.type.c93 = name_ref HasValue, %.loc80_42.7 [concrete = constants.%Optional.HasValue.32a]
// CHECK:STDOUT:   %Optional.HasValue.bound: <bound method> = bound_method %.loc80_42.6, %HasValue.ref
// CHECK:STDOUT:   %Optional.HasValue.specific_fn: <specific function> = specific_function %HasValue.ref, @Optional.HasValue(constants.%OptionalStorage.facet.193) [concrete = constants.%Optional.HasValue.specific_fn.faa]
// CHECK:STDOUT:   %bound_method.loc80_42.5: <bound method> = bound_method %.loc80_42.6, %Optional.HasValue.specific_fn
// CHECK:STDOUT:   %.loc80_42.8: %Optional.164 = acquire_value %.loc80_42.6
// CHECK:STDOUT:   %Optional.HasValue.call: init bool = call %bound_method.loc80_42.5(%.loc80_42.8)
// CHECK:STDOUT:   %.loc80_42.9: bool = value_of_initializer %Optional.HasValue.call
// CHECK:STDOUT:   %.loc80_42.10: bool = converted %Optional.HasValue.call, %.loc80_42.9
// CHECK:STDOUT:   if %.loc80_42.10 br !for.body else br !for.done
// CHECK:STDOUT:
// CHECK:STDOUT: !for.body:
// CHECK:STDOUT:   %.loc80_42.11: %Optional.Get.type.7cd = specific_constant imports.%Core.import_ref.dab, @Optional(constants.%OptionalStorage.facet.193) [concrete = constants.%Optional.Get.4de]
// CHECK:STDOUT:   %Get.ref: %Optional.Get.type.7cd = name_ref Get, %.loc80_42.11 [concrete = constants.%Optional.Get.4de]
// CHECK:STDOUT:   %Optional.Get.bound: <bound method> = bound_method %.loc80_42.6, %Get.ref
// CHECK:STDOUT:   %Optional.Get.specific_fn: <specific function> = specific_function %Get.ref, @Optional.Get(constants.%OptionalStorage.facet.193) [concrete = constants.%Optional.Get.specific_fn.7ee]
// CHECK:STDOUT:   %bound_method.loc80_42.6: <bound method> = bound_method %.loc80_42.6, %Optional.Get.specific_fn
// CHECK:STDOUT:   %.loc80_42.12: ref %ValueType = temporary_storage
// CHECK:STDOUT:   %.loc80_42.13: %Optional.164 = acquire_value %.loc80_42.6
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
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc80_42.1: <bound method> = bound_method %.loc80_42.14, constants.%Destroy.WithSelf.SelfDestruct.2f2
// CHECK:STDOUT:   %Destroy.facet.loc80_42.1: %Destroy.type = facet_value constants.%ValueType, (constants.%custom_witness.0aa) [concrete = constants.%Destroy.facet.0bd]
// CHECK:STDOUT:   %.loc80_42.16: %Destroy.type = converted constants.%ValueType, %Destroy.facet.loc80_42.1 [concrete = constants.%Destroy.facet.0bd]
// CHECK:STDOUT:   %Destroy.facet.loc80_42.2: %Destroy.type = facet_value constants.%ValueType, (constants.%custom_witness.0aa) [concrete = constants.%Destroy.facet.0bd]
// CHECK:STDOUT:   %.loc80_42.17: %Destroy.type = converted constants.%ValueType, %Destroy.facet.loc80_42.2 [concrete = constants.%Destroy.facet.0bd]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc80_42.7: <bound method> = bound_method %.loc80_42.14, %Destroy.WithSelf.SelfDestruct.specific_fn.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc80_42.1: init %empty_tuple.type = call %bound_method.loc80_42.7(%.loc80_42.14)
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc80_42.2: <bound method> = bound_method %.loc80_42.6, constants.%Destroy.WithSelf.SelfDestruct.3ed
// CHECK:STDOUT:   %Destroy.facet.loc80_42.3: %Destroy.type = facet_value constants.%Optional.164, (constants.%custom_witness.eb5) [concrete = constants.%Destroy.facet.8c1]
// CHECK:STDOUT:   %.loc80_42.18: %Destroy.type = converted constants.%Optional.164, %Destroy.facet.loc80_42.3 [concrete = constants.%Destroy.facet.8c1]
// CHECK:STDOUT:   %Destroy.facet.loc80_42.4: %Destroy.type = facet_value constants.%Optional.164, (constants.%custom_witness.eb5) [concrete = constants.%Destroy.facet.8c1]
// CHECK:STDOUT:   %.loc80_42.19: %Destroy.type = converted constants.%Optional.164, %Destroy.facet.loc80_42.4 [concrete = constants.%Destroy.facet.8c1]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc80_42.8: <bound method> = bound_method %.loc80_42.6, %Destroy.WithSelf.SelfDestruct.specific_fn.2
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc80_42.2: init %empty_tuple.type = call %bound_method.loc80_42.8(%.loc80_42.6)
// CHECK:STDOUT:   br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT: !for.done:
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc80_42.3: <bound method> = bound_method %.loc80_42.6, constants.%Destroy.WithSelf.SelfDestruct.3ed
// CHECK:STDOUT:   %Destroy.facet.loc80_42.5: %Destroy.type = facet_value constants.%Optional.164, (constants.%custom_witness.eb5) [concrete = constants.%Destroy.facet.8c1]
// CHECK:STDOUT:   %.loc80_42.20: %Destroy.type = converted constants.%Optional.164, %Destroy.facet.loc80_42.5 [concrete = constants.%Destroy.facet.8c1]
// CHECK:STDOUT:   %Destroy.facet.loc80_42.6: %Destroy.type = facet_value constants.%Optional.164, (constants.%custom_witness.eb5) [concrete = constants.%Destroy.facet.8c1]
// CHECK:STDOUT:   %.loc80_42.21: %Destroy.type = converted constants.%Optional.164, %Destroy.facet.loc80_42.6 [concrete = constants.%Destroy.facet.8c1]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc80_42.9: <bound method> = bound_method %.loc80_42.6, %Destroy.WithSelf.SelfDestruct.specific_fn.3
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc80_42.3: init %empty_tuple.type = call %bound_method.loc80_42.9(%.loc80_42.6)
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc80_42.4: <bound method> = bound_method %var, constants.%Destroy.WithSelf.SelfDestruct.f74
// CHECK:STDOUT:   %Destroy.facet.loc80_42.7: %Destroy.type = facet_value constants.%tuple.type.de6, (constants.%custom_witness.a91) [concrete = constants.%Destroy.facet.992]
// CHECK:STDOUT:   %.loc80_42.22: %Destroy.type = converted constants.%tuple.type.de6, %Destroy.facet.loc80_42.7 [concrete = constants.%Destroy.facet.992]
// CHECK:STDOUT:   %Destroy.facet.loc80_42.8: %Destroy.type = facet_value constants.%tuple.type.de6, (constants.%custom_witness.a91) [concrete = constants.%Destroy.facet.992]
// CHECK:STDOUT:   %.loc80_42.23: %Destroy.type = converted constants.%tuple.type.de6, %Destroy.facet.loc80_42.8 [concrete = constants.%Destroy.facet.992]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc80_42.10: <bound method> = bound_method %var, %Destroy.WithSelf.SelfDestruct.specific_fn.4
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc80_42.4: init %empty_tuple.type = call %bound_method.loc80_42.10(%var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestIterate(constants.%R) {
// CHECK:STDOUT:   %R.patt.loc34_17.2 => constants.%R.patt
// CHECK:STDOUT:   %R.loc34_17.1 => constants.%R
// CHECK:STDOUT:   %R.as_type.loc34_83.1 => constants.%R.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.dac
// CHECK:STDOUT:   %r.param_patt.loc34_81.2 => constants.%r.param_patt.769
// CHECK:STDOUT:   %r.patt.loc34_81.2 => constants.%r.patt.6b2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestIterate(constants.%facet_value.f2e) {
// CHECK:STDOUT:   %R.patt.loc34_17.2 => constants.%R.patt
// CHECK:STDOUT:   %R.loc34_17.1 => constants.%facet_value.f2e
// CHECK:STDOUT:   %R.as_type.loc34_83.1 => constants.%MutableAndConstRange
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.fdb
// CHECK:STDOUT:   %r.param_patt.loc34_81.2 => constants.%r.param_patt.3a6
// CHECK:STDOUT:   %r.patt.loc34_81.2 => constants.%r.patt.403
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc34 => constants.%complete_type.357
// CHECK:STDOUT:   %Iterate.lookup_impl_witness => constants.%Iterate.impl_witness.afa
// CHECK:STDOUT:   %Iterate.facet.loc38_41.5 => constants.%Iterate.facet.3f8
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type => constants.%Iterate.WithSelf.NewCursor.type.735
// CHECK:STDOUT:   %.loc38_41.26 => constants.%.bd7
// CHECK:STDOUT:   %impl.elem2.loc38_41.3 => constants.%T.as_type.as.Iterate.impl.NewCursor.415
// CHECK:STDOUT:   %specific_impl_fn.loc38_41.4 => constants.%T.as_type.as.Iterate.impl.NewCursor.specific_fn
// CHECK:STDOUT:   %impl.elem1 => constants.%Destroy.facet.992
// CHECK:STDOUT:   %as_type => constants.%tuple.type.de6
// CHECK:STDOUT:   %require_complete.1 => constants.%complete_type.95d
// CHECK:STDOUT:   %ptr => constants.%ptr.b95
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type => constants.%Iterate.WithSelf.Next.type.9c9
// CHECK:STDOUT:   %.loc38_41.27 => constants.%.2c8
// CHECK:STDOUT:   %impl.elem3.loc38_41.2 => constants.%T.as_type.as.Iterate.impl.Next.872
// CHECK:STDOUT:   %specific_impl_fn.loc38_41.5 => constants.%T.as_type.as.Iterate.impl.Next.specific_fn
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type => constants.%Destroy.WithSelf.SelfDestruct.type.42d
// CHECK:STDOUT:   %.loc38_41.28 => constants.%.5c5
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.a91
// CHECK:STDOUT:   %impl.elem2.loc38_41.4 => constants.%Destroy.WithSelf.SelfDestruct.f74
// CHECK:STDOUT:   %specific_impl_fn.loc38_41.6 => constants.%Destroy.WithSelf.SelfDestruct.specific_fn.13b
// CHECK:STDOUT: }
// CHECK:STDOUT:
