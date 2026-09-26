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
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %Iterate.type: type = facet_type <@Iterate> [concrete]
// CHECK:STDOUT:   %OptionalStorage.type: type = facet_type <@OptionalStorage> [concrete]
// CHECK:STDOUT:   %pattern_type.831: type = pattern_type bool [concrete]
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
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.1: type = fn_type @Destroy.WithSelf.SelfDestruct.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.1: %Destroy.WithSelf.SelfDestruct.type.fbceb5.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.1: type = fn_type @Destroy.WithSelf.SubobjectDestroy.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.1: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.1: type = fn_type @Destroy.WithSelf.Op.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.1: %Destroy.WithSelf.Op.type.ef016f.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.f8f19d.1: <witness> = custom_witness (%Destroy.WithSelf.Op.403171.1, %Destroy.WithSelf.SubobjectDestroy.d01daf.1, %Destroy.WithSelf.SelfDestruct.db3fdb.1), @Destroy [concrete]
// CHECK:STDOUT:   %Destroy.facet.b41: %Destroy.type = facet_value %i32, (%custom_witness.f8f19d.1) [concrete]
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
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.ac8: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N.fe9) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.5e0: %Int.as.Copy.impl.Op.type.ac8 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.b51: <witness> = impl_witness imports.%Copy.impl_witness_table.8d2, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %facet_value.178: %facet_type.f48 = facet_value %i32, (%custom_witness.f8f19d.1, %Copy.impl_witness.b51) [concrete]
// CHECK:STDOUT:   %Iterate.lookup_impl_witness.1d8: <witness> = lookup_impl_witness %.Self.500, @Iterate [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.294: %facet_type.f48 = impl_witness_access %Iterate.lookup_impl_witness.1d8, element0 [symbolic_self]
// CHECK:STDOUT:   %Iterate_where.type.16e: type = facet_type <@Iterate where %impl.elem0.294 = %facet_value.178> [concrete]
// CHECK:STDOUT:   %pattern_type.c32: type = pattern_type %Iterate_where.type.16e [concrete]
// CHECK:STDOUT:   %R.patt: %pattern_type.c32 = symbolic_binding_pattern R, 0 [symbolic]
// CHECK:STDOUT:   %R: %Iterate_where.type.16e = symbolic_binding R, 0 [symbolic]
// CHECK:STDOUT:   %R.as_type: type = facet_access_type %R [symbolic]
// CHECK:STDOUT:   %pattern_type.6f9: type = pattern_type %R.as_type [symbolic]
// CHECK:STDOUT:   %r.param_patt.170: %pattern_type.6f9 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %r.patt.893: %pattern_type.6f9 = wrapper_binding_pattern r, %r.param_patt.170 [symbolic]
// CHECK:STDOUT:   %facet_type.037: type = facet_type <@IntFitsIn, @IntFitsIn(%Int.67af24.2) & @AnyInt> [symbolic]
// CHECK:STDOUT:   %From.6a4: %facet_type.037 = symbolic_binding From, 1 [symbolic]
// CHECK:STDOUT:   %From.as_type.as.ImplicitAs.impl.Convert.type.7fa: type = fn_type @From.as_type.as.ImplicitAs.impl.Convert, @From.as_type.as.ImplicitAs.impl(%To, %From.6a4) [symbolic]
// CHECK:STDOUT:   %From.as_type.as.ImplicitAs.impl.Convert.643: %From.as_type.as.ImplicitAs.impl.Convert.type.7fa = struct_value () [symbolic]
// CHECK:STDOUT:   %i.patt: %pattern_type.6b6 = value_binding_pattern i [concrete]
// CHECK:STDOUT:   %Iterate.lookup_impl_witness.462: <witness> = lookup_impl_witness %R, @Iterate [symbolic]
// CHECK:STDOUT:   %Iterate.facet.e5e: %Iterate.type = facet_value %R.as_type, (%Iterate.lookup_impl_witness.462) [symbolic]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.62b: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.e5e) [symbolic]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.6c3: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.e5e) [symbolic]
// CHECK:STDOUT:   %.1a4: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.62b, %Iterate.facet.e5e [symbolic]
// CHECK:STDOUT:   %impl.elem2.3eb: %.1a4 = impl_witness_access %Iterate.lookup_impl_witness.462, element2 [symbolic]
// CHECK:STDOUT:   %impl.elem1.a11: %Destroy.type = impl_witness_access %Iterate.lookup_impl_witness.462, element1 [symbolic]
// CHECK:STDOUT:   %as_type.f79: type = facet_access_type %impl.elem1.a11 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.a09: <specific function> = specific_impl_function %impl.elem2.3eb, @Iterate.WithSelf.NewCursor(%Iterate.facet.e5e) [symbolic]
// CHECK:STDOUT:   %ptr.b18: type = ptr_type %as_type.f79 [symbolic]
// CHECK:STDOUT:   %.218: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.6c3, %Iterate.facet.e5e [symbolic]
// CHECK:STDOUT:   %impl.elem3.b50: %.218 = impl_witness_access %Iterate.lookup_impl_witness.462, element3 [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.9f1: type = class_type @DefaultOptionalStorage, @DefaultOptionalStorage(%facet_value.178) [concrete]
// CHECK:STDOUT:   %OptionalStorage.impl_witness.3c7: <witness> = impl_witness imports.%OptionalStorage.impl_witness_table.460, @T.as_type.as.OptionalStorage.impl(%facet_value.178) [concrete]
// CHECK:STDOUT:   %OptionalStorage.facet.6c5: %OptionalStorage.type = facet_value %i32, (%OptionalStorage.impl_witness.3c7) [concrete]
// CHECK:STDOUT:   %Optional.0ac: type = class_type @Optional, @Optional(%OptionalStorage.facet.6c5) [concrete]
// CHECK:STDOUT:   %pattern_type.022: type = pattern_type %Optional.0ac [concrete]
// CHECK:STDOUT:   %specific_impl_fn.598: <specific function> = specific_impl_function %impl.elem3.b50, @Iterate.WithSelf.Next(%Iterate.facet.e5e) [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.type.dbd: type = fn_type @Optional.HasValue, @Optional(%OptionalStorage.facet.6c5) [concrete]
// CHECK:STDOUT:   %Optional.HasValue.ed9: %Optional.HasValue.type.dbd = struct_value () [concrete]
// CHECK:STDOUT:   %Optional.Get.type.2b4: type = fn_type @Optional.Get, @Optional(%OptionalStorage.facet.6c5) [concrete]
// CHECK:STDOUT:   %Optional.Get.529: %Optional.Get.type.2b4 = struct_value () [concrete]
// CHECK:STDOUT:   %MaybeUnformed.515: type = class_type @MaybeUnformed, @MaybeUnformed(%Destroy.facet.b41) [concrete]
// CHECK:STDOUT:   %.1d6a7: type = maybe_unformed_type %i32 [concrete]
// CHECK:STDOUT:   %struct_type.value.has_value.f6a: type = struct_type {.value: %MaybeUnformed.515, .has_value: bool} [concrete]
// CHECK:STDOUT:   %Optional.HasValue.specific_fn.15f: <specific function> = specific_function %Optional.HasValue.ed9, @Optional.HasValue(%OptionalStorage.facet.6c5) [concrete]
// CHECK:STDOUT:   %Optional.Get.specific_fn.0dd: <specific function> = specific_function %Optional.Get.529, @Optional.Get(%OptionalStorage.facet.6c5) [concrete]
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
// CHECK:STDOUT:   %pattern_type.f40: type = pattern_type %.1d6a7 [concrete]
// CHECK:STDOUT:   %self.param_patt.af3: %pattern_type.f40 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.18f: %pattern_type.f40 = wrapper_binding_pattern self, %self.param_patt.af3 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.3: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc34_19.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.3: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.3 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.3: type = fn_type @Destroy.WithSelf.Op.loc34_19.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.3: %Destroy.WithSelf.Op.type.ef016f.3 = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.7cb: type = pattern_type %MaybeUnformed.515 [concrete]
// CHECK:STDOUT:   %self.param_patt.5b1: %pattern_type.7cb = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.797: %pattern_type.7cb = wrapper_binding_pattern self, %self.param_patt.5b1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.4: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc34_19.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.4: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.4 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.4: type = fn_type @Destroy.WithSelf.Op.loc34_19.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.4: %Destroy.WithSelf.Op.type.ef016f.4 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.4: type = fn_type @Destroy.WithSelf.SelfDestruct.loc34_19.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.4: %Destroy.WithSelf.SelfDestruct.type.fbceb5.4 = struct_value () [concrete]
// CHECK:STDOUT:   %self.param_patt.40d: %pattern_type.831 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.560: %pattern_type.831 = wrapper_binding_pattern self, %self.param_patt.40d [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.5: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc34_19.3 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.5: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.5 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.5: type = fn_type @Destroy.WithSelf.Op.loc34_19.3 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.5: %Destroy.WithSelf.Op.type.ef016f.5 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.5: type = fn_type @Destroy.WithSelf.SelfDestruct.loc34_19.3 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.5: %Destroy.WithSelf.SelfDestruct.type.fbceb5.5 = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.260: type = pattern_type %struct_type.value.has_value.f6a [concrete]
// CHECK:STDOUT:   %self.param_patt.e1f: %pattern_type.260 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.d8d: %pattern_type.260 = wrapper_binding_pattern self, %self.param_patt.e1f [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.6: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc34_19.4 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.6: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.6 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.6: type = fn_type @Destroy.WithSelf.Op.loc34_19.4 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.6: %Destroy.WithSelf.Op.type.ef016f.6 = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.96a: type = pattern_type %DefaultOptionalStorage.9f1 [concrete]
// CHECK:STDOUT:   %self.param_patt.333: %pattern_type.96a = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.00e: %pattern_type.96a = wrapper_binding_pattern self, %self.param_patt.333 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.7: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc34_19.5 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.7: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.7 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.7: type = fn_type @Destroy.WithSelf.Op.loc34_19.5 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.7: %Destroy.WithSelf.Op.type.ef016f.7 = struct_value () [concrete]
// CHECK:STDOUT:   %self.param_patt.d39: %pattern_type.022 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.046: %pattern_type.022 = wrapper_binding_pattern self, %self.param_patt.d39 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.8: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc34_19.6 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.8: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.8 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.8: type = fn_type @Destroy.WithSelf.Op.loc34_19.6 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.8: %Destroy.WithSelf.Op.type.ef016f.8 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.8: type = fn_type @Destroy.WithSelf.SelfDestruct.loc34_19.6 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.8: %Destroy.WithSelf.SelfDestruct.type.fbceb5.8 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.caf: <witness> = lookup_impl_witness %impl.elem1.a11, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.041: type = fn_type @Destroy.WithSelf.SelfDestruct.1, @Destroy.WithSelf(%impl.elem1.a11) [symbolic]
// CHECK:STDOUT:   %.b28: type = fn_type_with_self_type %Destroy.WithSelf.SelfDestruct.type.041, %impl.elem1.a11 [symbolic]
// CHECK:STDOUT:   %impl.elem2.9a0: %.b28 = impl_witness_access %Destroy.lookup_impl_witness.caf, element2 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.ea2: <specific function> = specific_impl_function %impl.elem2.9a0, @Destroy.WithSelf.SelfDestruct.1(%impl.elem1.a11) [symbolic]
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
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.9: type = fn_type @Destroy.WithSelf.Op.3 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.9: %Destroy.WithSelf.Op.type.ef016f.9 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.9: type = fn_type @Destroy.WithSelf.SelfDestruct.3 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.9: %Destroy.WithSelf.SelfDestruct.type.fbceb5.9 = struct_value () [concrete]
// CHECK:STDOUT:   %Iterator.SubobjectDestroy.type.9ec: type = fn_type @Iterator.SubobjectDestroy.1 [concrete]
// CHECK:STDOUT:   %Iterator.SubobjectDestroy.e52: %Iterator.SubobjectDestroy.type.9ec = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.8e1: <witness> = custom_witness (%Destroy.WithSelf.Op.403171.9, %Iterator.SubobjectDestroy.e52, %Destroy.WithSelf.SelfDestruct.db3fdb.9), @Destroy [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.0aed5d.2: type = fn_type @Iterator.Op.2 [concrete]
// CHECK:STDOUT:   %Iterator.Op.a2a866.2: %Iterator.Op.type.0aed5d.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.444: <witness> = custom_witness (%Iterator.Op.a2a866.2), @Inc [concrete]
// CHECK:STDOUT:   %Equal.type.ca8475.1: type = fn_type @Equal.1 [concrete]
// CHECK:STDOUT:   %Equal.285b15.1: %Equal.type.ca8475.1 = struct_value () [concrete]
// CHECK:STDOUT:   %NotEqual.type.53d9ee.1: type = fn_type @NotEqual.1 [concrete]
// CHECK:STDOUT:   %NotEqual.119263.1: %NotEqual.type.53d9ee.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.981: <witness> = custom_witness (%Equal.285b15.1, %NotEqual.119263.1), @EqWith, @EqWith(%Iterator.3ee) [concrete]
// CHECK:STDOUT:   %facet_value.474: %CppRange_where.type.d682d6.1 = facet_value %MutableRange, (%custom_witness.f8f19d.1, %Copy.impl_witness.b51, %custom_witness.ed3, %custom_witness.8e1, %custom_witness.ef8, %custom_witness.444, %custom_witness.981) [concrete]
// CHECK:STDOUT:   %tuple.type.508: type = tuple_type (%Iterator.3ee, %Iterator.3ee) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.9: type = fn_type @Destroy.WithSelf.SubobjectDestroy.3 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.9: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.9 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.10: type = fn_type @Destroy.WithSelf.Op.4 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.10: %Destroy.WithSelf.Op.type.ef016f.10 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.10: type = fn_type @Destroy.WithSelf.SelfDestruct.4 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.10: %Destroy.WithSelf.SelfDestruct.type.fbceb5.10 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.f8f19d.9: <witness> = custom_witness (%Destroy.WithSelf.Op.403171.10, %Destroy.WithSelf.SubobjectDestroy.d01daf.9, %Destroy.WithSelf.SelfDestruct.db3fdb.10), @Destroy [concrete]
// CHECK:STDOUT:   %Destroy.facet.0fe: %Destroy.type = facet_value %tuple.type.508, (%custom_witness.f8f19d.9) [concrete]
// CHECK:STDOUT:   %Iterate.impl_witness.2bb: <witness> = impl_witness imports.%Iterate.impl_witness_table.121, @T.as_type.as.Iterate.impl(%facet_value.474) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.type.2c7: type = fn_type @T.as_type.as.Iterate.impl.NewCursor, @T.as_type.as.Iterate.impl(%facet_value.474) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.e00: %T.as_type.as.Iterate.impl.NewCursor.type.2c7 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.type.245: type = fn_type @T.as_type.as.Iterate.impl.Next, @T.as_type.as.Iterate.impl(%facet_value.474) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.311: %T.as_type.as.Iterate.impl.Next.type.245 = struct_value () [concrete]
// CHECK:STDOUT:   %Iterate.facet.854: %Iterate.type = facet_value %MutableRange, (%Iterate.impl_witness.2bb) [concrete]
// CHECK:STDOUT:   %facet_value.26c: %Iterate_where.type.16e = facet_value %MutableRange, (%Iterate.impl_witness.2bb) [concrete]
// CHECK:STDOUT:   %r.param_patt.c79: %pattern_type.e635 = value_param_pattern [concrete]
// CHECK:STDOUT:   %r.patt.94f: %pattern_type.e635 = wrapper_binding_pattern r, %r.param_patt.c79 [concrete]
// CHECK:STDOUT:   %Iterator.260: type = class_type @Iterator.2 [concrete]
// CHECK:STDOUT:   %ConstRange.Begin.type: type = fn_type @ConstRange.Begin [concrete]
// CHECK:STDOUT:   %ConstRange.Begin: %ConstRange.Begin.type = struct_value () [concrete]
// CHECK:STDOUT:   %ConstRange.End.type: type = fn_type @ConstRange.End [concrete]
// CHECK:STDOUT:   %ConstRange.End: %ConstRange.End.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.23e: <witness> = custom_witness (%Iterator.260, %Iterator.260, %ConstRange.Begin, %ConstRange.End), @CppRangeForIterate [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.0f34b8.1: type = fn_type @Iterator.Op.3 [concrete]
// CHECK:STDOUT:   %Iterator.Op.1c3e4a.1: %Iterator.Op.type.0f34b8.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.adb: <witness> = custom_witness (%i32, %Iterator.Op.1c3e4a.1), @CppUnsafeDeref [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.11: type = fn_type @Destroy.WithSelf.Op.5 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.11: %Destroy.WithSelf.Op.type.ef016f.11 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.11: type = fn_type @Destroy.WithSelf.SelfDestruct.5 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.11: %Destroy.WithSelf.SelfDestruct.type.fbceb5.11 = struct_value () [concrete]
// CHECK:STDOUT:   %Iterator.SubobjectDestroy.type.128: type = fn_type @Iterator.SubobjectDestroy.2 [concrete]
// CHECK:STDOUT:   %Iterator.SubobjectDestroy.37b: %Iterator.SubobjectDestroy.type.128 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.0f3: <witness> = custom_witness (%Destroy.WithSelf.Op.403171.11, %Iterator.SubobjectDestroy.37b, %Destroy.WithSelf.SelfDestruct.db3fdb.11), @Destroy [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.0f34b8.2: type = fn_type @Iterator.Op.4 [concrete]
// CHECK:STDOUT:   %Iterator.Op.1c3e4a.2: %Iterator.Op.type.0f34b8.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.746: <witness> = custom_witness (%Iterator.Op.1c3e4a.2), @Inc [concrete]
// CHECK:STDOUT:   %Equal.type.ca8475.2: type = fn_type @Equal.2 [concrete]
// CHECK:STDOUT:   %Equal.285b15.2: %Equal.type.ca8475.2 = struct_value () [concrete]
// CHECK:STDOUT:   %NotEqual.type.53d9ee.2: type = fn_type @NotEqual.2 [concrete]
// CHECK:STDOUT:   %NotEqual.119263.2: %NotEqual.type.53d9ee.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.941: <witness> = custom_witness (%Equal.285b15.2, %NotEqual.119263.2), @EqWith, @EqWith(%Iterator.260) [concrete]
// CHECK:STDOUT:   %facet_value.3e1: %CppRange_where.type.d682d6.1 = facet_value %ConstRange, (%custom_witness.f8f19d.1, %Copy.impl_witness.b51, %custom_witness.23e, %custom_witness.0f3, %custom_witness.adb, %custom_witness.746, %custom_witness.941) [concrete]
// CHECK:STDOUT:   %tuple.type.69f: type = tuple_type (%Iterator.260, %Iterator.260) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.10: type = fn_type @Destroy.WithSelf.SubobjectDestroy.4 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.10: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.10 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.12: type = fn_type @Destroy.WithSelf.Op.6 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.12: %Destroy.WithSelf.Op.type.ef016f.12 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.12: type = fn_type @Destroy.WithSelf.SelfDestruct.6 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.12: %Destroy.WithSelf.SelfDestruct.type.fbceb5.12 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.f8f19d.10: <witness> = custom_witness (%Destroy.WithSelf.Op.403171.12, %Destroy.WithSelf.SubobjectDestroy.d01daf.10, %Destroy.WithSelf.SelfDestruct.db3fdb.12), @Destroy [concrete]
// CHECK:STDOUT:   %Destroy.facet.c6d: %Destroy.type = facet_value %tuple.type.69f, (%custom_witness.f8f19d.10) [concrete]
// CHECK:STDOUT:   %Iterate.impl_witness.1a6: <witness> = impl_witness imports.%Iterate.impl_witness_table.121, @T.as_type.as.Iterate.impl(%facet_value.3e1) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.type.31f: type = fn_type @T.as_type.as.Iterate.impl.NewCursor, @T.as_type.as.Iterate.impl(%facet_value.3e1) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.19e: %T.as_type.as.Iterate.impl.NewCursor.type.31f = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.type.aeb: type = fn_type @T.as_type.as.Iterate.impl.Next, @T.as_type.as.Iterate.impl(%facet_value.3e1) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.372: %T.as_type.as.Iterate.impl.Next.type.aeb = struct_value () [concrete]
// CHECK:STDOUT:   %Iterate.facet.7d1: %Iterate.type = facet_value %ConstRange, (%Iterate.impl_witness.1a6) [concrete]
// CHECK:STDOUT:   %facet_value.75e: %Iterate_where.type.16e = facet_value %ConstRange, (%Iterate.impl_witness.1a6) [concrete]
// CHECK:STDOUT:   %r.param_patt.83e: %pattern_type.0ba = value_param_pattern [concrete]
// CHECK:STDOUT:   %r.patt.c21: %pattern_type.0ba = wrapper_binding_pattern r, %r.param_patt.83e [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.4e8: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.854) [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.713: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.854) [concrete]
// CHECK:STDOUT:   %.cad: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.4e8, %Iterate.facet.854 [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.specific_fn.ac2: <specific function> = specific_function %T.as_type.as.Iterate.impl.NewCursor.e00, @T.as_type.as.Iterate.impl.NewCursor(%facet_value.474) [concrete]
// CHECK:STDOUT:   %ptr.45c: type = ptr_type %tuple.type.508 [concrete]
// CHECK:STDOUT:   %.640: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.713, %Iterate.facet.854 [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.specific_fn.b01: <specific function> = specific_function %T.as_type.as.Iterate.impl.Next.311, @T.as_type.as.Iterate.impl.Next(%facet_value.474) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.471: type = fn_type @Destroy.WithSelf.SelfDestruct.1, @Destroy.WithSelf(%Destroy.facet.0fe) [concrete]
// CHECK:STDOUT:   %.463: type = fn_type_with_self_type %Destroy.WithSelf.SelfDestruct.type.471, %Destroy.facet.0fe [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.488: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.7d1) [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.aee: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.7d1) [concrete]
// CHECK:STDOUT:   %.312: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.488, %Iterate.facet.7d1 [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.specific_fn.91a: <specific function> = specific_function %T.as_type.as.Iterate.impl.NewCursor.19e, @T.as_type.as.Iterate.impl.NewCursor(%facet_value.3e1) [concrete]
// CHECK:STDOUT:   %ptr.c5c: type = ptr_type %tuple.type.69f [concrete]
// CHECK:STDOUT:   %.5b5: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.aee, %Iterate.facet.7d1 [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.specific_fn.5ef: <specific function> = specific_function %T.as_type.as.Iterate.impl.Next.372, @T.as_type.as.Iterate.impl.Next(%facet_value.3e1) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.c53: type = fn_type @Destroy.WithSelf.SelfDestruct.1, @Destroy.WithSelf(%Destroy.facet.c6d) [concrete]
// CHECK:STDOUT:   %.331: type = fn_type_with_self_type %Destroy.WithSelf.SelfDestruct.type.c53, %Destroy.facet.c6d [concrete]
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
// CHECK:STDOUT: generated {
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.2: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.3 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc34_19.1 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.3] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.f40 = ref_param_pattern [concrete = constants.%self.param_patt.af3]
// CHECK:STDOUT:     %self.patt: %pattern_type.f40 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.18f]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %.1d6a7 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %.1d6a7 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.2: %Destroy.WithSelf.Op.type.ef016f.3 = fn_decl @Destroy.WithSelf.Op.loc34_19.1 [concrete = constants.%Destroy.WithSelf.Op.403171.3] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.f40 = ref_param_pattern [concrete = constants.%self.param_patt.af3]
// CHECK:STDOUT:     %self.patt: %pattern_type.f40 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.18f]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %.1d6a7 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %.1d6a7 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.3: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.4 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc34_19.2 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.4] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.7cb = ref_param_pattern [concrete = constants.%self.param_patt.5b1]
// CHECK:STDOUT:     %self.patt: %pattern_type.7cb = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.797]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %MaybeUnformed.515 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %MaybeUnformed.515 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.3: %Destroy.WithSelf.Op.type.ef016f.4 = fn_decl @Destroy.WithSelf.Op.loc34_19.2 [concrete = constants.%Destroy.WithSelf.Op.403171.4] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.7cb = ref_param_pattern [concrete = constants.%self.param_patt.5b1]
// CHECK:STDOUT:     %self.patt: %pattern_type.7cb = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.797]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %MaybeUnformed.515 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %MaybeUnformed.515 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.4: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.5 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc34_19.3 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.5] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.831 = ref_param_pattern [concrete = constants.%self.param_patt.40d]
// CHECK:STDOUT:     %self.patt: %pattern_type.831 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.560]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref bool = ref_param call_param0
// CHECK:STDOUT:     %self: ref bool = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.4: %Destroy.WithSelf.Op.type.ef016f.5 = fn_decl @Destroy.WithSelf.Op.loc34_19.3 [concrete = constants.%Destroy.WithSelf.Op.403171.5] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.831 = ref_param_pattern [concrete = constants.%self.param_patt.40d]
// CHECK:STDOUT:     %self.patt: %pattern_type.831 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.560]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref bool = ref_param call_param0
// CHECK:STDOUT:     %self: ref bool = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.5: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.6 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc34_19.4 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.6] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.260 = ref_param_pattern [concrete = constants.%self.param_patt.e1f]
// CHECK:STDOUT:     %self.patt: %pattern_type.260 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.d8d]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %struct_type.value.has_value.f6a = ref_param call_param0
// CHECK:STDOUT:     %self: ref %struct_type.value.has_value.f6a = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.5: %Destroy.WithSelf.Op.type.ef016f.6 = fn_decl @Destroy.WithSelf.Op.loc34_19.4 [concrete = constants.%Destroy.WithSelf.Op.403171.6] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.260 = ref_param_pattern [concrete = constants.%self.param_patt.e1f]
// CHECK:STDOUT:     %self.patt: %pattern_type.260 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.d8d]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %struct_type.value.has_value.f6a = ref_param call_param0
// CHECK:STDOUT:     %self: ref %struct_type.value.has_value.f6a = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.6: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.7 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc34_19.5 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.7] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.96a = ref_param_pattern [concrete = constants.%self.param_patt.333]
// CHECK:STDOUT:     %self.patt: %pattern_type.96a = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.00e]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %DefaultOptionalStorage.9f1 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %DefaultOptionalStorage.9f1 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.6: %Destroy.WithSelf.Op.type.ef016f.7 = fn_decl @Destroy.WithSelf.Op.loc34_19.5 [concrete = constants.%Destroy.WithSelf.Op.403171.7] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.96a = ref_param_pattern [concrete = constants.%self.param_patt.333]
// CHECK:STDOUT:     %self.patt: %pattern_type.96a = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.00e]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %DefaultOptionalStorage.9f1 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %DefaultOptionalStorage.9f1 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.7: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.8 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc34_19.6 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.8] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.022 = ref_param_pattern [concrete = constants.%self.param_patt.d39]
// CHECK:STDOUT:     %self.patt: %pattern_type.022 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.046]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %Optional.0ac = ref_param call_param0
// CHECK:STDOUT:     %self: ref %Optional.0ac = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.7: %Destroy.WithSelf.Op.type.ef016f.8 = fn_decl @Destroy.WithSelf.Op.loc34_19.6 [concrete = constants.%Destroy.WithSelf.Op.403171.8] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.022 = ref_param_pattern [concrete = constants.%self.param_patt.d39]
// CHECK:STDOUT:     %self.patt: %pattern_type.022 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.046]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %Optional.0ac = ref_param call_param0
// CHECK:STDOUT:     %self: ref %Optional.0ac = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @TestIterate(%R.loc30_17.2: %Iterate_where.type.16e) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Iterate.lookup_impl_witness: <witness> = lookup_impl_witness %R.loc30_17.1, @Iterate [symbolic = %Iterate.lookup_impl_witness (constants.%Iterate.lookup_impl_witness.462)]
// CHECK:STDOUT:   %Iterate.facet.loc34_19.5: %Iterate.type = facet_value %R.as_type.loc30_61.1, (%Iterate.lookup_impl_witness) [symbolic = %Iterate.facet.loc34_19.5 (constants.%Iterate.facet.e5e)]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.loc34_19.5) [symbolic = %Iterate.WithSelf.NewCursor.type (constants.%Iterate.WithSelf.NewCursor.type.62b)]
// CHECK:STDOUT:   %.loc34_19.19: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type, %Iterate.facet.loc34_19.5 [symbolic = %.loc34_19.19 (constants.%.1a4)]
// CHECK:STDOUT:   %impl.elem2.loc34_19.3: @TestIterate.%.loc34_19.19 (%.1a4) = impl_witness_access %Iterate.lookup_impl_witness, element2 [symbolic = %impl.elem2.loc34_19.3 (constants.%impl.elem2.3eb)]
// CHECK:STDOUT:   %specific_impl_fn.loc34_19.4: <specific function> = specific_impl_function %impl.elem2.loc34_19.3, @Iterate.WithSelf.NewCursor(%Iterate.facet.loc34_19.5) [symbolic = %specific_impl_fn.loc34_19.4 (constants.%specific_impl_fn.a09)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %ptr: type = ptr_type %as_type [symbolic = %ptr (constants.%ptr.b18)]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.loc34_19.5) [symbolic = %Iterate.WithSelf.Next.type (constants.%Iterate.WithSelf.Next.type.6c3)]
// CHECK:STDOUT:   %.loc34_19.20: type = fn_type_with_self_type %Iterate.WithSelf.Next.type, %Iterate.facet.loc34_19.5 [symbolic = %.loc34_19.20 (constants.%.218)]
// CHECK:STDOUT:   %impl.elem3.loc34_19.2: @TestIterate.%.loc34_19.20 (%.218) = impl_witness_access %Iterate.lookup_impl_witness, element3 [symbolic = %impl.elem3.loc34_19.2 (constants.%impl.elem3.b50)]
// CHECK:STDOUT:   %specific_impl_fn.loc34_19.5: <specific function> = specific_impl_function %impl.elem3.loc34_19.2, @Iterate.WithSelf.Next(%Iterate.facet.loc34_19.5) [symbolic = %specific_impl_fn.loc34_19.5 (constants.%specific_impl_fn.598)]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type: type = fn_type @Destroy.WithSelf.SelfDestruct.1, @Destroy.WithSelf(%impl.elem1) [symbolic = %Destroy.WithSelf.SelfDestruct.type (constants.%Destroy.WithSelf.SelfDestruct.type.041)]
// CHECK:STDOUT:   %.loc34_19.21: type = fn_type_with_self_type %Destroy.WithSelf.SelfDestruct.type, %impl.elem1 [symbolic = %.loc34_19.21 (constants.%.b28)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %impl.elem1, @Destroy [symbolic = %Destroy.lookup_impl_witness (constants.%Destroy.lookup_impl_witness.caf)]
// CHECK:STDOUT:   %impl.elem2.loc34_19.4: @TestIterate.%.loc34_19.21 (%.b28) = impl_witness_access %Destroy.lookup_impl_witness, element2 [symbolic = %impl.elem2.loc34_19.4 (constants.%impl.elem2.9a0)]
// CHECK:STDOUT:   %specific_impl_fn.loc34_19.6: <specific function> = specific_impl_function %impl.elem2.loc34_19.4, @Destroy.WithSelf.SelfDestruct.1(%impl.elem1) [symbolic = %specific_impl_fn.loc34_19.6 (constants.%specific_impl_fn.ea2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%r.param: @TestIterate.%R.as_type.loc30_61.1 (%R.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %i.patt: %pattern_type.6b6 = value_binding_pattern i [concrete = constants.%i.patt]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %r.ref: @TestIterate.%R.as_type.loc30_61.1 (%R.as_type) = name_ref r, %r
// CHECK:STDOUT:     %impl.elem2.loc34_19.1: @TestIterate.%.loc34_19.19 (%.1a4) = impl_witness_access constants.%Iterate.lookup_impl_witness.462, element2 [symbolic = %impl.elem2.loc34_19.3 (constants.%impl.elem2.3eb)]
// CHECK:STDOUT:     %bound_method.loc34_19.1: <bound method> = bound_method %r.ref, %impl.elem2.loc34_19.1
// CHECK:STDOUT:     %Iterate.facet.loc34_19.1: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.462) [symbolic = %Iterate.facet.loc34_19.5 (constants.%Iterate.facet.e5e)]
// CHECK:STDOUT:     %.loc34_19.1: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc34_19.1 [symbolic = %Iterate.facet.loc34_19.5 (constants.%Iterate.facet.e5e)]
// CHECK:STDOUT:     %Iterate.facet.loc34_19.2: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.462) [symbolic = %Iterate.facet.loc34_19.5 (constants.%Iterate.facet.e5e)]
// CHECK:STDOUT:     %.loc34_19.2: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc34_19.2 [symbolic = %Iterate.facet.loc34_19.5 (constants.%Iterate.facet.e5e)]
// CHECK:STDOUT:     %specific_impl_fn.loc34_19.1: <specific function> = specific_impl_function %impl.elem2.loc34_19.1, @Iterate.WithSelf.NewCursor(constants.%Iterate.facet.e5e) [symbolic = %specific_impl_fn.loc34_19.4 (constants.%specific_impl_fn.a09)]
// CHECK:STDOUT:     %bound_method.loc34_19.2: <bound method> = bound_method %r.ref, %specific_impl_fn.loc34_19.1
// CHECK:STDOUT:     %var: ref @TestIterate.%as_type (%as_type.f79) = var_storage invalid
// CHECK:STDOUT:     %Iterate.WithSelf.NewCursor.call: init @TestIterate.%as_type (%as_type.f79) to %var = call %bound_method.loc34_19.2(%r.ref)
// CHECK:STDOUT:     assign %var, %Iterate.WithSelf.NewCursor.call
// CHECK:STDOUT:     br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.next:
// CHECK:STDOUT:     %addr: @TestIterate.%ptr (%ptr.b18) = addr_of %var
// CHECK:STDOUT:     %impl.elem3.loc34_19.1: @TestIterate.%.loc34_19.20 (%.218) = impl_witness_access constants.%Iterate.lookup_impl_witness.462, element3 [symbolic = %impl.elem3.loc34_19.2 (constants.%impl.elem3.b50)]
// CHECK:STDOUT:     %bound_method.loc34_19.3: <bound method> = bound_method %r.ref, %impl.elem3.loc34_19.1
// CHECK:STDOUT:     %Iterate.facet.loc34_19.3: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.462) [symbolic = %Iterate.facet.loc34_19.5 (constants.%Iterate.facet.e5e)]
// CHECK:STDOUT:     %.loc34_19.3: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc34_19.3 [symbolic = %Iterate.facet.loc34_19.5 (constants.%Iterate.facet.e5e)]
// CHECK:STDOUT:     %Iterate.facet.loc34_19.4: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.462) [symbolic = %Iterate.facet.loc34_19.5 (constants.%Iterate.facet.e5e)]
// CHECK:STDOUT:     %.loc34_19.4: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc34_19.4 [symbolic = %Iterate.facet.loc34_19.5 (constants.%Iterate.facet.e5e)]
// CHECK:STDOUT:     %.loc34_19.5: %Destroy.type = converted constants.%as_type.f79, constants.%impl.elem1.a11 [symbolic = %impl.elem1 (constants.%impl.elem1.a11)]
// CHECK:STDOUT:     %.loc34_19.6: %Destroy.type = converted constants.%as_type.f79, constants.%impl.elem1.a11 [symbolic = %impl.elem1 (constants.%impl.elem1.a11)]
// CHECK:STDOUT:     %specific_impl_fn.loc34_19.2: <specific function> = specific_impl_function %impl.elem3.loc34_19.1, @Iterate.WithSelf.Next(constants.%Iterate.facet.e5e) [symbolic = %specific_impl_fn.loc34_19.5 (constants.%specific_impl_fn.598)]
// CHECK:STDOUT:     %bound_method.loc34_19.4: <bound method> = bound_method %r.ref, %specific_impl_fn.loc34_19.2
// CHECK:STDOUT:     %.loc34_19.7: ref %Optional.0ac = temporary_storage
// CHECK:STDOUT:     %Iterate.WithSelf.Next.call: init %Optional.0ac to %.loc34_19.7 = call %bound_method.loc34_19.4(%r.ref, %addr)
// CHECK:STDOUT:     %.loc34_19.8: ref %Optional.0ac = temporary %.loc34_19.7, %Iterate.WithSelf.Next.call
// CHECK:STDOUT:     %.loc34_19.9: %Optional.HasValue.type.dbd = specific_constant imports.%Core.import_ref.bb6, @Optional(constants.%OptionalStorage.facet.6c5) [concrete = constants.%Optional.HasValue.ed9]
// CHECK:STDOUT:     %HasValue.ref: %Optional.HasValue.type.dbd = name_ref HasValue, %.loc34_19.9 [concrete = constants.%Optional.HasValue.ed9]
// CHECK:STDOUT:     %Optional.HasValue.bound: <bound method> = bound_method %.loc34_19.8, %HasValue.ref
// CHECK:STDOUT:     %Optional.HasValue.specific_fn: <specific function> = specific_function %HasValue.ref, @Optional.HasValue(constants.%OptionalStorage.facet.6c5) [concrete = constants.%Optional.HasValue.specific_fn.15f]
// CHECK:STDOUT:     %bound_method.loc34_19.5: <bound method> = bound_method %.loc34_19.8, %Optional.HasValue.specific_fn
// CHECK:STDOUT:     %.loc34_19.10: %Optional.0ac = acquire_value %.loc34_19.8
// CHECK:STDOUT:     %Optional.HasValue.call: init bool = call %bound_method.loc34_19.5(%.loc34_19.10)
// CHECK:STDOUT:     %.loc34_19.11: bool = value_of_initializer %Optional.HasValue.call
// CHECK:STDOUT:     %.loc34_19.12: bool = converted %Optional.HasValue.call, %.loc34_19.11
// CHECK:STDOUT:     if %.loc34_19.12 br !for.body else br !for.done
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.body:
// CHECK:STDOUT:     %.loc34_19.13: %Optional.Get.type.2b4 = specific_constant imports.%Core.import_ref.dab, @Optional(constants.%OptionalStorage.facet.6c5) [concrete = constants.%Optional.Get.529]
// CHECK:STDOUT:     %Get.ref: %Optional.Get.type.2b4 = name_ref Get, %.loc34_19.13 [concrete = constants.%Optional.Get.529]
// CHECK:STDOUT:     %Optional.Get.bound: <bound method> = bound_method %.loc34_19.8, %Get.ref
// CHECK:STDOUT:     %Optional.Get.specific_fn: <specific function> = specific_function %Get.ref, @Optional.Get(constants.%OptionalStorage.facet.6c5) [concrete = constants.%Optional.Get.specific_fn.0dd]
// CHECK:STDOUT:     %bound_method.loc34_19.6: <bound method> = bound_method %.loc34_19.8, %Optional.Get.specific_fn
// CHECK:STDOUT:     %.loc34_19.14: %Optional.0ac = acquire_value %.loc34_19.8
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
// CHECK:STDOUT:     %Destroy.WithSelf.SelfDestruct.bound.loc34_19.1: <bound method> = bound_method %.loc34_19.8, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.8
// CHECK:STDOUT:     %Destroy.WithSelf.SelfDestruct.call.loc34_19.1: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc34_19.1(%.loc34_19.8)
// CHECK:STDOUT:     br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.done:
// CHECK:STDOUT:     %Destroy.WithSelf.SelfDestruct.bound.loc34_19.2: <bound method> = bound_method %.loc34_19.8, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.8
// CHECK:STDOUT:     %Destroy.WithSelf.SelfDestruct.call.loc34_19.2: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc34_19.2(%.loc34_19.8)
// CHECK:STDOUT:     %impl.elem2.loc34_19.2: @TestIterate.%.loc34_19.21 (%.b28) = impl_witness_access constants.%Destroy.lookup_impl_witness.caf, element2 [symbolic = %impl.elem2.loc34_19.4 (constants.%impl.elem2.9a0)]
// CHECK:STDOUT:     %bound_method.loc34_19.7: <bound method> = bound_method %var, %impl.elem2.loc34_19.2
// CHECK:STDOUT:     %.loc34_19.17: %Destroy.type = converted constants.%as_type.f79, constants.%impl.elem1.a11 [symbolic = %impl.elem1 (constants.%impl.elem1.a11)]
// CHECK:STDOUT:     %.loc34_19.18: %Destroy.type = converted constants.%as_type.f79, constants.%impl.elem1.a11 [symbolic = %impl.elem1 (constants.%impl.elem1.a11)]
// CHECK:STDOUT:     %specific_impl_fn.loc34_19.3: <specific function> = specific_impl_function %impl.elem2.loc34_19.2, @Destroy.WithSelf.SelfDestruct.1(constants.%impl.elem1.a11) [symbolic = %specific_impl_fn.loc34_19.6 (constants.%specific_impl_fn.ea2)]
// CHECK:STDOUT:     %bound_method.loc34_19.8: <bound method> = bound_method %var, %specific_impl_fn.loc34_19.3
// CHECK:STDOUT:     %Destroy.WithSelf.SelfDestruct.call.loc34_19.3: init %empty_tuple.type = call %bound_method.loc34_19.8(%var)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc34_19.1(%self.param: ref %.1d6a7) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc34_19.1(%self.param: ref %.1d6a7) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc34_19.1(%self.param: ref %.1d6a7) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.2(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.2(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc34_19.2(%self.param: ref %MaybeUnformed.515) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc34_19.2(%self.param: ref %MaybeUnformed.515) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc34_19.2(%self.param: ref %MaybeUnformed.515) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.3(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.3(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc34_19.3(%self.param: ref bool) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc34_19.3(%self.param: ref bool) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc34_19.3(%self.param: ref bool) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.4(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.4(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc34_19.4(%self.param: ref %struct_type.value.has_value.f6a) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc34_19.1: ref %MaybeUnformed.515 = struct_access %self.param, element0
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc34_19.1: <bound method> = bound_method %.loc34_19.1, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.4
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc34_19.1: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc34_19.1(%.loc34_19.1)
// CHECK:STDOUT:   %.loc34_19.2: ref bool = struct_access %self.param, element1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc34_19.2: <bound method> = bound_method %.loc34_19.2, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.5
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc34_19.2: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc34_19.2(%.loc34_19.2)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc34_19.4(%self.param: ref %struct_type.value.has_value.f6a) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc34_19.4(%self.param: ref %struct_type.value.has_value.f6a) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.5(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.5(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc34_19.5(%self.param: ref %DefaultOptionalStorage.9f1) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc34_19.5(%self.param: ref %DefaultOptionalStorage.9f1) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc34_19.5(%self.param: ref %DefaultOptionalStorage.9f1) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.6(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.6(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc34_19.6(%self.param: ref %Optional.0ac) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc34_19.6(%self.param: ref %Optional.0ac) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc34_19.6(%self.param: ref %Optional.0ac) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.7(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.7(%self.param)
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
// CHECK:STDOUT:   %impl.elem2.loc49: %.cad = impl_witness_access constants.%Iterate.impl_witness.2bb, element2 [concrete = constants.%T.as_type.as.Iterate.impl.NewCursor.e00]
// CHECK:STDOUT:   %bound_method.loc49_19.1: <bound method> = bound_method %m.ref, %impl.elem2.loc49
// CHECK:STDOUT:   %facet_value.loc49_19.1: %CppRange_where.type.d682d6.1 = facet_value constants.%MutableRange, (constants.%custom_witness.f8f19d.1, constants.%Copy.impl_witness.b51, constants.%custom_witness.ed3, constants.%custom_witness.8e1, constants.%custom_witness.ef8, constants.%custom_witness.444, constants.%custom_witness.981) [concrete = constants.%facet_value.474]
// CHECK:STDOUT:   %.loc49_19.1: %CppRange_where.type.d682d6.1 = converted constants.%MutableRange, %facet_value.loc49_19.1 [concrete = constants.%facet_value.474]
// CHECK:STDOUT:   %facet_value.loc49_19.2: %CppRange_where.type.d682d6.1 = facet_value constants.%MutableRange, (constants.%custom_witness.f8f19d.1, constants.%Copy.impl_witness.b51, constants.%custom_witness.ed3, constants.%custom_witness.8e1, constants.%custom_witness.ef8, constants.%custom_witness.444, constants.%custom_witness.981) [concrete = constants.%facet_value.474]
// CHECK:STDOUT:   %.loc49_19.2: %CppRange_where.type.d682d6.1 = converted constants.%MutableRange, %facet_value.loc49_19.2 [concrete = constants.%facet_value.474]
// CHECK:STDOUT:   %specific_fn.loc49_19.1: <specific function> = specific_function %impl.elem2.loc49, @T.as_type.as.Iterate.impl.NewCursor(constants.%facet_value.474) [concrete = constants.%T.as_type.as.Iterate.impl.NewCursor.specific_fn.ac2]
// CHECK:STDOUT:   %bound_method.loc49_19.2: <bound method> = bound_method %m.ref, %specific_fn.loc49_19.1
// CHECK:STDOUT:   %var.loc49: ref %tuple.type.508 = var_storage invalid
// CHECK:STDOUT:   %.loc49_18.1: %MutableRange = acquire_value %m.ref
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.call.loc49: init %tuple.type.508 to %var.loc49 = call %bound_method.loc49_19.2(%.loc49_18.1)
// CHECK:STDOUT:   assign %var.loc49, %T.as_type.as.Iterate.impl.NewCursor.call.loc49
// CHECK:STDOUT:   br !for.next.loc49
// CHECK:STDOUT:
// CHECK:STDOUT: !for.next.loc49:
// CHECK:STDOUT:   %addr.loc49: %ptr.45c = addr_of %var.loc49
// CHECK:STDOUT:   %impl.elem3.loc49: %.640 = impl_witness_access constants.%Iterate.impl_witness.2bb, element3 [concrete = constants.%T.as_type.as.Iterate.impl.Next.311]
// CHECK:STDOUT:   %bound_method.loc49_19.3: <bound method> = bound_method %m.ref, %impl.elem3.loc49
// CHECK:STDOUT:   %facet_value.loc49_19.3: %CppRange_where.type.d682d6.1 = facet_value constants.%MutableRange, (constants.%custom_witness.f8f19d.1, constants.%Copy.impl_witness.b51, constants.%custom_witness.ed3, constants.%custom_witness.8e1, constants.%custom_witness.ef8, constants.%custom_witness.444, constants.%custom_witness.981) [concrete = constants.%facet_value.474]
// CHECK:STDOUT:   %.loc49_19.3: %CppRange_where.type.d682d6.1 = converted constants.%MutableRange, %facet_value.loc49_19.3 [concrete = constants.%facet_value.474]
// CHECK:STDOUT:   %facet_value.loc49_19.4: %CppRange_where.type.d682d6.1 = facet_value constants.%MutableRange, (constants.%custom_witness.f8f19d.1, constants.%Copy.impl_witness.b51, constants.%custom_witness.ed3, constants.%custom_witness.8e1, constants.%custom_witness.ef8, constants.%custom_witness.444, constants.%custom_witness.981) [concrete = constants.%facet_value.474]
// CHECK:STDOUT:   %.loc49_19.4: %CppRange_where.type.d682d6.1 = converted constants.%MutableRange, %facet_value.loc49_19.4 [concrete = constants.%facet_value.474]
// CHECK:STDOUT:   %specific_fn.loc49_19.2: <specific function> = specific_function %impl.elem3.loc49, @T.as_type.as.Iterate.impl.Next(constants.%facet_value.474) [concrete = constants.%T.as_type.as.Iterate.impl.Next.specific_fn.b01]
// CHECK:STDOUT:   %bound_method.loc49_19.4: <bound method> = bound_method %m.ref, %specific_fn.loc49_19.2
// CHECK:STDOUT:   %.loc49_19.5: ref %Optional.0ac = temporary_storage
// CHECK:STDOUT:   %.loc49_18.2: %MutableRange = acquire_value %m.ref
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.call.loc49: init %Optional.0ac to %.loc49_19.5 = call %bound_method.loc49_19.4(%.loc49_18.2, %addr.loc49)
// CHECK:STDOUT:   %.loc49_19.6: ref %Optional.0ac = temporary %.loc49_19.5, %T.as_type.as.Iterate.impl.Next.call.loc49
// CHECK:STDOUT:   %.loc49_19.7: %Optional.HasValue.type.dbd = specific_constant imports.%Core.import_ref.bb6, @Optional(constants.%OptionalStorage.facet.6c5) [concrete = constants.%Optional.HasValue.ed9]
// CHECK:STDOUT:   %HasValue.ref.loc49: %Optional.HasValue.type.dbd = name_ref HasValue, %.loc49_19.7 [concrete = constants.%Optional.HasValue.ed9]
// CHECK:STDOUT:   %Optional.HasValue.bound.loc49: <bound method> = bound_method %.loc49_19.6, %HasValue.ref.loc49
// CHECK:STDOUT:   %Optional.HasValue.specific_fn.loc49: <specific function> = specific_function %HasValue.ref.loc49, @Optional.HasValue(constants.%OptionalStorage.facet.6c5) [concrete = constants.%Optional.HasValue.specific_fn.15f]
// CHECK:STDOUT:   %bound_method.loc49_19.5: <bound method> = bound_method %.loc49_19.6, %Optional.HasValue.specific_fn.loc49
// CHECK:STDOUT:   %.loc49_19.8: %Optional.0ac = acquire_value %.loc49_19.6
// CHECK:STDOUT:   %Optional.HasValue.call.loc49: init bool = call %bound_method.loc49_19.5(%.loc49_19.8)
// CHECK:STDOUT:   %.loc49_19.9: bool = value_of_initializer %Optional.HasValue.call.loc49
// CHECK:STDOUT:   %.loc49_19.10: bool = converted %Optional.HasValue.call.loc49, %.loc49_19.9
// CHECK:STDOUT:   if %.loc49_19.10 br !for.body.loc49 else br !for.done.loc49
// CHECK:STDOUT:
// CHECK:STDOUT: !for.body.loc49:
// CHECK:STDOUT:   %.loc49_19.11: %Optional.Get.type.2b4 = specific_constant imports.%Core.import_ref.dab, @Optional(constants.%OptionalStorage.facet.6c5) [concrete = constants.%Optional.Get.529]
// CHECK:STDOUT:   %Get.ref.loc49: %Optional.Get.type.2b4 = name_ref Get, %.loc49_19.11 [concrete = constants.%Optional.Get.529]
// CHECK:STDOUT:   %Optional.Get.bound.loc49: <bound method> = bound_method %.loc49_19.6, %Get.ref.loc49
// CHECK:STDOUT:   %Optional.Get.specific_fn.loc49: <specific function> = specific_function %Get.ref.loc49, @Optional.Get(constants.%OptionalStorage.facet.6c5) [concrete = constants.%Optional.Get.specific_fn.0dd]
// CHECK:STDOUT:   %bound_method.loc49_19.6: <bound method> = bound_method %.loc49_19.6, %Optional.Get.specific_fn.loc49
// CHECK:STDOUT:   %.loc49_19.12: %Optional.0ac = acquire_value %.loc49_19.6
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
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc49_19.1: <bound method> = bound_method %.loc49_19.6, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.8
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc49_19.1: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc49_19.1(%.loc49_19.6)
// CHECK:STDOUT:   br !for.next.loc49
// CHECK:STDOUT:
// CHECK:STDOUT: !for.done.loc49:
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc49_19.2: <bound method> = bound_method %.loc49_19.6, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.8
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc49_19.2: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc49_19.2(%.loc49_19.6)
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc49_19.3: <bound method> = bound_method %var.loc49, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.10
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc49_19.3: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc49_19.3(%var.loc49)
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %i.patt.loc55: %pattern_type.6b6 = value_binding_pattern i [concrete = constants.%i.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c.ref: %ConstRange = name_ref c, %c
// CHECK:STDOUT:   %impl.elem2.loc55: %.312 = impl_witness_access constants.%Iterate.impl_witness.1a6, element2 [concrete = constants.%T.as_type.as.Iterate.impl.NewCursor.19e]
// CHECK:STDOUT:   %bound_method.loc55_19.1: <bound method> = bound_method %c.ref, %impl.elem2.loc55
// CHECK:STDOUT:   %facet_value.loc55_19.1: %CppRange_where.type.d682d6.1 = facet_value constants.%ConstRange, (constants.%custom_witness.f8f19d.1, constants.%Copy.impl_witness.b51, constants.%custom_witness.23e, constants.%custom_witness.0f3, constants.%custom_witness.adb, constants.%custom_witness.746, constants.%custom_witness.941) [concrete = constants.%facet_value.3e1]
// CHECK:STDOUT:   %.loc55_19.1: %CppRange_where.type.d682d6.1 = converted constants.%ConstRange, %facet_value.loc55_19.1 [concrete = constants.%facet_value.3e1]
// CHECK:STDOUT:   %facet_value.loc55_19.2: %CppRange_where.type.d682d6.1 = facet_value constants.%ConstRange, (constants.%custom_witness.f8f19d.1, constants.%Copy.impl_witness.b51, constants.%custom_witness.23e, constants.%custom_witness.0f3, constants.%custom_witness.adb, constants.%custom_witness.746, constants.%custom_witness.941) [concrete = constants.%facet_value.3e1]
// CHECK:STDOUT:   %.loc55_19.2: %CppRange_where.type.d682d6.1 = converted constants.%ConstRange, %facet_value.loc55_19.2 [concrete = constants.%facet_value.3e1]
// CHECK:STDOUT:   %specific_fn.loc55_19.1: <specific function> = specific_function %impl.elem2.loc55, @T.as_type.as.Iterate.impl.NewCursor(constants.%facet_value.3e1) [concrete = constants.%T.as_type.as.Iterate.impl.NewCursor.specific_fn.91a]
// CHECK:STDOUT:   %bound_method.loc55_19.2: <bound method> = bound_method %c.ref, %specific_fn.loc55_19.1
// CHECK:STDOUT:   %var.loc55: ref %tuple.type.69f = var_storage invalid
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.call.loc55: init %tuple.type.69f to %var.loc55 = call %bound_method.loc55_19.2(%c.ref)
// CHECK:STDOUT:   assign %var.loc55, %T.as_type.as.Iterate.impl.NewCursor.call.loc55
// CHECK:STDOUT:   br !for.next.loc55
// CHECK:STDOUT:
// CHECK:STDOUT: !for.next.loc55:
// CHECK:STDOUT:   %addr.loc55: %ptr.c5c = addr_of %var.loc55
// CHECK:STDOUT:   %impl.elem3.loc55: %.5b5 = impl_witness_access constants.%Iterate.impl_witness.1a6, element3 [concrete = constants.%T.as_type.as.Iterate.impl.Next.372]
// CHECK:STDOUT:   %bound_method.loc55_19.3: <bound method> = bound_method %c.ref, %impl.elem3.loc55
// CHECK:STDOUT:   %facet_value.loc55_19.3: %CppRange_where.type.d682d6.1 = facet_value constants.%ConstRange, (constants.%custom_witness.f8f19d.1, constants.%Copy.impl_witness.b51, constants.%custom_witness.23e, constants.%custom_witness.0f3, constants.%custom_witness.adb, constants.%custom_witness.746, constants.%custom_witness.941) [concrete = constants.%facet_value.3e1]
// CHECK:STDOUT:   %.loc55_19.3: %CppRange_where.type.d682d6.1 = converted constants.%ConstRange, %facet_value.loc55_19.3 [concrete = constants.%facet_value.3e1]
// CHECK:STDOUT:   %facet_value.loc55_19.4: %CppRange_where.type.d682d6.1 = facet_value constants.%ConstRange, (constants.%custom_witness.f8f19d.1, constants.%Copy.impl_witness.b51, constants.%custom_witness.23e, constants.%custom_witness.0f3, constants.%custom_witness.adb, constants.%custom_witness.746, constants.%custom_witness.941) [concrete = constants.%facet_value.3e1]
// CHECK:STDOUT:   %.loc55_19.4: %CppRange_where.type.d682d6.1 = converted constants.%ConstRange, %facet_value.loc55_19.4 [concrete = constants.%facet_value.3e1]
// CHECK:STDOUT:   %specific_fn.loc55_19.2: <specific function> = specific_function %impl.elem3.loc55, @T.as_type.as.Iterate.impl.Next(constants.%facet_value.3e1) [concrete = constants.%T.as_type.as.Iterate.impl.Next.specific_fn.5ef]
// CHECK:STDOUT:   %bound_method.loc55_19.4: <bound method> = bound_method %c.ref, %specific_fn.loc55_19.2
// CHECK:STDOUT:   %.loc55_19.5: ref %Optional.0ac = temporary_storage
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.call.loc55: init %Optional.0ac to %.loc55_19.5 = call %bound_method.loc55_19.4(%c.ref, %addr.loc55)
// CHECK:STDOUT:   %.loc55_19.6: ref %Optional.0ac = temporary %.loc55_19.5, %T.as_type.as.Iterate.impl.Next.call.loc55
// CHECK:STDOUT:   %.loc55_19.7: %Optional.HasValue.type.dbd = specific_constant imports.%Core.import_ref.bb6, @Optional(constants.%OptionalStorage.facet.6c5) [concrete = constants.%Optional.HasValue.ed9]
// CHECK:STDOUT:   %HasValue.ref.loc55: %Optional.HasValue.type.dbd = name_ref HasValue, %.loc55_19.7 [concrete = constants.%Optional.HasValue.ed9]
// CHECK:STDOUT:   %Optional.HasValue.bound.loc55: <bound method> = bound_method %.loc55_19.6, %HasValue.ref.loc55
// CHECK:STDOUT:   %Optional.HasValue.specific_fn.loc55: <specific function> = specific_function %HasValue.ref.loc55, @Optional.HasValue(constants.%OptionalStorage.facet.6c5) [concrete = constants.%Optional.HasValue.specific_fn.15f]
// CHECK:STDOUT:   %bound_method.loc55_19.5: <bound method> = bound_method %.loc55_19.6, %Optional.HasValue.specific_fn.loc55
// CHECK:STDOUT:   %.loc55_19.8: %Optional.0ac = acquire_value %.loc55_19.6
// CHECK:STDOUT:   %Optional.HasValue.call.loc55: init bool = call %bound_method.loc55_19.5(%.loc55_19.8)
// CHECK:STDOUT:   %.loc55_19.9: bool = value_of_initializer %Optional.HasValue.call.loc55
// CHECK:STDOUT:   %.loc55_19.10: bool = converted %Optional.HasValue.call.loc55, %.loc55_19.9
// CHECK:STDOUT:   if %.loc55_19.10 br !for.body.loc55 else br !for.done.loc55
// CHECK:STDOUT:
// CHECK:STDOUT: !for.body.loc55:
// CHECK:STDOUT:   %.loc55_19.11: %Optional.Get.type.2b4 = specific_constant imports.%Core.import_ref.dab, @Optional(constants.%OptionalStorage.facet.6c5) [concrete = constants.%Optional.Get.529]
// CHECK:STDOUT:   %Get.ref.loc55: %Optional.Get.type.2b4 = name_ref Get, %.loc55_19.11 [concrete = constants.%Optional.Get.529]
// CHECK:STDOUT:   %Optional.Get.bound.loc55: <bound method> = bound_method %.loc55_19.6, %Get.ref.loc55
// CHECK:STDOUT:   %Optional.Get.specific_fn.loc55: <specific function> = specific_function %Get.ref.loc55, @Optional.Get(constants.%OptionalStorage.facet.6c5) [concrete = constants.%Optional.Get.specific_fn.0dd]
// CHECK:STDOUT:   %bound_method.loc55_19.6: <bound method> = bound_method %.loc55_19.6, %Optional.Get.specific_fn.loc55
// CHECK:STDOUT:   %.loc55_19.12: %Optional.0ac = acquire_value %.loc55_19.6
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
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc55_19.1: <bound method> = bound_method %.loc55_19.6, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.8
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc55_19.1: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc55_19.1(%.loc55_19.6)
// CHECK:STDOUT:   br !for.next.loc55
// CHECK:STDOUT:
// CHECK:STDOUT: !for.done.loc55:
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc55_19.2: <bound method> = bound_method %.loc55_19.6, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.8
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc55_19.2: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc55_19.2(%.loc55_19.6)
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc55_19.3: <bound method> = bound_method %var.loc55, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.12
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc55_19.3: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc55_19.3(%var.loc55)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestIterate(constants.%R) {
// CHECK:STDOUT:   %R.patt.loc30_17.2 => constants.%R.patt
// CHECK:STDOUT:   %R.loc30_17.1 => constants.%R
// CHECK:STDOUT:   %R.as_type.loc30_61.1 => constants.%R.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.6f9
// CHECK:STDOUT:   %r.param_patt.loc30_59.2 => constants.%r.param_patt.170
// CHECK:STDOUT:   %r.patt.loc30_59.2 => constants.%r.patt.893
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestIterate(constants.%facet_value.26c) {
// CHECK:STDOUT:   %R.patt.loc30_17.2 => constants.%R.patt
// CHECK:STDOUT:   %R.loc30_17.1 => constants.%facet_value.26c
// CHECK:STDOUT:   %R.as_type.loc30_61.1 => constants.%MutableRange
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.e635
// CHECK:STDOUT:   %r.param_patt.loc30_59.2 => constants.%r.param_patt.c79
// CHECK:STDOUT:   %r.patt.loc30_59.2 => constants.%r.patt.94f
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc30 => constants.%complete_type.357
// CHECK:STDOUT:   %Iterate.lookup_impl_witness => constants.%Iterate.impl_witness.2bb
// CHECK:STDOUT:   %Iterate.facet.loc34_19.5 => constants.%Iterate.facet.854
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type => constants.%Iterate.WithSelf.NewCursor.type.4e8
// CHECK:STDOUT:   %.loc34_19.19 => constants.%.cad
// CHECK:STDOUT:   %impl.elem2.loc34_19.3 => constants.%T.as_type.as.Iterate.impl.NewCursor.e00
// CHECK:STDOUT:   %specific_impl_fn.loc34_19.4 => constants.%T.as_type.as.Iterate.impl.NewCursor.specific_fn.ac2
// CHECK:STDOUT:   %impl.elem1 => constants.%Destroy.facet.0fe
// CHECK:STDOUT:   %as_type => constants.%tuple.type.508
// CHECK:STDOUT:   %require_complete.1 => constants.%complete_type.c28
// CHECK:STDOUT:   %ptr => constants.%ptr.45c
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type => constants.%Iterate.WithSelf.Next.type.713
// CHECK:STDOUT:   %.loc34_19.20 => constants.%.640
// CHECK:STDOUT:   %impl.elem3.loc34_19.2 => constants.%T.as_type.as.Iterate.impl.Next.311
// CHECK:STDOUT:   %specific_impl_fn.loc34_19.5 => constants.%T.as_type.as.Iterate.impl.Next.specific_fn.b01
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type => constants.%Destroy.WithSelf.SelfDestruct.type.471
// CHECK:STDOUT:   %.loc34_19.21 => constants.%.463
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.f8f19d.9
// CHECK:STDOUT:   %impl.elem2.loc34_19.4 => constants.%Destroy.WithSelf.SelfDestruct.db3fdb.10
// CHECK:STDOUT:   %specific_impl_fn.loc34_19.6 => constants.%Destroy.WithSelf.SelfDestruct.db3fdb.10
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestIterate(constants.%facet_value.75e) {
// CHECK:STDOUT:   %R.patt.loc30_17.2 => constants.%R.patt
// CHECK:STDOUT:   %R.loc30_17.1 => constants.%facet_value.75e
// CHECK:STDOUT:   %R.as_type.loc30_61.1 => constants.%ConstRange
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.0ba
// CHECK:STDOUT:   %r.param_patt.loc30_59.2 => constants.%r.param_patt.83e
// CHECK:STDOUT:   %r.patt.loc30_59.2 => constants.%r.patt.c21
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc30 => constants.%complete_type.357
// CHECK:STDOUT:   %Iterate.lookup_impl_witness => constants.%Iterate.impl_witness.1a6
// CHECK:STDOUT:   %Iterate.facet.loc34_19.5 => constants.%Iterate.facet.7d1
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type => constants.%Iterate.WithSelf.NewCursor.type.488
// CHECK:STDOUT:   %.loc34_19.19 => constants.%.312
// CHECK:STDOUT:   %impl.elem2.loc34_19.3 => constants.%T.as_type.as.Iterate.impl.NewCursor.19e
// CHECK:STDOUT:   %specific_impl_fn.loc34_19.4 => constants.%T.as_type.as.Iterate.impl.NewCursor.specific_fn.91a
// CHECK:STDOUT:   %impl.elem1 => constants.%Destroy.facet.c6d
// CHECK:STDOUT:   %as_type => constants.%tuple.type.69f
// CHECK:STDOUT:   %require_complete.1 => constants.%complete_type.8d5
// CHECK:STDOUT:   %ptr => constants.%ptr.c5c
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type => constants.%Iterate.WithSelf.Next.type.aee
// CHECK:STDOUT:   %.loc34_19.20 => constants.%.5b5
// CHECK:STDOUT:   %impl.elem3.loc34_19.2 => constants.%T.as_type.as.Iterate.impl.Next.372
// CHECK:STDOUT:   %specific_impl_fn.loc34_19.5 => constants.%T.as_type.as.Iterate.impl.Next.specific_fn.5ef
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type => constants.%Destroy.WithSelf.SelfDestruct.type.c53
// CHECK:STDOUT:   %.loc34_19.21 => constants.%.331
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.f8f19d.10
// CHECK:STDOUT:   %impl.elem2.loc34_19.4 => constants.%Destroy.WithSelf.SelfDestruct.db3fdb.12
// CHECK:STDOUT:   %specific_impl_fn.loc34_19.6 => constants.%Destroy.WithSelf.SelfDestruct.db3fdb.12
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- methods_return_different_types.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %Iterate.type: type = facet_type <@Iterate> [concrete]
// CHECK:STDOUT:   %OptionalStorage.type: type = facet_type <@OptionalStorage> [concrete]
// CHECK:STDOUT:   %pattern_type.831: type = pattern_type bool [concrete]
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
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.2: type = fn_type @Destroy.WithSelf.Op.loc40_56 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.2: %Destroy.WithSelf.Op.type.ef016f.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.2: type = fn_type @Destroy.WithSelf.SelfDestruct.loc40_56 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.2: %Destroy.WithSelf.SelfDestruct.type.fbceb5.2 = struct_value () [concrete]
// CHECK:STDOUT:   %DiffValueType.SubobjectDestroy.type: type = fn_type @DiffValueType.SubobjectDestroy [concrete]
// CHECK:STDOUT:   %DiffValueType.SubobjectDestroy: %DiffValueType.SubobjectDestroy.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.1bfa2a.1: <witness> = custom_witness (%Destroy.WithSelf.Op.403171.2, %DiffValueType.SubobjectDestroy, %Destroy.WithSelf.SelfDestruct.db3fdb.2), @Destroy [concrete]
// CHECK:STDOUT:   %const.879: type = const_type %DiffValueType [concrete]
// CHECK:STDOUT:   %ptr.816: type = ptr_type %const.879 [concrete]
// CHECK:STDOUT:   %ptr.cd0: type = ptr_type %DiffValueType [concrete]
// CHECK:STDOUT:   %DiffValueType.Op.type: type = fn_type @DiffValueType.Op [concrete]
// CHECK:STDOUT:   %DiffValueType.Op: %DiffValueType.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.d0da43.1: <witness> = custom_witness (%DiffValueType.Op), @Copy [concrete]
// CHECK:STDOUT:   %facet_value.cca: %facet_type.f48 = facet_value %DiffValueType, (%custom_witness.1bfa2a.1, %custom_witness.d0da43.1) [concrete]
// CHECK:STDOUT:   %Iterate.lookup_impl_witness.1d8: <witness> = lookup_impl_witness %.Self.500, @Iterate [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.294: %facet_type.f48 = impl_witness_access %Iterate.lookup_impl_witness.1d8, element0 [symbolic_self]
// CHECK:STDOUT:   %Iterate_where.type.b96: type = facet_type <@Iterate where %impl.elem0.294 = %facet_value.cca> [concrete]
// CHECK:STDOUT:   %pattern_type.975: type = pattern_type %Iterate_where.type.b96 [concrete]
// CHECK:STDOUT:   %R.patt: %pattern_type.975 = symbolic_binding_pattern R, 0 [symbolic]
// CHECK:STDOUT:   %R: %Iterate_where.type.b96 = symbolic_binding R, 0 [symbolic]
// CHECK:STDOUT:   %R.as_type: type = facet_access_type %R [symbolic]
// CHECK:STDOUT:   %pattern_type.dc2: type = pattern_type %R.as_type [symbolic]
// CHECK:STDOUT:   %r.patt.917: %pattern_type.dc2 = ref_binding_pattern r [symbolic]
// CHECK:STDOUT:   %r.param_patt.9a1: %pattern_type.dc2 = var_param_pattern %r.patt.917 [symbolic]
// CHECK:STDOUT:   %i.patt: %pattern_type.149 = value_binding_pattern i [concrete]
// CHECK:STDOUT:   %Iterate.lookup_impl_witness.d00: <witness> = lookup_impl_witness %R, @Iterate [symbolic]
// CHECK:STDOUT:   %Iterate.facet.f03: %Iterate.type = facet_value %R.as_type, (%Iterate.lookup_impl_witness.d00) [symbolic]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.de8: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.f03) [symbolic]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.e6f: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.f03) [symbolic]
// CHECK:STDOUT:   %.c20: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.de8, %Iterate.facet.f03 [symbolic]
// CHECK:STDOUT:   %impl.elem2.7d0: %.c20 = impl_witness_access %Iterate.lookup_impl_witness.d00, element2 [symbolic]
// CHECK:STDOUT:   %impl.elem1.fbc: %Destroy.type = impl_witness_access %Iterate.lookup_impl_witness.d00, element1 [symbolic]
// CHECK:STDOUT:   %as_type.389: type = facet_access_type %impl.elem1.fbc [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.1c3: <specific function> = specific_impl_function %impl.elem2.7d0, @Iterate.WithSelf.NewCursor(%Iterate.facet.f03) [symbolic]
// CHECK:STDOUT:   %ptr.02d: type = ptr_type %as_type.389 [symbolic]
// CHECK:STDOUT:   %.89f: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.e6f, %Iterate.facet.f03 [symbolic]
// CHECK:STDOUT:   %impl.elem3.73e: %.89f = impl_witness_access %Iterate.lookup_impl_witness.d00, element3 [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.9a0: type = class_type @DefaultOptionalStorage, @DefaultOptionalStorage(%facet_value.cca) [concrete]
// CHECK:STDOUT:   %OptionalStorage.impl_witness.e8d: <witness> = impl_witness imports.%OptionalStorage.impl_witness_table.460, @T.as_type.as.OptionalStorage.impl(%facet_value.cca) [concrete]
// CHECK:STDOUT:   %OptionalStorage.facet.beb: %OptionalStorage.type = facet_value %DiffValueType, (%OptionalStorage.impl_witness.e8d) [concrete]
// CHECK:STDOUT:   %Optional.d06: type = class_type @Optional, @Optional(%OptionalStorage.facet.beb) [concrete]
// CHECK:STDOUT:   %pattern_type.196: type = pattern_type %Optional.d06 [concrete]
// CHECK:STDOUT:   %specific_impl_fn.73e: <specific function> = specific_impl_function %impl.elem3.73e, @Iterate.WithSelf.Next(%Iterate.facet.f03) [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.type.821: type = fn_type @Optional.HasValue, @Optional(%OptionalStorage.facet.beb) [concrete]
// CHECK:STDOUT:   %Optional.HasValue.960: %Optional.HasValue.type.821 = struct_value () [concrete]
// CHECK:STDOUT:   %Optional.Get.type.f9c: type = fn_type @Optional.Get, @Optional(%OptionalStorage.facet.beb) [concrete]
// CHECK:STDOUT:   %Optional.Get.294: %Optional.Get.type.f9c = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.facet.be4: %Destroy.type = facet_value %DiffValueType, (%custom_witness.1bfa2a.1) [concrete]
// CHECK:STDOUT:   %MaybeUnformed.632: type = class_type @MaybeUnformed, @MaybeUnformed(%Destroy.facet.be4) [concrete]
// CHECK:STDOUT:   %.a18: type = maybe_unformed_type %DiffValueType [concrete]
// CHECK:STDOUT:   %struct_type.value.has_value.851: type = struct_type {.value: %MaybeUnformed.632, .has_value: bool} [concrete]
// CHECK:STDOUT:   %Optional.HasValue.specific_fn.a20: <specific function> = specific_function %Optional.HasValue.960, @Optional.HasValue(%OptionalStorage.facet.beb) [concrete]
// CHECK:STDOUT:   %Optional.Get.specific_fn.084: <specific function> = specific_function %Optional.Get.294, @Optional.Get(%OptionalStorage.facet.beb) [concrete]
// CHECK:STDOUT:   %operator_PlusEqual__carbon_thunk.type: type = fn_type @operator_PlusEqual__carbon_thunk [concrete]
// CHECK:STDOUT:   %operator_PlusEqual__carbon_thunk: %operator_PlusEqual__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.98c: type = pattern_type %.a18 [concrete]
// CHECK:STDOUT:   %self.param_patt.47c: %pattern_type.98c = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.90b: %pattern_type.98c = wrapper_binding_pattern self, %self.param_patt.47c [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc44_33.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.2: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.3: type = fn_type @Destroy.WithSelf.Op.loc44_33.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.3: %Destroy.WithSelf.Op.type.ef016f.3 = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.ea0: type = pattern_type %MaybeUnformed.632 [concrete]
// CHECK:STDOUT:   %self.param_patt.4d6: %pattern_type.ea0 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.5e0: %pattern_type.ea0 = wrapper_binding_pattern self, %self.param_patt.4d6 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.3: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc44_33.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.3: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.3 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.4: type = fn_type @Destroy.WithSelf.Op.loc44_33.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.4: %Destroy.WithSelf.Op.type.ef016f.4 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.4: type = fn_type @Destroy.WithSelf.SelfDestruct.loc44_33.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.4: %Destroy.WithSelf.SelfDestruct.type.fbceb5.4 = struct_value () [concrete]
// CHECK:STDOUT:   %self.param_patt.40d: %pattern_type.831 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.560: %pattern_type.831 = wrapper_binding_pattern self, %self.param_patt.40d [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.4: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc44_33.3 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.4: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.4 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.5: type = fn_type @Destroy.WithSelf.Op.loc44_33.3 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.5: %Destroy.WithSelf.Op.type.ef016f.5 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.5: type = fn_type @Destroy.WithSelf.SelfDestruct.loc44_33.3 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.5: %Destroy.WithSelf.SelfDestruct.type.fbceb5.5 = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.f90: type = pattern_type %struct_type.value.has_value.851 [concrete]
// CHECK:STDOUT:   %self.param_patt.067: %pattern_type.f90 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.fc6: %pattern_type.f90 = wrapper_binding_pattern self, %self.param_patt.067 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.5: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc44_33.4 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.5: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.5 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.6: type = fn_type @Destroy.WithSelf.Op.loc44_33.4 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.6: %Destroy.WithSelf.Op.type.ef016f.6 = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.fe2: type = pattern_type %DefaultOptionalStorage.9a0 [concrete]
// CHECK:STDOUT:   %self.param_patt.566: %pattern_type.fe2 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.691: %pattern_type.fe2 = wrapper_binding_pattern self, %self.param_patt.566 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.6: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc44_33.5 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.6: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.6 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.7: type = fn_type @Destroy.WithSelf.Op.loc44_33.5 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.7: %Destroy.WithSelf.Op.type.ef016f.7 = struct_value () [concrete]
// CHECK:STDOUT:   %self.param_patt.be1: %pattern_type.196 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.763: %pattern_type.196 = wrapper_binding_pattern self, %self.param_patt.be1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.7: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc44_33.6 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.7: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.7 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.8: type = fn_type @Destroy.WithSelf.Op.loc44_33.6 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.8: %Destroy.WithSelf.Op.type.ef016f.8 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.8: type = fn_type @Destroy.WithSelf.SelfDestruct.loc44_33.6 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.8: %Destroy.WithSelf.SelfDestruct.type.fbceb5.8 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.662: <witness> = lookup_impl_witness %impl.elem1.fbc, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.51e: type = fn_type @Destroy.WithSelf.SelfDestruct.1, @Destroy.WithSelf(%impl.elem1.fbc) [symbolic]
// CHECK:STDOUT:   %.6f9: type = fn_type_with_self_type %Destroy.WithSelf.SelfDestruct.type.51e, %impl.elem1.fbc [symbolic]
// CHECK:STDOUT:   %impl.elem2.a34: %.6f9 = impl_witness_access %Destroy.lookup_impl_witness.662, element2 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.f40: <specific function> = specific_impl_function %impl.elem2.a34, @Destroy.WithSelf.SelfDestruct.1(%impl.elem1.fbc) [symbolic]
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
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.9: type = fn_type @Destroy.WithSelf.Op.3 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.9: %Destroy.WithSelf.Op.type.ef016f.9 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.9: type = fn_type @Destroy.WithSelf.SelfDestruct.3 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.9: %Destroy.WithSelf.SelfDestruct.type.fbceb5.9 = struct_value () [concrete]
// CHECK:STDOUT:   %Iterator.SubobjectDestroy.type.45a: type = fn_type @Iterator.SubobjectDestroy.1 [concrete]
// CHECK:STDOUT:   %Iterator.SubobjectDestroy.d6a: %Iterator.SubobjectDestroy.type.45a = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.db9baa.1: <witness> = custom_witness (%Destroy.WithSelf.Op.403171.9, %Iterator.SubobjectDestroy.d6a, %Destroy.WithSelf.SelfDestruct.db3fdb.9), @Destroy [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.41bbea.2: type = fn_type @Iterator.Op.2 [concrete]
// CHECK:STDOUT:   %Iterator.Op.1caa36.2: %Iterator.Op.type.41bbea.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.1c9: <witness> = custom_witness (%Iterator.Op.1caa36.2), @Inc [concrete]
// CHECK:STDOUT:   %Equal.type.ca8475.1: type = fn_type @Equal.1 [concrete]
// CHECK:STDOUT:   %Equal.285b15.1: %Equal.type.ca8475.1 = struct_value () [concrete]
// CHECK:STDOUT:   %NotEqual.type.53d9ee.1: type = fn_type @NotEqual.1 [concrete]
// CHECK:STDOUT:   %NotEqual.119263.1: %NotEqual.type.53d9ee.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.723: <witness> = custom_witness (%Equal.285b15.1, %NotEqual.119263.1), @EqWith, @EqWith(%Sentinel.b4c) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.10: type = fn_type @Destroy.WithSelf.Op.loc51 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.10: %Destroy.WithSelf.Op.type.ef016f.10 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.10: type = fn_type @Destroy.WithSelf.SelfDestruct.loc51 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.10: %Destroy.WithSelf.SelfDestruct.type.fbceb5.10 = struct_value () [concrete]
// CHECK:STDOUT:   %Sentinel.SubobjectDestroy.type.45a: type = fn_type @Sentinel.SubobjectDestroy.1 [concrete]
// CHECK:STDOUT:   %Sentinel.SubobjectDestroy.d6a: %Sentinel.SubobjectDestroy.type.45a = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.db9baa.2: <witness> = custom_witness (%Destroy.WithSelf.Op.403171.10, %Sentinel.SubobjectDestroy.d6a, %Destroy.WithSelf.SelfDestruct.db3fdb.10), @Destroy [concrete]
// CHECK:STDOUT:   %facet_value.39a: %CppRange_where.type.d682d6.1 = facet_value %DiffMutableRange, (%custom_witness.1bfa2a.1, %custom_witness.d0da43.1, %custom_witness.aa8, %custom_witness.db9baa.1, %custom_witness.392, %custom_witness.1c9, %custom_witness.723, %custom_witness.db9baa.2) [concrete]
// CHECK:STDOUT:   %tuple.type.b18: type = tuple_type (%Iterator.f25, %Sentinel.b4c) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.8: type = fn_type @Destroy.WithSelf.SubobjectDestroy.3 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.8: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.8 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.11: type = fn_type @Destroy.WithSelf.Op.4 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.11: %Destroy.WithSelf.Op.type.ef016f.11 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.11: type = fn_type @Destroy.WithSelf.SelfDestruct.4 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.11: %Destroy.WithSelf.SelfDestruct.type.fbceb5.11 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.f8f19d.8: <witness> = custom_witness (%Destroy.WithSelf.Op.403171.11, %Destroy.WithSelf.SubobjectDestroy.d01daf.8, %Destroy.WithSelf.SelfDestruct.db3fdb.11), @Destroy [concrete]
// CHECK:STDOUT:   %Destroy.facet.645: %Destroy.type = facet_value %tuple.type.b18, (%custom_witness.f8f19d.8) [concrete]
// CHECK:STDOUT:   %Iterate.impl_witness.dbb: <witness> = impl_witness imports.%Iterate.impl_witness_table.121, @T.as_type.as.Iterate.impl(%facet_value.39a) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.type.0c6: type = fn_type @T.as_type.as.Iterate.impl.NewCursor, @T.as_type.as.Iterate.impl(%facet_value.39a) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.755: %T.as_type.as.Iterate.impl.NewCursor.type.0c6 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.type.f5a: type = fn_type @T.as_type.as.Iterate.impl.Next, @T.as_type.as.Iterate.impl(%facet_value.39a) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.8e7: %T.as_type.as.Iterate.impl.Next.type.f5a = struct_value () [concrete]
// CHECK:STDOUT:   %Iterate.facet.9e0: %Iterate.type = facet_value %DiffMutableRange, (%Iterate.impl_witness.dbb) [concrete]
// CHECK:STDOUT:   %facet_value.70f: %Iterate_where.type.b96 = facet_value %DiffMutableRange, (%Iterate.impl_witness.dbb) [concrete]
// CHECK:STDOUT:   %r.patt.5fe: %pattern_type.611 = ref_binding_pattern r [concrete]
// CHECK:STDOUT:   %r.param_patt.a4b: %pattern_type.611 = var_param_pattern %r.patt.5fe [concrete]
// CHECK:STDOUT:   %Iterator.a59: type = class_type @Iterator.2 [concrete]
// CHECK:STDOUT:   %Sentinel.b33: type = class_type @Sentinel.2 [concrete]
// CHECK:STDOUT:   %DiffConstRange.Begin.type: type = fn_type @DiffConstRange.Begin [concrete]
// CHECK:STDOUT:   %DiffConstRange.Begin: %DiffConstRange.Begin.type = struct_value () [concrete]
// CHECK:STDOUT:   %DiffConstRange.End.type: type = fn_type @DiffConstRange.End [concrete]
// CHECK:STDOUT:   %DiffConstRange.End: %DiffConstRange.End.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.7b4: <witness> = custom_witness (%Iterator.a59, %Sentinel.b33, %DiffConstRange.Begin, %DiffConstRange.End), @CppRangeForIterate [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.b7ae80.1: type = fn_type @Iterator.Op.3 [concrete]
// CHECK:STDOUT:   %Iterator.Op.690fee.1: %Iterator.Op.type.b7ae80.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.0c0: <witness> = custom_witness (%DiffValueType, %Iterator.Op.690fee.1), @CppUnsafeDeref [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.13: type = fn_type @Destroy.WithSelf.Op.5 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.13: %Destroy.WithSelf.Op.type.ef016f.13 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.13: type = fn_type @Destroy.WithSelf.SelfDestruct.5 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.13: %Destroy.WithSelf.SelfDestruct.type.fbceb5.13 = struct_value () [concrete]
// CHECK:STDOUT:   %Iterator.SubobjectDestroy.type.e2b: type = fn_type @Iterator.SubobjectDestroy.2 [concrete]
// CHECK:STDOUT:   %Iterator.SubobjectDestroy.21a: %Iterator.SubobjectDestroy.type.e2b = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.48b6e9.1: <witness> = custom_witness (%Destroy.WithSelf.Op.403171.13, %Iterator.SubobjectDestroy.21a, %Destroy.WithSelf.SelfDestruct.db3fdb.13), @Destroy [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.b7ae80.2: type = fn_type @Iterator.Op.4 [concrete]
// CHECK:STDOUT:   %Iterator.Op.690fee.2: %Iterator.Op.type.b7ae80.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.ffb: <witness> = custom_witness (%Iterator.Op.690fee.2), @Inc [concrete]
// CHECK:STDOUT:   %Equal.type.ca8475.2: type = fn_type @Equal.2 [concrete]
// CHECK:STDOUT:   %Equal.285b15.2: %Equal.type.ca8475.2 = struct_value () [concrete]
// CHECK:STDOUT:   %NotEqual.type.53d9ee.2: type = fn_type @NotEqual.2 [concrete]
// CHECK:STDOUT:   %NotEqual.119263.2: %NotEqual.type.53d9ee.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.a26: <witness> = custom_witness (%Equal.285b15.2, %NotEqual.119263.2), @EqWith, @EqWith(%Sentinel.b33) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.14: type = fn_type @Destroy.WithSelf.Op.loc52 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.14: %Destroy.WithSelf.Op.type.ef016f.14 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.14: type = fn_type @Destroy.WithSelf.SelfDestruct.loc52 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.14: %Destroy.WithSelf.SelfDestruct.type.fbceb5.14 = struct_value () [concrete]
// CHECK:STDOUT:   %Sentinel.SubobjectDestroy.type.e2b: type = fn_type @Sentinel.SubobjectDestroy.2 [concrete]
// CHECK:STDOUT:   %Sentinel.SubobjectDestroy.21a: %Sentinel.SubobjectDestroy.type.e2b = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.48b6e9.2: <witness> = custom_witness (%Destroy.WithSelf.Op.403171.14, %Sentinel.SubobjectDestroy.21a, %Destroy.WithSelf.SelfDestruct.db3fdb.14), @Destroy [concrete]
// CHECK:STDOUT:   %facet_value.9a7: %CppRange_where.type.d682d6.1 = facet_value %DiffConstRange, (%custom_witness.1bfa2a.1, %custom_witness.d0da43.1, %custom_witness.7b4, %custom_witness.48b6e9.1, %custom_witness.0c0, %custom_witness.ffb, %custom_witness.a26, %custom_witness.48b6e9.2) [concrete]
// CHECK:STDOUT:   %tuple.type.df5: type = tuple_type (%Iterator.a59, %Sentinel.b33) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.9: type = fn_type @Destroy.WithSelf.SubobjectDestroy.4 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.9: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.9 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.15: type = fn_type @Destroy.WithSelf.Op.6 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.15: %Destroy.WithSelf.Op.type.ef016f.15 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.15: type = fn_type @Destroy.WithSelf.SelfDestruct.6 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.15: %Destroy.WithSelf.SelfDestruct.type.fbceb5.15 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.f8f19d.9: <witness> = custom_witness (%Destroy.WithSelf.Op.403171.15, %Destroy.WithSelf.SubobjectDestroy.d01daf.9, %Destroy.WithSelf.SelfDestruct.db3fdb.15), @Destroy [concrete]
// CHECK:STDOUT:   %Destroy.facet.206: %Destroy.type = facet_value %tuple.type.df5, (%custom_witness.f8f19d.9) [concrete]
// CHECK:STDOUT:   %Iterate.impl_witness.1bb: <witness> = impl_witness imports.%Iterate.impl_witness_table.121, @T.as_type.as.Iterate.impl(%facet_value.9a7) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.type.b08: type = fn_type @T.as_type.as.Iterate.impl.NewCursor, @T.as_type.as.Iterate.impl(%facet_value.9a7) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.094: %T.as_type.as.Iterate.impl.NewCursor.type.b08 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.type.dc8: type = fn_type @T.as_type.as.Iterate.impl.Next, @T.as_type.as.Iterate.impl(%facet_value.9a7) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.33b: %T.as_type.as.Iterate.impl.Next.type.dc8 = struct_value () [concrete]
// CHECK:STDOUT:   %Iterate.facet.873: %Iterate.type = facet_value %DiffConstRange, (%Iterate.impl_witness.1bb) [concrete]
// CHECK:STDOUT:   %facet_value.4ce: %Iterate_where.type.b96 = facet_value %DiffConstRange, (%Iterate.impl_witness.1bb) [concrete]
// CHECK:STDOUT:   %r.patt.9c0: %pattern_type.7dd = ref_binding_pattern r [concrete]
// CHECK:STDOUT:   %r.param_patt.992: %pattern_type.7dd = var_param_pattern %r.patt.9c0 [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.d66: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.9e0) [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.37e: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.9e0) [concrete]
// CHECK:STDOUT:   %.ae2: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.d66, %Iterate.facet.9e0 [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.specific_fn.210: <specific function> = specific_function %T.as_type.as.Iterate.impl.NewCursor.755, @T.as_type.as.Iterate.impl.NewCursor(%facet_value.39a) [concrete]
// CHECK:STDOUT:   %ptr.3ce: type = ptr_type %tuple.type.b18 [concrete]
// CHECK:STDOUT:   %.0f5: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.37e, %Iterate.facet.9e0 [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.specific_fn.f32: <specific function> = specific_function %T.as_type.as.Iterate.impl.Next.8e7, @T.as_type.as.Iterate.impl.Next(%facet_value.39a) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.690: type = fn_type @Destroy.WithSelf.SelfDestruct.1, @Destroy.WithSelf(%Destroy.facet.645) [concrete]
// CHECK:STDOUT:   %.95e: type = fn_type_with_self_type %Destroy.WithSelf.SelfDestruct.type.690, %Destroy.facet.645 [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.7e8: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.873) [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.b3c: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.873) [concrete]
// CHECK:STDOUT:   %.88e: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.7e8, %Iterate.facet.873 [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.specific_fn.ac7: <specific function> = specific_function %T.as_type.as.Iterate.impl.NewCursor.094, @T.as_type.as.Iterate.impl.NewCursor(%facet_value.9a7) [concrete]
// CHECK:STDOUT:   %ptr.310: type = ptr_type %tuple.type.df5 [concrete]
// CHECK:STDOUT:   %.0fb: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.b3c, %Iterate.facet.873 [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.specific_fn.53e: <specific function> = specific_function %T.as_type.as.Iterate.impl.Next.33b, @T.as_type.as.Iterate.impl.Next(%facet_value.9a7) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.8c6: type = fn_type @Destroy.WithSelf.SelfDestruct.1, @Destroy.WithSelf(%Destroy.facet.206) [concrete]
// CHECK:STDOUT:   %.426: type = fn_type_with_self_type %Destroy.WithSelf.SelfDestruct.type.8c6, %Destroy.facet.206 [concrete]
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
// CHECK:STDOUT: generated {
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.1: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc44_33.1 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.2] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.98c = ref_param_pattern [concrete = constants.%self.param_patt.47c]
// CHECK:STDOUT:     %self.patt: %pattern_type.98c = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.90b]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %.a18 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %.a18 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.2: %Destroy.WithSelf.Op.type.ef016f.3 = fn_decl @Destroy.WithSelf.Op.loc44_33.1 [concrete = constants.%Destroy.WithSelf.Op.403171.3] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.98c = ref_param_pattern [concrete = constants.%self.param_patt.47c]
// CHECK:STDOUT:     %self.patt: %pattern_type.98c = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.90b]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %.a18 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %.a18 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.2: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.3 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc44_33.2 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.3] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.ea0 = ref_param_pattern [concrete = constants.%self.param_patt.4d6]
// CHECK:STDOUT:     %self.patt: %pattern_type.ea0 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.5e0]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %MaybeUnformed.632 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %MaybeUnformed.632 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.3: %Destroy.WithSelf.Op.type.ef016f.4 = fn_decl @Destroy.WithSelf.Op.loc44_33.2 [concrete = constants.%Destroy.WithSelf.Op.403171.4] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.ea0 = ref_param_pattern [concrete = constants.%self.param_patt.4d6]
// CHECK:STDOUT:     %self.patt: %pattern_type.ea0 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.5e0]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %MaybeUnformed.632 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %MaybeUnformed.632 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.3: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.4 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc44_33.3 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.4] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.831 = ref_param_pattern [concrete = constants.%self.param_patt.40d]
// CHECK:STDOUT:     %self.patt: %pattern_type.831 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.560]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref bool = ref_param call_param0
// CHECK:STDOUT:     %self: ref bool = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.4: %Destroy.WithSelf.Op.type.ef016f.5 = fn_decl @Destroy.WithSelf.Op.loc44_33.3 [concrete = constants.%Destroy.WithSelf.Op.403171.5] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.831 = ref_param_pattern [concrete = constants.%self.param_patt.40d]
// CHECK:STDOUT:     %self.patt: %pattern_type.831 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.560]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref bool = ref_param call_param0
// CHECK:STDOUT:     %self: ref bool = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.4: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.5 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc44_33.4 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.5] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.f90 = ref_param_pattern [concrete = constants.%self.param_patt.067]
// CHECK:STDOUT:     %self.patt: %pattern_type.f90 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.fc6]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %struct_type.value.has_value.851 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %struct_type.value.has_value.851 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.5: %Destroy.WithSelf.Op.type.ef016f.6 = fn_decl @Destroy.WithSelf.Op.loc44_33.4 [concrete = constants.%Destroy.WithSelf.Op.403171.6] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.f90 = ref_param_pattern [concrete = constants.%self.param_patt.067]
// CHECK:STDOUT:     %self.patt: %pattern_type.f90 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.fc6]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %struct_type.value.has_value.851 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %struct_type.value.has_value.851 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.5: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.6 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc44_33.5 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.6] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.fe2 = ref_param_pattern [concrete = constants.%self.param_patt.566]
// CHECK:STDOUT:     %self.patt: %pattern_type.fe2 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.691]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %DefaultOptionalStorage.9a0 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %DefaultOptionalStorage.9a0 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.6: %Destroy.WithSelf.Op.type.ef016f.7 = fn_decl @Destroy.WithSelf.Op.loc44_33.5 [concrete = constants.%Destroy.WithSelf.Op.403171.7] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.fe2 = ref_param_pattern [concrete = constants.%self.param_patt.566]
// CHECK:STDOUT:     %self.patt: %pattern_type.fe2 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.691]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %DefaultOptionalStorage.9a0 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %DefaultOptionalStorage.9a0 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.6: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.7 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc44_33.6 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.7] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.196 = ref_param_pattern [concrete = constants.%self.param_patt.be1]
// CHECK:STDOUT:     %self.patt: %pattern_type.196 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.763]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %Optional.d06 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %Optional.d06 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.7: %Destroy.WithSelf.Op.type.ef016f.8 = fn_decl @Destroy.WithSelf.Op.loc44_33.6 [concrete = constants.%Destroy.WithSelf.Op.403171.8] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.196 = ref_param_pattern [concrete = constants.%self.param_patt.be1]
// CHECK:STDOUT:     %self.patt: %pattern_type.196 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.763]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %Optional.d06 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %Optional.d06 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @TestIterate(%R.loc40_17.2: %Iterate_where.type.b96) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Iterate.lookup_impl_witness: <witness> = lookup_impl_witness %R.loc40_17.1, @Iterate [symbolic = %Iterate.lookup_impl_witness (constants.%Iterate.lookup_impl_witness.d00)]
// CHECK:STDOUT:   %Iterate.facet.loc44_33.5: %Iterate.type = facet_value %R.as_type.loc40_79.1, (%Iterate.lookup_impl_witness) [symbolic = %Iterate.facet.loc44_33.5 (constants.%Iterate.facet.f03)]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.loc44_33.5) [symbolic = %Iterate.WithSelf.NewCursor.type (constants.%Iterate.WithSelf.NewCursor.type.de8)]
// CHECK:STDOUT:   %.loc44_33.20: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type, %Iterate.facet.loc44_33.5 [symbolic = %.loc44_33.20 (constants.%.c20)]
// CHECK:STDOUT:   %impl.elem2.loc44_33.3: @TestIterate.%.loc44_33.20 (%.c20) = impl_witness_access %Iterate.lookup_impl_witness, element2 [symbolic = %impl.elem2.loc44_33.3 (constants.%impl.elem2.7d0)]
// CHECK:STDOUT:   %specific_impl_fn.loc44_33.4: <specific function> = specific_impl_function %impl.elem2.loc44_33.3, @Iterate.WithSelf.NewCursor(%Iterate.facet.loc44_33.5) [symbolic = %specific_impl_fn.loc44_33.4 (constants.%specific_impl_fn.1c3)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %ptr: type = ptr_type %as_type [symbolic = %ptr (constants.%ptr.02d)]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.loc44_33.5) [symbolic = %Iterate.WithSelf.Next.type (constants.%Iterate.WithSelf.Next.type.e6f)]
// CHECK:STDOUT:   %.loc44_33.21: type = fn_type_with_self_type %Iterate.WithSelf.Next.type, %Iterate.facet.loc44_33.5 [symbolic = %.loc44_33.21 (constants.%.89f)]
// CHECK:STDOUT:   %impl.elem3.loc44_33.2: @TestIterate.%.loc44_33.21 (%.89f) = impl_witness_access %Iterate.lookup_impl_witness, element3 [symbolic = %impl.elem3.loc44_33.2 (constants.%impl.elem3.73e)]
// CHECK:STDOUT:   %specific_impl_fn.loc44_33.5: <specific function> = specific_impl_function %impl.elem3.loc44_33.2, @Iterate.WithSelf.Next(%Iterate.facet.loc44_33.5) [symbolic = %specific_impl_fn.loc44_33.5 (constants.%specific_impl_fn.73e)]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type: type = fn_type @Destroy.WithSelf.SelfDestruct.1, @Destroy.WithSelf(%impl.elem1) [symbolic = %Destroy.WithSelf.SelfDestruct.type (constants.%Destroy.WithSelf.SelfDestruct.type.51e)]
// CHECK:STDOUT:   %.loc44_33.22: type = fn_type_with_self_type %Destroy.WithSelf.SelfDestruct.type, %impl.elem1 [symbolic = %.loc44_33.22 (constants.%.6f9)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %impl.elem1, @Destroy [symbolic = %Destroy.lookup_impl_witness (constants.%Destroy.lookup_impl_witness.662)]
// CHECK:STDOUT:   %impl.elem2.loc44_33.4: @TestIterate.%.loc44_33.22 (%.6f9) = impl_witness_access %Destroy.lookup_impl_witness, element2 [symbolic = %impl.elem2.loc44_33.4 (constants.%impl.elem2.a34)]
// CHECK:STDOUT:   %specific_impl_fn.loc44_33.6: <specific function> = specific_impl_function %impl.elem2.loc44_33.4, @Destroy.WithSelf.SelfDestruct.1(%impl.elem1) [symbolic = %specific_impl_fn.loc44_33.6 (constants.%specific_impl_fn.f40)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%r.param: ref @TestIterate.%R.as_type.loc40_79.1 (%R.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %i.patt: %pattern_type.149 = value_binding_pattern i [concrete = constants.%i.patt]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %r.ref: ref @TestIterate.%R.as_type.loc40_79.1 (%R.as_type) = name_ref r, %r
// CHECK:STDOUT:     %impl.elem2.loc44_33.1: @TestIterate.%.loc44_33.20 (%.c20) = impl_witness_access constants.%Iterate.lookup_impl_witness.d00, element2 [symbolic = %impl.elem2.loc44_33.3 (constants.%impl.elem2.7d0)]
// CHECK:STDOUT:     %bound_method.loc44_33.1: <bound method> = bound_method %r.ref, %impl.elem2.loc44_33.1
// CHECK:STDOUT:     %Iterate.facet.loc44_33.1: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.d00) [symbolic = %Iterate.facet.loc44_33.5 (constants.%Iterate.facet.f03)]
// CHECK:STDOUT:     %.loc44_33.1: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc44_33.1 [symbolic = %Iterate.facet.loc44_33.5 (constants.%Iterate.facet.f03)]
// CHECK:STDOUT:     %Iterate.facet.loc44_33.2: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.d00) [symbolic = %Iterate.facet.loc44_33.5 (constants.%Iterate.facet.f03)]
// CHECK:STDOUT:     %.loc44_33.2: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc44_33.2 [symbolic = %Iterate.facet.loc44_33.5 (constants.%Iterate.facet.f03)]
// CHECK:STDOUT:     %specific_impl_fn.loc44_33.1: <specific function> = specific_impl_function %impl.elem2.loc44_33.1, @Iterate.WithSelf.NewCursor(constants.%Iterate.facet.f03) [symbolic = %specific_impl_fn.loc44_33.4 (constants.%specific_impl_fn.1c3)]
// CHECK:STDOUT:     %bound_method.loc44_33.2: <bound method> = bound_method %r.ref, %specific_impl_fn.loc44_33.1
// CHECK:STDOUT:     %var: ref @TestIterate.%as_type (%as_type.389) = var_storage invalid
// CHECK:STDOUT:     %.loc44_32.1: @TestIterate.%R.as_type.loc40_79.1 (%R.as_type) = acquire_value %r.ref
// CHECK:STDOUT:     %Iterate.WithSelf.NewCursor.call: init @TestIterate.%as_type (%as_type.389) to %var = call %bound_method.loc44_33.2(%.loc44_32.1)
// CHECK:STDOUT:     assign %var, %Iterate.WithSelf.NewCursor.call
// CHECK:STDOUT:     br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.next:
// CHECK:STDOUT:     %addr.loc44: @TestIterate.%ptr (%ptr.02d) = addr_of %var
// CHECK:STDOUT:     %impl.elem3.loc44_33.1: @TestIterate.%.loc44_33.21 (%.89f) = impl_witness_access constants.%Iterate.lookup_impl_witness.d00, element3 [symbolic = %impl.elem3.loc44_33.2 (constants.%impl.elem3.73e)]
// CHECK:STDOUT:     %bound_method.loc44_33.3: <bound method> = bound_method %r.ref, %impl.elem3.loc44_33.1
// CHECK:STDOUT:     %Iterate.facet.loc44_33.3: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.d00) [symbolic = %Iterate.facet.loc44_33.5 (constants.%Iterate.facet.f03)]
// CHECK:STDOUT:     %.loc44_33.3: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc44_33.3 [symbolic = %Iterate.facet.loc44_33.5 (constants.%Iterate.facet.f03)]
// CHECK:STDOUT:     %Iterate.facet.loc44_33.4: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.d00) [symbolic = %Iterate.facet.loc44_33.5 (constants.%Iterate.facet.f03)]
// CHECK:STDOUT:     %.loc44_33.4: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc44_33.4 [symbolic = %Iterate.facet.loc44_33.5 (constants.%Iterate.facet.f03)]
// CHECK:STDOUT:     %.loc44_33.5: %Destroy.type = converted constants.%as_type.389, constants.%impl.elem1.fbc [symbolic = %impl.elem1 (constants.%impl.elem1.fbc)]
// CHECK:STDOUT:     %.loc44_33.6: %Destroy.type = converted constants.%as_type.389, constants.%impl.elem1.fbc [symbolic = %impl.elem1 (constants.%impl.elem1.fbc)]
// CHECK:STDOUT:     %specific_impl_fn.loc44_33.2: <specific function> = specific_impl_function %impl.elem3.loc44_33.1, @Iterate.WithSelf.Next(constants.%Iterate.facet.f03) [symbolic = %specific_impl_fn.loc44_33.5 (constants.%specific_impl_fn.73e)]
// CHECK:STDOUT:     %bound_method.loc44_33.4: <bound method> = bound_method %r.ref, %specific_impl_fn.loc44_33.2
// CHECK:STDOUT:     %.loc44_33.7: ref %Optional.d06 = temporary_storage
// CHECK:STDOUT:     %.loc44_32.2: @TestIterate.%R.as_type.loc40_79.1 (%R.as_type) = acquire_value %r.ref
// CHECK:STDOUT:     %Iterate.WithSelf.Next.call: init %Optional.d06 to %.loc44_33.7 = call %bound_method.loc44_33.4(%.loc44_32.2, %addr.loc44)
// CHECK:STDOUT:     %.loc44_33.8: ref %Optional.d06 = temporary %.loc44_33.7, %Iterate.WithSelf.Next.call
// CHECK:STDOUT:     %.loc44_33.9: %Optional.HasValue.type.821 = specific_constant imports.%Core.import_ref.bb6, @Optional(constants.%OptionalStorage.facet.beb) [concrete = constants.%Optional.HasValue.960]
// CHECK:STDOUT:     %HasValue.ref: %Optional.HasValue.type.821 = name_ref HasValue, %.loc44_33.9 [concrete = constants.%Optional.HasValue.960]
// CHECK:STDOUT:     %Optional.HasValue.bound: <bound method> = bound_method %.loc44_33.8, %HasValue.ref
// CHECK:STDOUT:     %Optional.HasValue.specific_fn: <specific function> = specific_function %HasValue.ref, @Optional.HasValue(constants.%OptionalStorage.facet.beb) [concrete = constants.%Optional.HasValue.specific_fn.a20]
// CHECK:STDOUT:     %bound_method.loc44_33.5: <bound method> = bound_method %.loc44_33.8, %Optional.HasValue.specific_fn
// CHECK:STDOUT:     %.loc44_33.10: %Optional.d06 = acquire_value %.loc44_33.8
// CHECK:STDOUT:     %Optional.HasValue.call: init bool = call %bound_method.loc44_33.5(%.loc44_33.10)
// CHECK:STDOUT:     %.loc44_33.11: bool = value_of_initializer %Optional.HasValue.call
// CHECK:STDOUT:     %.loc44_33.12: bool = converted %Optional.HasValue.call, %.loc44_33.11
// CHECK:STDOUT:     if %.loc44_33.12 br !for.body else br !for.done
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.body:
// CHECK:STDOUT:     %.loc44_33.13: %Optional.Get.type.f9c = specific_constant imports.%Core.import_ref.dab, @Optional(constants.%OptionalStorage.facet.beb) [concrete = constants.%Optional.Get.294]
// CHECK:STDOUT:     %Get.ref: %Optional.Get.type.f9c = name_ref Get, %.loc44_33.13 [concrete = constants.%Optional.Get.294]
// CHECK:STDOUT:     %Optional.Get.bound: <bound method> = bound_method %.loc44_33.8, %Get.ref
// CHECK:STDOUT:     %Optional.Get.specific_fn: <specific function> = specific_function %Get.ref, @Optional.Get(constants.%OptionalStorage.facet.beb) [concrete = constants.%Optional.Get.specific_fn.084]
// CHECK:STDOUT:     %bound_method.loc44_33.6: <bound method> = bound_method %.loc44_33.8, %Optional.Get.specific_fn
// CHECK:STDOUT:     %.loc44_33.14: ref %DiffValueType = temporary_storage
// CHECK:STDOUT:     %.loc44_33.15: %Optional.d06 = acquire_value %.loc44_33.8
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
// CHECK:STDOUT:     %Destroy.WithSelf.SelfDestruct.bound.loc44_33.1: <bound method> = bound_method %.loc44_33.16, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.2
// CHECK:STDOUT:     %Destroy.WithSelf.SelfDestruct.call.loc44_33.1: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc44_33.1(%.loc44_33.16)
// CHECK:STDOUT:     %Destroy.WithSelf.SelfDestruct.bound.loc44_33.2: <bound method> = bound_method %.loc44_33.8, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.8
// CHECK:STDOUT:     %Destroy.WithSelf.SelfDestruct.call.loc44_33.2: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc44_33.2(%.loc44_33.8)
// CHECK:STDOUT:     br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.done:
// CHECK:STDOUT:     %Destroy.WithSelf.SelfDestruct.bound.loc44_33.3: <bound method> = bound_method %.loc44_33.8, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.8
// CHECK:STDOUT:     %Destroy.WithSelf.SelfDestruct.call.loc44_33.3: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc44_33.3(%.loc44_33.8)
// CHECK:STDOUT:     %impl.elem2.loc44_33.2: @TestIterate.%.loc44_33.22 (%.6f9) = impl_witness_access constants.%Destroy.lookup_impl_witness.662, element2 [symbolic = %impl.elem2.loc44_33.4 (constants.%impl.elem2.a34)]
// CHECK:STDOUT:     %bound_method.loc44_33.7: <bound method> = bound_method %var, %impl.elem2.loc44_33.2
// CHECK:STDOUT:     %.loc44_33.18: %Destroy.type = converted constants.%as_type.389, constants.%impl.elem1.fbc [symbolic = %impl.elem1 (constants.%impl.elem1.fbc)]
// CHECK:STDOUT:     %.loc44_33.19: %Destroy.type = converted constants.%as_type.389, constants.%impl.elem1.fbc [symbolic = %impl.elem1 (constants.%impl.elem1.fbc)]
// CHECK:STDOUT:     %specific_impl_fn.loc44_33.3: <specific function> = specific_impl_function %impl.elem2.loc44_33.2, @Destroy.WithSelf.SelfDestruct.1(constants.%impl.elem1.fbc) [symbolic = %specific_impl_fn.loc44_33.6 (constants.%specific_impl_fn.f40)]
// CHECK:STDOUT:     %bound_method.loc44_33.8: <bound method> = bound_method %var, %specific_impl_fn.loc44_33.3
// CHECK:STDOUT:     %Destroy.WithSelf.SelfDestruct.call.loc44_33.4: init %empty_tuple.type = call %bound_method.loc44_33.8(%var)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc44_33.1(%self.param: ref %.a18) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc44_33.1(%self.param: ref %.a18) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc44_33.1(%self.param: ref %.a18) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.2(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.1(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc44_33.2(%self.param: ref %MaybeUnformed.632) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc44_33.2(%self.param: ref %MaybeUnformed.632) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc44_33.2(%self.param: ref %MaybeUnformed.632) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.3(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.2(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc44_33.3(%self.param: ref bool) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc44_33.3(%self.param: ref bool) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc44_33.3(%self.param: ref bool) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.4(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.3(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc44_33.4(%self.param: ref %struct_type.value.has_value.851) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc44_33.1: ref %MaybeUnformed.632 = struct_access %self.param, element0
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc44_33.1: <bound method> = bound_method %.loc44_33.1, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.4
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc44_33.1: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc44_33.1(%.loc44_33.1)
// CHECK:STDOUT:   %.loc44_33.2: ref bool = struct_access %self.param, element1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc44_33.2: <bound method> = bound_method %.loc44_33.2, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.5
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc44_33.2: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc44_33.2(%.loc44_33.2)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc44_33.4(%self.param: ref %struct_type.value.has_value.851) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc44_33.4(%self.param: ref %struct_type.value.has_value.851) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.5(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.4(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc44_33.5(%self.param: ref %DefaultOptionalStorage.9a0) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc44_33.5(%self.param: ref %DefaultOptionalStorage.9a0) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc44_33.5(%self.param: ref %DefaultOptionalStorage.9a0) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.6(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.5(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc44_33.6(%self.param: ref %Optional.d06) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc44_33.6(%self.param: ref %Optional.d06) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc44_33.6(%self.param: ref %Optional.d06) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.7(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.6(%self.param)
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
// CHECK:STDOUT:   %impl.elem2.loc59: %.ae2 = impl_witness_access constants.%Iterate.impl_witness.dbb, element2 [concrete = constants.%T.as_type.as.Iterate.impl.NewCursor.755]
// CHECK:STDOUT:   %bound_method.loc59_33.1: <bound method> = bound_method %m.ref, %impl.elem2.loc59
// CHECK:STDOUT:   %facet_value.loc59_33.1: %CppRange_where.type.d682d6.1 = facet_value constants.%DiffMutableRange, (constants.%custom_witness.1bfa2a.1, constants.%custom_witness.d0da43.1, constants.%custom_witness.aa8, constants.%custom_witness.db9baa.1, constants.%custom_witness.392, constants.%custom_witness.1c9, constants.%custom_witness.723, constants.%custom_witness.db9baa.2) [concrete = constants.%facet_value.39a]
// CHECK:STDOUT:   %.loc59_33.1: %CppRange_where.type.d682d6.1 = converted constants.%DiffMutableRange, %facet_value.loc59_33.1 [concrete = constants.%facet_value.39a]
// CHECK:STDOUT:   %facet_value.loc59_33.2: %CppRange_where.type.d682d6.1 = facet_value constants.%DiffMutableRange, (constants.%custom_witness.1bfa2a.1, constants.%custom_witness.d0da43.1, constants.%custom_witness.aa8, constants.%custom_witness.db9baa.1, constants.%custom_witness.392, constants.%custom_witness.1c9, constants.%custom_witness.723, constants.%custom_witness.db9baa.2) [concrete = constants.%facet_value.39a]
// CHECK:STDOUT:   %.loc59_33.2: %CppRange_where.type.d682d6.1 = converted constants.%DiffMutableRange, %facet_value.loc59_33.2 [concrete = constants.%facet_value.39a]
// CHECK:STDOUT:   %specific_fn.loc59_33.1: <specific function> = specific_function %impl.elem2.loc59, @T.as_type.as.Iterate.impl.NewCursor(constants.%facet_value.39a) [concrete = constants.%T.as_type.as.Iterate.impl.NewCursor.specific_fn.210]
// CHECK:STDOUT:   %bound_method.loc59_33.2: <bound method> = bound_method %m.ref, %specific_fn.loc59_33.1
// CHECK:STDOUT:   %var.loc59: ref %tuple.type.b18 = var_storage invalid
// CHECK:STDOUT:   %.loc59_32.1: %DiffMutableRange = acquire_value %m.ref
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.call.loc59: init %tuple.type.b18 to %var.loc59 = call %bound_method.loc59_33.2(%.loc59_32.1)
// CHECK:STDOUT:   assign %var.loc59, %T.as_type.as.Iterate.impl.NewCursor.call.loc59
// CHECK:STDOUT:   br !for.next.loc59
// CHECK:STDOUT:
// CHECK:STDOUT: !for.next.loc59:
// CHECK:STDOUT:   %addr.loc59: %ptr.3ce = addr_of %var.loc59
// CHECK:STDOUT:   %impl.elem3.loc59: %.0f5 = impl_witness_access constants.%Iterate.impl_witness.dbb, element3 [concrete = constants.%T.as_type.as.Iterate.impl.Next.8e7]
// CHECK:STDOUT:   %bound_method.loc59_33.3: <bound method> = bound_method %m.ref, %impl.elem3.loc59
// CHECK:STDOUT:   %facet_value.loc59_33.3: %CppRange_where.type.d682d6.1 = facet_value constants.%DiffMutableRange, (constants.%custom_witness.1bfa2a.1, constants.%custom_witness.d0da43.1, constants.%custom_witness.aa8, constants.%custom_witness.db9baa.1, constants.%custom_witness.392, constants.%custom_witness.1c9, constants.%custom_witness.723, constants.%custom_witness.db9baa.2) [concrete = constants.%facet_value.39a]
// CHECK:STDOUT:   %.loc59_33.3: %CppRange_where.type.d682d6.1 = converted constants.%DiffMutableRange, %facet_value.loc59_33.3 [concrete = constants.%facet_value.39a]
// CHECK:STDOUT:   %facet_value.loc59_33.4: %CppRange_where.type.d682d6.1 = facet_value constants.%DiffMutableRange, (constants.%custom_witness.1bfa2a.1, constants.%custom_witness.d0da43.1, constants.%custom_witness.aa8, constants.%custom_witness.db9baa.1, constants.%custom_witness.392, constants.%custom_witness.1c9, constants.%custom_witness.723, constants.%custom_witness.db9baa.2) [concrete = constants.%facet_value.39a]
// CHECK:STDOUT:   %.loc59_33.4: %CppRange_where.type.d682d6.1 = converted constants.%DiffMutableRange, %facet_value.loc59_33.4 [concrete = constants.%facet_value.39a]
// CHECK:STDOUT:   %specific_fn.loc59_33.2: <specific function> = specific_function %impl.elem3.loc59, @T.as_type.as.Iterate.impl.Next(constants.%facet_value.39a) [concrete = constants.%T.as_type.as.Iterate.impl.Next.specific_fn.f32]
// CHECK:STDOUT:   %bound_method.loc59_33.4: <bound method> = bound_method %m.ref, %specific_fn.loc59_33.2
// CHECK:STDOUT:   %.loc59_33.5: ref %Optional.d06 = temporary_storage
// CHECK:STDOUT:   %.loc59_32.2: %DiffMutableRange = acquire_value %m.ref
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.call.loc59: init %Optional.d06 to %.loc59_33.5 = call %bound_method.loc59_33.4(%.loc59_32.2, %addr.loc59)
// CHECK:STDOUT:   %.loc59_33.6: ref %Optional.d06 = temporary %.loc59_33.5, %T.as_type.as.Iterate.impl.Next.call.loc59
// CHECK:STDOUT:   %.loc59_33.7: %Optional.HasValue.type.821 = specific_constant imports.%Core.import_ref.bb6, @Optional(constants.%OptionalStorage.facet.beb) [concrete = constants.%Optional.HasValue.960]
// CHECK:STDOUT:   %HasValue.ref.loc59: %Optional.HasValue.type.821 = name_ref HasValue, %.loc59_33.7 [concrete = constants.%Optional.HasValue.960]
// CHECK:STDOUT:   %Optional.HasValue.bound.loc59: <bound method> = bound_method %.loc59_33.6, %HasValue.ref.loc59
// CHECK:STDOUT:   %Optional.HasValue.specific_fn.loc59: <specific function> = specific_function %HasValue.ref.loc59, @Optional.HasValue(constants.%OptionalStorage.facet.beb) [concrete = constants.%Optional.HasValue.specific_fn.a20]
// CHECK:STDOUT:   %bound_method.loc59_33.5: <bound method> = bound_method %.loc59_33.6, %Optional.HasValue.specific_fn.loc59
// CHECK:STDOUT:   %.loc59_33.8: %Optional.d06 = acquire_value %.loc59_33.6
// CHECK:STDOUT:   %Optional.HasValue.call.loc59: init bool = call %bound_method.loc59_33.5(%.loc59_33.8)
// CHECK:STDOUT:   %.loc59_33.9: bool = value_of_initializer %Optional.HasValue.call.loc59
// CHECK:STDOUT:   %.loc59_33.10: bool = converted %Optional.HasValue.call.loc59, %.loc59_33.9
// CHECK:STDOUT:   if %.loc59_33.10 br !for.body.loc59 else br !for.done.loc59
// CHECK:STDOUT:
// CHECK:STDOUT: !for.body.loc59:
// CHECK:STDOUT:   %.loc59_33.11: %Optional.Get.type.f9c = specific_constant imports.%Core.import_ref.dab, @Optional(constants.%OptionalStorage.facet.beb) [concrete = constants.%Optional.Get.294]
// CHECK:STDOUT:   %Get.ref.loc59: %Optional.Get.type.f9c = name_ref Get, %.loc59_33.11 [concrete = constants.%Optional.Get.294]
// CHECK:STDOUT:   %Optional.Get.bound.loc59: <bound method> = bound_method %.loc59_33.6, %Get.ref.loc59
// CHECK:STDOUT:   %Optional.Get.specific_fn.loc59: <specific function> = specific_function %Get.ref.loc59, @Optional.Get(constants.%OptionalStorage.facet.beb) [concrete = constants.%Optional.Get.specific_fn.084]
// CHECK:STDOUT:   %bound_method.loc59_33.6: <bound method> = bound_method %.loc59_33.6, %Optional.Get.specific_fn.loc59
// CHECK:STDOUT:   %.loc59_33.12: ref %DiffValueType = temporary_storage
// CHECK:STDOUT:   %.loc59_33.13: %Optional.d06 = acquire_value %.loc59_33.6
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
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc59_33.1: <bound method> = bound_method %.loc59_33.14, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.2
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc59_33.1: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc59_33.1(%.loc59_33.14)
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc59_33.2: <bound method> = bound_method %.loc59_33.6, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.8
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc59_33.2: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc59_33.2(%.loc59_33.6)
// CHECK:STDOUT:   br !for.next.loc59
// CHECK:STDOUT:
// CHECK:STDOUT: !for.done.loc59:
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc59_33.3: <bound method> = bound_method %.loc59_33.6, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.8
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc59_33.3: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc59_33.3(%.loc59_33.6)
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc59_33.4: <bound method> = bound_method %var.loc59, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.11
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc59_33.4: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc59_33.4(%var.loc59)
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %i.patt.loc63: %pattern_type.149 = value_binding_pattern i [concrete = constants.%i.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c.ref: %DiffConstRange = name_ref c, %c
// CHECK:STDOUT:   %impl.elem2.loc63: %.88e = impl_witness_access constants.%Iterate.impl_witness.1bb, element2 [concrete = constants.%T.as_type.as.Iterate.impl.NewCursor.094]
// CHECK:STDOUT:   %bound_method.loc63_33.1: <bound method> = bound_method %c.ref, %impl.elem2.loc63
// CHECK:STDOUT:   %facet_value.loc63_33.1: %CppRange_where.type.d682d6.1 = facet_value constants.%DiffConstRange, (constants.%custom_witness.1bfa2a.1, constants.%custom_witness.d0da43.1, constants.%custom_witness.7b4, constants.%custom_witness.48b6e9.1, constants.%custom_witness.0c0, constants.%custom_witness.ffb, constants.%custom_witness.a26, constants.%custom_witness.48b6e9.2) [concrete = constants.%facet_value.9a7]
// CHECK:STDOUT:   %.loc63_33.1: %CppRange_where.type.d682d6.1 = converted constants.%DiffConstRange, %facet_value.loc63_33.1 [concrete = constants.%facet_value.9a7]
// CHECK:STDOUT:   %facet_value.loc63_33.2: %CppRange_where.type.d682d6.1 = facet_value constants.%DiffConstRange, (constants.%custom_witness.1bfa2a.1, constants.%custom_witness.d0da43.1, constants.%custom_witness.7b4, constants.%custom_witness.48b6e9.1, constants.%custom_witness.0c0, constants.%custom_witness.ffb, constants.%custom_witness.a26, constants.%custom_witness.48b6e9.2) [concrete = constants.%facet_value.9a7]
// CHECK:STDOUT:   %.loc63_33.2: %CppRange_where.type.d682d6.1 = converted constants.%DiffConstRange, %facet_value.loc63_33.2 [concrete = constants.%facet_value.9a7]
// CHECK:STDOUT:   %specific_fn.loc63_33.1: <specific function> = specific_function %impl.elem2.loc63, @T.as_type.as.Iterate.impl.NewCursor(constants.%facet_value.9a7) [concrete = constants.%T.as_type.as.Iterate.impl.NewCursor.specific_fn.ac7]
// CHECK:STDOUT:   %bound_method.loc63_33.2: <bound method> = bound_method %c.ref, %specific_fn.loc63_33.1
// CHECK:STDOUT:   %var.loc63: ref %tuple.type.df5 = var_storage invalid
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.call.loc63: init %tuple.type.df5 to %var.loc63 = call %bound_method.loc63_33.2(%c.ref)
// CHECK:STDOUT:   assign %var.loc63, %T.as_type.as.Iterate.impl.NewCursor.call.loc63
// CHECK:STDOUT:   br !for.next.loc63
// CHECK:STDOUT:
// CHECK:STDOUT: !for.next.loc63:
// CHECK:STDOUT:   %addr.loc63: %ptr.310 = addr_of %var.loc63
// CHECK:STDOUT:   %impl.elem3.loc63: %.0fb = impl_witness_access constants.%Iterate.impl_witness.1bb, element3 [concrete = constants.%T.as_type.as.Iterate.impl.Next.33b]
// CHECK:STDOUT:   %bound_method.loc63_33.3: <bound method> = bound_method %c.ref, %impl.elem3.loc63
// CHECK:STDOUT:   %facet_value.loc63_33.3: %CppRange_where.type.d682d6.1 = facet_value constants.%DiffConstRange, (constants.%custom_witness.1bfa2a.1, constants.%custom_witness.d0da43.1, constants.%custom_witness.7b4, constants.%custom_witness.48b6e9.1, constants.%custom_witness.0c0, constants.%custom_witness.ffb, constants.%custom_witness.a26, constants.%custom_witness.48b6e9.2) [concrete = constants.%facet_value.9a7]
// CHECK:STDOUT:   %.loc63_33.3: %CppRange_where.type.d682d6.1 = converted constants.%DiffConstRange, %facet_value.loc63_33.3 [concrete = constants.%facet_value.9a7]
// CHECK:STDOUT:   %facet_value.loc63_33.4: %CppRange_where.type.d682d6.1 = facet_value constants.%DiffConstRange, (constants.%custom_witness.1bfa2a.1, constants.%custom_witness.d0da43.1, constants.%custom_witness.7b4, constants.%custom_witness.48b6e9.1, constants.%custom_witness.0c0, constants.%custom_witness.ffb, constants.%custom_witness.a26, constants.%custom_witness.48b6e9.2) [concrete = constants.%facet_value.9a7]
// CHECK:STDOUT:   %.loc63_33.4: %CppRange_where.type.d682d6.1 = converted constants.%DiffConstRange, %facet_value.loc63_33.4 [concrete = constants.%facet_value.9a7]
// CHECK:STDOUT:   %specific_fn.loc63_33.2: <specific function> = specific_function %impl.elem3.loc63, @T.as_type.as.Iterate.impl.Next(constants.%facet_value.9a7) [concrete = constants.%T.as_type.as.Iterate.impl.Next.specific_fn.53e]
// CHECK:STDOUT:   %bound_method.loc63_33.4: <bound method> = bound_method %c.ref, %specific_fn.loc63_33.2
// CHECK:STDOUT:   %.loc63_33.5: ref %Optional.d06 = temporary_storage
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.call.loc63: init %Optional.d06 to %.loc63_33.5 = call %bound_method.loc63_33.4(%c.ref, %addr.loc63)
// CHECK:STDOUT:   %.loc63_33.6: ref %Optional.d06 = temporary %.loc63_33.5, %T.as_type.as.Iterate.impl.Next.call.loc63
// CHECK:STDOUT:   %.loc63_33.7: %Optional.HasValue.type.821 = specific_constant imports.%Core.import_ref.bb6, @Optional(constants.%OptionalStorage.facet.beb) [concrete = constants.%Optional.HasValue.960]
// CHECK:STDOUT:   %HasValue.ref.loc63: %Optional.HasValue.type.821 = name_ref HasValue, %.loc63_33.7 [concrete = constants.%Optional.HasValue.960]
// CHECK:STDOUT:   %Optional.HasValue.bound.loc63: <bound method> = bound_method %.loc63_33.6, %HasValue.ref.loc63
// CHECK:STDOUT:   %Optional.HasValue.specific_fn.loc63: <specific function> = specific_function %HasValue.ref.loc63, @Optional.HasValue(constants.%OptionalStorage.facet.beb) [concrete = constants.%Optional.HasValue.specific_fn.a20]
// CHECK:STDOUT:   %bound_method.loc63_33.5: <bound method> = bound_method %.loc63_33.6, %Optional.HasValue.specific_fn.loc63
// CHECK:STDOUT:   %.loc63_33.8: %Optional.d06 = acquire_value %.loc63_33.6
// CHECK:STDOUT:   %Optional.HasValue.call.loc63: init bool = call %bound_method.loc63_33.5(%.loc63_33.8)
// CHECK:STDOUT:   %.loc63_33.9: bool = value_of_initializer %Optional.HasValue.call.loc63
// CHECK:STDOUT:   %.loc63_33.10: bool = converted %Optional.HasValue.call.loc63, %.loc63_33.9
// CHECK:STDOUT:   if %.loc63_33.10 br !for.body.loc63 else br !for.done.loc63
// CHECK:STDOUT:
// CHECK:STDOUT: !for.body.loc63:
// CHECK:STDOUT:   %.loc63_33.11: %Optional.Get.type.f9c = specific_constant imports.%Core.import_ref.dab, @Optional(constants.%OptionalStorage.facet.beb) [concrete = constants.%Optional.Get.294]
// CHECK:STDOUT:   %Get.ref.loc63: %Optional.Get.type.f9c = name_ref Get, %.loc63_33.11 [concrete = constants.%Optional.Get.294]
// CHECK:STDOUT:   %Optional.Get.bound.loc63: <bound method> = bound_method %.loc63_33.6, %Get.ref.loc63
// CHECK:STDOUT:   %Optional.Get.specific_fn.loc63: <specific function> = specific_function %Get.ref.loc63, @Optional.Get(constants.%OptionalStorage.facet.beb) [concrete = constants.%Optional.Get.specific_fn.084]
// CHECK:STDOUT:   %bound_method.loc63_33.6: <bound method> = bound_method %.loc63_33.6, %Optional.Get.specific_fn.loc63
// CHECK:STDOUT:   %.loc63_33.12: ref %DiffValueType = temporary_storage
// CHECK:STDOUT:   %.loc63_33.13: %Optional.d06 = acquire_value %.loc63_33.6
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
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc63_33.1: <bound method> = bound_method %.loc63_33.14, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.2
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc63_33.1: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc63_33.1(%.loc63_33.14)
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc63_33.2: <bound method> = bound_method %.loc63_33.6, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.8
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc63_33.2: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc63_33.2(%.loc63_33.6)
// CHECK:STDOUT:   br !for.next.loc63
// CHECK:STDOUT:
// CHECK:STDOUT: !for.done.loc63:
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc63_33.3: <bound method> = bound_method %.loc63_33.6, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.8
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc63_33.3: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc63_33.3(%.loc63_33.6)
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc63_33.4: <bound method> = bound_method %var.loc63, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.15
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc63_33.4: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc63_33.4(%var.loc63)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestIterate(constants.%R) {
// CHECK:STDOUT:   %R.patt.loc40_17.2 => constants.%R.patt
// CHECK:STDOUT:   %R.loc40_17.1 => constants.%R
// CHECK:STDOUT:   %R.as_type.loc40_79.1 => constants.%R.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.dc2
// CHECK:STDOUT:   %r.patt.loc40_77.2 => constants.%r.patt.917
// CHECK:STDOUT:   %r.param_patt.loc40_72.2 => constants.%r.param_patt.9a1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestIterate(constants.%facet_value.70f) {
// CHECK:STDOUT:   %R.patt.loc40_17.2 => constants.%R.patt
// CHECK:STDOUT:   %R.loc40_17.1 => constants.%facet_value.70f
// CHECK:STDOUT:   %R.as_type.loc40_79.1 => constants.%DiffMutableRange
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.611
// CHECK:STDOUT:   %r.patt.loc40_77.2 => constants.%r.patt.5fe
// CHECK:STDOUT:   %r.param_patt.loc40_72.2 => constants.%r.param_patt.a4b
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc40 => constants.%complete_type.357
// CHECK:STDOUT:   %Iterate.lookup_impl_witness => constants.%Iterate.impl_witness.dbb
// CHECK:STDOUT:   %Iterate.facet.loc44_33.5 => constants.%Iterate.facet.9e0
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type => constants.%Iterate.WithSelf.NewCursor.type.d66
// CHECK:STDOUT:   %.loc44_33.20 => constants.%.ae2
// CHECK:STDOUT:   %impl.elem2.loc44_33.3 => constants.%T.as_type.as.Iterate.impl.NewCursor.755
// CHECK:STDOUT:   %specific_impl_fn.loc44_33.4 => constants.%T.as_type.as.Iterate.impl.NewCursor.specific_fn.210
// CHECK:STDOUT:   %impl.elem1 => constants.%Destroy.facet.645
// CHECK:STDOUT:   %as_type => constants.%tuple.type.b18
// CHECK:STDOUT:   %require_complete.1 => constants.%complete_type.bfe
// CHECK:STDOUT:   %ptr => constants.%ptr.3ce
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type => constants.%Iterate.WithSelf.Next.type.37e
// CHECK:STDOUT:   %.loc44_33.21 => constants.%.0f5
// CHECK:STDOUT:   %impl.elem3.loc44_33.2 => constants.%T.as_type.as.Iterate.impl.Next.8e7
// CHECK:STDOUT:   %specific_impl_fn.loc44_33.5 => constants.%T.as_type.as.Iterate.impl.Next.specific_fn.f32
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type => constants.%Destroy.WithSelf.SelfDestruct.type.690
// CHECK:STDOUT:   %.loc44_33.22 => constants.%.95e
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.f8f19d.8
// CHECK:STDOUT:   %impl.elem2.loc44_33.4 => constants.%Destroy.WithSelf.SelfDestruct.db3fdb.11
// CHECK:STDOUT:   %specific_impl_fn.loc44_33.6 => constants.%Destroy.WithSelf.SelfDestruct.db3fdb.11
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestIterate(constants.%facet_value.4ce) {
// CHECK:STDOUT:   %R.patt.loc40_17.2 => constants.%R.patt
// CHECK:STDOUT:   %R.loc40_17.1 => constants.%facet_value.4ce
// CHECK:STDOUT:   %R.as_type.loc40_79.1 => constants.%DiffConstRange
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.7dd
// CHECK:STDOUT:   %r.patt.loc40_77.2 => constants.%r.patt.9c0
// CHECK:STDOUT:   %r.param_patt.loc40_72.2 => constants.%r.param_patt.992
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc40 => constants.%complete_type.357
// CHECK:STDOUT:   %Iterate.lookup_impl_witness => constants.%Iterate.impl_witness.1bb
// CHECK:STDOUT:   %Iterate.facet.loc44_33.5 => constants.%Iterate.facet.873
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type => constants.%Iterate.WithSelf.NewCursor.type.7e8
// CHECK:STDOUT:   %.loc44_33.20 => constants.%.88e
// CHECK:STDOUT:   %impl.elem2.loc44_33.3 => constants.%T.as_type.as.Iterate.impl.NewCursor.094
// CHECK:STDOUT:   %specific_impl_fn.loc44_33.4 => constants.%T.as_type.as.Iterate.impl.NewCursor.specific_fn.ac7
// CHECK:STDOUT:   %impl.elem1 => constants.%Destroy.facet.206
// CHECK:STDOUT:   %as_type => constants.%tuple.type.df5
// CHECK:STDOUT:   %require_complete.1 => constants.%complete_type.ad6
// CHECK:STDOUT:   %ptr => constants.%ptr.310
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type => constants.%Iterate.WithSelf.Next.type.b3c
// CHECK:STDOUT:   %.loc44_33.21 => constants.%.0fb
// CHECK:STDOUT:   %impl.elem3.loc44_33.2 => constants.%T.as_type.as.Iterate.impl.Next.33b
// CHECK:STDOUT:   %specific_impl_fn.loc44_33.5 => constants.%T.as_type.as.Iterate.impl.Next.specific_fn.53e
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type => constants.%Destroy.WithSelf.SelfDestruct.type.8c6
// CHECK:STDOUT:   %.loc44_33.22 => constants.%.426
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.f8f19d.9
// CHECK:STDOUT:   %impl.elem2.loc44_33.4 => constants.%Destroy.WithSelf.SelfDestruct.db3fdb.15
// CHECK:STDOUT:   %specific_impl_fn.loc44_33.6 => constants.%Destroy.WithSelf.SelfDestruct.db3fdb.15
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- method_overloads.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %Iterate.type: type = facet_type <@Iterate> [concrete]
// CHECK:STDOUT:   %OptionalStorage.type: type = facet_type <@OptionalStorage> [concrete]
// CHECK:STDOUT:   %pattern_type.831: type = pattern_type bool [concrete]
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
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.2: type = fn_type @Destroy.WithSelf.Op.loc25_56 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.2: %Destroy.WithSelf.Op.type.ef016f.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.2: type = fn_type @Destroy.WithSelf.SelfDestruct.loc25_56 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.2: %Destroy.WithSelf.SelfDestruct.type.fbceb5.2 = struct_value () [concrete]
// CHECK:STDOUT:   %OverloadValueType.SubobjectDestroy.type: type = fn_type @OverloadValueType.SubobjectDestroy [concrete]
// CHECK:STDOUT:   %OverloadValueType.SubobjectDestroy: %OverloadValueType.SubobjectDestroy.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.1bfa2a.1: <witness> = custom_witness (%Destroy.WithSelf.Op.403171.2, %OverloadValueType.SubobjectDestroy, %Destroy.WithSelf.SelfDestruct.db3fdb.2), @Destroy [concrete]
// CHECK:STDOUT:   %OverloadValueType.Op.type: type = fn_type @OverloadValueType.Op [concrete]
// CHECK:STDOUT:   %OverloadValueType.Op: %OverloadValueType.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.d0da43.1: <witness> = custom_witness (%OverloadValueType.Op), @Copy [concrete]
// CHECK:STDOUT:   %facet_value.8be: %facet_type.f48 = facet_value %OverloadValueType, (%custom_witness.1bfa2a.1, %custom_witness.d0da43.1) [concrete]
// CHECK:STDOUT:   %Iterate.lookup_impl_witness.1d8: <witness> = lookup_impl_witness %.Self.500, @Iterate [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.294: %facet_type.f48 = impl_witness_access %Iterate.lookup_impl_witness.1d8, element0 [symbolic_self]
// CHECK:STDOUT:   %Iterate_where.type.2ee: type = facet_type <@Iterate where %impl.elem0.294 = %facet_value.8be> [concrete]
// CHECK:STDOUT:   %pattern_type.e15: type = pattern_type %Iterate_where.type.2ee [concrete]
// CHECK:STDOUT:   %R.patt: %pattern_type.e15 = symbolic_binding_pattern R, 0 [symbolic]
// CHECK:STDOUT:   %R: %Iterate_where.type.2ee = symbolic_binding R, 0 [symbolic]
// CHECK:STDOUT:   %R.as_type: type = facet_access_type %R [symbolic]
// CHECK:STDOUT:   %pattern_type.a47: type = pattern_type %R.as_type [symbolic]
// CHECK:STDOUT:   %r.patt.24d: %pattern_type.a47 = ref_binding_pattern r [symbolic]
// CHECK:STDOUT:   %r.param_patt.d5b: %pattern_type.a47 = var_param_pattern %r.patt.24d [symbolic]
// CHECK:STDOUT:   %_.patt.a10: %pattern_type.804 = value_binding_pattern _ [concrete]
// CHECK:STDOUT:   %Iterate.lookup_impl_witness.e07: <witness> = lookup_impl_witness %R, @Iterate [symbolic]
// CHECK:STDOUT:   %Iterate.facet.8b0: %Iterate.type = facet_value %R.as_type, (%Iterate.lookup_impl_witness.e07) [symbolic]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.d55: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.8b0) [symbolic]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.627: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.8b0) [symbolic]
// CHECK:STDOUT:   %.489: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.d55, %Iterate.facet.8b0 [symbolic]
// CHECK:STDOUT:   %impl.elem2.348: %.489 = impl_witness_access %Iterate.lookup_impl_witness.e07, element2 [symbolic]
// CHECK:STDOUT:   %impl.elem1.0e3: %Destroy.type = impl_witness_access %Iterate.lookup_impl_witness.e07, element1 [symbolic]
// CHECK:STDOUT:   %as_type.400: type = facet_access_type %impl.elem1.0e3 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.e06: <specific function> = specific_impl_function %impl.elem2.348, @Iterate.WithSelf.NewCursor(%Iterate.facet.8b0) [symbolic]
// CHECK:STDOUT:   %ptr.196: type = ptr_type %as_type.400 [symbolic]
// CHECK:STDOUT:   %.fd2: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.627, %Iterate.facet.8b0 [symbolic]
// CHECK:STDOUT:   %impl.elem3.bb6: %.fd2 = impl_witness_access %Iterate.lookup_impl_witness.e07, element3 [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.fbc: type = class_type @DefaultOptionalStorage, @DefaultOptionalStorage(%facet_value.8be) [concrete]
// CHECK:STDOUT:   %OptionalStorage.impl_witness.bbc: <witness> = impl_witness imports.%OptionalStorage.impl_witness_table.460, @T.as_type.as.OptionalStorage.impl(%facet_value.8be) [concrete]
// CHECK:STDOUT:   %OptionalStorage.facet.50c: %OptionalStorage.type = facet_value %OverloadValueType, (%OptionalStorage.impl_witness.bbc) [concrete]
// CHECK:STDOUT:   %Optional.cb9: type = class_type @Optional, @Optional(%OptionalStorage.facet.50c) [concrete]
// CHECK:STDOUT:   %pattern_type.042: type = pattern_type %Optional.cb9 [concrete]
// CHECK:STDOUT:   %specific_impl_fn.622: <specific function> = specific_impl_function %impl.elem3.bb6, @Iterate.WithSelf.Next(%Iterate.facet.8b0) [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.type.800: type = fn_type @Optional.HasValue, @Optional(%OptionalStorage.facet.50c) [concrete]
// CHECK:STDOUT:   %Optional.HasValue.ab3: %Optional.HasValue.type.800 = struct_value () [concrete]
// CHECK:STDOUT:   %Optional.Get.type.466: type = fn_type @Optional.Get, @Optional(%OptionalStorage.facet.50c) [concrete]
// CHECK:STDOUT:   %Optional.Get.4ea: %Optional.Get.type.466 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.facet.d74: %Destroy.type = facet_value %OverloadValueType, (%custom_witness.1bfa2a.1) [concrete]
// CHECK:STDOUT:   %MaybeUnformed.d16: type = class_type @MaybeUnformed, @MaybeUnformed(%Destroy.facet.d74) [concrete]
// CHECK:STDOUT:   %.8a2: type = maybe_unformed_type %OverloadValueType [concrete]
// CHECK:STDOUT:   %struct_type.value.has_value.35f: type = struct_type {.value: %MaybeUnformed.d16, .has_value: bool} [concrete]
// CHECK:STDOUT:   %Optional.HasValue.specific_fn.730: <specific function> = specific_function %Optional.HasValue.ab3, @Optional.HasValue(%OptionalStorage.facet.50c) [concrete]
// CHECK:STDOUT:   %Optional.Get.specific_fn.f34: <specific function> = specific_function %Optional.Get.4ea, @Optional.Get(%OptionalStorage.facet.50c) [concrete]
// CHECK:STDOUT:   %pattern_type.6ff: type = pattern_type %.8a2 [concrete]
// CHECK:STDOUT:   %self.param_patt.b76: %pattern_type.6ff = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.079: %pattern_type.6ff = wrapper_binding_pattern self, %self.param_patt.b76 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc27_37.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.2: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.3: type = fn_type @Destroy.WithSelf.Op.loc27_37.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.3: %Destroy.WithSelf.Op.type.ef016f.3 = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.7a8: type = pattern_type %MaybeUnformed.d16 [concrete]
// CHECK:STDOUT:   %self.param_patt.d5e0: %pattern_type.7a8 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.7ed: %pattern_type.7a8 = wrapper_binding_pattern self, %self.param_patt.d5e0 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.3: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc27_37.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.3: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.3 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.4: type = fn_type @Destroy.WithSelf.Op.loc27_37.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.4: %Destroy.WithSelf.Op.type.ef016f.4 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.4: type = fn_type @Destroy.WithSelf.SelfDestruct.loc27_37.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.4: %Destroy.WithSelf.SelfDestruct.type.fbceb5.4 = struct_value () [concrete]
// CHECK:STDOUT:   %self.param_patt.40d: %pattern_type.831 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.560: %pattern_type.831 = wrapper_binding_pattern self, %self.param_patt.40d [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.4: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc27_37.3 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.4: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.4 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.5: type = fn_type @Destroy.WithSelf.Op.loc27_37.3 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.5: %Destroy.WithSelf.Op.type.ef016f.5 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.5: type = fn_type @Destroy.WithSelf.SelfDestruct.loc27_37.3 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.5: %Destroy.WithSelf.SelfDestruct.type.fbceb5.5 = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.d40: type = pattern_type %struct_type.value.has_value.35f [concrete]
// CHECK:STDOUT:   %self.param_patt.cda: %pattern_type.d40 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.8b2: %pattern_type.d40 = wrapper_binding_pattern self, %self.param_patt.cda [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.5: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc27_37.4 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.5: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.5 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.6: type = fn_type @Destroy.WithSelf.Op.loc27_37.4 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.6: %Destroy.WithSelf.Op.type.ef016f.6 = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.f46: type = pattern_type %DefaultOptionalStorage.fbc [concrete]
// CHECK:STDOUT:   %self.param_patt.c8b: %pattern_type.f46 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.cf5: %pattern_type.f46 = wrapper_binding_pattern self, %self.param_patt.c8b [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.6: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc27_37.5 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.6: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.6 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.7: type = fn_type @Destroy.WithSelf.Op.loc27_37.5 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.7: %Destroy.WithSelf.Op.type.ef016f.7 = struct_value () [concrete]
// CHECK:STDOUT:   %self.param_patt.594: %pattern_type.042 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.4a4: %pattern_type.042 = wrapper_binding_pattern self, %self.param_patt.594 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.7: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc27_37.6 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.7: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.7 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.8: type = fn_type @Destroy.WithSelf.Op.loc27_37.6 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.8: %Destroy.WithSelf.Op.type.ef016f.8 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.8: type = fn_type @Destroy.WithSelf.SelfDestruct.loc27_37.6 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.8: %Destroy.WithSelf.SelfDestruct.type.fbceb5.8 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.a78: <witness> = lookup_impl_witness %impl.elem1.0e3, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.360: type = fn_type @Destroy.WithSelf.SelfDestruct.1, @Destroy.WithSelf(%impl.elem1.0e3) [symbolic]
// CHECK:STDOUT:   %.73a: type = fn_type_with_self_type %Destroy.WithSelf.SelfDestruct.type.360, %impl.elem1.0e3 [symbolic]
// CHECK:STDOUT:   %impl.elem2.01e: %.73a = impl_witness_access %Destroy.lookup_impl_witness.a78, element2 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.dac: <specific function> = specific_impl_function %impl.elem2.01e, @Destroy.WithSelf.SelfDestruct.1(%impl.elem1.0e3) [symbolic]
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
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.9: type = fn_type @Destroy.WithSelf.Op.3 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.9: %Destroy.WithSelf.Op.type.ef016f.9 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.9: type = fn_type @Destroy.WithSelf.SelfDestruct.3 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.9: %Destroy.WithSelf.SelfDestruct.type.fbceb5.9 = struct_value () [concrete]
// CHECK:STDOUT:   %Iterator.SubobjectDestroy.type: type = fn_type @Iterator.SubobjectDestroy [concrete]
// CHECK:STDOUT:   %Iterator.SubobjectDestroy: %Iterator.SubobjectDestroy.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.eaa: <witness> = custom_witness (%Destroy.WithSelf.Op.403171.9, %Iterator.SubobjectDestroy, %Destroy.WithSelf.SelfDestruct.db3fdb.9), @Destroy [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.4232e0.2: type = fn_type @Iterator.Op.2 [concrete]
// CHECK:STDOUT:   %Iterator.Op.e8caec.2: %Iterator.Op.type.4232e0.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.716: <witness> = custom_witness (%Iterator.Op.e8caec.2), @Inc [concrete]
// CHECK:STDOUT:   %Equal.type: type = fn_type @Equal [concrete]
// CHECK:STDOUT:   %Equal: %Equal.type = struct_value () [concrete]
// CHECK:STDOUT:   %NotEqual.type: type = fn_type @NotEqual [concrete]
// CHECK:STDOUT:   %NotEqual: %NotEqual.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.617: <witness> = custom_witness (%Equal, %NotEqual), @EqWith, @EqWith(%Iterator) [concrete]
// CHECK:STDOUT:   %facet_value.670: %CppRange_where.type.d682d6.1 = facet_value %Range, (%custom_witness.1bfa2a.1, %custom_witness.d0da43.1, %custom_witness.090, %custom_witness.eaa, %custom_witness.6bd, %custom_witness.716, %custom_witness.617) [concrete]
// CHECK:STDOUT:   %tuple.type.9ef: type = tuple_type (%Iterator, %Iterator) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.8: type = fn_type @Destroy.WithSelf.SubobjectDestroy.3 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.8: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.8 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.10: type = fn_type @Destroy.WithSelf.Op.4 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.10: %Destroy.WithSelf.Op.type.ef016f.10 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.10: type = fn_type @Destroy.WithSelf.SelfDestruct.4 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.10: %Destroy.WithSelf.SelfDestruct.type.fbceb5.10 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.f8f19d.8: <witness> = custom_witness (%Destroy.WithSelf.Op.403171.10, %Destroy.WithSelf.SubobjectDestroy.d01daf.8, %Destroy.WithSelf.SelfDestruct.db3fdb.10), @Destroy [concrete]
// CHECK:STDOUT:   %Destroy.facet.c69: %Destroy.type = facet_value %tuple.type.9ef, (%custom_witness.f8f19d.8) [concrete]
// CHECK:STDOUT:   %Iterate.impl_witness.24f: <witness> = impl_witness imports.%Iterate.impl_witness_table.121, @T.as_type.as.Iterate.impl(%facet_value.670) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.type.1cd: type = fn_type @T.as_type.as.Iterate.impl.NewCursor, @T.as_type.as.Iterate.impl(%facet_value.670) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.8fd: %T.as_type.as.Iterate.impl.NewCursor.type.1cd = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.type.ac6: type = fn_type @T.as_type.as.Iterate.impl.Next, @T.as_type.as.Iterate.impl(%facet_value.670) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.4cb: %T.as_type.as.Iterate.impl.Next.type.ac6 = struct_value () [concrete]
// CHECK:STDOUT:   %Iterate.facet.726: %Iterate.type = facet_value %Range, (%Iterate.impl_witness.24f) [concrete]
// CHECK:STDOUT:   %facet_value.483: %Iterate_where.type.2ee = facet_value %Range, (%Iterate.impl_witness.24f) [concrete]
// CHECK:STDOUT:   %r.patt.b9f: %pattern_type.23f = ref_binding_pattern r [concrete]
// CHECK:STDOUT:   %r.param_patt.466: %pattern_type.23f = var_param_pattern %r.patt.b9f [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.292: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.726) [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.4f7: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.726) [concrete]
// CHECK:STDOUT:   %.b72: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.292, %Iterate.facet.726 [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.specific_fn: <specific function> = specific_function %T.as_type.as.Iterate.impl.NewCursor.8fd, @T.as_type.as.Iterate.impl.NewCursor(%facet_value.670) [concrete]
// CHECK:STDOUT:   %ptr.b18: type = ptr_type %tuple.type.9ef [concrete]
// CHECK:STDOUT:   %.0c0: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.4f7, %Iterate.facet.726 [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.specific_fn: <specific function> = specific_function %T.as_type.as.Iterate.impl.Next.4cb, @T.as_type.as.Iterate.impl.Next(%facet_value.670) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.6ef: type = fn_type @Destroy.WithSelf.SelfDestruct.1, @Destroy.WithSelf(%Destroy.facet.c69) [concrete]
// CHECK:STDOUT:   %.a39: type = fn_type_with_self_type %Destroy.WithSelf.SelfDestruct.type.6ef, %Destroy.facet.c69 [concrete]
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
// CHECK:STDOUT: generated {
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.1: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc27_37.1 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.2] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.6ff = ref_param_pattern [concrete = constants.%self.param_patt.b76]
// CHECK:STDOUT:     %self.patt: %pattern_type.6ff = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.079]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %.8a2 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %.8a2 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.2: %Destroy.WithSelf.Op.type.ef016f.3 = fn_decl @Destroy.WithSelf.Op.loc27_37.1 [concrete = constants.%Destroy.WithSelf.Op.403171.3] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.6ff = ref_param_pattern [concrete = constants.%self.param_patt.b76]
// CHECK:STDOUT:     %self.patt: %pattern_type.6ff = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.079]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %.8a2 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %.8a2 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.2: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.3 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc27_37.2 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.3] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.7a8 = ref_param_pattern [concrete = constants.%self.param_patt.d5e0]
// CHECK:STDOUT:     %self.patt: %pattern_type.7a8 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.7ed]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %MaybeUnformed.d16 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %MaybeUnformed.d16 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.3: %Destroy.WithSelf.Op.type.ef016f.4 = fn_decl @Destroy.WithSelf.Op.loc27_37.2 [concrete = constants.%Destroy.WithSelf.Op.403171.4] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.7a8 = ref_param_pattern [concrete = constants.%self.param_patt.d5e0]
// CHECK:STDOUT:     %self.patt: %pattern_type.7a8 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.7ed]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %MaybeUnformed.d16 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %MaybeUnformed.d16 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.3: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.4 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc27_37.3 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.4] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.831 = ref_param_pattern [concrete = constants.%self.param_patt.40d]
// CHECK:STDOUT:     %self.patt: %pattern_type.831 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.560]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref bool = ref_param call_param0
// CHECK:STDOUT:     %self: ref bool = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.4: %Destroy.WithSelf.Op.type.ef016f.5 = fn_decl @Destroy.WithSelf.Op.loc27_37.3 [concrete = constants.%Destroy.WithSelf.Op.403171.5] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.831 = ref_param_pattern [concrete = constants.%self.param_patt.40d]
// CHECK:STDOUT:     %self.patt: %pattern_type.831 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.560]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref bool = ref_param call_param0
// CHECK:STDOUT:     %self: ref bool = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.4: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.5 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc27_37.4 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.5] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.d40 = ref_param_pattern [concrete = constants.%self.param_patt.cda]
// CHECK:STDOUT:     %self.patt: %pattern_type.d40 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.8b2]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %struct_type.value.has_value.35f = ref_param call_param0
// CHECK:STDOUT:     %self: ref %struct_type.value.has_value.35f = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.5: %Destroy.WithSelf.Op.type.ef016f.6 = fn_decl @Destroy.WithSelf.Op.loc27_37.4 [concrete = constants.%Destroy.WithSelf.Op.403171.6] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.d40 = ref_param_pattern [concrete = constants.%self.param_patt.cda]
// CHECK:STDOUT:     %self.patt: %pattern_type.d40 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.8b2]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %struct_type.value.has_value.35f = ref_param call_param0
// CHECK:STDOUT:     %self: ref %struct_type.value.has_value.35f = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.5: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.6 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc27_37.5 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.6] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.f46 = ref_param_pattern [concrete = constants.%self.param_patt.c8b]
// CHECK:STDOUT:     %self.patt: %pattern_type.f46 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.cf5]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %DefaultOptionalStorage.fbc = ref_param call_param0
// CHECK:STDOUT:     %self: ref %DefaultOptionalStorage.fbc = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.6: %Destroy.WithSelf.Op.type.ef016f.7 = fn_decl @Destroy.WithSelf.Op.loc27_37.5 [concrete = constants.%Destroy.WithSelf.Op.403171.7] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.f46 = ref_param_pattern [concrete = constants.%self.param_patt.c8b]
// CHECK:STDOUT:     %self.patt: %pattern_type.f46 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.cf5]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %DefaultOptionalStorage.fbc = ref_param call_param0
// CHECK:STDOUT:     %self: ref %DefaultOptionalStorage.fbc = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.6: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.7 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc27_37.6 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.7] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.042 = ref_param_pattern [concrete = constants.%self.param_patt.594]
// CHECK:STDOUT:     %self.patt: %pattern_type.042 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.4a4]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %Optional.cb9 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %Optional.cb9 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.7: %Destroy.WithSelf.Op.type.ef016f.8 = fn_decl @Destroy.WithSelf.Op.loc27_37.6 [concrete = constants.%Destroy.WithSelf.Op.403171.8] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.042 = ref_param_pattern [concrete = constants.%self.param_patt.594]
// CHECK:STDOUT:     %self.patt: %pattern_type.042 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.4a4]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %Optional.cb9 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %Optional.cb9 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @TestIterate(%R.loc25_17.2: %Iterate_where.type.2ee) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Iterate.lookup_impl_witness: <witness> = lookup_impl_witness %R.loc25_17.1, @Iterate [symbolic = %Iterate.lookup_impl_witness (constants.%Iterate.lookup_impl_witness.e07)]
// CHECK:STDOUT:   %Iterate.facet.loc27_37.5: %Iterate.type = facet_value %R.as_type.loc25_83.1, (%Iterate.lookup_impl_witness) [symbolic = %Iterate.facet.loc27_37.5 (constants.%Iterate.facet.8b0)]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.loc27_37.5) [symbolic = %Iterate.WithSelf.NewCursor.type (constants.%Iterate.WithSelf.NewCursor.type.d55)]
// CHECK:STDOUT:   %.loc27_37.20: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type, %Iterate.facet.loc27_37.5 [symbolic = %.loc27_37.20 (constants.%.489)]
// CHECK:STDOUT:   %impl.elem2.loc27_37.3: @TestIterate.%.loc27_37.20 (%.489) = impl_witness_access %Iterate.lookup_impl_witness, element2 [symbolic = %impl.elem2.loc27_37.3 (constants.%impl.elem2.348)]
// CHECK:STDOUT:   %specific_impl_fn.loc27_37.4: <specific function> = specific_impl_function %impl.elem2.loc27_37.3, @Iterate.WithSelf.NewCursor(%Iterate.facet.loc27_37.5) [symbolic = %specific_impl_fn.loc27_37.4 (constants.%specific_impl_fn.e06)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %ptr: type = ptr_type %as_type [symbolic = %ptr (constants.%ptr.196)]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.loc27_37.5) [symbolic = %Iterate.WithSelf.Next.type (constants.%Iterate.WithSelf.Next.type.627)]
// CHECK:STDOUT:   %.loc27_37.21: type = fn_type_with_self_type %Iterate.WithSelf.Next.type, %Iterate.facet.loc27_37.5 [symbolic = %.loc27_37.21 (constants.%.fd2)]
// CHECK:STDOUT:   %impl.elem3.loc27_37.2: @TestIterate.%.loc27_37.21 (%.fd2) = impl_witness_access %Iterate.lookup_impl_witness, element3 [symbolic = %impl.elem3.loc27_37.2 (constants.%impl.elem3.bb6)]
// CHECK:STDOUT:   %specific_impl_fn.loc27_37.5: <specific function> = specific_impl_function %impl.elem3.loc27_37.2, @Iterate.WithSelf.Next(%Iterate.facet.loc27_37.5) [symbolic = %specific_impl_fn.loc27_37.5 (constants.%specific_impl_fn.622)]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type: type = fn_type @Destroy.WithSelf.SelfDestruct.1, @Destroy.WithSelf(%impl.elem1) [symbolic = %Destroy.WithSelf.SelfDestruct.type (constants.%Destroy.WithSelf.SelfDestruct.type.360)]
// CHECK:STDOUT:   %.loc27_37.22: type = fn_type_with_self_type %Destroy.WithSelf.SelfDestruct.type, %impl.elem1 [symbolic = %.loc27_37.22 (constants.%.73a)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %impl.elem1, @Destroy [symbolic = %Destroy.lookup_impl_witness (constants.%Destroy.lookup_impl_witness.a78)]
// CHECK:STDOUT:   %impl.elem2.loc27_37.4: @TestIterate.%.loc27_37.22 (%.73a) = impl_witness_access %Destroy.lookup_impl_witness, element2 [symbolic = %impl.elem2.loc27_37.4 (constants.%impl.elem2.01e)]
// CHECK:STDOUT:   %specific_impl_fn.loc27_37.6: <specific function> = specific_impl_function %impl.elem2.loc27_37.4, @Destroy.WithSelf.SelfDestruct.1(%impl.elem1) [symbolic = %specific_impl_fn.loc27_37.6 (constants.%specific_impl_fn.dac)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%r.param: ref @TestIterate.%R.as_type.loc25_83.1 (%R.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %_.patt: %pattern_type.804 = value_binding_pattern _ [concrete = constants.%_.patt.a10]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %r.ref: ref @TestIterate.%R.as_type.loc25_83.1 (%R.as_type) = name_ref r, %r
// CHECK:STDOUT:     %impl.elem2.loc27_37.1: @TestIterate.%.loc27_37.20 (%.489) = impl_witness_access constants.%Iterate.lookup_impl_witness.e07, element2 [symbolic = %impl.elem2.loc27_37.3 (constants.%impl.elem2.348)]
// CHECK:STDOUT:     %bound_method.loc27_37.1: <bound method> = bound_method %r.ref, %impl.elem2.loc27_37.1
// CHECK:STDOUT:     %Iterate.facet.loc27_37.1: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.e07) [symbolic = %Iterate.facet.loc27_37.5 (constants.%Iterate.facet.8b0)]
// CHECK:STDOUT:     %.loc27_37.1: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc27_37.1 [symbolic = %Iterate.facet.loc27_37.5 (constants.%Iterate.facet.8b0)]
// CHECK:STDOUT:     %Iterate.facet.loc27_37.2: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.e07) [symbolic = %Iterate.facet.loc27_37.5 (constants.%Iterate.facet.8b0)]
// CHECK:STDOUT:     %.loc27_37.2: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc27_37.2 [symbolic = %Iterate.facet.loc27_37.5 (constants.%Iterate.facet.8b0)]
// CHECK:STDOUT:     %specific_impl_fn.loc27_37.1: <specific function> = specific_impl_function %impl.elem2.loc27_37.1, @Iterate.WithSelf.NewCursor(constants.%Iterate.facet.8b0) [symbolic = %specific_impl_fn.loc27_37.4 (constants.%specific_impl_fn.e06)]
// CHECK:STDOUT:     %bound_method.loc27_37.2: <bound method> = bound_method %r.ref, %specific_impl_fn.loc27_37.1
// CHECK:STDOUT:     %var: ref @TestIterate.%as_type (%as_type.400) = var_storage invalid
// CHECK:STDOUT:     %.loc27_36.1: @TestIterate.%R.as_type.loc25_83.1 (%R.as_type) = acquire_value %r.ref
// CHECK:STDOUT:     %Iterate.WithSelf.NewCursor.call: init @TestIterate.%as_type (%as_type.400) to %var = call %bound_method.loc27_37.2(%.loc27_36.1)
// CHECK:STDOUT:     assign %var, %Iterate.WithSelf.NewCursor.call
// CHECK:STDOUT:     br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.next:
// CHECK:STDOUT:     %addr: @TestIterate.%ptr (%ptr.196) = addr_of %var
// CHECK:STDOUT:     %impl.elem3.loc27_37.1: @TestIterate.%.loc27_37.21 (%.fd2) = impl_witness_access constants.%Iterate.lookup_impl_witness.e07, element3 [symbolic = %impl.elem3.loc27_37.2 (constants.%impl.elem3.bb6)]
// CHECK:STDOUT:     %bound_method.loc27_37.3: <bound method> = bound_method %r.ref, %impl.elem3.loc27_37.1
// CHECK:STDOUT:     %Iterate.facet.loc27_37.3: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.e07) [symbolic = %Iterate.facet.loc27_37.5 (constants.%Iterate.facet.8b0)]
// CHECK:STDOUT:     %.loc27_37.3: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc27_37.3 [symbolic = %Iterate.facet.loc27_37.5 (constants.%Iterate.facet.8b0)]
// CHECK:STDOUT:     %Iterate.facet.loc27_37.4: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.e07) [symbolic = %Iterate.facet.loc27_37.5 (constants.%Iterate.facet.8b0)]
// CHECK:STDOUT:     %.loc27_37.4: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc27_37.4 [symbolic = %Iterate.facet.loc27_37.5 (constants.%Iterate.facet.8b0)]
// CHECK:STDOUT:     %.loc27_37.5: %Destroy.type = converted constants.%as_type.400, constants.%impl.elem1.0e3 [symbolic = %impl.elem1 (constants.%impl.elem1.0e3)]
// CHECK:STDOUT:     %.loc27_37.6: %Destroy.type = converted constants.%as_type.400, constants.%impl.elem1.0e3 [symbolic = %impl.elem1 (constants.%impl.elem1.0e3)]
// CHECK:STDOUT:     %specific_impl_fn.loc27_37.2: <specific function> = specific_impl_function %impl.elem3.loc27_37.1, @Iterate.WithSelf.Next(constants.%Iterate.facet.8b0) [symbolic = %specific_impl_fn.loc27_37.5 (constants.%specific_impl_fn.622)]
// CHECK:STDOUT:     %bound_method.loc27_37.4: <bound method> = bound_method %r.ref, %specific_impl_fn.loc27_37.2
// CHECK:STDOUT:     %.loc27_37.7: ref %Optional.cb9 = temporary_storage
// CHECK:STDOUT:     %.loc27_36.2: @TestIterate.%R.as_type.loc25_83.1 (%R.as_type) = acquire_value %r.ref
// CHECK:STDOUT:     %Iterate.WithSelf.Next.call: init %Optional.cb9 to %.loc27_37.7 = call %bound_method.loc27_37.4(%.loc27_36.2, %addr)
// CHECK:STDOUT:     %.loc27_37.8: ref %Optional.cb9 = temporary %.loc27_37.7, %Iterate.WithSelf.Next.call
// CHECK:STDOUT:     %.loc27_37.9: %Optional.HasValue.type.800 = specific_constant imports.%Core.import_ref.bb6, @Optional(constants.%OptionalStorage.facet.50c) [concrete = constants.%Optional.HasValue.ab3]
// CHECK:STDOUT:     %HasValue.ref: %Optional.HasValue.type.800 = name_ref HasValue, %.loc27_37.9 [concrete = constants.%Optional.HasValue.ab3]
// CHECK:STDOUT:     %Optional.HasValue.bound: <bound method> = bound_method %.loc27_37.8, %HasValue.ref
// CHECK:STDOUT:     %Optional.HasValue.specific_fn: <specific function> = specific_function %HasValue.ref, @Optional.HasValue(constants.%OptionalStorage.facet.50c) [concrete = constants.%Optional.HasValue.specific_fn.730]
// CHECK:STDOUT:     %bound_method.loc27_37.5: <bound method> = bound_method %.loc27_37.8, %Optional.HasValue.specific_fn
// CHECK:STDOUT:     %.loc27_37.10: %Optional.cb9 = acquire_value %.loc27_37.8
// CHECK:STDOUT:     %Optional.HasValue.call: init bool = call %bound_method.loc27_37.5(%.loc27_37.10)
// CHECK:STDOUT:     %.loc27_37.11: bool = value_of_initializer %Optional.HasValue.call
// CHECK:STDOUT:     %.loc27_37.12: bool = converted %Optional.HasValue.call, %.loc27_37.11
// CHECK:STDOUT:     if %.loc27_37.12 br !for.body else br !for.done
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.body:
// CHECK:STDOUT:     %.loc27_37.13: %Optional.Get.type.466 = specific_constant imports.%Core.import_ref.dab, @Optional(constants.%OptionalStorage.facet.50c) [concrete = constants.%Optional.Get.4ea]
// CHECK:STDOUT:     %Get.ref: %Optional.Get.type.466 = name_ref Get, %.loc27_37.13 [concrete = constants.%Optional.Get.4ea]
// CHECK:STDOUT:     %Optional.Get.bound: <bound method> = bound_method %.loc27_37.8, %Get.ref
// CHECK:STDOUT:     %Optional.Get.specific_fn: <specific function> = specific_function %Get.ref, @Optional.Get(constants.%OptionalStorage.facet.50c) [concrete = constants.%Optional.Get.specific_fn.f34]
// CHECK:STDOUT:     %bound_method.loc27_37.6: <bound method> = bound_method %.loc27_37.8, %Optional.Get.specific_fn
// CHECK:STDOUT:     %.loc27_37.14: ref %OverloadValueType = temporary_storage
// CHECK:STDOUT:     %.loc27_37.15: %Optional.cb9 = acquire_value %.loc27_37.8
// CHECK:STDOUT:     %Optional.Get.call: init %OverloadValueType to %.loc27_37.14 = call %bound_method.loc27_37.6(%.loc27_37.15)
// CHECK:STDOUT:     %.loc27_37.16: ref %OverloadValueType = temporary %.loc27_37.14, %Optional.Get.call
// CHECK:STDOUT:     %.loc27_37.17: %OverloadValueType = acquire_value %.loc27_37.16
// CHECK:STDOUT:     %.loc27_14: type = splice_block %OverloadValueType.ref.loc27 [concrete = constants.%OverloadValueType] {
// CHECK:STDOUT:       %Cpp.ref.loc27: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:       %OverloadValueType.ref.loc27: type = name_ref OverloadValueType, imports.%OverloadValueType.decl [concrete = constants.%OverloadValueType]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %_: %OverloadValueType = wrapper_binding _, %.loc27_37.17
// CHECK:STDOUT:     %Destroy.WithSelf.SelfDestruct.bound.loc27_37.1: <bound method> = bound_method %.loc27_37.16, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.2
// CHECK:STDOUT:     %Destroy.WithSelf.SelfDestruct.call.loc27_37.1: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc27_37.1(%.loc27_37.16)
// CHECK:STDOUT:     %Destroy.WithSelf.SelfDestruct.bound.loc27_37.2: <bound method> = bound_method %.loc27_37.8, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.8
// CHECK:STDOUT:     %Destroy.WithSelf.SelfDestruct.call.loc27_37.2: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc27_37.2(%.loc27_37.8)
// CHECK:STDOUT:     br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.done:
// CHECK:STDOUT:     %Destroy.WithSelf.SelfDestruct.bound.loc27_37.3: <bound method> = bound_method %.loc27_37.8, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.8
// CHECK:STDOUT:     %Destroy.WithSelf.SelfDestruct.call.loc27_37.3: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc27_37.3(%.loc27_37.8)
// CHECK:STDOUT:     %impl.elem2.loc27_37.2: @TestIterate.%.loc27_37.22 (%.73a) = impl_witness_access constants.%Destroy.lookup_impl_witness.a78, element2 [symbolic = %impl.elem2.loc27_37.4 (constants.%impl.elem2.01e)]
// CHECK:STDOUT:     %bound_method.loc27_37.7: <bound method> = bound_method %var, %impl.elem2.loc27_37.2
// CHECK:STDOUT:     %.loc27_37.18: %Destroy.type = converted constants.%as_type.400, constants.%impl.elem1.0e3 [symbolic = %impl.elem1 (constants.%impl.elem1.0e3)]
// CHECK:STDOUT:     %.loc27_37.19: %Destroy.type = converted constants.%as_type.400, constants.%impl.elem1.0e3 [symbolic = %impl.elem1 (constants.%impl.elem1.0e3)]
// CHECK:STDOUT:     %specific_impl_fn.loc27_37.3: <specific function> = specific_impl_function %impl.elem2.loc27_37.2, @Destroy.WithSelf.SelfDestruct.1(constants.%impl.elem1.0e3) [symbolic = %specific_impl_fn.loc27_37.6 (constants.%specific_impl_fn.dac)]
// CHECK:STDOUT:     %bound_method.loc27_37.8: <bound method> = bound_method %var, %specific_impl_fn.loc27_37.3
// CHECK:STDOUT:     %Destroy.WithSelf.SelfDestruct.call.loc27_37.4: init %empty_tuple.type = call %bound_method.loc27_37.8(%var)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc27_37.1(%self.param: ref %.8a2) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc27_37.1(%self.param: ref %.8a2) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc27_37.1(%self.param: ref %.8a2) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.2(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.1(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc27_37.2(%self.param: ref %MaybeUnformed.d16) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc27_37.2(%self.param: ref %MaybeUnformed.d16) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc27_37.2(%self.param: ref %MaybeUnformed.d16) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.3(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.2(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc27_37.3(%self.param: ref bool) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc27_37.3(%self.param: ref bool) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc27_37.3(%self.param: ref bool) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.4(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.3(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc27_37.4(%self.param: ref %struct_type.value.has_value.35f) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc27_37.1: ref %MaybeUnformed.d16 = struct_access %self.param, element0
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc27_37.1: <bound method> = bound_method %.loc27_37.1, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.4
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc27_37.1: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc27_37.1(%.loc27_37.1)
// CHECK:STDOUT:   %.loc27_37.2: ref bool = struct_access %self.param, element1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc27_37.2: <bound method> = bound_method %.loc27_37.2, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.5
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc27_37.2: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc27_37.2(%.loc27_37.2)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc27_37.4(%self.param: ref %struct_type.value.has_value.35f) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc27_37.4(%self.param: ref %struct_type.value.has_value.35f) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.5(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.4(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc27_37.5(%self.param: ref %DefaultOptionalStorage.fbc) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc27_37.5(%self.param: ref %DefaultOptionalStorage.fbc) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc27_37.5(%self.param: ref %DefaultOptionalStorage.fbc) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.6(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.5(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc27_37.6(%self.param: ref %Optional.cb9) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc27_37.6(%self.param: ref %Optional.cb9) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc27_37.6(%self.param: ref %Optional.cb9) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.7(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.6(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestRangeFor(%r.param: ref %Range) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %_.patt: %pattern_type.804 = value_binding_pattern _ [concrete = constants.%_.patt.a10]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %r.ref: ref %Range = name_ref r, %r
// CHECK:STDOUT:   %impl.elem2: %.b72 = impl_witness_access constants.%Iterate.impl_witness.24f, element2 [concrete = constants.%T.as_type.as.Iterate.impl.NewCursor.8fd]
// CHECK:STDOUT:   %bound_method.loc37_37.1: <bound method> = bound_method %r.ref, %impl.elem2
// CHECK:STDOUT:   %facet_value.loc37_37.1: %CppRange_where.type.d682d6.1 = facet_value constants.%Range, (constants.%custom_witness.1bfa2a.1, constants.%custom_witness.d0da43.1, constants.%custom_witness.090, constants.%custom_witness.eaa, constants.%custom_witness.6bd, constants.%custom_witness.716, constants.%custom_witness.617) [concrete = constants.%facet_value.670]
// CHECK:STDOUT:   %.loc37_37.1: %CppRange_where.type.d682d6.1 = converted constants.%Range, %facet_value.loc37_37.1 [concrete = constants.%facet_value.670]
// CHECK:STDOUT:   %facet_value.loc37_37.2: %CppRange_where.type.d682d6.1 = facet_value constants.%Range, (constants.%custom_witness.1bfa2a.1, constants.%custom_witness.d0da43.1, constants.%custom_witness.090, constants.%custom_witness.eaa, constants.%custom_witness.6bd, constants.%custom_witness.716, constants.%custom_witness.617) [concrete = constants.%facet_value.670]
// CHECK:STDOUT:   %.loc37_37.2: %CppRange_where.type.d682d6.1 = converted constants.%Range, %facet_value.loc37_37.2 [concrete = constants.%facet_value.670]
// CHECK:STDOUT:   %specific_fn.loc37_37.1: <specific function> = specific_function %impl.elem2, @T.as_type.as.Iterate.impl.NewCursor(constants.%facet_value.670) [concrete = constants.%T.as_type.as.Iterate.impl.NewCursor.specific_fn]
// CHECK:STDOUT:   %bound_method.loc37_37.2: <bound method> = bound_method %r.ref, %specific_fn.loc37_37.1
// CHECK:STDOUT:   %var: ref %tuple.type.9ef = var_storage invalid
// CHECK:STDOUT:   %.loc37_36.1: %Range = acquire_value %r.ref
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.call: init %tuple.type.9ef to %var = call %bound_method.loc37_37.2(%.loc37_36.1)
// CHECK:STDOUT:   assign %var, %T.as_type.as.Iterate.impl.NewCursor.call
// CHECK:STDOUT:   br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT: !for.next:
// CHECK:STDOUT:   %addr: %ptr.b18 = addr_of %var
// CHECK:STDOUT:   %impl.elem3: %.0c0 = impl_witness_access constants.%Iterate.impl_witness.24f, element3 [concrete = constants.%T.as_type.as.Iterate.impl.Next.4cb]
// CHECK:STDOUT:   %bound_method.loc37_37.3: <bound method> = bound_method %r.ref, %impl.elem3
// CHECK:STDOUT:   %facet_value.loc37_37.3: %CppRange_where.type.d682d6.1 = facet_value constants.%Range, (constants.%custom_witness.1bfa2a.1, constants.%custom_witness.d0da43.1, constants.%custom_witness.090, constants.%custom_witness.eaa, constants.%custom_witness.6bd, constants.%custom_witness.716, constants.%custom_witness.617) [concrete = constants.%facet_value.670]
// CHECK:STDOUT:   %.loc37_37.3: %CppRange_where.type.d682d6.1 = converted constants.%Range, %facet_value.loc37_37.3 [concrete = constants.%facet_value.670]
// CHECK:STDOUT:   %facet_value.loc37_37.4: %CppRange_where.type.d682d6.1 = facet_value constants.%Range, (constants.%custom_witness.1bfa2a.1, constants.%custom_witness.d0da43.1, constants.%custom_witness.090, constants.%custom_witness.eaa, constants.%custom_witness.6bd, constants.%custom_witness.716, constants.%custom_witness.617) [concrete = constants.%facet_value.670]
// CHECK:STDOUT:   %.loc37_37.4: %CppRange_where.type.d682d6.1 = converted constants.%Range, %facet_value.loc37_37.4 [concrete = constants.%facet_value.670]
// CHECK:STDOUT:   %specific_fn.loc37_37.2: <specific function> = specific_function %impl.elem3, @T.as_type.as.Iterate.impl.Next(constants.%facet_value.670) [concrete = constants.%T.as_type.as.Iterate.impl.Next.specific_fn]
// CHECK:STDOUT:   %bound_method.loc37_37.4: <bound method> = bound_method %r.ref, %specific_fn.loc37_37.2
// CHECK:STDOUT:   %.loc37_37.5: ref %Optional.cb9 = temporary_storage
// CHECK:STDOUT:   %.loc37_36.2: %Range = acquire_value %r.ref
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.call: init %Optional.cb9 to %.loc37_37.5 = call %bound_method.loc37_37.4(%.loc37_36.2, %addr)
// CHECK:STDOUT:   %.loc37_37.6: ref %Optional.cb9 = temporary %.loc37_37.5, %T.as_type.as.Iterate.impl.Next.call
// CHECK:STDOUT:   %.loc37_37.7: %Optional.HasValue.type.800 = specific_constant imports.%Core.import_ref.bb6, @Optional(constants.%OptionalStorage.facet.50c) [concrete = constants.%Optional.HasValue.ab3]
// CHECK:STDOUT:   %HasValue.ref: %Optional.HasValue.type.800 = name_ref HasValue, %.loc37_37.7 [concrete = constants.%Optional.HasValue.ab3]
// CHECK:STDOUT:   %Optional.HasValue.bound: <bound method> = bound_method %.loc37_37.6, %HasValue.ref
// CHECK:STDOUT:   %Optional.HasValue.specific_fn: <specific function> = specific_function %HasValue.ref, @Optional.HasValue(constants.%OptionalStorage.facet.50c) [concrete = constants.%Optional.HasValue.specific_fn.730]
// CHECK:STDOUT:   %bound_method.loc37_37.5: <bound method> = bound_method %.loc37_37.6, %Optional.HasValue.specific_fn
// CHECK:STDOUT:   %.loc37_37.8: %Optional.cb9 = acquire_value %.loc37_37.6
// CHECK:STDOUT:   %Optional.HasValue.call: init bool = call %bound_method.loc37_37.5(%.loc37_37.8)
// CHECK:STDOUT:   %.loc37_37.9: bool = value_of_initializer %Optional.HasValue.call
// CHECK:STDOUT:   %.loc37_37.10: bool = converted %Optional.HasValue.call, %.loc37_37.9
// CHECK:STDOUT:   if %.loc37_37.10 br !for.body else br !for.done
// CHECK:STDOUT:
// CHECK:STDOUT: !for.body:
// CHECK:STDOUT:   %.loc37_37.11: %Optional.Get.type.466 = specific_constant imports.%Core.import_ref.dab, @Optional(constants.%OptionalStorage.facet.50c) [concrete = constants.%Optional.Get.4ea]
// CHECK:STDOUT:   %Get.ref: %Optional.Get.type.466 = name_ref Get, %.loc37_37.11 [concrete = constants.%Optional.Get.4ea]
// CHECK:STDOUT:   %Optional.Get.bound: <bound method> = bound_method %.loc37_37.6, %Get.ref
// CHECK:STDOUT:   %Optional.Get.specific_fn: <specific function> = specific_function %Get.ref, @Optional.Get(constants.%OptionalStorage.facet.50c) [concrete = constants.%Optional.Get.specific_fn.f34]
// CHECK:STDOUT:   %bound_method.loc37_37.6: <bound method> = bound_method %.loc37_37.6, %Optional.Get.specific_fn
// CHECK:STDOUT:   %.loc37_37.12: ref %OverloadValueType = temporary_storage
// CHECK:STDOUT:   %.loc37_37.13: %Optional.cb9 = acquire_value %.loc37_37.6
// CHECK:STDOUT:   %Optional.Get.call: init %OverloadValueType to %.loc37_37.12 = call %bound_method.loc37_37.6(%.loc37_37.13)
// CHECK:STDOUT:   %.loc37_37.14: ref %OverloadValueType = temporary %.loc37_37.12, %Optional.Get.call
// CHECK:STDOUT:   %.loc37_37.15: %OverloadValueType = acquire_value %.loc37_37.14
// CHECK:STDOUT:   %.loc37_14: type = splice_block %OverloadValueType.ref [concrete = constants.%OverloadValueType] {
// CHECK:STDOUT:     %Cpp.ref.loc37: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %OverloadValueType.ref: type = name_ref OverloadValueType, imports.%OverloadValueType.decl [concrete = constants.%OverloadValueType]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %_: %OverloadValueType = wrapper_binding _, %.loc37_37.15
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc37_37.1: <bound method> = bound_method %.loc37_37.14, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.2
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc37_37.1: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc37_37.1(%.loc37_37.14)
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc37_37.2: <bound method> = bound_method %.loc37_37.6, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.8
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc37_37.2: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc37_37.2(%.loc37_37.6)
// CHECK:STDOUT:   br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT: !for.done:
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc37_37.3: <bound method> = bound_method %.loc37_37.6, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.8
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc37_37.3: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc37_37.3(%.loc37_37.6)
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc37_37.4: <bound method> = bound_method %var, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.10
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc37_37.4: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc37_37.4(%var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestIterate(constants.%R) {
// CHECK:STDOUT:   %R.patt.loc25_17.2 => constants.%R.patt
// CHECK:STDOUT:   %R.loc25_17.1 => constants.%R
// CHECK:STDOUT:   %R.as_type.loc25_83.1 => constants.%R.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.a47
// CHECK:STDOUT:   %r.patt.loc25_81.2 => constants.%r.patt.24d
// CHECK:STDOUT:   %r.param_patt.loc25_76.2 => constants.%r.param_patt.d5b
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestIterate(constants.%facet_value.483) {
// CHECK:STDOUT:   %R.patt.loc25_17.2 => constants.%R.patt
// CHECK:STDOUT:   %R.loc25_17.1 => constants.%facet_value.483
// CHECK:STDOUT:   %R.as_type.loc25_83.1 => constants.%Range
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.23f
// CHECK:STDOUT:   %r.patt.loc25_81.2 => constants.%r.patt.b9f
// CHECK:STDOUT:   %r.param_patt.loc25_76.2 => constants.%r.param_patt.466
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc25 => constants.%complete_type.357
// CHECK:STDOUT:   %Iterate.lookup_impl_witness => constants.%Iterate.impl_witness.24f
// CHECK:STDOUT:   %Iterate.facet.loc27_37.5 => constants.%Iterate.facet.726
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type => constants.%Iterate.WithSelf.NewCursor.type.292
// CHECK:STDOUT:   %.loc27_37.20 => constants.%.b72
// CHECK:STDOUT:   %impl.elem2.loc27_37.3 => constants.%T.as_type.as.Iterate.impl.NewCursor.8fd
// CHECK:STDOUT:   %specific_impl_fn.loc27_37.4 => constants.%T.as_type.as.Iterate.impl.NewCursor.specific_fn
// CHECK:STDOUT:   %impl.elem1 => constants.%Destroy.facet.c69
// CHECK:STDOUT:   %as_type => constants.%tuple.type.9ef
// CHECK:STDOUT:   %require_complete.1 => constants.%complete_type.e35
// CHECK:STDOUT:   %ptr => constants.%ptr.b18
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type => constants.%Iterate.WithSelf.Next.type.4f7
// CHECK:STDOUT:   %.loc27_37.21 => constants.%.0c0
// CHECK:STDOUT:   %impl.elem3.loc27_37.2 => constants.%T.as_type.as.Iterate.impl.Next.4cb
// CHECK:STDOUT:   %specific_impl_fn.loc27_37.5 => constants.%T.as_type.as.Iterate.impl.Next.specific_fn
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type => constants.%Destroy.WithSelf.SelfDestruct.type.6ef
// CHECK:STDOUT:   %.loc27_37.22 => constants.%.a39
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.f8f19d.8
// CHECK:STDOUT:   %impl.elem2.loc27_37.4 => constants.%Destroy.WithSelf.SelfDestruct.db3fdb.10
// CHECK:STDOUT:   %specific_impl_fn.loc27_37.6 => constants.%Destroy.WithSelf.SelfDestruct.db3fdb.10
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- adl_return_same_types.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %Iterate.type: type = facet_type <@Iterate> [concrete]
// CHECK:STDOUT:   %OptionalStorage.type: type = facet_type <@OptionalStorage> [concrete]
// CHECK:STDOUT:   %pattern_type.831: type = pattern_type bool [concrete]
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
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.3: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc20_53.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.3: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.3 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.3: type = fn_type @Destroy.WithSelf.Op.loc20_53.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.3: %Destroy.WithSelf.Op.type.ef016f.3 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.3: type = fn_type @Destroy.WithSelf.SelfDestruct.loc20_53.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.3: %Destroy.WithSelf.SelfDestruct.type.fbceb5.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.f8f19d.3: <witness> = custom_witness (%Destroy.WithSelf.Op.403171.3, %Destroy.WithSelf.SubobjectDestroy.d01daf.3, %Destroy.WithSelf.SelfDestruct.db3fdb.3), @Destroy [concrete]
// CHECK:STDOUT:   %Float.as.Copy.impl.Op.type.ff3: type = fn_type @Float.as.Copy.impl.Op, @Float.as.Copy.impl(%N.fe9) [symbolic]
// CHECK:STDOUT:   %Float.as.Copy.impl.Op.d93: %Float.as.Copy.impl.Op.type.ff3 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.0d6: <witness> = impl_witness imports.%Copy.impl_witness_table.8dd, @Float.as.Copy.impl(%int_64) [concrete]
// CHECK:STDOUT:   %facet_value.b6e: %facet_type.f48 = facet_value %f64.dc1, (%custom_witness.f8f19d.3, %Copy.impl_witness.0d6) [concrete]
// CHECK:STDOUT:   %Iterate.lookup_impl_witness.1d8: <witness> = lookup_impl_witness %.Self.500, @Iterate [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.294: %facet_type.f48 = impl_witness_access %Iterate.lookup_impl_witness.1d8, element0 [symbolic_self]
// CHECK:STDOUT:   %Iterate_where.type.028: type = facet_type <@Iterate where %impl.elem0.294 = %facet_value.b6e> [concrete]
// CHECK:STDOUT:   %pattern_type.522: type = pattern_type %Iterate_where.type.028 [concrete]
// CHECK:STDOUT:   %R.patt: %pattern_type.522 = symbolic_binding_pattern R, 0 [symbolic]
// CHECK:STDOUT:   %R: %Iterate_where.type.028 = symbolic_binding R, 0 [symbolic]
// CHECK:STDOUT:   %R.as_type: type = facet_access_type %R [symbolic]
// CHECK:STDOUT:   %pattern_type.6f7: type = pattern_type %R.as_type [symbolic]
// CHECK:STDOUT:   %r.param_patt.19a: %pattern_type.6f7 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %r.patt.f97: %pattern_type.6f7 = wrapper_binding_pattern r, %r.param_patt.19a [symbolic]
// CHECK:STDOUT:   %i.patt: %pattern_type.fb7 = value_binding_pattern i [concrete]
// CHECK:STDOUT:   %Iterate.lookup_impl_witness.9e0: <witness> = lookup_impl_witness %R, @Iterate [symbolic]
// CHECK:STDOUT:   %Iterate.facet.f88: %Iterate.type = facet_value %R.as_type, (%Iterate.lookup_impl_witness.9e0) [symbolic]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.3d0: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.f88) [symbolic]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.160: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.f88) [symbolic]
// CHECK:STDOUT:   %.afc: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.3d0, %Iterate.facet.f88 [symbolic]
// CHECK:STDOUT:   %impl.elem2.895: %.afc = impl_witness_access %Iterate.lookup_impl_witness.9e0, element2 [symbolic]
// CHECK:STDOUT:   %impl.elem1.0cf: %Destroy.type = impl_witness_access %Iterate.lookup_impl_witness.9e0, element1 [symbolic]
// CHECK:STDOUT:   %as_type.ce3: type = facet_access_type %impl.elem1.0cf [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.999: <specific function> = specific_impl_function %impl.elem2.895, @Iterate.WithSelf.NewCursor(%Iterate.facet.f88) [symbolic]
// CHECK:STDOUT:   %ptr.44a: type = ptr_type %as_type.ce3 [symbolic]
// CHECK:STDOUT:   %.b00: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.160, %Iterate.facet.f88 [symbolic]
// CHECK:STDOUT:   %impl.elem3.c16: %.b00 = impl_witness_access %Iterate.lookup_impl_witness.9e0, element3 [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.3e2: type = class_type @DefaultOptionalStorage, @DefaultOptionalStorage(%facet_value.b6e) [concrete]
// CHECK:STDOUT:   %OptionalStorage.impl_witness.4b3: <witness> = impl_witness imports.%OptionalStorage.impl_witness_table.460, @T.as_type.as.OptionalStorage.impl(%facet_value.b6e) [concrete]
// CHECK:STDOUT:   %OptionalStorage.facet.405: %OptionalStorage.type = facet_value %f64.dc1, (%OptionalStorage.impl_witness.4b3) [concrete]
// CHECK:STDOUT:   %Optional.049: type = class_type @Optional, @Optional(%OptionalStorage.facet.405) [concrete]
// CHECK:STDOUT:   %pattern_type.449: type = pattern_type %Optional.049 [concrete]
// CHECK:STDOUT:   %specific_impl_fn.b6b: <specific function> = specific_impl_function %impl.elem3.c16, @Iterate.WithSelf.Next(%Iterate.facet.f88) [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.type.794: type = fn_type @Optional.HasValue, @Optional(%OptionalStorage.facet.405) [concrete]
// CHECK:STDOUT:   %Optional.HasValue.776: %Optional.HasValue.type.794 = struct_value () [concrete]
// CHECK:STDOUT:   %Optional.Get.type.e6c: type = fn_type @Optional.Get, @Optional(%OptionalStorage.facet.405) [concrete]
// CHECK:STDOUT:   %Optional.Get.cb6: %Optional.Get.type.e6c = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.facet.d87: %Destroy.type = facet_value %f64.dc1, (%custom_witness.f8f19d.3) [concrete]
// CHECK:STDOUT:   %MaybeUnformed.8ff: type = class_type @MaybeUnformed, @MaybeUnformed(%Destroy.facet.d87) [concrete]
// CHECK:STDOUT:   %.cc0: type = maybe_unformed_type %f64.dc1 [concrete]
// CHECK:STDOUT:   %struct_type.value.has_value.58e: type = struct_type {.value: %MaybeUnformed.8ff, .has_value: bool} [concrete]
// CHECK:STDOUT:   %Optional.HasValue.specific_fn.9fc: <specific function> = specific_function %Optional.HasValue.776, @Optional.HasValue(%OptionalStorage.facet.405) [concrete]
// CHECK:STDOUT:   %Optional.Get.specific_fn.e69: <specific function> = specific_function %Optional.Get.cb6, @Optional.Get(%OptionalStorage.facet.405) [concrete]
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
// CHECK:STDOUT:   %pattern_type.1fb: type = pattern_type %.cc0 [concrete]
// CHECK:STDOUT:   %self.param_patt.4fd: %pattern_type.1fb = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.69f: %pattern_type.1fb = wrapper_binding_pattern self, %self.param_patt.4fd [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.4: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc24_19.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.4: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.4 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.4: type = fn_type @Destroy.WithSelf.Op.loc24_19.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.4: %Destroy.WithSelf.Op.type.ef016f.4 = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.80d: type = pattern_type %MaybeUnformed.8ff [concrete]
// CHECK:STDOUT:   %self.param_patt.70c: %pattern_type.80d = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.120: %pattern_type.80d = wrapper_binding_pattern self, %self.param_patt.70c [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.5: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc24_19.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.5: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.5 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.5: type = fn_type @Destroy.WithSelf.Op.loc24_19.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.5: %Destroy.WithSelf.Op.type.ef016f.5 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.5: type = fn_type @Destroy.WithSelf.SelfDestruct.loc24_19.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.5: %Destroy.WithSelf.SelfDestruct.type.fbceb5.5 = struct_value () [concrete]
// CHECK:STDOUT:   %self.param_patt.40d: %pattern_type.831 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.560: %pattern_type.831 = wrapper_binding_pattern self, %self.param_patt.40d [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.6: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc24_19.3 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.6: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.6 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.6: type = fn_type @Destroy.WithSelf.Op.loc24_19.3 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.6: %Destroy.WithSelf.Op.type.ef016f.6 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.6: type = fn_type @Destroy.WithSelf.SelfDestruct.loc24_19.3 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.6: %Destroy.WithSelf.SelfDestruct.type.fbceb5.6 = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.00a: type = pattern_type %struct_type.value.has_value.58e [concrete]
// CHECK:STDOUT:   %self.param_patt.96d: %pattern_type.00a = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.602: %pattern_type.00a = wrapper_binding_pattern self, %self.param_patt.96d [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.7: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc24_19.4 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.7: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.7 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.7: type = fn_type @Destroy.WithSelf.Op.loc24_19.4 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.7: %Destroy.WithSelf.Op.type.ef016f.7 = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.406: type = pattern_type %DefaultOptionalStorage.3e2 [concrete]
// CHECK:STDOUT:   %self.param_patt.73b: %pattern_type.406 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.bd7: %pattern_type.406 = wrapper_binding_pattern self, %self.param_patt.73b [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.8: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc24_19.5 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.8: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.8 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.8: type = fn_type @Destroy.WithSelf.Op.loc24_19.5 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.8: %Destroy.WithSelf.Op.type.ef016f.8 = struct_value () [concrete]
// CHECK:STDOUT:   %self.param_patt.a67: %pattern_type.449 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.521: %pattern_type.449 = wrapper_binding_pattern self, %self.param_patt.a67 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.9: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc24_19.6 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.9: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.9 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.9: type = fn_type @Destroy.WithSelf.Op.loc24_19.6 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.9: %Destroy.WithSelf.Op.type.ef016f.9 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.9: type = fn_type @Destroy.WithSelf.SelfDestruct.loc24_19.6 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.9: %Destroy.WithSelf.SelfDestruct.type.fbceb5.9 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.068: <witness> = lookup_impl_witness %impl.elem1.0cf, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.b85: type = fn_type @Destroy.WithSelf.SelfDestruct.1, @Destroy.WithSelf(%impl.elem1.0cf) [symbolic]
// CHECK:STDOUT:   %.056: type = fn_type_with_self_type %Destroy.WithSelf.SelfDestruct.type.b85, %impl.elem1.0cf [symbolic]
// CHECK:STDOUT:   %impl.elem2.5f8: %.056 = impl_witness_access %Destroy.lookup_impl_witness.068, element2 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.b61: <specific function> = specific_impl_function %impl.elem2.5f8, @Destroy.WithSelf.SelfDestruct.1(%impl.elem1.0cf) [symbolic]
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
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.10: type = fn_type @Destroy.WithSelf.Op.3 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.10: %Destroy.WithSelf.Op.type.ef016f.10 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.10: type = fn_type @Destroy.WithSelf.SelfDestruct.3 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.10: %Destroy.WithSelf.SelfDestruct.type.fbceb5.10 = struct_value () [concrete]
// CHECK:STDOUT:   %Iterator.SubobjectDestroy.type: type = fn_type @Iterator.SubobjectDestroy [concrete]
// CHECK:STDOUT:   %Iterator.SubobjectDestroy: %Iterator.SubobjectDestroy.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b76: <witness> = custom_witness (%Destroy.WithSelf.Op.403171.10, %Iterator.SubobjectDestroy, %Destroy.WithSelf.SelfDestruct.db3fdb.10), @Destroy [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.ac7e09.2: type = fn_type @Iterator.Op.2 [concrete]
// CHECK:STDOUT:   %Iterator.Op.583fda.2: %Iterator.Op.type.ac7e09.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.393: <witness> = custom_witness (%Iterator.Op.583fda.2), @Inc [concrete]
// CHECK:STDOUT:   %Equal.type: type = fn_type @Equal [concrete]
// CHECK:STDOUT:   %Equal: %Equal.type = struct_value () [concrete]
// CHECK:STDOUT:   %NotEqual.type: type = fn_type @NotEqual [concrete]
// CHECK:STDOUT:   %NotEqual: %NotEqual.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.821: <witness> = custom_witness (%Equal, %NotEqual), @EqWith, @EqWith(%Iterator) [concrete]
// CHECK:STDOUT:   %facet_value.e81: %CppRange_where.type.d682d6.1 = facet_value %ConstRange, (%custom_witness.f8f19d.3, %Copy.impl_witness.0d6, %custom_witness.5d5, %custom_witness.b76, %custom_witness.994, %custom_witness.393, %custom_witness.821) [concrete]
// CHECK:STDOUT:   %tuple.type.005: type = tuple_type (%Iterator, %Iterator) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.10: type = fn_type @Destroy.WithSelf.SubobjectDestroy.3 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.10: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.10 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.11: type = fn_type @Destroy.WithSelf.Op.4 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.11: %Destroy.WithSelf.Op.type.ef016f.11 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.11: type = fn_type @Destroy.WithSelf.SelfDestruct.4 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.11: %Destroy.WithSelf.SelfDestruct.type.fbceb5.11 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.f8f19d.10: <witness> = custom_witness (%Destroy.WithSelf.Op.403171.11, %Destroy.WithSelf.SubobjectDestroy.d01daf.10, %Destroy.WithSelf.SelfDestruct.db3fdb.11), @Destroy [concrete]
// CHECK:STDOUT:   %Destroy.facet.362: %Destroy.type = facet_value %tuple.type.005, (%custom_witness.f8f19d.10) [concrete]
// CHECK:STDOUT:   %Iterate.impl_witness.2d1: <witness> = impl_witness imports.%Iterate.impl_witness_table.121, @T.as_type.as.Iterate.impl(%facet_value.e81) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.type.2c9: type = fn_type @T.as_type.as.Iterate.impl.NewCursor, @T.as_type.as.Iterate.impl(%facet_value.e81) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.c9a: %T.as_type.as.Iterate.impl.NewCursor.type.2c9 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.type.08d: type = fn_type @T.as_type.as.Iterate.impl.Next, @T.as_type.as.Iterate.impl(%facet_value.e81) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.751: %T.as_type.as.Iterate.impl.Next.type.08d = struct_value () [concrete]
// CHECK:STDOUT:   %Iterate.facet.e86: %Iterate.type = facet_value %ConstRange, (%Iterate.impl_witness.2d1) [concrete]
// CHECK:STDOUT:   %facet_value.58e: %Iterate_where.type.028 = facet_value %ConstRange, (%Iterate.impl_witness.2d1) [concrete]
// CHECK:STDOUT:   %r.param_patt.a73: %pattern_type.025 = value_param_pattern [concrete]
// CHECK:STDOUT:   %r.patt.0b5: %pattern_type.025 = wrapper_binding_pattern r, %r.param_patt.a73 [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.bcd: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.e86) [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.44e: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.e86) [concrete]
// CHECK:STDOUT:   %.3f0: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.bcd, %Iterate.facet.e86 [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.specific_fn: <specific function> = specific_function %T.as_type.as.Iterate.impl.NewCursor.c9a, @T.as_type.as.Iterate.impl.NewCursor(%facet_value.e81) [concrete]
// CHECK:STDOUT:   %ptr.d1a: type = ptr_type %tuple.type.005 [concrete]
// CHECK:STDOUT:   %.343: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.44e, %Iterate.facet.e86 [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.specific_fn: <specific function> = specific_function %T.as_type.as.Iterate.impl.Next.751, @T.as_type.as.Iterate.impl.Next(%facet_value.e81) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.873: type = fn_type @Destroy.WithSelf.SelfDestruct.1, @Destroy.WithSelf(%Destroy.facet.362) [concrete]
// CHECK:STDOUT:   %.9a7: type = fn_type_with_self_type %Destroy.WithSelf.SelfDestruct.type.873, %Destroy.facet.362 [concrete]
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
// CHECK:STDOUT: generated {
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.3: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.4 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc24_19.1 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.4] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.1fb = ref_param_pattern [concrete = constants.%self.param_patt.4fd]
// CHECK:STDOUT:     %self.patt: %pattern_type.1fb = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.69f]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %.cc0 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %.cc0 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.3: %Destroy.WithSelf.Op.type.ef016f.4 = fn_decl @Destroy.WithSelf.Op.loc24_19.1 [concrete = constants.%Destroy.WithSelf.Op.403171.4] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.1fb = ref_param_pattern [concrete = constants.%self.param_patt.4fd]
// CHECK:STDOUT:     %self.patt: %pattern_type.1fb = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.69f]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %.cc0 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %.cc0 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.4: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.5 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc24_19.2 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.5] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.80d = ref_param_pattern [concrete = constants.%self.param_patt.70c]
// CHECK:STDOUT:     %self.patt: %pattern_type.80d = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.120]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %MaybeUnformed.8ff = ref_param call_param0
// CHECK:STDOUT:     %self: ref %MaybeUnformed.8ff = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.4: %Destroy.WithSelf.Op.type.ef016f.5 = fn_decl @Destroy.WithSelf.Op.loc24_19.2 [concrete = constants.%Destroy.WithSelf.Op.403171.5] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.80d = ref_param_pattern [concrete = constants.%self.param_patt.70c]
// CHECK:STDOUT:     %self.patt: %pattern_type.80d = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.120]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %MaybeUnformed.8ff = ref_param call_param0
// CHECK:STDOUT:     %self: ref %MaybeUnformed.8ff = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.5: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.6 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc24_19.3 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.6] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.831 = ref_param_pattern [concrete = constants.%self.param_patt.40d]
// CHECK:STDOUT:     %self.patt: %pattern_type.831 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.560]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref bool = ref_param call_param0
// CHECK:STDOUT:     %self: ref bool = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.5: %Destroy.WithSelf.Op.type.ef016f.6 = fn_decl @Destroy.WithSelf.Op.loc24_19.3 [concrete = constants.%Destroy.WithSelf.Op.403171.6] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.831 = ref_param_pattern [concrete = constants.%self.param_patt.40d]
// CHECK:STDOUT:     %self.patt: %pattern_type.831 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.560]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref bool = ref_param call_param0
// CHECK:STDOUT:     %self: ref bool = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.6: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.7 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc24_19.4 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.7] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.00a = ref_param_pattern [concrete = constants.%self.param_patt.96d]
// CHECK:STDOUT:     %self.patt: %pattern_type.00a = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.602]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %struct_type.value.has_value.58e = ref_param call_param0
// CHECK:STDOUT:     %self: ref %struct_type.value.has_value.58e = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.6: %Destroy.WithSelf.Op.type.ef016f.7 = fn_decl @Destroy.WithSelf.Op.loc24_19.4 [concrete = constants.%Destroy.WithSelf.Op.403171.7] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.00a = ref_param_pattern [concrete = constants.%self.param_patt.96d]
// CHECK:STDOUT:     %self.patt: %pattern_type.00a = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.602]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %struct_type.value.has_value.58e = ref_param call_param0
// CHECK:STDOUT:     %self: ref %struct_type.value.has_value.58e = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.7: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.8 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc24_19.5 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.8] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.406 = ref_param_pattern [concrete = constants.%self.param_patt.73b]
// CHECK:STDOUT:     %self.patt: %pattern_type.406 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.bd7]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %DefaultOptionalStorage.3e2 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %DefaultOptionalStorage.3e2 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.7: %Destroy.WithSelf.Op.type.ef016f.8 = fn_decl @Destroy.WithSelf.Op.loc24_19.5 [concrete = constants.%Destroy.WithSelf.Op.403171.8] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.406 = ref_param_pattern [concrete = constants.%self.param_patt.73b]
// CHECK:STDOUT:     %self.patt: %pattern_type.406 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.bd7]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %DefaultOptionalStorage.3e2 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %DefaultOptionalStorage.3e2 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.8: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.9 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc24_19.6 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.9] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.449 = ref_param_pattern [concrete = constants.%self.param_patt.a67]
// CHECK:STDOUT:     %self.patt: %pattern_type.449 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.521]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %Optional.049 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %Optional.049 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.8: %Destroy.WithSelf.Op.type.ef016f.9 = fn_decl @Destroy.WithSelf.Op.loc24_19.6 [concrete = constants.%Destroy.WithSelf.Op.403171.9] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.449 = ref_param_pattern [concrete = constants.%self.param_patt.a67]
// CHECK:STDOUT:     %self.patt: %pattern_type.449 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.521]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %Optional.049 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %Optional.049 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @TestIterate(%R.loc20_17.2: %Iterate_where.type.028) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Iterate.lookup_impl_witness: <witness> = lookup_impl_witness %R.loc20_17.1, @Iterate [symbolic = %Iterate.lookup_impl_witness (constants.%Iterate.lookup_impl_witness.9e0)]
// CHECK:STDOUT:   %Iterate.facet.loc24_19.5: %Iterate.type = facet_value %R.as_type.loc20_61.1, (%Iterate.lookup_impl_witness) [symbolic = %Iterate.facet.loc24_19.5 (constants.%Iterate.facet.f88)]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.loc24_19.5) [symbolic = %Iterate.WithSelf.NewCursor.type (constants.%Iterate.WithSelf.NewCursor.type.3d0)]
// CHECK:STDOUT:   %.loc24_19.19: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type, %Iterate.facet.loc24_19.5 [symbolic = %.loc24_19.19 (constants.%.afc)]
// CHECK:STDOUT:   %impl.elem2.loc24_19.3: @TestIterate.%.loc24_19.19 (%.afc) = impl_witness_access %Iterate.lookup_impl_witness, element2 [symbolic = %impl.elem2.loc24_19.3 (constants.%impl.elem2.895)]
// CHECK:STDOUT:   %specific_impl_fn.loc24_19.4: <specific function> = specific_impl_function %impl.elem2.loc24_19.3, @Iterate.WithSelf.NewCursor(%Iterate.facet.loc24_19.5) [symbolic = %specific_impl_fn.loc24_19.4 (constants.%specific_impl_fn.999)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %ptr: type = ptr_type %as_type [symbolic = %ptr (constants.%ptr.44a)]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.loc24_19.5) [symbolic = %Iterate.WithSelf.Next.type (constants.%Iterate.WithSelf.Next.type.160)]
// CHECK:STDOUT:   %.loc24_19.20: type = fn_type_with_self_type %Iterate.WithSelf.Next.type, %Iterate.facet.loc24_19.5 [symbolic = %.loc24_19.20 (constants.%.b00)]
// CHECK:STDOUT:   %impl.elem3.loc24_19.2: @TestIterate.%.loc24_19.20 (%.b00) = impl_witness_access %Iterate.lookup_impl_witness, element3 [symbolic = %impl.elem3.loc24_19.2 (constants.%impl.elem3.c16)]
// CHECK:STDOUT:   %specific_impl_fn.loc24_19.5: <specific function> = specific_impl_function %impl.elem3.loc24_19.2, @Iterate.WithSelf.Next(%Iterate.facet.loc24_19.5) [symbolic = %specific_impl_fn.loc24_19.5 (constants.%specific_impl_fn.b6b)]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type: type = fn_type @Destroy.WithSelf.SelfDestruct.1, @Destroy.WithSelf(%impl.elem1) [symbolic = %Destroy.WithSelf.SelfDestruct.type (constants.%Destroy.WithSelf.SelfDestruct.type.b85)]
// CHECK:STDOUT:   %.loc24_19.21: type = fn_type_with_self_type %Destroy.WithSelf.SelfDestruct.type, %impl.elem1 [symbolic = %.loc24_19.21 (constants.%.056)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %impl.elem1, @Destroy [symbolic = %Destroy.lookup_impl_witness (constants.%Destroy.lookup_impl_witness.068)]
// CHECK:STDOUT:   %impl.elem2.loc24_19.4: @TestIterate.%.loc24_19.21 (%.056) = impl_witness_access %Destroy.lookup_impl_witness, element2 [symbolic = %impl.elem2.loc24_19.4 (constants.%impl.elem2.5f8)]
// CHECK:STDOUT:   %specific_impl_fn.loc24_19.6: <specific function> = specific_impl_function %impl.elem2.loc24_19.4, @Destroy.WithSelf.SelfDestruct.1(%impl.elem1) [symbolic = %specific_impl_fn.loc24_19.6 (constants.%specific_impl_fn.b61)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%r.param: @TestIterate.%R.as_type.loc20_61.1 (%R.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %i.patt: %pattern_type.fb7 = value_binding_pattern i [concrete = constants.%i.patt]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %r.ref: @TestIterate.%R.as_type.loc20_61.1 (%R.as_type) = name_ref r, %r
// CHECK:STDOUT:     %impl.elem2.loc24_19.1: @TestIterate.%.loc24_19.19 (%.afc) = impl_witness_access constants.%Iterate.lookup_impl_witness.9e0, element2 [symbolic = %impl.elem2.loc24_19.3 (constants.%impl.elem2.895)]
// CHECK:STDOUT:     %bound_method.loc24_19.1: <bound method> = bound_method %r.ref, %impl.elem2.loc24_19.1
// CHECK:STDOUT:     %Iterate.facet.loc24_19.1: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.9e0) [symbolic = %Iterate.facet.loc24_19.5 (constants.%Iterate.facet.f88)]
// CHECK:STDOUT:     %.loc24_19.1: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc24_19.1 [symbolic = %Iterate.facet.loc24_19.5 (constants.%Iterate.facet.f88)]
// CHECK:STDOUT:     %Iterate.facet.loc24_19.2: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.9e0) [symbolic = %Iterate.facet.loc24_19.5 (constants.%Iterate.facet.f88)]
// CHECK:STDOUT:     %.loc24_19.2: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc24_19.2 [symbolic = %Iterate.facet.loc24_19.5 (constants.%Iterate.facet.f88)]
// CHECK:STDOUT:     %specific_impl_fn.loc24_19.1: <specific function> = specific_impl_function %impl.elem2.loc24_19.1, @Iterate.WithSelf.NewCursor(constants.%Iterate.facet.f88) [symbolic = %specific_impl_fn.loc24_19.4 (constants.%specific_impl_fn.999)]
// CHECK:STDOUT:     %bound_method.loc24_19.2: <bound method> = bound_method %r.ref, %specific_impl_fn.loc24_19.1
// CHECK:STDOUT:     %var: ref @TestIterate.%as_type (%as_type.ce3) = var_storage invalid
// CHECK:STDOUT:     %Iterate.WithSelf.NewCursor.call: init @TestIterate.%as_type (%as_type.ce3) to %var = call %bound_method.loc24_19.2(%r.ref)
// CHECK:STDOUT:     assign %var, %Iterate.WithSelf.NewCursor.call
// CHECK:STDOUT:     br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.next:
// CHECK:STDOUT:     %addr: @TestIterate.%ptr (%ptr.44a) = addr_of %var
// CHECK:STDOUT:     %impl.elem3.loc24_19.1: @TestIterate.%.loc24_19.20 (%.b00) = impl_witness_access constants.%Iterate.lookup_impl_witness.9e0, element3 [symbolic = %impl.elem3.loc24_19.2 (constants.%impl.elem3.c16)]
// CHECK:STDOUT:     %bound_method.loc24_19.3: <bound method> = bound_method %r.ref, %impl.elem3.loc24_19.1
// CHECK:STDOUT:     %Iterate.facet.loc24_19.3: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.9e0) [symbolic = %Iterate.facet.loc24_19.5 (constants.%Iterate.facet.f88)]
// CHECK:STDOUT:     %.loc24_19.3: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc24_19.3 [symbolic = %Iterate.facet.loc24_19.5 (constants.%Iterate.facet.f88)]
// CHECK:STDOUT:     %Iterate.facet.loc24_19.4: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.9e0) [symbolic = %Iterate.facet.loc24_19.5 (constants.%Iterate.facet.f88)]
// CHECK:STDOUT:     %.loc24_19.4: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc24_19.4 [symbolic = %Iterate.facet.loc24_19.5 (constants.%Iterate.facet.f88)]
// CHECK:STDOUT:     %.loc24_19.5: %Destroy.type = converted constants.%as_type.ce3, constants.%impl.elem1.0cf [symbolic = %impl.elem1 (constants.%impl.elem1.0cf)]
// CHECK:STDOUT:     %.loc24_19.6: %Destroy.type = converted constants.%as_type.ce3, constants.%impl.elem1.0cf [symbolic = %impl.elem1 (constants.%impl.elem1.0cf)]
// CHECK:STDOUT:     %specific_impl_fn.loc24_19.2: <specific function> = specific_impl_function %impl.elem3.loc24_19.1, @Iterate.WithSelf.Next(constants.%Iterate.facet.f88) [symbolic = %specific_impl_fn.loc24_19.5 (constants.%specific_impl_fn.b6b)]
// CHECK:STDOUT:     %bound_method.loc24_19.4: <bound method> = bound_method %r.ref, %specific_impl_fn.loc24_19.2
// CHECK:STDOUT:     %.loc24_19.7: ref %Optional.049 = temporary_storage
// CHECK:STDOUT:     %Iterate.WithSelf.Next.call: init %Optional.049 to %.loc24_19.7 = call %bound_method.loc24_19.4(%r.ref, %addr)
// CHECK:STDOUT:     %.loc24_19.8: ref %Optional.049 = temporary %.loc24_19.7, %Iterate.WithSelf.Next.call
// CHECK:STDOUT:     %.loc24_19.9: %Optional.HasValue.type.794 = specific_constant imports.%Core.import_ref.bb6, @Optional(constants.%OptionalStorage.facet.405) [concrete = constants.%Optional.HasValue.776]
// CHECK:STDOUT:     %HasValue.ref: %Optional.HasValue.type.794 = name_ref HasValue, %.loc24_19.9 [concrete = constants.%Optional.HasValue.776]
// CHECK:STDOUT:     %Optional.HasValue.bound: <bound method> = bound_method %.loc24_19.8, %HasValue.ref
// CHECK:STDOUT:     %Optional.HasValue.specific_fn: <specific function> = specific_function %HasValue.ref, @Optional.HasValue(constants.%OptionalStorage.facet.405) [concrete = constants.%Optional.HasValue.specific_fn.9fc]
// CHECK:STDOUT:     %bound_method.loc24_19.5: <bound method> = bound_method %.loc24_19.8, %Optional.HasValue.specific_fn
// CHECK:STDOUT:     %.loc24_19.10: %Optional.049 = acquire_value %.loc24_19.8
// CHECK:STDOUT:     %Optional.HasValue.call: init bool = call %bound_method.loc24_19.5(%.loc24_19.10)
// CHECK:STDOUT:     %.loc24_19.11: bool = value_of_initializer %Optional.HasValue.call
// CHECK:STDOUT:     %.loc24_19.12: bool = converted %Optional.HasValue.call, %.loc24_19.11
// CHECK:STDOUT:     if %.loc24_19.12 br !for.body else br !for.done
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.body:
// CHECK:STDOUT:     %.loc24_19.13: %Optional.Get.type.e6c = specific_constant imports.%Core.import_ref.dab, @Optional(constants.%OptionalStorage.facet.405) [concrete = constants.%Optional.Get.cb6]
// CHECK:STDOUT:     %Get.ref: %Optional.Get.type.e6c = name_ref Get, %.loc24_19.13 [concrete = constants.%Optional.Get.cb6]
// CHECK:STDOUT:     %Optional.Get.bound: <bound method> = bound_method %.loc24_19.8, %Get.ref
// CHECK:STDOUT:     %Optional.Get.specific_fn: <specific function> = specific_function %Get.ref, @Optional.Get(constants.%OptionalStorage.facet.405) [concrete = constants.%Optional.Get.specific_fn.e69]
// CHECK:STDOUT:     %bound_method.loc24_19.6: <bound method> = bound_method %.loc24_19.8, %Optional.Get.specific_fn
// CHECK:STDOUT:     %.loc24_19.14: %Optional.049 = acquire_value %.loc24_19.8
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
// CHECK:STDOUT:     %Destroy.WithSelf.SelfDestruct.bound.loc24_19.1: <bound method> = bound_method %.loc24_19.8, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.9
// CHECK:STDOUT:     %Destroy.WithSelf.SelfDestruct.call.loc24_19.1: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc24_19.1(%.loc24_19.8)
// CHECK:STDOUT:     br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.done:
// CHECK:STDOUT:     %Destroy.WithSelf.SelfDestruct.bound.loc24_19.2: <bound method> = bound_method %.loc24_19.8, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.9
// CHECK:STDOUT:     %Destroy.WithSelf.SelfDestruct.call.loc24_19.2: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc24_19.2(%.loc24_19.8)
// CHECK:STDOUT:     %impl.elem2.loc24_19.2: @TestIterate.%.loc24_19.21 (%.056) = impl_witness_access constants.%Destroy.lookup_impl_witness.068, element2 [symbolic = %impl.elem2.loc24_19.4 (constants.%impl.elem2.5f8)]
// CHECK:STDOUT:     %bound_method.loc24_19.7: <bound method> = bound_method %var, %impl.elem2.loc24_19.2
// CHECK:STDOUT:     %.loc24_19.17: %Destroy.type = converted constants.%as_type.ce3, constants.%impl.elem1.0cf [symbolic = %impl.elem1 (constants.%impl.elem1.0cf)]
// CHECK:STDOUT:     %.loc24_19.18: %Destroy.type = converted constants.%as_type.ce3, constants.%impl.elem1.0cf [symbolic = %impl.elem1 (constants.%impl.elem1.0cf)]
// CHECK:STDOUT:     %specific_impl_fn.loc24_19.3: <specific function> = specific_impl_function %impl.elem2.loc24_19.2, @Destroy.WithSelf.SelfDestruct.1(constants.%impl.elem1.0cf) [symbolic = %specific_impl_fn.loc24_19.6 (constants.%specific_impl_fn.b61)]
// CHECK:STDOUT:     %bound_method.loc24_19.8: <bound method> = bound_method %var, %specific_impl_fn.loc24_19.3
// CHECK:STDOUT:     %Destroy.WithSelf.SelfDestruct.call.loc24_19.3: init %empty_tuple.type = call %bound_method.loc24_19.8(%var)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc24_19.1(%self.param: ref %.cc0) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc24_19.1(%self.param: ref %.cc0) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc24_19.1(%self.param: ref %.cc0) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.3(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.3(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc24_19.2(%self.param: ref %MaybeUnformed.8ff) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc24_19.2(%self.param: ref %MaybeUnformed.8ff) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc24_19.2(%self.param: ref %MaybeUnformed.8ff) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.4(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.4(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc24_19.3(%self.param: ref bool) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc24_19.3(%self.param: ref bool) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc24_19.3(%self.param: ref bool) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.5(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.5(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc24_19.4(%self.param: ref %struct_type.value.has_value.58e) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc24_19.1: ref %MaybeUnformed.8ff = struct_access %self.param, element0
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc24_19.1: <bound method> = bound_method %.loc24_19.1, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.5
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc24_19.1: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc24_19.1(%.loc24_19.1)
// CHECK:STDOUT:   %.loc24_19.2: ref bool = struct_access %self.param, element1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc24_19.2: <bound method> = bound_method %.loc24_19.2, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.6
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc24_19.2: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc24_19.2(%.loc24_19.2)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc24_19.4(%self.param: ref %struct_type.value.has_value.58e) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc24_19.4(%self.param: ref %struct_type.value.has_value.58e) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.6(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.6(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc24_19.5(%self.param: ref %DefaultOptionalStorage.3e2) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc24_19.5(%self.param: ref %DefaultOptionalStorage.3e2) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc24_19.5(%self.param: ref %DefaultOptionalStorage.3e2) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.7(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.7(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc24_19.6(%self.param: ref %Optional.049) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc24_19.6(%self.param: ref %Optional.049) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc24_19.6(%self.param: ref %Optional.049) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.8(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.8(%self.param)
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
// CHECK:STDOUT:   %impl.elem2: %.3f0 = impl_witness_access constants.%Iterate.impl_witness.2d1, element2 [concrete = constants.%T.as_type.as.Iterate.impl.NewCursor.c9a]
// CHECK:STDOUT:   %bound_method.loc38_19.1: <bound method> = bound_method %c.ref, %impl.elem2
// CHECK:STDOUT:   %facet_value.loc38_19.1: %CppRange_where.type.d682d6.1 = facet_value constants.%ConstRange, (constants.%custom_witness.f8f19d.3, constants.%Copy.impl_witness.0d6, constants.%custom_witness.5d5, constants.%custom_witness.b76, constants.%custom_witness.994, constants.%custom_witness.393, constants.%custom_witness.821) [concrete = constants.%facet_value.e81]
// CHECK:STDOUT:   %.loc38_19.1: %CppRange_where.type.d682d6.1 = converted constants.%ConstRange, %facet_value.loc38_19.1 [concrete = constants.%facet_value.e81]
// CHECK:STDOUT:   %facet_value.loc38_19.2: %CppRange_where.type.d682d6.1 = facet_value constants.%ConstRange, (constants.%custom_witness.f8f19d.3, constants.%Copy.impl_witness.0d6, constants.%custom_witness.5d5, constants.%custom_witness.b76, constants.%custom_witness.994, constants.%custom_witness.393, constants.%custom_witness.821) [concrete = constants.%facet_value.e81]
// CHECK:STDOUT:   %.loc38_19.2: %CppRange_where.type.d682d6.1 = converted constants.%ConstRange, %facet_value.loc38_19.2 [concrete = constants.%facet_value.e81]
// CHECK:STDOUT:   %specific_fn.loc38_19.1: <specific function> = specific_function %impl.elem2, @T.as_type.as.Iterate.impl.NewCursor(constants.%facet_value.e81) [concrete = constants.%T.as_type.as.Iterate.impl.NewCursor.specific_fn]
// CHECK:STDOUT:   %bound_method.loc38_19.2: <bound method> = bound_method %c.ref, %specific_fn.loc38_19.1
// CHECK:STDOUT:   %var: ref %tuple.type.005 = var_storage invalid
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.call: init %tuple.type.005 to %var = call %bound_method.loc38_19.2(%c.ref)
// CHECK:STDOUT:   assign %var, %T.as_type.as.Iterate.impl.NewCursor.call
// CHECK:STDOUT:   br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT: !for.next:
// CHECK:STDOUT:   %addr: %ptr.d1a = addr_of %var
// CHECK:STDOUT:   %impl.elem3: %.343 = impl_witness_access constants.%Iterate.impl_witness.2d1, element3 [concrete = constants.%T.as_type.as.Iterate.impl.Next.751]
// CHECK:STDOUT:   %bound_method.loc38_19.3: <bound method> = bound_method %c.ref, %impl.elem3
// CHECK:STDOUT:   %facet_value.loc38_19.3: %CppRange_where.type.d682d6.1 = facet_value constants.%ConstRange, (constants.%custom_witness.f8f19d.3, constants.%Copy.impl_witness.0d6, constants.%custom_witness.5d5, constants.%custom_witness.b76, constants.%custom_witness.994, constants.%custom_witness.393, constants.%custom_witness.821) [concrete = constants.%facet_value.e81]
// CHECK:STDOUT:   %.loc38_19.3: %CppRange_where.type.d682d6.1 = converted constants.%ConstRange, %facet_value.loc38_19.3 [concrete = constants.%facet_value.e81]
// CHECK:STDOUT:   %facet_value.loc38_19.4: %CppRange_where.type.d682d6.1 = facet_value constants.%ConstRange, (constants.%custom_witness.f8f19d.3, constants.%Copy.impl_witness.0d6, constants.%custom_witness.5d5, constants.%custom_witness.b76, constants.%custom_witness.994, constants.%custom_witness.393, constants.%custom_witness.821) [concrete = constants.%facet_value.e81]
// CHECK:STDOUT:   %.loc38_19.4: %CppRange_where.type.d682d6.1 = converted constants.%ConstRange, %facet_value.loc38_19.4 [concrete = constants.%facet_value.e81]
// CHECK:STDOUT:   %specific_fn.loc38_19.2: <specific function> = specific_function %impl.elem3, @T.as_type.as.Iterate.impl.Next(constants.%facet_value.e81) [concrete = constants.%T.as_type.as.Iterate.impl.Next.specific_fn]
// CHECK:STDOUT:   %bound_method.loc38_19.4: <bound method> = bound_method %c.ref, %specific_fn.loc38_19.2
// CHECK:STDOUT:   %.loc38_19.5: ref %Optional.049 = temporary_storage
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.call: init %Optional.049 to %.loc38_19.5 = call %bound_method.loc38_19.4(%c.ref, %addr)
// CHECK:STDOUT:   %.loc38_19.6: ref %Optional.049 = temporary %.loc38_19.5, %T.as_type.as.Iterate.impl.Next.call
// CHECK:STDOUT:   %.loc38_19.7: %Optional.HasValue.type.794 = specific_constant imports.%Core.import_ref.bb6, @Optional(constants.%OptionalStorage.facet.405) [concrete = constants.%Optional.HasValue.776]
// CHECK:STDOUT:   %HasValue.ref: %Optional.HasValue.type.794 = name_ref HasValue, %.loc38_19.7 [concrete = constants.%Optional.HasValue.776]
// CHECK:STDOUT:   %Optional.HasValue.bound: <bound method> = bound_method %.loc38_19.6, %HasValue.ref
// CHECK:STDOUT:   %Optional.HasValue.specific_fn: <specific function> = specific_function %HasValue.ref, @Optional.HasValue(constants.%OptionalStorage.facet.405) [concrete = constants.%Optional.HasValue.specific_fn.9fc]
// CHECK:STDOUT:   %bound_method.loc38_19.5: <bound method> = bound_method %.loc38_19.6, %Optional.HasValue.specific_fn
// CHECK:STDOUT:   %.loc38_19.8: %Optional.049 = acquire_value %.loc38_19.6
// CHECK:STDOUT:   %Optional.HasValue.call: init bool = call %bound_method.loc38_19.5(%.loc38_19.8)
// CHECK:STDOUT:   %.loc38_19.9: bool = value_of_initializer %Optional.HasValue.call
// CHECK:STDOUT:   %.loc38_19.10: bool = converted %Optional.HasValue.call, %.loc38_19.9
// CHECK:STDOUT:   if %.loc38_19.10 br !for.body else br !for.done
// CHECK:STDOUT:
// CHECK:STDOUT: !for.body:
// CHECK:STDOUT:   %.loc38_19.11: %Optional.Get.type.e6c = specific_constant imports.%Core.import_ref.dab, @Optional(constants.%OptionalStorage.facet.405) [concrete = constants.%Optional.Get.cb6]
// CHECK:STDOUT:   %Get.ref: %Optional.Get.type.e6c = name_ref Get, %.loc38_19.11 [concrete = constants.%Optional.Get.cb6]
// CHECK:STDOUT:   %Optional.Get.bound: <bound method> = bound_method %.loc38_19.6, %Get.ref
// CHECK:STDOUT:   %Optional.Get.specific_fn: <specific function> = specific_function %Get.ref, @Optional.Get(constants.%OptionalStorage.facet.405) [concrete = constants.%Optional.Get.specific_fn.e69]
// CHECK:STDOUT:   %bound_method.loc38_19.6: <bound method> = bound_method %.loc38_19.6, %Optional.Get.specific_fn
// CHECK:STDOUT:   %.loc38_19.12: %Optional.049 = acquire_value %.loc38_19.6
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
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc38_19.1: <bound method> = bound_method %.loc38_19.6, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.9
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc38_19.1: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc38_19.1(%.loc38_19.6)
// CHECK:STDOUT:   br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT: !for.done:
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc38_19.2: <bound method> = bound_method %.loc38_19.6, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.9
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc38_19.2: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc38_19.2(%.loc38_19.6)
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc38_19.3: <bound method> = bound_method %var, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.11
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc38_19.3: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc38_19.3(%var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestIterate(constants.%R) {
// CHECK:STDOUT:   %R.patt.loc20_17.2 => constants.%R.patt
// CHECK:STDOUT:   %R.loc20_17.1 => constants.%R
// CHECK:STDOUT:   %R.as_type.loc20_61.1 => constants.%R.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.6f7
// CHECK:STDOUT:   %r.param_patt.loc20_59.2 => constants.%r.param_patt.19a
// CHECK:STDOUT:   %r.patt.loc20_59.2 => constants.%r.patt.f97
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestIterate(constants.%facet_value.58e) {
// CHECK:STDOUT:   %R.patt.loc20_17.2 => constants.%R.patt
// CHECK:STDOUT:   %R.loc20_17.1 => constants.%facet_value.58e
// CHECK:STDOUT:   %R.as_type.loc20_61.1 => constants.%ConstRange
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.025
// CHECK:STDOUT:   %r.param_patt.loc20_59.2 => constants.%r.param_patt.a73
// CHECK:STDOUT:   %r.patt.loc20_59.2 => constants.%r.patt.0b5
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc20 => constants.%complete_type.357
// CHECK:STDOUT:   %Iterate.lookup_impl_witness => constants.%Iterate.impl_witness.2d1
// CHECK:STDOUT:   %Iterate.facet.loc24_19.5 => constants.%Iterate.facet.e86
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type => constants.%Iterate.WithSelf.NewCursor.type.bcd
// CHECK:STDOUT:   %.loc24_19.19 => constants.%.3f0
// CHECK:STDOUT:   %impl.elem2.loc24_19.3 => constants.%T.as_type.as.Iterate.impl.NewCursor.c9a
// CHECK:STDOUT:   %specific_impl_fn.loc24_19.4 => constants.%T.as_type.as.Iterate.impl.NewCursor.specific_fn
// CHECK:STDOUT:   %impl.elem1 => constants.%Destroy.facet.362
// CHECK:STDOUT:   %as_type => constants.%tuple.type.005
// CHECK:STDOUT:   %require_complete.1 => constants.%complete_type.320
// CHECK:STDOUT:   %ptr => constants.%ptr.d1a
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type => constants.%Iterate.WithSelf.Next.type.44e
// CHECK:STDOUT:   %.loc24_19.20 => constants.%.343
// CHECK:STDOUT:   %impl.elem3.loc24_19.2 => constants.%T.as_type.as.Iterate.impl.Next.751
// CHECK:STDOUT:   %specific_impl_fn.loc24_19.5 => constants.%T.as_type.as.Iterate.impl.Next.specific_fn
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type => constants.%Destroy.WithSelf.SelfDestruct.type.873
// CHECK:STDOUT:   %.loc24_19.21 => constants.%.9a7
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.f8f19d.10
// CHECK:STDOUT:   %impl.elem2.loc24_19.4 => constants.%Destroy.WithSelf.SelfDestruct.db3fdb.11
// CHECK:STDOUT:   %specific_impl_fn.loc24_19.6 => constants.%Destroy.WithSelf.SelfDestruct.db3fdb.11
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_adl_returns_same_types_mutable.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %Iterate.type: type = facet_type <@Iterate> [concrete]
// CHECK:STDOUT:   %OptionalStorage.type: type = facet_type <@OptionalStorage> [concrete]
// CHECK:STDOUT:   %pattern_type.831: type = pattern_type bool [concrete]
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
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.3: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc22_53.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.3: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.3 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.3: type = fn_type @Destroy.WithSelf.Op.loc22_53.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.3: %Destroy.WithSelf.Op.type.ef016f.3 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.3: type = fn_type @Destroy.WithSelf.SelfDestruct.loc22_53.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.3: %Destroy.WithSelf.SelfDestruct.type.fbceb5.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.f8f19d.3: <witness> = custom_witness (%Destroy.WithSelf.Op.403171.3, %Destroy.WithSelf.SubobjectDestroy.d01daf.3, %Destroy.WithSelf.SelfDestruct.db3fdb.3), @Destroy [concrete]
// CHECK:STDOUT:   %Float.as.Copy.impl.Op.type.ff3: type = fn_type @Float.as.Copy.impl.Op, @Float.as.Copy.impl(%N.fe9) [symbolic]
// CHECK:STDOUT:   %Float.as.Copy.impl.Op.d93: %Float.as.Copy.impl.Op.type.ff3 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.0d6: <witness> = impl_witness imports.%Copy.impl_witness_table.8dd, @Float.as.Copy.impl(%int_64) [concrete]
// CHECK:STDOUT:   %facet_value.b6e: %facet_type.f48 = facet_value %f64.dc1, (%custom_witness.f8f19d.3, %Copy.impl_witness.0d6) [concrete]
// CHECK:STDOUT:   %Iterate.lookup_impl_witness.1d8: <witness> = lookup_impl_witness %.Self.500, @Iterate [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.294: %facet_type.f48 = impl_witness_access %Iterate.lookup_impl_witness.1d8, element0 [symbolic_self]
// CHECK:STDOUT:   %Iterate_where.type.028: type = facet_type <@Iterate where %impl.elem0.294 = %facet_value.b6e> [concrete]
// CHECK:STDOUT:   %pattern_type.522: type = pattern_type %Iterate_where.type.028 [concrete]
// CHECK:STDOUT:   %R.patt: %pattern_type.522 = symbolic_binding_pattern R, 0 [symbolic]
// CHECK:STDOUT:   %R: %Iterate_where.type.028 = symbolic_binding R, 0 [symbolic]
// CHECK:STDOUT:   %R.as_type: type = facet_access_type %R [symbolic]
// CHECK:STDOUT:   %pattern_type.6f7: type = pattern_type %R.as_type [symbolic]
// CHECK:STDOUT:   %r.param_patt.19a: %pattern_type.6f7 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %r.patt.f97: %pattern_type.6f7 = wrapper_binding_pattern r, %r.param_patt.19a [symbolic]
// CHECK:STDOUT:   %i.patt: %pattern_type.fb7 = value_binding_pattern i [concrete]
// CHECK:STDOUT:   %Iterate.lookup_impl_witness.9e0: <witness> = lookup_impl_witness %R, @Iterate [symbolic]
// CHECK:STDOUT:   %Iterate.facet.f88: %Iterate.type = facet_value %R.as_type, (%Iterate.lookup_impl_witness.9e0) [symbolic]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.3d0: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.f88) [symbolic]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.160: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.f88) [symbolic]
// CHECK:STDOUT:   %.afc: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.3d0, %Iterate.facet.f88 [symbolic]
// CHECK:STDOUT:   %impl.elem2.895: %.afc = impl_witness_access %Iterate.lookup_impl_witness.9e0, element2 [symbolic]
// CHECK:STDOUT:   %impl.elem1.0cf: %Destroy.type = impl_witness_access %Iterate.lookup_impl_witness.9e0, element1 [symbolic]
// CHECK:STDOUT:   %as_type.ce3: type = facet_access_type %impl.elem1.0cf [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.999: <specific function> = specific_impl_function %impl.elem2.895, @Iterate.WithSelf.NewCursor(%Iterate.facet.f88) [symbolic]
// CHECK:STDOUT:   %ptr.44a: type = ptr_type %as_type.ce3 [symbolic]
// CHECK:STDOUT:   %.b00: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.160, %Iterate.facet.f88 [symbolic]
// CHECK:STDOUT:   %impl.elem3.c16: %.b00 = impl_witness_access %Iterate.lookup_impl_witness.9e0, element3 [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.3e2: type = class_type @DefaultOptionalStorage, @DefaultOptionalStorage(%facet_value.b6e) [concrete]
// CHECK:STDOUT:   %OptionalStorage.impl_witness.4b3: <witness> = impl_witness imports.%OptionalStorage.impl_witness_table.460, @T.as_type.as.OptionalStorage.impl(%facet_value.b6e) [concrete]
// CHECK:STDOUT:   %OptionalStorage.facet.405: %OptionalStorage.type = facet_value %f64.dc1, (%OptionalStorage.impl_witness.4b3) [concrete]
// CHECK:STDOUT:   %Optional.049: type = class_type @Optional, @Optional(%OptionalStorage.facet.405) [concrete]
// CHECK:STDOUT:   %pattern_type.449: type = pattern_type %Optional.049 [concrete]
// CHECK:STDOUT:   %specific_impl_fn.b6b: <specific function> = specific_impl_function %impl.elem3.c16, @Iterate.WithSelf.Next(%Iterate.facet.f88) [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.type.794: type = fn_type @Optional.HasValue, @Optional(%OptionalStorage.facet.405) [concrete]
// CHECK:STDOUT:   %Optional.HasValue.776: %Optional.HasValue.type.794 = struct_value () [concrete]
// CHECK:STDOUT:   %Optional.Get.type.e6c: type = fn_type @Optional.Get, @Optional(%OptionalStorage.facet.405) [concrete]
// CHECK:STDOUT:   %Optional.Get.cb6: %Optional.Get.type.e6c = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.facet.d87: %Destroy.type = facet_value %f64.dc1, (%custom_witness.f8f19d.3) [concrete]
// CHECK:STDOUT:   %MaybeUnformed.8ff: type = class_type @MaybeUnformed, @MaybeUnformed(%Destroy.facet.d87) [concrete]
// CHECK:STDOUT:   %.cc0: type = maybe_unformed_type %f64.dc1 [concrete]
// CHECK:STDOUT:   %struct_type.value.has_value.58e: type = struct_type {.value: %MaybeUnformed.8ff, .has_value: bool} [concrete]
// CHECK:STDOUT:   %Optional.HasValue.specific_fn.9fc: <specific function> = specific_function %Optional.HasValue.776, @Optional.HasValue(%OptionalStorage.facet.405) [concrete]
// CHECK:STDOUT:   %Optional.Get.specific_fn.e69: <specific function> = specific_function %Optional.Get.cb6, @Optional.Get(%OptionalStorage.facet.405) [concrete]
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
// CHECK:STDOUT:   %pattern_type.1fb: type = pattern_type %.cc0 [concrete]
// CHECK:STDOUT:   %self.param_patt.4fd: %pattern_type.1fb = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.69f: %pattern_type.1fb = wrapper_binding_pattern self, %self.param_patt.4fd [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.4: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc26_19.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.4: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.4 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.4: type = fn_type @Destroy.WithSelf.Op.loc26_19.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.4: %Destroy.WithSelf.Op.type.ef016f.4 = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.80d: type = pattern_type %MaybeUnformed.8ff [concrete]
// CHECK:STDOUT:   %self.param_patt.70c: %pattern_type.80d = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.120: %pattern_type.80d = wrapper_binding_pattern self, %self.param_patt.70c [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.5: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc26_19.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.5: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.5 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.5: type = fn_type @Destroy.WithSelf.Op.loc26_19.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.5: %Destroy.WithSelf.Op.type.ef016f.5 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.5: type = fn_type @Destroy.WithSelf.SelfDestruct.loc26_19.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.5: %Destroy.WithSelf.SelfDestruct.type.fbceb5.5 = struct_value () [concrete]
// CHECK:STDOUT:   %self.param_patt.40d: %pattern_type.831 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.560: %pattern_type.831 = wrapper_binding_pattern self, %self.param_patt.40d [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.6: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc26_19.3 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.6: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.6 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.6: type = fn_type @Destroy.WithSelf.Op.loc26_19.3 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.6: %Destroy.WithSelf.Op.type.ef016f.6 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.6: type = fn_type @Destroy.WithSelf.SelfDestruct.loc26_19.3 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.6: %Destroy.WithSelf.SelfDestruct.type.fbceb5.6 = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.00a: type = pattern_type %struct_type.value.has_value.58e [concrete]
// CHECK:STDOUT:   %self.param_patt.96d: %pattern_type.00a = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.602: %pattern_type.00a = wrapper_binding_pattern self, %self.param_patt.96d [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.7: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc26_19.4 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.7: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.7 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.7: type = fn_type @Destroy.WithSelf.Op.loc26_19.4 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.7: %Destroy.WithSelf.Op.type.ef016f.7 = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.406: type = pattern_type %DefaultOptionalStorage.3e2 [concrete]
// CHECK:STDOUT:   %self.param_patt.73b: %pattern_type.406 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.bd7: %pattern_type.406 = wrapper_binding_pattern self, %self.param_patt.73b [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.8: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc26_19.5 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.8: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.8 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.8: type = fn_type @Destroy.WithSelf.Op.loc26_19.5 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.8: %Destroy.WithSelf.Op.type.ef016f.8 = struct_value () [concrete]
// CHECK:STDOUT:   %self.param_patt.a67: %pattern_type.449 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.521: %pattern_type.449 = wrapper_binding_pattern self, %self.param_patt.a67 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.9: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc26_19.6 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.9: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.9 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.9: type = fn_type @Destroy.WithSelf.Op.loc26_19.6 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.9: %Destroy.WithSelf.Op.type.ef016f.9 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.9: type = fn_type @Destroy.WithSelf.SelfDestruct.loc26_19.6 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.9: %Destroy.WithSelf.SelfDestruct.type.fbceb5.9 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.068: <witness> = lookup_impl_witness %impl.elem1.0cf, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.b85: type = fn_type @Destroy.WithSelf.SelfDestruct.1, @Destroy.WithSelf(%impl.elem1.0cf) [symbolic]
// CHECK:STDOUT:   %.056: type = fn_type_with_self_type %Destroy.WithSelf.SelfDestruct.type.b85, %impl.elem1.0cf [symbolic]
// CHECK:STDOUT:   %impl.elem2.5f8: %.056 = impl_witness_access %Destroy.lookup_impl_witness.068, element2 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.b61: <specific function> = specific_impl_function %impl.elem2.5f8, @Destroy.WithSelf.SelfDestruct.1(%impl.elem1.0cf) [symbolic]
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
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.10: type = fn_type @Destroy.WithSelf.Op.3 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.10: %Destroy.WithSelf.Op.type.ef016f.10 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.10: type = fn_type @Destroy.WithSelf.SelfDestruct.3 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.10: %Destroy.WithSelf.SelfDestruct.type.fbceb5.10 = struct_value () [concrete]
// CHECK:STDOUT:   %Iterator.SubobjectDestroy.type: type = fn_type @Iterator.SubobjectDestroy [concrete]
// CHECK:STDOUT:   %Iterator.SubobjectDestroy: %Iterator.SubobjectDestroy.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.e74: <witness> = custom_witness (%Destroy.WithSelf.Op.403171.10, %Iterator.SubobjectDestroy, %Destroy.WithSelf.SelfDestruct.db3fdb.10), @Destroy [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.b9731e.2: type = fn_type @Iterator.Op.2 [concrete]
// CHECK:STDOUT:   %Iterator.Op.ac6aad.2: %Iterator.Op.type.b9731e.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.491: <witness> = custom_witness (%Iterator.Op.ac6aad.2), @Inc [concrete]
// CHECK:STDOUT:   %Equal.type: type = fn_type @Equal [concrete]
// CHECK:STDOUT:   %Equal: %Equal.type = struct_value () [concrete]
// CHECK:STDOUT:   %NotEqual.type: type = fn_type @NotEqual [concrete]
// CHECK:STDOUT:   %NotEqual: %NotEqual.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.ed8: <witness> = custom_witness (%Equal, %NotEqual), @EqWith, @EqWith(%Iterator) [concrete]
// CHECK:STDOUT:   %facet_value.efd: %CppRange_where.type.d682d6.1 = facet_value %MutableRange, (%custom_witness.f8f19d.3, %Copy.impl_witness.0d6, %custom_witness.902, %custom_witness.e74, %custom_witness.05b, %custom_witness.491, %custom_witness.ed8) [concrete]
// CHECK:STDOUT:   %tuple.type.dd0: type = tuple_type (%Iterator, %Iterator) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.10: type = fn_type @Destroy.WithSelf.SubobjectDestroy.3 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.10: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.10 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.11: type = fn_type @Destroy.WithSelf.Op.4 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.11: %Destroy.WithSelf.Op.type.ef016f.11 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.11: type = fn_type @Destroy.WithSelf.SelfDestruct.4 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.11: %Destroy.WithSelf.SelfDestruct.type.fbceb5.11 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.f8f19d.10: <witness> = custom_witness (%Destroy.WithSelf.Op.403171.11, %Destroy.WithSelf.SubobjectDestroy.d01daf.10, %Destroy.WithSelf.SelfDestruct.db3fdb.11), @Destroy [concrete]
// CHECK:STDOUT:   %Destroy.facet.f0b: %Destroy.type = facet_value %tuple.type.dd0, (%custom_witness.f8f19d.10) [concrete]
// CHECK:STDOUT:   %Iterate.impl_witness.6cb: <witness> = impl_witness imports.%Iterate.impl_witness_table.121, @T.as_type.as.Iterate.impl(%facet_value.efd) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.type.eb5: type = fn_type @T.as_type.as.Iterate.impl.NewCursor, @T.as_type.as.Iterate.impl(%facet_value.efd) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.fcd: %T.as_type.as.Iterate.impl.NewCursor.type.eb5 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.type.ebe: type = fn_type @T.as_type.as.Iterate.impl.Next, @T.as_type.as.Iterate.impl(%facet_value.efd) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.c88: %T.as_type.as.Iterate.impl.Next.type.ebe = struct_value () [concrete]
// CHECK:STDOUT:   %Iterate.facet.d07: %Iterate.type = facet_value %MutableRange, (%Iterate.impl_witness.6cb) [concrete]
// CHECK:STDOUT:   %facet_value.30f: %Iterate_where.type.028 = facet_value %MutableRange, (%Iterate.impl_witness.6cb) [concrete]
// CHECK:STDOUT:   %r.param_patt.aeb: %pattern_type.992 = value_param_pattern [concrete]
// CHECK:STDOUT:   %r.patt.892: %pattern_type.992 = wrapper_binding_pattern r, %r.param_patt.aeb [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.0ef: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.d07) [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.7af: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.d07) [concrete]
// CHECK:STDOUT:   %.698: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.0ef, %Iterate.facet.d07 [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.specific_fn: <specific function> = specific_function %T.as_type.as.Iterate.impl.NewCursor.fcd, @T.as_type.as.Iterate.impl.NewCursor(%facet_value.efd) [concrete]
// CHECK:STDOUT:   %ptr.a4f: type = ptr_type %tuple.type.dd0 [concrete]
// CHECK:STDOUT:   %.782: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.7af, %Iterate.facet.d07 [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.specific_fn: <specific function> = specific_function %T.as_type.as.Iterate.impl.Next.c88, @T.as_type.as.Iterate.impl.Next(%facet_value.efd) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.e6e: type = fn_type @Destroy.WithSelf.SelfDestruct.1, @Destroy.WithSelf(%Destroy.facet.f0b) [concrete]
// CHECK:STDOUT:   %.94d: type = fn_type_with_self_type %Destroy.WithSelf.SelfDestruct.type.e6e, %Destroy.facet.f0b [concrete]
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
// CHECK:STDOUT: generated {
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.3: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.4 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc26_19.1 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.4] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.1fb = ref_param_pattern [concrete = constants.%self.param_patt.4fd]
// CHECK:STDOUT:     %self.patt: %pattern_type.1fb = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.69f]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %.cc0 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %.cc0 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.3: %Destroy.WithSelf.Op.type.ef016f.4 = fn_decl @Destroy.WithSelf.Op.loc26_19.1 [concrete = constants.%Destroy.WithSelf.Op.403171.4] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.1fb = ref_param_pattern [concrete = constants.%self.param_patt.4fd]
// CHECK:STDOUT:     %self.patt: %pattern_type.1fb = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.69f]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %.cc0 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %.cc0 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.4: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.5 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc26_19.2 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.5] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.80d = ref_param_pattern [concrete = constants.%self.param_patt.70c]
// CHECK:STDOUT:     %self.patt: %pattern_type.80d = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.120]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %MaybeUnformed.8ff = ref_param call_param0
// CHECK:STDOUT:     %self: ref %MaybeUnformed.8ff = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.4: %Destroy.WithSelf.Op.type.ef016f.5 = fn_decl @Destroy.WithSelf.Op.loc26_19.2 [concrete = constants.%Destroy.WithSelf.Op.403171.5] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.80d = ref_param_pattern [concrete = constants.%self.param_patt.70c]
// CHECK:STDOUT:     %self.patt: %pattern_type.80d = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.120]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %MaybeUnformed.8ff = ref_param call_param0
// CHECK:STDOUT:     %self: ref %MaybeUnformed.8ff = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.5: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.6 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc26_19.3 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.6] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.831 = ref_param_pattern [concrete = constants.%self.param_patt.40d]
// CHECK:STDOUT:     %self.patt: %pattern_type.831 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.560]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref bool = ref_param call_param0
// CHECK:STDOUT:     %self: ref bool = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.5: %Destroy.WithSelf.Op.type.ef016f.6 = fn_decl @Destroy.WithSelf.Op.loc26_19.3 [concrete = constants.%Destroy.WithSelf.Op.403171.6] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.831 = ref_param_pattern [concrete = constants.%self.param_patt.40d]
// CHECK:STDOUT:     %self.patt: %pattern_type.831 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.560]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref bool = ref_param call_param0
// CHECK:STDOUT:     %self: ref bool = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.6: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.7 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc26_19.4 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.7] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.00a = ref_param_pattern [concrete = constants.%self.param_patt.96d]
// CHECK:STDOUT:     %self.patt: %pattern_type.00a = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.602]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %struct_type.value.has_value.58e = ref_param call_param0
// CHECK:STDOUT:     %self: ref %struct_type.value.has_value.58e = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.6: %Destroy.WithSelf.Op.type.ef016f.7 = fn_decl @Destroy.WithSelf.Op.loc26_19.4 [concrete = constants.%Destroy.WithSelf.Op.403171.7] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.00a = ref_param_pattern [concrete = constants.%self.param_patt.96d]
// CHECK:STDOUT:     %self.patt: %pattern_type.00a = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.602]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %struct_type.value.has_value.58e = ref_param call_param0
// CHECK:STDOUT:     %self: ref %struct_type.value.has_value.58e = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.7: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.8 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc26_19.5 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.8] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.406 = ref_param_pattern [concrete = constants.%self.param_patt.73b]
// CHECK:STDOUT:     %self.patt: %pattern_type.406 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.bd7]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %DefaultOptionalStorage.3e2 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %DefaultOptionalStorage.3e2 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.7: %Destroy.WithSelf.Op.type.ef016f.8 = fn_decl @Destroy.WithSelf.Op.loc26_19.5 [concrete = constants.%Destroy.WithSelf.Op.403171.8] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.406 = ref_param_pattern [concrete = constants.%self.param_patt.73b]
// CHECK:STDOUT:     %self.patt: %pattern_type.406 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.bd7]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %DefaultOptionalStorage.3e2 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %DefaultOptionalStorage.3e2 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.8: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.9 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc26_19.6 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.9] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.449 = ref_param_pattern [concrete = constants.%self.param_patt.a67]
// CHECK:STDOUT:     %self.patt: %pattern_type.449 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.521]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %Optional.049 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %Optional.049 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.8: %Destroy.WithSelf.Op.type.ef016f.9 = fn_decl @Destroy.WithSelf.Op.loc26_19.6 [concrete = constants.%Destroy.WithSelf.Op.403171.9] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.449 = ref_param_pattern [concrete = constants.%self.param_patt.a67]
// CHECK:STDOUT:     %self.patt: %pattern_type.449 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.521]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %Optional.049 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %Optional.049 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @TestIterate(%R.loc22_17.2: %Iterate_where.type.028) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Iterate.lookup_impl_witness: <witness> = lookup_impl_witness %R.loc22_17.1, @Iterate [symbolic = %Iterate.lookup_impl_witness (constants.%Iterate.lookup_impl_witness.9e0)]
// CHECK:STDOUT:   %Iterate.facet.loc26_19.5: %Iterate.type = facet_value %R.as_type.loc22_61.1, (%Iterate.lookup_impl_witness) [symbolic = %Iterate.facet.loc26_19.5 (constants.%Iterate.facet.f88)]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.loc26_19.5) [symbolic = %Iterate.WithSelf.NewCursor.type (constants.%Iterate.WithSelf.NewCursor.type.3d0)]
// CHECK:STDOUT:   %.loc26_19.19: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type, %Iterate.facet.loc26_19.5 [symbolic = %.loc26_19.19 (constants.%.afc)]
// CHECK:STDOUT:   %impl.elem2.loc26_19.3: @TestIterate.%.loc26_19.19 (%.afc) = impl_witness_access %Iterate.lookup_impl_witness, element2 [symbolic = %impl.elem2.loc26_19.3 (constants.%impl.elem2.895)]
// CHECK:STDOUT:   %specific_impl_fn.loc26_19.4: <specific function> = specific_impl_function %impl.elem2.loc26_19.3, @Iterate.WithSelf.NewCursor(%Iterate.facet.loc26_19.5) [symbolic = %specific_impl_fn.loc26_19.4 (constants.%specific_impl_fn.999)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %ptr: type = ptr_type %as_type [symbolic = %ptr (constants.%ptr.44a)]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.loc26_19.5) [symbolic = %Iterate.WithSelf.Next.type (constants.%Iterate.WithSelf.Next.type.160)]
// CHECK:STDOUT:   %.loc26_19.20: type = fn_type_with_self_type %Iterate.WithSelf.Next.type, %Iterate.facet.loc26_19.5 [symbolic = %.loc26_19.20 (constants.%.b00)]
// CHECK:STDOUT:   %impl.elem3.loc26_19.2: @TestIterate.%.loc26_19.20 (%.b00) = impl_witness_access %Iterate.lookup_impl_witness, element3 [symbolic = %impl.elem3.loc26_19.2 (constants.%impl.elem3.c16)]
// CHECK:STDOUT:   %specific_impl_fn.loc26_19.5: <specific function> = specific_impl_function %impl.elem3.loc26_19.2, @Iterate.WithSelf.Next(%Iterate.facet.loc26_19.5) [symbolic = %specific_impl_fn.loc26_19.5 (constants.%specific_impl_fn.b6b)]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type: type = fn_type @Destroy.WithSelf.SelfDestruct.1, @Destroy.WithSelf(%impl.elem1) [symbolic = %Destroy.WithSelf.SelfDestruct.type (constants.%Destroy.WithSelf.SelfDestruct.type.b85)]
// CHECK:STDOUT:   %.loc26_19.21: type = fn_type_with_self_type %Destroy.WithSelf.SelfDestruct.type, %impl.elem1 [symbolic = %.loc26_19.21 (constants.%.056)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %impl.elem1, @Destroy [symbolic = %Destroy.lookup_impl_witness (constants.%Destroy.lookup_impl_witness.068)]
// CHECK:STDOUT:   %impl.elem2.loc26_19.4: @TestIterate.%.loc26_19.21 (%.056) = impl_witness_access %Destroy.lookup_impl_witness, element2 [symbolic = %impl.elem2.loc26_19.4 (constants.%impl.elem2.5f8)]
// CHECK:STDOUT:   %specific_impl_fn.loc26_19.6: <specific function> = specific_impl_function %impl.elem2.loc26_19.4, @Destroy.WithSelf.SelfDestruct.1(%impl.elem1) [symbolic = %specific_impl_fn.loc26_19.6 (constants.%specific_impl_fn.b61)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%r.param: @TestIterate.%R.as_type.loc22_61.1 (%R.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %i.patt: %pattern_type.fb7 = value_binding_pattern i [concrete = constants.%i.patt]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %r.ref: @TestIterate.%R.as_type.loc22_61.1 (%R.as_type) = name_ref r, %r
// CHECK:STDOUT:     %impl.elem2.loc26_19.1: @TestIterate.%.loc26_19.19 (%.afc) = impl_witness_access constants.%Iterate.lookup_impl_witness.9e0, element2 [symbolic = %impl.elem2.loc26_19.3 (constants.%impl.elem2.895)]
// CHECK:STDOUT:     %bound_method.loc26_19.1: <bound method> = bound_method %r.ref, %impl.elem2.loc26_19.1
// CHECK:STDOUT:     %Iterate.facet.loc26_19.1: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.9e0) [symbolic = %Iterate.facet.loc26_19.5 (constants.%Iterate.facet.f88)]
// CHECK:STDOUT:     %.loc26_19.1: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc26_19.1 [symbolic = %Iterate.facet.loc26_19.5 (constants.%Iterate.facet.f88)]
// CHECK:STDOUT:     %Iterate.facet.loc26_19.2: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.9e0) [symbolic = %Iterate.facet.loc26_19.5 (constants.%Iterate.facet.f88)]
// CHECK:STDOUT:     %.loc26_19.2: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc26_19.2 [symbolic = %Iterate.facet.loc26_19.5 (constants.%Iterate.facet.f88)]
// CHECK:STDOUT:     %specific_impl_fn.loc26_19.1: <specific function> = specific_impl_function %impl.elem2.loc26_19.1, @Iterate.WithSelf.NewCursor(constants.%Iterate.facet.f88) [symbolic = %specific_impl_fn.loc26_19.4 (constants.%specific_impl_fn.999)]
// CHECK:STDOUT:     %bound_method.loc26_19.2: <bound method> = bound_method %r.ref, %specific_impl_fn.loc26_19.1
// CHECK:STDOUT:     %var: ref @TestIterate.%as_type (%as_type.ce3) = var_storage invalid
// CHECK:STDOUT:     %Iterate.WithSelf.NewCursor.call: init @TestIterate.%as_type (%as_type.ce3) to %var = call %bound_method.loc26_19.2(%r.ref)
// CHECK:STDOUT:     assign %var, %Iterate.WithSelf.NewCursor.call
// CHECK:STDOUT:     br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.next:
// CHECK:STDOUT:     %addr: @TestIterate.%ptr (%ptr.44a) = addr_of %var
// CHECK:STDOUT:     %impl.elem3.loc26_19.1: @TestIterate.%.loc26_19.20 (%.b00) = impl_witness_access constants.%Iterate.lookup_impl_witness.9e0, element3 [symbolic = %impl.elem3.loc26_19.2 (constants.%impl.elem3.c16)]
// CHECK:STDOUT:     %bound_method.loc26_19.3: <bound method> = bound_method %r.ref, %impl.elem3.loc26_19.1
// CHECK:STDOUT:     %Iterate.facet.loc26_19.3: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.9e0) [symbolic = %Iterate.facet.loc26_19.5 (constants.%Iterate.facet.f88)]
// CHECK:STDOUT:     %.loc26_19.3: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc26_19.3 [symbolic = %Iterate.facet.loc26_19.5 (constants.%Iterate.facet.f88)]
// CHECK:STDOUT:     %Iterate.facet.loc26_19.4: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.9e0) [symbolic = %Iterate.facet.loc26_19.5 (constants.%Iterate.facet.f88)]
// CHECK:STDOUT:     %.loc26_19.4: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc26_19.4 [symbolic = %Iterate.facet.loc26_19.5 (constants.%Iterate.facet.f88)]
// CHECK:STDOUT:     %.loc26_19.5: %Destroy.type = converted constants.%as_type.ce3, constants.%impl.elem1.0cf [symbolic = %impl.elem1 (constants.%impl.elem1.0cf)]
// CHECK:STDOUT:     %.loc26_19.6: %Destroy.type = converted constants.%as_type.ce3, constants.%impl.elem1.0cf [symbolic = %impl.elem1 (constants.%impl.elem1.0cf)]
// CHECK:STDOUT:     %specific_impl_fn.loc26_19.2: <specific function> = specific_impl_function %impl.elem3.loc26_19.1, @Iterate.WithSelf.Next(constants.%Iterate.facet.f88) [symbolic = %specific_impl_fn.loc26_19.5 (constants.%specific_impl_fn.b6b)]
// CHECK:STDOUT:     %bound_method.loc26_19.4: <bound method> = bound_method %r.ref, %specific_impl_fn.loc26_19.2
// CHECK:STDOUT:     %.loc26_19.7: ref %Optional.049 = temporary_storage
// CHECK:STDOUT:     %Iterate.WithSelf.Next.call: init %Optional.049 to %.loc26_19.7 = call %bound_method.loc26_19.4(%r.ref, %addr)
// CHECK:STDOUT:     %.loc26_19.8: ref %Optional.049 = temporary %.loc26_19.7, %Iterate.WithSelf.Next.call
// CHECK:STDOUT:     %.loc26_19.9: %Optional.HasValue.type.794 = specific_constant imports.%Core.import_ref.bb6, @Optional(constants.%OptionalStorage.facet.405) [concrete = constants.%Optional.HasValue.776]
// CHECK:STDOUT:     %HasValue.ref: %Optional.HasValue.type.794 = name_ref HasValue, %.loc26_19.9 [concrete = constants.%Optional.HasValue.776]
// CHECK:STDOUT:     %Optional.HasValue.bound: <bound method> = bound_method %.loc26_19.8, %HasValue.ref
// CHECK:STDOUT:     %Optional.HasValue.specific_fn: <specific function> = specific_function %HasValue.ref, @Optional.HasValue(constants.%OptionalStorage.facet.405) [concrete = constants.%Optional.HasValue.specific_fn.9fc]
// CHECK:STDOUT:     %bound_method.loc26_19.5: <bound method> = bound_method %.loc26_19.8, %Optional.HasValue.specific_fn
// CHECK:STDOUT:     %.loc26_19.10: %Optional.049 = acquire_value %.loc26_19.8
// CHECK:STDOUT:     %Optional.HasValue.call: init bool = call %bound_method.loc26_19.5(%.loc26_19.10)
// CHECK:STDOUT:     %.loc26_19.11: bool = value_of_initializer %Optional.HasValue.call
// CHECK:STDOUT:     %.loc26_19.12: bool = converted %Optional.HasValue.call, %.loc26_19.11
// CHECK:STDOUT:     if %.loc26_19.12 br !for.body else br !for.done
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.body:
// CHECK:STDOUT:     %.loc26_19.13: %Optional.Get.type.e6c = specific_constant imports.%Core.import_ref.dab, @Optional(constants.%OptionalStorage.facet.405) [concrete = constants.%Optional.Get.cb6]
// CHECK:STDOUT:     %Get.ref: %Optional.Get.type.e6c = name_ref Get, %.loc26_19.13 [concrete = constants.%Optional.Get.cb6]
// CHECK:STDOUT:     %Optional.Get.bound: <bound method> = bound_method %.loc26_19.8, %Get.ref
// CHECK:STDOUT:     %Optional.Get.specific_fn: <specific function> = specific_function %Get.ref, @Optional.Get(constants.%OptionalStorage.facet.405) [concrete = constants.%Optional.Get.specific_fn.e69]
// CHECK:STDOUT:     %bound_method.loc26_19.6: <bound method> = bound_method %.loc26_19.8, %Optional.Get.specific_fn
// CHECK:STDOUT:     %.loc26_19.14: %Optional.049 = acquire_value %.loc26_19.8
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
// CHECK:STDOUT:     %Destroy.WithSelf.SelfDestruct.bound.loc26_19.1: <bound method> = bound_method %.loc26_19.8, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.9
// CHECK:STDOUT:     %Destroy.WithSelf.SelfDestruct.call.loc26_19.1: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc26_19.1(%.loc26_19.8)
// CHECK:STDOUT:     br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.done:
// CHECK:STDOUT:     %Destroy.WithSelf.SelfDestruct.bound.loc26_19.2: <bound method> = bound_method %.loc26_19.8, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.9
// CHECK:STDOUT:     %Destroy.WithSelf.SelfDestruct.call.loc26_19.2: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc26_19.2(%.loc26_19.8)
// CHECK:STDOUT:     %impl.elem2.loc26_19.2: @TestIterate.%.loc26_19.21 (%.056) = impl_witness_access constants.%Destroy.lookup_impl_witness.068, element2 [symbolic = %impl.elem2.loc26_19.4 (constants.%impl.elem2.5f8)]
// CHECK:STDOUT:     %bound_method.loc26_19.7: <bound method> = bound_method %var, %impl.elem2.loc26_19.2
// CHECK:STDOUT:     %.loc26_19.17: %Destroy.type = converted constants.%as_type.ce3, constants.%impl.elem1.0cf [symbolic = %impl.elem1 (constants.%impl.elem1.0cf)]
// CHECK:STDOUT:     %.loc26_19.18: %Destroy.type = converted constants.%as_type.ce3, constants.%impl.elem1.0cf [symbolic = %impl.elem1 (constants.%impl.elem1.0cf)]
// CHECK:STDOUT:     %specific_impl_fn.loc26_19.3: <specific function> = specific_impl_function %impl.elem2.loc26_19.2, @Destroy.WithSelf.SelfDestruct.1(constants.%impl.elem1.0cf) [symbolic = %specific_impl_fn.loc26_19.6 (constants.%specific_impl_fn.b61)]
// CHECK:STDOUT:     %bound_method.loc26_19.8: <bound method> = bound_method %var, %specific_impl_fn.loc26_19.3
// CHECK:STDOUT:     %Destroy.WithSelf.SelfDestruct.call.loc26_19.3: init %empty_tuple.type = call %bound_method.loc26_19.8(%var)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc26_19.1(%self.param: ref %.cc0) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc26_19.1(%self.param: ref %.cc0) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc26_19.1(%self.param: ref %.cc0) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.3(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.3(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc26_19.2(%self.param: ref %MaybeUnformed.8ff) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc26_19.2(%self.param: ref %MaybeUnformed.8ff) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc26_19.2(%self.param: ref %MaybeUnformed.8ff) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.4(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.4(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc26_19.3(%self.param: ref bool) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc26_19.3(%self.param: ref bool) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc26_19.3(%self.param: ref bool) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.5(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.5(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc26_19.4(%self.param: ref %struct_type.value.has_value.58e) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc26_19.1: ref %MaybeUnformed.8ff = struct_access %self.param, element0
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc26_19.1: <bound method> = bound_method %.loc26_19.1, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.5
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc26_19.1: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc26_19.1(%.loc26_19.1)
// CHECK:STDOUT:   %.loc26_19.2: ref bool = struct_access %self.param, element1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc26_19.2: <bound method> = bound_method %.loc26_19.2, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.6
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc26_19.2: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc26_19.2(%.loc26_19.2)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc26_19.4(%self.param: ref %struct_type.value.has_value.58e) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc26_19.4(%self.param: ref %struct_type.value.has_value.58e) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.6(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.6(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc26_19.5(%self.param: ref %DefaultOptionalStorage.3e2) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc26_19.5(%self.param: ref %DefaultOptionalStorage.3e2) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc26_19.5(%self.param: ref %DefaultOptionalStorage.3e2) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.7(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.7(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc26_19.6(%self.param: ref %Optional.049) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc26_19.6(%self.param: ref %Optional.049) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc26_19.6(%self.param: ref %Optional.049) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.8(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.8(%self.param)
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
// CHECK:STDOUT:   %impl.elem2: %.698 = impl_witness_access constants.%Iterate.impl_witness.6cb, element2 [concrete = constants.%T.as_type.as.Iterate.impl.NewCursor.fcd]
// CHECK:STDOUT:   %bound_method.loc68_19.1: <bound method> = bound_method %m.ref, %impl.elem2
// CHECK:STDOUT:   %facet_value.loc68_19.1: %CppRange_where.type.d682d6.1 = facet_value constants.%MutableRange, (constants.%custom_witness.f8f19d.3, constants.%Copy.impl_witness.0d6, constants.%custom_witness.902, constants.%custom_witness.e74, constants.%custom_witness.05b, constants.%custom_witness.491, constants.%custom_witness.ed8) [concrete = constants.%facet_value.efd]
// CHECK:STDOUT:   %.loc68_19.1: %CppRange_where.type.d682d6.1 = converted constants.%MutableRange, %facet_value.loc68_19.1 [concrete = constants.%facet_value.efd]
// CHECK:STDOUT:   %facet_value.loc68_19.2: %CppRange_where.type.d682d6.1 = facet_value constants.%MutableRange, (constants.%custom_witness.f8f19d.3, constants.%Copy.impl_witness.0d6, constants.%custom_witness.902, constants.%custom_witness.e74, constants.%custom_witness.05b, constants.%custom_witness.491, constants.%custom_witness.ed8) [concrete = constants.%facet_value.efd]
// CHECK:STDOUT:   %.loc68_19.2: %CppRange_where.type.d682d6.1 = converted constants.%MutableRange, %facet_value.loc68_19.2 [concrete = constants.%facet_value.efd]
// CHECK:STDOUT:   %specific_fn.loc68_19.1: <specific function> = specific_function %impl.elem2, @T.as_type.as.Iterate.impl.NewCursor(constants.%facet_value.efd) [concrete = constants.%T.as_type.as.Iterate.impl.NewCursor.specific_fn]
// CHECK:STDOUT:   %bound_method.loc68_19.2: <bound method> = bound_method %m.ref, %specific_fn.loc68_19.1
// CHECK:STDOUT:   %var: ref %tuple.type.dd0 = var_storage invalid
// CHECK:STDOUT:   %.loc68_18.1: %MutableRange = acquire_value %m.ref
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.call: init %tuple.type.dd0 to %var = call %bound_method.loc68_19.2(%.loc68_18.1)
// CHECK:STDOUT:   assign %var, %T.as_type.as.Iterate.impl.NewCursor.call
// CHECK:STDOUT:   br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT: !for.next:
// CHECK:STDOUT:   %addr: %ptr.a4f = addr_of %var
// CHECK:STDOUT:   %impl.elem3: %.782 = impl_witness_access constants.%Iterate.impl_witness.6cb, element3 [concrete = constants.%T.as_type.as.Iterate.impl.Next.c88]
// CHECK:STDOUT:   %bound_method.loc68_19.3: <bound method> = bound_method %m.ref, %impl.elem3
// CHECK:STDOUT:   %facet_value.loc68_19.3: %CppRange_where.type.d682d6.1 = facet_value constants.%MutableRange, (constants.%custom_witness.f8f19d.3, constants.%Copy.impl_witness.0d6, constants.%custom_witness.902, constants.%custom_witness.e74, constants.%custom_witness.05b, constants.%custom_witness.491, constants.%custom_witness.ed8) [concrete = constants.%facet_value.efd]
// CHECK:STDOUT:   %.loc68_19.3: %CppRange_where.type.d682d6.1 = converted constants.%MutableRange, %facet_value.loc68_19.3 [concrete = constants.%facet_value.efd]
// CHECK:STDOUT:   %facet_value.loc68_19.4: %CppRange_where.type.d682d6.1 = facet_value constants.%MutableRange, (constants.%custom_witness.f8f19d.3, constants.%Copy.impl_witness.0d6, constants.%custom_witness.902, constants.%custom_witness.e74, constants.%custom_witness.05b, constants.%custom_witness.491, constants.%custom_witness.ed8) [concrete = constants.%facet_value.efd]
// CHECK:STDOUT:   %.loc68_19.4: %CppRange_where.type.d682d6.1 = converted constants.%MutableRange, %facet_value.loc68_19.4 [concrete = constants.%facet_value.efd]
// CHECK:STDOUT:   %specific_fn.loc68_19.2: <specific function> = specific_function %impl.elem3, @T.as_type.as.Iterate.impl.Next(constants.%facet_value.efd) [concrete = constants.%T.as_type.as.Iterate.impl.Next.specific_fn]
// CHECK:STDOUT:   %bound_method.loc68_19.4: <bound method> = bound_method %m.ref, %specific_fn.loc68_19.2
// CHECK:STDOUT:   %.loc68_19.5: ref %Optional.049 = temporary_storage
// CHECK:STDOUT:   %.loc68_18.2: %MutableRange = acquire_value %m.ref
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.call: init %Optional.049 to %.loc68_19.5 = call %bound_method.loc68_19.4(%.loc68_18.2, %addr)
// CHECK:STDOUT:   %.loc68_19.6: ref %Optional.049 = temporary %.loc68_19.5, %T.as_type.as.Iterate.impl.Next.call
// CHECK:STDOUT:   %.loc68_19.7: %Optional.HasValue.type.794 = specific_constant imports.%Core.import_ref.bb6, @Optional(constants.%OptionalStorage.facet.405) [concrete = constants.%Optional.HasValue.776]
// CHECK:STDOUT:   %HasValue.ref: %Optional.HasValue.type.794 = name_ref HasValue, %.loc68_19.7 [concrete = constants.%Optional.HasValue.776]
// CHECK:STDOUT:   %Optional.HasValue.bound: <bound method> = bound_method %.loc68_19.6, %HasValue.ref
// CHECK:STDOUT:   %Optional.HasValue.specific_fn: <specific function> = specific_function %HasValue.ref, @Optional.HasValue(constants.%OptionalStorage.facet.405) [concrete = constants.%Optional.HasValue.specific_fn.9fc]
// CHECK:STDOUT:   %bound_method.loc68_19.5: <bound method> = bound_method %.loc68_19.6, %Optional.HasValue.specific_fn
// CHECK:STDOUT:   %.loc68_19.8: %Optional.049 = acquire_value %.loc68_19.6
// CHECK:STDOUT:   %Optional.HasValue.call: init bool = call %bound_method.loc68_19.5(%.loc68_19.8)
// CHECK:STDOUT:   %.loc68_19.9: bool = value_of_initializer %Optional.HasValue.call
// CHECK:STDOUT:   %.loc68_19.10: bool = converted %Optional.HasValue.call, %.loc68_19.9
// CHECK:STDOUT:   if %.loc68_19.10 br !for.body else br !for.done
// CHECK:STDOUT:
// CHECK:STDOUT: !for.body:
// CHECK:STDOUT:   %.loc68_19.11: %Optional.Get.type.e6c = specific_constant imports.%Core.import_ref.dab, @Optional(constants.%OptionalStorage.facet.405) [concrete = constants.%Optional.Get.cb6]
// CHECK:STDOUT:   %Get.ref: %Optional.Get.type.e6c = name_ref Get, %.loc68_19.11 [concrete = constants.%Optional.Get.cb6]
// CHECK:STDOUT:   %Optional.Get.bound: <bound method> = bound_method %.loc68_19.6, %Get.ref
// CHECK:STDOUT:   %Optional.Get.specific_fn: <specific function> = specific_function %Get.ref, @Optional.Get(constants.%OptionalStorage.facet.405) [concrete = constants.%Optional.Get.specific_fn.e69]
// CHECK:STDOUT:   %bound_method.loc68_19.6: <bound method> = bound_method %.loc68_19.6, %Optional.Get.specific_fn
// CHECK:STDOUT:   %.loc68_19.12: %Optional.049 = acquire_value %.loc68_19.6
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
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc68_19.1: <bound method> = bound_method %.loc68_19.6, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.9
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc68_19.1: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc68_19.1(%.loc68_19.6)
// CHECK:STDOUT:   br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT: !for.done:
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc68_19.2: <bound method> = bound_method %.loc68_19.6, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.9
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc68_19.2: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc68_19.2(%.loc68_19.6)
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc68_19.3: <bound method> = bound_method %var, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.11
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc68_19.3: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc68_19.3(%var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestIterate(constants.%R) {
// CHECK:STDOUT:   %R.patt.loc22_17.2 => constants.%R.patt
// CHECK:STDOUT:   %R.loc22_17.1 => constants.%R
// CHECK:STDOUT:   %R.as_type.loc22_61.1 => constants.%R.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.6f7
// CHECK:STDOUT:   %r.param_patt.loc22_59.2 => constants.%r.param_patt.19a
// CHECK:STDOUT:   %r.patt.loc22_59.2 => constants.%r.patt.f97
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestIterate(constants.%facet_value.30f) {
// CHECK:STDOUT:   %R.patt.loc22_17.2 => constants.%R.patt
// CHECK:STDOUT:   %R.loc22_17.1 => constants.%facet_value.30f
// CHECK:STDOUT:   %R.as_type.loc22_61.1 => constants.%MutableRange
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.992
// CHECK:STDOUT:   %r.param_patt.loc22_59.2 => constants.%r.param_patt.aeb
// CHECK:STDOUT:   %r.patt.loc22_59.2 => constants.%r.patt.892
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc22 => constants.%complete_type.357
// CHECK:STDOUT:   %Iterate.lookup_impl_witness => constants.%Iterate.impl_witness.6cb
// CHECK:STDOUT:   %Iterate.facet.loc26_19.5 => constants.%Iterate.facet.d07
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type => constants.%Iterate.WithSelf.NewCursor.type.0ef
// CHECK:STDOUT:   %.loc26_19.19 => constants.%.698
// CHECK:STDOUT:   %impl.elem2.loc26_19.3 => constants.%T.as_type.as.Iterate.impl.NewCursor.fcd
// CHECK:STDOUT:   %specific_impl_fn.loc26_19.4 => constants.%T.as_type.as.Iterate.impl.NewCursor.specific_fn
// CHECK:STDOUT:   %impl.elem1 => constants.%Destroy.facet.f0b
// CHECK:STDOUT:   %as_type => constants.%tuple.type.dd0
// CHECK:STDOUT:   %require_complete.1 => constants.%complete_type.843
// CHECK:STDOUT:   %ptr => constants.%ptr.a4f
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type => constants.%Iterate.WithSelf.Next.type.7af
// CHECK:STDOUT:   %.loc26_19.20 => constants.%.782
// CHECK:STDOUT:   %impl.elem3.loc26_19.2 => constants.%T.as_type.as.Iterate.impl.Next.c88
// CHECK:STDOUT:   %specific_impl_fn.loc26_19.5 => constants.%T.as_type.as.Iterate.impl.Next.specific_fn
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type => constants.%Destroy.WithSelf.SelfDestruct.type.e6e
// CHECK:STDOUT:   %.loc26_19.21 => constants.%.94d
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.f8f19d.10
// CHECK:STDOUT:   %impl.elem2.loc26_19.4 => constants.%Destroy.WithSelf.SelfDestruct.db3fdb.11
// CHECK:STDOUT:   %specific_impl_fn.loc26_19.6 => constants.%Destroy.WithSelf.SelfDestruct.db3fdb.11
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- adl_returns_different_types.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %Iterate.type: type = facet_type <@Iterate> [concrete]
// CHECK:STDOUT:   %OptionalStorage.type: type = facet_type <@OptionalStorage> [concrete]
// CHECK:STDOUT:   %pattern_type.831: type = pattern_type bool [concrete]
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
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.2: type = fn_type @Destroy.WithSelf.Op.loc27 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.2: %Destroy.WithSelf.Op.type.ef016f.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.2: type = fn_type @Destroy.WithSelf.SelfDestruct.loc27 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.2: %Destroy.WithSelf.SelfDestruct.type.fbceb5.2 = struct_value () [concrete]
// CHECK:STDOUT:   %ValueType.SubobjectDestroy.type: type = fn_type @ValueType.SubobjectDestroy [concrete]
// CHECK:STDOUT:   %ValueType.SubobjectDestroy: %ValueType.SubobjectDestroy.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.226: <witness> = custom_witness (%Destroy.WithSelf.Op.403171.2, %ValueType.SubobjectDestroy, %Destroy.WithSelf.SelfDestruct.db3fdb.2), @Destroy [concrete]
// CHECK:STDOUT:   %const.dd6: type = const_type %ValueType [concrete]
// CHECK:STDOUT:   %ptr.69c: type = ptr_type %const.dd6 [concrete]
// CHECK:STDOUT:   %ptr.233: type = ptr_type %ValueType [concrete]
// CHECK:STDOUT:   %ValueType.Op.type: type = fn_type @ValueType.Op [concrete]
// CHECK:STDOUT:   %ValueType.Op: %ValueType.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.202: <witness> = custom_witness (%ValueType.Op), @Copy [concrete]
// CHECK:STDOUT:   %facet_value.978: %facet_type.f48 = facet_value %ValueType, (%custom_witness.226, %custom_witness.202) [concrete]
// CHECK:STDOUT:   %Iterate.lookup_impl_witness.1d8: <witness> = lookup_impl_witness %.Self.500, @Iterate [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.294: %facet_type.f48 = impl_witness_access %Iterate.lookup_impl_witness.1d8, element0 [symbolic_self]
// CHECK:STDOUT:   %Iterate_where.type.87e: type = facet_type <@Iterate where %impl.elem0.294 = %facet_value.978> [concrete]
// CHECK:STDOUT:   %pattern_type.ba8: type = pattern_type %Iterate_where.type.87e [concrete]
// CHECK:STDOUT:   %R.patt: %pattern_type.ba8 = symbolic_binding_pattern R, 0 [symbolic]
// CHECK:STDOUT:   %R: %Iterate_where.type.87e = symbolic_binding R, 0 [symbolic]
// CHECK:STDOUT:   %R.as_type: type = facet_access_type %R [symbolic]
// CHECK:STDOUT:   %pattern_type.053: type = pattern_type %R.as_type [symbolic]
// CHECK:STDOUT:   %r.param_patt.75e: %pattern_type.053 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %r.patt.82a: %pattern_type.053 = wrapper_binding_pattern r, %r.param_patt.75e [symbolic]
// CHECK:STDOUT:   %i.patt: %pattern_type.c43 = value_binding_pattern i [concrete]
// CHECK:STDOUT:   %Iterate.lookup_impl_witness.01e: <witness> = lookup_impl_witness %R, @Iterate [symbolic]
// CHECK:STDOUT:   %Iterate.facet.467: %Iterate.type = facet_value %R.as_type, (%Iterate.lookup_impl_witness.01e) [symbolic]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.6eb: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.467) [symbolic]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.bb3: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.467) [symbolic]
// CHECK:STDOUT:   %.9d7: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.6eb, %Iterate.facet.467 [symbolic]
// CHECK:STDOUT:   %impl.elem2.b2a: %.9d7 = impl_witness_access %Iterate.lookup_impl_witness.01e, element2 [symbolic]
// CHECK:STDOUT:   %impl.elem1.d56: %Destroy.type = impl_witness_access %Iterate.lookup_impl_witness.01e, element1 [symbolic]
// CHECK:STDOUT:   %as_type.db9: type = facet_access_type %impl.elem1.d56 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.2f4: <specific function> = specific_impl_function %impl.elem2.b2a, @Iterate.WithSelf.NewCursor(%Iterate.facet.467) [symbolic]
// CHECK:STDOUT:   %ptr.eca: type = ptr_type %as_type.db9 [symbolic]
// CHECK:STDOUT:   %.2d6: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.bb3, %Iterate.facet.467 [symbolic]
// CHECK:STDOUT:   %impl.elem3.c97: %.2d6 = impl_witness_access %Iterate.lookup_impl_witness.01e, element3 [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.53f: type = class_type @DefaultOptionalStorage, @DefaultOptionalStorage(%facet_value.978) [concrete]
// CHECK:STDOUT:   %OptionalStorage.impl_witness.6c3: <witness> = impl_witness imports.%OptionalStorage.impl_witness_table.460, @T.as_type.as.OptionalStorage.impl(%facet_value.978) [concrete]
// CHECK:STDOUT:   %OptionalStorage.facet.b3c: %OptionalStorage.type = facet_value %ValueType, (%OptionalStorage.impl_witness.6c3) [concrete]
// CHECK:STDOUT:   %Optional.e69: type = class_type @Optional, @Optional(%OptionalStorage.facet.b3c) [concrete]
// CHECK:STDOUT:   %pattern_type.62b: type = pattern_type %Optional.e69 [concrete]
// CHECK:STDOUT:   %specific_impl_fn.5ea: <specific function> = specific_impl_function %impl.elem3.c97, @Iterate.WithSelf.Next(%Iterate.facet.467) [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.type.c61: type = fn_type @Optional.HasValue, @Optional(%OptionalStorage.facet.b3c) [concrete]
// CHECK:STDOUT:   %Optional.HasValue.ba9d: %Optional.HasValue.type.c61 = struct_value () [concrete]
// CHECK:STDOUT:   %Optional.Get.type.394: type = fn_type @Optional.Get, @Optional(%OptionalStorage.facet.b3c) [concrete]
// CHECK:STDOUT:   %Optional.Get.da9: %Optional.Get.type.394 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.facet.c1d: %Destroy.type = facet_value %ValueType, (%custom_witness.226) [concrete]
// CHECK:STDOUT:   %MaybeUnformed.e45: type = class_type @MaybeUnformed, @MaybeUnformed(%Destroy.facet.c1d) [concrete]
// CHECK:STDOUT:   %.be1: type = maybe_unformed_type %ValueType [concrete]
// CHECK:STDOUT:   %struct_type.value.has_value.619: type = struct_type {.value: %MaybeUnformed.e45, .has_value: bool} [concrete]
// CHECK:STDOUT:   %Optional.HasValue.specific_fn.524: <specific function> = specific_function %Optional.HasValue.ba9d, @Optional.HasValue(%OptionalStorage.facet.b3c) [concrete]
// CHECK:STDOUT:   %Optional.Get.specific_fn.f41: <specific function> = specific_function %Optional.Get.da9, @Optional.Get(%OptionalStorage.facet.b3c) [concrete]
// CHECK:STDOUT:   %operator_PlusEqual__carbon_thunk.type: type = fn_type @operator_PlusEqual__carbon_thunk [concrete]
// CHECK:STDOUT:   %operator_PlusEqual__carbon_thunk: %operator_PlusEqual__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.ce5: type = pattern_type %.be1 [concrete]
// CHECK:STDOUT:   %self.param_patt.36c: %pattern_type.ce5 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.548: %pattern_type.ce5 = wrapper_binding_pattern self, %self.param_patt.36c [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc31_37.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.2: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.3: type = fn_type @Destroy.WithSelf.Op.loc31_37.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.3: %Destroy.WithSelf.Op.type.ef016f.3 = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.759: type = pattern_type %MaybeUnformed.e45 [concrete]
// CHECK:STDOUT:   %self.param_patt.039: %pattern_type.759 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.4c2: %pattern_type.759 = wrapper_binding_pattern self, %self.param_patt.039 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.3: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc31_37.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.3: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.3 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.4: type = fn_type @Destroy.WithSelf.Op.loc31_37.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.4: %Destroy.WithSelf.Op.type.ef016f.4 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.4: type = fn_type @Destroy.WithSelf.SelfDestruct.loc31_37.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.4: %Destroy.WithSelf.SelfDestruct.type.fbceb5.4 = struct_value () [concrete]
// CHECK:STDOUT:   %self.param_patt.40d: %pattern_type.831 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.560: %pattern_type.831 = wrapper_binding_pattern self, %self.param_patt.40d [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.4: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc31_37.3 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.4: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.4 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.5: type = fn_type @Destroy.WithSelf.Op.loc31_37.3 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.5: %Destroy.WithSelf.Op.type.ef016f.5 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.5: type = fn_type @Destroy.WithSelf.SelfDestruct.loc31_37.3 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.5: %Destroy.WithSelf.SelfDestruct.type.fbceb5.5 = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.a92: type = pattern_type %struct_type.value.has_value.619 [concrete]
// CHECK:STDOUT:   %self.param_patt.565: %pattern_type.a92 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.1d7: %pattern_type.a92 = wrapper_binding_pattern self, %self.param_patt.565 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.5: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc31_37.4 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.5: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.5 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.6: type = fn_type @Destroy.WithSelf.Op.loc31_37.4 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.6: %Destroy.WithSelf.Op.type.ef016f.6 = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.fdb: type = pattern_type %DefaultOptionalStorage.53f [concrete]
// CHECK:STDOUT:   %self.param_patt.ace: %pattern_type.fdb = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.b05: %pattern_type.fdb = wrapper_binding_pattern self, %self.param_patt.ace [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.6: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc31_37.5 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.6: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.6 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.7: type = fn_type @Destroy.WithSelf.Op.loc31_37.5 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.7: %Destroy.WithSelf.Op.type.ef016f.7 = struct_value () [concrete]
// CHECK:STDOUT:   %self.param_patt.a8e: %pattern_type.62b = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.24c: %pattern_type.62b = wrapper_binding_pattern self, %self.param_patt.a8e [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.7: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc31_37.6 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.7: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.7 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.8: type = fn_type @Destroy.WithSelf.Op.loc31_37.6 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.8: %Destroy.WithSelf.Op.type.ef016f.8 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.8: type = fn_type @Destroy.WithSelf.SelfDestruct.loc31_37.6 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.8: %Destroy.WithSelf.SelfDestruct.type.fbceb5.8 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.d0e: <witness> = lookup_impl_witness %impl.elem1.d56, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.c82: type = fn_type @Destroy.WithSelf.SelfDestruct.1, @Destroy.WithSelf(%impl.elem1.d56) [symbolic]
// CHECK:STDOUT:   %.c10: type = fn_type_with_self_type %Destroy.WithSelf.SelfDestruct.type.c82, %impl.elem1.d56 [symbolic]
// CHECK:STDOUT:   %impl.elem2.5c5: %.c10 = impl_witness_access %Destroy.lookup_impl_witness.d0e, element2 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.17d: <specific function> = specific_impl_function %impl.elem2.5c5, @Destroy.WithSelf.SelfDestruct.1(%impl.elem1.d56) [symbolic]
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
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.9: type = fn_type @Destroy.WithSelf.Op.3 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.9: %Destroy.WithSelf.Op.type.ef016f.9 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.9: type = fn_type @Destroy.WithSelf.SelfDestruct.3 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.9: %Destroy.WithSelf.SelfDestruct.type.fbceb5.9 = struct_value () [concrete]
// CHECK:STDOUT:   %Iterator.SubobjectDestroy.type: type = fn_type @Iterator.SubobjectDestroy [concrete]
// CHECK:STDOUT:   %Iterator.SubobjectDestroy: %Iterator.SubobjectDestroy.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.429ac1.1: <witness> = custom_witness (%Destroy.WithSelf.Op.403171.9, %Iterator.SubobjectDestroy, %Destroy.WithSelf.SelfDestruct.db3fdb.9), @Destroy [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.3f6dc2.2: type = fn_type @Iterator.Op.2 [concrete]
// CHECK:STDOUT:   %Iterator.Op.240e50.2: %Iterator.Op.type.3f6dc2.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.c31: <witness> = custom_witness (%Iterator.Op.240e50.2), @Inc [concrete]
// CHECK:STDOUT:   %Equal.type: type = fn_type @Equal [concrete]
// CHECK:STDOUT:   %Equal: %Equal.type = struct_value () [concrete]
// CHECK:STDOUT:   %NotEqual.type: type = fn_type @NotEqual [concrete]
// CHECK:STDOUT:   %NotEqual: %NotEqual.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.e21: <witness> = custom_witness (%Equal, %NotEqual), @EqWith, @EqWith(%Sentinel) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.10: type = fn_type @Destroy.WithSelf.Op.loc38 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.10: %Destroy.WithSelf.Op.type.ef016f.10 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.10: type = fn_type @Destroy.WithSelf.SelfDestruct.loc38 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.10: %Destroy.WithSelf.SelfDestruct.type.fbceb5.10 = struct_value () [concrete]
// CHECK:STDOUT:   %Sentinel.SubobjectDestroy.type: type = fn_type @Sentinel.SubobjectDestroy [concrete]
// CHECK:STDOUT:   %Sentinel.SubobjectDestroy: %Sentinel.SubobjectDestroy.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.429ac1.2: <witness> = custom_witness (%Destroy.WithSelf.Op.403171.10, %Sentinel.SubobjectDestroy, %Destroy.WithSelf.SelfDestruct.db3fdb.10), @Destroy [concrete]
// CHECK:STDOUT:   %facet_value.e8b: %CppRange_where.type.d682d6.1 = facet_value %ConstRange, (%custom_witness.226, %custom_witness.202, %custom_witness.116, %custom_witness.429ac1.1, %custom_witness.5be, %custom_witness.c31, %custom_witness.e21, %custom_witness.429ac1.2) [concrete]
// CHECK:STDOUT:   %tuple.type.7a6: type = tuple_type (%Iterator, %Sentinel) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.8: type = fn_type @Destroy.WithSelf.SubobjectDestroy.3 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.8: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.8 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.11: type = fn_type @Destroy.WithSelf.Op.4 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.11: %Destroy.WithSelf.Op.type.ef016f.11 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.11: type = fn_type @Destroy.WithSelf.SelfDestruct.4 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.11: %Destroy.WithSelf.SelfDestruct.type.fbceb5.11 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.f8f19d.8: <witness> = custom_witness (%Destroy.WithSelf.Op.403171.11, %Destroy.WithSelf.SubobjectDestroy.d01daf.8, %Destroy.WithSelf.SelfDestruct.db3fdb.11), @Destroy [concrete]
// CHECK:STDOUT:   %Destroy.facet.d83: %Destroy.type = facet_value %tuple.type.7a6, (%custom_witness.f8f19d.8) [concrete]
// CHECK:STDOUT:   %Iterate.impl_witness.6b0: <witness> = impl_witness imports.%Iterate.impl_witness_table.121, @T.as_type.as.Iterate.impl(%facet_value.e8b) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.type.0f8: type = fn_type @T.as_type.as.Iterate.impl.NewCursor, @T.as_type.as.Iterate.impl(%facet_value.e8b) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.fa5: %T.as_type.as.Iterate.impl.NewCursor.type.0f8 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.type.e47: type = fn_type @T.as_type.as.Iterate.impl.Next, @T.as_type.as.Iterate.impl(%facet_value.e8b) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.e9b: %T.as_type.as.Iterate.impl.Next.type.e47 = struct_value () [concrete]
// CHECK:STDOUT:   %Iterate.facet.80d: %Iterate.type = facet_value %ConstRange, (%Iterate.impl_witness.6b0) [concrete]
// CHECK:STDOUT:   %facet_value.c77: %Iterate_where.type.87e = facet_value %ConstRange, (%Iterate.impl_witness.6b0) [concrete]
// CHECK:STDOUT:   %r.param_patt.ac5: %pattern_type.c1d = value_param_pattern [concrete]
// CHECK:STDOUT:   %r.patt.97f: %pattern_type.c1d = wrapper_binding_pattern r, %r.param_patt.ac5 [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.c59: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.80d) [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.5e6: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.80d) [concrete]
// CHECK:STDOUT:   %.182: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.c59, %Iterate.facet.80d [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.specific_fn: <specific function> = specific_function %T.as_type.as.Iterate.impl.NewCursor.fa5, @T.as_type.as.Iterate.impl.NewCursor(%facet_value.e8b) [concrete]
// CHECK:STDOUT:   %ptr.d33: type = ptr_type %tuple.type.7a6 [concrete]
// CHECK:STDOUT:   %.883: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.5e6, %Iterate.facet.80d [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.specific_fn: <specific function> = specific_function %T.as_type.as.Iterate.impl.Next.e9b, @T.as_type.as.Iterate.impl.Next(%facet_value.e8b) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.036: type = fn_type @Destroy.WithSelf.SelfDestruct.1, @Destroy.WithSelf(%Destroy.facet.d83) [concrete]
// CHECK:STDOUT:   %.6cc: type = fn_type_with_self_type %Destroy.WithSelf.SelfDestruct.type.036, %Destroy.facet.d83 [concrete]
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
// CHECK:STDOUT: generated {
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.1: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc31_37.1 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.2] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.ce5 = ref_param_pattern [concrete = constants.%self.param_patt.36c]
// CHECK:STDOUT:     %self.patt: %pattern_type.ce5 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.548]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %.be1 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %.be1 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.2: %Destroy.WithSelf.Op.type.ef016f.3 = fn_decl @Destroy.WithSelf.Op.loc31_37.1 [concrete = constants.%Destroy.WithSelf.Op.403171.3] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.ce5 = ref_param_pattern [concrete = constants.%self.param_patt.36c]
// CHECK:STDOUT:     %self.patt: %pattern_type.ce5 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.548]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %.be1 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %.be1 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.2: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.3 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc31_37.2 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.3] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.759 = ref_param_pattern [concrete = constants.%self.param_patt.039]
// CHECK:STDOUT:     %self.patt: %pattern_type.759 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.4c2]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %MaybeUnformed.e45 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %MaybeUnformed.e45 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.3: %Destroy.WithSelf.Op.type.ef016f.4 = fn_decl @Destroy.WithSelf.Op.loc31_37.2 [concrete = constants.%Destroy.WithSelf.Op.403171.4] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.759 = ref_param_pattern [concrete = constants.%self.param_patt.039]
// CHECK:STDOUT:     %self.patt: %pattern_type.759 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.4c2]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %MaybeUnformed.e45 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %MaybeUnformed.e45 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.3: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.4 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc31_37.3 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.4] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.831 = ref_param_pattern [concrete = constants.%self.param_patt.40d]
// CHECK:STDOUT:     %self.patt: %pattern_type.831 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.560]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref bool = ref_param call_param0
// CHECK:STDOUT:     %self: ref bool = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.4: %Destroy.WithSelf.Op.type.ef016f.5 = fn_decl @Destroy.WithSelf.Op.loc31_37.3 [concrete = constants.%Destroy.WithSelf.Op.403171.5] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.831 = ref_param_pattern [concrete = constants.%self.param_patt.40d]
// CHECK:STDOUT:     %self.patt: %pattern_type.831 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.560]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref bool = ref_param call_param0
// CHECK:STDOUT:     %self: ref bool = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.4: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.5 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc31_37.4 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.5] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.a92 = ref_param_pattern [concrete = constants.%self.param_patt.565]
// CHECK:STDOUT:     %self.patt: %pattern_type.a92 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.1d7]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %struct_type.value.has_value.619 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %struct_type.value.has_value.619 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.5: %Destroy.WithSelf.Op.type.ef016f.6 = fn_decl @Destroy.WithSelf.Op.loc31_37.4 [concrete = constants.%Destroy.WithSelf.Op.403171.6] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.a92 = ref_param_pattern [concrete = constants.%self.param_patt.565]
// CHECK:STDOUT:     %self.patt: %pattern_type.a92 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.1d7]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %struct_type.value.has_value.619 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %struct_type.value.has_value.619 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.5: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.6 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc31_37.5 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.6] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.fdb = ref_param_pattern [concrete = constants.%self.param_patt.ace]
// CHECK:STDOUT:     %self.patt: %pattern_type.fdb = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.b05]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %DefaultOptionalStorage.53f = ref_param call_param0
// CHECK:STDOUT:     %self: ref %DefaultOptionalStorage.53f = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.6: %Destroy.WithSelf.Op.type.ef016f.7 = fn_decl @Destroy.WithSelf.Op.loc31_37.5 [concrete = constants.%Destroy.WithSelf.Op.403171.7] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.fdb = ref_param_pattern [concrete = constants.%self.param_patt.ace]
// CHECK:STDOUT:     %self.patt: %pattern_type.fdb = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.b05]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %DefaultOptionalStorage.53f = ref_param call_param0
// CHECK:STDOUT:     %self: ref %DefaultOptionalStorage.53f = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.6: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.7 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc31_37.6 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.7] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.62b = ref_param_pattern [concrete = constants.%self.param_patt.a8e]
// CHECK:STDOUT:     %self.patt: %pattern_type.62b = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.24c]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %Optional.e69 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %Optional.e69 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.7: %Destroy.WithSelf.Op.type.ef016f.8 = fn_decl @Destroy.WithSelf.Op.loc31_37.6 [concrete = constants.%Destroy.WithSelf.Op.403171.8] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.62b = ref_param_pattern [concrete = constants.%self.param_patt.a8e]
// CHECK:STDOUT:     %self.patt: %pattern_type.62b = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.24c]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %Optional.e69 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %Optional.e69 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @TestIterate(%R.loc27_17.2: %Iterate_where.type.87e) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Iterate.lookup_impl_witness: <witness> = lookup_impl_witness %R.loc27_17.1, @Iterate [symbolic = %Iterate.lookup_impl_witness (constants.%Iterate.lookup_impl_witness.01e)]
// CHECK:STDOUT:   %Iterate.facet.loc31_37.5: %Iterate.type = facet_value %R.as_type.loc27_79.1, (%Iterate.lookup_impl_witness) [symbolic = %Iterate.facet.loc31_37.5 (constants.%Iterate.facet.467)]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.loc31_37.5) [symbolic = %Iterate.WithSelf.NewCursor.type (constants.%Iterate.WithSelf.NewCursor.type.6eb)]
// CHECK:STDOUT:   %.loc31_37.20: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type, %Iterate.facet.loc31_37.5 [symbolic = %.loc31_37.20 (constants.%.9d7)]
// CHECK:STDOUT:   %impl.elem2.loc31_37.3: @TestIterate.%.loc31_37.20 (%.9d7) = impl_witness_access %Iterate.lookup_impl_witness, element2 [symbolic = %impl.elem2.loc31_37.3 (constants.%impl.elem2.b2a)]
// CHECK:STDOUT:   %specific_impl_fn.loc31_37.4: <specific function> = specific_impl_function %impl.elem2.loc31_37.3, @Iterate.WithSelf.NewCursor(%Iterate.facet.loc31_37.5) [symbolic = %specific_impl_fn.loc31_37.4 (constants.%specific_impl_fn.2f4)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %ptr: type = ptr_type %as_type [symbolic = %ptr (constants.%ptr.eca)]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.loc31_37.5) [symbolic = %Iterate.WithSelf.Next.type (constants.%Iterate.WithSelf.Next.type.bb3)]
// CHECK:STDOUT:   %.loc31_37.21: type = fn_type_with_self_type %Iterate.WithSelf.Next.type, %Iterate.facet.loc31_37.5 [symbolic = %.loc31_37.21 (constants.%.2d6)]
// CHECK:STDOUT:   %impl.elem3.loc31_37.2: @TestIterate.%.loc31_37.21 (%.2d6) = impl_witness_access %Iterate.lookup_impl_witness, element3 [symbolic = %impl.elem3.loc31_37.2 (constants.%impl.elem3.c97)]
// CHECK:STDOUT:   %specific_impl_fn.loc31_37.5: <specific function> = specific_impl_function %impl.elem3.loc31_37.2, @Iterate.WithSelf.Next(%Iterate.facet.loc31_37.5) [symbolic = %specific_impl_fn.loc31_37.5 (constants.%specific_impl_fn.5ea)]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type: type = fn_type @Destroy.WithSelf.SelfDestruct.1, @Destroy.WithSelf(%impl.elem1) [symbolic = %Destroy.WithSelf.SelfDestruct.type (constants.%Destroy.WithSelf.SelfDestruct.type.c82)]
// CHECK:STDOUT:   %.loc31_37.22: type = fn_type_with_self_type %Destroy.WithSelf.SelfDestruct.type, %impl.elem1 [symbolic = %.loc31_37.22 (constants.%.c10)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %impl.elem1, @Destroy [symbolic = %Destroy.lookup_impl_witness (constants.%Destroy.lookup_impl_witness.d0e)]
// CHECK:STDOUT:   %impl.elem2.loc31_37.4: @TestIterate.%.loc31_37.22 (%.c10) = impl_witness_access %Destroy.lookup_impl_witness, element2 [symbolic = %impl.elem2.loc31_37.4 (constants.%impl.elem2.5c5)]
// CHECK:STDOUT:   %specific_impl_fn.loc31_37.6: <specific function> = specific_impl_function %impl.elem2.loc31_37.4, @Destroy.WithSelf.SelfDestruct.1(%impl.elem1) [symbolic = %specific_impl_fn.loc31_37.6 (constants.%specific_impl_fn.17d)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%r.param: @TestIterate.%R.as_type.loc27_79.1 (%R.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %i.patt: %pattern_type.c43 = value_binding_pattern i [concrete = constants.%i.patt]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %r.ref: @TestIterate.%R.as_type.loc27_79.1 (%R.as_type) = name_ref r, %r
// CHECK:STDOUT:     %impl.elem2.loc31_37.1: @TestIterate.%.loc31_37.20 (%.9d7) = impl_witness_access constants.%Iterate.lookup_impl_witness.01e, element2 [symbolic = %impl.elem2.loc31_37.3 (constants.%impl.elem2.b2a)]
// CHECK:STDOUT:     %bound_method.loc31_37.1: <bound method> = bound_method %r.ref, %impl.elem2.loc31_37.1
// CHECK:STDOUT:     %Iterate.facet.loc31_37.1: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.01e) [symbolic = %Iterate.facet.loc31_37.5 (constants.%Iterate.facet.467)]
// CHECK:STDOUT:     %.loc31_37.1: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc31_37.1 [symbolic = %Iterate.facet.loc31_37.5 (constants.%Iterate.facet.467)]
// CHECK:STDOUT:     %Iterate.facet.loc31_37.2: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.01e) [symbolic = %Iterate.facet.loc31_37.5 (constants.%Iterate.facet.467)]
// CHECK:STDOUT:     %.loc31_37.2: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc31_37.2 [symbolic = %Iterate.facet.loc31_37.5 (constants.%Iterate.facet.467)]
// CHECK:STDOUT:     %specific_impl_fn.loc31_37.1: <specific function> = specific_impl_function %impl.elem2.loc31_37.1, @Iterate.WithSelf.NewCursor(constants.%Iterate.facet.467) [symbolic = %specific_impl_fn.loc31_37.4 (constants.%specific_impl_fn.2f4)]
// CHECK:STDOUT:     %bound_method.loc31_37.2: <bound method> = bound_method %r.ref, %specific_impl_fn.loc31_37.1
// CHECK:STDOUT:     %var: ref @TestIterate.%as_type (%as_type.db9) = var_storage invalid
// CHECK:STDOUT:     %Iterate.WithSelf.NewCursor.call: init @TestIterate.%as_type (%as_type.db9) to %var = call %bound_method.loc31_37.2(%r.ref)
// CHECK:STDOUT:     assign %var, %Iterate.WithSelf.NewCursor.call
// CHECK:STDOUT:     br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.next:
// CHECK:STDOUT:     %addr.loc31: @TestIterate.%ptr (%ptr.eca) = addr_of %var
// CHECK:STDOUT:     %impl.elem3.loc31_37.1: @TestIterate.%.loc31_37.21 (%.2d6) = impl_witness_access constants.%Iterate.lookup_impl_witness.01e, element3 [symbolic = %impl.elem3.loc31_37.2 (constants.%impl.elem3.c97)]
// CHECK:STDOUT:     %bound_method.loc31_37.3: <bound method> = bound_method %r.ref, %impl.elem3.loc31_37.1
// CHECK:STDOUT:     %Iterate.facet.loc31_37.3: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.01e) [symbolic = %Iterate.facet.loc31_37.5 (constants.%Iterate.facet.467)]
// CHECK:STDOUT:     %.loc31_37.3: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc31_37.3 [symbolic = %Iterate.facet.loc31_37.5 (constants.%Iterate.facet.467)]
// CHECK:STDOUT:     %Iterate.facet.loc31_37.4: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.01e) [symbolic = %Iterate.facet.loc31_37.5 (constants.%Iterate.facet.467)]
// CHECK:STDOUT:     %.loc31_37.4: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc31_37.4 [symbolic = %Iterate.facet.loc31_37.5 (constants.%Iterate.facet.467)]
// CHECK:STDOUT:     %.loc31_37.5: %Destroy.type = converted constants.%as_type.db9, constants.%impl.elem1.d56 [symbolic = %impl.elem1 (constants.%impl.elem1.d56)]
// CHECK:STDOUT:     %.loc31_37.6: %Destroy.type = converted constants.%as_type.db9, constants.%impl.elem1.d56 [symbolic = %impl.elem1 (constants.%impl.elem1.d56)]
// CHECK:STDOUT:     %specific_impl_fn.loc31_37.2: <specific function> = specific_impl_function %impl.elem3.loc31_37.1, @Iterate.WithSelf.Next(constants.%Iterate.facet.467) [symbolic = %specific_impl_fn.loc31_37.5 (constants.%specific_impl_fn.5ea)]
// CHECK:STDOUT:     %bound_method.loc31_37.4: <bound method> = bound_method %r.ref, %specific_impl_fn.loc31_37.2
// CHECK:STDOUT:     %.loc31_37.7: ref %Optional.e69 = temporary_storage
// CHECK:STDOUT:     %Iterate.WithSelf.Next.call: init %Optional.e69 to %.loc31_37.7 = call %bound_method.loc31_37.4(%r.ref, %addr.loc31)
// CHECK:STDOUT:     %.loc31_37.8: ref %Optional.e69 = temporary %.loc31_37.7, %Iterate.WithSelf.Next.call
// CHECK:STDOUT:     %.loc31_37.9: %Optional.HasValue.type.c61 = specific_constant imports.%Core.import_ref.bb6, @Optional(constants.%OptionalStorage.facet.b3c) [concrete = constants.%Optional.HasValue.ba9d]
// CHECK:STDOUT:     %HasValue.ref: %Optional.HasValue.type.c61 = name_ref HasValue, %.loc31_37.9 [concrete = constants.%Optional.HasValue.ba9d]
// CHECK:STDOUT:     %Optional.HasValue.bound: <bound method> = bound_method %.loc31_37.8, %HasValue.ref
// CHECK:STDOUT:     %Optional.HasValue.specific_fn: <specific function> = specific_function %HasValue.ref, @Optional.HasValue(constants.%OptionalStorage.facet.b3c) [concrete = constants.%Optional.HasValue.specific_fn.524]
// CHECK:STDOUT:     %bound_method.loc31_37.5: <bound method> = bound_method %.loc31_37.8, %Optional.HasValue.specific_fn
// CHECK:STDOUT:     %.loc31_37.10: %Optional.e69 = acquire_value %.loc31_37.8
// CHECK:STDOUT:     %Optional.HasValue.call: init bool = call %bound_method.loc31_37.5(%.loc31_37.10)
// CHECK:STDOUT:     %.loc31_37.11: bool = value_of_initializer %Optional.HasValue.call
// CHECK:STDOUT:     %.loc31_37.12: bool = converted %Optional.HasValue.call, %.loc31_37.11
// CHECK:STDOUT:     if %.loc31_37.12 br !for.body else br !for.done
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.body:
// CHECK:STDOUT:     %.loc31_37.13: %Optional.Get.type.394 = specific_constant imports.%Core.import_ref.dab, @Optional(constants.%OptionalStorage.facet.b3c) [concrete = constants.%Optional.Get.da9]
// CHECK:STDOUT:     %Get.ref: %Optional.Get.type.394 = name_ref Get, %.loc31_37.13 [concrete = constants.%Optional.Get.da9]
// CHECK:STDOUT:     %Optional.Get.bound: <bound method> = bound_method %.loc31_37.8, %Get.ref
// CHECK:STDOUT:     %Optional.Get.specific_fn: <specific function> = specific_function %Get.ref, @Optional.Get(constants.%OptionalStorage.facet.b3c) [concrete = constants.%Optional.Get.specific_fn.f41]
// CHECK:STDOUT:     %bound_method.loc31_37.6: <bound method> = bound_method %.loc31_37.8, %Optional.Get.specific_fn
// CHECK:STDOUT:     %.loc31_37.14: ref %ValueType = temporary_storage
// CHECK:STDOUT:     %.loc31_37.15: %Optional.e69 = acquire_value %.loc31_37.8
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
// CHECK:STDOUT:     %Destroy.WithSelf.SelfDestruct.bound.loc31_37.1: <bound method> = bound_method %.loc31_37.16, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.2
// CHECK:STDOUT:     %Destroy.WithSelf.SelfDestruct.call.loc31_37.1: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc31_37.1(%.loc31_37.16)
// CHECK:STDOUT:     %Destroy.WithSelf.SelfDestruct.bound.loc31_37.2: <bound method> = bound_method %.loc31_37.8, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.8
// CHECK:STDOUT:     %Destroy.WithSelf.SelfDestruct.call.loc31_37.2: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc31_37.2(%.loc31_37.8)
// CHECK:STDOUT:     br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.done:
// CHECK:STDOUT:     %Destroy.WithSelf.SelfDestruct.bound.loc31_37.3: <bound method> = bound_method %.loc31_37.8, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.8
// CHECK:STDOUT:     %Destroy.WithSelf.SelfDestruct.call.loc31_37.3: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc31_37.3(%.loc31_37.8)
// CHECK:STDOUT:     %impl.elem2.loc31_37.2: @TestIterate.%.loc31_37.22 (%.c10) = impl_witness_access constants.%Destroy.lookup_impl_witness.d0e, element2 [symbolic = %impl.elem2.loc31_37.4 (constants.%impl.elem2.5c5)]
// CHECK:STDOUT:     %bound_method.loc31_37.7: <bound method> = bound_method %var, %impl.elem2.loc31_37.2
// CHECK:STDOUT:     %.loc31_37.18: %Destroy.type = converted constants.%as_type.db9, constants.%impl.elem1.d56 [symbolic = %impl.elem1 (constants.%impl.elem1.d56)]
// CHECK:STDOUT:     %.loc31_37.19: %Destroy.type = converted constants.%as_type.db9, constants.%impl.elem1.d56 [symbolic = %impl.elem1 (constants.%impl.elem1.d56)]
// CHECK:STDOUT:     %specific_impl_fn.loc31_37.3: <specific function> = specific_impl_function %impl.elem2.loc31_37.2, @Destroy.WithSelf.SelfDestruct.1(constants.%impl.elem1.d56) [symbolic = %specific_impl_fn.loc31_37.6 (constants.%specific_impl_fn.17d)]
// CHECK:STDOUT:     %bound_method.loc31_37.8: <bound method> = bound_method %var, %specific_impl_fn.loc31_37.3
// CHECK:STDOUT:     %Destroy.WithSelf.SelfDestruct.call.loc31_37.4: init %empty_tuple.type = call %bound_method.loc31_37.8(%var)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc31_37.1(%self.param: ref %.be1) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc31_37.1(%self.param: ref %.be1) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc31_37.1(%self.param: ref %.be1) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.2(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.1(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc31_37.2(%self.param: ref %MaybeUnformed.e45) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc31_37.2(%self.param: ref %MaybeUnformed.e45) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc31_37.2(%self.param: ref %MaybeUnformed.e45) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.3(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.2(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc31_37.3(%self.param: ref bool) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc31_37.3(%self.param: ref bool) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc31_37.3(%self.param: ref bool) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.4(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.3(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc31_37.4(%self.param: ref %struct_type.value.has_value.619) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc31_37.1: ref %MaybeUnformed.e45 = struct_access %self.param, element0
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc31_37.1: <bound method> = bound_method %.loc31_37.1, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.4
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc31_37.1: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc31_37.1(%.loc31_37.1)
// CHECK:STDOUT:   %.loc31_37.2: ref bool = struct_access %self.param, element1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc31_37.2: <bound method> = bound_method %.loc31_37.2, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.5
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc31_37.2: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc31_37.2(%.loc31_37.2)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc31_37.4(%self.param: ref %struct_type.value.has_value.619) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc31_37.4(%self.param: ref %struct_type.value.has_value.619) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.5(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.4(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc31_37.5(%self.param: ref %DefaultOptionalStorage.53f) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc31_37.5(%self.param: ref %DefaultOptionalStorage.53f) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc31_37.5(%self.param: ref %DefaultOptionalStorage.53f) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.6(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.5(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc31_37.6(%self.param: ref %Optional.e69) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc31_37.6(%self.param: ref %Optional.e69) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc31_37.6(%self.param: ref %Optional.e69) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.7(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.6(%self.param)
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
// CHECK:STDOUT:   %impl.elem2: %.182 = impl_witness_access constants.%Iterate.impl_witness.6b0, element2 [concrete = constants.%T.as_type.as.Iterate.impl.NewCursor.fa5]
// CHECK:STDOUT:   %bound_method.loc45_37.1: <bound method> = bound_method %c.ref, %impl.elem2
// CHECK:STDOUT:   %facet_value.loc45_37.1: %CppRange_where.type.d682d6.1 = facet_value constants.%ConstRange, (constants.%custom_witness.226, constants.%custom_witness.202, constants.%custom_witness.116, constants.%custom_witness.429ac1.1, constants.%custom_witness.5be, constants.%custom_witness.c31, constants.%custom_witness.e21, constants.%custom_witness.429ac1.2) [concrete = constants.%facet_value.e8b]
// CHECK:STDOUT:   %.loc45_37.1: %CppRange_where.type.d682d6.1 = converted constants.%ConstRange, %facet_value.loc45_37.1 [concrete = constants.%facet_value.e8b]
// CHECK:STDOUT:   %facet_value.loc45_37.2: %CppRange_where.type.d682d6.1 = facet_value constants.%ConstRange, (constants.%custom_witness.226, constants.%custom_witness.202, constants.%custom_witness.116, constants.%custom_witness.429ac1.1, constants.%custom_witness.5be, constants.%custom_witness.c31, constants.%custom_witness.e21, constants.%custom_witness.429ac1.2) [concrete = constants.%facet_value.e8b]
// CHECK:STDOUT:   %.loc45_37.2: %CppRange_where.type.d682d6.1 = converted constants.%ConstRange, %facet_value.loc45_37.2 [concrete = constants.%facet_value.e8b]
// CHECK:STDOUT:   %specific_fn.loc45_37.1: <specific function> = specific_function %impl.elem2, @T.as_type.as.Iterate.impl.NewCursor(constants.%facet_value.e8b) [concrete = constants.%T.as_type.as.Iterate.impl.NewCursor.specific_fn]
// CHECK:STDOUT:   %bound_method.loc45_37.2: <bound method> = bound_method %c.ref, %specific_fn.loc45_37.1
// CHECK:STDOUT:   %var: ref %tuple.type.7a6 = var_storage invalid
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.call: init %tuple.type.7a6 to %var = call %bound_method.loc45_37.2(%c.ref)
// CHECK:STDOUT:   assign %var, %T.as_type.as.Iterate.impl.NewCursor.call
// CHECK:STDOUT:   br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT: !for.next:
// CHECK:STDOUT:   %addr.loc45: %ptr.d33 = addr_of %var
// CHECK:STDOUT:   %impl.elem3: %.883 = impl_witness_access constants.%Iterate.impl_witness.6b0, element3 [concrete = constants.%T.as_type.as.Iterate.impl.Next.e9b]
// CHECK:STDOUT:   %bound_method.loc45_37.3: <bound method> = bound_method %c.ref, %impl.elem3
// CHECK:STDOUT:   %facet_value.loc45_37.3: %CppRange_where.type.d682d6.1 = facet_value constants.%ConstRange, (constants.%custom_witness.226, constants.%custom_witness.202, constants.%custom_witness.116, constants.%custom_witness.429ac1.1, constants.%custom_witness.5be, constants.%custom_witness.c31, constants.%custom_witness.e21, constants.%custom_witness.429ac1.2) [concrete = constants.%facet_value.e8b]
// CHECK:STDOUT:   %.loc45_37.3: %CppRange_where.type.d682d6.1 = converted constants.%ConstRange, %facet_value.loc45_37.3 [concrete = constants.%facet_value.e8b]
// CHECK:STDOUT:   %facet_value.loc45_37.4: %CppRange_where.type.d682d6.1 = facet_value constants.%ConstRange, (constants.%custom_witness.226, constants.%custom_witness.202, constants.%custom_witness.116, constants.%custom_witness.429ac1.1, constants.%custom_witness.5be, constants.%custom_witness.c31, constants.%custom_witness.e21, constants.%custom_witness.429ac1.2) [concrete = constants.%facet_value.e8b]
// CHECK:STDOUT:   %.loc45_37.4: %CppRange_where.type.d682d6.1 = converted constants.%ConstRange, %facet_value.loc45_37.4 [concrete = constants.%facet_value.e8b]
// CHECK:STDOUT:   %specific_fn.loc45_37.2: <specific function> = specific_function %impl.elem3, @T.as_type.as.Iterate.impl.Next(constants.%facet_value.e8b) [concrete = constants.%T.as_type.as.Iterate.impl.Next.specific_fn]
// CHECK:STDOUT:   %bound_method.loc45_37.4: <bound method> = bound_method %c.ref, %specific_fn.loc45_37.2
// CHECK:STDOUT:   %.loc45_37.5: ref %Optional.e69 = temporary_storage
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.call: init %Optional.e69 to %.loc45_37.5 = call %bound_method.loc45_37.4(%c.ref, %addr.loc45)
// CHECK:STDOUT:   %.loc45_37.6: ref %Optional.e69 = temporary %.loc45_37.5, %T.as_type.as.Iterate.impl.Next.call
// CHECK:STDOUT:   %.loc45_37.7: %Optional.HasValue.type.c61 = specific_constant imports.%Core.import_ref.bb6, @Optional(constants.%OptionalStorage.facet.b3c) [concrete = constants.%Optional.HasValue.ba9d]
// CHECK:STDOUT:   %HasValue.ref: %Optional.HasValue.type.c61 = name_ref HasValue, %.loc45_37.7 [concrete = constants.%Optional.HasValue.ba9d]
// CHECK:STDOUT:   %Optional.HasValue.bound: <bound method> = bound_method %.loc45_37.6, %HasValue.ref
// CHECK:STDOUT:   %Optional.HasValue.specific_fn: <specific function> = specific_function %HasValue.ref, @Optional.HasValue(constants.%OptionalStorage.facet.b3c) [concrete = constants.%Optional.HasValue.specific_fn.524]
// CHECK:STDOUT:   %bound_method.loc45_37.5: <bound method> = bound_method %.loc45_37.6, %Optional.HasValue.specific_fn
// CHECK:STDOUT:   %.loc45_37.8: %Optional.e69 = acquire_value %.loc45_37.6
// CHECK:STDOUT:   %Optional.HasValue.call: init bool = call %bound_method.loc45_37.5(%.loc45_37.8)
// CHECK:STDOUT:   %.loc45_37.9: bool = value_of_initializer %Optional.HasValue.call
// CHECK:STDOUT:   %.loc45_37.10: bool = converted %Optional.HasValue.call, %.loc45_37.9
// CHECK:STDOUT:   if %.loc45_37.10 br !for.body else br !for.done
// CHECK:STDOUT:
// CHECK:STDOUT: !for.body:
// CHECK:STDOUT:   %.loc45_37.11: %Optional.Get.type.394 = specific_constant imports.%Core.import_ref.dab, @Optional(constants.%OptionalStorage.facet.b3c) [concrete = constants.%Optional.Get.da9]
// CHECK:STDOUT:   %Get.ref: %Optional.Get.type.394 = name_ref Get, %.loc45_37.11 [concrete = constants.%Optional.Get.da9]
// CHECK:STDOUT:   %Optional.Get.bound: <bound method> = bound_method %.loc45_37.6, %Get.ref
// CHECK:STDOUT:   %Optional.Get.specific_fn: <specific function> = specific_function %Get.ref, @Optional.Get(constants.%OptionalStorage.facet.b3c) [concrete = constants.%Optional.Get.specific_fn.f41]
// CHECK:STDOUT:   %bound_method.loc45_37.6: <bound method> = bound_method %.loc45_37.6, %Optional.Get.specific_fn
// CHECK:STDOUT:   %.loc45_37.12: ref %ValueType = temporary_storage
// CHECK:STDOUT:   %.loc45_37.13: %Optional.e69 = acquire_value %.loc45_37.6
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
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc45_37.1: <bound method> = bound_method %.loc45_37.14, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.2
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc45_37.1: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc45_37.1(%.loc45_37.14)
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc45_37.2: <bound method> = bound_method %.loc45_37.6, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.8
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc45_37.2: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc45_37.2(%.loc45_37.6)
// CHECK:STDOUT:   br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT: !for.done:
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc45_37.3: <bound method> = bound_method %.loc45_37.6, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.8
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc45_37.3: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc45_37.3(%.loc45_37.6)
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc45_37.4: <bound method> = bound_method %var, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.11
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc45_37.4: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc45_37.4(%var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestIterate(constants.%R) {
// CHECK:STDOUT:   %R.patt.loc27_17.2 => constants.%R.patt
// CHECK:STDOUT:   %R.loc27_17.1 => constants.%R
// CHECK:STDOUT:   %R.as_type.loc27_79.1 => constants.%R.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.053
// CHECK:STDOUT:   %r.param_patt.loc27_77.2 => constants.%r.param_patt.75e
// CHECK:STDOUT:   %r.patt.loc27_77.2 => constants.%r.patt.82a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestIterate(constants.%facet_value.c77) {
// CHECK:STDOUT:   %R.patt.loc27_17.2 => constants.%R.patt
// CHECK:STDOUT:   %R.loc27_17.1 => constants.%facet_value.c77
// CHECK:STDOUT:   %R.as_type.loc27_79.1 => constants.%ConstRange
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.c1d
// CHECK:STDOUT:   %r.param_patt.loc27_77.2 => constants.%r.param_patt.ac5
// CHECK:STDOUT:   %r.patt.loc27_77.2 => constants.%r.patt.97f
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc27 => constants.%complete_type.357
// CHECK:STDOUT:   %Iterate.lookup_impl_witness => constants.%Iterate.impl_witness.6b0
// CHECK:STDOUT:   %Iterate.facet.loc31_37.5 => constants.%Iterate.facet.80d
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type => constants.%Iterate.WithSelf.NewCursor.type.c59
// CHECK:STDOUT:   %.loc31_37.20 => constants.%.182
// CHECK:STDOUT:   %impl.elem2.loc31_37.3 => constants.%T.as_type.as.Iterate.impl.NewCursor.fa5
// CHECK:STDOUT:   %specific_impl_fn.loc31_37.4 => constants.%T.as_type.as.Iterate.impl.NewCursor.specific_fn
// CHECK:STDOUT:   %impl.elem1 => constants.%Destroy.facet.d83
// CHECK:STDOUT:   %as_type => constants.%tuple.type.7a6
// CHECK:STDOUT:   %require_complete.1 => constants.%complete_type.cf0
// CHECK:STDOUT:   %ptr => constants.%ptr.d33
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type => constants.%Iterate.WithSelf.Next.type.5e6
// CHECK:STDOUT:   %.loc31_37.21 => constants.%.883
// CHECK:STDOUT:   %impl.elem3.loc31_37.2 => constants.%T.as_type.as.Iterate.impl.Next.e9b
// CHECK:STDOUT:   %specific_impl_fn.loc31_37.5 => constants.%T.as_type.as.Iterate.impl.Next.specific_fn
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type => constants.%Destroy.WithSelf.SelfDestruct.type.036
// CHECK:STDOUT:   %.loc31_37.22 => constants.%.6cc
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.f8f19d.8
// CHECK:STDOUT:   %impl.elem2.loc31_37.4 => constants.%Destroy.WithSelf.SelfDestruct.db3fdb.11
// CHECK:STDOUT:   %specific_impl_fn.loc31_37.6 => constants.%Destroy.WithSelf.SelfDestruct.db3fdb.11
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_adl_returns_different_types_mutable.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %Iterate.type: type = facet_type <@Iterate> [concrete]
// CHECK:STDOUT:   %OptionalStorage.type: type = facet_type <@OptionalStorage> [concrete]
// CHECK:STDOUT:   %pattern_type.831: type = pattern_type bool [concrete]
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
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.2: type = fn_type @Destroy.WithSelf.Op.loc27 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.2: %Destroy.WithSelf.Op.type.ef016f.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.2: type = fn_type @Destroy.WithSelf.SelfDestruct.loc27 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.2: %Destroy.WithSelf.SelfDestruct.type.fbceb5.2 = struct_value () [concrete]
// CHECK:STDOUT:   %ValueType.SubobjectDestroy.type: type = fn_type @ValueType.SubobjectDestroy [concrete]
// CHECK:STDOUT:   %ValueType.SubobjectDestroy: %ValueType.SubobjectDestroy.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.3ec: <witness> = custom_witness (%Destroy.WithSelf.Op.403171.2, %ValueType.SubobjectDestroy, %Destroy.WithSelf.SelfDestruct.db3fdb.2), @Destroy [concrete]
// CHECK:STDOUT:   %const.91f: type = const_type %ValueType [concrete]
// CHECK:STDOUT:   %ptr.5f4: type = ptr_type %const.91f [concrete]
// CHECK:STDOUT:   %ptr.9c6: type = ptr_type %ValueType [concrete]
// CHECK:STDOUT:   %ValueType.Op.type: type = fn_type @ValueType.Op [concrete]
// CHECK:STDOUT:   %ValueType.Op: %ValueType.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.587: <witness> = custom_witness (%ValueType.Op), @Copy [concrete]
// CHECK:STDOUT:   %facet_value.14c: %facet_type.f48 = facet_value %ValueType, (%custom_witness.3ec, %custom_witness.587) [concrete]
// CHECK:STDOUT:   %Iterate.lookup_impl_witness.1d8: <witness> = lookup_impl_witness %.Self.500, @Iterate [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.294: %facet_type.f48 = impl_witness_access %Iterate.lookup_impl_witness.1d8, element0 [symbolic_self]
// CHECK:STDOUT:   %Iterate_where.type.9df: type = facet_type <@Iterate where %impl.elem0.294 = %facet_value.14c> [concrete]
// CHECK:STDOUT:   %pattern_type.e3a: type = pattern_type %Iterate_where.type.9df [concrete]
// CHECK:STDOUT:   %R.patt: %pattern_type.e3a = symbolic_binding_pattern R, 0 [symbolic]
// CHECK:STDOUT:   %R: %Iterate_where.type.9df = symbolic_binding R, 0 [symbolic]
// CHECK:STDOUT:   %R.as_type: type = facet_access_type %R [symbolic]
// CHECK:STDOUT:   %pattern_type.0bc: type = pattern_type %R.as_type [symbolic]
// CHECK:STDOUT:   %r.param_patt: %pattern_type.0bc = value_param_pattern [symbolic]
// CHECK:STDOUT:   %r.patt: %pattern_type.0bc = wrapper_binding_pattern r, %r.param_patt [symbolic]
// CHECK:STDOUT:   %i.patt: %pattern_type.d2d = value_binding_pattern i [concrete]
// CHECK:STDOUT:   %Iterate.lookup_impl_witness.711: <witness> = lookup_impl_witness %R, @Iterate [symbolic]
// CHECK:STDOUT:   %Iterate.facet.04e: %Iterate.type = facet_value %R.as_type, (%Iterate.lookup_impl_witness.711) [symbolic]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.aea: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.04e) [symbolic]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.9d6: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.04e) [symbolic]
// CHECK:STDOUT:   %.d1d: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.aea, %Iterate.facet.04e [symbolic]
// CHECK:STDOUT:   %impl.elem2.bf8: %.d1d = impl_witness_access %Iterate.lookup_impl_witness.711, element2 [symbolic]
// CHECK:STDOUT:   %impl.elem1.0e0: %Destroy.type = impl_witness_access %Iterate.lookup_impl_witness.711, element1 [symbolic]
// CHECK:STDOUT:   %as_type.b30: type = facet_access_type %impl.elem1.0e0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.d24: <specific function> = specific_impl_function %impl.elem2.bf8, @Iterate.WithSelf.NewCursor(%Iterate.facet.04e) [symbolic]
// CHECK:STDOUT:   %ptr.46a: type = ptr_type %as_type.b30 [symbolic]
// CHECK:STDOUT:   %.b83: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.9d6, %Iterate.facet.04e [symbolic]
// CHECK:STDOUT:   %impl.elem3.351: %.b83 = impl_witness_access %Iterate.lookup_impl_witness.711, element3 [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.6f8: type = class_type @DefaultOptionalStorage, @DefaultOptionalStorage(%facet_value.14c) [concrete]
// CHECK:STDOUT:   %OptionalStorage.impl_witness.323: <witness> = impl_witness imports.%OptionalStorage.impl_witness_table.460, @T.as_type.as.OptionalStorage.impl(%facet_value.14c) [concrete]
// CHECK:STDOUT:   %OptionalStorage.facet.864: %OptionalStorage.type = facet_value %ValueType, (%OptionalStorage.impl_witness.323) [concrete]
// CHECK:STDOUT:   %Optional.59e: type = class_type @Optional, @Optional(%OptionalStorage.facet.864) [concrete]
// CHECK:STDOUT:   %pattern_type.9bf: type = pattern_type %Optional.59e [concrete]
// CHECK:STDOUT:   %specific_impl_fn.946: <specific function> = specific_impl_function %impl.elem3.351, @Iterate.WithSelf.Next(%Iterate.facet.04e) [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.type.22f: type = fn_type @Optional.HasValue, @Optional(%OptionalStorage.facet.864) [concrete]
// CHECK:STDOUT:   %Optional.HasValue.c86: %Optional.HasValue.type.22f = struct_value () [concrete]
// CHECK:STDOUT:   %Optional.Get.type.e57: type = fn_type @Optional.Get, @Optional(%OptionalStorage.facet.864) [concrete]
// CHECK:STDOUT:   %Optional.Get.d4a: %Optional.Get.type.e57 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.facet.6cb: %Destroy.type = facet_value %ValueType, (%custom_witness.3ec) [concrete]
// CHECK:STDOUT:   %MaybeUnformed.8ca: type = class_type @MaybeUnformed, @MaybeUnformed(%Destroy.facet.6cb) [concrete]
// CHECK:STDOUT:   %.dd1: type = maybe_unformed_type %ValueType [concrete]
// CHECK:STDOUT:   %struct_type.value.has_value.786: type = struct_type {.value: %MaybeUnformed.8ca, .has_value: bool} [concrete]
// CHECK:STDOUT:   %Optional.HasValue.specific_fn: <specific function> = specific_function %Optional.HasValue.c86, @Optional.HasValue(%OptionalStorage.facet.864) [concrete]
// CHECK:STDOUT:   %Optional.Get.specific_fn: <specific function> = specific_function %Optional.Get.d4a, @Optional.Get(%OptionalStorage.facet.864) [concrete]
// CHECK:STDOUT:   %operator_PlusEqual__carbon_thunk.type: type = fn_type @operator_PlusEqual__carbon_thunk [concrete]
// CHECK:STDOUT:   %operator_PlusEqual__carbon_thunk: %operator_PlusEqual__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.1d1: type = pattern_type %.dd1 [concrete]
// CHECK:STDOUT:   %self.param_patt.964: %pattern_type.1d1 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.82f: %pattern_type.1d1 = wrapper_binding_pattern self, %self.param_patt.964 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc31_44.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.2: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.3: type = fn_type @Destroy.WithSelf.Op.loc31_44.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.3: %Destroy.WithSelf.Op.type.ef016f.3 = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.c42: type = pattern_type %MaybeUnformed.8ca [concrete]
// CHECK:STDOUT:   %self.param_patt.7e6: %pattern_type.c42 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.c0f: %pattern_type.c42 = wrapper_binding_pattern self, %self.param_patt.7e6 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.3: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc31_44.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.3: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.3 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.4: type = fn_type @Destroy.WithSelf.Op.loc31_44.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.4: %Destroy.WithSelf.Op.type.ef016f.4 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.4: type = fn_type @Destroy.WithSelf.SelfDestruct.loc31_44.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.4: %Destroy.WithSelf.SelfDestruct.type.fbceb5.4 = struct_value () [concrete]
// CHECK:STDOUT:   %self.param_patt.40d: %pattern_type.831 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.560: %pattern_type.831 = wrapper_binding_pattern self, %self.param_patt.40d [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.4: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc31_44.3 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.4: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.4 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.5: type = fn_type @Destroy.WithSelf.Op.loc31_44.3 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.5: %Destroy.WithSelf.Op.type.ef016f.5 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.5: type = fn_type @Destroy.WithSelf.SelfDestruct.loc31_44.3 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.5: %Destroy.WithSelf.SelfDestruct.type.fbceb5.5 = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.8f3: type = pattern_type %struct_type.value.has_value.786 [concrete]
// CHECK:STDOUT:   %self.param_patt.8d5: %pattern_type.8f3 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.5e5: %pattern_type.8f3 = wrapper_binding_pattern self, %self.param_patt.8d5 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.5: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc31_44.4 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.5: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.5 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.6: type = fn_type @Destroy.WithSelf.Op.loc31_44.4 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.6: %Destroy.WithSelf.Op.type.ef016f.6 = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.e17: type = pattern_type %DefaultOptionalStorage.6f8 [concrete]
// CHECK:STDOUT:   %self.param_patt.993: %pattern_type.e17 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.c64: %pattern_type.e17 = wrapper_binding_pattern self, %self.param_patt.993 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.6: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc31_44.5 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.6: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.6 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.7: type = fn_type @Destroy.WithSelf.Op.loc31_44.5 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.7: %Destroy.WithSelf.Op.type.ef016f.7 = struct_value () [concrete]
// CHECK:STDOUT:   %self.param_patt.d84: %pattern_type.9bf = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.81b: %pattern_type.9bf = wrapper_binding_pattern self, %self.param_patt.d84 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.7: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc31_44.6 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.7: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.7 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.8: type = fn_type @Destroy.WithSelf.Op.loc31_44.6 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.8: %Destroy.WithSelf.Op.type.ef016f.8 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.8: type = fn_type @Destroy.WithSelf.SelfDestruct.loc31_44.6 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.8: %Destroy.WithSelf.SelfDestruct.type.fbceb5.8 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.696: <witness> = lookup_impl_witness %impl.elem1.0e0, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.bb9: type = fn_type @Destroy.WithSelf.SelfDestruct.1, @Destroy.WithSelf(%impl.elem1.0e0) [symbolic]
// CHECK:STDOUT:   %.cd9: type = fn_type_with_self_type %Destroy.WithSelf.SelfDestruct.type.bb9, %impl.elem1.0e0 [symbolic]
// CHECK:STDOUT:   %impl.elem2.75b: %.cd9 = impl_witness_access %Destroy.lookup_impl_witness.696, element2 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.65c: <specific function> = specific_impl_function %impl.elem2.75b, @Destroy.WithSelf.SelfDestruct.1(%impl.elem1.0e0) [symbolic]
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
// CHECK:STDOUT: generated {
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.1: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc31_44.1 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.2] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.1d1 = ref_param_pattern [concrete = constants.%self.param_patt.964]
// CHECK:STDOUT:     %self.patt: %pattern_type.1d1 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.82f]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %.dd1 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %.dd1 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.2: %Destroy.WithSelf.Op.type.ef016f.3 = fn_decl @Destroy.WithSelf.Op.loc31_44.1 [concrete = constants.%Destroy.WithSelf.Op.403171.3] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.1d1 = ref_param_pattern [concrete = constants.%self.param_patt.964]
// CHECK:STDOUT:     %self.patt: %pattern_type.1d1 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.82f]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %.dd1 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %.dd1 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.2: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.3 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc31_44.2 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.3] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.c42 = ref_param_pattern [concrete = constants.%self.param_patt.7e6]
// CHECK:STDOUT:     %self.patt: %pattern_type.c42 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.c0f]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %MaybeUnformed.8ca = ref_param call_param0
// CHECK:STDOUT:     %self: ref %MaybeUnformed.8ca = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.3: %Destroy.WithSelf.Op.type.ef016f.4 = fn_decl @Destroy.WithSelf.Op.loc31_44.2 [concrete = constants.%Destroy.WithSelf.Op.403171.4] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.c42 = ref_param_pattern [concrete = constants.%self.param_patt.7e6]
// CHECK:STDOUT:     %self.patt: %pattern_type.c42 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.c0f]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %MaybeUnformed.8ca = ref_param call_param0
// CHECK:STDOUT:     %self: ref %MaybeUnformed.8ca = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.3: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.4 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc31_44.3 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.4] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.831 = ref_param_pattern [concrete = constants.%self.param_patt.40d]
// CHECK:STDOUT:     %self.patt: %pattern_type.831 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.560]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref bool = ref_param call_param0
// CHECK:STDOUT:     %self: ref bool = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.4: %Destroy.WithSelf.Op.type.ef016f.5 = fn_decl @Destroy.WithSelf.Op.loc31_44.3 [concrete = constants.%Destroy.WithSelf.Op.403171.5] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.831 = ref_param_pattern [concrete = constants.%self.param_patt.40d]
// CHECK:STDOUT:     %self.patt: %pattern_type.831 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.560]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref bool = ref_param call_param0
// CHECK:STDOUT:     %self: ref bool = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.4: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.5 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc31_44.4 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.5] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.8f3 = ref_param_pattern [concrete = constants.%self.param_patt.8d5]
// CHECK:STDOUT:     %self.patt: %pattern_type.8f3 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.5e5]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %struct_type.value.has_value.786 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %struct_type.value.has_value.786 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.5: %Destroy.WithSelf.Op.type.ef016f.6 = fn_decl @Destroy.WithSelf.Op.loc31_44.4 [concrete = constants.%Destroy.WithSelf.Op.403171.6] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.8f3 = ref_param_pattern [concrete = constants.%self.param_patt.8d5]
// CHECK:STDOUT:     %self.patt: %pattern_type.8f3 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.5e5]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %struct_type.value.has_value.786 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %struct_type.value.has_value.786 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.5: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.6 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc31_44.5 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.6] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.e17 = ref_param_pattern [concrete = constants.%self.param_patt.993]
// CHECK:STDOUT:     %self.patt: %pattern_type.e17 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.c64]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %DefaultOptionalStorage.6f8 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %DefaultOptionalStorage.6f8 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.6: %Destroy.WithSelf.Op.type.ef016f.7 = fn_decl @Destroy.WithSelf.Op.loc31_44.5 [concrete = constants.%Destroy.WithSelf.Op.403171.7] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.e17 = ref_param_pattern [concrete = constants.%self.param_patt.993]
// CHECK:STDOUT:     %self.patt: %pattern_type.e17 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.c64]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %DefaultOptionalStorage.6f8 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %DefaultOptionalStorage.6f8 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.6: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.7 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc31_44.6 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.7] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.9bf = ref_param_pattern [concrete = constants.%self.param_patt.d84]
// CHECK:STDOUT:     %self.patt: %pattern_type.9bf = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.81b]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %Optional.59e = ref_param call_param0
// CHECK:STDOUT:     %self: ref %Optional.59e = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.7: %Destroy.WithSelf.Op.type.ef016f.8 = fn_decl @Destroy.WithSelf.Op.loc31_44.6 [concrete = constants.%Destroy.WithSelf.Op.403171.8] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.9bf = ref_param_pattern [concrete = constants.%self.param_patt.d84]
// CHECK:STDOUT:     %self.patt: %pattern_type.9bf = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.81b]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %Optional.59e = ref_param call_param0
// CHECK:STDOUT:     %self: ref %Optional.59e = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @TestIterate(%R.loc27_17.2: %Iterate_where.type.9df) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Iterate.lookup_impl_witness: <witness> = lookup_impl_witness %R.loc27_17.1, @Iterate [symbolic = %Iterate.lookup_impl_witness (constants.%Iterate.lookup_impl_witness.711)]
// CHECK:STDOUT:   %Iterate.facet.loc31_44.5: %Iterate.type = facet_value %R.as_type.loc27_86.1, (%Iterate.lookup_impl_witness) [symbolic = %Iterate.facet.loc31_44.5 (constants.%Iterate.facet.04e)]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.loc31_44.5) [symbolic = %Iterate.WithSelf.NewCursor.type (constants.%Iterate.WithSelf.NewCursor.type.aea)]
// CHECK:STDOUT:   %.loc31_44.20: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type, %Iterate.facet.loc31_44.5 [symbolic = %.loc31_44.20 (constants.%.d1d)]
// CHECK:STDOUT:   %impl.elem2.loc31_44.3: @TestIterate.%.loc31_44.20 (%.d1d) = impl_witness_access %Iterate.lookup_impl_witness, element2 [symbolic = %impl.elem2.loc31_44.3 (constants.%impl.elem2.bf8)]
// CHECK:STDOUT:   %specific_impl_fn.loc31_44.4: <specific function> = specific_impl_function %impl.elem2.loc31_44.3, @Iterate.WithSelf.NewCursor(%Iterate.facet.loc31_44.5) [symbolic = %specific_impl_fn.loc31_44.4 (constants.%specific_impl_fn.d24)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %ptr: type = ptr_type %as_type [symbolic = %ptr (constants.%ptr.46a)]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.loc31_44.5) [symbolic = %Iterate.WithSelf.Next.type (constants.%Iterate.WithSelf.Next.type.9d6)]
// CHECK:STDOUT:   %.loc31_44.21: type = fn_type_with_self_type %Iterate.WithSelf.Next.type, %Iterate.facet.loc31_44.5 [symbolic = %.loc31_44.21 (constants.%.b83)]
// CHECK:STDOUT:   %impl.elem3.loc31_44.2: @TestIterate.%.loc31_44.21 (%.b83) = impl_witness_access %Iterate.lookup_impl_witness, element3 [symbolic = %impl.elem3.loc31_44.2 (constants.%impl.elem3.351)]
// CHECK:STDOUT:   %specific_impl_fn.loc31_44.5: <specific function> = specific_impl_function %impl.elem3.loc31_44.2, @Iterate.WithSelf.Next(%Iterate.facet.loc31_44.5) [symbolic = %specific_impl_fn.loc31_44.5 (constants.%specific_impl_fn.946)]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type: type = fn_type @Destroy.WithSelf.SelfDestruct.1, @Destroy.WithSelf(%impl.elem1) [symbolic = %Destroy.WithSelf.SelfDestruct.type (constants.%Destroy.WithSelf.SelfDestruct.type.bb9)]
// CHECK:STDOUT:   %.loc31_44.22: type = fn_type_with_self_type %Destroy.WithSelf.SelfDestruct.type, %impl.elem1 [symbolic = %.loc31_44.22 (constants.%.cd9)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %impl.elem1, @Destroy [symbolic = %Destroy.lookup_impl_witness (constants.%Destroy.lookup_impl_witness.696)]
// CHECK:STDOUT:   %impl.elem2.loc31_44.4: @TestIterate.%.loc31_44.22 (%.cd9) = impl_witness_access %Destroy.lookup_impl_witness, element2 [symbolic = %impl.elem2.loc31_44.4 (constants.%impl.elem2.75b)]
// CHECK:STDOUT:   %specific_impl_fn.loc31_44.6: <specific function> = specific_impl_function %impl.elem2.loc31_44.4, @Destroy.WithSelf.SelfDestruct.1(%impl.elem1) [symbolic = %specific_impl_fn.loc31_44.6 (constants.%specific_impl_fn.65c)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%r.param: @TestIterate.%R.as_type.loc27_86.1 (%R.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %i.patt: %pattern_type.d2d = value_binding_pattern i [concrete = constants.%i.patt]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %r.ref: @TestIterate.%R.as_type.loc27_86.1 (%R.as_type) = name_ref r, %r
// CHECK:STDOUT:     %impl.elem2.loc31_44.1: @TestIterate.%.loc31_44.20 (%.d1d) = impl_witness_access constants.%Iterate.lookup_impl_witness.711, element2 [symbolic = %impl.elem2.loc31_44.3 (constants.%impl.elem2.bf8)]
// CHECK:STDOUT:     %bound_method.loc31_44.1: <bound method> = bound_method %r.ref, %impl.elem2.loc31_44.1
// CHECK:STDOUT:     %Iterate.facet.loc31_44.1: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.711) [symbolic = %Iterate.facet.loc31_44.5 (constants.%Iterate.facet.04e)]
// CHECK:STDOUT:     %.loc31_44.1: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc31_44.1 [symbolic = %Iterate.facet.loc31_44.5 (constants.%Iterate.facet.04e)]
// CHECK:STDOUT:     %Iterate.facet.loc31_44.2: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.711) [symbolic = %Iterate.facet.loc31_44.5 (constants.%Iterate.facet.04e)]
// CHECK:STDOUT:     %.loc31_44.2: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc31_44.2 [symbolic = %Iterate.facet.loc31_44.5 (constants.%Iterate.facet.04e)]
// CHECK:STDOUT:     %specific_impl_fn.loc31_44.1: <specific function> = specific_impl_function %impl.elem2.loc31_44.1, @Iterate.WithSelf.NewCursor(constants.%Iterate.facet.04e) [symbolic = %specific_impl_fn.loc31_44.4 (constants.%specific_impl_fn.d24)]
// CHECK:STDOUT:     %bound_method.loc31_44.2: <bound method> = bound_method %r.ref, %specific_impl_fn.loc31_44.1
// CHECK:STDOUT:     %var: ref @TestIterate.%as_type (%as_type.b30) = var_storage invalid
// CHECK:STDOUT:     %Iterate.WithSelf.NewCursor.call: init @TestIterate.%as_type (%as_type.b30) to %var = call %bound_method.loc31_44.2(%r.ref)
// CHECK:STDOUT:     assign %var, %Iterate.WithSelf.NewCursor.call
// CHECK:STDOUT:     br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.next:
// CHECK:STDOUT:     %addr.loc31: @TestIterate.%ptr (%ptr.46a) = addr_of %var
// CHECK:STDOUT:     %impl.elem3.loc31_44.1: @TestIterate.%.loc31_44.21 (%.b83) = impl_witness_access constants.%Iterate.lookup_impl_witness.711, element3 [symbolic = %impl.elem3.loc31_44.2 (constants.%impl.elem3.351)]
// CHECK:STDOUT:     %bound_method.loc31_44.3: <bound method> = bound_method %r.ref, %impl.elem3.loc31_44.1
// CHECK:STDOUT:     %Iterate.facet.loc31_44.3: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.711) [symbolic = %Iterate.facet.loc31_44.5 (constants.%Iterate.facet.04e)]
// CHECK:STDOUT:     %.loc31_44.3: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc31_44.3 [symbolic = %Iterate.facet.loc31_44.5 (constants.%Iterate.facet.04e)]
// CHECK:STDOUT:     %Iterate.facet.loc31_44.4: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.711) [symbolic = %Iterate.facet.loc31_44.5 (constants.%Iterate.facet.04e)]
// CHECK:STDOUT:     %.loc31_44.4: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc31_44.4 [symbolic = %Iterate.facet.loc31_44.5 (constants.%Iterate.facet.04e)]
// CHECK:STDOUT:     %.loc31_44.5: %Destroy.type = converted constants.%as_type.b30, constants.%impl.elem1.0e0 [symbolic = %impl.elem1 (constants.%impl.elem1.0e0)]
// CHECK:STDOUT:     %.loc31_44.6: %Destroy.type = converted constants.%as_type.b30, constants.%impl.elem1.0e0 [symbolic = %impl.elem1 (constants.%impl.elem1.0e0)]
// CHECK:STDOUT:     %specific_impl_fn.loc31_44.2: <specific function> = specific_impl_function %impl.elem3.loc31_44.1, @Iterate.WithSelf.Next(constants.%Iterate.facet.04e) [symbolic = %specific_impl_fn.loc31_44.5 (constants.%specific_impl_fn.946)]
// CHECK:STDOUT:     %bound_method.loc31_44.4: <bound method> = bound_method %r.ref, %specific_impl_fn.loc31_44.2
// CHECK:STDOUT:     %.loc31_44.7: ref %Optional.59e = temporary_storage
// CHECK:STDOUT:     %Iterate.WithSelf.Next.call: init %Optional.59e to %.loc31_44.7 = call %bound_method.loc31_44.4(%r.ref, %addr.loc31)
// CHECK:STDOUT:     %.loc31_44.8: ref %Optional.59e = temporary %.loc31_44.7, %Iterate.WithSelf.Next.call
// CHECK:STDOUT:     %.loc31_44.9: %Optional.HasValue.type.22f = specific_constant imports.%Core.import_ref.bb6, @Optional(constants.%OptionalStorage.facet.864) [concrete = constants.%Optional.HasValue.c86]
// CHECK:STDOUT:     %HasValue.ref: %Optional.HasValue.type.22f = name_ref HasValue, %.loc31_44.9 [concrete = constants.%Optional.HasValue.c86]
// CHECK:STDOUT:     %Optional.HasValue.bound: <bound method> = bound_method %.loc31_44.8, %HasValue.ref
// CHECK:STDOUT:     %Optional.HasValue.specific_fn: <specific function> = specific_function %HasValue.ref, @Optional.HasValue(constants.%OptionalStorage.facet.864) [concrete = constants.%Optional.HasValue.specific_fn]
// CHECK:STDOUT:     %bound_method.loc31_44.5: <bound method> = bound_method %.loc31_44.8, %Optional.HasValue.specific_fn
// CHECK:STDOUT:     %.loc31_44.10: %Optional.59e = acquire_value %.loc31_44.8
// CHECK:STDOUT:     %Optional.HasValue.call: init bool = call %bound_method.loc31_44.5(%.loc31_44.10)
// CHECK:STDOUT:     %.loc31_44.11: bool = value_of_initializer %Optional.HasValue.call
// CHECK:STDOUT:     %.loc31_44.12: bool = converted %Optional.HasValue.call, %.loc31_44.11
// CHECK:STDOUT:     if %.loc31_44.12 br !for.body else br !for.done
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.body:
// CHECK:STDOUT:     %.loc31_44.13: %Optional.Get.type.e57 = specific_constant imports.%Core.import_ref.dab, @Optional(constants.%OptionalStorage.facet.864) [concrete = constants.%Optional.Get.d4a]
// CHECK:STDOUT:     %Get.ref: %Optional.Get.type.e57 = name_ref Get, %.loc31_44.13 [concrete = constants.%Optional.Get.d4a]
// CHECK:STDOUT:     %Optional.Get.bound: <bound method> = bound_method %.loc31_44.8, %Get.ref
// CHECK:STDOUT:     %Optional.Get.specific_fn: <specific function> = specific_function %Get.ref, @Optional.Get(constants.%OptionalStorage.facet.864) [concrete = constants.%Optional.Get.specific_fn]
// CHECK:STDOUT:     %bound_method.loc31_44.6: <bound method> = bound_method %.loc31_44.8, %Optional.Get.specific_fn
// CHECK:STDOUT:     %.loc31_44.14: ref %ValueType = temporary_storage
// CHECK:STDOUT:     %.loc31_44.15: %Optional.59e = acquire_value %.loc31_44.8
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
// CHECK:STDOUT:     %Destroy.WithSelf.SelfDestruct.bound.loc31_44.1: <bound method> = bound_method %.loc31_44.16, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.2
// CHECK:STDOUT:     %Destroy.WithSelf.SelfDestruct.call.loc31_44.1: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc31_44.1(%.loc31_44.16)
// CHECK:STDOUT:     %Destroy.WithSelf.SelfDestruct.bound.loc31_44.2: <bound method> = bound_method %.loc31_44.8, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.8
// CHECK:STDOUT:     %Destroy.WithSelf.SelfDestruct.call.loc31_44.2: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc31_44.2(%.loc31_44.8)
// CHECK:STDOUT:     br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.done:
// CHECK:STDOUT:     %Destroy.WithSelf.SelfDestruct.bound.loc31_44.3: <bound method> = bound_method %.loc31_44.8, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.8
// CHECK:STDOUT:     %Destroy.WithSelf.SelfDestruct.call.loc31_44.3: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc31_44.3(%.loc31_44.8)
// CHECK:STDOUT:     %impl.elem2.loc31_44.2: @TestIterate.%.loc31_44.22 (%.cd9) = impl_witness_access constants.%Destroy.lookup_impl_witness.696, element2 [symbolic = %impl.elem2.loc31_44.4 (constants.%impl.elem2.75b)]
// CHECK:STDOUT:     %bound_method.loc31_44.7: <bound method> = bound_method %var, %impl.elem2.loc31_44.2
// CHECK:STDOUT:     %.loc31_44.18: %Destroy.type = converted constants.%as_type.b30, constants.%impl.elem1.0e0 [symbolic = %impl.elem1 (constants.%impl.elem1.0e0)]
// CHECK:STDOUT:     %.loc31_44.19: %Destroy.type = converted constants.%as_type.b30, constants.%impl.elem1.0e0 [symbolic = %impl.elem1 (constants.%impl.elem1.0e0)]
// CHECK:STDOUT:     %specific_impl_fn.loc31_44.3: <specific function> = specific_impl_function %impl.elem2.loc31_44.2, @Destroy.WithSelf.SelfDestruct.1(constants.%impl.elem1.0e0) [symbolic = %specific_impl_fn.loc31_44.6 (constants.%specific_impl_fn.65c)]
// CHECK:STDOUT:     %bound_method.loc31_44.8: <bound method> = bound_method %var, %specific_impl_fn.loc31_44.3
// CHECK:STDOUT:     %Destroy.WithSelf.SelfDestruct.call.loc31_44.4: init %empty_tuple.type = call %bound_method.loc31_44.8(%var)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc31_44.1(%self.param: ref %.dd1) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc31_44.1(%self.param: ref %.dd1) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc31_44.1(%self.param: ref %.dd1) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.2(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.1(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc31_44.2(%self.param: ref %MaybeUnformed.8ca) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc31_44.2(%self.param: ref %MaybeUnformed.8ca) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc31_44.2(%self.param: ref %MaybeUnformed.8ca) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.3(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.2(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc31_44.3(%self.param: ref bool) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc31_44.3(%self.param: ref bool) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc31_44.3(%self.param: ref bool) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.4(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.3(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc31_44.4(%self.param: ref %struct_type.value.has_value.786) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc31_44.1: ref %MaybeUnformed.8ca = struct_access %self.param, element0
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc31_44.1: <bound method> = bound_method %.loc31_44.1, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.4
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc31_44.1: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc31_44.1(%.loc31_44.1)
// CHECK:STDOUT:   %.loc31_44.2: ref bool = struct_access %self.param, element1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc31_44.2: <bound method> = bound_method %.loc31_44.2, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.5
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc31_44.2: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc31_44.2(%.loc31_44.2)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc31_44.4(%self.param: ref %struct_type.value.has_value.786) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc31_44.4(%self.param: ref %struct_type.value.has_value.786) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.5(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.4(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc31_44.5(%self.param: ref %DefaultOptionalStorage.6f8) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc31_44.5(%self.param: ref %DefaultOptionalStorage.6f8) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc31_44.5(%self.param: ref %DefaultOptionalStorage.6f8) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.6(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.5(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc31_44.6(%self.param: ref %Optional.59e) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc31_44.6(%self.param: ref %Optional.59e) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc31_44.6(%self.param: ref %Optional.59e) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.7(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.6(%self.param)
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
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.0bc
// CHECK:STDOUT:   %r.param_patt.loc27_84.2 => constants.%r.param_patt
// CHECK:STDOUT:   %r.patt.loc27_84.2 => constants.%r.patt
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_adl_overloads.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %Iterate.type: type = facet_type <@Iterate> [concrete]
// CHECK:STDOUT:   %OptionalStorage.type: type = facet_type <@OptionalStorage> [concrete]
// CHECK:STDOUT:   %pattern_type.831: type = pattern_type bool [concrete]
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
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.2: type = fn_type @Destroy.WithSelf.Op.loc34 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.2: %Destroy.WithSelf.Op.type.ef016f.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.2: type = fn_type @Destroy.WithSelf.SelfDestruct.loc34 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.2: %Destroy.WithSelf.SelfDestruct.type.fbceb5.2 = struct_value () [concrete]
// CHECK:STDOUT:   %ValueType.SubobjectDestroy.type: type = fn_type @ValueType.SubobjectDestroy [concrete]
// CHECK:STDOUT:   %ValueType.SubobjectDestroy: %ValueType.SubobjectDestroy.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.dc3: <witness> = custom_witness (%Destroy.WithSelf.Op.403171.2, %ValueType.SubobjectDestroy, %Destroy.WithSelf.SelfDestruct.db3fdb.2), @Destroy [concrete]
// CHECK:STDOUT:   %const.924: type = const_type %ValueType [concrete]
// CHECK:STDOUT:   %ptr.403: type = ptr_type %const.924 [concrete]
// CHECK:STDOUT:   %ptr.6ed: type = ptr_type %ValueType [concrete]
// CHECK:STDOUT:   %ValueType.Op.type: type = fn_type @ValueType.Op [concrete]
// CHECK:STDOUT:   %ValueType.Op: %ValueType.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.1c5: <witness> = custom_witness (%ValueType.Op), @Copy [concrete]
// CHECK:STDOUT:   %facet_value.36a: %facet_type.f48 = facet_value %ValueType, (%custom_witness.dc3, %custom_witness.1c5) [concrete]
// CHECK:STDOUT:   %Iterate.lookup_impl_witness.1d8: <witness> = lookup_impl_witness %.Self.500, @Iterate [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.294: %facet_type.f48 = impl_witness_access %Iterate.lookup_impl_witness.1d8, element0 [symbolic_self]
// CHECK:STDOUT:   %Iterate_where.type.8b0: type = facet_type <@Iterate where %impl.elem0.294 = %facet_value.36a> [concrete]
// CHECK:STDOUT:   %pattern_type.5f2: type = pattern_type %Iterate_where.type.8b0 [concrete]
// CHECK:STDOUT:   %R.patt: %pattern_type.5f2 = symbolic_binding_pattern R, 0 [symbolic]
// CHECK:STDOUT:   %R: %Iterate_where.type.8b0 = symbolic_binding R, 0 [symbolic]
// CHECK:STDOUT:   %R.as_type: type = facet_access_type %R [symbolic]
// CHECK:STDOUT:   %pattern_type.611: type = pattern_type %R.as_type [symbolic]
// CHECK:STDOUT:   %r.param_patt.e64: %pattern_type.611 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %r.patt.dcb: %pattern_type.611 = wrapper_binding_pattern r, %r.param_patt.e64 [symbolic]
// CHECK:STDOUT:   %i.patt: %pattern_type.e09 = value_binding_pattern i [concrete]
// CHECK:STDOUT:   %Iterate.lookup_impl_witness.a99: <witness> = lookup_impl_witness %R, @Iterate [symbolic]
// CHECK:STDOUT:   %Iterate.facet.c55: %Iterate.type = facet_value %R.as_type, (%Iterate.lookup_impl_witness.a99) [symbolic]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.a6c: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.c55) [symbolic]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.cd7: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.c55) [symbolic]
// CHECK:STDOUT:   %.57a: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.a6c, %Iterate.facet.c55 [symbolic]
// CHECK:STDOUT:   %impl.elem2.51b: %.57a = impl_witness_access %Iterate.lookup_impl_witness.a99, element2 [symbolic]
// CHECK:STDOUT:   %impl.elem1.751: %Destroy.type = impl_witness_access %Iterate.lookup_impl_witness.a99, element1 [symbolic]
// CHECK:STDOUT:   %as_type.733: type = facet_access_type %impl.elem1.751 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.986: <specific function> = specific_impl_function %impl.elem2.51b, @Iterate.WithSelf.NewCursor(%Iterate.facet.c55) [symbolic]
// CHECK:STDOUT:   %ptr.696: type = ptr_type %as_type.733 [symbolic]
// CHECK:STDOUT:   %.c1e: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.cd7, %Iterate.facet.c55 [symbolic]
// CHECK:STDOUT:   %impl.elem3.576: %.c1e = impl_witness_access %Iterate.lookup_impl_witness.a99, element3 [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.58e: type = class_type @DefaultOptionalStorage, @DefaultOptionalStorage(%facet_value.36a) [concrete]
// CHECK:STDOUT:   %OptionalStorage.impl_witness.913: <witness> = impl_witness imports.%OptionalStorage.impl_witness_table.460, @T.as_type.as.OptionalStorage.impl(%facet_value.36a) [concrete]
// CHECK:STDOUT:   %OptionalStorage.facet.e68: %OptionalStorage.type = facet_value %ValueType, (%OptionalStorage.impl_witness.913) [concrete]
// CHECK:STDOUT:   %Optional.70f: type = class_type @Optional, @Optional(%OptionalStorage.facet.e68) [concrete]
// CHECK:STDOUT:   %pattern_type.540: type = pattern_type %Optional.70f [concrete]
// CHECK:STDOUT:   %specific_impl_fn.1db: <specific function> = specific_impl_function %impl.elem3.576, @Iterate.WithSelf.Next(%Iterate.facet.c55) [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.type.74b: type = fn_type @Optional.HasValue, @Optional(%OptionalStorage.facet.e68) [concrete]
// CHECK:STDOUT:   %Optional.HasValue.f75: %Optional.HasValue.type.74b = struct_value () [concrete]
// CHECK:STDOUT:   %Optional.Get.type.316: type = fn_type @Optional.Get, @Optional(%OptionalStorage.facet.e68) [concrete]
// CHECK:STDOUT:   %Optional.Get.483: %Optional.Get.type.316 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.facet.745: %Destroy.type = facet_value %ValueType, (%custom_witness.dc3) [concrete]
// CHECK:STDOUT:   %MaybeUnformed.0af: type = class_type @MaybeUnformed, @MaybeUnformed(%Destroy.facet.745) [concrete]
// CHECK:STDOUT:   %.8f3: type = maybe_unformed_type %ValueType [concrete]
// CHECK:STDOUT:   %struct_type.value.has_value.ac3: type = struct_type {.value: %MaybeUnformed.0af, .has_value: bool} [concrete]
// CHECK:STDOUT:   %Optional.HasValue.specific_fn.e2e: <specific function> = specific_function %Optional.HasValue.f75, @Optional.HasValue(%OptionalStorage.facet.e68) [concrete]
// CHECK:STDOUT:   %Optional.Get.specific_fn.296: <specific function> = specific_function %Optional.Get.483, @Optional.Get(%OptionalStorage.facet.e68) [concrete]
// CHECK:STDOUT:   %operator_PlusEqual__carbon_thunk.type: type = fn_type @operator_PlusEqual__carbon_thunk [concrete]
// CHECK:STDOUT:   %operator_PlusEqual__carbon_thunk: %operator_PlusEqual__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.60e: type = pattern_type %.8f3 [concrete]
// CHECK:STDOUT:   %self.param_patt.61b3: %pattern_type.60e = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.ecc: %pattern_type.60e = wrapper_binding_pattern self, %self.param_patt.61b3 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc38_41.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.2: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.3: type = fn_type @Destroy.WithSelf.Op.loc38_41.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.3: %Destroy.WithSelf.Op.type.ef016f.3 = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.e4c: type = pattern_type %MaybeUnformed.0af [concrete]
// CHECK:STDOUT:   %self.param_patt.cb2: %pattern_type.e4c = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.32b: %pattern_type.e4c = wrapper_binding_pattern self, %self.param_patt.cb2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.3: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc38_41.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.3: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.3 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.4: type = fn_type @Destroy.WithSelf.Op.loc38_41.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.4: %Destroy.WithSelf.Op.type.ef016f.4 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.4: type = fn_type @Destroy.WithSelf.SelfDestruct.loc38_41.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.4: %Destroy.WithSelf.SelfDestruct.type.fbceb5.4 = struct_value () [concrete]
// CHECK:STDOUT:   %self.param_patt.40d: %pattern_type.831 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.560: %pattern_type.831 = wrapper_binding_pattern self, %self.param_patt.40d [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.4: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc38_41.3 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.4: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.4 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.5: type = fn_type @Destroy.WithSelf.Op.loc38_41.3 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.5: %Destroy.WithSelf.Op.type.ef016f.5 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.5: type = fn_type @Destroy.WithSelf.SelfDestruct.loc38_41.3 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.5: %Destroy.WithSelf.SelfDestruct.type.fbceb5.5 = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.95d: type = pattern_type %struct_type.value.has_value.ac3 [concrete]
// CHECK:STDOUT:   %self.param_patt.432: %pattern_type.95d = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.25a: %pattern_type.95d = wrapper_binding_pattern self, %self.param_patt.432 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.5: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc38_41.4 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.5: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.5 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.6: type = fn_type @Destroy.WithSelf.Op.loc38_41.4 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.6: %Destroy.WithSelf.Op.type.ef016f.6 = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.e96: type = pattern_type %DefaultOptionalStorage.58e [concrete]
// CHECK:STDOUT:   %self.param_patt.861: %pattern_type.e96 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.c75: %pattern_type.e96 = wrapper_binding_pattern self, %self.param_patt.861 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.6: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc38_41.5 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.6: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.6 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.7: type = fn_type @Destroy.WithSelf.Op.loc38_41.5 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.7: %Destroy.WithSelf.Op.type.ef016f.7 = struct_value () [concrete]
// CHECK:STDOUT:   %self.param_patt.422: %pattern_type.540 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.f61: %pattern_type.540 = wrapper_binding_pattern self, %self.param_patt.422 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.7: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc38_41.6 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.7: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.7 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.8: type = fn_type @Destroy.WithSelf.Op.loc38_41.6 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.8: %Destroy.WithSelf.Op.type.ef016f.8 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.8: type = fn_type @Destroy.WithSelf.SelfDestruct.loc38_41.6 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.8: %Destroy.WithSelf.SelfDestruct.type.fbceb5.8 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.eb0: <witness> = lookup_impl_witness %impl.elem1.751, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.57f: type = fn_type @Destroy.WithSelf.SelfDestruct.1, @Destroy.WithSelf(%impl.elem1.751) [symbolic]
// CHECK:STDOUT:   %.b9e: type = fn_type_with_self_type %Destroy.WithSelf.SelfDestruct.type.57f, %impl.elem1.751 [symbolic]
// CHECK:STDOUT:   %impl.elem2.570: %.b9e = impl_witness_access %Destroy.lookup_impl_witness.eb0, element2 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.74c: <specific function> = specific_impl_function %impl.elem2.570, @Destroy.WithSelf.SelfDestruct.1(%impl.elem1.751) [symbolic]
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
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.9: type = fn_type @Destroy.WithSelf.Op.3 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.9: %Destroy.WithSelf.Op.type.ef016f.9 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.9: type = fn_type @Destroy.WithSelf.SelfDestruct.3 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.9: %Destroy.WithSelf.SelfDestruct.type.fbceb5.9 = struct_value () [concrete]
// CHECK:STDOUT:   %Iterator.SubobjectDestroy.type: type = fn_type @Iterator.SubobjectDestroy [concrete]
// CHECK:STDOUT:   %Iterator.SubobjectDestroy: %Iterator.SubobjectDestroy.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.809: <witness> = custom_witness (%Destroy.WithSelf.Op.403171.9, %Iterator.SubobjectDestroy, %Destroy.WithSelf.SelfDestruct.db3fdb.9), @Destroy [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.c95ff6.2: type = fn_type @Iterator.Op.2 [concrete]
// CHECK:STDOUT:   %Iterator.Op.c90ac2.2: %Iterator.Op.type.c95ff6.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.68a: <witness> = custom_witness (%Iterator.Op.c90ac2.2), @Inc [concrete]
// CHECK:STDOUT:   %Equal.type: type = fn_type @Equal [concrete]
// CHECK:STDOUT:   %Equal: %Equal.type = struct_value () [concrete]
// CHECK:STDOUT:   %NotEqual.type: type = fn_type @NotEqual [concrete]
// CHECK:STDOUT:   %NotEqual: %NotEqual.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.2ef: <witness> = custom_witness (%Equal, %NotEqual), @EqWith, @EqWith(%Iterator) [concrete]
// CHECK:STDOUT:   %facet_value.cc4: %CppRange_where.type.d682d6.1 = facet_value %MutableAndConstRange, (%custom_witness.dc3, %custom_witness.1c5, %custom_witness.660, %custom_witness.809, %custom_witness.d56, %custom_witness.68a, %custom_witness.2ef) [concrete]
// CHECK:STDOUT:   %tuple.type.de6: type = tuple_type (%Iterator, %Iterator) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.8: type = fn_type @Destroy.WithSelf.SubobjectDestroy.3 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.8: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.8 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.10: type = fn_type @Destroy.WithSelf.Op.4 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.10: %Destroy.WithSelf.Op.type.ef016f.10 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.10: type = fn_type @Destroy.WithSelf.SelfDestruct.4 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.10: %Destroy.WithSelf.SelfDestruct.type.fbceb5.10 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.f8f19d.8: <witness> = custom_witness (%Destroy.WithSelf.Op.403171.10, %Destroy.WithSelf.SubobjectDestroy.d01daf.8, %Destroy.WithSelf.SelfDestruct.db3fdb.10), @Destroy [concrete]
// CHECK:STDOUT:   %Destroy.facet.c9a: %Destroy.type = facet_value %tuple.type.de6, (%custom_witness.f8f19d.8) [concrete]
// CHECK:STDOUT:   %Iterate.impl_witness.42b: <witness> = impl_witness imports.%Iterate.impl_witness_table.121, @T.as_type.as.Iterate.impl(%facet_value.cc4) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.type.d47: type = fn_type @T.as_type.as.Iterate.impl.NewCursor, @T.as_type.as.Iterate.impl(%facet_value.cc4) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.4ab: %T.as_type.as.Iterate.impl.NewCursor.type.d47 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.type.7d9: type = fn_type @T.as_type.as.Iterate.impl.Next, @T.as_type.as.Iterate.impl(%facet_value.cc4) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.214: %T.as_type.as.Iterate.impl.Next.type.7d9 = struct_value () [concrete]
// CHECK:STDOUT:   %Iterate.facet.c14: %Iterate.type = facet_value %MutableAndConstRange, (%Iterate.impl_witness.42b) [concrete]
// CHECK:STDOUT:   %facet_value.caf: %Iterate_where.type.8b0 = facet_value %MutableAndConstRange, (%Iterate.impl_witness.42b) [concrete]
// CHECK:STDOUT:   %r.param_patt.3a6: %pattern_type.fdb = value_param_pattern [concrete]
// CHECK:STDOUT:   %r.patt.403: %pattern_type.fdb = wrapper_binding_pattern r, %r.param_patt.3a6 [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.e77: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.c14) [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.c27: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.c14) [concrete]
// CHECK:STDOUT:   %.c8f: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.e77, %Iterate.facet.c14 [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.specific_fn: <specific function> = specific_function %T.as_type.as.Iterate.impl.NewCursor.4ab, @T.as_type.as.Iterate.impl.NewCursor(%facet_value.cc4) [concrete]
// CHECK:STDOUT:   %ptr.b95: type = ptr_type %tuple.type.de6 [concrete]
// CHECK:STDOUT:   %.d7e: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.c27, %Iterate.facet.c14 [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.specific_fn: <specific function> = specific_function %T.as_type.as.Iterate.impl.Next.214, @T.as_type.as.Iterate.impl.Next(%facet_value.cc4) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.c7c: type = fn_type @Destroy.WithSelf.SelfDestruct.1, @Destroy.WithSelf(%Destroy.facet.c9a) [concrete]
// CHECK:STDOUT:   %.c56: type = fn_type_with_self_type %Destroy.WithSelf.SelfDestruct.type.c7c, %Destroy.facet.c9a [concrete]
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
// CHECK:STDOUT: generated {
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.1: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc38_41.1 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.2] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.60e = ref_param_pattern [concrete = constants.%self.param_patt.61b3]
// CHECK:STDOUT:     %self.patt: %pattern_type.60e = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.ecc]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %.8f3 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %.8f3 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.2: %Destroy.WithSelf.Op.type.ef016f.3 = fn_decl @Destroy.WithSelf.Op.loc38_41.1 [concrete = constants.%Destroy.WithSelf.Op.403171.3] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.60e = ref_param_pattern [concrete = constants.%self.param_patt.61b3]
// CHECK:STDOUT:     %self.patt: %pattern_type.60e = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.ecc]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %.8f3 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %.8f3 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.2: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.3 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc38_41.2 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.3] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.e4c = ref_param_pattern [concrete = constants.%self.param_patt.cb2]
// CHECK:STDOUT:     %self.patt: %pattern_type.e4c = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.32b]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %MaybeUnformed.0af = ref_param call_param0
// CHECK:STDOUT:     %self: ref %MaybeUnformed.0af = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.3: %Destroy.WithSelf.Op.type.ef016f.4 = fn_decl @Destroy.WithSelf.Op.loc38_41.2 [concrete = constants.%Destroy.WithSelf.Op.403171.4] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.e4c = ref_param_pattern [concrete = constants.%self.param_patt.cb2]
// CHECK:STDOUT:     %self.patt: %pattern_type.e4c = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.32b]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %MaybeUnformed.0af = ref_param call_param0
// CHECK:STDOUT:     %self: ref %MaybeUnformed.0af = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.3: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.4 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc38_41.3 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.4] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.831 = ref_param_pattern [concrete = constants.%self.param_patt.40d]
// CHECK:STDOUT:     %self.patt: %pattern_type.831 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.560]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref bool = ref_param call_param0
// CHECK:STDOUT:     %self: ref bool = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.4: %Destroy.WithSelf.Op.type.ef016f.5 = fn_decl @Destroy.WithSelf.Op.loc38_41.3 [concrete = constants.%Destroy.WithSelf.Op.403171.5] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.831 = ref_param_pattern [concrete = constants.%self.param_patt.40d]
// CHECK:STDOUT:     %self.patt: %pattern_type.831 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.560]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref bool = ref_param call_param0
// CHECK:STDOUT:     %self: ref bool = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.4: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.5 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc38_41.4 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.5] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.95d = ref_param_pattern [concrete = constants.%self.param_patt.432]
// CHECK:STDOUT:     %self.patt: %pattern_type.95d = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.25a]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %struct_type.value.has_value.ac3 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %struct_type.value.has_value.ac3 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.5: %Destroy.WithSelf.Op.type.ef016f.6 = fn_decl @Destroy.WithSelf.Op.loc38_41.4 [concrete = constants.%Destroy.WithSelf.Op.403171.6] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.95d = ref_param_pattern [concrete = constants.%self.param_patt.432]
// CHECK:STDOUT:     %self.patt: %pattern_type.95d = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.25a]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %struct_type.value.has_value.ac3 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %struct_type.value.has_value.ac3 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.5: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.6 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc38_41.5 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.6] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.e96 = ref_param_pattern [concrete = constants.%self.param_patt.861]
// CHECK:STDOUT:     %self.patt: %pattern_type.e96 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.c75]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %DefaultOptionalStorage.58e = ref_param call_param0
// CHECK:STDOUT:     %self: ref %DefaultOptionalStorage.58e = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.6: %Destroy.WithSelf.Op.type.ef016f.7 = fn_decl @Destroy.WithSelf.Op.loc38_41.5 [concrete = constants.%Destroy.WithSelf.Op.403171.7] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.e96 = ref_param_pattern [concrete = constants.%self.param_patt.861]
// CHECK:STDOUT:     %self.patt: %pattern_type.e96 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.c75]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %DefaultOptionalStorage.58e = ref_param call_param0
// CHECK:STDOUT:     %self: ref %DefaultOptionalStorage.58e = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.6: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.7 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc38_41.6 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.7] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.540 = ref_param_pattern [concrete = constants.%self.param_patt.422]
// CHECK:STDOUT:     %self.patt: %pattern_type.540 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.f61]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %Optional.70f = ref_param call_param0
// CHECK:STDOUT:     %self: ref %Optional.70f = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.7: %Destroy.WithSelf.Op.type.ef016f.8 = fn_decl @Destroy.WithSelf.Op.loc38_41.6 [concrete = constants.%Destroy.WithSelf.Op.403171.8] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.540 = ref_param_pattern [concrete = constants.%self.param_patt.422]
// CHECK:STDOUT:     %self.patt: %pattern_type.540 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.f61]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %Optional.70f = ref_param call_param0
// CHECK:STDOUT:     %self: ref %Optional.70f = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @TestIterate(%R.loc34_17.2: %Iterate_where.type.8b0) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Iterate.lookup_impl_witness: <witness> = lookup_impl_witness %R.loc34_17.1, @Iterate [symbolic = %Iterate.lookup_impl_witness (constants.%Iterate.lookup_impl_witness.a99)]
// CHECK:STDOUT:   %Iterate.facet.loc38_41.5: %Iterate.type = facet_value %R.as_type.loc34_83.1, (%Iterate.lookup_impl_witness) [symbolic = %Iterate.facet.loc38_41.5 (constants.%Iterate.facet.c55)]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.loc38_41.5) [symbolic = %Iterate.WithSelf.NewCursor.type (constants.%Iterate.WithSelf.NewCursor.type.a6c)]
// CHECK:STDOUT:   %.loc38_41.20: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type, %Iterate.facet.loc38_41.5 [symbolic = %.loc38_41.20 (constants.%.57a)]
// CHECK:STDOUT:   %impl.elem2.loc38_41.3: @TestIterate.%.loc38_41.20 (%.57a) = impl_witness_access %Iterate.lookup_impl_witness, element2 [symbolic = %impl.elem2.loc38_41.3 (constants.%impl.elem2.51b)]
// CHECK:STDOUT:   %specific_impl_fn.loc38_41.4: <specific function> = specific_impl_function %impl.elem2.loc38_41.3, @Iterate.WithSelf.NewCursor(%Iterate.facet.loc38_41.5) [symbolic = %specific_impl_fn.loc38_41.4 (constants.%specific_impl_fn.986)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %ptr: type = ptr_type %as_type [symbolic = %ptr (constants.%ptr.696)]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.loc38_41.5) [symbolic = %Iterate.WithSelf.Next.type (constants.%Iterate.WithSelf.Next.type.cd7)]
// CHECK:STDOUT:   %.loc38_41.21: type = fn_type_with_self_type %Iterate.WithSelf.Next.type, %Iterate.facet.loc38_41.5 [symbolic = %.loc38_41.21 (constants.%.c1e)]
// CHECK:STDOUT:   %impl.elem3.loc38_41.2: @TestIterate.%.loc38_41.21 (%.c1e) = impl_witness_access %Iterate.lookup_impl_witness, element3 [symbolic = %impl.elem3.loc38_41.2 (constants.%impl.elem3.576)]
// CHECK:STDOUT:   %specific_impl_fn.loc38_41.5: <specific function> = specific_impl_function %impl.elem3.loc38_41.2, @Iterate.WithSelf.Next(%Iterate.facet.loc38_41.5) [symbolic = %specific_impl_fn.loc38_41.5 (constants.%specific_impl_fn.1db)]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type: type = fn_type @Destroy.WithSelf.SelfDestruct.1, @Destroy.WithSelf(%impl.elem1) [symbolic = %Destroy.WithSelf.SelfDestruct.type (constants.%Destroy.WithSelf.SelfDestruct.type.57f)]
// CHECK:STDOUT:   %.loc38_41.22: type = fn_type_with_self_type %Destroy.WithSelf.SelfDestruct.type, %impl.elem1 [symbolic = %.loc38_41.22 (constants.%.b9e)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %impl.elem1, @Destroy [symbolic = %Destroy.lookup_impl_witness (constants.%Destroy.lookup_impl_witness.eb0)]
// CHECK:STDOUT:   %impl.elem2.loc38_41.4: @TestIterate.%.loc38_41.22 (%.b9e) = impl_witness_access %Destroy.lookup_impl_witness, element2 [symbolic = %impl.elem2.loc38_41.4 (constants.%impl.elem2.570)]
// CHECK:STDOUT:   %specific_impl_fn.loc38_41.6: <specific function> = specific_impl_function %impl.elem2.loc38_41.4, @Destroy.WithSelf.SelfDestruct.1(%impl.elem1) [symbolic = %specific_impl_fn.loc38_41.6 (constants.%specific_impl_fn.74c)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%r.param: @TestIterate.%R.as_type.loc34_83.1 (%R.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %i.patt: %pattern_type.e09 = value_binding_pattern i [concrete = constants.%i.patt]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %r.ref: @TestIterate.%R.as_type.loc34_83.1 (%R.as_type) = name_ref r, %r
// CHECK:STDOUT:     %impl.elem2.loc38_41.1: @TestIterate.%.loc38_41.20 (%.57a) = impl_witness_access constants.%Iterate.lookup_impl_witness.a99, element2 [symbolic = %impl.elem2.loc38_41.3 (constants.%impl.elem2.51b)]
// CHECK:STDOUT:     %bound_method.loc38_41.1: <bound method> = bound_method %r.ref, %impl.elem2.loc38_41.1
// CHECK:STDOUT:     %Iterate.facet.loc38_41.1: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.a99) [symbolic = %Iterate.facet.loc38_41.5 (constants.%Iterate.facet.c55)]
// CHECK:STDOUT:     %.loc38_41.1: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc38_41.1 [symbolic = %Iterate.facet.loc38_41.5 (constants.%Iterate.facet.c55)]
// CHECK:STDOUT:     %Iterate.facet.loc38_41.2: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.a99) [symbolic = %Iterate.facet.loc38_41.5 (constants.%Iterate.facet.c55)]
// CHECK:STDOUT:     %.loc38_41.2: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc38_41.2 [symbolic = %Iterate.facet.loc38_41.5 (constants.%Iterate.facet.c55)]
// CHECK:STDOUT:     %specific_impl_fn.loc38_41.1: <specific function> = specific_impl_function %impl.elem2.loc38_41.1, @Iterate.WithSelf.NewCursor(constants.%Iterate.facet.c55) [symbolic = %specific_impl_fn.loc38_41.4 (constants.%specific_impl_fn.986)]
// CHECK:STDOUT:     %bound_method.loc38_41.2: <bound method> = bound_method %r.ref, %specific_impl_fn.loc38_41.1
// CHECK:STDOUT:     %var: ref @TestIterate.%as_type (%as_type.733) = var_storage invalid
// CHECK:STDOUT:     %Iterate.WithSelf.NewCursor.call: init @TestIterate.%as_type (%as_type.733) to %var = call %bound_method.loc38_41.2(%r.ref)
// CHECK:STDOUT:     assign %var, %Iterate.WithSelf.NewCursor.call
// CHECK:STDOUT:     br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.next:
// CHECK:STDOUT:     %addr.loc38: @TestIterate.%ptr (%ptr.696) = addr_of %var
// CHECK:STDOUT:     %impl.elem3.loc38_41.1: @TestIterate.%.loc38_41.21 (%.c1e) = impl_witness_access constants.%Iterate.lookup_impl_witness.a99, element3 [symbolic = %impl.elem3.loc38_41.2 (constants.%impl.elem3.576)]
// CHECK:STDOUT:     %bound_method.loc38_41.3: <bound method> = bound_method %r.ref, %impl.elem3.loc38_41.1
// CHECK:STDOUT:     %Iterate.facet.loc38_41.3: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.a99) [symbolic = %Iterate.facet.loc38_41.5 (constants.%Iterate.facet.c55)]
// CHECK:STDOUT:     %.loc38_41.3: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc38_41.3 [symbolic = %Iterate.facet.loc38_41.5 (constants.%Iterate.facet.c55)]
// CHECK:STDOUT:     %Iterate.facet.loc38_41.4: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.a99) [symbolic = %Iterate.facet.loc38_41.5 (constants.%Iterate.facet.c55)]
// CHECK:STDOUT:     %.loc38_41.4: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc38_41.4 [symbolic = %Iterate.facet.loc38_41.5 (constants.%Iterate.facet.c55)]
// CHECK:STDOUT:     %.loc38_41.5: %Destroy.type = converted constants.%as_type.733, constants.%impl.elem1.751 [symbolic = %impl.elem1 (constants.%impl.elem1.751)]
// CHECK:STDOUT:     %.loc38_41.6: %Destroy.type = converted constants.%as_type.733, constants.%impl.elem1.751 [symbolic = %impl.elem1 (constants.%impl.elem1.751)]
// CHECK:STDOUT:     %specific_impl_fn.loc38_41.2: <specific function> = specific_impl_function %impl.elem3.loc38_41.1, @Iterate.WithSelf.Next(constants.%Iterate.facet.c55) [symbolic = %specific_impl_fn.loc38_41.5 (constants.%specific_impl_fn.1db)]
// CHECK:STDOUT:     %bound_method.loc38_41.4: <bound method> = bound_method %r.ref, %specific_impl_fn.loc38_41.2
// CHECK:STDOUT:     %.loc38_41.7: ref %Optional.70f = temporary_storage
// CHECK:STDOUT:     %Iterate.WithSelf.Next.call: init %Optional.70f to %.loc38_41.7 = call %bound_method.loc38_41.4(%r.ref, %addr.loc38)
// CHECK:STDOUT:     %.loc38_41.8: ref %Optional.70f = temporary %.loc38_41.7, %Iterate.WithSelf.Next.call
// CHECK:STDOUT:     %.loc38_41.9: %Optional.HasValue.type.74b = specific_constant imports.%Core.import_ref.bb6, @Optional(constants.%OptionalStorage.facet.e68) [concrete = constants.%Optional.HasValue.f75]
// CHECK:STDOUT:     %HasValue.ref: %Optional.HasValue.type.74b = name_ref HasValue, %.loc38_41.9 [concrete = constants.%Optional.HasValue.f75]
// CHECK:STDOUT:     %Optional.HasValue.bound: <bound method> = bound_method %.loc38_41.8, %HasValue.ref
// CHECK:STDOUT:     %Optional.HasValue.specific_fn: <specific function> = specific_function %HasValue.ref, @Optional.HasValue(constants.%OptionalStorage.facet.e68) [concrete = constants.%Optional.HasValue.specific_fn.e2e]
// CHECK:STDOUT:     %bound_method.loc38_41.5: <bound method> = bound_method %.loc38_41.8, %Optional.HasValue.specific_fn
// CHECK:STDOUT:     %.loc38_41.10: %Optional.70f = acquire_value %.loc38_41.8
// CHECK:STDOUT:     %Optional.HasValue.call: init bool = call %bound_method.loc38_41.5(%.loc38_41.10)
// CHECK:STDOUT:     %.loc38_41.11: bool = value_of_initializer %Optional.HasValue.call
// CHECK:STDOUT:     %.loc38_41.12: bool = converted %Optional.HasValue.call, %.loc38_41.11
// CHECK:STDOUT:     if %.loc38_41.12 br !for.body else br !for.done
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.body:
// CHECK:STDOUT:     %.loc38_41.13: %Optional.Get.type.316 = specific_constant imports.%Core.import_ref.dab, @Optional(constants.%OptionalStorage.facet.e68) [concrete = constants.%Optional.Get.483]
// CHECK:STDOUT:     %Get.ref: %Optional.Get.type.316 = name_ref Get, %.loc38_41.13 [concrete = constants.%Optional.Get.483]
// CHECK:STDOUT:     %Optional.Get.bound: <bound method> = bound_method %.loc38_41.8, %Get.ref
// CHECK:STDOUT:     %Optional.Get.specific_fn: <specific function> = specific_function %Get.ref, @Optional.Get(constants.%OptionalStorage.facet.e68) [concrete = constants.%Optional.Get.specific_fn.296]
// CHECK:STDOUT:     %bound_method.loc38_41.6: <bound method> = bound_method %.loc38_41.8, %Optional.Get.specific_fn
// CHECK:STDOUT:     %.loc38_41.14: ref %ValueType = temporary_storage
// CHECK:STDOUT:     %.loc38_41.15: %Optional.70f = acquire_value %.loc38_41.8
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
// CHECK:STDOUT:     %Destroy.WithSelf.SelfDestruct.bound.loc38_41.1: <bound method> = bound_method %.loc38_41.16, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.2
// CHECK:STDOUT:     %Destroy.WithSelf.SelfDestruct.call.loc38_41.1: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc38_41.1(%.loc38_41.16)
// CHECK:STDOUT:     %Destroy.WithSelf.SelfDestruct.bound.loc38_41.2: <bound method> = bound_method %.loc38_41.8, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.8
// CHECK:STDOUT:     %Destroy.WithSelf.SelfDestruct.call.loc38_41.2: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc38_41.2(%.loc38_41.8)
// CHECK:STDOUT:     br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.done:
// CHECK:STDOUT:     %Destroy.WithSelf.SelfDestruct.bound.loc38_41.3: <bound method> = bound_method %.loc38_41.8, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.8
// CHECK:STDOUT:     %Destroy.WithSelf.SelfDestruct.call.loc38_41.3: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc38_41.3(%.loc38_41.8)
// CHECK:STDOUT:     %impl.elem2.loc38_41.2: @TestIterate.%.loc38_41.22 (%.b9e) = impl_witness_access constants.%Destroy.lookup_impl_witness.eb0, element2 [symbolic = %impl.elem2.loc38_41.4 (constants.%impl.elem2.570)]
// CHECK:STDOUT:     %bound_method.loc38_41.7: <bound method> = bound_method %var, %impl.elem2.loc38_41.2
// CHECK:STDOUT:     %.loc38_41.18: %Destroy.type = converted constants.%as_type.733, constants.%impl.elem1.751 [symbolic = %impl.elem1 (constants.%impl.elem1.751)]
// CHECK:STDOUT:     %.loc38_41.19: %Destroy.type = converted constants.%as_type.733, constants.%impl.elem1.751 [symbolic = %impl.elem1 (constants.%impl.elem1.751)]
// CHECK:STDOUT:     %specific_impl_fn.loc38_41.3: <specific function> = specific_impl_function %impl.elem2.loc38_41.2, @Destroy.WithSelf.SelfDestruct.1(constants.%impl.elem1.751) [symbolic = %specific_impl_fn.loc38_41.6 (constants.%specific_impl_fn.74c)]
// CHECK:STDOUT:     %bound_method.loc38_41.8: <bound method> = bound_method %var, %specific_impl_fn.loc38_41.3
// CHECK:STDOUT:     %Destroy.WithSelf.SelfDestruct.call.loc38_41.4: init %empty_tuple.type = call %bound_method.loc38_41.8(%var)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc38_41.1(%self.param: ref %.8f3) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc38_41.1(%self.param: ref %.8f3) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc38_41.1(%self.param: ref %.8f3) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.2(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.1(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc38_41.2(%self.param: ref %MaybeUnformed.0af) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc38_41.2(%self.param: ref %MaybeUnformed.0af) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc38_41.2(%self.param: ref %MaybeUnformed.0af) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.3(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.2(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc38_41.3(%self.param: ref bool) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc38_41.3(%self.param: ref bool) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc38_41.3(%self.param: ref bool) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.4(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.3(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc38_41.4(%self.param: ref %struct_type.value.has_value.ac3) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc38_41.1: ref %MaybeUnformed.0af = struct_access %self.param, element0
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc38_41.1: <bound method> = bound_method %.loc38_41.1, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.4
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc38_41.1: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc38_41.1(%.loc38_41.1)
// CHECK:STDOUT:   %.loc38_41.2: ref bool = struct_access %self.param, element1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc38_41.2: <bound method> = bound_method %.loc38_41.2, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.5
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc38_41.2: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc38_41.2(%.loc38_41.2)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc38_41.4(%self.param: ref %struct_type.value.has_value.ac3) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc38_41.4(%self.param: ref %struct_type.value.has_value.ac3) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.5(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.4(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc38_41.5(%self.param: ref %DefaultOptionalStorage.58e) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc38_41.5(%self.param: ref %DefaultOptionalStorage.58e) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc38_41.5(%self.param: ref %DefaultOptionalStorage.58e) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.6(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.5(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc38_41.6(%self.param: ref %Optional.70f) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc38_41.6(%self.param: ref %Optional.70f) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc38_41.6(%self.param: ref %Optional.70f) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.7(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.6(%self.param)
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
// CHECK:STDOUT:   %impl.elem2: %.c8f = impl_witness_access constants.%Iterate.impl_witness.42b, element2 [concrete = constants.%T.as_type.as.Iterate.impl.NewCursor.4ab]
// CHECK:STDOUT:   %bound_method.loc80_42.1: <bound method> = bound_method %mc.ref, %impl.elem2
// CHECK:STDOUT:   %facet_value.loc80_42.1: %CppRange_where.type.d682d6.1 = facet_value constants.%MutableAndConstRange, (constants.%custom_witness.dc3, constants.%custom_witness.1c5, constants.%custom_witness.660, constants.%custom_witness.809, constants.%custom_witness.d56, constants.%custom_witness.68a, constants.%custom_witness.2ef) [concrete = constants.%facet_value.cc4]
// CHECK:STDOUT:   %.loc80_42.1: %CppRange_where.type.d682d6.1 = converted constants.%MutableAndConstRange, %facet_value.loc80_42.1 [concrete = constants.%facet_value.cc4]
// CHECK:STDOUT:   %facet_value.loc80_42.2: %CppRange_where.type.d682d6.1 = facet_value constants.%MutableAndConstRange, (constants.%custom_witness.dc3, constants.%custom_witness.1c5, constants.%custom_witness.660, constants.%custom_witness.809, constants.%custom_witness.d56, constants.%custom_witness.68a, constants.%custom_witness.2ef) [concrete = constants.%facet_value.cc4]
// CHECK:STDOUT:   %.loc80_42.2: %CppRange_where.type.d682d6.1 = converted constants.%MutableAndConstRange, %facet_value.loc80_42.2 [concrete = constants.%facet_value.cc4]
// CHECK:STDOUT:   %specific_fn.loc80_42.1: <specific function> = specific_function %impl.elem2, @T.as_type.as.Iterate.impl.NewCursor(constants.%facet_value.cc4) [concrete = constants.%T.as_type.as.Iterate.impl.NewCursor.specific_fn]
// CHECK:STDOUT:   %bound_method.loc80_42.2: <bound method> = bound_method %mc.ref, %specific_fn.loc80_42.1
// CHECK:STDOUT:   %var: ref %tuple.type.de6 = var_storage invalid
// CHECK:STDOUT:   %.loc80_40.1: %MutableAndConstRange = acquire_value %mc.ref
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.call: init %tuple.type.de6 to %var = call %bound_method.loc80_42.2(%.loc80_40.1)
// CHECK:STDOUT:   assign %var, %T.as_type.as.Iterate.impl.NewCursor.call
// CHECK:STDOUT:   br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT: !for.next:
// CHECK:STDOUT:   %addr.loc80: %ptr.b95 = addr_of %var
// CHECK:STDOUT:   %impl.elem3: %.d7e = impl_witness_access constants.%Iterate.impl_witness.42b, element3 [concrete = constants.%T.as_type.as.Iterate.impl.Next.214]
// CHECK:STDOUT:   %bound_method.loc80_42.3: <bound method> = bound_method %mc.ref, %impl.elem3
// CHECK:STDOUT:   %facet_value.loc80_42.3: %CppRange_where.type.d682d6.1 = facet_value constants.%MutableAndConstRange, (constants.%custom_witness.dc3, constants.%custom_witness.1c5, constants.%custom_witness.660, constants.%custom_witness.809, constants.%custom_witness.d56, constants.%custom_witness.68a, constants.%custom_witness.2ef) [concrete = constants.%facet_value.cc4]
// CHECK:STDOUT:   %.loc80_42.3: %CppRange_where.type.d682d6.1 = converted constants.%MutableAndConstRange, %facet_value.loc80_42.3 [concrete = constants.%facet_value.cc4]
// CHECK:STDOUT:   %facet_value.loc80_42.4: %CppRange_where.type.d682d6.1 = facet_value constants.%MutableAndConstRange, (constants.%custom_witness.dc3, constants.%custom_witness.1c5, constants.%custom_witness.660, constants.%custom_witness.809, constants.%custom_witness.d56, constants.%custom_witness.68a, constants.%custom_witness.2ef) [concrete = constants.%facet_value.cc4]
// CHECK:STDOUT:   %.loc80_42.4: %CppRange_where.type.d682d6.1 = converted constants.%MutableAndConstRange, %facet_value.loc80_42.4 [concrete = constants.%facet_value.cc4]
// CHECK:STDOUT:   %specific_fn.loc80_42.2: <specific function> = specific_function %impl.elem3, @T.as_type.as.Iterate.impl.Next(constants.%facet_value.cc4) [concrete = constants.%T.as_type.as.Iterate.impl.Next.specific_fn]
// CHECK:STDOUT:   %bound_method.loc80_42.4: <bound method> = bound_method %mc.ref, %specific_fn.loc80_42.2
// CHECK:STDOUT:   %.loc80_42.5: ref %Optional.70f = temporary_storage
// CHECK:STDOUT:   %.loc80_40.2: %MutableAndConstRange = acquire_value %mc.ref
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.call: init %Optional.70f to %.loc80_42.5 = call %bound_method.loc80_42.4(%.loc80_40.2, %addr.loc80)
// CHECK:STDOUT:   %.loc80_42.6: ref %Optional.70f = temporary %.loc80_42.5, %T.as_type.as.Iterate.impl.Next.call
// CHECK:STDOUT:   %.loc80_42.7: %Optional.HasValue.type.74b = specific_constant imports.%Core.import_ref.bb6, @Optional(constants.%OptionalStorage.facet.e68) [concrete = constants.%Optional.HasValue.f75]
// CHECK:STDOUT:   %HasValue.ref: %Optional.HasValue.type.74b = name_ref HasValue, %.loc80_42.7 [concrete = constants.%Optional.HasValue.f75]
// CHECK:STDOUT:   %Optional.HasValue.bound: <bound method> = bound_method %.loc80_42.6, %HasValue.ref
// CHECK:STDOUT:   %Optional.HasValue.specific_fn: <specific function> = specific_function %HasValue.ref, @Optional.HasValue(constants.%OptionalStorage.facet.e68) [concrete = constants.%Optional.HasValue.specific_fn.e2e]
// CHECK:STDOUT:   %bound_method.loc80_42.5: <bound method> = bound_method %.loc80_42.6, %Optional.HasValue.specific_fn
// CHECK:STDOUT:   %.loc80_42.8: %Optional.70f = acquire_value %.loc80_42.6
// CHECK:STDOUT:   %Optional.HasValue.call: init bool = call %bound_method.loc80_42.5(%.loc80_42.8)
// CHECK:STDOUT:   %.loc80_42.9: bool = value_of_initializer %Optional.HasValue.call
// CHECK:STDOUT:   %.loc80_42.10: bool = converted %Optional.HasValue.call, %.loc80_42.9
// CHECK:STDOUT:   if %.loc80_42.10 br !for.body else br !for.done
// CHECK:STDOUT:
// CHECK:STDOUT: !for.body:
// CHECK:STDOUT:   %.loc80_42.11: %Optional.Get.type.316 = specific_constant imports.%Core.import_ref.dab, @Optional(constants.%OptionalStorage.facet.e68) [concrete = constants.%Optional.Get.483]
// CHECK:STDOUT:   %Get.ref: %Optional.Get.type.316 = name_ref Get, %.loc80_42.11 [concrete = constants.%Optional.Get.483]
// CHECK:STDOUT:   %Optional.Get.bound: <bound method> = bound_method %.loc80_42.6, %Get.ref
// CHECK:STDOUT:   %Optional.Get.specific_fn: <specific function> = specific_function %Get.ref, @Optional.Get(constants.%OptionalStorage.facet.e68) [concrete = constants.%Optional.Get.specific_fn.296]
// CHECK:STDOUT:   %bound_method.loc80_42.6: <bound method> = bound_method %.loc80_42.6, %Optional.Get.specific_fn
// CHECK:STDOUT:   %.loc80_42.12: ref %ValueType = temporary_storage
// CHECK:STDOUT:   %.loc80_42.13: %Optional.70f = acquire_value %.loc80_42.6
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
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc80_42.1: <bound method> = bound_method %.loc80_42.14, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.2
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc80_42.1: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc80_42.1(%.loc80_42.14)
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc80_42.2: <bound method> = bound_method %.loc80_42.6, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.8
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc80_42.2: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc80_42.2(%.loc80_42.6)
// CHECK:STDOUT:   br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT: !for.done:
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc80_42.3: <bound method> = bound_method %.loc80_42.6, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.8
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc80_42.3: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc80_42.3(%.loc80_42.6)
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc80_42.4: <bound method> = bound_method %var, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.10
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc80_42.4: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc80_42.4(%var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestIterate(constants.%R) {
// CHECK:STDOUT:   %R.patt.loc34_17.2 => constants.%R.patt
// CHECK:STDOUT:   %R.loc34_17.1 => constants.%R
// CHECK:STDOUT:   %R.as_type.loc34_83.1 => constants.%R.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.611
// CHECK:STDOUT:   %r.param_patt.loc34_81.2 => constants.%r.param_patt.e64
// CHECK:STDOUT:   %r.patt.loc34_81.2 => constants.%r.patt.dcb
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestIterate(constants.%facet_value.caf) {
// CHECK:STDOUT:   %R.patt.loc34_17.2 => constants.%R.patt
// CHECK:STDOUT:   %R.loc34_17.1 => constants.%facet_value.caf
// CHECK:STDOUT:   %R.as_type.loc34_83.1 => constants.%MutableAndConstRange
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.fdb
// CHECK:STDOUT:   %r.param_patt.loc34_81.2 => constants.%r.param_patt.3a6
// CHECK:STDOUT:   %r.patt.loc34_81.2 => constants.%r.patt.403
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc34 => constants.%complete_type.357
// CHECK:STDOUT:   %Iterate.lookup_impl_witness => constants.%Iterate.impl_witness.42b
// CHECK:STDOUT:   %Iterate.facet.loc38_41.5 => constants.%Iterate.facet.c14
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type => constants.%Iterate.WithSelf.NewCursor.type.e77
// CHECK:STDOUT:   %.loc38_41.20 => constants.%.c8f
// CHECK:STDOUT:   %impl.elem2.loc38_41.3 => constants.%T.as_type.as.Iterate.impl.NewCursor.4ab
// CHECK:STDOUT:   %specific_impl_fn.loc38_41.4 => constants.%T.as_type.as.Iterate.impl.NewCursor.specific_fn
// CHECK:STDOUT:   %impl.elem1 => constants.%Destroy.facet.c9a
// CHECK:STDOUT:   %as_type => constants.%tuple.type.de6
// CHECK:STDOUT:   %require_complete.1 => constants.%complete_type.95d
// CHECK:STDOUT:   %ptr => constants.%ptr.b95
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type => constants.%Iterate.WithSelf.Next.type.c27
// CHECK:STDOUT:   %.loc38_41.21 => constants.%.d7e
// CHECK:STDOUT:   %impl.elem3.loc38_41.2 => constants.%T.as_type.as.Iterate.impl.Next.214
// CHECK:STDOUT:   %specific_impl_fn.loc38_41.5 => constants.%T.as_type.as.Iterate.impl.Next.specific_fn
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type => constants.%Destroy.WithSelf.SelfDestruct.type.c7c
// CHECK:STDOUT:   %.loc38_41.22 => constants.%.c56
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.f8f19d.8
// CHECK:STDOUT:   %impl.elem2.loc38_41.4 => constants.%Destroy.WithSelf.SelfDestruct.db3fdb.10
// CHECK:STDOUT:   %specific_impl_fn.loc38_41.6 => constants.%Destroy.WithSelf.SelfDestruct.db3fdb.10
// CHECK:STDOUT: }
// CHECK:STDOUT:
