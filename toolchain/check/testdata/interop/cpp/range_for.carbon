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
  // CHECK:STDERR: fail_todo_adl_returns_different_types_mutable.carbon:[[@LINE+7]]:3: error: cannot convert type `Cpp.DiffMutableADL.MutableRange` into type implementing `Core.Iterate where .(Core.Iterate.ElementType) = Cpp.DiffMutableADL.ValueType as Core.SubobjectDestroy & Core.DestroyImpl & Core.Copy` [ConversionFailureTypeToFacet]
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
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %facet_type.d61: type = facet_type <@SubobjectDestroy & @DestroyImpl & @Copy> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %T.220: %OptionalStorage.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Optional.Get.type.d97: type = fn_type @Optional.Get, @Optional(%T.220) [symbolic]
// CHECK:STDOUT:   %Optional.Get.3e9: %Optional.Get.type.d97 = struct_value () [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.type.cef: type = fn_type @Optional.HasValue, @Optional(%T.220) [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.a74: %Optional.HasValue.type.cef = struct_value () [symbolic]
// CHECK:STDOUT:   %facet_type.8a3: type = facet_type <@SubobjectDestroy & @DestroyImpl> [concrete]
// CHECK:STDOUT:   %T.9ab: %facet_type.d61 = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.19c: type = facet_access_type %T.9ab [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.lookup_impl_witness.855: <witness> = lookup_impl_witness %T.9ab, @SubobjectDestroy [symbolic]
// CHECK:STDOUT:   %DestroyImpl.lookup_impl_witness.ff9: <witness> = lookup_impl_witness %T.9ab, @DestroyImpl [symbolic]
// CHECK:STDOUT:   %facet_value.b35: %facet_type.8a3 = facet_value %T.as_type.19c, (%SubobjectDestroy.lookup_impl_witness.855, %DestroyImpl.lookup_impl_witness.ff9) [symbolic]
// CHECK:STDOUT:   %MaybeUnformed.413: type = class_type @MaybeUnformed, @MaybeUnformed(%facet_value.b35) [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.a6b: type = class_type @DefaultOptionalStorage, @DefaultOptionalStorage(%T.9ab) [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.elem.94e: type = unbound_element_type %DefaultOptionalStorage.a6b, bool [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.elem.c3f: type = unbound_element_type %DefaultOptionalStorage.a6b, %MaybeUnformed.413 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Get.type.b32: type = fn_type @T.as_type.as.OptionalStorage.impl.Get, @T.as_type.as.OptionalStorage.impl(%T.9ab) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Get.8b0: %T.as_type.as.OptionalStorage.impl.Get.type.b32 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Has.type.8e5: type = fn_type @T.as_type.as.OptionalStorage.impl.Has, @T.as_type.as.OptionalStorage.impl(%T.9ab) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Has.2d5: %T.as_type.as.OptionalStorage.impl.Has.type.8e5 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Some.type.61d: type = fn_type @T.as_type.as.OptionalStorage.impl.Some, @T.as_type.as.OptionalStorage.impl(%T.9ab) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Some.ba7: %T.as_type.as.OptionalStorage.impl.Some.type.61d = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.None.type.a9f: type = fn_type @T.as_type.as.OptionalStorage.impl.None, @T.as_type.as.OptionalStorage.impl(%T.9ab) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.None.083: %T.as_type.as.OptionalStorage.impl.None.type.a9f = struct_value () [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %N.fe9: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %Int.682b94.1: type = class_type @Int, @Int(%N.fe9) [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.df9: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.1: type = fn_type @SubobjectDestroy.WithSelf.Op.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.1: %SubobjectDestroy.WithSelf.Op.type.6870fd.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.1: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.1), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.273: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.5b5: %T.as_type.as.DestroyImpl.impl.Op.type.273 = struct_value () [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.facet.6df: %SubobjectDestroy.type = facet_value %i32, (%custom_witness.b80ef6.1) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.52c: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.6df) [concrete]
// CHECK:STDOUT:   %facet_value.e43: %facet_type.8a3 = facet_value %i32, (%custom_witness.b80ef6.1, %DestroyImpl.impl_witness.52c) [concrete]
// CHECK:STDOUT:   %.Self.500: %Iterate.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Int.682b94.2: type = class_type @Int, @Int(%To) [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.cb7: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
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
// CHECK:STDOUT:   %SubobjectDestroy.lookup_impl_witness.dc9: <witness> = lookup_impl_witness %tuple.type.1ba, @SubobjectDestroy [symbolic]
// CHECK:STDOUT:   %DestroyImpl.lookup_impl_witness.504: <witness> = lookup_impl_witness %tuple.type.1ba, @DestroyImpl [symbolic]
// CHECK:STDOUT:   %facet_value.bf6: %facet_type.8a3 = facet_value %tuple.type.1ba, (%SubobjectDestroy.lookup_impl_witness.dc9, %DestroyImpl.lookup_impl_witness.504) [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.lookup_impl_witness.b15da7.2: <witness> = lookup_impl_witness %impl.elem0.493765.2, @SubobjectDestroy [symbolic]
// CHECK:STDOUT:   %DestroyImpl.lookup_impl_witness.4e7f44.2: <witness> = lookup_impl_witness %impl.elem0.493765.2, @DestroyImpl [symbolic]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.167: <witness> = lookup_impl_witness %impl.elem0.493765.2, @Copy [symbolic]
// CHECK:STDOUT:   %facet_value.9da: %facet_type.d61 = facet_value %impl.elem0.493765.2, (%SubobjectDestroy.lookup_impl_witness.b15da7.2, %DestroyImpl.lookup_impl_witness.4e7f44.2, %Copy.lookup_impl_witness.167) [symbolic]
// CHECK:STDOUT:   %SelfDestruct.type: type = fn_type @SelfDestruct [concrete]
// CHECK:STDOUT:   %SelfDestruct: %SelfDestruct.type = struct_value () [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.b5e: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N.fe9) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.f38: %Int.as.Copy.impl.Op.type.b5e = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.4c3: <witness> = impl_witness imports.%Copy.impl_witness_table.c8d, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %facet_value.c8d: %facet_type.d61 = facet_value %i32, (%custom_witness.b80ef6.1, %DestroyImpl.impl_witness.52c, %Copy.impl_witness.4c3) [concrete]
// CHECK:STDOUT:   %Iterate.lookup_impl_witness.1d8: <witness> = lookup_impl_witness %.Self.500, @Iterate [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.8a4: %facet_type.d61 = impl_witness_access %Iterate.lookup_impl_witness.1d8, element0 [symbolic_self]
// CHECK:STDOUT:   %Iterate_where.type.7a4: type = facet_type <@Iterate where %impl.elem0.8a4 = %facet_value.c8d> [concrete]
// CHECK:STDOUT:   %pattern_type.e85: type = pattern_type %Iterate_where.type.7a4 [concrete]
// CHECK:STDOUT:   %R.patt: %pattern_type.e85 = symbolic_binding_pattern R, 0 [symbolic]
// CHECK:STDOUT:   %R: %Iterate_where.type.7a4 = symbolic_binding R, 0 [symbolic]
// CHECK:STDOUT:   %R.as_type: type = facet_access_type %R [symbolic]
// CHECK:STDOUT:   %pattern_type.249: type = pattern_type %R.as_type [symbolic]
// CHECK:STDOUT:   %r.param_patt.8a5: %pattern_type.249 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %r.patt.f98: %pattern_type.249 = wrapper_binding_pattern r, %r.param_patt.8a5 [symbolic]
// CHECK:STDOUT:   %facet_type.425: type = facet_type <@IntFitsIn, @IntFitsIn(%Int.682b94.2) & @AnyInt> [symbolic]
// CHECK:STDOUT:   %From.157: %facet_type.425 = symbolic_binding From, 1 [symbolic]
// CHECK:STDOUT:   %From.as_type.as.ImplicitAs.impl.Convert.type.58f: type = fn_type @From.as_type.as.ImplicitAs.impl.Convert, @From.as_type.as.ImplicitAs.impl(%To, %From.157) [symbolic]
// CHECK:STDOUT:   %From.as_type.as.ImplicitAs.impl.Convert.aad: %From.as_type.as.ImplicitAs.impl.Convert.type.58f = struct_value () [symbolic]
// CHECK:STDOUT:   %i.patt: %pattern_type.df9 = value_binding_pattern i [concrete]
// CHECK:STDOUT:   %Iterate.lookup_impl_witness.a61: <witness> = lookup_impl_witness %R, @Iterate [symbolic]
// CHECK:STDOUT:   %Iterate.facet.da6: %Iterate.type = facet_value %R.as_type, (%Iterate.lookup_impl_witness.a61) [symbolic]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.bf7: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.da6) [symbolic]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.f9b: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.da6) [symbolic]
// CHECK:STDOUT:   %.fb4: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.bf7, %Iterate.facet.da6 [symbolic]
// CHECK:STDOUT:   %impl.elem2.069: %.fb4 = impl_witness_access %Iterate.lookup_impl_witness.a61, element2 [symbolic]
// CHECK:STDOUT:   %impl.elem1.f53: %facet_type.8a3 = impl_witness_access %Iterate.lookup_impl_witness.a61, element1 [symbolic]
// CHECK:STDOUT:   %as_type.d8a: type = facet_access_type %impl.elem1.f53 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.a50: <specific function> = specific_impl_function %impl.elem2.069, @Iterate.WithSelf.NewCursor(%Iterate.facet.da6) [symbolic]
// CHECK:STDOUT:   %.f41e: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.f9b, %Iterate.facet.da6 [symbolic]
// CHECK:STDOUT:   %impl.elem3.735: %.f41e = impl_witness_access %Iterate.lookup_impl_witness.a61, element3 [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.84b: type = class_type @DefaultOptionalStorage, @DefaultOptionalStorage(%facet_value.c8d) [concrete]
// CHECK:STDOUT:   %OptionalStorage.impl_witness.434: <witness> = impl_witness imports.%OptionalStorage.impl_witness_table.968, @T.as_type.as.OptionalStorage.impl(%facet_value.c8d) [concrete]
// CHECK:STDOUT:   %OptionalStorage.facet.7f6: %OptionalStorage.type = facet_value %i32, (%OptionalStorage.impl_witness.434) [concrete]
// CHECK:STDOUT:   %Optional.656: type = class_type @Optional, @Optional(%OptionalStorage.facet.7f6) [concrete]
// CHECK:STDOUT:   %specific_impl_fn.5aa: <specific function> = specific_impl_function %impl.elem3.735, @Iterate.WithSelf.Next(%Iterate.facet.da6) [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.type.201: type = fn_type @Optional.HasValue, @Optional(%OptionalStorage.facet.7f6) [concrete]
// CHECK:STDOUT:   %Optional.HasValue.392: %Optional.HasValue.type.201 = struct_value () [concrete]
// CHECK:STDOUT:   %Optional.Get.type.059: type = fn_type @Optional.Get, @Optional(%OptionalStorage.facet.7f6) [concrete]
// CHECK:STDOUT:   %Optional.Get.3cd: %Optional.Get.type.059 = struct_value () [concrete]
// CHECK:STDOUT:   %MaybeUnformed.3cf: type = class_type @MaybeUnformed, @MaybeUnformed(%facet_value.e43) [concrete]
// CHECK:STDOUT:   %.1ff: type = maybe_unformed_type %i32 [concrete]
// CHECK:STDOUT:   %DefaultOptionalStorage.elem.8fe: type = unbound_element_type %DefaultOptionalStorage.84b, %MaybeUnformed.3cf [concrete]
// CHECK:STDOUT:   %DefaultOptionalStorage.elem.4e2: type = unbound_element_type %DefaultOptionalStorage.84b, bool [concrete]
// CHECK:STDOUT:   %struct_type.value.has_value.00c: type = struct_type {.value: %MaybeUnformed.3cf, .has_value: bool} [concrete]
// CHECK:STDOUT:   %Optional.HasValue.specific_fn.3a9: <specific function> = specific_function %Optional.HasValue.392, @Optional.HasValue(%OptionalStorage.facet.7f6) [concrete]
// CHECK:STDOUT:   %Optional.Get.specific_fn.8ba: <specific function> = specific_function %Optional.Get.3cd, @Optional.Get(%OptionalStorage.facet.7f6) [concrete]
// CHECK:STDOUT:   %AddAssignWith.type.7cc: type = facet_type <@AddAssignWith, @AddAssignWith(%i32)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.35985e.2: type = facet_type <@ImplicitAs, @ImplicitAs(%Int.682b94.1)> [symbolic]
// CHECK:STDOUT:   %U.500: %ImplicitAs.type.35985e.2 = symbolic_binding U, 1 [symbolic]
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.type.e95403.1: type = fn_type @Int.as.AddAssignWith.impl.Op.1, @Int.as.AddAssignWith.impl(%N.fe9, %U.500) [symbolic]
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.78e00b.1: %Int.as.AddAssignWith.impl.Op.type.e95403.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.type.e95403.2: type = fn_type @Int.as.AddAssignWith.impl.Op.2, @Int.as.AddAssignWith.impl(%N.fe9, %U.500) [symbolic]
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.78e00b.2: %Int.as.AddAssignWith.impl.Op.type.e95403.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %facet_type.733: type = facet_type <@IntFitsIn, @IntFitsIn(%i32) & @AnyInt> [concrete]
// CHECK:STDOUT:   %custom_witness.39d: <witness> = custom_witness (), @IntFitsIn, @IntFitsIn(%i32) [concrete]
// CHECK:STDOUT:   %Int.as.AnyInt.impl.AsInt.type.425: type = fn_type @Int.as.AnyInt.impl.AsInt, @Int.as.AnyInt.impl(%N.fe9) [symbolic]
// CHECK:STDOUT:   %Int.as.AnyInt.impl.AsInt.830: %Int.as.AnyInt.impl.AsInt.type.425 = struct_value () [symbolic]
// CHECK:STDOUT:   %AnyInt.impl_witness.6f2: <witness> = impl_witness imports.%AnyInt.impl_witness_table, @Int.as.AnyInt.impl(%int_32) [concrete]
// CHECK:STDOUT:   %facet_value.004: %facet_type.733 = facet_value %i32, (%custom_witness.39d, %AnyInt.impl_witness.6f2) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.c7b: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.85f, @From.as_type.as.ImplicitAs.impl(%int_32, %facet_value.004) [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.749: %ImplicitAs.type.cb7 = facet_value %i32, (%ImplicitAs.impl_witness.c7b) [concrete]
// CHECK:STDOUT:   %AddAssignWith.impl_witness.f97: <witness> = impl_witness imports.%AddAssignWith.impl_witness_table, @Int.as.AddAssignWith.impl(%int_32, %ImplicitAs.facet.749) [concrete]
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.type.ae3c0a.1: type = fn_type @Int.as.AddAssignWith.impl.Op.2, @Int.as.AddAssignWith.impl(%int_32, %ImplicitAs.facet.749) [concrete]
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.e1a1fc.1: %Int.as.AddAssignWith.impl.Op.type.ae3c0a.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.type.ae3c0a.2: type = fn_type @Int.as.AddAssignWith.impl.Op.1, @Int.as.AddAssignWith.impl(%int_32, %ImplicitAs.facet.749) [concrete]
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.e1a1fc.2: %Int.as.AddAssignWith.impl.Op.type.ae3c0a.2 = struct_value () [concrete]
// CHECK:STDOUT:   %AddAssignWith.facet.3b4: %AddAssignWith.type.7cc = facet_value %i32, (%AddAssignWith.impl_witness.f97) [concrete]
// CHECK:STDOUT:   %AddAssignWith.WithSelf.Op.type.340: type = fn_type @AddAssignWith.WithSelf.Op, @AddAssignWith.WithSelf(%i32, %AddAssignWith.facet.3b4) [concrete]
// CHECK:STDOUT:   %.2a7: type = fn_type_with_self_type %AddAssignWith.WithSelf.Op.type.340, %AddAssignWith.facet.3b4 [concrete]
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.specific_fn.9595aa.1: <specific function> = specific_function %Int.as.AddAssignWith.impl.Op.e1a1fc.2, @Int.as.AddAssignWith.impl.Op.1(%int_32, %ImplicitAs.facet.749) [concrete]
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.specific_fn.9595aa.2: <specific function> = specific_function %Int.as.AddAssignWith.impl.Op.e1a1fc.1, @Int.as.AddAssignWith.impl.Op.2(%int_32, %ImplicitAs.facet.749) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.3: type = fn_type @SubobjectDestroy.WithSelf.Op.loc34_19.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.3: %SubobjectDestroy.WithSelf.Op.type.6870fd.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.3: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.3), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.4: type = fn_type @SubobjectDestroy.WithSelf.Op.loc34_19.3 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.4: %SubobjectDestroy.WithSelf.Op.type.6870fd.4 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.4: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.4), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.a87: %SubobjectDestroy.type = facet_value bool, (%custom_witness.b80ef6.4) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.deb: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.a87) [concrete]
// CHECK:STDOUT:   %facet_value.a1b: %facet_type.8a3 = facet_value bool, (%custom_witness.b80ef6.4, %DestroyImpl.impl_witness.deb) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.5e1: %SubobjectDestroy.type = facet_value %MaybeUnformed.3cf, (%custom_witness.b80ef6.3) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.a00: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.5e1) [concrete]
// CHECK:STDOUT:   %facet_value.c63: %facet_type.8a3 = facet_value %MaybeUnformed.3cf, (%custom_witness.b80ef6.3, %DestroyImpl.impl_witness.a00) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.7: type = fn_type @SubobjectDestroy.WithSelf.Op.loc34_19.6 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.7: %SubobjectDestroy.WithSelf.Op.type.6870fd.7 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.7: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.7), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.d1d: %SubobjectDestroy.type = facet_value %Optional.656, (%custom_witness.b80ef6.7) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.06e: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.d1d) [concrete]
// CHECK:STDOUT:   %facet_value.d9c: %facet_type.8a3 = facet_value %Optional.656, (%custom_witness.b80ef6.7, %DestroyImpl.impl_witness.06e) [concrete]
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
// CHECK:STDOUT:   %custom_witness.52e: <witness> = custom_witness (%i32, %Iterator.Op.a2a866.1), @CppUnsafeDeref [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.0aed5d.2: type = fn_type @Iterator.Op.2 [concrete]
// CHECK:STDOUT:   %Iterator.Op.a2a866.2: %Iterator.Op.type.0aed5d.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.da7: <witness> = custom_witness (%Iterator.Op.a2a866.2), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.2d2: %SubobjectDestroy.type = facet_value %Iterator.3ee, (%custom_witness.da7) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.048: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.2d2) [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.0aed5d.3: type = fn_type @Iterator.Op.3 [concrete]
// CHECK:STDOUT:   %Iterator.Op.a2a866.3: %Iterator.Op.type.0aed5d.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.444: <witness> = custom_witness (%Iterator.Op.a2a866.3), @Inc [concrete]
// CHECK:STDOUT:   %Equal.type.ca8475.1: type = fn_type @Equal.1 [concrete]
// CHECK:STDOUT:   %Equal.285b15.1: %Equal.type.ca8475.1 = struct_value () [concrete]
// CHECK:STDOUT:   %NotEqual.type.53d9ee.1: type = fn_type @NotEqual.1 [concrete]
// CHECK:STDOUT:   %NotEqual.119263.1: %NotEqual.type.53d9ee.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.981: <witness> = custom_witness (%Equal.285b15.1, %NotEqual.119263.1), @EqWith, @EqWith(%Iterator.3ee) [concrete]
// CHECK:STDOUT:   %facet_value.88a: %CppRange_where.type.d682d6.1 = facet_value %MutableRange, (%custom_witness.b80ef6.1, %DestroyImpl.impl_witness.52c, %Copy.impl_witness.4c3, %custom_witness.ed3, %custom_witness.da7, %DestroyImpl.impl_witness.048, %custom_witness.52e, %custom_witness.444, %custom_witness.981) [concrete]
// CHECK:STDOUT:   %tuple.type.508: type = tuple_type (%Iterator.3ee, %Iterator.3ee) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.8: type = fn_type @SubobjectDestroy.WithSelf.Op.3 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.8: %SubobjectDestroy.WithSelf.Op.type.6870fd.8 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.8: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.8), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.669: %SubobjectDestroy.type = facet_value %tuple.type.508, (%custom_witness.b80ef6.8) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.b78: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.669) [concrete]
// CHECK:STDOUT:   %complete_type.c28: <witness> = complete_type_witness %tuple.type.508 [concrete]
// CHECK:STDOUT:   %facet_value.ddb: %facet_type.8a3 = facet_value %tuple.type.508, (%custom_witness.b80ef6.8, %DestroyImpl.impl_witness.b78) [concrete]
// CHECK:STDOUT:   %Iterate.impl_witness.581: <witness> = impl_witness imports.%Iterate.impl_witness_table.121, @T.as_type.as.Iterate.impl(%facet_value.88a) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.type.455: type = fn_type @T.as_type.as.Iterate.impl.NewCursor, @T.as_type.as.Iterate.impl(%facet_value.88a) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.18b: %T.as_type.as.Iterate.impl.NewCursor.type.455 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.type.da3: type = fn_type @T.as_type.as.Iterate.impl.Next, @T.as_type.as.Iterate.impl(%facet_value.88a) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.2fa: %T.as_type.as.Iterate.impl.Next.type.da3 = struct_value () [concrete]
// CHECK:STDOUT:   %Iterate.facet.7da: %Iterate.type = facet_value %MutableRange, (%Iterate.impl_witness.581) [concrete]
// CHECK:STDOUT:   %facet_value.125: %Iterate_where.type.7a4 = facet_value %MutableRange, (%Iterate.impl_witness.581) [concrete]
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
// CHECK:STDOUT:   %custom_witness.290: <witness> = custom_witness (%i32, %Iterator.Op.1c3e4a.1), @CppUnsafeDeref [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.0f34b8.2: type = fn_type @Iterator.Op.5 [concrete]
// CHECK:STDOUT:   %Iterator.Op.1c3e4a.2: %Iterator.Op.type.0f34b8.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.fcc: <witness> = custom_witness (%Iterator.Op.1c3e4a.2), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.909: %SubobjectDestroy.type = facet_value %Iterator.260, (%custom_witness.fcc) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.f7e: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.909) [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.0f34b8.3: type = fn_type @Iterator.Op.6 [concrete]
// CHECK:STDOUT:   %Iterator.Op.1c3e4a.3: %Iterator.Op.type.0f34b8.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.746: <witness> = custom_witness (%Iterator.Op.1c3e4a.3), @Inc [concrete]
// CHECK:STDOUT:   %Equal.type.ca8475.2: type = fn_type @Equal.2 [concrete]
// CHECK:STDOUT:   %Equal.285b15.2: %Equal.type.ca8475.2 = struct_value () [concrete]
// CHECK:STDOUT:   %NotEqual.type.53d9ee.2: type = fn_type @NotEqual.2 [concrete]
// CHECK:STDOUT:   %NotEqual.119263.2: %NotEqual.type.53d9ee.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.941: <witness> = custom_witness (%Equal.285b15.2, %NotEqual.119263.2), @EqWith, @EqWith(%Iterator.260) [concrete]
// CHECK:STDOUT:   %facet_value.390: %CppRange_where.type.d682d6.1 = facet_value %ConstRange, (%custom_witness.b80ef6.1, %DestroyImpl.impl_witness.52c, %Copy.impl_witness.4c3, %custom_witness.23e, %custom_witness.fcc, %DestroyImpl.impl_witness.f7e, %custom_witness.290, %custom_witness.746, %custom_witness.941) [concrete]
// CHECK:STDOUT:   %tuple.type.69f: type = tuple_type (%Iterator.260, %Iterator.260) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.9: type = fn_type @SubobjectDestroy.WithSelf.Op.4 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.9: %SubobjectDestroy.WithSelf.Op.type.6870fd.9 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.9: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.9), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.100: %SubobjectDestroy.type = facet_value %tuple.type.69f, (%custom_witness.b80ef6.9) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.ee4: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.100) [concrete]
// CHECK:STDOUT:   %complete_type.8d5: <witness> = complete_type_witness %tuple.type.69f [concrete]
// CHECK:STDOUT:   %facet_value.537: %facet_type.8a3 = facet_value %tuple.type.69f, (%custom_witness.b80ef6.9, %DestroyImpl.impl_witness.ee4) [concrete]
// CHECK:STDOUT:   %Iterate.impl_witness.6c6: <witness> = impl_witness imports.%Iterate.impl_witness_table.121, @T.as_type.as.Iterate.impl(%facet_value.390) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.type.fce: type = fn_type @T.as_type.as.Iterate.impl.NewCursor, @T.as_type.as.Iterate.impl(%facet_value.390) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.3cb: %T.as_type.as.Iterate.impl.NewCursor.type.fce = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.type.488: type = fn_type @T.as_type.as.Iterate.impl.Next, @T.as_type.as.Iterate.impl(%facet_value.390) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.b76: %T.as_type.as.Iterate.impl.Next.type.488 = struct_value () [concrete]
// CHECK:STDOUT:   %Iterate.facet.b23: %Iterate.type = facet_value %ConstRange, (%Iterate.impl_witness.6c6) [concrete]
// CHECK:STDOUT:   %facet_value.e21: %Iterate_where.type.7a4 = facet_value %ConstRange, (%Iterate.impl_witness.6c6) [concrete]
// CHECK:STDOUT:   %r.param_patt.83e: %pattern_type.0ba = value_param_pattern [concrete]
// CHECK:STDOUT:   %r.patt.c21: %pattern_type.0ba = wrapper_binding_pattern r, %r.param_patt.83e [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.b77: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.7da) [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.b77: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.7da) [concrete]
// CHECK:STDOUT:   %.c7d: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.b77, %Iterate.facet.7da [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.specific_fn.7e1: <specific function> = specific_function %T.as_type.as.Iterate.impl.NewCursor.18b, @T.as_type.as.Iterate.impl.NewCursor(%facet_value.88a) [concrete]
// CHECK:STDOUT:   %.e7e: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.b77, %Iterate.facet.7da [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.specific_fn.935: <specific function> = specific_function %T.as_type.as.Iterate.impl.Next.2fa, @T.as_type.as.Iterate.impl.Next(%facet_value.88a) [concrete]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.80f: <specific function> = specific_function %SelfDestruct, @SelfDestruct(%facet_value.ddb) [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.33c: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.b23) [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.037: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.b23) [concrete]
// CHECK:STDOUT:   %.a75: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.33c, %Iterate.facet.b23 [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.specific_fn.704: <specific function> = specific_function %T.as_type.as.Iterate.impl.NewCursor.3cb, @T.as_type.as.Iterate.impl.NewCursor(%facet_value.390) [concrete]
// CHECK:STDOUT:   %.e3f: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.037, %Iterate.facet.b23 [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.specific_fn.392: <specific function> = specific_function %T.as_type.as.Iterate.impl.Next.b76, @T.as_type.as.Iterate.impl.Next(%facet_value.390) [concrete]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.df4: <specific function> = specific_function %SelfDestruct, @SelfDestruct(%facet_value.537) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.4d4: @Optional.%Optional.HasValue.type (%Optional.HasValue.type.cef) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.HasValue (constants.%Optional.HasValue.a74)]
// CHECK:STDOUT:   %Core.import_ref.059: @Optional.%Optional.Get.type (%Optional.Get.type.d97) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.Get (constants.%Optional.Get.3e9)]
// CHECK:STDOUT:   %Core.import_ref.7bb: type = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%DefaultOptionalStorage (constants.%DefaultOptionalStorage.a6b)]
// CHECK:STDOUT:   %Core.import_ref.61e: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.None.type (%T.as_type.as.OptionalStorage.impl.None.type.a9f) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.None (constants.%T.as_type.as.OptionalStorage.impl.None.083)]
// CHECK:STDOUT:   %Core.import_ref.ca1: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Some.type (%T.as_type.as.OptionalStorage.impl.Some.type.61d) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Some (constants.%T.as_type.as.OptionalStorage.impl.Some.ba7)]
// CHECK:STDOUT:   %Core.import_ref.6ba4: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Has.type (%T.as_type.as.OptionalStorage.impl.Has.type.8e5) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Has (constants.%T.as_type.as.OptionalStorage.impl.Has.2d5)]
// CHECK:STDOUT:   %Core.import_ref.73d: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Get.type (%T.as_type.as.OptionalStorage.impl.Get.type.b32) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Get (constants.%T.as_type.as.OptionalStorage.impl.Get.8b0)]
// CHECK:STDOUT:   %Core.import_ref.778 = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], unloaded
// CHECK:STDOUT:   %OptionalStorage.impl_witness_table.968 = impl_witness_table (%Core.import_ref.7bb, %Core.import_ref.61e, %Core.import_ref.ca1, %Core.import_ref.6ba4, %Core.import_ref.73d, %Core.import_ref.778), @T.as_type.as.OptionalStorage.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.146b: @DefaultOptionalStorage.%DefaultOptionalStorage.elem.1 (%DefaultOptionalStorage.elem.c3f) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [concrete = %field_decl.31e]
// CHECK:STDOUT:   %Core.import_ref.f1da: @DefaultOptionalStorage.%DefaultOptionalStorage.elem.2 (%DefaultOptionalStorage.elem.94e) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [concrete = %field_decl.431]
// CHECK:STDOUT:   %Core.import_ref.14b: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.273) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.5b5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.14b), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.d6c: %facet_type.d61 = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.Iterate.impl.%facet_value.1 (constants.%facet_value.9da)]
// CHECK:STDOUT:   %Core.import_ref.2ac: %facet_type.8a3 = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.Iterate.impl.%facet_value.2 (constants.%facet_value.bf6)]
// CHECK:STDOUT:   %Core.import_ref.fd1: @T.as_type.as.Iterate.impl.%T.as_type.as.Iterate.impl.NewCursor.type (%T.as_type.as.Iterate.impl.NewCursor.type.953) = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.Iterate.impl.%T.as_type.as.Iterate.impl.NewCursor (constants.%T.as_type.as.Iterate.impl.NewCursor.544)]
// CHECK:STDOUT:   %Core.import_ref.273: @T.as_type.as.Iterate.impl.%T.as_type.as.Iterate.impl.Next.type (%T.as_type.as.Iterate.impl.Next.type.b36) = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.Iterate.impl.%T.as_type.as.Iterate.impl.Next (constants.%T.as_type.as.Iterate.impl.Next.d6b)]
// CHECK:STDOUT:   %Iterate.impl_witness_table.121 = impl_witness_table (%Core.import_ref.d6c, %Core.import_ref.2ac, %Core.import_ref.fd1, %Core.import_ref.273), @T.as_type.as.Iterate.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.7f2: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.b5e) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.f38)]
// CHECK:STDOUT:   %Copy.impl_witness_table.c8d = impl_witness_table (%Core.import_ref.7f2), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.606: @From.as_type.as.ImplicitAs.impl.%From.as_type.as.ImplicitAs.impl.Convert.type (%From.as_type.as.ImplicitAs.impl.Convert.type.58f) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @From.as_type.as.ImplicitAs.impl.%From.as_type.as.ImplicitAs.impl.Convert (constants.%From.as_type.as.ImplicitAs.impl.Convert.aad)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.85f = impl_witness_table (%Core.import_ref.606), @From.as_type.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.5f18: @Int.as.AddAssignWith.impl.%Int.as.AddAssignWith.impl.Op.type.2 (%Int.as.AddAssignWith.impl.Op.type.e95403.1) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.AddAssignWith.impl.%Int.as.AddAssignWith.impl.Op.2 (constants.%Int.as.AddAssignWith.impl.Op.78e00b.1)]
// CHECK:STDOUT:   %AddAssignWith.impl_witness_table = impl_witness_table (%Core.import_ref.5f18), @Int.as.AddAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.8ea: @Int.as.AddAssignWith.impl.%Int.as.AddAssignWith.impl.Op.type.1 (%Int.as.AddAssignWith.impl.Op.type.e95403.2) = import_ref Core//prelude/types/int, Op, loaded [symbolic = @Int.as.AddAssignWith.impl.%Int.as.AddAssignWith.impl.Op.1 (constants.%Int.as.AddAssignWith.impl.Op.78e00b.2)]
// CHECK:STDOUT:   %Core.import_ref.47b: Core.IntLiteral = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.AnyInt.impl.%N (constants.%N.fe9)]
// CHECK:STDOUT:   %Core.import_ref.1e0: @Int.as.AnyInt.impl.%Int.as.AnyInt.impl.AsInt.type (%Int.as.AnyInt.impl.AsInt.type.425) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.AnyInt.impl.%Int.as.AnyInt.impl.AsInt (constants.%Int.as.AnyInt.impl.AsInt.830)]
// CHECK:STDOUT:   %AnyInt.impl_witness_table = impl_witness_table (%Core.import_ref.47b, %Core.import_ref.1e0), @Int.as.AnyInt.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.b0d = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %field_decl.431: @DefaultOptionalStorage.%DefaultOptionalStorage.elem.2 (%DefaultOptionalStorage.elem.94e) = field_decl has_value, element1, %Core.import_ref.b0d in [concrete] {}
// CHECK:STDOUT:   %Core.import_ref.164c = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %field_decl.31e: @DefaultOptionalStorage.%DefaultOptionalStorage.elem.1 (%DefaultOptionalStorage.elem.c3f) = field_decl value, element0, %Core.import_ref.164c in [concrete] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @TestIterate(%R.loc30_17.2: %Iterate_where.type.7a4) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Iterate.lookup_impl_witness: <witness> = lookup_impl_witness %R.loc30_17.1, @Iterate [symbolic = %Iterate.lookup_impl_witness (constants.%Iterate.lookup_impl_witness.a61)]
// CHECK:STDOUT:   %Iterate.facet.loc34_19.5: %Iterate.type = facet_value %R.as_type.loc30_61.1, (%Iterate.lookup_impl_witness) [symbolic = %Iterate.facet.loc34_19.5 (constants.%Iterate.facet.da6)]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.loc34_19.5) [symbolic = %Iterate.WithSelf.NewCursor.type (constants.%Iterate.WithSelf.NewCursor.type.bf7)]
// CHECK:STDOUT:   %.loc34_19.23: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type, %Iterate.facet.loc34_19.5 [symbolic = %.loc34_19.23 (constants.%.fb4)]
// CHECK:STDOUT:   %impl.elem2.loc34_19.2: @TestIterate.%.loc34_19.23 (%.fb4) = impl_witness_access %Iterate.lookup_impl_witness, element2 [symbolic = %impl.elem2.loc34_19.2 (constants.%impl.elem2.069)]
// CHECK:STDOUT:   %specific_impl_fn.loc34_19.3: <specific function> = specific_impl_function %impl.elem2.loc34_19.2, @Iterate.WithSelf.NewCursor(%Iterate.facet.loc34_19.5) [symbolic = %specific_impl_fn.loc34_19.3 (constants.%specific_impl_fn.a50)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.loc34_19.5) [symbolic = %Iterate.WithSelf.Next.type (constants.%Iterate.WithSelf.Next.type.f9b)]
// CHECK:STDOUT:   %.loc34_19.24: type = fn_type_with_self_type %Iterate.WithSelf.Next.type, %Iterate.facet.loc34_19.5 [symbolic = %.loc34_19.24 (constants.%.f41e)]
// CHECK:STDOUT:   %impl.elem3.loc34_19.2: @TestIterate.%.loc34_19.24 (%.f41e) = impl_witness_access %Iterate.lookup_impl_witness, element3 [symbolic = %impl.elem3.loc34_19.2 (constants.%impl.elem3.735)]
// CHECK:STDOUT:   %specific_impl_fn.loc34_19.4: <specific function> = specific_impl_function %impl.elem3.loc34_19.2, @Iterate.WithSelf.Next(%Iterate.facet.loc34_19.5) [symbolic = %specific_impl_fn.loc34_19.4 (constants.%specific_impl_fn.5aa)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%r.param: @TestIterate.%R.as_type.loc30_61.1 (%R.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %i.patt: %pattern_type.df9 = value_binding_pattern i [concrete = constants.%i.patt]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %r.ref: @TestIterate.%R.as_type.loc30_61.1 (%R.as_type) = name_ref r, %r
// CHECK:STDOUT:     %impl.elem2.loc34_19.1: @TestIterate.%.loc34_19.23 (%.fb4) = impl_witness_access constants.%Iterate.lookup_impl_witness.a61, element2 [symbolic = %impl.elem2.loc34_19.2 (constants.%impl.elem2.069)]
// CHECK:STDOUT:     %bound_method.loc34_19.1: <bound method> = bound_method %r.ref, %impl.elem2.loc34_19.1
// CHECK:STDOUT:     %Iterate.facet.loc34_19.1: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.a61) [symbolic = %Iterate.facet.loc34_19.5 (constants.%Iterate.facet.da6)]
// CHECK:STDOUT:     %.loc34_19.1: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc34_19.1 [symbolic = %Iterate.facet.loc34_19.5 (constants.%Iterate.facet.da6)]
// CHECK:STDOUT:     %Iterate.facet.loc34_19.2: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.a61) [symbolic = %Iterate.facet.loc34_19.5 (constants.%Iterate.facet.da6)]
// CHECK:STDOUT:     %.loc34_19.2: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc34_19.2 [symbolic = %Iterate.facet.loc34_19.5 (constants.%Iterate.facet.da6)]
// CHECK:STDOUT:     %specific_impl_fn.loc34_19.1: <specific function> = specific_impl_function %impl.elem2.loc34_19.1, @Iterate.WithSelf.NewCursor(constants.%Iterate.facet.da6) [symbolic = %specific_impl_fn.loc34_19.3 (constants.%specific_impl_fn.a50)]
// CHECK:STDOUT:     %bound_method.loc34_19.2: <bound method> = bound_method %r.ref, %specific_impl_fn.loc34_19.1
// CHECK:STDOUT:     %var: ref @TestIterate.%as_type (%as_type.d8a) = var_storage invalid
// CHECK:STDOUT:     %Iterate.WithSelf.NewCursor.call: init @TestIterate.%as_type (%as_type.d8a) to %var = call %bound_method.loc34_19.2(%r.ref)
// CHECK:STDOUT:     assign %var, %Iterate.WithSelf.NewCursor.call
// CHECK:STDOUT:     br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.next:
// CHECK:STDOUT:     %impl.elem3.loc34_19.1: @TestIterate.%.loc34_19.24 (%.f41e) = impl_witness_access constants.%Iterate.lookup_impl_witness.a61, element3 [symbolic = %impl.elem3.loc34_19.2 (constants.%impl.elem3.735)]
// CHECK:STDOUT:     %bound_method.loc34_19.3: <bound method> = bound_method %r.ref, %impl.elem3.loc34_19.1
// CHECK:STDOUT:     %Iterate.facet.loc34_19.3: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.a61) [symbolic = %Iterate.facet.loc34_19.5 (constants.%Iterate.facet.da6)]
// CHECK:STDOUT:     %.loc34_19.3: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc34_19.3 [symbolic = %Iterate.facet.loc34_19.5 (constants.%Iterate.facet.da6)]
// CHECK:STDOUT:     %Iterate.facet.loc34_19.4: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.a61) [symbolic = %Iterate.facet.loc34_19.5 (constants.%Iterate.facet.da6)]
// CHECK:STDOUT:     %.loc34_19.4: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc34_19.4 [symbolic = %Iterate.facet.loc34_19.5 (constants.%Iterate.facet.da6)]
// CHECK:STDOUT:     %.loc34_19.5: %facet_type.8a3 = converted constants.%as_type.d8a, constants.%impl.elem1.f53 [symbolic = %impl.elem1 (constants.%impl.elem1.f53)]
// CHECK:STDOUT:     %.loc34_19.6: %facet_type.8a3 = converted constants.%as_type.d8a, constants.%impl.elem1.f53 [symbolic = %impl.elem1 (constants.%impl.elem1.f53)]
// CHECK:STDOUT:     %specific_impl_fn.loc34_19.2: <specific function> = specific_impl_function %impl.elem3.loc34_19.1, @Iterate.WithSelf.Next(constants.%Iterate.facet.da6) [symbolic = %specific_impl_fn.loc34_19.4 (constants.%specific_impl_fn.5aa)]
// CHECK:STDOUT:     %bound_method.loc34_19.4: <bound method> = bound_method %r.ref, %specific_impl_fn.loc34_19.2
// CHECK:STDOUT:     %.loc34_19.7: ref %Optional.656 = temporary_storage
// CHECK:STDOUT:     %Iterate.WithSelf.Next.call: init %Optional.656 to %.loc34_19.7 = call %bound_method.loc34_19.4(%r.ref, %var)
// CHECK:STDOUT:     %.loc34_19.8: ref %Optional.656 = temporary %.loc34_19.7, %Iterate.WithSelf.Next.call
// CHECK:STDOUT:     %.loc34_19.9: %Optional.HasValue.type.201 = specific_constant imports.%Core.import_ref.4d4, @Optional(constants.%OptionalStorage.facet.7f6) [concrete = constants.%Optional.HasValue.392]
// CHECK:STDOUT:     %HasValue.ref: %Optional.HasValue.type.201 = name_ref HasValue, %.loc34_19.9 [concrete = constants.%Optional.HasValue.392]
// CHECK:STDOUT:     %Optional.HasValue.bound: <bound method> = bound_method %.loc34_19.8, %HasValue.ref
// CHECK:STDOUT:     %Optional.HasValue.specific_fn: <specific function> = specific_function %HasValue.ref, @Optional.HasValue(constants.%OptionalStorage.facet.7f6) [concrete = constants.%Optional.HasValue.specific_fn.3a9]
// CHECK:STDOUT:     %bound_method.loc34_19.5: <bound method> = bound_method %.loc34_19.8, %Optional.HasValue.specific_fn
// CHECK:STDOUT:     %.loc34_19.10: %Optional.656 = acquire_value %.loc34_19.8
// CHECK:STDOUT:     %Optional.HasValue.call: init bool = call %bound_method.loc34_19.5(%.loc34_19.10)
// CHECK:STDOUT:     %.loc34_19.11: bool = value_of_initializer %Optional.HasValue.call
// CHECK:STDOUT:     %.loc34_19.12: bool = converted %Optional.HasValue.call, %.loc34_19.11
// CHECK:STDOUT:     if %.loc34_19.12 br !for.body else br !for.done
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.body:
// CHECK:STDOUT:     %.loc34_19.13: %Optional.Get.type.059 = specific_constant imports.%Core.import_ref.059, @Optional(constants.%OptionalStorage.facet.7f6) [concrete = constants.%Optional.Get.3cd]
// CHECK:STDOUT:     %Get.ref: %Optional.Get.type.059 = name_ref Get, %.loc34_19.13 [concrete = constants.%Optional.Get.3cd]
// CHECK:STDOUT:     %Optional.Get.bound: <bound method> = bound_method %.loc34_19.8, %Get.ref
// CHECK:STDOUT:     %Optional.Get.specific_fn: <specific function> = specific_function %Get.ref, @Optional.Get(constants.%OptionalStorage.facet.7f6) [concrete = constants.%Optional.Get.specific_fn.8ba]
// CHECK:STDOUT:     %bound_method.loc34_19.6: <bound method> = bound_method %.loc34_19.8, %Optional.Get.specific_fn
// CHECK:STDOUT:     %.loc34_19.14: %Optional.656 = acquire_value %.loc34_19.8
// CHECK:STDOUT:     %Optional.Get.call: init %i32 = call %bound_method.loc34_19.6(%.loc34_19.14)
// CHECK:STDOUT:     %.loc34_19.15: %i32 = value_of_initializer %Optional.Get.call
// CHECK:STDOUT:     %.loc34_19.16: %i32 = converted %Optional.Get.call, %.loc34_19.15
// CHECK:STDOUT:     %i32.loc34: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %i: %i32 = wrapper_binding i, %.loc34_19.16
// CHECK:STDOUT:     %sum.ref: ref %i32 = name_ref sum, %sum
// CHECK:STDOUT:     %i.ref: %i32 = name_ref i, %i
// CHECK:STDOUT:     %impl.elem0.loc35: %.2a7 = impl_witness_access constants.%AddAssignWith.impl_witness.f97, element0 [concrete = constants.%Int.as.AddAssignWith.impl.Op.e1a1fc.2]
// CHECK:STDOUT:     %bound_method.loc35_9.1: <bound method> = bound_method %sum.ref, %impl.elem0.loc35
// CHECK:STDOUT:     %specific_fn.loc35: <specific function> = specific_function %impl.elem0.loc35, @Int.as.AddAssignWith.impl.Op.1(constants.%int_32, constants.%ImplicitAs.facet.749) [concrete = constants.%Int.as.AddAssignWith.impl.Op.specific_fn.9595aa.1]
// CHECK:STDOUT:     %bound_method.loc35_9.2: <bound method> = bound_method %sum.ref, %specific_fn.loc35
// CHECK:STDOUT:     %.loc35: %Int.as.AddAssignWith.impl.Op.type.ae3c0a.1 = specific_constant imports.%Core.Op.8ea, @Int.as.AddAssignWith.impl(constants.%int_32, constants.%ImplicitAs.facet.749) [concrete = constants.%Int.as.AddAssignWith.impl.Op.e1a1fc.1]
// CHECK:STDOUT:     %Op.ref: %Int.as.AddAssignWith.impl.Op.type.ae3c0a.1 = name_ref Op, %.loc35 [concrete = constants.%Int.as.AddAssignWith.impl.Op.e1a1fc.1]
// CHECK:STDOUT:     %Int.as.AddAssignWith.impl.Op.bound: <bound method> = bound_method %sum.ref, %Op.ref
// CHECK:STDOUT:     %Int.as.AddAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref, @Int.as.AddAssignWith.impl.Op.2(constants.%int_32, constants.%ImplicitAs.facet.749) [concrete = constants.%Int.as.AddAssignWith.impl.Op.specific_fn.9595aa.2]
// CHECK:STDOUT:     %bound_method.loc35_9.3: <bound method> = bound_method %sum.ref, %Int.as.AddAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:     %Int.as.AddAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc35_9.3(%sum.ref, %i.ref)
// CHECK:STDOUT:     %facet_value.loc34_19.1: %facet_type.8a3 = facet_value constants.%Optional.656, (constants.%custom_witness.b80ef6.7, constants.%DestroyImpl.impl_witness.06e) [concrete = constants.%facet_value.d9c]
// CHECK:STDOUT:     %.loc34_19.17: %facet_type.8a3 = converted constants.%Optional.656, %facet_value.loc34_19.1 [concrete = constants.%facet_value.d9c]
// CHECK:STDOUT:     %facet_value.loc34_19.2: %facet_type.8a3 = facet_value constants.%Optional.656, (constants.%custom_witness.b80ef6.7, constants.%DestroyImpl.impl_witness.06e) [concrete = constants.%facet_value.d9c]
// CHECK:STDOUT:     %.loc34_19.18: %facet_type.8a3 = converted constants.%Optional.656, %facet_value.loc34_19.2 [concrete = constants.%facet_value.d9c]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %SelfDestruct.call.loc34_19.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc34_19.8)
// CHECK:STDOUT:     br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.done:
// CHECK:STDOUT:     %facet_value.loc34_19.3: %facet_type.8a3 = facet_value constants.%Optional.656, (constants.%custom_witness.b80ef6.7, constants.%DestroyImpl.impl_witness.06e) [concrete = constants.%facet_value.d9c]
// CHECK:STDOUT:     %.loc34_19.19: %facet_type.8a3 = converted constants.%Optional.656, %facet_value.loc34_19.3 [concrete = constants.%facet_value.d9c]
// CHECK:STDOUT:     %facet_value.loc34_19.4: %facet_type.8a3 = facet_value constants.%Optional.656, (constants.%custom_witness.b80ef6.7, constants.%DestroyImpl.impl_witness.06e) [concrete = constants.%facet_value.d9c]
// CHECK:STDOUT:     %.loc34_19.20: %facet_type.8a3 = converted constants.%Optional.656, %facet_value.loc34_19.4 [concrete = constants.%facet_value.d9c]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %SelfDestruct.call.loc34_19.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc34_19.8)
// CHECK:STDOUT:     %.loc34_19.21: %facet_type.8a3 = converted constants.%as_type.d8a, constants.%impl.elem1.f53 [symbolic = %impl.elem1 (constants.%impl.elem1.f53)]
// CHECK:STDOUT:     %.loc34_19.22: %facet_type.8a3 = converted constants.%as_type.d8a, constants.%impl.elem1.f53 [symbolic = %impl.elem1 (constants.%impl.elem1.f53)]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %SelfDestruct.call.loc34_19.3: init %empty_tuple.type = call %SelfDestruct.specific_fn.3(%var)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc34_19.1(%self.param: ref %.1ff) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc34_19.2(%self.param: ref %MaybeUnformed.3cf) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc34_19.3(%self.param: ref bool) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc34_19.4(%self.param: ref %struct_type.value.has_value.00c) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc34_19.1: ref bool = struct_access %self.param, element1
// CHECK:STDOUT:   %facet_value.loc34_19.1: %facet_type.8a3 = facet_value bool, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.deb) [concrete = constants.%facet_value.a1b]
// CHECK:STDOUT:   %.loc34_19.2: %facet_type.8a3 = converted bool, %facet_value.loc34_19.1 [concrete = constants.%facet_value.a1b]
// CHECK:STDOUT:   %facet_value.loc34_19.2: %facet_type.8a3 = facet_value bool, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.deb) [concrete = constants.%facet_value.a1b]
// CHECK:STDOUT:   %.loc34_19.3: %facet_type.8a3 = converted bool, %facet_value.loc34_19.2 [concrete = constants.%facet_value.a1b]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc34_19.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc34_19.1)
// CHECK:STDOUT:   %.loc34_19.4: ref %MaybeUnformed.3cf = struct_access %self.param, element0
// CHECK:STDOUT:   %facet_value.loc34_19.3: %facet_type.8a3 = facet_value constants.%MaybeUnformed.3cf, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.a00) [concrete = constants.%facet_value.c63]
// CHECK:STDOUT:   %.loc34_19.5: %facet_type.8a3 = converted constants.%MaybeUnformed.3cf, %facet_value.loc34_19.3 [concrete = constants.%facet_value.c63]
// CHECK:STDOUT:   %facet_value.loc34_19.4: %facet_type.8a3 = facet_value constants.%MaybeUnformed.3cf, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.a00) [concrete = constants.%facet_value.c63]
// CHECK:STDOUT:   %.loc34_19.6: %facet_type.8a3 = converted constants.%MaybeUnformed.3cf, %facet_value.loc34_19.4 [concrete = constants.%facet_value.c63]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc34_19.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc34_19.4)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc34_19.5(%self.param: ref %DefaultOptionalStorage.84b) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %has_value.ref: %DefaultOptionalStorage.elem.4e2 = name_ref has_value, imports.%Core.import_ref.f1da [concrete = imports.%field_decl.431]
// CHECK:STDOUT:   %.loc34_19.1: ref bool = class_element_access %self.param, element1
// CHECK:STDOUT:   %facet_value.loc34_19.1: %facet_type.8a3 = facet_value bool, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.deb) [concrete = constants.%facet_value.a1b]
// CHECK:STDOUT:   %.loc34_19.2: %facet_type.8a3 = converted bool, %facet_value.loc34_19.1 [concrete = constants.%facet_value.a1b]
// CHECK:STDOUT:   %facet_value.loc34_19.2: %facet_type.8a3 = facet_value bool, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.deb) [concrete = constants.%facet_value.a1b]
// CHECK:STDOUT:   %.loc34_19.3: %facet_type.8a3 = converted bool, %facet_value.loc34_19.2 [concrete = constants.%facet_value.a1b]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc34_19.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc34_19.1)
// CHECK:STDOUT:   %value.ref: %DefaultOptionalStorage.elem.8fe = name_ref value, imports.%Core.import_ref.146b [concrete = imports.%field_decl.31e]
// CHECK:STDOUT:   %.loc34_19.4: ref %MaybeUnformed.3cf = class_element_access %self.param, element0
// CHECK:STDOUT:   %facet_value.loc34_19.3: %facet_type.8a3 = facet_value constants.%MaybeUnformed.3cf, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.a00) [concrete = constants.%facet_value.c63]
// CHECK:STDOUT:   %.loc34_19.5: %facet_type.8a3 = converted constants.%MaybeUnformed.3cf, %facet_value.loc34_19.3 [concrete = constants.%facet_value.c63]
// CHECK:STDOUT:   %facet_value.loc34_19.4: %facet_type.8a3 = facet_value constants.%MaybeUnformed.3cf, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.a00) [concrete = constants.%facet_value.c63]
// CHECK:STDOUT:   %.loc34_19.6: %facet_type.8a3 = converted constants.%MaybeUnformed.3cf, %facet_value.loc34_19.4 [concrete = constants.%facet_value.c63]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc34_19.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc34_19.4)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc34_19.6(%self.param: ref %Optional.656) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestRangeFor(%m.param: ref %MutableRange, %c.param: %ConstRange) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %i.patt.loc49: %pattern_type.df9 = value_binding_pattern i [concrete = constants.%i.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %m.ref: ref %MutableRange = name_ref m, %m
// CHECK:STDOUT:   %impl.elem2.loc49: %.c7d = impl_witness_access constants.%Iterate.impl_witness.581, element2 [concrete = constants.%T.as_type.as.Iterate.impl.NewCursor.18b]
// CHECK:STDOUT:   %bound_method.loc49_19.1: <bound method> = bound_method %m.ref, %impl.elem2.loc49
// CHECK:STDOUT:   %facet_value.loc49_19.1: %CppRange_where.type.d682d6.1 = facet_value constants.%MutableRange, (constants.%custom_witness.b80ef6.1, constants.%DestroyImpl.impl_witness.52c, constants.%Copy.impl_witness.4c3, constants.%custom_witness.ed3, constants.%custom_witness.da7, constants.%DestroyImpl.impl_witness.048, constants.%custom_witness.52e, constants.%custom_witness.444, constants.%custom_witness.981) [concrete = constants.%facet_value.88a]
// CHECK:STDOUT:   %.loc49_19.1: %CppRange_where.type.d682d6.1 = converted constants.%MutableRange, %facet_value.loc49_19.1 [concrete = constants.%facet_value.88a]
// CHECK:STDOUT:   %facet_value.loc49_19.2: %CppRange_where.type.d682d6.1 = facet_value constants.%MutableRange, (constants.%custom_witness.b80ef6.1, constants.%DestroyImpl.impl_witness.52c, constants.%Copy.impl_witness.4c3, constants.%custom_witness.ed3, constants.%custom_witness.da7, constants.%DestroyImpl.impl_witness.048, constants.%custom_witness.52e, constants.%custom_witness.444, constants.%custom_witness.981) [concrete = constants.%facet_value.88a]
// CHECK:STDOUT:   %.loc49_19.2: %CppRange_where.type.d682d6.1 = converted constants.%MutableRange, %facet_value.loc49_19.2 [concrete = constants.%facet_value.88a]
// CHECK:STDOUT:   %specific_fn.loc49_19.1: <specific function> = specific_function %impl.elem2.loc49, @T.as_type.as.Iterate.impl.NewCursor(constants.%facet_value.88a) [concrete = constants.%T.as_type.as.Iterate.impl.NewCursor.specific_fn.7e1]
// CHECK:STDOUT:   %bound_method.loc49_19.2: <bound method> = bound_method %m.ref, %specific_fn.loc49_19.1
// CHECK:STDOUT:   %var.loc49: ref %tuple.type.508 = var_storage invalid
// CHECK:STDOUT:   %.loc49_18.1: %MutableRange = acquire_value %m.ref
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.call.loc49: init %tuple.type.508 to %var.loc49 = call %bound_method.loc49_19.2(%.loc49_18.1)
// CHECK:STDOUT:   assign %var.loc49, %T.as_type.as.Iterate.impl.NewCursor.call.loc49
// CHECK:STDOUT:   br !for.next.loc49
// CHECK:STDOUT:
// CHECK:STDOUT: !for.next.loc49:
// CHECK:STDOUT:   %impl.elem3.loc49: %.e7e = impl_witness_access constants.%Iterate.impl_witness.581, element3 [concrete = constants.%T.as_type.as.Iterate.impl.Next.2fa]
// CHECK:STDOUT:   %bound_method.loc49_19.3: <bound method> = bound_method %m.ref, %impl.elem3.loc49
// CHECK:STDOUT:   %facet_value.loc49_19.3: %CppRange_where.type.d682d6.1 = facet_value constants.%MutableRange, (constants.%custom_witness.b80ef6.1, constants.%DestroyImpl.impl_witness.52c, constants.%Copy.impl_witness.4c3, constants.%custom_witness.ed3, constants.%custom_witness.da7, constants.%DestroyImpl.impl_witness.048, constants.%custom_witness.52e, constants.%custom_witness.444, constants.%custom_witness.981) [concrete = constants.%facet_value.88a]
// CHECK:STDOUT:   %.loc49_19.3: %CppRange_where.type.d682d6.1 = converted constants.%MutableRange, %facet_value.loc49_19.3 [concrete = constants.%facet_value.88a]
// CHECK:STDOUT:   %facet_value.loc49_19.4: %CppRange_where.type.d682d6.1 = facet_value constants.%MutableRange, (constants.%custom_witness.b80ef6.1, constants.%DestroyImpl.impl_witness.52c, constants.%Copy.impl_witness.4c3, constants.%custom_witness.ed3, constants.%custom_witness.da7, constants.%DestroyImpl.impl_witness.048, constants.%custom_witness.52e, constants.%custom_witness.444, constants.%custom_witness.981) [concrete = constants.%facet_value.88a]
// CHECK:STDOUT:   %.loc49_19.4: %CppRange_where.type.d682d6.1 = converted constants.%MutableRange, %facet_value.loc49_19.4 [concrete = constants.%facet_value.88a]
// CHECK:STDOUT:   %specific_fn.loc49_19.2: <specific function> = specific_function %impl.elem3.loc49, @T.as_type.as.Iterate.impl.Next(constants.%facet_value.88a) [concrete = constants.%T.as_type.as.Iterate.impl.Next.specific_fn.935]
// CHECK:STDOUT:   %bound_method.loc49_19.4: <bound method> = bound_method %m.ref, %specific_fn.loc49_19.2
// CHECK:STDOUT:   %.loc49_19.5: ref %Optional.656 = temporary_storage
// CHECK:STDOUT:   %.loc49_18.2: %MutableRange = acquire_value %m.ref
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.call.loc49: init %Optional.656 to %.loc49_19.5 = call %bound_method.loc49_19.4(%.loc49_18.2, %var.loc49)
// CHECK:STDOUT:   %.loc49_19.6: ref %Optional.656 = temporary %.loc49_19.5, %T.as_type.as.Iterate.impl.Next.call.loc49
// CHECK:STDOUT:   %.loc49_19.7: %Optional.HasValue.type.201 = specific_constant imports.%Core.import_ref.4d4, @Optional(constants.%OptionalStorage.facet.7f6) [concrete = constants.%Optional.HasValue.392]
// CHECK:STDOUT:   %HasValue.ref.loc49: %Optional.HasValue.type.201 = name_ref HasValue, %.loc49_19.7 [concrete = constants.%Optional.HasValue.392]
// CHECK:STDOUT:   %Optional.HasValue.bound.loc49: <bound method> = bound_method %.loc49_19.6, %HasValue.ref.loc49
// CHECK:STDOUT:   %Optional.HasValue.specific_fn.loc49: <specific function> = specific_function %HasValue.ref.loc49, @Optional.HasValue(constants.%OptionalStorage.facet.7f6) [concrete = constants.%Optional.HasValue.specific_fn.3a9]
// CHECK:STDOUT:   %bound_method.loc49_19.5: <bound method> = bound_method %.loc49_19.6, %Optional.HasValue.specific_fn.loc49
// CHECK:STDOUT:   %.loc49_19.8: %Optional.656 = acquire_value %.loc49_19.6
// CHECK:STDOUT:   %Optional.HasValue.call.loc49: init bool = call %bound_method.loc49_19.5(%.loc49_19.8)
// CHECK:STDOUT:   %.loc49_19.9: bool = value_of_initializer %Optional.HasValue.call.loc49
// CHECK:STDOUT:   %.loc49_19.10: bool = converted %Optional.HasValue.call.loc49, %.loc49_19.9
// CHECK:STDOUT:   if %.loc49_19.10 br !for.body.loc49 else br !for.done.loc49
// CHECK:STDOUT:
// CHECK:STDOUT: !for.body.loc49:
// CHECK:STDOUT:   %.loc49_19.11: %Optional.Get.type.059 = specific_constant imports.%Core.import_ref.059, @Optional(constants.%OptionalStorage.facet.7f6) [concrete = constants.%Optional.Get.3cd]
// CHECK:STDOUT:   %Get.ref.loc49: %Optional.Get.type.059 = name_ref Get, %.loc49_19.11 [concrete = constants.%Optional.Get.3cd]
// CHECK:STDOUT:   %Optional.Get.bound.loc49: <bound method> = bound_method %.loc49_19.6, %Get.ref.loc49
// CHECK:STDOUT:   %Optional.Get.specific_fn.loc49: <specific function> = specific_function %Get.ref.loc49, @Optional.Get(constants.%OptionalStorage.facet.7f6) [concrete = constants.%Optional.Get.specific_fn.8ba]
// CHECK:STDOUT:   %bound_method.loc49_19.6: <bound method> = bound_method %.loc49_19.6, %Optional.Get.specific_fn.loc49
// CHECK:STDOUT:   %.loc49_19.12: %Optional.656 = acquire_value %.loc49_19.6
// CHECK:STDOUT:   %Optional.Get.call.loc49: init %i32 = call %bound_method.loc49_19.6(%.loc49_19.12)
// CHECK:STDOUT:   %.loc49_19.13: %i32 = value_of_initializer %Optional.Get.call.loc49
// CHECK:STDOUT:   %.loc49_19.14: %i32 = converted %Optional.Get.call.loc49, %.loc49_19.13
// CHECK:STDOUT:   %i32.loc49: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %i.loc49: %i32 = wrapper_binding i, %.loc49_19.14
// CHECK:STDOUT:   %sum.ref.loc50: ref %i32 = name_ref sum, %sum
// CHECK:STDOUT:   %i.ref.loc50: %i32 = name_ref i, %i.loc49
// CHECK:STDOUT:   %impl.elem0.loc50: %.2a7 = impl_witness_access constants.%AddAssignWith.impl_witness.f97, element0 [concrete = constants.%Int.as.AddAssignWith.impl.Op.e1a1fc.2]
// CHECK:STDOUT:   %bound_method.loc50_9.1: <bound method> = bound_method %sum.ref.loc50, %impl.elem0.loc50
// CHECK:STDOUT:   %specific_fn.loc50: <specific function> = specific_function %impl.elem0.loc50, @Int.as.AddAssignWith.impl.Op.1(constants.%int_32, constants.%ImplicitAs.facet.749) [concrete = constants.%Int.as.AddAssignWith.impl.Op.specific_fn.9595aa.1]
// CHECK:STDOUT:   %bound_method.loc50_9.2: <bound method> = bound_method %sum.ref.loc50, %specific_fn.loc50
// CHECK:STDOUT:   %.loc50: %Int.as.AddAssignWith.impl.Op.type.ae3c0a.1 = specific_constant imports.%Core.Op.8ea, @Int.as.AddAssignWith.impl(constants.%int_32, constants.%ImplicitAs.facet.749) [concrete = constants.%Int.as.AddAssignWith.impl.Op.e1a1fc.1]
// CHECK:STDOUT:   %Op.ref.loc50: %Int.as.AddAssignWith.impl.Op.type.ae3c0a.1 = name_ref Op, %.loc50 [concrete = constants.%Int.as.AddAssignWith.impl.Op.e1a1fc.1]
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.bound.loc50: <bound method> = bound_method %sum.ref.loc50, %Op.ref.loc50
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.specific_fn.loc50: <specific function> = specific_function %Op.ref.loc50, @Int.as.AddAssignWith.impl.Op.2(constants.%int_32, constants.%ImplicitAs.facet.749) [concrete = constants.%Int.as.AddAssignWith.impl.Op.specific_fn.9595aa.2]
// CHECK:STDOUT:   %bound_method.loc50_9.3: <bound method> = bound_method %sum.ref.loc50, %Int.as.AddAssignWith.impl.Op.specific_fn.loc50
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.call.loc50: init %empty_tuple.type = call %bound_method.loc50_9.3(%sum.ref.loc50, %i.ref.loc50)
// CHECK:STDOUT:   %facet_value.loc49_19.5: %facet_type.8a3 = facet_value constants.%Optional.656, (constants.%custom_witness.b80ef6.7, constants.%DestroyImpl.impl_witness.06e) [concrete = constants.%facet_value.d9c]
// CHECK:STDOUT:   %.loc49_19.15: %facet_type.8a3 = converted constants.%Optional.656, %facet_value.loc49_19.5 [concrete = constants.%facet_value.d9c]
// CHECK:STDOUT:   %facet_value.loc49_19.6: %facet_type.8a3 = facet_value constants.%Optional.656, (constants.%custom_witness.b80ef6.7, constants.%DestroyImpl.impl_witness.06e) [concrete = constants.%facet_value.d9c]
// CHECK:STDOUT:   %.loc49_19.16: %facet_type.8a3 = converted constants.%Optional.656, %facet_value.loc49_19.6 [concrete = constants.%facet_value.d9c]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc49_19.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc49_19.6)
// CHECK:STDOUT:   br !for.next.loc49
// CHECK:STDOUT:
// CHECK:STDOUT: !for.done.loc49:
// CHECK:STDOUT:   %facet_value.loc49_19.7: %facet_type.8a3 = facet_value constants.%Optional.656, (constants.%custom_witness.b80ef6.7, constants.%DestroyImpl.impl_witness.06e) [concrete = constants.%facet_value.d9c]
// CHECK:STDOUT:   %.loc49_19.17: %facet_type.8a3 = converted constants.%Optional.656, %facet_value.loc49_19.7 [concrete = constants.%facet_value.d9c]
// CHECK:STDOUT:   %facet_value.loc49_19.8: %facet_type.8a3 = facet_value constants.%Optional.656, (constants.%custom_witness.b80ef6.7, constants.%DestroyImpl.impl_witness.06e) [concrete = constants.%facet_value.d9c]
// CHECK:STDOUT:   %.loc49_19.18: %facet_type.8a3 = converted constants.%Optional.656, %facet_value.loc49_19.8 [concrete = constants.%facet_value.d9c]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc49_19.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc49_19.6)
// CHECK:STDOUT:   %facet_value.loc49_19.9: %facet_type.8a3 = facet_value constants.%tuple.type.508, (constants.%custom_witness.b80ef6.8, constants.%DestroyImpl.impl_witness.b78) [concrete = constants.%facet_value.ddb]
// CHECK:STDOUT:   %.loc49_19.19: %facet_type.8a3 = converted constants.%tuple.type.508, %facet_value.loc49_19.9 [concrete = constants.%facet_value.ddb]
// CHECK:STDOUT:   %facet_value.loc49_19.10: %facet_type.8a3 = facet_value constants.%tuple.type.508, (constants.%custom_witness.b80ef6.8, constants.%DestroyImpl.impl_witness.b78) [concrete = constants.%facet_value.ddb]
// CHECK:STDOUT:   %.loc49_19.20: %facet_type.8a3 = converted constants.%tuple.type.508, %facet_value.loc49_19.10 [concrete = constants.%facet_value.ddb]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc49_19.3: init %empty_tuple.type = call %SelfDestruct.specific_fn.3(%var.loc49)
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %i.patt.loc55: %pattern_type.df9 = value_binding_pattern i [concrete = constants.%i.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c.ref: %ConstRange = name_ref c, %c
// CHECK:STDOUT:   %impl.elem2.loc55: %.a75 = impl_witness_access constants.%Iterate.impl_witness.6c6, element2 [concrete = constants.%T.as_type.as.Iterate.impl.NewCursor.3cb]
// CHECK:STDOUT:   %bound_method.loc55_19.1: <bound method> = bound_method %c.ref, %impl.elem2.loc55
// CHECK:STDOUT:   %facet_value.loc55_19.1: %CppRange_where.type.d682d6.1 = facet_value constants.%ConstRange, (constants.%custom_witness.b80ef6.1, constants.%DestroyImpl.impl_witness.52c, constants.%Copy.impl_witness.4c3, constants.%custom_witness.23e, constants.%custom_witness.fcc, constants.%DestroyImpl.impl_witness.f7e, constants.%custom_witness.290, constants.%custom_witness.746, constants.%custom_witness.941) [concrete = constants.%facet_value.390]
// CHECK:STDOUT:   %.loc55_19.1: %CppRange_where.type.d682d6.1 = converted constants.%ConstRange, %facet_value.loc55_19.1 [concrete = constants.%facet_value.390]
// CHECK:STDOUT:   %facet_value.loc55_19.2: %CppRange_where.type.d682d6.1 = facet_value constants.%ConstRange, (constants.%custom_witness.b80ef6.1, constants.%DestroyImpl.impl_witness.52c, constants.%Copy.impl_witness.4c3, constants.%custom_witness.23e, constants.%custom_witness.fcc, constants.%DestroyImpl.impl_witness.f7e, constants.%custom_witness.290, constants.%custom_witness.746, constants.%custom_witness.941) [concrete = constants.%facet_value.390]
// CHECK:STDOUT:   %.loc55_19.2: %CppRange_where.type.d682d6.1 = converted constants.%ConstRange, %facet_value.loc55_19.2 [concrete = constants.%facet_value.390]
// CHECK:STDOUT:   %specific_fn.loc55_19.1: <specific function> = specific_function %impl.elem2.loc55, @T.as_type.as.Iterate.impl.NewCursor(constants.%facet_value.390) [concrete = constants.%T.as_type.as.Iterate.impl.NewCursor.specific_fn.704]
// CHECK:STDOUT:   %bound_method.loc55_19.2: <bound method> = bound_method %c.ref, %specific_fn.loc55_19.1
// CHECK:STDOUT:   %var.loc55: ref %tuple.type.69f = var_storage invalid
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.call.loc55: init %tuple.type.69f to %var.loc55 = call %bound_method.loc55_19.2(%c.ref)
// CHECK:STDOUT:   assign %var.loc55, %T.as_type.as.Iterate.impl.NewCursor.call.loc55
// CHECK:STDOUT:   br !for.next.loc55
// CHECK:STDOUT:
// CHECK:STDOUT: !for.next.loc55:
// CHECK:STDOUT:   %impl.elem3.loc55: %.e3f = impl_witness_access constants.%Iterate.impl_witness.6c6, element3 [concrete = constants.%T.as_type.as.Iterate.impl.Next.b76]
// CHECK:STDOUT:   %bound_method.loc55_19.3: <bound method> = bound_method %c.ref, %impl.elem3.loc55
// CHECK:STDOUT:   %facet_value.loc55_19.3: %CppRange_where.type.d682d6.1 = facet_value constants.%ConstRange, (constants.%custom_witness.b80ef6.1, constants.%DestroyImpl.impl_witness.52c, constants.%Copy.impl_witness.4c3, constants.%custom_witness.23e, constants.%custom_witness.fcc, constants.%DestroyImpl.impl_witness.f7e, constants.%custom_witness.290, constants.%custom_witness.746, constants.%custom_witness.941) [concrete = constants.%facet_value.390]
// CHECK:STDOUT:   %.loc55_19.3: %CppRange_where.type.d682d6.1 = converted constants.%ConstRange, %facet_value.loc55_19.3 [concrete = constants.%facet_value.390]
// CHECK:STDOUT:   %facet_value.loc55_19.4: %CppRange_where.type.d682d6.1 = facet_value constants.%ConstRange, (constants.%custom_witness.b80ef6.1, constants.%DestroyImpl.impl_witness.52c, constants.%Copy.impl_witness.4c3, constants.%custom_witness.23e, constants.%custom_witness.fcc, constants.%DestroyImpl.impl_witness.f7e, constants.%custom_witness.290, constants.%custom_witness.746, constants.%custom_witness.941) [concrete = constants.%facet_value.390]
// CHECK:STDOUT:   %.loc55_19.4: %CppRange_where.type.d682d6.1 = converted constants.%ConstRange, %facet_value.loc55_19.4 [concrete = constants.%facet_value.390]
// CHECK:STDOUT:   %specific_fn.loc55_19.2: <specific function> = specific_function %impl.elem3.loc55, @T.as_type.as.Iterate.impl.Next(constants.%facet_value.390) [concrete = constants.%T.as_type.as.Iterate.impl.Next.specific_fn.392]
// CHECK:STDOUT:   %bound_method.loc55_19.4: <bound method> = bound_method %c.ref, %specific_fn.loc55_19.2
// CHECK:STDOUT:   %.loc55_19.5: ref %Optional.656 = temporary_storage
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.call.loc55: init %Optional.656 to %.loc55_19.5 = call %bound_method.loc55_19.4(%c.ref, %var.loc55)
// CHECK:STDOUT:   %.loc55_19.6: ref %Optional.656 = temporary %.loc55_19.5, %T.as_type.as.Iterate.impl.Next.call.loc55
// CHECK:STDOUT:   %.loc55_19.7: %Optional.HasValue.type.201 = specific_constant imports.%Core.import_ref.4d4, @Optional(constants.%OptionalStorage.facet.7f6) [concrete = constants.%Optional.HasValue.392]
// CHECK:STDOUT:   %HasValue.ref.loc55: %Optional.HasValue.type.201 = name_ref HasValue, %.loc55_19.7 [concrete = constants.%Optional.HasValue.392]
// CHECK:STDOUT:   %Optional.HasValue.bound.loc55: <bound method> = bound_method %.loc55_19.6, %HasValue.ref.loc55
// CHECK:STDOUT:   %Optional.HasValue.specific_fn.loc55: <specific function> = specific_function %HasValue.ref.loc55, @Optional.HasValue(constants.%OptionalStorage.facet.7f6) [concrete = constants.%Optional.HasValue.specific_fn.3a9]
// CHECK:STDOUT:   %bound_method.loc55_19.5: <bound method> = bound_method %.loc55_19.6, %Optional.HasValue.specific_fn.loc55
// CHECK:STDOUT:   %.loc55_19.8: %Optional.656 = acquire_value %.loc55_19.6
// CHECK:STDOUT:   %Optional.HasValue.call.loc55: init bool = call %bound_method.loc55_19.5(%.loc55_19.8)
// CHECK:STDOUT:   %.loc55_19.9: bool = value_of_initializer %Optional.HasValue.call.loc55
// CHECK:STDOUT:   %.loc55_19.10: bool = converted %Optional.HasValue.call.loc55, %.loc55_19.9
// CHECK:STDOUT:   if %.loc55_19.10 br !for.body.loc55 else br !for.done.loc55
// CHECK:STDOUT:
// CHECK:STDOUT: !for.body.loc55:
// CHECK:STDOUT:   %.loc55_19.11: %Optional.Get.type.059 = specific_constant imports.%Core.import_ref.059, @Optional(constants.%OptionalStorage.facet.7f6) [concrete = constants.%Optional.Get.3cd]
// CHECK:STDOUT:   %Get.ref.loc55: %Optional.Get.type.059 = name_ref Get, %.loc55_19.11 [concrete = constants.%Optional.Get.3cd]
// CHECK:STDOUT:   %Optional.Get.bound.loc55: <bound method> = bound_method %.loc55_19.6, %Get.ref.loc55
// CHECK:STDOUT:   %Optional.Get.specific_fn.loc55: <specific function> = specific_function %Get.ref.loc55, @Optional.Get(constants.%OptionalStorage.facet.7f6) [concrete = constants.%Optional.Get.specific_fn.8ba]
// CHECK:STDOUT:   %bound_method.loc55_19.6: <bound method> = bound_method %.loc55_19.6, %Optional.Get.specific_fn.loc55
// CHECK:STDOUT:   %.loc55_19.12: %Optional.656 = acquire_value %.loc55_19.6
// CHECK:STDOUT:   %Optional.Get.call.loc55: init %i32 = call %bound_method.loc55_19.6(%.loc55_19.12)
// CHECK:STDOUT:   %.loc55_19.13: %i32 = value_of_initializer %Optional.Get.call.loc55
// CHECK:STDOUT:   %.loc55_19.14: %i32 = converted %Optional.Get.call.loc55, %.loc55_19.13
// CHECK:STDOUT:   %i32.loc55: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %i.loc55: %i32 = wrapper_binding i, %.loc55_19.14
// CHECK:STDOUT:   %sum.ref.loc56: ref %i32 = name_ref sum, %sum
// CHECK:STDOUT:   %i.ref.loc56: %i32 = name_ref i, %i.loc55
// CHECK:STDOUT:   %impl.elem0.loc56: %.2a7 = impl_witness_access constants.%AddAssignWith.impl_witness.f97, element0 [concrete = constants.%Int.as.AddAssignWith.impl.Op.e1a1fc.2]
// CHECK:STDOUT:   %bound_method.loc56_9.1: <bound method> = bound_method %sum.ref.loc56, %impl.elem0.loc56
// CHECK:STDOUT:   %specific_fn.loc56: <specific function> = specific_function %impl.elem0.loc56, @Int.as.AddAssignWith.impl.Op.1(constants.%int_32, constants.%ImplicitAs.facet.749) [concrete = constants.%Int.as.AddAssignWith.impl.Op.specific_fn.9595aa.1]
// CHECK:STDOUT:   %bound_method.loc56_9.2: <bound method> = bound_method %sum.ref.loc56, %specific_fn.loc56
// CHECK:STDOUT:   %.loc56: %Int.as.AddAssignWith.impl.Op.type.ae3c0a.1 = specific_constant imports.%Core.Op.8ea, @Int.as.AddAssignWith.impl(constants.%int_32, constants.%ImplicitAs.facet.749) [concrete = constants.%Int.as.AddAssignWith.impl.Op.e1a1fc.1]
// CHECK:STDOUT:   %Op.ref.loc56: %Int.as.AddAssignWith.impl.Op.type.ae3c0a.1 = name_ref Op, %.loc56 [concrete = constants.%Int.as.AddAssignWith.impl.Op.e1a1fc.1]
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.bound.loc56: <bound method> = bound_method %sum.ref.loc56, %Op.ref.loc56
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.specific_fn.loc56: <specific function> = specific_function %Op.ref.loc56, @Int.as.AddAssignWith.impl.Op.2(constants.%int_32, constants.%ImplicitAs.facet.749) [concrete = constants.%Int.as.AddAssignWith.impl.Op.specific_fn.9595aa.2]
// CHECK:STDOUT:   %bound_method.loc56_9.3: <bound method> = bound_method %sum.ref.loc56, %Int.as.AddAssignWith.impl.Op.specific_fn.loc56
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.call.loc56: init %empty_tuple.type = call %bound_method.loc56_9.3(%sum.ref.loc56, %i.ref.loc56)
// CHECK:STDOUT:   %facet_value.loc55_19.5: %facet_type.8a3 = facet_value constants.%Optional.656, (constants.%custom_witness.b80ef6.7, constants.%DestroyImpl.impl_witness.06e) [concrete = constants.%facet_value.d9c]
// CHECK:STDOUT:   %.loc55_19.15: %facet_type.8a3 = converted constants.%Optional.656, %facet_value.loc55_19.5 [concrete = constants.%facet_value.d9c]
// CHECK:STDOUT:   %facet_value.loc55_19.6: %facet_type.8a3 = facet_value constants.%Optional.656, (constants.%custom_witness.b80ef6.7, constants.%DestroyImpl.impl_witness.06e) [concrete = constants.%facet_value.d9c]
// CHECK:STDOUT:   %.loc55_19.16: %facet_type.8a3 = converted constants.%Optional.656, %facet_value.loc55_19.6 [concrete = constants.%facet_value.d9c]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc55_19.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.4(%.loc55_19.6)
// CHECK:STDOUT:   br !for.next.loc55
// CHECK:STDOUT:
// CHECK:STDOUT: !for.done.loc55:
// CHECK:STDOUT:   %facet_value.loc55_19.7: %facet_type.8a3 = facet_value constants.%Optional.656, (constants.%custom_witness.b80ef6.7, constants.%DestroyImpl.impl_witness.06e) [concrete = constants.%facet_value.d9c]
// CHECK:STDOUT:   %.loc55_19.17: %facet_type.8a3 = converted constants.%Optional.656, %facet_value.loc55_19.7 [concrete = constants.%facet_value.d9c]
// CHECK:STDOUT:   %facet_value.loc55_19.8: %facet_type.8a3 = facet_value constants.%Optional.656, (constants.%custom_witness.b80ef6.7, constants.%DestroyImpl.impl_witness.06e) [concrete = constants.%facet_value.d9c]
// CHECK:STDOUT:   %.loc55_19.18: %facet_type.8a3 = converted constants.%Optional.656, %facet_value.loc55_19.8 [concrete = constants.%facet_value.d9c]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc55_19.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.5(%.loc55_19.6)
// CHECK:STDOUT:   %facet_value.loc55_19.9: %facet_type.8a3 = facet_value constants.%tuple.type.69f, (constants.%custom_witness.b80ef6.9, constants.%DestroyImpl.impl_witness.ee4) [concrete = constants.%facet_value.537]
// CHECK:STDOUT:   %.loc55_19.19: %facet_type.8a3 = converted constants.%tuple.type.69f, %facet_value.loc55_19.9 [concrete = constants.%facet_value.537]
// CHECK:STDOUT:   %facet_value.loc55_19.10: %facet_type.8a3 = facet_value constants.%tuple.type.69f, (constants.%custom_witness.b80ef6.9, constants.%DestroyImpl.impl_witness.ee4) [concrete = constants.%facet_value.537]
// CHECK:STDOUT:   %.loc55_19.20: %facet_type.8a3 = converted constants.%tuple.type.69f, %facet_value.loc55_19.10 [concrete = constants.%facet_value.537]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc55_19.3: init %empty_tuple.type = call %SelfDestruct.specific_fn.6(%var.loc55)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestIterate(constants.%R) {
// CHECK:STDOUT:   %R.patt.loc30_17.2 => constants.%R.patt
// CHECK:STDOUT:   %R.loc30_17.1 => constants.%R
// CHECK:STDOUT:   %R.as_type.loc30_61.1 => constants.%R.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.249
// CHECK:STDOUT:   %r.param_patt.loc30_59.2 => constants.%r.param_patt.8a5
// CHECK:STDOUT:   %r.patt.loc30_59.2 => constants.%r.patt.f98
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestIterate(constants.%facet_value.125) {
// CHECK:STDOUT:   %R.patt.loc30_17.2 => constants.%R.patt
// CHECK:STDOUT:   %R.loc30_17.1 => constants.%facet_value.125
// CHECK:STDOUT:   %R.as_type.loc30_61.1 => constants.%MutableRange
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.e635
// CHECK:STDOUT:   %r.param_patt.loc30_59.2 => constants.%r.param_patt.c79
// CHECK:STDOUT:   %r.patt.loc30_59.2 => constants.%r.patt.94f
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc30 => constants.%complete_type.357
// CHECK:STDOUT:   %Iterate.lookup_impl_witness => constants.%Iterate.impl_witness.581
// CHECK:STDOUT:   %Iterate.facet.loc34_19.5 => constants.%Iterate.facet.7da
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type => constants.%Iterate.WithSelf.NewCursor.type.b77
// CHECK:STDOUT:   %.loc34_19.23 => constants.%.c7d
// CHECK:STDOUT:   %impl.elem2.loc34_19.2 => constants.%T.as_type.as.Iterate.impl.NewCursor.18b
// CHECK:STDOUT:   %specific_impl_fn.loc34_19.3 => constants.%T.as_type.as.Iterate.impl.NewCursor.specific_fn.7e1
// CHECK:STDOUT:   %impl.elem1 => constants.%facet_value.ddb
// CHECK:STDOUT:   %as_type => constants.%tuple.type.508
// CHECK:STDOUT:   %require_complete.1 => constants.%complete_type.c28
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type => constants.%Iterate.WithSelf.Next.type.b77
// CHECK:STDOUT:   %.loc34_19.24 => constants.%.e7e
// CHECK:STDOUT:   %impl.elem3.loc34_19.2 => constants.%T.as_type.as.Iterate.impl.Next.2fa
// CHECK:STDOUT:   %specific_impl_fn.loc34_19.4 => constants.%T.as_type.as.Iterate.impl.Next.specific_fn.935
// CHECK:STDOUT:   %SelfDestruct.specific_fn.5 => constants.%SelfDestruct.specific_fn.80f
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestIterate(constants.%facet_value.e21) {
// CHECK:STDOUT:   %R.patt.loc30_17.2 => constants.%R.patt
// CHECK:STDOUT:   %R.loc30_17.1 => constants.%facet_value.e21
// CHECK:STDOUT:   %R.as_type.loc30_61.1 => constants.%ConstRange
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.0ba
// CHECK:STDOUT:   %r.param_patt.loc30_59.2 => constants.%r.param_patt.83e
// CHECK:STDOUT:   %r.patt.loc30_59.2 => constants.%r.patt.c21
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc30 => constants.%complete_type.357
// CHECK:STDOUT:   %Iterate.lookup_impl_witness => constants.%Iterate.impl_witness.6c6
// CHECK:STDOUT:   %Iterate.facet.loc34_19.5 => constants.%Iterate.facet.b23
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type => constants.%Iterate.WithSelf.NewCursor.type.33c
// CHECK:STDOUT:   %.loc34_19.23 => constants.%.a75
// CHECK:STDOUT:   %impl.elem2.loc34_19.2 => constants.%T.as_type.as.Iterate.impl.NewCursor.3cb
// CHECK:STDOUT:   %specific_impl_fn.loc34_19.3 => constants.%T.as_type.as.Iterate.impl.NewCursor.specific_fn.704
// CHECK:STDOUT:   %impl.elem1 => constants.%facet_value.537
// CHECK:STDOUT:   %as_type => constants.%tuple.type.69f
// CHECK:STDOUT:   %require_complete.1 => constants.%complete_type.8d5
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type => constants.%Iterate.WithSelf.Next.type.037
// CHECK:STDOUT:   %.loc34_19.24 => constants.%.e3f
// CHECK:STDOUT:   %impl.elem3.loc34_19.2 => constants.%T.as_type.as.Iterate.impl.Next.b76
// CHECK:STDOUT:   %specific_impl_fn.loc34_19.4 => constants.%T.as_type.as.Iterate.impl.Next.specific_fn.392
// CHECK:STDOUT:   %SelfDestruct.specific_fn.5 => constants.%SelfDestruct.specific_fn.df4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- methods_return_different_types.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %Iterate.type: type = facet_type <@Iterate> [concrete]
// CHECK:STDOUT:   %OptionalStorage.type: type = facet_type <@OptionalStorage> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %facet_type.d61: type = facet_type <@SubobjectDestroy & @DestroyImpl & @Copy> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %T.220: %OptionalStorage.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Optional.Get.type.d97: type = fn_type @Optional.Get, @Optional(%T.220) [symbolic]
// CHECK:STDOUT:   %Optional.Get.3e9: %Optional.Get.type.d97 = struct_value () [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.type.cef: type = fn_type @Optional.HasValue, @Optional(%T.220) [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.a74: %Optional.HasValue.type.cef = struct_value () [symbolic]
// CHECK:STDOUT:   %facet_type.8a3: type = facet_type <@SubobjectDestroy & @DestroyImpl> [concrete]
// CHECK:STDOUT:   %T.9ab: %facet_type.d61 = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.19c: type = facet_access_type %T.9ab [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.lookup_impl_witness.855: <witness> = lookup_impl_witness %T.9ab, @SubobjectDestroy [symbolic]
// CHECK:STDOUT:   %DestroyImpl.lookup_impl_witness.ff9: <witness> = lookup_impl_witness %T.9ab, @DestroyImpl [symbolic]
// CHECK:STDOUT:   %facet_value.b35: %facet_type.8a3 = facet_value %T.as_type.19c, (%SubobjectDestroy.lookup_impl_witness.855, %DestroyImpl.lookup_impl_witness.ff9) [symbolic]
// CHECK:STDOUT:   %MaybeUnformed.413: type = class_type @MaybeUnformed, @MaybeUnformed(%facet_value.b35) [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.a6b: type = class_type @DefaultOptionalStorage, @DefaultOptionalStorage(%T.9ab) [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.elem.94e: type = unbound_element_type %DefaultOptionalStorage.a6b, bool [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.elem.c3f: type = unbound_element_type %DefaultOptionalStorage.a6b, %MaybeUnformed.413 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Get.type.b32: type = fn_type @T.as_type.as.OptionalStorage.impl.Get, @T.as_type.as.OptionalStorage.impl(%T.9ab) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Get.8b0: %T.as_type.as.OptionalStorage.impl.Get.type.b32 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Has.type.8e5: type = fn_type @T.as_type.as.OptionalStorage.impl.Has, @T.as_type.as.OptionalStorage.impl(%T.9ab) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Has.2d5: %T.as_type.as.OptionalStorage.impl.Has.type.8e5 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Some.type.61d: type = fn_type @T.as_type.as.OptionalStorage.impl.Some, @T.as_type.as.OptionalStorage.impl(%T.9ab) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Some.ba7: %T.as_type.as.OptionalStorage.impl.Some.type.61d = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.None.type.a9f: type = fn_type @T.as_type.as.OptionalStorage.impl.None, @T.as_type.as.OptionalStorage.impl(%T.9ab) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.None.083: %T.as_type.as.OptionalStorage.impl.None.type.a9f = struct_value () [symbolic]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.273: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.5b5: %T.as_type.as.DestroyImpl.impl.Op.type.273 = struct_value () [symbolic]
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
// CHECK:STDOUT:   %SubobjectDestroy.lookup_impl_witness.dc9: <witness> = lookup_impl_witness %tuple.type.1ba, @SubobjectDestroy [symbolic]
// CHECK:STDOUT:   %DestroyImpl.lookup_impl_witness.504: <witness> = lookup_impl_witness %tuple.type.1ba, @DestroyImpl [symbolic]
// CHECK:STDOUT:   %facet_value.bf6: %facet_type.8a3 = facet_value %tuple.type.1ba, (%SubobjectDestroy.lookup_impl_witness.dc9, %DestroyImpl.lookup_impl_witness.504) [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.lookup_impl_witness.b15da7.2: <witness> = lookup_impl_witness %impl.elem0.493765.2, @SubobjectDestroy [symbolic]
// CHECK:STDOUT:   %DestroyImpl.lookup_impl_witness.4e7f44.2: <witness> = lookup_impl_witness %impl.elem0.493765.2, @DestroyImpl [symbolic]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.167: <witness> = lookup_impl_witness %impl.elem0.493765.2, @Copy [symbolic]
// CHECK:STDOUT:   %facet_value.9da: %facet_type.d61 = facet_value %impl.elem0.493765.2, (%SubobjectDestroy.lookup_impl_witness.b15da7.2, %DestroyImpl.lookup_impl_witness.4e7f44.2, %Copy.lookup_impl_witness.167) [symbolic]
// CHECK:STDOUT:   %SelfDestruct.type: type = fn_type @SelfDestruct [concrete]
// CHECK:STDOUT:   %SelfDestruct: %SelfDestruct.type = struct_value () [concrete]
// CHECK:STDOUT:   %DiffValueType: type = class_type @DiffValueType [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %pattern_type.149: type = pattern_type %DiffValueType [concrete]
// CHECK:STDOUT:   %DiffValueType.Op.type.83af6e.1: type = fn_type @DiffValueType.Op.1 [concrete]
// CHECK:STDOUT:   %DiffValueType.Op.4dca31.1: %DiffValueType.Op.type.83af6e.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.5760f4.1: <witness> = custom_witness (%DiffValueType.Op.4dca31.1), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.7ea: %SubobjectDestroy.type = facet_value %DiffValueType, (%custom_witness.5760f4.1) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.39a: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.7ea) [concrete]
// CHECK:STDOUT:   %const.879: type = const_type %DiffValueType [concrete]
// CHECK:STDOUT:   %ptr.816: type = ptr_type %const.879 [concrete]
// CHECK:STDOUT:   %ptr.cd0: type = ptr_type %DiffValueType [concrete]
// CHECK:STDOUT:   %DiffValueType.Op.type.83af6e.2: type = fn_type @DiffValueType.Op.2 [concrete]
// CHECK:STDOUT:   %DiffValueType.Op.4dca31.2: %DiffValueType.Op.type.83af6e.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.d0da43.1: <witness> = custom_witness (%DiffValueType.Op.4dca31.2), @Copy [concrete]
// CHECK:STDOUT:   %facet_value.e45: %facet_type.d61 = facet_value %DiffValueType, (%custom_witness.5760f4.1, %DestroyImpl.impl_witness.39a, %custom_witness.d0da43.1) [concrete]
// CHECK:STDOUT:   %Iterate.lookup_impl_witness.1d8: <witness> = lookup_impl_witness %.Self.500, @Iterate [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.8a4: %facet_type.d61 = impl_witness_access %Iterate.lookup_impl_witness.1d8, element0 [symbolic_self]
// CHECK:STDOUT:   %Iterate_where.type.401: type = facet_type <@Iterate where %impl.elem0.8a4 = %facet_value.e45> [concrete]
// CHECK:STDOUT:   %pattern_type.abe: type = pattern_type %Iterate_where.type.401 [concrete]
// CHECK:STDOUT:   %R.patt: %pattern_type.abe = symbolic_binding_pattern R, 0 [symbolic]
// CHECK:STDOUT:   %R: %Iterate_where.type.401 = symbolic_binding R, 0 [symbolic]
// CHECK:STDOUT:   %R.as_type: type = facet_access_type %R [symbolic]
// CHECK:STDOUT:   %pattern_type.bac: type = pattern_type %R.as_type [symbolic]
// CHECK:STDOUT:   %r.patt.cb9: %pattern_type.bac = ref_binding_pattern r [symbolic]
// CHECK:STDOUT:   %r.param_patt.4be: %pattern_type.bac = var_param_pattern %r.patt.cb9 [symbolic]
// CHECK:STDOUT:   %i.patt: %pattern_type.149 = value_binding_pattern i [concrete]
// CHECK:STDOUT:   %Iterate.lookup_impl_witness.47b: <witness> = lookup_impl_witness %R, @Iterate [symbolic]
// CHECK:STDOUT:   %Iterate.facet.db9: %Iterate.type = facet_value %R.as_type, (%Iterate.lookup_impl_witness.47b) [symbolic]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.6ed: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.db9) [symbolic]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.752: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.db9) [symbolic]
// CHECK:STDOUT:   %.3e0: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.6ed, %Iterate.facet.db9 [symbolic]
// CHECK:STDOUT:   %impl.elem2.d6e: %.3e0 = impl_witness_access %Iterate.lookup_impl_witness.47b, element2 [symbolic]
// CHECK:STDOUT:   %impl.elem1.efb: %facet_type.8a3 = impl_witness_access %Iterate.lookup_impl_witness.47b, element1 [symbolic]
// CHECK:STDOUT:   %as_type.91d: type = facet_access_type %impl.elem1.efb [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.9bc: <specific function> = specific_impl_function %impl.elem2.d6e, @Iterate.WithSelf.NewCursor(%Iterate.facet.db9) [symbolic]
// CHECK:STDOUT:   %.38c: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.752, %Iterate.facet.db9 [symbolic]
// CHECK:STDOUT:   %impl.elem3.b40: %.38c = impl_witness_access %Iterate.lookup_impl_witness.47b, element3 [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.670: type = class_type @DefaultOptionalStorage, @DefaultOptionalStorage(%facet_value.e45) [concrete]
// CHECK:STDOUT:   %OptionalStorage.impl_witness.a9c: <witness> = impl_witness imports.%OptionalStorage.impl_witness_table.968, @T.as_type.as.OptionalStorage.impl(%facet_value.e45) [concrete]
// CHECK:STDOUT:   %OptionalStorage.facet.7d7: %OptionalStorage.type = facet_value %DiffValueType, (%OptionalStorage.impl_witness.a9c) [concrete]
// CHECK:STDOUT:   %Optional.91f: type = class_type @Optional, @Optional(%OptionalStorage.facet.7d7) [concrete]
// CHECK:STDOUT:   %specific_impl_fn.fcf: <specific function> = specific_impl_function %impl.elem3.b40, @Iterate.WithSelf.Next(%Iterate.facet.db9) [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.type.a6d: type = fn_type @Optional.HasValue, @Optional(%OptionalStorage.facet.7d7) [concrete]
// CHECK:STDOUT:   %Optional.HasValue.1f0: %Optional.HasValue.type.a6d = struct_value () [concrete]
// CHECK:STDOUT:   %Optional.Get.type.d3e: type = fn_type @Optional.Get, @Optional(%OptionalStorage.facet.7d7) [concrete]
// CHECK:STDOUT:   %Optional.Get.2d4: %Optional.Get.type.d3e = struct_value () [concrete]
// CHECK:STDOUT:   %facet_value.dc6: %facet_type.8a3 = facet_value %DiffValueType, (%custom_witness.5760f4.1, %DestroyImpl.impl_witness.39a) [concrete]
// CHECK:STDOUT:   %MaybeUnformed.699: type = class_type @MaybeUnformed, @MaybeUnformed(%facet_value.dc6) [concrete]
// CHECK:STDOUT:   %.a18: type = maybe_unformed_type %DiffValueType [concrete]
// CHECK:STDOUT:   %DefaultOptionalStorage.elem.7b1: type = unbound_element_type %DefaultOptionalStorage.670, %MaybeUnformed.699 [concrete]
// CHECK:STDOUT:   %DefaultOptionalStorage.elem.0c5: type = unbound_element_type %DefaultOptionalStorage.670, bool [concrete]
// CHECK:STDOUT:   %struct_type.value.has_value.608: type = struct_type {.value: %MaybeUnformed.699, .has_value: bool} [concrete]
// CHECK:STDOUT:   %Optional.HasValue.specific_fn.0fc: <specific function> = specific_function %Optional.HasValue.1f0, @Optional.HasValue(%OptionalStorage.facet.7d7) [concrete]
// CHECK:STDOUT:   %Optional.Get.specific_fn.c83: <specific function> = specific_function %Optional.Get.2d4, @Optional.Get(%OptionalStorage.facet.7d7) [concrete]
// CHECK:STDOUT:   %operator_PlusEqual__carbon_thunk.type: type = fn_type @operator_PlusEqual__carbon_thunk [concrete]
// CHECK:STDOUT:   %operator_PlusEqual__carbon_thunk: %operator_PlusEqual__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.3: type = fn_type @SubobjectDestroy.WithSelf.Op.loc44_33.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.3: %SubobjectDestroy.WithSelf.Op.type.6870fd.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.3: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.3), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.4: type = fn_type @SubobjectDestroy.WithSelf.Op.loc44_33.3 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.4: %SubobjectDestroy.WithSelf.Op.type.6870fd.4 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.4: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.4), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.a87: %SubobjectDestroy.type = facet_value bool, (%custom_witness.b80ef6.4) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.deb: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.a87) [concrete]
// CHECK:STDOUT:   %facet_value.a1b: %facet_type.8a3 = facet_value bool, (%custom_witness.b80ef6.4, %DestroyImpl.impl_witness.deb) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.be2: %SubobjectDestroy.type = facet_value %MaybeUnformed.699, (%custom_witness.b80ef6.3) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.38a: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.be2) [concrete]
// CHECK:STDOUT:   %facet_value.67f: %facet_type.8a3 = facet_value %MaybeUnformed.699, (%custom_witness.b80ef6.3, %DestroyImpl.impl_witness.38a) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.7: type = fn_type @SubobjectDestroy.WithSelf.Op.loc44_33.6 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.7: %SubobjectDestroy.WithSelf.Op.type.6870fd.7 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.7: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.7), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.beb: %SubobjectDestroy.type = facet_value %Optional.91f, (%custom_witness.b80ef6.7) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.f90: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.beb) [concrete]
// CHECK:STDOUT:   %facet_value.451: %facet_type.8a3 = facet_value %Optional.91f, (%custom_witness.b80ef6.7, %DestroyImpl.impl_witness.f90) [concrete]
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
// CHECK:STDOUT:   %custom_witness.3815f0.1: <witness> = custom_witness (%Iterator.Op.1caa36.2), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.131: %SubobjectDestroy.type = facet_value %Iterator.f25, (%custom_witness.3815f0.1) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.09a: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.131) [concrete]
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
// CHECK:STDOUT:   %custom_witness.3815f0.2: <witness> = custom_witness (%Sentinel.Op.1ca), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.d42: %SubobjectDestroy.type = facet_value %Sentinel.b4c, (%custom_witness.3815f0.2) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.71f: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.d42) [concrete]
// CHECK:STDOUT:   %facet_value.4f2: %CppRange_where.type.d682d6.1 = facet_value %DiffMutableRange, (%custom_witness.5760f4.1, %DestroyImpl.impl_witness.39a, %custom_witness.d0da43.1, %custom_witness.aa8, %custom_witness.3815f0.1, %DestroyImpl.impl_witness.09a, %custom_witness.392, %custom_witness.1c9, %custom_witness.723, %custom_witness.3815f0.2, %DestroyImpl.impl_witness.71f) [concrete]
// CHECK:STDOUT:   %tuple.type.b18: type = tuple_type (%Iterator.f25, %Sentinel.b4c) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.8: type = fn_type @SubobjectDestroy.WithSelf.Op.3 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.8: %SubobjectDestroy.WithSelf.Op.type.6870fd.8 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.8: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.8), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.d2e: %SubobjectDestroy.type = facet_value %tuple.type.b18, (%custom_witness.b80ef6.8) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.a87: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.d2e) [concrete]
// CHECK:STDOUT:   %complete_type.bfe: <witness> = complete_type_witness %tuple.type.b18 [concrete]
// CHECK:STDOUT:   %facet_value.f38: %facet_type.8a3 = facet_value %tuple.type.b18, (%custom_witness.b80ef6.8, %DestroyImpl.impl_witness.a87) [concrete]
// CHECK:STDOUT:   %Iterate.impl_witness.7b0: <witness> = impl_witness imports.%Iterate.impl_witness_table.121, @T.as_type.as.Iterate.impl(%facet_value.4f2) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.type.616: type = fn_type @T.as_type.as.Iterate.impl.NewCursor, @T.as_type.as.Iterate.impl(%facet_value.4f2) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.64a: %T.as_type.as.Iterate.impl.NewCursor.type.616 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.type.948: type = fn_type @T.as_type.as.Iterate.impl.Next, @T.as_type.as.Iterate.impl(%facet_value.4f2) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.ed2: %T.as_type.as.Iterate.impl.Next.type.948 = struct_value () [concrete]
// CHECK:STDOUT:   %Iterate.facet.441: %Iterate.type = facet_value %DiffMutableRange, (%Iterate.impl_witness.7b0) [concrete]
// CHECK:STDOUT:   %facet_value.8c6: %Iterate_where.type.401 = facet_value %DiffMutableRange, (%Iterate.impl_witness.7b0) [concrete]
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
// CHECK:STDOUT:   %custom_witness.3d0529.1: <witness> = custom_witness (%Iterator.Op.690fee.2), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.9b7: %SubobjectDestroy.type = facet_value %Iterator.a59, (%custom_witness.3d0529.1) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.b0f: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.9b7) [concrete]
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
// CHECK:STDOUT:   %custom_witness.3d0529.2: <witness> = custom_witness (%Sentinel.Op.690), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.b86: %SubobjectDestroy.type = facet_value %Sentinel.b33, (%custom_witness.3d0529.2) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.4ee: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.b86) [concrete]
// CHECK:STDOUT:   %facet_value.32d: %CppRange_where.type.d682d6.1 = facet_value %DiffConstRange, (%custom_witness.5760f4.1, %DestroyImpl.impl_witness.39a, %custom_witness.d0da43.1, %custom_witness.7b4, %custom_witness.3d0529.1, %DestroyImpl.impl_witness.b0f, %custom_witness.0c0, %custom_witness.ffb, %custom_witness.a26, %custom_witness.3d0529.2, %DestroyImpl.impl_witness.4ee) [concrete]
// CHECK:STDOUT:   %tuple.type.df5: type = tuple_type (%Iterator.a59, %Sentinel.b33) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.9: type = fn_type @SubobjectDestroy.WithSelf.Op.4 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.9: %SubobjectDestroy.WithSelf.Op.type.6870fd.9 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.9: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.9), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.79d: %SubobjectDestroy.type = facet_value %tuple.type.df5, (%custom_witness.b80ef6.9) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.af6: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.79d) [concrete]
// CHECK:STDOUT:   %complete_type.ad6: <witness> = complete_type_witness %tuple.type.df5 [concrete]
// CHECK:STDOUT:   %facet_value.591: %facet_type.8a3 = facet_value %tuple.type.df5, (%custom_witness.b80ef6.9, %DestroyImpl.impl_witness.af6) [concrete]
// CHECK:STDOUT:   %Iterate.impl_witness.532: <witness> = impl_witness imports.%Iterate.impl_witness_table.121, @T.as_type.as.Iterate.impl(%facet_value.32d) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.type.901: type = fn_type @T.as_type.as.Iterate.impl.NewCursor, @T.as_type.as.Iterate.impl(%facet_value.32d) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.5aa: %T.as_type.as.Iterate.impl.NewCursor.type.901 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.type.361: type = fn_type @T.as_type.as.Iterate.impl.Next, @T.as_type.as.Iterate.impl(%facet_value.32d) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.79c: %T.as_type.as.Iterate.impl.Next.type.361 = struct_value () [concrete]
// CHECK:STDOUT:   %Iterate.facet.001: %Iterate.type = facet_value %DiffConstRange, (%Iterate.impl_witness.532) [concrete]
// CHECK:STDOUT:   %facet_value.4df: %Iterate_where.type.401 = facet_value %DiffConstRange, (%Iterate.impl_witness.532) [concrete]
// CHECK:STDOUT:   %r.patt.9c0: %pattern_type.7dd = ref_binding_pattern r [concrete]
// CHECK:STDOUT:   %r.param_patt.992: %pattern_type.7dd = var_param_pattern %r.patt.9c0 [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.cc8: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.441) [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.0b4: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.441) [concrete]
// CHECK:STDOUT:   %.68d: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.cc8, %Iterate.facet.441 [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.specific_fn.1fd: <specific function> = specific_function %T.as_type.as.Iterate.impl.NewCursor.64a, @T.as_type.as.Iterate.impl.NewCursor(%facet_value.4f2) [concrete]
// CHECK:STDOUT:   %.b83: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.0b4, %Iterate.facet.441 [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.specific_fn.90b: <specific function> = specific_function %T.as_type.as.Iterate.impl.Next.ed2, @T.as_type.as.Iterate.impl.Next(%facet_value.4f2) [concrete]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.b15: <specific function> = specific_function %SelfDestruct, @SelfDestruct(%facet_value.f38) [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.7ef: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.001) [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.f2f: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.001) [concrete]
// CHECK:STDOUT:   %.edf: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.7ef, %Iterate.facet.001 [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.specific_fn.ce3: <specific function> = specific_function %T.as_type.as.Iterate.impl.NewCursor.5aa, @T.as_type.as.Iterate.impl.NewCursor(%facet_value.32d) [concrete]
// CHECK:STDOUT:   %.24c: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.f2f, %Iterate.facet.001 [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.specific_fn.d8c: <specific function> = specific_function %T.as_type.as.Iterate.impl.Next.79c, @T.as_type.as.Iterate.impl.Next(%facet_value.32d) [concrete]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.80e: <specific function> = specific_function %SelfDestruct, @SelfDestruct(%facet_value.591) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .DiffValueType = %DiffValueType.decl
// CHECK:STDOUT:     .DiffMutableRange = %DiffMutableRange.decl
// CHECK:STDOUT:     .DiffConstRange = %DiffConstRange.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.4d4: @Optional.%Optional.HasValue.type (%Optional.HasValue.type.cef) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.HasValue (constants.%Optional.HasValue.a74)]
// CHECK:STDOUT:   %Core.import_ref.059: @Optional.%Optional.Get.type (%Optional.Get.type.d97) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.Get (constants.%Optional.Get.3e9)]
// CHECK:STDOUT:   %Core.import_ref.7bb: type = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%DefaultOptionalStorage (constants.%DefaultOptionalStorage.a6b)]
// CHECK:STDOUT:   %Core.import_ref.61e: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.None.type (%T.as_type.as.OptionalStorage.impl.None.type.a9f) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.None (constants.%T.as_type.as.OptionalStorage.impl.None.083)]
// CHECK:STDOUT:   %Core.import_ref.ca1: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Some.type (%T.as_type.as.OptionalStorage.impl.Some.type.61d) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Some (constants.%T.as_type.as.OptionalStorage.impl.Some.ba7)]
// CHECK:STDOUT:   %Core.import_ref.6ba4: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Has.type (%T.as_type.as.OptionalStorage.impl.Has.type.8e5) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Has (constants.%T.as_type.as.OptionalStorage.impl.Has.2d5)]
// CHECK:STDOUT:   %Core.import_ref.73d: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Get.type (%T.as_type.as.OptionalStorage.impl.Get.type.b32) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Get (constants.%T.as_type.as.OptionalStorage.impl.Get.8b0)]
// CHECK:STDOUT:   %Core.import_ref.778 = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], unloaded
// CHECK:STDOUT:   %OptionalStorage.impl_witness_table.968 = impl_witness_table (%Core.import_ref.7bb, %Core.import_ref.61e, %Core.import_ref.ca1, %Core.import_ref.6ba4, %Core.import_ref.73d, %Core.import_ref.778), @T.as_type.as.OptionalStorage.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.146b: @DefaultOptionalStorage.%DefaultOptionalStorage.elem.1 (%DefaultOptionalStorage.elem.c3f) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [concrete = %field_decl.31e]
// CHECK:STDOUT:   %Core.import_ref.f1da: @DefaultOptionalStorage.%DefaultOptionalStorage.elem.2 (%DefaultOptionalStorage.elem.94e) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [concrete = %field_decl.431]
// CHECK:STDOUT:   %Core.import_ref.14b: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.273) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.5b5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.14b), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.d6c: %facet_type.d61 = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.Iterate.impl.%facet_value.1 (constants.%facet_value.9da)]
// CHECK:STDOUT:   %Core.import_ref.2ac: %facet_type.8a3 = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.Iterate.impl.%facet_value.2 (constants.%facet_value.bf6)]
// CHECK:STDOUT:   %Core.import_ref.fd1: @T.as_type.as.Iterate.impl.%T.as_type.as.Iterate.impl.NewCursor.type (%T.as_type.as.Iterate.impl.NewCursor.type.953) = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.Iterate.impl.%T.as_type.as.Iterate.impl.NewCursor (constants.%T.as_type.as.Iterate.impl.NewCursor.544)]
// CHECK:STDOUT:   %Core.import_ref.273: @T.as_type.as.Iterate.impl.%T.as_type.as.Iterate.impl.Next.type (%T.as_type.as.Iterate.impl.Next.type.b36) = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.Iterate.impl.%T.as_type.as.Iterate.impl.Next (constants.%T.as_type.as.Iterate.impl.Next.d6b)]
// CHECK:STDOUT:   %Iterate.impl_witness_table.121 = impl_witness_table (%Core.import_ref.d6c, %Core.import_ref.2ac, %Core.import_ref.fd1, %Core.import_ref.273), @T.as_type.as.Iterate.impl [concrete]
// CHECK:STDOUT:   %DiffValueType.decl: type = class_decl @DiffValueType [concrete = constants.%DiffValueType] {} {}
// CHECK:STDOUT:   %operator_PlusEqual__carbon_thunk.decl: %operator_PlusEqual__carbon_thunk.type = fn_decl @operator_PlusEqual__carbon_thunk [concrete = constants.%operator_PlusEqual__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.b0d = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %field_decl.431: @DefaultOptionalStorage.%DefaultOptionalStorage.elem.2 (%DefaultOptionalStorage.elem.94e) = field_decl has_value, element1, %Core.import_ref.b0d in [concrete] {}
// CHECK:STDOUT:   %Core.import_ref.164c = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %field_decl.31e: @DefaultOptionalStorage.%DefaultOptionalStorage.elem.1 (%DefaultOptionalStorage.elem.c3f) = field_decl value, element0, %Core.import_ref.164c in [concrete] {}
// CHECK:STDOUT:   %DiffMutableRange.decl: type = class_decl @DiffMutableRange [concrete = constants.%DiffMutableRange] {} {}
// CHECK:STDOUT:   %DiffConstRange.decl: type = class_decl @DiffConstRange [concrete = constants.%DiffConstRange] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @TestIterate(%R.loc40_17.2: %Iterate_where.type.401) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Iterate.lookup_impl_witness: <witness> = lookup_impl_witness %R.loc40_17.1, @Iterate [symbolic = %Iterate.lookup_impl_witness (constants.%Iterate.lookup_impl_witness.47b)]
// CHECK:STDOUT:   %Iterate.facet.loc44_33.5: %Iterate.type = facet_value %R.as_type.loc40_79.1, (%Iterate.lookup_impl_witness) [symbolic = %Iterate.facet.loc44_33.5 (constants.%Iterate.facet.db9)]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.loc44_33.5) [symbolic = %Iterate.WithSelf.NewCursor.type (constants.%Iterate.WithSelf.NewCursor.type.6ed)]
// CHECK:STDOUT:   %.loc44_33.26: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type, %Iterate.facet.loc44_33.5 [symbolic = %.loc44_33.26 (constants.%.3e0)]
// CHECK:STDOUT:   %impl.elem2.loc44_33.2: @TestIterate.%.loc44_33.26 (%.3e0) = impl_witness_access %Iterate.lookup_impl_witness, element2 [symbolic = %impl.elem2.loc44_33.2 (constants.%impl.elem2.d6e)]
// CHECK:STDOUT:   %specific_impl_fn.loc44_33.3: <specific function> = specific_impl_function %impl.elem2.loc44_33.2, @Iterate.WithSelf.NewCursor(%Iterate.facet.loc44_33.5) [symbolic = %specific_impl_fn.loc44_33.3 (constants.%specific_impl_fn.9bc)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.loc44_33.5) [symbolic = %Iterate.WithSelf.Next.type (constants.%Iterate.WithSelf.Next.type.752)]
// CHECK:STDOUT:   %.loc44_33.27: type = fn_type_with_self_type %Iterate.WithSelf.Next.type, %Iterate.facet.loc44_33.5 [symbolic = %.loc44_33.27 (constants.%.38c)]
// CHECK:STDOUT:   %impl.elem3.loc44_33.2: @TestIterate.%.loc44_33.27 (%.38c) = impl_witness_access %Iterate.lookup_impl_witness, element3 [symbolic = %impl.elem3.loc44_33.2 (constants.%impl.elem3.b40)]
// CHECK:STDOUT:   %specific_impl_fn.loc44_33.4: <specific function> = specific_impl_function %impl.elem3.loc44_33.2, @Iterate.WithSelf.Next(%Iterate.facet.loc44_33.5) [symbolic = %specific_impl_fn.loc44_33.4 (constants.%specific_impl_fn.fcf)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%r.param: ref @TestIterate.%R.as_type.loc40_79.1 (%R.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %i.patt: %pattern_type.149 = value_binding_pattern i [concrete = constants.%i.patt]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %r.ref: ref @TestIterate.%R.as_type.loc40_79.1 (%R.as_type) = name_ref r, %r
// CHECK:STDOUT:     %impl.elem2.loc44_33.1: @TestIterate.%.loc44_33.26 (%.3e0) = impl_witness_access constants.%Iterate.lookup_impl_witness.47b, element2 [symbolic = %impl.elem2.loc44_33.2 (constants.%impl.elem2.d6e)]
// CHECK:STDOUT:     %bound_method.loc44_33.1: <bound method> = bound_method %r.ref, %impl.elem2.loc44_33.1
// CHECK:STDOUT:     %Iterate.facet.loc44_33.1: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.47b) [symbolic = %Iterate.facet.loc44_33.5 (constants.%Iterate.facet.db9)]
// CHECK:STDOUT:     %.loc44_33.1: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc44_33.1 [symbolic = %Iterate.facet.loc44_33.5 (constants.%Iterate.facet.db9)]
// CHECK:STDOUT:     %Iterate.facet.loc44_33.2: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.47b) [symbolic = %Iterate.facet.loc44_33.5 (constants.%Iterate.facet.db9)]
// CHECK:STDOUT:     %.loc44_33.2: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc44_33.2 [symbolic = %Iterate.facet.loc44_33.5 (constants.%Iterate.facet.db9)]
// CHECK:STDOUT:     %specific_impl_fn.loc44_33.1: <specific function> = specific_impl_function %impl.elem2.loc44_33.1, @Iterate.WithSelf.NewCursor(constants.%Iterate.facet.db9) [symbolic = %specific_impl_fn.loc44_33.3 (constants.%specific_impl_fn.9bc)]
// CHECK:STDOUT:     %bound_method.loc44_33.2: <bound method> = bound_method %r.ref, %specific_impl_fn.loc44_33.1
// CHECK:STDOUT:     %var: ref @TestIterate.%as_type (%as_type.91d) = var_storage invalid
// CHECK:STDOUT:     %.loc44_32.1: @TestIterate.%R.as_type.loc40_79.1 (%R.as_type) = acquire_value %r.ref
// CHECK:STDOUT:     %Iterate.WithSelf.NewCursor.call: init @TestIterate.%as_type (%as_type.91d) to %var = call %bound_method.loc44_33.2(%.loc44_32.1)
// CHECK:STDOUT:     assign %var, %Iterate.WithSelf.NewCursor.call
// CHECK:STDOUT:     br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.next:
// CHECK:STDOUT:     %impl.elem3.loc44_33.1: @TestIterate.%.loc44_33.27 (%.38c) = impl_witness_access constants.%Iterate.lookup_impl_witness.47b, element3 [symbolic = %impl.elem3.loc44_33.2 (constants.%impl.elem3.b40)]
// CHECK:STDOUT:     %bound_method.loc44_33.3: <bound method> = bound_method %r.ref, %impl.elem3.loc44_33.1
// CHECK:STDOUT:     %Iterate.facet.loc44_33.3: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.47b) [symbolic = %Iterate.facet.loc44_33.5 (constants.%Iterate.facet.db9)]
// CHECK:STDOUT:     %.loc44_33.3: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc44_33.3 [symbolic = %Iterate.facet.loc44_33.5 (constants.%Iterate.facet.db9)]
// CHECK:STDOUT:     %Iterate.facet.loc44_33.4: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.47b) [symbolic = %Iterate.facet.loc44_33.5 (constants.%Iterate.facet.db9)]
// CHECK:STDOUT:     %.loc44_33.4: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc44_33.4 [symbolic = %Iterate.facet.loc44_33.5 (constants.%Iterate.facet.db9)]
// CHECK:STDOUT:     %.loc44_33.5: %facet_type.8a3 = converted constants.%as_type.91d, constants.%impl.elem1.efb [symbolic = %impl.elem1 (constants.%impl.elem1.efb)]
// CHECK:STDOUT:     %.loc44_33.6: %facet_type.8a3 = converted constants.%as_type.91d, constants.%impl.elem1.efb [symbolic = %impl.elem1 (constants.%impl.elem1.efb)]
// CHECK:STDOUT:     %specific_impl_fn.loc44_33.2: <specific function> = specific_impl_function %impl.elem3.loc44_33.1, @Iterate.WithSelf.Next(constants.%Iterate.facet.db9) [symbolic = %specific_impl_fn.loc44_33.4 (constants.%specific_impl_fn.fcf)]
// CHECK:STDOUT:     %bound_method.loc44_33.4: <bound method> = bound_method %r.ref, %specific_impl_fn.loc44_33.2
// CHECK:STDOUT:     %.loc44_33.7: ref %Optional.91f = temporary_storage
// CHECK:STDOUT:     %.loc44_32.2: @TestIterate.%R.as_type.loc40_79.1 (%R.as_type) = acquire_value %r.ref
// CHECK:STDOUT:     %Iterate.WithSelf.Next.call: init %Optional.91f to %.loc44_33.7 = call %bound_method.loc44_33.4(%.loc44_32.2, %var)
// CHECK:STDOUT:     %.loc44_33.8: ref %Optional.91f = temporary %.loc44_33.7, %Iterate.WithSelf.Next.call
// CHECK:STDOUT:     %.loc44_33.9: %Optional.HasValue.type.a6d = specific_constant imports.%Core.import_ref.4d4, @Optional(constants.%OptionalStorage.facet.7d7) [concrete = constants.%Optional.HasValue.1f0]
// CHECK:STDOUT:     %HasValue.ref: %Optional.HasValue.type.a6d = name_ref HasValue, %.loc44_33.9 [concrete = constants.%Optional.HasValue.1f0]
// CHECK:STDOUT:     %Optional.HasValue.bound: <bound method> = bound_method %.loc44_33.8, %HasValue.ref
// CHECK:STDOUT:     %Optional.HasValue.specific_fn: <specific function> = specific_function %HasValue.ref, @Optional.HasValue(constants.%OptionalStorage.facet.7d7) [concrete = constants.%Optional.HasValue.specific_fn.0fc]
// CHECK:STDOUT:     %bound_method.loc44_33.5: <bound method> = bound_method %.loc44_33.8, %Optional.HasValue.specific_fn
// CHECK:STDOUT:     %.loc44_33.10: %Optional.91f = acquire_value %.loc44_33.8
// CHECK:STDOUT:     %Optional.HasValue.call: init bool = call %bound_method.loc44_33.5(%.loc44_33.10)
// CHECK:STDOUT:     %.loc44_33.11: bool = value_of_initializer %Optional.HasValue.call
// CHECK:STDOUT:     %.loc44_33.12: bool = converted %Optional.HasValue.call, %.loc44_33.11
// CHECK:STDOUT:     if %.loc44_33.12 br !for.body else br !for.done
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.body:
// CHECK:STDOUT:     %.loc44_33.13: %Optional.Get.type.d3e = specific_constant imports.%Core.import_ref.059, @Optional(constants.%OptionalStorage.facet.7d7) [concrete = constants.%Optional.Get.2d4]
// CHECK:STDOUT:     %Get.ref: %Optional.Get.type.d3e = name_ref Get, %.loc44_33.13 [concrete = constants.%Optional.Get.2d4]
// CHECK:STDOUT:     %Optional.Get.bound: <bound method> = bound_method %.loc44_33.8, %Get.ref
// CHECK:STDOUT:     %Optional.Get.specific_fn: <specific function> = specific_function %Get.ref, @Optional.Get(constants.%OptionalStorage.facet.7d7) [concrete = constants.%Optional.Get.specific_fn.c83]
// CHECK:STDOUT:     %bound_method.loc44_33.6: <bound method> = bound_method %.loc44_33.8, %Optional.Get.specific_fn
// CHECK:STDOUT:     %.loc44_33.14: ref %DiffValueType = temporary_storage
// CHECK:STDOUT:     %.loc44_33.15: %Optional.91f = acquire_value %.loc44_33.8
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
// CHECK:STDOUT:     %facet_value.loc44_33.1: %facet_type.8a3 = facet_value constants.%DiffValueType, (constants.%custom_witness.5760f4.1, constants.%DestroyImpl.impl_witness.39a) [concrete = constants.%facet_value.dc6]
// CHECK:STDOUT:     %.loc44_33.18: %facet_type.8a3 = converted constants.%DiffValueType, %facet_value.loc44_33.1 [concrete = constants.%facet_value.dc6]
// CHECK:STDOUT:     %facet_value.loc44_33.2: %facet_type.8a3 = facet_value constants.%DiffValueType, (constants.%custom_witness.5760f4.1, constants.%DestroyImpl.impl_witness.39a) [concrete = constants.%facet_value.dc6]
// CHECK:STDOUT:     %.loc44_33.19: %facet_type.8a3 = converted constants.%DiffValueType, %facet_value.loc44_33.2 [concrete = constants.%facet_value.dc6]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %SelfDestruct.call.loc44_33.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc44_33.16)
// CHECK:STDOUT:     %facet_value.loc44_33.3: %facet_type.8a3 = facet_value constants.%Optional.91f, (constants.%custom_witness.b80ef6.7, constants.%DestroyImpl.impl_witness.f90) [concrete = constants.%facet_value.451]
// CHECK:STDOUT:     %.loc44_33.20: %facet_type.8a3 = converted constants.%Optional.91f, %facet_value.loc44_33.3 [concrete = constants.%facet_value.451]
// CHECK:STDOUT:     %facet_value.loc44_33.4: %facet_type.8a3 = facet_value constants.%Optional.91f, (constants.%custom_witness.b80ef6.7, constants.%DestroyImpl.impl_witness.f90) [concrete = constants.%facet_value.451]
// CHECK:STDOUT:     %.loc44_33.21: %facet_type.8a3 = converted constants.%Optional.91f, %facet_value.loc44_33.4 [concrete = constants.%facet_value.451]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %SelfDestruct.call.loc44_33.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc44_33.8)
// CHECK:STDOUT:     br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.done:
// CHECK:STDOUT:     %facet_value.loc44_33.5: %facet_type.8a3 = facet_value constants.%Optional.91f, (constants.%custom_witness.b80ef6.7, constants.%DestroyImpl.impl_witness.f90) [concrete = constants.%facet_value.451]
// CHECK:STDOUT:     %.loc44_33.22: %facet_type.8a3 = converted constants.%Optional.91f, %facet_value.loc44_33.5 [concrete = constants.%facet_value.451]
// CHECK:STDOUT:     %facet_value.loc44_33.6: %facet_type.8a3 = facet_value constants.%Optional.91f, (constants.%custom_witness.b80ef6.7, constants.%DestroyImpl.impl_witness.f90) [concrete = constants.%facet_value.451]
// CHECK:STDOUT:     %.loc44_33.23: %facet_type.8a3 = converted constants.%Optional.91f, %facet_value.loc44_33.6 [concrete = constants.%facet_value.451]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %SelfDestruct.call.loc44_33.3: init %empty_tuple.type = call %SelfDestruct.specific_fn.3(%.loc44_33.8)
// CHECK:STDOUT:     %.loc44_33.24: %facet_type.8a3 = converted constants.%as_type.91d, constants.%impl.elem1.efb [symbolic = %impl.elem1 (constants.%impl.elem1.efb)]
// CHECK:STDOUT:     %.loc44_33.25: %facet_type.8a3 = converted constants.%as_type.91d, constants.%impl.elem1.efb [symbolic = %impl.elem1 (constants.%impl.elem1.efb)]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %SelfDestruct.call.loc44_33.4: init %empty_tuple.type = call %SelfDestruct.specific_fn.4(%var)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc44_33.1(%self.param: ref %.a18) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc44_33.2(%self.param: ref %MaybeUnformed.699) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc44_33.3(%self.param: ref bool) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc44_33.4(%self.param: ref %struct_type.value.has_value.608) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc44_33.1: ref bool = struct_access %self.param, element1
// CHECK:STDOUT:   %facet_value.loc44_33.1: %facet_type.8a3 = facet_value bool, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.deb) [concrete = constants.%facet_value.a1b]
// CHECK:STDOUT:   %.loc44_33.2: %facet_type.8a3 = converted bool, %facet_value.loc44_33.1 [concrete = constants.%facet_value.a1b]
// CHECK:STDOUT:   %facet_value.loc44_33.2: %facet_type.8a3 = facet_value bool, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.deb) [concrete = constants.%facet_value.a1b]
// CHECK:STDOUT:   %.loc44_33.3: %facet_type.8a3 = converted bool, %facet_value.loc44_33.2 [concrete = constants.%facet_value.a1b]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc44_33.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc44_33.1)
// CHECK:STDOUT:   %.loc44_33.4: ref %MaybeUnformed.699 = struct_access %self.param, element0
// CHECK:STDOUT:   %facet_value.loc44_33.3: %facet_type.8a3 = facet_value constants.%MaybeUnformed.699, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.38a) [concrete = constants.%facet_value.67f]
// CHECK:STDOUT:   %.loc44_33.5: %facet_type.8a3 = converted constants.%MaybeUnformed.699, %facet_value.loc44_33.3 [concrete = constants.%facet_value.67f]
// CHECK:STDOUT:   %facet_value.loc44_33.4: %facet_type.8a3 = facet_value constants.%MaybeUnformed.699, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.38a) [concrete = constants.%facet_value.67f]
// CHECK:STDOUT:   %.loc44_33.6: %facet_type.8a3 = converted constants.%MaybeUnformed.699, %facet_value.loc44_33.4 [concrete = constants.%facet_value.67f]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc44_33.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc44_33.4)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc44_33.5(%self.param: ref %DefaultOptionalStorage.670) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %has_value.ref: %DefaultOptionalStorage.elem.0c5 = name_ref has_value, imports.%Core.import_ref.f1da [concrete = imports.%field_decl.431]
// CHECK:STDOUT:   %.loc44_33.1: ref bool = class_element_access %self.param, element1
// CHECK:STDOUT:   %facet_value.loc44_33.1: %facet_type.8a3 = facet_value bool, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.deb) [concrete = constants.%facet_value.a1b]
// CHECK:STDOUT:   %.loc44_33.2: %facet_type.8a3 = converted bool, %facet_value.loc44_33.1 [concrete = constants.%facet_value.a1b]
// CHECK:STDOUT:   %facet_value.loc44_33.2: %facet_type.8a3 = facet_value bool, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.deb) [concrete = constants.%facet_value.a1b]
// CHECK:STDOUT:   %.loc44_33.3: %facet_type.8a3 = converted bool, %facet_value.loc44_33.2 [concrete = constants.%facet_value.a1b]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc44_33.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc44_33.1)
// CHECK:STDOUT:   %value.ref: %DefaultOptionalStorage.elem.7b1 = name_ref value, imports.%Core.import_ref.146b [concrete = imports.%field_decl.31e]
// CHECK:STDOUT:   %.loc44_33.4: ref %MaybeUnformed.699 = class_element_access %self.param, element0
// CHECK:STDOUT:   %facet_value.loc44_33.3: %facet_type.8a3 = facet_value constants.%MaybeUnformed.699, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.38a) [concrete = constants.%facet_value.67f]
// CHECK:STDOUT:   %.loc44_33.5: %facet_type.8a3 = converted constants.%MaybeUnformed.699, %facet_value.loc44_33.3 [concrete = constants.%facet_value.67f]
// CHECK:STDOUT:   %facet_value.loc44_33.4: %facet_type.8a3 = facet_value constants.%MaybeUnformed.699, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.38a) [concrete = constants.%facet_value.67f]
// CHECK:STDOUT:   %.loc44_33.6: %facet_type.8a3 = converted constants.%MaybeUnformed.699, %facet_value.loc44_33.4 [concrete = constants.%facet_value.67f]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc44_33.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc44_33.4)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc44_33.6(%self.param: ref %Optional.91f) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestRangeFor(%m.param: ref %DiffMutableRange, %c.param: %DiffConstRange) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %i.patt.loc59: %pattern_type.149 = value_binding_pattern i [concrete = constants.%i.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %m.ref: ref %DiffMutableRange = name_ref m, %m
// CHECK:STDOUT:   %impl.elem2.loc59: %.68d = impl_witness_access constants.%Iterate.impl_witness.7b0, element2 [concrete = constants.%T.as_type.as.Iterate.impl.NewCursor.64a]
// CHECK:STDOUT:   %bound_method.loc59_33.1: <bound method> = bound_method %m.ref, %impl.elem2.loc59
// CHECK:STDOUT:   %facet_value.loc59_33.1: %CppRange_where.type.d682d6.1 = facet_value constants.%DiffMutableRange, (constants.%custom_witness.5760f4.1, constants.%DestroyImpl.impl_witness.39a, constants.%custom_witness.d0da43.1, constants.%custom_witness.aa8, constants.%custom_witness.3815f0.1, constants.%DestroyImpl.impl_witness.09a, constants.%custom_witness.392, constants.%custom_witness.1c9, constants.%custom_witness.723, constants.%custom_witness.3815f0.2, constants.%DestroyImpl.impl_witness.71f) [concrete = constants.%facet_value.4f2]
// CHECK:STDOUT:   %.loc59_33.1: %CppRange_where.type.d682d6.1 = converted constants.%DiffMutableRange, %facet_value.loc59_33.1 [concrete = constants.%facet_value.4f2]
// CHECK:STDOUT:   %facet_value.loc59_33.2: %CppRange_where.type.d682d6.1 = facet_value constants.%DiffMutableRange, (constants.%custom_witness.5760f4.1, constants.%DestroyImpl.impl_witness.39a, constants.%custom_witness.d0da43.1, constants.%custom_witness.aa8, constants.%custom_witness.3815f0.1, constants.%DestroyImpl.impl_witness.09a, constants.%custom_witness.392, constants.%custom_witness.1c9, constants.%custom_witness.723, constants.%custom_witness.3815f0.2, constants.%DestroyImpl.impl_witness.71f) [concrete = constants.%facet_value.4f2]
// CHECK:STDOUT:   %.loc59_33.2: %CppRange_where.type.d682d6.1 = converted constants.%DiffMutableRange, %facet_value.loc59_33.2 [concrete = constants.%facet_value.4f2]
// CHECK:STDOUT:   %specific_fn.loc59_33.1: <specific function> = specific_function %impl.elem2.loc59, @T.as_type.as.Iterate.impl.NewCursor(constants.%facet_value.4f2) [concrete = constants.%T.as_type.as.Iterate.impl.NewCursor.specific_fn.1fd]
// CHECK:STDOUT:   %bound_method.loc59_33.2: <bound method> = bound_method %m.ref, %specific_fn.loc59_33.1
// CHECK:STDOUT:   %var.loc59: ref %tuple.type.b18 = var_storage invalid
// CHECK:STDOUT:   %.loc59_32.1: %DiffMutableRange = acquire_value %m.ref
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.call.loc59: init %tuple.type.b18 to %var.loc59 = call %bound_method.loc59_33.2(%.loc59_32.1)
// CHECK:STDOUT:   assign %var.loc59, %T.as_type.as.Iterate.impl.NewCursor.call.loc59
// CHECK:STDOUT:   br !for.next.loc59
// CHECK:STDOUT:
// CHECK:STDOUT: !for.next.loc59:
// CHECK:STDOUT:   %impl.elem3.loc59: %.b83 = impl_witness_access constants.%Iterate.impl_witness.7b0, element3 [concrete = constants.%T.as_type.as.Iterate.impl.Next.ed2]
// CHECK:STDOUT:   %bound_method.loc59_33.3: <bound method> = bound_method %m.ref, %impl.elem3.loc59
// CHECK:STDOUT:   %facet_value.loc59_33.3: %CppRange_where.type.d682d6.1 = facet_value constants.%DiffMutableRange, (constants.%custom_witness.5760f4.1, constants.%DestroyImpl.impl_witness.39a, constants.%custom_witness.d0da43.1, constants.%custom_witness.aa8, constants.%custom_witness.3815f0.1, constants.%DestroyImpl.impl_witness.09a, constants.%custom_witness.392, constants.%custom_witness.1c9, constants.%custom_witness.723, constants.%custom_witness.3815f0.2, constants.%DestroyImpl.impl_witness.71f) [concrete = constants.%facet_value.4f2]
// CHECK:STDOUT:   %.loc59_33.3: %CppRange_where.type.d682d6.1 = converted constants.%DiffMutableRange, %facet_value.loc59_33.3 [concrete = constants.%facet_value.4f2]
// CHECK:STDOUT:   %facet_value.loc59_33.4: %CppRange_where.type.d682d6.1 = facet_value constants.%DiffMutableRange, (constants.%custom_witness.5760f4.1, constants.%DestroyImpl.impl_witness.39a, constants.%custom_witness.d0da43.1, constants.%custom_witness.aa8, constants.%custom_witness.3815f0.1, constants.%DestroyImpl.impl_witness.09a, constants.%custom_witness.392, constants.%custom_witness.1c9, constants.%custom_witness.723, constants.%custom_witness.3815f0.2, constants.%DestroyImpl.impl_witness.71f) [concrete = constants.%facet_value.4f2]
// CHECK:STDOUT:   %.loc59_33.4: %CppRange_where.type.d682d6.1 = converted constants.%DiffMutableRange, %facet_value.loc59_33.4 [concrete = constants.%facet_value.4f2]
// CHECK:STDOUT:   %specific_fn.loc59_33.2: <specific function> = specific_function %impl.elem3.loc59, @T.as_type.as.Iterate.impl.Next(constants.%facet_value.4f2) [concrete = constants.%T.as_type.as.Iterate.impl.Next.specific_fn.90b]
// CHECK:STDOUT:   %bound_method.loc59_33.4: <bound method> = bound_method %m.ref, %specific_fn.loc59_33.2
// CHECK:STDOUT:   %.loc59_33.5: ref %Optional.91f = temporary_storage
// CHECK:STDOUT:   %.loc59_32.2: %DiffMutableRange = acquire_value %m.ref
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.call.loc59: init %Optional.91f to %.loc59_33.5 = call %bound_method.loc59_33.4(%.loc59_32.2, %var.loc59)
// CHECK:STDOUT:   %.loc59_33.6: ref %Optional.91f = temporary %.loc59_33.5, %T.as_type.as.Iterate.impl.Next.call.loc59
// CHECK:STDOUT:   %.loc59_33.7: %Optional.HasValue.type.a6d = specific_constant imports.%Core.import_ref.4d4, @Optional(constants.%OptionalStorage.facet.7d7) [concrete = constants.%Optional.HasValue.1f0]
// CHECK:STDOUT:   %HasValue.ref.loc59: %Optional.HasValue.type.a6d = name_ref HasValue, %.loc59_33.7 [concrete = constants.%Optional.HasValue.1f0]
// CHECK:STDOUT:   %Optional.HasValue.bound.loc59: <bound method> = bound_method %.loc59_33.6, %HasValue.ref.loc59
// CHECK:STDOUT:   %Optional.HasValue.specific_fn.loc59: <specific function> = specific_function %HasValue.ref.loc59, @Optional.HasValue(constants.%OptionalStorage.facet.7d7) [concrete = constants.%Optional.HasValue.specific_fn.0fc]
// CHECK:STDOUT:   %bound_method.loc59_33.5: <bound method> = bound_method %.loc59_33.6, %Optional.HasValue.specific_fn.loc59
// CHECK:STDOUT:   %.loc59_33.8: %Optional.91f = acquire_value %.loc59_33.6
// CHECK:STDOUT:   %Optional.HasValue.call.loc59: init bool = call %bound_method.loc59_33.5(%.loc59_33.8)
// CHECK:STDOUT:   %.loc59_33.9: bool = value_of_initializer %Optional.HasValue.call.loc59
// CHECK:STDOUT:   %.loc59_33.10: bool = converted %Optional.HasValue.call.loc59, %.loc59_33.9
// CHECK:STDOUT:   if %.loc59_33.10 br !for.body.loc59 else br !for.done.loc59
// CHECK:STDOUT:
// CHECK:STDOUT: !for.body.loc59:
// CHECK:STDOUT:   %.loc59_33.11: %Optional.Get.type.d3e = specific_constant imports.%Core.import_ref.059, @Optional(constants.%OptionalStorage.facet.7d7) [concrete = constants.%Optional.Get.2d4]
// CHECK:STDOUT:   %Get.ref.loc59: %Optional.Get.type.d3e = name_ref Get, %.loc59_33.11 [concrete = constants.%Optional.Get.2d4]
// CHECK:STDOUT:   %Optional.Get.bound.loc59: <bound method> = bound_method %.loc59_33.6, %Get.ref.loc59
// CHECK:STDOUT:   %Optional.Get.specific_fn.loc59: <specific function> = specific_function %Get.ref.loc59, @Optional.Get(constants.%OptionalStorage.facet.7d7) [concrete = constants.%Optional.Get.specific_fn.c83]
// CHECK:STDOUT:   %bound_method.loc59_33.6: <bound method> = bound_method %.loc59_33.6, %Optional.Get.specific_fn.loc59
// CHECK:STDOUT:   %.loc59_33.12: ref %DiffValueType = temporary_storage
// CHECK:STDOUT:   %.loc59_33.13: %Optional.91f = acquire_value %.loc59_33.6
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
// CHECK:STDOUT:   %facet_value.loc59_33.5: %facet_type.8a3 = facet_value constants.%DiffValueType, (constants.%custom_witness.5760f4.1, constants.%DestroyImpl.impl_witness.39a) [concrete = constants.%facet_value.dc6]
// CHECK:STDOUT:   %.loc59_33.16: %facet_type.8a3 = converted constants.%DiffValueType, %facet_value.loc59_33.5 [concrete = constants.%facet_value.dc6]
// CHECK:STDOUT:   %facet_value.loc59_33.6: %facet_type.8a3 = facet_value constants.%DiffValueType, (constants.%custom_witness.5760f4.1, constants.%DestroyImpl.impl_witness.39a) [concrete = constants.%facet_value.dc6]
// CHECK:STDOUT:   %.loc59_33.17: %facet_type.8a3 = converted constants.%DiffValueType, %facet_value.loc59_33.6 [concrete = constants.%facet_value.dc6]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc59_33.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc59_33.14)
// CHECK:STDOUT:   %facet_value.loc59_33.7: %facet_type.8a3 = facet_value constants.%Optional.91f, (constants.%custom_witness.b80ef6.7, constants.%DestroyImpl.impl_witness.f90) [concrete = constants.%facet_value.451]
// CHECK:STDOUT:   %.loc59_33.18: %facet_type.8a3 = converted constants.%Optional.91f, %facet_value.loc59_33.7 [concrete = constants.%facet_value.451]
// CHECK:STDOUT:   %facet_value.loc59_33.8: %facet_type.8a3 = facet_value constants.%Optional.91f, (constants.%custom_witness.b80ef6.7, constants.%DestroyImpl.impl_witness.f90) [concrete = constants.%facet_value.451]
// CHECK:STDOUT:   %.loc59_33.19: %facet_type.8a3 = converted constants.%Optional.91f, %facet_value.loc59_33.8 [concrete = constants.%facet_value.451]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc59_33.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc59_33.6)
// CHECK:STDOUT:   br !for.next.loc59
// CHECK:STDOUT:
// CHECK:STDOUT: !for.done.loc59:
// CHECK:STDOUT:   %facet_value.loc59_33.9: %facet_type.8a3 = facet_value constants.%Optional.91f, (constants.%custom_witness.b80ef6.7, constants.%DestroyImpl.impl_witness.f90) [concrete = constants.%facet_value.451]
// CHECK:STDOUT:   %.loc59_33.20: %facet_type.8a3 = converted constants.%Optional.91f, %facet_value.loc59_33.9 [concrete = constants.%facet_value.451]
// CHECK:STDOUT:   %facet_value.loc59_33.10: %facet_type.8a3 = facet_value constants.%Optional.91f, (constants.%custom_witness.b80ef6.7, constants.%DestroyImpl.impl_witness.f90) [concrete = constants.%facet_value.451]
// CHECK:STDOUT:   %.loc59_33.21: %facet_type.8a3 = converted constants.%Optional.91f, %facet_value.loc59_33.10 [concrete = constants.%facet_value.451]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc59_33.3: init %empty_tuple.type = call %SelfDestruct.specific_fn.3(%.loc59_33.6)
// CHECK:STDOUT:   %facet_value.loc59_33.11: %facet_type.8a3 = facet_value constants.%tuple.type.b18, (constants.%custom_witness.b80ef6.8, constants.%DestroyImpl.impl_witness.a87) [concrete = constants.%facet_value.f38]
// CHECK:STDOUT:   %.loc59_33.22: %facet_type.8a3 = converted constants.%tuple.type.b18, %facet_value.loc59_33.11 [concrete = constants.%facet_value.f38]
// CHECK:STDOUT:   %facet_value.loc59_33.12: %facet_type.8a3 = facet_value constants.%tuple.type.b18, (constants.%custom_witness.b80ef6.8, constants.%DestroyImpl.impl_witness.a87) [concrete = constants.%facet_value.f38]
// CHECK:STDOUT:   %.loc59_33.23: %facet_type.8a3 = converted constants.%tuple.type.b18, %facet_value.loc59_33.12 [concrete = constants.%facet_value.f38]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc59_33.4: init %empty_tuple.type = call %SelfDestruct.specific_fn.4(%var.loc59)
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %i.patt.loc63: %pattern_type.149 = value_binding_pattern i [concrete = constants.%i.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c.ref: %DiffConstRange = name_ref c, %c
// CHECK:STDOUT:   %impl.elem2.loc63: %.edf = impl_witness_access constants.%Iterate.impl_witness.532, element2 [concrete = constants.%T.as_type.as.Iterate.impl.NewCursor.5aa]
// CHECK:STDOUT:   %bound_method.loc63_33.1: <bound method> = bound_method %c.ref, %impl.elem2.loc63
// CHECK:STDOUT:   %facet_value.loc63_33.1: %CppRange_where.type.d682d6.1 = facet_value constants.%DiffConstRange, (constants.%custom_witness.5760f4.1, constants.%DestroyImpl.impl_witness.39a, constants.%custom_witness.d0da43.1, constants.%custom_witness.7b4, constants.%custom_witness.3d0529.1, constants.%DestroyImpl.impl_witness.b0f, constants.%custom_witness.0c0, constants.%custom_witness.ffb, constants.%custom_witness.a26, constants.%custom_witness.3d0529.2, constants.%DestroyImpl.impl_witness.4ee) [concrete = constants.%facet_value.32d]
// CHECK:STDOUT:   %.loc63_33.1: %CppRange_where.type.d682d6.1 = converted constants.%DiffConstRange, %facet_value.loc63_33.1 [concrete = constants.%facet_value.32d]
// CHECK:STDOUT:   %facet_value.loc63_33.2: %CppRange_where.type.d682d6.1 = facet_value constants.%DiffConstRange, (constants.%custom_witness.5760f4.1, constants.%DestroyImpl.impl_witness.39a, constants.%custom_witness.d0da43.1, constants.%custom_witness.7b4, constants.%custom_witness.3d0529.1, constants.%DestroyImpl.impl_witness.b0f, constants.%custom_witness.0c0, constants.%custom_witness.ffb, constants.%custom_witness.a26, constants.%custom_witness.3d0529.2, constants.%DestroyImpl.impl_witness.4ee) [concrete = constants.%facet_value.32d]
// CHECK:STDOUT:   %.loc63_33.2: %CppRange_where.type.d682d6.1 = converted constants.%DiffConstRange, %facet_value.loc63_33.2 [concrete = constants.%facet_value.32d]
// CHECK:STDOUT:   %specific_fn.loc63_33.1: <specific function> = specific_function %impl.elem2.loc63, @T.as_type.as.Iterate.impl.NewCursor(constants.%facet_value.32d) [concrete = constants.%T.as_type.as.Iterate.impl.NewCursor.specific_fn.ce3]
// CHECK:STDOUT:   %bound_method.loc63_33.2: <bound method> = bound_method %c.ref, %specific_fn.loc63_33.1
// CHECK:STDOUT:   %var.loc63: ref %tuple.type.df5 = var_storage invalid
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.call.loc63: init %tuple.type.df5 to %var.loc63 = call %bound_method.loc63_33.2(%c.ref)
// CHECK:STDOUT:   assign %var.loc63, %T.as_type.as.Iterate.impl.NewCursor.call.loc63
// CHECK:STDOUT:   br !for.next.loc63
// CHECK:STDOUT:
// CHECK:STDOUT: !for.next.loc63:
// CHECK:STDOUT:   %impl.elem3.loc63: %.24c = impl_witness_access constants.%Iterate.impl_witness.532, element3 [concrete = constants.%T.as_type.as.Iterate.impl.Next.79c]
// CHECK:STDOUT:   %bound_method.loc63_33.3: <bound method> = bound_method %c.ref, %impl.elem3.loc63
// CHECK:STDOUT:   %facet_value.loc63_33.3: %CppRange_where.type.d682d6.1 = facet_value constants.%DiffConstRange, (constants.%custom_witness.5760f4.1, constants.%DestroyImpl.impl_witness.39a, constants.%custom_witness.d0da43.1, constants.%custom_witness.7b4, constants.%custom_witness.3d0529.1, constants.%DestroyImpl.impl_witness.b0f, constants.%custom_witness.0c0, constants.%custom_witness.ffb, constants.%custom_witness.a26, constants.%custom_witness.3d0529.2, constants.%DestroyImpl.impl_witness.4ee) [concrete = constants.%facet_value.32d]
// CHECK:STDOUT:   %.loc63_33.3: %CppRange_where.type.d682d6.1 = converted constants.%DiffConstRange, %facet_value.loc63_33.3 [concrete = constants.%facet_value.32d]
// CHECK:STDOUT:   %facet_value.loc63_33.4: %CppRange_where.type.d682d6.1 = facet_value constants.%DiffConstRange, (constants.%custom_witness.5760f4.1, constants.%DestroyImpl.impl_witness.39a, constants.%custom_witness.d0da43.1, constants.%custom_witness.7b4, constants.%custom_witness.3d0529.1, constants.%DestroyImpl.impl_witness.b0f, constants.%custom_witness.0c0, constants.%custom_witness.ffb, constants.%custom_witness.a26, constants.%custom_witness.3d0529.2, constants.%DestroyImpl.impl_witness.4ee) [concrete = constants.%facet_value.32d]
// CHECK:STDOUT:   %.loc63_33.4: %CppRange_where.type.d682d6.1 = converted constants.%DiffConstRange, %facet_value.loc63_33.4 [concrete = constants.%facet_value.32d]
// CHECK:STDOUT:   %specific_fn.loc63_33.2: <specific function> = specific_function %impl.elem3.loc63, @T.as_type.as.Iterate.impl.Next(constants.%facet_value.32d) [concrete = constants.%T.as_type.as.Iterate.impl.Next.specific_fn.d8c]
// CHECK:STDOUT:   %bound_method.loc63_33.4: <bound method> = bound_method %c.ref, %specific_fn.loc63_33.2
// CHECK:STDOUT:   %.loc63_33.5: ref %Optional.91f = temporary_storage
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.call.loc63: init %Optional.91f to %.loc63_33.5 = call %bound_method.loc63_33.4(%c.ref, %var.loc63)
// CHECK:STDOUT:   %.loc63_33.6: ref %Optional.91f = temporary %.loc63_33.5, %T.as_type.as.Iterate.impl.Next.call.loc63
// CHECK:STDOUT:   %.loc63_33.7: %Optional.HasValue.type.a6d = specific_constant imports.%Core.import_ref.4d4, @Optional(constants.%OptionalStorage.facet.7d7) [concrete = constants.%Optional.HasValue.1f0]
// CHECK:STDOUT:   %HasValue.ref.loc63: %Optional.HasValue.type.a6d = name_ref HasValue, %.loc63_33.7 [concrete = constants.%Optional.HasValue.1f0]
// CHECK:STDOUT:   %Optional.HasValue.bound.loc63: <bound method> = bound_method %.loc63_33.6, %HasValue.ref.loc63
// CHECK:STDOUT:   %Optional.HasValue.specific_fn.loc63: <specific function> = specific_function %HasValue.ref.loc63, @Optional.HasValue(constants.%OptionalStorage.facet.7d7) [concrete = constants.%Optional.HasValue.specific_fn.0fc]
// CHECK:STDOUT:   %bound_method.loc63_33.5: <bound method> = bound_method %.loc63_33.6, %Optional.HasValue.specific_fn.loc63
// CHECK:STDOUT:   %.loc63_33.8: %Optional.91f = acquire_value %.loc63_33.6
// CHECK:STDOUT:   %Optional.HasValue.call.loc63: init bool = call %bound_method.loc63_33.5(%.loc63_33.8)
// CHECK:STDOUT:   %.loc63_33.9: bool = value_of_initializer %Optional.HasValue.call.loc63
// CHECK:STDOUT:   %.loc63_33.10: bool = converted %Optional.HasValue.call.loc63, %.loc63_33.9
// CHECK:STDOUT:   if %.loc63_33.10 br !for.body.loc63 else br !for.done.loc63
// CHECK:STDOUT:
// CHECK:STDOUT: !for.body.loc63:
// CHECK:STDOUT:   %.loc63_33.11: %Optional.Get.type.d3e = specific_constant imports.%Core.import_ref.059, @Optional(constants.%OptionalStorage.facet.7d7) [concrete = constants.%Optional.Get.2d4]
// CHECK:STDOUT:   %Get.ref.loc63: %Optional.Get.type.d3e = name_ref Get, %.loc63_33.11 [concrete = constants.%Optional.Get.2d4]
// CHECK:STDOUT:   %Optional.Get.bound.loc63: <bound method> = bound_method %.loc63_33.6, %Get.ref.loc63
// CHECK:STDOUT:   %Optional.Get.specific_fn.loc63: <specific function> = specific_function %Get.ref.loc63, @Optional.Get(constants.%OptionalStorage.facet.7d7) [concrete = constants.%Optional.Get.specific_fn.c83]
// CHECK:STDOUT:   %bound_method.loc63_33.6: <bound method> = bound_method %.loc63_33.6, %Optional.Get.specific_fn.loc63
// CHECK:STDOUT:   %.loc63_33.12: ref %DiffValueType = temporary_storage
// CHECK:STDOUT:   %.loc63_33.13: %Optional.91f = acquire_value %.loc63_33.6
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
// CHECK:STDOUT:   %facet_value.loc63_33.5: %facet_type.8a3 = facet_value constants.%DiffValueType, (constants.%custom_witness.5760f4.1, constants.%DestroyImpl.impl_witness.39a) [concrete = constants.%facet_value.dc6]
// CHECK:STDOUT:   %.loc63_33.16: %facet_type.8a3 = converted constants.%DiffValueType, %facet_value.loc63_33.5 [concrete = constants.%facet_value.dc6]
// CHECK:STDOUT:   %facet_value.loc63_33.6: %facet_type.8a3 = facet_value constants.%DiffValueType, (constants.%custom_witness.5760f4.1, constants.%DestroyImpl.impl_witness.39a) [concrete = constants.%facet_value.dc6]
// CHECK:STDOUT:   %.loc63_33.17: %facet_type.8a3 = converted constants.%DiffValueType, %facet_value.loc63_33.6 [concrete = constants.%facet_value.dc6]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc63_33.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.5(%.loc63_33.14)
// CHECK:STDOUT:   %facet_value.loc63_33.7: %facet_type.8a3 = facet_value constants.%Optional.91f, (constants.%custom_witness.b80ef6.7, constants.%DestroyImpl.impl_witness.f90) [concrete = constants.%facet_value.451]
// CHECK:STDOUT:   %.loc63_33.18: %facet_type.8a3 = converted constants.%Optional.91f, %facet_value.loc63_33.7 [concrete = constants.%facet_value.451]
// CHECK:STDOUT:   %facet_value.loc63_33.8: %facet_type.8a3 = facet_value constants.%Optional.91f, (constants.%custom_witness.b80ef6.7, constants.%DestroyImpl.impl_witness.f90) [concrete = constants.%facet_value.451]
// CHECK:STDOUT:   %.loc63_33.19: %facet_type.8a3 = converted constants.%Optional.91f, %facet_value.loc63_33.8 [concrete = constants.%facet_value.451]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc63_33.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.6(%.loc63_33.6)
// CHECK:STDOUT:   br !for.next.loc63
// CHECK:STDOUT:
// CHECK:STDOUT: !for.done.loc63:
// CHECK:STDOUT:   %facet_value.loc63_33.9: %facet_type.8a3 = facet_value constants.%Optional.91f, (constants.%custom_witness.b80ef6.7, constants.%DestroyImpl.impl_witness.f90) [concrete = constants.%facet_value.451]
// CHECK:STDOUT:   %.loc63_33.20: %facet_type.8a3 = converted constants.%Optional.91f, %facet_value.loc63_33.9 [concrete = constants.%facet_value.451]
// CHECK:STDOUT:   %facet_value.loc63_33.10: %facet_type.8a3 = facet_value constants.%Optional.91f, (constants.%custom_witness.b80ef6.7, constants.%DestroyImpl.impl_witness.f90) [concrete = constants.%facet_value.451]
// CHECK:STDOUT:   %.loc63_33.21: %facet_type.8a3 = converted constants.%Optional.91f, %facet_value.loc63_33.10 [concrete = constants.%facet_value.451]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc63_33.3: init %empty_tuple.type = call %SelfDestruct.specific_fn.7(%.loc63_33.6)
// CHECK:STDOUT:   %facet_value.loc63_33.11: %facet_type.8a3 = facet_value constants.%tuple.type.df5, (constants.%custom_witness.b80ef6.9, constants.%DestroyImpl.impl_witness.af6) [concrete = constants.%facet_value.591]
// CHECK:STDOUT:   %.loc63_33.22: %facet_type.8a3 = converted constants.%tuple.type.df5, %facet_value.loc63_33.11 [concrete = constants.%facet_value.591]
// CHECK:STDOUT:   %facet_value.loc63_33.12: %facet_type.8a3 = facet_value constants.%tuple.type.df5, (constants.%custom_witness.b80ef6.9, constants.%DestroyImpl.impl_witness.af6) [concrete = constants.%facet_value.591]
// CHECK:STDOUT:   %.loc63_33.23: %facet_type.8a3 = converted constants.%tuple.type.df5, %facet_value.loc63_33.12 [concrete = constants.%facet_value.591]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc63_33.4: init %empty_tuple.type = call %SelfDestruct.specific_fn.8(%var.loc63)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestIterate(constants.%R) {
// CHECK:STDOUT:   %R.patt.loc40_17.2 => constants.%R.patt
// CHECK:STDOUT:   %R.loc40_17.1 => constants.%R
// CHECK:STDOUT:   %R.as_type.loc40_79.1 => constants.%R.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.bac
// CHECK:STDOUT:   %r.patt.loc40_77.2 => constants.%r.patt.cb9
// CHECK:STDOUT:   %r.param_patt.loc40_72.2 => constants.%r.param_patt.4be
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestIterate(constants.%facet_value.8c6) {
// CHECK:STDOUT:   %R.patt.loc40_17.2 => constants.%R.patt
// CHECK:STDOUT:   %R.loc40_17.1 => constants.%facet_value.8c6
// CHECK:STDOUT:   %R.as_type.loc40_79.1 => constants.%DiffMutableRange
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.611
// CHECK:STDOUT:   %r.patt.loc40_77.2 => constants.%r.patt.5fe
// CHECK:STDOUT:   %r.param_patt.loc40_72.2 => constants.%r.param_patt.a4b
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc40 => constants.%complete_type.357
// CHECK:STDOUT:   %Iterate.lookup_impl_witness => constants.%Iterate.impl_witness.7b0
// CHECK:STDOUT:   %Iterate.facet.loc44_33.5 => constants.%Iterate.facet.441
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type => constants.%Iterate.WithSelf.NewCursor.type.cc8
// CHECK:STDOUT:   %.loc44_33.26 => constants.%.68d
// CHECK:STDOUT:   %impl.elem2.loc44_33.2 => constants.%T.as_type.as.Iterate.impl.NewCursor.64a
// CHECK:STDOUT:   %specific_impl_fn.loc44_33.3 => constants.%T.as_type.as.Iterate.impl.NewCursor.specific_fn.1fd
// CHECK:STDOUT:   %impl.elem1 => constants.%facet_value.f38
// CHECK:STDOUT:   %as_type => constants.%tuple.type.b18
// CHECK:STDOUT:   %require_complete.1 => constants.%complete_type.bfe
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type => constants.%Iterate.WithSelf.Next.type.0b4
// CHECK:STDOUT:   %.loc44_33.27 => constants.%.b83
// CHECK:STDOUT:   %impl.elem3.loc44_33.2 => constants.%T.as_type.as.Iterate.impl.Next.ed2
// CHECK:STDOUT:   %specific_impl_fn.loc44_33.4 => constants.%T.as_type.as.Iterate.impl.Next.specific_fn.90b
// CHECK:STDOUT:   %SelfDestruct.specific_fn.6 => constants.%SelfDestruct.specific_fn.b15
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestIterate(constants.%facet_value.4df) {
// CHECK:STDOUT:   %R.patt.loc40_17.2 => constants.%R.patt
// CHECK:STDOUT:   %R.loc40_17.1 => constants.%facet_value.4df
// CHECK:STDOUT:   %R.as_type.loc40_79.1 => constants.%DiffConstRange
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.7dd
// CHECK:STDOUT:   %r.patt.loc40_77.2 => constants.%r.patt.9c0
// CHECK:STDOUT:   %r.param_patt.loc40_72.2 => constants.%r.param_patt.992
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc40 => constants.%complete_type.357
// CHECK:STDOUT:   %Iterate.lookup_impl_witness => constants.%Iterate.impl_witness.532
// CHECK:STDOUT:   %Iterate.facet.loc44_33.5 => constants.%Iterate.facet.001
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type => constants.%Iterate.WithSelf.NewCursor.type.7ef
// CHECK:STDOUT:   %.loc44_33.26 => constants.%.edf
// CHECK:STDOUT:   %impl.elem2.loc44_33.2 => constants.%T.as_type.as.Iterate.impl.NewCursor.5aa
// CHECK:STDOUT:   %specific_impl_fn.loc44_33.3 => constants.%T.as_type.as.Iterate.impl.NewCursor.specific_fn.ce3
// CHECK:STDOUT:   %impl.elem1 => constants.%facet_value.591
// CHECK:STDOUT:   %as_type => constants.%tuple.type.df5
// CHECK:STDOUT:   %require_complete.1 => constants.%complete_type.ad6
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type => constants.%Iterate.WithSelf.Next.type.f2f
// CHECK:STDOUT:   %.loc44_33.27 => constants.%.24c
// CHECK:STDOUT:   %impl.elem3.loc44_33.2 => constants.%T.as_type.as.Iterate.impl.Next.79c
// CHECK:STDOUT:   %specific_impl_fn.loc44_33.4 => constants.%T.as_type.as.Iterate.impl.Next.specific_fn.d8c
// CHECK:STDOUT:   %SelfDestruct.specific_fn.6 => constants.%SelfDestruct.specific_fn.80e
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- method_overloads.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %Iterate.type: type = facet_type <@Iterate> [concrete]
// CHECK:STDOUT:   %OptionalStorage.type: type = facet_type <@OptionalStorage> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %facet_type.d61: type = facet_type <@SubobjectDestroy & @DestroyImpl & @Copy> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %T.220: %OptionalStorage.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Optional.Get.type.d97: type = fn_type @Optional.Get, @Optional(%T.220) [symbolic]
// CHECK:STDOUT:   %Optional.Get.3e9: %Optional.Get.type.d97 = struct_value () [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.type.cef: type = fn_type @Optional.HasValue, @Optional(%T.220) [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.a74: %Optional.HasValue.type.cef = struct_value () [symbolic]
// CHECK:STDOUT:   %facet_type.8a3: type = facet_type <@SubobjectDestroy & @DestroyImpl> [concrete]
// CHECK:STDOUT:   %T.9ab: %facet_type.d61 = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.19c: type = facet_access_type %T.9ab [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.lookup_impl_witness.855: <witness> = lookup_impl_witness %T.9ab, @SubobjectDestroy [symbolic]
// CHECK:STDOUT:   %DestroyImpl.lookup_impl_witness.ff9: <witness> = lookup_impl_witness %T.9ab, @DestroyImpl [symbolic]
// CHECK:STDOUT:   %facet_value.b35: %facet_type.8a3 = facet_value %T.as_type.19c, (%SubobjectDestroy.lookup_impl_witness.855, %DestroyImpl.lookup_impl_witness.ff9) [symbolic]
// CHECK:STDOUT:   %MaybeUnformed.413: type = class_type @MaybeUnformed, @MaybeUnformed(%facet_value.b35) [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.a6b: type = class_type @DefaultOptionalStorage, @DefaultOptionalStorage(%T.9ab) [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.elem.94e: type = unbound_element_type %DefaultOptionalStorage.a6b, bool [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.elem.c3f: type = unbound_element_type %DefaultOptionalStorage.a6b, %MaybeUnformed.413 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Get.type.b32: type = fn_type @T.as_type.as.OptionalStorage.impl.Get, @T.as_type.as.OptionalStorage.impl(%T.9ab) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Get.8b0: %T.as_type.as.OptionalStorage.impl.Get.type.b32 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Has.type.8e5: type = fn_type @T.as_type.as.OptionalStorage.impl.Has, @T.as_type.as.OptionalStorage.impl(%T.9ab) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Has.2d5: %T.as_type.as.OptionalStorage.impl.Has.type.8e5 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Some.type.61d: type = fn_type @T.as_type.as.OptionalStorage.impl.Some, @T.as_type.as.OptionalStorage.impl(%T.9ab) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Some.ba7: %T.as_type.as.OptionalStorage.impl.Some.type.61d = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.None.type.a9f: type = fn_type @T.as_type.as.OptionalStorage.impl.None, @T.as_type.as.OptionalStorage.impl(%T.9ab) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.None.083: %T.as_type.as.OptionalStorage.impl.None.type.a9f = struct_value () [symbolic]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.273: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.5b5: %T.as_type.as.DestroyImpl.impl.Op.type.273 = struct_value () [symbolic]
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
// CHECK:STDOUT:   %SubobjectDestroy.lookup_impl_witness.dc9: <witness> = lookup_impl_witness %tuple.type.1ba, @SubobjectDestroy [symbolic]
// CHECK:STDOUT:   %DestroyImpl.lookup_impl_witness.504: <witness> = lookup_impl_witness %tuple.type.1ba, @DestroyImpl [symbolic]
// CHECK:STDOUT:   %facet_value.bf6: %facet_type.8a3 = facet_value %tuple.type.1ba, (%SubobjectDestroy.lookup_impl_witness.dc9, %DestroyImpl.lookup_impl_witness.504) [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.lookup_impl_witness.b15da7.2: <witness> = lookup_impl_witness %impl.elem0.493765.2, @SubobjectDestroy [symbolic]
// CHECK:STDOUT:   %DestroyImpl.lookup_impl_witness.4e7f44.2: <witness> = lookup_impl_witness %impl.elem0.493765.2, @DestroyImpl [symbolic]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.167: <witness> = lookup_impl_witness %impl.elem0.493765.2, @Copy [symbolic]
// CHECK:STDOUT:   %facet_value.9da: %facet_type.d61 = facet_value %impl.elem0.493765.2, (%SubobjectDestroy.lookup_impl_witness.b15da7.2, %DestroyImpl.lookup_impl_witness.4e7f44.2, %Copy.lookup_impl_witness.167) [symbolic]
// CHECK:STDOUT:   %SelfDestruct.type: type = fn_type @SelfDestruct [concrete]
// CHECK:STDOUT:   %SelfDestruct: %SelfDestruct.type = struct_value () [concrete]
// CHECK:STDOUT:   %OverloadValueType: type = class_type @OverloadValueType [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %pattern_type.804: type = pattern_type %OverloadValueType [concrete]
// CHECK:STDOUT:   %OverloadValueType.Op.type.83af6e.1: type = fn_type @OverloadValueType.Op.1 [concrete]
// CHECK:STDOUT:   %OverloadValueType.Op.4dca31.1: %OverloadValueType.Op.type.83af6e.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.5760f4.1: <witness> = custom_witness (%OverloadValueType.Op.4dca31.1), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.ef0: %SubobjectDestroy.type = facet_value %OverloadValueType, (%custom_witness.5760f4.1) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.3a6: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.ef0) [concrete]
// CHECK:STDOUT:   %OverloadValueType.Op.type.83af6e.2: type = fn_type @OverloadValueType.Op.2 [concrete]
// CHECK:STDOUT:   %OverloadValueType.Op.4dca31.2: %OverloadValueType.Op.type.83af6e.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.d0da43.1: <witness> = custom_witness (%OverloadValueType.Op.4dca31.2), @Copy [concrete]
// CHECK:STDOUT:   %facet_value.eb1: %facet_type.d61 = facet_value %OverloadValueType, (%custom_witness.5760f4.1, %DestroyImpl.impl_witness.3a6, %custom_witness.d0da43.1) [concrete]
// CHECK:STDOUT:   %Iterate.lookup_impl_witness.1d8: <witness> = lookup_impl_witness %.Self.500, @Iterate [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.8a4: %facet_type.d61 = impl_witness_access %Iterate.lookup_impl_witness.1d8, element0 [symbolic_self]
// CHECK:STDOUT:   %Iterate_where.type.e27: type = facet_type <@Iterate where %impl.elem0.8a4 = %facet_value.eb1> [concrete]
// CHECK:STDOUT:   %pattern_type.383: type = pattern_type %Iterate_where.type.e27 [concrete]
// CHECK:STDOUT:   %R.patt: %pattern_type.383 = symbolic_binding_pattern R, 0 [symbolic]
// CHECK:STDOUT:   %R: %Iterate_where.type.e27 = symbolic_binding R, 0 [symbolic]
// CHECK:STDOUT:   %R.as_type: type = facet_access_type %R [symbolic]
// CHECK:STDOUT:   %pattern_type.64f: type = pattern_type %R.as_type [symbolic]
// CHECK:STDOUT:   %r.patt.b94: %pattern_type.64f = ref_binding_pattern r [symbolic]
// CHECK:STDOUT:   %r.param_patt.026: %pattern_type.64f = var_param_pattern %r.patt.b94 [symbolic]
// CHECK:STDOUT:   %_.patt.a10: %pattern_type.804 = value_binding_pattern _ [concrete]
// CHECK:STDOUT:   %Iterate.lookup_impl_witness.673: <witness> = lookup_impl_witness %R, @Iterate [symbolic]
// CHECK:STDOUT:   %Iterate.facet.857: %Iterate.type = facet_value %R.as_type, (%Iterate.lookup_impl_witness.673) [symbolic]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.826: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.857) [symbolic]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.7e6: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.857) [symbolic]
// CHECK:STDOUT:   %.a27: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.826, %Iterate.facet.857 [symbolic]
// CHECK:STDOUT:   %impl.elem2.615: %.a27 = impl_witness_access %Iterate.lookup_impl_witness.673, element2 [symbolic]
// CHECK:STDOUT:   %impl.elem1.96a: %facet_type.8a3 = impl_witness_access %Iterate.lookup_impl_witness.673, element1 [symbolic]
// CHECK:STDOUT:   %as_type.b9b: type = facet_access_type %impl.elem1.96a [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.964: <specific function> = specific_impl_function %impl.elem2.615, @Iterate.WithSelf.NewCursor(%Iterate.facet.857) [symbolic]
// CHECK:STDOUT:   %.8d5: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.7e6, %Iterate.facet.857 [symbolic]
// CHECK:STDOUT:   %impl.elem3.9aa: %.8d5 = impl_witness_access %Iterate.lookup_impl_witness.673, element3 [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.eff: type = class_type @DefaultOptionalStorage, @DefaultOptionalStorage(%facet_value.eb1) [concrete]
// CHECK:STDOUT:   %OptionalStorage.impl_witness.1f0: <witness> = impl_witness imports.%OptionalStorage.impl_witness_table.968, @T.as_type.as.OptionalStorage.impl(%facet_value.eb1) [concrete]
// CHECK:STDOUT:   %OptionalStorage.facet.cef: %OptionalStorage.type = facet_value %OverloadValueType, (%OptionalStorage.impl_witness.1f0) [concrete]
// CHECK:STDOUT:   %Optional.06a: type = class_type @Optional, @Optional(%OptionalStorage.facet.cef) [concrete]
// CHECK:STDOUT:   %specific_impl_fn.841: <specific function> = specific_impl_function %impl.elem3.9aa, @Iterate.WithSelf.Next(%Iterate.facet.857) [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.type.b8f: type = fn_type @Optional.HasValue, @Optional(%OptionalStorage.facet.cef) [concrete]
// CHECK:STDOUT:   %Optional.HasValue.ba8: %Optional.HasValue.type.b8f = struct_value () [concrete]
// CHECK:STDOUT:   %Optional.Get.type.e84: type = fn_type @Optional.Get, @Optional(%OptionalStorage.facet.cef) [concrete]
// CHECK:STDOUT:   %Optional.Get.56d: %Optional.Get.type.e84 = struct_value () [concrete]
// CHECK:STDOUT:   %facet_value.ca6: %facet_type.8a3 = facet_value %OverloadValueType, (%custom_witness.5760f4.1, %DestroyImpl.impl_witness.3a6) [concrete]
// CHECK:STDOUT:   %MaybeUnformed.ed8: type = class_type @MaybeUnformed, @MaybeUnformed(%facet_value.ca6) [concrete]
// CHECK:STDOUT:   %.8a2: type = maybe_unformed_type %OverloadValueType [concrete]
// CHECK:STDOUT:   %DefaultOptionalStorage.elem.f53: type = unbound_element_type %DefaultOptionalStorage.eff, %MaybeUnformed.ed8 [concrete]
// CHECK:STDOUT:   %DefaultOptionalStorage.elem.be2: type = unbound_element_type %DefaultOptionalStorage.eff, bool [concrete]
// CHECK:STDOUT:   %struct_type.value.has_value.d40: type = struct_type {.value: %MaybeUnformed.ed8, .has_value: bool} [concrete]
// CHECK:STDOUT:   %Optional.HasValue.specific_fn.cfe: <specific function> = specific_function %Optional.HasValue.ba8, @Optional.HasValue(%OptionalStorage.facet.cef) [concrete]
// CHECK:STDOUT:   %Optional.Get.specific_fn.2f6: <specific function> = specific_function %Optional.Get.56d, @Optional.Get(%OptionalStorage.facet.cef) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.3: type = fn_type @SubobjectDestroy.WithSelf.Op.loc27_37.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.3: %SubobjectDestroy.WithSelf.Op.type.6870fd.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.3: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.3), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.4: type = fn_type @SubobjectDestroy.WithSelf.Op.loc27_37.3 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.4: %SubobjectDestroy.WithSelf.Op.type.6870fd.4 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.4: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.4), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.a87: %SubobjectDestroy.type = facet_value bool, (%custom_witness.b80ef6.4) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.deb: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.a87) [concrete]
// CHECK:STDOUT:   %facet_value.a1b: %facet_type.8a3 = facet_value bool, (%custom_witness.b80ef6.4, %DestroyImpl.impl_witness.deb) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.3a8: %SubobjectDestroy.type = facet_value %MaybeUnformed.ed8, (%custom_witness.b80ef6.3) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.354: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.3a8) [concrete]
// CHECK:STDOUT:   %facet_value.521: %facet_type.8a3 = facet_value %MaybeUnformed.ed8, (%custom_witness.b80ef6.3, %DestroyImpl.impl_witness.354) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.7: type = fn_type @SubobjectDestroy.WithSelf.Op.loc27_37.6 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.7: %SubobjectDestroy.WithSelf.Op.type.6870fd.7 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.7: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.7), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.1f5: %SubobjectDestroy.type = facet_value %Optional.06a, (%custom_witness.b80ef6.7) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.601: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.1f5) [concrete]
// CHECK:STDOUT:   %facet_value.4a3: %facet_type.8a3 = facet_value %Optional.06a, (%custom_witness.b80ef6.7, %DestroyImpl.impl_witness.601) [concrete]
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
// CHECK:STDOUT:   %custom_witness.51e: <witness> = custom_witness (%Iterator.Op.e8caec.2), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.ea6: %SubobjectDestroy.type = facet_value %Iterator, (%custom_witness.51e) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.903: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.ea6) [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.4232e0.3: type = fn_type @Iterator.Op.3 [concrete]
// CHECK:STDOUT:   %Iterator.Op.e8caec.3: %Iterator.Op.type.4232e0.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.716: <witness> = custom_witness (%Iterator.Op.e8caec.3), @Inc [concrete]
// CHECK:STDOUT:   %Equal.type: type = fn_type @Equal [concrete]
// CHECK:STDOUT:   %Equal: %Equal.type = struct_value () [concrete]
// CHECK:STDOUT:   %NotEqual.type: type = fn_type @NotEqual [concrete]
// CHECK:STDOUT:   %NotEqual: %NotEqual.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.617: <witness> = custom_witness (%Equal, %NotEqual), @EqWith, @EqWith(%Iterator) [concrete]
// CHECK:STDOUT:   %facet_value.f25: %CppRange_where.type.d682d6.1 = facet_value %Range, (%custom_witness.5760f4.1, %DestroyImpl.impl_witness.3a6, %custom_witness.d0da43.1, %custom_witness.090, %custom_witness.51e, %DestroyImpl.impl_witness.903, %custom_witness.6bd, %custom_witness.716, %custom_witness.617) [concrete]
// CHECK:STDOUT:   %tuple.type.9ef: type = tuple_type (%Iterator, %Iterator) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.8: type = fn_type @SubobjectDestroy.WithSelf.Op.3 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.8: %SubobjectDestroy.WithSelf.Op.type.6870fd.8 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.8: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.8), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.43f: %SubobjectDestroy.type = facet_value %tuple.type.9ef, (%custom_witness.b80ef6.8) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.d8b: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.43f) [concrete]
// CHECK:STDOUT:   %complete_type.e35: <witness> = complete_type_witness %tuple.type.9ef [concrete]
// CHECK:STDOUT:   %facet_value.b15: %facet_type.8a3 = facet_value %tuple.type.9ef, (%custom_witness.b80ef6.8, %DestroyImpl.impl_witness.d8b) [concrete]
// CHECK:STDOUT:   %Iterate.impl_witness.5b6: <witness> = impl_witness imports.%Iterate.impl_witness_table.121, @T.as_type.as.Iterate.impl(%facet_value.f25) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.type.1e9: type = fn_type @T.as_type.as.Iterate.impl.NewCursor, @T.as_type.as.Iterate.impl(%facet_value.f25) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.bb9: %T.as_type.as.Iterate.impl.NewCursor.type.1e9 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.type.4c1: type = fn_type @T.as_type.as.Iterate.impl.Next, @T.as_type.as.Iterate.impl(%facet_value.f25) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.e07: %T.as_type.as.Iterate.impl.Next.type.4c1 = struct_value () [concrete]
// CHECK:STDOUT:   %Iterate.facet.12c: %Iterate.type = facet_value %Range, (%Iterate.impl_witness.5b6) [concrete]
// CHECK:STDOUT:   %facet_value.1fb: %Iterate_where.type.e27 = facet_value %Range, (%Iterate.impl_witness.5b6) [concrete]
// CHECK:STDOUT:   %r.patt.b9f: %pattern_type.23f = ref_binding_pattern r [concrete]
// CHECK:STDOUT:   %r.param_patt.466: %pattern_type.23f = var_param_pattern %r.patt.b9f [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.5cf: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.12c) [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.f29: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.12c) [concrete]
// CHECK:STDOUT:   %.19f: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.5cf, %Iterate.facet.12c [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.specific_fn: <specific function> = specific_function %T.as_type.as.Iterate.impl.NewCursor.bb9, @T.as_type.as.Iterate.impl.NewCursor(%facet_value.f25) [concrete]
// CHECK:STDOUT:   %.2ed: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.f29, %Iterate.facet.12c [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.specific_fn: <specific function> = specific_function %T.as_type.as.Iterate.impl.Next.e07, @T.as_type.as.Iterate.impl.Next(%facet_value.f25) [concrete]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.e65: <specific function> = specific_function %SelfDestruct, @SelfDestruct(%facet_value.b15) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .OverloadValueType = %OverloadValueType.decl
// CHECK:STDOUT:     .Range = %Range.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.4d4: @Optional.%Optional.HasValue.type (%Optional.HasValue.type.cef) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.HasValue (constants.%Optional.HasValue.a74)]
// CHECK:STDOUT:   %Core.import_ref.059: @Optional.%Optional.Get.type (%Optional.Get.type.d97) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.Get (constants.%Optional.Get.3e9)]
// CHECK:STDOUT:   %Core.import_ref.7bb: type = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%DefaultOptionalStorage (constants.%DefaultOptionalStorage.a6b)]
// CHECK:STDOUT:   %Core.import_ref.61e: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.None.type (%T.as_type.as.OptionalStorage.impl.None.type.a9f) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.None (constants.%T.as_type.as.OptionalStorage.impl.None.083)]
// CHECK:STDOUT:   %Core.import_ref.ca1: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Some.type (%T.as_type.as.OptionalStorage.impl.Some.type.61d) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Some (constants.%T.as_type.as.OptionalStorage.impl.Some.ba7)]
// CHECK:STDOUT:   %Core.import_ref.6ba4: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Has.type (%T.as_type.as.OptionalStorage.impl.Has.type.8e5) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Has (constants.%T.as_type.as.OptionalStorage.impl.Has.2d5)]
// CHECK:STDOUT:   %Core.import_ref.73d: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Get.type (%T.as_type.as.OptionalStorage.impl.Get.type.b32) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Get (constants.%T.as_type.as.OptionalStorage.impl.Get.8b0)]
// CHECK:STDOUT:   %Core.import_ref.778 = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], unloaded
// CHECK:STDOUT:   %OptionalStorage.impl_witness_table.968 = impl_witness_table (%Core.import_ref.7bb, %Core.import_ref.61e, %Core.import_ref.ca1, %Core.import_ref.6ba4, %Core.import_ref.73d, %Core.import_ref.778), @T.as_type.as.OptionalStorage.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.146b: @DefaultOptionalStorage.%DefaultOptionalStorage.elem.1 (%DefaultOptionalStorage.elem.c3f) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [concrete = %field_decl.31e]
// CHECK:STDOUT:   %Core.import_ref.f1da: @DefaultOptionalStorage.%DefaultOptionalStorage.elem.2 (%DefaultOptionalStorage.elem.94e) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [concrete = %field_decl.431]
// CHECK:STDOUT:   %Core.import_ref.14b: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.273) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.5b5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.14b), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.d6c: %facet_type.d61 = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.Iterate.impl.%facet_value.1 (constants.%facet_value.9da)]
// CHECK:STDOUT:   %Core.import_ref.2ac: %facet_type.8a3 = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.Iterate.impl.%facet_value.2 (constants.%facet_value.bf6)]
// CHECK:STDOUT:   %Core.import_ref.fd1: @T.as_type.as.Iterate.impl.%T.as_type.as.Iterate.impl.NewCursor.type (%T.as_type.as.Iterate.impl.NewCursor.type.953) = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.Iterate.impl.%T.as_type.as.Iterate.impl.NewCursor (constants.%T.as_type.as.Iterate.impl.NewCursor.544)]
// CHECK:STDOUT:   %Core.import_ref.273: @T.as_type.as.Iterate.impl.%T.as_type.as.Iterate.impl.Next.type (%T.as_type.as.Iterate.impl.Next.type.b36) = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.Iterate.impl.%T.as_type.as.Iterate.impl.Next (constants.%T.as_type.as.Iterate.impl.Next.d6b)]
// CHECK:STDOUT:   %Iterate.impl_witness_table.121 = impl_witness_table (%Core.import_ref.d6c, %Core.import_ref.2ac, %Core.import_ref.fd1, %Core.import_ref.273), @T.as_type.as.Iterate.impl [concrete]
// CHECK:STDOUT:   %OverloadValueType.decl: type = class_decl @OverloadValueType [concrete = constants.%OverloadValueType] {} {}
// CHECK:STDOUT:   %Core.import_ref.b0d = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %field_decl.431: @DefaultOptionalStorage.%DefaultOptionalStorage.elem.2 (%DefaultOptionalStorage.elem.94e) = field_decl has_value, element1, %Core.import_ref.b0d in [concrete] {}
// CHECK:STDOUT:   %Core.import_ref.164c = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %field_decl.31e: @DefaultOptionalStorage.%DefaultOptionalStorage.elem.1 (%DefaultOptionalStorage.elem.c3f) = field_decl value, element0, %Core.import_ref.164c in [concrete] {}
// CHECK:STDOUT:   %Range.decl: type = class_decl @Range [concrete = constants.%Range] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @TestIterate(%R.loc25_17.2: %Iterate_where.type.e27) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Iterate.lookup_impl_witness: <witness> = lookup_impl_witness %R.loc25_17.1, @Iterate [symbolic = %Iterate.lookup_impl_witness (constants.%Iterate.lookup_impl_witness.673)]
// CHECK:STDOUT:   %Iterate.facet.loc27_37.5: %Iterate.type = facet_value %R.as_type.loc25_83.1, (%Iterate.lookup_impl_witness) [symbolic = %Iterate.facet.loc27_37.5 (constants.%Iterate.facet.857)]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.loc27_37.5) [symbolic = %Iterate.WithSelf.NewCursor.type (constants.%Iterate.WithSelf.NewCursor.type.826)]
// CHECK:STDOUT:   %.loc27_37.26: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type, %Iterate.facet.loc27_37.5 [symbolic = %.loc27_37.26 (constants.%.a27)]
// CHECK:STDOUT:   %impl.elem2.loc27_37.2: @TestIterate.%.loc27_37.26 (%.a27) = impl_witness_access %Iterate.lookup_impl_witness, element2 [symbolic = %impl.elem2.loc27_37.2 (constants.%impl.elem2.615)]
// CHECK:STDOUT:   %specific_impl_fn.loc27_37.3: <specific function> = specific_impl_function %impl.elem2.loc27_37.2, @Iterate.WithSelf.NewCursor(%Iterate.facet.loc27_37.5) [symbolic = %specific_impl_fn.loc27_37.3 (constants.%specific_impl_fn.964)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.loc27_37.5) [symbolic = %Iterate.WithSelf.Next.type (constants.%Iterate.WithSelf.Next.type.7e6)]
// CHECK:STDOUT:   %.loc27_37.27: type = fn_type_with_self_type %Iterate.WithSelf.Next.type, %Iterate.facet.loc27_37.5 [symbolic = %.loc27_37.27 (constants.%.8d5)]
// CHECK:STDOUT:   %impl.elem3.loc27_37.2: @TestIterate.%.loc27_37.27 (%.8d5) = impl_witness_access %Iterate.lookup_impl_witness, element3 [symbolic = %impl.elem3.loc27_37.2 (constants.%impl.elem3.9aa)]
// CHECK:STDOUT:   %specific_impl_fn.loc27_37.4: <specific function> = specific_impl_function %impl.elem3.loc27_37.2, @Iterate.WithSelf.Next(%Iterate.facet.loc27_37.5) [symbolic = %specific_impl_fn.loc27_37.4 (constants.%specific_impl_fn.841)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%r.param: ref @TestIterate.%R.as_type.loc25_83.1 (%R.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %_.patt: %pattern_type.804 = value_binding_pattern _ [concrete = constants.%_.patt.a10]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %r.ref: ref @TestIterate.%R.as_type.loc25_83.1 (%R.as_type) = name_ref r, %r
// CHECK:STDOUT:     %impl.elem2.loc27_37.1: @TestIterate.%.loc27_37.26 (%.a27) = impl_witness_access constants.%Iterate.lookup_impl_witness.673, element2 [symbolic = %impl.elem2.loc27_37.2 (constants.%impl.elem2.615)]
// CHECK:STDOUT:     %bound_method.loc27_37.1: <bound method> = bound_method %r.ref, %impl.elem2.loc27_37.1
// CHECK:STDOUT:     %Iterate.facet.loc27_37.1: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.673) [symbolic = %Iterate.facet.loc27_37.5 (constants.%Iterate.facet.857)]
// CHECK:STDOUT:     %.loc27_37.1: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc27_37.1 [symbolic = %Iterate.facet.loc27_37.5 (constants.%Iterate.facet.857)]
// CHECK:STDOUT:     %Iterate.facet.loc27_37.2: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.673) [symbolic = %Iterate.facet.loc27_37.5 (constants.%Iterate.facet.857)]
// CHECK:STDOUT:     %.loc27_37.2: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc27_37.2 [symbolic = %Iterate.facet.loc27_37.5 (constants.%Iterate.facet.857)]
// CHECK:STDOUT:     %specific_impl_fn.loc27_37.1: <specific function> = specific_impl_function %impl.elem2.loc27_37.1, @Iterate.WithSelf.NewCursor(constants.%Iterate.facet.857) [symbolic = %specific_impl_fn.loc27_37.3 (constants.%specific_impl_fn.964)]
// CHECK:STDOUT:     %bound_method.loc27_37.2: <bound method> = bound_method %r.ref, %specific_impl_fn.loc27_37.1
// CHECK:STDOUT:     %var: ref @TestIterate.%as_type (%as_type.b9b) = var_storage invalid
// CHECK:STDOUT:     %.loc27_36.1: @TestIterate.%R.as_type.loc25_83.1 (%R.as_type) = acquire_value %r.ref
// CHECK:STDOUT:     %Iterate.WithSelf.NewCursor.call: init @TestIterate.%as_type (%as_type.b9b) to %var = call %bound_method.loc27_37.2(%.loc27_36.1)
// CHECK:STDOUT:     assign %var, %Iterate.WithSelf.NewCursor.call
// CHECK:STDOUT:     br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.next:
// CHECK:STDOUT:     %impl.elem3.loc27_37.1: @TestIterate.%.loc27_37.27 (%.8d5) = impl_witness_access constants.%Iterate.lookup_impl_witness.673, element3 [symbolic = %impl.elem3.loc27_37.2 (constants.%impl.elem3.9aa)]
// CHECK:STDOUT:     %bound_method.loc27_37.3: <bound method> = bound_method %r.ref, %impl.elem3.loc27_37.1
// CHECK:STDOUT:     %Iterate.facet.loc27_37.3: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.673) [symbolic = %Iterate.facet.loc27_37.5 (constants.%Iterate.facet.857)]
// CHECK:STDOUT:     %.loc27_37.3: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc27_37.3 [symbolic = %Iterate.facet.loc27_37.5 (constants.%Iterate.facet.857)]
// CHECK:STDOUT:     %Iterate.facet.loc27_37.4: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.673) [symbolic = %Iterate.facet.loc27_37.5 (constants.%Iterate.facet.857)]
// CHECK:STDOUT:     %.loc27_37.4: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc27_37.4 [symbolic = %Iterate.facet.loc27_37.5 (constants.%Iterate.facet.857)]
// CHECK:STDOUT:     %.loc27_37.5: %facet_type.8a3 = converted constants.%as_type.b9b, constants.%impl.elem1.96a [symbolic = %impl.elem1 (constants.%impl.elem1.96a)]
// CHECK:STDOUT:     %.loc27_37.6: %facet_type.8a3 = converted constants.%as_type.b9b, constants.%impl.elem1.96a [symbolic = %impl.elem1 (constants.%impl.elem1.96a)]
// CHECK:STDOUT:     %specific_impl_fn.loc27_37.2: <specific function> = specific_impl_function %impl.elem3.loc27_37.1, @Iterate.WithSelf.Next(constants.%Iterate.facet.857) [symbolic = %specific_impl_fn.loc27_37.4 (constants.%specific_impl_fn.841)]
// CHECK:STDOUT:     %bound_method.loc27_37.4: <bound method> = bound_method %r.ref, %specific_impl_fn.loc27_37.2
// CHECK:STDOUT:     %.loc27_37.7: ref %Optional.06a = temporary_storage
// CHECK:STDOUT:     %.loc27_36.2: @TestIterate.%R.as_type.loc25_83.1 (%R.as_type) = acquire_value %r.ref
// CHECK:STDOUT:     %Iterate.WithSelf.Next.call: init %Optional.06a to %.loc27_37.7 = call %bound_method.loc27_37.4(%.loc27_36.2, %var)
// CHECK:STDOUT:     %.loc27_37.8: ref %Optional.06a = temporary %.loc27_37.7, %Iterate.WithSelf.Next.call
// CHECK:STDOUT:     %.loc27_37.9: %Optional.HasValue.type.b8f = specific_constant imports.%Core.import_ref.4d4, @Optional(constants.%OptionalStorage.facet.cef) [concrete = constants.%Optional.HasValue.ba8]
// CHECK:STDOUT:     %HasValue.ref: %Optional.HasValue.type.b8f = name_ref HasValue, %.loc27_37.9 [concrete = constants.%Optional.HasValue.ba8]
// CHECK:STDOUT:     %Optional.HasValue.bound: <bound method> = bound_method %.loc27_37.8, %HasValue.ref
// CHECK:STDOUT:     %Optional.HasValue.specific_fn: <specific function> = specific_function %HasValue.ref, @Optional.HasValue(constants.%OptionalStorage.facet.cef) [concrete = constants.%Optional.HasValue.specific_fn.cfe]
// CHECK:STDOUT:     %bound_method.loc27_37.5: <bound method> = bound_method %.loc27_37.8, %Optional.HasValue.specific_fn
// CHECK:STDOUT:     %.loc27_37.10: %Optional.06a = acquire_value %.loc27_37.8
// CHECK:STDOUT:     %Optional.HasValue.call: init bool = call %bound_method.loc27_37.5(%.loc27_37.10)
// CHECK:STDOUT:     %.loc27_37.11: bool = value_of_initializer %Optional.HasValue.call
// CHECK:STDOUT:     %.loc27_37.12: bool = converted %Optional.HasValue.call, %.loc27_37.11
// CHECK:STDOUT:     if %.loc27_37.12 br !for.body else br !for.done
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.body:
// CHECK:STDOUT:     %.loc27_37.13: %Optional.Get.type.e84 = specific_constant imports.%Core.import_ref.059, @Optional(constants.%OptionalStorage.facet.cef) [concrete = constants.%Optional.Get.56d]
// CHECK:STDOUT:     %Get.ref: %Optional.Get.type.e84 = name_ref Get, %.loc27_37.13 [concrete = constants.%Optional.Get.56d]
// CHECK:STDOUT:     %Optional.Get.bound: <bound method> = bound_method %.loc27_37.8, %Get.ref
// CHECK:STDOUT:     %Optional.Get.specific_fn: <specific function> = specific_function %Get.ref, @Optional.Get(constants.%OptionalStorage.facet.cef) [concrete = constants.%Optional.Get.specific_fn.2f6]
// CHECK:STDOUT:     %bound_method.loc27_37.6: <bound method> = bound_method %.loc27_37.8, %Optional.Get.specific_fn
// CHECK:STDOUT:     %.loc27_37.14: ref %OverloadValueType = temporary_storage
// CHECK:STDOUT:     %.loc27_37.15: %Optional.06a = acquire_value %.loc27_37.8
// CHECK:STDOUT:     %Optional.Get.call: init %OverloadValueType to %.loc27_37.14 = call %bound_method.loc27_37.6(%.loc27_37.15)
// CHECK:STDOUT:     %.loc27_37.16: ref %OverloadValueType = temporary %.loc27_37.14, %Optional.Get.call
// CHECK:STDOUT:     %.loc27_37.17: %OverloadValueType = acquire_value %.loc27_37.16
// CHECK:STDOUT:     %.loc27_14: type = splice_block %OverloadValueType.ref.loc27 [concrete = constants.%OverloadValueType] {
// CHECK:STDOUT:       %Cpp.ref.loc27: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:       %OverloadValueType.ref.loc27: type = name_ref OverloadValueType, imports.%OverloadValueType.decl [concrete = constants.%OverloadValueType]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %_: %OverloadValueType = wrapper_binding _, %.loc27_37.17
// CHECK:STDOUT:     %facet_value.loc27_37.1: %facet_type.8a3 = facet_value constants.%OverloadValueType, (constants.%custom_witness.5760f4.1, constants.%DestroyImpl.impl_witness.3a6) [concrete = constants.%facet_value.ca6]
// CHECK:STDOUT:     %.loc27_37.18: %facet_type.8a3 = converted constants.%OverloadValueType, %facet_value.loc27_37.1 [concrete = constants.%facet_value.ca6]
// CHECK:STDOUT:     %facet_value.loc27_37.2: %facet_type.8a3 = facet_value constants.%OverloadValueType, (constants.%custom_witness.5760f4.1, constants.%DestroyImpl.impl_witness.3a6) [concrete = constants.%facet_value.ca6]
// CHECK:STDOUT:     %.loc27_37.19: %facet_type.8a3 = converted constants.%OverloadValueType, %facet_value.loc27_37.2 [concrete = constants.%facet_value.ca6]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %SelfDestruct.call.loc27_37.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc27_37.16)
// CHECK:STDOUT:     %facet_value.loc27_37.3: %facet_type.8a3 = facet_value constants.%Optional.06a, (constants.%custom_witness.b80ef6.7, constants.%DestroyImpl.impl_witness.601) [concrete = constants.%facet_value.4a3]
// CHECK:STDOUT:     %.loc27_37.20: %facet_type.8a3 = converted constants.%Optional.06a, %facet_value.loc27_37.3 [concrete = constants.%facet_value.4a3]
// CHECK:STDOUT:     %facet_value.loc27_37.4: %facet_type.8a3 = facet_value constants.%Optional.06a, (constants.%custom_witness.b80ef6.7, constants.%DestroyImpl.impl_witness.601) [concrete = constants.%facet_value.4a3]
// CHECK:STDOUT:     %.loc27_37.21: %facet_type.8a3 = converted constants.%Optional.06a, %facet_value.loc27_37.4 [concrete = constants.%facet_value.4a3]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %SelfDestruct.call.loc27_37.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc27_37.8)
// CHECK:STDOUT:     br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.done:
// CHECK:STDOUT:     %facet_value.loc27_37.5: %facet_type.8a3 = facet_value constants.%Optional.06a, (constants.%custom_witness.b80ef6.7, constants.%DestroyImpl.impl_witness.601) [concrete = constants.%facet_value.4a3]
// CHECK:STDOUT:     %.loc27_37.22: %facet_type.8a3 = converted constants.%Optional.06a, %facet_value.loc27_37.5 [concrete = constants.%facet_value.4a3]
// CHECK:STDOUT:     %facet_value.loc27_37.6: %facet_type.8a3 = facet_value constants.%Optional.06a, (constants.%custom_witness.b80ef6.7, constants.%DestroyImpl.impl_witness.601) [concrete = constants.%facet_value.4a3]
// CHECK:STDOUT:     %.loc27_37.23: %facet_type.8a3 = converted constants.%Optional.06a, %facet_value.loc27_37.6 [concrete = constants.%facet_value.4a3]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %SelfDestruct.call.loc27_37.3: init %empty_tuple.type = call %SelfDestruct.specific_fn.3(%.loc27_37.8)
// CHECK:STDOUT:     %.loc27_37.24: %facet_type.8a3 = converted constants.%as_type.b9b, constants.%impl.elem1.96a [symbolic = %impl.elem1 (constants.%impl.elem1.96a)]
// CHECK:STDOUT:     %.loc27_37.25: %facet_type.8a3 = converted constants.%as_type.b9b, constants.%impl.elem1.96a [symbolic = %impl.elem1 (constants.%impl.elem1.96a)]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %SelfDestruct.call.loc27_37.4: init %empty_tuple.type = call %SelfDestruct.specific_fn.4(%var)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc27_37.1(%self.param: ref %.8a2) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc27_37.2(%self.param: ref %MaybeUnformed.ed8) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc27_37.3(%self.param: ref bool) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc27_37.4(%self.param: ref %struct_type.value.has_value.d40) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc27_37.1: ref bool = struct_access %self.param, element1
// CHECK:STDOUT:   %facet_value.loc27_37.1: %facet_type.8a3 = facet_value bool, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.deb) [concrete = constants.%facet_value.a1b]
// CHECK:STDOUT:   %.loc27_37.2: %facet_type.8a3 = converted bool, %facet_value.loc27_37.1 [concrete = constants.%facet_value.a1b]
// CHECK:STDOUT:   %facet_value.loc27_37.2: %facet_type.8a3 = facet_value bool, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.deb) [concrete = constants.%facet_value.a1b]
// CHECK:STDOUT:   %.loc27_37.3: %facet_type.8a3 = converted bool, %facet_value.loc27_37.2 [concrete = constants.%facet_value.a1b]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc27_37.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc27_37.1)
// CHECK:STDOUT:   %.loc27_37.4: ref %MaybeUnformed.ed8 = struct_access %self.param, element0
// CHECK:STDOUT:   %facet_value.loc27_37.3: %facet_type.8a3 = facet_value constants.%MaybeUnformed.ed8, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.354) [concrete = constants.%facet_value.521]
// CHECK:STDOUT:   %.loc27_37.5: %facet_type.8a3 = converted constants.%MaybeUnformed.ed8, %facet_value.loc27_37.3 [concrete = constants.%facet_value.521]
// CHECK:STDOUT:   %facet_value.loc27_37.4: %facet_type.8a3 = facet_value constants.%MaybeUnformed.ed8, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.354) [concrete = constants.%facet_value.521]
// CHECK:STDOUT:   %.loc27_37.6: %facet_type.8a3 = converted constants.%MaybeUnformed.ed8, %facet_value.loc27_37.4 [concrete = constants.%facet_value.521]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc27_37.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc27_37.4)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc27_37.5(%self.param: ref %DefaultOptionalStorage.eff) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %has_value.ref: %DefaultOptionalStorage.elem.be2 = name_ref has_value, imports.%Core.import_ref.f1da [concrete = imports.%field_decl.431]
// CHECK:STDOUT:   %.loc27_37.1: ref bool = class_element_access %self.param, element1
// CHECK:STDOUT:   %facet_value.loc27_37.1: %facet_type.8a3 = facet_value bool, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.deb) [concrete = constants.%facet_value.a1b]
// CHECK:STDOUT:   %.loc27_37.2: %facet_type.8a3 = converted bool, %facet_value.loc27_37.1 [concrete = constants.%facet_value.a1b]
// CHECK:STDOUT:   %facet_value.loc27_37.2: %facet_type.8a3 = facet_value bool, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.deb) [concrete = constants.%facet_value.a1b]
// CHECK:STDOUT:   %.loc27_37.3: %facet_type.8a3 = converted bool, %facet_value.loc27_37.2 [concrete = constants.%facet_value.a1b]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc27_37.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc27_37.1)
// CHECK:STDOUT:   %value.ref: %DefaultOptionalStorage.elem.f53 = name_ref value, imports.%Core.import_ref.146b [concrete = imports.%field_decl.31e]
// CHECK:STDOUT:   %.loc27_37.4: ref %MaybeUnformed.ed8 = class_element_access %self.param, element0
// CHECK:STDOUT:   %facet_value.loc27_37.3: %facet_type.8a3 = facet_value constants.%MaybeUnformed.ed8, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.354) [concrete = constants.%facet_value.521]
// CHECK:STDOUT:   %.loc27_37.5: %facet_type.8a3 = converted constants.%MaybeUnformed.ed8, %facet_value.loc27_37.3 [concrete = constants.%facet_value.521]
// CHECK:STDOUT:   %facet_value.loc27_37.4: %facet_type.8a3 = facet_value constants.%MaybeUnformed.ed8, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.354) [concrete = constants.%facet_value.521]
// CHECK:STDOUT:   %.loc27_37.6: %facet_type.8a3 = converted constants.%MaybeUnformed.ed8, %facet_value.loc27_37.4 [concrete = constants.%facet_value.521]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc27_37.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc27_37.4)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc27_37.6(%self.param: ref %Optional.06a) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestRangeFor(%r.param: ref %Range) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %_.patt: %pattern_type.804 = value_binding_pattern _ [concrete = constants.%_.patt.a10]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %r.ref: ref %Range = name_ref r, %r
// CHECK:STDOUT:   %impl.elem2: %.19f = impl_witness_access constants.%Iterate.impl_witness.5b6, element2 [concrete = constants.%T.as_type.as.Iterate.impl.NewCursor.bb9]
// CHECK:STDOUT:   %bound_method.loc37_37.1: <bound method> = bound_method %r.ref, %impl.elem2
// CHECK:STDOUT:   %facet_value.loc37_37.1: %CppRange_where.type.d682d6.1 = facet_value constants.%Range, (constants.%custom_witness.5760f4.1, constants.%DestroyImpl.impl_witness.3a6, constants.%custom_witness.d0da43.1, constants.%custom_witness.090, constants.%custom_witness.51e, constants.%DestroyImpl.impl_witness.903, constants.%custom_witness.6bd, constants.%custom_witness.716, constants.%custom_witness.617) [concrete = constants.%facet_value.f25]
// CHECK:STDOUT:   %.loc37_37.1: %CppRange_where.type.d682d6.1 = converted constants.%Range, %facet_value.loc37_37.1 [concrete = constants.%facet_value.f25]
// CHECK:STDOUT:   %facet_value.loc37_37.2: %CppRange_where.type.d682d6.1 = facet_value constants.%Range, (constants.%custom_witness.5760f4.1, constants.%DestroyImpl.impl_witness.3a6, constants.%custom_witness.d0da43.1, constants.%custom_witness.090, constants.%custom_witness.51e, constants.%DestroyImpl.impl_witness.903, constants.%custom_witness.6bd, constants.%custom_witness.716, constants.%custom_witness.617) [concrete = constants.%facet_value.f25]
// CHECK:STDOUT:   %.loc37_37.2: %CppRange_where.type.d682d6.1 = converted constants.%Range, %facet_value.loc37_37.2 [concrete = constants.%facet_value.f25]
// CHECK:STDOUT:   %specific_fn.loc37_37.1: <specific function> = specific_function %impl.elem2, @T.as_type.as.Iterate.impl.NewCursor(constants.%facet_value.f25) [concrete = constants.%T.as_type.as.Iterate.impl.NewCursor.specific_fn]
// CHECK:STDOUT:   %bound_method.loc37_37.2: <bound method> = bound_method %r.ref, %specific_fn.loc37_37.1
// CHECK:STDOUT:   %var: ref %tuple.type.9ef = var_storage invalid
// CHECK:STDOUT:   %.loc37_36.1: %Range = acquire_value %r.ref
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.call: init %tuple.type.9ef to %var = call %bound_method.loc37_37.2(%.loc37_36.1)
// CHECK:STDOUT:   assign %var, %T.as_type.as.Iterate.impl.NewCursor.call
// CHECK:STDOUT:   br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT: !for.next:
// CHECK:STDOUT:   %impl.elem3: %.2ed = impl_witness_access constants.%Iterate.impl_witness.5b6, element3 [concrete = constants.%T.as_type.as.Iterate.impl.Next.e07]
// CHECK:STDOUT:   %bound_method.loc37_37.3: <bound method> = bound_method %r.ref, %impl.elem3
// CHECK:STDOUT:   %facet_value.loc37_37.3: %CppRange_where.type.d682d6.1 = facet_value constants.%Range, (constants.%custom_witness.5760f4.1, constants.%DestroyImpl.impl_witness.3a6, constants.%custom_witness.d0da43.1, constants.%custom_witness.090, constants.%custom_witness.51e, constants.%DestroyImpl.impl_witness.903, constants.%custom_witness.6bd, constants.%custom_witness.716, constants.%custom_witness.617) [concrete = constants.%facet_value.f25]
// CHECK:STDOUT:   %.loc37_37.3: %CppRange_where.type.d682d6.1 = converted constants.%Range, %facet_value.loc37_37.3 [concrete = constants.%facet_value.f25]
// CHECK:STDOUT:   %facet_value.loc37_37.4: %CppRange_where.type.d682d6.1 = facet_value constants.%Range, (constants.%custom_witness.5760f4.1, constants.%DestroyImpl.impl_witness.3a6, constants.%custom_witness.d0da43.1, constants.%custom_witness.090, constants.%custom_witness.51e, constants.%DestroyImpl.impl_witness.903, constants.%custom_witness.6bd, constants.%custom_witness.716, constants.%custom_witness.617) [concrete = constants.%facet_value.f25]
// CHECK:STDOUT:   %.loc37_37.4: %CppRange_where.type.d682d6.1 = converted constants.%Range, %facet_value.loc37_37.4 [concrete = constants.%facet_value.f25]
// CHECK:STDOUT:   %specific_fn.loc37_37.2: <specific function> = specific_function %impl.elem3, @T.as_type.as.Iterate.impl.Next(constants.%facet_value.f25) [concrete = constants.%T.as_type.as.Iterate.impl.Next.specific_fn]
// CHECK:STDOUT:   %bound_method.loc37_37.4: <bound method> = bound_method %r.ref, %specific_fn.loc37_37.2
// CHECK:STDOUT:   %.loc37_37.5: ref %Optional.06a = temporary_storage
// CHECK:STDOUT:   %.loc37_36.2: %Range = acquire_value %r.ref
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.call: init %Optional.06a to %.loc37_37.5 = call %bound_method.loc37_37.4(%.loc37_36.2, %var)
// CHECK:STDOUT:   %.loc37_37.6: ref %Optional.06a = temporary %.loc37_37.5, %T.as_type.as.Iterate.impl.Next.call
// CHECK:STDOUT:   %.loc37_37.7: %Optional.HasValue.type.b8f = specific_constant imports.%Core.import_ref.4d4, @Optional(constants.%OptionalStorage.facet.cef) [concrete = constants.%Optional.HasValue.ba8]
// CHECK:STDOUT:   %HasValue.ref: %Optional.HasValue.type.b8f = name_ref HasValue, %.loc37_37.7 [concrete = constants.%Optional.HasValue.ba8]
// CHECK:STDOUT:   %Optional.HasValue.bound: <bound method> = bound_method %.loc37_37.6, %HasValue.ref
// CHECK:STDOUT:   %Optional.HasValue.specific_fn: <specific function> = specific_function %HasValue.ref, @Optional.HasValue(constants.%OptionalStorage.facet.cef) [concrete = constants.%Optional.HasValue.specific_fn.cfe]
// CHECK:STDOUT:   %bound_method.loc37_37.5: <bound method> = bound_method %.loc37_37.6, %Optional.HasValue.specific_fn
// CHECK:STDOUT:   %.loc37_37.8: %Optional.06a = acquire_value %.loc37_37.6
// CHECK:STDOUT:   %Optional.HasValue.call: init bool = call %bound_method.loc37_37.5(%.loc37_37.8)
// CHECK:STDOUT:   %.loc37_37.9: bool = value_of_initializer %Optional.HasValue.call
// CHECK:STDOUT:   %.loc37_37.10: bool = converted %Optional.HasValue.call, %.loc37_37.9
// CHECK:STDOUT:   if %.loc37_37.10 br !for.body else br !for.done
// CHECK:STDOUT:
// CHECK:STDOUT: !for.body:
// CHECK:STDOUT:   %.loc37_37.11: %Optional.Get.type.e84 = specific_constant imports.%Core.import_ref.059, @Optional(constants.%OptionalStorage.facet.cef) [concrete = constants.%Optional.Get.56d]
// CHECK:STDOUT:   %Get.ref: %Optional.Get.type.e84 = name_ref Get, %.loc37_37.11 [concrete = constants.%Optional.Get.56d]
// CHECK:STDOUT:   %Optional.Get.bound: <bound method> = bound_method %.loc37_37.6, %Get.ref
// CHECK:STDOUT:   %Optional.Get.specific_fn: <specific function> = specific_function %Get.ref, @Optional.Get(constants.%OptionalStorage.facet.cef) [concrete = constants.%Optional.Get.specific_fn.2f6]
// CHECK:STDOUT:   %bound_method.loc37_37.6: <bound method> = bound_method %.loc37_37.6, %Optional.Get.specific_fn
// CHECK:STDOUT:   %.loc37_37.12: ref %OverloadValueType = temporary_storage
// CHECK:STDOUT:   %.loc37_37.13: %Optional.06a = acquire_value %.loc37_37.6
// CHECK:STDOUT:   %Optional.Get.call: init %OverloadValueType to %.loc37_37.12 = call %bound_method.loc37_37.6(%.loc37_37.13)
// CHECK:STDOUT:   %.loc37_37.14: ref %OverloadValueType = temporary %.loc37_37.12, %Optional.Get.call
// CHECK:STDOUT:   %.loc37_37.15: %OverloadValueType = acquire_value %.loc37_37.14
// CHECK:STDOUT:   %.loc37_14: type = splice_block %OverloadValueType.ref [concrete = constants.%OverloadValueType] {
// CHECK:STDOUT:     %Cpp.ref.loc37: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %OverloadValueType.ref: type = name_ref OverloadValueType, imports.%OverloadValueType.decl [concrete = constants.%OverloadValueType]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %_: %OverloadValueType = wrapper_binding _, %.loc37_37.15
// CHECK:STDOUT:   %facet_value.loc37_37.5: %facet_type.8a3 = facet_value constants.%OverloadValueType, (constants.%custom_witness.5760f4.1, constants.%DestroyImpl.impl_witness.3a6) [concrete = constants.%facet_value.ca6]
// CHECK:STDOUT:   %.loc37_37.16: %facet_type.8a3 = converted constants.%OverloadValueType, %facet_value.loc37_37.5 [concrete = constants.%facet_value.ca6]
// CHECK:STDOUT:   %facet_value.loc37_37.6: %facet_type.8a3 = facet_value constants.%OverloadValueType, (constants.%custom_witness.5760f4.1, constants.%DestroyImpl.impl_witness.3a6) [concrete = constants.%facet_value.ca6]
// CHECK:STDOUT:   %.loc37_37.17: %facet_type.8a3 = converted constants.%OverloadValueType, %facet_value.loc37_37.6 [concrete = constants.%facet_value.ca6]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc37_37.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc37_37.14)
// CHECK:STDOUT:   %facet_value.loc37_37.7: %facet_type.8a3 = facet_value constants.%Optional.06a, (constants.%custom_witness.b80ef6.7, constants.%DestroyImpl.impl_witness.601) [concrete = constants.%facet_value.4a3]
// CHECK:STDOUT:   %.loc37_37.18: %facet_type.8a3 = converted constants.%Optional.06a, %facet_value.loc37_37.7 [concrete = constants.%facet_value.4a3]
// CHECK:STDOUT:   %facet_value.loc37_37.8: %facet_type.8a3 = facet_value constants.%Optional.06a, (constants.%custom_witness.b80ef6.7, constants.%DestroyImpl.impl_witness.601) [concrete = constants.%facet_value.4a3]
// CHECK:STDOUT:   %.loc37_37.19: %facet_type.8a3 = converted constants.%Optional.06a, %facet_value.loc37_37.8 [concrete = constants.%facet_value.4a3]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc37_37.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc37_37.6)
// CHECK:STDOUT:   br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT: !for.done:
// CHECK:STDOUT:   %facet_value.loc37_37.9: %facet_type.8a3 = facet_value constants.%Optional.06a, (constants.%custom_witness.b80ef6.7, constants.%DestroyImpl.impl_witness.601) [concrete = constants.%facet_value.4a3]
// CHECK:STDOUT:   %.loc37_37.20: %facet_type.8a3 = converted constants.%Optional.06a, %facet_value.loc37_37.9 [concrete = constants.%facet_value.4a3]
// CHECK:STDOUT:   %facet_value.loc37_37.10: %facet_type.8a3 = facet_value constants.%Optional.06a, (constants.%custom_witness.b80ef6.7, constants.%DestroyImpl.impl_witness.601) [concrete = constants.%facet_value.4a3]
// CHECK:STDOUT:   %.loc37_37.21: %facet_type.8a3 = converted constants.%Optional.06a, %facet_value.loc37_37.10 [concrete = constants.%facet_value.4a3]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc37_37.3: init %empty_tuple.type = call %SelfDestruct.specific_fn.3(%.loc37_37.6)
// CHECK:STDOUT:   %facet_value.loc37_37.11: %facet_type.8a3 = facet_value constants.%tuple.type.9ef, (constants.%custom_witness.b80ef6.8, constants.%DestroyImpl.impl_witness.d8b) [concrete = constants.%facet_value.b15]
// CHECK:STDOUT:   %.loc37_37.22: %facet_type.8a3 = converted constants.%tuple.type.9ef, %facet_value.loc37_37.11 [concrete = constants.%facet_value.b15]
// CHECK:STDOUT:   %facet_value.loc37_37.12: %facet_type.8a3 = facet_value constants.%tuple.type.9ef, (constants.%custom_witness.b80ef6.8, constants.%DestroyImpl.impl_witness.d8b) [concrete = constants.%facet_value.b15]
// CHECK:STDOUT:   %.loc37_37.23: %facet_type.8a3 = converted constants.%tuple.type.9ef, %facet_value.loc37_37.12 [concrete = constants.%facet_value.b15]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc37_37.4: init %empty_tuple.type = call %SelfDestruct.specific_fn.4(%var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestIterate(constants.%R) {
// CHECK:STDOUT:   %R.patt.loc25_17.2 => constants.%R.patt
// CHECK:STDOUT:   %R.loc25_17.1 => constants.%R
// CHECK:STDOUT:   %R.as_type.loc25_83.1 => constants.%R.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.64f
// CHECK:STDOUT:   %r.patt.loc25_81.2 => constants.%r.patt.b94
// CHECK:STDOUT:   %r.param_patt.loc25_76.2 => constants.%r.param_patt.026
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestIterate(constants.%facet_value.1fb) {
// CHECK:STDOUT:   %R.patt.loc25_17.2 => constants.%R.patt
// CHECK:STDOUT:   %R.loc25_17.1 => constants.%facet_value.1fb
// CHECK:STDOUT:   %R.as_type.loc25_83.1 => constants.%Range
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.23f
// CHECK:STDOUT:   %r.patt.loc25_81.2 => constants.%r.patt.b9f
// CHECK:STDOUT:   %r.param_patt.loc25_76.2 => constants.%r.param_patt.466
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc25 => constants.%complete_type.357
// CHECK:STDOUT:   %Iterate.lookup_impl_witness => constants.%Iterate.impl_witness.5b6
// CHECK:STDOUT:   %Iterate.facet.loc27_37.5 => constants.%Iterate.facet.12c
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type => constants.%Iterate.WithSelf.NewCursor.type.5cf
// CHECK:STDOUT:   %.loc27_37.26 => constants.%.19f
// CHECK:STDOUT:   %impl.elem2.loc27_37.2 => constants.%T.as_type.as.Iterate.impl.NewCursor.bb9
// CHECK:STDOUT:   %specific_impl_fn.loc27_37.3 => constants.%T.as_type.as.Iterate.impl.NewCursor.specific_fn
// CHECK:STDOUT:   %impl.elem1 => constants.%facet_value.b15
// CHECK:STDOUT:   %as_type => constants.%tuple.type.9ef
// CHECK:STDOUT:   %require_complete.1 => constants.%complete_type.e35
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type => constants.%Iterate.WithSelf.Next.type.f29
// CHECK:STDOUT:   %.loc27_37.27 => constants.%.2ed
// CHECK:STDOUT:   %impl.elem3.loc27_37.2 => constants.%T.as_type.as.Iterate.impl.Next.e07
// CHECK:STDOUT:   %specific_impl_fn.loc27_37.4 => constants.%T.as_type.as.Iterate.impl.Next.specific_fn
// CHECK:STDOUT:   %SelfDestruct.specific_fn.5 => constants.%SelfDestruct.specific_fn.e65
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- adl_return_same_types.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %Iterate.type: type = facet_type <@Iterate> [concrete]
// CHECK:STDOUT:   %OptionalStorage.type: type = facet_type <@OptionalStorage> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %facet_type.d61: type = facet_type <@SubobjectDestroy & @DestroyImpl & @Copy> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %T.220: %OptionalStorage.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Optional.Get.type.d97: type = fn_type @Optional.Get, @Optional(%T.220) [symbolic]
// CHECK:STDOUT:   %Optional.Get.3e9: %Optional.Get.type.d97 = struct_value () [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.type.cef: type = fn_type @Optional.HasValue, @Optional(%T.220) [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.a74: %Optional.HasValue.type.cef = struct_value () [symbolic]
// CHECK:STDOUT:   %facet_type.8a3: type = facet_type <@SubobjectDestroy & @DestroyImpl> [concrete]
// CHECK:STDOUT:   %T.9ab: %facet_type.d61 = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.19c: type = facet_access_type %T.9ab [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.lookup_impl_witness.855: <witness> = lookup_impl_witness %T.9ab, @SubobjectDestroy [symbolic]
// CHECK:STDOUT:   %DestroyImpl.lookup_impl_witness.ff9: <witness> = lookup_impl_witness %T.9ab, @DestroyImpl [symbolic]
// CHECK:STDOUT:   %facet_value.b35: %facet_type.8a3 = facet_value %T.as_type.19c, (%SubobjectDestroy.lookup_impl_witness.855, %DestroyImpl.lookup_impl_witness.ff9) [symbolic]
// CHECK:STDOUT:   %MaybeUnformed.413: type = class_type @MaybeUnformed, @MaybeUnformed(%facet_value.b35) [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.a6b: type = class_type @DefaultOptionalStorage, @DefaultOptionalStorage(%T.9ab) [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.elem.94e: type = unbound_element_type %DefaultOptionalStorage.a6b, bool [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.elem.c3f: type = unbound_element_type %DefaultOptionalStorage.a6b, %MaybeUnformed.413 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Get.type.b32: type = fn_type @T.as_type.as.OptionalStorage.impl.Get, @T.as_type.as.OptionalStorage.impl(%T.9ab) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Get.8b0: %T.as_type.as.OptionalStorage.impl.Get.type.b32 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Has.type.8e5: type = fn_type @T.as_type.as.OptionalStorage.impl.Has, @T.as_type.as.OptionalStorage.impl(%T.9ab) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Has.2d5: %T.as_type.as.OptionalStorage.impl.Has.type.8e5 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Some.type.61d: type = fn_type @T.as_type.as.OptionalStorage.impl.Some, @T.as_type.as.OptionalStorage.impl(%T.9ab) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Some.ba7: %T.as_type.as.OptionalStorage.impl.Some.type.61d = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.None.type.a9f: type = fn_type @T.as_type.as.OptionalStorage.impl.None, @T.as_type.as.OptionalStorage.impl(%T.9ab) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.None.083: %T.as_type.as.OptionalStorage.impl.None.type.a9f = struct_value () [symbolic]
// CHECK:STDOUT:   %N.fe9: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.273: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.5b5: %T.as_type.as.DestroyImpl.impl.Op.type.273 = struct_value () [symbolic]
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
// CHECK:STDOUT:   %SubobjectDestroy.lookup_impl_witness.dc9: <witness> = lookup_impl_witness %tuple.type.1ba, @SubobjectDestroy [symbolic]
// CHECK:STDOUT:   %DestroyImpl.lookup_impl_witness.504: <witness> = lookup_impl_witness %tuple.type.1ba, @DestroyImpl [symbolic]
// CHECK:STDOUT:   %facet_value.bf6: %facet_type.8a3 = facet_value %tuple.type.1ba, (%SubobjectDestroy.lookup_impl_witness.dc9, %DestroyImpl.lookup_impl_witness.504) [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.lookup_impl_witness.b15da7.2: <witness> = lookup_impl_witness %impl.elem0.493765.2, @SubobjectDestroy [symbolic]
// CHECK:STDOUT:   %DestroyImpl.lookup_impl_witness.4e7f44.2: <witness> = lookup_impl_witness %impl.elem0.493765.2, @DestroyImpl [symbolic]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.167: <witness> = lookup_impl_witness %impl.elem0.493765.2, @Copy [symbolic]
// CHECK:STDOUT:   %facet_value.9da: %facet_type.d61 = facet_value %impl.elem0.493765.2, (%SubobjectDestroy.lookup_impl_witness.b15da7.2, %DestroyImpl.lookup_impl_witness.4e7f44.2, %Copy.lookup_impl_witness.167) [symbolic]
// CHECK:STDOUT:   %SelfDestruct.type: type = fn_type @SelfDestruct [concrete]
// CHECK:STDOUT:   %SelfDestruct: %SelfDestruct.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete]
// CHECK:STDOUT:   %f64.978: type = class_type @Float, @Float(%int_64) [concrete]
// CHECK:STDOUT:   %pattern_type.522: type = pattern_type %f64.978 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.2: type = fn_type @SubobjectDestroy.WithSelf.Op.loc20 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.2: %SubobjectDestroy.WithSelf.Op.type.6870fd.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.2: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.2), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.76ce: %SubobjectDestroy.type = facet_value %f64.978, (%custom_witness.b80ef6.2) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.868: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.76ce) [concrete]
// CHECK:STDOUT:   %Float.as.Copy.impl.Op.type.afa: type = fn_type @Float.as.Copy.impl.Op, @Float.as.Copy.impl(%N.fe9) [symbolic]
// CHECK:STDOUT:   %Float.as.Copy.impl.Op.e99: %Float.as.Copy.impl.Op.type.afa = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.e6c: <witness> = impl_witness imports.%Copy.impl_witness_table.f03, @Float.as.Copy.impl(%int_64) [concrete]
// CHECK:STDOUT:   %facet_value.83c: %facet_type.d61 = facet_value %f64.978, (%custom_witness.b80ef6.2, %DestroyImpl.impl_witness.868, %Copy.impl_witness.e6c) [concrete]
// CHECK:STDOUT:   %Iterate.lookup_impl_witness.1d8: <witness> = lookup_impl_witness %.Self.500, @Iterate [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.8a4: %facet_type.d61 = impl_witness_access %Iterate.lookup_impl_witness.1d8, element0 [symbolic_self]
// CHECK:STDOUT:   %Iterate_where.type.cd5: type = facet_type <@Iterate where %impl.elem0.8a4 = %facet_value.83c> [concrete]
// CHECK:STDOUT:   %pattern_type.1b8: type = pattern_type %Iterate_where.type.cd5 [concrete]
// CHECK:STDOUT:   %R.patt: %pattern_type.1b8 = symbolic_binding_pattern R, 0 [symbolic]
// CHECK:STDOUT:   %R: %Iterate_where.type.cd5 = symbolic_binding R, 0 [symbolic]
// CHECK:STDOUT:   %R.as_type: type = facet_access_type %R [symbolic]
// CHECK:STDOUT:   %pattern_type.bac: type = pattern_type %R.as_type [symbolic]
// CHECK:STDOUT:   %r.param_patt.954: %pattern_type.bac = value_param_pattern [symbolic]
// CHECK:STDOUT:   %r.patt.41e: %pattern_type.bac = wrapper_binding_pattern r, %r.param_patt.954 [symbolic]
// CHECK:STDOUT:   %i.patt: %pattern_type.522 = value_binding_pattern i [concrete]
// CHECK:STDOUT:   %Iterate.lookup_impl_witness.935: <witness> = lookup_impl_witness %R, @Iterate [symbolic]
// CHECK:STDOUT:   %Iterate.facet.c16: %Iterate.type = facet_value %R.as_type, (%Iterate.lookup_impl_witness.935) [symbolic]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.b97: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.c16) [symbolic]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.ff4: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.c16) [symbolic]
// CHECK:STDOUT:   %.e93: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.b97, %Iterate.facet.c16 [symbolic]
// CHECK:STDOUT:   %impl.elem2.7a3: %.e93 = impl_witness_access %Iterate.lookup_impl_witness.935, element2 [symbolic]
// CHECK:STDOUT:   %impl.elem1.c65: %facet_type.8a3 = impl_witness_access %Iterate.lookup_impl_witness.935, element1 [symbolic]
// CHECK:STDOUT:   %as_type.b5f: type = facet_access_type %impl.elem1.c65 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.46d: <specific function> = specific_impl_function %impl.elem2.7a3, @Iterate.WithSelf.NewCursor(%Iterate.facet.c16) [symbolic]
// CHECK:STDOUT:   %.d61: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.ff4, %Iterate.facet.c16 [symbolic]
// CHECK:STDOUT:   %impl.elem3.09c: %.d61 = impl_witness_access %Iterate.lookup_impl_witness.935, element3 [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.cd2: type = class_type @DefaultOptionalStorage, @DefaultOptionalStorage(%facet_value.83c) [concrete]
// CHECK:STDOUT:   %OptionalStorage.impl_witness.310: <witness> = impl_witness imports.%OptionalStorage.impl_witness_table.968, @T.as_type.as.OptionalStorage.impl(%facet_value.83c) [concrete]
// CHECK:STDOUT:   %OptionalStorage.facet.040: %OptionalStorage.type = facet_value %f64.978, (%OptionalStorage.impl_witness.310) [concrete]
// CHECK:STDOUT:   %Optional.d01: type = class_type @Optional, @Optional(%OptionalStorage.facet.040) [concrete]
// CHECK:STDOUT:   %specific_impl_fn.8a2: <specific function> = specific_impl_function %impl.elem3.09c, @Iterate.WithSelf.Next(%Iterate.facet.c16) [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.type.4ec: type = fn_type @Optional.HasValue, @Optional(%OptionalStorage.facet.040) [concrete]
// CHECK:STDOUT:   %Optional.HasValue.d29: %Optional.HasValue.type.4ec = struct_value () [concrete]
// CHECK:STDOUT:   %Optional.Get.type.53f: type = fn_type @Optional.Get, @Optional(%OptionalStorage.facet.040) [concrete]
// CHECK:STDOUT:   %Optional.Get.cb5: %Optional.Get.type.53f = struct_value () [concrete]
// CHECK:STDOUT:   %facet_value.348: %facet_type.8a3 = facet_value %f64.978, (%custom_witness.b80ef6.2, %DestroyImpl.impl_witness.868) [concrete]
// CHECK:STDOUT:   %MaybeUnformed.609: type = class_type @MaybeUnformed, @MaybeUnformed(%facet_value.348) [concrete]
// CHECK:STDOUT:   %.b81: type = maybe_unformed_type %f64.978 [concrete]
// CHECK:STDOUT:   %DefaultOptionalStorage.elem.fb9: type = unbound_element_type %DefaultOptionalStorage.cd2, %MaybeUnformed.609 [concrete]
// CHECK:STDOUT:   %DefaultOptionalStorage.elem.81c: type = unbound_element_type %DefaultOptionalStorage.cd2, bool [concrete]
// CHECK:STDOUT:   %struct_type.value.has_value.e85: type = struct_type {.value: %MaybeUnformed.609, .has_value: bool} [concrete]
// CHECK:STDOUT:   %Optional.HasValue.specific_fn.6ca: <specific function> = specific_function %Optional.HasValue.d29, @Optional.HasValue(%OptionalStorage.facet.040) [concrete]
// CHECK:STDOUT:   %Optional.Get.specific_fn.e9a: <specific function> = specific_function %Optional.Get.cb5, @Optional.Get(%OptionalStorage.facet.040) [concrete]
// CHECK:STDOUT:   %AddAssignWith.type.8e8: type = facet_type <@AddAssignWith, @AddAssignWith(%f64.978)> [concrete]
// CHECK:STDOUT:   %Float.as.AddAssignWith.impl.Op.type.bd0: type = fn_type @Float.as.AddAssignWith.impl.Op, @Float.as.AddAssignWith.impl(%N.fe9) [symbolic]
// CHECK:STDOUT:   %Float.as.AddAssignWith.impl.Op.f48: %Float.as.AddAssignWith.impl.Op.type.bd0 = struct_value () [symbolic]
// CHECK:STDOUT:   %AddAssignWith.impl_witness.614: <witness> = impl_witness imports.%AddAssignWith.impl_witness_table, @Float.as.AddAssignWith.impl(%int_64) [concrete]
// CHECK:STDOUT:   %Float.as.AddAssignWith.impl.Op.type.223: type = fn_type @Float.as.AddAssignWith.impl.Op, @Float.as.AddAssignWith.impl(%int_64) [concrete]
// CHECK:STDOUT:   %Float.as.AddAssignWith.impl.Op.cd7: %Float.as.AddAssignWith.impl.Op.type.223 = struct_value () [concrete]
// CHECK:STDOUT:   %AddAssignWith.facet: %AddAssignWith.type.8e8 = facet_value %f64.978, (%AddAssignWith.impl_witness.614) [concrete]
// CHECK:STDOUT:   %AddAssignWith.WithSelf.Op.type.58f: type = fn_type @AddAssignWith.WithSelf.Op, @AddAssignWith.WithSelf(%f64.978, %AddAssignWith.facet) [concrete]
// CHECK:STDOUT:   %.74b: type = fn_type_with_self_type %AddAssignWith.WithSelf.Op.type.58f, %AddAssignWith.facet [concrete]
// CHECK:STDOUT:   %Float.as.AddAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Float.as.AddAssignWith.impl.Op.cd7, @Float.as.AddAssignWith.impl.Op(%int_64) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.4: type = fn_type @SubobjectDestroy.WithSelf.Op.loc24_19.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.4: %SubobjectDestroy.WithSelf.Op.type.6870fd.4 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.4: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.4), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.5: type = fn_type @SubobjectDestroy.WithSelf.Op.loc24_19.3 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.5: %SubobjectDestroy.WithSelf.Op.type.6870fd.5 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.5: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.5), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.a87: %SubobjectDestroy.type = facet_value bool, (%custom_witness.b80ef6.5) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.deb: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.a87) [concrete]
// CHECK:STDOUT:   %facet_value.a1b: %facet_type.8a3 = facet_value bool, (%custom_witness.b80ef6.5, %DestroyImpl.impl_witness.deb) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.3e6: %SubobjectDestroy.type = facet_value %MaybeUnformed.609, (%custom_witness.b80ef6.4) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.4ed: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.3e6) [concrete]
// CHECK:STDOUT:   %facet_value.40c: %facet_type.8a3 = facet_value %MaybeUnformed.609, (%custom_witness.b80ef6.4, %DestroyImpl.impl_witness.4ed) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.8: type = fn_type @SubobjectDestroy.WithSelf.Op.loc24_19.6 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.8: %SubobjectDestroy.WithSelf.Op.type.6870fd.8 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.8: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.8), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.76c9: %SubobjectDestroy.type = facet_value %Optional.d01, (%custom_witness.b80ef6.8) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.06a: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.76c9) [concrete]
// CHECK:STDOUT:   %facet_value.bb3: %facet_type.8a3 = facet_value %Optional.d01, (%custom_witness.b80ef6.8, %DestroyImpl.impl_witness.06a) [concrete]
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
// CHECK:STDOUT:   %custom_witness.ed9: <witness> = custom_witness (%f64.978, %Iterator.Op.583fda.1), @CppUnsafeDeref [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.ac7e09.2: type = fn_type @Iterator.Op.2 [concrete]
// CHECK:STDOUT:   %Iterator.Op.583fda.2: %Iterator.Op.type.ac7e09.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.1fc: <witness> = custom_witness (%Iterator.Op.583fda.2), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.d46: %SubobjectDestroy.type = facet_value %Iterator, (%custom_witness.1fc) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.b32: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.d46) [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.ac7e09.3: type = fn_type @Iterator.Op.3 [concrete]
// CHECK:STDOUT:   %Iterator.Op.583fda.3: %Iterator.Op.type.ac7e09.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.393: <witness> = custom_witness (%Iterator.Op.583fda.3), @Inc [concrete]
// CHECK:STDOUT:   %Equal.type: type = fn_type @Equal [concrete]
// CHECK:STDOUT:   %Equal: %Equal.type = struct_value () [concrete]
// CHECK:STDOUT:   %NotEqual.type: type = fn_type @NotEqual [concrete]
// CHECK:STDOUT:   %NotEqual: %NotEqual.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.821: <witness> = custom_witness (%Equal, %NotEqual), @EqWith, @EqWith(%Iterator) [concrete]
// CHECK:STDOUT:   %facet_value.731: %CppRange_where.type.d682d6.1 = facet_value %ConstRange, (%custom_witness.b80ef6.2, %DestroyImpl.impl_witness.868, %Copy.impl_witness.e6c, %custom_witness.5d5, %custom_witness.1fc, %DestroyImpl.impl_witness.b32, %custom_witness.ed9, %custom_witness.393, %custom_witness.821) [concrete]
// CHECK:STDOUT:   %tuple.type.005: type = tuple_type (%Iterator, %Iterator) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.9: type = fn_type @SubobjectDestroy.WithSelf.Op.3 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.9: %SubobjectDestroy.WithSelf.Op.type.6870fd.9 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.9: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.9), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.7c6: %SubobjectDestroy.type = facet_value %tuple.type.005, (%custom_witness.b80ef6.9) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.7493: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.7c6) [concrete]
// CHECK:STDOUT:   %complete_type.320: <witness> = complete_type_witness %tuple.type.005 [concrete]
// CHECK:STDOUT:   %facet_value.6a6: %facet_type.8a3 = facet_value %tuple.type.005, (%custom_witness.b80ef6.9, %DestroyImpl.impl_witness.7493) [concrete]
// CHECK:STDOUT:   %Iterate.impl_witness.2a3: <witness> = impl_witness imports.%Iterate.impl_witness_table.121, @T.as_type.as.Iterate.impl(%facet_value.731) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.type.4fa: type = fn_type @T.as_type.as.Iterate.impl.NewCursor, @T.as_type.as.Iterate.impl(%facet_value.731) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.266: %T.as_type.as.Iterate.impl.NewCursor.type.4fa = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.type.0dd: type = fn_type @T.as_type.as.Iterate.impl.Next, @T.as_type.as.Iterate.impl(%facet_value.731) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.b00: %T.as_type.as.Iterate.impl.Next.type.0dd = struct_value () [concrete]
// CHECK:STDOUT:   %Iterate.facet.7d5: %Iterate.type = facet_value %ConstRange, (%Iterate.impl_witness.2a3) [concrete]
// CHECK:STDOUT:   %facet_value.062: %Iterate_where.type.cd5 = facet_value %ConstRange, (%Iterate.impl_witness.2a3) [concrete]
// CHECK:STDOUT:   %r.param_patt.a73: %pattern_type.025 = value_param_pattern [concrete]
// CHECK:STDOUT:   %r.patt.0b5: %pattern_type.025 = wrapper_binding_pattern r, %r.param_patt.a73 [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.0fc: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.7d5) [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.170: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.7d5) [concrete]
// CHECK:STDOUT:   %.35a: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.0fc, %Iterate.facet.7d5 [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.specific_fn: <specific function> = specific_function %T.as_type.as.Iterate.impl.NewCursor.266, @T.as_type.as.Iterate.impl.NewCursor(%facet_value.731) [concrete]
// CHECK:STDOUT:   %.e4c: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.170, %Iterate.facet.7d5 [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.specific_fn: <specific function> = specific_function %T.as_type.as.Iterate.impl.Next.b00, @T.as_type.as.Iterate.impl.Next(%facet_value.731) [concrete]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.b5d: <specific function> = specific_function %SelfDestruct, @SelfDestruct(%facet_value.6a6) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.4d4: @Optional.%Optional.HasValue.type (%Optional.HasValue.type.cef) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.HasValue (constants.%Optional.HasValue.a74)]
// CHECK:STDOUT:   %Core.import_ref.059: @Optional.%Optional.Get.type (%Optional.Get.type.d97) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.Get (constants.%Optional.Get.3e9)]
// CHECK:STDOUT:   %Core.import_ref.7bb: type = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%DefaultOptionalStorage (constants.%DefaultOptionalStorage.a6b)]
// CHECK:STDOUT:   %Core.import_ref.61e: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.None.type (%T.as_type.as.OptionalStorage.impl.None.type.a9f) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.None (constants.%T.as_type.as.OptionalStorage.impl.None.083)]
// CHECK:STDOUT:   %Core.import_ref.ca1: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Some.type (%T.as_type.as.OptionalStorage.impl.Some.type.61d) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Some (constants.%T.as_type.as.OptionalStorage.impl.Some.ba7)]
// CHECK:STDOUT:   %Core.import_ref.6ba4: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Has.type (%T.as_type.as.OptionalStorage.impl.Has.type.8e5) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Has (constants.%T.as_type.as.OptionalStorage.impl.Has.2d5)]
// CHECK:STDOUT:   %Core.import_ref.73d: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Get.type (%T.as_type.as.OptionalStorage.impl.Get.type.b32) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Get (constants.%T.as_type.as.OptionalStorage.impl.Get.8b0)]
// CHECK:STDOUT:   %Core.import_ref.778 = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], unloaded
// CHECK:STDOUT:   %OptionalStorage.impl_witness_table.968 = impl_witness_table (%Core.import_ref.7bb, %Core.import_ref.61e, %Core.import_ref.ca1, %Core.import_ref.6ba4, %Core.import_ref.73d, %Core.import_ref.778), @T.as_type.as.OptionalStorage.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.146b: @DefaultOptionalStorage.%DefaultOptionalStorage.elem.1 (%DefaultOptionalStorage.elem.c3f) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [concrete = %field_decl.31e]
// CHECK:STDOUT:   %Core.import_ref.f1da: @DefaultOptionalStorage.%DefaultOptionalStorage.elem.2 (%DefaultOptionalStorage.elem.94e) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [concrete = %field_decl.431]
// CHECK:STDOUT:   %Core.import_ref.14b: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.273) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.5b5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.14b), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.d6c: %facet_type.d61 = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.Iterate.impl.%facet_value.1 (constants.%facet_value.9da)]
// CHECK:STDOUT:   %Core.import_ref.2ac: %facet_type.8a3 = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.Iterate.impl.%facet_value.2 (constants.%facet_value.bf6)]
// CHECK:STDOUT:   %Core.import_ref.fd1a: @T.as_type.as.Iterate.impl.%T.as_type.as.Iterate.impl.NewCursor.type (%T.as_type.as.Iterate.impl.NewCursor.type.953) = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.Iterate.impl.%T.as_type.as.Iterate.impl.NewCursor (constants.%T.as_type.as.Iterate.impl.NewCursor.544)]
// CHECK:STDOUT:   %Core.import_ref.273: @T.as_type.as.Iterate.impl.%T.as_type.as.Iterate.impl.Next.type (%T.as_type.as.Iterate.impl.Next.type.b36) = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.Iterate.impl.%T.as_type.as.Iterate.impl.Next (constants.%T.as_type.as.Iterate.impl.Next.d6b)]
// CHECK:STDOUT:   %Iterate.impl_witness_table.121 = impl_witness_table (%Core.import_ref.d6c, %Core.import_ref.2ac, %Core.import_ref.fd1a, %Core.import_ref.273), @T.as_type.as.Iterate.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.bd6: @Float.as.Copy.impl.%Float.as.Copy.impl.Op.type (%Float.as.Copy.impl.Op.type.afa) = import_ref Core//prelude/types/float, loc{{\d+_\d+}}, loaded [symbolic = @Float.as.Copy.impl.%Float.as.Copy.impl.Op (constants.%Float.as.Copy.impl.Op.e99)]
// CHECK:STDOUT:   %Copy.impl_witness_table.f03 = impl_witness_table (%Core.import_ref.bd6), @Float.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.4e0: @Float.as.AddAssignWith.impl.%Float.as.AddAssignWith.impl.Op.type (%Float.as.AddAssignWith.impl.Op.type.bd0) = import_ref Core//prelude/types/float, loc{{\d+_\d+}}, loaded [symbolic = @Float.as.AddAssignWith.impl.%Float.as.AddAssignWith.impl.Op (constants.%Float.as.AddAssignWith.impl.Op.f48)]
// CHECK:STDOUT:   %AddAssignWith.impl_witness_table = impl_witness_table (%Core.import_ref.4e0), @Float.as.AddAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.b0d = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %field_decl.431: @DefaultOptionalStorage.%DefaultOptionalStorage.elem.2 (%DefaultOptionalStorage.elem.94e) = field_decl has_value, element1, %Core.import_ref.b0d in [concrete] {}
// CHECK:STDOUT:   %Core.import_ref.164c = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %field_decl.31e: @DefaultOptionalStorage.%DefaultOptionalStorage.elem.1 (%DefaultOptionalStorage.elem.c3f) = field_decl value, element0, %Core.import_ref.164c in [concrete] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @TestIterate(%R.loc20_17.2: %Iterate_where.type.cd5) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Iterate.lookup_impl_witness: <witness> = lookup_impl_witness %R.loc20_17.1, @Iterate [symbolic = %Iterate.lookup_impl_witness (constants.%Iterate.lookup_impl_witness.935)]
// CHECK:STDOUT:   %Iterate.facet.loc24_19.5: %Iterate.type = facet_value %R.as_type.loc20_61.1, (%Iterate.lookup_impl_witness) [symbolic = %Iterate.facet.loc24_19.5 (constants.%Iterate.facet.c16)]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.loc24_19.5) [symbolic = %Iterate.WithSelf.NewCursor.type (constants.%Iterate.WithSelf.NewCursor.type.b97)]
// CHECK:STDOUT:   %.loc24_19.23: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type, %Iterate.facet.loc24_19.5 [symbolic = %.loc24_19.23 (constants.%.e93)]
// CHECK:STDOUT:   %impl.elem2.loc24_19.2: @TestIterate.%.loc24_19.23 (%.e93) = impl_witness_access %Iterate.lookup_impl_witness, element2 [symbolic = %impl.elem2.loc24_19.2 (constants.%impl.elem2.7a3)]
// CHECK:STDOUT:   %specific_impl_fn.loc24_19.3: <specific function> = specific_impl_function %impl.elem2.loc24_19.2, @Iterate.WithSelf.NewCursor(%Iterate.facet.loc24_19.5) [symbolic = %specific_impl_fn.loc24_19.3 (constants.%specific_impl_fn.46d)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.loc24_19.5) [symbolic = %Iterate.WithSelf.Next.type (constants.%Iterate.WithSelf.Next.type.ff4)]
// CHECK:STDOUT:   %.loc24_19.24: type = fn_type_with_self_type %Iterate.WithSelf.Next.type, %Iterate.facet.loc24_19.5 [symbolic = %.loc24_19.24 (constants.%.d61)]
// CHECK:STDOUT:   %impl.elem3.loc24_19.2: @TestIterate.%.loc24_19.24 (%.d61) = impl_witness_access %Iterate.lookup_impl_witness, element3 [symbolic = %impl.elem3.loc24_19.2 (constants.%impl.elem3.09c)]
// CHECK:STDOUT:   %specific_impl_fn.loc24_19.4: <specific function> = specific_impl_function %impl.elem3.loc24_19.2, @Iterate.WithSelf.Next(%Iterate.facet.loc24_19.5) [symbolic = %specific_impl_fn.loc24_19.4 (constants.%specific_impl_fn.8a2)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%r.param: @TestIterate.%R.as_type.loc20_61.1 (%R.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %i.patt: %pattern_type.522 = value_binding_pattern i [concrete = constants.%i.patt]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %r.ref: @TestIterate.%R.as_type.loc20_61.1 (%R.as_type) = name_ref r, %r
// CHECK:STDOUT:     %impl.elem2.loc24_19.1: @TestIterate.%.loc24_19.23 (%.e93) = impl_witness_access constants.%Iterate.lookup_impl_witness.935, element2 [symbolic = %impl.elem2.loc24_19.2 (constants.%impl.elem2.7a3)]
// CHECK:STDOUT:     %bound_method.loc24_19.1: <bound method> = bound_method %r.ref, %impl.elem2.loc24_19.1
// CHECK:STDOUT:     %Iterate.facet.loc24_19.1: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.935) [symbolic = %Iterate.facet.loc24_19.5 (constants.%Iterate.facet.c16)]
// CHECK:STDOUT:     %.loc24_19.1: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc24_19.1 [symbolic = %Iterate.facet.loc24_19.5 (constants.%Iterate.facet.c16)]
// CHECK:STDOUT:     %Iterate.facet.loc24_19.2: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.935) [symbolic = %Iterate.facet.loc24_19.5 (constants.%Iterate.facet.c16)]
// CHECK:STDOUT:     %.loc24_19.2: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc24_19.2 [symbolic = %Iterate.facet.loc24_19.5 (constants.%Iterate.facet.c16)]
// CHECK:STDOUT:     %specific_impl_fn.loc24_19.1: <specific function> = specific_impl_function %impl.elem2.loc24_19.1, @Iterate.WithSelf.NewCursor(constants.%Iterate.facet.c16) [symbolic = %specific_impl_fn.loc24_19.3 (constants.%specific_impl_fn.46d)]
// CHECK:STDOUT:     %bound_method.loc24_19.2: <bound method> = bound_method %r.ref, %specific_impl_fn.loc24_19.1
// CHECK:STDOUT:     %var: ref @TestIterate.%as_type (%as_type.b5f) = var_storage invalid
// CHECK:STDOUT:     %Iterate.WithSelf.NewCursor.call: init @TestIterate.%as_type (%as_type.b5f) to %var = call %bound_method.loc24_19.2(%r.ref)
// CHECK:STDOUT:     assign %var, %Iterate.WithSelf.NewCursor.call
// CHECK:STDOUT:     br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.next:
// CHECK:STDOUT:     %impl.elem3.loc24_19.1: @TestIterate.%.loc24_19.24 (%.d61) = impl_witness_access constants.%Iterate.lookup_impl_witness.935, element3 [symbolic = %impl.elem3.loc24_19.2 (constants.%impl.elem3.09c)]
// CHECK:STDOUT:     %bound_method.loc24_19.3: <bound method> = bound_method %r.ref, %impl.elem3.loc24_19.1
// CHECK:STDOUT:     %Iterate.facet.loc24_19.3: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.935) [symbolic = %Iterate.facet.loc24_19.5 (constants.%Iterate.facet.c16)]
// CHECK:STDOUT:     %.loc24_19.3: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc24_19.3 [symbolic = %Iterate.facet.loc24_19.5 (constants.%Iterate.facet.c16)]
// CHECK:STDOUT:     %Iterate.facet.loc24_19.4: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.935) [symbolic = %Iterate.facet.loc24_19.5 (constants.%Iterate.facet.c16)]
// CHECK:STDOUT:     %.loc24_19.4: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc24_19.4 [symbolic = %Iterate.facet.loc24_19.5 (constants.%Iterate.facet.c16)]
// CHECK:STDOUT:     %.loc24_19.5: %facet_type.8a3 = converted constants.%as_type.b5f, constants.%impl.elem1.c65 [symbolic = %impl.elem1 (constants.%impl.elem1.c65)]
// CHECK:STDOUT:     %.loc24_19.6: %facet_type.8a3 = converted constants.%as_type.b5f, constants.%impl.elem1.c65 [symbolic = %impl.elem1 (constants.%impl.elem1.c65)]
// CHECK:STDOUT:     %specific_impl_fn.loc24_19.2: <specific function> = specific_impl_function %impl.elem3.loc24_19.1, @Iterate.WithSelf.Next(constants.%Iterate.facet.c16) [symbolic = %specific_impl_fn.loc24_19.4 (constants.%specific_impl_fn.8a2)]
// CHECK:STDOUT:     %bound_method.loc24_19.4: <bound method> = bound_method %r.ref, %specific_impl_fn.loc24_19.2
// CHECK:STDOUT:     %.loc24_19.7: ref %Optional.d01 = temporary_storage
// CHECK:STDOUT:     %Iterate.WithSelf.Next.call: init %Optional.d01 to %.loc24_19.7 = call %bound_method.loc24_19.4(%r.ref, %var)
// CHECK:STDOUT:     %.loc24_19.8: ref %Optional.d01 = temporary %.loc24_19.7, %Iterate.WithSelf.Next.call
// CHECK:STDOUT:     %.loc24_19.9: %Optional.HasValue.type.4ec = specific_constant imports.%Core.import_ref.4d4, @Optional(constants.%OptionalStorage.facet.040) [concrete = constants.%Optional.HasValue.d29]
// CHECK:STDOUT:     %HasValue.ref: %Optional.HasValue.type.4ec = name_ref HasValue, %.loc24_19.9 [concrete = constants.%Optional.HasValue.d29]
// CHECK:STDOUT:     %Optional.HasValue.bound: <bound method> = bound_method %.loc24_19.8, %HasValue.ref
// CHECK:STDOUT:     %Optional.HasValue.specific_fn: <specific function> = specific_function %HasValue.ref, @Optional.HasValue(constants.%OptionalStorage.facet.040) [concrete = constants.%Optional.HasValue.specific_fn.6ca]
// CHECK:STDOUT:     %bound_method.loc24_19.5: <bound method> = bound_method %.loc24_19.8, %Optional.HasValue.specific_fn
// CHECK:STDOUT:     %.loc24_19.10: %Optional.d01 = acquire_value %.loc24_19.8
// CHECK:STDOUT:     %Optional.HasValue.call: init bool = call %bound_method.loc24_19.5(%.loc24_19.10)
// CHECK:STDOUT:     %.loc24_19.11: bool = value_of_initializer %Optional.HasValue.call
// CHECK:STDOUT:     %.loc24_19.12: bool = converted %Optional.HasValue.call, %.loc24_19.11
// CHECK:STDOUT:     if %.loc24_19.12 br !for.body else br !for.done
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.body:
// CHECK:STDOUT:     %.loc24_19.13: %Optional.Get.type.53f = specific_constant imports.%Core.import_ref.059, @Optional(constants.%OptionalStorage.facet.040) [concrete = constants.%Optional.Get.cb5]
// CHECK:STDOUT:     %Get.ref: %Optional.Get.type.53f = name_ref Get, %.loc24_19.13 [concrete = constants.%Optional.Get.cb5]
// CHECK:STDOUT:     %Optional.Get.bound: <bound method> = bound_method %.loc24_19.8, %Get.ref
// CHECK:STDOUT:     %Optional.Get.specific_fn: <specific function> = specific_function %Get.ref, @Optional.Get(constants.%OptionalStorage.facet.040) [concrete = constants.%Optional.Get.specific_fn.e9a]
// CHECK:STDOUT:     %bound_method.loc24_19.6: <bound method> = bound_method %.loc24_19.8, %Optional.Get.specific_fn
// CHECK:STDOUT:     %.loc24_19.14: %Optional.d01 = acquire_value %.loc24_19.8
// CHECK:STDOUT:     %Optional.Get.call: init %f64.978 = call %bound_method.loc24_19.6(%.loc24_19.14)
// CHECK:STDOUT:     %.loc24_19.15: %f64.978 = value_of_initializer %Optional.Get.call
// CHECK:STDOUT:     %.loc24_19.16: %f64.978 = converted %Optional.Get.call, %.loc24_19.15
// CHECK:STDOUT:     %f64.loc24: type = type_literal constants.%f64.978 [concrete = constants.%f64.978]
// CHECK:STDOUT:     %i: %f64.978 = wrapper_binding i, %.loc24_19.16
// CHECK:STDOUT:     %sum.ref: ref %f64.978 = name_ref sum, %sum
// CHECK:STDOUT:     %i.ref: %f64.978 = name_ref i, %i
// CHECK:STDOUT:     %impl.elem0.loc25: %.74b = impl_witness_access constants.%AddAssignWith.impl_witness.614, element0 [concrete = constants.%Float.as.AddAssignWith.impl.Op.cd7]
// CHECK:STDOUT:     %bound_method.loc25_9.1: <bound method> = bound_method %sum.ref, %impl.elem0.loc25
// CHECK:STDOUT:     %specific_fn.loc25: <specific function> = specific_function %impl.elem0.loc25, @Float.as.AddAssignWith.impl.Op(constants.%int_64) [concrete = constants.%Float.as.AddAssignWith.impl.Op.specific_fn]
// CHECK:STDOUT:     %bound_method.loc25_9.2: <bound method> = bound_method %sum.ref, %specific_fn.loc25
// CHECK:STDOUT:     %Float.as.AddAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc25_9.2(%sum.ref, %i.ref)
// CHECK:STDOUT:     %facet_value.loc24_19.1: %facet_type.8a3 = facet_value constants.%Optional.d01, (constants.%custom_witness.b80ef6.8, constants.%DestroyImpl.impl_witness.06a) [concrete = constants.%facet_value.bb3]
// CHECK:STDOUT:     %.loc24_19.17: %facet_type.8a3 = converted constants.%Optional.d01, %facet_value.loc24_19.1 [concrete = constants.%facet_value.bb3]
// CHECK:STDOUT:     %facet_value.loc24_19.2: %facet_type.8a3 = facet_value constants.%Optional.d01, (constants.%custom_witness.b80ef6.8, constants.%DestroyImpl.impl_witness.06a) [concrete = constants.%facet_value.bb3]
// CHECK:STDOUT:     %.loc24_19.18: %facet_type.8a3 = converted constants.%Optional.d01, %facet_value.loc24_19.2 [concrete = constants.%facet_value.bb3]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %SelfDestruct.call.loc24_19.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc24_19.8)
// CHECK:STDOUT:     br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.done:
// CHECK:STDOUT:     %facet_value.loc24_19.3: %facet_type.8a3 = facet_value constants.%Optional.d01, (constants.%custom_witness.b80ef6.8, constants.%DestroyImpl.impl_witness.06a) [concrete = constants.%facet_value.bb3]
// CHECK:STDOUT:     %.loc24_19.19: %facet_type.8a3 = converted constants.%Optional.d01, %facet_value.loc24_19.3 [concrete = constants.%facet_value.bb3]
// CHECK:STDOUT:     %facet_value.loc24_19.4: %facet_type.8a3 = facet_value constants.%Optional.d01, (constants.%custom_witness.b80ef6.8, constants.%DestroyImpl.impl_witness.06a) [concrete = constants.%facet_value.bb3]
// CHECK:STDOUT:     %.loc24_19.20: %facet_type.8a3 = converted constants.%Optional.d01, %facet_value.loc24_19.4 [concrete = constants.%facet_value.bb3]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %SelfDestruct.call.loc24_19.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc24_19.8)
// CHECK:STDOUT:     %.loc24_19.21: %facet_type.8a3 = converted constants.%as_type.b5f, constants.%impl.elem1.c65 [symbolic = %impl.elem1 (constants.%impl.elem1.c65)]
// CHECK:STDOUT:     %.loc24_19.22: %facet_type.8a3 = converted constants.%as_type.b5f, constants.%impl.elem1.c65 [symbolic = %impl.elem1 (constants.%impl.elem1.c65)]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %SelfDestruct.call.loc24_19.3: init %empty_tuple.type = call %SelfDestruct.specific_fn.3(%var)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc24_19.1(%self.param: ref %.b81) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc24_19.2(%self.param: ref %MaybeUnformed.609) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc24_19.3(%self.param: ref bool) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc24_19.4(%self.param: ref %struct_type.value.has_value.e85) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc24_19.1: ref bool = struct_access %self.param, element1
// CHECK:STDOUT:   %facet_value.loc24_19.1: %facet_type.8a3 = facet_value bool, (constants.%custom_witness.b80ef6.5, constants.%DestroyImpl.impl_witness.deb) [concrete = constants.%facet_value.a1b]
// CHECK:STDOUT:   %.loc24_19.2: %facet_type.8a3 = converted bool, %facet_value.loc24_19.1 [concrete = constants.%facet_value.a1b]
// CHECK:STDOUT:   %facet_value.loc24_19.2: %facet_type.8a3 = facet_value bool, (constants.%custom_witness.b80ef6.5, constants.%DestroyImpl.impl_witness.deb) [concrete = constants.%facet_value.a1b]
// CHECK:STDOUT:   %.loc24_19.3: %facet_type.8a3 = converted bool, %facet_value.loc24_19.2 [concrete = constants.%facet_value.a1b]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc24_19.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc24_19.1)
// CHECK:STDOUT:   %.loc24_19.4: ref %MaybeUnformed.609 = struct_access %self.param, element0
// CHECK:STDOUT:   %facet_value.loc24_19.3: %facet_type.8a3 = facet_value constants.%MaybeUnformed.609, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.4ed) [concrete = constants.%facet_value.40c]
// CHECK:STDOUT:   %.loc24_19.5: %facet_type.8a3 = converted constants.%MaybeUnformed.609, %facet_value.loc24_19.3 [concrete = constants.%facet_value.40c]
// CHECK:STDOUT:   %facet_value.loc24_19.4: %facet_type.8a3 = facet_value constants.%MaybeUnformed.609, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.4ed) [concrete = constants.%facet_value.40c]
// CHECK:STDOUT:   %.loc24_19.6: %facet_type.8a3 = converted constants.%MaybeUnformed.609, %facet_value.loc24_19.4 [concrete = constants.%facet_value.40c]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc24_19.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc24_19.4)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc24_19.5(%self.param: ref %DefaultOptionalStorage.cd2) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %has_value.ref: %DefaultOptionalStorage.elem.81c = name_ref has_value, imports.%Core.import_ref.f1da [concrete = imports.%field_decl.431]
// CHECK:STDOUT:   %.loc24_19.1: ref bool = class_element_access %self.param, element1
// CHECK:STDOUT:   %facet_value.loc24_19.1: %facet_type.8a3 = facet_value bool, (constants.%custom_witness.b80ef6.5, constants.%DestroyImpl.impl_witness.deb) [concrete = constants.%facet_value.a1b]
// CHECK:STDOUT:   %.loc24_19.2: %facet_type.8a3 = converted bool, %facet_value.loc24_19.1 [concrete = constants.%facet_value.a1b]
// CHECK:STDOUT:   %facet_value.loc24_19.2: %facet_type.8a3 = facet_value bool, (constants.%custom_witness.b80ef6.5, constants.%DestroyImpl.impl_witness.deb) [concrete = constants.%facet_value.a1b]
// CHECK:STDOUT:   %.loc24_19.3: %facet_type.8a3 = converted bool, %facet_value.loc24_19.2 [concrete = constants.%facet_value.a1b]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc24_19.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc24_19.1)
// CHECK:STDOUT:   %value.ref: %DefaultOptionalStorage.elem.fb9 = name_ref value, imports.%Core.import_ref.146b [concrete = imports.%field_decl.31e]
// CHECK:STDOUT:   %.loc24_19.4: ref %MaybeUnformed.609 = class_element_access %self.param, element0
// CHECK:STDOUT:   %facet_value.loc24_19.3: %facet_type.8a3 = facet_value constants.%MaybeUnformed.609, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.4ed) [concrete = constants.%facet_value.40c]
// CHECK:STDOUT:   %.loc24_19.5: %facet_type.8a3 = converted constants.%MaybeUnformed.609, %facet_value.loc24_19.3 [concrete = constants.%facet_value.40c]
// CHECK:STDOUT:   %facet_value.loc24_19.4: %facet_type.8a3 = facet_value constants.%MaybeUnformed.609, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.4ed) [concrete = constants.%facet_value.40c]
// CHECK:STDOUT:   %.loc24_19.6: %facet_type.8a3 = converted constants.%MaybeUnformed.609, %facet_value.loc24_19.4 [concrete = constants.%facet_value.40c]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc24_19.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc24_19.4)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc24_19.6(%self.param: ref %Optional.d01) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestRangeFor(%c.param: %ConstRange) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %i.patt: %pattern_type.522 = value_binding_pattern i [concrete = constants.%i.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c.ref: %ConstRange = name_ref c, %c
// CHECK:STDOUT:   %impl.elem2: %.35a = impl_witness_access constants.%Iterate.impl_witness.2a3, element2 [concrete = constants.%T.as_type.as.Iterate.impl.NewCursor.266]
// CHECK:STDOUT:   %bound_method.loc38_19.1: <bound method> = bound_method %c.ref, %impl.elem2
// CHECK:STDOUT:   %facet_value.loc38_19.1: %CppRange_where.type.d682d6.1 = facet_value constants.%ConstRange, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.868, constants.%Copy.impl_witness.e6c, constants.%custom_witness.5d5, constants.%custom_witness.1fc, constants.%DestroyImpl.impl_witness.b32, constants.%custom_witness.ed9, constants.%custom_witness.393, constants.%custom_witness.821) [concrete = constants.%facet_value.731]
// CHECK:STDOUT:   %.loc38_19.1: %CppRange_where.type.d682d6.1 = converted constants.%ConstRange, %facet_value.loc38_19.1 [concrete = constants.%facet_value.731]
// CHECK:STDOUT:   %facet_value.loc38_19.2: %CppRange_where.type.d682d6.1 = facet_value constants.%ConstRange, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.868, constants.%Copy.impl_witness.e6c, constants.%custom_witness.5d5, constants.%custom_witness.1fc, constants.%DestroyImpl.impl_witness.b32, constants.%custom_witness.ed9, constants.%custom_witness.393, constants.%custom_witness.821) [concrete = constants.%facet_value.731]
// CHECK:STDOUT:   %.loc38_19.2: %CppRange_where.type.d682d6.1 = converted constants.%ConstRange, %facet_value.loc38_19.2 [concrete = constants.%facet_value.731]
// CHECK:STDOUT:   %specific_fn.loc38_19.1: <specific function> = specific_function %impl.elem2, @T.as_type.as.Iterate.impl.NewCursor(constants.%facet_value.731) [concrete = constants.%T.as_type.as.Iterate.impl.NewCursor.specific_fn]
// CHECK:STDOUT:   %bound_method.loc38_19.2: <bound method> = bound_method %c.ref, %specific_fn.loc38_19.1
// CHECK:STDOUT:   %var: ref %tuple.type.005 = var_storage invalid
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.call: init %tuple.type.005 to %var = call %bound_method.loc38_19.2(%c.ref)
// CHECK:STDOUT:   assign %var, %T.as_type.as.Iterate.impl.NewCursor.call
// CHECK:STDOUT:   br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT: !for.next:
// CHECK:STDOUT:   %impl.elem3: %.e4c = impl_witness_access constants.%Iterate.impl_witness.2a3, element3 [concrete = constants.%T.as_type.as.Iterate.impl.Next.b00]
// CHECK:STDOUT:   %bound_method.loc38_19.3: <bound method> = bound_method %c.ref, %impl.elem3
// CHECK:STDOUT:   %facet_value.loc38_19.3: %CppRange_where.type.d682d6.1 = facet_value constants.%ConstRange, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.868, constants.%Copy.impl_witness.e6c, constants.%custom_witness.5d5, constants.%custom_witness.1fc, constants.%DestroyImpl.impl_witness.b32, constants.%custom_witness.ed9, constants.%custom_witness.393, constants.%custom_witness.821) [concrete = constants.%facet_value.731]
// CHECK:STDOUT:   %.loc38_19.3: %CppRange_where.type.d682d6.1 = converted constants.%ConstRange, %facet_value.loc38_19.3 [concrete = constants.%facet_value.731]
// CHECK:STDOUT:   %facet_value.loc38_19.4: %CppRange_where.type.d682d6.1 = facet_value constants.%ConstRange, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.868, constants.%Copy.impl_witness.e6c, constants.%custom_witness.5d5, constants.%custom_witness.1fc, constants.%DestroyImpl.impl_witness.b32, constants.%custom_witness.ed9, constants.%custom_witness.393, constants.%custom_witness.821) [concrete = constants.%facet_value.731]
// CHECK:STDOUT:   %.loc38_19.4: %CppRange_where.type.d682d6.1 = converted constants.%ConstRange, %facet_value.loc38_19.4 [concrete = constants.%facet_value.731]
// CHECK:STDOUT:   %specific_fn.loc38_19.2: <specific function> = specific_function %impl.elem3, @T.as_type.as.Iterate.impl.Next(constants.%facet_value.731) [concrete = constants.%T.as_type.as.Iterate.impl.Next.specific_fn]
// CHECK:STDOUT:   %bound_method.loc38_19.4: <bound method> = bound_method %c.ref, %specific_fn.loc38_19.2
// CHECK:STDOUT:   %.loc38_19.5: ref %Optional.d01 = temporary_storage
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.call: init %Optional.d01 to %.loc38_19.5 = call %bound_method.loc38_19.4(%c.ref, %var)
// CHECK:STDOUT:   %.loc38_19.6: ref %Optional.d01 = temporary %.loc38_19.5, %T.as_type.as.Iterate.impl.Next.call
// CHECK:STDOUT:   %.loc38_19.7: %Optional.HasValue.type.4ec = specific_constant imports.%Core.import_ref.4d4, @Optional(constants.%OptionalStorage.facet.040) [concrete = constants.%Optional.HasValue.d29]
// CHECK:STDOUT:   %HasValue.ref: %Optional.HasValue.type.4ec = name_ref HasValue, %.loc38_19.7 [concrete = constants.%Optional.HasValue.d29]
// CHECK:STDOUT:   %Optional.HasValue.bound: <bound method> = bound_method %.loc38_19.6, %HasValue.ref
// CHECK:STDOUT:   %Optional.HasValue.specific_fn: <specific function> = specific_function %HasValue.ref, @Optional.HasValue(constants.%OptionalStorage.facet.040) [concrete = constants.%Optional.HasValue.specific_fn.6ca]
// CHECK:STDOUT:   %bound_method.loc38_19.5: <bound method> = bound_method %.loc38_19.6, %Optional.HasValue.specific_fn
// CHECK:STDOUT:   %.loc38_19.8: %Optional.d01 = acquire_value %.loc38_19.6
// CHECK:STDOUT:   %Optional.HasValue.call: init bool = call %bound_method.loc38_19.5(%.loc38_19.8)
// CHECK:STDOUT:   %.loc38_19.9: bool = value_of_initializer %Optional.HasValue.call
// CHECK:STDOUT:   %.loc38_19.10: bool = converted %Optional.HasValue.call, %.loc38_19.9
// CHECK:STDOUT:   if %.loc38_19.10 br !for.body else br !for.done
// CHECK:STDOUT:
// CHECK:STDOUT: !for.body:
// CHECK:STDOUT:   %.loc38_19.11: %Optional.Get.type.53f = specific_constant imports.%Core.import_ref.059, @Optional(constants.%OptionalStorage.facet.040) [concrete = constants.%Optional.Get.cb5]
// CHECK:STDOUT:   %Get.ref: %Optional.Get.type.53f = name_ref Get, %.loc38_19.11 [concrete = constants.%Optional.Get.cb5]
// CHECK:STDOUT:   %Optional.Get.bound: <bound method> = bound_method %.loc38_19.6, %Get.ref
// CHECK:STDOUT:   %Optional.Get.specific_fn: <specific function> = specific_function %Get.ref, @Optional.Get(constants.%OptionalStorage.facet.040) [concrete = constants.%Optional.Get.specific_fn.e9a]
// CHECK:STDOUT:   %bound_method.loc38_19.6: <bound method> = bound_method %.loc38_19.6, %Optional.Get.specific_fn
// CHECK:STDOUT:   %.loc38_19.12: %Optional.d01 = acquire_value %.loc38_19.6
// CHECK:STDOUT:   %Optional.Get.call: init %f64.978 = call %bound_method.loc38_19.6(%.loc38_19.12)
// CHECK:STDOUT:   %.loc38_19.13: %f64.978 = value_of_initializer %Optional.Get.call
// CHECK:STDOUT:   %.loc38_19.14: %f64.978 = converted %Optional.Get.call, %.loc38_19.13
// CHECK:STDOUT:   %f64.loc38: type = type_literal constants.%f64.978 [concrete = constants.%f64.978]
// CHECK:STDOUT:   %i: %f64.978 = wrapper_binding i, %.loc38_19.14
// CHECK:STDOUT:   %sum.ref: ref %f64.978 = name_ref sum, %sum
// CHECK:STDOUT:   %i.ref: %f64.978 = name_ref i, %i
// CHECK:STDOUT:   %impl.elem0.loc39: %.74b = impl_witness_access constants.%AddAssignWith.impl_witness.614, element0 [concrete = constants.%Float.as.AddAssignWith.impl.Op.cd7]
// CHECK:STDOUT:   %bound_method.loc39_9.1: <bound method> = bound_method %sum.ref, %impl.elem0.loc39
// CHECK:STDOUT:   %specific_fn.loc39: <specific function> = specific_function %impl.elem0.loc39, @Float.as.AddAssignWith.impl.Op(constants.%int_64) [concrete = constants.%Float.as.AddAssignWith.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc39_9.2: <bound method> = bound_method %sum.ref, %specific_fn.loc39
// CHECK:STDOUT:   %Float.as.AddAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc39_9.2(%sum.ref, %i.ref)
// CHECK:STDOUT:   %facet_value.loc38_19.5: %facet_type.8a3 = facet_value constants.%Optional.d01, (constants.%custom_witness.b80ef6.8, constants.%DestroyImpl.impl_witness.06a) [concrete = constants.%facet_value.bb3]
// CHECK:STDOUT:   %.loc38_19.15: %facet_type.8a3 = converted constants.%Optional.d01, %facet_value.loc38_19.5 [concrete = constants.%facet_value.bb3]
// CHECK:STDOUT:   %facet_value.loc38_19.6: %facet_type.8a3 = facet_value constants.%Optional.d01, (constants.%custom_witness.b80ef6.8, constants.%DestroyImpl.impl_witness.06a) [concrete = constants.%facet_value.bb3]
// CHECK:STDOUT:   %.loc38_19.16: %facet_type.8a3 = converted constants.%Optional.d01, %facet_value.loc38_19.6 [concrete = constants.%facet_value.bb3]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc38_19.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc38_19.6)
// CHECK:STDOUT:   br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT: !for.done:
// CHECK:STDOUT:   %facet_value.loc38_19.7: %facet_type.8a3 = facet_value constants.%Optional.d01, (constants.%custom_witness.b80ef6.8, constants.%DestroyImpl.impl_witness.06a) [concrete = constants.%facet_value.bb3]
// CHECK:STDOUT:   %.loc38_19.17: %facet_type.8a3 = converted constants.%Optional.d01, %facet_value.loc38_19.7 [concrete = constants.%facet_value.bb3]
// CHECK:STDOUT:   %facet_value.loc38_19.8: %facet_type.8a3 = facet_value constants.%Optional.d01, (constants.%custom_witness.b80ef6.8, constants.%DestroyImpl.impl_witness.06a) [concrete = constants.%facet_value.bb3]
// CHECK:STDOUT:   %.loc38_19.18: %facet_type.8a3 = converted constants.%Optional.d01, %facet_value.loc38_19.8 [concrete = constants.%facet_value.bb3]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc38_19.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc38_19.6)
// CHECK:STDOUT:   %facet_value.loc38_19.9: %facet_type.8a3 = facet_value constants.%tuple.type.005, (constants.%custom_witness.b80ef6.9, constants.%DestroyImpl.impl_witness.7493) [concrete = constants.%facet_value.6a6]
// CHECK:STDOUT:   %.loc38_19.19: %facet_type.8a3 = converted constants.%tuple.type.005, %facet_value.loc38_19.9 [concrete = constants.%facet_value.6a6]
// CHECK:STDOUT:   %facet_value.loc38_19.10: %facet_type.8a3 = facet_value constants.%tuple.type.005, (constants.%custom_witness.b80ef6.9, constants.%DestroyImpl.impl_witness.7493) [concrete = constants.%facet_value.6a6]
// CHECK:STDOUT:   %.loc38_19.20: %facet_type.8a3 = converted constants.%tuple.type.005, %facet_value.loc38_19.10 [concrete = constants.%facet_value.6a6]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc38_19.3: init %empty_tuple.type = call %SelfDestruct.specific_fn.3(%var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestIterate(constants.%R) {
// CHECK:STDOUT:   %R.patt.loc20_17.2 => constants.%R.patt
// CHECK:STDOUT:   %R.loc20_17.1 => constants.%R
// CHECK:STDOUT:   %R.as_type.loc20_61.1 => constants.%R.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.bac
// CHECK:STDOUT:   %r.param_patt.loc20_59.2 => constants.%r.param_patt.954
// CHECK:STDOUT:   %r.patt.loc20_59.2 => constants.%r.patt.41e
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestIterate(constants.%facet_value.062) {
// CHECK:STDOUT:   %R.patt.loc20_17.2 => constants.%R.patt
// CHECK:STDOUT:   %R.loc20_17.1 => constants.%facet_value.062
// CHECK:STDOUT:   %R.as_type.loc20_61.1 => constants.%ConstRange
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.025
// CHECK:STDOUT:   %r.param_patt.loc20_59.2 => constants.%r.param_patt.a73
// CHECK:STDOUT:   %r.patt.loc20_59.2 => constants.%r.patt.0b5
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc20 => constants.%complete_type.357
// CHECK:STDOUT:   %Iterate.lookup_impl_witness => constants.%Iterate.impl_witness.2a3
// CHECK:STDOUT:   %Iterate.facet.loc24_19.5 => constants.%Iterate.facet.7d5
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type => constants.%Iterate.WithSelf.NewCursor.type.0fc
// CHECK:STDOUT:   %.loc24_19.23 => constants.%.35a
// CHECK:STDOUT:   %impl.elem2.loc24_19.2 => constants.%T.as_type.as.Iterate.impl.NewCursor.266
// CHECK:STDOUT:   %specific_impl_fn.loc24_19.3 => constants.%T.as_type.as.Iterate.impl.NewCursor.specific_fn
// CHECK:STDOUT:   %impl.elem1 => constants.%facet_value.6a6
// CHECK:STDOUT:   %as_type => constants.%tuple.type.005
// CHECK:STDOUT:   %require_complete.1 => constants.%complete_type.320
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type => constants.%Iterate.WithSelf.Next.type.170
// CHECK:STDOUT:   %.loc24_19.24 => constants.%.e4c
// CHECK:STDOUT:   %impl.elem3.loc24_19.2 => constants.%T.as_type.as.Iterate.impl.Next.b00
// CHECK:STDOUT:   %specific_impl_fn.loc24_19.4 => constants.%T.as_type.as.Iterate.impl.Next.specific_fn
// CHECK:STDOUT:   %SelfDestruct.specific_fn.5 => constants.%SelfDestruct.specific_fn.b5d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_adl_returns_same_types_mutable.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %Iterate.type: type = facet_type <@Iterate> [concrete]
// CHECK:STDOUT:   %OptionalStorage.type: type = facet_type <@OptionalStorage> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %facet_type.d61: type = facet_type <@SubobjectDestroy & @DestroyImpl & @Copy> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %T.220: %OptionalStorage.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Optional.Get.type.d97: type = fn_type @Optional.Get, @Optional(%T.220) [symbolic]
// CHECK:STDOUT:   %Optional.Get.3e9: %Optional.Get.type.d97 = struct_value () [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.type.cef: type = fn_type @Optional.HasValue, @Optional(%T.220) [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.a74: %Optional.HasValue.type.cef = struct_value () [symbolic]
// CHECK:STDOUT:   %facet_type.8a3: type = facet_type <@SubobjectDestroy & @DestroyImpl> [concrete]
// CHECK:STDOUT:   %T.9ab: %facet_type.d61 = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.19c: type = facet_access_type %T.9ab [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.lookup_impl_witness.855: <witness> = lookup_impl_witness %T.9ab, @SubobjectDestroy [symbolic]
// CHECK:STDOUT:   %DestroyImpl.lookup_impl_witness.ff9: <witness> = lookup_impl_witness %T.9ab, @DestroyImpl [symbolic]
// CHECK:STDOUT:   %facet_value.b35: %facet_type.8a3 = facet_value %T.as_type.19c, (%SubobjectDestroy.lookup_impl_witness.855, %DestroyImpl.lookup_impl_witness.ff9) [symbolic]
// CHECK:STDOUT:   %MaybeUnformed.413: type = class_type @MaybeUnformed, @MaybeUnformed(%facet_value.b35) [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.a6b: type = class_type @DefaultOptionalStorage, @DefaultOptionalStorage(%T.9ab) [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.elem.94e: type = unbound_element_type %DefaultOptionalStorage.a6b, bool [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.elem.c3f: type = unbound_element_type %DefaultOptionalStorage.a6b, %MaybeUnformed.413 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Get.type.b32: type = fn_type @T.as_type.as.OptionalStorage.impl.Get, @T.as_type.as.OptionalStorage.impl(%T.9ab) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Get.8b0: %T.as_type.as.OptionalStorage.impl.Get.type.b32 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Has.type.8e5: type = fn_type @T.as_type.as.OptionalStorage.impl.Has, @T.as_type.as.OptionalStorage.impl(%T.9ab) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Has.2d5: %T.as_type.as.OptionalStorage.impl.Has.type.8e5 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Some.type.61d: type = fn_type @T.as_type.as.OptionalStorage.impl.Some, @T.as_type.as.OptionalStorage.impl(%T.9ab) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Some.ba7: %T.as_type.as.OptionalStorage.impl.Some.type.61d = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.None.type.a9f: type = fn_type @T.as_type.as.OptionalStorage.impl.None, @T.as_type.as.OptionalStorage.impl(%T.9ab) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.None.083: %T.as_type.as.OptionalStorage.impl.None.type.a9f = struct_value () [symbolic]
// CHECK:STDOUT:   %N.fe9: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.273: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.5b5: %T.as_type.as.DestroyImpl.impl.Op.type.273 = struct_value () [symbolic]
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
// CHECK:STDOUT:   %SubobjectDestroy.lookup_impl_witness.dc9: <witness> = lookup_impl_witness %tuple.type.1ba, @SubobjectDestroy [symbolic]
// CHECK:STDOUT:   %DestroyImpl.lookup_impl_witness.504: <witness> = lookup_impl_witness %tuple.type.1ba, @DestroyImpl [symbolic]
// CHECK:STDOUT:   %facet_value.bf6: %facet_type.8a3 = facet_value %tuple.type.1ba, (%SubobjectDestroy.lookup_impl_witness.dc9, %DestroyImpl.lookup_impl_witness.504) [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.lookup_impl_witness.b15da7.2: <witness> = lookup_impl_witness %impl.elem0.493765.2, @SubobjectDestroy [symbolic]
// CHECK:STDOUT:   %DestroyImpl.lookup_impl_witness.4e7f44.2: <witness> = lookup_impl_witness %impl.elem0.493765.2, @DestroyImpl [symbolic]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.167: <witness> = lookup_impl_witness %impl.elem0.493765.2, @Copy [symbolic]
// CHECK:STDOUT:   %facet_value.9da: %facet_type.d61 = facet_value %impl.elem0.493765.2, (%SubobjectDestroy.lookup_impl_witness.b15da7.2, %DestroyImpl.lookup_impl_witness.4e7f44.2, %Copy.lookup_impl_witness.167) [symbolic]
// CHECK:STDOUT:   %SelfDestruct.type: type = fn_type @SelfDestruct [concrete]
// CHECK:STDOUT:   %SelfDestruct: %SelfDestruct.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete]
// CHECK:STDOUT:   %f64.978: type = class_type @Float, @Float(%int_64) [concrete]
// CHECK:STDOUT:   %pattern_type.522: type = pattern_type %f64.978 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.2: type = fn_type @SubobjectDestroy.WithSelf.Op.loc22 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.2: %SubobjectDestroy.WithSelf.Op.type.6870fd.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.2: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.2), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.76ce: %SubobjectDestroy.type = facet_value %f64.978, (%custom_witness.b80ef6.2) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.868: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.76ce) [concrete]
// CHECK:STDOUT:   %Float.as.Copy.impl.Op.type.afa: type = fn_type @Float.as.Copy.impl.Op, @Float.as.Copy.impl(%N.fe9) [symbolic]
// CHECK:STDOUT:   %Float.as.Copy.impl.Op.e99: %Float.as.Copy.impl.Op.type.afa = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.e6c: <witness> = impl_witness imports.%Copy.impl_witness_table.f03, @Float.as.Copy.impl(%int_64) [concrete]
// CHECK:STDOUT:   %facet_value.83c: %facet_type.d61 = facet_value %f64.978, (%custom_witness.b80ef6.2, %DestroyImpl.impl_witness.868, %Copy.impl_witness.e6c) [concrete]
// CHECK:STDOUT:   %Iterate.lookup_impl_witness.1d8: <witness> = lookup_impl_witness %.Self.500, @Iterate [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.8a4: %facet_type.d61 = impl_witness_access %Iterate.lookup_impl_witness.1d8, element0 [symbolic_self]
// CHECK:STDOUT:   %Iterate_where.type.cd5: type = facet_type <@Iterate where %impl.elem0.8a4 = %facet_value.83c> [concrete]
// CHECK:STDOUT:   %pattern_type.1b8: type = pattern_type %Iterate_where.type.cd5 [concrete]
// CHECK:STDOUT:   %R.patt: %pattern_type.1b8 = symbolic_binding_pattern R, 0 [symbolic]
// CHECK:STDOUT:   %R: %Iterate_where.type.cd5 = symbolic_binding R, 0 [symbolic]
// CHECK:STDOUT:   %R.as_type: type = facet_access_type %R [symbolic]
// CHECK:STDOUT:   %pattern_type.bac: type = pattern_type %R.as_type [symbolic]
// CHECK:STDOUT:   %r.param_patt.954: %pattern_type.bac = value_param_pattern [symbolic]
// CHECK:STDOUT:   %r.patt.41e: %pattern_type.bac = wrapper_binding_pattern r, %r.param_patt.954 [symbolic]
// CHECK:STDOUT:   %i.patt: %pattern_type.522 = value_binding_pattern i [concrete]
// CHECK:STDOUT:   %Iterate.lookup_impl_witness.935: <witness> = lookup_impl_witness %R, @Iterate [symbolic]
// CHECK:STDOUT:   %Iterate.facet.c16: %Iterate.type = facet_value %R.as_type, (%Iterate.lookup_impl_witness.935) [symbolic]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.b97: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.c16) [symbolic]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.ff4: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.c16) [symbolic]
// CHECK:STDOUT:   %.e93: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.b97, %Iterate.facet.c16 [symbolic]
// CHECK:STDOUT:   %impl.elem2.7a3: %.e93 = impl_witness_access %Iterate.lookup_impl_witness.935, element2 [symbolic]
// CHECK:STDOUT:   %impl.elem1.c65: %facet_type.8a3 = impl_witness_access %Iterate.lookup_impl_witness.935, element1 [symbolic]
// CHECK:STDOUT:   %as_type.b5f: type = facet_access_type %impl.elem1.c65 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.46d: <specific function> = specific_impl_function %impl.elem2.7a3, @Iterate.WithSelf.NewCursor(%Iterate.facet.c16) [symbolic]
// CHECK:STDOUT:   %.d61: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.ff4, %Iterate.facet.c16 [symbolic]
// CHECK:STDOUT:   %impl.elem3.09c: %.d61 = impl_witness_access %Iterate.lookup_impl_witness.935, element3 [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.cd2: type = class_type @DefaultOptionalStorage, @DefaultOptionalStorage(%facet_value.83c) [concrete]
// CHECK:STDOUT:   %OptionalStorage.impl_witness.310: <witness> = impl_witness imports.%OptionalStorage.impl_witness_table.968, @T.as_type.as.OptionalStorage.impl(%facet_value.83c) [concrete]
// CHECK:STDOUT:   %OptionalStorage.facet.040: %OptionalStorage.type = facet_value %f64.978, (%OptionalStorage.impl_witness.310) [concrete]
// CHECK:STDOUT:   %Optional.d01: type = class_type @Optional, @Optional(%OptionalStorage.facet.040) [concrete]
// CHECK:STDOUT:   %specific_impl_fn.8a2: <specific function> = specific_impl_function %impl.elem3.09c, @Iterate.WithSelf.Next(%Iterate.facet.c16) [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.type.4ec: type = fn_type @Optional.HasValue, @Optional(%OptionalStorage.facet.040) [concrete]
// CHECK:STDOUT:   %Optional.HasValue.d29: %Optional.HasValue.type.4ec = struct_value () [concrete]
// CHECK:STDOUT:   %Optional.Get.type.53f: type = fn_type @Optional.Get, @Optional(%OptionalStorage.facet.040) [concrete]
// CHECK:STDOUT:   %Optional.Get.cb5: %Optional.Get.type.53f = struct_value () [concrete]
// CHECK:STDOUT:   %facet_value.348: %facet_type.8a3 = facet_value %f64.978, (%custom_witness.b80ef6.2, %DestroyImpl.impl_witness.868) [concrete]
// CHECK:STDOUT:   %MaybeUnformed.609: type = class_type @MaybeUnformed, @MaybeUnformed(%facet_value.348) [concrete]
// CHECK:STDOUT:   %.b81: type = maybe_unformed_type %f64.978 [concrete]
// CHECK:STDOUT:   %DefaultOptionalStorage.elem.fb9: type = unbound_element_type %DefaultOptionalStorage.cd2, %MaybeUnformed.609 [concrete]
// CHECK:STDOUT:   %DefaultOptionalStorage.elem.81c: type = unbound_element_type %DefaultOptionalStorage.cd2, bool [concrete]
// CHECK:STDOUT:   %struct_type.value.has_value.e85: type = struct_type {.value: %MaybeUnformed.609, .has_value: bool} [concrete]
// CHECK:STDOUT:   %Optional.HasValue.specific_fn.6ca: <specific function> = specific_function %Optional.HasValue.d29, @Optional.HasValue(%OptionalStorage.facet.040) [concrete]
// CHECK:STDOUT:   %Optional.Get.specific_fn.e9a: <specific function> = specific_function %Optional.Get.cb5, @Optional.Get(%OptionalStorage.facet.040) [concrete]
// CHECK:STDOUT:   %AddAssignWith.type.8e8: type = facet_type <@AddAssignWith, @AddAssignWith(%f64.978)> [concrete]
// CHECK:STDOUT:   %Float.as.AddAssignWith.impl.Op.type.bd0: type = fn_type @Float.as.AddAssignWith.impl.Op, @Float.as.AddAssignWith.impl(%N.fe9) [symbolic]
// CHECK:STDOUT:   %Float.as.AddAssignWith.impl.Op.f48: %Float.as.AddAssignWith.impl.Op.type.bd0 = struct_value () [symbolic]
// CHECK:STDOUT:   %AddAssignWith.impl_witness.614: <witness> = impl_witness imports.%AddAssignWith.impl_witness_table, @Float.as.AddAssignWith.impl(%int_64) [concrete]
// CHECK:STDOUT:   %Float.as.AddAssignWith.impl.Op.type.223: type = fn_type @Float.as.AddAssignWith.impl.Op, @Float.as.AddAssignWith.impl(%int_64) [concrete]
// CHECK:STDOUT:   %Float.as.AddAssignWith.impl.Op.cd7: %Float.as.AddAssignWith.impl.Op.type.223 = struct_value () [concrete]
// CHECK:STDOUT:   %AddAssignWith.facet: %AddAssignWith.type.8e8 = facet_value %f64.978, (%AddAssignWith.impl_witness.614) [concrete]
// CHECK:STDOUT:   %AddAssignWith.WithSelf.Op.type.58f: type = fn_type @AddAssignWith.WithSelf.Op, @AddAssignWith.WithSelf(%f64.978, %AddAssignWith.facet) [concrete]
// CHECK:STDOUT:   %.74b: type = fn_type_with_self_type %AddAssignWith.WithSelf.Op.type.58f, %AddAssignWith.facet [concrete]
// CHECK:STDOUT:   %Float.as.AddAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Float.as.AddAssignWith.impl.Op.cd7, @Float.as.AddAssignWith.impl.Op(%int_64) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.4: type = fn_type @SubobjectDestroy.WithSelf.Op.loc26_19.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.4: %SubobjectDestroy.WithSelf.Op.type.6870fd.4 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.4: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.4), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.5: type = fn_type @SubobjectDestroy.WithSelf.Op.loc26_19.3 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.5: %SubobjectDestroy.WithSelf.Op.type.6870fd.5 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.5: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.5), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.a87: %SubobjectDestroy.type = facet_value bool, (%custom_witness.b80ef6.5) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.deb: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.a87) [concrete]
// CHECK:STDOUT:   %facet_value.a1b: %facet_type.8a3 = facet_value bool, (%custom_witness.b80ef6.5, %DestroyImpl.impl_witness.deb) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.3e6: %SubobjectDestroy.type = facet_value %MaybeUnformed.609, (%custom_witness.b80ef6.4) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.4ed: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.3e6) [concrete]
// CHECK:STDOUT:   %facet_value.40c: %facet_type.8a3 = facet_value %MaybeUnformed.609, (%custom_witness.b80ef6.4, %DestroyImpl.impl_witness.4ed) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.8: type = fn_type @SubobjectDestroy.WithSelf.Op.loc26_19.6 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.8: %SubobjectDestroy.WithSelf.Op.type.6870fd.8 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.8: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.8), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.76c9: %SubobjectDestroy.type = facet_value %Optional.d01, (%custom_witness.b80ef6.8) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.06a: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.76c9) [concrete]
// CHECK:STDOUT:   %facet_value.bb3: %facet_type.8a3 = facet_value %Optional.d01, (%custom_witness.b80ef6.8, %DestroyImpl.impl_witness.06a) [concrete]
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
// CHECK:STDOUT:   %custom_witness.c50: <witness> = custom_witness (%f64.978, %Iterator.Op.ac6aad.1), @CppUnsafeDeref [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.b9731e.2: type = fn_type @Iterator.Op.2 [concrete]
// CHECK:STDOUT:   %Iterator.Op.ac6aad.2: %Iterator.Op.type.b9731e.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.f01: <witness> = custom_witness (%Iterator.Op.ac6aad.2), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.700: %SubobjectDestroy.type = facet_value %Iterator, (%custom_witness.f01) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.0af: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.700) [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.b9731e.3: type = fn_type @Iterator.Op.3 [concrete]
// CHECK:STDOUT:   %Iterator.Op.ac6aad.3: %Iterator.Op.type.b9731e.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.491: <witness> = custom_witness (%Iterator.Op.ac6aad.3), @Inc [concrete]
// CHECK:STDOUT:   %Equal.type: type = fn_type @Equal [concrete]
// CHECK:STDOUT:   %Equal: %Equal.type = struct_value () [concrete]
// CHECK:STDOUT:   %NotEqual.type: type = fn_type @NotEqual [concrete]
// CHECK:STDOUT:   %NotEqual: %NotEqual.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.ed8: <witness> = custom_witness (%Equal, %NotEqual), @EqWith, @EqWith(%Iterator) [concrete]
// CHECK:STDOUT:   %facet_value.86b: %CppRange_where.type.d682d6.1 = facet_value %MutableRange, (%custom_witness.b80ef6.2, %DestroyImpl.impl_witness.868, %Copy.impl_witness.e6c, %custom_witness.902, %custom_witness.f01, %DestroyImpl.impl_witness.0af, %custom_witness.c50, %custom_witness.491, %custom_witness.ed8) [concrete]
// CHECK:STDOUT:   %tuple.type.dd0: type = tuple_type (%Iterator, %Iterator) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.9: type = fn_type @SubobjectDestroy.WithSelf.Op.3 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.9: %SubobjectDestroy.WithSelf.Op.type.6870fd.9 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.9: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.9), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.0de: %SubobjectDestroy.type = facet_value %tuple.type.dd0, (%custom_witness.b80ef6.9) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.6b2: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.0de) [concrete]
// CHECK:STDOUT:   %complete_type.843: <witness> = complete_type_witness %tuple.type.dd0 [concrete]
// CHECK:STDOUT:   %facet_value.b59: %facet_type.8a3 = facet_value %tuple.type.dd0, (%custom_witness.b80ef6.9, %DestroyImpl.impl_witness.6b2) [concrete]
// CHECK:STDOUT:   %Iterate.impl_witness.43e: <witness> = impl_witness imports.%Iterate.impl_witness_table.121, @T.as_type.as.Iterate.impl(%facet_value.86b) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.type.d70: type = fn_type @T.as_type.as.Iterate.impl.NewCursor, @T.as_type.as.Iterate.impl(%facet_value.86b) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.5df: %T.as_type.as.Iterate.impl.NewCursor.type.d70 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.type.dba: type = fn_type @T.as_type.as.Iterate.impl.Next, @T.as_type.as.Iterate.impl(%facet_value.86b) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.b6d: %T.as_type.as.Iterate.impl.Next.type.dba = struct_value () [concrete]
// CHECK:STDOUT:   %Iterate.facet.b6a: %Iterate.type = facet_value %MutableRange, (%Iterate.impl_witness.43e) [concrete]
// CHECK:STDOUT:   %facet_value.1f1: %Iterate_where.type.cd5 = facet_value %MutableRange, (%Iterate.impl_witness.43e) [concrete]
// CHECK:STDOUT:   %r.param_patt.aeb: %pattern_type.992 = value_param_pattern [concrete]
// CHECK:STDOUT:   %r.patt.892: %pattern_type.992 = wrapper_binding_pattern r, %r.param_patt.aeb [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.2d6: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.b6a) [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.bcc: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.b6a) [concrete]
// CHECK:STDOUT:   %.5ac: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.2d6, %Iterate.facet.b6a [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.specific_fn: <specific function> = specific_function %T.as_type.as.Iterate.impl.NewCursor.5df, @T.as_type.as.Iterate.impl.NewCursor(%facet_value.86b) [concrete]
// CHECK:STDOUT:   %.854: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.bcc, %Iterate.facet.b6a [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.specific_fn: <specific function> = specific_function %T.as_type.as.Iterate.impl.Next.b6d, @T.as_type.as.Iterate.impl.Next(%facet_value.86b) [concrete]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.2f3: <specific function> = specific_function %SelfDestruct, @SelfDestruct(%facet_value.b59) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.4d4: @Optional.%Optional.HasValue.type (%Optional.HasValue.type.cef) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.HasValue (constants.%Optional.HasValue.a74)]
// CHECK:STDOUT:   %Core.import_ref.059: @Optional.%Optional.Get.type (%Optional.Get.type.d97) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.Get (constants.%Optional.Get.3e9)]
// CHECK:STDOUT:   %Core.import_ref.7bb: type = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%DefaultOptionalStorage (constants.%DefaultOptionalStorage.a6b)]
// CHECK:STDOUT:   %Core.import_ref.61e: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.None.type (%T.as_type.as.OptionalStorage.impl.None.type.a9f) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.None (constants.%T.as_type.as.OptionalStorage.impl.None.083)]
// CHECK:STDOUT:   %Core.import_ref.ca1: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Some.type (%T.as_type.as.OptionalStorage.impl.Some.type.61d) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Some (constants.%T.as_type.as.OptionalStorage.impl.Some.ba7)]
// CHECK:STDOUT:   %Core.import_ref.6ba4: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Has.type (%T.as_type.as.OptionalStorage.impl.Has.type.8e5) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Has (constants.%T.as_type.as.OptionalStorage.impl.Has.2d5)]
// CHECK:STDOUT:   %Core.import_ref.73d: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Get.type (%T.as_type.as.OptionalStorage.impl.Get.type.b32) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Get (constants.%T.as_type.as.OptionalStorage.impl.Get.8b0)]
// CHECK:STDOUT:   %Core.import_ref.778 = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], unloaded
// CHECK:STDOUT:   %OptionalStorage.impl_witness_table.968 = impl_witness_table (%Core.import_ref.7bb, %Core.import_ref.61e, %Core.import_ref.ca1, %Core.import_ref.6ba4, %Core.import_ref.73d, %Core.import_ref.778), @T.as_type.as.OptionalStorage.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.146b: @DefaultOptionalStorage.%DefaultOptionalStorage.elem.1 (%DefaultOptionalStorage.elem.c3f) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [concrete = %field_decl.31e]
// CHECK:STDOUT:   %Core.import_ref.f1da: @DefaultOptionalStorage.%DefaultOptionalStorage.elem.2 (%DefaultOptionalStorage.elem.94e) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [concrete = %field_decl.431]
// CHECK:STDOUT:   %Core.import_ref.14b: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.273) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.5b5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.14b), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.d6c: %facet_type.d61 = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.Iterate.impl.%facet_value.1 (constants.%facet_value.9da)]
// CHECK:STDOUT:   %Core.import_ref.2ac: %facet_type.8a3 = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.Iterate.impl.%facet_value.2 (constants.%facet_value.bf6)]
// CHECK:STDOUT:   %Core.import_ref.fd1a: @T.as_type.as.Iterate.impl.%T.as_type.as.Iterate.impl.NewCursor.type (%T.as_type.as.Iterate.impl.NewCursor.type.953) = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.Iterate.impl.%T.as_type.as.Iterate.impl.NewCursor (constants.%T.as_type.as.Iterate.impl.NewCursor.544)]
// CHECK:STDOUT:   %Core.import_ref.273: @T.as_type.as.Iterate.impl.%T.as_type.as.Iterate.impl.Next.type (%T.as_type.as.Iterate.impl.Next.type.b36) = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.Iterate.impl.%T.as_type.as.Iterate.impl.Next (constants.%T.as_type.as.Iterate.impl.Next.d6b)]
// CHECK:STDOUT:   %Iterate.impl_witness_table.121 = impl_witness_table (%Core.import_ref.d6c, %Core.import_ref.2ac, %Core.import_ref.fd1a, %Core.import_ref.273), @T.as_type.as.Iterate.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.bd6: @Float.as.Copy.impl.%Float.as.Copy.impl.Op.type (%Float.as.Copy.impl.Op.type.afa) = import_ref Core//prelude/types/float, loc{{\d+_\d+}}, loaded [symbolic = @Float.as.Copy.impl.%Float.as.Copy.impl.Op (constants.%Float.as.Copy.impl.Op.e99)]
// CHECK:STDOUT:   %Copy.impl_witness_table.f03 = impl_witness_table (%Core.import_ref.bd6), @Float.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.4e0: @Float.as.AddAssignWith.impl.%Float.as.AddAssignWith.impl.Op.type (%Float.as.AddAssignWith.impl.Op.type.bd0) = import_ref Core//prelude/types/float, loc{{\d+_\d+}}, loaded [symbolic = @Float.as.AddAssignWith.impl.%Float.as.AddAssignWith.impl.Op (constants.%Float.as.AddAssignWith.impl.Op.f48)]
// CHECK:STDOUT:   %AddAssignWith.impl_witness_table = impl_witness_table (%Core.import_ref.4e0), @Float.as.AddAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.b0d = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %field_decl.431: @DefaultOptionalStorage.%DefaultOptionalStorage.elem.2 (%DefaultOptionalStorage.elem.94e) = field_decl has_value, element1, %Core.import_ref.b0d in [concrete] {}
// CHECK:STDOUT:   %Core.import_ref.164c = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %field_decl.31e: @DefaultOptionalStorage.%DefaultOptionalStorage.elem.1 (%DefaultOptionalStorage.elem.c3f) = field_decl value, element0, %Core.import_ref.164c in [concrete] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @TestIterate(%R.loc22_17.2: %Iterate_where.type.cd5) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Iterate.lookup_impl_witness: <witness> = lookup_impl_witness %R.loc22_17.1, @Iterate [symbolic = %Iterate.lookup_impl_witness (constants.%Iterate.lookup_impl_witness.935)]
// CHECK:STDOUT:   %Iterate.facet.loc26_19.5: %Iterate.type = facet_value %R.as_type.loc22_61.1, (%Iterate.lookup_impl_witness) [symbolic = %Iterate.facet.loc26_19.5 (constants.%Iterate.facet.c16)]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.loc26_19.5) [symbolic = %Iterate.WithSelf.NewCursor.type (constants.%Iterate.WithSelf.NewCursor.type.b97)]
// CHECK:STDOUT:   %.loc26_19.23: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type, %Iterate.facet.loc26_19.5 [symbolic = %.loc26_19.23 (constants.%.e93)]
// CHECK:STDOUT:   %impl.elem2.loc26_19.2: @TestIterate.%.loc26_19.23 (%.e93) = impl_witness_access %Iterate.lookup_impl_witness, element2 [symbolic = %impl.elem2.loc26_19.2 (constants.%impl.elem2.7a3)]
// CHECK:STDOUT:   %specific_impl_fn.loc26_19.3: <specific function> = specific_impl_function %impl.elem2.loc26_19.2, @Iterate.WithSelf.NewCursor(%Iterate.facet.loc26_19.5) [symbolic = %specific_impl_fn.loc26_19.3 (constants.%specific_impl_fn.46d)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.loc26_19.5) [symbolic = %Iterate.WithSelf.Next.type (constants.%Iterate.WithSelf.Next.type.ff4)]
// CHECK:STDOUT:   %.loc26_19.24: type = fn_type_with_self_type %Iterate.WithSelf.Next.type, %Iterate.facet.loc26_19.5 [symbolic = %.loc26_19.24 (constants.%.d61)]
// CHECK:STDOUT:   %impl.elem3.loc26_19.2: @TestIterate.%.loc26_19.24 (%.d61) = impl_witness_access %Iterate.lookup_impl_witness, element3 [symbolic = %impl.elem3.loc26_19.2 (constants.%impl.elem3.09c)]
// CHECK:STDOUT:   %specific_impl_fn.loc26_19.4: <specific function> = specific_impl_function %impl.elem3.loc26_19.2, @Iterate.WithSelf.Next(%Iterate.facet.loc26_19.5) [symbolic = %specific_impl_fn.loc26_19.4 (constants.%specific_impl_fn.8a2)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%r.param: @TestIterate.%R.as_type.loc22_61.1 (%R.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %i.patt: %pattern_type.522 = value_binding_pattern i [concrete = constants.%i.patt]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %r.ref: @TestIterate.%R.as_type.loc22_61.1 (%R.as_type) = name_ref r, %r
// CHECK:STDOUT:     %impl.elem2.loc26_19.1: @TestIterate.%.loc26_19.23 (%.e93) = impl_witness_access constants.%Iterate.lookup_impl_witness.935, element2 [symbolic = %impl.elem2.loc26_19.2 (constants.%impl.elem2.7a3)]
// CHECK:STDOUT:     %bound_method.loc26_19.1: <bound method> = bound_method %r.ref, %impl.elem2.loc26_19.1
// CHECK:STDOUT:     %Iterate.facet.loc26_19.1: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.935) [symbolic = %Iterate.facet.loc26_19.5 (constants.%Iterate.facet.c16)]
// CHECK:STDOUT:     %.loc26_19.1: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc26_19.1 [symbolic = %Iterate.facet.loc26_19.5 (constants.%Iterate.facet.c16)]
// CHECK:STDOUT:     %Iterate.facet.loc26_19.2: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.935) [symbolic = %Iterate.facet.loc26_19.5 (constants.%Iterate.facet.c16)]
// CHECK:STDOUT:     %.loc26_19.2: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc26_19.2 [symbolic = %Iterate.facet.loc26_19.5 (constants.%Iterate.facet.c16)]
// CHECK:STDOUT:     %specific_impl_fn.loc26_19.1: <specific function> = specific_impl_function %impl.elem2.loc26_19.1, @Iterate.WithSelf.NewCursor(constants.%Iterate.facet.c16) [symbolic = %specific_impl_fn.loc26_19.3 (constants.%specific_impl_fn.46d)]
// CHECK:STDOUT:     %bound_method.loc26_19.2: <bound method> = bound_method %r.ref, %specific_impl_fn.loc26_19.1
// CHECK:STDOUT:     %var: ref @TestIterate.%as_type (%as_type.b5f) = var_storage invalid
// CHECK:STDOUT:     %Iterate.WithSelf.NewCursor.call: init @TestIterate.%as_type (%as_type.b5f) to %var = call %bound_method.loc26_19.2(%r.ref)
// CHECK:STDOUT:     assign %var, %Iterate.WithSelf.NewCursor.call
// CHECK:STDOUT:     br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.next:
// CHECK:STDOUT:     %impl.elem3.loc26_19.1: @TestIterate.%.loc26_19.24 (%.d61) = impl_witness_access constants.%Iterate.lookup_impl_witness.935, element3 [symbolic = %impl.elem3.loc26_19.2 (constants.%impl.elem3.09c)]
// CHECK:STDOUT:     %bound_method.loc26_19.3: <bound method> = bound_method %r.ref, %impl.elem3.loc26_19.1
// CHECK:STDOUT:     %Iterate.facet.loc26_19.3: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.935) [symbolic = %Iterate.facet.loc26_19.5 (constants.%Iterate.facet.c16)]
// CHECK:STDOUT:     %.loc26_19.3: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc26_19.3 [symbolic = %Iterate.facet.loc26_19.5 (constants.%Iterate.facet.c16)]
// CHECK:STDOUT:     %Iterate.facet.loc26_19.4: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.935) [symbolic = %Iterate.facet.loc26_19.5 (constants.%Iterate.facet.c16)]
// CHECK:STDOUT:     %.loc26_19.4: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc26_19.4 [symbolic = %Iterate.facet.loc26_19.5 (constants.%Iterate.facet.c16)]
// CHECK:STDOUT:     %.loc26_19.5: %facet_type.8a3 = converted constants.%as_type.b5f, constants.%impl.elem1.c65 [symbolic = %impl.elem1 (constants.%impl.elem1.c65)]
// CHECK:STDOUT:     %.loc26_19.6: %facet_type.8a3 = converted constants.%as_type.b5f, constants.%impl.elem1.c65 [symbolic = %impl.elem1 (constants.%impl.elem1.c65)]
// CHECK:STDOUT:     %specific_impl_fn.loc26_19.2: <specific function> = specific_impl_function %impl.elem3.loc26_19.1, @Iterate.WithSelf.Next(constants.%Iterate.facet.c16) [symbolic = %specific_impl_fn.loc26_19.4 (constants.%specific_impl_fn.8a2)]
// CHECK:STDOUT:     %bound_method.loc26_19.4: <bound method> = bound_method %r.ref, %specific_impl_fn.loc26_19.2
// CHECK:STDOUT:     %.loc26_19.7: ref %Optional.d01 = temporary_storage
// CHECK:STDOUT:     %Iterate.WithSelf.Next.call: init %Optional.d01 to %.loc26_19.7 = call %bound_method.loc26_19.4(%r.ref, %var)
// CHECK:STDOUT:     %.loc26_19.8: ref %Optional.d01 = temporary %.loc26_19.7, %Iterate.WithSelf.Next.call
// CHECK:STDOUT:     %.loc26_19.9: %Optional.HasValue.type.4ec = specific_constant imports.%Core.import_ref.4d4, @Optional(constants.%OptionalStorage.facet.040) [concrete = constants.%Optional.HasValue.d29]
// CHECK:STDOUT:     %HasValue.ref: %Optional.HasValue.type.4ec = name_ref HasValue, %.loc26_19.9 [concrete = constants.%Optional.HasValue.d29]
// CHECK:STDOUT:     %Optional.HasValue.bound: <bound method> = bound_method %.loc26_19.8, %HasValue.ref
// CHECK:STDOUT:     %Optional.HasValue.specific_fn: <specific function> = specific_function %HasValue.ref, @Optional.HasValue(constants.%OptionalStorage.facet.040) [concrete = constants.%Optional.HasValue.specific_fn.6ca]
// CHECK:STDOUT:     %bound_method.loc26_19.5: <bound method> = bound_method %.loc26_19.8, %Optional.HasValue.specific_fn
// CHECK:STDOUT:     %.loc26_19.10: %Optional.d01 = acquire_value %.loc26_19.8
// CHECK:STDOUT:     %Optional.HasValue.call: init bool = call %bound_method.loc26_19.5(%.loc26_19.10)
// CHECK:STDOUT:     %.loc26_19.11: bool = value_of_initializer %Optional.HasValue.call
// CHECK:STDOUT:     %.loc26_19.12: bool = converted %Optional.HasValue.call, %.loc26_19.11
// CHECK:STDOUT:     if %.loc26_19.12 br !for.body else br !for.done
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.body:
// CHECK:STDOUT:     %.loc26_19.13: %Optional.Get.type.53f = specific_constant imports.%Core.import_ref.059, @Optional(constants.%OptionalStorage.facet.040) [concrete = constants.%Optional.Get.cb5]
// CHECK:STDOUT:     %Get.ref: %Optional.Get.type.53f = name_ref Get, %.loc26_19.13 [concrete = constants.%Optional.Get.cb5]
// CHECK:STDOUT:     %Optional.Get.bound: <bound method> = bound_method %.loc26_19.8, %Get.ref
// CHECK:STDOUT:     %Optional.Get.specific_fn: <specific function> = specific_function %Get.ref, @Optional.Get(constants.%OptionalStorage.facet.040) [concrete = constants.%Optional.Get.specific_fn.e9a]
// CHECK:STDOUT:     %bound_method.loc26_19.6: <bound method> = bound_method %.loc26_19.8, %Optional.Get.specific_fn
// CHECK:STDOUT:     %.loc26_19.14: %Optional.d01 = acquire_value %.loc26_19.8
// CHECK:STDOUT:     %Optional.Get.call: init %f64.978 = call %bound_method.loc26_19.6(%.loc26_19.14)
// CHECK:STDOUT:     %.loc26_19.15: %f64.978 = value_of_initializer %Optional.Get.call
// CHECK:STDOUT:     %.loc26_19.16: %f64.978 = converted %Optional.Get.call, %.loc26_19.15
// CHECK:STDOUT:     %f64.loc26: type = type_literal constants.%f64.978 [concrete = constants.%f64.978]
// CHECK:STDOUT:     %i: %f64.978 = wrapper_binding i, %.loc26_19.16
// CHECK:STDOUT:     %sum.ref: ref %f64.978 = name_ref sum, %sum
// CHECK:STDOUT:     %i.ref: %f64.978 = name_ref i, %i
// CHECK:STDOUT:     %impl.elem0.loc27: %.74b = impl_witness_access constants.%AddAssignWith.impl_witness.614, element0 [concrete = constants.%Float.as.AddAssignWith.impl.Op.cd7]
// CHECK:STDOUT:     %bound_method.loc27_9.1: <bound method> = bound_method %sum.ref, %impl.elem0.loc27
// CHECK:STDOUT:     %specific_fn.loc27: <specific function> = specific_function %impl.elem0.loc27, @Float.as.AddAssignWith.impl.Op(constants.%int_64) [concrete = constants.%Float.as.AddAssignWith.impl.Op.specific_fn]
// CHECK:STDOUT:     %bound_method.loc27_9.2: <bound method> = bound_method %sum.ref, %specific_fn.loc27
// CHECK:STDOUT:     %Float.as.AddAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc27_9.2(%sum.ref, %i.ref)
// CHECK:STDOUT:     %facet_value.loc26_19.1: %facet_type.8a3 = facet_value constants.%Optional.d01, (constants.%custom_witness.b80ef6.8, constants.%DestroyImpl.impl_witness.06a) [concrete = constants.%facet_value.bb3]
// CHECK:STDOUT:     %.loc26_19.17: %facet_type.8a3 = converted constants.%Optional.d01, %facet_value.loc26_19.1 [concrete = constants.%facet_value.bb3]
// CHECK:STDOUT:     %facet_value.loc26_19.2: %facet_type.8a3 = facet_value constants.%Optional.d01, (constants.%custom_witness.b80ef6.8, constants.%DestroyImpl.impl_witness.06a) [concrete = constants.%facet_value.bb3]
// CHECK:STDOUT:     %.loc26_19.18: %facet_type.8a3 = converted constants.%Optional.d01, %facet_value.loc26_19.2 [concrete = constants.%facet_value.bb3]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %SelfDestruct.call.loc26_19.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc26_19.8)
// CHECK:STDOUT:     br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.done:
// CHECK:STDOUT:     %facet_value.loc26_19.3: %facet_type.8a3 = facet_value constants.%Optional.d01, (constants.%custom_witness.b80ef6.8, constants.%DestroyImpl.impl_witness.06a) [concrete = constants.%facet_value.bb3]
// CHECK:STDOUT:     %.loc26_19.19: %facet_type.8a3 = converted constants.%Optional.d01, %facet_value.loc26_19.3 [concrete = constants.%facet_value.bb3]
// CHECK:STDOUT:     %facet_value.loc26_19.4: %facet_type.8a3 = facet_value constants.%Optional.d01, (constants.%custom_witness.b80ef6.8, constants.%DestroyImpl.impl_witness.06a) [concrete = constants.%facet_value.bb3]
// CHECK:STDOUT:     %.loc26_19.20: %facet_type.8a3 = converted constants.%Optional.d01, %facet_value.loc26_19.4 [concrete = constants.%facet_value.bb3]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %SelfDestruct.call.loc26_19.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc26_19.8)
// CHECK:STDOUT:     %.loc26_19.21: %facet_type.8a3 = converted constants.%as_type.b5f, constants.%impl.elem1.c65 [symbolic = %impl.elem1 (constants.%impl.elem1.c65)]
// CHECK:STDOUT:     %.loc26_19.22: %facet_type.8a3 = converted constants.%as_type.b5f, constants.%impl.elem1.c65 [symbolic = %impl.elem1 (constants.%impl.elem1.c65)]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %SelfDestruct.call.loc26_19.3: init %empty_tuple.type = call %SelfDestruct.specific_fn.3(%var)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc26_19.1(%self.param: ref %.b81) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc26_19.2(%self.param: ref %MaybeUnformed.609) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc26_19.3(%self.param: ref bool) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc26_19.4(%self.param: ref %struct_type.value.has_value.e85) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc26_19.1: ref bool = struct_access %self.param, element1
// CHECK:STDOUT:   %facet_value.loc26_19.1: %facet_type.8a3 = facet_value bool, (constants.%custom_witness.b80ef6.5, constants.%DestroyImpl.impl_witness.deb) [concrete = constants.%facet_value.a1b]
// CHECK:STDOUT:   %.loc26_19.2: %facet_type.8a3 = converted bool, %facet_value.loc26_19.1 [concrete = constants.%facet_value.a1b]
// CHECK:STDOUT:   %facet_value.loc26_19.2: %facet_type.8a3 = facet_value bool, (constants.%custom_witness.b80ef6.5, constants.%DestroyImpl.impl_witness.deb) [concrete = constants.%facet_value.a1b]
// CHECK:STDOUT:   %.loc26_19.3: %facet_type.8a3 = converted bool, %facet_value.loc26_19.2 [concrete = constants.%facet_value.a1b]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc26_19.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc26_19.1)
// CHECK:STDOUT:   %.loc26_19.4: ref %MaybeUnformed.609 = struct_access %self.param, element0
// CHECK:STDOUT:   %facet_value.loc26_19.3: %facet_type.8a3 = facet_value constants.%MaybeUnformed.609, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.4ed) [concrete = constants.%facet_value.40c]
// CHECK:STDOUT:   %.loc26_19.5: %facet_type.8a3 = converted constants.%MaybeUnformed.609, %facet_value.loc26_19.3 [concrete = constants.%facet_value.40c]
// CHECK:STDOUT:   %facet_value.loc26_19.4: %facet_type.8a3 = facet_value constants.%MaybeUnformed.609, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.4ed) [concrete = constants.%facet_value.40c]
// CHECK:STDOUT:   %.loc26_19.6: %facet_type.8a3 = converted constants.%MaybeUnformed.609, %facet_value.loc26_19.4 [concrete = constants.%facet_value.40c]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc26_19.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc26_19.4)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc26_19.5(%self.param: ref %DefaultOptionalStorage.cd2) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %has_value.ref: %DefaultOptionalStorage.elem.81c = name_ref has_value, imports.%Core.import_ref.f1da [concrete = imports.%field_decl.431]
// CHECK:STDOUT:   %.loc26_19.1: ref bool = class_element_access %self.param, element1
// CHECK:STDOUT:   %facet_value.loc26_19.1: %facet_type.8a3 = facet_value bool, (constants.%custom_witness.b80ef6.5, constants.%DestroyImpl.impl_witness.deb) [concrete = constants.%facet_value.a1b]
// CHECK:STDOUT:   %.loc26_19.2: %facet_type.8a3 = converted bool, %facet_value.loc26_19.1 [concrete = constants.%facet_value.a1b]
// CHECK:STDOUT:   %facet_value.loc26_19.2: %facet_type.8a3 = facet_value bool, (constants.%custom_witness.b80ef6.5, constants.%DestroyImpl.impl_witness.deb) [concrete = constants.%facet_value.a1b]
// CHECK:STDOUT:   %.loc26_19.3: %facet_type.8a3 = converted bool, %facet_value.loc26_19.2 [concrete = constants.%facet_value.a1b]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc26_19.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc26_19.1)
// CHECK:STDOUT:   %value.ref: %DefaultOptionalStorage.elem.fb9 = name_ref value, imports.%Core.import_ref.146b [concrete = imports.%field_decl.31e]
// CHECK:STDOUT:   %.loc26_19.4: ref %MaybeUnformed.609 = class_element_access %self.param, element0
// CHECK:STDOUT:   %facet_value.loc26_19.3: %facet_type.8a3 = facet_value constants.%MaybeUnformed.609, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.4ed) [concrete = constants.%facet_value.40c]
// CHECK:STDOUT:   %.loc26_19.5: %facet_type.8a3 = converted constants.%MaybeUnformed.609, %facet_value.loc26_19.3 [concrete = constants.%facet_value.40c]
// CHECK:STDOUT:   %facet_value.loc26_19.4: %facet_type.8a3 = facet_value constants.%MaybeUnformed.609, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.4ed) [concrete = constants.%facet_value.40c]
// CHECK:STDOUT:   %.loc26_19.6: %facet_type.8a3 = converted constants.%MaybeUnformed.609, %facet_value.loc26_19.4 [concrete = constants.%facet_value.40c]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc26_19.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc26_19.4)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc26_19.6(%self.param: ref %Optional.d01) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestRangeFor(%m.param: ref %MutableRange) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %i.patt: %pattern_type.522 = value_binding_pattern i [concrete = constants.%i.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %m.ref: ref %MutableRange = name_ref m, %m
// CHECK:STDOUT:   %impl.elem2: %.5ac = impl_witness_access constants.%Iterate.impl_witness.43e, element2 [concrete = constants.%T.as_type.as.Iterate.impl.NewCursor.5df]
// CHECK:STDOUT:   %bound_method.loc68_19.1: <bound method> = bound_method %m.ref, %impl.elem2
// CHECK:STDOUT:   %facet_value.loc68_19.1: %CppRange_where.type.d682d6.1 = facet_value constants.%MutableRange, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.868, constants.%Copy.impl_witness.e6c, constants.%custom_witness.902, constants.%custom_witness.f01, constants.%DestroyImpl.impl_witness.0af, constants.%custom_witness.c50, constants.%custom_witness.491, constants.%custom_witness.ed8) [concrete = constants.%facet_value.86b]
// CHECK:STDOUT:   %.loc68_19.1: %CppRange_where.type.d682d6.1 = converted constants.%MutableRange, %facet_value.loc68_19.1 [concrete = constants.%facet_value.86b]
// CHECK:STDOUT:   %facet_value.loc68_19.2: %CppRange_where.type.d682d6.1 = facet_value constants.%MutableRange, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.868, constants.%Copy.impl_witness.e6c, constants.%custom_witness.902, constants.%custom_witness.f01, constants.%DestroyImpl.impl_witness.0af, constants.%custom_witness.c50, constants.%custom_witness.491, constants.%custom_witness.ed8) [concrete = constants.%facet_value.86b]
// CHECK:STDOUT:   %.loc68_19.2: %CppRange_where.type.d682d6.1 = converted constants.%MutableRange, %facet_value.loc68_19.2 [concrete = constants.%facet_value.86b]
// CHECK:STDOUT:   %specific_fn.loc68_19.1: <specific function> = specific_function %impl.elem2, @T.as_type.as.Iterate.impl.NewCursor(constants.%facet_value.86b) [concrete = constants.%T.as_type.as.Iterate.impl.NewCursor.specific_fn]
// CHECK:STDOUT:   %bound_method.loc68_19.2: <bound method> = bound_method %m.ref, %specific_fn.loc68_19.1
// CHECK:STDOUT:   %var: ref %tuple.type.dd0 = var_storage invalid
// CHECK:STDOUT:   %.loc68_18.1: %MutableRange = acquire_value %m.ref
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.call: init %tuple.type.dd0 to %var = call %bound_method.loc68_19.2(%.loc68_18.1)
// CHECK:STDOUT:   assign %var, %T.as_type.as.Iterate.impl.NewCursor.call
// CHECK:STDOUT:   br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT: !for.next:
// CHECK:STDOUT:   %impl.elem3: %.854 = impl_witness_access constants.%Iterate.impl_witness.43e, element3 [concrete = constants.%T.as_type.as.Iterate.impl.Next.b6d]
// CHECK:STDOUT:   %bound_method.loc68_19.3: <bound method> = bound_method %m.ref, %impl.elem3
// CHECK:STDOUT:   %facet_value.loc68_19.3: %CppRange_where.type.d682d6.1 = facet_value constants.%MutableRange, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.868, constants.%Copy.impl_witness.e6c, constants.%custom_witness.902, constants.%custom_witness.f01, constants.%DestroyImpl.impl_witness.0af, constants.%custom_witness.c50, constants.%custom_witness.491, constants.%custom_witness.ed8) [concrete = constants.%facet_value.86b]
// CHECK:STDOUT:   %.loc68_19.3: %CppRange_where.type.d682d6.1 = converted constants.%MutableRange, %facet_value.loc68_19.3 [concrete = constants.%facet_value.86b]
// CHECK:STDOUT:   %facet_value.loc68_19.4: %CppRange_where.type.d682d6.1 = facet_value constants.%MutableRange, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.868, constants.%Copy.impl_witness.e6c, constants.%custom_witness.902, constants.%custom_witness.f01, constants.%DestroyImpl.impl_witness.0af, constants.%custom_witness.c50, constants.%custom_witness.491, constants.%custom_witness.ed8) [concrete = constants.%facet_value.86b]
// CHECK:STDOUT:   %.loc68_19.4: %CppRange_where.type.d682d6.1 = converted constants.%MutableRange, %facet_value.loc68_19.4 [concrete = constants.%facet_value.86b]
// CHECK:STDOUT:   %specific_fn.loc68_19.2: <specific function> = specific_function %impl.elem3, @T.as_type.as.Iterate.impl.Next(constants.%facet_value.86b) [concrete = constants.%T.as_type.as.Iterate.impl.Next.specific_fn]
// CHECK:STDOUT:   %bound_method.loc68_19.4: <bound method> = bound_method %m.ref, %specific_fn.loc68_19.2
// CHECK:STDOUT:   %.loc68_19.5: ref %Optional.d01 = temporary_storage
// CHECK:STDOUT:   %.loc68_18.2: %MutableRange = acquire_value %m.ref
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.call: init %Optional.d01 to %.loc68_19.5 = call %bound_method.loc68_19.4(%.loc68_18.2, %var)
// CHECK:STDOUT:   %.loc68_19.6: ref %Optional.d01 = temporary %.loc68_19.5, %T.as_type.as.Iterate.impl.Next.call
// CHECK:STDOUT:   %.loc68_19.7: %Optional.HasValue.type.4ec = specific_constant imports.%Core.import_ref.4d4, @Optional(constants.%OptionalStorage.facet.040) [concrete = constants.%Optional.HasValue.d29]
// CHECK:STDOUT:   %HasValue.ref: %Optional.HasValue.type.4ec = name_ref HasValue, %.loc68_19.7 [concrete = constants.%Optional.HasValue.d29]
// CHECK:STDOUT:   %Optional.HasValue.bound: <bound method> = bound_method %.loc68_19.6, %HasValue.ref
// CHECK:STDOUT:   %Optional.HasValue.specific_fn: <specific function> = specific_function %HasValue.ref, @Optional.HasValue(constants.%OptionalStorage.facet.040) [concrete = constants.%Optional.HasValue.specific_fn.6ca]
// CHECK:STDOUT:   %bound_method.loc68_19.5: <bound method> = bound_method %.loc68_19.6, %Optional.HasValue.specific_fn
// CHECK:STDOUT:   %.loc68_19.8: %Optional.d01 = acquire_value %.loc68_19.6
// CHECK:STDOUT:   %Optional.HasValue.call: init bool = call %bound_method.loc68_19.5(%.loc68_19.8)
// CHECK:STDOUT:   %.loc68_19.9: bool = value_of_initializer %Optional.HasValue.call
// CHECK:STDOUT:   %.loc68_19.10: bool = converted %Optional.HasValue.call, %.loc68_19.9
// CHECK:STDOUT:   if %.loc68_19.10 br !for.body else br !for.done
// CHECK:STDOUT:
// CHECK:STDOUT: !for.body:
// CHECK:STDOUT:   %.loc68_19.11: %Optional.Get.type.53f = specific_constant imports.%Core.import_ref.059, @Optional(constants.%OptionalStorage.facet.040) [concrete = constants.%Optional.Get.cb5]
// CHECK:STDOUT:   %Get.ref: %Optional.Get.type.53f = name_ref Get, %.loc68_19.11 [concrete = constants.%Optional.Get.cb5]
// CHECK:STDOUT:   %Optional.Get.bound: <bound method> = bound_method %.loc68_19.6, %Get.ref
// CHECK:STDOUT:   %Optional.Get.specific_fn: <specific function> = specific_function %Get.ref, @Optional.Get(constants.%OptionalStorage.facet.040) [concrete = constants.%Optional.Get.specific_fn.e9a]
// CHECK:STDOUT:   %bound_method.loc68_19.6: <bound method> = bound_method %.loc68_19.6, %Optional.Get.specific_fn
// CHECK:STDOUT:   %.loc68_19.12: %Optional.d01 = acquire_value %.loc68_19.6
// CHECK:STDOUT:   %Optional.Get.call: init %f64.978 = call %bound_method.loc68_19.6(%.loc68_19.12)
// CHECK:STDOUT:   %.loc68_19.13: %f64.978 = value_of_initializer %Optional.Get.call
// CHECK:STDOUT:   %.loc68_19.14: %f64.978 = converted %Optional.Get.call, %.loc68_19.13
// CHECK:STDOUT:   %f64.loc68: type = type_literal constants.%f64.978 [concrete = constants.%f64.978]
// CHECK:STDOUT:   %i: %f64.978 = wrapper_binding i, %.loc68_19.14
// CHECK:STDOUT:   %sum.ref: ref %f64.978 = name_ref sum, %sum
// CHECK:STDOUT:   %i.ref: %f64.978 = name_ref i, %i
// CHECK:STDOUT:   %impl.elem0.loc69: %.74b = impl_witness_access constants.%AddAssignWith.impl_witness.614, element0 [concrete = constants.%Float.as.AddAssignWith.impl.Op.cd7]
// CHECK:STDOUT:   %bound_method.loc69_9.1: <bound method> = bound_method %sum.ref, %impl.elem0.loc69
// CHECK:STDOUT:   %specific_fn.loc69: <specific function> = specific_function %impl.elem0.loc69, @Float.as.AddAssignWith.impl.Op(constants.%int_64) [concrete = constants.%Float.as.AddAssignWith.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc69_9.2: <bound method> = bound_method %sum.ref, %specific_fn.loc69
// CHECK:STDOUT:   %Float.as.AddAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc69_9.2(%sum.ref, %i.ref)
// CHECK:STDOUT:   %facet_value.loc68_19.5: %facet_type.8a3 = facet_value constants.%Optional.d01, (constants.%custom_witness.b80ef6.8, constants.%DestroyImpl.impl_witness.06a) [concrete = constants.%facet_value.bb3]
// CHECK:STDOUT:   %.loc68_19.15: %facet_type.8a3 = converted constants.%Optional.d01, %facet_value.loc68_19.5 [concrete = constants.%facet_value.bb3]
// CHECK:STDOUT:   %facet_value.loc68_19.6: %facet_type.8a3 = facet_value constants.%Optional.d01, (constants.%custom_witness.b80ef6.8, constants.%DestroyImpl.impl_witness.06a) [concrete = constants.%facet_value.bb3]
// CHECK:STDOUT:   %.loc68_19.16: %facet_type.8a3 = converted constants.%Optional.d01, %facet_value.loc68_19.6 [concrete = constants.%facet_value.bb3]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc68_19.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc68_19.6)
// CHECK:STDOUT:   br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT: !for.done:
// CHECK:STDOUT:   %facet_value.loc68_19.7: %facet_type.8a3 = facet_value constants.%Optional.d01, (constants.%custom_witness.b80ef6.8, constants.%DestroyImpl.impl_witness.06a) [concrete = constants.%facet_value.bb3]
// CHECK:STDOUT:   %.loc68_19.17: %facet_type.8a3 = converted constants.%Optional.d01, %facet_value.loc68_19.7 [concrete = constants.%facet_value.bb3]
// CHECK:STDOUT:   %facet_value.loc68_19.8: %facet_type.8a3 = facet_value constants.%Optional.d01, (constants.%custom_witness.b80ef6.8, constants.%DestroyImpl.impl_witness.06a) [concrete = constants.%facet_value.bb3]
// CHECK:STDOUT:   %.loc68_19.18: %facet_type.8a3 = converted constants.%Optional.d01, %facet_value.loc68_19.8 [concrete = constants.%facet_value.bb3]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc68_19.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc68_19.6)
// CHECK:STDOUT:   %facet_value.loc68_19.9: %facet_type.8a3 = facet_value constants.%tuple.type.dd0, (constants.%custom_witness.b80ef6.9, constants.%DestroyImpl.impl_witness.6b2) [concrete = constants.%facet_value.b59]
// CHECK:STDOUT:   %.loc68_19.19: %facet_type.8a3 = converted constants.%tuple.type.dd0, %facet_value.loc68_19.9 [concrete = constants.%facet_value.b59]
// CHECK:STDOUT:   %facet_value.loc68_19.10: %facet_type.8a3 = facet_value constants.%tuple.type.dd0, (constants.%custom_witness.b80ef6.9, constants.%DestroyImpl.impl_witness.6b2) [concrete = constants.%facet_value.b59]
// CHECK:STDOUT:   %.loc68_19.20: %facet_type.8a3 = converted constants.%tuple.type.dd0, %facet_value.loc68_19.10 [concrete = constants.%facet_value.b59]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc68_19.3: init %empty_tuple.type = call %SelfDestruct.specific_fn.3(%var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestIterate(constants.%R) {
// CHECK:STDOUT:   %R.patt.loc22_17.2 => constants.%R.patt
// CHECK:STDOUT:   %R.loc22_17.1 => constants.%R
// CHECK:STDOUT:   %R.as_type.loc22_61.1 => constants.%R.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.bac
// CHECK:STDOUT:   %r.param_patt.loc22_59.2 => constants.%r.param_patt.954
// CHECK:STDOUT:   %r.patt.loc22_59.2 => constants.%r.patt.41e
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestIterate(constants.%facet_value.1f1) {
// CHECK:STDOUT:   %R.patt.loc22_17.2 => constants.%R.patt
// CHECK:STDOUT:   %R.loc22_17.1 => constants.%facet_value.1f1
// CHECK:STDOUT:   %R.as_type.loc22_61.1 => constants.%MutableRange
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.992
// CHECK:STDOUT:   %r.param_patt.loc22_59.2 => constants.%r.param_patt.aeb
// CHECK:STDOUT:   %r.patt.loc22_59.2 => constants.%r.patt.892
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc22 => constants.%complete_type.357
// CHECK:STDOUT:   %Iterate.lookup_impl_witness => constants.%Iterate.impl_witness.43e
// CHECK:STDOUT:   %Iterate.facet.loc26_19.5 => constants.%Iterate.facet.b6a
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type => constants.%Iterate.WithSelf.NewCursor.type.2d6
// CHECK:STDOUT:   %.loc26_19.23 => constants.%.5ac
// CHECK:STDOUT:   %impl.elem2.loc26_19.2 => constants.%T.as_type.as.Iterate.impl.NewCursor.5df
// CHECK:STDOUT:   %specific_impl_fn.loc26_19.3 => constants.%T.as_type.as.Iterate.impl.NewCursor.specific_fn
// CHECK:STDOUT:   %impl.elem1 => constants.%facet_value.b59
// CHECK:STDOUT:   %as_type => constants.%tuple.type.dd0
// CHECK:STDOUT:   %require_complete.1 => constants.%complete_type.843
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type => constants.%Iterate.WithSelf.Next.type.bcc
// CHECK:STDOUT:   %.loc26_19.24 => constants.%.854
// CHECK:STDOUT:   %impl.elem3.loc26_19.2 => constants.%T.as_type.as.Iterate.impl.Next.b6d
// CHECK:STDOUT:   %specific_impl_fn.loc26_19.4 => constants.%T.as_type.as.Iterate.impl.Next.specific_fn
// CHECK:STDOUT:   %SelfDestruct.specific_fn.5 => constants.%SelfDestruct.specific_fn.2f3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- adl_returns_different_types.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %Iterate.type: type = facet_type <@Iterate> [concrete]
// CHECK:STDOUT:   %OptionalStorage.type: type = facet_type <@OptionalStorage> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %facet_type.d61: type = facet_type <@SubobjectDestroy & @DestroyImpl & @Copy> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %T.220: %OptionalStorage.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Optional.Get.type.d97: type = fn_type @Optional.Get, @Optional(%T.220) [symbolic]
// CHECK:STDOUT:   %Optional.Get.3e9: %Optional.Get.type.d97 = struct_value () [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.type.cef: type = fn_type @Optional.HasValue, @Optional(%T.220) [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.a74: %Optional.HasValue.type.cef = struct_value () [symbolic]
// CHECK:STDOUT:   %facet_type.8a3: type = facet_type <@SubobjectDestroy & @DestroyImpl> [concrete]
// CHECK:STDOUT:   %T.9ab: %facet_type.d61 = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.19c: type = facet_access_type %T.9ab [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.lookup_impl_witness.855: <witness> = lookup_impl_witness %T.9ab, @SubobjectDestroy [symbolic]
// CHECK:STDOUT:   %DestroyImpl.lookup_impl_witness.ff9: <witness> = lookup_impl_witness %T.9ab, @DestroyImpl [symbolic]
// CHECK:STDOUT:   %facet_value.b35: %facet_type.8a3 = facet_value %T.as_type.19c, (%SubobjectDestroy.lookup_impl_witness.855, %DestroyImpl.lookup_impl_witness.ff9) [symbolic]
// CHECK:STDOUT:   %MaybeUnformed.413: type = class_type @MaybeUnformed, @MaybeUnformed(%facet_value.b35) [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.a6b: type = class_type @DefaultOptionalStorage, @DefaultOptionalStorage(%T.9ab) [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.elem.94e: type = unbound_element_type %DefaultOptionalStorage.a6b, bool [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.elem.c3f: type = unbound_element_type %DefaultOptionalStorage.a6b, %MaybeUnformed.413 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Get.type.b32: type = fn_type @T.as_type.as.OptionalStorage.impl.Get, @T.as_type.as.OptionalStorage.impl(%T.9ab) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Get.8b0: %T.as_type.as.OptionalStorage.impl.Get.type.b32 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Has.type.8e5: type = fn_type @T.as_type.as.OptionalStorage.impl.Has, @T.as_type.as.OptionalStorage.impl(%T.9ab) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Has.2d5: %T.as_type.as.OptionalStorage.impl.Has.type.8e5 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Some.type.61d: type = fn_type @T.as_type.as.OptionalStorage.impl.Some, @T.as_type.as.OptionalStorage.impl(%T.9ab) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Some.ba7: %T.as_type.as.OptionalStorage.impl.Some.type.61d = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.None.type.a9f: type = fn_type @T.as_type.as.OptionalStorage.impl.None, @T.as_type.as.OptionalStorage.impl(%T.9ab) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.None.083: %T.as_type.as.OptionalStorage.impl.None.type.a9f = struct_value () [symbolic]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.273: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.5b5b: %T.as_type.as.DestroyImpl.impl.Op.type.273 = struct_value () [symbolic]
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
// CHECK:STDOUT:   %SubobjectDestroy.lookup_impl_witness.dc9: <witness> = lookup_impl_witness %tuple.type.1ba, @SubobjectDestroy [symbolic]
// CHECK:STDOUT:   %DestroyImpl.lookup_impl_witness.504: <witness> = lookup_impl_witness %tuple.type.1ba, @DestroyImpl [symbolic]
// CHECK:STDOUT:   %facet_value.bf6: %facet_type.8a3 = facet_value %tuple.type.1ba, (%SubobjectDestroy.lookup_impl_witness.dc9, %DestroyImpl.lookup_impl_witness.504) [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.lookup_impl_witness.b15da7.2: <witness> = lookup_impl_witness %impl.elem0.493765.2, @SubobjectDestroy [symbolic]
// CHECK:STDOUT:   %DestroyImpl.lookup_impl_witness.4e7f44.2: <witness> = lookup_impl_witness %impl.elem0.493765.2, @DestroyImpl [symbolic]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.167: <witness> = lookup_impl_witness %impl.elem0.493765.2, @Copy [symbolic]
// CHECK:STDOUT:   %facet_value.9da: %facet_type.d61 = facet_value %impl.elem0.493765.2, (%SubobjectDestroy.lookup_impl_witness.b15da7.2, %DestroyImpl.lookup_impl_witness.4e7f44.2, %Copy.lookup_impl_witness.167) [symbolic]
// CHECK:STDOUT:   %SelfDestruct.type: type = fn_type @SelfDestruct [concrete]
// CHECK:STDOUT:   %SelfDestruct: %SelfDestruct.type = struct_value () [concrete]
// CHECK:STDOUT:   %ValueType: type = class_type @ValueType [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %pattern_type.c433: type = pattern_type %ValueType [concrete]
// CHECK:STDOUT:   %ValueType.Op.type.4c0699.1: type = fn_type @ValueType.Op.1 [concrete]
// CHECK:STDOUT:   %ValueType.Op.4ec7f2.1: %ValueType.Op.type.4c0699.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.0a1: <witness> = custom_witness (%ValueType.Op.4ec7f2.1), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.bf6: %SubobjectDestroy.type = facet_value %ValueType, (%custom_witness.0a1) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.0ef: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.bf6) [concrete]
// CHECK:STDOUT:   %const.dd6: type = const_type %ValueType [concrete]
// CHECK:STDOUT:   %ptr.69c: type = ptr_type %const.dd6 [concrete]
// CHECK:STDOUT:   %ptr.233: type = ptr_type %ValueType [concrete]
// CHECK:STDOUT:   %ValueType.Op.type.4c0699.2: type = fn_type @ValueType.Op.2 [concrete]
// CHECK:STDOUT:   %ValueType.Op.4ec7f2.2: %ValueType.Op.type.4c0699.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.202: <witness> = custom_witness (%ValueType.Op.4ec7f2.2), @Copy [concrete]
// CHECK:STDOUT:   %facet_value.4b1: %facet_type.d61 = facet_value %ValueType, (%custom_witness.0a1, %DestroyImpl.impl_witness.0ef, %custom_witness.202) [concrete]
// CHECK:STDOUT:   %Iterate.lookup_impl_witness.1d8: <witness> = lookup_impl_witness %.Self.500, @Iterate [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.8a4: %facet_type.d61 = impl_witness_access %Iterate.lookup_impl_witness.1d8, element0 [symbolic_self]
// CHECK:STDOUT:   %Iterate_where.type.60f: type = facet_type <@Iterate where %impl.elem0.8a4 = %facet_value.4b1> [concrete]
// CHECK:STDOUT:   %pattern_type.eef: type = pattern_type %Iterate_where.type.60f [concrete]
// CHECK:STDOUT:   %R.patt: %pattern_type.eef = symbolic_binding_pattern R, 0 [symbolic]
// CHECK:STDOUT:   %R: %Iterate_where.type.60f = symbolic_binding R, 0 [symbolic]
// CHECK:STDOUT:   %R.as_type: type = facet_access_type %R [symbolic]
// CHECK:STDOUT:   %pattern_type.323: type = pattern_type %R.as_type [symbolic]
// CHECK:STDOUT:   %r.param_patt.77f: %pattern_type.323 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %r.patt.e3b: %pattern_type.323 = wrapper_binding_pattern r, %r.param_patt.77f [symbolic]
// CHECK:STDOUT:   %i.patt: %pattern_type.c433 = value_binding_pattern i [concrete]
// CHECK:STDOUT:   %Iterate.lookup_impl_witness.235: <witness> = lookup_impl_witness %R, @Iterate [symbolic]
// CHECK:STDOUT:   %Iterate.facet.7ef: %Iterate.type = facet_value %R.as_type, (%Iterate.lookup_impl_witness.235) [symbolic]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.ee6: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.7ef) [symbolic]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.0a0: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.7ef) [symbolic]
// CHECK:STDOUT:   %.68c: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.ee6, %Iterate.facet.7ef [symbolic]
// CHECK:STDOUT:   %impl.elem2.69b: %.68c = impl_witness_access %Iterate.lookup_impl_witness.235, element2 [symbolic]
// CHECK:STDOUT:   %impl.elem1.230: %facet_type.8a3 = impl_witness_access %Iterate.lookup_impl_witness.235, element1 [symbolic]
// CHECK:STDOUT:   %as_type.9c6: type = facet_access_type %impl.elem1.230 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.e2ae: <specific function> = specific_impl_function %impl.elem2.69b, @Iterate.WithSelf.NewCursor(%Iterate.facet.7ef) [symbolic]
// CHECK:STDOUT:   %.4d0: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.0a0, %Iterate.facet.7ef [symbolic]
// CHECK:STDOUT:   %impl.elem3.201: %.4d0 = impl_witness_access %Iterate.lookup_impl_witness.235, element3 [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.1a2: type = class_type @DefaultOptionalStorage, @DefaultOptionalStorage(%facet_value.4b1) [concrete]
// CHECK:STDOUT:   %OptionalStorage.impl_witness.d6e: <witness> = impl_witness imports.%OptionalStorage.impl_witness_table.968, @T.as_type.as.OptionalStorage.impl(%facet_value.4b1) [concrete]
// CHECK:STDOUT:   %OptionalStorage.facet.9e7: %OptionalStorage.type = facet_value %ValueType, (%OptionalStorage.impl_witness.d6e) [concrete]
// CHECK:STDOUT:   %Optional.f64: type = class_type @Optional, @Optional(%OptionalStorage.facet.9e7) [concrete]
// CHECK:STDOUT:   %specific_impl_fn.437: <specific function> = specific_impl_function %impl.elem3.201, @Iterate.WithSelf.Next(%Iterate.facet.7ef) [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.type.c39: type = fn_type @Optional.HasValue, @Optional(%OptionalStorage.facet.9e7) [concrete]
// CHECK:STDOUT:   %Optional.HasValue.8b0: %Optional.HasValue.type.c39 = struct_value () [concrete]
// CHECK:STDOUT:   %Optional.Get.type.7e7: type = fn_type @Optional.Get, @Optional(%OptionalStorage.facet.9e7) [concrete]
// CHECK:STDOUT:   %Optional.Get.a0e: %Optional.Get.type.7e7 = struct_value () [concrete]
// CHECK:STDOUT:   %facet_value.6d5: %facet_type.8a3 = facet_value %ValueType, (%custom_witness.0a1, %DestroyImpl.impl_witness.0ef) [concrete]
// CHECK:STDOUT:   %MaybeUnformed.9a0: type = class_type @MaybeUnformed, @MaybeUnformed(%facet_value.6d5) [concrete]
// CHECK:STDOUT:   %.be1: type = maybe_unformed_type %ValueType [concrete]
// CHECK:STDOUT:   %DefaultOptionalStorage.elem.d17: type = unbound_element_type %DefaultOptionalStorage.1a2, %MaybeUnformed.9a0 [concrete]
// CHECK:STDOUT:   %DefaultOptionalStorage.elem.1ec: type = unbound_element_type %DefaultOptionalStorage.1a2, bool [concrete]
// CHECK:STDOUT:   %struct_type.value.has_value.f9d: type = struct_type {.value: %MaybeUnformed.9a0, .has_value: bool} [concrete]
// CHECK:STDOUT:   %Optional.HasValue.specific_fn.418: <specific function> = specific_function %Optional.HasValue.8b0, @Optional.HasValue(%OptionalStorage.facet.9e7) [concrete]
// CHECK:STDOUT:   %Optional.Get.specific_fn.cb7: <specific function> = specific_function %Optional.Get.a0e, @Optional.Get(%OptionalStorage.facet.9e7) [concrete]
// CHECK:STDOUT:   %operator_PlusEqual__carbon_thunk.type: type = fn_type @operator_PlusEqual__carbon_thunk [concrete]
// CHECK:STDOUT:   %operator_PlusEqual__carbon_thunk: %operator_PlusEqual__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.3: type = fn_type @SubobjectDestroy.WithSelf.Op.loc31_37.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.3: %SubobjectDestroy.WithSelf.Op.type.6870fd.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.3: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.3), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.4: type = fn_type @SubobjectDestroy.WithSelf.Op.loc31_37.3 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.4: %SubobjectDestroy.WithSelf.Op.type.6870fd.4 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.4: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.4), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.a87: %SubobjectDestroy.type = facet_value bool, (%custom_witness.b80ef6.4) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.deb: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.a87) [concrete]
// CHECK:STDOUT:   %facet_value.a1b: %facet_type.8a3 = facet_value bool, (%custom_witness.b80ef6.4, %DestroyImpl.impl_witness.deb) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.677: %SubobjectDestroy.type = facet_value %MaybeUnformed.9a0, (%custom_witness.b80ef6.3) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.c48: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.677) [concrete]
// CHECK:STDOUT:   %facet_value.585: %facet_type.8a3 = facet_value %MaybeUnformed.9a0, (%custom_witness.b80ef6.3, %DestroyImpl.impl_witness.c48) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.7: type = fn_type @SubobjectDestroy.WithSelf.Op.loc31_37.6 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.7: %SubobjectDestroy.WithSelf.Op.type.6870fd.7 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.7: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.7), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.deb: %SubobjectDestroy.type = facet_value %Optional.f64, (%custom_witness.b80ef6.7) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.59c: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.deb) [concrete]
// CHECK:STDOUT:   %facet_value.69a: %facet_type.8a3 = facet_value %Optional.f64, (%custom_witness.b80ef6.7, %DestroyImpl.impl_witness.59c) [concrete]
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
// CHECK:STDOUT:   %custom_witness.612b50.1: <witness> = custom_witness (%Iterator.Op.240e50.2), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.403: %SubobjectDestroy.type = facet_value %Iterator, (%custom_witness.612b50.1) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.3fc: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.403) [concrete]
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
// CHECK:STDOUT:   %custom_witness.612b50.2: <witness> = custom_witness (%Sentinel.Op), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.96b: %SubobjectDestroy.type = facet_value %Sentinel, (%custom_witness.612b50.2) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.72c: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.96b) [concrete]
// CHECK:STDOUT:   %facet_value.6f9: %CppRange_where.type.d682d6.1 = facet_value %ConstRange, (%custom_witness.0a1, %DestroyImpl.impl_witness.0ef, %custom_witness.202, %custom_witness.116, %custom_witness.612b50.1, %DestroyImpl.impl_witness.3fc, %custom_witness.5be, %custom_witness.c31, %custom_witness.e21, %custom_witness.612b50.2, %DestroyImpl.impl_witness.72c) [concrete]
// CHECK:STDOUT:   %tuple.type.7a6: type = tuple_type (%Iterator, %Sentinel) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.8: type = fn_type @SubobjectDestroy.WithSelf.Op.3 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.8: %SubobjectDestroy.WithSelf.Op.type.6870fd.8 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.8: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.8), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.b14: %SubobjectDestroy.type = facet_value %tuple.type.7a6, (%custom_witness.b80ef6.8) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.288: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.b14) [concrete]
// CHECK:STDOUT:   %complete_type.cf0: <witness> = complete_type_witness %tuple.type.7a6 [concrete]
// CHECK:STDOUT:   %facet_value.12b: %facet_type.8a3 = facet_value %tuple.type.7a6, (%custom_witness.b80ef6.8, %DestroyImpl.impl_witness.288) [concrete]
// CHECK:STDOUT:   %Iterate.impl_witness.acc: <witness> = impl_witness imports.%Iterate.impl_witness_table.121, @T.as_type.as.Iterate.impl(%facet_value.6f9) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.type.b24: type = fn_type @T.as_type.as.Iterate.impl.NewCursor, @T.as_type.as.Iterate.impl(%facet_value.6f9) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.239: %T.as_type.as.Iterate.impl.NewCursor.type.b24 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.type.039: type = fn_type @T.as_type.as.Iterate.impl.Next, @T.as_type.as.Iterate.impl(%facet_value.6f9) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.0d5: %T.as_type.as.Iterate.impl.Next.type.039 = struct_value () [concrete]
// CHECK:STDOUT:   %Iterate.facet.30f: %Iterate.type = facet_value %ConstRange, (%Iterate.impl_witness.acc) [concrete]
// CHECK:STDOUT:   %facet_value.c45: %Iterate_where.type.60f = facet_value %ConstRange, (%Iterate.impl_witness.acc) [concrete]
// CHECK:STDOUT:   %r.param_patt.ac5: %pattern_type.c1d = value_param_pattern [concrete]
// CHECK:STDOUT:   %r.patt.97f: %pattern_type.c1d = wrapper_binding_pattern r, %r.param_patt.ac5 [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.693: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.30f) [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.872: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.30f) [concrete]
// CHECK:STDOUT:   %.cf5: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.693, %Iterate.facet.30f [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.specific_fn: <specific function> = specific_function %T.as_type.as.Iterate.impl.NewCursor.239, @T.as_type.as.Iterate.impl.NewCursor(%facet_value.6f9) [concrete]
// CHECK:STDOUT:   %.ba3: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.872, %Iterate.facet.30f [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.specific_fn: <specific function> = specific_function %T.as_type.as.Iterate.impl.Next.0d5, @T.as_type.as.Iterate.impl.Next(%facet_value.6f9) [concrete]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.102: <specific function> = specific_function %SelfDestruct, @SelfDestruct(%facet_value.12b) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .DiffADL = %DiffADL
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.4d4: @Optional.%Optional.HasValue.type (%Optional.HasValue.type.cef) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.HasValue (constants.%Optional.HasValue.a74)]
// CHECK:STDOUT:   %Core.import_ref.059: @Optional.%Optional.Get.type (%Optional.Get.type.d97) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.Get (constants.%Optional.Get.3e9)]
// CHECK:STDOUT:   %Core.import_ref.7bb: type = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%DefaultOptionalStorage (constants.%DefaultOptionalStorage.a6b)]
// CHECK:STDOUT:   %Core.import_ref.61e: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.None.type (%T.as_type.as.OptionalStorage.impl.None.type.a9f) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.None (constants.%T.as_type.as.OptionalStorage.impl.None.083)]
// CHECK:STDOUT:   %Core.import_ref.ca1: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Some.type (%T.as_type.as.OptionalStorage.impl.Some.type.61d) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Some (constants.%T.as_type.as.OptionalStorage.impl.Some.ba7)]
// CHECK:STDOUT:   %Core.import_ref.6ba4: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Has.type (%T.as_type.as.OptionalStorage.impl.Has.type.8e5) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Has (constants.%T.as_type.as.OptionalStorage.impl.Has.2d5)]
// CHECK:STDOUT:   %Core.import_ref.73d: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Get.type (%T.as_type.as.OptionalStorage.impl.Get.type.b32) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Get (constants.%T.as_type.as.OptionalStorage.impl.Get.8b0)]
// CHECK:STDOUT:   %Core.import_ref.778 = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], unloaded
// CHECK:STDOUT:   %OptionalStorage.impl_witness_table.968 = impl_witness_table (%Core.import_ref.7bb, %Core.import_ref.61e, %Core.import_ref.ca1, %Core.import_ref.6ba4, %Core.import_ref.73d, %Core.import_ref.778), @T.as_type.as.OptionalStorage.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.146b: @DefaultOptionalStorage.%DefaultOptionalStorage.elem.1 (%DefaultOptionalStorage.elem.c3f) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [concrete = %field_decl.31e]
// CHECK:STDOUT:   %Core.import_ref.f1da: @DefaultOptionalStorage.%DefaultOptionalStorage.elem.2 (%DefaultOptionalStorage.elem.94e) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [concrete = %field_decl.431]
// CHECK:STDOUT:   %Core.import_ref.14b: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.273) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.5b5b)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.14b), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.d6c: %facet_type.d61 = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.Iterate.impl.%facet_value.1 (constants.%facet_value.9da)]
// CHECK:STDOUT:   %Core.import_ref.2ac: %facet_type.8a3 = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.Iterate.impl.%facet_value.2 (constants.%facet_value.bf6)]
// CHECK:STDOUT:   %Core.import_ref.fd1: @T.as_type.as.Iterate.impl.%T.as_type.as.Iterate.impl.NewCursor.type (%T.as_type.as.Iterate.impl.NewCursor.type.953) = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.Iterate.impl.%T.as_type.as.Iterate.impl.NewCursor (constants.%T.as_type.as.Iterate.impl.NewCursor.544)]
// CHECK:STDOUT:   %Core.import_ref.273: @T.as_type.as.Iterate.impl.%T.as_type.as.Iterate.impl.Next.type (%T.as_type.as.Iterate.impl.Next.type.b36) = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.Iterate.impl.%T.as_type.as.Iterate.impl.Next (constants.%T.as_type.as.Iterate.impl.Next.d6b)]
// CHECK:STDOUT:   %Iterate.impl_witness_table.121 = impl_witness_table (%Core.import_ref.d6c, %Core.import_ref.2ac, %Core.import_ref.fd1, %Core.import_ref.273), @T.as_type.as.Iterate.impl [concrete]
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
// CHECK:STDOUT:   %Core.import_ref.b0d = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %field_decl.431: @DefaultOptionalStorage.%DefaultOptionalStorage.elem.2 (%DefaultOptionalStorage.elem.94e) = field_decl has_value, element1, %Core.import_ref.b0d in [concrete] {}
// CHECK:STDOUT:   %Core.import_ref.164c = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %field_decl.31e: @DefaultOptionalStorage.%DefaultOptionalStorage.elem.1 (%DefaultOptionalStorage.elem.c3f) = field_decl value, element0, %Core.import_ref.164c in [concrete] {}
// CHECK:STDOUT:   %ConstRange.decl: type = class_decl @ConstRange [concrete = constants.%ConstRange] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @TestIterate(%R.loc27_17.2: %Iterate_where.type.60f) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Iterate.lookup_impl_witness: <witness> = lookup_impl_witness %R.loc27_17.1, @Iterate [symbolic = %Iterate.lookup_impl_witness (constants.%Iterate.lookup_impl_witness.235)]
// CHECK:STDOUT:   %Iterate.facet.loc31_37.5: %Iterate.type = facet_value %R.as_type.loc27_79.1, (%Iterate.lookup_impl_witness) [symbolic = %Iterate.facet.loc31_37.5 (constants.%Iterate.facet.7ef)]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.loc31_37.5) [symbolic = %Iterate.WithSelf.NewCursor.type (constants.%Iterate.WithSelf.NewCursor.type.ee6)]
// CHECK:STDOUT:   %.loc31_37.26: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type, %Iterate.facet.loc31_37.5 [symbolic = %.loc31_37.26 (constants.%.68c)]
// CHECK:STDOUT:   %impl.elem2.loc31_37.2: @TestIterate.%.loc31_37.26 (%.68c) = impl_witness_access %Iterate.lookup_impl_witness, element2 [symbolic = %impl.elem2.loc31_37.2 (constants.%impl.elem2.69b)]
// CHECK:STDOUT:   %specific_impl_fn.loc31_37.3: <specific function> = specific_impl_function %impl.elem2.loc31_37.2, @Iterate.WithSelf.NewCursor(%Iterate.facet.loc31_37.5) [symbolic = %specific_impl_fn.loc31_37.3 (constants.%specific_impl_fn.e2ae)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.loc31_37.5) [symbolic = %Iterate.WithSelf.Next.type (constants.%Iterate.WithSelf.Next.type.0a0)]
// CHECK:STDOUT:   %.loc31_37.27: type = fn_type_with_self_type %Iterate.WithSelf.Next.type, %Iterate.facet.loc31_37.5 [symbolic = %.loc31_37.27 (constants.%.4d0)]
// CHECK:STDOUT:   %impl.elem3.loc31_37.2: @TestIterate.%.loc31_37.27 (%.4d0) = impl_witness_access %Iterate.lookup_impl_witness, element3 [symbolic = %impl.elem3.loc31_37.2 (constants.%impl.elem3.201)]
// CHECK:STDOUT:   %specific_impl_fn.loc31_37.4: <specific function> = specific_impl_function %impl.elem3.loc31_37.2, @Iterate.WithSelf.Next(%Iterate.facet.loc31_37.5) [symbolic = %specific_impl_fn.loc31_37.4 (constants.%specific_impl_fn.437)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%r.param: @TestIterate.%R.as_type.loc27_79.1 (%R.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %i.patt: %pattern_type.c433 = value_binding_pattern i [concrete = constants.%i.patt]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %r.ref: @TestIterate.%R.as_type.loc27_79.1 (%R.as_type) = name_ref r, %r
// CHECK:STDOUT:     %impl.elem2.loc31_37.1: @TestIterate.%.loc31_37.26 (%.68c) = impl_witness_access constants.%Iterate.lookup_impl_witness.235, element2 [symbolic = %impl.elem2.loc31_37.2 (constants.%impl.elem2.69b)]
// CHECK:STDOUT:     %bound_method.loc31_37.1: <bound method> = bound_method %r.ref, %impl.elem2.loc31_37.1
// CHECK:STDOUT:     %Iterate.facet.loc31_37.1: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.235) [symbolic = %Iterate.facet.loc31_37.5 (constants.%Iterate.facet.7ef)]
// CHECK:STDOUT:     %.loc31_37.1: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc31_37.1 [symbolic = %Iterate.facet.loc31_37.5 (constants.%Iterate.facet.7ef)]
// CHECK:STDOUT:     %Iterate.facet.loc31_37.2: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.235) [symbolic = %Iterate.facet.loc31_37.5 (constants.%Iterate.facet.7ef)]
// CHECK:STDOUT:     %.loc31_37.2: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc31_37.2 [symbolic = %Iterate.facet.loc31_37.5 (constants.%Iterate.facet.7ef)]
// CHECK:STDOUT:     %specific_impl_fn.loc31_37.1: <specific function> = specific_impl_function %impl.elem2.loc31_37.1, @Iterate.WithSelf.NewCursor(constants.%Iterate.facet.7ef) [symbolic = %specific_impl_fn.loc31_37.3 (constants.%specific_impl_fn.e2ae)]
// CHECK:STDOUT:     %bound_method.loc31_37.2: <bound method> = bound_method %r.ref, %specific_impl_fn.loc31_37.1
// CHECK:STDOUT:     %var: ref @TestIterate.%as_type (%as_type.9c6) = var_storage invalid
// CHECK:STDOUT:     %Iterate.WithSelf.NewCursor.call: init @TestIterate.%as_type (%as_type.9c6) to %var = call %bound_method.loc31_37.2(%r.ref)
// CHECK:STDOUT:     assign %var, %Iterate.WithSelf.NewCursor.call
// CHECK:STDOUT:     br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.next:
// CHECK:STDOUT:     %impl.elem3.loc31_37.1: @TestIterate.%.loc31_37.27 (%.4d0) = impl_witness_access constants.%Iterate.lookup_impl_witness.235, element3 [symbolic = %impl.elem3.loc31_37.2 (constants.%impl.elem3.201)]
// CHECK:STDOUT:     %bound_method.loc31_37.3: <bound method> = bound_method %r.ref, %impl.elem3.loc31_37.1
// CHECK:STDOUT:     %Iterate.facet.loc31_37.3: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.235) [symbolic = %Iterate.facet.loc31_37.5 (constants.%Iterate.facet.7ef)]
// CHECK:STDOUT:     %.loc31_37.3: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc31_37.3 [symbolic = %Iterate.facet.loc31_37.5 (constants.%Iterate.facet.7ef)]
// CHECK:STDOUT:     %Iterate.facet.loc31_37.4: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.235) [symbolic = %Iterate.facet.loc31_37.5 (constants.%Iterate.facet.7ef)]
// CHECK:STDOUT:     %.loc31_37.4: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc31_37.4 [symbolic = %Iterate.facet.loc31_37.5 (constants.%Iterate.facet.7ef)]
// CHECK:STDOUT:     %.loc31_37.5: %facet_type.8a3 = converted constants.%as_type.9c6, constants.%impl.elem1.230 [symbolic = %impl.elem1 (constants.%impl.elem1.230)]
// CHECK:STDOUT:     %.loc31_37.6: %facet_type.8a3 = converted constants.%as_type.9c6, constants.%impl.elem1.230 [symbolic = %impl.elem1 (constants.%impl.elem1.230)]
// CHECK:STDOUT:     %specific_impl_fn.loc31_37.2: <specific function> = specific_impl_function %impl.elem3.loc31_37.1, @Iterate.WithSelf.Next(constants.%Iterate.facet.7ef) [symbolic = %specific_impl_fn.loc31_37.4 (constants.%specific_impl_fn.437)]
// CHECK:STDOUT:     %bound_method.loc31_37.4: <bound method> = bound_method %r.ref, %specific_impl_fn.loc31_37.2
// CHECK:STDOUT:     %.loc31_37.7: ref %Optional.f64 = temporary_storage
// CHECK:STDOUT:     %Iterate.WithSelf.Next.call: init %Optional.f64 to %.loc31_37.7 = call %bound_method.loc31_37.4(%r.ref, %var)
// CHECK:STDOUT:     %.loc31_37.8: ref %Optional.f64 = temporary %.loc31_37.7, %Iterate.WithSelf.Next.call
// CHECK:STDOUT:     %.loc31_37.9: %Optional.HasValue.type.c39 = specific_constant imports.%Core.import_ref.4d4, @Optional(constants.%OptionalStorage.facet.9e7) [concrete = constants.%Optional.HasValue.8b0]
// CHECK:STDOUT:     %HasValue.ref: %Optional.HasValue.type.c39 = name_ref HasValue, %.loc31_37.9 [concrete = constants.%Optional.HasValue.8b0]
// CHECK:STDOUT:     %Optional.HasValue.bound: <bound method> = bound_method %.loc31_37.8, %HasValue.ref
// CHECK:STDOUT:     %Optional.HasValue.specific_fn: <specific function> = specific_function %HasValue.ref, @Optional.HasValue(constants.%OptionalStorage.facet.9e7) [concrete = constants.%Optional.HasValue.specific_fn.418]
// CHECK:STDOUT:     %bound_method.loc31_37.5: <bound method> = bound_method %.loc31_37.8, %Optional.HasValue.specific_fn
// CHECK:STDOUT:     %.loc31_37.10: %Optional.f64 = acquire_value %.loc31_37.8
// CHECK:STDOUT:     %Optional.HasValue.call: init bool = call %bound_method.loc31_37.5(%.loc31_37.10)
// CHECK:STDOUT:     %.loc31_37.11: bool = value_of_initializer %Optional.HasValue.call
// CHECK:STDOUT:     %.loc31_37.12: bool = converted %Optional.HasValue.call, %.loc31_37.11
// CHECK:STDOUT:     if %.loc31_37.12 br !for.body else br !for.done
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.body:
// CHECK:STDOUT:     %.loc31_37.13: %Optional.Get.type.7e7 = specific_constant imports.%Core.import_ref.059, @Optional(constants.%OptionalStorage.facet.9e7) [concrete = constants.%Optional.Get.a0e]
// CHECK:STDOUT:     %Get.ref: %Optional.Get.type.7e7 = name_ref Get, %.loc31_37.13 [concrete = constants.%Optional.Get.a0e]
// CHECK:STDOUT:     %Optional.Get.bound: <bound method> = bound_method %.loc31_37.8, %Get.ref
// CHECK:STDOUT:     %Optional.Get.specific_fn: <specific function> = specific_function %Get.ref, @Optional.Get(constants.%OptionalStorage.facet.9e7) [concrete = constants.%Optional.Get.specific_fn.cb7]
// CHECK:STDOUT:     %bound_method.loc31_37.6: <bound method> = bound_method %.loc31_37.8, %Optional.Get.specific_fn
// CHECK:STDOUT:     %.loc31_37.14: ref %ValueType = temporary_storage
// CHECK:STDOUT:     %.loc31_37.15: %Optional.f64 = acquire_value %.loc31_37.8
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
// CHECK:STDOUT:     %facet_value.loc31_37.1: %facet_type.8a3 = facet_value constants.%ValueType, (constants.%custom_witness.0a1, constants.%DestroyImpl.impl_witness.0ef) [concrete = constants.%facet_value.6d5]
// CHECK:STDOUT:     %.loc31_37.18: %facet_type.8a3 = converted constants.%ValueType, %facet_value.loc31_37.1 [concrete = constants.%facet_value.6d5]
// CHECK:STDOUT:     %facet_value.loc31_37.2: %facet_type.8a3 = facet_value constants.%ValueType, (constants.%custom_witness.0a1, constants.%DestroyImpl.impl_witness.0ef) [concrete = constants.%facet_value.6d5]
// CHECK:STDOUT:     %.loc31_37.19: %facet_type.8a3 = converted constants.%ValueType, %facet_value.loc31_37.2 [concrete = constants.%facet_value.6d5]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %SelfDestruct.call.loc31_37.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc31_37.16)
// CHECK:STDOUT:     %facet_value.loc31_37.3: %facet_type.8a3 = facet_value constants.%Optional.f64, (constants.%custom_witness.b80ef6.7, constants.%DestroyImpl.impl_witness.59c) [concrete = constants.%facet_value.69a]
// CHECK:STDOUT:     %.loc31_37.20: %facet_type.8a3 = converted constants.%Optional.f64, %facet_value.loc31_37.3 [concrete = constants.%facet_value.69a]
// CHECK:STDOUT:     %facet_value.loc31_37.4: %facet_type.8a3 = facet_value constants.%Optional.f64, (constants.%custom_witness.b80ef6.7, constants.%DestroyImpl.impl_witness.59c) [concrete = constants.%facet_value.69a]
// CHECK:STDOUT:     %.loc31_37.21: %facet_type.8a3 = converted constants.%Optional.f64, %facet_value.loc31_37.4 [concrete = constants.%facet_value.69a]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %SelfDestruct.call.loc31_37.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc31_37.8)
// CHECK:STDOUT:     br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.done:
// CHECK:STDOUT:     %facet_value.loc31_37.5: %facet_type.8a3 = facet_value constants.%Optional.f64, (constants.%custom_witness.b80ef6.7, constants.%DestroyImpl.impl_witness.59c) [concrete = constants.%facet_value.69a]
// CHECK:STDOUT:     %.loc31_37.22: %facet_type.8a3 = converted constants.%Optional.f64, %facet_value.loc31_37.5 [concrete = constants.%facet_value.69a]
// CHECK:STDOUT:     %facet_value.loc31_37.6: %facet_type.8a3 = facet_value constants.%Optional.f64, (constants.%custom_witness.b80ef6.7, constants.%DestroyImpl.impl_witness.59c) [concrete = constants.%facet_value.69a]
// CHECK:STDOUT:     %.loc31_37.23: %facet_type.8a3 = converted constants.%Optional.f64, %facet_value.loc31_37.6 [concrete = constants.%facet_value.69a]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %SelfDestruct.call.loc31_37.3: init %empty_tuple.type = call %SelfDestruct.specific_fn.3(%.loc31_37.8)
// CHECK:STDOUT:     %.loc31_37.24: %facet_type.8a3 = converted constants.%as_type.9c6, constants.%impl.elem1.230 [symbolic = %impl.elem1 (constants.%impl.elem1.230)]
// CHECK:STDOUT:     %.loc31_37.25: %facet_type.8a3 = converted constants.%as_type.9c6, constants.%impl.elem1.230 [symbolic = %impl.elem1 (constants.%impl.elem1.230)]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %SelfDestruct.call.loc31_37.4: init %empty_tuple.type = call %SelfDestruct.specific_fn.4(%var)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc31_37.1(%self.param: ref %.be1) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc31_37.2(%self.param: ref %MaybeUnformed.9a0) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc31_37.3(%self.param: ref bool) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc31_37.4(%self.param: ref %struct_type.value.has_value.f9d) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc31_37.1: ref bool = struct_access %self.param, element1
// CHECK:STDOUT:   %facet_value.loc31_37.1: %facet_type.8a3 = facet_value bool, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.deb) [concrete = constants.%facet_value.a1b]
// CHECK:STDOUT:   %.loc31_37.2: %facet_type.8a3 = converted bool, %facet_value.loc31_37.1 [concrete = constants.%facet_value.a1b]
// CHECK:STDOUT:   %facet_value.loc31_37.2: %facet_type.8a3 = facet_value bool, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.deb) [concrete = constants.%facet_value.a1b]
// CHECK:STDOUT:   %.loc31_37.3: %facet_type.8a3 = converted bool, %facet_value.loc31_37.2 [concrete = constants.%facet_value.a1b]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc31_37.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc31_37.1)
// CHECK:STDOUT:   %.loc31_37.4: ref %MaybeUnformed.9a0 = struct_access %self.param, element0
// CHECK:STDOUT:   %facet_value.loc31_37.3: %facet_type.8a3 = facet_value constants.%MaybeUnformed.9a0, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.c48) [concrete = constants.%facet_value.585]
// CHECK:STDOUT:   %.loc31_37.5: %facet_type.8a3 = converted constants.%MaybeUnformed.9a0, %facet_value.loc31_37.3 [concrete = constants.%facet_value.585]
// CHECK:STDOUT:   %facet_value.loc31_37.4: %facet_type.8a3 = facet_value constants.%MaybeUnformed.9a0, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.c48) [concrete = constants.%facet_value.585]
// CHECK:STDOUT:   %.loc31_37.6: %facet_type.8a3 = converted constants.%MaybeUnformed.9a0, %facet_value.loc31_37.4 [concrete = constants.%facet_value.585]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc31_37.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc31_37.4)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc31_37.5(%self.param: ref %DefaultOptionalStorage.1a2) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %has_value.ref: %DefaultOptionalStorage.elem.1ec = name_ref has_value, imports.%Core.import_ref.f1da [concrete = imports.%field_decl.431]
// CHECK:STDOUT:   %.loc31_37.1: ref bool = class_element_access %self.param, element1
// CHECK:STDOUT:   %facet_value.loc31_37.1: %facet_type.8a3 = facet_value bool, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.deb) [concrete = constants.%facet_value.a1b]
// CHECK:STDOUT:   %.loc31_37.2: %facet_type.8a3 = converted bool, %facet_value.loc31_37.1 [concrete = constants.%facet_value.a1b]
// CHECK:STDOUT:   %facet_value.loc31_37.2: %facet_type.8a3 = facet_value bool, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.deb) [concrete = constants.%facet_value.a1b]
// CHECK:STDOUT:   %.loc31_37.3: %facet_type.8a3 = converted bool, %facet_value.loc31_37.2 [concrete = constants.%facet_value.a1b]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc31_37.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc31_37.1)
// CHECK:STDOUT:   %value.ref: %DefaultOptionalStorage.elem.d17 = name_ref value, imports.%Core.import_ref.146b [concrete = imports.%field_decl.31e]
// CHECK:STDOUT:   %.loc31_37.4: ref %MaybeUnformed.9a0 = class_element_access %self.param, element0
// CHECK:STDOUT:   %facet_value.loc31_37.3: %facet_type.8a3 = facet_value constants.%MaybeUnformed.9a0, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.c48) [concrete = constants.%facet_value.585]
// CHECK:STDOUT:   %.loc31_37.5: %facet_type.8a3 = converted constants.%MaybeUnformed.9a0, %facet_value.loc31_37.3 [concrete = constants.%facet_value.585]
// CHECK:STDOUT:   %facet_value.loc31_37.4: %facet_type.8a3 = facet_value constants.%MaybeUnformed.9a0, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.c48) [concrete = constants.%facet_value.585]
// CHECK:STDOUT:   %.loc31_37.6: %facet_type.8a3 = converted constants.%MaybeUnformed.9a0, %facet_value.loc31_37.4 [concrete = constants.%facet_value.585]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc31_37.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc31_37.4)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc31_37.6(%self.param: ref %Optional.f64) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestRangeFor(%c.param: %ConstRange) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %i.patt: %pattern_type.c433 = value_binding_pattern i [concrete = constants.%i.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c.ref: %ConstRange = name_ref c, %c
// CHECK:STDOUT:   %impl.elem2: %.cf5 = impl_witness_access constants.%Iterate.impl_witness.acc, element2 [concrete = constants.%T.as_type.as.Iterate.impl.NewCursor.239]
// CHECK:STDOUT:   %bound_method.loc45_37.1: <bound method> = bound_method %c.ref, %impl.elem2
// CHECK:STDOUT:   %facet_value.loc45_37.1: %CppRange_where.type.d682d6.1 = facet_value constants.%ConstRange, (constants.%custom_witness.0a1, constants.%DestroyImpl.impl_witness.0ef, constants.%custom_witness.202, constants.%custom_witness.116, constants.%custom_witness.612b50.1, constants.%DestroyImpl.impl_witness.3fc, constants.%custom_witness.5be, constants.%custom_witness.c31, constants.%custom_witness.e21, constants.%custom_witness.612b50.2, constants.%DestroyImpl.impl_witness.72c) [concrete = constants.%facet_value.6f9]
// CHECK:STDOUT:   %.loc45_37.1: %CppRange_where.type.d682d6.1 = converted constants.%ConstRange, %facet_value.loc45_37.1 [concrete = constants.%facet_value.6f9]
// CHECK:STDOUT:   %facet_value.loc45_37.2: %CppRange_where.type.d682d6.1 = facet_value constants.%ConstRange, (constants.%custom_witness.0a1, constants.%DestroyImpl.impl_witness.0ef, constants.%custom_witness.202, constants.%custom_witness.116, constants.%custom_witness.612b50.1, constants.%DestroyImpl.impl_witness.3fc, constants.%custom_witness.5be, constants.%custom_witness.c31, constants.%custom_witness.e21, constants.%custom_witness.612b50.2, constants.%DestroyImpl.impl_witness.72c) [concrete = constants.%facet_value.6f9]
// CHECK:STDOUT:   %.loc45_37.2: %CppRange_where.type.d682d6.1 = converted constants.%ConstRange, %facet_value.loc45_37.2 [concrete = constants.%facet_value.6f9]
// CHECK:STDOUT:   %specific_fn.loc45_37.1: <specific function> = specific_function %impl.elem2, @T.as_type.as.Iterate.impl.NewCursor(constants.%facet_value.6f9) [concrete = constants.%T.as_type.as.Iterate.impl.NewCursor.specific_fn]
// CHECK:STDOUT:   %bound_method.loc45_37.2: <bound method> = bound_method %c.ref, %specific_fn.loc45_37.1
// CHECK:STDOUT:   %var: ref %tuple.type.7a6 = var_storage invalid
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.call: init %tuple.type.7a6 to %var = call %bound_method.loc45_37.2(%c.ref)
// CHECK:STDOUT:   assign %var, %T.as_type.as.Iterate.impl.NewCursor.call
// CHECK:STDOUT:   br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT: !for.next:
// CHECK:STDOUT:   %impl.elem3: %.ba3 = impl_witness_access constants.%Iterate.impl_witness.acc, element3 [concrete = constants.%T.as_type.as.Iterate.impl.Next.0d5]
// CHECK:STDOUT:   %bound_method.loc45_37.3: <bound method> = bound_method %c.ref, %impl.elem3
// CHECK:STDOUT:   %facet_value.loc45_37.3: %CppRange_where.type.d682d6.1 = facet_value constants.%ConstRange, (constants.%custom_witness.0a1, constants.%DestroyImpl.impl_witness.0ef, constants.%custom_witness.202, constants.%custom_witness.116, constants.%custom_witness.612b50.1, constants.%DestroyImpl.impl_witness.3fc, constants.%custom_witness.5be, constants.%custom_witness.c31, constants.%custom_witness.e21, constants.%custom_witness.612b50.2, constants.%DestroyImpl.impl_witness.72c) [concrete = constants.%facet_value.6f9]
// CHECK:STDOUT:   %.loc45_37.3: %CppRange_where.type.d682d6.1 = converted constants.%ConstRange, %facet_value.loc45_37.3 [concrete = constants.%facet_value.6f9]
// CHECK:STDOUT:   %facet_value.loc45_37.4: %CppRange_where.type.d682d6.1 = facet_value constants.%ConstRange, (constants.%custom_witness.0a1, constants.%DestroyImpl.impl_witness.0ef, constants.%custom_witness.202, constants.%custom_witness.116, constants.%custom_witness.612b50.1, constants.%DestroyImpl.impl_witness.3fc, constants.%custom_witness.5be, constants.%custom_witness.c31, constants.%custom_witness.e21, constants.%custom_witness.612b50.2, constants.%DestroyImpl.impl_witness.72c) [concrete = constants.%facet_value.6f9]
// CHECK:STDOUT:   %.loc45_37.4: %CppRange_where.type.d682d6.1 = converted constants.%ConstRange, %facet_value.loc45_37.4 [concrete = constants.%facet_value.6f9]
// CHECK:STDOUT:   %specific_fn.loc45_37.2: <specific function> = specific_function %impl.elem3, @T.as_type.as.Iterate.impl.Next(constants.%facet_value.6f9) [concrete = constants.%T.as_type.as.Iterate.impl.Next.specific_fn]
// CHECK:STDOUT:   %bound_method.loc45_37.4: <bound method> = bound_method %c.ref, %specific_fn.loc45_37.2
// CHECK:STDOUT:   %.loc45_37.5: ref %Optional.f64 = temporary_storage
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.call: init %Optional.f64 to %.loc45_37.5 = call %bound_method.loc45_37.4(%c.ref, %var)
// CHECK:STDOUT:   %.loc45_37.6: ref %Optional.f64 = temporary %.loc45_37.5, %T.as_type.as.Iterate.impl.Next.call
// CHECK:STDOUT:   %.loc45_37.7: %Optional.HasValue.type.c39 = specific_constant imports.%Core.import_ref.4d4, @Optional(constants.%OptionalStorage.facet.9e7) [concrete = constants.%Optional.HasValue.8b0]
// CHECK:STDOUT:   %HasValue.ref: %Optional.HasValue.type.c39 = name_ref HasValue, %.loc45_37.7 [concrete = constants.%Optional.HasValue.8b0]
// CHECK:STDOUT:   %Optional.HasValue.bound: <bound method> = bound_method %.loc45_37.6, %HasValue.ref
// CHECK:STDOUT:   %Optional.HasValue.specific_fn: <specific function> = specific_function %HasValue.ref, @Optional.HasValue(constants.%OptionalStorage.facet.9e7) [concrete = constants.%Optional.HasValue.specific_fn.418]
// CHECK:STDOUT:   %bound_method.loc45_37.5: <bound method> = bound_method %.loc45_37.6, %Optional.HasValue.specific_fn
// CHECK:STDOUT:   %.loc45_37.8: %Optional.f64 = acquire_value %.loc45_37.6
// CHECK:STDOUT:   %Optional.HasValue.call: init bool = call %bound_method.loc45_37.5(%.loc45_37.8)
// CHECK:STDOUT:   %.loc45_37.9: bool = value_of_initializer %Optional.HasValue.call
// CHECK:STDOUT:   %.loc45_37.10: bool = converted %Optional.HasValue.call, %.loc45_37.9
// CHECK:STDOUT:   if %.loc45_37.10 br !for.body else br !for.done
// CHECK:STDOUT:
// CHECK:STDOUT: !for.body:
// CHECK:STDOUT:   %.loc45_37.11: %Optional.Get.type.7e7 = specific_constant imports.%Core.import_ref.059, @Optional(constants.%OptionalStorage.facet.9e7) [concrete = constants.%Optional.Get.a0e]
// CHECK:STDOUT:   %Get.ref: %Optional.Get.type.7e7 = name_ref Get, %.loc45_37.11 [concrete = constants.%Optional.Get.a0e]
// CHECK:STDOUT:   %Optional.Get.bound: <bound method> = bound_method %.loc45_37.6, %Get.ref
// CHECK:STDOUT:   %Optional.Get.specific_fn: <specific function> = specific_function %Get.ref, @Optional.Get(constants.%OptionalStorage.facet.9e7) [concrete = constants.%Optional.Get.specific_fn.cb7]
// CHECK:STDOUT:   %bound_method.loc45_37.6: <bound method> = bound_method %.loc45_37.6, %Optional.Get.specific_fn
// CHECK:STDOUT:   %.loc45_37.12: ref %ValueType = temporary_storage
// CHECK:STDOUT:   %.loc45_37.13: %Optional.f64 = acquire_value %.loc45_37.6
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
// CHECK:STDOUT:   %facet_value.loc45_37.5: %facet_type.8a3 = facet_value constants.%ValueType, (constants.%custom_witness.0a1, constants.%DestroyImpl.impl_witness.0ef) [concrete = constants.%facet_value.6d5]
// CHECK:STDOUT:   %.loc45_37.16: %facet_type.8a3 = converted constants.%ValueType, %facet_value.loc45_37.5 [concrete = constants.%facet_value.6d5]
// CHECK:STDOUT:   %facet_value.loc45_37.6: %facet_type.8a3 = facet_value constants.%ValueType, (constants.%custom_witness.0a1, constants.%DestroyImpl.impl_witness.0ef) [concrete = constants.%facet_value.6d5]
// CHECK:STDOUT:   %.loc45_37.17: %facet_type.8a3 = converted constants.%ValueType, %facet_value.loc45_37.6 [concrete = constants.%facet_value.6d5]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc45_37.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc45_37.14)
// CHECK:STDOUT:   %facet_value.loc45_37.7: %facet_type.8a3 = facet_value constants.%Optional.f64, (constants.%custom_witness.b80ef6.7, constants.%DestroyImpl.impl_witness.59c) [concrete = constants.%facet_value.69a]
// CHECK:STDOUT:   %.loc45_37.18: %facet_type.8a3 = converted constants.%Optional.f64, %facet_value.loc45_37.7 [concrete = constants.%facet_value.69a]
// CHECK:STDOUT:   %facet_value.loc45_37.8: %facet_type.8a3 = facet_value constants.%Optional.f64, (constants.%custom_witness.b80ef6.7, constants.%DestroyImpl.impl_witness.59c) [concrete = constants.%facet_value.69a]
// CHECK:STDOUT:   %.loc45_37.19: %facet_type.8a3 = converted constants.%Optional.f64, %facet_value.loc45_37.8 [concrete = constants.%facet_value.69a]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc45_37.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc45_37.6)
// CHECK:STDOUT:   br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT: !for.done:
// CHECK:STDOUT:   %facet_value.loc45_37.9: %facet_type.8a3 = facet_value constants.%Optional.f64, (constants.%custom_witness.b80ef6.7, constants.%DestroyImpl.impl_witness.59c) [concrete = constants.%facet_value.69a]
// CHECK:STDOUT:   %.loc45_37.20: %facet_type.8a3 = converted constants.%Optional.f64, %facet_value.loc45_37.9 [concrete = constants.%facet_value.69a]
// CHECK:STDOUT:   %facet_value.loc45_37.10: %facet_type.8a3 = facet_value constants.%Optional.f64, (constants.%custom_witness.b80ef6.7, constants.%DestroyImpl.impl_witness.59c) [concrete = constants.%facet_value.69a]
// CHECK:STDOUT:   %.loc45_37.21: %facet_type.8a3 = converted constants.%Optional.f64, %facet_value.loc45_37.10 [concrete = constants.%facet_value.69a]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc45_37.3: init %empty_tuple.type = call %SelfDestruct.specific_fn.3(%.loc45_37.6)
// CHECK:STDOUT:   %facet_value.loc45_37.11: %facet_type.8a3 = facet_value constants.%tuple.type.7a6, (constants.%custom_witness.b80ef6.8, constants.%DestroyImpl.impl_witness.288) [concrete = constants.%facet_value.12b]
// CHECK:STDOUT:   %.loc45_37.22: %facet_type.8a3 = converted constants.%tuple.type.7a6, %facet_value.loc45_37.11 [concrete = constants.%facet_value.12b]
// CHECK:STDOUT:   %facet_value.loc45_37.12: %facet_type.8a3 = facet_value constants.%tuple.type.7a6, (constants.%custom_witness.b80ef6.8, constants.%DestroyImpl.impl_witness.288) [concrete = constants.%facet_value.12b]
// CHECK:STDOUT:   %.loc45_37.23: %facet_type.8a3 = converted constants.%tuple.type.7a6, %facet_value.loc45_37.12 [concrete = constants.%facet_value.12b]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc45_37.4: init %empty_tuple.type = call %SelfDestruct.specific_fn.4(%var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestIterate(constants.%R) {
// CHECK:STDOUT:   %R.patt.loc27_17.2 => constants.%R.patt
// CHECK:STDOUT:   %R.loc27_17.1 => constants.%R
// CHECK:STDOUT:   %R.as_type.loc27_79.1 => constants.%R.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.323
// CHECK:STDOUT:   %r.param_patt.loc27_77.2 => constants.%r.param_patt.77f
// CHECK:STDOUT:   %r.patt.loc27_77.2 => constants.%r.patt.e3b
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestIterate(constants.%facet_value.c45) {
// CHECK:STDOUT:   %R.patt.loc27_17.2 => constants.%R.patt
// CHECK:STDOUT:   %R.loc27_17.1 => constants.%facet_value.c45
// CHECK:STDOUT:   %R.as_type.loc27_79.1 => constants.%ConstRange
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.c1d
// CHECK:STDOUT:   %r.param_patt.loc27_77.2 => constants.%r.param_patt.ac5
// CHECK:STDOUT:   %r.patt.loc27_77.2 => constants.%r.patt.97f
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc27 => constants.%complete_type.357
// CHECK:STDOUT:   %Iterate.lookup_impl_witness => constants.%Iterate.impl_witness.acc
// CHECK:STDOUT:   %Iterate.facet.loc31_37.5 => constants.%Iterate.facet.30f
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type => constants.%Iterate.WithSelf.NewCursor.type.693
// CHECK:STDOUT:   %.loc31_37.26 => constants.%.cf5
// CHECK:STDOUT:   %impl.elem2.loc31_37.2 => constants.%T.as_type.as.Iterate.impl.NewCursor.239
// CHECK:STDOUT:   %specific_impl_fn.loc31_37.3 => constants.%T.as_type.as.Iterate.impl.NewCursor.specific_fn
// CHECK:STDOUT:   %impl.elem1 => constants.%facet_value.12b
// CHECK:STDOUT:   %as_type => constants.%tuple.type.7a6
// CHECK:STDOUT:   %require_complete.1 => constants.%complete_type.cf0
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type => constants.%Iterate.WithSelf.Next.type.872
// CHECK:STDOUT:   %.loc31_37.27 => constants.%.ba3
// CHECK:STDOUT:   %impl.elem3.loc31_37.2 => constants.%T.as_type.as.Iterate.impl.Next.0d5
// CHECK:STDOUT:   %specific_impl_fn.loc31_37.4 => constants.%T.as_type.as.Iterate.impl.Next.specific_fn
// CHECK:STDOUT:   %SelfDestruct.specific_fn.6 => constants.%SelfDestruct.specific_fn.102
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_adl_returns_different_types_mutable.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %Iterate.type: type = facet_type <@Iterate> [concrete]
// CHECK:STDOUT:   %OptionalStorage.type: type = facet_type <@OptionalStorage> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %facet_type.d61: type = facet_type <@SubobjectDestroy & @DestroyImpl & @Copy> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %T.220: %OptionalStorage.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Optional.Get.type.d97: type = fn_type @Optional.Get, @Optional(%T.220) [symbolic]
// CHECK:STDOUT:   %Optional.Get.3e9: %Optional.Get.type.d97 = struct_value () [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.type.cef: type = fn_type @Optional.HasValue, @Optional(%T.220) [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.a74: %Optional.HasValue.type.cef = struct_value () [symbolic]
// CHECK:STDOUT:   %facet_type.8a3: type = facet_type <@SubobjectDestroy & @DestroyImpl> [concrete]
// CHECK:STDOUT:   %T.9ab: %facet_type.d61 = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.19c: type = facet_access_type %T.9ab [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.lookup_impl_witness.855: <witness> = lookup_impl_witness %T.9ab, @SubobjectDestroy [symbolic]
// CHECK:STDOUT:   %DestroyImpl.lookup_impl_witness.ff9: <witness> = lookup_impl_witness %T.9ab, @DestroyImpl [symbolic]
// CHECK:STDOUT:   %facet_value.b35: %facet_type.8a3 = facet_value %T.as_type.19c, (%SubobjectDestroy.lookup_impl_witness.855, %DestroyImpl.lookup_impl_witness.ff9) [symbolic]
// CHECK:STDOUT:   %MaybeUnformed.413: type = class_type @MaybeUnformed, @MaybeUnformed(%facet_value.b35) [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.a6b: type = class_type @DefaultOptionalStorage, @DefaultOptionalStorage(%T.9ab) [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.elem.94e: type = unbound_element_type %DefaultOptionalStorage.a6b, bool [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.elem.c3f: type = unbound_element_type %DefaultOptionalStorage.a6b, %MaybeUnformed.413 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Get.type.b32: type = fn_type @T.as_type.as.OptionalStorage.impl.Get, @T.as_type.as.OptionalStorage.impl(%T.9ab) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Get.8b0: %T.as_type.as.OptionalStorage.impl.Get.type.b32 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Has.type.8e5: type = fn_type @T.as_type.as.OptionalStorage.impl.Has, @T.as_type.as.OptionalStorage.impl(%T.9ab) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Has.2d5: %T.as_type.as.OptionalStorage.impl.Has.type.8e5 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.273: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.5b5: %T.as_type.as.DestroyImpl.impl.Op.type.273 = struct_value () [symbolic]
// CHECK:STDOUT:   %.Self.500: %Iterate.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %ValueType: type = class_type @ValueType [concrete]
// CHECK:STDOUT:   %pattern_type.d2d: type = pattern_type %ValueType [concrete]
// CHECK:STDOUT:   %ValueType.Op.type.db6aa9.1: type = fn_type @ValueType.Op.1 [concrete]
// CHECK:STDOUT:   %ValueType.Op.4590ed.1: %ValueType.Op.type.db6aa9.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.55b: <witness> = custom_witness (%ValueType.Op.4590ed.1), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.5fb: %SubobjectDestroy.type = facet_value %ValueType, (%custom_witness.55b) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.636: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.5fb) [concrete]
// CHECK:STDOUT:   %const.91f: type = const_type %ValueType [concrete]
// CHECK:STDOUT:   %ptr.5f4: type = ptr_type %const.91f [concrete]
// CHECK:STDOUT:   %ptr.9c6: type = ptr_type %ValueType [concrete]
// CHECK:STDOUT:   %ValueType.Op.type.db6aa9.2: type = fn_type @ValueType.Op.2 [concrete]
// CHECK:STDOUT:   %ValueType.Op.4590ed.2: %ValueType.Op.type.db6aa9.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.587: <witness> = custom_witness (%ValueType.Op.4590ed.2), @Copy [concrete]
// CHECK:STDOUT:   %facet_value.040: %facet_type.d61 = facet_value %ValueType, (%custom_witness.55b, %DestroyImpl.impl_witness.636, %custom_witness.587) [concrete]
// CHECK:STDOUT:   %Iterate.lookup_impl_witness.1d8: <witness> = lookup_impl_witness %.Self.500, @Iterate [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.8a4: %facet_type.d61 = impl_witness_access %Iterate.lookup_impl_witness.1d8, element0 [symbolic_self]
// CHECK:STDOUT:   %Iterate_where.type.db0: type = facet_type <@Iterate where %impl.elem0.8a4 = %facet_value.040> [concrete]
// CHECK:STDOUT:   %pattern_type.c3f: type = pattern_type %Iterate_where.type.db0 [concrete]
// CHECK:STDOUT:   %R.patt: %pattern_type.c3f = symbolic_binding_pattern R, 0 [symbolic]
// CHECK:STDOUT:   %R: %Iterate_where.type.db0 = symbolic_binding R, 0 [symbolic]
// CHECK:STDOUT:   %R.as_type: type = facet_access_type %R [symbolic]
// CHECK:STDOUT:   %pattern_type.fc7: type = pattern_type %R.as_type [symbolic]
// CHECK:STDOUT:   %r.param_patt: %pattern_type.fc7 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %r.patt: %pattern_type.fc7 = wrapper_binding_pattern r, %r.param_patt [symbolic]
// CHECK:STDOUT:   %i.patt: %pattern_type.d2d = value_binding_pattern i [concrete]
// CHECK:STDOUT:   %Iterate.lookup_impl_witness.63d: <witness> = lookup_impl_witness %R, @Iterate [symbolic]
// CHECK:STDOUT:   %Iterate.facet.561: %Iterate.type = facet_value %R.as_type, (%Iterate.lookup_impl_witness.63d) [symbolic]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.89f: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.561) [symbolic]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.b09: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.561) [symbolic]
// CHECK:STDOUT:   %.80d: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.89f, %Iterate.facet.561 [symbolic]
// CHECK:STDOUT:   %impl.elem2.6ca: %.80d = impl_witness_access %Iterate.lookup_impl_witness.63d, element2 [symbolic]
// CHECK:STDOUT:   %impl.elem1.de6: %facet_type.8a3 = impl_witness_access %Iterate.lookup_impl_witness.63d, element1 [symbolic]
// CHECK:STDOUT:   %as_type.922: type = facet_access_type %impl.elem1.de6 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.223: <specific function> = specific_impl_function %impl.elem2.6ca, @Iterate.WithSelf.NewCursor(%Iterate.facet.561) [symbolic]
// CHECK:STDOUT:   %.a4d: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.b09, %Iterate.facet.561 [symbolic]
// CHECK:STDOUT:   %impl.elem3.30f: %.a4d = impl_witness_access %Iterate.lookup_impl_witness.63d, element3 [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.4ef: type = class_type @DefaultOptionalStorage, @DefaultOptionalStorage(%facet_value.040) [concrete]
// CHECK:STDOUT:   %OptionalStorage.impl_witness.dc9: <witness> = impl_witness imports.%OptionalStorage.impl_witness_table.968, @T.as_type.as.OptionalStorage.impl(%facet_value.040) [concrete]
// CHECK:STDOUT:   %OptionalStorage.facet.546: %OptionalStorage.type = facet_value %ValueType, (%OptionalStorage.impl_witness.dc9) [concrete]
// CHECK:STDOUT:   %Optional.99c: type = class_type @Optional, @Optional(%OptionalStorage.facet.546) [concrete]
// CHECK:STDOUT:   %specific_impl_fn.da3: <specific function> = specific_impl_function %impl.elem3.30f, @Iterate.WithSelf.Next(%Iterate.facet.561) [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.type.37c: type = fn_type @Optional.HasValue, @Optional(%OptionalStorage.facet.546) [concrete]
// CHECK:STDOUT:   %Optional.HasValue.a23: %Optional.HasValue.type.37c = struct_value () [concrete]
// CHECK:STDOUT:   %Optional.Get.type.6e2: type = fn_type @Optional.Get, @Optional(%OptionalStorage.facet.546) [concrete]
// CHECK:STDOUT:   %Optional.Get.8ad: %Optional.Get.type.6e2 = struct_value () [concrete]
// CHECK:STDOUT:   %facet_value.b6f: %facet_type.8a3 = facet_value %ValueType, (%custom_witness.55b, %DestroyImpl.impl_witness.636) [concrete]
// CHECK:STDOUT:   %MaybeUnformed.5bf: type = class_type @MaybeUnformed, @MaybeUnformed(%facet_value.b6f) [concrete]
// CHECK:STDOUT:   %.dd1: type = maybe_unformed_type %ValueType [concrete]
// CHECK:STDOUT:   %DefaultOptionalStorage.elem.5fc: type = unbound_element_type %DefaultOptionalStorage.4ef, %MaybeUnformed.5bf [concrete]
// CHECK:STDOUT:   %DefaultOptionalStorage.elem.8a7: type = unbound_element_type %DefaultOptionalStorage.4ef, bool [concrete]
// CHECK:STDOUT:   %struct_type.value.has_value.dcf: type = struct_type {.value: %MaybeUnformed.5bf, .has_value: bool} [concrete]
// CHECK:STDOUT:   %Optional.HasValue.specific_fn: <specific function> = specific_function %Optional.HasValue.a23, @Optional.HasValue(%OptionalStorage.facet.546) [concrete]
// CHECK:STDOUT:   %Optional.Get.specific_fn: <specific function> = specific_function %Optional.Get.8ad, @Optional.Get(%OptionalStorage.facet.546) [concrete]
// CHECK:STDOUT:   %operator_PlusEqual__carbon_thunk.type: type = fn_type @operator_PlusEqual__carbon_thunk [concrete]
// CHECK:STDOUT:   %operator_PlusEqual__carbon_thunk: %operator_PlusEqual__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.3: type = fn_type @SubobjectDestroy.WithSelf.Op.loc31_44.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.3: %SubobjectDestroy.WithSelf.Op.type.6870fd.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.3: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.3), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.4: type = fn_type @SubobjectDestroy.WithSelf.Op.loc31_44.3 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.4: %SubobjectDestroy.WithSelf.Op.type.6870fd.4 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.4: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.4), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.a87: %SubobjectDestroy.type = facet_value bool, (%custom_witness.b80ef6.4) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.deb: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.a87) [concrete]
// CHECK:STDOUT:   %facet_value.a1b: %facet_type.8a3 = facet_value bool, (%custom_witness.b80ef6.4, %DestroyImpl.impl_witness.deb) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.2b7: %SubobjectDestroy.type = facet_value %MaybeUnformed.5bf, (%custom_witness.b80ef6.3) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.bc3: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.2b7) [concrete]
// CHECK:STDOUT:   %facet_value.8e9: %facet_type.8a3 = facet_value %MaybeUnformed.5bf, (%custom_witness.b80ef6.3, %DestroyImpl.impl_witness.bc3) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.7: type = fn_type @SubobjectDestroy.WithSelf.Op.loc31_44.6 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.7: %SubobjectDestroy.WithSelf.Op.type.6870fd.7 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.7: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.7), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.9e4: %SubobjectDestroy.type = facet_value %Optional.99c, (%custom_witness.b80ef6.7) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.788: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.9e4) [concrete]
// CHECK:STDOUT:   %facet_value.5d4: %facet_type.8a3 = facet_value %Optional.99c, (%custom_witness.b80ef6.7, %DestroyImpl.impl_witness.788) [concrete]
// CHECK:STDOUT:   %MutableRange: type = class_type @MutableRange [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .DiffMutableADL = %DiffMutableADL
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.4d4: @Optional.%Optional.HasValue.type (%Optional.HasValue.type.cef) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.HasValue (constants.%Optional.HasValue.a74)]
// CHECK:STDOUT:   %Core.import_ref.059: @Optional.%Optional.Get.type (%Optional.Get.type.d97) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.Get (constants.%Optional.Get.3e9)]
// CHECK:STDOUT:   %Core.import_ref.7bb: type = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%DefaultOptionalStorage (constants.%DefaultOptionalStorage.a6b)]
// CHECK:STDOUT:   %Core.import_ref.d2b = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], unloaded
// CHECK:STDOUT:   %Core.import_ref.370 = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], unloaded
// CHECK:STDOUT:   %Core.import_ref.6ba4: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Has.type (%T.as_type.as.OptionalStorage.impl.Has.type.8e5) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Has (constants.%T.as_type.as.OptionalStorage.impl.Has.2d5)]
// CHECK:STDOUT:   %Core.import_ref.73d: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Get.type (%T.as_type.as.OptionalStorage.impl.Get.type.b32) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Get (constants.%T.as_type.as.OptionalStorage.impl.Get.8b0)]
// CHECK:STDOUT:   %Core.import_ref.778 = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], unloaded
// CHECK:STDOUT:   %OptionalStorage.impl_witness_table.968 = impl_witness_table (%Core.import_ref.7bb, %Core.import_ref.d2b, %Core.import_ref.370, %Core.import_ref.6ba4, %Core.import_ref.73d, %Core.import_ref.778), @T.as_type.as.OptionalStorage.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.146b: @DefaultOptionalStorage.%DefaultOptionalStorage.elem.1 (%DefaultOptionalStorage.elem.c3f) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [concrete = %field_decl.31e]
// CHECK:STDOUT:   %Core.import_ref.f1da: @DefaultOptionalStorage.%DefaultOptionalStorage.elem.2 (%DefaultOptionalStorage.elem.94e) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [concrete = %field_decl.431]
// CHECK:STDOUT:   %Core.import_ref.14b: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.273) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.5b5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.14b), @T.as_type.as.DestroyImpl.impl [concrete]
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
// CHECK:STDOUT:   %Core.import_ref.b0d = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %field_decl.431: @DefaultOptionalStorage.%DefaultOptionalStorage.elem.2 (%DefaultOptionalStorage.elem.94e) = field_decl has_value, element1, %Core.import_ref.b0d in [concrete] {}
// CHECK:STDOUT:   %Core.import_ref.164c = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %field_decl.31e: @DefaultOptionalStorage.%DefaultOptionalStorage.elem.1 (%DefaultOptionalStorage.elem.c3f) = field_decl value, element0, %Core.import_ref.164c in [concrete] {}
// CHECK:STDOUT:   %MutableRange.decl: type = class_decl @MutableRange [concrete = constants.%MutableRange] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @TestIterate(%R.loc27_17.2: %Iterate_where.type.db0) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Iterate.lookup_impl_witness: <witness> = lookup_impl_witness %R.loc27_17.1, @Iterate [symbolic = %Iterate.lookup_impl_witness (constants.%Iterate.lookup_impl_witness.63d)]
// CHECK:STDOUT:   %Iterate.facet.loc31_44.5: %Iterate.type = facet_value %R.as_type.loc27_86.1, (%Iterate.lookup_impl_witness) [symbolic = %Iterate.facet.loc31_44.5 (constants.%Iterate.facet.561)]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.loc31_44.5) [symbolic = %Iterate.WithSelf.NewCursor.type (constants.%Iterate.WithSelf.NewCursor.type.89f)]
// CHECK:STDOUT:   %.loc31_44.26: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type, %Iterate.facet.loc31_44.5 [symbolic = %.loc31_44.26 (constants.%.80d)]
// CHECK:STDOUT:   %impl.elem2.loc31_44.2: @TestIterate.%.loc31_44.26 (%.80d) = impl_witness_access %Iterate.lookup_impl_witness, element2 [symbolic = %impl.elem2.loc31_44.2 (constants.%impl.elem2.6ca)]
// CHECK:STDOUT:   %specific_impl_fn.loc31_44.3: <specific function> = specific_impl_function %impl.elem2.loc31_44.2, @Iterate.WithSelf.NewCursor(%Iterate.facet.loc31_44.5) [symbolic = %specific_impl_fn.loc31_44.3 (constants.%specific_impl_fn.223)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.loc31_44.5) [symbolic = %Iterate.WithSelf.Next.type (constants.%Iterate.WithSelf.Next.type.b09)]
// CHECK:STDOUT:   %.loc31_44.27: type = fn_type_with_self_type %Iterate.WithSelf.Next.type, %Iterate.facet.loc31_44.5 [symbolic = %.loc31_44.27 (constants.%.a4d)]
// CHECK:STDOUT:   %impl.elem3.loc31_44.2: @TestIterate.%.loc31_44.27 (%.a4d) = impl_witness_access %Iterate.lookup_impl_witness, element3 [symbolic = %impl.elem3.loc31_44.2 (constants.%impl.elem3.30f)]
// CHECK:STDOUT:   %specific_impl_fn.loc31_44.4: <specific function> = specific_impl_function %impl.elem3.loc31_44.2, @Iterate.WithSelf.Next(%Iterate.facet.loc31_44.5) [symbolic = %specific_impl_fn.loc31_44.4 (constants.%specific_impl_fn.da3)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%r.param: @TestIterate.%R.as_type.loc27_86.1 (%R.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %i.patt: %pattern_type.d2d = value_binding_pattern i [concrete = constants.%i.patt]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %r.ref: @TestIterate.%R.as_type.loc27_86.1 (%R.as_type) = name_ref r, %r
// CHECK:STDOUT:     %impl.elem2.loc31_44.1: @TestIterate.%.loc31_44.26 (%.80d) = impl_witness_access constants.%Iterate.lookup_impl_witness.63d, element2 [symbolic = %impl.elem2.loc31_44.2 (constants.%impl.elem2.6ca)]
// CHECK:STDOUT:     %bound_method.loc31_44.1: <bound method> = bound_method %r.ref, %impl.elem2.loc31_44.1
// CHECK:STDOUT:     %Iterate.facet.loc31_44.1: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.63d) [symbolic = %Iterate.facet.loc31_44.5 (constants.%Iterate.facet.561)]
// CHECK:STDOUT:     %.loc31_44.1: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc31_44.1 [symbolic = %Iterate.facet.loc31_44.5 (constants.%Iterate.facet.561)]
// CHECK:STDOUT:     %Iterate.facet.loc31_44.2: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.63d) [symbolic = %Iterate.facet.loc31_44.5 (constants.%Iterate.facet.561)]
// CHECK:STDOUT:     %.loc31_44.2: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc31_44.2 [symbolic = %Iterate.facet.loc31_44.5 (constants.%Iterate.facet.561)]
// CHECK:STDOUT:     %specific_impl_fn.loc31_44.1: <specific function> = specific_impl_function %impl.elem2.loc31_44.1, @Iterate.WithSelf.NewCursor(constants.%Iterate.facet.561) [symbolic = %specific_impl_fn.loc31_44.3 (constants.%specific_impl_fn.223)]
// CHECK:STDOUT:     %bound_method.loc31_44.2: <bound method> = bound_method %r.ref, %specific_impl_fn.loc31_44.1
// CHECK:STDOUT:     %var: ref @TestIterate.%as_type (%as_type.922) = var_storage invalid
// CHECK:STDOUT:     %Iterate.WithSelf.NewCursor.call: init @TestIterate.%as_type (%as_type.922) to %var = call %bound_method.loc31_44.2(%r.ref)
// CHECK:STDOUT:     assign %var, %Iterate.WithSelf.NewCursor.call
// CHECK:STDOUT:     br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.next:
// CHECK:STDOUT:     %impl.elem3.loc31_44.1: @TestIterate.%.loc31_44.27 (%.a4d) = impl_witness_access constants.%Iterate.lookup_impl_witness.63d, element3 [symbolic = %impl.elem3.loc31_44.2 (constants.%impl.elem3.30f)]
// CHECK:STDOUT:     %bound_method.loc31_44.3: <bound method> = bound_method %r.ref, %impl.elem3.loc31_44.1
// CHECK:STDOUT:     %Iterate.facet.loc31_44.3: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.63d) [symbolic = %Iterate.facet.loc31_44.5 (constants.%Iterate.facet.561)]
// CHECK:STDOUT:     %.loc31_44.3: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc31_44.3 [symbolic = %Iterate.facet.loc31_44.5 (constants.%Iterate.facet.561)]
// CHECK:STDOUT:     %Iterate.facet.loc31_44.4: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.63d) [symbolic = %Iterate.facet.loc31_44.5 (constants.%Iterate.facet.561)]
// CHECK:STDOUT:     %.loc31_44.4: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc31_44.4 [symbolic = %Iterate.facet.loc31_44.5 (constants.%Iterate.facet.561)]
// CHECK:STDOUT:     %.loc31_44.5: %facet_type.8a3 = converted constants.%as_type.922, constants.%impl.elem1.de6 [symbolic = %impl.elem1 (constants.%impl.elem1.de6)]
// CHECK:STDOUT:     %.loc31_44.6: %facet_type.8a3 = converted constants.%as_type.922, constants.%impl.elem1.de6 [symbolic = %impl.elem1 (constants.%impl.elem1.de6)]
// CHECK:STDOUT:     %specific_impl_fn.loc31_44.2: <specific function> = specific_impl_function %impl.elem3.loc31_44.1, @Iterate.WithSelf.Next(constants.%Iterate.facet.561) [symbolic = %specific_impl_fn.loc31_44.4 (constants.%specific_impl_fn.da3)]
// CHECK:STDOUT:     %bound_method.loc31_44.4: <bound method> = bound_method %r.ref, %specific_impl_fn.loc31_44.2
// CHECK:STDOUT:     %.loc31_44.7: ref %Optional.99c = temporary_storage
// CHECK:STDOUT:     %Iterate.WithSelf.Next.call: init %Optional.99c to %.loc31_44.7 = call %bound_method.loc31_44.4(%r.ref, %var)
// CHECK:STDOUT:     %.loc31_44.8: ref %Optional.99c = temporary %.loc31_44.7, %Iterate.WithSelf.Next.call
// CHECK:STDOUT:     %.loc31_44.9: %Optional.HasValue.type.37c = specific_constant imports.%Core.import_ref.4d4, @Optional(constants.%OptionalStorage.facet.546) [concrete = constants.%Optional.HasValue.a23]
// CHECK:STDOUT:     %HasValue.ref: %Optional.HasValue.type.37c = name_ref HasValue, %.loc31_44.9 [concrete = constants.%Optional.HasValue.a23]
// CHECK:STDOUT:     %Optional.HasValue.bound: <bound method> = bound_method %.loc31_44.8, %HasValue.ref
// CHECK:STDOUT:     %Optional.HasValue.specific_fn: <specific function> = specific_function %HasValue.ref, @Optional.HasValue(constants.%OptionalStorage.facet.546) [concrete = constants.%Optional.HasValue.specific_fn]
// CHECK:STDOUT:     %bound_method.loc31_44.5: <bound method> = bound_method %.loc31_44.8, %Optional.HasValue.specific_fn
// CHECK:STDOUT:     %.loc31_44.10: %Optional.99c = acquire_value %.loc31_44.8
// CHECK:STDOUT:     %Optional.HasValue.call: init bool = call %bound_method.loc31_44.5(%.loc31_44.10)
// CHECK:STDOUT:     %.loc31_44.11: bool = value_of_initializer %Optional.HasValue.call
// CHECK:STDOUT:     %.loc31_44.12: bool = converted %Optional.HasValue.call, %.loc31_44.11
// CHECK:STDOUT:     if %.loc31_44.12 br !for.body else br !for.done
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.body:
// CHECK:STDOUT:     %.loc31_44.13: %Optional.Get.type.6e2 = specific_constant imports.%Core.import_ref.059, @Optional(constants.%OptionalStorage.facet.546) [concrete = constants.%Optional.Get.8ad]
// CHECK:STDOUT:     %Get.ref: %Optional.Get.type.6e2 = name_ref Get, %.loc31_44.13 [concrete = constants.%Optional.Get.8ad]
// CHECK:STDOUT:     %Optional.Get.bound: <bound method> = bound_method %.loc31_44.8, %Get.ref
// CHECK:STDOUT:     %Optional.Get.specific_fn: <specific function> = specific_function %Get.ref, @Optional.Get(constants.%OptionalStorage.facet.546) [concrete = constants.%Optional.Get.specific_fn]
// CHECK:STDOUT:     %bound_method.loc31_44.6: <bound method> = bound_method %.loc31_44.8, %Optional.Get.specific_fn
// CHECK:STDOUT:     %.loc31_44.14: ref %ValueType = temporary_storage
// CHECK:STDOUT:     %.loc31_44.15: %Optional.99c = acquire_value %.loc31_44.8
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
// CHECK:STDOUT:     %facet_value.loc31_44.1: %facet_type.8a3 = facet_value constants.%ValueType, (constants.%custom_witness.55b, constants.%DestroyImpl.impl_witness.636) [concrete = constants.%facet_value.b6f]
// CHECK:STDOUT:     %.loc31_44.18: %facet_type.8a3 = converted constants.%ValueType, %facet_value.loc31_44.1 [concrete = constants.%facet_value.b6f]
// CHECK:STDOUT:     %facet_value.loc31_44.2: %facet_type.8a3 = facet_value constants.%ValueType, (constants.%custom_witness.55b, constants.%DestroyImpl.impl_witness.636) [concrete = constants.%facet_value.b6f]
// CHECK:STDOUT:     %.loc31_44.19: %facet_type.8a3 = converted constants.%ValueType, %facet_value.loc31_44.2 [concrete = constants.%facet_value.b6f]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %SelfDestruct.call.loc31_44.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc31_44.16)
// CHECK:STDOUT:     %facet_value.loc31_44.3: %facet_type.8a3 = facet_value constants.%Optional.99c, (constants.%custom_witness.b80ef6.7, constants.%DestroyImpl.impl_witness.788) [concrete = constants.%facet_value.5d4]
// CHECK:STDOUT:     %.loc31_44.20: %facet_type.8a3 = converted constants.%Optional.99c, %facet_value.loc31_44.3 [concrete = constants.%facet_value.5d4]
// CHECK:STDOUT:     %facet_value.loc31_44.4: %facet_type.8a3 = facet_value constants.%Optional.99c, (constants.%custom_witness.b80ef6.7, constants.%DestroyImpl.impl_witness.788) [concrete = constants.%facet_value.5d4]
// CHECK:STDOUT:     %.loc31_44.21: %facet_type.8a3 = converted constants.%Optional.99c, %facet_value.loc31_44.4 [concrete = constants.%facet_value.5d4]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %SelfDestruct.call.loc31_44.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc31_44.8)
// CHECK:STDOUT:     br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.done:
// CHECK:STDOUT:     %facet_value.loc31_44.5: %facet_type.8a3 = facet_value constants.%Optional.99c, (constants.%custom_witness.b80ef6.7, constants.%DestroyImpl.impl_witness.788) [concrete = constants.%facet_value.5d4]
// CHECK:STDOUT:     %.loc31_44.22: %facet_type.8a3 = converted constants.%Optional.99c, %facet_value.loc31_44.5 [concrete = constants.%facet_value.5d4]
// CHECK:STDOUT:     %facet_value.loc31_44.6: %facet_type.8a3 = facet_value constants.%Optional.99c, (constants.%custom_witness.b80ef6.7, constants.%DestroyImpl.impl_witness.788) [concrete = constants.%facet_value.5d4]
// CHECK:STDOUT:     %.loc31_44.23: %facet_type.8a3 = converted constants.%Optional.99c, %facet_value.loc31_44.6 [concrete = constants.%facet_value.5d4]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %SelfDestruct.call.loc31_44.3: init %empty_tuple.type = call %SelfDestruct.specific_fn.3(%.loc31_44.8)
// CHECK:STDOUT:     %.loc31_44.24: %facet_type.8a3 = converted constants.%as_type.922, constants.%impl.elem1.de6 [symbolic = %impl.elem1 (constants.%impl.elem1.de6)]
// CHECK:STDOUT:     %.loc31_44.25: %facet_type.8a3 = converted constants.%as_type.922, constants.%impl.elem1.de6 [symbolic = %impl.elem1 (constants.%impl.elem1.de6)]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %SelfDestruct.call.loc31_44.4: init %empty_tuple.type = call %SelfDestruct.specific_fn.4(%var)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc31_44.1(%self.param: ref %.dd1) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc31_44.2(%self.param: ref %MaybeUnformed.5bf) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc31_44.3(%self.param: ref bool) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc31_44.4(%self.param: ref %struct_type.value.has_value.dcf) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc31_44.1: ref bool = struct_access %self.param, element1
// CHECK:STDOUT:   %facet_value.loc31_44.1: %facet_type.8a3 = facet_value bool, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.deb) [concrete = constants.%facet_value.a1b]
// CHECK:STDOUT:   %.loc31_44.2: %facet_type.8a3 = converted bool, %facet_value.loc31_44.1 [concrete = constants.%facet_value.a1b]
// CHECK:STDOUT:   %facet_value.loc31_44.2: %facet_type.8a3 = facet_value bool, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.deb) [concrete = constants.%facet_value.a1b]
// CHECK:STDOUT:   %.loc31_44.3: %facet_type.8a3 = converted bool, %facet_value.loc31_44.2 [concrete = constants.%facet_value.a1b]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc31_44.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc31_44.1)
// CHECK:STDOUT:   %.loc31_44.4: ref %MaybeUnformed.5bf = struct_access %self.param, element0
// CHECK:STDOUT:   %facet_value.loc31_44.3: %facet_type.8a3 = facet_value constants.%MaybeUnformed.5bf, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.bc3) [concrete = constants.%facet_value.8e9]
// CHECK:STDOUT:   %.loc31_44.5: %facet_type.8a3 = converted constants.%MaybeUnformed.5bf, %facet_value.loc31_44.3 [concrete = constants.%facet_value.8e9]
// CHECK:STDOUT:   %facet_value.loc31_44.4: %facet_type.8a3 = facet_value constants.%MaybeUnformed.5bf, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.bc3) [concrete = constants.%facet_value.8e9]
// CHECK:STDOUT:   %.loc31_44.6: %facet_type.8a3 = converted constants.%MaybeUnformed.5bf, %facet_value.loc31_44.4 [concrete = constants.%facet_value.8e9]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc31_44.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc31_44.4)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc31_44.5(%self.param: ref %DefaultOptionalStorage.4ef) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %has_value.ref: %DefaultOptionalStorage.elem.8a7 = name_ref has_value, imports.%Core.import_ref.f1da [concrete = imports.%field_decl.431]
// CHECK:STDOUT:   %.loc31_44.1: ref bool = class_element_access %self.param, element1
// CHECK:STDOUT:   %facet_value.loc31_44.1: %facet_type.8a3 = facet_value bool, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.deb) [concrete = constants.%facet_value.a1b]
// CHECK:STDOUT:   %.loc31_44.2: %facet_type.8a3 = converted bool, %facet_value.loc31_44.1 [concrete = constants.%facet_value.a1b]
// CHECK:STDOUT:   %facet_value.loc31_44.2: %facet_type.8a3 = facet_value bool, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.deb) [concrete = constants.%facet_value.a1b]
// CHECK:STDOUT:   %.loc31_44.3: %facet_type.8a3 = converted bool, %facet_value.loc31_44.2 [concrete = constants.%facet_value.a1b]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc31_44.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc31_44.1)
// CHECK:STDOUT:   %value.ref: %DefaultOptionalStorage.elem.5fc = name_ref value, imports.%Core.import_ref.146b [concrete = imports.%field_decl.31e]
// CHECK:STDOUT:   %.loc31_44.4: ref %MaybeUnformed.5bf = class_element_access %self.param, element0
// CHECK:STDOUT:   %facet_value.loc31_44.3: %facet_type.8a3 = facet_value constants.%MaybeUnformed.5bf, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.bc3) [concrete = constants.%facet_value.8e9]
// CHECK:STDOUT:   %.loc31_44.5: %facet_type.8a3 = converted constants.%MaybeUnformed.5bf, %facet_value.loc31_44.3 [concrete = constants.%facet_value.8e9]
// CHECK:STDOUT:   %facet_value.loc31_44.4: %facet_type.8a3 = facet_value constants.%MaybeUnformed.5bf, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.bc3) [concrete = constants.%facet_value.8e9]
// CHECK:STDOUT:   %.loc31_44.6: %facet_type.8a3 = converted constants.%MaybeUnformed.5bf, %facet_value.loc31_44.4 [concrete = constants.%facet_value.8e9]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc31_44.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc31_44.4)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc31_44.6(%self.param: ref %Optional.99c) = "no_op";
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
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.fc7
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
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %facet_type.d61: type = facet_type <@SubobjectDestroy & @DestroyImpl & @Copy> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %T.220: %OptionalStorage.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Optional.Get.type.d97: type = fn_type @Optional.Get, @Optional(%T.220) [symbolic]
// CHECK:STDOUT:   %Optional.Get.3e9: %Optional.Get.type.d97 = struct_value () [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.type.cef: type = fn_type @Optional.HasValue, @Optional(%T.220) [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.a74: %Optional.HasValue.type.cef = struct_value () [symbolic]
// CHECK:STDOUT:   %facet_type.8a3: type = facet_type <@SubobjectDestroy & @DestroyImpl> [concrete]
// CHECK:STDOUT:   %T.9ab: %facet_type.d61 = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.19c: type = facet_access_type %T.9ab [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.lookup_impl_witness.855: <witness> = lookup_impl_witness %T.9ab, @SubobjectDestroy [symbolic]
// CHECK:STDOUT:   %DestroyImpl.lookup_impl_witness.ff9: <witness> = lookup_impl_witness %T.9ab, @DestroyImpl [symbolic]
// CHECK:STDOUT:   %facet_value.b35: %facet_type.8a3 = facet_value %T.as_type.19c, (%SubobjectDestroy.lookup_impl_witness.855, %DestroyImpl.lookup_impl_witness.ff9) [symbolic]
// CHECK:STDOUT:   %MaybeUnformed.413: type = class_type @MaybeUnformed, @MaybeUnformed(%facet_value.b35) [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.a6b: type = class_type @DefaultOptionalStorage, @DefaultOptionalStorage(%T.9ab) [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.elem.94e: type = unbound_element_type %DefaultOptionalStorage.a6b, bool [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.elem.c3f: type = unbound_element_type %DefaultOptionalStorage.a6b, %MaybeUnformed.413 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Get.type.b32: type = fn_type @T.as_type.as.OptionalStorage.impl.Get, @T.as_type.as.OptionalStorage.impl(%T.9ab) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Get.8b0: %T.as_type.as.OptionalStorage.impl.Get.type.b32 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Has.type.8e5: type = fn_type @T.as_type.as.OptionalStorage.impl.Has, @T.as_type.as.OptionalStorage.impl(%T.9ab) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Has.2d5: %T.as_type.as.OptionalStorage.impl.Has.type.8e5 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Some.type.61d: type = fn_type @T.as_type.as.OptionalStorage.impl.Some, @T.as_type.as.OptionalStorage.impl(%T.9ab) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Some.ba7: %T.as_type.as.OptionalStorage.impl.Some.type.61d = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.None.type.a9f: type = fn_type @T.as_type.as.OptionalStorage.impl.None, @T.as_type.as.OptionalStorage.impl(%T.9ab) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.None.083: %T.as_type.as.OptionalStorage.impl.None.type.a9f = struct_value () [symbolic]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.273: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.5b5: %T.as_type.as.DestroyImpl.impl.Op.type.273 = struct_value () [symbolic]
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
// CHECK:STDOUT:   %SubobjectDestroy.lookup_impl_witness.dc9: <witness> = lookup_impl_witness %tuple.type.1ba, @SubobjectDestroy [symbolic]
// CHECK:STDOUT:   %DestroyImpl.lookup_impl_witness.504: <witness> = lookup_impl_witness %tuple.type.1ba, @DestroyImpl [symbolic]
// CHECK:STDOUT:   %facet_value.bf6: %facet_type.8a3 = facet_value %tuple.type.1ba, (%SubobjectDestroy.lookup_impl_witness.dc9, %DestroyImpl.lookup_impl_witness.504) [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.lookup_impl_witness.b15da7.2: <witness> = lookup_impl_witness %impl.elem0.493765.2, @SubobjectDestroy [symbolic]
// CHECK:STDOUT:   %DestroyImpl.lookup_impl_witness.4e7f44.2: <witness> = lookup_impl_witness %impl.elem0.493765.2, @DestroyImpl [symbolic]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.167: <witness> = lookup_impl_witness %impl.elem0.493765.2, @Copy [symbolic]
// CHECK:STDOUT:   %facet_value.9da: %facet_type.d61 = facet_value %impl.elem0.493765.2, (%SubobjectDestroy.lookup_impl_witness.b15da7.2, %DestroyImpl.lookup_impl_witness.4e7f44.2, %Copy.lookup_impl_witness.167) [symbolic]
// CHECK:STDOUT:   %SelfDestruct.type: type = fn_type @SelfDestruct [concrete]
// CHECK:STDOUT:   %SelfDestruct: %SelfDestruct.type = struct_value () [concrete]
// CHECK:STDOUT:   %ValueType: type = class_type @ValueType [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %pattern_type.e09: type = pattern_type %ValueType [concrete]
// CHECK:STDOUT:   %ValueType.Op.type.9ce105.1: type = fn_type @ValueType.Op.1 [concrete]
// CHECK:STDOUT:   %ValueType.Op.038ea9.1: %ValueType.Op.type.9ce105.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.656: <witness> = custom_witness (%ValueType.Op.038ea9.1), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.536: %SubobjectDestroy.type = facet_value %ValueType, (%custom_witness.656) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.261: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.536) [concrete]
// CHECK:STDOUT:   %const.924: type = const_type %ValueType [concrete]
// CHECK:STDOUT:   %ptr.403: type = ptr_type %const.924 [concrete]
// CHECK:STDOUT:   %ptr.6ed: type = ptr_type %ValueType [concrete]
// CHECK:STDOUT:   %ValueType.Op.type.9ce105.2: type = fn_type @ValueType.Op.2 [concrete]
// CHECK:STDOUT:   %ValueType.Op.038ea9.2: %ValueType.Op.type.9ce105.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.1c5: <witness> = custom_witness (%ValueType.Op.038ea9.2), @Copy [concrete]
// CHECK:STDOUT:   %facet_value.34c: %facet_type.d61 = facet_value %ValueType, (%custom_witness.656, %DestroyImpl.impl_witness.261, %custom_witness.1c5) [concrete]
// CHECK:STDOUT:   %Iterate.lookup_impl_witness.1d8: <witness> = lookup_impl_witness %.Self.500, @Iterate [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.8a4: %facet_type.d61 = impl_witness_access %Iterate.lookup_impl_witness.1d8, element0 [symbolic_self]
// CHECK:STDOUT:   %Iterate_where.type.fc8: type = facet_type <@Iterate where %impl.elem0.8a4 = %facet_value.34c> [concrete]
// CHECK:STDOUT:   %pattern_type.d98: type = pattern_type %Iterate_where.type.fc8 [concrete]
// CHECK:STDOUT:   %R.patt: %pattern_type.d98 = symbolic_binding_pattern R, 0 [symbolic]
// CHECK:STDOUT:   %R: %Iterate_where.type.fc8 = symbolic_binding R, 0 [symbolic]
// CHECK:STDOUT:   %R.as_type: type = facet_access_type %R [symbolic]
// CHECK:STDOUT:   %pattern_type.8d8: type = pattern_type %R.as_type [symbolic]
// CHECK:STDOUT:   %r.param_patt.420: %pattern_type.8d8 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %r.patt.a06: %pattern_type.8d8 = wrapper_binding_pattern r, %r.param_patt.420 [symbolic]
// CHECK:STDOUT:   %i.patt: %pattern_type.e09 = value_binding_pattern i [concrete]
// CHECK:STDOUT:   %Iterate.lookup_impl_witness.f25: <witness> = lookup_impl_witness %R, @Iterate [symbolic]
// CHECK:STDOUT:   %Iterate.facet.ad5: %Iterate.type = facet_value %R.as_type, (%Iterate.lookup_impl_witness.f25) [symbolic]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.b16: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.ad5) [symbolic]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.cb8: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.ad5) [symbolic]
// CHECK:STDOUT:   %.b4d: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.b16, %Iterate.facet.ad5 [symbolic]
// CHECK:STDOUT:   %impl.elem2.0ff: %.b4d = impl_witness_access %Iterate.lookup_impl_witness.f25, element2 [symbolic]
// CHECK:STDOUT:   %impl.elem1.1b2: %facet_type.8a3 = impl_witness_access %Iterate.lookup_impl_witness.f25, element1 [symbolic]
// CHECK:STDOUT:   %as_type.b52: type = facet_access_type %impl.elem1.1b2 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.149: <specific function> = specific_impl_function %impl.elem2.0ff, @Iterate.WithSelf.NewCursor(%Iterate.facet.ad5) [symbolic]
// CHECK:STDOUT:   %.897: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.cb8, %Iterate.facet.ad5 [symbolic]
// CHECK:STDOUT:   %impl.elem3.5c2: %.897 = impl_witness_access %Iterate.lookup_impl_witness.f25, element3 [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.ef9: type = class_type @DefaultOptionalStorage, @DefaultOptionalStorage(%facet_value.34c) [concrete]
// CHECK:STDOUT:   %OptionalStorage.impl_witness.f75: <witness> = impl_witness imports.%OptionalStorage.impl_witness_table.968, @T.as_type.as.OptionalStorage.impl(%facet_value.34c) [concrete]
// CHECK:STDOUT:   %OptionalStorage.facet.124: %OptionalStorage.type = facet_value %ValueType, (%OptionalStorage.impl_witness.f75) [concrete]
// CHECK:STDOUT:   %Optional.167: type = class_type @Optional, @Optional(%OptionalStorage.facet.124) [concrete]
// CHECK:STDOUT:   %specific_impl_fn.5cc: <specific function> = specific_impl_function %impl.elem3.5c2, @Iterate.WithSelf.Next(%Iterate.facet.ad5) [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.type.36f: type = fn_type @Optional.HasValue, @Optional(%OptionalStorage.facet.124) [concrete]
// CHECK:STDOUT:   %Optional.HasValue.6de: %Optional.HasValue.type.36f = struct_value () [concrete]
// CHECK:STDOUT:   %Optional.Get.type.f56: type = fn_type @Optional.Get, @Optional(%OptionalStorage.facet.124) [concrete]
// CHECK:STDOUT:   %Optional.Get.1ab: %Optional.Get.type.f56 = struct_value () [concrete]
// CHECK:STDOUT:   %facet_value.329: %facet_type.8a3 = facet_value %ValueType, (%custom_witness.656, %DestroyImpl.impl_witness.261) [concrete]
// CHECK:STDOUT:   %MaybeUnformed.f52: type = class_type @MaybeUnformed, @MaybeUnformed(%facet_value.329) [concrete]
// CHECK:STDOUT:   %.8f3: type = maybe_unformed_type %ValueType [concrete]
// CHECK:STDOUT:   %DefaultOptionalStorage.elem.37d: type = unbound_element_type %DefaultOptionalStorage.ef9, %MaybeUnformed.f52 [concrete]
// CHECK:STDOUT:   %DefaultOptionalStorage.elem.f06: type = unbound_element_type %DefaultOptionalStorage.ef9, bool [concrete]
// CHECK:STDOUT:   %struct_type.value.has_value.5ba: type = struct_type {.value: %MaybeUnformed.f52, .has_value: bool} [concrete]
// CHECK:STDOUT:   %Optional.HasValue.specific_fn.f7d: <specific function> = specific_function %Optional.HasValue.6de, @Optional.HasValue(%OptionalStorage.facet.124) [concrete]
// CHECK:STDOUT:   %Optional.Get.specific_fn.0c5: <specific function> = specific_function %Optional.Get.1ab, @Optional.Get(%OptionalStorage.facet.124) [concrete]
// CHECK:STDOUT:   %operator_PlusEqual__carbon_thunk.type: type = fn_type @operator_PlusEqual__carbon_thunk [concrete]
// CHECK:STDOUT:   %operator_PlusEqual__carbon_thunk: %operator_PlusEqual__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.3: type = fn_type @SubobjectDestroy.WithSelf.Op.loc38_41.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.3: %SubobjectDestroy.WithSelf.Op.type.6870fd.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.3: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.3), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.4: type = fn_type @SubobjectDestroy.WithSelf.Op.loc38_41.3 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.4: %SubobjectDestroy.WithSelf.Op.type.6870fd.4 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.4: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.4), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.a87: %SubobjectDestroy.type = facet_value bool, (%custom_witness.b80ef6.4) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.deb: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.a87) [concrete]
// CHECK:STDOUT:   %facet_value.a1b: %facet_type.8a3 = facet_value bool, (%custom_witness.b80ef6.4, %DestroyImpl.impl_witness.deb) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.65e: %SubobjectDestroy.type = facet_value %MaybeUnformed.f52, (%custom_witness.b80ef6.3) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.743: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.65e) [concrete]
// CHECK:STDOUT:   %facet_value.f4f: %facet_type.8a3 = facet_value %MaybeUnformed.f52, (%custom_witness.b80ef6.3, %DestroyImpl.impl_witness.743) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.7: type = fn_type @SubobjectDestroy.WithSelf.Op.loc38_41.6 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.7: %SubobjectDestroy.WithSelf.Op.type.6870fd.7 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.7: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.7), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.9f4: %SubobjectDestroy.type = facet_value %Optional.167, (%custom_witness.b80ef6.7) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.de7: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.9f4) [concrete]
// CHECK:STDOUT:   %facet_value.0ec: %facet_type.8a3 = facet_value %Optional.167, (%custom_witness.b80ef6.7, %DestroyImpl.impl_witness.de7) [concrete]
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
// CHECK:STDOUT:   %custom_witness.992: <witness> = custom_witness (%Iterator.Op.c90ac2.2), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.b40: %SubobjectDestroy.type = facet_value %Iterator, (%custom_witness.992) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.eb6: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.b40) [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.c95ff6.3: type = fn_type @Iterator.Op.3 [concrete]
// CHECK:STDOUT:   %Iterator.Op.c90ac2.3: %Iterator.Op.type.c95ff6.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.68a: <witness> = custom_witness (%Iterator.Op.c90ac2.3), @Inc [concrete]
// CHECK:STDOUT:   %Equal.type: type = fn_type @Equal [concrete]
// CHECK:STDOUT:   %Equal: %Equal.type = struct_value () [concrete]
// CHECK:STDOUT:   %NotEqual.type: type = fn_type @NotEqual [concrete]
// CHECK:STDOUT:   %NotEqual: %NotEqual.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.2ef: <witness> = custom_witness (%Equal, %NotEqual), @EqWith, @EqWith(%Iterator) [concrete]
// CHECK:STDOUT:   %facet_value.5f5: %CppRange_where.type.d682d6.1 = facet_value %MutableAndConstRange, (%custom_witness.656, %DestroyImpl.impl_witness.261, %custom_witness.1c5, %custom_witness.660, %custom_witness.992, %DestroyImpl.impl_witness.eb6, %custom_witness.d56, %custom_witness.68a, %custom_witness.2ef) [concrete]
// CHECK:STDOUT:   %tuple.type.de6: type = tuple_type (%Iterator, %Iterator) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.8: type = fn_type @SubobjectDestroy.WithSelf.Op.3 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.8: %SubobjectDestroy.WithSelf.Op.type.6870fd.8 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.8: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.8), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.3e6: %SubobjectDestroy.type = facet_value %tuple.type.de6, (%custom_witness.b80ef6.8) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.327: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.3e6) [concrete]
// CHECK:STDOUT:   %complete_type.95d: <witness> = complete_type_witness %tuple.type.de6 [concrete]
// CHECK:STDOUT:   %facet_value.fd1: %facet_type.8a3 = facet_value %tuple.type.de6, (%custom_witness.b80ef6.8, %DestroyImpl.impl_witness.327) [concrete]
// CHECK:STDOUT:   %Iterate.impl_witness.bf9: <witness> = impl_witness imports.%Iterate.impl_witness_table.121, @T.as_type.as.Iterate.impl(%facet_value.5f5) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.type.ed2: type = fn_type @T.as_type.as.Iterate.impl.NewCursor, @T.as_type.as.Iterate.impl(%facet_value.5f5) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.857: %T.as_type.as.Iterate.impl.NewCursor.type.ed2 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.type.049: type = fn_type @T.as_type.as.Iterate.impl.Next, @T.as_type.as.Iterate.impl(%facet_value.5f5) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.0cb: %T.as_type.as.Iterate.impl.Next.type.049 = struct_value () [concrete]
// CHECK:STDOUT:   %Iterate.facet.ef6: %Iterate.type = facet_value %MutableAndConstRange, (%Iterate.impl_witness.bf9) [concrete]
// CHECK:STDOUT:   %facet_value.e5a: %Iterate_where.type.fc8 = facet_value %MutableAndConstRange, (%Iterate.impl_witness.bf9) [concrete]
// CHECK:STDOUT:   %r.param_patt.3a6: %pattern_type.fdb = value_param_pattern [concrete]
// CHECK:STDOUT:   %r.patt.403: %pattern_type.fdb = wrapper_binding_pattern r, %r.param_patt.3a6 [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.76e: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.ef6) [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.a5f: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.ef6) [concrete]
// CHECK:STDOUT:   %.dba: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.76e, %Iterate.facet.ef6 [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.specific_fn: <specific function> = specific_function %T.as_type.as.Iterate.impl.NewCursor.857, @T.as_type.as.Iterate.impl.NewCursor(%facet_value.5f5) [concrete]
// CHECK:STDOUT:   %.f69: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.a5f, %Iterate.facet.ef6 [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.specific_fn: <specific function> = specific_function %T.as_type.as.Iterate.impl.Next.0cb, @T.as_type.as.Iterate.impl.Next(%facet_value.5f5) [concrete]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.bef: <specific function> = specific_function %SelfDestruct, @SelfDestruct(%facet_value.fd1) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .OverloadADL = %OverloadADL
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.4d4: @Optional.%Optional.HasValue.type (%Optional.HasValue.type.cef) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.HasValue (constants.%Optional.HasValue.a74)]
// CHECK:STDOUT:   %Core.import_ref.059: @Optional.%Optional.Get.type (%Optional.Get.type.d97) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.Get (constants.%Optional.Get.3e9)]
// CHECK:STDOUT:   %Core.import_ref.7bb: type = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%DefaultOptionalStorage (constants.%DefaultOptionalStorage.a6b)]
// CHECK:STDOUT:   %Core.import_ref.61e: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.None.type (%T.as_type.as.OptionalStorage.impl.None.type.a9f) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.None (constants.%T.as_type.as.OptionalStorage.impl.None.083)]
// CHECK:STDOUT:   %Core.import_ref.ca1: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Some.type (%T.as_type.as.OptionalStorage.impl.Some.type.61d) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Some (constants.%T.as_type.as.OptionalStorage.impl.Some.ba7)]
// CHECK:STDOUT:   %Core.import_ref.6ba4: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Has.type (%T.as_type.as.OptionalStorage.impl.Has.type.8e5) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Has (constants.%T.as_type.as.OptionalStorage.impl.Has.2d5)]
// CHECK:STDOUT:   %Core.import_ref.73d: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Get.type (%T.as_type.as.OptionalStorage.impl.Get.type.b32) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Get (constants.%T.as_type.as.OptionalStorage.impl.Get.8b0)]
// CHECK:STDOUT:   %Core.import_ref.778 = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], unloaded
// CHECK:STDOUT:   %OptionalStorage.impl_witness_table.968 = impl_witness_table (%Core.import_ref.7bb, %Core.import_ref.61e, %Core.import_ref.ca1, %Core.import_ref.6ba4, %Core.import_ref.73d, %Core.import_ref.778), @T.as_type.as.OptionalStorage.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.146b: @DefaultOptionalStorage.%DefaultOptionalStorage.elem.1 (%DefaultOptionalStorage.elem.c3f) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [concrete = %field_decl.31e]
// CHECK:STDOUT:   %Core.import_ref.f1da: @DefaultOptionalStorage.%DefaultOptionalStorage.elem.2 (%DefaultOptionalStorage.elem.94e) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [concrete = %field_decl.431]
// CHECK:STDOUT:   %Core.import_ref.14b: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.273) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.5b5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.14b), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.d6c: %facet_type.d61 = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.Iterate.impl.%facet_value.1 (constants.%facet_value.9da)]
// CHECK:STDOUT:   %Core.import_ref.2ac: %facet_type.8a3 = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.Iterate.impl.%facet_value.2 (constants.%facet_value.bf6)]
// CHECK:STDOUT:   %Core.import_ref.fd1: @T.as_type.as.Iterate.impl.%T.as_type.as.Iterate.impl.NewCursor.type (%T.as_type.as.Iterate.impl.NewCursor.type.953) = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.Iterate.impl.%T.as_type.as.Iterate.impl.NewCursor (constants.%T.as_type.as.Iterate.impl.NewCursor.544)]
// CHECK:STDOUT:   %Core.import_ref.273: @T.as_type.as.Iterate.impl.%T.as_type.as.Iterate.impl.Next.type (%T.as_type.as.Iterate.impl.Next.type.b36) = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.Iterate.impl.%T.as_type.as.Iterate.impl.Next (constants.%T.as_type.as.Iterate.impl.Next.d6b)]
// CHECK:STDOUT:   %Iterate.impl_witness_table.121 = impl_witness_table (%Core.import_ref.d6c, %Core.import_ref.2ac, %Core.import_ref.fd1, %Core.import_ref.273), @T.as_type.as.Iterate.impl [concrete]
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
// CHECK:STDOUT:   %Core.import_ref.b0d = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %field_decl.431: @DefaultOptionalStorage.%DefaultOptionalStorage.elem.2 (%DefaultOptionalStorage.elem.94e) = field_decl has_value, element1, %Core.import_ref.b0d in [concrete] {}
// CHECK:STDOUT:   %Core.import_ref.164c = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %field_decl.31e: @DefaultOptionalStorage.%DefaultOptionalStorage.elem.1 (%DefaultOptionalStorage.elem.c3f) = field_decl value, element0, %Core.import_ref.164c in [concrete] {}
// CHECK:STDOUT:   %MutableAndConstRange.decl: type = class_decl @MutableAndConstRange [concrete = constants.%MutableAndConstRange] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @TestIterate(%R.loc34_17.2: %Iterate_where.type.fc8) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Iterate.lookup_impl_witness: <witness> = lookup_impl_witness %R.loc34_17.1, @Iterate [symbolic = %Iterate.lookup_impl_witness (constants.%Iterate.lookup_impl_witness.f25)]
// CHECK:STDOUT:   %Iterate.facet.loc38_41.5: %Iterate.type = facet_value %R.as_type.loc34_83.1, (%Iterate.lookup_impl_witness) [symbolic = %Iterate.facet.loc38_41.5 (constants.%Iterate.facet.ad5)]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.loc38_41.5) [symbolic = %Iterate.WithSelf.NewCursor.type (constants.%Iterate.WithSelf.NewCursor.type.b16)]
// CHECK:STDOUT:   %.loc38_41.26: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type, %Iterate.facet.loc38_41.5 [symbolic = %.loc38_41.26 (constants.%.b4d)]
// CHECK:STDOUT:   %impl.elem2.loc38_41.2: @TestIterate.%.loc38_41.26 (%.b4d) = impl_witness_access %Iterate.lookup_impl_witness, element2 [symbolic = %impl.elem2.loc38_41.2 (constants.%impl.elem2.0ff)]
// CHECK:STDOUT:   %specific_impl_fn.loc38_41.3: <specific function> = specific_impl_function %impl.elem2.loc38_41.2, @Iterate.WithSelf.NewCursor(%Iterate.facet.loc38_41.5) [symbolic = %specific_impl_fn.loc38_41.3 (constants.%specific_impl_fn.149)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.loc38_41.5) [symbolic = %Iterate.WithSelf.Next.type (constants.%Iterate.WithSelf.Next.type.cb8)]
// CHECK:STDOUT:   %.loc38_41.27: type = fn_type_with_self_type %Iterate.WithSelf.Next.type, %Iterate.facet.loc38_41.5 [symbolic = %.loc38_41.27 (constants.%.897)]
// CHECK:STDOUT:   %impl.elem3.loc38_41.2: @TestIterate.%.loc38_41.27 (%.897) = impl_witness_access %Iterate.lookup_impl_witness, element3 [symbolic = %impl.elem3.loc38_41.2 (constants.%impl.elem3.5c2)]
// CHECK:STDOUT:   %specific_impl_fn.loc38_41.4: <specific function> = specific_impl_function %impl.elem3.loc38_41.2, @Iterate.WithSelf.Next(%Iterate.facet.loc38_41.5) [symbolic = %specific_impl_fn.loc38_41.4 (constants.%specific_impl_fn.5cc)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%r.param: @TestIterate.%R.as_type.loc34_83.1 (%R.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %i.patt: %pattern_type.e09 = value_binding_pattern i [concrete = constants.%i.patt]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %r.ref: @TestIterate.%R.as_type.loc34_83.1 (%R.as_type) = name_ref r, %r
// CHECK:STDOUT:     %impl.elem2.loc38_41.1: @TestIterate.%.loc38_41.26 (%.b4d) = impl_witness_access constants.%Iterate.lookup_impl_witness.f25, element2 [symbolic = %impl.elem2.loc38_41.2 (constants.%impl.elem2.0ff)]
// CHECK:STDOUT:     %bound_method.loc38_41.1: <bound method> = bound_method %r.ref, %impl.elem2.loc38_41.1
// CHECK:STDOUT:     %Iterate.facet.loc38_41.1: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.f25) [symbolic = %Iterate.facet.loc38_41.5 (constants.%Iterate.facet.ad5)]
// CHECK:STDOUT:     %.loc38_41.1: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc38_41.1 [symbolic = %Iterate.facet.loc38_41.5 (constants.%Iterate.facet.ad5)]
// CHECK:STDOUT:     %Iterate.facet.loc38_41.2: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.f25) [symbolic = %Iterate.facet.loc38_41.5 (constants.%Iterate.facet.ad5)]
// CHECK:STDOUT:     %.loc38_41.2: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc38_41.2 [symbolic = %Iterate.facet.loc38_41.5 (constants.%Iterate.facet.ad5)]
// CHECK:STDOUT:     %specific_impl_fn.loc38_41.1: <specific function> = specific_impl_function %impl.elem2.loc38_41.1, @Iterate.WithSelf.NewCursor(constants.%Iterate.facet.ad5) [symbolic = %specific_impl_fn.loc38_41.3 (constants.%specific_impl_fn.149)]
// CHECK:STDOUT:     %bound_method.loc38_41.2: <bound method> = bound_method %r.ref, %specific_impl_fn.loc38_41.1
// CHECK:STDOUT:     %var: ref @TestIterate.%as_type (%as_type.b52) = var_storage invalid
// CHECK:STDOUT:     %Iterate.WithSelf.NewCursor.call: init @TestIterate.%as_type (%as_type.b52) to %var = call %bound_method.loc38_41.2(%r.ref)
// CHECK:STDOUT:     assign %var, %Iterate.WithSelf.NewCursor.call
// CHECK:STDOUT:     br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.next:
// CHECK:STDOUT:     %impl.elem3.loc38_41.1: @TestIterate.%.loc38_41.27 (%.897) = impl_witness_access constants.%Iterate.lookup_impl_witness.f25, element3 [symbolic = %impl.elem3.loc38_41.2 (constants.%impl.elem3.5c2)]
// CHECK:STDOUT:     %bound_method.loc38_41.3: <bound method> = bound_method %r.ref, %impl.elem3.loc38_41.1
// CHECK:STDOUT:     %Iterate.facet.loc38_41.3: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.f25) [symbolic = %Iterate.facet.loc38_41.5 (constants.%Iterate.facet.ad5)]
// CHECK:STDOUT:     %.loc38_41.3: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc38_41.3 [symbolic = %Iterate.facet.loc38_41.5 (constants.%Iterate.facet.ad5)]
// CHECK:STDOUT:     %Iterate.facet.loc38_41.4: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.f25) [symbolic = %Iterate.facet.loc38_41.5 (constants.%Iterate.facet.ad5)]
// CHECK:STDOUT:     %.loc38_41.4: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc38_41.4 [symbolic = %Iterate.facet.loc38_41.5 (constants.%Iterate.facet.ad5)]
// CHECK:STDOUT:     %.loc38_41.5: %facet_type.8a3 = converted constants.%as_type.b52, constants.%impl.elem1.1b2 [symbolic = %impl.elem1 (constants.%impl.elem1.1b2)]
// CHECK:STDOUT:     %.loc38_41.6: %facet_type.8a3 = converted constants.%as_type.b52, constants.%impl.elem1.1b2 [symbolic = %impl.elem1 (constants.%impl.elem1.1b2)]
// CHECK:STDOUT:     %specific_impl_fn.loc38_41.2: <specific function> = specific_impl_function %impl.elem3.loc38_41.1, @Iterate.WithSelf.Next(constants.%Iterate.facet.ad5) [symbolic = %specific_impl_fn.loc38_41.4 (constants.%specific_impl_fn.5cc)]
// CHECK:STDOUT:     %bound_method.loc38_41.4: <bound method> = bound_method %r.ref, %specific_impl_fn.loc38_41.2
// CHECK:STDOUT:     %.loc38_41.7: ref %Optional.167 = temporary_storage
// CHECK:STDOUT:     %Iterate.WithSelf.Next.call: init %Optional.167 to %.loc38_41.7 = call %bound_method.loc38_41.4(%r.ref, %var)
// CHECK:STDOUT:     %.loc38_41.8: ref %Optional.167 = temporary %.loc38_41.7, %Iterate.WithSelf.Next.call
// CHECK:STDOUT:     %.loc38_41.9: %Optional.HasValue.type.36f = specific_constant imports.%Core.import_ref.4d4, @Optional(constants.%OptionalStorage.facet.124) [concrete = constants.%Optional.HasValue.6de]
// CHECK:STDOUT:     %HasValue.ref: %Optional.HasValue.type.36f = name_ref HasValue, %.loc38_41.9 [concrete = constants.%Optional.HasValue.6de]
// CHECK:STDOUT:     %Optional.HasValue.bound: <bound method> = bound_method %.loc38_41.8, %HasValue.ref
// CHECK:STDOUT:     %Optional.HasValue.specific_fn: <specific function> = specific_function %HasValue.ref, @Optional.HasValue(constants.%OptionalStorage.facet.124) [concrete = constants.%Optional.HasValue.specific_fn.f7d]
// CHECK:STDOUT:     %bound_method.loc38_41.5: <bound method> = bound_method %.loc38_41.8, %Optional.HasValue.specific_fn
// CHECK:STDOUT:     %.loc38_41.10: %Optional.167 = acquire_value %.loc38_41.8
// CHECK:STDOUT:     %Optional.HasValue.call: init bool = call %bound_method.loc38_41.5(%.loc38_41.10)
// CHECK:STDOUT:     %.loc38_41.11: bool = value_of_initializer %Optional.HasValue.call
// CHECK:STDOUT:     %.loc38_41.12: bool = converted %Optional.HasValue.call, %.loc38_41.11
// CHECK:STDOUT:     if %.loc38_41.12 br !for.body else br !for.done
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.body:
// CHECK:STDOUT:     %.loc38_41.13: %Optional.Get.type.f56 = specific_constant imports.%Core.import_ref.059, @Optional(constants.%OptionalStorage.facet.124) [concrete = constants.%Optional.Get.1ab]
// CHECK:STDOUT:     %Get.ref: %Optional.Get.type.f56 = name_ref Get, %.loc38_41.13 [concrete = constants.%Optional.Get.1ab]
// CHECK:STDOUT:     %Optional.Get.bound: <bound method> = bound_method %.loc38_41.8, %Get.ref
// CHECK:STDOUT:     %Optional.Get.specific_fn: <specific function> = specific_function %Get.ref, @Optional.Get(constants.%OptionalStorage.facet.124) [concrete = constants.%Optional.Get.specific_fn.0c5]
// CHECK:STDOUT:     %bound_method.loc38_41.6: <bound method> = bound_method %.loc38_41.8, %Optional.Get.specific_fn
// CHECK:STDOUT:     %.loc38_41.14: ref %ValueType = temporary_storage
// CHECK:STDOUT:     %.loc38_41.15: %Optional.167 = acquire_value %.loc38_41.8
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
// CHECK:STDOUT:     %facet_value.loc38_41.1: %facet_type.8a3 = facet_value constants.%ValueType, (constants.%custom_witness.656, constants.%DestroyImpl.impl_witness.261) [concrete = constants.%facet_value.329]
// CHECK:STDOUT:     %.loc38_41.18: %facet_type.8a3 = converted constants.%ValueType, %facet_value.loc38_41.1 [concrete = constants.%facet_value.329]
// CHECK:STDOUT:     %facet_value.loc38_41.2: %facet_type.8a3 = facet_value constants.%ValueType, (constants.%custom_witness.656, constants.%DestroyImpl.impl_witness.261) [concrete = constants.%facet_value.329]
// CHECK:STDOUT:     %.loc38_41.19: %facet_type.8a3 = converted constants.%ValueType, %facet_value.loc38_41.2 [concrete = constants.%facet_value.329]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %SelfDestruct.call.loc38_41.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc38_41.16)
// CHECK:STDOUT:     %facet_value.loc38_41.3: %facet_type.8a3 = facet_value constants.%Optional.167, (constants.%custom_witness.b80ef6.7, constants.%DestroyImpl.impl_witness.de7) [concrete = constants.%facet_value.0ec]
// CHECK:STDOUT:     %.loc38_41.20: %facet_type.8a3 = converted constants.%Optional.167, %facet_value.loc38_41.3 [concrete = constants.%facet_value.0ec]
// CHECK:STDOUT:     %facet_value.loc38_41.4: %facet_type.8a3 = facet_value constants.%Optional.167, (constants.%custom_witness.b80ef6.7, constants.%DestroyImpl.impl_witness.de7) [concrete = constants.%facet_value.0ec]
// CHECK:STDOUT:     %.loc38_41.21: %facet_type.8a3 = converted constants.%Optional.167, %facet_value.loc38_41.4 [concrete = constants.%facet_value.0ec]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %SelfDestruct.call.loc38_41.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc38_41.8)
// CHECK:STDOUT:     br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.done:
// CHECK:STDOUT:     %facet_value.loc38_41.5: %facet_type.8a3 = facet_value constants.%Optional.167, (constants.%custom_witness.b80ef6.7, constants.%DestroyImpl.impl_witness.de7) [concrete = constants.%facet_value.0ec]
// CHECK:STDOUT:     %.loc38_41.22: %facet_type.8a3 = converted constants.%Optional.167, %facet_value.loc38_41.5 [concrete = constants.%facet_value.0ec]
// CHECK:STDOUT:     %facet_value.loc38_41.6: %facet_type.8a3 = facet_value constants.%Optional.167, (constants.%custom_witness.b80ef6.7, constants.%DestroyImpl.impl_witness.de7) [concrete = constants.%facet_value.0ec]
// CHECK:STDOUT:     %.loc38_41.23: %facet_type.8a3 = converted constants.%Optional.167, %facet_value.loc38_41.6 [concrete = constants.%facet_value.0ec]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %SelfDestruct.call.loc38_41.3: init %empty_tuple.type = call %SelfDestruct.specific_fn.3(%.loc38_41.8)
// CHECK:STDOUT:     %.loc38_41.24: %facet_type.8a3 = converted constants.%as_type.b52, constants.%impl.elem1.1b2 [symbolic = %impl.elem1 (constants.%impl.elem1.1b2)]
// CHECK:STDOUT:     %.loc38_41.25: %facet_type.8a3 = converted constants.%as_type.b52, constants.%impl.elem1.1b2 [symbolic = %impl.elem1 (constants.%impl.elem1.1b2)]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %SelfDestruct.call.loc38_41.4: init %empty_tuple.type = call %SelfDestruct.specific_fn.4(%var)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc38_41.1(%self.param: ref %.8f3) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc38_41.2(%self.param: ref %MaybeUnformed.f52) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc38_41.3(%self.param: ref bool) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc38_41.4(%self.param: ref %struct_type.value.has_value.5ba) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc38_41.1: ref bool = struct_access %self.param, element1
// CHECK:STDOUT:   %facet_value.loc38_41.1: %facet_type.8a3 = facet_value bool, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.deb) [concrete = constants.%facet_value.a1b]
// CHECK:STDOUT:   %.loc38_41.2: %facet_type.8a3 = converted bool, %facet_value.loc38_41.1 [concrete = constants.%facet_value.a1b]
// CHECK:STDOUT:   %facet_value.loc38_41.2: %facet_type.8a3 = facet_value bool, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.deb) [concrete = constants.%facet_value.a1b]
// CHECK:STDOUT:   %.loc38_41.3: %facet_type.8a3 = converted bool, %facet_value.loc38_41.2 [concrete = constants.%facet_value.a1b]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc38_41.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc38_41.1)
// CHECK:STDOUT:   %.loc38_41.4: ref %MaybeUnformed.f52 = struct_access %self.param, element0
// CHECK:STDOUT:   %facet_value.loc38_41.3: %facet_type.8a3 = facet_value constants.%MaybeUnformed.f52, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.743) [concrete = constants.%facet_value.f4f]
// CHECK:STDOUT:   %.loc38_41.5: %facet_type.8a3 = converted constants.%MaybeUnformed.f52, %facet_value.loc38_41.3 [concrete = constants.%facet_value.f4f]
// CHECK:STDOUT:   %facet_value.loc38_41.4: %facet_type.8a3 = facet_value constants.%MaybeUnformed.f52, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.743) [concrete = constants.%facet_value.f4f]
// CHECK:STDOUT:   %.loc38_41.6: %facet_type.8a3 = converted constants.%MaybeUnformed.f52, %facet_value.loc38_41.4 [concrete = constants.%facet_value.f4f]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc38_41.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc38_41.4)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc38_41.5(%self.param: ref %DefaultOptionalStorage.ef9) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %has_value.ref: %DefaultOptionalStorage.elem.f06 = name_ref has_value, imports.%Core.import_ref.f1da [concrete = imports.%field_decl.431]
// CHECK:STDOUT:   %.loc38_41.1: ref bool = class_element_access %self.param, element1
// CHECK:STDOUT:   %facet_value.loc38_41.1: %facet_type.8a3 = facet_value bool, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.deb) [concrete = constants.%facet_value.a1b]
// CHECK:STDOUT:   %.loc38_41.2: %facet_type.8a3 = converted bool, %facet_value.loc38_41.1 [concrete = constants.%facet_value.a1b]
// CHECK:STDOUT:   %facet_value.loc38_41.2: %facet_type.8a3 = facet_value bool, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.deb) [concrete = constants.%facet_value.a1b]
// CHECK:STDOUT:   %.loc38_41.3: %facet_type.8a3 = converted bool, %facet_value.loc38_41.2 [concrete = constants.%facet_value.a1b]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc38_41.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc38_41.1)
// CHECK:STDOUT:   %value.ref: %DefaultOptionalStorage.elem.37d = name_ref value, imports.%Core.import_ref.146b [concrete = imports.%field_decl.31e]
// CHECK:STDOUT:   %.loc38_41.4: ref %MaybeUnformed.f52 = class_element_access %self.param, element0
// CHECK:STDOUT:   %facet_value.loc38_41.3: %facet_type.8a3 = facet_value constants.%MaybeUnformed.f52, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.743) [concrete = constants.%facet_value.f4f]
// CHECK:STDOUT:   %.loc38_41.5: %facet_type.8a3 = converted constants.%MaybeUnformed.f52, %facet_value.loc38_41.3 [concrete = constants.%facet_value.f4f]
// CHECK:STDOUT:   %facet_value.loc38_41.4: %facet_type.8a3 = facet_value constants.%MaybeUnformed.f52, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.743) [concrete = constants.%facet_value.f4f]
// CHECK:STDOUT:   %.loc38_41.6: %facet_type.8a3 = converted constants.%MaybeUnformed.f52, %facet_value.loc38_41.4 [concrete = constants.%facet_value.f4f]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc38_41.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc38_41.4)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc38_41.6(%self.param: ref %Optional.167) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestRangeFor(%mc.param: ref %MutableAndConstRange) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %i.patt: %pattern_type.e09 = value_binding_pattern i [concrete = constants.%i.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %mc.ref: ref %MutableAndConstRange = name_ref mc, %mc
// CHECK:STDOUT:   %impl.elem2: %.dba = impl_witness_access constants.%Iterate.impl_witness.bf9, element2 [concrete = constants.%T.as_type.as.Iterate.impl.NewCursor.857]
// CHECK:STDOUT:   %bound_method.loc80_42.1: <bound method> = bound_method %mc.ref, %impl.elem2
// CHECK:STDOUT:   %facet_value.loc80_42.1: %CppRange_where.type.d682d6.1 = facet_value constants.%MutableAndConstRange, (constants.%custom_witness.656, constants.%DestroyImpl.impl_witness.261, constants.%custom_witness.1c5, constants.%custom_witness.660, constants.%custom_witness.992, constants.%DestroyImpl.impl_witness.eb6, constants.%custom_witness.d56, constants.%custom_witness.68a, constants.%custom_witness.2ef) [concrete = constants.%facet_value.5f5]
// CHECK:STDOUT:   %.loc80_42.1: %CppRange_where.type.d682d6.1 = converted constants.%MutableAndConstRange, %facet_value.loc80_42.1 [concrete = constants.%facet_value.5f5]
// CHECK:STDOUT:   %facet_value.loc80_42.2: %CppRange_where.type.d682d6.1 = facet_value constants.%MutableAndConstRange, (constants.%custom_witness.656, constants.%DestroyImpl.impl_witness.261, constants.%custom_witness.1c5, constants.%custom_witness.660, constants.%custom_witness.992, constants.%DestroyImpl.impl_witness.eb6, constants.%custom_witness.d56, constants.%custom_witness.68a, constants.%custom_witness.2ef) [concrete = constants.%facet_value.5f5]
// CHECK:STDOUT:   %.loc80_42.2: %CppRange_where.type.d682d6.1 = converted constants.%MutableAndConstRange, %facet_value.loc80_42.2 [concrete = constants.%facet_value.5f5]
// CHECK:STDOUT:   %specific_fn.loc80_42.1: <specific function> = specific_function %impl.elem2, @T.as_type.as.Iterate.impl.NewCursor(constants.%facet_value.5f5) [concrete = constants.%T.as_type.as.Iterate.impl.NewCursor.specific_fn]
// CHECK:STDOUT:   %bound_method.loc80_42.2: <bound method> = bound_method %mc.ref, %specific_fn.loc80_42.1
// CHECK:STDOUT:   %var: ref %tuple.type.de6 = var_storage invalid
// CHECK:STDOUT:   %.loc80_40.1: %MutableAndConstRange = acquire_value %mc.ref
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.call: init %tuple.type.de6 to %var = call %bound_method.loc80_42.2(%.loc80_40.1)
// CHECK:STDOUT:   assign %var, %T.as_type.as.Iterate.impl.NewCursor.call
// CHECK:STDOUT:   br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT: !for.next:
// CHECK:STDOUT:   %impl.elem3: %.f69 = impl_witness_access constants.%Iterate.impl_witness.bf9, element3 [concrete = constants.%T.as_type.as.Iterate.impl.Next.0cb]
// CHECK:STDOUT:   %bound_method.loc80_42.3: <bound method> = bound_method %mc.ref, %impl.elem3
// CHECK:STDOUT:   %facet_value.loc80_42.3: %CppRange_where.type.d682d6.1 = facet_value constants.%MutableAndConstRange, (constants.%custom_witness.656, constants.%DestroyImpl.impl_witness.261, constants.%custom_witness.1c5, constants.%custom_witness.660, constants.%custom_witness.992, constants.%DestroyImpl.impl_witness.eb6, constants.%custom_witness.d56, constants.%custom_witness.68a, constants.%custom_witness.2ef) [concrete = constants.%facet_value.5f5]
// CHECK:STDOUT:   %.loc80_42.3: %CppRange_where.type.d682d6.1 = converted constants.%MutableAndConstRange, %facet_value.loc80_42.3 [concrete = constants.%facet_value.5f5]
// CHECK:STDOUT:   %facet_value.loc80_42.4: %CppRange_where.type.d682d6.1 = facet_value constants.%MutableAndConstRange, (constants.%custom_witness.656, constants.%DestroyImpl.impl_witness.261, constants.%custom_witness.1c5, constants.%custom_witness.660, constants.%custom_witness.992, constants.%DestroyImpl.impl_witness.eb6, constants.%custom_witness.d56, constants.%custom_witness.68a, constants.%custom_witness.2ef) [concrete = constants.%facet_value.5f5]
// CHECK:STDOUT:   %.loc80_42.4: %CppRange_where.type.d682d6.1 = converted constants.%MutableAndConstRange, %facet_value.loc80_42.4 [concrete = constants.%facet_value.5f5]
// CHECK:STDOUT:   %specific_fn.loc80_42.2: <specific function> = specific_function %impl.elem3, @T.as_type.as.Iterate.impl.Next(constants.%facet_value.5f5) [concrete = constants.%T.as_type.as.Iterate.impl.Next.specific_fn]
// CHECK:STDOUT:   %bound_method.loc80_42.4: <bound method> = bound_method %mc.ref, %specific_fn.loc80_42.2
// CHECK:STDOUT:   %.loc80_42.5: ref %Optional.167 = temporary_storage
// CHECK:STDOUT:   %.loc80_40.2: %MutableAndConstRange = acquire_value %mc.ref
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.call: init %Optional.167 to %.loc80_42.5 = call %bound_method.loc80_42.4(%.loc80_40.2, %var)
// CHECK:STDOUT:   %.loc80_42.6: ref %Optional.167 = temporary %.loc80_42.5, %T.as_type.as.Iterate.impl.Next.call
// CHECK:STDOUT:   %.loc80_42.7: %Optional.HasValue.type.36f = specific_constant imports.%Core.import_ref.4d4, @Optional(constants.%OptionalStorage.facet.124) [concrete = constants.%Optional.HasValue.6de]
// CHECK:STDOUT:   %HasValue.ref: %Optional.HasValue.type.36f = name_ref HasValue, %.loc80_42.7 [concrete = constants.%Optional.HasValue.6de]
// CHECK:STDOUT:   %Optional.HasValue.bound: <bound method> = bound_method %.loc80_42.6, %HasValue.ref
// CHECK:STDOUT:   %Optional.HasValue.specific_fn: <specific function> = specific_function %HasValue.ref, @Optional.HasValue(constants.%OptionalStorage.facet.124) [concrete = constants.%Optional.HasValue.specific_fn.f7d]
// CHECK:STDOUT:   %bound_method.loc80_42.5: <bound method> = bound_method %.loc80_42.6, %Optional.HasValue.specific_fn
// CHECK:STDOUT:   %.loc80_42.8: %Optional.167 = acquire_value %.loc80_42.6
// CHECK:STDOUT:   %Optional.HasValue.call: init bool = call %bound_method.loc80_42.5(%.loc80_42.8)
// CHECK:STDOUT:   %.loc80_42.9: bool = value_of_initializer %Optional.HasValue.call
// CHECK:STDOUT:   %.loc80_42.10: bool = converted %Optional.HasValue.call, %.loc80_42.9
// CHECK:STDOUT:   if %.loc80_42.10 br !for.body else br !for.done
// CHECK:STDOUT:
// CHECK:STDOUT: !for.body:
// CHECK:STDOUT:   %.loc80_42.11: %Optional.Get.type.f56 = specific_constant imports.%Core.import_ref.059, @Optional(constants.%OptionalStorage.facet.124) [concrete = constants.%Optional.Get.1ab]
// CHECK:STDOUT:   %Get.ref: %Optional.Get.type.f56 = name_ref Get, %.loc80_42.11 [concrete = constants.%Optional.Get.1ab]
// CHECK:STDOUT:   %Optional.Get.bound: <bound method> = bound_method %.loc80_42.6, %Get.ref
// CHECK:STDOUT:   %Optional.Get.specific_fn: <specific function> = specific_function %Get.ref, @Optional.Get(constants.%OptionalStorage.facet.124) [concrete = constants.%Optional.Get.specific_fn.0c5]
// CHECK:STDOUT:   %bound_method.loc80_42.6: <bound method> = bound_method %.loc80_42.6, %Optional.Get.specific_fn
// CHECK:STDOUT:   %.loc80_42.12: ref %ValueType = temporary_storage
// CHECK:STDOUT:   %.loc80_42.13: %Optional.167 = acquire_value %.loc80_42.6
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
// CHECK:STDOUT:   %facet_value.loc80_42.5: %facet_type.8a3 = facet_value constants.%ValueType, (constants.%custom_witness.656, constants.%DestroyImpl.impl_witness.261) [concrete = constants.%facet_value.329]
// CHECK:STDOUT:   %.loc80_42.16: %facet_type.8a3 = converted constants.%ValueType, %facet_value.loc80_42.5 [concrete = constants.%facet_value.329]
// CHECK:STDOUT:   %facet_value.loc80_42.6: %facet_type.8a3 = facet_value constants.%ValueType, (constants.%custom_witness.656, constants.%DestroyImpl.impl_witness.261) [concrete = constants.%facet_value.329]
// CHECK:STDOUT:   %.loc80_42.17: %facet_type.8a3 = converted constants.%ValueType, %facet_value.loc80_42.6 [concrete = constants.%facet_value.329]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc80_42.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc80_42.14)
// CHECK:STDOUT:   %facet_value.loc80_42.7: %facet_type.8a3 = facet_value constants.%Optional.167, (constants.%custom_witness.b80ef6.7, constants.%DestroyImpl.impl_witness.de7) [concrete = constants.%facet_value.0ec]
// CHECK:STDOUT:   %.loc80_42.18: %facet_type.8a3 = converted constants.%Optional.167, %facet_value.loc80_42.7 [concrete = constants.%facet_value.0ec]
// CHECK:STDOUT:   %facet_value.loc80_42.8: %facet_type.8a3 = facet_value constants.%Optional.167, (constants.%custom_witness.b80ef6.7, constants.%DestroyImpl.impl_witness.de7) [concrete = constants.%facet_value.0ec]
// CHECK:STDOUT:   %.loc80_42.19: %facet_type.8a3 = converted constants.%Optional.167, %facet_value.loc80_42.8 [concrete = constants.%facet_value.0ec]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc80_42.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc80_42.6)
// CHECK:STDOUT:   br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT: !for.done:
// CHECK:STDOUT:   %facet_value.loc80_42.9: %facet_type.8a3 = facet_value constants.%Optional.167, (constants.%custom_witness.b80ef6.7, constants.%DestroyImpl.impl_witness.de7) [concrete = constants.%facet_value.0ec]
// CHECK:STDOUT:   %.loc80_42.20: %facet_type.8a3 = converted constants.%Optional.167, %facet_value.loc80_42.9 [concrete = constants.%facet_value.0ec]
// CHECK:STDOUT:   %facet_value.loc80_42.10: %facet_type.8a3 = facet_value constants.%Optional.167, (constants.%custom_witness.b80ef6.7, constants.%DestroyImpl.impl_witness.de7) [concrete = constants.%facet_value.0ec]
// CHECK:STDOUT:   %.loc80_42.21: %facet_type.8a3 = converted constants.%Optional.167, %facet_value.loc80_42.10 [concrete = constants.%facet_value.0ec]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc80_42.3: init %empty_tuple.type = call %SelfDestruct.specific_fn.3(%.loc80_42.6)
// CHECK:STDOUT:   %facet_value.loc80_42.11: %facet_type.8a3 = facet_value constants.%tuple.type.de6, (constants.%custom_witness.b80ef6.8, constants.%DestroyImpl.impl_witness.327) [concrete = constants.%facet_value.fd1]
// CHECK:STDOUT:   %.loc80_42.22: %facet_type.8a3 = converted constants.%tuple.type.de6, %facet_value.loc80_42.11 [concrete = constants.%facet_value.fd1]
// CHECK:STDOUT:   %facet_value.loc80_42.12: %facet_type.8a3 = facet_value constants.%tuple.type.de6, (constants.%custom_witness.b80ef6.8, constants.%DestroyImpl.impl_witness.327) [concrete = constants.%facet_value.fd1]
// CHECK:STDOUT:   %.loc80_42.23: %facet_type.8a3 = converted constants.%tuple.type.de6, %facet_value.loc80_42.12 [concrete = constants.%facet_value.fd1]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc80_42.4: init %empty_tuple.type = call %SelfDestruct.specific_fn.4(%var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestIterate(constants.%R) {
// CHECK:STDOUT:   %R.patt.loc34_17.2 => constants.%R.patt
// CHECK:STDOUT:   %R.loc34_17.1 => constants.%R
// CHECK:STDOUT:   %R.as_type.loc34_83.1 => constants.%R.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.8d8
// CHECK:STDOUT:   %r.param_patt.loc34_81.2 => constants.%r.param_patt.420
// CHECK:STDOUT:   %r.patt.loc34_81.2 => constants.%r.patt.a06
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestIterate(constants.%facet_value.e5a) {
// CHECK:STDOUT:   %R.patt.loc34_17.2 => constants.%R.patt
// CHECK:STDOUT:   %R.loc34_17.1 => constants.%facet_value.e5a
// CHECK:STDOUT:   %R.as_type.loc34_83.1 => constants.%MutableAndConstRange
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.fdb
// CHECK:STDOUT:   %r.param_patt.loc34_81.2 => constants.%r.param_patt.3a6
// CHECK:STDOUT:   %r.patt.loc34_81.2 => constants.%r.patt.403
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc34 => constants.%complete_type.357
// CHECK:STDOUT:   %Iterate.lookup_impl_witness => constants.%Iterate.impl_witness.bf9
// CHECK:STDOUT:   %Iterate.facet.loc38_41.5 => constants.%Iterate.facet.ef6
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type => constants.%Iterate.WithSelf.NewCursor.type.76e
// CHECK:STDOUT:   %.loc38_41.26 => constants.%.dba
// CHECK:STDOUT:   %impl.elem2.loc38_41.2 => constants.%T.as_type.as.Iterate.impl.NewCursor.857
// CHECK:STDOUT:   %specific_impl_fn.loc38_41.3 => constants.%T.as_type.as.Iterate.impl.NewCursor.specific_fn
// CHECK:STDOUT:   %impl.elem1 => constants.%facet_value.fd1
// CHECK:STDOUT:   %as_type => constants.%tuple.type.de6
// CHECK:STDOUT:   %require_complete.1 => constants.%complete_type.95d
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type => constants.%Iterate.WithSelf.Next.type.a5f
// CHECK:STDOUT:   %.loc38_41.27 => constants.%.f69
// CHECK:STDOUT:   %impl.elem3.loc38_41.2 => constants.%T.as_type.as.Iterate.impl.Next.0cb
// CHECK:STDOUT:   %specific_impl_fn.loc38_41.4 => constants.%T.as_type.as.Iterate.impl.Next.specific_fn
// CHECK:STDOUT:   %SelfDestruct.specific_fn.6 => constants.%SelfDestruct.specific_fn.bef
// CHECK:STDOUT: }
// CHECK:STDOUT:
