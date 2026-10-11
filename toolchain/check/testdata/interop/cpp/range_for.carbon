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
// CHECK:STDOUT:   %facet_type.d61: type = facet_type <@SubobjectDestroy & @DestroyImpl & @Copy> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %T.220: %OptionalStorage.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Optional.Get.type.bb9: type = fn_type @Optional.Get, @Optional(%T.220) [symbolic]
// CHECK:STDOUT:   %Optional.Get.fa1: %Optional.Get.type.bb9 = struct_value () [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.type.e9e: type = fn_type @Optional.HasValue, @Optional(%T.220) [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.6cd: %Optional.HasValue.type.e9e = struct_value () [symbolic]
// CHECK:STDOUT:   %facet_type.8a3: type = facet_type <@SubobjectDestroy & @DestroyImpl> [concrete]
// CHECK:STDOUT:   %T.9ab: %facet_type.d61 = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.19c: type = facet_access_type %T.9ab [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.lookup_impl_witness.855: <witness> = lookup_impl_witness %T.9ab, @SubobjectDestroy [symbolic]
// CHECK:STDOUT:   %DestroyImpl.lookup_impl_witness.ff9: <witness> = lookup_impl_witness %T.9ab, @DestroyImpl [symbolic]
// CHECK:STDOUT:   %facet_value.b35: %facet_type.8a3 = facet_value %T.as_type.19c, (%SubobjectDestroy.lookup_impl_witness.855, %DestroyImpl.lookup_impl_witness.ff9) [symbolic]
// CHECK:STDOUT:   %MaybeUnformed.2cf: type = class_type @MaybeUnformed, @MaybeUnformed(%facet_value.b35) [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.7db: type = class_type @DefaultOptionalStorage, @DefaultOptionalStorage(%T.9ab) [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.elem.98d: type = unbound_element_type %DefaultOptionalStorage.7db, bool [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.elem.b5a: type = unbound_element_type %DefaultOptionalStorage.7db, %MaybeUnformed.2cf [symbolic]
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
// CHECK:STDOUT:   %Int.67af24.1: type = class_type @Int, @Int(%N.fe9) [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.6b6: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.1: type = fn_type @SubobjectDestroy.WithSelf.Op.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.1: %SubobjectDestroy.WithSelf.Op.type.6870fd.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.1: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.1), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.type.b77747.1: type = fn_type @DestroyImpl.WithSelf.Op.2 [concrete]
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.25aa57.1: %DestroyImpl.WithSelf.Op.type.b77747.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.2cb555.1: <witness> = custom_witness (%DestroyImpl.WithSelf.Op.25aa57.1), @DestroyImpl [concrete]
// CHECK:STDOUT:   %facet_value.0fe: %facet_type.8a3 = facet_value %i32, (%custom_witness.b80ef6.1, %custom_witness.2cb555.1) [concrete]
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
// CHECK:STDOUT:   %SubobjectDestroy.lookup_impl_witness.dc9: <witness> = lookup_impl_witness %tuple.type.1ba, @SubobjectDestroy [symbolic]
// CHECK:STDOUT:   %DestroyImpl.lookup_impl_witness.504: <witness> = lookup_impl_witness %tuple.type.1ba, @DestroyImpl [symbolic]
// CHECK:STDOUT:   %facet_value.bf6: %facet_type.8a3 = facet_value %tuple.type.1ba, (%SubobjectDestroy.lookup_impl_witness.dc9, %DestroyImpl.lookup_impl_witness.504) [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.lookup_impl_witness.b15da7.2: <witness> = lookup_impl_witness %impl.elem0.493765.2, @SubobjectDestroy [symbolic]
// CHECK:STDOUT:   %DestroyImpl.lookup_impl_witness.4e7f44.2: <witness> = lookup_impl_witness %impl.elem0.493765.2, @DestroyImpl [symbolic]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.167: <witness> = lookup_impl_witness %impl.elem0.493765.2, @Copy [symbolic]
// CHECK:STDOUT:   %facet_value.9da: %facet_type.d61 = facet_value %impl.elem0.493765.2, (%SubobjectDestroy.lookup_impl_witness.b15da7.2, %DestroyImpl.lookup_impl_witness.4e7f44.2, %Copy.lookup_impl_witness.167) [symbolic]
// CHECK:STDOUT:   %SelfDestruct.type: type = fn_type @SelfDestruct [concrete]
// CHECK:STDOUT:   %SelfDestruct: %SelfDestruct.type = struct_value () [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.ac8: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N.fe9) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.5e0: %Int.as.Copy.impl.Op.type.ac8 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.b51: <witness> = impl_witness imports.%Copy.impl_witness_table.8d2, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %facet_value.91d: %facet_type.d61 = facet_value %i32, (%custom_witness.b80ef6.1, %custom_witness.2cb555.1, %Copy.impl_witness.b51) [concrete]
// CHECK:STDOUT:   %Iterate.lookup_impl_witness.1d8: <witness> = lookup_impl_witness %.Self.500, @Iterate [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.8a4: %facet_type.d61 = impl_witness_access %Iterate.lookup_impl_witness.1d8, element0 [symbolic_self]
// CHECK:STDOUT:   %Iterate_where.type.e0e: type = facet_type <@Iterate where %impl.elem0.8a4 = %facet_value.91d> [concrete]
// CHECK:STDOUT:   %pattern_type.e3f: type = pattern_type %Iterate_where.type.e0e [concrete]
// CHECK:STDOUT:   %R.patt: %pattern_type.e3f = symbolic_binding_pattern R, 0 [symbolic]
// CHECK:STDOUT:   %R: %Iterate_where.type.e0e = symbolic_binding R, 0 [symbolic]
// CHECK:STDOUT:   %R.as_type: type = facet_access_type %R [symbolic]
// CHECK:STDOUT:   %pattern_type.42e: type = pattern_type %R.as_type [symbolic]
// CHECK:STDOUT:   %r.param_patt.1e5: %pattern_type.42e = value_param_pattern [symbolic]
// CHECK:STDOUT:   %r.patt.181: %pattern_type.42e = wrapper_binding_pattern r, %r.param_patt.1e5 [symbolic]
// CHECK:STDOUT:   %facet_type.037: type = facet_type <@IntFitsIn, @IntFitsIn(%Int.67af24.2) & @AnyInt> [symbolic]
// CHECK:STDOUT:   %From.6a4: %facet_type.037 = symbolic_binding From, 1 [symbolic]
// CHECK:STDOUT:   %From.as_type.as.ImplicitAs.impl.Convert.type.7fa: type = fn_type @From.as_type.as.ImplicitAs.impl.Convert, @From.as_type.as.ImplicitAs.impl(%To, %From.6a4) [symbolic]
// CHECK:STDOUT:   %From.as_type.as.ImplicitAs.impl.Convert.643: %From.as_type.as.ImplicitAs.impl.Convert.type.7fa = struct_value () [symbolic]
// CHECK:STDOUT:   %i.patt: %pattern_type.6b6 = value_binding_pattern i [concrete]
// CHECK:STDOUT:   %Iterate.lookup_impl_witness.0be: <witness> = lookup_impl_witness %R, @Iterate [symbolic]
// CHECK:STDOUT:   %Iterate.facet.b30: %Iterate.type = facet_value %R.as_type, (%Iterate.lookup_impl_witness.0be) [symbolic]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.489: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.b30) [symbolic]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.53d: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.b30) [symbolic]
// CHECK:STDOUT:   %.f68: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.489, %Iterate.facet.b30 [symbolic]
// CHECK:STDOUT:   %impl.elem2.fa4: %.f68 = impl_witness_access %Iterate.lookup_impl_witness.0be, element2 [symbolic]
// CHECK:STDOUT:   %impl.elem1.b52: %facet_type.8a3 = impl_witness_access %Iterate.lookup_impl_witness.0be, element1 [symbolic]
// CHECK:STDOUT:   %as_type.0eb: type = facet_access_type %impl.elem1.b52 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.725: <specific function> = specific_impl_function %impl.elem2.fa4, @Iterate.WithSelf.NewCursor(%Iterate.facet.b30) [symbolic]
// CHECK:STDOUT:   %.4f5: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.53d, %Iterate.facet.b30 [symbolic]
// CHECK:STDOUT:   %impl.elem3.084: %.4f5 = impl_witness_access %Iterate.lookup_impl_witness.0be, element3 [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.e2c: type = class_type @DefaultOptionalStorage, @DefaultOptionalStorage(%facet_value.91d) [concrete]
// CHECK:STDOUT:   %OptionalStorage.impl_witness.4b6: <witness> = impl_witness imports.%OptionalStorage.impl_witness_table.968, @T.as_type.as.OptionalStorage.impl(%facet_value.91d) [concrete]
// CHECK:STDOUT:   %OptionalStorage.facet.76b: %OptionalStorage.type = facet_value %i32, (%OptionalStorage.impl_witness.4b6) [concrete]
// CHECK:STDOUT:   %Optional.b4f: type = class_type @Optional, @Optional(%OptionalStorage.facet.76b) [concrete]
// CHECK:STDOUT:   %specific_impl_fn.50c: <specific function> = specific_impl_function %impl.elem3.084, @Iterate.WithSelf.Next(%Iterate.facet.b30) [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.type.203: type = fn_type @Optional.HasValue, @Optional(%OptionalStorage.facet.76b) [concrete]
// CHECK:STDOUT:   %Optional.HasValue.c3e: %Optional.HasValue.type.203 = struct_value () [concrete]
// CHECK:STDOUT:   %Optional.Get.type.957: type = fn_type @Optional.Get, @Optional(%OptionalStorage.facet.76b) [concrete]
// CHECK:STDOUT:   %Optional.Get.ae1: %Optional.Get.type.957 = struct_value () [concrete]
// CHECK:STDOUT:   %MaybeUnformed.1d2: type = class_type @MaybeUnformed, @MaybeUnformed(%facet_value.0fe) [concrete]
// CHECK:STDOUT:   %.1d6a7: type = maybe_unformed_type %i32 [concrete]
// CHECK:STDOUT:   %DefaultOptionalStorage.elem.f51: type = unbound_element_type %DefaultOptionalStorage.e2c, %MaybeUnformed.1d2 [concrete]
// CHECK:STDOUT:   %DefaultOptionalStorage.elem.32c: type = unbound_element_type %DefaultOptionalStorage.e2c, bool [concrete]
// CHECK:STDOUT:   %struct_type.value.has_value.11b: type = struct_type {.value: %MaybeUnformed.1d2, .has_value: bool} [concrete]
// CHECK:STDOUT:   %Optional.HasValue.specific_fn.1ea: <specific function> = specific_function %Optional.HasValue.c3e, @Optional.HasValue(%OptionalStorage.facet.76b) [concrete]
// CHECK:STDOUT:   %Optional.Get.specific_fn.c1c: <specific function> = specific_function %Optional.Get.ae1, @Optional.Get(%OptionalStorage.facet.76b) [concrete]
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
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.3: type = fn_type @SubobjectDestroy.WithSelf.Op.loc34_19.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.3: %SubobjectDestroy.WithSelf.Op.type.6870fd.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.3: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.3), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.4: type = fn_type @SubobjectDestroy.WithSelf.Op.loc34_19.3 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.4: %SubobjectDestroy.WithSelf.Op.type.6870fd.4 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.4: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.4), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.type.b77747.2: type = fn_type @DestroyImpl.WithSelf.Op.loc34_19.1 [concrete]
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.25aa57.2: %DestroyImpl.WithSelf.Op.type.b77747.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.2cb555.2: <witness> = custom_witness (%DestroyImpl.WithSelf.Op.25aa57.2), @DestroyImpl [concrete]
// CHECK:STDOUT:   %facet_value.ff1: %facet_type.8a3 = facet_value bool, (%custom_witness.b80ef6.4, %custom_witness.2cb555.2) [concrete]
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.type.b77747.4: type = fn_type @DestroyImpl.WithSelf.Op.loc34_19.3 [concrete]
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.25aa57.4: %DestroyImpl.WithSelf.Op.type.b77747.4 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.2cb555.4: <witness> = custom_witness (%DestroyImpl.WithSelf.Op.25aa57.4), @DestroyImpl [concrete]
// CHECK:STDOUT:   %facet_value.4fe: %facet_type.8a3 = facet_value %MaybeUnformed.1d2, (%custom_witness.b80ef6.3, %custom_witness.2cb555.4) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.7: type = fn_type @SubobjectDestroy.WithSelf.Op.loc34_19.6 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.7: %SubobjectDestroy.WithSelf.Op.type.6870fd.7 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.7: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.7), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.type.b77747.7: type = fn_type @DestroyImpl.WithSelf.Op.loc34_19.6 [concrete]
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.25aa57.7: %DestroyImpl.WithSelf.Op.type.b77747.7 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.2cb555.7: <witness> = custom_witness (%DestroyImpl.WithSelf.Op.25aa57.7), @DestroyImpl [concrete]
// CHECK:STDOUT:   %facet_value.b75: %facet_type.8a3 = facet_value %Optional.b4f, (%custom_witness.b80ef6.7, %custom_witness.2cb555.7) [concrete]
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
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.type.b77747.9: type = fn_type @DestroyImpl.WithSelf.Op.4 [concrete]
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.25aa57.9: %DestroyImpl.WithSelf.Op.type.b77747.9 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.2cb555.8: <witness> = custom_witness (%DestroyImpl.WithSelf.Op.25aa57.9), @DestroyImpl [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.0aed5d.2: type = fn_type @Iterator.Op.2 [concrete]
// CHECK:STDOUT:   %Iterator.Op.a2a866.2: %Iterator.Op.type.0aed5d.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.da7: <witness> = custom_witness (%Iterator.Op.a2a866.2), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.0aed5d.3: type = fn_type @Iterator.Op.3 [concrete]
// CHECK:STDOUT:   %Iterator.Op.a2a866.3: %Iterator.Op.type.0aed5d.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.444: <witness> = custom_witness (%Iterator.Op.a2a866.3), @Inc [concrete]
// CHECK:STDOUT:   %Equal.type.ca8475.1: type = fn_type @Equal.1 [concrete]
// CHECK:STDOUT:   %Equal.285b15.1: %Equal.type.ca8475.1 = struct_value () [concrete]
// CHECK:STDOUT:   %NotEqual.type.53d9ee.1: type = fn_type @NotEqual.1 [concrete]
// CHECK:STDOUT:   %NotEqual.119263.1: %NotEqual.type.53d9ee.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.981: <witness> = custom_witness (%Equal.285b15.1, %NotEqual.119263.1), @EqWith, @EqWith(%Iterator.3ee) [concrete]
// CHECK:STDOUT:   %facet_value.56c: %CppRange_where.type.d682d6.1 = facet_value %MutableRange, (%custom_witness.b80ef6.1, %custom_witness.2cb555.1, %Copy.impl_witness.b51, %custom_witness.ed3, %custom_witness.da7, %custom_witness.2cb555.8, %custom_witness.ef8, %custom_witness.444, %custom_witness.981) [concrete]
// CHECK:STDOUT:   %tuple.type.508: type = tuple_type (%Iterator.3ee, %Iterator.3ee) [concrete]
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.type.b77747.10: type = fn_type @DestroyImpl.WithSelf.Op.5 [concrete]
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.25aa57.10: %DestroyImpl.WithSelf.Op.type.b77747.10 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.2cb555.9: <witness> = custom_witness (%DestroyImpl.WithSelf.Op.25aa57.10), @DestroyImpl [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.8: type = fn_type @SubobjectDestroy.WithSelf.Op.3 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.8: %SubobjectDestroy.WithSelf.Op.type.6870fd.8 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.8: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.8), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %facet_value.ff6: %facet_type.8a3 = facet_value %tuple.type.508, (%custom_witness.b80ef6.8, %custom_witness.2cb555.9) [concrete]
// CHECK:STDOUT:   %Iterate.impl_witness.50b: <witness> = impl_witness imports.%Iterate.impl_witness_table.121, @T.as_type.as.Iterate.impl(%facet_value.56c) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.type.c92: type = fn_type @T.as_type.as.Iterate.impl.NewCursor, @T.as_type.as.Iterate.impl(%facet_value.56c) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.b3a: %T.as_type.as.Iterate.impl.NewCursor.type.c92 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.type.be3: type = fn_type @T.as_type.as.Iterate.impl.Next, @T.as_type.as.Iterate.impl(%facet_value.56c) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.551: %T.as_type.as.Iterate.impl.Next.type.be3 = struct_value () [concrete]
// CHECK:STDOUT:   %Iterate.facet.951: %Iterate.type = facet_value %MutableRange, (%Iterate.impl_witness.50b) [concrete]
// CHECK:STDOUT:   %facet_value.9b6: %Iterate_where.type.e0e = facet_value %MutableRange, (%Iterate.impl_witness.50b) [concrete]
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
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.type.b77747.12: type = fn_type @DestroyImpl.WithSelf.Op.7 [concrete]
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.25aa57.12: %DestroyImpl.WithSelf.Op.type.b77747.12 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.2cb555.10: <witness> = custom_witness (%DestroyImpl.WithSelf.Op.25aa57.12), @DestroyImpl [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.0f34b8.2: type = fn_type @Iterator.Op.5 [concrete]
// CHECK:STDOUT:   %Iterator.Op.1c3e4a.2: %Iterator.Op.type.0f34b8.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.fcc: <witness> = custom_witness (%Iterator.Op.1c3e4a.2), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.0f34b8.3: type = fn_type @Iterator.Op.6 [concrete]
// CHECK:STDOUT:   %Iterator.Op.1c3e4a.3: %Iterator.Op.type.0f34b8.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.746: <witness> = custom_witness (%Iterator.Op.1c3e4a.3), @Inc [concrete]
// CHECK:STDOUT:   %Equal.type.ca8475.2: type = fn_type @Equal.2 [concrete]
// CHECK:STDOUT:   %Equal.285b15.2: %Equal.type.ca8475.2 = struct_value () [concrete]
// CHECK:STDOUT:   %NotEqual.type.53d9ee.2: type = fn_type @NotEqual.2 [concrete]
// CHECK:STDOUT:   %NotEqual.119263.2: %NotEqual.type.53d9ee.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.941: <witness> = custom_witness (%Equal.285b15.2, %NotEqual.119263.2), @EqWith, @EqWith(%Iterator.260) [concrete]
// CHECK:STDOUT:   %facet_value.48d: %CppRange_where.type.d682d6.1 = facet_value %ConstRange, (%custom_witness.b80ef6.1, %custom_witness.2cb555.1, %Copy.impl_witness.b51, %custom_witness.23e, %custom_witness.fcc, %custom_witness.2cb555.10, %custom_witness.adb, %custom_witness.746, %custom_witness.941) [concrete]
// CHECK:STDOUT:   %tuple.type.69f: type = tuple_type (%Iterator.260, %Iterator.260) [concrete]
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.type.b77747.13: type = fn_type @DestroyImpl.WithSelf.Op.8 [concrete]
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.25aa57.13: %DestroyImpl.WithSelf.Op.type.b77747.13 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.2cb555.11: <witness> = custom_witness (%DestroyImpl.WithSelf.Op.25aa57.13), @DestroyImpl [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.9: type = fn_type @SubobjectDestroy.WithSelf.Op.4 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.9: %SubobjectDestroy.WithSelf.Op.type.6870fd.9 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.9: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.9), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %facet_value.57a: %facet_type.8a3 = facet_value %tuple.type.69f, (%custom_witness.b80ef6.9, %custom_witness.2cb555.11) [concrete]
// CHECK:STDOUT:   %Iterate.impl_witness.3f1: <witness> = impl_witness imports.%Iterate.impl_witness_table.121, @T.as_type.as.Iterate.impl(%facet_value.48d) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.type.0cd: type = fn_type @T.as_type.as.Iterate.impl.NewCursor, @T.as_type.as.Iterate.impl(%facet_value.48d) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.50d: %T.as_type.as.Iterate.impl.NewCursor.type.0cd = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.type.607: type = fn_type @T.as_type.as.Iterate.impl.Next, @T.as_type.as.Iterate.impl(%facet_value.48d) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.f15: %T.as_type.as.Iterate.impl.Next.type.607 = struct_value () [concrete]
// CHECK:STDOUT:   %Iterate.facet.aca: %Iterate.type = facet_value %ConstRange, (%Iterate.impl_witness.3f1) [concrete]
// CHECK:STDOUT:   %facet_value.6c5: %Iterate_where.type.e0e = facet_value %ConstRange, (%Iterate.impl_witness.3f1) [concrete]
// CHECK:STDOUT:   %r.param_patt.83e: %pattern_type.0ba = value_param_pattern [concrete]
// CHECK:STDOUT:   %r.patt.c21: %pattern_type.0ba = wrapper_binding_pattern r, %r.param_patt.83e [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.aa4: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.951) [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.659: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.951) [concrete]
// CHECK:STDOUT:   %.689: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.aa4, %Iterate.facet.951 [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.specific_fn.d29: <specific function> = specific_function %T.as_type.as.Iterate.impl.NewCursor.b3a, @T.as_type.as.Iterate.impl.NewCursor(%facet_value.56c) [concrete]
// CHECK:STDOUT:   %.65d: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.659, %Iterate.facet.951 [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.specific_fn.857: <specific function> = specific_function %T.as_type.as.Iterate.impl.Next.551, @T.as_type.as.Iterate.impl.Next(%facet_value.56c) [concrete]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.680: <specific function> = specific_function %SelfDestruct, @SelfDestruct(%facet_value.ff6) [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.970: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.aca) [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.ad8: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.aca) [concrete]
// CHECK:STDOUT:   %.77a: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.970, %Iterate.facet.aca [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.specific_fn.aed: <specific function> = specific_function %T.as_type.as.Iterate.impl.NewCursor.50d, @T.as_type.as.Iterate.impl.NewCursor(%facet_value.48d) [concrete]
// CHECK:STDOUT:   %.776: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.ad8, %Iterate.facet.aca [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.specific_fn.b75: <specific function> = specific_function %T.as_type.as.Iterate.impl.Next.f15, @T.as_type.as.Iterate.impl.Next(%facet_value.48d) [concrete]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.2b3: <specific function> = specific_function %SelfDestruct, @SelfDestruct(%facet_value.57a) [concrete]
// CHECK:STDOUT:   %complete_type.c28: <witness> = complete_type_witness %tuple.type.508 [concrete]
// CHECK:STDOUT:   %complete_type.8d5: <witness> = complete_type_witness %tuple.type.69f [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.bb6: @Optional.%Optional.HasValue.type (%Optional.HasValue.type.e9e) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.HasValue (constants.%Optional.HasValue.6cd)]
// CHECK:STDOUT:   %Core.import_ref.dab: @Optional.%Optional.Get.type (%Optional.Get.type.bb9) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.Get (constants.%Optional.Get.fa1)]
// CHECK:STDOUT:   %Core.import_ref.38a: type = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%DefaultOptionalStorage (constants.%DefaultOptionalStorage.7db)]
// CHECK:STDOUT:   %Core.import_ref.61e: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.None.type (%T.as_type.as.OptionalStorage.impl.None.type.a9f) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.None (constants.%T.as_type.as.OptionalStorage.impl.None.083)]
// CHECK:STDOUT:   %Core.import_ref.ca1: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Some.type (%T.as_type.as.OptionalStorage.impl.Some.type.61d) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Some (constants.%T.as_type.as.OptionalStorage.impl.Some.ba7)]
// CHECK:STDOUT:   %Core.import_ref.6ba4: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Has.type (%T.as_type.as.OptionalStorage.impl.Has.type.8e5) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Has (constants.%T.as_type.as.OptionalStorage.impl.Has.2d5)]
// CHECK:STDOUT:   %Core.import_ref.73d: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Get.type (%T.as_type.as.OptionalStorage.impl.Get.type.b32) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Get (constants.%T.as_type.as.OptionalStorage.impl.Get.8b0)]
// CHECK:STDOUT:   %Core.import_ref.778 = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], unloaded
// CHECK:STDOUT:   %OptionalStorage.impl_witness_table.968 = impl_witness_table (%Core.import_ref.38a, %Core.import_ref.61e, %Core.import_ref.ca1, %Core.import_ref.6ba4, %Core.import_ref.73d, %Core.import_ref.778), @T.as_type.as.OptionalStorage.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.1c5: @DefaultOptionalStorage.%DefaultOptionalStorage.elem.1 (%DefaultOptionalStorage.elem.b5a) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [concrete = %field_decl.649]
// CHECK:STDOUT:   %Core.import_ref.1cb2: @DefaultOptionalStorage.%DefaultOptionalStorage.elem.2 (%DefaultOptionalStorage.elem.98d) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [concrete = %field_decl.255]
// CHECK:STDOUT:   %Core.import_ref.d6c: %facet_type.d61 = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.Iterate.impl.%facet_value.1 (constants.%facet_value.9da)]
// CHECK:STDOUT:   %Core.import_ref.2ac: %facet_type.8a3 = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.Iterate.impl.%facet_value.2 (constants.%facet_value.bf6)]
// CHECK:STDOUT:   %Core.import_ref.fd1: @T.as_type.as.Iterate.impl.%T.as_type.as.Iterate.impl.NewCursor.type (%T.as_type.as.Iterate.impl.NewCursor.type.953) = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.Iterate.impl.%T.as_type.as.Iterate.impl.NewCursor (constants.%T.as_type.as.Iterate.impl.NewCursor.544)]
// CHECK:STDOUT:   %Core.import_ref.273: @T.as_type.as.Iterate.impl.%T.as_type.as.Iterate.impl.Next.type (%T.as_type.as.Iterate.impl.Next.type.b36) = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.Iterate.impl.%T.as_type.as.Iterate.impl.Next (constants.%T.as_type.as.Iterate.impl.Next.d6b)]
// CHECK:STDOUT:   %Iterate.impl_witness_table.121 = impl_witness_table (%Core.import_ref.d6c, %Core.import_ref.2ac, %Core.import_ref.fd1, %Core.import_ref.273), @T.as_type.as.Iterate.impl [concrete]
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
// CHECK:STDOUT:   %Core.import_ref.b0d = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %field_decl.255: @DefaultOptionalStorage.%DefaultOptionalStorage.elem.2 (%DefaultOptionalStorage.elem.98d) = field_decl has_value, element1, %Core.import_ref.b0d in [concrete] {}
// CHECK:STDOUT:   %Core.import_ref.4cb = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %field_decl.649: @DefaultOptionalStorage.%DefaultOptionalStorage.elem.1 (%DefaultOptionalStorage.elem.b5a) = field_decl value, element0, %Core.import_ref.4cb in [concrete] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @TestIterate(%R.loc30_17.2: %Iterate_where.type.e0e) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Iterate.lookup_impl_witness: <witness> = lookup_impl_witness %R.loc30_17.1, @Iterate [symbolic = %Iterate.lookup_impl_witness (constants.%Iterate.lookup_impl_witness.0be)]
// CHECK:STDOUT:   %Iterate.facet.loc34_19.5: %Iterate.type = facet_value %R.as_type.loc30_61.1, (%Iterate.lookup_impl_witness) [symbolic = %Iterate.facet.loc34_19.5 (constants.%Iterate.facet.b30)]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.loc34_19.5) [symbolic = %Iterate.WithSelf.NewCursor.type (constants.%Iterate.WithSelf.NewCursor.type.489)]
// CHECK:STDOUT:   %.loc34_19.23: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type, %Iterate.facet.loc34_19.5 [symbolic = %.loc34_19.23 (constants.%.f68)]
// CHECK:STDOUT:   %impl.elem2.loc34_19.2: @TestIterate.%.loc34_19.23 (%.f68) = impl_witness_access %Iterate.lookup_impl_witness, element2 [symbolic = %impl.elem2.loc34_19.2 (constants.%impl.elem2.fa4)]
// CHECK:STDOUT:   %specific_impl_fn.loc34_19.3: <specific function> = specific_impl_function %impl.elem2.loc34_19.2, @Iterate.WithSelf.NewCursor(%Iterate.facet.loc34_19.5) [symbolic = %specific_impl_fn.loc34_19.3 (constants.%specific_impl_fn.725)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.loc34_19.5) [symbolic = %Iterate.WithSelf.Next.type (constants.%Iterate.WithSelf.Next.type.53d)]
// CHECK:STDOUT:   %.loc34_19.24: type = fn_type_with_self_type %Iterate.WithSelf.Next.type, %Iterate.facet.loc34_19.5 [symbolic = %.loc34_19.24 (constants.%.4f5)]
// CHECK:STDOUT:   %impl.elem3.loc34_19.2: @TestIterate.%.loc34_19.24 (%.4f5) = impl_witness_access %Iterate.lookup_impl_witness, element3 [symbolic = %impl.elem3.loc34_19.2 (constants.%impl.elem3.084)]
// CHECK:STDOUT:   %specific_impl_fn.loc34_19.4: <specific function> = specific_impl_function %impl.elem3.loc34_19.2, @Iterate.WithSelf.Next(%Iterate.facet.loc34_19.5) [symbolic = %specific_impl_fn.loc34_19.4 (constants.%specific_impl_fn.50c)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%r.param: @TestIterate.%R.as_type.loc30_61.1 (%R.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %i.patt: %pattern_type.6b6 = value_binding_pattern i [concrete = constants.%i.patt]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %r.ref: @TestIterate.%R.as_type.loc30_61.1 (%R.as_type) = name_ref r, %r
// CHECK:STDOUT:     %impl.elem2.loc34_19.1: @TestIterate.%.loc34_19.23 (%.f68) = impl_witness_access constants.%Iterate.lookup_impl_witness.0be, element2 [symbolic = %impl.elem2.loc34_19.2 (constants.%impl.elem2.fa4)]
// CHECK:STDOUT:     %bound_method.loc34_19.1: <bound method> = bound_method %r.ref, %impl.elem2.loc34_19.1
// CHECK:STDOUT:     %Iterate.facet.loc34_19.1: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.0be) [symbolic = %Iterate.facet.loc34_19.5 (constants.%Iterate.facet.b30)]
// CHECK:STDOUT:     %.loc34_19.1: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc34_19.1 [symbolic = %Iterate.facet.loc34_19.5 (constants.%Iterate.facet.b30)]
// CHECK:STDOUT:     %Iterate.facet.loc34_19.2: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.0be) [symbolic = %Iterate.facet.loc34_19.5 (constants.%Iterate.facet.b30)]
// CHECK:STDOUT:     %.loc34_19.2: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc34_19.2 [symbolic = %Iterate.facet.loc34_19.5 (constants.%Iterate.facet.b30)]
// CHECK:STDOUT:     %specific_impl_fn.loc34_19.1: <specific function> = specific_impl_function %impl.elem2.loc34_19.1, @Iterate.WithSelf.NewCursor(constants.%Iterate.facet.b30) [symbolic = %specific_impl_fn.loc34_19.3 (constants.%specific_impl_fn.725)]
// CHECK:STDOUT:     %bound_method.loc34_19.2: <bound method> = bound_method %r.ref, %specific_impl_fn.loc34_19.1
// CHECK:STDOUT:     %var: ref @TestIterate.%as_type (%as_type.0eb) = var_storage invalid
// CHECK:STDOUT:     %Iterate.WithSelf.NewCursor.call: init @TestIterate.%as_type (%as_type.0eb) to %var = call %bound_method.loc34_19.2(%r.ref)
// CHECK:STDOUT:     assign %var, %Iterate.WithSelf.NewCursor.call
// CHECK:STDOUT:     br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.next:
// CHECK:STDOUT:     %impl.elem3.loc34_19.1: @TestIterate.%.loc34_19.24 (%.4f5) = impl_witness_access constants.%Iterate.lookup_impl_witness.0be, element3 [symbolic = %impl.elem3.loc34_19.2 (constants.%impl.elem3.084)]
// CHECK:STDOUT:     %bound_method.loc34_19.3: <bound method> = bound_method %r.ref, %impl.elem3.loc34_19.1
// CHECK:STDOUT:     %Iterate.facet.loc34_19.3: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.0be) [symbolic = %Iterate.facet.loc34_19.5 (constants.%Iterate.facet.b30)]
// CHECK:STDOUT:     %.loc34_19.3: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc34_19.3 [symbolic = %Iterate.facet.loc34_19.5 (constants.%Iterate.facet.b30)]
// CHECK:STDOUT:     %Iterate.facet.loc34_19.4: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.0be) [symbolic = %Iterate.facet.loc34_19.5 (constants.%Iterate.facet.b30)]
// CHECK:STDOUT:     %.loc34_19.4: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc34_19.4 [symbolic = %Iterate.facet.loc34_19.5 (constants.%Iterate.facet.b30)]
// CHECK:STDOUT:     %.loc34_19.5: %facet_type.8a3 = converted constants.%as_type.0eb, constants.%impl.elem1.b52 [symbolic = %impl.elem1 (constants.%impl.elem1.b52)]
// CHECK:STDOUT:     %.loc34_19.6: %facet_type.8a3 = converted constants.%as_type.0eb, constants.%impl.elem1.b52 [symbolic = %impl.elem1 (constants.%impl.elem1.b52)]
// CHECK:STDOUT:     %specific_impl_fn.loc34_19.2: <specific function> = specific_impl_function %impl.elem3.loc34_19.1, @Iterate.WithSelf.Next(constants.%Iterate.facet.b30) [symbolic = %specific_impl_fn.loc34_19.4 (constants.%specific_impl_fn.50c)]
// CHECK:STDOUT:     %bound_method.loc34_19.4: <bound method> = bound_method %r.ref, %specific_impl_fn.loc34_19.2
// CHECK:STDOUT:     %.loc34_19.7: ref %Optional.b4f = temporary_storage
// CHECK:STDOUT:     %Iterate.WithSelf.Next.call: init %Optional.b4f to %.loc34_19.7 = call %bound_method.loc34_19.4(%r.ref, %var)
// CHECK:STDOUT:     %.loc34_19.8: ref %Optional.b4f = temporary %.loc34_19.7, %Iterate.WithSelf.Next.call
// CHECK:STDOUT:     %.loc34_19.9: %Optional.HasValue.type.203 = specific_constant imports.%Core.import_ref.bb6, @Optional(constants.%OptionalStorage.facet.76b) [concrete = constants.%Optional.HasValue.c3e]
// CHECK:STDOUT:     %HasValue.ref: %Optional.HasValue.type.203 = name_ref HasValue, %.loc34_19.9 [concrete = constants.%Optional.HasValue.c3e]
// CHECK:STDOUT:     %Optional.HasValue.bound: <bound method> = bound_method %.loc34_19.8, %HasValue.ref
// CHECK:STDOUT:     %Optional.HasValue.specific_fn: <specific function> = specific_function %HasValue.ref, @Optional.HasValue(constants.%OptionalStorage.facet.76b) [concrete = constants.%Optional.HasValue.specific_fn.1ea]
// CHECK:STDOUT:     %bound_method.loc34_19.5: <bound method> = bound_method %.loc34_19.8, %Optional.HasValue.specific_fn
// CHECK:STDOUT:     %.loc34_19.10: %Optional.b4f = acquire_value %.loc34_19.8
// CHECK:STDOUT:     %Optional.HasValue.call: init bool = call %bound_method.loc34_19.5(%.loc34_19.10)
// CHECK:STDOUT:     %.loc34_19.11: bool = value_of_initializer %Optional.HasValue.call
// CHECK:STDOUT:     %.loc34_19.12: bool = converted %Optional.HasValue.call, %.loc34_19.11
// CHECK:STDOUT:     if %.loc34_19.12 br !for.body else br !for.done
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.body:
// CHECK:STDOUT:     %.loc34_19.13: %Optional.Get.type.957 = specific_constant imports.%Core.import_ref.dab, @Optional(constants.%OptionalStorage.facet.76b) [concrete = constants.%Optional.Get.ae1]
// CHECK:STDOUT:     %Get.ref: %Optional.Get.type.957 = name_ref Get, %.loc34_19.13 [concrete = constants.%Optional.Get.ae1]
// CHECK:STDOUT:     %Optional.Get.bound: <bound method> = bound_method %.loc34_19.8, %Get.ref
// CHECK:STDOUT:     %Optional.Get.specific_fn: <specific function> = specific_function %Get.ref, @Optional.Get(constants.%OptionalStorage.facet.76b) [concrete = constants.%Optional.Get.specific_fn.c1c]
// CHECK:STDOUT:     %bound_method.loc34_19.6: <bound method> = bound_method %.loc34_19.8, %Optional.Get.specific_fn
// CHECK:STDOUT:     %.loc34_19.14: %Optional.b4f = acquire_value %.loc34_19.8
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
// CHECK:STDOUT:     %facet_value.loc34_19.1: %facet_type.8a3 = facet_value constants.%Optional.b4f, (constants.%custom_witness.b80ef6.7, constants.%custom_witness.2cb555.7) [concrete = constants.%facet_value.b75]
// CHECK:STDOUT:     %.loc34_19.17: %facet_type.8a3 = converted constants.%Optional.b4f, %facet_value.loc34_19.1 [concrete = constants.%facet_value.b75]
// CHECK:STDOUT:     %facet_value.loc34_19.2: %facet_type.8a3 = facet_value constants.%Optional.b4f, (constants.%custom_witness.b80ef6.7, constants.%custom_witness.2cb555.7) [concrete = constants.%facet_value.b75]
// CHECK:STDOUT:     %.loc34_19.18: %facet_type.8a3 = converted constants.%Optional.b4f, %facet_value.loc34_19.2 [concrete = constants.%facet_value.b75]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %SelfDestruct.call.loc34_19.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc34_19.8)
// CHECK:STDOUT:     br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.done:
// CHECK:STDOUT:     %facet_value.loc34_19.3: %facet_type.8a3 = facet_value constants.%Optional.b4f, (constants.%custom_witness.b80ef6.7, constants.%custom_witness.2cb555.7) [concrete = constants.%facet_value.b75]
// CHECK:STDOUT:     %.loc34_19.19: %facet_type.8a3 = converted constants.%Optional.b4f, %facet_value.loc34_19.3 [concrete = constants.%facet_value.b75]
// CHECK:STDOUT:     %facet_value.loc34_19.4: %facet_type.8a3 = facet_value constants.%Optional.b4f, (constants.%custom_witness.b80ef6.7, constants.%custom_witness.2cb555.7) [concrete = constants.%facet_value.b75]
// CHECK:STDOUT:     %.loc34_19.20: %facet_type.8a3 = converted constants.%Optional.b4f, %facet_value.loc34_19.4 [concrete = constants.%facet_value.b75]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %SelfDestruct.call.loc34_19.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc34_19.8)
// CHECK:STDOUT:     %.loc34_19.21: %facet_type.8a3 = converted constants.%as_type.0eb, constants.%impl.elem1.b52 [symbolic = %impl.elem1 (constants.%impl.elem1.b52)]
// CHECK:STDOUT:     %.loc34_19.22: %facet_type.8a3 = converted constants.%as_type.0eb, constants.%impl.elem1.b52 [symbolic = %impl.elem1 (constants.%impl.elem1.b52)]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %SelfDestruct.call.loc34_19.3: init %empty_tuple.type = call %SelfDestruct.specific_fn.3(%var)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc34_19.1(%self.param: ref %.1d6a7) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc34_19.2(%self.param: ref %MaybeUnformed.1d2) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc34_19.3(%self.param: ref bool) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc34_19.4(%self.param: ref %struct_type.value.has_value.11b) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc34_19.1: ref bool = struct_access %self.param, element1
// CHECK:STDOUT:   %facet_value.loc34_19.1: %facet_type.8a3 = facet_value bool, (constants.%custom_witness.b80ef6.4, constants.%custom_witness.2cb555.2) [concrete = constants.%facet_value.ff1]
// CHECK:STDOUT:   %.loc34_19.2: %facet_type.8a3 = converted bool, %facet_value.loc34_19.1 [concrete = constants.%facet_value.ff1]
// CHECK:STDOUT:   %facet_value.loc34_19.2: %facet_type.8a3 = facet_value bool, (constants.%custom_witness.b80ef6.4, constants.%custom_witness.2cb555.2) [concrete = constants.%facet_value.ff1]
// CHECK:STDOUT:   %.loc34_19.3: %facet_type.8a3 = converted bool, %facet_value.loc34_19.2 [concrete = constants.%facet_value.ff1]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc34_19.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc34_19.1)
// CHECK:STDOUT:   %.loc34_19.4: ref %MaybeUnformed.1d2 = struct_access %self.param, element0
// CHECK:STDOUT:   %facet_value.loc34_19.3: %facet_type.8a3 = facet_value constants.%MaybeUnformed.1d2, (constants.%custom_witness.b80ef6.3, constants.%custom_witness.2cb555.4) [concrete = constants.%facet_value.4fe]
// CHECK:STDOUT:   %.loc34_19.5: %facet_type.8a3 = converted constants.%MaybeUnformed.1d2, %facet_value.loc34_19.3 [concrete = constants.%facet_value.4fe]
// CHECK:STDOUT:   %facet_value.loc34_19.4: %facet_type.8a3 = facet_value constants.%MaybeUnformed.1d2, (constants.%custom_witness.b80ef6.3, constants.%custom_witness.2cb555.4) [concrete = constants.%facet_value.4fe]
// CHECK:STDOUT:   %.loc34_19.6: %facet_type.8a3 = converted constants.%MaybeUnformed.1d2, %facet_value.loc34_19.4 [concrete = constants.%facet_value.4fe]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc34_19.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc34_19.4)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyImpl.WithSelf.Op.loc34_19.1(%self.param: ref bool) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyImpl.WithSelf.Op.loc34_19.2(%self.param: ref %.1d6a7) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyImpl.WithSelf.Op.loc34_19.3(%self.param: ref %MaybeUnformed.1d2) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc34_19.5(%self.param: ref %DefaultOptionalStorage.e2c) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %has_value.ref: %DefaultOptionalStorage.elem.32c = name_ref has_value, imports.%Core.import_ref.1cb2 [concrete = imports.%field_decl.255]
// CHECK:STDOUT:   %.loc34_19.1: ref bool = class_element_access %self.param, element1
// CHECK:STDOUT:   %facet_value.loc34_19.1: %facet_type.8a3 = facet_value bool, (constants.%custom_witness.b80ef6.4, constants.%custom_witness.2cb555.2) [concrete = constants.%facet_value.ff1]
// CHECK:STDOUT:   %.loc34_19.2: %facet_type.8a3 = converted bool, %facet_value.loc34_19.1 [concrete = constants.%facet_value.ff1]
// CHECK:STDOUT:   %facet_value.loc34_19.2: %facet_type.8a3 = facet_value bool, (constants.%custom_witness.b80ef6.4, constants.%custom_witness.2cb555.2) [concrete = constants.%facet_value.ff1]
// CHECK:STDOUT:   %.loc34_19.3: %facet_type.8a3 = converted bool, %facet_value.loc34_19.2 [concrete = constants.%facet_value.ff1]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc34_19.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc34_19.1)
// CHECK:STDOUT:   %value.ref: %DefaultOptionalStorage.elem.f51 = name_ref value, imports.%Core.import_ref.1c5 [concrete = imports.%field_decl.649]
// CHECK:STDOUT:   %.loc34_19.4: ref %MaybeUnformed.1d2 = class_element_access %self.param, element0
// CHECK:STDOUT:   %facet_value.loc34_19.3: %facet_type.8a3 = facet_value constants.%MaybeUnformed.1d2, (constants.%custom_witness.b80ef6.3, constants.%custom_witness.2cb555.4) [concrete = constants.%facet_value.4fe]
// CHECK:STDOUT:   %.loc34_19.5: %facet_type.8a3 = converted constants.%MaybeUnformed.1d2, %facet_value.loc34_19.3 [concrete = constants.%facet_value.4fe]
// CHECK:STDOUT:   %facet_value.loc34_19.4: %facet_type.8a3 = facet_value constants.%MaybeUnformed.1d2, (constants.%custom_witness.b80ef6.3, constants.%custom_witness.2cb555.4) [concrete = constants.%facet_value.4fe]
// CHECK:STDOUT:   %.loc34_19.6: %facet_type.8a3 = converted constants.%MaybeUnformed.1d2, %facet_value.loc34_19.4 [concrete = constants.%facet_value.4fe]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc34_19.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc34_19.4)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc34_19.6(%self.param: ref %Optional.b4f) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyImpl.WithSelf.Op.loc34_19.4(%self.param: ref %struct_type.value.has_value.11b) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyImpl.WithSelf.Op.loc34_19.5(%self.param: ref %DefaultOptionalStorage.e2c) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyImpl.WithSelf.Op.loc34_19.6(%self.param: ref %Optional.b4f) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestRangeFor(%m.param: ref %MutableRange, %c.param: %ConstRange) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %i.patt.loc49: %pattern_type.6b6 = value_binding_pattern i [concrete = constants.%i.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %m.ref: ref %MutableRange = name_ref m, %m
// CHECK:STDOUT:   %impl.elem2.loc49: %.689 = impl_witness_access constants.%Iterate.impl_witness.50b, element2 [concrete = constants.%T.as_type.as.Iterate.impl.NewCursor.b3a]
// CHECK:STDOUT:   %bound_method.loc49_19.1: <bound method> = bound_method %m.ref, %impl.elem2.loc49
// CHECK:STDOUT:   %facet_value.loc49_19.1: %CppRange_where.type.d682d6.1 = facet_value constants.%MutableRange, (constants.%custom_witness.b80ef6.1, constants.%custom_witness.2cb555.1, constants.%Copy.impl_witness.b51, constants.%custom_witness.ed3, constants.%custom_witness.da7, constants.%custom_witness.2cb555.8, constants.%custom_witness.ef8, constants.%custom_witness.444, constants.%custom_witness.981) [concrete = constants.%facet_value.56c]
// CHECK:STDOUT:   %.loc49_19.1: %CppRange_where.type.d682d6.1 = converted constants.%MutableRange, %facet_value.loc49_19.1 [concrete = constants.%facet_value.56c]
// CHECK:STDOUT:   %facet_value.loc49_19.2: %CppRange_where.type.d682d6.1 = facet_value constants.%MutableRange, (constants.%custom_witness.b80ef6.1, constants.%custom_witness.2cb555.1, constants.%Copy.impl_witness.b51, constants.%custom_witness.ed3, constants.%custom_witness.da7, constants.%custom_witness.2cb555.8, constants.%custom_witness.ef8, constants.%custom_witness.444, constants.%custom_witness.981) [concrete = constants.%facet_value.56c]
// CHECK:STDOUT:   %.loc49_19.2: %CppRange_where.type.d682d6.1 = converted constants.%MutableRange, %facet_value.loc49_19.2 [concrete = constants.%facet_value.56c]
// CHECK:STDOUT:   %specific_fn.loc49_19.1: <specific function> = specific_function %impl.elem2.loc49, @T.as_type.as.Iterate.impl.NewCursor(constants.%facet_value.56c) [concrete = constants.%T.as_type.as.Iterate.impl.NewCursor.specific_fn.d29]
// CHECK:STDOUT:   %bound_method.loc49_19.2: <bound method> = bound_method %m.ref, %specific_fn.loc49_19.1
// CHECK:STDOUT:   %var.loc49: ref %tuple.type.508 = var_storage invalid
// CHECK:STDOUT:   %.loc49_18.1: %MutableRange = acquire_value %m.ref
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.call.loc49: init %tuple.type.508 to %var.loc49 = call %bound_method.loc49_19.2(%.loc49_18.1)
// CHECK:STDOUT:   assign %var.loc49, %T.as_type.as.Iterate.impl.NewCursor.call.loc49
// CHECK:STDOUT:   br !for.next.loc49
// CHECK:STDOUT:
// CHECK:STDOUT: !for.next.loc49:
// CHECK:STDOUT:   %impl.elem3.loc49: %.65d = impl_witness_access constants.%Iterate.impl_witness.50b, element3 [concrete = constants.%T.as_type.as.Iterate.impl.Next.551]
// CHECK:STDOUT:   %bound_method.loc49_19.3: <bound method> = bound_method %m.ref, %impl.elem3.loc49
// CHECK:STDOUT:   %facet_value.loc49_19.3: %CppRange_where.type.d682d6.1 = facet_value constants.%MutableRange, (constants.%custom_witness.b80ef6.1, constants.%custom_witness.2cb555.1, constants.%Copy.impl_witness.b51, constants.%custom_witness.ed3, constants.%custom_witness.da7, constants.%custom_witness.2cb555.8, constants.%custom_witness.ef8, constants.%custom_witness.444, constants.%custom_witness.981) [concrete = constants.%facet_value.56c]
// CHECK:STDOUT:   %.loc49_19.3: %CppRange_where.type.d682d6.1 = converted constants.%MutableRange, %facet_value.loc49_19.3 [concrete = constants.%facet_value.56c]
// CHECK:STDOUT:   %facet_value.loc49_19.4: %CppRange_where.type.d682d6.1 = facet_value constants.%MutableRange, (constants.%custom_witness.b80ef6.1, constants.%custom_witness.2cb555.1, constants.%Copy.impl_witness.b51, constants.%custom_witness.ed3, constants.%custom_witness.da7, constants.%custom_witness.2cb555.8, constants.%custom_witness.ef8, constants.%custom_witness.444, constants.%custom_witness.981) [concrete = constants.%facet_value.56c]
// CHECK:STDOUT:   %.loc49_19.4: %CppRange_where.type.d682d6.1 = converted constants.%MutableRange, %facet_value.loc49_19.4 [concrete = constants.%facet_value.56c]
// CHECK:STDOUT:   %specific_fn.loc49_19.2: <specific function> = specific_function %impl.elem3.loc49, @T.as_type.as.Iterate.impl.Next(constants.%facet_value.56c) [concrete = constants.%T.as_type.as.Iterate.impl.Next.specific_fn.857]
// CHECK:STDOUT:   %bound_method.loc49_19.4: <bound method> = bound_method %m.ref, %specific_fn.loc49_19.2
// CHECK:STDOUT:   %.loc49_19.5: ref %Optional.b4f = temporary_storage
// CHECK:STDOUT:   %.loc49_18.2: %MutableRange = acquire_value %m.ref
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.call.loc49: init %Optional.b4f to %.loc49_19.5 = call %bound_method.loc49_19.4(%.loc49_18.2, %var.loc49)
// CHECK:STDOUT:   %.loc49_19.6: ref %Optional.b4f = temporary %.loc49_19.5, %T.as_type.as.Iterate.impl.Next.call.loc49
// CHECK:STDOUT:   %.loc49_19.7: %Optional.HasValue.type.203 = specific_constant imports.%Core.import_ref.bb6, @Optional(constants.%OptionalStorage.facet.76b) [concrete = constants.%Optional.HasValue.c3e]
// CHECK:STDOUT:   %HasValue.ref.loc49: %Optional.HasValue.type.203 = name_ref HasValue, %.loc49_19.7 [concrete = constants.%Optional.HasValue.c3e]
// CHECK:STDOUT:   %Optional.HasValue.bound.loc49: <bound method> = bound_method %.loc49_19.6, %HasValue.ref.loc49
// CHECK:STDOUT:   %Optional.HasValue.specific_fn.loc49: <specific function> = specific_function %HasValue.ref.loc49, @Optional.HasValue(constants.%OptionalStorage.facet.76b) [concrete = constants.%Optional.HasValue.specific_fn.1ea]
// CHECK:STDOUT:   %bound_method.loc49_19.5: <bound method> = bound_method %.loc49_19.6, %Optional.HasValue.specific_fn.loc49
// CHECK:STDOUT:   %.loc49_19.8: %Optional.b4f = acquire_value %.loc49_19.6
// CHECK:STDOUT:   %Optional.HasValue.call.loc49: init bool = call %bound_method.loc49_19.5(%.loc49_19.8)
// CHECK:STDOUT:   %.loc49_19.9: bool = value_of_initializer %Optional.HasValue.call.loc49
// CHECK:STDOUT:   %.loc49_19.10: bool = converted %Optional.HasValue.call.loc49, %.loc49_19.9
// CHECK:STDOUT:   if %.loc49_19.10 br !for.body.loc49 else br !for.done.loc49
// CHECK:STDOUT:
// CHECK:STDOUT: !for.body.loc49:
// CHECK:STDOUT:   %.loc49_19.11: %Optional.Get.type.957 = specific_constant imports.%Core.import_ref.dab, @Optional(constants.%OptionalStorage.facet.76b) [concrete = constants.%Optional.Get.ae1]
// CHECK:STDOUT:   %Get.ref.loc49: %Optional.Get.type.957 = name_ref Get, %.loc49_19.11 [concrete = constants.%Optional.Get.ae1]
// CHECK:STDOUT:   %Optional.Get.bound.loc49: <bound method> = bound_method %.loc49_19.6, %Get.ref.loc49
// CHECK:STDOUT:   %Optional.Get.specific_fn.loc49: <specific function> = specific_function %Get.ref.loc49, @Optional.Get(constants.%OptionalStorage.facet.76b) [concrete = constants.%Optional.Get.specific_fn.c1c]
// CHECK:STDOUT:   %bound_method.loc49_19.6: <bound method> = bound_method %.loc49_19.6, %Optional.Get.specific_fn.loc49
// CHECK:STDOUT:   %.loc49_19.12: %Optional.b4f = acquire_value %.loc49_19.6
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
// CHECK:STDOUT:   %facet_value.loc49_19.5: %facet_type.8a3 = facet_value constants.%Optional.b4f, (constants.%custom_witness.b80ef6.7, constants.%custom_witness.2cb555.7) [concrete = constants.%facet_value.b75]
// CHECK:STDOUT:   %.loc49_19.15: %facet_type.8a3 = converted constants.%Optional.b4f, %facet_value.loc49_19.5 [concrete = constants.%facet_value.b75]
// CHECK:STDOUT:   %facet_value.loc49_19.6: %facet_type.8a3 = facet_value constants.%Optional.b4f, (constants.%custom_witness.b80ef6.7, constants.%custom_witness.2cb555.7) [concrete = constants.%facet_value.b75]
// CHECK:STDOUT:   %.loc49_19.16: %facet_type.8a3 = converted constants.%Optional.b4f, %facet_value.loc49_19.6 [concrete = constants.%facet_value.b75]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc49_19.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc49_19.6)
// CHECK:STDOUT:   br !for.next.loc49
// CHECK:STDOUT:
// CHECK:STDOUT: !for.done.loc49:
// CHECK:STDOUT:   %facet_value.loc49_19.7: %facet_type.8a3 = facet_value constants.%Optional.b4f, (constants.%custom_witness.b80ef6.7, constants.%custom_witness.2cb555.7) [concrete = constants.%facet_value.b75]
// CHECK:STDOUT:   %.loc49_19.17: %facet_type.8a3 = converted constants.%Optional.b4f, %facet_value.loc49_19.7 [concrete = constants.%facet_value.b75]
// CHECK:STDOUT:   %facet_value.loc49_19.8: %facet_type.8a3 = facet_value constants.%Optional.b4f, (constants.%custom_witness.b80ef6.7, constants.%custom_witness.2cb555.7) [concrete = constants.%facet_value.b75]
// CHECK:STDOUT:   %.loc49_19.18: %facet_type.8a3 = converted constants.%Optional.b4f, %facet_value.loc49_19.8 [concrete = constants.%facet_value.b75]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc49_19.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc49_19.6)
// CHECK:STDOUT:   %facet_value.loc49_19.9: %facet_type.8a3 = facet_value constants.%tuple.type.508, (constants.%custom_witness.b80ef6.8, constants.%custom_witness.2cb555.9) [concrete = constants.%facet_value.ff6]
// CHECK:STDOUT:   %.loc49_19.19: %facet_type.8a3 = converted constants.%tuple.type.508, %facet_value.loc49_19.9 [concrete = constants.%facet_value.ff6]
// CHECK:STDOUT:   %facet_value.loc49_19.10: %facet_type.8a3 = facet_value constants.%tuple.type.508, (constants.%custom_witness.b80ef6.8, constants.%custom_witness.2cb555.9) [concrete = constants.%facet_value.ff6]
// CHECK:STDOUT:   %.loc49_19.20: %facet_type.8a3 = converted constants.%tuple.type.508, %facet_value.loc49_19.10 [concrete = constants.%facet_value.ff6]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc49_19.3: init %empty_tuple.type = call %SelfDestruct.specific_fn.3(%var.loc49)
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %i.patt.loc55: %pattern_type.6b6 = value_binding_pattern i [concrete = constants.%i.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c.ref: %ConstRange = name_ref c, %c
// CHECK:STDOUT:   %impl.elem2.loc55: %.77a = impl_witness_access constants.%Iterate.impl_witness.3f1, element2 [concrete = constants.%T.as_type.as.Iterate.impl.NewCursor.50d]
// CHECK:STDOUT:   %bound_method.loc55_19.1: <bound method> = bound_method %c.ref, %impl.elem2.loc55
// CHECK:STDOUT:   %facet_value.loc55_19.1: %CppRange_where.type.d682d6.1 = facet_value constants.%ConstRange, (constants.%custom_witness.b80ef6.1, constants.%custom_witness.2cb555.1, constants.%Copy.impl_witness.b51, constants.%custom_witness.23e, constants.%custom_witness.fcc, constants.%custom_witness.2cb555.10, constants.%custom_witness.adb, constants.%custom_witness.746, constants.%custom_witness.941) [concrete = constants.%facet_value.48d]
// CHECK:STDOUT:   %.loc55_19.1: %CppRange_where.type.d682d6.1 = converted constants.%ConstRange, %facet_value.loc55_19.1 [concrete = constants.%facet_value.48d]
// CHECK:STDOUT:   %facet_value.loc55_19.2: %CppRange_where.type.d682d6.1 = facet_value constants.%ConstRange, (constants.%custom_witness.b80ef6.1, constants.%custom_witness.2cb555.1, constants.%Copy.impl_witness.b51, constants.%custom_witness.23e, constants.%custom_witness.fcc, constants.%custom_witness.2cb555.10, constants.%custom_witness.adb, constants.%custom_witness.746, constants.%custom_witness.941) [concrete = constants.%facet_value.48d]
// CHECK:STDOUT:   %.loc55_19.2: %CppRange_where.type.d682d6.1 = converted constants.%ConstRange, %facet_value.loc55_19.2 [concrete = constants.%facet_value.48d]
// CHECK:STDOUT:   %specific_fn.loc55_19.1: <specific function> = specific_function %impl.elem2.loc55, @T.as_type.as.Iterate.impl.NewCursor(constants.%facet_value.48d) [concrete = constants.%T.as_type.as.Iterate.impl.NewCursor.specific_fn.aed]
// CHECK:STDOUT:   %bound_method.loc55_19.2: <bound method> = bound_method %c.ref, %specific_fn.loc55_19.1
// CHECK:STDOUT:   %var.loc55: ref %tuple.type.69f = var_storage invalid
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.call.loc55: init %tuple.type.69f to %var.loc55 = call %bound_method.loc55_19.2(%c.ref)
// CHECK:STDOUT:   assign %var.loc55, %T.as_type.as.Iterate.impl.NewCursor.call.loc55
// CHECK:STDOUT:   br !for.next.loc55
// CHECK:STDOUT:
// CHECK:STDOUT: !for.next.loc55:
// CHECK:STDOUT:   %impl.elem3.loc55: %.776 = impl_witness_access constants.%Iterate.impl_witness.3f1, element3 [concrete = constants.%T.as_type.as.Iterate.impl.Next.f15]
// CHECK:STDOUT:   %bound_method.loc55_19.3: <bound method> = bound_method %c.ref, %impl.elem3.loc55
// CHECK:STDOUT:   %facet_value.loc55_19.3: %CppRange_where.type.d682d6.1 = facet_value constants.%ConstRange, (constants.%custom_witness.b80ef6.1, constants.%custom_witness.2cb555.1, constants.%Copy.impl_witness.b51, constants.%custom_witness.23e, constants.%custom_witness.fcc, constants.%custom_witness.2cb555.10, constants.%custom_witness.adb, constants.%custom_witness.746, constants.%custom_witness.941) [concrete = constants.%facet_value.48d]
// CHECK:STDOUT:   %.loc55_19.3: %CppRange_where.type.d682d6.1 = converted constants.%ConstRange, %facet_value.loc55_19.3 [concrete = constants.%facet_value.48d]
// CHECK:STDOUT:   %facet_value.loc55_19.4: %CppRange_where.type.d682d6.1 = facet_value constants.%ConstRange, (constants.%custom_witness.b80ef6.1, constants.%custom_witness.2cb555.1, constants.%Copy.impl_witness.b51, constants.%custom_witness.23e, constants.%custom_witness.fcc, constants.%custom_witness.2cb555.10, constants.%custom_witness.adb, constants.%custom_witness.746, constants.%custom_witness.941) [concrete = constants.%facet_value.48d]
// CHECK:STDOUT:   %.loc55_19.4: %CppRange_where.type.d682d6.1 = converted constants.%ConstRange, %facet_value.loc55_19.4 [concrete = constants.%facet_value.48d]
// CHECK:STDOUT:   %specific_fn.loc55_19.2: <specific function> = specific_function %impl.elem3.loc55, @T.as_type.as.Iterate.impl.Next(constants.%facet_value.48d) [concrete = constants.%T.as_type.as.Iterate.impl.Next.specific_fn.b75]
// CHECK:STDOUT:   %bound_method.loc55_19.4: <bound method> = bound_method %c.ref, %specific_fn.loc55_19.2
// CHECK:STDOUT:   %.loc55_19.5: ref %Optional.b4f = temporary_storage
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.call.loc55: init %Optional.b4f to %.loc55_19.5 = call %bound_method.loc55_19.4(%c.ref, %var.loc55)
// CHECK:STDOUT:   %.loc55_19.6: ref %Optional.b4f = temporary %.loc55_19.5, %T.as_type.as.Iterate.impl.Next.call.loc55
// CHECK:STDOUT:   %.loc55_19.7: %Optional.HasValue.type.203 = specific_constant imports.%Core.import_ref.bb6, @Optional(constants.%OptionalStorage.facet.76b) [concrete = constants.%Optional.HasValue.c3e]
// CHECK:STDOUT:   %HasValue.ref.loc55: %Optional.HasValue.type.203 = name_ref HasValue, %.loc55_19.7 [concrete = constants.%Optional.HasValue.c3e]
// CHECK:STDOUT:   %Optional.HasValue.bound.loc55: <bound method> = bound_method %.loc55_19.6, %HasValue.ref.loc55
// CHECK:STDOUT:   %Optional.HasValue.specific_fn.loc55: <specific function> = specific_function %HasValue.ref.loc55, @Optional.HasValue(constants.%OptionalStorage.facet.76b) [concrete = constants.%Optional.HasValue.specific_fn.1ea]
// CHECK:STDOUT:   %bound_method.loc55_19.5: <bound method> = bound_method %.loc55_19.6, %Optional.HasValue.specific_fn.loc55
// CHECK:STDOUT:   %.loc55_19.8: %Optional.b4f = acquire_value %.loc55_19.6
// CHECK:STDOUT:   %Optional.HasValue.call.loc55: init bool = call %bound_method.loc55_19.5(%.loc55_19.8)
// CHECK:STDOUT:   %.loc55_19.9: bool = value_of_initializer %Optional.HasValue.call.loc55
// CHECK:STDOUT:   %.loc55_19.10: bool = converted %Optional.HasValue.call.loc55, %.loc55_19.9
// CHECK:STDOUT:   if %.loc55_19.10 br !for.body.loc55 else br !for.done.loc55
// CHECK:STDOUT:
// CHECK:STDOUT: !for.body.loc55:
// CHECK:STDOUT:   %.loc55_19.11: %Optional.Get.type.957 = specific_constant imports.%Core.import_ref.dab, @Optional(constants.%OptionalStorage.facet.76b) [concrete = constants.%Optional.Get.ae1]
// CHECK:STDOUT:   %Get.ref.loc55: %Optional.Get.type.957 = name_ref Get, %.loc55_19.11 [concrete = constants.%Optional.Get.ae1]
// CHECK:STDOUT:   %Optional.Get.bound.loc55: <bound method> = bound_method %.loc55_19.6, %Get.ref.loc55
// CHECK:STDOUT:   %Optional.Get.specific_fn.loc55: <specific function> = specific_function %Get.ref.loc55, @Optional.Get(constants.%OptionalStorage.facet.76b) [concrete = constants.%Optional.Get.specific_fn.c1c]
// CHECK:STDOUT:   %bound_method.loc55_19.6: <bound method> = bound_method %.loc55_19.6, %Optional.Get.specific_fn.loc55
// CHECK:STDOUT:   %.loc55_19.12: %Optional.b4f = acquire_value %.loc55_19.6
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
// CHECK:STDOUT:   %facet_value.loc55_19.5: %facet_type.8a3 = facet_value constants.%Optional.b4f, (constants.%custom_witness.b80ef6.7, constants.%custom_witness.2cb555.7) [concrete = constants.%facet_value.b75]
// CHECK:STDOUT:   %.loc55_19.15: %facet_type.8a3 = converted constants.%Optional.b4f, %facet_value.loc55_19.5 [concrete = constants.%facet_value.b75]
// CHECK:STDOUT:   %facet_value.loc55_19.6: %facet_type.8a3 = facet_value constants.%Optional.b4f, (constants.%custom_witness.b80ef6.7, constants.%custom_witness.2cb555.7) [concrete = constants.%facet_value.b75]
// CHECK:STDOUT:   %.loc55_19.16: %facet_type.8a3 = converted constants.%Optional.b4f, %facet_value.loc55_19.6 [concrete = constants.%facet_value.b75]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc55_19.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.4(%.loc55_19.6)
// CHECK:STDOUT:   br !for.next.loc55
// CHECK:STDOUT:
// CHECK:STDOUT: !for.done.loc55:
// CHECK:STDOUT:   %facet_value.loc55_19.7: %facet_type.8a3 = facet_value constants.%Optional.b4f, (constants.%custom_witness.b80ef6.7, constants.%custom_witness.2cb555.7) [concrete = constants.%facet_value.b75]
// CHECK:STDOUT:   %.loc55_19.17: %facet_type.8a3 = converted constants.%Optional.b4f, %facet_value.loc55_19.7 [concrete = constants.%facet_value.b75]
// CHECK:STDOUT:   %facet_value.loc55_19.8: %facet_type.8a3 = facet_value constants.%Optional.b4f, (constants.%custom_witness.b80ef6.7, constants.%custom_witness.2cb555.7) [concrete = constants.%facet_value.b75]
// CHECK:STDOUT:   %.loc55_19.18: %facet_type.8a3 = converted constants.%Optional.b4f, %facet_value.loc55_19.8 [concrete = constants.%facet_value.b75]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc55_19.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.5(%.loc55_19.6)
// CHECK:STDOUT:   %facet_value.loc55_19.9: %facet_type.8a3 = facet_value constants.%tuple.type.69f, (constants.%custom_witness.b80ef6.9, constants.%custom_witness.2cb555.11) [concrete = constants.%facet_value.57a]
// CHECK:STDOUT:   %.loc55_19.19: %facet_type.8a3 = converted constants.%tuple.type.69f, %facet_value.loc55_19.9 [concrete = constants.%facet_value.57a]
// CHECK:STDOUT:   %facet_value.loc55_19.10: %facet_type.8a3 = facet_value constants.%tuple.type.69f, (constants.%custom_witness.b80ef6.9, constants.%custom_witness.2cb555.11) [concrete = constants.%facet_value.57a]
// CHECK:STDOUT:   %.loc55_19.20: %facet_type.8a3 = converted constants.%tuple.type.69f, %facet_value.loc55_19.10 [concrete = constants.%facet_value.57a]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc55_19.3: init %empty_tuple.type = call %SelfDestruct.specific_fn.6(%var.loc55)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestIterate(constants.%R) {
// CHECK:STDOUT:   %R.patt.loc30_17.2 => constants.%R.patt
// CHECK:STDOUT:   %R.loc30_17.1 => constants.%R
// CHECK:STDOUT:   %R.as_type.loc30_61.1 => constants.%R.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.42e
// CHECK:STDOUT:   %r.param_patt.loc30_59.2 => constants.%r.param_patt.1e5
// CHECK:STDOUT:   %r.patt.loc30_59.2 => constants.%r.patt.181
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestIterate(constants.%facet_value.9b6) {
// CHECK:STDOUT:   %R.patt.loc30_17.2 => constants.%R.patt
// CHECK:STDOUT:   %R.loc30_17.1 => constants.%facet_value.9b6
// CHECK:STDOUT:   %R.as_type.loc30_61.1 => constants.%MutableRange
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.e635
// CHECK:STDOUT:   %r.param_patt.loc30_59.2 => constants.%r.param_patt.c79
// CHECK:STDOUT:   %r.patt.loc30_59.2 => constants.%r.patt.94f
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc30 => constants.%complete_type.357
// CHECK:STDOUT:   %Iterate.lookup_impl_witness => constants.%Iterate.impl_witness.50b
// CHECK:STDOUT:   %Iterate.facet.loc34_19.5 => constants.%Iterate.facet.951
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type => constants.%Iterate.WithSelf.NewCursor.type.aa4
// CHECK:STDOUT:   %.loc34_19.23 => constants.%.689
// CHECK:STDOUT:   %impl.elem2.loc34_19.2 => constants.%T.as_type.as.Iterate.impl.NewCursor.b3a
// CHECK:STDOUT:   %specific_impl_fn.loc34_19.3 => constants.%T.as_type.as.Iterate.impl.NewCursor.specific_fn.d29
// CHECK:STDOUT:   %impl.elem1 => constants.%facet_value.ff6
// CHECK:STDOUT:   %as_type => constants.%tuple.type.508
// CHECK:STDOUT:   %require_complete.1 => constants.%complete_type.c28
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type => constants.%Iterate.WithSelf.Next.type.659
// CHECK:STDOUT:   %.loc34_19.24 => constants.%.65d
// CHECK:STDOUT:   %impl.elem3.loc34_19.2 => constants.%T.as_type.as.Iterate.impl.Next.551
// CHECK:STDOUT:   %specific_impl_fn.loc34_19.4 => constants.%T.as_type.as.Iterate.impl.Next.specific_fn.857
// CHECK:STDOUT:   %SelfDestruct.specific_fn.5 => constants.%SelfDestruct.specific_fn.680
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestIterate(constants.%facet_value.6c5) {
// CHECK:STDOUT:   %R.patt.loc30_17.2 => constants.%R.patt
// CHECK:STDOUT:   %R.loc30_17.1 => constants.%facet_value.6c5
// CHECK:STDOUT:   %R.as_type.loc30_61.1 => constants.%ConstRange
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.0ba
// CHECK:STDOUT:   %r.param_patt.loc30_59.2 => constants.%r.param_patt.83e
// CHECK:STDOUT:   %r.patt.loc30_59.2 => constants.%r.patt.c21
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc30 => constants.%complete_type.357
// CHECK:STDOUT:   %Iterate.lookup_impl_witness => constants.%Iterate.impl_witness.3f1
// CHECK:STDOUT:   %Iterate.facet.loc34_19.5 => constants.%Iterate.facet.aca
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type => constants.%Iterate.WithSelf.NewCursor.type.970
// CHECK:STDOUT:   %.loc34_19.23 => constants.%.77a
// CHECK:STDOUT:   %impl.elem2.loc34_19.2 => constants.%T.as_type.as.Iterate.impl.NewCursor.50d
// CHECK:STDOUT:   %specific_impl_fn.loc34_19.3 => constants.%T.as_type.as.Iterate.impl.NewCursor.specific_fn.aed
// CHECK:STDOUT:   %impl.elem1 => constants.%facet_value.57a
// CHECK:STDOUT:   %as_type => constants.%tuple.type.69f
// CHECK:STDOUT:   %require_complete.1 => constants.%complete_type.8d5
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type => constants.%Iterate.WithSelf.Next.type.ad8
// CHECK:STDOUT:   %.loc34_19.24 => constants.%.776
// CHECK:STDOUT:   %impl.elem3.loc34_19.2 => constants.%T.as_type.as.Iterate.impl.Next.f15
// CHECK:STDOUT:   %specific_impl_fn.loc34_19.4 => constants.%T.as_type.as.Iterate.impl.Next.specific_fn.b75
// CHECK:STDOUT:   %SelfDestruct.specific_fn.5 => constants.%SelfDestruct.specific_fn.2b3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- methods_return_different_types.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %Iterate.type: type = facet_type <@Iterate> [concrete]
// CHECK:STDOUT:   %OptionalStorage.type: type = facet_type <@OptionalStorage> [concrete]
// CHECK:STDOUT:   %facet_type.d61: type = facet_type <@SubobjectDestroy & @DestroyImpl & @Copy> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %T.220: %OptionalStorage.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Optional.Get.type.bb9: type = fn_type @Optional.Get, @Optional(%T.220) [symbolic]
// CHECK:STDOUT:   %Optional.Get.fa1: %Optional.Get.type.bb9 = struct_value () [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.type.e9e: type = fn_type @Optional.HasValue, @Optional(%T.220) [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.6cd: %Optional.HasValue.type.e9e = struct_value () [symbolic]
// CHECK:STDOUT:   %facet_type.8a3: type = facet_type <@SubobjectDestroy & @DestroyImpl> [concrete]
// CHECK:STDOUT:   %T.9ab: %facet_type.d61 = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.19c: type = facet_access_type %T.9ab [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.lookup_impl_witness.855: <witness> = lookup_impl_witness %T.9ab, @SubobjectDestroy [symbolic]
// CHECK:STDOUT:   %DestroyImpl.lookup_impl_witness.ff9: <witness> = lookup_impl_witness %T.9ab, @DestroyImpl [symbolic]
// CHECK:STDOUT:   %facet_value.b35: %facet_type.8a3 = facet_value %T.as_type.19c, (%SubobjectDestroy.lookup_impl_witness.855, %DestroyImpl.lookup_impl_witness.ff9) [symbolic]
// CHECK:STDOUT:   %MaybeUnformed.2cf: type = class_type @MaybeUnformed, @MaybeUnformed(%facet_value.b35) [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.7db: type = class_type @DefaultOptionalStorage, @DefaultOptionalStorage(%T.9ab) [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.elem.98d: type = unbound_element_type %DefaultOptionalStorage.7db, bool [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.elem.b5a: type = unbound_element_type %DefaultOptionalStorage.7db, %MaybeUnformed.2cf [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Get.type.b32: type = fn_type @T.as_type.as.OptionalStorage.impl.Get, @T.as_type.as.OptionalStorage.impl(%T.9ab) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Get.8b0: %T.as_type.as.OptionalStorage.impl.Get.type.b32 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Has.type.8e5: type = fn_type @T.as_type.as.OptionalStorage.impl.Has, @T.as_type.as.OptionalStorage.impl(%T.9ab) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Has.2d5: %T.as_type.as.OptionalStorage.impl.Has.type.8e5 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Some.type.61d: type = fn_type @T.as_type.as.OptionalStorage.impl.Some, @T.as_type.as.OptionalStorage.impl(%T.9ab) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Some.ba7: %T.as_type.as.OptionalStorage.impl.Some.type.61d = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.None.type.a9f: type = fn_type @T.as_type.as.OptionalStorage.impl.None, @T.as_type.as.OptionalStorage.impl(%T.9ab) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.None.083: %T.as_type.as.OptionalStorage.impl.None.type.a9f = struct_value () [symbolic]
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
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.type.b77747.3: type = fn_type @DestroyImpl.WithSelf.Op.4 [concrete]
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.25aa57.3: %DestroyImpl.WithSelf.Op.type.b77747.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.2cb555.2: <witness> = custom_witness (%DestroyImpl.WithSelf.Op.25aa57.3), @DestroyImpl [concrete]
// CHECK:STDOUT:   %const.879: type = const_type %DiffValueType [concrete]
// CHECK:STDOUT:   %ptr.816: type = ptr_type %const.879 [concrete]
// CHECK:STDOUT:   %ptr.cd0: type = ptr_type %DiffValueType [concrete]
// CHECK:STDOUT:   %DiffValueType.Op.type.83af6e.2: type = fn_type @DiffValueType.Op.2 [concrete]
// CHECK:STDOUT:   %DiffValueType.Op.4dca31.2: %DiffValueType.Op.type.83af6e.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.d0da43.1: <witness> = custom_witness (%DiffValueType.Op.4dca31.2), @Copy [concrete]
// CHECK:STDOUT:   %facet_value.3c8: %facet_type.d61 = facet_value %DiffValueType, (%custom_witness.5760f4.1, %custom_witness.2cb555.2, %custom_witness.d0da43.1) [concrete]
// CHECK:STDOUT:   %Iterate.lookup_impl_witness.1d8: <witness> = lookup_impl_witness %.Self.500, @Iterate [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.8a4: %facet_type.d61 = impl_witness_access %Iterate.lookup_impl_witness.1d8, element0 [symbolic_self]
// CHECK:STDOUT:   %Iterate_where.type.d0c: type = facet_type <@Iterate where %impl.elem0.8a4 = %facet_value.3c8> [concrete]
// CHECK:STDOUT:   %pattern_type.92c: type = pattern_type %Iterate_where.type.d0c [concrete]
// CHECK:STDOUT:   %R.patt: %pattern_type.92c = symbolic_binding_pattern R, 0 [symbolic]
// CHECK:STDOUT:   %R: %Iterate_where.type.d0c = symbolic_binding R, 0 [symbolic]
// CHECK:STDOUT:   %R.as_type: type = facet_access_type %R [symbolic]
// CHECK:STDOUT:   %pattern_type.0a0: type = pattern_type %R.as_type [symbolic]
// CHECK:STDOUT:   %r.patt.029: %pattern_type.0a0 = ref_binding_pattern r [symbolic]
// CHECK:STDOUT:   %r.param_patt.b15: %pattern_type.0a0 = var_param_pattern %r.patt.029 [symbolic]
// CHECK:STDOUT:   %i.patt: %pattern_type.149 = value_binding_pattern i [concrete]
// CHECK:STDOUT:   %Iterate.lookup_impl_witness.47e: <witness> = lookup_impl_witness %R, @Iterate [symbolic]
// CHECK:STDOUT:   %Iterate.facet.dfb: %Iterate.type = facet_value %R.as_type, (%Iterate.lookup_impl_witness.47e) [symbolic]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.bd5: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.dfb) [symbolic]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.81f: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.dfb) [symbolic]
// CHECK:STDOUT:   %.5e9: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.bd5, %Iterate.facet.dfb [symbolic]
// CHECK:STDOUT:   %impl.elem2.621: %.5e9 = impl_witness_access %Iterate.lookup_impl_witness.47e, element2 [symbolic]
// CHECK:STDOUT:   %impl.elem1.f91: %facet_type.8a3 = impl_witness_access %Iterate.lookup_impl_witness.47e, element1 [symbolic]
// CHECK:STDOUT:   %as_type.9e2: type = facet_access_type %impl.elem1.f91 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.f6c: <specific function> = specific_impl_function %impl.elem2.621, @Iterate.WithSelf.NewCursor(%Iterate.facet.dfb) [symbolic]
// CHECK:STDOUT:   %.be1: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.81f, %Iterate.facet.dfb [symbolic]
// CHECK:STDOUT:   %impl.elem3.f98: %.be1 = impl_witness_access %Iterate.lookup_impl_witness.47e, element3 [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.fda: type = class_type @DefaultOptionalStorage, @DefaultOptionalStorage(%facet_value.3c8) [concrete]
// CHECK:STDOUT:   %OptionalStorage.impl_witness.aa2: <witness> = impl_witness imports.%OptionalStorage.impl_witness_table.968, @T.as_type.as.OptionalStorage.impl(%facet_value.3c8) [concrete]
// CHECK:STDOUT:   %OptionalStorage.facet.e5c: %OptionalStorage.type = facet_value %DiffValueType, (%OptionalStorage.impl_witness.aa2) [concrete]
// CHECK:STDOUT:   %Optional.57a: type = class_type @Optional, @Optional(%OptionalStorage.facet.e5c) [concrete]
// CHECK:STDOUT:   %specific_impl_fn.876: <specific function> = specific_impl_function %impl.elem3.f98, @Iterate.WithSelf.Next(%Iterate.facet.dfb) [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.type.85b: type = fn_type @Optional.HasValue, @Optional(%OptionalStorage.facet.e5c) [concrete]
// CHECK:STDOUT:   %Optional.HasValue.d89: %Optional.HasValue.type.85b = struct_value () [concrete]
// CHECK:STDOUT:   %Optional.Get.type.49a: type = fn_type @Optional.Get, @Optional(%OptionalStorage.facet.e5c) [concrete]
// CHECK:STDOUT:   %Optional.Get.c1e: %Optional.Get.type.49a = struct_value () [concrete]
// CHECK:STDOUT:   %facet_value.837: %facet_type.8a3 = facet_value %DiffValueType, (%custom_witness.5760f4.1, %custom_witness.2cb555.2) [concrete]
// CHECK:STDOUT:   %MaybeUnformed.589: type = class_type @MaybeUnformed, @MaybeUnformed(%facet_value.837) [concrete]
// CHECK:STDOUT:   %.a18: type = maybe_unformed_type %DiffValueType [concrete]
// CHECK:STDOUT:   %DefaultOptionalStorage.elem.75c: type = unbound_element_type %DefaultOptionalStorage.fda, %MaybeUnformed.589 [concrete]
// CHECK:STDOUT:   %DefaultOptionalStorage.elem.8a1: type = unbound_element_type %DefaultOptionalStorage.fda, bool [concrete]
// CHECK:STDOUT:   %struct_type.value.has_value.759: type = struct_type {.value: %MaybeUnformed.589, .has_value: bool} [concrete]
// CHECK:STDOUT:   %Optional.HasValue.specific_fn.449: <specific function> = specific_function %Optional.HasValue.d89, @Optional.HasValue(%OptionalStorage.facet.e5c) [concrete]
// CHECK:STDOUT:   %Optional.Get.specific_fn.27e: <specific function> = specific_function %Optional.Get.c1e, @Optional.Get(%OptionalStorage.facet.e5c) [concrete]
// CHECK:STDOUT:   %operator_PlusEqual__carbon_thunk.type: type = fn_type @operator_PlusEqual__carbon_thunk [concrete]
// CHECK:STDOUT:   %operator_PlusEqual__carbon_thunk: %operator_PlusEqual__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.3: type = fn_type @SubobjectDestroy.WithSelf.Op.loc44_33.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.3: %SubobjectDestroy.WithSelf.Op.type.6870fd.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.3: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.3), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.4: type = fn_type @SubobjectDestroy.WithSelf.Op.loc44_33.3 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.4: %SubobjectDestroy.WithSelf.Op.type.6870fd.4 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.4: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.4), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.type.b77747.4: type = fn_type @DestroyImpl.WithSelf.Op.loc44_33.1 [concrete]
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.25aa57.4: %DestroyImpl.WithSelf.Op.type.b77747.4 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.2cb555.3: <witness> = custom_witness (%DestroyImpl.WithSelf.Op.25aa57.4), @DestroyImpl [concrete]
// CHECK:STDOUT:   %facet_value.ff1: %facet_type.8a3 = facet_value bool, (%custom_witness.b80ef6.4, %custom_witness.2cb555.3) [concrete]
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.type.b77747.6: type = fn_type @DestroyImpl.WithSelf.Op.loc44_33.3 [concrete]
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.25aa57.6: %DestroyImpl.WithSelf.Op.type.b77747.6 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.2cb555.5: <witness> = custom_witness (%DestroyImpl.WithSelf.Op.25aa57.6), @DestroyImpl [concrete]
// CHECK:STDOUT:   %facet_value.a4f: %facet_type.8a3 = facet_value %MaybeUnformed.589, (%custom_witness.b80ef6.3, %custom_witness.2cb555.5) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.7: type = fn_type @SubobjectDestroy.WithSelf.Op.loc44_33.6 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.7: %SubobjectDestroy.WithSelf.Op.type.6870fd.7 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.7: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.7), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.type.b77747.9: type = fn_type @DestroyImpl.WithSelf.Op.loc44_33.6 [concrete]
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.25aa57.9: %DestroyImpl.WithSelf.Op.type.b77747.9 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.2cb555.8: <witness> = custom_witness (%DestroyImpl.WithSelf.Op.25aa57.9), @DestroyImpl [concrete]
// CHECK:STDOUT:   %facet_value.971: %facet_type.8a3 = facet_value %Optional.57a, (%custom_witness.b80ef6.7, %custom_witness.2cb555.8) [concrete]
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
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.type.b77747.11: type = fn_type @DestroyImpl.WithSelf.Op.6 [concrete]
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.25aa57.11: %DestroyImpl.WithSelf.Op.type.b77747.11 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.2cb555.9: <witness> = custom_witness (%DestroyImpl.WithSelf.Op.25aa57.11), @DestroyImpl [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.41bbea.2: type = fn_type @Iterator.Op.2 [concrete]
// CHECK:STDOUT:   %Iterator.Op.1caa36.2: %Iterator.Op.type.41bbea.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.3815f0.1: <witness> = custom_witness (%Iterator.Op.1caa36.2), @SubobjectDestroy [concrete]
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
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.type.b77747.13: type = fn_type @DestroyImpl.WithSelf.Op.8 [concrete]
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.25aa57.13: %DestroyImpl.WithSelf.Op.type.b77747.13 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.2cb555.10: <witness> = custom_witness (%DestroyImpl.WithSelf.Op.25aa57.13), @DestroyImpl [concrete]
// CHECK:STDOUT:   %facet_value.618: %CppRange_where.type.d682d6.1 = facet_value %DiffMutableRange, (%custom_witness.5760f4.1, %custom_witness.2cb555.2, %custom_witness.d0da43.1, %custom_witness.aa8, %custom_witness.3815f0.1, %custom_witness.2cb555.9, %custom_witness.392, %custom_witness.1c9, %custom_witness.723, %custom_witness.3815f0.2, %custom_witness.2cb555.10) [concrete]
// CHECK:STDOUT:   %tuple.type.b18: type = tuple_type (%Iterator.f25, %Sentinel.b4c) [concrete]
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.type.b77747.14: type = fn_type @DestroyImpl.WithSelf.Op.9 [concrete]
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.25aa57.14: %DestroyImpl.WithSelf.Op.type.b77747.14 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.2cb555.11: <witness> = custom_witness (%DestroyImpl.WithSelf.Op.25aa57.14), @DestroyImpl [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.8: type = fn_type @SubobjectDestroy.WithSelf.Op.3 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.8: %SubobjectDestroy.WithSelf.Op.type.6870fd.8 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.8: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.8), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %facet_value.e0e: %facet_type.8a3 = facet_value %tuple.type.b18, (%custom_witness.b80ef6.8, %custom_witness.2cb555.11) [concrete]
// CHECK:STDOUT:   %Iterate.impl_witness.feb: <witness> = impl_witness imports.%Iterate.impl_witness_table.121, @T.as_type.as.Iterate.impl(%facet_value.618) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.type.13e: type = fn_type @T.as_type.as.Iterate.impl.NewCursor, @T.as_type.as.Iterate.impl(%facet_value.618) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.737: %T.as_type.as.Iterate.impl.NewCursor.type.13e = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.type.754: type = fn_type @T.as_type.as.Iterate.impl.Next, @T.as_type.as.Iterate.impl(%facet_value.618) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.2f3: %T.as_type.as.Iterate.impl.Next.type.754 = struct_value () [concrete]
// CHECK:STDOUT:   %Iterate.facet.811: %Iterate.type = facet_value %DiffMutableRange, (%Iterate.impl_witness.feb) [concrete]
// CHECK:STDOUT:   %facet_value.7b7: %Iterate_where.type.d0c = facet_value %DiffMutableRange, (%Iterate.impl_witness.feb) [concrete]
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
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.type.b77747.18: type = fn_type @DestroyImpl.WithSelf.Op.13 [concrete]
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.25aa57.18: %DestroyImpl.WithSelf.Op.type.b77747.18 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.2cb555.13: <witness> = custom_witness (%DestroyImpl.WithSelf.Op.25aa57.18), @DestroyImpl [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.b7ae80.2: type = fn_type @Iterator.Op.5 [concrete]
// CHECK:STDOUT:   %Iterator.Op.690fee.2: %Iterator.Op.type.b7ae80.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.3d0529.1: <witness> = custom_witness (%Iterator.Op.690fee.2), @SubobjectDestroy [concrete]
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
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.type.b77747.20: type = fn_type @DestroyImpl.WithSelf.Op.15 [concrete]
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.25aa57.20: %DestroyImpl.WithSelf.Op.type.b77747.20 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.2cb555.14: <witness> = custom_witness (%DestroyImpl.WithSelf.Op.25aa57.20), @DestroyImpl [concrete]
// CHECK:STDOUT:   %facet_value.f7a: %CppRange_where.type.d682d6.1 = facet_value %DiffConstRange, (%custom_witness.5760f4.1, %custom_witness.2cb555.2, %custom_witness.d0da43.1, %custom_witness.7b4, %custom_witness.3d0529.1, %custom_witness.2cb555.13, %custom_witness.0c0, %custom_witness.ffb, %custom_witness.a26, %custom_witness.3d0529.2, %custom_witness.2cb555.14) [concrete]
// CHECK:STDOUT:   %tuple.type.df5: type = tuple_type (%Iterator.a59, %Sentinel.b33) [concrete]
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.type.b77747.21: type = fn_type @DestroyImpl.WithSelf.Op.16 [concrete]
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.25aa57.21: %DestroyImpl.WithSelf.Op.type.b77747.21 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.2cb555.15: <witness> = custom_witness (%DestroyImpl.WithSelf.Op.25aa57.21), @DestroyImpl [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.9: type = fn_type @SubobjectDestroy.WithSelf.Op.4 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.9: %SubobjectDestroy.WithSelf.Op.type.6870fd.9 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.9: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.9), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %facet_value.495: %facet_type.8a3 = facet_value %tuple.type.df5, (%custom_witness.b80ef6.9, %custom_witness.2cb555.15) [concrete]
// CHECK:STDOUT:   %Iterate.impl_witness.3b1: <witness> = impl_witness imports.%Iterate.impl_witness_table.121, @T.as_type.as.Iterate.impl(%facet_value.f7a) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.type.5e4: type = fn_type @T.as_type.as.Iterate.impl.NewCursor, @T.as_type.as.Iterate.impl(%facet_value.f7a) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.9f4: %T.as_type.as.Iterate.impl.NewCursor.type.5e4 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.type.26c: type = fn_type @T.as_type.as.Iterate.impl.Next, @T.as_type.as.Iterate.impl(%facet_value.f7a) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.0c4: %T.as_type.as.Iterate.impl.Next.type.26c = struct_value () [concrete]
// CHECK:STDOUT:   %Iterate.facet.bdf: %Iterate.type = facet_value %DiffConstRange, (%Iterate.impl_witness.3b1) [concrete]
// CHECK:STDOUT:   %facet_value.6e5: %Iterate_where.type.d0c = facet_value %DiffConstRange, (%Iterate.impl_witness.3b1) [concrete]
// CHECK:STDOUT:   %r.patt.9c0: %pattern_type.7dd = ref_binding_pattern r [concrete]
// CHECK:STDOUT:   %r.param_patt.992: %pattern_type.7dd = var_param_pattern %r.patt.9c0 [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.54d: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.811) [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.1fc: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.811) [concrete]
// CHECK:STDOUT:   %.9d5: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.54d, %Iterate.facet.811 [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.specific_fn.7fa: <specific function> = specific_function %T.as_type.as.Iterate.impl.NewCursor.737, @T.as_type.as.Iterate.impl.NewCursor(%facet_value.618) [concrete]
// CHECK:STDOUT:   %.d5f: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.1fc, %Iterate.facet.811 [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.specific_fn.ea2: <specific function> = specific_function %T.as_type.as.Iterate.impl.Next.2f3, @T.as_type.as.Iterate.impl.Next(%facet_value.618) [concrete]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.caa: <specific function> = specific_function %SelfDestruct, @SelfDestruct(%facet_value.e0e) [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.fe4: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.bdf) [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.b60: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.bdf) [concrete]
// CHECK:STDOUT:   %.7bf: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.fe4, %Iterate.facet.bdf [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.specific_fn.ccd: <specific function> = specific_function %T.as_type.as.Iterate.impl.NewCursor.9f4, @T.as_type.as.Iterate.impl.NewCursor(%facet_value.f7a) [concrete]
// CHECK:STDOUT:   %.68d: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.b60, %Iterate.facet.bdf [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.specific_fn.b8b: <specific function> = specific_function %T.as_type.as.Iterate.impl.Next.0c4, @T.as_type.as.Iterate.impl.Next(%facet_value.f7a) [concrete]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.6e2: <specific function> = specific_function %SelfDestruct, @SelfDestruct(%facet_value.495) [concrete]
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
// CHECK:STDOUT:   %Core.import_ref.38a: type = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%DefaultOptionalStorage (constants.%DefaultOptionalStorage.7db)]
// CHECK:STDOUT:   %Core.import_ref.61e: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.None.type (%T.as_type.as.OptionalStorage.impl.None.type.a9f) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.None (constants.%T.as_type.as.OptionalStorage.impl.None.083)]
// CHECK:STDOUT:   %Core.import_ref.ca1: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Some.type (%T.as_type.as.OptionalStorage.impl.Some.type.61d) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Some (constants.%T.as_type.as.OptionalStorage.impl.Some.ba7)]
// CHECK:STDOUT:   %Core.import_ref.6ba4: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Has.type (%T.as_type.as.OptionalStorage.impl.Has.type.8e5) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Has (constants.%T.as_type.as.OptionalStorage.impl.Has.2d5)]
// CHECK:STDOUT:   %Core.import_ref.73d: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Get.type (%T.as_type.as.OptionalStorage.impl.Get.type.b32) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Get (constants.%T.as_type.as.OptionalStorage.impl.Get.8b0)]
// CHECK:STDOUT:   %Core.import_ref.778 = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], unloaded
// CHECK:STDOUT:   %OptionalStorage.impl_witness_table.968 = impl_witness_table (%Core.import_ref.38a, %Core.import_ref.61e, %Core.import_ref.ca1, %Core.import_ref.6ba4, %Core.import_ref.73d, %Core.import_ref.778), @T.as_type.as.OptionalStorage.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.1c5: @DefaultOptionalStorage.%DefaultOptionalStorage.elem.1 (%DefaultOptionalStorage.elem.b5a) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [concrete = %field_decl.649]
// CHECK:STDOUT:   %Core.import_ref.1cb: @DefaultOptionalStorage.%DefaultOptionalStorage.elem.2 (%DefaultOptionalStorage.elem.98d) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [concrete = %field_decl.255]
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
// CHECK:STDOUT:   %field_decl.255: @DefaultOptionalStorage.%DefaultOptionalStorage.elem.2 (%DefaultOptionalStorage.elem.98d) = field_decl has_value, element1, %Core.import_ref.b0d in [concrete] {}
// CHECK:STDOUT:   %Core.import_ref.4cb = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %field_decl.649: @DefaultOptionalStorage.%DefaultOptionalStorage.elem.1 (%DefaultOptionalStorage.elem.b5a) = field_decl value, element0, %Core.import_ref.4cb in [concrete] {}
// CHECK:STDOUT:   %DiffMutableRange.decl: type = class_decl @DiffMutableRange [concrete = constants.%DiffMutableRange] {} {}
// CHECK:STDOUT:   %DiffConstRange.decl: type = class_decl @DiffConstRange [concrete = constants.%DiffConstRange] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @TestIterate(%R.loc40_17.2: %Iterate_where.type.d0c) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Iterate.lookup_impl_witness: <witness> = lookup_impl_witness %R.loc40_17.1, @Iterate [symbolic = %Iterate.lookup_impl_witness (constants.%Iterate.lookup_impl_witness.47e)]
// CHECK:STDOUT:   %Iterate.facet.loc44_33.5: %Iterate.type = facet_value %R.as_type.loc40_79.1, (%Iterate.lookup_impl_witness) [symbolic = %Iterate.facet.loc44_33.5 (constants.%Iterate.facet.dfb)]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.loc44_33.5) [symbolic = %Iterate.WithSelf.NewCursor.type (constants.%Iterate.WithSelf.NewCursor.type.bd5)]
// CHECK:STDOUT:   %.loc44_33.26: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type, %Iterate.facet.loc44_33.5 [symbolic = %.loc44_33.26 (constants.%.5e9)]
// CHECK:STDOUT:   %impl.elem2.loc44_33.2: @TestIterate.%.loc44_33.26 (%.5e9) = impl_witness_access %Iterate.lookup_impl_witness, element2 [symbolic = %impl.elem2.loc44_33.2 (constants.%impl.elem2.621)]
// CHECK:STDOUT:   %specific_impl_fn.loc44_33.3: <specific function> = specific_impl_function %impl.elem2.loc44_33.2, @Iterate.WithSelf.NewCursor(%Iterate.facet.loc44_33.5) [symbolic = %specific_impl_fn.loc44_33.3 (constants.%specific_impl_fn.f6c)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.loc44_33.5) [symbolic = %Iterate.WithSelf.Next.type (constants.%Iterate.WithSelf.Next.type.81f)]
// CHECK:STDOUT:   %.loc44_33.27: type = fn_type_with_self_type %Iterate.WithSelf.Next.type, %Iterate.facet.loc44_33.5 [symbolic = %.loc44_33.27 (constants.%.be1)]
// CHECK:STDOUT:   %impl.elem3.loc44_33.2: @TestIterate.%.loc44_33.27 (%.be1) = impl_witness_access %Iterate.lookup_impl_witness, element3 [symbolic = %impl.elem3.loc44_33.2 (constants.%impl.elem3.f98)]
// CHECK:STDOUT:   %specific_impl_fn.loc44_33.4: <specific function> = specific_impl_function %impl.elem3.loc44_33.2, @Iterate.WithSelf.Next(%Iterate.facet.loc44_33.5) [symbolic = %specific_impl_fn.loc44_33.4 (constants.%specific_impl_fn.876)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%r.param: ref @TestIterate.%R.as_type.loc40_79.1 (%R.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %i.patt: %pattern_type.149 = value_binding_pattern i [concrete = constants.%i.patt]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %r.ref: ref @TestIterate.%R.as_type.loc40_79.1 (%R.as_type) = name_ref r, %r
// CHECK:STDOUT:     %impl.elem2.loc44_33.1: @TestIterate.%.loc44_33.26 (%.5e9) = impl_witness_access constants.%Iterate.lookup_impl_witness.47e, element2 [symbolic = %impl.elem2.loc44_33.2 (constants.%impl.elem2.621)]
// CHECK:STDOUT:     %bound_method.loc44_33.1: <bound method> = bound_method %r.ref, %impl.elem2.loc44_33.1
// CHECK:STDOUT:     %Iterate.facet.loc44_33.1: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.47e) [symbolic = %Iterate.facet.loc44_33.5 (constants.%Iterate.facet.dfb)]
// CHECK:STDOUT:     %.loc44_33.1: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc44_33.1 [symbolic = %Iterate.facet.loc44_33.5 (constants.%Iterate.facet.dfb)]
// CHECK:STDOUT:     %Iterate.facet.loc44_33.2: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.47e) [symbolic = %Iterate.facet.loc44_33.5 (constants.%Iterate.facet.dfb)]
// CHECK:STDOUT:     %.loc44_33.2: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc44_33.2 [symbolic = %Iterate.facet.loc44_33.5 (constants.%Iterate.facet.dfb)]
// CHECK:STDOUT:     %specific_impl_fn.loc44_33.1: <specific function> = specific_impl_function %impl.elem2.loc44_33.1, @Iterate.WithSelf.NewCursor(constants.%Iterate.facet.dfb) [symbolic = %specific_impl_fn.loc44_33.3 (constants.%specific_impl_fn.f6c)]
// CHECK:STDOUT:     %bound_method.loc44_33.2: <bound method> = bound_method %r.ref, %specific_impl_fn.loc44_33.1
// CHECK:STDOUT:     %var: ref @TestIterate.%as_type (%as_type.9e2) = var_storage invalid
// CHECK:STDOUT:     %.loc44_32.1: @TestIterate.%R.as_type.loc40_79.1 (%R.as_type) = acquire_value %r.ref
// CHECK:STDOUT:     %Iterate.WithSelf.NewCursor.call: init @TestIterate.%as_type (%as_type.9e2) to %var = call %bound_method.loc44_33.2(%.loc44_32.1)
// CHECK:STDOUT:     assign %var, %Iterate.WithSelf.NewCursor.call
// CHECK:STDOUT:     br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.next:
// CHECK:STDOUT:     %impl.elem3.loc44_33.1: @TestIterate.%.loc44_33.27 (%.be1) = impl_witness_access constants.%Iterate.lookup_impl_witness.47e, element3 [symbolic = %impl.elem3.loc44_33.2 (constants.%impl.elem3.f98)]
// CHECK:STDOUT:     %bound_method.loc44_33.3: <bound method> = bound_method %r.ref, %impl.elem3.loc44_33.1
// CHECK:STDOUT:     %Iterate.facet.loc44_33.3: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.47e) [symbolic = %Iterate.facet.loc44_33.5 (constants.%Iterate.facet.dfb)]
// CHECK:STDOUT:     %.loc44_33.3: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc44_33.3 [symbolic = %Iterate.facet.loc44_33.5 (constants.%Iterate.facet.dfb)]
// CHECK:STDOUT:     %Iterate.facet.loc44_33.4: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.47e) [symbolic = %Iterate.facet.loc44_33.5 (constants.%Iterate.facet.dfb)]
// CHECK:STDOUT:     %.loc44_33.4: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc44_33.4 [symbolic = %Iterate.facet.loc44_33.5 (constants.%Iterate.facet.dfb)]
// CHECK:STDOUT:     %.loc44_33.5: %facet_type.8a3 = converted constants.%as_type.9e2, constants.%impl.elem1.f91 [symbolic = %impl.elem1 (constants.%impl.elem1.f91)]
// CHECK:STDOUT:     %.loc44_33.6: %facet_type.8a3 = converted constants.%as_type.9e2, constants.%impl.elem1.f91 [symbolic = %impl.elem1 (constants.%impl.elem1.f91)]
// CHECK:STDOUT:     %specific_impl_fn.loc44_33.2: <specific function> = specific_impl_function %impl.elem3.loc44_33.1, @Iterate.WithSelf.Next(constants.%Iterate.facet.dfb) [symbolic = %specific_impl_fn.loc44_33.4 (constants.%specific_impl_fn.876)]
// CHECK:STDOUT:     %bound_method.loc44_33.4: <bound method> = bound_method %r.ref, %specific_impl_fn.loc44_33.2
// CHECK:STDOUT:     %.loc44_33.7: ref %Optional.57a = temporary_storage
// CHECK:STDOUT:     %.loc44_32.2: @TestIterate.%R.as_type.loc40_79.1 (%R.as_type) = acquire_value %r.ref
// CHECK:STDOUT:     %Iterate.WithSelf.Next.call: init %Optional.57a to %.loc44_33.7 = call %bound_method.loc44_33.4(%.loc44_32.2, %var)
// CHECK:STDOUT:     %.loc44_33.8: ref %Optional.57a = temporary %.loc44_33.7, %Iterate.WithSelf.Next.call
// CHECK:STDOUT:     %.loc44_33.9: %Optional.HasValue.type.85b = specific_constant imports.%Core.import_ref.bb6, @Optional(constants.%OptionalStorage.facet.e5c) [concrete = constants.%Optional.HasValue.d89]
// CHECK:STDOUT:     %HasValue.ref: %Optional.HasValue.type.85b = name_ref HasValue, %.loc44_33.9 [concrete = constants.%Optional.HasValue.d89]
// CHECK:STDOUT:     %Optional.HasValue.bound: <bound method> = bound_method %.loc44_33.8, %HasValue.ref
// CHECK:STDOUT:     %Optional.HasValue.specific_fn: <specific function> = specific_function %HasValue.ref, @Optional.HasValue(constants.%OptionalStorage.facet.e5c) [concrete = constants.%Optional.HasValue.specific_fn.449]
// CHECK:STDOUT:     %bound_method.loc44_33.5: <bound method> = bound_method %.loc44_33.8, %Optional.HasValue.specific_fn
// CHECK:STDOUT:     %.loc44_33.10: %Optional.57a = acquire_value %.loc44_33.8
// CHECK:STDOUT:     %Optional.HasValue.call: init bool = call %bound_method.loc44_33.5(%.loc44_33.10)
// CHECK:STDOUT:     %.loc44_33.11: bool = value_of_initializer %Optional.HasValue.call
// CHECK:STDOUT:     %.loc44_33.12: bool = converted %Optional.HasValue.call, %.loc44_33.11
// CHECK:STDOUT:     if %.loc44_33.12 br !for.body else br !for.done
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.body:
// CHECK:STDOUT:     %.loc44_33.13: %Optional.Get.type.49a = specific_constant imports.%Core.import_ref.dab, @Optional(constants.%OptionalStorage.facet.e5c) [concrete = constants.%Optional.Get.c1e]
// CHECK:STDOUT:     %Get.ref: %Optional.Get.type.49a = name_ref Get, %.loc44_33.13 [concrete = constants.%Optional.Get.c1e]
// CHECK:STDOUT:     %Optional.Get.bound: <bound method> = bound_method %.loc44_33.8, %Get.ref
// CHECK:STDOUT:     %Optional.Get.specific_fn: <specific function> = specific_function %Get.ref, @Optional.Get(constants.%OptionalStorage.facet.e5c) [concrete = constants.%Optional.Get.specific_fn.27e]
// CHECK:STDOUT:     %bound_method.loc44_33.6: <bound method> = bound_method %.loc44_33.8, %Optional.Get.specific_fn
// CHECK:STDOUT:     %.loc44_33.14: ref %DiffValueType = temporary_storage
// CHECK:STDOUT:     %.loc44_33.15: %Optional.57a = acquire_value %.loc44_33.8
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
// CHECK:STDOUT:     %facet_value.loc44_33.1: %facet_type.8a3 = facet_value constants.%DiffValueType, (constants.%custom_witness.5760f4.1, constants.%custom_witness.2cb555.2) [concrete = constants.%facet_value.837]
// CHECK:STDOUT:     %.loc44_33.18: %facet_type.8a3 = converted constants.%DiffValueType, %facet_value.loc44_33.1 [concrete = constants.%facet_value.837]
// CHECK:STDOUT:     %facet_value.loc44_33.2: %facet_type.8a3 = facet_value constants.%DiffValueType, (constants.%custom_witness.5760f4.1, constants.%custom_witness.2cb555.2) [concrete = constants.%facet_value.837]
// CHECK:STDOUT:     %.loc44_33.19: %facet_type.8a3 = converted constants.%DiffValueType, %facet_value.loc44_33.2 [concrete = constants.%facet_value.837]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %SelfDestruct.call.loc44_33.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc44_33.16)
// CHECK:STDOUT:     %facet_value.loc44_33.3: %facet_type.8a3 = facet_value constants.%Optional.57a, (constants.%custom_witness.b80ef6.7, constants.%custom_witness.2cb555.8) [concrete = constants.%facet_value.971]
// CHECK:STDOUT:     %.loc44_33.20: %facet_type.8a3 = converted constants.%Optional.57a, %facet_value.loc44_33.3 [concrete = constants.%facet_value.971]
// CHECK:STDOUT:     %facet_value.loc44_33.4: %facet_type.8a3 = facet_value constants.%Optional.57a, (constants.%custom_witness.b80ef6.7, constants.%custom_witness.2cb555.8) [concrete = constants.%facet_value.971]
// CHECK:STDOUT:     %.loc44_33.21: %facet_type.8a3 = converted constants.%Optional.57a, %facet_value.loc44_33.4 [concrete = constants.%facet_value.971]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %SelfDestruct.call.loc44_33.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc44_33.8)
// CHECK:STDOUT:     br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.done:
// CHECK:STDOUT:     %facet_value.loc44_33.5: %facet_type.8a3 = facet_value constants.%Optional.57a, (constants.%custom_witness.b80ef6.7, constants.%custom_witness.2cb555.8) [concrete = constants.%facet_value.971]
// CHECK:STDOUT:     %.loc44_33.22: %facet_type.8a3 = converted constants.%Optional.57a, %facet_value.loc44_33.5 [concrete = constants.%facet_value.971]
// CHECK:STDOUT:     %facet_value.loc44_33.6: %facet_type.8a3 = facet_value constants.%Optional.57a, (constants.%custom_witness.b80ef6.7, constants.%custom_witness.2cb555.8) [concrete = constants.%facet_value.971]
// CHECK:STDOUT:     %.loc44_33.23: %facet_type.8a3 = converted constants.%Optional.57a, %facet_value.loc44_33.6 [concrete = constants.%facet_value.971]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %SelfDestruct.call.loc44_33.3: init %empty_tuple.type = call %SelfDestruct.specific_fn.3(%.loc44_33.8)
// CHECK:STDOUT:     %.loc44_33.24: %facet_type.8a3 = converted constants.%as_type.9e2, constants.%impl.elem1.f91 [symbolic = %impl.elem1 (constants.%impl.elem1.f91)]
// CHECK:STDOUT:     %.loc44_33.25: %facet_type.8a3 = converted constants.%as_type.9e2, constants.%impl.elem1.f91 [symbolic = %impl.elem1 (constants.%impl.elem1.f91)]
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
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc44_33.2(%self.param: ref %MaybeUnformed.589) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc44_33.3(%self.param: ref bool) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc44_33.4(%self.param: ref %struct_type.value.has_value.759) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc44_33.1: ref bool = struct_access %self.param, element1
// CHECK:STDOUT:   %facet_value.loc44_33.1: %facet_type.8a3 = facet_value bool, (constants.%custom_witness.b80ef6.4, constants.%custom_witness.2cb555.3) [concrete = constants.%facet_value.ff1]
// CHECK:STDOUT:   %.loc44_33.2: %facet_type.8a3 = converted bool, %facet_value.loc44_33.1 [concrete = constants.%facet_value.ff1]
// CHECK:STDOUT:   %facet_value.loc44_33.2: %facet_type.8a3 = facet_value bool, (constants.%custom_witness.b80ef6.4, constants.%custom_witness.2cb555.3) [concrete = constants.%facet_value.ff1]
// CHECK:STDOUT:   %.loc44_33.3: %facet_type.8a3 = converted bool, %facet_value.loc44_33.2 [concrete = constants.%facet_value.ff1]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc44_33.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc44_33.1)
// CHECK:STDOUT:   %.loc44_33.4: ref %MaybeUnformed.589 = struct_access %self.param, element0
// CHECK:STDOUT:   %facet_value.loc44_33.3: %facet_type.8a3 = facet_value constants.%MaybeUnformed.589, (constants.%custom_witness.b80ef6.3, constants.%custom_witness.2cb555.5) [concrete = constants.%facet_value.a4f]
// CHECK:STDOUT:   %.loc44_33.5: %facet_type.8a3 = converted constants.%MaybeUnformed.589, %facet_value.loc44_33.3 [concrete = constants.%facet_value.a4f]
// CHECK:STDOUT:   %facet_value.loc44_33.4: %facet_type.8a3 = facet_value constants.%MaybeUnformed.589, (constants.%custom_witness.b80ef6.3, constants.%custom_witness.2cb555.5) [concrete = constants.%facet_value.a4f]
// CHECK:STDOUT:   %.loc44_33.6: %facet_type.8a3 = converted constants.%MaybeUnformed.589, %facet_value.loc44_33.4 [concrete = constants.%facet_value.a4f]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc44_33.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc44_33.4)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyImpl.WithSelf.Op.loc44_33.1(%self.param: ref bool) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyImpl.WithSelf.Op.loc44_33.2(%self.param: ref %.a18) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyImpl.WithSelf.Op.loc44_33.3(%self.param: ref %MaybeUnformed.589) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc44_33.5(%self.param: ref %DefaultOptionalStorage.fda) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %has_value.ref: %DefaultOptionalStorage.elem.8a1 = name_ref has_value, imports.%Core.import_ref.1cb [concrete = imports.%field_decl.255]
// CHECK:STDOUT:   %.loc44_33.1: ref bool = class_element_access %self.param, element1
// CHECK:STDOUT:   %facet_value.loc44_33.1: %facet_type.8a3 = facet_value bool, (constants.%custom_witness.b80ef6.4, constants.%custom_witness.2cb555.3) [concrete = constants.%facet_value.ff1]
// CHECK:STDOUT:   %.loc44_33.2: %facet_type.8a3 = converted bool, %facet_value.loc44_33.1 [concrete = constants.%facet_value.ff1]
// CHECK:STDOUT:   %facet_value.loc44_33.2: %facet_type.8a3 = facet_value bool, (constants.%custom_witness.b80ef6.4, constants.%custom_witness.2cb555.3) [concrete = constants.%facet_value.ff1]
// CHECK:STDOUT:   %.loc44_33.3: %facet_type.8a3 = converted bool, %facet_value.loc44_33.2 [concrete = constants.%facet_value.ff1]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc44_33.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc44_33.1)
// CHECK:STDOUT:   %value.ref: %DefaultOptionalStorage.elem.75c = name_ref value, imports.%Core.import_ref.1c5 [concrete = imports.%field_decl.649]
// CHECK:STDOUT:   %.loc44_33.4: ref %MaybeUnformed.589 = class_element_access %self.param, element0
// CHECK:STDOUT:   %facet_value.loc44_33.3: %facet_type.8a3 = facet_value constants.%MaybeUnformed.589, (constants.%custom_witness.b80ef6.3, constants.%custom_witness.2cb555.5) [concrete = constants.%facet_value.a4f]
// CHECK:STDOUT:   %.loc44_33.5: %facet_type.8a3 = converted constants.%MaybeUnformed.589, %facet_value.loc44_33.3 [concrete = constants.%facet_value.a4f]
// CHECK:STDOUT:   %facet_value.loc44_33.4: %facet_type.8a3 = facet_value constants.%MaybeUnformed.589, (constants.%custom_witness.b80ef6.3, constants.%custom_witness.2cb555.5) [concrete = constants.%facet_value.a4f]
// CHECK:STDOUT:   %.loc44_33.6: %facet_type.8a3 = converted constants.%MaybeUnformed.589, %facet_value.loc44_33.4 [concrete = constants.%facet_value.a4f]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc44_33.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc44_33.4)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc44_33.6(%self.param: ref %Optional.57a) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyImpl.WithSelf.Op.loc44_33.4(%self.param: ref %struct_type.value.has_value.759) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyImpl.WithSelf.Op.loc44_33.5(%self.param: ref %DefaultOptionalStorage.fda) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyImpl.WithSelf.Op.loc44_33.6(%self.param: ref %Optional.57a) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestRangeFor(%m.param: ref %DiffMutableRange, %c.param: %DiffConstRange) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %i.patt.loc59: %pattern_type.149 = value_binding_pattern i [concrete = constants.%i.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %m.ref: ref %DiffMutableRange = name_ref m, %m
// CHECK:STDOUT:   %impl.elem2.loc59: %.9d5 = impl_witness_access constants.%Iterate.impl_witness.feb, element2 [concrete = constants.%T.as_type.as.Iterate.impl.NewCursor.737]
// CHECK:STDOUT:   %bound_method.loc59_33.1: <bound method> = bound_method %m.ref, %impl.elem2.loc59
// CHECK:STDOUT:   %facet_value.loc59_33.1: %CppRange_where.type.d682d6.1 = facet_value constants.%DiffMutableRange, (constants.%custom_witness.5760f4.1, constants.%custom_witness.2cb555.2, constants.%custom_witness.d0da43.1, constants.%custom_witness.aa8, constants.%custom_witness.3815f0.1, constants.%custom_witness.2cb555.9, constants.%custom_witness.392, constants.%custom_witness.1c9, constants.%custom_witness.723, constants.%custom_witness.3815f0.2, constants.%custom_witness.2cb555.10) [concrete = constants.%facet_value.618]
// CHECK:STDOUT:   %.loc59_33.1: %CppRange_where.type.d682d6.1 = converted constants.%DiffMutableRange, %facet_value.loc59_33.1 [concrete = constants.%facet_value.618]
// CHECK:STDOUT:   %facet_value.loc59_33.2: %CppRange_where.type.d682d6.1 = facet_value constants.%DiffMutableRange, (constants.%custom_witness.5760f4.1, constants.%custom_witness.2cb555.2, constants.%custom_witness.d0da43.1, constants.%custom_witness.aa8, constants.%custom_witness.3815f0.1, constants.%custom_witness.2cb555.9, constants.%custom_witness.392, constants.%custom_witness.1c9, constants.%custom_witness.723, constants.%custom_witness.3815f0.2, constants.%custom_witness.2cb555.10) [concrete = constants.%facet_value.618]
// CHECK:STDOUT:   %.loc59_33.2: %CppRange_where.type.d682d6.1 = converted constants.%DiffMutableRange, %facet_value.loc59_33.2 [concrete = constants.%facet_value.618]
// CHECK:STDOUT:   %specific_fn.loc59_33.1: <specific function> = specific_function %impl.elem2.loc59, @T.as_type.as.Iterate.impl.NewCursor(constants.%facet_value.618) [concrete = constants.%T.as_type.as.Iterate.impl.NewCursor.specific_fn.7fa]
// CHECK:STDOUT:   %bound_method.loc59_33.2: <bound method> = bound_method %m.ref, %specific_fn.loc59_33.1
// CHECK:STDOUT:   %var.loc59: ref %tuple.type.b18 = var_storage invalid
// CHECK:STDOUT:   %.loc59_32.1: %DiffMutableRange = acquire_value %m.ref
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.call.loc59: init %tuple.type.b18 to %var.loc59 = call %bound_method.loc59_33.2(%.loc59_32.1)
// CHECK:STDOUT:   assign %var.loc59, %T.as_type.as.Iterate.impl.NewCursor.call.loc59
// CHECK:STDOUT:   br !for.next.loc59
// CHECK:STDOUT:
// CHECK:STDOUT: !for.next.loc59:
// CHECK:STDOUT:   %impl.elem3.loc59: %.d5f = impl_witness_access constants.%Iterate.impl_witness.feb, element3 [concrete = constants.%T.as_type.as.Iterate.impl.Next.2f3]
// CHECK:STDOUT:   %bound_method.loc59_33.3: <bound method> = bound_method %m.ref, %impl.elem3.loc59
// CHECK:STDOUT:   %facet_value.loc59_33.3: %CppRange_where.type.d682d6.1 = facet_value constants.%DiffMutableRange, (constants.%custom_witness.5760f4.1, constants.%custom_witness.2cb555.2, constants.%custom_witness.d0da43.1, constants.%custom_witness.aa8, constants.%custom_witness.3815f0.1, constants.%custom_witness.2cb555.9, constants.%custom_witness.392, constants.%custom_witness.1c9, constants.%custom_witness.723, constants.%custom_witness.3815f0.2, constants.%custom_witness.2cb555.10) [concrete = constants.%facet_value.618]
// CHECK:STDOUT:   %.loc59_33.3: %CppRange_where.type.d682d6.1 = converted constants.%DiffMutableRange, %facet_value.loc59_33.3 [concrete = constants.%facet_value.618]
// CHECK:STDOUT:   %facet_value.loc59_33.4: %CppRange_where.type.d682d6.1 = facet_value constants.%DiffMutableRange, (constants.%custom_witness.5760f4.1, constants.%custom_witness.2cb555.2, constants.%custom_witness.d0da43.1, constants.%custom_witness.aa8, constants.%custom_witness.3815f0.1, constants.%custom_witness.2cb555.9, constants.%custom_witness.392, constants.%custom_witness.1c9, constants.%custom_witness.723, constants.%custom_witness.3815f0.2, constants.%custom_witness.2cb555.10) [concrete = constants.%facet_value.618]
// CHECK:STDOUT:   %.loc59_33.4: %CppRange_where.type.d682d6.1 = converted constants.%DiffMutableRange, %facet_value.loc59_33.4 [concrete = constants.%facet_value.618]
// CHECK:STDOUT:   %specific_fn.loc59_33.2: <specific function> = specific_function %impl.elem3.loc59, @T.as_type.as.Iterate.impl.Next(constants.%facet_value.618) [concrete = constants.%T.as_type.as.Iterate.impl.Next.specific_fn.ea2]
// CHECK:STDOUT:   %bound_method.loc59_33.4: <bound method> = bound_method %m.ref, %specific_fn.loc59_33.2
// CHECK:STDOUT:   %.loc59_33.5: ref %Optional.57a = temporary_storage
// CHECK:STDOUT:   %.loc59_32.2: %DiffMutableRange = acquire_value %m.ref
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.call.loc59: init %Optional.57a to %.loc59_33.5 = call %bound_method.loc59_33.4(%.loc59_32.2, %var.loc59)
// CHECK:STDOUT:   %.loc59_33.6: ref %Optional.57a = temporary %.loc59_33.5, %T.as_type.as.Iterate.impl.Next.call.loc59
// CHECK:STDOUT:   %.loc59_33.7: %Optional.HasValue.type.85b = specific_constant imports.%Core.import_ref.bb6, @Optional(constants.%OptionalStorage.facet.e5c) [concrete = constants.%Optional.HasValue.d89]
// CHECK:STDOUT:   %HasValue.ref.loc59: %Optional.HasValue.type.85b = name_ref HasValue, %.loc59_33.7 [concrete = constants.%Optional.HasValue.d89]
// CHECK:STDOUT:   %Optional.HasValue.bound.loc59: <bound method> = bound_method %.loc59_33.6, %HasValue.ref.loc59
// CHECK:STDOUT:   %Optional.HasValue.specific_fn.loc59: <specific function> = specific_function %HasValue.ref.loc59, @Optional.HasValue(constants.%OptionalStorage.facet.e5c) [concrete = constants.%Optional.HasValue.specific_fn.449]
// CHECK:STDOUT:   %bound_method.loc59_33.5: <bound method> = bound_method %.loc59_33.6, %Optional.HasValue.specific_fn.loc59
// CHECK:STDOUT:   %.loc59_33.8: %Optional.57a = acquire_value %.loc59_33.6
// CHECK:STDOUT:   %Optional.HasValue.call.loc59: init bool = call %bound_method.loc59_33.5(%.loc59_33.8)
// CHECK:STDOUT:   %.loc59_33.9: bool = value_of_initializer %Optional.HasValue.call.loc59
// CHECK:STDOUT:   %.loc59_33.10: bool = converted %Optional.HasValue.call.loc59, %.loc59_33.9
// CHECK:STDOUT:   if %.loc59_33.10 br !for.body.loc59 else br !for.done.loc59
// CHECK:STDOUT:
// CHECK:STDOUT: !for.body.loc59:
// CHECK:STDOUT:   %.loc59_33.11: %Optional.Get.type.49a = specific_constant imports.%Core.import_ref.dab, @Optional(constants.%OptionalStorage.facet.e5c) [concrete = constants.%Optional.Get.c1e]
// CHECK:STDOUT:   %Get.ref.loc59: %Optional.Get.type.49a = name_ref Get, %.loc59_33.11 [concrete = constants.%Optional.Get.c1e]
// CHECK:STDOUT:   %Optional.Get.bound.loc59: <bound method> = bound_method %.loc59_33.6, %Get.ref.loc59
// CHECK:STDOUT:   %Optional.Get.specific_fn.loc59: <specific function> = specific_function %Get.ref.loc59, @Optional.Get(constants.%OptionalStorage.facet.e5c) [concrete = constants.%Optional.Get.specific_fn.27e]
// CHECK:STDOUT:   %bound_method.loc59_33.6: <bound method> = bound_method %.loc59_33.6, %Optional.Get.specific_fn.loc59
// CHECK:STDOUT:   %.loc59_33.12: ref %DiffValueType = temporary_storage
// CHECK:STDOUT:   %.loc59_33.13: %Optional.57a = acquire_value %.loc59_33.6
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
// CHECK:STDOUT:   %facet_value.loc59_33.5: %facet_type.8a3 = facet_value constants.%DiffValueType, (constants.%custom_witness.5760f4.1, constants.%custom_witness.2cb555.2) [concrete = constants.%facet_value.837]
// CHECK:STDOUT:   %.loc59_33.16: %facet_type.8a3 = converted constants.%DiffValueType, %facet_value.loc59_33.5 [concrete = constants.%facet_value.837]
// CHECK:STDOUT:   %facet_value.loc59_33.6: %facet_type.8a3 = facet_value constants.%DiffValueType, (constants.%custom_witness.5760f4.1, constants.%custom_witness.2cb555.2) [concrete = constants.%facet_value.837]
// CHECK:STDOUT:   %.loc59_33.17: %facet_type.8a3 = converted constants.%DiffValueType, %facet_value.loc59_33.6 [concrete = constants.%facet_value.837]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc59_33.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc59_33.14)
// CHECK:STDOUT:   %facet_value.loc59_33.7: %facet_type.8a3 = facet_value constants.%Optional.57a, (constants.%custom_witness.b80ef6.7, constants.%custom_witness.2cb555.8) [concrete = constants.%facet_value.971]
// CHECK:STDOUT:   %.loc59_33.18: %facet_type.8a3 = converted constants.%Optional.57a, %facet_value.loc59_33.7 [concrete = constants.%facet_value.971]
// CHECK:STDOUT:   %facet_value.loc59_33.8: %facet_type.8a3 = facet_value constants.%Optional.57a, (constants.%custom_witness.b80ef6.7, constants.%custom_witness.2cb555.8) [concrete = constants.%facet_value.971]
// CHECK:STDOUT:   %.loc59_33.19: %facet_type.8a3 = converted constants.%Optional.57a, %facet_value.loc59_33.8 [concrete = constants.%facet_value.971]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc59_33.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc59_33.6)
// CHECK:STDOUT:   br !for.next.loc59
// CHECK:STDOUT:
// CHECK:STDOUT: !for.done.loc59:
// CHECK:STDOUT:   %facet_value.loc59_33.9: %facet_type.8a3 = facet_value constants.%Optional.57a, (constants.%custom_witness.b80ef6.7, constants.%custom_witness.2cb555.8) [concrete = constants.%facet_value.971]
// CHECK:STDOUT:   %.loc59_33.20: %facet_type.8a3 = converted constants.%Optional.57a, %facet_value.loc59_33.9 [concrete = constants.%facet_value.971]
// CHECK:STDOUT:   %facet_value.loc59_33.10: %facet_type.8a3 = facet_value constants.%Optional.57a, (constants.%custom_witness.b80ef6.7, constants.%custom_witness.2cb555.8) [concrete = constants.%facet_value.971]
// CHECK:STDOUT:   %.loc59_33.21: %facet_type.8a3 = converted constants.%Optional.57a, %facet_value.loc59_33.10 [concrete = constants.%facet_value.971]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc59_33.3: init %empty_tuple.type = call %SelfDestruct.specific_fn.3(%.loc59_33.6)
// CHECK:STDOUT:   %facet_value.loc59_33.11: %facet_type.8a3 = facet_value constants.%tuple.type.b18, (constants.%custom_witness.b80ef6.8, constants.%custom_witness.2cb555.11) [concrete = constants.%facet_value.e0e]
// CHECK:STDOUT:   %.loc59_33.22: %facet_type.8a3 = converted constants.%tuple.type.b18, %facet_value.loc59_33.11 [concrete = constants.%facet_value.e0e]
// CHECK:STDOUT:   %facet_value.loc59_33.12: %facet_type.8a3 = facet_value constants.%tuple.type.b18, (constants.%custom_witness.b80ef6.8, constants.%custom_witness.2cb555.11) [concrete = constants.%facet_value.e0e]
// CHECK:STDOUT:   %.loc59_33.23: %facet_type.8a3 = converted constants.%tuple.type.b18, %facet_value.loc59_33.12 [concrete = constants.%facet_value.e0e]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc59_33.4: init %empty_tuple.type = call %SelfDestruct.specific_fn.4(%var.loc59)
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %i.patt.loc63: %pattern_type.149 = value_binding_pattern i [concrete = constants.%i.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c.ref: %DiffConstRange = name_ref c, %c
// CHECK:STDOUT:   %impl.elem2.loc63: %.7bf = impl_witness_access constants.%Iterate.impl_witness.3b1, element2 [concrete = constants.%T.as_type.as.Iterate.impl.NewCursor.9f4]
// CHECK:STDOUT:   %bound_method.loc63_33.1: <bound method> = bound_method %c.ref, %impl.elem2.loc63
// CHECK:STDOUT:   %facet_value.loc63_33.1: %CppRange_where.type.d682d6.1 = facet_value constants.%DiffConstRange, (constants.%custom_witness.5760f4.1, constants.%custom_witness.2cb555.2, constants.%custom_witness.d0da43.1, constants.%custom_witness.7b4, constants.%custom_witness.3d0529.1, constants.%custom_witness.2cb555.13, constants.%custom_witness.0c0, constants.%custom_witness.ffb, constants.%custom_witness.a26, constants.%custom_witness.3d0529.2, constants.%custom_witness.2cb555.14) [concrete = constants.%facet_value.f7a]
// CHECK:STDOUT:   %.loc63_33.1: %CppRange_where.type.d682d6.1 = converted constants.%DiffConstRange, %facet_value.loc63_33.1 [concrete = constants.%facet_value.f7a]
// CHECK:STDOUT:   %facet_value.loc63_33.2: %CppRange_where.type.d682d6.1 = facet_value constants.%DiffConstRange, (constants.%custom_witness.5760f4.1, constants.%custom_witness.2cb555.2, constants.%custom_witness.d0da43.1, constants.%custom_witness.7b4, constants.%custom_witness.3d0529.1, constants.%custom_witness.2cb555.13, constants.%custom_witness.0c0, constants.%custom_witness.ffb, constants.%custom_witness.a26, constants.%custom_witness.3d0529.2, constants.%custom_witness.2cb555.14) [concrete = constants.%facet_value.f7a]
// CHECK:STDOUT:   %.loc63_33.2: %CppRange_where.type.d682d6.1 = converted constants.%DiffConstRange, %facet_value.loc63_33.2 [concrete = constants.%facet_value.f7a]
// CHECK:STDOUT:   %specific_fn.loc63_33.1: <specific function> = specific_function %impl.elem2.loc63, @T.as_type.as.Iterate.impl.NewCursor(constants.%facet_value.f7a) [concrete = constants.%T.as_type.as.Iterate.impl.NewCursor.specific_fn.ccd]
// CHECK:STDOUT:   %bound_method.loc63_33.2: <bound method> = bound_method %c.ref, %specific_fn.loc63_33.1
// CHECK:STDOUT:   %var.loc63: ref %tuple.type.df5 = var_storage invalid
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.call.loc63: init %tuple.type.df5 to %var.loc63 = call %bound_method.loc63_33.2(%c.ref)
// CHECK:STDOUT:   assign %var.loc63, %T.as_type.as.Iterate.impl.NewCursor.call.loc63
// CHECK:STDOUT:   br !for.next.loc63
// CHECK:STDOUT:
// CHECK:STDOUT: !for.next.loc63:
// CHECK:STDOUT:   %impl.elem3.loc63: %.68d = impl_witness_access constants.%Iterate.impl_witness.3b1, element3 [concrete = constants.%T.as_type.as.Iterate.impl.Next.0c4]
// CHECK:STDOUT:   %bound_method.loc63_33.3: <bound method> = bound_method %c.ref, %impl.elem3.loc63
// CHECK:STDOUT:   %facet_value.loc63_33.3: %CppRange_where.type.d682d6.1 = facet_value constants.%DiffConstRange, (constants.%custom_witness.5760f4.1, constants.%custom_witness.2cb555.2, constants.%custom_witness.d0da43.1, constants.%custom_witness.7b4, constants.%custom_witness.3d0529.1, constants.%custom_witness.2cb555.13, constants.%custom_witness.0c0, constants.%custom_witness.ffb, constants.%custom_witness.a26, constants.%custom_witness.3d0529.2, constants.%custom_witness.2cb555.14) [concrete = constants.%facet_value.f7a]
// CHECK:STDOUT:   %.loc63_33.3: %CppRange_where.type.d682d6.1 = converted constants.%DiffConstRange, %facet_value.loc63_33.3 [concrete = constants.%facet_value.f7a]
// CHECK:STDOUT:   %facet_value.loc63_33.4: %CppRange_where.type.d682d6.1 = facet_value constants.%DiffConstRange, (constants.%custom_witness.5760f4.1, constants.%custom_witness.2cb555.2, constants.%custom_witness.d0da43.1, constants.%custom_witness.7b4, constants.%custom_witness.3d0529.1, constants.%custom_witness.2cb555.13, constants.%custom_witness.0c0, constants.%custom_witness.ffb, constants.%custom_witness.a26, constants.%custom_witness.3d0529.2, constants.%custom_witness.2cb555.14) [concrete = constants.%facet_value.f7a]
// CHECK:STDOUT:   %.loc63_33.4: %CppRange_where.type.d682d6.1 = converted constants.%DiffConstRange, %facet_value.loc63_33.4 [concrete = constants.%facet_value.f7a]
// CHECK:STDOUT:   %specific_fn.loc63_33.2: <specific function> = specific_function %impl.elem3.loc63, @T.as_type.as.Iterate.impl.Next(constants.%facet_value.f7a) [concrete = constants.%T.as_type.as.Iterate.impl.Next.specific_fn.b8b]
// CHECK:STDOUT:   %bound_method.loc63_33.4: <bound method> = bound_method %c.ref, %specific_fn.loc63_33.2
// CHECK:STDOUT:   %.loc63_33.5: ref %Optional.57a = temporary_storage
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.call.loc63: init %Optional.57a to %.loc63_33.5 = call %bound_method.loc63_33.4(%c.ref, %var.loc63)
// CHECK:STDOUT:   %.loc63_33.6: ref %Optional.57a = temporary %.loc63_33.5, %T.as_type.as.Iterate.impl.Next.call.loc63
// CHECK:STDOUT:   %.loc63_33.7: %Optional.HasValue.type.85b = specific_constant imports.%Core.import_ref.bb6, @Optional(constants.%OptionalStorage.facet.e5c) [concrete = constants.%Optional.HasValue.d89]
// CHECK:STDOUT:   %HasValue.ref.loc63: %Optional.HasValue.type.85b = name_ref HasValue, %.loc63_33.7 [concrete = constants.%Optional.HasValue.d89]
// CHECK:STDOUT:   %Optional.HasValue.bound.loc63: <bound method> = bound_method %.loc63_33.6, %HasValue.ref.loc63
// CHECK:STDOUT:   %Optional.HasValue.specific_fn.loc63: <specific function> = specific_function %HasValue.ref.loc63, @Optional.HasValue(constants.%OptionalStorage.facet.e5c) [concrete = constants.%Optional.HasValue.specific_fn.449]
// CHECK:STDOUT:   %bound_method.loc63_33.5: <bound method> = bound_method %.loc63_33.6, %Optional.HasValue.specific_fn.loc63
// CHECK:STDOUT:   %.loc63_33.8: %Optional.57a = acquire_value %.loc63_33.6
// CHECK:STDOUT:   %Optional.HasValue.call.loc63: init bool = call %bound_method.loc63_33.5(%.loc63_33.8)
// CHECK:STDOUT:   %.loc63_33.9: bool = value_of_initializer %Optional.HasValue.call.loc63
// CHECK:STDOUT:   %.loc63_33.10: bool = converted %Optional.HasValue.call.loc63, %.loc63_33.9
// CHECK:STDOUT:   if %.loc63_33.10 br !for.body.loc63 else br !for.done.loc63
// CHECK:STDOUT:
// CHECK:STDOUT: !for.body.loc63:
// CHECK:STDOUT:   %.loc63_33.11: %Optional.Get.type.49a = specific_constant imports.%Core.import_ref.dab, @Optional(constants.%OptionalStorage.facet.e5c) [concrete = constants.%Optional.Get.c1e]
// CHECK:STDOUT:   %Get.ref.loc63: %Optional.Get.type.49a = name_ref Get, %.loc63_33.11 [concrete = constants.%Optional.Get.c1e]
// CHECK:STDOUT:   %Optional.Get.bound.loc63: <bound method> = bound_method %.loc63_33.6, %Get.ref.loc63
// CHECK:STDOUT:   %Optional.Get.specific_fn.loc63: <specific function> = specific_function %Get.ref.loc63, @Optional.Get(constants.%OptionalStorage.facet.e5c) [concrete = constants.%Optional.Get.specific_fn.27e]
// CHECK:STDOUT:   %bound_method.loc63_33.6: <bound method> = bound_method %.loc63_33.6, %Optional.Get.specific_fn.loc63
// CHECK:STDOUT:   %.loc63_33.12: ref %DiffValueType = temporary_storage
// CHECK:STDOUT:   %.loc63_33.13: %Optional.57a = acquire_value %.loc63_33.6
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
// CHECK:STDOUT:   %facet_value.loc63_33.5: %facet_type.8a3 = facet_value constants.%DiffValueType, (constants.%custom_witness.5760f4.1, constants.%custom_witness.2cb555.2) [concrete = constants.%facet_value.837]
// CHECK:STDOUT:   %.loc63_33.16: %facet_type.8a3 = converted constants.%DiffValueType, %facet_value.loc63_33.5 [concrete = constants.%facet_value.837]
// CHECK:STDOUT:   %facet_value.loc63_33.6: %facet_type.8a3 = facet_value constants.%DiffValueType, (constants.%custom_witness.5760f4.1, constants.%custom_witness.2cb555.2) [concrete = constants.%facet_value.837]
// CHECK:STDOUT:   %.loc63_33.17: %facet_type.8a3 = converted constants.%DiffValueType, %facet_value.loc63_33.6 [concrete = constants.%facet_value.837]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc63_33.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.5(%.loc63_33.14)
// CHECK:STDOUT:   %facet_value.loc63_33.7: %facet_type.8a3 = facet_value constants.%Optional.57a, (constants.%custom_witness.b80ef6.7, constants.%custom_witness.2cb555.8) [concrete = constants.%facet_value.971]
// CHECK:STDOUT:   %.loc63_33.18: %facet_type.8a3 = converted constants.%Optional.57a, %facet_value.loc63_33.7 [concrete = constants.%facet_value.971]
// CHECK:STDOUT:   %facet_value.loc63_33.8: %facet_type.8a3 = facet_value constants.%Optional.57a, (constants.%custom_witness.b80ef6.7, constants.%custom_witness.2cb555.8) [concrete = constants.%facet_value.971]
// CHECK:STDOUT:   %.loc63_33.19: %facet_type.8a3 = converted constants.%Optional.57a, %facet_value.loc63_33.8 [concrete = constants.%facet_value.971]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc63_33.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.6(%.loc63_33.6)
// CHECK:STDOUT:   br !for.next.loc63
// CHECK:STDOUT:
// CHECK:STDOUT: !for.done.loc63:
// CHECK:STDOUT:   %facet_value.loc63_33.9: %facet_type.8a3 = facet_value constants.%Optional.57a, (constants.%custom_witness.b80ef6.7, constants.%custom_witness.2cb555.8) [concrete = constants.%facet_value.971]
// CHECK:STDOUT:   %.loc63_33.20: %facet_type.8a3 = converted constants.%Optional.57a, %facet_value.loc63_33.9 [concrete = constants.%facet_value.971]
// CHECK:STDOUT:   %facet_value.loc63_33.10: %facet_type.8a3 = facet_value constants.%Optional.57a, (constants.%custom_witness.b80ef6.7, constants.%custom_witness.2cb555.8) [concrete = constants.%facet_value.971]
// CHECK:STDOUT:   %.loc63_33.21: %facet_type.8a3 = converted constants.%Optional.57a, %facet_value.loc63_33.10 [concrete = constants.%facet_value.971]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc63_33.3: init %empty_tuple.type = call %SelfDestruct.specific_fn.7(%.loc63_33.6)
// CHECK:STDOUT:   %facet_value.loc63_33.11: %facet_type.8a3 = facet_value constants.%tuple.type.df5, (constants.%custom_witness.b80ef6.9, constants.%custom_witness.2cb555.15) [concrete = constants.%facet_value.495]
// CHECK:STDOUT:   %.loc63_33.22: %facet_type.8a3 = converted constants.%tuple.type.df5, %facet_value.loc63_33.11 [concrete = constants.%facet_value.495]
// CHECK:STDOUT:   %facet_value.loc63_33.12: %facet_type.8a3 = facet_value constants.%tuple.type.df5, (constants.%custom_witness.b80ef6.9, constants.%custom_witness.2cb555.15) [concrete = constants.%facet_value.495]
// CHECK:STDOUT:   %.loc63_33.23: %facet_type.8a3 = converted constants.%tuple.type.df5, %facet_value.loc63_33.12 [concrete = constants.%facet_value.495]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc63_33.4: init %empty_tuple.type = call %SelfDestruct.specific_fn.8(%var.loc63)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestIterate(constants.%R) {
// CHECK:STDOUT:   %R.patt.loc40_17.2 => constants.%R.patt
// CHECK:STDOUT:   %R.loc40_17.1 => constants.%R
// CHECK:STDOUT:   %R.as_type.loc40_79.1 => constants.%R.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.0a0
// CHECK:STDOUT:   %r.patt.loc40_77.2 => constants.%r.patt.029
// CHECK:STDOUT:   %r.param_patt.loc40_72.2 => constants.%r.param_patt.b15
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestIterate(constants.%facet_value.7b7) {
// CHECK:STDOUT:   %R.patt.loc40_17.2 => constants.%R.patt
// CHECK:STDOUT:   %R.loc40_17.1 => constants.%facet_value.7b7
// CHECK:STDOUT:   %R.as_type.loc40_79.1 => constants.%DiffMutableRange
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.611
// CHECK:STDOUT:   %r.patt.loc40_77.2 => constants.%r.patt.5fe
// CHECK:STDOUT:   %r.param_patt.loc40_72.2 => constants.%r.param_patt.a4b
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc40 => constants.%complete_type.357
// CHECK:STDOUT:   %Iterate.lookup_impl_witness => constants.%Iterate.impl_witness.feb
// CHECK:STDOUT:   %Iterate.facet.loc44_33.5 => constants.%Iterate.facet.811
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type => constants.%Iterate.WithSelf.NewCursor.type.54d
// CHECK:STDOUT:   %.loc44_33.26 => constants.%.9d5
// CHECK:STDOUT:   %impl.elem2.loc44_33.2 => constants.%T.as_type.as.Iterate.impl.NewCursor.737
// CHECK:STDOUT:   %specific_impl_fn.loc44_33.3 => constants.%T.as_type.as.Iterate.impl.NewCursor.specific_fn.7fa
// CHECK:STDOUT:   %impl.elem1 => constants.%facet_value.e0e
// CHECK:STDOUT:   %as_type => constants.%tuple.type.b18
// CHECK:STDOUT:   %require_complete.1 => constants.%complete_type.bfe
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type => constants.%Iterate.WithSelf.Next.type.1fc
// CHECK:STDOUT:   %.loc44_33.27 => constants.%.d5f
// CHECK:STDOUT:   %impl.elem3.loc44_33.2 => constants.%T.as_type.as.Iterate.impl.Next.2f3
// CHECK:STDOUT:   %specific_impl_fn.loc44_33.4 => constants.%T.as_type.as.Iterate.impl.Next.specific_fn.ea2
// CHECK:STDOUT:   %SelfDestruct.specific_fn.6 => constants.%SelfDestruct.specific_fn.caa
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestIterate(constants.%facet_value.6e5) {
// CHECK:STDOUT:   %R.patt.loc40_17.2 => constants.%R.patt
// CHECK:STDOUT:   %R.loc40_17.1 => constants.%facet_value.6e5
// CHECK:STDOUT:   %R.as_type.loc40_79.1 => constants.%DiffConstRange
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.7dd
// CHECK:STDOUT:   %r.patt.loc40_77.2 => constants.%r.patt.9c0
// CHECK:STDOUT:   %r.param_patt.loc40_72.2 => constants.%r.param_patt.992
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc40 => constants.%complete_type.357
// CHECK:STDOUT:   %Iterate.lookup_impl_witness => constants.%Iterate.impl_witness.3b1
// CHECK:STDOUT:   %Iterate.facet.loc44_33.5 => constants.%Iterate.facet.bdf
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type => constants.%Iterate.WithSelf.NewCursor.type.fe4
// CHECK:STDOUT:   %.loc44_33.26 => constants.%.7bf
// CHECK:STDOUT:   %impl.elem2.loc44_33.2 => constants.%T.as_type.as.Iterate.impl.NewCursor.9f4
// CHECK:STDOUT:   %specific_impl_fn.loc44_33.3 => constants.%T.as_type.as.Iterate.impl.NewCursor.specific_fn.ccd
// CHECK:STDOUT:   %impl.elem1 => constants.%facet_value.495
// CHECK:STDOUT:   %as_type => constants.%tuple.type.df5
// CHECK:STDOUT:   %require_complete.1 => constants.%complete_type.ad6
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type => constants.%Iterate.WithSelf.Next.type.b60
// CHECK:STDOUT:   %.loc44_33.27 => constants.%.68d
// CHECK:STDOUT:   %impl.elem3.loc44_33.2 => constants.%T.as_type.as.Iterate.impl.Next.0c4
// CHECK:STDOUT:   %specific_impl_fn.loc44_33.4 => constants.%T.as_type.as.Iterate.impl.Next.specific_fn.b8b
// CHECK:STDOUT:   %SelfDestruct.specific_fn.6 => constants.%SelfDestruct.specific_fn.6e2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- method_overloads.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %Iterate.type: type = facet_type <@Iterate> [concrete]
// CHECK:STDOUT:   %OptionalStorage.type: type = facet_type <@OptionalStorage> [concrete]
// CHECK:STDOUT:   %facet_type.d61: type = facet_type <@SubobjectDestroy & @DestroyImpl & @Copy> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %T.220: %OptionalStorage.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Optional.Get.type.bb9: type = fn_type @Optional.Get, @Optional(%T.220) [symbolic]
// CHECK:STDOUT:   %Optional.Get.fa1: %Optional.Get.type.bb9 = struct_value () [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.type.e9e: type = fn_type @Optional.HasValue, @Optional(%T.220) [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.6cd: %Optional.HasValue.type.e9e = struct_value () [symbolic]
// CHECK:STDOUT:   %facet_type.8a3: type = facet_type <@SubobjectDestroy & @DestroyImpl> [concrete]
// CHECK:STDOUT:   %T.9ab: %facet_type.d61 = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.19c: type = facet_access_type %T.9ab [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.lookup_impl_witness.855: <witness> = lookup_impl_witness %T.9ab, @SubobjectDestroy [symbolic]
// CHECK:STDOUT:   %DestroyImpl.lookup_impl_witness.ff9: <witness> = lookup_impl_witness %T.9ab, @DestroyImpl [symbolic]
// CHECK:STDOUT:   %facet_value.b35: %facet_type.8a3 = facet_value %T.as_type.19c, (%SubobjectDestroy.lookup_impl_witness.855, %DestroyImpl.lookup_impl_witness.ff9) [symbolic]
// CHECK:STDOUT:   %MaybeUnformed.2cf: type = class_type @MaybeUnformed, @MaybeUnformed(%facet_value.b35) [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.7db: type = class_type @DefaultOptionalStorage, @DefaultOptionalStorage(%T.9ab) [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.elem.98d: type = unbound_element_type %DefaultOptionalStorage.7db, bool [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.elem.b5a: type = unbound_element_type %DefaultOptionalStorage.7db, %MaybeUnformed.2cf [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Get.type.b32: type = fn_type @T.as_type.as.OptionalStorage.impl.Get, @T.as_type.as.OptionalStorage.impl(%T.9ab) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Get.8b0: %T.as_type.as.OptionalStorage.impl.Get.type.b32 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Has.type.8e5: type = fn_type @T.as_type.as.OptionalStorage.impl.Has, @T.as_type.as.OptionalStorage.impl(%T.9ab) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Has.2d5: %T.as_type.as.OptionalStorage.impl.Has.type.8e5 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Some.type.61d: type = fn_type @T.as_type.as.OptionalStorage.impl.Some, @T.as_type.as.OptionalStorage.impl(%T.9ab) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Some.ba7: %T.as_type.as.OptionalStorage.impl.Some.type.61d = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.None.type.a9f: type = fn_type @T.as_type.as.OptionalStorage.impl.None, @T.as_type.as.OptionalStorage.impl(%T.9ab) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.None.083: %T.as_type.as.OptionalStorage.impl.None.type.a9f = struct_value () [symbolic]
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
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.type.b77747.3: type = fn_type @DestroyImpl.WithSelf.Op.4 [concrete]
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.25aa57.3: %DestroyImpl.WithSelf.Op.type.b77747.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.2cb555.2: <witness> = custom_witness (%DestroyImpl.WithSelf.Op.25aa57.3), @DestroyImpl [concrete]
// CHECK:STDOUT:   %OverloadValueType.Op.type.83af6e.2: type = fn_type @OverloadValueType.Op.2 [concrete]
// CHECK:STDOUT:   %OverloadValueType.Op.4dca31.2: %OverloadValueType.Op.type.83af6e.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.d0da43.1: <witness> = custom_witness (%OverloadValueType.Op.4dca31.2), @Copy [concrete]
// CHECK:STDOUT:   %facet_value.26a: %facet_type.d61 = facet_value %OverloadValueType, (%custom_witness.5760f4.1, %custom_witness.2cb555.2, %custom_witness.d0da43.1) [concrete]
// CHECK:STDOUT:   %Iterate.lookup_impl_witness.1d8: <witness> = lookup_impl_witness %.Self.500, @Iterate [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.8a4: %facet_type.d61 = impl_witness_access %Iterate.lookup_impl_witness.1d8, element0 [symbolic_self]
// CHECK:STDOUT:   %Iterate_where.type.537: type = facet_type <@Iterate where %impl.elem0.8a4 = %facet_value.26a> [concrete]
// CHECK:STDOUT:   %pattern_type.847: type = pattern_type %Iterate_where.type.537 [concrete]
// CHECK:STDOUT:   %R.patt: %pattern_type.847 = symbolic_binding_pattern R, 0 [symbolic]
// CHECK:STDOUT:   %R: %Iterate_where.type.537 = symbolic_binding R, 0 [symbolic]
// CHECK:STDOUT:   %R.as_type: type = facet_access_type %R [symbolic]
// CHECK:STDOUT:   %pattern_type.f1e: type = pattern_type %R.as_type [symbolic]
// CHECK:STDOUT:   %r.patt.b92: %pattern_type.f1e = ref_binding_pattern r [symbolic]
// CHECK:STDOUT:   %r.param_patt.1ca: %pattern_type.f1e = var_param_pattern %r.patt.b92 [symbolic]
// CHECK:STDOUT:   %_.patt.a10: %pattern_type.804 = value_binding_pattern _ [concrete]
// CHECK:STDOUT:   %Iterate.lookup_impl_witness.7ec: <witness> = lookup_impl_witness %R, @Iterate [symbolic]
// CHECK:STDOUT:   %Iterate.facet.1d6: %Iterate.type = facet_value %R.as_type, (%Iterate.lookup_impl_witness.7ec) [symbolic]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.b13: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.1d6) [symbolic]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.a34: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.1d6) [symbolic]
// CHECK:STDOUT:   %.204: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.b13, %Iterate.facet.1d6 [symbolic]
// CHECK:STDOUT:   %impl.elem2.098: %.204 = impl_witness_access %Iterate.lookup_impl_witness.7ec, element2 [symbolic]
// CHECK:STDOUT:   %impl.elem1.8d0: %facet_type.8a3 = impl_witness_access %Iterate.lookup_impl_witness.7ec, element1 [symbolic]
// CHECK:STDOUT:   %as_type.5f7: type = facet_access_type %impl.elem1.8d0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.389: <specific function> = specific_impl_function %impl.elem2.098, @Iterate.WithSelf.NewCursor(%Iterate.facet.1d6) [symbolic]
// CHECK:STDOUT:   %.5cd: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.a34, %Iterate.facet.1d6 [symbolic]
// CHECK:STDOUT:   %impl.elem3.815: %.5cd = impl_witness_access %Iterate.lookup_impl_witness.7ec, element3 [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.0cb: type = class_type @DefaultOptionalStorage, @DefaultOptionalStorage(%facet_value.26a) [concrete]
// CHECK:STDOUT:   %OptionalStorage.impl_witness.505: <witness> = impl_witness imports.%OptionalStorage.impl_witness_table.968, @T.as_type.as.OptionalStorage.impl(%facet_value.26a) [concrete]
// CHECK:STDOUT:   %OptionalStorage.facet.2bc: %OptionalStorage.type = facet_value %OverloadValueType, (%OptionalStorage.impl_witness.505) [concrete]
// CHECK:STDOUT:   %Optional.328: type = class_type @Optional, @Optional(%OptionalStorage.facet.2bc) [concrete]
// CHECK:STDOUT:   %specific_impl_fn.0aa: <specific function> = specific_impl_function %impl.elem3.815, @Iterate.WithSelf.Next(%Iterate.facet.1d6) [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.type.e8d: type = fn_type @Optional.HasValue, @Optional(%OptionalStorage.facet.2bc) [concrete]
// CHECK:STDOUT:   %Optional.HasValue.4bf: %Optional.HasValue.type.e8d = struct_value () [concrete]
// CHECK:STDOUT:   %Optional.Get.type.b0b: type = fn_type @Optional.Get, @Optional(%OptionalStorage.facet.2bc) [concrete]
// CHECK:STDOUT:   %Optional.Get.30b: %Optional.Get.type.b0b = struct_value () [concrete]
// CHECK:STDOUT:   %facet_value.a01: %facet_type.8a3 = facet_value %OverloadValueType, (%custom_witness.5760f4.1, %custom_witness.2cb555.2) [concrete]
// CHECK:STDOUT:   %MaybeUnformed.d03: type = class_type @MaybeUnformed, @MaybeUnformed(%facet_value.a01) [concrete]
// CHECK:STDOUT:   %.8a2: type = maybe_unformed_type %OverloadValueType [concrete]
// CHECK:STDOUT:   %DefaultOptionalStorage.elem.6a5: type = unbound_element_type %DefaultOptionalStorage.0cb, %MaybeUnformed.d03 [concrete]
// CHECK:STDOUT:   %DefaultOptionalStorage.elem.9fe: type = unbound_element_type %DefaultOptionalStorage.0cb, bool [concrete]
// CHECK:STDOUT:   %struct_type.value.has_value.3d3: type = struct_type {.value: %MaybeUnformed.d03, .has_value: bool} [concrete]
// CHECK:STDOUT:   %Optional.HasValue.specific_fn.555: <specific function> = specific_function %Optional.HasValue.4bf, @Optional.HasValue(%OptionalStorage.facet.2bc) [concrete]
// CHECK:STDOUT:   %Optional.Get.specific_fn.519: <specific function> = specific_function %Optional.Get.30b, @Optional.Get(%OptionalStorage.facet.2bc) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.3: type = fn_type @SubobjectDestroy.WithSelf.Op.loc27_37.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.3: %SubobjectDestroy.WithSelf.Op.type.6870fd.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.3: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.3), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.4: type = fn_type @SubobjectDestroy.WithSelf.Op.loc27_37.3 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.4: %SubobjectDestroy.WithSelf.Op.type.6870fd.4 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.4: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.4), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.type.b77747.4: type = fn_type @DestroyImpl.WithSelf.Op.loc27_37.1 [concrete]
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.25aa57.4: %DestroyImpl.WithSelf.Op.type.b77747.4 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.2cb555.3: <witness> = custom_witness (%DestroyImpl.WithSelf.Op.25aa57.4), @DestroyImpl [concrete]
// CHECK:STDOUT:   %facet_value.ff1: %facet_type.8a3 = facet_value bool, (%custom_witness.b80ef6.4, %custom_witness.2cb555.3) [concrete]
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.type.b77747.6: type = fn_type @DestroyImpl.WithSelf.Op.loc27_37.3 [concrete]
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.25aa57.6: %DestroyImpl.WithSelf.Op.type.b77747.6 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.2cb555.5: <witness> = custom_witness (%DestroyImpl.WithSelf.Op.25aa57.6), @DestroyImpl [concrete]
// CHECK:STDOUT:   %facet_value.bc3: %facet_type.8a3 = facet_value %MaybeUnformed.d03, (%custom_witness.b80ef6.3, %custom_witness.2cb555.5) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.7: type = fn_type @SubobjectDestroy.WithSelf.Op.loc27_37.6 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.7: %SubobjectDestroy.WithSelf.Op.type.6870fd.7 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.7: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.7), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.type.b77747.9: type = fn_type @DestroyImpl.WithSelf.Op.loc27_37.6 [concrete]
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.25aa57.9: %DestroyImpl.WithSelf.Op.type.b77747.9 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.2cb555.8: <witness> = custom_witness (%DestroyImpl.WithSelf.Op.25aa57.9), @DestroyImpl [concrete]
// CHECK:STDOUT:   %facet_value.4c2: %facet_type.8a3 = facet_value %Optional.328, (%custom_witness.b80ef6.7, %custom_witness.2cb555.8) [concrete]
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
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.type.b77747.11: type = fn_type @DestroyImpl.WithSelf.Op.6 [concrete]
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.25aa57.11: %DestroyImpl.WithSelf.Op.type.b77747.11 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.2cb555.9: <witness> = custom_witness (%DestroyImpl.WithSelf.Op.25aa57.11), @DestroyImpl [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.4232e0.2: type = fn_type @Iterator.Op.2 [concrete]
// CHECK:STDOUT:   %Iterator.Op.e8caec.2: %Iterator.Op.type.4232e0.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.51e: <witness> = custom_witness (%Iterator.Op.e8caec.2), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.4232e0.3: type = fn_type @Iterator.Op.3 [concrete]
// CHECK:STDOUT:   %Iterator.Op.e8caec.3: %Iterator.Op.type.4232e0.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.716: <witness> = custom_witness (%Iterator.Op.e8caec.3), @Inc [concrete]
// CHECK:STDOUT:   %Equal.type: type = fn_type @Equal [concrete]
// CHECK:STDOUT:   %Equal: %Equal.type = struct_value () [concrete]
// CHECK:STDOUT:   %NotEqual.type: type = fn_type @NotEqual [concrete]
// CHECK:STDOUT:   %NotEqual: %NotEqual.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.617: <witness> = custom_witness (%Equal, %NotEqual), @EqWith, @EqWith(%Iterator) [concrete]
// CHECK:STDOUT:   %facet_value.de8: %CppRange_where.type.d682d6.1 = facet_value %Range, (%custom_witness.5760f4.1, %custom_witness.2cb555.2, %custom_witness.d0da43.1, %custom_witness.090, %custom_witness.51e, %custom_witness.2cb555.9, %custom_witness.6bd, %custom_witness.716, %custom_witness.617) [concrete]
// CHECK:STDOUT:   %tuple.type.9ef: type = tuple_type (%Iterator, %Iterator) [concrete]
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.type.b77747.12: type = fn_type @DestroyImpl.WithSelf.Op.7 [concrete]
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.25aa57.12: %DestroyImpl.WithSelf.Op.type.b77747.12 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.2cb555.10: <witness> = custom_witness (%DestroyImpl.WithSelf.Op.25aa57.12), @DestroyImpl [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.8: type = fn_type @SubobjectDestroy.WithSelf.Op.3 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.8: %SubobjectDestroy.WithSelf.Op.type.6870fd.8 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.8: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.8), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %facet_value.ee1: %facet_type.8a3 = facet_value %tuple.type.9ef, (%custom_witness.b80ef6.8, %custom_witness.2cb555.10) [concrete]
// CHECK:STDOUT:   %Iterate.impl_witness.3aa: <witness> = impl_witness imports.%Iterate.impl_witness_table.121, @T.as_type.as.Iterate.impl(%facet_value.de8) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.type.295: type = fn_type @T.as_type.as.Iterate.impl.NewCursor, @T.as_type.as.Iterate.impl(%facet_value.de8) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.880: %T.as_type.as.Iterate.impl.NewCursor.type.295 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.type.562: type = fn_type @T.as_type.as.Iterate.impl.Next, @T.as_type.as.Iterate.impl(%facet_value.de8) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.1cb: %T.as_type.as.Iterate.impl.Next.type.562 = struct_value () [concrete]
// CHECK:STDOUT:   %Iterate.facet.b70: %Iterate.type = facet_value %Range, (%Iterate.impl_witness.3aa) [concrete]
// CHECK:STDOUT:   %facet_value.f01: %Iterate_where.type.537 = facet_value %Range, (%Iterate.impl_witness.3aa) [concrete]
// CHECK:STDOUT:   %r.patt.b9f: %pattern_type.23f = ref_binding_pattern r [concrete]
// CHECK:STDOUT:   %r.param_patt.466: %pattern_type.23f = var_param_pattern %r.patt.b9f [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.719: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.b70) [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.da8: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.b70) [concrete]
// CHECK:STDOUT:   %.3d5: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.719, %Iterate.facet.b70 [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.specific_fn: <specific function> = specific_function %T.as_type.as.Iterate.impl.NewCursor.880, @T.as_type.as.Iterate.impl.NewCursor(%facet_value.de8) [concrete]
// CHECK:STDOUT:   %.569: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.da8, %Iterate.facet.b70 [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.specific_fn: <specific function> = specific_function %T.as_type.as.Iterate.impl.Next.1cb, @T.as_type.as.Iterate.impl.Next(%facet_value.de8) [concrete]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.1d5: <specific function> = specific_function %SelfDestruct, @SelfDestruct(%facet_value.ee1) [concrete]
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
// CHECK:STDOUT:   %Core.import_ref.38a: type = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%DefaultOptionalStorage (constants.%DefaultOptionalStorage.7db)]
// CHECK:STDOUT:   %Core.import_ref.61e: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.None.type (%T.as_type.as.OptionalStorage.impl.None.type.a9f) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.None (constants.%T.as_type.as.OptionalStorage.impl.None.083)]
// CHECK:STDOUT:   %Core.import_ref.ca1: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Some.type (%T.as_type.as.OptionalStorage.impl.Some.type.61d) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Some (constants.%T.as_type.as.OptionalStorage.impl.Some.ba7)]
// CHECK:STDOUT:   %Core.import_ref.6ba4: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Has.type (%T.as_type.as.OptionalStorage.impl.Has.type.8e5) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Has (constants.%T.as_type.as.OptionalStorage.impl.Has.2d5)]
// CHECK:STDOUT:   %Core.import_ref.73d: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Get.type (%T.as_type.as.OptionalStorage.impl.Get.type.b32) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Get (constants.%T.as_type.as.OptionalStorage.impl.Get.8b0)]
// CHECK:STDOUT:   %Core.import_ref.778 = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], unloaded
// CHECK:STDOUT:   %OptionalStorage.impl_witness_table.968 = impl_witness_table (%Core.import_ref.38a, %Core.import_ref.61e, %Core.import_ref.ca1, %Core.import_ref.6ba4, %Core.import_ref.73d, %Core.import_ref.778), @T.as_type.as.OptionalStorage.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.1c5: @DefaultOptionalStorage.%DefaultOptionalStorage.elem.1 (%DefaultOptionalStorage.elem.b5a) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [concrete = %field_decl.649]
// CHECK:STDOUT:   %Core.import_ref.1cb: @DefaultOptionalStorage.%DefaultOptionalStorage.elem.2 (%DefaultOptionalStorage.elem.98d) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [concrete = %field_decl.255]
// CHECK:STDOUT:   %Core.import_ref.d6c: %facet_type.d61 = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.Iterate.impl.%facet_value.1 (constants.%facet_value.9da)]
// CHECK:STDOUT:   %Core.import_ref.2ac: %facet_type.8a3 = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.Iterate.impl.%facet_value.2 (constants.%facet_value.bf6)]
// CHECK:STDOUT:   %Core.import_ref.fd1: @T.as_type.as.Iterate.impl.%T.as_type.as.Iterate.impl.NewCursor.type (%T.as_type.as.Iterate.impl.NewCursor.type.953) = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.Iterate.impl.%T.as_type.as.Iterate.impl.NewCursor (constants.%T.as_type.as.Iterate.impl.NewCursor.544)]
// CHECK:STDOUT:   %Core.import_ref.273: @T.as_type.as.Iterate.impl.%T.as_type.as.Iterate.impl.Next.type (%T.as_type.as.Iterate.impl.Next.type.b36) = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.Iterate.impl.%T.as_type.as.Iterate.impl.Next (constants.%T.as_type.as.Iterate.impl.Next.d6b)]
// CHECK:STDOUT:   %Iterate.impl_witness_table.121 = impl_witness_table (%Core.import_ref.d6c, %Core.import_ref.2ac, %Core.import_ref.fd1, %Core.import_ref.273), @T.as_type.as.Iterate.impl [concrete]
// CHECK:STDOUT:   %OverloadValueType.decl: type = class_decl @OverloadValueType [concrete = constants.%OverloadValueType] {} {}
// CHECK:STDOUT:   %Core.import_ref.b0d = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %field_decl.255: @DefaultOptionalStorage.%DefaultOptionalStorage.elem.2 (%DefaultOptionalStorage.elem.98d) = field_decl has_value, element1, %Core.import_ref.b0d in [concrete] {}
// CHECK:STDOUT:   %Core.import_ref.4cb = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %field_decl.649: @DefaultOptionalStorage.%DefaultOptionalStorage.elem.1 (%DefaultOptionalStorage.elem.b5a) = field_decl value, element0, %Core.import_ref.4cb in [concrete] {}
// CHECK:STDOUT:   %Range.decl: type = class_decl @Range [concrete = constants.%Range] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @TestIterate(%R.loc25_17.2: %Iterate_where.type.537) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Iterate.lookup_impl_witness: <witness> = lookup_impl_witness %R.loc25_17.1, @Iterate [symbolic = %Iterate.lookup_impl_witness (constants.%Iterate.lookup_impl_witness.7ec)]
// CHECK:STDOUT:   %Iterate.facet.loc27_37.5: %Iterate.type = facet_value %R.as_type.loc25_83.1, (%Iterate.lookup_impl_witness) [symbolic = %Iterate.facet.loc27_37.5 (constants.%Iterate.facet.1d6)]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.loc27_37.5) [symbolic = %Iterate.WithSelf.NewCursor.type (constants.%Iterate.WithSelf.NewCursor.type.b13)]
// CHECK:STDOUT:   %.loc27_37.26: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type, %Iterate.facet.loc27_37.5 [symbolic = %.loc27_37.26 (constants.%.204)]
// CHECK:STDOUT:   %impl.elem2.loc27_37.2: @TestIterate.%.loc27_37.26 (%.204) = impl_witness_access %Iterate.lookup_impl_witness, element2 [symbolic = %impl.elem2.loc27_37.2 (constants.%impl.elem2.098)]
// CHECK:STDOUT:   %specific_impl_fn.loc27_37.3: <specific function> = specific_impl_function %impl.elem2.loc27_37.2, @Iterate.WithSelf.NewCursor(%Iterate.facet.loc27_37.5) [symbolic = %specific_impl_fn.loc27_37.3 (constants.%specific_impl_fn.389)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.loc27_37.5) [symbolic = %Iterate.WithSelf.Next.type (constants.%Iterate.WithSelf.Next.type.a34)]
// CHECK:STDOUT:   %.loc27_37.27: type = fn_type_with_self_type %Iterate.WithSelf.Next.type, %Iterate.facet.loc27_37.5 [symbolic = %.loc27_37.27 (constants.%.5cd)]
// CHECK:STDOUT:   %impl.elem3.loc27_37.2: @TestIterate.%.loc27_37.27 (%.5cd) = impl_witness_access %Iterate.lookup_impl_witness, element3 [symbolic = %impl.elem3.loc27_37.2 (constants.%impl.elem3.815)]
// CHECK:STDOUT:   %specific_impl_fn.loc27_37.4: <specific function> = specific_impl_function %impl.elem3.loc27_37.2, @Iterate.WithSelf.Next(%Iterate.facet.loc27_37.5) [symbolic = %specific_impl_fn.loc27_37.4 (constants.%specific_impl_fn.0aa)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%r.param: ref @TestIterate.%R.as_type.loc25_83.1 (%R.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %_.patt: %pattern_type.804 = value_binding_pattern _ [concrete = constants.%_.patt.a10]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %r.ref: ref @TestIterate.%R.as_type.loc25_83.1 (%R.as_type) = name_ref r, %r
// CHECK:STDOUT:     %impl.elem2.loc27_37.1: @TestIterate.%.loc27_37.26 (%.204) = impl_witness_access constants.%Iterate.lookup_impl_witness.7ec, element2 [symbolic = %impl.elem2.loc27_37.2 (constants.%impl.elem2.098)]
// CHECK:STDOUT:     %bound_method.loc27_37.1: <bound method> = bound_method %r.ref, %impl.elem2.loc27_37.1
// CHECK:STDOUT:     %Iterate.facet.loc27_37.1: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.7ec) [symbolic = %Iterate.facet.loc27_37.5 (constants.%Iterate.facet.1d6)]
// CHECK:STDOUT:     %.loc27_37.1: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc27_37.1 [symbolic = %Iterate.facet.loc27_37.5 (constants.%Iterate.facet.1d6)]
// CHECK:STDOUT:     %Iterate.facet.loc27_37.2: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.7ec) [symbolic = %Iterate.facet.loc27_37.5 (constants.%Iterate.facet.1d6)]
// CHECK:STDOUT:     %.loc27_37.2: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc27_37.2 [symbolic = %Iterate.facet.loc27_37.5 (constants.%Iterate.facet.1d6)]
// CHECK:STDOUT:     %specific_impl_fn.loc27_37.1: <specific function> = specific_impl_function %impl.elem2.loc27_37.1, @Iterate.WithSelf.NewCursor(constants.%Iterate.facet.1d6) [symbolic = %specific_impl_fn.loc27_37.3 (constants.%specific_impl_fn.389)]
// CHECK:STDOUT:     %bound_method.loc27_37.2: <bound method> = bound_method %r.ref, %specific_impl_fn.loc27_37.1
// CHECK:STDOUT:     %var: ref @TestIterate.%as_type (%as_type.5f7) = var_storage invalid
// CHECK:STDOUT:     %.loc27_36.1: @TestIterate.%R.as_type.loc25_83.1 (%R.as_type) = acquire_value %r.ref
// CHECK:STDOUT:     %Iterate.WithSelf.NewCursor.call: init @TestIterate.%as_type (%as_type.5f7) to %var = call %bound_method.loc27_37.2(%.loc27_36.1)
// CHECK:STDOUT:     assign %var, %Iterate.WithSelf.NewCursor.call
// CHECK:STDOUT:     br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.next:
// CHECK:STDOUT:     %impl.elem3.loc27_37.1: @TestIterate.%.loc27_37.27 (%.5cd) = impl_witness_access constants.%Iterate.lookup_impl_witness.7ec, element3 [symbolic = %impl.elem3.loc27_37.2 (constants.%impl.elem3.815)]
// CHECK:STDOUT:     %bound_method.loc27_37.3: <bound method> = bound_method %r.ref, %impl.elem3.loc27_37.1
// CHECK:STDOUT:     %Iterate.facet.loc27_37.3: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.7ec) [symbolic = %Iterate.facet.loc27_37.5 (constants.%Iterate.facet.1d6)]
// CHECK:STDOUT:     %.loc27_37.3: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc27_37.3 [symbolic = %Iterate.facet.loc27_37.5 (constants.%Iterate.facet.1d6)]
// CHECK:STDOUT:     %Iterate.facet.loc27_37.4: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.7ec) [symbolic = %Iterate.facet.loc27_37.5 (constants.%Iterate.facet.1d6)]
// CHECK:STDOUT:     %.loc27_37.4: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc27_37.4 [symbolic = %Iterate.facet.loc27_37.5 (constants.%Iterate.facet.1d6)]
// CHECK:STDOUT:     %.loc27_37.5: %facet_type.8a3 = converted constants.%as_type.5f7, constants.%impl.elem1.8d0 [symbolic = %impl.elem1 (constants.%impl.elem1.8d0)]
// CHECK:STDOUT:     %.loc27_37.6: %facet_type.8a3 = converted constants.%as_type.5f7, constants.%impl.elem1.8d0 [symbolic = %impl.elem1 (constants.%impl.elem1.8d0)]
// CHECK:STDOUT:     %specific_impl_fn.loc27_37.2: <specific function> = specific_impl_function %impl.elem3.loc27_37.1, @Iterate.WithSelf.Next(constants.%Iterate.facet.1d6) [symbolic = %specific_impl_fn.loc27_37.4 (constants.%specific_impl_fn.0aa)]
// CHECK:STDOUT:     %bound_method.loc27_37.4: <bound method> = bound_method %r.ref, %specific_impl_fn.loc27_37.2
// CHECK:STDOUT:     %.loc27_37.7: ref %Optional.328 = temporary_storage
// CHECK:STDOUT:     %.loc27_36.2: @TestIterate.%R.as_type.loc25_83.1 (%R.as_type) = acquire_value %r.ref
// CHECK:STDOUT:     %Iterate.WithSelf.Next.call: init %Optional.328 to %.loc27_37.7 = call %bound_method.loc27_37.4(%.loc27_36.2, %var)
// CHECK:STDOUT:     %.loc27_37.8: ref %Optional.328 = temporary %.loc27_37.7, %Iterate.WithSelf.Next.call
// CHECK:STDOUT:     %.loc27_37.9: %Optional.HasValue.type.e8d = specific_constant imports.%Core.import_ref.bb6, @Optional(constants.%OptionalStorage.facet.2bc) [concrete = constants.%Optional.HasValue.4bf]
// CHECK:STDOUT:     %HasValue.ref: %Optional.HasValue.type.e8d = name_ref HasValue, %.loc27_37.9 [concrete = constants.%Optional.HasValue.4bf]
// CHECK:STDOUT:     %Optional.HasValue.bound: <bound method> = bound_method %.loc27_37.8, %HasValue.ref
// CHECK:STDOUT:     %Optional.HasValue.specific_fn: <specific function> = specific_function %HasValue.ref, @Optional.HasValue(constants.%OptionalStorage.facet.2bc) [concrete = constants.%Optional.HasValue.specific_fn.555]
// CHECK:STDOUT:     %bound_method.loc27_37.5: <bound method> = bound_method %.loc27_37.8, %Optional.HasValue.specific_fn
// CHECK:STDOUT:     %.loc27_37.10: %Optional.328 = acquire_value %.loc27_37.8
// CHECK:STDOUT:     %Optional.HasValue.call: init bool = call %bound_method.loc27_37.5(%.loc27_37.10)
// CHECK:STDOUT:     %.loc27_37.11: bool = value_of_initializer %Optional.HasValue.call
// CHECK:STDOUT:     %.loc27_37.12: bool = converted %Optional.HasValue.call, %.loc27_37.11
// CHECK:STDOUT:     if %.loc27_37.12 br !for.body else br !for.done
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.body:
// CHECK:STDOUT:     %.loc27_37.13: %Optional.Get.type.b0b = specific_constant imports.%Core.import_ref.dab, @Optional(constants.%OptionalStorage.facet.2bc) [concrete = constants.%Optional.Get.30b]
// CHECK:STDOUT:     %Get.ref: %Optional.Get.type.b0b = name_ref Get, %.loc27_37.13 [concrete = constants.%Optional.Get.30b]
// CHECK:STDOUT:     %Optional.Get.bound: <bound method> = bound_method %.loc27_37.8, %Get.ref
// CHECK:STDOUT:     %Optional.Get.specific_fn: <specific function> = specific_function %Get.ref, @Optional.Get(constants.%OptionalStorage.facet.2bc) [concrete = constants.%Optional.Get.specific_fn.519]
// CHECK:STDOUT:     %bound_method.loc27_37.6: <bound method> = bound_method %.loc27_37.8, %Optional.Get.specific_fn
// CHECK:STDOUT:     %.loc27_37.14: ref %OverloadValueType = temporary_storage
// CHECK:STDOUT:     %.loc27_37.15: %Optional.328 = acquire_value %.loc27_37.8
// CHECK:STDOUT:     %Optional.Get.call: init %OverloadValueType to %.loc27_37.14 = call %bound_method.loc27_37.6(%.loc27_37.15)
// CHECK:STDOUT:     %.loc27_37.16: ref %OverloadValueType = temporary %.loc27_37.14, %Optional.Get.call
// CHECK:STDOUT:     %.loc27_37.17: %OverloadValueType = acquire_value %.loc27_37.16
// CHECK:STDOUT:     %.loc27_14: type = splice_block %OverloadValueType.ref.loc27 [concrete = constants.%OverloadValueType] {
// CHECK:STDOUT:       %Cpp.ref.loc27: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:       %OverloadValueType.ref.loc27: type = name_ref OverloadValueType, imports.%OverloadValueType.decl [concrete = constants.%OverloadValueType]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %_: %OverloadValueType = wrapper_binding _, %.loc27_37.17
// CHECK:STDOUT:     %facet_value.loc27_37.1: %facet_type.8a3 = facet_value constants.%OverloadValueType, (constants.%custom_witness.5760f4.1, constants.%custom_witness.2cb555.2) [concrete = constants.%facet_value.a01]
// CHECK:STDOUT:     %.loc27_37.18: %facet_type.8a3 = converted constants.%OverloadValueType, %facet_value.loc27_37.1 [concrete = constants.%facet_value.a01]
// CHECK:STDOUT:     %facet_value.loc27_37.2: %facet_type.8a3 = facet_value constants.%OverloadValueType, (constants.%custom_witness.5760f4.1, constants.%custom_witness.2cb555.2) [concrete = constants.%facet_value.a01]
// CHECK:STDOUT:     %.loc27_37.19: %facet_type.8a3 = converted constants.%OverloadValueType, %facet_value.loc27_37.2 [concrete = constants.%facet_value.a01]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %SelfDestruct.call.loc27_37.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc27_37.16)
// CHECK:STDOUT:     %facet_value.loc27_37.3: %facet_type.8a3 = facet_value constants.%Optional.328, (constants.%custom_witness.b80ef6.7, constants.%custom_witness.2cb555.8) [concrete = constants.%facet_value.4c2]
// CHECK:STDOUT:     %.loc27_37.20: %facet_type.8a3 = converted constants.%Optional.328, %facet_value.loc27_37.3 [concrete = constants.%facet_value.4c2]
// CHECK:STDOUT:     %facet_value.loc27_37.4: %facet_type.8a3 = facet_value constants.%Optional.328, (constants.%custom_witness.b80ef6.7, constants.%custom_witness.2cb555.8) [concrete = constants.%facet_value.4c2]
// CHECK:STDOUT:     %.loc27_37.21: %facet_type.8a3 = converted constants.%Optional.328, %facet_value.loc27_37.4 [concrete = constants.%facet_value.4c2]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %SelfDestruct.call.loc27_37.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc27_37.8)
// CHECK:STDOUT:     br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.done:
// CHECK:STDOUT:     %facet_value.loc27_37.5: %facet_type.8a3 = facet_value constants.%Optional.328, (constants.%custom_witness.b80ef6.7, constants.%custom_witness.2cb555.8) [concrete = constants.%facet_value.4c2]
// CHECK:STDOUT:     %.loc27_37.22: %facet_type.8a3 = converted constants.%Optional.328, %facet_value.loc27_37.5 [concrete = constants.%facet_value.4c2]
// CHECK:STDOUT:     %facet_value.loc27_37.6: %facet_type.8a3 = facet_value constants.%Optional.328, (constants.%custom_witness.b80ef6.7, constants.%custom_witness.2cb555.8) [concrete = constants.%facet_value.4c2]
// CHECK:STDOUT:     %.loc27_37.23: %facet_type.8a3 = converted constants.%Optional.328, %facet_value.loc27_37.6 [concrete = constants.%facet_value.4c2]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %SelfDestruct.call.loc27_37.3: init %empty_tuple.type = call %SelfDestruct.specific_fn.3(%.loc27_37.8)
// CHECK:STDOUT:     %.loc27_37.24: %facet_type.8a3 = converted constants.%as_type.5f7, constants.%impl.elem1.8d0 [symbolic = %impl.elem1 (constants.%impl.elem1.8d0)]
// CHECK:STDOUT:     %.loc27_37.25: %facet_type.8a3 = converted constants.%as_type.5f7, constants.%impl.elem1.8d0 [symbolic = %impl.elem1 (constants.%impl.elem1.8d0)]
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
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc27_37.2(%self.param: ref %MaybeUnformed.d03) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc27_37.3(%self.param: ref bool) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc27_37.4(%self.param: ref %struct_type.value.has_value.3d3) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc27_37.1: ref bool = struct_access %self.param, element1
// CHECK:STDOUT:   %facet_value.loc27_37.1: %facet_type.8a3 = facet_value bool, (constants.%custom_witness.b80ef6.4, constants.%custom_witness.2cb555.3) [concrete = constants.%facet_value.ff1]
// CHECK:STDOUT:   %.loc27_37.2: %facet_type.8a3 = converted bool, %facet_value.loc27_37.1 [concrete = constants.%facet_value.ff1]
// CHECK:STDOUT:   %facet_value.loc27_37.2: %facet_type.8a3 = facet_value bool, (constants.%custom_witness.b80ef6.4, constants.%custom_witness.2cb555.3) [concrete = constants.%facet_value.ff1]
// CHECK:STDOUT:   %.loc27_37.3: %facet_type.8a3 = converted bool, %facet_value.loc27_37.2 [concrete = constants.%facet_value.ff1]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc27_37.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc27_37.1)
// CHECK:STDOUT:   %.loc27_37.4: ref %MaybeUnformed.d03 = struct_access %self.param, element0
// CHECK:STDOUT:   %facet_value.loc27_37.3: %facet_type.8a3 = facet_value constants.%MaybeUnformed.d03, (constants.%custom_witness.b80ef6.3, constants.%custom_witness.2cb555.5) [concrete = constants.%facet_value.bc3]
// CHECK:STDOUT:   %.loc27_37.5: %facet_type.8a3 = converted constants.%MaybeUnformed.d03, %facet_value.loc27_37.3 [concrete = constants.%facet_value.bc3]
// CHECK:STDOUT:   %facet_value.loc27_37.4: %facet_type.8a3 = facet_value constants.%MaybeUnformed.d03, (constants.%custom_witness.b80ef6.3, constants.%custom_witness.2cb555.5) [concrete = constants.%facet_value.bc3]
// CHECK:STDOUT:   %.loc27_37.6: %facet_type.8a3 = converted constants.%MaybeUnformed.d03, %facet_value.loc27_37.4 [concrete = constants.%facet_value.bc3]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc27_37.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc27_37.4)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyImpl.WithSelf.Op.loc27_37.1(%self.param: ref bool) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyImpl.WithSelf.Op.loc27_37.2(%self.param: ref %.8a2) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyImpl.WithSelf.Op.loc27_37.3(%self.param: ref %MaybeUnformed.d03) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc27_37.5(%self.param: ref %DefaultOptionalStorage.0cb) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %has_value.ref: %DefaultOptionalStorage.elem.9fe = name_ref has_value, imports.%Core.import_ref.1cb [concrete = imports.%field_decl.255]
// CHECK:STDOUT:   %.loc27_37.1: ref bool = class_element_access %self.param, element1
// CHECK:STDOUT:   %facet_value.loc27_37.1: %facet_type.8a3 = facet_value bool, (constants.%custom_witness.b80ef6.4, constants.%custom_witness.2cb555.3) [concrete = constants.%facet_value.ff1]
// CHECK:STDOUT:   %.loc27_37.2: %facet_type.8a3 = converted bool, %facet_value.loc27_37.1 [concrete = constants.%facet_value.ff1]
// CHECK:STDOUT:   %facet_value.loc27_37.2: %facet_type.8a3 = facet_value bool, (constants.%custom_witness.b80ef6.4, constants.%custom_witness.2cb555.3) [concrete = constants.%facet_value.ff1]
// CHECK:STDOUT:   %.loc27_37.3: %facet_type.8a3 = converted bool, %facet_value.loc27_37.2 [concrete = constants.%facet_value.ff1]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc27_37.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc27_37.1)
// CHECK:STDOUT:   %value.ref: %DefaultOptionalStorage.elem.6a5 = name_ref value, imports.%Core.import_ref.1c5 [concrete = imports.%field_decl.649]
// CHECK:STDOUT:   %.loc27_37.4: ref %MaybeUnformed.d03 = class_element_access %self.param, element0
// CHECK:STDOUT:   %facet_value.loc27_37.3: %facet_type.8a3 = facet_value constants.%MaybeUnformed.d03, (constants.%custom_witness.b80ef6.3, constants.%custom_witness.2cb555.5) [concrete = constants.%facet_value.bc3]
// CHECK:STDOUT:   %.loc27_37.5: %facet_type.8a3 = converted constants.%MaybeUnformed.d03, %facet_value.loc27_37.3 [concrete = constants.%facet_value.bc3]
// CHECK:STDOUT:   %facet_value.loc27_37.4: %facet_type.8a3 = facet_value constants.%MaybeUnformed.d03, (constants.%custom_witness.b80ef6.3, constants.%custom_witness.2cb555.5) [concrete = constants.%facet_value.bc3]
// CHECK:STDOUT:   %.loc27_37.6: %facet_type.8a3 = converted constants.%MaybeUnformed.d03, %facet_value.loc27_37.4 [concrete = constants.%facet_value.bc3]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc27_37.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc27_37.4)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc27_37.6(%self.param: ref %Optional.328) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyImpl.WithSelf.Op.loc27_37.4(%self.param: ref %struct_type.value.has_value.3d3) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyImpl.WithSelf.Op.loc27_37.5(%self.param: ref %DefaultOptionalStorage.0cb) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyImpl.WithSelf.Op.loc27_37.6(%self.param: ref %Optional.328) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestRangeFor(%r.param: ref %Range) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %_.patt: %pattern_type.804 = value_binding_pattern _ [concrete = constants.%_.patt.a10]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %r.ref: ref %Range = name_ref r, %r
// CHECK:STDOUT:   %impl.elem2: %.3d5 = impl_witness_access constants.%Iterate.impl_witness.3aa, element2 [concrete = constants.%T.as_type.as.Iterate.impl.NewCursor.880]
// CHECK:STDOUT:   %bound_method.loc37_37.1: <bound method> = bound_method %r.ref, %impl.elem2
// CHECK:STDOUT:   %facet_value.loc37_37.1: %CppRange_where.type.d682d6.1 = facet_value constants.%Range, (constants.%custom_witness.5760f4.1, constants.%custom_witness.2cb555.2, constants.%custom_witness.d0da43.1, constants.%custom_witness.090, constants.%custom_witness.51e, constants.%custom_witness.2cb555.9, constants.%custom_witness.6bd, constants.%custom_witness.716, constants.%custom_witness.617) [concrete = constants.%facet_value.de8]
// CHECK:STDOUT:   %.loc37_37.1: %CppRange_where.type.d682d6.1 = converted constants.%Range, %facet_value.loc37_37.1 [concrete = constants.%facet_value.de8]
// CHECK:STDOUT:   %facet_value.loc37_37.2: %CppRange_where.type.d682d6.1 = facet_value constants.%Range, (constants.%custom_witness.5760f4.1, constants.%custom_witness.2cb555.2, constants.%custom_witness.d0da43.1, constants.%custom_witness.090, constants.%custom_witness.51e, constants.%custom_witness.2cb555.9, constants.%custom_witness.6bd, constants.%custom_witness.716, constants.%custom_witness.617) [concrete = constants.%facet_value.de8]
// CHECK:STDOUT:   %.loc37_37.2: %CppRange_where.type.d682d6.1 = converted constants.%Range, %facet_value.loc37_37.2 [concrete = constants.%facet_value.de8]
// CHECK:STDOUT:   %specific_fn.loc37_37.1: <specific function> = specific_function %impl.elem2, @T.as_type.as.Iterate.impl.NewCursor(constants.%facet_value.de8) [concrete = constants.%T.as_type.as.Iterate.impl.NewCursor.specific_fn]
// CHECK:STDOUT:   %bound_method.loc37_37.2: <bound method> = bound_method %r.ref, %specific_fn.loc37_37.1
// CHECK:STDOUT:   %var: ref %tuple.type.9ef = var_storage invalid
// CHECK:STDOUT:   %.loc37_36.1: %Range = acquire_value %r.ref
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.call: init %tuple.type.9ef to %var = call %bound_method.loc37_37.2(%.loc37_36.1)
// CHECK:STDOUT:   assign %var, %T.as_type.as.Iterate.impl.NewCursor.call
// CHECK:STDOUT:   br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT: !for.next:
// CHECK:STDOUT:   %impl.elem3: %.569 = impl_witness_access constants.%Iterate.impl_witness.3aa, element3 [concrete = constants.%T.as_type.as.Iterate.impl.Next.1cb]
// CHECK:STDOUT:   %bound_method.loc37_37.3: <bound method> = bound_method %r.ref, %impl.elem3
// CHECK:STDOUT:   %facet_value.loc37_37.3: %CppRange_where.type.d682d6.1 = facet_value constants.%Range, (constants.%custom_witness.5760f4.1, constants.%custom_witness.2cb555.2, constants.%custom_witness.d0da43.1, constants.%custom_witness.090, constants.%custom_witness.51e, constants.%custom_witness.2cb555.9, constants.%custom_witness.6bd, constants.%custom_witness.716, constants.%custom_witness.617) [concrete = constants.%facet_value.de8]
// CHECK:STDOUT:   %.loc37_37.3: %CppRange_where.type.d682d6.1 = converted constants.%Range, %facet_value.loc37_37.3 [concrete = constants.%facet_value.de8]
// CHECK:STDOUT:   %facet_value.loc37_37.4: %CppRange_where.type.d682d6.1 = facet_value constants.%Range, (constants.%custom_witness.5760f4.1, constants.%custom_witness.2cb555.2, constants.%custom_witness.d0da43.1, constants.%custom_witness.090, constants.%custom_witness.51e, constants.%custom_witness.2cb555.9, constants.%custom_witness.6bd, constants.%custom_witness.716, constants.%custom_witness.617) [concrete = constants.%facet_value.de8]
// CHECK:STDOUT:   %.loc37_37.4: %CppRange_where.type.d682d6.1 = converted constants.%Range, %facet_value.loc37_37.4 [concrete = constants.%facet_value.de8]
// CHECK:STDOUT:   %specific_fn.loc37_37.2: <specific function> = specific_function %impl.elem3, @T.as_type.as.Iterate.impl.Next(constants.%facet_value.de8) [concrete = constants.%T.as_type.as.Iterate.impl.Next.specific_fn]
// CHECK:STDOUT:   %bound_method.loc37_37.4: <bound method> = bound_method %r.ref, %specific_fn.loc37_37.2
// CHECK:STDOUT:   %.loc37_37.5: ref %Optional.328 = temporary_storage
// CHECK:STDOUT:   %.loc37_36.2: %Range = acquire_value %r.ref
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.call: init %Optional.328 to %.loc37_37.5 = call %bound_method.loc37_37.4(%.loc37_36.2, %var)
// CHECK:STDOUT:   %.loc37_37.6: ref %Optional.328 = temporary %.loc37_37.5, %T.as_type.as.Iterate.impl.Next.call
// CHECK:STDOUT:   %.loc37_37.7: %Optional.HasValue.type.e8d = specific_constant imports.%Core.import_ref.bb6, @Optional(constants.%OptionalStorage.facet.2bc) [concrete = constants.%Optional.HasValue.4bf]
// CHECK:STDOUT:   %HasValue.ref: %Optional.HasValue.type.e8d = name_ref HasValue, %.loc37_37.7 [concrete = constants.%Optional.HasValue.4bf]
// CHECK:STDOUT:   %Optional.HasValue.bound: <bound method> = bound_method %.loc37_37.6, %HasValue.ref
// CHECK:STDOUT:   %Optional.HasValue.specific_fn: <specific function> = specific_function %HasValue.ref, @Optional.HasValue(constants.%OptionalStorage.facet.2bc) [concrete = constants.%Optional.HasValue.specific_fn.555]
// CHECK:STDOUT:   %bound_method.loc37_37.5: <bound method> = bound_method %.loc37_37.6, %Optional.HasValue.specific_fn
// CHECK:STDOUT:   %.loc37_37.8: %Optional.328 = acquire_value %.loc37_37.6
// CHECK:STDOUT:   %Optional.HasValue.call: init bool = call %bound_method.loc37_37.5(%.loc37_37.8)
// CHECK:STDOUT:   %.loc37_37.9: bool = value_of_initializer %Optional.HasValue.call
// CHECK:STDOUT:   %.loc37_37.10: bool = converted %Optional.HasValue.call, %.loc37_37.9
// CHECK:STDOUT:   if %.loc37_37.10 br !for.body else br !for.done
// CHECK:STDOUT:
// CHECK:STDOUT: !for.body:
// CHECK:STDOUT:   %.loc37_37.11: %Optional.Get.type.b0b = specific_constant imports.%Core.import_ref.dab, @Optional(constants.%OptionalStorage.facet.2bc) [concrete = constants.%Optional.Get.30b]
// CHECK:STDOUT:   %Get.ref: %Optional.Get.type.b0b = name_ref Get, %.loc37_37.11 [concrete = constants.%Optional.Get.30b]
// CHECK:STDOUT:   %Optional.Get.bound: <bound method> = bound_method %.loc37_37.6, %Get.ref
// CHECK:STDOUT:   %Optional.Get.specific_fn: <specific function> = specific_function %Get.ref, @Optional.Get(constants.%OptionalStorage.facet.2bc) [concrete = constants.%Optional.Get.specific_fn.519]
// CHECK:STDOUT:   %bound_method.loc37_37.6: <bound method> = bound_method %.loc37_37.6, %Optional.Get.specific_fn
// CHECK:STDOUT:   %.loc37_37.12: ref %OverloadValueType = temporary_storage
// CHECK:STDOUT:   %.loc37_37.13: %Optional.328 = acquire_value %.loc37_37.6
// CHECK:STDOUT:   %Optional.Get.call: init %OverloadValueType to %.loc37_37.12 = call %bound_method.loc37_37.6(%.loc37_37.13)
// CHECK:STDOUT:   %.loc37_37.14: ref %OverloadValueType = temporary %.loc37_37.12, %Optional.Get.call
// CHECK:STDOUT:   %.loc37_37.15: %OverloadValueType = acquire_value %.loc37_37.14
// CHECK:STDOUT:   %.loc37_14: type = splice_block %OverloadValueType.ref [concrete = constants.%OverloadValueType] {
// CHECK:STDOUT:     %Cpp.ref.loc37: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %OverloadValueType.ref: type = name_ref OverloadValueType, imports.%OverloadValueType.decl [concrete = constants.%OverloadValueType]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %_: %OverloadValueType = wrapper_binding _, %.loc37_37.15
// CHECK:STDOUT:   %facet_value.loc37_37.5: %facet_type.8a3 = facet_value constants.%OverloadValueType, (constants.%custom_witness.5760f4.1, constants.%custom_witness.2cb555.2) [concrete = constants.%facet_value.a01]
// CHECK:STDOUT:   %.loc37_37.16: %facet_type.8a3 = converted constants.%OverloadValueType, %facet_value.loc37_37.5 [concrete = constants.%facet_value.a01]
// CHECK:STDOUT:   %facet_value.loc37_37.6: %facet_type.8a3 = facet_value constants.%OverloadValueType, (constants.%custom_witness.5760f4.1, constants.%custom_witness.2cb555.2) [concrete = constants.%facet_value.a01]
// CHECK:STDOUT:   %.loc37_37.17: %facet_type.8a3 = converted constants.%OverloadValueType, %facet_value.loc37_37.6 [concrete = constants.%facet_value.a01]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc37_37.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc37_37.14)
// CHECK:STDOUT:   %facet_value.loc37_37.7: %facet_type.8a3 = facet_value constants.%Optional.328, (constants.%custom_witness.b80ef6.7, constants.%custom_witness.2cb555.8) [concrete = constants.%facet_value.4c2]
// CHECK:STDOUT:   %.loc37_37.18: %facet_type.8a3 = converted constants.%Optional.328, %facet_value.loc37_37.7 [concrete = constants.%facet_value.4c2]
// CHECK:STDOUT:   %facet_value.loc37_37.8: %facet_type.8a3 = facet_value constants.%Optional.328, (constants.%custom_witness.b80ef6.7, constants.%custom_witness.2cb555.8) [concrete = constants.%facet_value.4c2]
// CHECK:STDOUT:   %.loc37_37.19: %facet_type.8a3 = converted constants.%Optional.328, %facet_value.loc37_37.8 [concrete = constants.%facet_value.4c2]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc37_37.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc37_37.6)
// CHECK:STDOUT:   br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT: !for.done:
// CHECK:STDOUT:   %facet_value.loc37_37.9: %facet_type.8a3 = facet_value constants.%Optional.328, (constants.%custom_witness.b80ef6.7, constants.%custom_witness.2cb555.8) [concrete = constants.%facet_value.4c2]
// CHECK:STDOUT:   %.loc37_37.20: %facet_type.8a3 = converted constants.%Optional.328, %facet_value.loc37_37.9 [concrete = constants.%facet_value.4c2]
// CHECK:STDOUT:   %facet_value.loc37_37.10: %facet_type.8a3 = facet_value constants.%Optional.328, (constants.%custom_witness.b80ef6.7, constants.%custom_witness.2cb555.8) [concrete = constants.%facet_value.4c2]
// CHECK:STDOUT:   %.loc37_37.21: %facet_type.8a3 = converted constants.%Optional.328, %facet_value.loc37_37.10 [concrete = constants.%facet_value.4c2]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc37_37.3: init %empty_tuple.type = call %SelfDestruct.specific_fn.3(%.loc37_37.6)
// CHECK:STDOUT:   %facet_value.loc37_37.11: %facet_type.8a3 = facet_value constants.%tuple.type.9ef, (constants.%custom_witness.b80ef6.8, constants.%custom_witness.2cb555.10) [concrete = constants.%facet_value.ee1]
// CHECK:STDOUT:   %.loc37_37.22: %facet_type.8a3 = converted constants.%tuple.type.9ef, %facet_value.loc37_37.11 [concrete = constants.%facet_value.ee1]
// CHECK:STDOUT:   %facet_value.loc37_37.12: %facet_type.8a3 = facet_value constants.%tuple.type.9ef, (constants.%custom_witness.b80ef6.8, constants.%custom_witness.2cb555.10) [concrete = constants.%facet_value.ee1]
// CHECK:STDOUT:   %.loc37_37.23: %facet_type.8a3 = converted constants.%tuple.type.9ef, %facet_value.loc37_37.12 [concrete = constants.%facet_value.ee1]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc37_37.4: init %empty_tuple.type = call %SelfDestruct.specific_fn.4(%var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestIterate(constants.%R) {
// CHECK:STDOUT:   %R.patt.loc25_17.2 => constants.%R.patt
// CHECK:STDOUT:   %R.loc25_17.1 => constants.%R
// CHECK:STDOUT:   %R.as_type.loc25_83.1 => constants.%R.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.f1e
// CHECK:STDOUT:   %r.patt.loc25_81.2 => constants.%r.patt.b92
// CHECK:STDOUT:   %r.param_patt.loc25_76.2 => constants.%r.param_patt.1ca
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestIterate(constants.%facet_value.f01) {
// CHECK:STDOUT:   %R.patt.loc25_17.2 => constants.%R.patt
// CHECK:STDOUT:   %R.loc25_17.1 => constants.%facet_value.f01
// CHECK:STDOUT:   %R.as_type.loc25_83.1 => constants.%Range
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.23f
// CHECK:STDOUT:   %r.patt.loc25_81.2 => constants.%r.patt.b9f
// CHECK:STDOUT:   %r.param_patt.loc25_76.2 => constants.%r.param_patt.466
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc25 => constants.%complete_type.357
// CHECK:STDOUT:   %Iterate.lookup_impl_witness => constants.%Iterate.impl_witness.3aa
// CHECK:STDOUT:   %Iterate.facet.loc27_37.5 => constants.%Iterate.facet.b70
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type => constants.%Iterate.WithSelf.NewCursor.type.719
// CHECK:STDOUT:   %.loc27_37.26 => constants.%.3d5
// CHECK:STDOUT:   %impl.elem2.loc27_37.2 => constants.%T.as_type.as.Iterate.impl.NewCursor.880
// CHECK:STDOUT:   %specific_impl_fn.loc27_37.3 => constants.%T.as_type.as.Iterate.impl.NewCursor.specific_fn
// CHECK:STDOUT:   %impl.elem1 => constants.%facet_value.ee1
// CHECK:STDOUT:   %as_type => constants.%tuple.type.9ef
// CHECK:STDOUT:   %require_complete.1 => constants.%complete_type.e35
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type => constants.%Iterate.WithSelf.Next.type.da8
// CHECK:STDOUT:   %.loc27_37.27 => constants.%.569
// CHECK:STDOUT:   %impl.elem3.loc27_37.2 => constants.%T.as_type.as.Iterate.impl.Next.1cb
// CHECK:STDOUT:   %specific_impl_fn.loc27_37.4 => constants.%T.as_type.as.Iterate.impl.Next.specific_fn
// CHECK:STDOUT:   %SelfDestruct.specific_fn.5 => constants.%SelfDestruct.specific_fn.1d5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- adl_return_same_types.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %Iterate.type: type = facet_type <@Iterate> [concrete]
// CHECK:STDOUT:   %OptionalStorage.type: type = facet_type <@OptionalStorage> [concrete]
// CHECK:STDOUT:   %facet_type.d61: type = facet_type <@SubobjectDestroy & @DestroyImpl & @Copy> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %T.220: %OptionalStorage.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Optional.Get.type.bb9: type = fn_type @Optional.Get, @Optional(%T.220) [symbolic]
// CHECK:STDOUT:   %Optional.Get.fa1: %Optional.Get.type.bb9 = struct_value () [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.type.e9e: type = fn_type @Optional.HasValue, @Optional(%T.220) [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.6cd: %Optional.HasValue.type.e9e = struct_value () [symbolic]
// CHECK:STDOUT:   %facet_type.8a3: type = facet_type <@SubobjectDestroy & @DestroyImpl> [concrete]
// CHECK:STDOUT:   %T.9ab: %facet_type.d61 = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.19c: type = facet_access_type %T.9ab [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.lookup_impl_witness.855: <witness> = lookup_impl_witness %T.9ab, @SubobjectDestroy [symbolic]
// CHECK:STDOUT:   %DestroyImpl.lookup_impl_witness.ff9: <witness> = lookup_impl_witness %T.9ab, @DestroyImpl [symbolic]
// CHECK:STDOUT:   %facet_value.b35: %facet_type.8a3 = facet_value %T.as_type.19c, (%SubobjectDestroy.lookup_impl_witness.855, %DestroyImpl.lookup_impl_witness.ff9) [symbolic]
// CHECK:STDOUT:   %MaybeUnformed.2cf: type = class_type @MaybeUnformed, @MaybeUnformed(%facet_value.b35) [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.7db: type = class_type @DefaultOptionalStorage, @DefaultOptionalStorage(%T.9ab) [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.elem.98d: type = unbound_element_type %DefaultOptionalStorage.7db, bool [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.elem.b5a: type = unbound_element_type %DefaultOptionalStorage.7db, %MaybeUnformed.2cf [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Get.type.b32: type = fn_type @T.as_type.as.OptionalStorage.impl.Get, @T.as_type.as.OptionalStorage.impl(%T.9ab) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Get.8b0: %T.as_type.as.OptionalStorage.impl.Get.type.b32 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Has.type.8e5: type = fn_type @T.as_type.as.OptionalStorage.impl.Has, @T.as_type.as.OptionalStorage.impl(%T.9ab) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Has.2d5: %T.as_type.as.OptionalStorage.impl.Has.type.8e5 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Some.type.61d: type = fn_type @T.as_type.as.OptionalStorage.impl.Some, @T.as_type.as.OptionalStorage.impl(%T.9ab) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Some.ba7: %T.as_type.as.OptionalStorage.impl.Some.type.61d = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.None.type.a9f: type = fn_type @T.as_type.as.OptionalStorage.impl.None, @T.as_type.as.OptionalStorage.impl(%T.9ab) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.None.083: %T.as_type.as.OptionalStorage.impl.None.type.a9f = struct_value () [symbolic]
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
// CHECK:STDOUT:   %f64.dc1: type = class_type @Float, @Float(%int_64) [concrete]
// CHECK:STDOUT:   %pattern_type.fb7: type = pattern_type %f64.dc1 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.2: type = fn_type @SubobjectDestroy.WithSelf.Op.loc20 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.2: %SubobjectDestroy.WithSelf.Op.type.6870fd.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.2: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.2), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.type.b77747.2: type = fn_type @DestroyImpl.WithSelf.Op.loc20 [concrete]
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.25aa57.2: %DestroyImpl.WithSelf.Op.type.b77747.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.2cb555.2: <witness> = custom_witness (%DestroyImpl.WithSelf.Op.25aa57.2), @DestroyImpl [concrete]
// CHECK:STDOUT:   %Float.as.Copy.impl.Op.type.ff3: type = fn_type @Float.as.Copy.impl.Op, @Float.as.Copy.impl(%N.fe9) [symbolic]
// CHECK:STDOUT:   %Float.as.Copy.impl.Op.d93: %Float.as.Copy.impl.Op.type.ff3 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.0d6: <witness> = impl_witness imports.%Copy.impl_witness_table.8dd, @Float.as.Copy.impl(%int_64) [concrete]
// CHECK:STDOUT:   %facet_value.340: %facet_type.d61 = facet_value %f64.dc1, (%custom_witness.b80ef6.2, %custom_witness.2cb555.2, %Copy.impl_witness.0d6) [concrete]
// CHECK:STDOUT:   %Iterate.lookup_impl_witness.1d8: <witness> = lookup_impl_witness %.Self.500, @Iterate [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.8a4: %facet_type.d61 = impl_witness_access %Iterate.lookup_impl_witness.1d8, element0 [symbolic_self]
// CHECK:STDOUT:   %Iterate_where.type.7ce: type = facet_type <@Iterate where %impl.elem0.8a4 = %facet_value.340> [concrete]
// CHECK:STDOUT:   %pattern_type.cb4: type = pattern_type %Iterate_where.type.7ce [concrete]
// CHECK:STDOUT:   %R.patt: %pattern_type.cb4 = symbolic_binding_pattern R, 0 [symbolic]
// CHECK:STDOUT:   %R: %Iterate_where.type.7ce = symbolic_binding R, 0 [symbolic]
// CHECK:STDOUT:   %R.as_type: type = facet_access_type %R [symbolic]
// CHECK:STDOUT:   %pattern_type.df6: type = pattern_type %R.as_type [symbolic]
// CHECK:STDOUT:   %r.param_patt.7c5: %pattern_type.df6 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %r.patt.598: %pattern_type.df6 = wrapper_binding_pattern r, %r.param_patt.7c5 [symbolic]
// CHECK:STDOUT:   %i.patt: %pattern_type.fb7 = value_binding_pattern i [concrete]
// CHECK:STDOUT:   %Iterate.lookup_impl_witness.03a: <witness> = lookup_impl_witness %R, @Iterate [symbolic]
// CHECK:STDOUT:   %Iterate.facet.183: %Iterate.type = facet_value %R.as_type, (%Iterate.lookup_impl_witness.03a) [symbolic]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.9bc: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.183) [symbolic]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.ffd: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.183) [symbolic]
// CHECK:STDOUT:   %.40e: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.9bc, %Iterate.facet.183 [symbolic]
// CHECK:STDOUT:   %impl.elem2.0a0: %.40e = impl_witness_access %Iterate.lookup_impl_witness.03a, element2 [symbolic]
// CHECK:STDOUT:   %impl.elem1.277: %facet_type.8a3 = impl_witness_access %Iterate.lookup_impl_witness.03a, element1 [symbolic]
// CHECK:STDOUT:   %as_type.dff: type = facet_access_type %impl.elem1.277 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.929: <specific function> = specific_impl_function %impl.elem2.0a0, @Iterate.WithSelf.NewCursor(%Iterate.facet.183) [symbolic]
// CHECK:STDOUT:   %.b8e: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.ffd, %Iterate.facet.183 [symbolic]
// CHECK:STDOUT:   %impl.elem3.64e: %.b8e = impl_witness_access %Iterate.lookup_impl_witness.03a, element3 [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.877: type = class_type @DefaultOptionalStorage, @DefaultOptionalStorage(%facet_value.340) [concrete]
// CHECK:STDOUT:   %OptionalStorage.impl_witness.ad2: <witness> = impl_witness imports.%OptionalStorage.impl_witness_table.968, @T.as_type.as.OptionalStorage.impl(%facet_value.340) [concrete]
// CHECK:STDOUT:   %OptionalStorage.facet.424: %OptionalStorage.type = facet_value %f64.dc1, (%OptionalStorage.impl_witness.ad2) [concrete]
// CHECK:STDOUT:   %Optional.49a: type = class_type @Optional, @Optional(%OptionalStorage.facet.424) [concrete]
// CHECK:STDOUT:   %specific_impl_fn.520: <specific function> = specific_impl_function %impl.elem3.64e, @Iterate.WithSelf.Next(%Iterate.facet.183) [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.type.79d: type = fn_type @Optional.HasValue, @Optional(%OptionalStorage.facet.424) [concrete]
// CHECK:STDOUT:   %Optional.HasValue.718: %Optional.HasValue.type.79d = struct_value () [concrete]
// CHECK:STDOUT:   %Optional.Get.type.3ba: type = fn_type @Optional.Get, @Optional(%OptionalStorage.facet.424) [concrete]
// CHECK:STDOUT:   %Optional.Get.a7f: %Optional.Get.type.3ba = struct_value () [concrete]
// CHECK:STDOUT:   %facet_value.391: %facet_type.8a3 = facet_value %f64.dc1, (%custom_witness.b80ef6.2, %custom_witness.2cb555.2) [concrete]
// CHECK:STDOUT:   %MaybeUnformed.15c: type = class_type @MaybeUnformed, @MaybeUnformed(%facet_value.391) [concrete]
// CHECK:STDOUT:   %.cc0: type = maybe_unformed_type %f64.dc1 [concrete]
// CHECK:STDOUT:   %DefaultOptionalStorage.elem.e01: type = unbound_element_type %DefaultOptionalStorage.877, %MaybeUnformed.15c [concrete]
// CHECK:STDOUT:   %DefaultOptionalStorage.elem.718: type = unbound_element_type %DefaultOptionalStorage.877, bool [concrete]
// CHECK:STDOUT:   %struct_type.value.has_value.12d: type = struct_type {.value: %MaybeUnformed.15c, .has_value: bool} [concrete]
// CHECK:STDOUT:   %Optional.HasValue.specific_fn.409: <specific function> = specific_function %Optional.HasValue.718, @Optional.HasValue(%OptionalStorage.facet.424) [concrete]
// CHECK:STDOUT:   %Optional.Get.specific_fn.f9b: <specific function> = specific_function %Optional.Get.a7f, @Optional.Get(%OptionalStorage.facet.424) [concrete]
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
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.4: type = fn_type @SubobjectDestroy.WithSelf.Op.loc24_19.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.4: %SubobjectDestroy.WithSelf.Op.type.6870fd.4 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.4: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.4), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.5: type = fn_type @SubobjectDestroy.WithSelf.Op.loc24_19.3 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.5: %SubobjectDestroy.WithSelf.Op.type.6870fd.5 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.5: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.5), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.type.b77747.3: type = fn_type @DestroyImpl.WithSelf.Op.loc24_19.1 [concrete]
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.25aa57.3: %DestroyImpl.WithSelf.Op.type.b77747.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.2cb555.3: <witness> = custom_witness (%DestroyImpl.WithSelf.Op.25aa57.3), @DestroyImpl [concrete]
// CHECK:STDOUT:   %facet_value.ff1: %facet_type.8a3 = facet_value bool, (%custom_witness.b80ef6.5, %custom_witness.2cb555.3) [concrete]
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.type.b77747.5: type = fn_type @DestroyImpl.WithSelf.Op.loc24_19.3 [concrete]
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.25aa57.5: %DestroyImpl.WithSelf.Op.type.b77747.5 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.2cb555.5: <witness> = custom_witness (%DestroyImpl.WithSelf.Op.25aa57.5), @DestroyImpl [concrete]
// CHECK:STDOUT:   %facet_value.f6f: %facet_type.8a3 = facet_value %MaybeUnformed.15c, (%custom_witness.b80ef6.4, %custom_witness.2cb555.5) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.8: type = fn_type @SubobjectDestroy.WithSelf.Op.loc24_19.6 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.8: %SubobjectDestroy.WithSelf.Op.type.6870fd.8 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.8: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.8), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.type.b77747.8: type = fn_type @DestroyImpl.WithSelf.Op.loc24_19.6 [concrete]
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.25aa57.8: %DestroyImpl.WithSelf.Op.type.b77747.8 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.2cb555.8: <witness> = custom_witness (%DestroyImpl.WithSelf.Op.25aa57.8), @DestroyImpl [concrete]
// CHECK:STDOUT:   %facet_value.153: %facet_type.8a3 = facet_value %Optional.49a, (%custom_witness.b80ef6.8, %custom_witness.2cb555.8) [concrete]
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
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.type.b77747.10: type = fn_type @DestroyImpl.WithSelf.Op.4 [concrete]
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.25aa57.10: %DestroyImpl.WithSelf.Op.type.b77747.10 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.2cb555.9: <witness> = custom_witness (%DestroyImpl.WithSelf.Op.25aa57.10), @DestroyImpl [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.ac7e09.2: type = fn_type @Iterator.Op.2 [concrete]
// CHECK:STDOUT:   %Iterator.Op.583fda.2: %Iterator.Op.type.ac7e09.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.1fc: <witness> = custom_witness (%Iterator.Op.583fda.2), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.ac7e09.3: type = fn_type @Iterator.Op.3 [concrete]
// CHECK:STDOUT:   %Iterator.Op.583fda.3: %Iterator.Op.type.ac7e09.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.393: <witness> = custom_witness (%Iterator.Op.583fda.3), @Inc [concrete]
// CHECK:STDOUT:   %Equal.type: type = fn_type @Equal [concrete]
// CHECK:STDOUT:   %Equal: %Equal.type = struct_value () [concrete]
// CHECK:STDOUT:   %NotEqual.type: type = fn_type @NotEqual [concrete]
// CHECK:STDOUT:   %NotEqual: %NotEqual.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.821: <witness> = custom_witness (%Equal, %NotEqual), @EqWith, @EqWith(%Iterator) [concrete]
// CHECK:STDOUT:   %facet_value.d36: %CppRange_where.type.d682d6.1 = facet_value %ConstRange, (%custom_witness.b80ef6.2, %custom_witness.2cb555.2, %Copy.impl_witness.0d6, %custom_witness.5d5, %custom_witness.1fc, %custom_witness.2cb555.9, %custom_witness.994, %custom_witness.393, %custom_witness.821) [concrete]
// CHECK:STDOUT:   %tuple.type.005: type = tuple_type (%Iterator, %Iterator) [concrete]
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.type.b77747.11: type = fn_type @DestroyImpl.WithSelf.Op.5 [concrete]
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.25aa57.11: %DestroyImpl.WithSelf.Op.type.b77747.11 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.2cb555.10: <witness> = custom_witness (%DestroyImpl.WithSelf.Op.25aa57.11), @DestroyImpl [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.9: type = fn_type @SubobjectDestroy.WithSelf.Op.3 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.9: %SubobjectDestroy.WithSelf.Op.type.6870fd.9 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.9: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.9), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %facet_value.54b: %facet_type.8a3 = facet_value %tuple.type.005, (%custom_witness.b80ef6.9, %custom_witness.2cb555.10) [concrete]
// CHECK:STDOUT:   %Iterate.impl_witness.17d: <witness> = impl_witness imports.%Iterate.impl_witness_table.121, @T.as_type.as.Iterate.impl(%facet_value.d36) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.type.2f3: type = fn_type @T.as_type.as.Iterate.impl.NewCursor, @T.as_type.as.Iterate.impl(%facet_value.d36) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.1ac: %T.as_type.as.Iterate.impl.NewCursor.type.2f3 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.type.d61: type = fn_type @T.as_type.as.Iterate.impl.Next, @T.as_type.as.Iterate.impl(%facet_value.d36) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.b00: %T.as_type.as.Iterate.impl.Next.type.d61 = struct_value () [concrete]
// CHECK:STDOUT:   %Iterate.facet.3bd: %Iterate.type = facet_value %ConstRange, (%Iterate.impl_witness.17d) [concrete]
// CHECK:STDOUT:   %facet_value.7c0: %Iterate_where.type.7ce = facet_value %ConstRange, (%Iterate.impl_witness.17d) [concrete]
// CHECK:STDOUT:   %r.param_patt.a73: %pattern_type.025 = value_param_pattern [concrete]
// CHECK:STDOUT:   %r.patt.0b5: %pattern_type.025 = wrapper_binding_pattern r, %r.param_patt.a73 [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.bf3: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.3bd) [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.3a6: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.3bd) [concrete]
// CHECK:STDOUT:   %.7b6: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.bf3, %Iterate.facet.3bd [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.specific_fn: <specific function> = specific_function %T.as_type.as.Iterate.impl.NewCursor.1ac, @T.as_type.as.Iterate.impl.NewCursor(%facet_value.d36) [concrete]
// CHECK:STDOUT:   %.4e6: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.3a6, %Iterate.facet.3bd [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.specific_fn: <specific function> = specific_function %T.as_type.as.Iterate.impl.Next.b00, @T.as_type.as.Iterate.impl.Next(%facet_value.d36) [concrete]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.770: <specific function> = specific_function %SelfDestruct, @SelfDestruct(%facet_value.54b) [concrete]
// CHECK:STDOUT:   %complete_type.320: <witness> = complete_type_witness %tuple.type.005 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.bb6: @Optional.%Optional.HasValue.type (%Optional.HasValue.type.e9e) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.HasValue (constants.%Optional.HasValue.6cd)]
// CHECK:STDOUT:   %Core.import_ref.dab: @Optional.%Optional.Get.type (%Optional.Get.type.bb9) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.Get (constants.%Optional.Get.fa1)]
// CHECK:STDOUT:   %Core.import_ref.38a: type = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%DefaultOptionalStorage (constants.%DefaultOptionalStorage.7db)]
// CHECK:STDOUT:   %Core.import_ref.61e: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.None.type (%T.as_type.as.OptionalStorage.impl.None.type.a9f) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.None (constants.%T.as_type.as.OptionalStorage.impl.None.083)]
// CHECK:STDOUT:   %Core.import_ref.ca1: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Some.type (%T.as_type.as.OptionalStorage.impl.Some.type.61d) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Some (constants.%T.as_type.as.OptionalStorage.impl.Some.ba7)]
// CHECK:STDOUT:   %Core.import_ref.6ba4: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Has.type (%T.as_type.as.OptionalStorage.impl.Has.type.8e5) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Has (constants.%T.as_type.as.OptionalStorage.impl.Has.2d5)]
// CHECK:STDOUT:   %Core.import_ref.73d: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Get.type (%T.as_type.as.OptionalStorage.impl.Get.type.b32) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Get (constants.%T.as_type.as.OptionalStorage.impl.Get.8b0)]
// CHECK:STDOUT:   %Core.import_ref.778 = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], unloaded
// CHECK:STDOUT:   %OptionalStorage.impl_witness_table.968 = impl_witness_table (%Core.import_ref.38a, %Core.import_ref.61e, %Core.import_ref.ca1, %Core.import_ref.6ba4, %Core.import_ref.73d, %Core.import_ref.778), @T.as_type.as.OptionalStorage.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.1c5: @DefaultOptionalStorage.%DefaultOptionalStorage.elem.1 (%DefaultOptionalStorage.elem.b5a) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [concrete = %field_decl.649]
// CHECK:STDOUT:   %Core.import_ref.1cb: @DefaultOptionalStorage.%DefaultOptionalStorage.elem.2 (%DefaultOptionalStorage.elem.98d) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [concrete = %field_decl.255]
// CHECK:STDOUT:   %Core.import_ref.d6c: %facet_type.d61 = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.Iterate.impl.%facet_value.1 (constants.%facet_value.9da)]
// CHECK:STDOUT:   %Core.import_ref.2ac: %facet_type.8a3 = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.Iterate.impl.%facet_value.2 (constants.%facet_value.bf6)]
// CHECK:STDOUT:   %Core.import_ref.fd1a: @T.as_type.as.Iterate.impl.%T.as_type.as.Iterate.impl.NewCursor.type (%T.as_type.as.Iterate.impl.NewCursor.type.953) = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.Iterate.impl.%T.as_type.as.Iterate.impl.NewCursor (constants.%T.as_type.as.Iterate.impl.NewCursor.544)]
// CHECK:STDOUT:   %Core.import_ref.273: @T.as_type.as.Iterate.impl.%T.as_type.as.Iterate.impl.Next.type (%T.as_type.as.Iterate.impl.Next.type.b36) = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.Iterate.impl.%T.as_type.as.Iterate.impl.Next (constants.%T.as_type.as.Iterate.impl.Next.d6b)]
// CHECK:STDOUT:   %Iterate.impl_witness_table.121 = impl_witness_table (%Core.import_ref.d6c, %Core.import_ref.2ac, %Core.import_ref.fd1a, %Core.import_ref.273), @T.as_type.as.Iterate.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.e69: @Float.as.Copy.impl.%Float.as.Copy.impl.Op.type (%Float.as.Copy.impl.Op.type.ff3) = import_ref Core//prelude/types/float, loc{{\d+_\d+}}, loaded [symbolic = @Float.as.Copy.impl.%Float.as.Copy.impl.Op (constants.%Float.as.Copy.impl.Op.d93)]
// CHECK:STDOUT:   %Copy.impl_witness_table.8dd = impl_witness_table (%Core.import_ref.e69), @Float.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.cb3: @Float.as.AddAssignWith.impl.%Float.as.AddAssignWith.impl.Op.type (%Float.as.AddAssignWith.impl.Op.type.fdd) = import_ref Core//prelude/types/float, loc{{\d+_\d+}}, loaded [symbolic = @Float.as.AddAssignWith.impl.%Float.as.AddAssignWith.impl.Op (constants.%Float.as.AddAssignWith.impl.Op.430)]
// CHECK:STDOUT:   %AddAssignWith.impl_witness_table = impl_witness_table (%Core.import_ref.cb3), @Float.as.AddAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.b0d = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %field_decl.255: @DefaultOptionalStorage.%DefaultOptionalStorage.elem.2 (%DefaultOptionalStorage.elem.98d) = field_decl has_value, element1, %Core.import_ref.b0d in [concrete] {}
// CHECK:STDOUT:   %Core.import_ref.4cb = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %field_decl.649: @DefaultOptionalStorage.%DefaultOptionalStorage.elem.1 (%DefaultOptionalStorage.elem.b5a) = field_decl value, element0, %Core.import_ref.4cb in [concrete] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @TestIterate(%R.loc20_17.2: %Iterate_where.type.7ce) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Iterate.lookup_impl_witness: <witness> = lookup_impl_witness %R.loc20_17.1, @Iterate [symbolic = %Iterate.lookup_impl_witness (constants.%Iterate.lookup_impl_witness.03a)]
// CHECK:STDOUT:   %Iterate.facet.loc24_19.5: %Iterate.type = facet_value %R.as_type.loc20_61.1, (%Iterate.lookup_impl_witness) [symbolic = %Iterate.facet.loc24_19.5 (constants.%Iterate.facet.183)]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.loc24_19.5) [symbolic = %Iterate.WithSelf.NewCursor.type (constants.%Iterate.WithSelf.NewCursor.type.9bc)]
// CHECK:STDOUT:   %.loc24_19.23: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type, %Iterate.facet.loc24_19.5 [symbolic = %.loc24_19.23 (constants.%.40e)]
// CHECK:STDOUT:   %impl.elem2.loc24_19.2: @TestIterate.%.loc24_19.23 (%.40e) = impl_witness_access %Iterate.lookup_impl_witness, element2 [symbolic = %impl.elem2.loc24_19.2 (constants.%impl.elem2.0a0)]
// CHECK:STDOUT:   %specific_impl_fn.loc24_19.3: <specific function> = specific_impl_function %impl.elem2.loc24_19.2, @Iterate.WithSelf.NewCursor(%Iterate.facet.loc24_19.5) [symbolic = %specific_impl_fn.loc24_19.3 (constants.%specific_impl_fn.929)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.loc24_19.5) [symbolic = %Iterate.WithSelf.Next.type (constants.%Iterate.WithSelf.Next.type.ffd)]
// CHECK:STDOUT:   %.loc24_19.24: type = fn_type_with_self_type %Iterate.WithSelf.Next.type, %Iterate.facet.loc24_19.5 [symbolic = %.loc24_19.24 (constants.%.b8e)]
// CHECK:STDOUT:   %impl.elem3.loc24_19.2: @TestIterate.%.loc24_19.24 (%.b8e) = impl_witness_access %Iterate.lookup_impl_witness, element3 [symbolic = %impl.elem3.loc24_19.2 (constants.%impl.elem3.64e)]
// CHECK:STDOUT:   %specific_impl_fn.loc24_19.4: <specific function> = specific_impl_function %impl.elem3.loc24_19.2, @Iterate.WithSelf.Next(%Iterate.facet.loc24_19.5) [symbolic = %specific_impl_fn.loc24_19.4 (constants.%specific_impl_fn.520)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%r.param: @TestIterate.%R.as_type.loc20_61.1 (%R.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %i.patt: %pattern_type.fb7 = value_binding_pattern i [concrete = constants.%i.patt]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %r.ref: @TestIterate.%R.as_type.loc20_61.1 (%R.as_type) = name_ref r, %r
// CHECK:STDOUT:     %impl.elem2.loc24_19.1: @TestIterate.%.loc24_19.23 (%.40e) = impl_witness_access constants.%Iterate.lookup_impl_witness.03a, element2 [symbolic = %impl.elem2.loc24_19.2 (constants.%impl.elem2.0a0)]
// CHECK:STDOUT:     %bound_method.loc24_19.1: <bound method> = bound_method %r.ref, %impl.elem2.loc24_19.1
// CHECK:STDOUT:     %Iterate.facet.loc24_19.1: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.03a) [symbolic = %Iterate.facet.loc24_19.5 (constants.%Iterate.facet.183)]
// CHECK:STDOUT:     %.loc24_19.1: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc24_19.1 [symbolic = %Iterate.facet.loc24_19.5 (constants.%Iterate.facet.183)]
// CHECK:STDOUT:     %Iterate.facet.loc24_19.2: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.03a) [symbolic = %Iterate.facet.loc24_19.5 (constants.%Iterate.facet.183)]
// CHECK:STDOUT:     %.loc24_19.2: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc24_19.2 [symbolic = %Iterate.facet.loc24_19.5 (constants.%Iterate.facet.183)]
// CHECK:STDOUT:     %specific_impl_fn.loc24_19.1: <specific function> = specific_impl_function %impl.elem2.loc24_19.1, @Iterate.WithSelf.NewCursor(constants.%Iterate.facet.183) [symbolic = %specific_impl_fn.loc24_19.3 (constants.%specific_impl_fn.929)]
// CHECK:STDOUT:     %bound_method.loc24_19.2: <bound method> = bound_method %r.ref, %specific_impl_fn.loc24_19.1
// CHECK:STDOUT:     %var: ref @TestIterate.%as_type (%as_type.dff) = var_storage invalid
// CHECK:STDOUT:     %Iterate.WithSelf.NewCursor.call: init @TestIterate.%as_type (%as_type.dff) to %var = call %bound_method.loc24_19.2(%r.ref)
// CHECK:STDOUT:     assign %var, %Iterate.WithSelf.NewCursor.call
// CHECK:STDOUT:     br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.next:
// CHECK:STDOUT:     %impl.elem3.loc24_19.1: @TestIterate.%.loc24_19.24 (%.b8e) = impl_witness_access constants.%Iterate.lookup_impl_witness.03a, element3 [symbolic = %impl.elem3.loc24_19.2 (constants.%impl.elem3.64e)]
// CHECK:STDOUT:     %bound_method.loc24_19.3: <bound method> = bound_method %r.ref, %impl.elem3.loc24_19.1
// CHECK:STDOUT:     %Iterate.facet.loc24_19.3: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.03a) [symbolic = %Iterate.facet.loc24_19.5 (constants.%Iterate.facet.183)]
// CHECK:STDOUT:     %.loc24_19.3: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc24_19.3 [symbolic = %Iterate.facet.loc24_19.5 (constants.%Iterate.facet.183)]
// CHECK:STDOUT:     %Iterate.facet.loc24_19.4: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.03a) [symbolic = %Iterate.facet.loc24_19.5 (constants.%Iterate.facet.183)]
// CHECK:STDOUT:     %.loc24_19.4: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc24_19.4 [symbolic = %Iterate.facet.loc24_19.5 (constants.%Iterate.facet.183)]
// CHECK:STDOUT:     %.loc24_19.5: %facet_type.8a3 = converted constants.%as_type.dff, constants.%impl.elem1.277 [symbolic = %impl.elem1 (constants.%impl.elem1.277)]
// CHECK:STDOUT:     %.loc24_19.6: %facet_type.8a3 = converted constants.%as_type.dff, constants.%impl.elem1.277 [symbolic = %impl.elem1 (constants.%impl.elem1.277)]
// CHECK:STDOUT:     %specific_impl_fn.loc24_19.2: <specific function> = specific_impl_function %impl.elem3.loc24_19.1, @Iterate.WithSelf.Next(constants.%Iterate.facet.183) [symbolic = %specific_impl_fn.loc24_19.4 (constants.%specific_impl_fn.520)]
// CHECK:STDOUT:     %bound_method.loc24_19.4: <bound method> = bound_method %r.ref, %specific_impl_fn.loc24_19.2
// CHECK:STDOUT:     %.loc24_19.7: ref %Optional.49a = temporary_storage
// CHECK:STDOUT:     %Iterate.WithSelf.Next.call: init %Optional.49a to %.loc24_19.7 = call %bound_method.loc24_19.4(%r.ref, %var)
// CHECK:STDOUT:     %.loc24_19.8: ref %Optional.49a = temporary %.loc24_19.7, %Iterate.WithSelf.Next.call
// CHECK:STDOUT:     %.loc24_19.9: %Optional.HasValue.type.79d = specific_constant imports.%Core.import_ref.bb6, @Optional(constants.%OptionalStorage.facet.424) [concrete = constants.%Optional.HasValue.718]
// CHECK:STDOUT:     %HasValue.ref: %Optional.HasValue.type.79d = name_ref HasValue, %.loc24_19.9 [concrete = constants.%Optional.HasValue.718]
// CHECK:STDOUT:     %Optional.HasValue.bound: <bound method> = bound_method %.loc24_19.8, %HasValue.ref
// CHECK:STDOUT:     %Optional.HasValue.specific_fn: <specific function> = specific_function %HasValue.ref, @Optional.HasValue(constants.%OptionalStorage.facet.424) [concrete = constants.%Optional.HasValue.specific_fn.409]
// CHECK:STDOUT:     %bound_method.loc24_19.5: <bound method> = bound_method %.loc24_19.8, %Optional.HasValue.specific_fn
// CHECK:STDOUT:     %.loc24_19.10: %Optional.49a = acquire_value %.loc24_19.8
// CHECK:STDOUT:     %Optional.HasValue.call: init bool = call %bound_method.loc24_19.5(%.loc24_19.10)
// CHECK:STDOUT:     %.loc24_19.11: bool = value_of_initializer %Optional.HasValue.call
// CHECK:STDOUT:     %.loc24_19.12: bool = converted %Optional.HasValue.call, %.loc24_19.11
// CHECK:STDOUT:     if %.loc24_19.12 br !for.body else br !for.done
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.body:
// CHECK:STDOUT:     %.loc24_19.13: %Optional.Get.type.3ba = specific_constant imports.%Core.import_ref.dab, @Optional(constants.%OptionalStorage.facet.424) [concrete = constants.%Optional.Get.a7f]
// CHECK:STDOUT:     %Get.ref: %Optional.Get.type.3ba = name_ref Get, %.loc24_19.13 [concrete = constants.%Optional.Get.a7f]
// CHECK:STDOUT:     %Optional.Get.bound: <bound method> = bound_method %.loc24_19.8, %Get.ref
// CHECK:STDOUT:     %Optional.Get.specific_fn: <specific function> = specific_function %Get.ref, @Optional.Get(constants.%OptionalStorage.facet.424) [concrete = constants.%Optional.Get.specific_fn.f9b]
// CHECK:STDOUT:     %bound_method.loc24_19.6: <bound method> = bound_method %.loc24_19.8, %Optional.Get.specific_fn
// CHECK:STDOUT:     %.loc24_19.14: %Optional.49a = acquire_value %.loc24_19.8
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
// CHECK:STDOUT:     %facet_value.loc24_19.1: %facet_type.8a3 = facet_value constants.%Optional.49a, (constants.%custom_witness.b80ef6.8, constants.%custom_witness.2cb555.8) [concrete = constants.%facet_value.153]
// CHECK:STDOUT:     %.loc24_19.17: %facet_type.8a3 = converted constants.%Optional.49a, %facet_value.loc24_19.1 [concrete = constants.%facet_value.153]
// CHECK:STDOUT:     %facet_value.loc24_19.2: %facet_type.8a3 = facet_value constants.%Optional.49a, (constants.%custom_witness.b80ef6.8, constants.%custom_witness.2cb555.8) [concrete = constants.%facet_value.153]
// CHECK:STDOUT:     %.loc24_19.18: %facet_type.8a3 = converted constants.%Optional.49a, %facet_value.loc24_19.2 [concrete = constants.%facet_value.153]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %SelfDestruct.call.loc24_19.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc24_19.8)
// CHECK:STDOUT:     br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.done:
// CHECK:STDOUT:     %facet_value.loc24_19.3: %facet_type.8a3 = facet_value constants.%Optional.49a, (constants.%custom_witness.b80ef6.8, constants.%custom_witness.2cb555.8) [concrete = constants.%facet_value.153]
// CHECK:STDOUT:     %.loc24_19.19: %facet_type.8a3 = converted constants.%Optional.49a, %facet_value.loc24_19.3 [concrete = constants.%facet_value.153]
// CHECK:STDOUT:     %facet_value.loc24_19.4: %facet_type.8a3 = facet_value constants.%Optional.49a, (constants.%custom_witness.b80ef6.8, constants.%custom_witness.2cb555.8) [concrete = constants.%facet_value.153]
// CHECK:STDOUT:     %.loc24_19.20: %facet_type.8a3 = converted constants.%Optional.49a, %facet_value.loc24_19.4 [concrete = constants.%facet_value.153]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %SelfDestruct.call.loc24_19.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc24_19.8)
// CHECK:STDOUT:     %.loc24_19.21: %facet_type.8a3 = converted constants.%as_type.dff, constants.%impl.elem1.277 [symbolic = %impl.elem1 (constants.%impl.elem1.277)]
// CHECK:STDOUT:     %.loc24_19.22: %facet_type.8a3 = converted constants.%as_type.dff, constants.%impl.elem1.277 [symbolic = %impl.elem1 (constants.%impl.elem1.277)]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %SelfDestruct.call.loc24_19.3: init %empty_tuple.type = call %SelfDestruct.specific_fn.3(%var)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc24_19.1(%self.param: ref %.cc0) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc24_19.2(%self.param: ref %MaybeUnformed.15c) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc24_19.3(%self.param: ref bool) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc24_19.4(%self.param: ref %struct_type.value.has_value.12d) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc24_19.1: ref bool = struct_access %self.param, element1
// CHECK:STDOUT:   %facet_value.loc24_19.1: %facet_type.8a3 = facet_value bool, (constants.%custom_witness.b80ef6.5, constants.%custom_witness.2cb555.3) [concrete = constants.%facet_value.ff1]
// CHECK:STDOUT:   %.loc24_19.2: %facet_type.8a3 = converted bool, %facet_value.loc24_19.1 [concrete = constants.%facet_value.ff1]
// CHECK:STDOUT:   %facet_value.loc24_19.2: %facet_type.8a3 = facet_value bool, (constants.%custom_witness.b80ef6.5, constants.%custom_witness.2cb555.3) [concrete = constants.%facet_value.ff1]
// CHECK:STDOUT:   %.loc24_19.3: %facet_type.8a3 = converted bool, %facet_value.loc24_19.2 [concrete = constants.%facet_value.ff1]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc24_19.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc24_19.1)
// CHECK:STDOUT:   %.loc24_19.4: ref %MaybeUnformed.15c = struct_access %self.param, element0
// CHECK:STDOUT:   %facet_value.loc24_19.3: %facet_type.8a3 = facet_value constants.%MaybeUnformed.15c, (constants.%custom_witness.b80ef6.4, constants.%custom_witness.2cb555.5) [concrete = constants.%facet_value.f6f]
// CHECK:STDOUT:   %.loc24_19.5: %facet_type.8a3 = converted constants.%MaybeUnformed.15c, %facet_value.loc24_19.3 [concrete = constants.%facet_value.f6f]
// CHECK:STDOUT:   %facet_value.loc24_19.4: %facet_type.8a3 = facet_value constants.%MaybeUnformed.15c, (constants.%custom_witness.b80ef6.4, constants.%custom_witness.2cb555.5) [concrete = constants.%facet_value.f6f]
// CHECK:STDOUT:   %.loc24_19.6: %facet_type.8a3 = converted constants.%MaybeUnformed.15c, %facet_value.loc24_19.4 [concrete = constants.%facet_value.f6f]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc24_19.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc24_19.4)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyImpl.WithSelf.Op.loc24_19.1(%self.param: ref bool) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyImpl.WithSelf.Op.loc24_19.2(%self.param: ref %.cc0) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyImpl.WithSelf.Op.loc24_19.3(%self.param: ref %MaybeUnformed.15c) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc24_19.5(%self.param: ref %DefaultOptionalStorage.877) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %has_value.ref: %DefaultOptionalStorage.elem.718 = name_ref has_value, imports.%Core.import_ref.1cb [concrete = imports.%field_decl.255]
// CHECK:STDOUT:   %.loc24_19.1: ref bool = class_element_access %self.param, element1
// CHECK:STDOUT:   %facet_value.loc24_19.1: %facet_type.8a3 = facet_value bool, (constants.%custom_witness.b80ef6.5, constants.%custom_witness.2cb555.3) [concrete = constants.%facet_value.ff1]
// CHECK:STDOUT:   %.loc24_19.2: %facet_type.8a3 = converted bool, %facet_value.loc24_19.1 [concrete = constants.%facet_value.ff1]
// CHECK:STDOUT:   %facet_value.loc24_19.2: %facet_type.8a3 = facet_value bool, (constants.%custom_witness.b80ef6.5, constants.%custom_witness.2cb555.3) [concrete = constants.%facet_value.ff1]
// CHECK:STDOUT:   %.loc24_19.3: %facet_type.8a3 = converted bool, %facet_value.loc24_19.2 [concrete = constants.%facet_value.ff1]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc24_19.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc24_19.1)
// CHECK:STDOUT:   %value.ref: %DefaultOptionalStorage.elem.e01 = name_ref value, imports.%Core.import_ref.1c5 [concrete = imports.%field_decl.649]
// CHECK:STDOUT:   %.loc24_19.4: ref %MaybeUnformed.15c = class_element_access %self.param, element0
// CHECK:STDOUT:   %facet_value.loc24_19.3: %facet_type.8a3 = facet_value constants.%MaybeUnformed.15c, (constants.%custom_witness.b80ef6.4, constants.%custom_witness.2cb555.5) [concrete = constants.%facet_value.f6f]
// CHECK:STDOUT:   %.loc24_19.5: %facet_type.8a3 = converted constants.%MaybeUnformed.15c, %facet_value.loc24_19.3 [concrete = constants.%facet_value.f6f]
// CHECK:STDOUT:   %facet_value.loc24_19.4: %facet_type.8a3 = facet_value constants.%MaybeUnformed.15c, (constants.%custom_witness.b80ef6.4, constants.%custom_witness.2cb555.5) [concrete = constants.%facet_value.f6f]
// CHECK:STDOUT:   %.loc24_19.6: %facet_type.8a3 = converted constants.%MaybeUnformed.15c, %facet_value.loc24_19.4 [concrete = constants.%facet_value.f6f]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc24_19.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc24_19.4)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc24_19.6(%self.param: ref %Optional.49a) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyImpl.WithSelf.Op.loc24_19.4(%self.param: ref %struct_type.value.has_value.12d) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyImpl.WithSelf.Op.loc24_19.5(%self.param: ref %DefaultOptionalStorage.877) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyImpl.WithSelf.Op.loc24_19.6(%self.param: ref %Optional.49a) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestRangeFor(%c.param: %ConstRange) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %i.patt: %pattern_type.fb7 = value_binding_pattern i [concrete = constants.%i.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c.ref: %ConstRange = name_ref c, %c
// CHECK:STDOUT:   %impl.elem2: %.7b6 = impl_witness_access constants.%Iterate.impl_witness.17d, element2 [concrete = constants.%T.as_type.as.Iterate.impl.NewCursor.1ac]
// CHECK:STDOUT:   %bound_method.loc38_19.1: <bound method> = bound_method %c.ref, %impl.elem2
// CHECK:STDOUT:   %facet_value.loc38_19.1: %CppRange_where.type.d682d6.1 = facet_value constants.%ConstRange, (constants.%custom_witness.b80ef6.2, constants.%custom_witness.2cb555.2, constants.%Copy.impl_witness.0d6, constants.%custom_witness.5d5, constants.%custom_witness.1fc, constants.%custom_witness.2cb555.9, constants.%custom_witness.994, constants.%custom_witness.393, constants.%custom_witness.821) [concrete = constants.%facet_value.d36]
// CHECK:STDOUT:   %.loc38_19.1: %CppRange_where.type.d682d6.1 = converted constants.%ConstRange, %facet_value.loc38_19.1 [concrete = constants.%facet_value.d36]
// CHECK:STDOUT:   %facet_value.loc38_19.2: %CppRange_where.type.d682d6.1 = facet_value constants.%ConstRange, (constants.%custom_witness.b80ef6.2, constants.%custom_witness.2cb555.2, constants.%Copy.impl_witness.0d6, constants.%custom_witness.5d5, constants.%custom_witness.1fc, constants.%custom_witness.2cb555.9, constants.%custom_witness.994, constants.%custom_witness.393, constants.%custom_witness.821) [concrete = constants.%facet_value.d36]
// CHECK:STDOUT:   %.loc38_19.2: %CppRange_where.type.d682d6.1 = converted constants.%ConstRange, %facet_value.loc38_19.2 [concrete = constants.%facet_value.d36]
// CHECK:STDOUT:   %specific_fn.loc38_19.1: <specific function> = specific_function %impl.elem2, @T.as_type.as.Iterate.impl.NewCursor(constants.%facet_value.d36) [concrete = constants.%T.as_type.as.Iterate.impl.NewCursor.specific_fn]
// CHECK:STDOUT:   %bound_method.loc38_19.2: <bound method> = bound_method %c.ref, %specific_fn.loc38_19.1
// CHECK:STDOUT:   %var: ref %tuple.type.005 = var_storage invalid
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.call: init %tuple.type.005 to %var = call %bound_method.loc38_19.2(%c.ref)
// CHECK:STDOUT:   assign %var, %T.as_type.as.Iterate.impl.NewCursor.call
// CHECK:STDOUT:   br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT: !for.next:
// CHECK:STDOUT:   %impl.elem3: %.4e6 = impl_witness_access constants.%Iterate.impl_witness.17d, element3 [concrete = constants.%T.as_type.as.Iterate.impl.Next.b00]
// CHECK:STDOUT:   %bound_method.loc38_19.3: <bound method> = bound_method %c.ref, %impl.elem3
// CHECK:STDOUT:   %facet_value.loc38_19.3: %CppRange_where.type.d682d6.1 = facet_value constants.%ConstRange, (constants.%custom_witness.b80ef6.2, constants.%custom_witness.2cb555.2, constants.%Copy.impl_witness.0d6, constants.%custom_witness.5d5, constants.%custom_witness.1fc, constants.%custom_witness.2cb555.9, constants.%custom_witness.994, constants.%custom_witness.393, constants.%custom_witness.821) [concrete = constants.%facet_value.d36]
// CHECK:STDOUT:   %.loc38_19.3: %CppRange_where.type.d682d6.1 = converted constants.%ConstRange, %facet_value.loc38_19.3 [concrete = constants.%facet_value.d36]
// CHECK:STDOUT:   %facet_value.loc38_19.4: %CppRange_where.type.d682d6.1 = facet_value constants.%ConstRange, (constants.%custom_witness.b80ef6.2, constants.%custom_witness.2cb555.2, constants.%Copy.impl_witness.0d6, constants.%custom_witness.5d5, constants.%custom_witness.1fc, constants.%custom_witness.2cb555.9, constants.%custom_witness.994, constants.%custom_witness.393, constants.%custom_witness.821) [concrete = constants.%facet_value.d36]
// CHECK:STDOUT:   %.loc38_19.4: %CppRange_where.type.d682d6.1 = converted constants.%ConstRange, %facet_value.loc38_19.4 [concrete = constants.%facet_value.d36]
// CHECK:STDOUT:   %specific_fn.loc38_19.2: <specific function> = specific_function %impl.elem3, @T.as_type.as.Iterate.impl.Next(constants.%facet_value.d36) [concrete = constants.%T.as_type.as.Iterate.impl.Next.specific_fn]
// CHECK:STDOUT:   %bound_method.loc38_19.4: <bound method> = bound_method %c.ref, %specific_fn.loc38_19.2
// CHECK:STDOUT:   %.loc38_19.5: ref %Optional.49a = temporary_storage
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.call: init %Optional.49a to %.loc38_19.5 = call %bound_method.loc38_19.4(%c.ref, %var)
// CHECK:STDOUT:   %.loc38_19.6: ref %Optional.49a = temporary %.loc38_19.5, %T.as_type.as.Iterate.impl.Next.call
// CHECK:STDOUT:   %.loc38_19.7: %Optional.HasValue.type.79d = specific_constant imports.%Core.import_ref.bb6, @Optional(constants.%OptionalStorage.facet.424) [concrete = constants.%Optional.HasValue.718]
// CHECK:STDOUT:   %HasValue.ref: %Optional.HasValue.type.79d = name_ref HasValue, %.loc38_19.7 [concrete = constants.%Optional.HasValue.718]
// CHECK:STDOUT:   %Optional.HasValue.bound: <bound method> = bound_method %.loc38_19.6, %HasValue.ref
// CHECK:STDOUT:   %Optional.HasValue.specific_fn: <specific function> = specific_function %HasValue.ref, @Optional.HasValue(constants.%OptionalStorage.facet.424) [concrete = constants.%Optional.HasValue.specific_fn.409]
// CHECK:STDOUT:   %bound_method.loc38_19.5: <bound method> = bound_method %.loc38_19.6, %Optional.HasValue.specific_fn
// CHECK:STDOUT:   %.loc38_19.8: %Optional.49a = acquire_value %.loc38_19.6
// CHECK:STDOUT:   %Optional.HasValue.call: init bool = call %bound_method.loc38_19.5(%.loc38_19.8)
// CHECK:STDOUT:   %.loc38_19.9: bool = value_of_initializer %Optional.HasValue.call
// CHECK:STDOUT:   %.loc38_19.10: bool = converted %Optional.HasValue.call, %.loc38_19.9
// CHECK:STDOUT:   if %.loc38_19.10 br !for.body else br !for.done
// CHECK:STDOUT:
// CHECK:STDOUT: !for.body:
// CHECK:STDOUT:   %.loc38_19.11: %Optional.Get.type.3ba = specific_constant imports.%Core.import_ref.dab, @Optional(constants.%OptionalStorage.facet.424) [concrete = constants.%Optional.Get.a7f]
// CHECK:STDOUT:   %Get.ref: %Optional.Get.type.3ba = name_ref Get, %.loc38_19.11 [concrete = constants.%Optional.Get.a7f]
// CHECK:STDOUT:   %Optional.Get.bound: <bound method> = bound_method %.loc38_19.6, %Get.ref
// CHECK:STDOUT:   %Optional.Get.specific_fn: <specific function> = specific_function %Get.ref, @Optional.Get(constants.%OptionalStorage.facet.424) [concrete = constants.%Optional.Get.specific_fn.f9b]
// CHECK:STDOUT:   %bound_method.loc38_19.6: <bound method> = bound_method %.loc38_19.6, %Optional.Get.specific_fn
// CHECK:STDOUT:   %.loc38_19.12: %Optional.49a = acquire_value %.loc38_19.6
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
// CHECK:STDOUT:   %facet_value.loc38_19.5: %facet_type.8a3 = facet_value constants.%Optional.49a, (constants.%custom_witness.b80ef6.8, constants.%custom_witness.2cb555.8) [concrete = constants.%facet_value.153]
// CHECK:STDOUT:   %.loc38_19.15: %facet_type.8a3 = converted constants.%Optional.49a, %facet_value.loc38_19.5 [concrete = constants.%facet_value.153]
// CHECK:STDOUT:   %facet_value.loc38_19.6: %facet_type.8a3 = facet_value constants.%Optional.49a, (constants.%custom_witness.b80ef6.8, constants.%custom_witness.2cb555.8) [concrete = constants.%facet_value.153]
// CHECK:STDOUT:   %.loc38_19.16: %facet_type.8a3 = converted constants.%Optional.49a, %facet_value.loc38_19.6 [concrete = constants.%facet_value.153]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc38_19.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc38_19.6)
// CHECK:STDOUT:   br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT: !for.done:
// CHECK:STDOUT:   %facet_value.loc38_19.7: %facet_type.8a3 = facet_value constants.%Optional.49a, (constants.%custom_witness.b80ef6.8, constants.%custom_witness.2cb555.8) [concrete = constants.%facet_value.153]
// CHECK:STDOUT:   %.loc38_19.17: %facet_type.8a3 = converted constants.%Optional.49a, %facet_value.loc38_19.7 [concrete = constants.%facet_value.153]
// CHECK:STDOUT:   %facet_value.loc38_19.8: %facet_type.8a3 = facet_value constants.%Optional.49a, (constants.%custom_witness.b80ef6.8, constants.%custom_witness.2cb555.8) [concrete = constants.%facet_value.153]
// CHECK:STDOUT:   %.loc38_19.18: %facet_type.8a3 = converted constants.%Optional.49a, %facet_value.loc38_19.8 [concrete = constants.%facet_value.153]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc38_19.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc38_19.6)
// CHECK:STDOUT:   %facet_value.loc38_19.9: %facet_type.8a3 = facet_value constants.%tuple.type.005, (constants.%custom_witness.b80ef6.9, constants.%custom_witness.2cb555.10) [concrete = constants.%facet_value.54b]
// CHECK:STDOUT:   %.loc38_19.19: %facet_type.8a3 = converted constants.%tuple.type.005, %facet_value.loc38_19.9 [concrete = constants.%facet_value.54b]
// CHECK:STDOUT:   %facet_value.loc38_19.10: %facet_type.8a3 = facet_value constants.%tuple.type.005, (constants.%custom_witness.b80ef6.9, constants.%custom_witness.2cb555.10) [concrete = constants.%facet_value.54b]
// CHECK:STDOUT:   %.loc38_19.20: %facet_type.8a3 = converted constants.%tuple.type.005, %facet_value.loc38_19.10 [concrete = constants.%facet_value.54b]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc38_19.3: init %empty_tuple.type = call %SelfDestruct.specific_fn.3(%var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestIterate(constants.%R) {
// CHECK:STDOUT:   %R.patt.loc20_17.2 => constants.%R.patt
// CHECK:STDOUT:   %R.loc20_17.1 => constants.%R
// CHECK:STDOUT:   %R.as_type.loc20_61.1 => constants.%R.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.df6
// CHECK:STDOUT:   %r.param_patt.loc20_59.2 => constants.%r.param_patt.7c5
// CHECK:STDOUT:   %r.patt.loc20_59.2 => constants.%r.patt.598
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestIterate(constants.%facet_value.7c0) {
// CHECK:STDOUT:   %R.patt.loc20_17.2 => constants.%R.patt
// CHECK:STDOUT:   %R.loc20_17.1 => constants.%facet_value.7c0
// CHECK:STDOUT:   %R.as_type.loc20_61.1 => constants.%ConstRange
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.025
// CHECK:STDOUT:   %r.param_patt.loc20_59.2 => constants.%r.param_patt.a73
// CHECK:STDOUT:   %r.patt.loc20_59.2 => constants.%r.patt.0b5
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc20 => constants.%complete_type.357
// CHECK:STDOUT:   %Iterate.lookup_impl_witness => constants.%Iterate.impl_witness.17d
// CHECK:STDOUT:   %Iterate.facet.loc24_19.5 => constants.%Iterate.facet.3bd
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type => constants.%Iterate.WithSelf.NewCursor.type.bf3
// CHECK:STDOUT:   %.loc24_19.23 => constants.%.7b6
// CHECK:STDOUT:   %impl.elem2.loc24_19.2 => constants.%T.as_type.as.Iterate.impl.NewCursor.1ac
// CHECK:STDOUT:   %specific_impl_fn.loc24_19.3 => constants.%T.as_type.as.Iterate.impl.NewCursor.specific_fn
// CHECK:STDOUT:   %impl.elem1 => constants.%facet_value.54b
// CHECK:STDOUT:   %as_type => constants.%tuple.type.005
// CHECK:STDOUT:   %require_complete.1 => constants.%complete_type.320
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type => constants.%Iterate.WithSelf.Next.type.3a6
// CHECK:STDOUT:   %.loc24_19.24 => constants.%.4e6
// CHECK:STDOUT:   %impl.elem3.loc24_19.2 => constants.%T.as_type.as.Iterate.impl.Next.b00
// CHECK:STDOUT:   %specific_impl_fn.loc24_19.4 => constants.%T.as_type.as.Iterate.impl.Next.specific_fn
// CHECK:STDOUT:   %SelfDestruct.specific_fn.5 => constants.%SelfDestruct.specific_fn.770
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_adl_returns_same_types_mutable.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %Iterate.type: type = facet_type <@Iterate> [concrete]
// CHECK:STDOUT:   %OptionalStorage.type: type = facet_type <@OptionalStorage> [concrete]
// CHECK:STDOUT:   %facet_type.d61: type = facet_type <@SubobjectDestroy & @DestroyImpl & @Copy> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %T.220: %OptionalStorage.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Optional.Get.type.bb9: type = fn_type @Optional.Get, @Optional(%T.220) [symbolic]
// CHECK:STDOUT:   %Optional.Get.fa1: %Optional.Get.type.bb9 = struct_value () [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.type.e9e: type = fn_type @Optional.HasValue, @Optional(%T.220) [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.6cd: %Optional.HasValue.type.e9e = struct_value () [symbolic]
// CHECK:STDOUT:   %facet_type.8a3: type = facet_type <@SubobjectDestroy & @DestroyImpl> [concrete]
// CHECK:STDOUT:   %T.9ab: %facet_type.d61 = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.19c: type = facet_access_type %T.9ab [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.lookup_impl_witness.855: <witness> = lookup_impl_witness %T.9ab, @SubobjectDestroy [symbolic]
// CHECK:STDOUT:   %DestroyImpl.lookup_impl_witness.ff9: <witness> = lookup_impl_witness %T.9ab, @DestroyImpl [symbolic]
// CHECK:STDOUT:   %facet_value.b35: %facet_type.8a3 = facet_value %T.as_type.19c, (%SubobjectDestroy.lookup_impl_witness.855, %DestroyImpl.lookup_impl_witness.ff9) [symbolic]
// CHECK:STDOUT:   %MaybeUnformed.2cf: type = class_type @MaybeUnformed, @MaybeUnformed(%facet_value.b35) [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.7db: type = class_type @DefaultOptionalStorage, @DefaultOptionalStorage(%T.9ab) [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.elem.98d: type = unbound_element_type %DefaultOptionalStorage.7db, bool [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.elem.b5a: type = unbound_element_type %DefaultOptionalStorage.7db, %MaybeUnformed.2cf [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Get.type.b32: type = fn_type @T.as_type.as.OptionalStorage.impl.Get, @T.as_type.as.OptionalStorage.impl(%T.9ab) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Get.8b0: %T.as_type.as.OptionalStorage.impl.Get.type.b32 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Has.type.8e5: type = fn_type @T.as_type.as.OptionalStorage.impl.Has, @T.as_type.as.OptionalStorage.impl(%T.9ab) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Has.2d5: %T.as_type.as.OptionalStorage.impl.Has.type.8e5 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Some.type.61d: type = fn_type @T.as_type.as.OptionalStorage.impl.Some, @T.as_type.as.OptionalStorage.impl(%T.9ab) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Some.ba7: %T.as_type.as.OptionalStorage.impl.Some.type.61d = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.None.type.a9f: type = fn_type @T.as_type.as.OptionalStorage.impl.None, @T.as_type.as.OptionalStorage.impl(%T.9ab) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.None.083: %T.as_type.as.OptionalStorage.impl.None.type.a9f = struct_value () [symbolic]
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
// CHECK:STDOUT:   %f64.dc1: type = class_type @Float, @Float(%int_64) [concrete]
// CHECK:STDOUT:   %pattern_type.fb7: type = pattern_type %f64.dc1 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.2: type = fn_type @SubobjectDestroy.WithSelf.Op.loc22 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.2: %SubobjectDestroy.WithSelf.Op.type.6870fd.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.2: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.2), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.type.b77747.2: type = fn_type @DestroyImpl.WithSelf.Op.loc22 [concrete]
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.25aa57.2: %DestroyImpl.WithSelf.Op.type.b77747.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.2cb555.2: <witness> = custom_witness (%DestroyImpl.WithSelf.Op.25aa57.2), @DestroyImpl [concrete]
// CHECK:STDOUT:   %Float.as.Copy.impl.Op.type.ff3: type = fn_type @Float.as.Copy.impl.Op, @Float.as.Copy.impl(%N.fe9) [symbolic]
// CHECK:STDOUT:   %Float.as.Copy.impl.Op.d93: %Float.as.Copy.impl.Op.type.ff3 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.0d6: <witness> = impl_witness imports.%Copy.impl_witness_table.8dd, @Float.as.Copy.impl(%int_64) [concrete]
// CHECK:STDOUT:   %facet_value.340: %facet_type.d61 = facet_value %f64.dc1, (%custom_witness.b80ef6.2, %custom_witness.2cb555.2, %Copy.impl_witness.0d6) [concrete]
// CHECK:STDOUT:   %Iterate.lookup_impl_witness.1d8: <witness> = lookup_impl_witness %.Self.500, @Iterate [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.8a4: %facet_type.d61 = impl_witness_access %Iterate.lookup_impl_witness.1d8, element0 [symbolic_self]
// CHECK:STDOUT:   %Iterate_where.type.7ce: type = facet_type <@Iterate where %impl.elem0.8a4 = %facet_value.340> [concrete]
// CHECK:STDOUT:   %pattern_type.cb4: type = pattern_type %Iterate_where.type.7ce [concrete]
// CHECK:STDOUT:   %R.patt: %pattern_type.cb4 = symbolic_binding_pattern R, 0 [symbolic]
// CHECK:STDOUT:   %R: %Iterate_where.type.7ce = symbolic_binding R, 0 [symbolic]
// CHECK:STDOUT:   %R.as_type: type = facet_access_type %R [symbolic]
// CHECK:STDOUT:   %pattern_type.df6: type = pattern_type %R.as_type [symbolic]
// CHECK:STDOUT:   %r.param_patt.7c5: %pattern_type.df6 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %r.patt.598: %pattern_type.df6 = wrapper_binding_pattern r, %r.param_patt.7c5 [symbolic]
// CHECK:STDOUT:   %i.patt: %pattern_type.fb7 = value_binding_pattern i [concrete]
// CHECK:STDOUT:   %Iterate.lookup_impl_witness.03a: <witness> = lookup_impl_witness %R, @Iterate [symbolic]
// CHECK:STDOUT:   %Iterate.facet.183: %Iterate.type = facet_value %R.as_type, (%Iterate.lookup_impl_witness.03a) [symbolic]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.9bc: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.183) [symbolic]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.ffd: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.183) [symbolic]
// CHECK:STDOUT:   %.40e: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.9bc, %Iterate.facet.183 [symbolic]
// CHECK:STDOUT:   %impl.elem2.0a0: %.40e = impl_witness_access %Iterate.lookup_impl_witness.03a, element2 [symbolic]
// CHECK:STDOUT:   %impl.elem1.277: %facet_type.8a3 = impl_witness_access %Iterate.lookup_impl_witness.03a, element1 [symbolic]
// CHECK:STDOUT:   %as_type.dff: type = facet_access_type %impl.elem1.277 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.929: <specific function> = specific_impl_function %impl.elem2.0a0, @Iterate.WithSelf.NewCursor(%Iterate.facet.183) [symbolic]
// CHECK:STDOUT:   %.b8e: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.ffd, %Iterate.facet.183 [symbolic]
// CHECK:STDOUT:   %impl.elem3.64e: %.b8e = impl_witness_access %Iterate.lookup_impl_witness.03a, element3 [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.877: type = class_type @DefaultOptionalStorage, @DefaultOptionalStorage(%facet_value.340) [concrete]
// CHECK:STDOUT:   %OptionalStorage.impl_witness.ad2: <witness> = impl_witness imports.%OptionalStorage.impl_witness_table.968, @T.as_type.as.OptionalStorage.impl(%facet_value.340) [concrete]
// CHECK:STDOUT:   %OptionalStorage.facet.424: %OptionalStorage.type = facet_value %f64.dc1, (%OptionalStorage.impl_witness.ad2) [concrete]
// CHECK:STDOUT:   %Optional.49a: type = class_type @Optional, @Optional(%OptionalStorage.facet.424) [concrete]
// CHECK:STDOUT:   %specific_impl_fn.520: <specific function> = specific_impl_function %impl.elem3.64e, @Iterate.WithSelf.Next(%Iterate.facet.183) [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.type.79d: type = fn_type @Optional.HasValue, @Optional(%OptionalStorage.facet.424) [concrete]
// CHECK:STDOUT:   %Optional.HasValue.718: %Optional.HasValue.type.79d = struct_value () [concrete]
// CHECK:STDOUT:   %Optional.Get.type.3ba: type = fn_type @Optional.Get, @Optional(%OptionalStorage.facet.424) [concrete]
// CHECK:STDOUT:   %Optional.Get.a7f: %Optional.Get.type.3ba = struct_value () [concrete]
// CHECK:STDOUT:   %facet_value.391: %facet_type.8a3 = facet_value %f64.dc1, (%custom_witness.b80ef6.2, %custom_witness.2cb555.2) [concrete]
// CHECK:STDOUT:   %MaybeUnformed.15c: type = class_type @MaybeUnformed, @MaybeUnformed(%facet_value.391) [concrete]
// CHECK:STDOUT:   %.cc0: type = maybe_unformed_type %f64.dc1 [concrete]
// CHECK:STDOUT:   %DefaultOptionalStorage.elem.e01: type = unbound_element_type %DefaultOptionalStorage.877, %MaybeUnformed.15c [concrete]
// CHECK:STDOUT:   %DefaultOptionalStorage.elem.718: type = unbound_element_type %DefaultOptionalStorage.877, bool [concrete]
// CHECK:STDOUT:   %struct_type.value.has_value.12d: type = struct_type {.value: %MaybeUnformed.15c, .has_value: bool} [concrete]
// CHECK:STDOUT:   %Optional.HasValue.specific_fn.409: <specific function> = specific_function %Optional.HasValue.718, @Optional.HasValue(%OptionalStorage.facet.424) [concrete]
// CHECK:STDOUT:   %Optional.Get.specific_fn.f9b: <specific function> = specific_function %Optional.Get.a7f, @Optional.Get(%OptionalStorage.facet.424) [concrete]
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
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.4: type = fn_type @SubobjectDestroy.WithSelf.Op.loc26_19.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.4: %SubobjectDestroy.WithSelf.Op.type.6870fd.4 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.4: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.4), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.5: type = fn_type @SubobjectDestroy.WithSelf.Op.loc26_19.3 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.5: %SubobjectDestroy.WithSelf.Op.type.6870fd.5 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.5: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.5), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.type.b77747.3: type = fn_type @DestroyImpl.WithSelf.Op.loc26_19.1 [concrete]
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.25aa57.3: %DestroyImpl.WithSelf.Op.type.b77747.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.2cb555.3: <witness> = custom_witness (%DestroyImpl.WithSelf.Op.25aa57.3), @DestroyImpl [concrete]
// CHECK:STDOUT:   %facet_value.ff1: %facet_type.8a3 = facet_value bool, (%custom_witness.b80ef6.5, %custom_witness.2cb555.3) [concrete]
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.type.b77747.5: type = fn_type @DestroyImpl.WithSelf.Op.loc26_19.3 [concrete]
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.25aa57.5: %DestroyImpl.WithSelf.Op.type.b77747.5 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.2cb555.5: <witness> = custom_witness (%DestroyImpl.WithSelf.Op.25aa57.5), @DestroyImpl [concrete]
// CHECK:STDOUT:   %facet_value.f6f: %facet_type.8a3 = facet_value %MaybeUnformed.15c, (%custom_witness.b80ef6.4, %custom_witness.2cb555.5) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.8: type = fn_type @SubobjectDestroy.WithSelf.Op.loc26_19.6 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.8: %SubobjectDestroy.WithSelf.Op.type.6870fd.8 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.8: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.8), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.type.b77747.8: type = fn_type @DestroyImpl.WithSelf.Op.loc26_19.6 [concrete]
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.25aa57.8: %DestroyImpl.WithSelf.Op.type.b77747.8 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.2cb555.8: <witness> = custom_witness (%DestroyImpl.WithSelf.Op.25aa57.8), @DestroyImpl [concrete]
// CHECK:STDOUT:   %facet_value.153: %facet_type.8a3 = facet_value %Optional.49a, (%custom_witness.b80ef6.8, %custom_witness.2cb555.8) [concrete]
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
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.type.b77747.10: type = fn_type @DestroyImpl.WithSelf.Op.4 [concrete]
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.25aa57.10: %DestroyImpl.WithSelf.Op.type.b77747.10 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.2cb555.9: <witness> = custom_witness (%DestroyImpl.WithSelf.Op.25aa57.10), @DestroyImpl [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.b9731e.2: type = fn_type @Iterator.Op.2 [concrete]
// CHECK:STDOUT:   %Iterator.Op.ac6aad.2: %Iterator.Op.type.b9731e.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.f01: <witness> = custom_witness (%Iterator.Op.ac6aad.2), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.b9731e.3: type = fn_type @Iterator.Op.3 [concrete]
// CHECK:STDOUT:   %Iterator.Op.ac6aad.3: %Iterator.Op.type.b9731e.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.491: <witness> = custom_witness (%Iterator.Op.ac6aad.3), @Inc [concrete]
// CHECK:STDOUT:   %Equal.type: type = fn_type @Equal [concrete]
// CHECK:STDOUT:   %Equal: %Equal.type = struct_value () [concrete]
// CHECK:STDOUT:   %NotEqual.type: type = fn_type @NotEqual [concrete]
// CHECK:STDOUT:   %NotEqual: %NotEqual.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.ed8: <witness> = custom_witness (%Equal, %NotEqual), @EqWith, @EqWith(%Iterator) [concrete]
// CHECK:STDOUT:   %facet_value.8a6: %CppRange_where.type.d682d6.1 = facet_value %MutableRange, (%custom_witness.b80ef6.2, %custom_witness.2cb555.2, %Copy.impl_witness.0d6, %custom_witness.902, %custom_witness.f01, %custom_witness.2cb555.9, %custom_witness.05b, %custom_witness.491, %custom_witness.ed8) [concrete]
// CHECK:STDOUT:   %tuple.type.dd0: type = tuple_type (%Iterator, %Iterator) [concrete]
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.type.b77747.11: type = fn_type @DestroyImpl.WithSelf.Op.5 [concrete]
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.25aa57.11: %DestroyImpl.WithSelf.Op.type.b77747.11 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.2cb555.10: <witness> = custom_witness (%DestroyImpl.WithSelf.Op.25aa57.11), @DestroyImpl [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.9: type = fn_type @SubobjectDestroy.WithSelf.Op.3 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.9: %SubobjectDestroy.WithSelf.Op.type.6870fd.9 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.9: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.9), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %facet_value.a79: %facet_type.8a3 = facet_value %tuple.type.dd0, (%custom_witness.b80ef6.9, %custom_witness.2cb555.10) [concrete]
// CHECK:STDOUT:   %Iterate.impl_witness.8fd: <witness> = impl_witness imports.%Iterate.impl_witness_table.121, @T.as_type.as.Iterate.impl(%facet_value.8a6) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.type.53a: type = fn_type @T.as_type.as.Iterate.impl.NewCursor, @T.as_type.as.Iterate.impl(%facet_value.8a6) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.7bd: %T.as_type.as.Iterate.impl.NewCursor.type.53a = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.type.bf1: type = fn_type @T.as_type.as.Iterate.impl.Next, @T.as_type.as.Iterate.impl(%facet_value.8a6) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.b0d: %T.as_type.as.Iterate.impl.Next.type.bf1 = struct_value () [concrete]
// CHECK:STDOUT:   %Iterate.facet.c39: %Iterate.type = facet_value %MutableRange, (%Iterate.impl_witness.8fd) [concrete]
// CHECK:STDOUT:   %facet_value.a70: %Iterate_where.type.7ce = facet_value %MutableRange, (%Iterate.impl_witness.8fd) [concrete]
// CHECK:STDOUT:   %r.param_patt.aeb: %pattern_type.992 = value_param_pattern [concrete]
// CHECK:STDOUT:   %r.patt.892: %pattern_type.992 = wrapper_binding_pattern r, %r.param_patt.aeb [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.582: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.c39) [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.12b: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.c39) [concrete]
// CHECK:STDOUT:   %.131: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.582, %Iterate.facet.c39 [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.specific_fn: <specific function> = specific_function %T.as_type.as.Iterate.impl.NewCursor.7bd, @T.as_type.as.Iterate.impl.NewCursor(%facet_value.8a6) [concrete]
// CHECK:STDOUT:   %.9fd: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.12b, %Iterate.facet.c39 [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.specific_fn: <specific function> = specific_function %T.as_type.as.Iterate.impl.Next.b0d, @T.as_type.as.Iterate.impl.Next(%facet_value.8a6) [concrete]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.c79: <specific function> = specific_function %SelfDestruct, @SelfDestruct(%facet_value.a79) [concrete]
// CHECK:STDOUT:   %complete_type.843: <witness> = complete_type_witness %tuple.type.dd0 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.bb6: @Optional.%Optional.HasValue.type (%Optional.HasValue.type.e9e) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.HasValue (constants.%Optional.HasValue.6cd)]
// CHECK:STDOUT:   %Core.import_ref.dab: @Optional.%Optional.Get.type (%Optional.Get.type.bb9) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.Get (constants.%Optional.Get.fa1)]
// CHECK:STDOUT:   %Core.import_ref.38a: type = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%DefaultOptionalStorage (constants.%DefaultOptionalStorage.7db)]
// CHECK:STDOUT:   %Core.import_ref.61e: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.None.type (%T.as_type.as.OptionalStorage.impl.None.type.a9f) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.None (constants.%T.as_type.as.OptionalStorage.impl.None.083)]
// CHECK:STDOUT:   %Core.import_ref.ca1: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Some.type (%T.as_type.as.OptionalStorage.impl.Some.type.61d) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Some (constants.%T.as_type.as.OptionalStorage.impl.Some.ba7)]
// CHECK:STDOUT:   %Core.import_ref.6ba4: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Has.type (%T.as_type.as.OptionalStorage.impl.Has.type.8e5) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Has (constants.%T.as_type.as.OptionalStorage.impl.Has.2d5)]
// CHECK:STDOUT:   %Core.import_ref.73d: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Get.type (%T.as_type.as.OptionalStorage.impl.Get.type.b32) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Get (constants.%T.as_type.as.OptionalStorage.impl.Get.8b0)]
// CHECK:STDOUT:   %Core.import_ref.778 = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], unloaded
// CHECK:STDOUT:   %OptionalStorage.impl_witness_table.968 = impl_witness_table (%Core.import_ref.38a, %Core.import_ref.61e, %Core.import_ref.ca1, %Core.import_ref.6ba4, %Core.import_ref.73d, %Core.import_ref.778), @T.as_type.as.OptionalStorage.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.1c5: @DefaultOptionalStorage.%DefaultOptionalStorage.elem.1 (%DefaultOptionalStorage.elem.b5a) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [concrete = %field_decl.649]
// CHECK:STDOUT:   %Core.import_ref.1cb: @DefaultOptionalStorage.%DefaultOptionalStorage.elem.2 (%DefaultOptionalStorage.elem.98d) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [concrete = %field_decl.255]
// CHECK:STDOUT:   %Core.import_ref.d6c: %facet_type.d61 = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.Iterate.impl.%facet_value.1 (constants.%facet_value.9da)]
// CHECK:STDOUT:   %Core.import_ref.2ac: %facet_type.8a3 = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.Iterate.impl.%facet_value.2 (constants.%facet_value.bf6)]
// CHECK:STDOUT:   %Core.import_ref.fd1a: @T.as_type.as.Iterate.impl.%T.as_type.as.Iterate.impl.NewCursor.type (%T.as_type.as.Iterate.impl.NewCursor.type.953) = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.Iterate.impl.%T.as_type.as.Iterate.impl.NewCursor (constants.%T.as_type.as.Iterate.impl.NewCursor.544)]
// CHECK:STDOUT:   %Core.import_ref.273: @T.as_type.as.Iterate.impl.%T.as_type.as.Iterate.impl.Next.type (%T.as_type.as.Iterate.impl.Next.type.b36) = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.Iterate.impl.%T.as_type.as.Iterate.impl.Next (constants.%T.as_type.as.Iterate.impl.Next.d6b)]
// CHECK:STDOUT:   %Iterate.impl_witness_table.121 = impl_witness_table (%Core.import_ref.d6c, %Core.import_ref.2ac, %Core.import_ref.fd1a, %Core.import_ref.273), @T.as_type.as.Iterate.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.e69: @Float.as.Copy.impl.%Float.as.Copy.impl.Op.type (%Float.as.Copy.impl.Op.type.ff3) = import_ref Core//prelude/types/float, loc{{\d+_\d+}}, loaded [symbolic = @Float.as.Copy.impl.%Float.as.Copy.impl.Op (constants.%Float.as.Copy.impl.Op.d93)]
// CHECK:STDOUT:   %Copy.impl_witness_table.8dd = impl_witness_table (%Core.import_ref.e69), @Float.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.cb3: @Float.as.AddAssignWith.impl.%Float.as.AddAssignWith.impl.Op.type (%Float.as.AddAssignWith.impl.Op.type.fdd) = import_ref Core//prelude/types/float, loc{{\d+_\d+}}, loaded [symbolic = @Float.as.AddAssignWith.impl.%Float.as.AddAssignWith.impl.Op (constants.%Float.as.AddAssignWith.impl.Op.430)]
// CHECK:STDOUT:   %AddAssignWith.impl_witness_table = impl_witness_table (%Core.import_ref.cb3), @Float.as.AddAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.b0d = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %field_decl.255: @DefaultOptionalStorage.%DefaultOptionalStorage.elem.2 (%DefaultOptionalStorage.elem.98d) = field_decl has_value, element1, %Core.import_ref.b0d in [concrete] {}
// CHECK:STDOUT:   %Core.import_ref.4cb = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %field_decl.649: @DefaultOptionalStorage.%DefaultOptionalStorage.elem.1 (%DefaultOptionalStorage.elem.b5a) = field_decl value, element0, %Core.import_ref.4cb in [concrete] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @TestIterate(%R.loc22_17.2: %Iterate_where.type.7ce) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Iterate.lookup_impl_witness: <witness> = lookup_impl_witness %R.loc22_17.1, @Iterate [symbolic = %Iterate.lookup_impl_witness (constants.%Iterate.lookup_impl_witness.03a)]
// CHECK:STDOUT:   %Iterate.facet.loc26_19.5: %Iterate.type = facet_value %R.as_type.loc22_61.1, (%Iterate.lookup_impl_witness) [symbolic = %Iterate.facet.loc26_19.5 (constants.%Iterate.facet.183)]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.loc26_19.5) [symbolic = %Iterate.WithSelf.NewCursor.type (constants.%Iterate.WithSelf.NewCursor.type.9bc)]
// CHECK:STDOUT:   %.loc26_19.23: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type, %Iterate.facet.loc26_19.5 [symbolic = %.loc26_19.23 (constants.%.40e)]
// CHECK:STDOUT:   %impl.elem2.loc26_19.2: @TestIterate.%.loc26_19.23 (%.40e) = impl_witness_access %Iterate.lookup_impl_witness, element2 [symbolic = %impl.elem2.loc26_19.2 (constants.%impl.elem2.0a0)]
// CHECK:STDOUT:   %specific_impl_fn.loc26_19.3: <specific function> = specific_impl_function %impl.elem2.loc26_19.2, @Iterate.WithSelf.NewCursor(%Iterate.facet.loc26_19.5) [symbolic = %specific_impl_fn.loc26_19.3 (constants.%specific_impl_fn.929)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.loc26_19.5) [symbolic = %Iterate.WithSelf.Next.type (constants.%Iterate.WithSelf.Next.type.ffd)]
// CHECK:STDOUT:   %.loc26_19.24: type = fn_type_with_self_type %Iterate.WithSelf.Next.type, %Iterate.facet.loc26_19.5 [symbolic = %.loc26_19.24 (constants.%.b8e)]
// CHECK:STDOUT:   %impl.elem3.loc26_19.2: @TestIterate.%.loc26_19.24 (%.b8e) = impl_witness_access %Iterate.lookup_impl_witness, element3 [symbolic = %impl.elem3.loc26_19.2 (constants.%impl.elem3.64e)]
// CHECK:STDOUT:   %specific_impl_fn.loc26_19.4: <specific function> = specific_impl_function %impl.elem3.loc26_19.2, @Iterate.WithSelf.Next(%Iterate.facet.loc26_19.5) [symbolic = %specific_impl_fn.loc26_19.4 (constants.%specific_impl_fn.520)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%r.param: @TestIterate.%R.as_type.loc22_61.1 (%R.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %i.patt: %pattern_type.fb7 = value_binding_pattern i [concrete = constants.%i.patt]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %r.ref: @TestIterate.%R.as_type.loc22_61.1 (%R.as_type) = name_ref r, %r
// CHECK:STDOUT:     %impl.elem2.loc26_19.1: @TestIterate.%.loc26_19.23 (%.40e) = impl_witness_access constants.%Iterate.lookup_impl_witness.03a, element2 [symbolic = %impl.elem2.loc26_19.2 (constants.%impl.elem2.0a0)]
// CHECK:STDOUT:     %bound_method.loc26_19.1: <bound method> = bound_method %r.ref, %impl.elem2.loc26_19.1
// CHECK:STDOUT:     %Iterate.facet.loc26_19.1: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.03a) [symbolic = %Iterate.facet.loc26_19.5 (constants.%Iterate.facet.183)]
// CHECK:STDOUT:     %.loc26_19.1: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc26_19.1 [symbolic = %Iterate.facet.loc26_19.5 (constants.%Iterate.facet.183)]
// CHECK:STDOUT:     %Iterate.facet.loc26_19.2: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.03a) [symbolic = %Iterate.facet.loc26_19.5 (constants.%Iterate.facet.183)]
// CHECK:STDOUT:     %.loc26_19.2: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc26_19.2 [symbolic = %Iterate.facet.loc26_19.5 (constants.%Iterate.facet.183)]
// CHECK:STDOUT:     %specific_impl_fn.loc26_19.1: <specific function> = specific_impl_function %impl.elem2.loc26_19.1, @Iterate.WithSelf.NewCursor(constants.%Iterate.facet.183) [symbolic = %specific_impl_fn.loc26_19.3 (constants.%specific_impl_fn.929)]
// CHECK:STDOUT:     %bound_method.loc26_19.2: <bound method> = bound_method %r.ref, %specific_impl_fn.loc26_19.1
// CHECK:STDOUT:     %var: ref @TestIterate.%as_type (%as_type.dff) = var_storage invalid
// CHECK:STDOUT:     %Iterate.WithSelf.NewCursor.call: init @TestIterate.%as_type (%as_type.dff) to %var = call %bound_method.loc26_19.2(%r.ref)
// CHECK:STDOUT:     assign %var, %Iterate.WithSelf.NewCursor.call
// CHECK:STDOUT:     br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.next:
// CHECK:STDOUT:     %impl.elem3.loc26_19.1: @TestIterate.%.loc26_19.24 (%.b8e) = impl_witness_access constants.%Iterate.lookup_impl_witness.03a, element3 [symbolic = %impl.elem3.loc26_19.2 (constants.%impl.elem3.64e)]
// CHECK:STDOUT:     %bound_method.loc26_19.3: <bound method> = bound_method %r.ref, %impl.elem3.loc26_19.1
// CHECK:STDOUT:     %Iterate.facet.loc26_19.3: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.03a) [symbolic = %Iterate.facet.loc26_19.5 (constants.%Iterate.facet.183)]
// CHECK:STDOUT:     %.loc26_19.3: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc26_19.3 [symbolic = %Iterate.facet.loc26_19.5 (constants.%Iterate.facet.183)]
// CHECK:STDOUT:     %Iterate.facet.loc26_19.4: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.03a) [symbolic = %Iterate.facet.loc26_19.5 (constants.%Iterate.facet.183)]
// CHECK:STDOUT:     %.loc26_19.4: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc26_19.4 [symbolic = %Iterate.facet.loc26_19.5 (constants.%Iterate.facet.183)]
// CHECK:STDOUT:     %.loc26_19.5: %facet_type.8a3 = converted constants.%as_type.dff, constants.%impl.elem1.277 [symbolic = %impl.elem1 (constants.%impl.elem1.277)]
// CHECK:STDOUT:     %.loc26_19.6: %facet_type.8a3 = converted constants.%as_type.dff, constants.%impl.elem1.277 [symbolic = %impl.elem1 (constants.%impl.elem1.277)]
// CHECK:STDOUT:     %specific_impl_fn.loc26_19.2: <specific function> = specific_impl_function %impl.elem3.loc26_19.1, @Iterate.WithSelf.Next(constants.%Iterate.facet.183) [symbolic = %specific_impl_fn.loc26_19.4 (constants.%specific_impl_fn.520)]
// CHECK:STDOUT:     %bound_method.loc26_19.4: <bound method> = bound_method %r.ref, %specific_impl_fn.loc26_19.2
// CHECK:STDOUT:     %.loc26_19.7: ref %Optional.49a = temporary_storage
// CHECK:STDOUT:     %Iterate.WithSelf.Next.call: init %Optional.49a to %.loc26_19.7 = call %bound_method.loc26_19.4(%r.ref, %var)
// CHECK:STDOUT:     %.loc26_19.8: ref %Optional.49a = temporary %.loc26_19.7, %Iterate.WithSelf.Next.call
// CHECK:STDOUT:     %.loc26_19.9: %Optional.HasValue.type.79d = specific_constant imports.%Core.import_ref.bb6, @Optional(constants.%OptionalStorage.facet.424) [concrete = constants.%Optional.HasValue.718]
// CHECK:STDOUT:     %HasValue.ref: %Optional.HasValue.type.79d = name_ref HasValue, %.loc26_19.9 [concrete = constants.%Optional.HasValue.718]
// CHECK:STDOUT:     %Optional.HasValue.bound: <bound method> = bound_method %.loc26_19.8, %HasValue.ref
// CHECK:STDOUT:     %Optional.HasValue.specific_fn: <specific function> = specific_function %HasValue.ref, @Optional.HasValue(constants.%OptionalStorage.facet.424) [concrete = constants.%Optional.HasValue.specific_fn.409]
// CHECK:STDOUT:     %bound_method.loc26_19.5: <bound method> = bound_method %.loc26_19.8, %Optional.HasValue.specific_fn
// CHECK:STDOUT:     %.loc26_19.10: %Optional.49a = acquire_value %.loc26_19.8
// CHECK:STDOUT:     %Optional.HasValue.call: init bool = call %bound_method.loc26_19.5(%.loc26_19.10)
// CHECK:STDOUT:     %.loc26_19.11: bool = value_of_initializer %Optional.HasValue.call
// CHECK:STDOUT:     %.loc26_19.12: bool = converted %Optional.HasValue.call, %.loc26_19.11
// CHECK:STDOUT:     if %.loc26_19.12 br !for.body else br !for.done
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.body:
// CHECK:STDOUT:     %.loc26_19.13: %Optional.Get.type.3ba = specific_constant imports.%Core.import_ref.dab, @Optional(constants.%OptionalStorage.facet.424) [concrete = constants.%Optional.Get.a7f]
// CHECK:STDOUT:     %Get.ref: %Optional.Get.type.3ba = name_ref Get, %.loc26_19.13 [concrete = constants.%Optional.Get.a7f]
// CHECK:STDOUT:     %Optional.Get.bound: <bound method> = bound_method %.loc26_19.8, %Get.ref
// CHECK:STDOUT:     %Optional.Get.specific_fn: <specific function> = specific_function %Get.ref, @Optional.Get(constants.%OptionalStorage.facet.424) [concrete = constants.%Optional.Get.specific_fn.f9b]
// CHECK:STDOUT:     %bound_method.loc26_19.6: <bound method> = bound_method %.loc26_19.8, %Optional.Get.specific_fn
// CHECK:STDOUT:     %.loc26_19.14: %Optional.49a = acquire_value %.loc26_19.8
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
// CHECK:STDOUT:     %facet_value.loc26_19.1: %facet_type.8a3 = facet_value constants.%Optional.49a, (constants.%custom_witness.b80ef6.8, constants.%custom_witness.2cb555.8) [concrete = constants.%facet_value.153]
// CHECK:STDOUT:     %.loc26_19.17: %facet_type.8a3 = converted constants.%Optional.49a, %facet_value.loc26_19.1 [concrete = constants.%facet_value.153]
// CHECK:STDOUT:     %facet_value.loc26_19.2: %facet_type.8a3 = facet_value constants.%Optional.49a, (constants.%custom_witness.b80ef6.8, constants.%custom_witness.2cb555.8) [concrete = constants.%facet_value.153]
// CHECK:STDOUT:     %.loc26_19.18: %facet_type.8a3 = converted constants.%Optional.49a, %facet_value.loc26_19.2 [concrete = constants.%facet_value.153]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %SelfDestruct.call.loc26_19.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc26_19.8)
// CHECK:STDOUT:     br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.done:
// CHECK:STDOUT:     %facet_value.loc26_19.3: %facet_type.8a3 = facet_value constants.%Optional.49a, (constants.%custom_witness.b80ef6.8, constants.%custom_witness.2cb555.8) [concrete = constants.%facet_value.153]
// CHECK:STDOUT:     %.loc26_19.19: %facet_type.8a3 = converted constants.%Optional.49a, %facet_value.loc26_19.3 [concrete = constants.%facet_value.153]
// CHECK:STDOUT:     %facet_value.loc26_19.4: %facet_type.8a3 = facet_value constants.%Optional.49a, (constants.%custom_witness.b80ef6.8, constants.%custom_witness.2cb555.8) [concrete = constants.%facet_value.153]
// CHECK:STDOUT:     %.loc26_19.20: %facet_type.8a3 = converted constants.%Optional.49a, %facet_value.loc26_19.4 [concrete = constants.%facet_value.153]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %SelfDestruct.call.loc26_19.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc26_19.8)
// CHECK:STDOUT:     %.loc26_19.21: %facet_type.8a3 = converted constants.%as_type.dff, constants.%impl.elem1.277 [symbolic = %impl.elem1 (constants.%impl.elem1.277)]
// CHECK:STDOUT:     %.loc26_19.22: %facet_type.8a3 = converted constants.%as_type.dff, constants.%impl.elem1.277 [symbolic = %impl.elem1 (constants.%impl.elem1.277)]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %SelfDestruct.call.loc26_19.3: init %empty_tuple.type = call %SelfDestruct.specific_fn.3(%var)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc26_19.1(%self.param: ref %.cc0) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc26_19.2(%self.param: ref %MaybeUnformed.15c) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc26_19.3(%self.param: ref bool) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc26_19.4(%self.param: ref %struct_type.value.has_value.12d) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc26_19.1: ref bool = struct_access %self.param, element1
// CHECK:STDOUT:   %facet_value.loc26_19.1: %facet_type.8a3 = facet_value bool, (constants.%custom_witness.b80ef6.5, constants.%custom_witness.2cb555.3) [concrete = constants.%facet_value.ff1]
// CHECK:STDOUT:   %.loc26_19.2: %facet_type.8a3 = converted bool, %facet_value.loc26_19.1 [concrete = constants.%facet_value.ff1]
// CHECK:STDOUT:   %facet_value.loc26_19.2: %facet_type.8a3 = facet_value bool, (constants.%custom_witness.b80ef6.5, constants.%custom_witness.2cb555.3) [concrete = constants.%facet_value.ff1]
// CHECK:STDOUT:   %.loc26_19.3: %facet_type.8a3 = converted bool, %facet_value.loc26_19.2 [concrete = constants.%facet_value.ff1]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc26_19.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc26_19.1)
// CHECK:STDOUT:   %.loc26_19.4: ref %MaybeUnformed.15c = struct_access %self.param, element0
// CHECK:STDOUT:   %facet_value.loc26_19.3: %facet_type.8a3 = facet_value constants.%MaybeUnformed.15c, (constants.%custom_witness.b80ef6.4, constants.%custom_witness.2cb555.5) [concrete = constants.%facet_value.f6f]
// CHECK:STDOUT:   %.loc26_19.5: %facet_type.8a3 = converted constants.%MaybeUnformed.15c, %facet_value.loc26_19.3 [concrete = constants.%facet_value.f6f]
// CHECK:STDOUT:   %facet_value.loc26_19.4: %facet_type.8a3 = facet_value constants.%MaybeUnformed.15c, (constants.%custom_witness.b80ef6.4, constants.%custom_witness.2cb555.5) [concrete = constants.%facet_value.f6f]
// CHECK:STDOUT:   %.loc26_19.6: %facet_type.8a3 = converted constants.%MaybeUnformed.15c, %facet_value.loc26_19.4 [concrete = constants.%facet_value.f6f]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc26_19.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc26_19.4)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyImpl.WithSelf.Op.loc26_19.1(%self.param: ref bool) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyImpl.WithSelf.Op.loc26_19.2(%self.param: ref %.cc0) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyImpl.WithSelf.Op.loc26_19.3(%self.param: ref %MaybeUnformed.15c) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc26_19.5(%self.param: ref %DefaultOptionalStorage.877) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %has_value.ref: %DefaultOptionalStorage.elem.718 = name_ref has_value, imports.%Core.import_ref.1cb [concrete = imports.%field_decl.255]
// CHECK:STDOUT:   %.loc26_19.1: ref bool = class_element_access %self.param, element1
// CHECK:STDOUT:   %facet_value.loc26_19.1: %facet_type.8a3 = facet_value bool, (constants.%custom_witness.b80ef6.5, constants.%custom_witness.2cb555.3) [concrete = constants.%facet_value.ff1]
// CHECK:STDOUT:   %.loc26_19.2: %facet_type.8a3 = converted bool, %facet_value.loc26_19.1 [concrete = constants.%facet_value.ff1]
// CHECK:STDOUT:   %facet_value.loc26_19.2: %facet_type.8a3 = facet_value bool, (constants.%custom_witness.b80ef6.5, constants.%custom_witness.2cb555.3) [concrete = constants.%facet_value.ff1]
// CHECK:STDOUT:   %.loc26_19.3: %facet_type.8a3 = converted bool, %facet_value.loc26_19.2 [concrete = constants.%facet_value.ff1]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc26_19.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc26_19.1)
// CHECK:STDOUT:   %value.ref: %DefaultOptionalStorage.elem.e01 = name_ref value, imports.%Core.import_ref.1c5 [concrete = imports.%field_decl.649]
// CHECK:STDOUT:   %.loc26_19.4: ref %MaybeUnformed.15c = class_element_access %self.param, element0
// CHECK:STDOUT:   %facet_value.loc26_19.3: %facet_type.8a3 = facet_value constants.%MaybeUnformed.15c, (constants.%custom_witness.b80ef6.4, constants.%custom_witness.2cb555.5) [concrete = constants.%facet_value.f6f]
// CHECK:STDOUT:   %.loc26_19.5: %facet_type.8a3 = converted constants.%MaybeUnformed.15c, %facet_value.loc26_19.3 [concrete = constants.%facet_value.f6f]
// CHECK:STDOUT:   %facet_value.loc26_19.4: %facet_type.8a3 = facet_value constants.%MaybeUnformed.15c, (constants.%custom_witness.b80ef6.4, constants.%custom_witness.2cb555.5) [concrete = constants.%facet_value.f6f]
// CHECK:STDOUT:   %.loc26_19.6: %facet_type.8a3 = converted constants.%MaybeUnformed.15c, %facet_value.loc26_19.4 [concrete = constants.%facet_value.f6f]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc26_19.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc26_19.4)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc26_19.6(%self.param: ref %Optional.49a) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyImpl.WithSelf.Op.loc26_19.4(%self.param: ref %struct_type.value.has_value.12d) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyImpl.WithSelf.Op.loc26_19.5(%self.param: ref %DefaultOptionalStorage.877) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyImpl.WithSelf.Op.loc26_19.6(%self.param: ref %Optional.49a) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestRangeFor(%m.param: ref %MutableRange) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %i.patt: %pattern_type.fb7 = value_binding_pattern i [concrete = constants.%i.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %m.ref: ref %MutableRange = name_ref m, %m
// CHECK:STDOUT:   %impl.elem2: %.131 = impl_witness_access constants.%Iterate.impl_witness.8fd, element2 [concrete = constants.%T.as_type.as.Iterate.impl.NewCursor.7bd]
// CHECK:STDOUT:   %bound_method.loc68_19.1: <bound method> = bound_method %m.ref, %impl.elem2
// CHECK:STDOUT:   %facet_value.loc68_19.1: %CppRange_where.type.d682d6.1 = facet_value constants.%MutableRange, (constants.%custom_witness.b80ef6.2, constants.%custom_witness.2cb555.2, constants.%Copy.impl_witness.0d6, constants.%custom_witness.902, constants.%custom_witness.f01, constants.%custom_witness.2cb555.9, constants.%custom_witness.05b, constants.%custom_witness.491, constants.%custom_witness.ed8) [concrete = constants.%facet_value.8a6]
// CHECK:STDOUT:   %.loc68_19.1: %CppRange_where.type.d682d6.1 = converted constants.%MutableRange, %facet_value.loc68_19.1 [concrete = constants.%facet_value.8a6]
// CHECK:STDOUT:   %facet_value.loc68_19.2: %CppRange_where.type.d682d6.1 = facet_value constants.%MutableRange, (constants.%custom_witness.b80ef6.2, constants.%custom_witness.2cb555.2, constants.%Copy.impl_witness.0d6, constants.%custom_witness.902, constants.%custom_witness.f01, constants.%custom_witness.2cb555.9, constants.%custom_witness.05b, constants.%custom_witness.491, constants.%custom_witness.ed8) [concrete = constants.%facet_value.8a6]
// CHECK:STDOUT:   %.loc68_19.2: %CppRange_where.type.d682d6.1 = converted constants.%MutableRange, %facet_value.loc68_19.2 [concrete = constants.%facet_value.8a6]
// CHECK:STDOUT:   %specific_fn.loc68_19.1: <specific function> = specific_function %impl.elem2, @T.as_type.as.Iterate.impl.NewCursor(constants.%facet_value.8a6) [concrete = constants.%T.as_type.as.Iterate.impl.NewCursor.specific_fn]
// CHECK:STDOUT:   %bound_method.loc68_19.2: <bound method> = bound_method %m.ref, %specific_fn.loc68_19.1
// CHECK:STDOUT:   %var: ref %tuple.type.dd0 = var_storage invalid
// CHECK:STDOUT:   %.loc68_18.1: %MutableRange = acquire_value %m.ref
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.call: init %tuple.type.dd0 to %var = call %bound_method.loc68_19.2(%.loc68_18.1)
// CHECK:STDOUT:   assign %var, %T.as_type.as.Iterate.impl.NewCursor.call
// CHECK:STDOUT:   br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT: !for.next:
// CHECK:STDOUT:   %impl.elem3: %.9fd = impl_witness_access constants.%Iterate.impl_witness.8fd, element3 [concrete = constants.%T.as_type.as.Iterate.impl.Next.b0d]
// CHECK:STDOUT:   %bound_method.loc68_19.3: <bound method> = bound_method %m.ref, %impl.elem3
// CHECK:STDOUT:   %facet_value.loc68_19.3: %CppRange_where.type.d682d6.1 = facet_value constants.%MutableRange, (constants.%custom_witness.b80ef6.2, constants.%custom_witness.2cb555.2, constants.%Copy.impl_witness.0d6, constants.%custom_witness.902, constants.%custom_witness.f01, constants.%custom_witness.2cb555.9, constants.%custom_witness.05b, constants.%custom_witness.491, constants.%custom_witness.ed8) [concrete = constants.%facet_value.8a6]
// CHECK:STDOUT:   %.loc68_19.3: %CppRange_where.type.d682d6.1 = converted constants.%MutableRange, %facet_value.loc68_19.3 [concrete = constants.%facet_value.8a6]
// CHECK:STDOUT:   %facet_value.loc68_19.4: %CppRange_where.type.d682d6.1 = facet_value constants.%MutableRange, (constants.%custom_witness.b80ef6.2, constants.%custom_witness.2cb555.2, constants.%Copy.impl_witness.0d6, constants.%custom_witness.902, constants.%custom_witness.f01, constants.%custom_witness.2cb555.9, constants.%custom_witness.05b, constants.%custom_witness.491, constants.%custom_witness.ed8) [concrete = constants.%facet_value.8a6]
// CHECK:STDOUT:   %.loc68_19.4: %CppRange_where.type.d682d6.1 = converted constants.%MutableRange, %facet_value.loc68_19.4 [concrete = constants.%facet_value.8a6]
// CHECK:STDOUT:   %specific_fn.loc68_19.2: <specific function> = specific_function %impl.elem3, @T.as_type.as.Iterate.impl.Next(constants.%facet_value.8a6) [concrete = constants.%T.as_type.as.Iterate.impl.Next.specific_fn]
// CHECK:STDOUT:   %bound_method.loc68_19.4: <bound method> = bound_method %m.ref, %specific_fn.loc68_19.2
// CHECK:STDOUT:   %.loc68_19.5: ref %Optional.49a = temporary_storage
// CHECK:STDOUT:   %.loc68_18.2: %MutableRange = acquire_value %m.ref
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.call: init %Optional.49a to %.loc68_19.5 = call %bound_method.loc68_19.4(%.loc68_18.2, %var)
// CHECK:STDOUT:   %.loc68_19.6: ref %Optional.49a = temporary %.loc68_19.5, %T.as_type.as.Iterate.impl.Next.call
// CHECK:STDOUT:   %.loc68_19.7: %Optional.HasValue.type.79d = specific_constant imports.%Core.import_ref.bb6, @Optional(constants.%OptionalStorage.facet.424) [concrete = constants.%Optional.HasValue.718]
// CHECK:STDOUT:   %HasValue.ref: %Optional.HasValue.type.79d = name_ref HasValue, %.loc68_19.7 [concrete = constants.%Optional.HasValue.718]
// CHECK:STDOUT:   %Optional.HasValue.bound: <bound method> = bound_method %.loc68_19.6, %HasValue.ref
// CHECK:STDOUT:   %Optional.HasValue.specific_fn: <specific function> = specific_function %HasValue.ref, @Optional.HasValue(constants.%OptionalStorage.facet.424) [concrete = constants.%Optional.HasValue.specific_fn.409]
// CHECK:STDOUT:   %bound_method.loc68_19.5: <bound method> = bound_method %.loc68_19.6, %Optional.HasValue.specific_fn
// CHECK:STDOUT:   %.loc68_19.8: %Optional.49a = acquire_value %.loc68_19.6
// CHECK:STDOUT:   %Optional.HasValue.call: init bool = call %bound_method.loc68_19.5(%.loc68_19.8)
// CHECK:STDOUT:   %.loc68_19.9: bool = value_of_initializer %Optional.HasValue.call
// CHECK:STDOUT:   %.loc68_19.10: bool = converted %Optional.HasValue.call, %.loc68_19.9
// CHECK:STDOUT:   if %.loc68_19.10 br !for.body else br !for.done
// CHECK:STDOUT:
// CHECK:STDOUT: !for.body:
// CHECK:STDOUT:   %.loc68_19.11: %Optional.Get.type.3ba = specific_constant imports.%Core.import_ref.dab, @Optional(constants.%OptionalStorage.facet.424) [concrete = constants.%Optional.Get.a7f]
// CHECK:STDOUT:   %Get.ref: %Optional.Get.type.3ba = name_ref Get, %.loc68_19.11 [concrete = constants.%Optional.Get.a7f]
// CHECK:STDOUT:   %Optional.Get.bound: <bound method> = bound_method %.loc68_19.6, %Get.ref
// CHECK:STDOUT:   %Optional.Get.specific_fn: <specific function> = specific_function %Get.ref, @Optional.Get(constants.%OptionalStorage.facet.424) [concrete = constants.%Optional.Get.specific_fn.f9b]
// CHECK:STDOUT:   %bound_method.loc68_19.6: <bound method> = bound_method %.loc68_19.6, %Optional.Get.specific_fn
// CHECK:STDOUT:   %.loc68_19.12: %Optional.49a = acquire_value %.loc68_19.6
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
// CHECK:STDOUT:   %facet_value.loc68_19.5: %facet_type.8a3 = facet_value constants.%Optional.49a, (constants.%custom_witness.b80ef6.8, constants.%custom_witness.2cb555.8) [concrete = constants.%facet_value.153]
// CHECK:STDOUT:   %.loc68_19.15: %facet_type.8a3 = converted constants.%Optional.49a, %facet_value.loc68_19.5 [concrete = constants.%facet_value.153]
// CHECK:STDOUT:   %facet_value.loc68_19.6: %facet_type.8a3 = facet_value constants.%Optional.49a, (constants.%custom_witness.b80ef6.8, constants.%custom_witness.2cb555.8) [concrete = constants.%facet_value.153]
// CHECK:STDOUT:   %.loc68_19.16: %facet_type.8a3 = converted constants.%Optional.49a, %facet_value.loc68_19.6 [concrete = constants.%facet_value.153]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc68_19.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc68_19.6)
// CHECK:STDOUT:   br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT: !for.done:
// CHECK:STDOUT:   %facet_value.loc68_19.7: %facet_type.8a3 = facet_value constants.%Optional.49a, (constants.%custom_witness.b80ef6.8, constants.%custom_witness.2cb555.8) [concrete = constants.%facet_value.153]
// CHECK:STDOUT:   %.loc68_19.17: %facet_type.8a3 = converted constants.%Optional.49a, %facet_value.loc68_19.7 [concrete = constants.%facet_value.153]
// CHECK:STDOUT:   %facet_value.loc68_19.8: %facet_type.8a3 = facet_value constants.%Optional.49a, (constants.%custom_witness.b80ef6.8, constants.%custom_witness.2cb555.8) [concrete = constants.%facet_value.153]
// CHECK:STDOUT:   %.loc68_19.18: %facet_type.8a3 = converted constants.%Optional.49a, %facet_value.loc68_19.8 [concrete = constants.%facet_value.153]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc68_19.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc68_19.6)
// CHECK:STDOUT:   %facet_value.loc68_19.9: %facet_type.8a3 = facet_value constants.%tuple.type.dd0, (constants.%custom_witness.b80ef6.9, constants.%custom_witness.2cb555.10) [concrete = constants.%facet_value.a79]
// CHECK:STDOUT:   %.loc68_19.19: %facet_type.8a3 = converted constants.%tuple.type.dd0, %facet_value.loc68_19.9 [concrete = constants.%facet_value.a79]
// CHECK:STDOUT:   %facet_value.loc68_19.10: %facet_type.8a3 = facet_value constants.%tuple.type.dd0, (constants.%custom_witness.b80ef6.9, constants.%custom_witness.2cb555.10) [concrete = constants.%facet_value.a79]
// CHECK:STDOUT:   %.loc68_19.20: %facet_type.8a3 = converted constants.%tuple.type.dd0, %facet_value.loc68_19.10 [concrete = constants.%facet_value.a79]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc68_19.3: init %empty_tuple.type = call %SelfDestruct.specific_fn.3(%var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestIterate(constants.%R) {
// CHECK:STDOUT:   %R.patt.loc22_17.2 => constants.%R.patt
// CHECK:STDOUT:   %R.loc22_17.1 => constants.%R
// CHECK:STDOUT:   %R.as_type.loc22_61.1 => constants.%R.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.df6
// CHECK:STDOUT:   %r.param_patt.loc22_59.2 => constants.%r.param_patt.7c5
// CHECK:STDOUT:   %r.patt.loc22_59.2 => constants.%r.patt.598
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestIterate(constants.%facet_value.a70) {
// CHECK:STDOUT:   %R.patt.loc22_17.2 => constants.%R.patt
// CHECK:STDOUT:   %R.loc22_17.1 => constants.%facet_value.a70
// CHECK:STDOUT:   %R.as_type.loc22_61.1 => constants.%MutableRange
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.992
// CHECK:STDOUT:   %r.param_patt.loc22_59.2 => constants.%r.param_patt.aeb
// CHECK:STDOUT:   %r.patt.loc22_59.2 => constants.%r.patt.892
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc22 => constants.%complete_type.357
// CHECK:STDOUT:   %Iterate.lookup_impl_witness => constants.%Iterate.impl_witness.8fd
// CHECK:STDOUT:   %Iterate.facet.loc26_19.5 => constants.%Iterate.facet.c39
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type => constants.%Iterate.WithSelf.NewCursor.type.582
// CHECK:STDOUT:   %.loc26_19.23 => constants.%.131
// CHECK:STDOUT:   %impl.elem2.loc26_19.2 => constants.%T.as_type.as.Iterate.impl.NewCursor.7bd
// CHECK:STDOUT:   %specific_impl_fn.loc26_19.3 => constants.%T.as_type.as.Iterate.impl.NewCursor.specific_fn
// CHECK:STDOUT:   %impl.elem1 => constants.%facet_value.a79
// CHECK:STDOUT:   %as_type => constants.%tuple.type.dd0
// CHECK:STDOUT:   %require_complete.1 => constants.%complete_type.843
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type => constants.%Iterate.WithSelf.Next.type.12b
// CHECK:STDOUT:   %.loc26_19.24 => constants.%.9fd
// CHECK:STDOUT:   %impl.elem3.loc26_19.2 => constants.%T.as_type.as.Iterate.impl.Next.b0d
// CHECK:STDOUT:   %specific_impl_fn.loc26_19.4 => constants.%T.as_type.as.Iterate.impl.Next.specific_fn
// CHECK:STDOUT:   %SelfDestruct.specific_fn.5 => constants.%SelfDestruct.specific_fn.c79
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- adl_returns_different_types.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %Iterate.type: type = facet_type <@Iterate> [concrete]
// CHECK:STDOUT:   %OptionalStorage.type: type = facet_type <@OptionalStorage> [concrete]
// CHECK:STDOUT:   %facet_type.d61: type = facet_type <@SubobjectDestroy & @DestroyImpl & @Copy> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %T.220: %OptionalStorage.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Optional.Get.type.bb9: type = fn_type @Optional.Get, @Optional(%T.220) [symbolic]
// CHECK:STDOUT:   %Optional.Get.fa1: %Optional.Get.type.bb9 = struct_value () [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.type.e9e: type = fn_type @Optional.HasValue, @Optional(%T.220) [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.6cd: %Optional.HasValue.type.e9e = struct_value () [symbolic]
// CHECK:STDOUT:   %facet_type.8a3: type = facet_type <@SubobjectDestroy & @DestroyImpl> [concrete]
// CHECK:STDOUT:   %T.9ab: %facet_type.d61 = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.19c: type = facet_access_type %T.9ab [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.lookup_impl_witness.855: <witness> = lookup_impl_witness %T.9ab, @SubobjectDestroy [symbolic]
// CHECK:STDOUT:   %DestroyImpl.lookup_impl_witness.ff9: <witness> = lookup_impl_witness %T.9ab, @DestroyImpl [symbolic]
// CHECK:STDOUT:   %facet_value.b35: %facet_type.8a3 = facet_value %T.as_type.19c, (%SubobjectDestroy.lookup_impl_witness.855, %DestroyImpl.lookup_impl_witness.ff9) [symbolic]
// CHECK:STDOUT:   %MaybeUnformed.2cf: type = class_type @MaybeUnformed, @MaybeUnformed(%facet_value.b35) [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.7db: type = class_type @DefaultOptionalStorage, @DefaultOptionalStorage(%T.9ab) [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.elem.98d: type = unbound_element_type %DefaultOptionalStorage.7db, bool [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.elem.b5a: type = unbound_element_type %DefaultOptionalStorage.7db, %MaybeUnformed.2cf [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Get.type.b32: type = fn_type @T.as_type.as.OptionalStorage.impl.Get, @T.as_type.as.OptionalStorage.impl(%T.9ab) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Get.8b0: %T.as_type.as.OptionalStorage.impl.Get.type.b32 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Has.type.8e5: type = fn_type @T.as_type.as.OptionalStorage.impl.Has, @T.as_type.as.OptionalStorage.impl(%T.9ab) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Has.2d5: %T.as_type.as.OptionalStorage.impl.Has.type.8e5 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Some.type.61d: type = fn_type @T.as_type.as.OptionalStorage.impl.Some, @T.as_type.as.OptionalStorage.impl(%T.9ab) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Some.ba7: %T.as_type.as.OptionalStorage.impl.Some.type.61d = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.None.type.a9f: type = fn_type @T.as_type.as.OptionalStorage.impl.None, @T.as_type.as.OptionalStorage.impl(%T.9ab) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.None.083: %T.as_type.as.OptionalStorage.impl.None.type.a9f = struct_value () [symbolic]
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
// CHECK:STDOUT:   %pattern_type.c43: type = pattern_type %ValueType [concrete]
// CHECK:STDOUT:   %ValueType.Op.type.4c0699.1: type = fn_type @ValueType.Op.1 [concrete]
// CHECK:STDOUT:   %ValueType.Op.4ec7f2.1: %ValueType.Op.type.4c0699.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.0a1: <witness> = custom_witness (%ValueType.Op.4ec7f2.1), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.type.b77747.3: type = fn_type @DestroyImpl.WithSelf.Op.4 [concrete]
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.25aa57.3: %DestroyImpl.WithSelf.Op.type.b77747.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.2cb555.2: <witness> = custom_witness (%DestroyImpl.WithSelf.Op.25aa57.3), @DestroyImpl [concrete]
// CHECK:STDOUT:   %const.dd6: type = const_type %ValueType [concrete]
// CHECK:STDOUT:   %ptr.69c: type = ptr_type %const.dd6 [concrete]
// CHECK:STDOUT:   %ptr.233: type = ptr_type %ValueType [concrete]
// CHECK:STDOUT:   %ValueType.Op.type.4c0699.2: type = fn_type @ValueType.Op.2 [concrete]
// CHECK:STDOUT:   %ValueType.Op.4ec7f2.2: %ValueType.Op.type.4c0699.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.202: <witness> = custom_witness (%ValueType.Op.4ec7f2.2), @Copy [concrete]
// CHECK:STDOUT:   %facet_value.a23: %facet_type.d61 = facet_value %ValueType, (%custom_witness.0a1, %custom_witness.2cb555.2, %custom_witness.202) [concrete]
// CHECK:STDOUT:   %Iterate.lookup_impl_witness.1d8: <witness> = lookup_impl_witness %.Self.500, @Iterate [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.8a4: %facet_type.d61 = impl_witness_access %Iterate.lookup_impl_witness.1d8, element0 [symbolic_self]
// CHECK:STDOUT:   %Iterate_where.type.b58: type = facet_type <@Iterate where %impl.elem0.8a4 = %facet_value.a23> [concrete]
// CHECK:STDOUT:   %pattern_type.455: type = pattern_type %Iterate_where.type.b58 [concrete]
// CHECK:STDOUT:   %R.patt: %pattern_type.455 = symbolic_binding_pattern R, 0 [symbolic]
// CHECK:STDOUT:   %R: %Iterate_where.type.b58 = symbolic_binding R, 0 [symbolic]
// CHECK:STDOUT:   %R.as_type: type = facet_access_type %R [symbolic]
// CHECK:STDOUT:   %pattern_type.cc4: type = pattern_type %R.as_type [symbolic]
// CHECK:STDOUT:   %r.param_patt.36e: %pattern_type.cc4 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %r.patt.021: %pattern_type.cc4 = wrapper_binding_pattern r, %r.param_patt.36e [symbolic]
// CHECK:STDOUT:   %i.patt: %pattern_type.c43 = value_binding_pattern i [concrete]
// CHECK:STDOUT:   %Iterate.lookup_impl_witness.fdd: <witness> = lookup_impl_witness %R, @Iterate [symbolic]
// CHECK:STDOUT:   %Iterate.facet.e0a: %Iterate.type = facet_value %R.as_type, (%Iterate.lookup_impl_witness.fdd) [symbolic]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.46d: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.e0a) [symbolic]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.9b0: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.e0a) [symbolic]
// CHECK:STDOUT:   %.adf: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.46d, %Iterate.facet.e0a [symbolic]
// CHECK:STDOUT:   %impl.elem2.b2b: %.adf = impl_witness_access %Iterate.lookup_impl_witness.fdd, element2 [symbolic]
// CHECK:STDOUT:   %impl.elem1.517: %facet_type.8a3 = impl_witness_access %Iterate.lookup_impl_witness.fdd, element1 [symbolic]
// CHECK:STDOUT:   %as_type.fe4: type = facet_access_type %impl.elem1.517 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.982: <specific function> = specific_impl_function %impl.elem2.b2b, @Iterate.WithSelf.NewCursor(%Iterate.facet.e0a) [symbolic]
// CHECK:STDOUT:   %.4d7: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.9b0, %Iterate.facet.e0a [symbolic]
// CHECK:STDOUT:   %impl.elem3.c22: %.4d7 = impl_witness_access %Iterate.lookup_impl_witness.fdd, element3 [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.712: type = class_type @DefaultOptionalStorage, @DefaultOptionalStorage(%facet_value.a23) [concrete]
// CHECK:STDOUT:   %OptionalStorage.impl_witness.763: <witness> = impl_witness imports.%OptionalStorage.impl_witness_table.968, @T.as_type.as.OptionalStorage.impl(%facet_value.a23) [concrete]
// CHECK:STDOUT:   %OptionalStorage.facet.e84: %OptionalStorage.type = facet_value %ValueType, (%OptionalStorage.impl_witness.763) [concrete]
// CHECK:STDOUT:   %Optional.a47: type = class_type @Optional, @Optional(%OptionalStorage.facet.e84) [concrete]
// CHECK:STDOUT:   %specific_impl_fn.ea6: <specific function> = specific_impl_function %impl.elem3.c22, @Iterate.WithSelf.Next(%Iterate.facet.e0a) [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.type.5b6: type = fn_type @Optional.HasValue, @Optional(%OptionalStorage.facet.e84) [concrete]
// CHECK:STDOUT:   %Optional.HasValue.032: %Optional.HasValue.type.5b6 = struct_value () [concrete]
// CHECK:STDOUT:   %Optional.Get.type.d0d: type = fn_type @Optional.Get, @Optional(%OptionalStorage.facet.e84) [concrete]
// CHECK:STDOUT:   %Optional.Get.2b4: %Optional.Get.type.d0d = struct_value () [concrete]
// CHECK:STDOUT:   %facet_value.594: %facet_type.8a3 = facet_value %ValueType, (%custom_witness.0a1, %custom_witness.2cb555.2) [concrete]
// CHECK:STDOUT:   %MaybeUnformed.5aa: type = class_type @MaybeUnformed, @MaybeUnformed(%facet_value.594) [concrete]
// CHECK:STDOUT:   %.be1: type = maybe_unformed_type %ValueType [concrete]
// CHECK:STDOUT:   %DefaultOptionalStorage.elem.0ae: type = unbound_element_type %DefaultOptionalStorage.712, %MaybeUnformed.5aa [concrete]
// CHECK:STDOUT:   %DefaultOptionalStorage.elem.a03: type = unbound_element_type %DefaultOptionalStorage.712, bool [concrete]
// CHECK:STDOUT:   %struct_type.value.has_value.607: type = struct_type {.value: %MaybeUnformed.5aa, .has_value: bool} [concrete]
// CHECK:STDOUT:   %Optional.HasValue.specific_fn.d86: <specific function> = specific_function %Optional.HasValue.032, @Optional.HasValue(%OptionalStorage.facet.e84) [concrete]
// CHECK:STDOUT:   %Optional.Get.specific_fn.a90: <specific function> = specific_function %Optional.Get.2b4, @Optional.Get(%OptionalStorage.facet.e84) [concrete]
// CHECK:STDOUT:   %operator_PlusEqual__carbon_thunk.type: type = fn_type @operator_PlusEqual__carbon_thunk [concrete]
// CHECK:STDOUT:   %operator_PlusEqual__carbon_thunk: %operator_PlusEqual__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.3: type = fn_type @SubobjectDestroy.WithSelf.Op.loc31_37.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.3: %SubobjectDestroy.WithSelf.Op.type.6870fd.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.3: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.3), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.4: type = fn_type @SubobjectDestroy.WithSelf.Op.loc31_37.3 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.4: %SubobjectDestroy.WithSelf.Op.type.6870fd.4 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.4: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.4), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.type.b77747.4: type = fn_type @DestroyImpl.WithSelf.Op.loc31_37.1 [concrete]
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.25aa57.4: %DestroyImpl.WithSelf.Op.type.b77747.4 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.2cb555.3: <witness> = custom_witness (%DestroyImpl.WithSelf.Op.25aa57.4), @DestroyImpl [concrete]
// CHECK:STDOUT:   %facet_value.ff1: %facet_type.8a3 = facet_value bool, (%custom_witness.b80ef6.4, %custom_witness.2cb555.3) [concrete]
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.type.b77747.6: type = fn_type @DestroyImpl.WithSelf.Op.loc31_37.3 [concrete]
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.25aa57.6: %DestroyImpl.WithSelf.Op.type.b77747.6 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.2cb555.5: <witness> = custom_witness (%DestroyImpl.WithSelf.Op.25aa57.6), @DestroyImpl [concrete]
// CHECK:STDOUT:   %facet_value.a18: %facet_type.8a3 = facet_value %MaybeUnformed.5aa, (%custom_witness.b80ef6.3, %custom_witness.2cb555.5) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.7: type = fn_type @SubobjectDestroy.WithSelf.Op.loc31_37.6 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.7: %SubobjectDestroy.WithSelf.Op.type.6870fd.7 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.7: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.7), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.type.b77747.9: type = fn_type @DestroyImpl.WithSelf.Op.loc31_37.6 [concrete]
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.25aa57.9: %DestroyImpl.WithSelf.Op.type.b77747.9 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.2cb555.8: <witness> = custom_witness (%DestroyImpl.WithSelf.Op.25aa57.9), @DestroyImpl [concrete]
// CHECK:STDOUT:   %facet_value.6ba: %facet_type.8a3 = facet_value %Optional.a47, (%custom_witness.b80ef6.7, %custom_witness.2cb555.8) [concrete]
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
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.type.b77747.11: type = fn_type @DestroyImpl.WithSelf.Op.6 [concrete]
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.25aa57.11: %DestroyImpl.WithSelf.Op.type.b77747.11 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.2cb555.9: <witness> = custom_witness (%DestroyImpl.WithSelf.Op.25aa57.11), @DestroyImpl [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.3f6dc2.2: type = fn_type @Iterator.Op.2 [concrete]
// CHECK:STDOUT:   %Iterator.Op.240e50.2: %Iterator.Op.type.3f6dc2.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.612b50.1: <witness> = custom_witness (%Iterator.Op.240e50.2), @SubobjectDestroy [concrete]
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
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.type.b77747.13: type = fn_type @DestroyImpl.WithSelf.Op.8 [concrete]
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.25aa57.13: %DestroyImpl.WithSelf.Op.type.b77747.13 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.2cb555.10: <witness> = custom_witness (%DestroyImpl.WithSelf.Op.25aa57.13), @DestroyImpl [concrete]
// CHECK:STDOUT:   %facet_value.dc2: %CppRange_where.type.d682d6.1 = facet_value %ConstRange, (%custom_witness.0a1, %custom_witness.2cb555.2, %custom_witness.202, %custom_witness.116, %custom_witness.612b50.1, %custom_witness.2cb555.9, %custom_witness.5be, %custom_witness.c31, %custom_witness.e21, %custom_witness.612b50.2, %custom_witness.2cb555.10) [concrete]
// CHECK:STDOUT:   %tuple.type.7a6: type = tuple_type (%Iterator, %Sentinel) [concrete]
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.type.b77747.14: type = fn_type @DestroyImpl.WithSelf.Op.9 [concrete]
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.25aa57.14: %DestroyImpl.WithSelf.Op.type.b77747.14 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.2cb555.11: <witness> = custom_witness (%DestroyImpl.WithSelf.Op.25aa57.14), @DestroyImpl [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.8: type = fn_type @SubobjectDestroy.WithSelf.Op.3 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.8: %SubobjectDestroy.WithSelf.Op.type.6870fd.8 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.8: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.8), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %facet_value.1c4: %facet_type.8a3 = facet_value %tuple.type.7a6, (%custom_witness.b80ef6.8, %custom_witness.2cb555.11) [concrete]
// CHECK:STDOUT:   %Iterate.impl_witness.c2b: <witness> = impl_witness imports.%Iterate.impl_witness_table.121, @T.as_type.as.Iterate.impl(%facet_value.dc2) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.type.b12: type = fn_type @T.as_type.as.Iterate.impl.NewCursor, @T.as_type.as.Iterate.impl(%facet_value.dc2) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.d0f: %T.as_type.as.Iterate.impl.NewCursor.type.b12 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.type.861: type = fn_type @T.as_type.as.Iterate.impl.Next, @T.as_type.as.Iterate.impl(%facet_value.dc2) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.fd6: %T.as_type.as.Iterate.impl.Next.type.861 = struct_value () [concrete]
// CHECK:STDOUT:   %Iterate.facet.7ba: %Iterate.type = facet_value %ConstRange, (%Iterate.impl_witness.c2b) [concrete]
// CHECK:STDOUT:   %facet_value.c7e: %Iterate_where.type.b58 = facet_value %ConstRange, (%Iterate.impl_witness.c2b) [concrete]
// CHECK:STDOUT:   %r.param_patt.ac5: %pattern_type.c1d = value_param_pattern [concrete]
// CHECK:STDOUT:   %r.patt.97f: %pattern_type.c1d = wrapper_binding_pattern r, %r.param_patt.ac5 [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.f65: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.7ba) [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.029: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.7ba) [concrete]
// CHECK:STDOUT:   %.894: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.f65, %Iterate.facet.7ba [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.specific_fn: <specific function> = specific_function %T.as_type.as.Iterate.impl.NewCursor.d0f, @T.as_type.as.Iterate.impl.NewCursor(%facet_value.dc2) [concrete]
// CHECK:STDOUT:   %.c31: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.029, %Iterate.facet.7ba [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.specific_fn: <specific function> = specific_function %T.as_type.as.Iterate.impl.Next.fd6, @T.as_type.as.Iterate.impl.Next(%facet_value.dc2) [concrete]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.23d: <specific function> = specific_function %SelfDestruct, @SelfDestruct(%facet_value.1c4) [concrete]
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
// CHECK:STDOUT:   %Core.import_ref.38a: type = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%DefaultOptionalStorage (constants.%DefaultOptionalStorage.7db)]
// CHECK:STDOUT:   %Core.import_ref.61e: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.None.type (%T.as_type.as.OptionalStorage.impl.None.type.a9f) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.None (constants.%T.as_type.as.OptionalStorage.impl.None.083)]
// CHECK:STDOUT:   %Core.import_ref.ca1: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Some.type (%T.as_type.as.OptionalStorage.impl.Some.type.61d) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Some (constants.%T.as_type.as.OptionalStorage.impl.Some.ba7)]
// CHECK:STDOUT:   %Core.import_ref.6ba4: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Has.type (%T.as_type.as.OptionalStorage.impl.Has.type.8e5) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Has (constants.%T.as_type.as.OptionalStorage.impl.Has.2d5)]
// CHECK:STDOUT:   %Core.import_ref.73d: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Get.type (%T.as_type.as.OptionalStorage.impl.Get.type.b32) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Get (constants.%T.as_type.as.OptionalStorage.impl.Get.8b0)]
// CHECK:STDOUT:   %Core.import_ref.778 = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], unloaded
// CHECK:STDOUT:   %OptionalStorage.impl_witness_table.968 = impl_witness_table (%Core.import_ref.38a, %Core.import_ref.61e, %Core.import_ref.ca1, %Core.import_ref.6ba4, %Core.import_ref.73d, %Core.import_ref.778), @T.as_type.as.OptionalStorage.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.1c5: @DefaultOptionalStorage.%DefaultOptionalStorage.elem.1 (%DefaultOptionalStorage.elem.b5a) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [concrete = %field_decl.649]
// CHECK:STDOUT:   %Core.import_ref.1cb: @DefaultOptionalStorage.%DefaultOptionalStorage.elem.2 (%DefaultOptionalStorage.elem.98d) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [concrete = %field_decl.255]
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
// CHECK:STDOUT:   %field_decl.255: @DefaultOptionalStorage.%DefaultOptionalStorage.elem.2 (%DefaultOptionalStorage.elem.98d) = field_decl has_value, element1, %Core.import_ref.b0d in [concrete] {}
// CHECK:STDOUT:   %Core.import_ref.4cb = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %field_decl.649: @DefaultOptionalStorage.%DefaultOptionalStorage.elem.1 (%DefaultOptionalStorage.elem.b5a) = field_decl value, element0, %Core.import_ref.4cb in [concrete] {}
// CHECK:STDOUT:   %ConstRange.decl: type = class_decl @ConstRange [concrete = constants.%ConstRange] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @TestIterate(%R.loc27_17.2: %Iterate_where.type.b58) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Iterate.lookup_impl_witness: <witness> = lookup_impl_witness %R.loc27_17.1, @Iterate [symbolic = %Iterate.lookup_impl_witness (constants.%Iterate.lookup_impl_witness.fdd)]
// CHECK:STDOUT:   %Iterate.facet.loc31_37.5: %Iterate.type = facet_value %R.as_type.loc27_79.1, (%Iterate.lookup_impl_witness) [symbolic = %Iterate.facet.loc31_37.5 (constants.%Iterate.facet.e0a)]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.loc31_37.5) [symbolic = %Iterate.WithSelf.NewCursor.type (constants.%Iterate.WithSelf.NewCursor.type.46d)]
// CHECK:STDOUT:   %.loc31_37.26: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type, %Iterate.facet.loc31_37.5 [symbolic = %.loc31_37.26 (constants.%.adf)]
// CHECK:STDOUT:   %impl.elem2.loc31_37.2: @TestIterate.%.loc31_37.26 (%.adf) = impl_witness_access %Iterate.lookup_impl_witness, element2 [symbolic = %impl.elem2.loc31_37.2 (constants.%impl.elem2.b2b)]
// CHECK:STDOUT:   %specific_impl_fn.loc31_37.3: <specific function> = specific_impl_function %impl.elem2.loc31_37.2, @Iterate.WithSelf.NewCursor(%Iterate.facet.loc31_37.5) [symbolic = %specific_impl_fn.loc31_37.3 (constants.%specific_impl_fn.982)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.loc31_37.5) [symbolic = %Iterate.WithSelf.Next.type (constants.%Iterate.WithSelf.Next.type.9b0)]
// CHECK:STDOUT:   %.loc31_37.27: type = fn_type_with_self_type %Iterate.WithSelf.Next.type, %Iterate.facet.loc31_37.5 [symbolic = %.loc31_37.27 (constants.%.4d7)]
// CHECK:STDOUT:   %impl.elem3.loc31_37.2: @TestIterate.%.loc31_37.27 (%.4d7) = impl_witness_access %Iterate.lookup_impl_witness, element3 [symbolic = %impl.elem3.loc31_37.2 (constants.%impl.elem3.c22)]
// CHECK:STDOUT:   %specific_impl_fn.loc31_37.4: <specific function> = specific_impl_function %impl.elem3.loc31_37.2, @Iterate.WithSelf.Next(%Iterate.facet.loc31_37.5) [symbolic = %specific_impl_fn.loc31_37.4 (constants.%specific_impl_fn.ea6)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%r.param: @TestIterate.%R.as_type.loc27_79.1 (%R.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %i.patt: %pattern_type.c43 = value_binding_pattern i [concrete = constants.%i.patt]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %r.ref: @TestIterate.%R.as_type.loc27_79.1 (%R.as_type) = name_ref r, %r
// CHECK:STDOUT:     %impl.elem2.loc31_37.1: @TestIterate.%.loc31_37.26 (%.adf) = impl_witness_access constants.%Iterate.lookup_impl_witness.fdd, element2 [symbolic = %impl.elem2.loc31_37.2 (constants.%impl.elem2.b2b)]
// CHECK:STDOUT:     %bound_method.loc31_37.1: <bound method> = bound_method %r.ref, %impl.elem2.loc31_37.1
// CHECK:STDOUT:     %Iterate.facet.loc31_37.1: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.fdd) [symbolic = %Iterate.facet.loc31_37.5 (constants.%Iterate.facet.e0a)]
// CHECK:STDOUT:     %.loc31_37.1: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc31_37.1 [symbolic = %Iterate.facet.loc31_37.5 (constants.%Iterate.facet.e0a)]
// CHECK:STDOUT:     %Iterate.facet.loc31_37.2: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.fdd) [symbolic = %Iterate.facet.loc31_37.5 (constants.%Iterate.facet.e0a)]
// CHECK:STDOUT:     %.loc31_37.2: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc31_37.2 [symbolic = %Iterate.facet.loc31_37.5 (constants.%Iterate.facet.e0a)]
// CHECK:STDOUT:     %specific_impl_fn.loc31_37.1: <specific function> = specific_impl_function %impl.elem2.loc31_37.1, @Iterate.WithSelf.NewCursor(constants.%Iterate.facet.e0a) [symbolic = %specific_impl_fn.loc31_37.3 (constants.%specific_impl_fn.982)]
// CHECK:STDOUT:     %bound_method.loc31_37.2: <bound method> = bound_method %r.ref, %specific_impl_fn.loc31_37.1
// CHECK:STDOUT:     %var: ref @TestIterate.%as_type (%as_type.fe4) = var_storage invalid
// CHECK:STDOUT:     %Iterate.WithSelf.NewCursor.call: init @TestIterate.%as_type (%as_type.fe4) to %var = call %bound_method.loc31_37.2(%r.ref)
// CHECK:STDOUT:     assign %var, %Iterate.WithSelf.NewCursor.call
// CHECK:STDOUT:     br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.next:
// CHECK:STDOUT:     %impl.elem3.loc31_37.1: @TestIterate.%.loc31_37.27 (%.4d7) = impl_witness_access constants.%Iterate.lookup_impl_witness.fdd, element3 [symbolic = %impl.elem3.loc31_37.2 (constants.%impl.elem3.c22)]
// CHECK:STDOUT:     %bound_method.loc31_37.3: <bound method> = bound_method %r.ref, %impl.elem3.loc31_37.1
// CHECK:STDOUT:     %Iterate.facet.loc31_37.3: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.fdd) [symbolic = %Iterate.facet.loc31_37.5 (constants.%Iterate.facet.e0a)]
// CHECK:STDOUT:     %.loc31_37.3: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc31_37.3 [symbolic = %Iterate.facet.loc31_37.5 (constants.%Iterate.facet.e0a)]
// CHECK:STDOUT:     %Iterate.facet.loc31_37.4: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.fdd) [symbolic = %Iterate.facet.loc31_37.5 (constants.%Iterate.facet.e0a)]
// CHECK:STDOUT:     %.loc31_37.4: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc31_37.4 [symbolic = %Iterate.facet.loc31_37.5 (constants.%Iterate.facet.e0a)]
// CHECK:STDOUT:     %.loc31_37.5: %facet_type.8a3 = converted constants.%as_type.fe4, constants.%impl.elem1.517 [symbolic = %impl.elem1 (constants.%impl.elem1.517)]
// CHECK:STDOUT:     %.loc31_37.6: %facet_type.8a3 = converted constants.%as_type.fe4, constants.%impl.elem1.517 [symbolic = %impl.elem1 (constants.%impl.elem1.517)]
// CHECK:STDOUT:     %specific_impl_fn.loc31_37.2: <specific function> = specific_impl_function %impl.elem3.loc31_37.1, @Iterate.WithSelf.Next(constants.%Iterate.facet.e0a) [symbolic = %specific_impl_fn.loc31_37.4 (constants.%specific_impl_fn.ea6)]
// CHECK:STDOUT:     %bound_method.loc31_37.4: <bound method> = bound_method %r.ref, %specific_impl_fn.loc31_37.2
// CHECK:STDOUT:     %.loc31_37.7: ref %Optional.a47 = temporary_storage
// CHECK:STDOUT:     %Iterate.WithSelf.Next.call: init %Optional.a47 to %.loc31_37.7 = call %bound_method.loc31_37.4(%r.ref, %var)
// CHECK:STDOUT:     %.loc31_37.8: ref %Optional.a47 = temporary %.loc31_37.7, %Iterate.WithSelf.Next.call
// CHECK:STDOUT:     %.loc31_37.9: %Optional.HasValue.type.5b6 = specific_constant imports.%Core.import_ref.bb6, @Optional(constants.%OptionalStorage.facet.e84) [concrete = constants.%Optional.HasValue.032]
// CHECK:STDOUT:     %HasValue.ref: %Optional.HasValue.type.5b6 = name_ref HasValue, %.loc31_37.9 [concrete = constants.%Optional.HasValue.032]
// CHECK:STDOUT:     %Optional.HasValue.bound: <bound method> = bound_method %.loc31_37.8, %HasValue.ref
// CHECK:STDOUT:     %Optional.HasValue.specific_fn: <specific function> = specific_function %HasValue.ref, @Optional.HasValue(constants.%OptionalStorage.facet.e84) [concrete = constants.%Optional.HasValue.specific_fn.d86]
// CHECK:STDOUT:     %bound_method.loc31_37.5: <bound method> = bound_method %.loc31_37.8, %Optional.HasValue.specific_fn
// CHECK:STDOUT:     %.loc31_37.10: %Optional.a47 = acquire_value %.loc31_37.8
// CHECK:STDOUT:     %Optional.HasValue.call: init bool = call %bound_method.loc31_37.5(%.loc31_37.10)
// CHECK:STDOUT:     %.loc31_37.11: bool = value_of_initializer %Optional.HasValue.call
// CHECK:STDOUT:     %.loc31_37.12: bool = converted %Optional.HasValue.call, %.loc31_37.11
// CHECK:STDOUT:     if %.loc31_37.12 br !for.body else br !for.done
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.body:
// CHECK:STDOUT:     %.loc31_37.13: %Optional.Get.type.d0d = specific_constant imports.%Core.import_ref.dab, @Optional(constants.%OptionalStorage.facet.e84) [concrete = constants.%Optional.Get.2b4]
// CHECK:STDOUT:     %Get.ref: %Optional.Get.type.d0d = name_ref Get, %.loc31_37.13 [concrete = constants.%Optional.Get.2b4]
// CHECK:STDOUT:     %Optional.Get.bound: <bound method> = bound_method %.loc31_37.8, %Get.ref
// CHECK:STDOUT:     %Optional.Get.specific_fn: <specific function> = specific_function %Get.ref, @Optional.Get(constants.%OptionalStorage.facet.e84) [concrete = constants.%Optional.Get.specific_fn.a90]
// CHECK:STDOUT:     %bound_method.loc31_37.6: <bound method> = bound_method %.loc31_37.8, %Optional.Get.specific_fn
// CHECK:STDOUT:     %.loc31_37.14: ref %ValueType = temporary_storage
// CHECK:STDOUT:     %.loc31_37.15: %Optional.a47 = acquire_value %.loc31_37.8
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
// CHECK:STDOUT:     %facet_value.loc31_37.1: %facet_type.8a3 = facet_value constants.%ValueType, (constants.%custom_witness.0a1, constants.%custom_witness.2cb555.2) [concrete = constants.%facet_value.594]
// CHECK:STDOUT:     %.loc31_37.18: %facet_type.8a3 = converted constants.%ValueType, %facet_value.loc31_37.1 [concrete = constants.%facet_value.594]
// CHECK:STDOUT:     %facet_value.loc31_37.2: %facet_type.8a3 = facet_value constants.%ValueType, (constants.%custom_witness.0a1, constants.%custom_witness.2cb555.2) [concrete = constants.%facet_value.594]
// CHECK:STDOUT:     %.loc31_37.19: %facet_type.8a3 = converted constants.%ValueType, %facet_value.loc31_37.2 [concrete = constants.%facet_value.594]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %SelfDestruct.call.loc31_37.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc31_37.16)
// CHECK:STDOUT:     %facet_value.loc31_37.3: %facet_type.8a3 = facet_value constants.%Optional.a47, (constants.%custom_witness.b80ef6.7, constants.%custom_witness.2cb555.8) [concrete = constants.%facet_value.6ba]
// CHECK:STDOUT:     %.loc31_37.20: %facet_type.8a3 = converted constants.%Optional.a47, %facet_value.loc31_37.3 [concrete = constants.%facet_value.6ba]
// CHECK:STDOUT:     %facet_value.loc31_37.4: %facet_type.8a3 = facet_value constants.%Optional.a47, (constants.%custom_witness.b80ef6.7, constants.%custom_witness.2cb555.8) [concrete = constants.%facet_value.6ba]
// CHECK:STDOUT:     %.loc31_37.21: %facet_type.8a3 = converted constants.%Optional.a47, %facet_value.loc31_37.4 [concrete = constants.%facet_value.6ba]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %SelfDestruct.call.loc31_37.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc31_37.8)
// CHECK:STDOUT:     br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.done:
// CHECK:STDOUT:     %facet_value.loc31_37.5: %facet_type.8a3 = facet_value constants.%Optional.a47, (constants.%custom_witness.b80ef6.7, constants.%custom_witness.2cb555.8) [concrete = constants.%facet_value.6ba]
// CHECK:STDOUT:     %.loc31_37.22: %facet_type.8a3 = converted constants.%Optional.a47, %facet_value.loc31_37.5 [concrete = constants.%facet_value.6ba]
// CHECK:STDOUT:     %facet_value.loc31_37.6: %facet_type.8a3 = facet_value constants.%Optional.a47, (constants.%custom_witness.b80ef6.7, constants.%custom_witness.2cb555.8) [concrete = constants.%facet_value.6ba]
// CHECK:STDOUT:     %.loc31_37.23: %facet_type.8a3 = converted constants.%Optional.a47, %facet_value.loc31_37.6 [concrete = constants.%facet_value.6ba]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %SelfDestruct.call.loc31_37.3: init %empty_tuple.type = call %SelfDestruct.specific_fn.3(%.loc31_37.8)
// CHECK:STDOUT:     %.loc31_37.24: %facet_type.8a3 = converted constants.%as_type.fe4, constants.%impl.elem1.517 [symbolic = %impl.elem1 (constants.%impl.elem1.517)]
// CHECK:STDOUT:     %.loc31_37.25: %facet_type.8a3 = converted constants.%as_type.fe4, constants.%impl.elem1.517 [symbolic = %impl.elem1 (constants.%impl.elem1.517)]
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
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc31_37.2(%self.param: ref %MaybeUnformed.5aa) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc31_37.3(%self.param: ref bool) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc31_37.4(%self.param: ref %struct_type.value.has_value.607) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc31_37.1: ref bool = struct_access %self.param, element1
// CHECK:STDOUT:   %facet_value.loc31_37.1: %facet_type.8a3 = facet_value bool, (constants.%custom_witness.b80ef6.4, constants.%custom_witness.2cb555.3) [concrete = constants.%facet_value.ff1]
// CHECK:STDOUT:   %.loc31_37.2: %facet_type.8a3 = converted bool, %facet_value.loc31_37.1 [concrete = constants.%facet_value.ff1]
// CHECK:STDOUT:   %facet_value.loc31_37.2: %facet_type.8a3 = facet_value bool, (constants.%custom_witness.b80ef6.4, constants.%custom_witness.2cb555.3) [concrete = constants.%facet_value.ff1]
// CHECK:STDOUT:   %.loc31_37.3: %facet_type.8a3 = converted bool, %facet_value.loc31_37.2 [concrete = constants.%facet_value.ff1]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc31_37.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc31_37.1)
// CHECK:STDOUT:   %.loc31_37.4: ref %MaybeUnformed.5aa = struct_access %self.param, element0
// CHECK:STDOUT:   %facet_value.loc31_37.3: %facet_type.8a3 = facet_value constants.%MaybeUnformed.5aa, (constants.%custom_witness.b80ef6.3, constants.%custom_witness.2cb555.5) [concrete = constants.%facet_value.a18]
// CHECK:STDOUT:   %.loc31_37.5: %facet_type.8a3 = converted constants.%MaybeUnformed.5aa, %facet_value.loc31_37.3 [concrete = constants.%facet_value.a18]
// CHECK:STDOUT:   %facet_value.loc31_37.4: %facet_type.8a3 = facet_value constants.%MaybeUnformed.5aa, (constants.%custom_witness.b80ef6.3, constants.%custom_witness.2cb555.5) [concrete = constants.%facet_value.a18]
// CHECK:STDOUT:   %.loc31_37.6: %facet_type.8a3 = converted constants.%MaybeUnformed.5aa, %facet_value.loc31_37.4 [concrete = constants.%facet_value.a18]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc31_37.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc31_37.4)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyImpl.WithSelf.Op.loc31_37.1(%self.param: ref bool) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyImpl.WithSelf.Op.loc31_37.2(%self.param: ref %.be1) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyImpl.WithSelf.Op.loc31_37.3(%self.param: ref %MaybeUnformed.5aa) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc31_37.5(%self.param: ref %DefaultOptionalStorage.712) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %has_value.ref: %DefaultOptionalStorage.elem.a03 = name_ref has_value, imports.%Core.import_ref.1cb [concrete = imports.%field_decl.255]
// CHECK:STDOUT:   %.loc31_37.1: ref bool = class_element_access %self.param, element1
// CHECK:STDOUT:   %facet_value.loc31_37.1: %facet_type.8a3 = facet_value bool, (constants.%custom_witness.b80ef6.4, constants.%custom_witness.2cb555.3) [concrete = constants.%facet_value.ff1]
// CHECK:STDOUT:   %.loc31_37.2: %facet_type.8a3 = converted bool, %facet_value.loc31_37.1 [concrete = constants.%facet_value.ff1]
// CHECK:STDOUT:   %facet_value.loc31_37.2: %facet_type.8a3 = facet_value bool, (constants.%custom_witness.b80ef6.4, constants.%custom_witness.2cb555.3) [concrete = constants.%facet_value.ff1]
// CHECK:STDOUT:   %.loc31_37.3: %facet_type.8a3 = converted bool, %facet_value.loc31_37.2 [concrete = constants.%facet_value.ff1]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc31_37.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc31_37.1)
// CHECK:STDOUT:   %value.ref: %DefaultOptionalStorage.elem.0ae = name_ref value, imports.%Core.import_ref.1c5 [concrete = imports.%field_decl.649]
// CHECK:STDOUT:   %.loc31_37.4: ref %MaybeUnformed.5aa = class_element_access %self.param, element0
// CHECK:STDOUT:   %facet_value.loc31_37.3: %facet_type.8a3 = facet_value constants.%MaybeUnformed.5aa, (constants.%custom_witness.b80ef6.3, constants.%custom_witness.2cb555.5) [concrete = constants.%facet_value.a18]
// CHECK:STDOUT:   %.loc31_37.5: %facet_type.8a3 = converted constants.%MaybeUnformed.5aa, %facet_value.loc31_37.3 [concrete = constants.%facet_value.a18]
// CHECK:STDOUT:   %facet_value.loc31_37.4: %facet_type.8a3 = facet_value constants.%MaybeUnformed.5aa, (constants.%custom_witness.b80ef6.3, constants.%custom_witness.2cb555.5) [concrete = constants.%facet_value.a18]
// CHECK:STDOUT:   %.loc31_37.6: %facet_type.8a3 = converted constants.%MaybeUnformed.5aa, %facet_value.loc31_37.4 [concrete = constants.%facet_value.a18]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc31_37.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc31_37.4)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc31_37.6(%self.param: ref %Optional.a47) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyImpl.WithSelf.Op.loc31_37.4(%self.param: ref %struct_type.value.has_value.607) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyImpl.WithSelf.Op.loc31_37.5(%self.param: ref %DefaultOptionalStorage.712) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyImpl.WithSelf.Op.loc31_37.6(%self.param: ref %Optional.a47) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestRangeFor(%c.param: %ConstRange) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %i.patt: %pattern_type.c43 = value_binding_pattern i [concrete = constants.%i.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c.ref: %ConstRange = name_ref c, %c
// CHECK:STDOUT:   %impl.elem2: %.894 = impl_witness_access constants.%Iterate.impl_witness.c2b, element2 [concrete = constants.%T.as_type.as.Iterate.impl.NewCursor.d0f]
// CHECK:STDOUT:   %bound_method.loc45_37.1: <bound method> = bound_method %c.ref, %impl.elem2
// CHECK:STDOUT:   %facet_value.loc45_37.1: %CppRange_where.type.d682d6.1 = facet_value constants.%ConstRange, (constants.%custom_witness.0a1, constants.%custom_witness.2cb555.2, constants.%custom_witness.202, constants.%custom_witness.116, constants.%custom_witness.612b50.1, constants.%custom_witness.2cb555.9, constants.%custom_witness.5be, constants.%custom_witness.c31, constants.%custom_witness.e21, constants.%custom_witness.612b50.2, constants.%custom_witness.2cb555.10) [concrete = constants.%facet_value.dc2]
// CHECK:STDOUT:   %.loc45_37.1: %CppRange_where.type.d682d6.1 = converted constants.%ConstRange, %facet_value.loc45_37.1 [concrete = constants.%facet_value.dc2]
// CHECK:STDOUT:   %facet_value.loc45_37.2: %CppRange_where.type.d682d6.1 = facet_value constants.%ConstRange, (constants.%custom_witness.0a1, constants.%custom_witness.2cb555.2, constants.%custom_witness.202, constants.%custom_witness.116, constants.%custom_witness.612b50.1, constants.%custom_witness.2cb555.9, constants.%custom_witness.5be, constants.%custom_witness.c31, constants.%custom_witness.e21, constants.%custom_witness.612b50.2, constants.%custom_witness.2cb555.10) [concrete = constants.%facet_value.dc2]
// CHECK:STDOUT:   %.loc45_37.2: %CppRange_where.type.d682d6.1 = converted constants.%ConstRange, %facet_value.loc45_37.2 [concrete = constants.%facet_value.dc2]
// CHECK:STDOUT:   %specific_fn.loc45_37.1: <specific function> = specific_function %impl.elem2, @T.as_type.as.Iterate.impl.NewCursor(constants.%facet_value.dc2) [concrete = constants.%T.as_type.as.Iterate.impl.NewCursor.specific_fn]
// CHECK:STDOUT:   %bound_method.loc45_37.2: <bound method> = bound_method %c.ref, %specific_fn.loc45_37.1
// CHECK:STDOUT:   %var: ref %tuple.type.7a6 = var_storage invalid
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.call: init %tuple.type.7a6 to %var = call %bound_method.loc45_37.2(%c.ref)
// CHECK:STDOUT:   assign %var, %T.as_type.as.Iterate.impl.NewCursor.call
// CHECK:STDOUT:   br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT: !for.next:
// CHECK:STDOUT:   %impl.elem3: %.c31 = impl_witness_access constants.%Iterate.impl_witness.c2b, element3 [concrete = constants.%T.as_type.as.Iterate.impl.Next.fd6]
// CHECK:STDOUT:   %bound_method.loc45_37.3: <bound method> = bound_method %c.ref, %impl.elem3
// CHECK:STDOUT:   %facet_value.loc45_37.3: %CppRange_where.type.d682d6.1 = facet_value constants.%ConstRange, (constants.%custom_witness.0a1, constants.%custom_witness.2cb555.2, constants.%custom_witness.202, constants.%custom_witness.116, constants.%custom_witness.612b50.1, constants.%custom_witness.2cb555.9, constants.%custom_witness.5be, constants.%custom_witness.c31, constants.%custom_witness.e21, constants.%custom_witness.612b50.2, constants.%custom_witness.2cb555.10) [concrete = constants.%facet_value.dc2]
// CHECK:STDOUT:   %.loc45_37.3: %CppRange_where.type.d682d6.1 = converted constants.%ConstRange, %facet_value.loc45_37.3 [concrete = constants.%facet_value.dc2]
// CHECK:STDOUT:   %facet_value.loc45_37.4: %CppRange_where.type.d682d6.1 = facet_value constants.%ConstRange, (constants.%custom_witness.0a1, constants.%custom_witness.2cb555.2, constants.%custom_witness.202, constants.%custom_witness.116, constants.%custom_witness.612b50.1, constants.%custom_witness.2cb555.9, constants.%custom_witness.5be, constants.%custom_witness.c31, constants.%custom_witness.e21, constants.%custom_witness.612b50.2, constants.%custom_witness.2cb555.10) [concrete = constants.%facet_value.dc2]
// CHECK:STDOUT:   %.loc45_37.4: %CppRange_where.type.d682d6.1 = converted constants.%ConstRange, %facet_value.loc45_37.4 [concrete = constants.%facet_value.dc2]
// CHECK:STDOUT:   %specific_fn.loc45_37.2: <specific function> = specific_function %impl.elem3, @T.as_type.as.Iterate.impl.Next(constants.%facet_value.dc2) [concrete = constants.%T.as_type.as.Iterate.impl.Next.specific_fn]
// CHECK:STDOUT:   %bound_method.loc45_37.4: <bound method> = bound_method %c.ref, %specific_fn.loc45_37.2
// CHECK:STDOUT:   %.loc45_37.5: ref %Optional.a47 = temporary_storage
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.call: init %Optional.a47 to %.loc45_37.5 = call %bound_method.loc45_37.4(%c.ref, %var)
// CHECK:STDOUT:   %.loc45_37.6: ref %Optional.a47 = temporary %.loc45_37.5, %T.as_type.as.Iterate.impl.Next.call
// CHECK:STDOUT:   %.loc45_37.7: %Optional.HasValue.type.5b6 = specific_constant imports.%Core.import_ref.bb6, @Optional(constants.%OptionalStorage.facet.e84) [concrete = constants.%Optional.HasValue.032]
// CHECK:STDOUT:   %HasValue.ref: %Optional.HasValue.type.5b6 = name_ref HasValue, %.loc45_37.7 [concrete = constants.%Optional.HasValue.032]
// CHECK:STDOUT:   %Optional.HasValue.bound: <bound method> = bound_method %.loc45_37.6, %HasValue.ref
// CHECK:STDOUT:   %Optional.HasValue.specific_fn: <specific function> = specific_function %HasValue.ref, @Optional.HasValue(constants.%OptionalStorage.facet.e84) [concrete = constants.%Optional.HasValue.specific_fn.d86]
// CHECK:STDOUT:   %bound_method.loc45_37.5: <bound method> = bound_method %.loc45_37.6, %Optional.HasValue.specific_fn
// CHECK:STDOUT:   %.loc45_37.8: %Optional.a47 = acquire_value %.loc45_37.6
// CHECK:STDOUT:   %Optional.HasValue.call: init bool = call %bound_method.loc45_37.5(%.loc45_37.8)
// CHECK:STDOUT:   %.loc45_37.9: bool = value_of_initializer %Optional.HasValue.call
// CHECK:STDOUT:   %.loc45_37.10: bool = converted %Optional.HasValue.call, %.loc45_37.9
// CHECK:STDOUT:   if %.loc45_37.10 br !for.body else br !for.done
// CHECK:STDOUT:
// CHECK:STDOUT: !for.body:
// CHECK:STDOUT:   %.loc45_37.11: %Optional.Get.type.d0d = specific_constant imports.%Core.import_ref.dab, @Optional(constants.%OptionalStorage.facet.e84) [concrete = constants.%Optional.Get.2b4]
// CHECK:STDOUT:   %Get.ref: %Optional.Get.type.d0d = name_ref Get, %.loc45_37.11 [concrete = constants.%Optional.Get.2b4]
// CHECK:STDOUT:   %Optional.Get.bound: <bound method> = bound_method %.loc45_37.6, %Get.ref
// CHECK:STDOUT:   %Optional.Get.specific_fn: <specific function> = specific_function %Get.ref, @Optional.Get(constants.%OptionalStorage.facet.e84) [concrete = constants.%Optional.Get.specific_fn.a90]
// CHECK:STDOUT:   %bound_method.loc45_37.6: <bound method> = bound_method %.loc45_37.6, %Optional.Get.specific_fn
// CHECK:STDOUT:   %.loc45_37.12: ref %ValueType = temporary_storage
// CHECK:STDOUT:   %.loc45_37.13: %Optional.a47 = acquire_value %.loc45_37.6
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
// CHECK:STDOUT:   %facet_value.loc45_37.5: %facet_type.8a3 = facet_value constants.%ValueType, (constants.%custom_witness.0a1, constants.%custom_witness.2cb555.2) [concrete = constants.%facet_value.594]
// CHECK:STDOUT:   %.loc45_37.16: %facet_type.8a3 = converted constants.%ValueType, %facet_value.loc45_37.5 [concrete = constants.%facet_value.594]
// CHECK:STDOUT:   %facet_value.loc45_37.6: %facet_type.8a3 = facet_value constants.%ValueType, (constants.%custom_witness.0a1, constants.%custom_witness.2cb555.2) [concrete = constants.%facet_value.594]
// CHECK:STDOUT:   %.loc45_37.17: %facet_type.8a3 = converted constants.%ValueType, %facet_value.loc45_37.6 [concrete = constants.%facet_value.594]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc45_37.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc45_37.14)
// CHECK:STDOUT:   %facet_value.loc45_37.7: %facet_type.8a3 = facet_value constants.%Optional.a47, (constants.%custom_witness.b80ef6.7, constants.%custom_witness.2cb555.8) [concrete = constants.%facet_value.6ba]
// CHECK:STDOUT:   %.loc45_37.18: %facet_type.8a3 = converted constants.%Optional.a47, %facet_value.loc45_37.7 [concrete = constants.%facet_value.6ba]
// CHECK:STDOUT:   %facet_value.loc45_37.8: %facet_type.8a3 = facet_value constants.%Optional.a47, (constants.%custom_witness.b80ef6.7, constants.%custom_witness.2cb555.8) [concrete = constants.%facet_value.6ba]
// CHECK:STDOUT:   %.loc45_37.19: %facet_type.8a3 = converted constants.%Optional.a47, %facet_value.loc45_37.8 [concrete = constants.%facet_value.6ba]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc45_37.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc45_37.6)
// CHECK:STDOUT:   br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT: !for.done:
// CHECK:STDOUT:   %facet_value.loc45_37.9: %facet_type.8a3 = facet_value constants.%Optional.a47, (constants.%custom_witness.b80ef6.7, constants.%custom_witness.2cb555.8) [concrete = constants.%facet_value.6ba]
// CHECK:STDOUT:   %.loc45_37.20: %facet_type.8a3 = converted constants.%Optional.a47, %facet_value.loc45_37.9 [concrete = constants.%facet_value.6ba]
// CHECK:STDOUT:   %facet_value.loc45_37.10: %facet_type.8a3 = facet_value constants.%Optional.a47, (constants.%custom_witness.b80ef6.7, constants.%custom_witness.2cb555.8) [concrete = constants.%facet_value.6ba]
// CHECK:STDOUT:   %.loc45_37.21: %facet_type.8a3 = converted constants.%Optional.a47, %facet_value.loc45_37.10 [concrete = constants.%facet_value.6ba]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc45_37.3: init %empty_tuple.type = call %SelfDestruct.specific_fn.3(%.loc45_37.6)
// CHECK:STDOUT:   %facet_value.loc45_37.11: %facet_type.8a3 = facet_value constants.%tuple.type.7a6, (constants.%custom_witness.b80ef6.8, constants.%custom_witness.2cb555.11) [concrete = constants.%facet_value.1c4]
// CHECK:STDOUT:   %.loc45_37.22: %facet_type.8a3 = converted constants.%tuple.type.7a6, %facet_value.loc45_37.11 [concrete = constants.%facet_value.1c4]
// CHECK:STDOUT:   %facet_value.loc45_37.12: %facet_type.8a3 = facet_value constants.%tuple.type.7a6, (constants.%custom_witness.b80ef6.8, constants.%custom_witness.2cb555.11) [concrete = constants.%facet_value.1c4]
// CHECK:STDOUT:   %.loc45_37.23: %facet_type.8a3 = converted constants.%tuple.type.7a6, %facet_value.loc45_37.12 [concrete = constants.%facet_value.1c4]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc45_37.4: init %empty_tuple.type = call %SelfDestruct.specific_fn.4(%var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestIterate(constants.%R) {
// CHECK:STDOUT:   %R.patt.loc27_17.2 => constants.%R.patt
// CHECK:STDOUT:   %R.loc27_17.1 => constants.%R
// CHECK:STDOUT:   %R.as_type.loc27_79.1 => constants.%R.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.cc4
// CHECK:STDOUT:   %r.param_patt.loc27_77.2 => constants.%r.param_patt.36e
// CHECK:STDOUT:   %r.patt.loc27_77.2 => constants.%r.patt.021
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestIterate(constants.%facet_value.c7e) {
// CHECK:STDOUT:   %R.patt.loc27_17.2 => constants.%R.patt
// CHECK:STDOUT:   %R.loc27_17.1 => constants.%facet_value.c7e
// CHECK:STDOUT:   %R.as_type.loc27_79.1 => constants.%ConstRange
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.c1d
// CHECK:STDOUT:   %r.param_patt.loc27_77.2 => constants.%r.param_patt.ac5
// CHECK:STDOUT:   %r.patt.loc27_77.2 => constants.%r.patt.97f
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc27 => constants.%complete_type.357
// CHECK:STDOUT:   %Iterate.lookup_impl_witness => constants.%Iterate.impl_witness.c2b
// CHECK:STDOUT:   %Iterate.facet.loc31_37.5 => constants.%Iterate.facet.7ba
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type => constants.%Iterate.WithSelf.NewCursor.type.f65
// CHECK:STDOUT:   %.loc31_37.26 => constants.%.894
// CHECK:STDOUT:   %impl.elem2.loc31_37.2 => constants.%T.as_type.as.Iterate.impl.NewCursor.d0f
// CHECK:STDOUT:   %specific_impl_fn.loc31_37.3 => constants.%T.as_type.as.Iterate.impl.NewCursor.specific_fn
// CHECK:STDOUT:   %impl.elem1 => constants.%facet_value.1c4
// CHECK:STDOUT:   %as_type => constants.%tuple.type.7a6
// CHECK:STDOUT:   %require_complete.1 => constants.%complete_type.cf0
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type => constants.%Iterate.WithSelf.Next.type.029
// CHECK:STDOUT:   %.loc31_37.27 => constants.%.c31
// CHECK:STDOUT:   %impl.elem3.loc31_37.2 => constants.%T.as_type.as.Iterate.impl.Next.fd6
// CHECK:STDOUT:   %specific_impl_fn.loc31_37.4 => constants.%T.as_type.as.Iterate.impl.Next.specific_fn
// CHECK:STDOUT:   %SelfDestruct.specific_fn.6 => constants.%SelfDestruct.specific_fn.23d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_adl_returns_different_types_mutable.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %Iterate.type: type = facet_type <@Iterate> [concrete]
// CHECK:STDOUT:   %OptionalStorage.type: type = facet_type <@OptionalStorage> [concrete]
// CHECK:STDOUT:   %facet_type.d61: type = facet_type <@SubobjectDestroy & @DestroyImpl & @Copy> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %T.220: %OptionalStorage.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Optional.Get.type.bb9: type = fn_type @Optional.Get, @Optional(%T.220) [symbolic]
// CHECK:STDOUT:   %Optional.Get.fa1: %Optional.Get.type.bb9 = struct_value () [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.type.e9e: type = fn_type @Optional.HasValue, @Optional(%T.220) [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.6cd: %Optional.HasValue.type.e9e = struct_value () [symbolic]
// CHECK:STDOUT:   %facet_type.8a3: type = facet_type <@SubobjectDestroy & @DestroyImpl> [concrete]
// CHECK:STDOUT:   %T.9ab: %facet_type.d61 = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.19c: type = facet_access_type %T.9ab [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.lookup_impl_witness.855: <witness> = lookup_impl_witness %T.9ab, @SubobjectDestroy [symbolic]
// CHECK:STDOUT:   %DestroyImpl.lookup_impl_witness.ff9: <witness> = lookup_impl_witness %T.9ab, @DestroyImpl [symbolic]
// CHECK:STDOUT:   %facet_value.b35: %facet_type.8a3 = facet_value %T.as_type.19c, (%SubobjectDestroy.lookup_impl_witness.855, %DestroyImpl.lookup_impl_witness.ff9) [symbolic]
// CHECK:STDOUT:   %MaybeUnformed.2cf: type = class_type @MaybeUnformed, @MaybeUnformed(%facet_value.b35) [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.7db: type = class_type @DefaultOptionalStorage, @DefaultOptionalStorage(%T.9ab) [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.elem.98d: type = unbound_element_type %DefaultOptionalStorage.7db, bool [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.elem.b5a: type = unbound_element_type %DefaultOptionalStorage.7db, %MaybeUnformed.2cf [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Get.type.b32: type = fn_type @T.as_type.as.OptionalStorage.impl.Get, @T.as_type.as.OptionalStorage.impl(%T.9ab) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Get.8b0: %T.as_type.as.OptionalStorage.impl.Get.type.b32 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Has.type.8e5: type = fn_type @T.as_type.as.OptionalStorage.impl.Has, @T.as_type.as.OptionalStorage.impl(%T.9ab) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Has.2d5: %T.as_type.as.OptionalStorage.impl.Has.type.8e5 = struct_value () [symbolic]
// CHECK:STDOUT:   %.Self.500: %Iterate.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %ValueType: type = class_type @ValueType [concrete]
// CHECK:STDOUT:   %pattern_type.d2d: type = pattern_type %ValueType [concrete]
// CHECK:STDOUT:   %ValueType.Op.type.db6aa9.1: type = fn_type @ValueType.Op.1 [concrete]
// CHECK:STDOUT:   %ValueType.Op.4590ed.1: %ValueType.Op.type.db6aa9.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.55b: <witness> = custom_witness (%ValueType.Op.4590ed.1), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.type.b77747.3: type = fn_type @DestroyImpl.WithSelf.Op.4 [concrete]
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.25aa57.3: %DestroyImpl.WithSelf.Op.type.b77747.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.2cb555.2: <witness> = custom_witness (%DestroyImpl.WithSelf.Op.25aa57.3), @DestroyImpl [concrete]
// CHECK:STDOUT:   %const.91f: type = const_type %ValueType [concrete]
// CHECK:STDOUT:   %ptr.5f4: type = ptr_type %const.91f [concrete]
// CHECK:STDOUT:   %ptr.9c6: type = ptr_type %ValueType [concrete]
// CHECK:STDOUT:   %ValueType.Op.type.db6aa9.2: type = fn_type @ValueType.Op.2 [concrete]
// CHECK:STDOUT:   %ValueType.Op.4590ed.2: %ValueType.Op.type.db6aa9.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.587: <witness> = custom_witness (%ValueType.Op.4590ed.2), @Copy [concrete]
// CHECK:STDOUT:   %facet_value.8de: %facet_type.d61 = facet_value %ValueType, (%custom_witness.55b, %custom_witness.2cb555.2, %custom_witness.587) [concrete]
// CHECK:STDOUT:   %Iterate.lookup_impl_witness.1d8: <witness> = lookup_impl_witness %.Self.500, @Iterate [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.8a4: %facet_type.d61 = impl_witness_access %Iterate.lookup_impl_witness.1d8, element0 [symbolic_self]
// CHECK:STDOUT:   %Iterate_where.type.f8a: type = facet_type <@Iterate where %impl.elem0.8a4 = %facet_value.8de> [concrete]
// CHECK:STDOUT:   %pattern_type.86b: type = pattern_type %Iterate_where.type.f8a [concrete]
// CHECK:STDOUT:   %R.patt: %pattern_type.86b = symbolic_binding_pattern R, 0 [symbolic]
// CHECK:STDOUT:   %R: %Iterate_where.type.f8a = symbolic_binding R, 0 [symbolic]
// CHECK:STDOUT:   %R.as_type: type = facet_access_type %R [symbolic]
// CHECK:STDOUT:   %pattern_type.002: type = pattern_type %R.as_type [symbolic]
// CHECK:STDOUT:   %r.param_patt: %pattern_type.002 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %r.patt: %pattern_type.002 = wrapper_binding_pattern r, %r.param_patt [symbolic]
// CHECK:STDOUT:   %i.patt: %pattern_type.d2d = value_binding_pattern i [concrete]
// CHECK:STDOUT:   %Iterate.lookup_impl_witness.a0e: <witness> = lookup_impl_witness %R, @Iterate [symbolic]
// CHECK:STDOUT:   %Iterate.facet.c1a: %Iterate.type = facet_value %R.as_type, (%Iterate.lookup_impl_witness.a0e) [symbolic]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.d83: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.c1a) [symbolic]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.1dc: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.c1a) [symbolic]
// CHECK:STDOUT:   %.54c: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.d83, %Iterate.facet.c1a [symbolic]
// CHECK:STDOUT:   %impl.elem2.e98: %.54c = impl_witness_access %Iterate.lookup_impl_witness.a0e, element2 [symbolic]
// CHECK:STDOUT:   %impl.elem1.de0: %facet_type.8a3 = impl_witness_access %Iterate.lookup_impl_witness.a0e, element1 [symbolic]
// CHECK:STDOUT:   %as_type.a37: type = facet_access_type %impl.elem1.de0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.bea: <specific function> = specific_impl_function %impl.elem2.e98, @Iterate.WithSelf.NewCursor(%Iterate.facet.c1a) [symbolic]
// CHECK:STDOUT:   %.6c6: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.1dc, %Iterate.facet.c1a [symbolic]
// CHECK:STDOUT:   %impl.elem3.017: %.6c6 = impl_witness_access %Iterate.lookup_impl_witness.a0e, element3 [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.d2b: type = class_type @DefaultOptionalStorage, @DefaultOptionalStorage(%facet_value.8de) [concrete]
// CHECK:STDOUT:   %OptionalStorage.impl_witness.1b6: <witness> = impl_witness imports.%OptionalStorage.impl_witness_table.968, @T.as_type.as.OptionalStorage.impl(%facet_value.8de) [concrete]
// CHECK:STDOUT:   %OptionalStorage.facet.eb3: %OptionalStorage.type = facet_value %ValueType, (%OptionalStorage.impl_witness.1b6) [concrete]
// CHECK:STDOUT:   %Optional.216: type = class_type @Optional, @Optional(%OptionalStorage.facet.eb3) [concrete]
// CHECK:STDOUT:   %specific_impl_fn.a80: <specific function> = specific_impl_function %impl.elem3.017, @Iterate.WithSelf.Next(%Iterate.facet.c1a) [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.type.6b8: type = fn_type @Optional.HasValue, @Optional(%OptionalStorage.facet.eb3) [concrete]
// CHECK:STDOUT:   %Optional.HasValue.aad: %Optional.HasValue.type.6b8 = struct_value () [concrete]
// CHECK:STDOUT:   %Optional.Get.type.42e: type = fn_type @Optional.Get, @Optional(%OptionalStorage.facet.eb3) [concrete]
// CHECK:STDOUT:   %Optional.Get.41e: %Optional.Get.type.42e = struct_value () [concrete]
// CHECK:STDOUT:   %facet_value.b9a: %facet_type.8a3 = facet_value %ValueType, (%custom_witness.55b, %custom_witness.2cb555.2) [concrete]
// CHECK:STDOUT:   %MaybeUnformed.9a0: type = class_type @MaybeUnformed, @MaybeUnformed(%facet_value.b9a) [concrete]
// CHECK:STDOUT:   %.dd1: type = maybe_unformed_type %ValueType [concrete]
// CHECK:STDOUT:   %DefaultOptionalStorage.elem.f67: type = unbound_element_type %DefaultOptionalStorage.d2b, %MaybeUnformed.9a0 [concrete]
// CHECK:STDOUT:   %DefaultOptionalStorage.elem.74a: type = unbound_element_type %DefaultOptionalStorage.d2b, bool [concrete]
// CHECK:STDOUT:   %struct_type.value.has_value.db2: type = struct_type {.value: %MaybeUnformed.9a0, .has_value: bool} [concrete]
// CHECK:STDOUT:   %Optional.HasValue.specific_fn: <specific function> = specific_function %Optional.HasValue.aad, @Optional.HasValue(%OptionalStorage.facet.eb3) [concrete]
// CHECK:STDOUT:   %Optional.Get.specific_fn: <specific function> = specific_function %Optional.Get.41e, @Optional.Get(%OptionalStorage.facet.eb3) [concrete]
// CHECK:STDOUT:   %operator_PlusEqual__carbon_thunk.type: type = fn_type @operator_PlusEqual__carbon_thunk [concrete]
// CHECK:STDOUT:   %operator_PlusEqual__carbon_thunk: %operator_PlusEqual__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.3: type = fn_type @SubobjectDestroy.WithSelf.Op.loc31_44.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.3: %SubobjectDestroy.WithSelf.Op.type.6870fd.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.3: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.3), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.4: type = fn_type @SubobjectDestroy.WithSelf.Op.loc31_44.3 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.4: %SubobjectDestroy.WithSelf.Op.type.6870fd.4 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.4: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.4), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.type.b77747.4: type = fn_type @DestroyImpl.WithSelf.Op.loc31_44.1 [concrete]
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.25aa57.4: %DestroyImpl.WithSelf.Op.type.b77747.4 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.2cb555.3: <witness> = custom_witness (%DestroyImpl.WithSelf.Op.25aa57.4), @DestroyImpl [concrete]
// CHECK:STDOUT:   %facet_value.ff1: %facet_type.8a3 = facet_value bool, (%custom_witness.b80ef6.4, %custom_witness.2cb555.3) [concrete]
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.type.b77747.6: type = fn_type @DestroyImpl.WithSelf.Op.loc31_44.3 [concrete]
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.25aa57.6: %DestroyImpl.WithSelf.Op.type.b77747.6 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.2cb555.5: <witness> = custom_witness (%DestroyImpl.WithSelf.Op.25aa57.6), @DestroyImpl [concrete]
// CHECK:STDOUT:   %facet_value.c9c: %facet_type.8a3 = facet_value %MaybeUnformed.9a0, (%custom_witness.b80ef6.3, %custom_witness.2cb555.5) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.7: type = fn_type @SubobjectDestroy.WithSelf.Op.loc31_44.6 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.7: %SubobjectDestroy.WithSelf.Op.type.6870fd.7 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.7: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.7), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.type.b77747.9: type = fn_type @DestroyImpl.WithSelf.Op.loc31_44.6 [concrete]
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.25aa57.9: %DestroyImpl.WithSelf.Op.type.b77747.9 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.2cb555.8: <witness> = custom_witness (%DestroyImpl.WithSelf.Op.25aa57.9), @DestroyImpl [concrete]
// CHECK:STDOUT:   %facet_value.35f: %facet_type.8a3 = facet_value %Optional.216, (%custom_witness.b80ef6.7, %custom_witness.2cb555.8) [concrete]
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
// CHECK:STDOUT:   %Core.import_ref.38a: type = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%DefaultOptionalStorage (constants.%DefaultOptionalStorage.7db)]
// CHECK:STDOUT:   %Core.import_ref.d2b = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], unloaded
// CHECK:STDOUT:   %Core.import_ref.370 = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], unloaded
// CHECK:STDOUT:   %Core.import_ref.6ba4: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Has.type (%T.as_type.as.OptionalStorage.impl.Has.type.8e5) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Has (constants.%T.as_type.as.OptionalStorage.impl.Has.2d5)]
// CHECK:STDOUT:   %Core.import_ref.73d: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Get.type (%T.as_type.as.OptionalStorage.impl.Get.type.b32) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Get (constants.%T.as_type.as.OptionalStorage.impl.Get.8b0)]
// CHECK:STDOUT:   %Core.import_ref.778 = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], unloaded
// CHECK:STDOUT:   %OptionalStorage.impl_witness_table.968 = impl_witness_table (%Core.import_ref.38a, %Core.import_ref.d2b, %Core.import_ref.370, %Core.import_ref.6ba4, %Core.import_ref.73d, %Core.import_ref.778), @T.as_type.as.OptionalStorage.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.1c5: @DefaultOptionalStorage.%DefaultOptionalStorage.elem.1 (%DefaultOptionalStorage.elem.b5a) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [concrete = %field_decl.649]
// CHECK:STDOUT:   %Core.import_ref.1cb: @DefaultOptionalStorage.%DefaultOptionalStorage.elem.2 (%DefaultOptionalStorage.elem.98d) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [concrete = %field_decl.255]
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
// CHECK:STDOUT:   %field_decl.255: @DefaultOptionalStorage.%DefaultOptionalStorage.elem.2 (%DefaultOptionalStorage.elem.98d) = field_decl has_value, element1, %Core.import_ref.b0d in [concrete] {}
// CHECK:STDOUT:   %Core.import_ref.4cb = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %field_decl.649: @DefaultOptionalStorage.%DefaultOptionalStorage.elem.1 (%DefaultOptionalStorage.elem.b5a) = field_decl value, element0, %Core.import_ref.4cb in [concrete] {}
// CHECK:STDOUT:   %MutableRange.decl: type = class_decl @MutableRange [concrete = constants.%MutableRange] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @TestIterate(%R.loc27_17.2: %Iterate_where.type.f8a) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Iterate.lookup_impl_witness: <witness> = lookup_impl_witness %R.loc27_17.1, @Iterate [symbolic = %Iterate.lookup_impl_witness (constants.%Iterate.lookup_impl_witness.a0e)]
// CHECK:STDOUT:   %Iterate.facet.loc31_44.5: %Iterate.type = facet_value %R.as_type.loc27_86.1, (%Iterate.lookup_impl_witness) [symbolic = %Iterate.facet.loc31_44.5 (constants.%Iterate.facet.c1a)]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.loc31_44.5) [symbolic = %Iterate.WithSelf.NewCursor.type (constants.%Iterate.WithSelf.NewCursor.type.d83)]
// CHECK:STDOUT:   %.loc31_44.26: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type, %Iterate.facet.loc31_44.5 [symbolic = %.loc31_44.26 (constants.%.54c)]
// CHECK:STDOUT:   %impl.elem2.loc31_44.2: @TestIterate.%.loc31_44.26 (%.54c) = impl_witness_access %Iterate.lookup_impl_witness, element2 [symbolic = %impl.elem2.loc31_44.2 (constants.%impl.elem2.e98)]
// CHECK:STDOUT:   %specific_impl_fn.loc31_44.3: <specific function> = specific_impl_function %impl.elem2.loc31_44.2, @Iterate.WithSelf.NewCursor(%Iterate.facet.loc31_44.5) [symbolic = %specific_impl_fn.loc31_44.3 (constants.%specific_impl_fn.bea)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.loc31_44.5) [symbolic = %Iterate.WithSelf.Next.type (constants.%Iterate.WithSelf.Next.type.1dc)]
// CHECK:STDOUT:   %.loc31_44.27: type = fn_type_with_self_type %Iterate.WithSelf.Next.type, %Iterate.facet.loc31_44.5 [symbolic = %.loc31_44.27 (constants.%.6c6)]
// CHECK:STDOUT:   %impl.elem3.loc31_44.2: @TestIterate.%.loc31_44.27 (%.6c6) = impl_witness_access %Iterate.lookup_impl_witness, element3 [symbolic = %impl.elem3.loc31_44.2 (constants.%impl.elem3.017)]
// CHECK:STDOUT:   %specific_impl_fn.loc31_44.4: <specific function> = specific_impl_function %impl.elem3.loc31_44.2, @Iterate.WithSelf.Next(%Iterate.facet.loc31_44.5) [symbolic = %specific_impl_fn.loc31_44.4 (constants.%specific_impl_fn.a80)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%r.param: @TestIterate.%R.as_type.loc27_86.1 (%R.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %i.patt: %pattern_type.d2d = value_binding_pattern i [concrete = constants.%i.patt]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %r.ref: @TestIterate.%R.as_type.loc27_86.1 (%R.as_type) = name_ref r, %r
// CHECK:STDOUT:     %impl.elem2.loc31_44.1: @TestIterate.%.loc31_44.26 (%.54c) = impl_witness_access constants.%Iterate.lookup_impl_witness.a0e, element2 [symbolic = %impl.elem2.loc31_44.2 (constants.%impl.elem2.e98)]
// CHECK:STDOUT:     %bound_method.loc31_44.1: <bound method> = bound_method %r.ref, %impl.elem2.loc31_44.1
// CHECK:STDOUT:     %Iterate.facet.loc31_44.1: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.a0e) [symbolic = %Iterate.facet.loc31_44.5 (constants.%Iterate.facet.c1a)]
// CHECK:STDOUT:     %.loc31_44.1: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc31_44.1 [symbolic = %Iterate.facet.loc31_44.5 (constants.%Iterate.facet.c1a)]
// CHECK:STDOUT:     %Iterate.facet.loc31_44.2: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.a0e) [symbolic = %Iterate.facet.loc31_44.5 (constants.%Iterate.facet.c1a)]
// CHECK:STDOUT:     %.loc31_44.2: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc31_44.2 [symbolic = %Iterate.facet.loc31_44.5 (constants.%Iterate.facet.c1a)]
// CHECK:STDOUT:     %specific_impl_fn.loc31_44.1: <specific function> = specific_impl_function %impl.elem2.loc31_44.1, @Iterate.WithSelf.NewCursor(constants.%Iterate.facet.c1a) [symbolic = %specific_impl_fn.loc31_44.3 (constants.%specific_impl_fn.bea)]
// CHECK:STDOUT:     %bound_method.loc31_44.2: <bound method> = bound_method %r.ref, %specific_impl_fn.loc31_44.1
// CHECK:STDOUT:     %var: ref @TestIterate.%as_type (%as_type.a37) = var_storage invalid
// CHECK:STDOUT:     %Iterate.WithSelf.NewCursor.call: init @TestIterate.%as_type (%as_type.a37) to %var = call %bound_method.loc31_44.2(%r.ref)
// CHECK:STDOUT:     assign %var, %Iterate.WithSelf.NewCursor.call
// CHECK:STDOUT:     br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.next:
// CHECK:STDOUT:     %impl.elem3.loc31_44.1: @TestIterate.%.loc31_44.27 (%.6c6) = impl_witness_access constants.%Iterate.lookup_impl_witness.a0e, element3 [symbolic = %impl.elem3.loc31_44.2 (constants.%impl.elem3.017)]
// CHECK:STDOUT:     %bound_method.loc31_44.3: <bound method> = bound_method %r.ref, %impl.elem3.loc31_44.1
// CHECK:STDOUT:     %Iterate.facet.loc31_44.3: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.a0e) [symbolic = %Iterate.facet.loc31_44.5 (constants.%Iterate.facet.c1a)]
// CHECK:STDOUT:     %.loc31_44.3: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc31_44.3 [symbolic = %Iterate.facet.loc31_44.5 (constants.%Iterate.facet.c1a)]
// CHECK:STDOUT:     %Iterate.facet.loc31_44.4: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.a0e) [symbolic = %Iterate.facet.loc31_44.5 (constants.%Iterate.facet.c1a)]
// CHECK:STDOUT:     %.loc31_44.4: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc31_44.4 [symbolic = %Iterate.facet.loc31_44.5 (constants.%Iterate.facet.c1a)]
// CHECK:STDOUT:     %.loc31_44.5: %facet_type.8a3 = converted constants.%as_type.a37, constants.%impl.elem1.de0 [symbolic = %impl.elem1 (constants.%impl.elem1.de0)]
// CHECK:STDOUT:     %.loc31_44.6: %facet_type.8a3 = converted constants.%as_type.a37, constants.%impl.elem1.de0 [symbolic = %impl.elem1 (constants.%impl.elem1.de0)]
// CHECK:STDOUT:     %specific_impl_fn.loc31_44.2: <specific function> = specific_impl_function %impl.elem3.loc31_44.1, @Iterate.WithSelf.Next(constants.%Iterate.facet.c1a) [symbolic = %specific_impl_fn.loc31_44.4 (constants.%specific_impl_fn.a80)]
// CHECK:STDOUT:     %bound_method.loc31_44.4: <bound method> = bound_method %r.ref, %specific_impl_fn.loc31_44.2
// CHECK:STDOUT:     %.loc31_44.7: ref %Optional.216 = temporary_storage
// CHECK:STDOUT:     %Iterate.WithSelf.Next.call: init %Optional.216 to %.loc31_44.7 = call %bound_method.loc31_44.4(%r.ref, %var)
// CHECK:STDOUT:     %.loc31_44.8: ref %Optional.216 = temporary %.loc31_44.7, %Iterate.WithSelf.Next.call
// CHECK:STDOUT:     %.loc31_44.9: %Optional.HasValue.type.6b8 = specific_constant imports.%Core.import_ref.bb6, @Optional(constants.%OptionalStorage.facet.eb3) [concrete = constants.%Optional.HasValue.aad]
// CHECK:STDOUT:     %HasValue.ref: %Optional.HasValue.type.6b8 = name_ref HasValue, %.loc31_44.9 [concrete = constants.%Optional.HasValue.aad]
// CHECK:STDOUT:     %Optional.HasValue.bound: <bound method> = bound_method %.loc31_44.8, %HasValue.ref
// CHECK:STDOUT:     %Optional.HasValue.specific_fn: <specific function> = specific_function %HasValue.ref, @Optional.HasValue(constants.%OptionalStorage.facet.eb3) [concrete = constants.%Optional.HasValue.specific_fn]
// CHECK:STDOUT:     %bound_method.loc31_44.5: <bound method> = bound_method %.loc31_44.8, %Optional.HasValue.specific_fn
// CHECK:STDOUT:     %.loc31_44.10: %Optional.216 = acquire_value %.loc31_44.8
// CHECK:STDOUT:     %Optional.HasValue.call: init bool = call %bound_method.loc31_44.5(%.loc31_44.10)
// CHECK:STDOUT:     %.loc31_44.11: bool = value_of_initializer %Optional.HasValue.call
// CHECK:STDOUT:     %.loc31_44.12: bool = converted %Optional.HasValue.call, %.loc31_44.11
// CHECK:STDOUT:     if %.loc31_44.12 br !for.body else br !for.done
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.body:
// CHECK:STDOUT:     %.loc31_44.13: %Optional.Get.type.42e = specific_constant imports.%Core.import_ref.dab, @Optional(constants.%OptionalStorage.facet.eb3) [concrete = constants.%Optional.Get.41e]
// CHECK:STDOUT:     %Get.ref: %Optional.Get.type.42e = name_ref Get, %.loc31_44.13 [concrete = constants.%Optional.Get.41e]
// CHECK:STDOUT:     %Optional.Get.bound: <bound method> = bound_method %.loc31_44.8, %Get.ref
// CHECK:STDOUT:     %Optional.Get.specific_fn: <specific function> = specific_function %Get.ref, @Optional.Get(constants.%OptionalStorage.facet.eb3) [concrete = constants.%Optional.Get.specific_fn]
// CHECK:STDOUT:     %bound_method.loc31_44.6: <bound method> = bound_method %.loc31_44.8, %Optional.Get.specific_fn
// CHECK:STDOUT:     %.loc31_44.14: ref %ValueType = temporary_storage
// CHECK:STDOUT:     %.loc31_44.15: %Optional.216 = acquire_value %.loc31_44.8
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
// CHECK:STDOUT:     %facet_value.loc31_44.1: %facet_type.8a3 = facet_value constants.%ValueType, (constants.%custom_witness.55b, constants.%custom_witness.2cb555.2) [concrete = constants.%facet_value.b9a]
// CHECK:STDOUT:     %.loc31_44.18: %facet_type.8a3 = converted constants.%ValueType, %facet_value.loc31_44.1 [concrete = constants.%facet_value.b9a]
// CHECK:STDOUT:     %facet_value.loc31_44.2: %facet_type.8a3 = facet_value constants.%ValueType, (constants.%custom_witness.55b, constants.%custom_witness.2cb555.2) [concrete = constants.%facet_value.b9a]
// CHECK:STDOUT:     %.loc31_44.19: %facet_type.8a3 = converted constants.%ValueType, %facet_value.loc31_44.2 [concrete = constants.%facet_value.b9a]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %SelfDestruct.call.loc31_44.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc31_44.16)
// CHECK:STDOUT:     %facet_value.loc31_44.3: %facet_type.8a3 = facet_value constants.%Optional.216, (constants.%custom_witness.b80ef6.7, constants.%custom_witness.2cb555.8) [concrete = constants.%facet_value.35f]
// CHECK:STDOUT:     %.loc31_44.20: %facet_type.8a3 = converted constants.%Optional.216, %facet_value.loc31_44.3 [concrete = constants.%facet_value.35f]
// CHECK:STDOUT:     %facet_value.loc31_44.4: %facet_type.8a3 = facet_value constants.%Optional.216, (constants.%custom_witness.b80ef6.7, constants.%custom_witness.2cb555.8) [concrete = constants.%facet_value.35f]
// CHECK:STDOUT:     %.loc31_44.21: %facet_type.8a3 = converted constants.%Optional.216, %facet_value.loc31_44.4 [concrete = constants.%facet_value.35f]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %SelfDestruct.call.loc31_44.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc31_44.8)
// CHECK:STDOUT:     br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.done:
// CHECK:STDOUT:     %facet_value.loc31_44.5: %facet_type.8a3 = facet_value constants.%Optional.216, (constants.%custom_witness.b80ef6.7, constants.%custom_witness.2cb555.8) [concrete = constants.%facet_value.35f]
// CHECK:STDOUT:     %.loc31_44.22: %facet_type.8a3 = converted constants.%Optional.216, %facet_value.loc31_44.5 [concrete = constants.%facet_value.35f]
// CHECK:STDOUT:     %facet_value.loc31_44.6: %facet_type.8a3 = facet_value constants.%Optional.216, (constants.%custom_witness.b80ef6.7, constants.%custom_witness.2cb555.8) [concrete = constants.%facet_value.35f]
// CHECK:STDOUT:     %.loc31_44.23: %facet_type.8a3 = converted constants.%Optional.216, %facet_value.loc31_44.6 [concrete = constants.%facet_value.35f]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %SelfDestruct.call.loc31_44.3: init %empty_tuple.type = call %SelfDestruct.specific_fn.3(%.loc31_44.8)
// CHECK:STDOUT:     %.loc31_44.24: %facet_type.8a3 = converted constants.%as_type.a37, constants.%impl.elem1.de0 [symbolic = %impl.elem1 (constants.%impl.elem1.de0)]
// CHECK:STDOUT:     %.loc31_44.25: %facet_type.8a3 = converted constants.%as_type.a37, constants.%impl.elem1.de0 [symbolic = %impl.elem1 (constants.%impl.elem1.de0)]
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
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc31_44.2(%self.param: ref %MaybeUnformed.9a0) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc31_44.3(%self.param: ref bool) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc31_44.4(%self.param: ref %struct_type.value.has_value.db2) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc31_44.1: ref bool = struct_access %self.param, element1
// CHECK:STDOUT:   %facet_value.loc31_44.1: %facet_type.8a3 = facet_value bool, (constants.%custom_witness.b80ef6.4, constants.%custom_witness.2cb555.3) [concrete = constants.%facet_value.ff1]
// CHECK:STDOUT:   %.loc31_44.2: %facet_type.8a3 = converted bool, %facet_value.loc31_44.1 [concrete = constants.%facet_value.ff1]
// CHECK:STDOUT:   %facet_value.loc31_44.2: %facet_type.8a3 = facet_value bool, (constants.%custom_witness.b80ef6.4, constants.%custom_witness.2cb555.3) [concrete = constants.%facet_value.ff1]
// CHECK:STDOUT:   %.loc31_44.3: %facet_type.8a3 = converted bool, %facet_value.loc31_44.2 [concrete = constants.%facet_value.ff1]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc31_44.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc31_44.1)
// CHECK:STDOUT:   %.loc31_44.4: ref %MaybeUnformed.9a0 = struct_access %self.param, element0
// CHECK:STDOUT:   %facet_value.loc31_44.3: %facet_type.8a3 = facet_value constants.%MaybeUnformed.9a0, (constants.%custom_witness.b80ef6.3, constants.%custom_witness.2cb555.5) [concrete = constants.%facet_value.c9c]
// CHECK:STDOUT:   %.loc31_44.5: %facet_type.8a3 = converted constants.%MaybeUnformed.9a0, %facet_value.loc31_44.3 [concrete = constants.%facet_value.c9c]
// CHECK:STDOUT:   %facet_value.loc31_44.4: %facet_type.8a3 = facet_value constants.%MaybeUnformed.9a0, (constants.%custom_witness.b80ef6.3, constants.%custom_witness.2cb555.5) [concrete = constants.%facet_value.c9c]
// CHECK:STDOUT:   %.loc31_44.6: %facet_type.8a3 = converted constants.%MaybeUnformed.9a0, %facet_value.loc31_44.4 [concrete = constants.%facet_value.c9c]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc31_44.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc31_44.4)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyImpl.WithSelf.Op.loc31_44.1(%self.param: ref bool) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyImpl.WithSelf.Op.loc31_44.2(%self.param: ref %.dd1) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyImpl.WithSelf.Op.loc31_44.3(%self.param: ref %MaybeUnformed.9a0) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc31_44.5(%self.param: ref %DefaultOptionalStorage.d2b) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %has_value.ref: %DefaultOptionalStorage.elem.74a = name_ref has_value, imports.%Core.import_ref.1cb [concrete = imports.%field_decl.255]
// CHECK:STDOUT:   %.loc31_44.1: ref bool = class_element_access %self.param, element1
// CHECK:STDOUT:   %facet_value.loc31_44.1: %facet_type.8a3 = facet_value bool, (constants.%custom_witness.b80ef6.4, constants.%custom_witness.2cb555.3) [concrete = constants.%facet_value.ff1]
// CHECK:STDOUT:   %.loc31_44.2: %facet_type.8a3 = converted bool, %facet_value.loc31_44.1 [concrete = constants.%facet_value.ff1]
// CHECK:STDOUT:   %facet_value.loc31_44.2: %facet_type.8a3 = facet_value bool, (constants.%custom_witness.b80ef6.4, constants.%custom_witness.2cb555.3) [concrete = constants.%facet_value.ff1]
// CHECK:STDOUT:   %.loc31_44.3: %facet_type.8a3 = converted bool, %facet_value.loc31_44.2 [concrete = constants.%facet_value.ff1]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc31_44.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc31_44.1)
// CHECK:STDOUT:   %value.ref: %DefaultOptionalStorage.elem.f67 = name_ref value, imports.%Core.import_ref.1c5 [concrete = imports.%field_decl.649]
// CHECK:STDOUT:   %.loc31_44.4: ref %MaybeUnformed.9a0 = class_element_access %self.param, element0
// CHECK:STDOUT:   %facet_value.loc31_44.3: %facet_type.8a3 = facet_value constants.%MaybeUnformed.9a0, (constants.%custom_witness.b80ef6.3, constants.%custom_witness.2cb555.5) [concrete = constants.%facet_value.c9c]
// CHECK:STDOUT:   %.loc31_44.5: %facet_type.8a3 = converted constants.%MaybeUnformed.9a0, %facet_value.loc31_44.3 [concrete = constants.%facet_value.c9c]
// CHECK:STDOUT:   %facet_value.loc31_44.4: %facet_type.8a3 = facet_value constants.%MaybeUnformed.9a0, (constants.%custom_witness.b80ef6.3, constants.%custom_witness.2cb555.5) [concrete = constants.%facet_value.c9c]
// CHECK:STDOUT:   %.loc31_44.6: %facet_type.8a3 = converted constants.%MaybeUnformed.9a0, %facet_value.loc31_44.4 [concrete = constants.%facet_value.c9c]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc31_44.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc31_44.4)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc31_44.6(%self.param: ref %Optional.216) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyImpl.WithSelf.Op.loc31_44.4(%self.param: ref %struct_type.value.has_value.db2) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyImpl.WithSelf.Op.loc31_44.5(%self.param: ref %DefaultOptionalStorage.d2b) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyImpl.WithSelf.Op.loc31_44.6(%self.param: ref %Optional.216) = "no_op";
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
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.002
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
// CHECK:STDOUT:   %facet_type.d61: type = facet_type <@SubobjectDestroy & @DestroyImpl & @Copy> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %T.220: %OptionalStorage.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Optional.Get.type.bb9: type = fn_type @Optional.Get, @Optional(%T.220) [symbolic]
// CHECK:STDOUT:   %Optional.Get.fa1: %Optional.Get.type.bb9 = struct_value () [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.type.e9e: type = fn_type @Optional.HasValue, @Optional(%T.220) [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.6cd: %Optional.HasValue.type.e9e = struct_value () [symbolic]
// CHECK:STDOUT:   %facet_type.8a3: type = facet_type <@SubobjectDestroy & @DestroyImpl> [concrete]
// CHECK:STDOUT:   %T.9ab: %facet_type.d61 = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.19c: type = facet_access_type %T.9ab [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.lookup_impl_witness.855: <witness> = lookup_impl_witness %T.9ab, @SubobjectDestroy [symbolic]
// CHECK:STDOUT:   %DestroyImpl.lookup_impl_witness.ff9: <witness> = lookup_impl_witness %T.9ab, @DestroyImpl [symbolic]
// CHECK:STDOUT:   %facet_value.b35: %facet_type.8a3 = facet_value %T.as_type.19c, (%SubobjectDestroy.lookup_impl_witness.855, %DestroyImpl.lookup_impl_witness.ff9) [symbolic]
// CHECK:STDOUT:   %MaybeUnformed.2cf: type = class_type @MaybeUnformed, @MaybeUnformed(%facet_value.b35) [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.7db: type = class_type @DefaultOptionalStorage, @DefaultOptionalStorage(%T.9ab) [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.elem.98d: type = unbound_element_type %DefaultOptionalStorage.7db, bool [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.elem.b5a: type = unbound_element_type %DefaultOptionalStorage.7db, %MaybeUnformed.2cf [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Get.type.b32: type = fn_type @T.as_type.as.OptionalStorage.impl.Get, @T.as_type.as.OptionalStorage.impl(%T.9ab) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Get.8b0: %T.as_type.as.OptionalStorage.impl.Get.type.b32 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Has.type.8e5: type = fn_type @T.as_type.as.OptionalStorage.impl.Has, @T.as_type.as.OptionalStorage.impl(%T.9ab) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Has.2d5: %T.as_type.as.OptionalStorage.impl.Has.type.8e5 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Some.type.61d: type = fn_type @T.as_type.as.OptionalStorage.impl.Some, @T.as_type.as.OptionalStorage.impl(%T.9ab) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Some.ba7: %T.as_type.as.OptionalStorage.impl.Some.type.61d = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.None.type.a9f: type = fn_type @T.as_type.as.OptionalStorage.impl.None, @T.as_type.as.OptionalStorage.impl(%T.9ab) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.None.083: %T.as_type.as.OptionalStorage.impl.None.type.a9f = struct_value () [symbolic]
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
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.type.b77747.3: type = fn_type @DestroyImpl.WithSelf.Op.4 [concrete]
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.25aa57.3: %DestroyImpl.WithSelf.Op.type.b77747.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.2cb555.2: <witness> = custom_witness (%DestroyImpl.WithSelf.Op.25aa57.3), @DestroyImpl [concrete]
// CHECK:STDOUT:   %const.924: type = const_type %ValueType [concrete]
// CHECK:STDOUT:   %ptr.403: type = ptr_type %const.924 [concrete]
// CHECK:STDOUT:   %ptr.6ed: type = ptr_type %ValueType [concrete]
// CHECK:STDOUT:   %ValueType.Op.type.9ce105.2: type = fn_type @ValueType.Op.2 [concrete]
// CHECK:STDOUT:   %ValueType.Op.038ea9.2: %ValueType.Op.type.9ce105.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.1c5: <witness> = custom_witness (%ValueType.Op.038ea9.2), @Copy [concrete]
// CHECK:STDOUT:   %facet_value.88c: %facet_type.d61 = facet_value %ValueType, (%custom_witness.656, %custom_witness.2cb555.2, %custom_witness.1c5) [concrete]
// CHECK:STDOUT:   %Iterate.lookup_impl_witness.1d8: <witness> = lookup_impl_witness %.Self.500, @Iterate [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.8a4: %facet_type.d61 = impl_witness_access %Iterate.lookup_impl_witness.1d8, element0 [symbolic_self]
// CHECK:STDOUT:   %Iterate_where.type.e3a: type = facet_type <@Iterate where %impl.elem0.8a4 = %facet_value.88c> [concrete]
// CHECK:STDOUT:   %pattern_type.31b: type = pattern_type %Iterate_where.type.e3a [concrete]
// CHECK:STDOUT:   %R.patt: %pattern_type.31b = symbolic_binding_pattern R, 0 [symbolic]
// CHECK:STDOUT:   %R: %Iterate_where.type.e3a = symbolic_binding R, 0 [symbolic]
// CHECK:STDOUT:   %R.as_type: type = facet_access_type %R [symbolic]
// CHECK:STDOUT:   %pattern_type.e1d: type = pattern_type %R.as_type [symbolic]
// CHECK:STDOUT:   %r.param_patt.b01: %pattern_type.e1d = value_param_pattern [symbolic]
// CHECK:STDOUT:   %r.patt.0bf: %pattern_type.e1d = wrapper_binding_pattern r, %r.param_patt.b01 [symbolic]
// CHECK:STDOUT:   %i.patt: %pattern_type.e09 = value_binding_pattern i [concrete]
// CHECK:STDOUT:   %Iterate.lookup_impl_witness.bcc: <witness> = lookup_impl_witness %R, @Iterate [symbolic]
// CHECK:STDOUT:   %Iterate.facet.dcb: %Iterate.type = facet_value %R.as_type, (%Iterate.lookup_impl_witness.bcc) [symbolic]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.23e: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.dcb) [symbolic]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.e6c: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.dcb) [symbolic]
// CHECK:STDOUT:   %.57e: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.23e, %Iterate.facet.dcb [symbolic]
// CHECK:STDOUT:   %impl.elem2.b07: %.57e = impl_witness_access %Iterate.lookup_impl_witness.bcc, element2 [symbolic]
// CHECK:STDOUT:   %impl.elem1.016: %facet_type.8a3 = impl_witness_access %Iterate.lookup_impl_witness.bcc, element1 [symbolic]
// CHECK:STDOUT:   %as_type.76d: type = facet_access_type %impl.elem1.016 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.f70: <specific function> = specific_impl_function %impl.elem2.b07, @Iterate.WithSelf.NewCursor(%Iterate.facet.dcb) [symbolic]
// CHECK:STDOUT:   %.dd9: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.e6c, %Iterate.facet.dcb [symbolic]
// CHECK:STDOUT:   %impl.elem3.6ef: %.dd9 = impl_witness_access %Iterate.lookup_impl_witness.bcc, element3 [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.c7f: type = class_type @DefaultOptionalStorage, @DefaultOptionalStorage(%facet_value.88c) [concrete]
// CHECK:STDOUT:   %OptionalStorage.impl_witness.16c: <witness> = impl_witness imports.%OptionalStorage.impl_witness_table.968, @T.as_type.as.OptionalStorage.impl(%facet_value.88c) [concrete]
// CHECK:STDOUT:   %OptionalStorage.facet.87b: %OptionalStorage.type = facet_value %ValueType, (%OptionalStorage.impl_witness.16c) [concrete]
// CHECK:STDOUT:   %Optional.d79: type = class_type @Optional, @Optional(%OptionalStorage.facet.87b) [concrete]
// CHECK:STDOUT:   %specific_impl_fn.a59: <specific function> = specific_impl_function %impl.elem3.6ef, @Iterate.WithSelf.Next(%Iterate.facet.dcb) [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.type.257: type = fn_type @Optional.HasValue, @Optional(%OptionalStorage.facet.87b) [concrete]
// CHECK:STDOUT:   %Optional.HasValue.566: %Optional.HasValue.type.257 = struct_value () [concrete]
// CHECK:STDOUT:   %Optional.Get.type.506: type = fn_type @Optional.Get, @Optional(%OptionalStorage.facet.87b) [concrete]
// CHECK:STDOUT:   %Optional.Get.197: %Optional.Get.type.506 = struct_value () [concrete]
// CHECK:STDOUT:   %facet_value.807: %facet_type.8a3 = facet_value %ValueType, (%custom_witness.656, %custom_witness.2cb555.2) [concrete]
// CHECK:STDOUT:   %MaybeUnformed.3e6: type = class_type @MaybeUnformed, @MaybeUnformed(%facet_value.807) [concrete]
// CHECK:STDOUT:   %.8f3: type = maybe_unformed_type %ValueType [concrete]
// CHECK:STDOUT:   %DefaultOptionalStorage.elem.bcc: type = unbound_element_type %DefaultOptionalStorage.c7f, %MaybeUnformed.3e6 [concrete]
// CHECK:STDOUT:   %DefaultOptionalStorage.elem.64c: type = unbound_element_type %DefaultOptionalStorage.c7f, bool [concrete]
// CHECK:STDOUT:   %struct_type.value.has_value.3ab: type = struct_type {.value: %MaybeUnformed.3e6, .has_value: bool} [concrete]
// CHECK:STDOUT:   %Optional.HasValue.specific_fn.7e0: <specific function> = specific_function %Optional.HasValue.566, @Optional.HasValue(%OptionalStorage.facet.87b) [concrete]
// CHECK:STDOUT:   %Optional.Get.specific_fn.3f1: <specific function> = specific_function %Optional.Get.197, @Optional.Get(%OptionalStorage.facet.87b) [concrete]
// CHECK:STDOUT:   %operator_PlusEqual__carbon_thunk.type: type = fn_type @operator_PlusEqual__carbon_thunk [concrete]
// CHECK:STDOUT:   %operator_PlusEqual__carbon_thunk: %operator_PlusEqual__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.3: type = fn_type @SubobjectDestroy.WithSelf.Op.loc38_41.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.3: %SubobjectDestroy.WithSelf.Op.type.6870fd.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.3: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.3), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.4: type = fn_type @SubobjectDestroy.WithSelf.Op.loc38_41.3 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.4: %SubobjectDestroy.WithSelf.Op.type.6870fd.4 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.4: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.4), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.type.b77747.4: type = fn_type @DestroyImpl.WithSelf.Op.loc38_41.1 [concrete]
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.25aa57.4: %DestroyImpl.WithSelf.Op.type.b77747.4 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.2cb555.3: <witness> = custom_witness (%DestroyImpl.WithSelf.Op.25aa57.4), @DestroyImpl [concrete]
// CHECK:STDOUT:   %facet_value.ff1: %facet_type.8a3 = facet_value bool, (%custom_witness.b80ef6.4, %custom_witness.2cb555.3) [concrete]
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.type.b77747.6: type = fn_type @DestroyImpl.WithSelf.Op.loc38_41.3 [concrete]
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.25aa57.6: %DestroyImpl.WithSelf.Op.type.b77747.6 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.2cb555.5: <witness> = custom_witness (%DestroyImpl.WithSelf.Op.25aa57.6), @DestroyImpl [concrete]
// CHECK:STDOUT:   %facet_value.8e9: %facet_type.8a3 = facet_value %MaybeUnformed.3e6, (%custom_witness.b80ef6.3, %custom_witness.2cb555.5) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.7: type = fn_type @SubobjectDestroy.WithSelf.Op.loc38_41.6 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.7: %SubobjectDestroy.WithSelf.Op.type.6870fd.7 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.7: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.7), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.type.b77747.9: type = fn_type @DestroyImpl.WithSelf.Op.loc38_41.6 [concrete]
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.25aa57.9: %DestroyImpl.WithSelf.Op.type.b77747.9 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.2cb555.8: <witness> = custom_witness (%DestroyImpl.WithSelf.Op.25aa57.9), @DestroyImpl [concrete]
// CHECK:STDOUT:   %facet_value.c59: %facet_type.8a3 = facet_value %Optional.d79, (%custom_witness.b80ef6.7, %custom_witness.2cb555.8) [concrete]
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
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.type.b77747.11: type = fn_type @DestroyImpl.WithSelf.Op.6 [concrete]
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.25aa57.11: %DestroyImpl.WithSelf.Op.type.b77747.11 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.2cb555.9: <witness> = custom_witness (%DestroyImpl.WithSelf.Op.25aa57.11), @DestroyImpl [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.c95ff6.2: type = fn_type @Iterator.Op.2 [concrete]
// CHECK:STDOUT:   %Iterator.Op.c90ac2.2: %Iterator.Op.type.c95ff6.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.992: <witness> = custom_witness (%Iterator.Op.c90ac2.2), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.c95ff6.3: type = fn_type @Iterator.Op.3 [concrete]
// CHECK:STDOUT:   %Iterator.Op.c90ac2.3: %Iterator.Op.type.c95ff6.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.68a: <witness> = custom_witness (%Iterator.Op.c90ac2.3), @Inc [concrete]
// CHECK:STDOUT:   %Equal.type: type = fn_type @Equal [concrete]
// CHECK:STDOUT:   %Equal: %Equal.type = struct_value () [concrete]
// CHECK:STDOUT:   %NotEqual.type: type = fn_type @NotEqual [concrete]
// CHECK:STDOUT:   %NotEqual: %NotEqual.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.2ef: <witness> = custom_witness (%Equal, %NotEqual), @EqWith, @EqWith(%Iterator) [concrete]
// CHECK:STDOUT:   %facet_value.a0d: %CppRange_where.type.d682d6.1 = facet_value %MutableAndConstRange, (%custom_witness.656, %custom_witness.2cb555.2, %custom_witness.1c5, %custom_witness.660, %custom_witness.992, %custom_witness.2cb555.9, %custom_witness.d56, %custom_witness.68a, %custom_witness.2ef) [concrete]
// CHECK:STDOUT:   %tuple.type.de6: type = tuple_type (%Iterator, %Iterator) [concrete]
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.type.b77747.12: type = fn_type @DestroyImpl.WithSelf.Op.7 [concrete]
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.25aa57.12: %DestroyImpl.WithSelf.Op.type.b77747.12 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.2cb555.10: <witness> = custom_witness (%DestroyImpl.WithSelf.Op.25aa57.12), @DestroyImpl [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.8: type = fn_type @SubobjectDestroy.WithSelf.Op.3 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.8: %SubobjectDestroy.WithSelf.Op.type.6870fd.8 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.8: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.8), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %facet_value.d95: %facet_type.8a3 = facet_value %tuple.type.de6, (%custom_witness.b80ef6.8, %custom_witness.2cb555.10) [concrete]
// CHECK:STDOUT:   %Iterate.impl_witness.74b: <witness> = impl_witness imports.%Iterate.impl_witness_table.121, @T.as_type.as.Iterate.impl(%facet_value.a0d) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.type.957: type = fn_type @T.as_type.as.Iterate.impl.NewCursor, @T.as_type.as.Iterate.impl(%facet_value.a0d) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.9e2: %T.as_type.as.Iterate.impl.NewCursor.type.957 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.type.f5c: type = fn_type @T.as_type.as.Iterate.impl.Next, @T.as_type.as.Iterate.impl(%facet_value.a0d) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.5f3: %T.as_type.as.Iterate.impl.Next.type.f5c = struct_value () [concrete]
// CHECK:STDOUT:   %Iterate.facet.83d: %Iterate.type = facet_value %MutableAndConstRange, (%Iterate.impl_witness.74b) [concrete]
// CHECK:STDOUT:   %facet_value.8d5: %Iterate_where.type.e3a = facet_value %MutableAndConstRange, (%Iterate.impl_witness.74b) [concrete]
// CHECK:STDOUT:   %r.param_patt.3a6: %pattern_type.fdb = value_param_pattern [concrete]
// CHECK:STDOUT:   %r.patt.403: %pattern_type.fdb = wrapper_binding_pattern r, %r.param_patt.3a6 [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.8a4: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.83d) [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.47e: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.83d) [concrete]
// CHECK:STDOUT:   %.431: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.8a4, %Iterate.facet.83d [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.specific_fn: <specific function> = specific_function %T.as_type.as.Iterate.impl.NewCursor.9e2, @T.as_type.as.Iterate.impl.NewCursor(%facet_value.a0d) [concrete]
// CHECK:STDOUT:   %.651: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.47e, %Iterate.facet.83d [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.specific_fn: <specific function> = specific_function %T.as_type.as.Iterate.impl.Next.5f3, @T.as_type.as.Iterate.impl.Next(%facet_value.a0d) [concrete]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.892: <specific function> = specific_function %SelfDestruct, @SelfDestruct(%facet_value.d95) [concrete]
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
// CHECK:STDOUT:   %Core.import_ref.38a: type = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%DefaultOptionalStorage (constants.%DefaultOptionalStorage.7db)]
// CHECK:STDOUT:   %Core.import_ref.61e: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.None.type (%T.as_type.as.OptionalStorage.impl.None.type.a9f) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.None (constants.%T.as_type.as.OptionalStorage.impl.None.083)]
// CHECK:STDOUT:   %Core.import_ref.ca1: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Some.type (%T.as_type.as.OptionalStorage.impl.Some.type.61d) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Some (constants.%T.as_type.as.OptionalStorage.impl.Some.ba7)]
// CHECK:STDOUT:   %Core.import_ref.6ba4: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Has.type (%T.as_type.as.OptionalStorage.impl.Has.type.8e5) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Has (constants.%T.as_type.as.OptionalStorage.impl.Has.2d5)]
// CHECK:STDOUT:   %Core.import_ref.73d: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Get.type (%T.as_type.as.OptionalStorage.impl.Get.type.b32) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Get (constants.%T.as_type.as.OptionalStorage.impl.Get.8b0)]
// CHECK:STDOUT:   %Core.import_ref.778 = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], unloaded
// CHECK:STDOUT:   %OptionalStorage.impl_witness_table.968 = impl_witness_table (%Core.import_ref.38a, %Core.import_ref.61e, %Core.import_ref.ca1, %Core.import_ref.6ba4, %Core.import_ref.73d, %Core.import_ref.778), @T.as_type.as.OptionalStorage.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.1c5: @DefaultOptionalStorage.%DefaultOptionalStorage.elem.1 (%DefaultOptionalStorage.elem.b5a) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [concrete = %field_decl.649]
// CHECK:STDOUT:   %Core.import_ref.1cb: @DefaultOptionalStorage.%DefaultOptionalStorage.elem.2 (%DefaultOptionalStorage.elem.98d) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [concrete = %field_decl.255]
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
// CHECK:STDOUT:   %field_decl.255: @DefaultOptionalStorage.%DefaultOptionalStorage.elem.2 (%DefaultOptionalStorage.elem.98d) = field_decl has_value, element1, %Core.import_ref.b0d in [concrete] {}
// CHECK:STDOUT:   %Core.import_ref.4cb = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %field_decl.649: @DefaultOptionalStorage.%DefaultOptionalStorage.elem.1 (%DefaultOptionalStorage.elem.b5a) = field_decl value, element0, %Core.import_ref.4cb in [concrete] {}
// CHECK:STDOUT:   %MutableAndConstRange.decl: type = class_decl @MutableAndConstRange [concrete = constants.%MutableAndConstRange] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @TestIterate(%R.loc34_17.2: %Iterate_where.type.e3a) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Iterate.lookup_impl_witness: <witness> = lookup_impl_witness %R.loc34_17.1, @Iterate [symbolic = %Iterate.lookup_impl_witness (constants.%Iterate.lookup_impl_witness.bcc)]
// CHECK:STDOUT:   %Iterate.facet.loc38_41.5: %Iterate.type = facet_value %R.as_type.loc34_83.1, (%Iterate.lookup_impl_witness) [symbolic = %Iterate.facet.loc38_41.5 (constants.%Iterate.facet.dcb)]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.loc38_41.5) [symbolic = %Iterate.WithSelf.NewCursor.type (constants.%Iterate.WithSelf.NewCursor.type.23e)]
// CHECK:STDOUT:   %.loc38_41.26: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type, %Iterate.facet.loc38_41.5 [symbolic = %.loc38_41.26 (constants.%.57e)]
// CHECK:STDOUT:   %impl.elem2.loc38_41.2: @TestIterate.%.loc38_41.26 (%.57e) = impl_witness_access %Iterate.lookup_impl_witness, element2 [symbolic = %impl.elem2.loc38_41.2 (constants.%impl.elem2.b07)]
// CHECK:STDOUT:   %specific_impl_fn.loc38_41.3: <specific function> = specific_impl_function %impl.elem2.loc38_41.2, @Iterate.WithSelf.NewCursor(%Iterate.facet.loc38_41.5) [symbolic = %specific_impl_fn.loc38_41.3 (constants.%specific_impl_fn.f70)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.loc38_41.5) [symbolic = %Iterate.WithSelf.Next.type (constants.%Iterate.WithSelf.Next.type.e6c)]
// CHECK:STDOUT:   %.loc38_41.27: type = fn_type_with_self_type %Iterate.WithSelf.Next.type, %Iterate.facet.loc38_41.5 [symbolic = %.loc38_41.27 (constants.%.dd9)]
// CHECK:STDOUT:   %impl.elem3.loc38_41.2: @TestIterate.%.loc38_41.27 (%.dd9) = impl_witness_access %Iterate.lookup_impl_witness, element3 [symbolic = %impl.elem3.loc38_41.2 (constants.%impl.elem3.6ef)]
// CHECK:STDOUT:   %specific_impl_fn.loc38_41.4: <specific function> = specific_impl_function %impl.elem3.loc38_41.2, @Iterate.WithSelf.Next(%Iterate.facet.loc38_41.5) [symbolic = %specific_impl_fn.loc38_41.4 (constants.%specific_impl_fn.a59)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%r.param: @TestIterate.%R.as_type.loc34_83.1 (%R.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %i.patt: %pattern_type.e09 = value_binding_pattern i [concrete = constants.%i.patt]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %r.ref: @TestIterate.%R.as_type.loc34_83.1 (%R.as_type) = name_ref r, %r
// CHECK:STDOUT:     %impl.elem2.loc38_41.1: @TestIterate.%.loc38_41.26 (%.57e) = impl_witness_access constants.%Iterate.lookup_impl_witness.bcc, element2 [symbolic = %impl.elem2.loc38_41.2 (constants.%impl.elem2.b07)]
// CHECK:STDOUT:     %bound_method.loc38_41.1: <bound method> = bound_method %r.ref, %impl.elem2.loc38_41.1
// CHECK:STDOUT:     %Iterate.facet.loc38_41.1: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.bcc) [symbolic = %Iterate.facet.loc38_41.5 (constants.%Iterate.facet.dcb)]
// CHECK:STDOUT:     %.loc38_41.1: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc38_41.1 [symbolic = %Iterate.facet.loc38_41.5 (constants.%Iterate.facet.dcb)]
// CHECK:STDOUT:     %Iterate.facet.loc38_41.2: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.bcc) [symbolic = %Iterate.facet.loc38_41.5 (constants.%Iterate.facet.dcb)]
// CHECK:STDOUT:     %.loc38_41.2: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc38_41.2 [symbolic = %Iterate.facet.loc38_41.5 (constants.%Iterate.facet.dcb)]
// CHECK:STDOUT:     %specific_impl_fn.loc38_41.1: <specific function> = specific_impl_function %impl.elem2.loc38_41.1, @Iterate.WithSelf.NewCursor(constants.%Iterate.facet.dcb) [symbolic = %specific_impl_fn.loc38_41.3 (constants.%specific_impl_fn.f70)]
// CHECK:STDOUT:     %bound_method.loc38_41.2: <bound method> = bound_method %r.ref, %specific_impl_fn.loc38_41.1
// CHECK:STDOUT:     %var: ref @TestIterate.%as_type (%as_type.76d) = var_storage invalid
// CHECK:STDOUT:     %Iterate.WithSelf.NewCursor.call: init @TestIterate.%as_type (%as_type.76d) to %var = call %bound_method.loc38_41.2(%r.ref)
// CHECK:STDOUT:     assign %var, %Iterate.WithSelf.NewCursor.call
// CHECK:STDOUT:     br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.next:
// CHECK:STDOUT:     %impl.elem3.loc38_41.1: @TestIterate.%.loc38_41.27 (%.dd9) = impl_witness_access constants.%Iterate.lookup_impl_witness.bcc, element3 [symbolic = %impl.elem3.loc38_41.2 (constants.%impl.elem3.6ef)]
// CHECK:STDOUT:     %bound_method.loc38_41.3: <bound method> = bound_method %r.ref, %impl.elem3.loc38_41.1
// CHECK:STDOUT:     %Iterate.facet.loc38_41.3: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.bcc) [symbolic = %Iterate.facet.loc38_41.5 (constants.%Iterate.facet.dcb)]
// CHECK:STDOUT:     %.loc38_41.3: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc38_41.3 [symbolic = %Iterate.facet.loc38_41.5 (constants.%Iterate.facet.dcb)]
// CHECK:STDOUT:     %Iterate.facet.loc38_41.4: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.bcc) [symbolic = %Iterate.facet.loc38_41.5 (constants.%Iterate.facet.dcb)]
// CHECK:STDOUT:     %.loc38_41.4: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc38_41.4 [symbolic = %Iterate.facet.loc38_41.5 (constants.%Iterate.facet.dcb)]
// CHECK:STDOUT:     %.loc38_41.5: %facet_type.8a3 = converted constants.%as_type.76d, constants.%impl.elem1.016 [symbolic = %impl.elem1 (constants.%impl.elem1.016)]
// CHECK:STDOUT:     %.loc38_41.6: %facet_type.8a3 = converted constants.%as_type.76d, constants.%impl.elem1.016 [symbolic = %impl.elem1 (constants.%impl.elem1.016)]
// CHECK:STDOUT:     %specific_impl_fn.loc38_41.2: <specific function> = specific_impl_function %impl.elem3.loc38_41.1, @Iterate.WithSelf.Next(constants.%Iterate.facet.dcb) [symbolic = %specific_impl_fn.loc38_41.4 (constants.%specific_impl_fn.a59)]
// CHECK:STDOUT:     %bound_method.loc38_41.4: <bound method> = bound_method %r.ref, %specific_impl_fn.loc38_41.2
// CHECK:STDOUT:     %.loc38_41.7: ref %Optional.d79 = temporary_storage
// CHECK:STDOUT:     %Iterate.WithSelf.Next.call: init %Optional.d79 to %.loc38_41.7 = call %bound_method.loc38_41.4(%r.ref, %var)
// CHECK:STDOUT:     %.loc38_41.8: ref %Optional.d79 = temporary %.loc38_41.7, %Iterate.WithSelf.Next.call
// CHECK:STDOUT:     %.loc38_41.9: %Optional.HasValue.type.257 = specific_constant imports.%Core.import_ref.bb6, @Optional(constants.%OptionalStorage.facet.87b) [concrete = constants.%Optional.HasValue.566]
// CHECK:STDOUT:     %HasValue.ref: %Optional.HasValue.type.257 = name_ref HasValue, %.loc38_41.9 [concrete = constants.%Optional.HasValue.566]
// CHECK:STDOUT:     %Optional.HasValue.bound: <bound method> = bound_method %.loc38_41.8, %HasValue.ref
// CHECK:STDOUT:     %Optional.HasValue.specific_fn: <specific function> = specific_function %HasValue.ref, @Optional.HasValue(constants.%OptionalStorage.facet.87b) [concrete = constants.%Optional.HasValue.specific_fn.7e0]
// CHECK:STDOUT:     %bound_method.loc38_41.5: <bound method> = bound_method %.loc38_41.8, %Optional.HasValue.specific_fn
// CHECK:STDOUT:     %.loc38_41.10: %Optional.d79 = acquire_value %.loc38_41.8
// CHECK:STDOUT:     %Optional.HasValue.call: init bool = call %bound_method.loc38_41.5(%.loc38_41.10)
// CHECK:STDOUT:     %.loc38_41.11: bool = value_of_initializer %Optional.HasValue.call
// CHECK:STDOUT:     %.loc38_41.12: bool = converted %Optional.HasValue.call, %.loc38_41.11
// CHECK:STDOUT:     if %.loc38_41.12 br !for.body else br !for.done
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.body:
// CHECK:STDOUT:     %.loc38_41.13: %Optional.Get.type.506 = specific_constant imports.%Core.import_ref.dab, @Optional(constants.%OptionalStorage.facet.87b) [concrete = constants.%Optional.Get.197]
// CHECK:STDOUT:     %Get.ref: %Optional.Get.type.506 = name_ref Get, %.loc38_41.13 [concrete = constants.%Optional.Get.197]
// CHECK:STDOUT:     %Optional.Get.bound: <bound method> = bound_method %.loc38_41.8, %Get.ref
// CHECK:STDOUT:     %Optional.Get.specific_fn: <specific function> = specific_function %Get.ref, @Optional.Get(constants.%OptionalStorage.facet.87b) [concrete = constants.%Optional.Get.specific_fn.3f1]
// CHECK:STDOUT:     %bound_method.loc38_41.6: <bound method> = bound_method %.loc38_41.8, %Optional.Get.specific_fn
// CHECK:STDOUT:     %.loc38_41.14: ref %ValueType = temporary_storage
// CHECK:STDOUT:     %.loc38_41.15: %Optional.d79 = acquire_value %.loc38_41.8
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
// CHECK:STDOUT:     %facet_value.loc38_41.1: %facet_type.8a3 = facet_value constants.%ValueType, (constants.%custom_witness.656, constants.%custom_witness.2cb555.2) [concrete = constants.%facet_value.807]
// CHECK:STDOUT:     %.loc38_41.18: %facet_type.8a3 = converted constants.%ValueType, %facet_value.loc38_41.1 [concrete = constants.%facet_value.807]
// CHECK:STDOUT:     %facet_value.loc38_41.2: %facet_type.8a3 = facet_value constants.%ValueType, (constants.%custom_witness.656, constants.%custom_witness.2cb555.2) [concrete = constants.%facet_value.807]
// CHECK:STDOUT:     %.loc38_41.19: %facet_type.8a3 = converted constants.%ValueType, %facet_value.loc38_41.2 [concrete = constants.%facet_value.807]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %SelfDestruct.call.loc38_41.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc38_41.16)
// CHECK:STDOUT:     %facet_value.loc38_41.3: %facet_type.8a3 = facet_value constants.%Optional.d79, (constants.%custom_witness.b80ef6.7, constants.%custom_witness.2cb555.8) [concrete = constants.%facet_value.c59]
// CHECK:STDOUT:     %.loc38_41.20: %facet_type.8a3 = converted constants.%Optional.d79, %facet_value.loc38_41.3 [concrete = constants.%facet_value.c59]
// CHECK:STDOUT:     %facet_value.loc38_41.4: %facet_type.8a3 = facet_value constants.%Optional.d79, (constants.%custom_witness.b80ef6.7, constants.%custom_witness.2cb555.8) [concrete = constants.%facet_value.c59]
// CHECK:STDOUT:     %.loc38_41.21: %facet_type.8a3 = converted constants.%Optional.d79, %facet_value.loc38_41.4 [concrete = constants.%facet_value.c59]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %SelfDestruct.call.loc38_41.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc38_41.8)
// CHECK:STDOUT:     br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.done:
// CHECK:STDOUT:     %facet_value.loc38_41.5: %facet_type.8a3 = facet_value constants.%Optional.d79, (constants.%custom_witness.b80ef6.7, constants.%custom_witness.2cb555.8) [concrete = constants.%facet_value.c59]
// CHECK:STDOUT:     %.loc38_41.22: %facet_type.8a3 = converted constants.%Optional.d79, %facet_value.loc38_41.5 [concrete = constants.%facet_value.c59]
// CHECK:STDOUT:     %facet_value.loc38_41.6: %facet_type.8a3 = facet_value constants.%Optional.d79, (constants.%custom_witness.b80ef6.7, constants.%custom_witness.2cb555.8) [concrete = constants.%facet_value.c59]
// CHECK:STDOUT:     %.loc38_41.23: %facet_type.8a3 = converted constants.%Optional.d79, %facet_value.loc38_41.6 [concrete = constants.%facet_value.c59]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %SelfDestruct.call.loc38_41.3: init %empty_tuple.type = call %SelfDestruct.specific_fn.3(%.loc38_41.8)
// CHECK:STDOUT:     %.loc38_41.24: %facet_type.8a3 = converted constants.%as_type.76d, constants.%impl.elem1.016 [symbolic = %impl.elem1 (constants.%impl.elem1.016)]
// CHECK:STDOUT:     %.loc38_41.25: %facet_type.8a3 = converted constants.%as_type.76d, constants.%impl.elem1.016 [symbolic = %impl.elem1 (constants.%impl.elem1.016)]
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
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc38_41.2(%self.param: ref %MaybeUnformed.3e6) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc38_41.3(%self.param: ref bool) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc38_41.4(%self.param: ref %struct_type.value.has_value.3ab) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc38_41.1: ref bool = struct_access %self.param, element1
// CHECK:STDOUT:   %facet_value.loc38_41.1: %facet_type.8a3 = facet_value bool, (constants.%custom_witness.b80ef6.4, constants.%custom_witness.2cb555.3) [concrete = constants.%facet_value.ff1]
// CHECK:STDOUT:   %.loc38_41.2: %facet_type.8a3 = converted bool, %facet_value.loc38_41.1 [concrete = constants.%facet_value.ff1]
// CHECK:STDOUT:   %facet_value.loc38_41.2: %facet_type.8a3 = facet_value bool, (constants.%custom_witness.b80ef6.4, constants.%custom_witness.2cb555.3) [concrete = constants.%facet_value.ff1]
// CHECK:STDOUT:   %.loc38_41.3: %facet_type.8a3 = converted bool, %facet_value.loc38_41.2 [concrete = constants.%facet_value.ff1]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc38_41.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc38_41.1)
// CHECK:STDOUT:   %.loc38_41.4: ref %MaybeUnformed.3e6 = struct_access %self.param, element0
// CHECK:STDOUT:   %facet_value.loc38_41.3: %facet_type.8a3 = facet_value constants.%MaybeUnformed.3e6, (constants.%custom_witness.b80ef6.3, constants.%custom_witness.2cb555.5) [concrete = constants.%facet_value.8e9]
// CHECK:STDOUT:   %.loc38_41.5: %facet_type.8a3 = converted constants.%MaybeUnformed.3e6, %facet_value.loc38_41.3 [concrete = constants.%facet_value.8e9]
// CHECK:STDOUT:   %facet_value.loc38_41.4: %facet_type.8a3 = facet_value constants.%MaybeUnformed.3e6, (constants.%custom_witness.b80ef6.3, constants.%custom_witness.2cb555.5) [concrete = constants.%facet_value.8e9]
// CHECK:STDOUT:   %.loc38_41.6: %facet_type.8a3 = converted constants.%MaybeUnformed.3e6, %facet_value.loc38_41.4 [concrete = constants.%facet_value.8e9]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc38_41.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc38_41.4)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyImpl.WithSelf.Op.loc38_41.1(%self.param: ref bool) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyImpl.WithSelf.Op.loc38_41.2(%self.param: ref %.8f3) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyImpl.WithSelf.Op.loc38_41.3(%self.param: ref %MaybeUnformed.3e6) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc38_41.5(%self.param: ref %DefaultOptionalStorage.c7f) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %has_value.ref: %DefaultOptionalStorage.elem.64c = name_ref has_value, imports.%Core.import_ref.1cb [concrete = imports.%field_decl.255]
// CHECK:STDOUT:   %.loc38_41.1: ref bool = class_element_access %self.param, element1
// CHECK:STDOUT:   %facet_value.loc38_41.1: %facet_type.8a3 = facet_value bool, (constants.%custom_witness.b80ef6.4, constants.%custom_witness.2cb555.3) [concrete = constants.%facet_value.ff1]
// CHECK:STDOUT:   %.loc38_41.2: %facet_type.8a3 = converted bool, %facet_value.loc38_41.1 [concrete = constants.%facet_value.ff1]
// CHECK:STDOUT:   %facet_value.loc38_41.2: %facet_type.8a3 = facet_value bool, (constants.%custom_witness.b80ef6.4, constants.%custom_witness.2cb555.3) [concrete = constants.%facet_value.ff1]
// CHECK:STDOUT:   %.loc38_41.3: %facet_type.8a3 = converted bool, %facet_value.loc38_41.2 [concrete = constants.%facet_value.ff1]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc38_41.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc38_41.1)
// CHECK:STDOUT:   %value.ref: %DefaultOptionalStorage.elem.bcc = name_ref value, imports.%Core.import_ref.1c5 [concrete = imports.%field_decl.649]
// CHECK:STDOUT:   %.loc38_41.4: ref %MaybeUnformed.3e6 = class_element_access %self.param, element0
// CHECK:STDOUT:   %facet_value.loc38_41.3: %facet_type.8a3 = facet_value constants.%MaybeUnformed.3e6, (constants.%custom_witness.b80ef6.3, constants.%custom_witness.2cb555.5) [concrete = constants.%facet_value.8e9]
// CHECK:STDOUT:   %.loc38_41.5: %facet_type.8a3 = converted constants.%MaybeUnformed.3e6, %facet_value.loc38_41.3 [concrete = constants.%facet_value.8e9]
// CHECK:STDOUT:   %facet_value.loc38_41.4: %facet_type.8a3 = facet_value constants.%MaybeUnformed.3e6, (constants.%custom_witness.b80ef6.3, constants.%custom_witness.2cb555.5) [concrete = constants.%facet_value.8e9]
// CHECK:STDOUT:   %.loc38_41.6: %facet_type.8a3 = converted constants.%MaybeUnformed.3e6, %facet_value.loc38_41.4 [concrete = constants.%facet_value.8e9]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc38_41.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc38_41.4)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc38_41.6(%self.param: ref %Optional.d79) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyImpl.WithSelf.Op.loc38_41.4(%self.param: ref %struct_type.value.has_value.3ab) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyImpl.WithSelf.Op.loc38_41.5(%self.param: ref %DefaultOptionalStorage.c7f) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyImpl.WithSelf.Op.loc38_41.6(%self.param: ref %Optional.d79) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestRangeFor(%mc.param: ref %MutableAndConstRange) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %i.patt: %pattern_type.e09 = value_binding_pattern i [concrete = constants.%i.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %mc.ref: ref %MutableAndConstRange = name_ref mc, %mc
// CHECK:STDOUT:   %impl.elem2: %.431 = impl_witness_access constants.%Iterate.impl_witness.74b, element2 [concrete = constants.%T.as_type.as.Iterate.impl.NewCursor.9e2]
// CHECK:STDOUT:   %bound_method.loc80_42.1: <bound method> = bound_method %mc.ref, %impl.elem2
// CHECK:STDOUT:   %facet_value.loc80_42.1: %CppRange_where.type.d682d6.1 = facet_value constants.%MutableAndConstRange, (constants.%custom_witness.656, constants.%custom_witness.2cb555.2, constants.%custom_witness.1c5, constants.%custom_witness.660, constants.%custom_witness.992, constants.%custom_witness.2cb555.9, constants.%custom_witness.d56, constants.%custom_witness.68a, constants.%custom_witness.2ef) [concrete = constants.%facet_value.a0d]
// CHECK:STDOUT:   %.loc80_42.1: %CppRange_where.type.d682d6.1 = converted constants.%MutableAndConstRange, %facet_value.loc80_42.1 [concrete = constants.%facet_value.a0d]
// CHECK:STDOUT:   %facet_value.loc80_42.2: %CppRange_where.type.d682d6.1 = facet_value constants.%MutableAndConstRange, (constants.%custom_witness.656, constants.%custom_witness.2cb555.2, constants.%custom_witness.1c5, constants.%custom_witness.660, constants.%custom_witness.992, constants.%custom_witness.2cb555.9, constants.%custom_witness.d56, constants.%custom_witness.68a, constants.%custom_witness.2ef) [concrete = constants.%facet_value.a0d]
// CHECK:STDOUT:   %.loc80_42.2: %CppRange_where.type.d682d6.1 = converted constants.%MutableAndConstRange, %facet_value.loc80_42.2 [concrete = constants.%facet_value.a0d]
// CHECK:STDOUT:   %specific_fn.loc80_42.1: <specific function> = specific_function %impl.elem2, @T.as_type.as.Iterate.impl.NewCursor(constants.%facet_value.a0d) [concrete = constants.%T.as_type.as.Iterate.impl.NewCursor.specific_fn]
// CHECK:STDOUT:   %bound_method.loc80_42.2: <bound method> = bound_method %mc.ref, %specific_fn.loc80_42.1
// CHECK:STDOUT:   %var: ref %tuple.type.de6 = var_storage invalid
// CHECK:STDOUT:   %.loc80_40.1: %MutableAndConstRange = acquire_value %mc.ref
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.call: init %tuple.type.de6 to %var = call %bound_method.loc80_42.2(%.loc80_40.1)
// CHECK:STDOUT:   assign %var, %T.as_type.as.Iterate.impl.NewCursor.call
// CHECK:STDOUT:   br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT: !for.next:
// CHECK:STDOUT:   %impl.elem3: %.651 = impl_witness_access constants.%Iterate.impl_witness.74b, element3 [concrete = constants.%T.as_type.as.Iterate.impl.Next.5f3]
// CHECK:STDOUT:   %bound_method.loc80_42.3: <bound method> = bound_method %mc.ref, %impl.elem3
// CHECK:STDOUT:   %facet_value.loc80_42.3: %CppRange_where.type.d682d6.1 = facet_value constants.%MutableAndConstRange, (constants.%custom_witness.656, constants.%custom_witness.2cb555.2, constants.%custom_witness.1c5, constants.%custom_witness.660, constants.%custom_witness.992, constants.%custom_witness.2cb555.9, constants.%custom_witness.d56, constants.%custom_witness.68a, constants.%custom_witness.2ef) [concrete = constants.%facet_value.a0d]
// CHECK:STDOUT:   %.loc80_42.3: %CppRange_where.type.d682d6.1 = converted constants.%MutableAndConstRange, %facet_value.loc80_42.3 [concrete = constants.%facet_value.a0d]
// CHECK:STDOUT:   %facet_value.loc80_42.4: %CppRange_where.type.d682d6.1 = facet_value constants.%MutableAndConstRange, (constants.%custom_witness.656, constants.%custom_witness.2cb555.2, constants.%custom_witness.1c5, constants.%custom_witness.660, constants.%custom_witness.992, constants.%custom_witness.2cb555.9, constants.%custom_witness.d56, constants.%custom_witness.68a, constants.%custom_witness.2ef) [concrete = constants.%facet_value.a0d]
// CHECK:STDOUT:   %.loc80_42.4: %CppRange_where.type.d682d6.1 = converted constants.%MutableAndConstRange, %facet_value.loc80_42.4 [concrete = constants.%facet_value.a0d]
// CHECK:STDOUT:   %specific_fn.loc80_42.2: <specific function> = specific_function %impl.elem3, @T.as_type.as.Iterate.impl.Next(constants.%facet_value.a0d) [concrete = constants.%T.as_type.as.Iterate.impl.Next.specific_fn]
// CHECK:STDOUT:   %bound_method.loc80_42.4: <bound method> = bound_method %mc.ref, %specific_fn.loc80_42.2
// CHECK:STDOUT:   %.loc80_42.5: ref %Optional.d79 = temporary_storage
// CHECK:STDOUT:   %.loc80_40.2: %MutableAndConstRange = acquire_value %mc.ref
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.call: init %Optional.d79 to %.loc80_42.5 = call %bound_method.loc80_42.4(%.loc80_40.2, %var)
// CHECK:STDOUT:   %.loc80_42.6: ref %Optional.d79 = temporary %.loc80_42.5, %T.as_type.as.Iterate.impl.Next.call
// CHECK:STDOUT:   %.loc80_42.7: %Optional.HasValue.type.257 = specific_constant imports.%Core.import_ref.bb6, @Optional(constants.%OptionalStorage.facet.87b) [concrete = constants.%Optional.HasValue.566]
// CHECK:STDOUT:   %HasValue.ref: %Optional.HasValue.type.257 = name_ref HasValue, %.loc80_42.7 [concrete = constants.%Optional.HasValue.566]
// CHECK:STDOUT:   %Optional.HasValue.bound: <bound method> = bound_method %.loc80_42.6, %HasValue.ref
// CHECK:STDOUT:   %Optional.HasValue.specific_fn: <specific function> = specific_function %HasValue.ref, @Optional.HasValue(constants.%OptionalStorage.facet.87b) [concrete = constants.%Optional.HasValue.specific_fn.7e0]
// CHECK:STDOUT:   %bound_method.loc80_42.5: <bound method> = bound_method %.loc80_42.6, %Optional.HasValue.specific_fn
// CHECK:STDOUT:   %.loc80_42.8: %Optional.d79 = acquire_value %.loc80_42.6
// CHECK:STDOUT:   %Optional.HasValue.call: init bool = call %bound_method.loc80_42.5(%.loc80_42.8)
// CHECK:STDOUT:   %.loc80_42.9: bool = value_of_initializer %Optional.HasValue.call
// CHECK:STDOUT:   %.loc80_42.10: bool = converted %Optional.HasValue.call, %.loc80_42.9
// CHECK:STDOUT:   if %.loc80_42.10 br !for.body else br !for.done
// CHECK:STDOUT:
// CHECK:STDOUT: !for.body:
// CHECK:STDOUT:   %.loc80_42.11: %Optional.Get.type.506 = specific_constant imports.%Core.import_ref.dab, @Optional(constants.%OptionalStorage.facet.87b) [concrete = constants.%Optional.Get.197]
// CHECK:STDOUT:   %Get.ref: %Optional.Get.type.506 = name_ref Get, %.loc80_42.11 [concrete = constants.%Optional.Get.197]
// CHECK:STDOUT:   %Optional.Get.bound: <bound method> = bound_method %.loc80_42.6, %Get.ref
// CHECK:STDOUT:   %Optional.Get.specific_fn: <specific function> = specific_function %Get.ref, @Optional.Get(constants.%OptionalStorage.facet.87b) [concrete = constants.%Optional.Get.specific_fn.3f1]
// CHECK:STDOUT:   %bound_method.loc80_42.6: <bound method> = bound_method %.loc80_42.6, %Optional.Get.specific_fn
// CHECK:STDOUT:   %.loc80_42.12: ref %ValueType = temporary_storage
// CHECK:STDOUT:   %.loc80_42.13: %Optional.d79 = acquire_value %.loc80_42.6
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
// CHECK:STDOUT:   %facet_value.loc80_42.5: %facet_type.8a3 = facet_value constants.%ValueType, (constants.%custom_witness.656, constants.%custom_witness.2cb555.2) [concrete = constants.%facet_value.807]
// CHECK:STDOUT:   %.loc80_42.16: %facet_type.8a3 = converted constants.%ValueType, %facet_value.loc80_42.5 [concrete = constants.%facet_value.807]
// CHECK:STDOUT:   %facet_value.loc80_42.6: %facet_type.8a3 = facet_value constants.%ValueType, (constants.%custom_witness.656, constants.%custom_witness.2cb555.2) [concrete = constants.%facet_value.807]
// CHECK:STDOUT:   %.loc80_42.17: %facet_type.8a3 = converted constants.%ValueType, %facet_value.loc80_42.6 [concrete = constants.%facet_value.807]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc80_42.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc80_42.14)
// CHECK:STDOUT:   %facet_value.loc80_42.7: %facet_type.8a3 = facet_value constants.%Optional.d79, (constants.%custom_witness.b80ef6.7, constants.%custom_witness.2cb555.8) [concrete = constants.%facet_value.c59]
// CHECK:STDOUT:   %.loc80_42.18: %facet_type.8a3 = converted constants.%Optional.d79, %facet_value.loc80_42.7 [concrete = constants.%facet_value.c59]
// CHECK:STDOUT:   %facet_value.loc80_42.8: %facet_type.8a3 = facet_value constants.%Optional.d79, (constants.%custom_witness.b80ef6.7, constants.%custom_witness.2cb555.8) [concrete = constants.%facet_value.c59]
// CHECK:STDOUT:   %.loc80_42.19: %facet_type.8a3 = converted constants.%Optional.d79, %facet_value.loc80_42.8 [concrete = constants.%facet_value.c59]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc80_42.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc80_42.6)
// CHECK:STDOUT:   br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT: !for.done:
// CHECK:STDOUT:   %facet_value.loc80_42.9: %facet_type.8a3 = facet_value constants.%Optional.d79, (constants.%custom_witness.b80ef6.7, constants.%custom_witness.2cb555.8) [concrete = constants.%facet_value.c59]
// CHECK:STDOUT:   %.loc80_42.20: %facet_type.8a3 = converted constants.%Optional.d79, %facet_value.loc80_42.9 [concrete = constants.%facet_value.c59]
// CHECK:STDOUT:   %facet_value.loc80_42.10: %facet_type.8a3 = facet_value constants.%Optional.d79, (constants.%custom_witness.b80ef6.7, constants.%custom_witness.2cb555.8) [concrete = constants.%facet_value.c59]
// CHECK:STDOUT:   %.loc80_42.21: %facet_type.8a3 = converted constants.%Optional.d79, %facet_value.loc80_42.10 [concrete = constants.%facet_value.c59]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc80_42.3: init %empty_tuple.type = call %SelfDestruct.specific_fn.3(%.loc80_42.6)
// CHECK:STDOUT:   %facet_value.loc80_42.11: %facet_type.8a3 = facet_value constants.%tuple.type.de6, (constants.%custom_witness.b80ef6.8, constants.%custom_witness.2cb555.10) [concrete = constants.%facet_value.d95]
// CHECK:STDOUT:   %.loc80_42.22: %facet_type.8a3 = converted constants.%tuple.type.de6, %facet_value.loc80_42.11 [concrete = constants.%facet_value.d95]
// CHECK:STDOUT:   %facet_value.loc80_42.12: %facet_type.8a3 = facet_value constants.%tuple.type.de6, (constants.%custom_witness.b80ef6.8, constants.%custom_witness.2cb555.10) [concrete = constants.%facet_value.d95]
// CHECK:STDOUT:   %.loc80_42.23: %facet_type.8a3 = converted constants.%tuple.type.de6, %facet_value.loc80_42.12 [concrete = constants.%facet_value.d95]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc80_42.4: init %empty_tuple.type = call %SelfDestruct.specific_fn.4(%var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestIterate(constants.%R) {
// CHECK:STDOUT:   %R.patt.loc34_17.2 => constants.%R.patt
// CHECK:STDOUT:   %R.loc34_17.1 => constants.%R
// CHECK:STDOUT:   %R.as_type.loc34_83.1 => constants.%R.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.e1d
// CHECK:STDOUT:   %r.param_patt.loc34_81.2 => constants.%r.param_patt.b01
// CHECK:STDOUT:   %r.patt.loc34_81.2 => constants.%r.patt.0bf
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestIterate(constants.%facet_value.8d5) {
// CHECK:STDOUT:   %R.patt.loc34_17.2 => constants.%R.patt
// CHECK:STDOUT:   %R.loc34_17.1 => constants.%facet_value.8d5
// CHECK:STDOUT:   %R.as_type.loc34_83.1 => constants.%MutableAndConstRange
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.fdb
// CHECK:STDOUT:   %r.param_patt.loc34_81.2 => constants.%r.param_patt.3a6
// CHECK:STDOUT:   %r.patt.loc34_81.2 => constants.%r.patt.403
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc34 => constants.%complete_type.357
// CHECK:STDOUT:   %Iterate.lookup_impl_witness => constants.%Iterate.impl_witness.74b
// CHECK:STDOUT:   %Iterate.facet.loc38_41.5 => constants.%Iterate.facet.83d
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type => constants.%Iterate.WithSelf.NewCursor.type.8a4
// CHECK:STDOUT:   %.loc38_41.26 => constants.%.431
// CHECK:STDOUT:   %impl.elem2.loc38_41.2 => constants.%T.as_type.as.Iterate.impl.NewCursor.9e2
// CHECK:STDOUT:   %specific_impl_fn.loc38_41.3 => constants.%T.as_type.as.Iterate.impl.NewCursor.specific_fn
// CHECK:STDOUT:   %impl.elem1 => constants.%facet_value.d95
// CHECK:STDOUT:   %as_type => constants.%tuple.type.de6
// CHECK:STDOUT:   %require_complete.1 => constants.%complete_type.95d
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type => constants.%Iterate.WithSelf.Next.type.47e
// CHECK:STDOUT:   %.loc38_41.27 => constants.%.651
// CHECK:STDOUT:   %impl.elem3.loc38_41.2 => constants.%T.as_type.as.Iterate.impl.Next.5f3
// CHECK:STDOUT:   %specific_impl_fn.loc38_41.4 => constants.%T.as_type.as.Iterate.impl.Next.specific_fn
// CHECK:STDOUT:   %SelfDestruct.specific_fn.6 => constants.%SelfDestruct.specific_fn.892
// CHECK:STDOUT: }
// CHECK:STDOUT:
