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

fn TestIterate[R: Core.Iterate where .ElementType = Cpp.ValueType](var r: R) {
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

fn TestIterate[R: Core.Iterate where .ElementType = Cpp.N.ValueType](r: R) {
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

fn TestIterate[R: Core.Iterate where .ElementType = Cpp.N.ValueType](r: R) {
  var sum: Cpp.N.ValueType = Cpp.N.ValueType.ValueType();

  //@dump-sem-ir-begin
  for (i: Cpp.N.ValueType in r) {
    sum += i;
  }
  //@dump-sem-ir-end
}

fn TestIterateDriver(var m: Cpp.N.MutableRange) {
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
  // CHECK:STDERR: fn TestIterate[R: Core.Iterate where .ElementType = Cpp.N.ValueType](r: R) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
  // CHECK:STDERR: fn TestIterate[R: Core.Iterate where .ElementType = Cpp.N.ValueType](r: R) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_todo_adl_returns_different_types_mutable.carbon:[[@LINE+17]]:3: error: facet type `Core.CppRange where .(Core.CppRangeForIterate.Iterator).(Core.CppUnsafeDeref.Result) impls Core.Copy` can not be identified [ImplLookupInUnidentifiedFacetType]
  // CHECK:STDERR:   TestIterate(m);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_adl_returns_different_types_mutable.carbon:[[@LINE+14]]:3: note: semantics TODO: `Rewriting operator== using operator== is not supported` [SemanticsTodo]
  // CHECK:STDERR:   TestIterate(m);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_adl_returns_different_types_mutable.carbon:[[@LINE-45]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn TestIterate[R: Core.Iterate where .ElementType = Cpp.N.ValueType](r: R) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_todo_adl_returns_different_types_mutable.carbon:[[@LINE+7]]:3: error: cannot convert type `Cpp.N.MutableRange` into type implementing `Core.Iterate where .(Core.Iterate.ElementType) = Cpp.N.ValueType as Core.Destroy & Core.Copy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   TestIterate(m);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_adl_returns_different_types_mutable.carbon:[[@LINE-52]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn TestIterate[R: Core.Iterate where .ElementType = Cpp.N.ValueType](r: R) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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

fn TestIterate[R: Core.Iterate where .ElementType = Cpp.N.ValueType](r: R) {
  var sum: Cpp.N.ValueType = Cpp.N.ValueType.ValueType();

  //@dump-sem-ir-begin
  for (i: Cpp.N.ValueType in r) {
    sum += i;
  }
  //@dump-sem-ir-end
}

fn TestIterateDriver(var mc: Cpp.N.MutableAndConstRange) {
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
  // CHECK:STDERR: fn TestIterate[R: Core.Iterate where .ElementType = Cpp.N.ValueType](r: R) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
  // CHECK:STDERR: fn TestIterate[R: Core.Iterate where .ElementType = Cpp.N.ValueType](r: R) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Get.type.d3c: type = fn_type @T.as_type.as.OptionalStorage.impl.Get, @T.as_type.as.OptionalStorage.impl(%T.283) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Get.da2: %T.as_type.as.OptionalStorage.impl.Get.type.d3c = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Has.type.eb9: type = fn_type @T.as_type.as.OptionalStorage.impl.Has, @T.as_type.as.OptionalStorage.impl(%T.283) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Has.c7e: %T.as_type.as.OptionalStorage.impl.Has.type.eb9 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Some.type.7eb: type = fn_type @T.as_type.as.OptionalStorage.impl.Some, @T.as_type.as.OptionalStorage.impl(%T.283) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Some.2b4: %T.as_type.as.OptionalStorage.impl.Some.type.7eb = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.None.type.d4d: type = fn_type @T.as_type.as.OptionalStorage.impl.None, @T.as_type.as.OptionalStorage.impl(%T.283) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.None.d64: %T.as_type.as.OptionalStorage.impl.None.type.d4d = struct_value () [symbolic]
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
// CHECK:STDOUT:   %CppRangeForIterate.lookup_impl_witness.b581a8.2: <witness> = lookup_impl_witness %T.f45530.2, @CppRangeForIterate [symbolic]
// CHECK:STDOUT:   %impl.elem1.c2dc15.2: type = impl_witness_access %CppRangeForIterate.lookup_impl_witness.b581a8.2, element1 [symbolic]
// CHECK:STDOUT:   %impl.elem0.5f971c.2: type = impl_witness_access %CppRangeForIterate.lookup_impl_witness.b581a8.2, element0 [symbolic]
// CHECK:STDOUT:   %tuple.type.1ba: type = tuple_type (%impl.elem0.5f971c.2, %impl.elem1.c2dc15.2) [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.812: <witness> = lookup_impl_witness %tuple.type.1ba, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.9cd: %Destroy.type = facet_value %tuple.type.1ba, (%Destroy.lookup_impl_witness.812) [symbolic]
// CHECK:STDOUT:   %CppUnsafeDeref.lookup_impl_witness.87e543.2: <witness> = lookup_impl_witness %impl.elem0.5f971c.2, @CppUnsafeDeref [symbolic]
// CHECK:STDOUT:   %impl.elem0.493765.2: type = impl_witness_access %CppUnsafeDeref.lookup_impl_witness.87e543.2, element0 [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.35816c.2: <witness> = lookup_impl_witness %impl.elem0.493765.2, @Destroy [symbolic]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.167: <witness> = lookup_impl_witness %impl.elem0.493765.2, @Copy [symbolic]
// CHECK:STDOUT:   %facet_value.f93: %facet_type.f48 = facet_value %impl.elem0.493765.2, (%Destroy.lookup_impl_witness.35816c.2, %Copy.lookup_impl_witness.167) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.type.5dc: type = fn_type @T.as_type.as.Iterate.impl.Next, @T.as_type.as.Iterate.impl(%T.f45530.2) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.b73: %T.as_type.as.Iterate.impl.Next.type.5dc = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.type.b5d: type = fn_type @T.as_type.as.Iterate.impl.NewCursor, @T.as_type.as.Iterate.impl(%T.f45530.2) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.21f: %T.as_type.as.Iterate.impl.NewCursor.type.b5d = struct_value () [symbolic]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.3: type = fn_type @Destroy.Op.loc30_53.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.3: %Destroy.Op.type.1d8f74.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.df9cc1.3: <witness> = custom_witness (%Destroy.Op.1a2547.3), @Destroy [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.ac8: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N.fe9) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.5e0: %Int.as.Copy.impl.Op.type.ac8 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.0e0: <witness> = impl_witness imports.%Copy.impl_witness_table.367, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %facet_value.ba2: %facet_type.f48 = facet_value %i32, (%custom_witness.df9cc1.3, %Copy.impl_witness.0e0) [concrete]
// CHECK:STDOUT:   %Iterate.lookup_impl_witness.1d8: <witness> = lookup_impl_witness %.Self.500, @Iterate [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.294: %facet_type.f48 = impl_witness_access %Iterate.lookup_impl_witness.1d8, element0 [symbolic_self]
// CHECK:STDOUT:   %Iterate_where.type.ec1: type = facet_type <@Iterate where %impl.elem0.294 = %facet_value.ba2> [concrete]
// CHECK:STDOUT:   %pattern_type.a4d: type = pattern_type %Iterate_where.type.ec1 [concrete]
// CHECK:STDOUT:   %R.patt: %pattern_type.a4d = symbolic_binding_pattern R, 0 [symbolic]
// CHECK:STDOUT:   %R: %Iterate_where.type.ec1 = symbolic_binding R, 0 [symbolic]
// CHECK:STDOUT:   %R.as_type: type = facet_access_type %R [symbolic]
// CHECK:STDOUT:   %pattern_type.b2da: type = pattern_type %R.as_type [symbolic]
// CHECK:STDOUT:   %r.param_patt.774: %pattern_type.b2da = value_param_pattern [symbolic]
// CHECK:STDOUT:   %r.patt.54b: %pattern_type.b2da = at_binding_pattern r, %r.param_patt.774 [symbolic]
// CHECK:STDOUT:   %facet_type.037: type = facet_type <@IntFitsIn, @IntFitsIn(%Int.67af24.2) & @AnyInt> [symbolic]
// CHECK:STDOUT:   %From.6a4: %facet_type.037 = symbolic_binding From, 1 [symbolic]
// CHECK:STDOUT:   %From.as_type.as.ImplicitAs.impl.Convert.type.7fa: type = fn_type @From.as_type.as.ImplicitAs.impl.Convert, @From.as_type.as.ImplicitAs.impl(%To, %From.6a4) [symbolic]
// CHECK:STDOUT:   %From.as_type.as.ImplicitAs.impl.Convert.643: %From.as_type.as.ImplicitAs.impl.Convert.type.7fa = struct_value () [symbolic]
// CHECK:STDOUT:   %i.patt: %pattern_type.6b6 = value_binding_pattern i [concrete]
// CHECK:STDOUT:   %Iterate.lookup_impl_witness.18b: <witness> = lookup_impl_witness %R, @Iterate [symbolic]
// CHECK:STDOUT:   %Iterate.facet.353: %Iterate.type = facet_value %R.as_type, (%Iterate.lookup_impl_witness.18b) [symbolic]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.8fb: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.353) [symbolic]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.830: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.353) [symbolic]
// CHECK:STDOUT:   %.985: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.8fb, %Iterate.facet.353 [symbolic]
// CHECK:STDOUT:   %impl.elem2.8ec: %.985 = impl_witness_access %Iterate.lookup_impl_witness.18b, element2 [symbolic]
// CHECK:STDOUT:   %impl.elem1.295: %Destroy.type = impl_witness_access %Iterate.lookup_impl_witness.18b, element1 [symbolic]
// CHECK:STDOUT:   %as_type.c8a: type = facet_access_type %impl.elem1.295 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.9ca: <specific function> = specific_impl_function %impl.elem2.8ec, @Iterate.WithSelf.NewCursor(%Iterate.facet.353) [symbolic]
// CHECK:STDOUT:   %ptr.2ef: type = ptr_type %as_type.c8a [symbolic]
// CHECK:STDOUT:   %.f86: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.830, %Iterate.facet.353 [symbolic]
// CHECK:STDOUT:   %impl.elem3.5f7: %.f86 = impl_witness_access %Iterate.lookup_impl_witness.18b, element3 [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.970: type = class_type @DefaultOptionalStorage, @DefaultOptionalStorage(%facet_value.ba2) [concrete]
// CHECK:STDOUT:   %OptionalStorage.impl_witness.9bd: <witness> = impl_witness imports.%OptionalStorage.impl_witness_table.80d, @T.as_type.as.OptionalStorage.impl(%facet_value.ba2) [concrete]
// CHECK:STDOUT:   %OptionalStorage.facet.1a2: %OptionalStorage.type = facet_value %i32, (%OptionalStorage.impl_witness.9bd) [concrete]
// CHECK:STDOUT:   %Optional.5cf: type = class_type @Optional, @Optional(%OptionalStorage.facet.1a2) [concrete]
// CHECK:STDOUT:   %specific_impl_fn.9a8: <specific function> = specific_impl_function %impl.elem3.5f7, @Iterate.WithSelf.Next(%Iterate.facet.353) [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.type.47f: type = fn_type @Optional.HasValue, @Optional(%OptionalStorage.facet.1a2) [concrete]
// CHECK:STDOUT:   %Optional.HasValue.1ad: %Optional.HasValue.type.47f = struct_value () [concrete]
// CHECK:STDOUT:   %Optional.Get.type.bc1: type = fn_type @Optional.Get, @Optional(%OptionalStorage.facet.1a2) [concrete]
// CHECK:STDOUT:   %Optional.Get.dda: %Optional.Get.type.bc1 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.facet.d07318.2: %Destroy.type = facet_value %i32, (%custom_witness.df9cc1.3) [concrete]
// CHECK:STDOUT:   %MaybeUnformed.023: type = class_type @MaybeUnformed, @MaybeUnformed(%Destroy.facet.d07318.2) [concrete]
// CHECK:STDOUT:   %.1d6a7: type = maybe_unformed_type %i32 [concrete]
// CHECK:STDOUT:   %struct_type.value.has_value.ad0: type = struct_type {.value: %MaybeUnformed.023, .has_value: bool} [concrete]
// CHECK:STDOUT:   %Optional.HasValue.specific_fn.9b9: <specific function> = specific_function %Optional.HasValue.1ad, @Optional.HasValue(%OptionalStorage.facet.1a2) [concrete]
// CHECK:STDOUT:   %Optional.Get.specific_fn.e58: <specific function> = specific_function %Optional.Get.dda, @Optional.Get(%OptionalStorage.facet.1a2) [concrete]
// CHECK:STDOUT:   %AddAssignWith.type.941: type = facet_type <@AddAssignWith, @AddAssignWith(%i32)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.f7de31.2: type = facet_type <@ImplicitAs, @ImplicitAs(%Int.67af24.1)> [symbolic]
// CHECK:STDOUT:   %U.4e6: %ImplicitAs.type.f7de31.2 = symbolic_binding U, 1 [symbolic]
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.type.a58fbe.1: type = fn_type @Int.as.AddAssignWith.impl.Op.1, @Int.as.AddAssignWith.impl(%N.fe9, %U.4e6) [symbolic]
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.45b360.1: %Int.as.AddAssignWith.impl.Op.type.a58fbe.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.type.a58fbe.2: type = fn_type @Int.as.AddAssignWith.impl.Op.2, @Int.as.AddAssignWith.impl(%N.fe9, %U.4e6) [symbolic]
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.45b360.2: %Int.as.AddAssignWith.impl.Op.type.a58fbe.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %facet_type.138: type = facet_type <@IntFitsIn, @IntFitsIn(%i32) & @AnyInt> [concrete]
// CHECK:STDOUT:   %custom_witness.36a: <witness> = custom_witness (), @IntFitsIn, @IntFitsIn(%i32) [concrete]
// CHECK:STDOUT:   %Int.as.AnyInt.impl.AsInt.type.2ea: type = fn_type @Int.as.AnyInt.impl.AsInt, @Int.as.AnyInt.impl(%N.fe9) [symbolic]
// CHECK:STDOUT:   %Int.as.AnyInt.impl.AsInt.5e2: %Int.as.AnyInt.impl.AsInt.type.2ea = struct_value () [symbolic]
// CHECK:STDOUT:   %AnyInt.impl_witness.c83: <witness> = impl_witness imports.%AnyInt.impl_witness_table, @Int.as.AnyInt.impl(%int_32) [concrete]
// CHECK:STDOUT:   %facet_value.226: %facet_type.138 = facet_value %i32, (%custom_witness.36a, %AnyInt.impl_witness.c83) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.8eb: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.3b9, @From.as_type.as.ImplicitAs.impl(%int_32, %facet_value.226) [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.354: %ImplicitAs.type.914 = facet_value %i32, (%ImplicitAs.impl_witness.8eb) [concrete]
// CHECK:STDOUT:   %AddAssignWith.impl_witness.211: <witness> = impl_witness imports.%AddAssignWith.impl_witness_table, @Int.as.AddAssignWith.impl(%int_32, %ImplicitAs.facet.354) [concrete]
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.type.56278f.1: type = fn_type @Int.as.AddAssignWith.impl.Op.2, @Int.as.AddAssignWith.impl(%int_32, %ImplicitAs.facet.354) [concrete]
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.f51aaf.1: %Int.as.AddAssignWith.impl.Op.type.56278f.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.type.56278f.2: type = fn_type @Int.as.AddAssignWith.impl.Op.1, @Int.as.AddAssignWith.impl(%int_32, %ImplicitAs.facet.354) [concrete]
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.f51aaf.2: %Int.as.AddAssignWith.impl.Op.type.56278f.2 = struct_value () [concrete]
// CHECK:STDOUT:   %AddAssignWith.facet.a68: %AddAssignWith.type.941 = facet_value %i32, (%AddAssignWith.impl_witness.211) [concrete]
// CHECK:STDOUT:   %AddAssignWith.WithSelf.Op.type.ae7: type = fn_type @AddAssignWith.WithSelf.Op, @AddAssignWith.WithSelf(%i32, %AddAssignWith.facet.a68) [concrete]
// CHECK:STDOUT:   %.583: type = fn_type_with_self_type %AddAssignWith.WithSelf.Op.type.ae7, %AddAssignWith.facet.a68 [concrete]
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.specific_fn.bd8c6b.1: <specific function> = specific_function %Int.as.AddAssignWith.impl.Op.f51aaf.2, @Int.as.AddAssignWith.impl.Op.1(%int_32, %ImplicitAs.facet.354) [concrete]
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.specific_fn.bd8c6b.2: <specific function> = specific_function %Int.as.AddAssignWith.impl.Op.f51aaf.1, @Int.as.AddAssignWith.impl.Op.2(%int_32, %ImplicitAs.facet.354) [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.9: type = fn_type @Destroy.Op.loc34_19.6 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.9: %Destroy.Op.type.1d8f74.9 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.d26: <witness> = lookup_impl_witness %impl.elem1.295, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.71c: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%impl.elem1.295) [symbolic]
// CHECK:STDOUT:   %.fa0: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.71c, %impl.elem1.295 [symbolic]
// CHECK:STDOUT:   %impl.elem0.544: %.fa0 = impl_witness_access %Destroy.lookup_impl_witness.d26, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.fa9: <specific function> = specific_impl_function %impl.elem0.544, @Destroy.WithSelf.Op(%impl.elem1.295) [symbolic]
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
// CHECK:STDOUT:   %facet_value.036: %CppRange_where.type.d682d6.1 = facet_value %MutableRange, (%custom_witness.df9cc1.3, %Copy.impl_witness.0e0, %custom_witness.ed3, %custom_witness.955, %custom_witness.ef8, %custom_witness.444, %custom_witness.981) [concrete]
// CHECK:STDOUT:   %tuple.type.508: type = tuple_type (%Iterator.3ee, %Iterator.3ee) [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.10: type = fn_type @Destroy.Op.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.10: %Destroy.Op.type.1d8f74.10 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.df9cc1.10: <witness> = custom_witness (%Destroy.Op.1a2547.10), @Destroy [concrete]
// CHECK:STDOUT:   %Destroy.facet.30f: %Destroy.type = facet_value %tuple.type.508, (%custom_witness.df9cc1.10) [concrete]
// CHECK:STDOUT:   %Iterate.impl_witness.55e: <witness> = impl_witness imports.%Iterate.impl_witness_table.478, @T.as_type.as.Iterate.impl(%facet_value.036) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.type.ef8: type = fn_type @T.as_type.as.Iterate.impl.NewCursor, @T.as_type.as.Iterate.impl(%facet_value.036) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.9cc: %T.as_type.as.Iterate.impl.NewCursor.type.ef8 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.type.a62: type = fn_type @T.as_type.as.Iterate.impl.Next, @T.as_type.as.Iterate.impl(%facet_value.036) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.6ab: %T.as_type.as.Iterate.impl.Next.type.a62 = struct_value () [concrete]
// CHECK:STDOUT:   %Iterate.facet.523: %Iterate.type = facet_value %MutableRange, (%Iterate.impl_witness.55e) [concrete]
// CHECK:STDOUT:   %facet_value.0de: %Iterate_where.type.ec1 = facet_value %MutableRange, (%Iterate.impl_witness.55e) [concrete]
// CHECK:STDOUT:   %r.param_patt.c79: %pattern_type.e635 = value_param_pattern [concrete]
// CHECK:STDOUT:   %r.patt.16f: %pattern_type.e635 = at_binding_pattern r, %r.param_patt.c79 [concrete]
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
// CHECK:STDOUT:   %facet_value.f06: %CppRange_where.type.d682d6.1 = facet_value %ConstRange, (%custom_witness.df9cc1.3, %Copy.impl_witness.0e0, %custom_witness.23e, %custom_witness.6e8, %custom_witness.adb, %custom_witness.746, %custom_witness.941) [concrete]
// CHECK:STDOUT:   %tuple.type.69f: type = tuple_type (%Iterator.260, %Iterator.260) [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.11: type = fn_type @Destroy.Op.3 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.11: %Destroy.Op.type.1d8f74.11 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.df9cc1.11: <witness> = custom_witness (%Destroy.Op.1a2547.11), @Destroy [concrete]
// CHECK:STDOUT:   %Destroy.facet.e4c: %Destroy.type = facet_value %tuple.type.69f, (%custom_witness.df9cc1.11) [concrete]
// CHECK:STDOUT:   %Iterate.impl_witness.1f3: <witness> = impl_witness imports.%Iterate.impl_witness_table.478, @T.as_type.as.Iterate.impl(%facet_value.f06) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.type.4d6: type = fn_type @T.as_type.as.Iterate.impl.NewCursor, @T.as_type.as.Iterate.impl(%facet_value.f06) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.2b7: %T.as_type.as.Iterate.impl.NewCursor.type.4d6 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.type.dce: type = fn_type @T.as_type.as.Iterate.impl.Next, @T.as_type.as.Iterate.impl(%facet_value.f06) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.e83: %T.as_type.as.Iterate.impl.Next.type.dce = struct_value () [concrete]
// CHECK:STDOUT:   %Iterate.facet.910: %Iterate.type = facet_value %ConstRange, (%Iterate.impl_witness.1f3) [concrete]
// CHECK:STDOUT:   %facet_value.b62: %Iterate_where.type.ec1 = facet_value %ConstRange, (%Iterate.impl_witness.1f3) [concrete]
// CHECK:STDOUT:   %r.param_patt.83e: %pattern_type.0ba = value_param_pattern [concrete]
// CHECK:STDOUT:   %r.patt.116: %pattern_type.0ba = at_binding_pattern r, %r.param_patt.83e [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.dfd: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.523) [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.b64: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.523) [concrete]
// CHECK:STDOUT:   %.9e1: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.dfd, %Iterate.facet.523 [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.specific_fn.6a8: <specific function> = specific_function %T.as_type.as.Iterate.impl.NewCursor.9cc, @T.as_type.as.Iterate.impl.NewCursor(%facet_value.036) [concrete]
// CHECK:STDOUT:   %ptr.45c: type = ptr_type %tuple.type.508 [concrete]
// CHECK:STDOUT:   %.4e4: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.b64, %Iterate.facet.523 [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.specific_fn.891: <specific function> = specific_function %T.as_type.as.Iterate.impl.Next.6ab, @T.as_type.as.Iterate.impl.Next(%facet_value.036) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.c6c: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.30f) [concrete]
// CHECK:STDOUT:   %.0a4: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.c6c, %Destroy.facet.30f [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.ba4: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.910) [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.dec: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.910) [concrete]
// CHECK:STDOUT:   %.5d9: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.ba4, %Iterate.facet.910 [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.specific_fn.4ae: <specific function> = specific_function %T.as_type.as.Iterate.impl.NewCursor.2b7, @T.as_type.as.Iterate.impl.NewCursor(%facet_value.f06) [concrete]
// CHECK:STDOUT:   %ptr.c5c: type = ptr_type %tuple.type.69f [concrete]
// CHECK:STDOUT:   %.712: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.dec, %Iterate.facet.910 [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.specific_fn.0ce: <specific function> = specific_function %T.as_type.as.Iterate.impl.Next.e83, @T.as_type.as.Iterate.impl.Next(%facet_value.f06) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.7ec: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.e4c) [concrete]
// CHECK:STDOUT:   %.0e7: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.7ec, %Destroy.facet.e4c [concrete]
// CHECK:STDOUT:   %complete_type.c28: <witness> = complete_type_witness %tuple.type.508 [concrete]
// CHECK:STDOUT:   %complete_type.8d5: <witness> = complete_type_witness %tuple.type.69f [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.bb6: @Optional.%Optional.HasValue.type (%Optional.HasValue.type.e9e) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.HasValue (constants.%Optional.HasValue.6cd)]
// CHECK:STDOUT:   %Core.import_ref.daba: @Optional.%Optional.Get.type (%Optional.Get.type.bb9) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.Get (constants.%Optional.Get.fa1)]
// CHECK:STDOUT:   %Core.import_ref.0c6: type = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%DefaultOptionalStorage (constants.%DefaultOptionalStorage.d50)]
// CHECK:STDOUT:   %Core.import_ref.13b: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.None.type (%T.as_type.as.OptionalStorage.impl.None.type.d4d) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.None (constants.%T.as_type.as.OptionalStorage.impl.None.d64)]
// CHECK:STDOUT:   %Core.import_ref.c43f: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Some.type (%T.as_type.as.OptionalStorage.impl.Some.type.7eb) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Some (constants.%T.as_type.as.OptionalStorage.impl.Some.2b4)]
// CHECK:STDOUT:   %Core.import_ref.61e: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Has.type (%T.as_type.as.OptionalStorage.impl.Has.type.eb9) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Has (constants.%T.as_type.as.OptionalStorage.impl.Has.c7e)]
// CHECK:STDOUT:   %Core.import_ref.ce2e: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Get.type (%T.as_type.as.OptionalStorage.impl.Get.type.d3c) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Get (constants.%T.as_type.as.OptionalStorage.impl.Get.da2)]
// CHECK:STDOUT:   %Core.import_ref.8fc = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], unloaded
// CHECK:STDOUT:   %OptionalStorage.impl_witness_table.80d = impl_witness_table (%Core.import_ref.0c6, %Core.import_ref.13b, %Core.import_ref.c43f, %Core.import_ref.61e, %Core.import_ref.ce2e, %Core.import_ref.8fc), @T.as_type.as.OptionalStorage.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.eef: %facet_type.f48 = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.Iterate.impl.%facet_value (constants.%facet_value.f93)]
// CHECK:STDOUT:   %Core.import_ref.fcc: %Destroy.type = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.Iterate.impl.%Destroy.facet (constants.%Destroy.facet.9cd)]
// CHECK:STDOUT:   %Core.import_ref.f50: @T.as_type.as.Iterate.impl.%T.as_type.as.Iterate.impl.NewCursor.type (%T.as_type.as.Iterate.impl.NewCursor.type.b5d) = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.Iterate.impl.%T.as_type.as.Iterate.impl.NewCursor (constants.%T.as_type.as.Iterate.impl.NewCursor.21f)]
// CHECK:STDOUT:   %Core.import_ref.590: @T.as_type.as.Iterate.impl.%T.as_type.as.Iterate.impl.Next.type (%T.as_type.as.Iterate.impl.Next.type.5dc) = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.Iterate.impl.%T.as_type.as.Iterate.impl.Next (constants.%T.as_type.as.Iterate.impl.Next.b73)]
// CHECK:STDOUT:   %Iterate.impl_witness_table.478 = impl_witness_table (%Core.import_ref.eef, %Core.import_ref.fcc, %Core.import_ref.f50, %Core.import_ref.590), @T.as_type.as.Iterate.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.cd6: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.ac8) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.5e0)]
// CHECK:STDOUT:   %Copy.impl_witness_table.367 = impl_witness_table (%Core.import_ref.cd6), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.2a4: @From.as_type.as.ImplicitAs.impl.%From.as_type.as.ImplicitAs.impl.Convert.type (%From.as_type.as.ImplicitAs.impl.Convert.type.7fa) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @From.as_type.as.ImplicitAs.impl.%From.as_type.as.ImplicitAs.impl.Convert (constants.%From.as_type.as.ImplicitAs.impl.Convert.643)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.3b9 = impl_witness_table (%Core.import_ref.2a4), @From.as_type.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.098: @Int.as.AddAssignWith.impl.%Int.as.AddAssignWith.impl.Op.type.2 (%Int.as.AddAssignWith.impl.Op.type.a58fbe.1) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.AddAssignWith.impl.%Int.as.AddAssignWith.impl.Op.2 (constants.%Int.as.AddAssignWith.impl.Op.45b360.1)]
// CHECK:STDOUT:   %AddAssignWith.impl_witness_table = impl_witness_table (%Core.import_ref.098), @Int.as.AddAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.1b1: @Int.as.AddAssignWith.impl.%Int.as.AddAssignWith.impl.Op.type.1 (%Int.as.AddAssignWith.impl.Op.type.a58fbe.2) = import_ref Core//prelude/types/int, Op, loaded [symbolic = @Int.as.AddAssignWith.impl.%Int.as.AddAssignWith.impl.Op.1 (constants.%Int.as.AddAssignWith.impl.Op.45b360.2)]
// CHECK:STDOUT:   %Core.import_ref.47b: Core.IntLiteral = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.AnyInt.impl.%N (constants.%N.fe9)]
// CHECK:STDOUT:   %Core.import_ref.895: @Int.as.AnyInt.impl.%Int.as.AnyInt.impl.AsInt.type (%Int.as.AnyInt.impl.AsInt.type.2ea) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.AnyInt.impl.%Int.as.AnyInt.impl.AsInt (constants.%Int.as.AnyInt.impl.AsInt.5e2)]
// CHECK:STDOUT:   %AnyInt.impl_witness_table = impl_witness_table (%Core.import_ref.47b, %Core.import_ref.895), @Int.as.AnyInt.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @TestIterate(%R.loc30_17.2: %Iterate_where.type.ec1) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Iterate.lookup_impl_witness: <witness> = lookup_impl_witness %R.loc30_17.1, @Iterate [symbolic = %Iterate.lookup_impl_witness (constants.%Iterate.lookup_impl_witness.18b)]
// CHECK:STDOUT:   %Iterate.facet.loc34_19.5: %Iterate.type = facet_value %R.as_type.loc30_61.1, (%Iterate.lookup_impl_witness) [symbolic = %Iterate.facet.loc34_19.5 (constants.%Iterate.facet.353)]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.loc34_19.5) [symbolic = %Iterate.WithSelf.NewCursor.type (constants.%Iterate.WithSelf.NewCursor.type.8fb)]
// CHECK:STDOUT:   %.loc34_19.19: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type, %Iterate.facet.loc34_19.5 [symbolic = %.loc34_19.19 (constants.%.985)]
// CHECK:STDOUT:   %impl.elem2.loc34_19.2: @TestIterate.%.loc34_19.19 (%.985) = impl_witness_access %Iterate.lookup_impl_witness, element2 [symbolic = %impl.elem2.loc34_19.2 (constants.%impl.elem2.8ec)]
// CHECK:STDOUT:   %specific_impl_fn.loc34_19.4: <specific function> = specific_impl_function %impl.elem2.loc34_19.2, @Iterate.WithSelf.NewCursor(%Iterate.facet.loc34_19.5) [symbolic = %specific_impl_fn.loc34_19.4 (constants.%specific_impl_fn.9ca)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %ptr: type = ptr_type %as_type [symbolic = %ptr (constants.%ptr.2ef)]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.loc34_19.5) [symbolic = %Iterate.WithSelf.Next.type (constants.%Iterate.WithSelf.Next.type.830)]
// CHECK:STDOUT:   %.loc34_19.20: type = fn_type_with_self_type %Iterate.WithSelf.Next.type, %Iterate.facet.loc34_19.5 [symbolic = %.loc34_19.20 (constants.%.f86)]
// CHECK:STDOUT:   %impl.elem3.loc34_19.2: @TestIterate.%.loc34_19.20 (%.f86) = impl_witness_access %Iterate.lookup_impl_witness, element3 [symbolic = %impl.elem3.loc34_19.2 (constants.%impl.elem3.5f7)]
// CHECK:STDOUT:   %specific_impl_fn.loc34_19.5: <specific function> = specific_impl_function %impl.elem3.loc34_19.2, @Iterate.WithSelf.Next(%Iterate.facet.loc34_19.5) [symbolic = %specific_impl_fn.loc34_19.5 (constants.%specific_impl_fn.9a8)]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%impl.elem1) [symbolic = %Destroy.WithSelf.Op.type (constants.%Destroy.WithSelf.Op.type.71c)]
// CHECK:STDOUT:   %.loc34_19.21: type = fn_type_with_self_type %Destroy.WithSelf.Op.type, %impl.elem1 [symbolic = %.loc34_19.21 (constants.%.fa0)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %impl.elem1, @Destroy [symbolic = %Destroy.lookup_impl_witness (constants.%Destroy.lookup_impl_witness.d26)]
// CHECK:STDOUT:   %impl.elem0.loc34_19.2: @TestIterate.%.loc34_19.21 (%.fa0) = impl_witness_access %Destroy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc34_19.2 (constants.%impl.elem0.544)]
// CHECK:STDOUT:   %specific_impl_fn.loc34_19.6: <specific function> = specific_impl_function %impl.elem0.loc34_19.2, @Destroy.WithSelf.Op(%impl.elem1) [symbolic = %specific_impl_fn.loc34_19.6 (constants.%specific_impl_fn.fa9)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%r.param: @TestIterate.%R.as_type.loc30_61.1 (%R.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %i.patt: %pattern_type.6b6 = value_binding_pattern i [concrete = constants.%i.patt]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %r.ref: @TestIterate.%R.as_type.loc30_61.1 (%R.as_type) = name_ref r, %r
// CHECK:STDOUT:     %impl.elem2.loc34_19.1: @TestIterate.%.loc34_19.19 (%.985) = impl_witness_access constants.%Iterate.lookup_impl_witness.18b, element2 [symbolic = %impl.elem2.loc34_19.2 (constants.%impl.elem2.8ec)]
// CHECK:STDOUT:     %bound_method.loc34_19.1: <bound method> = bound_method %r.ref, %impl.elem2.loc34_19.1
// CHECK:STDOUT:     %Iterate.facet.loc34_19.1: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.18b) [symbolic = %Iterate.facet.loc34_19.5 (constants.%Iterate.facet.353)]
// CHECK:STDOUT:     %.loc34_19.1: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc34_19.1 [symbolic = %Iterate.facet.loc34_19.5 (constants.%Iterate.facet.353)]
// CHECK:STDOUT:     %Iterate.facet.loc34_19.2: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.18b) [symbolic = %Iterate.facet.loc34_19.5 (constants.%Iterate.facet.353)]
// CHECK:STDOUT:     %.loc34_19.2: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc34_19.2 [symbolic = %Iterate.facet.loc34_19.5 (constants.%Iterate.facet.353)]
// CHECK:STDOUT:     %specific_impl_fn.loc34_19.1: <specific function> = specific_impl_function %impl.elem2.loc34_19.1, @Iterate.WithSelf.NewCursor(constants.%Iterate.facet.353) [symbolic = %specific_impl_fn.loc34_19.4 (constants.%specific_impl_fn.9ca)]
// CHECK:STDOUT:     %bound_method.loc34_19.2: <bound method> = bound_method %r.ref, %specific_impl_fn.loc34_19.1
// CHECK:STDOUT:     %var: ref @TestIterate.%as_type (%as_type.c8a) = var_storage invalid
// CHECK:STDOUT:     %Iterate.WithSelf.NewCursor.call: init @TestIterate.%as_type (%as_type.c8a) to %var = call %bound_method.loc34_19.2(%r.ref)
// CHECK:STDOUT:     assign %var, %Iterate.WithSelf.NewCursor.call
// CHECK:STDOUT:     br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.next:
// CHECK:STDOUT:     %addr: @TestIterate.%ptr (%ptr.2ef) = addr_of %var
// CHECK:STDOUT:     %impl.elem3.loc34_19.1: @TestIterate.%.loc34_19.20 (%.f86) = impl_witness_access constants.%Iterate.lookup_impl_witness.18b, element3 [symbolic = %impl.elem3.loc34_19.2 (constants.%impl.elem3.5f7)]
// CHECK:STDOUT:     %bound_method.loc34_19.3: <bound method> = bound_method %r.ref, %impl.elem3.loc34_19.1
// CHECK:STDOUT:     %Iterate.facet.loc34_19.3: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.18b) [symbolic = %Iterate.facet.loc34_19.5 (constants.%Iterate.facet.353)]
// CHECK:STDOUT:     %.loc34_19.3: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc34_19.3 [symbolic = %Iterate.facet.loc34_19.5 (constants.%Iterate.facet.353)]
// CHECK:STDOUT:     %Iterate.facet.loc34_19.4: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.18b) [symbolic = %Iterate.facet.loc34_19.5 (constants.%Iterate.facet.353)]
// CHECK:STDOUT:     %.loc34_19.4: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc34_19.4 [symbolic = %Iterate.facet.loc34_19.5 (constants.%Iterate.facet.353)]
// CHECK:STDOUT:     %.loc34_19.5: %Destroy.type = converted constants.%as_type.c8a, constants.%impl.elem1.295 [symbolic = %impl.elem1 (constants.%impl.elem1.295)]
// CHECK:STDOUT:     %.loc34_19.6: %Destroy.type = converted constants.%as_type.c8a, constants.%impl.elem1.295 [symbolic = %impl.elem1 (constants.%impl.elem1.295)]
// CHECK:STDOUT:     %specific_impl_fn.loc34_19.2: <specific function> = specific_impl_function %impl.elem3.loc34_19.1, @Iterate.WithSelf.Next(constants.%Iterate.facet.353) [symbolic = %specific_impl_fn.loc34_19.5 (constants.%specific_impl_fn.9a8)]
// CHECK:STDOUT:     %bound_method.loc34_19.4: <bound method> = bound_method %r.ref, %specific_impl_fn.loc34_19.2
// CHECK:STDOUT:     %.loc34_19.7: ref %Optional.5cf = temporary_storage
// CHECK:STDOUT:     %Iterate.WithSelf.Next.call: init %Optional.5cf to %.loc34_19.7 = call %bound_method.loc34_19.4(%r.ref, %addr)
// CHECK:STDOUT:     %.loc34_19.8: ref %Optional.5cf = temporary %.loc34_19.7, %Iterate.WithSelf.Next.call
// CHECK:STDOUT:     %.loc34_19.9: %Optional.HasValue.type.47f = specific_constant imports.%Core.import_ref.bb6, @Optional(constants.%OptionalStorage.facet.1a2) [concrete = constants.%Optional.HasValue.1ad]
// CHECK:STDOUT:     %HasValue.ref: %Optional.HasValue.type.47f = name_ref HasValue, %.loc34_19.9 [concrete = constants.%Optional.HasValue.1ad]
// CHECK:STDOUT:     %Optional.HasValue.bound: <bound method> = bound_method %.loc34_19.8, %HasValue.ref
// CHECK:STDOUT:     %Optional.HasValue.specific_fn: <specific function> = specific_function %HasValue.ref, @Optional.HasValue(constants.%OptionalStorage.facet.1a2) [concrete = constants.%Optional.HasValue.specific_fn.9b9]
// CHECK:STDOUT:     %bound_method.loc34_19.5: <bound method> = bound_method %.loc34_19.8, %Optional.HasValue.specific_fn
// CHECK:STDOUT:     %.loc34_19.10: %Optional.5cf = acquire_value %.loc34_19.8
// CHECK:STDOUT:     %Optional.HasValue.call: init bool = call %bound_method.loc34_19.5(%.loc34_19.10)
// CHECK:STDOUT:     %.loc34_19.11: bool = value_of_initializer %Optional.HasValue.call
// CHECK:STDOUT:     %.loc34_19.12: bool = converted %Optional.HasValue.call, %.loc34_19.11
// CHECK:STDOUT:     if %.loc34_19.12 br !for.body else br !for.done
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.body:
// CHECK:STDOUT:     %.loc34_19.13: %Optional.Get.type.bc1 = specific_constant imports.%Core.import_ref.daba, @Optional(constants.%OptionalStorage.facet.1a2) [concrete = constants.%Optional.Get.dda]
// CHECK:STDOUT:     %Get.ref: %Optional.Get.type.bc1 = name_ref Get, %.loc34_19.13 [concrete = constants.%Optional.Get.dda]
// CHECK:STDOUT:     %Optional.Get.bound: <bound method> = bound_method %.loc34_19.8, %Get.ref
// CHECK:STDOUT:     %Optional.Get.specific_fn: <specific function> = specific_function %Get.ref, @Optional.Get(constants.%OptionalStorage.facet.1a2) [concrete = constants.%Optional.Get.specific_fn.e58]
// CHECK:STDOUT:     %bound_method.loc34_19.6: <bound method> = bound_method %.loc34_19.8, %Optional.Get.specific_fn
// CHECK:STDOUT:     %.loc34_19.14: %Optional.5cf = acquire_value %.loc34_19.8
// CHECK:STDOUT:     %Optional.Get.call: init %i32 = call %bound_method.loc34_19.6(%.loc34_19.14)
// CHECK:STDOUT:     %.loc34_19.15: %i32 = value_of_initializer %Optional.Get.call
// CHECK:STDOUT:     %.loc34_19.16: %i32 = converted %Optional.Get.call, %.loc34_19.15
// CHECK:STDOUT:     %i32.loc34: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %i: %i32 = wrapper_binding i, %.loc34_19.16
// CHECK:STDOUT:     %sum.ref: ref %i32 = name_ref sum, %sum
// CHECK:STDOUT:     %i.ref: %i32 = name_ref i, %i
// CHECK:STDOUT:     %impl.elem0.loc35: %.583 = impl_witness_access constants.%AddAssignWith.impl_witness.211, element0 [concrete = constants.%Int.as.AddAssignWith.impl.Op.f51aaf.2]
// CHECK:STDOUT:     %bound_method.loc35_9.1: <bound method> = bound_method %sum.ref, %impl.elem0.loc35
// CHECK:STDOUT:     %specific_fn.loc35: <specific function> = specific_function %impl.elem0.loc35, @Int.as.AddAssignWith.impl.Op.1(constants.%int_32, constants.%ImplicitAs.facet.354) [concrete = constants.%Int.as.AddAssignWith.impl.Op.specific_fn.bd8c6b.1]
// CHECK:STDOUT:     %bound_method.loc35_9.2: <bound method> = bound_method %sum.ref, %specific_fn.loc35
// CHECK:STDOUT:     %.loc35: %Int.as.AddAssignWith.impl.Op.type.56278f.1 = specific_constant imports.%Core.Op.1b1, @Int.as.AddAssignWith.impl(constants.%int_32, constants.%ImplicitAs.facet.354) [concrete = constants.%Int.as.AddAssignWith.impl.Op.f51aaf.1]
// CHECK:STDOUT:     %Op.ref: %Int.as.AddAssignWith.impl.Op.type.56278f.1 = name_ref Op, %.loc35 [concrete = constants.%Int.as.AddAssignWith.impl.Op.f51aaf.1]
// CHECK:STDOUT:     %Int.as.AddAssignWith.impl.Op.bound: <bound method> = bound_method %sum.ref, %Op.ref
// CHECK:STDOUT:     %Int.as.AddAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref, @Int.as.AddAssignWith.impl.Op.2(constants.%int_32, constants.%ImplicitAs.facet.354) [concrete = constants.%Int.as.AddAssignWith.impl.Op.specific_fn.bd8c6b.2]
// CHECK:STDOUT:     %bound_method.loc35_9.3: <bound method> = bound_method %sum.ref, %Int.as.AddAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:     %Int.as.AddAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc35_9.3(%sum.ref, %i.ref)
// CHECK:STDOUT:     %Destroy.Op.bound.loc34_19.1: <bound method> = bound_method %.loc34_19.8, constants.%Destroy.Op.1a2547.9
// CHECK:STDOUT:     %Destroy.Op.call.loc34_19.1: init %empty_tuple.type = call %Destroy.Op.bound.loc34_19.1(%.loc34_19.8)
// CHECK:STDOUT:     br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.done:
// CHECK:STDOUT:     %Destroy.Op.bound.loc34_19.2: <bound method> = bound_method %.loc34_19.8, constants.%Destroy.Op.1a2547.9
// CHECK:STDOUT:     %Destroy.Op.call.loc34_19.2: init %empty_tuple.type = call %Destroy.Op.bound.loc34_19.2(%.loc34_19.8)
// CHECK:STDOUT:     %impl.elem0.loc34_19.1: @TestIterate.%.loc34_19.21 (%.fa0) = impl_witness_access constants.%Destroy.lookup_impl_witness.d26, element0 [symbolic = %impl.elem0.loc34_19.2 (constants.%impl.elem0.544)]
// CHECK:STDOUT:     %bound_method.loc34_19.7: <bound method> = bound_method %var, %impl.elem0.loc34_19.1
// CHECK:STDOUT:     %.loc34_19.17: %Destroy.type = converted constants.%as_type.c8a, constants.%impl.elem1.295 [symbolic = %impl.elem1 (constants.%impl.elem1.295)]
// CHECK:STDOUT:     %.loc34_19.18: %Destroy.type = converted constants.%as_type.c8a, constants.%impl.elem1.295 [symbolic = %impl.elem1 (constants.%impl.elem1.295)]
// CHECK:STDOUT:     %specific_impl_fn.loc34_19.3: <specific function> = specific_impl_function %impl.elem0.loc34_19.1, @Destroy.WithSelf.Op(constants.%impl.elem1.295) [symbolic = %specific_impl_fn.loc34_19.6 (constants.%specific_impl_fn.fa9)]
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
// CHECK:STDOUT: fn @Destroy.Op.loc34_19.6(%self.param: ref %Optional.5cf) {
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
// CHECK:STDOUT:   %impl.elem2.loc49: %.9e1 = impl_witness_access constants.%Iterate.impl_witness.55e, element2 [concrete = constants.%T.as_type.as.Iterate.impl.NewCursor.9cc]
// CHECK:STDOUT:   %bound_method.loc49_19.1: <bound method> = bound_method %m.ref, %impl.elem2.loc49
// CHECK:STDOUT:   %facet_value.loc49_19.1: %CppRange_where.type.d682d6.1 = facet_value constants.%MutableRange, (constants.%custom_witness.df9cc1.3, constants.%Copy.impl_witness.0e0, constants.%custom_witness.ed3, constants.%custom_witness.955, constants.%custom_witness.ef8, constants.%custom_witness.444, constants.%custom_witness.981) [concrete = constants.%facet_value.036]
// CHECK:STDOUT:   %.loc49_19.1: %CppRange_where.type.d682d6.1 = converted constants.%MutableRange, %facet_value.loc49_19.1 [concrete = constants.%facet_value.036]
// CHECK:STDOUT:   %facet_value.loc49_19.2: %CppRange_where.type.d682d6.1 = facet_value constants.%MutableRange, (constants.%custom_witness.df9cc1.3, constants.%Copy.impl_witness.0e0, constants.%custom_witness.ed3, constants.%custom_witness.955, constants.%custom_witness.ef8, constants.%custom_witness.444, constants.%custom_witness.981) [concrete = constants.%facet_value.036]
// CHECK:STDOUT:   %.loc49_19.2: %CppRange_where.type.d682d6.1 = converted constants.%MutableRange, %facet_value.loc49_19.2 [concrete = constants.%facet_value.036]
// CHECK:STDOUT:   %specific_fn.loc49_19.1: <specific function> = specific_function %impl.elem2.loc49, @T.as_type.as.Iterate.impl.NewCursor(constants.%facet_value.036) [concrete = constants.%T.as_type.as.Iterate.impl.NewCursor.specific_fn.6a8]
// CHECK:STDOUT:   %bound_method.loc49_19.2: <bound method> = bound_method %m.ref, %specific_fn.loc49_19.1
// CHECK:STDOUT:   %var.loc49: ref %tuple.type.508 = var_storage invalid
// CHECK:STDOUT:   %.loc49_18.1: %MutableRange = acquire_value %m.ref
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.call.loc49: init %tuple.type.508 to %var.loc49 = call %bound_method.loc49_19.2(%.loc49_18.1)
// CHECK:STDOUT:   assign %var.loc49, %T.as_type.as.Iterate.impl.NewCursor.call.loc49
// CHECK:STDOUT:   br !for.next.loc49
// CHECK:STDOUT:
// CHECK:STDOUT: !for.next.loc49:
// CHECK:STDOUT:   %addr.loc49: %ptr.45c = addr_of %var.loc49
// CHECK:STDOUT:   %impl.elem3.loc49: %.4e4 = impl_witness_access constants.%Iterate.impl_witness.55e, element3 [concrete = constants.%T.as_type.as.Iterate.impl.Next.6ab]
// CHECK:STDOUT:   %bound_method.loc49_19.3: <bound method> = bound_method %m.ref, %impl.elem3.loc49
// CHECK:STDOUT:   %facet_value.loc49_19.3: %CppRange_where.type.d682d6.1 = facet_value constants.%MutableRange, (constants.%custom_witness.df9cc1.3, constants.%Copy.impl_witness.0e0, constants.%custom_witness.ed3, constants.%custom_witness.955, constants.%custom_witness.ef8, constants.%custom_witness.444, constants.%custom_witness.981) [concrete = constants.%facet_value.036]
// CHECK:STDOUT:   %.loc49_19.3: %CppRange_where.type.d682d6.1 = converted constants.%MutableRange, %facet_value.loc49_19.3 [concrete = constants.%facet_value.036]
// CHECK:STDOUT:   %facet_value.loc49_19.4: %CppRange_where.type.d682d6.1 = facet_value constants.%MutableRange, (constants.%custom_witness.df9cc1.3, constants.%Copy.impl_witness.0e0, constants.%custom_witness.ed3, constants.%custom_witness.955, constants.%custom_witness.ef8, constants.%custom_witness.444, constants.%custom_witness.981) [concrete = constants.%facet_value.036]
// CHECK:STDOUT:   %.loc49_19.4: %CppRange_where.type.d682d6.1 = converted constants.%MutableRange, %facet_value.loc49_19.4 [concrete = constants.%facet_value.036]
// CHECK:STDOUT:   %specific_fn.loc49_19.2: <specific function> = specific_function %impl.elem3.loc49, @T.as_type.as.Iterate.impl.Next(constants.%facet_value.036) [concrete = constants.%T.as_type.as.Iterate.impl.Next.specific_fn.891]
// CHECK:STDOUT:   %bound_method.loc49_19.4: <bound method> = bound_method %m.ref, %specific_fn.loc49_19.2
// CHECK:STDOUT:   %.loc49_19.5: ref %Optional.5cf = temporary_storage
// CHECK:STDOUT:   %.loc49_18.2: %MutableRange = acquire_value %m.ref
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.call.loc49: init %Optional.5cf to %.loc49_19.5 = call %bound_method.loc49_19.4(%.loc49_18.2, %addr.loc49)
// CHECK:STDOUT:   %.loc49_19.6: ref %Optional.5cf = temporary %.loc49_19.5, %T.as_type.as.Iterate.impl.Next.call.loc49
// CHECK:STDOUT:   %.loc49_19.7: %Optional.HasValue.type.47f = specific_constant imports.%Core.import_ref.bb6, @Optional(constants.%OptionalStorage.facet.1a2) [concrete = constants.%Optional.HasValue.1ad]
// CHECK:STDOUT:   %HasValue.ref.loc49: %Optional.HasValue.type.47f = name_ref HasValue, %.loc49_19.7 [concrete = constants.%Optional.HasValue.1ad]
// CHECK:STDOUT:   %Optional.HasValue.bound.loc49: <bound method> = bound_method %.loc49_19.6, %HasValue.ref.loc49
// CHECK:STDOUT:   %Optional.HasValue.specific_fn.loc49: <specific function> = specific_function %HasValue.ref.loc49, @Optional.HasValue(constants.%OptionalStorage.facet.1a2) [concrete = constants.%Optional.HasValue.specific_fn.9b9]
// CHECK:STDOUT:   %bound_method.loc49_19.5: <bound method> = bound_method %.loc49_19.6, %Optional.HasValue.specific_fn.loc49
// CHECK:STDOUT:   %.loc49_19.8: %Optional.5cf = acquire_value %.loc49_19.6
// CHECK:STDOUT:   %Optional.HasValue.call.loc49: init bool = call %bound_method.loc49_19.5(%.loc49_19.8)
// CHECK:STDOUT:   %.loc49_19.9: bool = value_of_initializer %Optional.HasValue.call.loc49
// CHECK:STDOUT:   %.loc49_19.10: bool = converted %Optional.HasValue.call.loc49, %.loc49_19.9
// CHECK:STDOUT:   if %.loc49_19.10 br !for.body.loc49 else br !for.done.loc49
// CHECK:STDOUT:
// CHECK:STDOUT: !for.body.loc49:
// CHECK:STDOUT:   %.loc49_19.11: %Optional.Get.type.bc1 = specific_constant imports.%Core.import_ref.daba, @Optional(constants.%OptionalStorage.facet.1a2) [concrete = constants.%Optional.Get.dda]
// CHECK:STDOUT:   %Get.ref.loc49: %Optional.Get.type.bc1 = name_ref Get, %.loc49_19.11 [concrete = constants.%Optional.Get.dda]
// CHECK:STDOUT:   %Optional.Get.bound.loc49: <bound method> = bound_method %.loc49_19.6, %Get.ref.loc49
// CHECK:STDOUT:   %Optional.Get.specific_fn.loc49: <specific function> = specific_function %Get.ref.loc49, @Optional.Get(constants.%OptionalStorage.facet.1a2) [concrete = constants.%Optional.Get.specific_fn.e58]
// CHECK:STDOUT:   %bound_method.loc49_19.6: <bound method> = bound_method %.loc49_19.6, %Optional.Get.specific_fn.loc49
// CHECK:STDOUT:   %.loc49_19.12: %Optional.5cf = acquire_value %.loc49_19.6
// CHECK:STDOUT:   %Optional.Get.call.loc49: init %i32 = call %bound_method.loc49_19.6(%.loc49_19.12)
// CHECK:STDOUT:   %.loc49_19.13: %i32 = value_of_initializer %Optional.Get.call.loc49
// CHECK:STDOUT:   %.loc49_19.14: %i32 = converted %Optional.Get.call.loc49, %.loc49_19.13
// CHECK:STDOUT:   %i32.loc49: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %i.loc49: %i32 = wrapper_binding i, %.loc49_19.14
// CHECK:STDOUT:   %sum.ref.loc50: ref %i32 = name_ref sum, %sum
// CHECK:STDOUT:   %i.ref.loc50: %i32 = name_ref i, %i.loc49
// CHECK:STDOUT:   %impl.elem0.loc50: %.583 = impl_witness_access constants.%AddAssignWith.impl_witness.211, element0 [concrete = constants.%Int.as.AddAssignWith.impl.Op.f51aaf.2]
// CHECK:STDOUT:   %bound_method.loc50_9.1: <bound method> = bound_method %sum.ref.loc50, %impl.elem0.loc50
// CHECK:STDOUT:   %specific_fn.loc50: <specific function> = specific_function %impl.elem0.loc50, @Int.as.AddAssignWith.impl.Op.1(constants.%int_32, constants.%ImplicitAs.facet.354) [concrete = constants.%Int.as.AddAssignWith.impl.Op.specific_fn.bd8c6b.1]
// CHECK:STDOUT:   %bound_method.loc50_9.2: <bound method> = bound_method %sum.ref.loc50, %specific_fn.loc50
// CHECK:STDOUT:   %.loc50: %Int.as.AddAssignWith.impl.Op.type.56278f.1 = specific_constant imports.%Core.Op.1b1, @Int.as.AddAssignWith.impl(constants.%int_32, constants.%ImplicitAs.facet.354) [concrete = constants.%Int.as.AddAssignWith.impl.Op.f51aaf.1]
// CHECK:STDOUT:   %Op.ref.loc50: %Int.as.AddAssignWith.impl.Op.type.56278f.1 = name_ref Op, %.loc50 [concrete = constants.%Int.as.AddAssignWith.impl.Op.f51aaf.1]
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.bound.loc50: <bound method> = bound_method %sum.ref.loc50, %Op.ref.loc50
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.specific_fn.loc50: <specific function> = specific_function %Op.ref.loc50, @Int.as.AddAssignWith.impl.Op.2(constants.%int_32, constants.%ImplicitAs.facet.354) [concrete = constants.%Int.as.AddAssignWith.impl.Op.specific_fn.bd8c6b.2]
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
// CHECK:STDOUT:   %impl.elem2.loc55: %.5d9 = impl_witness_access constants.%Iterate.impl_witness.1f3, element2 [concrete = constants.%T.as_type.as.Iterate.impl.NewCursor.2b7]
// CHECK:STDOUT:   %bound_method.loc55_19.1: <bound method> = bound_method %c.ref, %impl.elem2.loc55
// CHECK:STDOUT:   %facet_value.loc55_19.1: %CppRange_where.type.d682d6.1 = facet_value constants.%ConstRange, (constants.%custom_witness.df9cc1.3, constants.%Copy.impl_witness.0e0, constants.%custom_witness.23e, constants.%custom_witness.6e8, constants.%custom_witness.adb, constants.%custom_witness.746, constants.%custom_witness.941) [concrete = constants.%facet_value.f06]
// CHECK:STDOUT:   %.loc55_19.1: %CppRange_where.type.d682d6.1 = converted constants.%ConstRange, %facet_value.loc55_19.1 [concrete = constants.%facet_value.f06]
// CHECK:STDOUT:   %facet_value.loc55_19.2: %CppRange_where.type.d682d6.1 = facet_value constants.%ConstRange, (constants.%custom_witness.df9cc1.3, constants.%Copy.impl_witness.0e0, constants.%custom_witness.23e, constants.%custom_witness.6e8, constants.%custom_witness.adb, constants.%custom_witness.746, constants.%custom_witness.941) [concrete = constants.%facet_value.f06]
// CHECK:STDOUT:   %.loc55_19.2: %CppRange_where.type.d682d6.1 = converted constants.%ConstRange, %facet_value.loc55_19.2 [concrete = constants.%facet_value.f06]
// CHECK:STDOUT:   %specific_fn.loc55_19.1: <specific function> = specific_function %impl.elem2.loc55, @T.as_type.as.Iterate.impl.NewCursor(constants.%facet_value.f06) [concrete = constants.%T.as_type.as.Iterate.impl.NewCursor.specific_fn.4ae]
// CHECK:STDOUT:   %bound_method.loc55_19.2: <bound method> = bound_method %c.ref, %specific_fn.loc55_19.1
// CHECK:STDOUT:   %var.loc55: ref %tuple.type.69f = var_storage invalid
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.call.loc55: init %tuple.type.69f to %var.loc55 = call %bound_method.loc55_19.2(%c.ref)
// CHECK:STDOUT:   assign %var.loc55, %T.as_type.as.Iterate.impl.NewCursor.call.loc55
// CHECK:STDOUT:   br !for.next.loc55
// CHECK:STDOUT:
// CHECK:STDOUT: !for.next.loc55:
// CHECK:STDOUT:   %addr.loc55: %ptr.c5c = addr_of %var.loc55
// CHECK:STDOUT:   %impl.elem3.loc55: %.712 = impl_witness_access constants.%Iterate.impl_witness.1f3, element3 [concrete = constants.%T.as_type.as.Iterate.impl.Next.e83]
// CHECK:STDOUT:   %bound_method.loc55_19.3: <bound method> = bound_method %c.ref, %impl.elem3.loc55
// CHECK:STDOUT:   %facet_value.loc55_19.3: %CppRange_where.type.d682d6.1 = facet_value constants.%ConstRange, (constants.%custom_witness.df9cc1.3, constants.%Copy.impl_witness.0e0, constants.%custom_witness.23e, constants.%custom_witness.6e8, constants.%custom_witness.adb, constants.%custom_witness.746, constants.%custom_witness.941) [concrete = constants.%facet_value.f06]
// CHECK:STDOUT:   %.loc55_19.3: %CppRange_where.type.d682d6.1 = converted constants.%ConstRange, %facet_value.loc55_19.3 [concrete = constants.%facet_value.f06]
// CHECK:STDOUT:   %facet_value.loc55_19.4: %CppRange_where.type.d682d6.1 = facet_value constants.%ConstRange, (constants.%custom_witness.df9cc1.3, constants.%Copy.impl_witness.0e0, constants.%custom_witness.23e, constants.%custom_witness.6e8, constants.%custom_witness.adb, constants.%custom_witness.746, constants.%custom_witness.941) [concrete = constants.%facet_value.f06]
// CHECK:STDOUT:   %.loc55_19.4: %CppRange_where.type.d682d6.1 = converted constants.%ConstRange, %facet_value.loc55_19.4 [concrete = constants.%facet_value.f06]
// CHECK:STDOUT:   %specific_fn.loc55_19.2: <specific function> = specific_function %impl.elem3.loc55, @T.as_type.as.Iterate.impl.Next(constants.%facet_value.f06) [concrete = constants.%T.as_type.as.Iterate.impl.Next.specific_fn.0ce]
// CHECK:STDOUT:   %bound_method.loc55_19.4: <bound method> = bound_method %c.ref, %specific_fn.loc55_19.2
// CHECK:STDOUT:   %.loc55_19.5: ref %Optional.5cf = temporary_storage
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.call.loc55: init %Optional.5cf to %.loc55_19.5 = call %bound_method.loc55_19.4(%c.ref, %addr.loc55)
// CHECK:STDOUT:   %.loc55_19.6: ref %Optional.5cf = temporary %.loc55_19.5, %T.as_type.as.Iterate.impl.Next.call.loc55
// CHECK:STDOUT:   %.loc55_19.7: %Optional.HasValue.type.47f = specific_constant imports.%Core.import_ref.bb6, @Optional(constants.%OptionalStorage.facet.1a2) [concrete = constants.%Optional.HasValue.1ad]
// CHECK:STDOUT:   %HasValue.ref.loc55: %Optional.HasValue.type.47f = name_ref HasValue, %.loc55_19.7 [concrete = constants.%Optional.HasValue.1ad]
// CHECK:STDOUT:   %Optional.HasValue.bound.loc55: <bound method> = bound_method %.loc55_19.6, %HasValue.ref.loc55
// CHECK:STDOUT:   %Optional.HasValue.specific_fn.loc55: <specific function> = specific_function %HasValue.ref.loc55, @Optional.HasValue(constants.%OptionalStorage.facet.1a2) [concrete = constants.%Optional.HasValue.specific_fn.9b9]
// CHECK:STDOUT:   %bound_method.loc55_19.5: <bound method> = bound_method %.loc55_19.6, %Optional.HasValue.specific_fn.loc55
// CHECK:STDOUT:   %.loc55_19.8: %Optional.5cf = acquire_value %.loc55_19.6
// CHECK:STDOUT:   %Optional.HasValue.call.loc55: init bool = call %bound_method.loc55_19.5(%.loc55_19.8)
// CHECK:STDOUT:   %.loc55_19.9: bool = value_of_initializer %Optional.HasValue.call.loc55
// CHECK:STDOUT:   %.loc55_19.10: bool = converted %Optional.HasValue.call.loc55, %.loc55_19.9
// CHECK:STDOUT:   if %.loc55_19.10 br !for.body.loc55 else br !for.done.loc55
// CHECK:STDOUT:
// CHECK:STDOUT: !for.body.loc55:
// CHECK:STDOUT:   %.loc55_19.11: %Optional.Get.type.bc1 = specific_constant imports.%Core.import_ref.daba, @Optional(constants.%OptionalStorage.facet.1a2) [concrete = constants.%Optional.Get.dda]
// CHECK:STDOUT:   %Get.ref.loc55: %Optional.Get.type.bc1 = name_ref Get, %.loc55_19.11 [concrete = constants.%Optional.Get.dda]
// CHECK:STDOUT:   %Optional.Get.bound.loc55: <bound method> = bound_method %.loc55_19.6, %Get.ref.loc55
// CHECK:STDOUT:   %Optional.Get.specific_fn.loc55: <specific function> = specific_function %Get.ref.loc55, @Optional.Get(constants.%OptionalStorage.facet.1a2) [concrete = constants.%Optional.Get.specific_fn.e58]
// CHECK:STDOUT:   %bound_method.loc55_19.6: <bound method> = bound_method %.loc55_19.6, %Optional.Get.specific_fn.loc55
// CHECK:STDOUT:   %.loc55_19.12: %Optional.5cf = acquire_value %.loc55_19.6
// CHECK:STDOUT:   %Optional.Get.call.loc55: init %i32 = call %bound_method.loc55_19.6(%.loc55_19.12)
// CHECK:STDOUT:   %.loc55_19.13: %i32 = value_of_initializer %Optional.Get.call.loc55
// CHECK:STDOUT:   %.loc55_19.14: %i32 = converted %Optional.Get.call.loc55, %.loc55_19.13
// CHECK:STDOUT:   %i32.loc55: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %i.loc55: %i32 = wrapper_binding i, %.loc55_19.14
// CHECK:STDOUT:   %sum.ref.loc56: ref %i32 = name_ref sum, %sum
// CHECK:STDOUT:   %i.ref.loc56: %i32 = name_ref i, %i.loc55
// CHECK:STDOUT:   %impl.elem0.loc56: %.583 = impl_witness_access constants.%AddAssignWith.impl_witness.211, element0 [concrete = constants.%Int.as.AddAssignWith.impl.Op.f51aaf.2]
// CHECK:STDOUT:   %bound_method.loc56_9.1: <bound method> = bound_method %sum.ref.loc56, %impl.elem0.loc56
// CHECK:STDOUT:   %specific_fn.loc56: <specific function> = specific_function %impl.elem0.loc56, @Int.as.AddAssignWith.impl.Op.1(constants.%int_32, constants.%ImplicitAs.facet.354) [concrete = constants.%Int.as.AddAssignWith.impl.Op.specific_fn.bd8c6b.1]
// CHECK:STDOUT:   %bound_method.loc56_9.2: <bound method> = bound_method %sum.ref.loc56, %specific_fn.loc56
// CHECK:STDOUT:   %.loc56: %Int.as.AddAssignWith.impl.Op.type.56278f.1 = specific_constant imports.%Core.Op.1b1, @Int.as.AddAssignWith.impl(constants.%int_32, constants.%ImplicitAs.facet.354) [concrete = constants.%Int.as.AddAssignWith.impl.Op.f51aaf.1]
// CHECK:STDOUT:   %Op.ref.loc56: %Int.as.AddAssignWith.impl.Op.type.56278f.1 = name_ref Op, %.loc56 [concrete = constants.%Int.as.AddAssignWith.impl.Op.f51aaf.1]
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.bound.loc56: <bound method> = bound_method %sum.ref.loc56, %Op.ref.loc56
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.specific_fn.loc56: <specific function> = specific_function %Op.ref.loc56, @Int.as.AddAssignWith.impl.Op.2(constants.%int_32, constants.%ImplicitAs.facet.354) [concrete = constants.%Int.as.AddAssignWith.impl.Op.specific_fn.bd8c6b.2]
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
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.b2da
// CHECK:STDOUT:   %r.param_patt.loc30_59.2 => constants.%r.param_patt.774
// CHECK:STDOUT:   %r.patt.loc30_59.2 => constants.%r.patt.54b
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestIterate(constants.%facet_value.0de) {
// CHECK:STDOUT:   %R.patt.loc30_17.2 => constants.%R.patt
// CHECK:STDOUT:   %R.loc30_17.1 => constants.%facet_value.0de
// CHECK:STDOUT:   %R.as_type.loc30_61.1 => constants.%MutableRange
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.e635
// CHECK:STDOUT:   %r.param_patt.loc30_59.2 => constants.%r.param_patt.c79
// CHECK:STDOUT:   %r.patt.loc30_59.2 => constants.%r.patt.16f
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc30 => constants.%complete_type.357
// CHECK:STDOUT:   %Iterate.lookup_impl_witness => constants.%Iterate.impl_witness.55e
// CHECK:STDOUT:   %Iterate.facet.loc34_19.5 => constants.%Iterate.facet.523
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type => constants.%Iterate.WithSelf.NewCursor.type.dfd
// CHECK:STDOUT:   %.loc34_19.19 => constants.%.9e1
// CHECK:STDOUT:   %impl.elem2.loc34_19.2 => constants.%T.as_type.as.Iterate.impl.NewCursor.9cc
// CHECK:STDOUT:   %specific_impl_fn.loc34_19.4 => constants.%T.as_type.as.Iterate.impl.NewCursor.specific_fn.6a8
// CHECK:STDOUT:   %impl.elem1 => constants.%Destroy.facet.30f
// CHECK:STDOUT:   %as_type => constants.%tuple.type.508
// CHECK:STDOUT:   %require_complete.1 => constants.%complete_type.c28
// CHECK:STDOUT:   %ptr => constants.%ptr.45c
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type => constants.%Iterate.WithSelf.Next.type.b64
// CHECK:STDOUT:   %.loc34_19.20 => constants.%.4e4
// CHECK:STDOUT:   %impl.elem3.loc34_19.2 => constants.%T.as_type.as.Iterate.impl.Next.6ab
// CHECK:STDOUT:   %specific_impl_fn.loc34_19.5 => constants.%T.as_type.as.Iterate.impl.Next.specific_fn.891
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.c6c
// CHECK:STDOUT:   %.loc34_19.21 => constants.%.0a4
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.df9cc1.10
// CHECK:STDOUT:   %impl.elem0.loc34_19.2 => constants.%Destroy.Op.1a2547.10
// CHECK:STDOUT:   %specific_impl_fn.loc34_19.6 => constants.%Destroy.Op.1a2547.10
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestIterate(constants.%facet_value.b62) {
// CHECK:STDOUT:   %R.patt.loc30_17.2 => constants.%R.patt
// CHECK:STDOUT:   %R.loc30_17.1 => constants.%facet_value.b62
// CHECK:STDOUT:   %R.as_type.loc30_61.1 => constants.%ConstRange
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.0ba
// CHECK:STDOUT:   %r.param_patt.loc30_59.2 => constants.%r.param_patt.83e
// CHECK:STDOUT:   %r.patt.loc30_59.2 => constants.%r.patt.116
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc30 => constants.%complete_type.357
// CHECK:STDOUT:   %Iterate.lookup_impl_witness => constants.%Iterate.impl_witness.1f3
// CHECK:STDOUT:   %Iterate.facet.loc34_19.5 => constants.%Iterate.facet.910
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type => constants.%Iterate.WithSelf.NewCursor.type.ba4
// CHECK:STDOUT:   %.loc34_19.19 => constants.%.5d9
// CHECK:STDOUT:   %impl.elem2.loc34_19.2 => constants.%T.as_type.as.Iterate.impl.NewCursor.2b7
// CHECK:STDOUT:   %specific_impl_fn.loc34_19.4 => constants.%T.as_type.as.Iterate.impl.NewCursor.specific_fn.4ae
// CHECK:STDOUT:   %impl.elem1 => constants.%Destroy.facet.e4c
// CHECK:STDOUT:   %as_type => constants.%tuple.type.69f
// CHECK:STDOUT:   %require_complete.1 => constants.%complete_type.8d5
// CHECK:STDOUT:   %ptr => constants.%ptr.c5c
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type => constants.%Iterate.WithSelf.Next.type.dec
// CHECK:STDOUT:   %.loc34_19.20 => constants.%.712
// CHECK:STDOUT:   %impl.elem3.loc34_19.2 => constants.%T.as_type.as.Iterate.impl.Next.e83
// CHECK:STDOUT:   %specific_impl_fn.loc34_19.5 => constants.%T.as_type.as.Iterate.impl.Next.specific_fn.0ce
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
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Get.type.d3c: type = fn_type @T.as_type.as.OptionalStorage.impl.Get, @T.as_type.as.OptionalStorage.impl(%T.283) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Get.da2: %T.as_type.as.OptionalStorage.impl.Get.type.d3c = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Has.type.eb9: type = fn_type @T.as_type.as.OptionalStorage.impl.Has, @T.as_type.as.OptionalStorage.impl(%T.283) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Has.c7e: %T.as_type.as.OptionalStorage.impl.Has.type.eb9 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Some.type.7eb: type = fn_type @T.as_type.as.OptionalStorage.impl.Some, @T.as_type.as.OptionalStorage.impl(%T.283) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Some.2b4: %T.as_type.as.OptionalStorage.impl.Some.type.7eb = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.None.type.d4d: type = fn_type @T.as_type.as.OptionalStorage.impl.None, @T.as_type.as.OptionalStorage.impl(%T.283) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.None.d64: %T.as_type.as.OptionalStorage.impl.None.type.d4d = struct_value () [symbolic]
// CHECK:STDOUT:   %.Self.500: %Iterate.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %CppRange.type: type = facet_type <@CppRange> [concrete]
// CHECK:STDOUT:   %.Self.e0b: %CppRange.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %CppRangeForIterate.lookup_impl_witness.50b: <witness> = lookup_impl_witness %.Self.e0b, @CppRangeForIterate [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.32d: type = impl_witness_access %CppRangeForIterate.lookup_impl_witness.50b, element0 [symbolic_self]
// CHECK:STDOUT:   %CppUnsafeDeref.lookup_impl_witness.851: <witness> = lookup_impl_witness %impl.elem0.32d, @CppUnsafeDeref [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.10d: type = impl_witness_access %CppUnsafeDeref.lookup_impl_witness.851, element0 [symbolic_self]
// CHECK:STDOUT:   %CppRange_where.type.d682d6.1: type = facet_type <@CppRange where %impl.elem0.10d impls @Copy> [concrete]
// CHECK:STDOUT:   %T.f45530.2: %CppRange_where.type.d682d6.1 = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %CppRangeForIterate.lookup_impl_witness.b581a8.2: <witness> = lookup_impl_witness %T.f45530.2, @CppRangeForIterate [symbolic]
// CHECK:STDOUT:   %impl.elem1.c2dc15.2: type = impl_witness_access %CppRangeForIterate.lookup_impl_witness.b581a8.2, element1 [symbolic]
// CHECK:STDOUT:   %impl.elem0.5f971c.2: type = impl_witness_access %CppRangeForIterate.lookup_impl_witness.b581a8.2, element0 [symbolic]
// CHECK:STDOUT:   %tuple.type.1ba: type = tuple_type (%impl.elem0.5f971c.2, %impl.elem1.c2dc15.2) [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.812: <witness> = lookup_impl_witness %tuple.type.1ba, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.9cd: %Destroy.type = facet_value %tuple.type.1ba, (%Destroy.lookup_impl_witness.812) [symbolic]
// CHECK:STDOUT:   %CppUnsafeDeref.lookup_impl_witness.87e543.2: <witness> = lookup_impl_witness %impl.elem0.5f971c.2, @CppUnsafeDeref [symbolic]
// CHECK:STDOUT:   %impl.elem0.493765.2: type = impl_witness_access %CppUnsafeDeref.lookup_impl_witness.87e543.2, element0 [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.35816c.2: <witness> = lookup_impl_witness %impl.elem0.493765.2, @Destroy [symbolic]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.167: <witness> = lookup_impl_witness %impl.elem0.493765.2, @Copy [symbolic]
// CHECK:STDOUT:   %facet_value.f93: %facet_type.f48 = facet_value %impl.elem0.493765.2, (%Destroy.lookup_impl_witness.35816c.2, %Copy.lookup_impl_witness.167) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.type.5dc: type = fn_type @T.as_type.as.Iterate.impl.Next, @T.as_type.as.Iterate.impl(%T.f45530.2) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.b73: %T.as_type.as.Iterate.impl.Next.type.5dc = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.type.b5d: type = fn_type @T.as_type.as.Iterate.impl.NewCursor, @T.as_type.as.Iterate.impl(%T.f45530.2) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.21f: %T.as_type.as.Iterate.impl.NewCursor.type.b5d = struct_value () [symbolic]
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
// CHECK:STDOUT:   %facet_value.bf1: %facet_type.f48 = facet_value %ValueType, (%custom_witness.e7a31c.1, %custom_witness.d0da43.1) [concrete]
// CHECK:STDOUT:   %Iterate.lookup_impl_witness.1d8: <witness> = lookup_impl_witness %.Self.500, @Iterate [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.294: %facet_type.f48 = impl_witness_access %Iterate.lookup_impl_witness.1d8, element0 [symbolic_self]
// CHECK:STDOUT:   %Iterate_where.type.502: type = facet_type <@Iterate where %impl.elem0.294 = %facet_value.bf1> [concrete]
// CHECK:STDOUT:   %pattern_type.4c7: type = pattern_type %Iterate_where.type.502 [concrete]
// CHECK:STDOUT:   %R.patt: %pattern_type.4c7 = symbolic_binding_pattern R, 0 [symbolic]
// CHECK:STDOUT:   %R: %Iterate_where.type.502 = symbolic_binding R, 0 [symbolic]
// CHECK:STDOUT:   %R.as_type: type = facet_access_type %R [symbolic]
// CHECK:STDOUT:   %pattern_type.d2d: type = pattern_type %R.as_type [symbolic]
// CHECK:STDOUT:   %r.patt.d27: %pattern_type.d2d = ref_binding_pattern r [symbolic]
// CHECK:STDOUT:   %r.param_patt.a41: %pattern_type.d2d = var_param_pattern %r.patt.d27 [symbolic]
// CHECK:STDOUT:   %i.patt: %pattern_type.e9f = value_binding_pattern i [concrete]
// CHECK:STDOUT:   %Iterate.lookup_impl_witness.06b: <witness> = lookup_impl_witness %R, @Iterate [symbolic]
// CHECK:STDOUT:   %Iterate.facet.dbf: %Iterate.type = facet_value %R.as_type, (%Iterate.lookup_impl_witness.06b) [symbolic]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.1b1: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.dbf) [symbolic]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.705: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.dbf) [symbolic]
// CHECK:STDOUT:   %.1c2: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.1b1, %Iterate.facet.dbf [symbolic]
// CHECK:STDOUT:   %impl.elem2.49b: %.1c2 = impl_witness_access %Iterate.lookup_impl_witness.06b, element2 [symbolic]
// CHECK:STDOUT:   %impl.elem1.3cf: %Destroy.type = impl_witness_access %Iterate.lookup_impl_witness.06b, element1 [symbolic]
// CHECK:STDOUT:   %as_type.dbb: type = facet_access_type %impl.elem1.3cf [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.6f3: <specific function> = specific_impl_function %impl.elem2.49b, @Iterate.WithSelf.NewCursor(%Iterate.facet.dbf) [symbolic]
// CHECK:STDOUT:   %ptr.bde: type = ptr_type %as_type.dbb [symbolic]
// CHECK:STDOUT:   %.bd3: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.705, %Iterate.facet.dbf [symbolic]
// CHECK:STDOUT:   %impl.elem3.d57: %.bd3 = impl_witness_access %Iterate.lookup_impl_witness.06b, element3 [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.84c: type = class_type @DefaultOptionalStorage, @DefaultOptionalStorage(%facet_value.bf1) [concrete]
// CHECK:STDOUT:   %OptionalStorage.impl_witness.7d0: <witness> = impl_witness imports.%OptionalStorage.impl_witness_table.80d, @T.as_type.as.OptionalStorage.impl(%facet_value.bf1) [concrete]
// CHECK:STDOUT:   %OptionalStorage.facet.ba4: %OptionalStorage.type = facet_value %ValueType, (%OptionalStorage.impl_witness.7d0) [concrete]
// CHECK:STDOUT:   %Optional.aa2: type = class_type @Optional, @Optional(%OptionalStorage.facet.ba4) [concrete]
// CHECK:STDOUT:   %specific_impl_fn.fca: <specific function> = specific_impl_function %impl.elem3.d57, @Iterate.WithSelf.Next(%Iterate.facet.dbf) [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.type.5f9: type = fn_type @Optional.HasValue, @Optional(%OptionalStorage.facet.ba4) [concrete]
// CHECK:STDOUT:   %Optional.HasValue.fa8: %Optional.HasValue.type.5f9 = struct_value () [concrete]
// CHECK:STDOUT:   %Optional.Get.type.74f: type = fn_type @Optional.Get, @Optional(%OptionalStorage.facet.ba4) [concrete]
// CHECK:STDOUT:   %Optional.Get.ec5: %Optional.Get.type.74f = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.facet.4e5: %Destroy.type = facet_value %ValueType, (%custom_witness.e7a31c.1) [concrete]
// CHECK:STDOUT:   %MaybeUnformed.02e: type = class_type @MaybeUnformed, @MaybeUnformed(%Destroy.facet.4e5) [concrete]
// CHECK:STDOUT:   %.98c: type = maybe_unformed_type %ValueType [concrete]
// CHECK:STDOUT:   %struct_type.value.has_value.761: type = struct_type {.value: %MaybeUnformed.02e, .has_value: bool} [concrete]
// CHECK:STDOUT:   %Optional.HasValue.specific_fn.c5b: <specific function> = specific_function %Optional.HasValue.fa8, @Optional.HasValue(%OptionalStorage.facet.ba4) [concrete]
// CHECK:STDOUT:   %Optional.Get.specific_fn.a74: <specific function> = specific_function %Optional.Get.ec5, @Optional.Get(%OptionalStorage.facet.ba4) [concrete]
// CHECK:STDOUT:   %operator_PlusEqual__carbon_thunk.type: type = fn_type @operator_PlusEqual__carbon_thunk [concrete]
// CHECK:STDOUT:   %operator_PlusEqual__carbon_thunk: %operator_PlusEqual__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.7: type = fn_type @Destroy.Op.loc44_29.6 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.7: %Destroy.Op.type.1d8f74.7 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.b29: <witness> = lookup_impl_witness %impl.elem1.3cf, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.d9c: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%impl.elem1.3cf) [symbolic]
// CHECK:STDOUT:   %.98a: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.d9c, %impl.elem1.3cf [symbolic]
// CHECK:STDOUT:   %impl.elem0.81b: %.98a = impl_witness_access %Destroy.lookup_impl_witness.b29, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.bf3: <specific function> = specific_impl_function %impl.elem0.81b, @Destroy.WithSelf.Op(%impl.elem1.3cf) [symbolic]
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
// CHECK:STDOUT:   %custom_witness.444: <witness> = custom_witness (%Iterator.Op.a2a866.3), @Inc [concrete]
// CHECK:STDOUT:   %Equal.type.ca8475.1: type = fn_type @Equal.1 [concrete]
// CHECK:STDOUT:   %Equal.285b15.1: %Equal.type.ca8475.1 = struct_value () [concrete]
// CHECK:STDOUT:   %NotEqual.type.53d9ee.1: type = fn_type @NotEqual.1 [concrete]
// CHECK:STDOUT:   %NotEqual.119263.1: %NotEqual.type.53d9ee.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.3e8: <witness> = custom_witness (%Equal.285b15.1, %NotEqual.119263.1), @EqWith, @EqWith(%Sentinel.9b2) [concrete]
// CHECK:STDOUT:   %Sentinel.Op.type.0ae: type = fn_type @Sentinel.Op.1 [concrete]
// CHECK:STDOUT:   %Sentinel.Op.a2a: %Sentinel.Op.type.0ae = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.95581e.2: <witness> = custom_witness (%Sentinel.Op.a2a), @Destroy [concrete]
// CHECK:STDOUT:   %facet_value.9f5: %CppRange_where.type.d682d6.1 = facet_value %MutableRange, (%custom_witness.e7a31c.1, %custom_witness.d0da43.1, %custom_witness.6b6, %custom_witness.95581e.1, %custom_witness.90a, %custom_witness.444, %custom_witness.3e8, %custom_witness.95581e.2) [concrete]
// CHECK:STDOUT:   %tuple.type.829: type = tuple_type (%Iterator.3ee, %Sentinel.9b2) [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.8: type = fn_type @Destroy.Op.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.8: %Destroy.Op.type.1d8f74.8 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.df9cc1.8: <witness> = custom_witness (%Destroy.Op.1a2547.8), @Destroy [concrete]
// CHECK:STDOUT:   %Destroy.facet.305: %Destroy.type = facet_value %tuple.type.829, (%custom_witness.df9cc1.8) [concrete]
// CHECK:STDOUT:   %Iterate.impl_witness.00a: <witness> = impl_witness imports.%Iterate.impl_witness_table.478, @T.as_type.as.Iterate.impl(%facet_value.9f5) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.type.f22: type = fn_type @T.as_type.as.Iterate.impl.NewCursor, @T.as_type.as.Iterate.impl(%facet_value.9f5) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.a3f: %T.as_type.as.Iterate.impl.NewCursor.type.f22 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.type.69d: type = fn_type @T.as_type.as.Iterate.impl.Next, @T.as_type.as.Iterate.impl(%facet_value.9f5) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.db5: %T.as_type.as.Iterate.impl.Next.type.69d = struct_value () [concrete]
// CHECK:STDOUT:   %Iterate.facet.a01: %Iterate.type = facet_value %MutableRange, (%Iterate.impl_witness.00a) [concrete]
// CHECK:STDOUT:   %facet_value.b35: %Iterate_where.type.502 = facet_value %MutableRange, (%Iterate.impl_witness.00a) [concrete]
// CHECK:STDOUT:   %r.patt.f69: %pattern_type.e635 = ref_binding_pattern r [concrete]
// CHECK:STDOUT:   %r.param_patt.91e: %pattern_type.e635 = var_param_pattern %r.patt.f69 [concrete]
// CHECK:STDOUT:   %Iterator.260: type = class_type @Iterator.2 [concrete]
// CHECK:STDOUT:   %Sentinel.4c4: type = class_type @Sentinel.2 [concrete]
// CHECK:STDOUT:   %ConstRange.Begin.type: type = fn_type @ConstRange.Begin [concrete]
// CHECK:STDOUT:   %ConstRange.Begin: %ConstRange.Begin.type = struct_value () [concrete]
// CHECK:STDOUT:   %ConstRange.End.type: type = fn_type @ConstRange.End [concrete]
// CHECK:STDOUT:   %ConstRange.End: %ConstRange.End.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.a03: <witness> = custom_witness (%Iterator.260, %Sentinel.4c4, %ConstRange.Begin, %ConstRange.End), @CppRangeForIterate [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.0f34b8.1: type = fn_type @Iterator.Op.4 [concrete]
// CHECK:STDOUT:   %Iterator.Op.1c3e4a.1: %Iterator.Op.type.0f34b8.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.78b: <witness> = custom_witness (%ValueType, %Iterator.Op.1c3e4a.1), @CppUnsafeDeref [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.0f34b8.2: type = fn_type @Iterator.Op.5 [concrete]
// CHECK:STDOUT:   %Iterator.Op.1c3e4a.2: %Iterator.Op.type.0f34b8.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.6e8a2e.1: <witness> = custom_witness (%Iterator.Op.1c3e4a.2), @Destroy [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.0f34b8.3: type = fn_type @Iterator.Op.6 [concrete]
// CHECK:STDOUT:   %Iterator.Op.1c3e4a.3: %Iterator.Op.type.0f34b8.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.746: <witness> = custom_witness (%Iterator.Op.1c3e4a.3), @Inc [concrete]
// CHECK:STDOUT:   %Equal.type.ca8475.2: type = fn_type @Equal.2 [concrete]
// CHECK:STDOUT:   %Equal.285b15.2: %Equal.type.ca8475.2 = struct_value () [concrete]
// CHECK:STDOUT:   %NotEqual.type.53d9ee.2: type = fn_type @NotEqual.2 [concrete]
// CHECK:STDOUT:   %NotEqual.119263.2: %NotEqual.type.53d9ee.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.c55: <witness> = custom_witness (%Equal.285b15.2, %NotEqual.119263.2), @EqWith, @EqWith(%Sentinel.4c4) [concrete]
// CHECK:STDOUT:   %Sentinel.Op.type.0f3: type = fn_type @Sentinel.Op.2 [concrete]
// CHECK:STDOUT:   %Sentinel.Op.1c3: %Sentinel.Op.type.0f3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.6e8a2e.2: <witness> = custom_witness (%Sentinel.Op.1c3), @Destroy [concrete]
// CHECK:STDOUT:   %facet_value.782: %CppRange_where.type.d682d6.1 = facet_value %ConstRange, (%custom_witness.e7a31c.1, %custom_witness.d0da43.1, %custom_witness.a03, %custom_witness.6e8a2e.1, %custom_witness.78b, %custom_witness.746, %custom_witness.c55, %custom_witness.6e8a2e.2) [concrete]
// CHECK:STDOUT:   %tuple.type.99f: type = tuple_type (%Iterator.260, %Sentinel.4c4) [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.9: type = fn_type @Destroy.Op.3 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.9: %Destroy.Op.type.1d8f74.9 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.df9cc1.9: <witness> = custom_witness (%Destroy.Op.1a2547.9), @Destroy [concrete]
// CHECK:STDOUT:   %Destroy.facet.e08: %Destroy.type = facet_value %tuple.type.99f, (%custom_witness.df9cc1.9) [concrete]
// CHECK:STDOUT:   %Iterate.impl_witness.8b4: <witness> = impl_witness imports.%Iterate.impl_witness_table.478, @T.as_type.as.Iterate.impl(%facet_value.782) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.type.897: type = fn_type @T.as_type.as.Iterate.impl.NewCursor, @T.as_type.as.Iterate.impl(%facet_value.782) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.4af: %T.as_type.as.Iterate.impl.NewCursor.type.897 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.type.0ac: type = fn_type @T.as_type.as.Iterate.impl.Next, @T.as_type.as.Iterate.impl(%facet_value.782) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.988: %T.as_type.as.Iterate.impl.Next.type.0ac = struct_value () [concrete]
// CHECK:STDOUT:   %Iterate.facet.d09: %Iterate.type = facet_value %ConstRange, (%Iterate.impl_witness.8b4) [concrete]
// CHECK:STDOUT:   %facet_value.fba: %Iterate_where.type.502 = facet_value %ConstRange, (%Iterate.impl_witness.8b4) [concrete]
// CHECK:STDOUT:   %r.patt.36a: %pattern_type.0ba = ref_binding_pattern r [concrete]
// CHECK:STDOUT:   %r.param_patt.501: %pattern_type.0ba = var_param_pattern %r.patt.36a [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.7e6: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.a01) [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.f85: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.a01) [concrete]
// CHECK:STDOUT:   %.0ed: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.7e6, %Iterate.facet.a01 [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.specific_fn.a99: <specific function> = specific_function %T.as_type.as.Iterate.impl.NewCursor.a3f, @T.as_type.as.Iterate.impl.NewCursor(%facet_value.9f5) [concrete]
// CHECK:STDOUT:   %ptr.d6b: type = ptr_type %tuple.type.829 [concrete]
// CHECK:STDOUT:   %.1c5: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.f85, %Iterate.facet.a01 [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.specific_fn.b98: <specific function> = specific_function %T.as_type.as.Iterate.impl.Next.db5, @T.as_type.as.Iterate.impl.Next(%facet_value.9f5) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.c89: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.305) [concrete]
// CHECK:STDOUT:   %.e8a: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.c89, %Destroy.facet.305 [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.a99: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.d09) [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.c30: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.d09) [concrete]
// CHECK:STDOUT:   %.a00: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.a99, %Iterate.facet.d09 [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.specific_fn.e9b: <specific function> = specific_function %T.as_type.as.Iterate.impl.NewCursor.4af, @T.as_type.as.Iterate.impl.NewCursor(%facet_value.782) [concrete]
// CHECK:STDOUT:   %ptr.2f7: type = ptr_type %tuple.type.99f [concrete]
// CHECK:STDOUT:   %.c34: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.c30, %Iterate.facet.d09 [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.specific_fn.4f8: <specific function> = specific_function %T.as_type.as.Iterate.impl.Next.988, @T.as_type.as.Iterate.impl.Next(%facet_value.782) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.e30: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.e08) [concrete]
// CHECK:STDOUT:   %.987: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.e30, %Destroy.facet.e08 [concrete]
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
// CHECK:STDOUT:   %Core.import_ref.13b: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.None.type (%T.as_type.as.OptionalStorage.impl.None.type.d4d) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.None (constants.%T.as_type.as.OptionalStorage.impl.None.d64)]
// CHECK:STDOUT:   %Core.import_ref.c43f: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Some.type (%T.as_type.as.OptionalStorage.impl.Some.type.7eb) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Some (constants.%T.as_type.as.OptionalStorage.impl.Some.2b4)]
// CHECK:STDOUT:   %Core.import_ref.61e: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Has.type (%T.as_type.as.OptionalStorage.impl.Has.type.eb9) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Has (constants.%T.as_type.as.OptionalStorage.impl.Has.c7e)]
// CHECK:STDOUT:   %Core.import_ref.ce2e: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Get.type (%T.as_type.as.OptionalStorage.impl.Get.type.d3c) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Get (constants.%T.as_type.as.OptionalStorage.impl.Get.da2)]
// CHECK:STDOUT:   %Core.import_ref.8fc = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], unloaded
// CHECK:STDOUT:   %OptionalStorage.impl_witness_table.80d = impl_witness_table (%Core.import_ref.0c6, %Core.import_ref.13b, %Core.import_ref.c43f, %Core.import_ref.61e, %Core.import_ref.ce2e, %Core.import_ref.8fc), @T.as_type.as.OptionalStorage.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.eef: %facet_type.f48 = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.Iterate.impl.%facet_value (constants.%facet_value.f93)]
// CHECK:STDOUT:   %Core.import_ref.fcc: %Destroy.type = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.Iterate.impl.%Destroy.facet (constants.%Destroy.facet.9cd)]
// CHECK:STDOUT:   %Core.import_ref.f50: @T.as_type.as.Iterate.impl.%T.as_type.as.Iterate.impl.NewCursor.type (%T.as_type.as.Iterate.impl.NewCursor.type.b5d) = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.Iterate.impl.%T.as_type.as.Iterate.impl.NewCursor (constants.%T.as_type.as.Iterate.impl.NewCursor.21f)]
// CHECK:STDOUT:   %Core.import_ref.590: @T.as_type.as.Iterate.impl.%T.as_type.as.Iterate.impl.Next.type (%T.as_type.as.Iterate.impl.Next.type.5dc) = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.Iterate.impl.%T.as_type.as.Iterate.impl.Next (constants.%T.as_type.as.Iterate.impl.Next.b73)]
// CHECK:STDOUT:   %Iterate.impl_witness_table.478 = impl_witness_table (%Core.import_ref.eef, %Core.import_ref.fcc, %Core.import_ref.f50, %Core.import_ref.590), @T.as_type.as.Iterate.impl [concrete]
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
// CHECK:STDOUT:   %ConstRange.decl: type = class_decl @ConstRange [concrete = constants.%ConstRange] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @TestIterate(%R.loc40_17.2: %Iterate_where.type.502) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Iterate.lookup_impl_witness: <witness> = lookup_impl_witness %R.loc40_17.1, @Iterate [symbolic = %Iterate.lookup_impl_witness (constants.%Iterate.lookup_impl_witness.06b)]
// CHECK:STDOUT:   %Iterate.facet.loc44_29.5: %Iterate.type = facet_value %R.as_type.loc40_75.1, (%Iterate.lookup_impl_witness) [symbolic = %Iterate.facet.loc44_29.5 (constants.%Iterate.facet.dbf)]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.loc44_29.5) [symbolic = %Iterate.WithSelf.NewCursor.type (constants.%Iterate.WithSelf.NewCursor.type.1b1)]
// CHECK:STDOUT:   %.loc44_29.20: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type, %Iterate.facet.loc44_29.5 [symbolic = %.loc44_29.20 (constants.%.1c2)]
// CHECK:STDOUT:   %impl.elem2.loc44_29.2: @TestIterate.%.loc44_29.20 (%.1c2) = impl_witness_access %Iterate.lookup_impl_witness, element2 [symbolic = %impl.elem2.loc44_29.2 (constants.%impl.elem2.49b)]
// CHECK:STDOUT:   %specific_impl_fn.loc44_29.4: <specific function> = specific_impl_function %impl.elem2.loc44_29.2, @Iterate.WithSelf.NewCursor(%Iterate.facet.loc44_29.5) [symbolic = %specific_impl_fn.loc44_29.4 (constants.%specific_impl_fn.6f3)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %ptr: type = ptr_type %as_type [symbolic = %ptr (constants.%ptr.bde)]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.loc44_29.5) [symbolic = %Iterate.WithSelf.Next.type (constants.%Iterate.WithSelf.Next.type.705)]
// CHECK:STDOUT:   %.loc44_29.21: type = fn_type_with_self_type %Iterate.WithSelf.Next.type, %Iterate.facet.loc44_29.5 [symbolic = %.loc44_29.21 (constants.%.bd3)]
// CHECK:STDOUT:   %impl.elem3.loc44_29.2: @TestIterate.%.loc44_29.21 (%.bd3) = impl_witness_access %Iterate.lookup_impl_witness, element3 [symbolic = %impl.elem3.loc44_29.2 (constants.%impl.elem3.d57)]
// CHECK:STDOUT:   %specific_impl_fn.loc44_29.5: <specific function> = specific_impl_function %impl.elem3.loc44_29.2, @Iterate.WithSelf.Next(%Iterate.facet.loc44_29.5) [symbolic = %specific_impl_fn.loc44_29.5 (constants.%specific_impl_fn.fca)]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%impl.elem1) [symbolic = %Destroy.WithSelf.Op.type (constants.%Destroy.WithSelf.Op.type.d9c)]
// CHECK:STDOUT:   %.loc44_29.22: type = fn_type_with_self_type %Destroy.WithSelf.Op.type, %impl.elem1 [symbolic = %.loc44_29.22 (constants.%.98a)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %impl.elem1, @Destroy [symbolic = %Destroy.lookup_impl_witness (constants.%Destroy.lookup_impl_witness.b29)]
// CHECK:STDOUT:   %impl.elem0.loc44_29.2: @TestIterate.%.loc44_29.22 (%.98a) = impl_witness_access %Destroy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc44_29.2 (constants.%impl.elem0.81b)]
// CHECK:STDOUT:   %specific_impl_fn.loc44_29.6: <specific function> = specific_impl_function %impl.elem0.loc44_29.2, @Destroy.WithSelf.Op(%impl.elem1) [symbolic = %specific_impl_fn.loc44_29.6 (constants.%specific_impl_fn.bf3)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%r.param: ref @TestIterate.%R.as_type.loc40_75.1 (%R.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %i.patt: %pattern_type.e9f = value_binding_pattern i [concrete = constants.%i.patt]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %r.ref: ref @TestIterate.%R.as_type.loc40_75.1 (%R.as_type) = name_ref r, %r
// CHECK:STDOUT:     %impl.elem2.loc44_29.1: @TestIterate.%.loc44_29.20 (%.1c2) = impl_witness_access constants.%Iterate.lookup_impl_witness.06b, element2 [symbolic = %impl.elem2.loc44_29.2 (constants.%impl.elem2.49b)]
// CHECK:STDOUT:     %bound_method.loc44_29.1: <bound method> = bound_method %r.ref, %impl.elem2.loc44_29.1
// CHECK:STDOUT:     %Iterate.facet.loc44_29.1: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.06b) [symbolic = %Iterate.facet.loc44_29.5 (constants.%Iterate.facet.dbf)]
// CHECK:STDOUT:     %.loc44_29.1: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc44_29.1 [symbolic = %Iterate.facet.loc44_29.5 (constants.%Iterate.facet.dbf)]
// CHECK:STDOUT:     %Iterate.facet.loc44_29.2: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.06b) [symbolic = %Iterate.facet.loc44_29.5 (constants.%Iterate.facet.dbf)]
// CHECK:STDOUT:     %.loc44_29.2: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc44_29.2 [symbolic = %Iterate.facet.loc44_29.5 (constants.%Iterate.facet.dbf)]
// CHECK:STDOUT:     %specific_impl_fn.loc44_29.1: <specific function> = specific_impl_function %impl.elem2.loc44_29.1, @Iterate.WithSelf.NewCursor(constants.%Iterate.facet.dbf) [symbolic = %specific_impl_fn.loc44_29.4 (constants.%specific_impl_fn.6f3)]
// CHECK:STDOUT:     %bound_method.loc44_29.2: <bound method> = bound_method %r.ref, %specific_impl_fn.loc44_29.1
// CHECK:STDOUT:     %var: ref @TestIterate.%as_type (%as_type.dbb) = var_storage invalid
// CHECK:STDOUT:     %.loc44_28.1: @TestIterate.%R.as_type.loc40_75.1 (%R.as_type) = acquire_value %r.ref
// CHECK:STDOUT:     %Iterate.WithSelf.NewCursor.call: init @TestIterate.%as_type (%as_type.dbb) to %var = call %bound_method.loc44_29.2(%.loc44_28.1)
// CHECK:STDOUT:     assign %var, %Iterate.WithSelf.NewCursor.call
// CHECK:STDOUT:     br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.next:
// CHECK:STDOUT:     %addr.loc44: @TestIterate.%ptr (%ptr.bde) = addr_of %var
// CHECK:STDOUT:     %impl.elem3.loc44_29.1: @TestIterate.%.loc44_29.21 (%.bd3) = impl_witness_access constants.%Iterate.lookup_impl_witness.06b, element3 [symbolic = %impl.elem3.loc44_29.2 (constants.%impl.elem3.d57)]
// CHECK:STDOUT:     %bound_method.loc44_29.3: <bound method> = bound_method %r.ref, %impl.elem3.loc44_29.1
// CHECK:STDOUT:     %Iterate.facet.loc44_29.3: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.06b) [symbolic = %Iterate.facet.loc44_29.5 (constants.%Iterate.facet.dbf)]
// CHECK:STDOUT:     %.loc44_29.3: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc44_29.3 [symbolic = %Iterate.facet.loc44_29.5 (constants.%Iterate.facet.dbf)]
// CHECK:STDOUT:     %Iterate.facet.loc44_29.4: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.06b) [symbolic = %Iterate.facet.loc44_29.5 (constants.%Iterate.facet.dbf)]
// CHECK:STDOUT:     %.loc44_29.4: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc44_29.4 [symbolic = %Iterate.facet.loc44_29.5 (constants.%Iterate.facet.dbf)]
// CHECK:STDOUT:     %.loc44_29.5: %Destroy.type = converted constants.%as_type.dbb, constants.%impl.elem1.3cf [symbolic = %impl.elem1 (constants.%impl.elem1.3cf)]
// CHECK:STDOUT:     %.loc44_29.6: %Destroy.type = converted constants.%as_type.dbb, constants.%impl.elem1.3cf [symbolic = %impl.elem1 (constants.%impl.elem1.3cf)]
// CHECK:STDOUT:     %specific_impl_fn.loc44_29.2: <specific function> = specific_impl_function %impl.elem3.loc44_29.1, @Iterate.WithSelf.Next(constants.%Iterate.facet.dbf) [symbolic = %specific_impl_fn.loc44_29.5 (constants.%specific_impl_fn.fca)]
// CHECK:STDOUT:     %bound_method.loc44_29.4: <bound method> = bound_method %r.ref, %specific_impl_fn.loc44_29.2
// CHECK:STDOUT:     %.loc44_29.7: ref %Optional.aa2 = temporary_storage
// CHECK:STDOUT:     %.loc44_28.2: @TestIterate.%R.as_type.loc40_75.1 (%R.as_type) = acquire_value %r.ref
// CHECK:STDOUT:     %Iterate.WithSelf.Next.call: init %Optional.aa2 to %.loc44_29.7 = call %bound_method.loc44_29.4(%.loc44_28.2, %addr.loc44)
// CHECK:STDOUT:     %.loc44_29.8: ref %Optional.aa2 = temporary %.loc44_29.7, %Iterate.WithSelf.Next.call
// CHECK:STDOUT:     %.loc44_29.9: %Optional.HasValue.type.5f9 = specific_constant imports.%Core.import_ref.bb6, @Optional(constants.%OptionalStorage.facet.ba4) [concrete = constants.%Optional.HasValue.fa8]
// CHECK:STDOUT:     %HasValue.ref: %Optional.HasValue.type.5f9 = name_ref HasValue, %.loc44_29.9 [concrete = constants.%Optional.HasValue.fa8]
// CHECK:STDOUT:     %Optional.HasValue.bound: <bound method> = bound_method %.loc44_29.8, %HasValue.ref
// CHECK:STDOUT:     %Optional.HasValue.specific_fn: <specific function> = specific_function %HasValue.ref, @Optional.HasValue(constants.%OptionalStorage.facet.ba4) [concrete = constants.%Optional.HasValue.specific_fn.c5b]
// CHECK:STDOUT:     %bound_method.loc44_29.5: <bound method> = bound_method %.loc44_29.8, %Optional.HasValue.specific_fn
// CHECK:STDOUT:     %.loc44_29.10: %Optional.aa2 = acquire_value %.loc44_29.8
// CHECK:STDOUT:     %Optional.HasValue.call: init bool = call %bound_method.loc44_29.5(%.loc44_29.10)
// CHECK:STDOUT:     %.loc44_29.11: bool = value_of_initializer %Optional.HasValue.call
// CHECK:STDOUT:     %.loc44_29.12: bool = converted %Optional.HasValue.call, %.loc44_29.11
// CHECK:STDOUT:     if %.loc44_29.12 br !for.body else br !for.done
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.body:
// CHECK:STDOUT:     %.loc44_29.13: %Optional.Get.type.74f = specific_constant imports.%Core.import_ref.daba, @Optional(constants.%OptionalStorage.facet.ba4) [concrete = constants.%Optional.Get.ec5]
// CHECK:STDOUT:     %Get.ref: %Optional.Get.type.74f = name_ref Get, %.loc44_29.13 [concrete = constants.%Optional.Get.ec5]
// CHECK:STDOUT:     %Optional.Get.bound: <bound method> = bound_method %.loc44_29.8, %Get.ref
// CHECK:STDOUT:     %Optional.Get.specific_fn: <specific function> = specific_function %Get.ref, @Optional.Get(constants.%OptionalStorage.facet.ba4) [concrete = constants.%Optional.Get.specific_fn.a74]
// CHECK:STDOUT:     %bound_method.loc44_29.6: <bound method> = bound_method %.loc44_29.8, %Optional.Get.specific_fn
// CHECK:STDOUT:     %.loc44_29.14: ref %ValueType = temporary_storage
// CHECK:STDOUT:     %.loc44_29.15: %Optional.aa2 = acquire_value %.loc44_29.8
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
// CHECK:STDOUT:     %ValueType.Op.bound.loc44: <bound method> = bound_method %.loc44_29.16, constants.%ValueType.Op.4dca31.1
// CHECK:STDOUT:     %Op.ref.loc44: %ValueType.cpp_destructor.type = name_ref Op, imports.%ValueType.cpp_destructor.decl [concrete = constants.%ValueType.cpp_destructor]
// CHECK:STDOUT:     %ValueType.cpp_destructor.bound.loc44: <bound method> = bound_method %.loc44_29.16, %Op.ref.loc44
// CHECK:STDOUT:     %ValueType.cpp_destructor.call.loc44: init %empty_tuple.type = call %ValueType.cpp_destructor.bound.loc44(%.loc44_29.16)
// CHECK:STDOUT:     %Destroy.Op.bound.loc44_29.1: <bound method> = bound_method %.loc44_29.8, constants.%Destroy.Op.1a2547.7
// CHECK:STDOUT:     %Destroy.Op.call.loc44_29.1: init %empty_tuple.type = call %Destroy.Op.bound.loc44_29.1(%.loc44_29.8)
// CHECK:STDOUT:     br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.done:
// CHECK:STDOUT:     %Destroy.Op.bound.loc44_29.2: <bound method> = bound_method %.loc44_29.8, constants.%Destroy.Op.1a2547.7
// CHECK:STDOUT:     %Destroy.Op.call.loc44_29.2: init %empty_tuple.type = call %Destroy.Op.bound.loc44_29.2(%.loc44_29.8)
// CHECK:STDOUT:     %impl.elem0.loc44_29.1: @TestIterate.%.loc44_29.22 (%.98a) = impl_witness_access constants.%Destroy.lookup_impl_witness.b29, element0 [symbolic = %impl.elem0.loc44_29.2 (constants.%impl.elem0.81b)]
// CHECK:STDOUT:     %bound_method.loc44_29.7: <bound method> = bound_method %var, %impl.elem0.loc44_29.1
// CHECK:STDOUT:     %.loc44_29.18: %Destroy.type = converted constants.%as_type.dbb, constants.%impl.elem1.3cf [symbolic = %impl.elem1 (constants.%impl.elem1.3cf)]
// CHECK:STDOUT:     %.loc44_29.19: %Destroy.type = converted constants.%as_type.dbb, constants.%impl.elem1.3cf [symbolic = %impl.elem1 (constants.%impl.elem1.3cf)]
// CHECK:STDOUT:     %specific_impl_fn.loc44_29.3: <specific function> = specific_impl_function %impl.elem0.loc44_29.1, @Destroy.WithSelf.Op(constants.%impl.elem1.3cf) [symbolic = %specific_impl_fn.loc44_29.6 (constants.%specific_impl_fn.bf3)]
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
// CHECK:STDOUT: fn @Destroy.Op.loc44_29.6(%self.param: ref %Optional.aa2) {
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
// CHECK:STDOUT:   %impl.elem2.loc59: %.0ed = impl_witness_access constants.%Iterate.impl_witness.00a, element2 [concrete = constants.%T.as_type.as.Iterate.impl.NewCursor.a3f]
// CHECK:STDOUT:   %bound_method.loc59_29.1: <bound method> = bound_method %m.ref, %impl.elem2.loc59
// CHECK:STDOUT:   %facet_value.loc59_29.1: %CppRange_where.type.d682d6.1 = facet_value constants.%MutableRange, (constants.%custom_witness.e7a31c.1, constants.%custom_witness.d0da43.1, constants.%custom_witness.6b6, constants.%custom_witness.95581e.1, constants.%custom_witness.90a, constants.%custom_witness.444, constants.%custom_witness.3e8, constants.%custom_witness.95581e.2) [concrete = constants.%facet_value.9f5]
// CHECK:STDOUT:   %.loc59_29.1: %CppRange_where.type.d682d6.1 = converted constants.%MutableRange, %facet_value.loc59_29.1 [concrete = constants.%facet_value.9f5]
// CHECK:STDOUT:   %facet_value.loc59_29.2: %CppRange_where.type.d682d6.1 = facet_value constants.%MutableRange, (constants.%custom_witness.e7a31c.1, constants.%custom_witness.d0da43.1, constants.%custom_witness.6b6, constants.%custom_witness.95581e.1, constants.%custom_witness.90a, constants.%custom_witness.444, constants.%custom_witness.3e8, constants.%custom_witness.95581e.2) [concrete = constants.%facet_value.9f5]
// CHECK:STDOUT:   %.loc59_29.2: %CppRange_where.type.d682d6.1 = converted constants.%MutableRange, %facet_value.loc59_29.2 [concrete = constants.%facet_value.9f5]
// CHECK:STDOUT:   %specific_fn.loc59_29.1: <specific function> = specific_function %impl.elem2.loc59, @T.as_type.as.Iterate.impl.NewCursor(constants.%facet_value.9f5) [concrete = constants.%T.as_type.as.Iterate.impl.NewCursor.specific_fn.a99]
// CHECK:STDOUT:   %bound_method.loc59_29.2: <bound method> = bound_method %m.ref, %specific_fn.loc59_29.1
// CHECK:STDOUT:   %var.loc59: ref %tuple.type.829 = var_storage invalid
// CHECK:STDOUT:   %.loc59_28.1: %MutableRange = acquire_value %m.ref
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.call.loc59: init %tuple.type.829 to %var.loc59 = call %bound_method.loc59_29.2(%.loc59_28.1)
// CHECK:STDOUT:   assign %var.loc59, %T.as_type.as.Iterate.impl.NewCursor.call.loc59
// CHECK:STDOUT:   br !for.next.loc59
// CHECK:STDOUT:
// CHECK:STDOUT: !for.next.loc59:
// CHECK:STDOUT:   %addr.loc59: %ptr.d6b = addr_of %var.loc59
// CHECK:STDOUT:   %impl.elem3.loc59: %.1c5 = impl_witness_access constants.%Iterate.impl_witness.00a, element3 [concrete = constants.%T.as_type.as.Iterate.impl.Next.db5]
// CHECK:STDOUT:   %bound_method.loc59_29.3: <bound method> = bound_method %m.ref, %impl.elem3.loc59
// CHECK:STDOUT:   %facet_value.loc59_29.3: %CppRange_where.type.d682d6.1 = facet_value constants.%MutableRange, (constants.%custom_witness.e7a31c.1, constants.%custom_witness.d0da43.1, constants.%custom_witness.6b6, constants.%custom_witness.95581e.1, constants.%custom_witness.90a, constants.%custom_witness.444, constants.%custom_witness.3e8, constants.%custom_witness.95581e.2) [concrete = constants.%facet_value.9f5]
// CHECK:STDOUT:   %.loc59_29.3: %CppRange_where.type.d682d6.1 = converted constants.%MutableRange, %facet_value.loc59_29.3 [concrete = constants.%facet_value.9f5]
// CHECK:STDOUT:   %facet_value.loc59_29.4: %CppRange_where.type.d682d6.1 = facet_value constants.%MutableRange, (constants.%custom_witness.e7a31c.1, constants.%custom_witness.d0da43.1, constants.%custom_witness.6b6, constants.%custom_witness.95581e.1, constants.%custom_witness.90a, constants.%custom_witness.444, constants.%custom_witness.3e8, constants.%custom_witness.95581e.2) [concrete = constants.%facet_value.9f5]
// CHECK:STDOUT:   %.loc59_29.4: %CppRange_where.type.d682d6.1 = converted constants.%MutableRange, %facet_value.loc59_29.4 [concrete = constants.%facet_value.9f5]
// CHECK:STDOUT:   %specific_fn.loc59_29.2: <specific function> = specific_function %impl.elem3.loc59, @T.as_type.as.Iterate.impl.Next(constants.%facet_value.9f5) [concrete = constants.%T.as_type.as.Iterate.impl.Next.specific_fn.b98]
// CHECK:STDOUT:   %bound_method.loc59_29.4: <bound method> = bound_method %m.ref, %specific_fn.loc59_29.2
// CHECK:STDOUT:   %.loc59_29.5: ref %Optional.aa2 = temporary_storage
// CHECK:STDOUT:   %.loc59_28.2: %MutableRange = acquire_value %m.ref
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.call.loc59: init %Optional.aa2 to %.loc59_29.5 = call %bound_method.loc59_29.4(%.loc59_28.2, %addr.loc59)
// CHECK:STDOUT:   %.loc59_29.6: ref %Optional.aa2 = temporary %.loc59_29.5, %T.as_type.as.Iterate.impl.Next.call.loc59
// CHECK:STDOUT:   %.loc59_29.7: %Optional.HasValue.type.5f9 = specific_constant imports.%Core.import_ref.bb6, @Optional(constants.%OptionalStorage.facet.ba4) [concrete = constants.%Optional.HasValue.fa8]
// CHECK:STDOUT:   %HasValue.ref.loc59: %Optional.HasValue.type.5f9 = name_ref HasValue, %.loc59_29.7 [concrete = constants.%Optional.HasValue.fa8]
// CHECK:STDOUT:   %Optional.HasValue.bound.loc59: <bound method> = bound_method %.loc59_29.6, %HasValue.ref.loc59
// CHECK:STDOUT:   %Optional.HasValue.specific_fn.loc59: <specific function> = specific_function %HasValue.ref.loc59, @Optional.HasValue(constants.%OptionalStorage.facet.ba4) [concrete = constants.%Optional.HasValue.specific_fn.c5b]
// CHECK:STDOUT:   %bound_method.loc59_29.5: <bound method> = bound_method %.loc59_29.6, %Optional.HasValue.specific_fn.loc59
// CHECK:STDOUT:   %.loc59_29.8: %Optional.aa2 = acquire_value %.loc59_29.6
// CHECK:STDOUT:   %Optional.HasValue.call.loc59: init bool = call %bound_method.loc59_29.5(%.loc59_29.8)
// CHECK:STDOUT:   %.loc59_29.9: bool = value_of_initializer %Optional.HasValue.call.loc59
// CHECK:STDOUT:   %.loc59_29.10: bool = converted %Optional.HasValue.call.loc59, %.loc59_29.9
// CHECK:STDOUT:   if %.loc59_29.10 br !for.body.loc59 else br !for.done.loc59
// CHECK:STDOUT:
// CHECK:STDOUT: !for.body.loc59:
// CHECK:STDOUT:   %.loc59_29.11: %Optional.Get.type.74f = specific_constant imports.%Core.import_ref.daba, @Optional(constants.%OptionalStorage.facet.ba4) [concrete = constants.%Optional.Get.ec5]
// CHECK:STDOUT:   %Get.ref.loc59: %Optional.Get.type.74f = name_ref Get, %.loc59_29.11 [concrete = constants.%Optional.Get.ec5]
// CHECK:STDOUT:   %Optional.Get.bound.loc59: <bound method> = bound_method %.loc59_29.6, %Get.ref.loc59
// CHECK:STDOUT:   %Optional.Get.specific_fn.loc59: <specific function> = specific_function %Get.ref.loc59, @Optional.Get(constants.%OptionalStorage.facet.ba4) [concrete = constants.%Optional.Get.specific_fn.a74]
// CHECK:STDOUT:   %bound_method.loc59_29.6: <bound method> = bound_method %.loc59_29.6, %Optional.Get.specific_fn.loc59
// CHECK:STDOUT:   %.loc59_29.12: ref %ValueType = temporary_storage
// CHECK:STDOUT:   %.loc59_29.13: %Optional.aa2 = acquire_value %.loc59_29.6
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
// CHECK:STDOUT:   %ValueType.Op.bound.loc59: <bound method> = bound_method %.loc59_29.14, constants.%ValueType.Op.4dca31.1
// CHECK:STDOUT:   %Op.ref.loc59: %ValueType.cpp_destructor.type = name_ref Op, imports.%ValueType.cpp_destructor.decl [concrete = constants.%ValueType.cpp_destructor]
// CHECK:STDOUT:   %ValueType.cpp_destructor.bound.loc59: <bound method> = bound_method %.loc59_29.14, %Op.ref.loc59
// CHECK:STDOUT:   %ValueType.cpp_destructor.call.loc59: init %empty_tuple.type = call %ValueType.cpp_destructor.bound.loc59(%.loc59_29.14)
// CHECK:STDOUT:   %Destroy.Op.bound.loc59_29.1: <bound method> = bound_method %.loc59_29.6, constants.%Destroy.Op.1a2547.7
// CHECK:STDOUT:   %Destroy.Op.call.loc59_29.1: init %empty_tuple.type = call %Destroy.Op.bound.loc59_29.1(%.loc59_29.6)
// CHECK:STDOUT:   br !for.next.loc59
// CHECK:STDOUT:
// CHECK:STDOUT: !for.done.loc59:
// CHECK:STDOUT:   %Destroy.Op.bound.loc59_29.2: <bound method> = bound_method %.loc59_29.6, constants.%Destroy.Op.1a2547.7
// CHECK:STDOUT:   %Destroy.Op.call.loc59_29.2: init %empty_tuple.type = call %Destroy.Op.bound.loc59_29.2(%.loc59_29.6)
// CHECK:STDOUT:   %Destroy.Op.bound.loc59_29.3: <bound method> = bound_method %var.loc59, constants.%Destroy.Op.1a2547.8
// CHECK:STDOUT:   %Destroy.Op.call.loc59_29.3: init %empty_tuple.type = call %Destroy.Op.bound.loc59_29.3(%var.loc59)
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %i.patt.loc63: %pattern_type.e9f = value_binding_pattern i [concrete = constants.%i.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c.ref: %ConstRange = name_ref c, %c
// CHECK:STDOUT:   %impl.elem2.loc63: %.a00 = impl_witness_access constants.%Iterate.impl_witness.8b4, element2 [concrete = constants.%T.as_type.as.Iterate.impl.NewCursor.4af]
// CHECK:STDOUT:   %bound_method.loc63_29.1: <bound method> = bound_method %c.ref, %impl.elem2.loc63
// CHECK:STDOUT:   %facet_value.loc63_29.1: %CppRange_where.type.d682d6.1 = facet_value constants.%ConstRange, (constants.%custom_witness.e7a31c.1, constants.%custom_witness.d0da43.1, constants.%custom_witness.a03, constants.%custom_witness.6e8a2e.1, constants.%custom_witness.78b, constants.%custom_witness.746, constants.%custom_witness.c55, constants.%custom_witness.6e8a2e.2) [concrete = constants.%facet_value.782]
// CHECK:STDOUT:   %.loc63_29.1: %CppRange_where.type.d682d6.1 = converted constants.%ConstRange, %facet_value.loc63_29.1 [concrete = constants.%facet_value.782]
// CHECK:STDOUT:   %facet_value.loc63_29.2: %CppRange_where.type.d682d6.1 = facet_value constants.%ConstRange, (constants.%custom_witness.e7a31c.1, constants.%custom_witness.d0da43.1, constants.%custom_witness.a03, constants.%custom_witness.6e8a2e.1, constants.%custom_witness.78b, constants.%custom_witness.746, constants.%custom_witness.c55, constants.%custom_witness.6e8a2e.2) [concrete = constants.%facet_value.782]
// CHECK:STDOUT:   %.loc63_29.2: %CppRange_where.type.d682d6.1 = converted constants.%ConstRange, %facet_value.loc63_29.2 [concrete = constants.%facet_value.782]
// CHECK:STDOUT:   %specific_fn.loc63_29.1: <specific function> = specific_function %impl.elem2.loc63, @T.as_type.as.Iterate.impl.NewCursor(constants.%facet_value.782) [concrete = constants.%T.as_type.as.Iterate.impl.NewCursor.specific_fn.e9b]
// CHECK:STDOUT:   %bound_method.loc63_29.2: <bound method> = bound_method %c.ref, %specific_fn.loc63_29.1
// CHECK:STDOUT:   %var.loc63: ref %tuple.type.99f = var_storage invalid
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.call.loc63: init %tuple.type.99f to %var.loc63 = call %bound_method.loc63_29.2(%c.ref)
// CHECK:STDOUT:   assign %var.loc63, %T.as_type.as.Iterate.impl.NewCursor.call.loc63
// CHECK:STDOUT:   br !for.next.loc63
// CHECK:STDOUT:
// CHECK:STDOUT: !for.next.loc63:
// CHECK:STDOUT:   %addr.loc63: %ptr.2f7 = addr_of %var.loc63
// CHECK:STDOUT:   %impl.elem3.loc63: %.c34 = impl_witness_access constants.%Iterate.impl_witness.8b4, element3 [concrete = constants.%T.as_type.as.Iterate.impl.Next.988]
// CHECK:STDOUT:   %bound_method.loc63_29.3: <bound method> = bound_method %c.ref, %impl.elem3.loc63
// CHECK:STDOUT:   %facet_value.loc63_29.3: %CppRange_where.type.d682d6.1 = facet_value constants.%ConstRange, (constants.%custom_witness.e7a31c.1, constants.%custom_witness.d0da43.1, constants.%custom_witness.a03, constants.%custom_witness.6e8a2e.1, constants.%custom_witness.78b, constants.%custom_witness.746, constants.%custom_witness.c55, constants.%custom_witness.6e8a2e.2) [concrete = constants.%facet_value.782]
// CHECK:STDOUT:   %.loc63_29.3: %CppRange_where.type.d682d6.1 = converted constants.%ConstRange, %facet_value.loc63_29.3 [concrete = constants.%facet_value.782]
// CHECK:STDOUT:   %facet_value.loc63_29.4: %CppRange_where.type.d682d6.1 = facet_value constants.%ConstRange, (constants.%custom_witness.e7a31c.1, constants.%custom_witness.d0da43.1, constants.%custom_witness.a03, constants.%custom_witness.6e8a2e.1, constants.%custom_witness.78b, constants.%custom_witness.746, constants.%custom_witness.c55, constants.%custom_witness.6e8a2e.2) [concrete = constants.%facet_value.782]
// CHECK:STDOUT:   %.loc63_29.4: %CppRange_where.type.d682d6.1 = converted constants.%ConstRange, %facet_value.loc63_29.4 [concrete = constants.%facet_value.782]
// CHECK:STDOUT:   %specific_fn.loc63_29.2: <specific function> = specific_function %impl.elem3.loc63, @T.as_type.as.Iterate.impl.Next(constants.%facet_value.782) [concrete = constants.%T.as_type.as.Iterate.impl.Next.specific_fn.4f8]
// CHECK:STDOUT:   %bound_method.loc63_29.4: <bound method> = bound_method %c.ref, %specific_fn.loc63_29.2
// CHECK:STDOUT:   %.loc63_29.5: ref %Optional.aa2 = temporary_storage
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.call.loc63: init %Optional.aa2 to %.loc63_29.5 = call %bound_method.loc63_29.4(%c.ref, %addr.loc63)
// CHECK:STDOUT:   %.loc63_29.6: ref %Optional.aa2 = temporary %.loc63_29.5, %T.as_type.as.Iterate.impl.Next.call.loc63
// CHECK:STDOUT:   %.loc63_29.7: %Optional.HasValue.type.5f9 = specific_constant imports.%Core.import_ref.bb6, @Optional(constants.%OptionalStorage.facet.ba4) [concrete = constants.%Optional.HasValue.fa8]
// CHECK:STDOUT:   %HasValue.ref.loc63: %Optional.HasValue.type.5f9 = name_ref HasValue, %.loc63_29.7 [concrete = constants.%Optional.HasValue.fa8]
// CHECK:STDOUT:   %Optional.HasValue.bound.loc63: <bound method> = bound_method %.loc63_29.6, %HasValue.ref.loc63
// CHECK:STDOUT:   %Optional.HasValue.specific_fn.loc63: <specific function> = specific_function %HasValue.ref.loc63, @Optional.HasValue(constants.%OptionalStorage.facet.ba4) [concrete = constants.%Optional.HasValue.specific_fn.c5b]
// CHECK:STDOUT:   %bound_method.loc63_29.5: <bound method> = bound_method %.loc63_29.6, %Optional.HasValue.specific_fn.loc63
// CHECK:STDOUT:   %.loc63_29.8: %Optional.aa2 = acquire_value %.loc63_29.6
// CHECK:STDOUT:   %Optional.HasValue.call.loc63: init bool = call %bound_method.loc63_29.5(%.loc63_29.8)
// CHECK:STDOUT:   %.loc63_29.9: bool = value_of_initializer %Optional.HasValue.call.loc63
// CHECK:STDOUT:   %.loc63_29.10: bool = converted %Optional.HasValue.call.loc63, %.loc63_29.9
// CHECK:STDOUT:   if %.loc63_29.10 br !for.body.loc63 else br !for.done.loc63
// CHECK:STDOUT:
// CHECK:STDOUT: !for.body.loc63:
// CHECK:STDOUT:   %.loc63_29.11: %Optional.Get.type.74f = specific_constant imports.%Core.import_ref.daba, @Optional(constants.%OptionalStorage.facet.ba4) [concrete = constants.%Optional.Get.ec5]
// CHECK:STDOUT:   %Get.ref.loc63: %Optional.Get.type.74f = name_ref Get, %.loc63_29.11 [concrete = constants.%Optional.Get.ec5]
// CHECK:STDOUT:   %Optional.Get.bound.loc63: <bound method> = bound_method %.loc63_29.6, %Get.ref.loc63
// CHECK:STDOUT:   %Optional.Get.specific_fn.loc63: <specific function> = specific_function %Get.ref.loc63, @Optional.Get(constants.%OptionalStorage.facet.ba4) [concrete = constants.%Optional.Get.specific_fn.a74]
// CHECK:STDOUT:   %bound_method.loc63_29.6: <bound method> = bound_method %.loc63_29.6, %Optional.Get.specific_fn.loc63
// CHECK:STDOUT:   %.loc63_29.12: ref %ValueType = temporary_storage
// CHECK:STDOUT:   %.loc63_29.13: %Optional.aa2 = acquire_value %.loc63_29.6
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
// CHECK:STDOUT:   %ValueType.Op.bound.loc63: <bound method> = bound_method %.loc63_29.14, constants.%ValueType.Op.4dca31.1
// CHECK:STDOUT:   %Op.ref.loc63: %ValueType.cpp_destructor.type = name_ref Op, imports.%ValueType.cpp_destructor.decl [concrete = constants.%ValueType.cpp_destructor]
// CHECK:STDOUT:   %ValueType.cpp_destructor.bound.loc63: <bound method> = bound_method %.loc63_29.14, %Op.ref.loc63
// CHECK:STDOUT:   %ValueType.cpp_destructor.call.loc63: init %empty_tuple.type = call %ValueType.cpp_destructor.bound.loc63(%.loc63_29.14)
// CHECK:STDOUT:   %Destroy.Op.bound.loc63_29.1: <bound method> = bound_method %.loc63_29.6, constants.%Destroy.Op.1a2547.7
// CHECK:STDOUT:   %Destroy.Op.call.loc63_29.1: init %empty_tuple.type = call %Destroy.Op.bound.loc63_29.1(%.loc63_29.6)
// CHECK:STDOUT:   br !for.next.loc63
// CHECK:STDOUT:
// CHECK:STDOUT: !for.done.loc63:
// CHECK:STDOUT:   %Destroy.Op.bound.loc63_29.2: <bound method> = bound_method %.loc63_29.6, constants.%Destroy.Op.1a2547.7
// CHECK:STDOUT:   %Destroy.Op.call.loc63_29.2: init %empty_tuple.type = call %Destroy.Op.bound.loc63_29.2(%.loc63_29.6)
// CHECK:STDOUT:   %Destroy.Op.bound.loc63_29.3: <bound method> = bound_method %var.loc63, constants.%Destroy.Op.1a2547.9
// CHECK:STDOUT:   %Destroy.Op.call.loc63_29.3: init %empty_tuple.type = call %Destroy.Op.bound.loc63_29.3(%var.loc63)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestIterate(constants.%R) {
// CHECK:STDOUT:   %R.patt.loc40_17.2 => constants.%R.patt
// CHECK:STDOUT:   %R.loc40_17.1 => constants.%R
// CHECK:STDOUT:   %R.as_type.loc40_75.1 => constants.%R.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.d2d
// CHECK:STDOUT:   %r.patt.loc40_73.2 => constants.%r.patt.d27
// CHECK:STDOUT:   %r.param_patt.loc40_68.2 => constants.%r.param_patt.a41
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestIterate(constants.%facet_value.b35) {
// CHECK:STDOUT:   %R.patt.loc40_17.2 => constants.%R.patt
// CHECK:STDOUT:   %R.loc40_17.1 => constants.%facet_value.b35
// CHECK:STDOUT:   %R.as_type.loc40_75.1 => constants.%MutableRange
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.e635
// CHECK:STDOUT:   %r.patt.loc40_73.2 => constants.%r.patt.f69
// CHECK:STDOUT:   %r.param_patt.loc40_68.2 => constants.%r.param_patt.91e
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc40 => constants.%complete_type.357
// CHECK:STDOUT:   %Iterate.lookup_impl_witness => constants.%Iterate.impl_witness.00a
// CHECK:STDOUT:   %Iterate.facet.loc44_29.5 => constants.%Iterate.facet.a01
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type => constants.%Iterate.WithSelf.NewCursor.type.7e6
// CHECK:STDOUT:   %.loc44_29.20 => constants.%.0ed
// CHECK:STDOUT:   %impl.elem2.loc44_29.2 => constants.%T.as_type.as.Iterate.impl.NewCursor.a3f
// CHECK:STDOUT:   %specific_impl_fn.loc44_29.4 => constants.%T.as_type.as.Iterate.impl.NewCursor.specific_fn.a99
// CHECK:STDOUT:   %impl.elem1 => constants.%Destroy.facet.305
// CHECK:STDOUT:   %as_type => constants.%tuple.type.829
// CHECK:STDOUT:   %require_complete.1 => constants.%complete_type.94f
// CHECK:STDOUT:   %ptr => constants.%ptr.d6b
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type => constants.%Iterate.WithSelf.Next.type.f85
// CHECK:STDOUT:   %.loc44_29.21 => constants.%.1c5
// CHECK:STDOUT:   %impl.elem3.loc44_29.2 => constants.%T.as_type.as.Iterate.impl.Next.db5
// CHECK:STDOUT:   %specific_impl_fn.loc44_29.5 => constants.%T.as_type.as.Iterate.impl.Next.specific_fn.b98
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.c89
// CHECK:STDOUT:   %.loc44_29.22 => constants.%.e8a
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.df9cc1.8
// CHECK:STDOUT:   %impl.elem0.loc44_29.2 => constants.%Destroy.Op.1a2547.8
// CHECK:STDOUT:   %specific_impl_fn.loc44_29.6 => constants.%Destroy.Op.1a2547.8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestIterate(constants.%facet_value.fba) {
// CHECK:STDOUT:   %R.patt.loc40_17.2 => constants.%R.patt
// CHECK:STDOUT:   %R.loc40_17.1 => constants.%facet_value.fba
// CHECK:STDOUT:   %R.as_type.loc40_75.1 => constants.%ConstRange
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.0ba
// CHECK:STDOUT:   %r.patt.loc40_73.2 => constants.%r.patt.36a
// CHECK:STDOUT:   %r.param_patt.loc40_68.2 => constants.%r.param_patt.501
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc40 => constants.%complete_type.357
// CHECK:STDOUT:   %Iterate.lookup_impl_witness => constants.%Iterate.impl_witness.8b4
// CHECK:STDOUT:   %Iterate.facet.loc44_29.5 => constants.%Iterate.facet.d09
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type => constants.%Iterate.WithSelf.NewCursor.type.a99
// CHECK:STDOUT:   %.loc44_29.20 => constants.%.a00
// CHECK:STDOUT:   %impl.elem2.loc44_29.2 => constants.%T.as_type.as.Iterate.impl.NewCursor.4af
// CHECK:STDOUT:   %specific_impl_fn.loc44_29.4 => constants.%T.as_type.as.Iterate.impl.NewCursor.specific_fn.e9b
// CHECK:STDOUT:   %impl.elem1 => constants.%Destroy.facet.e08
// CHECK:STDOUT:   %as_type => constants.%tuple.type.99f
// CHECK:STDOUT:   %require_complete.1 => constants.%complete_type.dea
// CHECK:STDOUT:   %ptr => constants.%ptr.2f7
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type => constants.%Iterate.WithSelf.Next.type.c30
// CHECK:STDOUT:   %.loc44_29.21 => constants.%.c34
// CHECK:STDOUT:   %impl.elem3.loc44_29.2 => constants.%T.as_type.as.Iterate.impl.Next.988
// CHECK:STDOUT:   %specific_impl_fn.loc44_29.5 => constants.%T.as_type.as.Iterate.impl.Next.specific_fn.4f8
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
// CHECK:STDOUT:   %facet_type.f48: type = facet_type <@Destroy & @Copy> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %T.220: %OptionalStorage.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Optional.Get.type.bb9: type = fn_type @Optional.Get, @Optional(%T.220) [symbolic]
// CHECK:STDOUT:   %Optional.Get.fa1: %Optional.Get.type.bb9 = struct_value () [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.type.e9e: type = fn_type @Optional.HasValue, @Optional(%T.220) [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.6cd: %Optional.HasValue.type.e9e = struct_value () [symbolic]
// CHECK:STDOUT:   %T.283: %facet_type.f48 = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.d50: type = class_type @DefaultOptionalStorage, @DefaultOptionalStorage(%T.283) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Get.type.d3c: type = fn_type @T.as_type.as.OptionalStorage.impl.Get, @T.as_type.as.OptionalStorage.impl(%T.283) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Get.da2: %T.as_type.as.OptionalStorage.impl.Get.type.d3c = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Has.type.eb9: type = fn_type @T.as_type.as.OptionalStorage.impl.Has, @T.as_type.as.OptionalStorage.impl(%T.283) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Has.c7e: %T.as_type.as.OptionalStorage.impl.Has.type.eb9 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Some.type.7eb: type = fn_type @T.as_type.as.OptionalStorage.impl.Some, @T.as_type.as.OptionalStorage.impl(%T.283) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Some.2b4: %T.as_type.as.OptionalStorage.impl.Some.type.7eb = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.None.type.d4d: type = fn_type @T.as_type.as.OptionalStorage.impl.None, @T.as_type.as.OptionalStorage.impl(%T.283) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.None.d64: %T.as_type.as.OptionalStorage.impl.None.type.d4d = struct_value () [symbolic]
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
// CHECK:STDOUT:   %CppRangeForIterate.lookup_impl_witness.b581a8.2: <witness> = lookup_impl_witness %T.f45530.2, @CppRangeForIterate [symbolic]
// CHECK:STDOUT:   %impl.elem1.c2dc15.2: type = impl_witness_access %CppRangeForIterate.lookup_impl_witness.b581a8.2, element1 [symbolic]
// CHECK:STDOUT:   %impl.elem0.5f971c.2: type = impl_witness_access %CppRangeForIterate.lookup_impl_witness.b581a8.2, element0 [symbolic]
// CHECK:STDOUT:   %tuple.type.1ba: type = tuple_type (%impl.elem0.5f971c.2, %impl.elem1.c2dc15.2) [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.812: <witness> = lookup_impl_witness %tuple.type.1ba, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.9cd: %Destroy.type = facet_value %tuple.type.1ba, (%Destroy.lookup_impl_witness.812) [symbolic]
// CHECK:STDOUT:   %CppUnsafeDeref.lookup_impl_witness.87e543.2: <witness> = lookup_impl_witness %impl.elem0.5f971c.2, @CppUnsafeDeref [symbolic]
// CHECK:STDOUT:   %impl.elem0.493765.2: type = impl_witness_access %CppUnsafeDeref.lookup_impl_witness.87e543.2, element0 [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.35816c.2: <witness> = lookup_impl_witness %impl.elem0.493765.2, @Destroy [symbolic]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.167: <witness> = lookup_impl_witness %impl.elem0.493765.2, @Copy [symbolic]
// CHECK:STDOUT:   %facet_value.f93: %facet_type.f48 = facet_value %impl.elem0.493765.2, (%Destroy.lookup_impl_witness.35816c.2, %Copy.lookup_impl_witness.167) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.type.5dc: type = fn_type @T.as_type.as.Iterate.impl.Next, @T.as_type.as.Iterate.impl(%T.f45530.2) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.b73: %T.as_type.as.Iterate.impl.Next.type.5dc = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.type.b5d: type = fn_type @T.as_type.as.Iterate.impl.NewCursor, @T.as_type.as.Iterate.impl(%T.f45530.2) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.21f: %T.as_type.as.Iterate.impl.NewCursor.type.b5d = struct_value () [symbolic]
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete]
// CHECK:STDOUT:   %f64.dc1: type = class_type @Float, @Float(%int_64) [concrete]
// CHECK:STDOUT:   %pattern_type.fb7: type = pattern_type %f64.dc1 [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.3: type = fn_type @Destroy.Op.loc20_53.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.3: %Destroy.Op.type.1d8f74.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.df9cc1.3: <witness> = custom_witness (%Destroy.Op.1a2547.3), @Destroy [concrete]
// CHECK:STDOUT:   %Float.as.Copy.impl.Op.type.ff3: type = fn_type @Float.as.Copy.impl.Op, @Float.as.Copy.impl(%N.fe9) [symbolic]
// CHECK:STDOUT:   %Float.as.Copy.impl.Op.d93: %Float.as.Copy.impl.Op.type.ff3 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.873: <witness> = impl_witness imports.%Copy.impl_witness_table.4e8, @Float.as.Copy.impl(%int_64) [concrete]
// CHECK:STDOUT:   %facet_value.209: %facet_type.f48 = facet_value %f64.dc1, (%custom_witness.df9cc1.3, %Copy.impl_witness.873) [concrete]
// CHECK:STDOUT:   %Iterate.lookup_impl_witness.1d8: <witness> = lookup_impl_witness %.Self.500, @Iterate [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.294: %facet_type.f48 = impl_witness_access %Iterate.lookup_impl_witness.1d8, element0 [symbolic_self]
// CHECK:STDOUT:   %Iterate_where.type.a32: type = facet_type <@Iterate where %impl.elem0.294 = %facet_value.209> [concrete]
// CHECK:STDOUT:   %pattern_type.ff5: type = pattern_type %Iterate_where.type.a32 [concrete]
// CHECK:STDOUT:   %R.patt: %pattern_type.ff5 = symbolic_binding_pattern R, 0 [symbolic]
// CHECK:STDOUT:   %R: %Iterate_where.type.a32 = symbolic_binding R, 0 [symbolic]
// CHECK:STDOUT:   %R.as_type: type = facet_access_type %R [symbolic]
// CHECK:STDOUT:   %pattern_type.3d7: type = pattern_type %R.as_type [symbolic]
// CHECK:STDOUT:   %r.param_patt.03a: %pattern_type.3d7 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %r.patt.b86: %pattern_type.3d7 = at_binding_pattern r, %r.param_patt.03a [symbolic]
// CHECK:STDOUT:   %i.patt: %pattern_type.fb7 = value_binding_pattern i [concrete]
// CHECK:STDOUT:   %Iterate.lookup_impl_witness.93c: <witness> = lookup_impl_witness %R, @Iterate [symbolic]
// CHECK:STDOUT:   %Iterate.facet.94d: %Iterate.type = facet_value %R.as_type, (%Iterate.lookup_impl_witness.93c) [symbolic]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.4c7: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.94d) [symbolic]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.275: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.94d) [symbolic]
// CHECK:STDOUT:   %.a62: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.4c7, %Iterate.facet.94d [symbolic]
// CHECK:STDOUT:   %impl.elem2.c7b: %.a62 = impl_witness_access %Iterate.lookup_impl_witness.93c, element2 [symbolic]
// CHECK:STDOUT:   %impl.elem1.7a9: %Destroy.type = impl_witness_access %Iterate.lookup_impl_witness.93c, element1 [symbolic]
// CHECK:STDOUT:   %as_type.8e4: type = facet_access_type %impl.elem1.7a9 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.1b8: <specific function> = specific_impl_function %impl.elem2.c7b, @Iterate.WithSelf.NewCursor(%Iterate.facet.94d) [symbolic]
// CHECK:STDOUT:   %ptr.bac: type = ptr_type %as_type.8e4 [symbolic]
// CHECK:STDOUT:   %.a06: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.275, %Iterate.facet.94d [symbolic]
// CHECK:STDOUT:   %impl.elem3.7bd: %.a06 = impl_witness_access %Iterate.lookup_impl_witness.93c, element3 [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.541: type = class_type @DefaultOptionalStorage, @DefaultOptionalStorage(%facet_value.209) [concrete]
// CHECK:STDOUT:   %OptionalStorage.impl_witness.a56: <witness> = impl_witness imports.%OptionalStorage.impl_witness_table.80d, @T.as_type.as.OptionalStorage.impl(%facet_value.209) [concrete]
// CHECK:STDOUT:   %OptionalStorage.facet.63c: %OptionalStorage.type = facet_value %f64.dc1, (%OptionalStorage.impl_witness.a56) [concrete]
// CHECK:STDOUT:   %Optional.e65: type = class_type @Optional, @Optional(%OptionalStorage.facet.63c) [concrete]
// CHECK:STDOUT:   %specific_impl_fn.666: <specific function> = specific_impl_function %impl.elem3.7bd, @Iterate.WithSelf.Next(%Iterate.facet.94d) [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.type.731: type = fn_type @Optional.HasValue, @Optional(%OptionalStorage.facet.63c) [concrete]
// CHECK:STDOUT:   %Optional.HasValue.904: %Optional.HasValue.type.731 = struct_value () [concrete]
// CHECK:STDOUT:   %Optional.Get.type.811: type = fn_type @Optional.Get, @Optional(%OptionalStorage.facet.63c) [concrete]
// CHECK:STDOUT:   %Optional.Get.423: %Optional.Get.type.811 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.facet.339: %Destroy.type = facet_value %f64.dc1, (%custom_witness.df9cc1.3) [concrete]
// CHECK:STDOUT:   %MaybeUnformed.b7e: type = class_type @MaybeUnformed, @MaybeUnformed(%Destroy.facet.339) [concrete]
// CHECK:STDOUT:   %.cc0: type = maybe_unformed_type %f64.dc1 [concrete]
// CHECK:STDOUT:   %struct_type.value.has_value.ad9: type = struct_type {.value: %MaybeUnformed.b7e, .has_value: bool} [concrete]
// CHECK:STDOUT:   %Optional.HasValue.specific_fn.b6b: <specific function> = specific_function %Optional.HasValue.904, @Optional.HasValue(%OptionalStorage.facet.63c) [concrete]
// CHECK:STDOUT:   %Optional.Get.specific_fn.f0c: <specific function> = specific_function %Optional.Get.423, @Optional.Get(%OptionalStorage.facet.63c) [concrete]
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
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.def: <witness> = lookup_impl_witness %impl.elem1.7a9, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.41f: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%impl.elem1.7a9) [symbolic]
// CHECK:STDOUT:   %.148: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.41f, %impl.elem1.7a9 [symbolic]
// CHECK:STDOUT:   %impl.elem0.461: %.148 = impl_witness_access %Destroy.lookup_impl_witness.def, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.095b: <specific function> = specific_impl_function %impl.elem0.461, @Destroy.WithSelf.Op(%impl.elem1.7a9) [symbolic]
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
// CHECK:STDOUT:   %facet_value.f27: %CppRange_where.type.d682d6.1 = facet_value %ConstRange, (%custom_witness.df9cc1.3, %Copy.impl_witness.873, %custom_witness.5d5, %custom_witness.213, %custom_witness.994, %custom_witness.393, %custom_witness.821) [concrete]
// CHECK:STDOUT:   %tuple.type.005: type = tuple_type (%Iterator, %Iterator) [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.10: type = fn_type @Destroy.Op.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.10: %Destroy.Op.type.1d8f74.10 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.df9cc1.10: <witness> = custom_witness (%Destroy.Op.1a2547.10), @Destroy [concrete]
// CHECK:STDOUT:   %Destroy.facet.40b: %Destroy.type = facet_value %tuple.type.005, (%custom_witness.df9cc1.10) [concrete]
// CHECK:STDOUT:   %Iterate.impl_witness.ca4: <witness> = impl_witness imports.%Iterate.impl_witness_table.478, @T.as_type.as.Iterate.impl(%facet_value.f27) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.type.259: type = fn_type @T.as_type.as.Iterate.impl.NewCursor, @T.as_type.as.Iterate.impl(%facet_value.f27) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.980: %T.as_type.as.Iterate.impl.NewCursor.type.259 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.type.2ba: type = fn_type @T.as_type.as.Iterate.impl.Next, @T.as_type.as.Iterate.impl(%facet_value.f27) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.16a: %T.as_type.as.Iterate.impl.Next.type.2ba = struct_value () [concrete]
// CHECK:STDOUT:   %Iterate.facet.1cc: %Iterate.type = facet_value %ConstRange, (%Iterate.impl_witness.ca4) [concrete]
// CHECK:STDOUT:   %facet_value.a64: %Iterate_where.type.a32 = facet_value %ConstRange, (%Iterate.impl_witness.ca4) [concrete]
// CHECK:STDOUT:   %r.param_patt.a73: %pattern_type.025 = value_param_pattern [concrete]
// CHECK:STDOUT:   %r.patt.0e2: %pattern_type.025 = at_binding_pattern r, %r.param_patt.a73 [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.769: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.1cc) [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.bfd: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.1cc) [concrete]
// CHECK:STDOUT:   %.166: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.769, %Iterate.facet.1cc [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.specific_fn: <specific function> = specific_function %T.as_type.as.Iterate.impl.NewCursor.980, @T.as_type.as.Iterate.impl.NewCursor(%facet_value.f27) [concrete]
// CHECK:STDOUT:   %ptr.d1a: type = ptr_type %tuple.type.005 [concrete]
// CHECK:STDOUT:   %.131: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.bfd, %Iterate.facet.1cc [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.specific_fn: <specific function> = specific_function %T.as_type.as.Iterate.impl.Next.16a, @T.as_type.as.Iterate.impl.Next(%facet_value.f27) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.117: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.40b) [concrete]
// CHECK:STDOUT:   %.5a4: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.117, %Destroy.facet.40b [concrete]
// CHECK:STDOUT:   %complete_type.320: <witness> = complete_type_witness %tuple.type.005 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.bb6: @Optional.%Optional.HasValue.type (%Optional.HasValue.type.e9e) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.HasValue (constants.%Optional.HasValue.6cd)]
// CHECK:STDOUT:   %Core.import_ref.daba: @Optional.%Optional.Get.type (%Optional.Get.type.bb9) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.Get (constants.%Optional.Get.fa1)]
// CHECK:STDOUT:   %Core.import_ref.0c6: type = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%DefaultOptionalStorage (constants.%DefaultOptionalStorage.d50)]
// CHECK:STDOUT:   %Core.import_ref.13b: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.None.type (%T.as_type.as.OptionalStorage.impl.None.type.d4d) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.None (constants.%T.as_type.as.OptionalStorage.impl.None.d64)]
// CHECK:STDOUT:   %Core.import_ref.c43: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Some.type (%T.as_type.as.OptionalStorage.impl.Some.type.7eb) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Some (constants.%T.as_type.as.OptionalStorage.impl.Some.2b4)]
// CHECK:STDOUT:   %Core.import_ref.61e: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Has.type (%T.as_type.as.OptionalStorage.impl.Has.type.eb9) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Has (constants.%T.as_type.as.OptionalStorage.impl.Has.c7e)]
// CHECK:STDOUT:   %Core.import_ref.ce2e: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Get.type (%T.as_type.as.OptionalStorage.impl.Get.type.d3c) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Get (constants.%T.as_type.as.OptionalStorage.impl.Get.da2)]
// CHECK:STDOUT:   %Core.import_ref.8fc = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], unloaded
// CHECK:STDOUT:   %OptionalStorage.impl_witness_table.80d = impl_witness_table (%Core.import_ref.0c6, %Core.import_ref.13b, %Core.import_ref.c43, %Core.import_ref.61e, %Core.import_ref.ce2e, %Core.import_ref.8fc), @T.as_type.as.OptionalStorage.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.eef: %facet_type.f48 = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.Iterate.impl.%facet_value (constants.%facet_value.f93)]
// CHECK:STDOUT:   %Core.import_ref.fcc: %Destroy.type = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.Iterate.impl.%Destroy.facet (constants.%Destroy.facet.9cd)]
// CHECK:STDOUT:   %Core.import_ref.f50: @T.as_type.as.Iterate.impl.%T.as_type.as.Iterate.impl.NewCursor.type (%T.as_type.as.Iterate.impl.NewCursor.type.b5d) = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.Iterate.impl.%T.as_type.as.Iterate.impl.NewCursor (constants.%T.as_type.as.Iterate.impl.NewCursor.21f)]
// CHECK:STDOUT:   %Core.import_ref.590: @T.as_type.as.Iterate.impl.%T.as_type.as.Iterate.impl.Next.type (%T.as_type.as.Iterate.impl.Next.type.5dc) = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.Iterate.impl.%T.as_type.as.Iterate.impl.Next (constants.%T.as_type.as.Iterate.impl.Next.b73)]
// CHECK:STDOUT:   %Iterate.impl_witness_table.478 = impl_witness_table (%Core.import_ref.eef, %Core.import_ref.fcc, %Core.import_ref.f50, %Core.import_ref.590), @T.as_type.as.Iterate.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.e69: @Float.as.Copy.impl.%Float.as.Copy.impl.Op.type (%Float.as.Copy.impl.Op.type.ff3) = import_ref Core//prelude/types/float, loc{{\d+_\d+}}, loaded [symbolic = @Float.as.Copy.impl.%Float.as.Copy.impl.Op (constants.%Float.as.Copy.impl.Op.d93)]
// CHECK:STDOUT:   %Copy.impl_witness_table.4e8 = impl_witness_table (%Core.import_ref.e69), @Float.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.cb3: @Float.as.AddAssignWith.impl.%Float.as.AddAssignWith.impl.Op.type (%Float.as.AddAssignWith.impl.Op.type.fdd) = import_ref Core//prelude/types/float, loc{{\d+_\d+}}, loaded [symbolic = @Float.as.AddAssignWith.impl.%Float.as.AddAssignWith.impl.Op (constants.%Float.as.AddAssignWith.impl.Op.430)]
// CHECK:STDOUT:   %AddAssignWith.impl_witness_table = impl_witness_table (%Core.import_ref.cb3), @Float.as.AddAssignWith.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @TestIterate(%R.loc20_17.2: %Iterate_where.type.a32) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Iterate.lookup_impl_witness: <witness> = lookup_impl_witness %R.loc20_17.1, @Iterate [symbolic = %Iterate.lookup_impl_witness (constants.%Iterate.lookup_impl_witness.93c)]
// CHECK:STDOUT:   %Iterate.facet.loc24_19.5: %Iterate.type = facet_value %R.as_type.loc20_61.1, (%Iterate.lookup_impl_witness) [symbolic = %Iterate.facet.loc24_19.5 (constants.%Iterate.facet.94d)]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.loc24_19.5) [symbolic = %Iterate.WithSelf.NewCursor.type (constants.%Iterate.WithSelf.NewCursor.type.4c7)]
// CHECK:STDOUT:   %.loc24_19.19: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type, %Iterate.facet.loc24_19.5 [symbolic = %.loc24_19.19 (constants.%.a62)]
// CHECK:STDOUT:   %impl.elem2.loc24_19.2: @TestIterate.%.loc24_19.19 (%.a62) = impl_witness_access %Iterate.lookup_impl_witness, element2 [symbolic = %impl.elem2.loc24_19.2 (constants.%impl.elem2.c7b)]
// CHECK:STDOUT:   %specific_impl_fn.loc24_19.4: <specific function> = specific_impl_function %impl.elem2.loc24_19.2, @Iterate.WithSelf.NewCursor(%Iterate.facet.loc24_19.5) [symbolic = %specific_impl_fn.loc24_19.4 (constants.%specific_impl_fn.1b8)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %ptr: type = ptr_type %as_type [symbolic = %ptr (constants.%ptr.bac)]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.loc24_19.5) [symbolic = %Iterate.WithSelf.Next.type (constants.%Iterate.WithSelf.Next.type.275)]
// CHECK:STDOUT:   %.loc24_19.20: type = fn_type_with_self_type %Iterate.WithSelf.Next.type, %Iterate.facet.loc24_19.5 [symbolic = %.loc24_19.20 (constants.%.a06)]
// CHECK:STDOUT:   %impl.elem3.loc24_19.2: @TestIterate.%.loc24_19.20 (%.a06) = impl_witness_access %Iterate.lookup_impl_witness, element3 [symbolic = %impl.elem3.loc24_19.2 (constants.%impl.elem3.7bd)]
// CHECK:STDOUT:   %specific_impl_fn.loc24_19.5: <specific function> = specific_impl_function %impl.elem3.loc24_19.2, @Iterate.WithSelf.Next(%Iterate.facet.loc24_19.5) [symbolic = %specific_impl_fn.loc24_19.5 (constants.%specific_impl_fn.666)]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%impl.elem1) [symbolic = %Destroy.WithSelf.Op.type (constants.%Destroy.WithSelf.Op.type.41f)]
// CHECK:STDOUT:   %.loc24_19.21: type = fn_type_with_self_type %Destroy.WithSelf.Op.type, %impl.elem1 [symbolic = %.loc24_19.21 (constants.%.148)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %impl.elem1, @Destroy [symbolic = %Destroy.lookup_impl_witness (constants.%Destroy.lookup_impl_witness.def)]
// CHECK:STDOUT:   %impl.elem0.loc24_19.2: @TestIterate.%.loc24_19.21 (%.148) = impl_witness_access %Destroy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc24_19.2 (constants.%impl.elem0.461)]
// CHECK:STDOUT:   %specific_impl_fn.loc24_19.6: <specific function> = specific_impl_function %impl.elem0.loc24_19.2, @Destroy.WithSelf.Op(%impl.elem1) [symbolic = %specific_impl_fn.loc24_19.6 (constants.%specific_impl_fn.095b)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%r.param: @TestIterate.%R.as_type.loc20_61.1 (%R.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %i.patt: %pattern_type.fb7 = value_binding_pattern i [concrete = constants.%i.patt]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %r.ref: @TestIterate.%R.as_type.loc20_61.1 (%R.as_type) = name_ref r, %r
// CHECK:STDOUT:     %impl.elem2.loc24_19.1: @TestIterate.%.loc24_19.19 (%.a62) = impl_witness_access constants.%Iterate.lookup_impl_witness.93c, element2 [symbolic = %impl.elem2.loc24_19.2 (constants.%impl.elem2.c7b)]
// CHECK:STDOUT:     %bound_method.loc24_19.1: <bound method> = bound_method %r.ref, %impl.elem2.loc24_19.1
// CHECK:STDOUT:     %Iterate.facet.loc24_19.1: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.93c) [symbolic = %Iterate.facet.loc24_19.5 (constants.%Iterate.facet.94d)]
// CHECK:STDOUT:     %.loc24_19.1: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc24_19.1 [symbolic = %Iterate.facet.loc24_19.5 (constants.%Iterate.facet.94d)]
// CHECK:STDOUT:     %Iterate.facet.loc24_19.2: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.93c) [symbolic = %Iterate.facet.loc24_19.5 (constants.%Iterate.facet.94d)]
// CHECK:STDOUT:     %.loc24_19.2: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc24_19.2 [symbolic = %Iterate.facet.loc24_19.5 (constants.%Iterate.facet.94d)]
// CHECK:STDOUT:     %specific_impl_fn.loc24_19.1: <specific function> = specific_impl_function %impl.elem2.loc24_19.1, @Iterate.WithSelf.NewCursor(constants.%Iterate.facet.94d) [symbolic = %specific_impl_fn.loc24_19.4 (constants.%specific_impl_fn.1b8)]
// CHECK:STDOUT:     %bound_method.loc24_19.2: <bound method> = bound_method %r.ref, %specific_impl_fn.loc24_19.1
// CHECK:STDOUT:     %var: ref @TestIterate.%as_type (%as_type.8e4) = var_storage invalid
// CHECK:STDOUT:     %Iterate.WithSelf.NewCursor.call: init @TestIterate.%as_type (%as_type.8e4) to %var = call %bound_method.loc24_19.2(%r.ref)
// CHECK:STDOUT:     assign %var, %Iterate.WithSelf.NewCursor.call
// CHECK:STDOUT:     br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.next:
// CHECK:STDOUT:     %addr: @TestIterate.%ptr (%ptr.bac) = addr_of %var
// CHECK:STDOUT:     %impl.elem3.loc24_19.1: @TestIterate.%.loc24_19.20 (%.a06) = impl_witness_access constants.%Iterate.lookup_impl_witness.93c, element3 [symbolic = %impl.elem3.loc24_19.2 (constants.%impl.elem3.7bd)]
// CHECK:STDOUT:     %bound_method.loc24_19.3: <bound method> = bound_method %r.ref, %impl.elem3.loc24_19.1
// CHECK:STDOUT:     %Iterate.facet.loc24_19.3: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.93c) [symbolic = %Iterate.facet.loc24_19.5 (constants.%Iterate.facet.94d)]
// CHECK:STDOUT:     %.loc24_19.3: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc24_19.3 [symbolic = %Iterate.facet.loc24_19.5 (constants.%Iterate.facet.94d)]
// CHECK:STDOUT:     %Iterate.facet.loc24_19.4: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.93c) [symbolic = %Iterate.facet.loc24_19.5 (constants.%Iterate.facet.94d)]
// CHECK:STDOUT:     %.loc24_19.4: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc24_19.4 [symbolic = %Iterate.facet.loc24_19.5 (constants.%Iterate.facet.94d)]
// CHECK:STDOUT:     %.loc24_19.5: %Destroy.type = converted constants.%as_type.8e4, constants.%impl.elem1.7a9 [symbolic = %impl.elem1 (constants.%impl.elem1.7a9)]
// CHECK:STDOUT:     %.loc24_19.6: %Destroy.type = converted constants.%as_type.8e4, constants.%impl.elem1.7a9 [symbolic = %impl.elem1 (constants.%impl.elem1.7a9)]
// CHECK:STDOUT:     %specific_impl_fn.loc24_19.2: <specific function> = specific_impl_function %impl.elem3.loc24_19.1, @Iterate.WithSelf.Next(constants.%Iterate.facet.94d) [symbolic = %specific_impl_fn.loc24_19.5 (constants.%specific_impl_fn.666)]
// CHECK:STDOUT:     %bound_method.loc24_19.4: <bound method> = bound_method %r.ref, %specific_impl_fn.loc24_19.2
// CHECK:STDOUT:     %.loc24_19.7: ref %Optional.e65 = temporary_storage
// CHECK:STDOUT:     %Iterate.WithSelf.Next.call: init %Optional.e65 to %.loc24_19.7 = call %bound_method.loc24_19.4(%r.ref, %addr)
// CHECK:STDOUT:     %.loc24_19.8: ref %Optional.e65 = temporary %.loc24_19.7, %Iterate.WithSelf.Next.call
// CHECK:STDOUT:     %.loc24_19.9: %Optional.HasValue.type.731 = specific_constant imports.%Core.import_ref.bb6, @Optional(constants.%OptionalStorage.facet.63c) [concrete = constants.%Optional.HasValue.904]
// CHECK:STDOUT:     %HasValue.ref: %Optional.HasValue.type.731 = name_ref HasValue, %.loc24_19.9 [concrete = constants.%Optional.HasValue.904]
// CHECK:STDOUT:     %Optional.HasValue.bound: <bound method> = bound_method %.loc24_19.8, %HasValue.ref
// CHECK:STDOUT:     %Optional.HasValue.specific_fn: <specific function> = specific_function %HasValue.ref, @Optional.HasValue(constants.%OptionalStorage.facet.63c) [concrete = constants.%Optional.HasValue.specific_fn.b6b]
// CHECK:STDOUT:     %bound_method.loc24_19.5: <bound method> = bound_method %.loc24_19.8, %Optional.HasValue.specific_fn
// CHECK:STDOUT:     %.loc24_19.10: %Optional.e65 = acquire_value %.loc24_19.8
// CHECK:STDOUT:     %Optional.HasValue.call: init bool = call %bound_method.loc24_19.5(%.loc24_19.10)
// CHECK:STDOUT:     %.loc24_19.11: bool = value_of_initializer %Optional.HasValue.call
// CHECK:STDOUT:     %.loc24_19.12: bool = converted %Optional.HasValue.call, %.loc24_19.11
// CHECK:STDOUT:     if %.loc24_19.12 br !for.body else br !for.done
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.body:
// CHECK:STDOUT:     %.loc24_19.13: %Optional.Get.type.811 = specific_constant imports.%Core.import_ref.daba, @Optional(constants.%OptionalStorage.facet.63c) [concrete = constants.%Optional.Get.423]
// CHECK:STDOUT:     %Get.ref: %Optional.Get.type.811 = name_ref Get, %.loc24_19.13 [concrete = constants.%Optional.Get.423]
// CHECK:STDOUT:     %Optional.Get.bound: <bound method> = bound_method %.loc24_19.8, %Get.ref
// CHECK:STDOUT:     %Optional.Get.specific_fn: <specific function> = specific_function %Get.ref, @Optional.Get(constants.%OptionalStorage.facet.63c) [concrete = constants.%Optional.Get.specific_fn.f0c]
// CHECK:STDOUT:     %bound_method.loc24_19.6: <bound method> = bound_method %.loc24_19.8, %Optional.Get.specific_fn
// CHECK:STDOUT:     %.loc24_19.14: %Optional.e65 = acquire_value %.loc24_19.8
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
// CHECK:STDOUT:     %Destroy.Op.bound.loc24_19.1: <bound method> = bound_method %.loc24_19.8, constants.%Destroy.Op.1a2547.9
// CHECK:STDOUT:     %Destroy.Op.call.loc24_19.1: init %empty_tuple.type = call %Destroy.Op.bound.loc24_19.1(%.loc24_19.8)
// CHECK:STDOUT:     br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.done:
// CHECK:STDOUT:     %Destroy.Op.bound.loc24_19.2: <bound method> = bound_method %.loc24_19.8, constants.%Destroy.Op.1a2547.9
// CHECK:STDOUT:     %Destroy.Op.call.loc24_19.2: init %empty_tuple.type = call %Destroy.Op.bound.loc24_19.2(%.loc24_19.8)
// CHECK:STDOUT:     %impl.elem0.loc24_19.1: @TestIterate.%.loc24_19.21 (%.148) = impl_witness_access constants.%Destroy.lookup_impl_witness.def, element0 [symbolic = %impl.elem0.loc24_19.2 (constants.%impl.elem0.461)]
// CHECK:STDOUT:     %bound_method.loc24_19.7: <bound method> = bound_method %var, %impl.elem0.loc24_19.1
// CHECK:STDOUT:     %.loc24_19.17: %Destroy.type = converted constants.%as_type.8e4, constants.%impl.elem1.7a9 [symbolic = %impl.elem1 (constants.%impl.elem1.7a9)]
// CHECK:STDOUT:     %.loc24_19.18: %Destroy.type = converted constants.%as_type.8e4, constants.%impl.elem1.7a9 [symbolic = %impl.elem1 (constants.%impl.elem1.7a9)]
// CHECK:STDOUT:     %specific_impl_fn.loc24_19.3: <specific function> = specific_impl_function %impl.elem0.loc24_19.1, @Destroy.WithSelf.Op(constants.%impl.elem1.7a9) [symbolic = %specific_impl_fn.loc24_19.6 (constants.%specific_impl_fn.095b)]
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
// CHECK:STDOUT: fn @Destroy.Op.loc24_19.6(%self.param: ref %Optional.e65) {
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
// CHECK:STDOUT:   %impl.elem2: %.166 = impl_witness_access constants.%Iterate.impl_witness.ca4, element2 [concrete = constants.%T.as_type.as.Iterate.impl.NewCursor.980]
// CHECK:STDOUT:   %bound_method.loc38_19.1: <bound method> = bound_method %c.ref, %impl.elem2
// CHECK:STDOUT:   %facet_value.loc38_19.1: %CppRange_where.type.d682d6.1 = facet_value constants.%ConstRange, (constants.%custom_witness.df9cc1.3, constants.%Copy.impl_witness.873, constants.%custom_witness.5d5, constants.%custom_witness.213, constants.%custom_witness.994, constants.%custom_witness.393, constants.%custom_witness.821) [concrete = constants.%facet_value.f27]
// CHECK:STDOUT:   %.loc38_19.1: %CppRange_where.type.d682d6.1 = converted constants.%ConstRange, %facet_value.loc38_19.1 [concrete = constants.%facet_value.f27]
// CHECK:STDOUT:   %facet_value.loc38_19.2: %CppRange_where.type.d682d6.1 = facet_value constants.%ConstRange, (constants.%custom_witness.df9cc1.3, constants.%Copy.impl_witness.873, constants.%custom_witness.5d5, constants.%custom_witness.213, constants.%custom_witness.994, constants.%custom_witness.393, constants.%custom_witness.821) [concrete = constants.%facet_value.f27]
// CHECK:STDOUT:   %.loc38_19.2: %CppRange_where.type.d682d6.1 = converted constants.%ConstRange, %facet_value.loc38_19.2 [concrete = constants.%facet_value.f27]
// CHECK:STDOUT:   %specific_fn.loc38_19.1: <specific function> = specific_function %impl.elem2, @T.as_type.as.Iterate.impl.NewCursor(constants.%facet_value.f27) [concrete = constants.%T.as_type.as.Iterate.impl.NewCursor.specific_fn]
// CHECK:STDOUT:   %bound_method.loc38_19.2: <bound method> = bound_method %c.ref, %specific_fn.loc38_19.1
// CHECK:STDOUT:   %var: ref %tuple.type.005 = var_storage invalid
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.call: init %tuple.type.005 to %var = call %bound_method.loc38_19.2(%c.ref)
// CHECK:STDOUT:   assign %var, %T.as_type.as.Iterate.impl.NewCursor.call
// CHECK:STDOUT:   br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT: !for.next:
// CHECK:STDOUT:   %addr: %ptr.d1a = addr_of %var
// CHECK:STDOUT:   %impl.elem3: %.131 = impl_witness_access constants.%Iterate.impl_witness.ca4, element3 [concrete = constants.%T.as_type.as.Iterate.impl.Next.16a]
// CHECK:STDOUT:   %bound_method.loc38_19.3: <bound method> = bound_method %c.ref, %impl.elem3
// CHECK:STDOUT:   %facet_value.loc38_19.3: %CppRange_where.type.d682d6.1 = facet_value constants.%ConstRange, (constants.%custom_witness.df9cc1.3, constants.%Copy.impl_witness.873, constants.%custom_witness.5d5, constants.%custom_witness.213, constants.%custom_witness.994, constants.%custom_witness.393, constants.%custom_witness.821) [concrete = constants.%facet_value.f27]
// CHECK:STDOUT:   %.loc38_19.3: %CppRange_where.type.d682d6.1 = converted constants.%ConstRange, %facet_value.loc38_19.3 [concrete = constants.%facet_value.f27]
// CHECK:STDOUT:   %facet_value.loc38_19.4: %CppRange_where.type.d682d6.1 = facet_value constants.%ConstRange, (constants.%custom_witness.df9cc1.3, constants.%Copy.impl_witness.873, constants.%custom_witness.5d5, constants.%custom_witness.213, constants.%custom_witness.994, constants.%custom_witness.393, constants.%custom_witness.821) [concrete = constants.%facet_value.f27]
// CHECK:STDOUT:   %.loc38_19.4: %CppRange_where.type.d682d6.1 = converted constants.%ConstRange, %facet_value.loc38_19.4 [concrete = constants.%facet_value.f27]
// CHECK:STDOUT:   %specific_fn.loc38_19.2: <specific function> = specific_function %impl.elem3, @T.as_type.as.Iterate.impl.Next(constants.%facet_value.f27) [concrete = constants.%T.as_type.as.Iterate.impl.Next.specific_fn]
// CHECK:STDOUT:   %bound_method.loc38_19.4: <bound method> = bound_method %c.ref, %specific_fn.loc38_19.2
// CHECK:STDOUT:   %.loc38_19.5: ref %Optional.e65 = temporary_storage
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.call: init %Optional.e65 to %.loc38_19.5 = call %bound_method.loc38_19.4(%c.ref, %addr)
// CHECK:STDOUT:   %.loc38_19.6: ref %Optional.e65 = temporary %.loc38_19.5, %T.as_type.as.Iterate.impl.Next.call
// CHECK:STDOUT:   %.loc38_19.7: %Optional.HasValue.type.731 = specific_constant imports.%Core.import_ref.bb6, @Optional(constants.%OptionalStorage.facet.63c) [concrete = constants.%Optional.HasValue.904]
// CHECK:STDOUT:   %HasValue.ref: %Optional.HasValue.type.731 = name_ref HasValue, %.loc38_19.7 [concrete = constants.%Optional.HasValue.904]
// CHECK:STDOUT:   %Optional.HasValue.bound: <bound method> = bound_method %.loc38_19.6, %HasValue.ref
// CHECK:STDOUT:   %Optional.HasValue.specific_fn: <specific function> = specific_function %HasValue.ref, @Optional.HasValue(constants.%OptionalStorage.facet.63c) [concrete = constants.%Optional.HasValue.specific_fn.b6b]
// CHECK:STDOUT:   %bound_method.loc38_19.5: <bound method> = bound_method %.loc38_19.6, %Optional.HasValue.specific_fn
// CHECK:STDOUT:   %.loc38_19.8: %Optional.e65 = acquire_value %.loc38_19.6
// CHECK:STDOUT:   %Optional.HasValue.call: init bool = call %bound_method.loc38_19.5(%.loc38_19.8)
// CHECK:STDOUT:   %.loc38_19.9: bool = value_of_initializer %Optional.HasValue.call
// CHECK:STDOUT:   %.loc38_19.10: bool = converted %Optional.HasValue.call, %.loc38_19.9
// CHECK:STDOUT:   if %.loc38_19.10 br !for.body else br !for.done
// CHECK:STDOUT:
// CHECK:STDOUT: !for.body:
// CHECK:STDOUT:   %.loc38_19.11: %Optional.Get.type.811 = specific_constant imports.%Core.import_ref.daba, @Optional(constants.%OptionalStorage.facet.63c) [concrete = constants.%Optional.Get.423]
// CHECK:STDOUT:   %Get.ref: %Optional.Get.type.811 = name_ref Get, %.loc38_19.11 [concrete = constants.%Optional.Get.423]
// CHECK:STDOUT:   %Optional.Get.bound: <bound method> = bound_method %.loc38_19.6, %Get.ref
// CHECK:STDOUT:   %Optional.Get.specific_fn: <specific function> = specific_function %Get.ref, @Optional.Get(constants.%OptionalStorage.facet.63c) [concrete = constants.%Optional.Get.specific_fn.f0c]
// CHECK:STDOUT:   %bound_method.loc38_19.6: <bound method> = bound_method %.loc38_19.6, %Optional.Get.specific_fn
// CHECK:STDOUT:   %.loc38_19.12: %Optional.e65 = acquire_value %.loc38_19.6
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
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.3d7
// CHECK:STDOUT:   %r.param_patt.loc20_59.2 => constants.%r.param_patt.03a
// CHECK:STDOUT:   %r.patt.loc20_59.2 => constants.%r.patt.b86
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestIterate(constants.%facet_value.a64) {
// CHECK:STDOUT:   %R.patt.loc20_17.2 => constants.%R.patt
// CHECK:STDOUT:   %R.loc20_17.1 => constants.%facet_value.a64
// CHECK:STDOUT:   %R.as_type.loc20_61.1 => constants.%ConstRange
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.025
// CHECK:STDOUT:   %r.param_patt.loc20_59.2 => constants.%r.param_patt.a73
// CHECK:STDOUT:   %r.patt.loc20_59.2 => constants.%r.patt.0e2
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc20 => constants.%complete_type.357
// CHECK:STDOUT:   %Iterate.lookup_impl_witness => constants.%Iterate.impl_witness.ca4
// CHECK:STDOUT:   %Iterate.facet.loc24_19.5 => constants.%Iterate.facet.1cc
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type => constants.%Iterate.WithSelf.NewCursor.type.769
// CHECK:STDOUT:   %.loc24_19.19 => constants.%.166
// CHECK:STDOUT:   %impl.elem2.loc24_19.2 => constants.%T.as_type.as.Iterate.impl.NewCursor.980
// CHECK:STDOUT:   %specific_impl_fn.loc24_19.4 => constants.%T.as_type.as.Iterate.impl.NewCursor.specific_fn
// CHECK:STDOUT:   %impl.elem1 => constants.%Destroy.facet.40b
// CHECK:STDOUT:   %as_type => constants.%tuple.type.005
// CHECK:STDOUT:   %require_complete.1 => constants.%complete_type.320
// CHECK:STDOUT:   %ptr => constants.%ptr.d1a
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type => constants.%Iterate.WithSelf.Next.type.bfd
// CHECK:STDOUT:   %.loc24_19.20 => constants.%.131
// CHECK:STDOUT:   %impl.elem3.loc24_19.2 => constants.%T.as_type.as.Iterate.impl.Next.16a
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
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Get.type.d3c: type = fn_type @T.as_type.as.OptionalStorage.impl.Get, @T.as_type.as.OptionalStorage.impl(%T.283) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Get.da2: %T.as_type.as.OptionalStorage.impl.Get.type.d3c = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Has.type.eb9: type = fn_type @T.as_type.as.OptionalStorage.impl.Has, @T.as_type.as.OptionalStorage.impl(%T.283) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Has.c7e: %T.as_type.as.OptionalStorage.impl.Has.type.eb9 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Some.type.7eb: type = fn_type @T.as_type.as.OptionalStorage.impl.Some, @T.as_type.as.OptionalStorage.impl(%T.283) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Some.2b4: %T.as_type.as.OptionalStorage.impl.Some.type.7eb = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.None.type.d4d: type = fn_type @T.as_type.as.OptionalStorage.impl.None, @T.as_type.as.OptionalStorage.impl(%T.283) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.None.d64: %T.as_type.as.OptionalStorage.impl.None.type.d4d = struct_value () [symbolic]
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
// CHECK:STDOUT:   %CppRangeForIterate.lookup_impl_witness.b581a8.2: <witness> = lookup_impl_witness %T.f45530.2, @CppRangeForIterate [symbolic]
// CHECK:STDOUT:   %impl.elem1.c2dc15.2: type = impl_witness_access %CppRangeForIterate.lookup_impl_witness.b581a8.2, element1 [symbolic]
// CHECK:STDOUT:   %impl.elem0.5f971c.2: type = impl_witness_access %CppRangeForIterate.lookup_impl_witness.b581a8.2, element0 [symbolic]
// CHECK:STDOUT:   %tuple.type.1ba: type = tuple_type (%impl.elem0.5f971c.2, %impl.elem1.c2dc15.2) [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.812: <witness> = lookup_impl_witness %tuple.type.1ba, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.9cd: %Destroy.type = facet_value %tuple.type.1ba, (%Destroy.lookup_impl_witness.812) [symbolic]
// CHECK:STDOUT:   %CppUnsafeDeref.lookup_impl_witness.87e543.2: <witness> = lookup_impl_witness %impl.elem0.5f971c.2, @CppUnsafeDeref [symbolic]
// CHECK:STDOUT:   %impl.elem0.493765.2: type = impl_witness_access %CppUnsafeDeref.lookup_impl_witness.87e543.2, element0 [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.35816c.2: <witness> = lookup_impl_witness %impl.elem0.493765.2, @Destroy [symbolic]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.167: <witness> = lookup_impl_witness %impl.elem0.493765.2, @Copy [symbolic]
// CHECK:STDOUT:   %facet_value.f93: %facet_type.f48 = facet_value %impl.elem0.493765.2, (%Destroy.lookup_impl_witness.35816c.2, %Copy.lookup_impl_witness.167) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.type.5dc: type = fn_type @T.as_type.as.Iterate.impl.Next, @T.as_type.as.Iterate.impl(%T.f45530.2) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.b73: %T.as_type.as.Iterate.impl.Next.type.5dc = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.type.b5d: type = fn_type @T.as_type.as.Iterate.impl.NewCursor, @T.as_type.as.Iterate.impl(%T.f45530.2) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.21f: %T.as_type.as.Iterate.impl.NewCursor.type.b5d = struct_value () [symbolic]
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete]
// CHECK:STDOUT:   %f64.dc1: type = class_type @Float, @Float(%int_64) [concrete]
// CHECK:STDOUT:   %pattern_type.fb7: type = pattern_type %f64.dc1 [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.3: type = fn_type @Destroy.Op.loc22_53.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.3: %Destroy.Op.type.1d8f74.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.df9cc1.3: <witness> = custom_witness (%Destroy.Op.1a2547.3), @Destroy [concrete]
// CHECK:STDOUT:   %Float.as.Copy.impl.Op.type.ff3: type = fn_type @Float.as.Copy.impl.Op, @Float.as.Copy.impl(%N.fe9) [symbolic]
// CHECK:STDOUT:   %Float.as.Copy.impl.Op.d93: %Float.as.Copy.impl.Op.type.ff3 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.873: <witness> = impl_witness imports.%Copy.impl_witness_table.4e8, @Float.as.Copy.impl(%int_64) [concrete]
// CHECK:STDOUT:   %facet_value.209: %facet_type.f48 = facet_value %f64.dc1, (%custom_witness.df9cc1.3, %Copy.impl_witness.873) [concrete]
// CHECK:STDOUT:   %Iterate.lookup_impl_witness.1d8: <witness> = lookup_impl_witness %.Self.500, @Iterate [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.294: %facet_type.f48 = impl_witness_access %Iterate.lookup_impl_witness.1d8, element0 [symbolic_self]
// CHECK:STDOUT:   %Iterate_where.type.a32: type = facet_type <@Iterate where %impl.elem0.294 = %facet_value.209> [concrete]
// CHECK:STDOUT:   %pattern_type.ff5: type = pattern_type %Iterate_where.type.a32 [concrete]
// CHECK:STDOUT:   %R.patt: %pattern_type.ff5 = symbolic_binding_pattern R, 0 [symbolic]
// CHECK:STDOUT:   %R: %Iterate_where.type.a32 = symbolic_binding R, 0 [symbolic]
// CHECK:STDOUT:   %R.as_type: type = facet_access_type %R [symbolic]
// CHECK:STDOUT:   %pattern_type.3d7: type = pattern_type %R.as_type [symbolic]
// CHECK:STDOUT:   %r.param_patt.03a: %pattern_type.3d7 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %r.patt.b86: %pattern_type.3d7 = at_binding_pattern r, %r.param_patt.03a [symbolic]
// CHECK:STDOUT:   %i.patt: %pattern_type.fb7 = value_binding_pattern i [concrete]
// CHECK:STDOUT:   %Iterate.lookup_impl_witness.93c: <witness> = lookup_impl_witness %R, @Iterate [symbolic]
// CHECK:STDOUT:   %Iterate.facet.94d: %Iterate.type = facet_value %R.as_type, (%Iterate.lookup_impl_witness.93c) [symbolic]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.4c7: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.94d) [symbolic]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.275: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.94d) [symbolic]
// CHECK:STDOUT:   %.a62: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.4c7, %Iterate.facet.94d [symbolic]
// CHECK:STDOUT:   %impl.elem2.c7b: %.a62 = impl_witness_access %Iterate.lookup_impl_witness.93c, element2 [symbolic]
// CHECK:STDOUT:   %impl.elem1.7a9: %Destroy.type = impl_witness_access %Iterate.lookup_impl_witness.93c, element1 [symbolic]
// CHECK:STDOUT:   %as_type.8e4: type = facet_access_type %impl.elem1.7a9 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.1b8: <specific function> = specific_impl_function %impl.elem2.c7b, @Iterate.WithSelf.NewCursor(%Iterate.facet.94d) [symbolic]
// CHECK:STDOUT:   %ptr.bac: type = ptr_type %as_type.8e4 [symbolic]
// CHECK:STDOUT:   %.a06: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.275, %Iterate.facet.94d [symbolic]
// CHECK:STDOUT:   %impl.elem3.7bd: %.a06 = impl_witness_access %Iterate.lookup_impl_witness.93c, element3 [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.541: type = class_type @DefaultOptionalStorage, @DefaultOptionalStorage(%facet_value.209) [concrete]
// CHECK:STDOUT:   %OptionalStorage.impl_witness.a56: <witness> = impl_witness imports.%OptionalStorage.impl_witness_table.80d, @T.as_type.as.OptionalStorage.impl(%facet_value.209) [concrete]
// CHECK:STDOUT:   %OptionalStorage.facet.63c: %OptionalStorage.type = facet_value %f64.dc1, (%OptionalStorage.impl_witness.a56) [concrete]
// CHECK:STDOUT:   %Optional.e65: type = class_type @Optional, @Optional(%OptionalStorage.facet.63c) [concrete]
// CHECK:STDOUT:   %specific_impl_fn.666: <specific function> = specific_impl_function %impl.elem3.7bd, @Iterate.WithSelf.Next(%Iterate.facet.94d) [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.type.731: type = fn_type @Optional.HasValue, @Optional(%OptionalStorage.facet.63c) [concrete]
// CHECK:STDOUT:   %Optional.HasValue.904: %Optional.HasValue.type.731 = struct_value () [concrete]
// CHECK:STDOUT:   %Optional.Get.type.811: type = fn_type @Optional.Get, @Optional(%OptionalStorage.facet.63c) [concrete]
// CHECK:STDOUT:   %Optional.Get.423: %Optional.Get.type.811 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.facet.339: %Destroy.type = facet_value %f64.dc1, (%custom_witness.df9cc1.3) [concrete]
// CHECK:STDOUT:   %MaybeUnformed.b7e: type = class_type @MaybeUnformed, @MaybeUnformed(%Destroy.facet.339) [concrete]
// CHECK:STDOUT:   %.cc0: type = maybe_unformed_type %f64.dc1 [concrete]
// CHECK:STDOUT:   %struct_type.value.has_value.ad9: type = struct_type {.value: %MaybeUnformed.b7e, .has_value: bool} [concrete]
// CHECK:STDOUT:   %Optional.HasValue.specific_fn.b6b: <specific function> = specific_function %Optional.HasValue.904, @Optional.HasValue(%OptionalStorage.facet.63c) [concrete]
// CHECK:STDOUT:   %Optional.Get.specific_fn.f0c: <specific function> = specific_function %Optional.Get.423, @Optional.Get(%OptionalStorage.facet.63c) [concrete]
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
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.def: <witness> = lookup_impl_witness %impl.elem1.7a9, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.41f: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%impl.elem1.7a9) [symbolic]
// CHECK:STDOUT:   %.148: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.41f, %impl.elem1.7a9 [symbolic]
// CHECK:STDOUT:   %impl.elem0.461: %.148 = impl_witness_access %Destroy.lookup_impl_witness.def, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.095b: <specific function> = specific_impl_function %impl.elem0.461, @Destroy.WithSelf.Op(%impl.elem1.7a9) [symbolic]
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
// CHECK:STDOUT:   %facet_value.cb9: %CppRange_where.type.d682d6.1 = facet_value %MutableRange, (%custom_witness.df9cc1.3, %Copy.impl_witness.873, %custom_witness.902, %custom_witness.c08, %custom_witness.05b, %custom_witness.491, %custom_witness.ed8) [concrete]
// CHECK:STDOUT:   %tuple.type.dd0: type = tuple_type (%Iterator, %Iterator) [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.10: type = fn_type @Destroy.Op.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.10: %Destroy.Op.type.1d8f74.10 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.df9cc1.10: <witness> = custom_witness (%Destroy.Op.1a2547.10), @Destroy [concrete]
// CHECK:STDOUT:   %Destroy.facet.e1e: %Destroy.type = facet_value %tuple.type.dd0, (%custom_witness.df9cc1.10) [concrete]
// CHECK:STDOUT:   %Iterate.impl_witness.4d3: <witness> = impl_witness imports.%Iterate.impl_witness_table.478, @T.as_type.as.Iterate.impl(%facet_value.cb9) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.type.09d: type = fn_type @T.as_type.as.Iterate.impl.NewCursor, @T.as_type.as.Iterate.impl(%facet_value.cb9) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.c5d: %T.as_type.as.Iterate.impl.NewCursor.type.09d = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.type.a4b: type = fn_type @T.as_type.as.Iterate.impl.Next, @T.as_type.as.Iterate.impl(%facet_value.cb9) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.9ad: %T.as_type.as.Iterate.impl.Next.type.a4b = struct_value () [concrete]
// CHECK:STDOUT:   %Iterate.facet.a58: %Iterate.type = facet_value %MutableRange, (%Iterate.impl_witness.4d3) [concrete]
// CHECK:STDOUT:   %facet_value.064: %Iterate_where.type.a32 = facet_value %MutableRange, (%Iterate.impl_witness.4d3) [concrete]
// CHECK:STDOUT:   %r.param_patt.aeb: %pattern_type.992 = value_param_pattern [concrete]
// CHECK:STDOUT:   %r.patt.fe7: %pattern_type.992 = at_binding_pattern r, %r.param_patt.aeb [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.dd9: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.a58) [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.ec7: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.a58) [concrete]
// CHECK:STDOUT:   %.835: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.dd9, %Iterate.facet.a58 [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.specific_fn: <specific function> = specific_function %T.as_type.as.Iterate.impl.NewCursor.c5d, @T.as_type.as.Iterate.impl.NewCursor(%facet_value.cb9) [concrete]
// CHECK:STDOUT:   %ptr.a4f: type = ptr_type %tuple.type.dd0 [concrete]
// CHECK:STDOUT:   %.5d5: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.ec7, %Iterate.facet.a58 [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.specific_fn: <specific function> = specific_function %T.as_type.as.Iterate.impl.Next.9ad, @T.as_type.as.Iterate.impl.Next(%facet_value.cb9) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.452: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.e1e) [concrete]
// CHECK:STDOUT:   %.899: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.452, %Destroy.facet.e1e [concrete]
// CHECK:STDOUT:   %complete_type.843: <witness> = complete_type_witness %tuple.type.dd0 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.bb6: @Optional.%Optional.HasValue.type (%Optional.HasValue.type.e9e) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.HasValue (constants.%Optional.HasValue.6cd)]
// CHECK:STDOUT:   %Core.import_ref.daba: @Optional.%Optional.Get.type (%Optional.Get.type.bb9) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.Get (constants.%Optional.Get.fa1)]
// CHECK:STDOUT:   %Core.import_ref.0c6: type = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%DefaultOptionalStorage (constants.%DefaultOptionalStorage.d50)]
// CHECK:STDOUT:   %Core.import_ref.13b: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.None.type (%T.as_type.as.OptionalStorage.impl.None.type.d4d) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.None (constants.%T.as_type.as.OptionalStorage.impl.None.d64)]
// CHECK:STDOUT:   %Core.import_ref.c43: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Some.type (%T.as_type.as.OptionalStorage.impl.Some.type.7eb) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Some (constants.%T.as_type.as.OptionalStorage.impl.Some.2b4)]
// CHECK:STDOUT:   %Core.import_ref.61e: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Has.type (%T.as_type.as.OptionalStorage.impl.Has.type.eb9) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Has (constants.%T.as_type.as.OptionalStorage.impl.Has.c7e)]
// CHECK:STDOUT:   %Core.import_ref.ce2e: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Get.type (%T.as_type.as.OptionalStorage.impl.Get.type.d3c) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Get (constants.%T.as_type.as.OptionalStorage.impl.Get.da2)]
// CHECK:STDOUT:   %Core.import_ref.8fc = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], unloaded
// CHECK:STDOUT:   %OptionalStorage.impl_witness_table.80d = impl_witness_table (%Core.import_ref.0c6, %Core.import_ref.13b, %Core.import_ref.c43, %Core.import_ref.61e, %Core.import_ref.ce2e, %Core.import_ref.8fc), @T.as_type.as.OptionalStorage.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.eef: %facet_type.f48 = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.Iterate.impl.%facet_value (constants.%facet_value.f93)]
// CHECK:STDOUT:   %Core.import_ref.fcc: %Destroy.type = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.Iterate.impl.%Destroy.facet (constants.%Destroy.facet.9cd)]
// CHECK:STDOUT:   %Core.import_ref.f50: @T.as_type.as.Iterate.impl.%T.as_type.as.Iterate.impl.NewCursor.type (%T.as_type.as.Iterate.impl.NewCursor.type.b5d) = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.Iterate.impl.%T.as_type.as.Iterate.impl.NewCursor (constants.%T.as_type.as.Iterate.impl.NewCursor.21f)]
// CHECK:STDOUT:   %Core.import_ref.590: @T.as_type.as.Iterate.impl.%T.as_type.as.Iterate.impl.Next.type (%T.as_type.as.Iterate.impl.Next.type.5dc) = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.Iterate.impl.%T.as_type.as.Iterate.impl.Next (constants.%T.as_type.as.Iterate.impl.Next.b73)]
// CHECK:STDOUT:   %Iterate.impl_witness_table.478 = impl_witness_table (%Core.import_ref.eef, %Core.import_ref.fcc, %Core.import_ref.f50, %Core.import_ref.590), @T.as_type.as.Iterate.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.e69: @Float.as.Copy.impl.%Float.as.Copy.impl.Op.type (%Float.as.Copy.impl.Op.type.ff3) = import_ref Core//prelude/types/float, loc{{\d+_\d+}}, loaded [symbolic = @Float.as.Copy.impl.%Float.as.Copy.impl.Op (constants.%Float.as.Copy.impl.Op.d93)]
// CHECK:STDOUT:   %Copy.impl_witness_table.4e8 = impl_witness_table (%Core.import_ref.e69), @Float.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.cb3: @Float.as.AddAssignWith.impl.%Float.as.AddAssignWith.impl.Op.type (%Float.as.AddAssignWith.impl.Op.type.fdd) = import_ref Core//prelude/types/float, loc{{\d+_\d+}}, loaded [symbolic = @Float.as.AddAssignWith.impl.%Float.as.AddAssignWith.impl.Op (constants.%Float.as.AddAssignWith.impl.Op.430)]
// CHECK:STDOUT:   %AddAssignWith.impl_witness_table = impl_witness_table (%Core.import_ref.cb3), @Float.as.AddAssignWith.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @TestIterate(%R.loc22_17.2: %Iterate_where.type.a32) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Iterate.lookup_impl_witness: <witness> = lookup_impl_witness %R.loc22_17.1, @Iterate [symbolic = %Iterate.lookup_impl_witness (constants.%Iterate.lookup_impl_witness.93c)]
// CHECK:STDOUT:   %Iterate.facet.loc26_19.5: %Iterate.type = facet_value %R.as_type.loc22_61.1, (%Iterate.lookup_impl_witness) [symbolic = %Iterate.facet.loc26_19.5 (constants.%Iterate.facet.94d)]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.loc26_19.5) [symbolic = %Iterate.WithSelf.NewCursor.type (constants.%Iterate.WithSelf.NewCursor.type.4c7)]
// CHECK:STDOUT:   %.loc26_19.19: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type, %Iterate.facet.loc26_19.5 [symbolic = %.loc26_19.19 (constants.%.a62)]
// CHECK:STDOUT:   %impl.elem2.loc26_19.2: @TestIterate.%.loc26_19.19 (%.a62) = impl_witness_access %Iterate.lookup_impl_witness, element2 [symbolic = %impl.elem2.loc26_19.2 (constants.%impl.elem2.c7b)]
// CHECK:STDOUT:   %specific_impl_fn.loc26_19.4: <specific function> = specific_impl_function %impl.elem2.loc26_19.2, @Iterate.WithSelf.NewCursor(%Iterate.facet.loc26_19.5) [symbolic = %specific_impl_fn.loc26_19.4 (constants.%specific_impl_fn.1b8)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %ptr: type = ptr_type %as_type [symbolic = %ptr (constants.%ptr.bac)]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.loc26_19.5) [symbolic = %Iterate.WithSelf.Next.type (constants.%Iterate.WithSelf.Next.type.275)]
// CHECK:STDOUT:   %.loc26_19.20: type = fn_type_with_self_type %Iterate.WithSelf.Next.type, %Iterate.facet.loc26_19.5 [symbolic = %.loc26_19.20 (constants.%.a06)]
// CHECK:STDOUT:   %impl.elem3.loc26_19.2: @TestIterate.%.loc26_19.20 (%.a06) = impl_witness_access %Iterate.lookup_impl_witness, element3 [symbolic = %impl.elem3.loc26_19.2 (constants.%impl.elem3.7bd)]
// CHECK:STDOUT:   %specific_impl_fn.loc26_19.5: <specific function> = specific_impl_function %impl.elem3.loc26_19.2, @Iterate.WithSelf.Next(%Iterate.facet.loc26_19.5) [symbolic = %specific_impl_fn.loc26_19.5 (constants.%specific_impl_fn.666)]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%impl.elem1) [symbolic = %Destroy.WithSelf.Op.type (constants.%Destroy.WithSelf.Op.type.41f)]
// CHECK:STDOUT:   %.loc26_19.21: type = fn_type_with_self_type %Destroy.WithSelf.Op.type, %impl.elem1 [symbolic = %.loc26_19.21 (constants.%.148)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %impl.elem1, @Destroy [symbolic = %Destroy.lookup_impl_witness (constants.%Destroy.lookup_impl_witness.def)]
// CHECK:STDOUT:   %impl.elem0.loc26_19.2: @TestIterate.%.loc26_19.21 (%.148) = impl_witness_access %Destroy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc26_19.2 (constants.%impl.elem0.461)]
// CHECK:STDOUT:   %specific_impl_fn.loc26_19.6: <specific function> = specific_impl_function %impl.elem0.loc26_19.2, @Destroy.WithSelf.Op(%impl.elem1) [symbolic = %specific_impl_fn.loc26_19.6 (constants.%specific_impl_fn.095b)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%r.param: @TestIterate.%R.as_type.loc22_61.1 (%R.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %i.patt: %pattern_type.fb7 = value_binding_pattern i [concrete = constants.%i.patt]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %r.ref: @TestIterate.%R.as_type.loc22_61.1 (%R.as_type) = name_ref r, %r
// CHECK:STDOUT:     %impl.elem2.loc26_19.1: @TestIterate.%.loc26_19.19 (%.a62) = impl_witness_access constants.%Iterate.lookup_impl_witness.93c, element2 [symbolic = %impl.elem2.loc26_19.2 (constants.%impl.elem2.c7b)]
// CHECK:STDOUT:     %bound_method.loc26_19.1: <bound method> = bound_method %r.ref, %impl.elem2.loc26_19.1
// CHECK:STDOUT:     %Iterate.facet.loc26_19.1: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.93c) [symbolic = %Iterate.facet.loc26_19.5 (constants.%Iterate.facet.94d)]
// CHECK:STDOUT:     %.loc26_19.1: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc26_19.1 [symbolic = %Iterate.facet.loc26_19.5 (constants.%Iterate.facet.94d)]
// CHECK:STDOUT:     %Iterate.facet.loc26_19.2: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.93c) [symbolic = %Iterate.facet.loc26_19.5 (constants.%Iterate.facet.94d)]
// CHECK:STDOUT:     %.loc26_19.2: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc26_19.2 [symbolic = %Iterate.facet.loc26_19.5 (constants.%Iterate.facet.94d)]
// CHECK:STDOUT:     %specific_impl_fn.loc26_19.1: <specific function> = specific_impl_function %impl.elem2.loc26_19.1, @Iterate.WithSelf.NewCursor(constants.%Iterate.facet.94d) [symbolic = %specific_impl_fn.loc26_19.4 (constants.%specific_impl_fn.1b8)]
// CHECK:STDOUT:     %bound_method.loc26_19.2: <bound method> = bound_method %r.ref, %specific_impl_fn.loc26_19.1
// CHECK:STDOUT:     %var: ref @TestIterate.%as_type (%as_type.8e4) = var_storage invalid
// CHECK:STDOUT:     %Iterate.WithSelf.NewCursor.call: init @TestIterate.%as_type (%as_type.8e4) to %var = call %bound_method.loc26_19.2(%r.ref)
// CHECK:STDOUT:     assign %var, %Iterate.WithSelf.NewCursor.call
// CHECK:STDOUT:     br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.next:
// CHECK:STDOUT:     %addr: @TestIterate.%ptr (%ptr.bac) = addr_of %var
// CHECK:STDOUT:     %impl.elem3.loc26_19.1: @TestIterate.%.loc26_19.20 (%.a06) = impl_witness_access constants.%Iterate.lookup_impl_witness.93c, element3 [symbolic = %impl.elem3.loc26_19.2 (constants.%impl.elem3.7bd)]
// CHECK:STDOUT:     %bound_method.loc26_19.3: <bound method> = bound_method %r.ref, %impl.elem3.loc26_19.1
// CHECK:STDOUT:     %Iterate.facet.loc26_19.3: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.93c) [symbolic = %Iterate.facet.loc26_19.5 (constants.%Iterate.facet.94d)]
// CHECK:STDOUT:     %.loc26_19.3: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc26_19.3 [symbolic = %Iterate.facet.loc26_19.5 (constants.%Iterate.facet.94d)]
// CHECK:STDOUT:     %Iterate.facet.loc26_19.4: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.93c) [symbolic = %Iterate.facet.loc26_19.5 (constants.%Iterate.facet.94d)]
// CHECK:STDOUT:     %.loc26_19.4: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc26_19.4 [symbolic = %Iterate.facet.loc26_19.5 (constants.%Iterate.facet.94d)]
// CHECK:STDOUT:     %.loc26_19.5: %Destroy.type = converted constants.%as_type.8e4, constants.%impl.elem1.7a9 [symbolic = %impl.elem1 (constants.%impl.elem1.7a9)]
// CHECK:STDOUT:     %.loc26_19.6: %Destroy.type = converted constants.%as_type.8e4, constants.%impl.elem1.7a9 [symbolic = %impl.elem1 (constants.%impl.elem1.7a9)]
// CHECK:STDOUT:     %specific_impl_fn.loc26_19.2: <specific function> = specific_impl_function %impl.elem3.loc26_19.1, @Iterate.WithSelf.Next(constants.%Iterate.facet.94d) [symbolic = %specific_impl_fn.loc26_19.5 (constants.%specific_impl_fn.666)]
// CHECK:STDOUT:     %bound_method.loc26_19.4: <bound method> = bound_method %r.ref, %specific_impl_fn.loc26_19.2
// CHECK:STDOUT:     %.loc26_19.7: ref %Optional.e65 = temporary_storage
// CHECK:STDOUT:     %Iterate.WithSelf.Next.call: init %Optional.e65 to %.loc26_19.7 = call %bound_method.loc26_19.4(%r.ref, %addr)
// CHECK:STDOUT:     %.loc26_19.8: ref %Optional.e65 = temporary %.loc26_19.7, %Iterate.WithSelf.Next.call
// CHECK:STDOUT:     %.loc26_19.9: %Optional.HasValue.type.731 = specific_constant imports.%Core.import_ref.bb6, @Optional(constants.%OptionalStorage.facet.63c) [concrete = constants.%Optional.HasValue.904]
// CHECK:STDOUT:     %HasValue.ref: %Optional.HasValue.type.731 = name_ref HasValue, %.loc26_19.9 [concrete = constants.%Optional.HasValue.904]
// CHECK:STDOUT:     %Optional.HasValue.bound: <bound method> = bound_method %.loc26_19.8, %HasValue.ref
// CHECK:STDOUT:     %Optional.HasValue.specific_fn: <specific function> = specific_function %HasValue.ref, @Optional.HasValue(constants.%OptionalStorage.facet.63c) [concrete = constants.%Optional.HasValue.specific_fn.b6b]
// CHECK:STDOUT:     %bound_method.loc26_19.5: <bound method> = bound_method %.loc26_19.8, %Optional.HasValue.specific_fn
// CHECK:STDOUT:     %.loc26_19.10: %Optional.e65 = acquire_value %.loc26_19.8
// CHECK:STDOUT:     %Optional.HasValue.call: init bool = call %bound_method.loc26_19.5(%.loc26_19.10)
// CHECK:STDOUT:     %.loc26_19.11: bool = value_of_initializer %Optional.HasValue.call
// CHECK:STDOUT:     %.loc26_19.12: bool = converted %Optional.HasValue.call, %.loc26_19.11
// CHECK:STDOUT:     if %.loc26_19.12 br !for.body else br !for.done
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.body:
// CHECK:STDOUT:     %.loc26_19.13: %Optional.Get.type.811 = specific_constant imports.%Core.import_ref.daba, @Optional(constants.%OptionalStorage.facet.63c) [concrete = constants.%Optional.Get.423]
// CHECK:STDOUT:     %Get.ref: %Optional.Get.type.811 = name_ref Get, %.loc26_19.13 [concrete = constants.%Optional.Get.423]
// CHECK:STDOUT:     %Optional.Get.bound: <bound method> = bound_method %.loc26_19.8, %Get.ref
// CHECK:STDOUT:     %Optional.Get.specific_fn: <specific function> = specific_function %Get.ref, @Optional.Get(constants.%OptionalStorage.facet.63c) [concrete = constants.%Optional.Get.specific_fn.f0c]
// CHECK:STDOUT:     %bound_method.loc26_19.6: <bound method> = bound_method %.loc26_19.8, %Optional.Get.specific_fn
// CHECK:STDOUT:     %.loc26_19.14: %Optional.e65 = acquire_value %.loc26_19.8
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
// CHECK:STDOUT:     %Destroy.Op.bound.loc26_19.1: <bound method> = bound_method %.loc26_19.8, constants.%Destroy.Op.1a2547.9
// CHECK:STDOUT:     %Destroy.Op.call.loc26_19.1: init %empty_tuple.type = call %Destroy.Op.bound.loc26_19.1(%.loc26_19.8)
// CHECK:STDOUT:     br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.done:
// CHECK:STDOUT:     %Destroy.Op.bound.loc26_19.2: <bound method> = bound_method %.loc26_19.8, constants.%Destroy.Op.1a2547.9
// CHECK:STDOUT:     %Destroy.Op.call.loc26_19.2: init %empty_tuple.type = call %Destroy.Op.bound.loc26_19.2(%.loc26_19.8)
// CHECK:STDOUT:     %impl.elem0.loc26_19.1: @TestIterate.%.loc26_19.21 (%.148) = impl_witness_access constants.%Destroy.lookup_impl_witness.def, element0 [symbolic = %impl.elem0.loc26_19.2 (constants.%impl.elem0.461)]
// CHECK:STDOUT:     %bound_method.loc26_19.7: <bound method> = bound_method %var, %impl.elem0.loc26_19.1
// CHECK:STDOUT:     %.loc26_19.17: %Destroy.type = converted constants.%as_type.8e4, constants.%impl.elem1.7a9 [symbolic = %impl.elem1 (constants.%impl.elem1.7a9)]
// CHECK:STDOUT:     %.loc26_19.18: %Destroy.type = converted constants.%as_type.8e4, constants.%impl.elem1.7a9 [symbolic = %impl.elem1 (constants.%impl.elem1.7a9)]
// CHECK:STDOUT:     %specific_impl_fn.loc26_19.3: <specific function> = specific_impl_function %impl.elem0.loc26_19.1, @Destroy.WithSelf.Op(constants.%impl.elem1.7a9) [symbolic = %specific_impl_fn.loc26_19.6 (constants.%specific_impl_fn.095b)]
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
// CHECK:STDOUT: fn @Destroy.Op.loc26_19.6(%self.param: ref %Optional.e65) {
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
// CHECK:STDOUT:   %impl.elem2: %.835 = impl_witness_access constants.%Iterate.impl_witness.4d3, element2 [concrete = constants.%T.as_type.as.Iterate.impl.NewCursor.c5d]
// CHECK:STDOUT:   %bound_method.loc68_19.1: <bound method> = bound_method %m.ref, %impl.elem2
// CHECK:STDOUT:   %facet_value.loc68_19.1: %CppRange_where.type.d682d6.1 = facet_value constants.%MutableRange, (constants.%custom_witness.df9cc1.3, constants.%Copy.impl_witness.873, constants.%custom_witness.902, constants.%custom_witness.c08, constants.%custom_witness.05b, constants.%custom_witness.491, constants.%custom_witness.ed8) [concrete = constants.%facet_value.cb9]
// CHECK:STDOUT:   %.loc68_19.1: %CppRange_where.type.d682d6.1 = converted constants.%MutableRange, %facet_value.loc68_19.1 [concrete = constants.%facet_value.cb9]
// CHECK:STDOUT:   %facet_value.loc68_19.2: %CppRange_where.type.d682d6.1 = facet_value constants.%MutableRange, (constants.%custom_witness.df9cc1.3, constants.%Copy.impl_witness.873, constants.%custom_witness.902, constants.%custom_witness.c08, constants.%custom_witness.05b, constants.%custom_witness.491, constants.%custom_witness.ed8) [concrete = constants.%facet_value.cb9]
// CHECK:STDOUT:   %.loc68_19.2: %CppRange_where.type.d682d6.1 = converted constants.%MutableRange, %facet_value.loc68_19.2 [concrete = constants.%facet_value.cb9]
// CHECK:STDOUT:   %specific_fn.loc68_19.1: <specific function> = specific_function %impl.elem2, @T.as_type.as.Iterate.impl.NewCursor(constants.%facet_value.cb9) [concrete = constants.%T.as_type.as.Iterate.impl.NewCursor.specific_fn]
// CHECK:STDOUT:   %bound_method.loc68_19.2: <bound method> = bound_method %m.ref, %specific_fn.loc68_19.1
// CHECK:STDOUT:   %var: ref %tuple.type.dd0 = var_storage invalid
// CHECK:STDOUT:   %.loc68_18.1: %MutableRange = acquire_value %m.ref
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.call: init %tuple.type.dd0 to %var = call %bound_method.loc68_19.2(%.loc68_18.1)
// CHECK:STDOUT:   assign %var, %T.as_type.as.Iterate.impl.NewCursor.call
// CHECK:STDOUT:   br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT: !for.next:
// CHECK:STDOUT:   %addr: %ptr.a4f = addr_of %var
// CHECK:STDOUT:   %impl.elem3: %.5d5 = impl_witness_access constants.%Iterate.impl_witness.4d3, element3 [concrete = constants.%T.as_type.as.Iterate.impl.Next.9ad]
// CHECK:STDOUT:   %bound_method.loc68_19.3: <bound method> = bound_method %m.ref, %impl.elem3
// CHECK:STDOUT:   %facet_value.loc68_19.3: %CppRange_where.type.d682d6.1 = facet_value constants.%MutableRange, (constants.%custom_witness.df9cc1.3, constants.%Copy.impl_witness.873, constants.%custom_witness.902, constants.%custom_witness.c08, constants.%custom_witness.05b, constants.%custom_witness.491, constants.%custom_witness.ed8) [concrete = constants.%facet_value.cb9]
// CHECK:STDOUT:   %.loc68_19.3: %CppRange_where.type.d682d6.1 = converted constants.%MutableRange, %facet_value.loc68_19.3 [concrete = constants.%facet_value.cb9]
// CHECK:STDOUT:   %facet_value.loc68_19.4: %CppRange_where.type.d682d6.1 = facet_value constants.%MutableRange, (constants.%custom_witness.df9cc1.3, constants.%Copy.impl_witness.873, constants.%custom_witness.902, constants.%custom_witness.c08, constants.%custom_witness.05b, constants.%custom_witness.491, constants.%custom_witness.ed8) [concrete = constants.%facet_value.cb9]
// CHECK:STDOUT:   %.loc68_19.4: %CppRange_where.type.d682d6.1 = converted constants.%MutableRange, %facet_value.loc68_19.4 [concrete = constants.%facet_value.cb9]
// CHECK:STDOUT:   %specific_fn.loc68_19.2: <specific function> = specific_function %impl.elem3, @T.as_type.as.Iterate.impl.Next(constants.%facet_value.cb9) [concrete = constants.%T.as_type.as.Iterate.impl.Next.specific_fn]
// CHECK:STDOUT:   %bound_method.loc68_19.4: <bound method> = bound_method %m.ref, %specific_fn.loc68_19.2
// CHECK:STDOUT:   %.loc68_19.5: ref %Optional.e65 = temporary_storage
// CHECK:STDOUT:   %.loc68_18.2: %MutableRange = acquire_value %m.ref
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.call: init %Optional.e65 to %.loc68_19.5 = call %bound_method.loc68_19.4(%.loc68_18.2, %addr)
// CHECK:STDOUT:   %.loc68_19.6: ref %Optional.e65 = temporary %.loc68_19.5, %T.as_type.as.Iterate.impl.Next.call
// CHECK:STDOUT:   %.loc68_19.7: %Optional.HasValue.type.731 = specific_constant imports.%Core.import_ref.bb6, @Optional(constants.%OptionalStorage.facet.63c) [concrete = constants.%Optional.HasValue.904]
// CHECK:STDOUT:   %HasValue.ref: %Optional.HasValue.type.731 = name_ref HasValue, %.loc68_19.7 [concrete = constants.%Optional.HasValue.904]
// CHECK:STDOUT:   %Optional.HasValue.bound: <bound method> = bound_method %.loc68_19.6, %HasValue.ref
// CHECK:STDOUT:   %Optional.HasValue.specific_fn: <specific function> = specific_function %HasValue.ref, @Optional.HasValue(constants.%OptionalStorage.facet.63c) [concrete = constants.%Optional.HasValue.specific_fn.b6b]
// CHECK:STDOUT:   %bound_method.loc68_19.5: <bound method> = bound_method %.loc68_19.6, %Optional.HasValue.specific_fn
// CHECK:STDOUT:   %.loc68_19.8: %Optional.e65 = acquire_value %.loc68_19.6
// CHECK:STDOUT:   %Optional.HasValue.call: init bool = call %bound_method.loc68_19.5(%.loc68_19.8)
// CHECK:STDOUT:   %.loc68_19.9: bool = value_of_initializer %Optional.HasValue.call
// CHECK:STDOUT:   %.loc68_19.10: bool = converted %Optional.HasValue.call, %.loc68_19.9
// CHECK:STDOUT:   if %.loc68_19.10 br !for.body else br !for.done
// CHECK:STDOUT:
// CHECK:STDOUT: !for.body:
// CHECK:STDOUT:   %.loc68_19.11: %Optional.Get.type.811 = specific_constant imports.%Core.import_ref.daba, @Optional(constants.%OptionalStorage.facet.63c) [concrete = constants.%Optional.Get.423]
// CHECK:STDOUT:   %Get.ref: %Optional.Get.type.811 = name_ref Get, %.loc68_19.11 [concrete = constants.%Optional.Get.423]
// CHECK:STDOUT:   %Optional.Get.bound: <bound method> = bound_method %.loc68_19.6, %Get.ref
// CHECK:STDOUT:   %Optional.Get.specific_fn: <specific function> = specific_function %Get.ref, @Optional.Get(constants.%OptionalStorage.facet.63c) [concrete = constants.%Optional.Get.specific_fn.f0c]
// CHECK:STDOUT:   %bound_method.loc68_19.6: <bound method> = bound_method %.loc68_19.6, %Optional.Get.specific_fn
// CHECK:STDOUT:   %.loc68_19.12: %Optional.e65 = acquire_value %.loc68_19.6
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
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.3d7
// CHECK:STDOUT:   %r.param_patt.loc22_59.2 => constants.%r.param_patt.03a
// CHECK:STDOUT:   %r.patt.loc22_59.2 => constants.%r.patt.b86
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestIterate(constants.%facet_value.064) {
// CHECK:STDOUT:   %R.patt.loc22_17.2 => constants.%R.patt
// CHECK:STDOUT:   %R.loc22_17.1 => constants.%facet_value.064
// CHECK:STDOUT:   %R.as_type.loc22_61.1 => constants.%MutableRange
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.992
// CHECK:STDOUT:   %r.param_patt.loc22_59.2 => constants.%r.param_patt.aeb
// CHECK:STDOUT:   %r.patt.loc22_59.2 => constants.%r.patt.fe7
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc22 => constants.%complete_type.357
// CHECK:STDOUT:   %Iterate.lookup_impl_witness => constants.%Iterate.impl_witness.4d3
// CHECK:STDOUT:   %Iterate.facet.loc26_19.5 => constants.%Iterate.facet.a58
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type => constants.%Iterate.WithSelf.NewCursor.type.dd9
// CHECK:STDOUT:   %.loc26_19.19 => constants.%.835
// CHECK:STDOUT:   %impl.elem2.loc26_19.2 => constants.%T.as_type.as.Iterate.impl.NewCursor.c5d
// CHECK:STDOUT:   %specific_impl_fn.loc26_19.4 => constants.%T.as_type.as.Iterate.impl.NewCursor.specific_fn
// CHECK:STDOUT:   %impl.elem1 => constants.%Destroy.facet.e1e
// CHECK:STDOUT:   %as_type => constants.%tuple.type.dd0
// CHECK:STDOUT:   %require_complete.1 => constants.%complete_type.843
// CHECK:STDOUT:   %ptr => constants.%ptr.a4f
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type => constants.%Iterate.WithSelf.Next.type.ec7
// CHECK:STDOUT:   %.loc26_19.20 => constants.%.5d5
// CHECK:STDOUT:   %impl.elem3.loc26_19.2 => constants.%T.as_type.as.Iterate.impl.Next.9ad
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
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Get.type.d3c: type = fn_type @T.as_type.as.OptionalStorage.impl.Get, @T.as_type.as.OptionalStorage.impl(%T.283) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Get.da2: %T.as_type.as.OptionalStorage.impl.Get.type.d3c = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Has.type.eb9: type = fn_type @T.as_type.as.OptionalStorage.impl.Has, @T.as_type.as.OptionalStorage.impl(%T.283) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Has.c7e: %T.as_type.as.OptionalStorage.impl.Has.type.eb9 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Some.type.7eb: type = fn_type @T.as_type.as.OptionalStorage.impl.Some, @T.as_type.as.OptionalStorage.impl(%T.283) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Some.2b4: %T.as_type.as.OptionalStorage.impl.Some.type.7eb = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.None.type.d4d: type = fn_type @T.as_type.as.OptionalStorage.impl.None, @T.as_type.as.OptionalStorage.impl(%T.283) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.None.d64: %T.as_type.as.OptionalStorage.impl.None.type.d4d = struct_value () [symbolic]
// CHECK:STDOUT:   %.Self.500: %Iterate.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %CppRange.type: type = facet_type <@CppRange> [concrete]
// CHECK:STDOUT:   %.Self.e0b: %CppRange.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %CppRangeForIterate.lookup_impl_witness.50b: <witness> = lookup_impl_witness %.Self.e0b, @CppRangeForIterate [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.32d: type = impl_witness_access %CppRangeForIterate.lookup_impl_witness.50b, element0 [symbolic_self]
// CHECK:STDOUT:   %CppUnsafeDeref.lookup_impl_witness.851: <witness> = lookup_impl_witness %impl.elem0.32d, @CppUnsafeDeref [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.10d: type = impl_witness_access %CppUnsafeDeref.lookup_impl_witness.851, element0 [symbolic_self]
// CHECK:STDOUT:   %CppRange_where.type.d682d6.1: type = facet_type <@CppRange where %impl.elem0.10d impls @Copy> [concrete]
// CHECK:STDOUT:   %T.f45530.2: %CppRange_where.type.d682d6.1 = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %CppRangeForIterate.lookup_impl_witness.b581a8.2: <witness> = lookup_impl_witness %T.f45530.2, @CppRangeForIterate [symbolic]
// CHECK:STDOUT:   %impl.elem1.c2dc15.2: type = impl_witness_access %CppRangeForIterate.lookup_impl_witness.b581a8.2, element1 [symbolic]
// CHECK:STDOUT:   %impl.elem0.5f971c.2: type = impl_witness_access %CppRangeForIterate.lookup_impl_witness.b581a8.2, element0 [symbolic]
// CHECK:STDOUT:   %tuple.type.1ba: type = tuple_type (%impl.elem0.5f971c.2, %impl.elem1.c2dc15.2) [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.812: <witness> = lookup_impl_witness %tuple.type.1ba, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.9cd: %Destroy.type = facet_value %tuple.type.1ba, (%Destroy.lookup_impl_witness.812) [symbolic]
// CHECK:STDOUT:   %CppUnsafeDeref.lookup_impl_witness.87e543.2: <witness> = lookup_impl_witness %impl.elem0.5f971c.2, @CppUnsafeDeref [symbolic]
// CHECK:STDOUT:   %impl.elem0.493765.2: type = impl_witness_access %CppUnsafeDeref.lookup_impl_witness.87e543.2, element0 [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.35816c.2: <witness> = lookup_impl_witness %impl.elem0.493765.2, @Destroy [symbolic]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.167: <witness> = lookup_impl_witness %impl.elem0.493765.2, @Copy [symbolic]
// CHECK:STDOUT:   %facet_value.f93: %facet_type.f48 = facet_value %impl.elem0.493765.2, (%Destroy.lookup_impl_witness.35816c.2, %Copy.lookup_impl_witness.167) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.type.5dc: type = fn_type @T.as_type.as.Iterate.impl.Next, @T.as_type.as.Iterate.impl(%T.f45530.2) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.b73: %T.as_type.as.Iterate.impl.Next.type.5dc = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.type.b5d: type = fn_type @T.as_type.as.Iterate.impl.NewCursor, @T.as_type.as.Iterate.impl(%T.f45530.2) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.21f: %T.as_type.as.Iterate.impl.NewCursor.type.b5d = struct_value () [symbolic]
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
// CHECK:STDOUT:   %facet_value.ccb: %facet_type.f48 = facet_value %ValueType, (%custom_witness.fef, %custom_witness.c19) [concrete]
// CHECK:STDOUT:   %Iterate.lookup_impl_witness.1d8: <witness> = lookup_impl_witness %.Self.500, @Iterate [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.294: %facet_type.f48 = impl_witness_access %Iterate.lookup_impl_witness.1d8, element0 [symbolic_self]
// CHECK:STDOUT:   %Iterate_where.type.f9d: type = facet_type <@Iterate where %impl.elem0.294 = %facet_value.ccb> [concrete]
// CHECK:STDOUT:   %pattern_type.c7b: type = pattern_type %Iterate_where.type.f9d [concrete]
// CHECK:STDOUT:   %R.patt: %pattern_type.c7b = symbolic_binding_pattern R, 0 [symbolic]
// CHECK:STDOUT:   %R: %Iterate_where.type.f9d = symbolic_binding R, 0 [symbolic]
// CHECK:STDOUT:   %R.as_type: type = facet_access_type %R [symbolic]
// CHECK:STDOUT:   %pattern_type.17d: type = pattern_type %R.as_type [symbolic]
// CHECK:STDOUT:   %r.param_patt.90c: %pattern_type.17d = value_param_pattern [symbolic]
// CHECK:STDOUT:   %r.patt.d6c: %pattern_type.17d = at_binding_pattern r, %r.param_patt.90c [symbolic]
// CHECK:STDOUT:   %i.patt: %pattern_type.e41 = value_binding_pattern i [concrete]
// CHECK:STDOUT:   %Iterate.lookup_impl_witness.173: <witness> = lookup_impl_witness %R, @Iterate [symbolic]
// CHECK:STDOUT:   %Iterate.facet.b38: %Iterate.type = facet_value %R.as_type, (%Iterate.lookup_impl_witness.173) [symbolic]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.5cd: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.b38) [symbolic]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.d51: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.b38) [symbolic]
// CHECK:STDOUT:   %.592: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.5cd, %Iterate.facet.b38 [symbolic]
// CHECK:STDOUT:   %impl.elem2.ae2: %.592 = impl_witness_access %Iterate.lookup_impl_witness.173, element2 [symbolic]
// CHECK:STDOUT:   %impl.elem1.834: %Destroy.type = impl_witness_access %Iterate.lookup_impl_witness.173, element1 [symbolic]
// CHECK:STDOUT:   %as_type.e41: type = facet_access_type %impl.elem1.834 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.06e: <specific function> = specific_impl_function %impl.elem2.ae2, @Iterate.WithSelf.NewCursor(%Iterate.facet.b38) [symbolic]
// CHECK:STDOUT:   %ptr.d27: type = ptr_type %as_type.e41 [symbolic]
// CHECK:STDOUT:   %.524: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.d51, %Iterate.facet.b38 [symbolic]
// CHECK:STDOUT:   %impl.elem3.7a5: %.524 = impl_witness_access %Iterate.lookup_impl_witness.173, element3 [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.257: type = class_type @DefaultOptionalStorage, @DefaultOptionalStorage(%facet_value.ccb) [concrete]
// CHECK:STDOUT:   %OptionalStorage.impl_witness.cf8: <witness> = impl_witness imports.%OptionalStorage.impl_witness_table.80d, @T.as_type.as.OptionalStorage.impl(%facet_value.ccb) [concrete]
// CHECK:STDOUT:   %OptionalStorage.facet.381: %OptionalStorage.type = facet_value %ValueType, (%OptionalStorage.impl_witness.cf8) [concrete]
// CHECK:STDOUT:   %Optional.dda: type = class_type @Optional, @Optional(%OptionalStorage.facet.381) [concrete]
// CHECK:STDOUT:   %specific_impl_fn.24b: <specific function> = specific_impl_function %impl.elem3.7a5, @Iterate.WithSelf.Next(%Iterate.facet.b38) [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.type.b33: type = fn_type @Optional.HasValue, @Optional(%OptionalStorage.facet.381) [concrete]
// CHECK:STDOUT:   %Optional.HasValue.1a5: %Optional.HasValue.type.b33 = struct_value () [concrete]
// CHECK:STDOUT:   %Optional.Get.type.341: type = fn_type @Optional.Get, @Optional(%OptionalStorage.facet.381) [concrete]
// CHECK:STDOUT:   %Optional.Get.733: %Optional.Get.type.341 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.facet.306: %Destroy.type = facet_value %ValueType, (%custom_witness.fef) [concrete]
// CHECK:STDOUT:   %MaybeUnformed.669: type = class_type @MaybeUnformed, @MaybeUnformed(%Destroy.facet.306) [concrete]
// CHECK:STDOUT:   %.b36: type = maybe_unformed_type %ValueType [concrete]
// CHECK:STDOUT:   %struct_type.value.has_value.5a1: type = struct_type {.value: %MaybeUnformed.669, .has_value: bool} [concrete]
// CHECK:STDOUT:   %Optional.HasValue.specific_fn.39a: <specific function> = specific_function %Optional.HasValue.1a5, @Optional.HasValue(%OptionalStorage.facet.381) [concrete]
// CHECK:STDOUT:   %Optional.Get.specific_fn.cd3: <specific function> = specific_function %Optional.Get.733, @Optional.Get(%OptionalStorage.facet.381) [concrete]
// CHECK:STDOUT:   %operator_PlusEqual__carbon_thunk.type: type = fn_type @operator_PlusEqual__carbon_thunk [concrete]
// CHECK:STDOUT:   %operator_PlusEqual__carbon_thunk: %operator_PlusEqual__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.7: type = fn_type @Destroy.Op.loc31_31.6 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.7: %Destroy.Op.type.1d8f74.7 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.198: <witness> = lookup_impl_witness %impl.elem1.834, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.9d6: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%impl.elem1.834) [symbolic]
// CHECK:STDOUT:   %.57c: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.9d6, %impl.elem1.834 [symbolic]
// CHECK:STDOUT:   %impl.elem0.152: %.57c = impl_witness_access %Destroy.lookup_impl_witness.198, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.9fb: <specific function> = specific_impl_function %impl.elem0.152, @Destroy.WithSelf.Op(%impl.elem1.834) [symbolic]
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
// CHECK:STDOUT:   %custom_witness.393: <witness> = custom_witness (%Iterator.Op.583fda.3), @Inc [concrete]
// CHECK:STDOUT:   %Equal.type: type = fn_type @Equal [concrete]
// CHECK:STDOUT:   %Equal: %Equal.type = struct_value () [concrete]
// CHECK:STDOUT:   %NotEqual.type: type = fn_type @NotEqual [concrete]
// CHECK:STDOUT:   %NotEqual: %NotEqual.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.aa4: <witness> = custom_witness (%Equal, %NotEqual), @EqWith, @EqWith(%Sentinel) [concrete]
// CHECK:STDOUT:   %Sentinel.Op.type: type = fn_type @Sentinel.Op [concrete]
// CHECK:STDOUT:   %Sentinel.Op: %Sentinel.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.21358e.2: <witness> = custom_witness (%Sentinel.Op), @Destroy [concrete]
// CHECK:STDOUT:   %facet_value.2b3: %CppRange_where.type.d682d6.1 = facet_value %ConstRange, (%custom_witness.fef, %custom_witness.c19, %custom_witness.c2f, %custom_witness.21358e.1, %custom_witness.1d3, %custom_witness.393, %custom_witness.aa4, %custom_witness.21358e.2) [concrete]
// CHECK:STDOUT:   %tuple.type.df8: type = tuple_type (%Iterator, %Sentinel) [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.8: type = fn_type @Destroy.Op.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.8: %Destroy.Op.type.1d8f74.8 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.df9cc1.8: <witness> = custom_witness (%Destroy.Op.1a2547.8), @Destroy [concrete]
// CHECK:STDOUT:   %Destroy.facet.777: %Destroy.type = facet_value %tuple.type.df8, (%custom_witness.df9cc1.8) [concrete]
// CHECK:STDOUT:   %Iterate.impl_witness.059: <witness> = impl_witness imports.%Iterate.impl_witness_table.478, @T.as_type.as.Iterate.impl(%facet_value.2b3) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.type.7b6: type = fn_type @T.as_type.as.Iterate.impl.NewCursor, @T.as_type.as.Iterate.impl(%facet_value.2b3) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.f2f: %T.as_type.as.Iterate.impl.NewCursor.type.7b6 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.type.da5: type = fn_type @T.as_type.as.Iterate.impl.Next, @T.as_type.as.Iterate.impl(%facet_value.2b3) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.47a: %T.as_type.as.Iterate.impl.Next.type.da5 = struct_value () [concrete]
// CHECK:STDOUT:   %Iterate.facet.11a: %Iterate.type = facet_value %ConstRange, (%Iterate.impl_witness.059) [concrete]
// CHECK:STDOUT:   %facet_value.883: %Iterate_where.type.f9d = facet_value %ConstRange, (%Iterate.impl_witness.059) [concrete]
// CHECK:STDOUT:   %r.param_patt.a73: %pattern_type.025 = value_param_pattern [concrete]
// CHECK:STDOUT:   %r.patt.0e2: %pattern_type.025 = at_binding_pattern r, %r.param_patt.a73 [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.f75: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.11a) [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.c96: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.11a) [concrete]
// CHECK:STDOUT:   %.c99: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.f75, %Iterate.facet.11a [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.specific_fn: <specific function> = specific_function %T.as_type.as.Iterate.impl.NewCursor.f2f, @T.as_type.as.Iterate.impl.NewCursor(%facet_value.2b3) [concrete]
// CHECK:STDOUT:   %ptr.fc0: type = ptr_type %tuple.type.df8 [concrete]
// CHECK:STDOUT:   %.858: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.c96, %Iterate.facet.11a [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.specific_fn: <specific function> = specific_function %T.as_type.as.Iterate.impl.Next.47a, @T.as_type.as.Iterate.impl.Next(%facet_value.2b3) [concrete]
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
// CHECK:STDOUT:   %Core.import_ref.13b: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.None.type (%T.as_type.as.OptionalStorage.impl.None.type.d4d) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.None (constants.%T.as_type.as.OptionalStorage.impl.None.d64)]
// CHECK:STDOUT:   %Core.import_ref.c43f: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Some.type (%T.as_type.as.OptionalStorage.impl.Some.type.7eb) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Some (constants.%T.as_type.as.OptionalStorage.impl.Some.2b4)]
// CHECK:STDOUT:   %Core.import_ref.61e: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Has.type (%T.as_type.as.OptionalStorage.impl.Has.type.eb9) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Has (constants.%T.as_type.as.OptionalStorage.impl.Has.c7e)]
// CHECK:STDOUT:   %Core.import_ref.ce2e: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Get.type (%T.as_type.as.OptionalStorage.impl.Get.type.d3c) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Get (constants.%T.as_type.as.OptionalStorage.impl.Get.da2)]
// CHECK:STDOUT:   %Core.import_ref.8fc = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], unloaded
// CHECK:STDOUT:   %OptionalStorage.impl_witness_table.80d = impl_witness_table (%Core.import_ref.0c6, %Core.import_ref.13b, %Core.import_ref.c43f, %Core.import_ref.61e, %Core.import_ref.ce2e, %Core.import_ref.8fc), @T.as_type.as.OptionalStorage.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.eef: %facet_type.f48 = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.Iterate.impl.%facet_value (constants.%facet_value.f93)]
// CHECK:STDOUT:   %Core.import_ref.fcc: %Destroy.type = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.Iterate.impl.%Destroy.facet (constants.%Destroy.facet.9cd)]
// CHECK:STDOUT:   %Core.import_ref.f50: @T.as_type.as.Iterate.impl.%T.as_type.as.Iterate.impl.NewCursor.type (%T.as_type.as.Iterate.impl.NewCursor.type.b5d) = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.Iterate.impl.%T.as_type.as.Iterate.impl.NewCursor (constants.%T.as_type.as.Iterate.impl.NewCursor.21f)]
// CHECK:STDOUT:   %Core.import_ref.590: @T.as_type.as.Iterate.impl.%T.as_type.as.Iterate.impl.Next.type (%T.as_type.as.Iterate.impl.Next.type.5dc) = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.Iterate.impl.%T.as_type.as.Iterate.impl.Next (constants.%T.as_type.as.Iterate.impl.Next.b73)]
// CHECK:STDOUT:   %Iterate.impl_witness_table.478 = impl_witness_table (%Core.import_ref.eef, %Core.import_ref.fcc, %Core.import_ref.f50, %Core.import_ref.590), @T.as_type.as.Iterate.impl [concrete]
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
// CHECK:STDOUT:   %operator_PlusEqual__carbon_thunk.decl: %operator_PlusEqual__carbon_thunk.type = fn_decl @operator_PlusEqual__carbon_thunk [concrete = constants.%operator_PlusEqual__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ConstRange.decl: type = class_decl @ConstRange [concrete = constants.%ConstRange] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @TestIterate(%R.loc27_17.2: %Iterate_where.type.f9d) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Iterate.lookup_impl_witness: <witness> = lookup_impl_witness %R.loc27_17.1, @Iterate [symbolic = %Iterate.lookup_impl_witness (constants.%Iterate.lookup_impl_witness.173)]
// CHECK:STDOUT:   %Iterate.facet.loc31_31.5: %Iterate.type = facet_value %R.as_type.loc27_73.1, (%Iterate.lookup_impl_witness) [symbolic = %Iterate.facet.loc31_31.5 (constants.%Iterate.facet.b38)]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.loc31_31.5) [symbolic = %Iterate.WithSelf.NewCursor.type (constants.%Iterate.WithSelf.NewCursor.type.5cd)]
// CHECK:STDOUT:   %.loc31_31.20: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type, %Iterate.facet.loc31_31.5 [symbolic = %.loc31_31.20 (constants.%.592)]
// CHECK:STDOUT:   %impl.elem2.loc31_31.2: @TestIterate.%.loc31_31.20 (%.592) = impl_witness_access %Iterate.lookup_impl_witness, element2 [symbolic = %impl.elem2.loc31_31.2 (constants.%impl.elem2.ae2)]
// CHECK:STDOUT:   %specific_impl_fn.loc31_31.4: <specific function> = specific_impl_function %impl.elem2.loc31_31.2, @Iterate.WithSelf.NewCursor(%Iterate.facet.loc31_31.5) [symbolic = %specific_impl_fn.loc31_31.4 (constants.%specific_impl_fn.06e)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %ptr: type = ptr_type %as_type [symbolic = %ptr (constants.%ptr.d27)]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.loc31_31.5) [symbolic = %Iterate.WithSelf.Next.type (constants.%Iterate.WithSelf.Next.type.d51)]
// CHECK:STDOUT:   %.loc31_31.21: type = fn_type_with_self_type %Iterate.WithSelf.Next.type, %Iterate.facet.loc31_31.5 [symbolic = %.loc31_31.21 (constants.%.524)]
// CHECK:STDOUT:   %impl.elem3.loc31_31.2: @TestIterate.%.loc31_31.21 (%.524) = impl_witness_access %Iterate.lookup_impl_witness, element3 [symbolic = %impl.elem3.loc31_31.2 (constants.%impl.elem3.7a5)]
// CHECK:STDOUT:   %specific_impl_fn.loc31_31.5: <specific function> = specific_impl_function %impl.elem3.loc31_31.2, @Iterate.WithSelf.Next(%Iterate.facet.loc31_31.5) [symbolic = %specific_impl_fn.loc31_31.5 (constants.%specific_impl_fn.24b)]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%impl.elem1) [symbolic = %Destroy.WithSelf.Op.type (constants.%Destroy.WithSelf.Op.type.9d6)]
// CHECK:STDOUT:   %.loc31_31.22: type = fn_type_with_self_type %Destroy.WithSelf.Op.type, %impl.elem1 [symbolic = %.loc31_31.22 (constants.%.57c)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %impl.elem1, @Destroy [symbolic = %Destroy.lookup_impl_witness (constants.%Destroy.lookup_impl_witness.198)]
// CHECK:STDOUT:   %impl.elem0.loc31_31.2: @TestIterate.%.loc31_31.22 (%.57c) = impl_witness_access %Destroy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc31_31.2 (constants.%impl.elem0.152)]
// CHECK:STDOUT:   %specific_impl_fn.loc31_31.6: <specific function> = specific_impl_function %impl.elem0.loc31_31.2, @Destroy.WithSelf.Op(%impl.elem1) [symbolic = %specific_impl_fn.loc31_31.6 (constants.%specific_impl_fn.9fb)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%r.param: @TestIterate.%R.as_type.loc27_73.1 (%R.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %i.patt: %pattern_type.e41 = value_binding_pattern i [concrete = constants.%i.patt]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %r.ref: @TestIterate.%R.as_type.loc27_73.1 (%R.as_type) = name_ref r, %r
// CHECK:STDOUT:     %impl.elem2.loc31_31.1: @TestIterate.%.loc31_31.20 (%.592) = impl_witness_access constants.%Iterate.lookup_impl_witness.173, element2 [symbolic = %impl.elem2.loc31_31.2 (constants.%impl.elem2.ae2)]
// CHECK:STDOUT:     %bound_method.loc31_31.1: <bound method> = bound_method %r.ref, %impl.elem2.loc31_31.1
// CHECK:STDOUT:     %Iterate.facet.loc31_31.1: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.173) [symbolic = %Iterate.facet.loc31_31.5 (constants.%Iterate.facet.b38)]
// CHECK:STDOUT:     %.loc31_31.1: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc31_31.1 [symbolic = %Iterate.facet.loc31_31.5 (constants.%Iterate.facet.b38)]
// CHECK:STDOUT:     %Iterate.facet.loc31_31.2: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.173) [symbolic = %Iterate.facet.loc31_31.5 (constants.%Iterate.facet.b38)]
// CHECK:STDOUT:     %.loc31_31.2: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc31_31.2 [symbolic = %Iterate.facet.loc31_31.5 (constants.%Iterate.facet.b38)]
// CHECK:STDOUT:     %specific_impl_fn.loc31_31.1: <specific function> = specific_impl_function %impl.elem2.loc31_31.1, @Iterate.WithSelf.NewCursor(constants.%Iterate.facet.b38) [symbolic = %specific_impl_fn.loc31_31.4 (constants.%specific_impl_fn.06e)]
// CHECK:STDOUT:     %bound_method.loc31_31.2: <bound method> = bound_method %r.ref, %specific_impl_fn.loc31_31.1
// CHECK:STDOUT:     %var: ref @TestIterate.%as_type (%as_type.e41) = var_storage invalid
// CHECK:STDOUT:     %Iterate.WithSelf.NewCursor.call: init @TestIterate.%as_type (%as_type.e41) to %var = call %bound_method.loc31_31.2(%r.ref)
// CHECK:STDOUT:     assign %var, %Iterate.WithSelf.NewCursor.call
// CHECK:STDOUT:     br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.next:
// CHECK:STDOUT:     %addr.loc31: @TestIterate.%ptr (%ptr.d27) = addr_of %var
// CHECK:STDOUT:     %impl.elem3.loc31_31.1: @TestIterate.%.loc31_31.21 (%.524) = impl_witness_access constants.%Iterate.lookup_impl_witness.173, element3 [symbolic = %impl.elem3.loc31_31.2 (constants.%impl.elem3.7a5)]
// CHECK:STDOUT:     %bound_method.loc31_31.3: <bound method> = bound_method %r.ref, %impl.elem3.loc31_31.1
// CHECK:STDOUT:     %Iterate.facet.loc31_31.3: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.173) [symbolic = %Iterate.facet.loc31_31.5 (constants.%Iterate.facet.b38)]
// CHECK:STDOUT:     %.loc31_31.3: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc31_31.3 [symbolic = %Iterate.facet.loc31_31.5 (constants.%Iterate.facet.b38)]
// CHECK:STDOUT:     %Iterate.facet.loc31_31.4: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.173) [symbolic = %Iterate.facet.loc31_31.5 (constants.%Iterate.facet.b38)]
// CHECK:STDOUT:     %.loc31_31.4: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc31_31.4 [symbolic = %Iterate.facet.loc31_31.5 (constants.%Iterate.facet.b38)]
// CHECK:STDOUT:     %.loc31_31.5: %Destroy.type = converted constants.%as_type.e41, constants.%impl.elem1.834 [symbolic = %impl.elem1 (constants.%impl.elem1.834)]
// CHECK:STDOUT:     %.loc31_31.6: %Destroy.type = converted constants.%as_type.e41, constants.%impl.elem1.834 [symbolic = %impl.elem1 (constants.%impl.elem1.834)]
// CHECK:STDOUT:     %specific_impl_fn.loc31_31.2: <specific function> = specific_impl_function %impl.elem3.loc31_31.1, @Iterate.WithSelf.Next(constants.%Iterate.facet.b38) [symbolic = %specific_impl_fn.loc31_31.5 (constants.%specific_impl_fn.24b)]
// CHECK:STDOUT:     %bound_method.loc31_31.4: <bound method> = bound_method %r.ref, %specific_impl_fn.loc31_31.2
// CHECK:STDOUT:     %.loc31_31.7: ref %Optional.dda = temporary_storage
// CHECK:STDOUT:     %Iterate.WithSelf.Next.call: init %Optional.dda to %.loc31_31.7 = call %bound_method.loc31_31.4(%r.ref, %addr.loc31)
// CHECK:STDOUT:     %.loc31_31.8: ref %Optional.dda = temporary %.loc31_31.7, %Iterate.WithSelf.Next.call
// CHECK:STDOUT:     %.loc31_31.9: %Optional.HasValue.type.b33 = specific_constant imports.%Core.import_ref.bb6, @Optional(constants.%OptionalStorage.facet.381) [concrete = constants.%Optional.HasValue.1a5]
// CHECK:STDOUT:     %HasValue.ref: %Optional.HasValue.type.b33 = name_ref HasValue, %.loc31_31.9 [concrete = constants.%Optional.HasValue.1a5]
// CHECK:STDOUT:     %Optional.HasValue.bound: <bound method> = bound_method %.loc31_31.8, %HasValue.ref
// CHECK:STDOUT:     %Optional.HasValue.specific_fn: <specific function> = specific_function %HasValue.ref, @Optional.HasValue(constants.%OptionalStorage.facet.381) [concrete = constants.%Optional.HasValue.specific_fn.39a]
// CHECK:STDOUT:     %bound_method.loc31_31.5: <bound method> = bound_method %.loc31_31.8, %Optional.HasValue.specific_fn
// CHECK:STDOUT:     %.loc31_31.10: %Optional.dda = acquire_value %.loc31_31.8
// CHECK:STDOUT:     %Optional.HasValue.call: init bool = call %bound_method.loc31_31.5(%.loc31_31.10)
// CHECK:STDOUT:     %.loc31_31.11: bool = value_of_initializer %Optional.HasValue.call
// CHECK:STDOUT:     %.loc31_31.12: bool = converted %Optional.HasValue.call, %.loc31_31.11
// CHECK:STDOUT:     if %.loc31_31.12 br !for.body else br !for.done
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.body:
// CHECK:STDOUT:     %.loc31_31.13: %Optional.Get.type.341 = specific_constant imports.%Core.import_ref.daba, @Optional(constants.%OptionalStorage.facet.381) [concrete = constants.%Optional.Get.733]
// CHECK:STDOUT:     %Get.ref: %Optional.Get.type.341 = name_ref Get, %.loc31_31.13 [concrete = constants.%Optional.Get.733]
// CHECK:STDOUT:     %Optional.Get.bound: <bound method> = bound_method %.loc31_31.8, %Get.ref
// CHECK:STDOUT:     %Optional.Get.specific_fn: <specific function> = specific_function %Get.ref, @Optional.Get(constants.%OptionalStorage.facet.381) [concrete = constants.%Optional.Get.specific_fn.cd3]
// CHECK:STDOUT:     %bound_method.loc31_31.6: <bound method> = bound_method %.loc31_31.8, %Optional.Get.specific_fn
// CHECK:STDOUT:     %.loc31_31.14: ref %ValueType = temporary_storage
// CHECK:STDOUT:     %.loc31_31.15: %Optional.dda = acquire_value %.loc31_31.8
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
// CHECK:STDOUT:     %ValueType.Op.bound.loc31: <bound method> = bound_method %.loc31_31.16, constants.%ValueType.Op.4c94b9.1
// CHECK:STDOUT:     %Op.ref.loc31: %ValueType.cpp_destructor.type = name_ref Op, imports.%ValueType.cpp_destructor.decl [concrete = constants.%ValueType.cpp_destructor]
// CHECK:STDOUT:     %ValueType.cpp_destructor.bound.loc31: <bound method> = bound_method %.loc31_31.16, %Op.ref.loc31
// CHECK:STDOUT:     %ValueType.cpp_destructor.call.loc31: init %empty_tuple.type = call %ValueType.cpp_destructor.bound.loc31(%.loc31_31.16)
// CHECK:STDOUT:     %Destroy.Op.bound.loc31_31.1: <bound method> = bound_method %.loc31_31.8, constants.%Destroy.Op.1a2547.7
// CHECK:STDOUT:     %Destroy.Op.call.loc31_31.1: init %empty_tuple.type = call %Destroy.Op.bound.loc31_31.1(%.loc31_31.8)
// CHECK:STDOUT:     br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.done:
// CHECK:STDOUT:     %Destroy.Op.bound.loc31_31.2: <bound method> = bound_method %.loc31_31.8, constants.%Destroy.Op.1a2547.7
// CHECK:STDOUT:     %Destroy.Op.call.loc31_31.2: init %empty_tuple.type = call %Destroy.Op.bound.loc31_31.2(%.loc31_31.8)
// CHECK:STDOUT:     %impl.elem0.loc31_31.1: @TestIterate.%.loc31_31.22 (%.57c) = impl_witness_access constants.%Destroy.lookup_impl_witness.198, element0 [symbolic = %impl.elem0.loc31_31.2 (constants.%impl.elem0.152)]
// CHECK:STDOUT:     %bound_method.loc31_31.7: <bound method> = bound_method %var, %impl.elem0.loc31_31.1
// CHECK:STDOUT:     %.loc31_31.18: %Destroy.type = converted constants.%as_type.e41, constants.%impl.elem1.834 [symbolic = %impl.elem1 (constants.%impl.elem1.834)]
// CHECK:STDOUT:     %.loc31_31.19: %Destroy.type = converted constants.%as_type.e41, constants.%impl.elem1.834 [symbolic = %impl.elem1 (constants.%impl.elem1.834)]
// CHECK:STDOUT:     %specific_impl_fn.loc31_31.3: <specific function> = specific_impl_function %impl.elem0.loc31_31.1, @Destroy.WithSelf.Op(constants.%impl.elem1.834) [symbolic = %specific_impl_fn.loc31_31.6 (constants.%specific_impl_fn.9fb)]
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
// CHECK:STDOUT: fn @Destroy.Op.loc31_31.6(%self.param: ref %Optional.dda) {
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
// CHECK:STDOUT:   %impl.elem2: %.c99 = impl_witness_access constants.%Iterate.impl_witness.059, element2 [concrete = constants.%T.as_type.as.Iterate.impl.NewCursor.f2f]
// CHECK:STDOUT:   %bound_method.loc45_31.1: <bound method> = bound_method %c.ref, %impl.elem2
// CHECK:STDOUT:   %facet_value.loc45_31.1: %CppRange_where.type.d682d6.1 = facet_value constants.%ConstRange, (constants.%custom_witness.fef, constants.%custom_witness.c19, constants.%custom_witness.c2f, constants.%custom_witness.21358e.1, constants.%custom_witness.1d3, constants.%custom_witness.393, constants.%custom_witness.aa4, constants.%custom_witness.21358e.2) [concrete = constants.%facet_value.2b3]
// CHECK:STDOUT:   %.loc45_31.1: %CppRange_where.type.d682d6.1 = converted constants.%ConstRange, %facet_value.loc45_31.1 [concrete = constants.%facet_value.2b3]
// CHECK:STDOUT:   %facet_value.loc45_31.2: %CppRange_where.type.d682d6.1 = facet_value constants.%ConstRange, (constants.%custom_witness.fef, constants.%custom_witness.c19, constants.%custom_witness.c2f, constants.%custom_witness.21358e.1, constants.%custom_witness.1d3, constants.%custom_witness.393, constants.%custom_witness.aa4, constants.%custom_witness.21358e.2) [concrete = constants.%facet_value.2b3]
// CHECK:STDOUT:   %.loc45_31.2: %CppRange_where.type.d682d6.1 = converted constants.%ConstRange, %facet_value.loc45_31.2 [concrete = constants.%facet_value.2b3]
// CHECK:STDOUT:   %specific_fn.loc45_31.1: <specific function> = specific_function %impl.elem2, @T.as_type.as.Iterate.impl.NewCursor(constants.%facet_value.2b3) [concrete = constants.%T.as_type.as.Iterate.impl.NewCursor.specific_fn]
// CHECK:STDOUT:   %bound_method.loc45_31.2: <bound method> = bound_method %c.ref, %specific_fn.loc45_31.1
// CHECK:STDOUT:   %var: ref %tuple.type.df8 = var_storage invalid
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.call: init %tuple.type.df8 to %var = call %bound_method.loc45_31.2(%c.ref)
// CHECK:STDOUT:   assign %var, %T.as_type.as.Iterate.impl.NewCursor.call
// CHECK:STDOUT:   br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT: !for.next:
// CHECK:STDOUT:   %addr.loc45: %ptr.fc0 = addr_of %var
// CHECK:STDOUT:   %impl.elem3: %.858 = impl_witness_access constants.%Iterate.impl_witness.059, element3 [concrete = constants.%T.as_type.as.Iterate.impl.Next.47a]
// CHECK:STDOUT:   %bound_method.loc45_31.3: <bound method> = bound_method %c.ref, %impl.elem3
// CHECK:STDOUT:   %facet_value.loc45_31.3: %CppRange_where.type.d682d6.1 = facet_value constants.%ConstRange, (constants.%custom_witness.fef, constants.%custom_witness.c19, constants.%custom_witness.c2f, constants.%custom_witness.21358e.1, constants.%custom_witness.1d3, constants.%custom_witness.393, constants.%custom_witness.aa4, constants.%custom_witness.21358e.2) [concrete = constants.%facet_value.2b3]
// CHECK:STDOUT:   %.loc45_31.3: %CppRange_where.type.d682d6.1 = converted constants.%ConstRange, %facet_value.loc45_31.3 [concrete = constants.%facet_value.2b3]
// CHECK:STDOUT:   %facet_value.loc45_31.4: %CppRange_where.type.d682d6.1 = facet_value constants.%ConstRange, (constants.%custom_witness.fef, constants.%custom_witness.c19, constants.%custom_witness.c2f, constants.%custom_witness.21358e.1, constants.%custom_witness.1d3, constants.%custom_witness.393, constants.%custom_witness.aa4, constants.%custom_witness.21358e.2) [concrete = constants.%facet_value.2b3]
// CHECK:STDOUT:   %.loc45_31.4: %CppRange_where.type.d682d6.1 = converted constants.%ConstRange, %facet_value.loc45_31.4 [concrete = constants.%facet_value.2b3]
// CHECK:STDOUT:   %specific_fn.loc45_31.2: <specific function> = specific_function %impl.elem3, @T.as_type.as.Iterate.impl.Next(constants.%facet_value.2b3) [concrete = constants.%T.as_type.as.Iterate.impl.Next.specific_fn]
// CHECK:STDOUT:   %bound_method.loc45_31.4: <bound method> = bound_method %c.ref, %specific_fn.loc45_31.2
// CHECK:STDOUT:   %.loc45_31.5: ref %Optional.dda = temporary_storage
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.call: init %Optional.dda to %.loc45_31.5 = call %bound_method.loc45_31.4(%c.ref, %addr.loc45)
// CHECK:STDOUT:   %.loc45_31.6: ref %Optional.dda = temporary %.loc45_31.5, %T.as_type.as.Iterate.impl.Next.call
// CHECK:STDOUT:   %.loc45_31.7: %Optional.HasValue.type.b33 = specific_constant imports.%Core.import_ref.bb6, @Optional(constants.%OptionalStorage.facet.381) [concrete = constants.%Optional.HasValue.1a5]
// CHECK:STDOUT:   %HasValue.ref: %Optional.HasValue.type.b33 = name_ref HasValue, %.loc45_31.7 [concrete = constants.%Optional.HasValue.1a5]
// CHECK:STDOUT:   %Optional.HasValue.bound: <bound method> = bound_method %.loc45_31.6, %HasValue.ref
// CHECK:STDOUT:   %Optional.HasValue.specific_fn: <specific function> = specific_function %HasValue.ref, @Optional.HasValue(constants.%OptionalStorage.facet.381) [concrete = constants.%Optional.HasValue.specific_fn.39a]
// CHECK:STDOUT:   %bound_method.loc45_31.5: <bound method> = bound_method %.loc45_31.6, %Optional.HasValue.specific_fn
// CHECK:STDOUT:   %.loc45_31.8: %Optional.dda = acquire_value %.loc45_31.6
// CHECK:STDOUT:   %Optional.HasValue.call: init bool = call %bound_method.loc45_31.5(%.loc45_31.8)
// CHECK:STDOUT:   %.loc45_31.9: bool = value_of_initializer %Optional.HasValue.call
// CHECK:STDOUT:   %.loc45_31.10: bool = converted %Optional.HasValue.call, %.loc45_31.9
// CHECK:STDOUT:   if %.loc45_31.10 br !for.body else br !for.done
// CHECK:STDOUT:
// CHECK:STDOUT: !for.body:
// CHECK:STDOUT:   %.loc45_31.11: %Optional.Get.type.341 = specific_constant imports.%Core.import_ref.daba, @Optional(constants.%OptionalStorage.facet.381) [concrete = constants.%Optional.Get.733]
// CHECK:STDOUT:   %Get.ref: %Optional.Get.type.341 = name_ref Get, %.loc45_31.11 [concrete = constants.%Optional.Get.733]
// CHECK:STDOUT:   %Optional.Get.bound: <bound method> = bound_method %.loc45_31.6, %Get.ref
// CHECK:STDOUT:   %Optional.Get.specific_fn: <specific function> = specific_function %Get.ref, @Optional.Get(constants.%OptionalStorage.facet.381) [concrete = constants.%Optional.Get.specific_fn.cd3]
// CHECK:STDOUT:   %bound_method.loc45_31.6: <bound method> = bound_method %.loc45_31.6, %Optional.Get.specific_fn
// CHECK:STDOUT:   %.loc45_31.12: ref %ValueType = temporary_storage
// CHECK:STDOUT:   %.loc45_31.13: %Optional.dda = acquire_value %.loc45_31.6
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
// CHECK:STDOUT:   %ValueType.Op.bound.loc45: <bound method> = bound_method %.loc45_31.14, constants.%ValueType.Op.4c94b9.1
// CHECK:STDOUT:   %Op.ref.loc45: %ValueType.cpp_destructor.type = name_ref Op, imports.%ValueType.cpp_destructor.decl [concrete = constants.%ValueType.cpp_destructor]
// CHECK:STDOUT:   %ValueType.cpp_destructor.bound.loc45: <bound method> = bound_method %.loc45_31.14, %Op.ref.loc45
// CHECK:STDOUT:   %ValueType.cpp_destructor.call.loc45: init %empty_tuple.type = call %ValueType.cpp_destructor.bound.loc45(%.loc45_31.14)
// CHECK:STDOUT:   %Destroy.Op.bound.loc45_31.1: <bound method> = bound_method %.loc45_31.6, constants.%Destroy.Op.1a2547.7
// CHECK:STDOUT:   %Destroy.Op.call.loc45_31.1: init %empty_tuple.type = call %Destroy.Op.bound.loc45_31.1(%.loc45_31.6)
// CHECK:STDOUT:   br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT: !for.done:
// CHECK:STDOUT:   %Destroy.Op.bound.loc45_31.2: <bound method> = bound_method %.loc45_31.6, constants.%Destroy.Op.1a2547.7
// CHECK:STDOUT:   %Destroy.Op.call.loc45_31.2: init %empty_tuple.type = call %Destroy.Op.bound.loc45_31.2(%.loc45_31.6)
// CHECK:STDOUT:   %Destroy.Op.bound.loc45_31.3: <bound method> = bound_method %var, constants.%Destroy.Op.1a2547.8
// CHECK:STDOUT:   %Destroy.Op.call.loc45_31.3: init %empty_tuple.type = call %Destroy.Op.bound.loc45_31.3(%var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestIterate(constants.%R) {
// CHECK:STDOUT:   %R.patt.loc27_17.2 => constants.%R.patt
// CHECK:STDOUT:   %R.loc27_17.1 => constants.%R
// CHECK:STDOUT:   %R.as_type.loc27_73.1 => constants.%R.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.17d
// CHECK:STDOUT:   %r.param_patt.loc27_71.2 => constants.%r.param_patt.90c
// CHECK:STDOUT:   %r.patt.loc27_71.2 => constants.%r.patt.d6c
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestIterate(constants.%facet_value.883) {
// CHECK:STDOUT:   %R.patt.loc27_17.2 => constants.%R.patt
// CHECK:STDOUT:   %R.loc27_17.1 => constants.%facet_value.883
// CHECK:STDOUT:   %R.as_type.loc27_73.1 => constants.%ConstRange
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.025
// CHECK:STDOUT:   %r.param_patt.loc27_71.2 => constants.%r.param_patt.a73
// CHECK:STDOUT:   %r.patt.loc27_71.2 => constants.%r.patt.0e2
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc27 => constants.%complete_type.357
// CHECK:STDOUT:   %Iterate.lookup_impl_witness => constants.%Iterate.impl_witness.059
// CHECK:STDOUT:   %Iterate.facet.loc31_31.5 => constants.%Iterate.facet.11a
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type => constants.%Iterate.WithSelf.NewCursor.type.f75
// CHECK:STDOUT:   %.loc31_31.20 => constants.%.c99
// CHECK:STDOUT:   %impl.elem2.loc31_31.2 => constants.%T.as_type.as.Iterate.impl.NewCursor.f2f
// CHECK:STDOUT:   %specific_impl_fn.loc31_31.4 => constants.%T.as_type.as.Iterate.impl.NewCursor.specific_fn
// CHECK:STDOUT:   %impl.elem1 => constants.%Destroy.facet.777
// CHECK:STDOUT:   %as_type => constants.%tuple.type.df8
// CHECK:STDOUT:   %require_complete.1 => constants.%complete_type.e35
// CHECK:STDOUT:   %ptr => constants.%ptr.fc0
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type => constants.%Iterate.WithSelf.Next.type.c96
// CHECK:STDOUT:   %.loc31_31.21 => constants.%.858
// CHECK:STDOUT:   %impl.elem3.loc31_31.2 => constants.%T.as_type.as.Iterate.impl.Next.47a
// CHECK:STDOUT:   %specific_impl_fn.loc31_31.5 => constants.%T.as_type.as.Iterate.impl.Next.specific_fn
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
// CHECK:STDOUT:   %facet_type.f48: type = facet_type <@Destroy & @Copy> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %T.220: %OptionalStorage.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Optional.Get.type.bb9: type = fn_type @Optional.Get, @Optional(%T.220) [symbolic]
// CHECK:STDOUT:   %Optional.Get.fa1: %Optional.Get.type.bb9 = struct_value () [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.type.e9e: type = fn_type @Optional.HasValue, @Optional(%T.220) [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.6cd: %Optional.HasValue.type.e9e = struct_value () [symbolic]
// CHECK:STDOUT:   %T.283: %facet_type.f48 = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.d50: type = class_type @DefaultOptionalStorage, @DefaultOptionalStorage(%T.283) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Get.type.d3c: type = fn_type @T.as_type.as.OptionalStorage.impl.Get, @T.as_type.as.OptionalStorage.impl(%T.283) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Get.da2: %T.as_type.as.OptionalStorage.impl.Get.type.d3c = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Has.type.eb9: type = fn_type @T.as_type.as.OptionalStorage.impl.Has, @T.as_type.as.OptionalStorage.impl(%T.283) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Has.c7e: %T.as_type.as.OptionalStorage.impl.Has.type.eb9 = struct_value () [symbolic]
// CHECK:STDOUT:   %.Self.500: %Iterate.type = symbolic_binding .Self [symbolic_self]
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
// CHECK:STDOUT:   %facet_value.ccb: %facet_type.f48 = facet_value %ValueType, (%custom_witness.fef, %custom_witness.c19) [concrete]
// CHECK:STDOUT:   %Iterate.lookup_impl_witness.1d8: <witness> = lookup_impl_witness %.Self.500, @Iterate [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.294: %facet_type.f48 = impl_witness_access %Iterate.lookup_impl_witness.1d8, element0 [symbolic_self]
// CHECK:STDOUT:   %Iterate_where.type.f9d: type = facet_type <@Iterate where %impl.elem0.294 = %facet_value.ccb> [concrete]
// CHECK:STDOUT:   %pattern_type.c7b: type = pattern_type %Iterate_where.type.f9d [concrete]
// CHECK:STDOUT:   %R.patt: %pattern_type.c7b = symbolic_binding_pattern R, 0 [symbolic]
// CHECK:STDOUT:   %R: %Iterate_where.type.f9d = symbolic_binding R, 0 [symbolic]
// CHECK:STDOUT:   %R.as_type: type = facet_access_type %R [symbolic]
// CHECK:STDOUT:   %pattern_type.17d: type = pattern_type %R.as_type [symbolic]
// CHECK:STDOUT:   %r.param_patt: %pattern_type.17d = value_param_pattern [symbolic]
// CHECK:STDOUT:   %r.patt: %pattern_type.17d = at_binding_pattern r, %r.param_patt [symbolic]
// CHECK:STDOUT:   %i.patt: %pattern_type.e41 = value_binding_pattern i [concrete]
// CHECK:STDOUT:   %Iterate.lookup_impl_witness.173: <witness> = lookup_impl_witness %R, @Iterate [symbolic]
// CHECK:STDOUT:   %Iterate.facet.b38: %Iterate.type = facet_value %R.as_type, (%Iterate.lookup_impl_witness.173) [symbolic]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.5cd: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.b38) [symbolic]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.d51: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.b38) [symbolic]
// CHECK:STDOUT:   %.592: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.5cd, %Iterate.facet.b38 [symbolic]
// CHECK:STDOUT:   %impl.elem2.ae2: %.592 = impl_witness_access %Iterate.lookup_impl_witness.173, element2 [symbolic]
// CHECK:STDOUT:   %impl.elem1.834: %Destroy.type = impl_witness_access %Iterate.lookup_impl_witness.173, element1 [symbolic]
// CHECK:STDOUT:   %as_type.e41: type = facet_access_type %impl.elem1.834 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.06e: <specific function> = specific_impl_function %impl.elem2.ae2, @Iterate.WithSelf.NewCursor(%Iterate.facet.b38) [symbolic]
// CHECK:STDOUT:   %ptr.d27: type = ptr_type %as_type.e41 [symbolic]
// CHECK:STDOUT:   %.524: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.d51, %Iterate.facet.b38 [symbolic]
// CHECK:STDOUT:   %impl.elem3.7a5: %.524 = impl_witness_access %Iterate.lookup_impl_witness.173, element3 [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.257: type = class_type @DefaultOptionalStorage, @DefaultOptionalStorage(%facet_value.ccb) [concrete]
// CHECK:STDOUT:   %OptionalStorage.impl_witness.2fc: <witness> = impl_witness imports.%OptionalStorage.impl_witness_table.faa, @T.as_type.as.OptionalStorage.impl(%facet_value.ccb) [concrete]
// CHECK:STDOUT:   %OptionalStorage.facet.7b8: %OptionalStorage.type = facet_value %ValueType, (%OptionalStorage.impl_witness.2fc) [concrete]
// CHECK:STDOUT:   %Optional.0a6: type = class_type @Optional, @Optional(%OptionalStorage.facet.7b8) [concrete]
// CHECK:STDOUT:   %specific_impl_fn.24b: <specific function> = specific_impl_function %impl.elem3.7a5, @Iterate.WithSelf.Next(%Iterate.facet.b38) [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.type.07e: type = fn_type @Optional.HasValue, @Optional(%OptionalStorage.facet.7b8) [concrete]
// CHECK:STDOUT:   %Optional.HasValue.77a: %Optional.HasValue.type.07e = struct_value () [concrete]
// CHECK:STDOUT:   %Optional.Get.type.fa3: type = fn_type @Optional.Get, @Optional(%OptionalStorage.facet.7b8) [concrete]
// CHECK:STDOUT:   %Optional.Get.e65: %Optional.Get.type.fa3 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.facet.306: %Destroy.type = facet_value %ValueType, (%custom_witness.fef) [concrete]
// CHECK:STDOUT:   %MaybeUnformed.669: type = class_type @MaybeUnformed, @MaybeUnformed(%Destroy.facet.306) [concrete]
// CHECK:STDOUT:   %.b36: type = maybe_unformed_type %ValueType [concrete]
// CHECK:STDOUT:   %struct_type.value.has_value.5a1: type = struct_type {.value: %MaybeUnformed.669, .has_value: bool} [concrete]
// CHECK:STDOUT:   %Optional.HasValue.specific_fn: <specific function> = specific_function %Optional.HasValue.77a, @Optional.HasValue(%OptionalStorage.facet.7b8) [concrete]
// CHECK:STDOUT:   %Optional.Get.specific_fn: <specific function> = specific_function %Optional.Get.e65, @Optional.Get(%OptionalStorage.facet.7b8) [concrete]
// CHECK:STDOUT:   %operator_PlusEqual__carbon_thunk.type: type = fn_type @operator_PlusEqual__carbon_thunk [concrete]
// CHECK:STDOUT:   %operator_PlusEqual__carbon_thunk: %operator_PlusEqual__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.7: type = fn_type @Destroy.Op.loc31_31.6 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.7: %Destroy.Op.type.1d8f74.7 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.198: <witness> = lookup_impl_witness %impl.elem1.834, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.9d6: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%impl.elem1.834) [symbolic]
// CHECK:STDOUT:   %.57c: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.9d6, %impl.elem1.834 [symbolic]
// CHECK:STDOUT:   %impl.elem0.152: %.57c = impl_witness_access %Destroy.lookup_impl_witness.198, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.9fb: <specific function> = specific_impl_function %impl.elem0.152, @Destroy.WithSelf.Op(%impl.elem1.834) [symbolic]
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
// CHECK:STDOUT:   %Core.import_ref.f4e = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], unloaded
// CHECK:STDOUT:   %Core.import_ref.81d = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], unloaded
// CHECK:STDOUT:   %Core.import_ref.61e: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Has.type (%T.as_type.as.OptionalStorage.impl.Has.type.eb9) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Has (constants.%T.as_type.as.OptionalStorage.impl.Has.c7e)]
// CHECK:STDOUT:   %Core.import_ref.ce2e: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Get.type (%T.as_type.as.OptionalStorage.impl.Get.type.d3c) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Get (constants.%T.as_type.as.OptionalStorage.impl.Get.da2)]
// CHECK:STDOUT:   %Core.import_ref.8fc = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], unloaded
// CHECK:STDOUT:   %OptionalStorage.impl_witness_table.faa = impl_witness_table (%Core.import_ref.0c6, %Core.import_ref.f4e, %Core.import_ref.81d, %Core.import_ref.61e, %Core.import_ref.ce2e, %Core.import_ref.8fc), @T.as_type.as.OptionalStorage.impl [concrete]
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
// CHECK:STDOUT: generic fn @TestIterate(%R.loc27_17.2: %Iterate_where.type.f9d) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Iterate.lookup_impl_witness: <witness> = lookup_impl_witness %R.loc27_17.1, @Iterate [symbolic = %Iterate.lookup_impl_witness (constants.%Iterate.lookup_impl_witness.173)]
// CHECK:STDOUT:   %Iterate.facet.loc31_31.5: %Iterate.type = facet_value %R.as_type.loc27_73.1, (%Iterate.lookup_impl_witness) [symbolic = %Iterate.facet.loc31_31.5 (constants.%Iterate.facet.b38)]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.loc31_31.5) [symbolic = %Iterate.WithSelf.NewCursor.type (constants.%Iterate.WithSelf.NewCursor.type.5cd)]
// CHECK:STDOUT:   %.loc31_31.20: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type, %Iterate.facet.loc31_31.5 [symbolic = %.loc31_31.20 (constants.%.592)]
// CHECK:STDOUT:   %impl.elem2.loc31_31.2: @TestIterate.%.loc31_31.20 (%.592) = impl_witness_access %Iterate.lookup_impl_witness, element2 [symbolic = %impl.elem2.loc31_31.2 (constants.%impl.elem2.ae2)]
// CHECK:STDOUT:   %specific_impl_fn.loc31_31.4: <specific function> = specific_impl_function %impl.elem2.loc31_31.2, @Iterate.WithSelf.NewCursor(%Iterate.facet.loc31_31.5) [symbolic = %specific_impl_fn.loc31_31.4 (constants.%specific_impl_fn.06e)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %ptr: type = ptr_type %as_type [symbolic = %ptr (constants.%ptr.d27)]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.loc31_31.5) [symbolic = %Iterate.WithSelf.Next.type (constants.%Iterate.WithSelf.Next.type.d51)]
// CHECK:STDOUT:   %.loc31_31.21: type = fn_type_with_self_type %Iterate.WithSelf.Next.type, %Iterate.facet.loc31_31.5 [symbolic = %.loc31_31.21 (constants.%.524)]
// CHECK:STDOUT:   %impl.elem3.loc31_31.2: @TestIterate.%.loc31_31.21 (%.524) = impl_witness_access %Iterate.lookup_impl_witness, element3 [symbolic = %impl.elem3.loc31_31.2 (constants.%impl.elem3.7a5)]
// CHECK:STDOUT:   %specific_impl_fn.loc31_31.5: <specific function> = specific_impl_function %impl.elem3.loc31_31.2, @Iterate.WithSelf.Next(%Iterate.facet.loc31_31.5) [symbolic = %specific_impl_fn.loc31_31.5 (constants.%specific_impl_fn.24b)]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%impl.elem1) [symbolic = %Destroy.WithSelf.Op.type (constants.%Destroy.WithSelf.Op.type.9d6)]
// CHECK:STDOUT:   %.loc31_31.22: type = fn_type_with_self_type %Destroy.WithSelf.Op.type, %impl.elem1 [symbolic = %.loc31_31.22 (constants.%.57c)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %impl.elem1, @Destroy [symbolic = %Destroy.lookup_impl_witness (constants.%Destroy.lookup_impl_witness.198)]
// CHECK:STDOUT:   %impl.elem0.loc31_31.2: @TestIterate.%.loc31_31.22 (%.57c) = impl_witness_access %Destroy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc31_31.2 (constants.%impl.elem0.152)]
// CHECK:STDOUT:   %specific_impl_fn.loc31_31.6: <specific function> = specific_impl_function %impl.elem0.loc31_31.2, @Destroy.WithSelf.Op(%impl.elem1) [symbolic = %specific_impl_fn.loc31_31.6 (constants.%specific_impl_fn.9fb)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%r.param: @TestIterate.%R.as_type.loc27_73.1 (%R.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %i.patt: %pattern_type.e41 = value_binding_pattern i [concrete = constants.%i.patt]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %r.ref: @TestIterate.%R.as_type.loc27_73.1 (%R.as_type) = name_ref r, %r
// CHECK:STDOUT:     %impl.elem2.loc31_31.1: @TestIterate.%.loc31_31.20 (%.592) = impl_witness_access constants.%Iterate.lookup_impl_witness.173, element2 [symbolic = %impl.elem2.loc31_31.2 (constants.%impl.elem2.ae2)]
// CHECK:STDOUT:     %bound_method.loc31_31.1: <bound method> = bound_method %r.ref, %impl.elem2.loc31_31.1
// CHECK:STDOUT:     %Iterate.facet.loc31_31.1: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.173) [symbolic = %Iterate.facet.loc31_31.5 (constants.%Iterate.facet.b38)]
// CHECK:STDOUT:     %.loc31_31.1: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc31_31.1 [symbolic = %Iterate.facet.loc31_31.5 (constants.%Iterate.facet.b38)]
// CHECK:STDOUT:     %Iterate.facet.loc31_31.2: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.173) [symbolic = %Iterate.facet.loc31_31.5 (constants.%Iterate.facet.b38)]
// CHECK:STDOUT:     %.loc31_31.2: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc31_31.2 [symbolic = %Iterate.facet.loc31_31.5 (constants.%Iterate.facet.b38)]
// CHECK:STDOUT:     %specific_impl_fn.loc31_31.1: <specific function> = specific_impl_function %impl.elem2.loc31_31.1, @Iterate.WithSelf.NewCursor(constants.%Iterate.facet.b38) [symbolic = %specific_impl_fn.loc31_31.4 (constants.%specific_impl_fn.06e)]
// CHECK:STDOUT:     %bound_method.loc31_31.2: <bound method> = bound_method %r.ref, %specific_impl_fn.loc31_31.1
// CHECK:STDOUT:     %var: ref @TestIterate.%as_type (%as_type.e41) = var_storage invalid
// CHECK:STDOUT:     %Iterate.WithSelf.NewCursor.call: init @TestIterate.%as_type (%as_type.e41) to %var = call %bound_method.loc31_31.2(%r.ref)
// CHECK:STDOUT:     assign %var, %Iterate.WithSelf.NewCursor.call
// CHECK:STDOUT:     br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.next:
// CHECK:STDOUT:     %addr.loc31: @TestIterate.%ptr (%ptr.d27) = addr_of %var
// CHECK:STDOUT:     %impl.elem3.loc31_31.1: @TestIterate.%.loc31_31.21 (%.524) = impl_witness_access constants.%Iterate.lookup_impl_witness.173, element3 [symbolic = %impl.elem3.loc31_31.2 (constants.%impl.elem3.7a5)]
// CHECK:STDOUT:     %bound_method.loc31_31.3: <bound method> = bound_method %r.ref, %impl.elem3.loc31_31.1
// CHECK:STDOUT:     %Iterate.facet.loc31_31.3: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.173) [symbolic = %Iterate.facet.loc31_31.5 (constants.%Iterate.facet.b38)]
// CHECK:STDOUT:     %.loc31_31.3: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc31_31.3 [symbolic = %Iterate.facet.loc31_31.5 (constants.%Iterate.facet.b38)]
// CHECK:STDOUT:     %Iterate.facet.loc31_31.4: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.173) [symbolic = %Iterate.facet.loc31_31.5 (constants.%Iterate.facet.b38)]
// CHECK:STDOUT:     %.loc31_31.4: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc31_31.4 [symbolic = %Iterate.facet.loc31_31.5 (constants.%Iterate.facet.b38)]
// CHECK:STDOUT:     %.loc31_31.5: %Destroy.type = converted constants.%as_type.e41, constants.%impl.elem1.834 [symbolic = %impl.elem1 (constants.%impl.elem1.834)]
// CHECK:STDOUT:     %.loc31_31.6: %Destroy.type = converted constants.%as_type.e41, constants.%impl.elem1.834 [symbolic = %impl.elem1 (constants.%impl.elem1.834)]
// CHECK:STDOUT:     %specific_impl_fn.loc31_31.2: <specific function> = specific_impl_function %impl.elem3.loc31_31.1, @Iterate.WithSelf.Next(constants.%Iterate.facet.b38) [symbolic = %specific_impl_fn.loc31_31.5 (constants.%specific_impl_fn.24b)]
// CHECK:STDOUT:     %bound_method.loc31_31.4: <bound method> = bound_method %r.ref, %specific_impl_fn.loc31_31.2
// CHECK:STDOUT:     %.loc31_31.7: ref %Optional.0a6 = temporary_storage
// CHECK:STDOUT:     %Iterate.WithSelf.Next.call: init %Optional.0a6 to %.loc31_31.7 = call %bound_method.loc31_31.4(%r.ref, %addr.loc31)
// CHECK:STDOUT:     %.loc31_31.8: ref %Optional.0a6 = temporary %.loc31_31.7, %Iterate.WithSelf.Next.call
// CHECK:STDOUT:     %.loc31_31.9: %Optional.HasValue.type.07e = specific_constant imports.%Core.import_ref.bb6, @Optional(constants.%OptionalStorage.facet.7b8) [concrete = constants.%Optional.HasValue.77a]
// CHECK:STDOUT:     %HasValue.ref: %Optional.HasValue.type.07e = name_ref HasValue, %.loc31_31.9 [concrete = constants.%Optional.HasValue.77a]
// CHECK:STDOUT:     %Optional.HasValue.bound: <bound method> = bound_method %.loc31_31.8, %HasValue.ref
// CHECK:STDOUT:     %Optional.HasValue.specific_fn: <specific function> = specific_function %HasValue.ref, @Optional.HasValue(constants.%OptionalStorage.facet.7b8) [concrete = constants.%Optional.HasValue.specific_fn]
// CHECK:STDOUT:     %bound_method.loc31_31.5: <bound method> = bound_method %.loc31_31.8, %Optional.HasValue.specific_fn
// CHECK:STDOUT:     %.loc31_31.10: %Optional.0a6 = acquire_value %.loc31_31.8
// CHECK:STDOUT:     %Optional.HasValue.call: init bool = call %bound_method.loc31_31.5(%.loc31_31.10)
// CHECK:STDOUT:     %.loc31_31.11: bool = value_of_initializer %Optional.HasValue.call
// CHECK:STDOUT:     %.loc31_31.12: bool = converted %Optional.HasValue.call, %.loc31_31.11
// CHECK:STDOUT:     if %.loc31_31.12 br !for.body else br !for.done
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.body:
// CHECK:STDOUT:     %.loc31_31.13: %Optional.Get.type.fa3 = specific_constant imports.%Core.import_ref.daba, @Optional(constants.%OptionalStorage.facet.7b8) [concrete = constants.%Optional.Get.e65]
// CHECK:STDOUT:     %Get.ref: %Optional.Get.type.fa3 = name_ref Get, %.loc31_31.13 [concrete = constants.%Optional.Get.e65]
// CHECK:STDOUT:     %Optional.Get.bound: <bound method> = bound_method %.loc31_31.8, %Get.ref
// CHECK:STDOUT:     %Optional.Get.specific_fn: <specific function> = specific_function %Get.ref, @Optional.Get(constants.%OptionalStorage.facet.7b8) [concrete = constants.%Optional.Get.specific_fn]
// CHECK:STDOUT:     %bound_method.loc31_31.6: <bound method> = bound_method %.loc31_31.8, %Optional.Get.specific_fn
// CHECK:STDOUT:     %.loc31_31.14: ref %ValueType = temporary_storage
// CHECK:STDOUT:     %.loc31_31.15: %Optional.0a6 = acquire_value %.loc31_31.8
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
// CHECK:STDOUT:     %ValueType.Op.bound.loc31: <bound method> = bound_method %.loc31_31.16, constants.%ValueType.Op.4c94b9.1
// CHECK:STDOUT:     %Op.ref.loc31: %ValueType.cpp_destructor.type = name_ref Op, imports.%ValueType.cpp_destructor.decl [concrete = constants.%ValueType.cpp_destructor]
// CHECK:STDOUT:     %ValueType.cpp_destructor.bound.loc31: <bound method> = bound_method %.loc31_31.16, %Op.ref.loc31
// CHECK:STDOUT:     %ValueType.cpp_destructor.call.loc31: init %empty_tuple.type = call %ValueType.cpp_destructor.bound.loc31(%.loc31_31.16)
// CHECK:STDOUT:     %Destroy.Op.bound.loc31_31.1: <bound method> = bound_method %.loc31_31.8, constants.%Destroy.Op.1a2547.7
// CHECK:STDOUT:     %Destroy.Op.call.loc31_31.1: init %empty_tuple.type = call %Destroy.Op.bound.loc31_31.1(%.loc31_31.8)
// CHECK:STDOUT:     br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.done:
// CHECK:STDOUT:     %Destroy.Op.bound.loc31_31.2: <bound method> = bound_method %.loc31_31.8, constants.%Destroy.Op.1a2547.7
// CHECK:STDOUT:     %Destroy.Op.call.loc31_31.2: init %empty_tuple.type = call %Destroy.Op.bound.loc31_31.2(%.loc31_31.8)
// CHECK:STDOUT:     %impl.elem0.loc31_31.1: @TestIterate.%.loc31_31.22 (%.57c) = impl_witness_access constants.%Destroy.lookup_impl_witness.198, element0 [symbolic = %impl.elem0.loc31_31.2 (constants.%impl.elem0.152)]
// CHECK:STDOUT:     %bound_method.loc31_31.7: <bound method> = bound_method %var, %impl.elem0.loc31_31.1
// CHECK:STDOUT:     %.loc31_31.18: %Destroy.type = converted constants.%as_type.e41, constants.%impl.elem1.834 [symbolic = %impl.elem1 (constants.%impl.elem1.834)]
// CHECK:STDOUT:     %.loc31_31.19: %Destroy.type = converted constants.%as_type.e41, constants.%impl.elem1.834 [symbolic = %impl.elem1 (constants.%impl.elem1.834)]
// CHECK:STDOUT:     %specific_impl_fn.loc31_31.3: <specific function> = specific_impl_function %impl.elem0.loc31_31.1, @Destroy.WithSelf.Op(constants.%impl.elem1.834) [symbolic = %specific_impl_fn.loc31_31.6 (constants.%specific_impl_fn.9fb)]
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
// CHECK:STDOUT: fn @Destroy.Op.loc31_31.6(%self.param: ref %Optional.0a6) {
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
// CHECK:STDOUT:   %var: ref <error> = var_storage invalid [concrete = <error>]
// CHECK:STDOUT:   assign <error>, <error>
// CHECK:STDOUT:   br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT: !for.next:
// CHECK:STDOUT:   %addr.loc98: <error> = addr_of <error> [concrete = <error>]
// CHECK:STDOUT:   %HasValue.ref: <error> = name_ref HasValue, <error> [concrete = <error>]
// CHECK:STDOUT:   if <error> br !for.body else br !for.done
// CHECK:STDOUT:
// CHECK:STDOUT: !for.body:
// CHECK:STDOUT:   %Get.ref: <error> = name_ref Get, <error> [concrete = <error>]
// CHECK:STDOUT:   %.loc98: type = splice_block %ValueType.ref.loc98 [concrete = constants.%ValueType] {
// CHECK:STDOUT:     %Cpp.ref.loc98: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %N.ref.loc98: <namespace> = name_ref N, imports.%N [concrete = imports.%N]
// CHECK:STDOUT:     %ValueType.ref.loc98: type = name_ref ValueType, imports.%ValueType.decl [concrete = constants.%ValueType]
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
// CHECK:STDOUT:   %R.as_type.loc27_73.1 => constants.%R.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.17d
// CHECK:STDOUT:   %r.param_patt.loc27_71.2 => constants.%r.param_patt
// CHECK:STDOUT:   %r.patt.loc27_71.2 => constants.%r.patt
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
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Get.type.d3c: type = fn_type @T.as_type.as.OptionalStorage.impl.Get, @T.as_type.as.OptionalStorage.impl(%T.283) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Get.da2: %T.as_type.as.OptionalStorage.impl.Get.type.d3c = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Has.type.eb9: type = fn_type @T.as_type.as.OptionalStorage.impl.Has, @T.as_type.as.OptionalStorage.impl(%T.283) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Has.c7e: %T.as_type.as.OptionalStorage.impl.Has.type.eb9 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Some.type.7eb: type = fn_type @T.as_type.as.OptionalStorage.impl.Some, @T.as_type.as.OptionalStorage.impl(%T.283) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Some.2b4: %T.as_type.as.OptionalStorage.impl.Some.type.7eb = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.None.type.d4d: type = fn_type @T.as_type.as.OptionalStorage.impl.None, @T.as_type.as.OptionalStorage.impl(%T.283) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.None.d64: %T.as_type.as.OptionalStorage.impl.None.type.d4d = struct_value () [symbolic]
// CHECK:STDOUT:   %.Self.500: %Iterate.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %CppRange.type: type = facet_type <@CppRange> [concrete]
// CHECK:STDOUT:   %.Self.e0b: %CppRange.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %CppRangeForIterate.lookup_impl_witness.50b: <witness> = lookup_impl_witness %.Self.e0b, @CppRangeForIterate [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.32d: type = impl_witness_access %CppRangeForIterate.lookup_impl_witness.50b, element0 [symbolic_self]
// CHECK:STDOUT:   %CppUnsafeDeref.lookup_impl_witness.851: <witness> = lookup_impl_witness %impl.elem0.32d, @CppUnsafeDeref [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.10d: type = impl_witness_access %CppUnsafeDeref.lookup_impl_witness.851, element0 [symbolic_self]
// CHECK:STDOUT:   %CppRange_where.type.d682d6.1: type = facet_type <@CppRange where %impl.elem0.10d impls @Copy> [concrete]
// CHECK:STDOUT:   %T.f45530.2: %CppRange_where.type.d682d6.1 = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %CppRangeForIterate.lookup_impl_witness.b581a8.2: <witness> = lookup_impl_witness %T.f45530.2, @CppRangeForIterate [symbolic]
// CHECK:STDOUT:   %impl.elem1.c2dc15.2: type = impl_witness_access %CppRangeForIterate.lookup_impl_witness.b581a8.2, element1 [symbolic]
// CHECK:STDOUT:   %impl.elem0.5f971c.2: type = impl_witness_access %CppRangeForIterate.lookup_impl_witness.b581a8.2, element0 [symbolic]
// CHECK:STDOUT:   %tuple.type.1ba: type = tuple_type (%impl.elem0.5f971c.2, %impl.elem1.c2dc15.2) [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.812: <witness> = lookup_impl_witness %tuple.type.1ba, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.9cd: %Destroy.type = facet_value %tuple.type.1ba, (%Destroy.lookup_impl_witness.812) [symbolic]
// CHECK:STDOUT:   %CppUnsafeDeref.lookup_impl_witness.87e543.2: <witness> = lookup_impl_witness %impl.elem0.5f971c.2, @CppUnsafeDeref [symbolic]
// CHECK:STDOUT:   %impl.elem0.493765.2: type = impl_witness_access %CppUnsafeDeref.lookup_impl_witness.87e543.2, element0 [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.35816c.2: <witness> = lookup_impl_witness %impl.elem0.493765.2, @Destroy [symbolic]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.167: <witness> = lookup_impl_witness %impl.elem0.493765.2, @Copy [symbolic]
// CHECK:STDOUT:   %facet_value.f93: %facet_type.f48 = facet_value %impl.elem0.493765.2, (%Destroy.lookup_impl_witness.35816c.2, %Copy.lookup_impl_witness.167) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.type.5dc: type = fn_type @T.as_type.as.Iterate.impl.Next, @T.as_type.as.Iterate.impl(%T.f45530.2) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.b73: %T.as_type.as.Iterate.impl.Next.type.5dc = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.type.b5d: type = fn_type @T.as_type.as.Iterate.impl.NewCursor, @T.as_type.as.Iterate.impl(%T.f45530.2) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.21f: %T.as_type.as.Iterate.impl.NewCursor.type.b5d = struct_value () [symbolic]
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
// CHECK:STDOUT:   %facet_value.ccb: %facet_type.f48 = facet_value %ValueType, (%custom_witness.fef, %custom_witness.c19) [concrete]
// CHECK:STDOUT:   %Iterate.lookup_impl_witness.1d8: <witness> = lookup_impl_witness %.Self.500, @Iterate [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.294: %facet_type.f48 = impl_witness_access %Iterate.lookup_impl_witness.1d8, element0 [symbolic_self]
// CHECK:STDOUT:   %Iterate_where.type.f9d: type = facet_type <@Iterate where %impl.elem0.294 = %facet_value.ccb> [concrete]
// CHECK:STDOUT:   %pattern_type.c7b: type = pattern_type %Iterate_where.type.f9d [concrete]
// CHECK:STDOUT:   %R.patt: %pattern_type.c7b = symbolic_binding_pattern R, 0 [symbolic]
// CHECK:STDOUT:   %R: %Iterate_where.type.f9d = symbolic_binding R, 0 [symbolic]
// CHECK:STDOUT:   %R.as_type: type = facet_access_type %R [symbolic]
// CHECK:STDOUT:   %pattern_type.17d: type = pattern_type %R.as_type [symbolic]
// CHECK:STDOUT:   %r.param_patt.90c: %pattern_type.17d = value_param_pattern [symbolic]
// CHECK:STDOUT:   %r.patt.d6c: %pattern_type.17d = at_binding_pattern r, %r.param_patt.90c [symbolic]
// CHECK:STDOUT:   %i.patt: %pattern_type.e41 = value_binding_pattern i [concrete]
// CHECK:STDOUT:   %Iterate.lookup_impl_witness.173: <witness> = lookup_impl_witness %R, @Iterate [symbolic]
// CHECK:STDOUT:   %Iterate.facet.b38: %Iterate.type = facet_value %R.as_type, (%Iterate.lookup_impl_witness.173) [symbolic]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.5cd: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.b38) [symbolic]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.d51: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.b38) [symbolic]
// CHECK:STDOUT:   %.592: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.5cd, %Iterate.facet.b38 [symbolic]
// CHECK:STDOUT:   %impl.elem2.ae2: %.592 = impl_witness_access %Iterate.lookup_impl_witness.173, element2 [symbolic]
// CHECK:STDOUT:   %impl.elem1.834: %Destroy.type = impl_witness_access %Iterate.lookup_impl_witness.173, element1 [symbolic]
// CHECK:STDOUT:   %as_type.e41: type = facet_access_type %impl.elem1.834 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.06e: <specific function> = specific_impl_function %impl.elem2.ae2, @Iterate.WithSelf.NewCursor(%Iterate.facet.b38) [symbolic]
// CHECK:STDOUT:   %ptr.d27: type = ptr_type %as_type.e41 [symbolic]
// CHECK:STDOUT:   %.524: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.d51, %Iterate.facet.b38 [symbolic]
// CHECK:STDOUT:   %impl.elem3.7a5: %.524 = impl_witness_access %Iterate.lookup_impl_witness.173, element3 [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.257: type = class_type @DefaultOptionalStorage, @DefaultOptionalStorage(%facet_value.ccb) [concrete]
// CHECK:STDOUT:   %OptionalStorage.impl_witness.cf8: <witness> = impl_witness imports.%OptionalStorage.impl_witness_table.80d, @T.as_type.as.OptionalStorage.impl(%facet_value.ccb) [concrete]
// CHECK:STDOUT:   %OptionalStorage.facet.381: %OptionalStorage.type = facet_value %ValueType, (%OptionalStorage.impl_witness.cf8) [concrete]
// CHECK:STDOUT:   %Optional.dda: type = class_type @Optional, @Optional(%OptionalStorage.facet.381) [concrete]
// CHECK:STDOUT:   %specific_impl_fn.24b: <specific function> = specific_impl_function %impl.elem3.7a5, @Iterate.WithSelf.Next(%Iterate.facet.b38) [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.type.b33: type = fn_type @Optional.HasValue, @Optional(%OptionalStorage.facet.381) [concrete]
// CHECK:STDOUT:   %Optional.HasValue.1a5: %Optional.HasValue.type.b33 = struct_value () [concrete]
// CHECK:STDOUT:   %Optional.Get.type.341: type = fn_type @Optional.Get, @Optional(%OptionalStorage.facet.381) [concrete]
// CHECK:STDOUT:   %Optional.Get.733: %Optional.Get.type.341 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.facet.306: %Destroy.type = facet_value %ValueType, (%custom_witness.fef) [concrete]
// CHECK:STDOUT:   %MaybeUnformed.669: type = class_type @MaybeUnformed, @MaybeUnformed(%Destroy.facet.306) [concrete]
// CHECK:STDOUT:   %.b36: type = maybe_unformed_type %ValueType [concrete]
// CHECK:STDOUT:   %struct_type.value.has_value.5a1: type = struct_type {.value: %MaybeUnformed.669, .has_value: bool} [concrete]
// CHECK:STDOUT:   %Optional.HasValue.specific_fn.39a: <specific function> = specific_function %Optional.HasValue.1a5, @Optional.HasValue(%OptionalStorage.facet.381) [concrete]
// CHECK:STDOUT:   %Optional.Get.specific_fn.cd3: <specific function> = specific_function %Optional.Get.733, @Optional.Get(%OptionalStorage.facet.381) [concrete]
// CHECK:STDOUT:   %operator_PlusEqual__carbon_thunk.type: type = fn_type @operator_PlusEqual__carbon_thunk [concrete]
// CHECK:STDOUT:   %operator_PlusEqual__carbon_thunk: %operator_PlusEqual__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.7: type = fn_type @Destroy.Op.loc38_31.6 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.7: %Destroy.Op.type.1d8f74.7 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.198: <witness> = lookup_impl_witness %impl.elem1.834, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.9d6: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%impl.elem1.834) [symbolic]
// CHECK:STDOUT:   %.57c: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.9d6, %impl.elem1.834 [symbolic]
// CHECK:STDOUT:   %impl.elem0.152: %.57c = impl_witness_access %Destroy.lookup_impl_witness.198, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.9fb: <specific function> = specific_impl_function %impl.elem0.152, @Destroy.WithSelf.Op(%impl.elem1.834) [symbolic]
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
// CHECK:STDOUT:   %custom_witness.01a: <witness> = custom_witness (%Iterator.Op.549055.3), @Inc [concrete]
// CHECK:STDOUT:   %Equal.type: type = fn_type @Equal [concrete]
// CHECK:STDOUT:   %Equal: %Equal.type = struct_value () [concrete]
// CHECK:STDOUT:   %NotEqual.type: type = fn_type @NotEqual [concrete]
// CHECK:STDOUT:   %NotEqual: %NotEqual.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.c66: <witness> = custom_witness (%Equal, %NotEqual), @EqWith, @EqWith(%Iterator) [concrete]
// CHECK:STDOUT:   %facet_value.32f: %CppRange_where.type.d682d6.1 = facet_value %MutableAndConstRange, (%custom_witness.fef, %custom_witness.c19, %custom_witness.a40, %custom_witness.04a, %custom_witness.dd5, %custom_witness.01a, %custom_witness.c66) [concrete]
// CHECK:STDOUT:   %tuple.type.56e: type = tuple_type (%Iterator, %Iterator) [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.8: type = fn_type @Destroy.Op.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.8: %Destroy.Op.type.1d8f74.8 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.df9cc1.8: <witness> = custom_witness (%Destroy.Op.1a2547.8), @Destroy [concrete]
// CHECK:STDOUT:   %Destroy.facet.7aa: %Destroy.type = facet_value %tuple.type.56e, (%custom_witness.df9cc1.8) [concrete]
// CHECK:STDOUT:   %Iterate.impl_witness.0a2: <witness> = impl_witness imports.%Iterate.impl_witness_table.478, @T.as_type.as.Iterate.impl(%facet_value.32f) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.type.b14: type = fn_type @T.as_type.as.Iterate.impl.NewCursor, @T.as_type.as.Iterate.impl(%facet_value.32f) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.01c: %T.as_type.as.Iterate.impl.NewCursor.type.b14 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.type.e03: type = fn_type @T.as_type.as.Iterate.impl.Next, @T.as_type.as.Iterate.impl(%facet_value.32f) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.ce6: %T.as_type.as.Iterate.impl.Next.type.e03 = struct_value () [concrete]
// CHECK:STDOUT:   %Iterate.facet.850: %Iterate.type = facet_value %MutableAndConstRange, (%Iterate.impl_witness.0a2) [concrete]
// CHECK:STDOUT:   %facet_value.385: %Iterate_where.type.f9d = facet_value %MutableAndConstRange, (%Iterate.impl_witness.0a2) [concrete]
// CHECK:STDOUT:   %r.param_patt.8bb: %pattern_type.394 = value_param_pattern [concrete]
// CHECK:STDOUT:   %r.patt.ac4: %pattern_type.394 = at_binding_pattern r, %r.param_patt.8bb [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.cb3: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.850) [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.6f7: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.850) [concrete]
// CHECK:STDOUT:   %.83f: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.cb3, %Iterate.facet.850 [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.specific_fn: <specific function> = specific_function %T.as_type.as.Iterate.impl.NewCursor.01c, @T.as_type.as.Iterate.impl.NewCursor(%facet_value.32f) [concrete]
// CHECK:STDOUT:   %ptr.484: type = ptr_type %tuple.type.56e [concrete]
// CHECK:STDOUT:   %.1b3: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.6f7, %Iterate.facet.850 [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.specific_fn: <specific function> = specific_function %T.as_type.as.Iterate.impl.Next.ce6, @T.as_type.as.Iterate.impl.Next(%facet_value.32f) [concrete]
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
// CHECK:STDOUT:   %Core.import_ref.13b: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.None.type (%T.as_type.as.OptionalStorage.impl.None.type.d4d) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.None (constants.%T.as_type.as.OptionalStorage.impl.None.d64)]
// CHECK:STDOUT:   %Core.import_ref.c43f: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Some.type (%T.as_type.as.OptionalStorage.impl.Some.type.7eb) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Some (constants.%T.as_type.as.OptionalStorage.impl.Some.2b4)]
// CHECK:STDOUT:   %Core.import_ref.61e: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Has.type (%T.as_type.as.OptionalStorage.impl.Has.type.eb9) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Has (constants.%T.as_type.as.OptionalStorage.impl.Has.c7e)]
// CHECK:STDOUT:   %Core.import_ref.ce2e: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Get.type (%T.as_type.as.OptionalStorage.impl.Get.type.d3c) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Get (constants.%T.as_type.as.OptionalStorage.impl.Get.da2)]
// CHECK:STDOUT:   %Core.import_ref.8fc = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], unloaded
// CHECK:STDOUT:   %OptionalStorage.impl_witness_table.80d = impl_witness_table (%Core.import_ref.0c6, %Core.import_ref.13b, %Core.import_ref.c43f, %Core.import_ref.61e, %Core.import_ref.ce2e, %Core.import_ref.8fc), @T.as_type.as.OptionalStorage.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.eef: %facet_type.f48 = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.Iterate.impl.%facet_value (constants.%facet_value.f93)]
// CHECK:STDOUT:   %Core.import_ref.fcc: %Destroy.type = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.Iterate.impl.%Destroy.facet (constants.%Destroy.facet.9cd)]
// CHECK:STDOUT:   %Core.import_ref.f50: @T.as_type.as.Iterate.impl.%T.as_type.as.Iterate.impl.NewCursor.type (%T.as_type.as.Iterate.impl.NewCursor.type.b5d) = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.Iterate.impl.%T.as_type.as.Iterate.impl.NewCursor (constants.%T.as_type.as.Iterate.impl.NewCursor.21f)]
// CHECK:STDOUT:   %Core.import_ref.590: @T.as_type.as.Iterate.impl.%T.as_type.as.Iterate.impl.Next.type (%T.as_type.as.Iterate.impl.Next.type.5dc) = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.Iterate.impl.%T.as_type.as.Iterate.impl.Next (constants.%T.as_type.as.Iterate.impl.Next.b73)]
// CHECK:STDOUT:   %Iterate.impl_witness_table.478 = impl_witness_table (%Core.import_ref.eef, %Core.import_ref.fcc, %Core.import_ref.f50, %Core.import_ref.590), @T.as_type.as.Iterate.impl [concrete]
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
// CHECK:STDOUT:   %operator_PlusEqual__carbon_thunk.decl: %operator_PlusEqual__carbon_thunk.type = fn_decl @operator_PlusEqual__carbon_thunk [concrete = constants.%operator_PlusEqual__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %MutableAndConstRange.decl: type = class_decl @MutableAndConstRange [concrete = constants.%MutableAndConstRange] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @TestIterate(%R.loc34_17.2: %Iterate_where.type.f9d) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Iterate.lookup_impl_witness: <witness> = lookup_impl_witness %R.loc34_17.1, @Iterate [symbolic = %Iterate.lookup_impl_witness (constants.%Iterate.lookup_impl_witness.173)]
// CHECK:STDOUT:   %Iterate.facet.loc38_31.5: %Iterate.type = facet_value %R.as_type.loc34_73.1, (%Iterate.lookup_impl_witness) [symbolic = %Iterate.facet.loc38_31.5 (constants.%Iterate.facet.b38)]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.loc38_31.5) [symbolic = %Iterate.WithSelf.NewCursor.type (constants.%Iterate.WithSelf.NewCursor.type.5cd)]
// CHECK:STDOUT:   %.loc38_31.20: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type, %Iterate.facet.loc38_31.5 [symbolic = %.loc38_31.20 (constants.%.592)]
// CHECK:STDOUT:   %impl.elem2.loc38_31.2: @TestIterate.%.loc38_31.20 (%.592) = impl_witness_access %Iterate.lookup_impl_witness, element2 [symbolic = %impl.elem2.loc38_31.2 (constants.%impl.elem2.ae2)]
// CHECK:STDOUT:   %specific_impl_fn.loc38_31.4: <specific function> = specific_impl_function %impl.elem2.loc38_31.2, @Iterate.WithSelf.NewCursor(%Iterate.facet.loc38_31.5) [symbolic = %specific_impl_fn.loc38_31.4 (constants.%specific_impl_fn.06e)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %ptr: type = ptr_type %as_type [symbolic = %ptr (constants.%ptr.d27)]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.loc38_31.5) [symbolic = %Iterate.WithSelf.Next.type (constants.%Iterate.WithSelf.Next.type.d51)]
// CHECK:STDOUT:   %.loc38_31.21: type = fn_type_with_self_type %Iterate.WithSelf.Next.type, %Iterate.facet.loc38_31.5 [symbolic = %.loc38_31.21 (constants.%.524)]
// CHECK:STDOUT:   %impl.elem3.loc38_31.2: @TestIterate.%.loc38_31.21 (%.524) = impl_witness_access %Iterate.lookup_impl_witness, element3 [symbolic = %impl.elem3.loc38_31.2 (constants.%impl.elem3.7a5)]
// CHECK:STDOUT:   %specific_impl_fn.loc38_31.5: <specific function> = specific_impl_function %impl.elem3.loc38_31.2, @Iterate.WithSelf.Next(%Iterate.facet.loc38_31.5) [symbolic = %specific_impl_fn.loc38_31.5 (constants.%specific_impl_fn.24b)]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%impl.elem1) [symbolic = %Destroy.WithSelf.Op.type (constants.%Destroy.WithSelf.Op.type.9d6)]
// CHECK:STDOUT:   %.loc38_31.22: type = fn_type_with_self_type %Destroy.WithSelf.Op.type, %impl.elem1 [symbolic = %.loc38_31.22 (constants.%.57c)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %impl.elem1, @Destroy [symbolic = %Destroy.lookup_impl_witness (constants.%Destroy.lookup_impl_witness.198)]
// CHECK:STDOUT:   %impl.elem0.loc38_31.2: @TestIterate.%.loc38_31.22 (%.57c) = impl_witness_access %Destroy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc38_31.2 (constants.%impl.elem0.152)]
// CHECK:STDOUT:   %specific_impl_fn.loc38_31.6: <specific function> = specific_impl_function %impl.elem0.loc38_31.2, @Destroy.WithSelf.Op(%impl.elem1) [symbolic = %specific_impl_fn.loc38_31.6 (constants.%specific_impl_fn.9fb)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%r.param: @TestIterate.%R.as_type.loc34_73.1 (%R.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %i.patt: %pattern_type.e41 = value_binding_pattern i [concrete = constants.%i.patt]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %r.ref: @TestIterate.%R.as_type.loc34_73.1 (%R.as_type) = name_ref r, %r
// CHECK:STDOUT:     %impl.elem2.loc38_31.1: @TestIterate.%.loc38_31.20 (%.592) = impl_witness_access constants.%Iterate.lookup_impl_witness.173, element2 [symbolic = %impl.elem2.loc38_31.2 (constants.%impl.elem2.ae2)]
// CHECK:STDOUT:     %bound_method.loc38_31.1: <bound method> = bound_method %r.ref, %impl.elem2.loc38_31.1
// CHECK:STDOUT:     %Iterate.facet.loc38_31.1: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.173) [symbolic = %Iterate.facet.loc38_31.5 (constants.%Iterate.facet.b38)]
// CHECK:STDOUT:     %.loc38_31.1: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc38_31.1 [symbolic = %Iterate.facet.loc38_31.5 (constants.%Iterate.facet.b38)]
// CHECK:STDOUT:     %Iterate.facet.loc38_31.2: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.173) [symbolic = %Iterate.facet.loc38_31.5 (constants.%Iterate.facet.b38)]
// CHECK:STDOUT:     %.loc38_31.2: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc38_31.2 [symbolic = %Iterate.facet.loc38_31.5 (constants.%Iterate.facet.b38)]
// CHECK:STDOUT:     %specific_impl_fn.loc38_31.1: <specific function> = specific_impl_function %impl.elem2.loc38_31.1, @Iterate.WithSelf.NewCursor(constants.%Iterate.facet.b38) [symbolic = %specific_impl_fn.loc38_31.4 (constants.%specific_impl_fn.06e)]
// CHECK:STDOUT:     %bound_method.loc38_31.2: <bound method> = bound_method %r.ref, %specific_impl_fn.loc38_31.1
// CHECK:STDOUT:     %var: ref @TestIterate.%as_type (%as_type.e41) = var_storage invalid
// CHECK:STDOUT:     %Iterate.WithSelf.NewCursor.call: init @TestIterate.%as_type (%as_type.e41) to %var = call %bound_method.loc38_31.2(%r.ref)
// CHECK:STDOUT:     assign %var, %Iterate.WithSelf.NewCursor.call
// CHECK:STDOUT:     br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.next:
// CHECK:STDOUT:     %addr.loc38: @TestIterate.%ptr (%ptr.d27) = addr_of %var
// CHECK:STDOUT:     %impl.elem3.loc38_31.1: @TestIterate.%.loc38_31.21 (%.524) = impl_witness_access constants.%Iterate.lookup_impl_witness.173, element3 [symbolic = %impl.elem3.loc38_31.2 (constants.%impl.elem3.7a5)]
// CHECK:STDOUT:     %bound_method.loc38_31.3: <bound method> = bound_method %r.ref, %impl.elem3.loc38_31.1
// CHECK:STDOUT:     %Iterate.facet.loc38_31.3: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.173) [symbolic = %Iterate.facet.loc38_31.5 (constants.%Iterate.facet.b38)]
// CHECK:STDOUT:     %.loc38_31.3: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc38_31.3 [symbolic = %Iterate.facet.loc38_31.5 (constants.%Iterate.facet.b38)]
// CHECK:STDOUT:     %Iterate.facet.loc38_31.4: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.173) [symbolic = %Iterate.facet.loc38_31.5 (constants.%Iterate.facet.b38)]
// CHECK:STDOUT:     %.loc38_31.4: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc38_31.4 [symbolic = %Iterate.facet.loc38_31.5 (constants.%Iterate.facet.b38)]
// CHECK:STDOUT:     %.loc38_31.5: %Destroy.type = converted constants.%as_type.e41, constants.%impl.elem1.834 [symbolic = %impl.elem1 (constants.%impl.elem1.834)]
// CHECK:STDOUT:     %.loc38_31.6: %Destroy.type = converted constants.%as_type.e41, constants.%impl.elem1.834 [symbolic = %impl.elem1 (constants.%impl.elem1.834)]
// CHECK:STDOUT:     %specific_impl_fn.loc38_31.2: <specific function> = specific_impl_function %impl.elem3.loc38_31.1, @Iterate.WithSelf.Next(constants.%Iterate.facet.b38) [symbolic = %specific_impl_fn.loc38_31.5 (constants.%specific_impl_fn.24b)]
// CHECK:STDOUT:     %bound_method.loc38_31.4: <bound method> = bound_method %r.ref, %specific_impl_fn.loc38_31.2
// CHECK:STDOUT:     %.loc38_31.7: ref %Optional.dda = temporary_storage
// CHECK:STDOUT:     %Iterate.WithSelf.Next.call: init %Optional.dda to %.loc38_31.7 = call %bound_method.loc38_31.4(%r.ref, %addr.loc38)
// CHECK:STDOUT:     %.loc38_31.8: ref %Optional.dda = temporary %.loc38_31.7, %Iterate.WithSelf.Next.call
// CHECK:STDOUT:     %.loc38_31.9: %Optional.HasValue.type.b33 = specific_constant imports.%Core.import_ref.bb6, @Optional(constants.%OptionalStorage.facet.381) [concrete = constants.%Optional.HasValue.1a5]
// CHECK:STDOUT:     %HasValue.ref: %Optional.HasValue.type.b33 = name_ref HasValue, %.loc38_31.9 [concrete = constants.%Optional.HasValue.1a5]
// CHECK:STDOUT:     %Optional.HasValue.bound: <bound method> = bound_method %.loc38_31.8, %HasValue.ref
// CHECK:STDOUT:     %Optional.HasValue.specific_fn: <specific function> = specific_function %HasValue.ref, @Optional.HasValue(constants.%OptionalStorage.facet.381) [concrete = constants.%Optional.HasValue.specific_fn.39a]
// CHECK:STDOUT:     %bound_method.loc38_31.5: <bound method> = bound_method %.loc38_31.8, %Optional.HasValue.specific_fn
// CHECK:STDOUT:     %.loc38_31.10: %Optional.dda = acquire_value %.loc38_31.8
// CHECK:STDOUT:     %Optional.HasValue.call: init bool = call %bound_method.loc38_31.5(%.loc38_31.10)
// CHECK:STDOUT:     %.loc38_31.11: bool = value_of_initializer %Optional.HasValue.call
// CHECK:STDOUT:     %.loc38_31.12: bool = converted %Optional.HasValue.call, %.loc38_31.11
// CHECK:STDOUT:     if %.loc38_31.12 br !for.body else br !for.done
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.body:
// CHECK:STDOUT:     %.loc38_31.13: %Optional.Get.type.341 = specific_constant imports.%Core.import_ref.daba, @Optional(constants.%OptionalStorage.facet.381) [concrete = constants.%Optional.Get.733]
// CHECK:STDOUT:     %Get.ref: %Optional.Get.type.341 = name_ref Get, %.loc38_31.13 [concrete = constants.%Optional.Get.733]
// CHECK:STDOUT:     %Optional.Get.bound: <bound method> = bound_method %.loc38_31.8, %Get.ref
// CHECK:STDOUT:     %Optional.Get.specific_fn: <specific function> = specific_function %Get.ref, @Optional.Get(constants.%OptionalStorage.facet.381) [concrete = constants.%Optional.Get.specific_fn.cd3]
// CHECK:STDOUT:     %bound_method.loc38_31.6: <bound method> = bound_method %.loc38_31.8, %Optional.Get.specific_fn
// CHECK:STDOUT:     %.loc38_31.14: ref %ValueType = temporary_storage
// CHECK:STDOUT:     %.loc38_31.15: %Optional.dda = acquire_value %.loc38_31.8
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
// CHECK:STDOUT:     %ValueType.Op.bound.loc38: <bound method> = bound_method %.loc38_31.16, constants.%ValueType.Op.4c94b9.1
// CHECK:STDOUT:     %Op.ref.loc38: %ValueType.cpp_destructor.type = name_ref Op, imports.%ValueType.cpp_destructor.decl [concrete = constants.%ValueType.cpp_destructor]
// CHECK:STDOUT:     %ValueType.cpp_destructor.bound.loc38: <bound method> = bound_method %.loc38_31.16, %Op.ref.loc38
// CHECK:STDOUT:     %ValueType.cpp_destructor.call.loc38: init %empty_tuple.type = call %ValueType.cpp_destructor.bound.loc38(%.loc38_31.16)
// CHECK:STDOUT:     %Destroy.Op.bound.loc38_31.1: <bound method> = bound_method %.loc38_31.8, constants.%Destroy.Op.1a2547.7
// CHECK:STDOUT:     %Destroy.Op.call.loc38_31.1: init %empty_tuple.type = call %Destroy.Op.bound.loc38_31.1(%.loc38_31.8)
// CHECK:STDOUT:     br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.done:
// CHECK:STDOUT:     %Destroy.Op.bound.loc38_31.2: <bound method> = bound_method %.loc38_31.8, constants.%Destroy.Op.1a2547.7
// CHECK:STDOUT:     %Destroy.Op.call.loc38_31.2: init %empty_tuple.type = call %Destroy.Op.bound.loc38_31.2(%.loc38_31.8)
// CHECK:STDOUT:     %impl.elem0.loc38_31.1: @TestIterate.%.loc38_31.22 (%.57c) = impl_witness_access constants.%Destroy.lookup_impl_witness.198, element0 [symbolic = %impl.elem0.loc38_31.2 (constants.%impl.elem0.152)]
// CHECK:STDOUT:     %bound_method.loc38_31.7: <bound method> = bound_method %var, %impl.elem0.loc38_31.1
// CHECK:STDOUT:     %.loc38_31.18: %Destroy.type = converted constants.%as_type.e41, constants.%impl.elem1.834 [symbolic = %impl.elem1 (constants.%impl.elem1.834)]
// CHECK:STDOUT:     %.loc38_31.19: %Destroy.type = converted constants.%as_type.e41, constants.%impl.elem1.834 [symbolic = %impl.elem1 (constants.%impl.elem1.834)]
// CHECK:STDOUT:     %specific_impl_fn.loc38_31.3: <specific function> = specific_impl_function %impl.elem0.loc38_31.1, @Destroy.WithSelf.Op(constants.%impl.elem1.834) [symbolic = %specific_impl_fn.loc38_31.6 (constants.%specific_impl_fn.9fb)]
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
// CHECK:STDOUT: fn @Destroy.Op.loc38_31.6(%self.param: ref %Optional.dda) {
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
// CHECK:STDOUT:   %impl.elem2: %.83f = impl_witness_access constants.%Iterate.impl_witness.0a2, element2 [concrete = constants.%T.as_type.as.Iterate.impl.NewCursor.01c]
// CHECK:STDOUT:   %bound_method.loc80_32.1: <bound method> = bound_method %mc.ref, %impl.elem2
// CHECK:STDOUT:   %facet_value.loc80_32.1: %CppRange_where.type.d682d6.1 = facet_value constants.%MutableAndConstRange, (constants.%custom_witness.fef, constants.%custom_witness.c19, constants.%custom_witness.a40, constants.%custom_witness.04a, constants.%custom_witness.dd5, constants.%custom_witness.01a, constants.%custom_witness.c66) [concrete = constants.%facet_value.32f]
// CHECK:STDOUT:   %.loc80_32.1: %CppRange_where.type.d682d6.1 = converted constants.%MutableAndConstRange, %facet_value.loc80_32.1 [concrete = constants.%facet_value.32f]
// CHECK:STDOUT:   %facet_value.loc80_32.2: %CppRange_where.type.d682d6.1 = facet_value constants.%MutableAndConstRange, (constants.%custom_witness.fef, constants.%custom_witness.c19, constants.%custom_witness.a40, constants.%custom_witness.04a, constants.%custom_witness.dd5, constants.%custom_witness.01a, constants.%custom_witness.c66) [concrete = constants.%facet_value.32f]
// CHECK:STDOUT:   %.loc80_32.2: %CppRange_where.type.d682d6.1 = converted constants.%MutableAndConstRange, %facet_value.loc80_32.2 [concrete = constants.%facet_value.32f]
// CHECK:STDOUT:   %specific_fn.loc80_32.1: <specific function> = specific_function %impl.elem2, @T.as_type.as.Iterate.impl.NewCursor(constants.%facet_value.32f) [concrete = constants.%T.as_type.as.Iterate.impl.NewCursor.specific_fn]
// CHECK:STDOUT:   %bound_method.loc80_32.2: <bound method> = bound_method %mc.ref, %specific_fn.loc80_32.1
// CHECK:STDOUT:   %var: ref %tuple.type.56e = var_storage invalid
// CHECK:STDOUT:   %.loc80_30.1: %MutableAndConstRange = acquire_value %mc.ref
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.call: init %tuple.type.56e to %var = call %bound_method.loc80_32.2(%.loc80_30.1)
// CHECK:STDOUT:   assign %var, %T.as_type.as.Iterate.impl.NewCursor.call
// CHECK:STDOUT:   br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT: !for.next:
// CHECK:STDOUT:   %addr.loc80: %ptr.484 = addr_of %var
// CHECK:STDOUT:   %impl.elem3: %.1b3 = impl_witness_access constants.%Iterate.impl_witness.0a2, element3 [concrete = constants.%T.as_type.as.Iterate.impl.Next.ce6]
// CHECK:STDOUT:   %bound_method.loc80_32.3: <bound method> = bound_method %mc.ref, %impl.elem3
// CHECK:STDOUT:   %facet_value.loc80_32.3: %CppRange_where.type.d682d6.1 = facet_value constants.%MutableAndConstRange, (constants.%custom_witness.fef, constants.%custom_witness.c19, constants.%custom_witness.a40, constants.%custom_witness.04a, constants.%custom_witness.dd5, constants.%custom_witness.01a, constants.%custom_witness.c66) [concrete = constants.%facet_value.32f]
// CHECK:STDOUT:   %.loc80_32.3: %CppRange_where.type.d682d6.1 = converted constants.%MutableAndConstRange, %facet_value.loc80_32.3 [concrete = constants.%facet_value.32f]
// CHECK:STDOUT:   %facet_value.loc80_32.4: %CppRange_where.type.d682d6.1 = facet_value constants.%MutableAndConstRange, (constants.%custom_witness.fef, constants.%custom_witness.c19, constants.%custom_witness.a40, constants.%custom_witness.04a, constants.%custom_witness.dd5, constants.%custom_witness.01a, constants.%custom_witness.c66) [concrete = constants.%facet_value.32f]
// CHECK:STDOUT:   %.loc80_32.4: %CppRange_where.type.d682d6.1 = converted constants.%MutableAndConstRange, %facet_value.loc80_32.4 [concrete = constants.%facet_value.32f]
// CHECK:STDOUT:   %specific_fn.loc80_32.2: <specific function> = specific_function %impl.elem3, @T.as_type.as.Iterate.impl.Next(constants.%facet_value.32f) [concrete = constants.%T.as_type.as.Iterate.impl.Next.specific_fn]
// CHECK:STDOUT:   %bound_method.loc80_32.4: <bound method> = bound_method %mc.ref, %specific_fn.loc80_32.2
// CHECK:STDOUT:   %.loc80_32.5: ref %Optional.dda = temporary_storage
// CHECK:STDOUT:   %.loc80_30.2: %MutableAndConstRange = acquire_value %mc.ref
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.call: init %Optional.dda to %.loc80_32.5 = call %bound_method.loc80_32.4(%.loc80_30.2, %addr.loc80)
// CHECK:STDOUT:   %.loc80_32.6: ref %Optional.dda = temporary %.loc80_32.5, %T.as_type.as.Iterate.impl.Next.call
// CHECK:STDOUT:   %.loc80_32.7: %Optional.HasValue.type.b33 = specific_constant imports.%Core.import_ref.bb6, @Optional(constants.%OptionalStorage.facet.381) [concrete = constants.%Optional.HasValue.1a5]
// CHECK:STDOUT:   %HasValue.ref: %Optional.HasValue.type.b33 = name_ref HasValue, %.loc80_32.7 [concrete = constants.%Optional.HasValue.1a5]
// CHECK:STDOUT:   %Optional.HasValue.bound: <bound method> = bound_method %.loc80_32.6, %HasValue.ref
// CHECK:STDOUT:   %Optional.HasValue.specific_fn: <specific function> = specific_function %HasValue.ref, @Optional.HasValue(constants.%OptionalStorage.facet.381) [concrete = constants.%Optional.HasValue.specific_fn.39a]
// CHECK:STDOUT:   %bound_method.loc80_32.5: <bound method> = bound_method %.loc80_32.6, %Optional.HasValue.specific_fn
// CHECK:STDOUT:   %.loc80_32.8: %Optional.dda = acquire_value %.loc80_32.6
// CHECK:STDOUT:   %Optional.HasValue.call: init bool = call %bound_method.loc80_32.5(%.loc80_32.8)
// CHECK:STDOUT:   %.loc80_32.9: bool = value_of_initializer %Optional.HasValue.call
// CHECK:STDOUT:   %.loc80_32.10: bool = converted %Optional.HasValue.call, %.loc80_32.9
// CHECK:STDOUT:   if %.loc80_32.10 br !for.body else br !for.done
// CHECK:STDOUT:
// CHECK:STDOUT: !for.body:
// CHECK:STDOUT:   %.loc80_32.11: %Optional.Get.type.341 = specific_constant imports.%Core.import_ref.daba, @Optional(constants.%OptionalStorage.facet.381) [concrete = constants.%Optional.Get.733]
// CHECK:STDOUT:   %Get.ref: %Optional.Get.type.341 = name_ref Get, %.loc80_32.11 [concrete = constants.%Optional.Get.733]
// CHECK:STDOUT:   %Optional.Get.bound: <bound method> = bound_method %.loc80_32.6, %Get.ref
// CHECK:STDOUT:   %Optional.Get.specific_fn: <specific function> = specific_function %Get.ref, @Optional.Get(constants.%OptionalStorage.facet.381) [concrete = constants.%Optional.Get.specific_fn.cd3]
// CHECK:STDOUT:   %bound_method.loc80_32.6: <bound method> = bound_method %.loc80_32.6, %Optional.Get.specific_fn
// CHECK:STDOUT:   %.loc80_32.12: ref %ValueType = temporary_storage
// CHECK:STDOUT:   %.loc80_32.13: %Optional.dda = acquire_value %.loc80_32.6
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
// CHECK:STDOUT:   %ValueType.Op.bound.loc80: <bound method> = bound_method %.loc80_32.14, constants.%ValueType.Op.4c94b9.1
// CHECK:STDOUT:   %Op.ref.loc80: %ValueType.cpp_destructor.type = name_ref Op, imports.%ValueType.cpp_destructor.decl [concrete = constants.%ValueType.cpp_destructor]
// CHECK:STDOUT:   %ValueType.cpp_destructor.bound.loc80: <bound method> = bound_method %.loc80_32.14, %Op.ref.loc80
// CHECK:STDOUT:   %ValueType.cpp_destructor.call.loc80: init %empty_tuple.type = call %ValueType.cpp_destructor.bound.loc80(%.loc80_32.14)
// CHECK:STDOUT:   %Destroy.Op.bound.loc80_32.1: <bound method> = bound_method %.loc80_32.6, constants.%Destroy.Op.1a2547.7
// CHECK:STDOUT:   %Destroy.Op.call.loc80_32.1: init %empty_tuple.type = call %Destroy.Op.bound.loc80_32.1(%.loc80_32.6)
// CHECK:STDOUT:   br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT: !for.done:
// CHECK:STDOUT:   %Destroy.Op.bound.loc80_32.2: <bound method> = bound_method %.loc80_32.6, constants.%Destroy.Op.1a2547.7
// CHECK:STDOUT:   %Destroy.Op.call.loc80_32.2: init %empty_tuple.type = call %Destroy.Op.bound.loc80_32.2(%.loc80_32.6)
// CHECK:STDOUT:   %Destroy.Op.bound.loc80_32.3: <bound method> = bound_method %var, constants.%Destroy.Op.1a2547.8
// CHECK:STDOUT:   %Destroy.Op.call.loc80_32.3: init %empty_tuple.type = call %Destroy.Op.bound.loc80_32.3(%var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestIterate(constants.%R) {
// CHECK:STDOUT:   %R.patt.loc34_17.2 => constants.%R.patt
// CHECK:STDOUT:   %R.loc34_17.1 => constants.%R
// CHECK:STDOUT:   %R.as_type.loc34_73.1 => constants.%R.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.17d
// CHECK:STDOUT:   %r.param_patt.loc34_71.2 => constants.%r.param_patt.90c
// CHECK:STDOUT:   %r.patt.loc34_71.2 => constants.%r.patt.d6c
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestIterate(constants.%facet_value.385) {
// CHECK:STDOUT:   %R.patt.loc34_17.2 => constants.%R.patt
// CHECK:STDOUT:   %R.loc34_17.1 => constants.%facet_value.385
// CHECK:STDOUT:   %R.as_type.loc34_73.1 => constants.%MutableAndConstRange
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.394
// CHECK:STDOUT:   %r.param_patt.loc34_71.2 => constants.%r.param_patt.8bb
// CHECK:STDOUT:   %r.patt.loc34_71.2 => constants.%r.patt.ac4
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc34 => constants.%complete_type.357
// CHECK:STDOUT:   %Iterate.lookup_impl_witness => constants.%Iterate.impl_witness.0a2
// CHECK:STDOUT:   %Iterate.facet.loc38_31.5 => constants.%Iterate.facet.850
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type => constants.%Iterate.WithSelf.NewCursor.type.cb3
// CHECK:STDOUT:   %.loc38_31.20 => constants.%.83f
// CHECK:STDOUT:   %impl.elem2.loc38_31.2 => constants.%T.as_type.as.Iterate.impl.NewCursor.01c
// CHECK:STDOUT:   %specific_impl_fn.loc38_31.4 => constants.%T.as_type.as.Iterate.impl.NewCursor.specific_fn
// CHECK:STDOUT:   %impl.elem1 => constants.%Destroy.facet.7aa
// CHECK:STDOUT:   %as_type => constants.%tuple.type.56e
// CHECK:STDOUT:   %require_complete.1 => constants.%complete_type.1b5
// CHECK:STDOUT:   %ptr => constants.%ptr.484
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type => constants.%Iterate.WithSelf.Next.type.6f7
// CHECK:STDOUT:   %.loc38_31.21 => constants.%.1b3
// CHECK:STDOUT:   %impl.elem3.loc38_31.2 => constants.%T.as_type.as.Iterate.impl.Next.ce6
// CHECK:STDOUT:   %specific_impl_fn.loc38_31.5 => constants.%T.as_type.as.Iterate.impl.Next.specific_fn
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.b7b
// CHECK:STDOUT:   %.loc38_31.22 => constants.%.1ad
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.df9cc1.8
// CHECK:STDOUT:   %impl.elem0.loc38_31.2 => constants.%Destroy.Op.1a2547.8
// CHECK:STDOUT:   %specific_impl_fn.loc38_31.6 => constants.%Destroy.Op.1a2547.8
// CHECK:STDOUT: }
// CHECK:STDOUT:
