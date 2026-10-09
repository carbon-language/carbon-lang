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
// CHECK:STDERR: fail_todo_adl_returns_same_types_mutable.carbon: error: value expression passed to reference parameter [ValueForRefParam]

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
  // CHECK:STDERR: fail_todo_adl_returns_same_types_mutable.carbon:[[@LINE+6]]:34: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR:   friend auto begin(MutableRange&) -> Iterator;
  // CHECK:STDERR:                                  ^
  // CHECK:STDERR: {{.*}}/prelude/iterate.carbon:41:3: note: while building thunk to match the signature of this function [ThunkSignature]
  // CHECK:STDERR:   fn Begin(self) -> Iterator;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~
  friend auto begin(MutableRange&) -> Iterator;
  friend auto end(MutableRange&) -> Iterator;
};
} // namespace N
''';

// CHECK:STDERR: fail_todo_adl_returns_same_types_mutable.carbon:[[@LINE+15]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
// CHECK:STDERR: fn TestIterate[R: Core.Iterate where .ElementType = f64](r: R) {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_todo_adl_returns_same_types_mutable.carbon: error: value expression passed to reference parameter [ValueForRefParam]
// CHECK:STDERR: fail_todo_adl_returns_same_types_mutable.carbon:[[@LINE-10]]:32: note: initializing function parameter [InCallToFunctionParam]
// CHECK:STDERR:   friend auto end(MutableRange&) -> Iterator;
// CHECK:STDERR:                                ^
// CHECK:STDERR: {{.*}}/prelude/iterate.carbon:42:3: note: while building thunk to match the signature of this function [ThunkSignature]
// CHECK:STDERR:   fn End(self) -> Sentinel;
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_todo_adl_returns_same_types_mutable.carbon:[[@LINE+4]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
// CHECK:STDERR: fn TestIterate[R: Core.Iterate where .ElementType = f64](r: R) {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn TestIterate[R: Core.Iterate where .ElementType = f64](r: R) {
  var sum: f64 = 0.0;

  //@dump-sem-ir-begin
  for (i: f64 in r) {
    sum += i;
  }
  //@dump-sem-ir-end
}

fn TestIterateDriver(var m: Cpp.N.MutableRange) {
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
// CHECK:STDERR: fail_todo_adl_returns_different_types_mutable.carbon: error: value expression passed to reference parameter [ValueForRefParam]

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
  // CHECK:STDERR: fail_todo_adl_returns_different_types_mutable.carbon:[[@LINE+6]]:34: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR:   friend auto begin(MutableRange&) -> Iterator;
  // CHECK:STDERR:                                  ^
  // CHECK:STDERR: {{.*}}/prelude/iterate.carbon:41:3: note: while building thunk to match the signature of this function [ThunkSignature]
  // CHECK:STDERR:   fn Begin(self) -> Iterator;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~
  friend auto begin(MutableRange&) -> Iterator;
  friend auto end(MutableRange&) -> Sentinel;
};
}
''';

// CHECK:STDERR: fail_todo_adl_returns_different_types_mutable.carbon:[[@LINE+15]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
// CHECK:STDERR: fn TestIterate[R: Core.Iterate where .ElementType = Cpp.DiffMutableADL.ValueType](r: R) {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_todo_adl_returns_different_types_mutable.carbon: error: value expression passed to reference parameter [ValueForRefParam]
// CHECK:STDERR: fail_todo_adl_returns_different_types_mutable.carbon:[[@LINE-10]]:32: note: initializing function parameter [InCallToFunctionParam]
// CHECK:STDERR:   friend auto end(MutableRange&) -> Sentinel;
// CHECK:STDERR:                                ^
// CHECK:STDERR: {{.*}}/prelude/iterate.carbon:42:3: note: while building thunk to match the signature of this function [ThunkSignature]
// CHECK:STDERR:   fn End(self) -> Sentinel;
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_todo_adl_returns_different_types_mutable.carbon:[[@LINE+4]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
// CHECK:STDERR: fn TestIterate[R: Core.Iterate where .ElementType = Cpp.DiffMutableADL.ValueType](r: R) {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn TestIterate[R: Core.Iterate where .ElementType = Cpp.DiffMutableADL.ValueType](r: R) {
  var sum: Cpp.DiffMutableADL.ValueType = Cpp.DiffMutableADL.ValueType.ValueType();

  //@dump-sem-ir-begin
  for (i: Cpp.DiffMutableADL.ValueType in r) {
    sum += i;
  }
  //@dump-sem-ir-end
}

fn TestIterateDriver(var m: Cpp.DiffMutableADL.MutableRange) {
  // CHECK:STDERR: fail_todo_adl_returns_different_types_mutable.carbon:[[@LINE+14]]:3: error: semantics TODO: `Rewriting operator== using operator== is not supported` [SemanticsTodo]
  // CHECK:STDERR:   TestIterate(m);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_adl_returns_different_types_mutable.carbon:[[@LINE-14]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn TestIterate[R: Core.Iterate where .ElementType = Cpp.DiffMutableADL.ValueType](r: R) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_todo_adl_returns_different_types_mutable.carbon:[[@LINE+7]]:3: error: cannot convert type `Cpp.DiffMutableADL.MutableRange` into type implementing `Core.Iterate where .(Core.Iterate.ElementType) = Cpp.DiffMutableADL.ValueType as Core.Destroy & Core.Copy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   TestIterate(m);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_adl_returns_different_types_mutable.carbon:[[@LINE-21]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
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
// CHECK:STDERR: fail_todo_adl_overloads.carbon: error: value expression passed to reference parameter [ValueForRefParam]

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
    // CHECK:STDERR: fail_todo_adl_overloads.carbon:[[@LINE+6]]:44: note: initializing function parameter [InCallToFunctionParam]
    // CHECK:STDERR:     friend auto begin(MutableAndConstRange&) -> Iterator;
    // CHECK:STDERR:                                            ^
    // CHECK:STDERR: {{.*}}/prelude/iterate.carbon:41:3: note: while building thunk to match the signature of this function [ThunkSignature]
    // CHECK:STDERR:   fn Begin(self) -> Iterator;
    // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~
    friend auto begin(MutableAndConstRange&) -> Iterator;
    friend auto end(MutableAndConstRange&) -> Iterator;

    friend auto begin(const MutableAndConstRange&) -> ConstIterator;
    friend auto end(const MutableAndConstRange&) -> ConstIterator;
};
}
''';

// CHECK:STDERR: fail_todo_adl_overloads.carbon:[[@LINE+15]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
// CHECK:STDERR: fn TestIterate[R: Core.Iterate where .ElementType = Cpp.OverloadADL.ValueType](r: R) {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_todo_adl_overloads.carbon: error: value expression passed to reference parameter [ValueForRefParam]
// CHECK:STDERR: fail_todo_adl_overloads.carbon:[[@LINE-13]]:42: note: initializing function parameter [InCallToFunctionParam]
// CHECK:STDERR:     friend auto end(MutableAndConstRange&) -> Iterator;
// CHECK:STDERR:                                          ^
// CHECK:STDERR: {{.*}}/prelude/iterate.carbon:42:3: note: while building thunk to match the signature of this function [ThunkSignature]
// CHECK:STDERR:   fn End(self) -> Sentinel;
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_todo_adl_overloads.carbon:[[@LINE+4]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
// CHECK:STDERR: fn TestIterate[R: Core.Iterate where .ElementType = Cpp.OverloadADL.ValueType](r: R) {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn TestIterate[R: Core.Iterate where .ElementType = Cpp.OverloadADL.ValueType](r: R) {
  var sum: Cpp.OverloadADL.ValueType = Cpp.OverloadADL.ValueType.ValueType();

  //@dump-sem-ir-begin
  for (i: Cpp.OverloadADL.ValueType in r) {
    sum += i;
  }
  //@dump-sem-ir-end
}

fn TestIterateDriver(var mc: Cpp.OverloadADL.MutableAndConstRange) {
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

  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE+17]]:28: error: attempt to use a deleted function [CppInteropParseError]
  // CHECK:STDERR:   178 |   Test(begin_method_deleted);
  // CHECK:STDERR:       |                            ^
  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE-120]]:8: note: 'begin' has been explicitly marked deleted here [CppInteropParseNote]
  // CHECK:STDERR:    44 |   auto begin() = delete;
  // CHECK:STDERR:       |        ^
  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE-103]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn Test[R: Core.Iterate](unused r: R) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE+7]]:3: error: cannot convert type `Cpp.BeginMethodDeleted` into type implementing `Core.Iterate` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   Test(begin_method_deleted);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE-110]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn Test[R: Core.Iterate](unused r: R) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Test(begin_method_deleted);

  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE+17]]:26: error: attempt to use a deleted function [CppInteropParseError]
  // CHECK:STDERR:   197 |   Test(end_method_deleted);
  // CHECK:STDERR:       |                          ^
  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE-133]]:8: note: 'end' has been explicitly marked deleted here [CppInteropParseNote]
  // CHECK:STDERR:    50 |   auto end() = delete;
  // CHECK:STDERR:       |        ^
  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE-122]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn Test[R: Core.Iterate](unused r: R) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE+7]]:3: error: cannot convert type `Cpp.EndMethodDeleted` into type implementing `Core.Iterate` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   Test(end_method_deleted);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE-129]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn Test[R: Core.Iterate](unused r: R) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Test(end_method_deleted);

  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE+17]]:30: error: call to deleted function 'begin' [CppInteropParseError]
  // CHECK:STDERR:   216 |   Test(begin_function_deleted);
  // CHECK:STDERR:       |                              ^
  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE-148]]:15: note: candidate function has been explicitly deleted [CppInteropParseNote]
  // CHECK:STDERR:    54 |   friend auto begin(const BeginFunctionDeleted&) = delete;
  // CHECK:STDERR:       |               ^
  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE-141]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn Test[R: Core.Iterate](unused r: R) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE+7]]:3: error: cannot convert type `Cpp.BeginFunctionDeleted` into type implementing `Core.Iterate` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   Test(begin_function_deleted);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE-148]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn Test[R: Core.Iterate](unused r: R) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Test(begin_function_deleted);

  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE+17]]:28: error: call to deleted function 'end' [CppInteropParseError]
  // CHECK:STDERR:   235 |   Test(end_function_deleted);
  // CHECK:STDERR:       |                            ^
  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE-161]]:15: note: candidate function has been explicitly deleted [CppInteropParseNote]
  // CHECK:STDERR:    60 |   friend auto end(const EndFunctionDeleted&) = delete;
  // CHECK:STDERR:       |               ^
  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE-160]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn Test[R: Core.Iterate](unused r: R) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE+7]]:3: error: cannot convert type `Cpp.EndFunctionDeleted` into type implementing `Core.Iterate` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   Test(end_function_deleted);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE-167]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
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
// CHECK:STDOUT:   %Optional.Get.type.c64: type = fn_type @Optional.Get, @Optional(%T.220) [symbolic]
// CHECK:STDOUT:   %Optional.Get.bcf: %Optional.Get.type.c64 = struct_value () [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.type.49e: type = fn_type @Optional.HasValue, @Optional(%T.220) [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.c20: %Optional.HasValue.type.49e = struct_value () [symbolic]
// CHECK:STDOUT:   %T.283: %facet_type.f48 = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.74b: type = facet_access_type %T.283 [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.28a: <witness> = lookup_impl_witness %T.283, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.16e: %Destroy.type = facet_value %T.as_type.74b, (%Destroy.lookup_impl_witness.28a) [symbolic]
// CHECK:STDOUT:   %MaybeUnformed.9b7: type = class_type @MaybeUnformed, @MaybeUnformed(%Destroy.facet.16e) [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.faa: type = class_type @DefaultOptionalStorage, @DefaultOptionalStorage(%T.283) [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.elem.06a: type = unbound_element_type %DefaultOptionalStorage.faa, bool [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.elem.9db: type = unbound_element_type %DefaultOptionalStorage.faa, %MaybeUnformed.9b7 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Get.type.2f2: type = fn_type @T.as_type.as.OptionalStorage.impl.Get, @T.as_type.as.OptionalStorage.impl(%T.283) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Get.884: %T.as_type.as.OptionalStorage.impl.Get.type.2f2 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Has.type.d28: type = fn_type @T.as_type.as.OptionalStorage.impl.Has, @T.as_type.as.OptionalStorage.impl(%T.283) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Has.f29: %T.as_type.as.OptionalStorage.impl.Has.type.d28 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Some.type.760: type = fn_type @T.as_type.as.OptionalStorage.impl.Some, @T.as_type.as.OptionalStorage.impl(%T.283) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Some.268: %T.as_type.as.OptionalStorage.impl.Some.type.760 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.None.type.925: type = fn_type @T.as_type.as.OptionalStorage.impl.None, @T.as_type.as.OptionalStorage.impl(%T.283) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.None.f42: %T.as_type.as.OptionalStorage.impl.None.type.925 = struct_value () [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %N.fe9: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %Int.ec967b.1: type = class_type @Int, @Int(%N.fe9) [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.9d9: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.1: type = fn_type @Destroy.WithSelf.SelfDestruct.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.1: %Destroy.WithSelf.SelfDestruct.type.fbceb5.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.1: type = fn_type @Destroy.WithSelf.SubobjectDestroy.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.1: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.1: type = fn_type @Destroy.WithSelf.Op.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.1: %Destroy.WithSelf.Op.type.ef016f.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.f8f19d.1: <witness> = custom_witness (%Destroy.WithSelf.Op.403171.1, %Destroy.WithSelf.SubobjectDestroy.d01daf.1, %Destroy.WithSelf.SelfDestruct.db3fdb.1), @Destroy [concrete]
// CHECK:STDOUT:   %Destroy.facet.2b3: %Destroy.type = facet_value %i32, (%custom_witness.f8f19d.1) [concrete]
// CHECK:STDOUT:   %.Self.500: %Iterate.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Int.ec967b.2: type = class_type @Int, @Int(%To) [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.09b: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %CppRange.type: type = facet_type <@CppRange> [concrete]
// CHECK:STDOUT:   %.Self.e0b: %CppRange.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %CppRangeForIterate.lookup_impl_witness.50b: <witness> = lookup_impl_witness %.Self.e0b, @CppRangeForIterate [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.32d: type = impl_witness_access %CppRangeForIterate.lookup_impl_witness.50b, element0 [symbolic_self]
// CHECK:STDOUT:   %CppUnsafeDeref.lookup_impl_witness.851: <witness> = lookup_impl_witness %impl.elem0.32d, @CppUnsafeDeref [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.10d: type = impl_witness_access %CppUnsafeDeref.lookup_impl_witness.851, element0 [symbolic_self]
// CHECK:STDOUT:   %CppRange_where.type.d682d6.1: type = facet_type <@CppRange where %impl.elem0.10d impls @Copy> [concrete]
// CHECK:STDOUT:   %T.f45530.2: %CppRange_where.type.d682d6.1 = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.type.d4a: type = fn_type @T.as_type.as.Iterate.impl.Next, @T.as_type.as.Iterate.impl(%T.f45530.2) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.5a8: %T.as_type.as.Iterate.impl.Next.type.d4a = struct_value () [symbolic]
// CHECK:STDOUT:   %CppRangeForIterate.lookup_impl_witness.b581a8.2: <witness> = lookup_impl_witness %T.f45530.2, @CppRangeForIterate [symbolic]
// CHECK:STDOUT:   %impl.elem0.5f971c.2: type = impl_witness_access %CppRangeForIterate.lookup_impl_witness.b581a8.2, element0 [symbolic]
// CHECK:STDOUT:   %CppUnsafeDeref.lookup_impl_witness.87e543.2: <witness> = lookup_impl_witness %impl.elem0.5f971c.2, @CppUnsafeDeref [symbolic]
// CHECK:STDOUT:   %impl.elem0.493765.2: type = impl_witness_access %CppUnsafeDeref.lookup_impl_witness.87e543.2, element0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.type.8fb: type = fn_type @T.as_type.as.Iterate.impl.NewCursor, @T.as_type.as.Iterate.impl(%T.f45530.2) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.8a1: %T.as_type.as.Iterate.impl.NewCursor.type.8fb = struct_value () [symbolic]
// CHECK:STDOUT:   %impl.elem1.c2dc15.2: type = impl_witness_access %CppRangeForIterate.lookup_impl_witness.b581a8.2, element1 [symbolic]
// CHECK:STDOUT:   %tuple.type.1ba: type = tuple_type (%impl.elem0.5f971c.2, %impl.elem1.c2dc15.2) [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.812: <witness> = lookup_impl_witness %tuple.type.1ba, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.9cd: %Destroy.type = facet_value %tuple.type.1ba, (%Destroy.lookup_impl_witness.812) [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.35816c.2: <witness> = lookup_impl_witness %impl.elem0.493765.2, @Destroy [symbolic]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.167: <witness> = lookup_impl_witness %impl.elem0.493765.2, @Copy [symbolic]
// CHECK:STDOUT:   %facet_value.f93: %facet_type.f48 = facet_value %impl.elem0.493765.2, (%Destroy.lookup_impl_witness.35816c.2, %Copy.lookup_impl_witness.167) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.d84: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N.fe9) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.cca: %Int.as.Copy.impl.Op.type.d84 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.0ff: <witness> = impl_witness imports.%Copy.impl_witness_table.f74, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %facet_value.7e1: %facet_type.f48 = facet_value %i32, (%custom_witness.f8f19d.1, %Copy.impl_witness.0ff) [concrete]
// CHECK:STDOUT:   %Iterate.lookup_impl_witness.1d8: <witness> = lookup_impl_witness %.Self.500, @Iterate [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.294: %facet_type.f48 = impl_witness_access %Iterate.lookup_impl_witness.1d8, element0 [symbolic_self]
// CHECK:STDOUT:   %Iterate_where.type.e81: type = facet_type <@Iterate where %impl.elem0.294 = %facet_value.7e1> [concrete]
// CHECK:STDOUT:   %pattern_type.555: type = pattern_type %Iterate_where.type.e81 [concrete]
// CHECK:STDOUT:   %R.patt: %pattern_type.555 = symbolic_binding_pattern R, 0 [symbolic]
// CHECK:STDOUT:   %R: %Iterate_where.type.e81 = symbolic_binding R, 0 [symbolic]
// CHECK:STDOUT:   %R.as_type: type = facet_access_type %R [symbolic]
// CHECK:STDOUT:   %pattern_type.3dc: type = pattern_type %R.as_type [symbolic]
// CHECK:STDOUT:   %r.param_patt.ab7: %pattern_type.3dc = value_param_pattern [symbolic]
// CHECK:STDOUT:   %r.patt.531: %pattern_type.3dc = wrapper_binding_pattern r, %r.param_patt.ab7 [symbolic]
// CHECK:STDOUT:   %facet_type.6c0: type = facet_type <@IntFitsIn, @IntFitsIn(%Int.ec967b.2) & @AnyInt> [symbolic]
// CHECK:STDOUT:   %From.e88: %facet_type.6c0 = symbolic_binding From, 1 [symbolic]
// CHECK:STDOUT:   %From.as_type.as.ImplicitAs.impl.Convert.type.830: type = fn_type @From.as_type.as.ImplicitAs.impl.Convert, @From.as_type.as.ImplicitAs.impl(%To, %From.e88) [symbolic]
// CHECK:STDOUT:   %From.as_type.as.ImplicitAs.impl.Convert.7b2: %From.as_type.as.ImplicitAs.impl.Convert.type.830 = struct_value () [symbolic]
// CHECK:STDOUT:   %i.patt: %pattern_type.9d9 = value_binding_pattern i [concrete]
// CHECK:STDOUT:   %Iterate.lookup_impl_witness.281: <witness> = lookup_impl_witness %R, @Iterate [symbolic]
// CHECK:STDOUT:   %Iterate.facet.6be: %Iterate.type = facet_value %R.as_type, (%Iterate.lookup_impl_witness.281) [symbolic]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.f51: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.6be) [symbolic]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.0a9: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.6be) [symbolic]
// CHECK:STDOUT:   %.5de: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.f51, %Iterate.facet.6be [symbolic]
// CHECK:STDOUT:   %impl.elem2.38c: %.5de = impl_witness_access %Iterate.lookup_impl_witness.281, element2 [symbolic]
// CHECK:STDOUT:   %impl.elem1.5af: %Destroy.type = impl_witness_access %Iterate.lookup_impl_witness.281, element1 [symbolic]
// CHECK:STDOUT:   %as_type.73e: type = facet_access_type %impl.elem1.5af [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.7fb: <specific function> = specific_impl_function %impl.elem2.38c, @Iterate.WithSelf.NewCursor(%Iterate.facet.6be) [symbolic]
// CHECK:STDOUT:   %.798: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.0a9, %Iterate.facet.6be [symbolic]
// CHECK:STDOUT:   %impl.elem3.5f6: %.798 = impl_witness_access %Iterate.lookup_impl_witness.281, element3 [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.78c: type = class_type @DefaultOptionalStorage, @DefaultOptionalStorage(%facet_value.7e1) [concrete]
// CHECK:STDOUT:   %OptionalStorage.impl_witness.d14: <witness> = impl_witness imports.%OptionalStorage.impl_witness_table.460, @T.as_type.as.OptionalStorage.impl(%facet_value.7e1) [concrete]
// CHECK:STDOUT:   %OptionalStorage.facet.ac7: %OptionalStorage.type = facet_value %i32, (%OptionalStorage.impl_witness.d14) [concrete]
// CHECK:STDOUT:   %Optional.117: type = class_type @Optional, @Optional(%OptionalStorage.facet.ac7) [concrete]
// CHECK:STDOUT:   %pattern_type.e75: type = pattern_type %Optional.117 [concrete]
// CHECK:STDOUT:   %specific_impl_fn.e73: <specific function> = specific_impl_function %impl.elem3.5f6, @Iterate.WithSelf.Next(%Iterate.facet.6be) [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.type.ced: type = fn_type @Optional.HasValue, @Optional(%OptionalStorage.facet.ac7) [concrete]
// CHECK:STDOUT:   %Optional.HasValue.615: %Optional.HasValue.type.ced = struct_value () [concrete]
// CHECK:STDOUT:   %Optional.Get.type.703: type = fn_type @Optional.Get, @Optional(%OptionalStorage.facet.ac7) [concrete]
// CHECK:STDOUT:   %Optional.Get.5fc: %Optional.Get.type.703 = struct_value () [concrete]
// CHECK:STDOUT:   %MaybeUnformed.0a2: type = class_type @MaybeUnformed, @MaybeUnformed(%Destroy.facet.2b3) [concrete]
// CHECK:STDOUT:   %.473: type = maybe_unformed_type %i32 [concrete]
// CHECK:STDOUT:   %DefaultOptionalStorage.elem.76a: type = unbound_element_type %DefaultOptionalStorage.78c, %MaybeUnformed.0a2 [concrete]
// CHECK:STDOUT:   %DefaultOptionalStorage.elem.ad8: type = unbound_element_type %DefaultOptionalStorage.78c, bool [concrete]
// CHECK:STDOUT:   %struct_type.value.has_value.88c: type = struct_type {.value: %MaybeUnformed.0a2, .has_value: bool} [concrete]
// CHECK:STDOUT:   %Optional.HasValue.specific_fn.bc7: <specific function> = specific_function %Optional.HasValue.615, @Optional.HasValue(%OptionalStorage.facet.ac7) [concrete]
// CHECK:STDOUT:   %Optional.Get.specific_fn.1c0: <specific function> = specific_function %Optional.Get.5fc, @Optional.Get(%OptionalStorage.facet.ac7) [concrete]
// CHECK:STDOUT:   %AddAssignWith.type.0c0: type = facet_type <@AddAssignWith, @AddAssignWith(%i32)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.3758b9.2: type = facet_type <@ImplicitAs, @ImplicitAs(%Int.ec967b.1)> [symbolic]
// CHECK:STDOUT:   %U.722: %ImplicitAs.type.3758b9.2 = symbolic_binding U, 1 [symbolic]
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.type.8adbfb.1: type = fn_type @Int.as.AddAssignWith.impl.Op.1, @Int.as.AddAssignWith.impl(%N.fe9, %U.722) [symbolic]
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.9b6333.1: %Int.as.AddAssignWith.impl.Op.type.8adbfb.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.type.8adbfb.2: type = fn_type @Int.as.AddAssignWith.impl.Op.2, @Int.as.AddAssignWith.impl(%N.fe9, %U.722) [symbolic]
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.9b6333.2: %Int.as.AddAssignWith.impl.Op.type.8adbfb.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %facet_type.1d7: type = facet_type <@IntFitsIn, @IntFitsIn(%i32) & @AnyInt> [concrete]
// CHECK:STDOUT:   %custom_witness.040: <witness> = custom_witness (), @IntFitsIn, @IntFitsIn(%i32) [concrete]
// CHECK:STDOUT:   %Int.as.AnyInt.impl.AsInt.type.212: type = fn_type @Int.as.AnyInt.impl.AsInt, @Int.as.AnyInt.impl(%N.fe9) [symbolic]
// CHECK:STDOUT:   %Int.as.AnyInt.impl.AsInt.c39: %Int.as.AnyInt.impl.AsInt.type.212 = struct_value () [symbolic]
// CHECK:STDOUT:   %AnyInt.impl_witness.59c: <witness> = impl_witness imports.%AnyInt.impl_witness_table, @Int.as.AnyInt.impl(%int_32) [concrete]
// CHECK:STDOUT:   %facet_value.fc2: %facet_type.1d7 = facet_value %i32, (%custom_witness.040, %AnyInt.impl_witness.59c) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.fc7: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.106, @From.as_type.as.ImplicitAs.impl(%int_32, %facet_value.fc2) [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.371: %ImplicitAs.type.09b = facet_value %i32, (%ImplicitAs.impl_witness.fc7) [concrete]
// CHECK:STDOUT:   %AddAssignWith.impl_witness.2a4: <witness> = impl_witness imports.%AddAssignWith.impl_witness_table, @Int.as.AddAssignWith.impl(%int_32, %ImplicitAs.facet.371) [concrete]
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.type.303b18.1: type = fn_type @Int.as.AddAssignWith.impl.Op.2, @Int.as.AddAssignWith.impl(%int_32, %ImplicitAs.facet.371) [concrete]
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.bd35de.1: %Int.as.AddAssignWith.impl.Op.type.303b18.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.type.303b18.2: type = fn_type @Int.as.AddAssignWith.impl.Op.1, @Int.as.AddAssignWith.impl(%int_32, %ImplicitAs.facet.371) [concrete]
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.bd35de.2: %Int.as.AddAssignWith.impl.Op.type.303b18.2 = struct_value () [concrete]
// CHECK:STDOUT:   %AddAssignWith.facet.834: %AddAssignWith.type.0c0 = facet_value %i32, (%AddAssignWith.impl_witness.2a4) [concrete]
// CHECK:STDOUT:   %AddAssignWith.WithSelf.Op.type.411: type = fn_type @AddAssignWith.WithSelf.Op, @AddAssignWith.WithSelf(%i32, %AddAssignWith.facet.834) [concrete]
// CHECK:STDOUT:   %.2f4: type = fn_type_with_self_type %AddAssignWith.WithSelf.Op.type.411, %AddAssignWith.facet.834 [concrete]
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.specific_fn.3157f3.1: <specific function> = specific_function %Int.as.AddAssignWith.impl.Op.bd35de.2, @Int.as.AddAssignWith.impl.Op.1(%int_32, %ImplicitAs.facet.371) [concrete]
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.specific_fn.3157f3.2: <specific function> = specific_function %Int.as.AddAssignWith.impl.Op.bd35de.1, @Int.as.AddAssignWith.impl.Op.2(%int_32, %ImplicitAs.facet.371) [concrete]
// CHECK:STDOUT:   %pattern_type.263: type = pattern_type %.473 [concrete]
// CHECK:STDOUT:   %self.param_patt.9c5: %pattern_type.263 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.d0e: %pattern_type.263 = wrapper_binding_pattern self, %self.param_patt.9c5 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc34_19.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.2: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.2: type = fn_type @Destroy.WithSelf.Op.loc34_19.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.2: %Destroy.WithSelf.Op.type.ef016f.2 = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.e91: type = pattern_type %MaybeUnformed.0a2 [concrete]
// CHECK:STDOUT:   %self.param_patt.bab: %pattern_type.e91 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.1af: %pattern_type.e91 = wrapper_binding_pattern self, %self.param_patt.bab [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.3: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc34_19.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.3: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.3 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.3: type = fn_type @Destroy.WithSelf.Op.loc34_19.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.3: %Destroy.WithSelf.Op.type.ef016f.3 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.3: type = fn_type @Destroy.WithSelf.SelfDestruct.loc34_19.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.3: %Destroy.WithSelf.SelfDestruct.type.fbceb5.3 = struct_value () [concrete]
// CHECK:STDOUT:   %self.param_patt.40d: %pattern_type.831 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.560: %pattern_type.831 = wrapper_binding_pattern self, %self.param_patt.40d [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.4: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc34_19.3 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.4: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.4 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.4: type = fn_type @Destroy.WithSelf.Op.loc34_19.3 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.4: %Destroy.WithSelf.Op.type.ef016f.4 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.4: type = fn_type @Destroy.WithSelf.SelfDestruct.loc34_19.3 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.4: %Destroy.WithSelf.SelfDestruct.type.fbceb5.4 = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.fa8: type = pattern_type %struct_type.value.has_value.88c [concrete]
// CHECK:STDOUT:   %self.param_patt.97f: %pattern_type.fa8 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.5f1: %pattern_type.fa8 = wrapper_binding_pattern self, %self.param_patt.97f [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.5: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc34_19.4 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.5: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.5 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.5: type = fn_type @Destroy.WithSelf.Op.loc34_19.4 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.5: %Destroy.WithSelf.Op.type.ef016f.5 = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.db6: type = pattern_type %DefaultOptionalStorage.78c [concrete]
// CHECK:STDOUT:   %self.param_patt.7ab: %pattern_type.db6 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.d35: %pattern_type.db6 = wrapper_binding_pattern self, %self.param_patt.7ab [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.6: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc34_19.5 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.6: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.6 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.6: type = fn_type @Destroy.WithSelf.Op.loc34_19.5 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.6: %Destroy.WithSelf.Op.type.ef016f.6 = struct_value () [concrete]
// CHECK:STDOUT:   %self.param_patt.142: %pattern_type.e75 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.59f: %pattern_type.e75 = wrapper_binding_pattern self, %self.param_patt.142 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.7: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc34_19.6 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.7: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.7 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.7: type = fn_type @Destroy.WithSelf.Op.loc34_19.6 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.7: %Destroy.WithSelf.Op.type.ef016f.7 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.7: type = fn_type @Destroy.WithSelf.SelfDestruct.loc34_19.6 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.7: %Destroy.WithSelf.SelfDestruct.type.fbceb5.7 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.690: <witness> = lookup_impl_witness %impl.elem1.5af, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.271: type = fn_type @Destroy.WithSelf.SelfDestruct.1, @Destroy.WithSelf(%impl.elem1.5af) [symbolic]
// CHECK:STDOUT:   %.ed4: type = fn_type_with_self_type %Destroy.WithSelf.SelfDestruct.type.271, %impl.elem1.5af [symbolic]
// CHECK:STDOUT:   %impl.elem2.e0f: %.ed4 = impl_witness_access %Destroy.lookup_impl_witness.690, element2 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.6b4: <specific function> = specific_impl_function %impl.elem2.e0f, @Destroy.WithSelf.SelfDestruct.1(%impl.elem1.5af) [symbolic]
// CHECK:STDOUT:   %MutableRange: type = class_type @MutableRange [concrete]
// CHECK:STDOUT:   %pattern_type.e63: type = pattern_type %MutableRange [concrete]
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
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.8: type = fn_type @Destroy.WithSelf.Op.loc41 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.8: %Destroy.WithSelf.Op.type.ef016f.8 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.8: type = fn_type @Destroy.WithSelf.SelfDestruct.loc41 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.8: %Destroy.WithSelf.SelfDestruct.type.fbceb5.8 = struct_value () [concrete]
// CHECK:STDOUT:   %Iterator.SubobjectDestroy.type.9ec: type = fn_type @Iterator.SubobjectDestroy.1 [concrete]
// CHECK:STDOUT:   %Iterator.SubobjectDestroy.e52: %Iterator.SubobjectDestroy.type.9ec = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.8e1: <witness> = custom_witness (%Destroy.WithSelf.Op.403171.8, %Iterator.SubobjectDestroy.e52, %Destroy.WithSelf.SelfDestruct.db3fdb.8), @Destroy [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.0aed5d.1: type = fn_type @Iterator.Op.1 [concrete]
// CHECK:STDOUT:   %Iterator.Op.a2a866.1: %Iterator.Op.type.0aed5d.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b38: <witness> = custom_witness (%i32, %Iterator.Op.a2a866.1), @CppUnsafeDeref [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.0aed5d.2: type = fn_type @Iterator.Op.2 [concrete]
// CHECK:STDOUT:   %Iterator.Op.a2a866.2: %Iterator.Op.type.0aed5d.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.444: <witness> = custom_witness (%Iterator.Op.a2a866.2), @Inc [concrete]
// CHECK:STDOUT:   %Equal.type.ca8475.1: type = fn_type @Equal.1 [concrete]
// CHECK:STDOUT:   %Equal.285b15.1: %Equal.type.ca8475.1 = struct_value () [concrete]
// CHECK:STDOUT:   %NotEqual.type.53d9ee.1: type = fn_type @NotEqual.1 [concrete]
// CHECK:STDOUT:   %NotEqual.119263.1: %NotEqual.type.53d9ee.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.c91: <witness> = custom_witness (%Equal.285b15.1, %NotEqual.119263.1), @EqWith, @EqWith(%Iterator.3ee) [concrete]
// CHECK:STDOUT:   %facet_value.84d: %CppRange_where.type.d682d6.1 = facet_value %MutableRange, (%custom_witness.ed3, %custom_witness.8e1, %custom_witness.b38, %custom_witness.444, %custom_witness.c91, %custom_witness.f8f19d.1, %Copy.impl_witness.0ff, %custom_witness.8e1) [concrete]
// CHECK:STDOUT:   %tuple.type.508: type = tuple_type (%Iterator.3ee, %Iterator.3ee) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.8: type = fn_type @Destroy.WithSelf.SubobjectDestroy.3 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.8: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.8 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.9: type = fn_type @Destroy.WithSelf.Op.3 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.9: %Destroy.WithSelf.Op.type.ef016f.9 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.9: type = fn_type @Destroy.WithSelf.SelfDestruct.3 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.9: %Destroy.WithSelf.SelfDestruct.type.fbceb5.9 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.f8f19d.8: <witness> = custom_witness (%Destroy.WithSelf.Op.403171.9, %Destroy.WithSelf.SubobjectDestroy.d01daf.8, %Destroy.WithSelf.SelfDestruct.db3fdb.9), @Destroy [concrete]
// CHECK:STDOUT:   %Destroy.facet.0fe: %Destroy.type = facet_value %tuple.type.508, (%custom_witness.f8f19d.8) [concrete]
// CHECK:STDOUT:   %Iterate.impl_witness.230: <witness> = impl_witness imports.%Iterate.impl_witness_table.121, @T.as_type.as.Iterate.impl(%facet_value.84d) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.type.ee9: type = fn_type @T.as_type.as.Iterate.impl.NewCursor, @T.as_type.as.Iterate.impl(%facet_value.84d) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.067: %T.as_type.as.Iterate.impl.NewCursor.type.ee9 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.type.fce: type = fn_type @T.as_type.as.Iterate.impl.Next, @T.as_type.as.Iterate.impl(%facet_value.84d) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.f95: %T.as_type.as.Iterate.impl.Next.type.fce = struct_value () [concrete]
// CHECK:STDOUT:   %Iterate.facet.af6: %Iterate.type = facet_value %MutableRange, (%Iterate.impl_witness.230) [concrete]
// CHECK:STDOUT:   %facet_value.904: %Iterate_where.type.e81 = facet_value %MutableRange, (%Iterate.impl_witness.230) [concrete]
// CHECK:STDOUT:   %r.param_patt.c79: %pattern_type.e63 = value_param_pattern [concrete]
// CHECK:STDOUT:   %r.patt.94f: %pattern_type.e63 = wrapper_binding_pattern r, %r.param_patt.c79 [concrete]
// CHECK:STDOUT:   %Iterator.260: type = class_type @Iterator.2 [concrete]
// CHECK:STDOUT:   %ConstRange.Begin.type: type = fn_type @ConstRange.Begin [concrete]
// CHECK:STDOUT:   %ConstRange.Begin: %ConstRange.Begin.type = struct_value () [concrete]
// CHECK:STDOUT:   %ConstRange.End.type: type = fn_type @ConstRange.End [concrete]
// CHECK:STDOUT:   %ConstRange.End: %ConstRange.End.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.23e: <witness> = custom_witness (%Iterator.260, %Iterator.260, %ConstRange.Begin, %ConstRange.End), @CppRangeForIterate [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.10: type = fn_type @Destroy.WithSelf.Op.loc42 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.10: %Destroy.WithSelf.Op.type.ef016f.10 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.10: type = fn_type @Destroy.WithSelf.SelfDestruct.loc42 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.10: %Destroy.WithSelf.SelfDestruct.type.fbceb5.10 = struct_value () [concrete]
// CHECK:STDOUT:   %Iterator.SubobjectDestroy.type.128: type = fn_type @Iterator.SubobjectDestroy.2 [concrete]
// CHECK:STDOUT:   %Iterator.SubobjectDestroy.37b: %Iterator.SubobjectDestroy.type.128 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.0f3: <witness> = custom_witness (%Destroy.WithSelf.Op.403171.10, %Iterator.SubobjectDestroy.37b, %Destroy.WithSelf.SelfDestruct.db3fdb.10), @Destroy [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.0f34b8.1: type = fn_type @Iterator.Op.3 [concrete]
// CHECK:STDOUT:   %Iterator.Op.1c3e4a.1: %Iterator.Op.type.0f34b8.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.c00: <witness> = custom_witness (%i32, %Iterator.Op.1c3e4a.1), @CppUnsafeDeref [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.0f34b8.2: type = fn_type @Iterator.Op.4 [concrete]
// CHECK:STDOUT:   %Iterator.Op.1c3e4a.2: %Iterator.Op.type.0f34b8.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.746: <witness> = custom_witness (%Iterator.Op.1c3e4a.2), @Inc [concrete]
// CHECK:STDOUT:   %Equal.type.ca8475.2: type = fn_type @Equal.2 [concrete]
// CHECK:STDOUT:   %Equal.285b15.2: %Equal.type.ca8475.2 = struct_value () [concrete]
// CHECK:STDOUT:   %NotEqual.type.53d9ee.2: type = fn_type @NotEqual.2 [concrete]
// CHECK:STDOUT:   %NotEqual.119263.2: %NotEqual.type.53d9ee.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.86f: <witness> = custom_witness (%Equal.285b15.2, %NotEqual.119263.2), @EqWith, @EqWith(%Iterator.260) [concrete]
// CHECK:STDOUT:   %facet_value.793: %CppRange_where.type.d682d6.1 = facet_value %ConstRange, (%custom_witness.23e, %custom_witness.0f3, %custom_witness.c00, %custom_witness.746, %custom_witness.86f, %custom_witness.f8f19d.1, %Copy.impl_witness.0ff, %custom_witness.0f3) [concrete]
// CHECK:STDOUT:   %tuple.type.69f: type = tuple_type (%Iterator.260, %Iterator.260) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.9: type = fn_type @Destroy.WithSelf.SubobjectDestroy.4 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.9: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.9 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.11: type = fn_type @Destroy.WithSelf.Op.4 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.11: %Destroy.WithSelf.Op.type.ef016f.11 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.11: type = fn_type @Destroy.WithSelf.SelfDestruct.4 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.11: %Destroy.WithSelf.SelfDestruct.type.fbceb5.11 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.f8f19d.9: <witness> = custom_witness (%Destroy.WithSelf.Op.403171.11, %Destroy.WithSelf.SubobjectDestroy.d01daf.9, %Destroy.WithSelf.SelfDestruct.db3fdb.11), @Destroy [concrete]
// CHECK:STDOUT:   %Destroy.facet.c6d: %Destroy.type = facet_value %tuple.type.69f, (%custom_witness.f8f19d.9) [concrete]
// CHECK:STDOUT:   %Iterate.impl_witness.01e: <witness> = impl_witness imports.%Iterate.impl_witness_table.121, @T.as_type.as.Iterate.impl(%facet_value.793) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.type.8e7: type = fn_type @T.as_type.as.Iterate.impl.NewCursor, @T.as_type.as.Iterate.impl(%facet_value.793) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.644: %T.as_type.as.Iterate.impl.NewCursor.type.8e7 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.type.9fb: type = fn_type @T.as_type.as.Iterate.impl.Next, @T.as_type.as.Iterate.impl(%facet_value.793) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.b53: %T.as_type.as.Iterate.impl.Next.type.9fb = struct_value () [concrete]
// CHECK:STDOUT:   %Iterate.facet.98a: %Iterate.type = facet_value %ConstRange, (%Iterate.impl_witness.01e) [concrete]
// CHECK:STDOUT:   %facet_value.82c: %Iterate_where.type.e81 = facet_value %ConstRange, (%Iterate.impl_witness.01e) [concrete]
// CHECK:STDOUT:   %r.param_patt.83e: %pattern_type.0ba = value_param_pattern [concrete]
// CHECK:STDOUT:   %r.patt.c21: %pattern_type.0ba = wrapper_binding_pattern r, %r.param_patt.83e [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.ade: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.af6) [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.949: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.af6) [concrete]
// CHECK:STDOUT:   %.d0d: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.ade, %Iterate.facet.af6 [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.specific_fn.212: <specific function> = specific_function %T.as_type.as.Iterate.impl.NewCursor.067, @T.as_type.as.Iterate.impl.NewCursor(%facet_value.84d) [concrete]
// CHECK:STDOUT:   %.0cc: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.949, %Iterate.facet.af6 [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.specific_fn.4c7: <specific function> = specific_function %T.as_type.as.Iterate.impl.Next.f95, @T.as_type.as.Iterate.impl.Next(%facet_value.84d) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.d1f: type = fn_type @Destroy.WithSelf.SelfDestruct.1, @Destroy.WithSelf(%Destroy.facet.0fe) [concrete]
// CHECK:STDOUT:   %.0dc: type = fn_type_with_self_type %Destroy.WithSelf.SelfDestruct.type.d1f, %Destroy.facet.0fe [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.867: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.98a) [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.5a5: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.98a) [concrete]
// CHECK:STDOUT:   %.fd0: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.867, %Iterate.facet.98a [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.specific_fn.b5f: <specific function> = specific_function %T.as_type.as.Iterate.impl.NewCursor.644, @T.as_type.as.Iterate.impl.NewCursor(%facet_value.793) [concrete]
// CHECK:STDOUT:   %.d49: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.5a5, %Iterate.facet.98a [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.specific_fn.f07: <specific function> = specific_function %T.as_type.as.Iterate.impl.Next.b53, @T.as_type.as.Iterate.impl.Next(%facet_value.793) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.b84: type = fn_type @Destroy.WithSelf.SelfDestruct.1, @Destroy.WithSelf(%Destroy.facet.c6d) [concrete]
// CHECK:STDOUT:   %.40c: type = fn_type_with_self_type %Destroy.WithSelf.SelfDestruct.type.b84, %Destroy.facet.c6d [concrete]
// CHECK:STDOUT:   %complete_type.c28: <witness> = complete_type_witness %tuple.type.508 [concrete]
// CHECK:STDOUT:   %complete_type.8d5: <witness> = complete_type_witness %tuple.type.69f [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.4b2: @Optional.%Optional.HasValue.type (%Optional.HasValue.type.49e) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.HasValue (constants.%Optional.HasValue.c20)]
// CHECK:STDOUT:   %Core.import_ref.52c: @Optional.%Optional.Get.type (%Optional.Get.type.c64) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.Get (constants.%Optional.Get.bcf)]
// CHECK:STDOUT:   %Core.import_ref.ace: type = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%DefaultOptionalStorage (constants.%DefaultOptionalStorage.faa)]
// CHECK:STDOUT:   %Core.import_ref.ef4: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.None.type (%T.as_type.as.OptionalStorage.impl.None.type.925) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.None (constants.%T.as_type.as.OptionalStorage.impl.None.f42)]
// CHECK:STDOUT:   %Core.import_ref.a20: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Some.type (%T.as_type.as.OptionalStorage.impl.Some.type.760) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Some (constants.%T.as_type.as.OptionalStorage.impl.Some.268)]
// CHECK:STDOUT:   %Core.import_ref.af7: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Has.type (%T.as_type.as.OptionalStorage.impl.Has.type.d28) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Has (constants.%T.as_type.as.OptionalStorage.impl.Has.f29)]
// CHECK:STDOUT:   %Core.import_ref.dd4: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Get.type (%T.as_type.as.OptionalStorage.impl.Get.type.2f2) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Get (constants.%T.as_type.as.OptionalStorage.impl.Get.884)]
// CHECK:STDOUT:   %Core.import_ref.554 = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], unloaded
// CHECK:STDOUT:   %OptionalStorage.impl_witness_table.460 = impl_witness_table (%Core.import_ref.ace, %Core.import_ref.ef4, %Core.import_ref.a20, %Core.import_ref.af7, %Core.import_ref.dd4, %Core.import_ref.554), @T.as_type.as.OptionalStorage.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.1d7: @DefaultOptionalStorage.%DefaultOptionalStorage.elem.1 (%DefaultOptionalStorage.elem.9db) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [concrete = %field_decl.5f0]
// CHECK:STDOUT:   %Core.import_ref.929: @DefaultOptionalStorage.%DefaultOptionalStorage.elem.2 (%DefaultOptionalStorage.elem.06a) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [concrete = %field_decl.6a8]
// CHECK:STDOUT:   %Core.import_ref.eef: %facet_type.f48 = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.Iterate.impl.%facet_value (constants.%facet_value.f93)]
// CHECK:STDOUT:   %Core.import_ref.fcc: %Destroy.type = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.Iterate.impl.%Destroy.facet (constants.%Destroy.facet.9cd)]
// CHECK:STDOUT:   %Core.import_ref.946: @T.as_type.as.Iterate.impl.%T.as_type.as.Iterate.impl.NewCursor.type (%T.as_type.as.Iterate.impl.NewCursor.type.8fb) = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.Iterate.impl.%T.as_type.as.Iterate.impl.NewCursor (constants.%T.as_type.as.Iterate.impl.NewCursor.8a1)]
// CHECK:STDOUT:   %Core.import_ref.cfb: @T.as_type.as.Iterate.impl.%T.as_type.as.Iterate.impl.Next.type (%T.as_type.as.Iterate.impl.Next.type.d4a) = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.Iterate.impl.%T.as_type.as.Iterate.impl.Next (constants.%T.as_type.as.Iterate.impl.Next.5a8)]
// CHECK:STDOUT:   %Iterate.impl_witness_table.121 = impl_witness_table (%Core.import_ref.eef, %Core.import_ref.fcc, %Core.import_ref.946, %Core.import_ref.cfb), @T.as_type.as.Iterate.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.dfa: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.d84) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.cca)]
// CHECK:STDOUT:   %Copy.impl_witness_table.f74 = impl_witness_table (%Core.import_ref.dfa), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.1c7: @From.as_type.as.ImplicitAs.impl.%From.as_type.as.ImplicitAs.impl.Convert.type (%From.as_type.as.ImplicitAs.impl.Convert.type.830) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @From.as_type.as.ImplicitAs.impl.%From.as_type.as.ImplicitAs.impl.Convert (constants.%From.as_type.as.ImplicitAs.impl.Convert.7b2)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.106 = impl_witness_table (%Core.import_ref.1c7), @From.as_type.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.3c3: @Int.as.AddAssignWith.impl.%Int.as.AddAssignWith.impl.Op.type.2 (%Int.as.AddAssignWith.impl.Op.type.8adbfb.1) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.AddAssignWith.impl.%Int.as.AddAssignWith.impl.Op.2 (constants.%Int.as.AddAssignWith.impl.Op.9b6333.1)]
// CHECK:STDOUT:   %AddAssignWith.impl_witness_table = impl_witness_table (%Core.import_ref.3c3), @Int.as.AddAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.b5f: @Int.as.AddAssignWith.impl.%Int.as.AddAssignWith.impl.Op.type.1 (%Int.as.AddAssignWith.impl.Op.type.8adbfb.2) = import_ref Core//prelude/types/int, Op, loaded [symbolic = @Int.as.AddAssignWith.impl.%Int.as.AddAssignWith.impl.Op.1 (constants.%Int.as.AddAssignWith.impl.Op.9b6333.2)]
// CHECK:STDOUT:   %Core.import_ref.47b: Core.IntLiteral = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.AnyInt.impl.%N (constants.%N.fe9)]
// CHECK:STDOUT:   %Core.import_ref.d8c: @Int.as.AnyInt.impl.%Int.as.AnyInt.impl.AsInt.type (%Int.as.AnyInt.impl.AsInt.type.212) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.AnyInt.impl.%Int.as.AnyInt.impl.AsInt (constants.%Int.as.AnyInt.impl.AsInt.c39)]
// CHECK:STDOUT:   %AnyInt.impl_witness_table = impl_witness_table (%Core.import_ref.47b, %Core.import_ref.d8c), @Int.as.AnyInt.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.b0d = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %field_decl.6a8: @DefaultOptionalStorage.%DefaultOptionalStorage.elem.2 (%DefaultOptionalStorage.elem.06a) = field_decl has_value, element1, %Core.import_ref.b0d in [concrete] {}
// CHECK:STDOUT:   %Core.import_ref.87f = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %field_decl.5f0: @DefaultOptionalStorage.%DefaultOptionalStorage.elem.1 (%DefaultOptionalStorage.elem.9db) = field_decl value, element0, %Core.import_ref.87f in [concrete] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generated {
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.1: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc34_19.1 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.2] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.263 = ref_param_pattern [concrete = constants.%self.param_patt.9c5]
// CHECK:STDOUT:     %self.patt: %pattern_type.263 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.d0e]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %.473 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %.473 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.1: %Destroy.WithSelf.Op.type.ef016f.2 = fn_decl @Destroy.WithSelf.Op.loc34_19.1 [concrete = constants.%Destroy.WithSelf.Op.403171.2] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.263 = ref_param_pattern [concrete = constants.%self.param_patt.9c5]
// CHECK:STDOUT:     %self.patt: %pattern_type.263 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.d0e]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %.473 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %.473 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.2: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.3 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc34_19.2 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.3] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.e91 = ref_param_pattern [concrete = constants.%self.param_patt.bab]
// CHECK:STDOUT:     %self.patt: %pattern_type.e91 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.1af]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %MaybeUnformed.0a2 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %MaybeUnformed.0a2 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.2: %Destroy.WithSelf.Op.type.ef016f.3 = fn_decl @Destroy.WithSelf.Op.loc34_19.2 [concrete = constants.%Destroy.WithSelf.Op.403171.3] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.e91 = ref_param_pattern [concrete = constants.%self.param_patt.bab]
// CHECK:STDOUT:     %self.patt: %pattern_type.e91 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.1af]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %MaybeUnformed.0a2 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %MaybeUnformed.0a2 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.3: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.4 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc34_19.3 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.4] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.831 = ref_param_pattern [concrete = constants.%self.param_patt.40d]
// CHECK:STDOUT:     %self.patt: %pattern_type.831 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.560]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref bool = ref_param call_param0
// CHECK:STDOUT:     %self: ref bool = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.3: %Destroy.WithSelf.Op.type.ef016f.4 = fn_decl @Destroy.WithSelf.Op.loc34_19.3 [concrete = constants.%Destroy.WithSelf.Op.403171.4] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.831 = ref_param_pattern [concrete = constants.%self.param_patt.40d]
// CHECK:STDOUT:     %self.patt: %pattern_type.831 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.560]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref bool = ref_param call_param0
// CHECK:STDOUT:     %self: ref bool = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.4: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.5 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc34_19.4 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.5] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.fa8 = ref_param_pattern [concrete = constants.%self.param_patt.97f]
// CHECK:STDOUT:     %self.patt: %pattern_type.fa8 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.5f1]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %struct_type.value.has_value.88c = ref_param call_param0
// CHECK:STDOUT:     %self: ref %struct_type.value.has_value.88c = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.4: %Destroy.WithSelf.Op.type.ef016f.5 = fn_decl @Destroy.WithSelf.Op.loc34_19.4 [concrete = constants.%Destroy.WithSelf.Op.403171.5] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.fa8 = ref_param_pattern [concrete = constants.%self.param_patt.97f]
// CHECK:STDOUT:     %self.patt: %pattern_type.fa8 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.5f1]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %struct_type.value.has_value.88c = ref_param call_param0
// CHECK:STDOUT:     %self: ref %struct_type.value.has_value.88c = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.5: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.6 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc34_19.5 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.6] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.db6 = ref_param_pattern [concrete = constants.%self.param_patt.7ab]
// CHECK:STDOUT:     %self.patt: %pattern_type.db6 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.d35]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %DefaultOptionalStorage.78c = ref_param call_param0
// CHECK:STDOUT:     %self: ref %DefaultOptionalStorage.78c = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.5: %Destroy.WithSelf.Op.type.ef016f.6 = fn_decl @Destroy.WithSelf.Op.loc34_19.5 [concrete = constants.%Destroy.WithSelf.Op.403171.6] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.db6 = ref_param_pattern [concrete = constants.%self.param_patt.7ab]
// CHECK:STDOUT:     %self.patt: %pattern_type.db6 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.d35]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %DefaultOptionalStorage.78c = ref_param call_param0
// CHECK:STDOUT:     %self: ref %DefaultOptionalStorage.78c = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.6: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.7 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc34_19.6 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.7] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.e75 = ref_param_pattern [concrete = constants.%self.param_patt.142]
// CHECK:STDOUT:     %self.patt: %pattern_type.e75 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.59f]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %Optional.117 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %Optional.117 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.6: %Destroy.WithSelf.Op.type.ef016f.7 = fn_decl @Destroy.WithSelf.Op.loc34_19.6 [concrete = constants.%Destroy.WithSelf.Op.403171.7] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.e75 = ref_param_pattern [concrete = constants.%self.param_patt.142]
// CHECK:STDOUT:     %self.patt: %pattern_type.e75 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.59f]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %Optional.117 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %Optional.117 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @TestIterate(%R.loc30_17.2: %Iterate_where.type.e81) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Iterate.lookup_impl_witness: <witness> = lookup_impl_witness %R.loc30_17.1, @Iterate [symbolic = %Iterate.lookup_impl_witness (constants.%Iterate.lookup_impl_witness.281)]
// CHECK:STDOUT:   %Iterate.facet.loc34_19.5: %Iterate.type = facet_value %R.as_type.loc30_61.1, (%Iterate.lookup_impl_witness) [symbolic = %Iterate.facet.loc34_19.5 (constants.%Iterate.facet.6be)]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.loc34_19.5) [symbolic = %Iterate.WithSelf.NewCursor.type (constants.%Iterate.WithSelf.NewCursor.type.f51)]
// CHECK:STDOUT:   %.loc34_19.19: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type, %Iterate.facet.loc34_19.5 [symbolic = %.loc34_19.19 (constants.%.5de)]
// CHECK:STDOUT:   %impl.elem2.loc34_19.3: @TestIterate.%.loc34_19.19 (%.5de) = impl_witness_access %Iterate.lookup_impl_witness, element2 [symbolic = %impl.elem2.loc34_19.3 (constants.%impl.elem2.38c)]
// CHECK:STDOUT:   %specific_impl_fn.loc34_19.4: <specific function> = specific_impl_function %impl.elem2.loc34_19.3, @Iterate.WithSelf.NewCursor(%Iterate.facet.loc34_19.5) [symbolic = %specific_impl_fn.loc34_19.4 (constants.%specific_impl_fn.7fb)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.loc34_19.5) [symbolic = %Iterate.WithSelf.Next.type (constants.%Iterate.WithSelf.Next.type.0a9)]
// CHECK:STDOUT:   %.loc34_19.20: type = fn_type_with_self_type %Iterate.WithSelf.Next.type, %Iterate.facet.loc34_19.5 [symbolic = %.loc34_19.20 (constants.%.798)]
// CHECK:STDOUT:   %impl.elem3.loc34_19.2: @TestIterate.%.loc34_19.20 (%.798) = impl_witness_access %Iterate.lookup_impl_witness, element3 [symbolic = %impl.elem3.loc34_19.2 (constants.%impl.elem3.5f6)]
// CHECK:STDOUT:   %specific_impl_fn.loc34_19.5: <specific function> = specific_impl_function %impl.elem3.loc34_19.2, @Iterate.WithSelf.Next(%Iterate.facet.loc34_19.5) [symbolic = %specific_impl_fn.loc34_19.5 (constants.%specific_impl_fn.e73)]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type: type = fn_type @Destroy.WithSelf.SelfDestruct.1, @Destroy.WithSelf(%impl.elem1) [symbolic = %Destroy.WithSelf.SelfDestruct.type (constants.%Destroy.WithSelf.SelfDestruct.type.271)]
// CHECK:STDOUT:   %.loc34_19.21: type = fn_type_with_self_type %Destroy.WithSelf.SelfDestruct.type, %impl.elem1 [symbolic = %.loc34_19.21 (constants.%.ed4)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %impl.elem1, @Destroy [symbolic = %Destroy.lookup_impl_witness (constants.%Destroy.lookup_impl_witness.690)]
// CHECK:STDOUT:   %impl.elem2.loc34_19.4: @TestIterate.%.loc34_19.21 (%.ed4) = impl_witness_access %Destroy.lookup_impl_witness, element2 [symbolic = %impl.elem2.loc34_19.4 (constants.%impl.elem2.e0f)]
// CHECK:STDOUT:   %specific_impl_fn.loc34_19.6: <specific function> = specific_impl_function %impl.elem2.loc34_19.4, @Destroy.WithSelf.SelfDestruct.1(%impl.elem1) [symbolic = %specific_impl_fn.loc34_19.6 (constants.%specific_impl_fn.6b4)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%r.param: @TestIterate.%R.as_type.loc30_61.1 (%R.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %i.patt: %pattern_type.9d9 = value_binding_pattern i [concrete = constants.%i.patt]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %r.ref: @TestIterate.%R.as_type.loc30_61.1 (%R.as_type) = name_ref r, %r
// CHECK:STDOUT:     %impl.elem2.loc34_19.1: @TestIterate.%.loc34_19.19 (%.5de) = impl_witness_access constants.%Iterate.lookup_impl_witness.281, element2 [symbolic = %impl.elem2.loc34_19.3 (constants.%impl.elem2.38c)]
// CHECK:STDOUT:     %bound_method.loc34_19.1: <bound method> = bound_method %r.ref, %impl.elem2.loc34_19.1
// CHECK:STDOUT:     %Iterate.facet.loc34_19.1: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.281) [symbolic = %Iterate.facet.loc34_19.5 (constants.%Iterate.facet.6be)]
// CHECK:STDOUT:     %.loc34_19.1: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc34_19.1 [symbolic = %Iterate.facet.loc34_19.5 (constants.%Iterate.facet.6be)]
// CHECK:STDOUT:     %Iterate.facet.loc34_19.2: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.281) [symbolic = %Iterate.facet.loc34_19.5 (constants.%Iterate.facet.6be)]
// CHECK:STDOUT:     %.loc34_19.2: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc34_19.2 [symbolic = %Iterate.facet.loc34_19.5 (constants.%Iterate.facet.6be)]
// CHECK:STDOUT:     %specific_impl_fn.loc34_19.1: <specific function> = specific_impl_function %impl.elem2.loc34_19.1, @Iterate.WithSelf.NewCursor(constants.%Iterate.facet.6be) [symbolic = %specific_impl_fn.loc34_19.4 (constants.%specific_impl_fn.7fb)]
// CHECK:STDOUT:     %bound_method.loc34_19.2: <bound method> = bound_method %r.ref, %specific_impl_fn.loc34_19.1
// CHECK:STDOUT:     %var: ref @TestIterate.%as_type (%as_type.73e) = var_storage invalid
// CHECK:STDOUT:     %Iterate.WithSelf.NewCursor.call: init @TestIterate.%as_type (%as_type.73e) to %var = call %bound_method.loc34_19.2(%r.ref)
// CHECK:STDOUT:     assign %var, %Iterate.WithSelf.NewCursor.call
// CHECK:STDOUT:     br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.next:
// CHECK:STDOUT:     %impl.elem3.loc34_19.1: @TestIterate.%.loc34_19.20 (%.798) = impl_witness_access constants.%Iterate.lookup_impl_witness.281, element3 [symbolic = %impl.elem3.loc34_19.2 (constants.%impl.elem3.5f6)]
// CHECK:STDOUT:     %bound_method.loc34_19.3: <bound method> = bound_method %r.ref, %impl.elem3.loc34_19.1
// CHECK:STDOUT:     %Iterate.facet.loc34_19.3: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.281) [symbolic = %Iterate.facet.loc34_19.5 (constants.%Iterate.facet.6be)]
// CHECK:STDOUT:     %.loc34_19.3: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc34_19.3 [symbolic = %Iterate.facet.loc34_19.5 (constants.%Iterate.facet.6be)]
// CHECK:STDOUT:     %Iterate.facet.loc34_19.4: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.281) [symbolic = %Iterate.facet.loc34_19.5 (constants.%Iterate.facet.6be)]
// CHECK:STDOUT:     %.loc34_19.4: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc34_19.4 [symbolic = %Iterate.facet.loc34_19.5 (constants.%Iterate.facet.6be)]
// CHECK:STDOUT:     %.loc34_19.5: %Destroy.type = converted constants.%as_type.73e, constants.%impl.elem1.5af [symbolic = %impl.elem1 (constants.%impl.elem1.5af)]
// CHECK:STDOUT:     %.loc34_19.6: %Destroy.type = converted constants.%as_type.73e, constants.%impl.elem1.5af [symbolic = %impl.elem1 (constants.%impl.elem1.5af)]
// CHECK:STDOUT:     %specific_impl_fn.loc34_19.2: <specific function> = specific_impl_function %impl.elem3.loc34_19.1, @Iterate.WithSelf.Next(constants.%Iterate.facet.6be) [symbolic = %specific_impl_fn.loc34_19.5 (constants.%specific_impl_fn.e73)]
// CHECK:STDOUT:     %bound_method.loc34_19.4: <bound method> = bound_method %r.ref, %specific_impl_fn.loc34_19.2
// CHECK:STDOUT:     %.loc34_19.7: ref %Optional.117 = temporary_storage
// CHECK:STDOUT:     %Iterate.WithSelf.Next.call: init %Optional.117 to %.loc34_19.7 = call %bound_method.loc34_19.4(%r.ref, %var)
// CHECK:STDOUT:     %.loc34_19.8: ref %Optional.117 = temporary %.loc34_19.7, %Iterate.WithSelf.Next.call
// CHECK:STDOUT:     %.loc34_19.9: %Optional.HasValue.type.ced = specific_constant imports.%Core.import_ref.4b2, @Optional(constants.%OptionalStorage.facet.ac7) [concrete = constants.%Optional.HasValue.615]
// CHECK:STDOUT:     %HasValue.ref: %Optional.HasValue.type.ced = name_ref HasValue, %.loc34_19.9 [concrete = constants.%Optional.HasValue.615]
// CHECK:STDOUT:     %Optional.HasValue.bound: <bound method> = bound_method %.loc34_19.8, %HasValue.ref
// CHECK:STDOUT:     %Optional.HasValue.specific_fn: <specific function> = specific_function %HasValue.ref, @Optional.HasValue(constants.%OptionalStorage.facet.ac7) [concrete = constants.%Optional.HasValue.specific_fn.bc7]
// CHECK:STDOUT:     %bound_method.loc34_19.5: <bound method> = bound_method %.loc34_19.8, %Optional.HasValue.specific_fn
// CHECK:STDOUT:     %.loc34_19.10: %Optional.117 = acquire_value %.loc34_19.8
// CHECK:STDOUT:     %Optional.HasValue.call: init bool = call %bound_method.loc34_19.5(%.loc34_19.10)
// CHECK:STDOUT:     %.loc34_19.11: bool = value_of_initializer %Optional.HasValue.call
// CHECK:STDOUT:     %.loc34_19.12: bool = converted %Optional.HasValue.call, %.loc34_19.11
// CHECK:STDOUT:     if %.loc34_19.12 br !for.body else br !for.done
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.body:
// CHECK:STDOUT:     %.loc34_19.13: %Optional.Get.type.703 = specific_constant imports.%Core.import_ref.52c, @Optional(constants.%OptionalStorage.facet.ac7) [concrete = constants.%Optional.Get.5fc]
// CHECK:STDOUT:     %Get.ref: %Optional.Get.type.703 = name_ref Get, %.loc34_19.13 [concrete = constants.%Optional.Get.5fc]
// CHECK:STDOUT:     %Optional.Get.bound: <bound method> = bound_method %.loc34_19.8, %Get.ref
// CHECK:STDOUT:     %Optional.Get.specific_fn: <specific function> = specific_function %Get.ref, @Optional.Get(constants.%OptionalStorage.facet.ac7) [concrete = constants.%Optional.Get.specific_fn.1c0]
// CHECK:STDOUT:     %bound_method.loc34_19.6: <bound method> = bound_method %.loc34_19.8, %Optional.Get.specific_fn
// CHECK:STDOUT:     %.loc34_19.14: %Optional.117 = acquire_value %.loc34_19.8
// CHECK:STDOUT:     %Optional.Get.call: init %i32 = call %bound_method.loc34_19.6(%.loc34_19.14)
// CHECK:STDOUT:     %.loc34_19.15: %i32 = value_of_initializer %Optional.Get.call
// CHECK:STDOUT:     %.loc34_19.16: %i32 = converted %Optional.Get.call, %.loc34_19.15
// CHECK:STDOUT:     %i32.loc34: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %i: %i32 = wrapper_binding i, %.loc34_19.16
// CHECK:STDOUT:     %sum.ref: ref %i32 = name_ref sum, %sum
// CHECK:STDOUT:     %i.ref: %i32 = name_ref i, %i
// CHECK:STDOUT:     %impl.elem0.loc35: %.2f4 = impl_witness_access constants.%AddAssignWith.impl_witness.2a4, element0 [concrete = constants.%Int.as.AddAssignWith.impl.Op.bd35de.2]
// CHECK:STDOUT:     %bound_method.loc35_9.1: <bound method> = bound_method %sum.ref, %impl.elem0.loc35
// CHECK:STDOUT:     %specific_fn.loc35: <specific function> = specific_function %impl.elem0.loc35, @Int.as.AddAssignWith.impl.Op.1(constants.%int_32, constants.%ImplicitAs.facet.371) [concrete = constants.%Int.as.AddAssignWith.impl.Op.specific_fn.3157f3.1]
// CHECK:STDOUT:     %bound_method.loc35_9.2: <bound method> = bound_method %sum.ref, %specific_fn.loc35
// CHECK:STDOUT:     %.loc35: %Int.as.AddAssignWith.impl.Op.type.303b18.1 = specific_constant imports.%Core.Op.b5f, @Int.as.AddAssignWith.impl(constants.%int_32, constants.%ImplicitAs.facet.371) [concrete = constants.%Int.as.AddAssignWith.impl.Op.bd35de.1]
// CHECK:STDOUT:     %Op.ref: %Int.as.AddAssignWith.impl.Op.type.303b18.1 = name_ref Op, %.loc35 [concrete = constants.%Int.as.AddAssignWith.impl.Op.bd35de.1]
// CHECK:STDOUT:     %Int.as.AddAssignWith.impl.Op.bound: <bound method> = bound_method %sum.ref, %Op.ref
// CHECK:STDOUT:     %Int.as.AddAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref, @Int.as.AddAssignWith.impl.Op.2(constants.%int_32, constants.%ImplicitAs.facet.371) [concrete = constants.%Int.as.AddAssignWith.impl.Op.specific_fn.3157f3.2]
// CHECK:STDOUT:     %bound_method.loc35_9.3: <bound method> = bound_method %sum.ref, %Int.as.AddAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:     %Int.as.AddAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc35_9.3(%sum.ref, %i.ref)
// CHECK:STDOUT:     %Destroy.WithSelf.SelfDestruct.bound.loc34_19.1: <bound method> = bound_method %.loc34_19.8, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.7
// CHECK:STDOUT:     %Destroy.WithSelf.SelfDestruct.call.loc34_19.1: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc34_19.1(%.loc34_19.8)
// CHECK:STDOUT:     br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.done:
// CHECK:STDOUT:     %Destroy.WithSelf.SelfDestruct.bound.loc34_19.2: <bound method> = bound_method %.loc34_19.8, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.7
// CHECK:STDOUT:     %Destroy.WithSelf.SelfDestruct.call.loc34_19.2: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc34_19.2(%.loc34_19.8)
// CHECK:STDOUT:     %impl.elem2.loc34_19.2: @TestIterate.%.loc34_19.21 (%.ed4) = impl_witness_access constants.%Destroy.lookup_impl_witness.690, element2 [symbolic = %impl.elem2.loc34_19.4 (constants.%impl.elem2.e0f)]
// CHECK:STDOUT:     %bound_method.loc34_19.7: <bound method> = bound_method %var, %impl.elem2.loc34_19.2
// CHECK:STDOUT:     %.loc34_19.17: %Destroy.type = converted constants.%as_type.73e, constants.%impl.elem1.5af [symbolic = %impl.elem1 (constants.%impl.elem1.5af)]
// CHECK:STDOUT:     %.loc34_19.18: %Destroy.type = converted constants.%as_type.73e, constants.%impl.elem1.5af [symbolic = %impl.elem1 (constants.%impl.elem1.5af)]
// CHECK:STDOUT:     %specific_impl_fn.loc34_19.3: <specific function> = specific_impl_function %impl.elem2.loc34_19.2, @Destroy.WithSelf.SelfDestruct.1(constants.%impl.elem1.5af) [symbolic = %specific_impl_fn.loc34_19.6 (constants.%specific_impl_fn.6b4)]
// CHECK:STDOUT:     %bound_method.loc34_19.8: <bound method> = bound_method %var, %specific_impl_fn.loc34_19.3
// CHECK:STDOUT:     %Destroy.WithSelf.SelfDestruct.call.loc34_19.3: init %empty_tuple.type = call %bound_method.loc34_19.8(%var)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc34_19.1(%self.param: ref %.473) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc34_19.1(%self.param: ref %.473) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc34_19.1(%self.param: ref %.473) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.1(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.1(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc34_19.2(%self.param: ref %MaybeUnformed.0a2) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc34_19.2(%self.param: ref %MaybeUnformed.0a2) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc34_19.2(%self.param: ref %MaybeUnformed.0a2) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.2(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.2(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc34_19.3(%self.param: ref bool) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc34_19.3(%self.param: ref bool) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc34_19.3(%self.param: ref bool) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.3(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.3(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc34_19.4(%self.param: ref %struct_type.value.has_value.88c) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc34_19.1: ref bool = struct_access %self.param, element1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc34_19.1: <bound method> = bound_method %.loc34_19.1, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.4
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc34_19.1: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc34_19.1(%.loc34_19.1)
// CHECK:STDOUT:   %.loc34_19.2: ref %MaybeUnformed.0a2 = struct_access %self.param, element0
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc34_19.2: <bound method> = bound_method %.loc34_19.2, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.3
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc34_19.2: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc34_19.2(%.loc34_19.2)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc34_19.4(%self.param: ref %struct_type.value.has_value.88c) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc34_19.4(%self.param: ref %struct_type.value.has_value.88c) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.4(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.4(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc34_19.5(%self.param: ref %DefaultOptionalStorage.78c) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %has_value.ref: %DefaultOptionalStorage.elem.ad8 = name_ref has_value, imports.%Core.import_ref.929 [concrete = imports.%field_decl.6a8]
// CHECK:STDOUT:   %.loc34_19.1: ref bool = class_element_access %self.param, element1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc34_19.1: <bound method> = bound_method %.loc34_19.1, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.4
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc34_19.1: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc34_19.1(%.loc34_19.1)
// CHECK:STDOUT:   %value.ref: %DefaultOptionalStorage.elem.76a = name_ref value, imports.%Core.import_ref.1d7 [concrete = imports.%field_decl.5f0]
// CHECK:STDOUT:   %.loc34_19.2: ref %MaybeUnformed.0a2 = class_element_access %self.param, element0
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc34_19.2: <bound method> = bound_method %.loc34_19.2, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.3
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc34_19.2: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc34_19.2(%.loc34_19.2)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc34_19.5(%self.param: ref %DefaultOptionalStorage.78c) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc34_19.5(%self.param: ref %DefaultOptionalStorage.78c) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.5(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.5(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc34_19.6(%self.param: ref %Optional.117) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc34_19.6(%self.param: ref %Optional.117) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc34_19.6(%self.param: ref %Optional.117) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.6(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.6(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestRangeFor(%m.param: ref %MutableRange, %c.param: %ConstRange) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %i.patt.loc49: %pattern_type.9d9 = value_binding_pattern i [concrete = constants.%i.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %m.ref: ref %MutableRange = name_ref m, %m
// CHECK:STDOUT:   %impl.elem2.loc49: %.d0d = impl_witness_access constants.%Iterate.impl_witness.230, element2 [concrete = constants.%T.as_type.as.Iterate.impl.NewCursor.067]
// CHECK:STDOUT:   %bound_method.loc49_19.1: <bound method> = bound_method %m.ref, %impl.elem2.loc49
// CHECK:STDOUT:   %facet_value.loc49_19.1: %CppRange_where.type.d682d6.1 = facet_value constants.%MutableRange, (constants.%custom_witness.ed3, constants.%custom_witness.8e1, constants.%custom_witness.b38, constants.%custom_witness.444, constants.%custom_witness.c91, constants.%custom_witness.f8f19d.1, constants.%Copy.impl_witness.0ff, constants.%custom_witness.8e1) [concrete = constants.%facet_value.84d]
// CHECK:STDOUT:   %.loc49_19.1: %CppRange_where.type.d682d6.1 = converted constants.%MutableRange, %facet_value.loc49_19.1 [concrete = constants.%facet_value.84d]
// CHECK:STDOUT:   %facet_value.loc49_19.2: %CppRange_where.type.d682d6.1 = facet_value constants.%MutableRange, (constants.%custom_witness.ed3, constants.%custom_witness.8e1, constants.%custom_witness.b38, constants.%custom_witness.444, constants.%custom_witness.c91, constants.%custom_witness.f8f19d.1, constants.%Copy.impl_witness.0ff, constants.%custom_witness.8e1) [concrete = constants.%facet_value.84d]
// CHECK:STDOUT:   %.loc49_19.2: %CppRange_where.type.d682d6.1 = converted constants.%MutableRange, %facet_value.loc49_19.2 [concrete = constants.%facet_value.84d]
// CHECK:STDOUT:   %specific_fn.loc49_19.1: <specific function> = specific_function %impl.elem2.loc49, @T.as_type.as.Iterate.impl.NewCursor(constants.%facet_value.84d) [concrete = constants.%T.as_type.as.Iterate.impl.NewCursor.specific_fn.212]
// CHECK:STDOUT:   %bound_method.loc49_19.2: <bound method> = bound_method %m.ref, %specific_fn.loc49_19.1
// CHECK:STDOUT:   %var.loc49: ref %tuple.type.508 = var_storage invalid
// CHECK:STDOUT:   %.loc49_18.1: %MutableRange = acquire_value %m.ref
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.call.loc49: init %tuple.type.508 to %var.loc49 = call %bound_method.loc49_19.2(%.loc49_18.1)
// CHECK:STDOUT:   assign %var.loc49, %T.as_type.as.Iterate.impl.NewCursor.call.loc49
// CHECK:STDOUT:   br !for.next.loc49
// CHECK:STDOUT:
// CHECK:STDOUT: !for.next.loc49:
// CHECK:STDOUT:   %impl.elem3.loc49: %.0cc = impl_witness_access constants.%Iterate.impl_witness.230, element3 [concrete = constants.%T.as_type.as.Iterate.impl.Next.f95]
// CHECK:STDOUT:   %bound_method.loc49_19.3: <bound method> = bound_method %m.ref, %impl.elem3.loc49
// CHECK:STDOUT:   %facet_value.loc49_19.3: %CppRange_where.type.d682d6.1 = facet_value constants.%MutableRange, (constants.%custom_witness.ed3, constants.%custom_witness.8e1, constants.%custom_witness.b38, constants.%custom_witness.444, constants.%custom_witness.c91, constants.%custom_witness.f8f19d.1, constants.%Copy.impl_witness.0ff, constants.%custom_witness.8e1) [concrete = constants.%facet_value.84d]
// CHECK:STDOUT:   %.loc49_19.3: %CppRange_where.type.d682d6.1 = converted constants.%MutableRange, %facet_value.loc49_19.3 [concrete = constants.%facet_value.84d]
// CHECK:STDOUT:   %facet_value.loc49_19.4: %CppRange_where.type.d682d6.1 = facet_value constants.%MutableRange, (constants.%custom_witness.ed3, constants.%custom_witness.8e1, constants.%custom_witness.b38, constants.%custom_witness.444, constants.%custom_witness.c91, constants.%custom_witness.f8f19d.1, constants.%Copy.impl_witness.0ff, constants.%custom_witness.8e1) [concrete = constants.%facet_value.84d]
// CHECK:STDOUT:   %.loc49_19.4: %CppRange_where.type.d682d6.1 = converted constants.%MutableRange, %facet_value.loc49_19.4 [concrete = constants.%facet_value.84d]
// CHECK:STDOUT:   %specific_fn.loc49_19.2: <specific function> = specific_function %impl.elem3.loc49, @T.as_type.as.Iterate.impl.Next(constants.%facet_value.84d) [concrete = constants.%T.as_type.as.Iterate.impl.Next.specific_fn.4c7]
// CHECK:STDOUT:   %bound_method.loc49_19.4: <bound method> = bound_method %m.ref, %specific_fn.loc49_19.2
// CHECK:STDOUT:   %.loc49_19.5: ref %Optional.117 = temporary_storage
// CHECK:STDOUT:   %.loc49_18.2: %MutableRange = acquire_value %m.ref
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.call.loc49: init %Optional.117 to %.loc49_19.5 = call %bound_method.loc49_19.4(%.loc49_18.2, %var.loc49)
// CHECK:STDOUT:   %.loc49_19.6: ref %Optional.117 = temporary %.loc49_19.5, %T.as_type.as.Iterate.impl.Next.call.loc49
// CHECK:STDOUT:   %.loc49_19.7: %Optional.HasValue.type.ced = specific_constant imports.%Core.import_ref.4b2, @Optional(constants.%OptionalStorage.facet.ac7) [concrete = constants.%Optional.HasValue.615]
// CHECK:STDOUT:   %HasValue.ref.loc49: %Optional.HasValue.type.ced = name_ref HasValue, %.loc49_19.7 [concrete = constants.%Optional.HasValue.615]
// CHECK:STDOUT:   %Optional.HasValue.bound.loc49: <bound method> = bound_method %.loc49_19.6, %HasValue.ref.loc49
// CHECK:STDOUT:   %Optional.HasValue.specific_fn.loc49: <specific function> = specific_function %HasValue.ref.loc49, @Optional.HasValue(constants.%OptionalStorage.facet.ac7) [concrete = constants.%Optional.HasValue.specific_fn.bc7]
// CHECK:STDOUT:   %bound_method.loc49_19.5: <bound method> = bound_method %.loc49_19.6, %Optional.HasValue.specific_fn.loc49
// CHECK:STDOUT:   %.loc49_19.8: %Optional.117 = acquire_value %.loc49_19.6
// CHECK:STDOUT:   %Optional.HasValue.call.loc49: init bool = call %bound_method.loc49_19.5(%.loc49_19.8)
// CHECK:STDOUT:   %.loc49_19.9: bool = value_of_initializer %Optional.HasValue.call.loc49
// CHECK:STDOUT:   %.loc49_19.10: bool = converted %Optional.HasValue.call.loc49, %.loc49_19.9
// CHECK:STDOUT:   if %.loc49_19.10 br !for.body.loc49 else br !for.done.loc49
// CHECK:STDOUT:
// CHECK:STDOUT: !for.body.loc49:
// CHECK:STDOUT:   %.loc49_19.11: %Optional.Get.type.703 = specific_constant imports.%Core.import_ref.52c, @Optional(constants.%OptionalStorage.facet.ac7) [concrete = constants.%Optional.Get.5fc]
// CHECK:STDOUT:   %Get.ref.loc49: %Optional.Get.type.703 = name_ref Get, %.loc49_19.11 [concrete = constants.%Optional.Get.5fc]
// CHECK:STDOUT:   %Optional.Get.bound.loc49: <bound method> = bound_method %.loc49_19.6, %Get.ref.loc49
// CHECK:STDOUT:   %Optional.Get.specific_fn.loc49: <specific function> = specific_function %Get.ref.loc49, @Optional.Get(constants.%OptionalStorage.facet.ac7) [concrete = constants.%Optional.Get.specific_fn.1c0]
// CHECK:STDOUT:   %bound_method.loc49_19.6: <bound method> = bound_method %.loc49_19.6, %Optional.Get.specific_fn.loc49
// CHECK:STDOUT:   %.loc49_19.12: %Optional.117 = acquire_value %.loc49_19.6
// CHECK:STDOUT:   %Optional.Get.call.loc49: init %i32 = call %bound_method.loc49_19.6(%.loc49_19.12)
// CHECK:STDOUT:   %.loc49_19.13: %i32 = value_of_initializer %Optional.Get.call.loc49
// CHECK:STDOUT:   %.loc49_19.14: %i32 = converted %Optional.Get.call.loc49, %.loc49_19.13
// CHECK:STDOUT:   %i32.loc49: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %i.loc49: %i32 = wrapper_binding i, %.loc49_19.14
// CHECK:STDOUT:   %sum.ref.loc50: ref %i32 = name_ref sum, %sum
// CHECK:STDOUT:   %i.ref.loc50: %i32 = name_ref i, %i.loc49
// CHECK:STDOUT:   %impl.elem0.loc50: %.2f4 = impl_witness_access constants.%AddAssignWith.impl_witness.2a4, element0 [concrete = constants.%Int.as.AddAssignWith.impl.Op.bd35de.2]
// CHECK:STDOUT:   %bound_method.loc50_9.1: <bound method> = bound_method %sum.ref.loc50, %impl.elem0.loc50
// CHECK:STDOUT:   %specific_fn.loc50: <specific function> = specific_function %impl.elem0.loc50, @Int.as.AddAssignWith.impl.Op.1(constants.%int_32, constants.%ImplicitAs.facet.371) [concrete = constants.%Int.as.AddAssignWith.impl.Op.specific_fn.3157f3.1]
// CHECK:STDOUT:   %bound_method.loc50_9.2: <bound method> = bound_method %sum.ref.loc50, %specific_fn.loc50
// CHECK:STDOUT:   %.loc50: %Int.as.AddAssignWith.impl.Op.type.303b18.1 = specific_constant imports.%Core.Op.b5f, @Int.as.AddAssignWith.impl(constants.%int_32, constants.%ImplicitAs.facet.371) [concrete = constants.%Int.as.AddAssignWith.impl.Op.bd35de.1]
// CHECK:STDOUT:   %Op.ref.loc50: %Int.as.AddAssignWith.impl.Op.type.303b18.1 = name_ref Op, %.loc50 [concrete = constants.%Int.as.AddAssignWith.impl.Op.bd35de.1]
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.bound.loc50: <bound method> = bound_method %sum.ref.loc50, %Op.ref.loc50
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.specific_fn.loc50: <specific function> = specific_function %Op.ref.loc50, @Int.as.AddAssignWith.impl.Op.2(constants.%int_32, constants.%ImplicitAs.facet.371) [concrete = constants.%Int.as.AddAssignWith.impl.Op.specific_fn.3157f3.2]
// CHECK:STDOUT:   %bound_method.loc50_9.3: <bound method> = bound_method %sum.ref.loc50, %Int.as.AddAssignWith.impl.Op.specific_fn.loc50
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.call.loc50: init %empty_tuple.type = call %bound_method.loc50_9.3(%sum.ref.loc50, %i.ref.loc50)
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc49_19.1: <bound method> = bound_method %.loc49_19.6, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.7
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc49_19.1: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc49_19.1(%.loc49_19.6)
// CHECK:STDOUT:   br !for.next.loc49
// CHECK:STDOUT:
// CHECK:STDOUT: !for.done.loc49:
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc49_19.2: <bound method> = bound_method %.loc49_19.6, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.7
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc49_19.2: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc49_19.2(%.loc49_19.6)
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc49_19.3: <bound method> = bound_method %var.loc49, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.9
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc49_19.3: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc49_19.3(%var.loc49)
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %i.patt.loc55: %pattern_type.9d9 = value_binding_pattern i [concrete = constants.%i.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c.ref: %ConstRange = name_ref c, %c
// CHECK:STDOUT:   %impl.elem2.loc55: %.fd0 = impl_witness_access constants.%Iterate.impl_witness.01e, element2 [concrete = constants.%T.as_type.as.Iterate.impl.NewCursor.644]
// CHECK:STDOUT:   %bound_method.loc55_19.1: <bound method> = bound_method %c.ref, %impl.elem2.loc55
// CHECK:STDOUT:   %facet_value.loc55_19.1: %CppRange_where.type.d682d6.1 = facet_value constants.%ConstRange, (constants.%custom_witness.23e, constants.%custom_witness.0f3, constants.%custom_witness.c00, constants.%custom_witness.746, constants.%custom_witness.86f, constants.%custom_witness.f8f19d.1, constants.%Copy.impl_witness.0ff, constants.%custom_witness.0f3) [concrete = constants.%facet_value.793]
// CHECK:STDOUT:   %.loc55_19.1: %CppRange_where.type.d682d6.1 = converted constants.%ConstRange, %facet_value.loc55_19.1 [concrete = constants.%facet_value.793]
// CHECK:STDOUT:   %facet_value.loc55_19.2: %CppRange_where.type.d682d6.1 = facet_value constants.%ConstRange, (constants.%custom_witness.23e, constants.%custom_witness.0f3, constants.%custom_witness.c00, constants.%custom_witness.746, constants.%custom_witness.86f, constants.%custom_witness.f8f19d.1, constants.%Copy.impl_witness.0ff, constants.%custom_witness.0f3) [concrete = constants.%facet_value.793]
// CHECK:STDOUT:   %.loc55_19.2: %CppRange_where.type.d682d6.1 = converted constants.%ConstRange, %facet_value.loc55_19.2 [concrete = constants.%facet_value.793]
// CHECK:STDOUT:   %specific_fn.loc55_19.1: <specific function> = specific_function %impl.elem2.loc55, @T.as_type.as.Iterate.impl.NewCursor(constants.%facet_value.793) [concrete = constants.%T.as_type.as.Iterate.impl.NewCursor.specific_fn.b5f]
// CHECK:STDOUT:   %bound_method.loc55_19.2: <bound method> = bound_method %c.ref, %specific_fn.loc55_19.1
// CHECK:STDOUT:   %var.loc55: ref %tuple.type.69f = var_storage invalid
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.call.loc55: init %tuple.type.69f to %var.loc55 = call %bound_method.loc55_19.2(%c.ref)
// CHECK:STDOUT:   assign %var.loc55, %T.as_type.as.Iterate.impl.NewCursor.call.loc55
// CHECK:STDOUT:   br !for.next.loc55
// CHECK:STDOUT:
// CHECK:STDOUT: !for.next.loc55:
// CHECK:STDOUT:   %impl.elem3.loc55: %.d49 = impl_witness_access constants.%Iterate.impl_witness.01e, element3 [concrete = constants.%T.as_type.as.Iterate.impl.Next.b53]
// CHECK:STDOUT:   %bound_method.loc55_19.3: <bound method> = bound_method %c.ref, %impl.elem3.loc55
// CHECK:STDOUT:   %facet_value.loc55_19.3: %CppRange_where.type.d682d6.1 = facet_value constants.%ConstRange, (constants.%custom_witness.23e, constants.%custom_witness.0f3, constants.%custom_witness.c00, constants.%custom_witness.746, constants.%custom_witness.86f, constants.%custom_witness.f8f19d.1, constants.%Copy.impl_witness.0ff, constants.%custom_witness.0f3) [concrete = constants.%facet_value.793]
// CHECK:STDOUT:   %.loc55_19.3: %CppRange_where.type.d682d6.1 = converted constants.%ConstRange, %facet_value.loc55_19.3 [concrete = constants.%facet_value.793]
// CHECK:STDOUT:   %facet_value.loc55_19.4: %CppRange_where.type.d682d6.1 = facet_value constants.%ConstRange, (constants.%custom_witness.23e, constants.%custom_witness.0f3, constants.%custom_witness.c00, constants.%custom_witness.746, constants.%custom_witness.86f, constants.%custom_witness.f8f19d.1, constants.%Copy.impl_witness.0ff, constants.%custom_witness.0f3) [concrete = constants.%facet_value.793]
// CHECK:STDOUT:   %.loc55_19.4: %CppRange_where.type.d682d6.1 = converted constants.%ConstRange, %facet_value.loc55_19.4 [concrete = constants.%facet_value.793]
// CHECK:STDOUT:   %specific_fn.loc55_19.2: <specific function> = specific_function %impl.elem3.loc55, @T.as_type.as.Iterate.impl.Next(constants.%facet_value.793) [concrete = constants.%T.as_type.as.Iterate.impl.Next.specific_fn.f07]
// CHECK:STDOUT:   %bound_method.loc55_19.4: <bound method> = bound_method %c.ref, %specific_fn.loc55_19.2
// CHECK:STDOUT:   %.loc55_19.5: ref %Optional.117 = temporary_storage
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.call.loc55: init %Optional.117 to %.loc55_19.5 = call %bound_method.loc55_19.4(%c.ref, %var.loc55)
// CHECK:STDOUT:   %.loc55_19.6: ref %Optional.117 = temporary %.loc55_19.5, %T.as_type.as.Iterate.impl.Next.call.loc55
// CHECK:STDOUT:   %.loc55_19.7: %Optional.HasValue.type.ced = specific_constant imports.%Core.import_ref.4b2, @Optional(constants.%OptionalStorage.facet.ac7) [concrete = constants.%Optional.HasValue.615]
// CHECK:STDOUT:   %HasValue.ref.loc55: %Optional.HasValue.type.ced = name_ref HasValue, %.loc55_19.7 [concrete = constants.%Optional.HasValue.615]
// CHECK:STDOUT:   %Optional.HasValue.bound.loc55: <bound method> = bound_method %.loc55_19.6, %HasValue.ref.loc55
// CHECK:STDOUT:   %Optional.HasValue.specific_fn.loc55: <specific function> = specific_function %HasValue.ref.loc55, @Optional.HasValue(constants.%OptionalStorage.facet.ac7) [concrete = constants.%Optional.HasValue.specific_fn.bc7]
// CHECK:STDOUT:   %bound_method.loc55_19.5: <bound method> = bound_method %.loc55_19.6, %Optional.HasValue.specific_fn.loc55
// CHECK:STDOUT:   %.loc55_19.8: %Optional.117 = acquire_value %.loc55_19.6
// CHECK:STDOUT:   %Optional.HasValue.call.loc55: init bool = call %bound_method.loc55_19.5(%.loc55_19.8)
// CHECK:STDOUT:   %.loc55_19.9: bool = value_of_initializer %Optional.HasValue.call.loc55
// CHECK:STDOUT:   %.loc55_19.10: bool = converted %Optional.HasValue.call.loc55, %.loc55_19.9
// CHECK:STDOUT:   if %.loc55_19.10 br !for.body.loc55 else br !for.done.loc55
// CHECK:STDOUT:
// CHECK:STDOUT: !for.body.loc55:
// CHECK:STDOUT:   %.loc55_19.11: %Optional.Get.type.703 = specific_constant imports.%Core.import_ref.52c, @Optional(constants.%OptionalStorage.facet.ac7) [concrete = constants.%Optional.Get.5fc]
// CHECK:STDOUT:   %Get.ref.loc55: %Optional.Get.type.703 = name_ref Get, %.loc55_19.11 [concrete = constants.%Optional.Get.5fc]
// CHECK:STDOUT:   %Optional.Get.bound.loc55: <bound method> = bound_method %.loc55_19.6, %Get.ref.loc55
// CHECK:STDOUT:   %Optional.Get.specific_fn.loc55: <specific function> = specific_function %Get.ref.loc55, @Optional.Get(constants.%OptionalStorage.facet.ac7) [concrete = constants.%Optional.Get.specific_fn.1c0]
// CHECK:STDOUT:   %bound_method.loc55_19.6: <bound method> = bound_method %.loc55_19.6, %Optional.Get.specific_fn.loc55
// CHECK:STDOUT:   %.loc55_19.12: %Optional.117 = acquire_value %.loc55_19.6
// CHECK:STDOUT:   %Optional.Get.call.loc55: init %i32 = call %bound_method.loc55_19.6(%.loc55_19.12)
// CHECK:STDOUT:   %.loc55_19.13: %i32 = value_of_initializer %Optional.Get.call.loc55
// CHECK:STDOUT:   %.loc55_19.14: %i32 = converted %Optional.Get.call.loc55, %.loc55_19.13
// CHECK:STDOUT:   %i32.loc55: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %i.loc55: %i32 = wrapper_binding i, %.loc55_19.14
// CHECK:STDOUT:   %sum.ref.loc56: ref %i32 = name_ref sum, %sum
// CHECK:STDOUT:   %i.ref.loc56: %i32 = name_ref i, %i.loc55
// CHECK:STDOUT:   %impl.elem0.loc56: %.2f4 = impl_witness_access constants.%AddAssignWith.impl_witness.2a4, element0 [concrete = constants.%Int.as.AddAssignWith.impl.Op.bd35de.2]
// CHECK:STDOUT:   %bound_method.loc56_9.1: <bound method> = bound_method %sum.ref.loc56, %impl.elem0.loc56
// CHECK:STDOUT:   %specific_fn.loc56: <specific function> = specific_function %impl.elem0.loc56, @Int.as.AddAssignWith.impl.Op.1(constants.%int_32, constants.%ImplicitAs.facet.371) [concrete = constants.%Int.as.AddAssignWith.impl.Op.specific_fn.3157f3.1]
// CHECK:STDOUT:   %bound_method.loc56_9.2: <bound method> = bound_method %sum.ref.loc56, %specific_fn.loc56
// CHECK:STDOUT:   %.loc56: %Int.as.AddAssignWith.impl.Op.type.303b18.1 = specific_constant imports.%Core.Op.b5f, @Int.as.AddAssignWith.impl(constants.%int_32, constants.%ImplicitAs.facet.371) [concrete = constants.%Int.as.AddAssignWith.impl.Op.bd35de.1]
// CHECK:STDOUT:   %Op.ref.loc56: %Int.as.AddAssignWith.impl.Op.type.303b18.1 = name_ref Op, %.loc56 [concrete = constants.%Int.as.AddAssignWith.impl.Op.bd35de.1]
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.bound.loc56: <bound method> = bound_method %sum.ref.loc56, %Op.ref.loc56
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.specific_fn.loc56: <specific function> = specific_function %Op.ref.loc56, @Int.as.AddAssignWith.impl.Op.2(constants.%int_32, constants.%ImplicitAs.facet.371) [concrete = constants.%Int.as.AddAssignWith.impl.Op.specific_fn.3157f3.2]
// CHECK:STDOUT:   %bound_method.loc56_9.3: <bound method> = bound_method %sum.ref.loc56, %Int.as.AddAssignWith.impl.Op.specific_fn.loc56
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.call.loc56: init %empty_tuple.type = call %bound_method.loc56_9.3(%sum.ref.loc56, %i.ref.loc56)
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc55_19.1: <bound method> = bound_method %.loc55_19.6, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.7
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc55_19.1: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc55_19.1(%.loc55_19.6)
// CHECK:STDOUT:   br !for.next.loc55
// CHECK:STDOUT:
// CHECK:STDOUT: !for.done.loc55:
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc55_19.2: <bound method> = bound_method %.loc55_19.6, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.7
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc55_19.2: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc55_19.2(%.loc55_19.6)
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc55_19.3: <bound method> = bound_method %var.loc55, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.11
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc55_19.3: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc55_19.3(%var.loc55)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestIterate(constants.%R) {
// CHECK:STDOUT:   %R.patt.loc30_17.2 => constants.%R.patt
// CHECK:STDOUT:   %R.loc30_17.1 => constants.%R
// CHECK:STDOUT:   %R.as_type.loc30_61.1 => constants.%R.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.3dc
// CHECK:STDOUT:   %r.param_patt.loc30_59.2 => constants.%r.param_patt.ab7
// CHECK:STDOUT:   %r.patt.loc30_59.2 => constants.%r.patt.531
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestIterate(constants.%facet_value.904) {
// CHECK:STDOUT:   %R.patt.loc30_17.2 => constants.%R.patt
// CHECK:STDOUT:   %R.loc30_17.1 => constants.%facet_value.904
// CHECK:STDOUT:   %R.as_type.loc30_61.1 => constants.%MutableRange
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.e63
// CHECK:STDOUT:   %r.param_patt.loc30_59.2 => constants.%r.param_patt.c79
// CHECK:STDOUT:   %r.patt.loc30_59.2 => constants.%r.patt.94f
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc30 => constants.%complete_type.357
// CHECK:STDOUT:   %Iterate.lookup_impl_witness => constants.%Iterate.impl_witness.230
// CHECK:STDOUT:   %Iterate.facet.loc34_19.5 => constants.%Iterate.facet.af6
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type => constants.%Iterate.WithSelf.NewCursor.type.ade
// CHECK:STDOUT:   %.loc34_19.19 => constants.%.d0d
// CHECK:STDOUT:   %impl.elem2.loc34_19.3 => constants.%T.as_type.as.Iterate.impl.NewCursor.067
// CHECK:STDOUT:   %specific_impl_fn.loc34_19.4 => constants.%T.as_type.as.Iterate.impl.NewCursor.specific_fn.212
// CHECK:STDOUT:   %impl.elem1 => constants.%Destroy.facet.0fe
// CHECK:STDOUT:   %as_type => constants.%tuple.type.508
// CHECK:STDOUT:   %require_complete.1 => constants.%complete_type.c28
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type => constants.%Iterate.WithSelf.Next.type.949
// CHECK:STDOUT:   %.loc34_19.20 => constants.%.0cc
// CHECK:STDOUT:   %impl.elem3.loc34_19.2 => constants.%T.as_type.as.Iterate.impl.Next.f95
// CHECK:STDOUT:   %specific_impl_fn.loc34_19.5 => constants.%T.as_type.as.Iterate.impl.Next.specific_fn.4c7
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type => constants.%Destroy.WithSelf.SelfDestruct.type.d1f
// CHECK:STDOUT:   %.loc34_19.21 => constants.%.0dc
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.f8f19d.8
// CHECK:STDOUT:   %impl.elem2.loc34_19.4 => constants.%Destroy.WithSelf.SelfDestruct.db3fdb.9
// CHECK:STDOUT:   %specific_impl_fn.loc34_19.6 => constants.%Destroy.WithSelf.SelfDestruct.db3fdb.9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestIterate(constants.%facet_value.82c) {
// CHECK:STDOUT:   %R.patt.loc30_17.2 => constants.%R.patt
// CHECK:STDOUT:   %R.loc30_17.1 => constants.%facet_value.82c
// CHECK:STDOUT:   %R.as_type.loc30_61.1 => constants.%ConstRange
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.0ba
// CHECK:STDOUT:   %r.param_patt.loc30_59.2 => constants.%r.param_patt.83e
// CHECK:STDOUT:   %r.patt.loc30_59.2 => constants.%r.patt.c21
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc30 => constants.%complete_type.357
// CHECK:STDOUT:   %Iterate.lookup_impl_witness => constants.%Iterate.impl_witness.01e
// CHECK:STDOUT:   %Iterate.facet.loc34_19.5 => constants.%Iterate.facet.98a
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type => constants.%Iterate.WithSelf.NewCursor.type.867
// CHECK:STDOUT:   %.loc34_19.19 => constants.%.fd0
// CHECK:STDOUT:   %impl.elem2.loc34_19.3 => constants.%T.as_type.as.Iterate.impl.NewCursor.644
// CHECK:STDOUT:   %specific_impl_fn.loc34_19.4 => constants.%T.as_type.as.Iterate.impl.NewCursor.specific_fn.b5f
// CHECK:STDOUT:   %impl.elem1 => constants.%Destroy.facet.c6d
// CHECK:STDOUT:   %as_type => constants.%tuple.type.69f
// CHECK:STDOUT:   %require_complete.1 => constants.%complete_type.8d5
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type => constants.%Iterate.WithSelf.Next.type.5a5
// CHECK:STDOUT:   %.loc34_19.20 => constants.%.d49
// CHECK:STDOUT:   %impl.elem3.loc34_19.2 => constants.%T.as_type.as.Iterate.impl.Next.b53
// CHECK:STDOUT:   %specific_impl_fn.loc34_19.5 => constants.%T.as_type.as.Iterate.impl.Next.specific_fn.f07
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type => constants.%Destroy.WithSelf.SelfDestruct.type.b84
// CHECK:STDOUT:   %.loc34_19.21 => constants.%.40c
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.f8f19d.9
// CHECK:STDOUT:   %impl.elem2.loc34_19.4 => constants.%Destroy.WithSelf.SelfDestruct.db3fdb.11
// CHECK:STDOUT:   %specific_impl_fn.loc34_19.6 => constants.%Destroy.WithSelf.SelfDestruct.db3fdb.11
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
// CHECK:STDOUT:   %Optional.Get.type.c64: type = fn_type @Optional.Get, @Optional(%T.220) [symbolic]
// CHECK:STDOUT:   %Optional.Get.bcf: %Optional.Get.type.c64 = struct_value () [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.type.49e: type = fn_type @Optional.HasValue, @Optional(%T.220) [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.c20: %Optional.HasValue.type.49e = struct_value () [symbolic]
// CHECK:STDOUT:   %T.283: %facet_type.f48 = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.74b: type = facet_access_type %T.283 [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.28a: <witness> = lookup_impl_witness %T.283, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.16e: %Destroy.type = facet_value %T.as_type.74b, (%Destroy.lookup_impl_witness.28a) [symbolic]
// CHECK:STDOUT:   %MaybeUnformed.9b7: type = class_type @MaybeUnformed, @MaybeUnformed(%Destroy.facet.16e) [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.faa: type = class_type @DefaultOptionalStorage, @DefaultOptionalStorage(%T.283) [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.elem.06a: type = unbound_element_type %DefaultOptionalStorage.faa, bool [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.elem.9db: type = unbound_element_type %DefaultOptionalStorage.faa, %MaybeUnformed.9b7 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Get.type.2f2: type = fn_type @T.as_type.as.OptionalStorage.impl.Get, @T.as_type.as.OptionalStorage.impl(%T.283) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Get.884: %T.as_type.as.OptionalStorage.impl.Get.type.2f2 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Has.type.d28: type = fn_type @T.as_type.as.OptionalStorage.impl.Has, @T.as_type.as.OptionalStorage.impl(%T.283) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Has.f29: %T.as_type.as.OptionalStorage.impl.Has.type.d28 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Some.type.760: type = fn_type @T.as_type.as.OptionalStorage.impl.Some, @T.as_type.as.OptionalStorage.impl(%T.283) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Some.268: %T.as_type.as.OptionalStorage.impl.Some.type.760 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.None.type.925: type = fn_type @T.as_type.as.OptionalStorage.impl.None, @T.as_type.as.OptionalStorage.impl(%T.283) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.None.f42: %T.as_type.as.OptionalStorage.impl.None.type.925 = struct_value () [symbolic]
// CHECK:STDOUT:   %.Self.500: %Iterate.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %CppRange.type: type = facet_type <@CppRange> [concrete]
// CHECK:STDOUT:   %.Self.e0b: %CppRange.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %CppRangeForIterate.lookup_impl_witness.50b: <witness> = lookup_impl_witness %.Self.e0b, @CppRangeForIterate [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.32d: type = impl_witness_access %CppRangeForIterate.lookup_impl_witness.50b, element0 [symbolic_self]
// CHECK:STDOUT:   %CppUnsafeDeref.lookup_impl_witness.851: <witness> = lookup_impl_witness %impl.elem0.32d, @CppUnsafeDeref [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.10d: type = impl_witness_access %CppUnsafeDeref.lookup_impl_witness.851, element0 [symbolic_self]
// CHECK:STDOUT:   %CppRange_where.type.d682d6.1: type = facet_type <@CppRange where %impl.elem0.10d impls @Copy> [concrete]
// CHECK:STDOUT:   %T.f45530.2: %CppRange_where.type.d682d6.1 = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.type.d4a: type = fn_type @T.as_type.as.Iterate.impl.Next, @T.as_type.as.Iterate.impl(%T.f45530.2) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.5a8: %T.as_type.as.Iterate.impl.Next.type.d4a = struct_value () [symbolic]
// CHECK:STDOUT:   %CppRangeForIterate.lookup_impl_witness.b581a8.2: <witness> = lookup_impl_witness %T.f45530.2, @CppRangeForIterate [symbolic]
// CHECK:STDOUT:   %impl.elem0.5f971c.2: type = impl_witness_access %CppRangeForIterate.lookup_impl_witness.b581a8.2, element0 [symbolic]
// CHECK:STDOUT:   %CppUnsafeDeref.lookup_impl_witness.87e543.2: <witness> = lookup_impl_witness %impl.elem0.5f971c.2, @CppUnsafeDeref [symbolic]
// CHECK:STDOUT:   %impl.elem0.493765.2: type = impl_witness_access %CppUnsafeDeref.lookup_impl_witness.87e543.2, element0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.type.8fb: type = fn_type @T.as_type.as.Iterate.impl.NewCursor, @T.as_type.as.Iterate.impl(%T.f45530.2) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.8a1: %T.as_type.as.Iterate.impl.NewCursor.type.8fb = struct_value () [symbolic]
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
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.b5e: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.f03) [symbolic]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.931: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.f03) [symbolic]
// CHECK:STDOUT:   %.c17: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.b5e, %Iterate.facet.f03 [symbolic]
// CHECK:STDOUT:   %impl.elem2.3ea: %.c17 = impl_witness_access %Iterate.lookup_impl_witness.d00, element2 [symbolic]
// CHECK:STDOUT:   %impl.elem1.fbc: %Destroy.type = impl_witness_access %Iterate.lookup_impl_witness.d00, element1 [symbolic]
// CHECK:STDOUT:   %as_type.389: type = facet_access_type %impl.elem1.fbc [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.b2c: <specific function> = specific_impl_function %impl.elem2.3ea, @Iterate.WithSelf.NewCursor(%Iterate.facet.f03) [symbolic]
// CHECK:STDOUT:   %.c9b: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.931, %Iterate.facet.f03 [symbolic]
// CHECK:STDOUT:   %impl.elem3.0ff: %.c9b = impl_witness_access %Iterate.lookup_impl_witness.d00, element3 [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.21a: type = class_type @DefaultOptionalStorage, @DefaultOptionalStorage(%facet_value.cca) [concrete]
// CHECK:STDOUT:   %OptionalStorage.impl_witness.9ee: <witness> = impl_witness imports.%OptionalStorage.impl_witness_table.460, @T.as_type.as.OptionalStorage.impl(%facet_value.cca) [concrete]
// CHECK:STDOUT:   %OptionalStorage.facet.9e6: %OptionalStorage.type = facet_value %DiffValueType, (%OptionalStorage.impl_witness.9ee) [concrete]
// CHECK:STDOUT:   %Optional.6ee: type = class_type @Optional, @Optional(%OptionalStorage.facet.9e6) [concrete]
// CHECK:STDOUT:   %pattern_type.054: type = pattern_type %Optional.6ee [concrete]
// CHECK:STDOUT:   %specific_impl_fn.20b: <specific function> = specific_impl_function %impl.elem3.0ff, @Iterate.WithSelf.Next(%Iterate.facet.f03) [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.type.868: type = fn_type @Optional.HasValue, @Optional(%OptionalStorage.facet.9e6) [concrete]
// CHECK:STDOUT:   %Optional.HasValue.7d8: %Optional.HasValue.type.868 = struct_value () [concrete]
// CHECK:STDOUT:   %Optional.Get.type.ab0: type = fn_type @Optional.Get, @Optional(%OptionalStorage.facet.9e6) [concrete]
// CHECK:STDOUT:   %Optional.Get.6e2: %Optional.Get.type.ab0 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.facet.be4: %Destroy.type = facet_value %DiffValueType, (%custom_witness.1bfa2a.1) [concrete]
// CHECK:STDOUT:   %MaybeUnformed.175: type = class_type @MaybeUnformed, @MaybeUnformed(%Destroy.facet.be4) [concrete]
// CHECK:STDOUT:   %.a18: type = maybe_unformed_type %DiffValueType [concrete]
// CHECK:STDOUT:   %DefaultOptionalStorage.elem.4f3: type = unbound_element_type %DefaultOptionalStorage.21a, %MaybeUnformed.175 [concrete]
// CHECK:STDOUT:   %DefaultOptionalStorage.elem.311: type = unbound_element_type %DefaultOptionalStorage.21a, bool [concrete]
// CHECK:STDOUT:   %struct_type.value.has_value.64e: type = struct_type {.value: %MaybeUnformed.175, .has_value: bool} [concrete]
// CHECK:STDOUT:   %Optional.HasValue.specific_fn.126: <specific function> = specific_function %Optional.HasValue.7d8, @Optional.HasValue(%OptionalStorage.facet.9e6) [concrete]
// CHECK:STDOUT:   %Optional.Get.specific_fn.e6d: <specific function> = specific_function %Optional.Get.6e2, @Optional.Get(%OptionalStorage.facet.9e6) [concrete]
// CHECK:STDOUT:   %operator_PlusEqual__carbon_thunk.type: type = fn_type @operator_PlusEqual__carbon_thunk [concrete]
// CHECK:STDOUT:   %operator_PlusEqual__carbon_thunk: %operator_PlusEqual__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.98c: type = pattern_type %.a18 [concrete]
// CHECK:STDOUT:   %self.param_patt.47c: %pattern_type.98c = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.90b: %pattern_type.98c = wrapper_binding_pattern self, %self.param_patt.47c [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc44_33.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.2: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.3: type = fn_type @Destroy.WithSelf.Op.loc44_33.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.3: %Destroy.WithSelf.Op.type.ef016f.3 = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.4ea: type = pattern_type %MaybeUnformed.175 [concrete]
// CHECK:STDOUT:   %self.param_patt.8b0: %pattern_type.4ea = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.5e7: %pattern_type.4ea = wrapper_binding_pattern self, %self.param_patt.8b0 [concrete]
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
// CHECK:STDOUT:   %pattern_type.b1a: type = pattern_type %struct_type.value.has_value.64e [concrete]
// CHECK:STDOUT:   %self.param_patt.ef3: %pattern_type.b1a = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.5e2: %pattern_type.b1a = wrapper_binding_pattern self, %self.param_patt.ef3 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.5: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc44_33.4 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.5: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.5 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.6: type = fn_type @Destroy.WithSelf.Op.loc44_33.4 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.6: %Destroy.WithSelf.Op.type.ef016f.6 = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.0a8: type = pattern_type %DefaultOptionalStorage.21a [concrete]
// CHECK:STDOUT:   %self.param_patt.276: %pattern_type.0a8 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.7c0: %pattern_type.0a8 = wrapper_binding_pattern self, %self.param_patt.276 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.6: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc44_33.5 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.6: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.6 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.7: type = fn_type @Destroy.WithSelf.Op.loc44_33.5 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.7: %Destroy.WithSelf.Op.type.ef016f.7 = struct_value () [concrete]
// CHECK:STDOUT:   %self.param_patt.b6b: %pattern_type.054 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.4c2: %pattern_type.054 = wrapper_binding_pattern self, %self.param_patt.b6b [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.7: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc44_33.6 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.7: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.7 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.8: type = fn_type @Destroy.WithSelf.Op.loc44_33.6 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.8: %Destroy.WithSelf.Op.type.ef016f.8 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.8: type = fn_type @Destroy.WithSelf.SelfDestruct.loc44_33.6 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.8: %Destroy.WithSelf.SelfDestruct.type.fbceb5.8 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.662: <witness> = lookup_impl_witness %impl.elem1.fbc, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.e70: type = fn_type @Destroy.WithSelf.SelfDestruct.1, @Destroy.WithSelf(%impl.elem1.fbc) [symbolic]
// CHECK:STDOUT:   %.99c: type = fn_type_with_self_type %Destroy.WithSelf.SelfDestruct.type.e70, %impl.elem1.fbc [symbolic]
// CHECK:STDOUT:   %impl.elem2.ff7: %.99c = impl_witness_access %Destroy.lookup_impl_witness.662, element2 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.ffc: <specific function> = specific_impl_function %impl.elem2.ff7, @Destroy.WithSelf.SelfDestruct.1(%impl.elem1.fbc) [symbolic]
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
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.9: type = fn_type @Destroy.WithSelf.Op.loc51_16.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.9: %Destroy.WithSelf.Op.type.ef016f.9 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.9: type = fn_type @Destroy.WithSelf.SelfDestruct.loc51_16.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.9: %Destroy.WithSelf.SelfDestruct.type.fbceb5.9 = struct_value () [concrete]
// CHECK:STDOUT:   %Iterator.SubobjectDestroy.type.45a: type = fn_type @Iterator.SubobjectDestroy.1 [concrete]
// CHECK:STDOUT:   %Iterator.SubobjectDestroy.d6a: %Iterator.SubobjectDestroy.type.45a = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.db9baa.1: <witness> = custom_witness (%Destroy.WithSelf.Op.403171.9, %Iterator.SubobjectDestroy.d6a, %Destroy.WithSelf.SelfDestruct.db3fdb.9), @Destroy [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.41bbea.1: type = fn_type @Iterator.Op.1 [concrete]
// CHECK:STDOUT:   %Iterator.Op.1caa36.1: %Iterator.Op.type.41bbea.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.392: <witness> = custom_witness (%DiffValueType, %Iterator.Op.1caa36.1), @CppUnsafeDeref [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.41bbea.2: type = fn_type @Iterator.Op.2 [concrete]
// CHECK:STDOUT:   %Iterator.Op.1caa36.2: %Iterator.Op.type.41bbea.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.1c9: <witness> = custom_witness (%Iterator.Op.1caa36.2), @Inc [concrete]
// CHECK:STDOUT:   %Equal.type.ca8475.1: type = fn_type @Equal.1 [concrete]
// CHECK:STDOUT:   %Equal.285b15.1: %Equal.type.ca8475.1 = struct_value () [concrete]
// CHECK:STDOUT:   %NotEqual.type.53d9ee.1: type = fn_type @NotEqual.1 [concrete]
// CHECK:STDOUT:   %NotEqual.119263.1: %NotEqual.type.53d9ee.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b32: <witness> = custom_witness (%Equal.285b15.1, %NotEqual.119263.1), @EqWith, @EqWith(%Sentinel.b4c) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.10: type = fn_type @Destroy.WithSelf.Op.loc51_16.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.10: %Destroy.WithSelf.Op.type.ef016f.10 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.10: type = fn_type @Destroy.WithSelf.SelfDestruct.loc51_16.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.10: %Destroy.WithSelf.SelfDestruct.type.fbceb5.10 = struct_value () [concrete]
// CHECK:STDOUT:   %Sentinel.SubobjectDestroy.type.45a: type = fn_type @Sentinel.SubobjectDestroy.1 [concrete]
// CHECK:STDOUT:   %Sentinel.SubobjectDestroy.d6a: %Sentinel.SubobjectDestroy.type.45a = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.db9baa.2: <witness> = custom_witness (%Destroy.WithSelf.Op.403171.10, %Sentinel.SubobjectDestroy.d6a, %Destroy.WithSelf.SelfDestruct.db3fdb.10), @Destroy [concrete]
// CHECK:STDOUT:   %facet_value.4bb: %CppRange_where.type.d682d6.1 = facet_value %DiffMutableRange, (%custom_witness.aa8, %custom_witness.db9baa.1, %custom_witness.392, %custom_witness.1c9, %custom_witness.b32, %custom_witness.1bfa2a.1, %custom_witness.d0da43.1, %custom_witness.db9baa.2) [concrete]
// CHECK:STDOUT:   %tuple.type.b18: type = tuple_type (%Iterator.f25, %Sentinel.b4c) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.8: type = fn_type @Destroy.WithSelf.SubobjectDestroy.3 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.8: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.8 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.11: type = fn_type @Destroy.WithSelf.Op.3 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.11: %Destroy.WithSelf.Op.type.ef016f.11 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.11: type = fn_type @Destroy.WithSelf.SelfDestruct.3 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.11: %Destroy.WithSelf.SelfDestruct.type.fbceb5.11 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.f8f19d.8: <witness> = custom_witness (%Destroy.WithSelf.Op.403171.11, %Destroy.WithSelf.SubobjectDestroy.d01daf.8, %Destroy.WithSelf.SelfDestruct.db3fdb.11), @Destroy [concrete]
// CHECK:STDOUT:   %Destroy.facet.645: %Destroy.type = facet_value %tuple.type.b18, (%custom_witness.f8f19d.8) [concrete]
// CHECK:STDOUT:   %Iterate.impl_witness.606: <witness> = impl_witness imports.%Iterate.impl_witness_table.121, @T.as_type.as.Iterate.impl(%facet_value.4bb) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.type.899: type = fn_type @T.as_type.as.Iterate.impl.NewCursor, @T.as_type.as.Iterate.impl(%facet_value.4bb) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.a7b: %T.as_type.as.Iterate.impl.NewCursor.type.899 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.type.355: type = fn_type @T.as_type.as.Iterate.impl.Next, @T.as_type.as.Iterate.impl(%facet_value.4bb) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.8ce: %T.as_type.as.Iterate.impl.Next.type.355 = struct_value () [concrete]
// CHECK:STDOUT:   %Iterate.facet.758: %Iterate.type = facet_value %DiffMutableRange, (%Iterate.impl_witness.606) [concrete]
// CHECK:STDOUT:   %facet_value.2c1: %Iterate_where.type.b96 = facet_value %DiffMutableRange, (%Iterate.impl_witness.606) [concrete]
// CHECK:STDOUT:   %r.patt.5fe: %pattern_type.611 = ref_binding_pattern r [concrete]
// CHECK:STDOUT:   %r.param_patt.a4b: %pattern_type.611 = var_param_pattern %r.patt.5fe [concrete]
// CHECK:STDOUT:   %Iterator.a59: type = class_type @Iterator.2 [concrete]
// CHECK:STDOUT:   %Sentinel.b33: type = class_type @Sentinel.2 [concrete]
// CHECK:STDOUT:   %DiffConstRange.Begin.type: type = fn_type @DiffConstRange.Begin [concrete]
// CHECK:STDOUT:   %DiffConstRange.Begin: %DiffConstRange.Begin.type = struct_value () [concrete]
// CHECK:STDOUT:   %DiffConstRange.End.type: type = fn_type @DiffConstRange.End [concrete]
// CHECK:STDOUT:   %DiffConstRange.End: %DiffConstRange.End.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.7b4: <witness> = custom_witness (%Iterator.a59, %Sentinel.b33, %DiffConstRange.Begin, %DiffConstRange.End), @CppRangeForIterate [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.13: type = fn_type @Destroy.WithSelf.Op.loc52_16.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.13: %Destroy.WithSelf.Op.type.ef016f.13 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.13: type = fn_type @Destroy.WithSelf.SelfDestruct.loc52_16.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.13: %Destroy.WithSelf.SelfDestruct.type.fbceb5.13 = struct_value () [concrete]
// CHECK:STDOUT:   %Iterator.SubobjectDestroy.type.e2b: type = fn_type @Iterator.SubobjectDestroy.2 [concrete]
// CHECK:STDOUT:   %Iterator.SubobjectDestroy.21a: %Iterator.SubobjectDestroy.type.e2b = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.48b6e9.1: <witness> = custom_witness (%Destroy.WithSelf.Op.403171.13, %Iterator.SubobjectDestroy.21a, %Destroy.WithSelf.SelfDestruct.db3fdb.13), @Destroy [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.b7ae80.1: type = fn_type @Iterator.Op.3 [concrete]
// CHECK:STDOUT:   %Iterator.Op.690fee.1: %Iterator.Op.type.b7ae80.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.0c0: <witness> = custom_witness (%DiffValueType, %Iterator.Op.690fee.1), @CppUnsafeDeref [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.b7ae80.2: type = fn_type @Iterator.Op.4 [concrete]
// CHECK:STDOUT:   %Iterator.Op.690fee.2: %Iterator.Op.type.b7ae80.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.ffb: <witness> = custom_witness (%Iterator.Op.690fee.2), @Inc [concrete]
// CHECK:STDOUT:   %Equal.type.ca8475.2: type = fn_type @Equal.2 [concrete]
// CHECK:STDOUT:   %Equal.285b15.2: %Equal.type.ca8475.2 = struct_value () [concrete]
// CHECK:STDOUT:   %NotEqual.type.53d9ee.2: type = fn_type @NotEqual.2 [concrete]
// CHECK:STDOUT:   %NotEqual.119263.2: %NotEqual.type.53d9ee.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.fc2: <witness> = custom_witness (%Equal.285b15.2, %NotEqual.119263.2), @EqWith, @EqWith(%Sentinel.b33) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.14: type = fn_type @Destroy.WithSelf.Op.loc52_16.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.14: %Destroy.WithSelf.Op.type.ef016f.14 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.14: type = fn_type @Destroy.WithSelf.SelfDestruct.loc52_16.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.14: %Destroy.WithSelf.SelfDestruct.type.fbceb5.14 = struct_value () [concrete]
// CHECK:STDOUT:   %Sentinel.SubobjectDestroy.type.e2b: type = fn_type @Sentinel.SubobjectDestroy.2 [concrete]
// CHECK:STDOUT:   %Sentinel.SubobjectDestroy.21a: %Sentinel.SubobjectDestroy.type.e2b = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.48b6e9.2: <witness> = custom_witness (%Destroy.WithSelf.Op.403171.14, %Sentinel.SubobjectDestroy.21a, %Destroy.WithSelf.SelfDestruct.db3fdb.14), @Destroy [concrete]
// CHECK:STDOUT:   %facet_value.683: %CppRange_where.type.d682d6.1 = facet_value %DiffConstRange, (%custom_witness.7b4, %custom_witness.48b6e9.1, %custom_witness.0c0, %custom_witness.ffb, %custom_witness.fc2, %custom_witness.1bfa2a.1, %custom_witness.d0da43.1, %custom_witness.48b6e9.2) [concrete]
// CHECK:STDOUT:   %tuple.type.df5: type = tuple_type (%Iterator.a59, %Sentinel.b33) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.9: type = fn_type @Destroy.WithSelf.SubobjectDestroy.4 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.9: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.9 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.15: type = fn_type @Destroy.WithSelf.Op.4 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.15: %Destroy.WithSelf.Op.type.ef016f.15 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.15: type = fn_type @Destroy.WithSelf.SelfDestruct.4 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.15: %Destroy.WithSelf.SelfDestruct.type.fbceb5.15 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.f8f19d.9: <witness> = custom_witness (%Destroy.WithSelf.Op.403171.15, %Destroy.WithSelf.SubobjectDestroy.d01daf.9, %Destroy.WithSelf.SelfDestruct.db3fdb.15), @Destroy [concrete]
// CHECK:STDOUT:   %Destroy.facet.206: %Destroy.type = facet_value %tuple.type.df5, (%custom_witness.f8f19d.9) [concrete]
// CHECK:STDOUT:   %Iterate.impl_witness.9b4: <witness> = impl_witness imports.%Iterate.impl_witness_table.121, @T.as_type.as.Iterate.impl(%facet_value.683) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.type.4f6: type = fn_type @T.as_type.as.Iterate.impl.NewCursor, @T.as_type.as.Iterate.impl(%facet_value.683) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.3cd: %T.as_type.as.Iterate.impl.NewCursor.type.4f6 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.type.885: type = fn_type @T.as_type.as.Iterate.impl.Next, @T.as_type.as.Iterate.impl(%facet_value.683) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.d18: %T.as_type.as.Iterate.impl.Next.type.885 = struct_value () [concrete]
// CHECK:STDOUT:   %Iterate.facet.930: %Iterate.type = facet_value %DiffConstRange, (%Iterate.impl_witness.9b4) [concrete]
// CHECK:STDOUT:   %facet_value.1fc: %Iterate_where.type.b96 = facet_value %DiffConstRange, (%Iterate.impl_witness.9b4) [concrete]
// CHECK:STDOUT:   %r.patt.9c0: %pattern_type.7dd = ref_binding_pattern r [concrete]
// CHECK:STDOUT:   %r.param_patt.992: %pattern_type.7dd = var_param_pattern %r.patt.9c0 [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.49d: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.758) [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.1f8: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.758) [concrete]
// CHECK:STDOUT:   %.c05: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.49d, %Iterate.facet.758 [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.specific_fn.1d2: <specific function> = specific_function %T.as_type.as.Iterate.impl.NewCursor.a7b, @T.as_type.as.Iterate.impl.NewCursor(%facet_value.4bb) [concrete]
// CHECK:STDOUT:   %.22d: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.1f8, %Iterate.facet.758 [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.specific_fn.288: <specific function> = specific_function %T.as_type.as.Iterate.impl.Next.8ce, @T.as_type.as.Iterate.impl.Next(%facet_value.4bb) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.baf: type = fn_type @Destroy.WithSelf.SelfDestruct.1, @Destroy.WithSelf(%Destroy.facet.645) [concrete]
// CHECK:STDOUT:   %.085: type = fn_type_with_self_type %Destroy.WithSelf.SelfDestruct.type.baf, %Destroy.facet.645 [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.4f6: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.930) [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.95d: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.930) [concrete]
// CHECK:STDOUT:   %.b0d: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.4f6, %Iterate.facet.930 [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.specific_fn.104: <specific function> = specific_function %T.as_type.as.Iterate.impl.NewCursor.3cd, @T.as_type.as.Iterate.impl.NewCursor(%facet_value.683) [concrete]
// CHECK:STDOUT:   %.d85: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.95d, %Iterate.facet.930 [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.specific_fn.8ee: <specific function> = specific_function %T.as_type.as.Iterate.impl.Next.d18, @T.as_type.as.Iterate.impl.Next(%facet_value.683) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.ee6: type = fn_type @Destroy.WithSelf.SelfDestruct.1, @Destroy.WithSelf(%Destroy.facet.206) [concrete]
// CHECK:STDOUT:   %.c1e: type = fn_type_with_self_type %Destroy.WithSelf.SelfDestruct.type.ee6, %Destroy.facet.206 [concrete]
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
// CHECK:STDOUT:   %Core.import_ref.4b2: @Optional.%Optional.HasValue.type (%Optional.HasValue.type.49e) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.HasValue (constants.%Optional.HasValue.c20)]
// CHECK:STDOUT:   %Core.import_ref.52c: @Optional.%Optional.Get.type (%Optional.Get.type.c64) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.Get (constants.%Optional.Get.bcf)]
// CHECK:STDOUT:   %Core.import_ref.ace: type = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%DefaultOptionalStorage (constants.%DefaultOptionalStorage.faa)]
// CHECK:STDOUT:   %Core.import_ref.ef4: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.None.type (%T.as_type.as.OptionalStorage.impl.None.type.925) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.None (constants.%T.as_type.as.OptionalStorage.impl.None.f42)]
// CHECK:STDOUT:   %Core.import_ref.a20: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Some.type (%T.as_type.as.OptionalStorage.impl.Some.type.760) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Some (constants.%T.as_type.as.OptionalStorage.impl.Some.268)]
// CHECK:STDOUT:   %Core.import_ref.af7: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Has.type (%T.as_type.as.OptionalStorage.impl.Has.type.d28) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Has (constants.%T.as_type.as.OptionalStorage.impl.Has.f29)]
// CHECK:STDOUT:   %Core.import_ref.dd4: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Get.type (%T.as_type.as.OptionalStorage.impl.Get.type.2f2) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Get (constants.%T.as_type.as.OptionalStorage.impl.Get.884)]
// CHECK:STDOUT:   %Core.import_ref.554 = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], unloaded
// CHECK:STDOUT:   %OptionalStorage.impl_witness_table.460 = impl_witness_table (%Core.import_ref.ace, %Core.import_ref.ef4, %Core.import_ref.a20, %Core.import_ref.af7, %Core.import_ref.dd4, %Core.import_ref.554), @T.as_type.as.OptionalStorage.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.1d7: @DefaultOptionalStorage.%DefaultOptionalStorage.elem.1 (%DefaultOptionalStorage.elem.9db) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [concrete = %field_decl.5f0]
// CHECK:STDOUT:   %Core.import_ref.929: @DefaultOptionalStorage.%DefaultOptionalStorage.elem.2 (%DefaultOptionalStorage.elem.06a) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [concrete = %field_decl.6a8]
// CHECK:STDOUT:   %Core.import_ref.eef: %facet_type.f48 = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.Iterate.impl.%facet_value (constants.%facet_value.f93)]
// CHECK:STDOUT:   %Core.import_ref.fcc: %Destroy.type = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.Iterate.impl.%Destroy.facet (constants.%Destroy.facet.9cd)]
// CHECK:STDOUT:   %Core.import_ref.946: @T.as_type.as.Iterate.impl.%T.as_type.as.Iterate.impl.NewCursor.type (%T.as_type.as.Iterate.impl.NewCursor.type.8fb) = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.Iterate.impl.%T.as_type.as.Iterate.impl.NewCursor (constants.%T.as_type.as.Iterate.impl.NewCursor.8a1)]
// CHECK:STDOUT:   %Core.import_ref.cfb: @T.as_type.as.Iterate.impl.%T.as_type.as.Iterate.impl.Next.type (%T.as_type.as.Iterate.impl.Next.type.d4a) = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.Iterate.impl.%T.as_type.as.Iterate.impl.Next (constants.%T.as_type.as.Iterate.impl.Next.5a8)]
// CHECK:STDOUT:   %Iterate.impl_witness_table.121 = impl_witness_table (%Core.import_ref.eef, %Core.import_ref.fcc, %Core.import_ref.946, %Core.import_ref.cfb), @T.as_type.as.Iterate.impl [concrete]
// CHECK:STDOUT:   %DiffValueType.decl: type = class_decl @DiffValueType [concrete = constants.%DiffValueType] {} {}
// CHECK:STDOUT:   %operator_PlusEqual__carbon_thunk.decl: %operator_PlusEqual__carbon_thunk.type = fn_decl @operator_PlusEqual__carbon_thunk [concrete = constants.%operator_PlusEqual__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.b0d = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %field_decl.6a8: @DefaultOptionalStorage.%DefaultOptionalStorage.elem.2 (%DefaultOptionalStorage.elem.06a) = field_decl has_value, element1, %Core.import_ref.b0d in [concrete] {}
// CHECK:STDOUT:   %Core.import_ref.87f = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %field_decl.5f0: @DefaultOptionalStorage.%DefaultOptionalStorage.elem.1 (%DefaultOptionalStorage.elem.9db) = field_decl value, element0, %Core.import_ref.87f in [concrete] {}
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
// CHECK:STDOUT:     %self.param_patt: %pattern_type.4ea = ref_param_pattern [concrete = constants.%self.param_patt.8b0]
// CHECK:STDOUT:     %self.patt: %pattern_type.4ea = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.5e7]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %MaybeUnformed.175 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %MaybeUnformed.175 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.3: %Destroy.WithSelf.Op.type.ef016f.4 = fn_decl @Destroy.WithSelf.Op.loc44_33.2 [concrete = constants.%Destroy.WithSelf.Op.403171.4] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.4ea = ref_param_pattern [concrete = constants.%self.param_patt.8b0]
// CHECK:STDOUT:     %self.patt: %pattern_type.4ea = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.5e7]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %MaybeUnformed.175 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %MaybeUnformed.175 = wrapper_binding self, %self.param
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
// CHECK:STDOUT:     %self.param_patt: %pattern_type.b1a = ref_param_pattern [concrete = constants.%self.param_patt.ef3]
// CHECK:STDOUT:     %self.patt: %pattern_type.b1a = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.5e2]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %struct_type.value.has_value.64e = ref_param call_param0
// CHECK:STDOUT:     %self: ref %struct_type.value.has_value.64e = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.5: %Destroy.WithSelf.Op.type.ef016f.6 = fn_decl @Destroy.WithSelf.Op.loc44_33.4 [concrete = constants.%Destroy.WithSelf.Op.403171.6] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.b1a = ref_param_pattern [concrete = constants.%self.param_patt.ef3]
// CHECK:STDOUT:     %self.patt: %pattern_type.b1a = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.5e2]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %struct_type.value.has_value.64e = ref_param call_param0
// CHECK:STDOUT:     %self: ref %struct_type.value.has_value.64e = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.5: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.6 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc44_33.5 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.6] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.0a8 = ref_param_pattern [concrete = constants.%self.param_patt.276]
// CHECK:STDOUT:     %self.patt: %pattern_type.0a8 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.7c0]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %DefaultOptionalStorage.21a = ref_param call_param0
// CHECK:STDOUT:     %self: ref %DefaultOptionalStorage.21a = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.6: %Destroy.WithSelf.Op.type.ef016f.7 = fn_decl @Destroy.WithSelf.Op.loc44_33.5 [concrete = constants.%Destroy.WithSelf.Op.403171.7] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.0a8 = ref_param_pattern [concrete = constants.%self.param_patt.276]
// CHECK:STDOUT:     %self.patt: %pattern_type.0a8 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.7c0]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %DefaultOptionalStorage.21a = ref_param call_param0
// CHECK:STDOUT:     %self: ref %DefaultOptionalStorage.21a = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.6: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.7 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc44_33.6 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.7] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.054 = ref_param_pattern [concrete = constants.%self.param_patt.b6b]
// CHECK:STDOUT:     %self.patt: %pattern_type.054 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.4c2]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %Optional.6ee = ref_param call_param0
// CHECK:STDOUT:     %self: ref %Optional.6ee = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.7: %Destroy.WithSelf.Op.type.ef016f.8 = fn_decl @Destroy.WithSelf.Op.loc44_33.6 [concrete = constants.%Destroy.WithSelf.Op.403171.8] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.054 = ref_param_pattern [concrete = constants.%self.param_patt.b6b]
// CHECK:STDOUT:     %self.patt: %pattern_type.054 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.4c2]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %Optional.6ee = ref_param call_param0
// CHECK:STDOUT:     %self: ref %Optional.6ee = wrapper_binding self, %self.param
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
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.loc44_33.5) [symbolic = %Iterate.WithSelf.NewCursor.type (constants.%Iterate.WithSelf.NewCursor.type.b5e)]
// CHECK:STDOUT:   %.loc44_33.20: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type, %Iterate.facet.loc44_33.5 [symbolic = %.loc44_33.20 (constants.%.c17)]
// CHECK:STDOUT:   %impl.elem2.loc44_33.3: @TestIterate.%.loc44_33.20 (%.c17) = impl_witness_access %Iterate.lookup_impl_witness, element2 [symbolic = %impl.elem2.loc44_33.3 (constants.%impl.elem2.3ea)]
// CHECK:STDOUT:   %specific_impl_fn.loc44_33.4: <specific function> = specific_impl_function %impl.elem2.loc44_33.3, @Iterate.WithSelf.NewCursor(%Iterate.facet.loc44_33.5) [symbolic = %specific_impl_fn.loc44_33.4 (constants.%specific_impl_fn.b2c)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.loc44_33.5) [symbolic = %Iterate.WithSelf.Next.type (constants.%Iterate.WithSelf.Next.type.931)]
// CHECK:STDOUT:   %.loc44_33.21: type = fn_type_with_self_type %Iterate.WithSelf.Next.type, %Iterate.facet.loc44_33.5 [symbolic = %.loc44_33.21 (constants.%.c9b)]
// CHECK:STDOUT:   %impl.elem3.loc44_33.2: @TestIterate.%.loc44_33.21 (%.c9b) = impl_witness_access %Iterate.lookup_impl_witness, element3 [symbolic = %impl.elem3.loc44_33.2 (constants.%impl.elem3.0ff)]
// CHECK:STDOUT:   %specific_impl_fn.loc44_33.5: <specific function> = specific_impl_function %impl.elem3.loc44_33.2, @Iterate.WithSelf.Next(%Iterate.facet.loc44_33.5) [symbolic = %specific_impl_fn.loc44_33.5 (constants.%specific_impl_fn.20b)]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type: type = fn_type @Destroy.WithSelf.SelfDestruct.1, @Destroy.WithSelf(%impl.elem1) [symbolic = %Destroy.WithSelf.SelfDestruct.type (constants.%Destroy.WithSelf.SelfDestruct.type.e70)]
// CHECK:STDOUT:   %.loc44_33.22: type = fn_type_with_self_type %Destroy.WithSelf.SelfDestruct.type, %impl.elem1 [symbolic = %.loc44_33.22 (constants.%.99c)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %impl.elem1, @Destroy [symbolic = %Destroy.lookup_impl_witness (constants.%Destroy.lookup_impl_witness.662)]
// CHECK:STDOUT:   %impl.elem2.loc44_33.4: @TestIterate.%.loc44_33.22 (%.99c) = impl_witness_access %Destroy.lookup_impl_witness, element2 [symbolic = %impl.elem2.loc44_33.4 (constants.%impl.elem2.ff7)]
// CHECK:STDOUT:   %specific_impl_fn.loc44_33.6: <specific function> = specific_impl_function %impl.elem2.loc44_33.4, @Destroy.WithSelf.SelfDestruct.1(%impl.elem1) [symbolic = %specific_impl_fn.loc44_33.6 (constants.%specific_impl_fn.ffc)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%r.param: ref @TestIterate.%R.as_type.loc40_79.1 (%R.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %i.patt: %pattern_type.149 = value_binding_pattern i [concrete = constants.%i.patt]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %r.ref: ref @TestIterate.%R.as_type.loc40_79.1 (%R.as_type) = name_ref r, %r
// CHECK:STDOUT:     %impl.elem2.loc44_33.1: @TestIterate.%.loc44_33.20 (%.c17) = impl_witness_access constants.%Iterate.lookup_impl_witness.d00, element2 [symbolic = %impl.elem2.loc44_33.3 (constants.%impl.elem2.3ea)]
// CHECK:STDOUT:     %bound_method.loc44_33.1: <bound method> = bound_method %r.ref, %impl.elem2.loc44_33.1
// CHECK:STDOUT:     %Iterate.facet.loc44_33.1: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.d00) [symbolic = %Iterate.facet.loc44_33.5 (constants.%Iterate.facet.f03)]
// CHECK:STDOUT:     %.loc44_33.1: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc44_33.1 [symbolic = %Iterate.facet.loc44_33.5 (constants.%Iterate.facet.f03)]
// CHECK:STDOUT:     %Iterate.facet.loc44_33.2: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.d00) [symbolic = %Iterate.facet.loc44_33.5 (constants.%Iterate.facet.f03)]
// CHECK:STDOUT:     %.loc44_33.2: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc44_33.2 [symbolic = %Iterate.facet.loc44_33.5 (constants.%Iterate.facet.f03)]
// CHECK:STDOUT:     %specific_impl_fn.loc44_33.1: <specific function> = specific_impl_function %impl.elem2.loc44_33.1, @Iterate.WithSelf.NewCursor(constants.%Iterate.facet.f03) [symbolic = %specific_impl_fn.loc44_33.4 (constants.%specific_impl_fn.b2c)]
// CHECK:STDOUT:     %bound_method.loc44_33.2: <bound method> = bound_method %r.ref, %specific_impl_fn.loc44_33.1
// CHECK:STDOUT:     %var: ref @TestIterate.%as_type (%as_type.389) = var_storage invalid
// CHECK:STDOUT:     %.loc44_32.1: @TestIterate.%R.as_type.loc40_79.1 (%R.as_type) = acquire_value %r.ref
// CHECK:STDOUT:     %Iterate.WithSelf.NewCursor.call: init @TestIterate.%as_type (%as_type.389) to %var = call %bound_method.loc44_33.2(%.loc44_32.1)
// CHECK:STDOUT:     assign %var, %Iterate.WithSelf.NewCursor.call
// CHECK:STDOUT:     br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.next:
// CHECK:STDOUT:     %impl.elem3.loc44_33.1: @TestIterate.%.loc44_33.21 (%.c9b) = impl_witness_access constants.%Iterate.lookup_impl_witness.d00, element3 [symbolic = %impl.elem3.loc44_33.2 (constants.%impl.elem3.0ff)]
// CHECK:STDOUT:     %bound_method.loc44_33.3: <bound method> = bound_method %r.ref, %impl.elem3.loc44_33.1
// CHECK:STDOUT:     %Iterate.facet.loc44_33.3: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.d00) [symbolic = %Iterate.facet.loc44_33.5 (constants.%Iterate.facet.f03)]
// CHECK:STDOUT:     %.loc44_33.3: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc44_33.3 [symbolic = %Iterate.facet.loc44_33.5 (constants.%Iterate.facet.f03)]
// CHECK:STDOUT:     %Iterate.facet.loc44_33.4: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.d00) [symbolic = %Iterate.facet.loc44_33.5 (constants.%Iterate.facet.f03)]
// CHECK:STDOUT:     %.loc44_33.4: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc44_33.4 [symbolic = %Iterate.facet.loc44_33.5 (constants.%Iterate.facet.f03)]
// CHECK:STDOUT:     %.loc44_33.5: %Destroy.type = converted constants.%as_type.389, constants.%impl.elem1.fbc [symbolic = %impl.elem1 (constants.%impl.elem1.fbc)]
// CHECK:STDOUT:     %.loc44_33.6: %Destroy.type = converted constants.%as_type.389, constants.%impl.elem1.fbc [symbolic = %impl.elem1 (constants.%impl.elem1.fbc)]
// CHECK:STDOUT:     %specific_impl_fn.loc44_33.2: <specific function> = specific_impl_function %impl.elem3.loc44_33.1, @Iterate.WithSelf.Next(constants.%Iterate.facet.f03) [symbolic = %specific_impl_fn.loc44_33.5 (constants.%specific_impl_fn.20b)]
// CHECK:STDOUT:     %bound_method.loc44_33.4: <bound method> = bound_method %r.ref, %specific_impl_fn.loc44_33.2
// CHECK:STDOUT:     %.loc44_33.7: ref %Optional.6ee = temporary_storage
// CHECK:STDOUT:     %.loc44_32.2: @TestIterate.%R.as_type.loc40_79.1 (%R.as_type) = acquire_value %r.ref
// CHECK:STDOUT:     %Iterate.WithSelf.Next.call: init %Optional.6ee to %.loc44_33.7 = call %bound_method.loc44_33.4(%.loc44_32.2, %var)
// CHECK:STDOUT:     %.loc44_33.8: ref %Optional.6ee = temporary %.loc44_33.7, %Iterate.WithSelf.Next.call
// CHECK:STDOUT:     %.loc44_33.9: %Optional.HasValue.type.868 = specific_constant imports.%Core.import_ref.4b2, @Optional(constants.%OptionalStorage.facet.9e6) [concrete = constants.%Optional.HasValue.7d8]
// CHECK:STDOUT:     %HasValue.ref: %Optional.HasValue.type.868 = name_ref HasValue, %.loc44_33.9 [concrete = constants.%Optional.HasValue.7d8]
// CHECK:STDOUT:     %Optional.HasValue.bound: <bound method> = bound_method %.loc44_33.8, %HasValue.ref
// CHECK:STDOUT:     %Optional.HasValue.specific_fn: <specific function> = specific_function %HasValue.ref, @Optional.HasValue(constants.%OptionalStorage.facet.9e6) [concrete = constants.%Optional.HasValue.specific_fn.126]
// CHECK:STDOUT:     %bound_method.loc44_33.5: <bound method> = bound_method %.loc44_33.8, %Optional.HasValue.specific_fn
// CHECK:STDOUT:     %.loc44_33.10: %Optional.6ee = acquire_value %.loc44_33.8
// CHECK:STDOUT:     %Optional.HasValue.call: init bool = call %bound_method.loc44_33.5(%.loc44_33.10)
// CHECK:STDOUT:     %.loc44_33.11: bool = value_of_initializer %Optional.HasValue.call
// CHECK:STDOUT:     %.loc44_33.12: bool = converted %Optional.HasValue.call, %.loc44_33.11
// CHECK:STDOUT:     if %.loc44_33.12 br !for.body else br !for.done
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.body:
// CHECK:STDOUT:     %.loc44_33.13: %Optional.Get.type.ab0 = specific_constant imports.%Core.import_ref.52c, @Optional(constants.%OptionalStorage.facet.9e6) [concrete = constants.%Optional.Get.6e2]
// CHECK:STDOUT:     %Get.ref: %Optional.Get.type.ab0 = name_ref Get, %.loc44_33.13 [concrete = constants.%Optional.Get.6e2]
// CHECK:STDOUT:     %Optional.Get.bound: <bound method> = bound_method %.loc44_33.8, %Get.ref
// CHECK:STDOUT:     %Optional.Get.specific_fn: <specific function> = specific_function %Get.ref, @Optional.Get(constants.%OptionalStorage.facet.9e6) [concrete = constants.%Optional.Get.specific_fn.e6d]
// CHECK:STDOUT:     %bound_method.loc44_33.6: <bound method> = bound_method %.loc44_33.8, %Optional.Get.specific_fn
// CHECK:STDOUT:     %.loc44_33.14: ref %DiffValueType = temporary_storage
// CHECK:STDOUT:     %.loc44_33.15: %Optional.6ee = acquire_value %.loc44_33.8
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
// CHECK:STDOUT:     %impl.elem2.loc44_33.2: @TestIterate.%.loc44_33.22 (%.99c) = impl_witness_access constants.%Destroy.lookup_impl_witness.662, element2 [symbolic = %impl.elem2.loc44_33.4 (constants.%impl.elem2.ff7)]
// CHECK:STDOUT:     %bound_method.loc44_33.7: <bound method> = bound_method %var, %impl.elem2.loc44_33.2
// CHECK:STDOUT:     %.loc44_33.18: %Destroy.type = converted constants.%as_type.389, constants.%impl.elem1.fbc [symbolic = %impl.elem1 (constants.%impl.elem1.fbc)]
// CHECK:STDOUT:     %.loc44_33.19: %Destroy.type = converted constants.%as_type.389, constants.%impl.elem1.fbc [symbolic = %impl.elem1 (constants.%impl.elem1.fbc)]
// CHECK:STDOUT:     %specific_impl_fn.loc44_33.3: <specific function> = specific_impl_function %impl.elem2.loc44_33.2, @Destroy.WithSelf.SelfDestruct.1(constants.%impl.elem1.fbc) [symbolic = %specific_impl_fn.loc44_33.6 (constants.%specific_impl_fn.ffc)]
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
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc44_33.2(%self.param: ref %MaybeUnformed.175) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc44_33.2(%self.param: ref %MaybeUnformed.175) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc44_33.2(%self.param: ref %MaybeUnformed.175) {
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
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc44_33.4(%self.param: ref %struct_type.value.has_value.64e) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc44_33.1: ref bool = struct_access %self.param, element1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc44_33.1: <bound method> = bound_method %.loc44_33.1, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.5
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc44_33.1: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc44_33.1(%.loc44_33.1)
// CHECK:STDOUT:   %.loc44_33.2: ref %MaybeUnformed.175 = struct_access %self.param, element0
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc44_33.2: <bound method> = bound_method %.loc44_33.2, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.4
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc44_33.2: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc44_33.2(%.loc44_33.2)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc44_33.4(%self.param: ref %struct_type.value.has_value.64e) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc44_33.4(%self.param: ref %struct_type.value.has_value.64e) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.5(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.4(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc44_33.5(%self.param: ref %DefaultOptionalStorage.21a) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %has_value.ref: %DefaultOptionalStorage.elem.311 = name_ref has_value, imports.%Core.import_ref.929 [concrete = imports.%field_decl.6a8]
// CHECK:STDOUT:   %.loc44_33.1: ref bool = class_element_access %self.param, element1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc44_33.1: <bound method> = bound_method %.loc44_33.1, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.5
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc44_33.1: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc44_33.1(%.loc44_33.1)
// CHECK:STDOUT:   %value.ref: %DefaultOptionalStorage.elem.4f3 = name_ref value, imports.%Core.import_ref.1d7 [concrete = imports.%field_decl.5f0]
// CHECK:STDOUT:   %.loc44_33.2: ref %MaybeUnformed.175 = class_element_access %self.param, element0
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc44_33.2: <bound method> = bound_method %.loc44_33.2, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.4
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc44_33.2: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc44_33.2(%.loc44_33.2)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc44_33.5(%self.param: ref %DefaultOptionalStorage.21a) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc44_33.5(%self.param: ref %DefaultOptionalStorage.21a) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.6(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.5(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc44_33.6(%self.param: ref %Optional.6ee) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc44_33.6(%self.param: ref %Optional.6ee) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc44_33.6(%self.param: ref %Optional.6ee) {
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
// CHECK:STDOUT:   %impl.elem2.loc59: %.c05 = impl_witness_access constants.%Iterate.impl_witness.606, element2 [concrete = constants.%T.as_type.as.Iterate.impl.NewCursor.a7b]
// CHECK:STDOUT:   %bound_method.loc59_33.1: <bound method> = bound_method %m.ref, %impl.elem2.loc59
// CHECK:STDOUT:   %facet_value.loc59_33.1: %CppRange_where.type.d682d6.1 = facet_value constants.%DiffMutableRange, (constants.%custom_witness.aa8, constants.%custom_witness.db9baa.1, constants.%custom_witness.392, constants.%custom_witness.1c9, constants.%custom_witness.b32, constants.%custom_witness.1bfa2a.1, constants.%custom_witness.d0da43.1, constants.%custom_witness.db9baa.2) [concrete = constants.%facet_value.4bb]
// CHECK:STDOUT:   %.loc59_33.1: %CppRange_where.type.d682d6.1 = converted constants.%DiffMutableRange, %facet_value.loc59_33.1 [concrete = constants.%facet_value.4bb]
// CHECK:STDOUT:   %facet_value.loc59_33.2: %CppRange_where.type.d682d6.1 = facet_value constants.%DiffMutableRange, (constants.%custom_witness.aa8, constants.%custom_witness.db9baa.1, constants.%custom_witness.392, constants.%custom_witness.1c9, constants.%custom_witness.b32, constants.%custom_witness.1bfa2a.1, constants.%custom_witness.d0da43.1, constants.%custom_witness.db9baa.2) [concrete = constants.%facet_value.4bb]
// CHECK:STDOUT:   %.loc59_33.2: %CppRange_where.type.d682d6.1 = converted constants.%DiffMutableRange, %facet_value.loc59_33.2 [concrete = constants.%facet_value.4bb]
// CHECK:STDOUT:   %specific_fn.loc59_33.1: <specific function> = specific_function %impl.elem2.loc59, @T.as_type.as.Iterate.impl.NewCursor(constants.%facet_value.4bb) [concrete = constants.%T.as_type.as.Iterate.impl.NewCursor.specific_fn.1d2]
// CHECK:STDOUT:   %bound_method.loc59_33.2: <bound method> = bound_method %m.ref, %specific_fn.loc59_33.1
// CHECK:STDOUT:   %var.loc59: ref %tuple.type.b18 = var_storage invalid
// CHECK:STDOUT:   %.loc59_32.1: %DiffMutableRange = acquire_value %m.ref
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.call.loc59: init %tuple.type.b18 to %var.loc59 = call %bound_method.loc59_33.2(%.loc59_32.1)
// CHECK:STDOUT:   assign %var.loc59, %T.as_type.as.Iterate.impl.NewCursor.call.loc59
// CHECK:STDOUT:   br !for.next.loc59
// CHECK:STDOUT:
// CHECK:STDOUT: !for.next.loc59:
// CHECK:STDOUT:   %impl.elem3.loc59: %.22d = impl_witness_access constants.%Iterate.impl_witness.606, element3 [concrete = constants.%T.as_type.as.Iterate.impl.Next.8ce]
// CHECK:STDOUT:   %bound_method.loc59_33.3: <bound method> = bound_method %m.ref, %impl.elem3.loc59
// CHECK:STDOUT:   %facet_value.loc59_33.3: %CppRange_where.type.d682d6.1 = facet_value constants.%DiffMutableRange, (constants.%custom_witness.aa8, constants.%custom_witness.db9baa.1, constants.%custom_witness.392, constants.%custom_witness.1c9, constants.%custom_witness.b32, constants.%custom_witness.1bfa2a.1, constants.%custom_witness.d0da43.1, constants.%custom_witness.db9baa.2) [concrete = constants.%facet_value.4bb]
// CHECK:STDOUT:   %.loc59_33.3: %CppRange_where.type.d682d6.1 = converted constants.%DiffMutableRange, %facet_value.loc59_33.3 [concrete = constants.%facet_value.4bb]
// CHECK:STDOUT:   %facet_value.loc59_33.4: %CppRange_where.type.d682d6.1 = facet_value constants.%DiffMutableRange, (constants.%custom_witness.aa8, constants.%custom_witness.db9baa.1, constants.%custom_witness.392, constants.%custom_witness.1c9, constants.%custom_witness.b32, constants.%custom_witness.1bfa2a.1, constants.%custom_witness.d0da43.1, constants.%custom_witness.db9baa.2) [concrete = constants.%facet_value.4bb]
// CHECK:STDOUT:   %.loc59_33.4: %CppRange_where.type.d682d6.1 = converted constants.%DiffMutableRange, %facet_value.loc59_33.4 [concrete = constants.%facet_value.4bb]
// CHECK:STDOUT:   %specific_fn.loc59_33.2: <specific function> = specific_function %impl.elem3.loc59, @T.as_type.as.Iterate.impl.Next(constants.%facet_value.4bb) [concrete = constants.%T.as_type.as.Iterate.impl.Next.specific_fn.288]
// CHECK:STDOUT:   %bound_method.loc59_33.4: <bound method> = bound_method %m.ref, %specific_fn.loc59_33.2
// CHECK:STDOUT:   %.loc59_33.5: ref %Optional.6ee = temporary_storage
// CHECK:STDOUT:   %.loc59_32.2: %DiffMutableRange = acquire_value %m.ref
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.call.loc59: init %Optional.6ee to %.loc59_33.5 = call %bound_method.loc59_33.4(%.loc59_32.2, %var.loc59)
// CHECK:STDOUT:   %.loc59_33.6: ref %Optional.6ee = temporary %.loc59_33.5, %T.as_type.as.Iterate.impl.Next.call.loc59
// CHECK:STDOUT:   %.loc59_33.7: %Optional.HasValue.type.868 = specific_constant imports.%Core.import_ref.4b2, @Optional(constants.%OptionalStorage.facet.9e6) [concrete = constants.%Optional.HasValue.7d8]
// CHECK:STDOUT:   %HasValue.ref.loc59: %Optional.HasValue.type.868 = name_ref HasValue, %.loc59_33.7 [concrete = constants.%Optional.HasValue.7d8]
// CHECK:STDOUT:   %Optional.HasValue.bound.loc59: <bound method> = bound_method %.loc59_33.6, %HasValue.ref.loc59
// CHECK:STDOUT:   %Optional.HasValue.specific_fn.loc59: <specific function> = specific_function %HasValue.ref.loc59, @Optional.HasValue(constants.%OptionalStorage.facet.9e6) [concrete = constants.%Optional.HasValue.specific_fn.126]
// CHECK:STDOUT:   %bound_method.loc59_33.5: <bound method> = bound_method %.loc59_33.6, %Optional.HasValue.specific_fn.loc59
// CHECK:STDOUT:   %.loc59_33.8: %Optional.6ee = acquire_value %.loc59_33.6
// CHECK:STDOUT:   %Optional.HasValue.call.loc59: init bool = call %bound_method.loc59_33.5(%.loc59_33.8)
// CHECK:STDOUT:   %.loc59_33.9: bool = value_of_initializer %Optional.HasValue.call.loc59
// CHECK:STDOUT:   %.loc59_33.10: bool = converted %Optional.HasValue.call.loc59, %.loc59_33.9
// CHECK:STDOUT:   if %.loc59_33.10 br !for.body.loc59 else br !for.done.loc59
// CHECK:STDOUT:
// CHECK:STDOUT: !for.body.loc59:
// CHECK:STDOUT:   %.loc59_33.11: %Optional.Get.type.ab0 = specific_constant imports.%Core.import_ref.52c, @Optional(constants.%OptionalStorage.facet.9e6) [concrete = constants.%Optional.Get.6e2]
// CHECK:STDOUT:   %Get.ref.loc59: %Optional.Get.type.ab0 = name_ref Get, %.loc59_33.11 [concrete = constants.%Optional.Get.6e2]
// CHECK:STDOUT:   %Optional.Get.bound.loc59: <bound method> = bound_method %.loc59_33.6, %Get.ref.loc59
// CHECK:STDOUT:   %Optional.Get.specific_fn.loc59: <specific function> = specific_function %Get.ref.loc59, @Optional.Get(constants.%OptionalStorage.facet.9e6) [concrete = constants.%Optional.Get.specific_fn.e6d]
// CHECK:STDOUT:   %bound_method.loc59_33.6: <bound method> = bound_method %.loc59_33.6, %Optional.Get.specific_fn.loc59
// CHECK:STDOUT:   %.loc59_33.12: ref %DiffValueType = temporary_storage
// CHECK:STDOUT:   %.loc59_33.13: %Optional.6ee = acquire_value %.loc59_33.6
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
// CHECK:STDOUT:   %impl.elem2.loc63: %.b0d = impl_witness_access constants.%Iterate.impl_witness.9b4, element2 [concrete = constants.%T.as_type.as.Iterate.impl.NewCursor.3cd]
// CHECK:STDOUT:   %bound_method.loc63_33.1: <bound method> = bound_method %c.ref, %impl.elem2.loc63
// CHECK:STDOUT:   %facet_value.loc63_33.1: %CppRange_where.type.d682d6.1 = facet_value constants.%DiffConstRange, (constants.%custom_witness.7b4, constants.%custom_witness.48b6e9.1, constants.%custom_witness.0c0, constants.%custom_witness.ffb, constants.%custom_witness.fc2, constants.%custom_witness.1bfa2a.1, constants.%custom_witness.d0da43.1, constants.%custom_witness.48b6e9.2) [concrete = constants.%facet_value.683]
// CHECK:STDOUT:   %.loc63_33.1: %CppRange_where.type.d682d6.1 = converted constants.%DiffConstRange, %facet_value.loc63_33.1 [concrete = constants.%facet_value.683]
// CHECK:STDOUT:   %facet_value.loc63_33.2: %CppRange_where.type.d682d6.1 = facet_value constants.%DiffConstRange, (constants.%custom_witness.7b4, constants.%custom_witness.48b6e9.1, constants.%custom_witness.0c0, constants.%custom_witness.ffb, constants.%custom_witness.fc2, constants.%custom_witness.1bfa2a.1, constants.%custom_witness.d0da43.1, constants.%custom_witness.48b6e9.2) [concrete = constants.%facet_value.683]
// CHECK:STDOUT:   %.loc63_33.2: %CppRange_where.type.d682d6.1 = converted constants.%DiffConstRange, %facet_value.loc63_33.2 [concrete = constants.%facet_value.683]
// CHECK:STDOUT:   %specific_fn.loc63_33.1: <specific function> = specific_function %impl.elem2.loc63, @T.as_type.as.Iterate.impl.NewCursor(constants.%facet_value.683) [concrete = constants.%T.as_type.as.Iterate.impl.NewCursor.specific_fn.104]
// CHECK:STDOUT:   %bound_method.loc63_33.2: <bound method> = bound_method %c.ref, %specific_fn.loc63_33.1
// CHECK:STDOUT:   %var.loc63: ref %tuple.type.df5 = var_storage invalid
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.call.loc63: init %tuple.type.df5 to %var.loc63 = call %bound_method.loc63_33.2(%c.ref)
// CHECK:STDOUT:   assign %var.loc63, %T.as_type.as.Iterate.impl.NewCursor.call.loc63
// CHECK:STDOUT:   br !for.next.loc63
// CHECK:STDOUT:
// CHECK:STDOUT: !for.next.loc63:
// CHECK:STDOUT:   %impl.elem3.loc63: %.d85 = impl_witness_access constants.%Iterate.impl_witness.9b4, element3 [concrete = constants.%T.as_type.as.Iterate.impl.Next.d18]
// CHECK:STDOUT:   %bound_method.loc63_33.3: <bound method> = bound_method %c.ref, %impl.elem3.loc63
// CHECK:STDOUT:   %facet_value.loc63_33.3: %CppRange_where.type.d682d6.1 = facet_value constants.%DiffConstRange, (constants.%custom_witness.7b4, constants.%custom_witness.48b6e9.1, constants.%custom_witness.0c0, constants.%custom_witness.ffb, constants.%custom_witness.fc2, constants.%custom_witness.1bfa2a.1, constants.%custom_witness.d0da43.1, constants.%custom_witness.48b6e9.2) [concrete = constants.%facet_value.683]
// CHECK:STDOUT:   %.loc63_33.3: %CppRange_where.type.d682d6.1 = converted constants.%DiffConstRange, %facet_value.loc63_33.3 [concrete = constants.%facet_value.683]
// CHECK:STDOUT:   %facet_value.loc63_33.4: %CppRange_where.type.d682d6.1 = facet_value constants.%DiffConstRange, (constants.%custom_witness.7b4, constants.%custom_witness.48b6e9.1, constants.%custom_witness.0c0, constants.%custom_witness.ffb, constants.%custom_witness.fc2, constants.%custom_witness.1bfa2a.1, constants.%custom_witness.d0da43.1, constants.%custom_witness.48b6e9.2) [concrete = constants.%facet_value.683]
// CHECK:STDOUT:   %.loc63_33.4: %CppRange_where.type.d682d6.1 = converted constants.%DiffConstRange, %facet_value.loc63_33.4 [concrete = constants.%facet_value.683]
// CHECK:STDOUT:   %specific_fn.loc63_33.2: <specific function> = specific_function %impl.elem3.loc63, @T.as_type.as.Iterate.impl.Next(constants.%facet_value.683) [concrete = constants.%T.as_type.as.Iterate.impl.Next.specific_fn.8ee]
// CHECK:STDOUT:   %bound_method.loc63_33.4: <bound method> = bound_method %c.ref, %specific_fn.loc63_33.2
// CHECK:STDOUT:   %.loc63_33.5: ref %Optional.6ee = temporary_storage
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.call.loc63: init %Optional.6ee to %.loc63_33.5 = call %bound_method.loc63_33.4(%c.ref, %var.loc63)
// CHECK:STDOUT:   %.loc63_33.6: ref %Optional.6ee = temporary %.loc63_33.5, %T.as_type.as.Iterate.impl.Next.call.loc63
// CHECK:STDOUT:   %.loc63_33.7: %Optional.HasValue.type.868 = specific_constant imports.%Core.import_ref.4b2, @Optional(constants.%OptionalStorage.facet.9e6) [concrete = constants.%Optional.HasValue.7d8]
// CHECK:STDOUT:   %HasValue.ref.loc63: %Optional.HasValue.type.868 = name_ref HasValue, %.loc63_33.7 [concrete = constants.%Optional.HasValue.7d8]
// CHECK:STDOUT:   %Optional.HasValue.bound.loc63: <bound method> = bound_method %.loc63_33.6, %HasValue.ref.loc63
// CHECK:STDOUT:   %Optional.HasValue.specific_fn.loc63: <specific function> = specific_function %HasValue.ref.loc63, @Optional.HasValue(constants.%OptionalStorage.facet.9e6) [concrete = constants.%Optional.HasValue.specific_fn.126]
// CHECK:STDOUT:   %bound_method.loc63_33.5: <bound method> = bound_method %.loc63_33.6, %Optional.HasValue.specific_fn.loc63
// CHECK:STDOUT:   %.loc63_33.8: %Optional.6ee = acquire_value %.loc63_33.6
// CHECK:STDOUT:   %Optional.HasValue.call.loc63: init bool = call %bound_method.loc63_33.5(%.loc63_33.8)
// CHECK:STDOUT:   %.loc63_33.9: bool = value_of_initializer %Optional.HasValue.call.loc63
// CHECK:STDOUT:   %.loc63_33.10: bool = converted %Optional.HasValue.call.loc63, %.loc63_33.9
// CHECK:STDOUT:   if %.loc63_33.10 br !for.body.loc63 else br !for.done.loc63
// CHECK:STDOUT:
// CHECK:STDOUT: !for.body.loc63:
// CHECK:STDOUT:   %.loc63_33.11: %Optional.Get.type.ab0 = specific_constant imports.%Core.import_ref.52c, @Optional(constants.%OptionalStorage.facet.9e6) [concrete = constants.%Optional.Get.6e2]
// CHECK:STDOUT:   %Get.ref.loc63: %Optional.Get.type.ab0 = name_ref Get, %.loc63_33.11 [concrete = constants.%Optional.Get.6e2]
// CHECK:STDOUT:   %Optional.Get.bound.loc63: <bound method> = bound_method %.loc63_33.6, %Get.ref.loc63
// CHECK:STDOUT:   %Optional.Get.specific_fn.loc63: <specific function> = specific_function %Get.ref.loc63, @Optional.Get(constants.%OptionalStorage.facet.9e6) [concrete = constants.%Optional.Get.specific_fn.e6d]
// CHECK:STDOUT:   %bound_method.loc63_33.6: <bound method> = bound_method %.loc63_33.6, %Optional.Get.specific_fn.loc63
// CHECK:STDOUT:   %.loc63_33.12: ref %DiffValueType = temporary_storage
// CHECK:STDOUT:   %.loc63_33.13: %Optional.6ee = acquire_value %.loc63_33.6
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
// CHECK:STDOUT: specific @TestIterate(constants.%facet_value.2c1) {
// CHECK:STDOUT:   %R.patt.loc40_17.2 => constants.%R.patt
// CHECK:STDOUT:   %R.loc40_17.1 => constants.%facet_value.2c1
// CHECK:STDOUT:   %R.as_type.loc40_79.1 => constants.%DiffMutableRange
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.611
// CHECK:STDOUT:   %r.patt.loc40_77.2 => constants.%r.patt.5fe
// CHECK:STDOUT:   %r.param_patt.loc40_72.2 => constants.%r.param_patt.a4b
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc40 => constants.%complete_type.357
// CHECK:STDOUT:   %Iterate.lookup_impl_witness => constants.%Iterate.impl_witness.606
// CHECK:STDOUT:   %Iterate.facet.loc44_33.5 => constants.%Iterate.facet.758
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type => constants.%Iterate.WithSelf.NewCursor.type.49d
// CHECK:STDOUT:   %.loc44_33.20 => constants.%.c05
// CHECK:STDOUT:   %impl.elem2.loc44_33.3 => constants.%T.as_type.as.Iterate.impl.NewCursor.a7b
// CHECK:STDOUT:   %specific_impl_fn.loc44_33.4 => constants.%T.as_type.as.Iterate.impl.NewCursor.specific_fn.1d2
// CHECK:STDOUT:   %impl.elem1 => constants.%Destroy.facet.645
// CHECK:STDOUT:   %as_type => constants.%tuple.type.b18
// CHECK:STDOUT:   %require_complete.1 => constants.%complete_type.bfe
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type => constants.%Iterate.WithSelf.Next.type.1f8
// CHECK:STDOUT:   %.loc44_33.21 => constants.%.22d
// CHECK:STDOUT:   %impl.elem3.loc44_33.2 => constants.%T.as_type.as.Iterate.impl.Next.8ce
// CHECK:STDOUT:   %specific_impl_fn.loc44_33.5 => constants.%T.as_type.as.Iterate.impl.Next.specific_fn.288
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type => constants.%Destroy.WithSelf.SelfDestruct.type.baf
// CHECK:STDOUT:   %.loc44_33.22 => constants.%.085
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.f8f19d.8
// CHECK:STDOUT:   %impl.elem2.loc44_33.4 => constants.%Destroy.WithSelf.SelfDestruct.db3fdb.11
// CHECK:STDOUT:   %specific_impl_fn.loc44_33.6 => constants.%Destroy.WithSelf.SelfDestruct.db3fdb.11
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestIterate(constants.%facet_value.1fc) {
// CHECK:STDOUT:   %R.patt.loc40_17.2 => constants.%R.patt
// CHECK:STDOUT:   %R.loc40_17.1 => constants.%facet_value.1fc
// CHECK:STDOUT:   %R.as_type.loc40_79.1 => constants.%DiffConstRange
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.7dd
// CHECK:STDOUT:   %r.patt.loc40_77.2 => constants.%r.patt.9c0
// CHECK:STDOUT:   %r.param_patt.loc40_72.2 => constants.%r.param_patt.992
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc40 => constants.%complete_type.357
// CHECK:STDOUT:   %Iterate.lookup_impl_witness => constants.%Iterate.impl_witness.9b4
// CHECK:STDOUT:   %Iterate.facet.loc44_33.5 => constants.%Iterate.facet.930
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type => constants.%Iterate.WithSelf.NewCursor.type.4f6
// CHECK:STDOUT:   %.loc44_33.20 => constants.%.b0d
// CHECK:STDOUT:   %impl.elem2.loc44_33.3 => constants.%T.as_type.as.Iterate.impl.NewCursor.3cd
// CHECK:STDOUT:   %specific_impl_fn.loc44_33.4 => constants.%T.as_type.as.Iterate.impl.NewCursor.specific_fn.104
// CHECK:STDOUT:   %impl.elem1 => constants.%Destroy.facet.206
// CHECK:STDOUT:   %as_type => constants.%tuple.type.df5
// CHECK:STDOUT:   %require_complete.1 => constants.%complete_type.ad6
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type => constants.%Iterate.WithSelf.Next.type.95d
// CHECK:STDOUT:   %.loc44_33.21 => constants.%.d85
// CHECK:STDOUT:   %impl.elem3.loc44_33.2 => constants.%T.as_type.as.Iterate.impl.Next.d18
// CHECK:STDOUT:   %specific_impl_fn.loc44_33.5 => constants.%T.as_type.as.Iterate.impl.Next.specific_fn.8ee
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type => constants.%Destroy.WithSelf.SelfDestruct.type.ee6
// CHECK:STDOUT:   %.loc44_33.22 => constants.%.c1e
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
// CHECK:STDOUT:   %Optional.Get.type.c64: type = fn_type @Optional.Get, @Optional(%T.220) [symbolic]
// CHECK:STDOUT:   %Optional.Get.bcf: %Optional.Get.type.c64 = struct_value () [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.type.49e: type = fn_type @Optional.HasValue, @Optional(%T.220) [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.c20: %Optional.HasValue.type.49e = struct_value () [symbolic]
// CHECK:STDOUT:   %T.283: %facet_type.f48 = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.74b: type = facet_access_type %T.283 [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.28a: <witness> = lookup_impl_witness %T.283, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.16e: %Destroy.type = facet_value %T.as_type.74b, (%Destroy.lookup_impl_witness.28a) [symbolic]
// CHECK:STDOUT:   %MaybeUnformed.9b7: type = class_type @MaybeUnformed, @MaybeUnformed(%Destroy.facet.16e) [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.faa: type = class_type @DefaultOptionalStorage, @DefaultOptionalStorage(%T.283) [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.elem.06a: type = unbound_element_type %DefaultOptionalStorage.faa, bool [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.elem.9db: type = unbound_element_type %DefaultOptionalStorage.faa, %MaybeUnformed.9b7 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Get.type.2f2: type = fn_type @T.as_type.as.OptionalStorage.impl.Get, @T.as_type.as.OptionalStorage.impl(%T.283) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Get.884: %T.as_type.as.OptionalStorage.impl.Get.type.2f2 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Has.type.d28: type = fn_type @T.as_type.as.OptionalStorage.impl.Has, @T.as_type.as.OptionalStorage.impl(%T.283) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Has.f29: %T.as_type.as.OptionalStorage.impl.Has.type.d28 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Some.type.760: type = fn_type @T.as_type.as.OptionalStorage.impl.Some, @T.as_type.as.OptionalStorage.impl(%T.283) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Some.268: %T.as_type.as.OptionalStorage.impl.Some.type.760 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.None.type.925: type = fn_type @T.as_type.as.OptionalStorage.impl.None, @T.as_type.as.OptionalStorage.impl(%T.283) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.None.f42: %T.as_type.as.OptionalStorage.impl.None.type.925 = struct_value () [symbolic]
// CHECK:STDOUT:   %.Self.500: %Iterate.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %CppRange.type: type = facet_type <@CppRange> [concrete]
// CHECK:STDOUT:   %.Self.e0b: %CppRange.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %CppRangeForIterate.lookup_impl_witness.50b: <witness> = lookup_impl_witness %.Self.e0b, @CppRangeForIterate [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.32d: type = impl_witness_access %CppRangeForIterate.lookup_impl_witness.50b, element0 [symbolic_self]
// CHECK:STDOUT:   %CppUnsafeDeref.lookup_impl_witness.851: <witness> = lookup_impl_witness %impl.elem0.32d, @CppUnsafeDeref [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.10d: type = impl_witness_access %CppUnsafeDeref.lookup_impl_witness.851, element0 [symbolic_self]
// CHECK:STDOUT:   %CppRange_where.type.d682d6.1: type = facet_type <@CppRange where %impl.elem0.10d impls @Copy> [concrete]
// CHECK:STDOUT:   %T.f45530.2: %CppRange_where.type.d682d6.1 = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.type.d4a: type = fn_type @T.as_type.as.Iterate.impl.Next, @T.as_type.as.Iterate.impl(%T.f45530.2) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.5a8: %T.as_type.as.Iterate.impl.Next.type.d4a = struct_value () [symbolic]
// CHECK:STDOUT:   %CppRangeForIterate.lookup_impl_witness.b581a8.2: <witness> = lookup_impl_witness %T.f45530.2, @CppRangeForIterate [symbolic]
// CHECK:STDOUT:   %impl.elem0.5f971c.2: type = impl_witness_access %CppRangeForIterate.lookup_impl_witness.b581a8.2, element0 [symbolic]
// CHECK:STDOUT:   %CppUnsafeDeref.lookup_impl_witness.87e543.2: <witness> = lookup_impl_witness %impl.elem0.5f971c.2, @CppUnsafeDeref [symbolic]
// CHECK:STDOUT:   %impl.elem0.493765.2: type = impl_witness_access %CppUnsafeDeref.lookup_impl_witness.87e543.2, element0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.type.8fb: type = fn_type @T.as_type.as.Iterate.impl.NewCursor, @T.as_type.as.Iterate.impl(%T.f45530.2) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.8a1: %T.as_type.as.Iterate.impl.NewCursor.type.8fb = struct_value () [symbolic]
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
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.13a: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.8b0) [symbolic]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.63b: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.8b0) [symbolic]
// CHECK:STDOUT:   %.d64: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.13a, %Iterate.facet.8b0 [symbolic]
// CHECK:STDOUT:   %impl.elem2.8fd: %.d64 = impl_witness_access %Iterate.lookup_impl_witness.e07, element2 [symbolic]
// CHECK:STDOUT:   %impl.elem1.0e3: %Destroy.type = impl_witness_access %Iterate.lookup_impl_witness.e07, element1 [symbolic]
// CHECK:STDOUT:   %as_type.400: type = facet_access_type %impl.elem1.0e3 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.176: <specific function> = specific_impl_function %impl.elem2.8fd, @Iterate.WithSelf.NewCursor(%Iterate.facet.8b0) [symbolic]
// CHECK:STDOUT:   %.6fa: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.63b, %Iterate.facet.8b0 [symbolic]
// CHECK:STDOUT:   %impl.elem3.a35: %.6fa = impl_witness_access %Iterate.lookup_impl_witness.e07, element3 [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.d09: type = class_type @DefaultOptionalStorage, @DefaultOptionalStorage(%facet_value.8be) [concrete]
// CHECK:STDOUT:   %OptionalStorage.impl_witness.774: <witness> = impl_witness imports.%OptionalStorage.impl_witness_table.460, @T.as_type.as.OptionalStorage.impl(%facet_value.8be) [concrete]
// CHECK:STDOUT:   %OptionalStorage.facet.f35: %OptionalStorage.type = facet_value %OverloadValueType, (%OptionalStorage.impl_witness.774) [concrete]
// CHECK:STDOUT:   %Optional.179: type = class_type @Optional, @Optional(%OptionalStorage.facet.f35) [concrete]
// CHECK:STDOUT:   %pattern_type.a04: type = pattern_type %Optional.179 [concrete]
// CHECK:STDOUT:   %specific_impl_fn.efd: <specific function> = specific_impl_function %impl.elem3.a35, @Iterate.WithSelf.Next(%Iterate.facet.8b0) [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.type.ccd: type = fn_type @Optional.HasValue, @Optional(%OptionalStorage.facet.f35) [concrete]
// CHECK:STDOUT:   %Optional.HasValue.6c8: %Optional.HasValue.type.ccd = struct_value () [concrete]
// CHECK:STDOUT:   %Optional.Get.type.7eb: type = fn_type @Optional.Get, @Optional(%OptionalStorage.facet.f35) [concrete]
// CHECK:STDOUT:   %Optional.Get.787: %Optional.Get.type.7eb = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.facet.d74: %Destroy.type = facet_value %OverloadValueType, (%custom_witness.1bfa2a.1) [concrete]
// CHECK:STDOUT:   %MaybeUnformed.9c1: type = class_type @MaybeUnformed, @MaybeUnformed(%Destroy.facet.d74) [concrete]
// CHECK:STDOUT:   %.8a2: type = maybe_unformed_type %OverloadValueType [concrete]
// CHECK:STDOUT:   %DefaultOptionalStorage.elem.da2: type = unbound_element_type %DefaultOptionalStorage.d09, %MaybeUnformed.9c1 [concrete]
// CHECK:STDOUT:   %DefaultOptionalStorage.elem.20b: type = unbound_element_type %DefaultOptionalStorage.d09, bool [concrete]
// CHECK:STDOUT:   %struct_type.value.has_value.567: type = struct_type {.value: %MaybeUnformed.9c1, .has_value: bool} [concrete]
// CHECK:STDOUT:   %Optional.HasValue.specific_fn.dd8: <specific function> = specific_function %Optional.HasValue.6c8, @Optional.HasValue(%OptionalStorage.facet.f35) [concrete]
// CHECK:STDOUT:   %Optional.Get.specific_fn.01a: <specific function> = specific_function %Optional.Get.787, @Optional.Get(%OptionalStorage.facet.f35) [concrete]
// CHECK:STDOUT:   %pattern_type.6ff: type = pattern_type %.8a2 [concrete]
// CHECK:STDOUT:   %self.param_patt.b76: %pattern_type.6ff = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.079: %pattern_type.6ff = wrapper_binding_pattern self, %self.param_patt.b76 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc27_37.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.2: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.3: type = fn_type @Destroy.WithSelf.Op.loc27_37.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.3: %Destroy.WithSelf.Op.type.ef016f.3 = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.e21: type = pattern_type %MaybeUnformed.9c1 [concrete]
// CHECK:STDOUT:   %self.param_patt.a96: %pattern_type.e21 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.8f5: %pattern_type.e21 = wrapper_binding_pattern self, %self.param_patt.a96 [concrete]
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
// CHECK:STDOUT:   %pattern_type.3a1: type = pattern_type %struct_type.value.has_value.567 [concrete]
// CHECK:STDOUT:   %self.param_patt.2f6: %pattern_type.3a1 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.d60: %pattern_type.3a1 = wrapper_binding_pattern self, %self.param_patt.2f6 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.5: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc27_37.4 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.5: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.5 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.6: type = fn_type @Destroy.WithSelf.Op.loc27_37.4 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.6: %Destroy.WithSelf.Op.type.ef016f.6 = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.a2d: type = pattern_type %DefaultOptionalStorage.d09 [concrete]
// CHECK:STDOUT:   %self.param_patt.622: %pattern_type.a2d = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.856: %pattern_type.a2d = wrapper_binding_pattern self, %self.param_patt.622 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.6: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc27_37.5 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.6: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.6 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.7: type = fn_type @Destroy.WithSelf.Op.loc27_37.5 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.7: %Destroy.WithSelf.Op.type.ef016f.7 = struct_value () [concrete]
// CHECK:STDOUT:   %self.param_patt.479: %pattern_type.a04 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.178: %pattern_type.a04 = wrapper_binding_pattern self, %self.param_patt.479 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.7: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc27_37.6 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.7: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.7 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.8: type = fn_type @Destroy.WithSelf.Op.loc27_37.6 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.8: %Destroy.WithSelf.Op.type.ef016f.8 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.8: type = fn_type @Destroy.WithSelf.SelfDestruct.loc27_37.6 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.8: %Destroy.WithSelf.SelfDestruct.type.fbceb5.8 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.a78: <witness> = lookup_impl_witness %impl.elem1.0e3, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.45d: type = fn_type @Destroy.WithSelf.SelfDestruct.1, @Destroy.WithSelf(%impl.elem1.0e3) [symbolic]
// CHECK:STDOUT:   %.817: type = fn_type_with_self_type %Destroy.WithSelf.SelfDestruct.type.45d, %impl.elem1.0e3 [symbolic]
// CHECK:STDOUT:   %impl.elem2.25c: %.817 = impl_witness_access %Destroy.lookup_impl_witness.a78, element2 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.26e: <specific function> = specific_impl_function %impl.elem2.25c, @Destroy.WithSelf.SelfDestruct.1(%impl.elem1.0e3) [symbolic]
// CHECK:STDOUT:   %Range: type = class_type @Range [concrete]
// CHECK:STDOUT:   %pattern_type.23f: type = pattern_type %Range [concrete]
// CHECK:STDOUT:   %Iterator: type = class_type @Iterator.1 [concrete]
// CHECK:STDOUT:   %Range.Begin.type: type = fn_type @Range.Begin [concrete]
// CHECK:STDOUT:   %Range.Begin: %Range.Begin.type = struct_value () [concrete]
// CHECK:STDOUT:   %Range.End.type: type = fn_type @Range.End [concrete]
// CHECK:STDOUT:   %Range.End: %Range.End.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.090: <witness> = custom_witness (%Iterator, %Iterator, %Range.Begin, %Range.End), @CppRangeForIterate [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.9: type = fn_type @Destroy.WithSelf.Op.loc32 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.9: %Destroy.WithSelf.Op.type.ef016f.9 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.9: type = fn_type @Destroy.WithSelf.SelfDestruct.loc32 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.9: %Destroy.WithSelf.SelfDestruct.type.fbceb5.9 = struct_value () [concrete]
// CHECK:STDOUT:   %Iterator.SubobjectDestroy.type: type = fn_type @Iterator.SubobjectDestroy [concrete]
// CHECK:STDOUT:   %Iterator.SubobjectDestroy: %Iterator.SubobjectDestroy.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.eaa: <witness> = custom_witness (%Destroy.WithSelf.Op.403171.9, %Iterator.SubobjectDestroy, %Destroy.WithSelf.SelfDestruct.db3fdb.9), @Destroy [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.4232e0.1: type = fn_type @Iterator.Op.1 [concrete]
// CHECK:STDOUT:   %Iterator.Op.e8caec.1: %Iterator.Op.type.4232e0.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.6bd: <witness> = custom_witness (%OverloadValueType, %Iterator.Op.e8caec.1), @CppUnsafeDeref [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.4232e0.2: type = fn_type @Iterator.Op.2 [concrete]
// CHECK:STDOUT:   %Iterator.Op.e8caec.2: %Iterator.Op.type.4232e0.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.716: <witness> = custom_witness (%Iterator.Op.e8caec.2), @Inc [concrete]
// CHECK:STDOUT:   %Equal.type: type = fn_type @Equal [concrete]
// CHECK:STDOUT:   %Equal: %Equal.type = struct_value () [concrete]
// CHECK:STDOUT:   %NotEqual.type: type = fn_type @NotEqual [concrete]
// CHECK:STDOUT:   %NotEqual: %NotEqual.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.f93: <witness> = custom_witness (%Equal, %NotEqual), @EqWith, @EqWith(%Iterator) [concrete]
// CHECK:STDOUT:   %facet_value.8e3: %CppRange_where.type.d682d6.1 = facet_value %Range, (%custom_witness.090, %custom_witness.eaa, %custom_witness.6bd, %custom_witness.716, %custom_witness.f93, %custom_witness.1bfa2a.1, %custom_witness.d0da43.1, %custom_witness.eaa) [concrete]
// CHECK:STDOUT:   %tuple.type.9ef: type = tuple_type (%Iterator, %Iterator) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.8: type = fn_type @Destroy.WithSelf.SubobjectDestroy.3 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.8: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.8 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.10: type = fn_type @Destroy.WithSelf.Op.3 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.10: %Destroy.WithSelf.Op.type.ef016f.10 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.10: type = fn_type @Destroy.WithSelf.SelfDestruct.3 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.10: %Destroy.WithSelf.SelfDestruct.type.fbceb5.10 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.f8f19d.8: <witness> = custom_witness (%Destroy.WithSelf.Op.403171.10, %Destroy.WithSelf.SubobjectDestroy.d01daf.8, %Destroy.WithSelf.SelfDestruct.db3fdb.10), @Destroy [concrete]
// CHECK:STDOUT:   %Destroy.facet.c69d: %Destroy.type = facet_value %tuple.type.9ef, (%custom_witness.f8f19d.8) [concrete]
// CHECK:STDOUT:   %Iterate.impl_witness.e03: <witness> = impl_witness imports.%Iterate.impl_witness_table.121, @T.as_type.as.Iterate.impl(%facet_value.8e3) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.type.549: type = fn_type @T.as_type.as.Iterate.impl.NewCursor, @T.as_type.as.Iterate.impl(%facet_value.8e3) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.60e: %T.as_type.as.Iterate.impl.NewCursor.type.549 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.type.c07: type = fn_type @T.as_type.as.Iterate.impl.Next, @T.as_type.as.Iterate.impl(%facet_value.8e3) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.23c: %T.as_type.as.Iterate.impl.Next.type.c07 = struct_value () [concrete]
// CHECK:STDOUT:   %Iterate.facet.a3e: %Iterate.type = facet_value %Range, (%Iterate.impl_witness.e03) [concrete]
// CHECK:STDOUT:   %facet_value.637: %Iterate_where.type.2ee = facet_value %Range, (%Iterate.impl_witness.e03) [concrete]
// CHECK:STDOUT:   %r.patt.b9f: %pattern_type.23f = ref_binding_pattern r [concrete]
// CHECK:STDOUT:   %r.param_patt.466: %pattern_type.23f = var_param_pattern %r.patt.b9f [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.b10: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.a3e) [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.42b: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.a3e) [concrete]
// CHECK:STDOUT:   %.f2a: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.b10, %Iterate.facet.a3e [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.specific_fn: <specific function> = specific_function %T.as_type.as.Iterate.impl.NewCursor.60e, @T.as_type.as.Iterate.impl.NewCursor(%facet_value.8e3) [concrete]
// CHECK:STDOUT:   %.96e: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.42b, %Iterate.facet.a3e [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.specific_fn: <specific function> = specific_function %T.as_type.as.Iterate.impl.Next.23c, @T.as_type.as.Iterate.impl.Next(%facet_value.8e3) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.106: type = fn_type @Destroy.WithSelf.SelfDestruct.1, @Destroy.WithSelf(%Destroy.facet.c69d) [concrete]
// CHECK:STDOUT:   %.227: type = fn_type_with_self_type %Destroy.WithSelf.SelfDestruct.type.106, %Destroy.facet.c69d [concrete]
// CHECK:STDOUT:   %complete_type.e35: <witness> = complete_type_witness %tuple.type.9ef [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .OverloadValueType = %OverloadValueType.decl
// CHECK:STDOUT:     .Range = %Range.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.4b2: @Optional.%Optional.HasValue.type (%Optional.HasValue.type.49e) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.HasValue (constants.%Optional.HasValue.c20)]
// CHECK:STDOUT:   %Core.import_ref.52c: @Optional.%Optional.Get.type (%Optional.Get.type.c64) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.Get (constants.%Optional.Get.bcf)]
// CHECK:STDOUT:   %Core.import_ref.ace: type = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%DefaultOptionalStorage (constants.%DefaultOptionalStorage.faa)]
// CHECK:STDOUT:   %Core.import_ref.ef4: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.None.type (%T.as_type.as.OptionalStorage.impl.None.type.925) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.None (constants.%T.as_type.as.OptionalStorage.impl.None.f42)]
// CHECK:STDOUT:   %Core.import_ref.a20: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Some.type (%T.as_type.as.OptionalStorage.impl.Some.type.760) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Some (constants.%T.as_type.as.OptionalStorage.impl.Some.268)]
// CHECK:STDOUT:   %Core.import_ref.af7: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Has.type (%T.as_type.as.OptionalStorage.impl.Has.type.d28) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Has (constants.%T.as_type.as.OptionalStorage.impl.Has.f29)]
// CHECK:STDOUT:   %Core.import_ref.dd4: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Get.type (%T.as_type.as.OptionalStorage.impl.Get.type.2f2) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Get (constants.%T.as_type.as.OptionalStorage.impl.Get.884)]
// CHECK:STDOUT:   %Core.import_ref.554 = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], unloaded
// CHECK:STDOUT:   %OptionalStorage.impl_witness_table.460 = impl_witness_table (%Core.import_ref.ace, %Core.import_ref.ef4, %Core.import_ref.a20, %Core.import_ref.af7, %Core.import_ref.dd4, %Core.import_ref.554), @T.as_type.as.OptionalStorage.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.1d7: @DefaultOptionalStorage.%DefaultOptionalStorage.elem.1 (%DefaultOptionalStorage.elem.9db) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [concrete = %field_decl.5f0]
// CHECK:STDOUT:   %Core.import_ref.929: @DefaultOptionalStorage.%DefaultOptionalStorage.elem.2 (%DefaultOptionalStorage.elem.06a) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [concrete = %field_decl.6a8]
// CHECK:STDOUT:   %Core.import_ref.eef: %facet_type.f48 = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.Iterate.impl.%facet_value (constants.%facet_value.f93)]
// CHECK:STDOUT:   %Core.import_ref.fcc: %Destroy.type = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.Iterate.impl.%Destroy.facet (constants.%Destroy.facet.9cd)]
// CHECK:STDOUT:   %Core.import_ref.946: @T.as_type.as.Iterate.impl.%T.as_type.as.Iterate.impl.NewCursor.type (%T.as_type.as.Iterate.impl.NewCursor.type.8fb) = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.Iterate.impl.%T.as_type.as.Iterate.impl.NewCursor (constants.%T.as_type.as.Iterate.impl.NewCursor.8a1)]
// CHECK:STDOUT:   %Core.import_ref.cfb: @T.as_type.as.Iterate.impl.%T.as_type.as.Iterate.impl.Next.type (%T.as_type.as.Iterate.impl.Next.type.d4a) = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.Iterate.impl.%T.as_type.as.Iterate.impl.Next (constants.%T.as_type.as.Iterate.impl.Next.5a8)]
// CHECK:STDOUT:   %Iterate.impl_witness_table.121 = impl_witness_table (%Core.import_ref.eef, %Core.import_ref.fcc, %Core.import_ref.946, %Core.import_ref.cfb), @T.as_type.as.Iterate.impl [concrete]
// CHECK:STDOUT:   %OverloadValueType.decl: type = class_decl @OverloadValueType [concrete = constants.%OverloadValueType] {} {}
// CHECK:STDOUT:   %Core.import_ref.b0d = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %field_decl.6a8: @DefaultOptionalStorage.%DefaultOptionalStorage.elem.2 (%DefaultOptionalStorage.elem.06a) = field_decl has_value, element1, %Core.import_ref.b0d in [concrete] {}
// CHECK:STDOUT:   %Core.import_ref.87f = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %field_decl.5f0: @DefaultOptionalStorage.%DefaultOptionalStorage.elem.1 (%DefaultOptionalStorage.elem.9db) = field_decl value, element0, %Core.import_ref.87f in [concrete] {}
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
// CHECK:STDOUT:     %self.param_patt: %pattern_type.e21 = ref_param_pattern [concrete = constants.%self.param_patt.a96]
// CHECK:STDOUT:     %self.patt: %pattern_type.e21 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.8f5]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %MaybeUnformed.9c1 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %MaybeUnformed.9c1 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.3: %Destroy.WithSelf.Op.type.ef016f.4 = fn_decl @Destroy.WithSelf.Op.loc27_37.2 [concrete = constants.%Destroy.WithSelf.Op.403171.4] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.e21 = ref_param_pattern [concrete = constants.%self.param_patt.a96]
// CHECK:STDOUT:     %self.patt: %pattern_type.e21 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.8f5]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %MaybeUnformed.9c1 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %MaybeUnformed.9c1 = wrapper_binding self, %self.param
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
// CHECK:STDOUT:     %self.param_patt: %pattern_type.3a1 = ref_param_pattern [concrete = constants.%self.param_patt.2f6]
// CHECK:STDOUT:     %self.patt: %pattern_type.3a1 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.d60]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %struct_type.value.has_value.567 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %struct_type.value.has_value.567 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.5: %Destroy.WithSelf.Op.type.ef016f.6 = fn_decl @Destroy.WithSelf.Op.loc27_37.4 [concrete = constants.%Destroy.WithSelf.Op.403171.6] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.3a1 = ref_param_pattern [concrete = constants.%self.param_patt.2f6]
// CHECK:STDOUT:     %self.patt: %pattern_type.3a1 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.d60]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %struct_type.value.has_value.567 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %struct_type.value.has_value.567 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.5: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.6 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc27_37.5 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.6] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.a2d = ref_param_pattern [concrete = constants.%self.param_patt.622]
// CHECK:STDOUT:     %self.patt: %pattern_type.a2d = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.856]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %DefaultOptionalStorage.d09 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %DefaultOptionalStorage.d09 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.6: %Destroy.WithSelf.Op.type.ef016f.7 = fn_decl @Destroy.WithSelf.Op.loc27_37.5 [concrete = constants.%Destroy.WithSelf.Op.403171.7] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.a2d = ref_param_pattern [concrete = constants.%self.param_patt.622]
// CHECK:STDOUT:     %self.patt: %pattern_type.a2d = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.856]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %DefaultOptionalStorage.d09 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %DefaultOptionalStorage.d09 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.6: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.7 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc27_37.6 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.7] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.a04 = ref_param_pattern [concrete = constants.%self.param_patt.479]
// CHECK:STDOUT:     %self.patt: %pattern_type.a04 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.178]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %Optional.179 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %Optional.179 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.7: %Destroy.WithSelf.Op.type.ef016f.8 = fn_decl @Destroy.WithSelf.Op.loc27_37.6 [concrete = constants.%Destroy.WithSelf.Op.403171.8] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.a04 = ref_param_pattern [concrete = constants.%self.param_patt.479]
// CHECK:STDOUT:     %self.patt: %pattern_type.a04 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.178]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %Optional.179 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %Optional.179 = wrapper_binding self, %self.param
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
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.loc27_37.5) [symbolic = %Iterate.WithSelf.NewCursor.type (constants.%Iterate.WithSelf.NewCursor.type.13a)]
// CHECK:STDOUT:   %.loc27_37.20: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type, %Iterate.facet.loc27_37.5 [symbolic = %.loc27_37.20 (constants.%.d64)]
// CHECK:STDOUT:   %impl.elem2.loc27_37.3: @TestIterate.%.loc27_37.20 (%.d64) = impl_witness_access %Iterate.lookup_impl_witness, element2 [symbolic = %impl.elem2.loc27_37.3 (constants.%impl.elem2.8fd)]
// CHECK:STDOUT:   %specific_impl_fn.loc27_37.4: <specific function> = specific_impl_function %impl.elem2.loc27_37.3, @Iterate.WithSelf.NewCursor(%Iterate.facet.loc27_37.5) [symbolic = %specific_impl_fn.loc27_37.4 (constants.%specific_impl_fn.176)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.loc27_37.5) [symbolic = %Iterate.WithSelf.Next.type (constants.%Iterate.WithSelf.Next.type.63b)]
// CHECK:STDOUT:   %.loc27_37.21: type = fn_type_with_self_type %Iterate.WithSelf.Next.type, %Iterate.facet.loc27_37.5 [symbolic = %.loc27_37.21 (constants.%.6fa)]
// CHECK:STDOUT:   %impl.elem3.loc27_37.2: @TestIterate.%.loc27_37.21 (%.6fa) = impl_witness_access %Iterate.lookup_impl_witness, element3 [symbolic = %impl.elem3.loc27_37.2 (constants.%impl.elem3.a35)]
// CHECK:STDOUT:   %specific_impl_fn.loc27_37.5: <specific function> = specific_impl_function %impl.elem3.loc27_37.2, @Iterate.WithSelf.Next(%Iterate.facet.loc27_37.5) [symbolic = %specific_impl_fn.loc27_37.5 (constants.%specific_impl_fn.efd)]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type: type = fn_type @Destroy.WithSelf.SelfDestruct.1, @Destroy.WithSelf(%impl.elem1) [symbolic = %Destroy.WithSelf.SelfDestruct.type (constants.%Destroy.WithSelf.SelfDestruct.type.45d)]
// CHECK:STDOUT:   %.loc27_37.22: type = fn_type_with_self_type %Destroy.WithSelf.SelfDestruct.type, %impl.elem1 [symbolic = %.loc27_37.22 (constants.%.817)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %impl.elem1, @Destroy [symbolic = %Destroy.lookup_impl_witness (constants.%Destroy.lookup_impl_witness.a78)]
// CHECK:STDOUT:   %impl.elem2.loc27_37.4: @TestIterate.%.loc27_37.22 (%.817) = impl_witness_access %Destroy.lookup_impl_witness, element2 [symbolic = %impl.elem2.loc27_37.4 (constants.%impl.elem2.25c)]
// CHECK:STDOUT:   %specific_impl_fn.loc27_37.6: <specific function> = specific_impl_function %impl.elem2.loc27_37.4, @Destroy.WithSelf.SelfDestruct.1(%impl.elem1) [symbolic = %specific_impl_fn.loc27_37.6 (constants.%specific_impl_fn.26e)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%r.param: ref @TestIterate.%R.as_type.loc25_83.1 (%R.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %_.patt: %pattern_type.804 = value_binding_pattern _ [concrete = constants.%_.patt.a10]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %r.ref: ref @TestIterate.%R.as_type.loc25_83.1 (%R.as_type) = name_ref r, %r
// CHECK:STDOUT:     %impl.elem2.loc27_37.1: @TestIterate.%.loc27_37.20 (%.d64) = impl_witness_access constants.%Iterate.lookup_impl_witness.e07, element2 [symbolic = %impl.elem2.loc27_37.3 (constants.%impl.elem2.8fd)]
// CHECK:STDOUT:     %bound_method.loc27_37.1: <bound method> = bound_method %r.ref, %impl.elem2.loc27_37.1
// CHECK:STDOUT:     %Iterate.facet.loc27_37.1: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.e07) [symbolic = %Iterate.facet.loc27_37.5 (constants.%Iterate.facet.8b0)]
// CHECK:STDOUT:     %.loc27_37.1: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc27_37.1 [symbolic = %Iterate.facet.loc27_37.5 (constants.%Iterate.facet.8b0)]
// CHECK:STDOUT:     %Iterate.facet.loc27_37.2: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.e07) [symbolic = %Iterate.facet.loc27_37.5 (constants.%Iterate.facet.8b0)]
// CHECK:STDOUT:     %.loc27_37.2: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc27_37.2 [symbolic = %Iterate.facet.loc27_37.5 (constants.%Iterate.facet.8b0)]
// CHECK:STDOUT:     %specific_impl_fn.loc27_37.1: <specific function> = specific_impl_function %impl.elem2.loc27_37.1, @Iterate.WithSelf.NewCursor(constants.%Iterate.facet.8b0) [symbolic = %specific_impl_fn.loc27_37.4 (constants.%specific_impl_fn.176)]
// CHECK:STDOUT:     %bound_method.loc27_37.2: <bound method> = bound_method %r.ref, %specific_impl_fn.loc27_37.1
// CHECK:STDOUT:     %var: ref @TestIterate.%as_type (%as_type.400) = var_storage invalid
// CHECK:STDOUT:     %.loc27_36.1: @TestIterate.%R.as_type.loc25_83.1 (%R.as_type) = acquire_value %r.ref
// CHECK:STDOUT:     %Iterate.WithSelf.NewCursor.call: init @TestIterate.%as_type (%as_type.400) to %var = call %bound_method.loc27_37.2(%.loc27_36.1)
// CHECK:STDOUT:     assign %var, %Iterate.WithSelf.NewCursor.call
// CHECK:STDOUT:     br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.next:
// CHECK:STDOUT:     %impl.elem3.loc27_37.1: @TestIterate.%.loc27_37.21 (%.6fa) = impl_witness_access constants.%Iterate.lookup_impl_witness.e07, element3 [symbolic = %impl.elem3.loc27_37.2 (constants.%impl.elem3.a35)]
// CHECK:STDOUT:     %bound_method.loc27_37.3: <bound method> = bound_method %r.ref, %impl.elem3.loc27_37.1
// CHECK:STDOUT:     %Iterate.facet.loc27_37.3: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.e07) [symbolic = %Iterate.facet.loc27_37.5 (constants.%Iterate.facet.8b0)]
// CHECK:STDOUT:     %.loc27_37.3: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc27_37.3 [symbolic = %Iterate.facet.loc27_37.5 (constants.%Iterate.facet.8b0)]
// CHECK:STDOUT:     %Iterate.facet.loc27_37.4: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.e07) [symbolic = %Iterate.facet.loc27_37.5 (constants.%Iterate.facet.8b0)]
// CHECK:STDOUT:     %.loc27_37.4: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc27_37.4 [symbolic = %Iterate.facet.loc27_37.5 (constants.%Iterate.facet.8b0)]
// CHECK:STDOUT:     %.loc27_37.5: %Destroy.type = converted constants.%as_type.400, constants.%impl.elem1.0e3 [symbolic = %impl.elem1 (constants.%impl.elem1.0e3)]
// CHECK:STDOUT:     %.loc27_37.6: %Destroy.type = converted constants.%as_type.400, constants.%impl.elem1.0e3 [symbolic = %impl.elem1 (constants.%impl.elem1.0e3)]
// CHECK:STDOUT:     %specific_impl_fn.loc27_37.2: <specific function> = specific_impl_function %impl.elem3.loc27_37.1, @Iterate.WithSelf.Next(constants.%Iterate.facet.8b0) [symbolic = %specific_impl_fn.loc27_37.5 (constants.%specific_impl_fn.efd)]
// CHECK:STDOUT:     %bound_method.loc27_37.4: <bound method> = bound_method %r.ref, %specific_impl_fn.loc27_37.2
// CHECK:STDOUT:     %.loc27_37.7: ref %Optional.179 = temporary_storage
// CHECK:STDOUT:     %.loc27_36.2: @TestIterate.%R.as_type.loc25_83.1 (%R.as_type) = acquire_value %r.ref
// CHECK:STDOUT:     %Iterate.WithSelf.Next.call: init %Optional.179 to %.loc27_37.7 = call %bound_method.loc27_37.4(%.loc27_36.2, %var)
// CHECK:STDOUT:     %.loc27_37.8: ref %Optional.179 = temporary %.loc27_37.7, %Iterate.WithSelf.Next.call
// CHECK:STDOUT:     %.loc27_37.9: %Optional.HasValue.type.ccd = specific_constant imports.%Core.import_ref.4b2, @Optional(constants.%OptionalStorage.facet.f35) [concrete = constants.%Optional.HasValue.6c8]
// CHECK:STDOUT:     %HasValue.ref: %Optional.HasValue.type.ccd = name_ref HasValue, %.loc27_37.9 [concrete = constants.%Optional.HasValue.6c8]
// CHECK:STDOUT:     %Optional.HasValue.bound: <bound method> = bound_method %.loc27_37.8, %HasValue.ref
// CHECK:STDOUT:     %Optional.HasValue.specific_fn: <specific function> = specific_function %HasValue.ref, @Optional.HasValue(constants.%OptionalStorage.facet.f35) [concrete = constants.%Optional.HasValue.specific_fn.dd8]
// CHECK:STDOUT:     %bound_method.loc27_37.5: <bound method> = bound_method %.loc27_37.8, %Optional.HasValue.specific_fn
// CHECK:STDOUT:     %.loc27_37.10: %Optional.179 = acquire_value %.loc27_37.8
// CHECK:STDOUT:     %Optional.HasValue.call: init bool = call %bound_method.loc27_37.5(%.loc27_37.10)
// CHECK:STDOUT:     %.loc27_37.11: bool = value_of_initializer %Optional.HasValue.call
// CHECK:STDOUT:     %.loc27_37.12: bool = converted %Optional.HasValue.call, %.loc27_37.11
// CHECK:STDOUT:     if %.loc27_37.12 br !for.body else br !for.done
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.body:
// CHECK:STDOUT:     %.loc27_37.13: %Optional.Get.type.7eb = specific_constant imports.%Core.import_ref.52c, @Optional(constants.%OptionalStorage.facet.f35) [concrete = constants.%Optional.Get.787]
// CHECK:STDOUT:     %Get.ref: %Optional.Get.type.7eb = name_ref Get, %.loc27_37.13 [concrete = constants.%Optional.Get.787]
// CHECK:STDOUT:     %Optional.Get.bound: <bound method> = bound_method %.loc27_37.8, %Get.ref
// CHECK:STDOUT:     %Optional.Get.specific_fn: <specific function> = specific_function %Get.ref, @Optional.Get(constants.%OptionalStorage.facet.f35) [concrete = constants.%Optional.Get.specific_fn.01a]
// CHECK:STDOUT:     %bound_method.loc27_37.6: <bound method> = bound_method %.loc27_37.8, %Optional.Get.specific_fn
// CHECK:STDOUT:     %.loc27_37.14: ref %OverloadValueType = temporary_storage
// CHECK:STDOUT:     %.loc27_37.15: %Optional.179 = acquire_value %.loc27_37.8
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
// CHECK:STDOUT:     %impl.elem2.loc27_37.2: @TestIterate.%.loc27_37.22 (%.817) = impl_witness_access constants.%Destroy.lookup_impl_witness.a78, element2 [symbolic = %impl.elem2.loc27_37.4 (constants.%impl.elem2.25c)]
// CHECK:STDOUT:     %bound_method.loc27_37.7: <bound method> = bound_method %var, %impl.elem2.loc27_37.2
// CHECK:STDOUT:     %.loc27_37.18: %Destroy.type = converted constants.%as_type.400, constants.%impl.elem1.0e3 [symbolic = %impl.elem1 (constants.%impl.elem1.0e3)]
// CHECK:STDOUT:     %.loc27_37.19: %Destroy.type = converted constants.%as_type.400, constants.%impl.elem1.0e3 [symbolic = %impl.elem1 (constants.%impl.elem1.0e3)]
// CHECK:STDOUT:     %specific_impl_fn.loc27_37.3: <specific function> = specific_impl_function %impl.elem2.loc27_37.2, @Destroy.WithSelf.SelfDestruct.1(constants.%impl.elem1.0e3) [symbolic = %specific_impl_fn.loc27_37.6 (constants.%specific_impl_fn.26e)]
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
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc27_37.2(%self.param: ref %MaybeUnformed.9c1) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc27_37.2(%self.param: ref %MaybeUnformed.9c1) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc27_37.2(%self.param: ref %MaybeUnformed.9c1) {
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
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc27_37.4(%self.param: ref %struct_type.value.has_value.567) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc27_37.1: ref bool = struct_access %self.param, element1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc27_37.1: <bound method> = bound_method %.loc27_37.1, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.5
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc27_37.1: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc27_37.1(%.loc27_37.1)
// CHECK:STDOUT:   %.loc27_37.2: ref %MaybeUnformed.9c1 = struct_access %self.param, element0
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc27_37.2: <bound method> = bound_method %.loc27_37.2, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.4
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc27_37.2: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc27_37.2(%.loc27_37.2)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc27_37.4(%self.param: ref %struct_type.value.has_value.567) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc27_37.4(%self.param: ref %struct_type.value.has_value.567) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.5(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.4(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc27_37.5(%self.param: ref %DefaultOptionalStorage.d09) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %has_value.ref: %DefaultOptionalStorage.elem.20b = name_ref has_value, imports.%Core.import_ref.929 [concrete = imports.%field_decl.6a8]
// CHECK:STDOUT:   %.loc27_37.1: ref bool = class_element_access %self.param, element1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc27_37.1: <bound method> = bound_method %.loc27_37.1, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.5
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc27_37.1: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc27_37.1(%.loc27_37.1)
// CHECK:STDOUT:   %value.ref: %DefaultOptionalStorage.elem.da2 = name_ref value, imports.%Core.import_ref.1d7 [concrete = imports.%field_decl.5f0]
// CHECK:STDOUT:   %.loc27_37.2: ref %MaybeUnformed.9c1 = class_element_access %self.param, element0
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc27_37.2: <bound method> = bound_method %.loc27_37.2, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.4
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc27_37.2: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc27_37.2(%.loc27_37.2)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc27_37.5(%self.param: ref %DefaultOptionalStorage.d09) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc27_37.5(%self.param: ref %DefaultOptionalStorage.d09) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.6(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.5(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc27_37.6(%self.param: ref %Optional.179) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc27_37.6(%self.param: ref %Optional.179) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc27_37.6(%self.param: ref %Optional.179) {
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
// CHECK:STDOUT:   %impl.elem2: %.f2a = impl_witness_access constants.%Iterate.impl_witness.e03, element2 [concrete = constants.%T.as_type.as.Iterate.impl.NewCursor.60e]
// CHECK:STDOUT:   %bound_method.loc37_37.1: <bound method> = bound_method %r.ref, %impl.elem2
// CHECK:STDOUT:   %facet_value.loc37_37.1: %CppRange_where.type.d682d6.1 = facet_value constants.%Range, (constants.%custom_witness.090, constants.%custom_witness.eaa, constants.%custom_witness.6bd, constants.%custom_witness.716, constants.%custom_witness.f93, constants.%custom_witness.1bfa2a.1, constants.%custom_witness.d0da43.1, constants.%custom_witness.eaa) [concrete = constants.%facet_value.8e3]
// CHECK:STDOUT:   %.loc37_37.1: %CppRange_where.type.d682d6.1 = converted constants.%Range, %facet_value.loc37_37.1 [concrete = constants.%facet_value.8e3]
// CHECK:STDOUT:   %facet_value.loc37_37.2: %CppRange_where.type.d682d6.1 = facet_value constants.%Range, (constants.%custom_witness.090, constants.%custom_witness.eaa, constants.%custom_witness.6bd, constants.%custom_witness.716, constants.%custom_witness.f93, constants.%custom_witness.1bfa2a.1, constants.%custom_witness.d0da43.1, constants.%custom_witness.eaa) [concrete = constants.%facet_value.8e3]
// CHECK:STDOUT:   %.loc37_37.2: %CppRange_where.type.d682d6.1 = converted constants.%Range, %facet_value.loc37_37.2 [concrete = constants.%facet_value.8e3]
// CHECK:STDOUT:   %specific_fn.loc37_37.1: <specific function> = specific_function %impl.elem2, @T.as_type.as.Iterate.impl.NewCursor(constants.%facet_value.8e3) [concrete = constants.%T.as_type.as.Iterate.impl.NewCursor.specific_fn]
// CHECK:STDOUT:   %bound_method.loc37_37.2: <bound method> = bound_method %r.ref, %specific_fn.loc37_37.1
// CHECK:STDOUT:   %var: ref %tuple.type.9ef = var_storage invalid
// CHECK:STDOUT:   %.loc37_36.1: %Range = acquire_value %r.ref
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.call: init %tuple.type.9ef to %var = call %bound_method.loc37_37.2(%.loc37_36.1)
// CHECK:STDOUT:   assign %var, %T.as_type.as.Iterate.impl.NewCursor.call
// CHECK:STDOUT:   br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT: !for.next:
// CHECK:STDOUT:   %impl.elem3: %.96e = impl_witness_access constants.%Iterate.impl_witness.e03, element3 [concrete = constants.%T.as_type.as.Iterate.impl.Next.23c]
// CHECK:STDOUT:   %bound_method.loc37_37.3: <bound method> = bound_method %r.ref, %impl.elem3
// CHECK:STDOUT:   %facet_value.loc37_37.3: %CppRange_where.type.d682d6.1 = facet_value constants.%Range, (constants.%custom_witness.090, constants.%custom_witness.eaa, constants.%custom_witness.6bd, constants.%custom_witness.716, constants.%custom_witness.f93, constants.%custom_witness.1bfa2a.1, constants.%custom_witness.d0da43.1, constants.%custom_witness.eaa) [concrete = constants.%facet_value.8e3]
// CHECK:STDOUT:   %.loc37_37.3: %CppRange_where.type.d682d6.1 = converted constants.%Range, %facet_value.loc37_37.3 [concrete = constants.%facet_value.8e3]
// CHECK:STDOUT:   %facet_value.loc37_37.4: %CppRange_where.type.d682d6.1 = facet_value constants.%Range, (constants.%custom_witness.090, constants.%custom_witness.eaa, constants.%custom_witness.6bd, constants.%custom_witness.716, constants.%custom_witness.f93, constants.%custom_witness.1bfa2a.1, constants.%custom_witness.d0da43.1, constants.%custom_witness.eaa) [concrete = constants.%facet_value.8e3]
// CHECK:STDOUT:   %.loc37_37.4: %CppRange_where.type.d682d6.1 = converted constants.%Range, %facet_value.loc37_37.4 [concrete = constants.%facet_value.8e3]
// CHECK:STDOUT:   %specific_fn.loc37_37.2: <specific function> = specific_function %impl.elem3, @T.as_type.as.Iterate.impl.Next(constants.%facet_value.8e3) [concrete = constants.%T.as_type.as.Iterate.impl.Next.specific_fn]
// CHECK:STDOUT:   %bound_method.loc37_37.4: <bound method> = bound_method %r.ref, %specific_fn.loc37_37.2
// CHECK:STDOUT:   %.loc37_37.5: ref %Optional.179 = temporary_storage
// CHECK:STDOUT:   %.loc37_36.2: %Range = acquire_value %r.ref
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.call: init %Optional.179 to %.loc37_37.5 = call %bound_method.loc37_37.4(%.loc37_36.2, %var)
// CHECK:STDOUT:   %.loc37_37.6: ref %Optional.179 = temporary %.loc37_37.5, %T.as_type.as.Iterate.impl.Next.call
// CHECK:STDOUT:   %.loc37_37.7: %Optional.HasValue.type.ccd = specific_constant imports.%Core.import_ref.4b2, @Optional(constants.%OptionalStorage.facet.f35) [concrete = constants.%Optional.HasValue.6c8]
// CHECK:STDOUT:   %HasValue.ref: %Optional.HasValue.type.ccd = name_ref HasValue, %.loc37_37.7 [concrete = constants.%Optional.HasValue.6c8]
// CHECK:STDOUT:   %Optional.HasValue.bound: <bound method> = bound_method %.loc37_37.6, %HasValue.ref
// CHECK:STDOUT:   %Optional.HasValue.specific_fn: <specific function> = specific_function %HasValue.ref, @Optional.HasValue(constants.%OptionalStorage.facet.f35) [concrete = constants.%Optional.HasValue.specific_fn.dd8]
// CHECK:STDOUT:   %bound_method.loc37_37.5: <bound method> = bound_method %.loc37_37.6, %Optional.HasValue.specific_fn
// CHECK:STDOUT:   %.loc37_37.8: %Optional.179 = acquire_value %.loc37_37.6
// CHECK:STDOUT:   %Optional.HasValue.call: init bool = call %bound_method.loc37_37.5(%.loc37_37.8)
// CHECK:STDOUT:   %.loc37_37.9: bool = value_of_initializer %Optional.HasValue.call
// CHECK:STDOUT:   %.loc37_37.10: bool = converted %Optional.HasValue.call, %.loc37_37.9
// CHECK:STDOUT:   if %.loc37_37.10 br !for.body else br !for.done
// CHECK:STDOUT:
// CHECK:STDOUT: !for.body:
// CHECK:STDOUT:   %.loc37_37.11: %Optional.Get.type.7eb = specific_constant imports.%Core.import_ref.52c, @Optional(constants.%OptionalStorage.facet.f35) [concrete = constants.%Optional.Get.787]
// CHECK:STDOUT:   %Get.ref: %Optional.Get.type.7eb = name_ref Get, %.loc37_37.11 [concrete = constants.%Optional.Get.787]
// CHECK:STDOUT:   %Optional.Get.bound: <bound method> = bound_method %.loc37_37.6, %Get.ref
// CHECK:STDOUT:   %Optional.Get.specific_fn: <specific function> = specific_function %Get.ref, @Optional.Get(constants.%OptionalStorage.facet.f35) [concrete = constants.%Optional.Get.specific_fn.01a]
// CHECK:STDOUT:   %bound_method.loc37_37.6: <bound method> = bound_method %.loc37_37.6, %Optional.Get.specific_fn
// CHECK:STDOUT:   %.loc37_37.12: ref %OverloadValueType = temporary_storage
// CHECK:STDOUT:   %.loc37_37.13: %Optional.179 = acquire_value %.loc37_37.6
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
// CHECK:STDOUT: specific @TestIterate(constants.%facet_value.637) {
// CHECK:STDOUT:   %R.patt.loc25_17.2 => constants.%R.patt
// CHECK:STDOUT:   %R.loc25_17.1 => constants.%facet_value.637
// CHECK:STDOUT:   %R.as_type.loc25_83.1 => constants.%Range
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.23f
// CHECK:STDOUT:   %r.patt.loc25_81.2 => constants.%r.patt.b9f
// CHECK:STDOUT:   %r.param_patt.loc25_76.2 => constants.%r.param_patt.466
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc25 => constants.%complete_type.357
// CHECK:STDOUT:   %Iterate.lookup_impl_witness => constants.%Iterate.impl_witness.e03
// CHECK:STDOUT:   %Iterate.facet.loc27_37.5 => constants.%Iterate.facet.a3e
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type => constants.%Iterate.WithSelf.NewCursor.type.b10
// CHECK:STDOUT:   %.loc27_37.20 => constants.%.f2a
// CHECK:STDOUT:   %impl.elem2.loc27_37.3 => constants.%T.as_type.as.Iterate.impl.NewCursor.60e
// CHECK:STDOUT:   %specific_impl_fn.loc27_37.4 => constants.%T.as_type.as.Iterate.impl.NewCursor.specific_fn
// CHECK:STDOUT:   %impl.elem1 => constants.%Destroy.facet.c69d
// CHECK:STDOUT:   %as_type => constants.%tuple.type.9ef
// CHECK:STDOUT:   %require_complete.1 => constants.%complete_type.e35
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type => constants.%Iterate.WithSelf.Next.type.42b
// CHECK:STDOUT:   %.loc27_37.21 => constants.%.96e
// CHECK:STDOUT:   %impl.elem3.loc27_37.2 => constants.%T.as_type.as.Iterate.impl.Next.23c
// CHECK:STDOUT:   %specific_impl_fn.loc27_37.5 => constants.%T.as_type.as.Iterate.impl.Next.specific_fn
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type => constants.%Destroy.WithSelf.SelfDestruct.type.106
// CHECK:STDOUT:   %.loc27_37.22 => constants.%.227
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
// CHECK:STDOUT:   %Optional.Get.type.c64: type = fn_type @Optional.Get, @Optional(%T.220) [symbolic]
// CHECK:STDOUT:   %Optional.Get.bcf: %Optional.Get.type.c64 = struct_value () [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.type.49e: type = fn_type @Optional.HasValue, @Optional(%T.220) [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.c20: %Optional.HasValue.type.49e = struct_value () [symbolic]
// CHECK:STDOUT:   %T.283: %facet_type.f48 = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.74b: type = facet_access_type %T.283 [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.28a: <witness> = lookup_impl_witness %T.283, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.16e: %Destroy.type = facet_value %T.as_type.74b, (%Destroy.lookup_impl_witness.28a) [symbolic]
// CHECK:STDOUT:   %MaybeUnformed.9b7: type = class_type @MaybeUnformed, @MaybeUnformed(%Destroy.facet.16e) [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.faa: type = class_type @DefaultOptionalStorage, @DefaultOptionalStorage(%T.283) [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.elem.06a: type = unbound_element_type %DefaultOptionalStorage.faa, bool [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.elem.9db: type = unbound_element_type %DefaultOptionalStorage.faa, %MaybeUnformed.9b7 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Get.type.2f2: type = fn_type @T.as_type.as.OptionalStorage.impl.Get, @T.as_type.as.OptionalStorage.impl(%T.283) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Get.884: %T.as_type.as.OptionalStorage.impl.Get.type.2f2 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Has.type.d28: type = fn_type @T.as_type.as.OptionalStorage.impl.Has, @T.as_type.as.OptionalStorage.impl(%T.283) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Has.f29: %T.as_type.as.OptionalStorage.impl.Has.type.d28 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Some.type.760: type = fn_type @T.as_type.as.OptionalStorage.impl.Some, @T.as_type.as.OptionalStorage.impl(%T.283) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Some.268: %T.as_type.as.OptionalStorage.impl.Some.type.760 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.None.type.925: type = fn_type @T.as_type.as.OptionalStorage.impl.None, @T.as_type.as.OptionalStorage.impl(%T.283) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.None.f42: %T.as_type.as.OptionalStorage.impl.None.type.925 = struct_value () [symbolic]
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
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.type.d4a: type = fn_type @T.as_type.as.Iterate.impl.Next, @T.as_type.as.Iterate.impl(%T.f45530.2) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.5a8: %T.as_type.as.Iterate.impl.Next.type.d4a = struct_value () [symbolic]
// CHECK:STDOUT:   %CppRangeForIterate.lookup_impl_witness.b581a8.2: <witness> = lookup_impl_witness %T.f45530.2, @CppRangeForIterate [symbolic]
// CHECK:STDOUT:   %impl.elem0.5f971c.2: type = impl_witness_access %CppRangeForIterate.lookup_impl_witness.b581a8.2, element0 [symbolic]
// CHECK:STDOUT:   %CppUnsafeDeref.lookup_impl_witness.87e543.2: <witness> = lookup_impl_witness %impl.elem0.5f971c.2, @CppUnsafeDeref [symbolic]
// CHECK:STDOUT:   %impl.elem0.493765.2: type = impl_witness_access %CppUnsafeDeref.lookup_impl_witness.87e543.2, element0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.type.8fb: type = fn_type @T.as_type.as.Iterate.impl.NewCursor, @T.as_type.as.Iterate.impl(%T.f45530.2) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.8a1: %T.as_type.as.Iterate.impl.NewCursor.type.8fb = struct_value () [symbolic]
// CHECK:STDOUT:   %impl.elem1.c2dc15.2: type = impl_witness_access %CppRangeForIterate.lookup_impl_witness.b581a8.2, element1 [symbolic]
// CHECK:STDOUT:   %tuple.type.1ba: type = tuple_type (%impl.elem0.5f971c.2, %impl.elem1.c2dc15.2) [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.812: <witness> = lookup_impl_witness %tuple.type.1ba, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.9cd: %Destroy.type = facet_value %tuple.type.1ba, (%Destroy.lookup_impl_witness.812) [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.35816c.2: <witness> = lookup_impl_witness %impl.elem0.493765.2, @Destroy [symbolic]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.167: <witness> = lookup_impl_witness %impl.elem0.493765.2, @Copy [symbolic]
// CHECK:STDOUT:   %facet_value.f93: %facet_type.f48 = facet_value %impl.elem0.493765.2, (%Destroy.lookup_impl_witness.35816c.2, %Copy.lookup_impl_witness.167) [symbolic]
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete]
// CHECK:STDOUT:   %f64.84c: type = class_type @Float, @Float(%int_64) [concrete]
// CHECK:STDOUT:   %pattern_type.39b: type = pattern_type %f64.84c [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc20 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.2: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.2: type = fn_type @Destroy.WithSelf.Op.loc20 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.2: %Destroy.WithSelf.Op.type.ef016f.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.2: type = fn_type @Destroy.WithSelf.SelfDestruct.loc20 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.2: %Destroy.WithSelf.SelfDestruct.type.fbceb5.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.f8f19d.2: <witness> = custom_witness (%Destroy.WithSelf.Op.403171.2, %Destroy.WithSelf.SubobjectDestroy.d01daf.2, %Destroy.WithSelf.SelfDestruct.db3fdb.2), @Destroy [concrete]
// CHECK:STDOUT:   %Float.as.Copy.impl.Op.type.98e: type = fn_type @Float.as.Copy.impl.Op, @Float.as.Copy.impl(%N.fe9) [symbolic]
// CHECK:STDOUT:   %Float.as.Copy.impl.Op.1a7: %Float.as.Copy.impl.Op.type.98e = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.e3b: <witness> = impl_witness imports.%Copy.impl_witness_table.75b, @Float.as.Copy.impl(%int_64) [concrete]
// CHECK:STDOUT:   %facet_value.198: %facet_type.f48 = facet_value %f64.84c, (%custom_witness.f8f19d.2, %Copy.impl_witness.e3b) [concrete]
// CHECK:STDOUT:   %Iterate.lookup_impl_witness.1d8: <witness> = lookup_impl_witness %.Self.500, @Iterate [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.294: %facet_type.f48 = impl_witness_access %Iterate.lookup_impl_witness.1d8, element0 [symbolic_self]
// CHECK:STDOUT:   %Iterate_where.type.d76: type = facet_type <@Iterate where %impl.elem0.294 = %facet_value.198> [concrete]
// CHECK:STDOUT:   %pattern_type.d54: type = pattern_type %Iterate_where.type.d76 [concrete]
// CHECK:STDOUT:   %R.patt: %pattern_type.d54 = symbolic_binding_pattern R, 0 [symbolic]
// CHECK:STDOUT:   %R: %Iterate_where.type.d76 = symbolic_binding R, 0 [symbolic]
// CHECK:STDOUT:   %R.as_type: type = facet_access_type %R [symbolic]
// CHECK:STDOUT:   %pattern_type.f0c: type = pattern_type %R.as_type [symbolic]
// CHECK:STDOUT:   %r.param_patt.9f4: %pattern_type.f0c = value_param_pattern [symbolic]
// CHECK:STDOUT:   %r.patt.48b: %pattern_type.f0c = wrapper_binding_pattern r, %r.param_patt.9f4 [symbolic]
// CHECK:STDOUT:   %i.patt: %pattern_type.39b = value_binding_pattern i [concrete]
// CHECK:STDOUT:   %Iterate.lookup_impl_witness.22a: <witness> = lookup_impl_witness %R, @Iterate [symbolic]
// CHECK:STDOUT:   %Iterate.facet.e88: %Iterate.type = facet_value %R.as_type, (%Iterate.lookup_impl_witness.22a) [symbolic]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.8c6: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.e88) [symbolic]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.03b: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.e88) [symbolic]
// CHECK:STDOUT:   %.10b: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.8c6, %Iterate.facet.e88 [symbolic]
// CHECK:STDOUT:   %impl.elem2.846: %.10b = impl_witness_access %Iterate.lookup_impl_witness.22a, element2 [symbolic]
// CHECK:STDOUT:   %impl.elem1.dbb: %Destroy.type = impl_witness_access %Iterate.lookup_impl_witness.22a, element1 [symbolic]
// CHECK:STDOUT:   %as_type.a72: type = facet_access_type %impl.elem1.dbb [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.145: <specific function> = specific_impl_function %impl.elem2.846, @Iterate.WithSelf.NewCursor(%Iterate.facet.e88) [symbolic]
// CHECK:STDOUT:   %.573: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.03b, %Iterate.facet.e88 [symbolic]
// CHECK:STDOUT:   %impl.elem3.31f: %.573 = impl_witness_access %Iterate.lookup_impl_witness.22a, element3 [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.c6e: type = class_type @DefaultOptionalStorage, @DefaultOptionalStorage(%facet_value.198) [concrete]
// CHECK:STDOUT:   %OptionalStorage.impl_witness.3ff: <witness> = impl_witness imports.%OptionalStorage.impl_witness_table.460, @T.as_type.as.OptionalStorage.impl(%facet_value.198) [concrete]
// CHECK:STDOUT:   %OptionalStorage.facet.1b5: %OptionalStorage.type = facet_value %f64.84c, (%OptionalStorage.impl_witness.3ff) [concrete]
// CHECK:STDOUT:   %Optional.74d: type = class_type @Optional, @Optional(%OptionalStorage.facet.1b5) [concrete]
// CHECK:STDOUT:   %pattern_type.14a: type = pattern_type %Optional.74d [concrete]
// CHECK:STDOUT:   %specific_impl_fn.863: <specific function> = specific_impl_function %impl.elem3.31f, @Iterate.WithSelf.Next(%Iterate.facet.e88) [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.type.0c6: type = fn_type @Optional.HasValue, @Optional(%OptionalStorage.facet.1b5) [concrete]
// CHECK:STDOUT:   %Optional.HasValue.32a: %Optional.HasValue.type.0c6 = struct_value () [concrete]
// CHECK:STDOUT:   %Optional.Get.type.253: type = fn_type @Optional.Get, @Optional(%OptionalStorage.facet.1b5) [concrete]
// CHECK:STDOUT:   %Optional.Get.642: %Optional.Get.type.253 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.facet.065: %Destroy.type = facet_value %f64.84c, (%custom_witness.f8f19d.2) [concrete]
// CHECK:STDOUT:   %MaybeUnformed.045: type = class_type @MaybeUnformed, @MaybeUnformed(%Destroy.facet.065) [concrete]
// CHECK:STDOUT:   %.442: type = maybe_unformed_type %f64.84c [concrete]
// CHECK:STDOUT:   %DefaultOptionalStorage.elem.465: type = unbound_element_type %DefaultOptionalStorage.c6e, %MaybeUnformed.045 [concrete]
// CHECK:STDOUT:   %DefaultOptionalStorage.elem.f8c: type = unbound_element_type %DefaultOptionalStorage.c6e, bool [concrete]
// CHECK:STDOUT:   %struct_type.value.has_value.d86: type = struct_type {.value: %MaybeUnformed.045, .has_value: bool} [concrete]
// CHECK:STDOUT:   %Optional.HasValue.specific_fn.03d: <specific function> = specific_function %Optional.HasValue.32a, @Optional.HasValue(%OptionalStorage.facet.1b5) [concrete]
// CHECK:STDOUT:   %Optional.Get.specific_fn.4b7: <specific function> = specific_function %Optional.Get.642, @Optional.Get(%OptionalStorage.facet.1b5) [concrete]
// CHECK:STDOUT:   %AddAssignWith.type.35e: type = facet_type <@AddAssignWith, @AddAssignWith(%f64.84c)> [concrete]
// CHECK:STDOUT:   %Float.as.AddAssignWith.impl.Op.type.db9: type = fn_type @Float.as.AddAssignWith.impl.Op, @Float.as.AddAssignWith.impl(%N.fe9) [symbolic]
// CHECK:STDOUT:   %Float.as.AddAssignWith.impl.Op.1bb: %Float.as.AddAssignWith.impl.Op.type.db9 = struct_value () [symbolic]
// CHECK:STDOUT:   %AddAssignWith.impl_witness.c16: <witness> = impl_witness imports.%AddAssignWith.impl_witness_table, @Float.as.AddAssignWith.impl(%int_64) [concrete]
// CHECK:STDOUT:   %Float.as.AddAssignWith.impl.Op.type.c22: type = fn_type @Float.as.AddAssignWith.impl.Op, @Float.as.AddAssignWith.impl(%int_64) [concrete]
// CHECK:STDOUT:   %Float.as.AddAssignWith.impl.Op.e48: %Float.as.AddAssignWith.impl.Op.type.c22 = struct_value () [concrete]
// CHECK:STDOUT:   %AddAssignWith.facet: %AddAssignWith.type.35e = facet_value %f64.84c, (%AddAssignWith.impl_witness.c16) [concrete]
// CHECK:STDOUT:   %AddAssignWith.WithSelf.Op.type.3d3: type = fn_type @AddAssignWith.WithSelf.Op, @AddAssignWith.WithSelf(%f64.84c, %AddAssignWith.facet) [concrete]
// CHECK:STDOUT:   %.57e: type = fn_type_with_self_type %AddAssignWith.WithSelf.Op.type.3d3, %AddAssignWith.facet [concrete]
// CHECK:STDOUT:   %Float.as.AddAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Float.as.AddAssignWith.impl.Op.e48, @Float.as.AddAssignWith.impl.Op(%int_64) [concrete]
// CHECK:STDOUT:   %pattern_type.dde: type = pattern_type %.442 [concrete]
// CHECK:STDOUT:   %self.param_patt.603: %pattern_type.dde = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.d22: %pattern_type.dde = wrapper_binding_pattern self, %self.param_patt.603 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.3: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc24_19.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.3: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.3 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.3: type = fn_type @Destroy.WithSelf.Op.loc24_19.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.3: %Destroy.WithSelf.Op.type.ef016f.3 = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.c18: type = pattern_type %MaybeUnformed.045 [concrete]
// CHECK:STDOUT:   %self.param_patt.5bd: %pattern_type.c18 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.26e: %pattern_type.c18 = wrapper_binding_pattern self, %self.param_patt.5bd [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.4: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc24_19.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.4: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.4 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.4: type = fn_type @Destroy.WithSelf.Op.loc24_19.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.4: %Destroy.WithSelf.Op.type.ef016f.4 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.4: type = fn_type @Destroy.WithSelf.SelfDestruct.loc24_19.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.4: %Destroy.WithSelf.SelfDestruct.type.fbceb5.4 = struct_value () [concrete]
// CHECK:STDOUT:   %self.param_patt.40d: %pattern_type.831 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.560: %pattern_type.831 = wrapper_binding_pattern self, %self.param_patt.40d [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.5: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc24_19.3 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.5: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.5 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.5: type = fn_type @Destroy.WithSelf.Op.loc24_19.3 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.5: %Destroy.WithSelf.Op.type.ef016f.5 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.5: type = fn_type @Destroy.WithSelf.SelfDestruct.loc24_19.3 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.5: %Destroy.WithSelf.SelfDestruct.type.fbceb5.5 = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.65a: type = pattern_type %struct_type.value.has_value.d86 [concrete]
// CHECK:STDOUT:   %self.param_patt.60d: %pattern_type.65a = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.d38: %pattern_type.65a = wrapper_binding_pattern self, %self.param_patt.60d [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.6: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc24_19.4 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.6: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.6 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.6: type = fn_type @Destroy.WithSelf.Op.loc24_19.4 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.6: %Destroy.WithSelf.Op.type.ef016f.6 = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.282: type = pattern_type %DefaultOptionalStorage.c6e [concrete]
// CHECK:STDOUT:   %self.param_patt.470: %pattern_type.282 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.e17: %pattern_type.282 = wrapper_binding_pattern self, %self.param_patt.470 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.7: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc24_19.5 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.7: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.7 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.7: type = fn_type @Destroy.WithSelf.Op.loc24_19.5 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.7: %Destroy.WithSelf.Op.type.ef016f.7 = struct_value () [concrete]
// CHECK:STDOUT:   %self.param_patt.5e1: %pattern_type.14a = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.327: %pattern_type.14a = wrapper_binding_pattern self, %self.param_patt.5e1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.8: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc24_19.6 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.8: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.8 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.8: type = fn_type @Destroy.WithSelf.Op.loc24_19.6 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.8: %Destroy.WithSelf.Op.type.ef016f.8 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.8: type = fn_type @Destroy.WithSelf.SelfDestruct.loc24_19.6 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.8: %Destroy.WithSelf.SelfDestruct.type.fbceb5.8 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.21f: <witness> = lookup_impl_witness %impl.elem1.dbb, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.7d4: type = fn_type @Destroy.WithSelf.SelfDestruct.1, @Destroy.WithSelf(%impl.elem1.dbb) [symbolic]
// CHECK:STDOUT:   %.a69e: type = fn_type_with_self_type %Destroy.WithSelf.SelfDestruct.type.7d4, %impl.elem1.dbb [symbolic]
// CHECK:STDOUT:   %impl.elem2.05e: %.a69e = impl_witness_access %Destroy.lookup_impl_witness.21f, element2 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.fe7: <specific function> = specific_impl_function %impl.elem2.05e, @Destroy.WithSelf.SelfDestruct.1(%impl.elem1.dbb) [symbolic]
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
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.9: type = fn_type @Destroy.WithSelf.Op.loc31 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.9: %Destroy.WithSelf.Op.type.ef016f.9 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.9: type = fn_type @Destroy.WithSelf.SelfDestruct.loc31 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.9: %Destroy.WithSelf.SelfDestruct.type.fbceb5.9 = struct_value () [concrete]
// CHECK:STDOUT:   %Iterator.SubobjectDestroy.type: type = fn_type @Iterator.SubobjectDestroy [concrete]
// CHECK:STDOUT:   %Iterator.SubobjectDestroy: %Iterator.SubobjectDestroy.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b76: <witness> = custom_witness (%Destroy.WithSelf.Op.403171.9, %Iterator.SubobjectDestroy, %Destroy.WithSelf.SelfDestruct.db3fdb.9), @Destroy [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.ac7e09.1: type = fn_type @Iterator.Op.1 [concrete]
// CHECK:STDOUT:   %Iterator.Op.583fda.1: %Iterator.Op.type.ac7e09.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.9e0: <witness> = custom_witness (%f64.84c, %Iterator.Op.583fda.1), @CppUnsafeDeref [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.ac7e09.2: type = fn_type @Iterator.Op.2 [concrete]
// CHECK:STDOUT:   %Iterator.Op.583fda.2: %Iterator.Op.type.ac7e09.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.393: <witness> = custom_witness (%Iterator.Op.583fda.2), @Inc [concrete]
// CHECK:STDOUT:   %Equal.type: type = fn_type @Equal [concrete]
// CHECK:STDOUT:   %Equal: %Equal.type = struct_value () [concrete]
// CHECK:STDOUT:   %NotEqual.type: type = fn_type @NotEqual [concrete]
// CHECK:STDOUT:   %NotEqual: %NotEqual.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.eb7: <witness> = custom_witness (%Equal, %NotEqual), @EqWith, @EqWith(%Iterator) [concrete]
// CHECK:STDOUT:   %facet_value.065: %CppRange_where.type.d682d6.1 = facet_value %ConstRange, (%custom_witness.5d5, %custom_witness.b76, %custom_witness.9e0, %custom_witness.393, %custom_witness.eb7, %custom_witness.f8f19d.2, %Copy.impl_witness.e3b, %custom_witness.b76) [concrete]
// CHECK:STDOUT:   %tuple.type.005: type = tuple_type (%Iterator, %Iterator) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.9: type = fn_type @Destroy.WithSelf.SubobjectDestroy.3 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.9: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.9 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.10: type = fn_type @Destroy.WithSelf.Op.3 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.10: %Destroy.WithSelf.Op.type.ef016f.10 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.10: type = fn_type @Destroy.WithSelf.SelfDestruct.3 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.10: %Destroy.WithSelf.SelfDestruct.type.fbceb5.10 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.f8f19d.9: <witness> = custom_witness (%Destroy.WithSelf.Op.403171.10, %Destroy.WithSelf.SubobjectDestroy.d01daf.9, %Destroy.WithSelf.SelfDestruct.db3fdb.10), @Destroy [concrete]
// CHECK:STDOUT:   %Destroy.facet.362: %Destroy.type = facet_value %tuple.type.005, (%custom_witness.f8f19d.9) [concrete]
// CHECK:STDOUT:   %Iterate.impl_witness.1e9: <witness> = impl_witness imports.%Iterate.impl_witness_table.121, @T.as_type.as.Iterate.impl(%facet_value.065) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.type.2dd: type = fn_type @T.as_type.as.Iterate.impl.NewCursor, @T.as_type.as.Iterate.impl(%facet_value.065) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.d46: %T.as_type.as.Iterate.impl.NewCursor.type.2dd = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.type.b70: type = fn_type @T.as_type.as.Iterate.impl.Next, @T.as_type.as.Iterate.impl(%facet_value.065) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.e9d: %T.as_type.as.Iterate.impl.Next.type.b70 = struct_value () [concrete]
// CHECK:STDOUT:   %Iterate.facet.795: %Iterate.type = facet_value %ConstRange, (%Iterate.impl_witness.1e9) [concrete]
// CHECK:STDOUT:   %facet_value.f70: %Iterate_where.type.d76 = facet_value %ConstRange, (%Iterate.impl_witness.1e9) [concrete]
// CHECK:STDOUT:   %r.param_patt.a73: %pattern_type.025 = value_param_pattern [concrete]
// CHECK:STDOUT:   %r.patt.0b5: %pattern_type.025 = wrapper_binding_pattern r, %r.param_patt.a73 [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.db9: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.795) [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.8a3: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.795) [concrete]
// CHECK:STDOUT:   %.c3e: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.db9, %Iterate.facet.795 [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.specific_fn: <specific function> = specific_function %T.as_type.as.Iterate.impl.NewCursor.d46, @T.as_type.as.Iterate.impl.NewCursor(%facet_value.065) [concrete]
// CHECK:STDOUT:   %.9bc: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.8a3, %Iterate.facet.795 [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.specific_fn: <specific function> = specific_function %T.as_type.as.Iterate.impl.Next.e9d, @T.as_type.as.Iterate.impl.Next(%facet_value.065) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.dd4: type = fn_type @Destroy.WithSelf.SelfDestruct.1, @Destroy.WithSelf(%Destroy.facet.362) [concrete]
// CHECK:STDOUT:   %.38e: type = fn_type_with_self_type %Destroy.WithSelf.SelfDestruct.type.dd4, %Destroy.facet.362 [concrete]
// CHECK:STDOUT:   %complete_type.320: <witness> = complete_type_witness %tuple.type.005 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.4b2: @Optional.%Optional.HasValue.type (%Optional.HasValue.type.49e) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.HasValue (constants.%Optional.HasValue.c20)]
// CHECK:STDOUT:   %Core.import_ref.52c: @Optional.%Optional.Get.type (%Optional.Get.type.c64) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.Get (constants.%Optional.Get.bcf)]
// CHECK:STDOUT:   %Core.import_ref.ace: type = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%DefaultOptionalStorage (constants.%DefaultOptionalStorage.faa)]
// CHECK:STDOUT:   %Core.import_ref.ef42: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.None.type (%T.as_type.as.OptionalStorage.impl.None.type.925) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.None (constants.%T.as_type.as.OptionalStorage.impl.None.f42)]
// CHECK:STDOUT:   %Core.import_ref.a20: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Some.type (%T.as_type.as.OptionalStorage.impl.Some.type.760) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Some (constants.%T.as_type.as.OptionalStorage.impl.Some.268)]
// CHECK:STDOUT:   %Core.import_ref.af7: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Has.type (%T.as_type.as.OptionalStorage.impl.Has.type.d28) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Has (constants.%T.as_type.as.OptionalStorage.impl.Has.f29)]
// CHECK:STDOUT:   %Core.import_ref.dd4: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Get.type (%T.as_type.as.OptionalStorage.impl.Get.type.2f2) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Get (constants.%T.as_type.as.OptionalStorage.impl.Get.884)]
// CHECK:STDOUT:   %Core.import_ref.554 = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], unloaded
// CHECK:STDOUT:   %OptionalStorage.impl_witness_table.460 = impl_witness_table (%Core.import_ref.ace, %Core.import_ref.ef42, %Core.import_ref.a20, %Core.import_ref.af7, %Core.import_ref.dd4, %Core.import_ref.554), @T.as_type.as.OptionalStorage.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.1d708: @DefaultOptionalStorage.%DefaultOptionalStorage.elem.1 (%DefaultOptionalStorage.elem.9db) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [concrete = %field_decl.5f0]
// CHECK:STDOUT:   %Core.import_ref.929: @DefaultOptionalStorage.%DefaultOptionalStorage.elem.2 (%DefaultOptionalStorage.elem.06a) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [concrete = %field_decl.6a8]
// CHECK:STDOUT:   %Core.import_ref.eef: %facet_type.f48 = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.Iterate.impl.%facet_value (constants.%facet_value.f93)]
// CHECK:STDOUT:   %Core.import_ref.fcc: %Destroy.type = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.Iterate.impl.%Destroy.facet (constants.%Destroy.facet.9cd)]
// CHECK:STDOUT:   %Core.import_ref.946: @T.as_type.as.Iterate.impl.%T.as_type.as.Iterate.impl.NewCursor.type (%T.as_type.as.Iterate.impl.NewCursor.type.8fb) = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.Iterate.impl.%T.as_type.as.Iterate.impl.NewCursor (constants.%T.as_type.as.Iterate.impl.NewCursor.8a1)]
// CHECK:STDOUT:   %Core.import_ref.cfb: @T.as_type.as.Iterate.impl.%T.as_type.as.Iterate.impl.Next.type (%T.as_type.as.Iterate.impl.Next.type.d4a) = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.Iterate.impl.%T.as_type.as.Iterate.impl.Next (constants.%T.as_type.as.Iterate.impl.Next.5a8)]
// CHECK:STDOUT:   %Iterate.impl_witness_table.121 = impl_witness_table (%Core.import_ref.eef, %Core.import_ref.fcc, %Core.import_ref.946, %Core.import_ref.cfb), @T.as_type.as.Iterate.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.176: @Float.as.Copy.impl.%Float.as.Copy.impl.Op.type (%Float.as.Copy.impl.Op.type.98e) = import_ref Core//prelude/types/float, loc{{\d+_\d+}}, loaded [symbolic = @Float.as.Copy.impl.%Float.as.Copy.impl.Op (constants.%Float.as.Copy.impl.Op.1a7)]
// CHECK:STDOUT:   %Copy.impl_witness_table.75b = impl_witness_table (%Core.import_ref.176), @Float.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.ee5: @Float.as.AddAssignWith.impl.%Float.as.AddAssignWith.impl.Op.type (%Float.as.AddAssignWith.impl.Op.type.db9) = import_ref Core//prelude/types/float, loc{{\d+_\d+}}, loaded [symbolic = @Float.as.AddAssignWith.impl.%Float.as.AddAssignWith.impl.Op (constants.%Float.as.AddAssignWith.impl.Op.1bb)]
// CHECK:STDOUT:   %AddAssignWith.impl_witness_table = impl_witness_table (%Core.import_ref.ee5), @Float.as.AddAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.b0d = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %field_decl.6a8: @DefaultOptionalStorage.%DefaultOptionalStorage.elem.2 (%DefaultOptionalStorage.elem.06a) = field_decl has_value, element1, %Core.import_ref.b0d in [concrete] {}
// CHECK:STDOUT:   %Core.import_ref.87f = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %field_decl.5f0: @DefaultOptionalStorage.%DefaultOptionalStorage.elem.1 (%DefaultOptionalStorage.elem.9db) = field_decl value, element0, %Core.import_ref.87f in [concrete] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generated {
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.2: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.3 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc24_19.1 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.3] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.dde = ref_param_pattern [concrete = constants.%self.param_patt.603]
// CHECK:STDOUT:     %self.patt: %pattern_type.dde = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.d22]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %.442 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %.442 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.2: %Destroy.WithSelf.Op.type.ef016f.3 = fn_decl @Destroy.WithSelf.Op.loc24_19.1 [concrete = constants.%Destroy.WithSelf.Op.403171.3] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.dde = ref_param_pattern [concrete = constants.%self.param_patt.603]
// CHECK:STDOUT:     %self.patt: %pattern_type.dde = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.d22]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %.442 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %.442 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.3: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.4 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc24_19.2 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.4] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.c18 = ref_param_pattern [concrete = constants.%self.param_patt.5bd]
// CHECK:STDOUT:     %self.patt: %pattern_type.c18 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.26e]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %MaybeUnformed.045 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %MaybeUnformed.045 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.3: %Destroy.WithSelf.Op.type.ef016f.4 = fn_decl @Destroy.WithSelf.Op.loc24_19.2 [concrete = constants.%Destroy.WithSelf.Op.403171.4] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.c18 = ref_param_pattern [concrete = constants.%self.param_patt.5bd]
// CHECK:STDOUT:     %self.patt: %pattern_type.c18 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.26e]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %MaybeUnformed.045 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %MaybeUnformed.045 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.4: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.5 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc24_19.3 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.5] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.831 = ref_param_pattern [concrete = constants.%self.param_patt.40d]
// CHECK:STDOUT:     %self.patt: %pattern_type.831 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.560]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref bool = ref_param call_param0
// CHECK:STDOUT:     %self: ref bool = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.4: %Destroy.WithSelf.Op.type.ef016f.5 = fn_decl @Destroy.WithSelf.Op.loc24_19.3 [concrete = constants.%Destroy.WithSelf.Op.403171.5] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.831 = ref_param_pattern [concrete = constants.%self.param_patt.40d]
// CHECK:STDOUT:     %self.patt: %pattern_type.831 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.560]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref bool = ref_param call_param0
// CHECK:STDOUT:     %self: ref bool = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.5: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.6 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc24_19.4 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.6] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.65a = ref_param_pattern [concrete = constants.%self.param_patt.60d]
// CHECK:STDOUT:     %self.patt: %pattern_type.65a = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.d38]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %struct_type.value.has_value.d86 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %struct_type.value.has_value.d86 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.5: %Destroy.WithSelf.Op.type.ef016f.6 = fn_decl @Destroy.WithSelf.Op.loc24_19.4 [concrete = constants.%Destroy.WithSelf.Op.403171.6] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.65a = ref_param_pattern [concrete = constants.%self.param_patt.60d]
// CHECK:STDOUT:     %self.patt: %pattern_type.65a = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.d38]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %struct_type.value.has_value.d86 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %struct_type.value.has_value.d86 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.6: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.7 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc24_19.5 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.7] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.282 = ref_param_pattern [concrete = constants.%self.param_patt.470]
// CHECK:STDOUT:     %self.patt: %pattern_type.282 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.e17]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %DefaultOptionalStorage.c6e = ref_param call_param0
// CHECK:STDOUT:     %self: ref %DefaultOptionalStorage.c6e = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.6: %Destroy.WithSelf.Op.type.ef016f.7 = fn_decl @Destroy.WithSelf.Op.loc24_19.5 [concrete = constants.%Destroy.WithSelf.Op.403171.7] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.282 = ref_param_pattern [concrete = constants.%self.param_patt.470]
// CHECK:STDOUT:     %self.patt: %pattern_type.282 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.e17]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %DefaultOptionalStorage.c6e = ref_param call_param0
// CHECK:STDOUT:     %self: ref %DefaultOptionalStorage.c6e = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.7: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.8 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc24_19.6 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.8] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.14a = ref_param_pattern [concrete = constants.%self.param_patt.5e1]
// CHECK:STDOUT:     %self.patt: %pattern_type.14a = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.327]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %Optional.74d = ref_param call_param0
// CHECK:STDOUT:     %self: ref %Optional.74d = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.7: %Destroy.WithSelf.Op.type.ef016f.8 = fn_decl @Destroy.WithSelf.Op.loc24_19.6 [concrete = constants.%Destroy.WithSelf.Op.403171.8] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.14a = ref_param_pattern [concrete = constants.%self.param_patt.5e1]
// CHECK:STDOUT:     %self.patt: %pattern_type.14a = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.327]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %Optional.74d = ref_param call_param0
// CHECK:STDOUT:     %self: ref %Optional.74d = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @TestIterate(%R.loc20_17.2: %Iterate_where.type.d76) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Iterate.lookup_impl_witness: <witness> = lookup_impl_witness %R.loc20_17.1, @Iterate [symbolic = %Iterate.lookup_impl_witness (constants.%Iterate.lookup_impl_witness.22a)]
// CHECK:STDOUT:   %Iterate.facet.loc24_19.5: %Iterate.type = facet_value %R.as_type.loc20_61.1, (%Iterate.lookup_impl_witness) [symbolic = %Iterate.facet.loc24_19.5 (constants.%Iterate.facet.e88)]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.loc24_19.5) [symbolic = %Iterate.WithSelf.NewCursor.type (constants.%Iterate.WithSelf.NewCursor.type.8c6)]
// CHECK:STDOUT:   %.loc24_19.19: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type, %Iterate.facet.loc24_19.5 [symbolic = %.loc24_19.19 (constants.%.10b)]
// CHECK:STDOUT:   %impl.elem2.loc24_19.3: @TestIterate.%.loc24_19.19 (%.10b) = impl_witness_access %Iterate.lookup_impl_witness, element2 [symbolic = %impl.elem2.loc24_19.3 (constants.%impl.elem2.846)]
// CHECK:STDOUT:   %specific_impl_fn.loc24_19.4: <specific function> = specific_impl_function %impl.elem2.loc24_19.3, @Iterate.WithSelf.NewCursor(%Iterate.facet.loc24_19.5) [symbolic = %specific_impl_fn.loc24_19.4 (constants.%specific_impl_fn.145)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.loc24_19.5) [symbolic = %Iterate.WithSelf.Next.type (constants.%Iterate.WithSelf.Next.type.03b)]
// CHECK:STDOUT:   %.loc24_19.20: type = fn_type_with_self_type %Iterate.WithSelf.Next.type, %Iterate.facet.loc24_19.5 [symbolic = %.loc24_19.20 (constants.%.573)]
// CHECK:STDOUT:   %impl.elem3.loc24_19.2: @TestIterate.%.loc24_19.20 (%.573) = impl_witness_access %Iterate.lookup_impl_witness, element3 [symbolic = %impl.elem3.loc24_19.2 (constants.%impl.elem3.31f)]
// CHECK:STDOUT:   %specific_impl_fn.loc24_19.5: <specific function> = specific_impl_function %impl.elem3.loc24_19.2, @Iterate.WithSelf.Next(%Iterate.facet.loc24_19.5) [symbolic = %specific_impl_fn.loc24_19.5 (constants.%specific_impl_fn.863)]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type: type = fn_type @Destroy.WithSelf.SelfDestruct.1, @Destroy.WithSelf(%impl.elem1) [symbolic = %Destroy.WithSelf.SelfDestruct.type (constants.%Destroy.WithSelf.SelfDestruct.type.7d4)]
// CHECK:STDOUT:   %.loc24_19.21: type = fn_type_with_self_type %Destroy.WithSelf.SelfDestruct.type, %impl.elem1 [symbolic = %.loc24_19.21 (constants.%.a69e)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %impl.elem1, @Destroy [symbolic = %Destroy.lookup_impl_witness (constants.%Destroy.lookup_impl_witness.21f)]
// CHECK:STDOUT:   %impl.elem2.loc24_19.4: @TestIterate.%.loc24_19.21 (%.a69e) = impl_witness_access %Destroy.lookup_impl_witness, element2 [symbolic = %impl.elem2.loc24_19.4 (constants.%impl.elem2.05e)]
// CHECK:STDOUT:   %specific_impl_fn.loc24_19.6: <specific function> = specific_impl_function %impl.elem2.loc24_19.4, @Destroy.WithSelf.SelfDestruct.1(%impl.elem1) [symbolic = %specific_impl_fn.loc24_19.6 (constants.%specific_impl_fn.fe7)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%r.param: @TestIterate.%R.as_type.loc20_61.1 (%R.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %i.patt: %pattern_type.39b = value_binding_pattern i [concrete = constants.%i.patt]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %r.ref: @TestIterate.%R.as_type.loc20_61.1 (%R.as_type) = name_ref r, %r
// CHECK:STDOUT:     %impl.elem2.loc24_19.1: @TestIterate.%.loc24_19.19 (%.10b) = impl_witness_access constants.%Iterate.lookup_impl_witness.22a, element2 [symbolic = %impl.elem2.loc24_19.3 (constants.%impl.elem2.846)]
// CHECK:STDOUT:     %bound_method.loc24_19.1: <bound method> = bound_method %r.ref, %impl.elem2.loc24_19.1
// CHECK:STDOUT:     %Iterate.facet.loc24_19.1: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.22a) [symbolic = %Iterate.facet.loc24_19.5 (constants.%Iterate.facet.e88)]
// CHECK:STDOUT:     %.loc24_19.1: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc24_19.1 [symbolic = %Iterate.facet.loc24_19.5 (constants.%Iterate.facet.e88)]
// CHECK:STDOUT:     %Iterate.facet.loc24_19.2: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.22a) [symbolic = %Iterate.facet.loc24_19.5 (constants.%Iterate.facet.e88)]
// CHECK:STDOUT:     %.loc24_19.2: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc24_19.2 [symbolic = %Iterate.facet.loc24_19.5 (constants.%Iterate.facet.e88)]
// CHECK:STDOUT:     %specific_impl_fn.loc24_19.1: <specific function> = specific_impl_function %impl.elem2.loc24_19.1, @Iterate.WithSelf.NewCursor(constants.%Iterate.facet.e88) [symbolic = %specific_impl_fn.loc24_19.4 (constants.%specific_impl_fn.145)]
// CHECK:STDOUT:     %bound_method.loc24_19.2: <bound method> = bound_method %r.ref, %specific_impl_fn.loc24_19.1
// CHECK:STDOUT:     %var: ref @TestIterate.%as_type (%as_type.a72) = var_storage invalid
// CHECK:STDOUT:     %Iterate.WithSelf.NewCursor.call: init @TestIterate.%as_type (%as_type.a72) to %var = call %bound_method.loc24_19.2(%r.ref)
// CHECK:STDOUT:     assign %var, %Iterate.WithSelf.NewCursor.call
// CHECK:STDOUT:     br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.next:
// CHECK:STDOUT:     %impl.elem3.loc24_19.1: @TestIterate.%.loc24_19.20 (%.573) = impl_witness_access constants.%Iterate.lookup_impl_witness.22a, element3 [symbolic = %impl.elem3.loc24_19.2 (constants.%impl.elem3.31f)]
// CHECK:STDOUT:     %bound_method.loc24_19.3: <bound method> = bound_method %r.ref, %impl.elem3.loc24_19.1
// CHECK:STDOUT:     %Iterate.facet.loc24_19.3: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.22a) [symbolic = %Iterate.facet.loc24_19.5 (constants.%Iterate.facet.e88)]
// CHECK:STDOUT:     %.loc24_19.3: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc24_19.3 [symbolic = %Iterate.facet.loc24_19.5 (constants.%Iterate.facet.e88)]
// CHECK:STDOUT:     %Iterate.facet.loc24_19.4: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.22a) [symbolic = %Iterate.facet.loc24_19.5 (constants.%Iterate.facet.e88)]
// CHECK:STDOUT:     %.loc24_19.4: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc24_19.4 [symbolic = %Iterate.facet.loc24_19.5 (constants.%Iterate.facet.e88)]
// CHECK:STDOUT:     %.loc24_19.5: %Destroy.type = converted constants.%as_type.a72, constants.%impl.elem1.dbb [symbolic = %impl.elem1 (constants.%impl.elem1.dbb)]
// CHECK:STDOUT:     %.loc24_19.6: %Destroy.type = converted constants.%as_type.a72, constants.%impl.elem1.dbb [symbolic = %impl.elem1 (constants.%impl.elem1.dbb)]
// CHECK:STDOUT:     %specific_impl_fn.loc24_19.2: <specific function> = specific_impl_function %impl.elem3.loc24_19.1, @Iterate.WithSelf.Next(constants.%Iterate.facet.e88) [symbolic = %specific_impl_fn.loc24_19.5 (constants.%specific_impl_fn.863)]
// CHECK:STDOUT:     %bound_method.loc24_19.4: <bound method> = bound_method %r.ref, %specific_impl_fn.loc24_19.2
// CHECK:STDOUT:     %.loc24_19.7: ref %Optional.74d = temporary_storage
// CHECK:STDOUT:     %Iterate.WithSelf.Next.call: init %Optional.74d to %.loc24_19.7 = call %bound_method.loc24_19.4(%r.ref, %var)
// CHECK:STDOUT:     %.loc24_19.8: ref %Optional.74d = temporary %.loc24_19.7, %Iterate.WithSelf.Next.call
// CHECK:STDOUT:     %.loc24_19.9: %Optional.HasValue.type.0c6 = specific_constant imports.%Core.import_ref.4b2, @Optional(constants.%OptionalStorage.facet.1b5) [concrete = constants.%Optional.HasValue.32a]
// CHECK:STDOUT:     %HasValue.ref: %Optional.HasValue.type.0c6 = name_ref HasValue, %.loc24_19.9 [concrete = constants.%Optional.HasValue.32a]
// CHECK:STDOUT:     %Optional.HasValue.bound: <bound method> = bound_method %.loc24_19.8, %HasValue.ref
// CHECK:STDOUT:     %Optional.HasValue.specific_fn: <specific function> = specific_function %HasValue.ref, @Optional.HasValue(constants.%OptionalStorage.facet.1b5) [concrete = constants.%Optional.HasValue.specific_fn.03d]
// CHECK:STDOUT:     %bound_method.loc24_19.5: <bound method> = bound_method %.loc24_19.8, %Optional.HasValue.specific_fn
// CHECK:STDOUT:     %.loc24_19.10: %Optional.74d = acquire_value %.loc24_19.8
// CHECK:STDOUT:     %Optional.HasValue.call: init bool = call %bound_method.loc24_19.5(%.loc24_19.10)
// CHECK:STDOUT:     %.loc24_19.11: bool = value_of_initializer %Optional.HasValue.call
// CHECK:STDOUT:     %.loc24_19.12: bool = converted %Optional.HasValue.call, %.loc24_19.11
// CHECK:STDOUT:     if %.loc24_19.12 br !for.body else br !for.done
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.body:
// CHECK:STDOUT:     %.loc24_19.13: %Optional.Get.type.253 = specific_constant imports.%Core.import_ref.52c, @Optional(constants.%OptionalStorage.facet.1b5) [concrete = constants.%Optional.Get.642]
// CHECK:STDOUT:     %Get.ref: %Optional.Get.type.253 = name_ref Get, %.loc24_19.13 [concrete = constants.%Optional.Get.642]
// CHECK:STDOUT:     %Optional.Get.bound: <bound method> = bound_method %.loc24_19.8, %Get.ref
// CHECK:STDOUT:     %Optional.Get.specific_fn: <specific function> = specific_function %Get.ref, @Optional.Get(constants.%OptionalStorage.facet.1b5) [concrete = constants.%Optional.Get.specific_fn.4b7]
// CHECK:STDOUT:     %bound_method.loc24_19.6: <bound method> = bound_method %.loc24_19.8, %Optional.Get.specific_fn
// CHECK:STDOUT:     %.loc24_19.14: %Optional.74d = acquire_value %.loc24_19.8
// CHECK:STDOUT:     %Optional.Get.call: init %f64.84c = call %bound_method.loc24_19.6(%.loc24_19.14)
// CHECK:STDOUT:     %.loc24_19.15: %f64.84c = value_of_initializer %Optional.Get.call
// CHECK:STDOUT:     %.loc24_19.16: %f64.84c = converted %Optional.Get.call, %.loc24_19.15
// CHECK:STDOUT:     %f64.loc24: type = type_literal constants.%f64.84c [concrete = constants.%f64.84c]
// CHECK:STDOUT:     %i: %f64.84c = wrapper_binding i, %.loc24_19.16
// CHECK:STDOUT:     %sum.ref: ref %f64.84c = name_ref sum, %sum
// CHECK:STDOUT:     %i.ref: %f64.84c = name_ref i, %i
// CHECK:STDOUT:     %impl.elem0.loc25: %.57e = impl_witness_access constants.%AddAssignWith.impl_witness.c16, element0 [concrete = constants.%Float.as.AddAssignWith.impl.Op.e48]
// CHECK:STDOUT:     %bound_method.loc25_9.1: <bound method> = bound_method %sum.ref, %impl.elem0.loc25
// CHECK:STDOUT:     %specific_fn.loc25: <specific function> = specific_function %impl.elem0.loc25, @Float.as.AddAssignWith.impl.Op(constants.%int_64) [concrete = constants.%Float.as.AddAssignWith.impl.Op.specific_fn]
// CHECK:STDOUT:     %bound_method.loc25_9.2: <bound method> = bound_method %sum.ref, %specific_fn.loc25
// CHECK:STDOUT:     %Float.as.AddAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc25_9.2(%sum.ref, %i.ref)
// CHECK:STDOUT:     %Destroy.WithSelf.SelfDestruct.bound.loc24_19.1: <bound method> = bound_method %.loc24_19.8, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.8
// CHECK:STDOUT:     %Destroy.WithSelf.SelfDestruct.call.loc24_19.1: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc24_19.1(%.loc24_19.8)
// CHECK:STDOUT:     br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.done:
// CHECK:STDOUT:     %Destroy.WithSelf.SelfDestruct.bound.loc24_19.2: <bound method> = bound_method %.loc24_19.8, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.8
// CHECK:STDOUT:     %Destroy.WithSelf.SelfDestruct.call.loc24_19.2: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc24_19.2(%.loc24_19.8)
// CHECK:STDOUT:     %impl.elem2.loc24_19.2: @TestIterate.%.loc24_19.21 (%.a69e) = impl_witness_access constants.%Destroy.lookup_impl_witness.21f, element2 [symbolic = %impl.elem2.loc24_19.4 (constants.%impl.elem2.05e)]
// CHECK:STDOUT:     %bound_method.loc24_19.7: <bound method> = bound_method %var, %impl.elem2.loc24_19.2
// CHECK:STDOUT:     %.loc24_19.17: %Destroy.type = converted constants.%as_type.a72, constants.%impl.elem1.dbb [symbolic = %impl.elem1 (constants.%impl.elem1.dbb)]
// CHECK:STDOUT:     %.loc24_19.18: %Destroy.type = converted constants.%as_type.a72, constants.%impl.elem1.dbb [symbolic = %impl.elem1 (constants.%impl.elem1.dbb)]
// CHECK:STDOUT:     %specific_impl_fn.loc24_19.3: <specific function> = specific_impl_function %impl.elem2.loc24_19.2, @Destroy.WithSelf.SelfDestruct.1(constants.%impl.elem1.dbb) [symbolic = %specific_impl_fn.loc24_19.6 (constants.%specific_impl_fn.fe7)]
// CHECK:STDOUT:     %bound_method.loc24_19.8: <bound method> = bound_method %var, %specific_impl_fn.loc24_19.3
// CHECK:STDOUT:     %Destroy.WithSelf.SelfDestruct.call.loc24_19.3: init %empty_tuple.type = call %bound_method.loc24_19.8(%var)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc24_19.1(%self.param: ref %.442) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc24_19.1(%self.param: ref %.442) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc24_19.1(%self.param: ref %.442) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.2(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.2(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc24_19.2(%self.param: ref %MaybeUnformed.045) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc24_19.2(%self.param: ref %MaybeUnformed.045) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc24_19.2(%self.param: ref %MaybeUnformed.045) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.3(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.3(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc24_19.3(%self.param: ref bool) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc24_19.3(%self.param: ref bool) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc24_19.3(%self.param: ref bool) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.4(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.4(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc24_19.4(%self.param: ref %struct_type.value.has_value.d86) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc24_19.1: ref bool = struct_access %self.param, element1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc24_19.1: <bound method> = bound_method %.loc24_19.1, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.5
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc24_19.1: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc24_19.1(%.loc24_19.1)
// CHECK:STDOUT:   %.loc24_19.2: ref %MaybeUnformed.045 = struct_access %self.param, element0
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc24_19.2: <bound method> = bound_method %.loc24_19.2, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.4
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc24_19.2: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc24_19.2(%.loc24_19.2)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc24_19.4(%self.param: ref %struct_type.value.has_value.d86) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc24_19.4(%self.param: ref %struct_type.value.has_value.d86) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.5(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.5(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc24_19.5(%self.param: ref %DefaultOptionalStorage.c6e) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %has_value.ref: %DefaultOptionalStorage.elem.f8c = name_ref has_value, imports.%Core.import_ref.929 [concrete = imports.%field_decl.6a8]
// CHECK:STDOUT:   %.loc24_19.1: ref bool = class_element_access %self.param, element1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc24_19.1: <bound method> = bound_method %.loc24_19.1, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.5
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc24_19.1: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc24_19.1(%.loc24_19.1)
// CHECK:STDOUT:   %value.ref: %DefaultOptionalStorage.elem.465 = name_ref value, imports.%Core.import_ref.1d708 [concrete = imports.%field_decl.5f0]
// CHECK:STDOUT:   %.loc24_19.2: ref %MaybeUnformed.045 = class_element_access %self.param, element0
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc24_19.2: <bound method> = bound_method %.loc24_19.2, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.4
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc24_19.2: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc24_19.2(%.loc24_19.2)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc24_19.5(%self.param: ref %DefaultOptionalStorage.c6e) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc24_19.5(%self.param: ref %DefaultOptionalStorage.c6e) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.6(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.6(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc24_19.6(%self.param: ref %Optional.74d) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc24_19.6(%self.param: ref %Optional.74d) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc24_19.6(%self.param: ref %Optional.74d) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.7(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.7(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestRangeFor(%c.param: %ConstRange) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %i.patt: %pattern_type.39b = value_binding_pattern i [concrete = constants.%i.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c.ref: %ConstRange = name_ref c, %c
// CHECK:STDOUT:   %impl.elem2: %.c3e = impl_witness_access constants.%Iterate.impl_witness.1e9, element2 [concrete = constants.%T.as_type.as.Iterate.impl.NewCursor.d46]
// CHECK:STDOUT:   %bound_method.loc38_19.1: <bound method> = bound_method %c.ref, %impl.elem2
// CHECK:STDOUT:   %facet_value.loc38_19.1: %CppRange_where.type.d682d6.1 = facet_value constants.%ConstRange, (constants.%custom_witness.5d5, constants.%custom_witness.b76, constants.%custom_witness.9e0, constants.%custom_witness.393, constants.%custom_witness.eb7, constants.%custom_witness.f8f19d.2, constants.%Copy.impl_witness.e3b, constants.%custom_witness.b76) [concrete = constants.%facet_value.065]
// CHECK:STDOUT:   %.loc38_19.1: %CppRange_where.type.d682d6.1 = converted constants.%ConstRange, %facet_value.loc38_19.1 [concrete = constants.%facet_value.065]
// CHECK:STDOUT:   %facet_value.loc38_19.2: %CppRange_where.type.d682d6.1 = facet_value constants.%ConstRange, (constants.%custom_witness.5d5, constants.%custom_witness.b76, constants.%custom_witness.9e0, constants.%custom_witness.393, constants.%custom_witness.eb7, constants.%custom_witness.f8f19d.2, constants.%Copy.impl_witness.e3b, constants.%custom_witness.b76) [concrete = constants.%facet_value.065]
// CHECK:STDOUT:   %.loc38_19.2: %CppRange_where.type.d682d6.1 = converted constants.%ConstRange, %facet_value.loc38_19.2 [concrete = constants.%facet_value.065]
// CHECK:STDOUT:   %specific_fn.loc38_19.1: <specific function> = specific_function %impl.elem2, @T.as_type.as.Iterate.impl.NewCursor(constants.%facet_value.065) [concrete = constants.%T.as_type.as.Iterate.impl.NewCursor.specific_fn]
// CHECK:STDOUT:   %bound_method.loc38_19.2: <bound method> = bound_method %c.ref, %specific_fn.loc38_19.1
// CHECK:STDOUT:   %var: ref %tuple.type.005 = var_storage invalid
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.call: init %tuple.type.005 to %var = call %bound_method.loc38_19.2(%c.ref)
// CHECK:STDOUT:   assign %var, %T.as_type.as.Iterate.impl.NewCursor.call
// CHECK:STDOUT:   br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT: !for.next:
// CHECK:STDOUT:   %impl.elem3: %.9bc = impl_witness_access constants.%Iterate.impl_witness.1e9, element3 [concrete = constants.%T.as_type.as.Iterate.impl.Next.e9d]
// CHECK:STDOUT:   %bound_method.loc38_19.3: <bound method> = bound_method %c.ref, %impl.elem3
// CHECK:STDOUT:   %facet_value.loc38_19.3: %CppRange_where.type.d682d6.1 = facet_value constants.%ConstRange, (constants.%custom_witness.5d5, constants.%custom_witness.b76, constants.%custom_witness.9e0, constants.%custom_witness.393, constants.%custom_witness.eb7, constants.%custom_witness.f8f19d.2, constants.%Copy.impl_witness.e3b, constants.%custom_witness.b76) [concrete = constants.%facet_value.065]
// CHECK:STDOUT:   %.loc38_19.3: %CppRange_where.type.d682d6.1 = converted constants.%ConstRange, %facet_value.loc38_19.3 [concrete = constants.%facet_value.065]
// CHECK:STDOUT:   %facet_value.loc38_19.4: %CppRange_where.type.d682d6.1 = facet_value constants.%ConstRange, (constants.%custom_witness.5d5, constants.%custom_witness.b76, constants.%custom_witness.9e0, constants.%custom_witness.393, constants.%custom_witness.eb7, constants.%custom_witness.f8f19d.2, constants.%Copy.impl_witness.e3b, constants.%custom_witness.b76) [concrete = constants.%facet_value.065]
// CHECK:STDOUT:   %.loc38_19.4: %CppRange_where.type.d682d6.1 = converted constants.%ConstRange, %facet_value.loc38_19.4 [concrete = constants.%facet_value.065]
// CHECK:STDOUT:   %specific_fn.loc38_19.2: <specific function> = specific_function %impl.elem3, @T.as_type.as.Iterate.impl.Next(constants.%facet_value.065) [concrete = constants.%T.as_type.as.Iterate.impl.Next.specific_fn]
// CHECK:STDOUT:   %bound_method.loc38_19.4: <bound method> = bound_method %c.ref, %specific_fn.loc38_19.2
// CHECK:STDOUT:   %.loc38_19.5: ref %Optional.74d = temporary_storage
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.call: init %Optional.74d to %.loc38_19.5 = call %bound_method.loc38_19.4(%c.ref, %var)
// CHECK:STDOUT:   %.loc38_19.6: ref %Optional.74d = temporary %.loc38_19.5, %T.as_type.as.Iterate.impl.Next.call
// CHECK:STDOUT:   %.loc38_19.7: %Optional.HasValue.type.0c6 = specific_constant imports.%Core.import_ref.4b2, @Optional(constants.%OptionalStorage.facet.1b5) [concrete = constants.%Optional.HasValue.32a]
// CHECK:STDOUT:   %HasValue.ref: %Optional.HasValue.type.0c6 = name_ref HasValue, %.loc38_19.7 [concrete = constants.%Optional.HasValue.32a]
// CHECK:STDOUT:   %Optional.HasValue.bound: <bound method> = bound_method %.loc38_19.6, %HasValue.ref
// CHECK:STDOUT:   %Optional.HasValue.specific_fn: <specific function> = specific_function %HasValue.ref, @Optional.HasValue(constants.%OptionalStorage.facet.1b5) [concrete = constants.%Optional.HasValue.specific_fn.03d]
// CHECK:STDOUT:   %bound_method.loc38_19.5: <bound method> = bound_method %.loc38_19.6, %Optional.HasValue.specific_fn
// CHECK:STDOUT:   %.loc38_19.8: %Optional.74d = acquire_value %.loc38_19.6
// CHECK:STDOUT:   %Optional.HasValue.call: init bool = call %bound_method.loc38_19.5(%.loc38_19.8)
// CHECK:STDOUT:   %.loc38_19.9: bool = value_of_initializer %Optional.HasValue.call
// CHECK:STDOUT:   %.loc38_19.10: bool = converted %Optional.HasValue.call, %.loc38_19.9
// CHECK:STDOUT:   if %.loc38_19.10 br !for.body else br !for.done
// CHECK:STDOUT:
// CHECK:STDOUT: !for.body:
// CHECK:STDOUT:   %.loc38_19.11: %Optional.Get.type.253 = specific_constant imports.%Core.import_ref.52c, @Optional(constants.%OptionalStorage.facet.1b5) [concrete = constants.%Optional.Get.642]
// CHECK:STDOUT:   %Get.ref: %Optional.Get.type.253 = name_ref Get, %.loc38_19.11 [concrete = constants.%Optional.Get.642]
// CHECK:STDOUT:   %Optional.Get.bound: <bound method> = bound_method %.loc38_19.6, %Get.ref
// CHECK:STDOUT:   %Optional.Get.specific_fn: <specific function> = specific_function %Get.ref, @Optional.Get(constants.%OptionalStorage.facet.1b5) [concrete = constants.%Optional.Get.specific_fn.4b7]
// CHECK:STDOUT:   %bound_method.loc38_19.6: <bound method> = bound_method %.loc38_19.6, %Optional.Get.specific_fn
// CHECK:STDOUT:   %.loc38_19.12: %Optional.74d = acquire_value %.loc38_19.6
// CHECK:STDOUT:   %Optional.Get.call: init %f64.84c = call %bound_method.loc38_19.6(%.loc38_19.12)
// CHECK:STDOUT:   %.loc38_19.13: %f64.84c = value_of_initializer %Optional.Get.call
// CHECK:STDOUT:   %.loc38_19.14: %f64.84c = converted %Optional.Get.call, %.loc38_19.13
// CHECK:STDOUT:   %f64.loc38: type = type_literal constants.%f64.84c [concrete = constants.%f64.84c]
// CHECK:STDOUT:   %i: %f64.84c = wrapper_binding i, %.loc38_19.14
// CHECK:STDOUT:   %sum.ref: ref %f64.84c = name_ref sum, %sum
// CHECK:STDOUT:   %i.ref: %f64.84c = name_ref i, %i
// CHECK:STDOUT:   %impl.elem0.loc39: %.57e = impl_witness_access constants.%AddAssignWith.impl_witness.c16, element0 [concrete = constants.%Float.as.AddAssignWith.impl.Op.e48]
// CHECK:STDOUT:   %bound_method.loc39_9.1: <bound method> = bound_method %sum.ref, %impl.elem0.loc39
// CHECK:STDOUT:   %specific_fn.loc39: <specific function> = specific_function %impl.elem0.loc39, @Float.as.AddAssignWith.impl.Op(constants.%int_64) [concrete = constants.%Float.as.AddAssignWith.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc39_9.2: <bound method> = bound_method %sum.ref, %specific_fn.loc39
// CHECK:STDOUT:   %Float.as.AddAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc39_9.2(%sum.ref, %i.ref)
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc38_19.1: <bound method> = bound_method %.loc38_19.6, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.8
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc38_19.1: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc38_19.1(%.loc38_19.6)
// CHECK:STDOUT:   br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT: !for.done:
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc38_19.2: <bound method> = bound_method %.loc38_19.6, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.8
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc38_19.2: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc38_19.2(%.loc38_19.6)
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc38_19.3: <bound method> = bound_method %var, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.10
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc38_19.3: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc38_19.3(%var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestIterate(constants.%R) {
// CHECK:STDOUT:   %R.patt.loc20_17.2 => constants.%R.patt
// CHECK:STDOUT:   %R.loc20_17.1 => constants.%R
// CHECK:STDOUT:   %R.as_type.loc20_61.1 => constants.%R.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.f0c
// CHECK:STDOUT:   %r.param_patt.loc20_59.2 => constants.%r.param_patt.9f4
// CHECK:STDOUT:   %r.patt.loc20_59.2 => constants.%r.patt.48b
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestIterate(constants.%facet_value.f70) {
// CHECK:STDOUT:   %R.patt.loc20_17.2 => constants.%R.patt
// CHECK:STDOUT:   %R.loc20_17.1 => constants.%facet_value.f70
// CHECK:STDOUT:   %R.as_type.loc20_61.1 => constants.%ConstRange
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.025
// CHECK:STDOUT:   %r.param_patt.loc20_59.2 => constants.%r.param_patt.a73
// CHECK:STDOUT:   %r.patt.loc20_59.2 => constants.%r.patt.0b5
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc20 => constants.%complete_type.357
// CHECK:STDOUT:   %Iterate.lookup_impl_witness => constants.%Iterate.impl_witness.1e9
// CHECK:STDOUT:   %Iterate.facet.loc24_19.5 => constants.%Iterate.facet.795
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type => constants.%Iterate.WithSelf.NewCursor.type.db9
// CHECK:STDOUT:   %.loc24_19.19 => constants.%.c3e
// CHECK:STDOUT:   %impl.elem2.loc24_19.3 => constants.%T.as_type.as.Iterate.impl.NewCursor.d46
// CHECK:STDOUT:   %specific_impl_fn.loc24_19.4 => constants.%T.as_type.as.Iterate.impl.NewCursor.specific_fn
// CHECK:STDOUT:   %impl.elem1 => constants.%Destroy.facet.362
// CHECK:STDOUT:   %as_type => constants.%tuple.type.005
// CHECK:STDOUT:   %require_complete.1 => constants.%complete_type.320
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type => constants.%Iterate.WithSelf.Next.type.8a3
// CHECK:STDOUT:   %.loc24_19.20 => constants.%.9bc
// CHECK:STDOUT:   %impl.elem3.loc24_19.2 => constants.%T.as_type.as.Iterate.impl.Next.e9d
// CHECK:STDOUT:   %specific_impl_fn.loc24_19.5 => constants.%T.as_type.as.Iterate.impl.Next.specific_fn
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type => constants.%Destroy.WithSelf.SelfDestruct.type.dd4
// CHECK:STDOUT:   %.loc24_19.21 => constants.%.38e
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.f8f19d.9
// CHECK:STDOUT:   %impl.elem2.loc24_19.4 => constants.%Destroy.WithSelf.SelfDestruct.db3fdb.10
// CHECK:STDOUT:   %specific_impl_fn.loc24_19.6 => constants.%Destroy.WithSelf.SelfDestruct.db3fdb.10
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
// CHECK:STDOUT:   %Optional.Get.type.c64: type = fn_type @Optional.Get, @Optional(%T.220) [symbolic]
// CHECK:STDOUT:   %Optional.Get.bcf: %Optional.Get.type.c64 = struct_value () [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.type.49e: type = fn_type @Optional.HasValue, @Optional(%T.220) [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.c20: %Optional.HasValue.type.49e = struct_value () [symbolic]
// CHECK:STDOUT:   %T.283: %facet_type.f48 = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.74b: type = facet_access_type %T.283 [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.28a: <witness> = lookup_impl_witness %T.283, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.16e: %Destroy.type = facet_value %T.as_type.74b, (%Destroy.lookup_impl_witness.28a) [symbolic]
// CHECK:STDOUT:   %MaybeUnformed.9b7: type = class_type @MaybeUnformed, @MaybeUnformed(%Destroy.facet.16e) [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.faa: type = class_type @DefaultOptionalStorage, @DefaultOptionalStorage(%T.283) [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.elem.06a: type = unbound_element_type %DefaultOptionalStorage.faa, bool [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.elem.9db: type = unbound_element_type %DefaultOptionalStorage.faa, %MaybeUnformed.9b7 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Get.type.2f2: type = fn_type @T.as_type.as.OptionalStorage.impl.Get, @T.as_type.as.OptionalStorage.impl(%T.283) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Get.884: %T.as_type.as.OptionalStorage.impl.Get.type.2f2 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Has.type.d28: type = fn_type @T.as_type.as.OptionalStorage.impl.Has, @T.as_type.as.OptionalStorage.impl(%T.283) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Has.f29: %T.as_type.as.OptionalStorage.impl.Has.type.d28 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Some.type.760: type = fn_type @T.as_type.as.OptionalStorage.impl.Some, @T.as_type.as.OptionalStorage.impl(%T.283) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Some.268: %T.as_type.as.OptionalStorage.impl.Some.type.760 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.None.type.925: type = fn_type @T.as_type.as.OptionalStorage.impl.None, @T.as_type.as.OptionalStorage.impl(%T.283) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.None.f42: %T.as_type.as.OptionalStorage.impl.None.type.925 = struct_value () [symbolic]
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
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.type.d4a: type = fn_type @T.as_type.as.Iterate.impl.Next, @T.as_type.as.Iterate.impl(%T.f45530.2) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.5a8: %T.as_type.as.Iterate.impl.Next.type.d4a = struct_value () [symbolic]
// CHECK:STDOUT:   %CppRangeForIterate.lookup_impl_witness.b581a8.2: <witness> = lookup_impl_witness %T.f45530.2, @CppRangeForIterate [symbolic]
// CHECK:STDOUT:   %impl.elem0.5f971c.2: type = impl_witness_access %CppRangeForIterate.lookup_impl_witness.b581a8.2, element0 [symbolic]
// CHECK:STDOUT:   %CppUnsafeDeref.lookup_impl_witness.87e543.2: <witness> = lookup_impl_witness %impl.elem0.5f971c.2, @CppUnsafeDeref [symbolic]
// CHECK:STDOUT:   %impl.elem0.493765.2: type = impl_witness_access %CppUnsafeDeref.lookup_impl_witness.87e543.2, element0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.type.8fb: type = fn_type @T.as_type.as.Iterate.impl.NewCursor, @T.as_type.as.Iterate.impl(%T.f45530.2) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.8a1: %T.as_type.as.Iterate.impl.NewCursor.type.8fb = struct_value () [symbolic]
// CHECK:STDOUT:   %impl.elem1.c2dc15.2: type = impl_witness_access %CppRangeForIterate.lookup_impl_witness.b581a8.2, element1 [symbolic]
// CHECK:STDOUT:   %tuple.type.1ba: type = tuple_type (%impl.elem0.5f971c.2, %impl.elem1.c2dc15.2) [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.812: <witness> = lookup_impl_witness %tuple.type.1ba, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.9cd: %Destroy.type = facet_value %tuple.type.1ba, (%Destroy.lookup_impl_witness.812) [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.35816c.2: <witness> = lookup_impl_witness %impl.elem0.493765.2, @Destroy [symbolic]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.167: <witness> = lookup_impl_witness %impl.elem0.493765.2, @Copy [symbolic]
// CHECK:STDOUT:   %facet_value.f93: %facet_type.f48 = facet_value %impl.elem0.493765.2, (%Destroy.lookup_impl_witness.35816c.2, %Copy.lookup_impl_witness.167) [symbolic]
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete]
// CHECK:STDOUT:   %f64.84c: type = class_type @Float, @Float(%int_64) [concrete]
// CHECK:STDOUT:   %pattern_type.39b: type = pattern_type %f64.84c [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc44 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.2: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.2: type = fn_type @Destroy.WithSelf.Op.loc44 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.2: %Destroy.WithSelf.Op.type.ef016f.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.2: type = fn_type @Destroy.WithSelf.SelfDestruct.loc44 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.2: %Destroy.WithSelf.SelfDestruct.type.fbceb5.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.f8f19d.2: <witness> = custom_witness (%Destroy.WithSelf.Op.403171.2, %Destroy.WithSelf.SubobjectDestroy.d01daf.2, %Destroy.WithSelf.SelfDestruct.db3fdb.2), @Destroy [concrete]
// CHECK:STDOUT:   %Float.as.Copy.impl.Op.type.98e: type = fn_type @Float.as.Copy.impl.Op, @Float.as.Copy.impl(%N.fe9) [symbolic]
// CHECK:STDOUT:   %Float.as.Copy.impl.Op.1a7: %Float.as.Copy.impl.Op.type.98e = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.e3b: <witness> = impl_witness imports.%Copy.impl_witness_table.75b, @Float.as.Copy.impl(%int_64) [concrete]
// CHECK:STDOUT:   %facet_value.198: %facet_type.f48 = facet_value %f64.84c, (%custom_witness.f8f19d.2, %Copy.impl_witness.e3b) [concrete]
// CHECK:STDOUT:   %Iterate.lookup_impl_witness.1d8: <witness> = lookup_impl_witness %.Self.500, @Iterate [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.294: %facet_type.f48 = impl_witness_access %Iterate.lookup_impl_witness.1d8, element0 [symbolic_self]
// CHECK:STDOUT:   %Iterate_where.type.d76: type = facet_type <@Iterate where %impl.elem0.294 = %facet_value.198> [concrete]
// CHECK:STDOUT:   %pattern_type.d54: type = pattern_type %Iterate_where.type.d76 [concrete]
// CHECK:STDOUT:   %R.patt: %pattern_type.d54 = symbolic_binding_pattern R, 0 [symbolic]
// CHECK:STDOUT:   %R: %Iterate_where.type.d76 = symbolic_binding R, 0 [symbolic]
// CHECK:STDOUT:   %R.as_type: type = facet_access_type %R [symbolic]
// CHECK:STDOUT:   %pattern_type.f0c: type = pattern_type %R.as_type [symbolic]
// CHECK:STDOUT:   %r.param_patt.9f4: %pattern_type.f0c = value_param_pattern [symbolic]
// CHECK:STDOUT:   %r.patt.48b: %pattern_type.f0c = wrapper_binding_pattern r, %r.param_patt.9f4 [symbolic]
// CHECK:STDOUT:   %i.patt: %pattern_type.39b = value_binding_pattern i [concrete]
// CHECK:STDOUT:   %Iterate.lookup_impl_witness.22a: <witness> = lookup_impl_witness %R, @Iterate [symbolic]
// CHECK:STDOUT:   %Iterate.facet.e88: %Iterate.type = facet_value %R.as_type, (%Iterate.lookup_impl_witness.22a) [symbolic]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.8c6: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.e88) [symbolic]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.03b: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.e88) [symbolic]
// CHECK:STDOUT:   %.10b: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.8c6, %Iterate.facet.e88 [symbolic]
// CHECK:STDOUT:   %impl.elem2.846: %.10b = impl_witness_access %Iterate.lookup_impl_witness.22a, element2 [symbolic]
// CHECK:STDOUT:   %impl.elem1.dbb: %Destroy.type = impl_witness_access %Iterate.lookup_impl_witness.22a, element1 [symbolic]
// CHECK:STDOUT:   %as_type.a72: type = facet_access_type %impl.elem1.dbb [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.145: <specific function> = specific_impl_function %impl.elem2.846, @Iterate.WithSelf.NewCursor(%Iterate.facet.e88) [symbolic]
// CHECK:STDOUT:   %.573: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.03b, %Iterate.facet.e88 [symbolic]
// CHECK:STDOUT:   %impl.elem3.31f: %.573 = impl_witness_access %Iterate.lookup_impl_witness.22a, element3 [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.c6e: type = class_type @DefaultOptionalStorage, @DefaultOptionalStorage(%facet_value.198) [concrete]
// CHECK:STDOUT:   %OptionalStorage.impl_witness.3ff: <witness> = impl_witness imports.%OptionalStorage.impl_witness_table.460, @T.as_type.as.OptionalStorage.impl(%facet_value.198) [concrete]
// CHECK:STDOUT:   %OptionalStorage.facet.1b5: %OptionalStorage.type = facet_value %f64.84c, (%OptionalStorage.impl_witness.3ff) [concrete]
// CHECK:STDOUT:   %Optional.74d: type = class_type @Optional, @Optional(%OptionalStorage.facet.1b5) [concrete]
// CHECK:STDOUT:   %pattern_type.14a: type = pattern_type %Optional.74d [concrete]
// CHECK:STDOUT:   %specific_impl_fn.863: <specific function> = specific_impl_function %impl.elem3.31f, @Iterate.WithSelf.Next(%Iterate.facet.e88) [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.type.0c6: type = fn_type @Optional.HasValue, @Optional(%OptionalStorage.facet.1b5) [concrete]
// CHECK:STDOUT:   %Optional.HasValue.32a: %Optional.HasValue.type.0c6 = struct_value () [concrete]
// CHECK:STDOUT:   %Optional.Get.type.253: type = fn_type @Optional.Get, @Optional(%OptionalStorage.facet.1b5) [concrete]
// CHECK:STDOUT:   %Optional.Get.642: %Optional.Get.type.253 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.facet.065: %Destroy.type = facet_value %f64.84c, (%custom_witness.f8f19d.2) [concrete]
// CHECK:STDOUT:   %MaybeUnformed.045: type = class_type @MaybeUnformed, @MaybeUnformed(%Destroy.facet.065) [concrete]
// CHECK:STDOUT:   %.442: type = maybe_unformed_type %f64.84c [concrete]
// CHECK:STDOUT:   %DefaultOptionalStorage.elem.465: type = unbound_element_type %DefaultOptionalStorage.c6e, %MaybeUnformed.045 [concrete]
// CHECK:STDOUT:   %DefaultOptionalStorage.elem.f8c: type = unbound_element_type %DefaultOptionalStorage.c6e, bool [concrete]
// CHECK:STDOUT:   %struct_type.value.has_value.d86: type = struct_type {.value: %MaybeUnformed.045, .has_value: bool} [concrete]
// CHECK:STDOUT:   %Optional.HasValue.specific_fn.03d: <specific function> = specific_function %Optional.HasValue.32a, @Optional.HasValue(%OptionalStorage.facet.1b5) [concrete]
// CHECK:STDOUT:   %Optional.Get.specific_fn.4b7: <specific function> = specific_function %Optional.Get.642, @Optional.Get(%OptionalStorage.facet.1b5) [concrete]
// CHECK:STDOUT:   %AddAssignWith.type.35e: type = facet_type <@AddAssignWith, @AddAssignWith(%f64.84c)> [concrete]
// CHECK:STDOUT:   %Float.as.AddAssignWith.impl.Op.type.db9: type = fn_type @Float.as.AddAssignWith.impl.Op, @Float.as.AddAssignWith.impl(%N.fe9) [symbolic]
// CHECK:STDOUT:   %Float.as.AddAssignWith.impl.Op.1bb: %Float.as.AddAssignWith.impl.Op.type.db9 = struct_value () [symbolic]
// CHECK:STDOUT:   %AddAssignWith.impl_witness.c16: <witness> = impl_witness imports.%AddAssignWith.impl_witness_table, @Float.as.AddAssignWith.impl(%int_64) [concrete]
// CHECK:STDOUT:   %Float.as.AddAssignWith.impl.Op.type.c22: type = fn_type @Float.as.AddAssignWith.impl.Op, @Float.as.AddAssignWith.impl(%int_64) [concrete]
// CHECK:STDOUT:   %Float.as.AddAssignWith.impl.Op.e48: %Float.as.AddAssignWith.impl.Op.type.c22 = struct_value () [concrete]
// CHECK:STDOUT:   %AddAssignWith.facet: %AddAssignWith.type.35e = facet_value %f64.84c, (%AddAssignWith.impl_witness.c16) [concrete]
// CHECK:STDOUT:   %AddAssignWith.WithSelf.Op.type.3d3: type = fn_type @AddAssignWith.WithSelf.Op, @AddAssignWith.WithSelf(%f64.84c, %AddAssignWith.facet) [concrete]
// CHECK:STDOUT:   %.57e: type = fn_type_with_self_type %AddAssignWith.WithSelf.Op.type.3d3, %AddAssignWith.facet [concrete]
// CHECK:STDOUT:   %Float.as.AddAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Float.as.AddAssignWith.impl.Op.e48, @Float.as.AddAssignWith.impl.Op(%int_64) [concrete]
// CHECK:STDOUT:   %pattern_type.dde: type = pattern_type %.442 [concrete]
// CHECK:STDOUT:   %self.param_patt.603: %pattern_type.dde = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.d22: %pattern_type.dde = wrapper_binding_pattern self, %self.param_patt.603 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.3: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc48_19.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.3: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.3 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.3: type = fn_type @Destroy.WithSelf.Op.loc48_19.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.3: %Destroy.WithSelf.Op.type.ef016f.3 = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.c18: type = pattern_type %MaybeUnformed.045 [concrete]
// CHECK:STDOUT:   %self.param_patt.5bd: %pattern_type.c18 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.26e: %pattern_type.c18 = wrapper_binding_pattern self, %self.param_patt.5bd [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.4: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc48_19.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.4: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.4 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.4: type = fn_type @Destroy.WithSelf.Op.loc48_19.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.4: %Destroy.WithSelf.Op.type.ef016f.4 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.4: type = fn_type @Destroy.WithSelf.SelfDestruct.loc48_19.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.4: %Destroy.WithSelf.SelfDestruct.type.fbceb5.4 = struct_value () [concrete]
// CHECK:STDOUT:   %self.param_patt.40d: %pattern_type.831 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.560: %pattern_type.831 = wrapper_binding_pattern self, %self.param_patt.40d [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.5: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc48_19.3 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.5: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.5 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.5: type = fn_type @Destroy.WithSelf.Op.loc48_19.3 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.5: %Destroy.WithSelf.Op.type.ef016f.5 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.5: type = fn_type @Destroy.WithSelf.SelfDestruct.loc48_19.3 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.5: %Destroy.WithSelf.SelfDestruct.type.fbceb5.5 = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.65a: type = pattern_type %struct_type.value.has_value.d86 [concrete]
// CHECK:STDOUT:   %self.param_patt.60d: %pattern_type.65a = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.d38: %pattern_type.65a = wrapper_binding_pattern self, %self.param_patt.60d [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.6: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc48_19.4 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.6: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.6 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.6: type = fn_type @Destroy.WithSelf.Op.loc48_19.4 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.6: %Destroy.WithSelf.Op.type.ef016f.6 = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.282: type = pattern_type %DefaultOptionalStorage.c6e [concrete]
// CHECK:STDOUT:   %self.param_patt.470: %pattern_type.282 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.e17: %pattern_type.282 = wrapper_binding_pattern self, %self.param_patt.470 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.7: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc48_19.5 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.7: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.7 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.7: type = fn_type @Destroy.WithSelf.Op.loc48_19.5 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.7: %Destroy.WithSelf.Op.type.ef016f.7 = struct_value () [concrete]
// CHECK:STDOUT:   %self.param_patt.5e1: %pattern_type.14a = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.327: %pattern_type.14a = wrapper_binding_pattern self, %self.param_patt.5e1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.8: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc48_19.6 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.8: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.8 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.8: type = fn_type @Destroy.WithSelf.Op.loc48_19.6 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.8: %Destroy.WithSelf.Op.type.ef016f.8 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.8: type = fn_type @Destroy.WithSelf.SelfDestruct.loc48_19.6 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.8: %Destroy.WithSelf.SelfDestruct.type.fbceb5.8 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.21f: <witness> = lookup_impl_witness %impl.elem1.dbb, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.7d4: type = fn_type @Destroy.WithSelf.SelfDestruct.1, @Destroy.WithSelf(%impl.elem1.dbb) [symbolic]
// CHECK:STDOUT:   %.a69e: type = fn_type_with_self_type %Destroy.WithSelf.SelfDestruct.type.7d4, %impl.elem1.dbb [symbolic]
// CHECK:STDOUT:   %impl.elem2.05e: %.a69e = impl_witness_access %Destroy.lookup_impl_witness.21f, element2 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.fe7: <specific function> = specific_impl_function %impl.elem2.05e, @Destroy.WithSelf.SelfDestruct.1(%impl.elem1.dbb) [symbolic]
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
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.9: type = fn_type @Destroy.WithSelf.Op.loc55 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.9: %Destroy.WithSelf.Op.type.ef016f.9 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.9: type = fn_type @Destroy.WithSelf.SelfDestruct.loc55 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.9: %Destroy.WithSelf.SelfDestruct.type.fbceb5.9 = struct_value () [concrete]
// CHECK:STDOUT:   %Iterator.SubobjectDestroy.type: type = fn_type @Iterator.SubobjectDestroy [concrete]
// CHECK:STDOUT:   %Iterator.SubobjectDestroy: %Iterator.SubobjectDestroy.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.e74: <witness> = custom_witness (%Destroy.WithSelf.Op.403171.9, %Iterator.SubobjectDestroy, %Destroy.WithSelf.SelfDestruct.db3fdb.9), @Destroy [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.b9731e.1: type = fn_type @Iterator.Op.1 [concrete]
// CHECK:STDOUT:   %Iterator.Op.ac6aad.1: %Iterator.Op.type.b9731e.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.288: <witness> = custom_witness (%f64.84c, %Iterator.Op.ac6aad.1), @CppUnsafeDeref [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.b9731e.2: type = fn_type @Iterator.Op.2 [concrete]
// CHECK:STDOUT:   %Iterator.Op.ac6aad.2: %Iterator.Op.type.b9731e.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.491: <witness> = custom_witness (%Iterator.Op.ac6aad.2), @Inc [concrete]
// CHECK:STDOUT:   %Equal.type: type = fn_type @Equal [concrete]
// CHECK:STDOUT:   %Equal: %Equal.type = struct_value () [concrete]
// CHECK:STDOUT:   %NotEqual.type: type = fn_type @NotEqual [concrete]
// CHECK:STDOUT:   %NotEqual: %NotEqual.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.068: <witness> = custom_witness (%Equal, %NotEqual), @EqWith, @EqWith(%Iterator) [concrete]
// CHECK:STDOUT:   %facet_value.ee1: %CppRange_where.type.d682d6.1 = facet_value %MutableRange, (%custom_witness.902, %custom_witness.e74, %custom_witness.288, %custom_witness.491, %custom_witness.068, %custom_witness.f8f19d.2, %Copy.impl_witness.e3b, %custom_witness.e74) [concrete]
// CHECK:STDOUT:   %tuple.type.dd0: type = tuple_type (%Iterator, %Iterator) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.9: type = fn_type @Destroy.WithSelf.SubobjectDestroy.3 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.9: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.9 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.10: type = fn_type @Destroy.WithSelf.Op.3 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.10: %Destroy.WithSelf.Op.type.ef016f.10 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.10: type = fn_type @Destroy.WithSelf.SelfDestruct.3 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.10: %Destroy.WithSelf.SelfDestruct.type.fbceb5.10 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.f8f19d.9: <witness> = custom_witness (%Destroy.WithSelf.Op.403171.10, %Destroy.WithSelf.SubobjectDestroy.d01daf.9, %Destroy.WithSelf.SelfDestruct.db3fdb.10), @Destroy [concrete]
// CHECK:STDOUT:   %Destroy.facet.f0b: %Destroy.type = facet_value %tuple.type.dd0, (%custom_witness.f8f19d.9) [concrete]
// CHECK:STDOUT:   %Iterate.impl_witness.fcb: <witness> = impl_witness imports.%Iterate.impl_witness_table.121, @T.as_type.as.Iterate.impl(%facet_value.ee1) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.type.071: type = fn_type @T.as_type.as.Iterate.impl.NewCursor, @T.as_type.as.Iterate.impl(%facet_value.ee1) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.0dc: %T.as_type.as.Iterate.impl.NewCursor.type.071 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.type.c56: type = fn_type @T.as_type.as.Iterate.impl.Next, @T.as_type.as.Iterate.impl(%facet_value.ee1) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.fcd: %T.as_type.as.Iterate.impl.Next.type.c56 = struct_value () [concrete]
// CHECK:STDOUT:   %Iterate.facet.34c: %Iterate.type = facet_value %MutableRange, (%Iterate.impl_witness.fcb) [concrete]
// CHECK:STDOUT:   %facet_value.496: %Iterate_where.type.d76 = facet_value %MutableRange, (%Iterate.impl_witness.fcb) [concrete]
// CHECK:STDOUT:   %r.param_patt.aeb: %pattern_type.992 = value_param_pattern [concrete]
// CHECK:STDOUT:   %r.patt.892: %pattern_type.992 = wrapper_binding_pattern r, %r.param_patt.aeb [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.363: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.34c) [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.8bc: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.34c) [concrete]
// CHECK:STDOUT:   %.d26: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.363, %Iterate.facet.34c [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.specific_fn: <specific function> = specific_function %T.as_type.as.Iterate.impl.NewCursor.0dc, @T.as_type.as.Iterate.impl.NewCursor(%facet_value.ee1) [concrete]
// CHECK:STDOUT:   %.43f: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.8bc, %Iterate.facet.34c [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.specific_fn: <specific function> = specific_function %T.as_type.as.Iterate.impl.Next.fcd, @T.as_type.as.Iterate.impl.Next(%facet_value.ee1) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.1a9: type = fn_type @Destroy.WithSelf.SelfDestruct.1, @Destroy.WithSelf(%Destroy.facet.f0b) [concrete]
// CHECK:STDOUT:   %.4a2: type = fn_type_with_self_type %Destroy.WithSelf.SelfDestruct.type.1a9, %Destroy.facet.f0b [concrete]
// CHECK:STDOUT:   %complete_type.843: <witness> = complete_type_witness %tuple.type.dd0 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.4b2: @Optional.%Optional.HasValue.type (%Optional.HasValue.type.49e) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.HasValue (constants.%Optional.HasValue.c20)]
// CHECK:STDOUT:   %Core.import_ref.52c: @Optional.%Optional.Get.type (%Optional.Get.type.c64) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.Get (constants.%Optional.Get.bcf)]
// CHECK:STDOUT:   %Core.import_ref.ace: type = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%DefaultOptionalStorage (constants.%DefaultOptionalStorage.faa)]
// CHECK:STDOUT:   %Core.import_ref.ef42: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.None.type (%T.as_type.as.OptionalStorage.impl.None.type.925) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.None (constants.%T.as_type.as.OptionalStorage.impl.None.f42)]
// CHECK:STDOUT:   %Core.import_ref.a20: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Some.type (%T.as_type.as.OptionalStorage.impl.Some.type.760) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Some (constants.%T.as_type.as.OptionalStorage.impl.Some.268)]
// CHECK:STDOUT:   %Core.import_ref.af7: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Has.type (%T.as_type.as.OptionalStorage.impl.Has.type.d28) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Has (constants.%T.as_type.as.OptionalStorage.impl.Has.f29)]
// CHECK:STDOUT:   %Core.import_ref.dd4: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Get.type (%T.as_type.as.OptionalStorage.impl.Get.type.2f2) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Get (constants.%T.as_type.as.OptionalStorage.impl.Get.884)]
// CHECK:STDOUT:   %Core.import_ref.554 = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], unloaded
// CHECK:STDOUT:   %OptionalStorage.impl_witness_table.460 = impl_witness_table (%Core.import_ref.ace, %Core.import_ref.ef42, %Core.import_ref.a20, %Core.import_ref.af7, %Core.import_ref.dd4, %Core.import_ref.554), @T.as_type.as.OptionalStorage.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.1d708: @DefaultOptionalStorage.%DefaultOptionalStorage.elem.1 (%DefaultOptionalStorage.elem.9db) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [concrete = %field_decl.5f0]
// CHECK:STDOUT:   %Core.import_ref.929: @DefaultOptionalStorage.%DefaultOptionalStorage.elem.2 (%DefaultOptionalStorage.elem.06a) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [concrete = %field_decl.6a8]
// CHECK:STDOUT:   %Core.import_ref.eef: %facet_type.f48 = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.Iterate.impl.%facet_value (constants.%facet_value.f93)]
// CHECK:STDOUT:   %Core.import_ref.fcc: %Destroy.type = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.Iterate.impl.%Destroy.facet (constants.%Destroy.facet.9cd)]
// CHECK:STDOUT:   %Core.import_ref.946: @T.as_type.as.Iterate.impl.%T.as_type.as.Iterate.impl.NewCursor.type (%T.as_type.as.Iterate.impl.NewCursor.type.8fb) = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.Iterate.impl.%T.as_type.as.Iterate.impl.NewCursor (constants.%T.as_type.as.Iterate.impl.NewCursor.8a1)]
// CHECK:STDOUT:   %Core.import_ref.cfb: @T.as_type.as.Iterate.impl.%T.as_type.as.Iterate.impl.Next.type (%T.as_type.as.Iterate.impl.Next.type.d4a) = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.Iterate.impl.%T.as_type.as.Iterate.impl.Next (constants.%T.as_type.as.Iterate.impl.Next.5a8)]
// CHECK:STDOUT:   %Iterate.impl_witness_table.121 = impl_witness_table (%Core.import_ref.eef, %Core.import_ref.fcc, %Core.import_ref.946, %Core.import_ref.cfb), @T.as_type.as.Iterate.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.176: @Float.as.Copy.impl.%Float.as.Copy.impl.Op.type (%Float.as.Copy.impl.Op.type.98e) = import_ref Core//prelude/types/float, loc{{\d+_\d+}}, loaded [symbolic = @Float.as.Copy.impl.%Float.as.Copy.impl.Op (constants.%Float.as.Copy.impl.Op.1a7)]
// CHECK:STDOUT:   %Copy.impl_witness_table.75b = impl_witness_table (%Core.import_ref.176), @Float.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.ee5: @Float.as.AddAssignWith.impl.%Float.as.AddAssignWith.impl.Op.type (%Float.as.AddAssignWith.impl.Op.type.db9) = import_ref Core//prelude/types/float, loc{{\d+_\d+}}, loaded [symbolic = @Float.as.AddAssignWith.impl.%Float.as.AddAssignWith.impl.Op (constants.%Float.as.AddAssignWith.impl.Op.1bb)]
// CHECK:STDOUT:   %AddAssignWith.impl_witness_table = impl_witness_table (%Core.import_ref.ee5), @Float.as.AddAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.b0d = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %field_decl.6a8: @DefaultOptionalStorage.%DefaultOptionalStorage.elem.2 (%DefaultOptionalStorage.elem.06a) = field_decl has_value, element1, %Core.import_ref.b0d in [concrete] {}
// CHECK:STDOUT:   %Core.import_ref.87f = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %field_decl.5f0: @DefaultOptionalStorage.%DefaultOptionalStorage.elem.1 (%DefaultOptionalStorage.elem.9db) = field_decl value, element0, %Core.import_ref.87f in [concrete] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generated {
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.2: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.3 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc48_19.1 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.3] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.dde = ref_param_pattern [concrete = constants.%self.param_patt.603]
// CHECK:STDOUT:     %self.patt: %pattern_type.dde = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.d22]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %.442 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %.442 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.2: %Destroy.WithSelf.Op.type.ef016f.3 = fn_decl @Destroy.WithSelf.Op.loc48_19.1 [concrete = constants.%Destroy.WithSelf.Op.403171.3] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.dde = ref_param_pattern [concrete = constants.%self.param_patt.603]
// CHECK:STDOUT:     %self.patt: %pattern_type.dde = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.d22]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %.442 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %.442 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.3: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.4 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc48_19.2 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.4] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.c18 = ref_param_pattern [concrete = constants.%self.param_patt.5bd]
// CHECK:STDOUT:     %self.patt: %pattern_type.c18 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.26e]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %MaybeUnformed.045 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %MaybeUnformed.045 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.3: %Destroy.WithSelf.Op.type.ef016f.4 = fn_decl @Destroy.WithSelf.Op.loc48_19.2 [concrete = constants.%Destroy.WithSelf.Op.403171.4] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.c18 = ref_param_pattern [concrete = constants.%self.param_patt.5bd]
// CHECK:STDOUT:     %self.patt: %pattern_type.c18 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.26e]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %MaybeUnformed.045 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %MaybeUnformed.045 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.4: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.5 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc48_19.3 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.5] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.831 = ref_param_pattern [concrete = constants.%self.param_patt.40d]
// CHECK:STDOUT:     %self.patt: %pattern_type.831 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.560]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref bool = ref_param call_param0
// CHECK:STDOUT:     %self: ref bool = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.4: %Destroy.WithSelf.Op.type.ef016f.5 = fn_decl @Destroy.WithSelf.Op.loc48_19.3 [concrete = constants.%Destroy.WithSelf.Op.403171.5] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.831 = ref_param_pattern [concrete = constants.%self.param_patt.40d]
// CHECK:STDOUT:     %self.patt: %pattern_type.831 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.560]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref bool = ref_param call_param0
// CHECK:STDOUT:     %self: ref bool = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.5: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.6 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc48_19.4 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.6] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.65a = ref_param_pattern [concrete = constants.%self.param_patt.60d]
// CHECK:STDOUT:     %self.patt: %pattern_type.65a = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.d38]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %struct_type.value.has_value.d86 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %struct_type.value.has_value.d86 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.5: %Destroy.WithSelf.Op.type.ef016f.6 = fn_decl @Destroy.WithSelf.Op.loc48_19.4 [concrete = constants.%Destroy.WithSelf.Op.403171.6] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.65a = ref_param_pattern [concrete = constants.%self.param_patt.60d]
// CHECK:STDOUT:     %self.patt: %pattern_type.65a = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.d38]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %struct_type.value.has_value.d86 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %struct_type.value.has_value.d86 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.6: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.7 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc48_19.5 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.7] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.282 = ref_param_pattern [concrete = constants.%self.param_patt.470]
// CHECK:STDOUT:     %self.patt: %pattern_type.282 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.e17]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %DefaultOptionalStorage.c6e = ref_param call_param0
// CHECK:STDOUT:     %self: ref %DefaultOptionalStorage.c6e = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.6: %Destroy.WithSelf.Op.type.ef016f.7 = fn_decl @Destroy.WithSelf.Op.loc48_19.5 [concrete = constants.%Destroy.WithSelf.Op.403171.7] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.282 = ref_param_pattern [concrete = constants.%self.param_patt.470]
// CHECK:STDOUT:     %self.patt: %pattern_type.282 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.e17]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %DefaultOptionalStorage.c6e = ref_param call_param0
// CHECK:STDOUT:     %self: ref %DefaultOptionalStorage.c6e = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.7: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.8 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc48_19.6 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.8] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.14a = ref_param_pattern [concrete = constants.%self.param_patt.5e1]
// CHECK:STDOUT:     %self.patt: %pattern_type.14a = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.327]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %Optional.74d = ref_param call_param0
// CHECK:STDOUT:     %self: ref %Optional.74d = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.7: %Destroy.WithSelf.Op.type.ef016f.8 = fn_decl @Destroy.WithSelf.Op.loc48_19.6 [concrete = constants.%Destroy.WithSelf.Op.403171.8] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.14a = ref_param_pattern [concrete = constants.%self.param_patt.5e1]
// CHECK:STDOUT:     %self.patt: %pattern_type.14a = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.327]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %Optional.74d = ref_param call_param0
// CHECK:STDOUT:     %self: ref %Optional.74d = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @TestIterate(%R.loc44_17.2: %Iterate_where.type.d76) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Iterate.lookup_impl_witness: <witness> = lookup_impl_witness %R.loc44_17.1, @Iterate [symbolic = %Iterate.lookup_impl_witness (constants.%Iterate.lookup_impl_witness.22a)]
// CHECK:STDOUT:   %Iterate.facet.loc48_19.5: %Iterate.type = facet_value %R.as_type.loc44_61.1, (%Iterate.lookup_impl_witness) [symbolic = %Iterate.facet.loc48_19.5 (constants.%Iterate.facet.e88)]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.loc48_19.5) [symbolic = %Iterate.WithSelf.NewCursor.type (constants.%Iterate.WithSelf.NewCursor.type.8c6)]
// CHECK:STDOUT:   %.loc48_19.19: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type, %Iterate.facet.loc48_19.5 [symbolic = %.loc48_19.19 (constants.%.10b)]
// CHECK:STDOUT:   %impl.elem2.loc48_19.3: @TestIterate.%.loc48_19.19 (%.10b) = impl_witness_access %Iterate.lookup_impl_witness, element2 [symbolic = %impl.elem2.loc48_19.3 (constants.%impl.elem2.846)]
// CHECK:STDOUT:   %specific_impl_fn.loc48_19.4: <specific function> = specific_impl_function %impl.elem2.loc48_19.3, @Iterate.WithSelf.NewCursor(%Iterate.facet.loc48_19.5) [symbolic = %specific_impl_fn.loc48_19.4 (constants.%specific_impl_fn.145)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.loc48_19.5) [symbolic = %Iterate.WithSelf.Next.type (constants.%Iterate.WithSelf.Next.type.03b)]
// CHECK:STDOUT:   %.loc48_19.20: type = fn_type_with_self_type %Iterate.WithSelf.Next.type, %Iterate.facet.loc48_19.5 [symbolic = %.loc48_19.20 (constants.%.573)]
// CHECK:STDOUT:   %impl.elem3.loc48_19.2: @TestIterate.%.loc48_19.20 (%.573) = impl_witness_access %Iterate.lookup_impl_witness, element3 [symbolic = %impl.elem3.loc48_19.2 (constants.%impl.elem3.31f)]
// CHECK:STDOUT:   %specific_impl_fn.loc48_19.5: <specific function> = specific_impl_function %impl.elem3.loc48_19.2, @Iterate.WithSelf.Next(%Iterate.facet.loc48_19.5) [symbolic = %specific_impl_fn.loc48_19.5 (constants.%specific_impl_fn.863)]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type: type = fn_type @Destroy.WithSelf.SelfDestruct.1, @Destroy.WithSelf(%impl.elem1) [symbolic = %Destroy.WithSelf.SelfDestruct.type (constants.%Destroy.WithSelf.SelfDestruct.type.7d4)]
// CHECK:STDOUT:   %.loc48_19.21: type = fn_type_with_self_type %Destroy.WithSelf.SelfDestruct.type, %impl.elem1 [symbolic = %.loc48_19.21 (constants.%.a69e)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %impl.elem1, @Destroy [symbolic = %Destroy.lookup_impl_witness (constants.%Destroy.lookup_impl_witness.21f)]
// CHECK:STDOUT:   %impl.elem2.loc48_19.4: @TestIterate.%.loc48_19.21 (%.a69e) = impl_witness_access %Destroy.lookup_impl_witness, element2 [symbolic = %impl.elem2.loc48_19.4 (constants.%impl.elem2.05e)]
// CHECK:STDOUT:   %specific_impl_fn.loc48_19.6: <specific function> = specific_impl_function %impl.elem2.loc48_19.4, @Destroy.WithSelf.SelfDestruct.1(%impl.elem1) [symbolic = %specific_impl_fn.loc48_19.6 (constants.%specific_impl_fn.fe7)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%r.param: @TestIterate.%R.as_type.loc44_61.1 (%R.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %i.patt: %pattern_type.39b = value_binding_pattern i [concrete = constants.%i.patt]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %r.ref: @TestIterate.%R.as_type.loc44_61.1 (%R.as_type) = name_ref r, %r
// CHECK:STDOUT:     %impl.elem2.loc48_19.1: @TestIterate.%.loc48_19.19 (%.10b) = impl_witness_access constants.%Iterate.lookup_impl_witness.22a, element2 [symbolic = %impl.elem2.loc48_19.3 (constants.%impl.elem2.846)]
// CHECK:STDOUT:     %bound_method.loc48_19.1: <bound method> = bound_method %r.ref, %impl.elem2.loc48_19.1
// CHECK:STDOUT:     %Iterate.facet.loc48_19.1: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.22a) [symbolic = %Iterate.facet.loc48_19.5 (constants.%Iterate.facet.e88)]
// CHECK:STDOUT:     %.loc48_19.1: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc48_19.1 [symbolic = %Iterate.facet.loc48_19.5 (constants.%Iterate.facet.e88)]
// CHECK:STDOUT:     %Iterate.facet.loc48_19.2: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.22a) [symbolic = %Iterate.facet.loc48_19.5 (constants.%Iterate.facet.e88)]
// CHECK:STDOUT:     %.loc48_19.2: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc48_19.2 [symbolic = %Iterate.facet.loc48_19.5 (constants.%Iterate.facet.e88)]
// CHECK:STDOUT:     %specific_impl_fn.loc48_19.1: <specific function> = specific_impl_function %impl.elem2.loc48_19.1, @Iterate.WithSelf.NewCursor(constants.%Iterate.facet.e88) [symbolic = %specific_impl_fn.loc48_19.4 (constants.%specific_impl_fn.145)]
// CHECK:STDOUT:     %bound_method.loc48_19.2: <bound method> = bound_method %r.ref, %specific_impl_fn.loc48_19.1
// CHECK:STDOUT:     %var: ref @TestIterate.%as_type (%as_type.a72) = var_storage invalid
// CHECK:STDOUT:     %Iterate.WithSelf.NewCursor.call: init @TestIterate.%as_type (%as_type.a72) to %var = call %bound_method.loc48_19.2(%r.ref)
// CHECK:STDOUT:     assign %var, %Iterate.WithSelf.NewCursor.call
// CHECK:STDOUT:     br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.next:
// CHECK:STDOUT:     %impl.elem3.loc48_19.1: @TestIterate.%.loc48_19.20 (%.573) = impl_witness_access constants.%Iterate.lookup_impl_witness.22a, element3 [symbolic = %impl.elem3.loc48_19.2 (constants.%impl.elem3.31f)]
// CHECK:STDOUT:     %bound_method.loc48_19.3: <bound method> = bound_method %r.ref, %impl.elem3.loc48_19.1
// CHECK:STDOUT:     %Iterate.facet.loc48_19.3: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.22a) [symbolic = %Iterate.facet.loc48_19.5 (constants.%Iterate.facet.e88)]
// CHECK:STDOUT:     %.loc48_19.3: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc48_19.3 [symbolic = %Iterate.facet.loc48_19.5 (constants.%Iterate.facet.e88)]
// CHECK:STDOUT:     %Iterate.facet.loc48_19.4: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.22a) [symbolic = %Iterate.facet.loc48_19.5 (constants.%Iterate.facet.e88)]
// CHECK:STDOUT:     %.loc48_19.4: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc48_19.4 [symbolic = %Iterate.facet.loc48_19.5 (constants.%Iterate.facet.e88)]
// CHECK:STDOUT:     %.loc48_19.5: %Destroy.type = converted constants.%as_type.a72, constants.%impl.elem1.dbb [symbolic = %impl.elem1 (constants.%impl.elem1.dbb)]
// CHECK:STDOUT:     %.loc48_19.6: %Destroy.type = converted constants.%as_type.a72, constants.%impl.elem1.dbb [symbolic = %impl.elem1 (constants.%impl.elem1.dbb)]
// CHECK:STDOUT:     %specific_impl_fn.loc48_19.2: <specific function> = specific_impl_function %impl.elem3.loc48_19.1, @Iterate.WithSelf.Next(constants.%Iterate.facet.e88) [symbolic = %specific_impl_fn.loc48_19.5 (constants.%specific_impl_fn.863)]
// CHECK:STDOUT:     %bound_method.loc48_19.4: <bound method> = bound_method %r.ref, %specific_impl_fn.loc48_19.2
// CHECK:STDOUT:     %.loc48_19.7: ref %Optional.74d = temporary_storage
// CHECK:STDOUT:     %Iterate.WithSelf.Next.call: init %Optional.74d to %.loc48_19.7 = call %bound_method.loc48_19.4(%r.ref, %var)
// CHECK:STDOUT:     %.loc48_19.8: ref %Optional.74d = temporary %.loc48_19.7, %Iterate.WithSelf.Next.call
// CHECK:STDOUT:     %.loc48_19.9: %Optional.HasValue.type.0c6 = specific_constant imports.%Core.import_ref.4b2, @Optional(constants.%OptionalStorage.facet.1b5) [concrete = constants.%Optional.HasValue.32a]
// CHECK:STDOUT:     %HasValue.ref: %Optional.HasValue.type.0c6 = name_ref HasValue, %.loc48_19.9 [concrete = constants.%Optional.HasValue.32a]
// CHECK:STDOUT:     %Optional.HasValue.bound: <bound method> = bound_method %.loc48_19.8, %HasValue.ref
// CHECK:STDOUT:     %Optional.HasValue.specific_fn: <specific function> = specific_function %HasValue.ref, @Optional.HasValue(constants.%OptionalStorage.facet.1b5) [concrete = constants.%Optional.HasValue.specific_fn.03d]
// CHECK:STDOUT:     %bound_method.loc48_19.5: <bound method> = bound_method %.loc48_19.8, %Optional.HasValue.specific_fn
// CHECK:STDOUT:     %.loc48_19.10: %Optional.74d = acquire_value %.loc48_19.8
// CHECK:STDOUT:     %Optional.HasValue.call: init bool = call %bound_method.loc48_19.5(%.loc48_19.10)
// CHECK:STDOUT:     %.loc48_19.11: bool = value_of_initializer %Optional.HasValue.call
// CHECK:STDOUT:     %.loc48_19.12: bool = converted %Optional.HasValue.call, %.loc48_19.11
// CHECK:STDOUT:     if %.loc48_19.12 br !for.body else br !for.done
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.body:
// CHECK:STDOUT:     %.loc48_19.13: %Optional.Get.type.253 = specific_constant imports.%Core.import_ref.52c, @Optional(constants.%OptionalStorage.facet.1b5) [concrete = constants.%Optional.Get.642]
// CHECK:STDOUT:     %Get.ref: %Optional.Get.type.253 = name_ref Get, %.loc48_19.13 [concrete = constants.%Optional.Get.642]
// CHECK:STDOUT:     %Optional.Get.bound: <bound method> = bound_method %.loc48_19.8, %Get.ref
// CHECK:STDOUT:     %Optional.Get.specific_fn: <specific function> = specific_function %Get.ref, @Optional.Get(constants.%OptionalStorage.facet.1b5) [concrete = constants.%Optional.Get.specific_fn.4b7]
// CHECK:STDOUT:     %bound_method.loc48_19.6: <bound method> = bound_method %.loc48_19.8, %Optional.Get.specific_fn
// CHECK:STDOUT:     %.loc48_19.14: %Optional.74d = acquire_value %.loc48_19.8
// CHECK:STDOUT:     %Optional.Get.call: init %f64.84c = call %bound_method.loc48_19.6(%.loc48_19.14)
// CHECK:STDOUT:     %.loc48_19.15: %f64.84c = value_of_initializer %Optional.Get.call
// CHECK:STDOUT:     %.loc48_19.16: %f64.84c = converted %Optional.Get.call, %.loc48_19.15
// CHECK:STDOUT:     %f64.loc48: type = type_literal constants.%f64.84c [concrete = constants.%f64.84c]
// CHECK:STDOUT:     %i: %f64.84c = wrapper_binding i, %.loc48_19.16
// CHECK:STDOUT:     %sum.ref: ref %f64.84c = name_ref sum, %sum
// CHECK:STDOUT:     %i.ref: %f64.84c = name_ref i, %i
// CHECK:STDOUT:     %impl.elem0.loc49: %.57e = impl_witness_access constants.%AddAssignWith.impl_witness.c16, element0 [concrete = constants.%Float.as.AddAssignWith.impl.Op.e48]
// CHECK:STDOUT:     %bound_method.loc49_9.1: <bound method> = bound_method %sum.ref, %impl.elem0.loc49
// CHECK:STDOUT:     %specific_fn.loc49: <specific function> = specific_function %impl.elem0.loc49, @Float.as.AddAssignWith.impl.Op(constants.%int_64) [concrete = constants.%Float.as.AddAssignWith.impl.Op.specific_fn]
// CHECK:STDOUT:     %bound_method.loc49_9.2: <bound method> = bound_method %sum.ref, %specific_fn.loc49
// CHECK:STDOUT:     %Float.as.AddAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc49_9.2(%sum.ref, %i.ref)
// CHECK:STDOUT:     %Destroy.WithSelf.SelfDestruct.bound.loc48_19.1: <bound method> = bound_method %.loc48_19.8, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.8
// CHECK:STDOUT:     %Destroy.WithSelf.SelfDestruct.call.loc48_19.1: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc48_19.1(%.loc48_19.8)
// CHECK:STDOUT:     br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.done:
// CHECK:STDOUT:     %Destroy.WithSelf.SelfDestruct.bound.loc48_19.2: <bound method> = bound_method %.loc48_19.8, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.8
// CHECK:STDOUT:     %Destroy.WithSelf.SelfDestruct.call.loc48_19.2: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc48_19.2(%.loc48_19.8)
// CHECK:STDOUT:     %impl.elem2.loc48_19.2: @TestIterate.%.loc48_19.21 (%.a69e) = impl_witness_access constants.%Destroy.lookup_impl_witness.21f, element2 [symbolic = %impl.elem2.loc48_19.4 (constants.%impl.elem2.05e)]
// CHECK:STDOUT:     %bound_method.loc48_19.7: <bound method> = bound_method %var, %impl.elem2.loc48_19.2
// CHECK:STDOUT:     %.loc48_19.17: %Destroy.type = converted constants.%as_type.a72, constants.%impl.elem1.dbb [symbolic = %impl.elem1 (constants.%impl.elem1.dbb)]
// CHECK:STDOUT:     %.loc48_19.18: %Destroy.type = converted constants.%as_type.a72, constants.%impl.elem1.dbb [symbolic = %impl.elem1 (constants.%impl.elem1.dbb)]
// CHECK:STDOUT:     %specific_impl_fn.loc48_19.3: <specific function> = specific_impl_function %impl.elem2.loc48_19.2, @Destroy.WithSelf.SelfDestruct.1(constants.%impl.elem1.dbb) [symbolic = %specific_impl_fn.loc48_19.6 (constants.%specific_impl_fn.fe7)]
// CHECK:STDOUT:     %bound_method.loc48_19.8: <bound method> = bound_method %var, %specific_impl_fn.loc48_19.3
// CHECK:STDOUT:     %Destroy.WithSelf.SelfDestruct.call.loc48_19.3: init %empty_tuple.type = call %bound_method.loc48_19.8(%var)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc48_19.1(%self.param: ref %.442) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc48_19.1(%self.param: ref %.442) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc48_19.1(%self.param: ref %.442) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.2(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.2(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc48_19.2(%self.param: ref %MaybeUnformed.045) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc48_19.2(%self.param: ref %MaybeUnformed.045) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc48_19.2(%self.param: ref %MaybeUnformed.045) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.3(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.3(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc48_19.3(%self.param: ref bool) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc48_19.3(%self.param: ref bool) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc48_19.3(%self.param: ref bool) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.4(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.4(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc48_19.4(%self.param: ref %struct_type.value.has_value.d86) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc48_19.1: ref bool = struct_access %self.param, element1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc48_19.1: <bound method> = bound_method %.loc48_19.1, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.5
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc48_19.1: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc48_19.1(%.loc48_19.1)
// CHECK:STDOUT:   %.loc48_19.2: ref %MaybeUnformed.045 = struct_access %self.param, element0
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc48_19.2: <bound method> = bound_method %.loc48_19.2, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.4
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc48_19.2: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc48_19.2(%.loc48_19.2)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc48_19.4(%self.param: ref %struct_type.value.has_value.d86) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc48_19.4(%self.param: ref %struct_type.value.has_value.d86) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.5(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.5(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc48_19.5(%self.param: ref %DefaultOptionalStorage.c6e) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %has_value.ref: %DefaultOptionalStorage.elem.f8c = name_ref has_value, imports.%Core.import_ref.929 [concrete = imports.%field_decl.6a8]
// CHECK:STDOUT:   %.loc48_19.1: ref bool = class_element_access %self.param, element1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc48_19.1: <bound method> = bound_method %.loc48_19.1, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.5
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc48_19.1: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc48_19.1(%.loc48_19.1)
// CHECK:STDOUT:   %value.ref: %DefaultOptionalStorage.elem.465 = name_ref value, imports.%Core.import_ref.1d708 [concrete = imports.%field_decl.5f0]
// CHECK:STDOUT:   %.loc48_19.2: ref %MaybeUnformed.045 = class_element_access %self.param, element0
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc48_19.2: <bound method> = bound_method %.loc48_19.2, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.4
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc48_19.2: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc48_19.2(%.loc48_19.2)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc48_19.5(%self.param: ref %DefaultOptionalStorage.c6e) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc48_19.5(%self.param: ref %DefaultOptionalStorage.c6e) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.6(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.6(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc48_19.6(%self.param: ref %Optional.74d) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc48_19.6(%self.param: ref %Optional.74d) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc48_19.6(%self.param: ref %Optional.74d) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.7(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.7(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestRangeFor(%m.param: ref %MutableRange) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %i.patt: %pattern_type.39b = value_binding_pattern i [concrete = constants.%i.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %m.ref: ref %MutableRange = name_ref m, %m
// CHECK:STDOUT:   %impl.elem2: %.d26 = impl_witness_access constants.%Iterate.impl_witness.fcb, element2 [concrete = constants.%T.as_type.as.Iterate.impl.NewCursor.0dc]
// CHECK:STDOUT:   %bound_method.loc62_19.1: <bound method> = bound_method %m.ref, %impl.elem2
// CHECK:STDOUT:   %facet_value.loc62_19.1: %CppRange_where.type.d682d6.1 = facet_value constants.%MutableRange, (constants.%custom_witness.902, constants.%custom_witness.e74, constants.%custom_witness.288, constants.%custom_witness.491, constants.%custom_witness.068, constants.%custom_witness.f8f19d.2, constants.%Copy.impl_witness.e3b, constants.%custom_witness.e74) [concrete = constants.%facet_value.ee1]
// CHECK:STDOUT:   %.loc62_19.1: %CppRange_where.type.d682d6.1 = converted constants.%MutableRange, %facet_value.loc62_19.1 [concrete = constants.%facet_value.ee1]
// CHECK:STDOUT:   %facet_value.loc62_19.2: %CppRange_where.type.d682d6.1 = facet_value constants.%MutableRange, (constants.%custom_witness.902, constants.%custom_witness.e74, constants.%custom_witness.288, constants.%custom_witness.491, constants.%custom_witness.068, constants.%custom_witness.f8f19d.2, constants.%Copy.impl_witness.e3b, constants.%custom_witness.e74) [concrete = constants.%facet_value.ee1]
// CHECK:STDOUT:   %.loc62_19.2: %CppRange_where.type.d682d6.1 = converted constants.%MutableRange, %facet_value.loc62_19.2 [concrete = constants.%facet_value.ee1]
// CHECK:STDOUT:   %specific_fn.loc62_19.1: <specific function> = specific_function %impl.elem2, @T.as_type.as.Iterate.impl.NewCursor(constants.%facet_value.ee1) [concrete = constants.%T.as_type.as.Iterate.impl.NewCursor.specific_fn]
// CHECK:STDOUT:   %bound_method.loc62_19.2: <bound method> = bound_method %m.ref, %specific_fn.loc62_19.1
// CHECK:STDOUT:   %var: ref %tuple.type.dd0 = var_storage invalid
// CHECK:STDOUT:   %.loc62_18.1: %MutableRange = acquire_value %m.ref
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.call: init %tuple.type.dd0 to %var = call %bound_method.loc62_19.2(%.loc62_18.1)
// CHECK:STDOUT:   assign %var, %T.as_type.as.Iterate.impl.NewCursor.call
// CHECK:STDOUT:   br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT: !for.next:
// CHECK:STDOUT:   %impl.elem3: %.43f = impl_witness_access constants.%Iterate.impl_witness.fcb, element3 [concrete = constants.%T.as_type.as.Iterate.impl.Next.fcd]
// CHECK:STDOUT:   %bound_method.loc62_19.3: <bound method> = bound_method %m.ref, %impl.elem3
// CHECK:STDOUT:   %facet_value.loc62_19.3: %CppRange_where.type.d682d6.1 = facet_value constants.%MutableRange, (constants.%custom_witness.902, constants.%custom_witness.e74, constants.%custom_witness.288, constants.%custom_witness.491, constants.%custom_witness.068, constants.%custom_witness.f8f19d.2, constants.%Copy.impl_witness.e3b, constants.%custom_witness.e74) [concrete = constants.%facet_value.ee1]
// CHECK:STDOUT:   %.loc62_19.3: %CppRange_where.type.d682d6.1 = converted constants.%MutableRange, %facet_value.loc62_19.3 [concrete = constants.%facet_value.ee1]
// CHECK:STDOUT:   %facet_value.loc62_19.4: %CppRange_where.type.d682d6.1 = facet_value constants.%MutableRange, (constants.%custom_witness.902, constants.%custom_witness.e74, constants.%custom_witness.288, constants.%custom_witness.491, constants.%custom_witness.068, constants.%custom_witness.f8f19d.2, constants.%Copy.impl_witness.e3b, constants.%custom_witness.e74) [concrete = constants.%facet_value.ee1]
// CHECK:STDOUT:   %.loc62_19.4: %CppRange_where.type.d682d6.1 = converted constants.%MutableRange, %facet_value.loc62_19.4 [concrete = constants.%facet_value.ee1]
// CHECK:STDOUT:   %specific_fn.loc62_19.2: <specific function> = specific_function %impl.elem3, @T.as_type.as.Iterate.impl.Next(constants.%facet_value.ee1) [concrete = constants.%T.as_type.as.Iterate.impl.Next.specific_fn]
// CHECK:STDOUT:   %bound_method.loc62_19.4: <bound method> = bound_method %m.ref, %specific_fn.loc62_19.2
// CHECK:STDOUT:   %.loc62_19.5: ref %Optional.74d = temporary_storage
// CHECK:STDOUT:   %.loc62_18.2: %MutableRange = acquire_value %m.ref
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.call: init %Optional.74d to %.loc62_19.5 = call %bound_method.loc62_19.4(%.loc62_18.2, %var)
// CHECK:STDOUT:   %.loc62_19.6: ref %Optional.74d = temporary %.loc62_19.5, %T.as_type.as.Iterate.impl.Next.call
// CHECK:STDOUT:   %.loc62_19.7: %Optional.HasValue.type.0c6 = specific_constant imports.%Core.import_ref.4b2, @Optional(constants.%OptionalStorage.facet.1b5) [concrete = constants.%Optional.HasValue.32a]
// CHECK:STDOUT:   %HasValue.ref: %Optional.HasValue.type.0c6 = name_ref HasValue, %.loc62_19.7 [concrete = constants.%Optional.HasValue.32a]
// CHECK:STDOUT:   %Optional.HasValue.bound: <bound method> = bound_method %.loc62_19.6, %HasValue.ref
// CHECK:STDOUT:   %Optional.HasValue.specific_fn: <specific function> = specific_function %HasValue.ref, @Optional.HasValue(constants.%OptionalStorage.facet.1b5) [concrete = constants.%Optional.HasValue.specific_fn.03d]
// CHECK:STDOUT:   %bound_method.loc62_19.5: <bound method> = bound_method %.loc62_19.6, %Optional.HasValue.specific_fn
// CHECK:STDOUT:   %.loc62_19.8: %Optional.74d = acquire_value %.loc62_19.6
// CHECK:STDOUT:   %Optional.HasValue.call: init bool = call %bound_method.loc62_19.5(%.loc62_19.8)
// CHECK:STDOUT:   %.loc62_19.9: bool = value_of_initializer %Optional.HasValue.call
// CHECK:STDOUT:   %.loc62_19.10: bool = converted %Optional.HasValue.call, %.loc62_19.9
// CHECK:STDOUT:   if %.loc62_19.10 br !for.body else br !for.done
// CHECK:STDOUT:
// CHECK:STDOUT: !for.body:
// CHECK:STDOUT:   %.loc62_19.11: %Optional.Get.type.253 = specific_constant imports.%Core.import_ref.52c, @Optional(constants.%OptionalStorage.facet.1b5) [concrete = constants.%Optional.Get.642]
// CHECK:STDOUT:   %Get.ref: %Optional.Get.type.253 = name_ref Get, %.loc62_19.11 [concrete = constants.%Optional.Get.642]
// CHECK:STDOUT:   %Optional.Get.bound: <bound method> = bound_method %.loc62_19.6, %Get.ref
// CHECK:STDOUT:   %Optional.Get.specific_fn: <specific function> = specific_function %Get.ref, @Optional.Get(constants.%OptionalStorage.facet.1b5) [concrete = constants.%Optional.Get.specific_fn.4b7]
// CHECK:STDOUT:   %bound_method.loc62_19.6: <bound method> = bound_method %.loc62_19.6, %Optional.Get.specific_fn
// CHECK:STDOUT:   %.loc62_19.12: %Optional.74d = acquire_value %.loc62_19.6
// CHECK:STDOUT:   %Optional.Get.call: init %f64.84c = call %bound_method.loc62_19.6(%.loc62_19.12)
// CHECK:STDOUT:   %.loc62_19.13: %f64.84c = value_of_initializer %Optional.Get.call
// CHECK:STDOUT:   %.loc62_19.14: %f64.84c = converted %Optional.Get.call, %.loc62_19.13
// CHECK:STDOUT:   %f64.loc62: type = type_literal constants.%f64.84c [concrete = constants.%f64.84c]
// CHECK:STDOUT:   %i: %f64.84c = wrapper_binding i, %.loc62_19.14
// CHECK:STDOUT:   %sum.ref: ref %f64.84c = name_ref sum, %sum
// CHECK:STDOUT:   %i.ref: %f64.84c = name_ref i, %i
// CHECK:STDOUT:   %impl.elem0.loc63: %.57e = impl_witness_access constants.%AddAssignWith.impl_witness.c16, element0 [concrete = constants.%Float.as.AddAssignWith.impl.Op.e48]
// CHECK:STDOUT:   %bound_method.loc63_9.1: <bound method> = bound_method %sum.ref, %impl.elem0.loc63
// CHECK:STDOUT:   %specific_fn.loc63: <specific function> = specific_function %impl.elem0.loc63, @Float.as.AddAssignWith.impl.Op(constants.%int_64) [concrete = constants.%Float.as.AddAssignWith.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc63_9.2: <bound method> = bound_method %sum.ref, %specific_fn.loc63
// CHECK:STDOUT:   %Float.as.AddAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc63_9.2(%sum.ref, %i.ref)
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc62_19.1: <bound method> = bound_method %.loc62_19.6, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.8
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc62_19.1: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc62_19.1(%.loc62_19.6)
// CHECK:STDOUT:   br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT: !for.done:
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc62_19.2: <bound method> = bound_method %.loc62_19.6, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.8
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc62_19.2: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc62_19.2(%.loc62_19.6)
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc62_19.3: <bound method> = bound_method %var, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.10
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc62_19.3: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc62_19.3(%var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestIterate(constants.%R) {
// CHECK:STDOUT:   %R.patt.loc44_17.2 => constants.%R.patt
// CHECK:STDOUT:   %R.loc44_17.1 => constants.%R
// CHECK:STDOUT:   %R.as_type.loc44_61.1 => constants.%R.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.f0c
// CHECK:STDOUT:   %r.param_patt.loc44_59.2 => constants.%r.param_patt.9f4
// CHECK:STDOUT:   %r.patt.loc44_59.2 => constants.%r.patt.48b
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestIterate(constants.%facet_value.496) {
// CHECK:STDOUT:   %R.patt.loc44_17.2 => constants.%R.patt
// CHECK:STDOUT:   %R.loc44_17.1 => constants.%facet_value.496
// CHECK:STDOUT:   %R.as_type.loc44_61.1 => constants.%MutableRange
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.992
// CHECK:STDOUT:   %r.param_patt.loc44_59.2 => constants.%r.param_patt.aeb
// CHECK:STDOUT:   %r.patt.loc44_59.2 => constants.%r.patt.892
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc44 => constants.%complete_type.357
// CHECK:STDOUT:   %Iterate.lookup_impl_witness => constants.%Iterate.impl_witness.fcb
// CHECK:STDOUT:   %Iterate.facet.loc48_19.5 => constants.%Iterate.facet.34c
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type => constants.%Iterate.WithSelf.NewCursor.type.363
// CHECK:STDOUT:   %.loc48_19.19 => constants.%.d26
// CHECK:STDOUT:   %impl.elem2.loc48_19.3 => constants.%T.as_type.as.Iterate.impl.NewCursor.0dc
// CHECK:STDOUT:   %specific_impl_fn.loc48_19.4 => constants.%T.as_type.as.Iterate.impl.NewCursor.specific_fn
// CHECK:STDOUT:   %impl.elem1 => constants.%Destroy.facet.f0b
// CHECK:STDOUT:   %as_type => constants.%tuple.type.dd0
// CHECK:STDOUT:   %require_complete.1 => constants.%complete_type.843
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type => constants.%Iterate.WithSelf.Next.type.8bc
// CHECK:STDOUT:   %.loc48_19.20 => constants.%.43f
// CHECK:STDOUT:   %impl.elem3.loc48_19.2 => constants.%T.as_type.as.Iterate.impl.Next.fcd
// CHECK:STDOUT:   %specific_impl_fn.loc48_19.5 => constants.%T.as_type.as.Iterate.impl.Next.specific_fn
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type => constants.%Destroy.WithSelf.SelfDestruct.type.1a9
// CHECK:STDOUT:   %.loc48_19.21 => constants.%.4a2
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.f8f19d.9
// CHECK:STDOUT:   %impl.elem2.loc48_19.4 => constants.%Destroy.WithSelf.SelfDestruct.db3fdb.10
// CHECK:STDOUT:   %specific_impl_fn.loc48_19.6 => constants.%Destroy.WithSelf.SelfDestruct.db3fdb.10
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
// CHECK:STDOUT:   %Optional.Get.type.c64: type = fn_type @Optional.Get, @Optional(%T.220) [symbolic]
// CHECK:STDOUT:   %Optional.Get.bcf: %Optional.Get.type.c64 = struct_value () [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.type.49e: type = fn_type @Optional.HasValue, @Optional(%T.220) [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.c20: %Optional.HasValue.type.49e = struct_value () [symbolic]
// CHECK:STDOUT:   %T.283: %facet_type.f48 = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.74b: type = facet_access_type %T.283 [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.28a: <witness> = lookup_impl_witness %T.283, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.16e: %Destroy.type = facet_value %T.as_type.74b, (%Destroy.lookup_impl_witness.28a) [symbolic]
// CHECK:STDOUT:   %MaybeUnformed.9b7: type = class_type @MaybeUnformed, @MaybeUnformed(%Destroy.facet.16e) [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.faa: type = class_type @DefaultOptionalStorage, @DefaultOptionalStorage(%T.283) [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.elem.06a: type = unbound_element_type %DefaultOptionalStorage.faa, bool [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.elem.9db: type = unbound_element_type %DefaultOptionalStorage.faa, %MaybeUnformed.9b7 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Get.type.2f2: type = fn_type @T.as_type.as.OptionalStorage.impl.Get, @T.as_type.as.OptionalStorage.impl(%T.283) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Get.884: %T.as_type.as.OptionalStorage.impl.Get.type.2f2 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Has.type.d28: type = fn_type @T.as_type.as.OptionalStorage.impl.Has, @T.as_type.as.OptionalStorage.impl(%T.283) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Has.f29: %T.as_type.as.OptionalStorage.impl.Has.type.d28 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Some.type.760: type = fn_type @T.as_type.as.OptionalStorage.impl.Some, @T.as_type.as.OptionalStorage.impl(%T.283) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Some.268: %T.as_type.as.OptionalStorage.impl.Some.type.760 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.None.type.925: type = fn_type @T.as_type.as.OptionalStorage.impl.None, @T.as_type.as.OptionalStorage.impl(%T.283) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.None.f42: %T.as_type.as.OptionalStorage.impl.None.type.925 = struct_value () [symbolic]
// CHECK:STDOUT:   %.Self.500: %Iterate.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %CppRange.type: type = facet_type <@CppRange> [concrete]
// CHECK:STDOUT:   %.Self.e0b: %CppRange.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %CppRangeForIterate.lookup_impl_witness.50b: <witness> = lookup_impl_witness %.Self.e0b, @CppRangeForIterate [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.32d: type = impl_witness_access %CppRangeForIterate.lookup_impl_witness.50b, element0 [symbolic_self]
// CHECK:STDOUT:   %CppUnsafeDeref.lookup_impl_witness.851: <witness> = lookup_impl_witness %impl.elem0.32d, @CppUnsafeDeref [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.10d: type = impl_witness_access %CppUnsafeDeref.lookup_impl_witness.851, element0 [symbolic_self]
// CHECK:STDOUT:   %CppRange_where.type.d682d6.1: type = facet_type <@CppRange where %impl.elem0.10d impls @Copy> [concrete]
// CHECK:STDOUT:   %T.f45530.2: %CppRange_where.type.d682d6.1 = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.type.d4a: type = fn_type @T.as_type.as.Iterate.impl.Next, @T.as_type.as.Iterate.impl(%T.f45530.2) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.5a8: %T.as_type.as.Iterate.impl.Next.type.d4a = struct_value () [symbolic]
// CHECK:STDOUT:   %CppRangeForIterate.lookup_impl_witness.b581a8.2: <witness> = lookup_impl_witness %T.f45530.2, @CppRangeForIterate [symbolic]
// CHECK:STDOUT:   %impl.elem0.5f971c.2: type = impl_witness_access %CppRangeForIterate.lookup_impl_witness.b581a8.2, element0 [symbolic]
// CHECK:STDOUT:   %CppUnsafeDeref.lookup_impl_witness.87e543.2: <witness> = lookup_impl_witness %impl.elem0.5f971c.2, @CppUnsafeDeref [symbolic]
// CHECK:STDOUT:   %impl.elem0.493765.2: type = impl_witness_access %CppUnsafeDeref.lookup_impl_witness.87e543.2, element0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.type.8fb: type = fn_type @T.as_type.as.Iterate.impl.NewCursor, @T.as_type.as.Iterate.impl(%T.f45530.2) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.8a1: %T.as_type.as.Iterate.impl.NewCursor.type.8fb = struct_value () [symbolic]
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
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.b87: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.467) [symbolic]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.526: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.467) [symbolic]
// CHECK:STDOUT:   %.c11: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.b87, %Iterate.facet.467 [symbolic]
// CHECK:STDOUT:   %impl.elem2.978: %.c11 = impl_witness_access %Iterate.lookup_impl_witness.01e, element2 [symbolic]
// CHECK:STDOUT:   %impl.elem1.d56: %Destroy.type = impl_witness_access %Iterate.lookup_impl_witness.01e, element1 [symbolic]
// CHECK:STDOUT:   %as_type.db9: type = facet_access_type %impl.elem1.d56 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.f0e: <specific function> = specific_impl_function %impl.elem2.978, @Iterate.WithSelf.NewCursor(%Iterate.facet.467) [symbolic]
// CHECK:STDOUT:   %.118: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.526, %Iterate.facet.467 [symbolic]
// CHECK:STDOUT:   %impl.elem3.2b1: %.118 = impl_witness_access %Iterate.lookup_impl_witness.01e, element3 [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.5b0: type = class_type @DefaultOptionalStorage, @DefaultOptionalStorage(%facet_value.978) [concrete]
// CHECK:STDOUT:   %OptionalStorage.impl_witness.13c: <witness> = impl_witness imports.%OptionalStorage.impl_witness_table.460, @T.as_type.as.OptionalStorage.impl(%facet_value.978) [concrete]
// CHECK:STDOUT:   %OptionalStorage.facet.0d1: %OptionalStorage.type = facet_value %ValueType, (%OptionalStorage.impl_witness.13c) [concrete]
// CHECK:STDOUT:   %Optional.873: type = class_type @Optional, @Optional(%OptionalStorage.facet.0d1) [concrete]
// CHECK:STDOUT:   %pattern_type.ece: type = pattern_type %Optional.873 [concrete]
// CHECK:STDOUT:   %specific_impl_fn.3f1: <specific function> = specific_impl_function %impl.elem3.2b1, @Iterate.WithSelf.Next(%Iterate.facet.467) [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.type.1e1: type = fn_type @Optional.HasValue, @Optional(%OptionalStorage.facet.0d1) [concrete]
// CHECK:STDOUT:   %Optional.HasValue.6d8: %Optional.HasValue.type.1e1 = struct_value () [concrete]
// CHECK:STDOUT:   %Optional.Get.type.915: type = fn_type @Optional.Get, @Optional(%OptionalStorage.facet.0d1) [concrete]
// CHECK:STDOUT:   %Optional.Get.788: %Optional.Get.type.915 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.facet.c1d: %Destroy.type = facet_value %ValueType, (%custom_witness.226) [concrete]
// CHECK:STDOUT:   %MaybeUnformed.4c8: type = class_type @MaybeUnformed, @MaybeUnformed(%Destroy.facet.c1d) [concrete]
// CHECK:STDOUT:   %.be1: type = maybe_unformed_type %ValueType [concrete]
// CHECK:STDOUT:   %DefaultOptionalStorage.elem.cbb: type = unbound_element_type %DefaultOptionalStorage.5b0, %MaybeUnformed.4c8 [concrete]
// CHECK:STDOUT:   %DefaultOptionalStorage.elem.917: type = unbound_element_type %DefaultOptionalStorage.5b0, bool [concrete]
// CHECK:STDOUT:   %struct_type.value.has_value.d90: type = struct_type {.value: %MaybeUnformed.4c8, .has_value: bool} [concrete]
// CHECK:STDOUT:   %Optional.HasValue.specific_fn.921: <specific function> = specific_function %Optional.HasValue.6d8, @Optional.HasValue(%OptionalStorage.facet.0d1) [concrete]
// CHECK:STDOUT:   %Optional.Get.specific_fn.0c4: <specific function> = specific_function %Optional.Get.788, @Optional.Get(%OptionalStorage.facet.0d1) [concrete]
// CHECK:STDOUT:   %operator_PlusEqual__carbon_thunk.type: type = fn_type @operator_PlusEqual__carbon_thunk [concrete]
// CHECK:STDOUT:   %operator_PlusEqual__carbon_thunk: %operator_PlusEqual__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.ce5: type = pattern_type %.be1 [concrete]
// CHECK:STDOUT:   %self.param_patt.36c: %pattern_type.ce5 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.548: %pattern_type.ce5 = wrapper_binding_pattern self, %self.param_patt.36c [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc31_37.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.2: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.3: type = fn_type @Destroy.WithSelf.Op.loc31_37.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.3: %Destroy.WithSelf.Op.type.ef016f.3 = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.12c: type = pattern_type %MaybeUnformed.4c8 [concrete]
// CHECK:STDOUT:   %self.param_patt.086: %pattern_type.12c = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.3d3: %pattern_type.12c = wrapper_binding_pattern self, %self.param_patt.086 [concrete]
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
// CHECK:STDOUT:   %pattern_type.34d: type = pattern_type %struct_type.value.has_value.d90 [concrete]
// CHECK:STDOUT:   %self.param_patt.620: %pattern_type.34d = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.868: %pattern_type.34d = wrapper_binding_pattern self, %self.param_patt.620 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.5: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc31_37.4 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.5: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.5 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.6: type = fn_type @Destroy.WithSelf.Op.loc31_37.4 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.6: %Destroy.WithSelf.Op.type.ef016f.6 = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.c01: type = pattern_type %DefaultOptionalStorage.5b0 [concrete]
// CHECK:STDOUT:   %self.param_patt.fb6: %pattern_type.c01 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.ae8: %pattern_type.c01 = wrapper_binding_pattern self, %self.param_patt.fb6 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.6: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc31_37.5 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.6: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.6 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.7: type = fn_type @Destroy.WithSelf.Op.loc31_37.5 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.7: %Destroy.WithSelf.Op.type.ef016f.7 = struct_value () [concrete]
// CHECK:STDOUT:   %self.param_patt.66b: %pattern_type.ece = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.fa3: %pattern_type.ece = wrapper_binding_pattern self, %self.param_patt.66b [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.7: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc31_37.6 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.7: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.7 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.8: type = fn_type @Destroy.WithSelf.Op.loc31_37.6 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.8: %Destroy.WithSelf.Op.type.ef016f.8 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.8: type = fn_type @Destroy.WithSelf.SelfDestruct.loc31_37.6 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.8: %Destroy.WithSelf.SelfDestruct.type.fbceb5.8 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.d0e: <witness> = lookup_impl_witness %impl.elem1.d56, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.726: type = fn_type @Destroy.WithSelf.SelfDestruct.1, @Destroy.WithSelf(%impl.elem1.d56) [symbolic]
// CHECK:STDOUT:   %.a54: type = fn_type_with_self_type %Destroy.WithSelf.SelfDestruct.type.726, %impl.elem1.d56 [symbolic]
// CHECK:STDOUT:   %impl.elem2.9bf: %.a54 = impl_witness_access %Destroy.lookup_impl_witness.d0e, element2 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.2dd: <specific function> = specific_impl_function %impl.elem2.9bf, @Destroy.WithSelf.SelfDestruct.1(%impl.elem1.d56) [symbolic]
// CHECK:STDOUT:   %ConstRange: type = class_type @ConstRange [concrete]
// CHECK:STDOUT:   %pattern_type.c1d: type = pattern_type %ConstRange [concrete]
// CHECK:STDOUT:   %Iterator: type = class_type @Iterator.1 [concrete]
// CHECK:STDOUT:   %Sentinel: type = class_type @Sentinel.1 [concrete]
// CHECK:STDOUT:   %Begin.type: type = fn_type @Begin [concrete]
// CHECK:STDOUT:   %Begin: %Begin.type = struct_value () [concrete]
// CHECK:STDOUT:   %End.type: type = fn_type @End [concrete]
// CHECK:STDOUT:   %End: %End.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.116: <witness> = custom_witness (%Iterator, %Sentinel, %Begin, %End), @CppRangeForIterate [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.9: type = fn_type @Destroy.WithSelf.Op.loc38_16.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.9: %Destroy.WithSelf.Op.type.ef016f.9 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.9: type = fn_type @Destroy.WithSelf.SelfDestruct.loc38_16.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.9: %Destroy.WithSelf.SelfDestruct.type.fbceb5.9 = struct_value () [concrete]
// CHECK:STDOUT:   %Iterator.SubobjectDestroy.type: type = fn_type @Iterator.SubobjectDestroy [concrete]
// CHECK:STDOUT:   %Iterator.SubobjectDestroy: %Iterator.SubobjectDestroy.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.429ac1.1: <witness> = custom_witness (%Destroy.WithSelf.Op.403171.9, %Iterator.SubobjectDestroy, %Destroy.WithSelf.SelfDestruct.db3fdb.9), @Destroy [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.3f6dc2.1: type = fn_type @Iterator.Op.1 [concrete]
// CHECK:STDOUT:   %Iterator.Op.240e50.1: %Iterator.Op.type.3f6dc2.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.5be: <witness> = custom_witness (%ValueType, %Iterator.Op.240e50.1), @CppUnsafeDeref [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.3f6dc2.2: type = fn_type @Iterator.Op.2 [concrete]
// CHECK:STDOUT:   %Iterator.Op.240e50.2: %Iterator.Op.type.3f6dc2.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.c31: <witness> = custom_witness (%Iterator.Op.240e50.2), @Inc [concrete]
// CHECK:STDOUT:   %Equal.type: type = fn_type @Equal [concrete]
// CHECK:STDOUT:   %Equal: %Equal.type = struct_value () [concrete]
// CHECK:STDOUT:   %NotEqual.type: type = fn_type @NotEqual [concrete]
// CHECK:STDOUT:   %NotEqual: %NotEqual.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.bf4: <witness> = custom_witness (%Equal, %NotEqual), @EqWith, @EqWith(%Sentinel) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.10: type = fn_type @Destroy.WithSelf.Op.loc38_16.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.10: %Destroy.WithSelf.Op.type.ef016f.10 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.10: type = fn_type @Destroy.WithSelf.SelfDestruct.loc38_16.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.10: %Destroy.WithSelf.SelfDestruct.type.fbceb5.10 = struct_value () [concrete]
// CHECK:STDOUT:   %Sentinel.SubobjectDestroy.type: type = fn_type @Sentinel.SubobjectDestroy [concrete]
// CHECK:STDOUT:   %Sentinel.SubobjectDestroy: %Sentinel.SubobjectDestroy.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.429ac1.2: <witness> = custom_witness (%Destroy.WithSelf.Op.403171.10, %Sentinel.SubobjectDestroy, %Destroy.WithSelf.SelfDestruct.db3fdb.10), @Destroy [concrete]
// CHECK:STDOUT:   %facet_value.d15: %CppRange_where.type.d682d6.1 = facet_value %ConstRange, (%custom_witness.116, %custom_witness.429ac1.1, %custom_witness.5be, %custom_witness.c31, %custom_witness.bf4, %custom_witness.226, %custom_witness.202, %custom_witness.429ac1.2) [concrete]
// CHECK:STDOUT:   %tuple.type.7a6: type = tuple_type (%Iterator, %Sentinel) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.8: type = fn_type @Destroy.WithSelf.SubobjectDestroy.3 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.8: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.8 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.11: type = fn_type @Destroy.WithSelf.Op.3 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.11: %Destroy.WithSelf.Op.type.ef016f.11 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.11: type = fn_type @Destroy.WithSelf.SelfDestruct.3 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.11: %Destroy.WithSelf.SelfDestruct.type.fbceb5.11 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.f8f19d.8: <witness> = custom_witness (%Destroy.WithSelf.Op.403171.11, %Destroy.WithSelf.SubobjectDestroy.d01daf.8, %Destroy.WithSelf.SelfDestruct.db3fdb.11), @Destroy [concrete]
// CHECK:STDOUT:   %Destroy.facet.d83: %Destroy.type = facet_value %tuple.type.7a6, (%custom_witness.f8f19d.8) [concrete]
// CHECK:STDOUT:   %Iterate.impl_witness.d1d: <witness> = impl_witness imports.%Iterate.impl_witness_table.121, @T.as_type.as.Iterate.impl(%facet_value.d15) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.type.755: type = fn_type @T.as_type.as.Iterate.impl.NewCursor, @T.as_type.as.Iterate.impl(%facet_value.d15) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.437: %T.as_type.as.Iterate.impl.NewCursor.type.755 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.type.b75: type = fn_type @T.as_type.as.Iterate.impl.Next, @T.as_type.as.Iterate.impl(%facet_value.d15) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.82c: %T.as_type.as.Iterate.impl.Next.type.b75 = struct_value () [concrete]
// CHECK:STDOUT:   %Iterate.facet.8d3: %Iterate.type = facet_value %ConstRange, (%Iterate.impl_witness.d1d) [concrete]
// CHECK:STDOUT:   %facet_value.1d8: %Iterate_where.type.87e = facet_value %ConstRange, (%Iterate.impl_witness.d1d) [concrete]
// CHECK:STDOUT:   %r.param_patt.ac5: %pattern_type.c1d = value_param_pattern [concrete]
// CHECK:STDOUT:   %r.patt.97f: %pattern_type.c1d = wrapper_binding_pattern r, %r.param_patt.ac5 [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.7f7: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.8d3) [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.035: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.8d3) [concrete]
// CHECK:STDOUT:   %.8c0: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.7f7, %Iterate.facet.8d3 [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.specific_fn: <specific function> = specific_function %T.as_type.as.Iterate.impl.NewCursor.437, @T.as_type.as.Iterate.impl.NewCursor(%facet_value.d15) [concrete]
// CHECK:STDOUT:   %.b64: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.035, %Iterate.facet.8d3 [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.specific_fn: <specific function> = specific_function %T.as_type.as.Iterate.impl.Next.82c, @T.as_type.as.Iterate.impl.Next(%facet_value.d15) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.000: type = fn_type @Destroy.WithSelf.SelfDestruct.1, @Destroy.WithSelf(%Destroy.facet.d83) [concrete]
// CHECK:STDOUT:   %.dc3: type = fn_type_with_self_type %Destroy.WithSelf.SelfDestruct.type.000, %Destroy.facet.d83 [concrete]
// CHECK:STDOUT:   %complete_type.cf0: <witness> = complete_type_witness %tuple.type.7a6 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .DiffADL = %DiffADL
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.4b2: @Optional.%Optional.HasValue.type (%Optional.HasValue.type.49e) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.HasValue (constants.%Optional.HasValue.c20)]
// CHECK:STDOUT:   %Core.import_ref.52c: @Optional.%Optional.Get.type (%Optional.Get.type.c64) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.Get (constants.%Optional.Get.bcf)]
// CHECK:STDOUT:   %Core.import_ref.ace: type = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%DefaultOptionalStorage (constants.%DefaultOptionalStorage.faa)]
// CHECK:STDOUT:   %Core.import_ref.ef4: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.None.type (%T.as_type.as.OptionalStorage.impl.None.type.925) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.None (constants.%T.as_type.as.OptionalStorage.impl.None.f42)]
// CHECK:STDOUT:   %Core.import_ref.a20: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Some.type (%T.as_type.as.OptionalStorage.impl.Some.type.760) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Some (constants.%T.as_type.as.OptionalStorage.impl.Some.268)]
// CHECK:STDOUT:   %Core.import_ref.af7: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Has.type (%T.as_type.as.OptionalStorage.impl.Has.type.d28) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Has (constants.%T.as_type.as.OptionalStorage.impl.Has.f29)]
// CHECK:STDOUT:   %Core.import_ref.dd4: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Get.type (%T.as_type.as.OptionalStorage.impl.Get.type.2f2) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Get (constants.%T.as_type.as.OptionalStorage.impl.Get.884)]
// CHECK:STDOUT:   %Core.import_ref.554 = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], unloaded
// CHECK:STDOUT:   %OptionalStorage.impl_witness_table.460 = impl_witness_table (%Core.import_ref.ace, %Core.import_ref.ef4, %Core.import_ref.a20, %Core.import_ref.af7, %Core.import_ref.dd4, %Core.import_ref.554), @T.as_type.as.OptionalStorage.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.1d7: @DefaultOptionalStorage.%DefaultOptionalStorage.elem.1 (%DefaultOptionalStorage.elem.9db) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [concrete = %field_decl.5f0]
// CHECK:STDOUT:   %Core.import_ref.929: @DefaultOptionalStorage.%DefaultOptionalStorage.elem.2 (%DefaultOptionalStorage.elem.06a) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [concrete = %field_decl.6a8]
// CHECK:STDOUT:   %Core.import_ref.eef: %facet_type.f48 = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.Iterate.impl.%facet_value (constants.%facet_value.f93)]
// CHECK:STDOUT:   %Core.import_ref.fcc: %Destroy.type = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.Iterate.impl.%Destroy.facet (constants.%Destroy.facet.9cd)]
// CHECK:STDOUT:   %Core.import_ref.946: @T.as_type.as.Iterate.impl.%T.as_type.as.Iterate.impl.NewCursor.type (%T.as_type.as.Iterate.impl.NewCursor.type.8fb) = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.Iterate.impl.%T.as_type.as.Iterate.impl.NewCursor (constants.%T.as_type.as.Iterate.impl.NewCursor.8a1)]
// CHECK:STDOUT:   %Core.import_ref.cfb: @T.as_type.as.Iterate.impl.%T.as_type.as.Iterate.impl.Next.type (%T.as_type.as.Iterate.impl.Next.type.d4a) = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.Iterate.impl.%T.as_type.as.Iterate.impl.Next (constants.%T.as_type.as.Iterate.impl.Next.5a8)]
// CHECK:STDOUT:   %Iterate.impl_witness_table.121 = impl_witness_table (%Core.import_ref.eef, %Core.import_ref.fcc, %Core.import_ref.946, %Core.import_ref.cfb), @T.as_type.as.Iterate.impl [concrete]
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
// CHECK:STDOUT:   %field_decl.6a8: @DefaultOptionalStorage.%DefaultOptionalStorage.elem.2 (%DefaultOptionalStorage.elem.06a) = field_decl has_value, element1, %Core.import_ref.b0d in [concrete] {}
// CHECK:STDOUT:   %Core.import_ref.87f = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %field_decl.5f0: @DefaultOptionalStorage.%DefaultOptionalStorage.elem.1 (%DefaultOptionalStorage.elem.9db) = field_decl value, element0, %Core.import_ref.87f in [concrete] {}
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
// CHECK:STDOUT:     %self.param_patt: %pattern_type.12c = ref_param_pattern [concrete = constants.%self.param_patt.086]
// CHECK:STDOUT:     %self.patt: %pattern_type.12c = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.3d3]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %MaybeUnformed.4c8 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %MaybeUnformed.4c8 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.3: %Destroy.WithSelf.Op.type.ef016f.4 = fn_decl @Destroy.WithSelf.Op.loc31_37.2 [concrete = constants.%Destroy.WithSelf.Op.403171.4] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.12c = ref_param_pattern [concrete = constants.%self.param_patt.086]
// CHECK:STDOUT:     %self.patt: %pattern_type.12c = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.3d3]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %MaybeUnformed.4c8 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %MaybeUnformed.4c8 = wrapper_binding self, %self.param
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
// CHECK:STDOUT:     %self.param_patt: %pattern_type.34d = ref_param_pattern [concrete = constants.%self.param_patt.620]
// CHECK:STDOUT:     %self.patt: %pattern_type.34d = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.868]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %struct_type.value.has_value.d90 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %struct_type.value.has_value.d90 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.5: %Destroy.WithSelf.Op.type.ef016f.6 = fn_decl @Destroy.WithSelf.Op.loc31_37.4 [concrete = constants.%Destroy.WithSelf.Op.403171.6] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.34d = ref_param_pattern [concrete = constants.%self.param_patt.620]
// CHECK:STDOUT:     %self.patt: %pattern_type.34d = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.868]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %struct_type.value.has_value.d90 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %struct_type.value.has_value.d90 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.5: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.6 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc31_37.5 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.6] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.c01 = ref_param_pattern [concrete = constants.%self.param_patt.fb6]
// CHECK:STDOUT:     %self.patt: %pattern_type.c01 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.ae8]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %DefaultOptionalStorage.5b0 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %DefaultOptionalStorage.5b0 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.6: %Destroy.WithSelf.Op.type.ef016f.7 = fn_decl @Destroy.WithSelf.Op.loc31_37.5 [concrete = constants.%Destroy.WithSelf.Op.403171.7] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.c01 = ref_param_pattern [concrete = constants.%self.param_patt.fb6]
// CHECK:STDOUT:     %self.patt: %pattern_type.c01 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.ae8]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %DefaultOptionalStorage.5b0 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %DefaultOptionalStorage.5b0 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.6: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.7 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc31_37.6 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.7] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.ece = ref_param_pattern [concrete = constants.%self.param_patt.66b]
// CHECK:STDOUT:     %self.patt: %pattern_type.ece = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.fa3]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %Optional.873 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %Optional.873 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.7: %Destroy.WithSelf.Op.type.ef016f.8 = fn_decl @Destroy.WithSelf.Op.loc31_37.6 [concrete = constants.%Destroy.WithSelf.Op.403171.8] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.ece = ref_param_pattern [concrete = constants.%self.param_patt.66b]
// CHECK:STDOUT:     %self.patt: %pattern_type.ece = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.fa3]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %Optional.873 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %Optional.873 = wrapper_binding self, %self.param
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
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.loc31_37.5) [symbolic = %Iterate.WithSelf.NewCursor.type (constants.%Iterate.WithSelf.NewCursor.type.b87)]
// CHECK:STDOUT:   %.loc31_37.20: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type, %Iterate.facet.loc31_37.5 [symbolic = %.loc31_37.20 (constants.%.c11)]
// CHECK:STDOUT:   %impl.elem2.loc31_37.3: @TestIterate.%.loc31_37.20 (%.c11) = impl_witness_access %Iterate.lookup_impl_witness, element2 [symbolic = %impl.elem2.loc31_37.3 (constants.%impl.elem2.978)]
// CHECK:STDOUT:   %specific_impl_fn.loc31_37.4: <specific function> = specific_impl_function %impl.elem2.loc31_37.3, @Iterate.WithSelf.NewCursor(%Iterate.facet.loc31_37.5) [symbolic = %specific_impl_fn.loc31_37.4 (constants.%specific_impl_fn.f0e)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.loc31_37.5) [symbolic = %Iterate.WithSelf.Next.type (constants.%Iterate.WithSelf.Next.type.526)]
// CHECK:STDOUT:   %.loc31_37.21: type = fn_type_with_self_type %Iterate.WithSelf.Next.type, %Iterate.facet.loc31_37.5 [symbolic = %.loc31_37.21 (constants.%.118)]
// CHECK:STDOUT:   %impl.elem3.loc31_37.2: @TestIterate.%.loc31_37.21 (%.118) = impl_witness_access %Iterate.lookup_impl_witness, element3 [symbolic = %impl.elem3.loc31_37.2 (constants.%impl.elem3.2b1)]
// CHECK:STDOUT:   %specific_impl_fn.loc31_37.5: <specific function> = specific_impl_function %impl.elem3.loc31_37.2, @Iterate.WithSelf.Next(%Iterate.facet.loc31_37.5) [symbolic = %specific_impl_fn.loc31_37.5 (constants.%specific_impl_fn.3f1)]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type: type = fn_type @Destroy.WithSelf.SelfDestruct.1, @Destroy.WithSelf(%impl.elem1) [symbolic = %Destroy.WithSelf.SelfDestruct.type (constants.%Destroy.WithSelf.SelfDestruct.type.726)]
// CHECK:STDOUT:   %.loc31_37.22: type = fn_type_with_self_type %Destroy.WithSelf.SelfDestruct.type, %impl.elem1 [symbolic = %.loc31_37.22 (constants.%.a54)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %impl.elem1, @Destroy [symbolic = %Destroy.lookup_impl_witness (constants.%Destroy.lookup_impl_witness.d0e)]
// CHECK:STDOUT:   %impl.elem2.loc31_37.4: @TestIterate.%.loc31_37.22 (%.a54) = impl_witness_access %Destroy.lookup_impl_witness, element2 [symbolic = %impl.elem2.loc31_37.4 (constants.%impl.elem2.9bf)]
// CHECK:STDOUT:   %specific_impl_fn.loc31_37.6: <specific function> = specific_impl_function %impl.elem2.loc31_37.4, @Destroy.WithSelf.SelfDestruct.1(%impl.elem1) [symbolic = %specific_impl_fn.loc31_37.6 (constants.%specific_impl_fn.2dd)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%r.param: @TestIterate.%R.as_type.loc27_79.1 (%R.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %i.patt: %pattern_type.c43 = value_binding_pattern i [concrete = constants.%i.patt]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %r.ref: @TestIterate.%R.as_type.loc27_79.1 (%R.as_type) = name_ref r, %r
// CHECK:STDOUT:     %impl.elem2.loc31_37.1: @TestIterate.%.loc31_37.20 (%.c11) = impl_witness_access constants.%Iterate.lookup_impl_witness.01e, element2 [symbolic = %impl.elem2.loc31_37.3 (constants.%impl.elem2.978)]
// CHECK:STDOUT:     %bound_method.loc31_37.1: <bound method> = bound_method %r.ref, %impl.elem2.loc31_37.1
// CHECK:STDOUT:     %Iterate.facet.loc31_37.1: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.01e) [symbolic = %Iterate.facet.loc31_37.5 (constants.%Iterate.facet.467)]
// CHECK:STDOUT:     %.loc31_37.1: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc31_37.1 [symbolic = %Iterate.facet.loc31_37.5 (constants.%Iterate.facet.467)]
// CHECK:STDOUT:     %Iterate.facet.loc31_37.2: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.01e) [symbolic = %Iterate.facet.loc31_37.5 (constants.%Iterate.facet.467)]
// CHECK:STDOUT:     %.loc31_37.2: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc31_37.2 [symbolic = %Iterate.facet.loc31_37.5 (constants.%Iterate.facet.467)]
// CHECK:STDOUT:     %specific_impl_fn.loc31_37.1: <specific function> = specific_impl_function %impl.elem2.loc31_37.1, @Iterate.WithSelf.NewCursor(constants.%Iterate.facet.467) [symbolic = %specific_impl_fn.loc31_37.4 (constants.%specific_impl_fn.f0e)]
// CHECK:STDOUT:     %bound_method.loc31_37.2: <bound method> = bound_method %r.ref, %specific_impl_fn.loc31_37.1
// CHECK:STDOUT:     %var: ref @TestIterate.%as_type (%as_type.db9) = var_storage invalid
// CHECK:STDOUT:     %Iterate.WithSelf.NewCursor.call: init @TestIterate.%as_type (%as_type.db9) to %var = call %bound_method.loc31_37.2(%r.ref)
// CHECK:STDOUT:     assign %var, %Iterate.WithSelf.NewCursor.call
// CHECK:STDOUT:     br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.next:
// CHECK:STDOUT:     %impl.elem3.loc31_37.1: @TestIterate.%.loc31_37.21 (%.118) = impl_witness_access constants.%Iterate.lookup_impl_witness.01e, element3 [symbolic = %impl.elem3.loc31_37.2 (constants.%impl.elem3.2b1)]
// CHECK:STDOUT:     %bound_method.loc31_37.3: <bound method> = bound_method %r.ref, %impl.elem3.loc31_37.1
// CHECK:STDOUT:     %Iterate.facet.loc31_37.3: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.01e) [symbolic = %Iterate.facet.loc31_37.5 (constants.%Iterate.facet.467)]
// CHECK:STDOUT:     %.loc31_37.3: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc31_37.3 [symbolic = %Iterate.facet.loc31_37.5 (constants.%Iterate.facet.467)]
// CHECK:STDOUT:     %Iterate.facet.loc31_37.4: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.01e) [symbolic = %Iterate.facet.loc31_37.5 (constants.%Iterate.facet.467)]
// CHECK:STDOUT:     %.loc31_37.4: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc31_37.4 [symbolic = %Iterate.facet.loc31_37.5 (constants.%Iterate.facet.467)]
// CHECK:STDOUT:     %.loc31_37.5: %Destroy.type = converted constants.%as_type.db9, constants.%impl.elem1.d56 [symbolic = %impl.elem1 (constants.%impl.elem1.d56)]
// CHECK:STDOUT:     %.loc31_37.6: %Destroy.type = converted constants.%as_type.db9, constants.%impl.elem1.d56 [symbolic = %impl.elem1 (constants.%impl.elem1.d56)]
// CHECK:STDOUT:     %specific_impl_fn.loc31_37.2: <specific function> = specific_impl_function %impl.elem3.loc31_37.1, @Iterate.WithSelf.Next(constants.%Iterate.facet.467) [symbolic = %specific_impl_fn.loc31_37.5 (constants.%specific_impl_fn.3f1)]
// CHECK:STDOUT:     %bound_method.loc31_37.4: <bound method> = bound_method %r.ref, %specific_impl_fn.loc31_37.2
// CHECK:STDOUT:     %.loc31_37.7: ref %Optional.873 = temporary_storage
// CHECK:STDOUT:     %Iterate.WithSelf.Next.call: init %Optional.873 to %.loc31_37.7 = call %bound_method.loc31_37.4(%r.ref, %var)
// CHECK:STDOUT:     %.loc31_37.8: ref %Optional.873 = temporary %.loc31_37.7, %Iterate.WithSelf.Next.call
// CHECK:STDOUT:     %.loc31_37.9: %Optional.HasValue.type.1e1 = specific_constant imports.%Core.import_ref.4b2, @Optional(constants.%OptionalStorage.facet.0d1) [concrete = constants.%Optional.HasValue.6d8]
// CHECK:STDOUT:     %HasValue.ref: %Optional.HasValue.type.1e1 = name_ref HasValue, %.loc31_37.9 [concrete = constants.%Optional.HasValue.6d8]
// CHECK:STDOUT:     %Optional.HasValue.bound: <bound method> = bound_method %.loc31_37.8, %HasValue.ref
// CHECK:STDOUT:     %Optional.HasValue.specific_fn: <specific function> = specific_function %HasValue.ref, @Optional.HasValue(constants.%OptionalStorage.facet.0d1) [concrete = constants.%Optional.HasValue.specific_fn.921]
// CHECK:STDOUT:     %bound_method.loc31_37.5: <bound method> = bound_method %.loc31_37.8, %Optional.HasValue.specific_fn
// CHECK:STDOUT:     %.loc31_37.10: %Optional.873 = acquire_value %.loc31_37.8
// CHECK:STDOUT:     %Optional.HasValue.call: init bool = call %bound_method.loc31_37.5(%.loc31_37.10)
// CHECK:STDOUT:     %.loc31_37.11: bool = value_of_initializer %Optional.HasValue.call
// CHECK:STDOUT:     %.loc31_37.12: bool = converted %Optional.HasValue.call, %.loc31_37.11
// CHECK:STDOUT:     if %.loc31_37.12 br !for.body else br !for.done
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.body:
// CHECK:STDOUT:     %.loc31_37.13: %Optional.Get.type.915 = specific_constant imports.%Core.import_ref.52c, @Optional(constants.%OptionalStorage.facet.0d1) [concrete = constants.%Optional.Get.788]
// CHECK:STDOUT:     %Get.ref: %Optional.Get.type.915 = name_ref Get, %.loc31_37.13 [concrete = constants.%Optional.Get.788]
// CHECK:STDOUT:     %Optional.Get.bound: <bound method> = bound_method %.loc31_37.8, %Get.ref
// CHECK:STDOUT:     %Optional.Get.specific_fn: <specific function> = specific_function %Get.ref, @Optional.Get(constants.%OptionalStorage.facet.0d1) [concrete = constants.%Optional.Get.specific_fn.0c4]
// CHECK:STDOUT:     %bound_method.loc31_37.6: <bound method> = bound_method %.loc31_37.8, %Optional.Get.specific_fn
// CHECK:STDOUT:     %.loc31_37.14: ref %ValueType = temporary_storage
// CHECK:STDOUT:     %.loc31_37.15: %Optional.873 = acquire_value %.loc31_37.8
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
// CHECK:STDOUT:     %impl.elem2.loc31_37.2: @TestIterate.%.loc31_37.22 (%.a54) = impl_witness_access constants.%Destroy.lookup_impl_witness.d0e, element2 [symbolic = %impl.elem2.loc31_37.4 (constants.%impl.elem2.9bf)]
// CHECK:STDOUT:     %bound_method.loc31_37.7: <bound method> = bound_method %var, %impl.elem2.loc31_37.2
// CHECK:STDOUT:     %.loc31_37.18: %Destroy.type = converted constants.%as_type.db9, constants.%impl.elem1.d56 [symbolic = %impl.elem1 (constants.%impl.elem1.d56)]
// CHECK:STDOUT:     %.loc31_37.19: %Destroy.type = converted constants.%as_type.db9, constants.%impl.elem1.d56 [symbolic = %impl.elem1 (constants.%impl.elem1.d56)]
// CHECK:STDOUT:     %specific_impl_fn.loc31_37.3: <specific function> = specific_impl_function %impl.elem2.loc31_37.2, @Destroy.WithSelf.SelfDestruct.1(constants.%impl.elem1.d56) [symbolic = %specific_impl_fn.loc31_37.6 (constants.%specific_impl_fn.2dd)]
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
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc31_37.2(%self.param: ref %MaybeUnformed.4c8) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc31_37.2(%self.param: ref %MaybeUnformed.4c8) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc31_37.2(%self.param: ref %MaybeUnformed.4c8) {
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
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc31_37.4(%self.param: ref %struct_type.value.has_value.d90) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc31_37.1: ref bool = struct_access %self.param, element1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc31_37.1: <bound method> = bound_method %.loc31_37.1, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.5
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc31_37.1: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc31_37.1(%.loc31_37.1)
// CHECK:STDOUT:   %.loc31_37.2: ref %MaybeUnformed.4c8 = struct_access %self.param, element0
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc31_37.2: <bound method> = bound_method %.loc31_37.2, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.4
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc31_37.2: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc31_37.2(%.loc31_37.2)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc31_37.4(%self.param: ref %struct_type.value.has_value.d90) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc31_37.4(%self.param: ref %struct_type.value.has_value.d90) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.5(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.4(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc31_37.5(%self.param: ref %DefaultOptionalStorage.5b0) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %has_value.ref: %DefaultOptionalStorage.elem.917 = name_ref has_value, imports.%Core.import_ref.929 [concrete = imports.%field_decl.6a8]
// CHECK:STDOUT:   %.loc31_37.1: ref bool = class_element_access %self.param, element1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc31_37.1: <bound method> = bound_method %.loc31_37.1, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.5
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc31_37.1: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc31_37.1(%.loc31_37.1)
// CHECK:STDOUT:   %value.ref: %DefaultOptionalStorage.elem.cbb = name_ref value, imports.%Core.import_ref.1d7 [concrete = imports.%field_decl.5f0]
// CHECK:STDOUT:   %.loc31_37.2: ref %MaybeUnformed.4c8 = class_element_access %self.param, element0
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc31_37.2: <bound method> = bound_method %.loc31_37.2, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.4
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc31_37.2: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc31_37.2(%.loc31_37.2)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc31_37.5(%self.param: ref %DefaultOptionalStorage.5b0) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc31_37.5(%self.param: ref %DefaultOptionalStorage.5b0) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.6(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.5(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc31_37.6(%self.param: ref %Optional.873) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc31_37.6(%self.param: ref %Optional.873) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc31_37.6(%self.param: ref %Optional.873) {
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
// CHECK:STDOUT:   %impl.elem2: %.8c0 = impl_witness_access constants.%Iterate.impl_witness.d1d, element2 [concrete = constants.%T.as_type.as.Iterate.impl.NewCursor.437]
// CHECK:STDOUT:   %bound_method.loc45_37.1: <bound method> = bound_method %c.ref, %impl.elem2
// CHECK:STDOUT:   %facet_value.loc45_37.1: %CppRange_where.type.d682d6.1 = facet_value constants.%ConstRange, (constants.%custom_witness.116, constants.%custom_witness.429ac1.1, constants.%custom_witness.5be, constants.%custom_witness.c31, constants.%custom_witness.bf4, constants.%custom_witness.226, constants.%custom_witness.202, constants.%custom_witness.429ac1.2) [concrete = constants.%facet_value.d15]
// CHECK:STDOUT:   %.loc45_37.1: %CppRange_where.type.d682d6.1 = converted constants.%ConstRange, %facet_value.loc45_37.1 [concrete = constants.%facet_value.d15]
// CHECK:STDOUT:   %facet_value.loc45_37.2: %CppRange_where.type.d682d6.1 = facet_value constants.%ConstRange, (constants.%custom_witness.116, constants.%custom_witness.429ac1.1, constants.%custom_witness.5be, constants.%custom_witness.c31, constants.%custom_witness.bf4, constants.%custom_witness.226, constants.%custom_witness.202, constants.%custom_witness.429ac1.2) [concrete = constants.%facet_value.d15]
// CHECK:STDOUT:   %.loc45_37.2: %CppRange_where.type.d682d6.1 = converted constants.%ConstRange, %facet_value.loc45_37.2 [concrete = constants.%facet_value.d15]
// CHECK:STDOUT:   %specific_fn.loc45_37.1: <specific function> = specific_function %impl.elem2, @T.as_type.as.Iterate.impl.NewCursor(constants.%facet_value.d15) [concrete = constants.%T.as_type.as.Iterate.impl.NewCursor.specific_fn]
// CHECK:STDOUT:   %bound_method.loc45_37.2: <bound method> = bound_method %c.ref, %specific_fn.loc45_37.1
// CHECK:STDOUT:   %var: ref %tuple.type.7a6 = var_storage invalid
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.call: init %tuple.type.7a6 to %var = call %bound_method.loc45_37.2(%c.ref)
// CHECK:STDOUT:   assign %var, %T.as_type.as.Iterate.impl.NewCursor.call
// CHECK:STDOUT:   br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT: !for.next:
// CHECK:STDOUT:   %impl.elem3: %.b64 = impl_witness_access constants.%Iterate.impl_witness.d1d, element3 [concrete = constants.%T.as_type.as.Iterate.impl.Next.82c]
// CHECK:STDOUT:   %bound_method.loc45_37.3: <bound method> = bound_method %c.ref, %impl.elem3
// CHECK:STDOUT:   %facet_value.loc45_37.3: %CppRange_where.type.d682d6.1 = facet_value constants.%ConstRange, (constants.%custom_witness.116, constants.%custom_witness.429ac1.1, constants.%custom_witness.5be, constants.%custom_witness.c31, constants.%custom_witness.bf4, constants.%custom_witness.226, constants.%custom_witness.202, constants.%custom_witness.429ac1.2) [concrete = constants.%facet_value.d15]
// CHECK:STDOUT:   %.loc45_37.3: %CppRange_where.type.d682d6.1 = converted constants.%ConstRange, %facet_value.loc45_37.3 [concrete = constants.%facet_value.d15]
// CHECK:STDOUT:   %facet_value.loc45_37.4: %CppRange_where.type.d682d6.1 = facet_value constants.%ConstRange, (constants.%custom_witness.116, constants.%custom_witness.429ac1.1, constants.%custom_witness.5be, constants.%custom_witness.c31, constants.%custom_witness.bf4, constants.%custom_witness.226, constants.%custom_witness.202, constants.%custom_witness.429ac1.2) [concrete = constants.%facet_value.d15]
// CHECK:STDOUT:   %.loc45_37.4: %CppRange_where.type.d682d6.1 = converted constants.%ConstRange, %facet_value.loc45_37.4 [concrete = constants.%facet_value.d15]
// CHECK:STDOUT:   %specific_fn.loc45_37.2: <specific function> = specific_function %impl.elem3, @T.as_type.as.Iterate.impl.Next(constants.%facet_value.d15) [concrete = constants.%T.as_type.as.Iterate.impl.Next.specific_fn]
// CHECK:STDOUT:   %bound_method.loc45_37.4: <bound method> = bound_method %c.ref, %specific_fn.loc45_37.2
// CHECK:STDOUT:   %.loc45_37.5: ref %Optional.873 = temporary_storage
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.call: init %Optional.873 to %.loc45_37.5 = call %bound_method.loc45_37.4(%c.ref, %var)
// CHECK:STDOUT:   %.loc45_37.6: ref %Optional.873 = temporary %.loc45_37.5, %T.as_type.as.Iterate.impl.Next.call
// CHECK:STDOUT:   %.loc45_37.7: %Optional.HasValue.type.1e1 = specific_constant imports.%Core.import_ref.4b2, @Optional(constants.%OptionalStorage.facet.0d1) [concrete = constants.%Optional.HasValue.6d8]
// CHECK:STDOUT:   %HasValue.ref: %Optional.HasValue.type.1e1 = name_ref HasValue, %.loc45_37.7 [concrete = constants.%Optional.HasValue.6d8]
// CHECK:STDOUT:   %Optional.HasValue.bound: <bound method> = bound_method %.loc45_37.6, %HasValue.ref
// CHECK:STDOUT:   %Optional.HasValue.specific_fn: <specific function> = specific_function %HasValue.ref, @Optional.HasValue(constants.%OptionalStorage.facet.0d1) [concrete = constants.%Optional.HasValue.specific_fn.921]
// CHECK:STDOUT:   %bound_method.loc45_37.5: <bound method> = bound_method %.loc45_37.6, %Optional.HasValue.specific_fn
// CHECK:STDOUT:   %.loc45_37.8: %Optional.873 = acquire_value %.loc45_37.6
// CHECK:STDOUT:   %Optional.HasValue.call: init bool = call %bound_method.loc45_37.5(%.loc45_37.8)
// CHECK:STDOUT:   %.loc45_37.9: bool = value_of_initializer %Optional.HasValue.call
// CHECK:STDOUT:   %.loc45_37.10: bool = converted %Optional.HasValue.call, %.loc45_37.9
// CHECK:STDOUT:   if %.loc45_37.10 br !for.body else br !for.done
// CHECK:STDOUT:
// CHECK:STDOUT: !for.body:
// CHECK:STDOUT:   %.loc45_37.11: %Optional.Get.type.915 = specific_constant imports.%Core.import_ref.52c, @Optional(constants.%OptionalStorage.facet.0d1) [concrete = constants.%Optional.Get.788]
// CHECK:STDOUT:   %Get.ref: %Optional.Get.type.915 = name_ref Get, %.loc45_37.11 [concrete = constants.%Optional.Get.788]
// CHECK:STDOUT:   %Optional.Get.bound: <bound method> = bound_method %.loc45_37.6, %Get.ref
// CHECK:STDOUT:   %Optional.Get.specific_fn: <specific function> = specific_function %Get.ref, @Optional.Get(constants.%OptionalStorage.facet.0d1) [concrete = constants.%Optional.Get.specific_fn.0c4]
// CHECK:STDOUT:   %bound_method.loc45_37.6: <bound method> = bound_method %.loc45_37.6, %Optional.Get.specific_fn
// CHECK:STDOUT:   %.loc45_37.12: ref %ValueType = temporary_storage
// CHECK:STDOUT:   %.loc45_37.13: %Optional.873 = acquire_value %.loc45_37.6
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
// CHECK:STDOUT: specific @TestIterate(constants.%facet_value.1d8) {
// CHECK:STDOUT:   %R.patt.loc27_17.2 => constants.%R.patt
// CHECK:STDOUT:   %R.loc27_17.1 => constants.%facet_value.1d8
// CHECK:STDOUT:   %R.as_type.loc27_79.1 => constants.%ConstRange
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.c1d
// CHECK:STDOUT:   %r.param_patt.loc27_77.2 => constants.%r.param_patt.ac5
// CHECK:STDOUT:   %r.patt.loc27_77.2 => constants.%r.patt.97f
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc27 => constants.%complete_type.357
// CHECK:STDOUT:   %Iterate.lookup_impl_witness => constants.%Iterate.impl_witness.d1d
// CHECK:STDOUT:   %Iterate.facet.loc31_37.5 => constants.%Iterate.facet.8d3
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type => constants.%Iterate.WithSelf.NewCursor.type.7f7
// CHECK:STDOUT:   %.loc31_37.20 => constants.%.8c0
// CHECK:STDOUT:   %impl.elem2.loc31_37.3 => constants.%T.as_type.as.Iterate.impl.NewCursor.437
// CHECK:STDOUT:   %specific_impl_fn.loc31_37.4 => constants.%T.as_type.as.Iterate.impl.NewCursor.specific_fn
// CHECK:STDOUT:   %impl.elem1 => constants.%Destroy.facet.d83
// CHECK:STDOUT:   %as_type => constants.%tuple.type.7a6
// CHECK:STDOUT:   %require_complete.1 => constants.%complete_type.cf0
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type => constants.%Iterate.WithSelf.Next.type.035
// CHECK:STDOUT:   %.loc31_37.21 => constants.%.b64
// CHECK:STDOUT:   %impl.elem3.loc31_37.2 => constants.%T.as_type.as.Iterate.impl.Next.82c
// CHECK:STDOUT:   %specific_impl_fn.loc31_37.5 => constants.%T.as_type.as.Iterate.impl.Next.specific_fn
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type => constants.%Destroy.WithSelf.SelfDestruct.type.000
// CHECK:STDOUT:   %.loc31_37.22 => constants.%.dc3
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
// CHECK:STDOUT:   %Optional.Get.type.c64: type = fn_type @Optional.Get, @Optional(%T.220) [symbolic]
// CHECK:STDOUT:   %Optional.Get.bcf: %Optional.Get.type.c64 = struct_value () [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.type.49e: type = fn_type @Optional.HasValue, @Optional(%T.220) [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.c20: %Optional.HasValue.type.49e = struct_value () [symbolic]
// CHECK:STDOUT:   %T.283: %facet_type.f48 = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.74b: type = facet_access_type %T.283 [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.28a: <witness> = lookup_impl_witness %T.283, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.16e: %Destroy.type = facet_value %T.as_type.74b, (%Destroy.lookup_impl_witness.28a) [symbolic]
// CHECK:STDOUT:   %MaybeUnformed.9b7: type = class_type @MaybeUnformed, @MaybeUnformed(%Destroy.facet.16e) [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.faa: type = class_type @DefaultOptionalStorage, @DefaultOptionalStorage(%T.283) [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.elem.06a: type = unbound_element_type %DefaultOptionalStorage.faa, bool [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.elem.9db: type = unbound_element_type %DefaultOptionalStorage.faa, %MaybeUnformed.9b7 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Get.type.2f2: type = fn_type @T.as_type.as.OptionalStorage.impl.Get, @T.as_type.as.OptionalStorage.impl(%T.283) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Get.884: %T.as_type.as.OptionalStorage.impl.Get.type.2f2 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Has.type.d28: type = fn_type @T.as_type.as.OptionalStorage.impl.Has, @T.as_type.as.OptionalStorage.impl(%T.283) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Has.f29: %T.as_type.as.OptionalStorage.impl.Has.type.d28 = struct_value () [symbolic]
// CHECK:STDOUT:   %.Self.500: %Iterate.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %ValueType: type = class_type @ValueType [concrete]
// CHECK:STDOUT:   %pattern_type.d2d: type = pattern_type %ValueType [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.2: type = fn_type @Destroy.WithSelf.Op.loc49 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.2: %Destroy.WithSelf.Op.type.ef016f.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.2: type = fn_type @Destroy.WithSelf.SelfDestruct.loc49 [concrete]
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
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.f6c: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.04e) [symbolic]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.d76: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.04e) [symbolic]
// CHECK:STDOUT:   %.2577: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.f6c, %Iterate.facet.04e [symbolic]
// CHECK:STDOUT:   %impl.elem2.f11: %.2577 = impl_witness_access %Iterate.lookup_impl_witness.711, element2 [symbolic]
// CHECK:STDOUT:   %impl.elem1.0e0: %Destroy.type = impl_witness_access %Iterate.lookup_impl_witness.711, element1 [symbolic]
// CHECK:STDOUT:   %as_type.b30: type = facet_access_type %impl.elem1.0e0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.d03: <specific function> = specific_impl_function %impl.elem2.f11, @Iterate.WithSelf.NewCursor(%Iterate.facet.04e) [symbolic]
// CHECK:STDOUT:   %.894: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.d76, %Iterate.facet.04e [symbolic]
// CHECK:STDOUT:   %impl.elem3.eed: %.894 = impl_witness_access %Iterate.lookup_impl_witness.711, element3 [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.022: type = class_type @DefaultOptionalStorage, @DefaultOptionalStorage(%facet_value.14c) [concrete]
// CHECK:STDOUT:   %OptionalStorage.impl_witness.90a: <witness> = impl_witness imports.%OptionalStorage.impl_witness_table.460, @T.as_type.as.OptionalStorage.impl(%facet_value.14c) [concrete]
// CHECK:STDOUT:   %OptionalStorage.facet.c95: %OptionalStorage.type = facet_value %ValueType, (%OptionalStorage.impl_witness.90a) [concrete]
// CHECK:STDOUT:   %Optional.61e: type = class_type @Optional, @Optional(%OptionalStorage.facet.c95) [concrete]
// CHECK:STDOUT:   %pattern_type.5a0: type = pattern_type %Optional.61e [concrete]
// CHECK:STDOUT:   %specific_impl_fn.303: <specific function> = specific_impl_function %impl.elem3.eed, @Iterate.WithSelf.Next(%Iterate.facet.04e) [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.type.dd2: type = fn_type @Optional.HasValue, @Optional(%OptionalStorage.facet.c95) [concrete]
// CHECK:STDOUT:   %Optional.HasValue.e45: %Optional.HasValue.type.dd2 = struct_value () [concrete]
// CHECK:STDOUT:   %Optional.Get.type.caf: type = fn_type @Optional.Get, @Optional(%OptionalStorage.facet.c95) [concrete]
// CHECK:STDOUT:   %Optional.Get.04c: %Optional.Get.type.caf = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.facet.6cb: %Destroy.type = facet_value %ValueType, (%custom_witness.3ec) [concrete]
// CHECK:STDOUT:   %MaybeUnformed.385: type = class_type @MaybeUnformed, @MaybeUnformed(%Destroy.facet.6cb) [concrete]
// CHECK:STDOUT:   %.dd1: type = maybe_unformed_type %ValueType [concrete]
// CHECK:STDOUT:   %DefaultOptionalStorage.elem.dae: type = unbound_element_type %DefaultOptionalStorage.022, %MaybeUnformed.385 [concrete]
// CHECK:STDOUT:   %DefaultOptionalStorage.elem.80f: type = unbound_element_type %DefaultOptionalStorage.022, bool [concrete]
// CHECK:STDOUT:   %struct_type.value.has_value.03f: type = struct_type {.value: %MaybeUnformed.385, .has_value: bool} [concrete]
// CHECK:STDOUT:   %Optional.HasValue.specific_fn: <specific function> = specific_function %Optional.HasValue.e45, @Optional.HasValue(%OptionalStorage.facet.c95) [concrete]
// CHECK:STDOUT:   %Optional.Get.specific_fn: <specific function> = specific_function %Optional.Get.04c, @Optional.Get(%OptionalStorage.facet.c95) [concrete]
// CHECK:STDOUT:   %operator_PlusEqual__carbon_thunk.type: type = fn_type @operator_PlusEqual__carbon_thunk [concrete]
// CHECK:STDOUT:   %operator_PlusEqual__carbon_thunk: %operator_PlusEqual__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.1d1: type = pattern_type %.dd1 [concrete]
// CHECK:STDOUT:   %self.param_patt.964: %pattern_type.1d1 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.82f: %pattern_type.1d1 = wrapper_binding_pattern self, %self.param_patt.964 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc53_44.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.2: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.3: type = fn_type @Destroy.WithSelf.Op.loc53_44.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.3: %Destroy.WithSelf.Op.type.ef016f.3 = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.0da: type = pattern_type %MaybeUnformed.385 [concrete]
// CHECK:STDOUT:   %self.param_patt.623: %pattern_type.0da = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.7ed: %pattern_type.0da = wrapper_binding_pattern self, %self.param_patt.623 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.3: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc53_44.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.3: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.3 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.4: type = fn_type @Destroy.WithSelf.Op.loc53_44.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.4: %Destroy.WithSelf.Op.type.ef016f.4 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.4: type = fn_type @Destroy.WithSelf.SelfDestruct.loc53_44.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.4: %Destroy.WithSelf.SelfDestruct.type.fbceb5.4 = struct_value () [concrete]
// CHECK:STDOUT:   %self.param_patt.40d: %pattern_type.831 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.560: %pattern_type.831 = wrapper_binding_pattern self, %self.param_patt.40d [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.4: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc53_44.3 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.4: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.4 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.5: type = fn_type @Destroy.WithSelf.Op.loc53_44.3 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.5: %Destroy.WithSelf.Op.type.ef016f.5 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.5: type = fn_type @Destroy.WithSelf.SelfDestruct.loc53_44.3 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.5: %Destroy.WithSelf.SelfDestruct.type.fbceb5.5 = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.cdb: type = pattern_type %struct_type.value.has_value.03f [concrete]
// CHECK:STDOUT:   %self.param_patt.31f: %pattern_type.cdb = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.692: %pattern_type.cdb = wrapper_binding_pattern self, %self.param_patt.31f [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.5: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc53_44.4 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.5: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.5 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.6: type = fn_type @Destroy.WithSelf.Op.loc53_44.4 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.6: %Destroy.WithSelf.Op.type.ef016f.6 = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.b4b: type = pattern_type %DefaultOptionalStorage.022 [concrete]
// CHECK:STDOUT:   %self.param_patt.48e: %pattern_type.b4b = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.724: %pattern_type.b4b = wrapper_binding_pattern self, %self.param_patt.48e [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.6: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc53_44.5 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.6: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.6 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.7: type = fn_type @Destroy.WithSelf.Op.loc53_44.5 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.7: %Destroy.WithSelf.Op.type.ef016f.7 = struct_value () [concrete]
// CHECK:STDOUT:   %self.param_patt.d85: %pattern_type.5a0 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.a14: %pattern_type.5a0 = wrapper_binding_pattern self, %self.param_patt.d85 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.7: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc53_44.6 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.7: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.7 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.8: type = fn_type @Destroy.WithSelf.Op.loc53_44.6 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.8: %Destroy.WithSelf.Op.type.ef016f.8 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.8: type = fn_type @Destroy.WithSelf.SelfDestruct.loc53_44.6 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.8: %Destroy.WithSelf.SelfDestruct.type.fbceb5.8 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.696: <witness> = lookup_impl_witness %impl.elem1.0e0, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.55d: type = fn_type @Destroy.WithSelf.SelfDestruct.1, @Destroy.WithSelf(%impl.elem1.0e0) [symbolic]
// CHECK:STDOUT:   %.a44: type = fn_type_with_self_type %Destroy.WithSelf.SelfDestruct.type.55d, %impl.elem1.0e0 [symbolic]
// CHECK:STDOUT:   %impl.elem2.1f1: %.a44 = impl_witness_access %Destroy.lookup_impl_witness.696, element2 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.cb8: <specific function> = specific_impl_function %impl.elem2.1f1, @Destroy.WithSelf.SelfDestruct.1(%impl.elem1.0e0) [symbolic]
// CHECK:STDOUT:   %MutableRange: type = class_type @MutableRange [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .DiffMutableADL = %DiffMutableADL
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.4b2: @Optional.%Optional.HasValue.type (%Optional.HasValue.type.49e) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.HasValue (constants.%Optional.HasValue.c20)]
// CHECK:STDOUT:   %Core.import_ref.52c: @Optional.%Optional.Get.type (%Optional.Get.type.c64) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.Get (constants.%Optional.Get.bcf)]
// CHECK:STDOUT:   %Core.import_ref.ace: type = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%DefaultOptionalStorage (constants.%DefaultOptionalStorage.faa)]
// CHECK:STDOUT:   %Core.import_ref.063 = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], unloaded
// CHECK:STDOUT:   %Core.import_ref.4b4 = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], unloaded
// CHECK:STDOUT:   %Core.import_ref.af7: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Has.type (%T.as_type.as.OptionalStorage.impl.Has.type.d28) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Has (constants.%T.as_type.as.OptionalStorage.impl.Has.f29)]
// CHECK:STDOUT:   %Core.import_ref.dd4: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Get.type (%T.as_type.as.OptionalStorage.impl.Get.type.2f2) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Get (constants.%T.as_type.as.OptionalStorage.impl.Get.884)]
// CHECK:STDOUT:   %Core.import_ref.554 = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], unloaded
// CHECK:STDOUT:   %OptionalStorage.impl_witness_table.460 = impl_witness_table (%Core.import_ref.ace, %Core.import_ref.063, %Core.import_ref.4b4, %Core.import_ref.af7, %Core.import_ref.dd4, %Core.import_ref.554), @T.as_type.as.OptionalStorage.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.1d7: @DefaultOptionalStorage.%DefaultOptionalStorage.elem.1 (%DefaultOptionalStorage.elem.9db) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [concrete = %field_decl.5f0]
// CHECK:STDOUT:   %Core.import_ref.929: @DefaultOptionalStorage.%DefaultOptionalStorage.elem.2 (%DefaultOptionalStorage.elem.06a) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [concrete = %field_decl.6a8]
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
// CHECK:STDOUT:   %field_decl.6a8: @DefaultOptionalStorage.%DefaultOptionalStorage.elem.2 (%DefaultOptionalStorage.elem.06a) = field_decl has_value, element1, %Core.import_ref.b0d in [concrete] {}
// CHECK:STDOUT:   %Core.import_ref.87f = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %field_decl.5f0: @DefaultOptionalStorage.%DefaultOptionalStorage.elem.1 (%DefaultOptionalStorage.elem.9db) = field_decl value, element0, %Core.import_ref.87f in [concrete] {}
// CHECK:STDOUT:   %MutableRange.decl: type = class_decl @MutableRange [concrete = constants.%MutableRange] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generated {
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.1: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc53_44.1 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.2] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.1d1 = ref_param_pattern [concrete = constants.%self.param_patt.964]
// CHECK:STDOUT:     %self.patt: %pattern_type.1d1 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.82f]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %.dd1 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %.dd1 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.2: %Destroy.WithSelf.Op.type.ef016f.3 = fn_decl @Destroy.WithSelf.Op.loc53_44.1 [concrete = constants.%Destroy.WithSelf.Op.403171.3] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.1d1 = ref_param_pattern [concrete = constants.%self.param_patt.964]
// CHECK:STDOUT:     %self.patt: %pattern_type.1d1 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.82f]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %.dd1 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %.dd1 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.2: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.3 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc53_44.2 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.3] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.0da = ref_param_pattern [concrete = constants.%self.param_patt.623]
// CHECK:STDOUT:     %self.patt: %pattern_type.0da = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.7ed]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %MaybeUnformed.385 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %MaybeUnformed.385 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.3: %Destroy.WithSelf.Op.type.ef016f.4 = fn_decl @Destroy.WithSelf.Op.loc53_44.2 [concrete = constants.%Destroy.WithSelf.Op.403171.4] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.0da = ref_param_pattern [concrete = constants.%self.param_patt.623]
// CHECK:STDOUT:     %self.patt: %pattern_type.0da = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.7ed]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %MaybeUnformed.385 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %MaybeUnformed.385 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.3: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.4 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc53_44.3 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.4] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.831 = ref_param_pattern [concrete = constants.%self.param_patt.40d]
// CHECK:STDOUT:     %self.patt: %pattern_type.831 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.560]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref bool = ref_param call_param0
// CHECK:STDOUT:     %self: ref bool = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.4: %Destroy.WithSelf.Op.type.ef016f.5 = fn_decl @Destroy.WithSelf.Op.loc53_44.3 [concrete = constants.%Destroy.WithSelf.Op.403171.5] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.831 = ref_param_pattern [concrete = constants.%self.param_patt.40d]
// CHECK:STDOUT:     %self.patt: %pattern_type.831 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.560]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref bool = ref_param call_param0
// CHECK:STDOUT:     %self: ref bool = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.4: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.5 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc53_44.4 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.5] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.cdb = ref_param_pattern [concrete = constants.%self.param_patt.31f]
// CHECK:STDOUT:     %self.patt: %pattern_type.cdb = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.692]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %struct_type.value.has_value.03f = ref_param call_param0
// CHECK:STDOUT:     %self: ref %struct_type.value.has_value.03f = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.5: %Destroy.WithSelf.Op.type.ef016f.6 = fn_decl @Destroy.WithSelf.Op.loc53_44.4 [concrete = constants.%Destroy.WithSelf.Op.403171.6] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.cdb = ref_param_pattern [concrete = constants.%self.param_patt.31f]
// CHECK:STDOUT:     %self.patt: %pattern_type.cdb = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.692]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %struct_type.value.has_value.03f = ref_param call_param0
// CHECK:STDOUT:     %self: ref %struct_type.value.has_value.03f = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.5: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.6 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc53_44.5 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.6] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.b4b = ref_param_pattern [concrete = constants.%self.param_patt.48e]
// CHECK:STDOUT:     %self.patt: %pattern_type.b4b = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.724]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %DefaultOptionalStorage.022 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %DefaultOptionalStorage.022 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.6: %Destroy.WithSelf.Op.type.ef016f.7 = fn_decl @Destroy.WithSelf.Op.loc53_44.5 [concrete = constants.%Destroy.WithSelf.Op.403171.7] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.b4b = ref_param_pattern [concrete = constants.%self.param_patt.48e]
// CHECK:STDOUT:     %self.patt: %pattern_type.b4b = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.724]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %DefaultOptionalStorage.022 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %DefaultOptionalStorage.022 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.6: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.7 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc53_44.6 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.7] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.5a0 = ref_param_pattern [concrete = constants.%self.param_patt.d85]
// CHECK:STDOUT:     %self.patt: %pattern_type.5a0 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.a14]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %Optional.61e = ref_param call_param0
// CHECK:STDOUT:     %self: ref %Optional.61e = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.7: %Destroy.WithSelf.Op.type.ef016f.8 = fn_decl @Destroy.WithSelf.Op.loc53_44.6 [concrete = constants.%Destroy.WithSelf.Op.403171.8] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.5a0 = ref_param_pattern [concrete = constants.%self.param_patt.d85]
// CHECK:STDOUT:     %self.patt: %pattern_type.5a0 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.a14]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %Optional.61e = ref_param call_param0
// CHECK:STDOUT:     %self: ref %Optional.61e = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @TestIterate(%R.loc49_17.2: %Iterate_where.type.9df) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Iterate.lookup_impl_witness: <witness> = lookup_impl_witness %R.loc49_17.1, @Iterate [symbolic = %Iterate.lookup_impl_witness (constants.%Iterate.lookup_impl_witness.711)]
// CHECK:STDOUT:   %Iterate.facet.loc53_44.5: %Iterate.type = facet_value %R.as_type.loc49_86.1, (%Iterate.lookup_impl_witness) [symbolic = %Iterate.facet.loc53_44.5 (constants.%Iterate.facet.04e)]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.loc53_44.5) [symbolic = %Iterate.WithSelf.NewCursor.type (constants.%Iterate.WithSelf.NewCursor.type.f6c)]
// CHECK:STDOUT:   %.loc53_44.20: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type, %Iterate.facet.loc53_44.5 [symbolic = %.loc53_44.20 (constants.%.2577)]
// CHECK:STDOUT:   %impl.elem2.loc53_44.3: @TestIterate.%.loc53_44.20 (%.2577) = impl_witness_access %Iterate.lookup_impl_witness, element2 [symbolic = %impl.elem2.loc53_44.3 (constants.%impl.elem2.f11)]
// CHECK:STDOUT:   %specific_impl_fn.loc53_44.4: <specific function> = specific_impl_function %impl.elem2.loc53_44.3, @Iterate.WithSelf.NewCursor(%Iterate.facet.loc53_44.5) [symbolic = %specific_impl_fn.loc53_44.4 (constants.%specific_impl_fn.d03)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.loc53_44.5) [symbolic = %Iterate.WithSelf.Next.type (constants.%Iterate.WithSelf.Next.type.d76)]
// CHECK:STDOUT:   %.loc53_44.21: type = fn_type_with_self_type %Iterate.WithSelf.Next.type, %Iterate.facet.loc53_44.5 [symbolic = %.loc53_44.21 (constants.%.894)]
// CHECK:STDOUT:   %impl.elem3.loc53_44.2: @TestIterate.%.loc53_44.21 (%.894) = impl_witness_access %Iterate.lookup_impl_witness, element3 [symbolic = %impl.elem3.loc53_44.2 (constants.%impl.elem3.eed)]
// CHECK:STDOUT:   %specific_impl_fn.loc53_44.5: <specific function> = specific_impl_function %impl.elem3.loc53_44.2, @Iterate.WithSelf.Next(%Iterate.facet.loc53_44.5) [symbolic = %specific_impl_fn.loc53_44.5 (constants.%specific_impl_fn.303)]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type: type = fn_type @Destroy.WithSelf.SelfDestruct.1, @Destroy.WithSelf(%impl.elem1) [symbolic = %Destroy.WithSelf.SelfDestruct.type (constants.%Destroy.WithSelf.SelfDestruct.type.55d)]
// CHECK:STDOUT:   %.loc53_44.22: type = fn_type_with_self_type %Destroy.WithSelf.SelfDestruct.type, %impl.elem1 [symbolic = %.loc53_44.22 (constants.%.a44)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %impl.elem1, @Destroy [symbolic = %Destroy.lookup_impl_witness (constants.%Destroy.lookup_impl_witness.696)]
// CHECK:STDOUT:   %impl.elem2.loc53_44.4: @TestIterate.%.loc53_44.22 (%.a44) = impl_witness_access %Destroy.lookup_impl_witness, element2 [symbolic = %impl.elem2.loc53_44.4 (constants.%impl.elem2.1f1)]
// CHECK:STDOUT:   %specific_impl_fn.loc53_44.6: <specific function> = specific_impl_function %impl.elem2.loc53_44.4, @Destroy.WithSelf.SelfDestruct.1(%impl.elem1) [symbolic = %specific_impl_fn.loc53_44.6 (constants.%specific_impl_fn.cb8)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%r.param: @TestIterate.%R.as_type.loc49_86.1 (%R.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %i.patt: %pattern_type.d2d = value_binding_pattern i [concrete = constants.%i.patt]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %r.ref: @TestIterate.%R.as_type.loc49_86.1 (%R.as_type) = name_ref r, %r
// CHECK:STDOUT:     %impl.elem2.loc53_44.1: @TestIterate.%.loc53_44.20 (%.2577) = impl_witness_access constants.%Iterate.lookup_impl_witness.711, element2 [symbolic = %impl.elem2.loc53_44.3 (constants.%impl.elem2.f11)]
// CHECK:STDOUT:     %bound_method.loc53_44.1: <bound method> = bound_method %r.ref, %impl.elem2.loc53_44.1
// CHECK:STDOUT:     %Iterate.facet.loc53_44.1: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.711) [symbolic = %Iterate.facet.loc53_44.5 (constants.%Iterate.facet.04e)]
// CHECK:STDOUT:     %.loc53_44.1: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc53_44.1 [symbolic = %Iterate.facet.loc53_44.5 (constants.%Iterate.facet.04e)]
// CHECK:STDOUT:     %Iterate.facet.loc53_44.2: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.711) [symbolic = %Iterate.facet.loc53_44.5 (constants.%Iterate.facet.04e)]
// CHECK:STDOUT:     %.loc53_44.2: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc53_44.2 [symbolic = %Iterate.facet.loc53_44.5 (constants.%Iterate.facet.04e)]
// CHECK:STDOUT:     %specific_impl_fn.loc53_44.1: <specific function> = specific_impl_function %impl.elem2.loc53_44.1, @Iterate.WithSelf.NewCursor(constants.%Iterate.facet.04e) [symbolic = %specific_impl_fn.loc53_44.4 (constants.%specific_impl_fn.d03)]
// CHECK:STDOUT:     %bound_method.loc53_44.2: <bound method> = bound_method %r.ref, %specific_impl_fn.loc53_44.1
// CHECK:STDOUT:     %var: ref @TestIterate.%as_type (%as_type.b30) = var_storage invalid
// CHECK:STDOUT:     %Iterate.WithSelf.NewCursor.call: init @TestIterate.%as_type (%as_type.b30) to %var = call %bound_method.loc53_44.2(%r.ref)
// CHECK:STDOUT:     assign %var, %Iterate.WithSelf.NewCursor.call
// CHECK:STDOUT:     br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.next:
// CHECK:STDOUT:     %impl.elem3.loc53_44.1: @TestIterate.%.loc53_44.21 (%.894) = impl_witness_access constants.%Iterate.lookup_impl_witness.711, element3 [symbolic = %impl.elem3.loc53_44.2 (constants.%impl.elem3.eed)]
// CHECK:STDOUT:     %bound_method.loc53_44.3: <bound method> = bound_method %r.ref, %impl.elem3.loc53_44.1
// CHECK:STDOUT:     %Iterate.facet.loc53_44.3: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.711) [symbolic = %Iterate.facet.loc53_44.5 (constants.%Iterate.facet.04e)]
// CHECK:STDOUT:     %.loc53_44.3: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc53_44.3 [symbolic = %Iterate.facet.loc53_44.5 (constants.%Iterate.facet.04e)]
// CHECK:STDOUT:     %Iterate.facet.loc53_44.4: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.711) [symbolic = %Iterate.facet.loc53_44.5 (constants.%Iterate.facet.04e)]
// CHECK:STDOUT:     %.loc53_44.4: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc53_44.4 [symbolic = %Iterate.facet.loc53_44.5 (constants.%Iterate.facet.04e)]
// CHECK:STDOUT:     %.loc53_44.5: %Destroy.type = converted constants.%as_type.b30, constants.%impl.elem1.0e0 [symbolic = %impl.elem1 (constants.%impl.elem1.0e0)]
// CHECK:STDOUT:     %.loc53_44.6: %Destroy.type = converted constants.%as_type.b30, constants.%impl.elem1.0e0 [symbolic = %impl.elem1 (constants.%impl.elem1.0e0)]
// CHECK:STDOUT:     %specific_impl_fn.loc53_44.2: <specific function> = specific_impl_function %impl.elem3.loc53_44.1, @Iterate.WithSelf.Next(constants.%Iterate.facet.04e) [symbolic = %specific_impl_fn.loc53_44.5 (constants.%specific_impl_fn.303)]
// CHECK:STDOUT:     %bound_method.loc53_44.4: <bound method> = bound_method %r.ref, %specific_impl_fn.loc53_44.2
// CHECK:STDOUT:     %.loc53_44.7: ref %Optional.61e = temporary_storage
// CHECK:STDOUT:     %Iterate.WithSelf.Next.call: init %Optional.61e to %.loc53_44.7 = call %bound_method.loc53_44.4(%r.ref, %var)
// CHECK:STDOUT:     %.loc53_44.8: ref %Optional.61e = temporary %.loc53_44.7, %Iterate.WithSelf.Next.call
// CHECK:STDOUT:     %.loc53_44.9: %Optional.HasValue.type.dd2 = specific_constant imports.%Core.import_ref.4b2, @Optional(constants.%OptionalStorage.facet.c95) [concrete = constants.%Optional.HasValue.e45]
// CHECK:STDOUT:     %HasValue.ref: %Optional.HasValue.type.dd2 = name_ref HasValue, %.loc53_44.9 [concrete = constants.%Optional.HasValue.e45]
// CHECK:STDOUT:     %Optional.HasValue.bound: <bound method> = bound_method %.loc53_44.8, %HasValue.ref
// CHECK:STDOUT:     %Optional.HasValue.specific_fn: <specific function> = specific_function %HasValue.ref, @Optional.HasValue(constants.%OptionalStorage.facet.c95) [concrete = constants.%Optional.HasValue.specific_fn]
// CHECK:STDOUT:     %bound_method.loc53_44.5: <bound method> = bound_method %.loc53_44.8, %Optional.HasValue.specific_fn
// CHECK:STDOUT:     %.loc53_44.10: %Optional.61e = acquire_value %.loc53_44.8
// CHECK:STDOUT:     %Optional.HasValue.call: init bool = call %bound_method.loc53_44.5(%.loc53_44.10)
// CHECK:STDOUT:     %.loc53_44.11: bool = value_of_initializer %Optional.HasValue.call
// CHECK:STDOUT:     %.loc53_44.12: bool = converted %Optional.HasValue.call, %.loc53_44.11
// CHECK:STDOUT:     if %.loc53_44.12 br !for.body else br !for.done
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.body:
// CHECK:STDOUT:     %.loc53_44.13: %Optional.Get.type.caf = specific_constant imports.%Core.import_ref.52c, @Optional(constants.%OptionalStorage.facet.c95) [concrete = constants.%Optional.Get.04c]
// CHECK:STDOUT:     %Get.ref: %Optional.Get.type.caf = name_ref Get, %.loc53_44.13 [concrete = constants.%Optional.Get.04c]
// CHECK:STDOUT:     %Optional.Get.bound: <bound method> = bound_method %.loc53_44.8, %Get.ref
// CHECK:STDOUT:     %Optional.Get.specific_fn: <specific function> = specific_function %Get.ref, @Optional.Get(constants.%OptionalStorage.facet.c95) [concrete = constants.%Optional.Get.specific_fn]
// CHECK:STDOUT:     %bound_method.loc53_44.6: <bound method> = bound_method %.loc53_44.8, %Optional.Get.specific_fn
// CHECK:STDOUT:     %.loc53_44.14: ref %ValueType = temporary_storage
// CHECK:STDOUT:     %.loc53_44.15: %Optional.61e = acquire_value %.loc53_44.8
// CHECK:STDOUT:     %Optional.Get.call: init %ValueType to %.loc53_44.14 = call %bound_method.loc53_44.6(%.loc53_44.15)
// CHECK:STDOUT:     %.loc53_44.16: ref %ValueType = temporary %.loc53_44.14, %Optional.Get.call
// CHECK:STDOUT:     %.loc53_44.17: %ValueType = acquire_value %.loc53_44.16
// CHECK:STDOUT:     %.loc53_29: type = splice_block %ValueType.ref.loc53 [concrete = constants.%ValueType] {
// CHECK:STDOUT:       %Cpp.ref.loc53: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:       %DiffMutableADL.ref.loc53: <namespace> = name_ref DiffMutableADL, imports.%DiffMutableADL [concrete = imports.%DiffMutableADL]
// CHECK:STDOUT:       %ValueType.ref.loc53: type = name_ref ValueType, imports.%ValueType.decl [concrete = constants.%ValueType]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %i: %ValueType = wrapper_binding i, %.loc53_44.17
// CHECK:STDOUT:     %sum.ref: ref %ValueType = name_ref sum, %sum
// CHECK:STDOUT:     %i.ref: %ValueType = name_ref i, %i
// CHECK:STDOUT:     %.loc54_12: ref %ValueType = value_as_ref %i.ref
// CHECK:STDOUT:     %addr.loc54: %ptr.9c6 = addr_of %.loc54_12
// CHECK:STDOUT:     %.loc54_9.1: %ptr.5f4 = as_compatible %addr.loc54
// CHECK:STDOUT:     %.loc54_9.2: %ptr.5f4 = converted %addr.loc54, %.loc54_9.1
// CHECK:STDOUT:     %operator_PlusEqual__carbon_thunk.call: ref %ValueType = call imports.%operator_PlusEqual__carbon_thunk.decl(%sum.ref, %.loc54_9.2)
// CHECK:STDOUT:     %Destroy.WithSelf.SelfDestruct.bound.loc53_44.1: <bound method> = bound_method %.loc53_44.16, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.2
// CHECK:STDOUT:     %Destroy.WithSelf.SelfDestruct.call.loc53_44.1: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc53_44.1(%.loc53_44.16)
// CHECK:STDOUT:     %Destroy.WithSelf.SelfDestruct.bound.loc53_44.2: <bound method> = bound_method %.loc53_44.8, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.8
// CHECK:STDOUT:     %Destroy.WithSelf.SelfDestruct.call.loc53_44.2: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc53_44.2(%.loc53_44.8)
// CHECK:STDOUT:     br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.done:
// CHECK:STDOUT:     %Destroy.WithSelf.SelfDestruct.bound.loc53_44.3: <bound method> = bound_method %.loc53_44.8, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.8
// CHECK:STDOUT:     %Destroy.WithSelf.SelfDestruct.call.loc53_44.3: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc53_44.3(%.loc53_44.8)
// CHECK:STDOUT:     %impl.elem2.loc53_44.2: @TestIterate.%.loc53_44.22 (%.a44) = impl_witness_access constants.%Destroy.lookup_impl_witness.696, element2 [symbolic = %impl.elem2.loc53_44.4 (constants.%impl.elem2.1f1)]
// CHECK:STDOUT:     %bound_method.loc53_44.7: <bound method> = bound_method %var, %impl.elem2.loc53_44.2
// CHECK:STDOUT:     %.loc53_44.18: %Destroy.type = converted constants.%as_type.b30, constants.%impl.elem1.0e0 [symbolic = %impl.elem1 (constants.%impl.elem1.0e0)]
// CHECK:STDOUT:     %.loc53_44.19: %Destroy.type = converted constants.%as_type.b30, constants.%impl.elem1.0e0 [symbolic = %impl.elem1 (constants.%impl.elem1.0e0)]
// CHECK:STDOUT:     %specific_impl_fn.loc53_44.3: <specific function> = specific_impl_function %impl.elem2.loc53_44.2, @Destroy.WithSelf.SelfDestruct.1(constants.%impl.elem1.0e0) [symbolic = %specific_impl_fn.loc53_44.6 (constants.%specific_impl_fn.cb8)]
// CHECK:STDOUT:     %bound_method.loc53_44.8: <bound method> = bound_method %var, %specific_impl_fn.loc53_44.3
// CHECK:STDOUT:     %Destroy.WithSelf.SelfDestruct.call.loc53_44.4: init %empty_tuple.type = call %bound_method.loc53_44.8(%var)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc53_44.1(%self.param: ref %.dd1) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc53_44.1(%self.param: ref %.dd1) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc53_44.1(%self.param: ref %.dd1) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.2(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.1(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc53_44.2(%self.param: ref %MaybeUnformed.385) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc53_44.2(%self.param: ref %MaybeUnformed.385) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc53_44.2(%self.param: ref %MaybeUnformed.385) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.3(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.2(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc53_44.3(%self.param: ref bool) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc53_44.3(%self.param: ref bool) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc53_44.3(%self.param: ref bool) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.4(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.3(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc53_44.4(%self.param: ref %struct_type.value.has_value.03f) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc53_44.1: ref bool = struct_access %self.param, element1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc53_44.1: <bound method> = bound_method %.loc53_44.1, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.5
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc53_44.1: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc53_44.1(%.loc53_44.1)
// CHECK:STDOUT:   %.loc53_44.2: ref %MaybeUnformed.385 = struct_access %self.param, element0
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc53_44.2: <bound method> = bound_method %.loc53_44.2, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.4
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc53_44.2: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc53_44.2(%.loc53_44.2)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc53_44.4(%self.param: ref %struct_type.value.has_value.03f) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc53_44.4(%self.param: ref %struct_type.value.has_value.03f) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.5(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.4(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc53_44.5(%self.param: ref %DefaultOptionalStorage.022) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %has_value.ref: %DefaultOptionalStorage.elem.80f = name_ref has_value, imports.%Core.import_ref.929 [concrete = imports.%field_decl.6a8]
// CHECK:STDOUT:   %.loc53_44.1: ref bool = class_element_access %self.param, element1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc53_44.1: <bound method> = bound_method %.loc53_44.1, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.5
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc53_44.1: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc53_44.1(%.loc53_44.1)
// CHECK:STDOUT:   %value.ref: %DefaultOptionalStorage.elem.dae = name_ref value, imports.%Core.import_ref.1d7 [concrete = imports.%field_decl.5f0]
// CHECK:STDOUT:   %.loc53_44.2: ref %MaybeUnformed.385 = class_element_access %self.param, element0
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc53_44.2: <bound method> = bound_method %.loc53_44.2, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.4
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc53_44.2: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc53_44.2(%.loc53_44.2)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc53_44.5(%self.param: ref %DefaultOptionalStorage.022) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc53_44.5(%self.param: ref %DefaultOptionalStorage.022) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.6(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.5(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc53_44.6(%self.param: ref %Optional.61e) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc53_44.6(%self.param: ref %Optional.61e) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc53_44.6(%self.param: ref %Optional.61e) {
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
// CHECK:STDOUT:   %HasValue.ref: <error> = name_ref HasValue, <error> [concrete = <error>]
// CHECK:STDOUT:   if <error> br !for.body else br !for.done
// CHECK:STDOUT:
// CHECK:STDOUT: !for.body:
// CHECK:STDOUT:   %Get.ref: <error> = name_ref Get, <error> [concrete = <error>]
// CHECK:STDOUT:   %.loc85: type = splice_block %ValueType.ref.loc85 [concrete = constants.%ValueType] {
// CHECK:STDOUT:     %Cpp.ref.loc85: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %DiffMutableADL.ref.loc85: <namespace> = name_ref DiffMutableADL, imports.%DiffMutableADL [concrete = imports.%DiffMutableADL]
// CHECK:STDOUT:     %ValueType.ref.loc85: type = name_ref ValueType, imports.%ValueType.decl [concrete = constants.%ValueType]
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
// CHECK:STDOUT:   %R.patt.loc49_17.2 => constants.%R.patt
// CHECK:STDOUT:   %R.loc49_17.1 => constants.%R
// CHECK:STDOUT:   %R.as_type.loc49_86.1 => constants.%R.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.0bc
// CHECK:STDOUT:   %r.param_patt.loc49_84.2 => constants.%r.param_patt
// CHECK:STDOUT:   %r.patt.loc49_84.2 => constants.%r.patt
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
// CHECK:STDOUT:   %Optional.Get.type.c64: type = fn_type @Optional.Get, @Optional(%T.220) [symbolic]
// CHECK:STDOUT:   %Optional.Get.bcf: %Optional.Get.type.c64 = struct_value () [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.type.49e: type = fn_type @Optional.HasValue, @Optional(%T.220) [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.c20: %Optional.HasValue.type.49e = struct_value () [symbolic]
// CHECK:STDOUT:   %T.283: %facet_type.f48 = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.74b: type = facet_access_type %T.283 [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.28a: <witness> = lookup_impl_witness %T.283, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.16e: %Destroy.type = facet_value %T.as_type.74b, (%Destroy.lookup_impl_witness.28a) [symbolic]
// CHECK:STDOUT:   %MaybeUnformed.9b7: type = class_type @MaybeUnformed, @MaybeUnformed(%Destroy.facet.16e) [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.faa: type = class_type @DefaultOptionalStorage, @DefaultOptionalStorage(%T.283) [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.elem.06a: type = unbound_element_type %DefaultOptionalStorage.faa, bool [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.elem.9db: type = unbound_element_type %DefaultOptionalStorage.faa, %MaybeUnformed.9b7 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Get.type.2f2: type = fn_type @T.as_type.as.OptionalStorage.impl.Get, @T.as_type.as.OptionalStorage.impl(%T.283) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Get.884: %T.as_type.as.OptionalStorage.impl.Get.type.2f2 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Has.type.d28: type = fn_type @T.as_type.as.OptionalStorage.impl.Has, @T.as_type.as.OptionalStorage.impl(%T.283) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Has.f29: %T.as_type.as.OptionalStorage.impl.Has.type.d28 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Some.type.760: type = fn_type @T.as_type.as.OptionalStorage.impl.Some, @T.as_type.as.OptionalStorage.impl(%T.283) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Some.268: %T.as_type.as.OptionalStorage.impl.Some.type.760 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.None.type.925: type = fn_type @T.as_type.as.OptionalStorage.impl.None, @T.as_type.as.OptionalStorage.impl(%T.283) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.None.f42: %T.as_type.as.OptionalStorage.impl.None.type.925 = struct_value () [symbolic]
// CHECK:STDOUT:   %.Self.500: %Iterate.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %CppRange.type: type = facet_type <@CppRange> [concrete]
// CHECK:STDOUT:   %.Self.e0b: %CppRange.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %CppRangeForIterate.lookup_impl_witness.50b: <witness> = lookup_impl_witness %.Self.e0b, @CppRangeForIterate [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.32d: type = impl_witness_access %CppRangeForIterate.lookup_impl_witness.50b, element0 [symbolic_self]
// CHECK:STDOUT:   %CppUnsafeDeref.lookup_impl_witness.851: <witness> = lookup_impl_witness %impl.elem0.32d, @CppUnsafeDeref [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.10d: type = impl_witness_access %CppUnsafeDeref.lookup_impl_witness.851, element0 [symbolic_self]
// CHECK:STDOUT:   %CppRange_where.type.d682d6.1: type = facet_type <@CppRange where %impl.elem0.10d impls @Copy> [concrete]
// CHECK:STDOUT:   %T.f45530.2: %CppRange_where.type.d682d6.1 = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.type.d4a: type = fn_type @T.as_type.as.Iterate.impl.Next, @T.as_type.as.Iterate.impl(%T.f45530.2) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.5a8: %T.as_type.as.Iterate.impl.Next.type.d4a = struct_value () [symbolic]
// CHECK:STDOUT:   %CppRangeForIterate.lookup_impl_witness.b581a8.2: <witness> = lookup_impl_witness %T.f45530.2, @CppRangeForIterate [symbolic]
// CHECK:STDOUT:   %impl.elem0.5f971c.2: type = impl_witness_access %CppRangeForIterate.lookup_impl_witness.b581a8.2, element0 [symbolic]
// CHECK:STDOUT:   %CppUnsafeDeref.lookup_impl_witness.87e543.2: <witness> = lookup_impl_witness %impl.elem0.5f971c.2, @CppUnsafeDeref [symbolic]
// CHECK:STDOUT:   %impl.elem0.493765.2: type = impl_witness_access %CppUnsafeDeref.lookup_impl_witness.87e543.2, element0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.type.8fb: type = fn_type @T.as_type.as.Iterate.impl.NewCursor, @T.as_type.as.Iterate.impl(%T.f45530.2) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.8a1: %T.as_type.as.Iterate.impl.NewCursor.type.8fb = struct_value () [symbolic]
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
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.2: type = fn_type @Destroy.WithSelf.Op.loc56 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.2: %Destroy.WithSelf.Op.type.ef016f.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.2: type = fn_type @Destroy.WithSelf.SelfDestruct.loc56 [concrete]
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
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.fb9: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.c55) [symbolic]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.5d8: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.c55) [symbolic]
// CHECK:STDOUT:   %.54f: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.fb9, %Iterate.facet.c55 [symbolic]
// CHECK:STDOUT:   %impl.elem2.ff2: %.54f = impl_witness_access %Iterate.lookup_impl_witness.a99, element2 [symbolic]
// CHECK:STDOUT:   %impl.elem1.751: %Destroy.type = impl_witness_access %Iterate.lookup_impl_witness.a99, element1 [symbolic]
// CHECK:STDOUT:   %as_type.733: type = facet_access_type %impl.elem1.751 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.541: <specific function> = specific_impl_function %impl.elem2.ff2, @Iterate.WithSelf.NewCursor(%Iterate.facet.c55) [symbolic]
// CHECK:STDOUT:   %.216: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.5d8, %Iterate.facet.c55 [symbolic]
// CHECK:STDOUT:   %impl.elem3.5f1: %.216 = impl_witness_access %Iterate.lookup_impl_witness.a99, element3 [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.ac6: type = class_type @DefaultOptionalStorage, @DefaultOptionalStorage(%facet_value.36a) [concrete]
// CHECK:STDOUT:   %OptionalStorage.impl_witness.120: <witness> = impl_witness imports.%OptionalStorage.impl_witness_table.460, @T.as_type.as.OptionalStorage.impl(%facet_value.36a) [concrete]
// CHECK:STDOUT:   %OptionalStorage.facet.adc: %OptionalStorage.type = facet_value %ValueType, (%OptionalStorage.impl_witness.120) [concrete]
// CHECK:STDOUT:   %Optional.8d2: type = class_type @Optional, @Optional(%OptionalStorage.facet.adc) [concrete]
// CHECK:STDOUT:   %pattern_type.f76: type = pattern_type %Optional.8d2 [concrete]
// CHECK:STDOUT:   %specific_impl_fn.eaf: <specific function> = specific_impl_function %impl.elem3.5f1, @Iterate.WithSelf.Next(%Iterate.facet.c55) [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.type.10c: type = fn_type @Optional.HasValue, @Optional(%OptionalStorage.facet.adc) [concrete]
// CHECK:STDOUT:   %Optional.HasValue.e84: %Optional.HasValue.type.10c = struct_value () [concrete]
// CHECK:STDOUT:   %Optional.Get.type.bc3: type = fn_type @Optional.Get, @Optional(%OptionalStorage.facet.adc) [concrete]
// CHECK:STDOUT:   %Optional.Get.d28: %Optional.Get.type.bc3 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.facet.745: %Destroy.type = facet_value %ValueType, (%custom_witness.dc3) [concrete]
// CHECK:STDOUT:   %MaybeUnformed.713: type = class_type @MaybeUnformed, @MaybeUnformed(%Destroy.facet.745) [concrete]
// CHECK:STDOUT:   %.8f3: type = maybe_unformed_type %ValueType [concrete]
// CHECK:STDOUT:   %DefaultOptionalStorage.elem.1eb: type = unbound_element_type %DefaultOptionalStorage.ac6, %MaybeUnformed.713 [concrete]
// CHECK:STDOUT:   %DefaultOptionalStorage.elem.a4b: type = unbound_element_type %DefaultOptionalStorage.ac6, bool [concrete]
// CHECK:STDOUT:   %struct_type.value.has_value.793: type = struct_type {.value: %MaybeUnformed.713, .has_value: bool} [concrete]
// CHECK:STDOUT:   %Optional.HasValue.specific_fn.a99: <specific function> = specific_function %Optional.HasValue.e84, @Optional.HasValue(%OptionalStorage.facet.adc) [concrete]
// CHECK:STDOUT:   %Optional.Get.specific_fn.22d: <specific function> = specific_function %Optional.Get.d28, @Optional.Get(%OptionalStorage.facet.adc) [concrete]
// CHECK:STDOUT:   %operator_PlusEqual__carbon_thunk.type: type = fn_type @operator_PlusEqual__carbon_thunk [concrete]
// CHECK:STDOUT:   %operator_PlusEqual__carbon_thunk: %operator_PlusEqual__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.60e: type = pattern_type %.8f3 [concrete]
// CHECK:STDOUT:   %self.param_patt.61b3: %pattern_type.60e = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.ecc: %pattern_type.60e = wrapper_binding_pattern self, %self.param_patt.61b3 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc60_41.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.2: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.3: type = fn_type @Destroy.WithSelf.Op.loc60_41.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.3: %Destroy.WithSelf.Op.type.ef016f.3 = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.90d: type = pattern_type %MaybeUnformed.713 [concrete]
// CHECK:STDOUT:   %self.param_patt.f43: %pattern_type.90d = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.162: %pattern_type.90d = wrapper_binding_pattern self, %self.param_patt.f43 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.3: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc60_41.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.3: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.3 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.4: type = fn_type @Destroy.WithSelf.Op.loc60_41.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.4: %Destroy.WithSelf.Op.type.ef016f.4 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.4: type = fn_type @Destroy.WithSelf.SelfDestruct.loc60_41.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.4: %Destroy.WithSelf.SelfDestruct.type.fbceb5.4 = struct_value () [concrete]
// CHECK:STDOUT:   %self.param_patt.40d: %pattern_type.831 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.560: %pattern_type.831 = wrapper_binding_pattern self, %self.param_patt.40d [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.4: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc60_41.3 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.4: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.4 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.5: type = fn_type @Destroy.WithSelf.Op.loc60_41.3 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.5: %Destroy.WithSelf.Op.type.ef016f.5 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.5: type = fn_type @Destroy.WithSelf.SelfDestruct.loc60_41.3 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.5: %Destroy.WithSelf.SelfDestruct.type.fbceb5.5 = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.030: type = pattern_type %struct_type.value.has_value.793 [concrete]
// CHECK:STDOUT:   %self.param_patt.c53: %pattern_type.030 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.250: %pattern_type.030 = wrapper_binding_pattern self, %self.param_patt.c53 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.5: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc60_41.4 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.5: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.5 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.6: type = fn_type @Destroy.WithSelf.Op.loc60_41.4 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.6: %Destroy.WithSelf.Op.type.ef016f.6 = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.6c5: type = pattern_type %DefaultOptionalStorage.ac6 [concrete]
// CHECK:STDOUT:   %self.param_patt.c66: %pattern_type.6c5 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.429: %pattern_type.6c5 = wrapper_binding_pattern self, %self.param_patt.c66 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.6: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc60_41.5 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.6: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.6 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.7: type = fn_type @Destroy.WithSelf.Op.loc60_41.5 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.7: %Destroy.WithSelf.Op.type.ef016f.7 = struct_value () [concrete]
// CHECK:STDOUT:   %self.param_patt.8f3: %pattern_type.f76 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.e6d: %pattern_type.f76 = wrapper_binding_pattern self, %self.param_patt.8f3 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.7: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc60_41.6 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.7: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.7 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.8: type = fn_type @Destroy.WithSelf.Op.loc60_41.6 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.8: %Destroy.WithSelf.Op.type.ef016f.8 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.8: type = fn_type @Destroy.WithSelf.SelfDestruct.loc60_41.6 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.8: %Destroy.WithSelf.SelfDestruct.type.fbceb5.8 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.eb0: <witness> = lookup_impl_witness %impl.elem1.751, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.4ed: type = fn_type @Destroy.WithSelf.SelfDestruct.1, @Destroy.WithSelf(%impl.elem1.751) [symbolic]
// CHECK:STDOUT:   %.97d: type = fn_type_with_self_type %Destroy.WithSelf.SelfDestruct.type.4ed, %impl.elem1.751 [symbolic]
// CHECK:STDOUT:   %impl.elem2.f7b: %.97d = impl_witness_access %Destroy.lookup_impl_witness.eb0, element2 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.577: <specific function> = specific_impl_function %impl.elem2.f7b, @Destroy.WithSelf.SelfDestruct.1(%impl.elem1.751) [symbolic]
// CHECK:STDOUT:   %MutableAndConstRange: type = class_type @MutableAndConstRange [concrete]
// CHECK:STDOUT:   %pattern_type.fdb: type = pattern_type %MutableAndConstRange [concrete]
// CHECK:STDOUT:   %Iterator: type = class_type @Iterator.1 [concrete]
// CHECK:STDOUT:   %Begin.type: type = fn_type @Begin [concrete]
// CHECK:STDOUT:   %Begin: %Begin.type = struct_value () [concrete]
// CHECK:STDOUT:   %End.type: type = fn_type @End [concrete]
// CHECK:STDOUT:   %End: %End.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.660: <witness> = custom_witness (%Iterator, %Iterator, %Begin, %End), @CppRangeForIterate [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.9: type = fn_type @Destroy.WithSelf.Op.loc67 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.9: %Destroy.WithSelf.Op.type.ef016f.9 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.9: type = fn_type @Destroy.WithSelf.SelfDestruct.loc67 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.9: %Destroy.WithSelf.SelfDestruct.type.fbceb5.9 = struct_value () [concrete]
// CHECK:STDOUT:   %Iterator.SubobjectDestroy.type: type = fn_type @Iterator.SubobjectDestroy [concrete]
// CHECK:STDOUT:   %Iterator.SubobjectDestroy: %Iterator.SubobjectDestroy.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.809: <witness> = custom_witness (%Destroy.WithSelf.Op.403171.9, %Iterator.SubobjectDestroy, %Destroy.WithSelf.SelfDestruct.db3fdb.9), @Destroy [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.c95ff6.1: type = fn_type @Iterator.Op.1 [concrete]
// CHECK:STDOUT:   %Iterator.Op.c90ac2.1: %Iterator.Op.type.c95ff6.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.d56: <witness> = custom_witness (%ValueType, %Iterator.Op.c90ac2.1), @CppUnsafeDeref [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.c95ff6.2: type = fn_type @Iterator.Op.2 [concrete]
// CHECK:STDOUT:   %Iterator.Op.c90ac2.2: %Iterator.Op.type.c95ff6.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.68a: <witness> = custom_witness (%Iterator.Op.c90ac2.2), @Inc [concrete]
// CHECK:STDOUT:   %Equal.type: type = fn_type @Equal [concrete]
// CHECK:STDOUT:   %Equal: %Equal.type = struct_value () [concrete]
// CHECK:STDOUT:   %NotEqual.type: type = fn_type @NotEqual [concrete]
// CHECK:STDOUT:   %NotEqual: %NotEqual.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.edd: <witness> = custom_witness (%Equal, %NotEqual), @EqWith, @EqWith(%Iterator) [concrete]
// CHECK:STDOUT:   %facet_value.24f: %CppRange_where.type.d682d6.1 = facet_value %MutableAndConstRange, (%custom_witness.660, %custom_witness.809, %custom_witness.d56, %custom_witness.68a, %custom_witness.edd, %custom_witness.dc3, %custom_witness.1c5, %custom_witness.809) [concrete]
// CHECK:STDOUT:   %tuple.type.de6: type = tuple_type (%Iterator, %Iterator) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.8: type = fn_type @Destroy.WithSelf.SubobjectDestroy.3 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.8: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.8 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.10: type = fn_type @Destroy.WithSelf.Op.3 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.10: %Destroy.WithSelf.Op.type.ef016f.10 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.10: type = fn_type @Destroy.WithSelf.SelfDestruct.3 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.10: %Destroy.WithSelf.SelfDestruct.type.fbceb5.10 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.f8f19d.8: <witness> = custom_witness (%Destroy.WithSelf.Op.403171.10, %Destroy.WithSelf.SubobjectDestroy.d01daf.8, %Destroy.WithSelf.SelfDestruct.db3fdb.10), @Destroy [concrete]
// CHECK:STDOUT:   %Destroy.facet.c9a: %Destroy.type = facet_value %tuple.type.de6, (%custom_witness.f8f19d.8) [concrete]
// CHECK:STDOUT:   %Iterate.impl_witness.198: <witness> = impl_witness imports.%Iterate.impl_witness_table.121, @T.as_type.as.Iterate.impl(%facet_value.24f) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.type.212: type = fn_type @T.as_type.as.Iterate.impl.NewCursor, @T.as_type.as.Iterate.impl(%facet_value.24f) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.160: %T.as_type.as.Iterate.impl.NewCursor.type.212 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.type.6dd: type = fn_type @T.as_type.as.Iterate.impl.Next, @T.as_type.as.Iterate.impl(%facet_value.24f) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.ddf: %T.as_type.as.Iterate.impl.Next.type.6dd = struct_value () [concrete]
// CHECK:STDOUT:   %Iterate.facet.b30: %Iterate.type = facet_value %MutableAndConstRange, (%Iterate.impl_witness.198) [concrete]
// CHECK:STDOUT:   %facet_value.fc8: %Iterate_where.type.8b0 = facet_value %MutableAndConstRange, (%Iterate.impl_witness.198) [concrete]
// CHECK:STDOUT:   %r.param_patt.3a6: %pattern_type.fdb = value_param_pattern [concrete]
// CHECK:STDOUT:   %r.patt.403: %pattern_type.fdb = wrapper_binding_pattern r, %r.param_patt.3a6 [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.b5a: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.b30) [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.3b0: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.b30) [concrete]
// CHECK:STDOUT:   %.bc1: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.b5a, %Iterate.facet.b30 [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.specific_fn: <specific function> = specific_function %T.as_type.as.Iterate.impl.NewCursor.160, @T.as_type.as.Iterate.impl.NewCursor(%facet_value.24f) [concrete]
// CHECK:STDOUT:   %.698: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.3b0, %Iterate.facet.b30 [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.specific_fn: <specific function> = specific_function %T.as_type.as.Iterate.impl.Next.ddf, @T.as_type.as.Iterate.impl.Next(%facet_value.24f) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.235: type = fn_type @Destroy.WithSelf.SelfDestruct.1, @Destroy.WithSelf(%Destroy.facet.c9a) [concrete]
// CHECK:STDOUT:   %.a38: type = fn_type_with_self_type %Destroy.WithSelf.SelfDestruct.type.235, %Destroy.facet.c9a [concrete]
// CHECK:STDOUT:   %complete_type.95d: <witness> = complete_type_witness %tuple.type.de6 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .OverloadADL = %OverloadADL
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.4b2: @Optional.%Optional.HasValue.type (%Optional.HasValue.type.49e) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.HasValue (constants.%Optional.HasValue.c20)]
// CHECK:STDOUT:   %Core.import_ref.52c: @Optional.%Optional.Get.type (%Optional.Get.type.c64) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.Get (constants.%Optional.Get.bcf)]
// CHECK:STDOUT:   %Core.import_ref.ace: type = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%DefaultOptionalStorage (constants.%DefaultOptionalStorage.faa)]
// CHECK:STDOUT:   %Core.import_ref.ef4: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.None.type (%T.as_type.as.OptionalStorage.impl.None.type.925) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.None (constants.%T.as_type.as.OptionalStorage.impl.None.f42)]
// CHECK:STDOUT:   %Core.import_ref.a20: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Some.type (%T.as_type.as.OptionalStorage.impl.Some.type.760) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Some (constants.%T.as_type.as.OptionalStorage.impl.Some.268)]
// CHECK:STDOUT:   %Core.import_ref.af7: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Has.type (%T.as_type.as.OptionalStorage.impl.Has.type.d28) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Has (constants.%T.as_type.as.OptionalStorage.impl.Has.f29)]
// CHECK:STDOUT:   %Core.import_ref.dd4: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Get.type (%T.as_type.as.OptionalStorage.impl.Get.type.2f2) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Get (constants.%T.as_type.as.OptionalStorage.impl.Get.884)]
// CHECK:STDOUT:   %Core.import_ref.554 = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], unloaded
// CHECK:STDOUT:   %OptionalStorage.impl_witness_table.460 = impl_witness_table (%Core.import_ref.ace, %Core.import_ref.ef4, %Core.import_ref.a20, %Core.import_ref.af7, %Core.import_ref.dd4, %Core.import_ref.554), @T.as_type.as.OptionalStorage.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.1d7: @DefaultOptionalStorage.%DefaultOptionalStorage.elem.1 (%DefaultOptionalStorage.elem.9db) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [concrete = %field_decl.5f0]
// CHECK:STDOUT:   %Core.import_ref.929: @DefaultOptionalStorage.%DefaultOptionalStorage.elem.2 (%DefaultOptionalStorage.elem.06a) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [concrete = %field_decl.6a8]
// CHECK:STDOUT:   %Core.import_ref.eef: %facet_type.f48 = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.Iterate.impl.%facet_value (constants.%facet_value.f93)]
// CHECK:STDOUT:   %Core.import_ref.fcc: %Destroy.type = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.Iterate.impl.%Destroy.facet (constants.%Destroy.facet.9cd)]
// CHECK:STDOUT:   %Core.import_ref.946: @T.as_type.as.Iterate.impl.%T.as_type.as.Iterate.impl.NewCursor.type (%T.as_type.as.Iterate.impl.NewCursor.type.8fb) = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.Iterate.impl.%T.as_type.as.Iterate.impl.NewCursor (constants.%T.as_type.as.Iterate.impl.NewCursor.8a1)]
// CHECK:STDOUT:   %Core.import_ref.cfb: @T.as_type.as.Iterate.impl.%T.as_type.as.Iterate.impl.Next.type (%T.as_type.as.Iterate.impl.Next.type.d4a) = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.Iterate.impl.%T.as_type.as.Iterate.impl.Next (constants.%T.as_type.as.Iterate.impl.Next.5a8)]
// CHECK:STDOUT:   %Iterate.impl_witness_table.121 = impl_witness_table (%Core.import_ref.eef, %Core.import_ref.fcc, %Core.import_ref.946, %Core.import_ref.cfb), @T.as_type.as.Iterate.impl [concrete]
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
// CHECK:STDOUT:   %field_decl.6a8: @DefaultOptionalStorage.%DefaultOptionalStorage.elem.2 (%DefaultOptionalStorage.elem.06a) = field_decl has_value, element1, %Core.import_ref.b0d in [concrete] {}
// CHECK:STDOUT:   %Core.import_ref.87f = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %field_decl.5f0: @DefaultOptionalStorage.%DefaultOptionalStorage.elem.1 (%DefaultOptionalStorage.elem.9db) = field_decl value, element0, %Core.import_ref.87f in [concrete] {}
// CHECK:STDOUT:   %MutableAndConstRange.decl: type = class_decl @MutableAndConstRange [concrete = constants.%MutableAndConstRange] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generated {
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.1: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc60_41.1 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.2] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.60e = ref_param_pattern [concrete = constants.%self.param_patt.61b3]
// CHECK:STDOUT:     %self.patt: %pattern_type.60e = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.ecc]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %.8f3 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %.8f3 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.2: %Destroy.WithSelf.Op.type.ef016f.3 = fn_decl @Destroy.WithSelf.Op.loc60_41.1 [concrete = constants.%Destroy.WithSelf.Op.403171.3] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.60e = ref_param_pattern [concrete = constants.%self.param_patt.61b3]
// CHECK:STDOUT:     %self.patt: %pattern_type.60e = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.ecc]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %.8f3 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %.8f3 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.2: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.3 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc60_41.2 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.3] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.90d = ref_param_pattern [concrete = constants.%self.param_patt.f43]
// CHECK:STDOUT:     %self.patt: %pattern_type.90d = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.162]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %MaybeUnformed.713 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %MaybeUnformed.713 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.3: %Destroy.WithSelf.Op.type.ef016f.4 = fn_decl @Destroy.WithSelf.Op.loc60_41.2 [concrete = constants.%Destroy.WithSelf.Op.403171.4] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.90d = ref_param_pattern [concrete = constants.%self.param_patt.f43]
// CHECK:STDOUT:     %self.patt: %pattern_type.90d = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.162]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %MaybeUnformed.713 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %MaybeUnformed.713 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.3: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.4 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc60_41.3 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.4] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.831 = ref_param_pattern [concrete = constants.%self.param_patt.40d]
// CHECK:STDOUT:     %self.patt: %pattern_type.831 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.560]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref bool = ref_param call_param0
// CHECK:STDOUT:     %self: ref bool = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.4: %Destroy.WithSelf.Op.type.ef016f.5 = fn_decl @Destroy.WithSelf.Op.loc60_41.3 [concrete = constants.%Destroy.WithSelf.Op.403171.5] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.831 = ref_param_pattern [concrete = constants.%self.param_patt.40d]
// CHECK:STDOUT:     %self.patt: %pattern_type.831 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.560]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref bool = ref_param call_param0
// CHECK:STDOUT:     %self: ref bool = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.4: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.5 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc60_41.4 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.5] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.030 = ref_param_pattern [concrete = constants.%self.param_patt.c53]
// CHECK:STDOUT:     %self.patt: %pattern_type.030 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.250]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %struct_type.value.has_value.793 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %struct_type.value.has_value.793 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.5: %Destroy.WithSelf.Op.type.ef016f.6 = fn_decl @Destroy.WithSelf.Op.loc60_41.4 [concrete = constants.%Destroy.WithSelf.Op.403171.6] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.030 = ref_param_pattern [concrete = constants.%self.param_patt.c53]
// CHECK:STDOUT:     %self.patt: %pattern_type.030 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.250]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %struct_type.value.has_value.793 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %struct_type.value.has_value.793 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.5: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.6 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc60_41.5 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.6] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.6c5 = ref_param_pattern [concrete = constants.%self.param_patt.c66]
// CHECK:STDOUT:     %self.patt: %pattern_type.6c5 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.429]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %DefaultOptionalStorage.ac6 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %DefaultOptionalStorage.ac6 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.6: %Destroy.WithSelf.Op.type.ef016f.7 = fn_decl @Destroy.WithSelf.Op.loc60_41.5 [concrete = constants.%Destroy.WithSelf.Op.403171.7] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.6c5 = ref_param_pattern [concrete = constants.%self.param_patt.c66]
// CHECK:STDOUT:     %self.patt: %pattern_type.6c5 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.429]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %DefaultOptionalStorage.ac6 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %DefaultOptionalStorage.ac6 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.6: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.7 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc60_41.6 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.7] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.f76 = ref_param_pattern [concrete = constants.%self.param_patt.8f3]
// CHECK:STDOUT:     %self.patt: %pattern_type.f76 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.e6d]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %Optional.8d2 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %Optional.8d2 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.7: %Destroy.WithSelf.Op.type.ef016f.8 = fn_decl @Destroy.WithSelf.Op.loc60_41.6 [concrete = constants.%Destroy.WithSelf.Op.403171.8] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.f76 = ref_param_pattern [concrete = constants.%self.param_patt.8f3]
// CHECK:STDOUT:     %self.patt: %pattern_type.f76 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.e6d]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %Optional.8d2 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %Optional.8d2 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @TestIterate(%R.loc56_17.2: %Iterate_where.type.8b0) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Iterate.lookup_impl_witness: <witness> = lookup_impl_witness %R.loc56_17.1, @Iterate [symbolic = %Iterate.lookup_impl_witness (constants.%Iterate.lookup_impl_witness.a99)]
// CHECK:STDOUT:   %Iterate.facet.loc60_41.5: %Iterate.type = facet_value %R.as_type.loc56_83.1, (%Iterate.lookup_impl_witness) [symbolic = %Iterate.facet.loc60_41.5 (constants.%Iterate.facet.c55)]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.loc60_41.5) [symbolic = %Iterate.WithSelf.NewCursor.type (constants.%Iterate.WithSelf.NewCursor.type.fb9)]
// CHECK:STDOUT:   %.loc60_41.20: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type, %Iterate.facet.loc60_41.5 [symbolic = %.loc60_41.20 (constants.%.54f)]
// CHECK:STDOUT:   %impl.elem2.loc60_41.3: @TestIterate.%.loc60_41.20 (%.54f) = impl_witness_access %Iterate.lookup_impl_witness, element2 [symbolic = %impl.elem2.loc60_41.3 (constants.%impl.elem2.ff2)]
// CHECK:STDOUT:   %specific_impl_fn.loc60_41.4: <specific function> = specific_impl_function %impl.elem2.loc60_41.3, @Iterate.WithSelf.NewCursor(%Iterate.facet.loc60_41.5) [symbolic = %specific_impl_fn.loc60_41.4 (constants.%specific_impl_fn.541)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.loc60_41.5) [symbolic = %Iterate.WithSelf.Next.type (constants.%Iterate.WithSelf.Next.type.5d8)]
// CHECK:STDOUT:   %.loc60_41.21: type = fn_type_with_self_type %Iterate.WithSelf.Next.type, %Iterate.facet.loc60_41.5 [symbolic = %.loc60_41.21 (constants.%.216)]
// CHECK:STDOUT:   %impl.elem3.loc60_41.2: @TestIterate.%.loc60_41.21 (%.216) = impl_witness_access %Iterate.lookup_impl_witness, element3 [symbolic = %impl.elem3.loc60_41.2 (constants.%impl.elem3.5f1)]
// CHECK:STDOUT:   %specific_impl_fn.loc60_41.5: <specific function> = specific_impl_function %impl.elem3.loc60_41.2, @Iterate.WithSelf.Next(%Iterate.facet.loc60_41.5) [symbolic = %specific_impl_fn.loc60_41.5 (constants.%specific_impl_fn.eaf)]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type: type = fn_type @Destroy.WithSelf.SelfDestruct.1, @Destroy.WithSelf(%impl.elem1) [symbolic = %Destroy.WithSelf.SelfDestruct.type (constants.%Destroy.WithSelf.SelfDestruct.type.4ed)]
// CHECK:STDOUT:   %.loc60_41.22: type = fn_type_with_self_type %Destroy.WithSelf.SelfDestruct.type, %impl.elem1 [symbolic = %.loc60_41.22 (constants.%.97d)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %impl.elem1, @Destroy [symbolic = %Destroy.lookup_impl_witness (constants.%Destroy.lookup_impl_witness.eb0)]
// CHECK:STDOUT:   %impl.elem2.loc60_41.4: @TestIterate.%.loc60_41.22 (%.97d) = impl_witness_access %Destroy.lookup_impl_witness, element2 [symbolic = %impl.elem2.loc60_41.4 (constants.%impl.elem2.f7b)]
// CHECK:STDOUT:   %specific_impl_fn.loc60_41.6: <specific function> = specific_impl_function %impl.elem2.loc60_41.4, @Destroy.WithSelf.SelfDestruct.1(%impl.elem1) [symbolic = %specific_impl_fn.loc60_41.6 (constants.%specific_impl_fn.577)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%r.param: @TestIterate.%R.as_type.loc56_83.1 (%R.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %i.patt: %pattern_type.e09 = value_binding_pattern i [concrete = constants.%i.patt]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %r.ref: @TestIterate.%R.as_type.loc56_83.1 (%R.as_type) = name_ref r, %r
// CHECK:STDOUT:     %impl.elem2.loc60_41.1: @TestIterate.%.loc60_41.20 (%.54f) = impl_witness_access constants.%Iterate.lookup_impl_witness.a99, element2 [symbolic = %impl.elem2.loc60_41.3 (constants.%impl.elem2.ff2)]
// CHECK:STDOUT:     %bound_method.loc60_41.1: <bound method> = bound_method %r.ref, %impl.elem2.loc60_41.1
// CHECK:STDOUT:     %Iterate.facet.loc60_41.1: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.a99) [symbolic = %Iterate.facet.loc60_41.5 (constants.%Iterate.facet.c55)]
// CHECK:STDOUT:     %.loc60_41.1: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc60_41.1 [symbolic = %Iterate.facet.loc60_41.5 (constants.%Iterate.facet.c55)]
// CHECK:STDOUT:     %Iterate.facet.loc60_41.2: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.a99) [symbolic = %Iterate.facet.loc60_41.5 (constants.%Iterate.facet.c55)]
// CHECK:STDOUT:     %.loc60_41.2: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc60_41.2 [symbolic = %Iterate.facet.loc60_41.5 (constants.%Iterate.facet.c55)]
// CHECK:STDOUT:     %specific_impl_fn.loc60_41.1: <specific function> = specific_impl_function %impl.elem2.loc60_41.1, @Iterate.WithSelf.NewCursor(constants.%Iterate.facet.c55) [symbolic = %specific_impl_fn.loc60_41.4 (constants.%specific_impl_fn.541)]
// CHECK:STDOUT:     %bound_method.loc60_41.2: <bound method> = bound_method %r.ref, %specific_impl_fn.loc60_41.1
// CHECK:STDOUT:     %var: ref @TestIterate.%as_type (%as_type.733) = var_storage invalid
// CHECK:STDOUT:     %Iterate.WithSelf.NewCursor.call: init @TestIterate.%as_type (%as_type.733) to %var = call %bound_method.loc60_41.2(%r.ref)
// CHECK:STDOUT:     assign %var, %Iterate.WithSelf.NewCursor.call
// CHECK:STDOUT:     br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.next:
// CHECK:STDOUT:     %impl.elem3.loc60_41.1: @TestIterate.%.loc60_41.21 (%.216) = impl_witness_access constants.%Iterate.lookup_impl_witness.a99, element3 [symbolic = %impl.elem3.loc60_41.2 (constants.%impl.elem3.5f1)]
// CHECK:STDOUT:     %bound_method.loc60_41.3: <bound method> = bound_method %r.ref, %impl.elem3.loc60_41.1
// CHECK:STDOUT:     %Iterate.facet.loc60_41.3: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.a99) [symbolic = %Iterate.facet.loc60_41.5 (constants.%Iterate.facet.c55)]
// CHECK:STDOUT:     %.loc60_41.3: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc60_41.3 [symbolic = %Iterate.facet.loc60_41.5 (constants.%Iterate.facet.c55)]
// CHECK:STDOUT:     %Iterate.facet.loc60_41.4: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.a99) [symbolic = %Iterate.facet.loc60_41.5 (constants.%Iterate.facet.c55)]
// CHECK:STDOUT:     %.loc60_41.4: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc60_41.4 [symbolic = %Iterate.facet.loc60_41.5 (constants.%Iterate.facet.c55)]
// CHECK:STDOUT:     %.loc60_41.5: %Destroy.type = converted constants.%as_type.733, constants.%impl.elem1.751 [symbolic = %impl.elem1 (constants.%impl.elem1.751)]
// CHECK:STDOUT:     %.loc60_41.6: %Destroy.type = converted constants.%as_type.733, constants.%impl.elem1.751 [symbolic = %impl.elem1 (constants.%impl.elem1.751)]
// CHECK:STDOUT:     %specific_impl_fn.loc60_41.2: <specific function> = specific_impl_function %impl.elem3.loc60_41.1, @Iterate.WithSelf.Next(constants.%Iterate.facet.c55) [symbolic = %specific_impl_fn.loc60_41.5 (constants.%specific_impl_fn.eaf)]
// CHECK:STDOUT:     %bound_method.loc60_41.4: <bound method> = bound_method %r.ref, %specific_impl_fn.loc60_41.2
// CHECK:STDOUT:     %.loc60_41.7: ref %Optional.8d2 = temporary_storage
// CHECK:STDOUT:     %Iterate.WithSelf.Next.call: init %Optional.8d2 to %.loc60_41.7 = call %bound_method.loc60_41.4(%r.ref, %var)
// CHECK:STDOUT:     %.loc60_41.8: ref %Optional.8d2 = temporary %.loc60_41.7, %Iterate.WithSelf.Next.call
// CHECK:STDOUT:     %.loc60_41.9: %Optional.HasValue.type.10c = specific_constant imports.%Core.import_ref.4b2, @Optional(constants.%OptionalStorage.facet.adc) [concrete = constants.%Optional.HasValue.e84]
// CHECK:STDOUT:     %HasValue.ref: %Optional.HasValue.type.10c = name_ref HasValue, %.loc60_41.9 [concrete = constants.%Optional.HasValue.e84]
// CHECK:STDOUT:     %Optional.HasValue.bound: <bound method> = bound_method %.loc60_41.8, %HasValue.ref
// CHECK:STDOUT:     %Optional.HasValue.specific_fn: <specific function> = specific_function %HasValue.ref, @Optional.HasValue(constants.%OptionalStorage.facet.adc) [concrete = constants.%Optional.HasValue.specific_fn.a99]
// CHECK:STDOUT:     %bound_method.loc60_41.5: <bound method> = bound_method %.loc60_41.8, %Optional.HasValue.specific_fn
// CHECK:STDOUT:     %.loc60_41.10: %Optional.8d2 = acquire_value %.loc60_41.8
// CHECK:STDOUT:     %Optional.HasValue.call: init bool = call %bound_method.loc60_41.5(%.loc60_41.10)
// CHECK:STDOUT:     %.loc60_41.11: bool = value_of_initializer %Optional.HasValue.call
// CHECK:STDOUT:     %.loc60_41.12: bool = converted %Optional.HasValue.call, %.loc60_41.11
// CHECK:STDOUT:     if %.loc60_41.12 br !for.body else br !for.done
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.body:
// CHECK:STDOUT:     %.loc60_41.13: %Optional.Get.type.bc3 = specific_constant imports.%Core.import_ref.52c, @Optional(constants.%OptionalStorage.facet.adc) [concrete = constants.%Optional.Get.d28]
// CHECK:STDOUT:     %Get.ref: %Optional.Get.type.bc3 = name_ref Get, %.loc60_41.13 [concrete = constants.%Optional.Get.d28]
// CHECK:STDOUT:     %Optional.Get.bound: <bound method> = bound_method %.loc60_41.8, %Get.ref
// CHECK:STDOUT:     %Optional.Get.specific_fn: <specific function> = specific_function %Get.ref, @Optional.Get(constants.%OptionalStorage.facet.adc) [concrete = constants.%Optional.Get.specific_fn.22d]
// CHECK:STDOUT:     %bound_method.loc60_41.6: <bound method> = bound_method %.loc60_41.8, %Optional.Get.specific_fn
// CHECK:STDOUT:     %.loc60_41.14: ref %ValueType = temporary_storage
// CHECK:STDOUT:     %.loc60_41.15: %Optional.8d2 = acquire_value %.loc60_41.8
// CHECK:STDOUT:     %Optional.Get.call: init %ValueType to %.loc60_41.14 = call %bound_method.loc60_41.6(%.loc60_41.15)
// CHECK:STDOUT:     %.loc60_41.16: ref %ValueType = temporary %.loc60_41.14, %Optional.Get.call
// CHECK:STDOUT:     %.loc60_41.17: %ValueType = acquire_value %.loc60_41.16
// CHECK:STDOUT:     %.loc60_26: type = splice_block %ValueType.ref.loc60 [concrete = constants.%ValueType] {
// CHECK:STDOUT:       %Cpp.ref.loc60: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:       %OverloadADL.ref.loc60: <namespace> = name_ref OverloadADL, imports.%OverloadADL [concrete = imports.%OverloadADL]
// CHECK:STDOUT:       %ValueType.ref.loc60: type = name_ref ValueType, imports.%ValueType.decl [concrete = constants.%ValueType]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %i: %ValueType = wrapper_binding i, %.loc60_41.17
// CHECK:STDOUT:     %sum.ref: ref %ValueType = name_ref sum, %sum
// CHECK:STDOUT:     %i.ref: %ValueType = name_ref i, %i
// CHECK:STDOUT:     %.loc61_12: ref %ValueType = value_as_ref %i.ref
// CHECK:STDOUT:     %addr.loc61: %ptr.6ed = addr_of %.loc61_12
// CHECK:STDOUT:     %.loc61_9.1: %ptr.403 = as_compatible %addr.loc61
// CHECK:STDOUT:     %.loc61_9.2: %ptr.403 = converted %addr.loc61, %.loc61_9.1
// CHECK:STDOUT:     %operator_PlusEqual__carbon_thunk.call: ref %ValueType = call imports.%operator_PlusEqual__carbon_thunk.decl(%sum.ref, %.loc61_9.2)
// CHECK:STDOUT:     %Destroy.WithSelf.SelfDestruct.bound.loc60_41.1: <bound method> = bound_method %.loc60_41.16, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.2
// CHECK:STDOUT:     %Destroy.WithSelf.SelfDestruct.call.loc60_41.1: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc60_41.1(%.loc60_41.16)
// CHECK:STDOUT:     %Destroy.WithSelf.SelfDestruct.bound.loc60_41.2: <bound method> = bound_method %.loc60_41.8, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.8
// CHECK:STDOUT:     %Destroy.WithSelf.SelfDestruct.call.loc60_41.2: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc60_41.2(%.loc60_41.8)
// CHECK:STDOUT:     br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.done:
// CHECK:STDOUT:     %Destroy.WithSelf.SelfDestruct.bound.loc60_41.3: <bound method> = bound_method %.loc60_41.8, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.8
// CHECK:STDOUT:     %Destroy.WithSelf.SelfDestruct.call.loc60_41.3: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc60_41.3(%.loc60_41.8)
// CHECK:STDOUT:     %impl.elem2.loc60_41.2: @TestIterate.%.loc60_41.22 (%.97d) = impl_witness_access constants.%Destroy.lookup_impl_witness.eb0, element2 [symbolic = %impl.elem2.loc60_41.4 (constants.%impl.elem2.f7b)]
// CHECK:STDOUT:     %bound_method.loc60_41.7: <bound method> = bound_method %var, %impl.elem2.loc60_41.2
// CHECK:STDOUT:     %.loc60_41.18: %Destroy.type = converted constants.%as_type.733, constants.%impl.elem1.751 [symbolic = %impl.elem1 (constants.%impl.elem1.751)]
// CHECK:STDOUT:     %.loc60_41.19: %Destroy.type = converted constants.%as_type.733, constants.%impl.elem1.751 [symbolic = %impl.elem1 (constants.%impl.elem1.751)]
// CHECK:STDOUT:     %specific_impl_fn.loc60_41.3: <specific function> = specific_impl_function %impl.elem2.loc60_41.2, @Destroy.WithSelf.SelfDestruct.1(constants.%impl.elem1.751) [symbolic = %specific_impl_fn.loc60_41.6 (constants.%specific_impl_fn.577)]
// CHECK:STDOUT:     %bound_method.loc60_41.8: <bound method> = bound_method %var, %specific_impl_fn.loc60_41.3
// CHECK:STDOUT:     %Destroy.WithSelf.SelfDestruct.call.loc60_41.4: init %empty_tuple.type = call %bound_method.loc60_41.8(%var)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc60_41.1(%self.param: ref %.8f3) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc60_41.1(%self.param: ref %.8f3) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc60_41.1(%self.param: ref %.8f3) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.2(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.1(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc60_41.2(%self.param: ref %MaybeUnformed.713) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc60_41.2(%self.param: ref %MaybeUnformed.713) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc60_41.2(%self.param: ref %MaybeUnformed.713) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.3(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.2(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc60_41.3(%self.param: ref bool) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc60_41.3(%self.param: ref bool) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc60_41.3(%self.param: ref bool) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.4(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.3(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc60_41.4(%self.param: ref %struct_type.value.has_value.793) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc60_41.1: ref bool = struct_access %self.param, element1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc60_41.1: <bound method> = bound_method %.loc60_41.1, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.5
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc60_41.1: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc60_41.1(%.loc60_41.1)
// CHECK:STDOUT:   %.loc60_41.2: ref %MaybeUnformed.713 = struct_access %self.param, element0
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc60_41.2: <bound method> = bound_method %.loc60_41.2, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.4
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc60_41.2: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc60_41.2(%.loc60_41.2)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc60_41.4(%self.param: ref %struct_type.value.has_value.793) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc60_41.4(%self.param: ref %struct_type.value.has_value.793) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.5(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.4(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc60_41.5(%self.param: ref %DefaultOptionalStorage.ac6) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %has_value.ref: %DefaultOptionalStorage.elem.a4b = name_ref has_value, imports.%Core.import_ref.929 [concrete = imports.%field_decl.6a8]
// CHECK:STDOUT:   %.loc60_41.1: ref bool = class_element_access %self.param, element1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc60_41.1: <bound method> = bound_method %.loc60_41.1, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.5
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc60_41.1: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc60_41.1(%.loc60_41.1)
// CHECK:STDOUT:   %value.ref: %DefaultOptionalStorage.elem.1eb = name_ref value, imports.%Core.import_ref.1d7 [concrete = imports.%field_decl.5f0]
// CHECK:STDOUT:   %.loc60_41.2: ref %MaybeUnformed.713 = class_element_access %self.param, element0
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc60_41.2: <bound method> = bound_method %.loc60_41.2, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.4
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc60_41.2: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc60_41.2(%.loc60_41.2)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc60_41.5(%self.param: ref %DefaultOptionalStorage.ac6) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc60_41.5(%self.param: ref %DefaultOptionalStorage.ac6) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.6(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.5(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc60_41.6(%self.param: ref %Optional.8d2) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc60_41.6(%self.param: ref %Optional.8d2) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc60_41.6(%self.param: ref %Optional.8d2) {
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
// CHECK:STDOUT:   %impl.elem2: %.bc1 = impl_witness_access constants.%Iterate.impl_witness.198, element2 [concrete = constants.%T.as_type.as.Iterate.impl.NewCursor.160]
// CHECK:STDOUT:   %bound_method.loc74_42.1: <bound method> = bound_method %mc.ref, %impl.elem2
// CHECK:STDOUT:   %facet_value.loc74_42.1: %CppRange_where.type.d682d6.1 = facet_value constants.%MutableAndConstRange, (constants.%custom_witness.660, constants.%custom_witness.809, constants.%custom_witness.d56, constants.%custom_witness.68a, constants.%custom_witness.edd, constants.%custom_witness.dc3, constants.%custom_witness.1c5, constants.%custom_witness.809) [concrete = constants.%facet_value.24f]
// CHECK:STDOUT:   %.loc74_42.1: %CppRange_where.type.d682d6.1 = converted constants.%MutableAndConstRange, %facet_value.loc74_42.1 [concrete = constants.%facet_value.24f]
// CHECK:STDOUT:   %facet_value.loc74_42.2: %CppRange_where.type.d682d6.1 = facet_value constants.%MutableAndConstRange, (constants.%custom_witness.660, constants.%custom_witness.809, constants.%custom_witness.d56, constants.%custom_witness.68a, constants.%custom_witness.edd, constants.%custom_witness.dc3, constants.%custom_witness.1c5, constants.%custom_witness.809) [concrete = constants.%facet_value.24f]
// CHECK:STDOUT:   %.loc74_42.2: %CppRange_where.type.d682d6.1 = converted constants.%MutableAndConstRange, %facet_value.loc74_42.2 [concrete = constants.%facet_value.24f]
// CHECK:STDOUT:   %specific_fn.loc74_42.1: <specific function> = specific_function %impl.elem2, @T.as_type.as.Iterate.impl.NewCursor(constants.%facet_value.24f) [concrete = constants.%T.as_type.as.Iterate.impl.NewCursor.specific_fn]
// CHECK:STDOUT:   %bound_method.loc74_42.2: <bound method> = bound_method %mc.ref, %specific_fn.loc74_42.1
// CHECK:STDOUT:   %var: ref %tuple.type.de6 = var_storage invalid
// CHECK:STDOUT:   %.loc74_40.1: %MutableAndConstRange = acquire_value %mc.ref
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.call: init %tuple.type.de6 to %var = call %bound_method.loc74_42.2(%.loc74_40.1)
// CHECK:STDOUT:   assign %var, %T.as_type.as.Iterate.impl.NewCursor.call
// CHECK:STDOUT:   br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT: !for.next:
// CHECK:STDOUT:   %impl.elem3: %.698 = impl_witness_access constants.%Iterate.impl_witness.198, element3 [concrete = constants.%T.as_type.as.Iterate.impl.Next.ddf]
// CHECK:STDOUT:   %bound_method.loc74_42.3: <bound method> = bound_method %mc.ref, %impl.elem3
// CHECK:STDOUT:   %facet_value.loc74_42.3: %CppRange_where.type.d682d6.1 = facet_value constants.%MutableAndConstRange, (constants.%custom_witness.660, constants.%custom_witness.809, constants.%custom_witness.d56, constants.%custom_witness.68a, constants.%custom_witness.edd, constants.%custom_witness.dc3, constants.%custom_witness.1c5, constants.%custom_witness.809) [concrete = constants.%facet_value.24f]
// CHECK:STDOUT:   %.loc74_42.3: %CppRange_where.type.d682d6.1 = converted constants.%MutableAndConstRange, %facet_value.loc74_42.3 [concrete = constants.%facet_value.24f]
// CHECK:STDOUT:   %facet_value.loc74_42.4: %CppRange_where.type.d682d6.1 = facet_value constants.%MutableAndConstRange, (constants.%custom_witness.660, constants.%custom_witness.809, constants.%custom_witness.d56, constants.%custom_witness.68a, constants.%custom_witness.edd, constants.%custom_witness.dc3, constants.%custom_witness.1c5, constants.%custom_witness.809) [concrete = constants.%facet_value.24f]
// CHECK:STDOUT:   %.loc74_42.4: %CppRange_where.type.d682d6.1 = converted constants.%MutableAndConstRange, %facet_value.loc74_42.4 [concrete = constants.%facet_value.24f]
// CHECK:STDOUT:   %specific_fn.loc74_42.2: <specific function> = specific_function %impl.elem3, @T.as_type.as.Iterate.impl.Next(constants.%facet_value.24f) [concrete = constants.%T.as_type.as.Iterate.impl.Next.specific_fn]
// CHECK:STDOUT:   %bound_method.loc74_42.4: <bound method> = bound_method %mc.ref, %specific_fn.loc74_42.2
// CHECK:STDOUT:   %.loc74_42.5: ref %Optional.8d2 = temporary_storage
// CHECK:STDOUT:   %.loc74_40.2: %MutableAndConstRange = acquire_value %mc.ref
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.call: init %Optional.8d2 to %.loc74_42.5 = call %bound_method.loc74_42.4(%.loc74_40.2, %var)
// CHECK:STDOUT:   %.loc74_42.6: ref %Optional.8d2 = temporary %.loc74_42.5, %T.as_type.as.Iterate.impl.Next.call
// CHECK:STDOUT:   %.loc74_42.7: %Optional.HasValue.type.10c = specific_constant imports.%Core.import_ref.4b2, @Optional(constants.%OptionalStorage.facet.adc) [concrete = constants.%Optional.HasValue.e84]
// CHECK:STDOUT:   %HasValue.ref: %Optional.HasValue.type.10c = name_ref HasValue, %.loc74_42.7 [concrete = constants.%Optional.HasValue.e84]
// CHECK:STDOUT:   %Optional.HasValue.bound: <bound method> = bound_method %.loc74_42.6, %HasValue.ref
// CHECK:STDOUT:   %Optional.HasValue.specific_fn: <specific function> = specific_function %HasValue.ref, @Optional.HasValue(constants.%OptionalStorage.facet.adc) [concrete = constants.%Optional.HasValue.specific_fn.a99]
// CHECK:STDOUT:   %bound_method.loc74_42.5: <bound method> = bound_method %.loc74_42.6, %Optional.HasValue.specific_fn
// CHECK:STDOUT:   %.loc74_42.8: %Optional.8d2 = acquire_value %.loc74_42.6
// CHECK:STDOUT:   %Optional.HasValue.call: init bool = call %bound_method.loc74_42.5(%.loc74_42.8)
// CHECK:STDOUT:   %.loc74_42.9: bool = value_of_initializer %Optional.HasValue.call
// CHECK:STDOUT:   %.loc74_42.10: bool = converted %Optional.HasValue.call, %.loc74_42.9
// CHECK:STDOUT:   if %.loc74_42.10 br !for.body else br !for.done
// CHECK:STDOUT:
// CHECK:STDOUT: !for.body:
// CHECK:STDOUT:   %.loc74_42.11: %Optional.Get.type.bc3 = specific_constant imports.%Core.import_ref.52c, @Optional(constants.%OptionalStorage.facet.adc) [concrete = constants.%Optional.Get.d28]
// CHECK:STDOUT:   %Get.ref: %Optional.Get.type.bc3 = name_ref Get, %.loc74_42.11 [concrete = constants.%Optional.Get.d28]
// CHECK:STDOUT:   %Optional.Get.bound: <bound method> = bound_method %.loc74_42.6, %Get.ref
// CHECK:STDOUT:   %Optional.Get.specific_fn: <specific function> = specific_function %Get.ref, @Optional.Get(constants.%OptionalStorage.facet.adc) [concrete = constants.%Optional.Get.specific_fn.22d]
// CHECK:STDOUT:   %bound_method.loc74_42.6: <bound method> = bound_method %.loc74_42.6, %Optional.Get.specific_fn
// CHECK:STDOUT:   %.loc74_42.12: ref %ValueType = temporary_storage
// CHECK:STDOUT:   %.loc74_42.13: %Optional.8d2 = acquire_value %.loc74_42.6
// CHECK:STDOUT:   %Optional.Get.call: init %ValueType to %.loc74_42.12 = call %bound_method.loc74_42.6(%.loc74_42.13)
// CHECK:STDOUT:   %.loc74_42.14: ref %ValueType = temporary %.loc74_42.12, %Optional.Get.call
// CHECK:STDOUT:   %.loc74_42.15: %ValueType = acquire_value %.loc74_42.14
// CHECK:STDOUT:   %.loc74_26: type = splice_block %ValueType.ref.loc74 [concrete = constants.%ValueType] {
// CHECK:STDOUT:     %Cpp.ref.loc74: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %OverloadADL.ref.loc74: <namespace> = name_ref OverloadADL, imports.%OverloadADL [concrete = imports.%OverloadADL]
// CHECK:STDOUT:     %ValueType.ref.loc74: type = name_ref ValueType, imports.%ValueType.decl [concrete = constants.%ValueType]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %i: %ValueType = wrapper_binding i, %.loc74_42.15
// CHECK:STDOUT:   %sum.ref: ref %ValueType = name_ref sum, %sum
// CHECK:STDOUT:   %i.ref: %ValueType = name_ref i, %i
// CHECK:STDOUT:   %.loc75_12: ref %ValueType = value_as_ref %i.ref
// CHECK:STDOUT:   %addr.loc75: %ptr.6ed = addr_of %.loc75_12
// CHECK:STDOUT:   %.loc75_9.1: %ptr.403 = as_compatible %addr.loc75
// CHECK:STDOUT:   %.loc75_9.2: %ptr.403 = converted %addr.loc75, %.loc75_9.1
// CHECK:STDOUT:   %operator_PlusEqual__carbon_thunk.call: ref %ValueType = call imports.%operator_PlusEqual__carbon_thunk.decl(%sum.ref, %.loc75_9.2)
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc74_42.1: <bound method> = bound_method %.loc74_42.14, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.2
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc74_42.1: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc74_42.1(%.loc74_42.14)
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc74_42.2: <bound method> = bound_method %.loc74_42.6, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.8
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc74_42.2: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc74_42.2(%.loc74_42.6)
// CHECK:STDOUT:   br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT: !for.done:
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc74_42.3: <bound method> = bound_method %.loc74_42.6, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.8
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc74_42.3: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc74_42.3(%.loc74_42.6)
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc74_42.4: <bound method> = bound_method %var, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.10
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc74_42.4: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc74_42.4(%var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestIterate(constants.%R) {
// CHECK:STDOUT:   %R.patt.loc56_17.2 => constants.%R.patt
// CHECK:STDOUT:   %R.loc56_17.1 => constants.%R
// CHECK:STDOUT:   %R.as_type.loc56_83.1 => constants.%R.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.611
// CHECK:STDOUT:   %r.param_patt.loc56_81.2 => constants.%r.param_patt.e64
// CHECK:STDOUT:   %r.patt.loc56_81.2 => constants.%r.patt.dcb
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestIterate(constants.%facet_value.fc8) {
// CHECK:STDOUT:   %R.patt.loc56_17.2 => constants.%R.patt
// CHECK:STDOUT:   %R.loc56_17.1 => constants.%facet_value.fc8
// CHECK:STDOUT:   %R.as_type.loc56_83.1 => constants.%MutableAndConstRange
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.fdb
// CHECK:STDOUT:   %r.param_patt.loc56_81.2 => constants.%r.param_patt.3a6
// CHECK:STDOUT:   %r.patt.loc56_81.2 => constants.%r.patt.403
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc56 => constants.%complete_type.357
// CHECK:STDOUT:   %Iterate.lookup_impl_witness => constants.%Iterate.impl_witness.198
// CHECK:STDOUT:   %Iterate.facet.loc60_41.5 => constants.%Iterate.facet.b30
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type => constants.%Iterate.WithSelf.NewCursor.type.b5a
// CHECK:STDOUT:   %.loc60_41.20 => constants.%.bc1
// CHECK:STDOUT:   %impl.elem2.loc60_41.3 => constants.%T.as_type.as.Iterate.impl.NewCursor.160
// CHECK:STDOUT:   %specific_impl_fn.loc60_41.4 => constants.%T.as_type.as.Iterate.impl.NewCursor.specific_fn
// CHECK:STDOUT:   %impl.elem1 => constants.%Destroy.facet.c9a
// CHECK:STDOUT:   %as_type => constants.%tuple.type.de6
// CHECK:STDOUT:   %require_complete.1 => constants.%complete_type.95d
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type => constants.%Iterate.WithSelf.Next.type.3b0
// CHECK:STDOUT:   %.loc60_41.21 => constants.%.698
// CHECK:STDOUT:   %impl.elem3.loc60_41.2 => constants.%T.as_type.as.Iterate.impl.Next.ddf
// CHECK:STDOUT:   %specific_impl_fn.loc60_41.5 => constants.%T.as_type.as.Iterate.impl.Next.specific_fn
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type => constants.%Destroy.WithSelf.SelfDestruct.type.235
// CHECK:STDOUT:   %.loc60_41.22 => constants.%.a38
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.f8f19d.8
// CHECK:STDOUT:   %impl.elem2.loc60_41.4 => constants.%Destroy.WithSelf.SelfDestruct.db3fdb.10
// CHECK:STDOUT:   %specific_impl_fn.loc60_41.6 => constants.%Destroy.WithSelf.SelfDestruct.db3fdb.10
// CHECK:STDOUT: }
// CHECK:STDOUT:
