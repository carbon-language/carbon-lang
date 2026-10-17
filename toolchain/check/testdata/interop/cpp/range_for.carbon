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
  // CHECK:STDERR: fail_todo_adl_returns_different_types_mutable.carbon:[[@LINE+7]]:3: error: cannot convert type `Cpp.DiffMutableADL.MutableRange` into type implementing `Core.Iterate where .(Core.Iterate.ElementType) = Cpp.DiffMutableADL.ValueType as Core.SubobjectDestroy & Core.DestroyImpl & Core.Copy` [ConversionFailureTypeToFacet]
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
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %facet_type.d61: type = facet_type <@SubobjectDestroy & @DestroyImpl & @Copy> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %T.220: %OptionalStorage.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Optional.Get.type.786: type = fn_type @Optional.Get, @Optional(%T.220) [symbolic]
// CHECK:STDOUT:   %Optional.Get.2d0: %Optional.Get.type.786 = struct_value () [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.type.3ea: type = fn_type @Optional.HasValue, @Optional(%T.220) [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.f2f: %Optional.HasValue.type.3ea = struct_value () [symbolic]
// CHECK:STDOUT:   %facet_type.8a3: type = facet_type <@SubobjectDestroy & @DestroyImpl> [concrete]
// CHECK:STDOUT:   %T.9ab: %facet_type.d61 = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.f15: type = class_type @DefaultOptionalStorage, @DefaultOptionalStorage(%T.9ab) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Get.type.3e7: type = fn_type @T.as_type.as.OptionalStorage.impl.Get, @T.as_type.as.OptionalStorage.impl(%T.9ab) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Get.f80: %T.as_type.as.OptionalStorage.impl.Get.type.3e7 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Has.type.371: type = fn_type @T.as_type.as.OptionalStorage.impl.Has, @T.as_type.as.OptionalStorage.impl(%T.9ab) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Has.8ce: %T.as_type.as.OptionalStorage.impl.Has.type.371 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Some.type.4eb: type = fn_type @T.as_type.as.OptionalStorage.impl.Some, @T.as_type.as.OptionalStorage.impl(%T.9ab) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Some.44a: %T.as_type.as.OptionalStorage.impl.Some.type.4eb = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.None.type.76a: type = fn_type @T.as_type.as.OptionalStorage.impl.None, @T.as_type.as.OptionalStorage.impl(%T.9ab) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.None.93b: %T.as_type.as.OptionalStorage.impl.None.type.76a = struct_value () [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %N.fe9: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %Int.d4bbe0.1: type = class_type @Int, @Int(%N.fe9) [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.d94: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.1: type = fn_type @SubobjectDestroy.WithSelf.Op.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.1: %SubobjectDestroy.WithSelf.Op.type.6870fd.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.1: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.1), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.facet.49a: %SubobjectDestroy.type = facet_value %i32, (%custom_witness.b80ef6.1) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.077: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.49a) [concrete]
// CHECK:STDOUT:   %facet_value.073: %facet_type.8a3 = facet_value %i32, (%custom_witness.b80ef6.1, %DestroyImpl.impl_witness.077) [concrete]
// CHECK:STDOUT:   %.Self.500: %Iterate.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Int.d4bbe0.2: type = class_type @Int, @Int(%To) [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.6b0: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
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
// CHECK:STDOUT:   %SubobjectDestroy.lookup_impl_witness.dc9: <witness> = lookup_impl_witness %tuple.type.1ba, @SubobjectDestroy [symbolic]
// CHECK:STDOUT:   %DestroyImpl.lookup_impl_witness.504: <witness> = lookup_impl_witness %tuple.type.1ba, @DestroyImpl [symbolic]
// CHECK:STDOUT:   %facet_value.bf6: %facet_type.8a3 = facet_value %tuple.type.1ba, (%SubobjectDestroy.lookup_impl_witness.dc9, %DestroyImpl.lookup_impl_witness.504) [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.lookup_impl_witness.b15da7.2: <witness> = lookup_impl_witness %impl.elem0.493765.2, @SubobjectDestroy [symbolic]
// CHECK:STDOUT:   %DestroyImpl.lookup_impl_witness.4e7f44.2: <witness> = lookup_impl_witness %impl.elem0.493765.2, @DestroyImpl [symbolic]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.167: <witness> = lookup_impl_witness %impl.elem0.493765.2, @Copy [symbolic]
// CHECK:STDOUT:   %facet_value.9da: %facet_type.d61 = facet_value %impl.elem0.493765.2, (%SubobjectDestroy.lookup_impl_witness.b15da7.2, %DestroyImpl.lookup_impl_witness.4e7f44.2, %Copy.lookup_impl_witness.167) [symbolic]
// CHECK:STDOUT:   %SelfDestruct.type: type = fn_type @SelfDestruct [concrete]
// CHECK:STDOUT:   %SelfDestruct: %SelfDestruct.type = struct_value () [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.a1d: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N.fe9) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.3c1: %Int.as.Copy.impl.Op.type.a1d = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.8af: <witness> = impl_witness imports.%Copy.impl_witness_table.f44, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %facet_value.344: %facet_type.d61 = facet_value %i32, (%custom_witness.b80ef6.1, %DestroyImpl.impl_witness.077, %Copy.impl_witness.8af) [concrete]
// CHECK:STDOUT:   %Iterate.lookup_impl_witness.1d8: <witness> = lookup_impl_witness %.Self.500, @Iterate [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.8a4: %facet_type.d61 = impl_witness_access %Iterate.lookup_impl_witness.1d8, element0 [symbolic_self]
// CHECK:STDOUT:   %Iterate_where.type.35a: type = facet_type <@Iterate where %impl.elem0.8a4 = %facet_value.344> [concrete]
// CHECK:STDOUT:   %pattern_type.0de: type = pattern_type %Iterate_where.type.35a [concrete]
// CHECK:STDOUT:   %R.patt: %pattern_type.0de = symbolic_binding_pattern R, 0 [symbolic]
// CHECK:STDOUT:   %R: %Iterate_where.type.35a = symbolic_binding R, 0 [symbolic]
// CHECK:STDOUT:   %R.as_type: type = facet_access_type %R [symbolic]
// CHECK:STDOUT:   %pattern_type.007: type = pattern_type %R.as_type [symbolic]
// CHECK:STDOUT:   %r.param_patt.553: %pattern_type.007 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %r.patt.b42: %pattern_type.007 = wrapper_binding_pattern r, %r.param_patt.553 [symbolic]
// CHECK:STDOUT:   %facet_type.08a: type = facet_type <@IntFitsIn, @IntFitsIn(%Int.d4bbe0.2) & @AnyInt> [symbolic]
// CHECK:STDOUT:   %From.170: %facet_type.08a = symbolic_binding From, 1 [symbolic]
// CHECK:STDOUT:   %From.as_type.as.ImplicitAs.impl.Convert.type.386: type = fn_type @From.as_type.as.ImplicitAs.impl.Convert, @From.as_type.as.ImplicitAs.impl(%To, %From.170) [symbolic]
// CHECK:STDOUT:   %From.as_type.as.ImplicitAs.impl.Convert.85a: %From.as_type.as.ImplicitAs.impl.Convert.type.386 = struct_value () [symbolic]
// CHECK:STDOUT:   %i.patt: %pattern_type.d94 = value_binding_pattern i [concrete]
// CHECK:STDOUT:   %Iterate.lookup_impl_witness.8f0: <witness> = lookup_impl_witness %R, @Iterate [symbolic]
// CHECK:STDOUT:   %Iterate.facet.39d: %Iterate.type = facet_value %R.as_type, (%Iterate.lookup_impl_witness.8f0) [symbolic]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.b8b: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.39d) [symbolic]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.f8e: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.39d) [symbolic]
// CHECK:STDOUT:   %.fd4: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.b8b, %Iterate.facet.39d [symbolic]
// CHECK:STDOUT:   %impl.elem2.c03: %.fd4 = impl_witness_access %Iterate.lookup_impl_witness.8f0, element2 [symbolic]
// CHECK:STDOUT:   %impl.elem1.f1d: %facet_type.8a3 = impl_witness_access %Iterate.lookup_impl_witness.8f0, element1 [symbolic]
// CHECK:STDOUT:   %as_type.9f4: type = facet_access_type %impl.elem1.f1d [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.9fbe: <specific function> = specific_impl_function %impl.elem2.c03, @Iterate.WithSelf.NewCursor(%Iterate.facet.39d) [symbolic]
// CHECK:STDOUT:   %.765: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.f8e, %Iterate.facet.39d [symbolic]
// CHECK:STDOUT:   %impl.elem3.297: %.765 = impl_witness_access %Iterate.lookup_impl_witness.8f0, element3 [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.6ca: type = class_type @DefaultOptionalStorage, @DefaultOptionalStorage(%facet_value.344) [concrete]
// CHECK:STDOUT:   %OptionalStorage.impl_witness.ae6: <witness> = impl_witness imports.%OptionalStorage.impl_witness_table.968, @T.as_type.as.OptionalStorage.impl(%facet_value.344) [concrete]
// CHECK:STDOUT:   %OptionalStorage.facet.c8d: %OptionalStorage.type = facet_value %i32, (%OptionalStorage.impl_witness.ae6) [concrete]
// CHECK:STDOUT:   %Optional.fb0: type = class_type @Optional, @Optional(%OptionalStorage.facet.c8d) [concrete]
// CHECK:STDOUT:   %specific_impl_fn.e3a: <specific function> = specific_impl_function %impl.elem3.297, @Iterate.WithSelf.Next(%Iterate.facet.39d) [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.type.be5: type = fn_type @Optional.HasValue, @Optional(%OptionalStorage.facet.c8d) [concrete]
// CHECK:STDOUT:   %Optional.HasValue.b32: %Optional.HasValue.type.be5 = struct_value () [concrete]
// CHECK:STDOUT:   %Optional.Get.type.186: type = fn_type @Optional.Get, @Optional(%OptionalStorage.facet.c8d) [concrete]
// CHECK:STDOUT:   %Optional.Get.998: %Optional.Get.type.186 = struct_value () [concrete]
// CHECK:STDOUT:   %MaybeUnformed.a56: type = class_type @MaybeUnformed, @MaybeUnformed(%facet_value.073) [concrete]
// CHECK:STDOUT:   %.b31: type = maybe_unformed_type %i32 [concrete]
// CHECK:STDOUT:   %struct_type.value.has_value.b29: type = struct_type {.value: %MaybeUnformed.a56, .has_value: bool} [concrete]
// CHECK:STDOUT:   %Optional.HasValue.specific_fn.d32: <specific function> = specific_function %Optional.HasValue.b32, @Optional.HasValue(%OptionalStorage.facet.c8d) [concrete]
// CHECK:STDOUT:   %Optional.Get.specific_fn.7da: <specific function> = specific_function %Optional.Get.998, @Optional.Get(%OptionalStorage.facet.c8d) [concrete]
// CHECK:STDOUT:   %AddAssignWith.type.1ad: type = facet_type <@AddAssignWith, @AddAssignWith(%i32)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.81c7d0.2: type = facet_type <@ImplicitAs, @ImplicitAs(%Int.d4bbe0.1)> [symbolic]
// CHECK:STDOUT:   %U.f16: %ImplicitAs.type.81c7d0.2 = symbolic_binding U, 1 [symbolic]
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.type.93140c.1: type = fn_type @Int.as.AddAssignWith.impl.Op.1, @Int.as.AddAssignWith.impl(%N.fe9, %U.f16) [symbolic]
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.75e08e.1: %Int.as.AddAssignWith.impl.Op.type.93140c.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.type.93140c.2: type = fn_type @Int.as.AddAssignWith.impl.Op.2, @Int.as.AddAssignWith.impl(%N.fe9, %U.f16) [symbolic]
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.75e08e.2: %Int.as.AddAssignWith.impl.Op.type.93140c.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %facet_type.0f0: type = facet_type <@IntFitsIn, @IntFitsIn(%i32) & @AnyInt> [concrete]
// CHECK:STDOUT:   %custom_witness.633: <witness> = custom_witness (), @IntFitsIn, @IntFitsIn(%i32) [concrete]
// CHECK:STDOUT:   %Int.as.AnyInt.impl.AsInt.type.0ee: type = fn_type @Int.as.AnyInt.impl.AsInt, @Int.as.AnyInt.impl(%N.fe9) [symbolic]
// CHECK:STDOUT:   %Int.as.AnyInt.impl.AsInt.497: %Int.as.AnyInt.impl.AsInt.type.0ee = struct_value () [symbolic]
// CHECK:STDOUT:   %AnyInt.impl_witness.83b: <witness> = impl_witness imports.%AnyInt.impl_witness_table, @Int.as.AnyInt.impl(%int_32) [concrete]
// CHECK:STDOUT:   %facet_value.709: %facet_type.0f0 = facet_value %i32, (%custom_witness.633, %AnyInt.impl_witness.83b) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.943: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.8f1, @From.as_type.as.ImplicitAs.impl(%int_32, %facet_value.709) [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.c18: %ImplicitAs.type.6b0 = facet_value %i32, (%ImplicitAs.impl_witness.943) [concrete]
// CHECK:STDOUT:   %AddAssignWith.impl_witness.95a: <witness> = impl_witness imports.%AddAssignWith.impl_witness_table, @Int.as.AddAssignWith.impl(%int_32, %ImplicitAs.facet.c18) [concrete]
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.type.47f4e6.1: type = fn_type @Int.as.AddAssignWith.impl.Op.2, @Int.as.AddAssignWith.impl(%int_32, %ImplicitAs.facet.c18) [concrete]
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.bff4f9.1: %Int.as.AddAssignWith.impl.Op.type.47f4e6.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.type.47f4e6.2: type = fn_type @Int.as.AddAssignWith.impl.Op.1, @Int.as.AddAssignWith.impl(%int_32, %ImplicitAs.facet.c18) [concrete]
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.bff4f9.2: %Int.as.AddAssignWith.impl.Op.type.47f4e6.2 = struct_value () [concrete]
// CHECK:STDOUT:   %AddAssignWith.facet.d53: %AddAssignWith.type.1ad = facet_value %i32, (%AddAssignWith.impl_witness.95a) [concrete]
// CHECK:STDOUT:   %AddAssignWith.WithSelf.Op.type.313: type = fn_type @AddAssignWith.WithSelf.Op, @AddAssignWith.WithSelf(%i32, %AddAssignWith.facet.d53) [concrete]
// CHECK:STDOUT:   %.316: type = fn_type_with_self_type %AddAssignWith.WithSelf.Op.type.313, %AddAssignWith.facet.d53 [concrete]
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.specific_fn.ec28e6.1: <specific function> = specific_function %Int.as.AddAssignWith.impl.Op.bff4f9.2, @Int.as.AddAssignWith.impl.Op.1(%int_32, %ImplicitAs.facet.c18) [concrete]
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.specific_fn.ec28e6.2: <specific function> = specific_function %Int.as.AddAssignWith.impl.Op.bff4f9.1, @Int.as.AddAssignWith.impl.Op.2(%int_32, %ImplicitAs.facet.c18) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.8: type = fn_type @SubobjectDestroy.WithSelf.Op.loc34_19.6 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.8: %SubobjectDestroy.WithSelf.Op.type.6870fd.8 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.8: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.8), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.3b7: %SubobjectDestroy.type = facet_value %Optional.fb0, (%custom_witness.b80ef6.8) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.d78: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.3b7) [concrete]
// CHECK:STDOUT:   %facet_value.5ba: %facet_type.8a3 = facet_value %Optional.fb0, (%custom_witness.b80ef6.8, %DestroyImpl.impl_witness.d78) [concrete]
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
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.9: type = fn_type @SubobjectDestroy.WithSelf.Op.loc41 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.9: %SubobjectDestroy.WithSelf.Op.type.6870fd.9 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.9: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.9), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.0c7: %SubobjectDestroy.type = facet_value %Iterator.3ee, (%custom_witness.b80ef6.9) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.728: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.0c7) [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.0aed5d.1: type = fn_type @Iterator.Op.1 [concrete]
// CHECK:STDOUT:   %Iterator.Op.a2a866.1: %Iterator.Op.type.0aed5d.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.43f: <witness> = custom_witness (%i32, %Iterator.Op.a2a866.1), @CppUnsafeDeref [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.0aed5d.2: type = fn_type @Iterator.Op.2 [concrete]
// CHECK:STDOUT:   %Iterator.Op.a2a866.2: %Iterator.Op.type.0aed5d.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.444: <witness> = custom_witness (%Iterator.Op.a2a866.2), @Inc [concrete]
// CHECK:STDOUT:   %Equal.type.ca8475.1: type = fn_type @Equal.1 [concrete]
// CHECK:STDOUT:   %Equal.285b15.1: %Equal.type.ca8475.1 = struct_value () [concrete]
// CHECK:STDOUT:   %NotEqual.type.53d9ee.1: type = fn_type @NotEqual.1 [concrete]
// CHECK:STDOUT:   %NotEqual.119263.1: %NotEqual.type.53d9ee.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.c91: <witness> = custom_witness (%Equal.285b15.1, %NotEqual.119263.1), @EqWith, @EqWith(%Iterator.3ee) [concrete]
// CHECK:STDOUT:   %facet_value.209: %CppRange_where.type.d682d6.1 = facet_value %MutableRange, (%custom_witness.ed3, %custom_witness.b80ef6.9, %DestroyImpl.impl_witness.728, %custom_witness.43f, %custom_witness.444, %custom_witness.c91, %custom_witness.b80ef6.1, %DestroyImpl.impl_witness.077, %Copy.impl_witness.8af, %custom_witness.b80ef6.9, %DestroyImpl.impl_witness.728) [concrete]
// CHECK:STDOUT:   %tuple.type.508: type = tuple_type (%Iterator.3ee, %Iterator.3ee) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.10: type = fn_type @SubobjectDestroy.WithSelf.Op.3 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.10: %SubobjectDestroy.WithSelf.Op.type.6870fd.10 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.10: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.10), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.669: %SubobjectDestroy.type = facet_value %tuple.type.508, (%custom_witness.b80ef6.10) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.91e: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.669) [concrete]
// CHECK:STDOUT:   %complete_type.c28: <witness> = complete_type_witness %tuple.type.508 [concrete]
// CHECK:STDOUT:   %facet_value.10d: %facet_type.8a3 = facet_value %tuple.type.508, (%custom_witness.b80ef6.10, %DestroyImpl.impl_witness.91e) [concrete]
// CHECK:STDOUT:   %Iterate.impl_witness.2ca: <witness> = impl_witness imports.%Iterate.impl_witness_table.121, @T.as_type.as.Iterate.impl(%facet_value.209) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.type.9f7: type = fn_type @T.as_type.as.Iterate.impl.NewCursor, @T.as_type.as.Iterate.impl(%facet_value.209) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.219: %T.as_type.as.Iterate.impl.NewCursor.type.9f7 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.type.80a: type = fn_type @T.as_type.as.Iterate.impl.Next, @T.as_type.as.Iterate.impl(%facet_value.209) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.a23: %T.as_type.as.Iterate.impl.Next.type.80a = struct_value () [concrete]
// CHECK:STDOUT:   %Iterate.facet.da9: %Iterate.type = facet_value %MutableRange, (%Iterate.impl_witness.2ca) [concrete]
// CHECK:STDOUT:   %facet_value.4e0: %Iterate_where.type.35a = facet_value %MutableRange, (%Iterate.impl_witness.2ca) [concrete]
// CHECK:STDOUT:   %r.param_patt.c79: %pattern_type.e63 = value_param_pattern [concrete]
// CHECK:STDOUT:   %r.patt.94f: %pattern_type.e63 = wrapper_binding_pattern r, %r.param_patt.c79 [concrete]
// CHECK:STDOUT:   %Iterator.260: type = class_type @Iterator.2 [concrete]
// CHECK:STDOUT:   %ConstRange.Begin.type: type = fn_type @ConstRange.Begin [concrete]
// CHECK:STDOUT:   %ConstRange.Begin: %ConstRange.Begin.type = struct_value () [concrete]
// CHECK:STDOUT:   %ConstRange.End.type: type = fn_type @ConstRange.End [concrete]
// CHECK:STDOUT:   %ConstRange.End: %ConstRange.End.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.23e: <witness> = custom_witness (%Iterator.260, %Iterator.260, %ConstRange.Begin, %ConstRange.End), @CppRangeForIterate [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.11: type = fn_type @SubobjectDestroy.WithSelf.Op.loc42 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.11: %SubobjectDestroy.WithSelf.Op.type.6870fd.11 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.11: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.11), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.4d4: %SubobjectDestroy.type = facet_value %Iterator.260, (%custom_witness.b80ef6.11) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.514: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.4d4) [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.0f34b8.1: type = fn_type @Iterator.Op.3 [concrete]
// CHECK:STDOUT:   %Iterator.Op.1c3e4a.1: %Iterator.Op.type.0f34b8.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.f5c: <witness> = custom_witness (%i32, %Iterator.Op.1c3e4a.1), @CppUnsafeDeref [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.0f34b8.2: type = fn_type @Iterator.Op.4 [concrete]
// CHECK:STDOUT:   %Iterator.Op.1c3e4a.2: %Iterator.Op.type.0f34b8.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.746: <witness> = custom_witness (%Iterator.Op.1c3e4a.2), @Inc [concrete]
// CHECK:STDOUT:   %Equal.type.ca8475.2: type = fn_type @Equal.2 [concrete]
// CHECK:STDOUT:   %Equal.285b15.2: %Equal.type.ca8475.2 = struct_value () [concrete]
// CHECK:STDOUT:   %NotEqual.type.53d9ee.2: type = fn_type @NotEqual.2 [concrete]
// CHECK:STDOUT:   %NotEqual.119263.2: %NotEqual.type.53d9ee.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.86f: <witness> = custom_witness (%Equal.285b15.2, %NotEqual.119263.2), @EqWith, @EqWith(%Iterator.260) [concrete]
// CHECK:STDOUT:   %facet_value.20b: %CppRange_where.type.d682d6.1 = facet_value %ConstRange, (%custom_witness.23e, %custom_witness.b80ef6.11, %DestroyImpl.impl_witness.514, %custom_witness.f5c, %custom_witness.746, %custom_witness.86f, %custom_witness.b80ef6.1, %DestroyImpl.impl_witness.077, %Copy.impl_witness.8af, %custom_witness.b80ef6.11, %DestroyImpl.impl_witness.514) [concrete]
// CHECK:STDOUT:   %tuple.type.69f: type = tuple_type (%Iterator.260, %Iterator.260) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.12: type = fn_type @SubobjectDestroy.WithSelf.Op.4 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.12: %SubobjectDestroy.WithSelf.Op.type.6870fd.12 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.12: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.12), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.100: %SubobjectDestroy.type = facet_value %tuple.type.69f, (%custom_witness.b80ef6.12) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.4ff: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.100) [concrete]
// CHECK:STDOUT:   %complete_type.8d5: <witness> = complete_type_witness %tuple.type.69f [concrete]
// CHECK:STDOUT:   %facet_value.f4d: %facet_type.8a3 = facet_value %tuple.type.69f, (%custom_witness.b80ef6.12, %DestroyImpl.impl_witness.4ff) [concrete]
// CHECK:STDOUT:   %Iterate.impl_witness.51a: <witness> = impl_witness imports.%Iterate.impl_witness_table.121, @T.as_type.as.Iterate.impl(%facet_value.20b) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.type.efe: type = fn_type @T.as_type.as.Iterate.impl.NewCursor, @T.as_type.as.Iterate.impl(%facet_value.20b) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.1b8: %T.as_type.as.Iterate.impl.NewCursor.type.efe = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.type.9b9: type = fn_type @T.as_type.as.Iterate.impl.Next, @T.as_type.as.Iterate.impl(%facet_value.20b) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.283: %T.as_type.as.Iterate.impl.Next.type.9b9 = struct_value () [concrete]
// CHECK:STDOUT:   %Iterate.facet.656: %Iterate.type = facet_value %ConstRange, (%Iterate.impl_witness.51a) [concrete]
// CHECK:STDOUT:   %facet_value.1df: %Iterate_where.type.35a = facet_value %ConstRange, (%Iterate.impl_witness.51a) [concrete]
// CHECK:STDOUT:   %r.param_patt.83e: %pattern_type.0ba = value_param_pattern [concrete]
// CHECK:STDOUT:   %r.patt.c21: %pattern_type.0ba = wrapper_binding_pattern r, %r.param_patt.83e [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.b65: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.da9) [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.88f: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.da9) [concrete]
// CHECK:STDOUT:   %.efe: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.b65, %Iterate.facet.da9 [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.specific_fn.d1a: <specific function> = specific_function %T.as_type.as.Iterate.impl.NewCursor.219, @T.as_type.as.Iterate.impl.NewCursor(%facet_value.209) [concrete]
// CHECK:STDOUT:   %.d9d2: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.88f, %Iterate.facet.da9 [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.specific_fn.e11: <specific function> = specific_function %T.as_type.as.Iterate.impl.Next.a23, @T.as_type.as.Iterate.impl.Next(%facet_value.209) [concrete]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.949: <specific function> = specific_function %SelfDestruct, @SelfDestruct(%facet_value.10d) [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.32a: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.656) [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.4e2: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.656) [concrete]
// CHECK:STDOUT:   %.4dd: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.32a, %Iterate.facet.656 [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.specific_fn.ece: <specific function> = specific_function %T.as_type.as.Iterate.impl.NewCursor.1b8, @T.as_type.as.Iterate.impl.NewCursor(%facet_value.20b) [concrete]
// CHECK:STDOUT:   %.4fc: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.4e2, %Iterate.facet.656 [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.specific_fn.d22: <specific function> = specific_function %T.as_type.as.Iterate.impl.Next.283, @T.as_type.as.Iterate.impl.Next(%facet_value.20b) [concrete]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.036: <specific function> = specific_function %SelfDestruct, @SelfDestruct(%facet_value.f4d) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.c82: @Optional.%Optional.HasValue.type (%Optional.HasValue.type.3ea) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.HasValue (constants.%Optional.HasValue.f2f)]
// CHECK:STDOUT:   %Core.import_ref.29e: @Optional.%Optional.Get.type (%Optional.Get.type.786) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.Get (constants.%Optional.Get.2d0)]
// CHECK:STDOUT:   %Core.import_ref.9a31: type = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%DefaultOptionalStorage (constants.%DefaultOptionalStorage.f15)]
// CHECK:STDOUT:   %Core.import_ref.819a: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.None.type (%T.as_type.as.OptionalStorage.impl.None.type.76a) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.None (constants.%T.as_type.as.OptionalStorage.impl.None.93b)]
// CHECK:STDOUT:   %Core.import_ref.14a: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Some.type (%T.as_type.as.OptionalStorage.impl.Some.type.4eb) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Some (constants.%T.as_type.as.OptionalStorage.impl.Some.44a)]
// CHECK:STDOUT:   %Core.import_ref.242: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Has.type (%T.as_type.as.OptionalStorage.impl.Has.type.371) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Has (constants.%T.as_type.as.OptionalStorage.impl.Has.8ce)]
// CHECK:STDOUT:   %Core.import_ref.a54: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Get.type (%T.as_type.as.OptionalStorage.impl.Get.type.3e7) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Get (constants.%T.as_type.as.OptionalStorage.impl.Get.f80)]
// CHECK:STDOUT:   %Core.import_ref.5bf = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], unloaded
// CHECK:STDOUT:   %OptionalStorage.impl_witness_table.968 = impl_witness_table (%Core.import_ref.9a31, %Core.import_ref.819a, %Core.import_ref.14a, %Core.import_ref.242, %Core.import_ref.a54, %Core.import_ref.5bf), @T.as_type.as.OptionalStorage.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.822: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.822), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.d6c: %facet_type.d61 = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.Iterate.impl.%facet_value.1 (constants.%facet_value.9da)]
// CHECK:STDOUT:   %Core.import_ref.2ac: %facet_type.8a3 = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.Iterate.impl.%facet_value.2 (constants.%facet_value.bf6)]
// CHECK:STDOUT:   %Core.import_ref.946: @T.as_type.as.Iterate.impl.%T.as_type.as.Iterate.impl.NewCursor.type (%T.as_type.as.Iterate.impl.NewCursor.type.8fb) = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.Iterate.impl.%T.as_type.as.Iterate.impl.NewCursor (constants.%T.as_type.as.Iterate.impl.NewCursor.8a1)]
// CHECK:STDOUT:   %Core.import_ref.cfb: @T.as_type.as.Iterate.impl.%T.as_type.as.Iterate.impl.Next.type (%T.as_type.as.Iterate.impl.Next.type.d4a) = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.Iterate.impl.%T.as_type.as.Iterate.impl.Next (constants.%T.as_type.as.Iterate.impl.Next.5a8)]
// CHECK:STDOUT:   %Iterate.impl_witness_table.121 = impl_witness_table (%Core.import_ref.d6c, %Core.import_ref.2ac, %Core.import_ref.946, %Core.import_ref.cfb), @T.as_type.as.Iterate.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.72f: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.a1d) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.3c1)]
// CHECK:STDOUT:   %Copy.impl_witness_table.f44 = impl_witness_table (%Core.import_ref.72f), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.39b: @From.as_type.as.ImplicitAs.impl.%From.as_type.as.ImplicitAs.impl.Convert.type (%From.as_type.as.ImplicitAs.impl.Convert.type.386) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @From.as_type.as.ImplicitAs.impl.%From.as_type.as.ImplicitAs.impl.Convert (constants.%From.as_type.as.ImplicitAs.impl.Convert.85a)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.8f1 = impl_witness_table (%Core.import_ref.39b), @From.as_type.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.b25: @Int.as.AddAssignWith.impl.%Int.as.AddAssignWith.impl.Op.type.2 (%Int.as.AddAssignWith.impl.Op.type.93140c.1) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.AddAssignWith.impl.%Int.as.AddAssignWith.impl.Op.2 (constants.%Int.as.AddAssignWith.impl.Op.75e08e.1)]
// CHECK:STDOUT:   %AddAssignWith.impl_witness_table = impl_witness_table (%Core.import_ref.b25), @Int.as.AddAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.da0: @Int.as.AddAssignWith.impl.%Int.as.AddAssignWith.impl.Op.type.1 (%Int.as.AddAssignWith.impl.Op.type.93140c.2) = import_ref Core//prelude/types/int, Op, loaded [symbolic = @Int.as.AddAssignWith.impl.%Int.as.AddAssignWith.impl.Op.1 (constants.%Int.as.AddAssignWith.impl.Op.75e08e.2)]
// CHECK:STDOUT:   %Core.import_ref.47b: Core.IntLiteral = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.AnyInt.impl.%N (constants.%N.fe9)]
// CHECK:STDOUT:   %Core.import_ref.323: @Int.as.AnyInt.impl.%Int.as.AnyInt.impl.AsInt.type (%Int.as.AnyInt.impl.AsInt.type.0ee) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.AnyInt.impl.%Int.as.AnyInt.impl.AsInt (constants.%Int.as.AnyInt.impl.AsInt.497)]
// CHECK:STDOUT:   %AnyInt.impl_witness_table = impl_witness_table (%Core.import_ref.47b, %Core.import_ref.323), @Int.as.AnyInt.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @TestIterate(%R.loc30_17.2: %Iterate_where.type.35a) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Iterate.lookup_impl_witness: <witness> = lookup_impl_witness %R.loc30_17.1, @Iterate [symbolic = %Iterate.lookup_impl_witness (constants.%Iterate.lookup_impl_witness.8f0)]
// CHECK:STDOUT:   %Iterate.facet.loc34_19.5: %Iterate.type = facet_value %R.as_type.loc30_61.1, (%Iterate.lookup_impl_witness) [symbolic = %Iterate.facet.loc34_19.5 (constants.%Iterate.facet.39d)]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.loc34_19.5) [symbolic = %Iterate.WithSelf.NewCursor.type (constants.%Iterate.WithSelf.NewCursor.type.b8b)]
// CHECK:STDOUT:   %.loc34_19.23: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type, %Iterate.facet.loc34_19.5 [symbolic = %.loc34_19.23 (constants.%.fd4)]
// CHECK:STDOUT:   %impl.elem2.loc34_19.2: @TestIterate.%.loc34_19.23 (%.fd4) = impl_witness_access %Iterate.lookup_impl_witness, element2 [symbolic = %impl.elem2.loc34_19.2 (constants.%impl.elem2.c03)]
// CHECK:STDOUT:   %specific_impl_fn.loc34_19.3: <specific function> = specific_impl_function %impl.elem2.loc34_19.2, @Iterate.WithSelf.NewCursor(%Iterate.facet.loc34_19.5) [symbolic = %specific_impl_fn.loc34_19.3 (constants.%specific_impl_fn.9fbe)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.loc34_19.5) [symbolic = %Iterate.WithSelf.Next.type (constants.%Iterate.WithSelf.Next.type.f8e)]
// CHECK:STDOUT:   %.loc34_19.24: type = fn_type_with_self_type %Iterate.WithSelf.Next.type, %Iterate.facet.loc34_19.5 [symbolic = %.loc34_19.24 (constants.%.765)]
// CHECK:STDOUT:   %impl.elem3.loc34_19.2: @TestIterate.%.loc34_19.24 (%.765) = impl_witness_access %Iterate.lookup_impl_witness, element3 [symbolic = %impl.elem3.loc34_19.2 (constants.%impl.elem3.297)]
// CHECK:STDOUT:   %specific_impl_fn.loc34_19.4: <specific function> = specific_impl_function %impl.elem3.loc34_19.2, @Iterate.WithSelf.Next(%Iterate.facet.loc34_19.5) [symbolic = %specific_impl_fn.loc34_19.4 (constants.%specific_impl_fn.e3a)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%r.param: @TestIterate.%R.as_type.loc30_61.1 (%R.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %i.patt: %pattern_type.d94 = value_binding_pattern i [concrete = constants.%i.patt]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %r.ref: @TestIterate.%R.as_type.loc30_61.1 (%R.as_type) = name_ref r, %r
// CHECK:STDOUT:     %impl.elem2.loc34_19.1: @TestIterate.%.loc34_19.23 (%.fd4) = impl_witness_access constants.%Iterate.lookup_impl_witness.8f0, element2 [symbolic = %impl.elem2.loc34_19.2 (constants.%impl.elem2.c03)]
// CHECK:STDOUT:     %bound_method.loc34_19.1: <bound method> = bound_method %r.ref, %impl.elem2.loc34_19.1
// CHECK:STDOUT:     %Iterate.facet.loc34_19.1: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.8f0) [symbolic = %Iterate.facet.loc34_19.5 (constants.%Iterate.facet.39d)]
// CHECK:STDOUT:     %.loc34_19.1: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc34_19.1 [symbolic = %Iterate.facet.loc34_19.5 (constants.%Iterate.facet.39d)]
// CHECK:STDOUT:     %Iterate.facet.loc34_19.2: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.8f0) [symbolic = %Iterate.facet.loc34_19.5 (constants.%Iterate.facet.39d)]
// CHECK:STDOUT:     %.loc34_19.2: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc34_19.2 [symbolic = %Iterate.facet.loc34_19.5 (constants.%Iterate.facet.39d)]
// CHECK:STDOUT:     %specific_impl_fn.loc34_19.1: <specific function> = specific_impl_function %impl.elem2.loc34_19.1, @Iterate.WithSelf.NewCursor(constants.%Iterate.facet.39d) [symbolic = %specific_impl_fn.loc34_19.3 (constants.%specific_impl_fn.9fbe)]
// CHECK:STDOUT:     %bound_method.loc34_19.2: <bound method> = bound_method %r.ref, %specific_impl_fn.loc34_19.1
// CHECK:STDOUT:     %var: ref @TestIterate.%as_type (%as_type.9f4) = var_storage invalid
// CHECK:STDOUT:     %Iterate.WithSelf.NewCursor.call: init @TestIterate.%as_type (%as_type.9f4) to %var = call %bound_method.loc34_19.2(%r.ref)
// CHECK:STDOUT:     assign %var, %Iterate.WithSelf.NewCursor.call
// CHECK:STDOUT:     br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.next:
// CHECK:STDOUT:     %impl.elem3.loc34_19.1: @TestIterate.%.loc34_19.24 (%.765) = impl_witness_access constants.%Iterate.lookup_impl_witness.8f0, element3 [symbolic = %impl.elem3.loc34_19.2 (constants.%impl.elem3.297)]
// CHECK:STDOUT:     %bound_method.loc34_19.3: <bound method> = bound_method %r.ref, %impl.elem3.loc34_19.1
// CHECK:STDOUT:     %Iterate.facet.loc34_19.3: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.8f0) [symbolic = %Iterate.facet.loc34_19.5 (constants.%Iterate.facet.39d)]
// CHECK:STDOUT:     %.loc34_19.3: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc34_19.3 [symbolic = %Iterate.facet.loc34_19.5 (constants.%Iterate.facet.39d)]
// CHECK:STDOUT:     %Iterate.facet.loc34_19.4: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.8f0) [symbolic = %Iterate.facet.loc34_19.5 (constants.%Iterate.facet.39d)]
// CHECK:STDOUT:     %.loc34_19.4: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc34_19.4 [symbolic = %Iterate.facet.loc34_19.5 (constants.%Iterate.facet.39d)]
// CHECK:STDOUT:     %.loc34_19.5: %facet_type.8a3 = converted constants.%as_type.9f4, constants.%impl.elem1.f1d [symbolic = %impl.elem1 (constants.%impl.elem1.f1d)]
// CHECK:STDOUT:     %.loc34_19.6: %facet_type.8a3 = converted constants.%as_type.9f4, constants.%impl.elem1.f1d [symbolic = %impl.elem1 (constants.%impl.elem1.f1d)]
// CHECK:STDOUT:     %specific_impl_fn.loc34_19.2: <specific function> = specific_impl_function %impl.elem3.loc34_19.1, @Iterate.WithSelf.Next(constants.%Iterate.facet.39d) [symbolic = %specific_impl_fn.loc34_19.4 (constants.%specific_impl_fn.e3a)]
// CHECK:STDOUT:     %bound_method.loc34_19.4: <bound method> = bound_method %r.ref, %specific_impl_fn.loc34_19.2
// CHECK:STDOUT:     %.loc34_19.7: ref %Optional.fb0 = temporary_storage
// CHECK:STDOUT:     %Iterate.WithSelf.Next.call: init %Optional.fb0 to %.loc34_19.7 = call %bound_method.loc34_19.4(%r.ref, %var)
// CHECK:STDOUT:     %.loc34_19.8: ref %Optional.fb0 = temporary %.loc34_19.7, %Iterate.WithSelf.Next.call
// CHECK:STDOUT:     %.loc34_19.9: %Optional.HasValue.type.be5 = specific_constant imports.%Core.import_ref.c82, @Optional(constants.%OptionalStorage.facet.c8d) [concrete = constants.%Optional.HasValue.b32]
// CHECK:STDOUT:     %HasValue.ref: %Optional.HasValue.type.be5 = name_ref HasValue, %.loc34_19.9 [concrete = constants.%Optional.HasValue.b32]
// CHECK:STDOUT:     %Optional.HasValue.bound: <bound method> = bound_method %.loc34_19.8, %HasValue.ref
// CHECK:STDOUT:     %Optional.HasValue.specific_fn: <specific function> = specific_function %HasValue.ref, @Optional.HasValue(constants.%OptionalStorage.facet.c8d) [concrete = constants.%Optional.HasValue.specific_fn.d32]
// CHECK:STDOUT:     %bound_method.loc34_19.5: <bound method> = bound_method %.loc34_19.8, %Optional.HasValue.specific_fn
// CHECK:STDOUT:     %.loc34_19.10: %Optional.fb0 = acquire_value %.loc34_19.8
// CHECK:STDOUT:     %Optional.HasValue.call: init bool = call %bound_method.loc34_19.5(%.loc34_19.10)
// CHECK:STDOUT:     %.loc34_19.11: bool = value_of_initializer %Optional.HasValue.call
// CHECK:STDOUT:     %.loc34_19.12: bool = converted %Optional.HasValue.call, %.loc34_19.11
// CHECK:STDOUT:     if %.loc34_19.12 br !for.body else br !for.done
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.body:
// CHECK:STDOUT:     %.loc34_19.13: %Optional.Get.type.186 = specific_constant imports.%Core.import_ref.29e, @Optional(constants.%OptionalStorage.facet.c8d) [concrete = constants.%Optional.Get.998]
// CHECK:STDOUT:     %Get.ref: %Optional.Get.type.186 = name_ref Get, %.loc34_19.13 [concrete = constants.%Optional.Get.998]
// CHECK:STDOUT:     %Optional.Get.bound: <bound method> = bound_method %.loc34_19.8, %Get.ref
// CHECK:STDOUT:     %Optional.Get.specific_fn: <specific function> = specific_function %Get.ref, @Optional.Get(constants.%OptionalStorage.facet.c8d) [concrete = constants.%Optional.Get.specific_fn.7da]
// CHECK:STDOUT:     %bound_method.loc34_19.6: <bound method> = bound_method %.loc34_19.8, %Optional.Get.specific_fn
// CHECK:STDOUT:     %.loc34_19.14: %Optional.fb0 = acquire_value %.loc34_19.8
// CHECK:STDOUT:     %Optional.Get.call: init %i32 = call %bound_method.loc34_19.6(%.loc34_19.14)
// CHECK:STDOUT:     %.loc34_19.15: %i32 = value_of_initializer %Optional.Get.call
// CHECK:STDOUT:     %.loc34_19.16: %i32 = converted %Optional.Get.call, %.loc34_19.15
// CHECK:STDOUT:     %i32.loc34: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %i: %i32 = wrapper_binding i, %.loc34_19.16
// CHECK:STDOUT:     %sum.ref: ref %i32 = name_ref sum, %sum
// CHECK:STDOUT:     %i.ref: %i32 = name_ref i, %i
// CHECK:STDOUT:     %impl.elem0.loc35: %.316 = impl_witness_access constants.%AddAssignWith.impl_witness.95a, element0 [concrete = constants.%Int.as.AddAssignWith.impl.Op.bff4f9.2]
// CHECK:STDOUT:     %bound_method.loc35_9.1: <bound method> = bound_method %sum.ref, %impl.elem0.loc35
// CHECK:STDOUT:     %specific_fn.loc35: <specific function> = specific_function %impl.elem0.loc35, @Int.as.AddAssignWith.impl.Op.1(constants.%int_32, constants.%ImplicitAs.facet.c18) [concrete = constants.%Int.as.AddAssignWith.impl.Op.specific_fn.ec28e6.1]
// CHECK:STDOUT:     %bound_method.loc35_9.2: <bound method> = bound_method %sum.ref, %specific_fn.loc35
// CHECK:STDOUT:     %.loc35: %Int.as.AddAssignWith.impl.Op.type.47f4e6.1 = specific_constant imports.%Core.Op.da0, @Int.as.AddAssignWith.impl(constants.%int_32, constants.%ImplicitAs.facet.c18) [concrete = constants.%Int.as.AddAssignWith.impl.Op.bff4f9.1]
// CHECK:STDOUT:     %Op.ref: %Int.as.AddAssignWith.impl.Op.type.47f4e6.1 = name_ref Op, %.loc35 [concrete = constants.%Int.as.AddAssignWith.impl.Op.bff4f9.1]
// CHECK:STDOUT:     %Int.as.AddAssignWith.impl.Op.bound: <bound method> = bound_method %sum.ref, %Op.ref
// CHECK:STDOUT:     %Int.as.AddAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Op.ref, @Int.as.AddAssignWith.impl.Op.2(constants.%int_32, constants.%ImplicitAs.facet.c18) [concrete = constants.%Int.as.AddAssignWith.impl.Op.specific_fn.ec28e6.2]
// CHECK:STDOUT:     %bound_method.loc35_9.3: <bound method> = bound_method %sum.ref, %Int.as.AddAssignWith.impl.Op.specific_fn
// CHECK:STDOUT:     %Int.as.AddAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc35_9.3(%sum.ref, %i.ref)
// CHECK:STDOUT:     %facet_value.loc34_19.1: %facet_type.8a3 = facet_value constants.%Optional.fb0, (constants.%custom_witness.b80ef6.8, constants.%DestroyImpl.impl_witness.d78) [concrete = constants.%facet_value.5ba]
// CHECK:STDOUT:     %.loc34_19.17: %facet_type.8a3 = converted constants.%Optional.fb0, %facet_value.loc34_19.1 [concrete = constants.%facet_value.5ba]
// CHECK:STDOUT:     %facet_value.loc34_19.2: %facet_type.8a3 = facet_value constants.%Optional.fb0, (constants.%custom_witness.b80ef6.8, constants.%DestroyImpl.impl_witness.d78) [concrete = constants.%facet_value.5ba]
// CHECK:STDOUT:     %.loc34_19.18: %facet_type.8a3 = converted constants.%Optional.fb0, %facet_value.loc34_19.2 [concrete = constants.%facet_value.5ba]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %SelfDestruct.call.loc34_19.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc34_19.8)
// CHECK:STDOUT:     br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.done:
// CHECK:STDOUT:     %facet_value.loc34_19.3: %facet_type.8a3 = facet_value constants.%Optional.fb0, (constants.%custom_witness.b80ef6.8, constants.%DestroyImpl.impl_witness.d78) [concrete = constants.%facet_value.5ba]
// CHECK:STDOUT:     %.loc34_19.19: %facet_type.8a3 = converted constants.%Optional.fb0, %facet_value.loc34_19.3 [concrete = constants.%facet_value.5ba]
// CHECK:STDOUT:     %facet_value.loc34_19.4: %facet_type.8a3 = facet_value constants.%Optional.fb0, (constants.%custom_witness.b80ef6.8, constants.%DestroyImpl.impl_witness.d78) [concrete = constants.%facet_value.5ba]
// CHECK:STDOUT:     %.loc34_19.20: %facet_type.8a3 = converted constants.%Optional.fb0, %facet_value.loc34_19.4 [concrete = constants.%facet_value.5ba]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %SelfDestruct.call.loc34_19.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc34_19.8)
// CHECK:STDOUT:     %.loc34_19.21: %facet_type.8a3 = converted constants.%as_type.9f4, constants.%impl.elem1.f1d [symbolic = %impl.elem1 (constants.%impl.elem1.f1d)]
// CHECK:STDOUT:     %.loc34_19.22: %facet_type.8a3 = converted constants.%as_type.9f4, constants.%impl.elem1.f1d [symbolic = %impl.elem1 (constants.%impl.elem1.f1d)]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %SelfDestruct.call.loc34_19.3: init %empty_tuple.type = call %SelfDestruct.specific_fn.3(%var)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc34_19.1(%self.param: ref %.b31) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc34_19.2(%self.param: ref %MaybeUnformed.a56) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc34_19.3(%self.param: ref bool) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc34_19.4(%self.param: ref %struct_type.value.has_value.b29) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc34_19.5(%self.param: ref %DefaultOptionalStorage.6ca) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc34_19.6(%self.param: ref %Optional.fb0) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestRangeFor(%m.param: ref %MutableRange, %c.param: %ConstRange) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %i.patt.loc49: %pattern_type.d94 = value_binding_pattern i [concrete = constants.%i.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %m.ref: ref %MutableRange = name_ref m, %m
// CHECK:STDOUT:   %impl.elem2.loc49: %.efe = impl_witness_access constants.%Iterate.impl_witness.2ca, element2 [concrete = constants.%T.as_type.as.Iterate.impl.NewCursor.219]
// CHECK:STDOUT:   %bound_method.loc49_19.1: <bound method> = bound_method %m.ref, %impl.elem2.loc49
// CHECK:STDOUT:   %facet_value.loc49_19.1: %CppRange_where.type.d682d6.1 = facet_value constants.%MutableRange, (constants.%custom_witness.ed3, constants.%custom_witness.b80ef6.9, constants.%DestroyImpl.impl_witness.728, constants.%custom_witness.43f, constants.%custom_witness.444, constants.%custom_witness.c91, constants.%custom_witness.b80ef6.1, constants.%DestroyImpl.impl_witness.077, constants.%Copy.impl_witness.8af, constants.%custom_witness.b80ef6.9, constants.%DestroyImpl.impl_witness.728) [concrete = constants.%facet_value.209]
// CHECK:STDOUT:   %.loc49_19.1: %CppRange_where.type.d682d6.1 = converted constants.%MutableRange, %facet_value.loc49_19.1 [concrete = constants.%facet_value.209]
// CHECK:STDOUT:   %facet_value.loc49_19.2: %CppRange_where.type.d682d6.1 = facet_value constants.%MutableRange, (constants.%custom_witness.ed3, constants.%custom_witness.b80ef6.9, constants.%DestroyImpl.impl_witness.728, constants.%custom_witness.43f, constants.%custom_witness.444, constants.%custom_witness.c91, constants.%custom_witness.b80ef6.1, constants.%DestroyImpl.impl_witness.077, constants.%Copy.impl_witness.8af, constants.%custom_witness.b80ef6.9, constants.%DestroyImpl.impl_witness.728) [concrete = constants.%facet_value.209]
// CHECK:STDOUT:   %.loc49_19.2: %CppRange_where.type.d682d6.1 = converted constants.%MutableRange, %facet_value.loc49_19.2 [concrete = constants.%facet_value.209]
// CHECK:STDOUT:   %specific_fn.loc49_19.1: <specific function> = specific_function %impl.elem2.loc49, @T.as_type.as.Iterate.impl.NewCursor(constants.%facet_value.209) [concrete = constants.%T.as_type.as.Iterate.impl.NewCursor.specific_fn.d1a]
// CHECK:STDOUT:   %bound_method.loc49_19.2: <bound method> = bound_method %m.ref, %specific_fn.loc49_19.1
// CHECK:STDOUT:   %var.loc49: ref %tuple.type.508 = var_storage invalid
// CHECK:STDOUT:   %.loc49_18.1: %MutableRange = acquire_value %m.ref
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.call.loc49: init %tuple.type.508 to %var.loc49 = call %bound_method.loc49_19.2(%.loc49_18.1)
// CHECK:STDOUT:   assign %var.loc49, %T.as_type.as.Iterate.impl.NewCursor.call.loc49
// CHECK:STDOUT:   br !for.next.loc49
// CHECK:STDOUT:
// CHECK:STDOUT: !for.next.loc49:
// CHECK:STDOUT:   %impl.elem3.loc49: %.d9d2 = impl_witness_access constants.%Iterate.impl_witness.2ca, element3 [concrete = constants.%T.as_type.as.Iterate.impl.Next.a23]
// CHECK:STDOUT:   %bound_method.loc49_19.3: <bound method> = bound_method %m.ref, %impl.elem3.loc49
// CHECK:STDOUT:   %facet_value.loc49_19.3: %CppRange_where.type.d682d6.1 = facet_value constants.%MutableRange, (constants.%custom_witness.ed3, constants.%custom_witness.b80ef6.9, constants.%DestroyImpl.impl_witness.728, constants.%custom_witness.43f, constants.%custom_witness.444, constants.%custom_witness.c91, constants.%custom_witness.b80ef6.1, constants.%DestroyImpl.impl_witness.077, constants.%Copy.impl_witness.8af, constants.%custom_witness.b80ef6.9, constants.%DestroyImpl.impl_witness.728) [concrete = constants.%facet_value.209]
// CHECK:STDOUT:   %.loc49_19.3: %CppRange_where.type.d682d6.1 = converted constants.%MutableRange, %facet_value.loc49_19.3 [concrete = constants.%facet_value.209]
// CHECK:STDOUT:   %facet_value.loc49_19.4: %CppRange_where.type.d682d6.1 = facet_value constants.%MutableRange, (constants.%custom_witness.ed3, constants.%custom_witness.b80ef6.9, constants.%DestroyImpl.impl_witness.728, constants.%custom_witness.43f, constants.%custom_witness.444, constants.%custom_witness.c91, constants.%custom_witness.b80ef6.1, constants.%DestroyImpl.impl_witness.077, constants.%Copy.impl_witness.8af, constants.%custom_witness.b80ef6.9, constants.%DestroyImpl.impl_witness.728) [concrete = constants.%facet_value.209]
// CHECK:STDOUT:   %.loc49_19.4: %CppRange_where.type.d682d6.1 = converted constants.%MutableRange, %facet_value.loc49_19.4 [concrete = constants.%facet_value.209]
// CHECK:STDOUT:   %specific_fn.loc49_19.2: <specific function> = specific_function %impl.elem3.loc49, @T.as_type.as.Iterate.impl.Next(constants.%facet_value.209) [concrete = constants.%T.as_type.as.Iterate.impl.Next.specific_fn.e11]
// CHECK:STDOUT:   %bound_method.loc49_19.4: <bound method> = bound_method %m.ref, %specific_fn.loc49_19.2
// CHECK:STDOUT:   %.loc49_19.5: ref %Optional.fb0 = temporary_storage
// CHECK:STDOUT:   %.loc49_18.2: %MutableRange = acquire_value %m.ref
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.call.loc49: init %Optional.fb0 to %.loc49_19.5 = call %bound_method.loc49_19.4(%.loc49_18.2, %var.loc49)
// CHECK:STDOUT:   %.loc49_19.6: ref %Optional.fb0 = temporary %.loc49_19.5, %T.as_type.as.Iterate.impl.Next.call.loc49
// CHECK:STDOUT:   %.loc49_19.7: %Optional.HasValue.type.be5 = specific_constant imports.%Core.import_ref.c82, @Optional(constants.%OptionalStorage.facet.c8d) [concrete = constants.%Optional.HasValue.b32]
// CHECK:STDOUT:   %HasValue.ref.loc49: %Optional.HasValue.type.be5 = name_ref HasValue, %.loc49_19.7 [concrete = constants.%Optional.HasValue.b32]
// CHECK:STDOUT:   %Optional.HasValue.bound.loc49: <bound method> = bound_method %.loc49_19.6, %HasValue.ref.loc49
// CHECK:STDOUT:   %Optional.HasValue.specific_fn.loc49: <specific function> = specific_function %HasValue.ref.loc49, @Optional.HasValue(constants.%OptionalStorage.facet.c8d) [concrete = constants.%Optional.HasValue.specific_fn.d32]
// CHECK:STDOUT:   %bound_method.loc49_19.5: <bound method> = bound_method %.loc49_19.6, %Optional.HasValue.specific_fn.loc49
// CHECK:STDOUT:   %.loc49_19.8: %Optional.fb0 = acquire_value %.loc49_19.6
// CHECK:STDOUT:   %Optional.HasValue.call.loc49: init bool = call %bound_method.loc49_19.5(%.loc49_19.8)
// CHECK:STDOUT:   %.loc49_19.9: bool = value_of_initializer %Optional.HasValue.call.loc49
// CHECK:STDOUT:   %.loc49_19.10: bool = converted %Optional.HasValue.call.loc49, %.loc49_19.9
// CHECK:STDOUT:   if %.loc49_19.10 br !for.body.loc49 else br !for.done.loc49
// CHECK:STDOUT:
// CHECK:STDOUT: !for.body.loc49:
// CHECK:STDOUT:   %.loc49_19.11: %Optional.Get.type.186 = specific_constant imports.%Core.import_ref.29e, @Optional(constants.%OptionalStorage.facet.c8d) [concrete = constants.%Optional.Get.998]
// CHECK:STDOUT:   %Get.ref.loc49: %Optional.Get.type.186 = name_ref Get, %.loc49_19.11 [concrete = constants.%Optional.Get.998]
// CHECK:STDOUT:   %Optional.Get.bound.loc49: <bound method> = bound_method %.loc49_19.6, %Get.ref.loc49
// CHECK:STDOUT:   %Optional.Get.specific_fn.loc49: <specific function> = specific_function %Get.ref.loc49, @Optional.Get(constants.%OptionalStorage.facet.c8d) [concrete = constants.%Optional.Get.specific_fn.7da]
// CHECK:STDOUT:   %bound_method.loc49_19.6: <bound method> = bound_method %.loc49_19.6, %Optional.Get.specific_fn.loc49
// CHECK:STDOUT:   %.loc49_19.12: %Optional.fb0 = acquire_value %.loc49_19.6
// CHECK:STDOUT:   %Optional.Get.call.loc49: init %i32 = call %bound_method.loc49_19.6(%.loc49_19.12)
// CHECK:STDOUT:   %.loc49_19.13: %i32 = value_of_initializer %Optional.Get.call.loc49
// CHECK:STDOUT:   %.loc49_19.14: %i32 = converted %Optional.Get.call.loc49, %.loc49_19.13
// CHECK:STDOUT:   %i32.loc49: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %i.loc49: %i32 = wrapper_binding i, %.loc49_19.14
// CHECK:STDOUT:   %sum.ref.loc50: ref %i32 = name_ref sum, %sum
// CHECK:STDOUT:   %i.ref.loc50: %i32 = name_ref i, %i.loc49
// CHECK:STDOUT:   %impl.elem0.loc50: %.316 = impl_witness_access constants.%AddAssignWith.impl_witness.95a, element0 [concrete = constants.%Int.as.AddAssignWith.impl.Op.bff4f9.2]
// CHECK:STDOUT:   %bound_method.loc50_9.1: <bound method> = bound_method %sum.ref.loc50, %impl.elem0.loc50
// CHECK:STDOUT:   %specific_fn.loc50: <specific function> = specific_function %impl.elem0.loc50, @Int.as.AddAssignWith.impl.Op.1(constants.%int_32, constants.%ImplicitAs.facet.c18) [concrete = constants.%Int.as.AddAssignWith.impl.Op.specific_fn.ec28e6.1]
// CHECK:STDOUT:   %bound_method.loc50_9.2: <bound method> = bound_method %sum.ref.loc50, %specific_fn.loc50
// CHECK:STDOUT:   %.loc50: %Int.as.AddAssignWith.impl.Op.type.47f4e6.1 = specific_constant imports.%Core.Op.da0, @Int.as.AddAssignWith.impl(constants.%int_32, constants.%ImplicitAs.facet.c18) [concrete = constants.%Int.as.AddAssignWith.impl.Op.bff4f9.1]
// CHECK:STDOUT:   %Op.ref.loc50: %Int.as.AddAssignWith.impl.Op.type.47f4e6.1 = name_ref Op, %.loc50 [concrete = constants.%Int.as.AddAssignWith.impl.Op.bff4f9.1]
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.bound.loc50: <bound method> = bound_method %sum.ref.loc50, %Op.ref.loc50
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.specific_fn.loc50: <specific function> = specific_function %Op.ref.loc50, @Int.as.AddAssignWith.impl.Op.2(constants.%int_32, constants.%ImplicitAs.facet.c18) [concrete = constants.%Int.as.AddAssignWith.impl.Op.specific_fn.ec28e6.2]
// CHECK:STDOUT:   %bound_method.loc50_9.3: <bound method> = bound_method %sum.ref.loc50, %Int.as.AddAssignWith.impl.Op.specific_fn.loc50
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.call.loc50: init %empty_tuple.type = call %bound_method.loc50_9.3(%sum.ref.loc50, %i.ref.loc50)
// CHECK:STDOUT:   %facet_value.loc49_19.5: %facet_type.8a3 = facet_value constants.%Optional.fb0, (constants.%custom_witness.b80ef6.8, constants.%DestroyImpl.impl_witness.d78) [concrete = constants.%facet_value.5ba]
// CHECK:STDOUT:   %.loc49_19.15: %facet_type.8a3 = converted constants.%Optional.fb0, %facet_value.loc49_19.5 [concrete = constants.%facet_value.5ba]
// CHECK:STDOUT:   %facet_value.loc49_19.6: %facet_type.8a3 = facet_value constants.%Optional.fb0, (constants.%custom_witness.b80ef6.8, constants.%DestroyImpl.impl_witness.d78) [concrete = constants.%facet_value.5ba]
// CHECK:STDOUT:   %.loc49_19.16: %facet_type.8a3 = converted constants.%Optional.fb0, %facet_value.loc49_19.6 [concrete = constants.%facet_value.5ba]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc49_19.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc49_19.6)
// CHECK:STDOUT:   br !for.next.loc49
// CHECK:STDOUT:
// CHECK:STDOUT: !for.done.loc49:
// CHECK:STDOUT:   %facet_value.loc49_19.7: %facet_type.8a3 = facet_value constants.%Optional.fb0, (constants.%custom_witness.b80ef6.8, constants.%DestroyImpl.impl_witness.d78) [concrete = constants.%facet_value.5ba]
// CHECK:STDOUT:   %.loc49_19.17: %facet_type.8a3 = converted constants.%Optional.fb0, %facet_value.loc49_19.7 [concrete = constants.%facet_value.5ba]
// CHECK:STDOUT:   %facet_value.loc49_19.8: %facet_type.8a3 = facet_value constants.%Optional.fb0, (constants.%custom_witness.b80ef6.8, constants.%DestroyImpl.impl_witness.d78) [concrete = constants.%facet_value.5ba]
// CHECK:STDOUT:   %.loc49_19.18: %facet_type.8a3 = converted constants.%Optional.fb0, %facet_value.loc49_19.8 [concrete = constants.%facet_value.5ba]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc49_19.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc49_19.6)
// CHECK:STDOUT:   %facet_value.loc49_19.9: %facet_type.8a3 = facet_value constants.%tuple.type.508, (constants.%custom_witness.b80ef6.10, constants.%DestroyImpl.impl_witness.91e) [concrete = constants.%facet_value.10d]
// CHECK:STDOUT:   %.loc49_19.19: %facet_type.8a3 = converted constants.%tuple.type.508, %facet_value.loc49_19.9 [concrete = constants.%facet_value.10d]
// CHECK:STDOUT:   %facet_value.loc49_19.10: %facet_type.8a3 = facet_value constants.%tuple.type.508, (constants.%custom_witness.b80ef6.10, constants.%DestroyImpl.impl_witness.91e) [concrete = constants.%facet_value.10d]
// CHECK:STDOUT:   %.loc49_19.20: %facet_type.8a3 = converted constants.%tuple.type.508, %facet_value.loc49_19.10 [concrete = constants.%facet_value.10d]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc49_19.3: init %empty_tuple.type = call %SelfDestruct.specific_fn.3(%var.loc49)
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %i.patt.loc55: %pattern_type.d94 = value_binding_pattern i [concrete = constants.%i.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c.ref: %ConstRange = name_ref c, %c
// CHECK:STDOUT:   %impl.elem2.loc55: %.4dd = impl_witness_access constants.%Iterate.impl_witness.51a, element2 [concrete = constants.%T.as_type.as.Iterate.impl.NewCursor.1b8]
// CHECK:STDOUT:   %bound_method.loc55_19.1: <bound method> = bound_method %c.ref, %impl.elem2.loc55
// CHECK:STDOUT:   %facet_value.loc55_19.1: %CppRange_where.type.d682d6.1 = facet_value constants.%ConstRange, (constants.%custom_witness.23e, constants.%custom_witness.b80ef6.11, constants.%DestroyImpl.impl_witness.514, constants.%custom_witness.f5c, constants.%custom_witness.746, constants.%custom_witness.86f, constants.%custom_witness.b80ef6.1, constants.%DestroyImpl.impl_witness.077, constants.%Copy.impl_witness.8af, constants.%custom_witness.b80ef6.11, constants.%DestroyImpl.impl_witness.514) [concrete = constants.%facet_value.20b]
// CHECK:STDOUT:   %.loc55_19.1: %CppRange_where.type.d682d6.1 = converted constants.%ConstRange, %facet_value.loc55_19.1 [concrete = constants.%facet_value.20b]
// CHECK:STDOUT:   %facet_value.loc55_19.2: %CppRange_where.type.d682d6.1 = facet_value constants.%ConstRange, (constants.%custom_witness.23e, constants.%custom_witness.b80ef6.11, constants.%DestroyImpl.impl_witness.514, constants.%custom_witness.f5c, constants.%custom_witness.746, constants.%custom_witness.86f, constants.%custom_witness.b80ef6.1, constants.%DestroyImpl.impl_witness.077, constants.%Copy.impl_witness.8af, constants.%custom_witness.b80ef6.11, constants.%DestroyImpl.impl_witness.514) [concrete = constants.%facet_value.20b]
// CHECK:STDOUT:   %.loc55_19.2: %CppRange_where.type.d682d6.1 = converted constants.%ConstRange, %facet_value.loc55_19.2 [concrete = constants.%facet_value.20b]
// CHECK:STDOUT:   %specific_fn.loc55_19.1: <specific function> = specific_function %impl.elem2.loc55, @T.as_type.as.Iterate.impl.NewCursor(constants.%facet_value.20b) [concrete = constants.%T.as_type.as.Iterate.impl.NewCursor.specific_fn.ece]
// CHECK:STDOUT:   %bound_method.loc55_19.2: <bound method> = bound_method %c.ref, %specific_fn.loc55_19.1
// CHECK:STDOUT:   %var.loc55: ref %tuple.type.69f = var_storage invalid
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.call.loc55: init %tuple.type.69f to %var.loc55 = call %bound_method.loc55_19.2(%c.ref)
// CHECK:STDOUT:   assign %var.loc55, %T.as_type.as.Iterate.impl.NewCursor.call.loc55
// CHECK:STDOUT:   br !for.next.loc55
// CHECK:STDOUT:
// CHECK:STDOUT: !for.next.loc55:
// CHECK:STDOUT:   %impl.elem3.loc55: %.4fc = impl_witness_access constants.%Iterate.impl_witness.51a, element3 [concrete = constants.%T.as_type.as.Iterate.impl.Next.283]
// CHECK:STDOUT:   %bound_method.loc55_19.3: <bound method> = bound_method %c.ref, %impl.elem3.loc55
// CHECK:STDOUT:   %facet_value.loc55_19.3: %CppRange_where.type.d682d6.1 = facet_value constants.%ConstRange, (constants.%custom_witness.23e, constants.%custom_witness.b80ef6.11, constants.%DestroyImpl.impl_witness.514, constants.%custom_witness.f5c, constants.%custom_witness.746, constants.%custom_witness.86f, constants.%custom_witness.b80ef6.1, constants.%DestroyImpl.impl_witness.077, constants.%Copy.impl_witness.8af, constants.%custom_witness.b80ef6.11, constants.%DestroyImpl.impl_witness.514) [concrete = constants.%facet_value.20b]
// CHECK:STDOUT:   %.loc55_19.3: %CppRange_where.type.d682d6.1 = converted constants.%ConstRange, %facet_value.loc55_19.3 [concrete = constants.%facet_value.20b]
// CHECK:STDOUT:   %facet_value.loc55_19.4: %CppRange_where.type.d682d6.1 = facet_value constants.%ConstRange, (constants.%custom_witness.23e, constants.%custom_witness.b80ef6.11, constants.%DestroyImpl.impl_witness.514, constants.%custom_witness.f5c, constants.%custom_witness.746, constants.%custom_witness.86f, constants.%custom_witness.b80ef6.1, constants.%DestroyImpl.impl_witness.077, constants.%Copy.impl_witness.8af, constants.%custom_witness.b80ef6.11, constants.%DestroyImpl.impl_witness.514) [concrete = constants.%facet_value.20b]
// CHECK:STDOUT:   %.loc55_19.4: %CppRange_where.type.d682d6.1 = converted constants.%ConstRange, %facet_value.loc55_19.4 [concrete = constants.%facet_value.20b]
// CHECK:STDOUT:   %specific_fn.loc55_19.2: <specific function> = specific_function %impl.elem3.loc55, @T.as_type.as.Iterate.impl.Next(constants.%facet_value.20b) [concrete = constants.%T.as_type.as.Iterate.impl.Next.specific_fn.d22]
// CHECK:STDOUT:   %bound_method.loc55_19.4: <bound method> = bound_method %c.ref, %specific_fn.loc55_19.2
// CHECK:STDOUT:   %.loc55_19.5: ref %Optional.fb0 = temporary_storage
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.call.loc55: init %Optional.fb0 to %.loc55_19.5 = call %bound_method.loc55_19.4(%c.ref, %var.loc55)
// CHECK:STDOUT:   %.loc55_19.6: ref %Optional.fb0 = temporary %.loc55_19.5, %T.as_type.as.Iterate.impl.Next.call.loc55
// CHECK:STDOUT:   %.loc55_19.7: %Optional.HasValue.type.be5 = specific_constant imports.%Core.import_ref.c82, @Optional(constants.%OptionalStorage.facet.c8d) [concrete = constants.%Optional.HasValue.b32]
// CHECK:STDOUT:   %HasValue.ref.loc55: %Optional.HasValue.type.be5 = name_ref HasValue, %.loc55_19.7 [concrete = constants.%Optional.HasValue.b32]
// CHECK:STDOUT:   %Optional.HasValue.bound.loc55: <bound method> = bound_method %.loc55_19.6, %HasValue.ref.loc55
// CHECK:STDOUT:   %Optional.HasValue.specific_fn.loc55: <specific function> = specific_function %HasValue.ref.loc55, @Optional.HasValue(constants.%OptionalStorage.facet.c8d) [concrete = constants.%Optional.HasValue.specific_fn.d32]
// CHECK:STDOUT:   %bound_method.loc55_19.5: <bound method> = bound_method %.loc55_19.6, %Optional.HasValue.specific_fn.loc55
// CHECK:STDOUT:   %.loc55_19.8: %Optional.fb0 = acquire_value %.loc55_19.6
// CHECK:STDOUT:   %Optional.HasValue.call.loc55: init bool = call %bound_method.loc55_19.5(%.loc55_19.8)
// CHECK:STDOUT:   %.loc55_19.9: bool = value_of_initializer %Optional.HasValue.call.loc55
// CHECK:STDOUT:   %.loc55_19.10: bool = converted %Optional.HasValue.call.loc55, %.loc55_19.9
// CHECK:STDOUT:   if %.loc55_19.10 br !for.body.loc55 else br !for.done.loc55
// CHECK:STDOUT:
// CHECK:STDOUT: !for.body.loc55:
// CHECK:STDOUT:   %.loc55_19.11: %Optional.Get.type.186 = specific_constant imports.%Core.import_ref.29e, @Optional(constants.%OptionalStorage.facet.c8d) [concrete = constants.%Optional.Get.998]
// CHECK:STDOUT:   %Get.ref.loc55: %Optional.Get.type.186 = name_ref Get, %.loc55_19.11 [concrete = constants.%Optional.Get.998]
// CHECK:STDOUT:   %Optional.Get.bound.loc55: <bound method> = bound_method %.loc55_19.6, %Get.ref.loc55
// CHECK:STDOUT:   %Optional.Get.specific_fn.loc55: <specific function> = specific_function %Get.ref.loc55, @Optional.Get(constants.%OptionalStorage.facet.c8d) [concrete = constants.%Optional.Get.specific_fn.7da]
// CHECK:STDOUT:   %bound_method.loc55_19.6: <bound method> = bound_method %.loc55_19.6, %Optional.Get.specific_fn.loc55
// CHECK:STDOUT:   %.loc55_19.12: %Optional.fb0 = acquire_value %.loc55_19.6
// CHECK:STDOUT:   %Optional.Get.call.loc55: init %i32 = call %bound_method.loc55_19.6(%.loc55_19.12)
// CHECK:STDOUT:   %.loc55_19.13: %i32 = value_of_initializer %Optional.Get.call.loc55
// CHECK:STDOUT:   %.loc55_19.14: %i32 = converted %Optional.Get.call.loc55, %.loc55_19.13
// CHECK:STDOUT:   %i32.loc55: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %i.loc55: %i32 = wrapper_binding i, %.loc55_19.14
// CHECK:STDOUT:   %sum.ref.loc56: ref %i32 = name_ref sum, %sum
// CHECK:STDOUT:   %i.ref.loc56: %i32 = name_ref i, %i.loc55
// CHECK:STDOUT:   %impl.elem0.loc56: %.316 = impl_witness_access constants.%AddAssignWith.impl_witness.95a, element0 [concrete = constants.%Int.as.AddAssignWith.impl.Op.bff4f9.2]
// CHECK:STDOUT:   %bound_method.loc56_9.1: <bound method> = bound_method %sum.ref.loc56, %impl.elem0.loc56
// CHECK:STDOUT:   %specific_fn.loc56: <specific function> = specific_function %impl.elem0.loc56, @Int.as.AddAssignWith.impl.Op.1(constants.%int_32, constants.%ImplicitAs.facet.c18) [concrete = constants.%Int.as.AddAssignWith.impl.Op.specific_fn.ec28e6.1]
// CHECK:STDOUT:   %bound_method.loc56_9.2: <bound method> = bound_method %sum.ref.loc56, %specific_fn.loc56
// CHECK:STDOUT:   %.loc56: %Int.as.AddAssignWith.impl.Op.type.47f4e6.1 = specific_constant imports.%Core.Op.da0, @Int.as.AddAssignWith.impl(constants.%int_32, constants.%ImplicitAs.facet.c18) [concrete = constants.%Int.as.AddAssignWith.impl.Op.bff4f9.1]
// CHECK:STDOUT:   %Op.ref.loc56: %Int.as.AddAssignWith.impl.Op.type.47f4e6.1 = name_ref Op, %.loc56 [concrete = constants.%Int.as.AddAssignWith.impl.Op.bff4f9.1]
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.bound.loc56: <bound method> = bound_method %sum.ref.loc56, %Op.ref.loc56
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.specific_fn.loc56: <specific function> = specific_function %Op.ref.loc56, @Int.as.AddAssignWith.impl.Op.2(constants.%int_32, constants.%ImplicitAs.facet.c18) [concrete = constants.%Int.as.AddAssignWith.impl.Op.specific_fn.ec28e6.2]
// CHECK:STDOUT:   %bound_method.loc56_9.3: <bound method> = bound_method %sum.ref.loc56, %Int.as.AddAssignWith.impl.Op.specific_fn.loc56
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.call.loc56: init %empty_tuple.type = call %bound_method.loc56_9.3(%sum.ref.loc56, %i.ref.loc56)
// CHECK:STDOUT:   %facet_value.loc55_19.5: %facet_type.8a3 = facet_value constants.%Optional.fb0, (constants.%custom_witness.b80ef6.8, constants.%DestroyImpl.impl_witness.d78) [concrete = constants.%facet_value.5ba]
// CHECK:STDOUT:   %.loc55_19.15: %facet_type.8a3 = converted constants.%Optional.fb0, %facet_value.loc55_19.5 [concrete = constants.%facet_value.5ba]
// CHECK:STDOUT:   %facet_value.loc55_19.6: %facet_type.8a3 = facet_value constants.%Optional.fb0, (constants.%custom_witness.b80ef6.8, constants.%DestroyImpl.impl_witness.d78) [concrete = constants.%facet_value.5ba]
// CHECK:STDOUT:   %.loc55_19.16: %facet_type.8a3 = converted constants.%Optional.fb0, %facet_value.loc55_19.6 [concrete = constants.%facet_value.5ba]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc55_19.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.4(%.loc55_19.6)
// CHECK:STDOUT:   br !for.next.loc55
// CHECK:STDOUT:
// CHECK:STDOUT: !for.done.loc55:
// CHECK:STDOUT:   %facet_value.loc55_19.7: %facet_type.8a3 = facet_value constants.%Optional.fb0, (constants.%custom_witness.b80ef6.8, constants.%DestroyImpl.impl_witness.d78) [concrete = constants.%facet_value.5ba]
// CHECK:STDOUT:   %.loc55_19.17: %facet_type.8a3 = converted constants.%Optional.fb0, %facet_value.loc55_19.7 [concrete = constants.%facet_value.5ba]
// CHECK:STDOUT:   %facet_value.loc55_19.8: %facet_type.8a3 = facet_value constants.%Optional.fb0, (constants.%custom_witness.b80ef6.8, constants.%DestroyImpl.impl_witness.d78) [concrete = constants.%facet_value.5ba]
// CHECK:STDOUT:   %.loc55_19.18: %facet_type.8a3 = converted constants.%Optional.fb0, %facet_value.loc55_19.8 [concrete = constants.%facet_value.5ba]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc55_19.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.5(%.loc55_19.6)
// CHECK:STDOUT:   %facet_value.loc55_19.9: %facet_type.8a3 = facet_value constants.%tuple.type.69f, (constants.%custom_witness.b80ef6.12, constants.%DestroyImpl.impl_witness.4ff) [concrete = constants.%facet_value.f4d]
// CHECK:STDOUT:   %.loc55_19.19: %facet_type.8a3 = converted constants.%tuple.type.69f, %facet_value.loc55_19.9 [concrete = constants.%facet_value.f4d]
// CHECK:STDOUT:   %facet_value.loc55_19.10: %facet_type.8a3 = facet_value constants.%tuple.type.69f, (constants.%custom_witness.b80ef6.12, constants.%DestroyImpl.impl_witness.4ff) [concrete = constants.%facet_value.f4d]
// CHECK:STDOUT:   %.loc55_19.20: %facet_type.8a3 = converted constants.%tuple.type.69f, %facet_value.loc55_19.10 [concrete = constants.%facet_value.f4d]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc55_19.3: init %empty_tuple.type = call %SelfDestruct.specific_fn.6(%var.loc55)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestIterate(constants.%R) {
// CHECK:STDOUT:   %R.patt.loc30_17.2 => constants.%R.patt
// CHECK:STDOUT:   %R.loc30_17.1 => constants.%R
// CHECK:STDOUT:   %R.as_type.loc30_61.1 => constants.%R.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.007
// CHECK:STDOUT:   %r.param_patt.loc30_59.2 => constants.%r.param_patt.553
// CHECK:STDOUT:   %r.patt.loc30_59.2 => constants.%r.patt.b42
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestIterate(constants.%facet_value.4e0) {
// CHECK:STDOUT:   %R.patt.loc30_17.2 => constants.%R.patt
// CHECK:STDOUT:   %R.loc30_17.1 => constants.%facet_value.4e0
// CHECK:STDOUT:   %R.as_type.loc30_61.1 => constants.%MutableRange
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.e63
// CHECK:STDOUT:   %r.param_patt.loc30_59.2 => constants.%r.param_patt.c79
// CHECK:STDOUT:   %r.patt.loc30_59.2 => constants.%r.patt.94f
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc30 => constants.%complete_type.357
// CHECK:STDOUT:   %Iterate.lookup_impl_witness => constants.%Iterate.impl_witness.2ca
// CHECK:STDOUT:   %Iterate.facet.loc34_19.5 => constants.%Iterate.facet.da9
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type => constants.%Iterate.WithSelf.NewCursor.type.b65
// CHECK:STDOUT:   %.loc34_19.23 => constants.%.efe
// CHECK:STDOUT:   %impl.elem2.loc34_19.2 => constants.%T.as_type.as.Iterate.impl.NewCursor.219
// CHECK:STDOUT:   %specific_impl_fn.loc34_19.3 => constants.%T.as_type.as.Iterate.impl.NewCursor.specific_fn.d1a
// CHECK:STDOUT:   %impl.elem1 => constants.%facet_value.10d
// CHECK:STDOUT:   %as_type => constants.%tuple.type.508
// CHECK:STDOUT:   %require_complete.1 => constants.%complete_type.c28
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type => constants.%Iterate.WithSelf.Next.type.88f
// CHECK:STDOUT:   %.loc34_19.24 => constants.%.d9d2
// CHECK:STDOUT:   %impl.elem3.loc34_19.2 => constants.%T.as_type.as.Iterate.impl.Next.a23
// CHECK:STDOUT:   %specific_impl_fn.loc34_19.4 => constants.%T.as_type.as.Iterate.impl.Next.specific_fn.e11
// CHECK:STDOUT:   %SelfDestruct.specific_fn.5 => constants.%SelfDestruct.specific_fn.949
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestIterate(constants.%facet_value.1df) {
// CHECK:STDOUT:   %R.patt.loc30_17.2 => constants.%R.patt
// CHECK:STDOUT:   %R.loc30_17.1 => constants.%facet_value.1df
// CHECK:STDOUT:   %R.as_type.loc30_61.1 => constants.%ConstRange
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.0ba
// CHECK:STDOUT:   %r.param_patt.loc30_59.2 => constants.%r.param_patt.83e
// CHECK:STDOUT:   %r.patt.loc30_59.2 => constants.%r.patt.c21
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc30 => constants.%complete_type.357
// CHECK:STDOUT:   %Iterate.lookup_impl_witness => constants.%Iterate.impl_witness.51a
// CHECK:STDOUT:   %Iterate.facet.loc34_19.5 => constants.%Iterate.facet.656
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type => constants.%Iterate.WithSelf.NewCursor.type.32a
// CHECK:STDOUT:   %.loc34_19.23 => constants.%.4dd
// CHECK:STDOUT:   %impl.elem2.loc34_19.2 => constants.%T.as_type.as.Iterate.impl.NewCursor.1b8
// CHECK:STDOUT:   %specific_impl_fn.loc34_19.3 => constants.%T.as_type.as.Iterate.impl.NewCursor.specific_fn.ece
// CHECK:STDOUT:   %impl.elem1 => constants.%facet_value.f4d
// CHECK:STDOUT:   %as_type => constants.%tuple.type.69f
// CHECK:STDOUT:   %require_complete.1 => constants.%complete_type.8d5
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type => constants.%Iterate.WithSelf.Next.type.4e2
// CHECK:STDOUT:   %.loc34_19.24 => constants.%.4fc
// CHECK:STDOUT:   %impl.elem3.loc34_19.2 => constants.%T.as_type.as.Iterate.impl.Next.283
// CHECK:STDOUT:   %specific_impl_fn.loc34_19.4 => constants.%T.as_type.as.Iterate.impl.Next.specific_fn.d22
// CHECK:STDOUT:   %SelfDestruct.specific_fn.5 => constants.%SelfDestruct.specific_fn.036
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
// CHECK:STDOUT:   %Optional.Get.type.786: type = fn_type @Optional.Get, @Optional(%T.220) [symbolic]
// CHECK:STDOUT:   %Optional.Get.2d0: %Optional.Get.type.786 = struct_value () [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.type.3ea: type = fn_type @Optional.HasValue, @Optional(%T.220) [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.f2f: %Optional.HasValue.type.3ea = struct_value () [symbolic]
// CHECK:STDOUT:   %facet_type.8a3: type = facet_type <@SubobjectDestroy & @DestroyImpl> [concrete]
// CHECK:STDOUT:   %T.9ab: %facet_type.d61 = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.f15: type = class_type @DefaultOptionalStorage, @DefaultOptionalStorage(%T.9ab) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Get.type.3e7: type = fn_type @T.as_type.as.OptionalStorage.impl.Get, @T.as_type.as.OptionalStorage.impl(%T.9ab) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Get.f80: %T.as_type.as.OptionalStorage.impl.Get.type.3e7 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Has.type.371: type = fn_type @T.as_type.as.OptionalStorage.impl.Has, @T.as_type.as.OptionalStorage.impl(%T.9ab) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Has.8ce: %T.as_type.as.OptionalStorage.impl.Has.type.371 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Some.type.4eb: type = fn_type @T.as_type.as.OptionalStorage.impl.Some, @T.as_type.as.OptionalStorage.impl(%T.9ab) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Some.44a: %T.as_type.as.OptionalStorage.impl.Some.type.4eb = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.None.type.76a: type = fn_type @T.as_type.as.OptionalStorage.impl.None, @T.as_type.as.OptionalStorage.impl(%T.9ab) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.None.93b: %T.as_type.as.OptionalStorage.impl.None.type.76a = struct_value () [symbolic]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
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
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.2: type = fn_type @SubobjectDestroy.WithSelf.Op.loc40_56 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.2: %SubobjectDestroy.WithSelf.Op.type.6870fd.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.2: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.2), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.eb5: %SubobjectDestroy.type = facet_value %DiffValueType, (%custom_witness.b80ef6.2) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.f43: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.eb5) [concrete]
// CHECK:STDOUT:   %const.879: type = const_type %DiffValueType [concrete]
// CHECK:STDOUT:   %ptr.816: type = ptr_type %const.879 [concrete]
// CHECK:STDOUT:   %ptr.cd0: type = ptr_type %DiffValueType [concrete]
// CHECK:STDOUT:   %DiffValueType.Op.type: type = fn_type @DiffValueType.Op [concrete]
// CHECK:STDOUT:   %DiffValueType.Op: %DiffValueType.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.d0da43.1: <witness> = custom_witness (%DiffValueType.Op), @Copy [concrete]
// CHECK:STDOUT:   %facet_value.a46: %facet_type.d61 = facet_value %DiffValueType, (%custom_witness.b80ef6.2, %DestroyImpl.impl_witness.f43, %custom_witness.d0da43.1) [concrete]
// CHECK:STDOUT:   %Iterate.lookup_impl_witness.1d8: <witness> = lookup_impl_witness %.Self.500, @Iterate [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.8a4: %facet_type.d61 = impl_witness_access %Iterate.lookup_impl_witness.1d8, element0 [symbolic_self]
// CHECK:STDOUT:   %Iterate_where.type.b70: type = facet_type <@Iterate where %impl.elem0.8a4 = %facet_value.a46> [concrete]
// CHECK:STDOUT:   %pattern_type.47d: type = pattern_type %Iterate_where.type.b70 [concrete]
// CHECK:STDOUT:   %R.patt: %pattern_type.47d = symbolic_binding_pattern R, 0 [symbolic]
// CHECK:STDOUT:   %R: %Iterate_where.type.b70 = symbolic_binding R, 0 [symbolic]
// CHECK:STDOUT:   %R.as_type: type = facet_access_type %R [symbolic]
// CHECK:STDOUT:   %pattern_type.e03: type = pattern_type %R.as_type [symbolic]
// CHECK:STDOUT:   %r.patt.ac0: %pattern_type.e03 = ref_binding_pattern r [symbolic]
// CHECK:STDOUT:   %r.param_patt.5e7: %pattern_type.e03 = var_param_pattern %r.patt.ac0 [symbolic]
// CHECK:STDOUT:   %i.patt: %pattern_type.149 = value_binding_pattern i [concrete]
// CHECK:STDOUT:   %Iterate.lookup_impl_witness.0b4: <witness> = lookup_impl_witness %R, @Iterate [symbolic]
// CHECK:STDOUT:   %Iterate.facet.162: %Iterate.type = facet_value %R.as_type, (%Iterate.lookup_impl_witness.0b4) [symbolic]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.073: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.162) [symbolic]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.5ba: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.162) [symbolic]
// CHECK:STDOUT:   %.701: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.073, %Iterate.facet.162 [symbolic]
// CHECK:STDOUT:   %impl.elem2.cbd: %.701 = impl_witness_access %Iterate.lookup_impl_witness.0b4, element2 [symbolic]
// CHECK:STDOUT:   %impl.elem1.7f1: %facet_type.8a3 = impl_witness_access %Iterate.lookup_impl_witness.0b4, element1 [symbolic]
// CHECK:STDOUT:   %as_type.903: type = facet_access_type %impl.elem1.7f1 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.34b: <specific function> = specific_impl_function %impl.elem2.cbd, @Iterate.WithSelf.NewCursor(%Iterate.facet.162) [symbolic]
// CHECK:STDOUT:   %.db2: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.5ba, %Iterate.facet.162 [symbolic]
// CHECK:STDOUT:   %impl.elem3.540: %.db2 = impl_witness_access %Iterate.lookup_impl_witness.0b4, element3 [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.a0f: type = class_type @DefaultOptionalStorage, @DefaultOptionalStorage(%facet_value.a46) [concrete]
// CHECK:STDOUT:   %OptionalStorage.impl_witness.d5f: <witness> = impl_witness imports.%OptionalStorage.impl_witness_table.968, @T.as_type.as.OptionalStorage.impl(%facet_value.a46) [concrete]
// CHECK:STDOUT:   %OptionalStorage.facet.591: %OptionalStorage.type = facet_value %DiffValueType, (%OptionalStorage.impl_witness.d5f) [concrete]
// CHECK:STDOUT:   %Optional.8c2: type = class_type @Optional, @Optional(%OptionalStorage.facet.591) [concrete]
// CHECK:STDOUT:   %specific_impl_fn.1b6: <specific function> = specific_impl_function %impl.elem3.540, @Iterate.WithSelf.Next(%Iterate.facet.162) [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.type.df4: type = fn_type @Optional.HasValue, @Optional(%OptionalStorage.facet.591) [concrete]
// CHECK:STDOUT:   %Optional.HasValue.7c9: %Optional.HasValue.type.df4 = struct_value () [concrete]
// CHECK:STDOUT:   %Optional.Get.type.18b: type = fn_type @Optional.Get, @Optional(%OptionalStorage.facet.591) [concrete]
// CHECK:STDOUT:   %Optional.Get.57b: %Optional.Get.type.18b = struct_value () [concrete]
// CHECK:STDOUT:   %facet_value.0ea: %facet_type.8a3 = facet_value %DiffValueType, (%custom_witness.b80ef6.2, %DestroyImpl.impl_witness.f43) [concrete]
// CHECK:STDOUT:   %MaybeUnformed.31b: type = class_type @MaybeUnformed, @MaybeUnformed(%facet_value.0ea) [concrete]
// CHECK:STDOUT:   %.a18: type = maybe_unformed_type %DiffValueType [concrete]
// CHECK:STDOUT:   %struct_type.value.has_value.08b: type = struct_type {.value: %MaybeUnformed.31b, .has_value: bool} [concrete]
// CHECK:STDOUT:   %Optional.HasValue.specific_fn.7b9: <specific function> = specific_function %Optional.HasValue.7c9, @Optional.HasValue(%OptionalStorage.facet.591) [concrete]
// CHECK:STDOUT:   %Optional.Get.specific_fn.c38: <specific function> = specific_function %Optional.Get.57b, @Optional.Get(%OptionalStorage.facet.591) [concrete]
// CHECK:STDOUT:   %operator_PlusEqual__carbon_thunk.type: type = fn_type @operator_PlusEqual__carbon_thunk [concrete]
// CHECK:STDOUT:   %operator_PlusEqual__carbon_thunk: %operator_PlusEqual__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.8: type = fn_type @SubobjectDestroy.WithSelf.Op.loc44_33.6 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.8: %SubobjectDestroy.WithSelf.Op.type.6870fd.8 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.8: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.8), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.ce6: %SubobjectDestroy.type = facet_value %Optional.8c2, (%custom_witness.b80ef6.8) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.355: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.ce6) [concrete]
// CHECK:STDOUT:   %facet_value.fb7: %facet_type.8a3 = facet_value %Optional.8c2, (%custom_witness.b80ef6.8, %DestroyImpl.impl_witness.355) [concrete]
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
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.9: type = fn_type @SubobjectDestroy.WithSelf.Op.loc51_16.1 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.9: %SubobjectDestroy.WithSelf.Op.type.6870fd.9 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.9: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.9), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.ee2: %SubobjectDestroy.type = facet_value %Iterator.f25, (%custom_witness.b80ef6.9) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.9dd: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.ee2) [concrete]
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
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.10: type = fn_type @SubobjectDestroy.WithSelf.Op.loc51_16.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.10: %SubobjectDestroy.WithSelf.Op.type.6870fd.10 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.10: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.10), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.599: %SubobjectDestroy.type = facet_value %Sentinel.b4c, (%custom_witness.b80ef6.10) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.fc2: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.599) [concrete]
// CHECK:STDOUT:   %facet_value.f44: %CppRange_where.type.d682d6.1 = facet_value %DiffMutableRange, (%custom_witness.aa8, %custom_witness.b80ef6.9, %DestroyImpl.impl_witness.9dd, %custom_witness.392, %custom_witness.1c9, %custom_witness.b32, %custom_witness.b80ef6.2, %DestroyImpl.impl_witness.f43, %custom_witness.d0da43.1, %custom_witness.b80ef6.10, %DestroyImpl.impl_witness.fc2) [concrete]
// CHECK:STDOUT:   %tuple.type.b18: type = tuple_type (%Iterator.f25, %Sentinel.b4c) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.11: type = fn_type @SubobjectDestroy.WithSelf.Op.3 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.11: %SubobjectDestroy.WithSelf.Op.type.6870fd.11 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.11: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.11), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.d2e: %SubobjectDestroy.type = facet_value %tuple.type.b18, (%custom_witness.b80ef6.11) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.c40: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.d2e) [concrete]
// CHECK:STDOUT:   %complete_type.bfe: <witness> = complete_type_witness %tuple.type.b18 [concrete]
// CHECK:STDOUT:   %facet_value.fcf: %facet_type.8a3 = facet_value %tuple.type.b18, (%custom_witness.b80ef6.11, %DestroyImpl.impl_witness.c40) [concrete]
// CHECK:STDOUT:   %Iterate.impl_witness.706: <witness> = impl_witness imports.%Iterate.impl_witness_table.121, @T.as_type.as.Iterate.impl(%facet_value.f44) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.type.594: type = fn_type @T.as_type.as.Iterate.impl.NewCursor, @T.as_type.as.Iterate.impl(%facet_value.f44) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.097: %T.as_type.as.Iterate.impl.NewCursor.type.594 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.type.033: type = fn_type @T.as_type.as.Iterate.impl.Next, @T.as_type.as.Iterate.impl(%facet_value.f44) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.c89: %T.as_type.as.Iterate.impl.Next.type.033 = struct_value () [concrete]
// CHECK:STDOUT:   %Iterate.facet.9f6: %Iterate.type = facet_value %DiffMutableRange, (%Iterate.impl_witness.706) [concrete]
// CHECK:STDOUT:   %facet_value.8db: %Iterate_where.type.b70 = facet_value %DiffMutableRange, (%Iterate.impl_witness.706) [concrete]
// CHECK:STDOUT:   %r.patt.5fe: %pattern_type.611 = ref_binding_pattern r [concrete]
// CHECK:STDOUT:   %r.param_patt.a4b: %pattern_type.611 = var_param_pattern %r.patt.5fe [concrete]
// CHECK:STDOUT:   %Iterator.a59: type = class_type @Iterator.2 [concrete]
// CHECK:STDOUT:   %Sentinel.b33: type = class_type @Sentinel.2 [concrete]
// CHECK:STDOUT:   %DiffConstRange.Begin.type: type = fn_type @DiffConstRange.Begin [concrete]
// CHECK:STDOUT:   %DiffConstRange.Begin: %DiffConstRange.Begin.type = struct_value () [concrete]
// CHECK:STDOUT:   %DiffConstRange.End.type: type = fn_type @DiffConstRange.End [concrete]
// CHECK:STDOUT:   %DiffConstRange.End: %DiffConstRange.End.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.7b4: <witness> = custom_witness (%Iterator.a59, %Sentinel.b33, %DiffConstRange.Begin, %DiffConstRange.End), @CppRangeForIterate [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.13: type = fn_type @SubobjectDestroy.WithSelf.Op.loc52_16.1 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.13: %SubobjectDestroy.WithSelf.Op.type.6870fd.13 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.13: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.13), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.0d0: %SubobjectDestroy.type = facet_value %Iterator.a59, (%custom_witness.b80ef6.13) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.791: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.0d0) [concrete]
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
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.14: type = fn_type @SubobjectDestroy.WithSelf.Op.loc52_16.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.14: %SubobjectDestroy.WithSelf.Op.type.6870fd.14 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.14: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.14), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.f42: %SubobjectDestroy.type = facet_value %Sentinel.b33, (%custom_witness.b80ef6.14) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.2a2: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.f42) [concrete]
// CHECK:STDOUT:   %facet_value.115: %CppRange_where.type.d682d6.1 = facet_value %DiffConstRange, (%custom_witness.7b4, %custom_witness.b80ef6.13, %DestroyImpl.impl_witness.791, %custom_witness.0c0, %custom_witness.ffb, %custom_witness.fc2, %custom_witness.b80ef6.2, %DestroyImpl.impl_witness.f43, %custom_witness.d0da43.1, %custom_witness.b80ef6.14, %DestroyImpl.impl_witness.2a2) [concrete]
// CHECK:STDOUT:   %tuple.type.df5: type = tuple_type (%Iterator.a59, %Sentinel.b33) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.15: type = fn_type @SubobjectDestroy.WithSelf.Op.4 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.15: %SubobjectDestroy.WithSelf.Op.type.6870fd.15 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.15: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.15), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.79d: %SubobjectDestroy.type = facet_value %tuple.type.df5, (%custom_witness.b80ef6.15) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.8c3: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.79d) [concrete]
// CHECK:STDOUT:   %complete_type.ad6: <witness> = complete_type_witness %tuple.type.df5 [concrete]
// CHECK:STDOUT:   %facet_value.4bc: %facet_type.8a3 = facet_value %tuple.type.df5, (%custom_witness.b80ef6.15, %DestroyImpl.impl_witness.8c3) [concrete]
// CHECK:STDOUT:   %Iterate.impl_witness.df4: <witness> = impl_witness imports.%Iterate.impl_witness_table.121, @T.as_type.as.Iterate.impl(%facet_value.115) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.type.18c: type = fn_type @T.as_type.as.Iterate.impl.NewCursor, @T.as_type.as.Iterate.impl(%facet_value.115) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.416: %T.as_type.as.Iterate.impl.NewCursor.type.18c = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.type.e78: type = fn_type @T.as_type.as.Iterate.impl.Next, @T.as_type.as.Iterate.impl(%facet_value.115) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.b68: %T.as_type.as.Iterate.impl.Next.type.e78 = struct_value () [concrete]
// CHECK:STDOUT:   %Iterate.facet.3d3: %Iterate.type = facet_value %DiffConstRange, (%Iterate.impl_witness.df4) [concrete]
// CHECK:STDOUT:   %facet_value.eda: %Iterate_where.type.b70 = facet_value %DiffConstRange, (%Iterate.impl_witness.df4) [concrete]
// CHECK:STDOUT:   %r.patt.9c0: %pattern_type.7dd = ref_binding_pattern r [concrete]
// CHECK:STDOUT:   %r.param_patt.992: %pattern_type.7dd = var_param_pattern %r.patt.9c0 [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.e12: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.9f6) [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.aad: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.9f6) [concrete]
// CHECK:STDOUT:   %.c87: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.e12, %Iterate.facet.9f6 [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.specific_fn.5a0: <specific function> = specific_function %T.as_type.as.Iterate.impl.NewCursor.097, @T.as_type.as.Iterate.impl.NewCursor(%facet_value.f44) [concrete]
// CHECK:STDOUT:   %.5db8: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.aad, %Iterate.facet.9f6 [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.specific_fn.134: <specific function> = specific_function %T.as_type.as.Iterate.impl.Next.c89, @T.as_type.as.Iterate.impl.Next(%facet_value.f44) [concrete]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.1d9: <specific function> = specific_function %SelfDestruct, @SelfDestruct(%facet_value.fcf) [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.c77: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.3d3) [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.f88: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.3d3) [concrete]
// CHECK:STDOUT:   %.afc: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.c77, %Iterate.facet.3d3 [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.specific_fn.0cc: <specific function> = specific_function %T.as_type.as.Iterate.impl.NewCursor.416, @T.as_type.as.Iterate.impl.NewCursor(%facet_value.115) [concrete]
// CHECK:STDOUT:   %.5be: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.f88, %Iterate.facet.3d3 [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.specific_fn.4d3: <specific function> = specific_function %T.as_type.as.Iterate.impl.Next.b68, @T.as_type.as.Iterate.impl.Next(%facet_value.115) [concrete]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.dea: <specific function> = specific_function %SelfDestruct, @SelfDestruct(%facet_value.4bc) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .DiffValueType = %DiffValueType.decl
// CHECK:STDOUT:     .DiffMutableRange = %DiffMutableRange.decl
// CHECK:STDOUT:     .DiffConstRange = %DiffConstRange.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.c82: @Optional.%Optional.HasValue.type (%Optional.HasValue.type.3ea) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.HasValue (constants.%Optional.HasValue.f2f)]
// CHECK:STDOUT:   %Core.import_ref.29e: @Optional.%Optional.Get.type (%Optional.Get.type.786) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.Get (constants.%Optional.Get.2d0)]
// CHECK:STDOUT:   %Core.import_ref.9a31: type = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%DefaultOptionalStorage (constants.%DefaultOptionalStorage.f15)]
// CHECK:STDOUT:   %Core.import_ref.819a: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.None.type (%T.as_type.as.OptionalStorage.impl.None.type.76a) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.None (constants.%T.as_type.as.OptionalStorage.impl.None.93b)]
// CHECK:STDOUT:   %Core.import_ref.14a: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Some.type (%T.as_type.as.OptionalStorage.impl.Some.type.4eb) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Some (constants.%T.as_type.as.OptionalStorage.impl.Some.44a)]
// CHECK:STDOUT:   %Core.import_ref.242: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Has.type (%T.as_type.as.OptionalStorage.impl.Has.type.371) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Has (constants.%T.as_type.as.OptionalStorage.impl.Has.8ce)]
// CHECK:STDOUT:   %Core.import_ref.a54: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Get.type (%T.as_type.as.OptionalStorage.impl.Get.type.3e7) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Get (constants.%T.as_type.as.OptionalStorage.impl.Get.f80)]
// CHECK:STDOUT:   %Core.import_ref.5bf = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], unloaded
// CHECK:STDOUT:   %OptionalStorage.impl_witness_table.968 = impl_witness_table (%Core.import_ref.9a31, %Core.import_ref.819a, %Core.import_ref.14a, %Core.import_ref.242, %Core.import_ref.a54, %Core.import_ref.5bf), @T.as_type.as.OptionalStorage.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.822: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.822), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.d6c: %facet_type.d61 = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.Iterate.impl.%facet_value.1 (constants.%facet_value.9da)]
// CHECK:STDOUT:   %Core.import_ref.2ac: %facet_type.8a3 = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.Iterate.impl.%facet_value.2 (constants.%facet_value.bf6)]
// CHECK:STDOUT:   %Core.import_ref.946: @T.as_type.as.Iterate.impl.%T.as_type.as.Iterate.impl.NewCursor.type (%T.as_type.as.Iterate.impl.NewCursor.type.8fb) = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.Iterate.impl.%T.as_type.as.Iterate.impl.NewCursor (constants.%T.as_type.as.Iterate.impl.NewCursor.8a1)]
// CHECK:STDOUT:   %Core.import_ref.cfb: @T.as_type.as.Iterate.impl.%T.as_type.as.Iterate.impl.Next.type (%T.as_type.as.Iterate.impl.Next.type.d4a) = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.Iterate.impl.%T.as_type.as.Iterate.impl.Next (constants.%T.as_type.as.Iterate.impl.Next.5a8)]
// CHECK:STDOUT:   %Iterate.impl_witness_table.121 = impl_witness_table (%Core.import_ref.d6c, %Core.import_ref.2ac, %Core.import_ref.946, %Core.import_ref.cfb), @T.as_type.as.Iterate.impl [concrete]
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
// CHECK:STDOUT: generic fn @TestIterate(%R.loc40_17.2: %Iterate_where.type.b70) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Iterate.lookup_impl_witness: <witness> = lookup_impl_witness %R.loc40_17.1, @Iterate [symbolic = %Iterate.lookup_impl_witness (constants.%Iterate.lookup_impl_witness.0b4)]
// CHECK:STDOUT:   %Iterate.facet.loc44_33.5: %Iterate.type = facet_value %R.as_type.loc40_79.1, (%Iterate.lookup_impl_witness) [symbolic = %Iterate.facet.loc44_33.5 (constants.%Iterate.facet.162)]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.loc44_33.5) [symbolic = %Iterate.WithSelf.NewCursor.type (constants.%Iterate.WithSelf.NewCursor.type.073)]
// CHECK:STDOUT:   %.loc44_33.26: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type, %Iterate.facet.loc44_33.5 [symbolic = %.loc44_33.26 (constants.%.701)]
// CHECK:STDOUT:   %impl.elem2.loc44_33.2: @TestIterate.%.loc44_33.26 (%.701) = impl_witness_access %Iterate.lookup_impl_witness, element2 [symbolic = %impl.elem2.loc44_33.2 (constants.%impl.elem2.cbd)]
// CHECK:STDOUT:   %specific_impl_fn.loc44_33.3: <specific function> = specific_impl_function %impl.elem2.loc44_33.2, @Iterate.WithSelf.NewCursor(%Iterate.facet.loc44_33.5) [symbolic = %specific_impl_fn.loc44_33.3 (constants.%specific_impl_fn.34b)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.loc44_33.5) [symbolic = %Iterate.WithSelf.Next.type (constants.%Iterate.WithSelf.Next.type.5ba)]
// CHECK:STDOUT:   %.loc44_33.27: type = fn_type_with_self_type %Iterate.WithSelf.Next.type, %Iterate.facet.loc44_33.5 [symbolic = %.loc44_33.27 (constants.%.db2)]
// CHECK:STDOUT:   %impl.elem3.loc44_33.2: @TestIterate.%.loc44_33.27 (%.db2) = impl_witness_access %Iterate.lookup_impl_witness, element3 [symbolic = %impl.elem3.loc44_33.2 (constants.%impl.elem3.540)]
// CHECK:STDOUT:   %specific_impl_fn.loc44_33.4: <specific function> = specific_impl_function %impl.elem3.loc44_33.2, @Iterate.WithSelf.Next(%Iterate.facet.loc44_33.5) [symbolic = %specific_impl_fn.loc44_33.4 (constants.%specific_impl_fn.1b6)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%r.param: ref @TestIterate.%R.as_type.loc40_79.1 (%R.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %i.patt: %pattern_type.149 = value_binding_pattern i [concrete = constants.%i.patt]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %r.ref: ref @TestIterate.%R.as_type.loc40_79.1 (%R.as_type) = name_ref r, %r
// CHECK:STDOUT:     %impl.elem2.loc44_33.1: @TestIterate.%.loc44_33.26 (%.701) = impl_witness_access constants.%Iterate.lookup_impl_witness.0b4, element2 [symbolic = %impl.elem2.loc44_33.2 (constants.%impl.elem2.cbd)]
// CHECK:STDOUT:     %bound_method.loc44_33.1: <bound method> = bound_method %r.ref, %impl.elem2.loc44_33.1
// CHECK:STDOUT:     %Iterate.facet.loc44_33.1: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.0b4) [symbolic = %Iterate.facet.loc44_33.5 (constants.%Iterate.facet.162)]
// CHECK:STDOUT:     %.loc44_33.1: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc44_33.1 [symbolic = %Iterate.facet.loc44_33.5 (constants.%Iterate.facet.162)]
// CHECK:STDOUT:     %Iterate.facet.loc44_33.2: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.0b4) [symbolic = %Iterate.facet.loc44_33.5 (constants.%Iterate.facet.162)]
// CHECK:STDOUT:     %.loc44_33.2: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc44_33.2 [symbolic = %Iterate.facet.loc44_33.5 (constants.%Iterate.facet.162)]
// CHECK:STDOUT:     %specific_impl_fn.loc44_33.1: <specific function> = specific_impl_function %impl.elem2.loc44_33.1, @Iterate.WithSelf.NewCursor(constants.%Iterate.facet.162) [symbolic = %specific_impl_fn.loc44_33.3 (constants.%specific_impl_fn.34b)]
// CHECK:STDOUT:     %bound_method.loc44_33.2: <bound method> = bound_method %r.ref, %specific_impl_fn.loc44_33.1
// CHECK:STDOUT:     %var: ref @TestIterate.%as_type (%as_type.903) = var_storage invalid
// CHECK:STDOUT:     %.loc44_32.1: @TestIterate.%R.as_type.loc40_79.1 (%R.as_type) = acquire_value %r.ref
// CHECK:STDOUT:     %Iterate.WithSelf.NewCursor.call: init @TestIterate.%as_type (%as_type.903) to %var = call %bound_method.loc44_33.2(%.loc44_32.1)
// CHECK:STDOUT:     assign %var, %Iterate.WithSelf.NewCursor.call
// CHECK:STDOUT:     br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.next:
// CHECK:STDOUT:     %impl.elem3.loc44_33.1: @TestIterate.%.loc44_33.27 (%.db2) = impl_witness_access constants.%Iterate.lookup_impl_witness.0b4, element3 [symbolic = %impl.elem3.loc44_33.2 (constants.%impl.elem3.540)]
// CHECK:STDOUT:     %bound_method.loc44_33.3: <bound method> = bound_method %r.ref, %impl.elem3.loc44_33.1
// CHECK:STDOUT:     %Iterate.facet.loc44_33.3: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.0b4) [symbolic = %Iterate.facet.loc44_33.5 (constants.%Iterate.facet.162)]
// CHECK:STDOUT:     %.loc44_33.3: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc44_33.3 [symbolic = %Iterate.facet.loc44_33.5 (constants.%Iterate.facet.162)]
// CHECK:STDOUT:     %Iterate.facet.loc44_33.4: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.0b4) [symbolic = %Iterate.facet.loc44_33.5 (constants.%Iterate.facet.162)]
// CHECK:STDOUT:     %.loc44_33.4: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc44_33.4 [symbolic = %Iterate.facet.loc44_33.5 (constants.%Iterate.facet.162)]
// CHECK:STDOUT:     %.loc44_33.5: %facet_type.8a3 = converted constants.%as_type.903, constants.%impl.elem1.7f1 [symbolic = %impl.elem1 (constants.%impl.elem1.7f1)]
// CHECK:STDOUT:     %.loc44_33.6: %facet_type.8a3 = converted constants.%as_type.903, constants.%impl.elem1.7f1 [symbolic = %impl.elem1 (constants.%impl.elem1.7f1)]
// CHECK:STDOUT:     %specific_impl_fn.loc44_33.2: <specific function> = specific_impl_function %impl.elem3.loc44_33.1, @Iterate.WithSelf.Next(constants.%Iterate.facet.162) [symbolic = %specific_impl_fn.loc44_33.4 (constants.%specific_impl_fn.1b6)]
// CHECK:STDOUT:     %bound_method.loc44_33.4: <bound method> = bound_method %r.ref, %specific_impl_fn.loc44_33.2
// CHECK:STDOUT:     %.loc44_33.7: ref %Optional.8c2 = temporary_storage
// CHECK:STDOUT:     %.loc44_32.2: @TestIterate.%R.as_type.loc40_79.1 (%R.as_type) = acquire_value %r.ref
// CHECK:STDOUT:     %Iterate.WithSelf.Next.call: init %Optional.8c2 to %.loc44_33.7 = call %bound_method.loc44_33.4(%.loc44_32.2, %var)
// CHECK:STDOUT:     %.loc44_33.8: ref %Optional.8c2 = temporary %.loc44_33.7, %Iterate.WithSelf.Next.call
// CHECK:STDOUT:     %.loc44_33.9: %Optional.HasValue.type.df4 = specific_constant imports.%Core.import_ref.c82, @Optional(constants.%OptionalStorage.facet.591) [concrete = constants.%Optional.HasValue.7c9]
// CHECK:STDOUT:     %HasValue.ref: %Optional.HasValue.type.df4 = name_ref HasValue, %.loc44_33.9 [concrete = constants.%Optional.HasValue.7c9]
// CHECK:STDOUT:     %Optional.HasValue.bound: <bound method> = bound_method %.loc44_33.8, %HasValue.ref
// CHECK:STDOUT:     %Optional.HasValue.specific_fn: <specific function> = specific_function %HasValue.ref, @Optional.HasValue(constants.%OptionalStorage.facet.591) [concrete = constants.%Optional.HasValue.specific_fn.7b9]
// CHECK:STDOUT:     %bound_method.loc44_33.5: <bound method> = bound_method %.loc44_33.8, %Optional.HasValue.specific_fn
// CHECK:STDOUT:     %.loc44_33.10: %Optional.8c2 = acquire_value %.loc44_33.8
// CHECK:STDOUT:     %Optional.HasValue.call: init bool = call %bound_method.loc44_33.5(%.loc44_33.10)
// CHECK:STDOUT:     %.loc44_33.11: bool = value_of_initializer %Optional.HasValue.call
// CHECK:STDOUT:     %.loc44_33.12: bool = converted %Optional.HasValue.call, %.loc44_33.11
// CHECK:STDOUT:     if %.loc44_33.12 br !for.body else br !for.done
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.body:
// CHECK:STDOUT:     %.loc44_33.13: %Optional.Get.type.18b = specific_constant imports.%Core.import_ref.29e, @Optional(constants.%OptionalStorage.facet.591) [concrete = constants.%Optional.Get.57b]
// CHECK:STDOUT:     %Get.ref: %Optional.Get.type.18b = name_ref Get, %.loc44_33.13 [concrete = constants.%Optional.Get.57b]
// CHECK:STDOUT:     %Optional.Get.bound: <bound method> = bound_method %.loc44_33.8, %Get.ref
// CHECK:STDOUT:     %Optional.Get.specific_fn: <specific function> = specific_function %Get.ref, @Optional.Get(constants.%OptionalStorage.facet.591) [concrete = constants.%Optional.Get.specific_fn.c38]
// CHECK:STDOUT:     %bound_method.loc44_33.6: <bound method> = bound_method %.loc44_33.8, %Optional.Get.specific_fn
// CHECK:STDOUT:     %.loc44_33.14: ref %DiffValueType = temporary_storage
// CHECK:STDOUT:     %.loc44_33.15: %Optional.8c2 = acquire_value %.loc44_33.8
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
// CHECK:STDOUT:     %facet_value.loc44_33.1: %facet_type.8a3 = facet_value constants.%DiffValueType, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.f43) [concrete = constants.%facet_value.0ea]
// CHECK:STDOUT:     %.loc44_33.18: %facet_type.8a3 = converted constants.%DiffValueType, %facet_value.loc44_33.1 [concrete = constants.%facet_value.0ea]
// CHECK:STDOUT:     %facet_value.loc44_33.2: %facet_type.8a3 = facet_value constants.%DiffValueType, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.f43) [concrete = constants.%facet_value.0ea]
// CHECK:STDOUT:     %.loc44_33.19: %facet_type.8a3 = converted constants.%DiffValueType, %facet_value.loc44_33.2 [concrete = constants.%facet_value.0ea]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %SelfDestruct.call.loc44_33.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc44_33.16)
// CHECK:STDOUT:     %facet_value.loc44_33.3: %facet_type.8a3 = facet_value constants.%Optional.8c2, (constants.%custom_witness.b80ef6.8, constants.%DestroyImpl.impl_witness.355) [concrete = constants.%facet_value.fb7]
// CHECK:STDOUT:     %.loc44_33.20: %facet_type.8a3 = converted constants.%Optional.8c2, %facet_value.loc44_33.3 [concrete = constants.%facet_value.fb7]
// CHECK:STDOUT:     %facet_value.loc44_33.4: %facet_type.8a3 = facet_value constants.%Optional.8c2, (constants.%custom_witness.b80ef6.8, constants.%DestroyImpl.impl_witness.355) [concrete = constants.%facet_value.fb7]
// CHECK:STDOUT:     %.loc44_33.21: %facet_type.8a3 = converted constants.%Optional.8c2, %facet_value.loc44_33.4 [concrete = constants.%facet_value.fb7]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %SelfDestruct.call.loc44_33.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc44_33.8)
// CHECK:STDOUT:     br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.done:
// CHECK:STDOUT:     %facet_value.loc44_33.5: %facet_type.8a3 = facet_value constants.%Optional.8c2, (constants.%custom_witness.b80ef6.8, constants.%DestroyImpl.impl_witness.355) [concrete = constants.%facet_value.fb7]
// CHECK:STDOUT:     %.loc44_33.22: %facet_type.8a3 = converted constants.%Optional.8c2, %facet_value.loc44_33.5 [concrete = constants.%facet_value.fb7]
// CHECK:STDOUT:     %facet_value.loc44_33.6: %facet_type.8a3 = facet_value constants.%Optional.8c2, (constants.%custom_witness.b80ef6.8, constants.%DestroyImpl.impl_witness.355) [concrete = constants.%facet_value.fb7]
// CHECK:STDOUT:     %.loc44_33.23: %facet_type.8a3 = converted constants.%Optional.8c2, %facet_value.loc44_33.6 [concrete = constants.%facet_value.fb7]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %SelfDestruct.call.loc44_33.3: init %empty_tuple.type = call %SelfDestruct.specific_fn.3(%.loc44_33.8)
// CHECK:STDOUT:     %.loc44_33.24: %facet_type.8a3 = converted constants.%as_type.903, constants.%impl.elem1.7f1 [symbolic = %impl.elem1 (constants.%impl.elem1.7f1)]
// CHECK:STDOUT:     %.loc44_33.25: %facet_type.8a3 = converted constants.%as_type.903, constants.%impl.elem1.7f1 [symbolic = %impl.elem1 (constants.%impl.elem1.7f1)]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %SelfDestruct.call.loc44_33.4: init %empty_tuple.type = call %SelfDestruct.specific_fn.4(%var)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc44_33.1(%self.param: ref %.a18) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc44_33.2(%self.param: ref %MaybeUnformed.31b) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc44_33.3(%self.param: ref bool) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc44_33.4(%self.param: ref %struct_type.value.has_value.08b) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc44_33.5(%self.param: ref %DefaultOptionalStorage.a0f) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc44_33.6(%self.param: ref %Optional.8c2) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestRangeFor(%m.param: ref %DiffMutableRange, %c.param: %DiffConstRange) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %i.patt.loc59: %pattern_type.149 = value_binding_pattern i [concrete = constants.%i.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %m.ref: ref %DiffMutableRange = name_ref m, %m
// CHECK:STDOUT:   %impl.elem2.loc59: %.c87 = impl_witness_access constants.%Iterate.impl_witness.706, element2 [concrete = constants.%T.as_type.as.Iterate.impl.NewCursor.097]
// CHECK:STDOUT:   %bound_method.loc59_33.1: <bound method> = bound_method %m.ref, %impl.elem2.loc59
// CHECK:STDOUT:   %facet_value.loc59_33.1: %CppRange_where.type.d682d6.1 = facet_value constants.%DiffMutableRange, (constants.%custom_witness.aa8, constants.%custom_witness.b80ef6.9, constants.%DestroyImpl.impl_witness.9dd, constants.%custom_witness.392, constants.%custom_witness.1c9, constants.%custom_witness.b32, constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.f43, constants.%custom_witness.d0da43.1, constants.%custom_witness.b80ef6.10, constants.%DestroyImpl.impl_witness.fc2) [concrete = constants.%facet_value.f44]
// CHECK:STDOUT:   %.loc59_33.1: %CppRange_where.type.d682d6.1 = converted constants.%DiffMutableRange, %facet_value.loc59_33.1 [concrete = constants.%facet_value.f44]
// CHECK:STDOUT:   %facet_value.loc59_33.2: %CppRange_where.type.d682d6.1 = facet_value constants.%DiffMutableRange, (constants.%custom_witness.aa8, constants.%custom_witness.b80ef6.9, constants.%DestroyImpl.impl_witness.9dd, constants.%custom_witness.392, constants.%custom_witness.1c9, constants.%custom_witness.b32, constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.f43, constants.%custom_witness.d0da43.1, constants.%custom_witness.b80ef6.10, constants.%DestroyImpl.impl_witness.fc2) [concrete = constants.%facet_value.f44]
// CHECK:STDOUT:   %.loc59_33.2: %CppRange_where.type.d682d6.1 = converted constants.%DiffMutableRange, %facet_value.loc59_33.2 [concrete = constants.%facet_value.f44]
// CHECK:STDOUT:   %specific_fn.loc59_33.1: <specific function> = specific_function %impl.elem2.loc59, @T.as_type.as.Iterate.impl.NewCursor(constants.%facet_value.f44) [concrete = constants.%T.as_type.as.Iterate.impl.NewCursor.specific_fn.5a0]
// CHECK:STDOUT:   %bound_method.loc59_33.2: <bound method> = bound_method %m.ref, %specific_fn.loc59_33.1
// CHECK:STDOUT:   %var.loc59: ref %tuple.type.b18 = var_storage invalid
// CHECK:STDOUT:   %.loc59_32.1: %DiffMutableRange = acquire_value %m.ref
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.call.loc59: init %tuple.type.b18 to %var.loc59 = call %bound_method.loc59_33.2(%.loc59_32.1)
// CHECK:STDOUT:   assign %var.loc59, %T.as_type.as.Iterate.impl.NewCursor.call.loc59
// CHECK:STDOUT:   br !for.next.loc59
// CHECK:STDOUT:
// CHECK:STDOUT: !for.next.loc59:
// CHECK:STDOUT:   %impl.elem3.loc59: %.5db8 = impl_witness_access constants.%Iterate.impl_witness.706, element3 [concrete = constants.%T.as_type.as.Iterate.impl.Next.c89]
// CHECK:STDOUT:   %bound_method.loc59_33.3: <bound method> = bound_method %m.ref, %impl.elem3.loc59
// CHECK:STDOUT:   %facet_value.loc59_33.3: %CppRange_where.type.d682d6.1 = facet_value constants.%DiffMutableRange, (constants.%custom_witness.aa8, constants.%custom_witness.b80ef6.9, constants.%DestroyImpl.impl_witness.9dd, constants.%custom_witness.392, constants.%custom_witness.1c9, constants.%custom_witness.b32, constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.f43, constants.%custom_witness.d0da43.1, constants.%custom_witness.b80ef6.10, constants.%DestroyImpl.impl_witness.fc2) [concrete = constants.%facet_value.f44]
// CHECK:STDOUT:   %.loc59_33.3: %CppRange_where.type.d682d6.1 = converted constants.%DiffMutableRange, %facet_value.loc59_33.3 [concrete = constants.%facet_value.f44]
// CHECK:STDOUT:   %facet_value.loc59_33.4: %CppRange_where.type.d682d6.1 = facet_value constants.%DiffMutableRange, (constants.%custom_witness.aa8, constants.%custom_witness.b80ef6.9, constants.%DestroyImpl.impl_witness.9dd, constants.%custom_witness.392, constants.%custom_witness.1c9, constants.%custom_witness.b32, constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.f43, constants.%custom_witness.d0da43.1, constants.%custom_witness.b80ef6.10, constants.%DestroyImpl.impl_witness.fc2) [concrete = constants.%facet_value.f44]
// CHECK:STDOUT:   %.loc59_33.4: %CppRange_where.type.d682d6.1 = converted constants.%DiffMutableRange, %facet_value.loc59_33.4 [concrete = constants.%facet_value.f44]
// CHECK:STDOUT:   %specific_fn.loc59_33.2: <specific function> = specific_function %impl.elem3.loc59, @T.as_type.as.Iterate.impl.Next(constants.%facet_value.f44) [concrete = constants.%T.as_type.as.Iterate.impl.Next.specific_fn.134]
// CHECK:STDOUT:   %bound_method.loc59_33.4: <bound method> = bound_method %m.ref, %specific_fn.loc59_33.2
// CHECK:STDOUT:   %.loc59_33.5: ref %Optional.8c2 = temporary_storage
// CHECK:STDOUT:   %.loc59_32.2: %DiffMutableRange = acquire_value %m.ref
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.call.loc59: init %Optional.8c2 to %.loc59_33.5 = call %bound_method.loc59_33.4(%.loc59_32.2, %var.loc59)
// CHECK:STDOUT:   %.loc59_33.6: ref %Optional.8c2 = temporary %.loc59_33.5, %T.as_type.as.Iterate.impl.Next.call.loc59
// CHECK:STDOUT:   %.loc59_33.7: %Optional.HasValue.type.df4 = specific_constant imports.%Core.import_ref.c82, @Optional(constants.%OptionalStorage.facet.591) [concrete = constants.%Optional.HasValue.7c9]
// CHECK:STDOUT:   %HasValue.ref.loc59: %Optional.HasValue.type.df4 = name_ref HasValue, %.loc59_33.7 [concrete = constants.%Optional.HasValue.7c9]
// CHECK:STDOUT:   %Optional.HasValue.bound.loc59: <bound method> = bound_method %.loc59_33.6, %HasValue.ref.loc59
// CHECK:STDOUT:   %Optional.HasValue.specific_fn.loc59: <specific function> = specific_function %HasValue.ref.loc59, @Optional.HasValue(constants.%OptionalStorage.facet.591) [concrete = constants.%Optional.HasValue.specific_fn.7b9]
// CHECK:STDOUT:   %bound_method.loc59_33.5: <bound method> = bound_method %.loc59_33.6, %Optional.HasValue.specific_fn.loc59
// CHECK:STDOUT:   %.loc59_33.8: %Optional.8c2 = acquire_value %.loc59_33.6
// CHECK:STDOUT:   %Optional.HasValue.call.loc59: init bool = call %bound_method.loc59_33.5(%.loc59_33.8)
// CHECK:STDOUT:   %.loc59_33.9: bool = value_of_initializer %Optional.HasValue.call.loc59
// CHECK:STDOUT:   %.loc59_33.10: bool = converted %Optional.HasValue.call.loc59, %.loc59_33.9
// CHECK:STDOUT:   if %.loc59_33.10 br !for.body.loc59 else br !for.done.loc59
// CHECK:STDOUT:
// CHECK:STDOUT: !for.body.loc59:
// CHECK:STDOUT:   %.loc59_33.11: %Optional.Get.type.18b = specific_constant imports.%Core.import_ref.29e, @Optional(constants.%OptionalStorage.facet.591) [concrete = constants.%Optional.Get.57b]
// CHECK:STDOUT:   %Get.ref.loc59: %Optional.Get.type.18b = name_ref Get, %.loc59_33.11 [concrete = constants.%Optional.Get.57b]
// CHECK:STDOUT:   %Optional.Get.bound.loc59: <bound method> = bound_method %.loc59_33.6, %Get.ref.loc59
// CHECK:STDOUT:   %Optional.Get.specific_fn.loc59: <specific function> = specific_function %Get.ref.loc59, @Optional.Get(constants.%OptionalStorage.facet.591) [concrete = constants.%Optional.Get.specific_fn.c38]
// CHECK:STDOUT:   %bound_method.loc59_33.6: <bound method> = bound_method %.loc59_33.6, %Optional.Get.specific_fn.loc59
// CHECK:STDOUT:   %.loc59_33.12: ref %DiffValueType = temporary_storage
// CHECK:STDOUT:   %.loc59_33.13: %Optional.8c2 = acquire_value %.loc59_33.6
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
// CHECK:STDOUT:   %facet_value.loc59_33.5: %facet_type.8a3 = facet_value constants.%DiffValueType, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.f43) [concrete = constants.%facet_value.0ea]
// CHECK:STDOUT:   %.loc59_33.16: %facet_type.8a3 = converted constants.%DiffValueType, %facet_value.loc59_33.5 [concrete = constants.%facet_value.0ea]
// CHECK:STDOUT:   %facet_value.loc59_33.6: %facet_type.8a3 = facet_value constants.%DiffValueType, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.f43) [concrete = constants.%facet_value.0ea]
// CHECK:STDOUT:   %.loc59_33.17: %facet_type.8a3 = converted constants.%DiffValueType, %facet_value.loc59_33.6 [concrete = constants.%facet_value.0ea]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc59_33.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc59_33.14)
// CHECK:STDOUT:   %facet_value.loc59_33.7: %facet_type.8a3 = facet_value constants.%Optional.8c2, (constants.%custom_witness.b80ef6.8, constants.%DestroyImpl.impl_witness.355) [concrete = constants.%facet_value.fb7]
// CHECK:STDOUT:   %.loc59_33.18: %facet_type.8a3 = converted constants.%Optional.8c2, %facet_value.loc59_33.7 [concrete = constants.%facet_value.fb7]
// CHECK:STDOUT:   %facet_value.loc59_33.8: %facet_type.8a3 = facet_value constants.%Optional.8c2, (constants.%custom_witness.b80ef6.8, constants.%DestroyImpl.impl_witness.355) [concrete = constants.%facet_value.fb7]
// CHECK:STDOUT:   %.loc59_33.19: %facet_type.8a3 = converted constants.%Optional.8c2, %facet_value.loc59_33.8 [concrete = constants.%facet_value.fb7]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc59_33.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc59_33.6)
// CHECK:STDOUT:   br !for.next.loc59
// CHECK:STDOUT:
// CHECK:STDOUT: !for.done.loc59:
// CHECK:STDOUT:   %facet_value.loc59_33.9: %facet_type.8a3 = facet_value constants.%Optional.8c2, (constants.%custom_witness.b80ef6.8, constants.%DestroyImpl.impl_witness.355) [concrete = constants.%facet_value.fb7]
// CHECK:STDOUT:   %.loc59_33.20: %facet_type.8a3 = converted constants.%Optional.8c2, %facet_value.loc59_33.9 [concrete = constants.%facet_value.fb7]
// CHECK:STDOUT:   %facet_value.loc59_33.10: %facet_type.8a3 = facet_value constants.%Optional.8c2, (constants.%custom_witness.b80ef6.8, constants.%DestroyImpl.impl_witness.355) [concrete = constants.%facet_value.fb7]
// CHECK:STDOUT:   %.loc59_33.21: %facet_type.8a3 = converted constants.%Optional.8c2, %facet_value.loc59_33.10 [concrete = constants.%facet_value.fb7]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc59_33.3: init %empty_tuple.type = call %SelfDestruct.specific_fn.3(%.loc59_33.6)
// CHECK:STDOUT:   %facet_value.loc59_33.11: %facet_type.8a3 = facet_value constants.%tuple.type.b18, (constants.%custom_witness.b80ef6.11, constants.%DestroyImpl.impl_witness.c40) [concrete = constants.%facet_value.fcf]
// CHECK:STDOUT:   %.loc59_33.22: %facet_type.8a3 = converted constants.%tuple.type.b18, %facet_value.loc59_33.11 [concrete = constants.%facet_value.fcf]
// CHECK:STDOUT:   %facet_value.loc59_33.12: %facet_type.8a3 = facet_value constants.%tuple.type.b18, (constants.%custom_witness.b80ef6.11, constants.%DestroyImpl.impl_witness.c40) [concrete = constants.%facet_value.fcf]
// CHECK:STDOUT:   %.loc59_33.23: %facet_type.8a3 = converted constants.%tuple.type.b18, %facet_value.loc59_33.12 [concrete = constants.%facet_value.fcf]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc59_33.4: init %empty_tuple.type = call %SelfDestruct.specific_fn.4(%var.loc59)
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %i.patt.loc63: %pattern_type.149 = value_binding_pattern i [concrete = constants.%i.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c.ref: %DiffConstRange = name_ref c, %c
// CHECK:STDOUT:   %impl.elem2.loc63: %.afc = impl_witness_access constants.%Iterate.impl_witness.df4, element2 [concrete = constants.%T.as_type.as.Iterate.impl.NewCursor.416]
// CHECK:STDOUT:   %bound_method.loc63_33.1: <bound method> = bound_method %c.ref, %impl.elem2.loc63
// CHECK:STDOUT:   %facet_value.loc63_33.1: %CppRange_where.type.d682d6.1 = facet_value constants.%DiffConstRange, (constants.%custom_witness.7b4, constants.%custom_witness.b80ef6.13, constants.%DestroyImpl.impl_witness.791, constants.%custom_witness.0c0, constants.%custom_witness.ffb, constants.%custom_witness.fc2, constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.f43, constants.%custom_witness.d0da43.1, constants.%custom_witness.b80ef6.14, constants.%DestroyImpl.impl_witness.2a2) [concrete = constants.%facet_value.115]
// CHECK:STDOUT:   %.loc63_33.1: %CppRange_where.type.d682d6.1 = converted constants.%DiffConstRange, %facet_value.loc63_33.1 [concrete = constants.%facet_value.115]
// CHECK:STDOUT:   %facet_value.loc63_33.2: %CppRange_where.type.d682d6.1 = facet_value constants.%DiffConstRange, (constants.%custom_witness.7b4, constants.%custom_witness.b80ef6.13, constants.%DestroyImpl.impl_witness.791, constants.%custom_witness.0c0, constants.%custom_witness.ffb, constants.%custom_witness.fc2, constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.f43, constants.%custom_witness.d0da43.1, constants.%custom_witness.b80ef6.14, constants.%DestroyImpl.impl_witness.2a2) [concrete = constants.%facet_value.115]
// CHECK:STDOUT:   %.loc63_33.2: %CppRange_where.type.d682d6.1 = converted constants.%DiffConstRange, %facet_value.loc63_33.2 [concrete = constants.%facet_value.115]
// CHECK:STDOUT:   %specific_fn.loc63_33.1: <specific function> = specific_function %impl.elem2.loc63, @T.as_type.as.Iterate.impl.NewCursor(constants.%facet_value.115) [concrete = constants.%T.as_type.as.Iterate.impl.NewCursor.specific_fn.0cc]
// CHECK:STDOUT:   %bound_method.loc63_33.2: <bound method> = bound_method %c.ref, %specific_fn.loc63_33.1
// CHECK:STDOUT:   %var.loc63: ref %tuple.type.df5 = var_storage invalid
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.call.loc63: init %tuple.type.df5 to %var.loc63 = call %bound_method.loc63_33.2(%c.ref)
// CHECK:STDOUT:   assign %var.loc63, %T.as_type.as.Iterate.impl.NewCursor.call.loc63
// CHECK:STDOUT:   br !for.next.loc63
// CHECK:STDOUT:
// CHECK:STDOUT: !for.next.loc63:
// CHECK:STDOUT:   %impl.elem3.loc63: %.5be = impl_witness_access constants.%Iterate.impl_witness.df4, element3 [concrete = constants.%T.as_type.as.Iterate.impl.Next.b68]
// CHECK:STDOUT:   %bound_method.loc63_33.3: <bound method> = bound_method %c.ref, %impl.elem3.loc63
// CHECK:STDOUT:   %facet_value.loc63_33.3: %CppRange_where.type.d682d6.1 = facet_value constants.%DiffConstRange, (constants.%custom_witness.7b4, constants.%custom_witness.b80ef6.13, constants.%DestroyImpl.impl_witness.791, constants.%custom_witness.0c0, constants.%custom_witness.ffb, constants.%custom_witness.fc2, constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.f43, constants.%custom_witness.d0da43.1, constants.%custom_witness.b80ef6.14, constants.%DestroyImpl.impl_witness.2a2) [concrete = constants.%facet_value.115]
// CHECK:STDOUT:   %.loc63_33.3: %CppRange_where.type.d682d6.1 = converted constants.%DiffConstRange, %facet_value.loc63_33.3 [concrete = constants.%facet_value.115]
// CHECK:STDOUT:   %facet_value.loc63_33.4: %CppRange_where.type.d682d6.1 = facet_value constants.%DiffConstRange, (constants.%custom_witness.7b4, constants.%custom_witness.b80ef6.13, constants.%DestroyImpl.impl_witness.791, constants.%custom_witness.0c0, constants.%custom_witness.ffb, constants.%custom_witness.fc2, constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.f43, constants.%custom_witness.d0da43.1, constants.%custom_witness.b80ef6.14, constants.%DestroyImpl.impl_witness.2a2) [concrete = constants.%facet_value.115]
// CHECK:STDOUT:   %.loc63_33.4: %CppRange_where.type.d682d6.1 = converted constants.%DiffConstRange, %facet_value.loc63_33.4 [concrete = constants.%facet_value.115]
// CHECK:STDOUT:   %specific_fn.loc63_33.2: <specific function> = specific_function %impl.elem3.loc63, @T.as_type.as.Iterate.impl.Next(constants.%facet_value.115) [concrete = constants.%T.as_type.as.Iterate.impl.Next.specific_fn.4d3]
// CHECK:STDOUT:   %bound_method.loc63_33.4: <bound method> = bound_method %c.ref, %specific_fn.loc63_33.2
// CHECK:STDOUT:   %.loc63_33.5: ref %Optional.8c2 = temporary_storage
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.call.loc63: init %Optional.8c2 to %.loc63_33.5 = call %bound_method.loc63_33.4(%c.ref, %var.loc63)
// CHECK:STDOUT:   %.loc63_33.6: ref %Optional.8c2 = temporary %.loc63_33.5, %T.as_type.as.Iterate.impl.Next.call.loc63
// CHECK:STDOUT:   %.loc63_33.7: %Optional.HasValue.type.df4 = specific_constant imports.%Core.import_ref.c82, @Optional(constants.%OptionalStorage.facet.591) [concrete = constants.%Optional.HasValue.7c9]
// CHECK:STDOUT:   %HasValue.ref.loc63: %Optional.HasValue.type.df4 = name_ref HasValue, %.loc63_33.7 [concrete = constants.%Optional.HasValue.7c9]
// CHECK:STDOUT:   %Optional.HasValue.bound.loc63: <bound method> = bound_method %.loc63_33.6, %HasValue.ref.loc63
// CHECK:STDOUT:   %Optional.HasValue.specific_fn.loc63: <specific function> = specific_function %HasValue.ref.loc63, @Optional.HasValue(constants.%OptionalStorage.facet.591) [concrete = constants.%Optional.HasValue.specific_fn.7b9]
// CHECK:STDOUT:   %bound_method.loc63_33.5: <bound method> = bound_method %.loc63_33.6, %Optional.HasValue.specific_fn.loc63
// CHECK:STDOUT:   %.loc63_33.8: %Optional.8c2 = acquire_value %.loc63_33.6
// CHECK:STDOUT:   %Optional.HasValue.call.loc63: init bool = call %bound_method.loc63_33.5(%.loc63_33.8)
// CHECK:STDOUT:   %.loc63_33.9: bool = value_of_initializer %Optional.HasValue.call.loc63
// CHECK:STDOUT:   %.loc63_33.10: bool = converted %Optional.HasValue.call.loc63, %.loc63_33.9
// CHECK:STDOUT:   if %.loc63_33.10 br !for.body.loc63 else br !for.done.loc63
// CHECK:STDOUT:
// CHECK:STDOUT: !for.body.loc63:
// CHECK:STDOUT:   %.loc63_33.11: %Optional.Get.type.18b = specific_constant imports.%Core.import_ref.29e, @Optional(constants.%OptionalStorage.facet.591) [concrete = constants.%Optional.Get.57b]
// CHECK:STDOUT:   %Get.ref.loc63: %Optional.Get.type.18b = name_ref Get, %.loc63_33.11 [concrete = constants.%Optional.Get.57b]
// CHECK:STDOUT:   %Optional.Get.bound.loc63: <bound method> = bound_method %.loc63_33.6, %Get.ref.loc63
// CHECK:STDOUT:   %Optional.Get.specific_fn.loc63: <specific function> = specific_function %Get.ref.loc63, @Optional.Get(constants.%OptionalStorage.facet.591) [concrete = constants.%Optional.Get.specific_fn.c38]
// CHECK:STDOUT:   %bound_method.loc63_33.6: <bound method> = bound_method %.loc63_33.6, %Optional.Get.specific_fn.loc63
// CHECK:STDOUT:   %.loc63_33.12: ref %DiffValueType = temporary_storage
// CHECK:STDOUT:   %.loc63_33.13: %Optional.8c2 = acquire_value %.loc63_33.6
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
// CHECK:STDOUT:   %facet_value.loc63_33.5: %facet_type.8a3 = facet_value constants.%DiffValueType, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.f43) [concrete = constants.%facet_value.0ea]
// CHECK:STDOUT:   %.loc63_33.16: %facet_type.8a3 = converted constants.%DiffValueType, %facet_value.loc63_33.5 [concrete = constants.%facet_value.0ea]
// CHECK:STDOUT:   %facet_value.loc63_33.6: %facet_type.8a3 = facet_value constants.%DiffValueType, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.f43) [concrete = constants.%facet_value.0ea]
// CHECK:STDOUT:   %.loc63_33.17: %facet_type.8a3 = converted constants.%DiffValueType, %facet_value.loc63_33.6 [concrete = constants.%facet_value.0ea]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc63_33.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.5(%.loc63_33.14)
// CHECK:STDOUT:   %facet_value.loc63_33.7: %facet_type.8a3 = facet_value constants.%Optional.8c2, (constants.%custom_witness.b80ef6.8, constants.%DestroyImpl.impl_witness.355) [concrete = constants.%facet_value.fb7]
// CHECK:STDOUT:   %.loc63_33.18: %facet_type.8a3 = converted constants.%Optional.8c2, %facet_value.loc63_33.7 [concrete = constants.%facet_value.fb7]
// CHECK:STDOUT:   %facet_value.loc63_33.8: %facet_type.8a3 = facet_value constants.%Optional.8c2, (constants.%custom_witness.b80ef6.8, constants.%DestroyImpl.impl_witness.355) [concrete = constants.%facet_value.fb7]
// CHECK:STDOUT:   %.loc63_33.19: %facet_type.8a3 = converted constants.%Optional.8c2, %facet_value.loc63_33.8 [concrete = constants.%facet_value.fb7]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc63_33.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.6(%.loc63_33.6)
// CHECK:STDOUT:   br !for.next.loc63
// CHECK:STDOUT:
// CHECK:STDOUT: !for.done.loc63:
// CHECK:STDOUT:   %facet_value.loc63_33.9: %facet_type.8a3 = facet_value constants.%Optional.8c2, (constants.%custom_witness.b80ef6.8, constants.%DestroyImpl.impl_witness.355) [concrete = constants.%facet_value.fb7]
// CHECK:STDOUT:   %.loc63_33.20: %facet_type.8a3 = converted constants.%Optional.8c2, %facet_value.loc63_33.9 [concrete = constants.%facet_value.fb7]
// CHECK:STDOUT:   %facet_value.loc63_33.10: %facet_type.8a3 = facet_value constants.%Optional.8c2, (constants.%custom_witness.b80ef6.8, constants.%DestroyImpl.impl_witness.355) [concrete = constants.%facet_value.fb7]
// CHECK:STDOUT:   %.loc63_33.21: %facet_type.8a3 = converted constants.%Optional.8c2, %facet_value.loc63_33.10 [concrete = constants.%facet_value.fb7]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc63_33.3: init %empty_tuple.type = call %SelfDestruct.specific_fn.7(%.loc63_33.6)
// CHECK:STDOUT:   %facet_value.loc63_33.11: %facet_type.8a3 = facet_value constants.%tuple.type.df5, (constants.%custom_witness.b80ef6.15, constants.%DestroyImpl.impl_witness.8c3) [concrete = constants.%facet_value.4bc]
// CHECK:STDOUT:   %.loc63_33.22: %facet_type.8a3 = converted constants.%tuple.type.df5, %facet_value.loc63_33.11 [concrete = constants.%facet_value.4bc]
// CHECK:STDOUT:   %facet_value.loc63_33.12: %facet_type.8a3 = facet_value constants.%tuple.type.df5, (constants.%custom_witness.b80ef6.15, constants.%DestroyImpl.impl_witness.8c3) [concrete = constants.%facet_value.4bc]
// CHECK:STDOUT:   %.loc63_33.23: %facet_type.8a3 = converted constants.%tuple.type.df5, %facet_value.loc63_33.12 [concrete = constants.%facet_value.4bc]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc63_33.4: init %empty_tuple.type = call %SelfDestruct.specific_fn.8(%var.loc63)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestIterate(constants.%R) {
// CHECK:STDOUT:   %R.patt.loc40_17.2 => constants.%R.patt
// CHECK:STDOUT:   %R.loc40_17.1 => constants.%R
// CHECK:STDOUT:   %R.as_type.loc40_79.1 => constants.%R.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.e03
// CHECK:STDOUT:   %r.patt.loc40_77.2 => constants.%r.patt.ac0
// CHECK:STDOUT:   %r.param_patt.loc40_72.2 => constants.%r.param_patt.5e7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestIterate(constants.%facet_value.8db) {
// CHECK:STDOUT:   %R.patt.loc40_17.2 => constants.%R.patt
// CHECK:STDOUT:   %R.loc40_17.1 => constants.%facet_value.8db
// CHECK:STDOUT:   %R.as_type.loc40_79.1 => constants.%DiffMutableRange
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.611
// CHECK:STDOUT:   %r.patt.loc40_77.2 => constants.%r.patt.5fe
// CHECK:STDOUT:   %r.param_patt.loc40_72.2 => constants.%r.param_patt.a4b
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc40 => constants.%complete_type.357
// CHECK:STDOUT:   %Iterate.lookup_impl_witness => constants.%Iterate.impl_witness.706
// CHECK:STDOUT:   %Iterate.facet.loc44_33.5 => constants.%Iterate.facet.9f6
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type => constants.%Iterate.WithSelf.NewCursor.type.e12
// CHECK:STDOUT:   %.loc44_33.26 => constants.%.c87
// CHECK:STDOUT:   %impl.elem2.loc44_33.2 => constants.%T.as_type.as.Iterate.impl.NewCursor.097
// CHECK:STDOUT:   %specific_impl_fn.loc44_33.3 => constants.%T.as_type.as.Iterate.impl.NewCursor.specific_fn.5a0
// CHECK:STDOUT:   %impl.elem1 => constants.%facet_value.fcf
// CHECK:STDOUT:   %as_type => constants.%tuple.type.b18
// CHECK:STDOUT:   %require_complete.1 => constants.%complete_type.bfe
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type => constants.%Iterate.WithSelf.Next.type.aad
// CHECK:STDOUT:   %.loc44_33.27 => constants.%.5db8
// CHECK:STDOUT:   %impl.elem3.loc44_33.2 => constants.%T.as_type.as.Iterate.impl.Next.c89
// CHECK:STDOUT:   %specific_impl_fn.loc44_33.4 => constants.%T.as_type.as.Iterate.impl.Next.specific_fn.134
// CHECK:STDOUT:   %SelfDestruct.specific_fn.6 => constants.%SelfDestruct.specific_fn.1d9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestIterate(constants.%facet_value.eda) {
// CHECK:STDOUT:   %R.patt.loc40_17.2 => constants.%R.patt
// CHECK:STDOUT:   %R.loc40_17.1 => constants.%facet_value.eda
// CHECK:STDOUT:   %R.as_type.loc40_79.1 => constants.%DiffConstRange
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.7dd
// CHECK:STDOUT:   %r.patt.loc40_77.2 => constants.%r.patt.9c0
// CHECK:STDOUT:   %r.param_patt.loc40_72.2 => constants.%r.param_patt.992
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc40 => constants.%complete_type.357
// CHECK:STDOUT:   %Iterate.lookup_impl_witness => constants.%Iterate.impl_witness.df4
// CHECK:STDOUT:   %Iterate.facet.loc44_33.5 => constants.%Iterate.facet.3d3
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type => constants.%Iterate.WithSelf.NewCursor.type.c77
// CHECK:STDOUT:   %.loc44_33.26 => constants.%.afc
// CHECK:STDOUT:   %impl.elem2.loc44_33.2 => constants.%T.as_type.as.Iterate.impl.NewCursor.416
// CHECK:STDOUT:   %specific_impl_fn.loc44_33.3 => constants.%T.as_type.as.Iterate.impl.NewCursor.specific_fn.0cc
// CHECK:STDOUT:   %impl.elem1 => constants.%facet_value.4bc
// CHECK:STDOUT:   %as_type => constants.%tuple.type.df5
// CHECK:STDOUT:   %require_complete.1 => constants.%complete_type.ad6
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type => constants.%Iterate.WithSelf.Next.type.f88
// CHECK:STDOUT:   %.loc44_33.27 => constants.%.5be
// CHECK:STDOUT:   %impl.elem3.loc44_33.2 => constants.%T.as_type.as.Iterate.impl.Next.b68
// CHECK:STDOUT:   %specific_impl_fn.loc44_33.4 => constants.%T.as_type.as.Iterate.impl.Next.specific_fn.4d3
// CHECK:STDOUT:   %SelfDestruct.specific_fn.6 => constants.%SelfDestruct.specific_fn.dea
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
// CHECK:STDOUT:   %Optional.Get.type.786: type = fn_type @Optional.Get, @Optional(%T.220) [symbolic]
// CHECK:STDOUT:   %Optional.Get.2d0: %Optional.Get.type.786 = struct_value () [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.type.3ea: type = fn_type @Optional.HasValue, @Optional(%T.220) [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.f2f: %Optional.HasValue.type.3ea = struct_value () [symbolic]
// CHECK:STDOUT:   %facet_type.8a3: type = facet_type <@SubobjectDestroy & @DestroyImpl> [concrete]
// CHECK:STDOUT:   %T.9ab: %facet_type.d61 = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.f15: type = class_type @DefaultOptionalStorage, @DefaultOptionalStorage(%T.9ab) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Get.type.3e7: type = fn_type @T.as_type.as.OptionalStorage.impl.Get, @T.as_type.as.OptionalStorage.impl(%T.9ab) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Get.f80: %T.as_type.as.OptionalStorage.impl.Get.type.3e7 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Has.type.371: type = fn_type @T.as_type.as.OptionalStorage.impl.Has, @T.as_type.as.OptionalStorage.impl(%T.9ab) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Has.8ce: %T.as_type.as.OptionalStorage.impl.Has.type.371 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Some.type.4eb: type = fn_type @T.as_type.as.OptionalStorage.impl.Some, @T.as_type.as.OptionalStorage.impl(%T.9ab) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Some.44a: %T.as_type.as.OptionalStorage.impl.Some.type.4eb = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.None.type.76a: type = fn_type @T.as_type.as.OptionalStorage.impl.None, @T.as_type.as.OptionalStorage.impl(%T.9ab) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.None.93b: %T.as_type.as.OptionalStorage.impl.None.type.76a = struct_value () [symbolic]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
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
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.2: type = fn_type @SubobjectDestroy.WithSelf.Op.loc25_56 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.2: %SubobjectDestroy.WithSelf.Op.type.6870fd.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.2: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.2), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.8fd: %SubobjectDestroy.type = facet_value %OverloadValueType, (%custom_witness.b80ef6.2) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.3a4: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.8fd) [concrete]
// CHECK:STDOUT:   %OverloadValueType.Op.type: type = fn_type @OverloadValueType.Op [concrete]
// CHECK:STDOUT:   %OverloadValueType.Op: %OverloadValueType.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.d0da43.1: <witness> = custom_witness (%OverloadValueType.Op), @Copy [concrete]
// CHECK:STDOUT:   %facet_value.cad: %facet_type.d61 = facet_value %OverloadValueType, (%custom_witness.b80ef6.2, %DestroyImpl.impl_witness.3a4, %custom_witness.d0da43.1) [concrete]
// CHECK:STDOUT:   %Iterate.lookup_impl_witness.1d8: <witness> = lookup_impl_witness %.Self.500, @Iterate [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.8a4: %facet_type.d61 = impl_witness_access %Iterate.lookup_impl_witness.1d8, element0 [symbolic_self]
// CHECK:STDOUT:   %Iterate_where.type.265: type = facet_type <@Iterate where %impl.elem0.8a4 = %facet_value.cad> [concrete]
// CHECK:STDOUT:   %pattern_type.3d0: type = pattern_type %Iterate_where.type.265 [concrete]
// CHECK:STDOUT:   %R.patt: %pattern_type.3d0 = symbolic_binding_pattern R, 0 [symbolic]
// CHECK:STDOUT:   %R: %Iterate_where.type.265 = symbolic_binding R, 0 [symbolic]
// CHECK:STDOUT:   %R.as_type: type = facet_access_type %R [symbolic]
// CHECK:STDOUT:   %pattern_type.7a2: type = pattern_type %R.as_type [symbolic]
// CHECK:STDOUT:   %r.patt.a30: %pattern_type.7a2 = ref_binding_pattern r [symbolic]
// CHECK:STDOUT:   %r.param_patt.d4d: %pattern_type.7a2 = var_param_pattern %r.patt.a30 [symbolic]
// CHECK:STDOUT:   %_.patt.a10: %pattern_type.804 = value_binding_pattern _ [concrete]
// CHECK:STDOUT:   %Iterate.lookup_impl_witness.cf4: <witness> = lookup_impl_witness %R, @Iterate [symbolic]
// CHECK:STDOUT:   %Iterate.facet.03d: %Iterate.type = facet_value %R.as_type, (%Iterate.lookup_impl_witness.cf4) [symbolic]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.954: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.03d) [symbolic]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.a9f: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.03d) [symbolic]
// CHECK:STDOUT:   %.b2c: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.954, %Iterate.facet.03d [symbolic]
// CHECK:STDOUT:   %impl.elem2.ac0: %.b2c = impl_witness_access %Iterate.lookup_impl_witness.cf4, element2 [symbolic]
// CHECK:STDOUT:   %impl.elem1.65a: %facet_type.8a3 = impl_witness_access %Iterate.lookup_impl_witness.cf4, element1 [symbolic]
// CHECK:STDOUT:   %as_type.d03: type = facet_access_type %impl.elem1.65a [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.a8e: <specific function> = specific_impl_function %impl.elem2.ac0, @Iterate.WithSelf.NewCursor(%Iterate.facet.03d) [symbolic]
// CHECK:STDOUT:   %.d96: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.a9f, %Iterate.facet.03d [symbolic]
// CHECK:STDOUT:   %impl.elem3.ae6: %.d96 = impl_witness_access %Iterate.lookup_impl_witness.cf4, element3 [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.f27: type = class_type @DefaultOptionalStorage, @DefaultOptionalStorage(%facet_value.cad) [concrete]
// CHECK:STDOUT:   %OptionalStorage.impl_witness.20f: <witness> = impl_witness imports.%OptionalStorage.impl_witness_table.968, @T.as_type.as.OptionalStorage.impl(%facet_value.cad) [concrete]
// CHECK:STDOUT:   %OptionalStorage.facet.094: %OptionalStorage.type = facet_value %OverloadValueType, (%OptionalStorage.impl_witness.20f) [concrete]
// CHECK:STDOUT:   %Optional.9dc: type = class_type @Optional, @Optional(%OptionalStorage.facet.094) [concrete]
// CHECK:STDOUT:   %specific_impl_fn.187: <specific function> = specific_impl_function %impl.elem3.ae6, @Iterate.WithSelf.Next(%Iterate.facet.03d) [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.type.ac3: type = fn_type @Optional.HasValue, @Optional(%OptionalStorage.facet.094) [concrete]
// CHECK:STDOUT:   %Optional.HasValue.ba1: %Optional.HasValue.type.ac3 = struct_value () [concrete]
// CHECK:STDOUT:   %Optional.Get.type.77c: type = fn_type @Optional.Get, @Optional(%OptionalStorage.facet.094) [concrete]
// CHECK:STDOUT:   %Optional.Get.322: %Optional.Get.type.77c = struct_value () [concrete]
// CHECK:STDOUT:   %facet_value.46d: %facet_type.8a3 = facet_value %OverloadValueType, (%custom_witness.b80ef6.2, %DestroyImpl.impl_witness.3a4) [concrete]
// CHECK:STDOUT:   %MaybeUnformed.261: type = class_type @MaybeUnformed, @MaybeUnformed(%facet_value.46d) [concrete]
// CHECK:STDOUT:   %.8a2: type = maybe_unformed_type %OverloadValueType [concrete]
// CHECK:STDOUT:   %struct_type.value.has_value.086: type = struct_type {.value: %MaybeUnformed.261, .has_value: bool} [concrete]
// CHECK:STDOUT:   %Optional.HasValue.specific_fn.4c3: <specific function> = specific_function %Optional.HasValue.ba1, @Optional.HasValue(%OptionalStorage.facet.094) [concrete]
// CHECK:STDOUT:   %Optional.Get.specific_fn.816: <specific function> = specific_function %Optional.Get.322, @Optional.Get(%OptionalStorage.facet.094) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.8: type = fn_type @SubobjectDestroy.WithSelf.Op.loc27_37.6 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.8: %SubobjectDestroy.WithSelf.Op.type.6870fd.8 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.8: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.8), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.d94: %SubobjectDestroy.type = facet_value %Optional.9dc, (%custom_witness.b80ef6.8) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.2d4: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.d94) [concrete]
// CHECK:STDOUT:   %facet_value.d59: %facet_type.8a3 = facet_value %Optional.9dc, (%custom_witness.b80ef6.8, %DestroyImpl.impl_witness.2d4) [concrete]
// CHECK:STDOUT:   %Range: type = class_type @Range [concrete]
// CHECK:STDOUT:   %pattern_type.23f: type = pattern_type %Range [concrete]
// CHECK:STDOUT:   %Iterator: type = class_type @Iterator.1 [concrete]
// CHECK:STDOUT:   %Range.Begin.type: type = fn_type @Range.Begin [concrete]
// CHECK:STDOUT:   %Range.Begin: %Range.Begin.type = struct_value () [concrete]
// CHECK:STDOUT:   %Range.End.type: type = fn_type @Range.End [concrete]
// CHECK:STDOUT:   %Range.End: %Range.End.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.090: <witness> = custom_witness (%Iterator, %Iterator, %Range.Begin, %Range.End), @CppRangeForIterate [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.9: type = fn_type @SubobjectDestroy.WithSelf.Op.loc32 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.9: %SubobjectDestroy.WithSelf.Op.type.6870fd.9 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.9: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.9), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.8fc: %SubobjectDestroy.type = facet_value %Iterator, (%custom_witness.b80ef6.9) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.ad1: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.8fc) [concrete]
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
// CHECK:STDOUT:   %facet_value.053: %CppRange_where.type.d682d6.1 = facet_value %Range, (%custom_witness.090, %custom_witness.b80ef6.9, %DestroyImpl.impl_witness.ad1, %custom_witness.6bd, %custom_witness.716, %custom_witness.f93, %custom_witness.b80ef6.2, %DestroyImpl.impl_witness.3a4, %custom_witness.d0da43.1, %custom_witness.b80ef6.9, %DestroyImpl.impl_witness.ad1) [concrete]
// CHECK:STDOUT:   %tuple.type.9ef: type = tuple_type (%Iterator, %Iterator) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.10: type = fn_type @SubobjectDestroy.WithSelf.Op.3 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.10: %SubobjectDestroy.WithSelf.Op.type.6870fd.10 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.10: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.10), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.43f: %SubobjectDestroy.type = facet_value %tuple.type.9ef, (%custom_witness.b80ef6.10) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.c0f: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.43f) [concrete]
// CHECK:STDOUT:   %complete_type.e35: <witness> = complete_type_witness %tuple.type.9ef [concrete]
// CHECK:STDOUT:   %facet_value.6b4: %facet_type.8a3 = facet_value %tuple.type.9ef, (%custom_witness.b80ef6.10, %DestroyImpl.impl_witness.c0f) [concrete]
// CHECK:STDOUT:   %Iterate.impl_witness.8a1: <witness> = impl_witness imports.%Iterate.impl_witness_table.121, @T.as_type.as.Iterate.impl(%facet_value.053) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.type.398: type = fn_type @T.as_type.as.Iterate.impl.NewCursor, @T.as_type.as.Iterate.impl(%facet_value.053) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.bdb: %T.as_type.as.Iterate.impl.NewCursor.type.398 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.type.565: type = fn_type @T.as_type.as.Iterate.impl.Next, @T.as_type.as.Iterate.impl(%facet_value.053) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.cb2: %T.as_type.as.Iterate.impl.Next.type.565 = struct_value () [concrete]
// CHECK:STDOUT:   %Iterate.facet.cdb: %Iterate.type = facet_value %Range, (%Iterate.impl_witness.8a1) [concrete]
// CHECK:STDOUT:   %facet_value.728: %Iterate_where.type.265 = facet_value %Range, (%Iterate.impl_witness.8a1) [concrete]
// CHECK:STDOUT:   %r.patt.b9f: %pattern_type.23f = ref_binding_pattern r [concrete]
// CHECK:STDOUT:   %r.param_patt.466: %pattern_type.23f = var_param_pattern %r.patt.b9f [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.012: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.cdb) [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.b7e: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.cdb) [concrete]
// CHECK:STDOUT:   %.11f: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.012, %Iterate.facet.cdb [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.specific_fn: <specific function> = specific_function %T.as_type.as.Iterate.impl.NewCursor.bdb, @T.as_type.as.Iterate.impl.NewCursor(%facet_value.053) [concrete]
// CHECK:STDOUT:   %.e6b: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.b7e, %Iterate.facet.cdb [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.specific_fn: <specific function> = specific_function %T.as_type.as.Iterate.impl.Next.cb2, @T.as_type.as.Iterate.impl.Next(%facet_value.053) [concrete]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.60c: <specific function> = specific_function %SelfDestruct, @SelfDestruct(%facet_value.6b4) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .OverloadValueType = %OverloadValueType.decl
// CHECK:STDOUT:     .Range = %Range.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.c82: @Optional.%Optional.HasValue.type (%Optional.HasValue.type.3ea) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.HasValue (constants.%Optional.HasValue.f2f)]
// CHECK:STDOUT:   %Core.import_ref.29e: @Optional.%Optional.Get.type (%Optional.Get.type.786) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.Get (constants.%Optional.Get.2d0)]
// CHECK:STDOUT:   %Core.import_ref.9a31: type = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%DefaultOptionalStorage (constants.%DefaultOptionalStorage.f15)]
// CHECK:STDOUT:   %Core.import_ref.819a: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.None.type (%T.as_type.as.OptionalStorage.impl.None.type.76a) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.None (constants.%T.as_type.as.OptionalStorage.impl.None.93b)]
// CHECK:STDOUT:   %Core.import_ref.14a: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Some.type (%T.as_type.as.OptionalStorage.impl.Some.type.4eb) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Some (constants.%T.as_type.as.OptionalStorage.impl.Some.44a)]
// CHECK:STDOUT:   %Core.import_ref.242: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Has.type (%T.as_type.as.OptionalStorage.impl.Has.type.371) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Has (constants.%T.as_type.as.OptionalStorage.impl.Has.8ce)]
// CHECK:STDOUT:   %Core.import_ref.a54: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Get.type (%T.as_type.as.OptionalStorage.impl.Get.type.3e7) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Get (constants.%T.as_type.as.OptionalStorage.impl.Get.f80)]
// CHECK:STDOUT:   %Core.import_ref.5bf = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], unloaded
// CHECK:STDOUT:   %OptionalStorage.impl_witness_table.968 = impl_witness_table (%Core.import_ref.9a31, %Core.import_ref.819a, %Core.import_ref.14a, %Core.import_ref.242, %Core.import_ref.a54, %Core.import_ref.5bf), @T.as_type.as.OptionalStorage.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.822: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.822), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.d6c: %facet_type.d61 = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.Iterate.impl.%facet_value.1 (constants.%facet_value.9da)]
// CHECK:STDOUT:   %Core.import_ref.2ac: %facet_type.8a3 = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.Iterate.impl.%facet_value.2 (constants.%facet_value.bf6)]
// CHECK:STDOUT:   %Core.import_ref.946: @T.as_type.as.Iterate.impl.%T.as_type.as.Iterate.impl.NewCursor.type (%T.as_type.as.Iterate.impl.NewCursor.type.8fb) = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.Iterate.impl.%T.as_type.as.Iterate.impl.NewCursor (constants.%T.as_type.as.Iterate.impl.NewCursor.8a1)]
// CHECK:STDOUT:   %Core.import_ref.cfb: @T.as_type.as.Iterate.impl.%T.as_type.as.Iterate.impl.Next.type (%T.as_type.as.Iterate.impl.Next.type.d4a) = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.Iterate.impl.%T.as_type.as.Iterate.impl.Next (constants.%T.as_type.as.Iterate.impl.Next.5a8)]
// CHECK:STDOUT:   %Iterate.impl_witness_table.121 = impl_witness_table (%Core.import_ref.d6c, %Core.import_ref.2ac, %Core.import_ref.946, %Core.import_ref.cfb), @T.as_type.as.Iterate.impl [concrete]
// CHECK:STDOUT:   %OverloadValueType.decl: type = class_decl @OverloadValueType [concrete = constants.%OverloadValueType] {} {}
// CHECK:STDOUT:   %Range.decl: type = class_decl @Range [concrete = constants.%Range] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @TestIterate(%R.loc25_17.2: %Iterate_where.type.265) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Iterate.lookup_impl_witness: <witness> = lookup_impl_witness %R.loc25_17.1, @Iterate [symbolic = %Iterate.lookup_impl_witness (constants.%Iterate.lookup_impl_witness.cf4)]
// CHECK:STDOUT:   %Iterate.facet.loc27_37.5: %Iterate.type = facet_value %R.as_type.loc25_83.1, (%Iterate.lookup_impl_witness) [symbolic = %Iterate.facet.loc27_37.5 (constants.%Iterate.facet.03d)]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.loc27_37.5) [symbolic = %Iterate.WithSelf.NewCursor.type (constants.%Iterate.WithSelf.NewCursor.type.954)]
// CHECK:STDOUT:   %.loc27_37.26: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type, %Iterate.facet.loc27_37.5 [symbolic = %.loc27_37.26 (constants.%.b2c)]
// CHECK:STDOUT:   %impl.elem2.loc27_37.2: @TestIterate.%.loc27_37.26 (%.b2c) = impl_witness_access %Iterate.lookup_impl_witness, element2 [symbolic = %impl.elem2.loc27_37.2 (constants.%impl.elem2.ac0)]
// CHECK:STDOUT:   %specific_impl_fn.loc27_37.3: <specific function> = specific_impl_function %impl.elem2.loc27_37.2, @Iterate.WithSelf.NewCursor(%Iterate.facet.loc27_37.5) [symbolic = %specific_impl_fn.loc27_37.3 (constants.%specific_impl_fn.a8e)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.loc27_37.5) [symbolic = %Iterate.WithSelf.Next.type (constants.%Iterate.WithSelf.Next.type.a9f)]
// CHECK:STDOUT:   %.loc27_37.27: type = fn_type_with_self_type %Iterate.WithSelf.Next.type, %Iterate.facet.loc27_37.5 [symbolic = %.loc27_37.27 (constants.%.d96)]
// CHECK:STDOUT:   %impl.elem3.loc27_37.2: @TestIterate.%.loc27_37.27 (%.d96) = impl_witness_access %Iterate.lookup_impl_witness, element3 [symbolic = %impl.elem3.loc27_37.2 (constants.%impl.elem3.ae6)]
// CHECK:STDOUT:   %specific_impl_fn.loc27_37.4: <specific function> = specific_impl_function %impl.elem3.loc27_37.2, @Iterate.WithSelf.Next(%Iterate.facet.loc27_37.5) [symbolic = %specific_impl_fn.loc27_37.4 (constants.%specific_impl_fn.187)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%r.param: ref @TestIterate.%R.as_type.loc25_83.1 (%R.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %_.patt: %pattern_type.804 = value_binding_pattern _ [concrete = constants.%_.patt.a10]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %r.ref: ref @TestIterate.%R.as_type.loc25_83.1 (%R.as_type) = name_ref r, %r
// CHECK:STDOUT:     %impl.elem2.loc27_37.1: @TestIterate.%.loc27_37.26 (%.b2c) = impl_witness_access constants.%Iterate.lookup_impl_witness.cf4, element2 [symbolic = %impl.elem2.loc27_37.2 (constants.%impl.elem2.ac0)]
// CHECK:STDOUT:     %bound_method.loc27_37.1: <bound method> = bound_method %r.ref, %impl.elem2.loc27_37.1
// CHECK:STDOUT:     %Iterate.facet.loc27_37.1: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.cf4) [symbolic = %Iterate.facet.loc27_37.5 (constants.%Iterate.facet.03d)]
// CHECK:STDOUT:     %.loc27_37.1: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc27_37.1 [symbolic = %Iterate.facet.loc27_37.5 (constants.%Iterate.facet.03d)]
// CHECK:STDOUT:     %Iterate.facet.loc27_37.2: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.cf4) [symbolic = %Iterate.facet.loc27_37.5 (constants.%Iterate.facet.03d)]
// CHECK:STDOUT:     %.loc27_37.2: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc27_37.2 [symbolic = %Iterate.facet.loc27_37.5 (constants.%Iterate.facet.03d)]
// CHECK:STDOUT:     %specific_impl_fn.loc27_37.1: <specific function> = specific_impl_function %impl.elem2.loc27_37.1, @Iterate.WithSelf.NewCursor(constants.%Iterate.facet.03d) [symbolic = %specific_impl_fn.loc27_37.3 (constants.%specific_impl_fn.a8e)]
// CHECK:STDOUT:     %bound_method.loc27_37.2: <bound method> = bound_method %r.ref, %specific_impl_fn.loc27_37.1
// CHECK:STDOUT:     %var: ref @TestIterate.%as_type (%as_type.d03) = var_storage invalid
// CHECK:STDOUT:     %.loc27_36.1: @TestIterate.%R.as_type.loc25_83.1 (%R.as_type) = acquire_value %r.ref
// CHECK:STDOUT:     %Iterate.WithSelf.NewCursor.call: init @TestIterate.%as_type (%as_type.d03) to %var = call %bound_method.loc27_37.2(%.loc27_36.1)
// CHECK:STDOUT:     assign %var, %Iterate.WithSelf.NewCursor.call
// CHECK:STDOUT:     br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.next:
// CHECK:STDOUT:     %impl.elem3.loc27_37.1: @TestIterate.%.loc27_37.27 (%.d96) = impl_witness_access constants.%Iterate.lookup_impl_witness.cf4, element3 [symbolic = %impl.elem3.loc27_37.2 (constants.%impl.elem3.ae6)]
// CHECK:STDOUT:     %bound_method.loc27_37.3: <bound method> = bound_method %r.ref, %impl.elem3.loc27_37.1
// CHECK:STDOUT:     %Iterate.facet.loc27_37.3: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.cf4) [symbolic = %Iterate.facet.loc27_37.5 (constants.%Iterate.facet.03d)]
// CHECK:STDOUT:     %.loc27_37.3: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc27_37.3 [symbolic = %Iterate.facet.loc27_37.5 (constants.%Iterate.facet.03d)]
// CHECK:STDOUT:     %Iterate.facet.loc27_37.4: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.cf4) [symbolic = %Iterate.facet.loc27_37.5 (constants.%Iterate.facet.03d)]
// CHECK:STDOUT:     %.loc27_37.4: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc27_37.4 [symbolic = %Iterate.facet.loc27_37.5 (constants.%Iterate.facet.03d)]
// CHECK:STDOUT:     %.loc27_37.5: %facet_type.8a3 = converted constants.%as_type.d03, constants.%impl.elem1.65a [symbolic = %impl.elem1 (constants.%impl.elem1.65a)]
// CHECK:STDOUT:     %.loc27_37.6: %facet_type.8a3 = converted constants.%as_type.d03, constants.%impl.elem1.65a [symbolic = %impl.elem1 (constants.%impl.elem1.65a)]
// CHECK:STDOUT:     %specific_impl_fn.loc27_37.2: <specific function> = specific_impl_function %impl.elem3.loc27_37.1, @Iterate.WithSelf.Next(constants.%Iterate.facet.03d) [symbolic = %specific_impl_fn.loc27_37.4 (constants.%specific_impl_fn.187)]
// CHECK:STDOUT:     %bound_method.loc27_37.4: <bound method> = bound_method %r.ref, %specific_impl_fn.loc27_37.2
// CHECK:STDOUT:     %.loc27_37.7: ref %Optional.9dc = temporary_storage
// CHECK:STDOUT:     %.loc27_36.2: @TestIterate.%R.as_type.loc25_83.1 (%R.as_type) = acquire_value %r.ref
// CHECK:STDOUT:     %Iterate.WithSelf.Next.call: init %Optional.9dc to %.loc27_37.7 = call %bound_method.loc27_37.4(%.loc27_36.2, %var)
// CHECK:STDOUT:     %.loc27_37.8: ref %Optional.9dc = temporary %.loc27_37.7, %Iterate.WithSelf.Next.call
// CHECK:STDOUT:     %.loc27_37.9: %Optional.HasValue.type.ac3 = specific_constant imports.%Core.import_ref.c82, @Optional(constants.%OptionalStorage.facet.094) [concrete = constants.%Optional.HasValue.ba1]
// CHECK:STDOUT:     %HasValue.ref: %Optional.HasValue.type.ac3 = name_ref HasValue, %.loc27_37.9 [concrete = constants.%Optional.HasValue.ba1]
// CHECK:STDOUT:     %Optional.HasValue.bound: <bound method> = bound_method %.loc27_37.8, %HasValue.ref
// CHECK:STDOUT:     %Optional.HasValue.specific_fn: <specific function> = specific_function %HasValue.ref, @Optional.HasValue(constants.%OptionalStorage.facet.094) [concrete = constants.%Optional.HasValue.specific_fn.4c3]
// CHECK:STDOUT:     %bound_method.loc27_37.5: <bound method> = bound_method %.loc27_37.8, %Optional.HasValue.specific_fn
// CHECK:STDOUT:     %.loc27_37.10: %Optional.9dc = acquire_value %.loc27_37.8
// CHECK:STDOUT:     %Optional.HasValue.call: init bool = call %bound_method.loc27_37.5(%.loc27_37.10)
// CHECK:STDOUT:     %.loc27_37.11: bool = value_of_initializer %Optional.HasValue.call
// CHECK:STDOUT:     %.loc27_37.12: bool = converted %Optional.HasValue.call, %.loc27_37.11
// CHECK:STDOUT:     if %.loc27_37.12 br !for.body else br !for.done
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.body:
// CHECK:STDOUT:     %.loc27_37.13: %Optional.Get.type.77c = specific_constant imports.%Core.import_ref.29e, @Optional(constants.%OptionalStorage.facet.094) [concrete = constants.%Optional.Get.322]
// CHECK:STDOUT:     %Get.ref: %Optional.Get.type.77c = name_ref Get, %.loc27_37.13 [concrete = constants.%Optional.Get.322]
// CHECK:STDOUT:     %Optional.Get.bound: <bound method> = bound_method %.loc27_37.8, %Get.ref
// CHECK:STDOUT:     %Optional.Get.specific_fn: <specific function> = specific_function %Get.ref, @Optional.Get(constants.%OptionalStorage.facet.094) [concrete = constants.%Optional.Get.specific_fn.816]
// CHECK:STDOUT:     %bound_method.loc27_37.6: <bound method> = bound_method %.loc27_37.8, %Optional.Get.specific_fn
// CHECK:STDOUT:     %.loc27_37.14: ref %OverloadValueType = temporary_storage
// CHECK:STDOUT:     %.loc27_37.15: %Optional.9dc = acquire_value %.loc27_37.8
// CHECK:STDOUT:     %Optional.Get.call: init %OverloadValueType to %.loc27_37.14 = call %bound_method.loc27_37.6(%.loc27_37.15)
// CHECK:STDOUT:     %.loc27_37.16: ref %OverloadValueType = temporary %.loc27_37.14, %Optional.Get.call
// CHECK:STDOUT:     %.loc27_37.17: %OverloadValueType = acquire_value %.loc27_37.16
// CHECK:STDOUT:     %.loc27_14: type = splice_block %OverloadValueType.ref.loc27 [concrete = constants.%OverloadValueType] {
// CHECK:STDOUT:       %Cpp.ref.loc27: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:       %OverloadValueType.ref.loc27: type = name_ref OverloadValueType, imports.%OverloadValueType.decl [concrete = constants.%OverloadValueType]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %_: %OverloadValueType = wrapper_binding _, %.loc27_37.17
// CHECK:STDOUT:     %facet_value.loc27_37.1: %facet_type.8a3 = facet_value constants.%OverloadValueType, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.3a4) [concrete = constants.%facet_value.46d]
// CHECK:STDOUT:     %.loc27_37.18: %facet_type.8a3 = converted constants.%OverloadValueType, %facet_value.loc27_37.1 [concrete = constants.%facet_value.46d]
// CHECK:STDOUT:     %facet_value.loc27_37.2: %facet_type.8a3 = facet_value constants.%OverloadValueType, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.3a4) [concrete = constants.%facet_value.46d]
// CHECK:STDOUT:     %.loc27_37.19: %facet_type.8a3 = converted constants.%OverloadValueType, %facet_value.loc27_37.2 [concrete = constants.%facet_value.46d]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %SelfDestruct.call.loc27_37.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc27_37.16)
// CHECK:STDOUT:     %facet_value.loc27_37.3: %facet_type.8a3 = facet_value constants.%Optional.9dc, (constants.%custom_witness.b80ef6.8, constants.%DestroyImpl.impl_witness.2d4) [concrete = constants.%facet_value.d59]
// CHECK:STDOUT:     %.loc27_37.20: %facet_type.8a3 = converted constants.%Optional.9dc, %facet_value.loc27_37.3 [concrete = constants.%facet_value.d59]
// CHECK:STDOUT:     %facet_value.loc27_37.4: %facet_type.8a3 = facet_value constants.%Optional.9dc, (constants.%custom_witness.b80ef6.8, constants.%DestroyImpl.impl_witness.2d4) [concrete = constants.%facet_value.d59]
// CHECK:STDOUT:     %.loc27_37.21: %facet_type.8a3 = converted constants.%Optional.9dc, %facet_value.loc27_37.4 [concrete = constants.%facet_value.d59]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %SelfDestruct.call.loc27_37.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc27_37.8)
// CHECK:STDOUT:     br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.done:
// CHECK:STDOUT:     %facet_value.loc27_37.5: %facet_type.8a3 = facet_value constants.%Optional.9dc, (constants.%custom_witness.b80ef6.8, constants.%DestroyImpl.impl_witness.2d4) [concrete = constants.%facet_value.d59]
// CHECK:STDOUT:     %.loc27_37.22: %facet_type.8a3 = converted constants.%Optional.9dc, %facet_value.loc27_37.5 [concrete = constants.%facet_value.d59]
// CHECK:STDOUT:     %facet_value.loc27_37.6: %facet_type.8a3 = facet_value constants.%Optional.9dc, (constants.%custom_witness.b80ef6.8, constants.%DestroyImpl.impl_witness.2d4) [concrete = constants.%facet_value.d59]
// CHECK:STDOUT:     %.loc27_37.23: %facet_type.8a3 = converted constants.%Optional.9dc, %facet_value.loc27_37.6 [concrete = constants.%facet_value.d59]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %SelfDestruct.call.loc27_37.3: init %empty_tuple.type = call %SelfDestruct.specific_fn.3(%.loc27_37.8)
// CHECK:STDOUT:     %.loc27_37.24: %facet_type.8a3 = converted constants.%as_type.d03, constants.%impl.elem1.65a [symbolic = %impl.elem1 (constants.%impl.elem1.65a)]
// CHECK:STDOUT:     %.loc27_37.25: %facet_type.8a3 = converted constants.%as_type.d03, constants.%impl.elem1.65a [symbolic = %impl.elem1 (constants.%impl.elem1.65a)]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %SelfDestruct.call.loc27_37.4: init %empty_tuple.type = call %SelfDestruct.specific_fn.4(%var)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc27_37.1(%self.param: ref %.8a2) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc27_37.2(%self.param: ref %MaybeUnformed.261) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc27_37.3(%self.param: ref bool) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc27_37.4(%self.param: ref %struct_type.value.has_value.086) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc27_37.5(%self.param: ref %DefaultOptionalStorage.f27) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc27_37.6(%self.param: ref %Optional.9dc) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestRangeFor(%r.param: ref %Range) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %_.patt: %pattern_type.804 = value_binding_pattern _ [concrete = constants.%_.patt.a10]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %r.ref: ref %Range = name_ref r, %r
// CHECK:STDOUT:   %impl.elem2: %.11f = impl_witness_access constants.%Iterate.impl_witness.8a1, element2 [concrete = constants.%T.as_type.as.Iterate.impl.NewCursor.bdb]
// CHECK:STDOUT:   %bound_method.loc37_37.1: <bound method> = bound_method %r.ref, %impl.elem2
// CHECK:STDOUT:   %facet_value.loc37_37.1: %CppRange_where.type.d682d6.1 = facet_value constants.%Range, (constants.%custom_witness.090, constants.%custom_witness.b80ef6.9, constants.%DestroyImpl.impl_witness.ad1, constants.%custom_witness.6bd, constants.%custom_witness.716, constants.%custom_witness.f93, constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.3a4, constants.%custom_witness.d0da43.1, constants.%custom_witness.b80ef6.9, constants.%DestroyImpl.impl_witness.ad1) [concrete = constants.%facet_value.053]
// CHECK:STDOUT:   %.loc37_37.1: %CppRange_where.type.d682d6.1 = converted constants.%Range, %facet_value.loc37_37.1 [concrete = constants.%facet_value.053]
// CHECK:STDOUT:   %facet_value.loc37_37.2: %CppRange_where.type.d682d6.1 = facet_value constants.%Range, (constants.%custom_witness.090, constants.%custom_witness.b80ef6.9, constants.%DestroyImpl.impl_witness.ad1, constants.%custom_witness.6bd, constants.%custom_witness.716, constants.%custom_witness.f93, constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.3a4, constants.%custom_witness.d0da43.1, constants.%custom_witness.b80ef6.9, constants.%DestroyImpl.impl_witness.ad1) [concrete = constants.%facet_value.053]
// CHECK:STDOUT:   %.loc37_37.2: %CppRange_where.type.d682d6.1 = converted constants.%Range, %facet_value.loc37_37.2 [concrete = constants.%facet_value.053]
// CHECK:STDOUT:   %specific_fn.loc37_37.1: <specific function> = specific_function %impl.elem2, @T.as_type.as.Iterate.impl.NewCursor(constants.%facet_value.053) [concrete = constants.%T.as_type.as.Iterate.impl.NewCursor.specific_fn]
// CHECK:STDOUT:   %bound_method.loc37_37.2: <bound method> = bound_method %r.ref, %specific_fn.loc37_37.1
// CHECK:STDOUT:   %var: ref %tuple.type.9ef = var_storage invalid
// CHECK:STDOUT:   %.loc37_36.1: %Range = acquire_value %r.ref
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.call: init %tuple.type.9ef to %var = call %bound_method.loc37_37.2(%.loc37_36.1)
// CHECK:STDOUT:   assign %var, %T.as_type.as.Iterate.impl.NewCursor.call
// CHECK:STDOUT:   br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT: !for.next:
// CHECK:STDOUT:   %impl.elem3: %.e6b = impl_witness_access constants.%Iterate.impl_witness.8a1, element3 [concrete = constants.%T.as_type.as.Iterate.impl.Next.cb2]
// CHECK:STDOUT:   %bound_method.loc37_37.3: <bound method> = bound_method %r.ref, %impl.elem3
// CHECK:STDOUT:   %facet_value.loc37_37.3: %CppRange_where.type.d682d6.1 = facet_value constants.%Range, (constants.%custom_witness.090, constants.%custom_witness.b80ef6.9, constants.%DestroyImpl.impl_witness.ad1, constants.%custom_witness.6bd, constants.%custom_witness.716, constants.%custom_witness.f93, constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.3a4, constants.%custom_witness.d0da43.1, constants.%custom_witness.b80ef6.9, constants.%DestroyImpl.impl_witness.ad1) [concrete = constants.%facet_value.053]
// CHECK:STDOUT:   %.loc37_37.3: %CppRange_where.type.d682d6.1 = converted constants.%Range, %facet_value.loc37_37.3 [concrete = constants.%facet_value.053]
// CHECK:STDOUT:   %facet_value.loc37_37.4: %CppRange_where.type.d682d6.1 = facet_value constants.%Range, (constants.%custom_witness.090, constants.%custom_witness.b80ef6.9, constants.%DestroyImpl.impl_witness.ad1, constants.%custom_witness.6bd, constants.%custom_witness.716, constants.%custom_witness.f93, constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.3a4, constants.%custom_witness.d0da43.1, constants.%custom_witness.b80ef6.9, constants.%DestroyImpl.impl_witness.ad1) [concrete = constants.%facet_value.053]
// CHECK:STDOUT:   %.loc37_37.4: %CppRange_where.type.d682d6.1 = converted constants.%Range, %facet_value.loc37_37.4 [concrete = constants.%facet_value.053]
// CHECK:STDOUT:   %specific_fn.loc37_37.2: <specific function> = specific_function %impl.elem3, @T.as_type.as.Iterate.impl.Next(constants.%facet_value.053) [concrete = constants.%T.as_type.as.Iterate.impl.Next.specific_fn]
// CHECK:STDOUT:   %bound_method.loc37_37.4: <bound method> = bound_method %r.ref, %specific_fn.loc37_37.2
// CHECK:STDOUT:   %.loc37_37.5: ref %Optional.9dc = temporary_storage
// CHECK:STDOUT:   %.loc37_36.2: %Range = acquire_value %r.ref
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.call: init %Optional.9dc to %.loc37_37.5 = call %bound_method.loc37_37.4(%.loc37_36.2, %var)
// CHECK:STDOUT:   %.loc37_37.6: ref %Optional.9dc = temporary %.loc37_37.5, %T.as_type.as.Iterate.impl.Next.call
// CHECK:STDOUT:   %.loc37_37.7: %Optional.HasValue.type.ac3 = specific_constant imports.%Core.import_ref.c82, @Optional(constants.%OptionalStorage.facet.094) [concrete = constants.%Optional.HasValue.ba1]
// CHECK:STDOUT:   %HasValue.ref: %Optional.HasValue.type.ac3 = name_ref HasValue, %.loc37_37.7 [concrete = constants.%Optional.HasValue.ba1]
// CHECK:STDOUT:   %Optional.HasValue.bound: <bound method> = bound_method %.loc37_37.6, %HasValue.ref
// CHECK:STDOUT:   %Optional.HasValue.specific_fn: <specific function> = specific_function %HasValue.ref, @Optional.HasValue(constants.%OptionalStorage.facet.094) [concrete = constants.%Optional.HasValue.specific_fn.4c3]
// CHECK:STDOUT:   %bound_method.loc37_37.5: <bound method> = bound_method %.loc37_37.6, %Optional.HasValue.specific_fn
// CHECK:STDOUT:   %.loc37_37.8: %Optional.9dc = acquire_value %.loc37_37.6
// CHECK:STDOUT:   %Optional.HasValue.call: init bool = call %bound_method.loc37_37.5(%.loc37_37.8)
// CHECK:STDOUT:   %.loc37_37.9: bool = value_of_initializer %Optional.HasValue.call
// CHECK:STDOUT:   %.loc37_37.10: bool = converted %Optional.HasValue.call, %.loc37_37.9
// CHECK:STDOUT:   if %.loc37_37.10 br !for.body else br !for.done
// CHECK:STDOUT:
// CHECK:STDOUT: !for.body:
// CHECK:STDOUT:   %.loc37_37.11: %Optional.Get.type.77c = specific_constant imports.%Core.import_ref.29e, @Optional(constants.%OptionalStorage.facet.094) [concrete = constants.%Optional.Get.322]
// CHECK:STDOUT:   %Get.ref: %Optional.Get.type.77c = name_ref Get, %.loc37_37.11 [concrete = constants.%Optional.Get.322]
// CHECK:STDOUT:   %Optional.Get.bound: <bound method> = bound_method %.loc37_37.6, %Get.ref
// CHECK:STDOUT:   %Optional.Get.specific_fn: <specific function> = specific_function %Get.ref, @Optional.Get(constants.%OptionalStorage.facet.094) [concrete = constants.%Optional.Get.specific_fn.816]
// CHECK:STDOUT:   %bound_method.loc37_37.6: <bound method> = bound_method %.loc37_37.6, %Optional.Get.specific_fn
// CHECK:STDOUT:   %.loc37_37.12: ref %OverloadValueType = temporary_storage
// CHECK:STDOUT:   %.loc37_37.13: %Optional.9dc = acquire_value %.loc37_37.6
// CHECK:STDOUT:   %Optional.Get.call: init %OverloadValueType to %.loc37_37.12 = call %bound_method.loc37_37.6(%.loc37_37.13)
// CHECK:STDOUT:   %.loc37_37.14: ref %OverloadValueType = temporary %.loc37_37.12, %Optional.Get.call
// CHECK:STDOUT:   %.loc37_37.15: %OverloadValueType = acquire_value %.loc37_37.14
// CHECK:STDOUT:   %.loc37_14: type = splice_block %OverloadValueType.ref [concrete = constants.%OverloadValueType] {
// CHECK:STDOUT:     %Cpp.ref.loc37: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %OverloadValueType.ref: type = name_ref OverloadValueType, imports.%OverloadValueType.decl [concrete = constants.%OverloadValueType]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %_: %OverloadValueType = wrapper_binding _, %.loc37_37.15
// CHECK:STDOUT:   %facet_value.loc37_37.5: %facet_type.8a3 = facet_value constants.%OverloadValueType, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.3a4) [concrete = constants.%facet_value.46d]
// CHECK:STDOUT:   %.loc37_37.16: %facet_type.8a3 = converted constants.%OverloadValueType, %facet_value.loc37_37.5 [concrete = constants.%facet_value.46d]
// CHECK:STDOUT:   %facet_value.loc37_37.6: %facet_type.8a3 = facet_value constants.%OverloadValueType, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.3a4) [concrete = constants.%facet_value.46d]
// CHECK:STDOUT:   %.loc37_37.17: %facet_type.8a3 = converted constants.%OverloadValueType, %facet_value.loc37_37.6 [concrete = constants.%facet_value.46d]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc37_37.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc37_37.14)
// CHECK:STDOUT:   %facet_value.loc37_37.7: %facet_type.8a3 = facet_value constants.%Optional.9dc, (constants.%custom_witness.b80ef6.8, constants.%DestroyImpl.impl_witness.2d4) [concrete = constants.%facet_value.d59]
// CHECK:STDOUT:   %.loc37_37.18: %facet_type.8a3 = converted constants.%Optional.9dc, %facet_value.loc37_37.7 [concrete = constants.%facet_value.d59]
// CHECK:STDOUT:   %facet_value.loc37_37.8: %facet_type.8a3 = facet_value constants.%Optional.9dc, (constants.%custom_witness.b80ef6.8, constants.%DestroyImpl.impl_witness.2d4) [concrete = constants.%facet_value.d59]
// CHECK:STDOUT:   %.loc37_37.19: %facet_type.8a3 = converted constants.%Optional.9dc, %facet_value.loc37_37.8 [concrete = constants.%facet_value.d59]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc37_37.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc37_37.6)
// CHECK:STDOUT:   br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT: !for.done:
// CHECK:STDOUT:   %facet_value.loc37_37.9: %facet_type.8a3 = facet_value constants.%Optional.9dc, (constants.%custom_witness.b80ef6.8, constants.%DestroyImpl.impl_witness.2d4) [concrete = constants.%facet_value.d59]
// CHECK:STDOUT:   %.loc37_37.20: %facet_type.8a3 = converted constants.%Optional.9dc, %facet_value.loc37_37.9 [concrete = constants.%facet_value.d59]
// CHECK:STDOUT:   %facet_value.loc37_37.10: %facet_type.8a3 = facet_value constants.%Optional.9dc, (constants.%custom_witness.b80ef6.8, constants.%DestroyImpl.impl_witness.2d4) [concrete = constants.%facet_value.d59]
// CHECK:STDOUT:   %.loc37_37.21: %facet_type.8a3 = converted constants.%Optional.9dc, %facet_value.loc37_37.10 [concrete = constants.%facet_value.d59]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc37_37.3: init %empty_tuple.type = call %SelfDestruct.specific_fn.3(%.loc37_37.6)
// CHECK:STDOUT:   %facet_value.loc37_37.11: %facet_type.8a3 = facet_value constants.%tuple.type.9ef, (constants.%custom_witness.b80ef6.10, constants.%DestroyImpl.impl_witness.c0f) [concrete = constants.%facet_value.6b4]
// CHECK:STDOUT:   %.loc37_37.22: %facet_type.8a3 = converted constants.%tuple.type.9ef, %facet_value.loc37_37.11 [concrete = constants.%facet_value.6b4]
// CHECK:STDOUT:   %facet_value.loc37_37.12: %facet_type.8a3 = facet_value constants.%tuple.type.9ef, (constants.%custom_witness.b80ef6.10, constants.%DestroyImpl.impl_witness.c0f) [concrete = constants.%facet_value.6b4]
// CHECK:STDOUT:   %.loc37_37.23: %facet_type.8a3 = converted constants.%tuple.type.9ef, %facet_value.loc37_37.12 [concrete = constants.%facet_value.6b4]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc37_37.4: init %empty_tuple.type = call %SelfDestruct.specific_fn.4(%var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestIterate(constants.%R) {
// CHECK:STDOUT:   %R.patt.loc25_17.2 => constants.%R.patt
// CHECK:STDOUT:   %R.loc25_17.1 => constants.%R
// CHECK:STDOUT:   %R.as_type.loc25_83.1 => constants.%R.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.7a2
// CHECK:STDOUT:   %r.patt.loc25_81.2 => constants.%r.patt.a30
// CHECK:STDOUT:   %r.param_patt.loc25_76.2 => constants.%r.param_patt.d4d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestIterate(constants.%facet_value.728) {
// CHECK:STDOUT:   %R.patt.loc25_17.2 => constants.%R.patt
// CHECK:STDOUT:   %R.loc25_17.1 => constants.%facet_value.728
// CHECK:STDOUT:   %R.as_type.loc25_83.1 => constants.%Range
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.23f
// CHECK:STDOUT:   %r.patt.loc25_81.2 => constants.%r.patt.b9f
// CHECK:STDOUT:   %r.param_patt.loc25_76.2 => constants.%r.param_patt.466
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc25 => constants.%complete_type.357
// CHECK:STDOUT:   %Iterate.lookup_impl_witness => constants.%Iterate.impl_witness.8a1
// CHECK:STDOUT:   %Iterate.facet.loc27_37.5 => constants.%Iterate.facet.cdb
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type => constants.%Iterate.WithSelf.NewCursor.type.012
// CHECK:STDOUT:   %.loc27_37.26 => constants.%.11f
// CHECK:STDOUT:   %impl.elem2.loc27_37.2 => constants.%T.as_type.as.Iterate.impl.NewCursor.bdb
// CHECK:STDOUT:   %specific_impl_fn.loc27_37.3 => constants.%T.as_type.as.Iterate.impl.NewCursor.specific_fn
// CHECK:STDOUT:   %impl.elem1 => constants.%facet_value.6b4
// CHECK:STDOUT:   %as_type => constants.%tuple.type.9ef
// CHECK:STDOUT:   %require_complete.1 => constants.%complete_type.e35
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type => constants.%Iterate.WithSelf.Next.type.b7e
// CHECK:STDOUT:   %.loc27_37.27 => constants.%.e6b
// CHECK:STDOUT:   %impl.elem3.loc27_37.2 => constants.%T.as_type.as.Iterate.impl.Next.cb2
// CHECK:STDOUT:   %specific_impl_fn.loc27_37.4 => constants.%T.as_type.as.Iterate.impl.Next.specific_fn
// CHECK:STDOUT:   %SelfDestruct.specific_fn.5 => constants.%SelfDestruct.specific_fn.60c
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
// CHECK:STDOUT:   %Optional.Get.type.786: type = fn_type @Optional.Get, @Optional(%T.220) [symbolic]
// CHECK:STDOUT:   %Optional.Get.2d0: %Optional.Get.type.786 = struct_value () [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.type.3ea: type = fn_type @Optional.HasValue, @Optional(%T.220) [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.f2f: %Optional.HasValue.type.3ea = struct_value () [symbolic]
// CHECK:STDOUT:   %facet_type.8a3: type = facet_type <@SubobjectDestroy & @DestroyImpl> [concrete]
// CHECK:STDOUT:   %T.9ab: %facet_type.d61 = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.f15: type = class_type @DefaultOptionalStorage, @DefaultOptionalStorage(%T.9ab) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Get.type.3e7: type = fn_type @T.as_type.as.OptionalStorage.impl.Get, @T.as_type.as.OptionalStorage.impl(%T.9ab) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Get.f80: %T.as_type.as.OptionalStorage.impl.Get.type.3e7 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Has.type.371: type = fn_type @T.as_type.as.OptionalStorage.impl.Has, @T.as_type.as.OptionalStorage.impl(%T.9ab) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Has.8ce: %T.as_type.as.OptionalStorage.impl.Has.type.371 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Some.type.4eb: type = fn_type @T.as_type.as.OptionalStorage.impl.Some, @T.as_type.as.OptionalStorage.impl(%T.9ab) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Some.44a: %T.as_type.as.OptionalStorage.impl.Some.type.4eb = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.None.type.76a: type = fn_type @T.as_type.as.OptionalStorage.impl.None, @T.as_type.as.OptionalStorage.impl(%T.9ab) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.None.93b: %T.as_type.as.OptionalStorage.impl.None.type.76a = struct_value () [symbolic]
// CHECK:STDOUT:   %N.fe9: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
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
// CHECK:STDOUT:   %f64.761: type = class_type @Float, @Float(%int_64) [concrete]
// CHECK:STDOUT:   %pattern_type.667: type = pattern_type %f64.761 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.3: type = fn_type @SubobjectDestroy.WithSelf.Op.loc20_53.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.3: %SubobjectDestroy.WithSelf.Op.type.6870fd.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.3: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.3), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.47f: %SubobjectDestroy.type = facet_value %f64.761, (%custom_witness.b80ef6.3) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.a23: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.47f) [concrete]
// CHECK:STDOUT:   %Float.as.Copy.impl.Op.type.e8d: type = fn_type @Float.as.Copy.impl.Op, @Float.as.Copy.impl(%N.fe9) [symbolic]
// CHECK:STDOUT:   %Float.as.Copy.impl.Op.fe9: %Float.as.Copy.impl.Op.type.e8d = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.eac: <witness> = impl_witness imports.%Copy.impl_witness_table.14d, @Float.as.Copy.impl(%int_64) [concrete]
// CHECK:STDOUT:   %facet_value.e8a: %facet_type.d61 = facet_value %f64.761, (%custom_witness.b80ef6.3, %DestroyImpl.impl_witness.a23, %Copy.impl_witness.eac) [concrete]
// CHECK:STDOUT:   %Iterate.lookup_impl_witness.1d8: <witness> = lookup_impl_witness %.Self.500, @Iterate [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.8a4: %facet_type.d61 = impl_witness_access %Iterate.lookup_impl_witness.1d8, element0 [symbolic_self]
// CHECK:STDOUT:   %Iterate_where.type.67d: type = facet_type <@Iterate where %impl.elem0.8a4 = %facet_value.e8a> [concrete]
// CHECK:STDOUT:   %pattern_type.942: type = pattern_type %Iterate_where.type.67d [concrete]
// CHECK:STDOUT:   %R.patt: %pattern_type.942 = symbolic_binding_pattern R, 0 [symbolic]
// CHECK:STDOUT:   %R: %Iterate_where.type.67d = symbolic_binding R, 0 [symbolic]
// CHECK:STDOUT:   %R.as_type: type = facet_access_type %R [symbolic]
// CHECK:STDOUT:   %pattern_type.639: type = pattern_type %R.as_type [symbolic]
// CHECK:STDOUT:   %r.param_patt.cf2: %pattern_type.639 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %r.patt.927: %pattern_type.639 = wrapper_binding_pattern r, %r.param_patt.cf2 [symbolic]
// CHECK:STDOUT:   %i.patt: %pattern_type.667 = value_binding_pattern i [concrete]
// CHECK:STDOUT:   %Iterate.lookup_impl_witness.ea0: <witness> = lookup_impl_witness %R, @Iterate [symbolic]
// CHECK:STDOUT:   %Iterate.facet.714: %Iterate.type = facet_value %R.as_type, (%Iterate.lookup_impl_witness.ea0) [symbolic]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.ed1: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.714) [symbolic]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.46a: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.714) [symbolic]
// CHECK:STDOUT:   %.e6b: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.ed1, %Iterate.facet.714 [symbolic]
// CHECK:STDOUT:   %impl.elem2.80a: %.e6b = impl_witness_access %Iterate.lookup_impl_witness.ea0, element2 [symbolic]
// CHECK:STDOUT:   %impl.elem1.af2: %facet_type.8a3 = impl_witness_access %Iterate.lookup_impl_witness.ea0, element1 [symbolic]
// CHECK:STDOUT:   %as_type.2b7: type = facet_access_type %impl.elem1.af2 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.da8: <specific function> = specific_impl_function %impl.elem2.80a, @Iterate.WithSelf.NewCursor(%Iterate.facet.714) [symbolic]
// CHECK:STDOUT:   %.e65: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.46a, %Iterate.facet.714 [symbolic]
// CHECK:STDOUT:   %impl.elem3.b6f: %.e65 = impl_witness_access %Iterate.lookup_impl_witness.ea0, element3 [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.f1a: type = class_type @DefaultOptionalStorage, @DefaultOptionalStorage(%facet_value.e8a) [concrete]
// CHECK:STDOUT:   %OptionalStorage.impl_witness.ca5: <witness> = impl_witness imports.%OptionalStorage.impl_witness_table.968, @T.as_type.as.OptionalStorage.impl(%facet_value.e8a) [concrete]
// CHECK:STDOUT:   %OptionalStorage.facet.be8: %OptionalStorage.type = facet_value %f64.761, (%OptionalStorage.impl_witness.ca5) [concrete]
// CHECK:STDOUT:   %Optional.694: type = class_type @Optional, @Optional(%OptionalStorage.facet.be8) [concrete]
// CHECK:STDOUT:   %specific_impl_fn.f91: <specific function> = specific_impl_function %impl.elem3.b6f, @Iterate.WithSelf.Next(%Iterate.facet.714) [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.type.eb1: type = fn_type @Optional.HasValue, @Optional(%OptionalStorage.facet.be8) [concrete]
// CHECK:STDOUT:   %Optional.HasValue.ad6: %Optional.HasValue.type.eb1 = struct_value () [concrete]
// CHECK:STDOUT:   %Optional.Get.type.f48: type = fn_type @Optional.Get, @Optional(%OptionalStorage.facet.be8) [concrete]
// CHECK:STDOUT:   %Optional.Get.999: %Optional.Get.type.f48 = struct_value () [concrete]
// CHECK:STDOUT:   %facet_value.6dd: %facet_type.8a3 = facet_value %f64.761, (%custom_witness.b80ef6.3, %DestroyImpl.impl_witness.a23) [concrete]
// CHECK:STDOUT:   %MaybeUnformed.ab3: type = class_type @MaybeUnformed, @MaybeUnformed(%facet_value.6dd) [concrete]
// CHECK:STDOUT:   %.ecc: type = maybe_unformed_type %f64.761 [concrete]
// CHECK:STDOUT:   %struct_type.value.has_value.4c7: type = struct_type {.value: %MaybeUnformed.ab3, .has_value: bool} [concrete]
// CHECK:STDOUT:   %Optional.HasValue.specific_fn.b33: <specific function> = specific_function %Optional.HasValue.ad6, @Optional.HasValue(%OptionalStorage.facet.be8) [concrete]
// CHECK:STDOUT:   %Optional.Get.specific_fn.aca: <specific function> = specific_function %Optional.Get.999, @Optional.Get(%OptionalStorage.facet.be8) [concrete]
// CHECK:STDOUT:   %AddAssignWith.type.0c5: type = facet_type <@AddAssignWith, @AddAssignWith(%f64.761)> [concrete]
// CHECK:STDOUT:   %Float.as.AddAssignWith.impl.Op.type.763: type = fn_type @Float.as.AddAssignWith.impl.Op, @Float.as.AddAssignWith.impl(%N.fe9) [symbolic]
// CHECK:STDOUT:   %Float.as.AddAssignWith.impl.Op.8aa: %Float.as.AddAssignWith.impl.Op.type.763 = struct_value () [symbolic]
// CHECK:STDOUT:   %AddAssignWith.impl_witness.5c7: <witness> = impl_witness imports.%AddAssignWith.impl_witness_table, @Float.as.AddAssignWith.impl(%int_64) [concrete]
// CHECK:STDOUT:   %Float.as.AddAssignWith.impl.Op.type.f20: type = fn_type @Float.as.AddAssignWith.impl.Op, @Float.as.AddAssignWith.impl(%int_64) [concrete]
// CHECK:STDOUT:   %Float.as.AddAssignWith.impl.Op.e04: %Float.as.AddAssignWith.impl.Op.type.f20 = struct_value () [concrete]
// CHECK:STDOUT:   %AddAssignWith.facet: %AddAssignWith.type.0c5 = facet_value %f64.761, (%AddAssignWith.impl_witness.5c7) [concrete]
// CHECK:STDOUT:   %AddAssignWith.WithSelf.Op.type.bb6: type = fn_type @AddAssignWith.WithSelf.Op, @AddAssignWith.WithSelf(%f64.761, %AddAssignWith.facet) [concrete]
// CHECK:STDOUT:   %.48b: type = fn_type_with_self_type %AddAssignWith.WithSelf.Op.type.bb6, %AddAssignWith.facet [concrete]
// CHECK:STDOUT:   %Float.as.AddAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Float.as.AddAssignWith.impl.Op.e04, @Float.as.AddAssignWith.impl.Op(%int_64) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.9: type = fn_type @SubobjectDestroy.WithSelf.Op.loc24_19.6 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.9: %SubobjectDestroy.WithSelf.Op.type.6870fd.9 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.9: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.9), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.2dd: %SubobjectDestroy.type = facet_value %Optional.694, (%custom_witness.b80ef6.9) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.dcd: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.2dd) [concrete]
// CHECK:STDOUT:   %facet_value.2ba: %facet_type.8a3 = facet_value %Optional.694, (%custom_witness.b80ef6.9, %DestroyImpl.impl_witness.dcd) [concrete]
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
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.10: type = fn_type @SubobjectDestroy.WithSelf.Op.loc31 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.10: %SubobjectDestroy.WithSelf.Op.type.6870fd.10 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.10: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.10), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.71a: %SubobjectDestroy.type = facet_value %Iterator, (%custom_witness.b80ef6.10) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.32e: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.71a) [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.ac7e09.1: type = fn_type @Iterator.Op.1 [concrete]
// CHECK:STDOUT:   %Iterator.Op.583fda.1: %Iterator.Op.type.ac7e09.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.f6f: <witness> = custom_witness (%f64.761, %Iterator.Op.583fda.1), @CppUnsafeDeref [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.ac7e09.2: type = fn_type @Iterator.Op.2 [concrete]
// CHECK:STDOUT:   %Iterator.Op.583fda.2: %Iterator.Op.type.ac7e09.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.393: <witness> = custom_witness (%Iterator.Op.583fda.2), @Inc [concrete]
// CHECK:STDOUT:   %Equal.type: type = fn_type @Equal [concrete]
// CHECK:STDOUT:   %Equal: %Equal.type = struct_value () [concrete]
// CHECK:STDOUT:   %NotEqual.type: type = fn_type @NotEqual [concrete]
// CHECK:STDOUT:   %NotEqual: %NotEqual.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.eb7: <witness> = custom_witness (%Equal, %NotEqual), @EqWith, @EqWith(%Iterator) [concrete]
// CHECK:STDOUT:   %facet_value.ecd: %CppRange_where.type.d682d6.1 = facet_value %ConstRange, (%custom_witness.5d5, %custom_witness.b80ef6.10, %DestroyImpl.impl_witness.32e, %custom_witness.f6f, %custom_witness.393, %custom_witness.eb7, %custom_witness.b80ef6.3, %DestroyImpl.impl_witness.a23, %Copy.impl_witness.eac, %custom_witness.b80ef6.10, %DestroyImpl.impl_witness.32e) [concrete]
// CHECK:STDOUT:   %tuple.type.005: type = tuple_type (%Iterator, %Iterator) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.11: type = fn_type @SubobjectDestroy.WithSelf.Op.3 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.11: %SubobjectDestroy.WithSelf.Op.type.6870fd.11 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.11: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.11), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.7c6: %SubobjectDestroy.type = facet_value %tuple.type.005, (%custom_witness.b80ef6.11) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.729: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.7c6) [concrete]
// CHECK:STDOUT:   %complete_type.320: <witness> = complete_type_witness %tuple.type.005 [concrete]
// CHECK:STDOUT:   %facet_value.1c3: %facet_type.8a3 = facet_value %tuple.type.005, (%custom_witness.b80ef6.11, %DestroyImpl.impl_witness.729) [concrete]
// CHECK:STDOUT:   %Iterate.impl_witness.5f4: <witness> = impl_witness imports.%Iterate.impl_witness_table.121, @T.as_type.as.Iterate.impl(%facet_value.ecd) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.type.79b: type = fn_type @T.as_type.as.Iterate.impl.NewCursor, @T.as_type.as.Iterate.impl(%facet_value.ecd) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.6f3: %T.as_type.as.Iterate.impl.NewCursor.type.79b = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.type.1d8: type = fn_type @T.as_type.as.Iterate.impl.Next, @T.as_type.as.Iterate.impl(%facet_value.ecd) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.984: %T.as_type.as.Iterate.impl.Next.type.1d8 = struct_value () [concrete]
// CHECK:STDOUT:   %Iterate.facet.10d: %Iterate.type = facet_value %ConstRange, (%Iterate.impl_witness.5f4) [concrete]
// CHECK:STDOUT:   %facet_value.283: %Iterate_where.type.67d = facet_value %ConstRange, (%Iterate.impl_witness.5f4) [concrete]
// CHECK:STDOUT:   %r.param_patt.a73: %pattern_type.025 = value_param_pattern [concrete]
// CHECK:STDOUT:   %r.patt.0b5: %pattern_type.025 = wrapper_binding_pattern r, %r.param_patt.a73 [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.008: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.10d) [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.d0b: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.10d) [concrete]
// CHECK:STDOUT:   %.46c: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.008, %Iterate.facet.10d [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.specific_fn: <specific function> = specific_function %T.as_type.as.Iterate.impl.NewCursor.6f3, @T.as_type.as.Iterate.impl.NewCursor(%facet_value.ecd) [concrete]
// CHECK:STDOUT:   %.a5a: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.d0b, %Iterate.facet.10d [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.specific_fn: <specific function> = specific_function %T.as_type.as.Iterate.impl.Next.984, @T.as_type.as.Iterate.impl.Next(%facet_value.ecd) [concrete]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.33d: <specific function> = specific_function %SelfDestruct, @SelfDestruct(%facet_value.1c3) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.c82: @Optional.%Optional.HasValue.type (%Optional.HasValue.type.3ea) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.HasValue (constants.%Optional.HasValue.f2f)]
// CHECK:STDOUT:   %Core.import_ref.29e: @Optional.%Optional.Get.type (%Optional.Get.type.786) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.Get (constants.%Optional.Get.2d0)]
// CHECK:STDOUT:   %Core.import_ref.9a31: type = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%DefaultOptionalStorage (constants.%DefaultOptionalStorage.f15)]
// CHECK:STDOUT:   %Core.import_ref.819a: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.None.type (%T.as_type.as.OptionalStorage.impl.None.type.76a) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.None (constants.%T.as_type.as.OptionalStorage.impl.None.93b)]
// CHECK:STDOUT:   %Core.import_ref.14a: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Some.type (%T.as_type.as.OptionalStorage.impl.Some.type.4eb) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Some (constants.%T.as_type.as.OptionalStorage.impl.Some.44a)]
// CHECK:STDOUT:   %Core.import_ref.242: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Has.type (%T.as_type.as.OptionalStorage.impl.Has.type.371) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Has (constants.%T.as_type.as.OptionalStorage.impl.Has.8ce)]
// CHECK:STDOUT:   %Core.import_ref.a54: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Get.type (%T.as_type.as.OptionalStorage.impl.Get.type.3e7) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Get (constants.%T.as_type.as.OptionalStorage.impl.Get.f80)]
// CHECK:STDOUT:   %Core.import_ref.5bf = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], unloaded
// CHECK:STDOUT:   %OptionalStorage.impl_witness_table.968 = impl_witness_table (%Core.import_ref.9a31, %Core.import_ref.819a, %Core.import_ref.14a, %Core.import_ref.242, %Core.import_ref.a54, %Core.import_ref.5bf), @T.as_type.as.OptionalStorage.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.822: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.822), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.d6c: %facet_type.d61 = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.Iterate.impl.%facet_value.1 (constants.%facet_value.9da)]
// CHECK:STDOUT:   %Core.import_ref.2ac: %facet_type.8a3 = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.Iterate.impl.%facet_value.2 (constants.%facet_value.bf6)]
// CHECK:STDOUT:   %Core.import_ref.946: @T.as_type.as.Iterate.impl.%T.as_type.as.Iterate.impl.NewCursor.type (%T.as_type.as.Iterate.impl.NewCursor.type.8fb) = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.Iterate.impl.%T.as_type.as.Iterate.impl.NewCursor (constants.%T.as_type.as.Iterate.impl.NewCursor.8a1)]
// CHECK:STDOUT:   %Core.import_ref.cfb: @T.as_type.as.Iterate.impl.%T.as_type.as.Iterate.impl.Next.type (%T.as_type.as.Iterate.impl.Next.type.d4a) = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.Iterate.impl.%T.as_type.as.Iterate.impl.Next (constants.%T.as_type.as.Iterate.impl.Next.5a8)]
// CHECK:STDOUT:   %Iterate.impl_witness_table.121 = impl_witness_table (%Core.import_ref.d6c, %Core.import_ref.2ac, %Core.import_ref.946, %Core.import_ref.cfb), @T.as_type.as.Iterate.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.19b: @Float.as.Copy.impl.%Float.as.Copy.impl.Op.type (%Float.as.Copy.impl.Op.type.e8d) = import_ref Core//prelude/types/float, loc{{\d+_\d+}}, loaded [symbolic = @Float.as.Copy.impl.%Float.as.Copy.impl.Op (constants.%Float.as.Copy.impl.Op.fe9)]
// CHECK:STDOUT:   %Copy.impl_witness_table.14d = impl_witness_table (%Core.import_ref.19b), @Float.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.e0d0: @Float.as.AddAssignWith.impl.%Float.as.AddAssignWith.impl.Op.type (%Float.as.AddAssignWith.impl.Op.type.763) = import_ref Core//prelude/types/float, loc{{\d+_\d+}}, loaded [symbolic = @Float.as.AddAssignWith.impl.%Float.as.AddAssignWith.impl.Op (constants.%Float.as.AddAssignWith.impl.Op.8aa)]
// CHECK:STDOUT:   %AddAssignWith.impl_witness_table = impl_witness_table (%Core.import_ref.e0d0), @Float.as.AddAssignWith.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @TestIterate(%R.loc20_17.2: %Iterate_where.type.67d) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Iterate.lookup_impl_witness: <witness> = lookup_impl_witness %R.loc20_17.1, @Iterate [symbolic = %Iterate.lookup_impl_witness (constants.%Iterate.lookup_impl_witness.ea0)]
// CHECK:STDOUT:   %Iterate.facet.loc24_19.5: %Iterate.type = facet_value %R.as_type.loc20_61.1, (%Iterate.lookup_impl_witness) [symbolic = %Iterate.facet.loc24_19.5 (constants.%Iterate.facet.714)]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.loc24_19.5) [symbolic = %Iterate.WithSelf.NewCursor.type (constants.%Iterate.WithSelf.NewCursor.type.ed1)]
// CHECK:STDOUT:   %.loc24_19.23: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type, %Iterate.facet.loc24_19.5 [symbolic = %.loc24_19.23 (constants.%.e6b)]
// CHECK:STDOUT:   %impl.elem2.loc24_19.2: @TestIterate.%.loc24_19.23 (%.e6b) = impl_witness_access %Iterate.lookup_impl_witness, element2 [symbolic = %impl.elem2.loc24_19.2 (constants.%impl.elem2.80a)]
// CHECK:STDOUT:   %specific_impl_fn.loc24_19.3: <specific function> = specific_impl_function %impl.elem2.loc24_19.2, @Iterate.WithSelf.NewCursor(%Iterate.facet.loc24_19.5) [symbolic = %specific_impl_fn.loc24_19.3 (constants.%specific_impl_fn.da8)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.loc24_19.5) [symbolic = %Iterate.WithSelf.Next.type (constants.%Iterate.WithSelf.Next.type.46a)]
// CHECK:STDOUT:   %.loc24_19.24: type = fn_type_with_self_type %Iterate.WithSelf.Next.type, %Iterate.facet.loc24_19.5 [symbolic = %.loc24_19.24 (constants.%.e65)]
// CHECK:STDOUT:   %impl.elem3.loc24_19.2: @TestIterate.%.loc24_19.24 (%.e65) = impl_witness_access %Iterate.lookup_impl_witness, element3 [symbolic = %impl.elem3.loc24_19.2 (constants.%impl.elem3.b6f)]
// CHECK:STDOUT:   %specific_impl_fn.loc24_19.4: <specific function> = specific_impl_function %impl.elem3.loc24_19.2, @Iterate.WithSelf.Next(%Iterate.facet.loc24_19.5) [symbolic = %specific_impl_fn.loc24_19.4 (constants.%specific_impl_fn.f91)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%r.param: @TestIterate.%R.as_type.loc20_61.1 (%R.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %i.patt: %pattern_type.667 = value_binding_pattern i [concrete = constants.%i.patt]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %r.ref: @TestIterate.%R.as_type.loc20_61.1 (%R.as_type) = name_ref r, %r
// CHECK:STDOUT:     %impl.elem2.loc24_19.1: @TestIterate.%.loc24_19.23 (%.e6b) = impl_witness_access constants.%Iterate.lookup_impl_witness.ea0, element2 [symbolic = %impl.elem2.loc24_19.2 (constants.%impl.elem2.80a)]
// CHECK:STDOUT:     %bound_method.loc24_19.1: <bound method> = bound_method %r.ref, %impl.elem2.loc24_19.1
// CHECK:STDOUT:     %Iterate.facet.loc24_19.1: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.ea0) [symbolic = %Iterate.facet.loc24_19.5 (constants.%Iterate.facet.714)]
// CHECK:STDOUT:     %.loc24_19.1: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc24_19.1 [symbolic = %Iterate.facet.loc24_19.5 (constants.%Iterate.facet.714)]
// CHECK:STDOUT:     %Iterate.facet.loc24_19.2: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.ea0) [symbolic = %Iterate.facet.loc24_19.5 (constants.%Iterate.facet.714)]
// CHECK:STDOUT:     %.loc24_19.2: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc24_19.2 [symbolic = %Iterate.facet.loc24_19.5 (constants.%Iterate.facet.714)]
// CHECK:STDOUT:     %specific_impl_fn.loc24_19.1: <specific function> = specific_impl_function %impl.elem2.loc24_19.1, @Iterate.WithSelf.NewCursor(constants.%Iterate.facet.714) [symbolic = %specific_impl_fn.loc24_19.3 (constants.%specific_impl_fn.da8)]
// CHECK:STDOUT:     %bound_method.loc24_19.2: <bound method> = bound_method %r.ref, %specific_impl_fn.loc24_19.1
// CHECK:STDOUT:     %var: ref @TestIterate.%as_type (%as_type.2b7) = var_storage invalid
// CHECK:STDOUT:     %Iterate.WithSelf.NewCursor.call: init @TestIterate.%as_type (%as_type.2b7) to %var = call %bound_method.loc24_19.2(%r.ref)
// CHECK:STDOUT:     assign %var, %Iterate.WithSelf.NewCursor.call
// CHECK:STDOUT:     br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.next:
// CHECK:STDOUT:     %impl.elem3.loc24_19.1: @TestIterate.%.loc24_19.24 (%.e65) = impl_witness_access constants.%Iterate.lookup_impl_witness.ea0, element3 [symbolic = %impl.elem3.loc24_19.2 (constants.%impl.elem3.b6f)]
// CHECK:STDOUT:     %bound_method.loc24_19.3: <bound method> = bound_method %r.ref, %impl.elem3.loc24_19.1
// CHECK:STDOUT:     %Iterate.facet.loc24_19.3: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.ea0) [symbolic = %Iterate.facet.loc24_19.5 (constants.%Iterate.facet.714)]
// CHECK:STDOUT:     %.loc24_19.3: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc24_19.3 [symbolic = %Iterate.facet.loc24_19.5 (constants.%Iterate.facet.714)]
// CHECK:STDOUT:     %Iterate.facet.loc24_19.4: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.ea0) [symbolic = %Iterate.facet.loc24_19.5 (constants.%Iterate.facet.714)]
// CHECK:STDOUT:     %.loc24_19.4: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc24_19.4 [symbolic = %Iterate.facet.loc24_19.5 (constants.%Iterate.facet.714)]
// CHECK:STDOUT:     %.loc24_19.5: %facet_type.8a3 = converted constants.%as_type.2b7, constants.%impl.elem1.af2 [symbolic = %impl.elem1 (constants.%impl.elem1.af2)]
// CHECK:STDOUT:     %.loc24_19.6: %facet_type.8a3 = converted constants.%as_type.2b7, constants.%impl.elem1.af2 [symbolic = %impl.elem1 (constants.%impl.elem1.af2)]
// CHECK:STDOUT:     %specific_impl_fn.loc24_19.2: <specific function> = specific_impl_function %impl.elem3.loc24_19.1, @Iterate.WithSelf.Next(constants.%Iterate.facet.714) [symbolic = %specific_impl_fn.loc24_19.4 (constants.%specific_impl_fn.f91)]
// CHECK:STDOUT:     %bound_method.loc24_19.4: <bound method> = bound_method %r.ref, %specific_impl_fn.loc24_19.2
// CHECK:STDOUT:     %.loc24_19.7: ref %Optional.694 = temporary_storage
// CHECK:STDOUT:     %Iterate.WithSelf.Next.call: init %Optional.694 to %.loc24_19.7 = call %bound_method.loc24_19.4(%r.ref, %var)
// CHECK:STDOUT:     %.loc24_19.8: ref %Optional.694 = temporary %.loc24_19.7, %Iterate.WithSelf.Next.call
// CHECK:STDOUT:     %.loc24_19.9: %Optional.HasValue.type.eb1 = specific_constant imports.%Core.import_ref.c82, @Optional(constants.%OptionalStorage.facet.be8) [concrete = constants.%Optional.HasValue.ad6]
// CHECK:STDOUT:     %HasValue.ref: %Optional.HasValue.type.eb1 = name_ref HasValue, %.loc24_19.9 [concrete = constants.%Optional.HasValue.ad6]
// CHECK:STDOUT:     %Optional.HasValue.bound: <bound method> = bound_method %.loc24_19.8, %HasValue.ref
// CHECK:STDOUT:     %Optional.HasValue.specific_fn: <specific function> = specific_function %HasValue.ref, @Optional.HasValue(constants.%OptionalStorage.facet.be8) [concrete = constants.%Optional.HasValue.specific_fn.b33]
// CHECK:STDOUT:     %bound_method.loc24_19.5: <bound method> = bound_method %.loc24_19.8, %Optional.HasValue.specific_fn
// CHECK:STDOUT:     %.loc24_19.10: %Optional.694 = acquire_value %.loc24_19.8
// CHECK:STDOUT:     %Optional.HasValue.call: init bool = call %bound_method.loc24_19.5(%.loc24_19.10)
// CHECK:STDOUT:     %.loc24_19.11: bool = value_of_initializer %Optional.HasValue.call
// CHECK:STDOUT:     %.loc24_19.12: bool = converted %Optional.HasValue.call, %.loc24_19.11
// CHECK:STDOUT:     if %.loc24_19.12 br !for.body else br !for.done
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.body:
// CHECK:STDOUT:     %.loc24_19.13: %Optional.Get.type.f48 = specific_constant imports.%Core.import_ref.29e, @Optional(constants.%OptionalStorage.facet.be8) [concrete = constants.%Optional.Get.999]
// CHECK:STDOUT:     %Get.ref: %Optional.Get.type.f48 = name_ref Get, %.loc24_19.13 [concrete = constants.%Optional.Get.999]
// CHECK:STDOUT:     %Optional.Get.bound: <bound method> = bound_method %.loc24_19.8, %Get.ref
// CHECK:STDOUT:     %Optional.Get.specific_fn: <specific function> = specific_function %Get.ref, @Optional.Get(constants.%OptionalStorage.facet.be8) [concrete = constants.%Optional.Get.specific_fn.aca]
// CHECK:STDOUT:     %bound_method.loc24_19.6: <bound method> = bound_method %.loc24_19.8, %Optional.Get.specific_fn
// CHECK:STDOUT:     %.loc24_19.14: %Optional.694 = acquire_value %.loc24_19.8
// CHECK:STDOUT:     %Optional.Get.call: init %f64.761 = call %bound_method.loc24_19.6(%.loc24_19.14)
// CHECK:STDOUT:     %.loc24_19.15: %f64.761 = value_of_initializer %Optional.Get.call
// CHECK:STDOUT:     %.loc24_19.16: %f64.761 = converted %Optional.Get.call, %.loc24_19.15
// CHECK:STDOUT:     %f64.loc24: type = type_literal constants.%f64.761 [concrete = constants.%f64.761]
// CHECK:STDOUT:     %i: %f64.761 = wrapper_binding i, %.loc24_19.16
// CHECK:STDOUT:     %sum.ref: ref %f64.761 = name_ref sum, %sum
// CHECK:STDOUT:     %i.ref: %f64.761 = name_ref i, %i
// CHECK:STDOUT:     %impl.elem0.loc25: %.48b = impl_witness_access constants.%AddAssignWith.impl_witness.5c7, element0 [concrete = constants.%Float.as.AddAssignWith.impl.Op.e04]
// CHECK:STDOUT:     %bound_method.loc25_9.1: <bound method> = bound_method %sum.ref, %impl.elem0.loc25
// CHECK:STDOUT:     %specific_fn.loc25: <specific function> = specific_function %impl.elem0.loc25, @Float.as.AddAssignWith.impl.Op(constants.%int_64) [concrete = constants.%Float.as.AddAssignWith.impl.Op.specific_fn]
// CHECK:STDOUT:     %bound_method.loc25_9.2: <bound method> = bound_method %sum.ref, %specific_fn.loc25
// CHECK:STDOUT:     %Float.as.AddAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc25_9.2(%sum.ref, %i.ref)
// CHECK:STDOUT:     %facet_value.loc24_19.1: %facet_type.8a3 = facet_value constants.%Optional.694, (constants.%custom_witness.b80ef6.9, constants.%DestroyImpl.impl_witness.dcd) [concrete = constants.%facet_value.2ba]
// CHECK:STDOUT:     %.loc24_19.17: %facet_type.8a3 = converted constants.%Optional.694, %facet_value.loc24_19.1 [concrete = constants.%facet_value.2ba]
// CHECK:STDOUT:     %facet_value.loc24_19.2: %facet_type.8a3 = facet_value constants.%Optional.694, (constants.%custom_witness.b80ef6.9, constants.%DestroyImpl.impl_witness.dcd) [concrete = constants.%facet_value.2ba]
// CHECK:STDOUT:     %.loc24_19.18: %facet_type.8a3 = converted constants.%Optional.694, %facet_value.loc24_19.2 [concrete = constants.%facet_value.2ba]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %SelfDestruct.call.loc24_19.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc24_19.8)
// CHECK:STDOUT:     br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.done:
// CHECK:STDOUT:     %facet_value.loc24_19.3: %facet_type.8a3 = facet_value constants.%Optional.694, (constants.%custom_witness.b80ef6.9, constants.%DestroyImpl.impl_witness.dcd) [concrete = constants.%facet_value.2ba]
// CHECK:STDOUT:     %.loc24_19.19: %facet_type.8a3 = converted constants.%Optional.694, %facet_value.loc24_19.3 [concrete = constants.%facet_value.2ba]
// CHECK:STDOUT:     %facet_value.loc24_19.4: %facet_type.8a3 = facet_value constants.%Optional.694, (constants.%custom_witness.b80ef6.9, constants.%DestroyImpl.impl_witness.dcd) [concrete = constants.%facet_value.2ba]
// CHECK:STDOUT:     %.loc24_19.20: %facet_type.8a3 = converted constants.%Optional.694, %facet_value.loc24_19.4 [concrete = constants.%facet_value.2ba]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %SelfDestruct.call.loc24_19.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc24_19.8)
// CHECK:STDOUT:     %.loc24_19.21: %facet_type.8a3 = converted constants.%as_type.2b7, constants.%impl.elem1.af2 [symbolic = %impl.elem1 (constants.%impl.elem1.af2)]
// CHECK:STDOUT:     %.loc24_19.22: %facet_type.8a3 = converted constants.%as_type.2b7, constants.%impl.elem1.af2 [symbolic = %impl.elem1 (constants.%impl.elem1.af2)]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %SelfDestruct.call.loc24_19.3: init %empty_tuple.type = call %SelfDestruct.specific_fn.3(%var)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc24_19.1(%self.param: ref %.ecc) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc24_19.2(%self.param: ref %MaybeUnformed.ab3) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc24_19.3(%self.param: ref bool) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc24_19.4(%self.param: ref %struct_type.value.has_value.4c7) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc24_19.5(%self.param: ref %DefaultOptionalStorage.f1a) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc24_19.6(%self.param: ref %Optional.694) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestRangeFor(%c.param: %ConstRange) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %i.patt: %pattern_type.667 = value_binding_pattern i [concrete = constants.%i.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c.ref: %ConstRange = name_ref c, %c
// CHECK:STDOUT:   %impl.elem2: %.46c = impl_witness_access constants.%Iterate.impl_witness.5f4, element2 [concrete = constants.%T.as_type.as.Iterate.impl.NewCursor.6f3]
// CHECK:STDOUT:   %bound_method.loc38_19.1: <bound method> = bound_method %c.ref, %impl.elem2
// CHECK:STDOUT:   %facet_value.loc38_19.1: %CppRange_where.type.d682d6.1 = facet_value constants.%ConstRange, (constants.%custom_witness.5d5, constants.%custom_witness.b80ef6.10, constants.%DestroyImpl.impl_witness.32e, constants.%custom_witness.f6f, constants.%custom_witness.393, constants.%custom_witness.eb7, constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.a23, constants.%Copy.impl_witness.eac, constants.%custom_witness.b80ef6.10, constants.%DestroyImpl.impl_witness.32e) [concrete = constants.%facet_value.ecd]
// CHECK:STDOUT:   %.loc38_19.1: %CppRange_where.type.d682d6.1 = converted constants.%ConstRange, %facet_value.loc38_19.1 [concrete = constants.%facet_value.ecd]
// CHECK:STDOUT:   %facet_value.loc38_19.2: %CppRange_where.type.d682d6.1 = facet_value constants.%ConstRange, (constants.%custom_witness.5d5, constants.%custom_witness.b80ef6.10, constants.%DestroyImpl.impl_witness.32e, constants.%custom_witness.f6f, constants.%custom_witness.393, constants.%custom_witness.eb7, constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.a23, constants.%Copy.impl_witness.eac, constants.%custom_witness.b80ef6.10, constants.%DestroyImpl.impl_witness.32e) [concrete = constants.%facet_value.ecd]
// CHECK:STDOUT:   %.loc38_19.2: %CppRange_where.type.d682d6.1 = converted constants.%ConstRange, %facet_value.loc38_19.2 [concrete = constants.%facet_value.ecd]
// CHECK:STDOUT:   %specific_fn.loc38_19.1: <specific function> = specific_function %impl.elem2, @T.as_type.as.Iterate.impl.NewCursor(constants.%facet_value.ecd) [concrete = constants.%T.as_type.as.Iterate.impl.NewCursor.specific_fn]
// CHECK:STDOUT:   %bound_method.loc38_19.2: <bound method> = bound_method %c.ref, %specific_fn.loc38_19.1
// CHECK:STDOUT:   %var: ref %tuple.type.005 = var_storage invalid
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.call: init %tuple.type.005 to %var = call %bound_method.loc38_19.2(%c.ref)
// CHECK:STDOUT:   assign %var, %T.as_type.as.Iterate.impl.NewCursor.call
// CHECK:STDOUT:   br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT: !for.next:
// CHECK:STDOUT:   %impl.elem3: %.a5a = impl_witness_access constants.%Iterate.impl_witness.5f4, element3 [concrete = constants.%T.as_type.as.Iterate.impl.Next.984]
// CHECK:STDOUT:   %bound_method.loc38_19.3: <bound method> = bound_method %c.ref, %impl.elem3
// CHECK:STDOUT:   %facet_value.loc38_19.3: %CppRange_where.type.d682d6.1 = facet_value constants.%ConstRange, (constants.%custom_witness.5d5, constants.%custom_witness.b80ef6.10, constants.%DestroyImpl.impl_witness.32e, constants.%custom_witness.f6f, constants.%custom_witness.393, constants.%custom_witness.eb7, constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.a23, constants.%Copy.impl_witness.eac, constants.%custom_witness.b80ef6.10, constants.%DestroyImpl.impl_witness.32e) [concrete = constants.%facet_value.ecd]
// CHECK:STDOUT:   %.loc38_19.3: %CppRange_where.type.d682d6.1 = converted constants.%ConstRange, %facet_value.loc38_19.3 [concrete = constants.%facet_value.ecd]
// CHECK:STDOUT:   %facet_value.loc38_19.4: %CppRange_where.type.d682d6.1 = facet_value constants.%ConstRange, (constants.%custom_witness.5d5, constants.%custom_witness.b80ef6.10, constants.%DestroyImpl.impl_witness.32e, constants.%custom_witness.f6f, constants.%custom_witness.393, constants.%custom_witness.eb7, constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.a23, constants.%Copy.impl_witness.eac, constants.%custom_witness.b80ef6.10, constants.%DestroyImpl.impl_witness.32e) [concrete = constants.%facet_value.ecd]
// CHECK:STDOUT:   %.loc38_19.4: %CppRange_where.type.d682d6.1 = converted constants.%ConstRange, %facet_value.loc38_19.4 [concrete = constants.%facet_value.ecd]
// CHECK:STDOUT:   %specific_fn.loc38_19.2: <specific function> = specific_function %impl.elem3, @T.as_type.as.Iterate.impl.Next(constants.%facet_value.ecd) [concrete = constants.%T.as_type.as.Iterate.impl.Next.specific_fn]
// CHECK:STDOUT:   %bound_method.loc38_19.4: <bound method> = bound_method %c.ref, %specific_fn.loc38_19.2
// CHECK:STDOUT:   %.loc38_19.5: ref %Optional.694 = temporary_storage
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.call: init %Optional.694 to %.loc38_19.5 = call %bound_method.loc38_19.4(%c.ref, %var)
// CHECK:STDOUT:   %.loc38_19.6: ref %Optional.694 = temporary %.loc38_19.5, %T.as_type.as.Iterate.impl.Next.call
// CHECK:STDOUT:   %.loc38_19.7: %Optional.HasValue.type.eb1 = specific_constant imports.%Core.import_ref.c82, @Optional(constants.%OptionalStorage.facet.be8) [concrete = constants.%Optional.HasValue.ad6]
// CHECK:STDOUT:   %HasValue.ref: %Optional.HasValue.type.eb1 = name_ref HasValue, %.loc38_19.7 [concrete = constants.%Optional.HasValue.ad6]
// CHECK:STDOUT:   %Optional.HasValue.bound: <bound method> = bound_method %.loc38_19.6, %HasValue.ref
// CHECK:STDOUT:   %Optional.HasValue.specific_fn: <specific function> = specific_function %HasValue.ref, @Optional.HasValue(constants.%OptionalStorage.facet.be8) [concrete = constants.%Optional.HasValue.specific_fn.b33]
// CHECK:STDOUT:   %bound_method.loc38_19.5: <bound method> = bound_method %.loc38_19.6, %Optional.HasValue.specific_fn
// CHECK:STDOUT:   %.loc38_19.8: %Optional.694 = acquire_value %.loc38_19.6
// CHECK:STDOUT:   %Optional.HasValue.call: init bool = call %bound_method.loc38_19.5(%.loc38_19.8)
// CHECK:STDOUT:   %.loc38_19.9: bool = value_of_initializer %Optional.HasValue.call
// CHECK:STDOUT:   %.loc38_19.10: bool = converted %Optional.HasValue.call, %.loc38_19.9
// CHECK:STDOUT:   if %.loc38_19.10 br !for.body else br !for.done
// CHECK:STDOUT:
// CHECK:STDOUT: !for.body:
// CHECK:STDOUT:   %.loc38_19.11: %Optional.Get.type.f48 = specific_constant imports.%Core.import_ref.29e, @Optional(constants.%OptionalStorage.facet.be8) [concrete = constants.%Optional.Get.999]
// CHECK:STDOUT:   %Get.ref: %Optional.Get.type.f48 = name_ref Get, %.loc38_19.11 [concrete = constants.%Optional.Get.999]
// CHECK:STDOUT:   %Optional.Get.bound: <bound method> = bound_method %.loc38_19.6, %Get.ref
// CHECK:STDOUT:   %Optional.Get.specific_fn: <specific function> = specific_function %Get.ref, @Optional.Get(constants.%OptionalStorage.facet.be8) [concrete = constants.%Optional.Get.specific_fn.aca]
// CHECK:STDOUT:   %bound_method.loc38_19.6: <bound method> = bound_method %.loc38_19.6, %Optional.Get.specific_fn
// CHECK:STDOUT:   %.loc38_19.12: %Optional.694 = acquire_value %.loc38_19.6
// CHECK:STDOUT:   %Optional.Get.call: init %f64.761 = call %bound_method.loc38_19.6(%.loc38_19.12)
// CHECK:STDOUT:   %.loc38_19.13: %f64.761 = value_of_initializer %Optional.Get.call
// CHECK:STDOUT:   %.loc38_19.14: %f64.761 = converted %Optional.Get.call, %.loc38_19.13
// CHECK:STDOUT:   %f64.loc38: type = type_literal constants.%f64.761 [concrete = constants.%f64.761]
// CHECK:STDOUT:   %i: %f64.761 = wrapper_binding i, %.loc38_19.14
// CHECK:STDOUT:   %sum.ref: ref %f64.761 = name_ref sum, %sum
// CHECK:STDOUT:   %i.ref: %f64.761 = name_ref i, %i
// CHECK:STDOUT:   %impl.elem0.loc39: %.48b = impl_witness_access constants.%AddAssignWith.impl_witness.5c7, element0 [concrete = constants.%Float.as.AddAssignWith.impl.Op.e04]
// CHECK:STDOUT:   %bound_method.loc39_9.1: <bound method> = bound_method %sum.ref, %impl.elem0.loc39
// CHECK:STDOUT:   %specific_fn.loc39: <specific function> = specific_function %impl.elem0.loc39, @Float.as.AddAssignWith.impl.Op(constants.%int_64) [concrete = constants.%Float.as.AddAssignWith.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc39_9.2: <bound method> = bound_method %sum.ref, %specific_fn.loc39
// CHECK:STDOUT:   %Float.as.AddAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc39_9.2(%sum.ref, %i.ref)
// CHECK:STDOUT:   %facet_value.loc38_19.5: %facet_type.8a3 = facet_value constants.%Optional.694, (constants.%custom_witness.b80ef6.9, constants.%DestroyImpl.impl_witness.dcd) [concrete = constants.%facet_value.2ba]
// CHECK:STDOUT:   %.loc38_19.15: %facet_type.8a3 = converted constants.%Optional.694, %facet_value.loc38_19.5 [concrete = constants.%facet_value.2ba]
// CHECK:STDOUT:   %facet_value.loc38_19.6: %facet_type.8a3 = facet_value constants.%Optional.694, (constants.%custom_witness.b80ef6.9, constants.%DestroyImpl.impl_witness.dcd) [concrete = constants.%facet_value.2ba]
// CHECK:STDOUT:   %.loc38_19.16: %facet_type.8a3 = converted constants.%Optional.694, %facet_value.loc38_19.6 [concrete = constants.%facet_value.2ba]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc38_19.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc38_19.6)
// CHECK:STDOUT:   br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT: !for.done:
// CHECK:STDOUT:   %facet_value.loc38_19.7: %facet_type.8a3 = facet_value constants.%Optional.694, (constants.%custom_witness.b80ef6.9, constants.%DestroyImpl.impl_witness.dcd) [concrete = constants.%facet_value.2ba]
// CHECK:STDOUT:   %.loc38_19.17: %facet_type.8a3 = converted constants.%Optional.694, %facet_value.loc38_19.7 [concrete = constants.%facet_value.2ba]
// CHECK:STDOUT:   %facet_value.loc38_19.8: %facet_type.8a3 = facet_value constants.%Optional.694, (constants.%custom_witness.b80ef6.9, constants.%DestroyImpl.impl_witness.dcd) [concrete = constants.%facet_value.2ba]
// CHECK:STDOUT:   %.loc38_19.18: %facet_type.8a3 = converted constants.%Optional.694, %facet_value.loc38_19.8 [concrete = constants.%facet_value.2ba]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc38_19.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc38_19.6)
// CHECK:STDOUT:   %facet_value.loc38_19.9: %facet_type.8a3 = facet_value constants.%tuple.type.005, (constants.%custom_witness.b80ef6.11, constants.%DestroyImpl.impl_witness.729) [concrete = constants.%facet_value.1c3]
// CHECK:STDOUT:   %.loc38_19.19: %facet_type.8a3 = converted constants.%tuple.type.005, %facet_value.loc38_19.9 [concrete = constants.%facet_value.1c3]
// CHECK:STDOUT:   %facet_value.loc38_19.10: %facet_type.8a3 = facet_value constants.%tuple.type.005, (constants.%custom_witness.b80ef6.11, constants.%DestroyImpl.impl_witness.729) [concrete = constants.%facet_value.1c3]
// CHECK:STDOUT:   %.loc38_19.20: %facet_type.8a3 = converted constants.%tuple.type.005, %facet_value.loc38_19.10 [concrete = constants.%facet_value.1c3]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc38_19.3: init %empty_tuple.type = call %SelfDestruct.specific_fn.3(%var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestIterate(constants.%R) {
// CHECK:STDOUT:   %R.patt.loc20_17.2 => constants.%R.patt
// CHECK:STDOUT:   %R.loc20_17.1 => constants.%R
// CHECK:STDOUT:   %R.as_type.loc20_61.1 => constants.%R.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.639
// CHECK:STDOUT:   %r.param_patt.loc20_59.2 => constants.%r.param_patt.cf2
// CHECK:STDOUT:   %r.patt.loc20_59.2 => constants.%r.patt.927
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestIterate(constants.%facet_value.283) {
// CHECK:STDOUT:   %R.patt.loc20_17.2 => constants.%R.patt
// CHECK:STDOUT:   %R.loc20_17.1 => constants.%facet_value.283
// CHECK:STDOUT:   %R.as_type.loc20_61.1 => constants.%ConstRange
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.025
// CHECK:STDOUT:   %r.param_patt.loc20_59.2 => constants.%r.param_patt.a73
// CHECK:STDOUT:   %r.patt.loc20_59.2 => constants.%r.patt.0b5
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc20 => constants.%complete_type.357
// CHECK:STDOUT:   %Iterate.lookup_impl_witness => constants.%Iterate.impl_witness.5f4
// CHECK:STDOUT:   %Iterate.facet.loc24_19.5 => constants.%Iterate.facet.10d
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type => constants.%Iterate.WithSelf.NewCursor.type.008
// CHECK:STDOUT:   %.loc24_19.23 => constants.%.46c
// CHECK:STDOUT:   %impl.elem2.loc24_19.2 => constants.%T.as_type.as.Iterate.impl.NewCursor.6f3
// CHECK:STDOUT:   %specific_impl_fn.loc24_19.3 => constants.%T.as_type.as.Iterate.impl.NewCursor.specific_fn
// CHECK:STDOUT:   %impl.elem1 => constants.%facet_value.1c3
// CHECK:STDOUT:   %as_type => constants.%tuple.type.005
// CHECK:STDOUT:   %require_complete.1 => constants.%complete_type.320
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type => constants.%Iterate.WithSelf.Next.type.d0b
// CHECK:STDOUT:   %.loc24_19.24 => constants.%.a5a
// CHECK:STDOUT:   %impl.elem3.loc24_19.2 => constants.%T.as_type.as.Iterate.impl.Next.984
// CHECK:STDOUT:   %specific_impl_fn.loc24_19.4 => constants.%T.as_type.as.Iterate.impl.Next.specific_fn
// CHECK:STDOUT:   %SelfDestruct.specific_fn.5 => constants.%SelfDestruct.specific_fn.33d
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
// CHECK:STDOUT:   %Optional.Get.type.786: type = fn_type @Optional.Get, @Optional(%T.220) [symbolic]
// CHECK:STDOUT:   %Optional.Get.2d0: %Optional.Get.type.786 = struct_value () [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.type.3ea: type = fn_type @Optional.HasValue, @Optional(%T.220) [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.f2f: %Optional.HasValue.type.3ea = struct_value () [symbolic]
// CHECK:STDOUT:   %facet_type.8a3: type = facet_type <@SubobjectDestroy & @DestroyImpl> [concrete]
// CHECK:STDOUT:   %T.9ab: %facet_type.d61 = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.f15: type = class_type @DefaultOptionalStorage, @DefaultOptionalStorage(%T.9ab) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Get.type.3e7: type = fn_type @T.as_type.as.OptionalStorage.impl.Get, @T.as_type.as.OptionalStorage.impl(%T.9ab) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Get.f80: %T.as_type.as.OptionalStorage.impl.Get.type.3e7 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Has.type.371: type = fn_type @T.as_type.as.OptionalStorage.impl.Has, @T.as_type.as.OptionalStorage.impl(%T.9ab) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Has.8ce: %T.as_type.as.OptionalStorage.impl.Has.type.371 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Some.type.4eb: type = fn_type @T.as_type.as.OptionalStorage.impl.Some, @T.as_type.as.OptionalStorage.impl(%T.9ab) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Some.44a: %T.as_type.as.OptionalStorage.impl.Some.type.4eb = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.None.type.76a: type = fn_type @T.as_type.as.OptionalStorage.impl.None, @T.as_type.as.OptionalStorage.impl(%T.9ab) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.None.93b: %T.as_type.as.OptionalStorage.impl.None.type.76a = struct_value () [symbolic]
// CHECK:STDOUT:   %N.fe9: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
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
// CHECK:STDOUT:   %f64.761: type = class_type @Float, @Float(%int_64) [concrete]
// CHECK:STDOUT:   %pattern_type.667: type = pattern_type %f64.761 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.3: type = fn_type @SubobjectDestroy.WithSelf.Op.loc44_53.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.3: %SubobjectDestroy.WithSelf.Op.type.6870fd.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.3: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.3), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.47f: %SubobjectDestroy.type = facet_value %f64.761, (%custom_witness.b80ef6.3) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.a23: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.47f) [concrete]
// CHECK:STDOUT:   %Float.as.Copy.impl.Op.type.e8d: type = fn_type @Float.as.Copy.impl.Op, @Float.as.Copy.impl(%N.fe9) [symbolic]
// CHECK:STDOUT:   %Float.as.Copy.impl.Op.fe9: %Float.as.Copy.impl.Op.type.e8d = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.eac: <witness> = impl_witness imports.%Copy.impl_witness_table.14d, @Float.as.Copy.impl(%int_64) [concrete]
// CHECK:STDOUT:   %facet_value.e8a: %facet_type.d61 = facet_value %f64.761, (%custom_witness.b80ef6.3, %DestroyImpl.impl_witness.a23, %Copy.impl_witness.eac) [concrete]
// CHECK:STDOUT:   %Iterate.lookup_impl_witness.1d8: <witness> = lookup_impl_witness %.Self.500, @Iterate [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.8a4: %facet_type.d61 = impl_witness_access %Iterate.lookup_impl_witness.1d8, element0 [symbolic_self]
// CHECK:STDOUT:   %Iterate_where.type.67d: type = facet_type <@Iterate where %impl.elem0.8a4 = %facet_value.e8a> [concrete]
// CHECK:STDOUT:   %pattern_type.942: type = pattern_type %Iterate_where.type.67d [concrete]
// CHECK:STDOUT:   %R.patt: %pattern_type.942 = symbolic_binding_pattern R, 0 [symbolic]
// CHECK:STDOUT:   %R: %Iterate_where.type.67d = symbolic_binding R, 0 [symbolic]
// CHECK:STDOUT:   %R.as_type: type = facet_access_type %R [symbolic]
// CHECK:STDOUT:   %pattern_type.639: type = pattern_type %R.as_type [symbolic]
// CHECK:STDOUT:   %r.param_patt.cf2: %pattern_type.639 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %r.patt.927: %pattern_type.639 = wrapper_binding_pattern r, %r.param_patt.cf2 [symbolic]
// CHECK:STDOUT:   %i.patt: %pattern_type.667 = value_binding_pattern i [concrete]
// CHECK:STDOUT:   %Iterate.lookup_impl_witness.ea0: <witness> = lookup_impl_witness %R, @Iterate [symbolic]
// CHECK:STDOUT:   %Iterate.facet.714: %Iterate.type = facet_value %R.as_type, (%Iterate.lookup_impl_witness.ea0) [symbolic]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.ed1: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.714) [symbolic]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.46a: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.714) [symbolic]
// CHECK:STDOUT:   %.e6b: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.ed1, %Iterate.facet.714 [symbolic]
// CHECK:STDOUT:   %impl.elem2.80a: %.e6b = impl_witness_access %Iterate.lookup_impl_witness.ea0, element2 [symbolic]
// CHECK:STDOUT:   %impl.elem1.af2: %facet_type.8a3 = impl_witness_access %Iterate.lookup_impl_witness.ea0, element1 [symbolic]
// CHECK:STDOUT:   %as_type.2b7: type = facet_access_type %impl.elem1.af2 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.da8: <specific function> = specific_impl_function %impl.elem2.80a, @Iterate.WithSelf.NewCursor(%Iterate.facet.714) [symbolic]
// CHECK:STDOUT:   %.e65: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.46a, %Iterate.facet.714 [symbolic]
// CHECK:STDOUT:   %impl.elem3.b6f: %.e65 = impl_witness_access %Iterate.lookup_impl_witness.ea0, element3 [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.f1a: type = class_type @DefaultOptionalStorage, @DefaultOptionalStorage(%facet_value.e8a) [concrete]
// CHECK:STDOUT:   %OptionalStorage.impl_witness.ca5: <witness> = impl_witness imports.%OptionalStorage.impl_witness_table.968, @T.as_type.as.OptionalStorage.impl(%facet_value.e8a) [concrete]
// CHECK:STDOUT:   %OptionalStorage.facet.be8: %OptionalStorage.type = facet_value %f64.761, (%OptionalStorage.impl_witness.ca5) [concrete]
// CHECK:STDOUT:   %Optional.694: type = class_type @Optional, @Optional(%OptionalStorage.facet.be8) [concrete]
// CHECK:STDOUT:   %specific_impl_fn.f91: <specific function> = specific_impl_function %impl.elem3.b6f, @Iterate.WithSelf.Next(%Iterate.facet.714) [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.type.eb1: type = fn_type @Optional.HasValue, @Optional(%OptionalStorage.facet.be8) [concrete]
// CHECK:STDOUT:   %Optional.HasValue.ad6: %Optional.HasValue.type.eb1 = struct_value () [concrete]
// CHECK:STDOUT:   %Optional.Get.type.f48: type = fn_type @Optional.Get, @Optional(%OptionalStorage.facet.be8) [concrete]
// CHECK:STDOUT:   %Optional.Get.999: %Optional.Get.type.f48 = struct_value () [concrete]
// CHECK:STDOUT:   %facet_value.6dd: %facet_type.8a3 = facet_value %f64.761, (%custom_witness.b80ef6.3, %DestroyImpl.impl_witness.a23) [concrete]
// CHECK:STDOUT:   %MaybeUnformed.ab3: type = class_type @MaybeUnformed, @MaybeUnformed(%facet_value.6dd) [concrete]
// CHECK:STDOUT:   %.ecc: type = maybe_unformed_type %f64.761 [concrete]
// CHECK:STDOUT:   %struct_type.value.has_value.4c7: type = struct_type {.value: %MaybeUnformed.ab3, .has_value: bool} [concrete]
// CHECK:STDOUT:   %Optional.HasValue.specific_fn.b33: <specific function> = specific_function %Optional.HasValue.ad6, @Optional.HasValue(%OptionalStorage.facet.be8) [concrete]
// CHECK:STDOUT:   %Optional.Get.specific_fn.aca: <specific function> = specific_function %Optional.Get.999, @Optional.Get(%OptionalStorage.facet.be8) [concrete]
// CHECK:STDOUT:   %AddAssignWith.type.0c5: type = facet_type <@AddAssignWith, @AddAssignWith(%f64.761)> [concrete]
// CHECK:STDOUT:   %Float.as.AddAssignWith.impl.Op.type.763: type = fn_type @Float.as.AddAssignWith.impl.Op, @Float.as.AddAssignWith.impl(%N.fe9) [symbolic]
// CHECK:STDOUT:   %Float.as.AddAssignWith.impl.Op.8aa: %Float.as.AddAssignWith.impl.Op.type.763 = struct_value () [symbolic]
// CHECK:STDOUT:   %AddAssignWith.impl_witness.5c7: <witness> = impl_witness imports.%AddAssignWith.impl_witness_table, @Float.as.AddAssignWith.impl(%int_64) [concrete]
// CHECK:STDOUT:   %Float.as.AddAssignWith.impl.Op.type.f20: type = fn_type @Float.as.AddAssignWith.impl.Op, @Float.as.AddAssignWith.impl(%int_64) [concrete]
// CHECK:STDOUT:   %Float.as.AddAssignWith.impl.Op.e04: %Float.as.AddAssignWith.impl.Op.type.f20 = struct_value () [concrete]
// CHECK:STDOUT:   %AddAssignWith.facet: %AddAssignWith.type.0c5 = facet_value %f64.761, (%AddAssignWith.impl_witness.5c7) [concrete]
// CHECK:STDOUT:   %AddAssignWith.WithSelf.Op.type.bb6: type = fn_type @AddAssignWith.WithSelf.Op, @AddAssignWith.WithSelf(%f64.761, %AddAssignWith.facet) [concrete]
// CHECK:STDOUT:   %.48b: type = fn_type_with_self_type %AddAssignWith.WithSelf.Op.type.bb6, %AddAssignWith.facet [concrete]
// CHECK:STDOUT:   %Float.as.AddAssignWith.impl.Op.specific_fn: <specific function> = specific_function %Float.as.AddAssignWith.impl.Op.e04, @Float.as.AddAssignWith.impl.Op(%int_64) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.9: type = fn_type @SubobjectDestroy.WithSelf.Op.loc48_19.6 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.9: %SubobjectDestroy.WithSelf.Op.type.6870fd.9 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.9: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.9), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.2dd: %SubobjectDestroy.type = facet_value %Optional.694, (%custom_witness.b80ef6.9) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.dcd: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.2dd) [concrete]
// CHECK:STDOUT:   %facet_value.2ba: %facet_type.8a3 = facet_value %Optional.694, (%custom_witness.b80ef6.9, %DestroyImpl.impl_witness.dcd) [concrete]
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
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.10: type = fn_type @SubobjectDestroy.WithSelf.Op.loc55 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.10: %SubobjectDestroy.WithSelf.Op.type.6870fd.10 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.10: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.10), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.03f: %SubobjectDestroy.type = facet_value %Iterator, (%custom_witness.b80ef6.10) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.452: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.03f) [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.b9731e.1: type = fn_type @Iterator.Op.1 [concrete]
// CHECK:STDOUT:   %Iterator.Op.ac6aad.1: %Iterator.Op.type.b9731e.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.0a8: <witness> = custom_witness (%f64.761, %Iterator.Op.ac6aad.1), @CppUnsafeDeref [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.b9731e.2: type = fn_type @Iterator.Op.2 [concrete]
// CHECK:STDOUT:   %Iterator.Op.ac6aad.2: %Iterator.Op.type.b9731e.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.491: <witness> = custom_witness (%Iterator.Op.ac6aad.2), @Inc [concrete]
// CHECK:STDOUT:   %Equal.type: type = fn_type @Equal [concrete]
// CHECK:STDOUT:   %Equal: %Equal.type = struct_value () [concrete]
// CHECK:STDOUT:   %NotEqual.type: type = fn_type @NotEqual [concrete]
// CHECK:STDOUT:   %NotEqual: %NotEqual.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.068: <witness> = custom_witness (%Equal, %NotEqual), @EqWith, @EqWith(%Iterator) [concrete]
// CHECK:STDOUT:   %facet_value.fd4: %CppRange_where.type.d682d6.1 = facet_value %MutableRange, (%custom_witness.902, %custom_witness.b80ef6.10, %DestroyImpl.impl_witness.452, %custom_witness.0a8, %custom_witness.491, %custom_witness.068, %custom_witness.b80ef6.3, %DestroyImpl.impl_witness.a23, %Copy.impl_witness.eac, %custom_witness.b80ef6.10, %DestroyImpl.impl_witness.452) [concrete]
// CHECK:STDOUT:   %tuple.type.dd0: type = tuple_type (%Iterator, %Iterator) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.11: type = fn_type @SubobjectDestroy.WithSelf.Op.3 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.11: %SubobjectDestroy.WithSelf.Op.type.6870fd.11 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.11: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.11), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.0de: %SubobjectDestroy.type = facet_value %tuple.type.dd0, (%custom_witness.b80ef6.11) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.1fa: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.0de) [concrete]
// CHECK:STDOUT:   %complete_type.843: <witness> = complete_type_witness %tuple.type.dd0 [concrete]
// CHECK:STDOUT:   %facet_value.efd: %facet_type.8a3 = facet_value %tuple.type.dd0, (%custom_witness.b80ef6.11, %DestroyImpl.impl_witness.1fa) [concrete]
// CHECK:STDOUT:   %Iterate.impl_witness.752: <witness> = impl_witness imports.%Iterate.impl_witness_table.121, @T.as_type.as.Iterate.impl(%facet_value.fd4) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.type.7d1: type = fn_type @T.as_type.as.Iterate.impl.NewCursor, @T.as_type.as.Iterate.impl(%facet_value.fd4) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.482: %T.as_type.as.Iterate.impl.NewCursor.type.7d1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.type.e77: type = fn_type @T.as_type.as.Iterate.impl.Next, @T.as_type.as.Iterate.impl(%facet_value.fd4) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.759: %T.as_type.as.Iterate.impl.Next.type.e77 = struct_value () [concrete]
// CHECK:STDOUT:   %Iterate.facet.c03: %Iterate.type = facet_value %MutableRange, (%Iterate.impl_witness.752) [concrete]
// CHECK:STDOUT:   %facet_value.9af: %Iterate_where.type.67d = facet_value %MutableRange, (%Iterate.impl_witness.752) [concrete]
// CHECK:STDOUT:   %r.param_patt.aeb: %pattern_type.992 = value_param_pattern [concrete]
// CHECK:STDOUT:   %r.patt.892: %pattern_type.992 = wrapper_binding_pattern r, %r.param_patt.aeb [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.b9c: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.c03) [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.692: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.c03) [concrete]
// CHECK:STDOUT:   %.d6e: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.b9c, %Iterate.facet.c03 [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.specific_fn: <specific function> = specific_function %T.as_type.as.Iterate.impl.NewCursor.482, @T.as_type.as.Iterate.impl.NewCursor(%facet_value.fd4) [concrete]
// CHECK:STDOUT:   %.446: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.692, %Iterate.facet.c03 [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.specific_fn: <specific function> = specific_function %T.as_type.as.Iterate.impl.Next.759, @T.as_type.as.Iterate.impl.Next(%facet_value.fd4) [concrete]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.4af: <specific function> = specific_function %SelfDestruct, @SelfDestruct(%facet_value.efd) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.c82: @Optional.%Optional.HasValue.type (%Optional.HasValue.type.3ea) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.HasValue (constants.%Optional.HasValue.f2f)]
// CHECK:STDOUT:   %Core.import_ref.29e: @Optional.%Optional.Get.type (%Optional.Get.type.786) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.Get (constants.%Optional.Get.2d0)]
// CHECK:STDOUT:   %Core.import_ref.9a31: type = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%DefaultOptionalStorage (constants.%DefaultOptionalStorage.f15)]
// CHECK:STDOUT:   %Core.import_ref.819a: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.None.type (%T.as_type.as.OptionalStorage.impl.None.type.76a) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.None (constants.%T.as_type.as.OptionalStorage.impl.None.93b)]
// CHECK:STDOUT:   %Core.import_ref.14a: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Some.type (%T.as_type.as.OptionalStorage.impl.Some.type.4eb) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Some (constants.%T.as_type.as.OptionalStorage.impl.Some.44a)]
// CHECK:STDOUT:   %Core.import_ref.242: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Has.type (%T.as_type.as.OptionalStorage.impl.Has.type.371) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Has (constants.%T.as_type.as.OptionalStorage.impl.Has.8ce)]
// CHECK:STDOUT:   %Core.import_ref.a54: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Get.type (%T.as_type.as.OptionalStorage.impl.Get.type.3e7) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Get (constants.%T.as_type.as.OptionalStorage.impl.Get.f80)]
// CHECK:STDOUT:   %Core.import_ref.5bf = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], unloaded
// CHECK:STDOUT:   %OptionalStorage.impl_witness_table.968 = impl_witness_table (%Core.import_ref.9a31, %Core.import_ref.819a, %Core.import_ref.14a, %Core.import_ref.242, %Core.import_ref.a54, %Core.import_ref.5bf), @T.as_type.as.OptionalStorage.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.822: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.822), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.d6c: %facet_type.d61 = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.Iterate.impl.%facet_value.1 (constants.%facet_value.9da)]
// CHECK:STDOUT:   %Core.import_ref.2ac: %facet_type.8a3 = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.Iterate.impl.%facet_value.2 (constants.%facet_value.bf6)]
// CHECK:STDOUT:   %Core.import_ref.946: @T.as_type.as.Iterate.impl.%T.as_type.as.Iterate.impl.NewCursor.type (%T.as_type.as.Iterate.impl.NewCursor.type.8fb) = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.Iterate.impl.%T.as_type.as.Iterate.impl.NewCursor (constants.%T.as_type.as.Iterate.impl.NewCursor.8a1)]
// CHECK:STDOUT:   %Core.import_ref.cfb: @T.as_type.as.Iterate.impl.%T.as_type.as.Iterate.impl.Next.type (%T.as_type.as.Iterate.impl.Next.type.d4a) = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.Iterate.impl.%T.as_type.as.Iterate.impl.Next (constants.%T.as_type.as.Iterate.impl.Next.5a8)]
// CHECK:STDOUT:   %Iterate.impl_witness_table.121 = impl_witness_table (%Core.import_ref.d6c, %Core.import_ref.2ac, %Core.import_ref.946, %Core.import_ref.cfb), @T.as_type.as.Iterate.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.19b: @Float.as.Copy.impl.%Float.as.Copy.impl.Op.type (%Float.as.Copy.impl.Op.type.e8d) = import_ref Core//prelude/types/float, loc{{\d+_\d+}}, loaded [symbolic = @Float.as.Copy.impl.%Float.as.Copy.impl.Op (constants.%Float.as.Copy.impl.Op.fe9)]
// CHECK:STDOUT:   %Copy.impl_witness_table.14d = impl_witness_table (%Core.import_ref.19b), @Float.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.e0d0: @Float.as.AddAssignWith.impl.%Float.as.AddAssignWith.impl.Op.type (%Float.as.AddAssignWith.impl.Op.type.763) = import_ref Core//prelude/types/float, loc{{\d+_\d+}}, loaded [symbolic = @Float.as.AddAssignWith.impl.%Float.as.AddAssignWith.impl.Op (constants.%Float.as.AddAssignWith.impl.Op.8aa)]
// CHECK:STDOUT:   %AddAssignWith.impl_witness_table = impl_witness_table (%Core.import_ref.e0d0), @Float.as.AddAssignWith.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @TestIterate(%R.loc44_17.2: %Iterate_where.type.67d) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Iterate.lookup_impl_witness: <witness> = lookup_impl_witness %R.loc44_17.1, @Iterate [symbolic = %Iterate.lookup_impl_witness (constants.%Iterate.lookup_impl_witness.ea0)]
// CHECK:STDOUT:   %Iterate.facet.loc48_19.5: %Iterate.type = facet_value %R.as_type.loc44_61.1, (%Iterate.lookup_impl_witness) [symbolic = %Iterate.facet.loc48_19.5 (constants.%Iterate.facet.714)]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.loc48_19.5) [symbolic = %Iterate.WithSelf.NewCursor.type (constants.%Iterate.WithSelf.NewCursor.type.ed1)]
// CHECK:STDOUT:   %.loc48_19.23: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type, %Iterate.facet.loc48_19.5 [symbolic = %.loc48_19.23 (constants.%.e6b)]
// CHECK:STDOUT:   %impl.elem2.loc48_19.2: @TestIterate.%.loc48_19.23 (%.e6b) = impl_witness_access %Iterate.lookup_impl_witness, element2 [symbolic = %impl.elem2.loc48_19.2 (constants.%impl.elem2.80a)]
// CHECK:STDOUT:   %specific_impl_fn.loc48_19.3: <specific function> = specific_impl_function %impl.elem2.loc48_19.2, @Iterate.WithSelf.NewCursor(%Iterate.facet.loc48_19.5) [symbolic = %specific_impl_fn.loc48_19.3 (constants.%specific_impl_fn.da8)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.loc48_19.5) [symbolic = %Iterate.WithSelf.Next.type (constants.%Iterate.WithSelf.Next.type.46a)]
// CHECK:STDOUT:   %.loc48_19.24: type = fn_type_with_self_type %Iterate.WithSelf.Next.type, %Iterate.facet.loc48_19.5 [symbolic = %.loc48_19.24 (constants.%.e65)]
// CHECK:STDOUT:   %impl.elem3.loc48_19.2: @TestIterate.%.loc48_19.24 (%.e65) = impl_witness_access %Iterate.lookup_impl_witness, element3 [symbolic = %impl.elem3.loc48_19.2 (constants.%impl.elem3.b6f)]
// CHECK:STDOUT:   %specific_impl_fn.loc48_19.4: <specific function> = specific_impl_function %impl.elem3.loc48_19.2, @Iterate.WithSelf.Next(%Iterate.facet.loc48_19.5) [symbolic = %specific_impl_fn.loc48_19.4 (constants.%specific_impl_fn.f91)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%r.param: @TestIterate.%R.as_type.loc44_61.1 (%R.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %i.patt: %pattern_type.667 = value_binding_pattern i [concrete = constants.%i.patt]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %r.ref: @TestIterate.%R.as_type.loc44_61.1 (%R.as_type) = name_ref r, %r
// CHECK:STDOUT:     %impl.elem2.loc48_19.1: @TestIterate.%.loc48_19.23 (%.e6b) = impl_witness_access constants.%Iterate.lookup_impl_witness.ea0, element2 [symbolic = %impl.elem2.loc48_19.2 (constants.%impl.elem2.80a)]
// CHECK:STDOUT:     %bound_method.loc48_19.1: <bound method> = bound_method %r.ref, %impl.elem2.loc48_19.1
// CHECK:STDOUT:     %Iterate.facet.loc48_19.1: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.ea0) [symbolic = %Iterate.facet.loc48_19.5 (constants.%Iterate.facet.714)]
// CHECK:STDOUT:     %.loc48_19.1: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc48_19.1 [symbolic = %Iterate.facet.loc48_19.5 (constants.%Iterate.facet.714)]
// CHECK:STDOUT:     %Iterate.facet.loc48_19.2: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.ea0) [symbolic = %Iterate.facet.loc48_19.5 (constants.%Iterate.facet.714)]
// CHECK:STDOUT:     %.loc48_19.2: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc48_19.2 [symbolic = %Iterate.facet.loc48_19.5 (constants.%Iterate.facet.714)]
// CHECK:STDOUT:     %specific_impl_fn.loc48_19.1: <specific function> = specific_impl_function %impl.elem2.loc48_19.1, @Iterate.WithSelf.NewCursor(constants.%Iterate.facet.714) [symbolic = %specific_impl_fn.loc48_19.3 (constants.%specific_impl_fn.da8)]
// CHECK:STDOUT:     %bound_method.loc48_19.2: <bound method> = bound_method %r.ref, %specific_impl_fn.loc48_19.1
// CHECK:STDOUT:     %var: ref @TestIterate.%as_type (%as_type.2b7) = var_storage invalid
// CHECK:STDOUT:     %Iterate.WithSelf.NewCursor.call: init @TestIterate.%as_type (%as_type.2b7) to %var = call %bound_method.loc48_19.2(%r.ref)
// CHECK:STDOUT:     assign %var, %Iterate.WithSelf.NewCursor.call
// CHECK:STDOUT:     br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.next:
// CHECK:STDOUT:     %impl.elem3.loc48_19.1: @TestIterate.%.loc48_19.24 (%.e65) = impl_witness_access constants.%Iterate.lookup_impl_witness.ea0, element3 [symbolic = %impl.elem3.loc48_19.2 (constants.%impl.elem3.b6f)]
// CHECK:STDOUT:     %bound_method.loc48_19.3: <bound method> = bound_method %r.ref, %impl.elem3.loc48_19.1
// CHECK:STDOUT:     %Iterate.facet.loc48_19.3: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.ea0) [symbolic = %Iterate.facet.loc48_19.5 (constants.%Iterate.facet.714)]
// CHECK:STDOUT:     %.loc48_19.3: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc48_19.3 [symbolic = %Iterate.facet.loc48_19.5 (constants.%Iterate.facet.714)]
// CHECK:STDOUT:     %Iterate.facet.loc48_19.4: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.ea0) [symbolic = %Iterate.facet.loc48_19.5 (constants.%Iterate.facet.714)]
// CHECK:STDOUT:     %.loc48_19.4: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc48_19.4 [symbolic = %Iterate.facet.loc48_19.5 (constants.%Iterate.facet.714)]
// CHECK:STDOUT:     %.loc48_19.5: %facet_type.8a3 = converted constants.%as_type.2b7, constants.%impl.elem1.af2 [symbolic = %impl.elem1 (constants.%impl.elem1.af2)]
// CHECK:STDOUT:     %.loc48_19.6: %facet_type.8a3 = converted constants.%as_type.2b7, constants.%impl.elem1.af2 [symbolic = %impl.elem1 (constants.%impl.elem1.af2)]
// CHECK:STDOUT:     %specific_impl_fn.loc48_19.2: <specific function> = specific_impl_function %impl.elem3.loc48_19.1, @Iterate.WithSelf.Next(constants.%Iterate.facet.714) [symbolic = %specific_impl_fn.loc48_19.4 (constants.%specific_impl_fn.f91)]
// CHECK:STDOUT:     %bound_method.loc48_19.4: <bound method> = bound_method %r.ref, %specific_impl_fn.loc48_19.2
// CHECK:STDOUT:     %.loc48_19.7: ref %Optional.694 = temporary_storage
// CHECK:STDOUT:     %Iterate.WithSelf.Next.call: init %Optional.694 to %.loc48_19.7 = call %bound_method.loc48_19.4(%r.ref, %var)
// CHECK:STDOUT:     %.loc48_19.8: ref %Optional.694 = temporary %.loc48_19.7, %Iterate.WithSelf.Next.call
// CHECK:STDOUT:     %.loc48_19.9: %Optional.HasValue.type.eb1 = specific_constant imports.%Core.import_ref.c82, @Optional(constants.%OptionalStorage.facet.be8) [concrete = constants.%Optional.HasValue.ad6]
// CHECK:STDOUT:     %HasValue.ref: %Optional.HasValue.type.eb1 = name_ref HasValue, %.loc48_19.9 [concrete = constants.%Optional.HasValue.ad6]
// CHECK:STDOUT:     %Optional.HasValue.bound: <bound method> = bound_method %.loc48_19.8, %HasValue.ref
// CHECK:STDOUT:     %Optional.HasValue.specific_fn: <specific function> = specific_function %HasValue.ref, @Optional.HasValue(constants.%OptionalStorage.facet.be8) [concrete = constants.%Optional.HasValue.specific_fn.b33]
// CHECK:STDOUT:     %bound_method.loc48_19.5: <bound method> = bound_method %.loc48_19.8, %Optional.HasValue.specific_fn
// CHECK:STDOUT:     %.loc48_19.10: %Optional.694 = acquire_value %.loc48_19.8
// CHECK:STDOUT:     %Optional.HasValue.call: init bool = call %bound_method.loc48_19.5(%.loc48_19.10)
// CHECK:STDOUT:     %.loc48_19.11: bool = value_of_initializer %Optional.HasValue.call
// CHECK:STDOUT:     %.loc48_19.12: bool = converted %Optional.HasValue.call, %.loc48_19.11
// CHECK:STDOUT:     if %.loc48_19.12 br !for.body else br !for.done
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.body:
// CHECK:STDOUT:     %.loc48_19.13: %Optional.Get.type.f48 = specific_constant imports.%Core.import_ref.29e, @Optional(constants.%OptionalStorage.facet.be8) [concrete = constants.%Optional.Get.999]
// CHECK:STDOUT:     %Get.ref: %Optional.Get.type.f48 = name_ref Get, %.loc48_19.13 [concrete = constants.%Optional.Get.999]
// CHECK:STDOUT:     %Optional.Get.bound: <bound method> = bound_method %.loc48_19.8, %Get.ref
// CHECK:STDOUT:     %Optional.Get.specific_fn: <specific function> = specific_function %Get.ref, @Optional.Get(constants.%OptionalStorage.facet.be8) [concrete = constants.%Optional.Get.specific_fn.aca]
// CHECK:STDOUT:     %bound_method.loc48_19.6: <bound method> = bound_method %.loc48_19.8, %Optional.Get.specific_fn
// CHECK:STDOUT:     %.loc48_19.14: %Optional.694 = acquire_value %.loc48_19.8
// CHECK:STDOUT:     %Optional.Get.call: init %f64.761 = call %bound_method.loc48_19.6(%.loc48_19.14)
// CHECK:STDOUT:     %.loc48_19.15: %f64.761 = value_of_initializer %Optional.Get.call
// CHECK:STDOUT:     %.loc48_19.16: %f64.761 = converted %Optional.Get.call, %.loc48_19.15
// CHECK:STDOUT:     %f64.loc48: type = type_literal constants.%f64.761 [concrete = constants.%f64.761]
// CHECK:STDOUT:     %i: %f64.761 = wrapper_binding i, %.loc48_19.16
// CHECK:STDOUT:     %sum.ref: ref %f64.761 = name_ref sum, %sum
// CHECK:STDOUT:     %i.ref: %f64.761 = name_ref i, %i
// CHECK:STDOUT:     %impl.elem0.loc49: %.48b = impl_witness_access constants.%AddAssignWith.impl_witness.5c7, element0 [concrete = constants.%Float.as.AddAssignWith.impl.Op.e04]
// CHECK:STDOUT:     %bound_method.loc49_9.1: <bound method> = bound_method %sum.ref, %impl.elem0.loc49
// CHECK:STDOUT:     %specific_fn.loc49: <specific function> = specific_function %impl.elem0.loc49, @Float.as.AddAssignWith.impl.Op(constants.%int_64) [concrete = constants.%Float.as.AddAssignWith.impl.Op.specific_fn]
// CHECK:STDOUT:     %bound_method.loc49_9.2: <bound method> = bound_method %sum.ref, %specific_fn.loc49
// CHECK:STDOUT:     %Float.as.AddAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc49_9.2(%sum.ref, %i.ref)
// CHECK:STDOUT:     %facet_value.loc48_19.1: %facet_type.8a3 = facet_value constants.%Optional.694, (constants.%custom_witness.b80ef6.9, constants.%DestroyImpl.impl_witness.dcd) [concrete = constants.%facet_value.2ba]
// CHECK:STDOUT:     %.loc48_19.17: %facet_type.8a3 = converted constants.%Optional.694, %facet_value.loc48_19.1 [concrete = constants.%facet_value.2ba]
// CHECK:STDOUT:     %facet_value.loc48_19.2: %facet_type.8a3 = facet_value constants.%Optional.694, (constants.%custom_witness.b80ef6.9, constants.%DestroyImpl.impl_witness.dcd) [concrete = constants.%facet_value.2ba]
// CHECK:STDOUT:     %.loc48_19.18: %facet_type.8a3 = converted constants.%Optional.694, %facet_value.loc48_19.2 [concrete = constants.%facet_value.2ba]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %SelfDestruct.call.loc48_19.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc48_19.8)
// CHECK:STDOUT:     br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.done:
// CHECK:STDOUT:     %facet_value.loc48_19.3: %facet_type.8a3 = facet_value constants.%Optional.694, (constants.%custom_witness.b80ef6.9, constants.%DestroyImpl.impl_witness.dcd) [concrete = constants.%facet_value.2ba]
// CHECK:STDOUT:     %.loc48_19.19: %facet_type.8a3 = converted constants.%Optional.694, %facet_value.loc48_19.3 [concrete = constants.%facet_value.2ba]
// CHECK:STDOUT:     %facet_value.loc48_19.4: %facet_type.8a3 = facet_value constants.%Optional.694, (constants.%custom_witness.b80ef6.9, constants.%DestroyImpl.impl_witness.dcd) [concrete = constants.%facet_value.2ba]
// CHECK:STDOUT:     %.loc48_19.20: %facet_type.8a3 = converted constants.%Optional.694, %facet_value.loc48_19.4 [concrete = constants.%facet_value.2ba]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %SelfDestruct.call.loc48_19.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc48_19.8)
// CHECK:STDOUT:     %.loc48_19.21: %facet_type.8a3 = converted constants.%as_type.2b7, constants.%impl.elem1.af2 [symbolic = %impl.elem1 (constants.%impl.elem1.af2)]
// CHECK:STDOUT:     %.loc48_19.22: %facet_type.8a3 = converted constants.%as_type.2b7, constants.%impl.elem1.af2 [symbolic = %impl.elem1 (constants.%impl.elem1.af2)]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %SelfDestruct.call.loc48_19.3: init %empty_tuple.type = call %SelfDestruct.specific_fn.3(%var)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc48_19.1(%self.param: ref %.ecc) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc48_19.2(%self.param: ref %MaybeUnformed.ab3) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc48_19.3(%self.param: ref bool) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc48_19.4(%self.param: ref %struct_type.value.has_value.4c7) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc48_19.5(%self.param: ref %DefaultOptionalStorage.f1a) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc48_19.6(%self.param: ref %Optional.694) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestRangeFor(%m.param: ref %MutableRange) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %i.patt: %pattern_type.667 = value_binding_pattern i [concrete = constants.%i.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %m.ref: ref %MutableRange = name_ref m, %m
// CHECK:STDOUT:   %impl.elem2: %.d6e = impl_witness_access constants.%Iterate.impl_witness.752, element2 [concrete = constants.%T.as_type.as.Iterate.impl.NewCursor.482]
// CHECK:STDOUT:   %bound_method.loc62_19.1: <bound method> = bound_method %m.ref, %impl.elem2
// CHECK:STDOUT:   %facet_value.loc62_19.1: %CppRange_where.type.d682d6.1 = facet_value constants.%MutableRange, (constants.%custom_witness.902, constants.%custom_witness.b80ef6.10, constants.%DestroyImpl.impl_witness.452, constants.%custom_witness.0a8, constants.%custom_witness.491, constants.%custom_witness.068, constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.a23, constants.%Copy.impl_witness.eac, constants.%custom_witness.b80ef6.10, constants.%DestroyImpl.impl_witness.452) [concrete = constants.%facet_value.fd4]
// CHECK:STDOUT:   %.loc62_19.1: %CppRange_where.type.d682d6.1 = converted constants.%MutableRange, %facet_value.loc62_19.1 [concrete = constants.%facet_value.fd4]
// CHECK:STDOUT:   %facet_value.loc62_19.2: %CppRange_where.type.d682d6.1 = facet_value constants.%MutableRange, (constants.%custom_witness.902, constants.%custom_witness.b80ef6.10, constants.%DestroyImpl.impl_witness.452, constants.%custom_witness.0a8, constants.%custom_witness.491, constants.%custom_witness.068, constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.a23, constants.%Copy.impl_witness.eac, constants.%custom_witness.b80ef6.10, constants.%DestroyImpl.impl_witness.452) [concrete = constants.%facet_value.fd4]
// CHECK:STDOUT:   %.loc62_19.2: %CppRange_where.type.d682d6.1 = converted constants.%MutableRange, %facet_value.loc62_19.2 [concrete = constants.%facet_value.fd4]
// CHECK:STDOUT:   %specific_fn.loc62_19.1: <specific function> = specific_function %impl.elem2, @T.as_type.as.Iterate.impl.NewCursor(constants.%facet_value.fd4) [concrete = constants.%T.as_type.as.Iterate.impl.NewCursor.specific_fn]
// CHECK:STDOUT:   %bound_method.loc62_19.2: <bound method> = bound_method %m.ref, %specific_fn.loc62_19.1
// CHECK:STDOUT:   %var: ref %tuple.type.dd0 = var_storage invalid
// CHECK:STDOUT:   %.loc62_18.1: %MutableRange = acquire_value %m.ref
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.call: init %tuple.type.dd0 to %var = call %bound_method.loc62_19.2(%.loc62_18.1)
// CHECK:STDOUT:   assign %var, %T.as_type.as.Iterate.impl.NewCursor.call
// CHECK:STDOUT:   br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT: !for.next:
// CHECK:STDOUT:   %impl.elem3: %.446 = impl_witness_access constants.%Iterate.impl_witness.752, element3 [concrete = constants.%T.as_type.as.Iterate.impl.Next.759]
// CHECK:STDOUT:   %bound_method.loc62_19.3: <bound method> = bound_method %m.ref, %impl.elem3
// CHECK:STDOUT:   %facet_value.loc62_19.3: %CppRange_where.type.d682d6.1 = facet_value constants.%MutableRange, (constants.%custom_witness.902, constants.%custom_witness.b80ef6.10, constants.%DestroyImpl.impl_witness.452, constants.%custom_witness.0a8, constants.%custom_witness.491, constants.%custom_witness.068, constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.a23, constants.%Copy.impl_witness.eac, constants.%custom_witness.b80ef6.10, constants.%DestroyImpl.impl_witness.452) [concrete = constants.%facet_value.fd4]
// CHECK:STDOUT:   %.loc62_19.3: %CppRange_where.type.d682d6.1 = converted constants.%MutableRange, %facet_value.loc62_19.3 [concrete = constants.%facet_value.fd4]
// CHECK:STDOUT:   %facet_value.loc62_19.4: %CppRange_where.type.d682d6.1 = facet_value constants.%MutableRange, (constants.%custom_witness.902, constants.%custom_witness.b80ef6.10, constants.%DestroyImpl.impl_witness.452, constants.%custom_witness.0a8, constants.%custom_witness.491, constants.%custom_witness.068, constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.a23, constants.%Copy.impl_witness.eac, constants.%custom_witness.b80ef6.10, constants.%DestroyImpl.impl_witness.452) [concrete = constants.%facet_value.fd4]
// CHECK:STDOUT:   %.loc62_19.4: %CppRange_where.type.d682d6.1 = converted constants.%MutableRange, %facet_value.loc62_19.4 [concrete = constants.%facet_value.fd4]
// CHECK:STDOUT:   %specific_fn.loc62_19.2: <specific function> = specific_function %impl.elem3, @T.as_type.as.Iterate.impl.Next(constants.%facet_value.fd4) [concrete = constants.%T.as_type.as.Iterate.impl.Next.specific_fn]
// CHECK:STDOUT:   %bound_method.loc62_19.4: <bound method> = bound_method %m.ref, %specific_fn.loc62_19.2
// CHECK:STDOUT:   %.loc62_19.5: ref %Optional.694 = temporary_storage
// CHECK:STDOUT:   %.loc62_18.2: %MutableRange = acquire_value %m.ref
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.call: init %Optional.694 to %.loc62_19.5 = call %bound_method.loc62_19.4(%.loc62_18.2, %var)
// CHECK:STDOUT:   %.loc62_19.6: ref %Optional.694 = temporary %.loc62_19.5, %T.as_type.as.Iterate.impl.Next.call
// CHECK:STDOUT:   %.loc62_19.7: %Optional.HasValue.type.eb1 = specific_constant imports.%Core.import_ref.c82, @Optional(constants.%OptionalStorage.facet.be8) [concrete = constants.%Optional.HasValue.ad6]
// CHECK:STDOUT:   %HasValue.ref: %Optional.HasValue.type.eb1 = name_ref HasValue, %.loc62_19.7 [concrete = constants.%Optional.HasValue.ad6]
// CHECK:STDOUT:   %Optional.HasValue.bound: <bound method> = bound_method %.loc62_19.6, %HasValue.ref
// CHECK:STDOUT:   %Optional.HasValue.specific_fn: <specific function> = specific_function %HasValue.ref, @Optional.HasValue(constants.%OptionalStorage.facet.be8) [concrete = constants.%Optional.HasValue.specific_fn.b33]
// CHECK:STDOUT:   %bound_method.loc62_19.5: <bound method> = bound_method %.loc62_19.6, %Optional.HasValue.specific_fn
// CHECK:STDOUT:   %.loc62_19.8: %Optional.694 = acquire_value %.loc62_19.6
// CHECK:STDOUT:   %Optional.HasValue.call: init bool = call %bound_method.loc62_19.5(%.loc62_19.8)
// CHECK:STDOUT:   %.loc62_19.9: bool = value_of_initializer %Optional.HasValue.call
// CHECK:STDOUT:   %.loc62_19.10: bool = converted %Optional.HasValue.call, %.loc62_19.9
// CHECK:STDOUT:   if %.loc62_19.10 br !for.body else br !for.done
// CHECK:STDOUT:
// CHECK:STDOUT: !for.body:
// CHECK:STDOUT:   %.loc62_19.11: %Optional.Get.type.f48 = specific_constant imports.%Core.import_ref.29e, @Optional(constants.%OptionalStorage.facet.be8) [concrete = constants.%Optional.Get.999]
// CHECK:STDOUT:   %Get.ref: %Optional.Get.type.f48 = name_ref Get, %.loc62_19.11 [concrete = constants.%Optional.Get.999]
// CHECK:STDOUT:   %Optional.Get.bound: <bound method> = bound_method %.loc62_19.6, %Get.ref
// CHECK:STDOUT:   %Optional.Get.specific_fn: <specific function> = specific_function %Get.ref, @Optional.Get(constants.%OptionalStorage.facet.be8) [concrete = constants.%Optional.Get.specific_fn.aca]
// CHECK:STDOUT:   %bound_method.loc62_19.6: <bound method> = bound_method %.loc62_19.6, %Optional.Get.specific_fn
// CHECK:STDOUT:   %.loc62_19.12: %Optional.694 = acquire_value %.loc62_19.6
// CHECK:STDOUT:   %Optional.Get.call: init %f64.761 = call %bound_method.loc62_19.6(%.loc62_19.12)
// CHECK:STDOUT:   %.loc62_19.13: %f64.761 = value_of_initializer %Optional.Get.call
// CHECK:STDOUT:   %.loc62_19.14: %f64.761 = converted %Optional.Get.call, %.loc62_19.13
// CHECK:STDOUT:   %f64.loc62: type = type_literal constants.%f64.761 [concrete = constants.%f64.761]
// CHECK:STDOUT:   %i: %f64.761 = wrapper_binding i, %.loc62_19.14
// CHECK:STDOUT:   %sum.ref: ref %f64.761 = name_ref sum, %sum
// CHECK:STDOUT:   %i.ref: %f64.761 = name_ref i, %i
// CHECK:STDOUT:   %impl.elem0.loc63: %.48b = impl_witness_access constants.%AddAssignWith.impl_witness.5c7, element0 [concrete = constants.%Float.as.AddAssignWith.impl.Op.e04]
// CHECK:STDOUT:   %bound_method.loc63_9.1: <bound method> = bound_method %sum.ref, %impl.elem0.loc63
// CHECK:STDOUT:   %specific_fn.loc63: <specific function> = specific_function %impl.elem0.loc63, @Float.as.AddAssignWith.impl.Op(constants.%int_64) [concrete = constants.%Float.as.AddAssignWith.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc63_9.2: <bound method> = bound_method %sum.ref, %specific_fn.loc63
// CHECK:STDOUT:   %Float.as.AddAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc63_9.2(%sum.ref, %i.ref)
// CHECK:STDOUT:   %facet_value.loc62_19.5: %facet_type.8a3 = facet_value constants.%Optional.694, (constants.%custom_witness.b80ef6.9, constants.%DestroyImpl.impl_witness.dcd) [concrete = constants.%facet_value.2ba]
// CHECK:STDOUT:   %.loc62_19.15: %facet_type.8a3 = converted constants.%Optional.694, %facet_value.loc62_19.5 [concrete = constants.%facet_value.2ba]
// CHECK:STDOUT:   %facet_value.loc62_19.6: %facet_type.8a3 = facet_value constants.%Optional.694, (constants.%custom_witness.b80ef6.9, constants.%DestroyImpl.impl_witness.dcd) [concrete = constants.%facet_value.2ba]
// CHECK:STDOUT:   %.loc62_19.16: %facet_type.8a3 = converted constants.%Optional.694, %facet_value.loc62_19.6 [concrete = constants.%facet_value.2ba]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc62_19.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc62_19.6)
// CHECK:STDOUT:   br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT: !for.done:
// CHECK:STDOUT:   %facet_value.loc62_19.7: %facet_type.8a3 = facet_value constants.%Optional.694, (constants.%custom_witness.b80ef6.9, constants.%DestroyImpl.impl_witness.dcd) [concrete = constants.%facet_value.2ba]
// CHECK:STDOUT:   %.loc62_19.17: %facet_type.8a3 = converted constants.%Optional.694, %facet_value.loc62_19.7 [concrete = constants.%facet_value.2ba]
// CHECK:STDOUT:   %facet_value.loc62_19.8: %facet_type.8a3 = facet_value constants.%Optional.694, (constants.%custom_witness.b80ef6.9, constants.%DestroyImpl.impl_witness.dcd) [concrete = constants.%facet_value.2ba]
// CHECK:STDOUT:   %.loc62_19.18: %facet_type.8a3 = converted constants.%Optional.694, %facet_value.loc62_19.8 [concrete = constants.%facet_value.2ba]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc62_19.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc62_19.6)
// CHECK:STDOUT:   %facet_value.loc62_19.9: %facet_type.8a3 = facet_value constants.%tuple.type.dd0, (constants.%custom_witness.b80ef6.11, constants.%DestroyImpl.impl_witness.1fa) [concrete = constants.%facet_value.efd]
// CHECK:STDOUT:   %.loc62_19.19: %facet_type.8a3 = converted constants.%tuple.type.dd0, %facet_value.loc62_19.9 [concrete = constants.%facet_value.efd]
// CHECK:STDOUT:   %facet_value.loc62_19.10: %facet_type.8a3 = facet_value constants.%tuple.type.dd0, (constants.%custom_witness.b80ef6.11, constants.%DestroyImpl.impl_witness.1fa) [concrete = constants.%facet_value.efd]
// CHECK:STDOUT:   %.loc62_19.20: %facet_type.8a3 = converted constants.%tuple.type.dd0, %facet_value.loc62_19.10 [concrete = constants.%facet_value.efd]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc62_19.3: init %empty_tuple.type = call %SelfDestruct.specific_fn.3(%var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestIterate(constants.%R) {
// CHECK:STDOUT:   %R.patt.loc44_17.2 => constants.%R.patt
// CHECK:STDOUT:   %R.loc44_17.1 => constants.%R
// CHECK:STDOUT:   %R.as_type.loc44_61.1 => constants.%R.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.639
// CHECK:STDOUT:   %r.param_patt.loc44_59.2 => constants.%r.param_patt.cf2
// CHECK:STDOUT:   %r.patt.loc44_59.2 => constants.%r.patt.927
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestIterate(constants.%facet_value.9af) {
// CHECK:STDOUT:   %R.patt.loc44_17.2 => constants.%R.patt
// CHECK:STDOUT:   %R.loc44_17.1 => constants.%facet_value.9af
// CHECK:STDOUT:   %R.as_type.loc44_61.1 => constants.%MutableRange
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.992
// CHECK:STDOUT:   %r.param_patt.loc44_59.2 => constants.%r.param_patt.aeb
// CHECK:STDOUT:   %r.patt.loc44_59.2 => constants.%r.patt.892
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc44 => constants.%complete_type.357
// CHECK:STDOUT:   %Iterate.lookup_impl_witness => constants.%Iterate.impl_witness.752
// CHECK:STDOUT:   %Iterate.facet.loc48_19.5 => constants.%Iterate.facet.c03
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type => constants.%Iterate.WithSelf.NewCursor.type.b9c
// CHECK:STDOUT:   %.loc48_19.23 => constants.%.d6e
// CHECK:STDOUT:   %impl.elem2.loc48_19.2 => constants.%T.as_type.as.Iterate.impl.NewCursor.482
// CHECK:STDOUT:   %specific_impl_fn.loc48_19.3 => constants.%T.as_type.as.Iterate.impl.NewCursor.specific_fn
// CHECK:STDOUT:   %impl.elem1 => constants.%facet_value.efd
// CHECK:STDOUT:   %as_type => constants.%tuple.type.dd0
// CHECK:STDOUT:   %require_complete.1 => constants.%complete_type.843
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type => constants.%Iterate.WithSelf.Next.type.692
// CHECK:STDOUT:   %.loc48_19.24 => constants.%.446
// CHECK:STDOUT:   %impl.elem3.loc48_19.2 => constants.%T.as_type.as.Iterate.impl.Next.759
// CHECK:STDOUT:   %specific_impl_fn.loc48_19.4 => constants.%T.as_type.as.Iterate.impl.Next.specific_fn
// CHECK:STDOUT:   %SelfDestruct.specific_fn.5 => constants.%SelfDestruct.specific_fn.4af
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
// CHECK:STDOUT:   %Optional.Get.type.786: type = fn_type @Optional.Get, @Optional(%T.220) [symbolic]
// CHECK:STDOUT:   %Optional.Get.2d0: %Optional.Get.type.786 = struct_value () [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.type.3ea: type = fn_type @Optional.HasValue, @Optional(%T.220) [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.f2f: %Optional.HasValue.type.3ea = struct_value () [symbolic]
// CHECK:STDOUT:   %facet_type.8a3: type = facet_type <@SubobjectDestroy & @DestroyImpl> [concrete]
// CHECK:STDOUT:   %T.9ab: %facet_type.d61 = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.f15: type = class_type @DefaultOptionalStorage, @DefaultOptionalStorage(%T.9ab) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Get.type.3e7: type = fn_type @T.as_type.as.OptionalStorage.impl.Get, @T.as_type.as.OptionalStorage.impl(%T.9ab) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Get.f80: %T.as_type.as.OptionalStorage.impl.Get.type.3e7 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Has.type.371: type = fn_type @T.as_type.as.OptionalStorage.impl.Has, @T.as_type.as.OptionalStorage.impl(%T.9ab) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Has.8ce: %T.as_type.as.OptionalStorage.impl.Has.type.371 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Some.type.4eb: type = fn_type @T.as_type.as.OptionalStorage.impl.Some, @T.as_type.as.OptionalStorage.impl(%T.9ab) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Some.44a: %T.as_type.as.OptionalStorage.impl.Some.type.4eb = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.None.type.76a: type = fn_type @T.as_type.as.OptionalStorage.impl.None, @T.as_type.as.OptionalStorage.impl(%T.9ab) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.None.93b: %T.as_type.as.OptionalStorage.impl.None.type.76a = struct_value () [symbolic]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
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
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.2: type = fn_type @SubobjectDestroy.WithSelf.Op.loc27 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.2: %SubobjectDestroy.WithSelf.Op.type.6870fd.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.2: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.2), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.251: %SubobjectDestroy.type = facet_value %ValueType, (%custom_witness.b80ef6.2) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.0a0: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.251) [concrete]
// CHECK:STDOUT:   %const.dd6: type = const_type %ValueType [concrete]
// CHECK:STDOUT:   %ptr.69c: type = ptr_type %const.dd6 [concrete]
// CHECK:STDOUT:   %ptr.233: type = ptr_type %ValueType [concrete]
// CHECK:STDOUT:   %ValueType.Op.type: type = fn_type @ValueType.Op [concrete]
// CHECK:STDOUT:   %ValueType.Op: %ValueType.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.202: <witness> = custom_witness (%ValueType.Op), @Copy [concrete]
// CHECK:STDOUT:   %facet_value.dce: %facet_type.d61 = facet_value %ValueType, (%custom_witness.b80ef6.2, %DestroyImpl.impl_witness.0a0, %custom_witness.202) [concrete]
// CHECK:STDOUT:   %Iterate.lookup_impl_witness.1d8: <witness> = lookup_impl_witness %.Self.500, @Iterate [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.8a4: %facet_type.d61 = impl_witness_access %Iterate.lookup_impl_witness.1d8, element0 [symbolic_self]
// CHECK:STDOUT:   %Iterate_where.type.415: type = facet_type <@Iterate where %impl.elem0.8a4 = %facet_value.dce> [concrete]
// CHECK:STDOUT:   %pattern_type.e44: type = pattern_type %Iterate_where.type.415 [concrete]
// CHECK:STDOUT:   %R.patt: %pattern_type.e44 = symbolic_binding_pattern R, 0 [symbolic]
// CHECK:STDOUT:   %R: %Iterate_where.type.415 = symbolic_binding R, 0 [symbolic]
// CHECK:STDOUT:   %R.as_type: type = facet_access_type %R [symbolic]
// CHECK:STDOUT:   %pattern_type.561: type = pattern_type %R.as_type [symbolic]
// CHECK:STDOUT:   %r.param_patt.05c: %pattern_type.561 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %r.patt.bdb: %pattern_type.561 = wrapper_binding_pattern r, %r.param_patt.05c [symbolic]
// CHECK:STDOUT:   %i.patt: %pattern_type.c43 = value_binding_pattern i [concrete]
// CHECK:STDOUT:   %Iterate.lookup_impl_witness.c3d: <witness> = lookup_impl_witness %R, @Iterate [symbolic]
// CHECK:STDOUT:   %Iterate.facet.8cd: %Iterate.type = facet_value %R.as_type, (%Iterate.lookup_impl_witness.c3d) [symbolic]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.ffc: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.8cd) [symbolic]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.3d4: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.8cd) [symbolic]
// CHECK:STDOUT:   %.3db: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.ffc, %Iterate.facet.8cd [symbolic]
// CHECK:STDOUT:   %impl.elem2.6fa: %.3db = impl_witness_access %Iterate.lookup_impl_witness.c3d, element2 [symbolic]
// CHECK:STDOUT:   %impl.elem1.d3f: %facet_type.8a3 = impl_witness_access %Iterate.lookup_impl_witness.c3d, element1 [symbolic]
// CHECK:STDOUT:   %as_type.338: type = facet_access_type %impl.elem1.d3f [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.fc2: <specific function> = specific_impl_function %impl.elem2.6fa, @Iterate.WithSelf.NewCursor(%Iterate.facet.8cd) [symbolic]
// CHECK:STDOUT:   %.557: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.3d4, %Iterate.facet.8cd [symbolic]
// CHECK:STDOUT:   %impl.elem3.9f4: %.557 = impl_witness_access %Iterate.lookup_impl_witness.c3d, element3 [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.75a: type = class_type @DefaultOptionalStorage, @DefaultOptionalStorage(%facet_value.dce) [concrete]
// CHECK:STDOUT:   %OptionalStorage.impl_witness.e36: <witness> = impl_witness imports.%OptionalStorage.impl_witness_table.968, @T.as_type.as.OptionalStorage.impl(%facet_value.dce) [concrete]
// CHECK:STDOUT:   %OptionalStorage.facet.afc: %OptionalStorage.type = facet_value %ValueType, (%OptionalStorage.impl_witness.e36) [concrete]
// CHECK:STDOUT:   %Optional.e74: type = class_type @Optional, @Optional(%OptionalStorage.facet.afc) [concrete]
// CHECK:STDOUT:   %specific_impl_fn.ad6: <specific function> = specific_impl_function %impl.elem3.9f4, @Iterate.WithSelf.Next(%Iterate.facet.8cd) [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.type.f61: type = fn_type @Optional.HasValue, @Optional(%OptionalStorage.facet.afc) [concrete]
// CHECK:STDOUT:   %Optional.HasValue.1fb: %Optional.HasValue.type.f61 = struct_value () [concrete]
// CHECK:STDOUT:   %Optional.Get.type.247: type = fn_type @Optional.Get, @Optional(%OptionalStorage.facet.afc) [concrete]
// CHECK:STDOUT:   %Optional.Get.594: %Optional.Get.type.247 = struct_value () [concrete]
// CHECK:STDOUT:   %facet_value.1c4: %facet_type.8a3 = facet_value %ValueType, (%custom_witness.b80ef6.2, %DestroyImpl.impl_witness.0a0) [concrete]
// CHECK:STDOUT:   %MaybeUnformed.d5b: type = class_type @MaybeUnformed, @MaybeUnformed(%facet_value.1c4) [concrete]
// CHECK:STDOUT:   %.be1: type = maybe_unformed_type %ValueType [concrete]
// CHECK:STDOUT:   %struct_type.value.has_value.495: type = struct_type {.value: %MaybeUnformed.d5b, .has_value: bool} [concrete]
// CHECK:STDOUT:   %Optional.HasValue.specific_fn.a80: <specific function> = specific_function %Optional.HasValue.1fb, @Optional.HasValue(%OptionalStorage.facet.afc) [concrete]
// CHECK:STDOUT:   %Optional.Get.specific_fn.8f8: <specific function> = specific_function %Optional.Get.594, @Optional.Get(%OptionalStorage.facet.afc) [concrete]
// CHECK:STDOUT:   %operator_PlusEqual__carbon_thunk.type: type = fn_type @operator_PlusEqual__carbon_thunk [concrete]
// CHECK:STDOUT:   %operator_PlusEqual__carbon_thunk: %operator_PlusEqual__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.8: type = fn_type @SubobjectDestroy.WithSelf.Op.loc31_37.6 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.8: %SubobjectDestroy.WithSelf.Op.type.6870fd.8 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.8: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.8), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.d5d: %SubobjectDestroy.type = facet_value %Optional.e74, (%custom_witness.b80ef6.8) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.186: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.d5d) [concrete]
// CHECK:STDOUT:   %facet_value.8b1: %facet_type.8a3 = facet_value %Optional.e74, (%custom_witness.b80ef6.8, %DestroyImpl.impl_witness.186) [concrete]
// CHECK:STDOUT:   %ConstRange: type = class_type @ConstRange [concrete]
// CHECK:STDOUT:   %pattern_type.c1d: type = pattern_type %ConstRange [concrete]
// CHECK:STDOUT:   %Iterator: type = class_type @Iterator.1 [concrete]
// CHECK:STDOUT:   %Sentinel: type = class_type @Sentinel.1 [concrete]
// CHECK:STDOUT:   %Begin.type: type = fn_type @Begin [concrete]
// CHECK:STDOUT:   %Begin: %Begin.type = struct_value () [concrete]
// CHECK:STDOUT:   %End.type: type = fn_type @End [concrete]
// CHECK:STDOUT:   %End: %End.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.116: <witness> = custom_witness (%Iterator, %Sentinel, %Begin, %End), @CppRangeForIterate [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.9: type = fn_type @SubobjectDestroy.WithSelf.Op.loc38_16.1 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.9: %SubobjectDestroy.WithSelf.Op.type.6870fd.9 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.9: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.9), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.ce7: %SubobjectDestroy.type = facet_value %Iterator, (%custom_witness.b80ef6.9) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.f28: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.ce7) [concrete]
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
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.10: type = fn_type @SubobjectDestroy.WithSelf.Op.loc38_16.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.10: %SubobjectDestroy.WithSelf.Op.type.6870fd.10 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.10: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.10), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.543: %SubobjectDestroy.type = facet_value %Sentinel, (%custom_witness.b80ef6.10) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.c16: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.543) [concrete]
// CHECK:STDOUT:   %facet_value.d10: %CppRange_where.type.d682d6.1 = facet_value %ConstRange, (%custom_witness.116, %custom_witness.b80ef6.9, %DestroyImpl.impl_witness.f28, %custom_witness.5be, %custom_witness.c31, %custom_witness.bf4, %custom_witness.b80ef6.2, %DestroyImpl.impl_witness.0a0, %custom_witness.202, %custom_witness.b80ef6.10, %DestroyImpl.impl_witness.c16) [concrete]
// CHECK:STDOUT:   %tuple.type.7a6: type = tuple_type (%Iterator, %Sentinel) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.11: type = fn_type @SubobjectDestroy.WithSelf.Op.3 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.11: %SubobjectDestroy.WithSelf.Op.type.6870fd.11 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.11: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.11), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.b14: %SubobjectDestroy.type = facet_value %tuple.type.7a6, (%custom_witness.b80ef6.11) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.b22: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.b14) [concrete]
// CHECK:STDOUT:   %complete_type.cf0: <witness> = complete_type_witness %tuple.type.7a6 [concrete]
// CHECK:STDOUT:   %facet_value.2f0: %facet_type.8a3 = facet_value %tuple.type.7a6, (%custom_witness.b80ef6.11, %DestroyImpl.impl_witness.b22) [concrete]
// CHECK:STDOUT:   %Iterate.impl_witness.491: <witness> = impl_witness imports.%Iterate.impl_witness_table.121, @T.as_type.as.Iterate.impl(%facet_value.d10) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.type.fef: type = fn_type @T.as_type.as.Iterate.impl.NewCursor, @T.as_type.as.Iterate.impl(%facet_value.d10) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.886: %T.as_type.as.Iterate.impl.NewCursor.type.fef = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.type.d72: type = fn_type @T.as_type.as.Iterate.impl.Next, @T.as_type.as.Iterate.impl(%facet_value.d10) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.2ea: %T.as_type.as.Iterate.impl.Next.type.d72 = struct_value () [concrete]
// CHECK:STDOUT:   %Iterate.facet.7a3: %Iterate.type = facet_value %ConstRange, (%Iterate.impl_witness.491) [concrete]
// CHECK:STDOUT:   %facet_value.c12: %Iterate_where.type.415 = facet_value %ConstRange, (%Iterate.impl_witness.491) [concrete]
// CHECK:STDOUT:   %r.param_patt.ac5: %pattern_type.c1d = value_param_pattern [concrete]
// CHECK:STDOUT:   %r.patt.97f: %pattern_type.c1d = wrapper_binding_pattern r, %r.param_patt.ac5 [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.1d5: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.7a3) [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.362: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.7a3) [concrete]
// CHECK:STDOUT:   %.83a: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.1d5, %Iterate.facet.7a3 [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.specific_fn: <specific function> = specific_function %T.as_type.as.Iterate.impl.NewCursor.886, @T.as_type.as.Iterate.impl.NewCursor(%facet_value.d10) [concrete]
// CHECK:STDOUT:   %.18a: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.362, %Iterate.facet.7a3 [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.specific_fn: <specific function> = specific_function %T.as_type.as.Iterate.impl.Next.2ea, @T.as_type.as.Iterate.impl.Next(%facet_value.d10) [concrete]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.00d: <specific function> = specific_function %SelfDestruct, @SelfDestruct(%facet_value.2f0) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .DiffADL = %DiffADL
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.c82: @Optional.%Optional.HasValue.type (%Optional.HasValue.type.3ea) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.HasValue (constants.%Optional.HasValue.f2f)]
// CHECK:STDOUT:   %Core.import_ref.29e: @Optional.%Optional.Get.type (%Optional.Get.type.786) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.Get (constants.%Optional.Get.2d0)]
// CHECK:STDOUT:   %Core.import_ref.9a31: type = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%DefaultOptionalStorage (constants.%DefaultOptionalStorage.f15)]
// CHECK:STDOUT:   %Core.import_ref.819a: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.None.type (%T.as_type.as.OptionalStorage.impl.None.type.76a) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.None (constants.%T.as_type.as.OptionalStorage.impl.None.93b)]
// CHECK:STDOUT:   %Core.import_ref.14a: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Some.type (%T.as_type.as.OptionalStorage.impl.Some.type.4eb) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Some (constants.%T.as_type.as.OptionalStorage.impl.Some.44a)]
// CHECK:STDOUT:   %Core.import_ref.242: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Has.type (%T.as_type.as.OptionalStorage.impl.Has.type.371) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Has (constants.%T.as_type.as.OptionalStorage.impl.Has.8ce)]
// CHECK:STDOUT:   %Core.import_ref.a54: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Get.type (%T.as_type.as.OptionalStorage.impl.Get.type.3e7) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Get (constants.%T.as_type.as.OptionalStorage.impl.Get.f80)]
// CHECK:STDOUT:   %Core.import_ref.5bf = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], unloaded
// CHECK:STDOUT:   %OptionalStorage.impl_witness_table.968 = impl_witness_table (%Core.import_ref.9a31, %Core.import_ref.819a, %Core.import_ref.14a, %Core.import_ref.242, %Core.import_ref.a54, %Core.import_ref.5bf), @T.as_type.as.OptionalStorage.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.822: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.822), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.d6c: %facet_type.d61 = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.Iterate.impl.%facet_value.1 (constants.%facet_value.9da)]
// CHECK:STDOUT:   %Core.import_ref.2ac: %facet_type.8a3 = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.Iterate.impl.%facet_value.2 (constants.%facet_value.bf6)]
// CHECK:STDOUT:   %Core.import_ref.946: @T.as_type.as.Iterate.impl.%T.as_type.as.Iterate.impl.NewCursor.type (%T.as_type.as.Iterate.impl.NewCursor.type.8fb) = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.Iterate.impl.%T.as_type.as.Iterate.impl.NewCursor (constants.%T.as_type.as.Iterate.impl.NewCursor.8a1)]
// CHECK:STDOUT:   %Core.import_ref.cfb: @T.as_type.as.Iterate.impl.%T.as_type.as.Iterate.impl.Next.type (%T.as_type.as.Iterate.impl.Next.type.d4a) = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.Iterate.impl.%T.as_type.as.Iterate.impl.Next (constants.%T.as_type.as.Iterate.impl.Next.5a8)]
// CHECK:STDOUT:   %Iterate.impl_witness_table.121 = impl_witness_table (%Core.import_ref.d6c, %Core.import_ref.2ac, %Core.import_ref.946, %Core.import_ref.cfb), @T.as_type.as.Iterate.impl [concrete]
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
// CHECK:STDOUT: generic fn @TestIterate(%R.loc27_17.2: %Iterate_where.type.415) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Iterate.lookup_impl_witness: <witness> = lookup_impl_witness %R.loc27_17.1, @Iterate [symbolic = %Iterate.lookup_impl_witness (constants.%Iterate.lookup_impl_witness.c3d)]
// CHECK:STDOUT:   %Iterate.facet.loc31_37.5: %Iterate.type = facet_value %R.as_type.loc27_79.1, (%Iterate.lookup_impl_witness) [symbolic = %Iterate.facet.loc31_37.5 (constants.%Iterate.facet.8cd)]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.loc31_37.5) [symbolic = %Iterate.WithSelf.NewCursor.type (constants.%Iterate.WithSelf.NewCursor.type.ffc)]
// CHECK:STDOUT:   %.loc31_37.26: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type, %Iterate.facet.loc31_37.5 [symbolic = %.loc31_37.26 (constants.%.3db)]
// CHECK:STDOUT:   %impl.elem2.loc31_37.2: @TestIterate.%.loc31_37.26 (%.3db) = impl_witness_access %Iterate.lookup_impl_witness, element2 [symbolic = %impl.elem2.loc31_37.2 (constants.%impl.elem2.6fa)]
// CHECK:STDOUT:   %specific_impl_fn.loc31_37.3: <specific function> = specific_impl_function %impl.elem2.loc31_37.2, @Iterate.WithSelf.NewCursor(%Iterate.facet.loc31_37.5) [symbolic = %specific_impl_fn.loc31_37.3 (constants.%specific_impl_fn.fc2)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.loc31_37.5) [symbolic = %Iterate.WithSelf.Next.type (constants.%Iterate.WithSelf.Next.type.3d4)]
// CHECK:STDOUT:   %.loc31_37.27: type = fn_type_with_self_type %Iterate.WithSelf.Next.type, %Iterate.facet.loc31_37.5 [symbolic = %.loc31_37.27 (constants.%.557)]
// CHECK:STDOUT:   %impl.elem3.loc31_37.2: @TestIterate.%.loc31_37.27 (%.557) = impl_witness_access %Iterate.lookup_impl_witness, element3 [symbolic = %impl.elem3.loc31_37.2 (constants.%impl.elem3.9f4)]
// CHECK:STDOUT:   %specific_impl_fn.loc31_37.4: <specific function> = specific_impl_function %impl.elem3.loc31_37.2, @Iterate.WithSelf.Next(%Iterate.facet.loc31_37.5) [symbolic = %specific_impl_fn.loc31_37.4 (constants.%specific_impl_fn.ad6)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%r.param: @TestIterate.%R.as_type.loc27_79.1 (%R.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %i.patt: %pattern_type.c43 = value_binding_pattern i [concrete = constants.%i.patt]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %r.ref: @TestIterate.%R.as_type.loc27_79.1 (%R.as_type) = name_ref r, %r
// CHECK:STDOUT:     %impl.elem2.loc31_37.1: @TestIterate.%.loc31_37.26 (%.3db) = impl_witness_access constants.%Iterate.lookup_impl_witness.c3d, element2 [symbolic = %impl.elem2.loc31_37.2 (constants.%impl.elem2.6fa)]
// CHECK:STDOUT:     %bound_method.loc31_37.1: <bound method> = bound_method %r.ref, %impl.elem2.loc31_37.1
// CHECK:STDOUT:     %Iterate.facet.loc31_37.1: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.c3d) [symbolic = %Iterate.facet.loc31_37.5 (constants.%Iterate.facet.8cd)]
// CHECK:STDOUT:     %.loc31_37.1: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc31_37.1 [symbolic = %Iterate.facet.loc31_37.5 (constants.%Iterate.facet.8cd)]
// CHECK:STDOUT:     %Iterate.facet.loc31_37.2: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.c3d) [symbolic = %Iterate.facet.loc31_37.5 (constants.%Iterate.facet.8cd)]
// CHECK:STDOUT:     %.loc31_37.2: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc31_37.2 [symbolic = %Iterate.facet.loc31_37.5 (constants.%Iterate.facet.8cd)]
// CHECK:STDOUT:     %specific_impl_fn.loc31_37.1: <specific function> = specific_impl_function %impl.elem2.loc31_37.1, @Iterate.WithSelf.NewCursor(constants.%Iterate.facet.8cd) [symbolic = %specific_impl_fn.loc31_37.3 (constants.%specific_impl_fn.fc2)]
// CHECK:STDOUT:     %bound_method.loc31_37.2: <bound method> = bound_method %r.ref, %specific_impl_fn.loc31_37.1
// CHECK:STDOUT:     %var: ref @TestIterate.%as_type (%as_type.338) = var_storage invalid
// CHECK:STDOUT:     %Iterate.WithSelf.NewCursor.call: init @TestIterate.%as_type (%as_type.338) to %var = call %bound_method.loc31_37.2(%r.ref)
// CHECK:STDOUT:     assign %var, %Iterate.WithSelf.NewCursor.call
// CHECK:STDOUT:     br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.next:
// CHECK:STDOUT:     %impl.elem3.loc31_37.1: @TestIterate.%.loc31_37.27 (%.557) = impl_witness_access constants.%Iterate.lookup_impl_witness.c3d, element3 [symbolic = %impl.elem3.loc31_37.2 (constants.%impl.elem3.9f4)]
// CHECK:STDOUT:     %bound_method.loc31_37.3: <bound method> = bound_method %r.ref, %impl.elem3.loc31_37.1
// CHECK:STDOUT:     %Iterate.facet.loc31_37.3: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.c3d) [symbolic = %Iterate.facet.loc31_37.5 (constants.%Iterate.facet.8cd)]
// CHECK:STDOUT:     %.loc31_37.3: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc31_37.3 [symbolic = %Iterate.facet.loc31_37.5 (constants.%Iterate.facet.8cd)]
// CHECK:STDOUT:     %Iterate.facet.loc31_37.4: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.c3d) [symbolic = %Iterate.facet.loc31_37.5 (constants.%Iterate.facet.8cd)]
// CHECK:STDOUT:     %.loc31_37.4: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc31_37.4 [symbolic = %Iterate.facet.loc31_37.5 (constants.%Iterate.facet.8cd)]
// CHECK:STDOUT:     %.loc31_37.5: %facet_type.8a3 = converted constants.%as_type.338, constants.%impl.elem1.d3f [symbolic = %impl.elem1 (constants.%impl.elem1.d3f)]
// CHECK:STDOUT:     %.loc31_37.6: %facet_type.8a3 = converted constants.%as_type.338, constants.%impl.elem1.d3f [symbolic = %impl.elem1 (constants.%impl.elem1.d3f)]
// CHECK:STDOUT:     %specific_impl_fn.loc31_37.2: <specific function> = specific_impl_function %impl.elem3.loc31_37.1, @Iterate.WithSelf.Next(constants.%Iterate.facet.8cd) [symbolic = %specific_impl_fn.loc31_37.4 (constants.%specific_impl_fn.ad6)]
// CHECK:STDOUT:     %bound_method.loc31_37.4: <bound method> = bound_method %r.ref, %specific_impl_fn.loc31_37.2
// CHECK:STDOUT:     %.loc31_37.7: ref %Optional.e74 = temporary_storage
// CHECK:STDOUT:     %Iterate.WithSelf.Next.call: init %Optional.e74 to %.loc31_37.7 = call %bound_method.loc31_37.4(%r.ref, %var)
// CHECK:STDOUT:     %.loc31_37.8: ref %Optional.e74 = temporary %.loc31_37.7, %Iterate.WithSelf.Next.call
// CHECK:STDOUT:     %.loc31_37.9: %Optional.HasValue.type.f61 = specific_constant imports.%Core.import_ref.c82, @Optional(constants.%OptionalStorage.facet.afc) [concrete = constants.%Optional.HasValue.1fb]
// CHECK:STDOUT:     %HasValue.ref: %Optional.HasValue.type.f61 = name_ref HasValue, %.loc31_37.9 [concrete = constants.%Optional.HasValue.1fb]
// CHECK:STDOUT:     %Optional.HasValue.bound: <bound method> = bound_method %.loc31_37.8, %HasValue.ref
// CHECK:STDOUT:     %Optional.HasValue.specific_fn: <specific function> = specific_function %HasValue.ref, @Optional.HasValue(constants.%OptionalStorage.facet.afc) [concrete = constants.%Optional.HasValue.specific_fn.a80]
// CHECK:STDOUT:     %bound_method.loc31_37.5: <bound method> = bound_method %.loc31_37.8, %Optional.HasValue.specific_fn
// CHECK:STDOUT:     %.loc31_37.10: %Optional.e74 = acquire_value %.loc31_37.8
// CHECK:STDOUT:     %Optional.HasValue.call: init bool = call %bound_method.loc31_37.5(%.loc31_37.10)
// CHECK:STDOUT:     %.loc31_37.11: bool = value_of_initializer %Optional.HasValue.call
// CHECK:STDOUT:     %.loc31_37.12: bool = converted %Optional.HasValue.call, %.loc31_37.11
// CHECK:STDOUT:     if %.loc31_37.12 br !for.body else br !for.done
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.body:
// CHECK:STDOUT:     %.loc31_37.13: %Optional.Get.type.247 = specific_constant imports.%Core.import_ref.29e, @Optional(constants.%OptionalStorage.facet.afc) [concrete = constants.%Optional.Get.594]
// CHECK:STDOUT:     %Get.ref: %Optional.Get.type.247 = name_ref Get, %.loc31_37.13 [concrete = constants.%Optional.Get.594]
// CHECK:STDOUT:     %Optional.Get.bound: <bound method> = bound_method %.loc31_37.8, %Get.ref
// CHECK:STDOUT:     %Optional.Get.specific_fn: <specific function> = specific_function %Get.ref, @Optional.Get(constants.%OptionalStorage.facet.afc) [concrete = constants.%Optional.Get.specific_fn.8f8]
// CHECK:STDOUT:     %bound_method.loc31_37.6: <bound method> = bound_method %.loc31_37.8, %Optional.Get.specific_fn
// CHECK:STDOUT:     %.loc31_37.14: ref %ValueType = temporary_storage
// CHECK:STDOUT:     %.loc31_37.15: %Optional.e74 = acquire_value %.loc31_37.8
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
// CHECK:STDOUT:     %facet_value.loc31_37.1: %facet_type.8a3 = facet_value constants.%ValueType, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.0a0) [concrete = constants.%facet_value.1c4]
// CHECK:STDOUT:     %.loc31_37.18: %facet_type.8a3 = converted constants.%ValueType, %facet_value.loc31_37.1 [concrete = constants.%facet_value.1c4]
// CHECK:STDOUT:     %facet_value.loc31_37.2: %facet_type.8a3 = facet_value constants.%ValueType, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.0a0) [concrete = constants.%facet_value.1c4]
// CHECK:STDOUT:     %.loc31_37.19: %facet_type.8a3 = converted constants.%ValueType, %facet_value.loc31_37.2 [concrete = constants.%facet_value.1c4]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %SelfDestruct.call.loc31_37.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc31_37.16)
// CHECK:STDOUT:     %facet_value.loc31_37.3: %facet_type.8a3 = facet_value constants.%Optional.e74, (constants.%custom_witness.b80ef6.8, constants.%DestroyImpl.impl_witness.186) [concrete = constants.%facet_value.8b1]
// CHECK:STDOUT:     %.loc31_37.20: %facet_type.8a3 = converted constants.%Optional.e74, %facet_value.loc31_37.3 [concrete = constants.%facet_value.8b1]
// CHECK:STDOUT:     %facet_value.loc31_37.4: %facet_type.8a3 = facet_value constants.%Optional.e74, (constants.%custom_witness.b80ef6.8, constants.%DestroyImpl.impl_witness.186) [concrete = constants.%facet_value.8b1]
// CHECK:STDOUT:     %.loc31_37.21: %facet_type.8a3 = converted constants.%Optional.e74, %facet_value.loc31_37.4 [concrete = constants.%facet_value.8b1]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %SelfDestruct.call.loc31_37.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc31_37.8)
// CHECK:STDOUT:     br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.done:
// CHECK:STDOUT:     %facet_value.loc31_37.5: %facet_type.8a3 = facet_value constants.%Optional.e74, (constants.%custom_witness.b80ef6.8, constants.%DestroyImpl.impl_witness.186) [concrete = constants.%facet_value.8b1]
// CHECK:STDOUT:     %.loc31_37.22: %facet_type.8a3 = converted constants.%Optional.e74, %facet_value.loc31_37.5 [concrete = constants.%facet_value.8b1]
// CHECK:STDOUT:     %facet_value.loc31_37.6: %facet_type.8a3 = facet_value constants.%Optional.e74, (constants.%custom_witness.b80ef6.8, constants.%DestroyImpl.impl_witness.186) [concrete = constants.%facet_value.8b1]
// CHECK:STDOUT:     %.loc31_37.23: %facet_type.8a3 = converted constants.%Optional.e74, %facet_value.loc31_37.6 [concrete = constants.%facet_value.8b1]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %SelfDestruct.call.loc31_37.3: init %empty_tuple.type = call %SelfDestruct.specific_fn.3(%.loc31_37.8)
// CHECK:STDOUT:     %.loc31_37.24: %facet_type.8a3 = converted constants.%as_type.338, constants.%impl.elem1.d3f [symbolic = %impl.elem1 (constants.%impl.elem1.d3f)]
// CHECK:STDOUT:     %.loc31_37.25: %facet_type.8a3 = converted constants.%as_type.338, constants.%impl.elem1.d3f [symbolic = %impl.elem1 (constants.%impl.elem1.d3f)]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %SelfDestruct.call.loc31_37.4: init %empty_tuple.type = call %SelfDestruct.specific_fn.4(%var)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc31_37.1(%self.param: ref %.be1) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc31_37.2(%self.param: ref %MaybeUnformed.d5b) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc31_37.3(%self.param: ref bool) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc31_37.4(%self.param: ref %struct_type.value.has_value.495) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc31_37.5(%self.param: ref %DefaultOptionalStorage.75a) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc31_37.6(%self.param: ref %Optional.e74) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestRangeFor(%c.param: %ConstRange) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %i.patt: %pattern_type.c43 = value_binding_pattern i [concrete = constants.%i.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c.ref: %ConstRange = name_ref c, %c
// CHECK:STDOUT:   %impl.elem2: %.83a = impl_witness_access constants.%Iterate.impl_witness.491, element2 [concrete = constants.%T.as_type.as.Iterate.impl.NewCursor.886]
// CHECK:STDOUT:   %bound_method.loc45_37.1: <bound method> = bound_method %c.ref, %impl.elem2
// CHECK:STDOUT:   %facet_value.loc45_37.1: %CppRange_where.type.d682d6.1 = facet_value constants.%ConstRange, (constants.%custom_witness.116, constants.%custom_witness.b80ef6.9, constants.%DestroyImpl.impl_witness.f28, constants.%custom_witness.5be, constants.%custom_witness.c31, constants.%custom_witness.bf4, constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.0a0, constants.%custom_witness.202, constants.%custom_witness.b80ef6.10, constants.%DestroyImpl.impl_witness.c16) [concrete = constants.%facet_value.d10]
// CHECK:STDOUT:   %.loc45_37.1: %CppRange_where.type.d682d6.1 = converted constants.%ConstRange, %facet_value.loc45_37.1 [concrete = constants.%facet_value.d10]
// CHECK:STDOUT:   %facet_value.loc45_37.2: %CppRange_where.type.d682d6.1 = facet_value constants.%ConstRange, (constants.%custom_witness.116, constants.%custom_witness.b80ef6.9, constants.%DestroyImpl.impl_witness.f28, constants.%custom_witness.5be, constants.%custom_witness.c31, constants.%custom_witness.bf4, constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.0a0, constants.%custom_witness.202, constants.%custom_witness.b80ef6.10, constants.%DestroyImpl.impl_witness.c16) [concrete = constants.%facet_value.d10]
// CHECK:STDOUT:   %.loc45_37.2: %CppRange_where.type.d682d6.1 = converted constants.%ConstRange, %facet_value.loc45_37.2 [concrete = constants.%facet_value.d10]
// CHECK:STDOUT:   %specific_fn.loc45_37.1: <specific function> = specific_function %impl.elem2, @T.as_type.as.Iterate.impl.NewCursor(constants.%facet_value.d10) [concrete = constants.%T.as_type.as.Iterate.impl.NewCursor.specific_fn]
// CHECK:STDOUT:   %bound_method.loc45_37.2: <bound method> = bound_method %c.ref, %specific_fn.loc45_37.1
// CHECK:STDOUT:   %var: ref %tuple.type.7a6 = var_storage invalid
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.call: init %tuple.type.7a6 to %var = call %bound_method.loc45_37.2(%c.ref)
// CHECK:STDOUT:   assign %var, %T.as_type.as.Iterate.impl.NewCursor.call
// CHECK:STDOUT:   br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT: !for.next:
// CHECK:STDOUT:   %impl.elem3: %.18a = impl_witness_access constants.%Iterate.impl_witness.491, element3 [concrete = constants.%T.as_type.as.Iterate.impl.Next.2ea]
// CHECK:STDOUT:   %bound_method.loc45_37.3: <bound method> = bound_method %c.ref, %impl.elem3
// CHECK:STDOUT:   %facet_value.loc45_37.3: %CppRange_where.type.d682d6.1 = facet_value constants.%ConstRange, (constants.%custom_witness.116, constants.%custom_witness.b80ef6.9, constants.%DestroyImpl.impl_witness.f28, constants.%custom_witness.5be, constants.%custom_witness.c31, constants.%custom_witness.bf4, constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.0a0, constants.%custom_witness.202, constants.%custom_witness.b80ef6.10, constants.%DestroyImpl.impl_witness.c16) [concrete = constants.%facet_value.d10]
// CHECK:STDOUT:   %.loc45_37.3: %CppRange_where.type.d682d6.1 = converted constants.%ConstRange, %facet_value.loc45_37.3 [concrete = constants.%facet_value.d10]
// CHECK:STDOUT:   %facet_value.loc45_37.4: %CppRange_where.type.d682d6.1 = facet_value constants.%ConstRange, (constants.%custom_witness.116, constants.%custom_witness.b80ef6.9, constants.%DestroyImpl.impl_witness.f28, constants.%custom_witness.5be, constants.%custom_witness.c31, constants.%custom_witness.bf4, constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.0a0, constants.%custom_witness.202, constants.%custom_witness.b80ef6.10, constants.%DestroyImpl.impl_witness.c16) [concrete = constants.%facet_value.d10]
// CHECK:STDOUT:   %.loc45_37.4: %CppRange_where.type.d682d6.1 = converted constants.%ConstRange, %facet_value.loc45_37.4 [concrete = constants.%facet_value.d10]
// CHECK:STDOUT:   %specific_fn.loc45_37.2: <specific function> = specific_function %impl.elem3, @T.as_type.as.Iterate.impl.Next(constants.%facet_value.d10) [concrete = constants.%T.as_type.as.Iterate.impl.Next.specific_fn]
// CHECK:STDOUT:   %bound_method.loc45_37.4: <bound method> = bound_method %c.ref, %specific_fn.loc45_37.2
// CHECK:STDOUT:   %.loc45_37.5: ref %Optional.e74 = temporary_storage
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.call: init %Optional.e74 to %.loc45_37.5 = call %bound_method.loc45_37.4(%c.ref, %var)
// CHECK:STDOUT:   %.loc45_37.6: ref %Optional.e74 = temporary %.loc45_37.5, %T.as_type.as.Iterate.impl.Next.call
// CHECK:STDOUT:   %.loc45_37.7: %Optional.HasValue.type.f61 = specific_constant imports.%Core.import_ref.c82, @Optional(constants.%OptionalStorage.facet.afc) [concrete = constants.%Optional.HasValue.1fb]
// CHECK:STDOUT:   %HasValue.ref: %Optional.HasValue.type.f61 = name_ref HasValue, %.loc45_37.7 [concrete = constants.%Optional.HasValue.1fb]
// CHECK:STDOUT:   %Optional.HasValue.bound: <bound method> = bound_method %.loc45_37.6, %HasValue.ref
// CHECK:STDOUT:   %Optional.HasValue.specific_fn: <specific function> = specific_function %HasValue.ref, @Optional.HasValue(constants.%OptionalStorage.facet.afc) [concrete = constants.%Optional.HasValue.specific_fn.a80]
// CHECK:STDOUT:   %bound_method.loc45_37.5: <bound method> = bound_method %.loc45_37.6, %Optional.HasValue.specific_fn
// CHECK:STDOUT:   %.loc45_37.8: %Optional.e74 = acquire_value %.loc45_37.6
// CHECK:STDOUT:   %Optional.HasValue.call: init bool = call %bound_method.loc45_37.5(%.loc45_37.8)
// CHECK:STDOUT:   %.loc45_37.9: bool = value_of_initializer %Optional.HasValue.call
// CHECK:STDOUT:   %.loc45_37.10: bool = converted %Optional.HasValue.call, %.loc45_37.9
// CHECK:STDOUT:   if %.loc45_37.10 br !for.body else br !for.done
// CHECK:STDOUT:
// CHECK:STDOUT: !for.body:
// CHECK:STDOUT:   %.loc45_37.11: %Optional.Get.type.247 = specific_constant imports.%Core.import_ref.29e, @Optional(constants.%OptionalStorage.facet.afc) [concrete = constants.%Optional.Get.594]
// CHECK:STDOUT:   %Get.ref: %Optional.Get.type.247 = name_ref Get, %.loc45_37.11 [concrete = constants.%Optional.Get.594]
// CHECK:STDOUT:   %Optional.Get.bound: <bound method> = bound_method %.loc45_37.6, %Get.ref
// CHECK:STDOUT:   %Optional.Get.specific_fn: <specific function> = specific_function %Get.ref, @Optional.Get(constants.%OptionalStorage.facet.afc) [concrete = constants.%Optional.Get.specific_fn.8f8]
// CHECK:STDOUT:   %bound_method.loc45_37.6: <bound method> = bound_method %.loc45_37.6, %Optional.Get.specific_fn
// CHECK:STDOUT:   %.loc45_37.12: ref %ValueType = temporary_storage
// CHECK:STDOUT:   %.loc45_37.13: %Optional.e74 = acquire_value %.loc45_37.6
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
// CHECK:STDOUT:   %facet_value.loc45_37.5: %facet_type.8a3 = facet_value constants.%ValueType, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.0a0) [concrete = constants.%facet_value.1c4]
// CHECK:STDOUT:   %.loc45_37.16: %facet_type.8a3 = converted constants.%ValueType, %facet_value.loc45_37.5 [concrete = constants.%facet_value.1c4]
// CHECK:STDOUT:   %facet_value.loc45_37.6: %facet_type.8a3 = facet_value constants.%ValueType, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.0a0) [concrete = constants.%facet_value.1c4]
// CHECK:STDOUT:   %.loc45_37.17: %facet_type.8a3 = converted constants.%ValueType, %facet_value.loc45_37.6 [concrete = constants.%facet_value.1c4]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc45_37.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc45_37.14)
// CHECK:STDOUT:   %facet_value.loc45_37.7: %facet_type.8a3 = facet_value constants.%Optional.e74, (constants.%custom_witness.b80ef6.8, constants.%DestroyImpl.impl_witness.186) [concrete = constants.%facet_value.8b1]
// CHECK:STDOUT:   %.loc45_37.18: %facet_type.8a3 = converted constants.%Optional.e74, %facet_value.loc45_37.7 [concrete = constants.%facet_value.8b1]
// CHECK:STDOUT:   %facet_value.loc45_37.8: %facet_type.8a3 = facet_value constants.%Optional.e74, (constants.%custom_witness.b80ef6.8, constants.%DestroyImpl.impl_witness.186) [concrete = constants.%facet_value.8b1]
// CHECK:STDOUT:   %.loc45_37.19: %facet_type.8a3 = converted constants.%Optional.e74, %facet_value.loc45_37.8 [concrete = constants.%facet_value.8b1]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc45_37.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc45_37.6)
// CHECK:STDOUT:   br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT: !for.done:
// CHECK:STDOUT:   %facet_value.loc45_37.9: %facet_type.8a3 = facet_value constants.%Optional.e74, (constants.%custom_witness.b80ef6.8, constants.%DestroyImpl.impl_witness.186) [concrete = constants.%facet_value.8b1]
// CHECK:STDOUT:   %.loc45_37.20: %facet_type.8a3 = converted constants.%Optional.e74, %facet_value.loc45_37.9 [concrete = constants.%facet_value.8b1]
// CHECK:STDOUT:   %facet_value.loc45_37.10: %facet_type.8a3 = facet_value constants.%Optional.e74, (constants.%custom_witness.b80ef6.8, constants.%DestroyImpl.impl_witness.186) [concrete = constants.%facet_value.8b1]
// CHECK:STDOUT:   %.loc45_37.21: %facet_type.8a3 = converted constants.%Optional.e74, %facet_value.loc45_37.10 [concrete = constants.%facet_value.8b1]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc45_37.3: init %empty_tuple.type = call %SelfDestruct.specific_fn.3(%.loc45_37.6)
// CHECK:STDOUT:   %facet_value.loc45_37.11: %facet_type.8a3 = facet_value constants.%tuple.type.7a6, (constants.%custom_witness.b80ef6.11, constants.%DestroyImpl.impl_witness.b22) [concrete = constants.%facet_value.2f0]
// CHECK:STDOUT:   %.loc45_37.22: %facet_type.8a3 = converted constants.%tuple.type.7a6, %facet_value.loc45_37.11 [concrete = constants.%facet_value.2f0]
// CHECK:STDOUT:   %facet_value.loc45_37.12: %facet_type.8a3 = facet_value constants.%tuple.type.7a6, (constants.%custom_witness.b80ef6.11, constants.%DestroyImpl.impl_witness.b22) [concrete = constants.%facet_value.2f0]
// CHECK:STDOUT:   %.loc45_37.23: %facet_type.8a3 = converted constants.%tuple.type.7a6, %facet_value.loc45_37.12 [concrete = constants.%facet_value.2f0]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc45_37.4: init %empty_tuple.type = call %SelfDestruct.specific_fn.4(%var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestIterate(constants.%R) {
// CHECK:STDOUT:   %R.patt.loc27_17.2 => constants.%R.patt
// CHECK:STDOUT:   %R.loc27_17.1 => constants.%R
// CHECK:STDOUT:   %R.as_type.loc27_79.1 => constants.%R.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.561
// CHECK:STDOUT:   %r.param_patt.loc27_77.2 => constants.%r.param_patt.05c
// CHECK:STDOUT:   %r.patt.loc27_77.2 => constants.%r.patt.bdb
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestIterate(constants.%facet_value.c12) {
// CHECK:STDOUT:   %R.patt.loc27_17.2 => constants.%R.patt
// CHECK:STDOUT:   %R.loc27_17.1 => constants.%facet_value.c12
// CHECK:STDOUT:   %R.as_type.loc27_79.1 => constants.%ConstRange
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.c1d
// CHECK:STDOUT:   %r.param_patt.loc27_77.2 => constants.%r.param_patt.ac5
// CHECK:STDOUT:   %r.patt.loc27_77.2 => constants.%r.patt.97f
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc27 => constants.%complete_type.357
// CHECK:STDOUT:   %Iterate.lookup_impl_witness => constants.%Iterate.impl_witness.491
// CHECK:STDOUT:   %Iterate.facet.loc31_37.5 => constants.%Iterate.facet.7a3
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type => constants.%Iterate.WithSelf.NewCursor.type.1d5
// CHECK:STDOUT:   %.loc31_37.26 => constants.%.83a
// CHECK:STDOUT:   %impl.elem2.loc31_37.2 => constants.%T.as_type.as.Iterate.impl.NewCursor.886
// CHECK:STDOUT:   %specific_impl_fn.loc31_37.3 => constants.%T.as_type.as.Iterate.impl.NewCursor.specific_fn
// CHECK:STDOUT:   %impl.elem1 => constants.%facet_value.2f0
// CHECK:STDOUT:   %as_type => constants.%tuple.type.7a6
// CHECK:STDOUT:   %require_complete.1 => constants.%complete_type.cf0
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type => constants.%Iterate.WithSelf.Next.type.362
// CHECK:STDOUT:   %.loc31_37.27 => constants.%.18a
// CHECK:STDOUT:   %impl.elem3.loc31_37.2 => constants.%T.as_type.as.Iterate.impl.Next.2ea
// CHECK:STDOUT:   %specific_impl_fn.loc31_37.4 => constants.%T.as_type.as.Iterate.impl.Next.specific_fn
// CHECK:STDOUT:   %SelfDestruct.specific_fn.6 => constants.%SelfDestruct.specific_fn.00d
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
// CHECK:STDOUT:   %Optional.Get.type.786: type = fn_type @Optional.Get, @Optional(%T.220) [symbolic]
// CHECK:STDOUT:   %Optional.Get.2d0: %Optional.Get.type.786 = struct_value () [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.type.3ea: type = fn_type @Optional.HasValue, @Optional(%T.220) [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.f2f: %Optional.HasValue.type.3ea = struct_value () [symbolic]
// CHECK:STDOUT:   %facet_type.8a3: type = facet_type <@SubobjectDestroy & @DestroyImpl> [concrete]
// CHECK:STDOUT:   %T.9ab: %facet_type.d61 = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.f15: type = class_type @DefaultOptionalStorage, @DefaultOptionalStorage(%T.9ab) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Get.type.3e7: type = fn_type @T.as_type.as.OptionalStorage.impl.Get, @T.as_type.as.OptionalStorage.impl(%T.9ab) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Get.f80: %T.as_type.as.OptionalStorage.impl.Get.type.3e7 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Has.type.371: type = fn_type @T.as_type.as.OptionalStorage.impl.Has, @T.as_type.as.OptionalStorage.impl(%T.9ab) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Has.8ce: %T.as_type.as.OptionalStorage.impl.Has.type.371 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %.Self.500: %Iterate.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %ValueType: type = class_type @ValueType [concrete]
// CHECK:STDOUT:   %pattern_type.d2d: type = pattern_type %ValueType [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.2: type = fn_type @SubobjectDestroy.WithSelf.Op.loc49 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.2: %SubobjectDestroy.WithSelf.Op.type.6870fd.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.2: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.2), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.d5a: %SubobjectDestroy.type = facet_value %ValueType, (%custom_witness.b80ef6.2) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.127: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.d5a) [concrete]
// CHECK:STDOUT:   %const.91f: type = const_type %ValueType [concrete]
// CHECK:STDOUT:   %ptr.5f4: type = ptr_type %const.91f [concrete]
// CHECK:STDOUT:   %ptr.9c6: type = ptr_type %ValueType [concrete]
// CHECK:STDOUT:   %ValueType.Op.type: type = fn_type @ValueType.Op [concrete]
// CHECK:STDOUT:   %ValueType.Op: %ValueType.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.587: <witness> = custom_witness (%ValueType.Op), @Copy [concrete]
// CHECK:STDOUT:   %facet_value.2bf: %facet_type.d61 = facet_value %ValueType, (%custom_witness.b80ef6.2, %DestroyImpl.impl_witness.127, %custom_witness.587) [concrete]
// CHECK:STDOUT:   %Iterate.lookup_impl_witness.1d8: <witness> = lookup_impl_witness %.Self.500, @Iterate [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.8a4: %facet_type.d61 = impl_witness_access %Iterate.lookup_impl_witness.1d8, element0 [symbolic_self]
// CHECK:STDOUT:   %Iterate_where.type.470: type = facet_type <@Iterate where %impl.elem0.8a4 = %facet_value.2bf> [concrete]
// CHECK:STDOUT:   %pattern_type.b76: type = pattern_type %Iterate_where.type.470 [concrete]
// CHECK:STDOUT:   %R.patt: %pattern_type.b76 = symbolic_binding_pattern R, 0 [symbolic]
// CHECK:STDOUT:   %R: %Iterate_where.type.470 = symbolic_binding R, 0 [symbolic]
// CHECK:STDOUT:   %R.as_type: type = facet_access_type %R [symbolic]
// CHECK:STDOUT:   %pattern_type.62c: type = pattern_type %R.as_type [symbolic]
// CHECK:STDOUT:   %r.param_patt: %pattern_type.62c = value_param_pattern [symbolic]
// CHECK:STDOUT:   %r.patt: %pattern_type.62c = wrapper_binding_pattern r, %r.param_patt [symbolic]
// CHECK:STDOUT:   %i.patt: %pattern_type.d2d = value_binding_pattern i [concrete]
// CHECK:STDOUT:   %Iterate.lookup_impl_witness.9ae: <witness> = lookup_impl_witness %R, @Iterate [symbolic]
// CHECK:STDOUT:   %Iterate.facet.c23: %Iterate.type = facet_value %R.as_type, (%Iterate.lookup_impl_witness.9ae) [symbolic]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.1fa: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.c23) [symbolic]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.a94: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.c23) [symbolic]
// CHECK:STDOUT:   %.88f: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.1fa, %Iterate.facet.c23 [symbolic]
// CHECK:STDOUT:   %impl.elem2.ef4: %.88f = impl_witness_access %Iterate.lookup_impl_witness.9ae, element2 [symbolic]
// CHECK:STDOUT:   %impl.elem1.ef7: %facet_type.8a3 = impl_witness_access %Iterate.lookup_impl_witness.9ae, element1 [symbolic]
// CHECK:STDOUT:   %as_type.458: type = facet_access_type %impl.elem1.ef7 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.72f: <specific function> = specific_impl_function %impl.elem2.ef4, @Iterate.WithSelf.NewCursor(%Iterate.facet.c23) [symbolic]
// CHECK:STDOUT:   %.a42: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.a94, %Iterate.facet.c23 [symbolic]
// CHECK:STDOUT:   %impl.elem3.164: %.a42 = impl_witness_access %Iterate.lookup_impl_witness.9ae, element3 [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.f91: type = class_type @DefaultOptionalStorage, @DefaultOptionalStorage(%facet_value.2bf) [concrete]
// CHECK:STDOUT:   %OptionalStorage.impl_witness.43a: <witness> = impl_witness imports.%OptionalStorage.impl_witness_table.968, @T.as_type.as.OptionalStorage.impl(%facet_value.2bf) [concrete]
// CHECK:STDOUT:   %OptionalStorage.facet.a50: %OptionalStorage.type = facet_value %ValueType, (%OptionalStorage.impl_witness.43a) [concrete]
// CHECK:STDOUT:   %Optional.8d7: type = class_type @Optional, @Optional(%OptionalStorage.facet.a50) [concrete]
// CHECK:STDOUT:   %specific_impl_fn.84d: <specific function> = specific_impl_function %impl.elem3.164, @Iterate.WithSelf.Next(%Iterate.facet.c23) [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.type.491: type = fn_type @Optional.HasValue, @Optional(%OptionalStorage.facet.a50) [concrete]
// CHECK:STDOUT:   %Optional.HasValue.163: %Optional.HasValue.type.491 = struct_value () [concrete]
// CHECK:STDOUT:   %Optional.Get.type.935: type = fn_type @Optional.Get, @Optional(%OptionalStorage.facet.a50) [concrete]
// CHECK:STDOUT:   %Optional.Get.ce9: %Optional.Get.type.935 = struct_value () [concrete]
// CHECK:STDOUT:   %facet_value.cfd: %facet_type.8a3 = facet_value %ValueType, (%custom_witness.b80ef6.2, %DestroyImpl.impl_witness.127) [concrete]
// CHECK:STDOUT:   %MaybeUnformed.547: type = class_type @MaybeUnformed, @MaybeUnformed(%facet_value.cfd) [concrete]
// CHECK:STDOUT:   %.dd1: type = maybe_unformed_type %ValueType [concrete]
// CHECK:STDOUT:   %struct_type.value.has_value.64c: type = struct_type {.value: %MaybeUnformed.547, .has_value: bool} [concrete]
// CHECK:STDOUT:   %Optional.HasValue.specific_fn: <specific function> = specific_function %Optional.HasValue.163, @Optional.HasValue(%OptionalStorage.facet.a50) [concrete]
// CHECK:STDOUT:   %Optional.Get.specific_fn: <specific function> = specific_function %Optional.Get.ce9, @Optional.Get(%OptionalStorage.facet.a50) [concrete]
// CHECK:STDOUT:   %operator_PlusEqual__carbon_thunk.type: type = fn_type @operator_PlusEqual__carbon_thunk [concrete]
// CHECK:STDOUT:   %operator_PlusEqual__carbon_thunk: %operator_PlusEqual__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.8: type = fn_type @SubobjectDestroy.WithSelf.Op.loc53_44.6 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.8: %SubobjectDestroy.WithSelf.Op.type.6870fd.8 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.8: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.8), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.372: %SubobjectDestroy.type = facet_value %Optional.8d7, (%custom_witness.b80ef6.8) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.b9d: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.372) [concrete]
// CHECK:STDOUT:   %facet_value.4d8: %facet_type.8a3 = facet_value %Optional.8d7, (%custom_witness.b80ef6.8, %DestroyImpl.impl_witness.b9d) [concrete]
// CHECK:STDOUT:   %MutableRange: type = class_type @MutableRange [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .DiffMutableADL = %DiffMutableADL
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.c82: @Optional.%Optional.HasValue.type (%Optional.HasValue.type.3ea) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.HasValue (constants.%Optional.HasValue.f2f)]
// CHECK:STDOUT:   %Core.import_ref.29e: @Optional.%Optional.Get.type (%Optional.Get.type.786) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.Get (constants.%Optional.Get.2d0)]
// CHECK:STDOUT:   %Core.import_ref.9a31: type = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%DefaultOptionalStorage (constants.%DefaultOptionalStorage.f15)]
// CHECK:STDOUT:   %Core.import_ref.505 = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], unloaded
// CHECK:STDOUT:   %Core.import_ref.71ea = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], unloaded
// CHECK:STDOUT:   %Core.import_ref.242: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Has.type (%T.as_type.as.OptionalStorage.impl.Has.type.371) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Has (constants.%T.as_type.as.OptionalStorage.impl.Has.8ce)]
// CHECK:STDOUT:   %Core.import_ref.a54: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Get.type (%T.as_type.as.OptionalStorage.impl.Get.type.3e7) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Get (constants.%T.as_type.as.OptionalStorage.impl.Get.f80)]
// CHECK:STDOUT:   %Core.import_ref.5bf = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], unloaded
// CHECK:STDOUT:   %OptionalStorage.impl_witness_table.968 = impl_witness_table (%Core.import_ref.9a31, %Core.import_ref.505, %Core.import_ref.71ea, %Core.import_ref.242, %Core.import_ref.a54, %Core.import_ref.5bf), @T.as_type.as.OptionalStorage.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.822: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.822), @T.as_type.as.DestroyImpl.impl [concrete]
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
// CHECK:STDOUT: generic fn @TestIterate(%R.loc49_17.2: %Iterate_where.type.470) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Iterate.lookup_impl_witness: <witness> = lookup_impl_witness %R.loc49_17.1, @Iterate [symbolic = %Iterate.lookup_impl_witness (constants.%Iterate.lookup_impl_witness.9ae)]
// CHECK:STDOUT:   %Iterate.facet.loc53_44.5: %Iterate.type = facet_value %R.as_type.loc49_86.1, (%Iterate.lookup_impl_witness) [symbolic = %Iterate.facet.loc53_44.5 (constants.%Iterate.facet.c23)]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.loc53_44.5) [symbolic = %Iterate.WithSelf.NewCursor.type (constants.%Iterate.WithSelf.NewCursor.type.1fa)]
// CHECK:STDOUT:   %.loc53_44.26: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type, %Iterate.facet.loc53_44.5 [symbolic = %.loc53_44.26 (constants.%.88f)]
// CHECK:STDOUT:   %impl.elem2.loc53_44.2: @TestIterate.%.loc53_44.26 (%.88f) = impl_witness_access %Iterate.lookup_impl_witness, element2 [symbolic = %impl.elem2.loc53_44.2 (constants.%impl.elem2.ef4)]
// CHECK:STDOUT:   %specific_impl_fn.loc53_44.3: <specific function> = specific_impl_function %impl.elem2.loc53_44.2, @Iterate.WithSelf.NewCursor(%Iterate.facet.loc53_44.5) [symbolic = %specific_impl_fn.loc53_44.3 (constants.%specific_impl_fn.72f)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.loc53_44.5) [symbolic = %Iterate.WithSelf.Next.type (constants.%Iterate.WithSelf.Next.type.a94)]
// CHECK:STDOUT:   %.loc53_44.27: type = fn_type_with_self_type %Iterate.WithSelf.Next.type, %Iterate.facet.loc53_44.5 [symbolic = %.loc53_44.27 (constants.%.a42)]
// CHECK:STDOUT:   %impl.elem3.loc53_44.2: @TestIterate.%.loc53_44.27 (%.a42) = impl_witness_access %Iterate.lookup_impl_witness, element3 [symbolic = %impl.elem3.loc53_44.2 (constants.%impl.elem3.164)]
// CHECK:STDOUT:   %specific_impl_fn.loc53_44.4: <specific function> = specific_impl_function %impl.elem3.loc53_44.2, @Iterate.WithSelf.Next(%Iterate.facet.loc53_44.5) [symbolic = %specific_impl_fn.loc53_44.4 (constants.%specific_impl_fn.84d)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%r.param: @TestIterate.%R.as_type.loc49_86.1 (%R.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %i.patt: %pattern_type.d2d = value_binding_pattern i [concrete = constants.%i.patt]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %r.ref: @TestIterate.%R.as_type.loc49_86.1 (%R.as_type) = name_ref r, %r
// CHECK:STDOUT:     %impl.elem2.loc53_44.1: @TestIterate.%.loc53_44.26 (%.88f) = impl_witness_access constants.%Iterate.lookup_impl_witness.9ae, element2 [symbolic = %impl.elem2.loc53_44.2 (constants.%impl.elem2.ef4)]
// CHECK:STDOUT:     %bound_method.loc53_44.1: <bound method> = bound_method %r.ref, %impl.elem2.loc53_44.1
// CHECK:STDOUT:     %Iterate.facet.loc53_44.1: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.9ae) [symbolic = %Iterate.facet.loc53_44.5 (constants.%Iterate.facet.c23)]
// CHECK:STDOUT:     %.loc53_44.1: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc53_44.1 [symbolic = %Iterate.facet.loc53_44.5 (constants.%Iterate.facet.c23)]
// CHECK:STDOUT:     %Iterate.facet.loc53_44.2: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.9ae) [symbolic = %Iterate.facet.loc53_44.5 (constants.%Iterate.facet.c23)]
// CHECK:STDOUT:     %.loc53_44.2: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc53_44.2 [symbolic = %Iterate.facet.loc53_44.5 (constants.%Iterate.facet.c23)]
// CHECK:STDOUT:     %specific_impl_fn.loc53_44.1: <specific function> = specific_impl_function %impl.elem2.loc53_44.1, @Iterate.WithSelf.NewCursor(constants.%Iterate.facet.c23) [symbolic = %specific_impl_fn.loc53_44.3 (constants.%specific_impl_fn.72f)]
// CHECK:STDOUT:     %bound_method.loc53_44.2: <bound method> = bound_method %r.ref, %specific_impl_fn.loc53_44.1
// CHECK:STDOUT:     %var: ref @TestIterate.%as_type (%as_type.458) = var_storage invalid
// CHECK:STDOUT:     %Iterate.WithSelf.NewCursor.call: init @TestIterate.%as_type (%as_type.458) to %var = call %bound_method.loc53_44.2(%r.ref)
// CHECK:STDOUT:     assign %var, %Iterate.WithSelf.NewCursor.call
// CHECK:STDOUT:     br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.next:
// CHECK:STDOUT:     %impl.elem3.loc53_44.1: @TestIterate.%.loc53_44.27 (%.a42) = impl_witness_access constants.%Iterate.lookup_impl_witness.9ae, element3 [symbolic = %impl.elem3.loc53_44.2 (constants.%impl.elem3.164)]
// CHECK:STDOUT:     %bound_method.loc53_44.3: <bound method> = bound_method %r.ref, %impl.elem3.loc53_44.1
// CHECK:STDOUT:     %Iterate.facet.loc53_44.3: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.9ae) [symbolic = %Iterate.facet.loc53_44.5 (constants.%Iterate.facet.c23)]
// CHECK:STDOUT:     %.loc53_44.3: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc53_44.3 [symbolic = %Iterate.facet.loc53_44.5 (constants.%Iterate.facet.c23)]
// CHECK:STDOUT:     %Iterate.facet.loc53_44.4: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.9ae) [symbolic = %Iterate.facet.loc53_44.5 (constants.%Iterate.facet.c23)]
// CHECK:STDOUT:     %.loc53_44.4: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc53_44.4 [symbolic = %Iterate.facet.loc53_44.5 (constants.%Iterate.facet.c23)]
// CHECK:STDOUT:     %.loc53_44.5: %facet_type.8a3 = converted constants.%as_type.458, constants.%impl.elem1.ef7 [symbolic = %impl.elem1 (constants.%impl.elem1.ef7)]
// CHECK:STDOUT:     %.loc53_44.6: %facet_type.8a3 = converted constants.%as_type.458, constants.%impl.elem1.ef7 [symbolic = %impl.elem1 (constants.%impl.elem1.ef7)]
// CHECK:STDOUT:     %specific_impl_fn.loc53_44.2: <specific function> = specific_impl_function %impl.elem3.loc53_44.1, @Iterate.WithSelf.Next(constants.%Iterate.facet.c23) [symbolic = %specific_impl_fn.loc53_44.4 (constants.%specific_impl_fn.84d)]
// CHECK:STDOUT:     %bound_method.loc53_44.4: <bound method> = bound_method %r.ref, %specific_impl_fn.loc53_44.2
// CHECK:STDOUT:     %.loc53_44.7: ref %Optional.8d7 = temporary_storage
// CHECK:STDOUT:     %Iterate.WithSelf.Next.call: init %Optional.8d7 to %.loc53_44.7 = call %bound_method.loc53_44.4(%r.ref, %var)
// CHECK:STDOUT:     %.loc53_44.8: ref %Optional.8d7 = temporary %.loc53_44.7, %Iterate.WithSelf.Next.call
// CHECK:STDOUT:     %.loc53_44.9: %Optional.HasValue.type.491 = specific_constant imports.%Core.import_ref.c82, @Optional(constants.%OptionalStorage.facet.a50) [concrete = constants.%Optional.HasValue.163]
// CHECK:STDOUT:     %HasValue.ref: %Optional.HasValue.type.491 = name_ref HasValue, %.loc53_44.9 [concrete = constants.%Optional.HasValue.163]
// CHECK:STDOUT:     %Optional.HasValue.bound: <bound method> = bound_method %.loc53_44.8, %HasValue.ref
// CHECK:STDOUT:     %Optional.HasValue.specific_fn: <specific function> = specific_function %HasValue.ref, @Optional.HasValue(constants.%OptionalStorage.facet.a50) [concrete = constants.%Optional.HasValue.specific_fn]
// CHECK:STDOUT:     %bound_method.loc53_44.5: <bound method> = bound_method %.loc53_44.8, %Optional.HasValue.specific_fn
// CHECK:STDOUT:     %.loc53_44.10: %Optional.8d7 = acquire_value %.loc53_44.8
// CHECK:STDOUT:     %Optional.HasValue.call: init bool = call %bound_method.loc53_44.5(%.loc53_44.10)
// CHECK:STDOUT:     %.loc53_44.11: bool = value_of_initializer %Optional.HasValue.call
// CHECK:STDOUT:     %.loc53_44.12: bool = converted %Optional.HasValue.call, %.loc53_44.11
// CHECK:STDOUT:     if %.loc53_44.12 br !for.body else br !for.done
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.body:
// CHECK:STDOUT:     %.loc53_44.13: %Optional.Get.type.935 = specific_constant imports.%Core.import_ref.29e, @Optional(constants.%OptionalStorage.facet.a50) [concrete = constants.%Optional.Get.ce9]
// CHECK:STDOUT:     %Get.ref: %Optional.Get.type.935 = name_ref Get, %.loc53_44.13 [concrete = constants.%Optional.Get.ce9]
// CHECK:STDOUT:     %Optional.Get.bound: <bound method> = bound_method %.loc53_44.8, %Get.ref
// CHECK:STDOUT:     %Optional.Get.specific_fn: <specific function> = specific_function %Get.ref, @Optional.Get(constants.%OptionalStorage.facet.a50) [concrete = constants.%Optional.Get.specific_fn]
// CHECK:STDOUT:     %bound_method.loc53_44.6: <bound method> = bound_method %.loc53_44.8, %Optional.Get.specific_fn
// CHECK:STDOUT:     %.loc53_44.14: ref %ValueType = temporary_storage
// CHECK:STDOUT:     %.loc53_44.15: %Optional.8d7 = acquire_value %.loc53_44.8
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
// CHECK:STDOUT:     %facet_value.loc53_44.1: %facet_type.8a3 = facet_value constants.%ValueType, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.127) [concrete = constants.%facet_value.cfd]
// CHECK:STDOUT:     %.loc53_44.18: %facet_type.8a3 = converted constants.%ValueType, %facet_value.loc53_44.1 [concrete = constants.%facet_value.cfd]
// CHECK:STDOUT:     %facet_value.loc53_44.2: %facet_type.8a3 = facet_value constants.%ValueType, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.127) [concrete = constants.%facet_value.cfd]
// CHECK:STDOUT:     %.loc53_44.19: %facet_type.8a3 = converted constants.%ValueType, %facet_value.loc53_44.2 [concrete = constants.%facet_value.cfd]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %SelfDestruct.call.loc53_44.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc53_44.16)
// CHECK:STDOUT:     %facet_value.loc53_44.3: %facet_type.8a3 = facet_value constants.%Optional.8d7, (constants.%custom_witness.b80ef6.8, constants.%DestroyImpl.impl_witness.b9d) [concrete = constants.%facet_value.4d8]
// CHECK:STDOUT:     %.loc53_44.20: %facet_type.8a3 = converted constants.%Optional.8d7, %facet_value.loc53_44.3 [concrete = constants.%facet_value.4d8]
// CHECK:STDOUT:     %facet_value.loc53_44.4: %facet_type.8a3 = facet_value constants.%Optional.8d7, (constants.%custom_witness.b80ef6.8, constants.%DestroyImpl.impl_witness.b9d) [concrete = constants.%facet_value.4d8]
// CHECK:STDOUT:     %.loc53_44.21: %facet_type.8a3 = converted constants.%Optional.8d7, %facet_value.loc53_44.4 [concrete = constants.%facet_value.4d8]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %SelfDestruct.call.loc53_44.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc53_44.8)
// CHECK:STDOUT:     br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.done:
// CHECK:STDOUT:     %facet_value.loc53_44.5: %facet_type.8a3 = facet_value constants.%Optional.8d7, (constants.%custom_witness.b80ef6.8, constants.%DestroyImpl.impl_witness.b9d) [concrete = constants.%facet_value.4d8]
// CHECK:STDOUT:     %.loc53_44.22: %facet_type.8a3 = converted constants.%Optional.8d7, %facet_value.loc53_44.5 [concrete = constants.%facet_value.4d8]
// CHECK:STDOUT:     %facet_value.loc53_44.6: %facet_type.8a3 = facet_value constants.%Optional.8d7, (constants.%custom_witness.b80ef6.8, constants.%DestroyImpl.impl_witness.b9d) [concrete = constants.%facet_value.4d8]
// CHECK:STDOUT:     %.loc53_44.23: %facet_type.8a3 = converted constants.%Optional.8d7, %facet_value.loc53_44.6 [concrete = constants.%facet_value.4d8]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %SelfDestruct.call.loc53_44.3: init %empty_tuple.type = call %SelfDestruct.specific_fn.3(%.loc53_44.8)
// CHECK:STDOUT:     %.loc53_44.24: %facet_type.8a3 = converted constants.%as_type.458, constants.%impl.elem1.ef7 [symbolic = %impl.elem1 (constants.%impl.elem1.ef7)]
// CHECK:STDOUT:     %.loc53_44.25: %facet_type.8a3 = converted constants.%as_type.458, constants.%impl.elem1.ef7 [symbolic = %impl.elem1 (constants.%impl.elem1.ef7)]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %SelfDestruct.call.loc53_44.4: init %empty_tuple.type = call %SelfDestruct.specific_fn.4(%var)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc53_44.1(%self.param: ref %.dd1) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc53_44.2(%self.param: ref %MaybeUnformed.547) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc53_44.3(%self.param: ref bool) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc53_44.4(%self.param: ref %struct_type.value.has_value.64c) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc53_44.5(%self.param: ref %DefaultOptionalStorage.f91) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc53_44.6(%self.param: ref %Optional.8d7) = "no_op";
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
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.62c
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
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %facet_type.d61: type = facet_type <@SubobjectDestroy & @DestroyImpl & @Copy> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %T.220: %OptionalStorage.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Optional.Get.type.786: type = fn_type @Optional.Get, @Optional(%T.220) [symbolic]
// CHECK:STDOUT:   %Optional.Get.2d0: %Optional.Get.type.786 = struct_value () [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.type.3ea: type = fn_type @Optional.HasValue, @Optional(%T.220) [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.f2f: %Optional.HasValue.type.3ea = struct_value () [symbolic]
// CHECK:STDOUT:   %facet_type.8a3: type = facet_type <@SubobjectDestroy & @DestroyImpl> [concrete]
// CHECK:STDOUT:   %T.9ab: %facet_type.d61 = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.f15: type = class_type @DefaultOptionalStorage, @DefaultOptionalStorage(%T.9ab) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Get.type.3e7: type = fn_type @T.as_type.as.OptionalStorage.impl.Get, @T.as_type.as.OptionalStorage.impl(%T.9ab) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Get.f80: %T.as_type.as.OptionalStorage.impl.Get.type.3e7 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Has.type.371: type = fn_type @T.as_type.as.OptionalStorage.impl.Has, @T.as_type.as.OptionalStorage.impl(%T.9ab) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Has.8ce: %T.as_type.as.OptionalStorage.impl.Has.type.371 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Some.type.4eb: type = fn_type @T.as_type.as.OptionalStorage.impl.Some, @T.as_type.as.OptionalStorage.impl(%T.9ab) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Some.44a: %T.as_type.as.OptionalStorage.impl.Some.type.4eb = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.None.type.76a: type = fn_type @T.as_type.as.OptionalStorage.impl.None, @T.as_type.as.OptionalStorage.impl(%T.9ab) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.None.93b: %T.as_type.as.OptionalStorage.impl.None.type.76a = struct_value () [symbolic]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
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
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.2: type = fn_type @SubobjectDestroy.WithSelf.Op.loc56 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.2: %SubobjectDestroy.WithSelf.Op.type.6870fd.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.2: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.2), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.c2a: %SubobjectDestroy.type = facet_value %ValueType, (%custom_witness.b80ef6.2) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.012: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.c2a) [concrete]
// CHECK:STDOUT:   %const.924: type = const_type %ValueType [concrete]
// CHECK:STDOUT:   %ptr.403: type = ptr_type %const.924 [concrete]
// CHECK:STDOUT:   %ptr.6ed: type = ptr_type %ValueType [concrete]
// CHECK:STDOUT:   %ValueType.Op.type: type = fn_type @ValueType.Op [concrete]
// CHECK:STDOUT:   %ValueType.Op: %ValueType.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.1c5: <witness> = custom_witness (%ValueType.Op), @Copy [concrete]
// CHECK:STDOUT:   %facet_value.3ab: %facet_type.d61 = facet_value %ValueType, (%custom_witness.b80ef6.2, %DestroyImpl.impl_witness.012, %custom_witness.1c5) [concrete]
// CHECK:STDOUT:   %Iterate.lookup_impl_witness.1d8: <witness> = lookup_impl_witness %.Self.500, @Iterate [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.8a4: %facet_type.d61 = impl_witness_access %Iterate.lookup_impl_witness.1d8, element0 [symbolic_self]
// CHECK:STDOUT:   %Iterate_where.type.030: type = facet_type <@Iterate where %impl.elem0.8a4 = %facet_value.3ab> [concrete]
// CHECK:STDOUT:   %pattern_type.5c4: type = pattern_type %Iterate_where.type.030 [concrete]
// CHECK:STDOUT:   %R.patt: %pattern_type.5c4 = symbolic_binding_pattern R, 0 [symbolic]
// CHECK:STDOUT:   %R: %Iterate_where.type.030 = symbolic_binding R, 0 [symbolic]
// CHECK:STDOUT:   %R.as_type: type = facet_access_type %R [symbolic]
// CHECK:STDOUT:   %pattern_type.3f8: type = pattern_type %R.as_type [symbolic]
// CHECK:STDOUT:   %r.param_patt.8aa: %pattern_type.3f8 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %r.patt.ac1: %pattern_type.3f8 = wrapper_binding_pattern r, %r.param_patt.8aa [symbolic]
// CHECK:STDOUT:   %i.patt: %pattern_type.e09 = value_binding_pattern i [concrete]
// CHECK:STDOUT:   %Iterate.lookup_impl_witness.27f: <witness> = lookup_impl_witness %R, @Iterate [symbolic]
// CHECK:STDOUT:   %Iterate.facet.258: %Iterate.type = facet_value %R.as_type, (%Iterate.lookup_impl_witness.27f) [symbolic]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.d3e: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.258) [symbolic]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.e72: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.258) [symbolic]
// CHECK:STDOUT:   %.a1e: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.d3e, %Iterate.facet.258 [symbolic]
// CHECK:STDOUT:   %impl.elem2.5cd: %.a1e = impl_witness_access %Iterate.lookup_impl_witness.27f, element2 [symbolic]
// CHECK:STDOUT:   %impl.elem1.b6f: %facet_type.8a3 = impl_witness_access %Iterate.lookup_impl_witness.27f, element1 [symbolic]
// CHECK:STDOUT:   %as_type.62b: type = facet_access_type %impl.elem1.b6f [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.cec: <specific function> = specific_impl_function %impl.elem2.5cd, @Iterate.WithSelf.NewCursor(%Iterate.facet.258) [symbolic]
// CHECK:STDOUT:   %.a31: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.e72, %Iterate.facet.258 [symbolic]
// CHECK:STDOUT:   %impl.elem3.221: %.a31 = impl_witness_access %Iterate.lookup_impl_witness.27f, element3 [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.a0d: type = class_type @DefaultOptionalStorage, @DefaultOptionalStorage(%facet_value.3ab) [concrete]
// CHECK:STDOUT:   %OptionalStorage.impl_witness.d0a: <witness> = impl_witness imports.%OptionalStorage.impl_witness_table.968, @T.as_type.as.OptionalStorage.impl(%facet_value.3ab) [concrete]
// CHECK:STDOUT:   %OptionalStorage.facet.1a0: %OptionalStorage.type = facet_value %ValueType, (%OptionalStorage.impl_witness.d0a) [concrete]
// CHECK:STDOUT:   %Optional.c62: type = class_type @Optional, @Optional(%OptionalStorage.facet.1a0) [concrete]
// CHECK:STDOUT:   %specific_impl_fn.aae: <specific function> = specific_impl_function %impl.elem3.221, @Iterate.WithSelf.Next(%Iterate.facet.258) [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.type.e82: type = fn_type @Optional.HasValue, @Optional(%OptionalStorage.facet.1a0) [concrete]
// CHECK:STDOUT:   %Optional.HasValue.8e1: %Optional.HasValue.type.e82 = struct_value () [concrete]
// CHECK:STDOUT:   %Optional.Get.type.741: type = fn_type @Optional.Get, @Optional(%OptionalStorage.facet.1a0) [concrete]
// CHECK:STDOUT:   %Optional.Get.776: %Optional.Get.type.741 = struct_value () [concrete]
// CHECK:STDOUT:   %facet_value.ea1: %facet_type.8a3 = facet_value %ValueType, (%custom_witness.b80ef6.2, %DestroyImpl.impl_witness.012) [concrete]
// CHECK:STDOUT:   %MaybeUnformed.388: type = class_type @MaybeUnformed, @MaybeUnformed(%facet_value.ea1) [concrete]
// CHECK:STDOUT:   %.8f3: type = maybe_unformed_type %ValueType [concrete]
// CHECK:STDOUT:   %struct_type.value.has_value.f72: type = struct_type {.value: %MaybeUnformed.388, .has_value: bool} [concrete]
// CHECK:STDOUT:   %Optional.HasValue.specific_fn.f9d: <specific function> = specific_function %Optional.HasValue.8e1, @Optional.HasValue(%OptionalStorage.facet.1a0) [concrete]
// CHECK:STDOUT:   %Optional.Get.specific_fn.c7b: <specific function> = specific_function %Optional.Get.776, @Optional.Get(%OptionalStorage.facet.1a0) [concrete]
// CHECK:STDOUT:   %operator_PlusEqual__carbon_thunk.type: type = fn_type @operator_PlusEqual__carbon_thunk [concrete]
// CHECK:STDOUT:   %operator_PlusEqual__carbon_thunk: %operator_PlusEqual__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.8: type = fn_type @SubobjectDestroy.WithSelf.Op.loc60_41.6 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.8: %SubobjectDestroy.WithSelf.Op.type.6870fd.8 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.8: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.8), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.488: %SubobjectDestroy.type = facet_value %Optional.c62, (%custom_witness.b80ef6.8) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.e4a: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.488) [concrete]
// CHECK:STDOUT:   %facet_value.4bd: %facet_type.8a3 = facet_value %Optional.c62, (%custom_witness.b80ef6.8, %DestroyImpl.impl_witness.e4a) [concrete]
// CHECK:STDOUT:   %MutableAndConstRange: type = class_type @MutableAndConstRange [concrete]
// CHECK:STDOUT:   %pattern_type.fdb: type = pattern_type %MutableAndConstRange [concrete]
// CHECK:STDOUT:   %Iterator: type = class_type @Iterator.1 [concrete]
// CHECK:STDOUT:   %Begin.type: type = fn_type @Begin [concrete]
// CHECK:STDOUT:   %Begin: %Begin.type = struct_value () [concrete]
// CHECK:STDOUT:   %End.type: type = fn_type @End [concrete]
// CHECK:STDOUT:   %End: %End.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.660: <witness> = custom_witness (%Iterator, %Iterator, %Begin, %End), @CppRangeForIterate [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.9: type = fn_type @SubobjectDestroy.WithSelf.Op.loc67 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.9: %SubobjectDestroy.WithSelf.Op.type.6870fd.9 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.9: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.9), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.e6d: %SubobjectDestroy.type = facet_value %Iterator, (%custom_witness.b80ef6.9) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.412: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.e6d) [concrete]
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
// CHECK:STDOUT:   %facet_value.958: %CppRange_where.type.d682d6.1 = facet_value %MutableAndConstRange, (%custom_witness.660, %custom_witness.b80ef6.9, %DestroyImpl.impl_witness.412, %custom_witness.d56, %custom_witness.68a, %custom_witness.edd, %custom_witness.b80ef6.2, %DestroyImpl.impl_witness.012, %custom_witness.1c5, %custom_witness.b80ef6.9, %DestroyImpl.impl_witness.412) [concrete]
// CHECK:STDOUT:   %tuple.type.de6: type = tuple_type (%Iterator, %Iterator) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.10: type = fn_type @SubobjectDestroy.WithSelf.Op.3 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.10: %SubobjectDestroy.WithSelf.Op.type.6870fd.10 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.10: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.10), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.3e6: %SubobjectDestroy.type = facet_value %tuple.type.de6, (%custom_witness.b80ef6.10) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.2c4: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.3e6) [concrete]
// CHECK:STDOUT:   %complete_type.95d: <witness> = complete_type_witness %tuple.type.de6 [concrete]
// CHECK:STDOUT:   %facet_value.916: %facet_type.8a3 = facet_value %tuple.type.de6, (%custom_witness.b80ef6.10, %DestroyImpl.impl_witness.2c4) [concrete]
// CHECK:STDOUT:   %Iterate.impl_witness.cd9: <witness> = impl_witness imports.%Iterate.impl_witness_table.121, @T.as_type.as.Iterate.impl(%facet_value.958) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.type.e6f: type = fn_type @T.as_type.as.Iterate.impl.NewCursor, @T.as_type.as.Iterate.impl(%facet_value.958) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.854: %T.as_type.as.Iterate.impl.NewCursor.type.e6f = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.type.40f: type = fn_type @T.as_type.as.Iterate.impl.Next, @T.as_type.as.Iterate.impl(%facet_value.958) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.2d9: %T.as_type.as.Iterate.impl.Next.type.40f = struct_value () [concrete]
// CHECK:STDOUT:   %Iterate.facet.b8c: %Iterate.type = facet_value %MutableAndConstRange, (%Iterate.impl_witness.cd9) [concrete]
// CHECK:STDOUT:   %facet_value.cd8: %Iterate_where.type.030 = facet_value %MutableAndConstRange, (%Iterate.impl_witness.cd9) [concrete]
// CHECK:STDOUT:   %r.param_patt.3a6: %pattern_type.fdb = value_param_pattern [concrete]
// CHECK:STDOUT:   %r.patt.403: %pattern_type.fdb = wrapper_binding_pattern r, %r.param_patt.3a6 [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.1e1: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.b8c) [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.7a3: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.b8c) [concrete]
// CHECK:STDOUT:   %.521: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.1e1, %Iterate.facet.b8c [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.specific_fn: <specific function> = specific_function %T.as_type.as.Iterate.impl.NewCursor.854, @T.as_type.as.Iterate.impl.NewCursor(%facet_value.958) [concrete]
// CHECK:STDOUT:   %.92f: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.7a3, %Iterate.facet.b8c [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.specific_fn: <specific function> = specific_function %T.as_type.as.Iterate.impl.Next.2d9, @T.as_type.as.Iterate.impl.Next(%facet_value.958) [concrete]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.358: <specific function> = specific_function %SelfDestruct, @SelfDestruct(%facet_value.916) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .OverloadADL = %OverloadADL
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.c82: @Optional.%Optional.HasValue.type (%Optional.HasValue.type.3ea) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.HasValue (constants.%Optional.HasValue.f2f)]
// CHECK:STDOUT:   %Core.import_ref.29e: @Optional.%Optional.Get.type (%Optional.Get.type.786) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.Get (constants.%Optional.Get.2d0)]
// CHECK:STDOUT:   %Core.import_ref.9a31: type = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%DefaultOptionalStorage (constants.%DefaultOptionalStorage.f15)]
// CHECK:STDOUT:   %Core.import_ref.819a: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.None.type (%T.as_type.as.OptionalStorage.impl.None.type.76a) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.None (constants.%T.as_type.as.OptionalStorage.impl.None.93b)]
// CHECK:STDOUT:   %Core.import_ref.14a: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Some.type (%T.as_type.as.OptionalStorage.impl.Some.type.4eb) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Some (constants.%T.as_type.as.OptionalStorage.impl.Some.44a)]
// CHECK:STDOUT:   %Core.import_ref.242: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Has.type (%T.as_type.as.OptionalStorage.impl.Has.type.371) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Has (constants.%T.as_type.as.OptionalStorage.impl.Has.8ce)]
// CHECK:STDOUT:   %Core.import_ref.a54: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Get.type (%T.as_type.as.OptionalStorage.impl.Get.type.3e7) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Get (constants.%T.as_type.as.OptionalStorage.impl.Get.f80)]
// CHECK:STDOUT:   %Core.import_ref.5bf = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], unloaded
// CHECK:STDOUT:   %OptionalStorage.impl_witness_table.968 = impl_witness_table (%Core.import_ref.9a31, %Core.import_ref.819a, %Core.import_ref.14a, %Core.import_ref.242, %Core.import_ref.a54, %Core.import_ref.5bf), @T.as_type.as.OptionalStorage.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.822: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.822), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.d6c: %facet_type.d61 = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.Iterate.impl.%facet_value.1 (constants.%facet_value.9da)]
// CHECK:STDOUT:   %Core.import_ref.2ac: %facet_type.8a3 = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.Iterate.impl.%facet_value.2 (constants.%facet_value.bf6)]
// CHECK:STDOUT:   %Core.import_ref.946: @T.as_type.as.Iterate.impl.%T.as_type.as.Iterate.impl.NewCursor.type (%T.as_type.as.Iterate.impl.NewCursor.type.8fb) = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.Iterate.impl.%T.as_type.as.Iterate.impl.NewCursor (constants.%T.as_type.as.Iterate.impl.NewCursor.8a1)]
// CHECK:STDOUT:   %Core.import_ref.cfb: @T.as_type.as.Iterate.impl.%T.as_type.as.Iterate.impl.Next.type (%T.as_type.as.Iterate.impl.Next.type.d4a) = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.Iterate.impl.%T.as_type.as.Iterate.impl.Next (constants.%T.as_type.as.Iterate.impl.Next.5a8)]
// CHECK:STDOUT:   %Iterate.impl_witness_table.121 = impl_witness_table (%Core.import_ref.d6c, %Core.import_ref.2ac, %Core.import_ref.946, %Core.import_ref.cfb), @T.as_type.as.Iterate.impl [concrete]
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
// CHECK:STDOUT: generic fn @TestIterate(%R.loc56_17.2: %Iterate_where.type.030) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Iterate.lookup_impl_witness: <witness> = lookup_impl_witness %R.loc56_17.1, @Iterate [symbolic = %Iterate.lookup_impl_witness (constants.%Iterate.lookup_impl_witness.27f)]
// CHECK:STDOUT:   %Iterate.facet.loc60_41.5: %Iterate.type = facet_value %R.as_type.loc56_83.1, (%Iterate.lookup_impl_witness) [symbolic = %Iterate.facet.loc60_41.5 (constants.%Iterate.facet.258)]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.loc60_41.5) [symbolic = %Iterate.WithSelf.NewCursor.type (constants.%Iterate.WithSelf.NewCursor.type.d3e)]
// CHECK:STDOUT:   %.loc60_41.26: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type, %Iterate.facet.loc60_41.5 [symbolic = %.loc60_41.26 (constants.%.a1e)]
// CHECK:STDOUT:   %impl.elem2.loc60_41.2: @TestIterate.%.loc60_41.26 (%.a1e) = impl_witness_access %Iterate.lookup_impl_witness, element2 [symbolic = %impl.elem2.loc60_41.2 (constants.%impl.elem2.5cd)]
// CHECK:STDOUT:   %specific_impl_fn.loc60_41.3: <specific function> = specific_impl_function %impl.elem2.loc60_41.2, @Iterate.WithSelf.NewCursor(%Iterate.facet.loc60_41.5) [symbolic = %specific_impl_fn.loc60_41.3 (constants.%specific_impl_fn.cec)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.loc60_41.5) [symbolic = %Iterate.WithSelf.Next.type (constants.%Iterate.WithSelf.Next.type.e72)]
// CHECK:STDOUT:   %.loc60_41.27: type = fn_type_with_self_type %Iterate.WithSelf.Next.type, %Iterate.facet.loc60_41.5 [symbolic = %.loc60_41.27 (constants.%.a31)]
// CHECK:STDOUT:   %impl.elem3.loc60_41.2: @TestIterate.%.loc60_41.27 (%.a31) = impl_witness_access %Iterate.lookup_impl_witness, element3 [symbolic = %impl.elem3.loc60_41.2 (constants.%impl.elem3.221)]
// CHECK:STDOUT:   %specific_impl_fn.loc60_41.4: <specific function> = specific_impl_function %impl.elem3.loc60_41.2, @Iterate.WithSelf.Next(%Iterate.facet.loc60_41.5) [symbolic = %specific_impl_fn.loc60_41.4 (constants.%specific_impl_fn.aae)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%r.param: @TestIterate.%R.as_type.loc56_83.1 (%R.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %i.patt: %pattern_type.e09 = value_binding_pattern i [concrete = constants.%i.patt]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %r.ref: @TestIterate.%R.as_type.loc56_83.1 (%R.as_type) = name_ref r, %r
// CHECK:STDOUT:     %impl.elem2.loc60_41.1: @TestIterate.%.loc60_41.26 (%.a1e) = impl_witness_access constants.%Iterate.lookup_impl_witness.27f, element2 [symbolic = %impl.elem2.loc60_41.2 (constants.%impl.elem2.5cd)]
// CHECK:STDOUT:     %bound_method.loc60_41.1: <bound method> = bound_method %r.ref, %impl.elem2.loc60_41.1
// CHECK:STDOUT:     %Iterate.facet.loc60_41.1: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.27f) [symbolic = %Iterate.facet.loc60_41.5 (constants.%Iterate.facet.258)]
// CHECK:STDOUT:     %.loc60_41.1: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc60_41.1 [symbolic = %Iterate.facet.loc60_41.5 (constants.%Iterate.facet.258)]
// CHECK:STDOUT:     %Iterate.facet.loc60_41.2: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.27f) [symbolic = %Iterate.facet.loc60_41.5 (constants.%Iterate.facet.258)]
// CHECK:STDOUT:     %.loc60_41.2: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc60_41.2 [symbolic = %Iterate.facet.loc60_41.5 (constants.%Iterate.facet.258)]
// CHECK:STDOUT:     %specific_impl_fn.loc60_41.1: <specific function> = specific_impl_function %impl.elem2.loc60_41.1, @Iterate.WithSelf.NewCursor(constants.%Iterate.facet.258) [symbolic = %specific_impl_fn.loc60_41.3 (constants.%specific_impl_fn.cec)]
// CHECK:STDOUT:     %bound_method.loc60_41.2: <bound method> = bound_method %r.ref, %specific_impl_fn.loc60_41.1
// CHECK:STDOUT:     %var: ref @TestIterate.%as_type (%as_type.62b) = var_storage invalid
// CHECK:STDOUT:     %Iterate.WithSelf.NewCursor.call: init @TestIterate.%as_type (%as_type.62b) to %var = call %bound_method.loc60_41.2(%r.ref)
// CHECK:STDOUT:     assign %var, %Iterate.WithSelf.NewCursor.call
// CHECK:STDOUT:     br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.next:
// CHECK:STDOUT:     %impl.elem3.loc60_41.1: @TestIterate.%.loc60_41.27 (%.a31) = impl_witness_access constants.%Iterate.lookup_impl_witness.27f, element3 [symbolic = %impl.elem3.loc60_41.2 (constants.%impl.elem3.221)]
// CHECK:STDOUT:     %bound_method.loc60_41.3: <bound method> = bound_method %r.ref, %impl.elem3.loc60_41.1
// CHECK:STDOUT:     %Iterate.facet.loc60_41.3: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.27f) [symbolic = %Iterate.facet.loc60_41.5 (constants.%Iterate.facet.258)]
// CHECK:STDOUT:     %.loc60_41.3: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc60_41.3 [symbolic = %Iterate.facet.loc60_41.5 (constants.%Iterate.facet.258)]
// CHECK:STDOUT:     %Iterate.facet.loc60_41.4: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.27f) [symbolic = %Iterate.facet.loc60_41.5 (constants.%Iterate.facet.258)]
// CHECK:STDOUT:     %.loc60_41.4: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc60_41.4 [symbolic = %Iterate.facet.loc60_41.5 (constants.%Iterate.facet.258)]
// CHECK:STDOUT:     %.loc60_41.5: %facet_type.8a3 = converted constants.%as_type.62b, constants.%impl.elem1.b6f [symbolic = %impl.elem1 (constants.%impl.elem1.b6f)]
// CHECK:STDOUT:     %.loc60_41.6: %facet_type.8a3 = converted constants.%as_type.62b, constants.%impl.elem1.b6f [symbolic = %impl.elem1 (constants.%impl.elem1.b6f)]
// CHECK:STDOUT:     %specific_impl_fn.loc60_41.2: <specific function> = specific_impl_function %impl.elem3.loc60_41.1, @Iterate.WithSelf.Next(constants.%Iterate.facet.258) [symbolic = %specific_impl_fn.loc60_41.4 (constants.%specific_impl_fn.aae)]
// CHECK:STDOUT:     %bound_method.loc60_41.4: <bound method> = bound_method %r.ref, %specific_impl_fn.loc60_41.2
// CHECK:STDOUT:     %.loc60_41.7: ref %Optional.c62 = temporary_storage
// CHECK:STDOUT:     %Iterate.WithSelf.Next.call: init %Optional.c62 to %.loc60_41.7 = call %bound_method.loc60_41.4(%r.ref, %var)
// CHECK:STDOUT:     %.loc60_41.8: ref %Optional.c62 = temporary %.loc60_41.7, %Iterate.WithSelf.Next.call
// CHECK:STDOUT:     %.loc60_41.9: %Optional.HasValue.type.e82 = specific_constant imports.%Core.import_ref.c82, @Optional(constants.%OptionalStorage.facet.1a0) [concrete = constants.%Optional.HasValue.8e1]
// CHECK:STDOUT:     %HasValue.ref: %Optional.HasValue.type.e82 = name_ref HasValue, %.loc60_41.9 [concrete = constants.%Optional.HasValue.8e1]
// CHECK:STDOUT:     %Optional.HasValue.bound: <bound method> = bound_method %.loc60_41.8, %HasValue.ref
// CHECK:STDOUT:     %Optional.HasValue.specific_fn: <specific function> = specific_function %HasValue.ref, @Optional.HasValue(constants.%OptionalStorage.facet.1a0) [concrete = constants.%Optional.HasValue.specific_fn.f9d]
// CHECK:STDOUT:     %bound_method.loc60_41.5: <bound method> = bound_method %.loc60_41.8, %Optional.HasValue.specific_fn
// CHECK:STDOUT:     %.loc60_41.10: %Optional.c62 = acquire_value %.loc60_41.8
// CHECK:STDOUT:     %Optional.HasValue.call: init bool = call %bound_method.loc60_41.5(%.loc60_41.10)
// CHECK:STDOUT:     %.loc60_41.11: bool = value_of_initializer %Optional.HasValue.call
// CHECK:STDOUT:     %.loc60_41.12: bool = converted %Optional.HasValue.call, %.loc60_41.11
// CHECK:STDOUT:     if %.loc60_41.12 br !for.body else br !for.done
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.body:
// CHECK:STDOUT:     %.loc60_41.13: %Optional.Get.type.741 = specific_constant imports.%Core.import_ref.29e, @Optional(constants.%OptionalStorage.facet.1a0) [concrete = constants.%Optional.Get.776]
// CHECK:STDOUT:     %Get.ref: %Optional.Get.type.741 = name_ref Get, %.loc60_41.13 [concrete = constants.%Optional.Get.776]
// CHECK:STDOUT:     %Optional.Get.bound: <bound method> = bound_method %.loc60_41.8, %Get.ref
// CHECK:STDOUT:     %Optional.Get.specific_fn: <specific function> = specific_function %Get.ref, @Optional.Get(constants.%OptionalStorage.facet.1a0) [concrete = constants.%Optional.Get.specific_fn.c7b]
// CHECK:STDOUT:     %bound_method.loc60_41.6: <bound method> = bound_method %.loc60_41.8, %Optional.Get.specific_fn
// CHECK:STDOUT:     %.loc60_41.14: ref %ValueType = temporary_storage
// CHECK:STDOUT:     %.loc60_41.15: %Optional.c62 = acquire_value %.loc60_41.8
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
// CHECK:STDOUT:     %facet_value.loc60_41.1: %facet_type.8a3 = facet_value constants.%ValueType, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.012) [concrete = constants.%facet_value.ea1]
// CHECK:STDOUT:     %.loc60_41.18: %facet_type.8a3 = converted constants.%ValueType, %facet_value.loc60_41.1 [concrete = constants.%facet_value.ea1]
// CHECK:STDOUT:     %facet_value.loc60_41.2: %facet_type.8a3 = facet_value constants.%ValueType, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.012) [concrete = constants.%facet_value.ea1]
// CHECK:STDOUT:     %.loc60_41.19: %facet_type.8a3 = converted constants.%ValueType, %facet_value.loc60_41.2 [concrete = constants.%facet_value.ea1]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %SelfDestruct.call.loc60_41.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc60_41.16)
// CHECK:STDOUT:     %facet_value.loc60_41.3: %facet_type.8a3 = facet_value constants.%Optional.c62, (constants.%custom_witness.b80ef6.8, constants.%DestroyImpl.impl_witness.e4a) [concrete = constants.%facet_value.4bd]
// CHECK:STDOUT:     %.loc60_41.20: %facet_type.8a3 = converted constants.%Optional.c62, %facet_value.loc60_41.3 [concrete = constants.%facet_value.4bd]
// CHECK:STDOUT:     %facet_value.loc60_41.4: %facet_type.8a3 = facet_value constants.%Optional.c62, (constants.%custom_witness.b80ef6.8, constants.%DestroyImpl.impl_witness.e4a) [concrete = constants.%facet_value.4bd]
// CHECK:STDOUT:     %.loc60_41.21: %facet_type.8a3 = converted constants.%Optional.c62, %facet_value.loc60_41.4 [concrete = constants.%facet_value.4bd]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %SelfDestruct.call.loc60_41.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc60_41.8)
// CHECK:STDOUT:     br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.done:
// CHECK:STDOUT:     %facet_value.loc60_41.5: %facet_type.8a3 = facet_value constants.%Optional.c62, (constants.%custom_witness.b80ef6.8, constants.%DestroyImpl.impl_witness.e4a) [concrete = constants.%facet_value.4bd]
// CHECK:STDOUT:     %.loc60_41.22: %facet_type.8a3 = converted constants.%Optional.c62, %facet_value.loc60_41.5 [concrete = constants.%facet_value.4bd]
// CHECK:STDOUT:     %facet_value.loc60_41.6: %facet_type.8a3 = facet_value constants.%Optional.c62, (constants.%custom_witness.b80ef6.8, constants.%DestroyImpl.impl_witness.e4a) [concrete = constants.%facet_value.4bd]
// CHECK:STDOUT:     %.loc60_41.23: %facet_type.8a3 = converted constants.%Optional.c62, %facet_value.loc60_41.6 [concrete = constants.%facet_value.4bd]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %SelfDestruct.call.loc60_41.3: init %empty_tuple.type = call %SelfDestruct.specific_fn.3(%.loc60_41.8)
// CHECK:STDOUT:     %.loc60_41.24: %facet_type.8a3 = converted constants.%as_type.62b, constants.%impl.elem1.b6f [symbolic = %impl.elem1 (constants.%impl.elem1.b6f)]
// CHECK:STDOUT:     %.loc60_41.25: %facet_type.8a3 = converted constants.%as_type.62b, constants.%impl.elem1.b6f [symbolic = %impl.elem1 (constants.%impl.elem1.b6f)]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %SelfDestruct.call.loc60_41.4: init %empty_tuple.type = call %SelfDestruct.specific_fn.4(%var)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc60_41.1(%self.param: ref %.8f3) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc60_41.2(%self.param: ref %MaybeUnformed.388) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc60_41.3(%self.param: ref bool) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc60_41.4(%self.param: ref %struct_type.value.has_value.f72) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc60_41.5(%self.param: ref %DefaultOptionalStorage.a0d) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc60_41.6(%self.param: ref %Optional.c62) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestRangeFor(%mc.param: ref %MutableAndConstRange) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %i.patt: %pattern_type.e09 = value_binding_pattern i [concrete = constants.%i.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %mc.ref: ref %MutableAndConstRange = name_ref mc, %mc
// CHECK:STDOUT:   %impl.elem2: %.521 = impl_witness_access constants.%Iterate.impl_witness.cd9, element2 [concrete = constants.%T.as_type.as.Iterate.impl.NewCursor.854]
// CHECK:STDOUT:   %bound_method.loc74_42.1: <bound method> = bound_method %mc.ref, %impl.elem2
// CHECK:STDOUT:   %facet_value.loc74_42.1: %CppRange_where.type.d682d6.1 = facet_value constants.%MutableAndConstRange, (constants.%custom_witness.660, constants.%custom_witness.b80ef6.9, constants.%DestroyImpl.impl_witness.412, constants.%custom_witness.d56, constants.%custom_witness.68a, constants.%custom_witness.edd, constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.012, constants.%custom_witness.1c5, constants.%custom_witness.b80ef6.9, constants.%DestroyImpl.impl_witness.412) [concrete = constants.%facet_value.958]
// CHECK:STDOUT:   %.loc74_42.1: %CppRange_where.type.d682d6.1 = converted constants.%MutableAndConstRange, %facet_value.loc74_42.1 [concrete = constants.%facet_value.958]
// CHECK:STDOUT:   %facet_value.loc74_42.2: %CppRange_where.type.d682d6.1 = facet_value constants.%MutableAndConstRange, (constants.%custom_witness.660, constants.%custom_witness.b80ef6.9, constants.%DestroyImpl.impl_witness.412, constants.%custom_witness.d56, constants.%custom_witness.68a, constants.%custom_witness.edd, constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.012, constants.%custom_witness.1c5, constants.%custom_witness.b80ef6.9, constants.%DestroyImpl.impl_witness.412) [concrete = constants.%facet_value.958]
// CHECK:STDOUT:   %.loc74_42.2: %CppRange_where.type.d682d6.1 = converted constants.%MutableAndConstRange, %facet_value.loc74_42.2 [concrete = constants.%facet_value.958]
// CHECK:STDOUT:   %specific_fn.loc74_42.1: <specific function> = specific_function %impl.elem2, @T.as_type.as.Iterate.impl.NewCursor(constants.%facet_value.958) [concrete = constants.%T.as_type.as.Iterate.impl.NewCursor.specific_fn]
// CHECK:STDOUT:   %bound_method.loc74_42.2: <bound method> = bound_method %mc.ref, %specific_fn.loc74_42.1
// CHECK:STDOUT:   %var: ref %tuple.type.de6 = var_storage invalid
// CHECK:STDOUT:   %.loc74_40.1: %MutableAndConstRange = acquire_value %mc.ref
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.call: init %tuple.type.de6 to %var = call %bound_method.loc74_42.2(%.loc74_40.1)
// CHECK:STDOUT:   assign %var, %T.as_type.as.Iterate.impl.NewCursor.call
// CHECK:STDOUT:   br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT: !for.next:
// CHECK:STDOUT:   %impl.elem3: %.92f = impl_witness_access constants.%Iterate.impl_witness.cd9, element3 [concrete = constants.%T.as_type.as.Iterate.impl.Next.2d9]
// CHECK:STDOUT:   %bound_method.loc74_42.3: <bound method> = bound_method %mc.ref, %impl.elem3
// CHECK:STDOUT:   %facet_value.loc74_42.3: %CppRange_where.type.d682d6.1 = facet_value constants.%MutableAndConstRange, (constants.%custom_witness.660, constants.%custom_witness.b80ef6.9, constants.%DestroyImpl.impl_witness.412, constants.%custom_witness.d56, constants.%custom_witness.68a, constants.%custom_witness.edd, constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.012, constants.%custom_witness.1c5, constants.%custom_witness.b80ef6.9, constants.%DestroyImpl.impl_witness.412) [concrete = constants.%facet_value.958]
// CHECK:STDOUT:   %.loc74_42.3: %CppRange_where.type.d682d6.1 = converted constants.%MutableAndConstRange, %facet_value.loc74_42.3 [concrete = constants.%facet_value.958]
// CHECK:STDOUT:   %facet_value.loc74_42.4: %CppRange_where.type.d682d6.1 = facet_value constants.%MutableAndConstRange, (constants.%custom_witness.660, constants.%custom_witness.b80ef6.9, constants.%DestroyImpl.impl_witness.412, constants.%custom_witness.d56, constants.%custom_witness.68a, constants.%custom_witness.edd, constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.012, constants.%custom_witness.1c5, constants.%custom_witness.b80ef6.9, constants.%DestroyImpl.impl_witness.412) [concrete = constants.%facet_value.958]
// CHECK:STDOUT:   %.loc74_42.4: %CppRange_where.type.d682d6.1 = converted constants.%MutableAndConstRange, %facet_value.loc74_42.4 [concrete = constants.%facet_value.958]
// CHECK:STDOUT:   %specific_fn.loc74_42.2: <specific function> = specific_function %impl.elem3, @T.as_type.as.Iterate.impl.Next(constants.%facet_value.958) [concrete = constants.%T.as_type.as.Iterate.impl.Next.specific_fn]
// CHECK:STDOUT:   %bound_method.loc74_42.4: <bound method> = bound_method %mc.ref, %specific_fn.loc74_42.2
// CHECK:STDOUT:   %.loc74_42.5: ref %Optional.c62 = temporary_storage
// CHECK:STDOUT:   %.loc74_40.2: %MutableAndConstRange = acquire_value %mc.ref
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.call: init %Optional.c62 to %.loc74_42.5 = call %bound_method.loc74_42.4(%.loc74_40.2, %var)
// CHECK:STDOUT:   %.loc74_42.6: ref %Optional.c62 = temporary %.loc74_42.5, %T.as_type.as.Iterate.impl.Next.call
// CHECK:STDOUT:   %.loc74_42.7: %Optional.HasValue.type.e82 = specific_constant imports.%Core.import_ref.c82, @Optional(constants.%OptionalStorage.facet.1a0) [concrete = constants.%Optional.HasValue.8e1]
// CHECK:STDOUT:   %HasValue.ref: %Optional.HasValue.type.e82 = name_ref HasValue, %.loc74_42.7 [concrete = constants.%Optional.HasValue.8e1]
// CHECK:STDOUT:   %Optional.HasValue.bound: <bound method> = bound_method %.loc74_42.6, %HasValue.ref
// CHECK:STDOUT:   %Optional.HasValue.specific_fn: <specific function> = specific_function %HasValue.ref, @Optional.HasValue(constants.%OptionalStorage.facet.1a0) [concrete = constants.%Optional.HasValue.specific_fn.f9d]
// CHECK:STDOUT:   %bound_method.loc74_42.5: <bound method> = bound_method %.loc74_42.6, %Optional.HasValue.specific_fn
// CHECK:STDOUT:   %.loc74_42.8: %Optional.c62 = acquire_value %.loc74_42.6
// CHECK:STDOUT:   %Optional.HasValue.call: init bool = call %bound_method.loc74_42.5(%.loc74_42.8)
// CHECK:STDOUT:   %.loc74_42.9: bool = value_of_initializer %Optional.HasValue.call
// CHECK:STDOUT:   %.loc74_42.10: bool = converted %Optional.HasValue.call, %.loc74_42.9
// CHECK:STDOUT:   if %.loc74_42.10 br !for.body else br !for.done
// CHECK:STDOUT:
// CHECK:STDOUT: !for.body:
// CHECK:STDOUT:   %.loc74_42.11: %Optional.Get.type.741 = specific_constant imports.%Core.import_ref.29e, @Optional(constants.%OptionalStorage.facet.1a0) [concrete = constants.%Optional.Get.776]
// CHECK:STDOUT:   %Get.ref: %Optional.Get.type.741 = name_ref Get, %.loc74_42.11 [concrete = constants.%Optional.Get.776]
// CHECK:STDOUT:   %Optional.Get.bound: <bound method> = bound_method %.loc74_42.6, %Get.ref
// CHECK:STDOUT:   %Optional.Get.specific_fn: <specific function> = specific_function %Get.ref, @Optional.Get(constants.%OptionalStorage.facet.1a0) [concrete = constants.%Optional.Get.specific_fn.c7b]
// CHECK:STDOUT:   %bound_method.loc74_42.6: <bound method> = bound_method %.loc74_42.6, %Optional.Get.specific_fn
// CHECK:STDOUT:   %.loc74_42.12: ref %ValueType = temporary_storage
// CHECK:STDOUT:   %.loc74_42.13: %Optional.c62 = acquire_value %.loc74_42.6
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
// CHECK:STDOUT:   %facet_value.loc74_42.5: %facet_type.8a3 = facet_value constants.%ValueType, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.012) [concrete = constants.%facet_value.ea1]
// CHECK:STDOUT:   %.loc74_42.16: %facet_type.8a3 = converted constants.%ValueType, %facet_value.loc74_42.5 [concrete = constants.%facet_value.ea1]
// CHECK:STDOUT:   %facet_value.loc74_42.6: %facet_type.8a3 = facet_value constants.%ValueType, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.012) [concrete = constants.%facet_value.ea1]
// CHECK:STDOUT:   %.loc74_42.17: %facet_type.8a3 = converted constants.%ValueType, %facet_value.loc74_42.6 [concrete = constants.%facet_value.ea1]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc74_42.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc74_42.14)
// CHECK:STDOUT:   %facet_value.loc74_42.7: %facet_type.8a3 = facet_value constants.%Optional.c62, (constants.%custom_witness.b80ef6.8, constants.%DestroyImpl.impl_witness.e4a) [concrete = constants.%facet_value.4bd]
// CHECK:STDOUT:   %.loc74_42.18: %facet_type.8a3 = converted constants.%Optional.c62, %facet_value.loc74_42.7 [concrete = constants.%facet_value.4bd]
// CHECK:STDOUT:   %facet_value.loc74_42.8: %facet_type.8a3 = facet_value constants.%Optional.c62, (constants.%custom_witness.b80ef6.8, constants.%DestroyImpl.impl_witness.e4a) [concrete = constants.%facet_value.4bd]
// CHECK:STDOUT:   %.loc74_42.19: %facet_type.8a3 = converted constants.%Optional.c62, %facet_value.loc74_42.8 [concrete = constants.%facet_value.4bd]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc74_42.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc74_42.6)
// CHECK:STDOUT:   br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT: !for.done:
// CHECK:STDOUT:   %facet_value.loc74_42.9: %facet_type.8a3 = facet_value constants.%Optional.c62, (constants.%custom_witness.b80ef6.8, constants.%DestroyImpl.impl_witness.e4a) [concrete = constants.%facet_value.4bd]
// CHECK:STDOUT:   %.loc74_42.20: %facet_type.8a3 = converted constants.%Optional.c62, %facet_value.loc74_42.9 [concrete = constants.%facet_value.4bd]
// CHECK:STDOUT:   %facet_value.loc74_42.10: %facet_type.8a3 = facet_value constants.%Optional.c62, (constants.%custom_witness.b80ef6.8, constants.%DestroyImpl.impl_witness.e4a) [concrete = constants.%facet_value.4bd]
// CHECK:STDOUT:   %.loc74_42.21: %facet_type.8a3 = converted constants.%Optional.c62, %facet_value.loc74_42.10 [concrete = constants.%facet_value.4bd]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc74_42.3: init %empty_tuple.type = call %SelfDestruct.specific_fn.3(%.loc74_42.6)
// CHECK:STDOUT:   %facet_value.loc74_42.11: %facet_type.8a3 = facet_value constants.%tuple.type.de6, (constants.%custom_witness.b80ef6.10, constants.%DestroyImpl.impl_witness.2c4) [concrete = constants.%facet_value.916]
// CHECK:STDOUT:   %.loc74_42.22: %facet_type.8a3 = converted constants.%tuple.type.de6, %facet_value.loc74_42.11 [concrete = constants.%facet_value.916]
// CHECK:STDOUT:   %facet_value.loc74_42.12: %facet_type.8a3 = facet_value constants.%tuple.type.de6, (constants.%custom_witness.b80ef6.10, constants.%DestroyImpl.impl_witness.2c4) [concrete = constants.%facet_value.916]
// CHECK:STDOUT:   %.loc74_42.23: %facet_type.8a3 = converted constants.%tuple.type.de6, %facet_value.loc74_42.12 [concrete = constants.%facet_value.916]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc74_42.4: init %empty_tuple.type = call %SelfDestruct.specific_fn.4(%var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestIterate(constants.%R) {
// CHECK:STDOUT:   %R.patt.loc56_17.2 => constants.%R.patt
// CHECK:STDOUT:   %R.loc56_17.1 => constants.%R
// CHECK:STDOUT:   %R.as_type.loc56_83.1 => constants.%R.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.3f8
// CHECK:STDOUT:   %r.param_patt.loc56_81.2 => constants.%r.param_patt.8aa
// CHECK:STDOUT:   %r.patt.loc56_81.2 => constants.%r.patt.ac1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestIterate(constants.%facet_value.cd8) {
// CHECK:STDOUT:   %R.patt.loc56_17.2 => constants.%R.patt
// CHECK:STDOUT:   %R.loc56_17.1 => constants.%facet_value.cd8
// CHECK:STDOUT:   %R.as_type.loc56_83.1 => constants.%MutableAndConstRange
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.fdb
// CHECK:STDOUT:   %r.param_patt.loc56_81.2 => constants.%r.param_patt.3a6
// CHECK:STDOUT:   %r.patt.loc56_81.2 => constants.%r.patt.403
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc56 => constants.%complete_type.357
// CHECK:STDOUT:   %Iterate.lookup_impl_witness => constants.%Iterate.impl_witness.cd9
// CHECK:STDOUT:   %Iterate.facet.loc60_41.5 => constants.%Iterate.facet.b8c
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type => constants.%Iterate.WithSelf.NewCursor.type.1e1
// CHECK:STDOUT:   %.loc60_41.26 => constants.%.521
// CHECK:STDOUT:   %impl.elem2.loc60_41.2 => constants.%T.as_type.as.Iterate.impl.NewCursor.854
// CHECK:STDOUT:   %specific_impl_fn.loc60_41.3 => constants.%T.as_type.as.Iterate.impl.NewCursor.specific_fn
// CHECK:STDOUT:   %impl.elem1 => constants.%facet_value.916
// CHECK:STDOUT:   %as_type => constants.%tuple.type.de6
// CHECK:STDOUT:   %require_complete.1 => constants.%complete_type.95d
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type => constants.%Iterate.WithSelf.Next.type.7a3
// CHECK:STDOUT:   %.loc60_41.27 => constants.%.92f
// CHECK:STDOUT:   %impl.elem3.loc60_41.2 => constants.%T.as_type.as.Iterate.impl.Next.2d9
// CHECK:STDOUT:   %specific_impl_fn.loc60_41.4 => constants.%T.as_type.as.Iterate.impl.Next.specific_fn
// CHECK:STDOUT:   %SelfDestruct.specific_fn.6 => constants.%SelfDestruct.specific_fn.358
// CHECK:STDOUT: }
// CHECK:STDOUT:
