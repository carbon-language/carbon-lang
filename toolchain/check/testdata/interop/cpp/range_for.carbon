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
  // CHECK:STDERR: fail_todo_adl_returns_same_types_mutable.carbon:[[@LINE+3]]:15: error: value expression passed to reference parameter [ValueForRefParam]
  // CHECK:STDERR: fail_todo_adl_returns_same_types_mutable.carbon:[[@LINE+2]]:34: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: {{.*}}/prelude/iterate.carbon:41:3-29: note: while building thunk to match the signature of this function [ThunkSignature]
  friend auto begin(MutableRange&) -> Iterator;
  friend auto end(MutableRange&) -> Iterator;
};
} // namespace N
''';

// CHECK:STDERR: fail_todo_adl_returns_same_types_mutable.carbon:[[@LINE+7]]:1-64: note: while deducing parameters of generic declared here [DeductionGenericHere]
// CHECK:STDERR:
// CHECK:STDERR: fail_todo_adl_returns_same_types_mutable.carbon:[[@LINE-7]]:15: error: value expression passed to reference parameter [ValueForRefParam]
// CHECK:STDERR: fail_todo_adl_returns_same_types_mutable.carbon:[[@LINE-8]]:32: note: initializing function parameter [InCallToFunctionParam]
// CHECK:STDERR: {{.*}}/prelude/iterate.carbon:42:3-27: note: while building thunk to match the signature of this function [ThunkSignature]
// CHECK:STDERR: fail_todo_adl_returns_same_types_mutable.carbon:[[@LINE+2]]:1-64: note: while deducing parameters of generic declared here [DeductionGenericHere]
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
  // CHECK:STDERR: fail_todo_adl_returns_different_types_mutable.carbon:[[@LINE+3]]:15: error: value expression passed to reference parameter [ValueForRefParam]
  // CHECK:STDERR: fail_todo_adl_returns_different_types_mutable.carbon:[[@LINE+2]]:34: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: {{.*}}/prelude/iterate.carbon:41:3-29: note: while building thunk to match the signature of this function [ThunkSignature]
  friend auto begin(MutableRange&) -> Iterator;
  friend auto end(MutableRange&) -> Sentinel;
};
}
''';

// CHECK:STDERR: fail_todo_adl_returns_different_types_mutable.carbon:[[@LINE+7]]:1-89: note: while deducing parameters of generic declared here [DeductionGenericHere]
// CHECK:STDERR:
// CHECK:STDERR: fail_todo_adl_returns_different_types_mutable.carbon:[[@LINE-7]]:15: error: value expression passed to reference parameter [ValueForRefParam]
// CHECK:STDERR: fail_todo_adl_returns_different_types_mutable.carbon:[[@LINE-8]]:32: note: initializing function parameter [InCallToFunctionParam]
// CHECK:STDERR: {{.*}}/prelude/iterate.carbon:42:3-27: note: while building thunk to match the signature of this function [ThunkSignature]
// CHECK:STDERR: fail_todo_adl_returns_different_types_mutable.carbon:[[@LINE+2]]:1-89: note: while deducing parameters of generic declared here [DeductionGenericHere]
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
  // CHECK:STDERR: fail_todo_adl_returns_different_types_mutable.carbon:[[@LINE+6]]:3-16: error: semantics TODO: `Rewriting operator== using operator== is not supported` [SemanticsTodo]
  // CHECK:STDERR: fail_todo_adl_returns_different_types_mutable.carbon:[[@LINE-12]]:1-89: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  // CHECK:STDERR: fail_todo_adl_returns_different_types_mutable.carbon:[[@LINE+3]]:3-16: error: cannot convert type `Cpp.DiffMutableADL.MutableRange` into type implementing `Core.Iterate where .(Core.Iterate.ElementType) = Cpp.DiffMutableADL.ValueType as Core.Copy & Core.Destroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR: fail_todo_adl_returns_different_types_mutable.carbon:[[@LINE-15]]:1-89: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  TestIterate(m);
}

fn TestRangeFor(var m: Cpp.DiffMutableADL.MutableRange) {
  var sum: Cpp.DiffMutableADL.ValueType = Cpp.DiffMutableADL.ValueType.ValueType();

  //@dump-sem-ir-begin
  // CHECK:STDERR: fail_todo_adl_returns_different_types_mutable.carbon:[[@LINE+2]]:7-44: error: cannot access member of interface `Core.Iterate` in type `Cpp.DiffMutableADL.MutableRange` that does not implement that interface [MissingImplInMemberAccess]
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
    // CHECK:STDERR: fail_todo_adl_overloads.carbon:[[@LINE+3]]:17: error: value expression passed to reference parameter [ValueForRefParam]
    // CHECK:STDERR: fail_todo_adl_overloads.carbon:[[@LINE+2]]:44: note: initializing function parameter [InCallToFunctionParam]
    // CHECK:STDERR: {{.*}}/prelude/iterate.carbon:41:3-29: note: while building thunk to match the signature of this function [ThunkSignature]
    friend auto begin(MutableAndConstRange&) -> Iterator;
    friend auto end(MutableAndConstRange&) -> Iterator;

    friend auto begin(const MutableAndConstRange&) -> ConstIterator;
    friend auto end(const MutableAndConstRange&) -> ConstIterator;
};
}
''';

// CHECK:STDERR: fail_todo_adl_overloads.carbon:[[@LINE+7]]:1-86: note: while deducing parameters of generic declared here [DeductionGenericHere]
// CHECK:STDERR:
// CHECK:STDERR: fail_todo_adl_overloads.carbon:[[@LINE-10]]:17: error: value expression passed to reference parameter [ValueForRefParam]
// CHECK:STDERR: fail_todo_adl_overloads.carbon:[[@LINE-11]]:42: note: initializing function parameter [InCallToFunctionParam]
// CHECK:STDERR: {{.*}}/prelude/iterate.carbon:42:3-27: note: while building thunk to match the signature of this function [ThunkSignature]
// CHECK:STDERR: fail_todo_adl_overloads.carbon:[[@LINE+2]]:1-86: note: while deducing parameters of generic declared here [DeductionGenericHere]
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
  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE+3]]:3-20: error: cannot convert type `Cpp.NoBeginEnd` into type implementing `Core.Iterate` [ConversionFailureTypeToFacet]
  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE-17]]:1-39: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  Test(no_begin_end);

  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE+3]]:3-23: error: cannot convert type `Cpp.NoBeginMethod` into type implementing `Core.Iterate` [ConversionFailureTypeToFacet]
  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE-22]]:1-39: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  Test(no_begin_method);

  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE+3]]:3-21: error: cannot convert type `Cpp.NoEndMethod` into type implementing `Core.Iterate` [ConversionFailureTypeToFacet]
  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE-27]]:1-39: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  Test(no_end_method);

  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE+3]]:3-20: error: cannot convert type `Cpp.NoBeginADL` into type implementing `Core.Iterate` [ConversionFailureTypeToFacet]
  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE-32]]:1-39: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  Test(no_begin_adl);

  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE+3]]:3-18: error: cannot convert type `Cpp.NoEndADL` into type implementing `Core.Iterate` [ConversionFailureTypeToFacet]
  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE-37]]:1-39: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  Test(no_end_adl);

  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE+3]]:3-28: error: cannot convert type `Cpp.BeginMethodEndADL` into type implementing `Core.Iterate` [ConversionFailureTypeToFacet]
  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE-42]]:1-39: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  Test(begin_method_end_adl);

  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE+3]]:3-28: error: cannot convert type `Cpp.BeginADLEndMethod` into type implementing `Core.Iterate` [ConversionFailureTypeToFacet]
  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE-47]]:1-39: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  Test(begin_adl_end_method);

  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE+3]]:3-30: error: cannot convert type `Cpp.BeginFieldEndMethod` into type implementing `Core.Iterate` [ConversionFailureTypeToFacet]
  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE-52]]:1-39: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  Test(begin_field_end_method);

  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE+3]]:3-30: error: cannot convert type `Cpp.BeginMethodEndField` into type implementing `Core.Iterate` [ConversionFailureTypeToFacet]
  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE-57]]:1-39: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  Test(begin_method_end_field);

  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE+7]]:28: error: attempt to use a deleted function [CppInteropParseError]
  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE-82]]:8: note: 'begin' has been explicitly marked deleted here [CppInteropParseNote]
  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE-63]]:1-39: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE+3]]:3-28: error: cannot convert type `Cpp.BeginMethodDeleted` into type implementing `Core.Iterate` [ConversionFailureTypeToFacet]
  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE-66]]:1-39: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  Test(begin_method_deleted);

  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE+7]]:26: error: attempt to use a deleted function [CppInteropParseError]
  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE-85]]:8: note: 'end' has been explicitly marked deleted here [CppInteropParseNote]
  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE-72]]:1-39: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE+3]]:3-26: error: cannot convert type `Cpp.EndMethodDeleted` into type implementing `Core.Iterate` [ConversionFailureTypeToFacet]
  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE-75]]:1-39: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  Test(end_method_deleted);

  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE+7]]:30: error: call to deleted function 'begin' [CppInteropParseError]
  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE-90]]:15: note: candidate function has been explicitly deleted [CppInteropParseNote]
  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE-81]]:1-39: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE+3]]:3-30: error: cannot convert type `Cpp.BeginFunctionDeleted` into type implementing `Core.Iterate` [ConversionFailureTypeToFacet]
  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE-84]]:1-39: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  Test(begin_function_deleted);

  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE+7]]:28: error: call to deleted function 'end' [CppInteropParseError]
  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE-93]]:15: note: candidate function has been explicitly deleted [CppInteropParseNote]
  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE-90]]:1-39: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE+3]]:3-28: error: cannot convert type `Cpp.EndFunctionDeleted` into type implementing `Core.Iterate` [ConversionFailureTypeToFacet]
  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE-93]]:1-39: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  Test(end_function_deleted);
}

// CHECK:STDOUT: --- methods_return_same_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %Iterate.type: type = facet_type <@Iterate> [concrete]
// CHECK:STDOUT:   %OptionalStorage.type: type = facet_type <@OptionalStorage> [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %facet_type.bd0: type = facet_type <@Copy & @Destroy> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %T.220e40.1: %OptionalStorage.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Optional.Get.type.786: type = fn_type @Optional.Get, @Optional(%T.220e40.1) [symbolic]
// CHECK:STDOUT:   %Optional.Get.2d0: %Optional.Get.type.786 = struct_value () [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.type.3ea: type = fn_type @Optional.HasValue, @Optional(%T.220e40.1) [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.f2f: %Optional.HasValue.type.3ea = struct_value () [symbolic]
// CHECK:STDOUT:   %T.f84392.1: %facet_type.bd0 = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.476: type = class_type @DefaultOptionalStorage, @DefaultOptionalStorage(%T.f84392.1) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Get.type.f27: type = fn_type @T.as_type.as.OptionalStorage.impl.Get, @T.as_type.as.OptionalStorage.impl(%T.f84392.1) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Get.e40: %T.as_type.as.OptionalStorage.impl.Get.type.f27 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Has.type.18e: type = fn_type @T.as_type.as.OptionalStorage.impl.Has, @T.as_type.as.OptionalStorage.impl(%T.f84392.1) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Has.463: %T.as_type.as.OptionalStorage.impl.Has.type.18e = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Some.type.292: type = fn_type @T.as_type.as.OptionalStorage.impl.Some, @T.as_type.as.OptionalStorage.impl(%T.f84392.1) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Some.ae9: %T.as_type.as.OptionalStorage.impl.Some.type.292 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.None.type.b57: type = fn_type @T.as_type.as.OptionalStorage.impl.None, @T.as_type.as.OptionalStorage.impl(%T.f84392.1) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.None.791: %T.as_type.as.OptionalStorage.impl.None.type.b57 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SelfDestruct.type: type = fn_type @SelfDestruct [concrete]
// CHECK:STDOUT:   %SelfDestruct: %SelfDestruct.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %N.fe9: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %Int.d4bbe0.1: type = class_type @Int, @Int(%N.fe9) [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.d94: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.1: type = fn_type @SubobjectDestroy.WithSelf.Op.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.1: %SubobjectDestroy.WithSelf.Op.type.6870fd.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.1: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.1), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.49a: %SubobjectDestroy.type = facet_value %i32, (%custom_witness.b80ef6.1) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.077: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.49a) [concrete]
// CHECK:STDOUT:   %Destroy.facet.046: %Destroy.type = facet_value %i32, (%custom_witness.b80ef6.1, %DestroyImpl.impl_witness.077) [concrete]
// CHECK:STDOUT:   %.Self.500: %Iterate.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Int.d4bbe0.2: type = class_type @Int, @Int(%To) [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.6b0: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %CppRange.type: type = facet_type <@CppRange> [concrete]
// CHECK:STDOUT:   %.Self.e0b: %CppRange.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %CppRangeForIterate.lookup_impl_witness.50b: <witness> = lookup_impl_witness %.Self.e0b, @CppRangeForIterate [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.32d3: type = impl_witness_access %CppRangeForIterate.lookup_impl_witness.50b, element0 [symbolic_self]
// CHECK:STDOUT:   %CppUnsafeDeref.lookup_impl_witness.851: <witness> = lookup_impl_witness %impl.elem0.32d3, @CppUnsafeDeref [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.10d: type = impl_witness_access %CppUnsafeDeref.lookup_impl_witness.851, element0 [symbolic_self]
// CHECK:STDOUT:   %CppRange_where.type.d682d6.1: type = facet_type <@CppRange where %impl.elem0.10d impls @Copy> [concrete]
// CHECK:STDOUT:   %T.f45530.3: %CppRange_where.type.d682d6.1 = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.type.d4a: type = fn_type @T.as_type.as.Iterate.impl.Next, @T.as_type.as.Iterate.impl(%T.f45530.3) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.5a8: %T.as_type.as.Iterate.impl.Next.type.d4a = struct_value () [symbolic]
// CHECK:STDOUT:   %CppRangeForIterate.lookup_impl_witness.b581a8.2: <witness> = lookup_impl_witness %T.f45530.3, @CppRangeForIterate [symbolic]
// CHECK:STDOUT:   %impl.elem0.5f971c.2: type = impl_witness_access %CppRangeForIterate.lookup_impl_witness.b581a8.2, element0 [symbolic]
// CHECK:STDOUT:   %CppUnsafeDeref.lookup_impl_witness.87e543.2: <witness> = lookup_impl_witness %impl.elem0.5f971c.2, @CppUnsafeDeref [symbolic]
// CHECK:STDOUT:   %impl.elem0.493765.2: type = impl_witness_access %CppUnsafeDeref.lookup_impl_witness.87e543.2, element0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.type.8fb: type = fn_type @T.as_type.as.Iterate.impl.NewCursor, @T.as_type.as.Iterate.impl(%T.f45530.3) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.8a1: %T.as_type.as.Iterate.impl.NewCursor.type.8fb = struct_value () [symbolic]
// CHECK:STDOUT:   %impl.elem1.c2dc15.2: type = impl_witness_access %CppRangeForIterate.lookup_impl_witness.b581a8.2, element1 [symbolic]
// CHECK:STDOUT:   %tuple.type.1ba: type = tuple_type (%impl.elem0.5f971c.2, %impl.elem1.c2dc15.2) [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.lookup_impl_witness.dc9: <witness> = lookup_impl_witness %tuple.type.1ba, @SubobjectDestroy [symbolic]
// CHECK:STDOUT:   %DestroyImpl.lookup_impl_witness.504: <witness> = lookup_impl_witness %tuple.type.1ba, @DestroyImpl [symbolic]
// CHECK:STDOUT:   %Destroy.facet.9b6: %Destroy.type = facet_value %tuple.type.1ba, (%SubobjectDestroy.lookup_impl_witness.dc9, %DestroyImpl.lookup_impl_witness.504) [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.lookup_impl_witness.b15da7.2: <witness> = lookup_impl_witness %impl.elem0.493765.2, @SubobjectDestroy [symbolic]
// CHECK:STDOUT:   %DestroyImpl.lookup_impl_witness.4e7f44.2: <witness> = lookup_impl_witness %impl.elem0.493765.2, @DestroyImpl [symbolic]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.167: <witness> = lookup_impl_witness %impl.elem0.493765.2, @Copy [symbolic]
// CHECK:STDOUT:   %facet_value.189: %facet_type.bd0 = facet_value %impl.elem0.493765.2, (%SubobjectDestroy.lookup_impl_witness.b15da7.2, %DestroyImpl.lookup_impl_witness.4e7f44.2, %Copy.lookup_impl_witness.167) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.a1d: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N.fe9) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.3c1: %Int.as.Copy.impl.Op.type.a1d = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.8af: <witness> = impl_witness imports.%Copy.impl_witness_table.f44, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %facet_value.521: %facet_type.bd0 = facet_value %i32, (%custom_witness.b80ef6.1, %DestroyImpl.impl_witness.077, %Copy.impl_witness.8af) [concrete]
// CHECK:STDOUT:   %Iterate.lookup_impl_witness.1d8: <witness> = lookup_impl_witness %.Self.500, @Iterate [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.4ce: %facet_type.bd0 = impl_witness_access %Iterate.lookup_impl_witness.1d8, element0 [symbolic_self]
// CHECK:STDOUT:   %Iterate_where.type.afe: type = facet_type <@Iterate where %impl.elem0.4ce = %facet_value.521> [concrete]
// CHECK:STDOUT:   %pattern_type.535: type = pattern_type %Iterate_where.type.afe [concrete]
// CHECK:STDOUT:   %R.patt: %pattern_type.535 = symbolic_binding_pattern R, 0 [symbolic]
// CHECK:STDOUT:   %R: %Iterate_where.type.afe = symbolic_binding R, 0 [symbolic]
// CHECK:STDOUT:   %R.as_type: type = facet_access_type %R [symbolic]
// CHECK:STDOUT:   %pattern_type.0e6: type = pattern_type %R.as_type [symbolic]
// CHECK:STDOUT:   %r.param_patt.874: %pattern_type.0e6 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %r.patt.97e: %pattern_type.0e6 = wrapper_binding_pattern r, %r.param_patt.874 [symbolic]
// CHECK:STDOUT:   %facet_type.08a: type = facet_type <@IntFitsIn, @IntFitsIn(%Int.d4bbe0.2) & @AnyInt> [symbolic]
// CHECK:STDOUT:   %From.170: %facet_type.08a = symbolic_binding From, 1 [symbolic]
// CHECK:STDOUT:   %From.as_type.as.ImplicitAs.impl.Convert.type.386: type = fn_type @From.as_type.as.ImplicitAs.impl.Convert, @From.as_type.as.ImplicitAs.impl(%To, %From.170) [symbolic]
// CHECK:STDOUT:   %From.as_type.as.ImplicitAs.impl.Convert.85a: %From.as_type.as.ImplicitAs.impl.Convert.type.386 = struct_value () [symbolic]
// CHECK:STDOUT:   %Int.as.AnyInt.impl.AsInt.type.0ee: type = fn_type @Int.as.AnyInt.impl.AsInt, @Int.as.AnyInt.impl(%N.fe9) [symbolic]
// CHECK:STDOUT:   %Int.as.AnyInt.impl.AsInt.497: %Int.as.AnyInt.impl.AsInt.type.0ee = struct_value () [symbolic]
// CHECK:STDOUT:   %i.patt: %pattern_type.d94 = value_binding_pattern i [concrete]
// CHECK:STDOUT:   %Iterate.lookup_impl_witness.d33: <witness> = lookup_impl_witness %R, @Iterate [symbolic]
// CHECK:STDOUT:   %Iterate.facet.342: %Iterate.type = facet_value %R.as_type, (%Iterate.lookup_impl_witness.d33) [symbolic]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.26d: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.342) [symbolic]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.1e7: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.342) [symbolic]
// CHECK:STDOUT:   %.ef3: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.26d, %Iterate.facet.342 [symbolic]
// CHECK:STDOUT:   %impl.elem2.ca9: %.ef3 = impl_witness_access %Iterate.lookup_impl_witness.d33, element2 [symbolic]
// CHECK:STDOUT:   %impl.elem1.732: %Destroy.type = impl_witness_access %Iterate.lookup_impl_witness.d33, element1 [symbolic]
// CHECK:STDOUT:   %as_type.5c5: type = facet_access_type %impl.elem1.732 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.c93: <specific function> = specific_impl_function %impl.elem2.ca9, @Iterate.WithSelf.NewCursor(%Iterate.facet.342) [symbolic]
// CHECK:STDOUT:   %.ab95: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.1e7, %Iterate.facet.342 [symbolic]
// CHECK:STDOUT:   %impl.elem3.e47: %.ab95 = impl_witness_access %Iterate.lookup_impl_witness.d33, element3 [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.af2: type = class_type @DefaultOptionalStorage, @DefaultOptionalStorage(%facet_value.521) [concrete]
// CHECK:STDOUT:   %OptionalStorage.impl_witness.78c: <witness> = impl_witness imports.%OptionalStorage.impl_witness_table.1c4, @T.as_type.as.OptionalStorage.impl(%facet_value.521) [concrete]
// CHECK:STDOUT:   %OptionalStorage.facet.072: %OptionalStorage.type = facet_value %i32, (%OptionalStorage.impl_witness.78c) [concrete]
// CHECK:STDOUT:   %Optional.561: type = class_type @Optional, @Optional(%OptionalStorage.facet.072) [concrete]
// CHECK:STDOUT:   %specific_impl_fn.c3b: <specific function> = specific_impl_function %impl.elem3.e47, @Iterate.WithSelf.Next(%Iterate.facet.342) [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.type.92b: type = fn_type @Optional.HasValue, @Optional(%OptionalStorage.facet.072) [concrete]
// CHECK:STDOUT:   %Optional.HasValue.104: %Optional.HasValue.type.92b = struct_value () [concrete]
// CHECK:STDOUT:   %Optional.Get.type.4e6: type = fn_type @Optional.Get, @Optional(%OptionalStorage.facet.072) [concrete]
// CHECK:STDOUT:   %Optional.Get.3a9: %Optional.Get.type.4e6 = struct_value () [concrete]
// CHECK:STDOUT:   %MaybeUnformed.f2c: type = class_type @MaybeUnformed, @MaybeUnformed(%Destroy.facet.046) [concrete]
// CHECK:STDOUT:   %.b31: type = maybe_unformed_type %i32 [concrete]
// CHECK:STDOUT:   %struct_type.value.has_value.6f1: type = struct_type {.value: %MaybeUnformed.f2c, .has_value: bool} [concrete]
// CHECK:STDOUT:   %Optional.HasValue.specific_fn.3c4: <specific function> = specific_function %Optional.HasValue.104, @Optional.HasValue(%OptionalStorage.facet.072) [concrete]
// CHECK:STDOUT:   %Optional.Get.specific_fn.65d: <specific function> = specific_function %Optional.Get.3a9, @Optional.Get(%OptionalStorage.facet.072) [concrete]
// CHECK:STDOUT:   %AddAssignWith.type.1ad: type = facet_type <@AddAssignWith, @AddAssignWith(%i32)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.81c7d0.2: type = facet_type <@ImplicitAs, @ImplicitAs(%Int.d4bbe0.1)> [symbolic]
// CHECK:STDOUT:   %U.f16: %ImplicitAs.type.81c7d0.2 = symbolic_binding U, 1 [symbolic]
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.type.93140c.1: type = fn_type @Int.as.AddAssignWith.impl.Op.1, @Int.as.AddAssignWith.impl(%N.fe9, %U.f16) [symbolic]
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.75e08e.1: %Int.as.AddAssignWith.impl.Op.type.93140c.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.type.93140c.2: type = fn_type @Int.as.AddAssignWith.impl.Op.2, @Int.as.AddAssignWith.impl(%N.fe9, %U.f16) [symbolic]
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.75e08e.2: %Int.as.AddAssignWith.impl.Op.type.93140c.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %facet_type.0f0: type = facet_type <@IntFitsIn, @IntFitsIn(%i32) & @AnyInt> [concrete]
// CHECK:STDOUT:   %custom_witness.633: <witness> = custom_witness (), @IntFitsIn, @IntFitsIn(%i32) [concrete]
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
// CHECK:STDOUT:   %SubobjectDestroy.facet.278: %SubobjectDestroy.type = facet_value %Optional.561, (%custom_witness.b80ef6.8) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.dd0: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.278) [concrete]
// CHECK:STDOUT:   %Destroy.facet.58d: %Destroy.type = facet_value %Optional.561, (%custom_witness.b80ef6.8, %DestroyImpl.impl_witness.dd0) [concrete]
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
// CHECK:STDOUT:   %Destroy.facet.70f: %Destroy.type = facet_value %tuple.type.508, (%custom_witness.b80ef6.10, %DestroyImpl.impl_witness.91e) [concrete]
// CHECK:STDOUT:   %Iterate.impl_witness.2ca: <witness> = impl_witness imports.%Iterate.impl_witness_table.121, @T.as_type.as.Iterate.impl(%facet_value.209) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.type.9f7: type = fn_type @T.as_type.as.Iterate.impl.NewCursor, @T.as_type.as.Iterate.impl(%facet_value.209) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.219: %T.as_type.as.Iterate.impl.NewCursor.type.9f7 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.type.80a: type = fn_type @T.as_type.as.Iterate.impl.Next, @T.as_type.as.Iterate.impl(%facet_value.209) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.a23: %T.as_type.as.Iterate.impl.Next.type.80a = struct_value () [concrete]
// CHECK:STDOUT:   %facet_value.0ca: %Iterate_where.type.afe = facet_value %MutableRange, (%Iterate.impl_witness.2ca) [concrete]
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
// CHECK:STDOUT:   %SubobjectDestroy.facet.4d44: %SubobjectDestroy.type = facet_value %Iterator.260, (%custom_witness.b80ef6.11) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.514: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.4d44) [concrete]
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
// CHECK:STDOUT:   %Destroy.facet.141: %Destroy.type = facet_value %tuple.type.69f, (%custom_witness.b80ef6.12, %DestroyImpl.impl_witness.4ff) [concrete]
// CHECK:STDOUT:   %Iterate.impl_witness.51a: <witness> = impl_witness imports.%Iterate.impl_witness_table.121, @T.as_type.as.Iterate.impl(%facet_value.20b) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.type.efe: type = fn_type @T.as_type.as.Iterate.impl.NewCursor, @T.as_type.as.Iterate.impl(%facet_value.20b) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.1b8: %T.as_type.as.Iterate.impl.NewCursor.type.efe = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.type.9b9: type = fn_type @T.as_type.as.Iterate.impl.Next, @T.as_type.as.Iterate.impl(%facet_value.20b) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.283: %T.as_type.as.Iterate.impl.Next.type.9b9 = struct_value () [concrete]
// CHECK:STDOUT:   %facet_value.2e7: %Iterate_where.type.afe = facet_value %ConstRange, (%Iterate.impl_witness.51a) [concrete]
// CHECK:STDOUT:   %r.param_patt.83e: %pattern_type.0ba = value_param_pattern [concrete]
// CHECK:STDOUT:   %r.patt.c21: %pattern_type.0ba = wrapper_binding_pattern r, %r.param_patt.83e [concrete]
// CHECK:STDOUT:   %Iterate.facet.da9: %Iterate.type = facet_value %MutableRange, (%Iterate.impl_witness.2ca) [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.b65: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.da9) [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.88f: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.da9) [concrete]
// CHECK:STDOUT:   %.efe: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.b65, %Iterate.facet.da9 [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.specific_fn.d1a: <specific function> = specific_function %T.as_type.as.Iterate.impl.NewCursor.219, @T.as_type.as.Iterate.impl.NewCursor(%facet_value.209) [concrete]
// CHECK:STDOUT:   %.d9d2: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.88f, %Iterate.facet.da9 [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.specific_fn.e11: <specific function> = specific_function %T.as_type.as.Iterate.impl.Next.a23, @T.as_type.as.Iterate.impl.Next(%facet_value.209) [concrete]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.a77: <specific function> = specific_function %SelfDestruct, @SelfDestruct(%Destroy.facet.70f) [concrete]
// CHECK:STDOUT:   %Iterate.facet.656: %Iterate.type = facet_value %ConstRange, (%Iterate.impl_witness.51a) [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.32a: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.656) [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.4e2: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.656) [concrete]
// CHECK:STDOUT:   %.4dd: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.32a, %Iterate.facet.656 [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.specific_fn.ece: <specific function> = specific_function %T.as_type.as.Iterate.impl.NewCursor.1b8, @T.as_type.as.Iterate.impl.NewCursor(%facet_value.20b) [concrete]
// CHECK:STDOUT:   %.4fc: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.4e2, %Iterate.facet.656 [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.specific_fn.d22: <specific function> = specific_function %T.as_type.as.Iterate.impl.Next.283, @T.as_type.as.Iterate.impl.Next(%facet_value.20b) [concrete]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.a15: <specific function> = specific_function %SelfDestruct, @SelfDestruct(%Destroy.facet.141) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.481: type = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%DefaultOptionalStorage (constants.%DefaultOptionalStorage.476)]
// CHECK:STDOUT:   %Core.import_ref.66d: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.None.type (%T.as_type.as.OptionalStorage.impl.None.type.b57) = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.None (constants.%T.as_type.as.OptionalStorage.impl.None.791)]
// CHECK:STDOUT:   %Core.import_ref.32c: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Some.type (%T.as_type.as.OptionalStorage.impl.Some.type.292) = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Some (constants.%T.as_type.as.OptionalStorage.impl.Some.ae9)]
// CHECK:STDOUT:   %Core.import_ref.801: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Has.type (%T.as_type.as.OptionalStorage.impl.Has.type.18e) = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Has (constants.%T.as_type.as.OptionalStorage.impl.Has.463)]
// CHECK:STDOUT:   %Core.import_ref.eec: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Get.type (%T.as_type.as.OptionalStorage.impl.Get.type.f27) = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Get (constants.%T.as_type.as.OptionalStorage.impl.Get.e40)]
// CHECK:STDOUT:   %Core.import_ref.a6a = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %OptionalStorage.impl_witness_table.1c4 = impl_witness_table (%Core.import_ref.481, %Core.import_ref.66d, %Core.import_ref.32c, %Core.import_ref.801, %Core.import_ref.eec, %Core.import_ref.a6a), @T.as_type.as.OptionalStorage.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.7b5: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/types/optional, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.7b5), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.c82: @Optional.%Optional.HasValue.type (%Optional.HasValue.type.3ea) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.HasValue (constants.%Optional.HasValue.f2f)]
// CHECK:STDOUT:   %Core.import_ref.29e: @Optional.%Optional.Get.type (%Optional.Get.type.786) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.Get (constants.%Optional.Get.2d0)]
// CHECK:STDOUT:   %Core.import_ref.724: %facet_type.bd0 = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.Iterate.impl.%facet_value (constants.%facet_value.189)]
// CHECK:STDOUT:   %Core.import_ref.dd9: %Destroy.type = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.Iterate.impl.%Destroy.facet (constants.%Destroy.facet.9b6)]
// CHECK:STDOUT:   %Core.import_ref.946: @T.as_type.as.Iterate.impl.%T.as_type.as.Iterate.impl.NewCursor.type (%T.as_type.as.Iterate.impl.NewCursor.type.8fb) = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.Iterate.impl.%T.as_type.as.Iterate.impl.NewCursor (constants.%T.as_type.as.Iterate.impl.NewCursor.8a1)]
// CHECK:STDOUT:   %Core.import_ref.cfb: @T.as_type.as.Iterate.impl.%T.as_type.as.Iterate.impl.Next.type (%T.as_type.as.Iterate.impl.Next.type.d4a) = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.Iterate.impl.%T.as_type.as.Iterate.impl.Next (constants.%T.as_type.as.Iterate.impl.Next.5a8)]
// CHECK:STDOUT:   %Iterate.impl_witness_table.121 = impl_witness_table (%Core.import_ref.724, %Core.import_ref.dd9, %Core.import_ref.946, %Core.import_ref.cfb), @T.as_type.as.Iterate.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.72f: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.a1d) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.3c1)]
// CHECK:STDOUT:   %Copy.impl_witness_table.f44 = impl_witness_table (%Core.import_ref.72f), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.39b: @From.as_type.as.ImplicitAs.impl.%From.as_type.as.ImplicitAs.impl.Convert.type (%From.as_type.as.ImplicitAs.impl.Convert.type.386) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @From.as_type.as.ImplicitAs.impl.%From.as_type.as.ImplicitAs.impl.Convert (constants.%From.as_type.as.ImplicitAs.impl.Convert.85a)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.8f1 = impl_witness_table (%Core.import_ref.39b), @From.as_type.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.47b: Core.IntLiteral = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.AnyInt.impl.%N (constants.%N.fe9)]
// CHECK:STDOUT:   %Core.import_ref.323: @Int.as.AnyInt.impl.%Int.as.AnyInt.impl.AsInt.type (%Int.as.AnyInt.impl.AsInt.type.0ee) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.AnyInt.impl.%Int.as.AnyInt.impl.AsInt (constants.%Int.as.AnyInt.impl.AsInt.497)]
// CHECK:STDOUT:   %AnyInt.impl_witness_table = impl_witness_table (%Core.import_ref.47b, %Core.import_ref.323), @Int.as.AnyInt.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.b25: @Int.as.AddAssignWith.impl.%Int.as.AddAssignWith.impl.Op.type.2 (%Int.as.AddAssignWith.impl.Op.type.93140c.1) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.AddAssignWith.impl.%Int.as.AddAssignWith.impl.Op.2 (constants.%Int.as.AddAssignWith.impl.Op.75e08e.1)]
// CHECK:STDOUT:   %AddAssignWith.impl_witness_table = impl_witness_table (%Core.import_ref.b25), @Int.as.AddAssignWith.impl [concrete]
// CHECK:STDOUT:   %Core.Op.da0: @Int.as.AddAssignWith.impl.%Int.as.AddAssignWith.impl.Op.type.1 (%Int.as.AddAssignWith.impl.Op.type.93140c.2) = import_ref Core//prelude/types/int, Op, loaded [symbolic = @Int.as.AddAssignWith.impl.%Int.as.AddAssignWith.impl.Op.1 (constants.%Int.as.AddAssignWith.impl.Op.75e08e.2)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @TestIterate(%R.loc30_17.2: %Iterate_where.type.afe) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Iterate.lookup_impl_witness: <witness> = lookup_impl_witness %R.loc30_17.1, @Iterate [symbolic = %Iterate.lookup_impl_witness (constants.%Iterate.lookup_impl_witness.d33)]
// CHECK:STDOUT:   %Iterate.facet.loc34_19.5: %Iterate.type = facet_value %R.as_type.loc30_61.1, (%Iterate.lookup_impl_witness) [symbolic = %Iterate.facet.loc34_19.5 (constants.%Iterate.facet.342)]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.loc34_19.5) [symbolic = %Iterate.WithSelf.NewCursor.type (constants.%Iterate.WithSelf.NewCursor.type.26d)]
// CHECK:STDOUT:   %.loc34_19.23: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type, %Iterate.facet.loc34_19.5 [symbolic = %.loc34_19.23 (constants.%.ef3)]
// CHECK:STDOUT:   %impl.elem2.loc34_19.2: @TestIterate.%.loc34_19.23 (%.ef3) = impl_witness_access %Iterate.lookup_impl_witness, element2 [symbolic = %impl.elem2.loc34_19.2 (constants.%impl.elem2.ca9)]
// CHECK:STDOUT:   %specific_impl_fn.loc34_19.3: <specific function> = specific_impl_function %impl.elem2.loc34_19.2, @Iterate.WithSelf.NewCursor(%Iterate.facet.loc34_19.5) [symbolic = %specific_impl_fn.loc34_19.3 (constants.%specific_impl_fn.c93)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.loc34_19.5) [symbolic = %Iterate.WithSelf.Next.type (constants.%Iterate.WithSelf.Next.type.1e7)]
// CHECK:STDOUT:   %.loc34_19.24: type = fn_type_with_self_type %Iterate.WithSelf.Next.type, %Iterate.facet.loc34_19.5 [symbolic = %.loc34_19.24 (constants.%.ab95)]
// CHECK:STDOUT:   %impl.elem3.loc34_19.2: @TestIterate.%.loc34_19.24 (%.ab95) = impl_witness_access %Iterate.lookup_impl_witness, element3 [symbolic = %impl.elem3.loc34_19.2 (constants.%impl.elem3.e47)]
// CHECK:STDOUT:   %specific_impl_fn.loc34_19.4: <specific function> = specific_impl_function %impl.elem3.loc34_19.2, @Iterate.WithSelf.Next(%Iterate.facet.loc34_19.5) [symbolic = %specific_impl_fn.loc34_19.4 (constants.%specific_impl_fn.c3b)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%r.param: @TestIterate.%R.as_type.loc30_61.1 (%R.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %i.patt: %pattern_type.d94 = value_binding_pattern i [concrete = constants.%i.patt]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %r.ref: @TestIterate.%R.as_type.loc30_61.1 (%R.as_type) = name_ref r, %r
// CHECK:STDOUT:     %impl.elem2.loc34_19.1: @TestIterate.%.loc34_19.23 (%.ef3) = impl_witness_access constants.%Iterate.lookup_impl_witness.d33, element2 [symbolic = %impl.elem2.loc34_19.2 (constants.%impl.elem2.ca9)]
// CHECK:STDOUT:     %bound_method.loc34_19.1: <bound method> = bound_method %r.ref, %impl.elem2.loc34_19.1
// CHECK:STDOUT:     %Iterate.facet.loc34_19.1: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.d33) [symbolic = %Iterate.facet.loc34_19.5 (constants.%Iterate.facet.342)]
// CHECK:STDOUT:     %.loc34_19.1: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc34_19.1 [symbolic = %Iterate.facet.loc34_19.5 (constants.%Iterate.facet.342)]
// CHECK:STDOUT:     %Iterate.facet.loc34_19.2: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.d33) [symbolic = %Iterate.facet.loc34_19.5 (constants.%Iterate.facet.342)]
// CHECK:STDOUT:     %.loc34_19.2: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc34_19.2 [symbolic = %Iterate.facet.loc34_19.5 (constants.%Iterate.facet.342)]
// CHECK:STDOUT:     %specific_impl_fn.loc34_19.1: <specific function> = specific_impl_function %impl.elem2.loc34_19.1, @Iterate.WithSelf.NewCursor(constants.%Iterate.facet.342) [symbolic = %specific_impl_fn.loc34_19.3 (constants.%specific_impl_fn.c93)]
// CHECK:STDOUT:     %bound_method.loc34_19.2: <bound method> = bound_method %r.ref, %specific_impl_fn.loc34_19.1
// CHECK:STDOUT:     %var: ref @TestIterate.%as_type (%as_type.5c5) = var_storage invalid
// CHECK:STDOUT:     %Iterate.WithSelf.NewCursor.call: init @TestIterate.%as_type (%as_type.5c5) to %var = call %bound_method.loc34_19.2(%r.ref)
// CHECK:STDOUT:     assign %var, %Iterate.WithSelf.NewCursor.call
// CHECK:STDOUT:     br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.next:
// CHECK:STDOUT:     %impl.elem3.loc34_19.1: @TestIterate.%.loc34_19.24 (%.ab95) = impl_witness_access constants.%Iterate.lookup_impl_witness.d33, element3 [symbolic = %impl.elem3.loc34_19.2 (constants.%impl.elem3.e47)]
// CHECK:STDOUT:     %bound_method.loc34_19.3: <bound method> = bound_method %r.ref, %impl.elem3.loc34_19.1
// CHECK:STDOUT:     %Iterate.facet.loc34_19.3: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.d33) [symbolic = %Iterate.facet.loc34_19.5 (constants.%Iterate.facet.342)]
// CHECK:STDOUT:     %.loc34_19.3: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc34_19.3 [symbolic = %Iterate.facet.loc34_19.5 (constants.%Iterate.facet.342)]
// CHECK:STDOUT:     %Iterate.facet.loc34_19.4: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.d33) [symbolic = %Iterate.facet.loc34_19.5 (constants.%Iterate.facet.342)]
// CHECK:STDOUT:     %.loc34_19.4: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc34_19.4 [symbolic = %Iterate.facet.loc34_19.5 (constants.%Iterate.facet.342)]
// CHECK:STDOUT:     %.loc34_19.5: %Destroy.type = converted constants.%as_type.5c5, constants.%impl.elem1.732 [symbolic = %impl.elem1 (constants.%impl.elem1.732)]
// CHECK:STDOUT:     %.loc34_19.6: %Destroy.type = converted constants.%as_type.5c5, constants.%impl.elem1.732 [symbolic = %impl.elem1 (constants.%impl.elem1.732)]
// CHECK:STDOUT:     %specific_impl_fn.loc34_19.2: <specific function> = specific_impl_function %impl.elem3.loc34_19.1, @Iterate.WithSelf.Next(constants.%Iterate.facet.342) [symbolic = %specific_impl_fn.loc34_19.4 (constants.%specific_impl_fn.c3b)]
// CHECK:STDOUT:     %bound_method.loc34_19.4: <bound method> = bound_method %r.ref, %specific_impl_fn.loc34_19.2
// CHECK:STDOUT:     %.loc34_19.7: ref %Optional.561 = temporary_storage
// CHECK:STDOUT:     %Iterate.WithSelf.Next.call: init %Optional.561 to %.loc34_19.7 = call %bound_method.loc34_19.4(%r.ref, %var)
// CHECK:STDOUT:     %.loc34_19.8: ref %Optional.561 = temporary %.loc34_19.7, %Iterate.WithSelf.Next.call
// CHECK:STDOUT:     %.loc34_19.9: %Optional.HasValue.type.92b = specific_constant imports.%Core.import_ref.c82, @Optional(constants.%OptionalStorage.facet.072) [concrete = constants.%Optional.HasValue.104]
// CHECK:STDOUT:     %HasValue.ref: %Optional.HasValue.type.92b = name_ref HasValue, %.loc34_19.9 [concrete = constants.%Optional.HasValue.104]
// CHECK:STDOUT:     %Optional.HasValue.bound: <bound method> = bound_method %.loc34_19.8, %HasValue.ref
// CHECK:STDOUT:     %Optional.HasValue.specific_fn: <specific function> = specific_function %HasValue.ref, @Optional.HasValue(constants.%OptionalStorage.facet.072) [concrete = constants.%Optional.HasValue.specific_fn.3c4]
// CHECK:STDOUT:     %bound_method.loc34_19.5: <bound method> = bound_method %.loc34_19.8, %Optional.HasValue.specific_fn
// CHECK:STDOUT:     %.loc34_19.10: %Optional.561 = acquire_value %.loc34_19.8
// CHECK:STDOUT:     %Optional.HasValue.call: init bool = call %bound_method.loc34_19.5(%.loc34_19.10)
// CHECK:STDOUT:     %.loc34_19.11: bool = value_of_initializer %Optional.HasValue.call
// CHECK:STDOUT:     %.loc34_19.12: bool = converted %Optional.HasValue.call, %.loc34_19.11
// CHECK:STDOUT:     if %.loc34_19.12 br !for.body else br !for.done
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.body:
// CHECK:STDOUT:     %.loc34_19.13: %Optional.Get.type.4e6 = specific_constant imports.%Core.import_ref.29e, @Optional(constants.%OptionalStorage.facet.072) [concrete = constants.%Optional.Get.3a9]
// CHECK:STDOUT:     %Get.ref: %Optional.Get.type.4e6 = name_ref Get, %.loc34_19.13 [concrete = constants.%Optional.Get.3a9]
// CHECK:STDOUT:     %Optional.Get.bound: <bound method> = bound_method %.loc34_19.8, %Get.ref
// CHECK:STDOUT:     %Optional.Get.specific_fn: <specific function> = specific_function %Get.ref, @Optional.Get(constants.%OptionalStorage.facet.072) [concrete = constants.%Optional.Get.specific_fn.65d]
// CHECK:STDOUT:     %bound_method.loc34_19.6: <bound method> = bound_method %.loc34_19.8, %Optional.Get.specific_fn
// CHECK:STDOUT:     %.loc34_19.14: %Optional.561 = acquire_value %.loc34_19.8
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
// CHECK:STDOUT:     %Destroy.facet.loc34_19.1: %Destroy.type = facet_value constants.%Optional.561, (constants.%custom_witness.b80ef6.8, constants.%DestroyImpl.impl_witness.dd0) [concrete = constants.%Destroy.facet.58d]
// CHECK:STDOUT:     %.loc34_19.17: %Destroy.type = converted constants.%Optional.561, %Destroy.facet.loc34_19.1 [concrete = constants.%Destroy.facet.58d]
// CHECK:STDOUT:     %Destroy.facet.loc34_19.2: %Destroy.type = facet_value constants.%Optional.561, (constants.%custom_witness.b80ef6.8, constants.%DestroyImpl.impl_witness.dd0) [concrete = constants.%Destroy.facet.58d]
// CHECK:STDOUT:     %.loc34_19.18: %Destroy.type = converted constants.%Optional.561, %Destroy.facet.loc34_19.2 [concrete = constants.%Destroy.facet.58d]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %SelfDestruct.call.loc34_19.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc34_19.8)
// CHECK:STDOUT:     br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.done:
// CHECK:STDOUT:     %Destroy.facet.loc34_19.3: %Destroy.type = facet_value constants.%Optional.561, (constants.%custom_witness.b80ef6.8, constants.%DestroyImpl.impl_witness.dd0) [concrete = constants.%Destroy.facet.58d]
// CHECK:STDOUT:     %.loc34_19.19: %Destroy.type = converted constants.%Optional.561, %Destroy.facet.loc34_19.3 [concrete = constants.%Destroy.facet.58d]
// CHECK:STDOUT:     %Destroy.facet.loc34_19.4: %Destroy.type = facet_value constants.%Optional.561, (constants.%custom_witness.b80ef6.8, constants.%DestroyImpl.impl_witness.dd0) [concrete = constants.%Destroy.facet.58d]
// CHECK:STDOUT:     %.loc34_19.20: %Destroy.type = converted constants.%Optional.561, %Destroy.facet.loc34_19.4 [concrete = constants.%Destroy.facet.58d]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %SelfDestruct.call.loc34_19.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc34_19.8)
// CHECK:STDOUT:     %.loc34_19.21: %Destroy.type = converted constants.%as_type.5c5, constants.%impl.elem1.732 [symbolic = %impl.elem1 (constants.%impl.elem1.732)]
// CHECK:STDOUT:     %.loc34_19.22: %Destroy.type = converted constants.%as_type.5c5, constants.%impl.elem1.732 [symbolic = %impl.elem1 (constants.%impl.elem1.732)]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %SelfDestruct.call.loc34_19.3: init %empty_tuple.type = call %SelfDestruct.specific_fn.3(%var)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc34_19.1(%self.param: ref %.b31) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc34_19.2(%self.param: ref %MaybeUnformed.f2c) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc34_19.3(%self.param: ref bool) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc34_19.4(%self.param: ref %struct_type.value.has_value.6f1) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc34_19.5(%self.param: ref %DefaultOptionalStorage.af2) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc34_19.6(%self.param: ref %Optional.561) = "no_op";
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
// CHECK:STDOUT:   %.loc49_19.5: ref %Optional.561 = temporary_storage
// CHECK:STDOUT:   %.loc49_18.2: %MutableRange = acquire_value %m.ref
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.call.loc49: init %Optional.561 to %.loc49_19.5 = call %bound_method.loc49_19.4(%.loc49_18.2, %var.loc49)
// CHECK:STDOUT:   %.loc49_19.6: ref %Optional.561 = temporary %.loc49_19.5, %T.as_type.as.Iterate.impl.Next.call.loc49
// CHECK:STDOUT:   %.loc49_19.7: %Optional.HasValue.type.92b = specific_constant imports.%Core.import_ref.c82, @Optional(constants.%OptionalStorage.facet.072) [concrete = constants.%Optional.HasValue.104]
// CHECK:STDOUT:   %HasValue.ref.loc49: %Optional.HasValue.type.92b = name_ref HasValue, %.loc49_19.7 [concrete = constants.%Optional.HasValue.104]
// CHECK:STDOUT:   %Optional.HasValue.bound.loc49: <bound method> = bound_method %.loc49_19.6, %HasValue.ref.loc49
// CHECK:STDOUT:   %Optional.HasValue.specific_fn.loc49: <specific function> = specific_function %HasValue.ref.loc49, @Optional.HasValue(constants.%OptionalStorage.facet.072) [concrete = constants.%Optional.HasValue.specific_fn.3c4]
// CHECK:STDOUT:   %bound_method.loc49_19.5: <bound method> = bound_method %.loc49_19.6, %Optional.HasValue.specific_fn.loc49
// CHECK:STDOUT:   %.loc49_19.8: %Optional.561 = acquire_value %.loc49_19.6
// CHECK:STDOUT:   %Optional.HasValue.call.loc49: init bool = call %bound_method.loc49_19.5(%.loc49_19.8)
// CHECK:STDOUT:   %.loc49_19.9: bool = value_of_initializer %Optional.HasValue.call.loc49
// CHECK:STDOUT:   %.loc49_19.10: bool = converted %Optional.HasValue.call.loc49, %.loc49_19.9
// CHECK:STDOUT:   if %.loc49_19.10 br !for.body.loc49 else br !for.done.loc49
// CHECK:STDOUT:
// CHECK:STDOUT: !for.body.loc49:
// CHECK:STDOUT:   %.loc49_19.11: %Optional.Get.type.4e6 = specific_constant imports.%Core.import_ref.29e, @Optional(constants.%OptionalStorage.facet.072) [concrete = constants.%Optional.Get.3a9]
// CHECK:STDOUT:   %Get.ref.loc49: %Optional.Get.type.4e6 = name_ref Get, %.loc49_19.11 [concrete = constants.%Optional.Get.3a9]
// CHECK:STDOUT:   %Optional.Get.bound.loc49: <bound method> = bound_method %.loc49_19.6, %Get.ref.loc49
// CHECK:STDOUT:   %Optional.Get.specific_fn.loc49: <specific function> = specific_function %Get.ref.loc49, @Optional.Get(constants.%OptionalStorage.facet.072) [concrete = constants.%Optional.Get.specific_fn.65d]
// CHECK:STDOUT:   %bound_method.loc49_19.6: <bound method> = bound_method %.loc49_19.6, %Optional.Get.specific_fn.loc49
// CHECK:STDOUT:   %.loc49_19.12: %Optional.561 = acquire_value %.loc49_19.6
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
// CHECK:STDOUT:   %Destroy.facet.loc49_19.1: %Destroy.type = facet_value constants.%Optional.561, (constants.%custom_witness.b80ef6.8, constants.%DestroyImpl.impl_witness.dd0) [concrete = constants.%Destroy.facet.58d]
// CHECK:STDOUT:   %.loc49_19.15: %Destroy.type = converted constants.%Optional.561, %Destroy.facet.loc49_19.1 [concrete = constants.%Destroy.facet.58d]
// CHECK:STDOUT:   %Destroy.facet.loc49_19.2: %Destroy.type = facet_value constants.%Optional.561, (constants.%custom_witness.b80ef6.8, constants.%DestroyImpl.impl_witness.dd0) [concrete = constants.%Destroy.facet.58d]
// CHECK:STDOUT:   %.loc49_19.16: %Destroy.type = converted constants.%Optional.561, %Destroy.facet.loc49_19.2 [concrete = constants.%Destroy.facet.58d]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc49_19.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc49_19.6)
// CHECK:STDOUT:   br !for.next.loc49
// CHECK:STDOUT:
// CHECK:STDOUT: !for.done.loc49:
// CHECK:STDOUT:   %Destroy.facet.loc49_19.3: %Destroy.type = facet_value constants.%Optional.561, (constants.%custom_witness.b80ef6.8, constants.%DestroyImpl.impl_witness.dd0) [concrete = constants.%Destroy.facet.58d]
// CHECK:STDOUT:   %.loc49_19.17: %Destroy.type = converted constants.%Optional.561, %Destroy.facet.loc49_19.3 [concrete = constants.%Destroy.facet.58d]
// CHECK:STDOUT:   %Destroy.facet.loc49_19.4: %Destroy.type = facet_value constants.%Optional.561, (constants.%custom_witness.b80ef6.8, constants.%DestroyImpl.impl_witness.dd0) [concrete = constants.%Destroy.facet.58d]
// CHECK:STDOUT:   %.loc49_19.18: %Destroy.type = converted constants.%Optional.561, %Destroy.facet.loc49_19.4 [concrete = constants.%Destroy.facet.58d]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc49_19.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc49_19.6)
// CHECK:STDOUT:   %Destroy.facet.loc49_19.5: %Destroy.type = facet_value constants.%tuple.type.508, (constants.%custom_witness.b80ef6.10, constants.%DestroyImpl.impl_witness.91e) [concrete = constants.%Destroy.facet.70f]
// CHECK:STDOUT:   %.loc49_19.19: %Destroy.type = converted constants.%tuple.type.508, %Destroy.facet.loc49_19.5 [concrete = constants.%Destroy.facet.70f]
// CHECK:STDOUT:   %Destroy.facet.loc49_19.6: %Destroy.type = facet_value constants.%tuple.type.508, (constants.%custom_witness.b80ef6.10, constants.%DestroyImpl.impl_witness.91e) [concrete = constants.%Destroy.facet.70f]
// CHECK:STDOUT:   %.loc49_19.20: %Destroy.type = converted constants.%tuple.type.508, %Destroy.facet.loc49_19.6 [concrete = constants.%Destroy.facet.70f]
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
// CHECK:STDOUT:   %.loc55_19.5: ref %Optional.561 = temporary_storage
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.call.loc55: init %Optional.561 to %.loc55_19.5 = call %bound_method.loc55_19.4(%c.ref, %var.loc55)
// CHECK:STDOUT:   %.loc55_19.6: ref %Optional.561 = temporary %.loc55_19.5, %T.as_type.as.Iterate.impl.Next.call.loc55
// CHECK:STDOUT:   %.loc55_19.7: %Optional.HasValue.type.92b = specific_constant imports.%Core.import_ref.c82, @Optional(constants.%OptionalStorage.facet.072) [concrete = constants.%Optional.HasValue.104]
// CHECK:STDOUT:   %HasValue.ref.loc55: %Optional.HasValue.type.92b = name_ref HasValue, %.loc55_19.7 [concrete = constants.%Optional.HasValue.104]
// CHECK:STDOUT:   %Optional.HasValue.bound.loc55: <bound method> = bound_method %.loc55_19.6, %HasValue.ref.loc55
// CHECK:STDOUT:   %Optional.HasValue.specific_fn.loc55: <specific function> = specific_function %HasValue.ref.loc55, @Optional.HasValue(constants.%OptionalStorage.facet.072) [concrete = constants.%Optional.HasValue.specific_fn.3c4]
// CHECK:STDOUT:   %bound_method.loc55_19.5: <bound method> = bound_method %.loc55_19.6, %Optional.HasValue.specific_fn.loc55
// CHECK:STDOUT:   %.loc55_19.8: %Optional.561 = acquire_value %.loc55_19.6
// CHECK:STDOUT:   %Optional.HasValue.call.loc55: init bool = call %bound_method.loc55_19.5(%.loc55_19.8)
// CHECK:STDOUT:   %.loc55_19.9: bool = value_of_initializer %Optional.HasValue.call.loc55
// CHECK:STDOUT:   %.loc55_19.10: bool = converted %Optional.HasValue.call.loc55, %.loc55_19.9
// CHECK:STDOUT:   if %.loc55_19.10 br !for.body.loc55 else br !for.done.loc55
// CHECK:STDOUT:
// CHECK:STDOUT: !for.body.loc55:
// CHECK:STDOUT:   %.loc55_19.11: %Optional.Get.type.4e6 = specific_constant imports.%Core.import_ref.29e, @Optional(constants.%OptionalStorage.facet.072) [concrete = constants.%Optional.Get.3a9]
// CHECK:STDOUT:   %Get.ref.loc55: %Optional.Get.type.4e6 = name_ref Get, %.loc55_19.11 [concrete = constants.%Optional.Get.3a9]
// CHECK:STDOUT:   %Optional.Get.bound.loc55: <bound method> = bound_method %.loc55_19.6, %Get.ref.loc55
// CHECK:STDOUT:   %Optional.Get.specific_fn.loc55: <specific function> = specific_function %Get.ref.loc55, @Optional.Get(constants.%OptionalStorage.facet.072) [concrete = constants.%Optional.Get.specific_fn.65d]
// CHECK:STDOUT:   %bound_method.loc55_19.6: <bound method> = bound_method %.loc55_19.6, %Optional.Get.specific_fn.loc55
// CHECK:STDOUT:   %.loc55_19.12: %Optional.561 = acquire_value %.loc55_19.6
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
// CHECK:STDOUT:   %Destroy.facet.loc55_19.1: %Destroy.type = facet_value constants.%Optional.561, (constants.%custom_witness.b80ef6.8, constants.%DestroyImpl.impl_witness.dd0) [concrete = constants.%Destroy.facet.58d]
// CHECK:STDOUT:   %.loc55_19.15: %Destroy.type = converted constants.%Optional.561, %Destroy.facet.loc55_19.1 [concrete = constants.%Destroy.facet.58d]
// CHECK:STDOUT:   %Destroy.facet.loc55_19.2: %Destroy.type = facet_value constants.%Optional.561, (constants.%custom_witness.b80ef6.8, constants.%DestroyImpl.impl_witness.dd0) [concrete = constants.%Destroy.facet.58d]
// CHECK:STDOUT:   %.loc55_19.16: %Destroy.type = converted constants.%Optional.561, %Destroy.facet.loc55_19.2 [concrete = constants.%Destroy.facet.58d]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc55_19.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.4(%.loc55_19.6)
// CHECK:STDOUT:   br !for.next.loc55
// CHECK:STDOUT:
// CHECK:STDOUT: !for.done.loc55:
// CHECK:STDOUT:   %Destroy.facet.loc55_19.3: %Destroy.type = facet_value constants.%Optional.561, (constants.%custom_witness.b80ef6.8, constants.%DestroyImpl.impl_witness.dd0) [concrete = constants.%Destroy.facet.58d]
// CHECK:STDOUT:   %.loc55_19.17: %Destroy.type = converted constants.%Optional.561, %Destroy.facet.loc55_19.3 [concrete = constants.%Destroy.facet.58d]
// CHECK:STDOUT:   %Destroy.facet.loc55_19.4: %Destroy.type = facet_value constants.%Optional.561, (constants.%custom_witness.b80ef6.8, constants.%DestroyImpl.impl_witness.dd0) [concrete = constants.%Destroy.facet.58d]
// CHECK:STDOUT:   %.loc55_19.18: %Destroy.type = converted constants.%Optional.561, %Destroy.facet.loc55_19.4 [concrete = constants.%Destroy.facet.58d]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc55_19.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.5(%.loc55_19.6)
// CHECK:STDOUT:   %Destroy.facet.loc55_19.5: %Destroy.type = facet_value constants.%tuple.type.69f, (constants.%custom_witness.b80ef6.12, constants.%DestroyImpl.impl_witness.4ff) [concrete = constants.%Destroy.facet.141]
// CHECK:STDOUT:   %.loc55_19.19: %Destroy.type = converted constants.%tuple.type.69f, %Destroy.facet.loc55_19.5 [concrete = constants.%Destroy.facet.141]
// CHECK:STDOUT:   %Destroy.facet.loc55_19.6: %Destroy.type = facet_value constants.%tuple.type.69f, (constants.%custom_witness.b80ef6.12, constants.%DestroyImpl.impl_witness.4ff) [concrete = constants.%Destroy.facet.141]
// CHECK:STDOUT:   %.loc55_19.20: %Destroy.type = converted constants.%tuple.type.69f, %Destroy.facet.loc55_19.6 [concrete = constants.%Destroy.facet.141]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc55_19.3: init %empty_tuple.type = call %SelfDestruct.specific_fn.6(%var.loc55)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestIterate(constants.%R) {
// CHECK:STDOUT:   %R.patt.loc30_17.2 => constants.%R.patt
// CHECK:STDOUT:   %R.loc30_17.1 => constants.%R
// CHECK:STDOUT:   %R.as_type.loc30_61.1 => constants.%R.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.0e6
// CHECK:STDOUT:   %r.param_patt.loc30_59.2 => constants.%r.param_patt.874
// CHECK:STDOUT:   %r.patt.loc30_59.2 => constants.%r.patt.97e
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestIterate(constants.%facet_value.0ca) {
// CHECK:STDOUT:   %R.patt.loc30_17.2 => constants.%R.patt
// CHECK:STDOUT:   %R.loc30_17.1 => constants.%facet_value.0ca
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
// CHECK:STDOUT:   %impl.elem1 => constants.%Destroy.facet.70f
// CHECK:STDOUT:   %as_type => constants.%tuple.type.508
// CHECK:STDOUT:   %require_complete.1 => constants.%complete_type.c28
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type => constants.%Iterate.WithSelf.Next.type.88f
// CHECK:STDOUT:   %.loc34_19.24 => constants.%.d9d2
// CHECK:STDOUT:   %impl.elem3.loc34_19.2 => constants.%T.as_type.as.Iterate.impl.Next.a23
// CHECK:STDOUT:   %specific_impl_fn.loc34_19.4 => constants.%T.as_type.as.Iterate.impl.Next.specific_fn.e11
// CHECK:STDOUT:   %SelfDestruct.specific_fn.5 => constants.%SelfDestruct.specific_fn.a77
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestIterate(constants.%facet_value.2e7) {
// CHECK:STDOUT:   %R.patt.loc30_17.2 => constants.%R.patt
// CHECK:STDOUT:   %R.loc30_17.1 => constants.%facet_value.2e7
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
// CHECK:STDOUT:   %impl.elem1 => constants.%Destroy.facet.141
// CHECK:STDOUT:   %as_type => constants.%tuple.type.69f
// CHECK:STDOUT:   %require_complete.1 => constants.%complete_type.8d5
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type => constants.%Iterate.WithSelf.Next.type.4e2
// CHECK:STDOUT:   %.loc34_19.24 => constants.%.4fc
// CHECK:STDOUT:   %impl.elem3.loc34_19.2 => constants.%T.as_type.as.Iterate.impl.Next.283
// CHECK:STDOUT:   %specific_impl_fn.loc34_19.4 => constants.%T.as_type.as.Iterate.impl.Next.specific_fn.d22
// CHECK:STDOUT:   %SelfDestruct.specific_fn.5 => constants.%SelfDestruct.specific_fn.a15
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- methods_return_different_types.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %Iterate.type: type = facet_type <@Iterate> [concrete]
// CHECK:STDOUT:   %OptionalStorage.type: type = facet_type <@OptionalStorage> [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %facet_type.bd0: type = facet_type <@Copy & @Destroy> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %T.220e40.1: %OptionalStorage.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Optional.Get.type.786: type = fn_type @Optional.Get, @Optional(%T.220e40.1) [symbolic]
// CHECK:STDOUT:   %Optional.Get.2d0: %Optional.Get.type.786 = struct_value () [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.type.3ea: type = fn_type @Optional.HasValue, @Optional(%T.220e40.1) [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.f2f: %Optional.HasValue.type.3ea = struct_value () [symbolic]
// CHECK:STDOUT:   %T.f84392.1: %facet_type.bd0 = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.476: type = class_type @DefaultOptionalStorage, @DefaultOptionalStorage(%T.f84392.1) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Get.type.f27: type = fn_type @T.as_type.as.OptionalStorage.impl.Get, @T.as_type.as.OptionalStorage.impl(%T.f84392.1) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Get.e40: %T.as_type.as.OptionalStorage.impl.Get.type.f27 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Has.type.18e: type = fn_type @T.as_type.as.OptionalStorage.impl.Has, @T.as_type.as.OptionalStorage.impl(%T.f84392.1) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Has.463: %T.as_type.as.OptionalStorage.impl.Has.type.18e = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Some.type.292: type = fn_type @T.as_type.as.OptionalStorage.impl.Some, @T.as_type.as.OptionalStorage.impl(%T.f84392.1) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Some.ae9: %T.as_type.as.OptionalStorage.impl.Some.type.292 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.None.type.b57: type = fn_type @T.as_type.as.OptionalStorage.impl.None, @T.as_type.as.OptionalStorage.impl(%T.f84392.1) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.None.791: %T.as_type.as.OptionalStorage.impl.None.type.b57 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SelfDestruct.type: type = fn_type @SelfDestruct [concrete]
// CHECK:STDOUT:   %SelfDestruct: %SelfDestruct.type = struct_value () [concrete]
// CHECK:STDOUT:   %.Self.500: %Iterate.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %CppRange.type: type = facet_type <@CppRange> [concrete]
// CHECK:STDOUT:   %.Self.e0b: %CppRange.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %CppRangeForIterate.lookup_impl_witness.50b: <witness> = lookup_impl_witness %.Self.e0b, @CppRangeForIterate [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.32d3: type = impl_witness_access %CppRangeForIterate.lookup_impl_witness.50b, element0 [symbolic_self]
// CHECK:STDOUT:   %CppUnsafeDeref.lookup_impl_witness.851: <witness> = lookup_impl_witness %impl.elem0.32d3, @CppUnsafeDeref [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.10d: type = impl_witness_access %CppUnsafeDeref.lookup_impl_witness.851, element0 [symbolic_self]
// CHECK:STDOUT:   %CppRange_where.type.d682d6.1: type = facet_type <@CppRange where %impl.elem0.10d impls @Copy> [concrete]
// CHECK:STDOUT:   %T.f45530.3: %CppRange_where.type.d682d6.1 = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.type.d4a: type = fn_type @T.as_type.as.Iterate.impl.Next, @T.as_type.as.Iterate.impl(%T.f45530.3) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.5a8: %T.as_type.as.Iterate.impl.Next.type.d4a = struct_value () [symbolic]
// CHECK:STDOUT:   %CppRangeForIterate.lookup_impl_witness.b581a8.2: <witness> = lookup_impl_witness %T.f45530.3, @CppRangeForIterate [symbolic]
// CHECK:STDOUT:   %impl.elem0.5f971c.2: type = impl_witness_access %CppRangeForIterate.lookup_impl_witness.b581a8.2, element0 [symbolic]
// CHECK:STDOUT:   %CppUnsafeDeref.lookup_impl_witness.87e543.2: <witness> = lookup_impl_witness %impl.elem0.5f971c.2, @CppUnsafeDeref [symbolic]
// CHECK:STDOUT:   %impl.elem0.493765.2: type = impl_witness_access %CppUnsafeDeref.lookup_impl_witness.87e543.2, element0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.type.8fb: type = fn_type @T.as_type.as.Iterate.impl.NewCursor, @T.as_type.as.Iterate.impl(%T.f45530.3) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.8a1: %T.as_type.as.Iterate.impl.NewCursor.type.8fb = struct_value () [symbolic]
// CHECK:STDOUT:   %impl.elem1.c2dc15.2: type = impl_witness_access %CppRangeForIterate.lookup_impl_witness.b581a8.2, element1 [symbolic]
// CHECK:STDOUT:   %tuple.type.1ba: type = tuple_type (%impl.elem0.5f971c.2, %impl.elem1.c2dc15.2) [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.lookup_impl_witness.dc9: <witness> = lookup_impl_witness %tuple.type.1ba, @SubobjectDestroy [symbolic]
// CHECK:STDOUT:   %DestroyImpl.lookup_impl_witness.504: <witness> = lookup_impl_witness %tuple.type.1ba, @DestroyImpl [symbolic]
// CHECK:STDOUT:   %Destroy.facet.9b6: %Destroy.type = facet_value %tuple.type.1ba, (%SubobjectDestroy.lookup_impl_witness.dc9, %DestroyImpl.lookup_impl_witness.504) [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.lookup_impl_witness.b15da7.2: <witness> = lookup_impl_witness %impl.elem0.493765.2, @SubobjectDestroy [symbolic]
// CHECK:STDOUT:   %DestroyImpl.lookup_impl_witness.4e7f44.2: <witness> = lookup_impl_witness %impl.elem0.493765.2, @DestroyImpl [symbolic]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.167: <witness> = lookup_impl_witness %impl.elem0.493765.2, @Copy [symbolic]
// CHECK:STDOUT:   %facet_value.189: %facet_type.bd0 = facet_value %impl.elem0.493765.2, (%SubobjectDestroy.lookup_impl_witness.b15da7.2, %DestroyImpl.lookup_impl_witness.4e7f44.2, %Copy.lookup_impl_witness.167) [symbolic]
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
// CHECK:STDOUT:   %facet_value.722: %facet_type.bd0 = facet_value %DiffValueType, (%custom_witness.b80ef6.2, %DestroyImpl.impl_witness.f43, %custom_witness.d0da43.1) [concrete]
// CHECK:STDOUT:   %Iterate.lookup_impl_witness.1d8: <witness> = lookup_impl_witness %.Self.500, @Iterate [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.4ce: %facet_type.bd0 = impl_witness_access %Iterate.lookup_impl_witness.1d8, element0 [symbolic_self]
// CHECK:STDOUT:   %Iterate_where.type.e6b: type = facet_type <@Iterate where %impl.elem0.4ce = %facet_value.722> [concrete]
// CHECK:STDOUT:   %pattern_type.4b1: type = pattern_type %Iterate_where.type.e6b [concrete]
// CHECK:STDOUT:   %R.patt: %pattern_type.4b1 = symbolic_binding_pattern R, 0 [symbolic]
// CHECK:STDOUT:   %R: %Iterate_where.type.e6b = symbolic_binding R, 0 [symbolic]
// CHECK:STDOUT:   %R.as_type: type = facet_access_type %R [symbolic]
// CHECK:STDOUT:   %pattern_type.db2: type = pattern_type %R.as_type [symbolic]
// CHECK:STDOUT:   %r.patt.4a5: %pattern_type.db2 = ref_binding_pattern r [symbolic]
// CHECK:STDOUT:   %r.param_patt.24f: %pattern_type.db2 = var_param_pattern %r.patt.4a5 [symbolic]
// CHECK:STDOUT:   %i.patt: %pattern_type.149 = value_binding_pattern i [concrete]
// CHECK:STDOUT:   %Iterate.lookup_impl_witness.1e9: <witness> = lookup_impl_witness %R, @Iterate [symbolic]
// CHECK:STDOUT:   %Iterate.facet.bc3: %Iterate.type = facet_value %R.as_type, (%Iterate.lookup_impl_witness.1e9) [symbolic]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.07b: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.bc3) [symbolic]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.c29: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.bc3) [symbolic]
// CHECK:STDOUT:   %.fd9: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.07b, %Iterate.facet.bc3 [symbolic]
// CHECK:STDOUT:   %impl.elem2.314: %.fd9 = impl_witness_access %Iterate.lookup_impl_witness.1e9, element2 [symbolic]
// CHECK:STDOUT:   %impl.elem1.4a5: %Destroy.type = impl_witness_access %Iterate.lookup_impl_witness.1e9, element1 [symbolic]
// CHECK:STDOUT:   %as_type.4e3: type = facet_access_type %impl.elem1.4a5 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.1b7: <specific function> = specific_impl_function %impl.elem2.314, @Iterate.WithSelf.NewCursor(%Iterate.facet.bc3) [symbolic]
// CHECK:STDOUT:   %.67c9: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.c29, %Iterate.facet.bc3 [symbolic]
// CHECK:STDOUT:   %impl.elem3.f93: %.67c9 = impl_witness_access %Iterate.lookup_impl_witness.1e9, element3 [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.dc7: type = class_type @DefaultOptionalStorage, @DefaultOptionalStorage(%facet_value.722) [concrete]
// CHECK:STDOUT:   %OptionalStorage.impl_witness.f27: <witness> = impl_witness imports.%OptionalStorage.impl_witness_table.1c4, @T.as_type.as.OptionalStorage.impl(%facet_value.722) [concrete]
// CHECK:STDOUT:   %OptionalStorage.facet.d73: %OptionalStorage.type = facet_value %DiffValueType, (%OptionalStorage.impl_witness.f27) [concrete]
// CHECK:STDOUT:   %Optional.750: type = class_type @Optional, @Optional(%OptionalStorage.facet.d73) [concrete]
// CHECK:STDOUT:   %specific_impl_fn.32c: <specific function> = specific_impl_function %impl.elem3.f93, @Iterate.WithSelf.Next(%Iterate.facet.bc3) [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.type.4da: type = fn_type @Optional.HasValue, @Optional(%OptionalStorage.facet.d73) [concrete]
// CHECK:STDOUT:   %Optional.HasValue.9f4: %Optional.HasValue.type.4da = struct_value () [concrete]
// CHECK:STDOUT:   %Optional.Get.type.b00: type = fn_type @Optional.Get, @Optional(%OptionalStorage.facet.d73) [concrete]
// CHECK:STDOUT:   %Optional.Get.e30: %Optional.Get.type.b00 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.facet.95d: %Destroy.type = facet_value %DiffValueType, (%custom_witness.b80ef6.2, %DestroyImpl.impl_witness.f43) [concrete]
// CHECK:STDOUT:   %MaybeUnformed.c01: type = class_type @MaybeUnformed, @MaybeUnformed(%Destroy.facet.95d) [concrete]
// CHECK:STDOUT:   %.a18: type = maybe_unformed_type %DiffValueType [concrete]
// CHECK:STDOUT:   %struct_type.value.has_value.732: type = struct_type {.value: %MaybeUnformed.c01, .has_value: bool} [concrete]
// CHECK:STDOUT:   %Optional.HasValue.specific_fn.57a: <specific function> = specific_function %Optional.HasValue.9f4, @Optional.HasValue(%OptionalStorage.facet.d73) [concrete]
// CHECK:STDOUT:   %Optional.Get.specific_fn.a58: <specific function> = specific_function %Optional.Get.e30, @Optional.Get(%OptionalStorage.facet.d73) [concrete]
// CHECK:STDOUT:   %operator_PlusEqual__carbon_thunk.type: type = fn_type @operator_PlusEqual__carbon_thunk [concrete]
// CHECK:STDOUT:   %operator_PlusEqual__carbon_thunk: %operator_PlusEqual__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.8: type = fn_type @SubobjectDestroy.WithSelf.Op.loc44_33.6 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.8: %SubobjectDestroy.WithSelf.Op.type.6870fd.8 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.8: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.8), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.527: %SubobjectDestroy.type = facet_value %Optional.750, (%custom_witness.b80ef6.8) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.41e: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.527) [concrete]
// CHECK:STDOUT:   %Destroy.facet.ee1: %Destroy.type = facet_value %Optional.750, (%custom_witness.b80ef6.8, %DestroyImpl.impl_witness.41e) [concrete]
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
// CHECK:STDOUT:   %Destroy.facet.4da: %Destroy.type = facet_value %tuple.type.b18, (%custom_witness.b80ef6.11, %DestroyImpl.impl_witness.c40) [concrete]
// CHECK:STDOUT:   %Iterate.impl_witness.706: <witness> = impl_witness imports.%Iterate.impl_witness_table.121, @T.as_type.as.Iterate.impl(%facet_value.f44) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.type.594: type = fn_type @T.as_type.as.Iterate.impl.NewCursor, @T.as_type.as.Iterate.impl(%facet_value.f44) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.097: %T.as_type.as.Iterate.impl.NewCursor.type.594 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.type.033: type = fn_type @T.as_type.as.Iterate.impl.Next, @T.as_type.as.Iterate.impl(%facet_value.f44) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.c89: %T.as_type.as.Iterate.impl.Next.type.033 = struct_value () [concrete]
// CHECK:STDOUT:   %facet_value.b26: %Iterate_where.type.e6b = facet_value %DiffMutableRange, (%Iterate.impl_witness.706) [concrete]
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
// CHECK:STDOUT:   %Destroy.facet.017: %Destroy.type = facet_value %tuple.type.df5, (%custom_witness.b80ef6.15, %DestroyImpl.impl_witness.8c3) [concrete]
// CHECK:STDOUT:   %Iterate.impl_witness.df4: <witness> = impl_witness imports.%Iterate.impl_witness_table.121, @T.as_type.as.Iterate.impl(%facet_value.115) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.type.18c: type = fn_type @T.as_type.as.Iterate.impl.NewCursor, @T.as_type.as.Iterate.impl(%facet_value.115) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.416: %T.as_type.as.Iterate.impl.NewCursor.type.18c = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.type.e78: type = fn_type @T.as_type.as.Iterate.impl.Next, @T.as_type.as.Iterate.impl(%facet_value.115) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.b68: %T.as_type.as.Iterate.impl.Next.type.e78 = struct_value () [concrete]
// CHECK:STDOUT:   %facet_value.635: %Iterate_where.type.e6b = facet_value %DiffConstRange, (%Iterate.impl_witness.df4) [concrete]
// CHECK:STDOUT:   %r.patt.9c0: %pattern_type.7dd = ref_binding_pattern r [concrete]
// CHECK:STDOUT:   %r.param_patt.992: %pattern_type.7dd = var_param_pattern %r.patt.9c0 [concrete]
// CHECK:STDOUT:   %Iterate.facet.9f6: %Iterate.type = facet_value %DiffMutableRange, (%Iterate.impl_witness.706) [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.e12: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.9f6) [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.aad: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.9f6) [concrete]
// CHECK:STDOUT:   %.c87: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.e12, %Iterate.facet.9f6 [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.specific_fn.5a0: <specific function> = specific_function %T.as_type.as.Iterate.impl.NewCursor.097, @T.as_type.as.Iterate.impl.NewCursor(%facet_value.f44) [concrete]
// CHECK:STDOUT:   %.5db8: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.aad, %Iterate.facet.9f6 [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.specific_fn.134: <specific function> = specific_function %T.as_type.as.Iterate.impl.Next.c89, @T.as_type.as.Iterate.impl.Next(%facet_value.f44) [concrete]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.9e00: <specific function> = specific_function %SelfDestruct, @SelfDestruct(%Destroy.facet.4da) [concrete]
// CHECK:STDOUT:   %Iterate.facet.3d3: %Iterate.type = facet_value %DiffConstRange, (%Iterate.impl_witness.df4) [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.c77: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.3d3) [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.f88: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.3d3) [concrete]
// CHECK:STDOUT:   %.afc: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.c77, %Iterate.facet.3d3 [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.specific_fn.0cc: <specific function> = specific_function %T.as_type.as.Iterate.impl.NewCursor.416, @T.as_type.as.Iterate.impl.NewCursor(%facet_value.115) [concrete]
// CHECK:STDOUT:   %.5be: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.f88, %Iterate.facet.3d3 [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.specific_fn.4d3: <specific function> = specific_function %T.as_type.as.Iterate.impl.Next.b68, @T.as_type.as.Iterate.impl.Next(%facet_value.115) [concrete]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.825: <specific function> = specific_function %SelfDestruct, @SelfDestruct(%Destroy.facet.017) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .DiffValueType = %DiffValueType.decl
// CHECK:STDOUT:     .DiffMutableRange = %DiffMutableRange.decl
// CHECK:STDOUT:     .DiffConstRange = %DiffConstRange.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.481: type = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%DefaultOptionalStorage (constants.%DefaultOptionalStorage.476)]
// CHECK:STDOUT:   %Core.import_ref.66d: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.None.type (%T.as_type.as.OptionalStorage.impl.None.type.b57) = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.None (constants.%T.as_type.as.OptionalStorage.impl.None.791)]
// CHECK:STDOUT:   %Core.import_ref.32c: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Some.type (%T.as_type.as.OptionalStorage.impl.Some.type.292) = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Some (constants.%T.as_type.as.OptionalStorage.impl.Some.ae9)]
// CHECK:STDOUT:   %Core.import_ref.801: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Has.type (%T.as_type.as.OptionalStorage.impl.Has.type.18e) = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Has (constants.%T.as_type.as.OptionalStorage.impl.Has.463)]
// CHECK:STDOUT:   %Core.import_ref.eec: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Get.type (%T.as_type.as.OptionalStorage.impl.Get.type.f27) = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Get (constants.%T.as_type.as.OptionalStorage.impl.Get.e40)]
// CHECK:STDOUT:   %Core.import_ref.a6a = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %OptionalStorage.impl_witness_table.1c4 = impl_witness_table (%Core.import_ref.481, %Core.import_ref.66d, %Core.import_ref.32c, %Core.import_ref.801, %Core.import_ref.eec, %Core.import_ref.a6a), @T.as_type.as.OptionalStorage.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.7b5: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/types/optional, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.7b5), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.c82: @Optional.%Optional.HasValue.type (%Optional.HasValue.type.3ea) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.HasValue (constants.%Optional.HasValue.f2f)]
// CHECK:STDOUT:   %Core.import_ref.29e: @Optional.%Optional.Get.type (%Optional.Get.type.786) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.Get (constants.%Optional.Get.2d0)]
// CHECK:STDOUT:   %Core.import_ref.724: %facet_type.bd0 = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.Iterate.impl.%facet_value (constants.%facet_value.189)]
// CHECK:STDOUT:   %Core.import_ref.dd9: %Destroy.type = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.Iterate.impl.%Destroy.facet (constants.%Destroy.facet.9b6)]
// CHECK:STDOUT:   %Core.import_ref.946: @T.as_type.as.Iterate.impl.%T.as_type.as.Iterate.impl.NewCursor.type (%T.as_type.as.Iterate.impl.NewCursor.type.8fb) = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.Iterate.impl.%T.as_type.as.Iterate.impl.NewCursor (constants.%T.as_type.as.Iterate.impl.NewCursor.8a1)]
// CHECK:STDOUT:   %Core.import_ref.cfb: @T.as_type.as.Iterate.impl.%T.as_type.as.Iterate.impl.Next.type (%T.as_type.as.Iterate.impl.Next.type.d4a) = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.Iterate.impl.%T.as_type.as.Iterate.impl.Next (constants.%T.as_type.as.Iterate.impl.Next.5a8)]
// CHECK:STDOUT:   %Iterate.impl_witness_table.121 = impl_witness_table (%Core.import_ref.724, %Core.import_ref.dd9, %Core.import_ref.946, %Core.import_ref.cfb), @T.as_type.as.Iterate.impl [concrete]
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
// CHECK:STDOUT: generic fn @TestIterate(%R.loc40_17.2: %Iterate_where.type.e6b) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Iterate.lookup_impl_witness: <witness> = lookup_impl_witness %R.loc40_17.1, @Iterate [symbolic = %Iterate.lookup_impl_witness (constants.%Iterate.lookup_impl_witness.1e9)]
// CHECK:STDOUT:   %Iterate.facet.loc44_33.5: %Iterate.type = facet_value %R.as_type.loc40_79.1, (%Iterate.lookup_impl_witness) [symbolic = %Iterate.facet.loc44_33.5 (constants.%Iterate.facet.bc3)]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.loc44_33.5) [symbolic = %Iterate.WithSelf.NewCursor.type (constants.%Iterate.WithSelf.NewCursor.type.07b)]
// CHECK:STDOUT:   %.loc44_33.26: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type, %Iterate.facet.loc44_33.5 [symbolic = %.loc44_33.26 (constants.%.fd9)]
// CHECK:STDOUT:   %impl.elem2.loc44_33.2: @TestIterate.%.loc44_33.26 (%.fd9) = impl_witness_access %Iterate.lookup_impl_witness, element2 [symbolic = %impl.elem2.loc44_33.2 (constants.%impl.elem2.314)]
// CHECK:STDOUT:   %specific_impl_fn.loc44_33.3: <specific function> = specific_impl_function %impl.elem2.loc44_33.2, @Iterate.WithSelf.NewCursor(%Iterate.facet.loc44_33.5) [symbolic = %specific_impl_fn.loc44_33.3 (constants.%specific_impl_fn.1b7)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.loc44_33.5) [symbolic = %Iterate.WithSelf.Next.type (constants.%Iterate.WithSelf.Next.type.c29)]
// CHECK:STDOUT:   %.loc44_33.27: type = fn_type_with_self_type %Iterate.WithSelf.Next.type, %Iterate.facet.loc44_33.5 [symbolic = %.loc44_33.27 (constants.%.67c9)]
// CHECK:STDOUT:   %impl.elem3.loc44_33.2: @TestIterate.%.loc44_33.27 (%.67c9) = impl_witness_access %Iterate.lookup_impl_witness, element3 [symbolic = %impl.elem3.loc44_33.2 (constants.%impl.elem3.f93)]
// CHECK:STDOUT:   %specific_impl_fn.loc44_33.4: <specific function> = specific_impl_function %impl.elem3.loc44_33.2, @Iterate.WithSelf.Next(%Iterate.facet.loc44_33.5) [symbolic = %specific_impl_fn.loc44_33.4 (constants.%specific_impl_fn.32c)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%r.param: ref @TestIterate.%R.as_type.loc40_79.1 (%R.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %i.patt: %pattern_type.149 = value_binding_pattern i [concrete = constants.%i.patt]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %r.ref: ref @TestIterate.%R.as_type.loc40_79.1 (%R.as_type) = name_ref r, %r
// CHECK:STDOUT:     %impl.elem2.loc44_33.1: @TestIterate.%.loc44_33.26 (%.fd9) = impl_witness_access constants.%Iterate.lookup_impl_witness.1e9, element2 [symbolic = %impl.elem2.loc44_33.2 (constants.%impl.elem2.314)]
// CHECK:STDOUT:     %bound_method.loc44_33.1: <bound method> = bound_method %r.ref, %impl.elem2.loc44_33.1
// CHECK:STDOUT:     %Iterate.facet.loc44_33.1: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.1e9) [symbolic = %Iterate.facet.loc44_33.5 (constants.%Iterate.facet.bc3)]
// CHECK:STDOUT:     %.loc44_33.1: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc44_33.1 [symbolic = %Iterate.facet.loc44_33.5 (constants.%Iterate.facet.bc3)]
// CHECK:STDOUT:     %Iterate.facet.loc44_33.2: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.1e9) [symbolic = %Iterate.facet.loc44_33.5 (constants.%Iterate.facet.bc3)]
// CHECK:STDOUT:     %.loc44_33.2: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc44_33.2 [symbolic = %Iterate.facet.loc44_33.5 (constants.%Iterate.facet.bc3)]
// CHECK:STDOUT:     %specific_impl_fn.loc44_33.1: <specific function> = specific_impl_function %impl.elem2.loc44_33.1, @Iterate.WithSelf.NewCursor(constants.%Iterate.facet.bc3) [symbolic = %specific_impl_fn.loc44_33.3 (constants.%specific_impl_fn.1b7)]
// CHECK:STDOUT:     %bound_method.loc44_33.2: <bound method> = bound_method %r.ref, %specific_impl_fn.loc44_33.1
// CHECK:STDOUT:     %var: ref @TestIterate.%as_type (%as_type.4e3) = var_storage invalid
// CHECK:STDOUT:     %.loc44_32.1: @TestIterate.%R.as_type.loc40_79.1 (%R.as_type) = acquire_value %r.ref
// CHECK:STDOUT:     %Iterate.WithSelf.NewCursor.call: init @TestIterate.%as_type (%as_type.4e3) to %var = call %bound_method.loc44_33.2(%.loc44_32.1)
// CHECK:STDOUT:     assign %var, %Iterate.WithSelf.NewCursor.call
// CHECK:STDOUT:     br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.next:
// CHECK:STDOUT:     %impl.elem3.loc44_33.1: @TestIterate.%.loc44_33.27 (%.67c9) = impl_witness_access constants.%Iterate.lookup_impl_witness.1e9, element3 [symbolic = %impl.elem3.loc44_33.2 (constants.%impl.elem3.f93)]
// CHECK:STDOUT:     %bound_method.loc44_33.3: <bound method> = bound_method %r.ref, %impl.elem3.loc44_33.1
// CHECK:STDOUT:     %Iterate.facet.loc44_33.3: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.1e9) [symbolic = %Iterate.facet.loc44_33.5 (constants.%Iterate.facet.bc3)]
// CHECK:STDOUT:     %.loc44_33.3: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc44_33.3 [symbolic = %Iterate.facet.loc44_33.5 (constants.%Iterate.facet.bc3)]
// CHECK:STDOUT:     %Iterate.facet.loc44_33.4: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.1e9) [symbolic = %Iterate.facet.loc44_33.5 (constants.%Iterate.facet.bc3)]
// CHECK:STDOUT:     %.loc44_33.4: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc44_33.4 [symbolic = %Iterate.facet.loc44_33.5 (constants.%Iterate.facet.bc3)]
// CHECK:STDOUT:     %.loc44_33.5: %Destroy.type = converted constants.%as_type.4e3, constants.%impl.elem1.4a5 [symbolic = %impl.elem1 (constants.%impl.elem1.4a5)]
// CHECK:STDOUT:     %.loc44_33.6: %Destroy.type = converted constants.%as_type.4e3, constants.%impl.elem1.4a5 [symbolic = %impl.elem1 (constants.%impl.elem1.4a5)]
// CHECK:STDOUT:     %specific_impl_fn.loc44_33.2: <specific function> = specific_impl_function %impl.elem3.loc44_33.1, @Iterate.WithSelf.Next(constants.%Iterate.facet.bc3) [symbolic = %specific_impl_fn.loc44_33.4 (constants.%specific_impl_fn.32c)]
// CHECK:STDOUT:     %bound_method.loc44_33.4: <bound method> = bound_method %r.ref, %specific_impl_fn.loc44_33.2
// CHECK:STDOUT:     %.loc44_33.7: ref %Optional.750 = temporary_storage
// CHECK:STDOUT:     %.loc44_32.2: @TestIterate.%R.as_type.loc40_79.1 (%R.as_type) = acquire_value %r.ref
// CHECK:STDOUT:     %Iterate.WithSelf.Next.call: init %Optional.750 to %.loc44_33.7 = call %bound_method.loc44_33.4(%.loc44_32.2, %var)
// CHECK:STDOUT:     %.loc44_33.8: ref %Optional.750 = temporary %.loc44_33.7, %Iterate.WithSelf.Next.call
// CHECK:STDOUT:     %.loc44_33.9: %Optional.HasValue.type.4da = specific_constant imports.%Core.import_ref.c82, @Optional(constants.%OptionalStorage.facet.d73) [concrete = constants.%Optional.HasValue.9f4]
// CHECK:STDOUT:     %HasValue.ref: %Optional.HasValue.type.4da = name_ref HasValue, %.loc44_33.9 [concrete = constants.%Optional.HasValue.9f4]
// CHECK:STDOUT:     %Optional.HasValue.bound: <bound method> = bound_method %.loc44_33.8, %HasValue.ref
// CHECK:STDOUT:     %Optional.HasValue.specific_fn: <specific function> = specific_function %HasValue.ref, @Optional.HasValue(constants.%OptionalStorage.facet.d73) [concrete = constants.%Optional.HasValue.specific_fn.57a]
// CHECK:STDOUT:     %bound_method.loc44_33.5: <bound method> = bound_method %.loc44_33.8, %Optional.HasValue.specific_fn
// CHECK:STDOUT:     %.loc44_33.10: %Optional.750 = acquire_value %.loc44_33.8
// CHECK:STDOUT:     %Optional.HasValue.call: init bool = call %bound_method.loc44_33.5(%.loc44_33.10)
// CHECK:STDOUT:     %.loc44_33.11: bool = value_of_initializer %Optional.HasValue.call
// CHECK:STDOUT:     %.loc44_33.12: bool = converted %Optional.HasValue.call, %.loc44_33.11
// CHECK:STDOUT:     if %.loc44_33.12 br !for.body else br !for.done
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.body:
// CHECK:STDOUT:     %.loc44_33.13: %Optional.Get.type.b00 = specific_constant imports.%Core.import_ref.29e, @Optional(constants.%OptionalStorage.facet.d73) [concrete = constants.%Optional.Get.e30]
// CHECK:STDOUT:     %Get.ref: %Optional.Get.type.b00 = name_ref Get, %.loc44_33.13 [concrete = constants.%Optional.Get.e30]
// CHECK:STDOUT:     %Optional.Get.bound: <bound method> = bound_method %.loc44_33.8, %Get.ref
// CHECK:STDOUT:     %Optional.Get.specific_fn: <specific function> = specific_function %Get.ref, @Optional.Get(constants.%OptionalStorage.facet.d73) [concrete = constants.%Optional.Get.specific_fn.a58]
// CHECK:STDOUT:     %bound_method.loc44_33.6: <bound method> = bound_method %.loc44_33.8, %Optional.Get.specific_fn
// CHECK:STDOUT:     %.loc44_33.14: ref %DiffValueType = temporary_storage
// CHECK:STDOUT:     %.loc44_33.15: %Optional.750 = acquire_value %.loc44_33.8
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
// CHECK:STDOUT:     %Destroy.facet.loc44_33.1: %Destroy.type = facet_value constants.%DiffValueType, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.f43) [concrete = constants.%Destroy.facet.95d]
// CHECK:STDOUT:     %.loc44_33.18: %Destroy.type = converted constants.%DiffValueType, %Destroy.facet.loc44_33.1 [concrete = constants.%Destroy.facet.95d]
// CHECK:STDOUT:     %Destroy.facet.loc44_33.2: %Destroy.type = facet_value constants.%DiffValueType, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.f43) [concrete = constants.%Destroy.facet.95d]
// CHECK:STDOUT:     %.loc44_33.19: %Destroy.type = converted constants.%DiffValueType, %Destroy.facet.loc44_33.2 [concrete = constants.%Destroy.facet.95d]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %SelfDestruct.call.loc44_33.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc44_33.16)
// CHECK:STDOUT:     %Destroy.facet.loc44_33.3: %Destroy.type = facet_value constants.%Optional.750, (constants.%custom_witness.b80ef6.8, constants.%DestroyImpl.impl_witness.41e) [concrete = constants.%Destroy.facet.ee1]
// CHECK:STDOUT:     %.loc44_33.20: %Destroy.type = converted constants.%Optional.750, %Destroy.facet.loc44_33.3 [concrete = constants.%Destroy.facet.ee1]
// CHECK:STDOUT:     %Destroy.facet.loc44_33.4: %Destroy.type = facet_value constants.%Optional.750, (constants.%custom_witness.b80ef6.8, constants.%DestroyImpl.impl_witness.41e) [concrete = constants.%Destroy.facet.ee1]
// CHECK:STDOUT:     %.loc44_33.21: %Destroy.type = converted constants.%Optional.750, %Destroy.facet.loc44_33.4 [concrete = constants.%Destroy.facet.ee1]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %SelfDestruct.call.loc44_33.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc44_33.8)
// CHECK:STDOUT:     br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.done:
// CHECK:STDOUT:     %Destroy.facet.loc44_33.5: %Destroy.type = facet_value constants.%Optional.750, (constants.%custom_witness.b80ef6.8, constants.%DestroyImpl.impl_witness.41e) [concrete = constants.%Destroy.facet.ee1]
// CHECK:STDOUT:     %.loc44_33.22: %Destroy.type = converted constants.%Optional.750, %Destroy.facet.loc44_33.5 [concrete = constants.%Destroy.facet.ee1]
// CHECK:STDOUT:     %Destroy.facet.loc44_33.6: %Destroy.type = facet_value constants.%Optional.750, (constants.%custom_witness.b80ef6.8, constants.%DestroyImpl.impl_witness.41e) [concrete = constants.%Destroy.facet.ee1]
// CHECK:STDOUT:     %.loc44_33.23: %Destroy.type = converted constants.%Optional.750, %Destroy.facet.loc44_33.6 [concrete = constants.%Destroy.facet.ee1]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %SelfDestruct.call.loc44_33.3: init %empty_tuple.type = call %SelfDestruct.specific_fn.3(%.loc44_33.8)
// CHECK:STDOUT:     %.loc44_33.24: %Destroy.type = converted constants.%as_type.4e3, constants.%impl.elem1.4a5 [symbolic = %impl.elem1 (constants.%impl.elem1.4a5)]
// CHECK:STDOUT:     %.loc44_33.25: %Destroy.type = converted constants.%as_type.4e3, constants.%impl.elem1.4a5 [symbolic = %impl.elem1 (constants.%impl.elem1.4a5)]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %SelfDestruct.call.loc44_33.4: init %empty_tuple.type = call %SelfDestruct.specific_fn.4(%var)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc44_33.1(%self.param: ref %.a18) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc44_33.2(%self.param: ref %MaybeUnformed.c01) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc44_33.3(%self.param: ref bool) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc44_33.4(%self.param: ref %struct_type.value.has_value.732) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc44_33.5(%self.param: ref %DefaultOptionalStorage.dc7) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc44_33.6(%self.param: ref %Optional.750) = "no_op";
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
// CHECK:STDOUT:   %.loc59_33.5: ref %Optional.750 = temporary_storage
// CHECK:STDOUT:   %.loc59_32.2: %DiffMutableRange = acquire_value %m.ref
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.call.loc59: init %Optional.750 to %.loc59_33.5 = call %bound_method.loc59_33.4(%.loc59_32.2, %var.loc59)
// CHECK:STDOUT:   %.loc59_33.6: ref %Optional.750 = temporary %.loc59_33.5, %T.as_type.as.Iterate.impl.Next.call.loc59
// CHECK:STDOUT:   %.loc59_33.7: %Optional.HasValue.type.4da = specific_constant imports.%Core.import_ref.c82, @Optional(constants.%OptionalStorage.facet.d73) [concrete = constants.%Optional.HasValue.9f4]
// CHECK:STDOUT:   %HasValue.ref.loc59: %Optional.HasValue.type.4da = name_ref HasValue, %.loc59_33.7 [concrete = constants.%Optional.HasValue.9f4]
// CHECK:STDOUT:   %Optional.HasValue.bound.loc59: <bound method> = bound_method %.loc59_33.6, %HasValue.ref.loc59
// CHECK:STDOUT:   %Optional.HasValue.specific_fn.loc59: <specific function> = specific_function %HasValue.ref.loc59, @Optional.HasValue(constants.%OptionalStorage.facet.d73) [concrete = constants.%Optional.HasValue.specific_fn.57a]
// CHECK:STDOUT:   %bound_method.loc59_33.5: <bound method> = bound_method %.loc59_33.6, %Optional.HasValue.specific_fn.loc59
// CHECK:STDOUT:   %.loc59_33.8: %Optional.750 = acquire_value %.loc59_33.6
// CHECK:STDOUT:   %Optional.HasValue.call.loc59: init bool = call %bound_method.loc59_33.5(%.loc59_33.8)
// CHECK:STDOUT:   %.loc59_33.9: bool = value_of_initializer %Optional.HasValue.call.loc59
// CHECK:STDOUT:   %.loc59_33.10: bool = converted %Optional.HasValue.call.loc59, %.loc59_33.9
// CHECK:STDOUT:   if %.loc59_33.10 br !for.body.loc59 else br !for.done.loc59
// CHECK:STDOUT:
// CHECK:STDOUT: !for.body.loc59:
// CHECK:STDOUT:   %.loc59_33.11: %Optional.Get.type.b00 = specific_constant imports.%Core.import_ref.29e, @Optional(constants.%OptionalStorage.facet.d73) [concrete = constants.%Optional.Get.e30]
// CHECK:STDOUT:   %Get.ref.loc59: %Optional.Get.type.b00 = name_ref Get, %.loc59_33.11 [concrete = constants.%Optional.Get.e30]
// CHECK:STDOUT:   %Optional.Get.bound.loc59: <bound method> = bound_method %.loc59_33.6, %Get.ref.loc59
// CHECK:STDOUT:   %Optional.Get.specific_fn.loc59: <specific function> = specific_function %Get.ref.loc59, @Optional.Get(constants.%OptionalStorage.facet.d73) [concrete = constants.%Optional.Get.specific_fn.a58]
// CHECK:STDOUT:   %bound_method.loc59_33.6: <bound method> = bound_method %.loc59_33.6, %Optional.Get.specific_fn.loc59
// CHECK:STDOUT:   %.loc59_33.12: ref %DiffValueType = temporary_storage
// CHECK:STDOUT:   %.loc59_33.13: %Optional.750 = acquire_value %.loc59_33.6
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
// CHECK:STDOUT:   %Destroy.facet.loc59_33.1: %Destroy.type = facet_value constants.%DiffValueType, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.f43) [concrete = constants.%Destroy.facet.95d]
// CHECK:STDOUT:   %.loc59_33.16: %Destroy.type = converted constants.%DiffValueType, %Destroy.facet.loc59_33.1 [concrete = constants.%Destroy.facet.95d]
// CHECK:STDOUT:   %Destroy.facet.loc59_33.2: %Destroy.type = facet_value constants.%DiffValueType, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.f43) [concrete = constants.%Destroy.facet.95d]
// CHECK:STDOUT:   %.loc59_33.17: %Destroy.type = converted constants.%DiffValueType, %Destroy.facet.loc59_33.2 [concrete = constants.%Destroy.facet.95d]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc59_33.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc59_33.14)
// CHECK:STDOUT:   %Destroy.facet.loc59_33.3: %Destroy.type = facet_value constants.%Optional.750, (constants.%custom_witness.b80ef6.8, constants.%DestroyImpl.impl_witness.41e) [concrete = constants.%Destroy.facet.ee1]
// CHECK:STDOUT:   %.loc59_33.18: %Destroy.type = converted constants.%Optional.750, %Destroy.facet.loc59_33.3 [concrete = constants.%Destroy.facet.ee1]
// CHECK:STDOUT:   %Destroy.facet.loc59_33.4: %Destroy.type = facet_value constants.%Optional.750, (constants.%custom_witness.b80ef6.8, constants.%DestroyImpl.impl_witness.41e) [concrete = constants.%Destroy.facet.ee1]
// CHECK:STDOUT:   %.loc59_33.19: %Destroy.type = converted constants.%Optional.750, %Destroy.facet.loc59_33.4 [concrete = constants.%Destroy.facet.ee1]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc59_33.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc59_33.6)
// CHECK:STDOUT:   br !for.next.loc59
// CHECK:STDOUT:
// CHECK:STDOUT: !for.done.loc59:
// CHECK:STDOUT:   %Destroy.facet.loc59_33.5: %Destroy.type = facet_value constants.%Optional.750, (constants.%custom_witness.b80ef6.8, constants.%DestroyImpl.impl_witness.41e) [concrete = constants.%Destroy.facet.ee1]
// CHECK:STDOUT:   %.loc59_33.20: %Destroy.type = converted constants.%Optional.750, %Destroy.facet.loc59_33.5 [concrete = constants.%Destroy.facet.ee1]
// CHECK:STDOUT:   %Destroy.facet.loc59_33.6: %Destroy.type = facet_value constants.%Optional.750, (constants.%custom_witness.b80ef6.8, constants.%DestroyImpl.impl_witness.41e) [concrete = constants.%Destroy.facet.ee1]
// CHECK:STDOUT:   %.loc59_33.21: %Destroy.type = converted constants.%Optional.750, %Destroy.facet.loc59_33.6 [concrete = constants.%Destroy.facet.ee1]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc59_33.3: init %empty_tuple.type = call %SelfDestruct.specific_fn.3(%.loc59_33.6)
// CHECK:STDOUT:   %Destroy.facet.loc59_33.7: %Destroy.type = facet_value constants.%tuple.type.b18, (constants.%custom_witness.b80ef6.11, constants.%DestroyImpl.impl_witness.c40) [concrete = constants.%Destroy.facet.4da]
// CHECK:STDOUT:   %.loc59_33.22: %Destroy.type = converted constants.%tuple.type.b18, %Destroy.facet.loc59_33.7 [concrete = constants.%Destroy.facet.4da]
// CHECK:STDOUT:   %Destroy.facet.loc59_33.8: %Destroy.type = facet_value constants.%tuple.type.b18, (constants.%custom_witness.b80ef6.11, constants.%DestroyImpl.impl_witness.c40) [concrete = constants.%Destroy.facet.4da]
// CHECK:STDOUT:   %.loc59_33.23: %Destroy.type = converted constants.%tuple.type.b18, %Destroy.facet.loc59_33.8 [concrete = constants.%Destroy.facet.4da]
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
// CHECK:STDOUT:   %.loc63_33.5: ref %Optional.750 = temporary_storage
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.call.loc63: init %Optional.750 to %.loc63_33.5 = call %bound_method.loc63_33.4(%c.ref, %var.loc63)
// CHECK:STDOUT:   %.loc63_33.6: ref %Optional.750 = temporary %.loc63_33.5, %T.as_type.as.Iterate.impl.Next.call.loc63
// CHECK:STDOUT:   %.loc63_33.7: %Optional.HasValue.type.4da = specific_constant imports.%Core.import_ref.c82, @Optional(constants.%OptionalStorage.facet.d73) [concrete = constants.%Optional.HasValue.9f4]
// CHECK:STDOUT:   %HasValue.ref.loc63: %Optional.HasValue.type.4da = name_ref HasValue, %.loc63_33.7 [concrete = constants.%Optional.HasValue.9f4]
// CHECK:STDOUT:   %Optional.HasValue.bound.loc63: <bound method> = bound_method %.loc63_33.6, %HasValue.ref.loc63
// CHECK:STDOUT:   %Optional.HasValue.specific_fn.loc63: <specific function> = specific_function %HasValue.ref.loc63, @Optional.HasValue(constants.%OptionalStorage.facet.d73) [concrete = constants.%Optional.HasValue.specific_fn.57a]
// CHECK:STDOUT:   %bound_method.loc63_33.5: <bound method> = bound_method %.loc63_33.6, %Optional.HasValue.specific_fn.loc63
// CHECK:STDOUT:   %.loc63_33.8: %Optional.750 = acquire_value %.loc63_33.6
// CHECK:STDOUT:   %Optional.HasValue.call.loc63: init bool = call %bound_method.loc63_33.5(%.loc63_33.8)
// CHECK:STDOUT:   %.loc63_33.9: bool = value_of_initializer %Optional.HasValue.call.loc63
// CHECK:STDOUT:   %.loc63_33.10: bool = converted %Optional.HasValue.call.loc63, %.loc63_33.9
// CHECK:STDOUT:   if %.loc63_33.10 br !for.body.loc63 else br !for.done.loc63
// CHECK:STDOUT:
// CHECK:STDOUT: !for.body.loc63:
// CHECK:STDOUT:   %.loc63_33.11: %Optional.Get.type.b00 = specific_constant imports.%Core.import_ref.29e, @Optional(constants.%OptionalStorage.facet.d73) [concrete = constants.%Optional.Get.e30]
// CHECK:STDOUT:   %Get.ref.loc63: %Optional.Get.type.b00 = name_ref Get, %.loc63_33.11 [concrete = constants.%Optional.Get.e30]
// CHECK:STDOUT:   %Optional.Get.bound.loc63: <bound method> = bound_method %.loc63_33.6, %Get.ref.loc63
// CHECK:STDOUT:   %Optional.Get.specific_fn.loc63: <specific function> = specific_function %Get.ref.loc63, @Optional.Get(constants.%OptionalStorage.facet.d73) [concrete = constants.%Optional.Get.specific_fn.a58]
// CHECK:STDOUT:   %bound_method.loc63_33.6: <bound method> = bound_method %.loc63_33.6, %Optional.Get.specific_fn.loc63
// CHECK:STDOUT:   %.loc63_33.12: ref %DiffValueType = temporary_storage
// CHECK:STDOUT:   %.loc63_33.13: %Optional.750 = acquire_value %.loc63_33.6
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
// CHECK:STDOUT:   %Destroy.facet.loc63_33.1: %Destroy.type = facet_value constants.%DiffValueType, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.f43) [concrete = constants.%Destroy.facet.95d]
// CHECK:STDOUT:   %.loc63_33.16: %Destroy.type = converted constants.%DiffValueType, %Destroy.facet.loc63_33.1 [concrete = constants.%Destroy.facet.95d]
// CHECK:STDOUT:   %Destroy.facet.loc63_33.2: %Destroy.type = facet_value constants.%DiffValueType, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.f43) [concrete = constants.%Destroy.facet.95d]
// CHECK:STDOUT:   %.loc63_33.17: %Destroy.type = converted constants.%DiffValueType, %Destroy.facet.loc63_33.2 [concrete = constants.%Destroy.facet.95d]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc63_33.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.5(%.loc63_33.14)
// CHECK:STDOUT:   %Destroy.facet.loc63_33.3: %Destroy.type = facet_value constants.%Optional.750, (constants.%custom_witness.b80ef6.8, constants.%DestroyImpl.impl_witness.41e) [concrete = constants.%Destroy.facet.ee1]
// CHECK:STDOUT:   %.loc63_33.18: %Destroy.type = converted constants.%Optional.750, %Destroy.facet.loc63_33.3 [concrete = constants.%Destroy.facet.ee1]
// CHECK:STDOUT:   %Destroy.facet.loc63_33.4: %Destroy.type = facet_value constants.%Optional.750, (constants.%custom_witness.b80ef6.8, constants.%DestroyImpl.impl_witness.41e) [concrete = constants.%Destroy.facet.ee1]
// CHECK:STDOUT:   %.loc63_33.19: %Destroy.type = converted constants.%Optional.750, %Destroy.facet.loc63_33.4 [concrete = constants.%Destroy.facet.ee1]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc63_33.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.6(%.loc63_33.6)
// CHECK:STDOUT:   br !for.next.loc63
// CHECK:STDOUT:
// CHECK:STDOUT: !for.done.loc63:
// CHECK:STDOUT:   %Destroy.facet.loc63_33.5: %Destroy.type = facet_value constants.%Optional.750, (constants.%custom_witness.b80ef6.8, constants.%DestroyImpl.impl_witness.41e) [concrete = constants.%Destroy.facet.ee1]
// CHECK:STDOUT:   %.loc63_33.20: %Destroy.type = converted constants.%Optional.750, %Destroy.facet.loc63_33.5 [concrete = constants.%Destroy.facet.ee1]
// CHECK:STDOUT:   %Destroy.facet.loc63_33.6: %Destroy.type = facet_value constants.%Optional.750, (constants.%custom_witness.b80ef6.8, constants.%DestroyImpl.impl_witness.41e) [concrete = constants.%Destroy.facet.ee1]
// CHECK:STDOUT:   %.loc63_33.21: %Destroy.type = converted constants.%Optional.750, %Destroy.facet.loc63_33.6 [concrete = constants.%Destroy.facet.ee1]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc63_33.3: init %empty_tuple.type = call %SelfDestruct.specific_fn.7(%.loc63_33.6)
// CHECK:STDOUT:   %Destroy.facet.loc63_33.7: %Destroy.type = facet_value constants.%tuple.type.df5, (constants.%custom_witness.b80ef6.15, constants.%DestroyImpl.impl_witness.8c3) [concrete = constants.%Destroy.facet.017]
// CHECK:STDOUT:   %.loc63_33.22: %Destroy.type = converted constants.%tuple.type.df5, %Destroy.facet.loc63_33.7 [concrete = constants.%Destroy.facet.017]
// CHECK:STDOUT:   %Destroy.facet.loc63_33.8: %Destroy.type = facet_value constants.%tuple.type.df5, (constants.%custom_witness.b80ef6.15, constants.%DestroyImpl.impl_witness.8c3) [concrete = constants.%Destroy.facet.017]
// CHECK:STDOUT:   %.loc63_33.23: %Destroy.type = converted constants.%tuple.type.df5, %Destroy.facet.loc63_33.8 [concrete = constants.%Destroy.facet.017]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc63_33.4: init %empty_tuple.type = call %SelfDestruct.specific_fn.8(%var.loc63)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestIterate(constants.%R) {
// CHECK:STDOUT:   %R.patt.loc40_17.2 => constants.%R.patt
// CHECK:STDOUT:   %R.loc40_17.1 => constants.%R
// CHECK:STDOUT:   %R.as_type.loc40_79.1 => constants.%R.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.db2
// CHECK:STDOUT:   %r.patt.loc40_77.2 => constants.%r.patt.4a5
// CHECK:STDOUT:   %r.param_patt.loc40_72.2 => constants.%r.param_patt.24f
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestIterate(constants.%facet_value.b26) {
// CHECK:STDOUT:   %R.patt.loc40_17.2 => constants.%R.patt
// CHECK:STDOUT:   %R.loc40_17.1 => constants.%facet_value.b26
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
// CHECK:STDOUT:   %impl.elem1 => constants.%Destroy.facet.4da
// CHECK:STDOUT:   %as_type => constants.%tuple.type.b18
// CHECK:STDOUT:   %require_complete.1 => constants.%complete_type.bfe
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type => constants.%Iterate.WithSelf.Next.type.aad
// CHECK:STDOUT:   %.loc44_33.27 => constants.%.5db8
// CHECK:STDOUT:   %impl.elem3.loc44_33.2 => constants.%T.as_type.as.Iterate.impl.Next.c89
// CHECK:STDOUT:   %specific_impl_fn.loc44_33.4 => constants.%T.as_type.as.Iterate.impl.Next.specific_fn.134
// CHECK:STDOUT:   %SelfDestruct.specific_fn.6 => constants.%SelfDestruct.specific_fn.9e00
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestIterate(constants.%facet_value.635) {
// CHECK:STDOUT:   %R.patt.loc40_17.2 => constants.%R.patt
// CHECK:STDOUT:   %R.loc40_17.1 => constants.%facet_value.635
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
// CHECK:STDOUT:   %impl.elem1 => constants.%Destroy.facet.017
// CHECK:STDOUT:   %as_type => constants.%tuple.type.df5
// CHECK:STDOUT:   %require_complete.1 => constants.%complete_type.ad6
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type => constants.%Iterate.WithSelf.Next.type.f88
// CHECK:STDOUT:   %.loc44_33.27 => constants.%.5be
// CHECK:STDOUT:   %impl.elem3.loc44_33.2 => constants.%T.as_type.as.Iterate.impl.Next.b68
// CHECK:STDOUT:   %specific_impl_fn.loc44_33.4 => constants.%T.as_type.as.Iterate.impl.Next.specific_fn.4d3
// CHECK:STDOUT:   %SelfDestruct.specific_fn.6 => constants.%SelfDestruct.specific_fn.825
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- method_overloads.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %Iterate.type: type = facet_type <@Iterate> [concrete]
// CHECK:STDOUT:   %OptionalStorage.type: type = facet_type <@OptionalStorage> [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %facet_type.bd0: type = facet_type <@Copy & @Destroy> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %T.220e40.1: %OptionalStorage.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Optional.Get.type.786: type = fn_type @Optional.Get, @Optional(%T.220e40.1) [symbolic]
// CHECK:STDOUT:   %Optional.Get.2d0: %Optional.Get.type.786 = struct_value () [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.type.3ea: type = fn_type @Optional.HasValue, @Optional(%T.220e40.1) [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.f2f: %Optional.HasValue.type.3ea = struct_value () [symbolic]
// CHECK:STDOUT:   %T.f84392.1: %facet_type.bd0 = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.476: type = class_type @DefaultOptionalStorage, @DefaultOptionalStorage(%T.f84392.1) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Get.type.f27: type = fn_type @T.as_type.as.OptionalStorage.impl.Get, @T.as_type.as.OptionalStorage.impl(%T.f84392.1) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Get.e40: %T.as_type.as.OptionalStorage.impl.Get.type.f27 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Has.type.18e: type = fn_type @T.as_type.as.OptionalStorage.impl.Has, @T.as_type.as.OptionalStorage.impl(%T.f84392.1) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Has.463: %T.as_type.as.OptionalStorage.impl.Has.type.18e = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Some.type.292: type = fn_type @T.as_type.as.OptionalStorage.impl.Some, @T.as_type.as.OptionalStorage.impl(%T.f84392.1) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Some.ae9: %T.as_type.as.OptionalStorage.impl.Some.type.292 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.None.type.b57: type = fn_type @T.as_type.as.OptionalStorage.impl.None, @T.as_type.as.OptionalStorage.impl(%T.f84392.1) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.None.791: %T.as_type.as.OptionalStorage.impl.None.type.b57 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SelfDestruct.type: type = fn_type @SelfDestruct [concrete]
// CHECK:STDOUT:   %SelfDestruct: %SelfDestruct.type = struct_value () [concrete]
// CHECK:STDOUT:   %.Self.500: %Iterate.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %CppRange.type: type = facet_type <@CppRange> [concrete]
// CHECK:STDOUT:   %.Self.e0b: %CppRange.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %CppRangeForIterate.lookup_impl_witness.50b: <witness> = lookup_impl_witness %.Self.e0b, @CppRangeForIterate [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.32d3: type = impl_witness_access %CppRangeForIterate.lookup_impl_witness.50b, element0 [symbolic_self]
// CHECK:STDOUT:   %CppUnsafeDeref.lookup_impl_witness.851: <witness> = lookup_impl_witness %impl.elem0.32d3, @CppUnsafeDeref [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.10d: type = impl_witness_access %CppUnsafeDeref.lookup_impl_witness.851, element0 [symbolic_self]
// CHECK:STDOUT:   %CppRange_where.type.d682d6.1: type = facet_type <@CppRange where %impl.elem0.10d impls @Copy> [concrete]
// CHECK:STDOUT:   %T.f45530.3: %CppRange_where.type.d682d6.1 = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.type.d4a: type = fn_type @T.as_type.as.Iterate.impl.Next, @T.as_type.as.Iterate.impl(%T.f45530.3) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.5a8: %T.as_type.as.Iterate.impl.Next.type.d4a = struct_value () [symbolic]
// CHECK:STDOUT:   %CppRangeForIterate.lookup_impl_witness.b581a8.2: <witness> = lookup_impl_witness %T.f45530.3, @CppRangeForIterate [symbolic]
// CHECK:STDOUT:   %impl.elem0.5f971c.2: type = impl_witness_access %CppRangeForIterate.lookup_impl_witness.b581a8.2, element0 [symbolic]
// CHECK:STDOUT:   %CppUnsafeDeref.lookup_impl_witness.87e543.2: <witness> = lookup_impl_witness %impl.elem0.5f971c.2, @CppUnsafeDeref [symbolic]
// CHECK:STDOUT:   %impl.elem0.493765.2: type = impl_witness_access %CppUnsafeDeref.lookup_impl_witness.87e543.2, element0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.type.8fb: type = fn_type @T.as_type.as.Iterate.impl.NewCursor, @T.as_type.as.Iterate.impl(%T.f45530.3) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.8a1: %T.as_type.as.Iterate.impl.NewCursor.type.8fb = struct_value () [symbolic]
// CHECK:STDOUT:   %impl.elem1.c2dc15.2: type = impl_witness_access %CppRangeForIterate.lookup_impl_witness.b581a8.2, element1 [symbolic]
// CHECK:STDOUT:   %tuple.type.1ba: type = tuple_type (%impl.elem0.5f971c.2, %impl.elem1.c2dc15.2) [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.lookup_impl_witness.dc9: <witness> = lookup_impl_witness %tuple.type.1ba, @SubobjectDestroy [symbolic]
// CHECK:STDOUT:   %DestroyImpl.lookup_impl_witness.504: <witness> = lookup_impl_witness %tuple.type.1ba, @DestroyImpl [symbolic]
// CHECK:STDOUT:   %Destroy.facet.9b6: %Destroy.type = facet_value %tuple.type.1ba, (%SubobjectDestroy.lookup_impl_witness.dc9, %DestroyImpl.lookup_impl_witness.504) [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.lookup_impl_witness.b15da7.2: <witness> = lookup_impl_witness %impl.elem0.493765.2, @SubobjectDestroy [symbolic]
// CHECK:STDOUT:   %DestroyImpl.lookup_impl_witness.4e7f44.2: <witness> = lookup_impl_witness %impl.elem0.493765.2, @DestroyImpl [symbolic]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.167: <witness> = lookup_impl_witness %impl.elem0.493765.2, @Copy [symbolic]
// CHECK:STDOUT:   %facet_value.189: %facet_type.bd0 = facet_value %impl.elem0.493765.2, (%SubobjectDestroy.lookup_impl_witness.b15da7.2, %DestroyImpl.lookup_impl_witness.4e7f44.2, %Copy.lookup_impl_witness.167) [symbolic]
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
// CHECK:STDOUT:   %facet_value.c4f: %facet_type.bd0 = facet_value %OverloadValueType, (%custom_witness.b80ef6.2, %DestroyImpl.impl_witness.3a4, %custom_witness.d0da43.1) [concrete]
// CHECK:STDOUT:   %Iterate.lookup_impl_witness.1d8: <witness> = lookup_impl_witness %.Self.500, @Iterate [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.4ce: %facet_type.bd0 = impl_witness_access %Iterate.lookup_impl_witness.1d8, element0 [symbolic_self]
// CHECK:STDOUT:   %Iterate_where.type.a23: type = facet_type <@Iterate where %impl.elem0.4ce = %facet_value.c4f> [concrete]
// CHECK:STDOUT:   %pattern_type.ba5: type = pattern_type %Iterate_where.type.a23 [concrete]
// CHECK:STDOUT:   %R.patt: %pattern_type.ba5 = symbolic_binding_pattern R, 0 [symbolic]
// CHECK:STDOUT:   %R: %Iterate_where.type.a23 = symbolic_binding R, 0 [symbolic]
// CHECK:STDOUT:   %R.as_type: type = facet_access_type %R [symbolic]
// CHECK:STDOUT:   %pattern_type.a43: type = pattern_type %R.as_type [symbolic]
// CHECK:STDOUT:   %r.patt.447: %pattern_type.a43 = ref_binding_pattern r [symbolic]
// CHECK:STDOUT:   %r.param_patt.a3f: %pattern_type.a43 = var_param_pattern %r.patt.447 [symbolic]
// CHECK:STDOUT:   %_.patt.a10: %pattern_type.804 = value_binding_pattern _ [concrete]
// CHECK:STDOUT:   %Iterate.lookup_impl_witness.1fa: <witness> = lookup_impl_witness %R, @Iterate [symbolic]
// CHECK:STDOUT:   %Iterate.facet.60f: %Iterate.type = facet_value %R.as_type, (%Iterate.lookup_impl_witness.1fa) [symbolic]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.a52: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.60f) [symbolic]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.042: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.60f) [symbolic]
// CHECK:STDOUT:   %.7a7: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.a52, %Iterate.facet.60f [symbolic]
// CHECK:STDOUT:   %impl.elem2.91e: %.7a7 = impl_witness_access %Iterate.lookup_impl_witness.1fa, element2 [symbolic]
// CHECK:STDOUT:   %impl.elem1.63f: %Destroy.type = impl_witness_access %Iterate.lookup_impl_witness.1fa, element1 [symbolic]
// CHECK:STDOUT:   %as_type.4d0: type = facet_access_type %impl.elem1.63f [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.7bb: <specific function> = specific_impl_function %impl.elem2.91e, @Iterate.WithSelf.NewCursor(%Iterate.facet.60f) [symbolic]
// CHECK:STDOUT:   %.e65: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.042, %Iterate.facet.60f [symbolic]
// CHECK:STDOUT:   %impl.elem3.a0b: %.e65 = impl_witness_access %Iterate.lookup_impl_witness.1fa, element3 [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.bba: type = class_type @DefaultOptionalStorage, @DefaultOptionalStorage(%facet_value.c4f) [concrete]
// CHECK:STDOUT:   %OptionalStorage.impl_witness.6a5: <witness> = impl_witness imports.%OptionalStorage.impl_witness_table.1c4, @T.as_type.as.OptionalStorage.impl(%facet_value.c4f) [concrete]
// CHECK:STDOUT:   %OptionalStorage.facet.62d: %OptionalStorage.type = facet_value %OverloadValueType, (%OptionalStorage.impl_witness.6a5) [concrete]
// CHECK:STDOUT:   %Optional.081: type = class_type @Optional, @Optional(%OptionalStorage.facet.62d) [concrete]
// CHECK:STDOUT:   %specific_impl_fn.e38: <specific function> = specific_impl_function %impl.elem3.a0b, @Iterate.WithSelf.Next(%Iterate.facet.60f) [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.type.82b: type = fn_type @Optional.HasValue, @Optional(%OptionalStorage.facet.62d) [concrete]
// CHECK:STDOUT:   %Optional.HasValue.3a5: %Optional.HasValue.type.82b = struct_value () [concrete]
// CHECK:STDOUT:   %Optional.Get.type.847: type = fn_type @Optional.Get, @Optional(%OptionalStorage.facet.62d) [concrete]
// CHECK:STDOUT:   %Optional.Get.b7a: %Optional.Get.type.847 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.facet.ddd: %Destroy.type = facet_value %OverloadValueType, (%custom_witness.b80ef6.2, %DestroyImpl.impl_witness.3a4) [concrete]
// CHECK:STDOUT:   %MaybeUnformed.ede: type = class_type @MaybeUnformed, @MaybeUnformed(%Destroy.facet.ddd) [concrete]
// CHECK:STDOUT:   %.8a2: type = maybe_unformed_type %OverloadValueType [concrete]
// CHECK:STDOUT:   %struct_type.value.has_value.fe4: type = struct_type {.value: %MaybeUnformed.ede, .has_value: bool} [concrete]
// CHECK:STDOUT:   %Optional.HasValue.specific_fn.d43: <specific function> = specific_function %Optional.HasValue.3a5, @Optional.HasValue(%OptionalStorage.facet.62d) [concrete]
// CHECK:STDOUT:   %Optional.Get.specific_fn.3ed: <specific function> = specific_function %Optional.Get.b7a, @Optional.Get(%OptionalStorage.facet.62d) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.8: type = fn_type @SubobjectDestroy.WithSelf.Op.loc27_37.6 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.8: %SubobjectDestroy.WithSelf.Op.type.6870fd.8 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.8: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.8), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.94b: %SubobjectDestroy.type = facet_value %Optional.081, (%custom_witness.b80ef6.8) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.388: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.94b) [concrete]
// CHECK:STDOUT:   %Destroy.facet.a25: %Destroy.type = facet_value %Optional.081, (%custom_witness.b80ef6.8, %DestroyImpl.impl_witness.388) [concrete]
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
// CHECK:STDOUT:   %Destroy.facet.76b: %Destroy.type = facet_value %tuple.type.9ef, (%custom_witness.b80ef6.10, %DestroyImpl.impl_witness.c0f) [concrete]
// CHECK:STDOUT:   %Iterate.impl_witness.8a1: <witness> = impl_witness imports.%Iterate.impl_witness_table.121, @T.as_type.as.Iterate.impl(%facet_value.053) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.type.398: type = fn_type @T.as_type.as.Iterate.impl.NewCursor, @T.as_type.as.Iterate.impl(%facet_value.053) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.bdb: %T.as_type.as.Iterate.impl.NewCursor.type.398 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.type.565: type = fn_type @T.as_type.as.Iterate.impl.Next, @T.as_type.as.Iterate.impl(%facet_value.053) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.cb2: %T.as_type.as.Iterate.impl.Next.type.565 = struct_value () [concrete]
// CHECK:STDOUT:   %facet_value.975: %Iterate_where.type.a23 = facet_value %Range, (%Iterate.impl_witness.8a1) [concrete]
// CHECK:STDOUT:   %r.patt.b9f: %pattern_type.23f = ref_binding_pattern r [concrete]
// CHECK:STDOUT:   %r.param_patt.466: %pattern_type.23f = var_param_pattern %r.patt.b9f [concrete]
// CHECK:STDOUT:   %Iterate.facet.cdb: %Iterate.type = facet_value %Range, (%Iterate.impl_witness.8a1) [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.012: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.cdb) [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.b7e: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.cdb) [concrete]
// CHECK:STDOUT:   %.11f: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.012, %Iterate.facet.cdb [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.specific_fn: <specific function> = specific_function %T.as_type.as.Iterate.impl.NewCursor.bdb, @T.as_type.as.Iterate.impl.NewCursor(%facet_value.053) [concrete]
// CHECK:STDOUT:   %.e6b: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.b7e, %Iterate.facet.cdb [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.specific_fn: <specific function> = specific_function %T.as_type.as.Iterate.impl.Next.cb2, @T.as_type.as.Iterate.impl.Next(%facet_value.053) [concrete]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.ddf: <specific function> = specific_function %SelfDestruct, @SelfDestruct(%Destroy.facet.76b) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .OverloadValueType = %OverloadValueType.decl
// CHECK:STDOUT:     .Range = %Range.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.481: type = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%DefaultOptionalStorage (constants.%DefaultOptionalStorage.476)]
// CHECK:STDOUT:   %Core.import_ref.66d: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.None.type (%T.as_type.as.OptionalStorage.impl.None.type.b57) = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.None (constants.%T.as_type.as.OptionalStorage.impl.None.791)]
// CHECK:STDOUT:   %Core.import_ref.32c: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Some.type (%T.as_type.as.OptionalStorage.impl.Some.type.292) = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Some (constants.%T.as_type.as.OptionalStorage.impl.Some.ae9)]
// CHECK:STDOUT:   %Core.import_ref.801: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Has.type (%T.as_type.as.OptionalStorage.impl.Has.type.18e) = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Has (constants.%T.as_type.as.OptionalStorage.impl.Has.463)]
// CHECK:STDOUT:   %Core.import_ref.eec: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Get.type (%T.as_type.as.OptionalStorage.impl.Get.type.f27) = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Get (constants.%T.as_type.as.OptionalStorage.impl.Get.e40)]
// CHECK:STDOUT:   %Core.import_ref.a6a = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %OptionalStorage.impl_witness_table.1c4 = impl_witness_table (%Core.import_ref.481, %Core.import_ref.66d, %Core.import_ref.32c, %Core.import_ref.801, %Core.import_ref.eec, %Core.import_ref.a6a), @T.as_type.as.OptionalStorage.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.7b5: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/types/optional, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.7b5), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.c82: @Optional.%Optional.HasValue.type (%Optional.HasValue.type.3ea) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.HasValue (constants.%Optional.HasValue.f2f)]
// CHECK:STDOUT:   %Core.import_ref.29e: @Optional.%Optional.Get.type (%Optional.Get.type.786) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.Get (constants.%Optional.Get.2d0)]
// CHECK:STDOUT:   %Core.import_ref.724: %facet_type.bd0 = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.Iterate.impl.%facet_value (constants.%facet_value.189)]
// CHECK:STDOUT:   %Core.import_ref.dd9: %Destroy.type = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.Iterate.impl.%Destroy.facet (constants.%Destroy.facet.9b6)]
// CHECK:STDOUT:   %Core.import_ref.946: @T.as_type.as.Iterate.impl.%T.as_type.as.Iterate.impl.NewCursor.type (%T.as_type.as.Iterate.impl.NewCursor.type.8fb) = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.Iterate.impl.%T.as_type.as.Iterate.impl.NewCursor (constants.%T.as_type.as.Iterate.impl.NewCursor.8a1)]
// CHECK:STDOUT:   %Core.import_ref.cfb: @T.as_type.as.Iterate.impl.%T.as_type.as.Iterate.impl.Next.type (%T.as_type.as.Iterate.impl.Next.type.d4a) = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.Iterate.impl.%T.as_type.as.Iterate.impl.Next (constants.%T.as_type.as.Iterate.impl.Next.5a8)]
// CHECK:STDOUT:   %Iterate.impl_witness_table.121 = impl_witness_table (%Core.import_ref.724, %Core.import_ref.dd9, %Core.import_ref.946, %Core.import_ref.cfb), @T.as_type.as.Iterate.impl [concrete]
// CHECK:STDOUT:   %OverloadValueType.decl: type = class_decl @OverloadValueType [concrete = constants.%OverloadValueType] {} {}
// CHECK:STDOUT:   %Range.decl: type = class_decl @Range [concrete = constants.%Range] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @TestIterate(%R.loc25_17.2: %Iterate_where.type.a23) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Iterate.lookup_impl_witness: <witness> = lookup_impl_witness %R.loc25_17.1, @Iterate [symbolic = %Iterate.lookup_impl_witness (constants.%Iterate.lookup_impl_witness.1fa)]
// CHECK:STDOUT:   %Iterate.facet.loc27_37.5: %Iterate.type = facet_value %R.as_type.loc25_83.1, (%Iterate.lookup_impl_witness) [symbolic = %Iterate.facet.loc27_37.5 (constants.%Iterate.facet.60f)]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.loc27_37.5) [symbolic = %Iterate.WithSelf.NewCursor.type (constants.%Iterate.WithSelf.NewCursor.type.a52)]
// CHECK:STDOUT:   %.loc27_37.26: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type, %Iterate.facet.loc27_37.5 [symbolic = %.loc27_37.26 (constants.%.7a7)]
// CHECK:STDOUT:   %impl.elem2.loc27_37.2: @TestIterate.%.loc27_37.26 (%.7a7) = impl_witness_access %Iterate.lookup_impl_witness, element2 [symbolic = %impl.elem2.loc27_37.2 (constants.%impl.elem2.91e)]
// CHECK:STDOUT:   %specific_impl_fn.loc27_37.3: <specific function> = specific_impl_function %impl.elem2.loc27_37.2, @Iterate.WithSelf.NewCursor(%Iterate.facet.loc27_37.5) [symbolic = %specific_impl_fn.loc27_37.3 (constants.%specific_impl_fn.7bb)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.loc27_37.5) [symbolic = %Iterate.WithSelf.Next.type (constants.%Iterate.WithSelf.Next.type.042)]
// CHECK:STDOUT:   %.loc27_37.27: type = fn_type_with_self_type %Iterate.WithSelf.Next.type, %Iterate.facet.loc27_37.5 [symbolic = %.loc27_37.27 (constants.%.e65)]
// CHECK:STDOUT:   %impl.elem3.loc27_37.2: @TestIterate.%.loc27_37.27 (%.e65) = impl_witness_access %Iterate.lookup_impl_witness, element3 [symbolic = %impl.elem3.loc27_37.2 (constants.%impl.elem3.a0b)]
// CHECK:STDOUT:   %specific_impl_fn.loc27_37.4: <specific function> = specific_impl_function %impl.elem3.loc27_37.2, @Iterate.WithSelf.Next(%Iterate.facet.loc27_37.5) [symbolic = %specific_impl_fn.loc27_37.4 (constants.%specific_impl_fn.e38)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%r.param: ref @TestIterate.%R.as_type.loc25_83.1 (%R.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %_.patt: %pattern_type.804 = value_binding_pattern _ [concrete = constants.%_.patt.a10]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %r.ref: ref @TestIterate.%R.as_type.loc25_83.1 (%R.as_type) = name_ref r, %r
// CHECK:STDOUT:     %impl.elem2.loc27_37.1: @TestIterate.%.loc27_37.26 (%.7a7) = impl_witness_access constants.%Iterate.lookup_impl_witness.1fa, element2 [symbolic = %impl.elem2.loc27_37.2 (constants.%impl.elem2.91e)]
// CHECK:STDOUT:     %bound_method.loc27_37.1: <bound method> = bound_method %r.ref, %impl.elem2.loc27_37.1
// CHECK:STDOUT:     %Iterate.facet.loc27_37.1: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.1fa) [symbolic = %Iterate.facet.loc27_37.5 (constants.%Iterate.facet.60f)]
// CHECK:STDOUT:     %.loc27_37.1: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc27_37.1 [symbolic = %Iterate.facet.loc27_37.5 (constants.%Iterate.facet.60f)]
// CHECK:STDOUT:     %Iterate.facet.loc27_37.2: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.1fa) [symbolic = %Iterate.facet.loc27_37.5 (constants.%Iterate.facet.60f)]
// CHECK:STDOUT:     %.loc27_37.2: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc27_37.2 [symbolic = %Iterate.facet.loc27_37.5 (constants.%Iterate.facet.60f)]
// CHECK:STDOUT:     %specific_impl_fn.loc27_37.1: <specific function> = specific_impl_function %impl.elem2.loc27_37.1, @Iterate.WithSelf.NewCursor(constants.%Iterate.facet.60f) [symbolic = %specific_impl_fn.loc27_37.3 (constants.%specific_impl_fn.7bb)]
// CHECK:STDOUT:     %bound_method.loc27_37.2: <bound method> = bound_method %r.ref, %specific_impl_fn.loc27_37.1
// CHECK:STDOUT:     %var: ref @TestIterate.%as_type (%as_type.4d0) = var_storage invalid
// CHECK:STDOUT:     %.loc27_36.1: @TestIterate.%R.as_type.loc25_83.1 (%R.as_type) = acquire_value %r.ref
// CHECK:STDOUT:     %Iterate.WithSelf.NewCursor.call: init @TestIterate.%as_type (%as_type.4d0) to %var = call %bound_method.loc27_37.2(%.loc27_36.1)
// CHECK:STDOUT:     assign %var, %Iterate.WithSelf.NewCursor.call
// CHECK:STDOUT:     br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.next:
// CHECK:STDOUT:     %impl.elem3.loc27_37.1: @TestIterate.%.loc27_37.27 (%.e65) = impl_witness_access constants.%Iterate.lookup_impl_witness.1fa, element3 [symbolic = %impl.elem3.loc27_37.2 (constants.%impl.elem3.a0b)]
// CHECK:STDOUT:     %bound_method.loc27_37.3: <bound method> = bound_method %r.ref, %impl.elem3.loc27_37.1
// CHECK:STDOUT:     %Iterate.facet.loc27_37.3: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.1fa) [symbolic = %Iterate.facet.loc27_37.5 (constants.%Iterate.facet.60f)]
// CHECK:STDOUT:     %.loc27_37.3: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc27_37.3 [symbolic = %Iterate.facet.loc27_37.5 (constants.%Iterate.facet.60f)]
// CHECK:STDOUT:     %Iterate.facet.loc27_37.4: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.1fa) [symbolic = %Iterate.facet.loc27_37.5 (constants.%Iterate.facet.60f)]
// CHECK:STDOUT:     %.loc27_37.4: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc27_37.4 [symbolic = %Iterate.facet.loc27_37.5 (constants.%Iterate.facet.60f)]
// CHECK:STDOUT:     %.loc27_37.5: %Destroy.type = converted constants.%as_type.4d0, constants.%impl.elem1.63f [symbolic = %impl.elem1 (constants.%impl.elem1.63f)]
// CHECK:STDOUT:     %.loc27_37.6: %Destroy.type = converted constants.%as_type.4d0, constants.%impl.elem1.63f [symbolic = %impl.elem1 (constants.%impl.elem1.63f)]
// CHECK:STDOUT:     %specific_impl_fn.loc27_37.2: <specific function> = specific_impl_function %impl.elem3.loc27_37.1, @Iterate.WithSelf.Next(constants.%Iterate.facet.60f) [symbolic = %specific_impl_fn.loc27_37.4 (constants.%specific_impl_fn.e38)]
// CHECK:STDOUT:     %bound_method.loc27_37.4: <bound method> = bound_method %r.ref, %specific_impl_fn.loc27_37.2
// CHECK:STDOUT:     %.loc27_37.7: ref %Optional.081 = temporary_storage
// CHECK:STDOUT:     %.loc27_36.2: @TestIterate.%R.as_type.loc25_83.1 (%R.as_type) = acquire_value %r.ref
// CHECK:STDOUT:     %Iterate.WithSelf.Next.call: init %Optional.081 to %.loc27_37.7 = call %bound_method.loc27_37.4(%.loc27_36.2, %var)
// CHECK:STDOUT:     %.loc27_37.8: ref %Optional.081 = temporary %.loc27_37.7, %Iterate.WithSelf.Next.call
// CHECK:STDOUT:     %.loc27_37.9: %Optional.HasValue.type.82b = specific_constant imports.%Core.import_ref.c82, @Optional(constants.%OptionalStorage.facet.62d) [concrete = constants.%Optional.HasValue.3a5]
// CHECK:STDOUT:     %HasValue.ref: %Optional.HasValue.type.82b = name_ref HasValue, %.loc27_37.9 [concrete = constants.%Optional.HasValue.3a5]
// CHECK:STDOUT:     %Optional.HasValue.bound: <bound method> = bound_method %.loc27_37.8, %HasValue.ref
// CHECK:STDOUT:     %Optional.HasValue.specific_fn: <specific function> = specific_function %HasValue.ref, @Optional.HasValue(constants.%OptionalStorage.facet.62d) [concrete = constants.%Optional.HasValue.specific_fn.d43]
// CHECK:STDOUT:     %bound_method.loc27_37.5: <bound method> = bound_method %.loc27_37.8, %Optional.HasValue.specific_fn
// CHECK:STDOUT:     %.loc27_37.10: %Optional.081 = acquire_value %.loc27_37.8
// CHECK:STDOUT:     %Optional.HasValue.call: init bool = call %bound_method.loc27_37.5(%.loc27_37.10)
// CHECK:STDOUT:     %.loc27_37.11: bool = value_of_initializer %Optional.HasValue.call
// CHECK:STDOUT:     %.loc27_37.12: bool = converted %Optional.HasValue.call, %.loc27_37.11
// CHECK:STDOUT:     if %.loc27_37.12 br !for.body else br !for.done
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.body:
// CHECK:STDOUT:     %.loc27_37.13: %Optional.Get.type.847 = specific_constant imports.%Core.import_ref.29e, @Optional(constants.%OptionalStorage.facet.62d) [concrete = constants.%Optional.Get.b7a]
// CHECK:STDOUT:     %Get.ref: %Optional.Get.type.847 = name_ref Get, %.loc27_37.13 [concrete = constants.%Optional.Get.b7a]
// CHECK:STDOUT:     %Optional.Get.bound: <bound method> = bound_method %.loc27_37.8, %Get.ref
// CHECK:STDOUT:     %Optional.Get.specific_fn: <specific function> = specific_function %Get.ref, @Optional.Get(constants.%OptionalStorage.facet.62d) [concrete = constants.%Optional.Get.specific_fn.3ed]
// CHECK:STDOUT:     %bound_method.loc27_37.6: <bound method> = bound_method %.loc27_37.8, %Optional.Get.specific_fn
// CHECK:STDOUT:     %.loc27_37.14: ref %OverloadValueType = temporary_storage
// CHECK:STDOUT:     %.loc27_37.15: %Optional.081 = acquire_value %.loc27_37.8
// CHECK:STDOUT:     %Optional.Get.call: init %OverloadValueType to %.loc27_37.14 = call %bound_method.loc27_37.6(%.loc27_37.15)
// CHECK:STDOUT:     %.loc27_37.16: ref %OverloadValueType = temporary %.loc27_37.14, %Optional.Get.call
// CHECK:STDOUT:     %.loc27_37.17: %OverloadValueType = acquire_value %.loc27_37.16
// CHECK:STDOUT:     %.loc27_14: type = splice_block %OverloadValueType.ref.loc27 [concrete = constants.%OverloadValueType] {
// CHECK:STDOUT:       %Cpp.ref.loc27: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:       %OverloadValueType.ref.loc27: type = name_ref OverloadValueType, imports.%OverloadValueType.decl [concrete = constants.%OverloadValueType]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %_: %OverloadValueType = wrapper_binding _, %.loc27_37.17
// CHECK:STDOUT:     %Destroy.facet.loc27_37.1: %Destroy.type = facet_value constants.%OverloadValueType, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.3a4) [concrete = constants.%Destroy.facet.ddd]
// CHECK:STDOUT:     %.loc27_37.18: %Destroy.type = converted constants.%OverloadValueType, %Destroy.facet.loc27_37.1 [concrete = constants.%Destroy.facet.ddd]
// CHECK:STDOUT:     %Destroy.facet.loc27_37.2: %Destroy.type = facet_value constants.%OverloadValueType, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.3a4) [concrete = constants.%Destroy.facet.ddd]
// CHECK:STDOUT:     %.loc27_37.19: %Destroy.type = converted constants.%OverloadValueType, %Destroy.facet.loc27_37.2 [concrete = constants.%Destroy.facet.ddd]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %SelfDestruct.call.loc27_37.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc27_37.16)
// CHECK:STDOUT:     %Destroy.facet.loc27_37.3: %Destroy.type = facet_value constants.%Optional.081, (constants.%custom_witness.b80ef6.8, constants.%DestroyImpl.impl_witness.388) [concrete = constants.%Destroy.facet.a25]
// CHECK:STDOUT:     %.loc27_37.20: %Destroy.type = converted constants.%Optional.081, %Destroy.facet.loc27_37.3 [concrete = constants.%Destroy.facet.a25]
// CHECK:STDOUT:     %Destroy.facet.loc27_37.4: %Destroy.type = facet_value constants.%Optional.081, (constants.%custom_witness.b80ef6.8, constants.%DestroyImpl.impl_witness.388) [concrete = constants.%Destroy.facet.a25]
// CHECK:STDOUT:     %.loc27_37.21: %Destroy.type = converted constants.%Optional.081, %Destroy.facet.loc27_37.4 [concrete = constants.%Destroy.facet.a25]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %SelfDestruct.call.loc27_37.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc27_37.8)
// CHECK:STDOUT:     br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.done:
// CHECK:STDOUT:     %Destroy.facet.loc27_37.5: %Destroy.type = facet_value constants.%Optional.081, (constants.%custom_witness.b80ef6.8, constants.%DestroyImpl.impl_witness.388) [concrete = constants.%Destroy.facet.a25]
// CHECK:STDOUT:     %.loc27_37.22: %Destroy.type = converted constants.%Optional.081, %Destroy.facet.loc27_37.5 [concrete = constants.%Destroy.facet.a25]
// CHECK:STDOUT:     %Destroy.facet.loc27_37.6: %Destroy.type = facet_value constants.%Optional.081, (constants.%custom_witness.b80ef6.8, constants.%DestroyImpl.impl_witness.388) [concrete = constants.%Destroy.facet.a25]
// CHECK:STDOUT:     %.loc27_37.23: %Destroy.type = converted constants.%Optional.081, %Destroy.facet.loc27_37.6 [concrete = constants.%Destroy.facet.a25]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %SelfDestruct.call.loc27_37.3: init %empty_tuple.type = call %SelfDestruct.specific_fn.3(%.loc27_37.8)
// CHECK:STDOUT:     %.loc27_37.24: %Destroy.type = converted constants.%as_type.4d0, constants.%impl.elem1.63f [symbolic = %impl.elem1 (constants.%impl.elem1.63f)]
// CHECK:STDOUT:     %.loc27_37.25: %Destroy.type = converted constants.%as_type.4d0, constants.%impl.elem1.63f [symbolic = %impl.elem1 (constants.%impl.elem1.63f)]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %SelfDestruct.call.loc27_37.4: init %empty_tuple.type = call %SelfDestruct.specific_fn.4(%var)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc27_37.1(%self.param: ref %.8a2) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc27_37.2(%self.param: ref %MaybeUnformed.ede) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc27_37.3(%self.param: ref bool) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc27_37.4(%self.param: ref %struct_type.value.has_value.fe4) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc27_37.5(%self.param: ref %DefaultOptionalStorage.bba) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc27_37.6(%self.param: ref %Optional.081) = "no_op";
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
// CHECK:STDOUT:   %.loc37_37.5: ref %Optional.081 = temporary_storage
// CHECK:STDOUT:   %.loc37_36.2: %Range = acquire_value %r.ref
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.call: init %Optional.081 to %.loc37_37.5 = call %bound_method.loc37_37.4(%.loc37_36.2, %var)
// CHECK:STDOUT:   %.loc37_37.6: ref %Optional.081 = temporary %.loc37_37.5, %T.as_type.as.Iterate.impl.Next.call
// CHECK:STDOUT:   %.loc37_37.7: %Optional.HasValue.type.82b = specific_constant imports.%Core.import_ref.c82, @Optional(constants.%OptionalStorage.facet.62d) [concrete = constants.%Optional.HasValue.3a5]
// CHECK:STDOUT:   %HasValue.ref: %Optional.HasValue.type.82b = name_ref HasValue, %.loc37_37.7 [concrete = constants.%Optional.HasValue.3a5]
// CHECK:STDOUT:   %Optional.HasValue.bound: <bound method> = bound_method %.loc37_37.6, %HasValue.ref
// CHECK:STDOUT:   %Optional.HasValue.specific_fn: <specific function> = specific_function %HasValue.ref, @Optional.HasValue(constants.%OptionalStorage.facet.62d) [concrete = constants.%Optional.HasValue.specific_fn.d43]
// CHECK:STDOUT:   %bound_method.loc37_37.5: <bound method> = bound_method %.loc37_37.6, %Optional.HasValue.specific_fn
// CHECK:STDOUT:   %.loc37_37.8: %Optional.081 = acquire_value %.loc37_37.6
// CHECK:STDOUT:   %Optional.HasValue.call: init bool = call %bound_method.loc37_37.5(%.loc37_37.8)
// CHECK:STDOUT:   %.loc37_37.9: bool = value_of_initializer %Optional.HasValue.call
// CHECK:STDOUT:   %.loc37_37.10: bool = converted %Optional.HasValue.call, %.loc37_37.9
// CHECK:STDOUT:   if %.loc37_37.10 br !for.body else br !for.done
// CHECK:STDOUT:
// CHECK:STDOUT: !for.body:
// CHECK:STDOUT:   %.loc37_37.11: %Optional.Get.type.847 = specific_constant imports.%Core.import_ref.29e, @Optional(constants.%OptionalStorage.facet.62d) [concrete = constants.%Optional.Get.b7a]
// CHECK:STDOUT:   %Get.ref: %Optional.Get.type.847 = name_ref Get, %.loc37_37.11 [concrete = constants.%Optional.Get.b7a]
// CHECK:STDOUT:   %Optional.Get.bound: <bound method> = bound_method %.loc37_37.6, %Get.ref
// CHECK:STDOUT:   %Optional.Get.specific_fn: <specific function> = specific_function %Get.ref, @Optional.Get(constants.%OptionalStorage.facet.62d) [concrete = constants.%Optional.Get.specific_fn.3ed]
// CHECK:STDOUT:   %bound_method.loc37_37.6: <bound method> = bound_method %.loc37_37.6, %Optional.Get.specific_fn
// CHECK:STDOUT:   %.loc37_37.12: ref %OverloadValueType = temporary_storage
// CHECK:STDOUT:   %.loc37_37.13: %Optional.081 = acquire_value %.loc37_37.6
// CHECK:STDOUT:   %Optional.Get.call: init %OverloadValueType to %.loc37_37.12 = call %bound_method.loc37_37.6(%.loc37_37.13)
// CHECK:STDOUT:   %.loc37_37.14: ref %OverloadValueType = temporary %.loc37_37.12, %Optional.Get.call
// CHECK:STDOUT:   %.loc37_37.15: %OverloadValueType = acquire_value %.loc37_37.14
// CHECK:STDOUT:   %.loc37_14: type = splice_block %OverloadValueType.ref [concrete = constants.%OverloadValueType] {
// CHECK:STDOUT:     %Cpp.ref.loc37: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %OverloadValueType.ref: type = name_ref OverloadValueType, imports.%OverloadValueType.decl [concrete = constants.%OverloadValueType]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %_: %OverloadValueType = wrapper_binding _, %.loc37_37.15
// CHECK:STDOUT:   %Destroy.facet.loc37_37.1: %Destroy.type = facet_value constants.%OverloadValueType, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.3a4) [concrete = constants.%Destroy.facet.ddd]
// CHECK:STDOUT:   %.loc37_37.16: %Destroy.type = converted constants.%OverloadValueType, %Destroy.facet.loc37_37.1 [concrete = constants.%Destroy.facet.ddd]
// CHECK:STDOUT:   %Destroy.facet.loc37_37.2: %Destroy.type = facet_value constants.%OverloadValueType, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.3a4) [concrete = constants.%Destroy.facet.ddd]
// CHECK:STDOUT:   %.loc37_37.17: %Destroy.type = converted constants.%OverloadValueType, %Destroy.facet.loc37_37.2 [concrete = constants.%Destroy.facet.ddd]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc37_37.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc37_37.14)
// CHECK:STDOUT:   %Destroy.facet.loc37_37.3: %Destroy.type = facet_value constants.%Optional.081, (constants.%custom_witness.b80ef6.8, constants.%DestroyImpl.impl_witness.388) [concrete = constants.%Destroy.facet.a25]
// CHECK:STDOUT:   %.loc37_37.18: %Destroy.type = converted constants.%Optional.081, %Destroy.facet.loc37_37.3 [concrete = constants.%Destroy.facet.a25]
// CHECK:STDOUT:   %Destroy.facet.loc37_37.4: %Destroy.type = facet_value constants.%Optional.081, (constants.%custom_witness.b80ef6.8, constants.%DestroyImpl.impl_witness.388) [concrete = constants.%Destroy.facet.a25]
// CHECK:STDOUT:   %.loc37_37.19: %Destroy.type = converted constants.%Optional.081, %Destroy.facet.loc37_37.4 [concrete = constants.%Destroy.facet.a25]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc37_37.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc37_37.6)
// CHECK:STDOUT:   br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT: !for.done:
// CHECK:STDOUT:   %Destroy.facet.loc37_37.5: %Destroy.type = facet_value constants.%Optional.081, (constants.%custom_witness.b80ef6.8, constants.%DestroyImpl.impl_witness.388) [concrete = constants.%Destroy.facet.a25]
// CHECK:STDOUT:   %.loc37_37.20: %Destroy.type = converted constants.%Optional.081, %Destroy.facet.loc37_37.5 [concrete = constants.%Destroy.facet.a25]
// CHECK:STDOUT:   %Destroy.facet.loc37_37.6: %Destroy.type = facet_value constants.%Optional.081, (constants.%custom_witness.b80ef6.8, constants.%DestroyImpl.impl_witness.388) [concrete = constants.%Destroy.facet.a25]
// CHECK:STDOUT:   %.loc37_37.21: %Destroy.type = converted constants.%Optional.081, %Destroy.facet.loc37_37.6 [concrete = constants.%Destroy.facet.a25]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc37_37.3: init %empty_tuple.type = call %SelfDestruct.specific_fn.3(%.loc37_37.6)
// CHECK:STDOUT:   %Destroy.facet.loc37_37.7: %Destroy.type = facet_value constants.%tuple.type.9ef, (constants.%custom_witness.b80ef6.10, constants.%DestroyImpl.impl_witness.c0f) [concrete = constants.%Destroy.facet.76b]
// CHECK:STDOUT:   %.loc37_37.22: %Destroy.type = converted constants.%tuple.type.9ef, %Destroy.facet.loc37_37.7 [concrete = constants.%Destroy.facet.76b]
// CHECK:STDOUT:   %Destroy.facet.loc37_37.8: %Destroy.type = facet_value constants.%tuple.type.9ef, (constants.%custom_witness.b80ef6.10, constants.%DestroyImpl.impl_witness.c0f) [concrete = constants.%Destroy.facet.76b]
// CHECK:STDOUT:   %.loc37_37.23: %Destroy.type = converted constants.%tuple.type.9ef, %Destroy.facet.loc37_37.8 [concrete = constants.%Destroy.facet.76b]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc37_37.4: init %empty_tuple.type = call %SelfDestruct.specific_fn.4(%var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestIterate(constants.%R) {
// CHECK:STDOUT:   %R.patt.loc25_17.2 => constants.%R.patt
// CHECK:STDOUT:   %R.loc25_17.1 => constants.%R
// CHECK:STDOUT:   %R.as_type.loc25_83.1 => constants.%R.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.a43
// CHECK:STDOUT:   %r.patt.loc25_81.2 => constants.%r.patt.447
// CHECK:STDOUT:   %r.param_patt.loc25_76.2 => constants.%r.param_patt.a3f
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestIterate(constants.%facet_value.975) {
// CHECK:STDOUT:   %R.patt.loc25_17.2 => constants.%R.patt
// CHECK:STDOUT:   %R.loc25_17.1 => constants.%facet_value.975
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
// CHECK:STDOUT:   %impl.elem1 => constants.%Destroy.facet.76b
// CHECK:STDOUT:   %as_type => constants.%tuple.type.9ef
// CHECK:STDOUT:   %require_complete.1 => constants.%complete_type.e35
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type => constants.%Iterate.WithSelf.Next.type.b7e
// CHECK:STDOUT:   %.loc27_37.27 => constants.%.e6b
// CHECK:STDOUT:   %impl.elem3.loc27_37.2 => constants.%T.as_type.as.Iterate.impl.Next.cb2
// CHECK:STDOUT:   %specific_impl_fn.loc27_37.4 => constants.%T.as_type.as.Iterate.impl.Next.specific_fn
// CHECK:STDOUT:   %SelfDestruct.specific_fn.5 => constants.%SelfDestruct.specific_fn.ddf
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- adl_return_same_types.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %Iterate.type: type = facet_type <@Iterate> [concrete]
// CHECK:STDOUT:   %OptionalStorage.type: type = facet_type <@OptionalStorage> [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %facet_type.bd0: type = facet_type <@Copy & @Destroy> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %T.220e40.1: %OptionalStorage.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Optional.Get.type.786: type = fn_type @Optional.Get, @Optional(%T.220e40.1) [symbolic]
// CHECK:STDOUT:   %Optional.Get.2d0: %Optional.Get.type.786 = struct_value () [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.type.3ea: type = fn_type @Optional.HasValue, @Optional(%T.220e40.1) [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.f2f: %Optional.HasValue.type.3ea = struct_value () [symbolic]
// CHECK:STDOUT:   %T.f84392.1: %facet_type.bd0 = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.476: type = class_type @DefaultOptionalStorage, @DefaultOptionalStorage(%T.f84392.1) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Get.type.f27: type = fn_type @T.as_type.as.OptionalStorage.impl.Get, @T.as_type.as.OptionalStorage.impl(%T.f84392.1) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Get.e40: %T.as_type.as.OptionalStorage.impl.Get.type.f27 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Has.type.18e: type = fn_type @T.as_type.as.OptionalStorage.impl.Has, @T.as_type.as.OptionalStorage.impl(%T.f84392.1) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Has.463: %T.as_type.as.OptionalStorage.impl.Has.type.18e = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Some.type.292: type = fn_type @T.as_type.as.OptionalStorage.impl.Some, @T.as_type.as.OptionalStorage.impl(%T.f84392.1) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Some.ae9: %T.as_type.as.OptionalStorage.impl.Some.type.292 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.None.type.b57: type = fn_type @T.as_type.as.OptionalStorage.impl.None, @T.as_type.as.OptionalStorage.impl(%T.f84392.1) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.None.791: %T.as_type.as.OptionalStorage.impl.None.type.b57 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SelfDestruct.type: type = fn_type @SelfDestruct [concrete]
// CHECK:STDOUT:   %SelfDestruct: %SelfDestruct.type = struct_value () [concrete]
// CHECK:STDOUT:   %N.fe9: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %.Self.500: %Iterate.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %CppRange.type: type = facet_type <@CppRange> [concrete]
// CHECK:STDOUT:   %.Self.e0b: %CppRange.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %CppRangeForIterate.lookup_impl_witness.50b: <witness> = lookup_impl_witness %.Self.e0b, @CppRangeForIterate [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.32d3: type = impl_witness_access %CppRangeForIterate.lookup_impl_witness.50b, element0 [symbolic_self]
// CHECK:STDOUT:   %CppUnsafeDeref.lookup_impl_witness.851: <witness> = lookup_impl_witness %impl.elem0.32d3, @CppUnsafeDeref [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.10d: type = impl_witness_access %CppUnsafeDeref.lookup_impl_witness.851, element0 [symbolic_self]
// CHECK:STDOUT:   %CppRange_where.type.d682d6.1: type = facet_type <@CppRange where %impl.elem0.10d impls @Copy> [concrete]
// CHECK:STDOUT:   %T.f45530.3: %CppRange_where.type.d682d6.1 = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.type.d4a: type = fn_type @T.as_type.as.Iterate.impl.Next, @T.as_type.as.Iterate.impl(%T.f45530.3) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.5a8: %T.as_type.as.Iterate.impl.Next.type.d4a = struct_value () [symbolic]
// CHECK:STDOUT:   %CppRangeForIterate.lookup_impl_witness.b581a8.2: <witness> = lookup_impl_witness %T.f45530.3, @CppRangeForIterate [symbolic]
// CHECK:STDOUT:   %impl.elem0.5f971c.2: type = impl_witness_access %CppRangeForIterate.lookup_impl_witness.b581a8.2, element0 [symbolic]
// CHECK:STDOUT:   %CppUnsafeDeref.lookup_impl_witness.87e543.2: <witness> = lookup_impl_witness %impl.elem0.5f971c.2, @CppUnsafeDeref [symbolic]
// CHECK:STDOUT:   %impl.elem0.493765.2: type = impl_witness_access %CppUnsafeDeref.lookup_impl_witness.87e543.2, element0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.type.8fb: type = fn_type @T.as_type.as.Iterate.impl.NewCursor, @T.as_type.as.Iterate.impl(%T.f45530.3) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.8a1: %T.as_type.as.Iterate.impl.NewCursor.type.8fb = struct_value () [symbolic]
// CHECK:STDOUT:   %impl.elem1.c2dc15.2: type = impl_witness_access %CppRangeForIterate.lookup_impl_witness.b581a8.2, element1 [symbolic]
// CHECK:STDOUT:   %tuple.type.1ba: type = tuple_type (%impl.elem0.5f971c.2, %impl.elem1.c2dc15.2) [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.lookup_impl_witness.dc9: <witness> = lookup_impl_witness %tuple.type.1ba, @SubobjectDestroy [symbolic]
// CHECK:STDOUT:   %DestroyImpl.lookup_impl_witness.504: <witness> = lookup_impl_witness %tuple.type.1ba, @DestroyImpl [symbolic]
// CHECK:STDOUT:   %Destroy.facet.9b6: %Destroy.type = facet_value %tuple.type.1ba, (%SubobjectDestroy.lookup_impl_witness.dc9, %DestroyImpl.lookup_impl_witness.504) [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.lookup_impl_witness.b15da7.2: <witness> = lookup_impl_witness %impl.elem0.493765.2, @SubobjectDestroy [symbolic]
// CHECK:STDOUT:   %DestroyImpl.lookup_impl_witness.4e7f44.2: <witness> = lookup_impl_witness %impl.elem0.493765.2, @DestroyImpl [symbolic]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.167: <witness> = lookup_impl_witness %impl.elem0.493765.2, @Copy [symbolic]
// CHECK:STDOUT:   %facet_value.189: %facet_type.bd0 = facet_value %impl.elem0.493765.2, (%SubobjectDestroy.lookup_impl_witness.b15da7.2, %DestroyImpl.lookup_impl_witness.4e7f44.2, %Copy.lookup_impl_witness.167) [symbolic]
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
// CHECK:STDOUT:   %facet_value.0d3: %facet_type.bd0 = facet_value %f64.761, (%custom_witness.b80ef6.3, %DestroyImpl.impl_witness.a23, %Copy.impl_witness.eac) [concrete]
// CHECK:STDOUT:   %Iterate.lookup_impl_witness.1d8: <witness> = lookup_impl_witness %.Self.500, @Iterate [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.4ce: %facet_type.bd0 = impl_witness_access %Iterate.lookup_impl_witness.1d8, element0 [symbolic_self]
// CHECK:STDOUT:   %Iterate_where.type.a66: type = facet_type <@Iterate where %impl.elem0.4ce = %facet_value.0d3> [concrete]
// CHECK:STDOUT:   %pattern_type.ec9: type = pattern_type %Iterate_where.type.a66 [concrete]
// CHECK:STDOUT:   %R.patt: %pattern_type.ec9 = symbolic_binding_pattern R, 0 [symbolic]
// CHECK:STDOUT:   %R: %Iterate_where.type.a66 = symbolic_binding R, 0 [symbolic]
// CHECK:STDOUT:   %R.as_type: type = facet_access_type %R [symbolic]
// CHECK:STDOUT:   %pattern_type.c3b: type = pattern_type %R.as_type [symbolic]
// CHECK:STDOUT:   %r.param_patt.304: %pattern_type.c3b = value_param_pattern [symbolic]
// CHECK:STDOUT:   %r.patt.240: %pattern_type.c3b = wrapper_binding_pattern r, %r.param_patt.304 [symbolic]
// CHECK:STDOUT:   %i.patt: %pattern_type.667 = value_binding_pattern i [concrete]
// CHECK:STDOUT:   %Iterate.lookup_impl_witness.6fa: <witness> = lookup_impl_witness %R, @Iterate [symbolic]
// CHECK:STDOUT:   %Iterate.facet.e8c: %Iterate.type = facet_value %R.as_type, (%Iterate.lookup_impl_witness.6fa) [symbolic]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.a57: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.e8c) [symbolic]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.01e: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.e8c) [symbolic]
// CHECK:STDOUT:   %.b10: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.a57, %Iterate.facet.e8c [symbolic]
// CHECK:STDOUT:   %impl.elem2.81a: %.b10 = impl_witness_access %Iterate.lookup_impl_witness.6fa, element2 [symbolic]
// CHECK:STDOUT:   %impl.elem1.7dd: %Destroy.type = impl_witness_access %Iterate.lookup_impl_witness.6fa, element1 [symbolic]
// CHECK:STDOUT:   %as_type.deb: type = facet_access_type %impl.elem1.7dd [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.6ec: <specific function> = specific_impl_function %impl.elem2.81a, @Iterate.WithSelf.NewCursor(%Iterate.facet.e8c) [symbolic]
// CHECK:STDOUT:   %.063: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.01e, %Iterate.facet.e8c [symbolic]
// CHECK:STDOUT:   %impl.elem3.ae5: %.063 = impl_witness_access %Iterate.lookup_impl_witness.6fa, element3 [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.f55: type = class_type @DefaultOptionalStorage, @DefaultOptionalStorage(%facet_value.0d3) [concrete]
// CHECK:STDOUT:   %OptionalStorage.impl_witness.050: <witness> = impl_witness imports.%OptionalStorage.impl_witness_table.1c4, @T.as_type.as.OptionalStorage.impl(%facet_value.0d3) [concrete]
// CHECK:STDOUT:   %OptionalStorage.facet.19f: %OptionalStorage.type = facet_value %f64.761, (%OptionalStorage.impl_witness.050) [concrete]
// CHECK:STDOUT:   %Optional.ebe: type = class_type @Optional, @Optional(%OptionalStorage.facet.19f) [concrete]
// CHECK:STDOUT:   %specific_impl_fn.2e1: <specific function> = specific_impl_function %impl.elem3.ae5, @Iterate.WithSelf.Next(%Iterate.facet.e8c) [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.type.1cf: type = fn_type @Optional.HasValue, @Optional(%OptionalStorage.facet.19f) [concrete]
// CHECK:STDOUT:   %Optional.HasValue.768: %Optional.HasValue.type.1cf = struct_value () [concrete]
// CHECK:STDOUT:   %Optional.Get.type.b2d: type = fn_type @Optional.Get, @Optional(%OptionalStorage.facet.19f) [concrete]
// CHECK:STDOUT:   %Optional.Get.f12: %Optional.Get.type.b2d = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.facet.941: %Destroy.type = facet_value %f64.761, (%custom_witness.b80ef6.3, %DestroyImpl.impl_witness.a23) [concrete]
// CHECK:STDOUT:   %MaybeUnformed.f92: type = class_type @MaybeUnformed, @MaybeUnformed(%Destroy.facet.941) [concrete]
// CHECK:STDOUT:   %.ecc: type = maybe_unformed_type %f64.761 [concrete]
// CHECK:STDOUT:   %struct_type.value.has_value.e0a: type = struct_type {.value: %MaybeUnformed.f92, .has_value: bool} [concrete]
// CHECK:STDOUT:   %Optional.HasValue.specific_fn.89a: <specific function> = specific_function %Optional.HasValue.768, @Optional.HasValue(%OptionalStorage.facet.19f) [concrete]
// CHECK:STDOUT:   %Optional.Get.specific_fn.67e: <specific function> = specific_function %Optional.Get.f12, @Optional.Get(%OptionalStorage.facet.19f) [concrete]
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
// CHECK:STDOUT:   %SubobjectDestroy.facet.cdd: %SubobjectDestroy.type = facet_value %Optional.ebe, (%custom_witness.b80ef6.9) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.6e5: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.cdd) [concrete]
// CHECK:STDOUT:   %Destroy.facet.54d: %Destroy.type = facet_value %Optional.ebe, (%custom_witness.b80ef6.9, %DestroyImpl.impl_witness.6e5) [concrete]
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
// CHECK:STDOUT:   %Destroy.facet.0b5: %Destroy.type = facet_value %tuple.type.005, (%custom_witness.b80ef6.11, %DestroyImpl.impl_witness.729) [concrete]
// CHECK:STDOUT:   %Iterate.impl_witness.5f4: <witness> = impl_witness imports.%Iterate.impl_witness_table.121, @T.as_type.as.Iterate.impl(%facet_value.ecd) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.type.79b: type = fn_type @T.as_type.as.Iterate.impl.NewCursor, @T.as_type.as.Iterate.impl(%facet_value.ecd) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.6f3: %T.as_type.as.Iterate.impl.NewCursor.type.79b = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.type.1d8: type = fn_type @T.as_type.as.Iterate.impl.Next, @T.as_type.as.Iterate.impl(%facet_value.ecd) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.984: %T.as_type.as.Iterate.impl.Next.type.1d8 = struct_value () [concrete]
// CHECK:STDOUT:   %facet_value.4f9: %Iterate_where.type.a66 = facet_value %ConstRange, (%Iterate.impl_witness.5f4) [concrete]
// CHECK:STDOUT:   %r.param_patt.a73: %pattern_type.025 = value_param_pattern [concrete]
// CHECK:STDOUT:   %r.patt.0b5: %pattern_type.025 = wrapper_binding_pattern r, %r.param_patt.a73 [concrete]
// CHECK:STDOUT:   %Iterate.facet.10d: %Iterate.type = facet_value %ConstRange, (%Iterate.impl_witness.5f4) [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.008: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.10d) [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.d0b: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.10d) [concrete]
// CHECK:STDOUT:   %.46c: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.008, %Iterate.facet.10d [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.specific_fn: <specific function> = specific_function %T.as_type.as.Iterate.impl.NewCursor.6f3, @T.as_type.as.Iterate.impl.NewCursor(%facet_value.ecd) [concrete]
// CHECK:STDOUT:   %.a5a: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.d0b, %Iterate.facet.10d [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.specific_fn: <specific function> = specific_function %T.as_type.as.Iterate.impl.Next.984, @T.as_type.as.Iterate.impl.Next(%facet_value.ecd) [concrete]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.c59: <specific function> = specific_function %SelfDestruct, @SelfDestruct(%Destroy.facet.0b5) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.481: type = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%DefaultOptionalStorage (constants.%DefaultOptionalStorage.476)]
// CHECK:STDOUT:   %Core.import_ref.66d: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.None.type (%T.as_type.as.OptionalStorage.impl.None.type.b57) = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.None (constants.%T.as_type.as.OptionalStorage.impl.None.791)]
// CHECK:STDOUT:   %Core.import_ref.32c: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Some.type (%T.as_type.as.OptionalStorage.impl.Some.type.292) = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Some (constants.%T.as_type.as.OptionalStorage.impl.Some.ae9)]
// CHECK:STDOUT:   %Core.import_ref.801: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Has.type (%T.as_type.as.OptionalStorage.impl.Has.type.18e) = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Has (constants.%T.as_type.as.OptionalStorage.impl.Has.463)]
// CHECK:STDOUT:   %Core.import_ref.eec: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Get.type (%T.as_type.as.OptionalStorage.impl.Get.type.f27) = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Get (constants.%T.as_type.as.OptionalStorage.impl.Get.e40)]
// CHECK:STDOUT:   %Core.import_ref.a6a = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %OptionalStorage.impl_witness_table.1c4 = impl_witness_table (%Core.import_ref.481, %Core.import_ref.66d, %Core.import_ref.32c, %Core.import_ref.801, %Core.import_ref.eec, %Core.import_ref.a6a), @T.as_type.as.OptionalStorage.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.7b5: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/types/optional, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.7b5), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.c82: @Optional.%Optional.HasValue.type (%Optional.HasValue.type.3ea) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.HasValue (constants.%Optional.HasValue.f2f)]
// CHECK:STDOUT:   %Core.import_ref.29e: @Optional.%Optional.Get.type (%Optional.Get.type.786) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.Get (constants.%Optional.Get.2d0)]
// CHECK:STDOUT:   %Core.import_ref.724: %facet_type.bd0 = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.Iterate.impl.%facet_value (constants.%facet_value.189)]
// CHECK:STDOUT:   %Core.import_ref.dd9: %Destroy.type = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.Iterate.impl.%Destroy.facet (constants.%Destroy.facet.9b6)]
// CHECK:STDOUT:   %Core.import_ref.946: @T.as_type.as.Iterate.impl.%T.as_type.as.Iterate.impl.NewCursor.type (%T.as_type.as.Iterate.impl.NewCursor.type.8fb) = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.Iterate.impl.%T.as_type.as.Iterate.impl.NewCursor (constants.%T.as_type.as.Iterate.impl.NewCursor.8a1)]
// CHECK:STDOUT:   %Core.import_ref.cfb: @T.as_type.as.Iterate.impl.%T.as_type.as.Iterate.impl.Next.type (%T.as_type.as.Iterate.impl.Next.type.d4a) = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.Iterate.impl.%T.as_type.as.Iterate.impl.Next (constants.%T.as_type.as.Iterate.impl.Next.5a8)]
// CHECK:STDOUT:   %Iterate.impl_witness_table.121 = impl_witness_table (%Core.import_ref.724, %Core.import_ref.dd9, %Core.import_ref.946, %Core.import_ref.cfb), @T.as_type.as.Iterate.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.19b: @Float.as.Copy.impl.%Float.as.Copy.impl.Op.type (%Float.as.Copy.impl.Op.type.e8d) = import_ref Core//prelude/types/float, loc{{\d+_\d+}}, loaded [symbolic = @Float.as.Copy.impl.%Float.as.Copy.impl.Op (constants.%Float.as.Copy.impl.Op.fe9)]
// CHECK:STDOUT:   %Copy.impl_witness_table.14d = impl_witness_table (%Core.import_ref.19b), @Float.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.e0d0: @Float.as.AddAssignWith.impl.%Float.as.AddAssignWith.impl.Op.type (%Float.as.AddAssignWith.impl.Op.type.763) = import_ref Core//prelude/types/float, loc{{\d+_\d+}}, loaded [symbolic = @Float.as.AddAssignWith.impl.%Float.as.AddAssignWith.impl.Op (constants.%Float.as.AddAssignWith.impl.Op.8aa)]
// CHECK:STDOUT:   %AddAssignWith.impl_witness_table = impl_witness_table (%Core.import_ref.e0d0), @Float.as.AddAssignWith.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @TestIterate(%R.loc20_17.2: %Iterate_where.type.a66) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Iterate.lookup_impl_witness: <witness> = lookup_impl_witness %R.loc20_17.1, @Iterate [symbolic = %Iterate.lookup_impl_witness (constants.%Iterate.lookup_impl_witness.6fa)]
// CHECK:STDOUT:   %Iterate.facet.loc24_19.5: %Iterate.type = facet_value %R.as_type.loc20_61.1, (%Iterate.lookup_impl_witness) [symbolic = %Iterate.facet.loc24_19.5 (constants.%Iterate.facet.e8c)]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.loc24_19.5) [symbolic = %Iterate.WithSelf.NewCursor.type (constants.%Iterate.WithSelf.NewCursor.type.a57)]
// CHECK:STDOUT:   %.loc24_19.23: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type, %Iterate.facet.loc24_19.5 [symbolic = %.loc24_19.23 (constants.%.b10)]
// CHECK:STDOUT:   %impl.elem2.loc24_19.2: @TestIterate.%.loc24_19.23 (%.b10) = impl_witness_access %Iterate.lookup_impl_witness, element2 [symbolic = %impl.elem2.loc24_19.2 (constants.%impl.elem2.81a)]
// CHECK:STDOUT:   %specific_impl_fn.loc24_19.3: <specific function> = specific_impl_function %impl.elem2.loc24_19.2, @Iterate.WithSelf.NewCursor(%Iterate.facet.loc24_19.5) [symbolic = %specific_impl_fn.loc24_19.3 (constants.%specific_impl_fn.6ec)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.loc24_19.5) [symbolic = %Iterate.WithSelf.Next.type (constants.%Iterate.WithSelf.Next.type.01e)]
// CHECK:STDOUT:   %.loc24_19.24: type = fn_type_with_self_type %Iterate.WithSelf.Next.type, %Iterate.facet.loc24_19.5 [symbolic = %.loc24_19.24 (constants.%.063)]
// CHECK:STDOUT:   %impl.elem3.loc24_19.2: @TestIterate.%.loc24_19.24 (%.063) = impl_witness_access %Iterate.lookup_impl_witness, element3 [symbolic = %impl.elem3.loc24_19.2 (constants.%impl.elem3.ae5)]
// CHECK:STDOUT:   %specific_impl_fn.loc24_19.4: <specific function> = specific_impl_function %impl.elem3.loc24_19.2, @Iterate.WithSelf.Next(%Iterate.facet.loc24_19.5) [symbolic = %specific_impl_fn.loc24_19.4 (constants.%specific_impl_fn.2e1)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%r.param: @TestIterate.%R.as_type.loc20_61.1 (%R.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %i.patt: %pattern_type.667 = value_binding_pattern i [concrete = constants.%i.patt]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %r.ref: @TestIterate.%R.as_type.loc20_61.1 (%R.as_type) = name_ref r, %r
// CHECK:STDOUT:     %impl.elem2.loc24_19.1: @TestIterate.%.loc24_19.23 (%.b10) = impl_witness_access constants.%Iterate.lookup_impl_witness.6fa, element2 [symbolic = %impl.elem2.loc24_19.2 (constants.%impl.elem2.81a)]
// CHECK:STDOUT:     %bound_method.loc24_19.1: <bound method> = bound_method %r.ref, %impl.elem2.loc24_19.1
// CHECK:STDOUT:     %Iterate.facet.loc24_19.1: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.6fa) [symbolic = %Iterate.facet.loc24_19.5 (constants.%Iterate.facet.e8c)]
// CHECK:STDOUT:     %.loc24_19.1: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc24_19.1 [symbolic = %Iterate.facet.loc24_19.5 (constants.%Iterate.facet.e8c)]
// CHECK:STDOUT:     %Iterate.facet.loc24_19.2: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.6fa) [symbolic = %Iterate.facet.loc24_19.5 (constants.%Iterate.facet.e8c)]
// CHECK:STDOUT:     %.loc24_19.2: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc24_19.2 [symbolic = %Iterate.facet.loc24_19.5 (constants.%Iterate.facet.e8c)]
// CHECK:STDOUT:     %specific_impl_fn.loc24_19.1: <specific function> = specific_impl_function %impl.elem2.loc24_19.1, @Iterate.WithSelf.NewCursor(constants.%Iterate.facet.e8c) [symbolic = %specific_impl_fn.loc24_19.3 (constants.%specific_impl_fn.6ec)]
// CHECK:STDOUT:     %bound_method.loc24_19.2: <bound method> = bound_method %r.ref, %specific_impl_fn.loc24_19.1
// CHECK:STDOUT:     %var: ref @TestIterate.%as_type (%as_type.deb) = var_storage invalid
// CHECK:STDOUT:     %Iterate.WithSelf.NewCursor.call: init @TestIterate.%as_type (%as_type.deb) to %var = call %bound_method.loc24_19.2(%r.ref)
// CHECK:STDOUT:     assign %var, %Iterate.WithSelf.NewCursor.call
// CHECK:STDOUT:     br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.next:
// CHECK:STDOUT:     %impl.elem3.loc24_19.1: @TestIterate.%.loc24_19.24 (%.063) = impl_witness_access constants.%Iterate.lookup_impl_witness.6fa, element3 [symbolic = %impl.elem3.loc24_19.2 (constants.%impl.elem3.ae5)]
// CHECK:STDOUT:     %bound_method.loc24_19.3: <bound method> = bound_method %r.ref, %impl.elem3.loc24_19.1
// CHECK:STDOUT:     %Iterate.facet.loc24_19.3: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.6fa) [symbolic = %Iterate.facet.loc24_19.5 (constants.%Iterate.facet.e8c)]
// CHECK:STDOUT:     %.loc24_19.3: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc24_19.3 [symbolic = %Iterate.facet.loc24_19.5 (constants.%Iterate.facet.e8c)]
// CHECK:STDOUT:     %Iterate.facet.loc24_19.4: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.6fa) [symbolic = %Iterate.facet.loc24_19.5 (constants.%Iterate.facet.e8c)]
// CHECK:STDOUT:     %.loc24_19.4: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc24_19.4 [symbolic = %Iterate.facet.loc24_19.5 (constants.%Iterate.facet.e8c)]
// CHECK:STDOUT:     %.loc24_19.5: %Destroy.type = converted constants.%as_type.deb, constants.%impl.elem1.7dd [symbolic = %impl.elem1 (constants.%impl.elem1.7dd)]
// CHECK:STDOUT:     %.loc24_19.6: %Destroy.type = converted constants.%as_type.deb, constants.%impl.elem1.7dd [symbolic = %impl.elem1 (constants.%impl.elem1.7dd)]
// CHECK:STDOUT:     %specific_impl_fn.loc24_19.2: <specific function> = specific_impl_function %impl.elem3.loc24_19.1, @Iterate.WithSelf.Next(constants.%Iterate.facet.e8c) [symbolic = %specific_impl_fn.loc24_19.4 (constants.%specific_impl_fn.2e1)]
// CHECK:STDOUT:     %bound_method.loc24_19.4: <bound method> = bound_method %r.ref, %specific_impl_fn.loc24_19.2
// CHECK:STDOUT:     %.loc24_19.7: ref %Optional.ebe = temporary_storage
// CHECK:STDOUT:     %Iterate.WithSelf.Next.call: init %Optional.ebe to %.loc24_19.7 = call %bound_method.loc24_19.4(%r.ref, %var)
// CHECK:STDOUT:     %.loc24_19.8: ref %Optional.ebe = temporary %.loc24_19.7, %Iterate.WithSelf.Next.call
// CHECK:STDOUT:     %.loc24_19.9: %Optional.HasValue.type.1cf = specific_constant imports.%Core.import_ref.c82, @Optional(constants.%OptionalStorage.facet.19f) [concrete = constants.%Optional.HasValue.768]
// CHECK:STDOUT:     %HasValue.ref: %Optional.HasValue.type.1cf = name_ref HasValue, %.loc24_19.9 [concrete = constants.%Optional.HasValue.768]
// CHECK:STDOUT:     %Optional.HasValue.bound: <bound method> = bound_method %.loc24_19.8, %HasValue.ref
// CHECK:STDOUT:     %Optional.HasValue.specific_fn: <specific function> = specific_function %HasValue.ref, @Optional.HasValue(constants.%OptionalStorage.facet.19f) [concrete = constants.%Optional.HasValue.specific_fn.89a]
// CHECK:STDOUT:     %bound_method.loc24_19.5: <bound method> = bound_method %.loc24_19.8, %Optional.HasValue.specific_fn
// CHECK:STDOUT:     %.loc24_19.10: %Optional.ebe = acquire_value %.loc24_19.8
// CHECK:STDOUT:     %Optional.HasValue.call: init bool = call %bound_method.loc24_19.5(%.loc24_19.10)
// CHECK:STDOUT:     %.loc24_19.11: bool = value_of_initializer %Optional.HasValue.call
// CHECK:STDOUT:     %.loc24_19.12: bool = converted %Optional.HasValue.call, %.loc24_19.11
// CHECK:STDOUT:     if %.loc24_19.12 br !for.body else br !for.done
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.body:
// CHECK:STDOUT:     %.loc24_19.13: %Optional.Get.type.b2d = specific_constant imports.%Core.import_ref.29e, @Optional(constants.%OptionalStorage.facet.19f) [concrete = constants.%Optional.Get.f12]
// CHECK:STDOUT:     %Get.ref: %Optional.Get.type.b2d = name_ref Get, %.loc24_19.13 [concrete = constants.%Optional.Get.f12]
// CHECK:STDOUT:     %Optional.Get.bound: <bound method> = bound_method %.loc24_19.8, %Get.ref
// CHECK:STDOUT:     %Optional.Get.specific_fn: <specific function> = specific_function %Get.ref, @Optional.Get(constants.%OptionalStorage.facet.19f) [concrete = constants.%Optional.Get.specific_fn.67e]
// CHECK:STDOUT:     %bound_method.loc24_19.6: <bound method> = bound_method %.loc24_19.8, %Optional.Get.specific_fn
// CHECK:STDOUT:     %.loc24_19.14: %Optional.ebe = acquire_value %.loc24_19.8
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
// CHECK:STDOUT:     %Destroy.facet.loc24_19.1: %Destroy.type = facet_value constants.%Optional.ebe, (constants.%custom_witness.b80ef6.9, constants.%DestroyImpl.impl_witness.6e5) [concrete = constants.%Destroy.facet.54d]
// CHECK:STDOUT:     %.loc24_19.17: %Destroy.type = converted constants.%Optional.ebe, %Destroy.facet.loc24_19.1 [concrete = constants.%Destroy.facet.54d]
// CHECK:STDOUT:     %Destroy.facet.loc24_19.2: %Destroy.type = facet_value constants.%Optional.ebe, (constants.%custom_witness.b80ef6.9, constants.%DestroyImpl.impl_witness.6e5) [concrete = constants.%Destroy.facet.54d]
// CHECK:STDOUT:     %.loc24_19.18: %Destroy.type = converted constants.%Optional.ebe, %Destroy.facet.loc24_19.2 [concrete = constants.%Destroy.facet.54d]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %SelfDestruct.call.loc24_19.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc24_19.8)
// CHECK:STDOUT:     br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.done:
// CHECK:STDOUT:     %Destroy.facet.loc24_19.3: %Destroy.type = facet_value constants.%Optional.ebe, (constants.%custom_witness.b80ef6.9, constants.%DestroyImpl.impl_witness.6e5) [concrete = constants.%Destroy.facet.54d]
// CHECK:STDOUT:     %.loc24_19.19: %Destroy.type = converted constants.%Optional.ebe, %Destroy.facet.loc24_19.3 [concrete = constants.%Destroy.facet.54d]
// CHECK:STDOUT:     %Destroy.facet.loc24_19.4: %Destroy.type = facet_value constants.%Optional.ebe, (constants.%custom_witness.b80ef6.9, constants.%DestroyImpl.impl_witness.6e5) [concrete = constants.%Destroy.facet.54d]
// CHECK:STDOUT:     %.loc24_19.20: %Destroy.type = converted constants.%Optional.ebe, %Destroy.facet.loc24_19.4 [concrete = constants.%Destroy.facet.54d]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %SelfDestruct.call.loc24_19.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc24_19.8)
// CHECK:STDOUT:     %.loc24_19.21: %Destroy.type = converted constants.%as_type.deb, constants.%impl.elem1.7dd [symbolic = %impl.elem1 (constants.%impl.elem1.7dd)]
// CHECK:STDOUT:     %.loc24_19.22: %Destroy.type = converted constants.%as_type.deb, constants.%impl.elem1.7dd [symbolic = %impl.elem1 (constants.%impl.elem1.7dd)]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %SelfDestruct.call.loc24_19.3: init %empty_tuple.type = call %SelfDestruct.specific_fn.3(%var)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc24_19.1(%self.param: ref %.ecc) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc24_19.2(%self.param: ref %MaybeUnformed.f92) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc24_19.3(%self.param: ref bool) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc24_19.4(%self.param: ref %struct_type.value.has_value.e0a) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc24_19.5(%self.param: ref %DefaultOptionalStorage.f55) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc24_19.6(%self.param: ref %Optional.ebe) = "no_op";
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
// CHECK:STDOUT:   %.loc38_19.5: ref %Optional.ebe = temporary_storage
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.call: init %Optional.ebe to %.loc38_19.5 = call %bound_method.loc38_19.4(%c.ref, %var)
// CHECK:STDOUT:   %.loc38_19.6: ref %Optional.ebe = temporary %.loc38_19.5, %T.as_type.as.Iterate.impl.Next.call
// CHECK:STDOUT:   %.loc38_19.7: %Optional.HasValue.type.1cf = specific_constant imports.%Core.import_ref.c82, @Optional(constants.%OptionalStorage.facet.19f) [concrete = constants.%Optional.HasValue.768]
// CHECK:STDOUT:   %HasValue.ref: %Optional.HasValue.type.1cf = name_ref HasValue, %.loc38_19.7 [concrete = constants.%Optional.HasValue.768]
// CHECK:STDOUT:   %Optional.HasValue.bound: <bound method> = bound_method %.loc38_19.6, %HasValue.ref
// CHECK:STDOUT:   %Optional.HasValue.specific_fn: <specific function> = specific_function %HasValue.ref, @Optional.HasValue(constants.%OptionalStorage.facet.19f) [concrete = constants.%Optional.HasValue.specific_fn.89a]
// CHECK:STDOUT:   %bound_method.loc38_19.5: <bound method> = bound_method %.loc38_19.6, %Optional.HasValue.specific_fn
// CHECK:STDOUT:   %.loc38_19.8: %Optional.ebe = acquire_value %.loc38_19.6
// CHECK:STDOUT:   %Optional.HasValue.call: init bool = call %bound_method.loc38_19.5(%.loc38_19.8)
// CHECK:STDOUT:   %.loc38_19.9: bool = value_of_initializer %Optional.HasValue.call
// CHECK:STDOUT:   %.loc38_19.10: bool = converted %Optional.HasValue.call, %.loc38_19.9
// CHECK:STDOUT:   if %.loc38_19.10 br !for.body else br !for.done
// CHECK:STDOUT:
// CHECK:STDOUT: !for.body:
// CHECK:STDOUT:   %.loc38_19.11: %Optional.Get.type.b2d = specific_constant imports.%Core.import_ref.29e, @Optional(constants.%OptionalStorage.facet.19f) [concrete = constants.%Optional.Get.f12]
// CHECK:STDOUT:   %Get.ref: %Optional.Get.type.b2d = name_ref Get, %.loc38_19.11 [concrete = constants.%Optional.Get.f12]
// CHECK:STDOUT:   %Optional.Get.bound: <bound method> = bound_method %.loc38_19.6, %Get.ref
// CHECK:STDOUT:   %Optional.Get.specific_fn: <specific function> = specific_function %Get.ref, @Optional.Get(constants.%OptionalStorage.facet.19f) [concrete = constants.%Optional.Get.specific_fn.67e]
// CHECK:STDOUT:   %bound_method.loc38_19.6: <bound method> = bound_method %.loc38_19.6, %Optional.Get.specific_fn
// CHECK:STDOUT:   %.loc38_19.12: %Optional.ebe = acquire_value %.loc38_19.6
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
// CHECK:STDOUT:   %Destroy.facet.loc38_19.1: %Destroy.type = facet_value constants.%Optional.ebe, (constants.%custom_witness.b80ef6.9, constants.%DestroyImpl.impl_witness.6e5) [concrete = constants.%Destroy.facet.54d]
// CHECK:STDOUT:   %.loc38_19.15: %Destroy.type = converted constants.%Optional.ebe, %Destroy.facet.loc38_19.1 [concrete = constants.%Destroy.facet.54d]
// CHECK:STDOUT:   %Destroy.facet.loc38_19.2: %Destroy.type = facet_value constants.%Optional.ebe, (constants.%custom_witness.b80ef6.9, constants.%DestroyImpl.impl_witness.6e5) [concrete = constants.%Destroy.facet.54d]
// CHECK:STDOUT:   %.loc38_19.16: %Destroy.type = converted constants.%Optional.ebe, %Destroy.facet.loc38_19.2 [concrete = constants.%Destroy.facet.54d]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc38_19.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc38_19.6)
// CHECK:STDOUT:   br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT: !for.done:
// CHECK:STDOUT:   %Destroy.facet.loc38_19.3: %Destroy.type = facet_value constants.%Optional.ebe, (constants.%custom_witness.b80ef6.9, constants.%DestroyImpl.impl_witness.6e5) [concrete = constants.%Destroy.facet.54d]
// CHECK:STDOUT:   %.loc38_19.17: %Destroy.type = converted constants.%Optional.ebe, %Destroy.facet.loc38_19.3 [concrete = constants.%Destroy.facet.54d]
// CHECK:STDOUT:   %Destroy.facet.loc38_19.4: %Destroy.type = facet_value constants.%Optional.ebe, (constants.%custom_witness.b80ef6.9, constants.%DestroyImpl.impl_witness.6e5) [concrete = constants.%Destroy.facet.54d]
// CHECK:STDOUT:   %.loc38_19.18: %Destroy.type = converted constants.%Optional.ebe, %Destroy.facet.loc38_19.4 [concrete = constants.%Destroy.facet.54d]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc38_19.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc38_19.6)
// CHECK:STDOUT:   %Destroy.facet.loc38_19.5: %Destroy.type = facet_value constants.%tuple.type.005, (constants.%custom_witness.b80ef6.11, constants.%DestroyImpl.impl_witness.729) [concrete = constants.%Destroy.facet.0b5]
// CHECK:STDOUT:   %.loc38_19.19: %Destroy.type = converted constants.%tuple.type.005, %Destroy.facet.loc38_19.5 [concrete = constants.%Destroy.facet.0b5]
// CHECK:STDOUT:   %Destroy.facet.loc38_19.6: %Destroy.type = facet_value constants.%tuple.type.005, (constants.%custom_witness.b80ef6.11, constants.%DestroyImpl.impl_witness.729) [concrete = constants.%Destroy.facet.0b5]
// CHECK:STDOUT:   %.loc38_19.20: %Destroy.type = converted constants.%tuple.type.005, %Destroy.facet.loc38_19.6 [concrete = constants.%Destroy.facet.0b5]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc38_19.3: init %empty_tuple.type = call %SelfDestruct.specific_fn.3(%var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestIterate(constants.%R) {
// CHECK:STDOUT:   %R.patt.loc20_17.2 => constants.%R.patt
// CHECK:STDOUT:   %R.loc20_17.1 => constants.%R
// CHECK:STDOUT:   %R.as_type.loc20_61.1 => constants.%R.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.c3b
// CHECK:STDOUT:   %r.param_patt.loc20_59.2 => constants.%r.param_patt.304
// CHECK:STDOUT:   %r.patt.loc20_59.2 => constants.%r.patt.240
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestIterate(constants.%facet_value.4f9) {
// CHECK:STDOUT:   %R.patt.loc20_17.2 => constants.%R.patt
// CHECK:STDOUT:   %R.loc20_17.1 => constants.%facet_value.4f9
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
// CHECK:STDOUT:   %impl.elem1 => constants.%Destroy.facet.0b5
// CHECK:STDOUT:   %as_type => constants.%tuple.type.005
// CHECK:STDOUT:   %require_complete.1 => constants.%complete_type.320
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type => constants.%Iterate.WithSelf.Next.type.d0b
// CHECK:STDOUT:   %.loc24_19.24 => constants.%.a5a
// CHECK:STDOUT:   %impl.elem3.loc24_19.2 => constants.%T.as_type.as.Iterate.impl.Next.984
// CHECK:STDOUT:   %specific_impl_fn.loc24_19.4 => constants.%T.as_type.as.Iterate.impl.Next.specific_fn
// CHECK:STDOUT:   %SelfDestruct.specific_fn.5 => constants.%SelfDestruct.specific_fn.c59
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_adl_returns_same_types_mutable.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %Iterate.type: type = facet_type <@Iterate> [concrete]
// CHECK:STDOUT:   %OptionalStorage.type: type = facet_type <@OptionalStorage> [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %facet_type.bd0: type = facet_type <@Copy & @Destroy> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %T.220e40.1: %OptionalStorage.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Optional.Get.type.786: type = fn_type @Optional.Get, @Optional(%T.220e40.1) [symbolic]
// CHECK:STDOUT:   %Optional.Get.2d0: %Optional.Get.type.786 = struct_value () [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.type.3ea: type = fn_type @Optional.HasValue, @Optional(%T.220e40.1) [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.f2f: %Optional.HasValue.type.3ea = struct_value () [symbolic]
// CHECK:STDOUT:   %T.f84392.1: %facet_type.bd0 = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.476: type = class_type @DefaultOptionalStorage, @DefaultOptionalStorage(%T.f84392.1) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Get.type.f27: type = fn_type @T.as_type.as.OptionalStorage.impl.Get, @T.as_type.as.OptionalStorage.impl(%T.f84392.1) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Get.e40: %T.as_type.as.OptionalStorage.impl.Get.type.f27 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Has.type.18e: type = fn_type @T.as_type.as.OptionalStorage.impl.Has, @T.as_type.as.OptionalStorage.impl(%T.f84392.1) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Has.463: %T.as_type.as.OptionalStorage.impl.Has.type.18e = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Some.type.292: type = fn_type @T.as_type.as.OptionalStorage.impl.Some, @T.as_type.as.OptionalStorage.impl(%T.f84392.1) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Some.ae9: %T.as_type.as.OptionalStorage.impl.Some.type.292 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.None.type.b57: type = fn_type @T.as_type.as.OptionalStorage.impl.None, @T.as_type.as.OptionalStorage.impl(%T.f84392.1) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.None.791: %T.as_type.as.OptionalStorage.impl.None.type.b57 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SelfDestruct.type: type = fn_type @SelfDestruct [concrete]
// CHECK:STDOUT:   %SelfDestruct: %SelfDestruct.type = struct_value () [concrete]
// CHECK:STDOUT:   %N.fe9: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %.Self.500: %Iterate.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %CppRange.type: type = facet_type <@CppRange> [concrete]
// CHECK:STDOUT:   %.Self.e0b: %CppRange.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %CppRangeForIterate.lookup_impl_witness.50b: <witness> = lookup_impl_witness %.Self.e0b, @CppRangeForIterate [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.32d3: type = impl_witness_access %CppRangeForIterate.lookup_impl_witness.50b, element0 [symbolic_self]
// CHECK:STDOUT:   %CppUnsafeDeref.lookup_impl_witness.851: <witness> = lookup_impl_witness %impl.elem0.32d3, @CppUnsafeDeref [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.10d: type = impl_witness_access %CppUnsafeDeref.lookup_impl_witness.851, element0 [symbolic_self]
// CHECK:STDOUT:   %CppRange_where.type.d682d6.1: type = facet_type <@CppRange where %impl.elem0.10d impls @Copy> [concrete]
// CHECK:STDOUT:   %T.f45530.3: %CppRange_where.type.d682d6.1 = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.type.d4a: type = fn_type @T.as_type.as.Iterate.impl.Next, @T.as_type.as.Iterate.impl(%T.f45530.3) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.5a8: %T.as_type.as.Iterate.impl.Next.type.d4a = struct_value () [symbolic]
// CHECK:STDOUT:   %CppRangeForIterate.lookup_impl_witness.b581a8.2: <witness> = lookup_impl_witness %T.f45530.3, @CppRangeForIterate [symbolic]
// CHECK:STDOUT:   %impl.elem0.5f971c.2: type = impl_witness_access %CppRangeForIterate.lookup_impl_witness.b581a8.2, element0 [symbolic]
// CHECK:STDOUT:   %CppUnsafeDeref.lookup_impl_witness.87e543.2: <witness> = lookup_impl_witness %impl.elem0.5f971c.2, @CppUnsafeDeref [symbolic]
// CHECK:STDOUT:   %impl.elem0.493765.2: type = impl_witness_access %CppUnsafeDeref.lookup_impl_witness.87e543.2, element0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.type.8fb: type = fn_type @T.as_type.as.Iterate.impl.NewCursor, @T.as_type.as.Iterate.impl(%T.f45530.3) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.8a1: %T.as_type.as.Iterate.impl.NewCursor.type.8fb = struct_value () [symbolic]
// CHECK:STDOUT:   %impl.elem1.c2dc15.2: type = impl_witness_access %CppRangeForIterate.lookup_impl_witness.b581a8.2, element1 [symbolic]
// CHECK:STDOUT:   %tuple.type.1ba: type = tuple_type (%impl.elem0.5f971c.2, %impl.elem1.c2dc15.2) [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.lookup_impl_witness.dc9: <witness> = lookup_impl_witness %tuple.type.1ba, @SubobjectDestroy [symbolic]
// CHECK:STDOUT:   %DestroyImpl.lookup_impl_witness.504: <witness> = lookup_impl_witness %tuple.type.1ba, @DestroyImpl [symbolic]
// CHECK:STDOUT:   %Destroy.facet.9b6: %Destroy.type = facet_value %tuple.type.1ba, (%SubobjectDestroy.lookup_impl_witness.dc9, %DestroyImpl.lookup_impl_witness.504) [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.lookup_impl_witness.b15da7.2: <witness> = lookup_impl_witness %impl.elem0.493765.2, @SubobjectDestroy [symbolic]
// CHECK:STDOUT:   %DestroyImpl.lookup_impl_witness.4e7f44.2: <witness> = lookup_impl_witness %impl.elem0.493765.2, @DestroyImpl [symbolic]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.167: <witness> = lookup_impl_witness %impl.elem0.493765.2, @Copy [symbolic]
// CHECK:STDOUT:   %facet_value.189: %facet_type.bd0 = facet_value %impl.elem0.493765.2, (%SubobjectDestroy.lookup_impl_witness.b15da7.2, %DestroyImpl.lookup_impl_witness.4e7f44.2, %Copy.lookup_impl_witness.167) [symbolic]
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete]
// CHECK:STDOUT:   %f64.761: type = class_type @Float, @Float(%int_64) [concrete]
// CHECK:STDOUT:   %pattern_type.667: type = pattern_type %f64.761 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.3: type = fn_type @SubobjectDestroy.WithSelf.Op.loc32_53.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.3: %SubobjectDestroy.WithSelf.Op.type.6870fd.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.3: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.3), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.47f: %SubobjectDestroy.type = facet_value %f64.761, (%custom_witness.b80ef6.3) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.a23: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.47f) [concrete]
// CHECK:STDOUT:   %Float.as.Copy.impl.Op.type.e8d: type = fn_type @Float.as.Copy.impl.Op, @Float.as.Copy.impl(%N.fe9) [symbolic]
// CHECK:STDOUT:   %Float.as.Copy.impl.Op.fe9: %Float.as.Copy.impl.Op.type.e8d = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.eac: <witness> = impl_witness imports.%Copy.impl_witness_table.14d, @Float.as.Copy.impl(%int_64) [concrete]
// CHECK:STDOUT:   %facet_value.0d3: %facet_type.bd0 = facet_value %f64.761, (%custom_witness.b80ef6.3, %DestroyImpl.impl_witness.a23, %Copy.impl_witness.eac) [concrete]
// CHECK:STDOUT:   %Iterate.lookup_impl_witness.1d8: <witness> = lookup_impl_witness %.Self.500, @Iterate [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.4ce: %facet_type.bd0 = impl_witness_access %Iterate.lookup_impl_witness.1d8, element0 [symbolic_self]
// CHECK:STDOUT:   %Iterate_where.type.a66: type = facet_type <@Iterate where %impl.elem0.4ce = %facet_value.0d3> [concrete]
// CHECK:STDOUT:   %pattern_type.ec9: type = pattern_type %Iterate_where.type.a66 [concrete]
// CHECK:STDOUT:   %R.patt: %pattern_type.ec9 = symbolic_binding_pattern R, 0 [symbolic]
// CHECK:STDOUT:   %R: %Iterate_where.type.a66 = symbolic_binding R, 0 [symbolic]
// CHECK:STDOUT:   %R.as_type: type = facet_access_type %R [symbolic]
// CHECK:STDOUT:   %pattern_type.c3b: type = pattern_type %R.as_type [symbolic]
// CHECK:STDOUT:   %r.param_patt.304: %pattern_type.c3b = value_param_pattern [symbolic]
// CHECK:STDOUT:   %r.patt.240: %pattern_type.c3b = wrapper_binding_pattern r, %r.param_patt.304 [symbolic]
// CHECK:STDOUT:   %i.patt: %pattern_type.667 = value_binding_pattern i [concrete]
// CHECK:STDOUT:   %Iterate.lookup_impl_witness.6fa: <witness> = lookup_impl_witness %R, @Iterate [symbolic]
// CHECK:STDOUT:   %Iterate.facet.e8c: %Iterate.type = facet_value %R.as_type, (%Iterate.lookup_impl_witness.6fa) [symbolic]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.a57: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.e8c) [symbolic]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.01e: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.e8c) [symbolic]
// CHECK:STDOUT:   %.b10: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.a57, %Iterate.facet.e8c [symbolic]
// CHECK:STDOUT:   %impl.elem2.81a: %.b10 = impl_witness_access %Iterate.lookup_impl_witness.6fa, element2 [symbolic]
// CHECK:STDOUT:   %impl.elem1.7dd: %Destroy.type = impl_witness_access %Iterate.lookup_impl_witness.6fa, element1 [symbolic]
// CHECK:STDOUT:   %as_type.deb: type = facet_access_type %impl.elem1.7dd [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.6ec: <specific function> = specific_impl_function %impl.elem2.81a, @Iterate.WithSelf.NewCursor(%Iterate.facet.e8c) [symbolic]
// CHECK:STDOUT:   %.063: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.01e, %Iterate.facet.e8c [symbolic]
// CHECK:STDOUT:   %impl.elem3.ae5: %.063 = impl_witness_access %Iterate.lookup_impl_witness.6fa, element3 [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.f55: type = class_type @DefaultOptionalStorage, @DefaultOptionalStorage(%facet_value.0d3) [concrete]
// CHECK:STDOUT:   %OptionalStorage.impl_witness.050: <witness> = impl_witness imports.%OptionalStorage.impl_witness_table.1c4, @T.as_type.as.OptionalStorage.impl(%facet_value.0d3) [concrete]
// CHECK:STDOUT:   %OptionalStorage.facet.19f: %OptionalStorage.type = facet_value %f64.761, (%OptionalStorage.impl_witness.050) [concrete]
// CHECK:STDOUT:   %Optional.ebe: type = class_type @Optional, @Optional(%OptionalStorage.facet.19f) [concrete]
// CHECK:STDOUT:   %specific_impl_fn.2e1: <specific function> = specific_impl_function %impl.elem3.ae5, @Iterate.WithSelf.Next(%Iterate.facet.e8c) [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.type.1cf: type = fn_type @Optional.HasValue, @Optional(%OptionalStorage.facet.19f) [concrete]
// CHECK:STDOUT:   %Optional.HasValue.768: %Optional.HasValue.type.1cf = struct_value () [concrete]
// CHECK:STDOUT:   %Optional.Get.type.b2d: type = fn_type @Optional.Get, @Optional(%OptionalStorage.facet.19f) [concrete]
// CHECK:STDOUT:   %Optional.Get.f12: %Optional.Get.type.b2d = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.facet.941: %Destroy.type = facet_value %f64.761, (%custom_witness.b80ef6.3, %DestroyImpl.impl_witness.a23) [concrete]
// CHECK:STDOUT:   %MaybeUnformed.f92: type = class_type @MaybeUnformed, @MaybeUnformed(%Destroy.facet.941) [concrete]
// CHECK:STDOUT:   %.ecc: type = maybe_unformed_type %f64.761 [concrete]
// CHECK:STDOUT:   %struct_type.value.has_value.e0a: type = struct_type {.value: %MaybeUnformed.f92, .has_value: bool} [concrete]
// CHECK:STDOUT:   %Optional.HasValue.specific_fn.89a: <specific function> = specific_function %Optional.HasValue.768, @Optional.HasValue(%OptionalStorage.facet.19f) [concrete]
// CHECK:STDOUT:   %Optional.Get.specific_fn.67e: <specific function> = specific_function %Optional.Get.f12, @Optional.Get(%OptionalStorage.facet.19f) [concrete]
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
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.9: type = fn_type @SubobjectDestroy.WithSelf.Op.loc36_19.6 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.9: %SubobjectDestroy.WithSelf.Op.type.6870fd.9 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.9: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.9), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.cdd: %SubobjectDestroy.type = facet_value %Optional.ebe, (%custom_witness.b80ef6.9) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.6e5: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.cdd) [concrete]
// CHECK:STDOUT:   %Destroy.facet.54d: %Destroy.type = facet_value %Optional.ebe, (%custom_witness.b80ef6.9, %DestroyImpl.impl_witness.6e5) [concrete]
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
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.10: type = fn_type @SubobjectDestroy.WithSelf.Op.loc43 [concrete]
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
// CHECK:STDOUT:   %Destroy.facet.5f38: %Destroy.type = facet_value %tuple.type.dd0, (%custom_witness.b80ef6.11, %DestroyImpl.impl_witness.1fa) [concrete]
// CHECK:STDOUT:   %Iterate.impl_witness.752: <witness> = impl_witness imports.%Iterate.impl_witness_table.121, @T.as_type.as.Iterate.impl(%facet_value.fd4) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.type.7d1: type = fn_type @T.as_type.as.Iterate.impl.NewCursor, @T.as_type.as.Iterate.impl(%facet_value.fd4) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.482: %T.as_type.as.Iterate.impl.NewCursor.type.7d1 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.type.e77: type = fn_type @T.as_type.as.Iterate.impl.Next, @T.as_type.as.Iterate.impl(%facet_value.fd4) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.759: %T.as_type.as.Iterate.impl.Next.type.e77 = struct_value () [concrete]
// CHECK:STDOUT:   %facet_value.463: %Iterate_where.type.a66 = facet_value %MutableRange, (%Iterate.impl_witness.752) [concrete]
// CHECK:STDOUT:   %r.param_patt.aeb: %pattern_type.992 = value_param_pattern [concrete]
// CHECK:STDOUT:   %r.patt.892: %pattern_type.992 = wrapper_binding_pattern r, %r.param_patt.aeb [concrete]
// CHECK:STDOUT:   %Iterate.facet.c03: %Iterate.type = facet_value %MutableRange, (%Iterate.impl_witness.752) [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.b9c: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.c03) [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.692: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.c03) [concrete]
// CHECK:STDOUT:   %.d6e: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.b9c, %Iterate.facet.c03 [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.specific_fn: <specific function> = specific_function %T.as_type.as.Iterate.impl.NewCursor.482, @T.as_type.as.Iterate.impl.NewCursor(%facet_value.fd4) [concrete]
// CHECK:STDOUT:   %.446: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.692, %Iterate.facet.c03 [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.specific_fn: <specific function> = specific_function %T.as_type.as.Iterate.impl.Next.759, @T.as_type.as.Iterate.impl.Next(%facet_value.fd4) [concrete]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.e1d: <specific function> = specific_function %SelfDestruct, @SelfDestruct(%Destroy.facet.5f38) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.481: type = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%DefaultOptionalStorage (constants.%DefaultOptionalStorage.476)]
// CHECK:STDOUT:   %Core.import_ref.66d: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.None.type (%T.as_type.as.OptionalStorage.impl.None.type.b57) = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.None (constants.%T.as_type.as.OptionalStorage.impl.None.791)]
// CHECK:STDOUT:   %Core.import_ref.32c: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Some.type (%T.as_type.as.OptionalStorage.impl.Some.type.292) = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Some (constants.%T.as_type.as.OptionalStorage.impl.Some.ae9)]
// CHECK:STDOUT:   %Core.import_ref.801: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Has.type (%T.as_type.as.OptionalStorage.impl.Has.type.18e) = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Has (constants.%T.as_type.as.OptionalStorage.impl.Has.463)]
// CHECK:STDOUT:   %Core.import_ref.eec: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Get.type (%T.as_type.as.OptionalStorage.impl.Get.type.f27) = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Get (constants.%T.as_type.as.OptionalStorage.impl.Get.e40)]
// CHECK:STDOUT:   %Core.import_ref.a6a = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %OptionalStorage.impl_witness_table.1c4 = impl_witness_table (%Core.import_ref.481, %Core.import_ref.66d, %Core.import_ref.32c, %Core.import_ref.801, %Core.import_ref.eec, %Core.import_ref.a6a), @T.as_type.as.OptionalStorage.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.7b5: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/types/optional, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.7b5), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.c82: @Optional.%Optional.HasValue.type (%Optional.HasValue.type.3ea) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.HasValue (constants.%Optional.HasValue.f2f)]
// CHECK:STDOUT:   %Core.import_ref.29e: @Optional.%Optional.Get.type (%Optional.Get.type.786) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.Get (constants.%Optional.Get.2d0)]
// CHECK:STDOUT:   %Core.import_ref.724: %facet_type.bd0 = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.Iterate.impl.%facet_value (constants.%facet_value.189)]
// CHECK:STDOUT:   %Core.import_ref.dd9: %Destroy.type = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.Iterate.impl.%Destroy.facet (constants.%Destroy.facet.9b6)]
// CHECK:STDOUT:   %Core.import_ref.946: @T.as_type.as.Iterate.impl.%T.as_type.as.Iterate.impl.NewCursor.type (%T.as_type.as.Iterate.impl.NewCursor.type.8fb) = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.Iterate.impl.%T.as_type.as.Iterate.impl.NewCursor (constants.%T.as_type.as.Iterate.impl.NewCursor.8a1)]
// CHECK:STDOUT:   %Core.import_ref.cfb: @T.as_type.as.Iterate.impl.%T.as_type.as.Iterate.impl.Next.type (%T.as_type.as.Iterate.impl.Next.type.d4a) = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.Iterate.impl.%T.as_type.as.Iterate.impl.Next (constants.%T.as_type.as.Iterate.impl.Next.5a8)]
// CHECK:STDOUT:   %Iterate.impl_witness_table.121 = impl_witness_table (%Core.import_ref.724, %Core.import_ref.dd9, %Core.import_ref.946, %Core.import_ref.cfb), @T.as_type.as.Iterate.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.19b: @Float.as.Copy.impl.%Float.as.Copy.impl.Op.type (%Float.as.Copy.impl.Op.type.e8d) = import_ref Core//prelude/types/float, loc{{\d+_\d+}}, loaded [symbolic = @Float.as.Copy.impl.%Float.as.Copy.impl.Op (constants.%Float.as.Copy.impl.Op.fe9)]
// CHECK:STDOUT:   %Copy.impl_witness_table.14d = impl_witness_table (%Core.import_ref.19b), @Float.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.e0d0: @Float.as.AddAssignWith.impl.%Float.as.AddAssignWith.impl.Op.type (%Float.as.AddAssignWith.impl.Op.type.763) = import_ref Core//prelude/types/float, loc{{\d+_\d+}}, loaded [symbolic = @Float.as.AddAssignWith.impl.%Float.as.AddAssignWith.impl.Op (constants.%Float.as.AddAssignWith.impl.Op.8aa)]
// CHECK:STDOUT:   %AddAssignWith.impl_witness_table = impl_witness_table (%Core.import_ref.e0d0), @Float.as.AddAssignWith.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @TestIterate(%R.loc32_17.2: %Iterate_where.type.a66) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Iterate.lookup_impl_witness: <witness> = lookup_impl_witness %R.loc32_17.1, @Iterate [symbolic = %Iterate.lookup_impl_witness (constants.%Iterate.lookup_impl_witness.6fa)]
// CHECK:STDOUT:   %Iterate.facet.loc36_19.5: %Iterate.type = facet_value %R.as_type.loc32_61.1, (%Iterate.lookup_impl_witness) [symbolic = %Iterate.facet.loc36_19.5 (constants.%Iterate.facet.e8c)]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.loc36_19.5) [symbolic = %Iterate.WithSelf.NewCursor.type (constants.%Iterate.WithSelf.NewCursor.type.a57)]
// CHECK:STDOUT:   %.loc36_19.23: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type, %Iterate.facet.loc36_19.5 [symbolic = %.loc36_19.23 (constants.%.b10)]
// CHECK:STDOUT:   %impl.elem2.loc36_19.2: @TestIterate.%.loc36_19.23 (%.b10) = impl_witness_access %Iterate.lookup_impl_witness, element2 [symbolic = %impl.elem2.loc36_19.2 (constants.%impl.elem2.81a)]
// CHECK:STDOUT:   %specific_impl_fn.loc36_19.3: <specific function> = specific_impl_function %impl.elem2.loc36_19.2, @Iterate.WithSelf.NewCursor(%Iterate.facet.loc36_19.5) [symbolic = %specific_impl_fn.loc36_19.3 (constants.%specific_impl_fn.6ec)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.loc36_19.5) [symbolic = %Iterate.WithSelf.Next.type (constants.%Iterate.WithSelf.Next.type.01e)]
// CHECK:STDOUT:   %.loc36_19.24: type = fn_type_with_self_type %Iterate.WithSelf.Next.type, %Iterate.facet.loc36_19.5 [symbolic = %.loc36_19.24 (constants.%.063)]
// CHECK:STDOUT:   %impl.elem3.loc36_19.2: @TestIterate.%.loc36_19.24 (%.063) = impl_witness_access %Iterate.lookup_impl_witness, element3 [symbolic = %impl.elem3.loc36_19.2 (constants.%impl.elem3.ae5)]
// CHECK:STDOUT:   %specific_impl_fn.loc36_19.4: <specific function> = specific_impl_function %impl.elem3.loc36_19.2, @Iterate.WithSelf.Next(%Iterate.facet.loc36_19.5) [symbolic = %specific_impl_fn.loc36_19.4 (constants.%specific_impl_fn.2e1)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%r.param: @TestIterate.%R.as_type.loc32_61.1 (%R.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %i.patt: %pattern_type.667 = value_binding_pattern i [concrete = constants.%i.patt]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %r.ref: @TestIterate.%R.as_type.loc32_61.1 (%R.as_type) = name_ref r, %r
// CHECK:STDOUT:     %impl.elem2.loc36_19.1: @TestIterate.%.loc36_19.23 (%.b10) = impl_witness_access constants.%Iterate.lookup_impl_witness.6fa, element2 [symbolic = %impl.elem2.loc36_19.2 (constants.%impl.elem2.81a)]
// CHECK:STDOUT:     %bound_method.loc36_19.1: <bound method> = bound_method %r.ref, %impl.elem2.loc36_19.1
// CHECK:STDOUT:     %Iterate.facet.loc36_19.1: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.6fa) [symbolic = %Iterate.facet.loc36_19.5 (constants.%Iterate.facet.e8c)]
// CHECK:STDOUT:     %.loc36_19.1: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc36_19.1 [symbolic = %Iterate.facet.loc36_19.5 (constants.%Iterate.facet.e8c)]
// CHECK:STDOUT:     %Iterate.facet.loc36_19.2: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.6fa) [symbolic = %Iterate.facet.loc36_19.5 (constants.%Iterate.facet.e8c)]
// CHECK:STDOUT:     %.loc36_19.2: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc36_19.2 [symbolic = %Iterate.facet.loc36_19.5 (constants.%Iterate.facet.e8c)]
// CHECK:STDOUT:     %specific_impl_fn.loc36_19.1: <specific function> = specific_impl_function %impl.elem2.loc36_19.1, @Iterate.WithSelf.NewCursor(constants.%Iterate.facet.e8c) [symbolic = %specific_impl_fn.loc36_19.3 (constants.%specific_impl_fn.6ec)]
// CHECK:STDOUT:     %bound_method.loc36_19.2: <bound method> = bound_method %r.ref, %specific_impl_fn.loc36_19.1
// CHECK:STDOUT:     %var: ref @TestIterate.%as_type (%as_type.deb) = var_storage invalid
// CHECK:STDOUT:     %Iterate.WithSelf.NewCursor.call: init @TestIterate.%as_type (%as_type.deb) to %var = call %bound_method.loc36_19.2(%r.ref)
// CHECK:STDOUT:     assign %var, %Iterate.WithSelf.NewCursor.call
// CHECK:STDOUT:     br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.next:
// CHECK:STDOUT:     %impl.elem3.loc36_19.1: @TestIterate.%.loc36_19.24 (%.063) = impl_witness_access constants.%Iterate.lookup_impl_witness.6fa, element3 [symbolic = %impl.elem3.loc36_19.2 (constants.%impl.elem3.ae5)]
// CHECK:STDOUT:     %bound_method.loc36_19.3: <bound method> = bound_method %r.ref, %impl.elem3.loc36_19.1
// CHECK:STDOUT:     %Iterate.facet.loc36_19.3: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.6fa) [symbolic = %Iterate.facet.loc36_19.5 (constants.%Iterate.facet.e8c)]
// CHECK:STDOUT:     %.loc36_19.3: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc36_19.3 [symbolic = %Iterate.facet.loc36_19.5 (constants.%Iterate.facet.e8c)]
// CHECK:STDOUT:     %Iterate.facet.loc36_19.4: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.6fa) [symbolic = %Iterate.facet.loc36_19.5 (constants.%Iterate.facet.e8c)]
// CHECK:STDOUT:     %.loc36_19.4: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc36_19.4 [symbolic = %Iterate.facet.loc36_19.5 (constants.%Iterate.facet.e8c)]
// CHECK:STDOUT:     %.loc36_19.5: %Destroy.type = converted constants.%as_type.deb, constants.%impl.elem1.7dd [symbolic = %impl.elem1 (constants.%impl.elem1.7dd)]
// CHECK:STDOUT:     %.loc36_19.6: %Destroy.type = converted constants.%as_type.deb, constants.%impl.elem1.7dd [symbolic = %impl.elem1 (constants.%impl.elem1.7dd)]
// CHECK:STDOUT:     %specific_impl_fn.loc36_19.2: <specific function> = specific_impl_function %impl.elem3.loc36_19.1, @Iterate.WithSelf.Next(constants.%Iterate.facet.e8c) [symbolic = %specific_impl_fn.loc36_19.4 (constants.%specific_impl_fn.2e1)]
// CHECK:STDOUT:     %bound_method.loc36_19.4: <bound method> = bound_method %r.ref, %specific_impl_fn.loc36_19.2
// CHECK:STDOUT:     %.loc36_19.7: ref %Optional.ebe = temporary_storage
// CHECK:STDOUT:     %Iterate.WithSelf.Next.call: init %Optional.ebe to %.loc36_19.7 = call %bound_method.loc36_19.4(%r.ref, %var)
// CHECK:STDOUT:     %.loc36_19.8: ref %Optional.ebe = temporary %.loc36_19.7, %Iterate.WithSelf.Next.call
// CHECK:STDOUT:     %.loc36_19.9: %Optional.HasValue.type.1cf = specific_constant imports.%Core.import_ref.c82, @Optional(constants.%OptionalStorage.facet.19f) [concrete = constants.%Optional.HasValue.768]
// CHECK:STDOUT:     %HasValue.ref: %Optional.HasValue.type.1cf = name_ref HasValue, %.loc36_19.9 [concrete = constants.%Optional.HasValue.768]
// CHECK:STDOUT:     %Optional.HasValue.bound: <bound method> = bound_method %.loc36_19.8, %HasValue.ref
// CHECK:STDOUT:     %Optional.HasValue.specific_fn: <specific function> = specific_function %HasValue.ref, @Optional.HasValue(constants.%OptionalStorage.facet.19f) [concrete = constants.%Optional.HasValue.specific_fn.89a]
// CHECK:STDOUT:     %bound_method.loc36_19.5: <bound method> = bound_method %.loc36_19.8, %Optional.HasValue.specific_fn
// CHECK:STDOUT:     %.loc36_19.10: %Optional.ebe = acquire_value %.loc36_19.8
// CHECK:STDOUT:     %Optional.HasValue.call: init bool = call %bound_method.loc36_19.5(%.loc36_19.10)
// CHECK:STDOUT:     %.loc36_19.11: bool = value_of_initializer %Optional.HasValue.call
// CHECK:STDOUT:     %.loc36_19.12: bool = converted %Optional.HasValue.call, %.loc36_19.11
// CHECK:STDOUT:     if %.loc36_19.12 br !for.body else br !for.done
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.body:
// CHECK:STDOUT:     %.loc36_19.13: %Optional.Get.type.b2d = specific_constant imports.%Core.import_ref.29e, @Optional(constants.%OptionalStorage.facet.19f) [concrete = constants.%Optional.Get.f12]
// CHECK:STDOUT:     %Get.ref: %Optional.Get.type.b2d = name_ref Get, %.loc36_19.13 [concrete = constants.%Optional.Get.f12]
// CHECK:STDOUT:     %Optional.Get.bound: <bound method> = bound_method %.loc36_19.8, %Get.ref
// CHECK:STDOUT:     %Optional.Get.specific_fn: <specific function> = specific_function %Get.ref, @Optional.Get(constants.%OptionalStorage.facet.19f) [concrete = constants.%Optional.Get.specific_fn.67e]
// CHECK:STDOUT:     %bound_method.loc36_19.6: <bound method> = bound_method %.loc36_19.8, %Optional.Get.specific_fn
// CHECK:STDOUT:     %.loc36_19.14: %Optional.ebe = acquire_value %.loc36_19.8
// CHECK:STDOUT:     %Optional.Get.call: init %f64.761 = call %bound_method.loc36_19.6(%.loc36_19.14)
// CHECK:STDOUT:     %.loc36_19.15: %f64.761 = value_of_initializer %Optional.Get.call
// CHECK:STDOUT:     %.loc36_19.16: %f64.761 = converted %Optional.Get.call, %.loc36_19.15
// CHECK:STDOUT:     %f64.loc36: type = type_literal constants.%f64.761 [concrete = constants.%f64.761]
// CHECK:STDOUT:     %i: %f64.761 = wrapper_binding i, %.loc36_19.16
// CHECK:STDOUT:     %sum.ref: ref %f64.761 = name_ref sum, %sum
// CHECK:STDOUT:     %i.ref: %f64.761 = name_ref i, %i
// CHECK:STDOUT:     %impl.elem0.loc37: %.48b = impl_witness_access constants.%AddAssignWith.impl_witness.5c7, element0 [concrete = constants.%Float.as.AddAssignWith.impl.Op.e04]
// CHECK:STDOUT:     %bound_method.loc37_9.1: <bound method> = bound_method %sum.ref, %impl.elem0.loc37
// CHECK:STDOUT:     %specific_fn.loc37: <specific function> = specific_function %impl.elem0.loc37, @Float.as.AddAssignWith.impl.Op(constants.%int_64) [concrete = constants.%Float.as.AddAssignWith.impl.Op.specific_fn]
// CHECK:STDOUT:     %bound_method.loc37_9.2: <bound method> = bound_method %sum.ref, %specific_fn.loc37
// CHECK:STDOUT:     %Float.as.AddAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc37_9.2(%sum.ref, %i.ref)
// CHECK:STDOUT:     %Destroy.facet.loc36_19.1: %Destroy.type = facet_value constants.%Optional.ebe, (constants.%custom_witness.b80ef6.9, constants.%DestroyImpl.impl_witness.6e5) [concrete = constants.%Destroy.facet.54d]
// CHECK:STDOUT:     %.loc36_19.17: %Destroy.type = converted constants.%Optional.ebe, %Destroy.facet.loc36_19.1 [concrete = constants.%Destroy.facet.54d]
// CHECK:STDOUT:     %Destroy.facet.loc36_19.2: %Destroy.type = facet_value constants.%Optional.ebe, (constants.%custom_witness.b80ef6.9, constants.%DestroyImpl.impl_witness.6e5) [concrete = constants.%Destroy.facet.54d]
// CHECK:STDOUT:     %.loc36_19.18: %Destroy.type = converted constants.%Optional.ebe, %Destroy.facet.loc36_19.2 [concrete = constants.%Destroy.facet.54d]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %SelfDestruct.call.loc36_19.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc36_19.8)
// CHECK:STDOUT:     br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.done:
// CHECK:STDOUT:     %Destroy.facet.loc36_19.3: %Destroy.type = facet_value constants.%Optional.ebe, (constants.%custom_witness.b80ef6.9, constants.%DestroyImpl.impl_witness.6e5) [concrete = constants.%Destroy.facet.54d]
// CHECK:STDOUT:     %.loc36_19.19: %Destroy.type = converted constants.%Optional.ebe, %Destroy.facet.loc36_19.3 [concrete = constants.%Destroy.facet.54d]
// CHECK:STDOUT:     %Destroy.facet.loc36_19.4: %Destroy.type = facet_value constants.%Optional.ebe, (constants.%custom_witness.b80ef6.9, constants.%DestroyImpl.impl_witness.6e5) [concrete = constants.%Destroy.facet.54d]
// CHECK:STDOUT:     %.loc36_19.20: %Destroy.type = converted constants.%Optional.ebe, %Destroy.facet.loc36_19.4 [concrete = constants.%Destroy.facet.54d]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %SelfDestruct.call.loc36_19.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc36_19.8)
// CHECK:STDOUT:     %.loc36_19.21: %Destroy.type = converted constants.%as_type.deb, constants.%impl.elem1.7dd [symbolic = %impl.elem1 (constants.%impl.elem1.7dd)]
// CHECK:STDOUT:     %.loc36_19.22: %Destroy.type = converted constants.%as_type.deb, constants.%impl.elem1.7dd [symbolic = %impl.elem1 (constants.%impl.elem1.7dd)]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %SelfDestruct.call.loc36_19.3: init %empty_tuple.type = call %SelfDestruct.specific_fn.3(%var)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc36_19.1(%self.param: ref %.ecc) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc36_19.2(%self.param: ref %MaybeUnformed.f92) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc36_19.3(%self.param: ref bool) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc36_19.4(%self.param: ref %struct_type.value.has_value.e0a) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc36_19.5(%self.param: ref %DefaultOptionalStorage.f55) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc36_19.6(%self.param: ref %Optional.ebe) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestRangeFor(%m.param: ref %MutableRange) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %i.patt: %pattern_type.667 = value_binding_pattern i [concrete = constants.%i.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %m.ref: ref %MutableRange = name_ref m, %m
// CHECK:STDOUT:   %impl.elem2: %.d6e = impl_witness_access constants.%Iterate.impl_witness.752, element2 [concrete = constants.%T.as_type.as.Iterate.impl.NewCursor.482]
// CHECK:STDOUT:   %bound_method.loc50_19.1: <bound method> = bound_method %m.ref, %impl.elem2
// CHECK:STDOUT:   %facet_value.loc50_19.1: %CppRange_where.type.d682d6.1 = facet_value constants.%MutableRange, (constants.%custom_witness.902, constants.%custom_witness.b80ef6.10, constants.%DestroyImpl.impl_witness.452, constants.%custom_witness.0a8, constants.%custom_witness.491, constants.%custom_witness.068, constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.a23, constants.%Copy.impl_witness.eac, constants.%custom_witness.b80ef6.10, constants.%DestroyImpl.impl_witness.452) [concrete = constants.%facet_value.fd4]
// CHECK:STDOUT:   %.loc50_19.1: %CppRange_where.type.d682d6.1 = converted constants.%MutableRange, %facet_value.loc50_19.1 [concrete = constants.%facet_value.fd4]
// CHECK:STDOUT:   %facet_value.loc50_19.2: %CppRange_where.type.d682d6.1 = facet_value constants.%MutableRange, (constants.%custom_witness.902, constants.%custom_witness.b80ef6.10, constants.%DestroyImpl.impl_witness.452, constants.%custom_witness.0a8, constants.%custom_witness.491, constants.%custom_witness.068, constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.a23, constants.%Copy.impl_witness.eac, constants.%custom_witness.b80ef6.10, constants.%DestroyImpl.impl_witness.452) [concrete = constants.%facet_value.fd4]
// CHECK:STDOUT:   %.loc50_19.2: %CppRange_where.type.d682d6.1 = converted constants.%MutableRange, %facet_value.loc50_19.2 [concrete = constants.%facet_value.fd4]
// CHECK:STDOUT:   %specific_fn.loc50_19.1: <specific function> = specific_function %impl.elem2, @T.as_type.as.Iterate.impl.NewCursor(constants.%facet_value.fd4) [concrete = constants.%T.as_type.as.Iterate.impl.NewCursor.specific_fn]
// CHECK:STDOUT:   %bound_method.loc50_19.2: <bound method> = bound_method %m.ref, %specific_fn.loc50_19.1
// CHECK:STDOUT:   %var: ref %tuple.type.dd0 = var_storage invalid
// CHECK:STDOUT:   %.loc50_18.1: %MutableRange = acquire_value %m.ref
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.call: init %tuple.type.dd0 to %var = call %bound_method.loc50_19.2(%.loc50_18.1)
// CHECK:STDOUT:   assign %var, %T.as_type.as.Iterate.impl.NewCursor.call
// CHECK:STDOUT:   br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT: !for.next:
// CHECK:STDOUT:   %impl.elem3: %.446 = impl_witness_access constants.%Iterate.impl_witness.752, element3 [concrete = constants.%T.as_type.as.Iterate.impl.Next.759]
// CHECK:STDOUT:   %bound_method.loc50_19.3: <bound method> = bound_method %m.ref, %impl.elem3
// CHECK:STDOUT:   %facet_value.loc50_19.3: %CppRange_where.type.d682d6.1 = facet_value constants.%MutableRange, (constants.%custom_witness.902, constants.%custom_witness.b80ef6.10, constants.%DestroyImpl.impl_witness.452, constants.%custom_witness.0a8, constants.%custom_witness.491, constants.%custom_witness.068, constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.a23, constants.%Copy.impl_witness.eac, constants.%custom_witness.b80ef6.10, constants.%DestroyImpl.impl_witness.452) [concrete = constants.%facet_value.fd4]
// CHECK:STDOUT:   %.loc50_19.3: %CppRange_where.type.d682d6.1 = converted constants.%MutableRange, %facet_value.loc50_19.3 [concrete = constants.%facet_value.fd4]
// CHECK:STDOUT:   %facet_value.loc50_19.4: %CppRange_where.type.d682d6.1 = facet_value constants.%MutableRange, (constants.%custom_witness.902, constants.%custom_witness.b80ef6.10, constants.%DestroyImpl.impl_witness.452, constants.%custom_witness.0a8, constants.%custom_witness.491, constants.%custom_witness.068, constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.a23, constants.%Copy.impl_witness.eac, constants.%custom_witness.b80ef6.10, constants.%DestroyImpl.impl_witness.452) [concrete = constants.%facet_value.fd4]
// CHECK:STDOUT:   %.loc50_19.4: %CppRange_where.type.d682d6.1 = converted constants.%MutableRange, %facet_value.loc50_19.4 [concrete = constants.%facet_value.fd4]
// CHECK:STDOUT:   %specific_fn.loc50_19.2: <specific function> = specific_function %impl.elem3, @T.as_type.as.Iterate.impl.Next(constants.%facet_value.fd4) [concrete = constants.%T.as_type.as.Iterate.impl.Next.specific_fn]
// CHECK:STDOUT:   %bound_method.loc50_19.4: <bound method> = bound_method %m.ref, %specific_fn.loc50_19.2
// CHECK:STDOUT:   %.loc50_19.5: ref %Optional.ebe = temporary_storage
// CHECK:STDOUT:   %.loc50_18.2: %MutableRange = acquire_value %m.ref
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.call: init %Optional.ebe to %.loc50_19.5 = call %bound_method.loc50_19.4(%.loc50_18.2, %var)
// CHECK:STDOUT:   %.loc50_19.6: ref %Optional.ebe = temporary %.loc50_19.5, %T.as_type.as.Iterate.impl.Next.call
// CHECK:STDOUT:   %.loc50_19.7: %Optional.HasValue.type.1cf = specific_constant imports.%Core.import_ref.c82, @Optional(constants.%OptionalStorage.facet.19f) [concrete = constants.%Optional.HasValue.768]
// CHECK:STDOUT:   %HasValue.ref: %Optional.HasValue.type.1cf = name_ref HasValue, %.loc50_19.7 [concrete = constants.%Optional.HasValue.768]
// CHECK:STDOUT:   %Optional.HasValue.bound: <bound method> = bound_method %.loc50_19.6, %HasValue.ref
// CHECK:STDOUT:   %Optional.HasValue.specific_fn: <specific function> = specific_function %HasValue.ref, @Optional.HasValue(constants.%OptionalStorage.facet.19f) [concrete = constants.%Optional.HasValue.specific_fn.89a]
// CHECK:STDOUT:   %bound_method.loc50_19.5: <bound method> = bound_method %.loc50_19.6, %Optional.HasValue.specific_fn
// CHECK:STDOUT:   %.loc50_19.8: %Optional.ebe = acquire_value %.loc50_19.6
// CHECK:STDOUT:   %Optional.HasValue.call: init bool = call %bound_method.loc50_19.5(%.loc50_19.8)
// CHECK:STDOUT:   %.loc50_19.9: bool = value_of_initializer %Optional.HasValue.call
// CHECK:STDOUT:   %.loc50_19.10: bool = converted %Optional.HasValue.call, %.loc50_19.9
// CHECK:STDOUT:   if %.loc50_19.10 br !for.body else br !for.done
// CHECK:STDOUT:
// CHECK:STDOUT: !for.body:
// CHECK:STDOUT:   %.loc50_19.11: %Optional.Get.type.b2d = specific_constant imports.%Core.import_ref.29e, @Optional(constants.%OptionalStorage.facet.19f) [concrete = constants.%Optional.Get.f12]
// CHECK:STDOUT:   %Get.ref: %Optional.Get.type.b2d = name_ref Get, %.loc50_19.11 [concrete = constants.%Optional.Get.f12]
// CHECK:STDOUT:   %Optional.Get.bound: <bound method> = bound_method %.loc50_19.6, %Get.ref
// CHECK:STDOUT:   %Optional.Get.specific_fn: <specific function> = specific_function %Get.ref, @Optional.Get(constants.%OptionalStorage.facet.19f) [concrete = constants.%Optional.Get.specific_fn.67e]
// CHECK:STDOUT:   %bound_method.loc50_19.6: <bound method> = bound_method %.loc50_19.6, %Optional.Get.specific_fn
// CHECK:STDOUT:   %.loc50_19.12: %Optional.ebe = acquire_value %.loc50_19.6
// CHECK:STDOUT:   %Optional.Get.call: init %f64.761 = call %bound_method.loc50_19.6(%.loc50_19.12)
// CHECK:STDOUT:   %.loc50_19.13: %f64.761 = value_of_initializer %Optional.Get.call
// CHECK:STDOUT:   %.loc50_19.14: %f64.761 = converted %Optional.Get.call, %.loc50_19.13
// CHECK:STDOUT:   %f64.loc50: type = type_literal constants.%f64.761 [concrete = constants.%f64.761]
// CHECK:STDOUT:   %i: %f64.761 = wrapper_binding i, %.loc50_19.14
// CHECK:STDOUT:   %sum.ref: ref %f64.761 = name_ref sum, %sum
// CHECK:STDOUT:   %i.ref: %f64.761 = name_ref i, %i
// CHECK:STDOUT:   %impl.elem0.loc51: %.48b = impl_witness_access constants.%AddAssignWith.impl_witness.5c7, element0 [concrete = constants.%Float.as.AddAssignWith.impl.Op.e04]
// CHECK:STDOUT:   %bound_method.loc51_9.1: <bound method> = bound_method %sum.ref, %impl.elem0.loc51
// CHECK:STDOUT:   %specific_fn.loc51: <specific function> = specific_function %impl.elem0.loc51, @Float.as.AddAssignWith.impl.Op(constants.%int_64) [concrete = constants.%Float.as.AddAssignWith.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc51_9.2: <bound method> = bound_method %sum.ref, %specific_fn.loc51
// CHECK:STDOUT:   %Float.as.AddAssignWith.impl.Op.call: init %empty_tuple.type = call %bound_method.loc51_9.2(%sum.ref, %i.ref)
// CHECK:STDOUT:   %Destroy.facet.loc50_19.1: %Destroy.type = facet_value constants.%Optional.ebe, (constants.%custom_witness.b80ef6.9, constants.%DestroyImpl.impl_witness.6e5) [concrete = constants.%Destroy.facet.54d]
// CHECK:STDOUT:   %.loc50_19.15: %Destroy.type = converted constants.%Optional.ebe, %Destroy.facet.loc50_19.1 [concrete = constants.%Destroy.facet.54d]
// CHECK:STDOUT:   %Destroy.facet.loc50_19.2: %Destroy.type = facet_value constants.%Optional.ebe, (constants.%custom_witness.b80ef6.9, constants.%DestroyImpl.impl_witness.6e5) [concrete = constants.%Destroy.facet.54d]
// CHECK:STDOUT:   %.loc50_19.16: %Destroy.type = converted constants.%Optional.ebe, %Destroy.facet.loc50_19.2 [concrete = constants.%Destroy.facet.54d]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc50_19.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc50_19.6)
// CHECK:STDOUT:   br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT: !for.done:
// CHECK:STDOUT:   %Destroy.facet.loc50_19.3: %Destroy.type = facet_value constants.%Optional.ebe, (constants.%custom_witness.b80ef6.9, constants.%DestroyImpl.impl_witness.6e5) [concrete = constants.%Destroy.facet.54d]
// CHECK:STDOUT:   %.loc50_19.17: %Destroy.type = converted constants.%Optional.ebe, %Destroy.facet.loc50_19.3 [concrete = constants.%Destroy.facet.54d]
// CHECK:STDOUT:   %Destroy.facet.loc50_19.4: %Destroy.type = facet_value constants.%Optional.ebe, (constants.%custom_witness.b80ef6.9, constants.%DestroyImpl.impl_witness.6e5) [concrete = constants.%Destroy.facet.54d]
// CHECK:STDOUT:   %.loc50_19.18: %Destroy.type = converted constants.%Optional.ebe, %Destroy.facet.loc50_19.4 [concrete = constants.%Destroy.facet.54d]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc50_19.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc50_19.6)
// CHECK:STDOUT:   %Destroy.facet.loc50_19.5: %Destroy.type = facet_value constants.%tuple.type.dd0, (constants.%custom_witness.b80ef6.11, constants.%DestroyImpl.impl_witness.1fa) [concrete = constants.%Destroy.facet.5f38]
// CHECK:STDOUT:   %.loc50_19.19: %Destroy.type = converted constants.%tuple.type.dd0, %Destroy.facet.loc50_19.5 [concrete = constants.%Destroy.facet.5f38]
// CHECK:STDOUT:   %Destroy.facet.loc50_19.6: %Destroy.type = facet_value constants.%tuple.type.dd0, (constants.%custom_witness.b80ef6.11, constants.%DestroyImpl.impl_witness.1fa) [concrete = constants.%Destroy.facet.5f38]
// CHECK:STDOUT:   %.loc50_19.20: %Destroy.type = converted constants.%tuple.type.dd0, %Destroy.facet.loc50_19.6 [concrete = constants.%Destroy.facet.5f38]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc50_19.3: init %empty_tuple.type = call %SelfDestruct.specific_fn.3(%var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestIterate(constants.%R) {
// CHECK:STDOUT:   %R.patt.loc32_17.2 => constants.%R.patt
// CHECK:STDOUT:   %R.loc32_17.1 => constants.%R
// CHECK:STDOUT:   %R.as_type.loc32_61.1 => constants.%R.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.c3b
// CHECK:STDOUT:   %r.param_patt.loc32_59.2 => constants.%r.param_patt.304
// CHECK:STDOUT:   %r.patt.loc32_59.2 => constants.%r.patt.240
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestIterate(constants.%facet_value.463) {
// CHECK:STDOUT:   %R.patt.loc32_17.2 => constants.%R.patt
// CHECK:STDOUT:   %R.loc32_17.1 => constants.%facet_value.463
// CHECK:STDOUT:   %R.as_type.loc32_61.1 => constants.%MutableRange
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.992
// CHECK:STDOUT:   %r.param_patt.loc32_59.2 => constants.%r.param_patt.aeb
// CHECK:STDOUT:   %r.patt.loc32_59.2 => constants.%r.patt.892
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc32 => constants.%complete_type.357
// CHECK:STDOUT:   %Iterate.lookup_impl_witness => constants.%Iterate.impl_witness.752
// CHECK:STDOUT:   %Iterate.facet.loc36_19.5 => constants.%Iterate.facet.c03
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type => constants.%Iterate.WithSelf.NewCursor.type.b9c
// CHECK:STDOUT:   %.loc36_19.23 => constants.%.d6e
// CHECK:STDOUT:   %impl.elem2.loc36_19.2 => constants.%T.as_type.as.Iterate.impl.NewCursor.482
// CHECK:STDOUT:   %specific_impl_fn.loc36_19.3 => constants.%T.as_type.as.Iterate.impl.NewCursor.specific_fn
// CHECK:STDOUT:   %impl.elem1 => constants.%Destroy.facet.5f38
// CHECK:STDOUT:   %as_type => constants.%tuple.type.dd0
// CHECK:STDOUT:   %require_complete.1 => constants.%complete_type.843
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type => constants.%Iterate.WithSelf.Next.type.692
// CHECK:STDOUT:   %.loc36_19.24 => constants.%.446
// CHECK:STDOUT:   %impl.elem3.loc36_19.2 => constants.%T.as_type.as.Iterate.impl.Next.759
// CHECK:STDOUT:   %specific_impl_fn.loc36_19.4 => constants.%T.as_type.as.Iterate.impl.Next.specific_fn
// CHECK:STDOUT:   %SelfDestruct.specific_fn.5 => constants.%SelfDestruct.specific_fn.e1d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- adl_returns_different_types.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %Iterate.type: type = facet_type <@Iterate> [concrete]
// CHECK:STDOUT:   %OptionalStorage.type: type = facet_type <@OptionalStorage> [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %facet_type.bd0: type = facet_type <@Copy & @Destroy> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %T.220e40.1: %OptionalStorage.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Optional.Get.type.786: type = fn_type @Optional.Get, @Optional(%T.220e40.1) [symbolic]
// CHECK:STDOUT:   %Optional.Get.2d0: %Optional.Get.type.786 = struct_value () [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.type.3ea: type = fn_type @Optional.HasValue, @Optional(%T.220e40.1) [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.f2f: %Optional.HasValue.type.3ea = struct_value () [symbolic]
// CHECK:STDOUT:   %T.f84392.1: %facet_type.bd0 = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.476: type = class_type @DefaultOptionalStorage, @DefaultOptionalStorage(%T.f84392.1) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Get.type.f27: type = fn_type @T.as_type.as.OptionalStorage.impl.Get, @T.as_type.as.OptionalStorage.impl(%T.f84392.1) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Get.e40: %T.as_type.as.OptionalStorage.impl.Get.type.f27 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Has.type.18e: type = fn_type @T.as_type.as.OptionalStorage.impl.Has, @T.as_type.as.OptionalStorage.impl(%T.f84392.1) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Has.463: %T.as_type.as.OptionalStorage.impl.Has.type.18e = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Some.type.292: type = fn_type @T.as_type.as.OptionalStorage.impl.Some, @T.as_type.as.OptionalStorage.impl(%T.f84392.1) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Some.ae9: %T.as_type.as.OptionalStorage.impl.Some.type.292 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.None.type.b57: type = fn_type @T.as_type.as.OptionalStorage.impl.None, @T.as_type.as.OptionalStorage.impl(%T.f84392.1) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.None.791: %T.as_type.as.OptionalStorage.impl.None.type.b57 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SelfDestruct.type: type = fn_type @SelfDestruct [concrete]
// CHECK:STDOUT:   %SelfDestruct: %SelfDestruct.type = struct_value () [concrete]
// CHECK:STDOUT:   %.Self.500: %Iterate.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %CppRange.type: type = facet_type <@CppRange> [concrete]
// CHECK:STDOUT:   %.Self.e0b: %CppRange.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %CppRangeForIterate.lookup_impl_witness.50b: <witness> = lookup_impl_witness %.Self.e0b, @CppRangeForIterate [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.32d3: type = impl_witness_access %CppRangeForIterate.lookup_impl_witness.50b, element0 [symbolic_self]
// CHECK:STDOUT:   %CppUnsafeDeref.lookup_impl_witness.851: <witness> = lookup_impl_witness %impl.elem0.32d3, @CppUnsafeDeref [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.10d: type = impl_witness_access %CppUnsafeDeref.lookup_impl_witness.851, element0 [symbolic_self]
// CHECK:STDOUT:   %CppRange_where.type.d682d6.1: type = facet_type <@CppRange where %impl.elem0.10d impls @Copy> [concrete]
// CHECK:STDOUT:   %T.f45530.3: %CppRange_where.type.d682d6.1 = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.type.d4a: type = fn_type @T.as_type.as.Iterate.impl.Next, @T.as_type.as.Iterate.impl(%T.f45530.3) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.5a8: %T.as_type.as.Iterate.impl.Next.type.d4a = struct_value () [symbolic]
// CHECK:STDOUT:   %CppRangeForIterate.lookup_impl_witness.b581a8.2: <witness> = lookup_impl_witness %T.f45530.3, @CppRangeForIterate [symbolic]
// CHECK:STDOUT:   %impl.elem0.5f971c.2: type = impl_witness_access %CppRangeForIterate.lookup_impl_witness.b581a8.2, element0 [symbolic]
// CHECK:STDOUT:   %CppUnsafeDeref.lookup_impl_witness.87e543.2: <witness> = lookup_impl_witness %impl.elem0.5f971c.2, @CppUnsafeDeref [symbolic]
// CHECK:STDOUT:   %impl.elem0.493765.2: type = impl_witness_access %CppUnsafeDeref.lookup_impl_witness.87e543.2, element0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.type.8fb: type = fn_type @T.as_type.as.Iterate.impl.NewCursor, @T.as_type.as.Iterate.impl(%T.f45530.3) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.8a1: %T.as_type.as.Iterate.impl.NewCursor.type.8fb = struct_value () [symbolic]
// CHECK:STDOUT:   %impl.elem1.c2dc15.2: type = impl_witness_access %CppRangeForIterate.lookup_impl_witness.b581a8.2, element1 [symbolic]
// CHECK:STDOUT:   %tuple.type.1ba: type = tuple_type (%impl.elem0.5f971c.2, %impl.elem1.c2dc15.2) [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.lookup_impl_witness.dc9: <witness> = lookup_impl_witness %tuple.type.1ba, @SubobjectDestroy [symbolic]
// CHECK:STDOUT:   %DestroyImpl.lookup_impl_witness.504: <witness> = lookup_impl_witness %tuple.type.1ba, @DestroyImpl [symbolic]
// CHECK:STDOUT:   %Destroy.facet.9b6: %Destroy.type = facet_value %tuple.type.1ba, (%SubobjectDestroy.lookup_impl_witness.dc9, %DestroyImpl.lookup_impl_witness.504) [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.lookup_impl_witness.b15da7.2: <witness> = lookup_impl_witness %impl.elem0.493765.2, @SubobjectDestroy [symbolic]
// CHECK:STDOUT:   %DestroyImpl.lookup_impl_witness.4e7f44.2: <witness> = lookup_impl_witness %impl.elem0.493765.2, @DestroyImpl [symbolic]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.167: <witness> = lookup_impl_witness %impl.elem0.493765.2, @Copy [symbolic]
// CHECK:STDOUT:   %facet_value.189: %facet_type.bd0 = facet_value %impl.elem0.493765.2, (%SubobjectDestroy.lookup_impl_witness.b15da7.2, %DestroyImpl.lookup_impl_witness.4e7f44.2, %Copy.lookup_impl_witness.167) [symbolic]
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
// CHECK:STDOUT:   %facet_value.e01: %facet_type.bd0 = facet_value %ValueType, (%custom_witness.b80ef6.2, %DestroyImpl.impl_witness.0a0, %custom_witness.202) [concrete]
// CHECK:STDOUT:   %Iterate.lookup_impl_witness.1d8: <witness> = lookup_impl_witness %.Self.500, @Iterate [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.4ce: %facet_type.bd0 = impl_witness_access %Iterate.lookup_impl_witness.1d8, element0 [symbolic_self]
// CHECK:STDOUT:   %Iterate_where.type.b87: type = facet_type <@Iterate where %impl.elem0.4ce = %facet_value.e01> [concrete]
// CHECK:STDOUT:   %pattern_type.35b: type = pattern_type %Iterate_where.type.b87 [concrete]
// CHECK:STDOUT:   %R.patt: %pattern_type.35b = symbolic_binding_pattern R, 0 [symbolic]
// CHECK:STDOUT:   %R: %Iterate_where.type.b87 = symbolic_binding R, 0 [symbolic]
// CHECK:STDOUT:   %R.as_type: type = facet_access_type %R [symbolic]
// CHECK:STDOUT:   %pattern_type.1f2: type = pattern_type %R.as_type [symbolic]
// CHECK:STDOUT:   %r.param_patt.c6c: %pattern_type.1f2 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %r.patt.a72: %pattern_type.1f2 = wrapper_binding_pattern r, %r.param_patt.c6c [symbolic]
// CHECK:STDOUT:   %i.patt: %pattern_type.c43 = value_binding_pattern i [concrete]
// CHECK:STDOUT:   %Iterate.lookup_impl_witness.030: <witness> = lookup_impl_witness %R, @Iterate [symbolic]
// CHECK:STDOUT:   %Iterate.facet.6e0: %Iterate.type = facet_value %R.as_type, (%Iterate.lookup_impl_witness.030) [symbolic]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.685: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.6e0) [symbolic]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.67b: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.6e0) [symbolic]
// CHECK:STDOUT:   %.ef0: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.685, %Iterate.facet.6e0 [symbolic]
// CHECK:STDOUT:   %impl.elem2.133: %.ef0 = impl_witness_access %Iterate.lookup_impl_witness.030, element2 [symbolic]
// CHECK:STDOUT:   %impl.elem1.ea0: %Destroy.type = impl_witness_access %Iterate.lookup_impl_witness.030, element1 [symbolic]
// CHECK:STDOUT:   %as_type.947: type = facet_access_type %impl.elem1.ea0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.1db: <specific function> = specific_impl_function %impl.elem2.133, @Iterate.WithSelf.NewCursor(%Iterate.facet.6e0) [symbolic]
// CHECK:STDOUT:   %.c62: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.67b, %Iterate.facet.6e0 [symbolic]
// CHECK:STDOUT:   %impl.elem3.089: %.c62 = impl_witness_access %Iterate.lookup_impl_witness.030, element3 [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.05d: type = class_type @DefaultOptionalStorage, @DefaultOptionalStorage(%facet_value.e01) [concrete]
// CHECK:STDOUT:   %OptionalStorage.impl_witness.77e: <witness> = impl_witness imports.%OptionalStorage.impl_witness_table.1c4, @T.as_type.as.OptionalStorage.impl(%facet_value.e01) [concrete]
// CHECK:STDOUT:   %OptionalStorage.facet.e07: %OptionalStorage.type = facet_value %ValueType, (%OptionalStorage.impl_witness.77e) [concrete]
// CHECK:STDOUT:   %Optional.e69: type = class_type @Optional, @Optional(%OptionalStorage.facet.e07) [concrete]
// CHECK:STDOUT:   %specific_impl_fn.dd5: <specific function> = specific_impl_function %impl.elem3.089, @Iterate.WithSelf.Next(%Iterate.facet.6e0) [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.type.13b: type = fn_type @Optional.HasValue, @Optional(%OptionalStorage.facet.e07) [concrete]
// CHECK:STDOUT:   %Optional.HasValue.7fb: %Optional.HasValue.type.13b = struct_value () [concrete]
// CHECK:STDOUT:   %Optional.Get.type.bc0: type = fn_type @Optional.Get, @Optional(%OptionalStorage.facet.e07) [concrete]
// CHECK:STDOUT:   %Optional.Get.9c2: %Optional.Get.type.bc0 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.facet.88c: %Destroy.type = facet_value %ValueType, (%custom_witness.b80ef6.2, %DestroyImpl.impl_witness.0a0) [concrete]
// CHECK:STDOUT:   %MaybeUnformed.17f: type = class_type @MaybeUnformed, @MaybeUnformed(%Destroy.facet.88c) [concrete]
// CHECK:STDOUT:   %.be1: type = maybe_unformed_type %ValueType [concrete]
// CHECK:STDOUT:   %struct_type.value.has_value.2a6: type = struct_type {.value: %MaybeUnformed.17f, .has_value: bool} [concrete]
// CHECK:STDOUT:   %Optional.HasValue.specific_fn.03c: <specific function> = specific_function %Optional.HasValue.7fb, @Optional.HasValue(%OptionalStorage.facet.e07) [concrete]
// CHECK:STDOUT:   %Optional.Get.specific_fn.a36: <specific function> = specific_function %Optional.Get.9c2, @Optional.Get(%OptionalStorage.facet.e07) [concrete]
// CHECK:STDOUT:   %operator_PlusEqual__carbon_thunk.type: type = fn_type @operator_PlusEqual__carbon_thunk [concrete]
// CHECK:STDOUT:   %operator_PlusEqual__carbon_thunk: %operator_PlusEqual__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.8: type = fn_type @SubobjectDestroy.WithSelf.Op.loc31_37.6 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.8: %SubobjectDestroy.WithSelf.Op.type.6870fd.8 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.8: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.8), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.f62: %SubobjectDestroy.type = facet_value %Optional.e69, (%custom_witness.b80ef6.8) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.7ab: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.f62) [concrete]
// CHECK:STDOUT:   %Destroy.facet.7e2: %Destroy.type = facet_value %Optional.e69, (%custom_witness.b80ef6.8, %DestroyImpl.impl_witness.7ab) [concrete]
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
// CHECK:STDOUT:   %Destroy.facet.959: %Destroy.type = facet_value %tuple.type.7a6, (%custom_witness.b80ef6.11, %DestroyImpl.impl_witness.b22) [concrete]
// CHECK:STDOUT:   %Iterate.impl_witness.491: <witness> = impl_witness imports.%Iterate.impl_witness_table.121, @T.as_type.as.Iterate.impl(%facet_value.d10) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.type.fef: type = fn_type @T.as_type.as.Iterate.impl.NewCursor, @T.as_type.as.Iterate.impl(%facet_value.d10) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.886: %T.as_type.as.Iterate.impl.NewCursor.type.fef = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.type.d72: type = fn_type @T.as_type.as.Iterate.impl.Next, @T.as_type.as.Iterate.impl(%facet_value.d10) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.2ea: %T.as_type.as.Iterate.impl.Next.type.d72 = struct_value () [concrete]
// CHECK:STDOUT:   %facet_value.18e: %Iterate_where.type.b87 = facet_value %ConstRange, (%Iterate.impl_witness.491) [concrete]
// CHECK:STDOUT:   %r.param_patt.ac5: %pattern_type.c1d = value_param_pattern [concrete]
// CHECK:STDOUT:   %r.patt.97f: %pattern_type.c1d = wrapper_binding_pattern r, %r.param_patt.ac5 [concrete]
// CHECK:STDOUT:   %Iterate.facet.7a3: %Iterate.type = facet_value %ConstRange, (%Iterate.impl_witness.491) [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.1d5: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.7a3) [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.362: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.7a3) [concrete]
// CHECK:STDOUT:   %.83a: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.1d5, %Iterate.facet.7a3 [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.specific_fn: <specific function> = specific_function %T.as_type.as.Iterate.impl.NewCursor.886, @T.as_type.as.Iterate.impl.NewCursor(%facet_value.d10) [concrete]
// CHECK:STDOUT:   %.18a: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.362, %Iterate.facet.7a3 [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.specific_fn: <specific function> = specific_function %T.as_type.as.Iterate.impl.Next.2ea, @T.as_type.as.Iterate.impl.Next(%facet_value.d10) [concrete]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.6b7: <specific function> = specific_function %SelfDestruct, @SelfDestruct(%Destroy.facet.959) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .DiffADL = %DiffADL
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.481: type = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%DefaultOptionalStorage (constants.%DefaultOptionalStorage.476)]
// CHECK:STDOUT:   %Core.import_ref.66d: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.None.type (%T.as_type.as.OptionalStorage.impl.None.type.b57) = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.None (constants.%T.as_type.as.OptionalStorage.impl.None.791)]
// CHECK:STDOUT:   %Core.import_ref.32c: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Some.type (%T.as_type.as.OptionalStorage.impl.Some.type.292) = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Some (constants.%T.as_type.as.OptionalStorage.impl.Some.ae9)]
// CHECK:STDOUT:   %Core.import_ref.801: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Has.type (%T.as_type.as.OptionalStorage.impl.Has.type.18e) = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Has (constants.%T.as_type.as.OptionalStorage.impl.Has.463)]
// CHECK:STDOUT:   %Core.import_ref.eec: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Get.type (%T.as_type.as.OptionalStorage.impl.Get.type.f27) = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Get (constants.%T.as_type.as.OptionalStorage.impl.Get.e40)]
// CHECK:STDOUT:   %Core.import_ref.a6a = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %OptionalStorage.impl_witness_table.1c4 = impl_witness_table (%Core.import_ref.481, %Core.import_ref.66d, %Core.import_ref.32c, %Core.import_ref.801, %Core.import_ref.eec, %Core.import_ref.a6a), @T.as_type.as.OptionalStorage.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.7b5: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/types/optional, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.7b5), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.c82: @Optional.%Optional.HasValue.type (%Optional.HasValue.type.3ea) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.HasValue (constants.%Optional.HasValue.f2f)]
// CHECK:STDOUT:   %Core.import_ref.29e: @Optional.%Optional.Get.type (%Optional.Get.type.786) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.Get (constants.%Optional.Get.2d0)]
// CHECK:STDOUT:   %Core.import_ref.724: %facet_type.bd0 = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.Iterate.impl.%facet_value (constants.%facet_value.189)]
// CHECK:STDOUT:   %Core.import_ref.dd9: %Destroy.type = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.Iterate.impl.%Destroy.facet (constants.%Destroy.facet.9b6)]
// CHECK:STDOUT:   %Core.import_ref.946: @T.as_type.as.Iterate.impl.%T.as_type.as.Iterate.impl.NewCursor.type (%T.as_type.as.Iterate.impl.NewCursor.type.8fb) = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.Iterate.impl.%T.as_type.as.Iterate.impl.NewCursor (constants.%T.as_type.as.Iterate.impl.NewCursor.8a1)]
// CHECK:STDOUT:   %Core.import_ref.cfb: @T.as_type.as.Iterate.impl.%T.as_type.as.Iterate.impl.Next.type (%T.as_type.as.Iterate.impl.Next.type.d4a) = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.Iterate.impl.%T.as_type.as.Iterate.impl.Next (constants.%T.as_type.as.Iterate.impl.Next.5a8)]
// CHECK:STDOUT:   %Iterate.impl_witness_table.121 = impl_witness_table (%Core.import_ref.724, %Core.import_ref.dd9, %Core.import_ref.946, %Core.import_ref.cfb), @T.as_type.as.Iterate.impl [concrete]
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
// CHECK:STDOUT: generic fn @TestIterate(%R.loc27_17.2: %Iterate_where.type.b87) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Iterate.lookup_impl_witness: <witness> = lookup_impl_witness %R.loc27_17.1, @Iterate [symbolic = %Iterate.lookup_impl_witness (constants.%Iterate.lookup_impl_witness.030)]
// CHECK:STDOUT:   %Iterate.facet.loc31_37.5: %Iterate.type = facet_value %R.as_type.loc27_79.1, (%Iterate.lookup_impl_witness) [symbolic = %Iterate.facet.loc31_37.5 (constants.%Iterate.facet.6e0)]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.loc31_37.5) [symbolic = %Iterate.WithSelf.NewCursor.type (constants.%Iterate.WithSelf.NewCursor.type.685)]
// CHECK:STDOUT:   %.loc31_37.26: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type, %Iterate.facet.loc31_37.5 [symbolic = %.loc31_37.26 (constants.%.ef0)]
// CHECK:STDOUT:   %impl.elem2.loc31_37.2: @TestIterate.%.loc31_37.26 (%.ef0) = impl_witness_access %Iterate.lookup_impl_witness, element2 [symbolic = %impl.elem2.loc31_37.2 (constants.%impl.elem2.133)]
// CHECK:STDOUT:   %specific_impl_fn.loc31_37.3: <specific function> = specific_impl_function %impl.elem2.loc31_37.2, @Iterate.WithSelf.NewCursor(%Iterate.facet.loc31_37.5) [symbolic = %specific_impl_fn.loc31_37.3 (constants.%specific_impl_fn.1db)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.loc31_37.5) [symbolic = %Iterate.WithSelf.Next.type (constants.%Iterate.WithSelf.Next.type.67b)]
// CHECK:STDOUT:   %.loc31_37.27: type = fn_type_with_self_type %Iterate.WithSelf.Next.type, %Iterate.facet.loc31_37.5 [symbolic = %.loc31_37.27 (constants.%.c62)]
// CHECK:STDOUT:   %impl.elem3.loc31_37.2: @TestIterate.%.loc31_37.27 (%.c62) = impl_witness_access %Iterate.lookup_impl_witness, element3 [symbolic = %impl.elem3.loc31_37.2 (constants.%impl.elem3.089)]
// CHECK:STDOUT:   %specific_impl_fn.loc31_37.4: <specific function> = specific_impl_function %impl.elem3.loc31_37.2, @Iterate.WithSelf.Next(%Iterate.facet.loc31_37.5) [symbolic = %specific_impl_fn.loc31_37.4 (constants.%specific_impl_fn.dd5)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%r.param: @TestIterate.%R.as_type.loc27_79.1 (%R.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %i.patt: %pattern_type.c43 = value_binding_pattern i [concrete = constants.%i.patt]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %r.ref: @TestIterate.%R.as_type.loc27_79.1 (%R.as_type) = name_ref r, %r
// CHECK:STDOUT:     %impl.elem2.loc31_37.1: @TestIterate.%.loc31_37.26 (%.ef0) = impl_witness_access constants.%Iterate.lookup_impl_witness.030, element2 [symbolic = %impl.elem2.loc31_37.2 (constants.%impl.elem2.133)]
// CHECK:STDOUT:     %bound_method.loc31_37.1: <bound method> = bound_method %r.ref, %impl.elem2.loc31_37.1
// CHECK:STDOUT:     %Iterate.facet.loc31_37.1: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.030) [symbolic = %Iterate.facet.loc31_37.5 (constants.%Iterate.facet.6e0)]
// CHECK:STDOUT:     %.loc31_37.1: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc31_37.1 [symbolic = %Iterate.facet.loc31_37.5 (constants.%Iterate.facet.6e0)]
// CHECK:STDOUT:     %Iterate.facet.loc31_37.2: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.030) [symbolic = %Iterate.facet.loc31_37.5 (constants.%Iterate.facet.6e0)]
// CHECK:STDOUT:     %.loc31_37.2: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc31_37.2 [symbolic = %Iterate.facet.loc31_37.5 (constants.%Iterate.facet.6e0)]
// CHECK:STDOUT:     %specific_impl_fn.loc31_37.1: <specific function> = specific_impl_function %impl.elem2.loc31_37.1, @Iterate.WithSelf.NewCursor(constants.%Iterate.facet.6e0) [symbolic = %specific_impl_fn.loc31_37.3 (constants.%specific_impl_fn.1db)]
// CHECK:STDOUT:     %bound_method.loc31_37.2: <bound method> = bound_method %r.ref, %specific_impl_fn.loc31_37.1
// CHECK:STDOUT:     %var: ref @TestIterate.%as_type (%as_type.947) = var_storage invalid
// CHECK:STDOUT:     %Iterate.WithSelf.NewCursor.call: init @TestIterate.%as_type (%as_type.947) to %var = call %bound_method.loc31_37.2(%r.ref)
// CHECK:STDOUT:     assign %var, %Iterate.WithSelf.NewCursor.call
// CHECK:STDOUT:     br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.next:
// CHECK:STDOUT:     %impl.elem3.loc31_37.1: @TestIterate.%.loc31_37.27 (%.c62) = impl_witness_access constants.%Iterate.lookup_impl_witness.030, element3 [symbolic = %impl.elem3.loc31_37.2 (constants.%impl.elem3.089)]
// CHECK:STDOUT:     %bound_method.loc31_37.3: <bound method> = bound_method %r.ref, %impl.elem3.loc31_37.1
// CHECK:STDOUT:     %Iterate.facet.loc31_37.3: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.030) [symbolic = %Iterate.facet.loc31_37.5 (constants.%Iterate.facet.6e0)]
// CHECK:STDOUT:     %.loc31_37.3: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc31_37.3 [symbolic = %Iterate.facet.loc31_37.5 (constants.%Iterate.facet.6e0)]
// CHECK:STDOUT:     %Iterate.facet.loc31_37.4: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.030) [symbolic = %Iterate.facet.loc31_37.5 (constants.%Iterate.facet.6e0)]
// CHECK:STDOUT:     %.loc31_37.4: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc31_37.4 [symbolic = %Iterate.facet.loc31_37.5 (constants.%Iterate.facet.6e0)]
// CHECK:STDOUT:     %.loc31_37.5: %Destroy.type = converted constants.%as_type.947, constants.%impl.elem1.ea0 [symbolic = %impl.elem1 (constants.%impl.elem1.ea0)]
// CHECK:STDOUT:     %.loc31_37.6: %Destroy.type = converted constants.%as_type.947, constants.%impl.elem1.ea0 [symbolic = %impl.elem1 (constants.%impl.elem1.ea0)]
// CHECK:STDOUT:     %specific_impl_fn.loc31_37.2: <specific function> = specific_impl_function %impl.elem3.loc31_37.1, @Iterate.WithSelf.Next(constants.%Iterate.facet.6e0) [symbolic = %specific_impl_fn.loc31_37.4 (constants.%specific_impl_fn.dd5)]
// CHECK:STDOUT:     %bound_method.loc31_37.4: <bound method> = bound_method %r.ref, %specific_impl_fn.loc31_37.2
// CHECK:STDOUT:     %.loc31_37.7: ref %Optional.e69 = temporary_storage
// CHECK:STDOUT:     %Iterate.WithSelf.Next.call: init %Optional.e69 to %.loc31_37.7 = call %bound_method.loc31_37.4(%r.ref, %var)
// CHECK:STDOUT:     %.loc31_37.8: ref %Optional.e69 = temporary %.loc31_37.7, %Iterate.WithSelf.Next.call
// CHECK:STDOUT:     %.loc31_37.9: %Optional.HasValue.type.13b = specific_constant imports.%Core.import_ref.c82, @Optional(constants.%OptionalStorage.facet.e07) [concrete = constants.%Optional.HasValue.7fb]
// CHECK:STDOUT:     %HasValue.ref: %Optional.HasValue.type.13b = name_ref HasValue, %.loc31_37.9 [concrete = constants.%Optional.HasValue.7fb]
// CHECK:STDOUT:     %Optional.HasValue.bound: <bound method> = bound_method %.loc31_37.8, %HasValue.ref
// CHECK:STDOUT:     %Optional.HasValue.specific_fn: <specific function> = specific_function %HasValue.ref, @Optional.HasValue(constants.%OptionalStorage.facet.e07) [concrete = constants.%Optional.HasValue.specific_fn.03c]
// CHECK:STDOUT:     %bound_method.loc31_37.5: <bound method> = bound_method %.loc31_37.8, %Optional.HasValue.specific_fn
// CHECK:STDOUT:     %.loc31_37.10: %Optional.e69 = acquire_value %.loc31_37.8
// CHECK:STDOUT:     %Optional.HasValue.call: init bool = call %bound_method.loc31_37.5(%.loc31_37.10)
// CHECK:STDOUT:     %.loc31_37.11: bool = value_of_initializer %Optional.HasValue.call
// CHECK:STDOUT:     %.loc31_37.12: bool = converted %Optional.HasValue.call, %.loc31_37.11
// CHECK:STDOUT:     if %.loc31_37.12 br !for.body else br !for.done
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.body:
// CHECK:STDOUT:     %.loc31_37.13: %Optional.Get.type.bc0 = specific_constant imports.%Core.import_ref.29e, @Optional(constants.%OptionalStorage.facet.e07) [concrete = constants.%Optional.Get.9c2]
// CHECK:STDOUT:     %Get.ref: %Optional.Get.type.bc0 = name_ref Get, %.loc31_37.13 [concrete = constants.%Optional.Get.9c2]
// CHECK:STDOUT:     %Optional.Get.bound: <bound method> = bound_method %.loc31_37.8, %Get.ref
// CHECK:STDOUT:     %Optional.Get.specific_fn: <specific function> = specific_function %Get.ref, @Optional.Get(constants.%OptionalStorage.facet.e07) [concrete = constants.%Optional.Get.specific_fn.a36]
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
// CHECK:STDOUT:     %Destroy.facet.loc31_37.1: %Destroy.type = facet_value constants.%ValueType, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.0a0) [concrete = constants.%Destroy.facet.88c]
// CHECK:STDOUT:     %.loc31_37.18: %Destroy.type = converted constants.%ValueType, %Destroy.facet.loc31_37.1 [concrete = constants.%Destroy.facet.88c]
// CHECK:STDOUT:     %Destroy.facet.loc31_37.2: %Destroy.type = facet_value constants.%ValueType, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.0a0) [concrete = constants.%Destroy.facet.88c]
// CHECK:STDOUT:     %.loc31_37.19: %Destroy.type = converted constants.%ValueType, %Destroy.facet.loc31_37.2 [concrete = constants.%Destroy.facet.88c]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %SelfDestruct.call.loc31_37.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc31_37.16)
// CHECK:STDOUT:     %Destroy.facet.loc31_37.3: %Destroy.type = facet_value constants.%Optional.e69, (constants.%custom_witness.b80ef6.8, constants.%DestroyImpl.impl_witness.7ab) [concrete = constants.%Destroy.facet.7e2]
// CHECK:STDOUT:     %.loc31_37.20: %Destroy.type = converted constants.%Optional.e69, %Destroy.facet.loc31_37.3 [concrete = constants.%Destroy.facet.7e2]
// CHECK:STDOUT:     %Destroy.facet.loc31_37.4: %Destroy.type = facet_value constants.%Optional.e69, (constants.%custom_witness.b80ef6.8, constants.%DestroyImpl.impl_witness.7ab) [concrete = constants.%Destroy.facet.7e2]
// CHECK:STDOUT:     %.loc31_37.21: %Destroy.type = converted constants.%Optional.e69, %Destroy.facet.loc31_37.4 [concrete = constants.%Destroy.facet.7e2]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %SelfDestruct.call.loc31_37.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc31_37.8)
// CHECK:STDOUT:     br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.done:
// CHECK:STDOUT:     %Destroy.facet.loc31_37.5: %Destroy.type = facet_value constants.%Optional.e69, (constants.%custom_witness.b80ef6.8, constants.%DestroyImpl.impl_witness.7ab) [concrete = constants.%Destroy.facet.7e2]
// CHECK:STDOUT:     %.loc31_37.22: %Destroy.type = converted constants.%Optional.e69, %Destroy.facet.loc31_37.5 [concrete = constants.%Destroy.facet.7e2]
// CHECK:STDOUT:     %Destroy.facet.loc31_37.6: %Destroy.type = facet_value constants.%Optional.e69, (constants.%custom_witness.b80ef6.8, constants.%DestroyImpl.impl_witness.7ab) [concrete = constants.%Destroy.facet.7e2]
// CHECK:STDOUT:     %.loc31_37.23: %Destroy.type = converted constants.%Optional.e69, %Destroy.facet.loc31_37.6 [concrete = constants.%Destroy.facet.7e2]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %SelfDestruct.call.loc31_37.3: init %empty_tuple.type = call %SelfDestruct.specific_fn.3(%.loc31_37.8)
// CHECK:STDOUT:     %.loc31_37.24: %Destroy.type = converted constants.%as_type.947, constants.%impl.elem1.ea0 [symbolic = %impl.elem1 (constants.%impl.elem1.ea0)]
// CHECK:STDOUT:     %.loc31_37.25: %Destroy.type = converted constants.%as_type.947, constants.%impl.elem1.ea0 [symbolic = %impl.elem1 (constants.%impl.elem1.ea0)]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %SelfDestruct.call.loc31_37.4: init %empty_tuple.type = call %SelfDestruct.specific_fn.4(%var)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc31_37.1(%self.param: ref %.be1) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc31_37.2(%self.param: ref %MaybeUnformed.17f) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc31_37.3(%self.param: ref bool) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc31_37.4(%self.param: ref %struct_type.value.has_value.2a6) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc31_37.5(%self.param: ref %DefaultOptionalStorage.05d) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc31_37.6(%self.param: ref %Optional.e69) = "no_op";
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
// CHECK:STDOUT:   %.loc45_37.5: ref %Optional.e69 = temporary_storage
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.call: init %Optional.e69 to %.loc45_37.5 = call %bound_method.loc45_37.4(%c.ref, %var)
// CHECK:STDOUT:   %.loc45_37.6: ref %Optional.e69 = temporary %.loc45_37.5, %T.as_type.as.Iterate.impl.Next.call
// CHECK:STDOUT:   %.loc45_37.7: %Optional.HasValue.type.13b = specific_constant imports.%Core.import_ref.c82, @Optional(constants.%OptionalStorage.facet.e07) [concrete = constants.%Optional.HasValue.7fb]
// CHECK:STDOUT:   %HasValue.ref: %Optional.HasValue.type.13b = name_ref HasValue, %.loc45_37.7 [concrete = constants.%Optional.HasValue.7fb]
// CHECK:STDOUT:   %Optional.HasValue.bound: <bound method> = bound_method %.loc45_37.6, %HasValue.ref
// CHECK:STDOUT:   %Optional.HasValue.specific_fn: <specific function> = specific_function %HasValue.ref, @Optional.HasValue(constants.%OptionalStorage.facet.e07) [concrete = constants.%Optional.HasValue.specific_fn.03c]
// CHECK:STDOUT:   %bound_method.loc45_37.5: <bound method> = bound_method %.loc45_37.6, %Optional.HasValue.specific_fn
// CHECK:STDOUT:   %.loc45_37.8: %Optional.e69 = acquire_value %.loc45_37.6
// CHECK:STDOUT:   %Optional.HasValue.call: init bool = call %bound_method.loc45_37.5(%.loc45_37.8)
// CHECK:STDOUT:   %.loc45_37.9: bool = value_of_initializer %Optional.HasValue.call
// CHECK:STDOUT:   %.loc45_37.10: bool = converted %Optional.HasValue.call, %.loc45_37.9
// CHECK:STDOUT:   if %.loc45_37.10 br !for.body else br !for.done
// CHECK:STDOUT:
// CHECK:STDOUT: !for.body:
// CHECK:STDOUT:   %.loc45_37.11: %Optional.Get.type.bc0 = specific_constant imports.%Core.import_ref.29e, @Optional(constants.%OptionalStorage.facet.e07) [concrete = constants.%Optional.Get.9c2]
// CHECK:STDOUT:   %Get.ref: %Optional.Get.type.bc0 = name_ref Get, %.loc45_37.11 [concrete = constants.%Optional.Get.9c2]
// CHECK:STDOUT:   %Optional.Get.bound: <bound method> = bound_method %.loc45_37.6, %Get.ref
// CHECK:STDOUT:   %Optional.Get.specific_fn: <specific function> = specific_function %Get.ref, @Optional.Get(constants.%OptionalStorage.facet.e07) [concrete = constants.%Optional.Get.specific_fn.a36]
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
// CHECK:STDOUT:   %Destroy.facet.loc45_37.1: %Destroy.type = facet_value constants.%ValueType, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.0a0) [concrete = constants.%Destroy.facet.88c]
// CHECK:STDOUT:   %.loc45_37.16: %Destroy.type = converted constants.%ValueType, %Destroy.facet.loc45_37.1 [concrete = constants.%Destroy.facet.88c]
// CHECK:STDOUT:   %Destroy.facet.loc45_37.2: %Destroy.type = facet_value constants.%ValueType, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.0a0) [concrete = constants.%Destroy.facet.88c]
// CHECK:STDOUT:   %.loc45_37.17: %Destroy.type = converted constants.%ValueType, %Destroy.facet.loc45_37.2 [concrete = constants.%Destroy.facet.88c]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc45_37.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc45_37.14)
// CHECK:STDOUT:   %Destroy.facet.loc45_37.3: %Destroy.type = facet_value constants.%Optional.e69, (constants.%custom_witness.b80ef6.8, constants.%DestroyImpl.impl_witness.7ab) [concrete = constants.%Destroy.facet.7e2]
// CHECK:STDOUT:   %.loc45_37.18: %Destroy.type = converted constants.%Optional.e69, %Destroy.facet.loc45_37.3 [concrete = constants.%Destroy.facet.7e2]
// CHECK:STDOUT:   %Destroy.facet.loc45_37.4: %Destroy.type = facet_value constants.%Optional.e69, (constants.%custom_witness.b80ef6.8, constants.%DestroyImpl.impl_witness.7ab) [concrete = constants.%Destroy.facet.7e2]
// CHECK:STDOUT:   %.loc45_37.19: %Destroy.type = converted constants.%Optional.e69, %Destroy.facet.loc45_37.4 [concrete = constants.%Destroy.facet.7e2]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc45_37.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc45_37.6)
// CHECK:STDOUT:   br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT: !for.done:
// CHECK:STDOUT:   %Destroy.facet.loc45_37.5: %Destroy.type = facet_value constants.%Optional.e69, (constants.%custom_witness.b80ef6.8, constants.%DestroyImpl.impl_witness.7ab) [concrete = constants.%Destroy.facet.7e2]
// CHECK:STDOUT:   %.loc45_37.20: %Destroy.type = converted constants.%Optional.e69, %Destroy.facet.loc45_37.5 [concrete = constants.%Destroy.facet.7e2]
// CHECK:STDOUT:   %Destroy.facet.loc45_37.6: %Destroy.type = facet_value constants.%Optional.e69, (constants.%custom_witness.b80ef6.8, constants.%DestroyImpl.impl_witness.7ab) [concrete = constants.%Destroy.facet.7e2]
// CHECK:STDOUT:   %.loc45_37.21: %Destroy.type = converted constants.%Optional.e69, %Destroy.facet.loc45_37.6 [concrete = constants.%Destroy.facet.7e2]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc45_37.3: init %empty_tuple.type = call %SelfDestruct.specific_fn.3(%.loc45_37.6)
// CHECK:STDOUT:   %Destroy.facet.loc45_37.7: %Destroy.type = facet_value constants.%tuple.type.7a6, (constants.%custom_witness.b80ef6.11, constants.%DestroyImpl.impl_witness.b22) [concrete = constants.%Destroy.facet.959]
// CHECK:STDOUT:   %.loc45_37.22: %Destroy.type = converted constants.%tuple.type.7a6, %Destroy.facet.loc45_37.7 [concrete = constants.%Destroy.facet.959]
// CHECK:STDOUT:   %Destroy.facet.loc45_37.8: %Destroy.type = facet_value constants.%tuple.type.7a6, (constants.%custom_witness.b80ef6.11, constants.%DestroyImpl.impl_witness.b22) [concrete = constants.%Destroy.facet.959]
// CHECK:STDOUT:   %.loc45_37.23: %Destroy.type = converted constants.%tuple.type.7a6, %Destroy.facet.loc45_37.8 [concrete = constants.%Destroy.facet.959]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc45_37.4: init %empty_tuple.type = call %SelfDestruct.specific_fn.4(%var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestIterate(constants.%R) {
// CHECK:STDOUT:   %R.patt.loc27_17.2 => constants.%R.patt
// CHECK:STDOUT:   %R.loc27_17.1 => constants.%R
// CHECK:STDOUT:   %R.as_type.loc27_79.1 => constants.%R.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.1f2
// CHECK:STDOUT:   %r.param_patt.loc27_77.2 => constants.%r.param_patt.c6c
// CHECK:STDOUT:   %r.patt.loc27_77.2 => constants.%r.patt.a72
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestIterate(constants.%facet_value.18e) {
// CHECK:STDOUT:   %R.patt.loc27_17.2 => constants.%R.patt
// CHECK:STDOUT:   %R.loc27_17.1 => constants.%facet_value.18e
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
// CHECK:STDOUT:   %impl.elem1 => constants.%Destroy.facet.959
// CHECK:STDOUT:   %as_type => constants.%tuple.type.7a6
// CHECK:STDOUT:   %require_complete.1 => constants.%complete_type.cf0
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type => constants.%Iterate.WithSelf.Next.type.362
// CHECK:STDOUT:   %.loc31_37.27 => constants.%.18a
// CHECK:STDOUT:   %impl.elem3.loc31_37.2 => constants.%T.as_type.as.Iterate.impl.Next.2ea
// CHECK:STDOUT:   %specific_impl_fn.loc31_37.4 => constants.%T.as_type.as.Iterate.impl.Next.specific_fn
// CHECK:STDOUT:   %SelfDestruct.specific_fn.6 => constants.%SelfDestruct.specific_fn.6b7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_adl_returns_different_types_mutable.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %Iterate.type: type = facet_type <@Iterate> [concrete]
// CHECK:STDOUT:   %OptionalStorage.type: type = facet_type <@OptionalStorage> [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %facet_type.bd0: type = facet_type <@Copy & @Destroy> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %T.220: %OptionalStorage.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Optional.Get.type.786: type = fn_type @Optional.Get, @Optional(%T.220) [symbolic]
// CHECK:STDOUT:   %Optional.Get.2d0: %Optional.Get.type.786 = struct_value () [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.type.3ea: type = fn_type @Optional.HasValue, @Optional(%T.220) [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.f2f: %Optional.HasValue.type.3ea = struct_value () [symbolic]
// CHECK:STDOUT:   %T.f84392.1: %facet_type.bd0 = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.476: type = class_type @DefaultOptionalStorage, @DefaultOptionalStorage(%T.f84392.1) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Get.type.f27: type = fn_type @T.as_type.as.OptionalStorage.impl.Get, @T.as_type.as.OptionalStorage.impl(%T.f84392.1) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Get.e40: %T.as_type.as.OptionalStorage.impl.Get.type.f27 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Has.type.18e: type = fn_type @T.as_type.as.OptionalStorage.impl.Has, @T.as_type.as.OptionalStorage.impl(%T.f84392.1) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Has.463: %T.as_type.as.OptionalStorage.impl.Has.type.18e = struct_value () [symbolic]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %.Self.500: %Iterate.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %ValueType: type = class_type @ValueType [concrete]
// CHECK:STDOUT:   %pattern_type.d2d: type = pattern_type %ValueType [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.2: type = fn_type @SubobjectDestroy.WithSelf.Op.loc37 [concrete]
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
// CHECK:STDOUT:   %facet_value.f82: %facet_type.bd0 = facet_value %ValueType, (%custom_witness.b80ef6.2, %DestroyImpl.impl_witness.127, %custom_witness.587) [concrete]
// CHECK:STDOUT:   %Iterate.lookup_impl_witness.1d8: <witness> = lookup_impl_witness %.Self.500, @Iterate [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.4ce: %facet_type.bd0 = impl_witness_access %Iterate.lookup_impl_witness.1d8, element0 [symbolic_self]
// CHECK:STDOUT:   %Iterate_where.type.f3d: type = facet_type <@Iterate where %impl.elem0.4ce = %facet_value.f82> [concrete]
// CHECK:STDOUT:   %pattern_type.c3b: type = pattern_type %Iterate_where.type.f3d [concrete]
// CHECK:STDOUT:   %R.patt: %pattern_type.c3b = symbolic_binding_pattern R, 0 [symbolic]
// CHECK:STDOUT:   %R: %Iterate_where.type.f3d = symbolic_binding R, 0 [symbolic]
// CHECK:STDOUT:   %R.as_type: type = facet_access_type %R [symbolic]
// CHECK:STDOUT:   %pattern_type.bac: type = pattern_type %R.as_type [symbolic]
// CHECK:STDOUT:   %r.param_patt: %pattern_type.bac = value_param_pattern [symbolic]
// CHECK:STDOUT:   %r.patt: %pattern_type.bac = wrapper_binding_pattern r, %r.param_patt [symbolic]
// CHECK:STDOUT:   %i.patt: %pattern_type.d2d = value_binding_pattern i [concrete]
// CHECK:STDOUT:   %Iterate.lookup_impl_witness.774: <witness> = lookup_impl_witness %R, @Iterate [symbolic]
// CHECK:STDOUT:   %Iterate.facet.49e: %Iterate.type = facet_value %R.as_type, (%Iterate.lookup_impl_witness.774) [symbolic]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.104: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.49e) [symbolic]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.05a: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.49e) [symbolic]
// CHECK:STDOUT:   %.00b: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.104, %Iterate.facet.49e [symbolic]
// CHECK:STDOUT:   %impl.elem2.2a9: %.00b = impl_witness_access %Iterate.lookup_impl_witness.774, element2 [symbolic]
// CHECK:STDOUT:   %impl.elem1.983: %Destroy.type = impl_witness_access %Iterate.lookup_impl_witness.774, element1 [symbolic]
// CHECK:STDOUT:   %as_type.7b4: type = facet_access_type %impl.elem1.983 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.cab: <specific function> = specific_impl_function %impl.elem2.2a9, @Iterate.WithSelf.NewCursor(%Iterate.facet.49e) [symbolic]
// CHECK:STDOUT:   %.536: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.05a, %Iterate.facet.49e [symbolic]
// CHECK:STDOUT:   %impl.elem3.e5b: %.536 = impl_witness_access %Iterate.lookup_impl_witness.774, element3 [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.ed6: type = class_type @DefaultOptionalStorage, @DefaultOptionalStorage(%facet_value.f82) [concrete]
// CHECK:STDOUT:   %OptionalStorage.impl_witness.82d: <witness> = impl_witness imports.%OptionalStorage.impl_witness_table.1c4, @T.as_type.as.OptionalStorage.impl(%facet_value.f82) [concrete]
// CHECK:STDOUT:   %OptionalStorage.facet.11f: %OptionalStorage.type = facet_value %ValueType, (%OptionalStorage.impl_witness.82d) [concrete]
// CHECK:STDOUT:   %Optional.5e5: type = class_type @Optional, @Optional(%OptionalStorage.facet.11f) [concrete]
// CHECK:STDOUT:   %specific_impl_fn.f7c: <specific function> = specific_impl_function %impl.elem3.e5b, @Iterate.WithSelf.Next(%Iterate.facet.49e) [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.type.e84: type = fn_type @Optional.HasValue, @Optional(%OptionalStorage.facet.11f) [concrete]
// CHECK:STDOUT:   %Optional.HasValue.cbc: %Optional.HasValue.type.e84 = struct_value () [concrete]
// CHECK:STDOUT:   %Optional.Get.type.e1b: type = fn_type @Optional.Get, @Optional(%OptionalStorage.facet.11f) [concrete]
// CHECK:STDOUT:   %Optional.Get.155: %Optional.Get.type.e1b = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.facet.3c5: %Destroy.type = facet_value %ValueType, (%custom_witness.b80ef6.2, %DestroyImpl.impl_witness.127) [concrete]
// CHECK:STDOUT:   %MaybeUnformed.753: type = class_type @MaybeUnformed, @MaybeUnformed(%Destroy.facet.3c5) [concrete]
// CHECK:STDOUT:   %.dd1: type = maybe_unformed_type %ValueType [concrete]
// CHECK:STDOUT:   %struct_type.value.has_value.9df: type = struct_type {.value: %MaybeUnformed.753, .has_value: bool} [concrete]
// CHECK:STDOUT:   %Optional.HasValue.specific_fn: <specific function> = specific_function %Optional.HasValue.cbc, @Optional.HasValue(%OptionalStorage.facet.11f) [concrete]
// CHECK:STDOUT:   %Optional.Get.specific_fn: <specific function> = specific_function %Optional.Get.155, @Optional.Get(%OptionalStorage.facet.11f) [concrete]
// CHECK:STDOUT:   %operator_PlusEqual__carbon_thunk.type: type = fn_type @operator_PlusEqual__carbon_thunk [concrete]
// CHECK:STDOUT:   %operator_PlusEqual__carbon_thunk: %operator_PlusEqual__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.8: type = fn_type @SubobjectDestroy.WithSelf.Op.loc41_44.6 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.8: %SubobjectDestroy.WithSelf.Op.type.6870fd.8 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.8: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.8), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.cb9: %SubobjectDestroy.type = facet_value %Optional.5e5, (%custom_witness.b80ef6.8) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.587: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.cb9) [concrete]
// CHECK:STDOUT:   %Destroy.facet.328: %Destroy.type = facet_value %Optional.5e5, (%custom_witness.b80ef6.8, %DestroyImpl.impl_witness.587) [concrete]
// CHECK:STDOUT:   %MutableRange: type = class_type @MutableRange [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .DiffMutableADL = %DiffMutableADL
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.481: type = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%DefaultOptionalStorage (constants.%DefaultOptionalStorage.476)]
// CHECK:STDOUT:   %Core.import_ref.138 = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.e46 = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.801: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Has.type (%T.as_type.as.OptionalStorage.impl.Has.type.18e) = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Has (constants.%T.as_type.as.OptionalStorage.impl.Has.463)]
// CHECK:STDOUT:   %Core.import_ref.eec: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Get.type (%T.as_type.as.OptionalStorage.impl.Get.type.f27) = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Get (constants.%T.as_type.as.OptionalStorage.impl.Get.e40)]
// CHECK:STDOUT:   %Core.import_ref.a6a = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %OptionalStorage.impl_witness_table.1c4 = impl_witness_table (%Core.import_ref.481, %Core.import_ref.138, %Core.import_ref.e46, %Core.import_ref.801, %Core.import_ref.eec, %Core.import_ref.a6a), @T.as_type.as.OptionalStorage.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.7b5: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/types/optional, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.7b5), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.c82: @Optional.%Optional.HasValue.type (%Optional.HasValue.type.3ea) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.HasValue (constants.%Optional.HasValue.f2f)]
// CHECK:STDOUT:   %Core.import_ref.29e: @Optional.%Optional.Get.type (%Optional.Get.type.786) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.Get (constants.%Optional.Get.2d0)]
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
// CHECK:STDOUT: generic fn @TestIterate(%R.loc37_17.2: %Iterate_where.type.f3d) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Iterate.lookup_impl_witness: <witness> = lookup_impl_witness %R.loc37_17.1, @Iterate [symbolic = %Iterate.lookup_impl_witness (constants.%Iterate.lookup_impl_witness.774)]
// CHECK:STDOUT:   %Iterate.facet.loc41_44.5: %Iterate.type = facet_value %R.as_type.loc37_86.1, (%Iterate.lookup_impl_witness) [symbolic = %Iterate.facet.loc41_44.5 (constants.%Iterate.facet.49e)]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.loc41_44.5) [symbolic = %Iterate.WithSelf.NewCursor.type (constants.%Iterate.WithSelf.NewCursor.type.104)]
// CHECK:STDOUT:   %.loc41_44.26: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type, %Iterate.facet.loc41_44.5 [symbolic = %.loc41_44.26 (constants.%.00b)]
// CHECK:STDOUT:   %impl.elem2.loc41_44.2: @TestIterate.%.loc41_44.26 (%.00b) = impl_witness_access %Iterate.lookup_impl_witness, element2 [symbolic = %impl.elem2.loc41_44.2 (constants.%impl.elem2.2a9)]
// CHECK:STDOUT:   %specific_impl_fn.loc41_44.3: <specific function> = specific_impl_function %impl.elem2.loc41_44.2, @Iterate.WithSelf.NewCursor(%Iterate.facet.loc41_44.5) [symbolic = %specific_impl_fn.loc41_44.3 (constants.%specific_impl_fn.cab)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.loc41_44.5) [symbolic = %Iterate.WithSelf.Next.type (constants.%Iterate.WithSelf.Next.type.05a)]
// CHECK:STDOUT:   %.loc41_44.27: type = fn_type_with_self_type %Iterate.WithSelf.Next.type, %Iterate.facet.loc41_44.5 [symbolic = %.loc41_44.27 (constants.%.536)]
// CHECK:STDOUT:   %impl.elem3.loc41_44.2: @TestIterate.%.loc41_44.27 (%.536) = impl_witness_access %Iterate.lookup_impl_witness, element3 [symbolic = %impl.elem3.loc41_44.2 (constants.%impl.elem3.e5b)]
// CHECK:STDOUT:   %specific_impl_fn.loc41_44.4: <specific function> = specific_impl_function %impl.elem3.loc41_44.2, @Iterate.WithSelf.Next(%Iterate.facet.loc41_44.5) [symbolic = %specific_impl_fn.loc41_44.4 (constants.%specific_impl_fn.f7c)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%r.param: @TestIterate.%R.as_type.loc37_86.1 (%R.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %i.patt: %pattern_type.d2d = value_binding_pattern i [concrete = constants.%i.patt]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %r.ref: @TestIterate.%R.as_type.loc37_86.1 (%R.as_type) = name_ref r, %r
// CHECK:STDOUT:     %impl.elem2.loc41_44.1: @TestIterate.%.loc41_44.26 (%.00b) = impl_witness_access constants.%Iterate.lookup_impl_witness.774, element2 [symbolic = %impl.elem2.loc41_44.2 (constants.%impl.elem2.2a9)]
// CHECK:STDOUT:     %bound_method.loc41_44.1: <bound method> = bound_method %r.ref, %impl.elem2.loc41_44.1
// CHECK:STDOUT:     %Iterate.facet.loc41_44.1: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.774) [symbolic = %Iterate.facet.loc41_44.5 (constants.%Iterate.facet.49e)]
// CHECK:STDOUT:     %.loc41_44.1: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc41_44.1 [symbolic = %Iterate.facet.loc41_44.5 (constants.%Iterate.facet.49e)]
// CHECK:STDOUT:     %Iterate.facet.loc41_44.2: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.774) [symbolic = %Iterate.facet.loc41_44.5 (constants.%Iterate.facet.49e)]
// CHECK:STDOUT:     %.loc41_44.2: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc41_44.2 [symbolic = %Iterate.facet.loc41_44.5 (constants.%Iterate.facet.49e)]
// CHECK:STDOUT:     %specific_impl_fn.loc41_44.1: <specific function> = specific_impl_function %impl.elem2.loc41_44.1, @Iterate.WithSelf.NewCursor(constants.%Iterate.facet.49e) [symbolic = %specific_impl_fn.loc41_44.3 (constants.%specific_impl_fn.cab)]
// CHECK:STDOUT:     %bound_method.loc41_44.2: <bound method> = bound_method %r.ref, %specific_impl_fn.loc41_44.1
// CHECK:STDOUT:     %var: ref @TestIterate.%as_type (%as_type.7b4) = var_storage invalid
// CHECK:STDOUT:     %Iterate.WithSelf.NewCursor.call: init @TestIterate.%as_type (%as_type.7b4) to %var = call %bound_method.loc41_44.2(%r.ref)
// CHECK:STDOUT:     assign %var, %Iterate.WithSelf.NewCursor.call
// CHECK:STDOUT:     br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.next:
// CHECK:STDOUT:     %impl.elem3.loc41_44.1: @TestIterate.%.loc41_44.27 (%.536) = impl_witness_access constants.%Iterate.lookup_impl_witness.774, element3 [symbolic = %impl.elem3.loc41_44.2 (constants.%impl.elem3.e5b)]
// CHECK:STDOUT:     %bound_method.loc41_44.3: <bound method> = bound_method %r.ref, %impl.elem3.loc41_44.1
// CHECK:STDOUT:     %Iterate.facet.loc41_44.3: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.774) [symbolic = %Iterate.facet.loc41_44.5 (constants.%Iterate.facet.49e)]
// CHECK:STDOUT:     %.loc41_44.3: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc41_44.3 [symbolic = %Iterate.facet.loc41_44.5 (constants.%Iterate.facet.49e)]
// CHECK:STDOUT:     %Iterate.facet.loc41_44.4: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.774) [symbolic = %Iterate.facet.loc41_44.5 (constants.%Iterate.facet.49e)]
// CHECK:STDOUT:     %.loc41_44.4: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc41_44.4 [symbolic = %Iterate.facet.loc41_44.5 (constants.%Iterate.facet.49e)]
// CHECK:STDOUT:     %.loc41_44.5: %Destroy.type = converted constants.%as_type.7b4, constants.%impl.elem1.983 [symbolic = %impl.elem1 (constants.%impl.elem1.983)]
// CHECK:STDOUT:     %.loc41_44.6: %Destroy.type = converted constants.%as_type.7b4, constants.%impl.elem1.983 [symbolic = %impl.elem1 (constants.%impl.elem1.983)]
// CHECK:STDOUT:     %specific_impl_fn.loc41_44.2: <specific function> = specific_impl_function %impl.elem3.loc41_44.1, @Iterate.WithSelf.Next(constants.%Iterate.facet.49e) [symbolic = %specific_impl_fn.loc41_44.4 (constants.%specific_impl_fn.f7c)]
// CHECK:STDOUT:     %bound_method.loc41_44.4: <bound method> = bound_method %r.ref, %specific_impl_fn.loc41_44.2
// CHECK:STDOUT:     %.loc41_44.7: ref %Optional.5e5 = temporary_storage
// CHECK:STDOUT:     %Iterate.WithSelf.Next.call: init %Optional.5e5 to %.loc41_44.7 = call %bound_method.loc41_44.4(%r.ref, %var)
// CHECK:STDOUT:     %.loc41_44.8: ref %Optional.5e5 = temporary %.loc41_44.7, %Iterate.WithSelf.Next.call
// CHECK:STDOUT:     %.loc41_44.9: %Optional.HasValue.type.e84 = specific_constant imports.%Core.import_ref.c82, @Optional(constants.%OptionalStorage.facet.11f) [concrete = constants.%Optional.HasValue.cbc]
// CHECK:STDOUT:     %HasValue.ref: %Optional.HasValue.type.e84 = name_ref HasValue, %.loc41_44.9 [concrete = constants.%Optional.HasValue.cbc]
// CHECK:STDOUT:     %Optional.HasValue.bound: <bound method> = bound_method %.loc41_44.8, %HasValue.ref
// CHECK:STDOUT:     %Optional.HasValue.specific_fn: <specific function> = specific_function %HasValue.ref, @Optional.HasValue(constants.%OptionalStorage.facet.11f) [concrete = constants.%Optional.HasValue.specific_fn]
// CHECK:STDOUT:     %bound_method.loc41_44.5: <bound method> = bound_method %.loc41_44.8, %Optional.HasValue.specific_fn
// CHECK:STDOUT:     %.loc41_44.10: %Optional.5e5 = acquire_value %.loc41_44.8
// CHECK:STDOUT:     %Optional.HasValue.call: init bool = call %bound_method.loc41_44.5(%.loc41_44.10)
// CHECK:STDOUT:     %.loc41_44.11: bool = value_of_initializer %Optional.HasValue.call
// CHECK:STDOUT:     %.loc41_44.12: bool = converted %Optional.HasValue.call, %.loc41_44.11
// CHECK:STDOUT:     if %.loc41_44.12 br !for.body else br !for.done
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.body:
// CHECK:STDOUT:     %.loc41_44.13: %Optional.Get.type.e1b = specific_constant imports.%Core.import_ref.29e, @Optional(constants.%OptionalStorage.facet.11f) [concrete = constants.%Optional.Get.155]
// CHECK:STDOUT:     %Get.ref: %Optional.Get.type.e1b = name_ref Get, %.loc41_44.13 [concrete = constants.%Optional.Get.155]
// CHECK:STDOUT:     %Optional.Get.bound: <bound method> = bound_method %.loc41_44.8, %Get.ref
// CHECK:STDOUT:     %Optional.Get.specific_fn: <specific function> = specific_function %Get.ref, @Optional.Get(constants.%OptionalStorage.facet.11f) [concrete = constants.%Optional.Get.specific_fn]
// CHECK:STDOUT:     %bound_method.loc41_44.6: <bound method> = bound_method %.loc41_44.8, %Optional.Get.specific_fn
// CHECK:STDOUT:     %.loc41_44.14: ref %ValueType = temporary_storage
// CHECK:STDOUT:     %.loc41_44.15: %Optional.5e5 = acquire_value %.loc41_44.8
// CHECK:STDOUT:     %Optional.Get.call: init %ValueType to %.loc41_44.14 = call %bound_method.loc41_44.6(%.loc41_44.15)
// CHECK:STDOUT:     %.loc41_44.16: ref %ValueType = temporary %.loc41_44.14, %Optional.Get.call
// CHECK:STDOUT:     %.loc41_44.17: %ValueType = acquire_value %.loc41_44.16
// CHECK:STDOUT:     %.loc41_29: type = splice_block %ValueType.ref.loc41 [concrete = constants.%ValueType] {
// CHECK:STDOUT:       %Cpp.ref.loc41: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:       %DiffMutableADL.ref.loc41: <namespace> = name_ref DiffMutableADL, imports.%DiffMutableADL [concrete = imports.%DiffMutableADL]
// CHECK:STDOUT:       %ValueType.ref.loc41: type = name_ref ValueType, imports.%ValueType.decl [concrete = constants.%ValueType]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %i: %ValueType = wrapper_binding i, %.loc41_44.17
// CHECK:STDOUT:     %sum.ref: ref %ValueType = name_ref sum, %sum
// CHECK:STDOUT:     %i.ref: %ValueType = name_ref i, %i
// CHECK:STDOUT:     %.loc42_12: ref %ValueType = value_as_ref %i.ref
// CHECK:STDOUT:     %addr.loc42: %ptr.9c6 = addr_of %.loc42_12
// CHECK:STDOUT:     %.loc42_9.1: %ptr.5f4 = as_compatible %addr.loc42
// CHECK:STDOUT:     %.loc42_9.2: %ptr.5f4 = converted %addr.loc42, %.loc42_9.1
// CHECK:STDOUT:     %operator_PlusEqual__carbon_thunk.call: ref %ValueType = call imports.%operator_PlusEqual__carbon_thunk.decl(%sum.ref, %.loc42_9.2)
// CHECK:STDOUT:     %Destroy.facet.loc41_44.1: %Destroy.type = facet_value constants.%ValueType, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.127) [concrete = constants.%Destroy.facet.3c5]
// CHECK:STDOUT:     %.loc41_44.18: %Destroy.type = converted constants.%ValueType, %Destroy.facet.loc41_44.1 [concrete = constants.%Destroy.facet.3c5]
// CHECK:STDOUT:     %Destroy.facet.loc41_44.2: %Destroy.type = facet_value constants.%ValueType, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.127) [concrete = constants.%Destroy.facet.3c5]
// CHECK:STDOUT:     %.loc41_44.19: %Destroy.type = converted constants.%ValueType, %Destroy.facet.loc41_44.2 [concrete = constants.%Destroy.facet.3c5]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %SelfDestruct.call.loc41_44.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc41_44.16)
// CHECK:STDOUT:     %Destroy.facet.loc41_44.3: %Destroy.type = facet_value constants.%Optional.5e5, (constants.%custom_witness.b80ef6.8, constants.%DestroyImpl.impl_witness.587) [concrete = constants.%Destroy.facet.328]
// CHECK:STDOUT:     %.loc41_44.20: %Destroy.type = converted constants.%Optional.5e5, %Destroy.facet.loc41_44.3 [concrete = constants.%Destroy.facet.328]
// CHECK:STDOUT:     %Destroy.facet.loc41_44.4: %Destroy.type = facet_value constants.%Optional.5e5, (constants.%custom_witness.b80ef6.8, constants.%DestroyImpl.impl_witness.587) [concrete = constants.%Destroy.facet.328]
// CHECK:STDOUT:     %.loc41_44.21: %Destroy.type = converted constants.%Optional.5e5, %Destroy.facet.loc41_44.4 [concrete = constants.%Destroy.facet.328]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %SelfDestruct.call.loc41_44.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc41_44.8)
// CHECK:STDOUT:     br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.done:
// CHECK:STDOUT:     %Destroy.facet.loc41_44.5: %Destroy.type = facet_value constants.%Optional.5e5, (constants.%custom_witness.b80ef6.8, constants.%DestroyImpl.impl_witness.587) [concrete = constants.%Destroy.facet.328]
// CHECK:STDOUT:     %.loc41_44.22: %Destroy.type = converted constants.%Optional.5e5, %Destroy.facet.loc41_44.5 [concrete = constants.%Destroy.facet.328]
// CHECK:STDOUT:     %Destroy.facet.loc41_44.6: %Destroy.type = facet_value constants.%Optional.5e5, (constants.%custom_witness.b80ef6.8, constants.%DestroyImpl.impl_witness.587) [concrete = constants.%Destroy.facet.328]
// CHECK:STDOUT:     %.loc41_44.23: %Destroy.type = converted constants.%Optional.5e5, %Destroy.facet.loc41_44.6 [concrete = constants.%Destroy.facet.328]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %SelfDestruct.call.loc41_44.3: init %empty_tuple.type = call %SelfDestruct.specific_fn.3(%.loc41_44.8)
// CHECK:STDOUT:     %.loc41_44.24: %Destroy.type = converted constants.%as_type.7b4, constants.%impl.elem1.983 [symbolic = %impl.elem1 (constants.%impl.elem1.983)]
// CHECK:STDOUT:     %.loc41_44.25: %Destroy.type = converted constants.%as_type.7b4, constants.%impl.elem1.983 [symbolic = %impl.elem1 (constants.%impl.elem1.983)]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %SelfDestruct.call.loc41_44.4: init %empty_tuple.type = call %SelfDestruct.specific_fn.4(%var)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc41_44.1(%self.param: ref %.dd1) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc41_44.2(%self.param: ref %MaybeUnformed.753) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc41_44.3(%self.param: ref bool) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc41_44.4(%self.param: ref %struct_type.value.has_value.9df) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc41_44.5(%self.param: ref %DefaultOptionalStorage.ed6) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc41_44.6(%self.param: ref %Optional.5e5) = "no_op";
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
// CHECK:STDOUT:   %.loc63: type = splice_block %ValueType.ref.loc63 [concrete = constants.%ValueType] {
// CHECK:STDOUT:     %Cpp.ref.loc63: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %DiffMutableADL.ref.loc63: <namespace> = name_ref DiffMutableADL, imports.%DiffMutableADL [concrete = imports.%DiffMutableADL]
// CHECK:STDOUT:     %ValueType.ref.loc63: type = name_ref ValueType, imports.%ValueType.decl [concrete = constants.%ValueType]
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
// CHECK:STDOUT:   %R.patt.loc37_17.2 => constants.%R.patt
// CHECK:STDOUT:   %R.loc37_17.1 => constants.%R
// CHECK:STDOUT:   %R.as_type.loc37_86.1 => constants.%R.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.bac
// CHECK:STDOUT:   %r.param_patt.loc37_84.2 => constants.%r.param_patt
// CHECK:STDOUT:   %r.patt.loc37_84.2 => constants.%r.patt
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_adl_overloads.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %Iterate.type: type = facet_type <@Iterate> [concrete]
// CHECK:STDOUT:   %OptionalStorage.type: type = facet_type <@OptionalStorage> [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %facet_type.bd0: type = facet_type <@Copy & @Destroy> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %T.220e40.1: %OptionalStorage.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Optional.Get.type.786: type = fn_type @Optional.Get, @Optional(%T.220e40.1) [symbolic]
// CHECK:STDOUT:   %Optional.Get.2d0: %Optional.Get.type.786 = struct_value () [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.type.3ea: type = fn_type @Optional.HasValue, @Optional(%T.220e40.1) [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.f2f: %Optional.HasValue.type.3ea = struct_value () [symbolic]
// CHECK:STDOUT:   %T.f84392.1: %facet_type.bd0 = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.476: type = class_type @DefaultOptionalStorage, @DefaultOptionalStorage(%T.f84392.1) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Get.type.f27: type = fn_type @T.as_type.as.OptionalStorage.impl.Get, @T.as_type.as.OptionalStorage.impl(%T.f84392.1) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Get.e40: %T.as_type.as.OptionalStorage.impl.Get.type.f27 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Has.type.18e: type = fn_type @T.as_type.as.OptionalStorage.impl.Has, @T.as_type.as.OptionalStorage.impl(%T.f84392.1) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Has.463: %T.as_type.as.OptionalStorage.impl.Has.type.18e = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Some.type.292: type = fn_type @T.as_type.as.OptionalStorage.impl.Some, @T.as_type.as.OptionalStorage.impl(%T.f84392.1) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.Some.ae9: %T.as_type.as.OptionalStorage.impl.Some.type.292 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.None.type.b57: type = fn_type @T.as_type.as.OptionalStorage.impl.None, @T.as_type.as.OptionalStorage.impl(%T.f84392.1) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalStorage.impl.None.791: %T.as_type.as.OptionalStorage.impl.None.type.b57 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SelfDestruct.type: type = fn_type @SelfDestruct [concrete]
// CHECK:STDOUT:   %SelfDestruct: %SelfDestruct.type = struct_value () [concrete]
// CHECK:STDOUT:   %.Self.500: %Iterate.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %CppRange.type: type = facet_type <@CppRange> [concrete]
// CHECK:STDOUT:   %.Self.e0b: %CppRange.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %CppRangeForIterate.lookup_impl_witness.50b: <witness> = lookup_impl_witness %.Self.e0b, @CppRangeForIterate [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.32d3: type = impl_witness_access %CppRangeForIterate.lookup_impl_witness.50b, element0 [symbolic_self]
// CHECK:STDOUT:   %CppUnsafeDeref.lookup_impl_witness.851: <witness> = lookup_impl_witness %impl.elem0.32d3, @CppUnsafeDeref [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.10d: type = impl_witness_access %CppUnsafeDeref.lookup_impl_witness.851, element0 [symbolic_self]
// CHECK:STDOUT:   %CppRange_where.type.d682d6.1: type = facet_type <@CppRange where %impl.elem0.10d impls @Copy> [concrete]
// CHECK:STDOUT:   %T.f45530.3: %CppRange_where.type.d682d6.1 = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.type.d4a: type = fn_type @T.as_type.as.Iterate.impl.Next, @T.as_type.as.Iterate.impl(%T.f45530.3) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.5a8: %T.as_type.as.Iterate.impl.Next.type.d4a = struct_value () [symbolic]
// CHECK:STDOUT:   %CppRangeForIterate.lookup_impl_witness.b581a8.2: <witness> = lookup_impl_witness %T.f45530.3, @CppRangeForIterate [symbolic]
// CHECK:STDOUT:   %impl.elem0.5f971c.2: type = impl_witness_access %CppRangeForIterate.lookup_impl_witness.b581a8.2, element0 [symbolic]
// CHECK:STDOUT:   %CppUnsafeDeref.lookup_impl_witness.87e543.2: <witness> = lookup_impl_witness %impl.elem0.5f971c.2, @CppUnsafeDeref [symbolic]
// CHECK:STDOUT:   %impl.elem0.493765.2: type = impl_witness_access %CppUnsafeDeref.lookup_impl_witness.87e543.2, element0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.type.8fb: type = fn_type @T.as_type.as.Iterate.impl.NewCursor, @T.as_type.as.Iterate.impl(%T.f45530.3) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.8a1: %T.as_type.as.Iterate.impl.NewCursor.type.8fb = struct_value () [symbolic]
// CHECK:STDOUT:   %impl.elem1.c2dc15.2: type = impl_witness_access %CppRangeForIterate.lookup_impl_witness.b581a8.2, element1 [symbolic]
// CHECK:STDOUT:   %tuple.type.1ba: type = tuple_type (%impl.elem0.5f971c.2, %impl.elem1.c2dc15.2) [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.lookup_impl_witness.dc9: <witness> = lookup_impl_witness %tuple.type.1ba, @SubobjectDestroy [symbolic]
// CHECK:STDOUT:   %DestroyImpl.lookup_impl_witness.504: <witness> = lookup_impl_witness %tuple.type.1ba, @DestroyImpl [symbolic]
// CHECK:STDOUT:   %Destroy.facet.9b6: %Destroy.type = facet_value %tuple.type.1ba, (%SubobjectDestroy.lookup_impl_witness.dc9, %DestroyImpl.lookup_impl_witness.504) [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.lookup_impl_witness.b15da7.2: <witness> = lookup_impl_witness %impl.elem0.493765.2, @SubobjectDestroy [symbolic]
// CHECK:STDOUT:   %DestroyImpl.lookup_impl_witness.4e7f44.2: <witness> = lookup_impl_witness %impl.elem0.493765.2, @DestroyImpl [symbolic]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.167: <witness> = lookup_impl_witness %impl.elem0.493765.2, @Copy [symbolic]
// CHECK:STDOUT:   %facet_value.189: %facet_type.bd0 = facet_value %impl.elem0.493765.2, (%SubobjectDestroy.lookup_impl_witness.b15da7.2, %DestroyImpl.lookup_impl_witness.4e7f44.2, %Copy.lookup_impl_witness.167) [symbolic]
// CHECK:STDOUT:   %ValueType: type = class_type @ValueType [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %pattern_type.e09: type = pattern_type %ValueType [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.2: type = fn_type @SubobjectDestroy.WithSelf.Op.loc44 [concrete]
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
// CHECK:STDOUT:   %facet_value.595: %facet_type.bd0 = facet_value %ValueType, (%custom_witness.b80ef6.2, %DestroyImpl.impl_witness.012, %custom_witness.1c5) [concrete]
// CHECK:STDOUT:   %Iterate.lookup_impl_witness.1d8: <witness> = lookup_impl_witness %.Self.500, @Iterate [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.4ce: %facet_type.bd0 = impl_witness_access %Iterate.lookup_impl_witness.1d8, element0 [symbolic_self]
// CHECK:STDOUT:   %Iterate_where.type.84c: type = facet_type <@Iterate where %impl.elem0.4ce = %facet_value.595> [concrete]
// CHECK:STDOUT:   %pattern_type.5f4: type = pattern_type %Iterate_where.type.84c [concrete]
// CHECK:STDOUT:   %R.patt: %pattern_type.5f4 = symbolic_binding_pattern R, 0 [symbolic]
// CHECK:STDOUT:   %R: %Iterate_where.type.84c = symbolic_binding R, 0 [symbolic]
// CHECK:STDOUT:   %R.as_type: type = facet_access_type %R [symbolic]
// CHECK:STDOUT:   %pattern_type.82f: type = pattern_type %R.as_type [symbolic]
// CHECK:STDOUT:   %r.param_patt.6fe: %pattern_type.82f = value_param_pattern [symbolic]
// CHECK:STDOUT:   %r.patt.a30: %pattern_type.82f = wrapper_binding_pattern r, %r.param_patt.6fe [symbolic]
// CHECK:STDOUT:   %i.patt: %pattern_type.e09 = value_binding_pattern i [concrete]
// CHECK:STDOUT:   %Iterate.lookup_impl_witness.b84: <witness> = lookup_impl_witness %R, @Iterate [symbolic]
// CHECK:STDOUT:   %Iterate.facet.e0b: %Iterate.type = facet_value %R.as_type, (%Iterate.lookup_impl_witness.b84) [symbolic]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.9a6: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.e0b) [symbolic]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.2fb: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.e0b) [symbolic]
// CHECK:STDOUT:   %.cec: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.9a6, %Iterate.facet.e0b [symbolic]
// CHECK:STDOUT:   %impl.elem2.be7: %.cec = impl_witness_access %Iterate.lookup_impl_witness.b84, element2 [symbolic]
// CHECK:STDOUT:   %impl.elem1.1bb: %Destroy.type = impl_witness_access %Iterate.lookup_impl_witness.b84, element1 [symbolic]
// CHECK:STDOUT:   %as_type.b8c: type = facet_access_type %impl.elem1.1bb [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.06d: <specific function> = specific_impl_function %impl.elem2.be7, @Iterate.WithSelf.NewCursor(%Iterate.facet.e0b) [symbolic]
// CHECK:STDOUT:   %.5c9: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.2fb, %Iterate.facet.e0b [symbolic]
// CHECK:STDOUT:   %impl.elem3.b74: %.5c9 = impl_witness_access %Iterate.lookup_impl_witness.b84, element3 [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.96b: type = class_type @DefaultOptionalStorage, @DefaultOptionalStorage(%facet_value.595) [concrete]
// CHECK:STDOUT:   %OptionalStorage.impl_witness.f75: <witness> = impl_witness imports.%OptionalStorage.impl_witness_table.1c4, @T.as_type.as.OptionalStorage.impl(%facet_value.595) [concrete]
// CHECK:STDOUT:   %OptionalStorage.facet.104: %OptionalStorage.type = facet_value %ValueType, (%OptionalStorage.impl_witness.f75) [concrete]
// CHECK:STDOUT:   %Optional.21a: type = class_type @Optional, @Optional(%OptionalStorage.facet.104) [concrete]
// CHECK:STDOUT:   %specific_impl_fn.37f: <specific function> = specific_impl_function %impl.elem3.b74, @Iterate.WithSelf.Next(%Iterate.facet.e0b) [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.type.0ea: type = fn_type @Optional.HasValue, @Optional(%OptionalStorage.facet.104) [concrete]
// CHECK:STDOUT:   %Optional.HasValue.bd5: %Optional.HasValue.type.0ea = struct_value () [concrete]
// CHECK:STDOUT:   %Optional.Get.type.eb7: type = fn_type @Optional.Get, @Optional(%OptionalStorage.facet.104) [concrete]
// CHECK:STDOUT:   %Optional.Get.bc2: %Optional.Get.type.eb7 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.facet.e20: %Destroy.type = facet_value %ValueType, (%custom_witness.b80ef6.2, %DestroyImpl.impl_witness.012) [concrete]
// CHECK:STDOUT:   %MaybeUnformed.efd: type = class_type @MaybeUnformed, @MaybeUnformed(%Destroy.facet.e20) [concrete]
// CHECK:STDOUT:   %.8f3: type = maybe_unformed_type %ValueType [concrete]
// CHECK:STDOUT:   %struct_type.value.has_value.eb3: type = struct_type {.value: %MaybeUnformed.efd, .has_value: bool} [concrete]
// CHECK:STDOUT:   %Optional.HasValue.specific_fn.0b1: <specific function> = specific_function %Optional.HasValue.bd5, @Optional.HasValue(%OptionalStorage.facet.104) [concrete]
// CHECK:STDOUT:   %Optional.Get.specific_fn.8b8: <specific function> = specific_function %Optional.Get.bc2, @Optional.Get(%OptionalStorage.facet.104) [concrete]
// CHECK:STDOUT:   %operator_PlusEqual__carbon_thunk.type: type = fn_type @operator_PlusEqual__carbon_thunk [concrete]
// CHECK:STDOUT:   %operator_PlusEqual__carbon_thunk: %operator_PlusEqual__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.8: type = fn_type @SubobjectDestroy.WithSelf.Op.loc48_41.6 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.8: %SubobjectDestroy.WithSelf.Op.type.6870fd.8 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.8: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.8), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.0c0: %SubobjectDestroy.type = facet_value %Optional.21a, (%custom_witness.b80ef6.8) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.8de: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.0c0) [concrete]
// CHECK:STDOUT:   %Destroy.facet.90d: %Destroy.type = facet_value %Optional.21a, (%custom_witness.b80ef6.8, %DestroyImpl.impl_witness.8de) [concrete]
// CHECK:STDOUT:   %MutableAndConstRange: type = class_type @MutableAndConstRange [concrete]
// CHECK:STDOUT:   %pattern_type.fdb: type = pattern_type %MutableAndConstRange [concrete]
// CHECK:STDOUT:   %Iterator: type = class_type @Iterator.1 [concrete]
// CHECK:STDOUT:   %Begin.type: type = fn_type @Begin [concrete]
// CHECK:STDOUT:   %Begin: %Begin.type = struct_value () [concrete]
// CHECK:STDOUT:   %End.type: type = fn_type @End [concrete]
// CHECK:STDOUT:   %End: %End.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.660: <witness> = custom_witness (%Iterator, %Iterator, %Begin, %End), @CppRangeForIterate [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.9: type = fn_type @SubobjectDestroy.WithSelf.Op.loc55 [concrete]
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
// CHECK:STDOUT:   %Destroy.facet.a88: %Destroy.type = facet_value %tuple.type.de6, (%custom_witness.b80ef6.10, %DestroyImpl.impl_witness.2c4) [concrete]
// CHECK:STDOUT:   %Iterate.impl_witness.cd9: <witness> = impl_witness imports.%Iterate.impl_witness_table.121, @T.as_type.as.Iterate.impl(%facet_value.958) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.type.e6f: type = fn_type @T.as_type.as.Iterate.impl.NewCursor, @T.as_type.as.Iterate.impl(%facet_value.958) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.854: %T.as_type.as.Iterate.impl.NewCursor.type.e6f = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.type.40f: type = fn_type @T.as_type.as.Iterate.impl.Next, @T.as_type.as.Iterate.impl(%facet_value.958) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.2d9: %T.as_type.as.Iterate.impl.Next.type.40f = struct_value () [concrete]
// CHECK:STDOUT:   %facet_value.cba: %Iterate_where.type.84c = facet_value %MutableAndConstRange, (%Iterate.impl_witness.cd9) [concrete]
// CHECK:STDOUT:   %r.param_patt.3a6: %pattern_type.fdb = value_param_pattern [concrete]
// CHECK:STDOUT:   %r.patt.403: %pattern_type.fdb = wrapper_binding_pattern r, %r.param_patt.3a6 [concrete]
// CHECK:STDOUT:   %Iterate.facet.b8c: %Iterate.type = facet_value %MutableAndConstRange, (%Iterate.impl_witness.cd9) [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.1e1: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.b8c) [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.7a3: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.b8c) [concrete]
// CHECK:STDOUT:   %.521: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.1e1, %Iterate.facet.b8c [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.specific_fn: <specific function> = specific_function %T.as_type.as.Iterate.impl.NewCursor.854, @T.as_type.as.Iterate.impl.NewCursor(%facet_value.958) [concrete]
// CHECK:STDOUT:   %.92f: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.7a3, %Iterate.facet.b8c [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.specific_fn: <specific function> = specific_function %T.as_type.as.Iterate.impl.Next.2d9, @T.as_type.as.Iterate.impl.Next(%facet_value.958) [concrete]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.28f: <specific function> = specific_function %SelfDestruct, @SelfDestruct(%Destroy.facet.a88) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .OverloadADL = %OverloadADL
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.481: type = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%DefaultOptionalStorage (constants.%DefaultOptionalStorage.476)]
// CHECK:STDOUT:   %Core.import_ref.66d: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.None.type (%T.as_type.as.OptionalStorage.impl.None.type.b57) = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.None (constants.%T.as_type.as.OptionalStorage.impl.None.791)]
// CHECK:STDOUT:   %Core.import_ref.32c: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Some.type (%T.as_type.as.OptionalStorage.impl.Some.type.292) = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Some (constants.%T.as_type.as.OptionalStorage.impl.Some.ae9)]
// CHECK:STDOUT:   %Core.import_ref.801: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Has.type (%T.as_type.as.OptionalStorage.impl.Has.type.18e) = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Has (constants.%T.as_type.as.OptionalStorage.impl.Has.463)]
// CHECK:STDOUT:   %Core.import_ref.eec: @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Get.type (%T.as_type.as.OptionalStorage.impl.Get.type.f27) = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.OptionalStorage.impl.%T.as_type.as.OptionalStorage.impl.Get (constants.%T.as_type.as.OptionalStorage.impl.Get.e40)]
// CHECK:STDOUT:   %Core.import_ref.a6a = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %OptionalStorage.impl_witness_table.1c4 = impl_witness_table (%Core.import_ref.481, %Core.import_ref.66d, %Core.import_ref.32c, %Core.import_ref.801, %Core.import_ref.eec, %Core.import_ref.a6a), @T.as_type.as.OptionalStorage.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.7b5: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/types/optional, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.7b5), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.c82: @Optional.%Optional.HasValue.type (%Optional.HasValue.type.3ea) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.HasValue (constants.%Optional.HasValue.f2f)]
// CHECK:STDOUT:   %Core.import_ref.29e: @Optional.%Optional.Get.type (%Optional.Get.type.786) = import_ref Core//prelude/iterate, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @Optional.%Optional.Get (constants.%Optional.Get.2d0)]
// CHECK:STDOUT:   %Core.import_ref.724: %facet_type.bd0 = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.Iterate.impl.%facet_value (constants.%facet_value.189)]
// CHECK:STDOUT:   %Core.import_ref.dd9: %Destroy.type = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.Iterate.impl.%Destroy.facet (constants.%Destroy.facet.9b6)]
// CHECK:STDOUT:   %Core.import_ref.946: @T.as_type.as.Iterate.impl.%T.as_type.as.Iterate.impl.NewCursor.type (%T.as_type.as.Iterate.impl.NewCursor.type.8fb) = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.Iterate.impl.%T.as_type.as.Iterate.impl.NewCursor (constants.%T.as_type.as.Iterate.impl.NewCursor.8a1)]
// CHECK:STDOUT:   %Core.import_ref.cfb: @T.as_type.as.Iterate.impl.%T.as_type.as.Iterate.impl.Next.type (%T.as_type.as.Iterate.impl.Next.type.d4a) = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.Iterate.impl.%T.as_type.as.Iterate.impl.Next (constants.%T.as_type.as.Iterate.impl.Next.5a8)]
// CHECK:STDOUT:   %Iterate.impl_witness_table.121 = impl_witness_table (%Core.import_ref.724, %Core.import_ref.dd9, %Core.import_ref.946, %Core.import_ref.cfb), @T.as_type.as.Iterate.impl [concrete]
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
// CHECK:STDOUT: generic fn @TestIterate(%R.loc44_17.2: %Iterate_where.type.84c) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Iterate.lookup_impl_witness: <witness> = lookup_impl_witness %R.loc44_17.1, @Iterate [symbolic = %Iterate.lookup_impl_witness (constants.%Iterate.lookup_impl_witness.b84)]
// CHECK:STDOUT:   %Iterate.facet.loc48_41.5: %Iterate.type = facet_value %R.as_type.loc44_83.1, (%Iterate.lookup_impl_witness) [symbolic = %Iterate.facet.loc48_41.5 (constants.%Iterate.facet.e0b)]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.loc48_41.5) [symbolic = %Iterate.WithSelf.NewCursor.type (constants.%Iterate.WithSelf.NewCursor.type.9a6)]
// CHECK:STDOUT:   %.loc48_41.26: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type, %Iterate.facet.loc48_41.5 [symbolic = %.loc48_41.26 (constants.%.cec)]
// CHECK:STDOUT:   %impl.elem2.loc48_41.2: @TestIterate.%.loc48_41.26 (%.cec) = impl_witness_access %Iterate.lookup_impl_witness, element2 [symbolic = %impl.elem2.loc48_41.2 (constants.%impl.elem2.be7)]
// CHECK:STDOUT:   %specific_impl_fn.loc48_41.3: <specific function> = specific_impl_function %impl.elem2.loc48_41.2, @Iterate.WithSelf.NewCursor(%Iterate.facet.loc48_41.5) [symbolic = %specific_impl_fn.loc48_41.3 (constants.%specific_impl_fn.06d)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.loc48_41.5) [symbolic = %Iterate.WithSelf.Next.type (constants.%Iterate.WithSelf.Next.type.2fb)]
// CHECK:STDOUT:   %.loc48_41.27: type = fn_type_with_self_type %Iterate.WithSelf.Next.type, %Iterate.facet.loc48_41.5 [symbolic = %.loc48_41.27 (constants.%.5c9)]
// CHECK:STDOUT:   %impl.elem3.loc48_41.2: @TestIterate.%.loc48_41.27 (%.5c9) = impl_witness_access %Iterate.lookup_impl_witness, element3 [symbolic = %impl.elem3.loc48_41.2 (constants.%impl.elem3.b74)]
// CHECK:STDOUT:   %specific_impl_fn.loc48_41.4: <specific function> = specific_impl_function %impl.elem3.loc48_41.2, @Iterate.WithSelf.Next(%Iterate.facet.loc48_41.5) [symbolic = %specific_impl_fn.loc48_41.4 (constants.%specific_impl_fn.37f)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%r.param: @TestIterate.%R.as_type.loc44_83.1 (%R.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %i.patt: %pattern_type.e09 = value_binding_pattern i [concrete = constants.%i.patt]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %r.ref: @TestIterate.%R.as_type.loc44_83.1 (%R.as_type) = name_ref r, %r
// CHECK:STDOUT:     %impl.elem2.loc48_41.1: @TestIterate.%.loc48_41.26 (%.cec) = impl_witness_access constants.%Iterate.lookup_impl_witness.b84, element2 [symbolic = %impl.elem2.loc48_41.2 (constants.%impl.elem2.be7)]
// CHECK:STDOUT:     %bound_method.loc48_41.1: <bound method> = bound_method %r.ref, %impl.elem2.loc48_41.1
// CHECK:STDOUT:     %Iterate.facet.loc48_41.1: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.b84) [symbolic = %Iterate.facet.loc48_41.5 (constants.%Iterate.facet.e0b)]
// CHECK:STDOUT:     %.loc48_41.1: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc48_41.1 [symbolic = %Iterate.facet.loc48_41.5 (constants.%Iterate.facet.e0b)]
// CHECK:STDOUT:     %Iterate.facet.loc48_41.2: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.b84) [symbolic = %Iterate.facet.loc48_41.5 (constants.%Iterate.facet.e0b)]
// CHECK:STDOUT:     %.loc48_41.2: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc48_41.2 [symbolic = %Iterate.facet.loc48_41.5 (constants.%Iterate.facet.e0b)]
// CHECK:STDOUT:     %specific_impl_fn.loc48_41.1: <specific function> = specific_impl_function %impl.elem2.loc48_41.1, @Iterate.WithSelf.NewCursor(constants.%Iterate.facet.e0b) [symbolic = %specific_impl_fn.loc48_41.3 (constants.%specific_impl_fn.06d)]
// CHECK:STDOUT:     %bound_method.loc48_41.2: <bound method> = bound_method %r.ref, %specific_impl_fn.loc48_41.1
// CHECK:STDOUT:     %var: ref @TestIterate.%as_type (%as_type.b8c) = var_storage invalid
// CHECK:STDOUT:     %Iterate.WithSelf.NewCursor.call: init @TestIterate.%as_type (%as_type.b8c) to %var = call %bound_method.loc48_41.2(%r.ref)
// CHECK:STDOUT:     assign %var, %Iterate.WithSelf.NewCursor.call
// CHECK:STDOUT:     br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.next:
// CHECK:STDOUT:     %impl.elem3.loc48_41.1: @TestIterate.%.loc48_41.27 (%.5c9) = impl_witness_access constants.%Iterate.lookup_impl_witness.b84, element3 [symbolic = %impl.elem3.loc48_41.2 (constants.%impl.elem3.b74)]
// CHECK:STDOUT:     %bound_method.loc48_41.3: <bound method> = bound_method %r.ref, %impl.elem3.loc48_41.1
// CHECK:STDOUT:     %Iterate.facet.loc48_41.3: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.b84) [symbolic = %Iterate.facet.loc48_41.5 (constants.%Iterate.facet.e0b)]
// CHECK:STDOUT:     %.loc48_41.3: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc48_41.3 [symbolic = %Iterate.facet.loc48_41.5 (constants.%Iterate.facet.e0b)]
// CHECK:STDOUT:     %Iterate.facet.loc48_41.4: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.b84) [symbolic = %Iterate.facet.loc48_41.5 (constants.%Iterate.facet.e0b)]
// CHECK:STDOUT:     %.loc48_41.4: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc48_41.4 [symbolic = %Iterate.facet.loc48_41.5 (constants.%Iterate.facet.e0b)]
// CHECK:STDOUT:     %.loc48_41.5: %Destroy.type = converted constants.%as_type.b8c, constants.%impl.elem1.1bb [symbolic = %impl.elem1 (constants.%impl.elem1.1bb)]
// CHECK:STDOUT:     %.loc48_41.6: %Destroy.type = converted constants.%as_type.b8c, constants.%impl.elem1.1bb [symbolic = %impl.elem1 (constants.%impl.elem1.1bb)]
// CHECK:STDOUT:     %specific_impl_fn.loc48_41.2: <specific function> = specific_impl_function %impl.elem3.loc48_41.1, @Iterate.WithSelf.Next(constants.%Iterate.facet.e0b) [symbolic = %specific_impl_fn.loc48_41.4 (constants.%specific_impl_fn.37f)]
// CHECK:STDOUT:     %bound_method.loc48_41.4: <bound method> = bound_method %r.ref, %specific_impl_fn.loc48_41.2
// CHECK:STDOUT:     %.loc48_41.7: ref %Optional.21a = temporary_storage
// CHECK:STDOUT:     %Iterate.WithSelf.Next.call: init %Optional.21a to %.loc48_41.7 = call %bound_method.loc48_41.4(%r.ref, %var)
// CHECK:STDOUT:     %.loc48_41.8: ref %Optional.21a = temporary %.loc48_41.7, %Iterate.WithSelf.Next.call
// CHECK:STDOUT:     %.loc48_41.9: %Optional.HasValue.type.0ea = specific_constant imports.%Core.import_ref.c82, @Optional(constants.%OptionalStorage.facet.104) [concrete = constants.%Optional.HasValue.bd5]
// CHECK:STDOUT:     %HasValue.ref: %Optional.HasValue.type.0ea = name_ref HasValue, %.loc48_41.9 [concrete = constants.%Optional.HasValue.bd5]
// CHECK:STDOUT:     %Optional.HasValue.bound: <bound method> = bound_method %.loc48_41.8, %HasValue.ref
// CHECK:STDOUT:     %Optional.HasValue.specific_fn: <specific function> = specific_function %HasValue.ref, @Optional.HasValue(constants.%OptionalStorage.facet.104) [concrete = constants.%Optional.HasValue.specific_fn.0b1]
// CHECK:STDOUT:     %bound_method.loc48_41.5: <bound method> = bound_method %.loc48_41.8, %Optional.HasValue.specific_fn
// CHECK:STDOUT:     %.loc48_41.10: %Optional.21a = acquire_value %.loc48_41.8
// CHECK:STDOUT:     %Optional.HasValue.call: init bool = call %bound_method.loc48_41.5(%.loc48_41.10)
// CHECK:STDOUT:     %.loc48_41.11: bool = value_of_initializer %Optional.HasValue.call
// CHECK:STDOUT:     %.loc48_41.12: bool = converted %Optional.HasValue.call, %.loc48_41.11
// CHECK:STDOUT:     if %.loc48_41.12 br !for.body else br !for.done
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.body:
// CHECK:STDOUT:     %.loc48_41.13: %Optional.Get.type.eb7 = specific_constant imports.%Core.import_ref.29e, @Optional(constants.%OptionalStorage.facet.104) [concrete = constants.%Optional.Get.bc2]
// CHECK:STDOUT:     %Get.ref: %Optional.Get.type.eb7 = name_ref Get, %.loc48_41.13 [concrete = constants.%Optional.Get.bc2]
// CHECK:STDOUT:     %Optional.Get.bound: <bound method> = bound_method %.loc48_41.8, %Get.ref
// CHECK:STDOUT:     %Optional.Get.specific_fn: <specific function> = specific_function %Get.ref, @Optional.Get(constants.%OptionalStorage.facet.104) [concrete = constants.%Optional.Get.specific_fn.8b8]
// CHECK:STDOUT:     %bound_method.loc48_41.6: <bound method> = bound_method %.loc48_41.8, %Optional.Get.specific_fn
// CHECK:STDOUT:     %.loc48_41.14: ref %ValueType = temporary_storage
// CHECK:STDOUT:     %.loc48_41.15: %Optional.21a = acquire_value %.loc48_41.8
// CHECK:STDOUT:     %Optional.Get.call: init %ValueType to %.loc48_41.14 = call %bound_method.loc48_41.6(%.loc48_41.15)
// CHECK:STDOUT:     %.loc48_41.16: ref %ValueType = temporary %.loc48_41.14, %Optional.Get.call
// CHECK:STDOUT:     %.loc48_41.17: %ValueType = acquire_value %.loc48_41.16
// CHECK:STDOUT:     %.loc48_26: type = splice_block %ValueType.ref.loc48 [concrete = constants.%ValueType] {
// CHECK:STDOUT:       %Cpp.ref.loc48: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:       %OverloadADL.ref.loc48: <namespace> = name_ref OverloadADL, imports.%OverloadADL [concrete = imports.%OverloadADL]
// CHECK:STDOUT:       %ValueType.ref.loc48: type = name_ref ValueType, imports.%ValueType.decl [concrete = constants.%ValueType]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %i: %ValueType = wrapper_binding i, %.loc48_41.17
// CHECK:STDOUT:     %sum.ref: ref %ValueType = name_ref sum, %sum
// CHECK:STDOUT:     %i.ref: %ValueType = name_ref i, %i
// CHECK:STDOUT:     %.loc49_12: ref %ValueType = value_as_ref %i.ref
// CHECK:STDOUT:     %addr.loc49: %ptr.6ed = addr_of %.loc49_12
// CHECK:STDOUT:     %.loc49_9.1: %ptr.403 = as_compatible %addr.loc49
// CHECK:STDOUT:     %.loc49_9.2: %ptr.403 = converted %addr.loc49, %.loc49_9.1
// CHECK:STDOUT:     %operator_PlusEqual__carbon_thunk.call: ref %ValueType = call imports.%operator_PlusEqual__carbon_thunk.decl(%sum.ref, %.loc49_9.2)
// CHECK:STDOUT:     %Destroy.facet.loc48_41.1: %Destroy.type = facet_value constants.%ValueType, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.012) [concrete = constants.%Destroy.facet.e20]
// CHECK:STDOUT:     %.loc48_41.18: %Destroy.type = converted constants.%ValueType, %Destroy.facet.loc48_41.1 [concrete = constants.%Destroy.facet.e20]
// CHECK:STDOUT:     %Destroy.facet.loc48_41.2: %Destroy.type = facet_value constants.%ValueType, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.012) [concrete = constants.%Destroy.facet.e20]
// CHECK:STDOUT:     %.loc48_41.19: %Destroy.type = converted constants.%ValueType, %Destroy.facet.loc48_41.2 [concrete = constants.%Destroy.facet.e20]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %SelfDestruct.call.loc48_41.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc48_41.16)
// CHECK:STDOUT:     %Destroy.facet.loc48_41.3: %Destroy.type = facet_value constants.%Optional.21a, (constants.%custom_witness.b80ef6.8, constants.%DestroyImpl.impl_witness.8de) [concrete = constants.%Destroy.facet.90d]
// CHECK:STDOUT:     %.loc48_41.20: %Destroy.type = converted constants.%Optional.21a, %Destroy.facet.loc48_41.3 [concrete = constants.%Destroy.facet.90d]
// CHECK:STDOUT:     %Destroy.facet.loc48_41.4: %Destroy.type = facet_value constants.%Optional.21a, (constants.%custom_witness.b80ef6.8, constants.%DestroyImpl.impl_witness.8de) [concrete = constants.%Destroy.facet.90d]
// CHECK:STDOUT:     %.loc48_41.21: %Destroy.type = converted constants.%Optional.21a, %Destroy.facet.loc48_41.4 [concrete = constants.%Destroy.facet.90d]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %SelfDestruct.call.loc48_41.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc48_41.8)
// CHECK:STDOUT:     br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.done:
// CHECK:STDOUT:     %Destroy.facet.loc48_41.5: %Destroy.type = facet_value constants.%Optional.21a, (constants.%custom_witness.b80ef6.8, constants.%DestroyImpl.impl_witness.8de) [concrete = constants.%Destroy.facet.90d]
// CHECK:STDOUT:     %.loc48_41.22: %Destroy.type = converted constants.%Optional.21a, %Destroy.facet.loc48_41.5 [concrete = constants.%Destroy.facet.90d]
// CHECK:STDOUT:     %Destroy.facet.loc48_41.6: %Destroy.type = facet_value constants.%Optional.21a, (constants.%custom_witness.b80ef6.8, constants.%DestroyImpl.impl_witness.8de) [concrete = constants.%Destroy.facet.90d]
// CHECK:STDOUT:     %.loc48_41.23: %Destroy.type = converted constants.%Optional.21a, %Destroy.facet.loc48_41.6 [concrete = constants.%Destroy.facet.90d]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %SelfDestruct.call.loc48_41.3: init %empty_tuple.type = call %SelfDestruct.specific_fn.3(%.loc48_41.8)
// CHECK:STDOUT:     %.loc48_41.24: %Destroy.type = converted constants.%as_type.b8c, constants.%impl.elem1.1bb [symbolic = %impl.elem1 (constants.%impl.elem1.1bb)]
// CHECK:STDOUT:     %.loc48_41.25: %Destroy.type = converted constants.%as_type.b8c, constants.%impl.elem1.1bb [symbolic = %impl.elem1 (constants.%impl.elem1.1bb)]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %SelfDestruct.call.loc48_41.4: init %empty_tuple.type = call %SelfDestruct.specific_fn.4(%var)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc48_41.1(%self.param: ref %.8f3) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc48_41.2(%self.param: ref %MaybeUnformed.efd) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc48_41.3(%self.param: ref bool) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc48_41.4(%self.param: ref %struct_type.value.has_value.eb3) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc48_41.5(%self.param: ref %DefaultOptionalStorage.96b) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc48_41.6(%self.param: ref %Optional.21a) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestRangeFor(%mc.param: ref %MutableAndConstRange) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %i.patt: %pattern_type.e09 = value_binding_pattern i [concrete = constants.%i.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %mc.ref: ref %MutableAndConstRange = name_ref mc, %mc
// CHECK:STDOUT:   %impl.elem2: %.521 = impl_witness_access constants.%Iterate.impl_witness.cd9, element2 [concrete = constants.%T.as_type.as.Iterate.impl.NewCursor.854]
// CHECK:STDOUT:   %bound_method.loc62_42.1: <bound method> = bound_method %mc.ref, %impl.elem2
// CHECK:STDOUT:   %facet_value.loc62_42.1: %CppRange_where.type.d682d6.1 = facet_value constants.%MutableAndConstRange, (constants.%custom_witness.660, constants.%custom_witness.b80ef6.9, constants.%DestroyImpl.impl_witness.412, constants.%custom_witness.d56, constants.%custom_witness.68a, constants.%custom_witness.edd, constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.012, constants.%custom_witness.1c5, constants.%custom_witness.b80ef6.9, constants.%DestroyImpl.impl_witness.412) [concrete = constants.%facet_value.958]
// CHECK:STDOUT:   %.loc62_42.1: %CppRange_where.type.d682d6.1 = converted constants.%MutableAndConstRange, %facet_value.loc62_42.1 [concrete = constants.%facet_value.958]
// CHECK:STDOUT:   %facet_value.loc62_42.2: %CppRange_where.type.d682d6.1 = facet_value constants.%MutableAndConstRange, (constants.%custom_witness.660, constants.%custom_witness.b80ef6.9, constants.%DestroyImpl.impl_witness.412, constants.%custom_witness.d56, constants.%custom_witness.68a, constants.%custom_witness.edd, constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.012, constants.%custom_witness.1c5, constants.%custom_witness.b80ef6.9, constants.%DestroyImpl.impl_witness.412) [concrete = constants.%facet_value.958]
// CHECK:STDOUT:   %.loc62_42.2: %CppRange_where.type.d682d6.1 = converted constants.%MutableAndConstRange, %facet_value.loc62_42.2 [concrete = constants.%facet_value.958]
// CHECK:STDOUT:   %specific_fn.loc62_42.1: <specific function> = specific_function %impl.elem2, @T.as_type.as.Iterate.impl.NewCursor(constants.%facet_value.958) [concrete = constants.%T.as_type.as.Iterate.impl.NewCursor.specific_fn]
// CHECK:STDOUT:   %bound_method.loc62_42.2: <bound method> = bound_method %mc.ref, %specific_fn.loc62_42.1
// CHECK:STDOUT:   %var: ref %tuple.type.de6 = var_storage invalid
// CHECK:STDOUT:   %.loc62_40.1: %MutableAndConstRange = acquire_value %mc.ref
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.call: init %tuple.type.de6 to %var = call %bound_method.loc62_42.2(%.loc62_40.1)
// CHECK:STDOUT:   assign %var, %T.as_type.as.Iterate.impl.NewCursor.call
// CHECK:STDOUT:   br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT: !for.next:
// CHECK:STDOUT:   %impl.elem3: %.92f = impl_witness_access constants.%Iterate.impl_witness.cd9, element3 [concrete = constants.%T.as_type.as.Iterate.impl.Next.2d9]
// CHECK:STDOUT:   %bound_method.loc62_42.3: <bound method> = bound_method %mc.ref, %impl.elem3
// CHECK:STDOUT:   %facet_value.loc62_42.3: %CppRange_where.type.d682d6.1 = facet_value constants.%MutableAndConstRange, (constants.%custom_witness.660, constants.%custom_witness.b80ef6.9, constants.%DestroyImpl.impl_witness.412, constants.%custom_witness.d56, constants.%custom_witness.68a, constants.%custom_witness.edd, constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.012, constants.%custom_witness.1c5, constants.%custom_witness.b80ef6.9, constants.%DestroyImpl.impl_witness.412) [concrete = constants.%facet_value.958]
// CHECK:STDOUT:   %.loc62_42.3: %CppRange_where.type.d682d6.1 = converted constants.%MutableAndConstRange, %facet_value.loc62_42.3 [concrete = constants.%facet_value.958]
// CHECK:STDOUT:   %facet_value.loc62_42.4: %CppRange_where.type.d682d6.1 = facet_value constants.%MutableAndConstRange, (constants.%custom_witness.660, constants.%custom_witness.b80ef6.9, constants.%DestroyImpl.impl_witness.412, constants.%custom_witness.d56, constants.%custom_witness.68a, constants.%custom_witness.edd, constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.012, constants.%custom_witness.1c5, constants.%custom_witness.b80ef6.9, constants.%DestroyImpl.impl_witness.412) [concrete = constants.%facet_value.958]
// CHECK:STDOUT:   %.loc62_42.4: %CppRange_where.type.d682d6.1 = converted constants.%MutableAndConstRange, %facet_value.loc62_42.4 [concrete = constants.%facet_value.958]
// CHECK:STDOUT:   %specific_fn.loc62_42.2: <specific function> = specific_function %impl.elem3, @T.as_type.as.Iterate.impl.Next(constants.%facet_value.958) [concrete = constants.%T.as_type.as.Iterate.impl.Next.specific_fn]
// CHECK:STDOUT:   %bound_method.loc62_42.4: <bound method> = bound_method %mc.ref, %specific_fn.loc62_42.2
// CHECK:STDOUT:   %.loc62_42.5: ref %Optional.21a = temporary_storage
// CHECK:STDOUT:   %.loc62_40.2: %MutableAndConstRange = acquire_value %mc.ref
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.call: init %Optional.21a to %.loc62_42.5 = call %bound_method.loc62_42.4(%.loc62_40.2, %var)
// CHECK:STDOUT:   %.loc62_42.6: ref %Optional.21a = temporary %.loc62_42.5, %T.as_type.as.Iterate.impl.Next.call
// CHECK:STDOUT:   %.loc62_42.7: %Optional.HasValue.type.0ea = specific_constant imports.%Core.import_ref.c82, @Optional(constants.%OptionalStorage.facet.104) [concrete = constants.%Optional.HasValue.bd5]
// CHECK:STDOUT:   %HasValue.ref: %Optional.HasValue.type.0ea = name_ref HasValue, %.loc62_42.7 [concrete = constants.%Optional.HasValue.bd5]
// CHECK:STDOUT:   %Optional.HasValue.bound: <bound method> = bound_method %.loc62_42.6, %HasValue.ref
// CHECK:STDOUT:   %Optional.HasValue.specific_fn: <specific function> = specific_function %HasValue.ref, @Optional.HasValue(constants.%OptionalStorage.facet.104) [concrete = constants.%Optional.HasValue.specific_fn.0b1]
// CHECK:STDOUT:   %bound_method.loc62_42.5: <bound method> = bound_method %.loc62_42.6, %Optional.HasValue.specific_fn
// CHECK:STDOUT:   %.loc62_42.8: %Optional.21a = acquire_value %.loc62_42.6
// CHECK:STDOUT:   %Optional.HasValue.call: init bool = call %bound_method.loc62_42.5(%.loc62_42.8)
// CHECK:STDOUT:   %.loc62_42.9: bool = value_of_initializer %Optional.HasValue.call
// CHECK:STDOUT:   %.loc62_42.10: bool = converted %Optional.HasValue.call, %.loc62_42.9
// CHECK:STDOUT:   if %.loc62_42.10 br !for.body else br !for.done
// CHECK:STDOUT:
// CHECK:STDOUT: !for.body:
// CHECK:STDOUT:   %.loc62_42.11: %Optional.Get.type.eb7 = specific_constant imports.%Core.import_ref.29e, @Optional(constants.%OptionalStorage.facet.104) [concrete = constants.%Optional.Get.bc2]
// CHECK:STDOUT:   %Get.ref: %Optional.Get.type.eb7 = name_ref Get, %.loc62_42.11 [concrete = constants.%Optional.Get.bc2]
// CHECK:STDOUT:   %Optional.Get.bound: <bound method> = bound_method %.loc62_42.6, %Get.ref
// CHECK:STDOUT:   %Optional.Get.specific_fn: <specific function> = specific_function %Get.ref, @Optional.Get(constants.%OptionalStorage.facet.104) [concrete = constants.%Optional.Get.specific_fn.8b8]
// CHECK:STDOUT:   %bound_method.loc62_42.6: <bound method> = bound_method %.loc62_42.6, %Optional.Get.specific_fn
// CHECK:STDOUT:   %.loc62_42.12: ref %ValueType = temporary_storage
// CHECK:STDOUT:   %.loc62_42.13: %Optional.21a = acquire_value %.loc62_42.6
// CHECK:STDOUT:   %Optional.Get.call: init %ValueType to %.loc62_42.12 = call %bound_method.loc62_42.6(%.loc62_42.13)
// CHECK:STDOUT:   %.loc62_42.14: ref %ValueType = temporary %.loc62_42.12, %Optional.Get.call
// CHECK:STDOUT:   %.loc62_42.15: %ValueType = acquire_value %.loc62_42.14
// CHECK:STDOUT:   %.loc62_26: type = splice_block %ValueType.ref.loc62 [concrete = constants.%ValueType] {
// CHECK:STDOUT:     %Cpp.ref.loc62: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %OverloadADL.ref.loc62: <namespace> = name_ref OverloadADL, imports.%OverloadADL [concrete = imports.%OverloadADL]
// CHECK:STDOUT:     %ValueType.ref.loc62: type = name_ref ValueType, imports.%ValueType.decl [concrete = constants.%ValueType]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %i: %ValueType = wrapper_binding i, %.loc62_42.15
// CHECK:STDOUT:   %sum.ref: ref %ValueType = name_ref sum, %sum
// CHECK:STDOUT:   %i.ref: %ValueType = name_ref i, %i
// CHECK:STDOUT:   %.loc63_12: ref %ValueType = value_as_ref %i.ref
// CHECK:STDOUT:   %addr.loc63: %ptr.6ed = addr_of %.loc63_12
// CHECK:STDOUT:   %.loc63_9.1: %ptr.403 = as_compatible %addr.loc63
// CHECK:STDOUT:   %.loc63_9.2: %ptr.403 = converted %addr.loc63, %.loc63_9.1
// CHECK:STDOUT:   %operator_PlusEqual__carbon_thunk.call: ref %ValueType = call imports.%operator_PlusEqual__carbon_thunk.decl(%sum.ref, %.loc63_9.2)
// CHECK:STDOUT:   %Destroy.facet.loc62_42.1: %Destroy.type = facet_value constants.%ValueType, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.012) [concrete = constants.%Destroy.facet.e20]
// CHECK:STDOUT:   %.loc62_42.16: %Destroy.type = converted constants.%ValueType, %Destroy.facet.loc62_42.1 [concrete = constants.%Destroy.facet.e20]
// CHECK:STDOUT:   %Destroy.facet.loc62_42.2: %Destroy.type = facet_value constants.%ValueType, (constants.%custom_witness.b80ef6.2, constants.%DestroyImpl.impl_witness.012) [concrete = constants.%Destroy.facet.e20]
// CHECK:STDOUT:   %.loc62_42.17: %Destroy.type = converted constants.%ValueType, %Destroy.facet.loc62_42.2 [concrete = constants.%Destroy.facet.e20]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc62_42.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc62_42.14)
// CHECK:STDOUT:   %Destroy.facet.loc62_42.3: %Destroy.type = facet_value constants.%Optional.21a, (constants.%custom_witness.b80ef6.8, constants.%DestroyImpl.impl_witness.8de) [concrete = constants.%Destroy.facet.90d]
// CHECK:STDOUT:   %.loc62_42.18: %Destroy.type = converted constants.%Optional.21a, %Destroy.facet.loc62_42.3 [concrete = constants.%Destroy.facet.90d]
// CHECK:STDOUT:   %Destroy.facet.loc62_42.4: %Destroy.type = facet_value constants.%Optional.21a, (constants.%custom_witness.b80ef6.8, constants.%DestroyImpl.impl_witness.8de) [concrete = constants.%Destroy.facet.90d]
// CHECK:STDOUT:   %.loc62_42.19: %Destroy.type = converted constants.%Optional.21a, %Destroy.facet.loc62_42.4 [concrete = constants.%Destroy.facet.90d]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc62_42.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc62_42.6)
// CHECK:STDOUT:   br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT: !for.done:
// CHECK:STDOUT:   %Destroy.facet.loc62_42.5: %Destroy.type = facet_value constants.%Optional.21a, (constants.%custom_witness.b80ef6.8, constants.%DestroyImpl.impl_witness.8de) [concrete = constants.%Destroy.facet.90d]
// CHECK:STDOUT:   %.loc62_42.20: %Destroy.type = converted constants.%Optional.21a, %Destroy.facet.loc62_42.5 [concrete = constants.%Destroy.facet.90d]
// CHECK:STDOUT:   %Destroy.facet.loc62_42.6: %Destroy.type = facet_value constants.%Optional.21a, (constants.%custom_witness.b80ef6.8, constants.%DestroyImpl.impl_witness.8de) [concrete = constants.%Destroy.facet.90d]
// CHECK:STDOUT:   %.loc62_42.21: %Destroy.type = converted constants.%Optional.21a, %Destroy.facet.loc62_42.6 [concrete = constants.%Destroy.facet.90d]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc62_42.3: init %empty_tuple.type = call %SelfDestruct.specific_fn.3(%.loc62_42.6)
// CHECK:STDOUT:   %Destroy.facet.loc62_42.7: %Destroy.type = facet_value constants.%tuple.type.de6, (constants.%custom_witness.b80ef6.10, constants.%DestroyImpl.impl_witness.2c4) [concrete = constants.%Destroy.facet.a88]
// CHECK:STDOUT:   %.loc62_42.22: %Destroy.type = converted constants.%tuple.type.de6, %Destroy.facet.loc62_42.7 [concrete = constants.%Destroy.facet.a88]
// CHECK:STDOUT:   %Destroy.facet.loc62_42.8: %Destroy.type = facet_value constants.%tuple.type.de6, (constants.%custom_witness.b80ef6.10, constants.%DestroyImpl.impl_witness.2c4) [concrete = constants.%Destroy.facet.a88]
// CHECK:STDOUT:   %.loc62_42.23: %Destroy.type = converted constants.%tuple.type.de6, %Destroy.facet.loc62_42.8 [concrete = constants.%Destroy.facet.a88]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc62_42.4: init %empty_tuple.type = call %SelfDestruct.specific_fn.4(%var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestIterate(constants.%R) {
// CHECK:STDOUT:   %R.patt.loc44_17.2 => constants.%R.patt
// CHECK:STDOUT:   %R.loc44_17.1 => constants.%R
// CHECK:STDOUT:   %R.as_type.loc44_83.1 => constants.%R.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.82f
// CHECK:STDOUT:   %r.param_patt.loc44_81.2 => constants.%r.param_patt.6fe
// CHECK:STDOUT:   %r.patt.loc44_81.2 => constants.%r.patt.a30
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestIterate(constants.%facet_value.cba) {
// CHECK:STDOUT:   %R.patt.loc44_17.2 => constants.%R.patt
// CHECK:STDOUT:   %R.loc44_17.1 => constants.%facet_value.cba
// CHECK:STDOUT:   %R.as_type.loc44_83.1 => constants.%MutableAndConstRange
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.fdb
// CHECK:STDOUT:   %r.param_patt.loc44_81.2 => constants.%r.param_patt.3a6
// CHECK:STDOUT:   %r.patt.loc44_81.2 => constants.%r.patt.403
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc44 => constants.%complete_type.357
// CHECK:STDOUT:   %Iterate.lookup_impl_witness => constants.%Iterate.impl_witness.cd9
// CHECK:STDOUT:   %Iterate.facet.loc48_41.5 => constants.%Iterate.facet.b8c
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type => constants.%Iterate.WithSelf.NewCursor.type.1e1
// CHECK:STDOUT:   %.loc48_41.26 => constants.%.521
// CHECK:STDOUT:   %impl.elem2.loc48_41.2 => constants.%T.as_type.as.Iterate.impl.NewCursor.854
// CHECK:STDOUT:   %specific_impl_fn.loc48_41.3 => constants.%T.as_type.as.Iterate.impl.NewCursor.specific_fn
// CHECK:STDOUT:   %impl.elem1 => constants.%Destroy.facet.a88
// CHECK:STDOUT:   %as_type => constants.%tuple.type.de6
// CHECK:STDOUT:   %require_complete.1 => constants.%complete_type.95d
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type => constants.%Iterate.WithSelf.Next.type.7a3
// CHECK:STDOUT:   %.loc48_41.27 => constants.%.92f
// CHECK:STDOUT:   %impl.elem3.loc48_41.2 => constants.%T.as_type.as.Iterate.impl.Next.2d9
// CHECK:STDOUT:   %specific_impl_fn.loc48_41.4 => constants.%T.as_type.as.Iterate.impl.Next.specific_fn
// CHECK:STDOUT:   %SelfDestruct.specific_fn.6 => constants.%SelfDestruct.specific_fn.28f
// CHECK:STDOUT: }
// CHECK:STDOUT:
