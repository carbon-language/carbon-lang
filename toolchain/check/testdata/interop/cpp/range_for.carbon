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
// CHECK:STDOUT:   %.ab9: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.1e7, %Iterate.facet.342 [symbolic]
// CHECK:STDOUT:   %impl.elem3.e47: %.ab9 = impl_witness_access %Iterate.lookup_impl_witness.d33, element3 [symbolic]
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
// CHECK:STDOUT:   %Iterator.Op.type.0aed5d.1: type = fn_type @Iterator.Op.1 [concrete]
// CHECK:STDOUT:   %Iterator.Op.a2a866.1: %Iterator.Op.type.0aed5d.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.da7: <witness> = custom_witness (%Iterator.Op.a2a866.1), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.2d2: %SubobjectDestroy.type = facet_value %Iterator.3ee, (%custom_witness.da7) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.eff: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.2d2) [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.0aed5d.2: type = fn_type @Iterator.Op.2 [concrete]
// CHECK:STDOUT:   %Iterator.Op.a2a866.2: %Iterator.Op.type.0aed5d.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.43f: <witness> = custom_witness (%i32, %Iterator.Op.a2a866.2), @CppUnsafeDeref [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.0aed5d.3: type = fn_type @Iterator.Op.3 [concrete]
// CHECK:STDOUT:   %Iterator.Op.a2a866.3: %Iterator.Op.type.0aed5d.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.444: <witness> = custom_witness (%Iterator.Op.a2a866.3), @Inc [concrete]
// CHECK:STDOUT:   %Equal.type.ca8475.1: type = fn_type @Equal.1 [concrete]
// CHECK:STDOUT:   %Equal.285b15.1: %Equal.type.ca8475.1 = struct_value () [concrete]
// CHECK:STDOUT:   %NotEqual.type.53d9ee.1: type = fn_type @NotEqual.1 [concrete]
// CHECK:STDOUT:   %NotEqual.119263.1: %NotEqual.type.53d9ee.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.c91: <witness> = custom_witness (%Equal.285b15.1, %NotEqual.119263.1), @EqWith, @EqWith(%Iterator.3ee) [concrete]
// CHECK:STDOUT:   %facet_value.941: %CppRange_where.type.d682d6.1 = facet_value %MutableRange, (%custom_witness.ed3, %custom_witness.da7, %DestroyImpl.impl_witness.eff, %custom_witness.43f, %custom_witness.444, %custom_witness.c91, %custom_witness.b80ef6.1, %DestroyImpl.impl_witness.077, %Copy.impl_witness.8af, %custom_witness.da7, %DestroyImpl.impl_witness.eff) [concrete]
// CHECK:STDOUT:   %tuple.type.508: type = tuple_type (%Iterator.3ee, %Iterator.3ee) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.9: type = fn_type @SubobjectDestroy.WithSelf.Op.3 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.9: %SubobjectDestroy.WithSelf.Op.type.6870fd.9 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.9: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.9), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.669: %SubobjectDestroy.type = facet_value %tuple.type.508, (%custom_witness.b80ef6.9) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.91e: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.669) [concrete]
// CHECK:STDOUT:   %complete_type.c28: <witness> = complete_type_witness %tuple.type.508 [concrete]
// CHECK:STDOUT:   %Destroy.facet.70f: %Destroy.type = facet_value %tuple.type.508, (%custom_witness.b80ef6.9, %DestroyImpl.impl_witness.91e) [concrete]
// CHECK:STDOUT:   %Iterate.impl_witness.d35: <witness> = impl_witness imports.%Iterate.impl_witness_table.121, @T.as_type.as.Iterate.impl(%facet_value.941) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.type.902: type = fn_type @T.as_type.as.Iterate.impl.NewCursor, @T.as_type.as.Iterate.impl(%facet_value.941) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.c31: %T.as_type.as.Iterate.impl.NewCursor.type.902 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.type.1ab: type = fn_type @T.as_type.as.Iterate.impl.Next, @T.as_type.as.Iterate.impl(%facet_value.941) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.7de: %T.as_type.as.Iterate.impl.Next.type.1ab = struct_value () [concrete]
// CHECK:STDOUT:   %facet_value.7cf: %Iterate_where.type.afe = facet_value %MutableRange, (%Iterate.impl_witness.d35) [concrete]
// CHECK:STDOUT:   %r.param_patt.c79: %pattern_type.e63 = value_param_pattern [concrete]
// CHECK:STDOUT:   %r.patt.94f: %pattern_type.e63 = wrapper_binding_pattern r, %r.param_patt.c79 [concrete]
// CHECK:STDOUT:   %Iterator.260: type = class_type @Iterator.2 [concrete]
// CHECK:STDOUT:   %ConstRange.Begin.type: type = fn_type @ConstRange.Begin [concrete]
// CHECK:STDOUT:   %ConstRange.Begin: %ConstRange.Begin.type = struct_value () [concrete]
// CHECK:STDOUT:   %ConstRange.End.type: type = fn_type @ConstRange.End [concrete]
// CHECK:STDOUT:   %ConstRange.End: %ConstRange.End.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.23e: <witness> = custom_witness (%Iterator.260, %Iterator.260, %ConstRange.Begin, %ConstRange.End), @CppRangeForIterate [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.0f34b8.1: type = fn_type @Iterator.Op.4 [concrete]
// CHECK:STDOUT:   %Iterator.Op.1c3e4a.1: %Iterator.Op.type.0f34b8.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.fcc: <witness> = custom_witness (%Iterator.Op.1c3e4a.1), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.909: %SubobjectDestroy.type = facet_value %Iterator.260, (%custom_witness.fcc) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.dfd: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.909) [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.0f34b8.2: type = fn_type @Iterator.Op.5 [concrete]
// CHECK:STDOUT:   %Iterator.Op.1c3e4a.2: %Iterator.Op.type.0f34b8.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.f5c: <witness> = custom_witness (%i32, %Iterator.Op.1c3e4a.2), @CppUnsafeDeref [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.0f34b8.3: type = fn_type @Iterator.Op.6 [concrete]
// CHECK:STDOUT:   %Iterator.Op.1c3e4a.3: %Iterator.Op.type.0f34b8.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.746: <witness> = custom_witness (%Iterator.Op.1c3e4a.3), @Inc [concrete]
// CHECK:STDOUT:   %Equal.type.ca8475.2: type = fn_type @Equal.2 [concrete]
// CHECK:STDOUT:   %Equal.285b15.2: %Equal.type.ca8475.2 = struct_value () [concrete]
// CHECK:STDOUT:   %NotEqual.type.53d9ee.2: type = fn_type @NotEqual.2 [concrete]
// CHECK:STDOUT:   %NotEqual.119263.2: %NotEqual.type.53d9ee.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.86f: <witness> = custom_witness (%Equal.285b15.2, %NotEqual.119263.2), @EqWith, @EqWith(%Iterator.260) [concrete]
// CHECK:STDOUT:   %facet_value.43d: %CppRange_where.type.d682d6.1 = facet_value %ConstRange, (%custom_witness.23e, %custom_witness.fcc, %DestroyImpl.impl_witness.dfd, %custom_witness.f5c, %custom_witness.746, %custom_witness.86f, %custom_witness.b80ef6.1, %DestroyImpl.impl_witness.077, %Copy.impl_witness.8af, %custom_witness.fcc, %DestroyImpl.impl_witness.dfd) [concrete]
// CHECK:STDOUT:   %tuple.type.69f: type = tuple_type (%Iterator.260, %Iterator.260) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.10: type = fn_type @SubobjectDestroy.WithSelf.Op.4 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.10: %SubobjectDestroy.WithSelf.Op.type.6870fd.10 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.10: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.10), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.100: %SubobjectDestroy.type = facet_value %tuple.type.69f, (%custom_witness.b80ef6.10) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.4ff: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.100) [concrete]
// CHECK:STDOUT:   %complete_type.8d5: <witness> = complete_type_witness %tuple.type.69f [concrete]
// CHECK:STDOUT:   %Destroy.facet.141: %Destroy.type = facet_value %tuple.type.69f, (%custom_witness.b80ef6.10, %DestroyImpl.impl_witness.4ff) [concrete]
// CHECK:STDOUT:   %Iterate.impl_witness.c9d: <witness> = impl_witness imports.%Iterate.impl_witness_table.121, @T.as_type.as.Iterate.impl(%facet_value.43d) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.type.14e: type = fn_type @T.as_type.as.Iterate.impl.NewCursor, @T.as_type.as.Iterate.impl(%facet_value.43d) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.649: %T.as_type.as.Iterate.impl.NewCursor.type.14e = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.type.088: type = fn_type @T.as_type.as.Iterate.impl.Next, @T.as_type.as.Iterate.impl(%facet_value.43d) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.320: %T.as_type.as.Iterate.impl.Next.type.088 = struct_value () [concrete]
// CHECK:STDOUT:   %facet_value.590: %Iterate_where.type.afe = facet_value %ConstRange, (%Iterate.impl_witness.c9d) [concrete]
// CHECK:STDOUT:   %r.param_patt.83e: %pattern_type.0ba = value_param_pattern [concrete]
// CHECK:STDOUT:   %r.patt.c21: %pattern_type.0ba = wrapper_binding_pattern r, %r.param_patt.83e [concrete]
// CHECK:STDOUT:   %Iterate.facet.e8f: %Iterate.type = facet_value %MutableRange, (%Iterate.impl_witness.d35) [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.b5d: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.e8f) [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.e6d: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.e8f) [concrete]
// CHECK:STDOUT:   %.6b7: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.b5d, %Iterate.facet.e8f [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.specific_fn.881: <specific function> = specific_function %T.as_type.as.Iterate.impl.NewCursor.c31, @T.as_type.as.Iterate.impl.NewCursor(%facet_value.941) [concrete]
// CHECK:STDOUT:   %.0ac: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.e6d, %Iterate.facet.e8f [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.specific_fn.e7d: <specific function> = specific_function %T.as_type.as.Iterate.impl.Next.7de, @T.as_type.as.Iterate.impl.Next(%facet_value.941) [concrete]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.a77: <specific function> = specific_function %SelfDestruct, @SelfDestruct(%Destroy.facet.70f) [concrete]
// CHECK:STDOUT:   %Iterate.facet.80e: %Iterate.type = facet_value %ConstRange, (%Iterate.impl_witness.c9d) [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.be9: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.80e) [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.4a4: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.80e) [concrete]
// CHECK:STDOUT:   %.b3c: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.be9, %Iterate.facet.80e [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.specific_fn.750: <specific function> = specific_function %T.as_type.as.Iterate.impl.NewCursor.649, @T.as_type.as.Iterate.impl.NewCursor(%facet_value.43d) [concrete]
// CHECK:STDOUT:   %.c69: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.4a4, %Iterate.facet.80e [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.specific_fn.d88: <specific function> = specific_function %T.as_type.as.Iterate.impl.Next.320, @T.as_type.as.Iterate.impl.Next(%facet_value.43d) [concrete]
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
// CHECK:STDOUT:   %.loc34_19.24: type = fn_type_with_self_type %Iterate.WithSelf.Next.type, %Iterate.facet.loc34_19.5 [symbolic = %.loc34_19.24 (constants.%.ab9)]
// CHECK:STDOUT:   %impl.elem3.loc34_19.2: @TestIterate.%.loc34_19.24 (%.ab9) = impl_witness_access %Iterate.lookup_impl_witness, element3 [symbolic = %impl.elem3.loc34_19.2 (constants.%impl.elem3.e47)]
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
// CHECK:STDOUT:     %impl.elem3.loc34_19.1: @TestIterate.%.loc34_19.24 (%.ab9) = impl_witness_access constants.%Iterate.lookup_impl_witness.d33, element3 [symbolic = %impl.elem3.loc34_19.2 (constants.%impl.elem3.e47)]
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
// CHECK:STDOUT:   %impl.elem2.loc49: %.6b7 = impl_witness_access constants.%Iterate.impl_witness.d35, element2 [concrete = constants.%T.as_type.as.Iterate.impl.NewCursor.c31]
// CHECK:STDOUT:   %bound_method.loc49_19.1: <bound method> = bound_method %m.ref, %impl.elem2.loc49
// CHECK:STDOUT:   %facet_value.loc49_19.1: %CppRange_where.type.d682d6.1 = facet_value constants.%MutableRange, (constants.%custom_witness.ed3, constants.%custom_witness.da7, constants.%DestroyImpl.impl_witness.eff, constants.%custom_witness.43f, constants.%custom_witness.444, constants.%custom_witness.c91, constants.%custom_witness.b80ef6.1, constants.%DestroyImpl.impl_witness.077, constants.%Copy.impl_witness.8af, constants.%custom_witness.da7, constants.%DestroyImpl.impl_witness.eff) [concrete = constants.%facet_value.941]
// CHECK:STDOUT:   %.loc49_19.1: %CppRange_where.type.d682d6.1 = converted constants.%MutableRange, %facet_value.loc49_19.1 [concrete = constants.%facet_value.941]
// CHECK:STDOUT:   %facet_value.loc49_19.2: %CppRange_where.type.d682d6.1 = facet_value constants.%MutableRange, (constants.%custom_witness.ed3, constants.%custom_witness.da7, constants.%DestroyImpl.impl_witness.eff, constants.%custom_witness.43f, constants.%custom_witness.444, constants.%custom_witness.c91, constants.%custom_witness.b80ef6.1, constants.%DestroyImpl.impl_witness.077, constants.%Copy.impl_witness.8af, constants.%custom_witness.da7, constants.%DestroyImpl.impl_witness.eff) [concrete = constants.%facet_value.941]
// CHECK:STDOUT:   %.loc49_19.2: %CppRange_where.type.d682d6.1 = converted constants.%MutableRange, %facet_value.loc49_19.2 [concrete = constants.%facet_value.941]
// CHECK:STDOUT:   %specific_fn.loc49_19.1: <specific function> = specific_function %impl.elem2.loc49, @T.as_type.as.Iterate.impl.NewCursor(constants.%facet_value.941) [concrete = constants.%T.as_type.as.Iterate.impl.NewCursor.specific_fn.881]
// CHECK:STDOUT:   %bound_method.loc49_19.2: <bound method> = bound_method %m.ref, %specific_fn.loc49_19.1
// CHECK:STDOUT:   %var.loc49: ref %tuple.type.508 = var_storage invalid
// CHECK:STDOUT:   %.loc49_18.1: %MutableRange = acquire_value %m.ref
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.call.loc49: init %tuple.type.508 to %var.loc49 = call %bound_method.loc49_19.2(%.loc49_18.1)
// CHECK:STDOUT:   assign %var.loc49, %T.as_type.as.Iterate.impl.NewCursor.call.loc49
// CHECK:STDOUT:   br !for.next.loc49
// CHECK:STDOUT:
// CHECK:STDOUT: !for.next.loc49:
// CHECK:STDOUT:   %impl.elem3.loc49: %.0ac = impl_witness_access constants.%Iterate.impl_witness.d35, element3 [concrete = constants.%T.as_type.as.Iterate.impl.Next.7de]
// CHECK:STDOUT:   %bound_method.loc49_19.3: <bound method> = bound_method %m.ref, %impl.elem3.loc49
// CHECK:STDOUT:   %facet_value.loc49_19.3: %CppRange_where.type.d682d6.1 = facet_value constants.%MutableRange, (constants.%custom_witness.ed3, constants.%custom_witness.da7, constants.%DestroyImpl.impl_witness.eff, constants.%custom_witness.43f, constants.%custom_witness.444, constants.%custom_witness.c91, constants.%custom_witness.b80ef6.1, constants.%DestroyImpl.impl_witness.077, constants.%Copy.impl_witness.8af, constants.%custom_witness.da7, constants.%DestroyImpl.impl_witness.eff) [concrete = constants.%facet_value.941]
// CHECK:STDOUT:   %.loc49_19.3: %CppRange_where.type.d682d6.1 = converted constants.%MutableRange, %facet_value.loc49_19.3 [concrete = constants.%facet_value.941]
// CHECK:STDOUT:   %facet_value.loc49_19.4: %CppRange_where.type.d682d6.1 = facet_value constants.%MutableRange, (constants.%custom_witness.ed3, constants.%custom_witness.da7, constants.%DestroyImpl.impl_witness.eff, constants.%custom_witness.43f, constants.%custom_witness.444, constants.%custom_witness.c91, constants.%custom_witness.b80ef6.1, constants.%DestroyImpl.impl_witness.077, constants.%Copy.impl_witness.8af, constants.%custom_witness.da7, constants.%DestroyImpl.impl_witness.eff) [concrete = constants.%facet_value.941]
// CHECK:STDOUT:   %.loc49_19.4: %CppRange_where.type.d682d6.1 = converted constants.%MutableRange, %facet_value.loc49_19.4 [concrete = constants.%facet_value.941]
// CHECK:STDOUT:   %specific_fn.loc49_19.2: <specific function> = specific_function %impl.elem3.loc49, @T.as_type.as.Iterate.impl.Next(constants.%facet_value.941) [concrete = constants.%T.as_type.as.Iterate.impl.Next.specific_fn.e7d]
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
// CHECK:STDOUT:   %Destroy.facet.loc49_19.5: %Destroy.type = facet_value constants.%tuple.type.508, (constants.%custom_witness.b80ef6.9, constants.%DestroyImpl.impl_witness.91e) [concrete = constants.%Destroy.facet.70f]
// CHECK:STDOUT:   %.loc49_19.19: %Destroy.type = converted constants.%tuple.type.508, %Destroy.facet.loc49_19.5 [concrete = constants.%Destroy.facet.70f]
// CHECK:STDOUT:   %Destroy.facet.loc49_19.6: %Destroy.type = facet_value constants.%tuple.type.508, (constants.%custom_witness.b80ef6.9, constants.%DestroyImpl.impl_witness.91e) [concrete = constants.%Destroy.facet.70f]
// CHECK:STDOUT:   %.loc49_19.20: %Destroy.type = converted constants.%tuple.type.508, %Destroy.facet.loc49_19.6 [concrete = constants.%Destroy.facet.70f]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc49_19.3: init %empty_tuple.type = call %SelfDestruct.specific_fn.3(%var.loc49)
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %i.patt.loc55: %pattern_type.d94 = value_binding_pattern i [concrete = constants.%i.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c.ref: %ConstRange = name_ref c, %c
// CHECK:STDOUT:   %impl.elem2.loc55: %.b3c = impl_witness_access constants.%Iterate.impl_witness.c9d, element2 [concrete = constants.%T.as_type.as.Iterate.impl.NewCursor.649]
// CHECK:STDOUT:   %bound_method.loc55_19.1: <bound method> = bound_method %c.ref, %impl.elem2.loc55
// CHECK:STDOUT:   %facet_value.loc55_19.1: %CppRange_where.type.d682d6.1 = facet_value constants.%ConstRange, (constants.%custom_witness.23e, constants.%custom_witness.fcc, constants.%DestroyImpl.impl_witness.dfd, constants.%custom_witness.f5c, constants.%custom_witness.746, constants.%custom_witness.86f, constants.%custom_witness.b80ef6.1, constants.%DestroyImpl.impl_witness.077, constants.%Copy.impl_witness.8af, constants.%custom_witness.fcc, constants.%DestroyImpl.impl_witness.dfd) [concrete = constants.%facet_value.43d]
// CHECK:STDOUT:   %.loc55_19.1: %CppRange_where.type.d682d6.1 = converted constants.%ConstRange, %facet_value.loc55_19.1 [concrete = constants.%facet_value.43d]
// CHECK:STDOUT:   %facet_value.loc55_19.2: %CppRange_where.type.d682d6.1 = facet_value constants.%ConstRange, (constants.%custom_witness.23e, constants.%custom_witness.fcc, constants.%DestroyImpl.impl_witness.dfd, constants.%custom_witness.f5c, constants.%custom_witness.746, constants.%custom_witness.86f, constants.%custom_witness.b80ef6.1, constants.%DestroyImpl.impl_witness.077, constants.%Copy.impl_witness.8af, constants.%custom_witness.fcc, constants.%DestroyImpl.impl_witness.dfd) [concrete = constants.%facet_value.43d]
// CHECK:STDOUT:   %.loc55_19.2: %CppRange_where.type.d682d6.1 = converted constants.%ConstRange, %facet_value.loc55_19.2 [concrete = constants.%facet_value.43d]
// CHECK:STDOUT:   %specific_fn.loc55_19.1: <specific function> = specific_function %impl.elem2.loc55, @T.as_type.as.Iterate.impl.NewCursor(constants.%facet_value.43d) [concrete = constants.%T.as_type.as.Iterate.impl.NewCursor.specific_fn.750]
// CHECK:STDOUT:   %bound_method.loc55_19.2: <bound method> = bound_method %c.ref, %specific_fn.loc55_19.1
// CHECK:STDOUT:   %var.loc55: ref %tuple.type.69f = var_storage invalid
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.call.loc55: init %tuple.type.69f to %var.loc55 = call %bound_method.loc55_19.2(%c.ref)
// CHECK:STDOUT:   assign %var.loc55, %T.as_type.as.Iterate.impl.NewCursor.call.loc55
// CHECK:STDOUT:   br !for.next.loc55
// CHECK:STDOUT:
// CHECK:STDOUT: !for.next.loc55:
// CHECK:STDOUT:   %impl.elem3.loc55: %.c69 = impl_witness_access constants.%Iterate.impl_witness.c9d, element3 [concrete = constants.%T.as_type.as.Iterate.impl.Next.320]
// CHECK:STDOUT:   %bound_method.loc55_19.3: <bound method> = bound_method %c.ref, %impl.elem3.loc55
// CHECK:STDOUT:   %facet_value.loc55_19.3: %CppRange_where.type.d682d6.1 = facet_value constants.%ConstRange, (constants.%custom_witness.23e, constants.%custom_witness.fcc, constants.%DestroyImpl.impl_witness.dfd, constants.%custom_witness.f5c, constants.%custom_witness.746, constants.%custom_witness.86f, constants.%custom_witness.b80ef6.1, constants.%DestroyImpl.impl_witness.077, constants.%Copy.impl_witness.8af, constants.%custom_witness.fcc, constants.%DestroyImpl.impl_witness.dfd) [concrete = constants.%facet_value.43d]
// CHECK:STDOUT:   %.loc55_19.3: %CppRange_where.type.d682d6.1 = converted constants.%ConstRange, %facet_value.loc55_19.3 [concrete = constants.%facet_value.43d]
// CHECK:STDOUT:   %facet_value.loc55_19.4: %CppRange_where.type.d682d6.1 = facet_value constants.%ConstRange, (constants.%custom_witness.23e, constants.%custom_witness.fcc, constants.%DestroyImpl.impl_witness.dfd, constants.%custom_witness.f5c, constants.%custom_witness.746, constants.%custom_witness.86f, constants.%custom_witness.b80ef6.1, constants.%DestroyImpl.impl_witness.077, constants.%Copy.impl_witness.8af, constants.%custom_witness.fcc, constants.%DestroyImpl.impl_witness.dfd) [concrete = constants.%facet_value.43d]
// CHECK:STDOUT:   %.loc55_19.4: %CppRange_where.type.d682d6.1 = converted constants.%ConstRange, %facet_value.loc55_19.4 [concrete = constants.%facet_value.43d]
// CHECK:STDOUT:   %specific_fn.loc55_19.2: <specific function> = specific_function %impl.elem3.loc55, @T.as_type.as.Iterate.impl.Next(constants.%facet_value.43d) [concrete = constants.%T.as_type.as.Iterate.impl.Next.specific_fn.d88]
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
// CHECK:STDOUT:   %Destroy.facet.loc55_19.5: %Destroy.type = facet_value constants.%tuple.type.69f, (constants.%custom_witness.b80ef6.10, constants.%DestroyImpl.impl_witness.4ff) [concrete = constants.%Destroy.facet.141]
// CHECK:STDOUT:   %.loc55_19.19: %Destroy.type = converted constants.%tuple.type.69f, %Destroy.facet.loc55_19.5 [concrete = constants.%Destroy.facet.141]
// CHECK:STDOUT:   %Destroy.facet.loc55_19.6: %Destroy.type = facet_value constants.%tuple.type.69f, (constants.%custom_witness.b80ef6.10, constants.%DestroyImpl.impl_witness.4ff) [concrete = constants.%Destroy.facet.141]
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
// CHECK:STDOUT: specific @TestIterate(constants.%facet_value.7cf) {
// CHECK:STDOUT:   %R.patt.loc30_17.2 => constants.%R.patt
// CHECK:STDOUT:   %R.loc30_17.1 => constants.%facet_value.7cf
// CHECK:STDOUT:   %R.as_type.loc30_61.1 => constants.%MutableRange
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.e63
// CHECK:STDOUT:   %r.param_patt.loc30_59.2 => constants.%r.param_patt.c79
// CHECK:STDOUT:   %r.patt.loc30_59.2 => constants.%r.patt.94f
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc30 => constants.%complete_type.357
// CHECK:STDOUT:   %Iterate.lookup_impl_witness => constants.%Iterate.impl_witness.d35
// CHECK:STDOUT:   %Iterate.facet.loc34_19.5 => constants.%Iterate.facet.e8f
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type => constants.%Iterate.WithSelf.NewCursor.type.b5d
// CHECK:STDOUT:   %.loc34_19.23 => constants.%.6b7
// CHECK:STDOUT:   %impl.elem2.loc34_19.2 => constants.%T.as_type.as.Iterate.impl.NewCursor.c31
// CHECK:STDOUT:   %specific_impl_fn.loc34_19.3 => constants.%T.as_type.as.Iterate.impl.NewCursor.specific_fn.881
// CHECK:STDOUT:   %impl.elem1 => constants.%Destroy.facet.70f
// CHECK:STDOUT:   %as_type => constants.%tuple.type.508
// CHECK:STDOUT:   %require_complete.1 => constants.%complete_type.c28
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type => constants.%Iterate.WithSelf.Next.type.e6d
// CHECK:STDOUT:   %.loc34_19.24 => constants.%.0ac
// CHECK:STDOUT:   %impl.elem3.loc34_19.2 => constants.%T.as_type.as.Iterate.impl.Next.7de
// CHECK:STDOUT:   %specific_impl_fn.loc34_19.4 => constants.%T.as_type.as.Iterate.impl.Next.specific_fn.e7d
// CHECK:STDOUT:   %SelfDestruct.specific_fn.5 => constants.%SelfDestruct.specific_fn.a77
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestIterate(constants.%facet_value.590) {
// CHECK:STDOUT:   %R.patt.loc30_17.2 => constants.%R.patt
// CHECK:STDOUT:   %R.loc30_17.1 => constants.%facet_value.590
// CHECK:STDOUT:   %R.as_type.loc30_61.1 => constants.%ConstRange
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.0ba
// CHECK:STDOUT:   %r.param_patt.loc30_59.2 => constants.%r.param_patt.83e
// CHECK:STDOUT:   %r.patt.loc30_59.2 => constants.%r.patt.c21
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc30 => constants.%complete_type.357
// CHECK:STDOUT:   %Iterate.lookup_impl_witness => constants.%Iterate.impl_witness.c9d
// CHECK:STDOUT:   %Iterate.facet.loc34_19.5 => constants.%Iterate.facet.80e
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type => constants.%Iterate.WithSelf.NewCursor.type.be9
// CHECK:STDOUT:   %.loc34_19.23 => constants.%.b3c
// CHECK:STDOUT:   %impl.elem2.loc34_19.2 => constants.%T.as_type.as.Iterate.impl.NewCursor.649
// CHECK:STDOUT:   %specific_impl_fn.loc34_19.3 => constants.%T.as_type.as.Iterate.impl.NewCursor.specific_fn.750
// CHECK:STDOUT:   %impl.elem1 => constants.%Destroy.facet.141
// CHECK:STDOUT:   %as_type => constants.%tuple.type.69f
// CHECK:STDOUT:   %require_complete.1 => constants.%complete_type.8d5
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type => constants.%Iterate.WithSelf.Next.type.4a4
// CHECK:STDOUT:   %.loc34_19.24 => constants.%.c69
// CHECK:STDOUT:   %impl.elem3.loc34_19.2 => constants.%T.as_type.as.Iterate.impl.Next.320
// CHECK:STDOUT:   %specific_impl_fn.loc34_19.4 => constants.%T.as_type.as.Iterate.impl.Next.specific_fn.d88
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
// CHECK:STDOUT:   %T.as_type.c1cc3e.1: type = facet_access_type %T.f84392.1 [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.lookup_impl_witness.70f: <witness> = lookup_impl_witness %T.f84392.1, @SubobjectDestroy [symbolic]
// CHECK:STDOUT:   %DestroyImpl.lookup_impl_witness.bf2: <witness> = lookup_impl_witness %T.f84392.1, @DestroyImpl [symbolic]
// CHECK:STDOUT:   %Destroy.facet.dc4: %Destroy.type = facet_value %T.as_type.c1cc3e.1, (%SubobjectDestroy.lookup_impl_witness.70f, %DestroyImpl.lookup_impl_witness.bf2) [symbolic]
// CHECK:STDOUT:   %MaybeUnformed.348: type = class_type @MaybeUnformed, @MaybeUnformed(%Destroy.facet.dc4) [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.476: type = class_type @DefaultOptionalStorage, @DefaultOptionalStorage(%T.f84392.1) [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.elem.4a2: type = unbound_element_type %DefaultOptionalStorage.476, bool [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.elem.ff3: type = unbound_element_type %DefaultOptionalStorage.476, %MaybeUnformed.348 [symbolic]
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
// CHECK:STDOUT:   %DiffValueType.Op.type.83af6e.1: type = fn_type @DiffValueType.Op.1 [concrete]
// CHECK:STDOUT:   %DiffValueType.Op.4dca31.1: %DiffValueType.Op.type.83af6e.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.5760f4.1: <witness> = custom_witness (%DiffValueType.Op.4dca31.1), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.7ea: %SubobjectDestroy.type = facet_value %DiffValueType, (%custom_witness.5760f4.1) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.c9a: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.7ea) [concrete]
// CHECK:STDOUT:   %const.879: type = const_type %DiffValueType [concrete]
// CHECK:STDOUT:   %ptr.816: type = ptr_type %const.879 [concrete]
// CHECK:STDOUT:   %ptr.cd0: type = ptr_type %DiffValueType [concrete]
// CHECK:STDOUT:   %DiffValueType.Op.type.83af6e.2: type = fn_type @DiffValueType.Op.2 [concrete]
// CHECK:STDOUT:   %DiffValueType.Op.4dca31.2: %DiffValueType.Op.type.83af6e.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.d0da43.1: <witness> = custom_witness (%DiffValueType.Op.4dca31.2), @Copy [concrete]
// CHECK:STDOUT:   %facet_value.873: %facet_type.bd0 = facet_value %DiffValueType, (%custom_witness.5760f4.1, %DestroyImpl.impl_witness.c9a, %custom_witness.d0da43.1) [concrete]
// CHECK:STDOUT:   %Iterate.lookup_impl_witness.1d8: <witness> = lookup_impl_witness %.Self.500, @Iterate [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.4ce: %facet_type.bd0 = impl_witness_access %Iterate.lookup_impl_witness.1d8, element0 [symbolic_self]
// CHECK:STDOUT:   %Iterate_where.type.1f9: type = facet_type <@Iterate where %impl.elem0.4ce = %facet_value.873> [concrete]
// CHECK:STDOUT:   %pattern_type.2de: type = pattern_type %Iterate_where.type.1f9 [concrete]
// CHECK:STDOUT:   %R.patt: %pattern_type.2de = symbolic_binding_pattern R, 0 [symbolic]
// CHECK:STDOUT:   %R: %Iterate_where.type.1f9 = symbolic_binding R, 0 [symbolic]
// CHECK:STDOUT:   %R.as_type: type = facet_access_type %R [symbolic]
// CHECK:STDOUT:   %pattern_type.172: type = pattern_type %R.as_type [symbolic]
// CHECK:STDOUT:   %r.patt.ea6: %pattern_type.172 = ref_binding_pattern r [symbolic]
// CHECK:STDOUT:   %r.param_patt.528: %pattern_type.172 = var_param_pattern %r.patt.ea6 [symbolic]
// CHECK:STDOUT:   %i.patt: %pattern_type.149 = value_binding_pattern i [concrete]
// CHECK:STDOUT:   %Iterate.lookup_impl_witness.b9d: <witness> = lookup_impl_witness %R, @Iterate [symbolic]
// CHECK:STDOUT:   %Iterate.facet.8f9: %Iterate.type = facet_value %R.as_type, (%Iterate.lookup_impl_witness.b9d) [symbolic]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.0c9: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.8f9) [symbolic]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.84e: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.8f9) [symbolic]
// CHECK:STDOUT:   %.3d4: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.0c9, %Iterate.facet.8f9 [symbolic]
// CHECK:STDOUT:   %impl.elem2.53f: %.3d4 = impl_witness_access %Iterate.lookup_impl_witness.b9d, element2 [symbolic]
// CHECK:STDOUT:   %impl.elem1.782: %Destroy.type = impl_witness_access %Iterate.lookup_impl_witness.b9d, element1 [symbolic]
// CHECK:STDOUT:   %as_type.240: type = facet_access_type %impl.elem1.782 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.b76: <specific function> = specific_impl_function %impl.elem2.53f, @Iterate.WithSelf.NewCursor(%Iterate.facet.8f9) [symbolic]
// CHECK:STDOUT:   %.73c: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.84e, %Iterate.facet.8f9 [symbolic]
// CHECK:STDOUT:   %impl.elem3.015: %.73c = impl_witness_access %Iterate.lookup_impl_witness.b9d, element3 [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.7fa: type = class_type @DefaultOptionalStorage, @DefaultOptionalStorage(%facet_value.873) [concrete]
// CHECK:STDOUT:   %OptionalStorage.impl_witness.f4e: <witness> = impl_witness imports.%OptionalStorage.impl_witness_table.1c4, @T.as_type.as.OptionalStorage.impl(%facet_value.873) [concrete]
// CHECK:STDOUT:   %OptionalStorage.facet.f87: %OptionalStorage.type = facet_value %DiffValueType, (%OptionalStorage.impl_witness.f4e) [concrete]
// CHECK:STDOUT:   %Optional.250: type = class_type @Optional, @Optional(%OptionalStorage.facet.f87) [concrete]
// CHECK:STDOUT:   %specific_impl_fn.f44: <specific function> = specific_impl_function %impl.elem3.015, @Iterate.WithSelf.Next(%Iterate.facet.8f9) [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.type.7a2: type = fn_type @Optional.HasValue, @Optional(%OptionalStorage.facet.f87) [concrete]
// CHECK:STDOUT:   %Optional.HasValue.0b9: %Optional.HasValue.type.7a2 = struct_value () [concrete]
// CHECK:STDOUT:   %Optional.Get.type.fa0: type = fn_type @Optional.Get, @Optional(%OptionalStorage.facet.f87) [concrete]
// CHECK:STDOUT:   %Optional.Get.f6a: %Optional.Get.type.fa0 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.facet.91b: %Destroy.type = facet_value %DiffValueType, (%custom_witness.5760f4.1, %DestroyImpl.impl_witness.c9a) [concrete]
// CHECK:STDOUT:   %MaybeUnformed.4a6: type = class_type @MaybeUnformed, @MaybeUnformed(%Destroy.facet.91b) [concrete]
// CHECK:STDOUT:   %.a18: type = maybe_unformed_type %DiffValueType [concrete]
// CHECK:STDOUT:   %DefaultOptionalStorage.elem.fb7: type = unbound_element_type %DefaultOptionalStorage.7fa, %MaybeUnformed.4a6 [concrete]
// CHECK:STDOUT:   %DefaultOptionalStorage.elem.54f: type = unbound_element_type %DefaultOptionalStorage.7fa, bool [concrete]
// CHECK:STDOUT:   %struct_type.value.has_value.b29: type = struct_type {.value: %MaybeUnformed.4a6, .has_value: bool} [concrete]
// CHECK:STDOUT:   %Optional.HasValue.specific_fn.366: <specific function> = specific_function %Optional.HasValue.0b9, @Optional.HasValue(%OptionalStorage.facet.f87) [concrete]
// CHECK:STDOUT:   %Optional.Get.specific_fn.29e: <specific function> = specific_function %Optional.Get.f6a, @Optional.Get(%OptionalStorage.facet.f87) [concrete]
// CHECK:STDOUT:   %operator_PlusEqual__carbon_thunk.type: type = fn_type @operator_PlusEqual__carbon_thunk [concrete]
// CHECK:STDOUT:   %operator_PlusEqual__carbon_thunk: %operator_PlusEqual__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.3: type = fn_type @SubobjectDestroy.WithSelf.Op.loc44_33.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.3: %SubobjectDestroy.WithSelf.Op.type.6870fd.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.3: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.3), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.4: type = fn_type @SubobjectDestroy.WithSelf.Op.loc44_33.3 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.4: %SubobjectDestroy.WithSelf.Op.type.6870fd.4 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.4: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.4), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.a87: %SubobjectDestroy.type = facet_value bool, (%custom_witness.b80ef6.4) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.778: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.a87) [concrete]
// CHECK:STDOUT:   %Destroy.facet.80c: %Destroy.type = facet_value bool, (%custom_witness.b80ef6.4, %DestroyImpl.impl_witness.778) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.1f6: %SubobjectDestroy.type = facet_value %MaybeUnformed.4a6, (%custom_witness.b80ef6.3) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.6f4: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.1f6) [concrete]
// CHECK:STDOUT:   %Destroy.facet.0a4: %Destroy.type = facet_value %MaybeUnformed.4a6, (%custom_witness.b80ef6.3, %DestroyImpl.impl_witness.6f4) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.7: type = fn_type @SubobjectDestroy.WithSelf.Op.loc44_33.6 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.7: %SubobjectDestroy.WithSelf.Op.type.6870fd.7 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.7: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.7), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.a85: %SubobjectDestroy.type = facet_value %Optional.250, (%custom_witness.b80ef6.7) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.dec: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.a85) [concrete]
// CHECK:STDOUT:   %Destroy.facet.ba0: %Destroy.type = facet_value %Optional.250, (%custom_witness.b80ef6.7, %DestroyImpl.impl_witness.dec) [concrete]
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
// CHECK:STDOUT:   %custom_witness.3815f0.1: <witness> = custom_witness (%Iterator.Op.1caa36.1), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.131: %SubobjectDestroy.type = facet_value %Iterator.f25, (%custom_witness.3815f0.1) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.03b: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.131) [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.41bbea.2: type = fn_type @Iterator.Op.2 [concrete]
// CHECK:STDOUT:   %Iterator.Op.1caa36.2: %Iterator.Op.type.41bbea.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.392: <witness> = custom_witness (%DiffValueType, %Iterator.Op.1caa36.2), @CppUnsafeDeref [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.41bbea.3: type = fn_type @Iterator.Op.3 [concrete]
// CHECK:STDOUT:   %Iterator.Op.1caa36.3: %Iterator.Op.type.41bbea.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.1c9: <witness> = custom_witness (%Iterator.Op.1caa36.3), @Inc [concrete]
// CHECK:STDOUT:   %Equal.type.ca8475.1: type = fn_type @Equal.1 [concrete]
// CHECK:STDOUT:   %Equal.285b15.1: %Equal.type.ca8475.1 = struct_value () [concrete]
// CHECK:STDOUT:   %NotEqual.type.53d9ee.1: type = fn_type @NotEqual.1 [concrete]
// CHECK:STDOUT:   %NotEqual.119263.1: %NotEqual.type.53d9ee.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b32: <witness> = custom_witness (%Equal.285b15.1, %NotEqual.119263.1), @EqWith, @EqWith(%Sentinel.b4c) [concrete]
// CHECK:STDOUT:   %Sentinel.Op.type.41b: type = fn_type @Sentinel.Op.1 [concrete]
// CHECK:STDOUT:   %Sentinel.Op.1ca: %Sentinel.Op.type.41b = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.3815f0.2: <witness> = custom_witness (%Sentinel.Op.1ca), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.d42: %SubobjectDestroy.type = facet_value %Sentinel.b4c, (%custom_witness.3815f0.2) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.508: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.d42) [concrete]
// CHECK:STDOUT:   %facet_value.6c7: %CppRange_where.type.d682d6.1 = facet_value %DiffMutableRange, (%custom_witness.aa8, %custom_witness.3815f0.1, %DestroyImpl.impl_witness.03b, %custom_witness.392, %custom_witness.1c9, %custom_witness.b32, %custom_witness.5760f4.1, %DestroyImpl.impl_witness.c9a, %custom_witness.d0da43.1, %custom_witness.3815f0.2, %DestroyImpl.impl_witness.508) [concrete]
// CHECK:STDOUT:   %tuple.type.b18: type = tuple_type (%Iterator.f25, %Sentinel.b4c) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.8: type = fn_type @SubobjectDestroy.WithSelf.Op.3 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.8: %SubobjectDestroy.WithSelf.Op.type.6870fd.8 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.8: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.8), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.d2e: %SubobjectDestroy.type = facet_value %tuple.type.b18, (%custom_witness.b80ef6.8) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.c40: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.d2e) [concrete]
// CHECK:STDOUT:   %complete_type.bfe: <witness> = complete_type_witness %tuple.type.b18 [concrete]
// CHECK:STDOUT:   %Destroy.facet.4da: %Destroy.type = facet_value %tuple.type.b18, (%custom_witness.b80ef6.8, %DestroyImpl.impl_witness.c40) [concrete]
// CHECK:STDOUT:   %Iterate.impl_witness.0fb: <witness> = impl_witness imports.%Iterate.impl_witness_table.121, @T.as_type.as.Iterate.impl(%facet_value.6c7) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.type.8f4: type = fn_type @T.as_type.as.Iterate.impl.NewCursor, @T.as_type.as.Iterate.impl(%facet_value.6c7) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.573: %T.as_type.as.Iterate.impl.NewCursor.type.8f4 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.type.8e9: type = fn_type @T.as_type.as.Iterate.impl.Next, @T.as_type.as.Iterate.impl(%facet_value.6c7) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.288: %T.as_type.as.Iterate.impl.Next.type.8e9 = struct_value () [concrete]
// CHECK:STDOUT:   %facet_value.847: %Iterate_where.type.1f9 = facet_value %DiffMutableRange, (%Iterate.impl_witness.0fb) [concrete]
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
// CHECK:STDOUT:   %custom_witness.3d0529.1: <witness> = custom_witness (%Iterator.Op.690fee.1), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.9b7: %SubobjectDestroy.type = facet_value %Iterator.a59, (%custom_witness.3d0529.1) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.bd5: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.9b7) [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.b7ae80.2: type = fn_type @Iterator.Op.5 [concrete]
// CHECK:STDOUT:   %Iterator.Op.690fee.2: %Iterator.Op.type.b7ae80.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.0c0: <witness> = custom_witness (%DiffValueType, %Iterator.Op.690fee.2), @CppUnsafeDeref [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.b7ae80.3: type = fn_type @Iterator.Op.6 [concrete]
// CHECK:STDOUT:   %Iterator.Op.690fee.3: %Iterator.Op.type.b7ae80.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.ffb: <witness> = custom_witness (%Iterator.Op.690fee.3), @Inc [concrete]
// CHECK:STDOUT:   %Equal.type.ca8475.2: type = fn_type @Equal.2 [concrete]
// CHECK:STDOUT:   %Equal.285b15.2: %Equal.type.ca8475.2 = struct_value () [concrete]
// CHECK:STDOUT:   %NotEqual.type.53d9ee.2: type = fn_type @NotEqual.2 [concrete]
// CHECK:STDOUT:   %NotEqual.119263.2: %NotEqual.type.53d9ee.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.fc2: <witness> = custom_witness (%Equal.285b15.2, %NotEqual.119263.2), @EqWith, @EqWith(%Sentinel.b33) [concrete]
// CHECK:STDOUT:   %Sentinel.Op.type.b7a: type = fn_type @Sentinel.Op.2 [concrete]
// CHECK:STDOUT:   %Sentinel.Op.690: %Sentinel.Op.type.b7a = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.3d0529.2: <witness> = custom_witness (%Sentinel.Op.690), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.b86: %SubobjectDestroy.type = facet_value %Sentinel.b33, (%custom_witness.3d0529.2) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.780: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.b86) [concrete]
// CHECK:STDOUT:   %facet_value.047: %CppRange_where.type.d682d6.1 = facet_value %DiffConstRange, (%custom_witness.7b4, %custom_witness.3d0529.1, %DestroyImpl.impl_witness.bd5, %custom_witness.0c0, %custom_witness.ffb, %custom_witness.fc2, %custom_witness.5760f4.1, %DestroyImpl.impl_witness.c9a, %custom_witness.d0da43.1, %custom_witness.3d0529.2, %DestroyImpl.impl_witness.780) [concrete]
// CHECK:STDOUT:   %tuple.type.df5: type = tuple_type (%Iterator.a59, %Sentinel.b33) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.9: type = fn_type @SubobjectDestroy.WithSelf.Op.4 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.9: %SubobjectDestroy.WithSelf.Op.type.6870fd.9 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.9: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.9), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.79d: %SubobjectDestroy.type = facet_value %tuple.type.df5, (%custom_witness.b80ef6.9) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.8c3: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.79d) [concrete]
// CHECK:STDOUT:   %complete_type.ad6: <witness> = complete_type_witness %tuple.type.df5 [concrete]
// CHECK:STDOUT:   %Destroy.facet.017: %Destroy.type = facet_value %tuple.type.df5, (%custom_witness.b80ef6.9, %DestroyImpl.impl_witness.8c3) [concrete]
// CHECK:STDOUT:   %Iterate.impl_witness.0fc: <witness> = impl_witness imports.%Iterate.impl_witness_table.121, @T.as_type.as.Iterate.impl(%facet_value.047) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.type.284: type = fn_type @T.as_type.as.Iterate.impl.NewCursor, @T.as_type.as.Iterate.impl(%facet_value.047) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.c96: %T.as_type.as.Iterate.impl.NewCursor.type.284 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.type.5e4: type = fn_type @T.as_type.as.Iterate.impl.Next, @T.as_type.as.Iterate.impl(%facet_value.047) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.264: %T.as_type.as.Iterate.impl.Next.type.5e4 = struct_value () [concrete]
// CHECK:STDOUT:   %facet_value.76f: %Iterate_where.type.1f9 = facet_value %DiffConstRange, (%Iterate.impl_witness.0fc) [concrete]
// CHECK:STDOUT:   %r.patt.9c0: %pattern_type.7dd = ref_binding_pattern r [concrete]
// CHECK:STDOUT:   %r.param_patt.992: %pattern_type.7dd = var_param_pattern %r.patt.9c0 [concrete]
// CHECK:STDOUT:   %Iterate.facet.012: %Iterate.type = facet_value %DiffMutableRange, (%Iterate.impl_witness.0fb) [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.ff0: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.012) [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.8fc: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.012) [concrete]
// CHECK:STDOUT:   %.6cf1: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.ff0, %Iterate.facet.012 [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.specific_fn.c3b: <specific function> = specific_function %T.as_type.as.Iterate.impl.NewCursor.573, @T.as_type.as.Iterate.impl.NewCursor(%facet_value.6c7) [concrete]
// CHECK:STDOUT:   %.d8c: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.8fc, %Iterate.facet.012 [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.specific_fn.fb4: <specific function> = specific_function %T.as_type.as.Iterate.impl.Next.288, @T.as_type.as.Iterate.impl.Next(%facet_value.6c7) [concrete]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.9e00: <specific function> = specific_function %SelfDestruct, @SelfDestruct(%Destroy.facet.4da) [concrete]
// CHECK:STDOUT:   %Iterate.facet.151: %Iterate.type = facet_value %DiffConstRange, (%Iterate.impl_witness.0fc) [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.cf5: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.151) [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.e49: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.151) [concrete]
// CHECK:STDOUT:   %.776: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.cf5, %Iterate.facet.151 [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.specific_fn.ba8: <specific function> = specific_function %T.as_type.as.Iterate.impl.NewCursor.c96, @T.as_type.as.Iterate.impl.NewCursor(%facet_value.047) [concrete]
// CHECK:STDOUT:   %.ee1: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.e49, %Iterate.facet.151 [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.specific_fn.3dc: <specific function> = specific_function %T.as_type.as.Iterate.impl.Next.264, @T.as_type.as.Iterate.impl.Next(%facet_value.047) [concrete]
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
// CHECK:STDOUT:   %Core.import_ref.e86: @DefaultOptionalStorage.%DefaultOptionalStorage.elem.1 (%DefaultOptionalStorage.elem.ff3) = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [concrete = %field_decl.bce]
// CHECK:STDOUT:   %Core.import_ref.9ce: @DefaultOptionalStorage.%DefaultOptionalStorage.elem.2 (%DefaultOptionalStorage.elem.4a2) = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [concrete = %field_decl.dbe]
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
// CHECK:STDOUT:   %Core.import_ref.b0d = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %field_decl.dbe: @DefaultOptionalStorage.%DefaultOptionalStorage.elem.2 (%DefaultOptionalStorage.elem.4a2) = field_decl has_value, element1, %Core.import_ref.b0d in [concrete] {}
// CHECK:STDOUT:   %Core.import_ref.45d = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %field_decl.bce: @DefaultOptionalStorage.%DefaultOptionalStorage.elem.1 (%DefaultOptionalStorage.elem.ff3) = field_decl value, element0, %Core.import_ref.45d in [concrete] {}
// CHECK:STDOUT:   %DiffMutableRange.decl: type = class_decl @DiffMutableRange [concrete = constants.%DiffMutableRange] {} {}
// CHECK:STDOUT:   %DiffConstRange.decl: type = class_decl @DiffConstRange [concrete = constants.%DiffConstRange] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @TestIterate(%R.loc40_17.2: %Iterate_where.type.1f9) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Iterate.lookup_impl_witness: <witness> = lookup_impl_witness %R.loc40_17.1, @Iterate [symbolic = %Iterate.lookup_impl_witness (constants.%Iterate.lookup_impl_witness.b9d)]
// CHECK:STDOUT:   %Iterate.facet.loc44_33.5: %Iterate.type = facet_value %R.as_type.loc40_79.1, (%Iterate.lookup_impl_witness) [symbolic = %Iterate.facet.loc44_33.5 (constants.%Iterate.facet.8f9)]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.loc44_33.5) [symbolic = %Iterate.WithSelf.NewCursor.type (constants.%Iterate.WithSelf.NewCursor.type.0c9)]
// CHECK:STDOUT:   %.loc44_33.26: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type, %Iterate.facet.loc44_33.5 [symbolic = %.loc44_33.26 (constants.%.3d4)]
// CHECK:STDOUT:   %impl.elem2.loc44_33.2: @TestIterate.%.loc44_33.26 (%.3d4) = impl_witness_access %Iterate.lookup_impl_witness, element2 [symbolic = %impl.elem2.loc44_33.2 (constants.%impl.elem2.53f)]
// CHECK:STDOUT:   %specific_impl_fn.loc44_33.3: <specific function> = specific_impl_function %impl.elem2.loc44_33.2, @Iterate.WithSelf.NewCursor(%Iterate.facet.loc44_33.5) [symbolic = %specific_impl_fn.loc44_33.3 (constants.%specific_impl_fn.b76)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.loc44_33.5) [symbolic = %Iterate.WithSelf.Next.type (constants.%Iterate.WithSelf.Next.type.84e)]
// CHECK:STDOUT:   %.loc44_33.27: type = fn_type_with_self_type %Iterate.WithSelf.Next.type, %Iterate.facet.loc44_33.5 [symbolic = %.loc44_33.27 (constants.%.73c)]
// CHECK:STDOUT:   %impl.elem3.loc44_33.2: @TestIterate.%.loc44_33.27 (%.73c) = impl_witness_access %Iterate.lookup_impl_witness, element3 [symbolic = %impl.elem3.loc44_33.2 (constants.%impl.elem3.015)]
// CHECK:STDOUT:   %specific_impl_fn.loc44_33.4: <specific function> = specific_impl_function %impl.elem3.loc44_33.2, @Iterate.WithSelf.Next(%Iterate.facet.loc44_33.5) [symbolic = %specific_impl_fn.loc44_33.4 (constants.%specific_impl_fn.f44)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%r.param: ref @TestIterate.%R.as_type.loc40_79.1 (%R.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %i.patt: %pattern_type.149 = value_binding_pattern i [concrete = constants.%i.patt]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %r.ref: ref @TestIterate.%R.as_type.loc40_79.1 (%R.as_type) = name_ref r, %r
// CHECK:STDOUT:     %impl.elem2.loc44_33.1: @TestIterate.%.loc44_33.26 (%.3d4) = impl_witness_access constants.%Iterate.lookup_impl_witness.b9d, element2 [symbolic = %impl.elem2.loc44_33.2 (constants.%impl.elem2.53f)]
// CHECK:STDOUT:     %bound_method.loc44_33.1: <bound method> = bound_method %r.ref, %impl.elem2.loc44_33.1
// CHECK:STDOUT:     %Iterate.facet.loc44_33.1: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.b9d) [symbolic = %Iterate.facet.loc44_33.5 (constants.%Iterate.facet.8f9)]
// CHECK:STDOUT:     %.loc44_33.1: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc44_33.1 [symbolic = %Iterate.facet.loc44_33.5 (constants.%Iterate.facet.8f9)]
// CHECK:STDOUT:     %Iterate.facet.loc44_33.2: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.b9d) [symbolic = %Iterate.facet.loc44_33.5 (constants.%Iterate.facet.8f9)]
// CHECK:STDOUT:     %.loc44_33.2: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc44_33.2 [symbolic = %Iterate.facet.loc44_33.5 (constants.%Iterate.facet.8f9)]
// CHECK:STDOUT:     %specific_impl_fn.loc44_33.1: <specific function> = specific_impl_function %impl.elem2.loc44_33.1, @Iterate.WithSelf.NewCursor(constants.%Iterate.facet.8f9) [symbolic = %specific_impl_fn.loc44_33.3 (constants.%specific_impl_fn.b76)]
// CHECK:STDOUT:     %bound_method.loc44_33.2: <bound method> = bound_method %r.ref, %specific_impl_fn.loc44_33.1
// CHECK:STDOUT:     %var: ref @TestIterate.%as_type (%as_type.240) = var_storage invalid
// CHECK:STDOUT:     %.loc44_32.1: @TestIterate.%R.as_type.loc40_79.1 (%R.as_type) = acquire_value %r.ref
// CHECK:STDOUT:     %Iterate.WithSelf.NewCursor.call: init @TestIterate.%as_type (%as_type.240) to %var = call %bound_method.loc44_33.2(%.loc44_32.1)
// CHECK:STDOUT:     assign %var, %Iterate.WithSelf.NewCursor.call
// CHECK:STDOUT:     br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.next:
// CHECK:STDOUT:     %impl.elem3.loc44_33.1: @TestIterate.%.loc44_33.27 (%.73c) = impl_witness_access constants.%Iterate.lookup_impl_witness.b9d, element3 [symbolic = %impl.elem3.loc44_33.2 (constants.%impl.elem3.015)]
// CHECK:STDOUT:     %bound_method.loc44_33.3: <bound method> = bound_method %r.ref, %impl.elem3.loc44_33.1
// CHECK:STDOUT:     %Iterate.facet.loc44_33.3: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.b9d) [symbolic = %Iterate.facet.loc44_33.5 (constants.%Iterate.facet.8f9)]
// CHECK:STDOUT:     %.loc44_33.3: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc44_33.3 [symbolic = %Iterate.facet.loc44_33.5 (constants.%Iterate.facet.8f9)]
// CHECK:STDOUT:     %Iterate.facet.loc44_33.4: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.b9d) [symbolic = %Iterate.facet.loc44_33.5 (constants.%Iterate.facet.8f9)]
// CHECK:STDOUT:     %.loc44_33.4: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc44_33.4 [symbolic = %Iterate.facet.loc44_33.5 (constants.%Iterate.facet.8f9)]
// CHECK:STDOUT:     %.loc44_33.5: %Destroy.type = converted constants.%as_type.240, constants.%impl.elem1.782 [symbolic = %impl.elem1 (constants.%impl.elem1.782)]
// CHECK:STDOUT:     %.loc44_33.6: %Destroy.type = converted constants.%as_type.240, constants.%impl.elem1.782 [symbolic = %impl.elem1 (constants.%impl.elem1.782)]
// CHECK:STDOUT:     %specific_impl_fn.loc44_33.2: <specific function> = specific_impl_function %impl.elem3.loc44_33.1, @Iterate.WithSelf.Next(constants.%Iterate.facet.8f9) [symbolic = %specific_impl_fn.loc44_33.4 (constants.%specific_impl_fn.f44)]
// CHECK:STDOUT:     %bound_method.loc44_33.4: <bound method> = bound_method %r.ref, %specific_impl_fn.loc44_33.2
// CHECK:STDOUT:     %.loc44_33.7: ref %Optional.250 = temporary_storage
// CHECK:STDOUT:     %.loc44_32.2: @TestIterate.%R.as_type.loc40_79.1 (%R.as_type) = acquire_value %r.ref
// CHECK:STDOUT:     %Iterate.WithSelf.Next.call: init %Optional.250 to %.loc44_33.7 = call %bound_method.loc44_33.4(%.loc44_32.2, %var)
// CHECK:STDOUT:     %.loc44_33.8: ref %Optional.250 = temporary %.loc44_33.7, %Iterate.WithSelf.Next.call
// CHECK:STDOUT:     %.loc44_33.9: %Optional.HasValue.type.7a2 = specific_constant imports.%Core.import_ref.c82, @Optional(constants.%OptionalStorage.facet.f87) [concrete = constants.%Optional.HasValue.0b9]
// CHECK:STDOUT:     %HasValue.ref: %Optional.HasValue.type.7a2 = name_ref HasValue, %.loc44_33.9 [concrete = constants.%Optional.HasValue.0b9]
// CHECK:STDOUT:     %Optional.HasValue.bound: <bound method> = bound_method %.loc44_33.8, %HasValue.ref
// CHECK:STDOUT:     %Optional.HasValue.specific_fn: <specific function> = specific_function %HasValue.ref, @Optional.HasValue(constants.%OptionalStorage.facet.f87) [concrete = constants.%Optional.HasValue.specific_fn.366]
// CHECK:STDOUT:     %bound_method.loc44_33.5: <bound method> = bound_method %.loc44_33.8, %Optional.HasValue.specific_fn
// CHECK:STDOUT:     %.loc44_33.10: %Optional.250 = acquire_value %.loc44_33.8
// CHECK:STDOUT:     %Optional.HasValue.call: init bool = call %bound_method.loc44_33.5(%.loc44_33.10)
// CHECK:STDOUT:     %.loc44_33.11: bool = value_of_initializer %Optional.HasValue.call
// CHECK:STDOUT:     %.loc44_33.12: bool = converted %Optional.HasValue.call, %.loc44_33.11
// CHECK:STDOUT:     if %.loc44_33.12 br !for.body else br !for.done
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.body:
// CHECK:STDOUT:     %.loc44_33.13: %Optional.Get.type.fa0 = specific_constant imports.%Core.import_ref.29e, @Optional(constants.%OptionalStorage.facet.f87) [concrete = constants.%Optional.Get.f6a]
// CHECK:STDOUT:     %Get.ref: %Optional.Get.type.fa0 = name_ref Get, %.loc44_33.13 [concrete = constants.%Optional.Get.f6a]
// CHECK:STDOUT:     %Optional.Get.bound: <bound method> = bound_method %.loc44_33.8, %Get.ref
// CHECK:STDOUT:     %Optional.Get.specific_fn: <specific function> = specific_function %Get.ref, @Optional.Get(constants.%OptionalStorage.facet.f87) [concrete = constants.%Optional.Get.specific_fn.29e]
// CHECK:STDOUT:     %bound_method.loc44_33.6: <bound method> = bound_method %.loc44_33.8, %Optional.Get.specific_fn
// CHECK:STDOUT:     %.loc44_33.14: ref %DiffValueType = temporary_storage
// CHECK:STDOUT:     %.loc44_33.15: %Optional.250 = acquire_value %.loc44_33.8
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
// CHECK:STDOUT:     %Destroy.facet.loc44_33.1: %Destroy.type = facet_value constants.%DiffValueType, (constants.%custom_witness.5760f4.1, constants.%DestroyImpl.impl_witness.c9a) [concrete = constants.%Destroy.facet.91b]
// CHECK:STDOUT:     %.loc44_33.18: %Destroy.type = converted constants.%DiffValueType, %Destroy.facet.loc44_33.1 [concrete = constants.%Destroy.facet.91b]
// CHECK:STDOUT:     %Destroy.facet.loc44_33.2: %Destroy.type = facet_value constants.%DiffValueType, (constants.%custom_witness.5760f4.1, constants.%DestroyImpl.impl_witness.c9a) [concrete = constants.%Destroy.facet.91b]
// CHECK:STDOUT:     %.loc44_33.19: %Destroy.type = converted constants.%DiffValueType, %Destroy.facet.loc44_33.2 [concrete = constants.%Destroy.facet.91b]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %SelfDestruct.call.loc44_33.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc44_33.16)
// CHECK:STDOUT:     %Destroy.facet.loc44_33.3: %Destroy.type = facet_value constants.%Optional.250, (constants.%custom_witness.b80ef6.7, constants.%DestroyImpl.impl_witness.dec) [concrete = constants.%Destroy.facet.ba0]
// CHECK:STDOUT:     %.loc44_33.20: %Destroy.type = converted constants.%Optional.250, %Destroy.facet.loc44_33.3 [concrete = constants.%Destroy.facet.ba0]
// CHECK:STDOUT:     %Destroy.facet.loc44_33.4: %Destroy.type = facet_value constants.%Optional.250, (constants.%custom_witness.b80ef6.7, constants.%DestroyImpl.impl_witness.dec) [concrete = constants.%Destroy.facet.ba0]
// CHECK:STDOUT:     %.loc44_33.21: %Destroy.type = converted constants.%Optional.250, %Destroy.facet.loc44_33.4 [concrete = constants.%Destroy.facet.ba0]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %SelfDestruct.call.loc44_33.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc44_33.8)
// CHECK:STDOUT:     br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.done:
// CHECK:STDOUT:     %Destroy.facet.loc44_33.5: %Destroy.type = facet_value constants.%Optional.250, (constants.%custom_witness.b80ef6.7, constants.%DestroyImpl.impl_witness.dec) [concrete = constants.%Destroy.facet.ba0]
// CHECK:STDOUT:     %.loc44_33.22: %Destroy.type = converted constants.%Optional.250, %Destroy.facet.loc44_33.5 [concrete = constants.%Destroy.facet.ba0]
// CHECK:STDOUT:     %Destroy.facet.loc44_33.6: %Destroy.type = facet_value constants.%Optional.250, (constants.%custom_witness.b80ef6.7, constants.%DestroyImpl.impl_witness.dec) [concrete = constants.%Destroy.facet.ba0]
// CHECK:STDOUT:     %.loc44_33.23: %Destroy.type = converted constants.%Optional.250, %Destroy.facet.loc44_33.6 [concrete = constants.%Destroy.facet.ba0]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %SelfDestruct.call.loc44_33.3: init %empty_tuple.type = call %SelfDestruct.specific_fn.3(%.loc44_33.8)
// CHECK:STDOUT:     %.loc44_33.24: %Destroy.type = converted constants.%as_type.240, constants.%impl.elem1.782 [symbolic = %impl.elem1 (constants.%impl.elem1.782)]
// CHECK:STDOUT:     %.loc44_33.25: %Destroy.type = converted constants.%as_type.240, constants.%impl.elem1.782 [symbolic = %impl.elem1 (constants.%impl.elem1.782)]
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
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc44_33.2(%self.param: ref %MaybeUnformed.4a6) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc44_33.3(%self.param: ref bool) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc44_33.4(%self.param: ref %struct_type.value.has_value.b29) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc44_33.1: ref bool = struct_access %self.param, element1
// CHECK:STDOUT:   %Destroy.facet.loc44_33.1: %Destroy.type = facet_value bool, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.778) [concrete = constants.%Destroy.facet.80c]
// CHECK:STDOUT:   %.loc44_33.2: %Destroy.type = converted bool, %Destroy.facet.loc44_33.1 [concrete = constants.%Destroy.facet.80c]
// CHECK:STDOUT:   %Destroy.facet.loc44_33.2: %Destroy.type = facet_value bool, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.778) [concrete = constants.%Destroy.facet.80c]
// CHECK:STDOUT:   %.loc44_33.3: %Destroy.type = converted bool, %Destroy.facet.loc44_33.2 [concrete = constants.%Destroy.facet.80c]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc44_33.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc44_33.1)
// CHECK:STDOUT:   %.loc44_33.4: ref %MaybeUnformed.4a6 = struct_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc44_33.3: %Destroy.type = facet_value constants.%MaybeUnformed.4a6, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.6f4) [concrete = constants.%Destroy.facet.0a4]
// CHECK:STDOUT:   %.loc44_33.5: %Destroy.type = converted constants.%MaybeUnformed.4a6, %Destroy.facet.loc44_33.3 [concrete = constants.%Destroy.facet.0a4]
// CHECK:STDOUT:   %Destroy.facet.loc44_33.4: %Destroy.type = facet_value constants.%MaybeUnformed.4a6, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.6f4) [concrete = constants.%Destroy.facet.0a4]
// CHECK:STDOUT:   %.loc44_33.6: %Destroy.type = converted constants.%MaybeUnformed.4a6, %Destroy.facet.loc44_33.4 [concrete = constants.%Destroy.facet.0a4]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc44_33.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc44_33.4)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc44_33.5(%self.param: ref %DefaultOptionalStorage.7fa) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %has_value.ref: %DefaultOptionalStorage.elem.54f = name_ref has_value, imports.%Core.import_ref.9ce [concrete = imports.%field_decl.dbe]
// CHECK:STDOUT:   %.loc44_33.1: ref bool = class_element_access %self.param, element1
// CHECK:STDOUT:   %Destroy.facet.loc44_33.1: %Destroy.type = facet_value bool, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.778) [concrete = constants.%Destroy.facet.80c]
// CHECK:STDOUT:   %.loc44_33.2: %Destroy.type = converted bool, %Destroy.facet.loc44_33.1 [concrete = constants.%Destroy.facet.80c]
// CHECK:STDOUT:   %Destroy.facet.loc44_33.2: %Destroy.type = facet_value bool, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.778) [concrete = constants.%Destroy.facet.80c]
// CHECK:STDOUT:   %.loc44_33.3: %Destroy.type = converted bool, %Destroy.facet.loc44_33.2 [concrete = constants.%Destroy.facet.80c]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc44_33.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc44_33.1)
// CHECK:STDOUT:   %value.ref: %DefaultOptionalStorage.elem.fb7 = name_ref value, imports.%Core.import_ref.e86 [concrete = imports.%field_decl.bce]
// CHECK:STDOUT:   %.loc44_33.4: ref %MaybeUnformed.4a6 = class_element_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc44_33.3: %Destroy.type = facet_value constants.%MaybeUnformed.4a6, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.6f4) [concrete = constants.%Destroy.facet.0a4]
// CHECK:STDOUT:   %.loc44_33.5: %Destroy.type = converted constants.%MaybeUnformed.4a6, %Destroy.facet.loc44_33.3 [concrete = constants.%Destroy.facet.0a4]
// CHECK:STDOUT:   %Destroy.facet.loc44_33.4: %Destroy.type = facet_value constants.%MaybeUnformed.4a6, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.6f4) [concrete = constants.%Destroy.facet.0a4]
// CHECK:STDOUT:   %.loc44_33.6: %Destroy.type = converted constants.%MaybeUnformed.4a6, %Destroy.facet.loc44_33.4 [concrete = constants.%Destroy.facet.0a4]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc44_33.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc44_33.4)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc44_33.6(%self.param: ref %Optional.250) {
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
// CHECK:STDOUT:   %impl.elem2.loc59: %.6cf1 = impl_witness_access constants.%Iterate.impl_witness.0fb, element2 [concrete = constants.%T.as_type.as.Iterate.impl.NewCursor.573]
// CHECK:STDOUT:   %bound_method.loc59_33.1: <bound method> = bound_method %m.ref, %impl.elem2.loc59
// CHECK:STDOUT:   %facet_value.loc59_33.1: %CppRange_where.type.d682d6.1 = facet_value constants.%DiffMutableRange, (constants.%custom_witness.aa8, constants.%custom_witness.3815f0.1, constants.%DestroyImpl.impl_witness.03b, constants.%custom_witness.392, constants.%custom_witness.1c9, constants.%custom_witness.b32, constants.%custom_witness.5760f4.1, constants.%DestroyImpl.impl_witness.c9a, constants.%custom_witness.d0da43.1, constants.%custom_witness.3815f0.2, constants.%DestroyImpl.impl_witness.508) [concrete = constants.%facet_value.6c7]
// CHECK:STDOUT:   %.loc59_33.1: %CppRange_where.type.d682d6.1 = converted constants.%DiffMutableRange, %facet_value.loc59_33.1 [concrete = constants.%facet_value.6c7]
// CHECK:STDOUT:   %facet_value.loc59_33.2: %CppRange_where.type.d682d6.1 = facet_value constants.%DiffMutableRange, (constants.%custom_witness.aa8, constants.%custom_witness.3815f0.1, constants.%DestroyImpl.impl_witness.03b, constants.%custom_witness.392, constants.%custom_witness.1c9, constants.%custom_witness.b32, constants.%custom_witness.5760f4.1, constants.%DestroyImpl.impl_witness.c9a, constants.%custom_witness.d0da43.1, constants.%custom_witness.3815f0.2, constants.%DestroyImpl.impl_witness.508) [concrete = constants.%facet_value.6c7]
// CHECK:STDOUT:   %.loc59_33.2: %CppRange_where.type.d682d6.1 = converted constants.%DiffMutableRange, %facet_value.loc59_33.2 [concrete = constants.%facet_value.6c7]
// CHECK:STDOUT:   %specific_fn.loc59_33.1: <specific function> = specific_function %impl.elem2.loc59, @T.as_type.as.Iterate.impl.NewCursor(constants.%facet_value.6c7) [concrete = constants.%T.as_type.as.Iterate.impl.NewCursor.specific_fn.c3b]
// CHECK:STDOUT:   %bound_method.loc59_33.2: <bound method> = bound_method %m.ref, %specific_fn.loc59_33.1
// CHECK:STDOUT:   %var.loc59: ref %tuple.type.b18 = var_storage invalid
// CHECK:STDOUT:   %.loc59_32.1: %DiffMutableRange = acquire_value %m.ref
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.call.loc59: init %tuple.type.b18 to %var.loc59 = call %bound_method.loc59_33.2(%.loc59_32.1)
// CHECK:STDOUT:   assign %var.loc59, %T.as_type.as.Iterate.impl.NewCursor.call.loc59
// CHECK:STDOUT:   br !for.next.loc59
// CHECK:STDOUT:
// CHECK:STDOUT: !for.next.loc59:
// CHECK:STDOUT:   %impl.elem3.loc59: %.d8c = impl_witness_access constants.%Iterate.impl_witness.0fb, element3 [concrete = constants.%T.as_type.as.Iterate.impl.Next.288]
// CHECK:STDOUT:   %bound_method.loc59_33.3: <bound method> = bound_method %m.ref, %impl.elem3.loc59
// CHECK:STDOUT:   %facet_value.loc59_33.3: %CppRange_where.type.d682d6.1 = facet_value constants.%DiffMutableRange, (constants.%custom_witness.aa8, constants.%custom_witness.3815f0.1, constants.%DestroyImpl.impl_witness.03b, constants.%custom_witness.392, constants.%custom_witness.1c9, constants.%custom_witness.b32, constants.%custom_witness.5760f4.1, constants.%DestroyImpl.impl_witness.c9a, constants.%custom_witness.d0da43.1, constants.%custom_witness.3815f0.2, constants.%DestroyImpl.impl_witness.508) [concrete = constants.%facet_value.6c7]
// CHECK:STDOUT:   %.loc59_33.3: %CppRange_where.type.d682d6.1 = converted constants.%DiffMutableRange, %facet_value.loc59_33.3 [concrete = constants.%facet_value.6c7]
// CHECK:STDOUT:   %facet_value.loc59_33.4: %CppRange_where.type.d682d6.1 = facet_value constants.%DiffMutableRange, (constants.%custom_witness.aa8, constants.%custom_witness.3815f0.1, constants.%DestroyImpl.impl_witness.03b, constants.%custom_witness.392, constants.%custom_witness.1c9, constants.%custom_witness.b32, constants.%custom_witness.5760f4.1, constants.%DestroyImpl.impl_witness.c9a, constants.%custom_witness.d0da43.1, constants.%custom_witness.3815f0.2, constants.%DestroyImpl.impl_witness.508) [concrete = constants.%facet_value.6c7]
// CHECK:STDOUT:   %.loc59_33.4: %CppRange_where.type.d682d6.1 = converted constants.%DiffMutableRange, %facet_value.loc59_33.4 [concrete = constants.%facet_value.6c7]
// CHECK:STDOUT:   %specific_fn.loc59_33.2: <specific function> = specific_function %impl.elem3.loc59, @T.as_type.as.Iterate.impl.Next(constants.%facet_value.6c7) [concrete = constants.%T.as_type.as.Iterate.impl.Next.specific_fn.fb4]
// CHECK:STDOUT:   %bound_method.loc59_33.4: <bound method> = bound_method %m.ref, %specific_fn.loc59_33.2
// CHECK:STDOUT:   %.loc59_33.5: ref %Optional.250 = temporary_storage
// CHECK:STDOUT:   %.loc59_32.2: %DiffMutableRange = acquire_value %m.ref
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.call.loc59: init %Optional.250 to %.loc59_33.5 = call %bound_method.loc59_33.4(%.loc59_32.2, %var.loc59)
// CHECK:STDOUT:   %.loc59_33.6: ref %Optional.250 = temporary %.loc59_33.5, %T.as_type.as.Iterate.impl.Next.call.loc59
// CHECK:STDOUT:   %.loc59_33.7: %Optional.HasValue.type.7a2 = specific_constant imports.%Core.import_ref.c82, @Optional(constants.%OptionalStorage.facet.f87) [concrete = constants.%Optional.HasValue.0b9]
// CHECK:STDOUT:   %HasValue.ref.loc59: %Optional.HasValue.type.7a2 = name_ref HasValue, %.loc59_33.7 [concrete = constants.%Optional.HasValue.0b9]
// CHECK:STDOUT:   %Optional.HasValue.bound.loc59: <bound method> = bound_method %.loc59_33.6, %HasValue.ref.loc59
// CHECK:STDOUT:   %Optional.HasValue.specific_fn.loc59: <specific function> = specific_function %HasValue.ref.loc59, @Optional.HasValue(constants.%OptionalStorage.facet.f87) [concrete = constants.%Optional.HasValue.specific_fn.366]
// CHECK:STDOUT:   %bound_method.loc59_33.5: <bound method> = bound_method %.loc59_33.6, %Optional.HasValue.specific_fn.loc59
// CHECK:STDOUT:   %.loc59_33.8: %Optional.250 = acquire_value %.loc59_33.6
// CHECK:STDOUT:   %Optional.HasValue.call.loc59: init bool = call %bound_method.loc59_33.5(%.loc59_33.8)
// CHECK:STDOUT:   %.loc59_33.9: bool = value_of_initializer %Optional.HasValue.call.loc59
// CHECK:STDOUT:   %.loc59_33.10: bool = converted %Optional.HasValue.call.loc59, %.loc59_33.9
// CHECK:STDOUT:   if %.loc59_33.10 br !for.body.loc59 else br !for.done.loc59
// CHECK:STDOUT:
// CHECK:STDOUT: !for.body.loc59:
// CHECK:STDOUT:   %.loc59_33.11: %Optional.Get.type.fa0 = specific_constant imports.%Core.import_ref.29e, @Optional(constants.%OptionalStorage.facet.f87) [concrete = constants.%Optional.Get.f6a]
// CHECK:STDOUT:   %Get.ref.loc59: %Optional.Get.type.fa0 = name_ref Get, %.loc59_33.11 [concrete = constants.%Optional.Get.f6a]
// CHECK:STDOUT:   %Optional.Get.bound.loc59: <bound method> = bound_method %.loc59_33.6, %Get.ref.loc59
// CHECK:STDOUT:   %Optional.Get.specific_fn.loc59: <specific function> = specific_function %Get.ref.loc59, @Optional.Get(constants.%OptionalStorage.facet.f87) [concrete = constants.%Optional.Get.specific_fn.29e]
// CHECK:STDOUT:   %bound_method.loc59_33.6: <bound method> = bound_method %.loc59_33.6, %Optional.Get.specific_fn.loc59
// CHECK:STDOUT:   %.loc59_33.12: ref %DiffValueType = temporary_storage
// CHECK:STDOUT:   %.loc59_33.13: %Optional.250 = acquire_value %.loc59_33.6
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
// CHECK:STDOUT:   %Destroy.facet.loc59_33.1: %Destroy.type = facet_value constants.%DiffValueType, (constants.%custom_witness.5760f4.1, constants.%DestroyImpl.impl_witness.c9a) [concrete = constants.%Destroy.facet.91b]
// CHECK:STDOUT:   %.loc59_33.16: %Destroy.type = converted constants.%DiffValueType, %Destroy.facet.loc59_33.1 [concrete = constants.%Destroy.facet.91b]
// CHECK:STDOUT:   %Destroy.facet.loc59_33.2: %Destroy.type = facet_value constants.%DiffValueType, (constants.%custom_witness.5760f4.1, constants.%DestroyImpl.impl_witness.c9a) [concrete = constants.%Destroy.facet.91b]
// CHECK:STDOUT:   %.loc59_33.17: %Destroy.type = converted constants.%DiffValueType, %Destroy.facet.loc59_33.2 [concrete = constants.%Destroy.facet.91b]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc59_33.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc59_33.14)
// CHECK:STDOUT:   %Destroy.facet.loc59_33.3: %Destroy.type = facet_value constants.%Optional.250, (constants.%custom_witness.b80ef6.7, constants.%DestroyImpl.impl_witness.dec) [concrete = constants.%Destroy.facet.ba0]
// CHECK:STDOUT:   %.loc59_33.18: %Destroy.type = converted constants.%Optional.250, %Destroy.facet.loc59_33.3 [concrete = constants.%Destroy.facet.ba0]
// CHECK:STDOUT:   %Destroy.facet.loc59_33.4: %Destroy.type = facet_value constants.%Optional.250, (constants.%custom_witness.b80ef6.7, constants.%DestroyImpl.impl_witness.dec) [concrete = constants.%Destroy.facet.ba0]
// CHECK:STDOUT:   %.loc59_33.19: %Destroy.type = converted constants.%Optional.250, %Destroy.facet.loc59_33.4 [concrete = constants.%Destroy.facet.ba0]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc59_33.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc59_33.6)
// CHECK:STDOUT:   br !for.next.loc59
// CHECK:STDOUT:
// CHECK:STDOUT: !for.done.loc59:
// CHECK:STDOUT:   %Destroy.facet.loc59_33.5: %Destroy.type = facet_value constants.%Optional.250, (constants.%custom_witness.b80ef6.7, constants.%DestroyImpl.impl_witness.dec) [concrete = constants.%Destroy.facet.ba0]
// CHECK:STDOUT:   %.loc59_33.20: %Destroy.type = converted constants.%Optional.250, %Destroy.facet.loc59_33.5 [concrete = constants.%Destroy.facet.ba0]
// CHECK:STDOUT:   %Destroy.facet.loc59_33.6: %Destroy.type = facet_value constants.%Optional.250, (constants.%custom_witness.b80ef6.7, constants.%DestroyImpl.impl_witness.dec) [concrete = constants.%Destroy.facet.ba0]
// CHECK:STDOUT:   %.loc59_33.21: %Destroy.type = converted constants.%Optional.250, %Destroy.facet.loc59_33.6 [concrete = constants.%Destroy.facet.ba0]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc59_33.3: init %empty_tuple.type = call %SelfDestruct.specific_fn.3(%.loc59_33.6)
// CHECK:STDOUT:   %Destroy.facet.loc59_33.7: %Destroy.type = facet_value constants.%tuple.type.b18, (constants.%custom_witness.b80ef6.8, constants.%DestroyImpl.impl_witness.c40) [concrete = constants.%Destroy.facet.4da]
// CHECK:STDOUT:   %.loc59_33.22: %Destroy.type = converted constants.%tuple.type.b18, %Destroy.facet.loc59_33.7 [concrete = constants.%Destroy.facet.4da]
// CHECK:STDOUT:   %Destroy.facet.loc59_33.8: %Destroy.type = facet_value constants.%tuple.type.b18, (constants.%custom_witness.b80ef6.8, constants.%DestroyImpl.impl_witness.c40) [concrete = constants.%Destroy.facet.4da]
// CHECK:STDOUT:   %.loc59_33.23: %Destroy.type = converted constants.%tuple.type.b18, %Destroy.facet.loc59_33.8 [concrete = constants.%Destroy.facet.4da]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc59_33.4: init %empty_tuple.type = call %SelfDestruct.specific_fn.4(%var.loc59)
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %i.patt.loc63: %pattern_type.149 = value_binding_pattern i [concrete = constants.%i.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c.ref: %DiffConstRange = name_ref c, %c
// CHECK:STDOUT:   %impl.elem2.loc63: %.776 = impl_witness_access constants.%Iterate.impl_witness.0fc, element2 [concrete = constants.%T.as_type.as.Iterate.impl.NewCursor.c96]
// CHECK:STDOUT:   %bound_method.loc63_33.1: <bound method> = bound_method %c.ref, %impl.elem2.loc63
// CHECK:STDOUT:   %facet_value.loc63_33.1: %CppRange_where.type.d682d6.1 = facet_value constants.%DiffConstRange, (constants.%custom_witness.7b4, constants.%custom_witness.3d0529.1, constants.%DestroyImpl.impl_witness.bd5, constants.%custom_witness.0c0, constants.%custom_witness.ffb, constants.%custom_witness.fc2, constants.%custom_witness.5760f4.1, constants.%DestroyImpl.impl_witness.c9a, constants.%custom_witness.d0da43.1, constants.%custom_witness.3d0529.2, constants.%DestroyImpl.impl_witness.780) [concrete = constants.%facet_value.047]
// CHECK:STDOUT:   %.loc63_33.1: %CppRange_where.type.d682d6.1 = converted constants.%DiffConstRange, %facet_value.loc63_33.1 [concrete = constants.%facet_value.047]
// CHECK:STDOUT:   %facet_value.loc63_33.2: %CppRange_where.type.d682d6.1 = facet_value constants.%DiffConstRange, (constants.%custom_witness.7b4, constants.%custom_witness.3d0529.1, constants.%DestroyImpl.impl_witness.bd5, constants.%custom_witness.0c0, constants.%custom_witness.ffb, constants.%custom_witness.fc2, constants.%custom_witness.5760f4.1, constants.%DestroyImpl.impl_witness.c9a, constants.%custom_witness.d0da43.1, constants.%custom_witness.3d0529.2, constants.%DestroyImpl.impl_witness.780) [concrete = constants.%facet_value.047]
// CHECK:STDOUT:   %.loc63_33.2: %CppRange_where.type.d682d6.1 = converted constants.%DiffConstRange, %facet_value.loc63_33.2 [concrete = constants.%facet_value.047]
// CHECK:STDOUT:   %specific_fn.loc63_33.1: <specific function> = specific_function %impl.elem2.loc63, @T.as_type.as.Iterate.impl.NewCursor(constants.%facet_value.047) [concrete = constants.%T.as_type.as.Iterate.impl.NewCursor.specific_fn.ba8]
// CHECK:STDOUT:   %bound_method.loc63_33.2: <bound method> = bound_method %c.ref, %specific_fn.loc63_33.1
// CHECK:STDOUT:   %var.loc63: ref %tuple.type.df5 = var_storage invalid
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.call.loc63: init %tuple.type.df5 to %var.loc63 = call %bound_method.loc63_33.2(%c.ref)
// CHECK:STDOUT:   assign %var.loc63, %T.as_type.as.Iterate.impl.NewCursor.call.loc63
// CHECK:STDOUT:   br !for.next.loc63
// CHECK:STDOUT:
// CHECK:STDOUT: !for.next.loc63:
// CHECK:STDOUT:   %impl.elem3.loc63: %.ee1 = impl_witness_access constants.%Iterate.impl_witness.0fc, element3 [concrete = constants.%T.as_type.as.Iterate.impl.Next.264]
// CHECK:STDOUT:   %bound_method.loc63_33.3: <bound method> = bound_method %c.ref, %impl.elem3.loc63
// CHECK:STDOUT:   %facet_value.loc63_33.3: %CppRange_where.type.d682d6.1 = facet_value constants.%DiffConstRange, (constants.%custom_witness.7b4, constants.%custom_witness.3d0529.1, constants.%DestroyImpl.impl_witness.bd5, constants.%custom_witness.0c0, constants.%custom_witness.ffb, constants.%custom_witness.fc2, constants.%custom_witness.5760f4.1, constants.%DestroyImpl.impl_witness.c9a, constants.%custom_witness.d0da43.1, constants.%custom_witness.3d0529.2, constants.%DestroyImpl.impl_witness.780) [concrete = constants.%facet_value.047]
// CHECK:STDOUT:   %.loc63_33.3: %CppRange_where.type.d682d6.1 = converted constants.%DiffConstRange, %facet_value.loc63_33.3 [concrete = constants.%facet_value.047]
// CHECK:STDOUT:   %facet_value.loc63_33.4: %CppRange_where.type.d682d6.1 = facet_value constants.%DiffConstRange, (constants.%custom_witness.7b4, constants.%custom_witness.3d0529.1, constants.%DestroyImpl.impl_witness.bd5, constants.%custom_witness.0c0, constants.%custom_witness.ffb, constants.%custom_witness.fc2, constants.%custom_witness.5760f4.1, constants.%DestroyImpl.impl_witness.c9a, constants.%custom_witness.d0da43.1, constants.%custom_witness.3d0529.2, constants.%DestroyImpl.impl_witness.780) [concrete = constants.%facet_value.047]
// CHECK:STDOUT:   %.loc63_33.4: %CppRange_where.type.d682d6.1 = converted constants.%DiffConstRange, %facet_value.loc63_33.4 [concrete = constants.%facet_value.047]
// CHECK:STDOUT:   %specific_fn.loc63_33.2: <specific function> = specific_function %impl.elem3.loc63, @T.as_type.as.Iterate.impl.Next(constants.%facet_value.047) [concrete = constants.%T.as_type.as.Iterate.impl.Next.specific_fn.3dc]
// CHECK:STDOUT:   %bound_method.loc63_33.4: <bound method> = bound_method %c.ref, %specific_fn.loc63_33.2
// CHECK:STDOUT:   %.loc63_33.5: ref %Optional.250 = temporary_storage
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.call.loc63: init %Optional.250 to %.loc63_33.5 = call %bound_method.loc63_33.4(%c.ref, %var.loc63)
// CHECK:STDOUT:   %.loc63_33.6: ref %Optional.250 = temporary %.loc63_33.5, %T.as_type.as.Iterate.impl.Next.call.loc63
// CHECK:STDOUT:   %.loc63_33.7: %Optional.HasValue.type.7a2 = specific_constant imports.%Core.import_ref.c82, @Optional(constants.%OptionalStorage.facet.f87) [concrete = constants.%Optional.HasValue.0b9]
// CHECK:STDOUT:   %HasValue.ref.loc63: %Optional.HasValue.type.7a2 = name_ref HasValue, %.loc63_33.7 [concrete = constants.%Optional.HasValue.0b9]
// CHECK:STDOUT:   %Optional.HasValue.bound.loc63: <bound method> = bound_method %.loc63_33.6, %HasValue.ref.loc63
// CHECK:STDOUT:   %Optional.HasValue.specific_fn.loc63: <specific function> = specific_function %HasValue.ref.loc63, @Optional.HasValue(constants.%OptionalStorage.facet.f87) [concrete = constants.%Optional.HasValue.specific_fn.366]
// CHECK:STDOUT:   %bound_method.loc63_33.5: <bound method> = bound_method %.loc63_33.6, %Optional.HasValue.specific_fn.loc63
// CHECK:STDOUT:   %.loc63_33.8: %Optional.250 = acquire_value %.loc63_33.6
// CHECK:STDOUT:   %Optional.HasValue.call.loc63: init bool = call %bound_method.loc63_33.5(%.loc63_33.8)
// CHECK:STDOUT:   %.loc63_33.9: bool = value_of_initializer %Optional.HasValue.call.loc63
// CHECK:STDOUT:   %.loc63_33.10: bool = converted %Optional.HasValue.call.loc63, %.loc63_33.9
// CHECK:STDOUT:   if %.loc63_33.10 br !for.body.loc63 else br !for.done.loc63
// CHECK:STDOUT:
// CHECK:STDOUT: !for.body.loc63:
// CHECK:STDOUT:   %.loc63_33.11: %Optional.Get.type.fa0 = specific_constant imports.%Core.import_ref.29e, @Optional(constants.%OptionalStorage.facet.f87) [concrete = constants.%Optional.Get.f6a]
// CHECK:STDOUT:   %Get.ref.loc63: %Optional.Get.type.fa0 = name_ref Get, %.loc63_33.11 [concrete = constants.%Optional.Get.f6a]
// CHECK:STDOUT:   %Optional.Get.bound.loc63: <bound method> = bound_method %.loc63_33.6, %Get.ref.loc63
// CHECK:STDOUT:   %Optional.Get.specific_fn.loc63: <specific function> = specific_function %Get.ref.loc63, @Optional.Get(constants.%OptionalStorage.facet.f87) [concrete = constants.%Optional.Get.specific_fn.29e]
// CHECK:STDOUT:   %bound_method.loc63_33.6: <bound method> = bound_method %.loc63_33.6, %Optional.Get.specific_fn.loc63
// CHECK:STDOUT:   %.loc63_33.12: ref %DiffValueType = temporary_storage
// CHECK:STDOUT:   %.loc63_33.13: %Optional.250 = acquire_value %.loc63_33.6
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
// CHECK:STDOUT:   %Destroy.facet.loc63_33.1: %Destroy.type = facet_value constants.%DiffValueType, (constants.%custom_witness.5760f4.1, constants.%DestroyImpl.impl_witness.c9a) [concrete = constants.%Destroy.facet.91b]
// CHECK:STDOUT:   %.loc63_33.16: %Destroy.type = converted constants.%DiffValueType, %Destroy.facet.loc63_33.1 [concrete = constants.%Destroy.facet.91b]
// CHECK:STDOUT:   %Destroy.facet.loc63_33.2: %Destroy.type = facet_value constants.%DiffValueType, (constants.%custom_witness.5760f4.1, constants.%DestroyImpl.impl_witness.c9a) [concrete = constants.%Destroy.facet.91b]
// CHECK:STDOUT:   %.loc63_33.17: %Destroy.type = converted constants.%DiffValueType, %Destroy.facet.loc63_33.2 [concrete = constants.%Destroy.facet.91b]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc63_33.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.5(%.loc63_33.14)
// CHECK:STDOUT:   %Destroy.facet.loc63_33.3: %Destroy.type = facet_value constants.%Optional.250, (constants.%custom_witness.b80ef6.7, constants.%DestroyImpl.impl_witness.dec) [concrete = constants.%Destroy.facet.ba0]
// CHECK:STDOUT:   %.loc63_33.18: %Destroy.type = converted constants.%Optional.250, %Destroy.facet.loc63_33.3 [concrete = constants.%Destroy.facet.ba0]
// CHECK:STDOUT:   %Destroy.facet.loc63_33.4: %Destroy.type = facet_value constants.%Optional.250, (constants.%custom_witness.b80ef6.7, constants.%DestroyImpl.impl_witness.dec) [concrete = constants.%Destroy.facet.ba0]
// CHECK:STDOUT:   %.loc63_33.19: %Destroy.type = converted constants.%Optional.250, %Destroy.facet.loc63_33.4 [concrete = constants.%Destroy.facet.ba0]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc63_33.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.6(%.loc63_33.6)
// CHECK:STDOUT:   br !for.next.loc63
// CHECK:STDOUT:
// CHECK:STDOUT: !for.done.loc63:
// CHECK:STDOUT:   %Destroy.facet.loc63_33.5: %Destroy.type = facet_value constants.%Optional.250, (constants.%custom_witness.b80ef6.7, constants.%DestroyImpl.impl_witness.dec) [concrete = constants.%Destroy.facet.ba0]
// CHECK:STDOUT:   %.loc63_33.20: %Destroy.type = converted constants.%Optional.250, %Destroy.facet.loc63_33.5 [concrete = constants.%Destroy.facet.ba0]
// CHECK:STDOUT:   %Destroy.facet.loc63_33.6: %Destroy.type = facet_value constants.%Optional.250, (constants.%custom_witness.b80ef6.7, constants.%DestroyImpl.impl_witness.dec) [concrete = constants.%Destroy.facet.ba0]
// CHECK:STDOUT:   %.loc63_33.21: %Destroy.type = converted constants.%Optional.250, %Destroy.facet.loc63_33.6 [concrete = constants.%Destroy.facet.ba0]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc63_33.3: init %empty_tuple.type = call %SelfDestruct.specific_fn.7(%.loc63_33.6)
// CHECK:STDOUT:   %Destroy.facet.loc63_33.7: %Destroy.type = facet_value constants.%tuple.type.df5, (constants.%custom_witness.b80ef6.9, constants.%DestroyImpl.impl_witness.8c3) [concrete = constants.%Destroy.facet.017]
// CHECK:STDOUT:   %.loc63_33.22: %Destroy.type = converted constants.%tuple.type.df5, %Destroy.facet.loc63_33.7 [concrete = constants.%Destroy.facet.017]
// CHECK:STDOUT:   %Destroy.facet.loc63_33.8: %Destroy.type = facet_value constants.%tuple.type.df5, (constants.%custom_witness.b80ef6.9, constants.%DestroyImpl.impl_witness.8c3) [concrete = constants.%Destroy.facet.017]
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
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.172
// CHECK:STDOUT:   %r.patt.loc40_77.2 => constants.%r.patt.ea6
// CHECK:STDOUT:   %r.param_patt.loc40_72.2 => constants.%r.param_patt.528
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestIterate(constants.%facet_value.847) {
// CHECK:STDOUT:   %R.patt.loc40_17.2 => constants.%R.patt
// CHECK:STDOUT:   %R.loc40_17.1 => constants.%facet_value.847
// CHECK:STDOUT:   %R.as_type.loc40_79.1 => constants.%DiffMutableRange
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.611
// CHECK:STDOUT:   %r.patt.loc40_77.2 => constants.%r.patt.5fe
// CHECK:STDOUT:   %r.param_patt.loc40_72.2 => constants.%r.param_patt.a4b
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc40 => constants.%complete_type.357
// CHECK:STDOUT:   %Iterate.lookup_impl_witness => constants.%Iterate.impl_witness.0fb
// CHECK:STDOUT:   %Iterate.facet.loc44_33.5 => constants.%Iterate.facet.012
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type => constants.%Iterate.WithSelf.NewCursor.type.ff0
// CHECK:STDOUT:   %.loc44_33.26 => constants.%.6cf1
// CHECK:STDOUT:   %impl.elem2.loc44_33.2 => constants.%T.as_type.as.Iterate.impl.NewCursor.573
// CHECK:STDOUT:   %specific_impl_fn.loc44_33.3 => constants.%T.as_type.as.Iterate.impl.NewCursor.specific_fn.c3b
// CHECK:STDOUT:   %impl.elem1 => constants.%Destroy.facet.4da
// CHECK:STDOUT:   %as_type => constants.%tuple.type.b18
// CHECK:STDOUT:   %require_complete.1 => constants.%complete_type.bfe
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type => constants.%Iterate.WithSelf.Next.type.8fc
// CHECK:STDOUT:   %.loc44_33.27 => constants.%.d8c
// CHECK:STDOUT:   %impl.elem3.loc44_33.2 => constants.%T.as_type.as.Iterate.impl.Next.288
// CHECK:STDOUT:   %specific_impl_fn.loc44_33.4 => constants.%T.as_type.as.Iterate.impl.Next.specific_fn.fb4
// CHECK:STDOUT:   %SelfDestruct.specific_fn.6 => constants.%SelfDestruct.specific_fn.9e00
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestIterate(constants.%facet_value.76f) {
// CHECK:STDOUT:   %R.patt.loc40_17.2 => constants.%R.patt
// CHECK:STDOUT:   %R.loc40_17.1 => constants.%facet_value.76f
// CHECK:STDOUT:   %R.as_type.loc40_79.1 => constants.%DiffConstRange
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.7dd
// CHECK:STDOUT:   %r.patt.loc40_77.2 => constants.%r.patt.9c0
// CHECK:STDOUT:   %r.param_patt.loc40_72.2 => constants.%r.param_patt.992
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc40 => constants.%complete_type.357
// CHECK:STDOUT:   %Iterate.lookup_impl_witness => constants.%Iterate.impl_witness.0fc
// CHECK:STDOUT:   %Iterate.facet.loc44_33.5 => constants.%Iterate.facet.151
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type => constants.%Iterate.WithSelf.NewCursor.type.cf5
// CHECK:STDOUT:   %.loc44_33.26 => constants.%.776
// CHECK:STDOUT:   %impl.elem2.loc44_33.2 => constants.%T.as_type.as.Iterate.impl.NewCursor.c96
// CHECK:STDOUT:   %specific_impl_fn.loc44_33.3 => constants.%T.as_type.as.Iterate.impl.NewCursor.specific_fn.ba8
// CHECK:STDOUT:   %impl.elem1 => constants.%Destroy.facet.017
// CHECK:STDOUT:   %as_type => constants.%tuple.type.df5
// CHECK:STDOUT:   %require_complete.1 => constants.%complete_type.ad6
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type => constants.%Iterate.WithSelf.Next.type.e49
// CHECK:STDOUT:   %.loc44_33.27 => constants.%.ee1
// CHECK:STDOUT:   %impl.elem3.loc44_33.2 => constants.%T.as_type.as.Iterate.impl.Next.264
// CHECK:STDOUT:   %specific_impl_fn.loc44_33.4 => constants.%T.as_type.as.Iterate.impl.Next.specific_fn.3dc
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
// CHECK:STDOUT:   %T.as_type.c1cc3e.1: type = facet_access_type %T.f84392.1 [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.lookup_impl_witness.70f: <witness> = lookup_impl_witness %T.f84392.1, @SubobjectDestroy [symbolic]
// CHECK:STDOUT:   %DestroyImpl.lookup_impl_witness.bf2: <witness> = lookup_impl_witness %T.f84392.1, @DestroyImpl [symbolic]
// CHECK:STDOUT:   %Destroy.facet.dc4: %Destroy.type = facet_value %T.as_type.c1cc3e.1, (%SubobjectDestroy.lookup_impl_witness.70f, %DestroyImpl.lookup_impl_witness.bf2) [symbolic]
// CHECK:STDOUT:   %MaybeUnformed.348: type = class_type @MaybeUnformed, @MaybeUnformed(%Destroy.facet.dc4) [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.476: type = class_type @DefaultOptionalStorage, @DefaultOptionalStorage(%T.f84392.1) [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.elem.4a2: type = unbound_element_type %DefaultOptionalStorage.476, bool [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.elem.ff3: type = unbound_element_type %DefaultOptionalStorage.476, %MaybeUnformed.348 [symbolic]
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
// CHECK:STDOUT:   %OverloadValueType.Op.type.83af6e.1: type = fn_type @OverloadValueType.Op.1 [concrete]
// CHECK:STDOUT:   %OverloadValueType.Op.4dca31.1: %OverloadValueType.Op.type.83af6e.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.5760f4.1: <witness> = custom_witness (%OverloadValueType.Op.4dca31.1), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.ef0: %SubobjectDestroy.type = facet_value %OverloadValueType, (%custom_witness.5760f4.1) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.efc: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.ef0) [concrete]
// CHECK:STDOUT:   %OverloadValueType.Op.type.83af6e.2: type = fn_type @OverloadValueType.Op.2 [concrete]
// CHECK:STDOUT:   %OverloadValueType.Op.4dca31.2: %OverloadValueType.Op.type.83af6e.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.d0da43.1: <witness> = custom_witness (%OverloadValueType.Op.4dca31.2), @Copy [concrete]
// CHECK:STDOUT:   %facet_value.249: %facet_type.bd0 = facet_value %OverloadValueType, (%custom_witness.5760f4.1, %DestroyImpl.impl_witness.efc, %custom_witness.d0da43.1) [concrete]
// CHECK:STDOUT:   %Iterate.lookup_impl_witness.1d8: <witness> = lookup_impl_witness %.Self.500, @Iterate [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.4ce: %facet_type.bd0 = impl_witness_access %Iterate.lookup_impl_witness.1d8, element0 [symbolic_self]
// CHECK:STDOUT:   %Iterate_where.type.4a9: type = facet_type <@Iterate where %impl.elem0.4ce = %facet_value.249> [concrete]
// CHECK:STDOUT:   %pattern_type.6c1: type = pattern_type %Iterate_where.type.4a9 [concrete]
// CHECK:STDOUT:   %R.patt: %pattern_type.6c1 = symbolic_binding_pattern R, 0 [symbolic]
// CHECK:STDOUT:   %R: %Iterate_where.type.4a9 = symbolic_binding R, 0 [symbolic]
// CHECK:STDOUT:   %R.as_type: type = facet_access_type %R [symbolic]
// CHECK:STDOUT:   %pattern_type.45b: type = pattern_type %R.as_type [symbolic]
// CHECK:STDOUT:   %r.patt.9b3: %pattern_type.45b = ref_binding_pattern r [symbolic]
// CHECK:STDOUT:   %r.param_patt.c29: %pattern_type.45b = var_param_pattern %r.patt.9b3 [symbolic]
// CHECK:STDOUT:   %_.patt.a10: %pattern_type.804 = value_binding_pattern _ [concrete]
// CHECK:STDOUT:   %Iterate.lookup_impl_witness.b22: <witness> = lookup_impl_witness %R, @Iterate [symbolic]
// CHECK:STDOUT:   %Iterate.facet.ea5: %Iterate.type = facet_value %R.as_type, (%Iterate.lookup_impl_witness.b22) [symbolic]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.268: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.ea5) [symbolic]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.6a9: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.ea5) [symbolic]
// CHECK:STDOUT:   %.202: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.268, %Iterate.facet.ea5 [symbolic]
// CHECK:STDOUT:   %impl.elem2.032: %.202 = impl_witness_access %Iterate.lookup_impl_witness.b22, element2 [symbolic]
// CHECK:STDOUT:   %impl.elem1.ccc: %Destroy.type = impl_witness_access %Iterate.lookup_impl_witness.b22, element1 [symbolic]
// CHECK:STDOUT:   %as_type.151: type = facet_access_type %impl.elem1.ccc [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.a09: <specific function> = specific_impl_function %impl.elem2.032, @Iterate.WithSelf.NewCursor(%Iterate.facet.ea5) [symbolic]
// CHECK:STDOUT:   %.003: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.6a9, %Iterate.facet.ea5 [symbolic]
// CHECK:STDOUT:   %impl.elem3.528: %.003 = impl_witness_access %Iterate.lookup_impl_witness.b22, element3 [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.827: type = class_type @DefaultOptionalStorage, @DefaultOptionalStorage(%facet_value.249) [concrete]
// CHECK:STDOUT:   %OptionalStorage.impl_witness.d26: <witness> = impl_witness imports.%OptionalStorage.impl_witness_table.1c4, @T.as_type.as.OptionalStorage.impl(%facet_value.249) [concrete]
// CHECK:STDOUT:   %OptionalStorage.facet.6b8: %OptionalStorage.type = facet_value %OverloadValueType, (%OptionalStorage.impl_witness.d26) [concrete]
// CHECK:STDOUT:   %Optional.410: type = class_type @Optional, @Optional(%OptionalStorage.facet.6b8) [concrete]
// CHECK:STDOUT:   %specific_impl_fn.a0d: <specific function> = specific_impl_function %impl.elem3.528, @Iterate.WithSelf.Next(%Iterate.facet.ea5) [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.type.6dc: type = fn_type @Optional.HasValue, @Optional(%OptionalStorage.facet.6b8) [concrete]
// CHECK:STDOUT:   %Optional.HasValue.5e3: %Optional.HasValue.type.6dc = struct_value () [concrete]
// CHECK:STDOUT:   %Optional.Get.type.ecf: type = fn_type @Optional.Get, @Optional(%OptionalStorage.facet.6b8) [concrete]
// CHECK:STDOUT:   %Optional.Get.e07: %Optional.Get.type.ecf = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.facet.45f: %Destroy.type = facet_value %OverloadValueType, (%custom_witness.5760f4.1, %DestroyImpl.impl_witness.efc) [concrete]
// CHECK:STDOUT:   %MaybeUnformed.c80: type = class_type @MaybeUnformed, @MaybeUnformed(%Destroy.facet.45f) [concrete]
// CHECK:STDOUT:   %.8a2: type = maybe_unformed_type %OverloadValueType [concrete]
// CHECK:STDOUT:   %DefaultOptionalStorage.elem.02a: type = unbound_element_type %DefaultOptionalStorage.827, %MaybeUnformed.c80 [concrete]
// CHECK:STDOUT:   %DefaultOptionalStorage.elem.099: type = unbound_element_type %DefaultOptionalStorage.827, bool [concrete]
// CHECK:STDOUT:   %struct_type.value.has_value.25d: type = struct_type {.value: %MaybeUnformed.c80, .has_value: bool} [concrete]
// CHECK:STDOUT:   %Optional.HasValue.specific_fn.c7c: <specific function> = specific_function %Optional.HasValue.5e3, @Optional.HasValue(%OptionalStorage.facet.6b8) [concrete]
// CHECK:STDOUT:   %Optional.Get.specific_fn.050: <specific function> = specific_function %Optional.Get.e07, @Optional.Get(%OptionalStorage.facet.6b8) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.3: type = fn_type @SubobjectDestroy.WithSelf.Op.loc27_37.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.3: %SubobjectDestroy.WithSelf.Op.type.6870fd.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.3: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.3), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.4: type = fn_type @SubobjectDestroy.WithSelf.Op.loc27_37.3 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.4: %SubobjectDestroy.WithSelf.Op.type.6870fd.4 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.4: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.4), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.a87: %SubobjectDestroy.type = facet_value bool, (%custom_witness.b80ef6.4) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.778: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.a87) [concrete]
// CHECK:STDOUT:   %Destroy.facet.80c: %Destroy.type = facet_value bool, (%custom_witness.b80ef6.4, %DestroyImpl.impl_witness.778) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.496: %SubobjectDestroy.type = facet_value %MaybeUnformed.c80, (%custom_witness.b80ef6.3) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.848: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.496) [concrete]
// CHECK:STDOUT:   %Destroy.facet.85c: %Destroy.type = facet_value %MaybeUnformed.c80, (%custom_witness.b80ef6.3, %DestroyImpl.impl_witness.848) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.7: type = fn_type @SubobjectDestroy.WithSelf.Op.loc27_37.6 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.7: %SubobjectDestroy.WithSelf.Op.type.6870fd.7 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.7: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.7), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.60e: %SubobjectDestroy.type = facet_value %Optional.410, (%custom_witness.b80ef6.7) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.1ac: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.60e) [concrete]
// CHECK:STDOUT:   %Destroy.facet.25e: %Destroy.type = facet_value %Optional.410, (%custom_witness.b80ef6.7, %DestroyImpl.impl_witness.1ac) [concrete]
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
// CHECK:STDOUT:   %custom_witness.51e: <witness> = custom_witness (%Iterator.Op.e8caec.1), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.ea6: %SubobjectDestroy.type = facet_value %Iterator, (%custom_witness.51e) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.e3a: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.ea6) [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.4232e0.2: type = fn_type @Iterator.Op.2 [concrete]
// CHECK:STDOUT:   %Iterator.Op.e8caec.2: %Iterator.Op.type.4232e0.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.6bd: <witness> = custom_witness (%OverloadValueType, %Iterator.Op.e8caec.2), @CppUnsafeDeref [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.4232e0.3: type = fn_type @Iterator.Op.3 [concrete]
// CHECK:STDOUT:   %Iterator.Op.e8caec.3: %Iterator.Op.type.4232e0.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.716: <witness> = custom_witness (%Iterator.Op.e8caec.3), @Inc [concrete]
// CHECK:STDOUT:   %Equal.type: type = fn_type @Equal [concrete]
// CHECK:STDOUT:   %Equal: %Equal.type = struct_value () [concrete]
// CHECK:STDOUT:   %NotEqual.type: type = fn_type @NotEqual [concrete]
// CHECK:STDOUT:   %NotEqual: %NotEqual.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.f93: <witness> = custom_witness (%Equal, %NotEqual), @EqWith, @EqWith(%Iterator) [concrete]
// CHECK:STDOUT:   %facet_value.732: %CppRange_where.type.d682d6.1 = facet_value %Range, (%custom_witness.090, %custom_witness.51e, %DestroyImpl.impl_witness.e3a, %custom_witness.6bd, %custom_witness.716, %custom_witness.f93, %custom_witness.5760f4.1, %DestroyImpl.impl_witness.efc, %custom_witness.d0da43.1, %custom_witness.51e, %DestroyImpl.impl_witness.e3a) [concrete]
// CHECK:STDOUT:   %tuple.type.9ef: type = tuple_type (%Iterator, %Iterator) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.8: type = fn_type @SubobjectDestroy.WithSelf.Op.3 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.8: %SubobjectDestroy.WithSelf.Op.type.6870fd.8 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.8: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.8), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.43f: %SubobjectDestroy.type = facet_value %tuple.type.9ef, (%custom_witness.b80ef6.8) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.c0f: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.43f) [concrete]
// CHECK:STDOUT:   %complete_type.e35: <witness> = complete_type_witness %tuple.type.9ef [concrete]
// CHECK:STDOUT:   %Destroy.facet.76b: %Destroy.type = facet_value %tuple.type.9ef, (%custom_witness.b80ef6.8, %DestroyImpl.impl_witness.c0f) [concrete]
// CHECK:STDOUT:   %Iterate.impl_witness.ac1: <witness> = impl_witness imports.%Iterate.impl_witness_table.121, @T.as_type.as.Iterate.impl(%facet_value.732) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.type.bae: type = fn_type @T.as_type.as.Iterate.impl.NewCursor, @T.as_type.as.Iterate.impl(%facet_value.732) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.edd: %T.as_type.as.Iterate.impl.NewCursor.type.bae = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.type.733: type = fn_type @T.as_type.as.Iterate.impl.Next, @T.as_type.as.Iterate.impl(%facet_value.732) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.842: %T.as_type.as.Iterate.impl.Next.type.733 = struct_value () [concrete]
// CHECK:STDOUT:   %facet_value.b75: %Iterate_where.type.4a9 = facet_value %Range, (%Iterate.impl_witness.ac1) [concrete]
// CHECK:STDOUT:   %r.patt.b9f: %pattern_type.23f = ref_binding_pattern r [concrete]
// CHECK:STDOUT:   %r.param_patt.466: %pattern_type.23f = var_param_pattern %r.patt.b9f [concrete]
// CHECK:STDOUT:   %Iterate.facet.5bf: %Iterate.type = facet_value %Range, (%Iterate.impl_witness.ac1) [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.a0e: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.5bf) [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.ce4: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.5bf) [concrete]
// CHECK:STDOUT:   %.da1: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.a0e, %Iterate.facet.5bf [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.specific_fn: <specific function> = specific_function %T.as_type.as.Iterate.impl.NewCursor.edd, @T.as_type.as.Iterate.impl.NewCursor(%facet_value.732) [concrete]
// CHECK:STDOUT:   %.604: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.ce4, %Iterate.facet.5bf [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.specific_fn: <specific function> = specific_function %T.as_type.as.Iterate.impl.Next.842, @T.as_type.as.Iterate.impl.Next(%facet_value.732) [concrete]
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
// CHECK:STDOUT:   %Core.import_ref.e86: @DefaultOptionalStorage.%DefaultOptionalStorage.elem.1 (%DefaultOptionalStorage.elem.ff3) = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [concrete = %field_decl.bce]
// CHECK:STDOUT:   %Core.import_ref.9ce: @DefaultOptionalStorage.%DefaultOptionalStorage.elem.2 (%DefaultOptionalStorage.elem.4a2) = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [concrete = %field_decl.dbe]
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
// CHECK:STDOUT:   %Core.import_ref.b0d = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %field_decl.dbe: @DefaultOptionalStorage.%DefaultOptionalStorage.elem.2 (%DefaultOptionalStorage.elem.4a2) = field_decl has_value, element1, %Core.import_ref.b0d in [concrete] {}
// CHECK:STDOUT:   %Core.import_ref.45d = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %field_decl.bce: @DefaultOptionalStorage.%DefaultOptionalStorage.elem.1 (%DefaultOptionalStorage.elem.ff3) = field_decl value, element0, %Core.import_ref.45d in [concrete] {}
// CHECK:STDOUT:   %Range.decl: type = class_decl @Range [concrete = constants.%Range] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @TestIterate(%R.loc25_17.2: %Iterate_where.type.4a9) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Iterate.lookup_impl_witness: <witness> = lookup_impl_witness %R.loc25_17.1, @Iterate [symbolic = %Iterate.lookup_impl_witness (constants.%Iterate.lookup_impl_witness.b22)]
// CHECK:STDOUT:   %Iterate.facet.loc27_37.5: %Iterate.type = facet_value %R.as_type.loc25_83.1, (%Iterate.lookup_impl_witness) [symbolic = %Iterate.facet.loc27_37.5 (constants.%Iterate.facet.ea5)]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.loc27_37.5) [symbolic = %Iterate.WithSelf.NewCursor.type (constants.%Iterate.WithSelf.NewCursor.type.268)]
// CHECK:STDOUT:   %.loc27_37.26: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type, %Iterate.facet.loc27_37.5 [symbolic = %.loc27_37.26 (constants.%.202)]
// CHECK:STDOUT:   %impl.elem2.loc27_37.2: @TestIterate.%.loc27_37.26 (%.202) = impl_witness_access %Iterate.lookup_impl_witness, element2 [symbolic = %impl.elem2.loc27_37.2 (constants.%impl.elem2.032)]
// CHECK:STDOUT:   %specific_impl_fn.loc27_37.3: <specific function> = specific_impl_function %impl.elem2.loc27_37.2, @Iterate.WithSelf.NewCursor(%Iterate.facet.loc27_37.5) [symbolic = %specific_impl_fn.loc27_37.3 (constants.%specific_impl_fn.a09)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.loc27_37.5) [symbolic = %Iterate.WithSelf.Next.type (constants.%Iterate.WithSelf.Next.type.6a9)]
// CHECK:STDOUT:   %.loc27_37.27: type = fn_type_with_self_type %Iterate.WithSelf.Next.type, %Iterate.facet.loc27_37.5 [symbolic = %.loc27_37.27 (constants.%.003)]
// CHECK:STDOUT:   %impl.elem3.loc27_37.2: @TestIterate.%.loc27_37.27 (%.003) = impl_witness_access %Iterate.lookup_impl_witness, element3 [symbolic = %impl.elem3.loc27_37.2 (constants.%impl.elem3.528)]
// CHECK:STDOUT:   %specific_impl_fn.loc27_37.4: <specific function> = specific_impl_function %impl.elem3.loc27_37.2, @Iterate.WithSelf.Next(%Iterate.facet.loc27_37.5) [symbolic = %specific_impl_fn.loc27_37.4 (constants.%specific_impl_fn.a0d)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%r.param: ref @TestIterate.%R.as_type.loc25_83.1 (%R.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %_.patt: %pattern_type.804 = value_binding_pattern _ [concrete = constants.%_.patt.a10]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %r.ref: ref @TestIterate.%R.as_type.loc25_83.1 (%R.as_type) = name_ref r, %r
// CHECK:STDOUT:     %impl.elem2.loc27_37.1: @TestIterate.%.loc27_37.26 (%.202) = impl_witness_access constants.%Iterate.lookup_impl_witness.b22, element2 [symbolic = %impl.elem2.loc27_37.2 (constants.%impl.elem2.032)]
// CHECK:STDOUT:     %bound_method.loc27_37.1: <bound method> = bound_method %r.ref, %impl.elem2.loc27_37.1
// CHECK:STDOUT:     %Iterate.facet.loc27_37.1: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.b22) [symbolic = %Iterate.facet.loc27_37.5 (constants.%Iterate.facet.ea5)]
// CHECK:STDOUT:     %.loc27_37.1: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc27_37.1 [symbolic = %Iterate.facet.loc27_37.5 (constants.%Iterate.facet.ea5)]
// CHECK:STDOUT:     %Iterate.facet.loc27_37.2: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.b22) [symbolic = %Iterate.facet.loc27_37.5 (constants.%Iterate.facet.ea5)]
// CHECK:STDOUT:     %.loc27_37.2: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc27_37.2 [symbolic = %Iterate.facet.loc27_37.5 (constants.%Iterate.facet.ea5)]
// CHECK:STDOUT:     %specific_impl_fn.loc27_37.1: <specific function> = specific_impl_function %impl.elem2.loc27_37.1, @Iterate.WithSelf.NewCursor(constants.%Iterate.facet.ea5) [symbolic = %specific_impl_fn.loc27_37.3 (constants.%specific_impl_fn.a09)]
// CHECK:STDOUT:     %bound_method.loc27_37.2: <bound method> = bound_method %r.ref, %specific_impl_fn.loc27_37.1
// CHECK:STDOUT:     %var: ref @TestIterate.%as_type (%as_type.151) = var_storage invalid
// CHECK:STDOUT:     %.loc27_36.1: @TestIterate.%R.as_type.loc25_83.1 (%R.as_type) = acquire_value %r.ref
// CHECK:STDOUT:     %Iterate.WithSelf.NewCursor.call: init @TestIterate.%as_type (%as_type.151) to %var = call %bound_method.loc27_37.2(%.loc27_36.1)
// CHECK:STDOUT:     assign %var, %Iterate.WithSelf.NewCursor.call
// CHECK:STDOUT:     br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.next:
// CHECK:STDOUT:     %impl.elem3.loc27_37.1: @TestIterate.%.loc27_37.27 (%.003) = impl_witness_access constants.%Iterate.lookup_impl_witness.b22, element3 [symbolic = %impl.elem3.loc27_37.2 (constants.%impl.elem3.528)]
// CHECK:STDOUT:     %bound_method.loc27_37.3: <bound method> = bound_method %r.ref, %impl.elem3.loc27_37.1
// CHECK:STDOUT:     %Iterate.facet.loc27_37.3: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.b22) [symbolic = %Iterate.facet.loc27_37.5 (constants.%Iterate.facet.ea5)]
// CHECK:STDOUT:     %.loc27_37.3: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc27_37.3 [symbolic = %Iterate.facet.loc27_37.5 (constants.%Iterate.facet.ea5)]
// CHECK:STDOUT:     %Iterate.facet.loc27_37.4: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.b22) [symbolic = %Iterate.facet.loc27_37.5 (constants.%Iterate.facet.ea5)]
// CHECK:STDOUT:     %.loc27_37.4: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc27_37.4 [symbolic = %Iterate.facet.loc27_37.5 (constants.%Iterate.facet.ea5)]
// CHECK:STDOUT:     %.loc27_37.5: %Destroy.type = converted constants.%as_type.151, constants.%impl.elem1.ccc [symbolic = %impl.elem1 (constants.%impl.elem1.ccc)]
// CHECK:STDOUT:     %.loc27_37.6: %Destroy.type = converted constants.%as_type.151, constants.%impl.elem1.ccc [symbolic = %impl.elem1 (constants.%impl.elem1.ccc)]
// CHECK:STDOUT:     %specific_impl_fn.loc27_37.2: <specific function> = specific_impl_function %impl.elem3.loc27_37.1, @Iterate.WithSelf.Next(constants.%Iterate.facet.ea5) [symbolic = %specific_impl_fn.loc27_37.4 (constants.%specific_impl_fn.a0d)]
// CHECK:STDOUT:     %bound_method.loc27_37.4: <bound method> = bound_method %r.ref, %specific_impl_fn.loc27_37.2
// CHECK:STDOUT:     %.loc27_37.7: ref %Optional.410 = temporary_storage
// CHECK:STDOUT:     %.loc27_36.2: @TestIterate.%R.as_type.loc25_83.1 (%R.as_type) = acquire_value %r.ref
// CHECK:STDOUT:     %Iterate.WithSelf.Next.call: init %Optional.410 to %.loc27_37.7 = call %bound_method.loc27_37.4(%.loc27_36.2, %var)
// CHECK:STDOUT:     %.loc27_37.8: ref %Optional.410 = temporary %.loc27_37.7, %Iterate.WithSelf.Next.call
// CHECK:STDOUT:     %.loc27_37.9: %Optional.HasValue.type.6dc = specific_constant imports.%Core.import_ref.c82, @Optional(constants.%OptionalStorage.facet.6b8) [concrete = constants.%Optional.HasValue.5e3]
// CHECK:STDOUT:     %HasValue.ref: %Optional.HasValue.type.6dc = name_ref HasValue, %.loc27_37.9 [concrete = constants.%Optional.HasValue.5e3]
// CHECK:STDOUT:     %Optional.HasValue.bound: <bound method> = bound_method %.loc27_37.8, %HasValue.ref
// CHECK:STDOUT:     %Optional.HasValue.specific_fn: <specific function> = specific_function %HasValue.ref, @Optional.HasValue(constants.%OptionalStorage.facet.6b8) [concrete = constants.%Optional.HasValue.specific_fn.c7c]
// CHECK:STDOUT:     %bound_method.loc27_37.5: <bound method> = bound_method %.loc27_37.8, %Optional.HasValue.specific_fn
// CHECK:STDOUT:     %.loc27_37.10: %Optional.410 = acquire_value %.loc27_37.8
// CHECK:STDOUT:     %Optional.HasValue.call: init bool = call %bound_method.loc27_37.5(%.loc27_37.10)
// CHECK:STDOUT:     %.loc27_37.11: bool = value_of_initializer %Optional.HasValue.call
// CHECK:STDOUT:     %.loc27_37.12: bool = converted %Optional.HasValue.call, %.loc27_37.11
// CHECK:STDOUT:     if %.loc27_37.12 br !for.body else br !for.done
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.body:
// CHECK:STDOUT:     %.loc27_37.13: %Optional.Get.type.ecf = specific_constant imports.%Core.import_ref.29e, @Optional(constants.%OptionalStorage.facet.6b8) [concrete = constants.%Optional.Get.e07]
// CHECK:STDOUT:     %Get.ref: %Optional.Get.type.ecf = name_ref Get, %.loc27_37.13 [concrete = constants.%Optional.Get.e07]
// CHECK:STDOUT:     %Optional.Get.bound: <bound method> = bound_method %.loc27_37.8, %Get.ref
// CHECK:STDOUT:     %Optional.Get.specific_fn: <specific function> = specific_function %Get.ref, @Optional.Get(constants.%OptionalStorage.facet.6b8) [concrete = constants.%Optional.Get.specific_fn.050]
// CHECK:STDOUT:     %bound_method.loc27_37.6: <bound method> = bound_method %.loc27_37.8, %Optional.Get.specific_fn
// CHECK:STDOUT:     %.loc27_37.14: ref %OverloadValueType = temporary_storage
// CHECK:STDOUT:     %.loc27_37.15: %Optional.410 = acquire_value %.loc27_37.8
// CHECK:STDOUT:     %Optional.Get.call: init %OverloadValueType to %.loc27_37.14 = call %bound_method.loc27_37.6(%.loc27_37.15)
// CHECK:STDOUT:     %.loc27_37.16: ref %OverloadValueType = temporary %.loc27_37.14, %Optional.Get.call
// CHECK:STDOUT:     %.loc27_37.17: %OverloadValueType = acquire_value %.loc27_37.16
// CHECK:STDOUT:     %.loc27_14: type = splice_block %OverloadValueType.ref.loc27 [concrete = constants.%OverloadValueType] {
// CHECK:STDOUT:       %Cpp.ref.loc27: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:       %OverloadValueType.ref.loc27: type = name_ref OverloadValueType, imports.%OverloadValueType.decl [concrete = constants.%OverloadValueType]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %_: %OverloadValueType = wrapper_binding _, %.loc27_37.17
// CHECK:STDOUT:     %Destroy.facet.loc27_37.1: %Destroy.type = facet_value constants.%OverloadValueType, (constants.%custom_witness.5760f4.1, constants.%DestroyImpl.impl_witness.efc) [concrete = constants.%Destroy.facet.45f]
// CHECK:STDOUT:     %.loc27_37.18: %Destroy.type = converted constants.%OverloadValueType, %Destroy.facet.loc27_37.1 [concrete = constants.%Destroy.facet.45f]
// CHECK:STDOUT:     %Destroy.facet.loc27_37.2: %Destroy.type = facet_value constants.%OverloadValueType, (constants.%custom_witness.5760f4.1, constants.%DestroyImpl.impl_witness.efc) [concrete = constants.%Destroy.facet.45f]
// CHECK:STDOUT:     %.loc27_37.19: %Destroy.type = converted constants.%OverloadValueType, %Destroy.facet.loc27_37.2 [concrete = constants.%Destroy.facet.45f]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %SelfDestruct.call.loc27_37.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc27_37.16)
// CHECK:STDOUT:     %Destroy.facet.loc27_37.3: %Destroy.type = facet_value constants.%Optional.410, (constants.%custom_witness.b80ef6.7, constants.%DestroyImpl.impl_witness.1ac) [concrete = constants.%Destroy.facet.25e]
// CHECK:STDOUT:     %.loc27_37.20: %Destroy.type = converted constants.%Optional.410, %Destroy.facet.loc27_37.3 [concrete = constants.%Destroy.facet.25e]
// CHECK:STDOUT:     %Destroy.facet.loc27_37.4: %Destroy.type = facet_value constants.%Optional.410, (constants.%custom_witness.b80ef6.7, constants.%DestroyImpl.impl_witness.1ac) [concrete = constants.%Destroy.facet.25e]
// CHECK:STDOUT:     %.loc27_37.21: %Destroy.type = converted constants.%Optional.410, %Destroy.facet.loc27_37.4 [concrete = constants.%Destroy.facet.25e]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %SelfDestruct.call.loc27_37.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc27_37.8)
// CHECK:STDOUT:     br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.done:
// CHECK:STDOUT:     %Destroy.facet.loc27_37.5: %Destroy.type = facet_value constants.%Optional.410, (constants.%custom_witness.b80ef6.7, constants.%DestroyImpl.impl_witness.1ac) [concrete = constants.%Destroy.facet.25e]
// CHECK:STDOUT:     %.loc27_37.22: %Destroy.type = converted constants.%Optional.410, %Destroy.facet.loc27_37.5 [concrete = constants.%Destroy.facet.25e]
// CHECK:STDOUT:     %Destroy.facet.loc27_37.6: %Destroy.type = facet_value constants.%Optional.410, (constants.%custom_witness.b80ef6.7, constants.%DestroyImpl.impl_witness.1ac) [concrete = constants.%Destroy.facet.25e]
// CHECK:STDOUT:     %.loc27_37.23: %Destroy.type = converted constants.%Optional.410, %Destroy.facet.loc27_37.6 [concrete = constants.%Destroy.facet.25e]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %SelfDestruct.call.loc27_37.3: init %empty_tuple.type = call %SelfDestruct.specific_fn.3(%.loc27_37.8)
// CHECK:STDOUT:     %.loc27_37.24: %Destroy.type = converted constants.%as_type.151, constants.%impl.elem1.ccc [symbolic = %impl.elem1 (constants.%impl.elem1.ccc)]
// CHECK:STDOUT:     %.loc27_37.25: %Destroy.type = converted constants.%as_type.151, constants.%impl.elem1.ccc [symbolic = %impl.elem1 (constants.%impl.elem1.ccc)]
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
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc27_37.2(%self.param: ref %MaybeUnformed.c80) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc27_37.3(%self.param: ref bool) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc27_37.4(%self.param: ref %struct_type.value.has_value.25d) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc27_37.1: ref bool = struct_access %self.param, element1
// CHECK:STDOUT:   %Destroy.facet.loc27_37.1: %Destroy.type = facet_value bool, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.778) [concrete = constants.%Destroy.facet.80c]
// CHECK:STDOUT:   %.loc27_37.2: %Destroy.type = converted bool, %Destroy.facet.loc27_37.1 [concrete = constants.%Destroy.facet.80c]
// CHECK:STDOUT:   %Destroy.facet.loc27_37.2: %Destroy.type = facet_value bool, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.778) [concrete = constants.%Destroy.facet.80c]
// CHECK:STDOUT:   %.loc27_37.3: %Destroy.type = converted bool, %Destroy.facet.loc27_37.2 [concrete = constants.%Destroy.facet.80c]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc27_37.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc27_37.1)
// CHECK:STDOUT:   %.loc27_37.4: ref %MaybeUnformed.c80 = struct_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc27_37.3: %Destroy.type = facet_value constants.%MaybeUnformed.c80, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.848) [concrete = constants.%Destroy.facet.85c]
// CHECK:STDOUT:   %.loc27_37.5: %Destroy.type = converted constants.%MaybeUnformed.c80, %Destroy.facet.loc27_37.3 [concrete = constants.%Destroy.facet.85c]
// CHECK:STDOUT:   %Destroy.facet.loc27_37.4: %Destroy.type = facet_value constants.%MaybeUnformed.c80, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.848) [concrete = constants.%Destroy.facet.85c]
// CHECK:STDOUT:   %.loc27_37.6: %Destroy.type = converted constants.%MaybeUnformed.c80, %Destroy.facet.loc27_37.4 [concrete = constants.%Destroy.facet.85c]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc27_37.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc27_37.4)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc27_37.5(%self.param: ref %DefaultOptionalStorage.827) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %has_value.ref: %DefaultOptionalStorage.elem.099 = name_ref has_value, imports.%Core.import_ref.9ce [concrete = imports.%field_decl.dbe]
// CHECK:STDOUT:   %.loc27_37.1: ref bool = class_element_access %self.param, element1
// CHECK:STDOUT:   %Destroy.facet.loc27_37.1: %Destroy.type = facet_value bool, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.778) [concrete = constants.%Destroy.facet.80c]
// CHECK:STDOUT:   %.loc27_37.2: %Destroy.type = converted bool, %Destroy.facet.loc27_37.1 [concrete = constants.%Destroy.facet.80c]
// CHECK:STDOUT:   %Destroy.facet.loc27_37.2: %Destroy.type = facet_value bool, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.778) [concrete = constants.%Destroy.facet.80c]
// CHECK:STDOUT:   %.loc27_37.3: %Destroy.type = converted bool, %Destroy.facet.loc27_37.2 [concrete = constants.%Destroy.facet.80c]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc27_37.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc27_37.1)
// CHECK:STDOUT:   %value.ref: %DefaultOptionalStorage.elem.02a = name_ref value, imports.%Core.import_ref.e86 [concrete = imports.%field_decl.bce]
// CHECK:STDOUT:   %.loc27_37.4: ref %MaybeUnformed.c80 = class_element_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc27_37.3: %Destroy.type = facet_value constants.%MaybeUnformed.c80, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.848) [concrete = constants.%Destroy.facet.85c]
// CHECK:STDOUT:   %.loc27_37.5: %Destroy.type = converted constants.%MaybeUnformed.c80, %Destroy.facet.loc27_37.3 [concrete = constants.%Destroy.facet.85c]
// CHECK:STDOUT:   %Destroy.facet.loc27_37.4: %Destroy.type = facet_value constants.%MaybeUnformed.c80, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.848) [concrete = constants.%Destroy.facet.85c]
// CHECK:STDOUT:   %.loc27_37.6: %Destroy.type = converted constants.%MaybeUnformed.c80, %Destroy.facet.loc27_37.4 [concrete = constants.%Destroy.facet.85c]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc27_37.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc27_37.4)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc27_37.6(%self.param: ref %Optional.410) {
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
// CHECK:STDOUT:   %impl.elem2: %.da1 = impl_witness_access constants.%Iterate.impl_witness.ac1, element2 [concrete = constants.%T.as_type.as.Iterate.impl.NewCursor.edd]
// CHECK:STDOUT:   %bound_method.loc37_37.1: <bound method> = bound_method %r.ref, %impl.elem2
// CHECK:STDOUT:   %facet_value.loc37_37.1: %CppRange_where.type.d682d6.1 = facet_value constants.%Range, (constants.%custom_witness.090, constants.%custom_witness.51e, constants.%DestroyImpl.impl_witness.e3a, constants.%custom_witness.6bd, constants.%custom_witness.716, constants.%custom_witness.f93, constants.%custom_witness.5760f4.1, constants.%DestroyImpl.impl_witness.efc, constants.%custom_witness.d0da43.1, constants.%custom_witness.51e, constants.%DestroyImpl.impl_witness.e3a) [concrete = constants.%facet_value.732]
// CHECK:STDOUT:   %.loc37_37.1: %CppRange_where.type.d682d6.1 = converted constants.%Range, %facet_value.loc37_37.1 [concrete = constants.%facet_value.732]
// CHECK:STDOUT:   %facet_value.loc37_37.2: %CppRange_where.type.d682d6.1 = facet_value constants.%Range, (constants.%custom_witness.090, constants.%custom_witness.51e, constants.%DestroyImpl.impl_witness.e3a, constants.%custom_witness.6bd, constants.%custom_witness.716, constants.%custom_witness.f93, constants.%custom_witness.5760f4.1, constants.%DestroyImpl.impl_witness.efc, constants.%custom_witness.d0da43.1, constants.%custom_witness.51e, constants.%DestroyImpl.impl_witness.e3a) [concrete = constants.%facet_value.732]
// CHECK:STDOUT:   %.loc37_37.2: %CppRange_where.type.d682d6.1 = converted constants.%Range, %facet_value.loc37_37.2 [concrete = constants.%facet_value.732]
// CHECK:STDOUT:   %specific_fn.loc37_37.1: <specific function> = specific_function %impl.elem2, @T.as_type.as.Iterate.impl.NewCursor(constants.%facet_value.732) [concrete = constants.%T.as_type.as.Iterate.impl.NewCursor.specific_fn]
// CHECK:STDOUT:   %bound_method.loc37_37.2: <bound method> = bound_method %r.ref, %specific_fn.loc37_37.1
// CHECK:STDOUT:   %var: ref %tuple.type.9ef = var_storage invalid
// CHECK:STDOUT:   %.loc37_36.1: %Range = acquire_value %r.ref
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.call: init %tuple.type.9ef to %var = call %bound_method.loc37_37.2(%.loc37_36.1)
// CHECK:STDOUT:   assign %var, %T.as_type.as.Iterate.impl.NewCursor.call
// CHECK:STDOUT:   br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT: !for.next:
// CHECK:STDOUT:   %impl.elem3: %.604 = impl_witness_access constants.%Iterate.impl_witness.ac1, element3 [concrete = constants.%T.as_type.as.Iterate.impl.Next.842]
// CHECK:STDOUT:   %bound_method.loc37_37.3: <bound method> = bound_method %r.ref, %impl.elem3
// CHECK:STDOUT:   %facet_value.loc37_37.3: %CppRange_where.type.d682d6.1 = facet_value constants.%Range, (constants.%custom_witness.090, constants.%custom_witness.51e, constants.%DestroyImpl.impl_witness.e3a, constants.%custom_witness.6bd, constants.%custom_witness.716, constants.%custom_witness.f93, constants.%custom_witness.5760f4.1, constants.%DestroyImpl.impl_witness.efc, constants.%custom_witness.d0da43.1, constants.%custom_witness.51e, constants.%DestroyImpl.impl_witness.e3a) [concrete = constants.%facet_value.732]
// CHECK:STDOUT:   %.loc37_37.3: %CppRange_where.type.d682d6.1 = converted constants.%Range, %facet_value.loc37_37.3 [concrete = constants.%facet_value.732]
// CHECK:STDOUT:   %facet_value.loc37_37.4: %CppRange_where.type.d682d6.1 = facet_value constants.%Range, (constants.%custom_witness.090, constants.%custom_witness.51e, constants.%DestroyImpl.impl_witness.e3a, constants.%custom_witness.6bd, constants.%custom_witness.716, constants.%custom_witness.f93, constants.%custom_witness.5760f4.1, constants.%DestroyImpl.impl_witness.efc, constants.%custom_witness.d0da43.1, constants.%custom_witness.51e, constants.%DestroyImpl.impl_witness.e3a) [concrete = constants.%facet_value.732]
// CHECK:STDOUT:   %.loc37_37.4: %CppRange_where.type.d682d6.1 = converted constants.%Range, %facet_value.loc37_37.4 [concrete = constants.%facet_value.732]
// CHECK:STDOUT:   %specific_fn.loc37_37.2: <specific function> = specific_function %impl.elem3, @T.as_type.as.Iterate.impl.Next(constants.%facet_value.732) [concrete = constants.%T.as_type.as.Iterate.impl.Next.specific_fn]
// CHECK:STDOUT:   %bound_method.loc37_37.4: <bound method> = bound_method %r.ref, %specific_fn.loc37_37.2
// CHECK:STDOUT:   %.loc37_37.5: ref %Optional.410 = temporary_storage
// CHECK:STDOUT:   %.loc37_36.2: %Range = acquire_value %r.ref
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.call: init %Optional.410 to %.loc37_37.5 = call %bound_method.loc37_37.4(%.loc37_36.2, %var)
// CHECK:STDOUT:   %.loc37_37.6: ref %Optional.410 = temporary %.loc37_37.5, %T.as_type.as.Iterate.impl.Next.call
// CHECK:STDOUT:   %.loc37_37.7: %Optional.HasValue.type.6dc = specific_constant imports.%Core.import_ref.c82, @Optional(constants.%OptionalStorage.facet.6b8) [concrete = constants.%Optional.HasValue.5e3]
// CHECK:STDOUT:   %HasValue.ref: %Optional.HasValue.type.6dc = name_ref HasValue, %.loc37_37.7 [concrete = constants.%Optional.HasValue.5e3]
// CHECK:STDOUT:   %Optional.HasValue.bound: <bound method> = bound_method %.loc37_37.6, %HasValue.ref
// CHECK:STDOUT:   %Optional.HasValue.specific_fn: <specific function> = specific_function %HasValue.ref, @Optional.HasValue(constants.%OptionalStorage.facet.6b8) [concrete = constants.%Optional.HasValue.specific_fn.c7c]
// CHECK:STDOUT:   %bound_method.loc37_37.5: <bound method> = bound_method %.loc37_37.6, %Optional.HasValue.specific_fn
// CHECK:STDOUT:   %.loc37_37.8: %Optional.410 = acquire_value %.loc37_37.6
// CHECK:STDOUT:   %Optional.HasValue.call: init bool = call %bound_method.loc37_37.5(%.loc37_37.8)
// CHECK:STDOUT:   %.loc37_37.9: bool = value_of_initializer %Optional.HasValue.call
// CHECK:STDOUT:   %.loc37_37.10: bool = converted %Optional.HasValue.call, %.loc37_37.9
// CHECK:STDOUT:   if %.loc37_37.10 br !for.body else br !for.done
// CHECK:STDOUT:
// CHECK:STDOUT: !for.body:
// CHECK:STDOUT:   %.loc37_37.11: %Optional.Get.type.ecf = specific_constant imports.%Core.import_ref.29e, @Optional(constants.%OptionalStorage.facet.6b8) [concrete = constants.%Optional.Get.e07]
// CHECK:STDOUT:   %Get.ref: %Optional.Get.type.ecf = name_ref Get, %.loc37_37.11 [concrete = constants.%Optional.Get.e07]
// CHECK:STDOUT:   %Optional.Get.bound: <bound method> = bound_method %.loc37_37.6, %Get.ref
// CHECK:STDOUT:   %Optional.Get.specific_fn: <specific function> = specific_function %Get.ref, @Optional.Get(constants.%OptionalStorage.facet.6b8) [concrete = constants.%Optional.Get.specific_fn.050]
// CHECK:STDOUT:   %bound_method.loc37_37.6: <bound method> = bound_method %.loc37_37.6, %Optional.Get.specific_fn
// CHECK:STDOUT:   %.loc37_37.12: ref %OverloadValueType = temporary_storage
// CHECK:STDOUT:   %.loc37_37.13: %Optional.410 = acquire_value %.loc37_37.6
// CHECK:STDOUT:   %Optional.Get.call: init %OverloadValueType to %.loc37_37.12 = call %bound_method.loc37_37.6(%.loc37_37.13)
// CHECK:STDOUT:   %.loc37_37.14: ref %OverloadValueType = temporary %.loc37_37.12, %Optional.Get.call
// CHECK:STDOUT:   %.loc37_37.15: %OverloadValueType = acquire_value %.loc37_37.14
// CHECK:STDOUT:   %.loc37_14: type = splice_block %OverloadValueType.ref [concrete = constants.%OverloadValueType] {
// CHECK:STDOUT:     %Cpp.ref.loc37: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %OverloadValueType.ref: type = name_ref OverloadValueType, imports.%OverloadValueType.decl [concrete = constants.%OverloadValueType]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %_: %OverloadValueType = wrapper_binding _, %.loc37_37.15
// CHECK:STDOUT:   %Destroy.facet.loc37_37.1: %Destroy.type = facet_value constants.%OverloadValueType, (constants.%custom_witness.5760f4.1, constants.%DestroyImpl.impl_witness.efc) [concrete = constants.%Destroy.facet.45f]
// CHECK:STDOUT:   %.loc37_37.16: %Destroy.type = converted constants.%OverloadValueType, %Destroy.facet.loc37_37.1 [concrete = constants.%Destroy.facet.45f]
// CHECK:STDOUT:   %Destroy.facet.loc37_37.2: %Destroy.type = facet_value constants.%OverloadValueType, (constants.%custom_witness.5760f4.1, constants.%DestroyImpl.impl_witness.efc) [concrete = constants.%Destroy.facet.45f]
// CHECK:STDOUT:   %.loc37_37.17: %Destroy.type = converted constants.%OverloadValueType, %Destroy.facet.loc37_37.2 [concrete = constants.%Destroy.facet.45f]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc37_37.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc37_37.14)
// CHECK:STDOUT:   %Destroy.facet.loc37_37.3: %Destroy.type = facet_value constants.%Optional.410, (constants.%custom_witness.b80ef6.7, constants.%DestroyImpl.impl_witness.1ac) [concrete = constants.%Destroy.facet.25e]
// CHECK:STDOUT:   %.loc37_37.18: %Destroy.type = converted constants.%Optional.410, %Destroy.facet.loc37_37.3 [concrete = constants.%Destroy.facet.25e]
// CHECK:STDOUT:   %Destroy.facet.loc37_37.4: %Destroy.type = facet_value constants.%Optional.410, (constants.%custom_witness.b80ef6.7, constants.%DestroyImpl.impl_witness.1ac) [concrete = constants.%Destroy.facet.25e]
// CHECK:STDOUT:   %.loc37_37.19: %Destroy.type = converted constants.%Optional.410, %Destroy.facet.loc37_37.4 [concrete = constants.%Destroy.facet.25e]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc37_37.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc37_37.6)
// CHECK:STDOUT:   br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT: !for.done:
// CHECK:STDOUT:   %Destroy.facet.loc37_37.5: %Destroy.type = facet_value constants.%Optional.410, (constants.%custom_witness.b80ef6.7, constants.%DestroyImpl.impl_witness.1ac) [concrete = constants.%Destroy.facet.25e]
// CHECK:STDOUT:   %.loc37_37.20: %Destroy.type = converted constants.%Optional.410, %Destroy.facet.loc37_37.5 [concrete = constants.%Destroy.facet.25e]
// CHECK:STDOUT:   %Destroy.facet.loc37_37.6: %Destroy.type = facet_value constants.%Optional.410, (constants.%custom_witness.b80ef6.7, constants.%DestroyImpl.impl_witness.1ac) [concrete = constants.%Destroy.facet.25e]
// CHECK:STDOUT:   %.loc37_37.21: %Destroy.type = converted constants.%Optional.410, %Destroy.facet.loc37_37.6 [concrete = constants.%Destroy.facet.25e]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc37_37.3: init %empty_tuple.type = call %SelfDestruct.specific_fn.3(%.loc37_37.6)
// CHECK:STDOUT:   %Destroy.facet.loc37_37.7: %Destroy.type = facet_value constants.%tuple.type.9ef, (constants.%custom_witness.b80ef6.8, constants.%DestroyImpl.impl_witness.c0f) [concrete = constants.%Destroy.facet.76b]
// CHECK:STDOUT:   %.loc37_37.22: %Destroy.type = converted constants.%tuple.type.9ef, %Destroy.facet.loc37_37.7 [concrete = constants.%Destroy.facet.76b]
// CHECK:STDOUT:   %Destroy.facet.loc37_37.8: %Destroy.type = facet_value constants.%tuple.type.9ef, (constants.%custom_witness.b80ef6.8, constants.%DestroyImpl.impl_witness.c0f) [concrete = constants.%Destroy.facet.76b]
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
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.45b
// CHECK:STDOUT:   %r.patt.loc25_81.2 => constants.%r.patt.9b3
// CHECK:STDOUT:   %r.param_patt.loc25_76.2 => constants.%r.param_patt.c29
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestIterate(constants.%facet_value.b75) {
// CHECK:STDOUT:   %R.patt.loc25_17.2 => constants.%R.patt
// CHECK:STDOUT:   %R.loc25_17.1 => constants.%facet_value.b75
// CHECK:STDOUT:   %R.as_type.loc25_83.1 => constants.%Range
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.23f
// CHECK:STDOUT:   %r.patt.loc25_81.2 => constants.%r.patt.b9f
// CHECK:STDOUT:   %r.param_patt.loc25_76.2 => constants.%r.param_patt.466
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc25 => constants.%complete_type.357
// CHECK:STDOUT:   %Iterate.lookup_impl_witness => constants.%Iterate.impl_witness.ac1
// CHECK:STDOUT:   %Iterate.facet.loc27_37.5 => constants.%Iterate.facet.5bf
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type => constants.%Iterate.WithSelf.NewCursor.type.a0e
// CHECK:STDOUT:   %.loc27_37.26 => constants.%.da1
// CHECK:STDOUT:   %impl.elem2.loc27_37.2 => constants.%T.as_type.as.Iterate.impl.NewCursor.edd
// CHECK:STDOUT:   %specific_impl_fn.loc27_37.3 => constants.%T.as_type.as.Iterate.impl.NewCursor.specific_fn
// CHECK:STDOUT:   %impl.elem1 => constants.%Destroy.facet.76b
// CHECK:STDOUT:   %as_type => constants.%tuple.type.9ef
// CHECK:STDOUT:   %require_complete.1 => constants.%complete_type.e35
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type => constants.%Iterate.WithSelf.Next.type.ce4
// CHECK:STDOUT:   %.loc27_37.27 => constants.%.604
// CHECK:STDOUT:   %impl.elem3.loc27_37.2 => constants.%T.as_type.as.Iterate.impl.Next.842
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
// CHECK:STDOUT:   %Iterator.Op.type.ac7e09.1: type = fn_type @Iterator.Op.1 [concrete]
// CHECK:STDOUT:   %Iterator.Op.583fda.1: %Iterator.Op.type.ac7e09.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.1fc: <witness> = custom_witness (%Iterator.Op.583fda.1), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.d46: %SubobjectDestroy.type = facet_value %Iterator, (%custom_witness.1fc) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.7ae: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.d46) [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.ac7e09.2: type = fn_type @Iterator.Op.2 [concrete]
// CHECK:STDOUT:   %Iterator.Op.583fda.2: %Iterator.Op.type.ac7e09.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.f6f: <witness> = custom_witness (%f64.761, %Iterator.Op.583fda.2), @CppUnsafeDeref [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.ac7e09.3: type = fn_type @Iterator.Op.3 [concrete]
// CHECK:STDOUT:   %Iterator.Op.583fda.3: %Iterator.Op.type.ac7e09.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.393: <witness> = custom_witness (%Iterator.Op.583fda.3), @Inc [concrete]
// CHECK:STDOUT:   %Equal.type: type = fn_type @Equal [concrete]
// CHECK:STDOUT:   %Equal: %Equal.type = struct_value () [concrete]
// CHECK:STDOUT:   %NotEqual.type: type = fn_type @NotEqual [concrete]
// CHECK:STDOUT:   %NotEqual: %NotEqual.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.eb7: <witness> = custom_witness (%Equal, %NotEqual), @EqWith, @EqWith(%Iterator) [concrete]
// CHECK:STDOUT:   %facet_value.310: %CppRange_where.type.d682d6.1 = facet_value %ConstRange, (%custom_witness.5d5, %custom_witness.1fc, %DestroyImpl.impl_witness.7ae, %custom_witness.f6f, %custom_witness.393, %custom_witness.eb7, %custom_witness.b80ef6.3, %DestroyImpl.impl_witness.a23, %Copy.impl_witness.eac, %custom_witness.1fc, %DestroyImpl.impl_witness.7ae) [concrete]
// CHECK:STDOUT:   %tuple.type.005: type = tuple_type (%Iterator, %Iterator) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.10: type = fn_type @SubobjectDestroy.WithSelf.Op.3 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.10: %SubobjectDestroy.WithSelf.Op.type.6870fd.10 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.10: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.10), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.7c6: %SubobjectDestroy.type = facet_value %tuple.type.005, (%custom_witness.b80ef6.10) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.729: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.7c6) [concrete]
// CHECK:STDOUT:   %complete_type.320: <witness> = complete_type_witness %tuple.type.005 [concrete]
// CHECK:STDOUT:   %Destroy.facet.0b5: %Destroy.type = facet_value %tuple.type.005, (%custom_witness.b80ef6.10, %DestroyImpl.impl_witness.729) [concrete]
// CHECK:STDOUT:   %Iterate.impl_witness.d5a: <witness> = impl_witness imports.%Iterate.impl_witness_table.121, @T.as_type.as.Iterate.impl(%facet_value.310) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.type.41f: type = fn_type @T.as_type.as.Iterate.impl.NewCursor, @T.as_type.as.Iterate.impl(%facet_value.310) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.af4: %T.as_type.as.Iterate.impl.NewCursor.type.41f = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.type.9b3: type = fn_type @T.as_type.as.Iterate.impl.Next, @T.as_type.as.Iterate.impl(%facet_value.310) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.1e7: %T.as_type.as.Iterate.impl.Next.type.9b3 = struct_value () [concrete]
// CHECK:STDOUT:   %facet_value.860: %Iterate_where.type.a66 = facet_value %ConstRange, (%Iterate.impl_witness.d5a) [concrete]
// CHECK:STDOUT:   %r.param_patt.a73: %pattern_type.025 = value_param_pattern [concrete]
// CHECK:STDOUT:   %r.patt.0b5: %pattern_type.025 = wrapper_binding_pattern r, %r.param_patt.a73 [concrete]
// CHECK:STDOUT:   %Iterate.facet.a66: %Iterate.type = facet_value %ConstRange, (%Iterate.impl_witness.d5a) [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.337: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.a66) [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.540: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.a66) [concrete]
// CHECK:STDOUT:   %.744: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.337, %Iterate.facet.a66 [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.specific_fn: <specific function> = specific_function %T.as_type.as.Iterate.impl.NewCursor.af4, @T.as_type.as.Iterate.impl.NewCursor(%facet_value.310) [concrete]
// CHECK:STDOUT:   %.68c: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.540, %Iterate.facet.a66 [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.specific_fn: <specific function> = specific_function %T.as_type.as.Iterate.impl.Next.1e7, @T.as_type.as.Iterate.impl.Next(%facet_value.310) [concrete]
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
// CHECK:STDOUT:   %impl.elem2: %.744 = impl_witness_access constants.%Iterate.impl_witness.d5a, element2 [concrete = constants.%T.as_type.as.Iterate.impl.NewCursor.af4]
// CHECK:STDOUT:   %bound_method.loc38_19.1: <bound method> = bound_method %c.ref, %impl.elem2
// CHECK:STDOUT:   %facet_value.loc38_19.1: %CppRange_where.type.d682d6.1 = facet_value constants.%ConstRange, (constants.%custom_witness.5d5, constants.%custom_witness.1fc, constants.%DestroyImpl.impl_witness.7ae, constants.%custom_witness.f6f, constants.%custom_witness.393, constants.%custom_witness.eb7, constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.a23, constants.%Copy.impl_witness.eac, constants.%custom_witness.1fc, constants.%DestroyImpl.impl_witness.7ae) [concrete = constants.%facet_value.310]
// CHECK:STDOUT:   %.loc38_19.1: %CppRange_where.type.d682d6.1 = converted constants.%ConstRange, %facet_value.loc38_19.1 [concrete = constants.%facet_value.310]
// CHECK:STDOUT:   %facet_value.loc38_19.2: %CppRange_where.type.d682d6.1 = facet_value constants.%ConstRange, (constants.%custom_witness.5d5, constants.%custom_witness.1fc, constants.%DestroyImpl.impl_witness.7ae, constants.%custom_witness.f6f, constants.%custom_witness.393, constants.%custom_witness.eb7, constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.a23, constants.%Copy.impl_witness.eac, constants.%custom_witness.1fc, constants.%DestroyImpl.impl_witness.7ae) [concrete = constants.%facet_value.310]
// CHECK:STDOUT:   %.loc38_19.2: %CppRange_where.type.d682d6.1 = converted constants.%ConstRange, %facet_value.loc38_19.2 [concrete = constants.%facet_value.310]
// CHECK:STDOUT:   %specific_fn.loc38_19.1: <specific function> = specific_function %impl.elem2, @T.as_type.as.Iterate.impl.NewCursor(constants.%facet_value.310) [concrete = constants.%T.as_type.as.Iterate.impl.NewCursor.specific_fn]
// CHECK:STDOUT:   %bound_method.loc38_19.2: <bound method> = bound_method %c.ref, %specific_fn.loc38_19.1
// CHECK:STDOUT:   %var: ref %tuple.type.005 = var_storage invalid
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.call: init %tuple.type.005 to %var = call %bound_method.loc38_19.2(%c.ref)
// CHECK:STDOUT:   assign %var, %T.as_type.as.Iterate.impl.NewCursor.call
// CHECK:STDOUT:   br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT: !for.next:
// CHECK:STDOUT:   %impl.elem3: %.68c = impl_witness_access constants.%Iterate.impl_witness.d5a, element3 [concrete = constants.%T.as_type.as.Iterate.impl.Next.1e7]
// CHECK:STDOUT:   %bound_method.loc38_19.3: <bound method> = bound_method %c.ref, %impl.elem3
// CHECK:STDOUT:   %facet_value.loc38_19.3: %CppRange_where.type.d682d6.1 = facet_value constants.%ConstRange, (constants.%custom_witness.5d5, constants.%custom_witness.1fc, constants.%DestroyImpl.impl_witness.7ae, constants.%custom_witness.f6f, constants.%custom_witness.393, constants.%custom_witness.eb7, constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.a23, constants.%Copy.impl_witness.eac, constants.%custom_witness.1fc, constants.%DestroyImpl.impl_witness.7ae) [concrete = constants.%facet_value.310]
// CHECK:STDOUT:   %.loc38_19.3: %CppRange_where.type.d682d6.1 = converted constants.%ConstRange, %facet_value.loc38_19.3 [concrete = constants.%facet_value.310]
// CHECK:STDOUT:   %facet_value.loc38_19.4: %CppRange_where.type.d682d6.1 = facet_value constants.%ConstRange, (constants.%custom_witness.5d5, constants.%custom_witness.1fc, constants.%DestroyImpl.impl_witness.7ae, constants.%custom_witness.f6f, constants.%custom_witness.393, constants.%custom_witness.eb7, constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.a23, constants.%Copy.impl_witness.eac, constants.%custom_witness.1fc, constants.%DestroyImpl.impl_witness.7ae) [concrete = constants.%facet_value.310]
// CHECK:STDOUT:   %.loc38_19.4: %CppRange_where.type.d682d6.1 = converted constants.%ConstRange, %facet_value.loc38_19.4 [concrete = constants.%facet_value.310]
// CHECK:STDOUT:   %specific_fn.loc38_19.2: <specific function> = specific_function %impl.elem3, @T.as_type.as.Iterate.impl.Next(constants.%facet_value.310) [concrete = constants.%T.as_type.as.Iterate.impl.Next.specific_fn]
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
// CHECK:STDOUT:   %Destroy.facet.loc38_19.5: %Destroy.type = facet_value constants.%tuple.type.005, (constants.%custom_witness.b80ef6.10, constants.%DestroyImpl.impl_witness.729) [concrete = constants.%Destroy.facet.0b5]
// CHECK:STDOUT:   %.loc38_19.19: %Destroy.type = converted constants.%tuple.type.005, %Destroy.facet.loc38_19.5 [concrete = constants.%Destroy.facet.0b5]
// CHECK:STDOUT:   %Destroy.facet.loc38_19.6: %Destroy.type = facet_value constants.%tuple.type.005, (constants.%custom_witness.b80ef6.10, constants.%DestroyImpl.impl_witness.729) [concrete = constants.%Destroy.facet.0b5]
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
// CHECK:STDOUT: specific @TestIterate(constants.%facet_value.860) {
// CHECK:STDOUT:   %R.patt.loc20_17.2 => constants.%R.patt
// CHECK:STDOUT:   %R.loc20_17.1 => constants.%facet_value.860
// CHECK:STDOUT:   %R.as_type.loc20_61.1 => constants.%ConstRange
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.025
// CHECK:STDOUT:   %r.param_patt.loc20_59.2 => constants.%r.param_patt.a73
// CHECK:STDOUT:   %r.patt.loc20_59.2 => constants.%r.patt.0b5
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc20 => constants.%complete_type.357
// CHECK:STDOUT:   %Iterate.lookup_impl_witness => constants.%Iterate.impl_witness.d5a
// CHECK:STDOUT:   %Iterate.facet.loc24_19.5 => constants.%Iterate.facet.a66
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type => constants.%Iterate.WithSelf.NewCursor.type.337
// CHECK:STDOUT:   %.loc24_19.23 => constants.%.744
// CHECK:STDOUT:   %impl.elem2.loc24_19.2 => constants.%T.as_type.as.Iterate.impl.NewCursor.af4
// CHECK:STDOUT:   %specific_impl_fn.loc24_19.3 => constants.%T.as_type.as.Iterate.impl.NewCursor.specific_fn
// CHECK:STDOUT:   %impl.elem1 => constants.%Destroy.facet.0b5
// CHECK:STDOUT:   %as_type => constants.%tuple.type.005
// CHECK:STDOUT:   %require_complete.1 => constants.%complete_type.320
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type => constants.%Iterate.WithSelf.Next.type.540
// CHECK:STDOUT:   %.loc24_19.24 => constants.%.68c
// CHECK:STDOUT:   %impl.elem3.loc24_19.2 => constants.%T.as_type.as.Iterate.impl.Next.1e7
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
// CHECK:STDOUT:   %Iterator.Op.type.b9731e.1: type = fn_type @Iterator.Op.1 [concrete]
// CHECK:STDOUT:   %Iterator.Op.ac6aad.1: %Iterator.Op.type.b9731e.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.f01: <witness> = custom_witness (%Iterator.Op.ac6aad.1), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.700: %SubobjectDestroy.type = facet_value %Iterator, (%custom_witness.f01) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.256: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.700) [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.b9731e.2: type = fn_type @Iterator.Op.2 [concrete]
// CHECK:STDOUT:   %Iterator.Op.ac6aad.2: %Iterator.Op.type.b9731e.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.0a8: <witness> = custom_witness (%f64.761, %Iterator.Op.ac6aad.2), @CppUnsafeDeref [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.b9731e.3: type = fn_type @Iterator.Op.3 [concrete]
// CHECK:STDOUT:   %Iterator.Op.ac6aad.3: %Iterator.Op.type.b9731e.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.491: <witness> = custom_witness (%Iterator.Op.ac6aad.3), @Inc [concrete]
// CHECK:STDOUT:   %Equal.type: type = fn_type @Equal [concrete]
// CHECK:STDOUT:   %Equal: %Equal.type = struct_value () [concrete]
// CHECK:STDOUT:   %NotEqual.type: type = fn_type @NotEqual [concrete]
// CHECK:STDOUT:   %NotEqual: %NotEqual.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.068: <witness> = custom_witness (%Equal, %NotEqual), @EqWith, @EqWith(%Iterator) [concrete]
// CHECK:STDOUT:   %facet_value.13f: %CppRange_where.type.d682d6.1 = facet_value %MutableRange, (%custom_witness.902, %custom_witness.f01, %DestroyImpl.impl_witness.256, %custom_witness.0a8, %custom_witness.491, %custom_witness.068, %custom_witness.b80ef6.3, %DestroyImpl.impl_witness.a23, %Copy.impl_witness.eac, %custom_witness.f01, %DestroyImpl.impl_witness.256) [concrete]
// CHECK:STDOUT:   %tuple.type.dd0: type = tuple_type (%Iterator, %Iterator) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.10: type = fn_type @SubobjectDestroy.WithSelf.Op.3 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.10: %SubobjectDestroy.WithSelf.Op.type.6870fd.10 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.10: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.10), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.0de: %SubobjectDestroy.type = facet_value %tuple.type.dd0, (%custom_witness.b80ef6.10) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.1fa: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.0de) [concrete]
// CHECK:STDOUT:   %complete_type.843: <witness> = complete_type_witness %tuple.type.dd0 [concrete]
// CHECK:STDOUT:   %Destroy.facet.5f38: %Destroy.type = facet_value %tuple.type.dd0, (%custom_witness.b80ef6.10, %DestroyImpl.impl_witness.1fa) [concrete]
// CHECK:STDOUT:   %Iterate.impl_witness.8a2: <witness> = impl_witness imports.%Iterate.impl_witness_table.121, @T.as_type.as.Iterate.impl(%facet_value.13f) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.type.82c: type = fn_type @T.as_type.as.Iterate.impl.NewCursor, @T.as_type.as.Iterate.impl(%facet_value.13f) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.3c2: %T.as_type.as.Iterate.impl.NewCursor.type.82c = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.type.a70: type = fn_type @T.as_type.as.Iterate.impl.Next, @T.as_type.as.Iterate.impl(%facet_value.13f) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.bc3: %T.as_type.as.Iterate.impl.Next.type.a70 = struct_value () [concrete]
// CHECK:STDOUT:   %facet_value.5f7: %Iterate_where.type.a66 = facet_value %MutableRange, (%Iterate.impl_witness.8a2) [concrete]
// CHECK:STDOUT:   %r.param_patt.aeb: %pattern_type.992 = value_param_pattern [concrete]
// CHECK:STDOUT:   %r.patt.892: %pattern_type.992 = wrapper_binding_pattern r, %r.param_patt.aeb [concrete]
// CHECK:STDOUT:   %Iterate.facet.602: %Iterate.type = facet_value %MutableRange, (%Iterate.impl_witness.8a2) [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.748: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.602) [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.b58: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.602) [concrete]
// CHECK:STDOUT:   %.03d: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.748, %Iterate.facet.602 [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.specific_fn: <specific function> = specific_function %T.as_type.as.Iterate.impl.NewCursor.3c2, @T.as_type.as.Iterate.impl.NewCursor(%facet_value.13f) [concrete]
// CHECK:STDOUT:   %.ab45: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.b58, %Iterate.facet.602 [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.specific_fn: <specific function> = specific_function %T.as_type.as.Iterate.impl.Next.bc3, @T.as_type.as.Iterate.impl.Next(%facet_value.13f) [concrete]
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
// CHECK:STDOUT:   %impl.elem2: %.03d = impl_witness_access constants.%Iterate.impl_witness.8a2, element2 [concrete = constants.%T.as_type.as.Iterate.impl.NewCursor.3c2]
// CHECK:STDOUT:   %bound_method.loc50_19.1: <bound method> = bound_method %m.ref, %impl.elem2
// CHECK:STDOUT:   %facet_value.loc50_19.1: %CppRange_where.type.d682d6.1 = facet_value constants.%MutableRange, (constants.%custom_witness.902, constants.%custom_witness.f01, constants.%DestroyImpl.impl_witness.256, constants.%custom_witness.0a8, constants.%custom_witness.491, constants.%custom_witness.068, constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.a23, constants.%Copy.impl_witness.eac, constants.%custom_witness.f01, constants.%DestroyImpl.impl_witness.256) [concrete = constants.%facet_value.13f]
// CHECK:STDOUT:   %.loc50_19.1: %CppRange_where.type.d682d6.1 = converted constants.%MutableRange, %facet_value.loc50_19.1 [concrete = constants.%facet_value.13f]
// CHECK:STDOUT:   %facet_value.loc50_19.2: %CppRange_where.type.d682d6.1 = facet_value constants.%MutableRange, (constants.%custom_witness.902, constants.%custom_witness.f01, constants.%DestroyImpl.impl_witness.256, constants.%custom_witness.0a8, constants.%custom_witness.491, constants.%custom_witness.068, constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.a23, constants.%Copy.impl_witness.eac, constants.%custom_witness.f01, constants.%DestroyImpl.impl_witness.256) [concrete = constants.%facet_value.13f]
// CHECK:STDOUT:   %.loc50_19.2: %CppRange_where.type.d682d6.1 = converted constants.%MutableRange, %facet_value.loc50_19.2 [concrete = constants.%facet_value.13f]
// CHECK:STDOUT:   %specific_fn.loc50_19.1: <specific function> = specific_function %impl.elem2, @T.as_type.as.Iterate.impl.NewCursor(constants.%facet_value.13f) [concrete = constants.%T.as_type.as.Iterate.impl.NewCursor.specific_fn]
// CHECK:STDOUT:   %bound_method.loc50_19.2: <bound method> = bound_method %m.ref, %specific_fn.loc50_19.1
// CHECK:STDOUT:   %var: ref %tuple.type.dd0 = var_storage invalid
// CHECK:STDOUT:   %.loc50_18.1: %MutableRange = acquire_value %m.ref
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.call: init %tuple.type.dd0 to %var = call %bound_method.loc50_19.2(%.loc50_18.1)
// CHECK:STDOUT:   assign %var, %T.as_type.as.Iterate.impl.NewCursor.call
// CHECK:STDOUT:   br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT: !for.next:
// CHECK:STDOUT:   %impl.elem3: %.ab45 = impl_witness_access constants.%Iterate.impl_witness.8a2, element3 [concrete = constants.%T.as_type.as.Iterate.impl.Next.bc3]
// CHECK:STDOUT:   %bound_method.loc50_19.3: <bound method> = bound_method %m.ref, %impl.elem3
// CHECK:STDOUT:   %facet_value.loc50_19.3: %CppRange_where.type.d682d6.1 = facet_value constants.%MutableRange, (constants.%custom_witness.902, constants.%custom_witness.f01, constants.%DestroyImpl.impl_witness.256, constants.%custom_witness.0a8, constants.%custom_witness.491, constants.%custom_witness.068, constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.a23, constants.%Copy.impl_witness.eac, constants.%custom_witness.f01, constants.%DestroyImpl.impl_witness.256) [concrete = constants.%facet_value.13f]
// CHECK:STDOUT:   %.loc50_19.3: %CppRange_where.type.d682d6.1 = converted constants.%MutableRange, %facet_value.loc50_19.3 [concrete = constants.%facet_value.13f]
// CHECK:STDOUT:   %facet_value.loc50_19.4: %CppRange_where.type.d682d6.1 = facet_value constants.%MutableRange, (constants.%custom_witness.902, constants.%custom_witness.f01, constants.%DestroyImpl.impl_witness.256, constants.%custom_witness.0a8, constants.%custom_witness.491, constants.%custom_witness.068, constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.a23, constants.%Copy.impl_witness.eac, constants.%custom_witness.f01, constants.%DestroyImpl.impl_witness.256) [concrete = constants.%facet_value.13f]
// CHECK:STDOUT:   %.loc50_19.4: %CppRange_where.type.d682d6.1 = converted constants.%MutableRange, %facet_value.loc50_19.4 [concrete = constants.%facet_value.13f]
// CHECK:STDOUT:   %specific_fn.loc50_19.2: <specific function> = specific_function %impl.elem3, @T.as_type.as.Iterate.impl.Next(constants.%facet_value.13f) [concrete = constants.%T.as_type.as.Iterate.impl.Next.specific_fn]
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
// CHECK:STDOUT:   %Destroy.facet.loc50_19.5: %Destroy.type = facet_value constants.%tuple.type.dd0, (constants.%custom_witness.b80ef6.10, constants.%DestroyImpl.impl_witness.1fa) [concrete = constants.%Destroy.facet.5f38]
// CHECK:STDOUT:   %.loc50_19.19: %Destroy.type = converted constants.%tuple.type.dd0, %Destroy.facet.loc50_19.5 [concrete = constants.%Destroy.facet.5f38]
// CHECK:STDOUT:   %Destroy.facet.loc50_19.6: %Destroy.type = facet_value constants.%tuple.type.dd0, (constants.%custom_witness.b80ef6.10, constants.%DestroyImpl.impl_witness.1fa) [concrete = constants.%Destroy.facet.5f38]
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
// CHECK:STDOUT: specific @TestIterate(constants.%facet_value.5f7) {
// CHECK:STDOUT:   %R.patt.loc32_17.2 => constants.%R.patt
// CHECK:STDOUT:   %R.loc32_17.1 => constants.%facet_value.5f7
// CHECK:STDOUT:   %R.as_type.loc32_61.1 => constants.%MutableRange
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.992
// CHECK:STDOUT:   %r.param_patt.loc32_59.2 => constants.%r.param_patt.aeb
// CHECK:STDOUT:   %r.patt.loc32_59.2 => constants.%r.patt.892
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc32 => constants.%complete_type.357
// CHECK:STDOUT:   %Iterate.lookup_impl_witness => constants.%Iterate.impl_witness.8a2
// CHECK:STDOUT:   %Iterate.facet.loc36_19.5 => constants.%Iterate.facet.602
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type => constants.%Iterate.WithSelf.NewCursor.type.748
// CHECK:STDOUT:   %.loc36_19.23 => constants.%.03d
// CHECK:STDOUT:   %impl.elem2.loc36_19.2 => constants.%T.as_type.as.Iterate.impl.NewCursor.3c2
// CHECK:STDOUT:   %specific_impl_fn.loc36_19.3 => constants.%T.as_type.as.Iterate.impl.NewCursor.specific_fn
// CHECK:STDOUT:   %impl.elem1 => constants.%Destroy.facet.5f38
// CHECK:STDOUT:   %as_type => constants.%tuple.type.dd0
// CHECK:STDOUT:   %require_complete.1 => constants.%complete_type.843
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type => constants.%Iterate.WithSelf.Next.type.b58
// CHECK:STDOUT:   %.loc36_19.24 => constants.%.ab45
// CHECK:STDOUT:   %impl.elem3.loc36_19.2 => constants.%T.as_type.as.Iterate.impl.Next.bc3
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
// CHECK:STDOUT:   %T.as_type.c1cc3e.1: type = facet_access_type %T.f84392.1 [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.lookup_impl_witness.70f: <witness> = lookup_impl_witness %T.f84392.1, @SubobjectDestroy [symbolic]
// CHECK:STDOUT:   %DestroyImpl.lookup_impl_witness.bf2: <witness> = lookup_impl_witness %T.f84392.1, @DestroyImpl [symbolic]
// CHECK:STDOUT:   %Destroy.facet.dc4: %Destroy.type = facet_value %T.as_type.c1cc3e.1, (%SubobjectDestroy.lookup_impl_witness.70f, %DestroyImpl.lookup_impl_witness.bf2) [symbolic]
// CHECK:STDOUT:   %MaybeUnformed.348: type = class_type @MaybeUnformed, @MaybeUnformed(%Destroy.facet.dc4) [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.476: type = class_type @DefaultOptionalStorage, @DefaultOptionalStorage(%T.f84392.1) [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.elem.4a2: type = unbound_element_type %DefaultOptionalStorage.476, bool [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.elem.ff3: type = unbound_element_type %DefaultOptionalStorage.476, %MaybeUnformed.348 [symbolic]
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
// CHECK:STDOUT:   %ValueType.Op.type.4c0699.1: type = fn_type @ValueType.Op.1 [concrete]
// CHECK:STDOUT:   %ValueType.Op.4ec7f2.1: %ValueType.Op.type.4c0699.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.0a1: <witness> = custom_witness (%ValueType.Op.4ec7f2.1), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.bf6: %SubobjectDestroy.type = facet_value %ValueType, (%custom_witness.0a1) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.488: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.bf6) [concrete]
// CHECK:STDOUT:   %const.dd6: type = const_type %ValueType [concrete]
// CHECK:STDOUT:   %ptr.69c: type = ptr_type %const.dd6 [concrete]
// CHECK:STDOUT:   %ptr.233: type = ptr_type %ValueType [concrete]
// CHECK:STDOUT:   %ValueType.Op.type.4c0699.2: type = fn_type @ValueType.Op.2 [concrete]
// CHECK:STDOUT:   %ValueType.Op.4ec7f2.2: %ValueType.Op.type.4c0699.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.202: <witness> = custom_witness (%ValueType.Op.4ec7f2.2), @Copy [concrete]
// CHECK:STDOUT:   %facet_value.0a5: %facet_type.bd0 = facet_value %ValueType, (%custom_witness.0a1, %DestroyImpl.impl_witness.488, %custom_witness.202) [concrete]
// CHECK:STDOUT:   %Iterate.lookup_impl_witness.1d8: <witness> = lookup_impl_witness %.Self.500, @Iterate [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.4ce: %facet_type.bd0 = impl_witness_access %Iterate.lookup_impl_witness.1d8, element0 [symbolic_self]
// CHECK:STDOUT:   %Iterate_where.type.360: type = facet_type <@Iterate where %impl.elem0.4ce = %facet_value.0a5> [concrete]
// CHECK:STDOUT:   %pattern_type.5d3: type = pattern_type %Iterate_where.type.360 [concrete]
// CHECK:STDOUT:   %R.patt: %pattern_type.5d3 = symbolic_binding_pattern R, 0 [symbolic]
// CHECK:STDOUT:   %R: %Iterate_where.type.360 = symbolic_binding R, 0 [symbolic]
// CHECK:STDOUT:   %R.as_type: type = facet_access_type %R [symbolic]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %R.as_type [symbolic]
// CHECK:STDOUT:   %r.param_patt.80f: %pattern_type.7ce = value_param_pattern [symbolic]
// CHECK:STDOUT:   %r.patt.7ca: %pattern_type.7ce = wrapper_binding_pattern r, %r.param_patt.80f [symbolic]
// CHECK:STDOUT:   %i.patt: %pattern_type.c43 = value_binding_pattern i [concrete]
// CHECK:STDOUT:   %Iterate.lookup_impl_witness.62d: <witness> = lookup_impl_witness %R, @Iterate [symbolic]
// CHECK:STDOUT:   %Iterate.facet.c11: %Iterate.type = facet_value %R.as_type, (%Iterate.lookup_impl_witness.62d) [symbolic]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.516: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.c11) [symbolic]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.7e2: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.c11) [symbolic]
// CHECK:STDOUT:   %.8b7: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.516, %Iterate.facet.c11 [symbolic]
// CHECK:STDOUT:   %impl.elem2.1dc: %.8b7 = impl_witness_access %Iterate.lookup_impl_witness.62d, element2 [symbolic]
// CHECK:STDOUT:   %impl.elem1.5b8: %Destroy.type = impl_witness_access %Iterate.lookup_impl_witness.62d, element1 [symbolic]
// CHECK:STDOUT:   %as_type.283: type = facet_access_type %impl.elem1.5b8 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.d13: <specific function> = specific_impl_function %impl.elem2.1dc, @Iterate.WithSelf.NewCursor(%Iterate.facet.c11) [symbolic]
// CHECK:STDOUT:   %.d82: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.7e2, %Iterate.facet.c11 [symbolic]
// CHECK:STDOUT:   %impl.elem3.902: %.d82 = impl_witness_access %Iterate.lookup_impl_witness.62d, element3 [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.70a: type = class_type @DefaultOptionalStorage, @DefaultOptionalStorage(%facet_value.0a5) [concrete]
// CHECK:STDOUT:   %OptionalStorage.impl_witness.253: <witness> = impl_witness imports.%OptionalStorage.impl_witness_table.1c4, @T.as_type.as.OptionalStorage.impl(%facet_value.0a5) [concrete]
// CHECK:STDOUT:   %OptionalStorage.facet.244: %OptionalStorage.type = facet_value %ValueType, (%OptionalStorage.impl_witness.253) [concrete]
// CHECK:STDOUT:   %Optional.d4a: type = class_type @Optional, @Optional(%OptionalStorage.facet.244) [concrete]
// CHECK:STDOUT:   %specific_impl_fn.4a7: <specific function> = specific_impl_function %impl.elem3.902, @Iterate.WithSelf.Next(%Iterate.facet.c11) [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.type.141: type = fn_type @Optional.HasValue, @Optional(%OptionalStorage.facet.244) [concrete]
// CHECK:STDOUT:   %Optional.HasValue.01b: %Optional.HasValue.type.141 = struct_value () [concrete]
// CHECK:STDOUT:   %Optional.Get.type.28b: type = fn_type @Optional.Get, @Optional(%OptionalStorage.facet.244) [concrete]
// CHECK:STDOUT:   %Optional.Get.90b: %Optional.Get.type.28b = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.facet.d1b: %Destroy.type = facet_value %ValueType, (%custom_witness.0a1, %DestroyImpl.impl_witness.488) [concrete]
// CHECK:STDOUT:   %MaybeUnformed.ceb: type = class_type @MaybeUnformed, @MaybeUnformed(%Destroy.facet.d1b) [concrete]
// CHECK:STDOUT:   %.be1: type = maybe_unformed_type %ValueType [concrete]
// CHECK:STDOUT:   %DefaultOptionalStorage.elem.bc1: type = unbound_element_type %DefaultOptionalStorage.70a, %MaybeUnformed.ceb [concrete]
// CHECK:STDOUT:   %DefaultOptionalStorage.elem.8c7: type = unbound_element_type %DefaultOptionalStorage.70a, bool [concrete]
// CHECK:STDOUT:   %struct_type.value.has_value.cec: type = struct_type {.value: %MaybeUnformed.ceb, .has_value: bool} [concrete]
// CHECK:STDOUT:   %Optional.HasValue.specific_fn.18d: <specific function> = specific_function %Optional.HasValue.01b, @Optional.HasValue(%OptionalStorage.facet.244) [concrete]
// CHECK:STDOUT:   %Optional.Get.specific_fn.f55: <specific function> = specific_function %Optional.Get.90b, @Optional.Get(%OptionalStorage.facet.244) [concrete]
// CHECK:STDOUT:   %operator_PlusEqual__carbon_thunk.type: type = fn_type @operator_PlusEqual__carbon_thunk [concrete]
// CHECK:STDOUT:   %operator_PlusEqual__carbon_thunk: %operator_PlusEqual__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.3: type = fn_type @SubobjectDestroy.WithSelf.Op.loc31_37.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.3: %SubobjectDestroy.WithSelf.Op.type.6870fd.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.3: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.3), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.4: type = fn_type @SubobjectDestroy.WithSelf.Op.loc31_37.3 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.4: %SubobjectDestroy.WithSelf.Op.type.6870fd.4 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.4: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.4), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.a87: %SubobjectDestroy.type = facet_value bool, (%custom_witness.b80ef6.4) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.778: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.a87) [concrete]
// CHECK:STDOUT:   %Destroy.facet.80c: %Destroy.type = facet_value bool, (%custom_witness.b80ef6.4, %DestroyImpl.impl_witness.778) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.84c: %SubobjectDestroy.type = facet_value %MaybeUnformed.ceb, (%custom_witness.b80ef6.3) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.3b3: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.84c) [concrete]
// CHECK:STDOUT:   %Destroy.facet.f67: %Destroy.type = facet_value %MaybeUnformed.ceb, (%custom_witness.b80ef6.3, %DestroyImpl.impl_witness.3b3) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.7: type = fn_type @SubobjectDestroy.WithSelf.Op.loc31_37.6 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.7: %SubobjectDestroy.WithSelf.Op.type.6870fd.7 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.7: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.7), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.7cd: %SubobjectDestroy.type = facet_value %Optional.d4a, (%custom_witness.b80ef6.7) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.ce9: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.7cd) [concrete]
// CHECK:STDOUT:   %Destroy.facet.7d9: %Destroy.type = facet_value %Optional.d4a, (%custom_witness.b80ef6.7, %DestroyImpl.impl_witness.ce9) [concrete]
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
// CHECK:STDOUT:   %custom_witness.612b50.1: <witness> = custom_witness (%Iterator.Op.240e50.1), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.403: %SubobjectDestroy.type = facet_value %Iterator, (%custom_witness.612b50.1) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.587: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.403) [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.3f6dc2.2: type = fn_type @Iterator.Op.2 [concrete]
// CHECK:STDOUT:   %Iterator.Op.240e50.2: %Iterator.Op.type.3f6dc2.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.5be: <witness> = custom_witness (%ValueType, %Iterator.Op.240e50.2), @CppUnsafeDeref [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.3f6dc2.3: type = fn_type @Iterator.Op.3 [concrete]
// CHECK:STDOUT:   %Iterator.Op.240e50.3: %Iterator.Op.type.3f6dc2.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.c31: <witness> = custom_witness (%Iterator.Op.240e50.3), @Inc [concrete]
// CHECK:STDOUT:   %Equal.type: type = fn_type @Equal [concrete]
// CHECK:STDOUT:   %Equal: %Equal.type = struct_value () [concrete]
// CHECK:STDOUT:   %NotEqual.type: type = fn_type @NotEqual [concrete]
// CHECK:STDOUT:   %NotEqual: %NotEqual.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.bf4: <witness> = custom_witness (%Equal, %NotEqual), @EqWith, @EqWith(%Sentinel) [concrete]
// CHECK:STDOUT:   %Sentinel.Op.type: type = fn_type @Sentinel.Op [concrete]
// CHECK:STDOUT:   %Sentinel.Op: %Sentinel.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.612b50.2: <witness> = custom_witness (%Sentinel.Op), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.96b: %SubobjectDestroy.type = facet_value %Sentinel, (%custom_witness.612b50.2) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.fa7: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.96b) [concrete]
// CHECK:STDOUT:   %facet_value.d86: %CppRange_where.type.d682d6.1 = facet_value %ConstRange, (%custom_witness.116, %custom_witness.612b50.1, %DestroyImpl.impl_witness.587, %custom_witness.5be, %custom_witness.c31, %custom_witness.bf4, %custom_witness.0a1, %DestroyImpl.impl_witness.488, %custom_witness.202, %custom_witness.612b50.2, %DestroyImpl.impl_witness.fa7) [concrete]
// CHECK:STDOUT:   %tuple.type.7a6: type = tuple_type (%Iterator, %Sentinel) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.8: type = fn_type @SubobjectDestroy.WithSelf.Op.3 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.8: %SubobjectDestroy.WithSelf.Op.type.6870fd.8 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.8: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.8), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.b14: %SubobjectDestroy.type = facet_value %tuple.type.7a6, (%custom_witness.b80ef6.8) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.b22: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.b14) [concrete]
// CHECK:STDOUT:   %complete_type.cf0: <witness> = complete_type_witness %tuple.type.7a6 [concrete]
// CHECK:STDOUT:   %Destroy.facet.959: %Destroy.type = facet_value %tuple.type.7a6, (%custom_witness.b80ef6.8, %DestroyImpl.impl_witness.b22) [concrete]
// CHECK:STDOUT:   %Iterate.impl_witness.939: <witness> = impl_witness imports.%Iterate.impl_witness_table.121, @T.as_type.as.Iterate.impl(%facet_value.d86) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.type.8ad: type = fn_type @T.as_type.as.Iterate.impl.NewCursor, @T.as_type.as.Iterate.impl(%facet_value.d86) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.d53: %T.as_type.as.Iterate.impl.NewCursor.type.8ad = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.type.f47: type = fn_type @T.as_type.as.Iterate.impl.Next, @T.as_type.as.Iterate.impl(%facet_value.d86) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.2f5: %T.as_type.as.Iterate.impl.Next.type.f47 = struct_value () [concrete]
// CHECK:STDOUT:   %facet_value.8da: %Iterate_where.type.360 = facet_value %ConstRange, (%Iterate.impl_witness.939) [concrete]
// CHECK:STDOUT:   %r.param_patt.ac5: %pattern_type.c1d = value_param_pattern [concrete]
// CHECK:STDOUT:   %r.patt.97f: %pattern_type.c1d = wrapper_binding_pattern r, %r.param_patt.ac5 [concrete]
// CHECK:STDOUT:   %Iterate.facet.ce6: %Iterate.type = facet_value %ConstRange, (%Iterate.impl_witness.939) [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.df3: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.ce6) [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.869: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.ce6) [concrete]
// CHECK:STDOUT:   %.472: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.df3, %Iterate.facet.ce6 [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.specific_fn: <specific function> = specific_function %T.as_type.as.Iterate.impl.NewCursor.d53, @T.as_type.as.Iterate.impl.NewCursor(%facet_value.d86) [concrete]
// CHECK:STDOUT:   %.135: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.869, %Iterate.facet.ce6 [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.specific_fn: <specific function> = specific_function %T.as_type.as.Iterate.impl.Next.2f5, @T.as_type.as.Iterate.impl.Next(%facet_value.d86) [concrete]
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
// CHECK:STDOUT:   %Core.import_ref.e86: @DefaultOptionalStorage.%DefaultOptionalStorage.elem.1 (%DefaultOptionalStorage.elem.ff3) = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [concrete = %field_decl.bce]
// CHECK:STDOUT:   %Core.import_ref.9ce: @DefaultOptionalStorage.%DefaultOptionalStorage.elem.2 (%DefaultOptionalStorage.elem.4a2) = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [concrete = %field_decl.dbe]
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
// CHECK:STDOUT:   %Core.import_ref.b0d = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %field_decl.dbe: @DefaultOptionalStorage.%DefaultOptionalStorage.elem.2 (%DefaultOptionalStorage.elem.4a2) = field_decl has_value, element1, %Core.import_ref.b0d in [concrete] {}
// CHECK:STDOUT:   %Core.import_ref.45d = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %field_decl.bce: @DefaultOptionalStorage.%DefaultOptionalStorage.elem.1 (%DefaultOptionalStorage.elem.ff3) = field_decl value, element0, %Core.import_ref.45d in [concrete] {}
// CHECK:STDOUT:   %ConstRange.decl: type = class_decl @ConstRange [concrete = constants.%ConstRange] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @TestIterate(%R.loc27_17.2: %Iterate_where.type.360) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Iterate.lookup_impl_witness: <witness> = lookup_impl_witness %R.loc27_17.1, @Iterate [symbolic = %Iterate.lookup_impl_witness (constants.%Iterate.lookup_impl_witness.62d)]
// CHECK:STDOUT:   %Iterate.facet.loc31_37.5: %Iterate.type = facet_value %R.as_type.loc27_79.1, (%Iterate.lookup_impl_witness) [symbolic = %Iterate.facet.loc31_37.5 (constants.%Iterate.facet.c11)]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.loc31_37.5) [symbolic = %Iterate.WithSelf.NewCursor.type (constants.%Iterate.WithSelf.NewCursor.type.516)]
// CHECK:STDOUT:   %.loc31_37.26: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type, %Iterate.facet.loc31_37.5 [symbolic = %.loc31_37.26 (constants.%.8b7)]
// CHECK:STDOUT:   %impl.elem2.loc31_37.2: @TestIterate.%.loc31_37.26 (%.8b7) = impl_witness_access %Iterate.lookup_impl_witness, element2 [symbolic = %impl.elem2.loc31_37.2 (constants.%impl.elem2.1dc)]
// CHECK:STDOUT:   %specific_impl_fn.loc31_37.3: <specific function> = specific_impl_function %impl.elem2.loc31_37.2, @Iterate.WithSelf.NewCursor(%Iterate.facet.loc31_37.5) [symbolic = %specific_impl_fn.loc31_37.3 (constants.%specific_impl_fn.d13)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.loc31_37.5) [symbolic = %Iterate.WithSelf.Next.type (constants.%Iterate.WithSelf.Next.type.7e2)]
// CHECK:STDOUT:   %.loc31_37.27: type = fn_type_with_self_type %Iterate.WithSelf.Next.type, %Iterate.facet.loc31_37.5 [symbolic = %.loc31_37.27 (constants.%.d82)]
// CHECK:STDOUT:   %impl.elem3.loc31_37.2: @TestIterate.%.loc31_37.27 (%.d82) = impl_witness_access %Iterate.lookup_impl_witness, element3 [symbolic = %impl.elem3.loc31_37.2 (constants.%impl.elem3.902)]
// CHECK:STDOUT:   %specific_impl_fn.loc31_37.4: <specific function> = specific_impl_function %impl.elem3.loc31_37.2, @Iterate.WithSelf.Next(%Iterate.facet.loc31_37.5) [symbolic = %specific_impl_fn.loc31_37.4 (constants.%specific_impl_fn.4a7)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%r.param: @TestIterate.%R.as_type.loc27_79.1 (%R.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %i.patt: %pattern_type.c43 = value_binding_pattern i [concrete = constants.%i.patt]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %r.ref: @TestIterate.%R.as_type.loc27_79.1 (%R.as_type) = name_ref r, %r
// CHECK:STDOUT:     %impl.elem2.loc31_37.1: @TestIterate.%.loc31_37.26 (%.8b7) = impl_witness_access constants.%Iterate.lookup_impl_witness.62d, element2 [symbolic = %impl.elem2.loc31_37.2 (constants.%impl.elem2.1dc)]
// CHECK:STDOUT:     %bound_method.loc31_37.1: <bound method> = bound_method %r.ref, %impl.elem2.loc31_37.1
// CHECK:STDOUT:     %Iterate.facet.loc31_37.1: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.62d) [symbolic = %Iterate.facet.loc31_37.5 (constants.%Iterate.facet.c11)]
// CHECK:STDOUT:     %.loc31_37.1: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc31_37.1 [symbolic = %Iterate.facet.loc31_37.5 (constants.%Iterate.facet.c11)]
// CHECK:STDOUT:     %Iterate.facet.loc31_37.2: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.62d) [symbolic = %Iterate.facet.loc31_37.5 (constants.%Iterate.facet.c11)]
// CHECK:STDOUT:     %.loc31_37.2: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc31_37.2 [symbolic = %Iterate.facet.loc31_37.5 (constants.%Iterate.facet.c11)]
// CHECK:STDOUT:     %specific_impl_fn.loc31_37.1: <specific function> = specific_impl_function %impl.elem2.loc31_37.1, @Iterate.WithSelf.NewCursor(constants.%Iterate.facet.c11) [symbolic = %specific_impl_fn.loc31_37.3 (constants.%specific_impl_fn.d13)]
// CHECK:STDOUT:     %bound_method.loc31_37.2: <bound method> = bound_method %r.ref, %specific_impl_fn.loc31_37.1
// CHECK:STDOUT:     %var: ref @TestIterate.%as_type (%as_type.283) = var_storage invalid
// CHECK:STDOUT:     %Iterate.WithSelf.NewCursor.call: init @TestIterate.%as_type (%as_type.283) to %var = call %bound_method.loc31_37.2(%r.ref)
// CHECK:STDOUT:     assign %var, %Iterate.WithSelf.NewCursor.call
// CHECK:STDOUT:     br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.next:
// CHECK:STDOUT:     %impl.elem3.loc31_37.1: @TestIterate.%.loc31_37.27 (%.d82) = impl_witness_access constants.%Iterate.lookup_impl_witness.62d, element3 [symbolic = %impl.elem3.loc31_37.2 (constants.%impl.elem3.902)]
// CHECK:STDOUT:     %bound_method.loc31_37.3: <bound method> = bound_method %r.ref, %impl.elem3.loc31_37.1
// CHECK:STDOUT:     %Iterate.facet.loc31_37.3: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.62d) [symbolic = %Iterate.facet.loc31_37.5 (constants.%Iterate.facet.c11)]
// CHECK:STDOUT:     %.loc31_37.3: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc31_37.3 [symbolic = %Iterate.facet.loc31_37.5 (constants.%Iterate.facet.c11)]
// CHECK:STDOUT:     %Iterate.facet.loc31_37.4: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.62d) [symbolic = %Iterate.facet.loc31_37.5 (constants.%Iterate.facet.c11)]
// CHECK:STDOUT:     %.loc31_37.4: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc31_37.4 [symbolic = %Iterate.facet.loc31_37.5 (constants.%Iterate.facet.c11)]
// CHECK:STDOUT:     %.loc31_37.5: %Destroy.type = converted constants.%as_type.283, constants.%impl.elem1.5b8 [symbolic = %impl.elem1 (constants.%impl.elem1.5b8)]
// CHECK:STDOUT:     %.loc31_37.6: %Destroy.type = converted constants.%as_type.283, constants.%impl.elem1.5b8 [symbolic = %impl.elem1 (constants.%impl.elem1.5b8)]
// CHECK:STDOUT:     %specific_impl_fn.loc31_37.2: <specific function> = specific_impl_function %impl.elem3.loc31_37.1, @Iterate.WithSelf.Next(constants.%Iterate.facet.c11) [symbolic = %specific_impl_fn.loc31_37.4 (constants.%specific_impl_fn.4a7)]
// CHECK:STDOUT:     %bound_method.loc31_37.4: <bound method> = bound_method %r.ref, %specific_impl_fn.loc31_37.2
// CHECK:STDOUT:     %.loc31_37.7: ref %Optional.d4a = temporary_storage
// CHECK:STDOUT:     %Iterate.WithSelf.Next.call: init %Optional.d4a to %.loc31_37.7 = call %bound_method.loc31_37.4(%r.ref, %var)
// CHECK:STDOUT:     %.loc31_37.8: ref %Optional.d4a = temporary %.loc31_37.7, %Iterate.WithSelf.Next.call
// CHECK:STDOUT:     %.loc31_37.9: %Optional.HasValue.type.141 = specific_constant imports.%Core.import_ref.c82, @Optional(constants.%OptionalStorage.facet.244) [concrete = constants.%Optional.HasValue.01b]
// CHECK:STDOUT:     %HasValue.ref: %Optional.HasValue.type.141 = name_ref HasValue, %.loc31_37.9 [concrete = constants.%Optional.HasValue.01b]
// CHECK:STDOUT:     %Optional.HasValue.bound: <bound method> = bound_method %.loc31_37.8, %HasValue.ref
// CHECK:STDOUT:     %Optional.HasValue.specific_fn: <specific function> = specific_function %HasValue.ref, @Optional.HasValue(constants.%OptionalStorage.facet.244) [concrete = constants.%Optional.HasValue.specific_fn.18d]
// CHECK:STDOUT:     %bound_method.loc31_37.5: <bound method> = bound_method %.loc31_37.8, %Optional.HasValue.specific_fn
// CHECK:STDOUT:     %.loc31_37.10: %Optional.d4a = acquire_value %.loc31_37.8
// CHECK:STDOUT:     %Optional.HasValue.call: init bool = call %bound_method.loc31_37.5(%.loc31_37.10)
// CHECK:STDOUT:     %.loc31_37.11: bool = value_of_initializer %Optional.HasValue.call
// CHECK:STDOUT:     %.loc31_37.12: bool = converted %Optional.HasValue.call, %.loc31_37.11
// CHECK:STDOUT:     if %.loc31_37.12 br !for.body else br !for.done
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.body:
// CHECK:STDOUT:     %.loc31_37.13: %Optional.Get.type.28b = specific_constant imports.%Core.import_ref.29e, @Optional(constants.%OptionalStorage.facet.244) [concrete = constants.%Optional.Get.90b]
// CHECK:STDOUT:     %Get.ref: %Optional.Get.type.28b = name_ref Get, %.loc31_37.13 [concrete = constants.%Optional.Get.90b]
// CHECK:STDOUT:     %Optional.Get.bound: <bound method> = bound_method %.loc31_37.8, %Get.ref
// CHECK:STDOUT:     %Optional.Get.specific_fn: <specific function> = specific_function %Get.ref, @Optional.Get(constants.%OptionalStorage.facet.244) [concrete = constants.%Optional.Get.specific_fn.f55]
// CHECK:STDOUT:     %bound_method.loc31_37.6: <bound method> = bound_method %.loc31_37.8, %Optional.Get.specific_fn
// CHECK:STDOUT:     %.loc31_37.14: ref %ValueType = temporary_storage
// CHECK:STDOUT:     %.loc31_37.15: %Optional.d4a = acquire_value %.loc31_37.8
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
// CHECK:STDOUT:     %Destroy.facet.loc31_37.1: %Destroy.type = facet_value constants.%ValueType, (constants.%custom_witness.0a1, constants.%DestroyImpl.impl_witness.488) [concrete = constants.%Destroy.facet.d1b]
// CHECK:STDOUT:     %.loc31_37.18: %Destroy.type = converted constants.%ValueType, %Destroy.facet.loc31_37.1 [concrete = constants.%Destroy.facet.d1b]
// CHECK:STDOUT:     %Destroy.facet.loc31_37.2: %Destroy.type = facet_value constants.%ValueType, (constants.%custom_witness.0a1, constants.%DestroyImpl.impl_witness.488) [concrete = constants.%Destroy.facet.d1b]
// CHECK:STDOUT:     %.loc31_37.19: %Destroy.type = converted constants.%ValueType, %Destroy.facet.loc31_37.2 [concrete = constants.%Destroy.facet.d1b]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %SelfDestruct.call.loc31_37.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc31_37.16)
// CHECK:STDOUT:     %Destroy.facet.loc31_37.3: %Destroy.type = facet_value constants.%Optional.d4a, (constants.%custom_witness.b80ef6.7, constants.%DestroyImpl.impl_witness.ce9) [concrete = constants.%Destroy.facet.7d9]
// CHECK:STDOUT:     %.loc31_37.20: %Destroy.type = converted constants.%Optional.d4a, %Destroy.facet.loc31_37.3 [concrete = constants.%Destroy.facet.7d9]
// CHECK:STDOUT:     %Destroy.facet.loc31_37.4: %Destroy.type = facet_value constants.%Optional.d4a, (constants.%custom_witness.b80ef6.7, constants.%DestroyImpl.impl_witness.ce9) [concrete = constants.%Destroy.facet.7d9]
// CHECK:STDOUT:     %.loc31_37.21: %Destroy.type = converted constants.%Optional.d4a, %Destroy.facet.loc31_37.4 [concrete = constants.%Destroy.facet.7d9]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %SelfDestruct.call.loc31_37.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc31_37.8)
// CHECK:STDOUT:     br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.done:
// CHECK:STDOUT:     %Destroy.facet.loc31_37.5: %Destroy.type = facet_value constants.%Optional.d4a, (constants.%custom_witness.b80ef6.7, constants.%DestroyImpl.impl_witness.ce9) [concrete = constants.%Destroy.facet.7d9]
// CHECK:STDOUT:     %.loc31_37.22: %Destroy.type = converted constants.%Optional.d4a, %Destroy.facet.loc31_37.5 [concrete = constants.%Destroy.facet.7d9]
// CHECK:STDOUT:     %Destroy.facet.loc31_37.6: %Destroy.type = facet_value constants.%Optional.d4a, (constants.%custom_witness.b80ef6.7, constants.%DestroyImpl.impl_witness.ce9) [concrete = constants.%Destroy.facet.7d9]
// CHECK:STDOUT:     %.loc31_37.23: %Destroy.type = converted constants.%Optional.d4a, %Destroy.facet.loc31_37.6 [concrete = constants.%Destroy.facet.7d9]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %SelfDestruct.call.loc31_37.3: init %empty_tuple.type = call %SelfDestruct.specific_fn.3(%.loc31_37.8)
// CHECK:STDOUT:     %.loc31_37.24: %Destroy.type = converted constants.%as_type.283, constants.%impl.elem1.5b8 [symbolic = %impl.elem1 (constants.%impl.elem1.5b8)]
// CHECK:STDOUT:     %.loc31_37.25: %Destroy.type = converted constants.%as_type.283, constants.%impl.elem1.5b8 [symbolic = %impl.elem1 (constants.%impl.elem1.5b8)]
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
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc31_37.2(%self.param: ref %MaybeUnformed.ceb) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc31_37.3(%self.param: ref bool) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc31_37.4(%self.param: ref %struct_type.value.has_value.cec) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc31_37.1: ref bool = struct_access %self.param, element1
// CHECK:STDOUT:   %Destroy.facet.loc31_37.1: %Destroy.type = facet_value bool, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.778) [concrete = constants.%Destroy.facet.80c]
// CHECK:STDOUT:   %.loc31_37.2: %Destroy.type = converted bool, %Destroy.facet.loc31_37.1 [concrete = constants.%Destroy.facet.80c]
// CHECK:STDOUT:   %Destroy.facet.loc31_37.2: %Destroy.type = facet_value bool, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.778) [concrete = constants.%Destroy.facet.80c]
// CHECK:STDOUT:   %.loc31_37.3: %Destroy.type = converted bool, %Destroy.facet.loc31_37.2 [concrete = constants.%Destroy.facet.80c]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc31_37.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc31_37.1)
// CHECK:STDOUT:   %.loc31_37.4: ref %MaybeUnformed.ceb = struct_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc31_37.3: %Destroy.type = facet_value constants.%MaybeUnformed.ceb, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.3b3) [concrete = constants.%Destroy.facet.f67]
// CHECK:STDOUT:   %.loc31_37.5: %Destroy.type = converted constants.%MaybeUnformed.ceb, %Destroy.facet.loc31_37.3 [concrete = constants.%Destroy.facet.f67]
// CHECK:STDOUT:   %Destroy.facet.loc31_37.4: %Destroy.type = facet_value constants.%MaybeUnformed.ceb, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.3b3) [concrete = constants.%Destroy.facet.f67]
// CHECK:STDOUT:   %.loc31_37.6: %Destroy.type = converted constants.%MaybeUnformed.ceb, %Destroy.facet.loc31_37.4 [concrete = constants.%Destroy.facet.f67]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc31_37.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc31_37.4)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc31_37.5(%self.param: ref %DefaultOptionalStorage.70a) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %has_value.ref: %DefaultOptionalStorage.elem.8c7 = name_ref has_value, imports.%Core.import_ref.9ce [concrete = imports.%field_decl.dbe]
// CHECK:STDOUT:   %.loc31_37.1: ref bool = class_element_access %self.param, element1
// CHECK:STDOUT:   %Destroy.facet.loc31_37.1: %Destroy.type = facet_value bool, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.778) [concrete = constants.%Destroy.facet.80c]
// CHECK:STDOUT:   %.loc31_37.2: %Destroy.type = converted bool, %Destroy.facet.loc31_37.1 [concrete = constants.%Destroy.facet.80c]
// CHECK:STDOUT:   %Destroy.facet.loc31_37.2: %Destroy.type = facet_value bool, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.778) [concrete = constants.%Destroy.facet.80c]
// CHECK:STDOUT:   %.loc31_37.3: %Destroy.type = converted bool, %Destroy.facet.loc31_37.2 [concrete = constants.%Destroy.facet.80c]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc31_37.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc31_37.1)
// CHECK:STDOUT:   %value.ref: %DefaultOptionalStorage.elem.bc1 = name_ref value, imports.%Core.import_ref.e86 [concrete = imports.%field_decl.bce]
// CHECK:STDOUT:   %.loc31_37.4: ref %MaybeUnformed.ceb = class_element_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc31_37.3: %Destroy.type = facet_value constants.%MaybeUnformed.ceb, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.3b3) [concrete = constants.%Destroy.facet.f67]
// CHECK:STDOUT:   %.loc31_37.5: %Destroy.type = converted constants.%MaybeUnformed.ceb, %Destroy.facet.loc31_37.3 [concrete = constants.%Destroy.facet.f67]
// CHECK:STDOUT:   %Destroy.facet.loc31_37.4: %Destroy.type = facet_value constants.%MaybeUnformed.ceb, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.3b3) [concrete = constants.%Destroy.facet.f67]
// CHECK:STDOUT:   %.loc31_37.6: %Destroy.type = converted constants.%MaybeUnformed.ceb, %Destroy.facet.loc31_37.4 [concrete = constants.%Destroy.facet.f67]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc31_37.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc31_37.4)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc31_37.6(%self.param: ref %Optional.d4a) {
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
// CHECK:STDOUT:   %impl.elem2: %.472 = impl_witness_access constants.%Iterate.impl_witness.939, element2 [concrete = constants.%T.as_type.as.Iterate.impl.NewCursor.d53]
// CHECK:STDOUT:   %bound_method.loc45_37.1: <bound method> = bound_method %c.ref, %impl.elem2
// CHECK:STDOUT:   %facet_value.loc45_37.1: %CppRange_where.type.d682d6.1 = facet_value constants.%ConstRange, (constants.%custom_witness.116, constants.%custom_witness.612b50.1, constants.%DestroyImpl.impl_witness.587, constants.%custom_witness.5be, constants.%custom_witness.c31, constants.%custom_witness.bf4, constants.%custom_witness.0a1, constants.%DestroyImpl.impl_witness.488, constants.%custom_witness.202, constants.%custom_witness.612b50.2, constants.%DestroyImpl.impl_witness.fa7) [concrete = constants.%facet_value.d86]
// CHECK:STDOUT:   %.loc45_37.1: %CppRange_where.type.d682d6.1 = converted constants.%ConstRange, %facet_value.loc45_37.1 [concrete = constants.%facet_value.d86]
// CHECK:STDOUT:   %facet_value.loc45_37.2: %CppRange_where.type.d682d6.1 = facet_value constants.%ConstRange, (constants.%custom_witness.116, constants.%custom_witness.612b50.1, constants.%DestroyImpl.impl_witness.587, constants.%custom_witness.5be, constants.%custom_witness.c31, constants.%custom_witness.bf4, constants.%custom_witness.0a1, constants.%DestroyImpl.impl_witness.488, constants.%custom_witness.202, constants.%custom_witness.612b50.2, constants.%DestroyImpl.impl_witness.fa7) [concrete = constants.%facet_value.d86]
// CHECK:STDOUT:   %.loc45_37.2: %CppRange_where.type.d682d6.1 = converted constants.%ConstRange, %facet_value.loc45_37.2 [concrete = constants.%facet_value.d86]
// CHECK:STDOUT:   %specific_fn.loc45_37.1: <specific function> = specific_function %impl.elem2, @T.as_type.as.Iterate.impl.NewCursor(constants.%facet_value.d86) [concrete = constants.%T.as_type.as.Iterate.impl.NewCursor.specific_fn]
// CHECK:STDOUT:   %bound_method.loc45_37.2: <bound method> = bound_method %c.ref, %specific_fn.loc45_37.1
// CHECK:STDOUT:   %var: ref %tuple.type.7a6 = var_storage invalid
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.call: init %tuple.type.7a6 to %var = call %bound_method.loc45_37.2(%c.ref)
// CHECK:STDOUT:   assign %var, %T.as_type.as.Iterate.impl.NewCursor.call
// CHECK:STDOUT:   br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT: !for.next:
// CHECK:STDOUT:   %impl.elem3: %.135 = impl_witness_access constants.%Iterate.impl_witness.939, element3 [concrete = constants.%T.as_type.as.Iterate.impl.Next.2f5]
// CHECK:STDOUT:   %bound_method.loc45_37.3: <bound method> = bound_method %c.ref, %impl.elem3
// CHECK:STDOUT:   %facet_value.loc45_37.3: %CppRange_where.type.d682d6.1 = facet_value constants.%ConstRange, (constants.%custom_witness.116, constants.%custom_witness.612b50.1, constants.%DestroyImpl.impl_witness.587, constants.%custom_witness.5be, constants.%custom_witness.c31, constants.%custom_witness.bf4, constants.%custom_witness.0a1, constants.%DestroyImpl.impl_witness.488, constants.%custom_witness.202, constants.%custom_witness.612b50.2, constants.%DestroyImpl.impl_witness.fa7) [concrete = constants.%facet_value.d86]
// CHECK:STDOUT:   %.loc45_37.3: %CppRange_where.type.d682d6.1 = converted constants.%ConstRange, %facet_value.loc45_37.3 [concrete = constants.%facet_value.d86]
// CHECK:STDOUT:   %facet_value.loc45_37.4: %CppRange_where.type.d682d6.1 = facet_value constants.%ConstRange, (constants.%custom_witness.116, constants.%custom_witness.612b50.1, constants.%DestroyImpl.impl_witness.587, constants.%custom_witness.5be, constants.%custom_witness.c31, constants.%custom_witness.bf4, constants.%custom_witness.0a1, constants.%DestroyImpl.impl_witness.488, constants.%custom_witness.202, constants.%custom_witness.612b50.2, constants.%DestroyImpl.impl_witness.fa7) [concrete = constants.%facet_value.d86]
// CHECK:STDOUT:   %.loc45_37.4: %CppRange_where.type.d682d6.1 = converted constants.%ConstRange, %facet_value.loc45_37.4 [concrete = constants.%facet_value.d86]
// CHECK:STDOUT:   %specific_fn.loc45_37.2: <specific function> = specific_function %impl.elem3, @T.as_type.as.Iterate.impl.Next(constants.%facet_value.d86) [concrete = constants.%T.as_type.as.Iterate.impl.Next.specific_fn]
// CHECK:STDOUT:   %bound_method.loc45_37.4: <bound method> = bound_method %c.ref, %specific_fn.loc45_37.2
// CHECK:STDOUT:   %.loc45_37.5: ref %Optional.d4a = temporary_storage
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.call: init %Optional.d4a to %.loc45_37.5 = call %bound_method.loc45_37.4(%c.ref, %var)
// CHECK:STDOUT:   %.loc45_37.6: ref %Optional.d4a = temporary %.loc45_37.5, %T.as_type.as.Iterate.impl.Next.call
// CHECK:STDOUT:   %.loc45_37.7: %Optional.HasValue.type.141 = specific_constant imports.%Core.import_ref.c82, @Optional(constants.%OptionalStorage.facet.244) [concrete = constants.%Optional.HasValue.01b]
// CHECK:STDOUT:   %HasValue.ref: %Optional.HasValue.type.141 = name_ref HasValue, %.loc45_37.7 [concrete = constants.%Optional.HasValue.01b]
// CHECK:STDOUT:   %Optional.HasValue.bound: <bound method> = bound_method %.loc45_37.6, %HasValue.ref
// CHECK:STDOUT:   %Optional.HasValue.specific_fn: <specific function> = specific_function %HasValue.ref, @Optional.HasValue(constants.%OptionalStorage.facet.244) [concrete = constants.%Optional.HasValue.specific_fn.18d]
// CHECK:STDOUT:   %bound_method.loc45_37.5: <bound method> = bound_method %.loc45_37.6, %Optional.HasValue.specific_fn
// CHECK:STDOUT:   %.loc45_37.8: %Optional.d4a = acquire_value %.loc45_37.6
// CHECK:STDOUT:   %Optional.HasValue.call: init bool = call %bound_method.loc45_37.5(%.loc45_37.8)
// CHECK:STDOUT:   %.loc45_37.9: bool = value_of_initializer %Optional.HasValue.call
// CHECK:STDOUT:   %.loc45_37.10: bool = converted %Optional.HasValue.call, %.loc45_37.9
// CHECK:STDOUT:   if %.loc45_37.10 br !for.body else br !for.done
// CHECK:STDOUT:
// CHECK:STDOUT: !for.body:
// CHECK:STDOUT:   %.loc45_37.11: %Optional.Get.type.28b = specific_constant imports.%Core.import_ref.29e, @Optional(constants.%OptionalStorage.facet.244) [concrete = constants.%Optional.Get.90b]
// CHECK:STDOUT:   %Get.ref: %Optional.Get.type.28b = name_ref Get, %.loc45_37.11 [concrete = constants.%Optional.Get.90b]
// CHECK:STDOUT:   %Optional.Get.bound: <bound method> = bound_method %.loc45_37.6, %Get.ref
// CHECK:STDOUT:   %Optional.Get.specific_fn: <specific function> = specific_function %Get.ref, @Optional.Get(constants.%OptionalStorage.facet.244) [concrete = constants.%Optional.Get.specific_fn.f55]
// CHECK:STDOUT:   %bound_method.loc45_37.6: <bound method> = bound_method %.loc45_37.6, %Optional.Get.specific_fn
// CHECK:STDOUT:   %.loc45_37.12: ref %ValueType = temporary_storage
// CHECK:STDOUT:   %.loc45_37.13: %Optional.d4a = acquire_value %.loc45_37.6
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
// CHECK:STDOUT:   %Destroy.facet.loc45_37.1: %Destroy.type = facet_value constants.%ValueType, (constants.%custom_witness.0a1, constants.%DestroyImpl.impl_witness.488) [concrete = constants.%Destroy.facet.d1b]
// CHECK:STDOUT:   %.loc45_37.16: %Destroy.type = converted constants.%ValueType, %Destroy.facet.loc45_37.1 [concrete = constants.%Destroy.facet.d1b]
// CHECK:STDOUT:   %Destroy.facet.loc45_37.2: %Destroy.type = facet_value constants.%ValueType, (constants.%custom_witness.0a1, constants.%DestroyImpl.impl_witness.488) [concrete = constants.%Destroy.facet.d1b]
// CHECK:STDOUT:   %.loc45_37.17: %Destroy.type = converted constants.%ValueType, %Destroy.facet.loc45_37.2 [concrete = constants.%Destroy.facet.d1b]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc45_37.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc45_37.14)
// CHECK:STDOUT:   %Destroy.facet.loc45_37.3: %Destroy.type = facet_value constants.%Optional.d4a, (constants.%custom_witness.b80ef6.7, constants.%DestroyImpl.impl_witness.ce9) [concrete = constants.%Destroy.facet.7d9]
// CHECK:STDOUT:   %.loc45_37.18: %Destroy.type = converted constants.%Optional.d4a, %Destroy.facet.loc45_37.3 [concrete = constants.%Destroy.facet.7d9]
// CHECK:STDOUT:   %Destroy.facet.loc45_37.4: %Destroy.type = facet_value constants.%Optional.d4a, (constants.%custom_witness.b80ef6.7, constants.%DestroyImpl.impl_witness.ce9) [concrete = constants.%Destroy.facet.7d9]
// CHECK:STDOUT:   %.loc45_37.19: %Destroy.type = converted constants.%Optional.d4a, %Destroy.facet.loc45_37.4 [concrete = constants.%Destroy.facet.7d9]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc45_37.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc45_37.6)
// CHECK:STDOUT:   br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT: !for.done:
// CHECK:STDOUT:   %Destroy.facet.loc45_37.5: %Destroy.type = facet_value constants.%Optional.d4a, (constants.%custom_witness.b80ef6.7, constants.%DestroyImpl.impl_witness.ce9) [concrete = constants.%Destroy.facet.7d9]
// CHECK:STDOUT:   %.loc45_37.20: %Destroy.type = converted constants.%Optional.d4a, %Destroy.facet.loc45_37.5 [concrete = constants.%Destroy.facet.7d9]
// CHECK:STDOUT:   %Destroy.facet.loc45_37.6: %Destroy.type = facet_value constants.%Optional.d4a, (constants.%custom_witness.b80ef6.7, constants.%DestroyImpl.impl_witness.ce9) [concrete = constants.%Destroy.facet.7d9]
// CHECK:STDOUT:   %.loc45_37.21: %Destroy.type = converted constants.%Optional.d4a, %Destroy.facet.loc45_37.6 [concrete = constants.%Destroy.facet.7d9]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc45_37.3: init %empty_tuple.type = call %SelfDestruct.specific_fn.3(%.loc45_37.6)
// CHECK:STDOUT:   %Destroy.facet.loc45_37.7: %Destroy.type = facet_value constants.%tuple.type.7a6, (constants.%custom_witness.b80ef6.8, constants.%DestroyImpl.impl_witness.b22) [concrete = constants.%Destroy.facet.959]
// CHECK:STDOUT:   %.loc45_37.22: %Destroy.type = converted constants.%tuple.type.7a6, %Destroy.facet.loc45_37.7 [concrete = constants.%Destroy.facet.959]
// CHECK:STDOUT:   %Destroy.facet.loc45_37.8: %Destroy.type = facet_value constants.%tuple.type.7a6, (constants.%custom_witness.b80ef6.8, constants.%DestroyImpl.impl_witness.b22) [concrete = constants.%Destroy.facet.959]
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
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.7ce
// CHECK:STDOUT:   %r.param_patt.loc27_77.2 => constants.%r.param_patt.80f
// CHECK:STDOUT:   %r.patt.loc27_77.2 => constants.%r.patt.7ca
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestIterate(constants.%facet_value.8da) {
// CHECK:STDOUT:   %R.patt.loc27_17.2 => constants.%R.patt
// CHECK:STDOUT:   %R.loc27_17.1 => constants.%facet_value.8da
// CHECK:STDOUT:   %R.as_type.loc27_79.1 => constants.%ConstRange
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.c1d
// CHECK:STDOUT:   %r.param_patt.loc27_77.2 => constants.%r.param_patt.ac5
// CHECK:STDOUT:   %r.patt.loc27_77.2 => constants.%r.patt.97f
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc27 => constants.%complete_type.357
// CHECK:STDOUT:   %Iterate.lookup_impl_witness => constants.%Iterate.impl_witness.939
// CHECK:STDOUT:   %Iterate.facet.loc31_37.5 => constants.%Iterate.facet.ce6
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type => constants.%Iterate.WithSelf.NewCursor.type.df3
// CHECK:STDOUT:   %.loc31_37.26 => constants.%.472
// CHECK:STDOUT:   %impl.elem2.loc31_37.2 => constants.%T.as_type.as.Iterate.impl.NewCursor.d53
// CHECK:STDOUT:   %specific_impl_fn.loc31_37.3 => constants.%T.as_type.as.Iterate.impl.NewCursor.specific_fn
// CHECK:STDOUT:   %impl.elem1 => constants.%Destroy.facet.959
// CHECK:STDOUT:   %as_type => constants.%tuple.type.7a6
// CHECK:STDOUT:   %require_complete.1 => constants.%complete_type.cf0
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type => constants.%Iterate.WithSelf.Next.type.869
// CHECK:STDOUT:   %.loc31_37.27 => constants.%.135
// CHECK:STDOUT:   %impl.elem3.loc31_37.2 => constants.%T.as_type.as.Iterate.impl.Next.2f5
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
// CHECK:STDOUT:   %T.as_type.c1cc3e.1: type = facet_access_type %T.f84392.1 [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.lookup_impl_witness.70f: <witness> = lookup_impl_witness %T.f84392.1, @SubobjectDestroy [symbolic]
// CHECK:STDOUT:   %DestroyImpl.lookup_impl_witness.bf2: <witness> = lookup_impl_witness %T.f84392.1, @DestroyImpl [symbolic]
// CHECK:STDOUT:   %Destroy.facet.dc4: %Destroy.type = facet_value %T.as_type.c1cc3e.1, (%SubobjectDestroy.lookup_impl_witness.70f, %DestroyImpl.lookup_impl_witness.bf2) [symbolic]
// CHECK:STDOUT:   %MaybeUnformed.348: type = class_type @MaybeUnformed, @MaybeUnformed(%Destroy.facet.dc4) [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.476: type = class_type @DefaultOptionalStorage, @DefaultOptionalStorage(%T.f84392.1) [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.elem.4a2: type = unbound_element_type %DefaultOptionalStorage.476, bool [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.elem.ff3: type = unbound_element_type %DefaultOptionalStorage.476, %MaybeUnformed.348 [symbolic]
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
// CHECK:STDOUT:   %ValueType.Op.type.db6aa9.1: type = fn_type @ValueType.Op.1 [concrete]
// CHECK:STDOUT:   %ValueType.Op.4590ed.1: %ValueType.Op.type.db6aa9.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.55b: <witness> = custom_witness (%ValueType.Op.4590ed.1), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.5fb: %SubobjectDestroy.type = facet_value %ValueType, (%custom_witness.55b) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.7f1: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.5fb) [concrete]
// CHECK:STDOUT:   %const.91f: type = const_type %ValueType [concrete]
// CHECK:STDOUT:   %ptr.5f4: type = ptr_type %const.91f [concrete]
// CHECK:STDOUT:   %ptr.9c6: type = ptr_type %ValueType [concrete]
// CHECK:STDOUT:   %ValueType.Op.type.db6aa9.2: type = fn_type @ValueType.Op.2 [concrete]
// CHECK:STDOUT:   %ValueType.Op.4590ed.2: %ValueType.Op.type.db6aa9.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.587: <witness> = custom_witness (%ValueType.Op.4590ed.2), @Copy [concrete]
// CHECK:STDOUT:   %facet_value.6d3: %facet_type.bd0 = facet_value %ValueType, (%custom_witness.55b, %DestroyImpl.impl_witness.7f1, %custom_witness.587) [concrete]
// CHECK:STDOUT:   %Iterate.lookup_impl_witness.1d8: <witness> = lookup_impl_witness %.Self.500, @Iterate [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.4ce: %facet_type.bd0 = impl_witness_access %Iterate.lookup_impl_witness.1d8, element0 [symbolic_self]
// CHECK:STDOUT:   %Iterate_where.type.ad0: type = facet_type <@Iterate where %impl.elem0.4ce = %facet_value.6d3> [concrete]
// CHECK:STDOUT:   %pattern_type.f28: type = pattern_type %Iterate_where.type.ad0 [concrete]
// CHECK:STDOUT:   %R.patt: %pattern_type.f28 = symbolic_binding_pattern R, 0 [symbolic]
// CHECK:STDOUT:   %R: %Iterate_where.type.ad0 = symbolic_binding R, 0 [symbolic]
// CHECK:STDOUT:   %R.as_type: type = facet_access_type %R [symbolic]
// CHECK:STDOUT:   %pattern_type.2f7: type = pattern_type %R.as_type [symbolic]
// CHECK:STDOUT:   %r.param_patt: %pattern_type.2f7 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %r.patt: %pattern_type.2f7 = wrapper_binding_pattern r, %r.param_patt [symbolic]
// CHECK:STDOUT:   %i.patt: %pattern_type.d2d = value_binding_pattern i [concrete]
// CHECK:STDOUT:   %Iterate.lookup_impl_witness.afc: <witness> = lookup_impl_witness %R, @Iterate [symbolic]
// CHECK:STDOUT:   %Iterate.facet.942: %Iterate.type = facet_value %R.as_type, (%Iterate.lookup_impl_witness.afc) [symbolic]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.73c: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.942) [symbolic]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.f8d: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.942) [symbolic]
// CHECK:STDOUT:   %.848: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.73c, %Iterate.facet.942 [symbolic]
// CHECK:STDOUT:   %impl.elem2.2ea: %.848 = impl_witness_access %Iterate.lookup_impl_witness.afc, element2 [symbolic]
// CHECK:STDOUT:   %impl.elem1.7d3: %Destroy.type = impl_witness_access %Iterate.lookup_impl_witness.afc, element1 [symbolic]
// CHECK:STDOUT:   %as_type.901: type = facet_access_type %impl.elem1.7d3 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.105: <specific function> = specific_impl_function %impl.elem2.2ea, @Iterate.WithSelf.NewCursor(%Iterate.facet.942) [symbolic]
// CHECK:STDOUT:   %.b21: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.f8d, %Iterate.facet.942 [symbolic]
// CHECK:STDOUT:   %impl.elem3.509: %.b21 = impl_witness_access %Iterate.lookup_impl_witness.afc, element3 [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.db0: type = class_type @DefaultOptionalStorage, @DefaultOptionalStorage(%facet_value.6d3) [concrete]
// CHECK:STDOUT:   %OptionalStorage.impl_witness.cc0: <witness> = impl_witness imports.%OptionalStorage.impl_witness_table.1c4, @T.as_type.as.OptionalStorage.impl(%facet_value.6d3) [concrete]
// CHECK:STDOUT:   %OptionalStorage.facet.86d: %OptionalStorage.type = facet_value %ValueType, (%OptionalStorage.impl_witness.cc0) [concrete]
// CHECK:STDOUT:   %Optional.35b: type = class_type @Optional, @Optional(%OptionalStorage.facet.86d) [concrete]
// CHECK:STDOUT:   %specific_impl_fn.ccf: <specific function> = specific_impl_function %impl.elem3.509, @Iterate.WithSelf.Next(%Iterate.facet.942) [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.type.5fc: type = fn_type @Optional.HasValue, @Optional(%OptionalStorage.facet.86d) [concrete]
// CHECK:STDOUT:   %Optional.HasValue.462: %Optional.HasValue.type.5fc = struct_value () [concrete]
// CHECK:STDOUT:   %Optional.Get.type.fad: type = fn_type @Optional.Get, @Optional(%OptionalStorage.facet.86d) [concrete]
// CHECK:STDOUT:   %Optional.Get.707: %Optional.Get.type.fad = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.facet.423: %Destroy.type = facet_value %ValueType, (%custom_witness.55b, %DestroyImpl.impl_witness.7f1) [concrete]
// CHECK:STDOUT:   %MaybeUnformed.149: type = class_type @MaybeUnformed, @MaybeUnformed(%Destroy.facet.423) [concrete]
// CHECK:STDOUT:   %.dd19: type = maybe_unformed_type %ValueType [concrete]
// CHECK:STDOUT:   %DefaultOptionalStorage.elem.649: type = unbound_element_type %DefaultOptionalStorage.db0, %MaybeUnformed.149 [concrete]
// CHECK:STDOUT:   %DefaultOptionalStorage.elem.21a: type = unbound_element_type %DefaultOptionalStorage.db0, bool [concrete]
// CHECK:STDOUT:   %struct_type.value.has_value.88e: type = struct_type {.value: %MaybeUnformed.149, .has_value: bool} [concrete]
// CHECK:STDOUT:   %Optional.HasValue.specific_fn: <specific function> = specific_function %Optional.HasValue.462, @Optional.HasValue(%OptionalStorage.facet.86d) [concrete]
// CHECK:STDOUT:   %Optional.Get.specific_fn: <specific function> = specific_function %Optional.Get.707, @Optional.Get(%OptionalStorage.facet.86d) [concrete]
// CHECK:STDOUT:   %operator_PlusEqual__carbon_thunk.type: type = fn_type @operator_PlusEqual__carbon_thunk [concrete]
// CHECK:STDOUT:   %operator_PlusEqual__carbon_thunk: %operator_PlusEqual__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.3: type = fn_type @SubobjectDestroy.WithSelf.Op.loc41_44.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.3: %SubobjectDestroy.WithSelf.Op.type.6870fd.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.3: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.3), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.4: type = fn_type @SubobjectDestroy.WithSelf.Op.loc41_44.3 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.4: %SubobjectDestroy.WithSelf.Op.type.6870fd.4 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.4: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.4), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.a87: %SubobjectDestroy.type = facet_value bool, (%custom_witness.b80ef6.4) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.778: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.a87) [concrete]
// CHECK:STDOUT:   %Destroy.facet.80c: %Destroy.type = facet_value bool, (%custom_witness.b80ef6.4, %DestroyImpl.impl_witness.778) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.e25: %SubobjectDestroy.type = facet_value %MaybeUnformed.149, (%custom_witness.b80ef6.3) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.faa: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.e25) [concrete]
// CHECK:STDOUT:   %Destroy.facet.6cf: %Destroy.type = facet_value %MaybeUnformed.149, (%custom_witness.b80ef6.3, %DestroyImpl.impl_witness.faa) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.7: type = fn_type @SubobjectDestroy.WithSelf.Op.loc41_44.6 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.7: %SubobjectDestroy.WithSelf.Op.type.6870fd.7 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.7: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.7), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.f77: %SubobjectDestroy.type = facet_value %Optional.35b, (%custom_witness.b80ef6.7) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.ed2: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.f77) [concrete]
// CHECK:STDOUT:   %Destroy.facet.610: %Destroy.type = facet_value %Optional.35b, (%custom_witness.b80ef6.7, %DestroyImpl.impl_witness.ed2) [concrete]
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
// CHECK:STDOUT:   %Core.import_ref.e86: @DefaultOptionalStorage.%DefaultOptionalStorage.elem.1 (%DefaultOptionalStorage.elem.ff3) = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [concrete = %field_decl.bce]
// CHECK:STDOUT:   %Core.import_ref.9ce: @DefaultOptionalStorage.%DefaultOptionalStorage.elem.2 (%DefaultOptionalStorage.elem.4a2) = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [concrete = %field_decl.dbe]
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
// CHECK:STDOUT:   %Core.import_ref.b0d = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %field_decl.dbe: @DefaultOptionalStorage.%DefaultOptionalStorage.elem.2 (%DefaultOptionalStorage.elem.4a2) = field_decl has_value, element1, %Core.import_ref.b0d in [concrete] {}
// CHECK:STDOUT:   %Core.import_ref.45d = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %field_decl.bce: @DefaultOptionalStorage.%DefaultOptionalStorage.elem.1 (%DefaultOptionalStorage.elem.ff3) = field_decl value, element0, %Core.import_ref.45d in [concrete] {}
// CHECK:STDOUT:   %MutableRange.decl: type = class_decl @MutableRange [concrete = constants.%MutableRange] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @TestIterate(%R.loc37_17.2: %Iterate_where.type.ad0) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Iterate.lookup_impl_witness: <witness> = lookup_impl_witness %R.loc37_17.1, @Iterate [symbolic = %Iterate.lookup_impl_witness (constants.%Iterate.lookup_impl_witness.afc)]
// CHECK:STDOUT:   %Iterate.facet.loc41_44.5: %Iterate.type = facet_value %R.as_type.loc37_86.1, (%Iterate.lookup_impl_witness) [symbolic = %Iterate.facet.loc41_44.5 (constants.%Iterate.facet.942)]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.loc41_44.5) [symbolic = %Iterate.WithSelf.NewCursor.type (constants.%Iterate.WithSelf.NewCursor.type.73c)]
// CHECK:STDOUT:   %.loc41_44.26: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type, %Iterate.facet.loc41_44.5 [symbolic = %.loc41_44.26 (constants.%.848)]
// CHECK:STDOUT:   %impl.elem2.loc41_44.2: @TestIterate.%.loc41_44.26 (%.848) = impl_witness_access %Iterate.lookup_impl_witness, element2 [symbolic = %impl.elem2.loc41_44.2 (constants.%impl.elem2.2ea)]
// CHECK:STDOUT:   %specific_impl_fn.loc41_44.3: <specific function> = specific_impl_function %impl.elem2.loc41_44.2, @Iterate.WithSelf.NewCursor(%Iterate.facet.loc41_44.5) [symbolic = %specific_impl_fn.loc41_44.3 (constants.%specific_impl_fn.105)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.loc41_44.5) [symbolic = %Iterate.WithSelf.Next.type (constants.%Iterate.WithSelf.Next.type.f8d)]
// CHECK:STDOUT:   %.loc41_44.27: type = fn_type_with_self_type %Iterate.WithSelf.Next.type, %Iterate.facet.loc41_44.5 [symbolic = %.loc41_44.27 (constants.%.b21)]
// CHECK:STDOUT:   %impl.elem3.loc41_44.2: @TestIterate.%.loc41_44.27 (%.b21) = impl_witness_access %Iterate.lookup_impl_witness, element3 [symbolic = %impl.elem3.loc41_44.2 (constants.%impl.elem3.509)]
// CHECK:STDOUT:   %specific_impl_fn.loc41_44.4: <specific function> = specific_impl_function %impl.elem3.loc41_44.2, @Iterate.WithSelf.Next(%Iterate.facet.loc41_44.5) [symbolic = %specific_impl_fn.loc41_44.4 (constants.%specific_impl_fn.ccf)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%r.param: @TestIterate.%R.as_type.loc37_86.1 (%R.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %i.patt: %pattern_type.d2d = value_binding_pattern i [concrete = constants.%i.patt]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %r.ref: @TestIterate.%R.as_type.loc37_86.1 (%R.as_type) = name_ref r, %r
// CHECK:STDOUT:     %impl.elem2.loc41_44.1: @TestIterate.%.loc41_44.26 (%.848) = impl_witness_access constants.%Iterate.lookup_impl_witness.afc, element2 [symbolic = %impl.elem2.loc41_44.2 (constants.%impl.elem2.2ea)]
// CHECK:STDOUT:     %bound_method.loc41_44.1: <bound method> = bound_method %r.ref, %impl.elem2.loc41_44.1
// CHECK:STDOUT:     %Iterate.facet.loc41_44.1: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.afc) [symbolic = %Iterate.facet.loc41_44.5 (constants.%Iterate.facet.942)]
// CHECK:STDOUT:     %.loc41_44.1: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc41_44.1 [symbolic = %Iterate.facet.loc41_44.5 (constants.%Iterate.facet.942)]
// CHECK:STDOUT:     %Iterate.facet.loc41_44.2: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.afc) [symbolic = %Iterate.facet.loc41_44.5 (constants.%Iterate.facet.942)]
// CHECK:STDOUT:     %.loc41_44.2: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc41_44.2 [symbolic = %Iterate.facet.loc41_44.5 (constants.%Iterate.facet.942)]
// CHECK:STDOUT:     %specific_impl_fn.loc41_44.1: <specific function> = specific_impl_function %impl.elem2.loc41_44.1, @Iterate.WithSelf.NewCursor(constants.%Iterate.facet.942) [symbolic = %specific_impl_fn.loc41_44.3 (constants.%specific_impl_fn.105)]
// CHECK:STDOUT:     %bound_method.loc41_44.2: <bound method> = bound_method %r.ref, %specific_impl_fn.loc41_44.1
// CHECK:STDOUT:     %var: ref @TestIterate.%as_type (%as_type.901) = var_storage invalid
// CHECK:STDOUT:     %Iterate.WithSelf.NewCursor.call: init @TestIterate.%as_type (%as_type.901) to %var = call %bound_method.loc41_44.2(%r.ref)
// CHECK:STDOUT:     assign %var, %Iterate.WithSelf.NewCursor.call
// CHECK:STDOUT:     br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.next:
// CHECK:STDOUT:     %impl.elem3.loc41_44.1: @TestIterate.%.loc41_44.27 (%.b21) = impl_witness_access constants.%Iterate.lookup_impl_witness.afc, element3 [symbolic = %impl.elem3.loc41_44.2 (constants.%impl.elem3.509)]
// CHECK:STDOUT:     %bound_method.loc41_44.3: <bound method> = bound_method %r.ref, %impl.elem3.loc41_44.1
// CHECK:STDOUT:     %Iterate.facet.loc41_44.3: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.afc) [symbolic = %Iterate.facet.loc41_44.5 (constants.%Iterate.facet.942)]
// CHECK:STDOUT:     %.loc41_44.3: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc41_44.3 [symbolic = %Iterate.facet.loc41_44.5 (constants.%Iterate.facet.942)]
// CHECK:STDOUT:     %Iterate.facet.loc41_44.4: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.afc) [symbolic = %Iterate.facet.loc41_44.5 (constants.%Iterate.facet.942)]
// CHECK:STDOUT:     %.loc41_44.4: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc41_44.4 [symbolic = %Iterate.facet.loc41_44.5 (constants.%Iterate.facet.942)]
// CHECK:STDOUT:     %.loc41_44.5: %Destroy.type = converted constants.%as_type.901, constants.%impl.elem1.7d3 [symbolic = %impl.elem1 (constants.%impl.elem1.7d3)]
// CHECK:STDOUT:     %.loc41_44.6: %Destroy.type = converted constants.%as_type.901, constants.%impl.elem1.7d3 [symbolic = %impl.elem1 (constants.%impl.elem1.7d3)]
// CHECK:STDOUT:     %specific_impl_fn.loc41_44.2: <specific function> = specific_impl_function %impl.elem3.loc41_44.1, @Iterate.WithSelf.Next(constants.%Iterate.facet.942) [symbolic = %specific_impl_fn.loc41_44.4 (constants.%specific_impl_fn.ccf)]
// CHECK:STDOUT:     %bound_method.loc41_44.4: <bound method> = bound_method %r.ref, %specific_impl_fn.loc41_44.2
// CHECK:STDOUT:     %.loc41_44.7: ref %Optional.35b = temporary_storage
// CHECK:STDOUT:     %Iterate.WithSelf.Next.call: init %Optional.35b to %.loc41_44.7 = call %bound_method.loc41_44.4(%r.ref, %var)
// CHECK:STDOUT:     %.loc41_44.8: ref %Optional.35b = temporary %.loc41_44.7, %Iterate.WithSelf.Next.call
// CHECK:STDOUT:     %.loc41_44.9: %Optional.HasValue.type.5fc = specific_constant imports.%Core.import_ref.c82, @Optional(constants.%OptionalStorage.facet.86d) [concrete = constants.%Optional.HasValue.462]
// CHECK:STDOUT:     %HasValue.ref: %Optional.HasValue.type.5fc = name_ref HasValue, %.loc41_44.9 [concrete = constants.%Optional.HasValue.462]
// CHECK:STDOUT:     %Optional.HasValue.bound: <bound method> = bound_method %.loc41_44.8, %HasValue.ref
// CHECK:STDOUT:     %Optional.HasValue.specific_fn: <specific function> = specific_function %HasValue.ref, @Optional.HasValue(constants.%OptionalStorage.facet.86d) [concrete = constants.%Optional.HasValue.specific_fn]
// CHECK:STDOUT:     %bound_method.loc41_44.5: <bound method> = bound_method %.loc41_44.8, %Optional.HasValue.specific_fn
// CHECK:STDOUT:     %.loc41_44.10: %Optional.35b = acquire_value %.loc41_44.8
// CHECK:STDOUT:     %Optional.HasValue.call: init bool = call %bound_method.loc41_44.5(%.loc41_44.10)
// CHECK:STDOUT:     %.loc41_44.11: bool = value_of_initializer %Optional.HasValue.call
// CHECK:STDOUT:     %.loc41_44.12: bool = converted %Optional.HasValue.call, %.loc41_44.11
// CHECK:STDOUT:     if %.loc41_44.12 br !for.body else br !for.done
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.body:
// CHECK:STDOUT:     %.loc41_44.13: %Optional.Get.type.fad = specific_constant imports.%Core.import_ref.29e, @Optional(constants.%OptionalStorage.facet.86d) [concrete = constants.%Optional.Get.707]
// CHECK:STDOUT:     %Get.ref: %Optional.Get.type.fad = name_ref Get, %.loc41_44.13 [concrete = constants.%Optional.Get.707]
// CHECK:STDOUT:     %Optional.Get.bound: <bound method> = bound_method %.loc41_44.8, %Get.ref
// CHECK:STDOUT:     %Optional.Get.specific_fn: <specific function> = specific_function %Get.ref, @Optional.Get(constants.%OptionalStorage.facet.86d) [concrete = constants.%Optional.Get.specific_fn]
// CHECK:STDOUT:     %bound_method.loc41_44.6: <bound method> = bound_method %.loc41_44.8, %Optional.Get.specific_fn
// CHECK:STDOUT:     %.loc41_44.14: ref %ValueType = temporary_storage
// CHECK:STDOUT:     %.loc41_44.15: %Optional.35b = acquire_value %.loc41_44.8
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
// CHECK:STDOUT:     %Destroy.facet.loc41_44.1: %Destroy.type = facet_value constants.%ValueType, (constants.%custom_witness.55b, constants.%DestroyImpl.impl_witness.7f1) [concrete = constants.%Destroy.facet.423]
// CHECK:STDOUT:     %.loc41_44.18: %Destroy.type = converted constants.%ValueType, %Destroy.facet.loc41_44.1 [concrete = constants.%Destroy.facet.423]
// CHECK:STDOUT:     %Destroy.facet.loc41_44.2: %Destroy.type = facet_value constants.%ValueType, (constants.%custom_witness.55b, constants.%DestroyImpl.impl_witness.7f1) [concrete = constants.%Destroy.facet.423]
// CHECK:STDOUT:     %.loc41_44.19: %Destroy.type = converted constants.%ValueType, %Destroy.facet.loc41_44.2 [concrete = constants.%Destroy.facet.423]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %SelfDestruct.call.loc41_44.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc41_44.16)
// CHECK:STDOUT:     %Destroy.facet.loc41_44.3: %Destroy.type = facet_value constants.%Optional.35b, (constants.%custom_witness.b80ef6.7, constants.%DestroyImpl.impl_witness.ed2) [concrete = constants.%Destroy.facet.610]
// CHECK:STDOUT:     %.loc41_44.20: %Destroy.type = converted constants.%Optional.35b, %Destroy.facet.loc41_44.3 [concrete = constants.%Destroy.facet.610]
// CHECK:STDOUT:     %Destroy.facet.loc41_44.4: %Destroy.type = facet_value constants.%Optional.35b, (constants.%custom_witness.b80ef6.7, constants.%DestroyImpl.impl_witness.ed2) [concrete = constants.%Destroy.facet.610]
// CHECK:STDOUT:     %.loc41_44.21: %Destroy.type = converted constants.%Optional.35b, %Destroy.facet.loc41_44.4 [concrete = constants.%Destroy.facet.610]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %SelfDestruct.call.loc41_44.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc41_44.8)
// CHECK:STDOUT:     br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.done:
// CHECK:STDOUT:     %Destroy.facet.loc41_44.5: %Destroy.type = facet_value constants.%Optional.35b, (constants.%custom_witness.b80ef6.7, constants.%DestroyImpl.impl_witness.ed2) [concrete = constants.%Destroy.facet.610]
// CHECK:STDOUT:     %.loc41_44.22: %Destroy.type = converted constants.%Optional.35b, %Destroy.facet.loc41_44.5 [concrete = constants.%Destroy.facet.610]
// CHECK:STDOUT:     %Destroy.facet.loc41_44.6: %Destroy.type = facet_value constants.%Optional.35b, (constants.%custom_witness.b80ef6.7, constants.%DestroyImpl.impl_witness.ed2) [concrete = constants.%Destroy.facet.610]
// CHECK:STDOUT:     %.loc41_44.23: %Destroy.type = converted constants.%Optional.35b, %Destroy.facet.loc41_44.6 [concrete = constants.%Destroy.facet.610]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %SelfDestruct.call.loc41_44.3: init %empty_tuple.type = call %SelfDestruct.specific_fn.3(%.loc41_44.8)
// CHECK:STDOUT:     %.loc41_44.24: %Destroy.type = converted constants.%as_type.901, constants.%impl.elem1.7d3 [symbolic = %impl.elem1 (constants.%impl.elem1.7d3)]
// CHECK:STDOUT:     %.loc41_44.25: %Destroy.type = converted constants.%as_type.901, constants.%impl.elem1.7d3 [symbolic = %impl.elem1 (constants.%impl.elem1.7d3)]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %SelfDestruct.call.loc41_44.4: init %empty_tuple.type = call %SelfDestruct.specific_fn.4(%var)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc41_44.1(%self.param: ref %.dd19) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc41_44.2(%self.param: ref %MaybeUnformed.149) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc41_44.3(%self.param: ref bool) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc41_44.4(%self.param: ref %struct_type.value.has_value.88e) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc41_44.1: ref bool = struct_access %self.param, element1
// CHECK:STDOUT:   %Destroy.facet.loc41_44.1: %Destroy.type = facet_value bool, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.778) [concrete = constants.%Destroy.facet.80c]
// CHECK:STDOUT:   %.loc41_44.2: %Destroy.type = converted bool, %Destroy.facet.loc41_44.1 [concrete = constants.%Destroy.facet.80c]
// CHECK:STDOUT:   %Destroy.facet.loc41_44.2: %Destroy.type = facet_value bool, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.778) [concrete = constants.%Destroy.facet.80c]
// CHECK:STDOUT:   %.loc41_44.3: %Destroy.type = converted bool, %Destroy.facet.loc41_44.2 [concrete = constants.%Destroy.facet.80c]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc41_44.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc41_44.1)
// CHECK:STDOUT:   %.loc41_44.4: ref %MaybeUnformed.149 = struct_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc41_44.3: %Destroy.type = facet_value constants.%MaybeUnformed.149, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.faa) [concrete = constants.%Destroy.facet.6cf]
// CHECK:STDOUT:   %.loc41_44.5: %Destroy.type = converted constants.%MaybeUnformed.149, %Destroy.facet.loc41_44.3 [concrete = constants.%Destroy.facet.6cf]
// CHECK:STDOUT:   %Destroy.facet.loc41_44.4: %Destroy.type = facet_value constants.%MaybeUnformed.149, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.faa) [concrete = constants.%Destroy.facet.6cf]
// CHECK:STDOUT:   %.loc41_44.6: %Destroy.type = converted constants.%MaybeUnformed.149, %Destroy.facet.loc41_44.4 [concrete = constants.%Destroy.facet.6cf]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc41_44.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc41_44.4)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc41_44.5(%self.param: ref %DefaultOptionalStorage.db0) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %has_value.ref: %DefaultOptionalStorage.elem.21a = name_ref has_value, imports.%Core.import_ref.9ce [concrete = imports.%field_decl.dbe]
// CHECK:STDOUT:   %.loc41_44.1: ref bool = class_element_access %self.param, element1
// CHECK:STDOUT:   %Destroy.facet.loc41_44.1: %Destroy.type = facet_value bool, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.778) [concrete = constants.%Destroy.facet.80c]
// CHECK:STDOUT:   %.loc41_44.2: %Destroy.type = converted bool, %Destroy.facet.loc41_44.1 [concrete = constants.%Destroy.facet.80c]
// CHECK:STDOUT:   %Destroy.facet.loc41_44.2: %Destroy.type = facet_value bool, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.778) [concrete = constants.%Destroy.facet.80c]
// CHECK:STDOUT:   %.loc41_44.3: %Destroy.type = converted bool, %Destroy.facet.loc41_44.2 [concrete = constants.%Destroy.facet.80c]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc41_44.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc41_44.1)
// CHECK:STDOUT:   %value.ref: %DefaultOptionalStorage.elem.649 = name_ref value, imports.%Core.import_ref.e86 [concrete = imports.%field_decl.bce]
// CHECK:STDOUT:   %.loc41_44.4: ref %MaybeUnformed.149 = class_element_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc41_44.3: %Destroy.type = facet_value constants.%MaybeUnformed.149, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.faa) [concrete = constants.%Destroy.facet.6cf]
// CHECK:STDOUT:   %.loc41_44.5: %Destroy.type = converted constants.%MaybeUnformed.149, %Destroy.facet.loc41_44.3 [concrete = constants.%Destroy.facet.6cf]
// CHECK:STDOUT:   %Destroy.facet.loc41_44.4: %Destroy.type = facet_value constants.%MaybeUnformed.149, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.faa) [concrete = constants.%Destroy.facet.6cf]
// CHECK:STDOUT:   %.loc41_44.6: %Destroy.type = converted constants.%MaybeUnformed.149, %Destroy.facet.loc41_44.4 [concrete = constants.%Destroy.facet.6cf]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc41_44.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc41_44.4)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc41_44.6(%self.param: ref %Optional.35b) {
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
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.2f7
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
// CHECK:STDOUT:   %T.as_type.c1cc3e.1: type = facet_access_type %T.f84392.1 [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.lookup_impl_witness.70f: <witness> = lookup_impl_witness %T.f84392.1, @SubobjectDestroy [symbolic]
// CHECK:STDOUT:   %DestroyImpl.lookup_impl_witness.bf2: <witness> = lookup_impl_witness %T.f84392.1, @DestroyImpl [symbolic]
// CHECK:STDOUT:   %Destroy.facet.dc4: %Destroy.type = facet_value %T.as_type.c1cc3e.1, (%SubobjectDestroy.lookup_impl_witness.70f, %DestroyImpl.lookup_impl_witness.bf2) [symbolic]
// CHECK:STDOUT:   %MaybeUnformed.348: type = class_type @MaybeUnformed, @MaybeUnformed(%Destroy.facet.dc4) [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.476: type = class_type @DefaultOptionalStorage, @DefaultOptionalStorage(%T.f84392.1) [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.elem.4a2: type = unbound_element_type %DefaultOptionalStorage.476, bool [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.elem.ff3: type = unbound_element_type %DefaultOptionalStorage.476, %MaybeUnformed.348 [symbolic]
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
// CHECK:STDOUT:   %ValueType.Op.type.9ce105.1: type = fn_type @ValueType.Op.1 [concrete]
// CHECK:STDOUT:   %ValueType.Op.038ea9.1: %ValueType.Op.type.9ce105.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.656: <witness> = custom_witness (%ValueType.Op.038ea9.1), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.536: %SubobjectDestroy.type = facet_value %ValueType, (%custom_witness.656) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.4c3: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.536) [concrete]
// CHECK:STDOUT:   %const.924: type = const_type %ValueType [concrete]
// CHECK:STDOUT:   %ptr.403: type = ptr_type %const.924 [concrete]
// CHECK:STDOUT:   %ptr.6ed: type = ptr_type %ValueType [concrete]
// CHECK:STDOUT:   %ValueType.Op.type.9ce105.2: type = fn_type @ValueType.Op.2 [concrete]
// CHECK:STDOUT:   %ValueType.Op.038ea9.2: %ValueType.Op.type.9ce105.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.1c5: <witness> = custom_witness (%ValueType.Op.038ea9.2), @Copy [concrete]
// CHECK:STDOUT:   %facet_value.282: %facet_type.bd0 = facet_value %ValueType, (%custom_witness.656, %DestroyImpl.impl_witness.4c3, %custom_witness.1c5) [concrete]
// CHECK:STDOUT:   %Iterate.lookup_impl_witness.1d8: <witness> = lookup_impl_witness %.Self.500, @Iterate [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.4ce: %facet_type.bd0 = impl_witness_access %Iterate.lookup_impl_witness.1d8, element0 [symbolic_self]
// CHECK:STDOUT:   %Iterate_where.type.97a: type = facet_type <@Iterate where %impl.elem0.4ce = %facet_value.282> [concrete]
// CHECK:STDOUT:   %pattern_type.4c1: type = pattern_type %Iterate_where.type.97a [concrete]
// CHECK:STDOUT:   %R.patt: %pattern_type.4c1 = symbolic_binding_pattern R, 0 [symbolic]
// CHECK:STDOUT:   %R: %Iterate_where.type.97a = symbolic_binding R, 0 [symbolic]
// CHECK:STDOUT:   %R.as_type: type = facet_access_type %R [symbolic]
// CHECK:STDOUT:   %pattern_type.2ae: type = pattern_type %R.as_type [symbolic]
// CHECK:STDOUT:   %r.param_patt.91c: %pattern_type.2ae = value_param_pattern [symbolic]
// CHECK:STDOUT:   %r.patt.03a: %pattern_type.2ae = wrapper_binding_pattern r, %r.param_patt.91c [symbolic]
// CHECK:STDOUT:   %i.patt: %pattern_type.e09 = value_binding_pattern i [concrete]
// CHECK:STDOUT:   %Iterate.lookup_impl_witness.0da: <witness> = lookup_impl_witness %R, @Iterate [symbolic]
// CHECK:STDOUT:   %Iterate.facet.394: %Iterate.type = facet_value %R.as_type, (%Iterate.lookup_impl_witness.0da) [symbolic]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.18c: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.394) [symbolic]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.f15: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.394) [symbolic]
// CHECK:STDOUT:   %.89f4: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.18c, %Iterate.facet.394 [symbolic]
// CHECK:STDOUT:   %impl.elem2.f19: %.89f4 = impl_witness_access %Iterate.lookup_impl_witness.0da, element2 [symbolic]
// CHECK:STDOUT:   %impl.elem1.626: %Destroy.type = impl_witness_access %Iterate.lookup_impl_witness.0da, element1 [symbolic]
// CHECK:STDOUT:   %as_type.71d: type = facet_access_type %impl.elem1.626 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.51c: <specific function> = specific_impl_function %impl.elem2.f19, @Iterate.WithSelf.NewCursor(%Iterate.facet.394) [symbolic]
// CHECK:STDOUT:   %.a1e: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.f15, %Iterate.facet.394 [symbolic]
// CHECK:STDOUT:   %impl.elem3.88b: %.a1e = impl_witness_access %Iterate.lookup_impl_witness.0da, element3 [symbolic]
// CHECK:STDOUT:   %DefaultOptionalStorage.dff: type = class_type @DefaultOptionalStorage, @DefaultOptionalStorage(%facet_value.282) [concrete]
// CHECK:STDOUT:   %OptionalStorage.impl_witness.5d6: <witness> = impl_witness imports.%OptionalStorage.impl_witness_table.1c4, @T.as_type.as.OptionalStorage.impl(%facet_value.282) [concrete]
// CHECK:STDOUT:   %OptionalStorage.facet.61a: %OptionalStorage.type = facet_value %ValueType, (%OptionalStorage.impl_witness.5d6) [concrete]
// CHECK:STDOUT:   %Optional.81f: type = class_type @Optional, @Optional(%OptionalStorage.facet.61a) [concrete]
// CHECK:STDOUT:   %specific_impl_fn.36a: <specific function> = specific_impl_function %impl.elem3.88b, @Iterate.WithSelf.Next(%Iterate.facet.394) [symbolic]
// CHECK:STDOUT:   %Optional.HasValue.type.db4: type = fn_type @Optional.HasValue, @Optional(%OptionalStorage.facet.61a) [concrete]
// CHECK:STDOUT:   %Optional.HasValue.97a: %Optional.HasValue.type.db4 = struct_value () [concrete]
// CHECK:STDOUT:   %Optional.Get.type.0d8: type = fn_type @Optional.Get, @Optional(%OptionalStorage.facet.61a) [concrete]
// CHECK:STDOUT:   %Optional.Get.2ea: %Optional.Get.type.0d8 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.facet.cac: %Destroy.type = facet_value %ValueType, (%custom_witness.656, %DestroyImpl.impl_witness.4c3) [concrete]
// CHECK:STDOUT:   %MaybeUnformed.4b9: type = class_type @MaybeUnformed, @MaybeUnformed(%Destroy.facet.cac) [concrete]
// CHECK:STDOUT:   %.8f3: type = maybe_unformed_type %ValueType [concrete]
// CHECK:STDOUT:   %DefaultOptionalStorage.elem.2ce: type = unbound_element_type %DefaultOptionalStorage.dff, %MaybeUnformed.4b9 [concrete]
// CHECK:STDOUT:   %DefaultOptionalStorage.elem.6f9: type = unbound_element_type %DefaultOptionalStorage.dff, bool [concrete]
// CHECK:STDOUT:   %struct_type.value.has_value.640: type = struct_type {.value: %MaybeUnformed.4b9, .has_value: bool} [concrete]
// CHECK:STDOUT:   %Optional.HasValue.specific_fn.92c: <specific function> = specific_function %Optional.HasValue.97a, @Optional.HasValue(%OptionalStorage.facet.61a) [concrete]
// CHECK:STDOUT:   %Optional.Get.specific_fn.132: <specific function> = specific_function %Optional.Get.2ea, @Optional.Get(%OptionalStorage.facet.61a) [concrete]
// CHECK:STDOUT:   %operator_PlusEqual__carbon_thunk.type: type = fn_type @operator_PlusEqual__carbon_thunk [concrete]
// CHECK:STDOUT:   %operator_PlusEqual__carbon_thunk: %operator_PlusEqual__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.3: type = fn_type @SubobjectDestroy.WithSelf.Op.loc48_41.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.3: %SubobjectDestroy.WithSelf.Op.type.6870fd.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.3: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.3), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.4: type = fn_type @SubobjectDestroy.WithSelf.Op.loc48_41.3 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.4: %SubobjectDestroy.WithSelf.Op.type.6870fd.4 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.4: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.4), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.a87: %SubobjectDestroy.type = facet_value bool, (%custom_witness.b80ef6.4) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.778: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.a87) [concrete]
// CHECK:STDOUT:   %Destroy.facet.80c: %Destroy.type = facet_value bool, (%custom_witness.b80ef6.4, %DestroyImpl.impl_witness.778) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.555: %SubobjectDestroy.type = facet_value %MaybeUnformed.4b9, (%custom_witness.b80ef6.3) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.e80: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.555) [concrete]
// CHECK:STDOUT:   %Destroy.facet.eca: %Destroy.type = facet_value %MaybeUnformed.4b9, (%custom_witness.b80ef6.3, %DestroyImpl.impl_witness.e80) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.7: type = fn_type @SubobjectDestroy.WithSelf.Op.loc48_41.6 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.7: %SubobjectDestroy.WithSelf.Op.type.6870fd.7 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.7: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.7), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.4ee: %SubobjectDestroy.type = facet_value %Optional.81f, (%custom_witness.b80ef6.7) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.e67: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.4ee) [concrete]
// CHECK:STDOUT:   %Destroy.facet.434: %Destroy.type = facet_value %Optional.81f, (%custom_witness.b80ef6.7, %DestroyImpl.impl_witness.e67) [concrete]
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
// CHECK:STDOUT:   %custom_witness.992: <witness> = custom_witness (%Iterator.Op.c90ac2.1), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.b40: %SubobjectDestroy.type = facet_value %Iterator, (%custom_witness.992) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.52a: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.b40) [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.c95ff6.2: type = fn_type @Iterator.Op.2 [concrete]
// CHECK:STDOUT:   %Iterator.Op.c90ac2.2: %Iterator.Op.type.c95ff6.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.d56: <witness> = custom_witness (%ValueType, %Iterator.Op.c90ac2.2), @CppUnsafeDeref [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.c95ff6.3: type = fn_type @Iterator.Op.3 [concrete]
// CHECK:STDOUT:   %Iterator.Op.c90ac2.3: %Iterator.Op.type.c95ff6.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.68a: <witness> = custom_witness (%Iterator.Op.c90ac2.3), @Inc [concrete]
// CHECK:STDOUT:   %Equal.type: type = fn_type @Equal [concrete]
// CHECK:STDOUT:   %Equal: %Equal.type = struct_value () [concrete]
// CHECK:STDOUT:   %NotEqual.type: type = fn_type @NotEqual [concrete]
// CHECK:STDOUT:   %NotEqual: %NotEqual.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.edd: <witness> = custom_witness (%Equal, %NotEqual), @EqWith, @EqWith(%Iterator) [concrete]
// CHECK:STDOUT:   %facet_value.b85: %CppRange_where.type.d682d6.1 = facet_value %MutableAndConstRange, (%custom_witness.660, %custom_witness.992, %DestroyImpl.impl_witness.52a, %custom_witness.d56, %custom_witness.68a, %custom_witness.edd, %custom_witness.656, %DestroyImpl.impl_witness.4c3, %custom_witness.1c5, %custom_witness.992, %DestroyImpl.impl_witness.52a) [concrete]
// CHECK:STDOUT:   %tuple.type.de6: type = tuple_type (%Iterator, %Iterator) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.8: type = fn_type @SubobjectDestroy.WithSelf.Op.3 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.8: %SubobjectDestroy.WithSelf.Op.type.6870fd.8 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.8: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.8), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.3e6: %SubobjectDestroy.type = facet_value %tuple.type.de6, (%custom_witness.b80ef6.8) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.2c4: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.3e6) [concrete]
// CHECK:STDOUT:   %complete_type.95d: <witness> = complete_type_witness %tuple.type.de6 [concrete]
// CHECK:STDOUT:   %Destroy.facet.a88: %Destroy.type = facet_value %tuple.type.de6, (%custom_witness.b80ef6.8, %DestroyImpl.impl_witness.2c4) [concrete]
// CHECK:STDOUT:   %Iterate.impl_witness.3b4: <witness> = impl_witness imports.%Iterate.impl_witness_table.121, @T.as_type.as.Iterate.impl(%facet_value.b85) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.type.316: type = fn_type @T.as_type.as.Iterate.impl.NewCursor, @T.as_type.as.Iterate.impl(%facet_value.b85) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.8e0: %T.as_type.as.Iterate.impl.NewCursor.type.316 = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.type.1bd: type = fn_type @T.as_type.as.Iterate.impl.Next, @T.as_type.as.Iterate.impl(%facet_value.b85) [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.353: %T.as_type.as.Iterate.impl.Next.type.1bd = struct_value () [concrete]
// CHECK:STDOUT:   %facet_value.62f: %Iterate_where.type.97a = facet_value %MutableAndConstRange, (%Iterate.impl_witness.3b4) [concrete]
// CHECK:STDOUT:   %r.param_patt.3a6: %pattern_type.fdb = value_param_pattern [concrete]
// CHECK:STDOUT:   %r.patt.403: %pattern_type.fdb = wrapper_binding_pattern r, %r.param_patt.3a6 [concrete]
// CHECK:STDOUT:   %Iterate.facet.748: %Iterate.type = facet_value %MutableAndConstRange, (%Iterate.impl_witness.3b4) [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type.24a: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.748) [concrete]
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type.f69: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.748) [concrete]
// CHECK:STDOUT:   %.227: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type.24a, %Iterate.facet.748 [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.specific_fn: <specific function> = specific_function %T.as_type.as.Iterate.impl.NewCursor.8e0, @T.as_type.as.Iterate.impl.NewCursor(%facet_value.b85) [concrete]
// CHECK:STDOUT:   %.343: type = fn_type_with_self_type %Iterate.WithSelf.Next.type.f69, %Iterate.facet.748 [concrete]
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.specific_fn: <specific function> = specific_function %T.as_type.as.Iterate.impl.Next.353, @T.as_type.as.Iterate.impl.Next(%facet_value.b85) [concrete]
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
// CHECK:STDOUT:   %Core.import_ref.e86: @DefaultOptionalStorage.%DefaultOptionalStorage.elem.1 (%DefaultOptionalStorage.elem.ff3) = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [concrete = %field_decl.bce]
// CHECK:STDOUT:   %Core.import_ref.9ce: @DefaultOptionalStorage.%DefaultOptionalStorage.elem.2 (%DefaultOptionalStorage.elem.4a2) = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [concrete = %field_decl.dbe]
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
// CHECK:STDOUT:   %Core.import_ref.b0d = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %field_decl.dbe: @DefaultOptionalStorage.%DefaultOptionalStorage.elem.2 (%DefaultOptionalStorage.elem.4a2) = field_decl has_value, element1, %Core.import_ref.b0d in [concrete] {}
// CHECK:STDOUT:   %Core.import_ref.45d = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %field_decl.bce: @DefaultOptionalStorage.%DefaultOptionalStorage.elem.1 (%DefaultOptionalStorage.elem.ff3) = field_decl value, element0, %Core.import_ref.45d in [concrete] {}
// CHECK:STDOUT:   %MutableAndConstRange.decl: type = class_decl @MutableAndConstRange [concrete = constants.%MutableAndConstRange] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @TestIterate(%R.loc44_17.2: %Iterate_where.type.97a) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Iterate.lookup_impl_witness: <witness> = lookup_impl_witness %R.loc44_17.1, @Iterate [symbolic = %Iterate.lookup_impl_witness (constants.%Iterate.lookup_impl_witness.0da)]
// CHECK:STDOUT:   %Iterate.facet.loc48_41.5: %Iterate.type = facet_value %R.as_type.loc44_83.1, (%Iterate.lookup_impl_witness) [symbolic = %Iterate.facet.loc48_41.5 (constants.%Iterate.facet.394)]
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type: type = fn_type @Iterate.WithSelf.NewCursor, @Iterate.WithSelf(%Iterate.facet.loc48_41.5) [symbolic = %Iterate.WithSelf.NewCursor.type (constants.%Iterate.WithSelf.NewCursor.type.18c)]
// CHECK:STDOUT:   %.loc48_41.26: type = fn_type_with_self_type %Iterate.WithSelf.NewCursor.type, %Iterate.facet.loc48_41.5 [symbolic = %.loc48_41.26 (constants.%.89f4)]
// CHECK:STDOUT:   %impl.elem2.loc48_41.2: @TestIterate.%.loc48_41.26 (%.89f4) = impl_witness_access %Iterate.lookup_impl_witness, element2 [symbolic = %impl.elem2.loc48_41.2 (constants.%impl.elem2.f19)]
// CHECK:STDOUT:   %specific_impl_fn.loc48_41.3: <specific function> = specific_impl_function %impl.elem2.loc48_41.2, @Iterate.WithSelf.NewCursor(%Iterate.facet.loc48_41.5) [symbolic = %specific_impl_fn.loc48_41.3 (constants.%specific_impl_fn.51c)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type: type = fn_type @Iterate.WithSelf.Next, @Iterate.WithSelf(%Iterate.facet.loc48_41.5) [symbolic = %Iterate.WithSelf.Next.type (constants.%Iterate.WithSelf.Next.type.f15)]
// CHECK:STDOUT:   %.loc48_41.27: type = fn_type_with_self_type %Iterate.WithSelf.Next.type, %Iterate.facet.loc48_41.5 [symbolic = %.loc48_41.27 (constants.%.a1e)]
// CHECK:STDOUT:   %impl.elem3.loc48_41.2: @TestIterate.%.loc48_41.27 (%.a1e) = impl_witness_access %Iterate.lookup_impl_witness, element3 [symbolic = %impl.elem3.loc48_41.2 (constants.%impl.elem3.88b)]
// CHECK:STDOUT:   %specific_impl_fn.loc48_41.4: <specific function> = specific_impl_function %impl.elem3.loc48_41.2, @Iterate.WithSelf.Next(%Iterate.facet.loc48_41.5) [symbolic = %specific_impl_fn.loc48_41.4 (constants.%specific_impl_fn.36a)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%r.param: @TestIterate.%R.as_type.loc44_83.1 (%R.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %i.patt: %pattern_type.e09 = value_binding_pattern i [concrete = constants.%i.patt]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %r.ref: @TestIterate.%R.as_type.loc44_83.1 (%R.as_type) = name_ref r, %r
// CHECK:STDOUT:     %impl.elem2.loc48_41.1: @TestIterate.%.loc48_41.26 (%.89f4) = impl_witness_access constants.%Iterate.lookup_impl_witness.0da, element2 [symbolic = %impl.elem2.loc48_41.2 (constants.%impl.elem2.f19)]
// CHECK:STDOUT:     %bound_method.loc48_41.1: <bound method> = bound_method %r.ref, %impl.elem2.loc48_41.1
// CHECK:STDOUT:     %Iterate.facet.loc48_41.1: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.0da) [symbolic = %Iterate.facet.loc48_41.5 (constants.%Iterate.facet.394)]
// CHECK:STDOUT:     %.loc48_41.1: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc48_41.1 [symbolic = %Iterate.facet.loc48_41.5 (constants.%Iterate.facet.394)]
// CHECK:STDOUT:     %Iterate.facet.loc48_41.2: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.0da) [symbolic = %Iterate.facet.loc48_41.5 (constants.%Iterate.facet.394)]
// CHECK:STDOUT:     %.loc48_41.2: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc48_41.2 [symbolic = %Iterate.facet.loc48_41.5 (constants.%Iterate.facet.394)]
// CHECK:STDOUT:     %specific_impl_fn.loc48_41.1: <specific function> = specific_impl_function %impl.elem2.loc48_41.1, @Iterate.WithSelf.NewCursor(constants.%Iterate.facet.394) [symbolic = %specific_impl_fn.loc48_41.3 (constants.%specific_impl_fn.51c)]
// CHECK:STDOUT:     %bound_method.loc48_41.2: <bound method> = bound_method %r.ref, %specific_impl_fn.loc48_41.1
// CHECK:STDOUT:     %var: ref @TestIterate.%as_type (%as_type.71d) = var_storage invalid
// CHECK:STDOUT:     %Iterate.WithSelf.NewCursor.call: init @TestIterate.%as_type (%as_type.71d) to %var = call %bound_method.loc48_41.2(%r.ref)
// CHECK:STDOUT:     assign %var, %Iterate.WithSelf.NewCursor.call
// CHECK:STDOUT:     br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.next:
// CHECK:STDOUT:     %impl.elem3.loc48_41.1: @TestIterate.%.loc48_41.27 (%.a1e) = impl_witness_access constants.%Iterate.lookup_impl_witness.0da, element3 [symbolic = %impl.elem3.loc48_41.2 (constants.%impl.elem3.88b)]
// CHECK:STDOUT:     %bound_method.loc48_41.3: <bound method> = bound_method %r.ref, %impl.elem3.loc48_41.1
// CHECK:STDOUT:     %Iterate.facet.loc48_41.3: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.0da) [symbolic = %Iterate.facet.loc48_41.5 (constants.%Iterate.facet.394)]
// CHECK:STDOUT:     %.loc48_41.3: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc48_41.3 [symbolic = %Iterate.facet.loc48_41.5 (constants.%Iterate.facet.394)]
// CHECK:STDOUT:     %Iterate.facet.loc48_41.4: %Iterate.type = facet_value constants.%R.as_type, (constants.%Iterate.lookup_impl_witness.0da) [symbolic = %Iterate.facet.loc48_41.5 (constants.%Iterate.facet.394)]
// CHECK:STDOUT:     %.loc48_41.4: %Iterate.type = converted constants.%R.as_type, %Iterate.facet.loc48_41.4 [symbolic = %Iterate.facet.loc48_41.5 (constants.%Iterate.facet.394)]
// CHECK:STDOUT:     %.loc48_41.5: %Destroy.type = converted constants.%as_type.71d, constants.%impl.elem1.626 [symbolic = %impl.elem1 (constants.%impl.elem1.626)]
// CHECK:STDOUT:     %.loc48_41.6: %Destroy.type = converted constants.%as_type.71d, constants.%impl.elem1.626 [symbolic = %impl.elem1 (constants.%impl.elem1.626)]
// CHECK:STDOUT:     %specific_impl_fn.loc48_41.2: <specific function> = specific_impl_function %impl.elem3.loc48_41.1, @Iterate.WithSelf.Next(constants.%Iterate.facet.394) [symbolic = %specific_impl_fn.loc48_41.4 (constants.%specific_impl_fn.36a)]
// CHECK:STDOUT:     %bound_method.loc48_41.4: <bound method> = bound_method %r.ref, %specific_impl_fn.loc48_41.2
// CHECK:STDOUT:     %.loc48_41.7: ref %Optional.81f = temporary_storage
// CHECK:STDOUT:     %Iterate.WithSelf.Next.call: init %Optional.81f to %.loc48_41.7 = call %bound_method.loc48_41.4(%r.ref, %var)
// CHECK:STDOUT:     %.loc48_41.8: ref %Optional.81f = temporary %.loc48_41.7, %Iterate.WithSelf.Next.call
// CHECK:STDOUT:     %.loc48_41.9: %Optional.HasValue.type.db4 = specific_constant imports.%Core.import_ref.c82, @Optional(constants.%OptionalStorage.facet.61a) [concrete = constants.%Optional.HasValue.97a]
// CHECK:STDOUT:     %HasValue.ref: %Optional.HasValue.type.db4 = name_ref HasValue, %.loc48_41.9 [concrete = constants.%Optional.HasValue.97a]
// CHECK:STDOUT:     %Optional.HasValue.bound: <bound method> = bound_method %.loc48_41.8, %HasValue.ref
// CHECK:STDOUT:     %Optional.HasValue.specific_fn: <specific function> = specific_function %HasValue.ref, @Optional.HasValue(constants.%OptionalStorage.facet.61a) [concrete = constants.%Optional.HasValue.specific_fn.92c]
// CHECK:STDOUT:     %bound_method.loc48_41.5: <bound method> = bound_method %.loc48_41.8, %Optional.HasValue.specific_fn
// CHECK:STDOUT:     %.loc48_41.10: %Optional.81f = acquire_value %.loc48_41.8
// CHECK:STDOUT:     %Optional.HasValue.call: init bool = call %bound_method.loc48_41.5(%.loc48_41.10)
// CHECK:STDOUT:     %.loc48_41.11: bool = value_of_initializer %Optional.HasValue.call
// CHECK:STDOUT:     %.loc48_41.12: bool = converted %Optional.HasValue.call, %.loc48_41.11
// CHECK:STDOUT:     if %.loc48_41.12 br !for.body else br !for.done
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.body:
// CHECK:STDOUT:     %.loc48_41.13: %Optional.Get.type.0d8 = specific_constant imports.%Core.import_ref.29e, @Optional(constants.%OptionalStorage.facet.61a) [concrete = constants.%Optional.Get.2ea]
// CHECK:STDOUT:     %Get.ref: %Optional.Get.type.0d8 = name_ref Get, %.loc48_41.13 [concrete = constants.%Optional.Get.2ea]
// CHECK:STDOUT:     %Optional.Get.bound: <bound method> = bound_method %.loc48_41.8, %Get.ref
// CHECK:STDOUT:     %Optional.Get.specific_fn: <specific function> = specific_function %Get.ref, @Optional.Get(constants.%OptionalStorage.facet.61a) [concrete = constants.%Optional.Get.specific_fn.132]
// CHECK:STDOUT:     %bound_method.loc48_41.6: <bound method> = bound_method %.loc48_41.8, %Optional.Get.specific_fn
// CHECK:STDOUT:     %.loc48_41.14: ref %ValueType = temporary_storage
// CHECK:STDOUT:     %.loc48_41.15: %Optional.81f = acquire_value %.loc48_41.8
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
// CHECK:STDOUT:     %Destroy.facet.loc48_41.1: %Destroy.type = facet_value constants.%ValueType, (constants.%custom_witness.656, constants.%DestroyImpl.impl_witness.4c3) [concrete = constants.%Destroy.facet.cac]
// CHECK:STDOUT:     %.loc48_41.18: %Destroy.type = converted constants.%ValueType, %Destroy.facet.loc48_41.1 [concrete = constants.%Destroy.facet.cac]
// CHECK:STDOUT:     %Destroy.facet.loc48_41.2: %Destroy.type = facet_value constants.%ValueType, (constants.%custom_witness.656, constants.%DestroyImpl.impl_witness.4c3) [concrete = constants.%Destroy.facet.cac]
// CHECK:STDOUT:     %.loc48_41.19: %Destroy.type = converted constants.%ValueType, %Destroy.facet.loc48_41.2 [concrete = constants.%Destroy.facet.cac]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %SelfDestruct.call.loc48_41.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc48_41.16)
// CHECK:STDOUT:     %Destroy.facet.loc48_41.3: %Destroy.type = facet_value constants.%Optional.81f, (constants.%custom_witness.b80ef6.7, constants.%DestroyImpl.impl_witness.e67) [concrete = constants.%Destroy.facet.434]
// CHECK:STDOUT:     %.loc48_41.20: %Destroy.type = converted constants.%Optional.81f, %Destroy.facet.loc48_41.3 [concrete = constants.%Destroy.facet.434]
// CHECK:STDOUT:     %Destroy.facet.loc48_41.4: %Destroy.type = facet_value constants.%Optional.81f, (constants.%custom_witness.b80ef6.7, constants.%DestroyImpl.impl_witness.e67) [concrete = constants.%Destroy.facet.434]
// CHECK:STDOUT:     %.loc48_41.21: %Destroy.type = converted constants.%Optional.81f, %Destroy.facet.loc48_41.4 [concrete = constants.%Destroy.facet.434]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %SelfDestruct.call.loc48_41.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc48_41.8)
// CHECK:STDOUT:     br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT:   !for.done:
// CHECK:STDOUT:     %Destroy.facet.loc48_41.5: %Destroy.type = facet_value constants.%Optional.81f, (constants.%custom_witness.b80ef6.7, constants.%DestroyImpl.impl_witness.e67) [concrete = constants.%Destroy.facet.434]
// CHECK:STDOUT:     %.loc48_41.22: %Destroy.type = converted constants.%Optional.81f, %Destroy.facet.loc48_41.5 [concrete = constants.%Destroy.facet.434]
// CHECK:STDOUT:     %Destroy.facet.loc48_41.6: %Destroy.type = facet_value constants.%Optional.81f, (constants.%custom_witness.b80ef6.7, constants.%DestroyImpl.impl_witness.e67) [concrete = constants.%Destroy.facet.434]
// CHECK:STDOUT:     %.loc48_41.23: %Destroy.type = converted constants.%Optional.81f, %Destroy.facet.loc48_41.6 [concrete = constants.%Destroy.facet.434]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %SelfDestruct.call.loc48_41.3: init %empty_tuple.type = call %SelfDestruct.specific_fn.3(%.loc48_41.8)
// CHECK:STDOUT:     %.loc48_41.24: %Destroy.type = converted constants.%as_type.71d, constants.%impl.elem1.626 [symbolic = %impl.elem1 (constants.%impl.elem1.626)]
// CHECK:STDOUT:     %.loc48_41.25: %Destroy.type = converted constants.%as_type.71d, constants.%impl.elem1.626 [symbolic = %impl.elem1 (constants.%impl.elem1.626)]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %SelfDestruct.call.loc48_41.4: init %empty_tuple.type = call %SelfDestruct.specific_fn.4(%var)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc48_41.1(%self.param: ref %.8f3) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc48_41.2(%self.param: ref %MaybeUnformed.4b9) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc48_41.3(%self.param: ref bool) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc48_41.4(%self.param: ref %struct_type.value.has_value.640) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc48_41.1: ref bool = struct_access %self.param, element1
// CHECK:STDOUT:   %Destroy.facet.loc48_41.1: %Destroy.type = facet_value bool, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.778) [concrete = constants.%Destroy.facet.80c]
// CHECK:STDOUT:   %.loc48_41.2: %Destroy.type = converted bool, %Destroy.facet.loc48_41.1 [concrete = constants.%Destroy.facet.80c]
// CHECK:STDOUT:   %Destroy.facet.loc48_41.2: %Destroy.type = facet_value bool, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.778) [concrete = constants.%Destroy.facet.80c]
// CHECK:STDOUT:   %.loc48_41.3: %Destroy.type = converted bool, %Destroy.facet.loc48_41.2 [concrete = constants.%Destroy.facet.80c]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc48_41.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc48_41.1)
// CHECK:STDOUT:   %.loc48_41.4: ref %MaybeUnformed.4b9 = struct_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc48_41.3: %Destroy.type = facet_value constants.%MaybeUnformed.4b9, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.e80) [concrete = constants.%Destroy.facet.eca]
// CHECK:STDOUT:   %.loc48_41.5: %Destroy.type = converted constants.%MaybeUnformed.4b9, %Destroy.facet.loc48_41.3 [concrete = constants.%Destroy.facet.eca]
// CHECK:STDOUT:   %Destroy.facet.loc48_41.4: %Destroy.type = facet_value constants.%MaybeUnformed.4b9, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.e80) [concrete = constants.%Destroy.facet.eca]
// CHECK:STDOUT:   %.loc48_41.6: %Destroy.type = converted constants.%MaybeUnformed.4b9, %Destroy.facet.loc48_41.4 [concrete = constants.%Destroy.facet.eca]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc48_41.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc48_41.4)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc48_41.5(%self.param: ref %DefaultOptionalStorage.dff) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %has_value.ref: %DefaultOptionalStorage.elem.6f9 = name_ref has_value, imports.%Core.import_ref.9ce [concrete = imports.%field_decl.dbe]
// CHECK:STDOUT:   %.loc48_41.1: ref bool = class_element_access %self.param, element1
// CHECK:STDOUT:   %Destroy.facet.loc48_41.1: %Destroy.type = facet_value bool, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.778) [concrete = constants.%Destroy.facet.80c]
// CHECK:STDOUT:   %.loc48_41.2: %Destroy.type = converted bool, %Destroy.facet.loc48_41.1 [concrete = constants.%Destroy.facet.80c]
// CHECK:STDOUT:   %Destroy.facet.loc48_41.2: %Destroy.type = facet_value bool, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.778) [concrete = constants.%Destroy.facet.80c]
// CHECK:STDOUT:   %.loc48_41.3: %Destroy.type = converted bool, %Destroy.facet.loc48_41.2 [concrete = constants.%Destroy.facet.80c]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc48_41.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc48_41.1)
// CHECK:STDOUT:   %value.ref: %DefaultOptionalStorage.elem.2ce = name_ref value, imports.%Core.import_ref.e86 [concrete = imports.%field_decl.bce]
// CHECK:STDOUT:   %.loc48_41.4: ref %MaybeUnformed.4b9 = class_element_access %self.param, element0
// CHECK:STDOUT:   %Destroy.facet.loc48_41.3: %Destroy.type = facet_value constants.%MaybeUnformed.4b9, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.e80) [concrete = constants.%Destroy.facet.eca]
// CHECK:STDOUT:   %.loc48_41.5: %Destroy.type = converted constants.%MaybeUnformed.4b9, %Destroy.facet.loc48_41.3 [concrete = constants.%Destroy.facet.eca]
// CHECK:STDOUT:   %Destroy.facet.loc48_41.4: %Destroy.type = facet_value constants.%MaybeUnformed.4b9, (constants.%custom_witness.b80ef6.3, constants.%DestroyImpl.impl_witness.e80) [concrete = constants.%Destroy.facet.eca]
// CHECK:STDOUT:   %.loc48_41.6: %Destroy.type = converted constants.%MaybeUnformed.4b9, %Destroy.facet.loc48_41.4 [concrete = constants.%Destroy.facet.eca]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc48_41.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc48_41.4)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc48_41.6(%self.param: ref %Optional.81f) {
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
// CHECK:STDOUT:   %impl.elem2: %.227 = impl_witness_access constants.%Iterate.impl_witness.3b4, element2 [concrete = constants.%T.as_type.as.Iterate.impl.NewCursor.8e0]
// CHECK:STDOUT:   %bound_method.loc62_42.1: <bound method> = bound_method %mc.ref, %impl.elem2
// CHECK:STDOUT:   %facet_value.loc62_42.1: %CppRange_where.type.d682d6.1 = facet_value constants.%MutableAndConstRange, (constants.%custom_witness.660, constants.%custom_witness.992, constants.%DestroyImpl.impl_witness.52a, constants.%custom_witness.d56, constants.%custom_witness.68a, constants.%custom_witness.edd, constants.%custom_witness.656, constants.%DestroyImpl.impl_witness.4c3, constants.%custom_witness.1c5, constants.%custom_witness.992, constants.%DestroyImpl.impl_witness.52a) [concrete = constants.%facet_value.b85]
// CHECK:STDOUT:   %.loc62_42.1: %CppRange_where.type.d682d6.1 = converted constants.%MutableAndConstRange, %facet_value.loc62_42.1 [concrete = constants.%facet_value.b85]
// CHECK:STDOUT:   %facet_value.loc62_42.2: %CppRange_where.type.d682d6.1 = facet_value constants.%MutableAndConstRange, (constants.%custom_witness.660, constants.%custom_witness.992, constants.%DestroyImpl.impl_witness.52a, constants.%custom_witness.d56, constants.%custom_witness.68a, constants.%custom_witness.edd, constants.%custom_witness.656, constants.%DestroyImpl.impl_witness.4c3, constants.%custom_witness.1c5, constants.%custom_witness.992, constants.%DestroyImpl.impl_witness.52a) [concrete = constants.%facet_value.b85]
// CHECK:STDOUT:   %.loc62_42.2: %CppRange_where.type.d682d6.1 = converted constants.%MutableAndConstRange, %facet_value.loc62_42.2 [concrete = constants.%facet_value.b85]
// CHECK:STDOUT:   %specific_fn.loc62_42.1: <specific function> = specific_function %impl.elem2, @T.as_type.as.Iterate.impl.NewCursor(constants.%facet_value.b85) [concrete = constants.%T.as_type.as.Iterate.impl.NewCursor.specific_fn]
// CHECK:STDOUT:   %bound_method.loc62_42.2: <bound method> = bound_method %mc.ref, %specific_fn.loc62_42.1
// CHECK:STDOUT:   %var: ref %tuple.type.de6 = var_storage invalid
// CHECK:STDOUT:   %.loc62_40.1: %MutableAndConstRange = acquire_value %mc.ref
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.NewCursor.call: init %tuple.type.de6 to %var = call %bound_method.loc62_42.2(%.loc62_40.1)
// CHECK:STDOUT:   assign %var, %T.as_type.as.Iterate.impl.NewCursor.call
// CHECK:STDOUT:   br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT: !for.next:
// CHECK:STDOUT:   %impl.elem3: %.343 = impl_witness_access constants.%Iterate.impl_witness.3b4, element3 [concrete = constants.%T.as_type.as.Iterate.impl.Next.353]
// CHECK:STDOUT:   %bound_method.loc62_42.3: <bound method> = bound_method %mc.ref, %impl.elem3
// CHECK:STDOUT:   %facet_value.loc62_42.3: %CppRange_where.type.d682d6.1 = facet_value constants.%MutableAndConstRange, (constants.%custom_witness.660, constants.%custom_witness.992, constants.%DestroyImpl.impl_witness.52a, constants.%custom_witness.d56, constants.%custom_witness.68a, constants.%custom_witness.edd, constants.%custom_witness.656, constants.%DestroyImpl.impl_witness.4c3, constants.%custom_witness.1c5, constants.%custom_witness.992, constants.%DestroyImpl.impl_witness.52a) [concrete = constants.%facet_value.b85]
// CHECK:STDOUT:   %.loc62_42.3: %CppRange_where.type.d682d6.1 = converted constants.%MutableAndConstRange, %facet_value.loc62_42.3 [concrete = constants.%facet_value.b85]
// CHECK:STDOUT:   %facet_value.loc62_42.4: %CppRange_where.type.d682d6.1 = facet_value constants.%MutableAndConstRange, (constants.%custom_witness.660, constants.%custom_witness.992, constants.%DestroyImpl.impl_witness.52a, constants.%custom_witness.d56, constants.%custom_witness.68a, constants.%custom_witness.edd, constants.%custom_witness.656, constants.%DestroyImpl.impl_witness.4c3, constants.%custom_witness.1c5, constants.%custom_witness.992, constants.%DestroyImpl.impl_witness.52a) [concrete = constants.%facet_value.b85]
// CHECK:STDOUT:   %.loc62_42.4: %CppRange_where.type.d682d6.1 = converted constants.%MutableAndConstRange, %facet_value.loc62_42.4 [concrete = constants.%facet_value.b85]
// CHECK:STDOUT:   %specific_fn.loc62_42.2: <specific function> = specific_function %impl.elem3, @T.as_type.as.Iterate.impl.Next(constants.%facet_value.b85) [concrete = constants.%T.as_type.as.Iterate.impl.Next.specific_fn]
// CHECK:STDOUT:   %bound_method.loc62_42.4: <bound method> = bound_method %mc.ref, %specific_fn.loc62_42.2
// CHECK:STDOUT:   %.loc62_42.5: ref %Optional.81f = temporary_storage
// CHECK:STDOUT:   %.loc62_40.2: %MutableAndConstRange = acquire_value %mc.ref
// CHECK:STDOUT:   %T.as_type.as.Iterate.impl.Next.call: init %Optional.81f to %.loc62_42.5 = call %bound_method.loc62_42.4(%.loc62_40.2, %var)
// CHECK:STDOUT:   %.loc62_42.6: ref %Optional.81f = temporary %.loc62_42.5, %T.as_type.as.Iterate.impl.Next.call
// CHECK:STDOUT:   %.loc62_42.7: %Optional.HasValue.type.db4 = specific_constant imports.%Core.import_ref.c82, @Optional(constants.%OptionalStorage.facet.61a) [concrete = constants.%Optional.HasValue.97a]
// CHECK:STDOUT:   %HasValue.ref: %Optional.HasValue.type.db4 = name_ref HasValue, %.loc62_42.7 [concrete = constants.%Optional.HasValue.97a]
// CHECK:STDOUT:   %Optional.HasValue.bound: <bound method> = bound_method %.loc62_42.6, %HasValue.ref
// CHECK:STDOUT:   %Optional.HasValue.specific_fn: <specific function> = specific_function %HasValue.ref, @Optional.HasValue(constants.%OptionalStorage.facet.61a) [concrete = constants.%Optional.HasValue.specific_fn.92c]
// CHECK:STDOUT:   %bound_method.loc62_42.5: <bound method> = bound_method %.loc62_42.6, %Optional.HasValue.specific_fn
// CHECK:STDOUT:   %.loc62_42.8: %Optional.81f = acquire_value %.loc62_42.6
// CHECK:STDOUT:   %Optional.HasValue.call: init bool = call %bound_method.loc62_42.5(%.loc62_42.8)
// CHECK:STDOUT:   %.loc62_42.9: bool = value_of_initializer %Optional.HasValue.call
// CHECK:STDOUT:   %.loc62_42.10: bool = converted %Optional.HasValue.call, %.loc62_42.9
// CHECK:STDOUT:   if %.loc62_42.10 br !for.body else br !for.done
// CHECK:STDOUT:
// CHECK:STDOUT: !for.body:
// CHECK:STDOUT:   %.loc62_42.11: %Optional.Get.type.0d8 = specific_constant imports.%Core.import_ref.29e, @Optional(constants.%OptionalStorage.facet.61a) [concrete = constants.%Optional.Get.2ea]
// CHECK:STDOUT:   %Get.ref: %Optional.Get.type.0d8 = name_ref Get, %.loc62_42.11 [concrete = constants.%Optional.Get.2ea]
// CHECK:STDOUT:   %Optional.Get.bound: <bound method> = bound_method %.loc62_42.6, %Get.ref
// CHECK:STDOUT:   %Optional.Get.specific_fn: <specific function> = specific_function %Get.ref, @Optional.Get(constants.%OptionalStorage.facet.61a) [concrete = constants.%Optional.Get.specific_fn.132]
// CHECK:STDOUT:   %bound_method.loc62_42.6: <bound method> = bound_method %.loc62_42.6, %Optional.Get.specific_fn
// CHECK:STDOUT:   %.loc62_42.12: ref %ValueType = temporary_storage
// CHECK:STDOUT:   %.loc62_42.13: %Optional.81f = acquire_value %.loc62_42.6
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
// CHECK:STDOUT:   %Destroy.facet.loc62_42.1: %Destroy.type = facet_value constants.%ValueType, (constants.%custom_witness.656, constants.%DestroyImpl.impl_witness.4c3) [concrete = constants.%Destroy.facet.cac]
// CHECK:STDOUT:   %.loc62_42.16: %Destroy.type = converted constants.%ValueType, %Destroy.facet.loc62_42.1 [concrete = constants.%Destroy.facet.cac]
// CHECK:STDOUT:   %Destroy.facet.loc62_42.2: %Destroy.type = facet_value constants.%ValueType, (constants.%custom_witness.656, constants.%DestroyImpl.impl_witness.4c3) [concrete = constants.%Destroy.facet.cac]
// CHECK:STDOUT:   %.loc62_42.17: %Destroy.type = converted constants.%ValueType, %Destroy.facet.loc62_42.2 [concrete = constants.%Destroy.facet.cac]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc62_42.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc62_42.14)
// CHECK:STDOUT:   %Destroy.facet.loc62_42.3: %Destroy.type = facet_value constants.%Optional.81f, (constants.%custom_witness.b80ef6.7, constants.%DestroyImpl.impl_witness.e67) [concrete = constants.%Destroy.facet.434]
// CHECK:STDOUT:   %.loc62_42.18: %Destroy.type = converted constants.%Optional.81f, %Destroy.facet.loc62_42.3 [concrete = constants.%Destroy.facet.434]
// CHECK:STDOUT:   %Destroy.facet.loc62_42.4: %Destroy.type = facet_value constants.%Optional.81f, (constants.%custom_witness.b80ef6.7, constants.%DestroyImpl.impl_witness.e67) [concrete = constants.%Destroy.facet.434]
// CHECK:STDOUT:   %.loc62_42.19: %Destroy.type = converted constants.%Optional.81f, %Destroy.facet.loc62_42.4 [concrete = constants.%Destroy.facet.434]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc62_42.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc62_42.6)
// CHECK:STDOUT:   br !for.next
// CHECK:STDOUT:
// CHECK:STDOUT: !for.done:
// CHECK:STDOUT:   %Destroy.facet.loc62_42.5: %Destroy.type = facet_value constants.%Optional.81f, (constants.%custom_witness.b80ef6.7, constants.%DestroyImpl.impl_witness.e67) [concrete = constants.%Destroy.facet.434]
// CHECK:STDOUT:   %.loc62_42.20: %Destroy.type = converted constants.%Optional.81f, %Destroy.facet.loc62_42.5 [concrete = constants.%Destroy.facet.434]
// CHECK:STDOUT:   %Destroy.facet.loc62_42.6: %Destroy.type = facet_value constants.%Optional.81f, (constants.%custom_witness.b80ef6.7, constants.%DestroyImpl.impl_witness.e67) [concrete = constants.%Destroy.facet.434]
// CHECK:STDOUT:   %.loc62_42.21: %Destroy.type = converted constants.%Optional.81f, %Destroy.facet.loc62_42.6 [concrete = constants.%Destroy.facet.434]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc62_42.3: init %empty_tuple.type = call %SelfDestruct.specific_fn.3(%.loc62_42.6)
// CHECK:STDOUT:   %Destroy.facet.loc62_42.7: %Destroy.type = facet_value constants.%tuple.type.de6, (constants.%custom_witness.b80ef6.8, constants.%DestroyImpl.impl_witness.2c4) [concrete = constants.%Destroy.facet.a88]
// CHECK:STDOUT:   %.loc62_42.22: %Destroy.type = converted constants.%tuple.type.de6, %Destroy.facet.loc62_42.7 [concrete = constants.%Destroy.facet.a88]
// CHECK:STDOUT:   %Destroy.facet.loc62_42.8: %Destroy.type = facet_value constants.%tuple.type.de6, (constants.%custom_witness.b80ef6.8, constants.%DestroyImpl.impl_witness.2c4) [concrete = constants.%Destroy.facet.a88]
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
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.2ae
// CHECK:STDOUT:   %r.param_patt.loc44_81.2 => constants.%r.param_patt.91c
// CHECK:STDOUT:   %r.patt.loc44_81.2 => constants.%r.patt.03a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestIterate(constants.%facet_value.62f) {
// CHECK:STDOUT:   %R.patt.loc44_17.2 => constants.%R.patt
// CHECK:STDOUT:   %R.loc44_17.1 => constants.%facet_value.62f
// CHECK:STDOUT:   %R.as_type.loc44_83.1 => constants.%MutableAndConstRange
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.fdb
// CHECK:STDOUT:   %r.param_patt.loc44_81.2 => constants.%r.param_patt.3a6
// CHECK:STDOUT:   %r.patt.loc44_81.2 => constants.%r.patt.403
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc44 => constants.%complete_type.357
// CHECK:STDOUT:   %Iterate.lookup_impl_witness => constants.%Iterate.impl_witness.3b4
// CHECK:STDOUT:   %Iterate.facet.loc48_41.5 => constants.%Iterate.facet.748
// CHECK:STDOUT:   %Iterate.WithSelf.NewCursor.type => constants.%Iterate.WithSelf.NewCursor.type.24a
// CHECK:STDOUT:   %.loc48_41.26 => constants.%.227
// CHECK:STDOUT:   %impl.elem2.loc48_41.2 => constants.%T.as_type.as.Iterate.impl.NewCursor.8e0
// CHECK:STDOUT:   %specific_impl_fn.loc48_41.3 => constants.%T.as_type.as.Iterate.impl.NewCursor.specific_fn
// CHECK:STDOUT:   %impl.elem1 => constants.%Destroy.facet.a88
// CHECK:STDOUT:   %as_type => constants.%tuple.type.de6
// CHECK:STDOUT:   %require_complete.1 => constants.%complete_type.95d
// CHECK:STDOUT:   %Iterate.WithSelf.Next.type => constants.%Iterate.WithSelf.Next.type.f69
// CHECK:STDOUT:   %.loc48_41.27 => constants.%.343
// CHECK:STDOUT:   %impl.elem3.loc48_41.2 => constants.%T.as_type.as.Iterate.impl.Next.353
// CHECK:STDOUT:   %specific_impl_fn.loc48_41.4 => constants.%T.as_type.as.Iterate.impl.Next.specific_fn
// CHECK:STDOUT:   %SelfDestruct.specific_fn.6 => constants.%SelfDestruct.specific_fn.28f
// CHECK:STDOUT: }
// CHECK:STDOUT:
