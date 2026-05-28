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

fn Test[R:! Core.CppRangeForIterate where .Iterator impls Core.Destroy and .Sentinel impls Core.Destroy](var r: R) {
  //@dump-sem-ir-begin
  r.Begin();
  r.End();
  //@dump-sem-ir-end
}

fn TestDriver(var m: Cpp.MutableRange, var c: Cpp.ConstRange) {
  Test(m);
  Test(c);
}

// --- methods_return_different_types.carbon

library "[[@TEST_NAME]]";

import Cpp inline '''c++
class MutableRange {
  struct ValueType {};

  struct Iterator {
    auto operator*() const -> ValueType&;
    auto operator++() -> Iterator&;
  };

  struct Sentinel {
    friend auto operator==(Sentinel, Iterator) -> bool;
    friend auto operator!=(Sentinel, Iterator) -> bool;
  };
public:
  auto begin() -> Iterator;
  auto end() -> Sentinel;
};

class ConstRange {
  struct ValueType {};

  struct Iterator {
    auto operator*() const -> ValueType;
    auto operator++() -> Iterator&;
  };

  struct Sentinel {
    friend auto operator==(Sentinel, Iterator) -> bool;
    friend auto operator!=(Sentinel, Iterator) -> bool;
  };
public:
  auto begin() const -> Iterator;
  auto end() const -> Sentinel;
};
''';

fn Test[R:! Core.CppRangeForIterate where .Iterator impls Core.Destroy and .Sentinel impls Core.Destroy](var r: R) {
  //@dump-sem-ir-begin
  r.Begin();
  r.End();
  //@dump-sem-ir-end
}

fn TestDriver(var m: Cpp.MutableRange, var c: Cpp.ConstRange) {
  Test(m);
  Test(c);
}

// --- adl_return_same_types.carbon

library "[[@TEST_NAME]]";

import Cpp inline '''c++
namespace N {
class MutableRange {
  struct Iterator {
    auto operator*() const -> int&;
    auto operator++() -> Iterator&;
    friend auto operator==(Iterator, Iterator) -> bool;
    friend auto operator!=(Iterator, Iterator) -> bool;
  };
public:
  friend auto begin(MutableRange&) -> Iterator;
  friend auto end(MutableRange&) -> Iterator;
};

class ConstRange {
  struct Iterator {
    auto operator*() const -> const int&;
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

fn Test[R:! Core.CppRangeForIterate where .Iterator impls Core.Destroy and .Sentinel impls Core.Destroy](var r: R) {
  //@dump-sem-ir-begin
  r.Begin();
  r.End();
  //@dump-sem-ir-end
}

fn TestDriver(var m: Cpp.N.MutableRange, var c: Cpp.N.ConstRange) {
  Test(m);
  Test(c);
}

// --- adl_returns_different_types.carbon

library "[[@TEST_NAME]]";

import Cpp inline '''c++
namespace N {
class MutableRange {
  struct ValueType {};

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

class ConstRange {
  struct ValueType {};

  struct Iterator {
    auto operator*() const -> ValueType;
    auto operator++() -> Iterator&;
  };

  struct Sentinel {
    friend auto operator==(Sentinel, Iterator) -> bool;
    friend auto operator!=(Sentinel, Iterator) -> bool;
  };
public:
  friend auto begin(const ConstRange&) -> Iterator;
  friend auto end(const ConstRange&) -> Sentinel;
};
} // namespace N
''';

fn Test[R:! Core.CppRangeForIterate where .Iterator impls Core.Destroy and .Sentinel impls Core.Destroy](var r: R) {
  //@dump-sem-ir-begin
  r.Begin();
  r.End();
  //@dump-sem-ir-end
}

fn TestDriver(var m: Cpp.N.MutableRange, var c: Cpp.N.ConstRange) {
  Test(m);
  Test(c);
}

// --- adl_overloads.carbon

import Cpp inline '''c++
namespace N {
class MutableAndConstRange {
  struct ValueType {};

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

fn Test[R:! Core.CppRangeForIterate where .Iterator impls Core.Destroy and .Sentinel impls Core.Destroy](var r: R) {
  //@dump-sem-ir-begin
  r.Begin();
  r.End();
  //@dump-sem-ir-end
}

fn TestDriver(var mc: Cpp.N.MutableAndConstRange) {
  Test(mc);
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

// CHECK:STDERR: fail_never_valid.carbon:[[@LINE+4]]:38: warning: binding `r` unused [UnusedBinding]
// CHECK:STDERR: fn Test[R:! Core.CppRangeForIterate](r: R) {
// CHECK:STDERR:                                      ^~~~
// CHECK:STDERR:
fn Test[R:! Core.CppRangeForIterate](r: R) {
}

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
  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE+7]]:3: error: cannot convert type `Cpp.NoBeginEnd` into type implementing `Core.CppRangeForIterate` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   Test(no_begin_end);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE-20]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn Test[R:! Core.CppRangeForIterate](r: R) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Test(no_begin_end);

  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE+7]]:3: error: cannot convert type `Cpp.NoBeginMethod` into type implementing `Core.CppRangeForIterate` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   Test(no_begin_method);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE-29]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn Test[R:! Core.CppRangeForIterate](r: R) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Test(no_begin_method);

  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE+7]]:3: error: cannot convert type `Cpp.NoEndMethod` into type implementing `Core.CppRangeForIterate` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   Test(no_end_method);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE-38]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn Test[R:! Core.CppRangeForIterate](r: R) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Test(no_end_method);

  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE+7]]:3: error: cannot convert type `Cpp.NoBeginADL` into type implementing `Core.CppRangeForIterate` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   Test(no_begin_adl);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE-47]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn Test[R:! Core.CppRangeForIterate](r: R) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Test(no_begin_adl);

  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE+7]]:3: error: cannot convert type `Cpp.NoEndADL` into type implementing `Core.CppRangeForIterate` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   Test(no_end_adl);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE-56]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn Test[R:! Core.CppRangeForIterate](r: R) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Test(no_end_adl);

  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE+7]]:3: error: cannot convert type `Cpp.BeginMethodEndADL` into type implementing `Core.CppRangeForIterate` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   Test(begin_method_end_adl);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE-65]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn Test[R:! Core.CppRangeForIterate](r: R) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Test(begin_method_end_adl);

  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE+7]]:3: error: cannot convert type `Cpp.BeginADLEndMethod` into type implementing `Core.CppRangeForIterate` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   Test(begin_adl_end_method);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE-74]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn Test[R:! Core.CppRangeForIterate](r: R) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Test(begin_adl_end_method);

  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE+7]]:3: error: cannot convert type `Cpp.BeginFieldEndMethod` into type implementing `Core.CppRangeForIterate` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   Test(begin_field_end_method);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE-83]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn Test[R:! Core.CppRangeForIterate](r: R) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Test(begin_field_end_method);

  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE+7]]:3: error: cannot convert type `Cpp.BeginMethodEndField` into type implementing `Core.CppRangeForIterate` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   Test(begin_method_end_field);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE-92]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn Test[R:! Core.CppRangeForIterate](r: R) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Test(begin_method_end_field);

  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE+10]]:28: error: attempt to use a deleted function [CppInteropParseError]
  // CHECK:STDERR:   176 |   Test(begin_method_deleted);
  // CHECK:STDERR:       |                            ^
  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE-125]]:8: note: 'begin' has been explicitly marked deleted here [CppInteropParseNote]
  // CHECK:STDERR:    44 |   auto begin() = delete;
  // CHECK:STDERR:       |        ^
  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE-104]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn Test[R:! Core.CppRangeForIterate](r: R) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Test(begin_method_deleted);

  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE+10]]:26: error: attempt to use a deleted function [CppInteropParseError]
  // CHECK:STDERR:   188 |   Test(end_method_deleted);
  // CHECK:STDERR:       |                          ^
  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE-131]]:8: note: 'end' has been explicitly marked deleted here [CppInteropParseNote]
  // CHECK:STDERR:    50 |   auto end() = delete;
  // CHECK:STDERR:       |        ^
  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE-116]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn Test[R:! Core.CppRangeForIterate](r: R) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Test(end_method_deleted);

  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE+10]]:30: error: call to deleted function 'begin' [CppInteropParseError]
  // CHECK:STDERR:   200 |   Test(begin_function_deleted);
  // CHECK:STDERR:       |                              ^
  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE-139]]:15: note: candidate function has been explicitly deleted [CppInteropParseNote]
  // CHECK:STDERR:    54 |   friend auto begin(const BeginFunctionDeleted&) = delete;
  // CHECK:STDERR:       |               ^
  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE-128]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn Test[R:! Core.CppRangeForIterate](r: R) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Test(begin_function_deleted);

  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE+10]]:28: error: call to deleted function 'end' [CppInteropParseError]
  // CHECK:STDERR:   212 |   Test(end_function_deleted);
  // CHECK:STDERR:       |                            ^
  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE-145]]:15: note: candidate function has been explicitly deleted [CppInteropParseNote]
  // CHECK:STDERR:    60 |   friend auto end(const EndFunctionDeleted&) = delete;
  // CHECK:STDERR:       |               ^
  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE-140]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn Test[R:! Core.CppRangeForIterate](r: R) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Test(end_function_deleted);
}

// CHECK:STDOUT: --- methods_return_same_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %CppRangeForIterate.type: type = facet_type <@CppRangeForIterate> [concrete]
// CHECK:STDOUT:   %Self.668: %CppRangeForIterate.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.End.type.45d9b7.1: type = fn_type @CppRangeForIterate.WithSelf.End, @CppRangeForIterate.WithSelf(%Self.668) [symbolic]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.End.e1b918.1: %CppRangeForIterate.WithSelf.End.type.45d9b7.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.Begin.type.e09a97.1: type = fn_type @CppRangeForIterate.WithSelf.Begin, @CppRangeForIterate.WithSelf(%Self.668) [symbolic]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.Begin.4c9a66.1: %CppRangeForIterate.WithSelf.Begin.type.e09a97.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %.Self.fda: %CppRangeForIterate.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %CppRangeForIterate.assoc_type: type = assoc_entity_type @CppRangeForIterate [concrete]
// CHECK:STDOUT:   %CppRangeForIterate.lookup_impl_witness.ae8: <witness> = lookup_impl_witness %.Self.fda, @CppRangeForIterate [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.354: type = impl_witness_access %CppRangeForIterate.lookup_impl_witness.ae8, element0 [symbolic_self]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %impl.elem1.fdc: type = impl_witness_access %CppRangeForIterate.lookup_impl_witness.ae8, element1 [symbolic_self]
// CHECK:STDOUT:   %CppRangeForIterate_where.type: type = facet_type <@CppRangeForIterate where %impl.elem0.354 impls @Destroy and %impl.elem1.fdc impls @Destroy> [concrete]
// CHECK:STDOUT:   %R: %CppRangeForIterate_where.type = symbolic_binding R, 0 [symbolic]
// CHECK:STDOUT:   %R.as_type: type = facet_access_type %R [symbolic]
// CHECK:STDOUT:   %pattern_type.ff859f.2: type = pattern_type %R.as_type [symbolic]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %assoc2: %CppRangeForIterate.assoc_type = assoc_entity element2, imports.%Core.import_ref.dad [concrete]
// CHECK:STDOUT:   %CppRangeForIterate.lookup_impl_witness.366fea.2: <witness> = lookup_impl_witness %R, @CppRangeForIterate [symbolic]
// CHECK:STDOUT:   %CppRangeForIterate.facet.527: %CppRangeForIterate.type = facet_value %R.as_type, (%CppRangeForIterate.lookup_impl_witness.366fea.2) [symbolic]
// CHECK:STDOUT:   %impl.elem0.b36147.2: type = impl_witness_access %CppRangeForIterate.lookup_impl_witness.366fea.2, element0 [symbolic]
// CHECK:STDOUT:   %impl.elem1.3ad37d.2: type = impl_witness_access %CppRangeForIterate.lookup_impl_witness.366fea.2, element1 [symbolic]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.Begin.type.aa6: type = fn_type @CppRangeForIterate.WithSelf.Begin, @CppRangeForIterate.WithSelf(%CppRangeForIterate.facet.527) [symbolic]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.End.type.ebe: type = fn_type @CppRangeForIterate.WithSelf.End, @CppRangeForIterate.WithSelf(%CppRangeForIterate.facet.527) [symbolic]
// CHECK:STDOUT:   %.227: type = fn_type_with_self_type %CppRangeForIterate.WithSelf.Begin.type.aa6, %CppRangeForIterate.facet.527 [symbolic]
// CHECK:STDOUT:   %impl.elem2: %.227 = impl_witness_access %CppRangeForIterate.lookup_impl_witness.366fea.2, element2 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.4a0: <specific function> = specific_impl_function %impl.elem2, @CppRangeForIterate.WithSelf.Begin(%CppRangeForIterate.facet.527) [symbolic]
// CHECK:STDOUT:   %assoc3: %CppRangeForIterate.assoc_type = assoc_entity element3, imports.%Core.import_ref.f0a [concrete]
// CHECK:STDOUT:   %.cab: type = fn_type_with_self_type %CppRangeForIterate.WithSelf.End.type.ebe, %CppRangeForIterate.facet.527 [symbolic]
// CHECK:STDOUT:   %impl.elem3: %.cab = impl_witness_access %CppRangeForIterate.lookup_impl_witness.366fea.2, element3 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.cc4: <specific function> = specific_impl_function %impl.elem3, @CppRangeForIterate.WithSelf.End(%CppRangeForIterate.facet.527) [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.4ab: <witness> = lookup_impl_witness %impl.elem1.3ad37d.2, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.553: %Destroy.type = facet_value %impl.elem1.3ad37d.2, (%Destroy.lookup_impl_witness.4ab) [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.b61: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.553) [symbolic]
// CHECK:STDOUT:   %.b17: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.b61, %Destroy.facet.553 [symbolic]
// CHECK:STDOUT:   %impl.elem0.449: %.b17 = impl_witness_access %Destroy.lookup_impl_witness.4ab, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.03f: <specific function> = specific_impl_function %impl.elem0.449, @Destroy.WithSelf.Op(%Destroy.facet.553) [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.e97: <witness> = lookup_impl_witness %impl.elem0.b36147.2, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.503: %Destroy.type = facet_value %impl.elem0.b36147.2, (%Destroy.lookup_impl_witness.e97) [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.4c8: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.503) [symbolic]
// CHECK:STDOUT:   %.d5a: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.4c8, %Destroy.facet.503 [symbolic]
// CHECK:STDOUT:   %impl.elem0.d1d: %.d5a = impl_witness_access %Destroy.lookup_impl_witness.e97, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.a5c: <specific function> = specific_impl_function %impl.elem0.d1d, @Destroy.WithSelf.Op(%Destroy.facet.503) [symbolic]
// CHECK:STDOUT:   %MutableRange: type = class_type @MutableRange [concrete]
// CHECK:STDOUT:   %pattern_type.607: type = pattern_type %MutableRange [concrete]
// CHECK:STDOUT:   %ConstRange: type = class_type @ConstRange [concrete]
// CHECK:STDOUT:   %pattern_type.03c: type = pattern_type %ConstRange [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %Iterator.a7566c.1: type = class_type @Iterator.1 [concrete]
// CHECK:STDOUT:   %MutableRange.Begin.type: type = fn_type @MutableRange.Begin [concrete]
// CHECK:STDOUT:   %MutableRange.Begin: %MutableRange.Begin.type = struct_value () [concrete]
// CHECK:STDOUT:   %MutableRange.End.type: type = fn_type @MutableRange.End [concrete]
// CHECK:STDOUT:   %MutableRange.End: %MutableRange.End.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.7e3fb2.1: <witness> = custom_witness (%Iterator.a7566c.1, %Iterator.a7566c.1, %MutableRange.Begin, %MutableRange.End), @CppRangeForIterate [concrete]
// CHECK:STDOUT:   %CppRangeForIterate.facet.d24: %CppRangeForIterate.type = facet_value %MutableRange, (%custom_witness.7e3fb2.1) [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.f11: type = fn_type @Iterator.Op.1 [concrete]
// CHECK:STDOUT:   %Iterator.Op.78e: %Iterator.Op.type.f11 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b91: <witness> = custom_witness (%Iterator.Op.78e), @Destroy [concrete]
// CHECK:STDOUT:   %facet_value.665: %CppRangeForIterate_where.type = facet_value %MutableRange, (%custom_witness.7e3fb2.1, %custom_witness.b91) [concrete]
// CHECK:STDOUT:   %Iterator.a7566c.2: type = class_type @Iterator.2 [concrete]
// CHECK:STDOUT:   %ConstRange.Begin.type: type = fn_type @ConstRange.Begin [concrete]
// CHECK:STDOUT:   %ConstRange.Begin: %ConstRange.Begin.type = struct_value () [concrete]
// CHECK:STDOUT:   %ConstRange.End.type: type = fn_type @ConstRange.End [concrete]
// CHECK:STDOUT:   %ConstRange.End: %ConstRange.End.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.7e3fb2.2: <witness> = custom_witness (%Iterator.a7566c.2, %Iterator.a7566c.2, %ConstRange.Begin, %ConstRange.End), @CppRangeForIterate [concrete]
// CHECK:STDOUT:   %CppRangeForIterate.facet.9f1: %CppRangeForIterate.type = facet_value %ConstRange, (%custom_witness.7e3fb2.2) [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.798: type = fn_type @Iterator.Op.2 [concrete]
// CHECK:STDOUT:   %Iterator.Op.250: %Iterator.Op.type.798 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.018: <witness> = custom_witness (%Iterator.Op.250), @Destroy [concrete]
// CHECK:STDOUT:   %facet_value.57a: %CppRangeForIterate_where.type = facet_value %ConstRange, (%custom_witness.7e3fb2.2, %custom_witness.018) [concrete]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.Begin.type.a13: type = fn_type @CppRangeForIterate.WithSelf.Begin, @CppRangeForIterate.WithSelf(%CppRangeForIterate.facet.d24) [concrete]
// CHECK:STDOUT:   %.a34: type = fn_type_with_self_type %CppRangeForIterate.WithSelf.Begin.type.a13, %CppRangeForIterate.facet.d24 [concrete]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.End.type.e9f: type = fn_type @CppRangeForIterate.WithSelf.End, @CppRangeForIterate.WithSelf(%CppRangeForIterate.facet.d24) [concrete]
// CHECK:STDOUT:   %.2ee: type = fn_type_with_self_type %CppRangeForIterate.WithSelf.End.type.e9f, %CppRangeForIterate.facet.d24 [concrete]
// CHECK:STDOUT:   %Destroy.facet.fb8: %Destroy.type = facet_value %Iterator.a7566c.1, (%custom_witness.b91) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.223: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.fb8) [concrete]
// CHECK:STDOUT:   %.438: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.223, %Destroy.facet.fb8 [concrete]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.Begin.type.70e: type = fn_type @CppRangeForIterate.WithSelf.Begin, @CppRangeForIterate.WithSelf(%CppRangeForIterate.facet.9f1) [concrete]
// CHECK:STDOUT:   %.a48: type = fn_type_with_self_type %CppRangeForIterate.WithSelf.Begin.type.70e, %CppRangeForIterate.facet.9f1 [concrete]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.End.type.870: type = fn_type @CppRangeForIterate.WithSelf.End, @CppRangeForIterate.WithSelf(%CppRangeForIterate.facet.9f1) [concrete]
// CHECK:STDOUT:   %.6d4: type = fn_type_with_self_type %CppRangeForIterate.WithSelf.End.type.870, %CppRangeForIterate.facet.9f1 [concrete]
// CHECK:STDOUT:   %Destroy.facet.10b: %Destroy.type = facet_value %Iterator.a7566c.2, (%custom_witness.018) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.7f7: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.10b) [concrete]
// CHECK:STDOUT:   %.cac: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.7f7, %Destroy.facet.10b [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.4af: %CppRangeForIterate.assoc_type = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [concrete = constants.%assoc2]
// CHECK:STDOUT:   %Core.import_ref.83c: %CppRangeForIterate.assoc_type = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [concrete = constants.%assoc3]
// CHECK:STDOUT:   %Core.import_ref.dad: @CppRangeForIterate.WithSelf.%CppRangeForIterate.WithSelf.Begin.type (%CppRangeForIterate.WithSelf.Begin.type.e09a97.1) = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @CppRangeForIterate.WithSelf.%CppRangeForIterate.WithSelf.Begin (constants.%CppRangeForIterate.WithSelf.Begin.4c9a66.1)]
// CHECK:STDOUT:   %Core.import_ref.f0a: @CppRangeForIterate.WithSelf.%CppRangeForIterate.WithSelf.End.type (%CppRangeForIterate.WithSelf.End.type.45d9b7.1) = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @CppRangeForIterate.WithSelf.%CppRangeForIterate.WithSelf.End (constants.%CppRangeForIterate.WithSelf.End.e1b918.1)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Test(%R.loc30_10.2: %CppRangeForIterate_where.type) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %CppRangeForIterate.lookup_impl_witness: <witness> = lookup_impl_witness %R.loc30_10.1, @CppRangeForIterate [symbolic = %CppRangeForIterate.lookup_impl_witness (constants.%CppRangeForIterate.lookup_impl_witness.366fea.2)]
// CHECK:STDOUT:   %CppRangeForIterate.facet: %CppRangeForIterate.type = facet_value %R.as_type.loc30_113.1, (%CppRangeForIterate.lookup_impl_witness) [symbolic = %CppRangeForIterate.facet (constants.%CppRangeForIterate.facet.527)]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.Begin.type: type = fn_type @CppRangeForIterate.WithSelf.Begin, @CppRangeForIterate.WithSelf(%CppRangeForIterate.facet) [symbolic = %CppRangeForIterate.WithSelf.Begin.type (constants.%CppRangeForIterate.WithSelf.Begin.type.aa6)]
// CHECK:STDOUT:   %.loc32_4: type = fn_type_with_self_type %CppRangeForIterate.WithSelf.Begin.type, %CppRangeForIterate.facet [symbolic = %.loc32_4 (constants.%.227)]
// CHECK:STDOUT:   %impl.elem2.loc32_4.2: @Test.%.loc32_4 (%.227) = impl_witness_access %CppRangeForIterate.lookup_impl_witness, element2 [symbolic = %impl.elem2.loc32_4.2 (constants.%impl.elem2)]
// CHECK:STDOUT:   %specific_impl_fn.loc32_4.2: <specific function> = specific_impl_function %impl.elem2.loc32_4.2, @CppRangeForIterate.WithSelf.Begin(%CppRangeForIterate.facet) [symbolic = %specific_impl_fn.loc32_4.2 (constants.%specific_impl_fn.4a0)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.End.type: type = fn_type @CppRangeForIterate.WithSelf.End, @CppRangeForIterate.WithSelf(%CppRangeForIterate.facet) [symbolic = %CppRangeForIterate.WithSelf.End.type (constants.%CppRangeForIterate.WithSelf.End.type.ebe)]
// CHECK:STDOUT:   %.loc33_4: type = fn_type_with_self_type %CppRangeForIterate.WithSelf.End.type, %CppRangeForIterate.facet [symbolic = %.loc33_4 (constants.%.cab)]
// CHECK:STDOUT:   %impl.elem3.loc33_4.2: @Test.%.loc33_4 (%.cab) = impl_witness_access %CppRangeForIterate.lookup_impl_witness, element3 [symbolic = %impl.elem3.loc33_4.2 (constants.%impl.elem3)]
// CHECK:STDOUT:   %specific_impl_fn.loc33_4.2: <specific function> = specific_impl_function %impl.elem3.loc33_4.2, @CppRangeForIterate.WithSelf.End(%CppRangeForIterate.facet) [symbolic = %specific_impl_fn.loc33_4.2 (constants.%specific_impl_fn.cc4)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.loc33: <witness> = lookup_impl_witness %impl.elem1.1, @Destroy [symbolic = %Destroy.lookup_impl_witness.loc33 (constants.%Destroy.lookup_impl_witness.4ab)]
// CHECK:STDOUT:   %Destroy.facet.loc33: %Destroy.type = facet_value %impl.elem1.1, (%Destroy.lookup_impl_witness.loc33) [symbolic = %Destroy.facet.loc33 (constants.%Destroy.facet.553)]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.loc33: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.loc33) [symbolic = %Destroy.WithSelf.Op.type.loc33 (constants.%Destroy.WithSelf.Op.type.b61)]
// CHECK:STDOUT:   %.loc33_9.3: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.loc33, %Destroy.facet.loc33 [symbolic = %.loc33_9.3 (constants.%.b17)]
// CHECK:STDOUT:   %impl.elem0.loc33_9.2: @Test.%.loc33_9.3 (%.b17) = impl_witness_access %Destroy.lookup_impl_witness.loc33, element0 [symbolic = %impl.elem0.loc33_9.2 (constants.%impl.elem0.449)]
// CHECK:STDOUT:   %specific_impl_fn.loc33_9.2: <specific function> = specific_impl_function %impl.elem0.loc33_9.2, @Destroy.WithSelf.Op(%Destroy.facet.loc33) [symbolic = %specific_impl_fn.loc33_9.2 (constants.%specific_impl_fn.03f)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.loc32: <witness> = lookup_impl_witness %impl.elem0.1, @Destroy [symbolic = %Destroy.lookup_impl_witness.loc32 (constants.%Destroy.lookup_impl_witness.e97)]
// CHECK:STDOUT:   %Destroy.facet.loc32: %Destroy.type = facet_value %impl.elem0.1, (%Destroy.lookup_impl_witness.loc32) [symbolic = %Destroy.facet.loc32 (constants.%Destroy.facet.503)]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.loc32: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.loc32) [symbolic = %Destroy.WithSelf.Op.type.loc32 (constants.%Destroy.WithSelf.Op.type.4c8)]
// CHECK:STDOUT:   %.loc32_11.3: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.loc32, %Destroy.facet.loc32 [symbolic = %.loc32_11.3 (constants.%.d5a)]
// CHECK:STDOUT:   %impl.elem0.loc32_11.2: @Test.%.loc32_11.3 (%.d5a) = impl_witness_access %Destroy.lookup_impl_witness.loc32, element0 [symbolic = %impl.elem0.loc32_11.2 (constants.%impl.elem0.d1d)]
// CHECK:STDOUT:   %specific_impl_fn.loc32_11.2: <specific function> = specific_impl_function %impl.elem0.loc32_11.2, @Destroy.WithSelf.Op(%Destroy.facet.loc32) [symbolic = %specific_impl_fn.loc32_11.2 (constants.%specific_impl_fn.a5c)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%r.param: ref @Test.%R.as_type.loc30_113.1 (%R.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %r.ref.loc32: ref @Test.%R.as_type.loc30_113.1 (%R.as_type) = name_ref r, %r
// CHECK:STDOUT:     %Begin.ref: %CppRangeForIterate.assoc_type = name_ref Begin, imports.%Core.import_ref.4af [concrete = constants.%assoc2]
// CHECK:STDOUT:     %impl.elem2.loc32_4.1: @Test.%.loc32_4 (%.227) = impl_witness_access constants.%CppRangeForIterate.lookup_impl_witness.366fea.2, element2 [symbolic = %impl.elem2.loc32_4.2 (constants.%impl.elem2)]
// CHECK:STDOUT:     %bound_method.loc32_4: <bound method> = bound_method %r.ref.loc32, %impl.elem2.loc32_4.1
// CHECK:STDOUT:     %specific_impl_fn.loc32_4.1: <specific function> = specific_impl_function %impl.elem2.loc32_4.1, @CppRangeForIterate.WithSelf.Begin(constants.%CppRangeForIterate.facet.527) [symbolic = %specific_impl_fn.loc32_4.2 (constants.%specific_impl_fn.4a0)]
// CHECK:STDOUT:     %bound_method.loc32_11.1: <bound method> = bound_method %r.ref.loc32, %specific_impl_fn.loc32_4.1
// CHECK:STDOUT:     %.loc32_11.1: ref @Test.%impl.elem0.1 (%impl.elem0.b36147.2) = temporary_storage
// CHECK:STDOUT:     %CppRangeForIterate.WithSelf.Begin.call: init @Test.%impl.elem0.1 (%impl.elem0.b36147.2) to %.loc32_11.1 = call %bound_method.loc32_11.1(%r.ref.loc32)
// CHECK:STDOUT:     %.loc32_11.2: ref @Test.%impl.elem0.1 (%impl.elem0.b36147.2) = temporary %.loc32_11.1, %CppRangeForIterate.WithSelf.Begin.call
// CHECK:STDOUT:     %r.ref.loc33: ref @Test.%R.as_type.loc30_113.1 (%R.as_type) = name_ref r, %r
// CHECK:STDOUT:     %End.ref: %CppRangeForIterate.assoc_type = name_ref End, imports.%Core.import_ref.83c [concrete = constants.%assoc3]
// CHECK:STDOUT:     %impl.elem3.loc33_4.1: @Test.%.loc33_4 (%.cab) = impl_witness_access constants.%CppRangeForIterate.lookup_impl_witness.366fea.2, element3 [symbolic = %impl.elem3.loc33_4.2 (constants.%impl.elem3)]
// CHECK:STDOUT:     %bound_method.loc33_4: <bound method> = bound_method %r.ref.loc33, %impl.elem3.loc33_4.1
// CHECK:STDOUT:     %specific_impl_fn.loc33_4.1: <specific function> = specific_impl_function %impl.elem3.loc33_4.1, @CppRangeForIterate.WithSelf.End(constants.%CppRangeForIterate.facet.527) [symbolic = %specific_impl_fn.loc33_4.2 (constants.%specific_impl_fn.cc4)]
// CHECK:STDOUT:     %bound_method.loc33_9.1: <bound method> = bound_method %r.ref.loc33, %specific_impl_fn.loc33_4.1
// CHECK:STDOUT:     %.loc33_9.1: ref @Test.%impl.elem1.1 (%impl.elem1.3ad37d.2) = temporary_storage
// CHECK:STDOUT:     %CppRangeForIterate.WithSelf.End.call: init @Test.%impl.elem1.1 (%impl.elem1.3ad37d.2) to %.loc33_9.1 = call %bound_method.loc33_9.1(%r.ref.loc33)
// CHECK:STDOUT:     %.loc33_9.2: ref @Test.%impl.elem1.1 (%impl.elem1.3ad37d.2) = temporary %.loc33_9.1, %CppRangeForIterate.WithSelf.End.call
// CHECK:STDOUT:     %impl.elem0.loc33_9.1: @Test.%.loc33_9.3 (%.b17) = impl_witness_access constants.%Destroy.lookup_impl_witness.4ab, element0 [symbolic = %impl.elem0.loc33_9.2 (constants.%impl.elem0.449)]
// CHECK:STDOUT:     %bound_method.loc33_9.2: <bound method> = bound_method %.loc33_9.2, %impl.elem0.loc33_9.1
// CHECK:STDOUT:     %specific_impl_fn.loc33_9.1: <specific function> = specific_impl_function %impl.elem0.loc33_9.1, @Destroy.WithSelf.Op(constants.%Destroy.facet.553) [symbolic = %specific_impl_fn.loc33_9.2 (constants.%specific_impl_fn.03f)]
// CHECK:STDOUT:     %bound_method.loc33_9.3: <bound method> = bound_method %.loc33_9.2, %specific_impl_fn.loc33_9.1
// CHECK:STDOUT:     %Destroy.WithSelf.Op.call.loc33: init %empty_tuple.type = call %bound_method.loc33_9.3(%.loc33_9.2)
// CHECK:STDOUT:     %impl.elem0.loc32_11.1: @Test.%.loc32_11.3 (%.d5a) = impl_witness_access constants.%Destroy.lookup_impl_witness.e97, element0 [symbolic = %impl.elem0.loc32_11.2 (constants.%impl.elem0.d1d)]
// CHECK:STDOUT:     %bound_method.loc32_11.2: <bound method> = bound_method %.loc32_11.2, %impl.elem0.loc32_11.1
// CHECK:STDOUT:     %specific_impl_fn.loc32_11.1: <specific function> = specific_impl_function %impl.elem0.loc32_11.1, @Destroy.WithSelf.Op(constants.%Destroy.facet.503) [symbolic = %specific_impl_fn.loc32_11.2 (constants.%specific_impl_fn.a5c)]
// CHECK:STDOUT:     %bound_method.loc32_11.3: <bound method> = bound_method %.loc32_11.2, %specific_impl_fn.loc32_11.1
// CHECK:STDOUT:     %Destroy.WithSelf.Op.call.loc32: init %empty_tuple.type = call %bound_method.loc32_11.3(%.loc32_11.2)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(constants.%R) {
// CHECK:STDOUT:   %R.loc30_10.1 => constants.%R
// CHECK:STDOUT:   %R.as_type.loc30_113.1 => constants.%R.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.ff859f.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(constants.%facet_value.665) {
// CHECK:STDOUT:   %R.loc30_10.1 => constants.%facet_value.665
// CHECK:STDOUT:   %R.as_type.loc30_113.1 => constants.%MutableRange
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.607
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc30 => constants.%complete_type
// CHECK:STDOUT:   %CppRangeForIterate.lookup_impl_witness => constants.%custom_witness.7e3fb2.1
// CHECK:STDOUT:   %CppRangeForIterate.facet => constants.%CppRangeForIterate.facet.d24
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.Begin.type => constants.%CppRangeForIterate.WithSelf.Begin.type.a13
// CHECK:STDOUT:   %.loc32_4 => constants.%.a34
// CHECK:STDOUT:   %impl.elem2.loc32_4.2 => constants.%MutableRange.Begin
// CHECK:STDOUT:   %specific_impl_fn.loc32_4.2 => constants.%MutableRange.Begin
// CHECK:STDOUT:   %impl.elem0.1 => constants.%Iterator.a7566c.1
// CHECK:STDOUT:   %require_complete.1 => constants.%complete_type
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.End.type => constants.%CppRangeForIterate.WithSelf.End.type.e9f
// CHECK:STDOUT:   %.loc33_4 => constants.%.2ee
// CHECK:STDOUT:   %impl.elem3.loc33_4.2 => constants.%MutableRange.End
// CHECK:STDOUT:   %specific_impl_fn.loc33_4.2 => constants.%MutableRange.End
// CHECK:STDOUT:   %impl.elem1.1 => constants.%Iterator.a7566c.1
// CHECK:STDOUT:   %require_complete.2 => constants.%complete_type
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.loc33 => constants.%custom_witness.b91
// CHECK:STDOUT:   %Destroy.facet.loc33 => constants.%Destroy.facet.fb8
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.loc33 => constants.%Destroy.WithSelf.Op.type.223
// CHECK:STDOUT:   %.loc33_9.3 => constants.%.438
// CHECK:STDOUT:   %impl.elem0.loc33_9.2 => constants.%Iterator.Op.78e
// CHECK:STDOUT:   %specific_impl_fn.loc33_9.2 => constants.%Iterator.Op.78e
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.loc32 => constants.%custom_witness.b91
// CHECK:STDOUT:   %Destroy.facet.loc32 => constants.%Destroy.facet.fb8
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.loc32 => constants.%Destroy.WithSelf.Op.type.223
// CHECK:STDOUT:   %.loc32_11.3 => constants.%.438
// CHECK:STDOUT:   %impl.elem0.loc32_11.2 => constants.%Iterator.Op.78e
// CHECK:STDOUT:   %specific_impl_fn.loc32_11.2 => constants.%Iterator.Op.78e
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(constants.%facet_value.57a) {
// CHECK:STDOUT:   %R.loc30_10.1 => constants.%facet_value.57a
// CHECK:STDOUT:   %R.as_type.loc30_113.1 => constants.%ConstRange
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.03c
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc30 => constants.%complete_type
// CHECK:STDOUT:   %CppRangeForIterate.lookup_impl_witness => constants.%custom_witness.7e3fb2.2
// CHECK:STDOUT:   %CppRangeForIterate.facet => constants.%CppRangeForIterate.facet.9f1
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.Begin.type => constants.%CppRangeForIterate.WithSelf.Begin.type.70e
// CHECK:STDOUT:   %.loc32_4 => constants.%.a48
// CHECK:STDOUT:   %impl.elem2.loc32_4.2 => constants.%ConstRange.Begin
// CHECK:STDOUT:   %specific_impl_fn.loc32_4.2 => constants.%ConstRange.Begin
// CHECK:STDOUT:   %impl.elem0.1 => constants.%Iterator.a7566c.2
// CHECK:STDOUT:   %require_complete.1 => constants.%complete_type
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.End.type => constants.%CppRangeForIterate.WithSelf.End.type.870
// CHECK:STDOUT:   %.loc33_4 => constants.%.6d4
// CHECK:STDOUT:   %impl.elem3.loc33_4.2 => constants.%ConstRange.End
// CHECK:STDOUT:   %specific_impl_fn.loc33_4.2 => constants.%ConstRange.End
// CHECK:STDOUT:   %impl.elem1.1 => constants.%Iterator.a7566c.2
// CHECK:STDOUT:   %require_complete.2 => constants.%complete_type
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.loc33 => constants.%custom_witness.018
// CHECK:STDOUT:   %Destroy.facet.loc33 => constants.%Destroy.facet.10b
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.loc33 => constants.%Destroy.WithSelf.Op.type.7f7
// CHECK:STDOUT:   %.loc33_9.3 => constants.%.cac
// CHECK:STDOUT:   %impl.elem0.loc33_9.2 => constants.%Iterator.Op.250
// CHECK:STDOUT:   %specific_impl_fn.loc33_9.2 => constants.%Iterator.Op.250
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.loc32 => constants.%custom_witness.018
// CHECK:STDOUT:   %Destroy.facet.loc32 => constants.%Destroy.facet.10b
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.loc32 => constants.%Destroy.WithSelf.Op.type.7f7
// CHECK:STDOUT:   %.loc32_11.3 => constants.%.cac
// CHECK:STDOUT:   %impl.elem0.loc32_11.2 => constants.%Iterator.Op.250
// CHECK:STDOUT:   %specific_impl_fn.loc32_11.2 => constants.%Iterator.Op.250
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- methods_return_different_types.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %CppRangeForIterate.type: type = facet_type <@CppRangeForIterate> [concrete]
// CHECK:STDOUT:   %Self.668: %CppRangeForIterate.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.End.type.45d9b7.1: type = fn_type @CppRangeForIterate.WithSelf.End, @CppRangeForIterate.WithSelf(%Self.668) [symbolic]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.End.e1b918.1: %CppRangeForIterate.WithSelf.End.type.45d9b7.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.Begin.type.e09a97.1: type = fn_type @CppRangeForIterate.WithSelf.Begin, @CppRangeForIterate.WithSelf(%Self.668) [symbolic]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.Begin.4c9a66.1: %CppRangeForIterate.WithSelf.Begin.type.e09a97.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %.Self.fda: %CppRangeForIterate.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %CppRangeForIterate.assoc_type: type = assoc_entity_type @CppRangeForIterate [concrete]
// CHECK:STDOUT:   %CppRangeForIterate.lookup_impl_witness.ae8: <witness> = lookup_impl_witness %.Self.fda, @CppRangeForIterate [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.354: type = impl_witness_access %CppRangeForIterate.lookup_impl_witness.ae8, element0 [symbolic_self]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %impl.elem1.fdc: type = impl_witness_access %CppRangeForIterate.lookup_impl_witness.ae8, element1 [symbolic_self]
// CHECK:STDOUT:   %CppRangeForIterate_where.type: type = facet_type <@CppRangeForIterate where %impl.elem0.354 impls @Destroy and %impl.elem1.fdc impls @Destroy> [concrete]
// CHECK:STDOUT:   %R: %CppRangeForIterate_where.type = symbolic_binding R, 0 [symbolic]
// CHECK:STDOUT:   %R.as_type: type = facet_access_type %R [symbolic]
// CHECK:STDOUT:   %pattern_type.ff859f.2: type = pattern_type %R.as_type [symbolic]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %assoc2: %CppRangeForIterate.assoc_type = assoc_entity element2, imports.%Core.import_ref.dad [concrete]
// CHECK:STDOUT:   %CppRangeForIterate.lookup_impl_witness.366fea.2: <witness> = lookup_impl_witness %R, @CppRangeForIterate [symbolic]
// CHECK:STDOUT:   %CppRangeForIterate.facet.527: %CppRangeForIterate.type = facet_value %R.as_type, (%CppRangeForIterate.lookup_impl_witness.366fea.2) [symbolic]
// CHECK:STDOUT:   %impl.elem0.b36147.2: type = impl_witness_access %CppRangeForIterate.lookup_impl_witness.366fea.2, element0 [symbolic]
// CHECK:STDOUT:   %impl.elem1.3ad37d.2: type = impl_witness_access %CppRangeForIterate.lookup_impl_witness.366fea.2, element1 [symbolic]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.Begin.type.aa6: type = fn_type @CppRangeForIterate.WithSelf.Begin, @CppRangeForIterate.WithSelf(%CppRangeForIterate.facet.527) [symbolic]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.End.type.ebe: type = fn_type @CppRangeForIterate.WithSelf.End, @CppRangeForIterate.WithSelf(%CppRangeForIterate.facet.527) [symbolic]
// CHECK:STDOUT:   %.227: type = fn_type_with_self_type %CppRangeForIterate.WithSelf.Begin.type.aa6, %CppRangeForIterate.facet.527 [symbolic]
// CHECK:STDOUT:   %impl.elem2: %.227 = impl_witness_access %CppRangeForIterate.lookup_impl_witness.366fea.2, element2 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.4a0: <specific function> = specific_impl_function %impl.elem2, @CppRangeForIterate.WithSelf.Begin(%CppRangeForIterate.facet.527) [symbolic]
// CHECK:STDOUT:   %assoc3: %CppRangeForIterate.assoc_type = assoc_entity element3, imports.%Core.import_ref.f0a [concrete]
// CHECK:STDOUT:   %.cab: type = fn_type_with_self_type %CppRangeForIterate.WithSelf.End.type.ebe, %CppRangeForIterate.facet.527 [symbolic]
// CHECK:STDOUT:   %impl.elem3: %.cab = impl_witness_access %CppRangeForIterate.lookup_impl_witness.366fea.2, element3 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.cc4: <specific function> = specific_impl_function %impl.elem3, @CppRangeForIterate.WithSelf.End(%CppRangeForIterate.facet.527) [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.4ab: <witness> = lookup_impl_witness %impl.elem1.3ad37d.2, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.553: %Destroy.type = facet_value %impl.elem1.3ad37d.2, (%Destroy.lookup_impl_witness.4ab) [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.b61: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.553) [symbolic]
// CHECK:STDOUT:   %.b17: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.b61, %Destroy.facet.553 [symbolic]
// CHECK:STDOUT:   %impl.elem0.449: %.b17 = impl_witness_access %Destroy.lookup_impl_witness.4ab, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.03f: <specific function> = specific_impl_function %impl.elem0.449, @Destroy.WithSelf.Op(%Destroy.facet.553) [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.e97: <witness> = lookup_impl_witness %impl.elem0.b36147.2, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.503: %Destroy.type = facet_value %impl.elem0.b36147.2, (%Destroy.lookup_impl_witness.e97) [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.4c8: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.503) [symbolic]
// CHECK:STDOUT:   %.d5a: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.4c8, %Destroy.facet.503 [symbolic]
// CHECK:STDOUT:   %impl.elem0.d1d: %.d5a = impl_witness_access %Destroy.lookup_impl_witness.e97, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.a5c: <specific function> = specific_impl_function %impl.elem0.d1d, @Destroy.WithSelf.Op(%Destroy.facet.503) [symbolic]
// CHECK:STDOUT:   %MutableRange: type = class_type @MutableRange [concrete]
// CHECK:STDOUT:   %pattern_type.607: type = pattern_type %MutableRange [concrete]
// CHECK:STDOUT:   %ConstRange: type = class_type @ConstRange [concrete]
// CHECK:STDOUT:   %pattern_type.03c: type = pattern_type %ConstRange [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %Iterator.a7566c.1: type = class_type @Iterator.1 [concrete]
// CHECK:STDOUT:   %Sentinel.885798.1: type = class_type @Sentinel.1 [concrete]
// CHECK:STDOUT:   %MutableRange.Begin.type: type = fn_type @MutableRange.Begin [concrete]
// CHECK:STDOUT:   %MutableRange.Begin: %MutableRange.Begin.type = struct_value () [concrete]
// CHECK:STDOUT:   %MutableRange.End.type: type = fn_type @MutableRange.End [concrete]
// CHECK:STDOUT:   %MutableRange.End: %MutableRange.End.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.607036.1: <witness> = custom_witness (%Iterator.a7566c.1, %Sentinel.885798.1, %MutableRange.Begin, %MutableRange.End), @CppRangeForIterate [concrete]
// CHECK:STDOUT:   %CppRangeForIterate.facet.5b8: %CppRangeForIterate.type = facet_value %MutableRange, (%custom_witness.607036.1) [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.f11: type = fn_type @Iterator.Op.1 [concrete]
// CHECK:STDOUT:   %Iterator.Op.78e: %Iterator.Op.type.f11 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b916b4.1: <witness> = custom_witness (%Iterator.Op.78e), @Destroy [concrete]
// CHECK:STDOUT:   %Sentinel.Op.type.f11: type = fn_type @Sentinel.Op.1 [concrete]
// CHECK:STDOUT:   %Sentinel.Op.78e: %Sentinel.Op.type.f11 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b916b4.2: <witness> = custom_witness (%Sentinel.Op.78e), @Destroy [concrete]
// CHECK:STDOUT:   %facet_value.8f5: %CppRangeForIterate_where.type = facet_value %MutableRange, (%custom_witness.607036.1, %custom_witness.b916b4.1, %custom_witness.b916b4.2) [concrete]
// CHECK:STDOUT:   %Iterator.a7566c.2: type = class_type @Iterator.2 [concrete]
// CHECK:STDOUT:   %Sentinel.885798.2: type = class_type @Sentinel.2 [concrete]
// CHECK:STDOUT:   %ConstRange.Begin.type: type = fn_type @ConstRange.Begin [concrete]
// CHECK:STDOUT:   %ConstRange.Begin: %ConstRange.Begin.type = struct_value () [concrete]
// CHECK:STDOUT:   %ConstRange.End.type: type = fn_type @ConstRange.End [concrete]
// CHECK:STDOUT:   %ConstRange.End: %ConstRange.End.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.607036.2: <witness> = custom_witness (%Iterator.a7566c.2, %Sentinel.885798.2, %ConstRange.Begin, %ConstRange.End), @CppRangeForIterate [concrete]
// CHECK:STDOUT:   %CppRangeForIterate.facet.3d4: %CppRangeForIterate.type = facet_value %ConstRange, (%custom_witness.607036.2) [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.798: type = fn_type @Iterator.Op.2 [concrete]
// CHECK:STDOUT:   %Iterator.Op.250: %Iterator.Op.type.798 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.0183d1.1: <witness> = custom_witness (%Iterator.Op.250), @Destroy [concrete]
// CHECK:STDOUT:   %Sentinel.Op.type.798: type = fn_type @Sentinel.Op.2 [concrete]
// CHECK:STDOUT:   %Sentinel.Op.250: %Sentinel.Op.type.798 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.0183d1.2: <witness> = custom_witness (%Sentinel.Op.250), @Destroy [concrete]
// CHECK:STDOUT:   %facet_value.058: %CppRangeForIterate_where.type = facet_value %ConstRange, (%custom_witness.607036.2, %custom_witness.0183d1.1, %custom_witness.0183d1.2) [concrete]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.Begin.type.dc4: type = fn_type @CppRangeForIterate.WithSelf.Begin, @CppRangeForIterate.WithSelf(%CppRangeForIterate.facet.5b8) [concrete]
// CHECK:STDOUT:   %.314: type = fn_type_with_self_type %CppRangeForIterate.WithSelf.Begin.type.dc4, %CppRangeForIterate.facet.5b8 [concrete]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.End.type.143: type = fn_type @CppRangeForIterate.WithSelf.End, @CppRangeForIterate.WithSelf(%CppRangeForIterate.facet.5b8) [concrete]
// CHECK:STDOUT:   %.0e0: type = fn_type_with_self_type %CppRangeForIterate.WithSelf.End.type.143, %CppRangeForIterate.facet.5b8 [concrete]
// CHECK:STDOUT:   %Destroy.facet.7db: %Destroy.type = facet_value %Sentinel.885798.1, (%custom_witness.b916b4.2) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.9c0: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.7db) [concrete]
// CHECK:STDOUT:   %.d81: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.9c0, %Destroy.facet.7db [concrete]
// CHECK:STDOUT:   %Destroy.facet.fb8: %Destroy.type = facet_value %Iterator.a7566c.1, (%custom_witness.b916b4.1) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.223: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.fb8) [concrete]
// CHECK:STDOUT:   %.438: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.223, %Destroy.facet.fb8 [concrete]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.Begin.type.c94: type = fn_type @CppRangeForIterate.WithSelf.Begin, @CppRangeForIterate.WithSelf(%CppRangeForIterate.facet.3d4) [concrete]
// CHECK:STDOUT:   %.6f0: type = fn_type_with_self_type %CppRangeForIterate.WithSelf.Begin.type.c94, %CppRangeForIterate.facet.3d4 [concrete]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.End.type.e75: type = fn_type @CppRangeForIterate.WithSelf.End, @CppRangeForIterate.WithSelf(%CppRangeForIterate.facet.3d4) [concrete]
// CHECK:STDOUT:   %.c37: type = fn_type_with_self_type %CppRangeForIterate.WithSelf.End.type.e75, %CppRangeForIterate.facet.3d4 [concrete]
// CHECK:STDOUT:   %Destroy.facet.e7c: %Destroy.type = facet_value %Sentinel.885798.2, (%custom_witness.0183d1.2) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.b01: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.e7c) [concrete]
// CHECK:STDOUT:   %.c9b: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.b01, %Destroy.facet.e7c [concrete]
// CHECK:STDOUT:   %Destroy.facet.10b: %Destroy.type = facet_value %Iterator.a7566c.2, (%custom_witness.0183d1.1) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.7f7: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.10b) [concrete]
// CHECK:STDOUT:   %.cac: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.7f7, %Destroy.facet.10b [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.4af: %CppRangeForIterate.assoc_type = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [concrete = constants.%assoc2]
// CHECK:STDOUT:   %Core.import_ref.83c: %CppRangeForIterate.assoc_type = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [concrete = constants.%assoc3]
// CHECK:STDOUT:   %Core.import_ref.dad: @CppRangeForIterate.WithSelf.%CppRangeForIterate.WithSelf.Begin.type (%CppRangeForIterate.WithSelf.Begin.type.e09a97.1) = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @CppRangeForIterate.WithSelf.%CppRangeForIterate.WithSelf.Begin (constants.%CppRangeForIterate.WithSelf.Begin.4c9a66.1)]
// CHECK:STDOUT:   %Core.import_ref.f0a: @CppRangeForIterate.WithSelf.%CppRangeForIterate.WithSelf.End.type (%CppRangeForIterate.WithSelf.End.type.45d9b7.1) = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @CppRangeForIterate.WithSelf.%CppRangeForIterate.WithSelf.End (constants.%CppRangeForIterate.WithSelf.End.e1b918.1)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Test(%R.loc40_10.2: %CppRangeForIterate_where.type) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %CppRangeForIterate.lookup_impl_witness: <witness> = lookup_impl_witness %R.loc40_10.1, @CppRangeForIterate [symbolic = %CppRangeForIterate.lookup_impl_witness (constants.%CppRangeForIterate.lookup_impl_witness.366fea.2)]
// CHECK:STDOUT:   %CppRangeForIterate.facet: %CppRangeForIterate.type = facet_value %R.as_type.loc40_113.1, (%CppRangeForIterate.lookup_impl_witness) [symbolic = %CppRangeForIterate.facet (constants.%CppRangeForIterate.facet.527)]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.Begin.type: type = fn_type @CppRangeForIterate.WithSelf.Begin, @CppRangeForIterate.WithSelf(%CppRangeForIterate.facet) [symbolic = %CppRangeForIterate.WithSelf.Begin.type (constants.%CppRangeForIterate.WithSelf.Begin.type.aa6)]
// CHECK:STDOUT:   %.loc42_4: type = fn_type_with_self_type %CppRangeForIterate.WithSelf.Begin.type, %CppRangeForIterate.facet [symbolic = %.loc42_4 (constants.%.227)]
// CHECK:STDOUT:   %impl.elem2.loc42_4.2: @Test.%.loc42_4 (%.227) = impl_witness_access %CppRangeForIterate.lookup_impl_witness, element2 [symbolic = %impl.elem2.loc42_4.2 (constants.%impl.elem2)]
// CHECK:STDOUT:   %specific_impl_fn.loc42_4.2: <specific function> = specific_impl_function %impl.elem2.loc42_4.2, @CppRangeForIterate.WithSelf.Begin(%CppRangeForIterate.facet) [symbolic = %specific_impl_fn.loc42_4.2 (constants.%specific_impl_fn.4a0)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.End.type: type = fn_type @CppRangeForIterate.WithSelf.End, @CppRangeForIterate.WithSelf(%CppRangeForIterate.facet) [symbolic = %CppRangeForIterate.WithSelf.End.type (constants.%CppRangeForIterate.WithSelf.End.type.ebe)]
// CHECK:STDOUT:   %.loc43_4: type = fn_type_with_self_type %CppRangeForIterate.WithSelf.End.type, %CppRangeForIterate.facet [symbolic = %.loc43_4 (constants.%.cab)]
// CHECK:STDOUT:   %impl.elem3.loc43_4.2: @Test.%.loc43_4 (%.cab) = impl_witness_access %CppRangeForIterate.lookup_impl_witness, element3 [symbolic = %impl.elem3.loc43_4.2 (constants.%impl.elem3)]
// CHECK:STDOUT:   %specific_impl_fn.loc43_4.2: <specific function> = specific_impl_function %impl.elem3.loc43_4.2, @CppRangeForIterate.WithSelf.End(%CppRangeForIterate.facet) [symbolic = %specific_impl_fn.loc43_4.2 (constants.%specific_impl_fn.cc4)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.loc43: <witness> = lookup_impl_witness %impl.elem1.1, @Destroy [symbolic = %Destroy.lookup_impl_witness.loc43 (constants.%Destroy.lookup_impl_witness.4ab)]
// CHECK:STDOUT:   %Destroy.facet.loc43: %Destroy.type = facet_value %impl.elem1.1, (%Destroy.lookup_impl_witness.loc43) [symbolic = %Destroy.facet.loc43 (constants.%Destroy.facet.553)]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.loc43: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.loc43) [symbolic = %Destroy.WithSelf.Op.type.loc43 (constants.%Destroy.WithSelf.Op.type.b61)]
// CHECK:STDOUT:   %.loc43_9.3: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.loc43, %Destroy.facet.loc43 [symbolic = %.loc43_9.3 (constants.%.b17)]
// CHECK:STDOUT:   %impl.elem0.loc43_9.2: @Test.%.loc43_9.3 (%.b17) = impl_witness_access %Destroy.lookup_impl_witness.loc43, element0 [symbolic = %impl.elem0.loc43_9.2 (constants.%impl.elem0.449)]
// CHECK:STDOUT:   %specific_impl_fn.loc43_9.2: <specific function> = specific_impl_function %impl.elem0.loc43_9.2, @Destroy.WithSelf.Op(%Destroy.facet.loc43) [symbolic = %specific_impl_fn.loc43_9.2 (constants.%specific_impl_fn.03f)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.loc42: <witness> = lookup_impl_witness %impl.elem0.1, @Destroy [symbolic = %Destroy.lookup_impl_witness.loc42 (constants.%Destroy.lookup_impl_witness.e97)]
// CHECK:STDOUT:   %Destroy.facet.loc42: %Destroy.type = facet_value %impl.elem0.1, (%Destroy.lookup_impl_witness.loc42) [symbolic = %Destroy.facet.loc42 (constants.%Destroy.facet.503)]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.loc42: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.loc42) [symbolic = %Destroy.WithSelf.Op.type.loc42 (constants.%Destroy.WithSelf.Op.type.4c8)]
// CHECK:STDOUT:   %.loc42_11.3: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.loc42, %Destroy.facet.loc42 [symbolic = %.loc42_11.3 (constants.%.d5a)]
// CHECK:STDOUT:   %impl.elem0.loc42_11.2: @Test.%.loc42_11.3 (%.d5a) = impl_witness_access %Destroy.lookup_impl_witness.loc42, element0 [symbolic = %impl.elem0.loc42_11.2 (constants.%impl.elem0.d1d)]
// CHECK:STDOUT:   %specific_impl_fn.loc42_11.2: <specific function> = specific_impl_function %impl.elem0.loc42_11.2, @Destroy.WithSelf.Op(%Destroy.facet.loc42) [symbolic = %specific_impl_fn.loc42_11.2 (constants.%specific_impl_fn.a5c)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%r.param: ref @Test.%R.as_type.loc40_113.1 (%R.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %r.ref.loc42: ref @Test.%R.as_type.loc40_113.1 (%R.as_type) = name_ref r, %r
// CHECK:STDOUT:     %Begin.ref: %CppRangeForIterate.assoc_type = name_ref Begin, imports.%Core.import_ref.4af [concrete = constants.%assoc2]
// CHECK:STDOUT:     %impl.elem2.loc42_4.1: @Test.%.loc42_4 (%.227) = impl_witness_access constants.%CppRangeForIterate.lookup_impl_witness.366fea.2, element2 [symbolic = %impl.elem2.loc42_4.2 (constants.%impl.elem2)]
// CHECK:STDOUT:     %bound_method.loc42_4: <bound method> = bound_method %r.ref.loc42, %impl.elem2.loc42_4.1
// CHECK:STDOUT:     %specific_impl_fn.loc42_4.1: <specific function> = specific_impl_function %impl.elem2.loc42_4.1, @CppRangeForIterate.WithSelf.Begin(constants.%CppRangeForIterate.facet.527) [symbolic = %specific_impl_fn.loc42_4.2 (constants.%specific_impl_fn.4a0)]
// CHECK:STDOUT:     %bound_method.loc42_11.1: <bound method> = bound_method %r.ref.loc42, %specific_impl_fn.loc42_4.1
// CHECK:STDOUT:     %.loc42_11.1: ref @Test.%impl.elem0.1 (%impl.elem0.b36147.2) = temporary_storage
// CHECK:STDOUT:     %CppRangeForIterate.WithSelf.Begin.call: init @Test.%impl.elem0.1 (%impl.elem0.b36147.2) to %.loc42_11.1 = call %bound_method.loc42_11.1(%r.ref.loc42)
// CHECK:STDOUT:     %.loc42_11.2: ref @Test.%impl.elem0.1 (%impl.elem0.b36147.2) = temporary %.loc42_11.1, %CppRangeForIterate.WithSelf.Begin.call
// CHECK:STDOUT:     %r.ref.loc43: ref @Test.%R.as_type.loc40_113.1 (%R.as_type) = name_ref r, %r
// CHECK:STDOUT:     %End.ref: %CppRangeForIterate.assoc_type = name_ref End, imports.%Core.import_ref.83c [concrete = constants.%assoc3]
// CHECK:STDOUT:     %impl.elem3.loc43_4.1: @Test.%.loc43_4 (%.cab) = impl_witness_access constants.%CppRangeForIterate.lookup_impl_witness.366fea.2, element3 [symbolic = %impl.elem3.loc43_4.2 (constants.%impl.elem3)]
// CHECK:STDOUT:     %bound_method.loc43_4: <bound method> = bound_method %r.ref.loc43, %impl.elem3.loc43_4.1
// CHECK:STDOUT:     %specific_impl_fn.loc43_4.1: <specific function> = specific_impl_function %impl.elem3.loc43_4.1, @CppRangeForIterate.WithSelf.End(constants.%CppRangeForIterate.facet.527) [symbolic = %specific_impl_fn.loc43_4.2 (constants.%specific_impl_fn.cc4)]
// CHECK:STDOUT:     %bound_method.loc43_9.1: <bound method> = bound_method %r.ref.loc43, %specific_impl_fn.loc43_4.1
// CHECK:STDOUT:     %.loc43_9.1: ref @Test.%impl.elem1.1 (%impl.elem1.3ad37d.2) = temporary_storage
// CHECK:STDOUT:     %CppRangeForIterate.WithSelf.End.call: init @Test.%impl.elem1.1 (%impl.elem1.3ad37d.2) to %.loc43_9.1 = call %bound_method.loc43_9.1(%r.ref.loc43)
// CHECK:STDOUT:     %.loc43_9.2: ref @Test.%impl.elem1.1 (%impl.elem1.3ad37d.2) = temporary %.loc43_9.1, %CppRangeForIterate.WithSelf.End.call
// CHECK:STDOUT:     %impl.elem0.loc43_9.1: @Test.%.loc43_9.3 (%.b17) = impl_witness_access constants.%Destroy.lookup_impl_witness.4ab, element0 [symbolic = %impl.elem0.loc43_9.2 (constants.%impl.elem0.449)]
// CHECK:STDOUT:     %bound_method.loc43_9.2: <bound method> = bound_method %.loc43_9.2, %impl.elem0.loc43_9.1
// CHECK:STDOUT:     %specific_impl_fn.loc43_9.1: <specific function> = specific_impl_function %impl.elem0.loc43_9.1, @Destroy.WithSelf.Op(constants.%Destroy.facet.553) [symbolic = %specific_impl_fn.loc43_9.2 (constants.%specific_impl_fn.03f)]
// CHECK:STDOUT:     %bound_method.loc43_9.3: <bound method> = bound_method %.loc43_9.2, %specific_impl_fn.loc43_9.1
// CHECK:STDOUT:     %Destroy.WithSelf.Op.call.loc43: init %empty_tuple.type = call %bound_method.loc43_9.3(%.loc43_9.2)
// CHECK:STDOUT:     %impl.elem0.loc42_11.1: @Test.%.loc42_11.3 (%.d5a) = impl_witness_access constants.%Destroy.lookup_impl_witness.e97, element0 [symbolic = %impl.elem0.loc42_11.2 (constants.%impl.elem0.d1d)]
// CHECK:STDOUT:     %bound_method.loc42_11.2: <bound method> = bound_method %.loc42_11.2, %impl.elem0.loc42_11.1
// CHECK:STDOUT:     %specific_impl_fn.loc42_11.1: <specific function> = specific_impl_function %impl.elem0.loc42_11.1, @Destroy.WithSelf.Op(constants.%Destroy.facet.503) [symbolic = %specific_impl_fn.loc42_11.2 (constants.%specific_impl_fn.a5c)]
// CHECK:STDOUT:     %bound_method.loc42_11.3: <bound method> = bound_method %.loc42_11.2, %specific_impl_fn.loc42_11.1
// CHECK:STDOUT:     %Destroy.WithSelf.Op.call.loc42: init %empty_tuple.type = call %bound_method.loc42_11.3(%.loc42_11.2)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(constants.%R) {
// CHECK:STDOUT:   %R.loc40_10.1 => constants.%R
// CHECK:STDOUT:   %R.as_type.loc40_113.1 => constants.%R.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.ff859f.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(constants.%facet_value.8f5) {
// CHECK:STDOUT:   %R.loc40_10.1 => constants.%facet_value.8f5
// CHECK:STDOUT:   %R.as_type.loc40_113.1 => constants.%MutableRange
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.607
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc40 => constants.%complete_type
// CHECK:STDOUT:   %CppRangeForIterate.lookup_impl_witness => constants.%custom_witness.607036.1
// CHECK:STDOUT:   %CppRangeForIterate.facet => constants.%CppRangeForIterate.facet.5b8
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.Begin.type => constants.%CppRangeForIterate.WithSelf.Begin.type.dc4
// CHECK:STDOUT:   %.loc42_4 => constants.%.314
// CHECK:STDOUT:   %impl.elem2.loc42_4.2 => constants.%MutableRange.Begin
// CHECK:STDOUT:   %specific_impl_fn.loc42_4.2 => constants.%MutableRange.Begin
// CHECK:STDOUT:   %impl.elem0.1 => constants.%Iterator.a7566c.1
// CHECK:STDOUT:   %require_complete.1 => constants.%complete_type
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.End.type => constants.%CppRangeForIterate.WithSelf.End.type.143
// CHECK:STDOUT:   %.loc43_4 => constants.%.0e0
// CHECK:STDOUT:   %impl.elem3.loc43_4.2 => constants.%MutableRange.End
// CHECK:STDOUT:   %specific_impl_fn.loc43_4.2 => constants.%MutableRange.End
// CHECK:STDOUT:   %impl.elem1.1 => constants.%Sentinel.885798.1
// CHECK:STDOUT:   %require_complete.2 => constants.%complete_type
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.loc43 => constants.%custom_witness.b916b4.2
// CHECK:STDOUT:   %Destroy.facet.loc43 => constants.%Destroy.facet.7db
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.loc43 => constants.%Destroy.WithSelf.Op.type.9c0
// CHECK:STDOUT:   %.loc43_9.3 => constants.%.d81
// CHECK:STDOUT:   %impl.elem0.loc43_9.2 => constants.%Sentinel.Op.78e
// CHECK:STDOUT:   %specific_impl_fn.loc43_9.2 => constants.%Sentinel.Op.78e
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.loc42 => constants.%custom_witness.b916b4.1
// CHECK:STDOUT:   %Destroy.facet.loc42 => constants.%Destroy.facet.fb8
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.loc42 => constants.%Destroy.WithSelf.Op.type.223
// CHECK:STDOUT:   %.loc42_11.3 => constants.%.438
// CHECK:STDOUT:   %impl.elem0.loc42_11.2 => constants.%Iterator.Op.78e
// CHECK:STDOUT:   %specific_impl_fn.loc42_11.2 => constants.%Iterator.Op.78e
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(constants.%facet_value.058) {
// CHECK:STDOUT:   %R.loc40_10.1 => constants.%facet_value.058
// CHECK:STDOUT:   %R.as_type.loc40_113.1 => constants.%ConstRange
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.03c
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc40 => constants.%complete_type
// CHECK:STDOUT:   %CppRangeForIterate.lookup_impl_witness => constants.%custom_witness.607036.2
// CHECK:STDOUT:   %CppRangeForIterate.facet => constants.%CppRangeForIterate.facet.3d4
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.Begin.type => constants.%CppRangeForIterate.WithSelf.Begin.type.c94
// CHECK:STDOUT:   %.loc42_4 => constants.%.6f0
// CHECK:STDOUT:   %impl.elem2.loc42_4.2 => constants.%ConstRange.Begin
// CHECK:STDOUT:   %specific_impl_fn.loc42_4.2 => constants.%ConstRange.Begin
// CHECK:STDOUT:   %impl.elem0.1 => constants.%Iterator.a7566c.2
// CHECK:STDOUT:   %require_complete.1 => constants.%complete_type
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.End.type => constants.%CppRangeForIterate.WithSelf.End.type.e75
// CHECK:STDOUT:   %.loc43_4 => constants.%.c37
// CHECK:STDOUT:   %impl.elem3.loc43_4.2 => constants.%ConstRange.End
// CHECK:STDOUT:   %specific_impl_fn.loc43_4.2 => constants.%ConstRange.End
// CHECK:STDOUT:   %impl.elem1.1 => constants.%Sentinel.885798.2
// CHECK:STDOUT:   %require_complete.2 => constants.%complete_type
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.loc43 => constants.%custom_witness.0183d1.2
// CHECK:STDOUT:   %Destroy.facet.loc43 => constants.%Destroy.facet.e7c
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.loc43 => constants.%Destroy.WithSelf.Op.type.b01
// CHECK:STDOUT:   %.loc43_9.3 => constants.%.c9b
// CHECK:STDOUT:   %impl.elem0.loc43_9.2 => constants.%Sentinel.Op.250
// CHECK:STDOUT:   %specific_impl_fn.loc43_9.2 => constants.%Sentinel.Op.250
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.loc42 => constants.%custom_witness.0183d1.1
// CHECK:STDOUT:   %Destroy.facet.loc42 => constants.%Destroy.facet.10b
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.loc42 => constants.%Destroy.WithSelf.Op.type.7f7
// CHECK:STDOUT:   %.loc42_11.3 => constants.%.cac
// CHECK:STDOUT:   %impl.elem0.loc42_11.2 => constants.%Iterator.Op.250
// CHECK:STDOUT:   %specific_impl_fn.loc42_11.2 => constants.%Iterator.Op.250
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- adl_return_same_types.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %CppRangeForIterate.type: type = facet_type <@CppRangeForIterate> [concrete]
// CHECK:STDOUT:   %Self.668: %CppRangeForIterate.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.End.type.45d9b7.1: type = fn_type @CppRangeForIterate.WithSelf.End, @CppRangeForIterate.WithSelf(%Self.668) [symbolic]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.End.e1b918.1: %CppRangeForIterate.WithSelf.End.type.45d9b7.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.Begin.type.e09a97.1: type = fn_type @CppRangeForIterate.WithSelf.Begin, @CppRangeForIterate.WithSelf(%Self.668) [symbolic]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.Begin.4c9a66.1: %CppRangeForIterate.WithSelf.Begin.type.e09a97.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %.Self.fda: %CppRangeForIterate.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %CppRangeForIterate.assoc_type: type = assoc_entity_type @CppRangeForIterate [concrete]
// CHECK:STDOUT:   %CppRangeForIterate.lookup_impl_witness.ae8: <witness> = lookup_impl_witness %.Self.fda, @CppRangeForIterate [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.354: type = impl_witness_access %CppRangeForIterate.lookup_impl_witness.ae8, element0 [symbolic_self]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %impl.elem1.fdc: type = impl_witness_access %CppRangeForIterate.lookup_impl_witness.ae8, element1 [symbolic_self]
// CHECK:STDOUT:   %CppRangeForIterate_where.type: type = facet_type <@CppRangeForIterate where %impl.elem0.354 impls @Destroy and %impl.elem1.fdc impls @Destroy> [concrete]
// CHECK:STDOUT:   %R: %CppRangeForIterate_where.type = symbolic_binding R, 0 [symbolic]
// CHECK:STDOUT:   %R.as_type: type = facet_access_type %R [symbolic]
// CHECK:STDOUT:   %pattern_type.ff859f.2: type = pattern_type %R.as_type [symbolic]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %assoc2: %CppRangeForIterate.assoc_type = assoc_entity element2, imports.%Core.import_ref.dad [concrete]
// CHECK:STDOUT:   %CppRangeForIterate.lookup_impl_witness.366fea.2: <witness> = lookup_impl_witness %R, @CppRangeForIterate [symbolic]
// CHECK:STDOUT:   %CppRangeForIterate.facet.527: %CppRangeForIterate.type = facet_value %R.as_type, (%CppRangeForIterate.lookup_impl_witness.366fea.2) [symbolic]
// CHECK:STDOUT:   %impl.elem0.b36147.2: type = impl_witness_access %CppRangeForIterate.lookup_impl_witness.366fea.2, element0 [symbolic]
// CHECK:STDOUT:   %impl.elem1.3ad37d.2: type = impl_witness_access %CppRangeForIterate.lookup_impl_witness.366fea.2, element1 [symbolic]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.Begin.type.aa6: type = fn_type @CppRangeForIterate.WithSelf.Begin, @CppRangeForIterate.WithSelf(%CppRangeForIterate.facet.527) [symbolic]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.End.type.ebe: type = fn_type @CppRangeForIterate.WithSelf.End, @CppRangeForIterate.WithSelf(%CppRangeForIterate.facet.527) [symbolic]
// CHECK:STDOUT:   %.227: type = fn_type_with_self_type %CppRangeForIterate.WithSelf.Begin.type.aa6, %CppRangeForIterate.facet.527 [symbolic]
// CHECK:STDOUT:   %impl.elem2: %.227 = impl_witness_access %CppRangeForIterate.lookup_impl_witness.366fea.2, element2 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.4a0: <specific function> = specific_impl_function %impl.elem2, @CppRangeForIterate.WithSelf.Begin(%CppRangeForIterate.facet.527) [symbolic]
// CHECK:STDOUT:   %assoc3: %CppRangeForIterate.assoc_type = assoc_entity element3, imports.%Core.import_ref.f0a [concrete]
// CHECK:STDOUT:   %.cab: type = fn_type_with_self_type %CppRangeForIterate.WithSelf.End.type.ebe, %CppRangeForIterate.facet.527 [symbolic]
// CHECK:STDOUT:   %impl.elem3: %.cab = impl_witness_access %CppRangeForIterate.lookup_impl_witness.366fea.2, element3 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.cc4: <specific function> = specific_impl_function %impl.elem3, @CppRangeForIterate.WithSelf.End(%CppRangeForIterate.facet.527) [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.4ab: <witness> = lookup_impl_witness %impl.elem1.3ad37d.2, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.553: %Destroy.type = facet_value %impl.elem1.3ad37d.2, (%Destroy.lookup_impl_witness.4ab) [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.b61: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.553) [symbolic]
// CHECK:STDOUT:   %.b17: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.b61, %Destroy.facet.553 [symbolic]
// CHECK:STDOUT:   %impl.elem0.449: %.b17 = impl_witness_access %Destroy.lookup_impl_witness.4ab, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.03f: <specific function> = specific_impl_function %impl.elem0.449, @Destroy.WithSelf.Op(%Destroy.facet.553) [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.e97: <witness> = lookup_impl_witness %impl.elem0.b36147.2, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.503: %Destroy.type = facet_value %impl.elem0.b36147.2, (%Destroy.lookup_impl_witness.e97) [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.4c8: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.503) [symbolic]
// CHECK:STDOUT:   %.d5a: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.4c8, %Destroy.facet.503 [symbolic]
// CHECK:STDOUT:   %impl.elem0.d1d: %.d5a = impl_witness_access %Destroy.lookup_impl_witness.e97, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.a5c: <specific function> = specific_impl_function %impl.elem0.d1d, @Destroy.WithSelf.Op(%Destroy.facet.503) [symbolic]
// CHECK:STDOUT:   %MutableRange: type = class_type @MutableRange [concrete]
// CHECK:STDOUT:   %pattern_type.f4c: type = pattern_type %MutableRange [concrete]
// CHECK:STDOUT:   %ConstRange: type = class_type @ConstRange [concrete]
// CHECK:STDOUT:   %pattern_type.e01: type = pattern_type %ConstRange [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %Iterator.b6bc8f.1: type = class_type @Iterator.1 [concrete]
// CHECK:STDOUT:   %Begin.type.793876.1: type = fn_type @Begin.1 [concrete]
// CHECK:STDOUT:   %Begin.1544bb.1: %Begin.type.793876.1 = struct_value () [concrete]
// CHECK:STDOUT:   %End.type.16398b.1: type = fn_type @End.1 [concrete]
// CHECK:STDOUT:   %End.be76e1.1: %End.type.16398b.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.827774.1: <witness> = custom_witness (%Iterator.b6bc8f.1, %Iterator.b6bc8f.1, %Begin.1544bb.1, %End.be76e1.1), @CppRangeForIterate [concrete]
// CHECK:STDOUT:   %CppRangeForIterate.facet.a7f: %CppRangeForIterate.type = facet_value %MutableRange, (%custom_witness.827774.1) [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.37d: type = fn_type @Iterator.Op.1 [concrete]
// CHECK:STDOUT:   %Iterator.Op.26b: %Iterator.Op.type.37d = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.415: <witness> = custom_witness (%Iterator.Op.26b), @Destroy [concrete]
// CHECK:STDOUT:   %facet_value.c35: %CppRangeForIterate_where.type = facet_value %MutableRange, (%custom_witness.827774.1, %custom_witness.415) [concrete]
// CHECK:STDOUT:   %Iterator.b6bc8f.2: type = class_type @Iterator.2 [concrete]
// CHECK:STDOUT:   %Begin.type.793876.2: type = fn_type @Begin.2 [concrete]
// CHECK:STDOUT:   %Begin.1544bb.2: %Begin.type.793876.2 = struct_value () [concrete]
// CHECK:STDOUT:   %End.type.16398b.2: type = fn_type @End.2 [concrete]
// CHECK:STDOUT:   %End.be76e1.2: %End.type.16398b.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.827774.2: <witness> = custom_witness (%Iterator.b6bc8f.2, %Iterator.b6bc8f.2, %Begin.1544bb.2, %End.be76e1.2), @CppRangeForIterate [concrete]
// CHECK:STDOUT:   %CppRangeForIterate.facet.5e6: %CppRangeForIterate.type = facet_value %ConstRange, (%custom_witness.827774.2) [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.152: type = fn_type @Iterator.Op.2 [concrete]
// CHECK:STDOUT:   %Iterator.Op.84b: %Iterator.Op.type.152 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.0d3: <witness> = custom_witness (%Iterator.Op.84b), @Destroy [concrete]
// CHECK:STDOUT:   %facet_value.057: %CppRangeForIterate_where.type = facet_value %ConstRange, (%custom_witness.827774.2, %custom_witness.0d3) [concrete]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.Begin.type.bfa: type = fn_type @CppRangeForIterate.WithSelf.Begin, @CppRangeForIterate.WithSelf(%CppRangeForIterate.facet.a7f) [concrete]
// CHECK:STDOUT:   %.73f: type = fn_type_with_self_type %CppRangeForIterate.WithSelf.Begin.type.bfa, %CppRangeForIterate.facet.a7f [concrete]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.End.type.c37: type = fn_type @CppRangeForIterate.WithSelf.End, @CppRangeForIterate.WithSelf(%CppRangeForIterate.facet.a7f) [concrete]
// CHECK:STDOUT:   %.03b: type = fn_type_with_self_type %CppRangeForIterate.WithSelf.End.type.c37, %CppRangeForIterate.facet.a7f [concrete]
// CHECK:STDOUT:   %Destroy.facet.bcd: %Destroy.type = facet_value %Iterator.b6bc8f.1, (%custom_witness.415) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.92e: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.bcd) [concrete]
// CHECK:STDOUT:   %.c1a: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.92e, %Destroy.facet.bcd [concrete]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.Begin.type.532: type = fn_type @CppRangeForIterate.WithSelf.Begin, @CppRangeForIterate.WithSelf(%CppRangeForIterate.facet.5e6) [concrete]
// CHECK:STDOUT:   %.bca: type = fn_type_with_self_type %CppRangeForIterate.WithSelf.Begin.type.532, %CppRangeForIterate.facet.5e6 [concrete]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.End.type.87b: type = fn_type @CppRangeForIterate.WithSelf.End, @CppRangeForIterate.WithSelf(%CppRangeForIterate.facet.5e6) [concrete]
// CHECK:STDOUT:   %.83b: type = fn_type_with_self_type %CppRangeForIterate.WithSelf.End.type.87b, %CppRangeForIterate.facet.5e6 [concrete]
// CHECK:STDOUT:   %Destroy.facet.e71: %Destroy.type = facet_value %Iterator.b6bc8f.2, (%custom_witness.0d3) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.1d5: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.e71) [concrete]
// CHECK:STDOUT:   %.bef: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.1d5, %Destroy.facet.e71 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.4af: %CppRangeForIterate.assoc_type = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [concrete = constants.%assoc2]
// CHECK:STDOUT:   %Core.import_ref.83c: %CppRangeForIterate.assoc_type = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [concrete = constants.%assoc3]
// CHECK:STDOUT:   %Core.import_ref.dad: @CppRangeForIterate.WithSelf.%CppRangeForIterate.WithSelf.Begin.type (%CppRangeForIterate.WithSelf.Begin.type.e09a97.1) = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @CppRangeForIterate.WithSelf.%CppRangeForIterate.WithSelf.Begin (constants.%CppRangeForIterate.WithSelf.Begin.4c9a66.1)]
// CHECK:STDOUT:   %Core.import_ref.f0a: @CppRangeForIterate.WithSelf.%CppRangeForIterate.WithSelf.End.type (%CppRangeForIterate.WithSelf.End.type.45d9b7.1) = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @CppRangeForIterate.WithSelf.%CppRangeForIterate.WithSelf.End (constants.%CppRangeForIterate.WithSelf.End.e1b918.1)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Test(%R.loc32_10.2: %CppRangeForIterate_where.type) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %CppRangeForIterate.lookup_impl_witness: <witness> = lookup_impl_witness %R.loc32_10.1, @CppRangeForIterate [symbolic = %CppRangeForIterate.lookup_impl_witness (constants.%CppRangeForIterate.lookup_impl_witness.366fea.2)]
// CHECK:STDOUT:   %CppRangeForIterate.facet: %CppRangeForIterate.type = facet_value %R.as_type.loc32_113.1, (%CppRangeForIterate.lookup_impl_witness) [symbolic = %CppRangeForIterate.facet (constants.%CppRangeForIterate.facet.527)]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.Begin.type: type = fn_type @CppRangeForIterate.WithSelf.Begin, @CppRangeForIterate.WithSelf(%CppRangeForIterate.facet) [symbolic = %CppRangeForIterate.WithSelf.Begin.type (constants.%CppRangeForIterate.WithSelf.Begin.type.aa6)]
// CHECK:STDOUT:   %.loc34_4: type = fn_type_with_self_type %CppRangeForIterate.WithSelf.Begin.type, %CppRangeForIterate.facet [symbolic = %.loc34_4 (constants.%.227)]
// CHECK:STDOUT:   %impl.elem2.loc34_4.2: @Test.%.loc34_4 (%.227) = impl_witness_access %CppRangeForIterate.lookup_impl_witness, element2 [symbolic = %impl.elem2.loc34_4.2 (constants.%impl.elem2)]
// CHECK:STDOUT:   %specific_impl_fn.loc34_4.2: <specific function> = specific_impl_function %impl.elem2.loc34_4.2, @CppRangeForIterate.WithSelf.Begin(%CppRangeForIterate.facet) [symbolic = %specific_impl_fn.loc34_4.2 (constants.%specific_impl_fn.4a0)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.End.type: type = fn_type @CppRangeForIterate.WithSelf.End, @CppRangeForIterate.WithSelf(%CppRangeForIterate.facet) [symbolic = %CppRangeForIterate.WithSelf.End.type (constants.%CppRangeForIterate.WithSelf.End.type.ebe)]
// CHECK:STDOUT:   %.loc35_4: type = fn_type_with_self_type %CppRangeForIterate.WithSelf.End.type, %CppRangeForIterate.facet [symbolic = %.loc35_4 (constants.%.cab)]
// CHECK:STDOUT:   %impl.elem3.loc35_4.2: @Test.%.loc35_4 (%.cab) = impl_witness_access %CppRangeForIterate.lookup_impl_witness, element3 [symbolic = %impl.elem3.loc35_4.2 (constants.%impl.elem3)]
// CHECK:STDOUT:   %specific_impl_fn.loc35_4.2: <specific function> = specific_impl_function %impl.elem3.loc35_4.2, @CppRangeForIterate.WithSelf.End(%CppRangeForIterate.facet) [symbolic = %specific_impl_fn.loc35_4.2 (constants.%specific_impl_fn.cc4)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.loc35: <witness> = lookup_impl_witness %impl.elem1.1, @Destroy [symbolic = %Destroy.lookup_impl_witness.loc35 (constants.%Destroy.lookup_impl_witness.4ab)]
// CHECK:STDOUT:   %Destroy.facet.loc35: %Destroy.type = facet_value %impl.elem1.1, (%Destroy.lookup_impl_witness.loc35) [symbolic = %Destroy.facet.loc35 (constants.%Destroy.facet.553)]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.loc35: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.loc35) [symbolic = %Destroy.WithSelf.Op.type.loc35 (constants.%Destroy.WithSelf.Op.type.b61)]
// CHECK:STDOUT:   %.loc35_9.3: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.loc35, %Destroy.facet.loc35 [symbolic = %.loc35_9.3 (constants.%.b17)]
// CHECK:STDOUT:   %impl.elem0.loc35_9.2: @Test.%.loc35_9.3 (%.b17) = impl_witness_access %Destroy.lookup_impl_witness.loc35, element0 [symbolic = %impl.elem0.loc35_9.2 (constants.%impl.elem0.449)]
// CHECK:STDOUT:   %specific_impl_fn.loc35_9.2: <specific function> = specific_impl_function %impl.elem0.loc35_9.2, @Destroy.WithSelf.Op(%Destroy.facet.loc35) [symbolic = %specific_impl_fn.loc35_9.2 (constants.%specific_impl_fn.03f)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.loc34: <witness> = lookup_impl_witness %impl.elem0.1, @Destroy [symbolic = %Destroy.lookup_impl_witness.loc34 (constants.%Destroy.lookup_impl_witness.e97)]
// CHECK:STDOUT:   %Destroy.facet.loc34: %Destroy.type = facet_value %impl.elem0.1, (%Destroy.lookup_impl_witness.loc34) [symbolic = %Destroy.facet.loc34 (constants.%Destroy.facet.503)]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.loc34: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.loc34) [symbolic = %Destroy.WithSelf.Op.type.loc34 (constants.%Destroy.WithSelf.Op.type.4c8)]
// CHECK:STDOUT:   %.loc34_11.3: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.loc34, %Destroy.facet.loc34 [symbolic = %.loc34_11.3 (constants.%.d5a)]
// CHECK:STDOUT:   %impl.elem0.loc34_11.2: @Test.%.loc34_11.3 (%.d5a) = impl_witness_access %Destroy.lookup_impl_witness.loc34, element0 [symbolic = %impl.elem0.loc34_11.2 (constants.%impl.elem0.d1d)]
// CHECK:STDOUT:   %specific_impl_fn.loc34_11.2: <specific function> = specific_impl_function %impl.elem0.loc34_11.2, @Destroy.WithSelf.Op(%Destroy.facet.loc34) [symbolic = %specific_impl_fn.loc34_11.2 (constants.%specific_impl_fn.a5c)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%r.param: ref @Test.%R.as_type.loc32_113.1 (%R.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %r.ref.loc34: ref @Test.%R.as_type.loc32_113.1 (%R.as_type) = name_ref r, %r
// CHECK:STDOUT:     %Begin.ref: %CppRangeForIterate.assoc_type = name_ref Begin, imports.%Core.import_ref.4af [concrete = constants.%assoc2]
// CHECK:STDOUT:     %impl.elem2.loc34_4.1: @Test.%.loc34_4 (%.227) = impl_witness_access constants.%CppRangeForIterate.lookup_impl_witness.366fea.2, element2 [symbolic = %impl.elem2.loc34_4.2 (constants.%impl.elem2)]
// CHECK:STDOUT:     %bound_method.loc34_4: <bound method> = bound_method %r.ref.loc34, %impl.elem2.loc34_4.1
// CHECK:STDOUT:     %specific_impl_fn.loc34_4.1: <specific function> = specific_impl_function %impl.elem2.loc34_4.1, @CppRangeForIterate.WithSelf.Begin(constants.%CppRangeForIterate.facet.527) [symbolic = %specific_impl_fn.loc34_4.2 (constants.%specific_impl_fn.4a0)]
// CHECK:STDOUT:     %bound_method.loc34_11.1: <bound method> = bound_method %r.ref.loc34, %specific_impl_fn.loc34_4.1
// CHECK:STDOUT:     %.loc34_11.1: ref @Test.%impl.elem0.1 (%impl.elem0.b36147.2) = temporary_storage
// CHECK:STDOUT:     %CppRangeForIterate.WithSelf.Begin.call: init @Test.%impl.elem0.1 (%impl.elem0.b36147.2) to %.loc34_11.1 = call %bound_method.loc34_11.1(%r.ref.loc34)
// CHECK:STDOUT:     %.loc34_11.2: ref @Test.%impl.elem0.1 (%impl.elem0.b36147.2) = temporary %.loc34_11.1, %CppRangeForIterate.WithSelf.Begin.call
// CHECK:STDOUT:     %r.ref.loc35: ref @Test.%R.as_type.loc32_113.1 (%R.as_type) = name_ref r, %r
// CHECK:STDOUT:     %End.ref: %CppRangeForIterate.assoc_type = name_ref End, imports.%Core.import_ref.83c [concrete = constants.%assoc3]
// CHECK:STDOUT:     %impl.elem3.loc35_4.1: @Test.%.loc35_4 (%.cab) = impl_witness_access constants.%CppRangeForIterate.lookup_impl_witness.366fea.2, element3 [symbolic = %impl.elem3.loc35_4.2 (constants.%impl.elem3)]
// CHECK:STDOUT:     %bound_method.loc35_4: <bound method> = bound_method %r.ref.loc35, %impl.elem3.loc35_4.1
// CHECK:STDOUT:     %specific_impl_fn.loc35_4.1: <specific function> = specific_impl_function %impl.elem3.loc35_4.1, @CppRangeForIterate.WithSelf.End(constants.%CppRangeForIterate.facet.527) [symbolic = %specific_impl_fn.loc35_4.2 (constants.%specific_impl_fn.cc4)]
// CHECK:STDOUT:     %bound_method.loc35_9.1: <bound method> = bound_method %r.ref.loc35, %specific_impl_fn.loc35_4.1
// CHECK:STDOUT:     %.loc35_9.1: ref @Test.%impl.elem1.1 (%impl.elem1.3ad37d.2) = temporary_storage
// CHECK:STDOUT:     %CppRangeForIterate.WithSelf.End.call: init @Test.%impl.elem1.1 (%impl.elem1.3ad37d.2) to %.loc35_9.1 = call %bound_method.loc35_9.1(%r.ref.loc35)
// CHECK:STDOUT:     %.loc35_9.2: ref @Test.%impl.elem1.1 (%impl.elem1.3ad37d.2) = temporary %.loc35_9.1, %CppRangeForIterate.WithSelf.End.call
// CHECK:STDOUT:     %impl.elem0.loc35_9.1: @Test.%.loc35_9.3 (%.b17) = impl_witness_access constants.%Destroy.lookup_impl_witness.4ab, element0 [symbolic = %impl.elem0.loc35_9.2 (constants.%impl.elem0.449)]
// CHECK:STDOUT:     %bound_method.loc35_9.2: <bound method> = bound_method %.loc35_9.2, %impl.elem0.loc35_9.1
// CHECK:STDOUT:     %specific_impl_fn.loc35_9.1: <specific function> = specific_impl_function %impl.elem0.loc35_9.1, @Destroy.WithSelf.Op(constants.%Destroy.facet.553) [symbolic = %specific_impl_fn.loc35_9.2 (constants.%specific_impl_fn.03f)]
// CHECK:STDOUT:     %bound_method.loc35_9.3: <bound method> = bound_method %.loc35_9.2, %specific_impl_fn.loc35_9.1
// CHECK:STDOUT:     %Destroy.WithSelf.Op.call.loc35: init %empty_tuple.type = call %bound_method.loc35_9.3(%.loc35_9.2)
// CHECK:STDOUT:     %impl.elem0.loc34_11.1: @Test.%.loc34_11.3 (%.d5a) = impl_witness_access constants.%Destroy.lookup_impl_witness.e97, element0 [symbolic = %impl.elem0.loc34_11.2 (constants.%impl.elem0.d1d)]
// CHECK:STDOUT:     %bound_method.loc34_11.2: <bound method> = bound_method %.loc34_11.2, %impl.elem0.loc34_11.1
// CHECK:STDOUT:     %specific_impl_fn.loc34_11.1: <specific function> = specific_impl_function %impl.elem0.loc34_11.1, @Destroy.WithSelf.Op(constants.%Destroy.facet.503) [symbolic = %specific_impl_fn.loc34_11.2 (constants.%specific_impl_fn.a5c)]
// CHECK:STDOUT:     %bound_method.loc34_11.3: <bound method> = bound_method %.loc34_11.2, %specific_impl_fn.loc34_11.1
// CHECK:STDOUT:     %Destroy.WithSelf.Op.call.loc34: init %empty_tuple.type = call %bound_method.loc34_11.3(%.loc34_11.2)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(constants.%R) {
// CHECK:STDOUT:   %R.loc32_10.1 => constants.%R
// CHECK:STDOUT:   %R.as_type.loc32_113.1 => constants.%R.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.ff859f.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(constants.%facet_value.c35) {
// CHECK:STDOUT:   %R.loc32_10.1 => constants.%facet_value.c35
// CHECK:STDOUT:   %R.as_type.loc32_113.1 => constants.%MutableRange
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.f4c
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc32 => constants.%complete_type
// CHECK:STDOUT:   %CppRangeForIterate.lookup_impl_witness => constants.%custom_witness.827774.1
// CHECK:STDOUT:   %CppRangeForIterate.facet => constants.%CppRangeForIterate.facet.a7f
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.Begin.type => constants.%CppRangeForIterate.WithSelf.Begin.type.bfa
// CHECK:STDOUT:   %.loc34_4 => constants.%.73f
// CHECK:STDOUT:   %impl.elem2.loc34_4.2 => constants.%Begin.1544bb.1
// CHECK:STDOUT:   %specific_impl_fn.loc34_4.2 => constants.%Begin.1544bb.1
// CHECK:STDOUT:   %impl.elem0.1 => constants.%Iterator.b6bc8f.1
// CHECK:STDOUT:   %require_complete.1 => constants.%complete_type
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.End.type => constants.%CppRangeForIterate.WithSelf.End.type.c37
// CHECK:STDOUT:   %.loc35_4 => constants.%.03b
// CHECK:STDOUT:   %impl.elem3.loc35_4.2 => constants.%End.be76e1.1
// CHECK:STDOUT:   %specific_impl_fn.loc35_4.2 => constants.%End.be76e1.1
// CHECK:STDOUT:   %impl.elem1.1 => constants.%Iterator.b6bc8f.1
// CHECK:STDOUT:   %require_complete.2 => constants.%complete_type
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.loc35 => constants.%custom_witness.415
// CHECK:STDOUT:   %Destroy.facet.loc35 => constants.%Destroy.facet.bcd
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.loc35 => constants.%Destroy.WithSelf.Op.type.92e
// CHECK:STDOUT:   %.loc35_9.3 => constants.%.c1a
// CHECK:STDOUT:   %impl.elem0.loc35_9.2 => constants.%Iterator.Op.26b
// CHECK:STDOUT:   %specific_impl_fn.loc35_9.2 => constants.%Iterator.Op.26b
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.loc34 => constants.%custom_witness.415
// CHECK:STDOUT:   %Destroy.facet.loc34 => constants.%Destroy.facet.bcd
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.loc34 => constants.%Destroy.WithSelf.Op.type.92e
// CHECK:STDOUT:   %.loc34_11.3 => constants.%.c1a
// CHECK:STDOUT:   %impl.elem0.loc34_11.2 => constants.%Iterator.Op.26b
// CHECK:STDOUT:   %specific_impl_fn.loc34_11.2 => constants.%Iterator.Op.26b
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(constants.%facet_value.057) {
// CHECK:STDOUT:   %R.loc32_10.1 => constants.%facet_value.057
// CHECK:STDOUT:   %R.as_type.loc32_113.1 => constants.%ConstRange
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.e01
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc32 => constants.%complete_type
// CHECK:STDOUT:   %CppRangeForIterate.lookup_impl_witness => constants.%custom_witness.827774.2
// CHECK:STDOUT:   %CppRangeForIterate.facet => constants.%CppRangeForIterate.facet.5e6
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.Begin.type => constants.%CppRangeForIterate.WithSelf.Begin.type.532
// CHECK:STDOUT:   %.loc34_4 => constants.%.bca
// CHECK:STDOUT:   %impl.elem2.loc34_4.2 => constants.%Begin.1544bb.2
// CHECK:STDOUT:   %specific_impl_fn.loc34_4.2 => constants.%Begin.1544bb.2
// CHECK:STDOUT:   %impl.elem0.1 => constants.%Iterator.b6bc8f.2
// CHECK:STDOUT:   %require_complete.1 => constants.%complete_type
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.End.type => constants.%CppRangeForIterate.WithSelf.End.type.87b
// CHECK:STDOUT:   %.loc35_4 => constants.%.83b
// CHECK:STDOUT:   %impl.elem3.loc35_4.2 => constants.%End.be76e1.2
// CHECK:STDOUT:   %specific_impl_fn.loc35_4.2 => constants.%End.be76e1.2
// CHECK:STDOUT:   %impl.elem1.1 => constants.%Iterator.b6bc8f.2
// CHECK:STDOUT:   %require_complete.2 => constants.%complete_type
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.loc35 => constants.%custom_witness.0d3
// CHECK:STDOUT:   %Destroy.facet.loc35 => constants.%Destroy.facet.e71
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.loc35 => constants.%Destroy.WithSelf.Op.type.1d5
// CHECK:STDOUT:   %.loc35_9.3 => constants.%.bef
// CHECK:STDOUT:   %impl.elem0.loc35_9.2 => constants.%Iterator.Op.84b
// CHECK:STDOUT:   %specific_impl_fn.loc35_9.2 => constants.%Iterator.Op.84b
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.loc34 => constants.%custom_witness.0d3
// CHECK:STDOUT:   %Destroy.facet.loc34 => constants.%Destroy.facet.e71
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.loc34 => constants.%Destroy.WithSelf.Op.type.1d5
// CHECK:STDOUT:   %.loc34_11.3 => constants.%.bef
// CHECK:STDOUT:   %impl.elem0.loc34_11.2 => constants.%Iterator.Op.84b
// CHECK:STDOUT:   %specific_impl_fn.loc34_11.2 => constants.%Iterator.Op.84b
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- adl_returns_different_types.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %CppRangeForIterate.type: type = facet_type <@CppRangeForIterate> [concrete]
// CHECK:STDOUT:   %Self.668: %CppRangeForIterate.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.End.type.45d9b7.1: type = fn_type @CppRangeForIterate.WithSelf.End, @CppRangeForIterate.WithSelf(%Self.668) [symbolic]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.End.e1b918.1: %CppRangeForIterate.WithSelf.End.type.45d9b7.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.Begin.type.e09a97.1: type = fn_type @CppRangeForIterate.WithSelf.Begin, @CppRangeForIterate.WithSelf(%Self.668) [symbolic]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.Begin.4c9a66.1: %CppRangeForIterate.WithSelf.Begin.type.e09a97.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %.Self.fda: %CppRangeForIterate.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %CppRangeForIterate.assoc_type: type = assoc_entity_type @CppRangeForIterate [concrete]
// CHECK:STDOUT:   %CppRangeForIterate.lookup_impl_witness.ae8: <witness> = lookup_impl_witness %.Self.fda, @CppRangeForIterate [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.354: type = impl_witness_access %CppRangeForIterate.lookup_impl_witness.ae8, element0 [symbolic_self]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %impl.elem1.fdc: type = impl_witness_access %CppRangeForIterate.lookup_impl_witness.ae8, element1 [symbolic_self]
// CHECK:STDOUT:   %CppRangeForIterate_where.type: type = facet_type <@CppRangeForIterate where %impl.elem0.354 impls @Destroy and %impl.elem1.fdc impls @Destroy> [concrete]
// CHECK:STDOUT:   %R: %CppRangeForIterate_where.type = symbolic_binding R, 0 [symbolic]
// CHECK:STDOUT:   %R.as_type: type = facet_access_type %R [symbolic]
// CHECK:STDOUT:   %pattern_type.ff859f.2: type = pattern_type %R.as_type [symbolic]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %assoc2: %CppRangeForIterate.assoc_type = assoc_entity element2, imports.%Core.import_ref.dad [concrete]
// CHECK:STDOUT:   %CppRangeForIterate.lookup_impl_witness.366fea.2: <witness> = lookup_impl_witness %R, @CppRangeForIterate [symbolic]
// CHECK:STDOUT:   %CppRangeForIterate.facet.527: %CppRangeForIterate.type = facet_value %R.as_type, (%CppRangeForIterate.lookup_impl_witness.366fea.2) [symbolic]
// CHECK:STDOUT:   %impl.elem0.b36147.2: type = impl_witness_access %CppRangeForIterate.lookup_impl_witness.366fea.2, element0 [symbolic]
// CHECK:STDOUT:   %impl.elem1.3ad37d.2: type = impl_witness_access %CppRangeForIterate.lookup_impl_witness.366fea.2, element1 [symbolic]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.Begin.type.aa6: type = fn_type @CppRangeForIterate.WithSelf.Begin, @CppRangeForIterate.WithSelf(%CppRangeForIterate.facet.527) [symbolic]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.End.type.ebe: type = fn_type @CppRangeForIterate.WithSelf.End, @CppRangeForIterate.WithSelf(%CppRangeForIterate.facet.527) [symbolic]
// CHECK:STDOUT:   %.227: type = fn_type_with_self_type %CppRangeForIterate.WithSelf.Begin.type.aa6, %CppRangeForIterate.facet.527 [symbolic]
// CHECK:STDOUT:   %impl.elem2: %.227 = impl_witness_access %CppRangeForIterate.lookup_impl_witness.366fea.2, element2 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.4a0: <specific function> = specific_impl_function %impl.elem2, @CppRangeForIterate.WithSelf.Begin(%CppRangeForIterate.facet.527) [symbolic]
// CHECK:STDOUT:   %assoc3: %CppRangeForIterate.assoc_type = assoc_entity element3, imports.%Core.import_ref.f0a [concrete]
// CHECK:STDOUT:   %.cab: type = fn_type_with_self_type %CppRangeForIterate.WithSelf.End.type.ebe, %CppRangeForIterate.facet.527 [symbolic]
// CHECK:STDOUT:   %impl.elem3: %.cab = impl_witness_access %CppRangeForIterate.lookup_impl_witness.366fea.2, element3 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.cc4: <specific function> = specific_impl_function %impl.elem3, @CppRangeForIterate.WithSelf.End(%CppRangeForIterate.facet.527) [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.4ab: <witness> = lookup_impl_witness %impl.elem1.3ad37d.2, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.553: %Destroy.type = facet_value %impl.elem1.3ad37d.2, (%Destroy.lookup_impl_witness.4ab) [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.b61: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.553) [symbolic]
// CHECK:STDOUT:   %.b17: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.b61, %Destroy.facet.553 [symbolic]
// CHECK:STDOUT:   %impl.elem0.449: %.b17 = impl_witness_access %Destroy.lookup_impl_witness.4ab, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.03f: <specific function> = specific_impl_function %impl.elem0.449, @Destroy.WithSelf.Op(%Destroy.facet.553) [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.e97: <witness> = lookup_impl_witness %impl.elem0.b36147.2, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.503: %Destroy.type = facet_value %impl.elem0.b36147.2, (%Destroy.lookup_impl_witness.e97) [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.4c8: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.503) [symbolic]
// CHECK:STDOUT:   %.d5a: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.4c8, %Destroy.facet.503 [symbolic]
// CHECK:STDOUT:   %impl.elem0.d1d: %.d5a = impl_witness_access %Destroy.lookup_impl_witness.e97, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.a5c: <specific function> = specific_impl_function %impl.elem0.d1d, @Destroy.WithSelf.Op(%Destroy.facet.503) [symbolic]
// CHECK:STDOUT:   %MutableRange: type = class_type @MutableRange [concrete]
// CHECK:STDOUT:   %pattern_type.f4c: type = pattern_type %MutableRange [concrete]
// CHECK:STDOUT:   %ConstRange: type = class_type @ConstRange [concrete]
// CHECK:STDOUT:   %pattern_type.e01: type = pattern_type %ConstRange [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %Iterator.b6bc8f.1: type = class_type @Iterator.1 [concrete]
// CHECK:STDOUT:   %Sentinel.50a788.1: type = class_type @Sentinel.1 [concrete]
// CHECK:STDOUT:   %Begin.type.793876.1: type = fn_type @Begin.1 [concrete]
// CHECK:STDOUT:   %Begin.1544bb.1: %Begin.type.793876.1 = struct_value () [concrete]
// CHECK:STDOUT:   %End.type.16398b.1: type = fn_type @End.1 [concrete]
// CHECK:STDOUT:   %End.be76e1.1: %End.type.16398b.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.1bf5e5.1: <witness> = custom_witness (%Iterator.b6bc8f.1, %Sentinel.50a788.1, %Begin.1544bb.1, %End.be76e1.1), @CppRangeForIterate [concrete]
// CHECK:STDOUT:   %CppRangeForIterate.facet.d1e: %CppRangeForIterate.type = facet_value %MutableRange, (%custom_witness.1bf5e5.1) [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.37d: type = fn_type @Iterator.Op.1 [concrete]
// CHECK:STDOUT:   %Iterator.Op.26b: %Iterator.Op.type.37d = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.415faa.1: <witness> = custom_witness (%Iterator.Op.26b), @Destroy [concrete]
// CHECK:STDOUT:   %Sentinel.Op.type.37d: type = fn_type @Sentinel.Op.1 [concrete]
// CHECK:STDOUT:   %Sentinel.Op.26b: %Sentinel.Op.type.37d = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.415faa.2: <witness> = custom_witness (%Sentinel.Op.26b), @Destroy [concrete]
// CHECK:STDOUT:   %facet_value.b34: %CppRangeForIterate_where.type = facet_value %MutableRange, (%custom_witness.1bf5e5.1, %custom_witness.415faa.1, %custom_witness.415faa.2) [concrete]
// CHECK:STDOUT:   %Iterator.b6bc8f.2: type = class_type @Iterator.2 [concrete]
// CHECK:STDOUT:   %Sentinel.50a788.2: type = class_type @Sentinel.2 [concrete]
// CHECK:STDOUT:   %Begin.type.793876.2: type = fn_type @Begin.2 [concrete]
// CHECK:STDOUT:   %Begin.1544bb.2: %Begin.type.793876.2 = struct_value () [concrete]
// CHECK:STDOUT:   %End.type.16398b.2: type = fn_type @End.2 [concrete]
// CHECK:STDOUT:   %End.be76e1.2: %End.type.16398b.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.1bf5e5.2: <witness> = custom_witness (%Iterator.b6bc8f.2, %Sentinel.50a788.2, %Begin.1544bb.2, %End.be76e1.2), @CppRangeForIterate [concrete]
// CHECK:STDOUT:   %CppRangeForIterate.facet.18f: %CppRangeForIterate.type = facet_value %ConstRange, (%custom_witness.1bf5e5.2) [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.152: type = fn_type @Iterator.Op.2 [concrete]
// CHECK:STDOUT:   %Iterator.Op.84b: %Iterator.Op.type.152 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.0d3cfa.1: <witness> = custom_witness (%Iterator.Op.84b), @Destroy [concrete]
// CHECK:STDOUT:   %Sentinel.Op.type.152: type = fn_type @Sentinel.Op.2 [concrete]
// CHECK:STDOUT:   %Sentinel.Op.84b: %Sentinel.Op.type.152 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.0d3cfa.2: <witness> = custom_witness (%Sentinel.Op.84b), @Destroy [concrete]
// CHECK:STDOUT:   %facet_value.11c: %CppRangeForIterate_where.type = facet_value %ConstRange, (%custom_witness.1bf5e5.2, %custom_witness.0d3cfa.1, %custom_witness.0d3cfa.2) [concrete]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.Begin.type.bf8: type = fn_type @CppRangeForIterate.WithSelf.Begin, @CppRangeForIterate.WithSelf(%CppRangeForIterate.facet.d1e) [concrete]
// CHECK:STDOUT:   %.504: type = fn_type_with_self_type %CppRangeForIterate.WithSelf.Begin.type.bf8, %CppRangeForIterate.facet.d1e [concrete]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.End.type.580: type = fn_type @CppRangeForIterate.WithSelf.End, @CppRangeForIterate.WithSelf(%CppRangeForIterate.facet.d1e) [concrete]
// CHECK:STDOUT:   %.1e1: type = fn_type_with_self_type %CppRangeForIterate.WithSelf.End.type.580, %CppRangeForIterate.facet.d1e [concrete]
// CHECK:STDOUT:   %Destroy.facet.eb0: %Destroy.type = facet_value %Sentinel.50a788.1, (%custom_witness.415faa.2) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.552: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.eb0) [concrete]
// CHECK:STDOUT:   %.9ca: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.552, %Destroy.facet.eb0 [concrete]
// CHECK:STDOUT:   %Destroy.facet.bcd: %Destroy.type = facet_value %Iterator.b6bc8f.1, (%custom_witness.415faa.1) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.92e: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.bcd) [concrete]
// CHECK:STDOUT:   %.c1a: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.92e, %Destroy.facet.bcd [concrete]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.Begin.type.a30: type = fn_type @CppRangeForIterate.WithSelf.Begin, @CppRangeForIterate.WithSelf(%CppRangeForIterate.facet.18f) [concrete]
// CHECK:STDOUT:   %.e4b: type = fn_type_with_self_type %CppRangeForIterate.WithSelf.Begin.type.a30, %CppRangeForIterate.facet.18f [concrete]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.End.type.aba: type = fn_type @CppRangeForIterate.WithSelf.End, @CppRangeForIterate.WithSelf(%CppRangeForIterate.facet.18f) [concrete]
// CHECK:STDOUT:   %.d10: type = fn_type_with_self_type %CppRangeForIterate.WithSelf.End.type.aba, %CppRangeForIterate.facet.18f [concrete]
// CHECK:STDOUT:   %Destroy.facet.a2a: %Destroy.type = facet_value %Sentinel.50a788.2, (%custom_witness.0d3cfa.2) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.1fe: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.a2a) [concrete]
// CHECK:STDOUT:   %.f48: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.1fe, %Destroy.facet.a2a [concrete]
// CHECK:STDOUT:   %Destroy.facet.e71: %Destroy.type = facet_value %Iterator.b6bc8f.2, (%custom_witness.0d3cfa.1) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.1d5: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.e71) [concrete]
// CHECK:STDOUT:   %.bef: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.1d5, %Destroy.facet.e71 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.4af: %CppRangeForIterate.assoc_type = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [concrete = constants.%assoc2]
// CHECK:STDOUT:   %Core.import_ref.83c: %CppRangeForIterate.assoc_type = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [concrete = constants.%assoc3]
// CHECK:STDOUT:   %Core.import_ref.dad: @CppRangeForIterate.WithSelf.%CppRangeForIterate.WithSelf.Begin.type (%CppRangeForIterate.WithSelf.Begin.type.e09a97.1) = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @CppRangeForIterate.WithSelf.%CppRangeForIterate.WithSelf.Begin (constants.%CppRangeForIterate.WithSelf.Begin.4c9a66.1)]
// CHECK:STDOUT:   %Core.import_ref.f0a: @CppRangeForIterate.WithSelf.%CppRangeForIterate.WithSelf.End.type (%CppRangeForIterate.WithSelf.End.type.45d9b7.1) = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @CppRangeForIterate.WithSelf.%CppRangeForIterate.WithSelf.End (constants.%CppRangeForIterate.WithSelf.End.e1b918.1)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Test(%R.loc42_10.2: %CppRangeForIterate_where.type) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %CppRangeForIterate.lookup_impl_witness: <witness> = lookup_impl_witness %R.loc42_10.1, @CppRangeForIterate [symbolic = %CppRangeForIterate.lookup_impl_witness (constants.%CppRangeForIterate.lookup_impl_witness.366fea.2)]
// CHECK:STDOUT:   %CppRangeForIterate.facet: %CppRangeForIterate.type = facet_value %R.as_type.loc42_113.1, (%CppRangeForIterate.lookup_impl_witness) [symbolic = %CppRangeForIterate.facet (constants.%CppRangeForIterate.facet.527)]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.Begin.type: type = fn_type @CppRangeForIterate.WithSelf.Begin, @CppRangeForIterate.WithSelf(%CppRangeForIterate.facet) [symbolic = %CppRangeForIterate.WithSelf.Begin.type (constants.%CppRangeForIterate.WithSelf.Begin.type.aa6)]
// CHECK:STDOUT:   %.loc44_4: type = fn_type_with_self_type %CppRangeForIterate.WithSelf.Begin.type, %CppRangeForIterate.facet [symbolic = %.loc44_4 (constants.%.227)]
// CHECK:STDOUT:   %impl.elem2.loc44_4.2: @Test.%.loc44_4 (%.227) = impl_witness_access %CppRangeForIterate.lookup_impl_witness, element2 [symbolic = %impl.elem2.loc44_4.2 (constants.%impl.elem2)]
// CHECK:STDOUT:   %specific_impl_fn.loc44_4.2: <specific function> = specific_impl_function %impl.elem2.loc44_4.2, @CppRangeForIterate.WithSelf.Begin(%CppRangeForIterate.facet) [symbolic = %specific_impl_fn.loc44_4.2 (constants.%specific_impl_fn.4a0)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.End.type: type = fn_type @CppRangeForIterate.WithSelf.End, @CppRangeForIterate.WithSelf(%CppRangeForIterate.facet) [symbolic = %CppRangeForIterate.WithSelf.End.type (constants.%CppRangeForIterate.WithSelf.End.type.ebe)]
// CHECK:STDOUT:   %.loc45_4: type = fn_type_with_self_type %CppRangeForIterate.WithSelf.End.type, %CppRangeForIterate.facet [symbolic = %.loc45_4 (constants.%.cab)]
// CHECK:STDOUT:   %impl.elem3.loc45_4.2: @Test.%.loc45_4 (%.cab) = impl_witness_access %CppRangeForIterate.lookup_impl_witness, element3 [symbolic = %impl.elem3.loc45_4.2 (constants.%impl.elem3)]
// CHECK:STDOUT:   %specific_impl_fn.loc45_4.2: <specific function> = specific_impl_function %impl.elem3.loc45_4.2, @CppRangeForIterate.WithSelf.End(%CppRangeForIterate.facet) [symbolic = %specific_impl_fn.loc45_4.2 (constants.%specific_impl_fn.cc4)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.loc45: <witness> = lookup_impl_witness %impl.elem1.1, @Destroy [symbolic = %Destroy.lookup_impl_witness.loc45 (constants.%Destroy.lookup_impl_witness.4ab)]
// CHECK:STDOUT:   %Destroy.facet.loc45: %Destroy.type = facet_value %impl.elem1.1, (%Destroy.lookup_impl_witness.loc45) [symbolic = %Destroy.facet.loc45 (constants.%Destroy.facet.553)]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.loc45: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.loc45) [symbolic = %Destroy.WithSelf.Op.type.loc45 (constants.%Destroy.WithSelf.Op.type.b61)]
// CHECK:STDOUT:   %.loc45_9.3: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.loc45, %Destroy.facet.loc45 [symbolic = %.loc45_9.3 (constants.%.b17)]
// CHECK:STDOUT:   %impl.elem0.loc45_9.2: @Test.%.loc45_9.3 (%.b17) = impl_witness_access %Destroy.lookup_impl_witness.loc45, element0 [symbolic = %impl.elem0.loc45_9.2 (constants.%impl.elem0.449)]
// CHECK:STDOUT:   %specific_impl_fn.loc45_9.2: <specific function> = specific_impl_function %impl.elem0.loc45_9.2, @Destroy.WithSelf.Op(%Destroy.facet.loc45) [symbolic = %specific_impl_fn.loc45_9.2 (constants.%specific_impl_fn.03f)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.loc44: <witness> = lookup_impl_witness %impl.elem0.1, @Destroy [symbolic = %Destroy.lookup_impl_witness.loc44 (constants.%Destroy.lookup_impl_witness.e97)]
// CHECK:STDOUT:   %Destroy.facet.loc44: %Destroy.type = facet_value %impl.elem0.1, (%Destroy.lookup_impl_witness.loc44) [symbolic = %Destroy.facet.loc44 (constants.%Destroy.facet.503)]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.loc44: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.loc44) [symbolic = %Destroy.WithSelf.Op.type.loc44 (constants.%Destroy.WithSelf.Op.type.4c8)]
// CHECK:STDOUT:   %.loc44_11.3: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.loc44, %Destroy.facet.loc44 [symbolic = %.loc44_11.3 (constants.%.d5a)]
// CHECK:STDOUT:   %impl.elem0.loc44_11.2: @Test.%.loc44_11.3 (%.d5a) = impl_witness_access %Destroy.lookup_impl_witness.loc44, element0 [symbolic = %impl.elem0.loc44_11.2 (constants.%impl.elem0.d1d)]
// CHECK:STDOUT:   %specific_impl_fn.loc44_11.2: <specific function> = specific_impl_function %impl.elem0.loc44_11.2, @Destroy.WithSelf.Op(%Destroy.facet.loc44) [symbolic = %specific_impl_fn.loc44_11.2 (constants.%specific_impl_fn.a5c)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%r.param: ref @Test.%R.as_type.loc42_113.1 (%R.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %r.ref.loc44: ref @Test.%R.as_type.loc42_113.1 (%R.as_type) = name_ref r, %r
// CHECK:STDOUT:     %Begin.ref: %CppRangeForIterate.assoc_type = name_ref Begin, imports.%Core.import_ref.4af [concrete = constants.%assoc2]
// CHECK:STDOUT:     %impl.elem2.loc44_4.1: @Test.%.loc44_4 (%.227) = impl_witness_access constants.%CppRangeForIterate.lookup_impl_witness.366fea.2, element2 [symbolic = %impl.elem2.loc44_4.2 (constants.%impl.elem2)]
// CHECK:STDOUT:     %bound_method.loc44_4: <bound method> = bound_method %r.ref.loc44, %impl.elem2.loc44_4.1
// CHECK:STDOUT:     %specific_impl_fn.loc44_4.1: <specific function> = specific_impl_function %impl.elem2.loc44_4.1, @CppRangeForIterate.WithSelf.Begin(constants.%CppRangeForIterate.facet.527) [symbolic = %specific_impl_fn.loc44_4.2 (constants.%specific_impl_fn.4a0)]
// CHECK:STDOUT:     %bound_method.loc44_11.1: <bound method> = bound_method %r.ref.loc44, %specific_impl_fn.loc44_4.1
// CHECK:STDOUT:     %.loc44_11.1: ref @Test.%impl.elem0.1 (%impl.elem0.b36147.2) = temporary_storage
// CHECK:STDOUT:     %CppRangeForIterate.WithSelf.Begin.call: init @Test.%impl.elem0.1 (%impl.elem0.b36147.2) to %.loc44_11.1 = call %bound_method.loc44_11.1(%r.ref.loc44)
// CHECK:STDOUT:     %.loc44_11.2: ref @Test.%impl.elem0.1 (%impl.elem0.b36147.2) = temporary %.loc44_11.1, %CppRangeForIterate.WithSelf.Begin.call
// CHECK:STDOUT:     %r.ref.loc45: ref @Test.%R.as_type.loc42_113.1 (%R.as_type) = name_ref r, %r
// CHECK:STDOUT:     %End.ref: %CppRangeForIterate.assoc_type = name_ref End, imports.%Core.import_ref.83c [concrete = constants.%assoc3]
// CHECK:STDOUT:     %impl.elem3.loc45_4.1: @Test.%.loc45_4 (%.cab) = impl_witness_access constants.%CppRangeForIterate.lookup_impl_witness.366fea.2, element3 [symbolic = %impl.elem3.loc45_4.2 (constants.%impl.elem3)]
// CHECK:STDOUT:     %bound_method.loc45_4: <bound method> = bound_method %r.ref.loc45, %impl.elem3.loc45_4.1
// CHECK:STDOUT:     %specific_impl_fn.loc45_4.1: <specific function> = specific_impl_function %impl.elem3.loc45_4.1, @CppRangeForIterate.WithSelf.End(constants.%CppRangeForIterate.facet.527) [symbolic = %specific_impl_fn.loc45_4.2 (constants.%specific_impl_fn.cc4)]
// CHECK:STDOUT:     %bound_method.loc45_9.1: <bound method> = bound_method %r.ref.loc45, %specific_impl_fn.loc45_4.1
// CHECK:STDOUT:     %.loc45_9.1: ref @Test.%impl.elem1.1 (%impl.elem1.3ad37d.2) = temporary_storage
// CHECK:STDOUT:     %CppRangeForIterate.WithSelf.End.call: init @Test.%impl.elem1.1 (%impl.elem1.3ad37d.2) to %.loc45_9.1 = call %bound_method.loc45_9.1(%r.ref.loc45)
// CHECK:STDOUT:     %.loc45_9.2: ref @Test.%impl.elem1.1 (%impl.elem1.3ad37d.2) = temporary %.loc45_9.1, %CppRangeForIterate.WithSelf.End.call
// CHECK:STDOUT:     %impl.elem0.loc45_9.1: @Test.%.loc45_9.3 (%.b17) = impl_witness_access constants.%Destroy.lookup_impl_witness.4ab, element0 [symbolic = %impl.elem0.loc45_9.2 (constants.%impl.elem0.449)]
// CHECK:STDOUT:     %bound_method.loc45_9.2: <bound method> = bound_method %.loc45_9.2, %impl.elem0.loc45_9.1
// CHECK:STDOUT:     %specific_impl_fn.loc45_9.1: <specific function> = specific_impl_function %impl.elem0.loc45_9.1, @Destroy.WithSelf.Op(constants.%Destroy.facet.553) [symbolic = %specific_impl_fn.loc45_9.2 (constants.%specific_impl_fn.03f)]
// CHECK:STDOUT:     %bound_method.loc45_9.3: <bound method> = bound_method %.loc45_9.2, %specific_impl_fn.loc45_9.1
// CHECK:STDOUT:     %Destroy.WithSelf.Op.call.loc45: init %empty_tuple.type = call %bound_method.loc45_9.3(%.loc45_9.2)
// CHECK:STDOUT:     %impl.elem0.loc44_11.1: @Test.%.loc44_11.3 (%.d5a) = impl_witness_access constants.%Destroy.lookup_impl_witness.e97, element0 [symbolic = %impl.elem0.loc44_11.2 (constants.%impl.elem0.d1d)]
// CHECK:STDOUT:     %bound_method.loc44_11.2: <bound method> = bound_method %.loc44_11.2, %impl.elem0.loc44_11.1
// CHECK:STDOUT:     %specific_impl_fn.loc44_11.1: <specific function> = specific_impl_function %impl.elem0.loc44_11.1, @Destroy.WithSelf.Op(constants.%Destroy.facet.503) [symbolic = %specific_impl_fn.loc44_11.2 (constants.%specific_impl_fn.a5c)]
// CHECK:STDOUT:     %bound_method.loc44_11.3: <bound method> = bound_method %.loc44_11.2, %specific_impl_fn.loc44_11.1
// CHECK:STDOUT:     %Destroy.WithSelf.Op.call.loc44: init %empty_tuple.type = call %bound_method.loc44_11.3(%.loc44_11.2)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(constants.%R) {
// CHECK:STDOUT:   %R.loc42_10.1 => constants.%R
// CHECK:STDOUT:   %R.as_type.loc42_113.1 => constants.%R.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.ff859f.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(constants.%facet_value.b34) {
// CHECK:STDOUT:   %R.loc42_10.1 => constants.%facet_value.b34
// CHECK:STDOUT:   %R.as_type.loc42_113.1 => constants.%MutableRange
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.f4c
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc42 => constants.%complete_type
// CHECK:STDOUT:   %CppRangeForIterate.lookup_impl_witness => constants.%custom_witness.1bf5e5.1
// CHECK:STDOUT:   %CppRangeForIterate.facet => constants.%CppRangeForIterate.facet.d1e
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.Begin.type => constants.%CppRangeForIterate.WithSelf.Begin.type.bf8
// CHECK:STDOUT:   %.loc44_4 => constants.%.504
// CHECK:STDOUT:   %impl.elem2.loc44_4.2 => constants.%Begin.1544bb.1
// CHECK:STDOUT:   %specific_impl_fn.loc44_4.2 => constants.%Begin.1544bb.1
// CHECK:STDOUT:   %impl.elem0.1 => constants.%Iterator.b6bc8f.1
// CHECK:STDOUT:   %require_complete.1 => constants.%complete_type
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.End.type => constants.%CppRangeForIterate.WithSelf.End.type.580
// CHECK:STDOUT:   %.loc45_4 => constants.%.1e1
// CHECK:STDOUT:   %impl.elem3.loc45_4.2 => constants.%End.be76e1.1
// CHECK:STDOUT:   %specific_impl_fn.loc45_4.2 => constants.%End.be76e1.1
// CHECK:STDOUT:   %impl.elem1.1 => constants.%Sentinel.50a788.1
// CHECK:STDOUT:   %require_complete.2 => constants.%complete_type
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.loc45 => constants.%custom_witness.415faa.2
// CHECK:STDOUT:   %Destroy.facet.loc45 => constants.%Destroy.facet.eb0
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.loc45 => constants.%Destroy.WithSelf.Op.type.552
// CHECK:STDOUT:   %.loc45_9.3 => constants.%.9ca
// CHECK:STDOUT:   %impl.elem0.loc45_9.2 => constants.%Sentinel.Op.26b
// CHECK:STDOUT:   %specific_impl_fn.loc45_9.2 => constants.%Sentinel.Op.26b
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.loc44 => constants.%custom_witness.415faa.1
// CHECK:STDOUT:   %Destroy.facet.loc44 => constants.%Destroy.facet.bcd
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.loc44 => constants.%Destroy.WithSelf.Op.type.92e
// CHECK:STDOUT:   %.loc44_11.3 => constants.%.c1a
// CHECK:STDOUT:   %impl.elem0.loc44_11.2 => constants.%Iterator.Op.26b
// CHECK:STDOUT:   %specific_impl_fn.loc44_11.2 => constants.%Iterator.Op.26b
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(constants.%facet_value.11c) {
// CHECK:STDOUT:   %R.loc42_10.1 => constants.%facet_value.11c
// CHECK:STDOUT:   %R.as_type.loc42_113.1 => constants.%ConstRange
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.e01
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc42 => constants.%complete_type
// CHECK:STDOUT:   %CppRangeForIterate.lookup_impl_witness => constants.%custom_witness.1bf5e5.2
// CHECK:STDOUT:   %CppRangeForIterate.facet => constants.%CppRangeForIterate.facet.18f
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.Begin.type => constants.%CppRangeForIterate.WithSelf.Begin.type.a30
// CHECK:STDOUT:   %.loc44_4 => constants.%.e4b
// CHECK:STDOUT:   %impl.elem2.loc44_4.2 => constants.%Begin.1544bb.2
// CHECK:STDOUT:   %specific_impl_fn.loc44_4.2 => constants.%Begin.1544bb.2
// CHECK:STDOUT:   %impl.elem0.1 => constants.%Iterator.b6bc8f.2
// CHECK:STDOUT:   %require_complete.1 => constants.%complete_type
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.End.type => constants.%CppRangeForIterate.WithSelf.End.type.aba
// CHECK:STDOUT:   %.loc45_4 => constants.%.d10
// CHECK:STDOUT:   %impl.elem3.loc45_4.2 => constants.%End.be76e1.2
// CHECK:STDOUT:   %specific_impl_fn.loc45_4.2 => constants.%End.be76e1.2
// CHECK:STDOUT:   %impl.elem1.1 => constants.%Sentinel.50a788.2
// CHECK:STDOUT:   %require_complete.2 => constants.%complete_type
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.loc45 => constants.%custom_witness.0d3cfa.2
// CHECK:STDOUT:   %Destroy.facet.loc45 => constants.%Destroy.facet.a2a
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.loc45 => constants.%Destroy.WithSelf.Op.type.1fe
// CHECK:STDOUT:   %.loc45_9.3 => constants.%.f48
// CHECK:STDOUT:   %impl.elem0.loc45_9.2 => constants.%Sentinel.Op.84b
// CHECK:STDOUT:   %specific_impl_fn.loc45_9.2 => constants.%Sentinel.Op.84b
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.loc44 => constants.%custom_witness.0d3cfa.1
// CHECK:STDOUT:   %Destroy.facet.loc44 => constants.%Destroy.facet.e71
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.loc44 => constants.%Destroy.WithSelf.Op.type.1d5
// CHECK:STDOUT:   %.loc44_11.3 => constants.%.bef
// CHECK:STDOUT:   %impl.elem0.loc44_11.2 => constants.%Iterator.Op.84b
// CHECK:STDOUT:   %specific_impl_fn.loc44_11.2 => constants.%Iterator.Op.84b
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- adl_overloads.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %CppRangeForIterate.type: type = facet_type <@CppRangeForIterate> [concrete]
// CHECK:STDOUT:   %Self.668: %CppRangeForIterate.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.End.type.45d9b7.1: type = fn_type @CppRangeForIterate.WithSelf.End, @CppRangeForIterate.WithSelf(%Self.668) [symbolic]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.End.e1b918.1: %CppRangeForIterate.WithSelf.End.type.45d9b7.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.Begin.type.e09a97.1: type = fn_type @CppRangeForIterate.WithSelf.Begin, @CppRangeForIterate.WithSelf(%Self.668) [symbolic]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.Begin.4c9a66.1: %CppRangeForIterate.WithSelf.Begin.type.e09a97.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %.Self.fda: %CppRangeForIterate.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %CppRangeForIterate.assoc_type: type = assoc_entity_type @CppRangeForIterate [concrete]
// CHECK:STDOUT:   %CppRangeForIterate.lookup_impl_witness.ae8: <witness> = lookup_impl_witness %.Self.fda, @CppRangeForIterate [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.354: type = impl_witness_access %CppRangeForIterate.lookup_impl_witness.ae8, element0 [symbolic_self]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %impl.elem1.fdc: type = impl_witness_access %CppRangeForIterate.lookup_impl_witness.ae8, element1 [symbolic_self]
// CHECK:STDOUT:   %CppRangeForIterate_where.type: type = facet_type <@CppRangeForIterate where %impl.elem0.354 impls @Destroy and %impl.elem1.fdc impls @Destroy> [concrete]
// CHECK:STDOUT:   %R: %CppRangeForIterate_where.type = symbolic_binding R, 0 [symbolic]
// CHECK:STDOUT:   %R.as_type: type = facet_access_type %R [symbolic]
// CHECK:STDOUT:   %pattern_type.ff859f.2: type = pattern_type %R.as_type [symbolic]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %assoc2: %CppRangeForIterate.assoc_type = assoc_entity element2, imports.%Core.import_ref.dad [concrete]
// CHECK:STDOUT:   %CppRangeForIterate.lookup_impl_witness.366fea.2: <witness> = lookup_impl_witness %R, @CppRangeForIterate [symbolic]
// CHECK:STDOUT:   %CppRangeForIterate.facet.527: %CppRangeForIterate.type = facet_value %R.as_type, (%CppRangeForIterate.lookup_impl_witness.366fea.2) [symbolic]
// CHECK:STDOUT:   %impl.elem0.b36147.2: type = impl_witness_access %CppRangeForIterate.lookup_impl_witness.366fea.2, element0 [symbolic]
// CHECK:STDOUT:   %impl.elem1.3ad37d.2: type = impl_witness_access %CppRangeForIterate.lookup_impl_witness.366fea.2, element1 [symbolic]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.Begin.type.aa6: type = fn_type @CppRangeForIterate.WithSelf.Begin, @CppRangeForIterate.WithSelf(%CppRangeForIterate.facet.527) [symbolic]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.End.type.ebe: type = fn_type @CppRangeForIterate.WithSelf.End, @CppRangeForIterate.WithSelf(%CppRangeForIterate.facet.527) [symbolic]
// CHECK:STDOUT:   %.227: type = fn_type_with_self_type %CppRangeForIterate.WithSelf.Begin.type.aa6, %CppRangeForIterate.facet.527 [symbolic]
// CHECK:STDOUT:   %impl.elem2: %.227 = impl_witness_access %CppRangeForIterate.lookup_impl_witness.366fea.2, element2 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.4a0: <specific function> = specific_impl_function %impl.elem2, @CppRangeForIterate.WithSelf.Begin(%CppRangeForIterate.facet.527) [symbolic]
// CHECK:STDOUT:   %assoc3: %CppRangeForIterate.assoc_type = assoc_entity element3, imports.%Core.import_ref.f0a [concrete]
// CHECK:STDOUT:   %.cab: type = fn_type_with_self_type %CppRangeForIterate.WithSelf.End.type.ebe, %CppRangeForIterate.facet.527 [symbolic]
// CHECK:STDOUT:   %impl.elem3: %.cab = impl_witness_access %CppRangeForIterate.lookup_impl_witness.366fea.2, element3 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.cc4: <specific function> = specific_impl_function %impl.elem3, @CppRangeForIterate.WithSelf.End(%CppRangeForIterate.facet.527) [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.4ab: <witness> = lookup_impl_witness %impl.elem1.3ad37d.2, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.553: %Destroy.type = facet_value %impl.elem1.3ad37d.2, (%Destroy.lookup_impl_witness.4ab) [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.b61: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.553) [symbolic]
// CHECK:STDOUT:   %.b17: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.b61, %Destroy.facet.553 [symbolic]
// CHECK:STDOUT:   %impl.elem0.449: %.b17 = impl_witness_access %Destroy.lookup_impl_witness.4ab, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.03f: <specific function> = specific_impl_function %impl.elem0.449, @Destroy.WithSelf.Op(%Destroy.facet.553) [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.e97: <witness> = lookup_impl_witness %impl.elem0.b36147.2, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.503: %Destroy.type = facet_value %impl.elem0.b36147.2, (%Destroy.lookup_impl_witness.e97) [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.4c8: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.503) [symbolic]
// CHECK:STDOUT:   %.d5a: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.4c8, %Destroy.facet.503 [symbolic]
// CHECK:STDOUT:   %impl.elem0.d1d: %.d5a = impl_witness_access %Destroy.lookup_impl_witness.e97, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.a5c: <specific function> = specific_impl_function %impl.elem0.d1d, @Destroy.WithSelf.Op(%Destroy.facet.503) [symbolic]
// CHECK:STDOUT:   %MutableAndConstRange: type = class_type @MutableAndConstRange [concrete]
// CHECK:STDOUT:   %pattern_type.d8b: type = pattern_type %MutableAndConstRange [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %Iterator: type = class_type @Iterator.1 [concrete]
// CHECK:STDOUT:   %Begin.type: type = fn_type @Begin [concrete]
// CHECK:STDOUT:   %Begin: %Begin.type = struct_value () [concrete]
// CHECK:STDOUT:   %End.type: type = fn_type @End [concrete]
// CHECK:STDOUT:   %End: %End.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.827: <witness> = custom_witness (%Iterator, %Iterator, %Begin, %End), @CppRangeForIterate [concrete]
// CHECK:STDOUT:   %CppRangeForIterate.facet.98b: %CppRangeForIterate.type = facet_value %MutableAndConstRange, (%custom_witness.827) [concrete]
// CHECK:STDOUT:   %Iterator.Op.type: type = fn_type @Iterator.Op [concrete]
// CHECK:STDOUT:   %Iterator.Op: %Iterator.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.2d3: <witness> = custom_witness (%Iterator.Op), @Destroy [concrete]
// CHECK:STDOUT:   %facet_value: %CppRangeForIterate_where.type = facet_value %MutableAndConstRange, (%custom_witness.827, %custom_witness.2d3) [concrete]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.Begin.type.706: type = fn_type @CppRangeForIterate.WithSelf.Begin, @CppRangeForIterate.WithSelf(%CppRangeForIterate.facet.98b) [concrete]
// CHECK:STDOUT:   %.ff3: type = fn_type_with_self_type %CppRangeForIterate.WithSelf.Begin.type.706, %CppRangeForIterate.facet.98b [concrete]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.End.type.5f8: type = fn_type @CppRangeForIterate.WithSelf.End, @CppRangeForIterate.WithSelf(%CppRangeForIterate.facet.98b) [concrete]
// CHECK:STDOUT:   %.c8e: type = fn_type_with_self_type %CppRangeForIterate.WithSelf.End.type.5f8, %CppRangeForIterate.facet.98b [concrete]
// CHECK:STDOUT:   %Destroy.facet.611: %Destroy.type = facet_value %Iterator, (%custom_witness.2d3) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.67f: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.611) [concrete]
// CHECK:STDOUT:   %.f62: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.67f, %Destroy.facet.611 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.4af: %CppRangeForIterate.assoc_type = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [concrete = constants.%assoc2]
// CHECK:STDOUT:   %Core.import_ref.83c: %CppRangeForIterate.assoc_type = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [concrete = constants.%assoc3]
// CHECK:STDOUT:   %Core.import_ref.dad: @CppRangeForIterate.WithSelf.%CppRangeForIterate.WithSelf.Begin.type (%CppRangeForIterate.WithSelf.Begin.type.e09a97.1) = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @CppRangeForIterate.WithSelf.%CppRangeForIterate.WithSelf.Begin (constants.%CppRangeForIterate.WithSelf.Begin.4c9a66.1)]
// CHECK:STDOUT:   %Core.import_ref.f0a: @CppRangeForIterate.WithSelf.%CppRangeForIterate.WithSelf.End.type (%CppRangeForIterate.WithSelf.End.type.45d9b7.1) = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @CppRangeForIterate.WithSelf.%CppRangeForIterate.WithSelf.End (constants.%CppRangeForIterate.WithSelf.End.e1b918.1)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Test(%R.loc32_10.2: %CppRangeForIterate_where.type) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %CppRangeForIterate.lookup_impl_witness: <witness> = lookup_impl_witness %R.loc32_10.1, @CppRangeForIterate [symbolic = %CppRangeForIterate.lookup_impl_witness (constants.%CppRangeForIterate.lookup_impl_witness.366fea.2)]
// CHECK:STDOUT:   %CppRangeForIterate.facet: %CppRangeForIterate.type = facet_value %R.as_type.loc32_113.1, (%CppRangeForIterate.lookup_impl_witness) [symbolic = %CppRangeForIterate.facet (constants.%CppRangeForIterate.facet.527)]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.Begin.type: type = fn_type @CppRangeForIterate.WithSelf.Begin, @CppRangeForIterate.WithSelf(%CppRangeForIterate.facet) [symbolic = %CppRangeForIterate.WithSelf.Begin.type (constants.%CppRangeForIterate.WithSelf.Begin.type.aa6)]
// CHECK:STDOUT:   %.loc34_4: type = fn_type_with_self_type %CppRangeForIterate.WithSelf.Begin.type, %CppRangeForIterate.facet [symbolic = %.loc34_4 (constants.%.227)]
// CHECK:STDOUT:   %impl.elem2.loc34_4.2: @Test.%.loc34_4 (%.227) = impl_witness_access %CppRangeForIterate.lookup_impl_witness, element2 [symbolic = %impl.elem2.loc34_4.2 (constants.%impl.elem2)]
// CHECK:STDOUT:   %specific_impl_fn.loc34_4.2: <specific function> = specific_impl_function %impl.elem2.loc34_4.2, @CppRangeForIterate.WithSelf.Begin(%CppRangeForIterate.facet) [symbolic = %specific_impl_fn.loc34_4.2 (constants.%specific_impl_fn.4a0)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.End.type: type = fn_type @CppRangeForIterate.WithSelf.End, @CppRangeForIterate.WithSelf(%CppRangeForIterate.facet) [symbolic = %CppRangeForIterate.WithSelf.End.type (constants.%CppRangeForIterate.WithSelf.End.type.ebe)]
// CHECK:STDOUT:   %.loc35_4: type = fn_type_with_self_type %CppRangeForIterate.WithSelf.End.type, %CppRangeForIterate.facet [symbolic = %.loc35_4 (constants.%.cab)]
// CHECK:STDOUT:   %impl.elem3.loc35_4.2: @Test.%.loc35_4 (%.cab) = impl_witness_access %CppRangeForIterate.lookup_impl_witness, element3 [symbolic = %impl.elem3.loc35_4.2 (constants.%impl.elem3)]
// CHECK:STDOUT:   %specific_impl_fn.loc35_4.2: <specific function> = specific_impl_function %impl.elem3.loc35_4.2, @CppRangeForIterate.WithSelf.End(%CppRangeForIterate.facet) [symbolic = %specific_impl_fn.loc35_4.2 (constants.%specific_impl_fn.cc4)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.loc35: <witness> = lookup_impl_witness %impl.elem1.1, @Destroy [symbolic = %Destroy.lookup_impl_witness.loc35 (constants.%Destroy.lookup_impl_witness.4ab)]
// CHECK:STDOUT:   %Destroy.facet.loc35: %Destroy.type = facet_value %impl.elem1.1, (%Destroy.lookup_impl_witness.loc35) [symbolic = %Destroy.facet.loc35 (constants.%Destroy.facet.553)]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.loc35: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.loc35) [symbolic = %Destroy.WithSelf.Op.type.loc35 (constants.%Destroy.WithSelf.Op.type.b61)]
// CHECK:STDOUT:   %.loc35_9.3: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.loc35, %Destroy.facet.loc35 [symbolic = %.loc35_9.3 (constants.%.b17)]
// CHECK:STDOUT:   %impl.elem0.loc35_9.2: @Test.%.loc35_9.3 (%.b17) = impl_witness_access %Destroy.lookup_impl_witness.loc35, element0 [symbolic = %impl.elem0.loc35_9.2 (constants.%impl.elem0.449)]
// CHECK:STDOUT:   %specific_impl_fn.loc35_9.2: <specific function> = specific_impl_function %impl.elem0.loc35_9.2, @Destroy.WithSelf.Op(%Destroy.facet.loc35) [symbolic = %specific_impl_fn.loc35_9.2 (constants.%specific_impl_fn.03f)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.loc34: <witness> = lookup_impl_witness %impl.elem0.1, @Destroy [symbolic = %Destroy.lookup_impl_witness.loc34 (constants.%Destroy.lookup_impl_witness.e97)]
// CHECK:STDOUT:   %Destroy.facet.loc34: %Destroy.type = facet_value %impl.elem0.1, (%Destroy.lookup_impl_witness.loc34) [symbolic = %Destroy.facet.loc34 (constants.%Destroy.facet.503)]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.loc34: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.loc34) [symbolic = %Destroy.WithSelf.Op.type.loc34 (constants.%Destroy.WithSelf.Op.type.4c8)]
// CHECK:STDOUT:   %.loc34_11.3: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.loc34, %Destroy.facet.loc34 [symbolic = %.loc34_11.3 (constants.%.d5a)]
// CHECK:STDOUT:   %impl.elem0.loc34_11.2: @Test.%.loc34_11.3 (%.d5a) = impl_witness_access %Destroy.lookup_impl_witness.loc34, element0 [symbolic = %impl.elem0.loc34_11.2 (constants.%impl.elem0.d1d)]
// CHECK:STDOUT:   %specific_impl_fn.loc34_11.2: <specific function> = specific_impl_function %impl.elem0.loc34_11.2, @Destroy.WithSelf.Op(%Destroy.facet.loc34) [symbolic = %specific_impl_fn.loc34_11.2 (constants.%specific_impl_fn.a5c)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%r.param: ref @Test.%R.as_type.loc32_113.1 (%R.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %r.ref.loc34: ref @Test.%R.as_type.loc32_113.1 (%R.as_type) = name_ref r, %r
// CHECK:STDOUT:     %Begin.ref: %CppRangeForIterate.assoc_type = name_ref Begin, imports.%Core.import_ref.4af [concrete = constants.%assoc2]
// CHECK:STDOUT:     %impl.elem2.loc34_4.1: @Test.%.loc34_4 (%.227) = impl_witness_access constants.%CppRangeForIterate.lookup_impl_witness.366fea.2, element2 [symbolic = %impl.elem2.loc34_4.2 (constants.%impl.elem2)]
// CHECK:STDOUT:     %bound_method.loc34_4: <bound method> = bound_method %r.ref.loc34, %impl.elem2.loc34_4.1
// CHECK:STDOUT:     %specific_impl_fn.loc34_4.1: <specific function> = specific_impl_function %impl.elem2.loc34_4.1, @CppRangeForIterate.WithSelf.Begin(constants.%CppRangeForIterate.facet.527) [symbolic = %specific_impl_fn.loc34_4.2 (constants.%specific_impl_fn.4a0)]
// CHECK:STDOUT:     %bound_method.loc34_11.1: <bound method> = bound_method %r.ref.loc34, %specific_impl_fn.loc34_4.1
// CHECK:STDOUT:     %.loc34_11.1: ref @Test.%impl.elem0.1 (%impl.elem0.b36147.2) = temporary_storage
// CHECK:STDOUT:     %CppRangeForIterate.WithSelf.Begin.call: init @Test.%impl.elem0.1 (%impl.elem0.b36147.2) to %.loc34_11.1 = call %bound_method.loc34_11.1(%r.ref.loc34)
// CHECK:STDOUT:     %.loc34_11.2: ref @Test.%impl.elem0.1 (%impl.elem0.b36147.2) = temporary %.loc34_11.1, %CppRangeForIterate.WithSelf.Begin.call
// CHECK:STDOUT:     %r.ref.loc35: ref @Test.%R.as_type.loc32_113.1 (%R.as_type) = name_ref r, %r
// CHECK:STDOUT:     %End.ref: %CppRangeForIterate.assoc_type = name_ref End, imports.%Core.import_ref.83c [concrete = constants.%assoc3]
// CHECK:STDOUT:     %impl.elem3.loc35_4.1: @Test.%.loc35_4 (%.cab) = impl_witness_access constants.%CppRangeForIterate.lookup_impl_witness.366fea.2, element3 [symbolic = %impl.elem3.loc35_4.2 (constants.%impl.elem3)]
// CHECK:STDOUT:     %bound_method.loc35_4: <bound method> = bound_method %r.ref.loc35, %impl.elem3.loc35_4.1
// CHECK:STDOUT:     %specific_impl_fn.loc35_4.1: <specific function> = specific_impl_function %impl.elem3.loc35_4.1, @CppRangeForIterate.WithSelf.End(constants.%CppRangeForIterate.facet.527) [symbolic = %specific_impl_fn.loc35_4.2 (constants.%specific_impl_fn.cc4)]
// CHECK:STDOUT:     %bound_method.loc35_9.1: <bound method> = bound_method %r.ref.loc35, %specific_impl_fn.loc35_4.1
// CHECK:STDOUT:     %.loc35_9.1: ref @Test.%impl.elem1.1 (%impl.elem1.3ad37d.2) = temporary_storage
// CHECK:STDOUT:     %CppRangeForIterate.WithSelf.End.call: init @Test.%impl.elem1.1 (%impl.elem1.3ad37d.2) to %.loc35_9.1 = call %bound_method.loc35_9.1(%r.ref.loc35)
// CHECK:STDOUT:     %.loc35_9.2: ref @Test.%impl.elem1.1 (%impl.elem1.3ad37d.2) = temporary %.loc35_9.1, %CppRangeForIterate.WithSelf.End.call
// CHECK:STDOUT:     %impl.elem0.loc35_9.1: @Test.%.loc35_9.3 (%.b17) = impl_witness_access constants.%Destroy.lookup_impl_witness.4ab, element0 [symbolic = %impl.elem0.loc35_9.2 (constants.%impl.elem0.449)]
// CHECK:STDOUT:     %bound_method.loc35_9.2: <bound method> = bound_method %.loc35_9.2, %impl.elem0.loc35_9.1
// CHECK:STDOUT:     %specific_impl_fn.loc35_9.1: <specific function> = specific_impl_function %impl.elem0.loc35_9.1, @Destroy.WithSelf.Op(constants.%Destroy.facet.553) [symbolic = %specific_impl_fn.loc35_9.2 (constants.%specific_impl_fn.03f)]
// CHECK:STDOUT:     %bound_method.loc35_9.3: <bound method> = bound_method %.loc35_9.2, %specific_impl_fn.loc35_9.1
// CHECK:STDOUT:     %Destroy.WithSelf.Op.call.loc35: init %empty_tuple.type = call %bound_method.loc35_9.3(%.loc35_9.2)
// CHECK:STDOUT:     %impl.elem0.loc34_11.1: @Test.%.loc34_11.3 (%.d5a) = impl_witness_access constants.%Destroy.lookup_impl_witness.e97, element0 [symbolic = %impl.elem0.loc34_11.2 (constants.%impl.elem0.d1d)]
// CHECK:STDOUT:     %bound_method.loc34_11.2: <bound method> = bound_method %.loc34_11.2, %impl.elem0.loc34_11.1
// CHECK:STDOUT:     %specific_impl_fn.loc34_11.1: <specific function> = specific_impl_function %impl.elem0.loc34_11.1, @Destroy.WithSelf.Op(constants.%Destroy.facet.503) [symbolic = %specific_impl_fn.loc34_11.2 (constants.%specific_impl_fn.a5c)]
// CHECK:STDOUT:     %bound_method.loc34_11.3: <bound method> = bound_method %.loc34_11.2, %specific_impl_fn.loc34_11.1
// CHECK:STDOUT:     %Destroy.WithSelf.Op.call.loc34: init %empty_tuple.type = call %bound_method.loc34_11.3(%.loc34_11.2)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(constants.%R) {
// CHECK:STDOUT:   %R.loc32_10.1 => constants.%R
// CHECK:STDOUT:   %R.as_type.loc32_113.1 => constants.%R.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.ff859f.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(constants.%facet_value) {
// CHECK:STDOUT:   %R.loc32_10.1 => constants.%facet_value
// CHECK:STDOUT:   %R.as_type.loc32_113.1 => constants.%MutableAndConstRange
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.d8b
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc32 => constants.%complete_type
// CHECK:STDOUT:   %CppRangeForIterate.lookup_impl_witness => constants.%custom_witness.827
// CHECK:STDOUT:   %CppRangeForIterate.facet => constants.%CppRangeForIterate.facet.98b
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.Begin.type => constants.%CppRangeForIterate.WithSelf.Begin.type.706
// CHECK:STDOUT:   %.loc34_4 => constants.%.ff3
// CHECK:STDOUT:   %impl.elem2.loc34_4.2 => constants.%Begin
// CHECK:STDOUT:   %specific_impl_fn.loc34_4.2 => constants.%Begin
// CHECK:STDOUT:   %impl.elem0.1 => constants.%Iterator
// CHECK:STDOUT:   %require_complete.1 => constants.%complete_type
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.End.type => constants.%CppRangeForIterate.WithSelf.End.type.5f8
// CHECK:STDOUT:   %.loc35_4 => constants.%.c8e
// CHECK:STDOUT:   %impl.elem3.loc35_4.2 => constants.%End
// CHECK:STDOUT:   %specific_impl_fn.loc35_4.2 => constants.%End
// CHECK:STDOUT:   %impl.elem1.1 => constants.%Iterator
// CHECK:STDOUT:   %require_complete.2 => constants.%complete_type
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.loc35 => constants.%custom_witness.2d3
// CHECK:STDOUT:   %Destroy.facet.loc35 => constants.%Destroy.facet.611
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.loc35 => constants.%Destroy.WithSelf.Op.type.67f
// CHECK:STDOUT:   %.loc35_9.3 => constants.%.f62
// CHECK:STDOUT:   %impl.elem0.loc35_9.2 => constants.%Iterator.Op
// CHECK:STDOUT:   %specific_impl_fn.loc35_9.2 => constants.%Iterator.Op
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.loc34 => constants.%custom_witness.2d3
// CHECK:STDOUT:   %Destroy.facet.loc34 => constants.%Destroy.facet.611
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.loc34 => constants.%Destroy.WithSelf.Op.type.67f
// CHECK:STDOUT:   %.loc34_11.3 => constants.%.f62
// CHECK:STDOUT:   %impl.elem0.loc34_11.2 => constants.%Iterator.Op
// CHECK:STDOUT:   %specific_impl_fn.loc34_11.2 => constants.%Iterator.Op
// CHECK:STDOUT: }
// CHECK:STDOUT:
