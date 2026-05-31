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
// CHECK:STDOUT:   %pattern_type.b76: type = pattern_type %MutableRange [concrete]
// CHECK:STDOUT:   %ConstRange: type = class_type @ConstRange [concrete]
// CHECK:STDOUT:   %pattern_type.631: type = pattern_type %ConstRange [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %Iterator.826: type = class_type @Iterator.1 [concrete]
// CHECK:STDOUT:   %MutableRange.Begin.type: type = fn_type @MutableRange.Begin [concrete]
// CHECK:STDOUT:   %MutableRange.Begin: %MutableRange.Begin.type = struct_value () [concrete]
// CHECK:STDOUT:   %MutableRange.End.type: type = fn_type @MutableRange.End [concrete]
// CHECK:STDOUT:   %MutableRange.End: %MutableRange.End.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.ea2: <witness> = custom_witness (%Iterator.826, %Iterator.826, %MutableRange.Begin, %MutableRange.End), @CppRangeForIterate [concrete]
// CHECK:STDOUT:   %CppRangeForIterate.facet.9d3: %CppRangeForIterate.type = facet_value %MutableRange, (%custom_witness.ea2) [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.134: type = fn_type @Iterator.Op.1 [concrete]
// CHECK:STDOUT:   %Iterator.Op.f05: %Iterator.Op.type.134 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.a97: <witness> = custom_witness (%Iterator.Op.f05), @Destroy [concrete]
// CHECK:STDOUT:   %facet_value.a96: %CppRangeForIterate_where.type = facet_value %MutableRange, (%custom_witness.ea2, %custom_witness.a97) [concrete]
// CHECK:STDOUT:   %Iterator.4f0: type = class_type @Iterator.2 [concrete]
// CHECK:STDOUT:   %ConstRange.Begin.type: type = fn_type @ConstRange.Begin [concrete]
// CHECK:STDOUT:   %ConstRange.Begin: %ConstRange.Begin.type = struct_value () [concrete]
// CHECK:STDOUT:   %ConstRange.End.type: type = fn_type @ConstRange.End [concrete]
// CHECK:STDOUT:   %ConstRange.End: %ConstRange.End.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.e4e: <witness> = custom_witness (%Iterator.4f0, %Iterator.4f0, %ConstRange.Begin, %ConstRange.End), @CppRangeForIterate [concrete]
// CHECK:STDOUT:   %CppRangeForIterate.facet.0b1: %CppRangeForIterate.type = facet_value %ConstRange, (%custom_witness.e4e) [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.9f2: type = fn_type @Iterator.Op.2 [concrete]
// CHECK:STDOUT:   %Iterator.Op.7b7: %Iterator.Op.type.9f2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.569: <witness> = custom_witness (%Iterator.Op.7b7), @Destroy [concrete]
// CHECK:STDOUT:   %facet_value.bc0: %CppRangeForIterate_where.type = facet_value %ConstRange, (%custom_witness.e4e, %custom_witness.569) [concrete]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.Begin.type.5c8: type = fn_type @CppRangeForIterate.WithSelf.Begin, @CppRangeForIterate.WithSelf(%CppRangeForIterate.facet.9d3) [concrete]
// CHECK:STDOUT:   %.a3d: type = fn_type_with_self_type %CppRangeForIterate.WithSelf.Begin.type.5c8, %CppRangeForIterate.facet.9d3 [concrete]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.End.type.a6b: type = fn_type @CppRangeForIterate.WithSelf.End, @CppRangeForIterate.WithSelf(%CppRangeForIterate.facet.9d3) [concrete]
// CHECK:STDOUT:   %.cf9: type = fn_type_with_self_type %CppRangeForIterate.WithSelf.End.type.a6b, %CppRangeForIterate.facet.9d3 [concrete]
// CHECK:STDOUT:   %Destroy.facet.41d: %Destroy.type = facet_value %Iterator.826, (%custom_witness.a97) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.323: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.41d) [concrete]
// CHECK:STDOUT:   %.458: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.323, %Destroy.facet.41d [concrete]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.Begin.type.9a2: type = fn_type @CppRangeForIterate.WithSelf.Begin, @CppRangeForIterate.WithSelf(%CppRangeForIterate.facet.0b1) [concrete]
// CHECK:STDOUT:   %.ce0: type = fn_type_with_self_type %CppRangeForIterate.WithSelf.Begin.type.9a2, %CppRangeForIterate.facet.0b1 [concrete]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.End.type.1ba: type = fn_type @CppRangeForIterate.WithSelf.End, @CppRangeForIterate.WithSelf(%CppRangeForIterate.facet.0b1) [concrete]
// CHECK:STDOUT:   %.3fc: type = fn_type_with_self_type %CppRangeForIterate.WithSelf.End.type.1ba, %CppRangeForIterate.facet.0b1 [concrete]
// CHECK:STDOUT:   %Destroy.facet.b0c: %Destroy.type = facet_value %Iterator.4f0, (%custom_witness.569) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.656: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.b0c) [concrete]
// CHECK:STDOUT:   %.80f: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.656, %Destroy.facet.b0c [concrete]
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
// CHECK:STDOUT: specific @Test(constants.%facet_value.a96) {
// CHECK:STDOUT:   %R.loc30_10.1 => constants.%facet_value.a96
// CHECK:STDOUT:   %R.as_type.loc30_113.1 => constants.%MutableRange
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.b76
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc30 => constants.%complete_type
// CHECK:STDOUT:   %CppRangeForIterate.lookup_impl_witness => constants.%custom_witness.ea2
// CHECK:STDOUT:   %CppRangeForIterate.facet => constants.%CppRangeForIterate.facet.9d3
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.Begin.type => constants.%CppRangeForIterate.WithSelf.Begin.type.5c8
// CHECK:STDOUT:   %.loc32_4 => constants.%.a3d
// CHECK:STDOUT:   %impl.elem2.loc32_4.2 => constants.%MutableRange.Begin
// CHECK:STDOUT:   %specific_impl_fn.loc32_4.2 => constants.%MutableRange.Begin
// CHECK:STDOUT:   %impl.elem0.1 => constants.%Iterator.826
// CHECK:STDOUT:   %require_complete.1 => constants.%complete_type
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.End.type => constants.%CppRangeForIterate.WithSelf.End.type.a6b
// CHECK:STDOUT:   %.loc33_4 => constants.%.cf9
// CHECK:STDOUT:   %impl.elem3.loc33_4.2 => constants.%MutableRange.End
// CHECK:STDOUT:   %specific_impl_fn.loc33_4.2 => constants.%MutableRange.End
// CHECK:STDOUT:   %impl.elem1.1 => constants.%Iterator.826
// CHECK:STDOUT:   %require_complete.2 => constants.%complete_type
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.loc33 => constants.%custom_witness.a97
// CHECK:STDOUT:   %Destroy.facet.loc33 => constants.%Destroy.facet.41d
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.loc33 => constants.%Destroy.WithSelf.Op.type.323
// CHECK:STDOUT:   %.loc33_9.3 => constants.%.458
// CHECK:STDOUT:   %impl.elem0.loc33_9.2 => constants.%Iterator.Op.f05
// CHECK:STDOUT:   %specific_impl_fn.loc33_9.2 => constants.%Iterator.Op.f05
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.loc32 => constants.%custom_witness.a97
// CHECK:STDOUT:   %Destroy.facet.loc32 => constants.%Destroy.facet.41d
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.loc32 => constants.%Destroy.WithSelf.Op.type.323
// CHECK:STDOUT:   %.loc32_11.3 => constants.%.458
// CHECK:STDOUT:   %impl.elem0.loc32_11.2 => constants.%Iterator.Op.f05
// CHECK:STDOUT:   %specific_impl_fn.loc32_11.2 => constants.%Iterator.Op.f05
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(constants.%facet_value.bc0) {
// CHECK:STDOUT:   %R.loc30_10.1 => constants.%facet_value.bc0
// CHECK:STDOUT:   %R.as_type.loc30_113.1 => constants.%ConstRange
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.631
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc30 => constants.%complete_type
// CHECK:STDOUT:   %CppRangeForIterate.lookup_impl_witness => constants.%custom_witness.e4e
// CHECK:STDOUT:   %CppRangeForIterate.facet => constants.%CppRangeForIterate.facet.0b1
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.Begin.type => constants.%CppRangeForIterate.WithSelf.Begin.type.9a2
// CHECK:STDOUT:   %.loc32_4 => constants.%.ce0
// CHECK:STDOUT:   %impl.elem2.loc32_4.2 => constants.%ConstRange.Begin
// CHECK:STDOUT:   %specific_impl_fn.loc32_4.2 => constants.%ConstRange.Begin
// CHECK:STDOUT:   %impl.elem0.1 => constants.%Iterator.4f0
// CHECK:STDOUT:   %require_complete.1 => constants.%complete_type
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.End.type => constants.%CppRangeForIterate.WithSelf.End.type.1ba
// CHECK:STDOUT:   %.loc33_4 => constants.%.3fc
// CHECK:STDOUT:   %impl.elem3.loc33_4.2 => constants.%ConstRange.End
// CHECK:STDOUT:   %specific_impl_fn.loc33_4.2 => constants.%ConstRange.End
// CHECK:STDOUT:   %impl.elem1.1 => constants.%Iterator.4f0
// CHECK:STDOUT:   %require_complete.2 => constants.%complete_type
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.loc33 => constants.%custom_witness.569
// CHECK:STDOUT:   %Destroy.facet.loc33 => constants.%Destroy.facet.b0c
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.loc33 => constants.%Destroy.WithSelf.Op.type.656
// CHECK:STDOUT:   %.loc33_9.3 => constants.%.80f
// CHECK:STDOUT:   %impl.elem0.loc33_9.2 => constants.%Iterator.Op.7b7
// CHECK:STDOUT:   %specific_impl_fn.loc33_9.2 => constants.%Iterator.Op.7b7
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.loc32 => constants.%custom_witness.569
// CHECK:STDOUT:   %Destroy.facet.loc32 => constants.%Destroy.facet.b0c
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.loc32 => constants.%Destroy.WithSelf.Op.type.656
// CHECK:STDOUT:   %.loc32_11.3 => constants.%.80f
// CHECK:STDOUT:   %impl.elem0.loc32_11.2 => constants.%Iterator.Op.7b7
// CHECK:STDOUT:   %specific_impl_fn.loc32_11.2 => constants.%Iterator.Op.7b7
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
// CHECK:STDOUT:   %pattern_type.b76: type = pattern_type %MutableRange [concrete]
// CHECK:STDOUT:   %ConstRange: type = class_type @ConstRange [concrete]
// CHECK:STDOUT:   %pattern_type.631: type = pattern_type %ConstRange [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %Iterator.826: type = class_type @Iterator.1 [concrete]
// CHECK:STDOUT:   %Sentinel.6eb: type = class_type @Sentinel.1 [concrete]
// CHECK:STDOUT:   %MutableRange.Begin.type: type = fn_type @MutableRange.Begin [concrete]
// CHECK:STDOUT:   %MutableRange.Begin: %MutableRange.Begin.type = struct_value () [concrete]
// CHECK:STDOUT:   %MutableRange.End.type: type = fn_type @MutableRange.End [concrete]
// CHECK:STDOUT:   %MutableRange.End: %MutableRange.End.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.556: <witness> = custom_witness (%Iterator.826, %Sentinel.6eb, %MutableRange.Begin, %MutableRange.End), @CppRangeForIterate [concrete]
// CHECK:STDOUT:   %CppRangeForIterate.facet.aae: %CppRangeForIterate.type = facet_value %MutableRange, (%custom_witness.556) [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.134: type = fn_type @Iterator.Op.1 [concrete]
// CHECK:STDOUT:   %Iterator.Op.f05: %Iterator.Op.type.134 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.a9753d.1: <witness> = custom_witness (%Iterator.Op.f05), @Destroy [concrete]
// CHECK:STDOUT:   %Sentinel.Op.type.134: type = fn_type @Sentinel.Op.1 [concrete]
// CHECK:STDOUT:   %Sentinel.Op.f05: %Sentinel.Op.type.134 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.a9753d.2: <witness> = custom_witness (%Sentinel.Op.f05), @Destroy [concrete]
// CHECK:STDOUT:   %facet_value.d71: %CppRangeForIterate_where.type = facet_value %MutableRange, (%custom_witness.556, %custom_witness.a9753d.1, %custom_witness.a9753d.2) [concrete]
// CHECK:STDOUT:   %Iterator.4f0: type = class_type @Iterator.2 [concrete]
// CHECK:STDOUT:   %Sentinel.45e: type = class_type @Sentinel.2 [concrete]
// CHECK:STDOUT:   %ConstRange.Begin.type: type = fn_type @ConstRange.Begin [concrete]
// CHECK:STDOUT:   %ConstRange.Begin: %ConstRange.Begin.type = struct_value () [concrete]
// CHECK:STDOUT:   %ConstRange.End.type: type = fn_type @ConstRange.End [concrete]
// CHECK:STDOUT:   %ConstRange.End: %ConstRange.End.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.d7a: <witness> = custom_witness (%Iterator.4f0, %Sentinel.45e, %ConstRange.Begin, %ConstRange.End), @CppRangeForIterate [concrete]
// CHECK:STDOUT:   %CppRangeForIterate.facet.77a: %CppRangeForIterate.type = facet_value %ConstRange, (%custom_witness.d7a) [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.9f2: type = fn_type @Iterator.Op.2 [concrete]
// CHECK:STDOUT:   %Iterator.Op.7b7: %Iterator.Op.type.9f2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.569415.1: <witness> = custom_witness (%Iterator.Op.7b7), @Destroy [concrete]
// CHECK:STDOUT:   %Sentinel.Op.type.9f2: type = fn_type @Sentinel.Op.2 [concrete]
// CHECK:STDOUT:   %Sentinel.Op.7b7: %Sentinel.Op.type.9f2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.569415.2: <witness> = custom_witness (%Sentinel.Op.7b7), @Destroy [concrete]
// CHECK:STDOUT:   %facet_value.2eb: %CppRangeForIterate_where.type = facet_value %ConstRange, (%custom_witness.d7a, %custom_witness.569415.1, %custom_witness.569415.2) [concrete]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.Begin.type.1ae: type = fn_type @CppRangeForIterate.WithSelf.Begin, @CppRangeForIterate.WithSelf(%CppRangeForIterate.facet.aae) [concrete]
// CHECK:STDOUT:   %.f5c: type = fn_type_with_self_type %CppRangeForIterate.WithSelf.Begin.type.1ae, %CppRangeForIterate.facet.aae [concrete]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.End.type.16f: type = fn_type @CppRangeForIterate.WithSelf.End, @CppRangeForIterate.WithSelf(%CppRangeForIterate.facet.aae) [concrete]
// CHECK:STDOUT:   %.f3e: type = fn_type_with_self_type %CppRangeForIterate.WithSelf.End.type.16f, %CppRangeForIterate.facet.aae [concrete]
// CHECK:STDOUT:   %Destroy.facet.ee2: %Destroy.type = facet_value %Sentinel.6eb, (%custom_witness.a9753d.2) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef7: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.ee2) [concrete]
// CHECK:STDOUT:   %.bef: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.ef7, %Destroy.facet.ee2 [concrete]
// CHECK:STDOUT:   %Destroy.facet.41d: %Destroy.type = facet_value %Iterator.826, (%custom_witness.a9753d.1) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.323: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.41d) [concrete]
// CHECK:STDOUT:   %.458: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.323, %Destroy.facet.41d [concrete]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.Begin.type.7ad: type = fn_type @CppRangeForIterate.WithSelf.Begin, @CppRangeForIterate.WithSelf(%CppRangeForIterate.facet.77a) [concrete]
// CHECK:STDOUT:   %.421: type = fn_type_with_self_type %CppRangeForIterate.WithSelf.Begin.type.7ad, %CppRangeForIterate.facet.77a [concrete]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.End.type.79b: type = fn_type @CppRangeForIterate.WithSelf.End, @CppRangeForIterate.WithSelf(%CppRangeForIterate.facet.77a) [concrete]
// CHECK:STDOUT:   %.b91: type = fn_type_with_self_type %CppRangeForIterate.WithSelf.End.type.79b, %CppRangeForIterate.facet.77a [concrete]
// CHECK:STDOUT:   %Destroy.facet.a41: %Destroy.type = facet_value %Sentinel.45e, (%custom_witness.569415.2) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.0c6: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.a41) [concrete]
// CHECK:STDOUT:   %.df3: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.0c6, %Destroy.facet.a41 [concrete]
// CHECK:STDOUT:   %Destroy.facet.b0c: %Destroy.type = facet_value %Iterator.4f0, (%custom_witness.569415.1) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.656: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.b0c) [concrete]
// CHECK:STDOUT:   %.80f: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.656, %Destroy.facet.b0c [concrete]
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
// CHECK:STDOUT: specific @Test(constants.%facet_value.d71) {
// CHECK:STDOUT:   %R.loc40_10.1 => constants.%facet_value.d71
// CHECK:STDOUT:   %R.as_type.loc40_113.1 => constants.%MutableRange
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.b76
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc40 => constants.%complete_type
// CHECK:STDOUT:   %CppRangeForIterate.lookup_impl_witness => constants.%custom_witness.556
// CHECK:STDOUT:   %CppRangeForIterate.facet => constants.%CppRangeForIterate.facet.aae
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.Begin.type => constants.%CppRangeForIterate.WithSelf.Begin.type.1ae
// CHECK:STDOUT:   %.loc42_4 => constants.%.f5c
// CHECK:STDOUT:   %impl.elem2.loc42_4.2 => constants.%MutableRange.Begin
// CHECK:STDOUT:   %specific_impl_fn.loc42_4.2 => constants.%MutableRange.Begin
// CHECK:STDOUT:   %impl.elem0.1 => constants.%Iterator.826
// CHECK:STDOUT:   %require_complete.1 => constants.%complete_type
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.End.type => constants.%CppRangeForIterate.WithSelf.End.type.16f
// CHECK:STDOUT:   %.loc43_4 => constants.%.f3e
// CHECK:STDOUT:   %impl.elem3.loc43_4.2 => constants.%MutableRange.End
// CHECK:STDOUT:   %specific_impl_fn.loc43_4.2 => constants.%MutableRange.End
// CHECK:STDOUT:   %impl.elem1.1 => constants.%Sentinel.6eb
// CHECK:STDOUT:   %require_complete.2 => constants.%complete_type
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.loc43 => constants.%custom_witness.a9753d.2
// CHECK:STDOUT:   %Destroy.facet.loc43 => constants.%Destroy.facet.ee2
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.loc43 => constants.%Destroy.WithSelf.Op.type.ef7
// CHECK:STDOUT:   %.loc43_9.3 => constants.%.bef
// CHECK:STDOUT:   %impl.elem0.loc43_9.2 => constants.%Sentinel.Op.f05
// CHECK:STDOUT:   %specific_impl_fn.loc43_9.2 => constants.%Sentinel.Op.f05
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.loc42 => constants.%custom_witness.a9753d.1
// CHECK:STDOUT:   %Destroy.facet.loc42 => constants.%Destroy.facet.41d
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.loc42 => constants.%Destroy.WithSelf.Op.type.323
// CHECK:STDOUT:   %.loc42_11.3 => constants.%.458
// CHECK:STDOUT:   %impl.elem0.loc42_11.2 => constants.%Iterator.Op.f05
// CHECK:STDOUT:   %specific_impl_fn.loc42_11.2 => constants.%Iterator.Op.f05
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(constants.%facet_value.2eb) {
// CHECK:STDOUT:   %R.loc40_10.1 => constants.%facet_value.2eb
// CHECK:STDOUT:   %R.as_type.loc40_113.1 => constants.%ConstRange
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.631
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc40 => constants.%complete_type
// CHECK:STDOUT:   %CppRangeForIterate.lookup_impl_witness => constants.%custom_witness.d7a
// CHECK:STDOUT:   %CppRangeForIterate.facet => constants.%CppRangeForIterate.facet.77a
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.Begin.type => constants.%CppRangeForIterate.WithSelf.Begin.type.7ad
// CHECK:STDOUT:   %.loc42_4 => constants.%.421
// CHECK:STDOUT:   %impl.elem2.loc42_4.2 => constants.%ConstRange.Begin
// CHECK:STDOUT:   %specific_impl_fn.loc42_4.2 => constants.%ConstRange.Begin
// CHECK:STDOUT:   %impl.elem0.1 => constants.%Iterator.4f0
// CHECK:STDOUT:   %require_complete.1 => constants.%complete_type
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.End.type => constants.%CppRangeForIterate.WithSelf.End.type.79b
// CHECK:STDOUT:   %.loc43_4 => constants.%.b91
// CHECK:STDOUT:   %impl.elem3.loc43_4.2 => constants.%ConstRange.End
// CHECK:STDOUT:   %specific_impl_fn.loc43_4.2 => constants.%ConstRange.End
// CHECK:STDOUT:   %impl.elem1.1 => constants.%Sentinel.45e
// CHECK:STDOUT:   %require_complete.2 => constants.%complete_type
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.loc43 => constants.%custom_witness.569415.2
// CHECK:STDOUT:   %Destroy.facet.loc43 => constants.%Destroy.facet.a41
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.loc43 => constants.%Destroy.WithSelf.Op.type.0c6
// CHECK:STDOUT:   %.loc43_9.3 => constants.%.df3
// CHECK:STDOUT:   %impl.elem0.loc43_9.2 => constants.%Sentinel.Op.7b7
// CHECK:STDOUT:   %specific_impl_fn.loc43_9.2 => constants.%Sentinel.Op.7b7
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.loc42 => constants.%custom_witness.569415.1
// CHECK:STDOUT:   %Destroy.facet.loc42 => constants.%Destroy.facet.b0c
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.loc42 => constants.%Destroy.WithSelf.Op.type.656
// CHECK:STDOUT:   %.loc42_11.3 => constants.%.80f
// CHECK:STDOUT:   %impl.elem0.loc42_11.2 => constants.%Iterator.Op.7b7
// CHECK:STDOUT:   %specific_impl_fn.loc42_11.2 => constants.%Iterator.Op.7b7
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
// CHECK:STDOUT:   %pattern_type.80f: type = pattern_type %MutableRange [concrete]
// CHECK:STDOUT:   %ConstRange: type = class_type @ConstRange [concrete]
// CHECK:STDOUT:   %pattern_type.88d: type = pattern_type %ConstRange [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %Iterator.b8c: type = class_type @Iterator.1 [concrete]
// CHECK:STDOUT:   %Begin.type.793876.1: type = fn_type @Begin.1 [concrete]
// CHECK:STDOUT:   %Begin.1544bb.1: %Begin.type.793876.1 = struct_value () [concrete]
// CHECK:STDOUT:   %End.type.16398b.1: type = fn_type @End.1 [concrete]
// CHECK:STDOUT:   %End.be76e1.1: %End.type.16398b.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.92f: <witness> = custom_witness (%Iterator.b8c, %Iterator.b8c, %Begin.1544bb.1, %End.be76e1.1), @CppRangeForIterate [concrete]
// CHECK:STDOUT:   %CppRangeForIterate.facet.068: %CppRangeForIterate.type = facet_value %MutableRange, (%custom_witness.92f) [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.910: type = fn_type @Iterator.Op.1 [concrete]
// CHECK:STDOUT:   %Iterator.Op.cbb: %Iterator.Op.type.910 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.d6a: <witness> = custom_witness (%Iterator.Op.cbb), @Destroy [concrete]
// CHECK:STDOUT:   %facet_value.eb2: %CppRangeForIterate_where.type = facet_value %MutableRange, (%custom_witness.92f, %custom_witness.d6a) [concrete]
// CHECK:STDOUT:   %Iterator.080: type = class_type @Iterator.2 [concrete]
// CHECK:STDOUT:   %Begin.type.793876.2: type = fn_type @Begin.2 [concrete]
// CHECK:STDOUT:   %Begin.1544bb.2: %Begin.type.793876.2 = struct_value () [concrete]
// CHECK:STDOUT:   %End.type.16398b.2: type = fn_type @End.2 [concrete]
// CHECK:STDOUT:   %End.be76e1.2: %End.type.16398b.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.532: <witness> = custom_witness (%Iterator.080, %Iterator.080, %Begin.1544bb.2, %End.be76e1.2), @CppRangeForIterate [concrete]
// CHECK:STDOUT:   %CppRangeForIterate.facet.984: %CppRangeForIterate.type = facet_value %ConstRange, (%custom_witness.532) [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.587: type = fn_type @Iterator.Op.2 [concrete]
// CHECK:STDOUT:   %Iterator.Op.dc6: %Iterator.Op.type.587 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b57: <witness> = custom_witness (%Iterator.Op.dc6), @Destroy [concrete]
// CHECK:STDOUT:   %facet_value.5d5: %CppRangeForIterate_where.type = facet_value %ConstRange, (%custom_witness.532, %custom_witness.b57) [concrete]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.Begin.type.949: type = fn_type @CppRangeForIterate.WithSelf.Begin, @CppRangeForIterate.WithSelf(%CppRangeForIterate.facet.068) [concrete]
// CHECK:STDOUT:   %.b32: type = fn_type_with_self_type %CppRangeForIterate.WithSelf.Begin.type.949, %CppRangeForIterate.facet.068 [concrete]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.End.type.e7a: type = fn_type @CppRangeForIterate.WithSelf.End, @CppRangeForIterate.WithSelf(%CppRangeForIterate.facet.068) [concrete]
// CHECK:STDOUT:   %.3d5: type = fn_type_with_self_type %CppRangeForIterate.WithSelf.End.type.e7a, %CppRangeForIterate.facet.068 [concrete]
// CHECK:STDOUT:   %Destroy.facet.44a: %Destroy.type = facet_value %Iterator.b8c, (%custom_witness.d6a) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.360: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.44a) [concrete]
// CHECK:STDOUT:   %.da3: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.360, %Destroy.facet.44a [concrete]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.Begin.type.973: type = fn_type @CppRangeForIterate.WithSelf.Begin, @CppRangeForIterate.WithSelf(%CppRangeForIterate.facet.984) [concrete]
// CHECK:STDOUT:   %.b08: type = fn_type_with_self_type %CppRangeForIterate.WithSelf.Begin.type.973, %CppRangeForIterate.facet.984 [concrete]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.End.type.4f9: type = fn_type @CppRangeForIterate.WithSelf.End, @CppRangeForIterate.WithSelf(%CppRangeForIterate.facet.984) [concrete]
// CHECK:STDOUT:   %.bf2: type = fn_type_with_self_type %CppRangeForIterate.WithSelf.End.type.4f9, %CppRangeForIterate.facet.984 [concrete]
// CHECK:STDOUT:   %Destroy.facet.eb0: %Destroy.type = facet_value %Iterator.080, (%custom_witness.b57) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.bbe: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.eb0) [concrete]
// CHECK:STDOUT:   %.cdf: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.bbe, %Destroy.facet.eb0 [concrete]
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
// CHECK:STDOUT: specific @Test(constants.%facet_value.eb2) {
// CHECK:STDOUT:   %R.loc32_10.1 => constants.%facet_value.eb2
// CHECK:STDOUT:   %R.as_type.loc32_113.1 => constants.%MutableRange
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.80f
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc32 => constants.%complete_type
// CHECK:STDOUT:   %CppRangeForIterate.lookup_impl_witness => constants.%custom_witness.92f
// CHECK:STDOUT:   %CppRangeForIterate.facet => constants.%CppRangeForIterate.facet.068
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.Begin.type => constants.%CppRangeForIterate.WithSelf.Begin.type.949
// CHECK:STDOUT:   %.loc34_4 => constants.%.b32
// CHECK:STDOUT:   %impl.elem2.loc34_4.2 => constants.%Begin.1544bb.1
// CHECK:STDOUT:   %specific_impl_fn.loc34_4.2 => constants.%Begin.1544bb.1
// CHECK:STDOUT:   %impl.elem0.1 => constants.%Iterator.b8c
// CHECK:STDOUT:   %require_complete.1 => constants.%complete_type
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.End.type => constants.%CppRangeForIterate.WithSelf.End.type.e7a
// CHECK:STDOUT:   %.loc35_4 => constants.%.3d5
// CHECK:STDOUT:   %impl.elem3.loc35_4.2 => constants.%End.be76e1.1
// CHECK:STDOUT:   %specific_impl_fn.loc35_4.2 => constants.%End.be76e1.1
// CHECK:STDOUT:   %impl.elem1.1 => constants.%Iterator.b8c
// CHECK:STDOUT:   %require_complete.2 => constants.%complete_type
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.loc35 => constants.%custom_witness.d6a
// CHECK:STDOUT:   %Destroy.facet.loc35 => constants.%Destroy.facet.44a
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.loc35 => constants.%Destroy.WithSelf.Op.type.360
// CHECK:STDOUT:   %.loc35_9.3 => constants.%.da3
// CHECK:STDOUT:   %impl.elem0.loc35_9.2 => constants.%Iterator.Op.cbb
// CHECK:STDOUT:   %specific_impl_fn.loc35_9.2 => constants.%Iterator.Op.cbb
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.loc34 => constants.%custom_witness.d6a
// CHECK:STDOUT:   %Destroy.facet.loc34 => constants.%Destroy.facet.44a
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.loc34 => constants.%Destroy.WithSelf.Op.type.360
// CHECK:STDOUT:   %.loc34_11.3 => constants.%.da3
// CHECK:STDOUT:   %impl.elem0.loc34_11.2 => constants.%Iterator.Op.cbb
// CHECK:STDOUT:   %specific_impl_fn.loc34_11.2 => constants.%Iterator.Op.cbb
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(constants.%facet_value.5d5) {
// CHECK:STDOUT:   %R.loc32_10.1 => constants.%facet_value.5d5
// CHECK:STDOUT:   %R.as_type.loc32_113.1 => constants.%ConstRange
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.88d
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc32 => constants.%complete_type
// CHECK:STDOUT:   %CppRangeForIterate.lookup_impl_witness => constants.%custom_witness.532
// CHECK:STDOUT:   %CppRangeForIterate.facet => constants.%CppRangeForIterate.facet.984
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.Begin.type => constants.%CppRangeForIterate.WithSelf.Begin.type.973
// CHECK:STDOUT:   %.loc34_4 => constants.%.b08
// CHECK:STDOUT:   %impl.elem2.loc34_4.2 => constants.%Begin.1544bb.2
// CHECK:STDOUT:   %specific_impl_fn.loc34_4.2 => constants.%Begin.1544bb.2
// CHECK:STDOUT:   %impl.elem0.1 => constants.%Iterator.080
// CHECK:STDOUT:   %require_complete.1 => constants.%complete_type
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.End.type => constants.%CppRangeForIterate.WithSelf.End.type.4f9
// CHECK:STDOUT:   %.loc35_4 => constants.%.bf2
// CHECK:STDOUT:   %impl.elem3.loc35_4.2 => constants.%End.be76e1.2
// CHECK:STDOUT:   %specific_impl_fn.loc35_4.2 => constants.%End.be76e1.2
// CHECK:STDOUT:   %impl.elem1.1 => constants.%Iterator.080
// CHECK:STDOUT:   %require_complete.2 => constants.%complete_type
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.loc35 => constants.%custom_witness.b57
// CHECK:STDOUT:   %Destroy.facet.loc35 => constants.%Destroy.facet.eb0
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.loc35 => constants.%Destroy.WithSelf.Op.type.bbe
// CHECK:STDOUT:   %.loc35_9.3 => constants.%.cdf
// CHECK:STDOUT:   %impl.elem0.loc35_9.2 => constants.%Iterator.Op.dc6
// CHECK:STDOUT:   %specific_impl_fn.loc35_9.2 => constants.%Iterator.Op.dc6
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.loc34 => constants.%custom_witness.b57
// CHECK:STDOUT:   %Destroy.facet.loc34 => constants.%Destroy.facet.eb0
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.loc34 => constants.%Destroy.WithSelf.Op.type.bbe
// CHECK:STDOUT:   %.loc34_11.3 => constants.%.cdf
// CHECK:STDOUT:   %impl.elem0.loc34_11.2 => constants.%Iterator.Op.dc6
// CHECK:STDOUT:   %specific_impl_fn.loc34_11.2 => constants.%Iterator.Op.dc6
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
// CHECK:STDOUT:   %pattern_type.80f: type = pattern_type %MutableRange [concrete]
// CHECK:STDOUT:   %ConstRange: type = class_type @ConstRange [concrete]
// CHECK:STDOUT:   %pattern_type.88d: type = pattern_type %ConstRange [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %Iterator.b8c: type = class_type @Iterator.1 [concrete]
// CHECK:STDOUT:   %Sentinel.501: type = class_type @Sentinel.1 [concrete]
// CHECK:STDOUT:   %Begin.type.793876.1: type = fn_type @Begin.1 [concrete]
// CHECK:STDOUT:   %Begin.1544bb.1: %Begin.type.793876.1 = struct_value () [concrete]
// CHECK:STDOUT:   %End.type.16398b.1: type = fn_type @End.1 [concrete]
// CHECK:STDOUT:   %End.be76e1.1: %End.type.16398b.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.480: <witness> = custom_witness (%Iterator.b8c, %Sentinel.501, %Begin.1544bb.1, %End.be76e1.1), @CppRangeForIterate [concrete]
// CHECK:STDOUT:   %CppRangeForIterate.facet.1f7: %CppRangeForIterate.type = facet_value %MutableRange, (%custom_witness.480) [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.910: type = fn_type @Iterator.Op.1 [concrete]
// CHECK:STDOUT:   %Iterator.Op.cbb: %Iterator.Op.type.910 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.d6afef.1: <witness> = custom_witness (%Iterator.Op.cbb), @Destroy [concrete]
// CHECK:STDOUT:   %Sentinel.Op.type.910: type = fn_type @Sentinel.Op.1 [concrete]
// CHECK:STDOUT:   %Sentinel.Op.cbb: %Sentinel.Op.type.910 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.d6afef.2: <witness> = custom_witness (%Sentinel.Op.cbb), @Destroy [concrete]
// CHECK:STDOUT:   %facet_value.388: %CppRangeForIterate_where.type = facet_value %MutableRange, (%custom_witness.480, %custom_witness.d6afef.1, %custom_witness.d6afef.2) [concrete]
// CHECK:STDOUT:   %Iterator.080: type = class_type @Iterator.2 [concrete]
// CHECK:STDOUT:   %Sentinel.a47: type = class_type @Sentinel.2 [concrete]
// CHECK:STDOUT:   %Begin.type.793876.2: type = fn_type @Begin.2 [concrete]
// CHECK:STDOUT:   %Begin.1544bb.2: %Begin.type.793876.2 = struct_value () [concrete]
// CHECK:STDOUT:   %End.type.16398b.2: type = fn_type @End.2 [concrete]
// CHECK:STDOUT:   %End.be76e1.2: %End.type.16398b.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.7a2: <witness> = custom_witness (%Iterator.080, %Sentinel.a47, %Begin.1544bb.2, %End.be76e1.2), @CppRangeForIterate [concrete]
// CHECK:STDOUT:   %CppRangeForIterate.facet.48c: %CppRangeForIterate.type = facet_value %ConstRange, (%custom_witness.7a2) [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.587: type = fn_type @Iterator.Op.2 [concrete]
// CHECK:STDOUT:   %Iterator.Op.dc6: %Iterator.Op.type.587 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b57f0b.1: <witness> = custom_witness (%Iterator.Op.dc6), @Destroy [concrete]
// CHECK:STDOUT:   %Sentinel.Op.type.587: type = fn_type @Sentinel.Op.2 [concrete]
// CHECK:STDOUT:   %Sentinel.Op.dc6: %Sentinel.Op.type.587 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b57f0b.2: <witness> = custom_witness (%Sentinel.Op.dc6), @Destroy [concrete]
// CHECK:STDOUT:   %facet_value.38d: %CppRangeForIterate_where.type = facet_value %ConstRange, (%custom_witness.7a2, %custom_witness.b57f0b.1, %custom_witness.b57f0b.2) [concrete]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.Begin.type.c44: type = fn_type @CppRangeForIterate.WithSelf.Begin, @CppRangeForIterate.WithSelf(%CppRangeForIterate.facet.1f7) [concrete]
// CHECK:STDOUT:   %.a44: type = fn_type_with_self_type %CppRangeForIterate.WithSelf.Begin.type.c44, %CppRangeForIterate.facet.1f7 [concrete]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.End.type.d16: type = fn_type @CppRangeForIterate.WithSelf.End, @CppRangeForIterate.WithSelf(%CppRangeForIterate.facet.1f7) [concrete]
// CHECK:STDOUT:   %.994: type = fn_type_with_self_type %CppRangeForIterate.WithSelf.End.type.d16, %CppRangeForIterate.facet.1f7 [concrete]
// CHECK:STDOUT:   %Destroy.facet.4b0: %Destroy.type = facet_value %Sentinel.501, (%custom_witness.d6afef.2) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.5e0: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.4b0) [concrete]
// CHECK:STDOUT:   %.132: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.5e0, %Destroy.facet.4b0 [concrete]
// CHECK:STDOUT:   %Destroy.facet.44a: %Destroy.type = facet_value %Iterator.b8c, (%custom_witness.d6afef.1) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.360: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.44a) [concrete]
// CHECK:STDOUT:   %.da3: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.360, %Destroy.facet.44a [concrete]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.Begin.type.6b1: type = fn_type @CppRangeForIterate.WithSelf.Begin, @CppRangeForIterate.WithSelf(%CppRangeForIterate.facet.48c) [concrete]
// CHECK:STDOUT:   %.2e1: type = fn_type_with_self_type %CppRangeForIterate.WithSelf.Begin.type.6b1, %CppRangeForIterate.facet.48c [concrete]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.End.type.4c6: type = fn_type @CppRangeForIterate.WithSelf.End, @CppRangeForIterate.WithSelf(%CppRangeForIterate.facet.48c) [concrete]
// CHECK:STDOUT:   %.94a: type = fn_type_with_self_type %CppRangeForIterate.WithSelf.End.type.4c6, %CppRangeForIterate.facet.48c [concrete]
// CHECK:STDOUT:   %Destroy.facet.7e7: %Destroy.type = facet_value %Sentinel.a47, (%custom_witness.b57f0b.2) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.04c: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.7e7) [concrete]
// CHECK:STDOUT:   %.bdc: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.04c, %Destroy.facet.7e7 [concrete]
// CHECK:STDOUT:   %Destroy.facet.eb0: %Destroy.type = facet_value %Iterator.080, (%custom_witness.b57f0b.1) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.bbe: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.eb0) [concrete]
// CHECK:STDOUT:   %.cdf: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.bbe, %Destroy.facet.eb0 [concrete]
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
// CHECK:STDOUT: specific @Test(constants.%facet_value.388) {
// CHECK:STDOUT:   %R.loc42_10.1 => constants.%facet_value.388
// CHECK:STDOUT:   %R.as_type.loc42_113.1 => constants.%MutableRange
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.80f
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc42 => constants.%complete_type
// CHECK:STDOUT:   %CppRangeForIterate.lookup_impl_witness => constants.%custom_witness.480
// CHECK:STDOUT:   %CppRangeForIterate.facet => constants.%CppRangeForIterate.facet.1f7
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.Begin.type => constants.%CppRangeForIterate.WithSelf.Begin.type.c44
// CHECK:STDOUT:   %.loc44_4 => constants.%.a44
// CHECK:STDOUT:   %impl.elem2.loc44_4.2 => constants.%Begin.1544bb.1
// CHECK:STDOUT:   %specific_impl_fn.loc44_4.2 => constants.%Begin.1544bb.1
// CHECK:STDOUT:   %impl.elem0.1 => constants.%Iterator.b8c
// CHECK:STDOUT:   %require_complete.1 => constants.%complete_type
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.End.type => constants.%CppRangeForIterate.WithSelf.End.type.d16
// CHECK:STDOUT:   %.loc45_4 => constants.%.994
// CHECK:STDOUT:   %impl.elem3.loc45_4.2 => constants.%End.be76e1.1
// CHECK:STDOUT:   %specific_impl_fn.loc45_4.2 => constants.%End.be76e1.1
// CHECK:STDOUT:   %impl.elem1.1 => constants.%Sentinel.501
// CHECK:STDOUT:   %require_complete.2 => constants.%complete_type
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.loc45 => constants.%custom_witness.d6afef.2
// CHECK:STDOUT:   %Destroy.facet.loc45 => constants.%Destroy.facet.4b0
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.loc45 => constants.%Destroy.WithSelf.Op.type.5e0
// CHECK:STDOUT:   %.loc45_9.3 => constants.%.132
// CHECK:STDOUT:   %impl.elem0.loc45_9.2 => constants.%Sentinel.Op.cbb
// CHECK:STDOUT:   %specific_impl_fn.loc45_9.2 => constants.%Sentinel.Op.cbb
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.loc44 => constants.%custom_witness.d6afef.1
// CHECK:STDOUT:   %Destroy.facet.loc44 => constants.%Destroy.facet.44a
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.loc44 => constants.%Destroy.WithSelf.Op.type.360
// CHECK:STDOUT:   %.loc44_11.3 => constants.%.da3
// CHECK:STDOUT:   %impl.elem0.loc44_11.2 => constants.%Iterator.Op.cbb
// CHECK:STDOUT:   %specific_impl_fn.loc44_11.2 => constants.%Iterator.Op.cbb
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(constants.%facet_value.38d) {
// CHECK:STDOUT:   %R.loc42_10.1 => constants.%facet_value.38d
// CHECK:STDOUT:   %R.as_type.loc42_113.1 => constants.%ConstRange
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.88d
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc42 => constants.%complete_type
// CHECK:STDOUT:   %CppRangeForIterate.lookup_impl_witness => constants.%custom_witness.7a2
// CHECK:STDOUT:   %CppRangeForIterate.facet => constants.%CppRangeForIterate.facet.48c
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.Begin.type => constants.%CppRangeForIterate.WithSelf.Begin.type.6b1
// CHECK:STDOUT:   %.loc44_4 => constants.%.2e1
// CHECK:STDOUT:   %impl.elem2.loc44_4.2 => constants.%Begin.1544bb.2
// CHECK:STDOUT:   %specific_impl_fn.loc44_4.2 => constants.%Begin.1544bb.2
// CHECK:STDOUT:   %impl.elem0.1 => constants.%Iterator.080
// CHECK:STDOUT:   %require_complete.1 => constants.%complete_type
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.End.type => constants.%CppRangeForIterate.WithSelf.End.type.4c6
// CHECK:STDOUT:   %.loc45_4 => constants.%.94a
// CHECK:STDOUT:   %impl.elem3.loc45_4.2 => constants.%End.be76e1.2
// CHECK:STDOUT:   %specific_impl_fn.loc45_4.2 => constants.%End.be76e1.2
// CHECK:STDOUT:   %impl.elem1.1 => constants.%Sentinel.a47
// CHECK:STDOUT:   %require_complete.2 => constants.%complete_type
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.loc45 => constants.%custom_witness.b57f0b.2
// CHECK:STDOUT:   %Destroy.facet.loc45 => constants.%Destroy.facet.7e7
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.loc45 => constants.%Destroy.WithSelf.Op.type.04c
// CHECK:STDOUT:   %.loc45_9.3 => constants.%.bdc
// CHECK:STDOUT:   %impl.elem0.loc45_9.2 => constants.%Sentinel.Op.dc6
// CHECK:STDOUT:   %specific_impl_fn.loc45_9.2 => constants.%Sentinel.Op.dc6
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.loc44 => constants.%custom_witness.b57f0b.1
// CHECK:STDOUT:   %Destroy.facet.loc44 => constants.%Destroy.facet.eb0
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.loc44 => constants.%Destroy.WithSelf.Op.type.bbe
// CHECK:STDOUT:   %.loc44_11.3 => constants.%.cdf
// CHECK:STDOUT:   %impl.elem0.loc44_11.2 => constants.%Iterator.Op.dc6
// CHECK:STDOUT:   %specific_impl_fn.loc44_11.2 => constants.%Iterator.Op.dc6
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
// CHECK:STDOUT:   %pattern_type.e6c: type = pattern_type %MutableAndConstRange [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %Iterator: type = class_type @Iterator.1 [concrete]
// CHECK:STDOUT:   %Begin.type: type = fn_type @Begin [concrete]
// CHECK:STDOUT:   %Begin: %Begin.type = struct_value () [concrete]
// CHECK:STDOUT:   %End.type: type = fn_type @End [concrete]
// CHECK:STDOUT:   %End: %End.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.0f5: <witness> = custom_witness (%Iterator, %Iterator, %Begin, %End), @CppRangeForIterate [concrete]
// CHECK:STDOUT:   %CppRangeForIterate.facet.4ed: %CppRangeForIterate.type = facet_value %MutableAndConstRange, (%custom_witness.0f5) [concrete]
// CHECK:STDOUT:   %Iterator.Op.type: type = fn_type @Iterator.Op [concrete]
// CHECK:STDOUT:   %Iterator.Op: %Iterator.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b15: <witness> = custom_witness (%Iterator.Op), @Destroy [concrete]
// CHECK:STDOUT:   %facet_value: %CppRangeForIterate_where.type = facet_value %MutableAndConstRange, (%custom_witness.0f5, %custom_witness.b15) [concrete]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.Begin.type.92e: type = fn_type @CppRangeForIterate.WithSelf.Begin, @CppRangeForIterate.WithSelf(%CppRangeForIterate.facet.4ed) [concrete]
// CHECK:STDOUT:   %.da6: type = fn_type_with_self_type %CppRangeForIterate.WithSelf.Begin.type.92e, %CppRangeForIterate.facet.4ed [concrete]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.End.type.576: type = fn_type @CppRangeForIterate.WithSelf.End, @CppRangeForIterate.WithSelf(%CppRangeForIterate.facet.4ed) [concrete]
// CHECK:STDOUT:   %.685: type = fn_type_with_self_type %CppRangeForIterate.WithSelf.End.type.576, %CppRangeForIterate.facet.4ed [concrete]
// CHECK:STDOUT:   %Destroy.facet.f9a: %Destroy.type = facet_value %Iterator, (%custom_witness.b15) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.343: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.f9a) [concrete]
// CHECK:STDOUT:   %.e83: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.343, %Destroy.facet.f9a [concrete]
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
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.e6c
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc32 => constants.%complete_type
// CHECK:STDOUT:   %CppRangeForIterate.lookup_impl_witness => constants.%custom_witness.0f5
// CHECK:STDOUT:   %CppRangeForIterate.facet => constants.%CppRangeForIterate.facet.4ed
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.Begin.type => constants.%CppRangeForIterate.WithSelf.Begin.type.92e
// CHECK:STDOUT:   %.loc34_4 => constants.%.da6
// CHECK:STDOUT:   %impl.elem2.loc34_4.2 => constants.%Begin
// CHECK:STDOUT:   %specific_impl_fn.loc34_4.2 => constants.%Begin
// CHECK:STDOUT:   %impl.elem0.1 => constants.%Iterator
// CHECK:STDOUT:   %require_complete.1 => constants.%complete_type
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.End.type => constants.%CppRangeForIterate.WithSelf.End.type.576
// CHECK:STDOUT:   %.loc35_4 => constants.%.685
// CHECK:STDOUT:   %impl.elem3.loc35_4.2 => constants.%End
// CHECK:STDOUT:   %specific_impl_fn.loc35_4.2 => constants.%End
// CHECK:STDOUT:   %impl.elem1.1 => constants.%Iterator
// CHECK:STDOUT:   %require_complete.2 => constants.%complete_type
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.loc35 => constants.%custom_witness.b15
// CHECK:STDOUT:   %Destroy.facet.loc35 => constants.%Destroy.facet.f9a
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.loc35 => constants.%Destroy.WithSelf.Op.type.343
// CHECK:STDOUT:   %.loc35_9.3 => constants.%.e83
// CHECK:STDOUT:   %impl.elem0.loc35_9.2 => constants.%Iterator.Op
// CHECK:STDOUT:   %specific_impl_fn.loc35_9.2 => constants.%Iterator.Op
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.loc34 => constants.%custom_witness.b15
// CHECK:STDOUT:   %Destroy.facet.loc34 => constants.%Destroy.facet.f9a
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.loc34 => constants.%Destroy.WithSelf.Op.type.343
// CHECK:STDOUT:   %.loc34_11.3 => constants.%.e83
// CHECK:STDOUT:   %impl.elem0.loc34_11.2 => constants.%Iterator.Op
// CHECK:STDOUT:   %specific_impl_fn.loc34_11.2 => constants.%Iterator.Op
// CHECK:STDOUT: }
// CHECK:STDOUT:
