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
// CHECK:STDOUT:   %Self.39b: %CppRangeForIterate.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.End.type.fe619a.1: type = fn_type @CppRangeForIterate.WithSelf.End, @CppRangeForIterate.WithSelf(%Self.39b) [symbolic]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.End.0f5092.1: %CppRangeForIterate.WithSelf.End.type.fe619a.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.Begin.type.3276c2.1: type = fn_type @CppRangeForIterate.WithSelf.Begin, @CppRangeForIterate.WithSelf(%Self.39b) [symbolic]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.Begin.39882e.1: %CppRangeForIterate.WithSelf.Begin.type.3276c2.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %.Self.6bd: %CppRangeForIterate.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %CppRangeForIterate.assoc_type: type = assoc_entity_type @CppRangeForIterate [concrete]
// CHECK:STDOUT:   %CppRangeForIterate.lookup_impl_witness.143: <witness> = lookup_impl_witness %.Self.6bd, @CppRangeForIterate [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.109: type = impl_witness_access %CppRangeForIterate.lookup_impl_witness.143, element0 [symbolic_self]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %impl.elem1.222: type = impl_witness_access %CppRangeForIterate.lookup_impl_witness.143, element1 [symbolic_self]
// CHECK:STDOUT:   %CppRangeForIterate_where.type: type = facet_type <@CppRangeForIterate where %impl.elem0.109 impls @Destroy and %impl.elem1.222 impls @Destroy> [concrete]
// CHECK:STDOUT:   %R: %CppRangeForIterate_where.type = symbolic_binding R, 0 [symbolic]
// CHECK:STDOUT:   %R.as_type: type = facet_access_type %R [symbolic]
// CHECK:STDOUT:   %pattern_type.48fa74.2: type = pattern_type %R.as_type [symbolic]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %assoc2: %CppRangeForIterate.assoc_type = assoc_entity element2, imports.%Core.import_ref.08d [concrete]
// CHECK:STDOUT:   %CppRangeForIterate.lookup_impl_witness.7e2712.2: <witness> = lookup_impl_witness %R, @CppRangeForIterate [symbolic]
// CHECK:STDOUT:   %CppRangeForIterate.facet.9d6: %CppRangeForIterate.type = facet_value %R.as_type, (%CppRangeForIterate.lookup_impl_witness.7e2712.2) [symbolic]
// CHECK:STDOUT:   %impl.elem0.73eb7f.2: type = impl_witness_access %CppRangeForIterate.lookup_impl_witness.7e2712.2, element0 [symbolic]
// CHECK:STDOUT:   %impl.elem1.7a0bff.2: type = impl_witness_access %CppRangeForIterate.lookup_impl_witness.7e2712.2, element1 [symbolic]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.Begin.type.848: type = fn_type @CppRangeForIterate.WithSelf.Begin, @CppRangeForIterate.WithSelf(%CppRangeForIterate.facet.9d6) [symbolic]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.End.type.2e5: type = fn_type @CppRangeForIterate.WithSelf.End, @CppRangeForIterate.WithSelf(%CppRangeForIterate.facet.9d6) [symbolic]
// CHECK:STDOUT:   %.be2: type = fn_type_with_self_type %CppRangeForIterate.WithSelf.Begin.type.848, %CppRangeForIterate.facet.9d6 [symbolic]
// CHECK:STDOUT:   %impl.elem2: %.be2 = impl_witness_access %CppRangeForIterate.lookup_impl_witness.7e2712.2, element2 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.b91: <specific function> = specific_impl_function %impl.elem2, @CppRangeForIterate.WithSelf.Begin(%CppRangeForIterate.facet.9d6) [symbolic]
// CHECK:STDOUT:   %assoc3: %CppRangeForIterate.assoc_type = assoc_entity element3, imports.%Core.import_ref.49d [concrete]
// CHECK:STDOUT:   %.fbd: type = fn_type_with_self_type %CppRangeForIterate.WithSelf.End.type.2e5, %CppRangeForIterate.facet.9d6 [symbolic]
// CHECK:STDOUT:   %impl.elem3: %.fbd = impl_witness_access %CppRangeForIterate.lookup_impl_witness.7e2712.2, element3 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.2ec: <specific function> = specific_impl_function %impl.elem3, @CppRangeForIterate.WithSelf.End(%CppRangeForIterate.facet.9d6) [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.afb: <witness> = lookup_impl_witness %impl.elem1.7a0bff.2, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.63a: %Destroy.type = facet_value %impl.elem1.7a0bff.2, (%Destroy.lookup_impl_witness.afb) [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.57d: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.63a) [symbolic]
// CHECK:STDOUT:   %.db6: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.57d, %Destroy.facet.63a [symbolic]
// CHECK:STDOUT:   %impl.elem0.180: %.db6 = impl_witness_access %Destroy.lookup_impl_witness.afb, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.a08: <specific function> = specific_impl_function %impl.elem0.180, @Destroy.WithSelf.Op(%Destroy.facet.63a) [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.75c: <witness> = lookup_impl_witness %impl.elem0.73eb7f.2, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.dae: %Destroy.type = facet_value %impl.elem0.73eb7f.2, (%Destroy.lookup_impl_witness.75c) [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.5c3: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.dae) [symbolic]
// CHECK:STDOUT:   %.51a: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.5c3, %Destroy.facet.dae [symbolic]
// CHECK:STDOUT:   %impl.elem0.2a2: %.51a = impl_witness_access %Destroy.lookup_impl_witness.75c, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.842: <specific function> = specific_impl_function %impl.elem0.2a2, @Destroy.WithSelf.Op(%Destroy.facet.dae) [symbolic]
// CHECK:STDOUT:   %MutableRange: type = class_type @MutableRange [concrete]
// CHECK:STDOUT:   %pattern_type.e63: type = pattern_type %MutableRange [concrete]
// CHECK:STDOUT:   %ConstRange: type = class_type @ConstRange [concrete]
// CHECK:STDOUT:   %pattern_type.0ba: type = pattern_type %ConstRange [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %Iterator.3ee: type = class_type @Iterator.1 [concrete]
// CHECK:STDOUT:   %MutableRange.Begin.type: type = fn_type @MutableRange.Begin [concrete]
// CHECK:STDOUT:   %MutableRange.Begin: %MutableRange.Begin.type = struct_value () [concrete]
// CHECK:STDOUT:   %MutableRange.End.type: type = fn_type @MutableRange.End [concrete]
// CHECK:STDOUT:   %MutableRange.End: %MutableRange.End.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.ed3: <witness> = custom_witness (%Iterator.3ee, %Iterator.3ee, %MutableRange.Begin, %MutableRange.End), @CppRangeForIterate [concrete]
// CHECK:STDOUT:   %CppRangeForIterate.facet.1e8: %CppRangeForIterate.type = facet_value %MutableRange, (%custom_witness.ed3) [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.0ae: type = fn_type @Iterator.Op.1 [concrete]
// CHECK:STDOUT:   %Iterator.Op.a2a: %Iterator.Op.type.0ae = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.955: <witness> = custom_witness (%Iterator.Op.a2a), @Destroy [concrete]
// CHECK:STDOUT:   %facet_value.a34: %CppRangeForIterate_where.type = facet_value %MutableRange, (%custom_witness.ed3, %custom_witness.955) [concrete]
// CHECK:STDOUT:   %Iterator.260: type = class_type @Iterator.2 [concrete]
// CHECK:STDOUT:   %ConstRange.Begin.type: type = fn_type @ConstRange.Begin [concrete]
// CHECK:STDOUT:   %ConstRange.Begin: %ConstRange.Begin.type = struct_value () [concrete]
// CHECK:STDOUT:   %ConstRange.End.type: type = fn_type @ConstRange.End [concrete]
// CHECK:STDOUT:   %ConstRange.End: %ConstRange.End.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.23e: <witness> = custom_witness (%Iterator.260, %Iterator.260, %ConstRange.Begin, %ConstRange.End), @CppRangeForIterate [concrete]
// CHECK:STDOUT:   %CppRangeForIterate.facet.ee3: %CppRangeForIterate.type = facet_value %ConstRange, (%custom_witness.23e) [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.0f3: type = fn_type @Iterator.Op.2 [concrete]
// CHECK:STDOUT:   %Iterator.Op.1c3: %Iterator.Op.type.0f3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.6e8: <witness> = custom_witness (%Iterator.Op.1c3), @Destroy [concrete]
// CHECK:STDOUT:   %facet_value.f5b: %CppRangeForIterate_where.type = facet_value %ConstRange, (%custom_witness.23e, %custom_witness.6e8) [concrete]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.Begin.type.0af: type = fn_type @CppRangeForIterate.WithSelf.Begin, @CppRangeForIterate.WithSelf(%CppRangeForIterate.facet.1e8) [concrete]
// CHECK:STDOUT:   %.14b: type = fn_type_with_self_type %CppRangeForIterate.WithSelf.Begin.type.0af, %CppRangeForIterate.facet.1e8 [concrete]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.End.type.47c: type = fn_type @CppRangeForIterate.WithSelf.End, @CppRangeForIterate.WithSelf(%CppRangeForIterate.facet.1e8) [concrete]
// CHECK:STDOUT:   %.d84: type = fn_type_with_self_type %CppRangeForIterate.WithSelf.End.type.47c, %CppRangeForIterate.facet.1e8 [concrete]
// CHECK:STDOUT:   %Destroy.facet.2de: %Destroy.type = facet_value %Iterator.3ee, (%custom_witness.955) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.69b: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.2de) [concrete]
// CHECK:STDOUT:   %.69f: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.69b, %Destroy.facet.2de [concrete]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.Begin.type.c00: type = fn_type @CppRangeForIterate.WithSelf.Begin, @CppRangeForIterate.WithSelf(%CppRangeForIterate.facet.ee3) [concrete]
// CHECK:STDOUT:   %.ebd: type = fn_type_with_self_type %CppRangeForIterate.WithSelf.Begin.type.c00, %CppRangeForIterate.facet.ee3 [concrete]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.End.type.b02: type = fn_type @CppRangeForIterate.WithSelf.End, @CppRangeForIterate.WithSelf(%CppRangeForIterate.facet.ee3) [concrete]
// CHECK:STDOUT:   %.066: type = fn_type_with_self_type %CppRangeForIterate.WithSelf.End.type.b02, %CppRangeForIterate.facet.ee3 [concrete]
// CHECK:STDOUT:   %Destroy.facet.a30: %Destroy.type = facet_value %Iterator.260, (%custom_witness.6e8) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.8b1: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.a30) [concrete]
// CHECK:STDOUT:   %.f73: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.8b1, %Destroy.facet.a30 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.a6a: %CppRangeForIterate.assoc_type = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [concrete = constants.%assoc2]
// CHECK:STDOUT:   %Core.import_ref.5b4: %CppRangeForIterate.assoc_type = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [concrete = constants.%assoc3]
// CHECK:STDOUT:   %Core.import_ref.08d: @CppRangeForIterate.WithSelf.%CppRangeForIterate.WithSelf.Begin.type (%CppRangeForIterate.WithSelf.Begin.type.3276c2.1) = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @CppRangeForIterate.WithSelf.%CppRangeForIterate.WithSelf.Begin (constants.%CppRangeForIterate.WithSelf.Begin.39882e.1)]
// CHECK:STDOUT:   %Core.import_ref.49d: @CppRangeForIterate.WithSelf.%CppRangeForIterate.WithSelf.End.type (%CppRangeForIterate.WithSelf.End.type.fe619a.1) = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @CppRangeForIterate.WithSelf.%CppRangeForIterate.WithSelf.End (constants.%CppRangeForIterate.WithSelf.End.0f5092.1)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Test(%R.loc30_10.2: %CppRangeForIterate_where.type) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %CppRangeForIterate.lookup_impl_witness: <witness> = lookup_impl_witness %R.loc30_10.1, @CppRangeForIterate [symbolic = %CppRangeForIterate.lookup_impl_witness (constants.%CppRangeForIterate.lookup_impl_witness.7e2712.2)]
// CHECK:STDOUT:   %CppRangeForIterate.facet: %CppRangeForIterate.type = facet_value %R.as_type.loc30_113.1, (%CppRangeForIterate.lookup_impl_witness) [symbolic = %CppRangeForIterate.facet (constants.%CppRangeForIterate.facet.9d6)]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.Begin.type: type = fn_type @CppRangeForIterate.WithSelf.Begin, @CppRangeForIterate.WithSelf(%CppRangeForIterate.facet) [symbolic = %CppRangeForIterate.WithSelf.Begin.type (constants.%CppRangeForIterate.WithSelf.Begin.type.848)]
// CHECK:STDOUT:   %.loc32_4: type = fn_type_with_self_type %CppRangeForIterate.WithSelf.Begin.type, %CppRangeForIterate.facet [symbolic = %.loc32_4 (constants.%.be2)]
// CHECK:STDOUT:   %impl.elem2.loc32_4.2: @Test.%.loc32_4 (%.be2) = impl_witness_access %CppRangeForIterate.lookup_impl_witness, element2 [symbolic = %impl.elem2.loc32_4.2 (constants.%impl.elem2)]
// CHECK:STDOUT:   %specific_impl_fn.loc32_4.2: <specific function> = specific_impl_function %impl.elem2.loc32_4.2, @CppRangeForIterate.WithSelf.Begin(%CppRangeForIterate.facet) [symbolic = %specific_impl_fn.loc32_4.2 (constants.%specific_impl_fn.b91)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.End.type: type = fn_type @CppRangeForIterate.WithSelf.End, @CppRangeForIterate.WithSelf(%CppRangeForIterate.facet) [symbolic = %CppRangeForIterate.WithSelf.End.type (constants.%CppRangeForIterate.WithSelf.End.type.2e5)]
// CHECK:STDOUT:   %.loc33_4: type = fn_type_with_self_type %CppRangeForIterate.WithSelf.End.type, %CppRangeForIterate.facet [symbolic = %.loc33_4 (constants.%.fbd)]
// CHECK:STDOUT:   %impl.elem3.loc33_4.2: @Test.%.loc33_4 (%.fbd) = impl_witness_access %CppRangeForIterate.lookup_impl_witness, element3 [symbolic = %impl.elem3.loc33_4.2 (constants.%impl.elem3)]
// CHECK:STDOUT:   %specific_impl_fn.loc33_4.2: <specific function> = specific_impl_function %impl.elem3.loc33_4.2, @CppRangeForIterate.WithSelf.End(%CppRangeForIterate.facet) [symbolic = %specific_impl_fn.loc33_4.2 (constants.%specific_impl_fn.2ec)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.loc33: <witness> = lookup_impl_witness %impl.elem1.1, @Destroy [symbolic = %Destroy.lookup_impl_witness.loc33 (constants.%Destroy.lookup_impl_witness.afb)]
// CHECK:STDOUT:   %Destroy.facet.loc33: %Destroy.type = facet_value %impl.elem1.1, (%Destroy.lookup_impl_witness.loc33) [symbolic = %Destroy.facet.loc33 (constants.%Destroy.facet.63a)]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.loc33: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.loc33) [symbolic = %Destroy.WithSelf.Op.type.loc33 (constants.%Destroy.WithSelf.Op.type.57d)]
// CHECK:STDOUT:   %.loc33_9.3: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.loc33, %Destroy.facet.loc33 [symbolic = %.loc33_9.3 (constants.%.db6)]
// CHECK:STDOUT:   %impl.elem0.loc33_9.2: @Test.%.loc33_9.3 (%.db6) = impl_witness_access %Destroy.lookup_impl_witness.loc33, element0 [symbolic = %impl.elem0.loc33_9.2 (constants.%impl.elem0.180)]
// CHECK:STDOUT:   %specific_impl_fn.loc33_9.2: <specific function> = specific_impl_function %impl.elem0.loc33_9.2, @Destroy.WithSelf.Op(%Destroy.facet.loc33) [symbolic = %specific_impl_fn.loc33_9.2 (constants.%specific_impl_fn.a08)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.loc32: <witness> = lookup_impl_witness %impl.elem0.1, @Destroy [symbolic = %Destroy.lookup_impl_witness.loc32 (constants.%Destroy.lookup_impl_witness.75c)]
// CHECK:STDOUT:   %Destroy.facet.loc32: %Destroy.type = facet_value %impl.elem0.1, (%Destroy.lookup_impl_witness.loc32) [symbolic = %Destroy.facet.loc32 (constants.%Destroy.facet.dae)]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.loc32: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.loc32) [symbolic = %Destroy.WithSelf.Op.type.loc32 (constants.%Destroy.WithSelf.Op.type.5c3)]
// CHECK:STDOUT:   %.loc32_11.3: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.loc32, %Destroy.facet.loc32 [symbolic = %.loc32_11.3 (constants.%.51a)]
// CHECK:STDOUT:   %impl.elem0.loc32_11.2: @Test.%.loc32_11.3 (%.51a) = impl_witness_access %Destroy.lookup_impl_witness.loc32, element0 [symbolic = %impl.elem0.loc32_11.2 (constants.%impl.elem0.2a2)]
// CHECK:STDOUT:   %specific_impl_fn.loc32_11.2: <specific function> = specific_impl_function %impl.elem0.loc32_11.2, @Destroy.WithSelf.Op(%Destroy.facet.loc32) [symbolic = %specific_impl_fn.loc32_11.2 (constants.%specific_impl_fn.842)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%r.param: ref @Test.%R.as_type.loc30_113.1 (%R.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %r.ref.loc32: ref @Test.%R.as_type.loc30_113.1 (%R.as_type) = name_ref r, %r
// CHECK:STDOUT:     %Begin.ref: %CppRangeForIterate.assoc_type = name_ref Begin, imports.%Core.import_ref.a6a [concrete = constants.%assoc2]
// CHECK:STDOUT:     %impl.elem2.loc32_4.1: @Test.%.loc32_4 (%.be2) = impl_witness_access constants.%CppRangeForIterate.lookup_impl_witness.7e2712.2, element2 [symbolic = %impl.elem2.loc32_4.2 (constants.%impl.elem2)]
// CHECK:STDOUT:     %bound_method.loc32_4: <bound method> = bound_method %r.ref.loc32, %impl.elem2.loc32_4.1
// CHECK:STDOUT:     %specific_impl_fn.loc32_4.1: <specific function> = specific_impl_function %impl.elem2.loc32_4.1, @CppRangeForIterate.WithSelf.Begin(constants.%CppRangeForIterate.facet.9d6) [symbolic = %specific_impl_fn.loc32_4.2 (constants.%specific_impl_fn.b91)]
// CHECK:STDOUT:     %bound_method.loc32_11.1: <bound method> = bound_method %r.ref.loc32, %specific_impl_fn.loc32_4.1
// CHECK:STDOUT:     %.loc32_11.1: ref @Test.%impl.elem0.1 (%impl.elem0.73eb7f.2) = temporary_storage
// CHECK:STDOUT:     %CppRangeForIterate.WithSelf.Begin.call: init @Test.%impl.elem0.1 (%impl.elem0.73eb7f.2) to %.loc32_11.1 = call %bound_method.loc32_11.1(%r.ref.loc32)
// CHECK:STDOUT:     %.loc32_11.2: ref @Test.%impl.elem0.1 (%impl.elem0.73eb7f.2) = temporary %.loc32_11.1, %CppRangeForIterate.WithSelf.Begin.call
// CHECK:STDOUT:     %r.ref.loc33: ref @Test.%R.as_type.loc30_113.1 (%R.as_type) = name_ref r, %r
// CHECK:STDOUT:     %End.ref: %CppRangeForIterate.assoc_type = name_ref End, imports.%Core.import_ref.5b4 [concrete = constants.%assoc3]
// CHECK:STDOUT:     %impl.elem3.loc33_4.1: @Test.%.loc33_4 (%.fbd) = impl_witness_access constants.%CppRangeForIterate.lookup_impl_witness.7e2712.2, element3 [symbolic = %impl.elem3.loc33_4.2 (constants.%impl.elem3)]
// CHECK:STDOUT:     %bound_method.loc33_4: <bound method> = bound_method %r.ref.loc33, %impl.elem3.loc33_4.1
// CHECK:STDOUT:     %specific_impl_fn.loc33_4.1: <specific function> = specific_impl_function %impl.elem3.loc33_4.1, @CppRangeForIterate.WithSelf.End(constants.%CppRangeForIterate.facet.9d6) [symbolic = %specific_impl_fn.loc33_4.2 (constants.%specific_impl_fn.2ec)]
// CHECK:STDOUT:     %bound_method.loc33_9.1: <bound method> = bound_method %r.ref.loc33, %specific_impl_fn.loc33_4.1
// CHECK:STDOUT:     %.loc33_9.1: ref @Test.%impl.elem1.1 (%impl.elem1.7a0bff.2) = temporary_storage
// CHECK:STDOUT:     %CppRangeForIterate.WithSelf.End.call: init @Test.%impl.elem1.1 (%impl.elem1.7a0bff.2) to %.loc33_9.1 = call %bound_method.loc33_9.1(%r.ref.loc33)
// CHECK:STDOUT:     %.loc33_9.2: ref @Test.%impl.elem1.1 (%impl.elem1.7a0bff.2) = temporary %.loc33_9.1, %CppRangeForIterate.WithSelf.End.call
// CHECK:STDOUT:     %impl.elem0.loc33_9.1: @Test.%.loc33_9.3 (%.db6) = impl_witness_access constants.%Destroy.lookup_impl_witness.afb, element0 [symbolic = %impl.elem0.loc33_9.2 (constants.%impl.elem0.180)]
// CHECK:STDOUT:     %bound_method.loc33_9.2: <bound method> = bound_method %.loc33_9.2, %impl.elem0.loc33_9.1
// CHECK:STDOUT:     %specific_impl_fn.loc33_9.1: <specific function> = specific_impl_function %impl.elem0.loc33_9.1, @Destroy.WithSelf.Op(constants.%Destroy.facet.63a) [symbolic = %specific_impl_fn.loc33_9.2 (constants.%specific_impl_fn.a08)]
// CHECK:STDOUT:     %bound_method.loc33_9.3: <bound method> = bound_method %.loc33_9.2, %specific_impl_fn.loc33_9.1
// CHECK:STDOUT:     %Destroy.WithSelf.Op.call.loc33: init %empty_tuple.type = call %bound_method.loc33_9.3(%.loc33_9.2)
// CHECK:STDOUT:     %impl.elem0.loc32_11.1: @Test.%.loc32_11.3 (%.51a) = impl_witness_access constants.%Destroy.lookup_impl_witness.75c, element0 [symbolic = %impl.elem0.loc32_11.2 (constants.%impl.elem0.2a2)]
// CHECK:STDOUT:     %bound_method.loc32_11.2: <bound method> = bound_method %.loc32_11.2, %impl.elem0.loc32_11.1
// CHECK:STDOUT:     %specific_impl_fn.loc32_11.1: <specific function> = specific_impl_function %impl.elem0.loc32_11.1, @Destroy.WithSelf.Op(constants.%Destroy.facet.dae) [symbolic = %specific_impl_fn.loc32_11.2 (constants.%specific_impl_fn.842)]
// CHECK:STDOUT:     %bound_method.loc32_11.3: <bound method> = bound_method %.loc32_11.2, %specific_impl_fn.loc32_11.1
// CHECK:STDOUT:     %Destroy.WithSelf.Op.call.loc32: init %empty_tuple.type = call %bound_method.loc32_11.3(%.loc32_11.2)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(constants.%R) {
// CHECK:STDOUT:   %R.loc30_10.1 => constants.%R
// CHECK:STDOUT:   %R.as_type.loc30_113.1 => constants.%R.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.48fa74.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(constants.%facet_value.a34) {
// CHECK:STDOUT:   %R.loc30_10.1 => constants.%facet_value.a34
// CHECK:STDOUT:   %R.as_type.loc30_113.1 => constants.%MutableRange
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.e63
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc30 => constants.%complete_type
// CHECK:STDOUT:   %CppRangeForIterate.lookup_impl_witness => constants.%custom_witness.ed3
// CHECK:STDOUT:   %CppRangeForIterate.facet => constants.%CppRangeForIterate.facet.1e8
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.Begin.type => constants.%CppRangeForIterate.WithSelf.Begin.type.0af
// CHECK:STDOUT:   %.loc32_4 => constants.%.14b
// CHECK:STDOUT:   %impl.elem2.loc32_4.2 => constants.%MutableRange.Begin
// CHECK:STDOUT:   %specific_impl_fn.loc32_4.2 => constants.%MutableRange.Begin
// CHECK:STDOUT:   %impl.elem0.1 => constants.%Iterator.3ee
// CHECK:STDOUT:   %require_complete.1 => constants.%complete_type
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.End.type => constants.%CppRangeForIterate.WithSelf.End.type.47c
// CHECK:STDOUT:   %.loc33_4 => constants.%.d84
// CHECK:STDOUT:   %impl.elem3.loc33_4.2 => constants.%MutableRange.End
// CHECK:STDOUT:   %specific_impl_fn.loc33_4.2 => constants.%MutableRange.End
// CHECK:STDOUT:   %impl.elem1.1 => constants.%Iterator.3ee
// CHECK:STDOUT:   %require_complete.2 => constants.%complete_type
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.loc33 => constants.%custom_witness.955
// CHECK:STDOUT:   %Destroy.facet.loc33 => constants.%Destroy.facet.2de
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.loc33 => constants.%Destroy.WithSelf.Op.type.69b
// CHECK:STDOUT:   %.loc33_9.3 => constants.%.69f
// CHECK:STDOUT:   %impl.elem0.loc33_9.2 => constants.%Iterator.Op.a2a
// CHECK:STDOUT:   %specific_impl_fn.loc33_9.2 => constants.%Iterator.Op.a2a
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.loc32 => constants.%custom_witness.955
// CHECK:STDOUT:   %Destroy.facet.loc32 => constants.%Destroy.facet.2de
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.loc32 => constants.%Destroy.WithSelf.Op.type.69b
// CHECK:STDOUT:   %.loc32_11.3 => constants.%.69f
// CHECK:STDOUT:   %impl.elem0.loc32_11.2 => constants.%Iterator.Op.a2a
// CHECK:STDOUT:   %specific_impl_fn.loc32_11.2 => constants.%Iterator.Op.a2a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(constants.%facet_value.f5b) {
// CHECK:STDOUT:   %R.loc30_10.1 => constants.%facet_value.f5b
// CHECK:STDOUT:   %R.as_type.loc30_113.1 => constants.%ConstRange
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.0ba
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc30 => constants.%complete_type
// CHECK:STDOUT:   %CppRangeForIterate.lookup_impl_witness => constants.%custom_witness.23e
// CHECK:STDOUT:   %CppRangeForIterate.facet => constants.%CppRangeForIterate.facet.ee3
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.Begin.type => constants.%CppRangeForIterate.WithSelf.Begin.type.c00
// CHECK:STDOUT:   %.loc32_4 => constants.%.ebd
// CHECK:STDOUT:   %impl.elem2.loc32_4.2 => constants.%ConstRange.Begin
// CHECK:STDOUT:   %specific_impl_fn.loc32_4.2 => constants.%ConstRange.Begin
// CHECK:STDOUT:   %impl.elem0.1 => constants.%Iterator.260
// CHECK:STDOUT:   %require_complete.1 => constants.%complete_type
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.End.type => constants.%CppRangeForIterate.WithSelf.End.type.b02
// CHECK:STDOUT:   %.loc33_4 => constants.%.066
// CHECK:STDOUT:   %impl.elem3.loc33_4.2 => constants.%ConstRange.End
// CHECK:STDOUT:   %specific_impl_fn.loc33_4.2 => constants.%ConstRange.End
// CHECK:STDOUT:   %impl.elem1.1 => constants.%Iterator.260
// CHECK:STDOUT:   %require_complete.2 => constants.%complete_type
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.loc33 => constants.%custom_witness.6e8
// CHECK:STDOUT:   %Destroy.facet.loc33 => constants.%Destroy.facet.a30
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.loc33 => constants.%Destroy.WithSelf.Op.type.8b1
// CHECK:STDOUT:   %.loc33_9.3 => constants.%.f73
// CHECK:STDOUT:   %impl.elem0.loc33_9.2 => constants.%Iterator.Op.1c3
// CHECK:STDOUT:   %specific_impl_fn.loc33_9.2 => constants.%Iterator.Op.1c3
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.loc32 => constants.%custom_witness.6e8
// CHECK:STDOUT:   %Destroy.facet.loc32 => constants.%Destroy.facet.a30
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.loc32 => constants.%Destroy.WithSelf.Op.type.8b1
// CHECK:STDOUT:   %.loc32_11.3 => constants.%.f73
// CHECK:STDOUT:   %impl.elem0.loc32_11.2 => constants.%Iterator.Op.1c3
// CHECK:STDOUT:   %specific_impl_fn.loc32_11.2 => constants.%Iterator.Op.1c3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- methods_return_different_types.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %CppRangeForIterate.type: type = facet_type <@CppRangeForIterate> [concrete]
// CHECK:STDOUT:   %Self.39b: %CppRangeForIterate.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.End.type.fe619a.1: type = fn_type @CppRangeForIterate.WithSelf.End, @CppRangeForIterate.WithSelf(%Self.39b) [symbolic]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.End.0f5092.1: %CppRangeForIterate.WithSelf.End.type.fe619a.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.Begin.type.3276c2.1: type = fn_type @CppRangeForIterate.WithSelf.Begin, @CppRangeForIterate.WithSelf(%Self.39b) [symbolic]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.Begin.39882e.1: %CppRangeForIterate.WithSelf.Begin.type.3276c2.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %.Self.6bd: %CppRangeForIterate.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %CppRangeForIterate.assoc_type: type = assoc_entity_type @CppRangeForIterate [concrete]
// CHECK:STDOUT:   %CppRangeForIterate.lookup_impl_witness.143: <witness> = lookup_impl_witness %.Self.6bd, @CppRangeForIterate [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.109: type = impl_witness_access %CppRangeForIterate.lookup_impl_witness.143, element0 [symbolic_self]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %impl.elem1.222: type = impl_witness_access %CppRangeForIterate.lookup_impl_witness.143, element1 [symbolic_self]
// CHECK:STDOUT:   %CppRangeForIterate_where.type: type = facet_type <@CppRangeForIterate where %impl.elem0.109 impls @Destroy and %impl.elem1.222 impls @Destroy> [concrete]
// CHECK:STDOUT:   %R: %CppRangeForIterate_where.type = symbolic_binding R, 0 [symbolic]
// CHECK:STDOUT:   %R.as_type: type = facet_access_type %R [symbolic]
// CHECK:STDOUT:   %pattern_type.48fa74.2: type = pattern_type %R.as_type [symbolic]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %assoc2: %CppRangeForIterate.assoc_type = assoc_entity element2, imports.%Core.import_ref.08d [concrete]
// CHECK:STDOUT:   %CppRangeForIterate.lookup_impl_witness.7e2712.2: <witness> = lookup_impl_witness %R, @CppRangeForIterate [symbolic]
// CHECK:STDOUT:   %CppRangeForIterate.facet.9d6: %CppRangeForIterate.type = facet_value %R.as_type, (%CppRangeForIterate.lookup_impl_witness.7e2712.2) [symbolic]
// CHECK:STDOUT:   %impl.elem0.73eb7f.2: type = impl_witness_access %CppRangeForIterate.lookup_impl_witness.7e2712.2, element0 [symbolic]
// CHECK:STDOUT:   %impl.elem1.7a0bff.2: type = impl_witness_access %CppRangeForIterate.lookup_impl_witness.7e2712.2, element1 [symbolic]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.Begin.type.848: type = fn_type @CppRangeForIterate.WithSelf.Begin, @CppRangeForIterate.WithSelf(%CppRangeForIterate.facet.9d6) [symbolic]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.End.type.2e5: type = fn_type @CppRangeForIterate.WithSelf.End, @CppRangeForIterate.WithSelf(%CppRangeForIterate.facet.9d6) [symbolic]
// CHECK:STDOUT:   %.be2: type = fn_type_with_self_type %CppRangeForIterate.WithSelf.Begin.type.848, %CppRangeForIterate.facet.9d6 [symbolic]
// CHECK:STDOUT:   %impl.elem2: %.be2 = impl_witness_access %CppRangeForIterate.lookup_impl_witness.7e2712.2, element2 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.b91: <specific function> = specific_impl_function %impl.elem2, @CppRangeForIterate.WithSelf.Begin(%CppRangeForIterate.facet.9d6) [symbolic]
// CHECK:STDOUT:   %assoc3: %CppRangeForIterate.assoc_type = assoc_entity element3, imports.%Core.import_ref.49d [concrete]
// CHECK:STDOUT:   %.fbd: type = fn_type_with_self_type %CppRangeForIterate.WithSelf.End.type.2e5, %CppRangeForIterate.facet.9d6 [symbolic]
// CHECK:STDOUT:   %impl.elem3: %.fbd = impl_witness_access %CppRangeForIterate.lookup_impl_witness.7e2712.2, element3 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.2ec: <specific function> = specific_impl_function %impl.elem3, @CppRangeForIterate.WithSelf.End(%CppRangeForIterate.facet.9d6) [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.afb: <witness> = lookup_impl_witness %impl.elem1.7a0bff.2, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.63a: %Destroy.type = facet_value %impl.elem1.7a0bff.2, (%Destroy.lookup_impl_witness.afb) [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.57d: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.63a) [symbolic]
// CHECK:STDOUT:   %.db6: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.57d, %Destroy.facet.63a [symbolic]
// CHECK:STDOUT:   %impl.elem0.180: %.db6 = impl_witness_access %Destroy.lookup_impl_witness.afb, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.a08: <specific function> = specific_impl_function %impl.elem0.180, @Destroy.WithSelf.Op(%Destroy.facet.63a) [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.75c: <witness> = lookup_impl_witness %impl.elem0.73eb7f.2, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.dae: %Destroy.type = facet_value %impl.elem0.73eb7f.2, (%Destroy.lookup_impl_witness.75c) [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.5c3: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.dae) [symbolic]
// CHECK:STDOUT:   %.51a: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.5c3, %Destroy.facet.dae [symbolic]
// CHECK:STDOUT:   %impl.elem0.2a2: %.51a = impl_witness_access %Destroy.lookup_impl_witness.75c, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.842: <specific function> = specific_impl_function %impl.elem0.2a2, @Destroy.WithSelf.Op(%Destroy.facet.dae) [symbolic]
// CHECK:STDOUT:   %MutableRange: type = class_type @MutableRange [concrete]
// CHECK:STDOUT:   %pattern_type.e63: type = pattern_type %MutableRange [concrete]
// CHECK:STDOUT:   %ConstRange: type = class_type @ConstRange [concrete]
// CHECK:STDOUT:   %pattern_type.0ba: type = pattern_type %ConstRange [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %Iterator.3ee: type = class_type @Iterator.1 [concrete]
// CHECK:STDOUT:   %Sentinel.9b2: type = class_type @Sentinel.1 [concrete]
// CHECK:STDOUT:   %MutableRange.Begin.type: type = fn_type @MutableRange.Begin [concrete]
// CHECK:STDOUT:   %MutableRange.Begin: %MutableRange.Begin.type = struct_value () [concrete]
// CHECK:STDOUT:   %MutableRange.End.type: type = fn_type @MutableRange.End [concrete]
// CHECK:STDOUT:   %MutableRange.End: %MutableRange.End.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.6b6: <witness> = custom_witness (%Iterator.3ee, %Sentinel.9b2, %MutableRange.Begin, %MutableRange.End), @CppRangeForIterate [concrete]
// CHECK:STDOUT:   %CppRangeForIterate.facet.502: %CppRangeForIterate.type = facet_value %MutableRange, (%custom_witness.6b6) [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.0ae: type = fn_type @Iterator.Op.1 [concrete]
// CHECK:STDOUT:   %Iterator.Op.a2a: %Iterator.Op.type.0ae = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.95581e.1: <witness> = custom_witness (%Iterator.Op.a2a), @Destroy [concrete]
// CHECK:STDOUT:   %Sentinel.Op.type.0ae: type = fn_type @Sentinel.Op.1 [concrete]
// CHECK:STDOUT:   %Sentinel.Op.a2a: %Sentinel.Op.type.0ae = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.95581e.2: <witness> = custom_witness (%Sentinel.Op.a2a), @Destroy [concrete]
// CHECK:STDOUT:   %facet_value.cdf: %CppRangeForIterate_where.type = facet_value %MutableRange, (%custom_witness.6b6, %custom_witness.95581e.1, %custom_witness.95581e.2) [concrete]
// CHECK:STDOUT:   %Iterator.260: type = class_type @Iterator.2 [concrete]
// CHECK:STDOUT:   %Sentinel.4c4: type = class_type @Sentinel.2 [concrete]
// CHECK:STDOUT:   %ConstRange.Begin.type: type = fn_type @ConstRange.Begin [concrete]
// CHECK:STDOUT:   %ConstRange.Begin: %ConstRange.Begin.type = struct_value () [concrete]
// CHECK:STDOUT:   %ConstRange.End.type: type = fn_type @ConstRange.End [concrete]
// CHECK:STDOUT:   %ConstRange.End: %ConstRange.End.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.a03: <witness> = custom_witness (%Iterator.260, %Sentinel.4c4, %ConstRange.Begin, %ConstRange.End), @CppRangeForIterate [concrete]
// CHECK:STDOUT:   %CppRangeForIterate.facet.117: %CppRangeForIterate.type = facet_value %ConstRange, (%custom_witness.a03) [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.0f3: type = fn_type @Iterator.Op.2 [concrete]
// CHECK:STDOUT:   %Iterator.Op.1c3: %Iterator.Op.type.0f3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.6e8a2e.1: <witness> = custom_witness (%Iterator.Op.1c3), @Destroy [concrete]
// CHECK:STDOUT:   %Sentinel.Op.type.0f3: type = fn_type @Sentinel.Op.2 [concrete]
// CHECK:STDOUT:   %Sentinel.Op.1c3: %Sentinel.Op.type.0f3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.6e8a2e.2: <witness> = custom_witness (%Sentinel.Op.1c3), @Destroy [concrete]
// CHECK:STDOUT:   %facet_value.2a7: %CppRangeForIterate_where.type = facet_value %ConstRange, (%custom_witness.a03, %custom_witness.6e8a2e.1, %custom_witness.6e8a2e.2) [concrete]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.Begin.type.211: type = fn_type @CppRangeForIterate.WithSelf.Begin, @CppRangeForIterate.WithSelf(%CppRangeForIterate.facet.502) [concrete]
// CHECK:STDOUT:   %.7ba: type = fn_type_with_self_type %CppRangeForIterate.WithSelf.Begin.type.211, %CppRangeForIterate.facet.502 [concrete]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.End.type.b32: type = fn_type @CppRangeForIterate.WithSelf.End, @CppRangeForIterate.WithSelf(%CppRangeForIterate.facet.502) [concrete]
// CHECK:STDOUT:   %.195: type = fn_type_with_self_type %CppRangeForIterate.WithSelf.End.type.b32, %CppRangeForIterate.facet.502 [concrete]
// CHECK:STDOUT:   %Destroy.facet.a31: %Destroy.type = facet_value %Sentinel.9b2, (%custom_witness.95581e.2) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.49b: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.a31) [concrete]
// CHECK:STDOUT:   %.3f3: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.49b, %Destroy.facet.a31 [concrete]
// CHECK:STDOUT:   %Destroy.facet.2de: %Destroy.type = facet_value %Iterator.3ee, (%custom_witness.95581e.1) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.69b: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.2de) [concrete]
// CHECK:STDOUT:   %.69f: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.69b, %Destroy.facet.2de [concrete]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.Begin.type.623: type = fn_type @CppRangeForIterate.WithSelf.Begin, @CppRangeForIterate.WithSelf(%CppRangeForIterate.facet.117) [concrete]
// CHECK:STDOUT:   %.274: type = fn_type_with_self_type %CppRangeForIterate.WithSelf.Begin.type.623, %CppRangeForIterate.facet.117 [concrete]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.End.type.e97: type = fn_type @CppRangeForIterate.WithSelf.End, @CppRangeForIterate.WithSelf(%CppRangeForIterate.facet.117) [concrete]
// CHECK:STDOUT:   %.b41: type = fn_type_with_self_type %CppRangeForIterate.WithSelf.End.type.e97, %CppRangeForIterate.facet.117 [concrete]
// CHECK:STDOUT:   %Destroy.facet.e72: %Destroy.type = facet_value %Sentinel.4c4, (%custom_witness.6e8a2e.2) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.af3: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.e72) [concrete]
// CHECK:STDOUT:   %.a51: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.af3, %Destroy.facet.e72 [concrete]
// CHECK:STDOUT:   %Destroy.facet.a30: %Destroy.type = facet_value %Iterator.260, (%custom_witness.6e8a2e.1) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.8b1: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.a30) [concrete]
// CHECK:STDOUT:   %.f73: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.8b1, %Destroy.facet.a30 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.a6a: %CppRangeForIterate.assoc_type = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [concrete = constants.%assoc2]
// CHECK:STDOUT:   %Core.import_ref.5b4: %CppRangeForIterate.assoc_type = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [concrete = constants.%assoc3]
// CHECK:STDOUT:   %Core.import_ref.08d: @CppRangeForIterate.WithSelf.%CppRangeForIterate.WithSelf.Begin.type (%CppRangeForIterate.WithSelf.Begin.type.3276c2.1) = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @CppRangeForIterate.WithSelf.%CppRangeForIterate.WithSelf.Begin (constants.%CppRangeForIterate.WithSelf.Begin.39882e.1)]
// CHECK:STDOUT:   %Core.import_ref.49d: @CppRangeForIterate.WithSelf.%CppRangeForIterate.WithSelf.End.type (%CppRangeForIterate.WithSelf.End.type.fe619a.1) = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @CppRangeForIterate.WithSelf.%CppRangeForIterate.WithSelf.End (constants.%CppRangeForIterate.WithSelf.End.0f5092.1)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Test(%R.loc40_10.2: %CppRangeForIterate_where.type) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %CppRangeForIterate.lookup_impl_witness: <witness> = lookup_impl_witness %R.loc40_10.1, @CppRangeForIterate [symbolic = %CppRangeForIterate.lookup_impl_witness (constants.%CppRangeForIterate.lookup_impl_witness.7e2712.2)]
// CHECK:STDOUT:   %CppRangeForIterate.facet: %CppRangeForIterate.type = facet_value %R.as_type.loc40_113.1, (%CppRangeForIterate.lookup_impl_witness) [symbolic = %CppRangeForIterate.facet (constants.%CppRangeForIterate.facet.9d6)]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.Begin.type: type = fn_type @CppRangeForIterate.WithSelf.Begin, @CppRangeForIterate.WithSelf(%CppRangeForIterate.facet) [symbolic = %CppRangeForIterate.WithSelf.Begin.type (constants.%CppRangeForIterate.WithSelf.Begin.type.848)]
// CHECK:STDOUT:   %.loc42_4: type = fn_type_with_self_type %CppRangeForIterate.WithSelf.Begin.type, %CppRangeForIterate.facet [symbolic = %.loc42_4 (constants.%.be2)]
// CHECK:STDOUT:   %impl.elem2.loc42_4.2: @Test.%.loc42_4 (%.be2) = impl_witness_access %CppRangeForIterate.lookup_impl_witness, element2 [symbolic = %impl.elem2.loc42_4.2 (constants.%impl.elem2)]
// CHECK:STDOUT:   %specific_impl_fn.loc42_4.2: <specific function> = specific_impl_function %impl.elem2.loc42_4.2, @CppRangeForIterate.WithSelf.Begin(%CppRangeForIterate.facet) [symbolic = %specific_impl_fn.loc42_4.2 (constants.%specific_impl_fn.b91)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.End.type: type = fn_type @CppRangeForIterate.WithSelf.End, @CppRangeForIterate.WithSelf(%CppRangeForIterate.facet) [symbolic = %CppRangeForIterate.WithSelf.End.type (constants.%CppRangeForIterate.WithSelf.End.type.2e5)]
// CHECK:STDOUT:   %.loc43_4: type = fn_type_with_self_type %CppRangeForIterate.WithSelf.End.type, %CppRangeForIterate.facet [symbolic = %.loc43_4 (constants.%.fbd)]
// CHECK:STDOUT:   %impl.elem3.loc43_4.2: @Test.%.loc43_4 (%.fbd) = impl_witness_access %CppRangeForIterate.lookup_impl_witness, element3 [symbolic = %impl.elem3.loc43_4.2 (constants.%impl.elem3)]
// CHECK:STDOUT:   %specific_impl_fn.loc43_4.2: <specific function> = specific_impl_function %impl.elem3.loc43_4.2, @CppRangeForIterate.WithSelf.End(%CppRangeForIterate.facet) [symbolic = %specific_impl_fn.loc43_4.2 (constants.%specific_impl_fn.2ec)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.loc43: <witness> = lookup_impl_witness %impl.elem1.1, @Destroy [symbolic = %Destroy.lookup_impl_witness.loc43 (constants.%Destroy.lookup_impl_witness.afb)]
// CHECK:STDOUT:   %Destroy.facet.loc43: %Destroy.type = facet_value %impl.elem1.1, (%Destroy.lookup_impl_witness.loc43) [symbolic = %Destroy.facet.loc43 (constants.%Destroy.facet.63a)]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.loc43: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.loc43) [symbolic = %Destroy.WithSelf.Op.type.loc43 (constants.%Destroy.WithSelf.Op.type.57d)]
// CHECK:STDOUT:   %.loc43_9.3: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.loc43, %Destroy.facet.loc43 [symbolic = %.loc43_9.3 (constants.%.db6)]
// CHECK:STDOUT:   %impl.elem0.loc43_9.2: @Test.%.loc43_9.3 (%.db6) = impl_witness_access %Destroy.lookup_impl_witness.loc43, element0 [symbolic = %impl.elem0.loc43_9.2 (constants.%impl.elem0.180)]
// CHECK:STDOUT:   %specific_impl_fn.loc43_9.2: <specific function> = specific_impl_function %impl.elem0.loc43_9.2, @Destroy.WithSelf.Op(%Destroy.facet.loc43) [symbolic = %specific_impl_fn.loc43_9.2 (constants.%specific_impl_fn.a08)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.loc42: <witness> = lookup_impl_witness %impl.elem0.1, @Destroy [symbolic = %Destroy.lookup_impl_witness.loc42 (constants.%Destroy.lookup_impl_witness.75c)]
// CHECK:STDOUT:   %Destroy.facet.loc42: %Destroy.type = facet_value %impl.elem0.1, (%Destroy.lookup_impl_witness.loc42) [symbolic = %Destroy.facet.loc42 (constants.%Destroy.facet.dae)]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.loc42: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.loc42) [symbolic = %Destroy.WithSelf.Op.type.loc42 (constants.%Destroy.WithSelf.Op.type.5c3)]
// CHECK:STDOUT:   %.loc42_11.3: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.loc42, %Destroy.facet.loc42 [symbolic = %.loc42_11.3 (constants.%.51a)]
// CHECK:STDOUT:   %impl.elem0.loc42_11.2: @Test.%.loc42_11.3 (%.51a) = impl_witness_access %Destroy.lookup_impl_witness.loc42, element0 [symbolic = %impl.elem0.loc42_11.2 (constants.%impl.elem0.2a2)]
// CHECK:STDOUT:   %specific_impl_fn.loc42_11.2: <specific function> = specific_impl_function %impl.elem0.loc42_11.2, @Destroy.WithSelf.Op(%Destroy.facet.loc42) [symbolic = %specific_impl_fn.loc42_11.2 (constants.%specific_impl_fn.842)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%r.param: ref @Test.%R.as_type.loc40_113.1 (%R.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %r.ref.loc42: ref @Test.%R.as_type.loc40_113.1 (%R.as_type) = name_ref r, %r
// CHECK:STDOUT:     %Begin.ref: %CppRangeForIterate.assoc_type = name_ref Begin, imports.%Core.import_ref.a6a [concrete = constants.%assoc2]
// CHECK:STDOUT:     %impl.elem2.loc42_4.1: @Test.%.loc42_4 (%.be2) = impl_witness_access constants.%CppRangeForIterate.lookup_impl_witness.7e2712.2, element2 [symbolic = %impl.elem2.loc42_4.2 (constants.%impl.elem2)]
// CHECK:STDOUT:     %bound_method.loc42_4: <bound method> = bound_method %r.ref.loc42, %impl.elem2.loc42_4.1
// CHECK:STDOUT:     %specific_impl_fn.loc42_4.1: <specific function> = specific_impl_function %impl.elem2.loc42_4.1, @CppRangeForIterate.WithSelf.Begin(constants.%CppRangeForIterate.facet.9d6) [symbolic = %specific_impl_fn.loc42_4.2 (constants.%specific_impl_fn.b91)]
// CHECK:STDOUT:     %bound_method.loc42_11.1: <bound method> = bound_method %r.ref.loc42, %specific_impl_fn.loc42_4.1
// CHECK:STDOUT:     %.loc42_11.1: ref @Test.%impl.elem0.1 (%impl.elem0.73eb7f.2) = temporary_storage
// CHECK:STDOUT:     %CppRangeForIterate.WithSelf.Begin.call: init @Test.%impl.elem0.1 (%impl.elem0.73eb7f.2) to %.loc42_11.1 = call %bound_method.loc42_11.1(%r.ref.loc42)
// CHECK:STDOUT:     %.loc42_11.2: ref @Test.%impl.elem0.1 (%impl.elem0.73eb7f.2) = temporary %.loc42_11.1, %CppRangeForIterate.WithSelf.Begin.call
// CHECK:STDOUT:     %r.ref.loc43: ref @Test.%R.as_type.loc40_113.1 (%R.as_type) = name_ref r, %r
// CHECK:STDOUT:     %End.ref: %CppRangeForIterate.assoc_type = name_ref End, imports.%Core.import_ref.5b4 [concrete = constants.%assoc3]
// CHECK:STDOUT:     %impl.elem3.loc43_4.1: @Test.%.loc43_4 (%.fbd) = impl_witness_access constants.%CppRangeForIterate.lookup_impl_witness.7e2712.2, element3 [symbolic = %impl.elem3.loc43_4.2 (constants.%impl.elem3)]
// CHECK:STDOUT:     %bound_method.loc43_4: <bound method> = bound_method %r.ref.loc43, %impl.elem3.loc43_4.1
// CHECK:STDOUT:     %specific_impl_fn.loc43_4.1: <specific function> = specific_impl_function %impl.elem3.loc43_4.1, @CppRangeForIterate.WithSelf.End(constants.%CppRangeForIterate.facet.9d6) [symbolic = %specific_impl_fn.loc43_4.2 (constants.%specific_impl_fn.2ec)]
// CHECK:STDOUT:     %bound_method.loc43_9.1: <bound method> = bound_method %r.ref.loc43, %specific_impl_fn.loc43_4.1
// CHECK:STDOUT:     %.loc43_9.1: ref @Test.%impl.elem1.1 (%impl.elem1.7a0bff.2) = temporary_storage
// CHECK:STDOUT:     %CppRangeForIterate.WithSelf.End.call: init @Test.%impl.elem1.1 (%impl.elem1.7a0bff.2) to %.loc43_9.1 = call %bound_method.loc43_9.1(%r.ref.loc43)
// CHECK:STDOUT:     %.loc43_9.2: ref @Test.%impl.elem1.1 (%impl.elem1.7a0bff.2) = temporary %.loc43_9.1, %CppRangeForIterate.WithSelf.End.call
// CHECK:STDOUT:     %impl.elem0.loc43_9.1: @Test.%.loc43_9.3 (%.db6) = impl_witness_access constants.%Destroy.lookup_impl_witness.afb, element0 [symbolic = %impl.elem0.loc43_9.2 (constants.%impl.elem0.180)]
// CHECK:STDOUT:     %bound_method.loc43_9.2: <bound method> = bound_method %.loc43_9.2, %impl.elem0.loc43_9.1
// CHECK:STDOUT:     %specific_impl_fn.loc43_9.1: <specific function> = specific_impl_function %impl.elem0.loc43_9.1, @Destroy.WithSelf.Op(constants.%Destroy.facet.63a) [symbolic = %specific_impl_fn.loc43_9.2 (constants.%specific_impl_fn.a08)]
// CHECK:STDOUT:     %bound_method.loc43_9.3: <bound method> = bound_method %.loc43_9.2, %specific_impl_fn.loc43_9.1
// CHECK:STDOUT:     %Destroy.WithSelf.Op.call.loc43: init %empty_tuple.type = call %bound_method.loc43_9.3(%.loc43_9.2)
// CHECK:STDOUT:     %impl.elem0.loc42_11.1: @Test.%.loc42_11.3 (%.51a) = impl_witness_access constants.%Destroy.lookup_impl_witness.75c, element0 [symbolic = %impl.elem0.loc42_11.2 (constants.%impl.elem0.2a2)]
// CHECK:STDOUT:     %bound_method.loc42_11.2: <bound method> = bound_method %.loc42_11.2, %impl.elem0.loc42_11.1
// CHECK:STDOUT:     %specific_impl_fn.loc42_11.1: <specific function> = specific_impl_function %impl.elem0.loc42_11.1, @Destroy.WithSelf.Op(constants.%Destroy.facet.dae) [symbolic = %specific_impl_fn.loc42_11.2 (constants.%specific_impl_fn.842)]
// CHECK:STDOUT:     %bound_method.loc42_11.3: <bound method> = bound_method %.loc42_11.2, %specific_impl_fn.loc42_11.1
// CHECK:STDOUT:     %Destroy.WithSelf.Op.call.loc42: init %empty_tuple.type = call %bound_method.loc42_11.3(%.loc42_11.2)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(constants.%R) {
// CHECK:STDOUT:   %R.loc40_10.1 => constants.%R
// CHECK:STDOUT:   %R.as_type.loc40_113.1 => constants.%R.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.48fa74.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(constants.%facet_value.cdf) {
// CHECK:STDOUT:   %R.loc40_10.1 => constants.%facet_value.cdf
// CHECK:STDOUT:   %R.as_type.loc40_113.1 => constants.%MutableRange
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.e63
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc40 => constants.%complete_type
// CHECK:STDOUT:   %CppRangeForIterate.lookup_impl_witness => constants.%custom_witness.6b6
// CHECK:STDOUT:   %CppRangeForIterate.facet => constants.%CppRangeForIterate.facet.502
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.Begin.type => constants.%CppRangeForIterate.WithSelf.Begin.type.211
// CHECK:STDOUT:   %.loc42_4 => constants.%.7ba
// CHECK:STDOUT:   %impl.elem2.loc42_4.2 => constants.%MutableRange.Begin
// CHECK:STDOUT:   %specific_impl_fn.loc42_4.2 => constants.%MutableRange.Begin
// CHECK:STDOUT:   %impl.elem0.1 => constants.%Iterator.3ee
// CHECK:STDOUT:   %require_complete.1 => constants.%complete_type
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.End.type => constants.%CppRangeForIterate.WithSelf.End.type.b32
// CHECK:STDOUT:   %.loc43_4 => constants.%.195
// CHECK:STDOUT:   %impl.elem3.loc43_4.2 => constants.%MutableRange.End
// CHECK:STDOUT:   %specific_impl_fn.loc43_4.2 => constants.%MutableRange.End
// CHECK:STDOUT:   %impl.elem1.1 => constants.%Sentinel.9b2
// CHECK:STDOUT:   %require_complete.2 => constants.%complete_type
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.loc43 => constants.%custom_witness.95581e.2
// CHECK:STDOUT:   %Destroy.facet.loc43 => constants.%Destroy.facet.a31
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.loc43 => constants.%Destroy.WithSelf.Op.type.49b
// CHECK:STDOUT:   %.loc43_9.3 => constants.%.3f3
// CHECK:STDOUT:   %impl.elem0.loc43_9.2 => constants.%Sentinel.Op.a2a
// CHECK:STDOUT:   %specific_impl_fn.loc43_9.2 => constants.%Sentinel.Op.a2a
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.loc42 => constants.%custom_witness.95581e.1
// CHECK:STDOUT:   %Destroy.facet.loc42 => constants.%Destroy.facet.2de
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.loc42 => constants.%Destroy.WithSelf.Op.type.69b
// CHECK:STDOUT:   %.loc42_11.3 => constants.%.69f
// CHECK:STDOUT:   %impl.elem0.loc42_11.2 => constants.%Iterator.Op.a2a
// CHECK:STDOUT:   %specific_impl_fn.loc42_11.2 => constants.%Iterator.Op.a2a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(constants.%facet_value.2a7) {
// CHECK:STDOUT:   %R.loc40_10.1 => constants.%facet_value.2a7
// CHECK:STDOUT:   %R.as_type.loc40_113.1 => constants.%ConstRange
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.0ba
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc40 => constants.%complete_type
// CHECK:STDOUT:   %CppRangeForIterate.lookup_impl_witness => constants.%custom_witness.a03
// CHECK:STDOUT:   %CppRangeForIterate.facet => constants.%CppRangeForIterate.facet.117
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.Begin.type => constants.%CppRangeForIterate.WithSelf.Begin.type.623
// CHECK:STDOUT:   %.loc42_4 => constants.%.274
// CHECK:STDOUT:   %impl.elem2.loc42_4.2 => constants.%ConstRange.Begin
// CHECK:STDOUT:   %specific_impl_fn.loc42_4.2 => constants.%ConstRange.Begin
// CHECK:STDOUT:   %impl.elem0.1 => constants.%Iterator.260
// CHECK:STDOUT:   %require_complete.1 => constants.%complete_type
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.End.type => constants.%CppRangeForIterate.WithSelf.End.type.e97
// CHECK:STDOUT:   %.loc43_4 => constants.%.b41
// CHECK:STDOUT:   %impl.elem3.loc43_4.2 => constants.%ConstRange.End
// CHECK:STDOUT:   %specific_impl_fn.loc43_4.2 => constants.%ConstRange.End
// CHECK:STDOUT:   %impl.elem1.1 => constants.%Sentinel.4c4
// CHECK:STDOUT:   %require_complete.2 => constants.%complete_type
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.loc43 => constants.%custom_witness.6e8a2e.2
// CHECK:STDOUT:   %Destroy.facet.loc43 => constants.%Destroy.facet.e72
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.loc43 => constants.%Destroy.WithSelf.Op.type.af3
// CHECK:STDOUT:   %.loc43_9.3 => constants.%.a51
// CHECK:STDOUT:   %impl.elem0.loc43_9.2 => constants.%Sentinel.Op.1c3
// CHECK:STDOUT:   %specific_impl_fn.loc43_9.2 => constants.%Sentinel.Op.1c3
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.loc42 => constants.%custom_witness.6e8a2e.1
// CHECK:STDOUT:   %Destroy.facet.loc42 => constants.%Destroy.facet.a30
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.loc42 => constants.%Destroy.WithSelf.Op.type.8b1
// CHECK:STDOUT:   %.loc42_11.3 => constants.%.f73
// CHECK:STDOUT:   %impl.elem0.loc42_11.2 => constants.%Iterator.Op.1c3
// CHECK:STDOUT:   %specific_impl_fn.loc42_11.2 => constants.%Iterator.Op.1c3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- adl_return_same_types.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %CppRangeForIterate.type: type = facet_type <@CppRangeForIterate> [concrete]
// CHECK:STDOUT:   %Self.39b: %CppRangeForIterate.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.End.type.fe619a.1: type = fn_type @CppRangeForIterate.WithSelf.End, @CppRangeForIterate.WithSelf(%Self.39b) [symbolic]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.End.0f5092.1: %CppRangeForIterate.WithSelf.End.type.fe619a.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.Begin.type.3276c2.1: type = fn_type @CppRangeForIterate.WithSelf.Begin, @CppRangeForIterate.WithSelf(%Self.39b) [symbolic]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.Begin.39882e.1: %CppRangeForIterate.WithSelf.Begin.type.3276c2.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %.Self.6bd: %CppRangeForIterate.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %CppRangeForIterate.assoc_type: type = assoc_entity_type @CppRangeForIterate [concrete]
// CHECK:STDOUT:   %CppRangeForIterate.lookup_impl_witness.143: <witness> = lookup_impl_witness %.Self.6bd, @CppRangeForIterate [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.109: type = impl_witness_access %CppRangeForIterate.lookup_impl_witness.143, element0 [symbolic_self]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %impl.elem1.222: type = impl_witness_access %CppRangeForIterate.lookup_impl_witness.143, element1 [symbolic_self]
// CHECK:STDOUT:   %CppRangeForIterate_where.type: type = facet_type <@CppRangeForIterate where %impl.elem0.109 impls @Destroy and %impl.elem1.222 impls @Destroy> [concrete]
// CHECK:STDOUT:   %R: %CppRangeForIterate_where.type = symbolic_binding R, 0 [symbolic]
// CHECK:STDOUT:   %R.as_type: type = facet_access_type %R [symbolic]
// CHECK:STDOUT:   %pattern_type.48fa74.2: type = pattern_type %R.as_type [symbolic]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %assoc2: %CppRangeForIterate.assoc_type = assoc_entity element2, imports.%Core.import_ref.08d [concrete]
// CHECK:STDOUT:   %CppRangeForIterate.lookup_impl_witness.7e2712.2: <witness> = lookup_impl_witness %R, @CppRangeForIterate [symbolic]
// CHECK:STDOUT:   %CppRangeForIterate.facet.9d6: %CppRangeForIterate.type = facet_value %R.as_type, (%CppRangeForIterate.lookup_impl_witness.7e2712.2) [symbolic]
// CHECK:STDOUT:   %impl.elem0.73eb7f.2: type = impl_witness_access %CppRangeForIterate.lookup_impl_witness.7e2712.2, element0 [symbolic]
// CHECK:STDOUT:   %impl.elem1.7a0bff.2: type = impl_witness_access %CppRangeForIterate.lookup_impl_witness.7e2712.2, element1 [symbolic]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.Begin.type.848: type = fn_type @CppRangeForIterate.WithSelf.Begin, @CppRangeForIterate.WithSelf(%CppRangeForIterate.facet.9d6) [symbolic]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.End.type.2e5: type = fn_type @CppRangeForIterate.WithSelf.End, @CppRangeForIterate.WithSelf(%CppRangeForIterate.facet.9d6) [symbolic]
// CHECK:STDOUT:   %.be2: type = fn_type_with_self_type %CppRangeForIterate.WithSelf.Begin.type.848, %CppRangeForIterate.facet.9d6 [symbolic]
// CHECK:STDOUT:   %impl.elem2: %.be2 = impl_witness_access %CppRangeForIterate.lookup_impl_witness.7e2712.2, element2 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.b91: <specific function> = specific_impl_function %impl.elem2, @CppRangeForIterate.WithSelf.Begin(%CppRangeForIterate.facet.9d6) [symbolic]
// CHECK:STDOUT:   %assoc3: %CppRangeForIterate.assoc_type = assoc_entity element3, imports.%Core.import_ref.49d [concrete]
// CHECK:STDOUT:   %.fbd: type = fn_type_with_self_type %CppRangeForIterate.WithSelf.End.type.2e5, %CppRangeForIterate.facet.9d6 [symbolic]
// CHECK:STDOUT:   %impl.elem3: %.fbd = impl_witness_access %CppRangeForIterate.lookup_impl_witness.7e2712.2, element3 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.2ec: <specific function> = specific_impl_function %impl.elem3, @CppRangeForIterate.WithSelf.End(%CppRangeForIterate.facet.9d6) [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.afb: <witness> = lookup_impl_witness %impl.elem1.7a0bff.2, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.63a: %Destroy.type = facet_value %impl.elem1.7a0bff.2, (%Destroy.lookup_impl_witness.afb) [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.57d: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.63a) [symbolic]
// CHECK:STDOUT:   %.db6: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.57d, %Destroy.facet.63a [symbolic]
// CHECK:STDOUT:   %impl.elem0.180: %.db6 = impl_witness_access %Destroy.lookup_impl_witness.afb, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.a08: <specific function> = specific_impl_function %impl.elem0.180, @Destroy.WithSelf.Op(%Destroy.facet.63a) [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.75c: <witness> = lookup_impl_witness %impl.elem0.73eb7f.2, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.dae: %Destroy.type = facet_value %impl.elem0.73eb7f.2, (%Destroy.lookup_impl_witness.75c) [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.5c3: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.dae) [symbolic]
// CHECK:STDOUT:   %.51a: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.5c3, %Destroy.facet.dae [symbolic]
// CHECK:STDOUT:   %impl.elem0.2a2: %.51a = impl_witness_access %Destroy.lookup_impl_witness.75c, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.842: <specific function> = specific_impl_function %impl.elem0.2a2, @Destroy.WithSelf.Op(%Destroy.facet.dae) [symbolic]
// CHECK:STDOUT:   %MutableRange: type = class_type @MutableRange [concrete]
// CHECK:STDOUT:   %pattern_type.992: type = pattern_type %MutableRange [concrete]
// CHECK:STDOUT:   %ConstRange: type = class_type @ConstRange [concrete]
// CHECK:STDOUT:   %pattern_type.025: type = pattern_type %ConstRange [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %Iterator.861: type = class_type @Iterator.1 [concrete]
// CHECK:STDOUT:   %Begin.type.4ad279.1: type = fn_type @Begin.1 [concrete]
// CHECK:STDOUT:   %Begin.f54f5f.1: %Begin.type.4ad279.1 = struct_value () [concrete]
// CHECK:STDOUT:   %End.type.f18d8e.1: type = fn_type @End.1 [concrete]
// CHECK:STDOUT:   %End.033e51.1: %End.type.f18d8e.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.902: <witness> = custom_witness (%Iterator.861, %Iterator.861, %Begin.f54f5f.1, %End.033e51.1), @CppRangeForIterate [concrete]
// CHECK:STDOUT:   %CppRangeForIterate.facet.199: %CppRangeForIterate.type = facet_value %MutableRange, (%custom_witness.902) [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.b97: type = fn_type @Iterator.Op.1 [concrete]
// CHECK:STDOUT:   %Iterator.Op.ac6: %Iterator.Op.type.b97 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.c08: <witness> = custom_witness (%Iterator.Op.ac6), @Destroy [concrete]
// CHECK:STDOUT:   %facet_value.703: %CppRangeForIterate_where.type = facet_value %MutableRange, (%custom_witness.902, %custom_witness.c08) [concrete]
// CHECK:STDOUT:   %Iterator.a00: type = class_type @Iterator.2 [concrete]
// CHECK:STDOUT:   %Begin.type.4ad279.2: type = fn_type @Begin.2 [concrete]
// CHECK:STDOUT:   %Begin.f54f5f.2: %Begin.type.4ad279.2 = struct_value () [concrete]
// CHECK:STDOUT:   %End.type.f18d8e.2: type = fn_type @End.2 [concrete]
// CHECK:STDOUT:   %End.033e51.2: %End.type.f18d8e.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.5d5: <witness> = custom_witness (%Iterator.a00, %Iterator.a00, %Begin.f54f5f.2, %End.033e51.2), @CppRangeForIterate [concrete]
// CHECK:STDOUT:   %CppRangeForIterate.facet.261: %CppRangeForIterate.type = facet_value %ConstRange, (%custom_witness.5d5) [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.ac7: type = fn_type @Iterator.Op.2 [concrete]
// CHECK:STDOUT:   %Iterator.Op.583: %Iterator.Op.type.ac7 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.213: <witness> = custom_witness (%Iterator.Op.583), @Destroy [concrete]
// CHECK:STDOUT:   %facet_value.94d: %CppRangeForIterate_where.type = facet_value %ConstRange, (%custom_witness.5d5, %custom_witness.213) [concrete]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.Begin.type.414: type = fn_type @CppRangeForIterate.WithSelf.Begin, @CppRangeForIterate.WithSelf(%CppRangeForIterate.facet.199) [concrete]
// CHECK:STDOUT:   %.976: type = fn_type_with_self_type %CppRangeForIterate.WithSelf.Begin.type.414, %CppRangeForIterate.facet.199 [concrete]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.End.type.a1a: type = fn_type @CppRangeForIterate.WithSelf.End, @CppRangeForIterate.WithSelf(%CppRangeForIterate.facet.199) [concrete]
// CHECK:STDOUT:   %.43e: type = fn_type_with_self_type %CppRangeForIterate.WithSelf.End.type.a1a, %CppRangeForIterate.facet.199 [concrete]
// CHECK:STDOUT:   %Destroy.facet.a62: %Destroy.type = facet_value %Iterator.861, (%custom_witness.c08) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.714: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.a62) [concrete]
// CHECK:STDOUT:   %.aab: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.714, %Destroy.facet.a62 [concrete]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.Begin.type.336: type = fn_type @CppRangeForIterate.WithSelf.Begin, @CppRangeForIterate.WithSelf(%CppRangeForIterate.facet.261) [concrete]
// CHECK:STDOUT:   %.264: type = fn_type_with_self_type %CppRangeForIterate.WithSelf.Begin.type.336, %CppRangeForIterate.facet.261 [concrete]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.End.type.4ff: type = fn_type @CppRangeForIterate.WithSelf.End, @CppRangeForIterate.WithSelf(%CppRangeForIterate.facet.261) [concrete]
// CHECK:STDOUT:   %.199: type = fn_type_with_self_type %CppRangeForIterate.WithSelf.End.type.4ff, %CppRangeForIterate.facet.261 [concrete]
// CHECK:STDOUT:   %Destroy.facet.6bc: %Destroy.type = facet_value %Iterator.a00, (%custom_witness.213) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.8bf6: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.6bc) [concrete]
// CHECK:STDOUT:   %.198: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.8bf6, %Destroy.facet.6bc [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.a6a: %CppRangeForIterate.assoc_type = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [concrete = constants.%assoc2]
// CHECK:STDOUT:   %Core.import_ref.5b4: %CppRangeForIterate.assoc_type = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [concrete = constants.%assoc3]
// CHECK:STDOUT:   %Core.import_ref.08d: @CppRangeForIterate.WithSelf.%CppRangeForIterate.WithSelf.Begin.type (%CppRangeForIterate.WithSelf.Begin.type.3276c2.1) = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @CppRangeForIterate.WithSelf.%CppRangeForIterate.WithSelf.Begin (constants.%CppRangeForIterate.WithSelf.Begin.39882e.1)]
// CHECK:STDOUT:   %Core.import_ref.49d: @CppRangeForIterate.WithSelf.%CppRangeForIterate.WithSelf.End.type (%CppRangeForIterate.WithSelf.End.type.fe619a.1) = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @CppRangeForIterate.WithSelf.%CppRangeForIterate.WithSelf.End (constants.%CppRangeForIterate.WithSelf.End.0f5092.1)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Test(%R.loc32_10.2: %CppRangeForIterate_where.type) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %CppRangeForIterate.lookup_impl_witness: <witness> = lookup_impl_witness %R.loc32_10.1, @CppRangeForIterate [symbolic = %CppRangeForIterate.lookup_impl_witness (constants.%CppRangeForIterate.lookup_impl_witness.7e2712.2)]
// CHECK:STDOUT:   %CppRangeForIterate.facet: %CppRangeForIterate.type = facet_value %R.as_type.loc32_113.1, (%CppRangeForIterate.lookup_impl_witness) [symbolic = %CppRangeForIterate.facet (constants.%CppRangeForIterate.facet.9d6)]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.Begin.type: type = fn_type @CppRangeForIterate.WithSelf.Begin, @CppRangeForIterate.WithSelf(%CppRangeForIterate.facet) [symbolic = %CppRangeForIterate.WithSelf.Begin.type (constants.%CppRangeForIterate.WithSelf.Begin.type.848)]
// CHECK:STDOUT:   %.loc34_4: type = fn_type_with_self_type %CppRangeForIterate.WithSelf.Begin.type, %CppRangeForIterate.facet [symbolic = %.loc34_4 (constants.%.be2)]
// CHECK:STDOUT:   %impl.elem2.loc34_4.2: @Test.%.loc34_4 (%.be2) = impl_witness_access %CppRangeForIterate.lookup_impl_witness, element2 [symbolic = %impl.elem2.loc34_4.2 (constants.%impl.elem2)]
// CHECK:STDOUT:   %specific_impl_fn.loc34_4.2: <specific function> = specific_impl_function %impl.elem2.loc34_4.2, @CppRangeForIterate.WithSelf.Begin(%CppRangeForIterate.facet) [symbolic = %specific_impl_fn.loc34_4.2 (constants.%specific_impl_fn.b91)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.End.type: type = fn_type @CppRangeForIterate.WithSelf.End, @CppRangeForIterate.WithSelf(%CppRangeForIterate.facet) [symbolic = %CppRangeForIterate.WithSelf.End.type (constants.%CppRangeForIterate.WithSelf.End.type.2e5)]
// CHECK:STDOUT:   %.loc35_4: type = fn_type_with_self_type %CppRangeForIterate.WithSelf.End.type, %CppRangeForIterate.facet [symbolic = %.loc35_4 (constants.%.fbd)]
// CHECK:STDOUT:   %impl.elem3.loc35_4.2: @Test.%.loc35_4 (%.fbd) = impl_witness_access %CppRangeForIterate.lookup_impl_witness, element3 [symbolic = %impl.elem3.loc35_4.2 (constants.%impl.elem3)]
// CHECK:STDOUT:   %specific_impl_fn.loc35_4.2: <specific function> = specific_impl_function %impl.elem3.loc35_4.2, @CppRangeForIterate.WithSelf.End(%CppRangeForIterate.facet) [symbolic = %specific_impl_fn.loc35_4.2 (constants.%specific_impl_fn.2ec)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.loc35: <witness> = lookup_impl_witness %impl.elem1.1, @Destroy [symbolic = %Destroy.lookup_impl_witness.loc35 (constants.%Destroy.lookup_impl_witness.afb)]
// CHECK:STDOUT:   %Destroy.facet.loc35: %Destroy.type = facet_value %impl.elem1.1, (%Destroy.lookup_impl_witness.loc35) [symbolic = %Destroy.facet.loc35 (constants.%Destroy.facet.63a)]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.loc35: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.loc35) [symbolic = %Destroy.WithSelf.Op.type.loc35 (constants.%Destroy.WithSelf.Op.type.57d)]
// CHECK:STDOUT:   %.loc35_9.3: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.loc35, %Destroy.facet.loc35 [symbolic = %.loc35_9.3 (constants.%.db6)]
// CHECK:STDOUT:   %impl.elem0.loc35_9.2: @Test.%.loc35_9.3 (%.db6) = impl_witness_access %Destroy.lookup_impl_witness.loc35, element0 [symbolic = %impl.elem0.loc35_9.2 (constants.%impl.elem0.180)]
// CHECK:STDOUT:   %specific_impl_fn.loc35_9.2: <specific function> = specific_impl_function %impl.elem0.loc35_9.2, @Destroy.WithSelf.Op(%Destroy.facet.loc35) [symbolic = %specific_impl_fn.loc35_9.2 (constants.%specific_impl_fn.a08)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.loc34: <witness> = lookup_impl_witness %impl.elem0.1, @Destroy [symbolic = %Destroy.lookup_impl_witness.loc34 (constants.%Destroy.lookup_impl_witness.75c)]
// CHECK:STDOUT:   %Destroy.facet.loc34: %Destroy.type = facet_value %impl.elem0.1, (%Destroy.lookup_impl_witness.loc34) [symbolic = %Destroy.facet.loc34 (constants.%Destroy.facet.dae)]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.loc34: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.loc34) [symbolic = %Destroy.WithSelf.Op.type.loc34 (constants.%Destroy.WithSelf.Op.type.5c3)]
// CHECK:STDOUT:   %.loc34_11.3: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.loc34, %Destroy.facet.loc34 [symbolic = %.loc34_11.3 (constants.%.51a)]
// CHECK:STDOUT:   %impl.elem0.loc34_11.2: @Test.%.loc34_11.3 (%.51a) = impl_witness_access %Destroy.lookup_impl_witness.loc34, element0 [symbolic = %impl.elem0.loc34_11.2 (constants.%impl.elem0.2a2)]
// CHECK:STDOUT:   %specific_impl_fn.loc34_11.2: <specific function> = specific_impl_function %impl.elem0.loc34_11.2, @Destroy.WithSelf.Op(%Destroy.facet.loc34) [symbolic = %specific_impl_fn.loc34_11.2 (constants.%specific_impl_fn.842)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%r.param: ref @Test.%R.as_type.loc32_113.1 (%R.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %r.ref.loc34: ref @Test.%R.as_type.loc32_113.1 (%R.as_type) = name_ref r, %r
// CHECK:STDOUT:     %Begin.ref: %CppRangeForIterate.assoc_type = name_ref Begin, imports.%Core.import_ref.a6a [concrete = constants.%assoc2]
// CHECK:STDOUT:     %impl.elem2.loc34_4.1: @Test.%.loc34_4 (%.be2) = impl_witness_access constants.%CppRangeForIterate.lookup_impl_witness.7e2712.2, element2 [symbolic = %impl.elem2.loc34_4.2 (constants.%impl.elem2)]
// CHECK:STDOUT:     %bound_method.loc34_4: <bound method> = bound_method %r.ref.loc34, %impl.elem2.loc34_4.1
// CHECK:STDOUT:     %specific_impl_fn.loc34_4.1: <specific function> = specific_impl_function %impl.elem2.loc34_4.1, @CppRangeForIterate.WithSelf.Begin(constants.%CppRangeForIterate.facet.9d6) [symbolic = %specific_impl_fn.loc34_4.2 (constants.%specific_impl_fn.b91)]
// CHECK:STDOUT:     %bound_method.loc34_11.1: <bound method> = bound_method %r.ref.loc34, %specific_impl_fn.loc34_4.1
// CHECK:STDOUT:     %.loc34_11.1: ref @Test.%impl.elem0.1 (%impl.elem0.73eb7f.2) = temporary_storage
// CHECK:STDOUT:     %CppRangeForIterate.WithSelf.Begin.call: init @Test.%impl.elem0.1 (%impl.elem0.73eb7f.2) to %.loc34_11.1 = call %bound_method.loc34_11.1(%r.ref.loc34)
// CHECK:STDOUT:     %.loc34_11.2: ref @Test.%impl.elem0.1 (%impl.elem0.73eb7f.2) = temporary %.loc34_11.1, %CppRangeForIterate.WithSelf.Begin.call
// CHECK:STDOUT:     %r.ref.loc35: ref @Test.%R.as_type.loc32_113.1 (%R.as_type) = name_ref r, %r
// CHECK:STDOUT:     %End.ref: %CppRangeForIterate.assoc_type = name_ref End, imports.%Core.import_ref.5b4 [concrete = constants.%assoc3]
// CHECK:STDOUT:     %impl.elem3.loc35_4.1: @Test.%.loc35_4 (%.fbd) = impl_witness_access constants.%CppRangeForIterate.lookup_impl_witness.7e2712.2, element3 [symbolic = %impl.elem3.loc35_4.2 (constants.%impl.elem3)]
// CHECK:STDOUT:     %bound_method.loc35_4: <bound method> = bound_method %r.ref.loc35, %impl.elem3.loc35_4.1
// CHECK:STDOUT:     %specific_impl_fn.loc35_4.1: <specific function> = specific_impl_function %impl.elem3.loc35_4.1, @CppRangeForIterate.WithSelf.End(constants.%CppRangeForIterate.facet.9d6) [symbolic = %specific_impl_fn.loc35_4.2 (constants.%specific_impl_fn.2ec)]
// CHECK:STDOUT:     %bound_method.loc35_9.1: <bound method> = bound_method %r.ref.loc35, %specific_impl_fn.loc35_4.1
// CHECK:STDOUT:     %.loc35_9.1: ref @Test.%impl.elem1.1 (%impl.elem1.7a0bff.2) = temporary_storage
// CHECK:STDOUT:     %CppRangeForIterate.WithSelf.End.call: init @Test.%impl.elem1.1 (%impl.elem1.7a0bff.2) to %.loc35_9.1 = call %bound_method.loc35_9.1(%r.ref.loc35)
// CHECK:STDOUT:     %.loc35_9.2: ref @Test.%impl.elem1.1 (%impl.elem1.7a0bff.2) = temporary %.loc35_9.1, %CppRangeForIterate.WithSelf.End.call
// CHECK:STDOUT:     %impl.elem0.loc35_9.1: @Test.%.loc35_9.3 (%.db6) = impl_witness_access constants.%Destroy.lookup_impl_witness.afb, element0 [symbolic = %impl.elem0.loc35_9.2 (constants.%impl.elem0.180)]
// CHECK:STDOUT:     %bound_method.loc35_9.2: <bound method> = bound_method %.loc35_9.2, %impl.elem0.loc35_9.1
// CHECK:STDOUT:     %specific_impl_fn.loc35_9.1: <specific function> = specific_impl_function %impl.elem0.loc35_9.1, @Destroy.WithSelf.Op(constants.%Destroy.facet.63a) [symbolic = %specific_impl_fn.loc35_9.2 (constants.%specific_impl_fn.a08)]
// CHECK:STDOUT:     %bound_method.loc35_9.3: <bound method> = bound_method %.loc35_9.2, %specific_impl_fn.loc35_9.1
// CHECK:STDOUT:     %Destroy.WithSelf.Op.call.loc35: init %empty_tuple.type = call %bound_method.loc35_9.3(%.loc35_9.2)
// CHECK:STDOUT:     %impl.elem0.loc34_11.1: @Test.%.loc34_11.3 (%.51a) = impl_witness_access constants.%Destroy.lookup_impl_witness.75c, element0 [symbolic = %impl.elem0.loc34_11.2 (constants.%impl.elem0.2a2)]
// CHECK:STDOUT:     %bound_method.loc34_11.2: <bound method> = bound_method %.loc34_11.2, %impl.elem0.loc34_11.1
// CHECK:STDOUT:     %specific_impl_fn.loc34_11.1: <specific function> = specific_impl_function %impl.elem0.loc34_11.1, @Destroy.WithSelf.Op(constants.%Destroy.facet.dae) [symbolic = %specific_impl_fn.loc34_11.2 (constants.%specific_impl_fn.842)]
// CHECK:STDOUT:     %bound_method.loc34_11.3: <bound method> = bound_method %.loc34_11.2, %specific_impl_fn.loc34_11.1
// CHECK:STDOUT:     %Destroy.WithSelf.Op.call.loc34: init %empty_tuple.type = call %bound_method.loc34_11.3(%.loc34_11.2)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(constants.%R) {
// CHECK:STDOUT:   %R.loc32_10.1 => constants.%R
// CHECK:STDOUT:   %R.as_type.loc32_113.1 => constants.%R.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.48fa74.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(constants.%facet_value.703) {
// CHECK:STDOUT:   %R.loc32_10.1 => constants.%facet_value.703
// CHECK:STDOUT:   %R.as_type.loc32_113.1 => constants.%MutableRange
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.992
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc32 => constants.%complete_type
// CHECK:STDOUT:   %CppRangeForIterate.lookup_impl_witness => constants.%custom_witness.902
// CHECK:STDOUT:   %CppRangeForIterate.facet => constants.%CppRangeForIterate.facet.199
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.Begin.type => constants.%CppRangeForIterate.WithSelf.Begin.type.414
// CHECK:STDOUT:   %.loc34_4 => constants.%.976
// CHECK:STDOUT:   %impl.elem2.loc34_4.2 => constants.%Begin.f54f5f.1
// CHECK:STDOUT:   %specific_impl_fn.loc34_4.2 => constants.%Begin.f54f5f.1
// CHECK:STDOUT:   %impl.elem0.1 => constants.%Iterator.861
// CHECK:STDOUT:   %require_complete.1 => constants.%complete_type
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.End.type => constants.%CppRangeForIterate.WithSelf.End.type.a1a
// CHECK:STDOUT:   %.loc35_4 => constants.%.43e
// CHECK:STDOUT:   %impl.elem3.loc35_4.2 => constants.%End.033e51.1
// CHECK:STDOUT:   %specific_impl_fn.loc35_4.2 => constants.%End.033e51.1
// CHECK:STDOUT:   %impl.elem1.1 => constants.%Iterator.861
// CHECK:STDOUT:   %require_complete.2 => constants.%complete_type
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.loc35 => constants.%custom_witness.c08
// CHECK:STDOUT:   %Destroy.facet.loc35 => constants.%Destroy.facet.a62
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.loc35 => constants.%Destroy.WithSelf.Op.type.714
// CHECK:STDOUT:   %.loc35_9.3 => constants.%.aab
// CHECK:STDOUT:   %impl.elem0.loc35_9.2 => constants.%Iterator.Op.ac6
// CHECK:STDOUT:   %specific_impl_fn.loc35_9.2 => constants.%Iterator.Op.ac6
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.loc34 => constants.%custom_witness.c08
// CHECK:STDOUT:   %Destroy.facet.loc34 => constants.%Destroy.facet.a62
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.loc34 => constants.%Destroy.WithSelf.Op.type.714
// CHECK:STDOUT:   %.loc34_11.3 => constants.%.aab
// CHECK:STDOUT:   %impl.elem0.loc34_11.2 => constants.%Iterator.Op.ac6
// CHECK:STDOUT:   %specific_impl_fn.loc34_11.2 => constants.%Iterator.Op.ac6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(constants.%facet_value.94d) {
// CHECK:STDOUT:   %R.loc32_10.1 => constants.%facet_value.94d
// CHECK:STDOUT:   %R.as_type.loc32_113.1 => constants.%ConstRange
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.025
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc32 => constants.%complete_type
// CHECK:STDOUT:   %CppRangeForIterate.lookup_impl_witness => constants.%custom_witness.5d5
// CHECK:STDOUT:   %CppRangeForIterate.facet => constants.%CppRangeForIterate.facet.261
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.Begin.type => constants.%CppRangeForIterate.WithSelf.Begin.type.336
// CHECK:STDOUT:   %.loc34_4 => constants.%.264
// CHECK:STDOUT:   %impl.elem2.loc34_4.2 => constants.%Begin.f54f5f.2
// CHECK:STDOUT:   %specific_impl_fn.loc34_4.2 => constants.%Begin.f54f5f.2
// CHECK:STDOUT:   %impl.elem0.1 => constants.%Iterator.a00
// CHECK:STDOUT:   %require_complete.1 => constants.%complete_type
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.End.type => constants.%CppRangeForIterate.WithSelf.End.type.4ff
// CHECK:STDOUT:   %.loc35_4 => constants.%.199
// CHECK:STDOUT:   %impl.elem3.loc35_4.2 => constants.%End.033e51.2
// CHECK:STDOUT:   %specific_impl_fn.loc35_4.2 => constants.%End.033e51.2
// CHECK:STDOUT:   %impl.elem1.1 => constants.%Iterator.a00
// CHECK:STDOUT:   %require_complete.2 => constants.%complete_type
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.loc35 => constants.%custom_witness.213
// CHECK:STDOUT:   %Destroy.facet.loc35 => constants.%Destroy.facet.6bc
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.loc35 => constants.%Destroy.WithSelf.Op.type.8bf6
// CHECK:STDOUT:   %.loc35_9.3 => constants.%.198
// CHECK:STDOUT:   %impl.elem0.loc35_9.2 => constants.%Iterator.Op.583
// CHECK:STDOUT:   %specific_impl_fn.loc35_9.2 => constants.%Iterator.Op.583
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.loc34 => constants.%custom_witness.213
// CHECK:STDOUT:   %Destroy.facet.loc34 => constants.%Destroy.facet.6bc
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.loc34 => constants.%Destroy.WithSelf.Op.type.8bf6
// CHECK:STDOUT:   %.loc34_11.3 => constants.%.198
// CHECK:STDOUT:   %impl.elem0.loc34_11.2 => constants.%Iterator.Op.583
// CHECK:STDOUT:   %specific_impl_fn.loc34_11.2 => constants.%Iterator.Op.583
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- adl_returns_different_types.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %CppRangeForIterate.type: type = facet_type <@CppRangeForIterate> [concrete]
// CHECK:STDOUT:   %Self.39b: %CppRangeForIterate.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.End.type.fe619a.1: type = fn_type @CppRangeForIterate.WithSelf.End, @CppRangeForIterate.WithSelf(%Self.39b) [symbolic]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.End.0f5092.1: %CppRangeForIterate.WithSelf.End.type.fe619a.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.Begin.type.3276c2.1: type = fn_type @CppRangeForIterate.WithSelf.Begin, @CppRangeForIterate.WithSelf(%Self.39b) [symbolic]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.Begin.39882e.1: %CppRangeForIterate.WithSelf.Begin.type.3276c2.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %.Self.6bd: %CppRangeForIterate.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %CppRangeForIterate.assoc_type: type = assoc_entity_type @CppRangeForIterate [concrete]
// CHECK:STDOUT:   %CppRangeForIterate.lookup_impl_witness.143: <witness> = lookup_impl_witness %.Self.6bd, @CppRangeForIterate [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.109: type = impl_witness_access %CppRangeForIterate.lookup_impl_witness.143, element0 [symbolic_self]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %impl.elem1.222: type = impl_witness_access %CppRangeForIterate.lookup_impl_witness.143, element1 [symbolic_self]
// CHECK:STDOUT:   %CppRangeForIterate_where.type: type = facet_type <@CppRangeForIterate where %impl.elem0.109 impls @Destroy and %impl.elem1.222 impls @Destroy> [concrete]
// CHECK:STDOUT:   %R: %CppRangeForIterate_where.type = symbolic_binding R, 0 [symbolic]
// CHECK:STDOUT:   %R.as_type: type = facet_access_type %R [symbolic]
// CHECK:STDOUT:   %pattern_type.48fa74.2: type = pattern_type %R.as_type [symbolic]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %assoc2: %CppRangeForIterate.assoc_type = assoc_entity element2, imports.%Core.import_ref.08d [concrete]
// CHECK:STDOUT:   %CppRangeForIterate.lookup_impl_witness.7e2712.2: <witness> = lookup_impl_witness %R, @CppRangeForIterate [symbolic]
// CHECK:STDOUT:   %CppRangeForIterate.facet.9d6: %CppRangeForIterate.type = facet_value %R.as_type, (%CppRangeForIterate.lookup_impl_witness.7e2712.2) [symbolic]
// CHECK:STDOUT:   %impl.elem0.73eb7f.2: type = impl_witness_access %CppRangeForIterate.lookup_impl_witness.7e2712.2, element0 [symbolic]
// CHECK:STDOUT:   %impl.elem1.7a0bff.2: type = impl_witness_access %CppRangeForIterate.lookup_impl_witness.7e2712.2, element1 [symbolic]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.Begin.type.848: type = fn_type @CppRangeForIterate.WithSelf.Begin, @CppRangeForIterate.WithSelf(%CppRangeForIterate.facet.9d6) [symbolic]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.End.type.2e5: type = fn_type @CppRangeForIterate.WithSelf.End, @CppRangeForIterate.WithSelf(%CppRangeForIterate.facet.9d6) [symbolic]
// CHECK:STDOUT:   %.be2: type = fn_type_with_self_type %CppRangeForIterate.WithSelf.Begin.type.848, %CppRangeForIterate.facet.9d6 [symbolic]
// CHECK:STDOUT:   %impl.elem2: %.be2 = impl_witness_access %CppRangeForIterate.lookup_impl_witness.7e2712.2, element2 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.b91: <specific function> = specific_impl_function %impl.elem2, @CppRangeForIterate.WithSelf.Begin(%CppRangeForIterate.facet.9d6) [symbolic]
// CHECK:STDOUT:   %assoc3: %CppRangeForIterate.assoc_type = assoc_entity element3, imports.%Core.import_ref.49d [concrete]
// CHECK:STDOUT:   %.fbd: type = fn_type_with_self_type %CppRangeForIterate.WithSelf.End.type.2e5, %CppRangeForIterate.facet.9d6 [symbolic]
// CHECK:STDOUT:   %impl.elem3: %.fbd = impl_witness_access %CppRangeForIterate.lookup_impl_witness.7e2712.2, element3 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.2ec: <specific function> = specific_impl_function %impl.elem3, @CppRangeForIterate.WithSelf.End(%CppRangeForIterate.facet.9d6) [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.afb: <witness> = lookup_impl_witness %impl.elem1.7a0bff.2, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.63a: %Destroy.type = facet_value %impl.elem1.7a0bff.2, (%Destroy.lookup_impl_witness.afb) [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.57d: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.63a) [symbolic]
// CHECK:STDOUT:   %.db6: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.57d, %Destroy.facet.63a [symbolic]
// CHECK:STDOUT:   %impl.elem0.180: %.db6 = impl_witness_access %Destroy.lookup_impl_witness.afb, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.a08: <specific function> = specific_impl_function %impl.elem0.180, @Destroy.WithSelf.Op(%Destroy.facet.63a) [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.75c: <witness> = lookup_impl_witness %impl.elem0.73eb7f.2, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.dae: %Destroy.type = facet_value %impl.elem0.73eb7f.2, (%Destroy.lookup_impl_witness.75c) [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.5c3: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.dae) [symbolic]
// CHECK:STDOUT:   %.51a: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.5c3, %Destroy.facet.dae [symbolic]
// CHECK:STDOUT:   %impl.elem0.2a2: %.51a = impl_witness_access %Destroy.lookup_impl_witness.75c, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.842: <specific function> = specific_impl_function %impl.elem0.2a2, @Destroy.WithSelf.Op(%Destroy.facet.dae) [symbolic]
// CHECK:STDOUT:   %MutableRange: type = class_type @MutableRange [concrete]
// CHECK:STDOUT:   %pattern_type.992: type = pattern_type %MutableRange [concrete]
// CHECK:STDOUT:   %ConstRange: type = class_type @ConstRange [concrete]
// CHECK:STDOUT:   %pattern_type.025: type = pattern_type %ConstRange [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %Iterator.861: type = class_type @Iterator.1 [concrete]
// CHECK:STDOUT:   %Sentinel.8e3: type = class_type @Sentinel.1 [concrete]
// CHECK:STDOUT:   %Begin.type.4ad279.1: type = fn_type @Begin.1 [concrete]
// CHECK:STDOUT:   %Begin.f54f5f.1: %Begin.type.4ad279.1 = struct_value () [concrete]
// CHECK:STDOUT:   %End.type.f18d8e.1: type = fn_type @End.1 [concrete]
// CHECK:STDOUT:   %End.033e51.1: %End.type.f18d8e.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.a73: <witness> = custom_witness (%Iterator.861, %Sentinel.8e3, %Begin.f54f5f.1, %End.033e51.1), @CppRangeForIterate [concrete]
// CHECK:STDOUT:   %CppRangeForIterate.facet.8fc: %CppRangeForIterate.type = facet_value %MutableRange, (%custom_witness.a73) [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.b97: type = fn_type @Iterator.Op.1 [concrete]
// CHECK:STDOUT:   %Iterator.Op.ac6: %Iterator.Op.type.b97 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.c08c51.1: <witness> = custom_witness (%Iterator.Op.ac6), @Destroy [concrete]
// CHECK:STDOUT:   %Sentinel.Op.type.b97: type = fn_type @Sentinel.Op.1 [concrete]
// CHECK:STDOUT:   %Sentinel.Op.ac6: %Sentinel.Op.type.b97 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.c08c51.2: <witness> = custom_witness (%Sentinel.Op.ac6), @Destroy [concrete]
// CHECK:STDOUT:   %facet_value.e10: %CppRangeForIterate_where.type = facet_value %MutableRange, (%custom_witness.a73, %custom_witness.c08c51.1, %custom_witness.c08c51.2) [concrete]
// CHECK:STDOUT:   %Iterator.a00: type = class_type @Iterator.2 [concrete]
// CHECK:STDOUT:   %Sentinel.6ef: type = class_type @Sentinel.2 [concrete]
// CHECK:STDOUT:   %Begin.type.4ad279.2: type = fn_type @Begin.2 [concrete]
// CHECK:STDOUT:   %Begin.f54f5f.2: %Begin.type.4ad279.2 = struct_value () [concrete]
// CHECK:STDOUT:   %End.type.f18d8e.2: type = fn_type @End.2 [concrete]
// CHECK:STDOUT:   %End.033e51.2: %End.type.f18d8e.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.c2f: <witness> = custom_witness (%Iterator.a00, %Sentinel.6ef, %Begin.f54f5f.2, %End.033e51.2), @CppRangeForIterate [concrete]
// CHECK:STDOUT:   %CppRangeForIterate.facet.c41: %CppRangeForIterate.type = facet_value %ConstRange, (%custom_witness.c2f) [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.ac7: type = fn_type @Iterator.Op.2 [concrete]
// CHECK:STDOUT:   %Iterator.Op.583: %Iterator.Op.type.ac7 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.21358e.1: <witness> = custom_witness (%Iterator.Op.583), @Destroy [concrete]
// CHECK:STDOUT:   %Sentinel.Op.type.ac7: type = fn_type @Sentinel.Op.2 [concrete]
// CHECK:STDOUT:   %Sentinel.Op.583: %Sentinel.Op.type.ac7 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.21358e.2: <witness> = custom_witness (%Sentinel.Op.583), @Destroy [concrete]
// CHECK:STDOUT:   %facet_value.5fd: %CppRangeForIterate_where.type = facet_value %ConstRange, (%custom_witness.c2f, %custom_witness.21358e.1, %custom_witness.21358e.2) [concrete]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.Begin.type.360: type = fn_type @CppRangeForIterate.WithSelf.Begin, @CppRangeForIterate.WithSelf(%CppRangeForIterate.facet.8fc) [concrete]
// CHECK:STDOUT:   %.e16: type = fn_type_with_self_type %CppRangeForIterate.WithSelf.Begin.type.360, %CppRangeForIterate.facet.8fc [concrete]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.End.type.5df: type = fn_type @CppRangeForIterate.WithSelf.End, @CppRangeForIterate.WithSelf(%CppRangeForIterate.facet.8fc) [concrete]
// CHECK:STDOUT:   %.fda: type = fn_type_with_self_type %CppRangeForIterate.WithSelf.End.type.5df, %CppRangeForIterate.facet.8fc [concrete]
// CHECK:STDOUT:   %Destroy.facet.bf1: %Destroy.type = facet_value %Sentinel.8e3, (%custom_witness.c08c51.2) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.113: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.bf1) [concrete]
// CHECK:STDOUT:   %.9a5: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.113, %Destroy.facet.bf1 [concrete]
// CHECK:STDOUT:   %Destroy.facet.a62: %Destroy.type = facet_value %Iterator.861, (%custom_witness.c08c51.1) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.714: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.a62) [concrete]
// CHECK:STDOUT:   %.aab: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.714, %Destroy.facet.a62 [concrete]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.Begin.type.302: type = fn_type @CppRangeForIterate.WithSelf.Begin, @CppRangeForIterate.WithSelf(%CppRangeForIterate.facet.c41) [concrete]
// CHECK:STDOUT:   %.45e: type = fn_type_with_self_type %CppRangeForIterate.WithSelf.Begin.type.302, %CppRangeForIterate.facet.c41 [concrete]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.End.type.19b: type = fn_type @CppRangeForIterate.WithSelf.End, @CppRangeForIterate.WithSelf(%CppRangeForIterate.facet.c41) [concrete]
// CHECK:STDOUT:   %.7a2: type = fn_type_with_self_type %CppRangeForIterate.WithSelf.End.type.19b, %CppRangeForIterate.facet.c41 [concrete]
// CHECK:STDOUT:   %Destroy.facet.711: %Destroy.type = facet_value %Sentinel.6ef, (%custom_witness.21358e.2) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.0e9: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.711) [concrete]
// CHECK:STDOUT:   %.558: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.0e9, %Destroy.facet.711 [concrete]
// CHECK:STDOUT:   %Destroy.facet.6bc: %Destroy.type = facet_value %Iterator.a00, (%custom_witness.21358e.1) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.8bf6: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.6bc) [concrete]
// CHECK:STDOUT:   %.198: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.8bf6, %Destroy.facet.6bc [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.a6a: %CppRangeForIterate.assoc_type = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [concrete = constants.%assoc2]
// CHECK:STDOUT:   %Core.import_ref.5b4: %CppRangeForIterate.assoc_type = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [concrete = constants.%assoc3]
// CHECK:STDOUT:   %Core.import_ref.08d: @CppRangeForIterate.WithSelf.%CppRangeForIterate.WithSelf.Begin.type (%CppRangeForIterate.WithSelf.Begin.type.3276c2.1) = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @CppRangeForIterate.WithSelf.%CppRangeForIterate.WithSelf.Begin (constants.%CppRangeForIterate.WithSelf.Begin.39882e.1)]
// CHECK:STDOUT:   %Core.import_ref.49d: @CppRangeForIterate.WithSelf.%CppRangeForIterate.WithSelf.End.type (%CppRangeForIterate.WithSelf.End.type.fe619a.1) = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @CppRangeForIterate.WithSelf.%CppRangeForIterate.WithSelf.End (constants.%CppRangeForIterate.WithSelf.End.0f5092.1)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Test(%R.loc42_10.2: %CppRangeForIterate_where.type) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %CppRangeForIterate.lookup_impl_witness: <witness> = lookup_impl_witness %R.loc42_10.1, @CppRangeForIterate [symbolic = %CppRangeForIterate.lookup_impl_witness (constants.%CppRangeForIterate.lookup_impl_witness.7e2712.2)]
// CHECK:STDOUT:   %CppRangeForIterate.facet: %CppRangeForIterate.type = facet_value %R.as_type.loc42_113.1, (%CppRangeForIterate.lookup_impl_witness) [symbolic = %CppRangeForIterate.facet (constants.%CppRangeForIterate.facet.9d6)]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.Begin.type: type = fn_type @CppRangeForIterate.WithSelf.Begin, @CppRangeForIterate.WithSelf(%CppRangeForIterate.facet) [symbolic = %CppRangeForIterate.WithSelf.Begin.type (constants.%CppRangeForIterate.WithSelf.Begin.type.848)]
// CHECK:STDOUT:   %.loc44_4: type = fn_type_with_self_type %CppRangeForIterate.WithSelf.Begin.type, %CppRangeForIterate.facet [symbolic = %.loc44_4 (constants.%.be2)]
// CHECK:STDOUT:   %impl.elem2.loc44_4.2: @Test.%.loc44_4 (%.be2) = impl_witness_access %CppRangeForIterate.lookup_impl_witness, element2 [symbolic = %impl.elem2.loc44_4.2 (constants.%impl.elem2)]
// CHECK:STDOUT:   %specific_impl_fn.loc44_4.2: <specific function> = specific_impl_function %impl.elem2.loc44_4.2, @CppRangeForIterate.WithSelf.Begin(%CppRangeForIterate.facet) [symbolic = %specific_impl_fn.loc44_4.2 (constants.%specific_impl_fn.b91)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.End.type: type = fn_type @CppRangeForIterate.WithSelf.End, @CppRangeForIterate.WithSelf(%CppRangeForIterate.facet) [symbolic = %CppRangeForIterate.WithSelf.End.type (constants.%CppRangeForIterate.WithSelf.End.type.2e5)]
// CHECK:STDOUT:   %.loc45_4: type = fn_type_with_self_type %CppRangeForIterate.WithSelf.End.type, %CppRangeForIterate.facet [symbolic = %.loc45_4 (constants.%.fbd)]
// CHECK:STDOUT:   %impl.elem3.loc45_4.2: @Test.%.loc45_4 (%.fbd) = impl_witness_access %CppRangeForIterate.lookup_impl_witness, element3 [symbolic = %impl.elem3.loc45_4.2 (constants.%impl.elem3)]
// CHECK:STDOUT:   %specific_impl_fn.loc45_4.2: <specific function> = specific_impl_function %impl.elem3.loc45_4.2, @CppRangeForIterate.WithSelf.End(%CppRangeForIterate.facet) [symbolic = %specific_impl_fn.loc45_4.2 (constants.%specific_impl_fn.2ec)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.loc45: <witness> = lookup_impl_witness %impl.elem1.1, @Destroy [symbolic = %Destroy.lookup_impl_witness.loc45 (constants.%Destroy.lookup_impl_witness.afb)]
// CHECK:STDOUT:   %Destroy.facet.loc45: %Destroy.type = facet_value %impl.elem1.1, (%Destroy.lookup_impl_witness.loc45) [symbolic = %Destroy.facet.loc45 (constants.%Destroy.facet.63a)]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.loc45: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.loc45) [symbolic = %Destroy.WithSelf.Op.type.loc45 (constants.%Destroy.WithSelf.Op.type.57d)]
// CHECK:STDOUT:   %.loc45_9.3: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.loc45, %Destroy.facet.loc45 [symbolic = %.loc45_9.3 (constants.%.db6)]
// CHECK:STDOUT:   %impl.elem0.loc45_9.2: @Test.%.loc45_9.3 (%.db6) = impl_witness_access %Destroy.lookup_impl_witness.loc45, element0 [symbolic = %impl.elem0.loc45_9.2 (constants.%impl.elem0.180)]
// CHECK:STDOUT:   %specific_impl_fn.loc45_9.2: <specific function> = specific_impl_function %impl.elem0.loc45_9.2, @Destroy.WithSelf.Op(%Destroy.facet.loc45) [symbolic = %specific_impl_fn.loc45_9.2 (constants.%specific_impl_fn.a08)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.loc44: <witness> = lookup_impl_witness %impl.elem0.1, @Destroy [symbolic = %Destroy.lookup_impl_witness.loc44 (constants.%Destroy.lookup_impl_witness.75c)]
// CHECK:STDOUT:   %Destroy.facet.loc44: %Destroy.type = facet_value %impl.elem0.1, (%Destroy.lookup_impl_witness.loc44) [symbolic = %Destroy.facet.loc44 (constants.%Destroy.facet.dae)]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.loc44: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.loc44) [symbolic = %Destroy.WithSelf.Op.type.loc44 (constants.%Destroy.WithSelf.Op.type.5c3)]
// CHECK:STDOUT:   %.loc44_11.3: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.loc44, %Destroy.facet.loc44 [symbolic = %.loc44_11.3 (constants.%.51a)]
// CHECK:STDOUT:   %impl.elem0.loc44_11.2: @Test.%.loc44_11.3 (%.51a) = impl_witness_access %Destroy.lookup_impl_witness.loc44, element0 [symbolic = %impl.elem0.loc44_11.2 (constants.%impl.elem0.2a2)]
// CHECK:STDOUT:   %specific_impl_fn.loc44_11.2: <specific function> = specific_impl_function %impl.elem0.loc44_11.2, @Destroy.WithSelf.Op(%Destroy.facet.loc44) [symbolic = %specific_impl_fn.loc44_11.2 (constants.%specific_impl_fn.842)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%r.param: ref @Test.%R.as_type.loc42_113.1 (%R.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %r.ref.loc44: ref @Test.%R.as_type.loc42_113.1 (%R.as_type) = name_ref r, %r
// CHECK:STDOUT:     %Begin.ref: %CppRangeForIterate.assoc_type = name_ref Begin, imports.%Core.import_ref.a6a [concrete = constants.%assoc2]
// CHECK:STDOUT:     %impl.elem2.loc44_4.1: @Test.%.loc44_4 (%.be2) = impl_witness_access constants.%CppRangeForIterate.lookup_impl_witness.7e2712.2, element2 [symbolic = %impl.elem2.loc44_4.2 (constants.%impl.elem2)]
// CHECK:STDOUT:     %bound_method.loc44_4: <bound method> = bound_method %r.ref.loc44, %impl.elem2.loc44_4.1
// CHECK:STDOUT:     %specific_impl_fn.loc44_4.1: <specific function> = specific_impl_function %impl.elem2.loc44_4.1, @CppRangeForIterate.WithSelf.Begin(constants.%CppRangeForIterate.facet.9d6) [symbolic = %specific_impl_fn.loc44_4.2 (constants.%specific_impl_fn.b91)]
// CHECK:STDOUT:     %bound_method.loc44_11.1: <bound method> = bound_method %r.ref.loc44, %specific_impl_fn.loc44_4.1
// CHECK:STDOUT:     %.loc44_11.1: ref @Test.%impl.elem0.1 (%impl.elem0.73eb7f.2) = temporary_storage
// CHECK:STDOUT:     %CppRangeForIterate.WithSelf.Begin.call: init @Test.%impl.elem0.1 (%impl.elem0.73eb7f.2) to %.loc44_11.1 = call %bound_method.loc44_11.1(%r.ref.loc44)
// CHECK:STDOUT:     %.loc44_11.2: ref @Test.%impl.elem0.1 (%impl.elem0.73eb7f.2) = temporary %.loc44_11.1, %CppRangeForIterate.WithSelf.Begin.call
// CHECK:STDOUT:     %r.ref.loc45: ref @Test.%R.as_type.loc42_113.1 (%R.as_type) = name_ref r, %r
// CHECK:STDOUT:     %End.ref: %CppRangeForIterate.assoc_type = name_ref End, imports.%Core.import_ref.5b4 [concrete = constants.%assoc3]
// CHECK:STDOUT:     %impl.elem3.loc45_4.1: @Test.%.loc45_4 (%.fbd) = impl_witness_access constants.%CppRangeForIterate.lookup_impl_witness.7e2712.2, element3 [symbolic = %impl.elem3.loc45_4.2 (constants.%impl.elem3)]
// CHECK:STDOUT:     %bound_method.loc45_4: <bound method> = bound_method %r.ref.loc45, %impl.elem3.loc45_4.1
// CHECK:STDOUT:     %specific_impl_fn.loc45_4.1: <specific function> = specific_impl_function %impl.elem3.loc45_4.1, @CppRangeForIterate.WithSelf.End(constants.%CppRangeForIterate.facet.9d6) [symbolic = %specific_impl_fn.loc45_4.2 (constants.%specific_impl_fn.2ec)]
// CHECK:STDOUT:     %bound_method.loc45_9.1: <bound method> = bound_method %r.ref.loc45, %specific_impl_fn.loc45_4.1
// CHECK:STDOUT:     %.loc45_9.1: ref @Test.%impl.elem1.1 (%impl.elem1.7a0bff.2) = temporary_storage
// CHECK:STDOUT:     %CppRangeForIterate.WithSelf.End.call: init @Test.%impl.elem1.1 (%impl.elem1.7a0bff.2) to %.loc45_9.1 = call %bound_method.loc45_9.1(%r.ref.loc45)
// CHECK:STDOUT:     %.loc45_9.2: ref @Test.%impl.elem1.1 (%impl.elem1.7a0bff.2) = temporary %.loc45_9.1, %CppRangeForIterate.WithSelf.End.call
// CHECK:STDOUT:     %impl.elem0.loc45_9.1: @Test.%.loc45_9.3 (%.db6) = impl_witness_access constants.%Destroy.lookup_impl_witness.afb, element0 [symbolic = %impl.elem0.loc45_9.2 (constants.%impl.elem0.180)]
// CHECK:STDOUT:     %bound_method.loc45_9.2: <bound method> = bound_method %.loc45_9.2, %impl.elem0.loc45_9.1
// CHECK:STDOUT:     %specific_impl_fn.loc45_9.1: <specific function> = specific_impl_function %impl.elem0.loc45_9.1, @Destroy.WithSelf.Op(constants.%Destroy.facet.63a) [symbolic = %specific_impl_fn.loc45_9.2 (constants.%specific_impl_fn.a08)]
// CHECK:STDOUT:     %bound_method.loc45_9.3: <bound method> = bound_method %.loc45_9.2, %specific_impl_fn.loc45_9.1
// CHECK:STDOUT:     %Destroy.WithSelf.Op.call.loc45: init %empty_tuple.type = call %bound_method.loc45_9.3(%.loc45_9.2)
// CHECK:STDOUT:     %impl.elem0.loc44_11.1: @Test.%.loc44_11.3 (%.51a) = impl_witness_access constants.%Destroy.lookup_impl_witness.75c, element0 [symbolic = %impl.elem0.loc44_11.2 (constants.%impl.elem0.2a2)]
// CHECK:STDOUT:     %bound_method.loc44_11.2: <bound method> = bound_method %.loc44_11.2, %impl.elem0.loc44_11.1
// CHECK:STDOUT:     %specific_impl_fn.loc44_11.1: <specific function> = specific_impl_function %impl.elem0.loc44_11.1, @Destroy.WithSelf.Op(constants.%Destroy.facet.dae) [symbolic = %specific_impl_fn.loc44_11.2 (constants.%specific_impl_fn.842)]
// CHECK:STDOUT:     %bound_method.loc44_11.3: <bound method> = bound_method %.loc44_11.2, %specific_impl_fn.loc44_11.1
// CHECK:STDOUT:     %Destroy.WithSelf.Op.call.loc44: init %empty_tuple.type = call %bound_method.loc44_11.3(%.loc44_11.2)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(constants.%R) {
// CHECK:STDOUT:   %R.loc42_10.1 => constants.%R
// CHECK:STDOUT:   %R.as_type.loc42_113.1 => constants.%R.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.48fa74.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(constants.%facet_value.e10) {
// CHECK:STDOUT:   %R.loc42_10.1 => constants.%facet_value.e10
// CHECK:STDOUT:   %R.as_type.loc42_113.1 => constants.%MutableRange
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.992
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc42 => constants.%complete_type
// CHECK:STDOUT:   %CppRangeForIterate.lookup_impl_witness => constants.%custom_witness.a73
// CHECK:STDOUT:   %CppRangeForIterate.facet => constants.%CppRangeForIterate.facet.8fc
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.Begin.type => constants.%CppRangeForIterate.WithSelf.Begin.type.360
// CHECK:STDOUT:   %.loc44_4 => constants.%.e16
// CHECK:STDOUT:   %impl.elem2.loc44_4.2 => constants.%Begin.f54f5f.1
// CHECK:STDOUT:   %specific_impl_fn.loc44_4.2 => constants.%Begin.f54f5f.1
// CHECK:STDOUT:   %impl.elem0.1 => constants.%Iterator.861
// CHECK:STDOUT:   %require_complete.1 => constants.%complete_type
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.End.type => constants.%CppRangeForIterate.WithSelf.End.type.5df
// CHECK:STDOUT:   %.loc45_4 => constants.%.fda
// CHECK:STDOUT:   %impl.elem3.loc45_4.2 => constants.%End.033e51.1
// CHECK:STDOUT:   %specific_impl_fn.loc45_4.2 => constants.%End.033e51.1
// CHECK:STDOUT:   %impl.elem1.1 => constants.%Sentinel.8e3
// CHECK:STDOUT:   %require_complete.2 => constants.%complete_type
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.loc45 => constants.%custom_witness.c08c51.2
// CHECK:STDOUT:   %Destroy.facet.loc45 => constants.%Destroy.facet.bf1
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.loc45 => constants.%Destroy.WithSelf.Op.type.113
// CHECK:STDOUT:   %.loc45_9.3 => constants.%.9a5
// CHECK:STDOUT:   %impl.elem0.loc45_9.2 => constants.%Sentinel.Op.ac6
// CHECK:STDOUT:   %specific_impl_fn.loc45_9.2 => constants.%Sentinel.Op.ac6
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.loc44 => constants.%custom_witness.c08c51.1
// CHECK:STDOUT:   %Destroy.facet.loc44 => constants.%Destroy.facet.a62
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.loc44 => constants.%Destroy.WithSelf.Op.type.714
// CHECK:STDOUT:   %.loc44_11.3 => constants.%.aab
// CHECK:STDOUT:   %impl.elem0.loc44_11.2 => constants.%Iterator.Op.ac6
// CHECK:STDOUT:   %specific_impl_fn.loc44_11.2 => constants.%Iterator.Op.ac6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(constants.%facet_value.5fd) {
// CHECK:STDOUT:   %R.loc42_10.1 => constants.%facet_value.5fd
// CHECK:STDOUT:   %R.as_type.loc42_113.1 => constants.%ConstRange
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.025
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc42 => constants.%complete_type
// CHECK:STDOUT:   %CppRangeForIterate.lookup_impl_witness => constants.%custom_witness.c2f
// CHECK:STDOUT:   %CppRangeForIterate.facet => constants.%CppRangeForIterate.facet.c41
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.Begin.type => constants.%CppRangeForIterate.WithSelf.Begin.type.302
// CHECK:STDOUT:   %.loc44_4 => constants.%.45e
// CHECK:STDOUT:   %impl.elem2.loc44_4.2 => constants.%Begin.f54f5f.2
// CHECK:STDOUT:   %specific_impl_fn.loc44_4.2 => constants.%Begin.f54f5f.2
// CHECK:STDOUT:   %impl.elem0.1 => constants.%Iterator.a00
// CHECK:STDOUT:   %require_complete.1 => constants.%complete_type
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.End.type => constants.%CppRangeForIterate.WithSelf.End.type.19b
// CHECK:STDOUT:   %.loc45_4 => constants.%.7a2
// CHECK:STDOUT:   %impl.elem3.loc45_4.2 => constants.%End.033e51.2
// CHECK:STDOUT:   %specific_impl_fn.loc45_4.2 => constants.%End.033e51.2
// CHECK:STDOUT:   %impl.elem1.1 => constants.%Sentinel.6ef
// CHECK:STDOUT:   %require_complete.2 => constants.%complete_type
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.loc45 => constants.%custom_witness.21358e.2
// CHECK:STDOUT:   %Destroy.facet.loc45 => constants.%Destroy.facet.711
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.loc45 => constants.%Destroy.WithSelf.Op.type.0e9
// CHECK:STDOUT:   %.loc45_9.3 => constants.%.558
// CHECK:STDOUT:   %impl.elem0.loc45_9.2 => constants.%Sentinel.Op.583
// CHECK:STDOUT:   %specific_impl_fn.loc45_9.2 => constants.%Sentinel.Op.583
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.loc44 => constants.%custom_witness.21358e.1
// CHECK:STDOUT:   %Destroy.facet.loc44 => constants.%Destroy.facet.6bc
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.loc44 => constants.%Destroy.WithSelf.Op.type.8bf6
// CHECK:STDOUT:   %.loc44_11.3 => constants.%.198
// CHECK:STDOUT:   %impl.elem0.loc44_11.2 => constants.%Iterator.Op.583
// CHECK:STDOUT:   %specific_impl_fn.loc44_11.2 => constants.%Iterator.Op.583
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- adl_overloads.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %CppRangeForIterate.type: type = facet_type <@CppRangeForIterate> [concrete]
// CHECK:STDOUT:   %Self.39b: %CppRangeForIterate.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.End.type.fe619a.1: type = fn_type @CppRangeForIterate.WithSelf.End, @CppRangeForIterate.WithSelf(%Self.39b) [symbolic]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.End.0f5092.1: %CppRangeForIterate.WithSelf.End.type.fe619a.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.Begin.type.3276c2.1: type = fn_type @CppRangeForIterate.WithSelf.Begin, @CppRangeForIterate.WithSelf(%Self.39b) [symbolic]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.Begin.39882e.1: %CppRangeForIterate.WithSelf.Begin.type.3276c2.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %.Self.6bd: %CppRangeForIterate.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %CppRangeForIterate.assoc_type: type = assoc_entity_type @CppRangeForIterate [concrete]
// CHECK:STDOUT:   %CppRangeForIterate.lookup_impl_witness.143: <witness> = lookup_impl_witness %.Self.6bd, @CppRangeForIterate [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.109: type = impl_witness_access %CppRangeForIterate.lookup_impl_witness.143, element0 [symbolic_self]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %impl.elem1.222: type = impl_witness_access %CppRangeForIterate.lookup_impl_witness.143, element1 [symbolic_self]
// CHECK:STDOUT:   %CppRangeForIterate_where.type: type = facet_type <@CppRangeForIterate where %impl.elem0.109 impls @Destroy and %impl.elem1.222 impls @Destroy> [concrete]
// CHECK:STDOUT:   %R: %CppRangeForIterate_where.type = symbolic_binding R, 0 [symbolic]
// CHECK:STDOUT:   %R.as_type: type = facet_access_type %R [symbolic]
// CHECK:STDOUT:   %pattern_type.48fa74.2: type = pattern_type %R.as_type [symbolic]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %assoc2: %CppRangeForIterate.assoc_type = assoc_entity element2, imports.%Core.import_ref.08d [concrete]
// CHECK:STDOUT:   %CppRangeForIterate.lookup_impl_witness.7e2712.2: <witness> = lookup_impl_witness %R, @CppRangeForIterate [symbolic]
// CHECK:STDOUT:   %CppRangeForIterate.facet.9d6: %CppRangeForIterate.type = facet_value %R.as_type, (%CppRangeForIterate.lookup_impl_witness.7e2712.2) [symbolic]
// CHECK:STDOUT:   %impl.elem0.73eb7f.2: type = impl_witness_access %CppRangeForIterate.lookup_impl_witness.7e2712.2, element0 [symbolic]
// CHECK:STDOUT:   %impl.elem1.7a0bff.2: type = impl_witness_access %CppRangeForIterate.lookup_impl_witness.7e2712.2, element1 [symbolic]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.Begin.type.848: type = fn_type @CppRangeForIterate.WithSelf.Begin, @CppRangeForIterate.WithSelf(%CppRangeForIterate.facet.9d6) [symbolic]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.End.type.2e5f: type = fn_type @CppRangeForIterate.WithSelf.End, @CppRangeForIterate.WithSelf(%CppRangeForIterate.facet.9d6) [symbolic]
// CHECK:STDOUT:   %.be2: type = fn_type_with_self_type %CppRangeForIterate.WithSelf.Begin.type.848, %CppRangeForIterate.facet.9d6 [symbolic]
// CHECK:STDOUT:   %impl.elem2: %.be2 = impl_witness_access %CppRangeForIterate.lookup_impl_witness.7e2712.2, element2 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.b91: <specific function> = specific_impl_function %impl.elem2, @CppRangeForIterate.WithSelf.Begin(%CppRangeForIterate.facet.9d6) [symbolic]
// CHECK:STDOUT:   %assoc3: %CppRangeForIterate.assoc_type = assoc_entity element3, imports.%Core.import_ref.49d [concrete]
// CHECK:STDOUT:   %.fbd: type = fn_type_with_self_type %CppRangeForIterate.WithSelf.End.type.2e5f, %CppRangeForIterate.facet.9d6 [symbolic]
// CHECK:STDOUT:   %impl.elem3: %.fbd = impl_witness_access %CppRangeForIterate.lookup_impl_witness.7e2712.2, element3 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.2ec: <specific function> = specific_impl_function %impl.elem3, @CppRangeForIterate.WithSelf.End(%CppRangeForIterate.facet.9d6) [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.afb: <witness> = lookup_impl_witness %impl.elem1.7a0bff.2, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.63a: %Destroy.type = facet_value %impl.elem1.7a0bff.2, (%Destroy.lookup_impl_witness.afb) [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.57d: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.63a) [symbolic]
// CHECK:STDOUT:   %.db6: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.57d, %Destroy.facet.63a [symbolic]
// CHECK:STDOUT:   %impl.elem0.180: %.db6 = impl_witness_access %Destroy.lookup_impl_witness.afb, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.a08: <specific function> = specific_impl_function %impl.elem0.180, @Destroy.WithSelf.Op(%Destroy.facet.63a) [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.75c: <witness> = lookup_impl_witness %impl.elem0.73eb7f.2, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.dae: %Destroy.type = facet_value %impl.elem0.73eb7f.2, (%Destroy.lookup_impl_witness.75c) [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.5c3: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.dae) [symbolic]
// CHECK:STDOUT:   %.51a: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.5c3, %Destroy.facet.dae [symbolic]
// CHECK:STDOUT:   %impl.elem0.2a2: %.51a = impl_witness_access %Destroy.lookup_impl_witness.75c, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.842: <specific function> = specific_impl_function %impl.elem0.2a2, @Destroy.WithSelf.Op(%Destroy.facet.dae) [symbolic]
// CHECK:STDOUT:   %MutableAndConstRange: type = class_type @MutableAndConstRange [concrete]
// CHECK:STDOUT:   %pattern_type.394: type = pattern_type %MutableAndConstRange [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %Iterator: type = class_type @Iterator.1 [concrete]
// CHECK:STDOUT:   %Begin.type: type = fn_type @Begin [concrete]
// CHECK:STDOUT:   %Begin: %Begin.type = struct_value () [concrete]
// CHECK:STDOUT:   %End.type: type = fn_type @End [concrete]
// CHECK:STDOUT:   %End: %End.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.a40: <witness> = custom_witness (%Iterator, %Iterator, %Begin, %End), @CppRangeForIterate [concrete]
// CHECK:STDOUT:   %CppRangeForIterate.facet.c41: %CppRangeForIterate.type = facet_value %MutableAndConstRange, (%custom_witness.a40) [concrete]
// CHECK:STDOUT:   %Iterator.Op.type: type = fn_type @Iterator.Op [concrete]
// CHECK:STDOUT:   %Iterator.Op: %Iterator.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.04a: <witness> = custom_witness (%Iterator.Op), @Destroy [concrete]
// CHECK:STDOUT:   %facet_value: %CppRangeForIterate_where.type = facet_value %MutableAndConstRange, (%custom_witness.a40, %custom_witness.04a) [concrete]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.Begin.type.d70: type = fn_type @CppRangeForIterate.WithSelf.Begin, @CppRangeForIterate.WithSelf(%CppRangeForIterate.facet.c41) [concrete]
// CHECK:STDOUT:   %.a97: type = fn_type_with_self_type %CppRangeForIterate.WithSelf.Begin.type.d70, %CppRangeForIterate.facet.c41 [concrete]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.End.type.e30: type = fn_type @CppRangeForIterate.WithSelf.End, @CppRangeForIterate.WithSelf(%CppRangeForIterate.facet.c41) [concrete]
// CHECK:STDOUT:   %.abe: type = fn_type_with_self_type %CppRangeForIterate.WithSelf.End.type.e30, %CppRangeForIterate.facet.c41 [concrete]
// CHECK:STDOUT:   %Destroy.facet.663: %Destroy.type = facet_value %Iterator, (%custom_witness.04a) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ea3: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.663) [concrete]
// CHECK:STDOUT:   %.57f: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.ea3, %Destroy.facet.663 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.a6a: %CppRangeForIterate.assoc_type = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [concrete = constants.%assoc2]
// CHECK:STDOUT:   %Core.import_ref.5b4: %CppRangeForIterate.assoc_type = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [concrete = constants.%assoc3]
// CHECK:STDOUT:   %Core.import_ref.08d: @CppRangeForIterate.WithSelf.%CppRangeForIterate.WithSelf.Begin.type (%CppRangeForIterate.WithSelf.Begin.type.3276c2.1) = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @CppRangeForIterate.WithSelf.%CppRangeForIterate.WithSelf.Begin (constants.%CppRangeForIterate.WithSelf.Begin.39882e.1)]
// CHECK:STDOUT:   %Core.import_ref.49d: @CppRangeForIterate.WithSelf.%CppRangeForIterate.WithSelf.End.type (%CppRangeForIterate.WithSelf.End.type.fe619a.1) = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @CppRangeForIterate.WithSelf.%CppRangeForIterate.WithSelf.End (constants.%CppRangeForIterate.WithSelf.End.0f5092.1)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Test(%R.loc32_10.2: %CppRangeForIterate_where.type) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %CppRangeForIterate.lookup_impl_witness: <witness> = lookup_impl_witness %R.loc32_10.1, @CppRangeForIterate [symbolic = %CppRangeForIterate.lookup_impl_witness (constants.%CppRangeForIterate.lookup_impl_witness.7e2712.2)]
// CHECK:STDOUT:   %CppRangeForIterate.facet: %CppRangeForIterate.type = facet_value %R.as_type.loc32_113.1, (%CppRangeForIterate.lookup_impl_witness) [symbolic = %CppRangeForIterate.facet (constants.%CppRangeForIterate.facet.9d6)]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.Begin.type: type = fn_type @CppRangeForIterate.WithSelf.Begin, @CppRangeForIterate.WithSelf(%CppRangeForIterate.facet) [symbolic = %CppRangeForIterate.WithSelf.Begin.type (constants.%CppRangeForIterate.WithSelf.Begin.type.848)]
// CHECK:STDOUT:   %.loc34_4: type = fn_type_with_self_type %CppRangeForIterate.WithSelf.Begin.type, %CppRangeForIterate.facet [symbolic = %.loc34_4 (constants.%.be2)]
// CHECK:STDOUT:   %impl.elem2.loc34_4.2: @Test.%.loc34_4 (%.be2) = impl_witness_access %CppRangeForIterate.lookup_impl_witness, element2 [symbolic = %impl.elem2.loc34_4.2 (constants.%impl.elem2)]
// CHECK:STDOUT:   %specific_impl_fn.loc34_4.2: <specific function> = specific_impl_function %impl.elem2.loc34_4.2, @CppRangeForIterate.WithSelf.Begin(%CppRangeForIterate.facet) [symbolic = %specific_impl_fn.loc34_4.2 (constants.%specific_impl_fn.b91)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.End.type: type = fn_type @CppRangeForIterate.WithSelf.End, @CppRangeForIterate.WithSelf(%CppRangeForIterate.facet) [symbolic = %CppRangeForIterate.WithSelf.End.type (constants.%CppRangeForIterate.WithSelf.End.type.2e5f)]
// CHECK:STDOUT:   %.loc35_4: type = fn_type_with_self_type %CppRangeForIterate.WithSelf.End.type, %CppRangeForIterate.facet [symbolic = %.loc35_4 (constants.%.fbd)]
// CHECK:STDOUT:   %impl.elem3.loc35_4.2: @Test.%.loc35_4 (%.fbd) = impl_witness_access %CppRangeForIterate.lookup_impl_witness, element3 [symbolic = %impl.elem3.loc35_4.2 (constants.%impl.elem3)]
// CHECK:STDOUT:   %specific_impl_fn.loc35_4.2: <specific function> = specific_impl_function %impl.elem3.loc35_4.2, @CppRangeForIterate.WithSelf.End(%CppRangeForIterate.facet) [symbolic = %specific_impl_fn.loc35_4.2 (constants.%specific_impl_fn.2ec)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.loc35: <witness> = lookup_impl_witness %impl.elem1.1, @Destroy [symbolic = %Destroy.lookup_impl_witness.loc35 (constants.%Destroy.lookup_impl_witness.afb)]
// CHECK:STDOUT:   %Destroy.facet.loc35: %Destroy.type = facet_value %impl.elem1.1, (%Destroy.lookup_impl_witness.loc35) [symbolic = %Destroy.facet.loc35 (constants.%Destroy.facet.63a)]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.loc35: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.loc35) [symbolic = %Destroy.WithSelf.Op.type.loc35 (constants.%Destroy.WithSelf.Op.type.57d)]
// CHECK:STDOUT:   %.loc35_9.3: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.loc35, %Destroy.facet.loc35 [symbolic = %.loc35_9.3 (constants.%.db6)]
// CHECK:STDOUT:   %impl.elem0.loc35_9.2: @Test.%.loc35_9.3 (%.db6) = impl_witness_access %Destroy.lookup_impl_witness.loc35, element0 [symbolic = %impl.elem0.loc35_9.2 (constants.%impl.elem0.180)]
// CHECK:STDOUT:   %specific_impl_fn.loc35_9.2: <specific function> = specific_impl_function %impl.elem0.loc35_9.2, @Destroy.WithSelf.Op(%Destroy.facet.loc35) [symbolic = %specific_impl_fn.loc35_9.2 (constants.%specific_impl_fn.a08)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.loc34: <witness> = lookup_impl_witness %impl.elem0.1, @Destroy [symbolic = %Destroy.lookup_impl_witness.loc34 (constants.%Destroy.lookup_impl_witness.75c)]
// CHECK:STDOUT:   %Destroy.facet.loc34: %Destroy.type = facet_value %impl.elem0.1, (%Destroy.lookup_impl_witness.loc34) [symbolic = %Destroy.facet.loc34 (constants.%Destroy.facet.dae)]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.loc34: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.loc34) [symbolic = %Destroy.WithSelf.Op.type.loc34 (constants.%Destroy.WithSelf.Op.type.5c3)]
// CHECK:STDOUT:   %.loc34_11.3: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.loc34, %Destroy.facet.loc34 [symbolic = %.loc34_11.3 (constants.%.51a)]
// CHECK:STDOUT:   %impl.elem0.loc34_11.2: @Test.%.loc34_11.3 (%.51a) = impl_witness_access %Destroy.lookup_impl_witness.loc34, element0 [symbolic = %impl.elem0.loc34_11.2 (constants.%impl.elem0.2a2)]
// CHECK:STDOUT:   %specific_impl_fn.loc34_11.2: <specific function> = specific_impl_function %impl.elem0.loc34_11.2, @Destroy.WithSelf.Op(%Destroy.facet.loc34) [symbolic = %specific_impl_fn.loc34_11.2 (constants.%specific_impl_fn.842)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%r.param: ref @Test.%R.as_type.loc32_113.1 (%R.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %r.ref.loc34: ref @Test.%R.as_type.loc32_113.1 (%R.as_type) = name_ref r, %r
// CHECK:STDOUT:     %Begin.ref: %CppRangeForIterate.assoc_type = name_ref Begin, imports.%Core.import_ref.a6a [concrete = constants.%assoc2]
// CHECK:STDOUT:     %impl.elem2.loc34_4.1: @Test.%.loc34_4 (%.be2) = impl_witness_access constants.%CppRangeForIterate.lookup_impl_witness.7e2712.2, element2 [symbolic = %impl.elem2.loc34_4.2 (constants.%impl.elem2)]
// CHECK:STDOUT:     %bound_method.loc34_4: <bound method> = bound_method %r.ref.loc34, %impl.elem2.loc34_4.1
// CHECK:STDOUT:     %specific_impl_fn.loc34_4.1: <specific function> = specific_impl_function %impl.elem2.loc34_4.1, @CppRangeForIterate.WithSelf.Begin(constants.%CppRangeForIterate.facet.9d6) [symbolic = %specific_impl_fn.loc34_4.2 (constants.%specific_impl_fn.b91)]
// CHECK:STDOUT:     %bound_method.loc34_11.1: <bound method> = bound_method %r.ref.loc34, %specific_impl_fn.loc34_4.1
// CHECK:STDOUT:     %.loc34_11.1: ref @Test.%impl.elem0.1 (%impl.elem0.73eb7f.2) = temporary_storage
// CHECK:STDOUT:     %CppRangeForIterate.WithSelf.Begin.call: init @Test.%impl.elem0.1 (%impl.elem0.73eb7f.2) to %.loc34_11.1 = call %bound_method.loc34_11.1(%r.ref.loc34)
// CHECK:STDOUT:     %.loc34_11.2: ref @Test.%impl.elem0.1 (%impl.elem0.73eb7f.2) = temporary %.loc34_11.1, %CppRangeForIterate.WithSelf.Begin.call
// CHECK:STDOUT:     %r.ref.loc35: ref @Test.%R.as_type.loc32_113.1 (%R.as_type) = name_ref r, %r
// CHECK:STDOUT:     %End.ref: %CppRangeForIterate.assoc_type = name_ref End, imports.%Core.import_ref.5b4 [concrete = constants.%assoc3]
// CHECK:STDOUT:     %impl.elem3.loc35_4.1: @Test.%.loc35_4 (%.fbd) = impl_witness_access constants.%CppRangeForIterate.lookup_impl_witness.7e2712.2, element3 [symbolic = %impl.elem3.loc35_4.2 (constants.%impl.elem3)]
// CHECK:STDOUT:     %bound_method.loc35_4: <bound method> = bound_method %r.ref.loc35, %impl.elem3.loc35_4.1
// CHECK:STDOUT:     %specific_impl_fn.loc35_4.1: <specific function> = specific_impl_function %impl.elem3.loc35_4.1, @CppRangeForIterate.WithSelf.End(constants.%CppRangeForIterate.facet.9d6) [symbolic = %specific_impl_fn.loc35_4.2 (constants.%specific_impl_fn.2ec)]
// CHECK:STDOUT:     %bound_method.loc35_9.1: <bound method> = bound_method %r.ref.loc35, %specific_impl_fn.loc35_4.1
// CHECK:STDOUT:     %.loc35_9.1: ref @Test.%impl.elem1.1 (%impl.elem1.7a0bff.2) = temporary_storage
// CHECK:STDOUT:     %CppRangeForIterate.WithSelf.End.call: init @Test.%impl.elem1.1 (%impl.elem1.7a0bff.2) to %.loc35_9.1 = call %bound_method.loc35_9.1(%r.ref.loc35)
// CHECK:STDOUT:     %.loc35_9.2: ref @Test.%impl.elem1.1 (%impl.elem1.7a0bff.2) = temporary %.loc35_9.1, %CppRangeForIterate.WithSelf.End.call
// CHECK:STDOUT:     %impl.elem0.loc35_9.1: @Test.%.loc35_9.3 (%.db6) = impl_witness_access constants.%Destroy.lookup_impl_witness.afb, element0 [symbolic = %impl.elem0.loc35_9.2 (constants.%impl.elem0.180)]
// CHECK:STDOUT:     %bound_method.loc35_9.2: <bound method> = bound_method %.loc35_9.2, %impl.elem0.loc35_9.1
// CHECK:STDOUT:     %specific_impl_fn.loc35_9.1: <specific function> = specific_impl_function %impl.elem0.loc35_9.1, @Destroy.WithSelf.Op(constants.%Destroy.facet.63a) [symbolic = %specific_impl_fn.loc35_9.2 (constants.%specific_impl_fn.a08)]
// CHECK:STDOUT:     %bound_method.loc35_9.3: <bound method> = bound_method %.loc35_9.2, %specific_impl_fn.loc35_9.1
// CHECK:STDOUT:     %Destroy.WithSelf.Op.call.loc35: init %empty_tuple.type = call %bound_method.loc35_9.3(%.loc35_9.2)
// CHECK:STDOUT:     %impl.elem0.loc34_11.1: @Test.%.loc34_11.3 (%.51a) = impl_witness_access constants.%Destroy.lookup_impl_witness.75c, element0 [symbolic = %impl.elem0.loc34_11.2 (constants.%impl.elem0.2a2)]
// CHECK:STDOUT:     %bound_method.loc34_11.2: <bound method> = bound_method %.loc34_11.2, %impl.elem0.loc34_11.1
// CHECK:STDOUT:     %specific_impl_fn.loc34_11.1: <specific function> = specific_impl_function %impl.elem0.loc34_11.1, @Destroy.WithSelf.Op(constants.%Destroy.facet.dae) [symbolic = %specific_impl_fn.loc34_11.2 (constants.%specific_impl_fn.842)]
// CHECK:STDOUT:     %bound_method.loc34_11.3: <bound method> = bound_method %.loc34_11.2, %specific_impl_fn.loc34_11.1
// CHECK:STDOUT:     %Destroy.WithSelf.Op.call.loc34: init %empty_tuple.type = call %bound_method.loc34_11.3(%.loc34_11.2)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(constants.%R) {
// CHECK:STDOUT:   %R.loc32_10.1 => constants.%R
// CHECK:STDOUT:   %R.as_type.loc32_113.1 => constants.%R.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.48fa74.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(constants.%facet_value) {
// CHECK:STDOUT:   %R.loc32_10.1 => constants.%facet_value
// CHECK:STDOUT:   %R.as_type.loc32_113.1 => constants.%MutableAndConstRange
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.394
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc32 => constants.%complete_type
// CHECK:STDOUT:   %CppRangeForIterate.lookup_impl_witness => constants.%custom_witness.a40
// CHECK:STDOUT:   %CppRangeForIterate.facet => constants.%CppRangeForIterate.facet.c41
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.Begin.type => constants.%CppRangeForIterate.WithSelf.Begin.type.d70
// CHECK:STDOUT:   %.loc34_4 => constants.%.a97
// CHECK:STDOUT:   %impl.elem2.loc34_4.2 => constants.%Begin
// CHECK:STDOUT:   %specific_impl_fn.loc34_4.2 => constants.%Begin
// CHECK:STDOUT:   %impl.elem0.1 => constants.%Iterator
// CHECK:STDOUT:   %require_complete.1 => constants.%complete_type
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.End.type => constants.%CppRangeForIterate.WithSelf.End.type.e30
// CHECK:STDOUT:   %.loc35_4 => constants.%.abe
// CHECK:STDOUT:   %impl.elem3.loc35_4.2 => constants.%End
// CHECK:STDOUT:   %specific_impl_fn.loc35_4.2 => constants.%End
// CHECK:STDOUT:   %impl.elem1.1 => constants.%Iterator
// CHECK:STDOUT:   %require_complete.2 => constants.%complete_type
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.loc35 => constants.%custom_witness.04a
// CHECK:STDOUT:   %Destroy.facet.loc35 => constants.%Destroy.facet.663
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.loc35 => constants.%Destroy.WithSelf.Op.type.ea3
// CHECK:STDOUT:   %.loc35_9.3 => constants.%.57f
// CHECK:STDOUT:   %impl.elem0.loc35_9.2 => constants.%Iterator.Op
// CHECK:STDOUT:   %specific_impl_fn.loc35_9.2 => constants.%Iterator.Op
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.loc34 => constants.%custom_witness.04a
// CHECK:STDOUT:   %Destroy.facet.loc34 => constants.%Destroy.facet.663
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.loc34 => constants.%Destroy.WithSelf.Op.type.ea3
// CHECK:STDOUT:   %.loc34_11.3 => constants.%.57f
// CHECK:STDOUT:   %impl.elem0.loc34_11.2 => constants.%Iterator.Op
// CHECK:STDOUT:   %specific_impl_fn.loc34_11.2 => constants.%Iterator.Op
// CHECK:STDOUT: }
// CHECK:STDOUT:
