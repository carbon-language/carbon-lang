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
// CHECK:STDOUT:   %Self.6c6: %CppRangeForIterate.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.End.type.056be7.1: type = fn_type @CppRangeForIterate.WithSelf.End, @CppRangeForIterate.WithSelf(%Self.6c6) [symbolic]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.End.a331a3.1: %CppRangeForIterate.WithSelf.End.type.056be7.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.Begin.type.753895.1: type = fn_type @CppRangeForIterate.WithSelf.Begin, @CppRangeForIterate.WithSelf(%Self.6c6) [symbolic]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.Begin.2b31f7.1: %CppRangeForIterate.WithSelf.Begin.type.753895.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %.Self.10b: %CppRangeForIterate.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %CppRangeForIterate.assoc_type: type = assoc_entity_type @CppRangeForIterate [concrete]
// CHECK:STDOUT:   %CppRangeForIterate.lookup_impl_witness.5bf: <witness> = lookup_impl_witness %.Self.10b, @CppRangeForIterate [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.d47: type = impl_witness_access %CppRangeForIterate.lookup_impl_witness.5bf, element0 [symbolic_self]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %impl.elem1.846: type = impl_witness_access %CppRangeForIterate.lookup_impl_witness.5bf, element1 [symbolic_self]
// CHECK:STDOUT:   %CppRangeForIterate_where.type: type = facet_type <@CppRangeForIterate where %impl.elem0.d47 impls @Destroy and %impl.elem1.846 impls @Destroy> [concrete]
// CHECK:STDOUT:   %R: %CppRangeForIterate_where.type = symbolic_binding R, 0 [symbolic]
// CHECK:STDOUT:   %R.as_type: type = facet_access_type %R [symbolic]
// CHECK:STDOUT:   %pattern_type.26c6ea.2: type = pattern_type %R.as_type [symbolic]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %assoc2: %CppRangeForIterate.assoc_type = assoc_entity element2, imports.%Core.import_ref.213 [concrete]
// CHECK:STDOUT:   %CppRangeForIterate.lookup_impl_witness.edd1fe.2: <witness> = lookup_impl_witness %R, @CppRangeForIterate [symbolic]
// CHECK:STDOUT:   %CppRangeForIterate.facet.0d2: %CppRangeForIterate.type = facet_value %R.as_type, (%CppRangeForIterate.lookup_impl_witness.edd1fe.2) [symbolic]
// CHECK:STDOUT:   %impl.elem0.d5966e.2: type = impl_witness_access %CppRangeForIterate.lookup_impl_witness.edd1fe.2, element0 [symbolic]
// CHECK:STDOUT:   %impl.elem1.bdcd2d.2: type = impl_witness_access %CppRangeForIterate.lookup_impl_witness.edd1fe.2, element1 [symbolic]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.Begin.type.57e: type = fn_type @CppRangeForIterate.WithSelf.Begin, @CppRangeForIterate.WithSelf(%CppRangeForIterate.facet.0d2) [symbolic]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.End.type.04d: type = fn_type @CppRangeForIterate.WithSelf.End, @CppRangeForIterate.WithSelf(%CppRangeForIterate.facet.0d2) [symbolic]
// CHECK:STDOUT:   %.b85: type = fn_type_with_self_type %CppRangeForIterate.WithSelf.Begin.type.57e, %CppRangeForIterate.facet.0d2 [symbolic]
// CHECK:STDOUT:   %impl.elem2: %.b85 = impl_witness_access %CppRangeForIterate.lookup_impl_witness.edd1fe.2, element2 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.c0f: <specific function> = specific_impl_function %impl.elem2, @CppRangeForIterate.WithSelf.Begin(%CppRangeForIterate.facet.0d2) [symbolic]
// CHECK:STDOUT:   %assoc3: %CppRangeForIterate.assoc_type = assoc_entity element3, imports.%Core.import_ref.730 [concrete]
// CHECK:STDOUT:   %.b40: type = fn_type_with_self_type %CppRangeForIterate.WithSelf.End.type.04d, %CppRangeForIterate.facet.0d2 [symbolic]
// CHECK:STDOUT:   %impl.elem3: %.b40 = impl_witness_access %CppRangeForIterate.lookup_impl_witness.edd1fe.2, element3 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.d6a: <specific function> = specific_impl_function %impl.elem3, @CppRangeForIterate.WithSelf.End(%CppRangeForIterate.facet.0d2) [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.a8f: <witness> = lookup_impl_witness %impl.elem1.bdcd2d.2, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.543: %Destroy.type = facet_value %impl.elem1.bdcd2d.2, (%Destroy.lookup_impl_witness.a8f) [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.366: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.543) [symbolic]
// CHECK:STDOUT:   %.b83: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.366, %Destroy.facet.543 [symbolic]
// CHECK:STDOUT:   %impl.elem0.acf: %.b83 = impl_witness_access %Destroy.lookup_impl_witness.a8f, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.add: <specific function> = specific_impl_function %impl.elem0.acf, @Destroy.WithSelf.Op(%Destroy.facet.543) [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.089: <witness> = lookup_impl_witness %impl.elem0.d5966e.2, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.5e8: %Destroy.type = facet_value %impl.elem0.d5966e.2, (%Destroy.lookup_impl_witness.089) [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.7cb: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.5e8) [symbolic]
// CHECK:STDOUT:   %.a68: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.7cb, %Destroy.facet.5e8 [symbolic]
// CHECK:STDOUT:   %impl.elem0.f59: %.a68 = impl_witness_access %Destroy.lookup_impl_witness.089, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.f06: <specific function> = specific_impl_function %impl.elem0.f59, @Destroy.WithSelf.Op(%Destroy.facet.5e8) [symbolic]
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
// CHECK:STDOUT:   %custom_witness.544: <witness> = custom_witness (%Iterator.826, %Iterator.826, %MutableRange.Begin, %MutableRange.End), @CppRangeForIterate [concrete]
// CHECK:STDOUT:   %CppRangeForIterate.facet.486: %CppRangeForIterate.type = facet_value %MutableRange, (%custom_witness.544) [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.134: type = fn_type @Iterator.Op.1 [concrete]
// CHECK:STDOUT:   %Iterator.Op.f05: %Iterator.Op.type.134 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.ef9: <witness> = custom_witness (%Iterator.Op.f05), @Destroy [concrete]
// CHECK:STDOUT:   %facet_value.848: %CppRangeForIterate_where.type = facet_value %MutableRange, (%custom_witness.544, %custom_witness.ef9) [concrete]
// CHECK:STDOUT:   %Iterator.4f0: type = class_type @Iterator.2 [concrete]
// CHECK:STDOUT:   %ConstRange.Begin.type: type = fn_type @ConstRange.Begin [concrete]
// CHECK:STDOUT:   %ConstRange.Begin: %ConstRange.Begin.type = struct_value () [concrete]
// CHECK:STDOUT:   %ConstRange.End.type: type = fn_type @ConstRange.End [concrete]
// CHECK:STDOUT:   %ConstRange.End: %ConstRange.End.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.83f: <witness> = custom_witness (%Iterator.4f0, %Iterator.4f0, %ConstRange.Begin, %ConstRange.End), @CppRangeForIterate [concrete]
// CHECK:STDOUT:   %CppRangeForIterate.facet.d14: %CppRangeForIterate.type = facet_value %ConstRange, (%custom_witness.83f) [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.9f2: type = fn_type @Iterator.Op.2 [concrete]
// CHECK:STDOUT:   %Iterator.Op.7b7: %Iterator.Op.type.9f2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.df0: <witness> = custom_witness (%Iterator.Op.7b7), @Destroy [concrete]
// CHECK:STDOUT:   %facet_value.d23: %CppRangeForIterate_where.type = facet_value %ConstRange, (%custom_witness.83f, %custom_witness.df0) [concrete]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.Begin.type.aea: type = fn_type @CppRangeForIterate.WithSelf.Begin, @CppRangeForIterate.WithSelf(%CppRangeForIterate.facet.486) [concrete]
// CHECK:STDOUT:   %.5be: type = fn_type_with_self_type %CppRangeForIterate.WithSelf.Begin.type.aea, %CppRangeForIterate.facet.486 [concrete]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.End.type.1c4: type = fn_type @CppRangeForIterate.WithSelf.End, @CppRangeForIterate.WithSelf(%CppRangeForIterate.facet.486) [concrete]
// CHECK:STDOUT:   %.e19: type = fn_type_with_self_type %CppRangeForIterate.WithSelf.End.type.1c4, %CppRangeForIterate.facet.486 [concrete]
// CHECK:STDOUT:   %Destroy.facet.f6b: %Destroy.type = facet_value %Iterator.826, (%custom_witness.ef9) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.a45: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.f6b) [concrete]
// CHECK:STDOUT:   %.ee6: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.a45, %Destroy.facet.f6b [concrete]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.Begin.type.4f8: type = fn_type @CppRangeForIterate.WithSelf.Begin, @CppRangeForIterate.WithSelf(%CppRangeForIterate.facet.d14) [concrete]
// CHECK:STDOUT:   %.3b9: type = fn_type_with_self_type %CppRangeForIterate.WithSelf.Begin.type.4f8, %CppRangeForIterate.facet.d14 [concrete]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.End.type.d4a: type = fn_type @CppRangeForIterate.WithSelf.End, @CppRangeForIterate.WithSelf(%CppRangeForIterate.facet.d14) [concrete]
// CHECK:STDOUT:   %.364: type = fn_type_with_self_type %CppRangeForIterate.WithSelf.End.type.d4a, %CppRangeForIterate.facet.d14 [concrete]
// CHECK:STDOUT:   %Destroy.facet.2fa: %Destroy.type = facet_value %Iterator.4f0, (%custom_witness.df0) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.993: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.2fa) [concrete]
// CHECK:STDOUT:   %.61a: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.993, %Destroy.facet.2fa [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.735: %CppRangeForIterate.assoc_type = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [concrete = constants.%assoc2]
// CHECK:STDOUT:   %Core.import_ref.572: %CppRangeForIterate.assoc_type = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [concrete = constants.%assoc3]
// CHECK:STDOUT:   %Core.import_ref.213: @CppRangeForIterate.WithSelf.%CppRangeForIterate.WithSelf.Begin.type (%CppRangeForIterate.WithSelf.Begin.type.753895.1) = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @CppRangeForIterate.WithSelf.%CppRangeForIterate.WithSelf.Begin (constants.%CppRangeForIterate.WithSelf.Begin.2b31f7.1)]
// CHECK:STDOUT:   %Core.import_ref.730: @CppRangeForIterate.WithSelf.%CppRangeForIterate.WithSelf.End.type (%CppRangeForIterate.WithSelf.End.type.056be7.1) = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @CppRangeForIterate.WithSelf.%CppRangeForIterate.WithSelf.End (constants.%CppRangeForIterate.WithSelf.End.a331a3.1)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Test(%R.loc30_10.2: %CppRangeForIterate_where.type) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %CppRangeForIterate.lookup_impl_witness: <witness> = lookup_impl_witness %R.loc30_10.1, @CppRangeForIterate [symbolic = %CppRangeForIterate.lookup_impl_witness (constants.%CppRangeForIterate.lookup_impl_witness.edd1fe.2)]
// CHECK:STDOUT:   %CppRangeForIterate.facet: %CppRangeForIterate.type = facet_value %R.as_type.loc30_113.1, (%CppRangeForIterate.lookup_impl_witness) [symbolic = %CppRangeForIterate.facet (constants.%CppRangeForIterate.facet.0d2)]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.Begin.type: type = fn_type @CppRangeForIterate.WithSelf.Begin, @CppRangeForIterate.WithSelf(%CppRangeForIterate.facet) [symbolic = %CppRangeForIterate.WithSelf.Begin.type (constants.%CppRangeForIterate.WithSelf.Begin.type.57e)]
// CHECK:STDOUT:   %.loc32_4: type = fn_type_with_self_type %CppRangeForIterate.WithSelf.Begin.type, %CppRangeForIterate.facet [symbolic = %.loc32_4 (constants.%.b85)]
// CHECK:STDOUT:   %impl.elem2.loc32_4.2: @Test.%.loc32_4 (%.b85) = impl_witness_access %CppRangeForIterate.lookup_impl_witness, element2 [symbolic = %impl.elem2.loc32_4.2 (constants.%impl.elem2)]
// CHECK:STDOUT:   %specific_impl_fn.loc32_4.2: <specific function> = specific_impl_function %impl.elem2.loc32_4.2, @CppRangeForIterate.WithSelf.Begin(%CppRangeForIterate.facet) [symbolic = %specific_impl_fn.loc32_4.2 (constants.%specific_impl_fn.c0f)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.End.type: type = fn_type @CppRangeForIterate.WithSelf.End, @CppRangeForIterate.WithSelf(%CppRangeForIterate.facet) [symbolic = %CppRangeForIterate.WithSelf.End.type (constants.%CppRangeForIterate.WithSelf.End.type.04d)]
// CHECK:STDOUT:   %.loc33_4: type = fn_type_with_self_type %CppRangeForIterate.WithSelf.End.type, %CppRangeForIterate.facet [symbolic = %.loc33_4 (constants.%.b40)]
// CHECK:STDOUT:   %impl.elem3.loc33_4.2: @Test.%.loc33_4 (%.b40) = impl_witness_access %CppRangeForIterate.lookup_impl_witness, element3 [symbolic = %impl.elem3.loc33_4.2 (constants.%impl.elem3)]
// CHECK:STDOUT:   %specific_impl_fn.loc33_4.2: <specific function> = specific_impl_function %impl.elem3.loc33_4.2, @CppRangeForIterate.WithSelf.End(%CppRangeForIterate.facet) [symbolic = %specific_impl_fn.loc33_4.2 (constants.%specific_impl_fn.d6a)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.loc33: <witness> = lookup_impl_witness %impl.elem1.1, @Destroy [symbolic = %Destroy.lookup_impl_witness.loc33 (constants.%Destroy.lookup_impl_witness.a8f)]
// CHECK:STDOUT:   %Destroy.facet.loc33: %Destroy.type = facet_value %impl.elem1.1, (%Destroy.lookup_impl_witness.loc33) [symbolic = %Destroy.facet.loc33 (constants.%Destroy.facet.543)]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.loc33: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.loc33) [symbolic = %Destroy.WithSelf.Op.type.loc33 (constants.%Destroy.WithSelf.Op.type.366)]
// CHECK:STDOUT:   %.loc33_9.3: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.loc33, %Destroy.facet.loc33 [symbolic = %.loc33_9.3 (constants.%.b83)]
// CHECK:STDOUT:   %impl.elem0.loc33_9.2: @Test.%.loc33_9.3 (%.b83) = impl_witness_access %Destroy.lookup_impl_witness.loc33, element0 [symbolic = %impl.elem0.loc33_9.2 (constants.%impl.elem0.acf)]
// CHECK:STDOUT:   %specific_impl_fn.loc33_9.2: <specific function> = specific_impl_function %impl.elem0.loc33_9.2, @Destroy.WithSelf.Op(%Destroy.facet.loc33) [symbolic = %specific_impl_fn.loc33_9.2 (constants.%specific_impl_fn.add)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.loc32: <witness> = lookup_impl_witness %impl.elem0.1, @Destroy [symbolic = %Destroy.lookup_impl_witness.loc32 (constants.%Destroy.lookup_impl_witness.089)]
// CHECK:STDOUT:   %Destroy.facet.loc32: %Destroy.type = facet_value %impl.elem0.1, (%Destroy.lookup_impl_witness.loc32) [symbolic = %Destroy.facet.loc32 (constants.%Destroy.facet.5e8)]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.loc32: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.loc32) [symbolic = %Destroy.WithSelf.Op.type.loc32 (constants.%Destroy.WithSelf.Op.type.7cb)]
// CHECK:STDOUT:   %.loc32_11.3: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.loc32, %Destroy.facet.loc32 [symbolic = %.loc32_11.3 (constants.%.a68)]
// CHECK:STDOUT:   %impl.elem0.loc32_11.2: @Test.%.loc32_11.3 (%.a68) = impl_witness_access %Destroy.lookup_impl_witness.loc32, element0 [symbolic = %impl.elem0.loc32_11.2 (constants.%impl.elem0.f59)]
// CHECK:STDOUT:   %specific_impl_fn.loc32_11.2: <specific function> = specific_impl_function %impl.elem0.loc32_11.2, @Destroy.WithSelf.Op(%Destroy.facet.loc32) [symbolic = %specific_impl_fn.loc32_11.2 (constants.%specific_impl_fn.f06)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%r.param: ref @Test.%R.as_type.loc30_113.1 (%R.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %r.ref.loc32: ref @Test.%R.as_type.loc30_113.1 (%R.as_type) = name_ref r, %r
// CHECK:STDOUT:     %Begin.ref: %CppRangeForIterate.assoc_type = name_ref Begin, imports.%Core.import_ref.735 [concrete = constants.%assoc2]
// CHECK:STDOUT:     %impl.elem2.loc32_4.1: @Test.%.loc32_4 (%.b85) = impl_witness_access constants.%CppRangeForIterate.lookup_impl_witness.edd1fe.2, element2 [symbolic = %impl.elem2.loc32_4.2 (constants.%impl.elem2)]
// CHECK:STDOUT:     %bound_method.loc32_4: <bound method> = bound_method %r.ref.loc32, %impl.elem2.loc32_4.1
// CHECK:STDOUT:     %specific_impl_fn.loc32_4.1: <specific function> = specific_impl_function %impl.elem2.loc32_4.1, @CppRangeForIterate.WithSelf.Begin(constants.%CppRangeForIterate.facet.0d2) [symbolic = %specific_impl_fn.loc32_4.2 (constants.%specific_impl_fn.c0f)]
// CHECK:STDOUT:     %bound_method.loc32_11.1: <bound method> = bound_method %r.ref.loc32, %specific_impl_fn.loc32_4.1
// CHECK:STDOUT:     %.loc32_11.1: ref @Test.%impl.elem0.1 (%impl.elem0.d5966e.2) = temporary_storage
// CHECK:STDOUT:     %CppRangeForIterate.WithSelf.Begin.call: init @Test.%impl.elem0.1 (%impl.elem0.d5966e.2) to %.loc32_11.1 = call %bound_method.loc32_11.1(%r.ref.loc32)
// CHECK:STDOUT:     %.loc32_11.2: ref @Test.%impl.elem0.1 (%impl.elem0.d5966e.2) = temporary %.loc32_11.1, %CppRangeForIterate.WithSelf.Begin.call
// CHECK:STDOUT:     %r.ref.loc33: ref @Test.%R.as_type.loc30_113.1 (%R.as_type) = name_ref r, %r
// CHECK:STDOUT:     %End.ref: %CppRangeForIterate.assoc_type = name_ref End, imports.%Core.import_ref.572 [concrete = constants.%assoc3]
// CHECK:STDOUT:     %impl.elem3.loc33_4.1: @Test.%.loc33_4 (%.b40) = impl_witness_access constants.%CppRangeForIterate.lookup_impl_witness.edd1fe.2, element3 [symbolic = %impl.elem3.loc33_4.2 (constants.%impl.elem3)]
// CHECK:STDOUT:     %bound_method.loc33_4: <bound method> = bound_method %r.ref.loc33, %impl.elem3.loc33_4.1
// CHECK:STDOUT:     %specific_impl_fn.loc33_4.1: <specific function> = specific_impl_function %impl.elem3.loc33_4.1, @CppRangeForIterate.WithSelf.End(constants.%CppRangeForIterate.facet.0d2) [symbolic = %specific_impl_fn.loc33_4.2 (constants.%specific_impl_fn.d6a)]
// CHECK:STDOUT:     %bound_method.loc33_9.1: <bound method> = bound_method %r.ref.loc33, %specific_impl_fn.loc33_4.1
// CHECK:STDOUT:     %.loc33_9.1: ref @Test.%impl.elem1.1 (%impl.elem1.bdcd2d.2) = temporary_storage
// CHECK:STDOUT:     %CppRangeForIterate.WithSelf.End.call: init @Test.%impl.elem1.1 (%impl.elem1.bdcd2d.2) to %.loc33_9.1 = call %bound_method.loc33_9.1(%r.ref.loc33)
// CHECK:STDOUT:     %.loc33_9.2: ref @Test.%impl.elem1.1 (%impl.elem1.bdcd2d.2) = temporary %.loc33_9.1, %CppRangeForIterate.WithSelf.End.call
// CHECK:STDOUT:     %impl.elem0.loc33_9.1: @Test.%.loc33_9.3 (%.b83) = impl_witness_access constants.%Destroy.lookup_impl_witness.a8f, element0 [symbolic = %impl.elem0.loc33_9.2 (constants.%impl.elem0.acf)]
// CHECK:STDOUT:     %bound_method.loc33_9.2: <bound method> = bound_method %.loc33_9.2, %impl.elem0.loc33_9.1
// CHECK:STDOUT:     %specific_impl_fn.loc33_9.1: <specific function> = specific_impl_function %impl.elem0.loc33_9.1, @Destroy.WithSelf.Op(constants.%Destroy.facet.543) [symbolic = %specific_impl_fn.loc33_9.2 (constants.%specific_impl_fn.add)]
// CHECK:STDOUT:     %bound_method.loc33_9.3: <bound method> = bound_method %.loc33_9.2, %specific_impl_fn.loc33_9.1
// CHECK:STDOUT:     %Destroy.WithSelf.Op.call.loc33: init %empty_tuple.type = call %bound_method.loc33_9.3(%.loc33_9.2)
// CHECK:STDOUT:     %impl.elem0.loc32_11.1: @Test.%.loc32_11.3 (%.a68) = impl_witness_access constants.%Destroy.lookup_impl_witness.089, element0 [symbolic = %impl.elem0.loc32_11.2 (constants.%impl.elem0.f59)]
// CHECK:STDOUT:     %bound_method.loc32_11.2: <bound method> = bound_method %.loc32_11.2, %impl.elem0.loc32_11.1
// CHECK:STDOUT:     %specific_impl_fn.loc32_11.1: <specific function> = specific_impl_function %impl.elem0.loc32_11.1, @Destroy.WithSelf.Op(constants.%Destroy.facet.5e8) [symbolic = %specific_impl_fn.loc32_11.2 (constants.%specific_impl_fn.f06)]
// CHECK:STDOUT:     %bound_method.loc32_11.3: <bound method> = bound_method %.loc32_11.2, %specific_impl_fn.loc32_11.1
// CHECK:STDOUT:     %Destroy.WithSelf.Op.call.loc32: init %empty_tuple.type = call %bound_method.loc32_11.3(%.loc32_11.2)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(constants.%R) {
// CHECK:STDOUT:   %R.loc30_10.1 => constants.%R
// CHECK:STDOUT:   %R.as_type.loc30_113.1 => constants.%R.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.26c6ea.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(constants.%facet_value.848) {
// CHECK:STDOUT:   %R.loc30_10.1 => constants.%facet_value.848
// CHECK:STDOUT:   %R.as_type.loc30_113.1 => constants.%MutableRange
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.b76
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc30 => constants.%complete_type
// CHECK:STDOUT:   %CppRangeForIterate.lookup_impl_witness => constants.%custom_witness.544
// CHECK:STDOUT:   %CppRangeForIterate.facet => constants.%CppRangeForIterate.facet.486
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.Begin.type => constants.%CppRangeForIterate.WithSelf.Begin.type.aea
// CHECK:STDOUT:   %.loc32_4 => constants.%.5be
// CHECK:STDOUT:   %impl.elem2.loc32_4.2 => constants.%MutableRange.Begin
// CHECK:STDOUT:   %specific_impl_fn.loc32_4.2 => constants.%MutableRange.Begin
// CHECK:STDOUT:   %impl.elem0.1 => constants.%Iterator.826
// CHECK:STDOUT:   %require_complete.1 => constants.%complete_type
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.End.type => constants.%CppRangeForIterate.WithSelf.End.type.1c4
// CHECK:STDOUT:   %.loc33_4 => constants.%.e19
// CHECK:STDOUT:   %impl.elem3.loc33_4.2 => constants.%MutableRange.End
// CHECK:STDOUT:   %specific_impl_fn.loc33_4.2 => constants.%MutableRange.End
// CHECK:STDOUT:   %impl.elem1.1 => constants.%Iterator.826
// CHECK:STDOUT:   %require_complete.2 => constants.%complete_type
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.loc33 => constants.%custom_witness.ef9
// CHECK:STDOUT:   %Destroy.facet.loc33 => constants.%Destroy.facet.f6b
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.loc33 => constants.%Destroy.WithSelf.Op.type.a45
// CHECK:STDOUT:   %.loc33_9.3 => constants.%.ee6
// CHECK:STDOUT:   %impl.elem0.loc33_9.2 => constants.%Iterator.Op.f05
// CHECK:STDOUT:   %specific_impl_fn.loc33_9.2 => constants.%Iterator.Op.f05
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.loc32 => constants.%custom_witness.ef9
// CHECK:STDOUT:   %Destroy.facet.loc32 => constants.%Destroy.facet.f6b
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.loc32 => constants.%Destroy.WithSelf.Op.type.a45
// CHECK:STDOUT:   %.loc32_11.3 => constants.%.ee6
// CHECK:STDOUT:   %impl.elem0.loc32_11.2 => constants.%Iterator.Op.f05
// CHECK:STDOUT:   %specific_impl_fn.loc32_11.2 => constants.%Iterator.Op.f05
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(constants.%facet_value.d23) {
// CHECK:STDOUT:   %R.loc30_10.1 => constants.%facet_value.d23
// CHECK:STDOUT:   %R.as_type.loc30_113.1 => constants.%ConstRange
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.631
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc30 => constants.%complete_type
// CHECK:STDOUT:   %CppRangeForIterate.lookup_impl_witness => constants.%custom_witness.83f
// CHECK:STDOUT:   %CppRangeForIterate.facet => constants.%CppRangeForIterate.facet.d14
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.Begin.type => constants.%CppRangeForIterate.WithSelf.Begin.type.4f8
// CHECK:STDOUT:   %.loc32_4 => constants.%.3b9
// CHECK:STDOUT:   %impl.elem2.loc32_4.2 => constants.%ConstRange.Begin
// CHECK:STDOUT:   %specific_impl_fn.loc32_4.2 => constants.%ConstRange.Begin
// CHECK:STDOUT:   %impl.elem0.1 => constants.%Iterator.4f0
// CHECK:STDOUT:   %require_complete.1 => constants.%complete_type
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.End.type => constants.%CppRangeForIterate.WithSelf.End.type.d4a
// CHECK:STDOUT:   %.loc33_4 => constants.%.364
// CHECK:STDOUT:   %impl.elem3.loc33_4.2 => constants.%ConstRange.End
// CHECK:STDOUT:   %specific_impl_fn.loc33_4.2 => constants.%ConstRange.End
// CHECK:STDOUT:   %impl.elem1.1 => constants.%Iterator.4f0
// CHECK:STDOUT:   %require_complete.2 => constants.%complete_type
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.loc33 => constants.%custom_witness.df0
// CHECK:STDOUT:   %Destroy.facet.loc33 => constants.%Destroy.facet.2fa
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.loc33 => constants.%Destroy.WithSelf.Op.type.993
// CHECK:STDOUT:   %.loc33_9.3 => constants.%.61a
// CHECK:STDOUT:   %impl.elem0.loc33_9.2 => constants.%Iterator.Op.7b7
// CHECK:STDOUT:   %specific_impl_fn.loc33_9.2 => constants.%Iterator.Op.7b7
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.loc32 => constants.%custom_witness.df0
// CHECK:STDOUT:   %Destroy.facet.loc32 => constants.%Destroy.facet.2fa
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.loc32 => constants.%Destroy.WithSelf.Op.type.993
// CHECK:STDOUT:   %.loc32_11.3 => constants.%.61a
// CHECK:STDOUT:   %impl.elem0.loc32_11.2 => constants.%Iterator.Op.7b7
// CHECK:STDOUT:   %specific_impl_fn.loc32_11.2 => constants.%Iterator.Op.7b7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- methods_return_different_types.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %CppRangeForIterate.type: type = facet_type <@CppRangeForIterate> [concrete]
// CHECK:STDOUT:   %Self.6c6: %CppRangeForIterate.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.End.type.056be7.1: type = fn_type @CppRangeForIterate.WithSelf.End, @CppRangeForIterate.WithSelf(%Self.6c6) [symbolic]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.End.a331a3.1: %CppRangeForIterate.WithSelf.End.type.056be7.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.Begin.type.753895.1: type = fn_type @CppRangeForIterate.WithSelf.Begin, @CppRangeForIterate.WithSelf(%Self.6c6) [symbolic]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.Begin.2b31f7.1: %CppRangeForIterate.WithSelf.Begin.type.753895.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %.Self.10b: %CppRangeForIterate.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %CppRangeForIterate.assoc_type: type = assoc_entity_type @CppRangeForIterate [concrete]
// CHECK:STDOUT:   %CppRangeForIterate.lookup_impl_witness.5bf: <witness> = lookup_impl_witness %.Self.10b, @CppRangeForIterate [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.d47: type = impl_witness_access %CppRangeForIterate.lookup_impl_witness.5bf, element0 [symbolic_self]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %impl.elem1.846: type = impl_witness_access %CppRangeForIterate.lookup_impl_witness.5bf, element1 [symbolic_self]
// CHECK:STDOUT:   %CppRangeForIterate_where.type: type = facet_type <@CppRangeForIterate where %impl.elem0.d47 impls @Destroy and %impl.elem1.846 impls @Destroy> [concrete]
// CHECK:STDOUT:   %R: %CppRangeForIterate_where.type = symbolic_binding R, 0 [symbolic]
// CHECK:STDOUT:   %R.as_type: type = facet_access_type %R [symbolic]
// CHECK:STDOUT:   %pattern_type.26c6ea.2: type = pattern_type %R.as_type [symbolic]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %assoc2: %CppRangeForIterate.assoc_type = assoc_entity element2, imports.%Core.import_ref.213 [concrete]
// CHECK:STDOUT:   %CppRangeForIterate.lookup_impl_witness.edd1fe.2: <witness> = lookup_impl_witness %R, @CppRangeForIterate [symbolic]
// CHECK:STDOUT:   %CppRangeForIterate.facet.0d2: %CppRangeForIterate.type = facet_value %R.as_type, (%CppRangeForIterate.lookup_impl_witness.edd1fe.2) [symbolic]
// CHECK:STDOUT:   %impl.elem0.d5966e.2: type = impl_witness_access %CppRangeForIterate.lookup_impl_witness.edd1fe.2, element0 [symbolic]
// CHECK:STDOUT:   %impl.elem1.bdcd2d.2: type = impl_witness_access %CppRangeForIterate.lookup_impl_witness.edd1fe.2, element1 [symbolic]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.Begin.type.57e: type = fn_type @CppRangeForIterate.WithSelf.Begin, @CppRangeForIterate.WithSelf(%CppRangeForIterate.facet.0d2) [symbolic]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.End.type.04d: type = fn_type @CppRangeForIterate.WithSelf.End, @CppRangeForIterate.WithSelf(%CppRangeForIterate.facet.0d2) [symbolic]
// CHECK:STDOUT:   %.b85: type = fn_type_with_self_type %CppRangeForIterate.WithSelf.Begin.type.57e, %CppRangeForIterate.facet.0d2 [symbolic]
// CHECK:STDOUT:   %impl.elem2: %.b85 = impl_witness_access %CppRangeForIterate.lookup_impl_witness.edd1fe.2, element2 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.c0f: <specific function> = specific_impl_function %impl.elem2, @CppRangeForIterate.WithSelf.Begin(%CppRangeForIterate.facet.0d2) [symbolic]
// CHECK:STDOUT:   %assoc3: %CppRangeForIterate.assoc_type = assoc_entity element3, imports.%Core.import_ref.730 [concrete]
// CHECK:STDOUT:   %.b40: type = fn_type_with_self_type %CppRangeForIterate.WithSelf.End.type.04d, %CppRangeForIterate.facet.0d2 [symbolic]
// CHECK:STDOUT:   %impl.elem3: %.b40 = impl_witness_access %CppRangeForIterate.lookup_impl_witness.edd1fe.2, element3 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.d6a: <specific function> = specific_impl_function %impl.elem3, @CppRangeForIterate.WithSelf.End(%CppRangeForIterate.facet.0d2) [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.a8f: <witness> = lookup_impl_witness %impl.elem1.bdcd2d.2, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.543: %Destroy.type = facet_value %impl.elem1.bdcd2d.2, (%Destroy.lookup_impl_witness.a8f) [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.366: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.543) [symbolic]
// CHECK:STDOUT:   %.b83: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.366, %Destroy.facet.543 [symbolic]
// CHECK:STDOUT:   %impl.elem0.acf: %.b83 = impl_witness_access %Destroy.lookup_impl_witness.a8f, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.add: <specific function> = specific_impl_function %impl.elem0.acf, @Destroy.WithSelf.Op(%Destroy.facet.543) [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.089: <witness> = lookup_impl_witness %impl.elem0.d5966e.2, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.5e8: %Destroy.type = facet_value %impl.elem0.d5966e.2, (%Destroy.lookup_impl_witness.089) [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.7cb: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.5e8) [symbolic]
// CHECK:STDOUT:   %.a68: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.7cb, %Destroy.facet.5e8 [symbolic]
// CHECK:STDOUT:   %impl.elem0.f59: %.a68 = impl_witness_access %Destroy.lookup_impl_witness.089, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.f06: <specific function> = specific_impl_function %impl.elem0.f59, @Destroy.WithSelf.Op(%Destroy.facet.5e8) [symbolic]
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
// CHECK:STDOUT:   %custom_witness.292: <witness> = custom_witness (%Iterator.826, %Sentinel.6eb, %MutableRange.Begin, %MutableRange.End), @CppRangeForIterate [concrete]
// CHECK:STDOUT:   %CppRangeForIterate.facet.de2: %CppRangeForIterate.type = facet_value %MutableRange, (%custom_witness.292) [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.134: type = fn_type @Iterator.Op.1 [concrete]
// CHECK:STDOUT:   %Iterator.Op.f05: %Iterator.Op.type.134 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.ef99ee.1: <witness> = custom_witness (%Iterator.Op.f05), @Destroy [concrete]
// CHECK:STDOUT:   %Sentinel.Op.type.134: type = fn_type @Sentinel.Op.1 [concrete]
// CHECK:STDOUT:   %Sentinel.Op.f05: %Sentinel.Op.type.134 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.ef99ee.2: <witness> = custom_witness (%Sentinel.Op.f05), @Destroy [concrete]
// CHECK:STDOUT:   %facet_value.99a: %CppRangeForIterate_where.type = facet_value %MutableRange, (%custom_witness.292, %custom_witness.ef99ee.1, %custom_witness.ef99ee.2) [concrete]
// CHECK:STDOUT:   %Iterator.4f0: type = class_type @Iterator.2 [concrete]
// CHECK:STDOUT:   %Sentinel.45e: type = class_type @Sentinel.2 [concrete]
// CHECK:STDOUT:   %ConstRange.Begin.type: type = fn_type @ConstRange.Begin [concrete]
// CHECK:STDOUT:   %ConstRange.Begin: %ConstRange.Begin.type = struct_value () [concrete]
// CHECK:STDOUT:   %ConstRange.End.type: type = fn_type @ConstRange.End [concrete]
// CHECK:STDOUT:   %ConstRange.End: %ConstRange.End.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.8cb: <witness> = custom_witness (%Iterator.4f0, %Sentinel.45e, %ConstRange.Begin, %ConstRange.End), @CppRangeForIterate [concrete]
// CHECK:STDOUT:   %CppRangeForIterate.facet.ef1: %CppRangeForIterate.type = facet_value %ConstRange, (%custom_witness.8cb) [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.9f2: type = fn_type @Iterator.Op.2 [concrete]
// CHECK:STDOUT:   %Iterator.Op.7b7: %Iterator.Op.type.9f2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.df0c4f.1: <witness> = custom_witness (%Iterator.Op.7b7), @Destroy [concrete]
// CHECK:STDOUT:   %Sentinel.Op.type.9f2: type = fn_type @Sentinel.Op.2 [concrete]
// CHECK:STDOUT:   %Sentinel.Op.7b7: %Sentinel.Op.type.9f2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.df0c4f.2: <witness> = custom_witness (%Sentinel.Op.7b7), @Destroy [concrete]
// CHECK:STDOUT:   %facet_value.b4b: %CppRangeForIterate_where.type = facet_value %ConstRange, (%custom_witness.8cb, %custom_witness.df0c4f.1, %custom_witness.df0c4f.2) [concrete]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.Begin.type.82a: type = fn_type @CppRangeForIterate.WithSelf.Begin, @CppRangeForIterate.WithSelf(%CppRangeForIterate.facet.de2) [concrete]
// CHECK:STDOUT:   %.26f: type = fn_type_with_self_type %CppRangeForIterate.WithSelf.Begin.type.82a, %CppRangeForIterate.facet.de2 [concrete]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.End.type.9fe: type = fn_type @CppRangeForIterate.WithSelf.End, @CppRangeForIterate.WithSelf(%CppRangeForIterate.facet.de2) [concrete]
// CHECK:STDOUT:   %.6d3: type = fn_type_with_self_type %CppRangeForIterate.WithSelf.End.type.9fe, %CppRangeForIterate.facet.de2 [concrete]
// CHECK:STDOUT:   %Destroy.facet.959: %Destroy.type = facet_value %Sentinel.6eb, (%custom_witness.ef99ee.2) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.6d7: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.959) [concrete]
// CHECK:STDOUT:   %.3e7: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.6d7, %Destroy.facet.959 [concrete]
// CHECK:STDOUT:   %Destroy.facet.f6b: %Destroy.type = facet_value %Iterator.826, (%custom_witness.ef99ee.1) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.a45: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.f6b) [concrete]
// CHECK:STDOUT:   %.ee6: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.a45, %Destroy.facet.f6b [concrete]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.Begin.type.317: type = fn_type @CppRangeForIterate.WithSelf.Begin, @CppRangeForIterate.WithSelf(%CppRangeForIterate.facet.ef1) [concrete]
// CHECK:STDOUT:   %.3f6: type = fn_type_with_self_type %CppRangeForIterate.WithSelf.Begin.type.317, %CppRangeForIterate.facet.ef1 [concrete]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.End.type.120: type = fn_type @CppRangeForIterate.WithSelf.End, @CppRangeForIterate.WithSelf(%CppRangeForIterate.facet.ef1) [concrete]
// CHECK:STDOUT:   %.cd4: type = fn_type_with_self_type %CppRangeForIterate.WithSelf.End.type.120, %CppRangeForIterate.facet.ef1 [concrete]
// CHECK:STDOUT:   %Destroy.facet.d9f: %Destroy.type = facet_value %Sentinel.45e, (%custom_witness.df0c4f.2) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.39d: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.d9f) [concrete]
// CHECK:STDOUT:   %.f49: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.39d, %Destroy.facet.d9f [concrete]
// CHECK:STDOUT:   %Destroy.facet.2fa: %Destroy.type = facet_value %Iterator.4f0, (%custom_witness.df0c4f.1) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.993: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.2fa) [concrete]
// CHECK:STDOUT:   %.61a: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.993, %Destroy.facet.2fa [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.735: %CppRangeForIterate.assoc_type = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [concrete = constants.%assoc2]
// CHECK:STDOUT:   %Core.import_ref.572: %CppRangeForIterate.assoc_type = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [concrete = constants.%assoc3]
// CHECK:STDOUT:   %Core.import_ref.213: @CppRangeForIterate.WithSelf.%CppRangeForIterate.WithSelf.Begin.type (%CppRangeForIterate.WithSelf.Begin.type.753895.1) = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @CppRangeForIterate.WithSelf.%CppRangeForIterate.WithSelf.Begin (constants.%CppRangeForIterate.WithSelf.Begin.2b31f7.1)]
// CHECK:STDOUT:   %Core.import_ref.730: @CppRangeForIterate.WithSelf.%CppRangeForIterate.WithSelf.End.type (%CppRangeForIterate.WithSelf.End.type.056be7.1) = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @CppRangeForIterate.WithSelf.%CppRangeForIterate.WithSelf.End (constants.%CppRangeForIterate.WithSelf.End.a331a3.1)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Test(%R.loc40_10.2: %CppRangeForIterate_where.type) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %CppRangeForIterate.lookup_impl_witness: <witness> = lookup_impl_witness %R.loc40_10.1, @CppRangeForIterate [symbolic = %CppRangeForIterate.lookup_impl_witness (constants.%CppRangeForIterate.lookup_impl_witness.edd1fe.2)]
// CHECK:STDOUT:   %CppRangeForIterate.facet: %CppRangeForIterate.type = facet_value %R.as_type.loc40_113.1, (%CppRangeForIterate.lookup_impl_witness) [symbolic = %CppRangeForIterate.facet (constants.%CppRangeForIterate.facet.0d2)]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.Begin.type: type = fn_type @CppRangeForIterate.WithSelf.Begin, @CppRangeForIterate.WithSelf(%CppRangeForIterate.facet) [symbolic = %CppRangeForIterate.WithSelf.Begin.type (constants.%CppRangeForIterate.WithSelf.Begin.type.57e)]
// CHECK:STDOUT:   %.loc42_4: type = fn_type_with_self_type %CppRangeForIterate.WithSelf.Begin.type, %CppRangeForIterate.facet [symbolic = %.loc42_4 (constants.%.b85)]
// CHECK:STDOUT:   %impl.elem2.loc42_4.2: @Test.%.loc42_4 (%.b85) = impl_witness_access %CppRangeForIterate.lookup_impl_witness, element2 [symbolic = %impl.elem2.loc42_4.2 (constants.%impl.elem2)]
// CHECK:STDOUT:   %specific_impl_fn.loc42_4.2: <specific function> = specific_impl_function %impl.elem2.loc42_4.2, @CppRangeForIterate.WithSelf.Begin(%CppRangeForIterate.facet) [symbolic = %specific_impl_fn.loc42_4.2 (constants.%specific_impl_fn.c0f)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.End.type: type = fn_type @CppRangeForIterate.WithSelf.End, @CppRangeForIterate.WithSelf(%CppRangeForIterate.facet) [symbolic = %CppRangeForIterate.WithSelf.End.type (constants.%CppRangeForIterate.WithSelf.End.type.04d)]
// CHECK:STDOUT:   %.loc43_4: type = fn_type_with_self_type %CppRangeForIterate.WithSelf.End.type, %CppRangeForIterate.facet [symbolic = %.loc43_4 (constants.%.b40)]
// CHECK:STDOUT:   %impl.elem3.loc43_4.2: @Test.%.loc43_4 (%.b40) = impl_witness_access %CppRangeForIterate.lookup_impl_witness, element3 [symbolic = %impl.elem3.loc43_4.2 (constants.%impl.elem3)]
// CHECK:STDOUT:   %specific_impl_fn.loc43_4.2: <specific function> = specific_impl_function %impl.elem3.loc43_4.2, @CppRangeForIterate.WithSelf.End(%CppRangeForIterate.facet) [symbolic = %specific_impl_fn.loc43_4.2 (constants.%specific_impl_fn.d6a)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.loc43: <witness> = lookup_impl_witness %impl.elem1.1, @Destroy [symbolic = %Destroy.lookup_impl_witness.loc43 (constants.%Destroy.lookup_impl_witness.a8f)]
// CHECK:STDOUT:   %Destroy.facet.loc43: %Destroy.type = facet_value %impl.elem1.1, (%Destroy.lookup_impl_witness.loc43) [symbolic = %Destroy.facet.loc43 (constants.%Destroy.facet.543)]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.loc43: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.loc43) [symbolic = %Destroy.WithSelf.Op.type.loc43 (constants.%Destroy.WithSelf.Op.type.366)]
// CHECK:STDOUT:   %.loc43_9.3: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.loc43, %Destroy.facet.loc43 [symbolic = %.loc43_9.3 (constants.%.b83)]
// CHECK:STDOUT:   %impl.elem0.loc43_9.2: @Test.%.loc43_9.3 (%.b83) = impl_witness_access %Destroy.lookup_impl_witness.loc43, element0 [symbolic = %impl.elem0.loc43_9.2 (constants.%impl.elem0.acf)]
// CHECK:STDOUT:   %specific_impl_fn.loc43_9.2: <specific function> = specific_impl_function %impl.elem0.loc43_9.2, @Destroy.WithSelf.Op(%Destroy.facet.loc43) [symbolic = %specific_impl_fn.loc43_9.2 (constants.%specific_impl_fn.add)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.loc42: <witness> = lookup_impl_witness %impl.elem0.1, @Destroy [symbolic = %Destroy.lookup_impl_witness.loc42 (constants.%Destroy.lookup_impl_witness.089)]
// CHECK:STDOUT:   %Destroy.facet.loc42: %Destroy.type = facet_value %impl.elem0.1, (%Destroy.lookup_impl_witness.loc42) [symbolic = %Destroy.facet.loc42 (constants.%Destroy.facet.5e8)]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.loc42: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.loc42) [symbolic = %Destroy.WithSelf.Op.type.loc42 (constants.%Destroy.WithSelf.Op.type.7cb)]
// CHECK:STDOUT:   %.loc42_11.3: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.loc42, %Destroy.facet.loc42 [symbolic = %.loc42_11.3 (constants.%.a68)]
// CHECK:STDOUT:   %impl.elem0.loc42_11.2: @Test.%.loc42_11.3 (%.a68) = impl_witness_access %Destroy.lookup_impl_witness.loc42, element0 [symbolic = %impl.elem0.loc42_11.2 (constants.%impl.elem0.f59)]
// CHECK:STDOUT:   %specific_impl_fn.loc42_11.2: <specific function> = specific_impl_function %impl.elem0.loc42_11.2, @Destroy.WithSelf.Op(%Destroy.facet.loc42) [symbolic = %specific_impl_fn.loc42_11.2 (constants.%specific_impl_fn.f06)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%r.param: ref @Test.%R.as_type.loc40_113.1 (%R.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %r.ref.loc42: ref @Test.%R.as_type.loc40_113.1 (%R.as_type) = name_ref r, %r
// CHECK:STDOUT:     %Begin.ref: %CppRangeForIterate.assoc_type = name_ref Begin, imports.%Core.import_ref.735 [concrete = constants.%assoc2]
// CHECK:STDOUT:     %impl.elem2.loc42_4.1: @Test.%.loc42_4 (%.b85) = impl_witness_access constants.%CppRangeForIterate.lookup_impl_witness.edd1fe.2, element2 [symbolic = %impl.elem2.loc42_4.2 (constants.%impl.elem2)]
// CHECK:STDOUT:     %bound_method.loc42_4: <bound method> = bound_method %r.ref.loc42, %impl.elem2.loc42_4.1
// CHECK:STDOUT:     %specific_impl_fn.loc42_4.1: <specific function> = specific_impl_function %impl.elem2.loc42_4.1, @CppRangeForIterate.WithSelf.Begin(constants.%CppRangeForIterate.facet.0d2) [symbolic = %specific_impl_fn.loc42_4.2 (constants.%specific_impl_fn.c0f)]
// CHECK:STDOUT:     %bound_method.loc42_11.1: <bound method> = bound_method %r.ref.loc42, %specific_impl_fn.loc42_4.1
// CHECK:STDOUT:     %.loc42_11.1: ref @Test.%impl.elem0.1 (%impl.elem0.d5966e.2) = temporary_storage
// CHECK:STDOUT:     %CppRangeForIterate.WithSelf.Begin.call: init @Test.%impl.elem0.1 (%impl.elem0.d5966e.2) to %.loc42_11.1 = call %bound_method.loc42_11.1(%r.ref.loc42)
// CHECK:STDOUT:     %.loc42_11.2: ref @Test.%impl.elem0.1 (%impl.elem0.d5966e.2) = temporary %.loc42_11.1, %CppRangeForIterate.WithSelf.Begin.call
// CHECK:STDOUT:     %r.ref.loc43: ref @Test.%R.as_type.loc40_113.1 (%R.as_type) = name_ref r, %r
// CHECK:STDOUT:     %End.ref: %CppRangeForIterate.assoc_type = name_ref End, imports.%Core.import_ref.572 [concrete = constants.%assoc3]
// CHECK:STDOUT:     %impl.elem3.loc43_4.1: @Test.%.loc43_4 (%.b40) = impl_witness_access constants.%CppRangeForIterate.lookup_impl_witness.edd1fe.2, element3 [symbolic = %impl.elem3.loc43_4.2 (constants.%impl.elem3)]
// CHECK:STDOUT:     %bound_method.loc43_4: <bound method> = bound_method %r.ref.loc43, %impl.elem3.loc43_4.1
// CHECK:STDOUT:     %specific_impl_fn.loc43_4.1: <specific function> = specific_impl_function %impl.elem3.loc43_4.1, @CppRangeForIterate.WithSelf.End(constants.%CppRangeForIterate.facet.0d2) [symbolic = %specific_impl_fn.loc43_4.2 (constants.%specific_impl_fn.d6a)]
// CHECK:STDOUT:     %bound_method.loc43_9.1: <bound method> = bound_method %r.ref.loc43, %specific_impl_fn.loc43_4.1
// CHECK:STDOUT:     %.loc43_9.1: ref @Test.%impl.elem1.1 (%impl.elem1.bdcd2d.2) = temporary_storage
// CHECK:STDOUT:     %CppRangeForIterate.WithSelf.End.call: init @Test.%impl.elem1.1 (%impl.elem1.bdcd2d.2) to %.loc43_9.1 = call %bound_method.loc43_9.1(%r.ref.loc43)
// CHECK:STDOUT:     %.loc43_9.2: ref @Test.%impl.elem1.1 (%impl.elem1.bdcd2d.2) = temporary %.loc43_9.1, %CppRangeForIterate.WithSelf.End.call
// CHECK:STDOUT:     %impl.elem0.loc43_9.1: @Test.%.loc43_9.3 (%.b83) = impl_witness_access constants.%Destroy.lookup_impl_witness.a8f, element0 [symbolic = %impl.elem0.loc43_9.2 (constants.%impl.elem0.acf)]
// CHECK:STDOUT:     %bound_method.loc43_9.2: <bound method> = bound_method %.loc43_9.2, %impl.elem0.loc43_9.1
// CHECK:STDOUT:     %specific_impl_fn.loc43_9.1: <specific function> = specific_impl_function %impl.elem0.loc43_9.1, @Destroy.WithSelf.Op(constants.%Destroy.facet.543) [symbolic = %specific_impl_fn.loc43_9.2 (constants.%specific_impl_fn.add)]
// CHECK:STDOUT:     %bound_method.loc43_9.3: <bound method> = bound_method %.loc43_9.2, %specific_impl_fn.loc43_9.1
// CHECK:STDOUT:     %Destroy.WithSelf.Op.call.loc43: init %empty_tuple.type = call %bound_method.loc43_9.3(%.loc43_9.2)
// CHECK:STDOUT:     %impl.elem0.loc42_11.1: @Test.%.loc42_11.3 (%.a68) = impl_witness_access constants.%Destroy.lookup_impl_witness.089, element0 [symbolic = %impl.elem0.loc42_11.2 (constants.%impl.elem0.f59)]
// CHECK:STDOUT:     %bound_method.loc42_11.2: <bound method> = bound_method %.loc42_11.2, %impl.elem0.loc42_11.1
// CHECK:STDOUT:     %specific_impl_fn.loc42_11.1: <specific function> = specific_impl_function %impl.elem0.loc42_11.1, @Destroy.WithSelf.Op(constants.%Destroy.facet.5e8) [symbolic = %specific_impl_fn.loc42_11.2 (constants.%specific_impl_fn.f06)]
// CHECK:STDOUT:     %bound_method.loc42_11.3: <bound method> = bound_method %.loc42_11.2, %specific_impl_fn.loc42_11.1
// CHECK:STDOUT:     %Destroy.WithSelf.Op.call.loc42: init %empty_tuple.type = call %bound_method.loc42_11.3(%.loc42_11.2)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(constants.%R) {
// CHECK:STDOUT:   %R.loc40_10.1 => constants.%R
// CHECK:STDOUT:   %R.as_type.loc40_113.1 => constants.%R.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.26c6ea.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(constants.%facet_value.99a) {
// CHECK:STDOUT:   %R.loc40_10.1 => constants.%facet_value.99a
// CHECK:STDOUT:   %R.as_type.loc40_113.1 => constants.%MutableRange
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.b76
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc40 => constants.%complete_type
// CHECK:STDOUT:   %CppRangeForIterate.lookup_impl_witness => constants.%custom_witness.292
// CHECK:STDOUT:   %CppRangeForIterate.facet => constants.%CppRangeForIterate.facet.de2
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.Begin.type => constants.%CppRangeForIterate.WithSelf.Begin.type.82a
// CHECK:STDOUT:   %.loc42_4 => constants.%.26f
// CHECK:STDOUT:   %impl.elem2.loc42_4.2 => constants.%MutableRange.Begin
// CHECK:STDOUT:   %specific_impl_fn.loc42_4.2 => constants.%MutableRange.Begin
// CHECK:STDOUT:   %impl.elem0.1 => constants.%Iterator.826
// CHECK:STDOUT:   %require_complete.1 => constants.%complete_type
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.End.type => constants.%CppRangeForIterate.WithSelf.End.type.9fe
// CHECK:STDOUT:   %.loc43_4 => constants.%.6d3
// CHECK:STDOUT:   %impl.elem3.loc43_4.2 => constants.%MutableRange.End
// CHECK:STDOUT:   %specific_impl_fn.loc43_4.2 => constants.%MutableRange.End
// CHECK:STDOUT:   %impl.elem1.1 => constants.%Sentinel.6eb
// CHECK:STDOUT:   %require_complete.2 => constants.%complete_type
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.loc43 => constants.%custom_witness.ef99ee.2
// CHECK:STDOUT:   %Destroy.facet.loc43 => constants.%Destroy.facet.959
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.loc43 => constants.%Destroy.WithSelf.Op.type.6d7
// CHECK:STDOUT:   %.loc43_9.3 => constants.%.3e7
// CHECK:STDOUT:   %impl.elem0.loc43_9.2 => constants.%Sentinel.Op.f05
// CHECK:STDOUT:   %specific_impl_fn.loc43_9.2 => constants.%Sentinel.Op.f05
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.loc42 => constants.%custom_witness.ef99ee.1
// CHECK:STDOUT:   %Destroy.facet.loc42 => constants.%Destroy.facet.f6b
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.loc42 => constants.%Destroy.WithSelf.Op.type.a45
// CHECK:STDOUT:   %.loc42_11.3 => constants.%.ee6
// CHECK:STDOUT:   %impl.elem0.loc42_11.2 => constants.%Iterator.Op.f05
// CHECK:STDOUT:   %specific_impl_fn.loc42_11.2 => constants.%Iterator.Op.f05
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(constants.%facet_value.b4b) {
// CHECK:STDOUT:   %R.loc40_10.1 => constants.%facet_value.b4b
// CHECK:STDOUT:   %R.as_type.loc40_113.1 => constants.%ConstRange
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.631
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc40 => constants.%complete_type
// CHECK:STDOUT:   %CppRangeForIterate.lookup_impl_witness => constants.%custom_witness.8cb
// CHECK:STDOUT:   %CppRangeForIterate.facet => constants.%CppRangeForIterate.facet.ef1
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.Begin.type => constants.%CppRangeForIterate.WithSelf.Begin.type.317
// CHECK:STDOUT:   %.loc42_4 => constants.%.3f6
// CHECK:STDOUT:   %impl.elem2.loc42_4.2 => constants.%ConstRange.Begin
// CHECK:STDOUT:   %specific_impl_fn.loc42_4.2 => constants.%ConstRange.Begin
// CHECK:STDOUT:   %impl.elem0.1 => constants.%Iterator.4f0
// CHECK:STDOUT:   %require_complete.1 => constants.%complete_type
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.End.type => constants.%CppRangeForIterate.WithSelf.End.type.120
// CHECK:STDOUT:   %.loc43_4 => constants.%.cd4
// CHECK:STDOUT:   %impl.elem3.loc43_4.2 => constants.%ConstRange.End
// CHECK:STDOUT:   %specific_impl_fn.loc43_4.2 => constants.%ConstRange.End
// CHECK:STDOUT:   %impl.elem1.1 => constants.%Sentinel.45e
// CHECK:STDOUT:   %require_complete.2 => constants.%complete_type
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.loc43 => constants.%custom_witness.df0c4f.2
// CHECK:STDOUT:   %Destroy.facet.loc43 => constants.%Destroy.facet.d9f
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.loc43 => constants.%Destroy.WithSelf.Op.type.39d
// CHECK:STDOUT:   %.loc43_9.3 => constants.%.f49
// CHECK:STDOUT:   %impl.elem0.loc43_9.2 => constants.%Sentinel.Op.7b7
// CHECK:STDOUT:   %specific_impl_fn.loc43_9.2 => constants.%Sentinel.Op.7b7
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.loc42 => constants.%custom_witness.df0c4f.1
// CHECK:STDOUT:   %Destroy.facet.loc42 => constants.%Destroy.facet.2fa
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.loc42 => constants.%Destroy.WithSelf.Op.type.993
// CHECK:STDOUT:   %.loc42_11.3 => constants.%.61a
// CHECK:STDOUT:   %impl.elem0.loc42_11.2 => constants.%Iterator.Op.7b7
// CHECK:STDOUT:   %specific_impl_fn.loc42_11.2 => constants.%Iterator.Op.7b7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- adl_return_same_types.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %CppRangeForIterate.type: type = facet_type <@CppRangeForIterate> [concrete]
// CHECK:STDOUT:   %Self.6c6: %CppRangeForIterate.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.End.type.056be7.1: type = fn_type @CppRangeForIterate.WithSelf.End, @CppRangeForIterate.WithSelf(%Self.6c6) [symbolic]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.End.a331a3.1: %CppRangeForIterate.WithSelf.End.type.056be7.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.Begin.type.753895.1: type = fn_type @CppRangeForIterate.WithSelf.Begin, @CppRangeForIterate.WithSelf(%Self.6c6) [symbolic]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.Begin.2b31f7.1: %CppRangeForIterate.WithSelf.Begin.type.753895.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %.Self.10b: %CppRangeForIterate.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %CppRangeForIterate.assoc_type: type = assoc_entity_type @CppRangeForIterate [concrete]
// CHECK:STDOUT:   %CppRangeForIterate.lookup_impl_witness.5bf: <witness> = lookup_impl_witness %.Self.10b, @CppRangeForIterate [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.d47: type = impl_witness_access %CppRangeForIterate.lookup_impl_witness.5bf, element0 [symbolic_self]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %impl.elem1.846: type = impl_witness_access %CppRangeForIterate.lookup_impl_witness.5bf, element1 [symbolic_self]
// CHECK:STDOUT:   %CppRangeForIterate_where.type: type = facet_type <@CppRangeForIterate where %impl.elem0.d47 impls @Destroy and %impl.elem1.846 impls @Destroy> [concrete]
// CHECK:STDOUT:   %R: %CppRangeForIterate_where.type = symbolic_binding R, 0 [symbolic]
// CHECK:STDOUT:   %R.as_type: type = facet_access_type %R [symbolic]
// CHECK:STDOUT:   %pattern_type.26c6ea.2: type = pattern_type %R.as_type [symbolic]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %assoc2: %CppRangeForIterate.assoc_type = assoc_entity element2, imports.%Core.import_ref.213 [concrete]
// CHECK:STDOUT:   %CppRangeForIterate.lookup_impl_witness.edd1fe.2: <witness> = lookup_impl_witness %R, @CppRangeForIterate [symbolic]
// CHECK:STDOUT:   %CppRangeForIterate.facet.0d2: %CppRangeForIterate.type = facet_value %R.as_type, (%CppRangeForIterate.lookup_impl_witness.edd1fe.2) [symbolic]
// CHECK:STDOUT:   %impl.elem0.d5966e.2: type = impl_witness_access %CppRangeForIterate.lookup_impl_witness.edd1fe.2, element0 [symbolic]
// CHECK:STDOUT:   %impl.elem1.bdcd2d.2: type = impl_witness_access %CppRangeForIterate.lookup_impl_witness.edd1fe.2, element1 [symbolic]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.Begin.type.57e: type = fn_type @CppRangeForIterate.WithSelf.Begin, @CppRangeForIterate.WithSelf(%CppRangeForIterate.facet.0d2) [symbolic]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.End.type.04d: type = fn_type @CppRangeForIterate.WithSelf.End, @CppRangeForIterate.WithSelf(%CppRangeForIterate.facet.0d2) [symbolic]
// CHECK:STDOUT:   %.b85: type = fn_type_with_self_type %CppRangeForIterate.WithSelf.Begin.type.57e, %CppRangeForIterate.facet.0d2 [symbolic]
// CHECK:STDOUT:   %impl.elem2: %.b85 = impl_witness_access %CppRangeForIterate.lookup_impl_witness.edd1fe.2, element2 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.c0f: <specific function> = specific_impl_function %impl.elem2, @CppRangeForIterate.WithSelf.Begin(%CppRangeForIterate.facet.0d2) [symbolic]
// CHECK:STDOUT:   %assoc3: %CppRangeForIterate.assoc_type = assoc_entity element3, imports.%Core.import_ref.730 [concrete]
// CHECK:STDOUT:   %.b40: type = fn_type_with_self_type %CppRangeForIterate.WithSelf.End.type.04d, %CppRangeForIterate.facet.0d2 [symbolic]
// CHECK:STDOUT:   %impl.elem3: %.b40 = impl_witness_access %CppRangeForIterate.lookup_impl_witness.edd1fe.2, element3 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.d6a: <specific function> = specific_impl_function %impl.elem3, @CppRangeForIterate.WithSelf.End(%CppRangeForIterate.facet.0d2) [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.a8f: <witness> = lookup_impl_witness %impl.elem1.bdcd2d.2, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.543: %Destroy.type = facet_value %impl.elem1.bdcd2d.2, (%Destroy.lookup_impl_witness.a8f) [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.366: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.543) [symbolic]
// CHECK:STDOUT:   %.b83: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.366, %Destroy.facet.543 [symbolic]
// CHECK:STDOUT:   %impl.elem0.acf: %.b83 = impl_witness_access %Destroy.lookup_impl_witness.a8f, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.add: <specific function> = specific_impl_function %impl.elem0.acf, @Destroy.WithSelf.Op(%Destroy.facet.543) [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.089: <witness> = lookup_impl_witness %impl.elem0.d5966e.2, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.5e8: %Destroy.type = facet_value %impl.elem0.d5966e.2, (%Destroy.lookup_impl_witness.089) [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.7cb: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.5e8) [symbolic]
// CHECK:STDOUT:   %.a68: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.7cb, %Destroy.facet.5e8 [symbolic]
// CHECK:STDOUT:   %impl.elem0.f59: %.a68 = impl_witness_access %Destroy.lookup_impl_witness.089, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.f06: <specific function> = specific_impl_function %impl.elem0.f59, @Destroy.WithSelf.Op(%Destroy.facet.5e8) [symbolic]
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
// CHECK:STDOUT:   %custom_witness.25d: <witness> = custom_witness (%Iterator.b8c, %Iterator.b8c, %Begin.1544bb.1, %End.be76e1.1), @CppRangeForIterate [concrete]
// CHECK:STDOUT:   %CppRangeForIterate.facet.5b0: %CppRangeForIterate.type = facet_value %MutableRange, (%custom_witness.25d) [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.910: type = fn_type @Iterator.Op.1 [concrete]
// CHECK:STDOUT:   %Iterator.Op.cbb: %Iterator.Op.type.910 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.fe6: <witness> = custom_witness (%Iterator.Op.cbb), @Destroy [concrete]
// CHECK:STDOUT:   %facet_value.589: %CppRangeForIterate_where.type = facet_value %MutableRange, (%custom_witness.25d, %custom_witness.fe6) [concrete]
// CHECK:STDOUT:   %Iterator.080: type = class_type @Iterator.2 [concrete]
// CHECK:STDOUT:   %Begin.type.793876.2: type = fn_type @Begin.2 [concrete]
// CHECK:STDOUT:   %Begin.1544bb.2: %Begin.type.793876.2 = struct_value () [concrete]
// CHECK:STDOUT:   %End.type.16398b.2: type = fn_type @End.2 [concrete]
// CHECK:STDOUT:   %End.be76e1.2: %End.type.16398b.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.4c3: <witness> = custom_witness (%Iterator.080, %Iterator.080, %Begin.1544bb.2, %End.be76e1.2), @CppRangeForIterate [concrete]
// CHECK:STDOUT:   %CppRangeForIterate.facet.379: %CppRangeForIterate.type = facet_value %ConstRange, (%custom_witness.4c3) [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.587: type = fn_type @Iterator.Op.2 [concrete]
// CHECK:STDOUT:   %Iterator.Op.dc6: %Iterator.Op.type.587 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.166: <witness> = custom_witness (%Iterator.Op.dc6), @Destroy [concrete]
// CHECK:STDOUT:   %facet_value.105: %CppRangeForIterate_where.type = facet_value %ConstRange, (%custom_witness.4c3, %custom_witness.166) [concrete]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.Begin.type.901: type = fn_type @CppRangeForIterate.WithSelf.Begin, @CppRangeForIterate.WithSelf(%CppRangeForIterate.facet.5b0) [concrete]
// CHECK:STDOUT:   %.8fa: type = fn_type_with_self_type %CppRangeForIterate.WithSelf.Begin.type.901, %CppRangeForIterate.facet.5b0 [concrete]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.End.type.3a5: type = fn_type @CppRangeForIterate.WithSelf.End, @CppRangeForIterate.WithSelf(%CppRangeForIterate.facet.5b0) [concrete]
// CHECK:STDOUT:   %.374: type = fn_type_with_self_type %CppRangeForIterate.WithSelf.End.type.3a5, %CppRangeForIterate.facet.5b0 [concrete]
// CHECK:STDOUT:   %Destroy.facet.6ae: %Destroy.type = facet_value %Iterator.b8c, (%custom_witness.fe6) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.018: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.6ae) [concrete]
// CHECK:STDOUT:   %.e53: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.018, %Destroy.facet.6ae [concrete]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.Begin.type.b5c: type = fn_type @CppRangeForIterate.WithSelf.Begin, @CppRangeForIterate.WithSelf(%CppRangeForIterate.facet.379) [concrete]
// CHECK:STDOUT:   %.093: type = fn_type_with_self_type %CppRangeForIterate.WithSelf.Begin.type.b5c, %CppRangeForIterate.facet.379 [concrete]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.End.type.735: type = fn_type @CppRangeForIterate.WithSelf.End, @CppRangeForIterate.WithSelf(%CppRangeForIterate.facet.379) [concrete]
// CHECK:STDOUT:   %.e31: type = fn_type_with_self_type %CppRangeForIterate.WithSelf.End.type.735, %CppRangeForIterate.facet.379 [concrete]
// CHECK:STDOUT:   %Destroy.facet.488: %Destroy.type = facet_value %Iterator.080, (%custom_witness.166) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.488: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.488) [concrete]
// CHECK:STDOUT:   %.8da: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.488, %Destroy.facet.488 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.735: %CppRangeForIterate.assoc_type = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [concrete = constants.%assoc2]
// CHECK:STDOUT:   %Core.import_ref.572: %CppRangeForIterate.assoc_type = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [concrete = constants.%assoc3]
// CHECK:STDOUT:   %Core.import_ref.213: @CppRangeForIterate.WithSelf.%CppRangeForIterate.WithSelf.Begin.type (%CppRangeForIterate.WithSelf.Begin.type.753895.1) = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @CppRangeForIterate.WithSelf.%CppRangeForIterate.WithSelf.Begin (constants.%CppRangeForIterate.WithSelf.Begin.2b31f7.1)]
// CHECK:STDOUT:   %Core.import_ref.730: @CppRangeForIterate.WithSelf.%CppRangeForIterate.WithSelf.End.type (%CppRangeForIterate.WithSelf.End.type.056be7.1) = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @CppRangeForIterate.WithSelf.%CppRangeForIterate.WithSelf.End (constants.%CppRangeForIterate.WithSelf.End.a331a3.1)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Test(%R.loc32_10.2: %CppRangeForIterate_where.type) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %CppRangeForIterate.lookup_impl_witness: <witness> = lookup_impl_witness %R.loc32_10.1, @CppRangeForIterate [symbolic = %CppRangeForIterate.lookup_impl_witness (constants.%CppRangeForIterate.lookup_impl_witness.edd1fe.2)]
// CHECK:STDOUT:   %CppRangeForIterate.facet: %CppRangeForIterate.type = facet_value %R.as_type.loc32_113.1, (%CppRangeForIterate.lookup_impl_witness) [symbolic = %CppRangeForIterate.facet (constants.%CppRangeForIterate.facet.0d2)]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.Begin.type: type = fn_type @CppRangeForIterate.WithSelf.Begin, @CppRangeForIterate.WithSelf(%CppRangeForIterate.facet) [symbolic = %CppRangeForIterate.WithSelf.Begin.type (constants.%CppRangeForIterate.WithSelf.Begin.type.57e)]
// CHECK:STDOUT:   %.loc34_4: type = fn_type_with_self_type %CppRangeForIterate.WithSelf.Begin.type, %CppRangeForIterate.facet [symbolic = %.loc34_4 (constants.%.b85)]
// CHECK:STDOUT:   %impl.elem2.loc34_4.2: @Test.%.loc34_4 (%.b85) = impl_witness_access %CppRangeForIterate.lookup_impl_witness, element2 [symbolic = %impl.elem2.loc34_4.2 (constants.%impl.elem2)]
// CHECK:STDOUT:   %specific_impl_fn.loc34_4.2: <specific function> = specific_impl_function %impl.elem2.loc34_4.2, @CppRangeForIterate.WithSelf.Begin(%CppRangeForIterate.facet) [symbolic = %specific_impl_fn.loc34_4.2 (constants.%specific_impl_fn.c0f)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.End.type: type = fn_type @CppRangeForIterate.WithSelf.End, @CppRangeForIterate.WithSelf(%CppRangeForIterate.facet) [symbolic = %CppRangeForIterate.WithSelf.End.type (constants.%CppRangeForIterate.WithSelf.End.type.04d)]
// CHECK:STDOUT:   %.loc35_4: type = fn_type_with_self_type %CppRangeForIterate.WithSelf.End.type, %CppRangeForIterate.facet [symbolic = %.loc35_4 (constants.%.b40)]
// CHECK:STDOUT:   %impl.elem3.loc35_4.2: @Test.%.loc35_4 (%.b40) = impl_witness_access %CppRangeForIterate.lookup_impl_witness, element3 [symbolic = %impl.elem3.loc35_4.2 (constants.%impl.elem3)]
// CHECK:STDOUT:   %specific_impl_fn.loc35_4.2: <specific function> = specific_impl_function %impl.elem3.loc35_4.2, @CppRangeForIterate.WithSelf.End(%CppRangeForIterate.facet) [symbolic = %specific_impl_fn.loc35_4.2 (constants.%specific_impl_fn.d6a)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.loc35: <witness> = lookup_impl_witness %impl.elem1.1, @Destroy [symbolic = %Destroy.lookup_impl_witness.loc35 (constants.%Destroy.lookup_impl_witness.a8f)]
// CHECK:STDOUT:   %Destroy.facet.loc35: %Destroy.type = facet_value %impl.elem1.1, (%Destroy.lookup_impl_witness.loc35) [symbolic = %Destroy.facet.loc35 (constants.%Destroy.facet.543)]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.loc35: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.loc35) [symbolic = %Destroy.WithSelf.Op.type.loc35 (constants.%Destroy.WithSelf.Op.type.366)]
// CHECK:STDOUT:   %.loc35_9.3: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.loc35, %Destroy.facet.loc35 [symbolic = %.loc35_9.3 (constants.%.b83)]
// CHECK:STDOUT:   %impl.elem0.loc35_9.2: @Test.%.loc35_9.3 (%.b83) = impl_witness_access %Destroy.lookup_impl_witness.loc35, element0 [symbolic = %impl.elem0.loc35_9.2 (constants.%impl.elem0.acf)]
// CHECK:STDOUT:   %specific_impl_fn.loc35_9.2: <specific function> = specific_impl_function %impl.elem0.loc35_9.2, @Destroy.WithSelf.Op(%Destroy.facet.loc35) [symbolic = %specific_impl_fn.loc35_9.2 (constants.%specific_impl_fn.add)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.loc34: <witness> = lookup_impl_witness %impl.elem0.1, @Destroy [symbolic = %Destroy.lookup_impl_witness.loc34 (constants.%Destroy.lookup_impl_witness.089)]
// CHECK:STDOUT:   %Destroy.facet.loc34: %Destroy.type = facet_value %impl.elem0.1, (%Destroy.lookup_impl_witness.loc34) [symbolic = %Destroy.facet.loc34 (constants.%Destroy.facet.5e8)]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.loc34: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.loc34) [symbolic = %Destroy.WithSelf.Op.type.loc34 (constants.%Destroy.WithSelf.Op.type.7cb)]
// CHECK:STDOUT:   %.loc34_11.3: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.loc34, %Destroy.facet.loc34 [symbolic = %.loc34_11.3 (constants.%.a68)]
// CHECK:STDOUT:   %impl.elem0.loc34_11.2: @Test.%.loc34_11.3 (%.a68) = impl_witness_access %Destroy.lookup_impl_witness.loc34, element0 [symbolic = %impl.elem0.loc34_11.2 (constants.%impl.elem0.f59)]
// CHECK:STDOUT:   %specific_impl_fn.loc34_11.2: <specific function> = specific_impl_function %impl.elem0.loc34_11.2, @Destroy.WithSelf.Op(%Destroy.facet.loc34) [symbolic = %specific_impl_fn.loc34_11.2 (constants.%specific_impl_fn.f06)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%r.param: ref @Test.%R.as_type.loc32_113.1 (%R.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %r.ref.loc34: ref @Test.%R.as_type.loc32_113.1 (%R.as_type) = name_ref r, %r
// CHECK:STDOUT:     %Begin.ref: %CppRangeForIterate.assoc_type = name_ref Begin, imports.%Core.import_ref.735 [concrete = constants.%assoc2]
// CHECK:STDOUT:     %impl.elem2.loc34_4.1: @Test.%.loc34_4 (%.b85) = impl_witness_access constants.%CppRangeForIterate.lookup_impl_witness.edd1fe.2, element2 [symbolic = %impl.elem2.loc34_4.2 (constants.%impl.elem2)]
// CHECK:STDOUT:     %bound_method.loc34_4: <bound method> = bound_method %r.ref.loc34, %impl.elem2.loc34_4.1
// CHECK:STDOUT:     %specific_impl_fn.loc34_4.1: <specific function> = specific_impl_function %impl.elem2.loc34_4.1, @CppRangeForIterate.WithSelf.Begin(constants.%CppRangeForIterate.facet.0d2) [symbolic = %specific_impl_fn.loc34_4.2 (constants.%specific_impl_fn.c0f)]
// CHECK:STDOUT:     %bound_method.loc34_11.1: <bound method> = bound_method %r.ref.loc34, %specific_impl_fn.loc34_4.1
// CHECK:STDOUT:     %.loc34_11.1: ref @Test.%impl.elem0.1 (%impl.elem0.d5966e.2) = temporary_storage
// CHECK:STDOUT:     %CppRangeForIterate.WithSelf.Begin.call: init @Test.%impl.elem0.1 (%impl.elem0.d5966e.2) to %.loc34_11.1 = call %bound_method.loc34_11.1(%r.ref.loc34)
// CHECK:STDOUT:     %.loc34_11.2: ref @Test.%impl.elem0.1 (%impl.elem0.d5966e.2) = temporary %.loc34_11.1, %CppRangeForIterate.WithSelf.Begin.call
// CHECK:STDOUT:     %r.ref.loc35: ref @Test.%R.as_type.loc32_113.1 (%R.as_type) = name_ref r, %r
// CHECK:STDOUT:     %End.ref: %CppRangeForIterate.assoc_type = name_ref End, imports.%Core.import_ref.572 [concrete = constants.%assoc3]
// CHECK:STDOUT:     %impl.elem3.loc35_4.1: @Test.%.loc35_4 (%.b40) = impl_witness_access constants.%CppRangeForIterate.lookup_impl_witness.edd1fe.2, element3 [symbolic = %impl.elem3.loc35_4.2 (constants.%impl.elem3)]
// CHECK:STDOUT:     %bound_method.loc35_4: <bound method> = bound_method %r.ref.loc35, %impl.elem3.loc35_4.1
// CHECK:STDOUT:     %specific_impl_fn.loc35_4.1: <specific function> = specific_impl_function %impl.elem3.loc35_4.1, @CppRangeForIterate.WithSelf.End(constants.%CppRangeForIterate.facet.0d2) [symbolic = %specific_impl_fn.loc35_4.2 (constants.%specific_impl_fn.d6a)]
// CHECK:STDOUT:     %bound_method.loc35_9.1: <bound method> = bound_method %r.ref.loc35, %specific_impl_fn.loc35_4.1
// CHECK:STDOUT:     %.loc35_9.1: ref @Test.%impl.elem1.1 (%impl.elem1.bdcd2d.2) = temporary_storage
// CHECK:STDOUT:     %CppRangeForIterate.WithSelf.End.call: init @Test.%impl.elem1.1 (%impl.elem1.bdcd2d.2) to %.loc35_9.1 = call %bound_method.loc35_9.1(%r.ref.loc35)
// CHECK:STDOUT:     %.loc35_9.2: ref @Test.%impl.elem1.1 (%impl.elem1.bdcd2d.2) = temporary %.loc35_9.1, %CppRangeForIterate.WithSelf.End.call
// CHECK:STDOUT:     %impl.elem0.loc35_9.1: @Test.%.loc35_9.3 (%.b83) = impl_witness_access constants.%Destroy.lookup_impl_witness.a8f, element0 [symbolic = %impl.elem0.loc35_9.2 (constants.%impl.elem0.acf)]
// CHECK:STDOUT:     %bound_method.loc35_9.2: <bound method> = bound_method %.loc35_9.2, %impl.elem0.loc35_9.1
// CHECK:STDOUT:     %specific_impl_fn.loc35_9.1: <specific function> = specific_impl_function %impl.elem0.loc35_9.1, @Destroy.WithSelf.Op(constants.%Destroy.facet.543) [symbolic = %specific_impl_fn.loc35_9.2 (constants.%specific_impl_fn.add)]
// CHECK:STDOUT:     %bound_method.loc35_9.3: <bound method> = bound_method %.loc35_9.2, %specific_impl_fn.loc35_9.1
// CHECK:STDOUT:     %Destroy.WithSelf.Op.call.loc35: init %empty_tuple.type = call %bound_method.loc35_9.3(%.loc35_9.2)
// CHECK:STDOUT:     %impl.elem0.loc34_11.1: @Test.%.loc34_11.3 (%.a68) = impl_witness_access constants.%Destroy.lookup_impl_witness.089, element0 [symbolic = %impl.elem0.loc34_11.2 (constants.%impl.elem0.f59)]
// CHECK:STDOUT:     %bound_method.loc34_11.2: <bound method> = bound_method %.loc34_11.2, %impl.elem0.loc34_11.1
// CHECK:STDOUT:     %specific_impl_fn.loc34_11.1: <specific function> = specific_impl_function %impl.elem0.loc34_11.1, @Destroy.WithSelf.Op(constants.%Destroy.facet.5e8) [symbolic = %specific_impl_fn.loc34_11.2 (constants.%specific_impl_fn.f06)]
// CHECK:STDOUT:     %bound_method.loc34_11.3: <bound method> = bound_method %.loc34_11.2, %specific_impl_fn.loc34_11.1
// CHECK:STDOUT:     %Destroy.WithSelf.Op.call.loc34: init %empty_tuple.type = call %bound_method.loc34_11.3(%.loc34_11.2)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(constants.%R) {
// CHECK:STDOUT:   %R.loc32_10.1 => constants.%R
// CHECK:STDOUT:   %R.as_type.loc32_113.1 => constants.%R.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.26c6ea.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(constants.%facet_value.589) {
// CHECK:STDOUT:   %R.loc32_10.1 => constants.%facet_value.589
// CHECK:STDOUT:   %R.as_type.loc32_113.1 => constants.%MutableRange
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.80f
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc32 => constants.%complete_type
// CHECK:STDOUT:   %CppRangeForIterate.lookup_impl_witness => constants.%custom_witness.25d
// CHECK:STDOUT:   %CppRangeForIterate.facet => constants.%CppRangeForIterate.facet.5b0
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.Begin.type => constants.%CppRangeForIterate.WithSelf.Begin.type.901
// CHECK:STDOUT:   %.loc34_4 => constants.%.8fa
// CHECK:STDOUT:   %impl.elem2.loc34_4.2 => constants.%Begin.1544bb.1
// CHECK:STDOUT:   %specific_impl_fn.loc34_4.2 => constants.%Begin.1544bb.1
// CHECK:STDOUT:   %impl.elem0.1 => constants.%Iterator.b8c
// CHECK:STDOUT:   %require_complete.1 => constants.%complete_type
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.End.type => constants.%CppRangeForIterate.WithSelf.End.type.3a5
// CHECK:STDOUT:   %.loc35_4 => constants.%.374
// CHECK:STDOUT:   %impl.elem3.loc35_4.2 => constants.%End.be76e1.1
// CHECK:STDOUT:   %specific_impl_fn.loc35_4.2 => constants.%End.be76e1.1
// CHECK:STDOUT:   %impl.elem1.1 => constants.%Iterator.b8c
// CHECK:STDOUT:   %require_complete.2 => constants.%complete_type
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.loc35 => constants.%custom_witness.fe6
// CHECK:STDOUT:   %Destroy.facet.loc35 => constants.%Destroy.facet.6ae
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.loc35 => constants.%Destroy.WithSelf.Op.type.018
// CHECK:STDOUT:   %.loc35_9.3 => constants.%.e53
// CHECK:STDOUT:   %impl.elem0.loc35_9.2 => constants.%Iterator.Op.cbb
// CHECK:STDOUT:   %specific_impl_fn.loc35_9.2 => constants.%Iterator.Op.cbb
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.loc34 => constants.%custom_witness.fe6
// CHECK:STDOUT:   %Destroy.facet.loc34 => constants.%Destroy.facet.6ae
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.loc34 => constants.%Destroy.WithSelf.Op.type.018
// CHECK:STDOUT:   %.loc34_11.3 => constants.%.e53
// CHECK:STDOUT:   %impl.elem0.loc34_11.2 => constants.%Iterator.Op.cbb
// CHECK:STDOUT:   %specific_impl_fn.loc34_11.2 => constants.%Iterator.Op.cbb
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(constants.%facet_value.105) {
// CHECK:STDOUT:   %R.loc32_10.1 => constants.%facet_value.105
// CHECK:STDOUT:   %R.as_type.loc32_113.1 => constants.%ConstRange
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.88d
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc32 => constants.%complete_type
// CHECK:STDOUT:   %CppRangeForIterate.lookup_impl_witness => constants.%custom_witness.4c3
// CHECK:STDOUT:   %CppRangeForIterate.facet => constants.%CppRangeForIterate.facet.379
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.Begin.type => constants.%CppRangeForIterate.WithSelf.Begin.type.b5c
// CHECK:STDOUT:   %.loc34_4 => constants.%.093
// CHECK:STDOUT:   %impl.elem2.loc34_4.2 => constants.%Begin.1544bb.2
// CHECK:STDOUT:   %specific_impl_fn.loc34_4.2 => constants.%Begin.1544bb.2
// CHECK:STDOUT:   %impl.elem0.1 => constants.%Iterator.080
// CHECK:STDOUT:   %require_complete.1 => constants.%complete_type
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.End.type => constants.%CppRangeForIterate.WithSelf.End.type.735
// CHECK:STDOUT:   %.loc35_4 => constants.%.e31
// CHECK:STDOUT:   %impl.elem3.loc35_4.2 => constants.%End.be76e1.2
// CHECK:STDOUT:   %specific_impl_fn.loc35_4.2 => constants.%End.be76e1.2
// CHECK:STDOUT:   %impl.elem1.1 => constants.%Iterator.080
// CHECK:STDOUT:   %require_complete.2 => constants.%complete_type
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.loc35 => constants.%custom_witness.166
// CHECK:STDOUT:   %Destroy.facet.loc35 => constants.%Destroy.facet.488
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.loc35 => constants.%Destroy.WithSelf.Op.type.488
// CHECK:STDOUT:   %.loc35_9.3 => constants.%.8da
// CHECK:STDOUT:   %impl.elem0.loc35_9.2 => constants.%Iterator.Op.dc6
// CHECK:STDOUT:   %specific_impl_fn.loc35_9.2 => constants.%Iterator.Op.dc6
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.loc34 => constants.%custom_witness.166
// CHECK:STDOUT:   %Destroy.facet.loc34 => constants.%Destroy.facet.488
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.loc34 => constants.%Destroy.WithSelf.Op.type.488
// CHECK:STDOUT:   %.loc34_11.3 => constants.%.8da
// CHECK:STDOUT:   %impl.elem0.loc34_11.2 => constants.%Iterator.Op.dc6
// CHECK:STDOUT:   %specific_impl_fn.loc34_11.2 => constants.%Iterator.Op.dc6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- adl_returns_different_types.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %CppRangeForIterate.type: type = facet_type <@CppRangeForIterate> [concrete]
// CHECK:STDOUT:   %Self.6c6: %CppRangeForIterate.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.End.type.056be7.1: type = fn_type @CppRangeForIterate.WithSelf.End, @CppRangeForIterate.WithSelf(%Self.6c6) [symbolic]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.End.a331a3.1: %CppRangeForIterate.WithSelf.End.type.056be7.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.Begin.type.753895.1: type = fn_type @CppRangeForIterate.WithSelf.Begin, @CppRangeForIterate.WithSelf(%Self.6c6) [symbolic]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.Begin.2b31f7.1: %CppRangeForIterate.WithSelf.Begin.type.753895.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %.Self.10b: %CppRangeForIterate.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %CppRangeForIterate.assoc_type: type = assoc_entity_type @CppRangeForIterate [concrete]
// CHECK:STDOUT:   %CppRangeForIterate.lookup_impl_witness.5bf: <witness> = lookup_impl_witness %.Self.10b, @CppRangeForIterate [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.d47: type = impl_witness_access %CppRangeForIterate.lookup_impl_witness.5bf, element0 [symbolic_self]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %impl.elem1.846: type = impl_witness_access %CppRangeForIterate.lookup_impl_witness.5bf, element1 [symbolic_self]
// CHECK:STDOUT:   %CppRangeForIterate_where.type: type = facet_type <@CppRangeForIterate where %impl.elem0.d47 impls @Destroy and %impl.elem1.846 impls @Destroy> [concrete]
// CHECK:STDOUT:   %R: %CppRangeForIterate_where.type = symbolic_binding R, 0 [symbolic]
// CHECK:STDOUT:   %R.as_type: type = facet_access_type %R [symbolic]
// CHECK:STDOUT:   %pattern_type.26c6ea.2: type = pattern_type %R.as_type [symbolic]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %assoc2: %CppRangeForIterate.assoc_type = assoc_entity element2, imports.%Core.import_ref.213 [concrete]
// CHECK:STDOUT:   %CppRangeForIterate.lookup_impl_witness.edd1fe.2: <witness> = lookup_impl_witness %R, @CppRangeForIterate [symbolic]
// CHECK:STDOUT:   %CppRangeForIterate.facet.0d2: %CppRangeForIterate.type = facet_value %R.as_type, (%CppRangeForIterate.lookup_impl_witness.edd1fe.2) [symbolic]
// CHECK:STDOUT:   %impl.elem0.d5966e.2: type = impl_witness_access %CppRangeForIterate.lookup_impl_witness.edd1fe.2, element0 [symbolic]
// CHECK:STDOUT:   %impl.elem1.bdcd2d.2: type = impl_witness_access %CppRangeForIterate.lookup_impl_witness.edd1fe.2, element1 [symbolic]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.Begin.type.57e: type = fn_type @CppRangeForIterate.WithSelf.Begin, @CppRangeForIterate.WithSelf(%CppRangeForIterate.facet.0d2) [symbolic]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.End.type.04d: type = fn_type @CppRangeForIterate.WithSelf.End, @CppRangeForIterate.WithSelf(%CppRangeForIterate.facet.0d2) [symbolic]
// CHECK:STDOUT:   %.b85: type = fn_type_with_self_type %CppRangeForIterate.WithSelf.Begin.type.57e, %CppRangeForIterate.facet.0d2 [symbolic]
// CHECK:STDOUT:   %impl.elem2: %.b85 = impl_witness_access %CppRangeForIterate.lookup_impl_witness.edd1fe.2, element2 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.c0f: <specific function> = specific_impl_function %impl.elem2, @CppRangeForIterate.WithSelf.Begin(%CppRangeForIterate.facet.0d2) [symbolic]
// CHECK:STDOUT:   %assoc3: %CppRangeForIterate.assoc_type = assoc_entity element3, imports.%Core.import_ref.730 [concrete]
// CHECK:STDOUT:   %.b40: type = fn_type_with_self_type %CppRangeForIterate.WithSelf.End.type.04d, %CppRangeForIterate.facet.0d2 [symbolic]
// CHECK:STDOUT:   %impl.elem3: %.b40 = impl_witness_access %CppRangeForIterate.lookup_impl_witness.edd1fe.2, element3 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.d6a: <specific function> = specific_impl_function %impl.elem3, @CppRangeForIterate.WithSelf.End(%CppRangeForIterate.facet.0d2) [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.a8f: <witness> = lookup_impl_witness %impl.elem1.bdcd2d.2, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.543: %Destroy.type = facet_value %impl.elem1.bdcd2d.2, (%Destroy.lookup_impl_witness.a8f) [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.366: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.543) [symbolic]
// CHECK:STDOUT:   %.b83: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.366, %Destroy.facet.543 [symbolic]
// CHECK:STDOUT:   %impl.elem0.acf: %.b83 = impl_witness_access %Destroy.lookup_impl_witness.a8f, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.add: <specific function> = specific_impl_function %impl.elem0.acf, @Destroy.WithSelf.Op(%Destroy.facet.543) [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.089: <witness> = lookup_impl_witness %impl.elem0.d5966e.2, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.5e8: %Destroy.type = facet_value %impl.elem0.d5966e.2, (%Destroy.lookup_impl_witness.089) [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.7cb: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.5e8) [symbolic]
// CHECK:STDOUT:   %.a68: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.7cb, %Destroy.facet.5e8 [symbolic]
// CHECK:STDOUT:   %impl.elem0.f59: %.a68 = impl_witness_access %Destroy.lookup_impl_witness.089, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.f06: <specific function> = specific_impl_function %impl.elem0.f59, @Destroy.WithSelf.Op(%Destroy.facet.5e8) [symbolic]
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
// CHECK:STDOUT:   %custom_witness.8be: <witness> = custom_witness (%Iterator.b8c, %Sentinel.501, %Begin.1544bb.1, %End.be76e1.1), @CppRangeForIterate [concrete]
// CHECK:STDOUT:   %CppRangeForIterate.facet.23e: %CppRangeForIterate.type = facet_value %MutableRange, (%custom_witness.8be) [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.910: type = fn_type @Iterator.Op.1 [concrete]
// CHECK:STDOUT:   %Iterator.Op.cbb: %Iterator.Op.type.910 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.fe6529.1: <witness> = custom_witness (%Iterator.Op.cbb), @Destroy [concrete]
// CHECK:STDOUT:   %Sentinel.Op.type.910: type = fn_type @Sentinel.Op.1 [concrete]
// CHECK:STDOUT:   %Sentinel.Op.cbb: %Sentinel.Op.type.910 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.fe6529.2: <witness> = custom_witness (%Sentinel.Op.cbb), @Destroy [concrete]
// CHECK:STDOUT:   %facet_value.2d2: %CppRangeForIterate_where.type = facet_value %MutableRange, (%custom_witness.8be, %custom_witness.fe6529.1, %custom_witness.fe6529.2) [concrete]
// CHECK:STDOUT:   %Iterator.080: type = class_type @Iterator.2 [concrete]
// CHECK:STDOUT:   %Sentinel.a47: type = class_type @Sentinel.2 [concrete]
// CHECK:STDOUT:   %Begin.type.793876.2: type = fn_type @Begin.2 [concrete]
// CHECK:STDOUT:   %Begin.1544bb.2: %Begin.type.793876.2 = struct_value () [concrete]
// CHECK:STDOUT:   %End.type.16398b.2: type = fn_type @End.2 [concrete]
// CHECK:STDOUT:   %End.be76e1.2: %End.type.16398b.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.2ff: <witness> = custom_witness (%Iterator.080, %Sentinel.a47, %Begin.1544bb.2, %End.be76e1.2), @CppRangeForIterate [concrete]
// CHECK:STDOUT:   %CppRangeForIterate.facet.b4e: %CppRangeForIterate.type = facet_value %ConstRange, (%custom_witness.2ff) [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.587: type = fn_type @Iterator.Op.2 [concrete]
// CHECK:STDOUT:   %Iterator.Op.dc6: %Iterator.Op.type.587 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.166783.1: <witness> = custom_witness (%Iterator.Op.dc6), @Destroy [concrete]
// CHECK:STDOUT:   %Sentinel.Op.type.587: type = fn_type @Sentinel.Op.2 [concrete]
// CHECK:STDOUT:   %Sentinel.Op.dc6: %Sentinel.Op.type.587 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.166783.2: <witness> = custom_witness (%Sentinel.Op.dc6), @Destroy [concrete]
// CHECK:STDOUT:   %facet_value.c34: %CppRangeForIterate_where.type = facet_value %ConstRange, (%custom_witness.2ff, %custom_witness.166783.1, %custom_witness.166783.2) [concrete]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.Begin.type.f2b: type = fn_type @CppRangeForIterate.WithSelf.Begin, @CppRangeForIterate.WithSelf(%CppRangeForIterate.facet.23e) [concrete]
// CHECK:STDOUT:   %.94e: type = fn_type_with_self_type %CppRangeForIterate.WithSelf.Begin.type.f2b, %CppRangeForIterate.facet.23e [concrete]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.End.type.a76: type = fn_type @CppRangeForIterate.WithSelf.End, @CppRangeForIterate.WithSelf(%CppRangeForIterate.facet.23e) [concrete]
// CHECK:STDOUT:   %.f95: type = fn_type_with_self_type %CppRangeForIterate.WithSelf.End.type.a76, %CppRangeForIterate.facet.23e [concrete]
// CHECK:STDOUT:   %Destroy.facet.15e: %Destroy.type = facet_value %Sentinel.501, (%custom_witness.fe6529.2) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.447: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.15e) [concrete]
// CHECK:STDOUT:   %.67d: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.447, %Destroy.facet.15e [concrete]
// CHECK:STDOUT:   %Destroy.facet.6ae: %Destroy.type = facet_value %Iterator.b8c, (%custom_witness.fe6529.1) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.018: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.6ae) [concrete]
// CHECK:STDOUT:   %.e53: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.018, %Destroy.facet.6ae [concrete]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.Begin.type.982: type = fn_type @CppRangeForIterate.WithSelf.Begin, @CppRangeForIterate.WithSelf(%CppRangeForIterate.facet.b4e) [concrete]
// CHECK:STDOUT:   %.8d9: type = fn_type_with_self_type %CppRangeForIterate.WithSelf.Begin.type.982, %CppRangeForIterate.facet.b4e [concrete]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.End.type.c86: type = fn_type @CppRangeForIterate.WithSelf.End, @CppRangeForIterate.WithSelf(%CppRangeForIterate.facet.b4e) [concrete]
// CHECK:STDOUT:   %.a33: type = fn_type_with_self_type %CppRangeForIterate.WithSelf.End.type.c86, %CppRangeForIterate.facet.b4e [concrete]
// CHECK:STDOUT:   %Destroy.facet.a7c: %Destroy.type = facet_value %Sentinel.a47, (%custom_witness.166783.2) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.89d: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.a7c) [concrete]
// CHECK:STDOUT:   %.514: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.89d, %Destroy.facet.a7c [concrete]
// CHECK:STDOUT:   %Destroy.facet.488: %Destroy.type = facet_value %Iterator.080, (%custom_witness.166783.1) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.488: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.488) [concrete]
// CHECK:STDOUT:   %.8da: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.488, %Destroy.facet.488 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.735: %CppRangeForIterate.assoc_type = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [concrete = constants.%assoc2]
// CHECK:STDOUT:   %Core.import_ref.572: %CppRangeForIterate.assoc_type = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [concrete = constants.%assoc3]
// CHECK:STDOUT:   %Core.import_ref.213: @CppRangeForIterate.WithSelf.%CppRangeForIterate.WithSelf.Begin.type (%CppRangeForIterate.WithSelf.Begin.type.753895.1) = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @CppRangeForIterate.WithSelf.%CppRangeForIterate.WithSelf.Begin (constants.%CppRangeForIterate.WithSelf.Begin.2b31f7.1)]
// CHECK:STDOUT:   %Core.import_ref.730: @CppRangeForIterate.WithSelf.%CppRangeForIterate.WithSelf.End.type (%CppRangeForIterate.WithSelf.End.type.056be7.1) = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @CppRangeForIterate.WithSelf.%CppRangeForIterate.WithSelf.End (constants.%CppRangeForIterate.WithSelf.End.a331a3.1)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Test(%R.loc42_10.2: %CppRangeForIterate_where.type) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %CppRangeForIterate.lookup_impl_witness: <witness> = lookup_impl_witness %R.loc42_10.1, @CppRangeForIterate [symbolic = %CppRangeForIterate.lookup_impl_witness (constants.%CppRangeForIterate.lookup_impl_witness.edd1fe.2)]
// CHECK:STDOUT:   %CppRangeForIterate.facet: %CppRangeForIterate.type = facet_value %R.as_type.loc42_113.1, (%CppRangeForIterate.lookup_impl_witness) [symbolic = %CppRangeForIterate.facet (constants.%CppRangeForIterate.facet.0d2)]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.Begin.type: type = fn_type @CppRangeForIterate.WithSelf.Begin, @CppRangeForIterate.WithSelf(%CppRangeForIterate.facet) [symbolic = %CppRangeForIterate.WithSelf.Begin.type (constants.%CppRangeForIterate.WithSelf.Begin.type.57e)]
// CHECK:STDOUT:   %.loc44_4: type = fn_type_with_self_type %CppRangeForIterate.WithSelf.Begin.type, %CppRangeForIterate.facet [symbolic = %.loc44_4 (constants.%.b85)]
// CHECK:STDOUT:   %impl.elem2.loc44_4.2: @Test.%.loc44_4 (%.b85) = impl_witness_access %CppRangeForIterate.lookup_impl_witness, element2 [symbolic = %impl.elem2.loc44_4.2 (constants.%impl.elem2)]
// CHECK:STDOUT:   %specific_impl_fn.loc44_4.2: <specific function> = specific_impl_function %impl.elem2.loc44_4.2, @CppRangeForIterate.WithSelf.Begin(%CppRangeForIterate.facet) [symbolic = %specific_impl_fn.loc44_4.2 (constants.%specific_impl_fn.c0f)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.End.type: type = fn_type @CppRangeForIterate.WithSelf.End, @CppRangeForIterate.WithSelf(%CppRangeForIterate.facet) [symbolic = %CppRangeForIterate.WithSelf.End.type (constants.%CppRangeForIterate.WithSelf.End.type.04d)]
// CHECK:STDOUT:   %.loc45_4: type = fn_type_with_self_type %CppRangeForIterate.WithSelf.End.type, %CppRangeForIterate.facet [symbolic = %.loc45_4 (constants.%.b40)]
// CHECK:STDOUT:   %impl.elem3.loc45_4.2: @Test.%.loc45_4 (%.b40) = impl_witness_access %CppRangeForIterate.lookup_impl_witness, element3 [symbolic = %impl.elem3.loc45_4.2 (constants.%impl.elem3)]
// CHECK:STDOUT:   %specific_impl_fn.loc45_4.2: <specific function> = specific_impl_function %impl.elem3.loc45_4.2, @CppRangeForIterate.WithSelf.End(%CppRangeForIterate.facet) [symbolic = %specific_impl_fn.loc45_4.2 (constants.%specific_impl_fn.d6a)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.loc45: <witness> = lookup_impl_witness %impl.elem1.1, @Destroy [symbolic = %Destroy.lookup_impl_witness.loc45 (constants.%Destroy.lookup_impl_witness.a8f)]
// CHECK:STDOUT:   %Destroy.facet.loc45: %Destroy.type = facet_value %impl.elem1.1, (%Destroy.lookup_impl_witness.loc45) [symbolic = %Destroy.facet.loc45 (constants.%Destroy.facet.543)]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.loc45: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.loc45) [symbolic = %Destroy.WithSelf.Op.type.loc45 (constants.%Destroy.WithSelf.Op.type.366)]
// CHECK:STDOUT:   %.loc45_9.3: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.loc45, %Destroy.facet.loc45 [symbolic = %.loc45_9.3 (constants.%.b83)]
// CHECK:STDOUT:   %impl.elem0.loc45_9.2: @Test.%.loc45_9.3 (%.b83) = impl_witness_access %Destroy.lookup_impl_witness.loc45, element0 [symbolic = %impl.elem0.loc45_9.2 (constants.%impl.elem0.acf)]
// CHECK:STDOUT:   %specific_impl_fn.loc45_9.2: <specific function> = specific_impl_function %impl.elem0.loc45_9.2, @Destroy.WithSelf.Op(%Destroy.facet.loc45) [symbolic = %specific_impl_fn.loc45_9.2 (constants.%specific_impl_fn.add)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.loc44: <witness> = lookup_impl_witness %impl.elem0.1, @Destroy [symbolic = %Destroy.lookup_impl_witness.loc44 (constants.%Destroy.lookup_impl_witness.089)]
// CHECK:STDOUT:   %Destroy.facet.loc44: %Destroy.type = facet_value %impl.elem0.1, (%Destroy.lookup_impl_witness.loc44) [symbolic = %Destroy.facet.loc44 (constants.%Destroy.facet.5e8)]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.loc44: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.loc44) [symbolic = %Destroy.WithSelf.Op.type.loc44 (constants.%Destroy.WithSelf.Op.type.7cb)]
// CHECK:STDOUT:   %.loc44_11.3: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.loc44, %Destroy.facet.loc44 [symbolic = %.loc44_11.3 (constants.%.a68)]
// CHECK:STDOUT:   %impl.elem0.loc44_11.2: @Test.%.loc44_11.3 (%.a68) = impl_witness_access %Destroy.lookup_impl_witness.loc44, element0 [symbolic = %impl.elem0.loc44_11.2 (constants.%impl.elem0.f59)]
// CHECK:STDOUT:   %specific_impl_fn.loc44_11.2: <specific function> = specific_impl_function %impl.elem0.loc44_11.2, @Destroy.WithSelf.Op(%Destroy.facet.loc44) [symbolic = %specific_impl_fn.loc44_11.2 (constants.%specific_impl_fn.f06)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%r.param: ref @Test.%R.as_type.loc42_113.1 (%R.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %r.ref.loc44: ref @Test.%R.as_type.loc42_113.1 (%R.as_type) = name_ref r, %r
// CHECK:STDOUT:     %Begin.ref: %CppRangeForIterate.assoc_type = name_ref Begin, imports.%Core.import_ref.735 [concrete = constants.%assoc2]
// CHECK:STDOUT:     %impl.elem2.loc44_4.1: @Test.%.loc44_4 (%.b85) = impl_witness_access constants.%CppRangeForIterate.lookup_impl_witness.edd1fe.2, element2 [symbolic = %impl.elem2.loc44_4.2 (constants.%impl.elem2)]
// CHECK:STDOUT:     %bound_method.loc44_4: <bound method> = bound_method %r.ref.loc44, %impl.elem2.loc44_4.1
// CHECK:STDOUT:     %specific_impl_fn.loc44_4.1: <specific function> = specific_impl_function %impl.elem2.loc44_4.1, @CppRangeForIterate.WithSelf.Begin(constants.%CppRangeForIterate.facet.0d2) [symbolic = %specific_impl_fn.loc44_4.2 (constants.%specific_impl_fn.c0f)]
// CHECK:STDOUT:     %bound_method.loc44_11.1: <bound method> = bound_method %r.ref.loc44, %specific_impl_fn.loc44_4.1
// CHECK:STDOUT:     %.loc44_11.1: ref @Test.%impl.elem0.1 (%impl.elem0.d5966e.2) = temporary_storage
// CHECK:STDOUT:     %CppRangeForIterate.WithSelf.Begin.call: init @Test.%impl.elem0.1 (%impl.elem0.d5966e.2) to %.loc44_11.1 = call %bound_method.loc44_11.1(%r.ref.loc44)
// CHECK:STDOUT:     %.loc44_11.2: ref @Test.%impl.elem0.1 (%impl.elem0.d5966e.2) = temporary %.loc44_11.1, %CppRangeForIterate.WithSelf.Begin.call
// CHECK:STDOUT:     %r.ref.loc45: ref @Test.%R.as_type.loc42_113.1 (%R.as_type) = name_ref r, %r
// CHECK:STDOUT:     %End.ref: %CppRangeForIterate.assoc_type = name_ref End, imports.%Core.import_ref.572 [concrete = constants.%assoc3]
// CHECK:STDOUT:     %impl.elem3.loc45_4.1: @Test.%.loc45_4 (%.b40) = impl_witness_access constants.%CppRangeForIterate.lookup_impl_witness.edd1fe.2, element3 [symbolic = %impl.elem3.loc45_4.2 (constants.%impl.elem3)]
// CHECK:STDOUT:     %bound_method.loc45_4: <bound method> = bound_method %r.ref.loc45, %impl.elem3.loc45_4.1
// CHECK:STDOUT:     %specific_impl_fn.loc45_4.1: <specific function> = specific_impl_function %impl.elem3.loc45_4.1, @CppRangeForIterate.WithSelf.End(constants.%CppRangeForIterate.facet.0d2) [symbolic = %specific_impl_fn.loc45_4.2 (constants.%specific_impl_fn.d6a)]
// CHECK:STDOUT:     %bound_method.loc45_9.1: <bound method> = bound_method %r.ref.loc45, %specific_impl_fn.loc45_4.1
// CHECK:STDOUT:     %.loc45_9.1: ref @Test.%impl.elem1.1 (%impl.elem1.bdcd2d.2) = temporary_storage
// CHECK:STDOUT:     %CppRangeForIterate.WithSelf.End.call: init @Test.%impl.elem1.1 (%impl.elem1.bdcd2d.2) to %.loc45_9.1 = call %bound_method.loc45_9.1(%r.ref.loc45)
// CHECK:STDOUT:     %.loc45_9.2: ref @Test.%impl.elem1.1 (%impl.elem1.bdcd2d.2) = temporary %.loc45_9.1, %CppRangeForIterate.WithSelf.End.call
// CHECK:STDOUT:     %impl.elem0.loc45_9.1: @Test.%.loc45_9.3 (%.b83) = impl_witness_access constants.%Destroy.lookup_impl_witness.a8f, element0 [symbolic = %impl.elem0.loc45_9.2 (constants.%impl.elem0.acf)]
// CHECK:STDOUT:     %bound_method.loc45_9.2: <bound method> = bound_method %.loc45_9.2, %impl.elem0.loc45_9.1
// CHECK:STDOUT:     %specific_impl_fn.loc45_9.1: <specific function> = specific_impl_function %impl.elem0.loc45_9.1, @Destroy.WithSelf.Op(constants.%Destroy.facet.543) [symbolic = %specific_impl_fn.loc45_9.2 (constants.%specific_impl_fn.add)]
// CHECK:STDOUT:     %bound_method.loc45_9.3: <bound method> = bound_method %.loc45_9.2, %specific_impl_fn.loc45_9.1
// CHECK:STDOUT:     %Destroy.WithSelf.Op.call.loc45: init %empty_tuple.type = call %bound_method.loc45_9.3(%.loc45_9.2)
// CHECK:STDOUT:     %impl.elem0.loc44_11.1: @Test.%.loc44_11.3 (%.a68) = impl_witness_access constants.%Destroy.lookup_impl_witness.089, element0 [symbolic = %impl.elem0.loc44_11.2 (constants.%impl.elem0.f59)]
// CHECK:STDOUT:     %bound_method.loc44_11.2: <bound method> = bound_method %.loc44_11.2, %impl.elem0.loc44_11.1
// CHECK:STDOUT:     %specific_impl_fn.loc44_11.1: <specific function> = specific_impl_function %impl.elem0.loc44_11.1, @Destroy.WithSelf.Op(constants.%Destroy.facet.5e8) [symbolic = %specific_impl_fn.loc44_11.2 (constants.%specific_impl_fn.f06)]
// CHECK:STDOUT:     %bound_method.loc44_11.3: <bound method> = bound_method %.loc44_11.2, %specific_impl_fn.loc44_11.1
// CHECK:STDOUT:     %Destroy.WithSelf.Op.call.loc44: init %empty_tuple.type = call %bound_method.loc44_11.3(%.loc44_11.2)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(constants.%R) {
// CHECK:STDOUT:   %R.loc42_10.1 => constants.%R
// CHECK:STDOUT:   %R.as_type.loc42_113.1 => constants.%R.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.26c6ea.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(constants.%facet_value.2d2) {
// CHECK:STDOUT:   %R.loc42_10.1 => constants.%facet_value.2d2
// CHECK:STDOUT:   %R.as_type.loc42_113.1 => constants.%MutableRange
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.80f
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc42 => constants.%complete_type
// CHECK:STDOUT:   %CppRangeForIterate.lookup_impl_witness => constants.%custom_witness.8be
// CHECK:STDOUT:   %CppRangeForIterate.facet => constants.%CppRangeForIterate.facet.23e
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.Begin.type => constants.%CppRangeForIterate.WithSelf.Begin.type.f2b
// CHECK:STDOUT:   %.loc44_4 => constants.%.94e
// CHECK:STDOUT:   %impl.elem2.loc44_4.2 => constants.%Begin.1544bb.1
// CHECK:STDOUT:   %specific_impl_fn.loc44_4.2 => constants.%Begin.1544bb.1
// CHECK:STDOUT:   %impl.elem0.1 => constants.%Iterator.b8c
// CHECK:STDOUT:   %require_complete.1 => constants.%complete_type
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.End.type => constants.%CppRangeForIterate.WithSelf.End.type.a76
// CHECK:STDOUT:   %.loc45_4 => constants.%.f95
// CHECK:STDOUT:   %impl.elem3.loc45_4.2 => constants.%End.be76e1.1
// CHECK:STDOUT:   %specific_impl_fn.loc45_4.2 => constants.%End.be76e1.1
// CHECK:STDOUT:   %impl.elem1.1 => constants.%Sentinel.501
// CHECK:STDOUT:   %require_complete.2 => constants.%complete_type
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.loc45 => constants.%custom_witness.fe6529.2
// CHECK:STDOUT:   %Destroy.facet.loc45 => constants.%Destroy.facet.15e
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.loc45 => constants.%Destroy.WithSelf.Op.type.447
// CHECK:STDOUT:   %.loc45_9.3 => constants.%.67d
// CHECK:STDOUT:   %impl.elem0.loc45_9.2 => constants.%Sentinel.Op.cbb
// CHECK:STDOUT:   %specific_impl_fn.loc45_9.2 => constants.%Sentinel.Op.cbb
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.loc44 => constants.%custom_witness.fe6529.1
// CHECK:STDOUT:   %Destroy.facet.loc44 => constants.%Destroy.facet.6ae
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.loc44 => constants.%Destroy.WithSelf.Op.type.018
// CHECK:STDOUT:   %.loc44_11.3 => constants.%.e53
// CHECK:STDOUT:   %impl.elem0.loc44_11.2 => constants.%Iterator.Op.cbb
// CHECK:STDOUT:   %specific_impl_fn.loc44_11.2 => constants.%Iterator.Op.cbb
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(constants.%facet_value.c34) {
// CHECK:STDOUT:   %R.loc42_10.1 => constants.%facet_value.c34
// CHECK:STDOUT:   %R.as_type.loc42_113.1 => constants.%ConstRange
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.88d
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc42 => constants.%complete_type
// CHECK:STDOUT:   %CppRangeForIterate.lookup_impl_witness => constants.%custom_witness.2ff
// CHECK:STDOUT:   %CppRangeForIterate.facet => constants.%CppRangeForIterate.facet.b4e
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.Begin.type => constants.%CppRangeForIterate.WithSelf.Begin.type.982
// CHECK:STDOUT:   %.loc44_4 => constants.%.8d9
// CHECK:STDOUT:   %impl.elem2.loc44_4.2 => constants.%Begin.1544bb.2
// CHECK:STDOUT:   %specific_impl_fn.loc44_4.2 => constants.%Begin.1544bb.2
// CHECK:STDOUT:   %impl.elem0.1 => constants.%Iterator.080
// CHECK:STDOUT:   %require_complete.1 => constants.%complete_type
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.End.type => constants.%CppRangeForIterate.WithSelf.End.type.c86
// CHECK:STDOUT:   %.loc45_4 => constants.%.a33
// CHECK:STDOUT:   %impl.elem3.loc45_4.2 => constants.%End.be76e1.2
// CHECK:STDOUT:   %specific_impl_fn.loc45_4.2 => constants.%End.be76e1.2
// CHECK:STDOUT:   %impl.elem1.1 => constants.%Sentinel.a47
// CHECK:STDOUT:   %require_complete.2 => constants.%complete_type
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.loc45 => constants.%custom_witness.166783.2
// CHECK:STDOUT:   %Destroy.facet.loc45 => constants.%Destroy.facet.a7c
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.loc45 => constants.%Destroy.WithSelf.Op.type.89d
// CHECK:STDOUT:   %.loc45_9.3 => constants.%.514
// CHECK:STDOUT:   %impl.elem0.loc45_9.2 => constants.%Sentinel.Op.dc6
// CHECK:STDOUT:   %specific_impl_fn.loc45_9.2 => constants.%Sentinel.Op.dc6
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.loc44 => constants.%custom_witness.166783.1
// CHECK:STDOUT:   %Destroy.facet.loc44 => constants.%Destroy.facet.488
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.loc44 => constants.%Destroy.WithSelf.Op.type.488
// CHECK:STDOUT:   %.loc44_11.3 => constants.%.8da
// CHECK:STDOUT:   %impl.elem0.loc44_11.2 => constants.%Iterator.Op.dc6
// CHECK:STDOUT:   %specific_impl_fn.loc44_11.2 => constants.%Iterator.Op.dc6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- adl_overloads.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %CppRangeForIterate.type: type = facet_type <@CppRangeForIterate> [concrete]
// CHECK:STDOUT:   %Self.6c6: %CppRangeForIterate.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.End.type.056be7.1: type = fn_type @CppRangeForIterate.WithSelf.End, @CppRangeForIterate.WithSelf(%Self.6c6) [symbolic]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.End.a331a3.1: %CppRangeForIterate.WithSelf.End.type.056be7.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.Begin.type.753895.1: type = fn_type @CppRangeForIterate.WithSelf.Begin, @CppRangeForIterate.WithSelf(%Self.6c6) [symbolic]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.Begin.2b31f7.1: %CppRangeForIterate.WithSelf.Begin.type.753895.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %.Self.10b: %CppRangeForIterate.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %CppRangeForIterate.assoc_type: type = assoc_entity_type @CppRangeForIterate [concrete]
// CHECK:STDOUT:   %CppRangeForIterate.lookup_impl_witness.5bf: <witness> = lookup_impl_witness %.Self.10b, @CppRangeForIterate [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.d47: type = impl_witness_access %CppRangeForIterate.lookup_impl_witness.5bf, element0 [symbolic_self]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %impl.elem1.846: type = impl_witness_access %CppRangeForIterate.lookup_impl_witness.5bf, element1 [symbolic_self]
// CHECK:STDOUT:   %CppRangeForIterate_where.type: type = facet_type <@CppRangeForIterate where %impl.elem0.d47 impls @Destroy and %impl.elem1.846 impls @Destroy> [concrete]
// CHECK:STDOUT:   %R: %CppRangeForIterate_where.type = symbolic_binding R, 0 [symbolic]
// CHECK:STDOUT:   %R.as_type: type = facet_access_type %R [symbolic]
// CHECK:STDOUT:   %pattern_type.26c6ea.2: type = pattern_type %R.as_type [symbolic]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %assoc2: %CppRangeForIterate.assoc_type = assoc_entity element2, imports.%Core.import_ref.213 [concrete]
// CHECK:STDOUT:   %CppRangeForIterate.lookup_impl_witness.edd1fe.2: <witness> = lookup_impl_witness %R, @CppRangeForIterate [symbolic]
// CHECK:STDOUT:   %CppRangeForIterate.facet.0d2: %CppRangeForIterate.type = facet_value %R.as_type, (%CppRangeForIterate.lookup_impl_witness.edd1fe.2) [symbolic]
// CHECK:STDOUT:   %impl.elem0.d5966e.2: type = impl_witness_access %CppRangeForIterate.lookup_impl_witness.edd1fe.2, element0 [symbolic]
// CHECK:STDOUT:   %impl.elem1.bdcd2d.2: type = impl_witness_access %CppRangeForIterate.lookup_impl_witness.edd1fe.2, element1 [symbolic]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.Begin.type.57e: type = fn_type @CppRangeForIterate.WithSelf.Begin, @CppRangeForIterate.WithSelf(%CppRangeForIterate.facet.0d2) [symbolic]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.End.type.04d: type = fn_type @CppRangeForIterate.WithSelf.End, @CppRangeForIterate.WithSelf(%CppRangeForIterate.facet.0d2) [symbolic]
// CHECK:STDOUT:   %.b85: type = fn_type_with_self_type %CppRangeForIterate.WithSelf.Begin.type.57e, %CppRangeForIterate.facet.0d2 [symbolic]
// CHECK:STDOUT:   %impl.elem2: %.b85 = impl_witness_access %CppRangeForIterate.lookup_impl_witness.edd1fe.2, element2 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.c0f: <specific function> = specific_impl_function %impl.elem2, @CppRangeForIterate.WithSelf.Begin(%CppRangeForIterate.facet.0d2) [symbolic]
// CHECK:STDOUT:   %assoc3: %CppRangeForIterate.assoc_type = assoc_entity element3, imports.%Core.import_ref.730 [concrete]
// CHECK:STDOUT:   %.b40: type = fn_type_with_self_type %CppRangeForIterate.WithSelf.End.type.04d, %CppRangeForIterate.facet.0d2 [symbolic]
// CHECK:STDOUT:   %impl.elem3: %.b40 = impl_witness_access %CppRangeForIterate.lookup_impl_witness.edd1fe.2, element3 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.d6a: <specific function> = specific_impl_function %impl.elem3, @CppRangeForIterate.WithSelf.End(%CppRangeForIterate.facet.0d2) [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.a8f: <witness> = lookup_impl_witness %impl.elem1.bdcd2d.2, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.543: %Destroy.type = facet_value %impl.elem1.bdcd2d.2, (%Destroy.lookup_impl_witness.a8f) [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.366: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.543) [symbolic]
// CHECK:STDOUT:   %.b83: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.366, %Destroy.facet.543 [symbolic]
// CHECK:STDOUT:   %impl.elem0.acf: %.b83 = impl_witness_access %Destroy.lookup_impl_witness.a8f, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.add: <specific function> = specific_impl_function %impl.elem0.acf, @Destroy.WithSelf.Op(%Destroy.facet.543) [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.089: <witness> = lookup_impl_witness %impl.elem0.d5966e.2, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.5e8: %Destroy.type = facet_value %impl.elem0.d5966e.2, (%Destroy.lookup_impl_witness.089) [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.7cb: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.5e8) [symbolic]
// CHECK:STDOUT:   %.a68: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.7cb, %Destroy.facet.5e8 [symbolic]
// CHECK:STDOUT:   %impl.elem0.f59: %.a68 = impl_witness_access %Destroy.lookup_impl_witness.089, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.f06: <specific function> = specific_impl_function %impl.elem0.f59, @Destroy.WithSelf.Op(%Destroy.facet.5e8) [symbolic]
// CHECK:STDOUT:   %MutableAndConstRange: type = class_type @MutableAndConstRange [concrete]
// CHECK:STDOUT:   %pattern_type.e6c: type = pattern_type %MutableAndConstRange [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %Iterator: type = class_type @Iterator.1 [concrete]
// CHECK:STDOUT:   %Begin.type: type = fn_type @Begin [concrete]
// CHECK:STDOUT:   %Begin: %Begin.type = struct_value () [concrete]
// CHECK:STDOUT:   %End.type: type = fn_type @End [concrete]
// CHECK:STDOUT:   %End: %End.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.af9: <witness> = custom_witness (%Iterator, %Iterator, %Begin, %End), @CppRangeForIterate [concrete]
// CHECK:STDOUT:   %CppRangeForIterate.facet.18a: %CppRangeForIterate.type = facet_value %MutableAndConstRange, (%custom_witness.af9) [concrete]
// CHECK:STDOUT:   %Iterator.Op.type: type = fn_type @Iterator.Op [concrete]
// CHECK:STDOUT:   %Iterator.Op: %Iterator.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.481: <witness> = custom_witness (%Iterator.Op), @Destroy [concrete]
// CHECK:STDOUT:   %facet_value: %CppRangeForIterate_where.type = facet_value %MutableAndConstRange, (%custom_witness.af9, %custom_witness.481) [concrete]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.Begin.type.3c7: type = fn_type @CppRangeForIterate.WithSelf.Begin, @CppRangeForIterate.WithSelf(%CppRangeForIterate.facet.18a) [concrete]
// CHECK:STDOUT:   %.1b5: type = fn_type_with_self_type %CppRangeForIterate.WithSelf.Begin.type.3c7, %CppRangeForIterate.facet.18a [concrete]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.End.type.011: type = fn_type @CppRangeForIterate.WithSelf.End, @CppRangeForIterate.WithSelf(%CppRangeForIterate.facet.18a) [concrete]
// CHECK:STDOUT:   %.fea: type = fn_type_with_self_type %CppRangeForIterate.WithSelf.End.type.011, %CppRangeForIterate.facet.18a [concrete]
// CHECK:STDOUT:   %Destroy.facet.5e7: %Destroy.type = facet_value %Iterator, (%custom_witness.481) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.594: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.5e7) [concrete]
// CHECK:STDOUT:   %.659: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.594, %Destroy.facet.5e7 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.735: %CppRangeForIterate.assoc_type = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [concrete = constants.%assoc2]
// CHECK:STDOUT:   %Core.import_ref.572: %CppRangeForIterate.assoc_type = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [concrete = constants.%assoc3]
// CHECK:STDOUT:   %Core.import_ref.213: @CppRangeForIterate.WithSelf.%CppRangeForIterate.WithSelf.Begin.type (%CppRangeForIterate.WithSelf.Begin.type.753895.1) = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @CppRangeForIterate.WithSelf.%CppRangeForIterate.WithSelf.Begin (constants.%CppRangeForIterate.WithSelf.Begin.2b31f7.1)]
// CHECK:STDOUT:   %Core.import_ref.730: @CppRangeForIterate.WithSelf.%CppRangeForIterate.WithSelf.End.type (%CppRangeForIterate.WithSelf.End.type.056be7.1) = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @CppRangeForIterate.WithSelf.%CppRangeForIterate.WithSelf.End (constants.%CppRangeForIterate.WithSelf.End.a331a3.1)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Test(%R.loc32_10.2: %CppRangeForIterate_where.type) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %CppRangeForIterate.lookup_impl_witness: <witness> = lookup_impl_witness %R.loc32_10.1, @CppRangeForIterate [symbolic = %CppRangeForIterate.lookup_impl_witness (constants.%CppRangeForIterate.lookup_impl_witness.edd1fe.2)]
// CHECK:STDOUT:   %CppRangeForIterate.facet: %CppRangeForIterate.type = facet_value %R.as_type.loc32_113.1, (%CppRangeForIterate.lookup_impl_witness) [symbolic = %CppRangeForIterate.facet (constants.%CppRangeForIterate.facet.0d2)]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.Begin.type: type = fn_type @CppRangeForIterate.WithSelf.Begin, @CppRangeForIterate.WithSelf(%CppRangeForIterate.facet) [symbolic = %CppRangeForIterate.WithSelf.Begin.type (constants.%CppRangeForIterate.WithSelf.Begin.type.57e)]
// CHECK:STDOUT:   %.loc34_4: type = fn_type_with_self_type %CppRangeForIterate.WithSelf.Begin.type, %CppRangeForIterate.facet [symbolic = %.loc34_4 (constants.%.b85)]
// CHECK:STDOUT:   %impl.elem2.loc34_4.2: @Test.%.loc34_4 (%.b85) = impl_witness_access %CppRangeForIterate.lookup_impl_witness, element2 [symbolic = %impl.elem2.loc34_4.2 (constants.%impl.elem2)]
// CHECK:STDOUT:   %specific_impl_fn.loc34_4.2: <specific function> = specific_impl_function %impl.elem2.loc34_4.2, @CppRangeForIterate.WithSelf.Begin(%CppRangeForIterate.facet) [symbolic = %specific_impl_fn.loc34_4.2 (constants.%specific_impl_fn.c0f)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.End.type: type = fn_type @CppRangeForIterate.WithSelf.End, @CppRangeForIterate.WithSelf(%CppRangeForIterate.facet) [symbolic = %CppRangeForIterate.WithSelf.End.type (constants.%CppRangeForIterate.WithSelf.End.type.04d)]
// CHECK:STDOUT:   %.loc35_4: type = fn_type_with_self_type %CppRangeForIterate.WithSelf.End.type, %CppRangeForIterate.facet [symbolic = %.loc35_4 (constants.%.b40)]
// CHECK:STDOUT:   %impl.elem3.loc35_4.2: @Test.%.loc35_4 (%.b40) = impl_witness_access %CppRangeForIterate.lookup_impl_witness, element3 [symbolic = %impl.elem3.loc35_4.2 (constants.%impl.elem3)]
// CHECK:STDOUT:   %specific_impl_fn.loc35_4.2: <specific function> = specific_impl_function %impl.elem3.loc35_4.2, @CppRangeForIterate.WithSelf.End(%CppRangeForIterate.facet) [symbolic = %specific_impl_fn.loc35_4.2 (constants.%specific_impl_fn.d6a)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.loc35: <witness> = lookup_impl_witness %impl.elem1.1, @Destroy [symbolic = %Destroy.lookup_impl_witness.loc35 (constants.%Destroy.lookup_impl_witness.a8f)]
// CHECK:STDOUT:   %Destroy.facet.loc35: %Destroy.type = facet_value %impl.elem1.1, (%Destroy.lookup_impl_witness.loc35) [symbolic = %Destroy.facet.loc35 (constants.%Destroy.facet.543)]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.loc35: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.loc35) [symbolic = %Destroy.WithSelf.Op.type.loc35 (constants.%Destroy.WithSelf.Op.type.366)]
// CHECK:STDOUT:   %.loc35_9.3: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.loc35, %Destroy.facet.loc35 [symbolic = %.loc35_9.3 (constants.%.b83)]
// CHECK:STDOUT:   %impl.elem0.loc35_9.2: @Test.%.loc35_9.3 (%.b83) = impl_witness_access %Destroy.lookup_impl_witness.loc35, element0 [symbolic = %impl.elem0.loc35_9.2 (constants.%impl.elem0.acf)]
// CHECK:STDOUT:   %specific_impl_fn.loc35_9.2: <specific function> = specific_impl_function %impl.elem0.loc35_9.2, @Destroy.WithSelf.Op(%Destroy.facet.loc35) [symbolic = %specific_impl_fn.loc35_9.2 (constants.%specific_impl_fn.add)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.loc34: <witness> = lookup_impl_witness %impl.elem0.1, @Destroy [symbolic = %Destroy.lookup_impl_witness.loc34 (constants.%Destroy.lookup_impl_witness.089)]
// CHECK:STDOUT:   %Destroy.facet.loc34: %Destroy.type = facet_value %impl.elem0.1, (%Destroy.lookup_impl_witness.loc34) [symbolic = %Destroy.facet.loc34 (constants.%Destroy.facet.5e8)]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.loc34: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.loc34) [symbolic = %Destroy.WithSelf.Op.type.loc34 (constants.%Destroy.WithSelf.Op.type.7cb)]
// CHECK:STDOUT:   %.loc34_11.3: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.loc34, %Destroy.facet.loc34 [symbolic = %.loc34_11.3 (constants.%.a68)]
// CHECK:STDOUT:   %impl.elem0.loc34_11.2: @Test.%.loc34_11.3 (%.a68) = impl_witness_access %Destroy.lookup_impl_witness.loc34, element0 [symbolic = %impl.elem0.loc34_11.2 (constants.%impl.elem0.f59)]
// CHECK:STDOUT:   %specific_impl_fn.loc34_11.2: <specific function> = specific_impl_function %impl.elem0.loc34_11.2, @Destroy.WithSelf.Op(%Destroy.facet.loc34) [symbolic = %specific_impl_fn.loc34_11.2 (constants.%specific_impl_fn.f06)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%r.param: ref @Test.%R.as_type.loc32_113.1 (%R.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %r.ref.loc34: ref @Test.%R.as_type.loc32_113.1 (%R.as_type) = name_ref r, %r
// CHECK:STDOUT:     %Begin.ref: %CppRangeForIterate.assoc_type = name_ref Begin, imports.%Core.import_ref.735 [concrete = constants.%assoc2]
// CHECK:STDOUT:     %impl.elem2.loc34_4.1: @Test.%.loc34_4 (%.b85) = impl_witness_access constants.%CppRangeForIterate.lookup_impl_witness.edd1fe.2, element2 [symbolic = %impl.elem2.loc34_4.2 (constants.%impl.elem2)]
// CHECK:STDOUT:     %bound_method.loc34_4: <bound method> = bound_method %r.ref.loc34, %impl.elem2.loc34_4.1
// CHECK:STDOUT:     %specific_impl_fn.loc34_4.1: <specific function> = specific_impl_function %impl.elem2.loc34_4.1, @CppRangeForIterate.WithSelf.Begin(constants.%CppRangeForIterate.facet.0d2) [symbolic = %specific_impl_fn.loc34_4.2 (constants.%specific_impl_fn.c0f)]
// CHECK:STDOUT:     %bound_method.loc34_11.1: <bound method> = bound_method %r.ref.loc34, %specific_impl_fn.loc34_4.1
// CHECK:STDOUT:     %.loc34_11.1: ref @Test.%impl.elem0.1 (%impl.elem0.d5966e.2) = temporary_storage
// CHECK:STDOUT:     %CppRangeForIterate.WithSelf.Begin.call: init @Test.%impl.elem0.1 (%impl.elem0.d5966e.2) to %.loc34_11.1 = call %bound_method.loc34_11.1(%r.ref.loc34)
// CHECK:STDOUT:     %.loc34_11.2: ref @Test.%impl.elem0.1 (%impl.elem0.d5966e.2) = temporary %.loc34_11.1, %CppRangeForIterate.WithSelf.Begin.call
// CHECK:STDOUT:     %r.ref.loc35: ref @Test.%R.as_type.loc32_113.1 (%R.as_type) = name_ref r, %r
// CHECK:STDOUT:     %End.ref: %CppRangeForIterate.assoc_type = name_ref End, imports.%Core.import_ref.572 [concrete = constants.%assoc3]
// CHECK:STDOUT:     %impl.elem3.loc35_4.1: @Test.%.loc35_4 (%.b40) = impl_witness_access constants.%CppRangeForIterate.lookup_impl_witness.edd1fe.2, element3 [symbolic = %impl.elem3.loc35_4.2 (constants.%impl.elem3)]
// CHECK:STDOUT:     %bound_method.loc35_4: <bound method> = bound_method %r.ref.loc35, %impl.elem3.loc35_4.1
// CHECK:STDOUT:     %specific_impl_fn.loc35_4.1: <specific function> = specific_impl_function %impl.elem3.loc35_4.1, @CppRangeForIterate.WithSelf.End(constants.%CppRangeForIterate.facet.0d2) [symbolic = %specific_impl_fn.loc35_4.2 (constants.%specific_impl_fn.d6a)]
// CHECK:STDOUT:     %bound_method.loc35_9.1: <bound method> = bound_method %r.ref.loc35, %specific_impl_fn.loc35_4.1
// CHECK:STDOUT:     %.loc35_9.1: ref @Test.%impl.elem1.1 (%impl.elem1.bdcd2d.2) = temporary_storage
// CHECK:STDOUT:     %CppRangeForIterate.WithSelf.End.call: init @Test.%impl.elem1.1 (%impl.elem1.bdcd2d.2) to %.loc35_9.1 = call %bound_method.loc35_9.1(%r.ref.loc35)
// CHECK:STDOUT:     %.loc35_9.2: ref @Test.%impl.elem1.1 (%impl.elem1.bdcd2d.2) = temporary %.loc35_9.1, %CppRangeForIterate.WithSelf.End.call
// CHECK:STDOUT:     %impl.elem0.loc35_9.1: @Test.%.loc35_9.3 (%.b83) = impl_witness_access constants.%Destroy.lookup_impl_witness.a8f, element0 [symbolic = %impl.elem0.loc35_9.2 (constants.%impl.elem0.acf)]
// CHECK:STDOUT:     %bound_method.loc35_9.2: <bound method> = bound_method %.loc35_9.2, %impl.elem0.loc35_9.1
// CHECK:STDOUT:     %specific_impl_fn.loc35_9.1: <specific function> = specific_impl_function %impl.elem0.loc35_9.1, @Destroy.WithSelf.Op(constants.%Destroy.facet.543) [symbolic = %specific_impl_fn.loc35_9.2 (constants.%specific_impl_fn.add)]
// CHECK:STDOUT:     %bound_method.loc35_9.3: <bound method> = bound_method %.loc35_9.2, %specific_impl_fn.loc35_9.1
// CHECK:STDOUT:     %Destroy.WithSelf.Op.call.loc35: init %empty_tuple.type = call %bound_method.loc35_9.3(%.loc35_9.2)
// CHECK:STDOUT:     %impl.elem0.loc34_11.1: @Test.%.loc34_11.3 (%.a68) = impl_witness_access constants.%Destroy.lookup_impl_witness.089, element0 [symbolic = %impl.elem0.loc34_11.2 (constants.%impl.elem0.f59)]
// CHECK:STDOUT:     %bound_method.loc34_11.2: <bound method> = bound_method %.loc34_11.2, %impl.elem0.loc34_11.1
// CHECK:STDOUT:     %specific_impl_fn.loc34_11.1: <specific function> = specific_impl_function %impl.elem0.loc34_11.1, @Destroy.WithSelf.Op(constants.%Destroy.facet.5e8) [symbolic = %specific_impl_fn.loc34_11.2 (constants.%specific_impl_fn.f06)]
// CHECK:STDOUT:     %bound_method.loc34_11.3: <bound method> = bound_method %.loc34_11.2, %specific_impl_fn.loc34_11.1
// CHECK:STDOUT:     %Destroy.WithSelf.Op.call.loc34: init %empty_tuple.type = call %bound_method.loc34_11.3(%.loc34_11.2)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(constants.%R) {
// CHECK:STDOUT:   %R.loc32_10.1 => constants.%R
// CHECK:STDOUT:   %R.as_type.loc32_113.1 => constants.%R.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.26c6ea.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(constants.%facet_value) {
// CHECK:STDOUT:   %R.loc32_10.1 => constants.%facet_value
// CHECK:STDOUT:   %R.as_type.loc32_113.1 => constants.%MutableAndConstRange
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.e6c
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc32 => constants.%complete_type
// CHECK:STDOUT:   %CppRangeForIterate.lookup_impl_witness => constants.%custom_witness.af9
// CHECK:STDOUT:   %CppRangeForIterate.facet => constants.%CppRangeForIterate.facet.18a
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.Begin.type => constants.%CppRangeForIterate.WithSelf.Begin.type.3c7
// CHECK:STDOUT:   %.loc34_4 => constants.%.1b5
// CHECK:STDOUT:   %impl.elem2.loc34_4.2 => constants.%Begin
// CHECK:STDOUT:   %specific_impl_fn.loc34_4.2 => constants.%Begin
// CHECK:STDOUT:   %impl.elem0.1 => constants.%Iterator
// CHECK:STDOUT:   %require_complete.1 => constants.%complete_type
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.End.type => constants.%CppRangeForIterate.WithSelf.End.type.011
// CHECK:STDOUT:   %.loc35_4 => constants.%.fea
// CHECK:STDOUT:   %impl.elem3.loc35_4.2 => constants.%End
// CHECK:STDOUT:   %specific_impl_fn.loc35_4.2 => constants.%End
// CHECK:STDOUT:   %impl.elem1.1 => constants.%Iterator
// CHECK:STDOUT:   %require_complete.2 => constants.%complete_type
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.loc35 => constants.%custom_witness.481
// CHECK:STDOUT:   %Destroy.facet.loc35 => constants.%Destroy.facet.5e7
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.loc35 => constants.%Destroy.WithSelf.Op.type.594
// CHECK:STDOUT:   %.loc35_9.3 => constants.%.659
// CHECK:STDOUT:   %impl.elem0.loc35_9.2 => constants.%Iterator.Op
// CHECK:STDOUT:   %specific_impl_fn.loc35_9.2 => constants.%Iterator.Op
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.loc34 => constants.%custom_witness.481
// CHECK:STDOUT:   %Destroy.facet.loc34 => constants.%Destroy.facet.5e7
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.loc34 => constants.%Destroy.WithSelf.Op.type.594
// CHECK:STDOUT:   %.loc34_11.3 => constants.%.659
// CHECK:STDOUT:   %impl.elem0.loc34_11.2 => constants.%Iterator.Op
// CHECK:STDOUT:   %specific_impl_fn.loc34_11.2 => constants.%Iterator.Op
// CHECK:STDOUT: }
// CHECK:STDOUT:
