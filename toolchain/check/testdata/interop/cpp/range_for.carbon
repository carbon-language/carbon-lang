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
              var begin_method_end_field: Cpp.BeginMethodEndField)
{
  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE+7]]:3: error: cannot convert type `Cpp.NoBeginEnd` into type implementing `Core.CppRangeForIterate` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   Test(no_begin_end);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE-16]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn Test[R:! Core.CppRangeForIterate](r: R) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Test(no_begin_end);

  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE+7]]:3: error: cannot convert type `Cpp.NoBeginMethod` into type implementing `Core.CppRangeForIterate` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   Test(no_begin_method);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE-25]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn Test[R:! Core.CppRangeForIterate](r: R) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Test(no_begin_method);

  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE+7]]:3: error: cannot convert type `Cpp.NoEndMethod` into type implementing `Core.CppRangeForIterate` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   Test(no_end_method);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE-34]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn Test[R:! Core.CppRangeForIterate](r: R) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Test(no_end_method);

  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE+7]]:3: error: cannot convert type `Cpp.NoBeginADL` into type implementing `Core.CppRangeForIterate` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   Test(no_begin_adl);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE-43]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn Test[R:! Core.CppRangeForIterate](r: R) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Test(no_begin_adl);

  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE+7]]:3: error: cannot convert type `Cpp.NoEndADL` into type implementing `Core.CppRangeForIterate` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   Test(no_end_adl);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE-52]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn Test[R:! Core.CppRangeForIterate](r: R) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Test(no_end_adl);

  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE+7]]:3: error: cannot convert type `Cpp.BeginMethodEndADL` into type implementing `Core.CppRangeForIterate` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   Test(begin_method_end_adl);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE-61]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn Test[R:! Core.CppRangeForIterate](r: R) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Test(begin_method_end_adl);

  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE+7]]:3: error: cannot convert type `Cpp.BeginADLEndMethod` into type implementing `Core.CppRangeForIterate` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   Test(begin_adl_end_method);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE-70]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn Test[R:! Core.CppRangeForIterate](r: R) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Test(begin_adl_end_method);

  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE+7]]:3: error: cannot convert type `Cpp.BeginFieldEndMethod` into type implementing `Core.CppRangeForIterate` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   Test(begin_field_end_method);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE-79]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn Test[R:! Core.CppRangeForIterate](r: R) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Test(begin_field_end_method);

  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE+7]]:3: error: cannot convert type `Cpp.BeginMethodEndField` into type implementing `Core.CppRangeForIterate` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   Test(begin_method_end_field);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_never_valid.carbon:[[@LINE-88]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn Test[R:! Core.CppRangeForIterate](r: R) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Test(begin_method_end_field);
}

// CHECK:STDOUT: --- methods_return_same_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %CppRangeForIterate.type: type = facet_type <@CppRangeForIterate> [concrete]
// CHECK:STDOUT:   %Self.981: %CppRangeForIterate.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.End.type.5610e6.1: type = fn_type @CppRangeForIterate.WithSelf.End, @CppRangeForIterate.WithSelf(%Self.981) [symbolic]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.End.65a3e4.1: %CppRangeForIterate.WithSelf.End.type.5610e6.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.Begin.type.91eecf.1: type = fn_type @CppRangeForIterate.WithSelf.Begin, @CppRangeForIterate.WithSelf(%Self.981) [symbolic]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.Begin.07315f.1: %CppRangeForIterate.WithSelf.Begin.type.91eecf.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %.Self.756: %CppRangeForIterate.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %CppRangeForIterate.assoc_type: type = assoc_entity_type @CppRangeForIterate [concrete]
// CHECK:STDOUT:   %CppRangeForIterate.lookup_impl_witness.a19: <witness> = lookup_impl_witness %.Self.756, @CppRangeForIterate [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.5fa: type = impl_witness_access %CppRangeForIterate.lookup_impl_witness.a19, element0 [symbolic_self]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %impl.elem1.b14: type = impl_witness_access %CppRangeForIterate.lookup_impl_witness.a19, element1 [symbolic_self]
// CHECK:STDOUT:   %CppRangeForIterate_where.type: type = facet_type <@CppRangeForIterate where %impl.elem0.5fa impls @Destroy and %impl.elem1.b14 impls @Destroy> [symbolic_self]
// CHECK:STDOUT:   %R: %CppRangeForIterate_where.type = symbolic_binding R, 0 [symbolic]
// CHECK:STDOUT:   %R.as_type: type = facet_access_type %R [symbolic]
// CHECK:STDOUT:   %pattern_type.61273a.2: type = pattern_type %R.as_type [symbolic]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %assoc2: %CppRangeForIterate.assoc_type = assoc_entity element2, imports.%Core.import_ref.a11 [concrete]
// CHECK:STDOUT:   %CppRangeForIterate.lookup_impl_witness.20309f.2: <witness> = lookup_impl_witness %R, @CppRangeForIterate [symbolic]
// CHECK:STDOUT:   %CppRangeForIterate.facet.ee6: %CppRangeForIterate.type = facet_value %R.as_type, (%CppRangeForIterate.lookup_impl_witness.20309f.2) [symbolic]
// CHECK:STDOUT:   %impl.elem0.8e47d6.2: type = impl_witness_access %CppRangeForIterate.lookup_impl_witness.20309f.2, element0 [symbolic]
// CHECK:STDOUT:   %impl.elem1.092b54.2: type = impl_witness_access %CppRangeForIterate.lookup_impl_witness.20309f.2, element1 [symbolic]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.Begin.type.71b: type = fn_type @CppRangeForIterate.WithSelf.Begin, @CppRangeForIterate.WithSelf(%CppRangeForIterate.facet.ee6) [symbolic]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.End.type.c1b: type = fn_type @CppRangeForIterate.WithSelf.End, @CppRangeForIterate.WithSelf(%CppRangeForIterate.facet.ee6) [symbolic]
// CHECK:STDOUT:   %.aa6: type = fn_type_with_self_type %CppRangeForIterate.WithSelf.Begin.type.71b, %CppRangeForIterate.facet.ee6 [symbolic]
// CHECK:STDOUT:   %impl.elem2: %.aa6 = impl_witness_access %CppRangeForIterate.lookup_impl_witness.20309f.2, element2 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.987: <specific function> = specific_impl_function %impl.elem2, @CppRangeForIterate.WithSelf.Begin(%CppRangeForIterate.facet.ee6) [symbolic]
// CHECK:STDOUT:   %assoc3: %CppRangeForIterate.assoc_type = assoc_entity element3, imports.%Core.import_ref.031 [concrete]
// CHECK:STDOUT:   %.5ea: type = fn_type_with_self_type %CppRangeForIterate.WithSelf.End.type.c1b, %CppRangeForIterate.facet.ee6 [symbolic]
// CHECK:STDOUT:   %impl.elem3: %.5ea = impl_witness_access %CppRangeForIterate.lookup_impl_witness.20309f.2, element3 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.748: <specific function> = specific_impl_function %impl.elem3, @CppRangeForIterate.WithSelf.End(%CppRangeForIterate.facet.ee6) [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.aba: <witness> = lookup_impl_witness %impl.elem1.092b54.2, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.ba7: %Destroy.type = facet_value %impl.elem1.092b54.2, (%Destroy.lookup_impl_witness.aba) [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.030: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.ba7) [symbolic]
// CHECK:STDOUT:   %.132: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.030, %Destroy.facet.ba7 [symbolic]
// CHECK:STDOUT:   %impl.elem0.74a: %.132 = impl_witness_access %Destroy.lookup_impl_witness.aba, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.044: <specific function> = specific_impl_function %impl.elem0.74a, @Destroy.WithSelf.Op(%Destroy.facet.ba7) [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.cf0: <witness> = lookup_impl_witness %impl.elem0.8e47d6.2, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.0a9: %Destroy.type = facet_value %impl.elem0.8e47d6.2, (%Destroy.lookup_impl_witness.cf0) [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.b91: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.0a9) [symbolic]
// CHECK:STDOUT:   %.d6b: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.b91, %Destroy.facet.0a9 [symbolic]
// CHECK:STDOUT:   %impl.elem0.a98: %.d6b = impl_witness_access %Destroy.lookup_impl_witness.cf0, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.168: <specific function> = specific_impl_function %impl.elem0.a98, @Destroy.WithSelf.Op(%Destroy.facet.0a9) [symbolic]
// CHECK:STDOUT:   %MutableRange: type = class_type @MutableRange [concrete]
// CHECK:STDOUT:   %pattern_type.457: type = pattern_type %MutableRange [concrete]
// CHECK:STDOUT:   %ConstRange: type = class_type @ConstRange [concrete]
// CHECK:STDOUT:   %pattern_type.07c: type = pattern_type %ConstRange [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %Iterator.612: type = class_type @Iterator.1 [concrete]
// CHECK:STDOUT:   %MutableRange.Begin.type: type = fn_type @MutableRange.Begin [concrete]
// CHECK:STDOUT:   %MutableRange.Begin: %MutableRange.Begin.type = struct_value () [concrete]
// CHECK:STDOUT:   %MutableRange.End.type: type = fn_type @MutableRange.End [concrete]
// CHECK:STDOUT:   %MutableRange.End: %MutableRange.End.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.5ab: <witness> = custom_witness (%Iterator.612, %Iterator.612, %MutableRange.Begin, %MutableRange.End), @CppRangeForIterate [concrete]
// CHECK:STDOUT:   %CppRangeForIterate.facet.085: %CppRangeForIterate.type = facet_value %MutableRange, (%custom_witness.5ab) [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.c22: type = fn_type @Iterator.Op.1 [concrete]
// CHECK:STDOUT:   %Iterator.Op.d55: %Iterator.Op.type.c22 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.9ce: <witness> = custom_witness (%Iterator.Op.d55), @Destroy [concrete]
// CHECK:STDOUT:   %facet_value.898: %CppRangeForIterate_where.type = facet_value %MutableRange, (%custom_witness.5ab, %custom_witness.9ce) [symbolic_self]
// CHECK:STDOUT:   %Iterator.cc3: type = class_type @Iterator.2 [concrete]
// CHECK:STDOUT:   %ConstRange.Begin.type: type = fn_type @ConstRange.Begin [concrete]
// CHECK:STDOUT:   %ConstRange.Begin: %ConstRange.Begin.type = struct_value () [concrete]
// CHECK:STDOUT:   %ConstRange.End.type: type = fn_type @ConstRange.End [concrete]
// CHECK:STDOUT:   %ConstRange.End: %ConstRange.End.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.dc1: <witness> = custom_witness (%Iterator.cc3, %Iterator.cc3, %ConstRange.Begin, %ConstRange.End), @CppRangeForIterate [concrete]
// CHECK:STDOUT:   %CppRangeForIterate.facet.eb3: %CppRangeForIterate.type = facet_value %ConstRange, (%custom_witness.dc1) [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.b67: type = fn_type @Iterator.Op.2 [concrete]
// CHECK:STDOUT:   %Iterator.Op.26d: %Iterator.Op.type.b67 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.a80: <witness> = custom_witness (%Iterator.Op.26d), @Destroy [concrete]
// CHECK:STDOUT:   %facet_value.78b: %CppRangeForIterate_where.type = facet_value %ConstRange, (%custom_witness.dc1, %custom_witness.a80) [symbolic_self]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.Begin.type.2da: type = fn_type @CppRangeForIterate.WithSelf.Begin, @CppRangeForIterate.WithSelf(%CppRangeForIterate.facet.085) [concrete]
// CHECK:STDOUT:   %.420: type = fn_type_with_self_type %CppRangeForIterate.WithSelf.Begin.type.2da, %CppRangeForIterate.facet.085 [concrete]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.End.type.900: type = fn_type @CppRangeForIterate.WithSelf.End, @CppRangeForIterate.WithSelf(%CppRangeForIterate.facet.085) [concrete]
// CHECK:STDOUT:   %.ce6: type = fn_type_with_self_type %CppRangeForIterate.WithSelf.End.type.900, %CppRangeForIterate.facet.085 [concrete]
// CHECK:STDOUT:   %Destroy.facet.fec: %Destroy.type = facet_value %Iterator.612, (%custom_witness.9ce) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.c56: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.fec) [concrete]
// CHECK:STDOUT:   %.235: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.c56, %Destroy.facet.fec [concrete]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.Begin.type.6ed: type = fn_type @CppRangeForIterate.WithSelf.Begin, @CppRangeForIterate.WithSelf(%CppRangeForIterate.facet.eb3) [concrete]
// CHECK:STDOUT:   %.d62: type = fn_type_with_self_type %CppRangeForIterate.WithSelf.Begin.type.6ed, %CppRangeForIterate.facet.eb3 [concrete]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.End.type.a9d: type = fn_type @CppRangeForIterate.WithSelf.End, @CppRangeForIterate.WithSelf(%CppRangeForIterate.facet.eb3) [concrete]
// CHECK:STDOUT:   %.ddd: type = fn_type_with_self_type %CppRangeForIterate.WithSelf.End.type.a9d, %CppRangeForIterate.facet.eb3 [concrete]
// CHECK:STDOUT:   %Destroy.facet.d38: %Destroy.type = facet_value %Iterator.cc3, (%custom_witness.a80) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.625: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.d38) [concrete]
// CHECK:STDOUT:   %.448: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.625, %Destroy.facet.d38 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.822: %CppRangeForIterate.assoc_type = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [concrete = constants.%assoc2]
// CHECK:STDOUT:   %Core.import_ref.685: %CppRangeForIterate.assoc_type = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [concrete = constants.%assoc3]
// CHECK:STDOUT:   %Core.import_ref.a11: @CppRangeForIterate.WithSelf.%CppRangeForIterate.WithSelf.Begin.type (%CppRangeForIterate.WithSelf.Begin.type.91eecf.1) = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @CppRangeForIterate.WithSelf.%CppRangeForIterate.WithSelf.Begin (constants.%CppRangeForIterate.WithSelf.Begin.07315f.1)]
// CHECK:STDOUT:   %Core.import_ref.031: @CppRangeForIterate.WithSelf.%CppRangeForIterate.WithSelf.End.type (%CppRangeForIterate.WithSelf.End.type.5610e6.1) = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @CppRangeForIterate.WithSelf.%CppRangeForIterate.WithSelf.End (constants.%CppRangeForIterate.WithSelf.End.65a3e4.1)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Test(%R.loc30_10.2: %CppRangeForIterate_where.type) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %CppRangeForIterate.lookup_impl_witness: <witness> = lookup_impl_witness %R.loc30_10.1, @CppRangeForIterate [symbolic = %CppRangeForIterate.lookup_impl_witness (constants.%CppRangeForIterate.lookup_impl_witness.20309f.2)]
// CHECK:STDOUT:   %CppRangeForIterate.facet: %CppRangeForIterate.type = facet_value %R.as_type.loc30_113.1, (%CppRangeForIterate.lookup_impl_witness) [symbolic = %CppRangeForIterate.facet (constants.%CppRangeForIterate.facet.ee6)]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.Begin.type: type = fn_type @CppRangeForIterate.WithSelf.Begin, @CppRangeForIterate.WithSelf(%CppRangeForIterate.facet) [symbolic = %CppRangeForIterate.WithSelf.Begin.type (constants.%CppRangeForIterate.WithSelf.Begin.type.71b)]
// CHECK:STDOUT:   %.loc32_4: type = fn_type_with_self_type %CppRangeForIterate.WithSelf.Begin.type, %CppRangeForIterate.facet [symbolic = %.loc32_4 (constants.%.aa6)]
// CHECK:STDOUT:   %impl.elem2.loc32_4.2: @Test.%.loc32_4 (%.aa6) = impl_witness_access %CppRangeForIterate.lookup_impl_witness, element2 [symbolic = %impl.elem2.loc32_4.2 (constants.%impl.elem2)]
// CHECK:STDOUT:   %specific_impl_fn.loc32_4.2: <specific function> = specific_impl_function %impl.elem2.loc32_4.2, @CppRangeForIterate.WithSelf.Begin(%CppRangeForIterate.facet) [symbolic = %specific_impl_fn.loc32_4.2 (constants.%specific_impl_fn.987)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.End.type: type = fn_type @CppRangeForIterate.WithSelf.End, @CppRangeForIterate.WithSelf(%CppRangeForIterate.facet) [symbolic = %CppRangeForIterate.WithSelf.End.type (constants.%CppRangeForIterate.WithSelf.End.type.c1b)]
// CHECK:STDOUT:   %.loc33_4: type = fn_type_with_self_type %CppRangeForIterate.WithSelf.End.type, %CppRangeForIterate.facet [symbolic = %.loc33_4 (constants.%.5ea)]
// CHECK:STDOUT:   %impl.elem3.loc33_4.2: @Test.%.loc33_4 (%.5ea) = impl_witness_access %CppRangeForIterate.lookup_impl_witness, element3 [symbolic = %impl.elem3.loc33_4.2 (constants.%impl.elem3)]
// CHECK:STDOUT:   %specific_impl_fn.loc33_4.2: <specific function> = specific_impl_function %impl.elem3.loc33_4.2, @CppRangeForIterate.WithSelf.End(%CppRangeForIterate.facet) [symbolic = %specific_impl_fn.loc33_4.2 (constants.%specific_impl_fn.748)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.loc33: <witness> = lookup_impl_witness %impl.elem1.1, @Destroy [symbolic = %Destroy.lookup_impl_witness.loc33 (constants.%Destroy.lookup_impl_witness.aba)]
// CHECK:STDOUT:   %Destroy.facet.loc33: %Destroy.type = facet_value %impl.elem1.1, (%Destroy.lookup_impl_witness.loc33) [symbolic = %Destroy.facet.loc33 (constants.%Destroy.facet.ba7)]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.loc33: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.loc33) [symbolic = %Destroy.WithSelf.Op.type.loc33 (constants.%Destroy.WithSelf.Op.type.030)]
// CHECK:STDOUT:   %.loc33_9.3: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.loc33, %Destroy.facet.loc33 [symbolic = %.loc33_9.3 (constants.%.132)]
// CHECK:STDOUT:   %impl.elem0.loc33_9.2: @Test.%.loc33_9.3 (%.132) = impl_witness_access %Destroy.lookup_impl_witness.loc33, element0 [symbolic = %impl.elem0.loc33_9.2 (constants.%impl.elem0.74a)]
// CHECK:STDOUT:   %specific_impl_fn.loc33_9.2: <specific function> = specific_impl_function %impl.elem0.loc33_9.2, @Destroy.WithSelf.Op(%Destroy.facet.loc33) [symbolic = %specific_impl_fn.loc33_9.2 (constants.%specific_impl_fn.044)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.loc32: <witness> = lookup_impl_witness %impl.elem0.1, @Destroy [symbolic = %Destroy.lookup_impl_witness.loc32 (constants.%Destroy.lookup_impl_witness.cf0)]
// CHECK:STDOUT:   %Destroy.facet.loc32: %Destroy.type = facet_value %impl.elem0.1, (%Destroy.lookup_impl_witness.loc32) [symbolic = %Destroy.facet.loc32 (constants.%Destroy.facet.0a9)]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.loc32: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.loc32) [symbolic = %Destroy.WithSelf.Op.type.loc32 (constants.%Destroy.WithSelf.Op.type.b91)]
// CHECK:STDOUT:   %.loc32_11.3: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.loc32, %Destroy.facet.loc32 [symbolic = %.loc32_11.3 (constants.%.d6b)]
// CHECK:STDOUT:   %impl.elem0.loc32_11.2: @Test.%.loc32_11.3 (%.d6b) = impl_witness_access %Destroy.lookup_impl_witness.loc32, element0 [symbolic = %impl.elem0.loc32_11.2 (constants.%impl.elem0.a98)]
// CHECK:STDOUT:   %specific_impl_fn.loc32_11.2: <specific function> = specific_impl_function %impl.elem0.loc32_11.2, @Destroy.WithSelf.Op(%Destroy.facet.loc32) [symbolic = %specific_impl_fn.loc32_11.2 (constants.%specific_impl_fn.168)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%r.param: ref @Test.%R.as_type.loc30_113.1 (%R.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %r.ref.loc32: ref @Test.%R.as_type.loc30_113.1 (%R.as_type) = name_ref r, %r
// CHECK:STDOUT:     %Begin.ref: %CppRangeForIterate.assoc_type = name_ref Begin, imports.%Core.import_ref.822 [concrete = constants.%assoc2]
// CHECK:STDOUT:     %impl.elem2.loc32_4.1: @Test.%.loc32_4 (%.aa6) = impl_witness_access constants.%CppRangeForIterate.lookup_impl_witness.20309f.2, element2 [symbolic = %impl.elem2.loc32_4.2 (constants.%impl.elem2)]
// CHECK:STDOUT:     %bound_method.loc32_4: <bound method> = bound_method %r.ref.loc32, %impl.elem2.loc32_4.1
// CHECK:STDOUT:     %specific_impl_fn.loc32_4.1: <specific function> = specific_impl_function %impl.elem2.loc32_4.1, @CppRangeForIterate.WithSelf.Begin(constants.%CppRangeForIterate.facet.ee6) [symbolic = %specific_impl_fn.loc32_4.2 (constants.%specific_impl_fn.987)]
// CHECK:STDOUT:     %bound_method.loc32_11.1: <bound method> = bound_method %r.ref.loc32, %specific_impl_fn.loc32_4.1
// CHECK:STDOUT:     %.loc32_11.1: ref @Test.%impl.elem0.1 (%impl.elem0.8e47d6.2) = temporary_storage
// CHECK:STDOUT:     %CppRangeForIterate.WithSelf.Begin.call: init @Test.%impl.elem0.1 (%impl.elem0.8e47d6.2) to %.loc32_11.1 = call %bound_method.loc32_11.1(%r.ref.loc32)
// CHECK:STDOUT:     %.loc32_11.2: ref @Test.%impl.elem0.1 (%impl.elem0.8e47d6.2) = temporary %.loc32_11.1, %CppRangeForIterate.WithSelf.Begin.call
// CHECK:STDOUT:     %r.ref.loc33: ref @Test.%R.as_type.loc30_113.1 (%R.as_type) = name_ref r, %r
// CHECK:STDOUT:     %End.ref: %CppRangeForIterate.assoc_type = name_ref End, imports.%Core.import_ref.685 [concrete = constants.%assoc3]
// CHECK:STDOUT:     %impl.elem3.loc33_4.1: @Test.%.loc33_4 (%.5ea) = impl_witness_access constants.%CppRangeForIterate.lookup_impl_witness.20309f.2, element3 [symbolic = %impl.elem3.loc33_4.2 (constants.%impl.elem3)]
// CHECK:STDOUT:     %bound_method.loc33_4: <bound method> = bound_method %r.ref.loc33, %impl.elem3.loc33_4.1
// CHECK:STDOUT:     %specific_impl_fn.loc33_4.1: <specific function> = specific_impl_function %impl.elem3.loc33_4.1, @CppRangeForIterate.WithSelf.End(constants.%CppRangeForIterate.facet.ee6) [symbolic = %specific_impl_fn.loc33_4.2 (constants.%specific_impl_fn.748)]
// CHECK:STDOUT:     %bound_method.loc33_9.1: <bound method> = bound_method %r.ref.loc33, %specific_impl_fn.loc33_4.1
// CHECK:STDOUT:     %.loc33_9.1: ref @Test.%impl.elem1.1 (%impl.elem1.092b54.2) = temporary_storage
// CHECK:STDOUT:     %CppRangeForIterate.WithSelf.End.call: init @Test.%impl.elem1.1 (%impl.elem1.092b54.2) to %.loc33_9.1 = call %bound_method.loc33_9.1(%r.ref.loc33)
// CHECK:STDOUT:     %.loc33_9.2: ref @Test.%impl.elem1.1 (%impl.elem1.092b54.2) = temporary %.loc33_9.1, %CppRangeForIterate.WithSelf.End.call
// CHECK:STDOUT:     %impl.elem0.loc33_9.1: @Test.%.loc33_9.3 (%.132) = impl_witness_access constants.%Destroy.lookup_impl_witness.aba, element0 [symbolic = %impl.elem0.loc33_9.2 (constants.%impl.elem0.74a)]
// CHECK:STDOUT:     %bound_method.loc33_9.2: <bound method> = bound_method %.loc33_9.2, %impl.elem0.loc33_9.1
// CHECK:STDOUT:     %specific_impl_fn.loc33_9.1: <specific function> = specific_impl_function %impl.elem0.loc33_9.1, @Destroy.WithSelf.Op(constants.%Destroy.facet.ba7) [symbolic = %specific_impl_fn.loc33_9.2 (constants.%specific_impl_fn.044)]
// CHECK:STDOUT:     %bound_method.loc33_9.3: <bound method> = bound_method %.loc33_9.2, %specific_impl_fn.loc33_9.1
// CHECK:STDOUT:     %Destroy.WithSelf.Op.call.loc33: init %empty_tuple.type = call %bound_method.loc33_9.3(%.loc33_9.2)
// CHECK:STDOUT:     %impl.elem0.loc32_11.1: @Test.%.loc32_11.3 (%.d6b) = impl_witness_access constants.%Destroy.lookup_impl_witness.cf0, element0 [symbolic = %impl.elem0.loc32_11.2 (constants.%impl.elem0.a98)]
// CHECK:STDOUT:     %bound_method.loc32_11.2: <bound method> = bound_method %.loc32_11.2, %impl.elem0.loc32_11.1
// CHECK:STDOUT:     %specific_impl_fn.loc32_11.1: <specific function> = specific_impl_function %impl.elem0.loc32_11.1, @Destroy.WithSelf.Op(constants.%Destroy.facet.0a9) [symbolic = %specific_impl_fn.loc32_11.2 (constants.%specific_impl_fn.168)]
// CHECK:STDOUT:     %bound_method.loc32_11.3: <bound method> = bound_method %.loc32_11.2, %specific_impl_fn.loc32_11.1
// CHECK:STDOUT:     %Destroy.WithSelf.Op.call.loc32: init %empty_tuple.type = call %bound_method.loc32_11.3(%.loc32_11.2)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(constants.%R) {
// CHECK:STDOUT:   %R.loc30_10.1 => constants.%R
// CHECK:STDOUT:   %R.as_type.loc30_113.1 => constants.%R.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.61273a.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(constants.%facet_value.898) {
// CHECK:STDOUT:   %R.loc30_10.1 => constants.%facet_value.898
// CHECK:STDOUT:   %R.as_type.loc30_113.1 => constants.%MutableRange
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.457
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc30 => constants.%complete_type
// CHECK:STDOUT:   %CppRangeForIterate.lookup_impl_witness => constants.%custom_witness.5ab
// CHECK:STDOUT:   %CppRangeForIterate.facet => constants.%CppRangeForIterate.facet.085
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.Begin.type => constants.%CppRangeForIterate.WithSelf.Begin.type.2da
// CHECK:STDOUT:   %.loc32_4 => constants.%.420
// CHECK:STDOUT:   %impl.elem2.loc32_4.2 => constants.%MutableRange.Begin
// CHECK:STDOUT:   %specific_impl_fn.loc32_4.2 => constants.%MutableRange.Begin
// CHECK:STDOUT:   %impl.elem0.1 => constants.%Iterator.612
// CHECK:STDOUT:   %require_complete.1 => constants.%complete_type
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.End.type => constants.%CppRangeForIterate.WithSelf.End.type.900
// CHECK:STDOUT:   %.loc33_4 => constants.%.ce6
// CHECK:STDOUT:   %impl.elem3.loc33_4.2 => constants.%MutableRange.End
// CHECK:STDOUT:   %specific_impl_fn.loc33_4.2 => constants.%MutableRange.End
// CHECK:STDOUT:   %impl.elem1.1 => constants.%Iterator.612
// CHECK:STDOUT:   %require_complete.2 => constants.%complete_type
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.loc33 => constants.%custom_witness.9ce
// CHECK:STDOUT:   %Destroy.facet.loc33 => constants.%Destroy.facet.fec
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.loc33 => constants.%Destroy.WithSelf.Op.type.c56
// CHECK:STDOUT:   %.loc33_9.3 => constants.%.235
// CHECK:STDOUT:   %impl.elem0.loc33_9.2 => constants.%Iterator.Op.d55
// CHECK:STDOUT:   %specific_impl_fn.loc33_9.2 => constants.%Iterator.Op.d55
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.loc32 => constants.%custom_witness.9ce
// CHECK:STDOUT:   %Destroy.facet.loc32 => constants.%Destroy.facet.fec
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.loc32 => constants.%Destroy.WithSelf.Op.type.c56
// CHECK:STDOUT:   %.loc32_11.3 => constants.%.235
// CHECK:STDOUT:   %impl.elem0.loc32_11.2 => constants.%Iterator.Op.d55
// CHECK:STDOUT:   %specific_impl_fn.loc32_11.2 => constants.%Iterator.Op.d55
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(constants.%facet_value.78b) {
// CHECK:STDOUT:   %R.loc30_10.1 => constants.%facet_value.78b
// CHECK:STDOUT:   %R.as_type.loc30_113.1 => constants.%ConstRange
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.07c
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc30 => constants.%complete_type
// CHECK:STDOUT:   %CppRangeForIterate.lookup_impl_witness => constants.%custom_witness.dc1
// CHECK:STDOUT:   %CppRangeForIterate.facet => constants.%CppRangeForIterate.facet.eb3
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.Begin.type => constants.%CppRangeForIterate.WithSelf.Begin.type.6ed
// CHECK:STDOUT:   %.loc32_4 => constants.%.d62
// CHECK:STDOUT:   %impl.elem2.loc32_4.2 => constants.%ConstRange.Begin
// CHECK:STDOUT:   %specific_impl_fn.loc32_4.2 => constants.%ConstRange.Begin
// CHECK:STDOUT:   %impl.elem0.1 => constants.%Iterator.cc3
// CHECK:STDOUT:   %require_complete.1 => constants.%complete_type
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.End.type => constants.%CppRangeForIterate.WithSelf.End.type.a9d
// CHECK:STDOUT:   %.loc33_4 => constants.%.ddd
// CHECK:STDOUT:   %impl.elem3.loc33_4.2 => constants.%ConstRange.End
// CHECK:STDOUT:   %specific_impl_fn.loc33_4.2 => constants.%ConstRange.End
// CHECK:STDOUT:   %impl.elem1.1 => constants.%Iterator.cc3
// CHECK:STDOUT:   %require_complete.2 => constants.%complete_type
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.loc33 => constants.%custom_witness.a80
// CHECK:STDOUT:   %Destroy.facet.loc33 => constants.%Destroy.facet.d38
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.loc33 => constants.%Destroy.WithSelf.Op.type.625
// CHECK:STDOUT:   %.loc33_9.3 => constants.%.448
// CHECK:STDOUT:   %impl.elem0.loc33_9.2 => constants.%Iterator.Op.26d
// CHECK:STDOUT:   %specific_impl_fn.loc33_9.2 => constants.%Iterator.Op.26d
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.loc32 => constants.%custom_witness.a80
// CHECK:STDOUT:   %Destroy.facet.loc32 => constants.%Destroy.facet.d38
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.loc32 => constants.%Destroy.WithSelf.Op.type.625
// CHECK:STDOUT:   %.loc32_11.3 => constants.%.448
// CHECK:STDOUT:   %impl.elem0.loc32_11.2 => constants.%Iterator.Op.26d
// CHECK:STDOUT:   %specific_impl_fn.loc32_11.2 => constants.%Iterator.Op.26d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- methods_return_different_types.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %CppRangeForIterate.type: type = facet_type <@CppRangeForIterate> [concrete]
// CHECK:STDOUT:   %Self.981: %CppRangeForIterate.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.End.type.5610e6.1: type = fn_type @CppRangeForIterate.WithSelf.End, @CppRangeForIterate.WithSelf(%Self.981) [symbolic]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.End.65a3e4.1: %CppRangeForIterate.WithSelf.End.type.5610e6.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.Begin.type.91eecf.1: type = fn_type @CppRangeForIterate.WithSelf.Begin, @CppRangeForIterate.WithSelf(%Self.981) [symbolic]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.Begin.07315f.1: %CppRangeForIterate.WithSelf.Begin.type.91eecf.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %.Self.756: %CppRangeForIterate.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %CppRangeForIterate.assoc_type: type = assoc_entity_type @CppRangeForIterate [concrete]
// CHECK:STDOUT:   %CppRangeForIterate.lookup_impl_witness.a19: <witness> = lookup_impl_witness %.Self.756, @CppRangeForIterate [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.5fa: type = impl_witness_access %CppRangeForIterate.lookup_impl_witness.a19, element0 [symbolic_self]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %impl.elem1.b14: type = impl_witness_access %CppRangeForIterate.lookup_impl_witness.a19, element1 [symbolic_self]
// CHECK:STDOUT:   %CppRangeForIterate_where.type: type = facet_type <@CppRangeForIterate where %impl.elem0.5fa impls @Destroy and %impl.elem1.b14 impls @Destroy> [symbolic_self]
// CHECK:STDOUT:   %R: %CppRangeForIterate_where.type = symbolic_binding R, 0 [symbolic]
// CHECK:STDOUT:   %R.as_type: type = facet_access_type %R [symbolic]
// CHECK:STDOUT:   %pattern_type.61273a.2: type = pattern_type %R.as_type [symbolic]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %assoc2: %CppRangeForIterate.assoc_type = assoc_entity element2, imports.%Core.import_ref.a11 [concrete]
// CHECK:STDOUT:   %CppRangeForIterate.lookup_impl_witness.20309f.2: <witness> = lookup_impl_witness %R, @CppRangeForIterate [symbolic]
// CHECK:STDOUT:   %CppRangeForIterate.facet.ee6: %CppRangeForIterate.type = facet_value %R.as_type, (%CppRangeForIterate.lookup_impl_witness.20309f.2) [symbolic]
// CHECK:STDOUT:   %impl.elem0.8e47d6.2: type = impl_witness_access %CppRangeForIterate.lookup_impl_witness.20309f.2, element0 [symbolic]
// CHECK:STDOUT:   %impl.elem1.092b54.2: type = impl_witness_access %CppRangeForIterate.lookup_impl_witness.20309f.2, element1 [symbolic]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.Begin.type.71b: type = fn_type @CppRangeForIterate.WithSelf.Begin, @CppRangeForIterate.WithSelf(%CppRangeForIterate.facet.ee6) [symbolic]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.End.type.c1b: type = fn_type @CppRangeForIterate.WithSelf.End, @CppRangeForIterate.WithSelf(%CppRangeForIterate.facet.ee6) [symbolic]
// CHECK:STDOUT:   %.aa6: type = fn_type_with_self_type %CppRangeForIterate.WithSelf.Begin.type.71b, %CppRangeForIterate.facet.ee6 [symbolic]
// CHECK:STDOUT:   %impl.elem2: %.aa6 = impl_witness_access %CppRangeForIterate.lookup_impl_witness.20309f.2, element2 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.987: <specific function> = specific_impl_function %impl.elem2, @CppRangeForIterate.WithSelf.Begin(%CppRangeForIterate.facet.ee6) [symbolic]
// CHECK:STDOUT:   %assoc3: %CppRangeForIterate.assoc_type = assoc_entity element3, imports.%Core.import_ref.031 [concrete]
// CHECK:STDOUT:   %.5ea: type = fn_type_with_self_type %CppRangeForIterate.WithSelf.End.type.c1b, %CppRangeForIterate.facet.ee6 [symbolic]
// CHECK:STDOUT:   %impl.elem3: %.5ea = impl_witness_access %CppRangeForIterate.lookup_impl_witness.20309f.2, element3 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.748: <specific function> = specific_impl_function %impl.elem3, @CppRangeForIterate.WithSelf.End(%CppRangeForIterate.facet.ee6) [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.aba: <witness> = lookup_impl_witness %impl.elem1.092b54.2, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.ba7: %Destroy.type = facet_value %impl.elem1.092b54.2, (%Destroy.lookup_impl_witness.aba) [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.030: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.ba7) [symbolic]
// CHECK:STDOUT:   %.132: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.030, %Destroy.facet.ba7 [symbolic]
// CHECK:STDOUT:   %impl.elem0.74a: %.132 = impl_witness_access %Destroy.lookup_impl_witness.aba, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.044: <specific function> = specific_impl_function %impl.elem0.74a, @Destroy.WithSelf.Op(%Destroy.facet.ba7) [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.cf0: <witness> = lookup_impl_witness %impl.elem0.8e47d6.2, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.0a9: %Destroy.type = facet_value %impl.elem0.8e47d6.2, (%Destroy.lookup_impl_witness.cf0) [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.b91: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.0a9) [symbolic]
// CHECK:STDOUT:   %.d6b: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.b91, %Destroy.facet.0a9 [symbolic]
// CHECK:STDOUT:   %impl.elem0.a98: %.d6b = impl_witness_access %Destroy.lookup_impl_witness.cf0, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.168: <specific function> = specific_impl_function %impl.elem0.a98, @Destroy.WithSelf.Op(%Destroy.facet.0a9) [symbolic]
// CHECK:STDOUT:   %MutableRange: type = class_type @MutableRange [concrete]
// CHECK:STDOUT:   %pattern_type.457: type = pattern_type %MutableRange [concrete]
// CHECK:STDOUT:   %ConstRange: type = class_type @ConstRange [concrete]
// CHECK:STDOUT:   %pattern_type.07c: type = pattern_type %ConstRange [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %Iterator.612: type = class_type @Iterator.1 [concrete]
// CHECK:STDOUT:   %Sentinel.818: type = class_type @Sentinel.1 [concrete]
// CHECK:STDOUT:   %MutableRange.Begin.type: type = fn_type @MutableRange.Begin [concrete]
// CHECK:STDOUT:   %MutableRange.Begin: %MutableRange.Begin.type = struct_value () [concrete]
// CHECK:STDOUT:   %MutableRange.End.type: type = fn_type @MutableRange.End [concrete]
// CHECK:STDOUT:   %MutableRange.End: %MutableRange.End.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.063: <witness> = custom_witness (%Iterator.612, %Sentinel.818, %MutableRange.Begin, %MutableRange.End), @CppRangeForIterate [concrete]
// CHECK:STDOUT:   %CppRangeForIterate.facet.f0e: %CppRangeForIterate.type = facet_value %MutableRange, (%custom_witness.063) [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.c22: type = fn_type @Iterator.Op.1 [concrete]
// CHECK:STDOUT:   %Iterator.Op.d55: %Iterator.Op.type.c22 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.9ce: <witness> = custom_witness (%Iterator.Op.d55), @Destroy [concrete]
// CHECK:STDOUT:   %Sentinel.Op.type.a55: type = fn_type @Sentinel.Op.1 [concrete]
// CHECK:STDOUT:   %Sentinel.Op.f79: %Sentinel.Op.type.a55 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.fb5: <witness> = custom_witness (%Sentinel.Op.f79), @Destroy [concrete]
// CHECK:STDOUT:   %facet_value.092: %CppRangeForIterate_where.type = facet_value %MutableRange, (%custom_witness.063, %custom_witness.9ce, %custom_witness.fb5) [symbolic_self]
// CHECK:STDOUT:   %Iterator.cc3: type = class_type @Iterator.2 [concrete]
// CHECK:STDOUT:   %Sentinel.deb: type = class_type @Sentinel.2 [concrete]
// CHECK:STDOUT:   %ConstRange.Begin.type: type = fn_type @ConstRange.Begin [concrete]
// CHECK:STDOUT:   %ConstRange.Begin: %ConstRange.Begin.type = struct_value () [concrete]
// CHECK:STDOUT:   %ConstRange.End.type: type = fn_type @ConstRange.End [concrete]
// CHECK:STDOUT:   %ConstRange.End: %ConstRange.End.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.522: <witness> = custom_witness (%Iterator.cc3, %Sentinel.deb, %ConstRange.Begin, %ConstRange.End), @CppRangeForIterate [concrete]
// CHECK:STDOUT:   %CppRangeForIterate.facet.8ca: %CppRangeForIterate.type = facet_value %ConstRange, (%custom_witness.522) [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.b67: type = fn_type @Iterator.Op.2 [concrete]
// CHECK:STDOUT:   %Iterator.Op.26d: %Iterator.Op.type.b67 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.a80: <witness> = custom_witness (%Iterator.Op.26d), @Destroy [concrete]
// CHECK:STDOUT:   %Sentinel.Op.type.171: type = fn_type @Sentinel.Op.2 [concrete]
// CHECK:STDOUT:   %Sentinel.Op.6c9: %Sentinel.Op.type.171 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.d98: <witness> = custom_witness (%Sentinel.Op.6c9), @Destroy [concrete]
// CHECK:STDOUT:   %facet_value.9ee: %CppRangeForIterate_where.type = facet_value %ConstRange, (%custom_witness.522, %custom_witness.a80, %custom_witness.d98) [symbolic_self]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.Begin.type.e46: type = fn_type @CppRangeForIterate.WithSelf.Begin, @CppRangeForIterate.WithSelf(%CppRangeForIterate.facet.f0e) [concrete]
// CHECK:STDOUT:   %.58f: type = fn_type_with_self_type %CppRangeForIterate.WithSelf.Begin.type.e46, %CppRangeForIterate.facet.f0e [concrete]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.End.type.d1b: type = fn_type @CppRangeForIterate.WithSelf.End, @CppRangeForIterate.WithSelf(%CppRangeForIterate.facet.f0e) [concrete]
// CHECK:STDOUT:   %.8bd: type = fn_type_with_self_type %CppRangeForIterate.WithSelf.End.type.d1b, %CppRangeForIterate.facet.f0e [concrete]
// CHECK:STDOUT:   %Destroy.facet.6ae: %Destroy.type = facet_value %Sentinel.818, (%custom_witness.fb5) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.45c: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.6ae) [concrete]
// CHECK:STDOUT:   %.759: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.45c, %Destroy.facet.6ae [concrete]
// CHECK:STDOUT:   %Destroy.facet.fec: %Destroy.type = facet_value %Iterator.612, (%custom_witness.9ce) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.c56: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.fec) [concrete]
// CHECK:STDOUT:   %.235: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.c56, %Destroy.facet.fec [concrete]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.Begin.type.1f3: type = fn_type @CppRangeForIterate.WithSelf.Begin, @CppRangeForIterate.WithSelf(%CppRangeForIterate.facet.8ca) [concrete]
// CHECK:STDOUT:   %.c0c: type = fn_type_with_self_type %CppRangeForIterate.WithSelf.Begin.type.1f3, %CppRangeForIterate.facet.8ca [concrete]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.End.type.0e5: type = fn_type @CppRangeForIterate.WithSelf.End, @CppRangeForIterate.WithSelf(%CppRangeForIterate.facet.8ca) [concrete]
// CHECK:STDOUT:   %.e09: type = fn_type_with_self_type %CppRangeForIterate.WithSelf.End.type.0e5, %CppRangeForIterate.facet.8ca [concrete]
// CHECK:STDOUT:   %Destroy.facet.787: %Destroy.type = facet_value %Sentinel.deb, (%custom_witness.d98) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.3ee: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.787) [concrete]
// CHECK:STDOUT:   %.452: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.3ee, %Destroy.facet.787 [concrete]
// CHECK:STDOUT:   %Destroy.facet.d38: %Destroy.type = facet_value %Iterator.cc3, (%custom_witness.a80) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.625: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.d38) [concrete]
// CHECK:STDOUT:   %.448: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.625, %Destroy.facet.d38 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.822: %CppRangeForIterate.assoc_type = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [concrete = constants.%assoc2]
// CHECK:STDOUT:   %Core.import_ref.685: %CppRangeForIterate.assoc_type = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [concrete = constants.%assoc3]
// CHECK:STDOUT:   %Core.import_ref.a11: @CppRangeForIterate.WithSelf.%CppRangeForIterate.WithSelf.Begin.type (%CppRangeForIterate.WithSelf.Begin.type.91eecf.1) = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @CppRangeForIterate.WithSelf.%CppRangeForIterate.WithSelf.Begin (constants.%CppRangeForIterate.WithSelf.Begin.07315f.1)]
// CHECK:STDOUT:   %Core.import_ref.031: @CppRangeForIterate.WithSelf.%CppRangeForIterate.WithSelf.End.type (%CppRangeForIterate.WithSelf.End.type.5610e6.1) = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @CppRangeForIterate.WithSelf.%CppRangeForIterate.WithSelf.End (constants.%CppRangeForIterate.WithSelf.End.65a3e4.1)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Test(%R.loc40_10.2: %CppRangeForIterate_where.type) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %CppRangeForIterate.lookup_impl_witness: <witness> = lookup_impl_witness %R.loc40_10.1, @CppRangeForIterate [symbolic = %CppRangeForIterate.lookup_impl_witness (constants.%CppRangeForIterate.lookup_impl_witness.20309f.2)]
// CHECK:STDOUT:   %CppRangeForIterate.facet: %CppRangeForIterate.type = facet_value %R.as_type.loc40_113.1, (%CppRangeForIterate.lookup_impl_witness) [symbolic = %CppRangeForIterate.facet (constants.%CppRangeForIterate.facet.ee6)]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.Begin.type: type = fn_type @CppRangeForIterate.WithSelf.Begin, @CppRangeForIterate.WithSelf(%CppRangeForIterate.facet) [symbolic = %CppRangeForIterate.WithSelf.Begin.type (constants.%CppRangeForIterate.WithSelf.Begin.type.71b)]
// CHECK:STDOUT:   %.loc42_4: type = fn_type_with_self_type %CppRangeForIterate.WithSelf.Begin.type, %CppRangeForIterate.facet [symbolic = %.loc42_4 (constants.%.aa6)]
// CHECK:STDOUT:   %impl.elem2.loc42_4.2: @Test.%.loc42_4 (%.aa6) = impl_witness_access %CppRangeForIterate.lookup_impl_witness, element2 [symbolic = %impl.elem2.loc42_4.2 (constants.%impl.elem2)]
// CHECK:STDOUT:   %specific_impl_fn.loc42_4.2: <specific function> = specific_impl_function %impl.elem2.loc42_4.2, @CppRangeForIterate.WithSelf.Begin(%CppRangeForIterate.facet) [symbolic = %specific_impl_fn.loc42_4.2 (constants.%specific_impl_fn.987)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.End.type: type = fn_type @CppRangeForIterate.WithSelf.End, @CppRangeForIterate.WithSelf(%CppRangeForIterate.facet) [symbolic = %CppRangeForIterate.WithSelf.End.type (constants.%CppRangeForIterate.WithSelf.End.type.c1b)]
// CHECK:STDOUT:   %.loc43_4: type = fn_type_with_self_type %CppRangeForIterate.WithSelf.End.type, %CppRangeForIterate.facet [symbolic = %.loc43_4 (constants.%.5ea)]
// CHECK:STDOUT:   %impl.elem3.loc43_4.2: @Test.%.loc43_4 (%.5ea) = impl_witness_access %CppRangeForIterate.lookup_impl_witness, element3 [symbolic = %impl.elem3.loc43_4.2 (constants.%impl.elem3)]
// CHECK:STDOUT:   %specific_impl_fn.loc43_4.2: <specific function> = specific_impl_function %impl.elem3.loc43_4.2, @CppRangeForIterate.WithSelf.End(%CppRangeForIterate.facet) [symbolic = %specific_impl_fn.loc43_4.2 (constants.%specific_impl_fn.748)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.loc43: <witness> = lookup_impl_witness %impl.elem1.1, @Destroy [symbolic = %Destroy.lookup_impl_witness.loc43 (constants.%Destroy.lookup_impl_witness.aba)]
// CHECK:STDOUT:   %Destroy.facet.loc43: %Destroy.type = facet_value %impl.elem1.1, (%Destroy.lookup_impl_witness.loc43) [symbolic = %Destroy.facet.loc43 (constants.%Destroy.facet.ba7)]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.loc43: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.loc43) [symbolic = %Destroy.WithSelf.Op.type.loc43 (constants.%Destroy.WithSelf.Op.type.030)]
// CHECK:STDOUT:   %.loc43_9.3: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.loc43, %Destroy.facet.loc43 [symbolic = %.loc43_9.3 (constants.%.132)]
// CHECK:STDOUT:   %impl.elem0.loc43_9.2: @Test.%.loc43_9.3 (%.132) = impl_witness_access %Destroy.lookup_impl_witness.loc43, element0 [symbolic = %impl.elem0.loc43_9.2 (constants.%impl.elem0.74a)]
// CHECK:STDOUT:   %specific_impl_fn.loc43_9.2: <specific function> = specific_impl_function %impl.elem0.loc43_9.2, @Destroy.WithSelf.Op(%Destroy.facet.loc43) [symbolic = %specific_impl_fn.loc43_9.2 (constants.%specific_impl_fn.044)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.loc42: <witness> = lookup_impl_witness %impl.elem0.1, @Destroy [symbolic = %Destroy.lookup_impl_witness.loc42 (constants.%Destroy.lookup_impl_witness.cf0)]
// CHECK:STDOUT:   %Destroy.facet.loc42: %Destroy.type = facet_value %impl.elem0.1, (%Destroy.lookup_impl_witness.loc42) [symbolic = %Destroy.facet.loc42 (constants.%Destroy.facet.0a9)]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.loc42: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.loc42) [symbolic = %Destroy.WithSelf.Op.type.loc42 (constants.%Destroy.WithSelf.Op.type.b91)]
// CHECK:STDOUT:   %.loc42_11.3: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.loc42, %Destroy.facet.loc42 [symbolic = %.loc42_11.3 (constants.%.d6b)]
// CHECK:STDOUT:   %impl.elem0.loc42_11.2: @Test.%.loc42_11.3 (%.d6b) = impl_witness_access %Destroy.lookup_impl_witness.loc42, element0 [symbolic = %impl.elem0.loc42_11.2 (constants.%impl.elem0.a98)]
// CHECK:STDOUT:   %specific_impl_fn.loc42_11.2: <specific function> = specific_impl_function %impl.elem0.loc42_11.2, @Destroy.WithSelf.Op(%Destroy.facet.loc42) [symbolic = %specific_impl_fn.loc42_11.2 (constants.%specific_impl_fn.168)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%r.param: ref @Test.%R.as_type.loc40_113.1 (%R.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %r.ref.loc42: ref @Test.%R.as_type.loc40_113.1 (%R.as_type) = name_ref r, %r
// CHECK:STDOUT:     %Begin.ref: %CppRangeForIterate.assoc_type = name_ref Begin, imports.%Core.import_ref.822 [concrete = constants.%assoc2]
// CHECK:STDOUT:     %impl.elem2.loc42_4.1: @Test.%.loc42_4 (%.aa6) = impl_witness_access constants.%CppRangeForIterate.lookup_impl_witness.20309f.2, element2 [symbolic = %impl.elem2.loc42_4.2 (constants.%impl.elem2)]
// CHECK:STDOUT:     %bound_method.loc42_4: <bound method> = bound_method %r.ref.loc42, %impl.elem2.loc42_4.1
// CHECK:STDOUT:     %specific_impl_fn.loc42_4.1: <specific function> = specific_impl_function %impl.elem2.loc42_4.1, @CppRangeForIterate.WithSelf.Begin(constants.%CppRangeForIterate.facet.ee6) [symbolic = %specific_impl_fn.loc42_4.2 (constants.%specific_impl_fn.987)]
// CHECK:STDOUT:     %bound_method.loc42_11.1: <bound method> = bound_method %r.ref.loc42, %specific_impl_fn.loc42_4.1
// CHECK:STDOUT:     %.loc42_11.1: ref @Test.%impl.elem0.1 (%impl.elem0.8e47d6.2) = temporary_storage
// CHECK:STDOUT:     %CppRangeForIterate.WithSelf.Begin.call: init @Test.%impl.elem0.1 (%impl.elem0.8e47d6.2) to %.loc42_11.1 = call %bound_method.loc42_11.1(%r.ref.loc42)
// CHECK:STDOUT:     %.loc42_11.2: ref @Test.%impl.elem0.1 (%impl.elem0.8e47d6.2) = temporary %.loc42_11.1, %CppRangeForIterate.WithSelf.Begin.call
// CHECK:STDOUT:     %r.ref.loc43: ref @Test.%R.as_type.loc40_113.1 (%R.as_type) = name_ref r, %r
// CHECK:STDOUT:     %End.ref: %CppRangeForIterate.assoc_type = name_ref End, imports.%Core.import_ref.685 [concrete = constants.%assoc3]
// CHECK:STDOUT:     %impl.elem3.loc43_4.1: @Test.%.loc43_4 (%.5ea) = impl_witness_access constants.%CppRangeForIterate.lookup_impl_witness.20309f.2, element3 [symbolic = %impl.elem3.loc43_4.2 (constants.%impl.elem3)]
// CHECK:STDOUT:     %bound_method.loc43_4: <bound method> = bound_method %r.ref.loc43, %impl.elem3.loc43_4.1
// CHECK:STDOUT:     %specific_impl_fn.loc43_4.1: <specific function> = specific_impl_function %impl.elem3.loc43_4.1, @CppRangeForIterate.WithSelf.End(constants.%CppRangeForIterate.facet.ee6) [symbolic = %specific_impl_fn.loc43_4.2 (constants.%specific_impl_fn.748)]
// CHECK:STDOUT:     %bound_method.loc43_9.1: <bound method> = bound_method %r.ref.loc43, %specific_impl_fn.loc43_4.1
// CHECK:STDOUT:     %.loc43_9.1: ref @Test.%impl.elem1.1 (%impl.elem1.092b54.2) = temporary_storage
// CHECK:STDOUT:     %CppRangeForIterate.WithSelf.End.call: init @Test.%impl.elem1.1 (%impl.elem1.092b54.2) to %.loc43_9.1 = call %bound_method.loc43_9.1(%r.ref.loc43)
// CHECK:STDOUT:     %.loc43_9.2: ref @Test.%impl.elem1.1 (%impl.elem1.092b54.2) = temporary %.loc43_9.1, %CppRangeForIterate.WithSelf.End.call
// CHECK:STDOUT:     %impl.elem0.loc43_9.1: @Test.%.loc43_9.3 (%.132) = impl_witness_access constants.%Destroy.lookup_impl_witness.aba, element0 [symbolic = %impl.elem0.loc43_9.2 (constants.%impl.elem0.74a)]
// CHECK:STDOUT:     %bound_method.loc43_9.2: <bound method> = bound_method %.loc43_9.2, %impl.elem0.loc43_9.1
// CHECK:STDOUT:     %specific_impl_fn.loc43_9.1: <specific function> = specific_impl_function %impl.elem0.loc43_9.1, @Destroy.WithSelf.Op(constants.%Destroy.facet.ba7) [symbolic = %specific_impl_fn.loc43_9.2 (constants.%specific_impl_fn.044)]
// CHECK:STDOUT:     %bound_method.loc43_9.3: <bound method> = bound_method %.loc43_9.2, %specific_impl_fn.loc43_9.1
// CHECK:STDOUT:     %Destroy.WithSelf.Op.call.loc43: init %empty_tuple.type = call %bound_method.loc43_9.3(%.loc43_9.2)
// CHECK:STDOUT:     %impl.elem0.loc42_11.1: @Test.%.loc42_11.3 (%.d6b) = impl_witness_access constants.%Destroy.lookup_impl_witness.cf0, element0 [symbolic = %impl.elem0.loc42_11.2 (constants.%impl.elem0.a98)]
// CHECK:STDOUT:     %bound_method.loc42_11.2: <bound method> = bound_method %.loc42_11.2, %impl.elem0.loc42_11.1
// CHECK:STDOUT:     %specific_impl_fn.loc42_11.1: <specific function> = specific_impl_function %impl.elem0.loc42_11.1, @Destroy.WithSelf.Op(constants.%Destroy.facet.0a9) [symbolic = %specific_impl_fn.loc42_11.2 (constants.%specific_impl_fn.168)]
// CHECK:STDOUT:     %bound_method.loc42_11.3: <bound method> = bound_method %.loc42_11.2, %specific_impl_fn.loc42_11.1
// CHECK:STDOUT:     %Destroy.WithSelf.Op.call.loc42: init %empty_tuple.type = call %bound_method.loc42_11.3(%.loc42_11.2)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(constants.%R) {
// CHECK:STDOUT:   %R.loc40_10.1 => constants.%R
// CHECK:STDOUT:   %R.as_type.loc40_113.1 => constants.%R.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.61273a.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(constants.%facet_value.092) {
// CHECK:STDOUT:   %R.loc40_10.1 => constants.%facet_value.092
// CHECK:STDOUT:   %R.as_type.loc40_113.1 => constants.%MutableRange
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.457
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc40 => constants.%complete_type
// CHECK:STDOUT:   %CppRangeForIterate.lookup_impl_witness => constants.%custom_witness.063
// CHECK:STDOUT:   %CppRangeForIterate.facet => constants.%CppRangeForIterate.facet.f0e
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.Begin.type => constants.%CppRangeForIterate.WithSelf.Begin.type.e46
// CHECK:STDOUT:   %.loc42_4 => constants.%.58f
// CHECK:STDOUT:   %impl.elem2.loc42_4.2 => constants.%MutableRange.Begin
// CHECK:STDOUT:   %specific_impl_fn.loc42_4.2 => constants.%MutableRange.Begin
// CHECK:STDOUT:   %impl.elem0.1 => constants.%Iterator.612
// CHECK:STDOUT:   %require_complete.1 => constants.%complete_type
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.End.type => constants.%CppRangeForIterate.WithSelf.End.type.d1b
// CHECK:STDOUT:   %.loc43_4 => constants.%.8bd
// CHECK:STDOUT:   %impl.elem3.loc43_4.2 => constants.%MutableRange.End
// CHECK:STDOUT:   %specific_impl_fn.loc43_4.2 => constants.%MutableRange.End
// CHECK:STDOUT:   %impl.elem1.1 => constants.%Sentinel.818
// CHECK:STDOUT:   %require_complete.2 => constants.%complete_type
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.loc43 => constants.%custom_witness.fb5
// CHECK:STDOUT:   %Destroy.facet.loc43 => constants.%Destroy.facet.6ae
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.loc43 => constants.%Destroy.WithSelf.Op.type.45c
// CHECK:STDOUT:   %.loc43_9.3 => constants.%.759
// CHECK:STDOUT:   %impl.elem0.loc43_9.2 => constants.%Sentinel.Op.f79
// CHECK:STDOUT:   %specific_impl_fn.loc43_9.2 => constants.%Sentinel.Op.f79
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.loc42 => constants.%custom_witness.9ce
// CHECK:STDOUT:   %Destroy.facet.loc42 => constants.%Destroy.facet.fec
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.loc42 => constants.%Destroy.WithSelf.Op.type.c56
// CHECK:STDOUT:   %.loc42_11.3 => constants.%.235
// CHECK:STDOUT:   %impl.elem0.loc42_11.2 => constants.%Iterator.Op.d55
// CHECK:STDOUT:   %specific_impl_fn.loc42_11.2 => constants.%Iterator.Op.d55
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(constants.%facet_value.9ee) {
// CHECK:STDOUT:   %R.loc40_10.1 => constants.%facet_value.9ee
// CHECK:STDOUT:   %R.as_type.loc40_113.1 => constants.%ConstRange
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.07c
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc40 => constants.%complete_type
// CHECK:STDOUT:   %CppRangeForIterate.lookup_impl_witness => constants.%custom_witness.522
// CHECK:STDOUT:   %CppRangeForIterate.facet => constants.%CppRangeForIterate.facet.8ca
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.Begin.type => constants.%CppRangeForIterate.WithSelf.Begin.type.1f3
// CHECK:STDOUT:   %.loc42_4 => constants.%.c0c
// CHECK:STDOUT:   %impl.elem2.loc42_4.2 => constants.%ConstRange.Begin
// CHECK:STDOUT:   %specific_impl_fn.loc42_4.2 => constants.%ConstRange.Begin
// CHECK:STDOUT:   %impl.elem0.1 => constants.%Iterator.cc3
// CHECK:STDOUT:   %require_complete.1 => constants.%complete_type
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.End.type => constants.%CppRangeForIterate.WithSelf.End.type.0e5
// CHECK:STDOUT:   %.loc43_4 => constants.%.e09
// CHECK:STDOUT:   %impl.elem3.loc43_4.2 => constants.%ConstRange.End
// CHECK:STDOUT:   %specific_impl_fn.loc43_4.2 => constants.%ConstRange.End
// CHECK:STDOUT:   %impl.elem1.1 => constants.%Sentinel.deb
// CHECK:STDOUT:   %require_complete.2 => constants.%complete_type
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.loc43 => constants.%custom_witness.d98
// CHECK:STDOUT:   %Destroy.facet.loc43 => constants.%Destroy.facet.787
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.loc43 => constants.%Destroy.WithSelf.Op.type.3ee
// CHECK:STDOUT:   %.loc43_9.3 => constants.%.452
// CHECK:STDOUT:   %impl.elem0.loc43_9.2 => constants.%Sentinel.Op.6c9
// CHECK:STDOUT:   %specific_impl_fn.loc43_9.2 => constants.%Sentinel.Op.6c9
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.loc42 => constants.%custom_witness.a80
// CHECK:STDOUT:   %Destroy.facet.loc42 => constants.%Destroy.facet.d38
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.loc42 => constants.%Destroy.WithSelf.Op.type.625
// CHECK:STDOUT:   %.loc42_11.3 => constants.%.448
// CHECK:STDOUT:   %impl.elem0.loc42_11.2 => constants.%Iterator.Op.26d
// CHECK:STDOUT:   %specific_impl_fn.loc42_11.2 => constants.%Iterator.Op.26d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- adl_return_same_types.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %CppRangeForIterate.type: type = facet_type <@CppRangeForIterate> [concrete]
// CHECK:STDOUT:   %Self.981: %CppRangeForIterate.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.End.type.5610e6.1: type = fn_type @CppRangeForIterate.WithSelf.End, @CppRangeForIterate.WithSelf(%Self.981) [symbolic]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.End.65a3e4.1: %CppRangeForIterate.WithSelf.End.type.5610e6.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.Begin.type.91eecf.1: type = fn_type @CppRangeForIterate.WithSelf.Begin, @CppRangeForIterate.WithSelf(%Self.981) [symbolic]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.Begin.07315f.1: %CppRangeForIterate.WithSelf.Begin.type.91eecf.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %.Self.756: %CppRangeForIterate.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %CppRangeForIterate.assoc_type: type = assoc_entity_type @CppRangeForIterate [concrete]
// CHECK:STDOUT:   %CppRangeForIterate.lookup_impl_witness.a19: <witness> = lookup_impl_witness %.Self.756, @CppRangeForIterate [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.5fa: type = impl_witness_access %CppRangeForIterate.lookup_impl_witness.a19, element0 [symbolic_self]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %impl.elem1.b14: type = impl_witness_access %CppRangeForIterate.lookup_impl_witness.a19, element1 [symbolic_self]
// CHECK:STDOUT:   %CppRangeForIterate_where.type: type = facet_type <@CppRangeForIterate where %impl.elem0.5fa impls @Destroy and %impl.elem1.b14 impls @Destroy> [symbolic_self]
// CHECK:STDOUT:   %R: %CppRangeForIterate_where.type = symbolic_binding R, 0 [symbolic]
// CHECK:STDOUT:   %R.as_type: type = facet_access_type %R [symbolic]
// CHECK:STDOUT:   %pattern_type.61273a.2: type = pattern_type %R.as_type [symbolic]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %assoc2: %CppRangeForIterate.assoc_type = assoc_entity element2, imports.%Core.import_ref.a11 [concrete]
// CHECK:STDOUT:   %CppRangeForIterate.lookup_impl_witness.20309f.2: <witness> = lookup_impl_witness %R, @CppRangeForIterate [symbolic]
// CHECK:STDOUT:   %CppRangeForIterate.facet.ee6: %CppRangeForIterate.type = facet_value %R.as_type, (%CppRangeForIterate.lookup_impl_witness.20309f.2) [symbolic]
// CHECK:STDOUT:   %impl.elem0.8e47d6.2: type = impl_witness_access %CppRangeForIterate.lookup_impl_witness.20309f.2, element0 [symbolic]
// CHECK:STDOUT:   %impl.elem1.092b54.2: type = impl_witness_access %CppRangeForIterate.lookup_impl_witness.20309f.2, element1 [symbolic]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.Begin.type.71b: type = fn_type @CppRangeForIterate.WithSelf.Begin, @CppRangeForIterate.WithSelf(%CppRangeForIterate.facet.ee6) [symbolic]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.End.type.c1b: type = fn_type @CppRangeForIterate.WithSelf.End, @CppRangeForIterate.WithSelf(%CppRangeForIterate.facet.ee6) [symbolic]
// CHECK:STDOUT:   %.aa6: type = fn_type_with_self_type %CppRangeForIterate.WithSelf.Begin.type.71b, %CppRangeForIterate.facet.ee6 [symbolic]
// CHECK:STDOUT:   %impl.elem2: %.aa6 = impl_witness_access %CppRangeForIterate.lookup_impl_witness.20309f.2, element2 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.987: <specific function> = specific_impl_function %impl.elem2, @CppRangeForIterate.WithSelf.Begin(%CppRangeForIterate.facet.ee6) [symbolic]
// CHECK:STDOUT:   %assoc3: %CppRangeForIterate.assoc_type = assoc_entity element3, imports.%Core.import_ref.031 [concrete]
// CHECK:STDOUT:   %.5ea: type = fn_type_with_self_type %CppRangeForIterate.WithSelf.End.type.c1b, %CppRangeForIterate.facet.ee6 [symbolic]
// CHECK:STDOUT:   %impl.elem3: %.5ea = impl_witness_access %CppRangeForIterate.lookup_impl_witness.20309f.2, element3 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.748: <specific function> = specific_impl_function %impl.elem3, @CppRangeForIterate.WithSelf.End(%CppRangeForIterate.facet.ee6) [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.aba: <witness> = lookup_impl_witness %impl.elem1.092b54.2, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.ba7: %Destroy.type = facet_value %impl.elem1.092b54.2, (%Destroy.lookup_impl_witness.aba) [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.030: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.ba7) [symbolic]
// CHECK:STDOUT:   %.132: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.030, %Destroy.facet.ba7 [symbolic]
// CHECK:STDOUT:   %impl.elem0.74a: %.132 = impl_witness_access %Destroy.lookup_impl_witness.aba, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.044: <specific function> = specific_impl_function %impl.elem0.74a, @Destroy.WithSelf.Op(%Destroy.facet.ba7) [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.cf0: <witness> = lookup_impl_witness %impl.elem0.8e47d6.2, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.0a9: %Destroy.type = facet_value %impl.elem0.8e47d6.2, (%Destroy.lookup_impl_witness.cf0) [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.b91: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.0a9) [symbolic]
// CHECK:STDOUT:   %.d6b: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.b91, %Destroy.facet.0a9 [symbolic]
// CHECK:STDOUT:   %impl.elem0.a98: %.d6b = impl_witness_access %Destroy.lookup_impl_witness.cf0, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.168: <specific function> = specific_impl_function %impl.elem0.a98, @Destroy.WithSelf.Op(%Destroy.facet.0a9) [symbolic]
// CHECK:STDOUT:   %MutableRange: type = class_type @MutableRange [concrete]
// CHECK:STDOUT:   %pattern_type.8be: type = pattern_type %MutableRange [concrete]
// CHECK:STDOUT:   %ConstRange: type = class_type @ConstRange [concrete]
// CHECK:STDOUT:   %pattern_type.8e2: type = pattern_type %ConstRange [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %Iterator.085: type = class_type @Iterator.1 [concrete]
// CHECK:STDOUT:   %Begin.type.ae21c2.1: type = fn_type @Begin.1 [concrete]
// CHECK:STDOUT:   %Begin.b992d6.1: %Begin.type.ae21c2.1 = struct_value () [concrete]
// CHECK:STDOUT:   %End.type.b365a7.1: type = fn_type @End.1 [concrete]
// CHECK:STDOUT:   %End.2270fc.1: %End.type.b365a7.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.de5: <witness> = custom_witness (%Iterator.085, %Iterator.085, %Begin.b992d6.1, %End.2270fc.1), @CppRangeForIterate [concrete]
// CHECK:STDOUT:   %CppRangeForIterate.facet.91b: %CppRangeForIterate.type = facet_value %MutableRange, (%custom_witness.de5) [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.1e7: type = fn_type @Iterator.Op.1 [concrete]
// CHECK:STDOUT:   %Iterator.Op.88c: %Iterator.Op.type.1e7 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.3b6: <witness> = custom_witness (%Iterator.Op.88c), @Destroy [concrete]
// CHECK:STDOUT:   %facet_value.302: %CppRangeForIterate_where.type = facet_value %MutableRange, (%custom_witness.de5, %custom_witness.3b6) [symbolic_self]
// CHECK:STDOUT:   %Iterator.268: type = class_type @Iterator.2 [concrete]
// CHECK:STDOUT:   %Begin.type.ae21c2.2: type = fn_type @Begin.2 [concrete]
// CHECK:STDOUT:   %Begin.b992d6.2: %Begin.type.ae21c2.2 = struct_value () [concrete]
// CHECK:STDOUT:   %End.type.b365a7.2: type = fn_type @End.2 [concrete]
// CHECK:STDOUT:   %End.2270fc.2: %End.type.b365a7.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.80b: <witness> = custom_witness (%Iterator.268, %Iterator.268, %Begin.b992d6.2, %End.2270fc.2), @CppRangeForIterate [concrete]
// CHECK:STDOUT:   %CppRangeForIterate.facet.dcf: %CppRangeForIterate.type = facet_value %ConstRange, (%custom_witness.80b) [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.635: type = fn_type @Iterator.Op.2 [concrete]
// CHECK:STDOUT:   %Iterator.Op.733: %Iterator.Op.type.635 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.14d: <witness> = custom_witness (%Iterator.Op.733), @Destroy [concrete]
// CHECK:STDOUT:   %facet_value.cdd: %CppRangeForIterate_where.type = facet_value %ConstRange, (%custom_witness.80b, %custom_witness.14d) [symbolic_self]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.Begin.type.781: type = fn_type @CppRangeForIterate.WithSelf.Begin, @CppRangeForIterate.WithSelf(%CppRangeForIterate.facet.91b) [concrete]
// CHECK:STDOUT:   %.508: type = fn_type_with_self_type %CppRangeForIterate.WithSelf.Begin.type.781, %CppRangeForIterate.facet.91b [concrete]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.End.type.2a0: type = fn_type @CppRangeForIterate.WithSelf.End, @CppRangeForIterate.WithSelf(%CppRangeForIterate.facet.91b) [concrete]
// CHECK:STDOUT:   %.253: type = fn_type_with_self_type %CppRangeForIterate.WithSelf.End.type.2a0, %CppRangeForIterate.facet.91b [concrete]
// CHECK:STDOUT:   %Destroy.facet.799: %Destroy.type = facet_value %Iterator.085, (%custom_witness.3b6) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.2fa: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.799) [concrete]
// CHECK:STDOUT:   %.a50: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.2fa, %Destroy.facet.799 [concrete]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.Begin.type.e01: type = fn_type @CppRangeForIterate.WithSelf.Begin, @CppRangeForIterate.WithSelf(%CppRangeForIterate.facet.dcf) [concrete]
// CHECK:STDOUT:   %.35b: type = fn_type_with_self_type %CppRangeForIterate.WithSelf.Begin.type.e01, %CppRangeForIterate.facet.dcf [concrete]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.End.type.66c: type = fn_type @CppRangeForIterate.WithSelf.End, @CppRangeForIterate.WithSelf(%CppRangeForIterate.facet.dcf) [concrete]
// CHECK:STDOUT:   %.14f: type = fn_type_with_self_type %CppRangeForIterate.WithSelf.End.type.66c, %CppRangeForIterate.facet.dcf [concrete]
// CHECK:STDOUT:   %Destroy.facet.6f7: %Destroy.type = facet_value %Iterator.268, (%custom_witness.14d) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.b31: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.6f7) [concrete]
// CHECK:STDOUT:   %.09c: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.b31, %Destroy.facet.6f7 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.822: %CppRangeForIterate.assoc_type = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [concrete = constants.%assoc2]
// CHECK:STDOUT:   %Core.import_ref.685: %CppRangeForIterate.assoc_type = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [concrete = constants.%assoc3]
// CHECK:STDOUT:   %Core.import_ref.a11: @CppRangeForIterate.WithSelf.%CppRangeForIterate.WithSelf.Begin.type (%CppRangeForIterate.WithSelf.Begin.type.91eecf.1) = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @CppRangeForIterate.WithSelf.%CppRangeForIterate.WithSelf.Begin (constants.%CppRangeForIterate.WithSelf.Begin.07315f.1)]
// CHECK:STDOUT:   %Core.import_ref.031: @CppRangeForIterate.WithSelf.%CppRangeForIterate.WithSelf.End.type (%CppRangeForIterate.WithSelf.End.type.5610e6.1) = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @CppRangeForIterate.WithSelf.%CppRangeForIterate.WithSelf.End (constants.%CppRangeForIterate.WithSelf.End.65a3e4.1)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Test(%R.loc32_10.2: %CppRangeForIterate_where.type) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %CppRangeForIterate.lookup_impl_witness: <witness> = lookup_impl_witness %R.loc32_10.1, @CppRangeForIterate [symbolic = %CppRangeForIterate.lookup_impl_witness (constants.%CppRangeForIterate.lookup_impl_witness.20309f.2)]
// CHECK:STDOUT:   %CppRangeForIterate.facet: %CppRangeForIterate.type = facet_value %R.as_type.loc32_113.1, (%CppRangeForIterate.lookup_impl_witness) [symbolic = %CppRangeForIterate.facet (constants.%CppRangeForIterate.facet.ee6)]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.Begin.type: type = fn_type @CppRangeForIterate.WithSelf.Begin, @CppRangeForIterate.WithSelf(%CppRangeForIterate.facet) [symbolic = %CppRangeForIterate.WithSelf.Begin.type (constants.%CppRangeForIterate.WithSelf.Begin.type.71b)]
// CHECK:STDOUT:   %.loc34_4: type = fn_type_with_self_type %CppRangeForIterate.WithSelf.Begin.type, %CppRangeForIterate.facet [symbolic = %.loc34_4 (constants.%.aa6)]
// CHECK:STDOUT:   %impl.elem2.loc34_4.2: @Test.%.loc34_4 (%.aa6) = impl_witness_access %CppRangeForIterate.lookup_impl_witness, element2 [symbolic = %impl.elem2.loc34_4.2 (constants.%impl.elem2)]
// CHECK:STDOUT:   %specific_impl_fn.loc34_4.2: <specific function> = specific_impl_function %impl.elem2.loc34_4.2, @CppRangeForIterate.WithSelf.Begin(%CppRangeForIterate.facet) [symbolic = %specific_impl_fn.loc34_4.2 (constants.%specific_impl_fn.987)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.End.type: type = fn_type @CppRangeForIterate.WithSelf.End, @CppRangeForIterate.WithSelf(%CppRangeForIterate.facet) [symbolic = %CppRangeForIterate.WithSelf.End.type (constants.%CppRangeForIterate.WithSelf.End.type.c1b)]
// CHECK:STDOUT:   %.loc35_4: type = fn_type_with_self_type %CppRangeForIterate.WithSelf.End.type, %CppRangeForIterate.facet [symbolic = %.loc35_4 (constants.%.5ea)]
// CHECK:STDOUT:   %impl.elem3.loc35_4.2: @Test.%.loc35_4 (%.5ea) = impl_witness_access %CppRangeForIterate.lookup_impl_witness, element3 [symbolic = %impl.elem3.loc35_4.2 (constants.%impl.elem3)]
// CHECK:STDOUT:   %specific_impl_fn.loc35_4.2: <specific function> = specific_impl_function %impl.elem3.loc35_4.2, @CppRangeForIterate.WithSelf.End(%CppRangeForIterate.facet) [symbolic = %specific_impl_fn.loc35_4.2 (constants.%specific_impl_fn.748)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.loc35: <witness> = lookup_impl_witness %impl.elem1.1, @Destroy [symbolic = %Destroy.lookup_impl_witness.loc35 (constants.%Destroy.lookup_impl_witness.aba)]
// CHECK:STDOUT:   %Destroy.facet.loc35: %Destroy.type = facet_value %impl.elem1.1, (%Destroy.lookup_impl_witness.loc35) [symbolic = %Destroy.facet.loc35 (constants.%Destroy.facet.ba7)]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.loc35: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.loc35) [symbolic = %Destroy.WithSelf.Op.type.loc35 (constants.%Destroy.WithSelf.Op.type.030)]
// CHECK:STDOUT:   %.loc35_9.3: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.loc35, %Destroy.facet.loc35 [symbolic = %.loc35_9.3 (constants.%.132)]
// CHECK:STDOUT:   %impl.elem0.loc35_9.2: @Test.%.loc35_9.3 (%.132) = impl_witness_access %Destroy.lookup_impl_witness.loc35, element0 [symbolic = %impl.elem0.loc35_9.2 (constants.%impl.elem0.74a)]
// CHECK:STDOUT:   %specific_impl_fn.loc35_9.2: <specific function> = specific_impl_function %impl.elem0.loc35_9.2, @Destroy.WithSelf.Op(%Destroy.facet.loc35) [symbolic = %specific_impl_fn.loc35_9.2 (constants.%specific_impl_fn.044)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.loc34: <witness> = lookup_impl_witness %impl.elem0.1, @Destroy [symbolic = %Destroy.lookup_impl_witness.loc34 (constants.%Destroy.lookup_impl_witness.cf0)]
// CHECK:STDOUT:   %Destroy.facet.loc34: %Destroy.type = facet_value %impl.elem0.1, (%Destroy.lookup_impl_witness.loc34) [symbolic = %Destroy.facet.loc34 (constants.%Destroy.facet.0a9)]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.loc34: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.loc34) [symbolic = %Destroy.WithSelf.Op.type.loc34 (constants.%Destroy.WithSelf.Op.type.b91)]
// CHECK:STDOUT:   %.loc34_11.3: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.loc34, %Destroy.facet.loc34 [symbolic = %.loc34_11.3 (constants.%.d6b)]
// CHECK:STDOUT:   %impl.elem0.loc34_11.2: @Test.%.loc34_11.3 (%.d6b) = impl_witness_access %Destroy.lookup_impl_witness.loc34, element0 [symbolic = %impl.elem0.loc34_11.2 (constants.%impl.elem0.a98)]
// CHECK:STDOUT:   %specific_impl_fn.loc34_11.2: <specific function> = specific_impl_function %impl.elem0.loc34_11.2, @Destroy.WithSelf.Op(%Destroy.facet.loc34) [symbolic = %specific_impl_fn.loc34_11.2 (constants.%specific_impl_fn.168)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%r.param: ref @Test.%R.as_type.loc32_113.1 (%R.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %r.ref.loc34: ref @Test.%R.as_type.loc32_113.1 (%R.as_type) = name_ref r, %r
// CHECK:STDOUT:     %Begin.ref: %CppRangeForIterate.assoc_type = name_ref Begin, imports.%Core.import_ref.822 [concrete = constants.%assoc2]
// CHECK:STDOUT:     %impl.elem2.loc34_4.1: @Test.%.loc34_4 (%.aa6) = impl_witness_access constants.%CppRangeForIterate.lookup_impl_witness.20309f.2, element2 [symbolic = %impl.elem2.loc34_4.2 (constants.%impl.elem2)]
// CHECK:STDOUT:     %bound_method.loc34_4: <bound method> = bound_method %r.ref.loc34, %impl.elem2.loc34_4.1
// CHECK:STDOUT:     %specific_impl_fn.loc34_4.1: <specific function> = specific_impl_function %impl.elem2.loc34_4.1, @CppRangeForIterate.WithSelf.Begin(constants.%CppRangeForIterate.facet.ee6) [symbolic = %specific_impl_fn.loc34_4.2 (constants.%specific_impl_fn.987)]
// CHECK:STDOUT:     %bound_method.loc34_11.1: <bound method> = bound_method %r.ref.loc34, %specific_impl_fn.loc34_4.1
// CHECK:STDOUT:     %.loc34_11.1: ref @Test.%impl.elem0.1 (%impl.elem0.8e47d6.2) = temporary_storage
// CHECK:STDOUT:     %CppRangeForIterate.WithSelf.Begin.call: init @Test.%impl.elem0.1 (%impl.elem0.8e47d6.2) to %.loc34_11.1 = call %bound_method.loc34_11.1(%r.ref.loc34)
// CHECK:STDOUT:     %.loc34_11.2: ref @Test.%impl.elem0.1 (%impl.elem0.8e47d6.2) = temporary %.loc34_11.1, %CppRangeForIterate.WithSelf.Begin.call
// CHECK:STDOUT:     %r.ref.loc35: ref @Test.%R.as_type.loc32_113.1 (%R.as_type) = name_ref r, %r
// CHECK:STDOUT:     %End.ref: %CppRangeForIterate.assoc_type = name_ref End, imports.%Core.import_ref.685 [concrete = constants.%assoc3]
// CHECK:STDOUT:     %impl.elem3.loc35_4.1: @Test.%.loc35_4 (%.5ea) = impl_witness_access constants.%CppRangeForIterate.lookup_impl_witness.20309f.2, element3 [symbolic = %impl.elem3.loc35_4.2 (constants.%impl.elem3)]
// CHECK:STDOUT:     %bound_method.loc35_4: <bound method> = bound_method %r.ref.loc35, %impl.elem3.loc35_4.1
// CHECK:STDOUT:     %specific_impl_fn.loc35_4.1: <specific function> = specific_impl_function %impl.elem3.loc35_4.1, @CppRangeForIterate.WithSelf.End(constants.%CppRangeForIterate.facet.ee6) [symbolic = %specific_impl_fn.loc35_4.2 (constants.%specific_impl_fn.748)]
// CHECK:STDOUT:     %bound_method.loc35_9.1: <bound method> = bound_method %r.ref.loc35, %specific_impl_fn.loc35_4.1
// CHECK:STDOUT:     %.loc35_9.1: ref @Test.%impl.elem1.1 (%impl.elem1.092b54.2) = temporary_storage
// CHECK:STDOUT:     %CppRangeForIterate.WithSelf.End.call: init @Test.%impl.elem1.1 (%impl.elem1.092b54.2) to %.loc35_9.1 = call %bound_method.loc35_9.1(%r.ref.loc35)
// CHECK:STDOUT:     %.loc35_9.2: ref @Test.%impl.elem1.1 (%impl.elem1.092b54.2) = temporary %.loc35_9.1, %CppRangeForIterate.WithSelf.End.call
// CHECK:STDOUT:     %impl.elem0.loc35_9.1: @Test.%.loc35_9.3 (%.132) = impl_witness_access constants.%Destroy.lookup_impl_witness.aba, element0 [symbolic = %impl.elem0.loc35_9.2 (constants.%impl.elem0.74a)]
// CHECK:STDOUT:     %bound_method.loc35_9.2: <bound method> = bound_method %.loc35_9.2, %impl.elem0.loc35_9.1
// CHECK:STDOUT:     %specific_impl_fn.loc35_9.1: <specific function> = specific_impl_function %impl.elem0.loc35_9.1, @Destroy.WithSelf.Op(constants.%Destroy.facet.ba7) [symbolic = %specific_impl_fn.loc35_9.2 (constants.%specific_impl_fn.044)]
// CHECK:STDOUT:     %bound_method.loc35_9.3: <bound method> = bound_method %.loc35_9.2, %specific_impl_fn.loc35_9.1
// CHECK:STDOUT:     %Destroy.WithSelf.Op.call.loc35: init %empty_tuple.type = call %bound_method.loc35_9.3(%.loc35_9.2)
// CHECK:STDOUT:     %impl.elem0.loc34_11.1: @Test.%.loc34_11.3 (%.d6b) = impl_witness_access constants.%Destroy.lookup_impl_witness.cf0, element0 [symbolic = %impl.elem0.loc34_11.2 (constants.%impl.elem0.a98)]
// CHECK:STDOUT:     %bound_method.loc34_11.2: <bound method> = bound_method %.loc34_11.2, %impl.elem0.loc34_11.1
// CHECK:STDOUT:     %specific_impl_fn.loc34_11.1: <specific function> = specific_impl_function %impl.elem0.loc34_11.1, @Destroy.WithSelf.Op(constants.%Destroy.facet.0a9) [symbolic = %specific_impl_fn.loc34_11.2 (constants.%specific_impl_fn.168)]
// CHECK:STDOUT:     %bound_method.loc34_11.3: <bound method> = bound_method %.loc34_11.2, %specific_impl_fn.loc34_11.1
// CHECK:STDOUT:     %Destroy.WithSelf.Op.call.loc34: init %empty_tuple.type = call %bound_method.loc34_11.3(%.loc34_11.2)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(constants.%R) {
// CHECK:STDOUT:   %R.loc32_10.1 => constants.%R
// CHECK:STDOUT:   %R.as_type.loc32_113.1 => constants.%R.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.61273a.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(constants.%facet_value.302) {
// CHECK:STDOUT:   %R.loc32_10.1 => constants.%facet_value.302
// CHECK:STDOUT:   %R.as_type.loc32_113.1 => constants.%MutableRange
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.8be
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc32 => constants.%complete_type
// CHECK:STDOUT:   %CppRangeForIterate.lookup_impl_witness => constants.%custom_witness.de5
// CHECK:STDOUT:   %CppRangeForIterate.facet => constants.%CppRangeForIterate.facet.91b
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.Begin.type => constants.%CppRangeForIterate.WithSelf.Begin.type.781
// CHECK:STDOUT:   %.loc34_4 => constants.%.508
// CHECK:STDOUT:   %impl.elem2.loc34_4.2 => constants.%Begin.b992d6.1
// CHECK:STDOUT:   %specific_impl_fn.loc34_4.2 => constants.%Begin.b992d6.1
// CHECK:STDOUT:   %impl.elem0.1 => constants.%Iterator.085
// CHECK:STDOUT:   %require_complete.1 => constants.%complete_type
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.End.type => constants.%CppRangeForIterate.WithSelf.End.type.2a0
// CHECK:STDOUT:   %.loc35_4 => constants.%.253
// CHECK:STDOUT:   %impl.elem3.loc35_4.2 => constants.%End.2270fc.1
// CHECK:STDOUT:   %specific_impl_fn.loc35_4.2 => constants.%End.2270fc.1
// CHECK:STDOUT:   %impl.elem1.1 => constants.%Iterator.085
// CHECK:STDOUT:   %require_complete.2 => constants.%complete_type
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.loc35 => constants.%custom_witness.3b6
// CHECK:STDOUT:   %Destroy.facet.loc35 => constants.%Destroy.facet.799
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.loc35 => constants.%Destroy.WithSelf.Op.type.2fa
// CHECK:STDOUT:   %.loc35_9.3 => constants.%.a50
// CHECK:STDOUT:   %impl.elem0.loc35_9.2 => constants.%Iterator.Op.88c
// CHECK:STDOUT:   %specific_impl_fn.loc35_9.2 => constants.%Iterator.Op.88c
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.loc34 => constants.%custom_witness.3b6
// CHECK:STDOUT:   %Destroy.facet.loc34 => constants.%Destroy.facet.799
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.loc34 => constants.%Destroy.WithSelf.Op.type.2fa
// CHECK:STDOUT:   %.loc34_11.3 => constants.%.a50
// CHECK:STDOUT:   %impl.elem0.loc34_11.2 => constants.%Iterator.Op.88c
// CHECK:STDOUT:   %specific_impl_fn.loc34_11.2 => constants.%Iterator.Op.88c
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(constants.%facet_value.cdd) {
// CHECK:STDOUT:   %R.loc32_10.1 => constants.%facet_value.cdd
// CHECK:STDOUT:   %R.as_type.loc32_113.1 => constants.%ConstRange
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.8e2
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc32 => constants.%complete_type
// CHECK:STDOUT:   %CppRangeForIterate.lookup_impl_witness => constants.%custom_witness.80b
// CHECK:STDOUT:   %CppRangeForIterate.facet => constants.%CppRangeForIterate.facet.dcf
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.Begin.type => constants.%CppRangeForIterate.WithSelf.Begin.type.e01
// CHECK:STDOUT:   %.loc34_4 => constants.%.35b
// CHECK:STDOUT:   %impl.elem2.loc34_4.2 => constants.%Begin.b992d6.2
// CHECK:STDOUT:   %specific_impl_fn.loc34_4.2 => constants.%Begin.b992d6.2
// CHECK:STDOUT:   %impl.elem0.1 => constants.%Iterator.268
// CHECK:STDOUT:   %require_complete.1 => constants.%complete_type
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.End.type => constants.%CppRangeForIterate.WithSelf.End.type.66c
// CHECK:STDOUT:   %.loc35_4 => constants.%.14f
// CHECK:STDOUT:   %impl.elem3.loc35_4.2 => constants.%End.2270fc.2
// CHECK:STDOUT:   %specific_impl_fn.loc35_4.2 => constants.%End.2270fc.2
// CHECK:STDOUT:   %impl.elem1.1 => constants.%Iterator.268
// CHECK:STDOUT:   %require_complete.2 => constants.%complete_type
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.loc35 => constants.%custom_witness.14d
// CHECK:STDOUT:   %Destroy.facet.loc35 => constants.%Destroy.facet.6f7
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.loc35 => constants.%Destroy.WithSelf.Op.type.b31
// CHECK:STDOUT:   %.loc35_9.3 => constants.%.09c
// CHECK:STDOUT:   %impl.elem0.loc35_9.2 => constants.%Iterator.Op.733
// CHECK:STDOUT:   %specific_impl_fn.loc35_9.2 => constants.%Iterator.Op.733
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.loc34 => constants.%custom_witness.14d
// CHECK:STDOUT:   %Destroy.facet.loc34 => constants.%Destroy.facet.6f7
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.loc34 => constants.%Destroy.WithSelf.Op.type.b31
// CHECK:STDOUT:   %.loc34_11.3 => constants.%.09c
// CHECK:STDOUT:   %impl.elem0.loc34_11.2 => constants.%Iterator.Op.733
// CHECK:STDOUT:   %specific_impl_fn.loc34_11.2 => constants.%Iterator.Op.733
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- adl_returns_different_types.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %CppRangeForIterate.type: type = facet_type <@CppRangeForIterate> [concrete]
// CHECK:STDOUT:   %Self.981: %CppRangeForIterate.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.End.type.5610e6.1: type = fn_type @CppRangeForIterate.WithSelf.End, @CppRangeForIterate.WithSelf(%Self.981) [symbolic]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.End.65a3e4.1: %CppRangeForIterate.WithSelf.End.type.5610e6.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.Begin.type.91eecf.1: type = fn_type @CppRangeForIterate.WithSelf.Begin, @CppRangeForIterate.WithSelf(%Self.981) [symbolic]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.Begin.07315f.1: %CppRangeForIterate.WithSelf.Begin.type.91eecf.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %.Self.756: %CppRangeForIterate.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %CppRangeForIterate.assoc_type: type = assoc_entity_type @CppRangeForIterate [concrete]
// CHECK:STDOUT:   %CppRangeForIterate.lookup_impl_witness.a19: <witness> = lookup_impl_witness %.Self.756, @CppRangeForIterate [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.5fa: type = impl_witness_access %CppRangeForIterate.lookup_impl_witness.a19, element0 [symbolic_self]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %impl.elem1.b14: type = impl_witness_access %CppRangeForIterate.lookup_impl_witness.a19, element1 [symbolic_self]
// CHECK:STDOUT:   %CppRangeForIterate_where.type: type = facet_type <@CppRangeForIterate where %impl.elem0.5fa impls @Destroy and %impl.elem1.b14 impls @Destroy> [symbolic_self]
// CHECK:STDOUT:   %R: %CppRangeForIterate_where.type = symbolic_binding R, 0 [symbolic]
// CHECK:STDOUT:   %R.as_type: type = facet_access_type %R [symbolic]
// CHECK:STDOUT:   %pattern_type.61273a.2: type = pattern_type %R.as_type [symbolic]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %assoc2: %CppRangeForIterate.assoc_type = assoc_entity element2, imports.%Core.import_ref.a11 [concrete]
// CHECK:STDOUT:   %CppRangeForIterate.lookup_impl_witness.20309f.2: <witness> = lookup_impl_witness %R, @CppRangeForIterate [symbolic]
// CHECK:STDOUT:   %CppRangeForIterate.facet.ee6: %CppRangeForIterate.type = facet_value %R.as_type, (%CppRangeForIterate.lookup_impl_witness.20309f.2) [symbolic]
// CHECK:STDOUT:   %impl.elem0.8e47d6.2: type = impl_witness_access %CppRangeForIterate.lookup_impl_witness.20309f.2, element0 [symbolic]
// CHECK:STDOUT:   %impl.elem1.092b54.2: type = impl_witness_access %CppRangeForIterate.lookup_impl_witness.20309f.2, element1 [symbolic]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.Begin.type.71b: type = fn_type @CppRangeForIterate.WithSelf.Begin, @CppRangeForIterate.WithSelf(%CppRangeForIterate.facet.ee6) [symbolic]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.End.type.c1b: type = fn_type @CppRangeForIterate.WithSelf.End, @CppRangeForIterate.WithSelf(%CppRangeForIterate.facet.ee6) [symbolic]
// CHECK:STDOUT:   %.aa6: type = fn_type_with_self_type %CppRangeForIterate.WithSelf.Begin.type.71b, %CppRangeForIterate.facet.ee6 [symbolic]
// CHECK:STDOUT:   %impl.elem2: %.aa6 = impl_witness_access %CppRangeForIterate.lookup_impl_witness.20309f.2, element2 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.987: <specific function> = specific_impl_function %impl.elem2, @CppRangeForIterate.WithSelf.Begin(%CppRangeForIterate.facet.ee6) [symbolic]
// CHECK:STDOUT:   %assoc3: %CppRangeForIterate.assoc_type = assoc_entity element3, imports.%Core.import_ref.031 [concrete]
// CHECK:STDOUT:   %.5ea: type = fn_type_with_self_type %CppRangeForIterate.WithSelf.End.type.c1b, %CppRangeForIterate.facet.ee6 [symbolic]
// CHECK:STDOUT:   %impl.elem3: %.5ea = impl_witness_access %CppRangeForIterate.lookup_impl_witness.20309f.2, element3 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.748: <specific function> = specific_impl_function %impl.elem3, @CppRangeForIterate.WithSelf.End(%CppRangeForIterate.facet.ee6) [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.aba: <witness> = lookup_impl_witness %impl.elem1.092b54.2, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.ba7: %Destroy.type = facet_value %impl.elem1.092b54.2, (%Destroy.lookup_impl_witness.aba) [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.030: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.ba7) [symbolic]
// CHECK:STDOUT:   %.132: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.030, %Destroy.facet.ba7 [symbolic]
// CHECK:STDOUT:   %impl.elem0.74a: %.132 = impl_witness_access %Destroy.lookup_impl_witness.aba, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.044: <specific function> = specific_impl_function %impl.elem0.74a, @Destroy.WithSelf.Op(%Destroy.facet.ba7) [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.cf0: <witness> = lookup_impl_witness %impl.elem0.8e47d6.2, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.0a9: %Destroy.type = facet_value %impl.elem0.8e47d6.2, (%Destroy.lookup_impl_witness.cf0) [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.b91: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.0a9) [symbolic]
// CHECK:STDOUT:   %.d6b: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.b91, %Destroy.facet.0a9 [symbolic]
// CHECK:STDOUT:   %impl.elem0.a98: %.d6b = impl_witness_access %Destroy.lookup_impl_witness.cf0, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.168: <specific function> = specific_impl_function %impl.elem0.a98, @Destroy.WithSelf.Op(%Destroy.facet.0a9) [symbolic]
// CHECK:STDOUT:   %MutableRange: type = class_type @MutableRange [concrete]
// CHECK:STDOUT:   %pattern_type.8be: type = pattern_type %MutableRange [concrete]
// CHECK:STDOUT:   %ConstRange: type = class_type @ConstRange [concrete]
// CHECK:STDOUT:   %pattern_type.8e2: type = pattern_type %ConstRange [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %Iterator.085: type = class_type @Iterator.1 [concrete]
// CHECK:STDOUT:   %Sentinel.788: type = class_type @Sentinel.1 [concrete]
// CHECK:STDOUT:   %Begin.type.ae21c2.1: type = fn_type @Begin.1 [concrete]
// CHECK:STDOUT:   %Begin.b992d6.1: %Begin.type.ae21c2.1 = struct_value () [concrete]
// CHECK:STDOUT:   %End.type.b365a7.1: type = fn_type @End.1 [concrete]
// CHECK:STDOUT:   %End.2270fc.1: %End.type.b365a7.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.4cf: <witness> = custom_witness (%Iterator.085, %Sentinel.788, %Begin.b992d6.1, %End.2270fc.1), @CppRangeForIterate [concrete]
// CHECK:STDOUT:   %CppRangeForIterate.facet.517: %CppRangeForIterate.type = facet_value %MutableRange, (%custom_witness.4cf) [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.1e7: type = fn_type @Iterator.Op.1 [concrete]
// CHECK:STDOUT:   %Iterator.Op.88c: %Iterator.Op.type.1e7 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.3b6: <witness> = custom_witness (%Iterator.Op.88c), @Destroy [concrete]
// CHECK:STDOUT:   %Sentinel.Op.type.f72: type = fn_type @Sentinel.Op.1 [concrete]
// CHECK:STDOUT:   %Sentinel.Op.f1e: %Sentinel.Op.type.f72 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.3af: <witness> = custom_witness (%Sentinel.Op.f1e), @Destroy [concrete]
// CHECK:STDOUT:   %facet_value.a6b: %CppRangeForIterate_where.type = facet_value %MutableRange, (%custom_witness.4cf, %custom_witness.3b6, %custom_witness.3af) [symbolic_self]
// CHECK:STDOUT:   %Iterator.268: type = class_type @Iterator.2 [concrete]
// CHECK:STDOUT:   %Sentinel.155: type = class_type @Sentinel.2 [concrete]
// CHECK:STDOUT:   %Begin.type.ae21c2.2: type = fn_type @Begin.2 [concrete]
// CHECK:STDOUT:   %Begin.b992d6.2: %Begin.type.ae21c2.2 = struct_value () [concrete]
// CHECK:STDOUT:   %End.type.b365a7.2: type = fn_type @End.2 [concrete]
// CHECK:STDOUT:   %End.2270fc.2: %End.type.b365a7.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.47c: <witness> = custom_witness (%Iterator.268, %Sentinel.155, %Begin.b992d6.2, %End.2270fc.2), @CppRangeForIterate [concrete]
// CHECK:STDOUT:   %CppRangeForIterate.facet.cba: %CppRangeForIterate.type = facet_value %ConstRange, (%custom_witness.47c) [concrete]
// CHECK:STDOUT:   %Iterator.Op.type.635: type = fn_type @Iterator.Op.2 [concrete]
// CHECK:STDOUT:   %Iterator.Op.733: %Iterator.Op.type.635 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.14d: <witness> = custom_witness (%Iterator.Op.733), @Destroy [concrete]
// CHECK:STDOUT:   %Sentinel.Op.type.a50: type = fn_type @Sentinel.Op.2 [concrete]
// CHECK:STDOUT:   %Sentinel.Op.0bd: %Sentinel.Op.type.a50 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.9cf: <witness> = custom_witness (%Sentinel.Op.0bd), @Destroy [concrete]
// CHECK:STDOUT:   %facet_value.c5b: %CppRangeForIterate_where.type = facet_value %ConstRange, (%custom_witness.47c, %custom_witness.14d, %custom_witness.9cf) [symbolic_self]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.Begin.type.80e: type = fn_type @CppRangeForIterate.WithSelf.Begin, @CppRangeForIterate.WithSelf(%CppRangeForIterate.facet.517) [concrete]
// CHECK:STDOUT:   %.3f3: type = fn_type_with_self_type %CppRangeForIterate.WithSelf.Begin.type.80e, %CppRangeForIterate.facet.517 [concrete]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.End.type.93c: type = fn_type @CppRangeForIterate.WithSelf.End, @CppRangeForIterate.WithSelf(%CppRangeForIterate.facet.517) [concrete]
// CHECK:STDOUT:   %.21b: type = fn_type_with_self_type %CppRangeForIterate.WithSelf.End.type.93c, %CppRangeForIterate.facet.517 [concrete]
// CHECK:STDOUT:   %Destroy.facet.c1b: %Destroy.type = facet_value %Sentinel.788, (%custom_witness.3af) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.835: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.c1b) [concrete]
// CHECK:STDOUT:   %.0cb: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.835, %Destroy.facet.c1b [concrete]
// CHECK:STDOUT:   %Destroy.facet.799: %Destroy.type = facet_value %Iterator.085, (%custom_witness.3b6) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.2fa: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.799) [concrete]
// CHECK:STDOUT:   %.a50: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.2fa, %Destroy.facet.799 [concrete]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.Begin.type.ae7: type = fn_type @CppRangeForIterate.WithSelf.Begin, @CppRangeForIterate.WithSelf(%CppRangeForIterate.facet.cba) [concrete]
// CHECK:STDOUT:   %.ace: type = fn_type_with_self_type %CppRangeForIterate.WithSelf.Begin.type.ae7, %CppRangeForIterate.facet.cba [concrete]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.End.type.2a0: type = fn_type @CppRangeForIterate.WithSelf.End, @CppRangeForIterate.WithSelf(%CppRangeForIterate.facet.cba) [concrete]
// CHECK:STDOUT:   %.956: type = fn_type_with_self_type %CppRangeForIterate.WithSelf.End.type.2a0, %CppRangeForIterate.facet.cba [concrete]
// CHECK:STDOUT:   %Destroy.facet.f4f: %Destroy.type = facet_value %Sentinel.155, (%custom_witness.9cf) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.923: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.f4f) [concrete]
// CHECK:STDOUT:   %.2f2: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.923, %Destroy.facet.f4f [concrete]
// CHECK:STDOUT:   %Destroy.facet.6f7: %Destroy.type = facet_value %Iterator.268, (%custom_witness.14d) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.b31: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.6f7) [concrete]
// CHECK:STDOUT:   %.09c: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.b31, %Destroy.facet.6f7 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.822: %CppRangeForIterate.assoc_type = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [concrete = constants.%assoc2]
// CHECK:STDOUT:   %Core.import_ref.685: %CppRangeForIterate.assoc_type = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [concrete = constants.%assoc3]
// CHECK:STDOUT:   %Core.import_ref.a11: @CppRangeForIterate.WithSelf.%CppRangeForIterate.WithSelf.Begin.type (%CppRangeForIterate.WithSelf.Begin.type.91eecf.1) = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @CppRangeForIterate.WithSelf.%CppRangeForIterate.WithSelf.Begin (constants.%CppRangeForIterate.WithSelf.Begin.07315f.1)]
// CHECK:STDOUT:   %Core.import_ref.031: @CppRangeForIterate.WithSelf.%CppRangeForIterate.WithSelf.End.type (%CppRangeForIterate.WithSelf.End.type.5610e6.1) = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @CppRangeForIterate.WithSelf.%CppRangeForIterate.WithSelf.End (constants.%CppRangeForIterate.WithSelf.End.65a3e4.1)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Test(%R.loc42_10.2: %CppRangeForIterate_where.type) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %CppRangeForIterate.lookup_impl_witness: <witness> = lookup_impl_witness %R.loc42_10.1, @CppRangeForIterate [symbolic = %CppRangeForIterate.lookup_impl_witness (constants.%CppRangeForIterate.lookup_impl_witness.20309f.2)]
// CHECK:STDOUT:   %CppRangeForIterate.facet: %CppRangeForIterate.type = facet_value %R.as_type.loc42_113.1, (%CppRangeForIterate.lookup_impl_witness) [symbolic = %CppRangeForIterate.facet (constants.%CppRangeForIterate.facet.ee6)]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.Begin.type: type = fn_type @CppRangeForIterate.WithSelf.Begin, @CppRangeForIterate.WithSelf(%CppRangeForIterate.facet) [symbolic = %CppRangeForIterate.WithSelf.Begin.type (constants.%CppRangeForIterate.WithSelf.Begin.type.71b)]
// CHECK:STDOUT:   %.loc44_4: type = fn_type_with_self_type %CppRangeForIterate.WithSelf.Begin.type, %CppRangeForIterate.facet [symbolic = %.loc44_4 (constants.%.aa6)]
// CHECK:STDOUT:   %impl.elem2.loc44_4.2: @Test.%.loc44_4 (%.aa6) = impl_witness_access %CppRangeForIterate.lookup_impl_witness, element2 [symbolic = %impl.elem2.loc44_4.2 (constants.%impl.elem2)]
// CHECK:STDOUT:   %specific_impl_fn.loc44_4.2: <specific function> = specific_impl_function %impl.elem2.loc44_4.2, @CppRangeForIterate.WithSelf.Begin(%CppRangeForIterate.facet) [symbolic = %specific_impl_fn.loc44_4.2 (constants.%specific_impl_fn.987)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.End.type: type = fn_type @CppRangeForIterate.WithSelf.End, @CppRangeForIterate.WithSelf(%CppRangeForIterate.facet) [symbolic = %CppRangeForIterate.WithSelf.End.type (constants.%CppRangeForIterate.WithSelf.End.type.c1b)]
// CHECK:STDOUT:   %.loc45_4: type = fn_type_with_self_type %CppRangeForIterate.WithSelf.End.type, %CppRangeForIterate.facet [symbolic = %.loc45_4 (constants.%.5ea)]
// CHECK:STDOUT:   %impl.elem3.loc45_4.2: @Test.%.loc45_4 (%.5ea) = impl_witness_access %CppRangeForIterate.lookup_impl_witness, element3 [symbolic = %impl.elem3.loc45_4.2 (constants.%impl.elem3)]
// CHECK:STDOUT:   %specific_impl_fn.loc45_4.2: <specific function> = specific_impl_function %impl.elem3.loc45_4.2, @CppRangeForIterate.WithSelf.End(%CppRangeForIterate.facet) [symbolic = %specific_impl_fn.loc45_4.2 (constants.%specific_impl_fn.748)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.loc45: <witness> = lookup_impl_witness %impl.elem1.1, @Destroy [symbolic = %Destroy.lookup_impl_witness.loc45 (constants.%Destroy.lookup_impl_witness.aba)]
// CHECK:STDOUT:   %Destroy.facet.loc45: %Destroy.type = facet_value %impl.elem1.1, (%Destroy.lookup_impl_witness.loc45) [symbolic = %Destroy.facet.loc45 (constants.%Destroy.facet.ba7)]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.loc45: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.loc45) [symbolic = %Destroy.WithSelf.Op.type.loc45 (constants.%Destroy.WithSelf.Op.type.030)]
// CHECK:STDOUT:   %.loc45_9.3: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.loc45, %Destroy.facet.loc45 [symbolic = %.loc45_9.3 (constants.%.132)]
// CHECK:STDOUT:   %impl.elem0.loc45_9.2: @Test.%.loc45_9.3 (%.132) = impl_witness_access %Destroy.lookup_impl_witness.loc45, element0 [symbolic = %impl.elem0.loc45_9.2 (constants.%impl.elem0.74a)]
// CHECK:STDOUT:   %specific_impl_fn.loc45_9.2: <specific function> = specific_impl_function %impl.elem0.loc45_9.2, @Destroy.WithSelf.Op(%Destroy.facet.loc45) [symbolic = %specific_impl_fn.loc45_9.2 (constants.%specific_impl_fn.044)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.loc44: <witness> = lookup_impl_witness %impl.elem0.1, @Destroy [symbolic = %Destroy.lookup_impl_witness.loc44 (constants.%Destroy.lookup_impl_witness.cf0)]
// CHECK:STDOUT:   %Destroy.facet.loc44: %Destroy.type = facet_value %impl.elem0.1, (%Destroy.lookup_impl_witness.loc44) [symbolic = %Destroy.facet.loc44 (constants.%Destroy.facet.0a9)]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.loc44: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.loc44) [symbolic = %Destroy.WithSelf.Op.type.loc44 (constants.%Destroy.WithSelf.Op.type.b91)]
// CHECK:STDOUT:   %.loc44_11.3: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.loc44, %Destroy.facet.loc44 [symbolic = %.loc44_11.3 (constants.%.d6b)]
// CHECK:STDOUT:   %impl.elem0.loc44_11.2: @Test.%.loc44_11.3 (%.d6b) = impl_witness_access %Destroy.lookup_impl_witness.loc44, element0 [symbolic = %impl.elem0.loc44_11.2 (constants.%impl.elem0.a98)]
// CHECK:STDOUT:   %specific_impl_fn.loc44_11.2: <specific function> = specific_impl_function %impl.elem0.loc44_11.2, @Destroy.WithSelf.Op(%Destroy.facet.loc44) [symbolic = %specific_impl_fn.loc44_11.2 (constants.%specific_impl_fn.168)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%r.param: ref @Test.%R.as_type.loc42_113.1 (%R.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %r.ref.loc44: ref @Test.%R.as_type.loc42_113.1 (%R.as_type) = name_ref r, %r
// CHECK:STDOUT:     %Begin.ref: %CppRangeForIterate.assoc_type = name_ref Begin, imports.%Core.import_ref.822 [concrete = constants.%assoc2]
// CHECK:STDOUT:     %impl.elem2.loc44_4.1: @Test.%.loc44_4 (%.aa6) = impl_witness_access constants.%CppRangeForIterate.lookup_impl_witness.20309f.2, element2 [symbolic = %impl.elem2.loc44_4.2 (constants.%impl.elem2)]
// CHECK:STDOUT:     %bound_method.loc44_4: <bound method> = bound_method %r.ref.loc44, %impl.elem2.loc44_4.1
// CHECK:STDOUT:     %specific_impl_fn.loc44_4.1: <specific function> = specific_impl_function %impl.elem2.loc44_4.1, @CppRangeForIterate.WithSelf.Begin(constants.%CppRangeForIterate.facet.ee6) [symbolic = %specific_impl_fn.loc44_4.2 (constants.%specific_impl_fn.987)]
// CHECK:STDOUT:     %bound_method.loc44_11.1: <bound method> = bound_method %r.ref.loc44, %specific_impl_fn.loc44_4.1
// CHECK:STDOUT:     %.loc44_11.1: ref @Test.%impl.elem0.1 (%impl.elem0.8e47d6.2) = temporary_storage
// CHECK:STDOUT:     %CppRangeForIterate.WithSelf.Begin.call: init @Test.%impl.elem0.1 (%impl.elem0.8e47d6.2) to %.loc44_11.1 = call %bound_method.loc44_11.1(%r.ref.loc44)
// CHECK:STDOUT:     %.loc44_11.2: ref @Test.%impl.elem0.1 (%impl.elem0.8e47d6.2) = temporary %.loc44_11.1, %CppRangeForIterate.WithSelf.Begin.call
// CHECK:STDOUT:     %r.ref.loc45: ref @Test.%R.as_type.loc42_113.1 (%R.as_type) = name_ref r, %r
// CHECK:STDOUT:     %End.ref: %CppRangeForIterate.assoc_type = name_ref End, imports.%Core.import_ref.685 [concrete = constants.%assoc3]
// CHECK:STDOUT:     %impl.elem3.loc45_4.1: @Test.%.loc45_4 (%.5ea) = impl_witness_access constants.%CppRangeForIterate.lookup_impl_witness.20309f.2, element3 [symbolic = %impl.elem3.loc45_4.2 (constants.%impl.elem3)]
// CHECK:STDOUT:     %bound_method.loc45_4: <bound method> = bound_method %r.ref.loc45, %impl.elem3.loc45_4.1
// CHECK:STDOUT:     %specific_impl_fn.loc45_4.1: <specific function> = specific_impl_function %impl.elem3.loc45_4.1, @CppRangeForIterate.WithSelf.End(constants.%CppRangeForIterate.facet.ee6) [symbolic = %specific_impl_fn.loc45_4.2 (constants.%specific_impl_fn.748)]
// CHECK:STDOUT:     %bound_method.loc45_9.1: <bound method> = bound_method %r.ref.loc45, %specific_impl_fn.loc45_4.1
// CHECK:STDOUT:     %.loc45_9.1: ref @Test.%impl.elem1.1 (%impl.elem1.092b54.2) = temporary_storage
// CHECK:STDOUT:     %CppRangeForIterate.WithSelf.End.call: init @Test.%impl.elem1.1 (%impl.elem1.092b54.2) to %.loc45_9.1 = call %bound_method.loc45_9.1(%r.ref.loc45)
// CHECK:STDOUT:     %.loc45_9.2: ref @Test.%impl.elem1.1 (%impl.elem1.092b54.2) = temporary %.loc45_9.1, %CppRangeForIterate.WithSelf.End.call
// CHECK:STDOUT:     %impl.elem0.loc45_9.1: @Test.%.loc45_9.3 (%.132) = impl_witness_access constants.%Destroy.lookup_impl_witness.aba, element0 [symbolic = %impl.elem0.loc45_9.2 (constants.%impl.elem0.74a)]
// CHECK:STDOUT:     %bound_method.loc45_9.2: <bound method> = bound_method %.loc45_9.2, %impl.elem0.loc45_9.1
// CHECK:STDOUT:     %specific_impl_fn.loc45_9.1: <specific function> = specific_impl_function %impl.elem0.loc45_9.1, @Destroy.WithSelf.Op(constants.%Destroy.facet.ba7) [symbolic = %specific_impl_fn.loc45_9.2 (constants.%specific_impl_fn.044)]
// CHECK:STDOUT:     %bound_method.loc45_9.3: <bound method> = bound_method %.loc45_9.2, %specific_impl_fn.loc45_9.1
// CHECK:STDOUT:     %Destroy.WithSelf.Op.call.loc45: init %empty_tuple.type = call %bound_method.loc45_9.3(%.loc45_9.2)
// CHECK:STDOUT:     %impl.elem0.loc44_11.1: @Test.%.loc44_11.3 (%.d6b) = impl_witness_access constants.%Destroy.lookup_impl_witness.cf0, element0 [symbolic = %impl.elem0.loc44_11.2 (constants.%impl.elem0.a98)]
// CHECK:STDOUT:     %bound_method.loc44_11.2: <bound method> = bound_method %.loc44_11.2, %impl.elem0.loc44_11.1
// CHECK:STDOUT:     %specific_impl_fn.loc44_11.1: <specific function> = specific_impl_function %impl.elem0.loc44_11.1, @Destroy.WithSelf.Op(constants.%Destroy.facet.0a9) [symbolic = %specific_impl_fn.loc44_11.2 (constants.%specific_impl_fn.168)]
// CHECK:STDOUT:     %bound_method.loc44_11.3: <bound method> = bound_method %.loc44_11.2, %specific_impl_fn.loc44_11.1
// CHECK:STDOUT:     %Destroy.WithSelf.Op.call.loc44: init %empty_tuple.type = call %bound_method.loc44_11.3(%.loc44_11.2)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(constants.%R) {
// CHECK:STDOUT:   %R.loc42_10.1 => constants.%R
// CHECK:STDOUT:   %R.as_type.loc42_113.1 => constants.%R.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.61273a.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(constants.%facet_value.a6b) {
// CHECK:STDOUT:   %R.loc42_10.1 => constants.%facet_value.a6b
// CHECK:STDOUT:   %R.as_type.loc42_113.1 => constants.%MutableRange
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.8be
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc42 => constants.%complete_type
// CHECK:STDOUT:   %CppRangeForIterate.lookup_impl_witness => constants.%custom_witness.4cf
// CHECK:STDOUT:   %CppRangeForIterate.facet => constants.%CppRangeForIterate.facet.517
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.Begin.type => constants.%CppRangeForIterate.WithSelf.Begin.type.80e
// CHECK:STDOUT:   %.loc44_4 => constants.%.3f3
// CHECK:STDOUT:   %impl.elem2.loc44_4.2 => constants.%Begin.b992d6.1
// CHECK:STDOUT:   %specific_impl_fn.loc44_4.2 => constants.%Begin.b992d6.1
// CHECK:STDOUT:   %impl.elem0.1 => constants.%Iterator.085
// CHECK:STDOUT:   %require_complete.1 => constants.%complete_type
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.End.type => constants.%CppRangeForIterate.WithSelf.End.type.93c
// CHECK:STDOUT:   %.loc45_4 => constants.%.21b
// CHECK:STDOUT:   %impl.elem3.loc45_4.2 => constants.%End.2270fc.1
// CHECK:STDOUT:   %specific_impl_fn.loc45_4.2 => constants.%End.2270fc.1
// CHECK:STDOUT:   %impl.elem1.1 => constants.%Sentinel.788
// CHECK:STDOUT:   %require_complete.2 => constants.%complete_type
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.loc45 => constants.%custom_witness.3af
// CHECK:STDOUT:   %Destroy.facet.loc45 => constants.%Destroy.facet.c1b
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.loc45 => constants.%Destroy.WithSelf.Op.type.835
// CHECK:STDOUT:   %.loc45_9.3 => constants.%.0cb
// CHECK:STDOUT:   %impl.elem0.loc45_9.2 => constants.%Sentinel.Op.f1e
// CHECK:STDOUT:   %specific_impl_fn.loc45_9.2 => constants.%Sentinel.Op.f1e
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.loc44 => constants.%custom_witness.3b6
// CHECK:STDOUT:   %Destroy.facet.loc44 => constants.%Destroy.facet.799
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.loc44 => constants.%Destroy.WithSelf.Op.type.2fa
// CHECK:STDOUT:   %.loc44_11.3 => constants.%.a50
// CHECK:STDOUT:   %impl.elem0.loc44_11.2 => constants.%Iterator.Op.88c
// CHECK:STDOUT:   %specific_impl_fn.loc44_11.2 => constants.%Iterator.Op.88c
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(constants.%facet_value.c5b) {
// CHECK:STDOUT:   %R.loc42_10.1 => constants.%facet_value.c5b
// CHECK:STDOUT:   %R.as_type.loc42_113.1 => constants.%ConstRange
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.8e2
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc42 => constants.%complete_type
// CHECK:STDOUT:   %CppRangeForIterate.lookup_impl_witness => constants.%custom_witness.47c
// CHECK:STDOUT:   %CppRangeForIterate.facet => constants.%CppRangeForIterate.facet.cba
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.Begin.type => constants.%CppRangeForIterate.WithSelf.Begin.type.ae7
// CHECK:STDOUT:   %.loc44_4 => constants.%.ace
// CHECK:STDOUT:   %impl.elem2.loc44_4.2 => constants.%Begin.b992d6.2
// CHECK:STDOUT:   %specific_impl_fn.loc44_4.2 => constants.%Begin.b992d6.2
// CHECK:STDOUT:   %impl.elem0.1 => constants.%Iterator.268
// CHECK:STDOUT:   %require_complete.1 => constants.%complete_type
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.End.type => constants.%CppRangeForIterate.WithSelf.End.type.2a0
// CHECK:STDOUT:   %.loc45_4 => constants.%.956
// CHECK:STDOUT:   %impl.elem3.loc45_4.2 => constants.%End.2270fc.2
// CHECK:STDOUT:   %specific_impl_fn.loc45_4.2 => constants.%End.2270fc.2
// CHECK:STDOUT:   %impl.elem1.1 => constants.%Sentinel.155
// CHECK:STDOUT:   %require_complete.2 => constants.%complete_type
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.loc45 => constants.%custom_witness.9cf
// CHECK:STDOUT:   %Destroy.facet.loc45 => constants.%Destroy.facet.f4f
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.loc45 => constants.%Destroy.WithSelf.Op.type.923
// CHECK:STDOUT:   %.loc45_9.3 => constants.%.2f2
// CHECK:STDOUT:   %impl.elem0.loc45_9.2 => constants.%Sentinel.Op.0bd
// CHECK:STDOUT:   %specific_impl_fn.loc45_9.2 => constants.%Sentinel.Op.0bd
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.loc44 => constants.%custom_witness.14d
// CHECK:STDOUT:   %Destroy.facet.loc44 => constants.%Destroy.facet.6f7
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.loc44 => constants.%Destroy.WithSelf.Op.type.b31
// CHECK:STDOUT:   %.loc44_11.3 => constants.%.09c
// CHECK:STDOUT:   %impl.elem0.loc44_11.2 => constants.%Iterator.Op.733
// CHECK:STDOUT:   %specific_impl_fn.loc44_11.2 => constants.%Iterator.Op.733
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- adl_overloads.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %CppRangeForIterate.type: type = facet_type <@CppRangeForIterate> [concrete]
// CHECK:STDOUT:   %Self.981: %CppRangeForIterate.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.End.type.5610e6.1: type = fn_type @CppRangeForIterate.WithSelf.End, @CppRangeForIterate.WithSelf(%Self.981) [symbolic]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.End.65a3e4.1: %CppRangeForIterate.WithSelf.End.type.5610e6.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.Begin.type.91eecf.1: type = fn_type @CppRangeForIterate.WithSelf.Begin, @CppRangeForIterate.WithSelf(%Self.981) [symbolic]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.Begin.07315f.1: %CppRangeForIterate.WithSelf.Begin.type.91eecf.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %.Self.756: %CppRangeForIterate.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %CppRangeForIterate.assoc_type: type = assoc_entity_type @CppRangeForIterate [concrete]
// CHECK:STDOUT:   %CppRangeForIterate.lookup_impl_witness.a19: <witness> = lookup_impl_witness %.Self.756, @CppRangeForIterate [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.5fa: type = impl_witness_access %CppRangeForIterate.lookup_impl_witness.a19, element0 [symbolic_self]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %impl.elem1.b14: type = impl_witness_access %CppRangeForIterate.lookup_impl_witness.a19, element1 [symbolic_self]
// CHECK:STDOUT:   %CppRangeForIterate_where.type: type = facet_type <@CppRangeForIterate where %impl.elem0.5fa impls @Destroy and %impl.elem1.b14 impls @Destroy> [symbolic_self]
// CHECK:STDOUT:   %R: %CppRangeForIterate_where.type = symbolic_binding R, 0 [symbolic]
// CHECK:STDOUT:   %R.as_type: type = facet_access_type %R [symbolic]
// CHECK:STDOUT:   %pattern_type.61273a.2: type = pattern_type %R.as_type [symbolic]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %assoc2: %CppRangeForIterate.assoc_type = assoc_entity element2, imports.%Core.import_ref.a11 [concrete]
// CHECK:STDOUT:   %CppRangeForIterate.lookup_impl_witness.20309f.2: <witness> = lookup_impl_witness %R, @CppRangeForIterate [symbolic]
// CHECK:STDOUT:   %CppRangeForIterate.facet.ee6: %CppRangeForIterate.type = facet_value %R.as_type, (%CppRangeForIterate.lookup_impl_witness.20309f.2) [symbolic]
// CHECK:STDOUT:   %impl.elem0.8e47d6.2: type = impl_witness_access %CppRangeForIterate.lookup_impl_witness.20309f.2, element0 [symbolic]
// CHECK:STDOUT:   %impl.elem1.092b54.2: type = impl_witness_access %CppRangeForIterate.lookup_impl_witness.20309f.2, element1 [symbolic]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.Begin.type.71b: type = fn_type @CppRangeForIterate.WithSelf.Begin, @CppRangeForIterate.WithSelf(%CppRangeForIterate.facet.ee6) [symbolic]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.End.type.c1b: type = fn_type @CppRangeForIterate.WithSelf.End, @CppRangeForIterate.WithSelf(%CppRangeForIterate.facet.ee6) [symbolic]
// CHECK:STDOUT:   %.aa6: type = fn_type_with_self_type %CppRangeForIterate.WithSelf.Begin.type.71b, %CppRangeForIterate.facet.ee6 [symbolic]
// CHECK:STDOUT:   %impl.elem2: %.aa6 = impl_witness_access %CppRangeForIterate.lookup_impl_witness.20309f.2, element2 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.987: <specific function> = specific_impl_function %impl.elem2, @CppRangeForIterate.WithSelf.Begin(%CppRangeForIterate.facet.ee6) [symbolic]
// CHECK:STDOUT:   %assoc3: %CppRangeForIterate.assoc_type = assoc_entity element3, imports.%Core.import_ref.031 [concrete]
// CHECK:STDOUT:   %.5ea: type = fn_type_with_self_type %CppRangeForIterate.WithSelf.End.type.c1b, %CppRangeForIterate.facet.ee6 [symbolic]
// CHECK:STDOUT:   %impl.elem3: %.5ea = impl_witness_access %CppRangeForIterate.lookup_impl_witness.20309f.2, element3 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.748: <specific function> = specific_impl_function %impl.elem3, @CppRangeForIterate.WithSelf.End(%CppRangeForIterate.facet.ee6) [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.aba: <witness> = lookup_impl_witness %impl.elem1.092b54.2, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.ba7: %Destroy.type = facet_value %impl.elem1.092b54.2, (%Destroy.lookup_impl_witness.aba) [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.030: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.ba7) [symbolic]
// CHECK:STDOUT:   %.132: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.030, %Destroy.facet.ba7 [symbolic]
// CHECK:STDOUT:   %impl.elem0.74a: %.132 = impl_witness_access %Destroy.lookup_impl_witness.aba, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.044: <specific function> = specific_impl_function %impl.elem0.74a, @Destroy.WithSelf.Op(%Destroy.facet.ba7) [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.cf0: <witness> = lookup_impl_witness %impl.elem0.8e47d6.2, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.0a9: %Destroy.type = facet_value %impl.elem0.8e47d6.2, (%Destroy.lookup_impl_witness.cf0) [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.b91: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.0a9) [symbolic]
// CHECK:STDOUT:   %.d6b: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.b91, %Destroy.facet.0a9 [symbolic]
// CHECK:STDOUT:   %impl.elem0.a98: %.d6b = impl_witness_access %Destroy.lookup_impl_witness.cf0, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.168: <specific function> = specific_impl_function %impl.elem0.a98, @Destroy.WithSelf.Op(%Destroy.facet.0a9) [symbolic]
// CHECK:STDOUT:   %MutableAndConstRange: type = class_type @MutableAndConstRange [concrete]
// CHECK:STDOUT:   %pattern_type.68f: type = pattern_type %MutableAndConstRange [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %Iterator: type = class_type @Iterator.1 [concrete]
// CHECK:STDOUT:   %Begin.type: type = fn_type @Begin [concrete]
// CHECK:STDOUT:   %Begin: %Begin.type = struct_value () [concrete]
// CHECK:STDOUT:   %End.type: type = fn_type @End [concrete]
// CHECK:STDOUT:   %End: %End.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.52d: <witness> = custom_witness (%Iterator, %Iterator, %Begin, %End), @CppRangeForIterate [concrete]
// CHECK:STDOUT:   %CppRangeForIterate.facet.fcd: %CppRangeForIterate.type = facet_value %MutableAndConstRange, (%custom_witness.52d) [concrete]
// CHECK:STDOUT:   %Iterator.Op.type: type = fn_type @Iterator.Op [concrete]
// CHECK:STDOUT:   %Iterator.Op: %Iterator.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.0d5: <witness> = custom_witness (%Iterator.Op), @Destroy [concrete]
// CHECK:STDOUT:   %facet_value: %CppRangeForIterate_where.type = facet_value %MutableAndConstRange, (%custom_witness.52d, %custom_witness.0d5) [symbolic_self]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.Begin.type.672: type = fn_type @CppRangeForIterate.WithSelf.Begin, @CppRangeForIterate.WithSelf(%CppRangeForIterate.facet.fcd) [concrete]
// CHECK:STDOUT:   %.d23: type = fn_type_with_self_type %CppRangeForIterate.WithSelf.Begin.type.672, %CppRangeForIterate.facet.fcd [concrete]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.End.type.bd4: type = fn_type @CppRangeForIterate.WithSelf.End, @CppRangeForIterate.WithSelf(%CppRangeForIterate.facet.fcd) [concrete]
// CHECK:STDOUT:   %.fbf: type = fn_type_with_self_type %CppRangeForIterate.WithSelf.End.type.bd4, %CppRangeForIterate.facet.fcd [concrete]
// CHECK:STDOUT:   %Destroy.facet.509: %Destroy.type = facet_value %Iterator, (%custom_witness.0d5) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.fa2: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.509) [concrete]
// CHECK:STDOUT:   %.eec: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.fa2, %Destroy.facet.509 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.822: %CppRangeForIterate.assoc_type = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [concrete = constants.%assoc2]
// CHECK:STDOUT:   %Core.import_ref.685: %CppRangeForIterate.assoc_type = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [concrete = constants.%assoc3]
// CHECK:STDOUT:   %Core.import_ref.a11: @CppRangeForIterate.WithSelf.%CppRangeForIterate.WithSelf.Begin.type (%CppRangeForIterate.WithSelf.Begin.type.91eecf.1) = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @CppRangeForIterate.WithSelf.%CppRangeForIterate.WithSelf.Begin (constants.%CppRangeForIterate.WithSelf.Begin.07315f.1)]
// CHECK:STDOUT:   %Core.import_ref.031: @CppRangeForIterate.WithSelf.%CppRangeForIterate.WithSelf.End.type (%CppRangeForIterate.WithSelf.End.type.5610e6.1) = import_ref Core//prelude/iterate, loc{{\d+_\d+}}, loaded [symbolic = @CppRangeForIterate.WithSelf.%CppRangeForIterate.WithSelf.End (constants.%CppRangeForIterate.WithSelf.End.65a3e4.1)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Test(%R.loc32_10.2: %CppRangeForIterate_where.type) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %CppRangeForIterate.lookup_impl_witness: <witness> = lookup_impl_witness %R.loc32_10.1, @CppRangeForIterate [symbolic = %CppRangeForIterate.lookup_impl_witness (constants.%CppRangeForIterate.lookup_impl_witness.20309f.2)]
// CHECK:STDOUT:   %CppRangeForIterate.facet: %CppRangeForIterate.type = facet_value %R.as_type.loc32_113.1, (%CppRangeForIterate.lookup_impl_witness) [symbolic = %CppRangeForIterate.facet (constants.%CppRangeForIterate.facet.ee6)]
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.Begin.type: type = fn_type @CppRangeForIterate.WithSelf.Begin, @CppRangeForIterate.WithSelf(%CppRangeForIterate.facet) [symbolic = %CppRangeForIterate.WithSelf.Begin.type (constants.%CppRangeForIterate.WithSelf.Begin.type.71b)]
// CHECK:STDOUT:   %.loc34_4: type = fn_type_with_self_type %CppRangeForIterate.WithSelf.Begin.type, %CppRangeForIterate.facet [symbolic = %.loc34_4 (constants.%.aa6)]
// CHECK:STDOUT:   %impl.elem2.loc34_4.2: @Test.%.loc34_4 (%.aa6) = impl_witness_access %CppRangeForIterate.lookup_impl_witness, element2 [symbolic = %impl.elem2.loc34_4.2 (constants.%impl.elem2)]
// CHECK:STDOUT:   %specific_impl_fn.loc34_4.2: <specific function> = specific_impl_function %impl.elem2.loc34_4.2, @CppRangeForIterate.WithSelf.Begin(%CppRangeForIterate.facet) [symbolic = %specific_impl_fn.loc34_4.2 (constants.%specific_impl_fn.987)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.End.type: type = fn_type @CppRangeForIterate.WithSelf.End, @CppRangeForIterate.WithSelf(%CppRangeForIterate.facet) [symbolic = %CppRangeForIterate.WithSelf.End.type (constants.%CppRangeForIterate.WithSelf.End.type.c1b)]
// CHECK:STDOUT:   %.loc35_4: type = fn_type_with_self_type %CppRangeForIterate.WithSelf.End.type, %CppRangeForIterate.facet [symbolic = %.loc35_4 (constants.%.5ea)]
// CHECK:STDOUT:   %impl.elem3.loc35_4.2: @Test.%.loc35_4 (%.5ea) = impl_witness_access %CppRangeForIterate.lookup_impl_witness, element3 [symbolic = %impl.elem3.loc35_4.2 (constants.%impl.elem3)]
// CHECK:STDOUT:   %specific_impl_fn.loc35_4.2: <specific function> = specific_impl_function %impl.elem3.loc35_4.2, @CppRangeForIterate.WithSelf.End(%CppRangeForIterate.facet) [symbolic = %specific_impl_fn.loc35_4.2 (constants.%specific_impl_fn.748)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.loc35: <witness> = lookup_impl_witness %impl.elem1.1, @Destroy [symbolic = %Destroy.lookup_impl_witness.loc35 (constants.%Destroy.lookup_impl_witness.aba)]
// CHECK:STDOUT:   %Destroy.facet.loc35: %Destroy.type = facet_value %impl.elem1.1, (%Destroy.lookup_impl_witness.loc35) [symbolic = %Destroy.facet.loc35 (constants.%Destroy.facet.ba7)]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.loc35: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.loc35) [symbolic = %Destroy.WithSelf.Op.type.loc35 (constants.%Destroy.WithSelf.Op.type.030)]
// CHECK:STDOUT:   %.loc35_9.3: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.loc35, %Destroy.facet.loc35 [symbolic = %.loc35_9.3 (constants.%.132)]
// CHECK:STDOUT:   %impl.elem0.loc35_9.2: @Test.%.loc35_9.3 (%.132) = impl_witness_access %Destroy.lookup_impl_witness.loc35, element0 [symbolic = %impl.elem0.loc35_9.2 (constants.%impl.elem0.74a)]
// CHECK:STDOUT:   %specific_impl_fn.loc35_9.2: <specific function> = specific_impl_function %impl.elem0.loc35_9.2, @Destroy.WithSelf.Op(%Destroy.facet.loc35) [symbolic = %specific_impl_fn.loc35_9.2 (constants.%specific_impl_fn.044)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.loc34: <witness> = lookup_impl_witness %impl.elem0.1, @Destroy [symbolic = %Destroy.lookup_impl_witness.loc34 (constants.%Destroy.lookup_impl_witness.cf0)]
// CHECK:STDOUT:   %Destroy.facet.loc34: %Destroy.type = facet_value %impl.elem0.1, (%Destroy.lookup_impl_witness.loc34) [symbolic = %Destroy.facet.loc34 (constants.%Destroy.facet.0a9)]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.loc34: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.loc34) [symbolic = %Destroy.WithSelf.Op.type.loc34 (constants.%Destroy.WithSelf.Op.type.b91)]
// CHECK:STDOUT:   %.loc34_11.3: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.loc34, %Destroy.facet.loc34 [symbolic = %.loc34_11.3 (constants.%.d6b)]
// CHECK:STDOUT:   %impl.elem0.loc34_11.2: @Test.%.loc34_11.3 (%.d6b) = impl_witness_access %Destroy.lookup_impl_witness.loc34, element0 [symbolic = %impl.elem0.loc34_11.2 (constants.%impl.elem0.a98)]
// CHECK:STDOUT:   %specific_impl_fn.loc34_11.2: <specific function> = specific_impl_function %impl.elem0.loc34_11.2, @Destroy.WithSelf.Op(%Destroy.facet.loc34) [symbolic = %specific_impl_fn.loc34_11.2 (constants.%specific_impl_fn.168)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%r.param: ref @Test.%R.as_type.loc32_113.1 (%R.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %r.ref.loc34: ref @Test.%R.as_type.loc32_113.1 (%R.as_type) = name_ref r, %r
// CHECK:STDOUT:     %Begin.ref: %CppRangeForIterate.assoc_type = name_ref Begin, imports.%Core.import_ref.822 [concrete = constants.%assoc2]
// CHECK:STDOUT:     %impl.elem2.loc34_4.1: @Test.%.loc34_4 (%.aa6) = impl_witness_access constants.%CppRangeForIterate.lookup_impl_witness.20309f.2, element2 [symbolic = %impl.elem2.loc34_4.2 (constants.%impl.elem2)]
// CHECK:STDOUT:     %bound_method.loc34_4: <bound method> = bound_method %r.ref.loc34, %impl.elem2.loc34_4.1
// CHECK:STDOUT:     %specific_impl_fn.loc34_4.1: <specific function> = specific_impl_function %impl.elem2.loc34_4.1, @CppRangeForIterate.WithSelf.Begin(constants.%CppRangeForIterate.facet.ee6) [symbolic = %specific_impl_fn.loc34_4.2 (constants.%specific_impl_fn.987)]
// CHECK:STDOUT:     %bound_method.loc34_11.1: <bound method> = bound_method %r.ref.loc34, %specific_impl_fn.loc34_4.1
// CHECK:STDOUT:     %.loc34_11.1: ref @Test.%impl.elem0.1 (%impl.elem0.8e47d6.2) = temporary_storage
// CHECK:STDOUT:     %CppRangeForIterate.WithSelf.Begin.call: init @Test.%impl.elem0.1 (%impl.elem0.8e47d6.2) to %.loc34_11.1 = call %bound_method.loc34_11.1(%r.ref.loc34)
// CHECK:STDOUT:     %.loc34_11.2: ref @Test.%impl.elem0.1 (%impl.elem0.8e47d6.2) = temporary %.loc34_11.1, %CppRangeForIterate.WithSelf.Begin.call
// CHECK:STDOUT:     %r.ref.loc35: ref @Test.%R.as_type.loc32_113.1 (%R.as_type) = name_ref r, %r
// CHECK:STDOUT:     %End.ref: %CppRangeForIterate.assoc_type = name_ref End, imports.%Core.import_ref.685 [concrete = constants.%assoc3]
// CHECK:STDOUT:     %impl.elem3.loc35_4.1: @Test.%.loc35_4 (%.5ea) = impl_witness_access constants.%CppRangeForIterate.lookup_impl_witness.20309f.2, element3 [symbolic = %impl.elem3.loc35_4.2 (constants.%impl.elem3)]
// CHECK:STDOUT:     %bound_method.loc35_4: <bound method> = bound_method %r.ref.loc35, %impl.elem3.loc35_4.1
// CHECK:STDOUT:     %specific_impl_fn.loc35_4.1: <specific function> = specific_impl_function %impl.elem3.loc35_4.1, @CppRangeForIterate.WithSelf.End(constants.%CppRangeForIterate.facet.ee6) [symbolic = %specific_impl_fn.loc35_4.2 (constants.%specific_impl_fn.748)]
// CHECK:STDOUT:     %bound_method.loc35_9.1: <bound method> = bound_method %r.ref.loc35, %specific_impl_fn.loc35_4.1
// CHECK:STDOUT:     %.loc35_9.1: ref @Test.%impl.elem1.1 (%impl.elem1.092b54.2) = temporary_storage
// CHECK:STDOUT:     %CppRangeForIterate.WithSelf.End.call: init @Test.%impl.elem1.1 (%impl.elem1.092b54.2) to %.loc35_9.1 = call %bound_method.loc35_9.1(%r.ref.loc35)
// CHECK:STDOUT:     %.loc35_9.2: ref @Test.%impl.elem1.1 (%impl.elem1.092b54.2) = temporary %.loc35_9.1, %CppRangeForIterate.WithSelf.End.call
// CHECK:STDOUT:     %impl.elem0.loc35_9.1: @Test.%.loc35_9.3 (%.132) = impl_witness_access constants.%Destroy.lookup_impl_witness.aba, element0 [symbolic = %impl.elem0.loc35_9.2 (constants.%impl.elem0.74a)]
// CHECK:STDOUT:     %bound_method.loc35_9.2: <bound method> = bound_method %.loc35_9.2, %impl.elem0.loc35_9.1
// CHECK:STDOUT:     %specific_impl_fn.loc35_9.1: <specific function> = specific_impl_function %impl.elem0.loc35_9.1, @Destroy.WithSelf.Op(constants.%Destroy.facet.ba7) [symbolic = %specific_impl_fn.loc35_9.2 (constants.%specific_impl_fn.044)]
// CHECK:STDOUT:     %bound_method.loc35_9.3: <bound method> = bound_method %.loc35_9.2, %specific_impl_fn.loc35_9.1
// CHECK:STDOUT:     %Destroy.WithSelf.Op.call.loc35: init %empty_tuple.type = call %bound_method.loc35_9.3(%.loc35_9.2)
// CHECK:STDOUT:     %impl.elem0.loc34_11.1: @Test.%.loc34_11.3 (%.d6b) = impl_witness_access constants.%Destroy.lookup_impl_witness.cf0, element0 [symbolic = %impl.elem0.loc34_11.2 (constants.%impl.elem0.a98)]
// CHECK:STDOUT:     %bound_method.loc34_11.2: <bound method> = bound_method %.loc34_11.2, %impl.elem0.loc34_11.1
// CHECK:STDOUT:     %specific_impl_fn.loc34_11.1: <specific function> = specific_impl_function %impl.elem0.loc34_11.1, @Destroy.WithSelf.Op(constants.%Destroy.facet.0a9) [symbolic = %specific_impl_fn.loc34_11.2 (constants.%specific_impl_fn.168)]
// CHECK:STDOUT:     %bound_method.loc34_11.3: <bound method> = bound_method %.loc34_11.2, %specific_impl_fn.loc34_11.1
// CHECK:STDOUT:     %Destroy.WithSelf.Op.call.loc34: init %empty_tuple.type = call %bound_method.loc34_11.3(%.loc34_11.2)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(constants.%R) {
// CHECK:STDOUT:   %R.loc32_10.1 => constants.%R
// CHECK:STDOUT:   %R.as_type.loc32_113.1 => constants.%R.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.61273a.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(constants.%facet_value) {
// CHECK:STDOUT:   %R.loc32_10.1 => constants.%facet_value
// CHECK:STDOUT:   %R.as_type.loc32_113.1 => constants.%MutableAndConstRange
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.68f
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc32 => constants.%complete_type
// CHECK:STDOUT:   %CppRangeForIterate.lookup_impl_witness => constants.%custom_witness.52d
// CHECK:STDOUT:   %CppRangeForIterate.facet => constants.%CppRangeForIterate.facet.fcd
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.Begin.type => constants.%CppRangeForIterate.WithSelf.Begin.type.672
// CHECK:STDOUT:   %.loc34_4 => constants.%.d23
// CHECK:STDOUT:   %impl.elem2.loc34_4.2 => constants.%Begin
// CHECK:STDOUT:   %specific_impl_fn.loc34_4.2 => constants.%Begin
// CHECK:STDOUT:   %impl.elem0.1 => constants.%Iterator
// CHECK:STDOUT:   %require_complete.1 => constants.%complete_type
// CHECK:STDOUT:   %CppRangeForIterate.WithSelf.End.type => constants.%CppRangeForIterate.WithSelf.End.type.bd4
// CHECK:STDOUT:   %.loc35_4 => constants.%.fbf
// CHECK:STDOUT:   %impl.elem3.loc35_4.2 => constants.%End
// CHECK:STDOUT:   %specific_impl_fn.loc35_4.2 => constants.%End
// CHECK:STDOUT:   %impl.elem1.1 => constants.%Iterator
// CHECK:STDOUT:   %require_complete.2 => constants.%complete_type
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.loc35 => constants.%custom_witness.0d5
// CHECK:STDOUT:   %Destroy.facet.loc35 => constants.%Destroy.facet.509
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.loc35 => constants.%Destroy.WithSelf.Op.type.fa2
// CHECK:STDOUT:   %.loc35_9.3 => constants.%.eec
// CHECK:STDOUT:   %impl.elem0.loc35_9.2 => constants.%Iterator.Op
// CHECK:STDOUT:   %specific_impl_fn.loc35_9.2 => constants.%Iterator.Op
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.loc34 => constants.%custom_witness.0d5
// CHECK:STDOUT:   %Destroy.facet.loc34 => constants.%Destroy.facet.509
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.loc34 => constants.%Destroy.WithSelf.Op.type.fa2
// CHECK:STDOUT:   %.loc34_11.3 => constants.%.eec
// CHECK:STDOUT:   %impl.elem0.loc34_11.2 => constants.%Iterator.Op
// CHECK:STDOUT:   %specific_impl_fn.loc34_11.2 => constants.%Iterator.Op
// CHECK:STDOUT: }
// CHECK:STDOUT:
