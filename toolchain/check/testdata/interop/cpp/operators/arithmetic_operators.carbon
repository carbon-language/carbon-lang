// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
// EXTRA-ARGS: --target=x86_64-linux-gnu --clang-arg=-std=c++20
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/operators/arithmetic_operators.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/operators/arithmetic_operators.carbon

// TODO: test implicit conversions when supported

library "[[@TEST_NAME]]";
import Cpp inline '''c++
class Int16;
class Int32;
class Int64;

class Int16 {
public:
  // TODO: support non-const qualified operators
  auto operator+(Int32) const -> Int32;
  auto operator-(Int32) const -> Int32;
  auto operator*(Int32) const -> Int32;
  auto operator/(Int32) const -> Int32;
  auto operator%(Int32) const -> Int32;

  // Tests `class + builtin` works
  auto operator+(short) const -> Int16;
  auto operator-(short) const -> Int16;
  auto operator*(short) const -> Int16;
  auto operator/(short) const -> Int16;
  auto operator%(short) const -> Int16;

  auto operator+=(short) const -> Int16&;
  auto operator-=(short) const -> Int16&;
  auto operator*=(short) const -> Int16&;
  auto operator/=(short) const -> Int16&;
  auto operator%=(short) const -> Int16&;

  auto operator-() const -> Int64;
};

auto operator+(short, Int16) -> Int16;
auto operator-(short, Int16) -> Int16;
auto operator*(short, Int16) -> Int16;
auto operator/(short, Int16) -> Int16;
auto operator%(short, Int16) -> Int16;

class Int32 {
public:
  // Intentionally not const-qualified
  auto operator+(Int16) const -> Int32;
  auto operator-(Int16) const -> Int32;
  auto operator*(Int16) const -> Int32;
  auto operator/(Int16) const -> Int32;
  auto operator%(Int16) const -> Int32;

  auto operator+(Int32) const -> Int32;
  auto operator-(Int32) const -> Int32;
  auto operator*(Int32) const -> Int32;
  auto operator/(Int32) const -> Int32;
  auto operator%(Int32) const -> Int32;

  auto operator-() const -> Int32;

  auto operator++() -> void;
  auto operator--() -> void;

  auto operator+=(Int32) -> void;
  auto operator-=(Int32) -> void;
  auto operator*=(Int32) -> void;
  auto operator/=(Int32) -> void;
  auto operator%=(Int32) -> void;
};

class Int64;
auto operator+(Int64, Int32) -> Int64;
auto operator-(Int64, Int32) -> Int64;
auto operator*(Int64, Int32) -> Int64;
auto operator/(Int64, Int32) -> Int64;
auto operator%(Int64, Int32) -> Int64;

class Int64 {
public:
  auto operator+=(Int32) -> void;
  auto operator-=(Int32) -> void;
  auto operator*=(Int32) -> void;
  auto operator/=(Int32) -> void;
  auto operator%=(Int32) -> void;

  // Tests hidden friends work
  friend auto operator+(Int32 x, Int64 y) -> Int64 {
    return y + x;
  }
  friend auto operator-(Int32 x, Int64 y) -> Int64 {
    return y + x;
  }
  friend auto operator*(Int32 x, Int64 y) -> Int64 {
    return y + x;
  }
  friend auto operator/(Int32 x, Int64 y) -> Int64 {
    return y + x;
  }
  friend auto operator%(Int32 x, Int64 y) -> Int64 {
    return y + x;
  }
};
''';

// TODO: Refactor when `.Result impls Core.Destroy` is possible.
// `Result` is a workaround for `.Result impls Core.Destroy`. This funciton
// ought to be defined as
// ```
// AddWith[U:! type, T:! Core.AddWith(U) where .Result impls Core.Destroy](x: T, y: T)
// ```
fn AddWith[
  Result:! Core.Destroy,
  U:! Core.Destroy,
  T:! Core.Destroy & Core.AddWith(U) where .Result = Result
](unused result: Result, x: T, y: U) {
  //@dump-sem-ir-begin
  x + y;
  //@dump-sem-ir-end
}

// TODO: Refactor when `.Result impls Core.Destroy` is possible.
fn SubWith[
  Result:! Core.Destroy,
  U:! Core.Destroy,
  T:! Core.Destroy & Core.SubWith(U) where .Result = Result
](unused result: Result, x: T, y: U) {
  //@dump-sem-ir-begin
  x - y;
  //@dump-sem-ir-end
}

// TODO: Refactor when `.Result impls Core.Destroy` is possible.
fn MulWith[
  Result:! Core.Destroy,
  U:! Core.Destroy,
  T:! Core.Destroy & Core.MulWith(U) where .Result = Result
](unused result: Result, x: T, y: U) {
  //@dump-sem-ir-begin
  x * y;
  //@dump-sem-ir-end
}

// TODO: Refactor when `.Result impls Core.Destroy` is possible.
fn DivWith[
  Result:! Core.Destroy,
  U:! Core.Destroy,
  T:! Core.Destroy & Core.DivWith(U) where .Result = Result
](unused result: Result, x: T, y: U) {
  //@dump-sem-ir-begin
  x / y;
  //@dump-sem-ir-end
}

// TODO: Refactor when `.Result impls Core.Destroy` is possible.
fn ModWith[
  Result:! Core.Destroy,
  U:! Core.Destroy,
  T:! Core.Destroy & Core.ModWith(U) where .Result = Result
](unused result: Result, x: T, y: U) {
  //@dump-sem-ir-begin
  x % y;
  //@dump-sem-ir-end
}

fn TestOpWith(a: Cpp.Int16, b: Cpp.Int32, c: Cpp.Int32, d: Cpp.Int64, x: i16) {
  // i16 `op` Int16
  AddWith(Cpp.Int16.Int16(), x, a);
  SubWith(Cpp.Int16.Int16(), x, a);
  MulWith(Cpp.Int16.Int16(), x, a);
  DivWith(Cpp.Int16.Int16(), x, a);
  ModWith(Cpp.Int16.Int16(), x, a);

  // Int16 `op` i16
  AddWith(Cpp.Int16.Int16(), a, x);
  SubWith(Cpp.Int16.Int16(), a, x);
  MulWith(Cpp.Int16.Int16(), a, x);
  DivWith(Cpp.Int16.Int16(), a, x);
  ModWith(Cpp.Int16.Int16(), a, x);

  // Int16 `op` Int32
  AddWith(Cpp.Int32.Int32(), a, b);
  SubWith(Cpp.Int32.Int32(), a, b);
  MulWith(Cpp.Int32.Int32(), a, b);
  DivWith(Cpp.Int32.Int32(), a, b);
  ModWith(Cpp.Int32.Int32(), a, b);

  // Int32 `op` Int16
  AddWith(Cpp.Int32.Int32(), b, a);
  SubWith(Cpp.Int32.Int32(), b, a);
  MulWith(Cpp.Int32.Int32(), b, a);
  DivWith(Cpp.Int32.Int32(), b, a);
  ModWith(Cpp.Int32.Int32(), b, a);

  // Int32 `op` Int32
  AddWith(Cpp.Int32.Int32(), b, c);
  SubWith(Cpp.Int32.Int32(), b, c);
  MulWith(Cpp.Int32.Int32(), b, c);
  DivWith(Cpp.Int32.Int32(), b, c);
  ModWith(Cpp.Int32.Int32(), b, c);

  // Int32 `op` Int64
  AddWith(Cpp.Int64.Int64(), c, d);
  SubWith(Cpp.Int64.Int64(), c, d);
  MulWith(Cpp.Int64.Int64(), c, d);
  DivWith(Cpp.Int64.Int64(), c, d);
  ModWith(Cpp.Int64.Int64(), c, d);

  // Int64 `op` Int32
  AddWith(Cpp.Int64.Int64(), d, c);
  SubWith(Cpp.Int64.Int64(), d, c);
  MulWith(Cpp.Int64.Int64(), d, c);
  DivWith(Cpp.Int64.Int64(), d, c);
  ModWith(Cpp.Int64.Int64(), d, c);
}

fn AddAssignWith[U:! type, T:! Core.AddAssignWith(U)](var x: T, y: U) {
  //@dump-sem-ir-begin
  x += y;
  //@dump-sem-ir-end
}

fn SubAssignWith[U:! type, T:! Core.SubAssignWith(U)](var x: T, y: U) {
  //@dump-sem-ir-begin
  x -= y;
  //@dump-sem-ir-end
}

fn MulAssignWith[U:! type, T:! Core.MulAssignWith(U)](var x: T, y: U) {
  //@dump-sem-ir-begin
  x *= y;
  //@dump-sem-ir-end
}

fn DivAssignWith[U:! type, T:! Core.DivAssignWith(U)](var x: T, y: U) {
  //@dump-sem-ir-begin
  x /= y;
  //@dump-sem-ir-end
}

fn ModAssignWith[U:! type, T:! Core.ModAssignWith(U)](var x: T, y: U) {
  //@dump-sem-ir-begin
  x %= y;
  //@dump-sem-ir-end
}

fn TestOpAssignWith(int16: Cpp.Int16, int32: Cpp.Int32, int64: Cpp.Int64, x: i16) {
  // Int16 `op` i16
  AddAssignWith(int16, x);
  SubAssignWith(int16, x);
  MulAssignWith(int16, x);
  DivAssignWith(int16, x);
  ModAssignWith(int16, x);

  // Int32 `op` Int32
  AddAssignWith(int32, int32);
  SubAssignWith(int32, int32);
  MulAssignWith(int32, int32);
  DivAssignWith(int32, int32);
  ModAssignWith(int32, int32);

  // Int64 `op` Int32
  AddAssignWith(int64, int32);
  SubAssignWith(int64, int32);
  MulAssignWith(int64, int32);
  DivAssignWith(int64, int32);
  ModAssignWith(int64, int32);
}

fn TestInc[T:! Core.Inc](var x: T) {
  //@dump-sem-ir-begin
  ++x;
  //@dump-sem-ir-end
}

fn TestDec[T:! Core.Dec](var x: T) {
  //@dump-sem-ir-begin
  --x;
  //@dump-sem-ir-end
}

// TODO: Refactor when `.Result impls Core.Destroy` is possible.
fn TestNegate[
  Result:! Core.Destroy,
  T:! Core.Destroy & Core.Negate where .Result = Result
](unused y: Result, x: T) {
  //@dump-sem-ir-begin
  -x;
  //@dump-sem-ir-end
}

fn TestUnaryOperators(a: Cpp.Int16, b: Cpp.Int32) {
  TestInc(b);
  TestDec(b);

  TestNegate(Cpp.Int64.Int64(), a);
  TestNegate(Cpp.Int32.Int32(), b);
}

// CHECK:STDOUT: --- arithmetic_operators.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Result: %Destroy.type = symbolic_binding Result, 0 [symbolic]
// CHECK:STDOUT:   %U.0bd: %Destroy.type = symbolic_binding U, 1 [symbolic]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Other: type = symbolic_binding Other, 0 [symbolic]
// CHECK:STDOUT:   %AddWith.type.6d9: type = facet_type <@AddWith.1, @AddWith.1(%Other)> [symbolic]
// CHECK:STDOUT:   %Self.b7c: %AddWith.type.6d9 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %AddWith.assoc_type.b6a: type = assoc_entity_type @AddWith.1, @AddWith.1(%Other) [symbolic]
// CHECK:STDOUT:   %AddWith.WithSelf.Op.type.08f: type = fn_type @AddWith.WithSelf.Op, @AddWith.WithSelf(%Other, %Self.b7c) [symbolic]
// CHECK:STDOUT:   %AddWith.WithSelf.Op.fb8: %AddWith.WithSelf.Op.type.08f = struct_value () [symbolic]
// CHECK:STDOUT:   %U.as_type.2b8: type = facet_access_type %U.0bd [symbolic]
// CHECK:STDOUT:   %AddWith.type.07d: type = facet_type <@AddWith.1, @AddWith.1(%U.as_type.2b8)> [symbolic]
// CHECK:STDOUT:   %facet_type.74d: type = facet_type <@Destroy & @AddWith.1, @AddWith.1(%U.as_type.2b8)> [symbolic]
// CHECK:STDOUT:   %.Self.5c0: %facet_type.74d = symbolic_binding .Self [symbolic]
// CHECK:STDOUT:   %AddWith.assoc_type.8c2: type = assoc_entity_type @AddWith.1, @AddWith.1(%U.as_type.2b8) [symbolic]
// CHECK:STDOUT:   %assoc0.7d3: %AddWith.assoc_type.8c2 = assoc_entity element0, imports.%Core.import_ref.6e9 [symbolic]
// CHECK:STDOUT:   %assoc1.0f4: %AddWith.assoc_type.8c2 = assoc_entity element1, imports.%Core.import_ref.1b1 [symbolic]
// CHECK:STDOUT:   %require_complete.311: <witness> = require_complete_type %facet_type.74d [symbolic]
// CHECK:STDOUT:   %.Self.as_type.f21: type = facet_access_type %.Self.5c0 [symbolic]
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.5a0: <witness> = lookup_impl_witness %.Self.5c0, @AddWith.1, @AddWith.1(%U.as_type.2b8) [symbolic]
// CHECK:STDOUT:   %impl.elem0.08a: type = impl_witness_access %AddWith.lookup_impl_witness.5a0, element0 [symbolic]
// CHECK:STDOUT:   %Result.as_type: type = facet_access_type %Result [symbolic]
// CHECK:STDOUT:   %facet_type.024: type = facet_type <@Destroy & @AddWith.1, @AddWith.1(%U.as_type.2b8) where %impl.elem0.08a = %Result.as_type> [symbolic]
// CHECK:STDOUT:   %pattern_type.077a: type = pattern_type %facet_type.024 [symbolic]
// CHECK:STDOUT:   %T.25f: %facet_type.024 = symbolic_binding T, 2 [symbolic]
// CHECK:STDOUT:   %pattern_type.eca0e4.2: type = pattern_type %Result.as_type [symbolic]
// CHECK:STDOUT:   %T.as_type.ab9: type = facet_access_type %T.25f [symbolic]
// CHECK:STDOUT:   %pattern_type.7bb: type = pattern_type %T.as_type.ab9 [symbolic]
// CHECK:STDOUT:   %pattern_type.711: type = pattern_type %U.as_type.2b8 [symbolic]
// CHECK:STDOUT:   %require_complete.8bd: <witness> = require_complete_type %AddWith.type.07d [symbolic]
// CHECK:STDOUT:   %assoc1.4d4: %AddWith.assoc_type.b6a = assoc_entity element1, imports.%Core.import_ref.a84 [symbolic]
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.3c7: <witness> = lookup_impl_witness %T.25f, @AddWith.1, @AddWith.1(%U.as_type.2b8) [symbolic]
// CHECK:STDOUT:   %AddWith.facet.570: %AddWith.type.07d = facet_value %T.as_type.ab9, (%AddWith.lookup_impl_witness.3c7) [symbolic]
// CHECK:STDOUT:   %AddWith.WithSelf.Op.type.82f: type = fn_type @AddWith.WithSelf.Op, @AddWith.WithSelf(%U.as_type.2b8, %AddWith.facet.570) [symbolic]
// CHECK:STDOUT:   %.97d: type = fn_type_with_self_type %AddWith.WithSelf.Op.type.82f, %AddWith.facet.570 [symbolic]
// CHECK:STDOUT:   %impl.elem1.07a: %.97d = impl_witness_access %AddWith.lookup_impl_witness.3c7, element1 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.dbc: <specific function> = specific_impl_function %impl.elem1.07a, @AddWith.WithSelf.Op(%U.as_type.2b8, %AddWith.facet.570) [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.d86: <witness> = lookup_impl_witness %Result, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.cb2e47.2: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Result) [symbolic]
// CHECK:STDOUT:   %.d5f: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.cb2e47.2, %Result [symbolic]
// CHECK:STDOUT:   %impl.elem0.a6b: %.d5f = impl_witness_access %Destroy.lookup_impl_witness.d86, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.761: <specific function> = specific_impl_function %impl.elem0.a6b, @Destroy.WithSelf.Op(%Result) [symbolic]
// CHECK:STDOUT:   %SubWith.type.99c: type = facet_type <@SubWith.1, @SubWith.1(%Other)> [symbolic]
// CHECK:STDOUT:   %Self.ddf: %SubWith.type.99c = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %SubWith.assoc_type.645: type = assoc_entity_type @SubWith.1, @SubWith.1(%Other) [symbolic]
// CHECK:STDOUT:   %SubWith.WithSelf.Op.type.0bf: type = fn_type @SubWith.WithSelf.Op, @SubWith.WithSelf(%Other, %Self.ddf) [symbolic]
// CHECK:STDOUT:   %SubWith.WithSelf.Op.bd9: %SubWith.WithSelf.Op.type.0bf = struct_value () [symbolic]
// CHECK:STDOUT:   %SubWith.type.593: type = facet_type <@SubWith.1, @SubWith.1(%U.as_type.2b8)> [symbolic]
// CHECK:STDOUT:   %facet_type.db3: type = facet_type <@Destroy & @SubWith.1, @SubWith.1(%U.as_type.2b8)> [symbolic]
// CHECK:STDOUT:   %.Self.db3: %facet_type.db3 = symbolic_binding .Self [symbolic]
// CHECK:STDOUT:   %SubWith.assoc_type.065: type = assoc_entity_type @SubWith.1, @SubWith.1(%U.as_type.2b8) [symbolic]
// CHECK:STDOUT:   %assoc0.55d: %SubWith.assoc_type.065 = assoc_entity element0, imports.%Core.import_ref.7e5 [symbolic]
// CHECK:STDOUT:   %assoc1.173: %SubWith.assoc_type.065 = assoc_entity element1, imports.%Core.import_ref.abb [symbolic]
// CHECK:STDOUT:   %require_complete.b71: <witness> = require_complete_type %facet_type.db3 [symbolic]
// CHECK:STDOUT:   %.Self.as_type.2a6: type = facet_access_type %.Self.db3 [symbolic]
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.98d: <witness> = lookup_impl_witness %.Self.db3, @SubWith.1, @SubWith.1(%U.as_type.2b8) [symbolic]
// CHECK:STDOUT:   %impl.elem0.bbb: type = impl_witness_access %SubWith.lookup_impl_witness.98d, element0 [symbolic]
// CHECK:STDOUT:   %facet_type.bea: type = facet_type <@Destroy & @SubWith.1, @SubWith.1(%U.as_type.2b8) where %impl.elem0.bbb = %Result.as_type> [symbolic]
// CHECK:STDOUT:   %pattern_type.265: type = pattern_type %facet_type.bea [symbolic]
// CHECK:STDOUT:   %T.970: %facet_type.bea = symbolic_binding T, 2 [symbolic]
// CHECK:STDOUT:   %T.as_type.474: type = facet_access_type %T.970 [symbolic]
// CHECK:STDOUT:   %pattern_type.b6a: type = pattern_type %T.as_type.474 [symbolic]
// CHECK:STDOUT:   %require_complete.ae1: <witness> = require_complete_type %SubWith.type.593 [symbolic]
// CHECK:STDOUT:   %assoc1.f09: %SubWith.assoc_type.645 = assoc_entity element1, imports.%Core.import_ref.e83 [symbolic]
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.db9: <witness> = lookup_impl_witness %T.970, @SubWith.1, @SubWith.1(%U.as_type.2b8) [symbolic]
// CHECK:STDOUT:   %SubWith.facet.7f1: %SubWith.type.593 = facet_value %T.as_type.474, (%SubWith.lookup_impl_witness.db9) [symbolic]
// CHECK:STDOUT:   %SubWith.WithSelf.Op.type.e59: type = fn_type @SubWith.WithSelf.Op, @SubWith.WithSelf(%U.as_type.2b8, %SubWith.facet.7f1) [symbolic]
// CHECK:STDOUT:   %.4b6: type = fn_type_with_self_type %SubWith.WithSelf.Op.type.e59, %SubWith.facet.7f1 [symbolic]
// CHECK:STDOUT:   %impl.elem1.6e4: %.4b6 = impl_witness_access %SubWith.lookup_impl_witness.db9, element1 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.0d3: <specific function> = specific_impl_function %impl.elem1.6e4, @SubWith.WithSelf.Op(%U.as_type.2b8, %SubWith.facet.7f1) [symbolic]
// CHECK:STDOUT:   %MulWith.type.9a3: type = facet_type <@MulWith.1, @MulWith.1(%Other)> [symbolic]
// CHECK:STDOUT:   %Self.9c4: %MulWith.type.9a3 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %MulWith.assoc_type.b5d: type = assoc_entity_type @MulWith.1, @MulWith.1(%Other) [symbolic]
// CHECK:STDOUT:   %MulWith.WithSelf.Op.type.00c: type = fn_type @MulWith.WithSelf.Op, @MulWith.WithSelf(%Other, %Self.9c4) [symbolic]
// CHECK:STDOUT:   %MulWith.WithSelf.Op.07a: %MulWith.WithSelf.Op.type.00c = struct_value () [symbolic]
// CHECK:STDOUT:   %MulWith.type.9532: type = facet_type <@MulWith.1, @MulWith.1(%U.as_type.2b8)> [symbolic]
// CHECK:STDOUT:   %facet_type.17c: type = facet_type <@Destroy & @MulWith.1, @MulWith.1(%U.as_type.2b8)> [symbolic]
// CHECK:STDOUT:   %.Self.afa: %facet_type.17c = symbolic_binding .Self [symbolic]
// CHECK:STDOUT:   %MulWith.assoc_type.97c: type = assoc_entity_type @MulWith.1, @MulWith.1(%U.as_type.2b8) [symbolic]
// CHECK:STDOUT:   %assoc0.1354: %MulWith.assoc_type.97c = assoc_entity element0, imports.%Core.import_ref.d9b [symbolic]
// CHECK:STDOUT:   %assoc1.496: %MulWith.assoc_type.97c = assoc_entity element1, imports.%Core.import_ref.abc [symbolic]
// CHECK:STDOUT:   %require_complete.665: <witness> = require_complete_type %facet_type.17c [symbolic]
// CHECK:STDOUT:   %.Self.as_type.9cb: type = facet_access_type %.Self.afa [symbolic]
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.05e: <witness> = lookup_impl_witness %.Self.afa, @MulWith.1, @MulWith.1(%U.as_type.2b8) [symbolic]
// CHECK:STDOUT:   %impl.elem0.972: type = impl_witness_access %MulWith.lookup_impl_witness.05e, element0 [symbolic]
// CHECK:STDOUT:   %facet_type.91e: type = facet_type <@Destroy & @MulWith.1, @MulWith.1(%U.as_type.2b8) where %impl.elem0.972 = %Result.as_type> [symbolic]
// CHECK:STDOUT:   %pattern_type.b93: type = pattern_type %facet_type.91e [symbolic]
// CHECK:STDOUT:   %T.438: %facet_type.91e = symbolic_binding T, 2 [symbolic]
// CHECK:STDOUT:   %T.as_type.68f: type = facet_access_type %T.438 [symbolic]
// CHECK:STDOUT:   %pattern_type.b74: type = pattern_type %T.as_type.68f [symbolic]
// CHECK:STDOUT:   %require_complete.d9d: <witness> = require_complete_type %MulWith.type.9532 [symbolic]
// CHECK:STDOUT:   %assoc1.ccf: %MulWith.assoc_type.b5d = assoc_entity element1, imports.%Core.import_ref.d5a [symbolic]
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.009: <witness> = lookup_impl_witness %T.438, @MulWith.1, @MulWith.1(%U.as_type.2b8) [symbolic]
// CHECK:STDOUT:   %MulWith.facet.2d0: %MulWith.type.9532 = facet_value %T.as_type.68f, (%MulWith.lookup_impl_witness.009) [symbolic]
// CHECK:STDOUT:   %MulWith.WithSelf.Op.type.283: type = fn_type @MulWith.WithSelf.Op, @MulWith.WithSelf(%U.as_type.2b8, %MulWith.facet.2d0) [symbolic]
// CHECK:STDOUT:   %.978: type = fn_type_with_self_type %MulWith.WithSelf.Op.type.283, %MulWith.facet.2d0 [symbolic]
// CHECK:STDOUT:   %impl.elem1.80d: %.978 = impl_witness_access %MulWith.lookup_impl_witness.009, element1 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.889: <specific function> = specific_impl_function %impl.elem1.80d, @MulWith.WithSelf.Op(%U.as_type.2b8, %MulWith.facet.2d0) [symbolic]
// CHECK:STDOUT:   %DivWith.type.871d: type = facet_type <@DivWith.1, @DivWith.1(%Other)> [symbolic]
// CHECK:STDOUT:   %Self.d71: %DivWith.type.871d = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %DivWith.assoc_type.16c: type = assoc_entity_type @DivWith.1, @DivWith.1(%Other) [symbolic]
// CHECK:STDOUT:   %DivWith.WithSelf.Op.type.8a1: type = fn_type @DivWith.WithSelf.Op, @DivWith.WithSelf(%Other, %Self.d71) [symbolic]
// CHECK:STDOUT:   %DivWith.WithSelf.Op.0a9: %DivWith.WithSelf.Op.type.8a1 = struct_value () [symbolic]
// CHECK:STDOUT:   %DivWith.type.0e2: type = facet_type <@DivWith.1, @DivWith.1(%U.as_type.2b8)> [symbolic]
// CHECK:STDOUT:   %facet_type.94d: type = facet_type <@Destroy & @DivWith.1, @DivWith.1(%U.as_type.2b8)> [symbolic]
// CHECK:STDOUT:   %.Self.5a5: %facet_type.94d = symbolic_binding .Self [symbolic]
// CHECK:STDOUT:   %DivWith.assoc_type.06d: type = assoc_entity_type @DivWith.1, @DivWith.1(%U.as_type.2b8) [symbolic]
// CHECK:STDOUT:   %assoc0.1db: %DivWith.assoc_type.06d = assoc_entity element0, imports.%Core.import_ref.14e [symbolic]
// CHECK:STDOUT:   %assoc1.49b: %DivWith.assoc_type.06d = assoc_entity element1, imports.%Core.import_ref.bb2c [symbolic]
// CHECK:STDOUT:   %require_complete.3a4: <witness> = require_complete_type %facet_type.94d [symbolic]
// CHECK:STDOUT:   %.Self.as_type.dad: type = facet_access_type %.Self.5a5 [symbolic]
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.fbf: <witness> = lookup_impl_witness %.Self.5a5, @DivWith.1, @DivWith.1(%U.as_type.2b8) [symbolic]
// CHECK:STDOUT:   %impl.elem0.409: type = impl_witness_access %DivWith.lookup_impl_witness.fbf, element0 [symbolic]
// CHECK:STDOUT:   %facet_type.6e6: type = facet_type <@Destroy & @DivWith.1, @DivWith.1(%U.as_type.2b8) where %impl.elem0.409 = %Result.as_type> [symbolic]
// CHECK:STDOUT:   %pattern_type.4ef: type = pattern_type %facet_type.6e6 [symbolic]
// CHECK:STDOUT:   %T.30e: %facet_type.6e6 = symbolic_binding T, 2 [symbolic]
// CHECK:STDOUT:   %T.as_type.cfe: type = facet_access_type %T.30e [symbolic]
// CHECK:STDOUT:   %pattern_type.daa: type = pattern_type %T.as_type.cfe [symbolic]
// CHECK:STDOUT:   %require_complete.9a1: <witness> = require_complete_type %DivWith.type.0e2 [symbolic]
// CHECK:STDOUT:   %assoc1.b83: %DivWith.assoc_type.16c = assoc_entity element1, imports.%Core.import_ref.4c4 [symbolic]
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.b69: <witness> = lookup_impl_witness %T.30e, @DivWith.1, @DivWith.1(%U.as_type.2b8) [symbolic]
// CHECK:STDOUT:   %DivWith.facet.490: %DivWith.type.0e2 = facet_value %T.as_type.cfe, (%DivWith.lookup_impl_witness.b69) [symbolic]
// CHECK:STDOUT:   %DivWith.WithSelf.Op.type.2ad: type = fn_type @DivWith.WithSelf.Op, @DivWith.WithSelf(%U.as_type.2b8, %DivWith.facet.490) [symbolic]
// CHECK:STDOUT:   %.0bf: type = fn_type_with_self_type %DivWith.WithSelf.Op.type.2ad, %DivWith.facet.490 [symbolic]
// CHECK:STDOUT:   %impl.elem1.b07: %.0bf = impl_witness_access %DivWith.lookup_impl_witness.b69, element1 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.dbd: <specific function> = specific_impl_function %impl.elem1.b07, @DivWith.WithSelf.Op(%U.as_type.2b8, %DivWith.facet.490) [symbolic]
// CHECK:STDOUT:   %ModWith.type.155: type = facet_type <@ModWith.1, @ModWith.1(%Other)> [symbolic]
// CHECK:STDOUT:   %Self.797: %ModWith.type.155 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %ModWith.assoc_type.e6f: type = assoc_entity_type @ModWith.1, @ModWith.1(%Other) [symbolic]
// CHECK:STDOUT:   %ModWith.WithSelf.Op.type.f61: type = fn_type @ModWith.WithSelf.Op, @ModWith.WithSelf(%Other, %Self.797) [symbolic]
// CHECK:STDOUT:   %ModWith.WithSelf.Op.3ac: %ModWith.WithSelf.Op.type.f61 = struct_value () [symbolic]
// CHECK:STDOUT:   %ModWith.type.c6f: type = facet_type <@ModWith.1, @ModWith.1(%U.as_type.2b8)> [symbolic]
// CHECK:STDOUT:   %facet_type.428: type = facet_type <@Destroy & @ModWith.1, @ModWith.1(%U.as_type.2b8)> [symbolic]
// CHECK:STDOUT:   %.Self.2d4: %facet_type.428 = symbolic_binding .Self [symbolic]
// CHECK:STDOUT:   %ModWith.assoc_type.66e: type = assoc_entity_type @ModWith.1, @ModWith.1(%U.as_type.2b8) [symbolic]
// CHECK:STDOUT:   %assoc0.fd6: %ModWith.assoc_type.66e = assoc_entity element0, imports.%Core.import_ref.a0d [symbolic]
// CHECK:STDOUT:   %assoc1.3da: %ModWith.assoc_type.66e = assoc_entity element1, imports.%Core.import_ref.c05 [symbolic]
// CHECK:STDOUT:   %require_complete.9441: <witness> = require_complete_type %facet_type.428 [symbolic]
// CHECK:STDOUT:   %.Self.as_type.12a6: type = facet_access_type %.Self.2d4 [symbolic]
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.afa: <witness> = lookup_impl_witness %.Self.2d4, @ModWith.1, @ModWith.1(%U.as_type.2b8) [symbolic]
// CHECK:STDOUT:   %impl.elem0.44b: type = impl_witness_access %ModWith.lookup_impl_witness.afa, element0 [symbolic]
// CHECK:STDOUT:   %facet_type.3a6: type = facet_type <@Destroy & @ModWith.1, @ModWith.1(%U.as_type.2b8) where %impl.elem0.44b = %Result.as_type> [symbolic]
// CHECK:STDOUT:   %pattern_type.50d: type = pattern_type %facet_type.3a6 [symbolic]
// CHECK:STDOUT:   %T.a77: %facet_type.3a6 = symbolic_binding T, 2 [symbolic]
// CHECK:STDOUT:   %T.as_type.10f: type = facet_access_type %T.a77 [symbolic]
// CHECK:STDOUT:   %pattern_type.8db: type = pattern_type %T.as_type.10f [symbolic]
// CHECK:STDOUT:   %require_complete.9b1: <witness> = require_complete_type %ModWith.type.c6f [symbolic]
// CHECK:STDOUT:   %assoc1.e77: %ModWith.assoc_type.e6f = assoc_entity element1, imports.%Core.import_ref.7b7 [symbolic]
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.e3f: <witness> = lookup_impl_witness %T.a77, @ModWith.1, @ModWith.1(%U.as_type.2b8) [symbolic]
// CHECK:STDOUT:   %ModWith.facet.430: %ModWith.type.c6f = facet_value %T.as_type.10f, (%ModWith.lookup_impl_witness.e3f) [symbolic]
// CHECK:STDOUT:   %ModWith.WithSelf.Op.type.5b5: type = fn_type @ModWith.WithSelf.Op, @ModWith.WithSelf(%U.as_type.2b8, %ModWith.facet.430) [symbolic]
// CHECK:STDOUT:   %.feb: type = fn_type_with_self_type %ModWith.WithSelf.Op.type.5b5, %ModWith.facet.430 [symbolic]
// CHECK:STDOUT:   %impl.elem1.1af: %.feb = impl_witness_access %ModWith.lookup_impl_witness.e3f, element1 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.772: <specific function> = specific_impl_function %impl.elem1.1af, @ModWith.WithSelf.Op(%U.as_type.2b8, %ModWith.facet.430) [symbolic]
// CHECK:STDOUT:   %Int16: type = class_type @Int16 [concrete]
// CHECK:STDOUT:   %pattern_type.9a4: type = pattern_type %Int16 [concrete]
// CHECK:STDOUT:   %Int32: type = class_type @Int32 [concrete]
// CHECK:STDOUT:   %pattern_type.9c9: type = pattern_type %Int32 [concrete]
// CHECK:STDOUT:   %Int64: type = class_type @Int64 [concrete]
// CHECK:STDOUT:   %pattern_type.f42: type = pattern_type %Int64 [concrete]
// CHECK:STDOUT:   %int_16: Core.IntLiteral = int_value 16 [concrete]
// CHECK:STDOUT:   %i16: type = class_type @Int, @Int(%int_16) [concrete]
// CHECK:STDOUT:   %pattern_type.2f8: type = pattern_type %i16 [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %i16.builtin: type = int_type signed, %int_16 [concrete]
// CHECK:STDOUT:   %complete_type.2da: <witness> = complete_type_witness %i16.builtin [concrete]
// CHECK:STDOUT:   %Int16.Op.type.180c8b.1: type = fn_type @Int16.Op.1 [concrete]
// CHECK:STDOUT:   %Int16.Op.c3e49c.1: %Int16.Op.type.180c8b.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.e5b: <witness> = custom_witness (%Int16.Op.c3e49c.1), @Destroy [concrete]
// CHECK:STDOUT:   %Destroy.facet.e06: %Destroy.type = facet_value %Int16, (%custom_witness.e5b) [concrete]
// CHECK:STDOUT:   %facet_type.a81: type = facet_type <@Destroy & @AddWith.1, @AddWith.1(%Int16)> [concrete]
// CHECK:STDOUT:   %.Self.0ae: %facet_type.a81 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.db4: <witness> = lookup_impl_witness %.Self.0ae, @AddWith.1, @AddWith.1(%Int16) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.cdb: type = impl_witness_access %AddWith.lookup_impl_witness.db4, element0 [symbolic_self]
// CHECK:STDOUT:   %facet_type.e97: type = facet_type <@Destroy & @AddWith.1, @AddWith.1(%Int16) where %impl.elem0.cdb = %Int16> [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.bae255.2: type = fn_type @Destroy.Op.loc176_34.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.651ba6.2: %Destroy.Op.type.bae255.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.8d7fae.2: <witness> = custom_witness (%Destroy.Op.651ba6.2), @Destroy [concrete]
// CHECK:STDOUT:   %AddWith.type.344: type = facet_type <@AddWith.1, @AddWith.1(%Int16)> [concrete]
// CHECK:STDOUT:   %AddWith.assoc_type.f75: type = assoc_entity_type @AddWith.1, @AddWith.1(%Int16) [concrete]
// CHECK:STDOUT:   %assoc0.dc9: %AddWith.assoc_type.f75 = assoc_entity element0, imports.%Core.import_ref.6e9 [concrete]
// CHECK:STDOUT:   %assoc1.646: %AddWith.assoc_type.f75 = assoc_entity element1, imports.%Core.import_ref.1b1 [concrete]
// CHECK:STDOUT:   %Op.type.67547d.1: type = fn_type @Op.1 [concrete]
// CHECK:STDOUT:   %Op.8fb557.1: %Op.type.67547d.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.facet.77e: %Destroy.type = facet_value %i16, (%custom_witness.8d7fae.2) [concrete]
// CHECK:STDOUT:   %custom_witness.47f: <witness> = custom_witness (%Int16, %Op.8fb557.1), @AddWith.1, @AddWith.1(%Int16) [concrete]
// CHECK:STDOUT:   %facet_value.c79: %facet_type.e97 = facet_value %i16, (%custom_witness.8d7fae.2, %custom_witness.47f) [concrete]
// CHECK:STDOUT:   %complete_type.f61: <witness> = complete_type_witness %facet_type.a81 [concrete]
// CHECK:STDOUT:   %.Self.as_type.12a0: type = facet_access_type %.Self.0ae [symbolic_self]
// CHECK:STDOUT:   %pattern_type.690: type = pattern_type %facet_type.e97 [concrete]
// CHECK:STDOUT:   %facet_type.b37: type = facet_type <@Destroy & @SubWith.1, @SubWith.1(%Int16)> [concrete]
// CHECK:STDOUT:   %.Self.107: %facet_type.b37 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.a35: <witness> = lookup_impl_witness %.Self.107, @SubWith.1, @SubWith.1(%Int16) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.345: type = impl_witness_access %SubWith.lookup_impl_witness.a35, element0 [symbolic_self]
// CHECK:STDOUT:   %facet_type.281: type = facet_type <@Destroy & @SubWith.1, @SubWith.1(%Int16) where %impl.elem0.345 = %Int16> [concrete]
// CHECK:STDOUT:   %SubWith.type.306: type = facet_type <@SubWith.1, @SubWith.1(%Int16)> [concrete]
// CHECK:STDOUT:   %SubWith.assoc_type.7a2: type = assoc_entity_type @SubWith.1, @SubWith.1(%Int16) [concrete]
// CHECK:STDOUT:   %assoc0.f42: %SubWith.assoc_type.7a2 = assoc_entity element0, imports.%Core.import_ref.7e5 [concrete]
// CHECK:STDOUT:   %assoc1.2c3: %SubWith.assoc_type.7a2 = assoc_entity element1, imports.%Core.import_ref.abb [concrete]
// CHECK:STDOUT:   %Op.type.67547d.2: type = fn_type @Op.2 [concrete]
// CHECK:STDOUT:   %Op.8fb557.2: %Op.type.67547d.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.a7b: <witness> = custom_witness (%Int16, %Op.8fb557.2), @SubWith.1, @SubWith.1(%Int16) [concrete]
// CHECK:STDOUT:   %facet_value.501: %facet_type.281 = facet_value %i16, (%custom_witness.8d7fae.2, %custom_witness.a7b) [concrete]
// CHECK:STDOUT:   %complete_type.5cf: <witness> = complete_type_witness %facet_type.b37 [concrete]
// CHECK:STDOUT:   %.Self.as_type.e34: type = facet_access_type %.Self.107 [symbolic_self]
// CHECK:STDOUT:   %pattern_type.926: type = pattern_type %facet_type.281 [concrete]
// CHECK:STDOUT:   %facet_type.2dd: type = facet_type <@Destroy & @MulWith.1, @MulWith.1(%Int16)> [concrete]
// CHECK:STDOUT:   %.Self.594: %facet_type.2dd = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.834: <witness> = lookup_impl_witness %.Self.594, @MulWith.1, @MulWith.1(%Int16) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.6fb: type = impl_witness_access %MulWith.lookup_impl_witness.834, element0 [symbolic_self]
// CHECK:STDOUT:   %facet_type.b7c: type = facet_type <@Destroy & @MulWith.1, @MulWith.1(%Int16) where %impl.elem0.6fb = %Int16> [concrete]
// CHECK:STDOUT:   %MulWith.type.0cd: type = facet_type <@MulWith.1, @MulWith.1(%Int16)> [concrete]
// CHECK:STDOUT:   %MulWith.assoc_type.f1b: type = assoc_entity_type @MulWith.1, @MulWith.1(%Int16) [concrete]
// CHECK:STDOUT:   %assoc0.3ee: %MulWith.assoc_type.f1b = assoc_entity element0, imports.%Core.import_ref.d9b [concrete]
// CHECK:STDOUT:   %assoc1.988: %MulWith.assoc_type.f1b = assoc_entity element1, imports.%Core.import_ref.abc [concrete]
// CHECK:STDOUT:   %Op.type.67547d.3: type = fn_type @Op.3 [concrete]
// CHECK:STDOUT:   %Op.8fb557.3: %Op.type.67547d.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.e40: <witness> = custom_witness (%Int16, %Op.8fb557.3), @MulWith.1, @MulWith.1(%Int16) [concrete]
// CHECK:STDOUT:   %facet_value.86a: %facet_type.b7c = facet_value %i16, (%custom_witness.8d7fae.2, %custom_witness.e40) [concrete]
// CHECK:STDOUT:   %complete_type.b32: <witness> = complete_type_witness %facet_type.2dd [concrete]
// CHECK:STDOUT:   %.Self.as_type.674: type = facet_access_type %.Self.594 [symbolic_self]
// CHECK:STDOUT:   %pattern_type.401: type = pattern_type %facet_type.b7c [concrete]
// CHECK:STDOUT:   %facet_type.e82: type = facet_type <@Destroy & @DivWith.1, @DivWith.1(%Int16)> [concrete]
// CHECK:STDOUT:   %.Self.c70: %facet_type.e82 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.dbb: <witness> = lookup_impl_witness %.Self.c70, @DivWith.1, @DivWith.1(%Int16) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.e43: type = impl_witness_access %DivWith.lookup_impl_witness.dbb, element0 [symbolic_self]
// CHECK:STDOUT:   %facet_type.a08: type = facet_type <@Destroy & @DivWith.1, @DivWith.1(%Int16) where %impl.elem0.e43 = %Int16> [concrete]
// CHECK:STDOUT:   %DivWith.type.4c2: type = facet_type <@DivWith.1, @DivWith.1(%Int16)> [concrete]
// CHECK:STDOUT:   %DivWith.assoc_type.9fa: type = assoc_entity_type @DivWith.1, @DivWith.1(%Int16) [concrete]
// CHECK:STDOUT:   %assoc0.cf5: %DivWith.assoc_type.9fa = assoc_entity element0, imports.%Core.import_ref.14e [concrete]
// CHECK:STDOUT:   %assoc1.cda: %DivWith.assoc_type.9fa = assoc_entity element1, imports.%Core.import_ref.bb2c [concrete]
// CHECK:STDOUT:   %Op.type.67547d.4: type = fn_type @Op.4 [concrete]
// CHECK:STDOUT:   %Op.8fb557.4: %Op.type.67547d.4 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.fb3: <witness> = custom_witness (%Int16, %Op.8fb557.4), @DivWith.1, @DivWith.1(%Int16) [concrete]
// CHECK:STDOUT:   %facet_value.018: %facet_type.a08 = facet_value %i16, (%custom_witness.8d7fae.2, %custom_witness.fb3) [concrete]
// CHECK:STDOUT:   %complete_type.8b32: <witness> = complete_type_witness %facet_type.e82 [concrete]
// CHECK:STDOUT:   %.Self.as_type.e0e: type = facet_access_type %.Self.c70 [symbolic_self]
// CHECK:STDOUT:   %pattern_type.6c0: type = pattern_type %facet_type.a08 [concrete]
// CHECK:STDOUT:   %facet_type.e1e: type = facet_type <@Destroy & @ModWith.1, @ModWith.1(%Int16)> [concrete]
// CHECK:STDOUT:   %.Self.76e: %facet_type.e1e = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.c6b: <witness> = lookup_impl_witness %.Self.76e, @ModWith.1, @ModWith.1(%Int16) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.cba: type = impl_witness_access %ModWith.lookup_impl_witness.c6b, element0 [symbolic_self]
// CHECK:STDOUT:   %facet_type.469: type = facet_type <@Destroy & @ModWith.1, @ModWith.1(%Int16) where %impl.elem0.cba = %Int16> [concrete]
// CHECK:STDOUT:   %ModWith.type.009: type = facet_type <@ModWith.1, @ModWith.1(%Int16)> [concrete]
// CHECK:STDOUT:   %ModWith.assoc_type.355: type = assoc_entity_type @ModWith.1, @ModWith.1(%Int16) [concrete]
// CHECK:STDOUT:   %assoc0.0f7: %ModWith.assoc_type.355 = assoc_entity element0, imports.%Core.import_ref.a0d [concrete]
// CHECK:STDOUT:   %assoc1.c82: %ModWith.assoc_type.355 = assoc_entity element1, imports.%Core.import_ref.c05 [concrete]
// CHECK:STDOUT:   %Op.type.67547d.5: type = fn_type @Op.5 [concrete]
// CHECK:STDOUT:   %Op.8fb557.5: %Op.type.67547d.5 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.69e: <witness> = custom_witness (%Int16, %Op.8fb557.5), @ModWith.1, @ModWith.1(%Int16) [concrete]
// CHECK:STDOUT:   %facet_value.ee6: %facet_type.469 = facet_value %i16, (%custom_witness.8d7fae.2, %custom_witness.69e) [concrete]
// CHECK:STDOUT:   %complete_type.5b6: <witness> = complete_type_witness %facet_type.e1e [concrete]
// CHECK:STDOUT:   %.Self.as_type.242: type = facet_access_type %.Self.76e [symbolic_self]
// CHECK:STDOUT:   %pattern_type.13c: type = pattern_type %facet_type.469 [concrete]
// CHECK:STDOUT:   %facet_type.8b1: type = facet_type <@Destroy & @AddWith.1, @AddWith.1(%i16)> [concrete]
// CHECK:STDOUT:   %.Self.ba7: %facet_type.8b1 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.d8b: <witness> = lookup_impl_witness %.Self.ba7, @AddWith.1, @AddWith.1(%i16) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.265: type = impl_witness_access %AddWith.lookup_impl_witness.d8b, element0 [symbolic_self]
// CHECK:STDOUT:   %facet_type.731: type = facet_type <@Destroy & @AddWith.1, @AddWith.1(%i16) where %impl.elem0.265 = %Int16> [concrete]
// CHECK:STDOUT:   %AddWith.type.a18: type = facet_type <@AddWith.1, @AddWith.1(%i16)> [concrete]
// CHECK:STDOUT:   %AddWith.assoc_type.4f0: type = assoc_entity_type @AddWith.1, @AddWith.1(%i16) [concrete]
// CHECK:STDOUT:   %assoc0.ba6: %AddWith.assoc_type.4f0 = assoc_entity element0, imports.%Core.import_ref.6e9 [concrete]
// CHECK:STDOUT:   %assoc1.88d: %AddWith.assoc_type.4f0 = assoc_entity element1, imports.%Core.import_ref.1b1 [concrete]
// CHECK:STDOUT:   %Int16.Op.type.180c8b.3: type = fn_type @Int16.Op.3 [concrete]
// CHECK:STDOUT:   %Int16.Op.c3e49c.3: %Int16.Op.type.180c8b.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.096: <witness> = custom_witness (%Int16, %Int16.Op.c3e49c.3), @AddWith.1, @AddWith.1(%i16) [concrete]
// CHECK:STDOUT:   %facet_value.dd6: %facet_type.731 = facet_value %Int16, (%custom_witness.e5b, %custom_witness.096) [concrete]
// CHECK:STDOUT:   %complete_type.59d: <witness> = complete_type_witness %facet_type.8b1 [concrete]
// CHECK:STDOUT:   %.Self.as_type.f28: type = facet_access_type %.Self.ba7 [symbolic_self]
// CHECK:STDOUT:   %pattern_type.336: type = pattern_type %facet_type.731 [concrete]
// CHECK:STDOUT:   %facet_type.bd5: type = facet_type <@Destroy & @SubWith.1, @SubWith.1(%i16)> [concrete]
// CHECK:STDOUT:   %.Self.d04: %facet_type.bd5 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.029: <witness> = lookup_impl_witness %.Self.d04, @SubWith.1, @SubWith.1(%i16) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.e59: type = impl_witness_access %SubWith.lookup_impl_witness.029, element0 [symbolic_self]
// CHECK:STDOUT:   %facet_type.257: type = facet_type <@Destroy & @SubWith.1, @SubWith.1(%i16) where %impl.elem0.e59 = %Int16> [concrete]
// CHECK:STDOUT:   %SubWith.type.a1b: type = facet_type <@SubWith.1, @SubWith.1(%i16)> [concrete]
// CHECK:STDOUT:   %SubWith.assoc_type.108: type = assoc_entity_type @SubWith.1, @SubWith.1(%i16) [concrete]
// CHECK:STDOUT:   %assoc0.053: %SubWith.assoc_type.108 = assoc_entity element0, imports.%Core.import_ref.7e5 [concrete]
// CHECK:STDOUT:   %assoc1.ef4: %SubWith.assoc_type.108 = assoc_entity element1, imports.%Core.import_ref.abb [concrete]
// CHECK:STDOUT:   %Int16.Op.type.180c8b.4: type = fn_type @Int16.Op.4 [concrete]
// CHECK:STDOUT:   %Int16.Op.c3e49c.4: %Int16.Op.type.180c8b.4 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.ced: <witness> = custom_witness (%Int16, %Int16.Op.c3e49c.4), @SubWith.1, @SubWith.1(%i16) [concrete]
// CHECK:STDOUT:   %facet_value.3f5: %facet_type.257 = facet_value %Int16, (%custom_witness.e5b, %custom_witness.ced) [concrete]
// CHECK:STDOUT:   %complete_type.d4c: <witness> = complete_type_witness %facet_type.bd5 [concrete]
// CHECK:STDOUT:   %.Self.as_type.d0c7: type = facet_access_type %.Self.d04 [symbolic_self]
// CHECK:STDOUT:   %pattern_type.021: type = pattern_type %facet_type.257 [concrete]
// CHECK:STDOUT:   %facet_type.5ca: type = facet_type <@Destroy & @MulWith.1, @MulWith.1(%i16)> [concrete]
// CHECK:STDOUT:   %.Self.3cf: %facet_type.5ca = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.aee: <witness> = lookup_impl_witness %.Self.3cf, @MulWith.1, @MulWith.1(%i16) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.5d1: type = impl_witness_access %MulWith.lookup_impl_witness.aee, element0 [symbolic_self]
// CHECK:STDOUT:   %facet_type.865: type = facet_type <@Destroy & @MulWith.1, @MulWith.1(%i16) where %impl.elem0.5d1 = %Int16> [concrete]
// CHECK:STDOUT:   %MulWith.type.ee9: type = facet_type <@MulWith.1, @MulWith.1(%i16)> [concrete]
// CHECK:STDOUT:   %MulWith.assoc_type.aae: type = assoc_entity_type @MulWith.1, @MulWith.1(%i16) [concrete]
// CHECK:STDOUT:   %assoc0.4d95: %MulWith.assoc_type.aae = assoc_entity element0, imports.%Core.import_ref.d9b [concrete]
// CHECK:STDOUT:   %assoc1.587: %MulWith.assoc_type.aae = assoc_entity element1, imports.%Core.import_ref.abc [concrete]
// CHECK:STDOUT:   %Int16.Op.type.180c8b.5: type = fn_type @Int16.Op.5 [concrete]
// CHECK:STDOUT:   %Int16.Op.c3e49c.5: %Int16.Op.type.180c8b.5 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.270: <witness> = custom_witness (%Int16, %Int16.Op.c3e49c.5), @MulWith.1, @MulWith.1(%i16) [concrete]
// CHECK:STDOUT:   %facet_value.4ca: %facet_type.865 = facet_value %Int16, (%custom_witness.e5b, %custom_witness.270) [concrete]
// CHECK:STDOUT:   %complete_type.d8f: <witness> = complete_type_witness %facet_type.5ca [concrete]
// CHECK:STDOUT:   %.Self.as_type.4dc: type = facet_access_type %.Self.3cf [symbolic_self]
// CHECK:STDOUT:   %pattern_type.038: type = pattern_type %facet_type.865 [concrete]
// CHECK:STDOUT:   %facet_type.014: type = facet_type <@Destroy & @DivWith.1, @DivWith.1(%i16)> [concrete]
// CHECK:STDOUT:   %.Self.197: %facet_type.014 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.834: <witness> = lookup_impl_witness %.Self.197, @DivWith.1, @DivWith.1(%i16) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.146: type = impl_witness_access %DivWith.lookup_impl_witness.834, element0 [symbolic_self]
// CHECK:STDOUT:   %facet_type.c20: type = facet_type <@Destroy & @DivWith.1, @DivWith.1(%i16) where %impl.elem0.146 = %Int16> [concrete]
// CHECK:STDOUT:   %DivWith.type.9ae: type = facet_type <@DivWith.1, @DivWith.1(%i16)> [concrete]
// CHECK:STDOUT:   %DivWith.assoc_type.e80: type = assoc_entity_type @DivWith.1, @DivWith.1(%i16) [concrete]
// CHECK:STDOUT:   %assoc0.75e: %DivWith.assoc_type.e80 = assoc_entity element0, imports.%Core.import_ref.14e [concrete]
// CHECK:STDOUT:   %assoc1.9ae: %DivWith.assoc_type.e80 = assoc_entity element1, imports.%Core.import_ref.bb2c [concrete]
// CHECK:STDOUT:   %Int16.Op.type.180c8b.6: type = fn_type @Int16.Op.6 [concrete]
// CHECK:STDOUT:   %Int16.Op.c3e49c.6: %Int16.Op.type.180c8b.6 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.462: <witness> = custom_witness (%Int16, %Int16.Op.c3e49c.6), @DivWith.1, @DivWith.1(%i16) [concrete]
// CHECK:STDOUT:   %facet_value.ff9: %facet_type.c20 = facet_value %Int16, (%custom_witness.e5b, %custom_witness.462) [concrete]
// CHECK:STDOUT:   %complete_type.b92: <witness> = complete_type_witness %facet_type.014 [concrete]
// CHECK:STDOUT:   %.Self.as_type.2e1: type = facet_access_type %.Self.197 [symbolic_self]
// CHECK:STDOUT:   %pattern_type.8c5: type = pattern_type %facet_type.c20 [concrete]
// CHECK:STDOUT:   %facet_type.fe7: type = facet_type <@Destroy & @ModWith.1, @ModWith.1(%i16)> [concrete]
// CHECK:STDOUT:   %.Self.7d0: %facet_type.fe7 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.ff6: <witness> = lookup_impl_witness %.Self.7d0, @ModWith.1, @ModWith.1(%i16) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.f78: type = impl_witness_access %ModWith.lookup_impl_witness.ff6, element0 [symbolic_self]
// CHECK:STDOUT:   %facet_type.a93: type = facet_type <@Destroy & @ModWith.1, @ModWith.1(%i16) where %impl.elem0.f78 = %Int16> [concrete]
// CHECK:STDOUT:   %ModWith.type.406: type = facet_type <@ModWith.1, @ModWith.1(%i16)> [concrete]
// CHECK:STDOUT:   %ModWith.assoc_type.ffe: type = assoc_entity_type @ModWith.1, @ModWith.1(%i16) [concrete]
// CHECK:STDOUT:   %assoc0.b49: %ModWith.assoc_type.ffe = assoc_entity element0, imports.%Core.import_ref.a0d [concrete]
// CHECK:STDOUT:   %assoc1.d40: %ModWith.assoc_type.ffe = assoc_entity element1, imports.%Core.import_ref.c05 [concrete]
// CHECK:STDOUT:   %Int16.Op.type.180c8b.7: type = fn_type @Int16.Op.7 [concrete]
// CHECK:STDOUT:   %Int16.Op.c3e49c.7: %Int16.Op.type.180c8b.7 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.84d2: <witness> = custom_witness (%Int16, %Int16.Op.c3e49c.7), @ModWith.1, @ModWith.1(%i16) [concrete]
// CHECK:STDOUT:   %facet_value.f56: %facet_type.a93 = facet_value %Int16, (%custom_witness.e5b, %custom_witness.84d2) [concrete]
// CHECK:STDOUT:   %complete_type.dfc: <witness> = complete_type_witness %facet_type.fe7 [concrete]
// CHECK:STDOUT:   %.Self.as_type.81b: type = facet_access_type %.Self.7d0 [symbolic_self]
// CHECK:STDOUT:   %pattern_type.0e2: type = pattern_type %facet_type.a93 [concrete]
// CHECK:STDOUT:   %Int32.Op.type.68b858.1: type = fn_type @Int32.Op.1 [concrete]
// CHECK:STDOUT:   %Int32.Op.6afff1.1: %Int32.Op.type.68b858.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.ce1: <witness> = custom_witness (%Int32.Op.6afff1.1), @Destroy [concrete]
// CHECK:STDOUT:   %Destroy.facet.c06: %Destroy.type = facet_value %Int32, (%custom_witness.ce1) [concrete]
// CHECK:STDOUT:   %facet_type.4a7: type = facet_type <@Destroy & @AddWith.1, @AddWith.1(%Int32)> [concrete]
// CHECK:STDOUT:   %.Self.3d9: %facet_type.4a7 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.632: <witness> = lookup_impl_witness %.Self.3d9, @AddWith.1, @AddWith.1(%Int32) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.cee: type = impl_witness_access %AddWith.lookup_impl_witness.632, element0 [symbolic_self]
// CHECK:STDOUT:   %facet_type.14d: type = facet_type <@Destroy & @AddWith.1, @AddWith.1(%Int32) where %impl.elem0.cee = %Int32> [concrete]
// CHECK:STDOUT:   %AddWith.type.e9e: type = facet_type <@AddWith.1, @AddWith.1(%Int32)> [concrete]
// CHECK:STDOUT:   %AddWith.assoc_type.29a: type = assoc_entity_type @AddWith.1, @AddWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %assoc0.701: %AddWith.assoc_type.29a = assoc_entity element0, imports.%Core.import_ref.6e9 [concrete]
// CHECK:STDOUT:   %assoc1.38b: %AddWith.assoc_type.29a = assoc_entity element1, imports.%Core.import_ref.1b1 [concrete]
// CHECK:STDOUT:   %Int16.Op.type.180c8b.8: type = fn_type @Int16.Op.8 [concrete]
// CHECK:STDOUT:   %Int16.Op.c3e49c.8: %Int16.Op.type.180c8b.8 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.200: <witness> = custom_witness (%Int32, %Int16.Op.c3e49c.8), @AddWith.1, @AddWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %facet_value.2b7: %facet_type.14d = facet_value %Int16, (%custom_witness.e5b, %custom_witness.200) [concrete]
// CHECK:STDOUT:   %complete_type.0d9: <witness> = complete_type_witness %facet_type.4a7 [concrete]
// CHECK:STDOUT:   %.Self.as_type.7ba: type = facet_access_type %.Self.3d9 [symbolic_self]
// CHECK:STDOUT:   %pattern_type.3f9: type = pattern_type %facet_type.14d [concrete]
// CHECK:STDOUT:   %facet_type.e7a: type = facet_type <@Destroy & @SubWith.1, @SubWith.1(%Int32)> [concrete]
// CHECK:STDOUT:   %.Self.384: %facet_type.e7a = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.b61: <witness> = lookup_impl_witness %.Self.384, @SubWith.1, @SubWith.1(%Int32) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.edd: type = impl_witness_access %SubWith.lookup_impl_witness.b61, element0 [symbolic_self]
// CHECK:STDOUT:   %facet_type.f15: type = facet_type <@Destroy & @SubWith.1, @SubWith.1(%Int32) where %impl.elem0.edd = %Int32> [concrete]
// CHECK:STDOUT:   %SubWith.type.84c: type = facet_type <@SubWith.1, @SubWith.1(%Int32)> [concrete]
// CHECK:STDOUT:   %SubWith.assoc_type.0f9: type = assoc_entity_type @SubWith.1, @SubWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %assoc0.1e9: %SubWith.assoc_type.0f9 = assoc_entity element0, imports.%Core.import_ref.7e5 [concrete]
// CHECK:STDOUT:   %assoc1.714: %SubWith.assoc_type.0f9 = assoc_entity element1, imports.%Core.import_ref.abb [concrete]
// CHECK:STDOUT:   %Int16.Op.type.180c8b.9: type = fn_type @Int16.Op.9 [concrete]
// CHECK:STDOUT:   %Int16.Op.c3e49c.9: %Int16.Op.type.180c8b.9 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.103: <witness> = custom_witness (%Int32, %Int16.Op.c3e49c.9), @SubWith.1, @SubWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %facet_value.7d7: %facet_type.f15 = facet_value %Int16, (%custom_witness.e5b, %custom_witness.103) [concrete]
// CHECK:STDOUT:   %complete_type.a34: <witness> = complete_type_witness %facet_type.e7a [concrete]
// CHECK:STDOUT:   %.Self.as_type.45d: type = facet_access_type %.Self.384 [symbolic_self]
// CHECK:STDOUT:   %pattern_type.56e: type = pattern_type %facet_type.f15 [concrete]
// CHECK:STDOUT:   %facet_type.2e6: type = facet_type <@Destroy & @MulWith.1, @MulWith.1(%Int32)> [concrete]
// CHECK:STDOUT:   %.Self.84b: %facet_type.2e6 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.539: <witness> = lookup_impl_witness %.Self.84b, @MulWith.1, @MulWith.1(%Int32) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.e63: type = impl_witness_access %MulWith.lookup_impl_witness.539, element0 [symbolic_self]
// CHECK:STDOUT:   %facet_type.c22: type = facet_type <@Destroy & @MulWith.1, @MulWith.1(%Int32) where %impl.elem0.e63 = %Int32> [concrete]
// CHECK:STDOUT:   %MulWith.type.d1b: type = facet_type <@MulWith.1, @MulWith.1(%Int32)> [concrete]
// CHECK:STDOUT:   %MulWith.assoc_type.8c6: type = assoc_entity_type @MulWith.1, @MulWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %assoc0.a8f: %MulWith.assoc_type.8c6 = assoc_entity element0, imports.%Core.import_ref.d9b [concrete]
// CHECK:STDOUT:   %assoc1.6bc: %MulWith.assoc_type.8c6 = assoc_entity element1, imports.%Core.import_ref.abc [concrete]
// CHECK:STDOUT:   %Int16.Op.type.180c8b.10: type = fn_type @Int16.Op.10 [concrete]
// CHECK:STDOUT:   %Int16.Op.c3e49c.10: %Int16.Op.type.180c8b.10 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.a6d: <witness> = custom_witness (%Int32, %Int16.Op.c3e49c.10), @MulWith.1, @MulWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %facet_value.75c: %facet_type.c22 = facet_value %Int16, (%custom_witness.e5b, %custom_witness.a6d) [concrete]
// CHECK:STDOUT:   %complete_type.043: <witness> = complete_type_witness %facet_type.2e6 [concrete]
// CHECK:STDOUT:   %.Self.as_type.d0cd: type = facet_access_type %.Self.84b [symbolic_self]
// CHECK:STDOUT:   %pattern_type.965: type = pattern_type %facet_type.c22 [concrete]
// CHECK:STDOUT:   %facet_type.eb5: type = facet_type <@Destroy & @DivWith.1, @DivWith.1(%Int32)> [concrete]
// CHECK:STDOUT:   %.Self.203: %facet_type.eb5 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.512: <witness> = lookup_impl_witness %.Self.203, @DivWith.1, @DivWith.1(%Int32) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.232: type = impl_witness_access %DivWith.lookup_impl_witness.512, element0 [symbolic_self]
// CHECK:STDOUT:   %facet_type.529: type = facet_type <@Destroy & @DivWith.1, @DivWith.1(%Int32) where %impl.elem0.232 = %Int32> [concrete]
// CHECK:STDOUT:   %DivWith.type.26e: type = facet_type <@DivWith.1, @DivWith.1(%Int32)> [concrete]
// CHECK:STDOUT:   %DivWith.assoc_type.109: type = assoc_entity_type @DivWith.1, @DivWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %assoc0.d76: %DivWith.assoc_type.109 = assoc_entity element0, imports.%Core.import_ref.14e [concrete]
// CHECK:STDOUT:   %assoc1.4a0: %DivWith.assoc_type.109 = assoc_entity element1, imports.%Core.import_ref.bb2c [concrete]
// CHECK:STDOUT:   %Int16.Op.type.180c8b.11: type = fn_type @Int16.Op.11 [concrete]
// CHECK:STDOUT:   %Int16.Op.c3e49c.11: %Int16.Op.type.180c8b.11 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.510: <witness> = custom_witness (%Int32, %Int16.Op.c3e49c.11), @DivWith.1, @DivWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %facet_value.85a: %facet_type.529 = facet_value %Int16, (%custom_witness.e5b, %custom_witness.510) [concrete]
// CHECK:STDOUT:   %complete_type.d76: <witness> = complete_type_witness %facet_type.eb5 [concrete]
// CHECK:STDOUT:   %.Self.as_type.f8e: type = facet_access_type %.Self.203 [symbolic_self]
// CHECK:STDOUT:   %pattern_type.04c: type = pattern_type %facet_type.529 [concrete]
// CHECK:STDOUT:   %facet_type.1e6: type = facet_type <@Destroy & @ModWith.1, @ModWith.1(%Int32)> [concrete]
// CHECK:STDOUT:   %.Self.0c5: %facet_type.1e6 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.9f7: <witness> = lookup_impl_witness %.Self.0c5, @ModWith.1, @ModWith.1(%Int32) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.919: type = impl_witness_access %ModWith.lookup_impl_witness.9f7, element0 [symbolic_self]
// CHECK:STDOUT:   %facet_type.422: type = facet_type <@Destroy & @ModWith.1, @ModWith.1(%Int32) where %impl.elem0.919 = %Int32> [concrete]
// CHECK:STDOUT:   %ModWith.type.5c4: type = facet_type <@ModWith.1, @ModWith.1(%Int32)> [concrete]
// CHECK:STDOUT:   %ModWith.assoc_type.1db: type = assoc_entity_type @ModWith.1, @ModWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %assoc0.8e2: %ModWith.assoc_type.1db = assoc_entity element0, imports.%Core.import_ref.a0d [concrete]
// CHECK:STDOUT:   %assoc1.826: %ModWith.assoc_type.1db = assoc_entity element1, imports.%Core.import_ref.c05 [concrete]
// CHECK:STDOUT:   %Int16.Op.type.180c8b.12: type = fn_type @Int16.Op.12 [concrete]
// CHECK:STDOUT:   %Int16.Op.c3e49c.12: %Int16.Op.type.180c8b.12 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.a62: <witness> = custom_witness (%Int32, %Int16.Op.c3e49c.12), @ModWith.1, @ModWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %facet_value.dd3: %facet_type.422 = facet_value %Int16, (%custom_witness.e5b, %custom_witness.a62) [concrete]
// CHECK:STDOUT:   %complete_type.1bb: <witness> = complete_type_witness %facet_type.1e6 [concrete]
// CHECK:STDOUT:   %.Self.as_type.ef1: type = facet_access_type %.Self.0c5 [symbolic_self]
// CHECK:STDOUT:   %pattern_type.4d3: type = pattern_type %facet_type.422 [concrete]
// CHECK:STDOUT:   %facet_type.b0a: type = facet_type <@Destroy & @AddWith.1, @AddWith.1(%Int16) where %impl.elem0.cdb = %Int32> [concrete]
// CHECK:STDOUT:   %Int32.Op.type.68b858.2: type = fn_type @Int32.Op.2 [concrete]
// CHECK:STDOUT:   %Int32.Op.6afff1.2: %Int32.Op.type.68b858.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.116: <witness> = custom_witness (%Int32, %Int32.Op.6afff1.2), @AddWith.1, @AddWith.1(%Int16) [concrete]
// CHECK:STDOUT:   %facet_value.741: %facet_type.b0a = facet_value %Int32, (%custom_witness.ce1, %custom_witness.116) [concrete]
// CHECK:STDOUT:   %pattern_type.e21: type = pattern_type %facet_type.b0a [concrete]
// CHECK:STDOUT:   %facet_type.afb: type = facet_type <@Destroy & @SubWith.1, @SubWith.1(%Int16) where %impl.elem0.345 = %Int32> [concrete]
// CHECK:STDOUT:   %Int32.Op.type.68b858.3: type = fn_type @Int32.Op.3 [concrete]
// CHECK:STDOUT:   %Int32.Op.6afff1.3: %Int32.Op.type.68b858.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.904: <witness> = custom_witness (%Int32, %Int32.Op.6afff1.3), @SubWith.1, @SubWith.1(%Int16) [concrete]
// CHECK:STDOUT:   %facet_value.f31: %facet_type.afb = facet_value %Int32, (%custom_witness.ce1, %custom_witness.904) [concrete]
// CHECK:STDOUT:   %pattern_type.6ad: type = pattern_type %facet_type.afb [concrete]
// CHECK:STDOUT:   %facet_type.bfe: type = facet_type <@Destroy & @MulWith.1, @MulWith.1(%Int16) where %impl.elem0.6fb = %Int32> [concrete]
// CHECK:STDOUT:   %Int32.Op.type.68b858.4: type = fn_type @Int32.Op.4 [concrete]
// CHECK:STDOUT:   %Int32.Op.6afff1.4: %Int32.Op.type.68b858.4 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.04f: <witness> = custom_witness (%Int32, %Int32.Op.6afff1.4), @MulWith.1, @MulWith.1(%Int16) [concrete]
// CHECK:STDOUT:   %facet_value.daa: %facet_type.bfe = facet_value %Int32, (%custom_witness.ce1, %custom_witness.04f) [concrete]
// CHECK:STDOUT:   %pattern_type.8a2: type = pattern_type %facet_type.bfe [concrete]
// CHECK:STDOUT:   %facet_type.917: type = facet_type <@Destroy & @DivWith.1, @DivWith.1(%Int16) where %impl.elem0.e43 = %Int32> [concrete]
// CHECK:STDOUT:   %Int32.Op.type.68b858.5: type = fn_type @Int32.Op.5 [concrete]
// CHECK:STDOUT:   %Int32.Op.6afff1.5: %Int32.Op.type.68b858.5 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.53b: <witness> = custom_witness (%Int32, %Int32.Op.6afff1.5), @DivWith.1, @DivWith.1(%Int16) [concrete]
// CHECK:STDOUT:   %facet_value.437: %facet_type.917 = facet_value %Int32, (%custom_witness.ce1, %custom_witness.53b) [concrete]
// CHECK:STDOUT:   %pattern_type.b7f: type = pattern_type %facet_type.917 [concrete]
// CHECK:STDOUT:   %facet_type.d08: type = facet_type <@Destroy & @ModWith.1, @ModWith.1(%Int16) where %impl.elem0.cba = %Int32> [concrete]
// CHECK:STDOUT:   %Int32.Op.type.68b858.6: type = fn_type @Int32.Op.6 [concrete]
// CHECK:STDOUT:   %Int32.Op.6afff1.6: %Int32.Op.type.68b858.6 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.895: <witness> = custom_witness (%Int32, %Int32.Op.6afff1.6), @ModWith.1, @ModWith.1(%Int16) [concrete]
// CHECK:STDOUT:   %facet_value.9f1: %facet_type.d08 = facet_value %Int32, (%custom_witness.ce1, %custom_witness.895) [concrete]
// CHECK:STDOUT:   %pattern_type.a04: type = pattern_type %facet_type.d08 [concrete]
// CHECK:STDOUT:   %Int32.Op.type.68b858.7: type = fn_type @Int32.Op.7 [concrete]
// CHECK:STDOUT:   %Int32.Op.6afff1.7: %Int32.Op.type.68b858.7 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.78d: <witness> = custom_witness (%Int32, %Int32.Op.6afff1.7), @AddWith.1, @AddWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %facet_value.fca: %facet_type.14d = facet_value %Int32, (%custom_witness.ce1, %custom_witness.78d) [concrete]
// CHECK:STDOUT:   %Int32.Op.type.68b858.8: type = fn_type @Int32.Op.8 [concrete]
// CHECK:STDOUT:   %Int32.Op.6afff1.8: %Int32.Op.type.68b858.8 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.64d: <witness> = custom_witness (%Int32, %Int32.Op.6afff1.8), @SubWith.1, @SubWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %facet_value.9a9: %facet_type.f15 = facet_value %Int32, (%custom_witness.ce1, %custom_witness.64d) [concrete]
// CHECK:STDOUT:   %Int32.Op.type.68b858.9: type = fn_type @Int32.Op.9 [concrete]
// CHECK:STDOUT:   %Int32.Op.6afff1.9: %Int32.Op.type.68b858.9 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.7b6d: <witness> = custom_witness (%Int32, %Int32.Op.6afff1.9), @MulWith.1, @MulWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %facet_value.03b: %facet_type.c22 = facet_value %Int32, (%custom_witness.ce1, %custom_witness.7b6d) [concrete]
// CHECK:STDOUT:   %Int32.Op.type.68b858.10: type = fn_type @Int32.Op.10 [concrete]
// CHECK:STDOUT:   %Int32.Op.6afff1.10: %Int32.Op.type.68b858.10 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.db6: <witness> = custom_witness (%Int32, %Int32.Op.6afff1.10), @DivWith.1, @DivWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %facet_value.a2b: %facet_type.529 = facet_value %Int32, (%custom_witness.ce1, %custom_witness.db6) [concrete]
// CHECK:STDOUT:   %Int32.Op.type.68b858.11: type = fn_type @Int32.Op.11 [concrete]
// CHECK:STDOUT:   %Int32.Op.6afff1.11: %Int32.Op.type.68b858.11 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.e01: <witness> = custom_witness (%Int32, %Int32.Op.6afff1.11), @ModWith.1, @ModWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %facet_value.a4c: %facet_type.422 = facet_value %Int32, (%custom_witness.ce1, %custom_witness.e01) [concrete]
// CHECK:STDOUT:   %Int64.Op.type.445ce8.1: type = fn_type @Int64.Op.1 [concrete]
// CHECK:STDOUT:   %Int64.Op.b941da.1: %Int64.Op.type.445ce8.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.d7e: <witness> = custom_witness (%Int64.Op.b941da.1), @Destroy [concrete]
// CHECK:STDOUT:   %Destroy.facet.b0a: %Destroy.type = facet_value %Int64, (%custom_witness.d7e) [concrete]
// CHECK:STDOUT:   %facet_type.221: type = facet_type <@Destroy & @AddWith.1, @AddWith.1(%Int64)> [concrete]
// CHECK:STDOUT:   %.Self.954: %facet_type.221 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.076: <witness> = lookup_impl_witness %.Self.954, @AddWith.1, @AddWith.1(%Int64) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.d09: type = impl_witness_access %AddWith.lookup_impl_witness.076, element0 [symbolic_self]
// CHECK:STDOUT:   %facet_type.7b1: type = facet_type <@Destroy & @AddWith.1, @AddWith.1(%Int64) where %impl.elem0.d09 = %Int64> [concrete]
// CHECK:STDOUT:   %AddWith.type.ee5: type = facet_type <@AddWith.1, @AddWith.1(%Int64)> [concrete]
// CHECK:STDOUT:   %AddWith.assoc_type.330: type = assoc_entity_type @AddWith.1, @AddWith.1(%Int64) [concrete]
// CHECK:STDOUT:   %assoc0.1fe: %AddWith.assoc_type.330 = assoc_entity element0, imports.%Core.import_ref.6e9 [concrete]
// CHECK:STDOUT:   %assoc1.cd6: %AddWith.assoc_type.330 = assoc_entity element1, imports.%Core.import_ref.1b1 [concrete]
// CHECK:STDOUT:   %Op.type.67547d.6: type = fn_type @Op.6 [concrete]
// CHECK:STDOUT:   %Op.8fb557.6: %Op.type.67547d.6 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.1b7: <witness> = custom_witness (%Int64, %Op.8fb557.6), @AddWith.1, @AddWith.1(%Int64) [concrete]
// CHECK:STDOUT:   %facet_value.054: %facet_type.7b1 = facet_value %Int32, (%custom_witness.ce1, %custom_witness.1b7) [concrete]
// CHECK:STDOUT:   %complete_type.7fa: <witness> = complete_type_witness %facet_type.221 [concrete]
// CHECK:STDOUT:   %.Self.as_type.083: type = facet_access_type %.Self.954 [symbolic_self]
// CHECK:STDOUT:   %pattern_type.aa1: type = pattern_type %facet_type.7b1 [concrete]
// CHECK:STDOUT:   %facet_type.660: type = facet_type <@Destroy & @SubWith.1, @SubWith.1(%Int64)> [concrete]
// CHECK:STDOUT:   %.Self.ea7: %facet_type.660 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.894: <witness> = lookup_impl_witness %.Self.ea7, @SubWith.1, @SubWith.1(%Int64) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.9dd: type = impl_witness_access %SubWith.lookup_impl_witness.894, element0 [symbolic_self]
// CHECK:STDOUT:   %facet_type.17a: type = facet_type <@Destroy & @SubWith.1, @SubWith.1(%Int64) where %impl.elem0.9dd = %Int64> [concrete]
// CHECK:STDOUT:   %SubWith.type.b41: type = facet_type <@SubWith.1, @SubWith.1(%Int64)> [concrete]
// CHECK:STDOUT:   %SubWith.assoc_type.b7b: type = assoc_entity_type @SubWith.1, @SubWith.1(%Int64) [concrete]
// CHECK:STDOUT:   %assoc0.979: %SubWith.assoc_type.b7b = assoc_entity element0, imports.%Core.import_ref.7e5 [concrete]
// CHECK:STDOUT:   %assoc1.80d: %SubWith.assoc_type.b7b = assoc_entity element1, imports.%Core.import_ref.abb [concrete]
// CHECK:STDOUT:   %Op.type.67547d.7: type = fn_type @Op.7 [concrete]
// CHECK:STDOUT:   %Op.8fb557.7: %Op.type.67547d.7 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.cab: <witness> = custom_witness (%Int64, %Op.8fb557.7), @SubWith.1, @SubWith.1(%Int64) [concrete]
// CHECK:STDOUT:   %facet_value.f7f: %facet_type.17a = facet_value %Int32, (%custom_witness.ce1, %custom_witness.cab) [concrete]
// CHECK:STDOUT:   %complete_type.6f3: <witness> = complete_type_witness %facet_type.660 [concrete]
// CHECK:STDOUT:   %.Self.as_type.d4d: type = facet_access_type %.Self.ea7 [symbolic_self]
// CHECK:STDOUT:   %pattern_type.30e: type = pattern_type %facet_type.17a [concrete]
// CHECK:STDOUT:   %facet_type.60a: type = facet_type <@Destroy & @MulWith.1, @MulWith.1(%Int64)> [concrete]
// CHECK:STDOUT:   %.Self.39e: %facet_type.60a = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.a30: <witness> = lookup_impl_witness %.Self.39e, @MulWith.1, @MulWith.1(%Int64) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.3f6: type = impl_witness_access %MulWith.lookup_impl_witness.a30, element0 [symbolic_self]
// CHECK:STDOUT:   %facet_type.9d2: type = facet_type <@Destroy & @MulWith.1, @MulWith.1(%Int64) where %impl.elem0.3f6 = %Int64> [concrete]
// CHECK:STDOUT:   %MulWith.type.166: type = facet_type <@MulWith.1, @MulWith.1(%Int64)> [concrete]
// CHECK:STDOUT:   %MulWith.assoc_type.cbc: type = assoc_entity_type @MulWith.1, @MulWith.1(%Int64) [concrete]
// CHECK:STDOUT:   %assoc0.c58: %MulWith.assoc_type.cbc = assoc_entity element0, imports.%Core.import_ref.d9b [concrete]
// CHECK:STDOUT:   %assoc1.279: %MulWith.assoc_type.cbc = assoc_entity element1, imports.%Core.import_ref.abc [concrete]
// CHECK:STDOUT:   %Op.type.67547d.8: type = fn_type @Op.8 [concrete]
// CHECK:STDOUT:   %Op.8fb557.8: %Op.type.67547d.8 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.f53: <witness> = custom_witness (%Int64, %Op.8fb557.8), @MulWith.1, @MulWith.1(%Int64) [concrete]
// CHECK:STDOUT:   %facet_value.823: %facet_type.9d2 = facet_value %Int32, (%custom_witness.ce1, %custom_witness.f53) [concrete]
// CHECK:STDOUT:   %complete_type.7a4: <witness> = complete_type_witness %facet_type.60a [concrete]
// CHECK:STDOUT:   %.Self.as_type.604: type = facet_access_type %.Self.39e [symbolic_self]
// CHECK:STDOUT:   %pattern_type.eb5: type = pattern_type %facet_type.9d2 [concrete]
// CHECK:STDOUT:   %facet_type.f02: type = facet_type <@Destroy & @DivWith.1, @DivWith.1(%Int64)> [concrete]
// CHECK:STDOUT:   %.Self.ad5: %facet_type.f02 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.cc2: <witness> = lookup_impl_witness %.Self.ad5, @DivWith.1, @DivWith.1(%Int64) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.2d2: type = impl_witness_access %DivWith.lookup_impl_witness.cc2, element0 [symbolic_self]
// CHECK:STDOUT:   %facet_type.6aa: type = facet_type <@Destroy & @DivWith.1, @DivWith.1(%Int64) where %impl.elem0.2d2 = %Int64> [concrete]
// CHECK:STDOUT:   %DivWith.type.70f: type = facet_type <@DivWith.1, @DivWith.1(%Int64)> [concrete]
// CHECK:STDOUT:   %DivWith.assoc_type.03e: type = assoc_entity_type @DivWith.1, @DivWith.1(%Int64) [concrete]
// CHECK:STDOUT:   %assoc0.a25: %DivWith.assoc_type.03e = assoc_entity element0, imports.%Core.import_ref.14e [concrete]
// CHECK:STDOUT:   %assoc1.f42: %DivWith.assoc_type.03e = assoc_entity element1, imports.%Core.import_ref.bb2c [concrete]
// CHECK:STDOUT:   %Op.type.67547d.9: type = fn_type @Op.9 [concrete]
// CHECK:STDOUT:   %Op.8fb557.9: %Op.type.67547d.9 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.bca: <witness> = custom_witness (%Int64, %Op.8fb557.9), @DivWith.1, @DivWith.1(%Int64) [concrete]
// CHECK:STDOUT:   %facet_value.cba: %facet_type.6aa = facet_value %Int32, (%custom_witness.ce1, %custom_witness.bca) [concrete]
// CHECK:STDOUT:   %complete_type.e31: <witness> = complete_type_witness %facet_type.f02 [concrete]
// CHECK:STDOUT:   %.Self.as_type.245: type = facet_access_type %.Self.ad5 [symbolic_self]
// CHECK:STDOUT:   %pattern_type.c59: type = pattern_type %facet_type.6aa [concrete]
// CHECK:STDOUT:   %facet_type.b13: type = facet_type <@Destroy & @ModWith.1, @ModWith.1(%Int64)> [concrete]
// CHECK:STDOUT:   %.Self.856: %facet_type.b13 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.81f: <witness> = lookup_impl_witness %.Self.856, @ModWith.1, @ModWith.1(%Int64) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.369: type = impl_witness_access %ModWith.lookup_impl_witness.81f, element0 [symbolic_self]
// CHECK:STDOUT:   %facet_type.41c: type = facet_type <@Destroy & @ModWith.1, @ModWith.1(%Int64) where %impl.elem0.369 = %Int64> [concrete]
// CHECK:STDOUT:   %ModWith.type.a92: type = facet_type <@ModWith.1, @ModWith.1(%Int64)> [concrete]
// CHECK:STDOUT:   %ModWith.assoc_type.f50: type = assoc_entity_type @ModWith.1, @ModWith.1(%Int64) [concrete]
// CHECK:STDOUT:   %assoc0.c89: %ModWith.assoc_type.f50 = assoc_entity element0, imports.%Core.import_ref.a0d [concrete]
// CHECK:STDOUT:   %assoc1.a80: %ModWith.assoc_type.f50 = assoc_entity element1, imports.%Core.import_ref.c05 [concrete]
// CHECK:STDOUT:   %Op.type.67547d.10: type = fn_type @Op.10 [concrete]
// CHECK:STDOUT:   %Op.8fb557.10: %Op.type.67547d.10 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.1b0: <witness> = custom_witness (%Int64, %Op.8fb557.10), @ModWith.1, @ModWith.1(%Int64) [concrete]
// CHECK:STDOUT:   %facet_value.403: %facet_type.41c = facet_value %Int32, (%custom_witness.ce1, %custom_witness.1b0) [concrete]
// CHECK:STDOUT:   %complete_type.780: <witness> = complete_type_witness %facet_type.b13 [concrete]
// CHECK:STDOUT:   %.Self.as_type.6bd: type = facet_access_type %.Self.856 [symbolic_self]
// CHECK:STDOUT:   %pattern_type.4ab: type = pattern_type %facet_type.41c [concrete]
// CHECK:STDOUT:   %facet_type.a53: type = facet_type <@Destroy & @AddWith.1, @AddWith.1(%Int32) where %impl.elem0.cee = %Int64> [concrete]
// CHECK:STDOUT:   %Op.type.67547d.11: type = fn_type @Op.11 [concrete]
// CHECK:STDOUT:   %Op.8fb557.11: %Op.type.67547d.11 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.fc1: <witness> = custom_witness (%Int64, %Op.8fb557.11), @AddWith.1, @AddWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %facet_value.8dc: %facet_type.a53 = facet_value %Int64, (%custom_witness.d7e, %custom_witness.fc1) [concrete]
// CHECK:STDOUT:   %pattern_type.7f1: type = pattern_type %facet_type.a53 [concrete]
// CHECK:STDOUT:   %facet_type.364: type = facet_type <@Destroy & @SubWith.1, @SubWith.1(%Int32) where %impl.elem0.edd = %Int64> [concrete]
// CHECK:STDOUT:   %Op.type.67547d.12: type = fn_type @Op.12 [concrete]
// CHECK:STDOUT:   %Op.8fb557.12: %Op.type.67547d.12 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.31e: <witness> = custom_witness (%Int64, %Op.8fb557.12), @SubWith.1, @SubWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %facet_value.c40: %facet_type.364 = facet_value %Int64, (%custom_witness.d7e, %custom_witness.31e) [concrete]
// CHECK:STDOUT:   %pattern_type.c45: type = pattern_type %facet_type.364 [concrete]
// CHECK:STDOUT:   %facet_type.d7d: type = facet_type <@Destroy & @MulWith.1, @MulWith.1(%Int32) where %impl.elem0.e63 = %Int64> [concrete]
// CHECK:STDOUT:   %Op.type.67547d.13: type = fn_type @Op.13 [concrete]
// CHECK:STDOUT:   %Op.8fb557.13: %Op.type.67547d.13 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.709: <witness> = custom_witness (%Int64, %Op.8fb557.13), @MulWith.1, @MulWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %facet_value.9d5: %facet_type.d7d = facet_value %Int64, (%custom_witness.d7e, %custom_witness.709) [concrete]
// CHECK:STDOUT:   %pattern_type.450: type = pattern_type %facet_type.d7d [concrete]
// CHECK:STDOUT:   %facet_type.a9f: type = facet_type <@Destroy & @DivWith.1, @DivWith.1(%Int32) where %impl.elem0.232 = %Int64> [concrete]
// CHECK:STDOUT:   %Op.type.67547d.14: type = fn_type @Op.14 [concrete]
// CHECK:STDOUT:   %Op.8fb557.14: %Op.type.67547d.14 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.47b: <witness> = custom_witness (%Int64, %Op.8fb557.14), @DivWith.1, @DivWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %facet_value.c21: %facet_type.a9f = facet_value %Int64, (%custom_witness.d7e, %custom_witness.47b) [concrete]
// CHECK:STDOUT:   %pattern_type.0b0: type = pattern_type %facet_type.a9f [concrete]
// CHECK:STDOUT:   %facet_type.f0f: type = facet_type <@Destroy & @ModWith.1, @ModWith.1(%Int32) where %impl.elem0.919 = %Int64> [concrete]
// CHECK:STDOUT:   %Op.type.67547d.15: type = fn_type @Op.15 [concrete]
// CHECK:STDOUT:   %Op.8fb557.15: %Op.type.67547d.15 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.ad6: <witness> = custom_witness (%Int64, %Op.8fb557.15), @ModWith.1, @ModWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %facet_value.766: %facet_type.f0f = facet_value %Int64, (%custom_witness.d7e, %custom_witness.ad6) [concrete]
// CHECK:STDOUT:   %pattern_type.bd2: type = pattern_type %facet_type.f0f [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.f16: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.b0a) [concrete]
// CHECK:STDOUT:   %.db1: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.f16, %Destroy.facet.b0a [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.2e1: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.c06) [concrete]
// CHECK:STDOUT:   %.e6c: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.2e1, %Destroy.facet.c06 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.833: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.e06) [concrete]
// CHECK:STDOUT:   %.368: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.833, %Destroy.facet.e06 [concrete]
// CHECK:STDOUT:   %U.67d: type = symbolic_binding U, 0 [symbolic]
// CHECK:STDOUT:   %AddAssignWith.type.4d3b38.1: type = facet_type <@AddAssignWith.1, @AddAssignWith.1(%Other)> [symbolic]
// CHECK:STDOUT:   %Self.a00362.1: %AddAssignWith.type.4d3b38.1 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %AddAssignWith.assoc_type.ca2a80.1: type = assoc_entity_type @AddAssignWith.1, @AddAssignWith.1(%Other) [symbolic]
// CHECK:STDOUT:   %AddAssignWith.WithSelf.Op.type.df0722.1: type = fn_type @AddAssignWith.WithSelf.Op, @AddAssignWith.WithSelf(%Other, %Self.a00362.1) [symbolic]
// CHECK:STDOUT:   %AddAssignWith.WithSelf.Op.b33928.1: %AddAssignWith.WithSelf.Op.type.df0722.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %AddAssignWith.type.4d3b38.2: type = facet_type <@AddAssignWith.1, @AddAssignWith.1(%U.67d)> [symbolic]
// CHECK:STDOUT:   %pattern_type.092: type = pattern_type %AddAssignWith.type.4d3b38.2 [symbolic]
// CHECK:STDOUT:   %T.a00: %AddAssignWith.type.4d3b38.2 = symbolic_binding T, 1 [symbolic]
// CHECK:STDOUT:   %AddAssignWith.assoc_type.ca2a80.2: type = assoc_entity_type @AddAssignWith.1, @AddAssignWith.1(%U.67d) [symbolic]
// CHECK:STDOUT:   %assoc0.f6c0ab.2: %AddAssignWith.assoc_type.ca2a80.2 = assoc_entity element0, imports.%Core.import_ref.7e1 [symbolic]
// CHECK:STDOUT:   %T.as_type.191: type = facet_access_type %T.a00 [symbolic]
// CHECK:STDOUT:   %pattern_type.03496c.2: type = pattern_type %T.as_type.191 [symbolic]
// CHECK:STDOUT:   %pattern_type.51d1c4.4: type = pattern_type %U.67d [symbolic]
// CHECK:STDOUT:   %require_complete.cb0: <witness> = require_complete_type %AddAssignWith.type.4d3b38.2 [symbolic]
// CHECK:STDOUT:   %assoc0.5d2: %AddAssignWith.assoc_type.ca2a80.1 = assoc_entity element0, imports.%Core.import_ref.f18 [symbolic]
// CHECK:STDOUT:   %AddAssignWith.lookup_impl_witness: <witness> = lookup_impl_witness %T.a00, @AddAssignWith.1, @AddAssignWith.1(%U.67d) [symbolic]
// CHECK:STDOUT:   %AddAssignWith.WithSelf.Op.type.df0722.3: type = fn_type @AddAssignWith.WithSelf.Op, @AddAssignWith.WithSelf(%U.67d, %T.a00) [symbolic]
// CHECK:STDOUT:   %.8f6: type = fn_type_with_self_type %AddAssignWith.WithSelf.Op.type.df0722.3, %T.a00 [symbolic]
// CHECK:STDOUT:   %impl.elem0.c7b: %.8f6 = impl_witness_access %AddAssignWith.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.fca: <specific function> = specific_impl_function %impl.elem0.c7b, @AddAssignWith.WithSelf.Op(%U.67d, %T.a00) [symbolic]
// CHECK:STDOUT:   %SubAssignWith.type.969cbb.1: type = facet_type <@SubAssignWith.1, @SubAssignWith.1(%Other)> [symbolic]
// CHECK:STDOUT:   %Self.1de093.1: %SubAssignWith.type.969cbb.1 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %SubAssignWith.assoc_type.4cb9a1.1: type = assoc_entity_type @SubAssignWith.1, @SubAssignWith.1(%Other) [symbolic]
// CHECK:STDOUT:   %SubAssignWith.WithSelf.Op.type.317c02.1: type = fn_type @SubAssignWith.WithSelf.Op, @SubAssignWith.WithSelf(%Other, %Self.1de093.1) [symbolic]
// CHECK:STDOUT:   %SubAssignWith.WithSelf.Op.26e13f.1: %SubAssignWith.WithSelf.Op.type.317c02.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %SubAssignWith.type.969cbb.2: type = facet_type <@SubAssignWith.1, @SubAssignWith.1(%U.67d)> [symbolic]
// CHECK:STDOUT:   %pattern_type.abb: type = pattern_type %SubAssignWith.type.969cbb.2 [symbolic]
// CHECK:STDOUT:   %T.1de: %SubAssignWith.type.969cbb.2 = symbolic_binding T, 1 [symbolic]
// CHECK:STDOUT:   %SubAssignWith.assoc_type.4cb9a1.2: type = assoc_entity_type @SubAssignWith.1, @SubAssignWith.1(%U.67d) [symbolic]
// CHECK:STDOUT:   %assoc0.6115cc.2: %SubAssignWith.assoc_type.4cb9a1.2 = assoc_entity element0, imports.%Core.import_ref.5b4 [symbolic]
// CHECK:STDOUT:   %T.as_type.b8f: type = facet_access_type %T.1de [symbolic]
// CHECK:STDOUT:   %pattern_type.2f1979.2: type = pattern_type %T.as_type.b8f [symbolic]
// CHECK:STDOUT:   %require_complete.22d3: <witness> = require_complete_type %SubAssignWith.type.969cbb.2 [symbolic]
// CHECK:STDOUT:   %assoc0.fd1: %SubAssignWith.assoc_type.4cb9a1.1 = assoc_entity element0, imports.%Core.import_ref.a26 [symbolic]
// CHECK:STDOUT:   %SubAssignWith.lookup_impl_witness: <witness> = lookup_impl_witness %T.1de, @SubAssignWith.1, @SubAssignWith.1(%U.67d) [symbolic]
// CHECK:STDOUT:   %SubAssignWith.WithSelf.Op.type.317c02.3: type = fn_type @SubAssignWith.WithSelf.Op, @SubAssignWith.WithSelf(%U.67d, %T.1de) [symbolic]
// CHECK:STDOUT:   %.307: type = fn_type_with_self_type %SubAssignWith.WithSelf.Op.type.317c02.3, %T.1de [symbolic]
// CHECK:STDOUT:   %impl.elem0.e10: %.307 = impl_witness_access %SubAssignWith.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.c8d: <specific function> = specific_impl_function %impl.elem0.e10, @SubAssignWith.WithSelf.Op(%U.67d, %T.1de) [symbolic]
// CHECK:STDOUT:   %MulAssignWith.type.ccc9e0.1: type = facet_type <@MulAssignWith.1, @MulAssignWith.1(%Other)> [symbolic]
// CHECK:STDOUT:   %Self.059e93.1: %MulAssignWith.type.ccc9e0.1 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %MulAssignWith.assoc_type.595561.1: type = assoc_entity_type @MulAssignWith.1, @MulAssignWith.1(%Other) [symbolic]
// CHECK:STDOUT:   %MulAssignWith.WithSelf.Op.type.31cf70.1: type = fn_type @MulAssignWith.WithSelf.Op, @MulAssignWith.WithSelf(%Other, %Self.059e93.1) [symbolic]
// CHECK:STDOUT:   %MulAssignWith.WithSelf.Op.cdf083.1: %MulAssignWith.WithSelf.Op.type.31cf70.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %MulAssignWith.type.ccc9e0.2: type = facet_type <@MulAssignWith.1, @MulAssignWith.1(%U.67d)> [symbolic]
// CHECK:STDOUT:   %pattern_type.043: type = pattern_type %MulAssignWith.type.ccc9e0.2 [symbolic]
// CHECK:STDOUT:   %T.059: %MulAssignWith.type.ccc9e0.2 = symbolic_binding T, 1 [symbolic]
// CHECK:STDOUT:   %MulAssignWith.assoc_type.595561.2: type = assoc_entity_type @MulAssignWith.1, @MulAssignWith.1(%U.67d) [symbolic]
// CHECK:STDOUT:   %assoc0.71e6fb.2: %MulAssignWith.assoc_type.595561.2 = assoc_entity element0, imports.%Core.import_ref.69b [symbolic]
// CHECK:STDOUT:   %T.as_type.6a0: type = facet_access_type %T.059 [symbolic]
// CHECK:STDOUT:   %pattern_type.587662.2: type = pattern_type %T.as_type.6a0 [symbolic]
// CHECK:STDOUT:   %require_complete.6dc: <witness> = require_complete_type %MulAssignWith.type.ccc9e0.2 [symbolic]
// CHECK:STDOUT:   %assoc0.fc1: %MulAssignWith.assoc_type.595561.1 = assoc_entity element0, imports.%Core.import_ref.2e6 [symbolic]
// CHECK:STDOUT:   %MulAssignWith.lookup_impl_witness: <witness> = lookup_impl_witness %T.059, @MulAssignWith.1, @MulAssignWith.1(%U.67d) [symbolic]
// CHECK:STDOUT:   %MulAssignWith.WithSelf.Op.type.31cf70.3: type = fn_type @MulAssignWith.WithSelf.Op, @MulAssignWith.WithSelf(%U.67d, %T.059) [symbolic]
// CHECK:STDOUT:   %.1b8: type = fn_type_with_self_type %MulAssignWith.WithSelf.Op.type.31cf70.3, %T.059 [symbolic]
// CHECK:STDOUT:   %impl.elem0.cd0: %.1b8 = impl_witness_access %MulAssignWith.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.397: <specific function> = specific_impl_function %impl.elem0.cd0, @MulAssignWith.WithSelf.Op(%U.67d, %T.059) [symbolic]
// CHECK:STDOUT:   %DivAssignWith.type.be08dd.1: type = facet_type <@DivAssignWith.1, @DivAssignWith.1(%Other)> [symbolic]
// CHECK:STDOUT:   %Self.aa8835.1: %DivAssignWith.type.be08dd.1 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %DivAssignWith.assoc_type.8b4b5b.1: type = assoc_entity_type @DivAssignWith.1, @DivAssignWith.1(%Other) [symbolic]
// CHECK:STDOUT:   %DivAssignWith.WithSelf.Op.type.ded5d7.1: type = fn_type @DivAssignWith.WithSelf.Op, @DivAssignWith.WithSelf(%Other, %Self.aa8835.1) [symbolic]
// CHECK:STDOUT:   %DivAssignWith.WithSelf.Op.6f2821.1: %DivAssignWith.WithSelf.Op.type.ded5d7.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %DivAssignWith.type.be08dd.2: type = facet_type <@DivAssignWith.1, @DivAssignWith.1(%U.67d)> [symbolic]
// CHECK:STDOUT:   %pattern_type.8e3: type = pattern_type %DivAssignWith.type.be08dd.2 [symbolic]
// CHECK:STDOUT:   %T.aa8: %DivAssignWith.type.be08dd.2 = symbolic_binding T, 1 [symbolic]
// CHECK:STDOUT:   %DivAssignWith.assoc_type.8b4b5b.2: type = assoc_entity_type @DivAssignWith.1, @DivAssignWith.1(%U.67d) [symbolic]
// CHECK:STDOUT:   %assoc0.399004.2: %DivAssignWith.assoc_type.8b4b5b.2 = assoc_entity element0, imports.%Core.import_ref.799 [symbolic]
// CHECK:STDOUT:   %T.as_type.efd: type = facet_access_type %T.aa8 [symbolic]
// CHECK:STDOUT:   %pattern_type.593823.2: type = pattern_type %T.as_type.efd [symbolic]
// CHECK:STDOUT:   %require_complete.542: <witness> = require_complete_type %DivAssignWith.type.be08dd.2 [symbolic]
// CHECK:STDOUT:   %assoc0.b19: %DivAssignWith.assoc_type.8b4b5b.1 = assoc_entity element0, imports.%Core.import_ref.7f9 [symbolic]
// CHECK:STDOUT:   %DivAssignWith.lookup_impl_witness: <witness> = lookup_impl_witness %T.aa8, @DivAssignWith.1, @DivAssignWith.1(%U.67d) [symbolic]
// CHECK:STDOUT:   %DivAssignWith.WithSelf.Op.type.ded5d7.3: type = fn_type @DivAssignWith.WithSelf.Op, @DivAssignWith.WithSelf(%U.67d, %T.aa8) [symbolic]
// CHECK:STDOUT:   %.607: type = fn_type_with_self_type %DivAssignWith.WithSelf.Op.type.ded5d7.3, %T.aa8 [symbolic]
// CHECK:STDOUT:   %impl.elem0.46f: %.607 = impl_witness_access %DivAssignWith.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.666: <specific function> = specific_impl_function %impl.elem0.46f, @DivAssignWith.WithSelf.Op(%U.67d, %T.aa8) [symbolic]
// CHECK:STDOUT:   %ModAssignWith.type.c65b77.1: type = facet_type <@ModAssignWith.1, @ModAssignWith.1(%Other)> [symbolic]
// CHECK:STDOUT:   %Self.d35c46.1: %ModAssignWith.type.c65b77.1 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %ModAssignWith.assoc_type.1462b9.1: type = assoc_entity_type @ModAssignWith.1, @ModAssignWith.1(%Other) [symbolic]
// CHECK:STDOUT:   %ModAssignWith.WithSelf.Op.type.5452e9.1: type = fn_type @ModAssignWith.WithSelf.Op, @ModAssignWith.WithSelf(%Other, %Self.d35c46.1) [symbolic]
// CHECK:STDOUT:   %ModAssignWith.WithSelf.Op.c5cc9b.1: %ModAssignWith.WithSelf.Op.type.5452e9.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %ModAssignWith.type.c65b77.2: type = facet_type <@ModAssignWith.1, @ModAssignWith.1(%U.67d)> [symbolic]
// CHECK:STDOUT:   %pattern_type.26a: type = pattern_type %ModAssignWith.type.c65b77.2 [symbolic]
// CHECK:STDOUT:   %T.d35: %ModAssignWith.type.c65b77.2 = symbolic_binding T, 1 [symbolic]
// CHECK:STDOUT:   %ModAssignWith.assoc_type.1462b9.2: type = assoc_entity_type @ModAssignWith.1, @ModAssignWith.1(%U.67d) [symbolic]
// CHECK:STDOUT:   %assoc0.378445.2: %ModAssignWith.assoc_type.1462b9.2 = assoc_entity element0, imports.%Core.import_ref.51c [symbolic]
// CHECK:STDOUT:   %T.as_type.4c2: type = facet_access_type %T.d35 [symbolic]
// CHECK:STDOUT:   %pattern_type.6ec6ce.2: type = pattern_type %T.as_type.4c2 [symbolic]
// CHECK:STDOUT:   %require_complete.8bf: <witness> = require_complete_type %ModAssignWith.type.c65b77.2 [symbolic]
// CHECK:STDOUT:   %assoc0.0e6: %ModAssignWith.assoc_type.1462b9.1 = assoc_entity element0, imports.%Core.import_ref.fa5 [symbolic]
// CHECK:STDOUT:   %ModAssignWith.lookup_impl_witness: <witness> = lookup_impl_witness %T.d35, @ModAssignWith.1, @ModAssignWith.1(%U.67d) [symbolic]
// CHECK:STDOUT:   %ModAssignWith.WithSelf.Op.type.5452e9.3: type = fn_type @ModAssignWith.WithSelf.Op, @ModAssignWith.WithSelf(%U.67d, %T.d35) [symbolic]
// CHECK:STDOUT:   %.dfa: type = fn_type_with_self_type %ModAssignWith.WithSelf.Op.type.5452e9.3, %T.d35 [symbolic]
// CHECK:STDOUT:   %impl.elem0.18c: %.dfa = impl_witness_access %ModAssignWith.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.28f: <specific function> = specific_impl_function %impl.elem0.18c, @ModAssignWith.WithSelf.Op(%U.67d, %T.d35) [symbolic]
// CHECK:STDOUT:   %AddAssignWith.type.704: type = facet_type <@AddAssignWith.1, @AddAssignWith.1(%i16)> [concrete]
// CHECK:STDOUT:   %AddAssignWith.assoc_type.23b: type = assoc_entity_type @AddAssignWith.1, @AddAssignWith.1(%i16) [concrete]
// CHECK:STDOUT:   %assoc0.08d: %AddAssignWith.assoc_type.23b = assoc_entity element0, imports.%Core.import_ref.7e1 [concrete]
// CHECK:STDOUT:   %Int16.Op.type.180c8b.13: type = fn_type @Int16.Op.13 [concrete]
// CHECK:STDOUT:   %Int16.Op.c3e49c.13: %Int16.Op.type.180c8b.13 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.6da: <witness> = custom_witness (%Int16.Op.c3e49c.13), @AddAssignWith.1, @AddAssignWith.1(%i16) [concrete]
// CHECK:STDOUT:   %AddAssignWith.facet.23f: %AddAssignWith.type.704 = facet_value %Int16, (%custom_witness.6da) [concrete]
// CHECK:STDOUT:   %pattern_type.139: type = pattern_type %AddAssignWith.type.704 [concrete]
// CHECK:STDOUT:   %SubAssignWith.type.fed: type = facet_type <@SubAssignWith.1, @SubAssignWith.1(%i16)> [concrete]
// CHECK:STDOUT:   %SubAssignWith.assoc_type.81a: type = assoc_entity_type @SubAssignWith.1, @SubAssignWith.1(%i16) [concrete]
// CHECK:STDOUT:   %assoc0.0a4: %SubAssignWith.assoc_type.81a = assoc_entity element0, imports.%Core.import_ref.5b4 [concrete]
// CHECK:STDOUT:   %Int16.Op.type.180c8b.14: type = fn_type @Int16.Op.14 [concrete]
// CHECK:STDOUT:   %Int16.Op.c3e49c.14: %Int16.Op.type.180c8b.14 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.509: <witness> = custom_witness (%Int16.Op.c3e49c.14), @SubAssignWith.1, @SubAssignWith.1(%i16) [concrete]
// CHECK:STDOUT:   %SubAssignWith.facet.3e7: %SubAssignWith.type.fed = facet_value %Int16, (%custom_witness.509) [concrete]
// CHECK:STDOUT:   %pattern_type.912: type = pattern_type %SubAssignWith.type.fed [concrete]
// CHECK:STDOUT:   %MulAssignWith.type.6c0: type = facet_type <@MulAssignWith.1, @MulAssignWith.1(%i16)> [concrete]
// CHECK:STDOUT:   %MulAssignWith.assoc_type.af2: type = assoc_entity_type @MulAssignWith.1, @MulAssignWith.1(%i16) [concrete]
// CHECK:STDOUT:   %assoc0.e50: %MulAssignWith.assoc_type.af2 = assoc_entity element0, imports.%Core.import_ref.69b [concrete]
// CHECK:STDOUT:   %Int16.Op.type.180c8b.15: type = fn_type @Int16.Op.15 [concrete]
// CHECK:STDOUT:   %Int16.Op.c3e49c.15: %Int16.Op.type.180c8b.15 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.a1f: <witness> = custom_witness (%Int16.Op.c3e49c.15), @MulAssignWith.1, @MulAssignWith.1(%i16) [concrete]
// CHECK:STDOUT:   %MulAssignWith.facet.78f: %MulAssignWith.type.6c0 = facet_value %Int16, (%custom_witness.a1f) [concrete]
// CHECK:STDOUT:   %pattern_type.578: type = pattern_type %MulAssignWith.type.6c0 [concrete]
// CHECK:STDOUT:   %DivAssignWith.type.527: type = facet_type <@DivAssignWith.1, @DivAssignWith.1(%i16)> [concrete]
// CHECK:STDOUT:   %DivAssignWith.assoc_type.5b8: type = assoc_entity_type @DivAssignWith.1, @DivAssignWith.1(%i16) [concrete]
// CHECK:STDOUT:   %assoc0.ced: %DivAssignWith.assoc_type.5b8 = assoc_entity element0, imports.%Core.import_ref.799 [concrete]
// CHECK:STDOUT:   %Int16.Op.type.180c8b.16: type = fn_type @Int16.Op.16 [concrete]
// CHECK:STDOUT:   %Int16.Op.c3e49c.16: %Int16.Op.type.180c8b.16 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.deb: <witness> = custom_witness (%Int16.Op.c3e49c.16), @DivAssignWith.1, @DivAssignWith.1(%i16) [concrete]
// CHECK:STDOUT:   %DivAssignWith.facet.b0d: %DivAssignWith.type.527 = facet_value %Int16, (%custom_witness.deb) [concrete]
// CHECK:STDOUT:   %pattern_type.5f7: type = pattern_type %DivAssignWith.type.527 [concrete]
// CHECK:STDOUT:   %ModAssignWith.type.ead: type = facet_type <@ModAssignWith.1, @ModAssignWith.1(%i16)> [concrete]
// CHECK:STDOUT:   %ModAssignWith.assoc_type.556: type = assoc_entity_type @ModAssignWith.1, @ModAssignWith.1(%i16) [concrete]
// CHECK:STDOUT:   %assoc0.b4c: %ModAssignWith.assoc_type.556 = assoc_entity element0, imports.%Core.import_ref.51c [concrete]
// CHECK:STDOUT:   %Int16.Op.type.180c8b.17: type = fn_type @Int16.Op.17 [concrete]
// CHECK:STDOUT:   %Int16.Op.c3e49c.17: %Int16.Op.type.180c8b.17 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.c26: <witness> = custom_witness (%Int16.Op.c3e49c.17), @ModAssignWith.1, @ModAssignWith.1(%i16) [concrete]
// CHECK:STDOUT:   %ModAssignWith.facet.ada: %ModAssignWith.type.ead = facet_value %Int16, (%custom_witness.c26) [concrete]
// CHECK:STDOUT:   %pattern_type.bf6: type = pattern_type %ModAssignWith.type.ead [concrete]
// CHECK:STDOUT:   %AddAssignWith.type.a71: type = facet_type <@AddAssignWith.1, @AddAssignWith.1(%Int32)> [concrete]
// CHECK:STDOUT:   %AddAssignWith.assoc_type.b4b: type = assoc_entity_type @AddAssignWith.1, @AddAssignWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %assoc0.e09: %AddAssignWith.assoc_type.b4b = assoc_entity element0, imports.%Core.import_ref.7e1 [concrete]
// CHECK:STDOUT:   %Int32.Op.type.68b858.12: type = fn_type @Int32.Op.12 [concrete]
// CHECK:STDOUT:   %Int32.Op.6afff1.12: %Int32.Op.type.68b858.12 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.c02: <witness> = custom_witness (%Int32.Op.6afff1.12), @AddAssignWith.1, @AddAssignWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %AddAssignWith.facet.d85: %AddAssignWith.type.a71 = facet_value %Int32, (%custom_witness.c02) [concrete]
// CHECK:STDOUT:   %pattern_type.736: type = pattern_type %AddAssignWith.type.a71 [concrete]
// CHECK:STDOUT:   %SubAssignWith.type.35f: type = facet_type <@SubAssignWith.1, @SubAssignWith.1(%Int32)> [concrete]
// CHECK:STDOUT:   %SubAssignWith.assoc_type.867: type = assoc_entity_type @SubAssignWith.1, @SubAssignWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %assoc0.aa5: %SubAssignWith.assoc_type.867 = assoc_entity element0, imports.%Core.import_ref.5b4 [concrete]
// CHECK:STDOUT:   %Int32.Op.type.68b858.14: type = fn_type @Int32.Op.14 [concrete]
// CHECK:STDOUT:   %Int32.Op.6afff1.14: %Int32.Op.type.68b858.14 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.062: <witness> = custom_witness (%Int32.Op.6afff1.14), @SubAssignWith.1, @SubAssignWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %SubAssignWith.facet.187: %SubAssignWith.type.35f = facet_value %Int32, (%custom_witness.062) [concrete]
// CHECK:STDOUT:   %pattern_type.870: type = pattern_type %SubAssignWith.type.35f [concrete]
// CHECK:STDOUT:   %MulAssignWith.type.dd2: type = facet_type <@MulAssignWith.1, @MulAssignWith.1(%Int32)> [concrete]
// CHECK:STDOUT:   %MulAssignWith.assoc_type.5c3: type = assoc_entity_type @MulAssignWith.1, @MulAssignWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %assoc0.6d0: %MulAssignWith.assoc_type.5c3 = assoc_entity element0, imports.%Core.import_ref.69b [concrete]
// CHECK:STDOUT:   %Int32.Op.type.68b858.15: type = fn_type @Int32.Op.15 [concrete]
// CHECK:STDOUT:   %Int32.Op.6afff1.15: %Int32.Op.type.68b858.15 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.03af: <witness> = custom_witness (%Int32.Op.6afff1.15), @MulAssignWith.1, @MulAssignWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %MulAssignWith.facet.775: %MulAssignWith.type.dd2 = facet_value %Int32, (%custom_witness.03af) [concrete]
// CHECK:STDOUT:   %pattern_type.e94: type = pattern_type %MulAssignWith.type.dd2 [concrete]
// CHECK:STDOUT:   %DivAssignWith.type.a38: type = facet_type <@DivAssignWith.1, @DivAssignWith.1(%Int32)> [concrete]
// CHECK:STDOUT:   %DivAssignWith.assoc_type.86d: type = assoc_entity_type @DivAssignWith.1, @DivAssignWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %assoc0.b6e: %DivAssignWith.assoc_type.86d = assoc_entity element0, imports.%Core.import_ref.799 [concrete]
// CHECK:STDOUT:   %Int32.Op.type.68b858.16: type = fn_type @Int32.Op.16 [concrete]
// CHECK:STDOUT:   %Int32.Op.6afff1.16: %Int32.Op.type.68b858.16 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.be6: <witness> = custom_witness (%Int32.Op.6afff1.16), @DivAssignWith.1, @DivAssignWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %DivAssignWith.facet.f5b: %DivAssignWith.type.a38 = facet_value %Int32, (%custom_witness.be6) [concrete]
// CHECK:STDOUT:   %pattern_type.689: type = pattern_type %DivAssignWith.type.a38 [concrete]
// CHECK:STDOUT:   %ModAssignWith.type.36f: type = facet_type <@ModAssignWith.1, @ModAssignWith.1(%Int32)> [concrete]
// CHECK:STDOUT:   %ModAssignWith.assoc_type.1a5: type = assoc_entity_type @ModAssignWith.1, @ModAssignWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %assoc0.609: %ModAssignWith.assoc_type.1a5 = assoc_entity element0, imports.%Core.import_ref.51c [concrete]
// CHECK:STDOUT:   %Int32.Op.type.68b858.17: type = fn_type @Int32.Op.17 [concrete]
// CHECK:STDOUT:   %Int32.Op.6afff1.17: %Int32.Op.type.68b858.17 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.203: <witness> = custom_witness (%Int32.Op.6afff1.17), @ModAssignWith.1, @ModAssignWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %ModAssignWith.facet.b82: %ModAssignWith.type.36f = facet_value %Int32, (%custom_witness.203) [concrete]
// CHECK:STDOUT:   %pattern_type.ebe: type = pattern_type %ModAssignWith.type.36f [concrete]
// CHECK:STDOUT:   %Int64.Op.type.445ce8.2: type = fn_type @Int64.Op.2 [concrete]
// CHECK:STDOUT:   %Int64.Op.b941da.2: %Int64.Op.type.445ce8.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.3a4: <witness> = custom_witness (%Int64.Op.b941da.2), @AddAssignWith.1, @AddAssignWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %AddAssignWith.facet.b70: %AddAssignWith.type.a71 = facet_value %Int64, (%custom_witness.3a4) [concrete]
// CHECK:STDOUT:   %Int64.Op.type.445ce8.4: type = fn_type @Int64.Op.4 [concrete]
// CHECK:STDOUT:   %Int64.Op.b941da.4: %Int64.Op.type.445ce8.4 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.de3: <witness> = custom_witness (%Int64.Op.b941da.4), @SubAssignWith.1, @SubAssignWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %SubAssignWith.facet.bd0: %SubAssignWith.type.35f = facet_value %Int64, (%custom_witness.de3) [concrete]
// CHECK:STDOUT:   %Int64.Op.type.445ce8.5: type = fn_type @Int64.Op.5 [concrete]
// CHECK:STDOUT:   %Int64.Op.b941da.5: %Int64.Op.type.445ce8.5 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.917: <witness> = custom_witness (%Int64.Op.b941da.5), @MulAssignWith.1, @MulAssignWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %MulAssignWith.facet.982: %MulAssignWith.type.dd2 = facet_value %Int64, (%custom_witness.917) [concrete]
// CHECK:STDOUT:   %Int64.Op.type.445ce8.6: type = fn_type @Int64.Op.6 [concrete]
// CHECK:STDOUT:   %Int64.Op.b941da.6: %Int64.Op.type.445ce8.6 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.f47: <witness> = custom_witness (%Int64.Op.b941da.6), @DivAssignWith.1, @DivAssignWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %DivAssignWith.facet.d3b: %DivAssignWith.type.a38 = facet_value %Int64, (%custom_witness.f47) [concrete]
// CHECK:STDOUT:   %Int64.Op.type.445ce8.7: type = fn_type @Int64.Op.7 [concrete]
// CHECK:STDOUT:   %Int64.Op.b941da.7: %Int64.Op.type.445ce8.7 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.ad7: <witness> = custom_witness (%Int64.Op.b941da.7), @ModAssignWith.1, @ModAssignWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %ModAssignWith.facet.1c6: %ModAssignWith.type.36f = facet_value %Int64, (%custom_witness.ad7) [concrete]
// CHECK:STDOUT:   %Inc.type: type = facet_type <@Inc> [concrete]
// CHECK:STDOUT:   %T.b7e: %Inc.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.3f5: type = facet_access_type %T.b7e [symbolic]
// CHECK:STDOUT:   %pattern_type.fb5eac.2: type = pattern_type %T.as_type.3f5 [symbolic]
// CHECK:STDOUT:   %Inc.lookup_impl_witness: <witness> = lookup_impl_witness %T.b7e, @Inc [symbolic]
// CHECK:STDOUT:   %Inc.WithSelf.Op.type.623399.2: type = fn_type @Inc.WithSelf.Op, @Inc.WithSelf(%T.b7e) [symbolic]
// CHECK:STDOUT:   %.ad4: type = fn_type_with_self_type %Inc.WithSelf.Op.type.623399.2, %T.b7e [symbolic]
// CHECK:STDOUT:   %impl.elem0.489: %.ad4 = impl_witness_access %Inc.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.71c: <specific function> = specific_impl_function %impl.elem0.489, @Inc.WithSelf.Op(%T.b7e) [symbolic]
// CHECK:STDOUT:   %Dec.type: type = facet_type <@Dec> [concrete]
// CHECK:STDOUT:   %T.ce1: %Dec.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.ead: type = facet_access_type %T.ce1 [symbolic]
// CHECK:STDOUT:   %pattern_type.f51ad8.2: type = pattern_type %T.as_type.ead [symbolic]
// CHECK:STDOUT:   %Dec.lookup_impl_witness: <witness> = lookup_impl_witness %T.ce1, @Dec [symbolic]
// CHECK:STDOUT:   %Dec.WithSelf.Op.type.41adba.2: type = fn_type @Dec.WithSelf.Op, @Dec.WithSelf(%T.ce1) [symbolic]
// CHECK:STDOUT:   %.41c: type = fn_type_with_self_type %Dec.WithSelf.Op.type.41adba.2, %T.ce1 [symbolic]
// CHECK:STDOUT:   %impl.elem0.64b: %.41c = impl_witness_access %Dec.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.91c: <specific function> = specific_impl_function %impl.elem0.64b, @Dec.WithSelf.Op(%T.ce1) [symbolic]
// CHECK:STDOUT:   %Negate.type: type = facet_type <@Negate> [concrete]
// CHECK:STDOUT:   %facet_type.86d: type = facet_type <@Destroy & @Negate> [concrete]
// CHECK:STDOUT:   %.Self.c4b: %facet_type.86d = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %Negate.lookup_impl_witness.3a4: <witness> = lookup_impl_witness %.Self.c4b, @Negate [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.f00: type = impl_witness_access %Negate.lookup_impl_witness.3a4, element0 [symbolic_self]
// CHECK:STDOUT:   %facet_type.0fc: type = facet_type <@Destroy & @Negate where %impl.elem0.f00 = %Result.as_type> [symbolic]
// CHECK:STDOUT:   %pattern_type.5c4: type = pattern_type %facet_type.0fc [symbolic]
// CHECK:STDOUT:   %T.abc: %facet_type.0fc = symbolic_binding T, 1 [symbolic]
// CHECK:STDOUT:   %T.as_type.c82: type = facet_access_type %T.abc [symbolic]
// CHECK:STDOUT:   %pattern_type.ece: type = pattern_type %T.as_type.c82 [symbolic]
// CHECK:STDOUT:   %Negate.lookup_impl_witness.783: <witness> = lookup_impl_witness %T.abc, @Negate [symbolic]
// CHECK:STDOUT:   %Negate.facet.758: %Negate.type = facet_value %T.as_type.c82, (%Negate.lookup_impl_witness.783) [symbolic]
// CHECK:STDOUT:   %Negate.WithSelf.Op.type.899: type = fn_type @Negate.WithSelf.Op, @Negate.WithSelf(%Negate.facet.758) [symbolic]
// CHECK:STDOUT:   %.0d7: type = fn_type_with_self_type %Negate.WithSelf.Op.type.899, %Negate.facet.758 [symbolic]
// CHECK:STDOUT:   %impl.elem1.ec8: %.0d7 = impl_witness_access %Negate.lookup_impl_witness.783, element1 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.f22: <specific function> = specific_impl_function %impl.elem1.ec8, @Negate.WithSelf.Op(%Negate.facet.758) [symbolic]
// CHECK:STDOUT:   %Int32.cpp_operator.type.49e401.16: type = fn_type @Int32.cpp_operator.16 [concrete]
// CHECK:STDOUT:   %Int32.cpp_operator.0010d3.16: %Int32.cpp_operator.type.49e401.16 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.934: <witness> = custom_witness (%Int32.cpp_operator.0010d3.16), @Inc [concrete]
// CHECK:STDOUT:   %Inc.facet.390: %Inc.type = facet_value %Int32, (%custom_witness.934) [concrete]
// CHECK:STDOUT:   %Int32.cpp_operator.type.49e401.17: type = fn_type @Int32.cpp_operator.17 [concrete]
// CHECK:STDOUT:   %Int32.cpp_operator.0010d3.17: %Int32.cpp_operator.type.49e401.17 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.99f: <witness> = custom_witness (%Int32.cpp_operator.0010d3.17), @Dec [concrete]
// CHECK:STDOUT:   %Dec.facet.961: %Dec.type = facet_value %Int32, (%custom_witness.99f) [concrete]
// CHECK:STDOUT:   %facet_type.89a: type = facet_type <@Destroy & @Negate where %impl.elem0.f00 = %Int64> [concrete]
// CHECK:STDOUT:   %Int16.Op.type.180c8b.18: type = fn_type @Int16.Op.18 [concrete]
// CHECK:STDOUT:   %Int16.Op.c3e49c.18: %Int16.Op.type.180c8b.18 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.e58: <witness> = custom_witness (%Int64, %Int16.Op.c3e49c.18), @Negate [concrete]
// CHECK:STDOUT:   %facet_value.70a: %facet_type.89a = facet_value %Int16, (%custom_witness.e5b, %custom_witness.e58) [concrete]
// CHECK:STDOUT:   %pattern_type.c49: type = pattern_type %facet_type.89a [concrete]
// CHECK:STDOUT:   %facet_type.d5d: type = facet_type <@Destroy & @Negate where %impl.elem0.f00 = %Int32> [concrete]
// CHECK:STDOUT:   %Int32.Op.type.68b858.18: type = fn_type @Int32.Op.18 [concrete]
// CHECK:STDOUT:   %Int32.Op.6afff1.18: %Int32.Op.type.68b858.18 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.e8a: <witness> = custom_witness (%Int32, %Int32.Op.6afff1.18), @Negate [concrete]
// CHECK:STDOUT:   %facet_value.396: %facet_type.d5d = facet_value %Int32, (%custom_witness.ce1, %custom_witness.e8a) [concrete]
// CHECK:STDOUT:   %pattern_type.0ec: type = pattern_type %facet_type.d5d [concrete]
// CHECK:STDOUT:   %complete_type.f1c: <witness> = complete_type_witness %AddWith.type.344 [concrete]
// CHECK:STDOUT:   %AddWith.facet.ffe: %AddWith.type.344 = facet_value %i16, (%custom_witness.47f) [concrete]
// CHECK:STDOUT:   %AddWith.WithSelf.Op.type.894: type = fn_type @AddWith.WithSelf.Op, @AddWith.WithSelf(%Int16, %AddWith.facet.ffe) [concrete]
// CHECK:STDOUT:   %.b10: type = fn_type_with_self_type %AddWith.WithSelf.Op.type.894, %AddWith.facet.ffe [concrete]
// CHECK:STDOUT:   %complete_type.9b3: <witness> = complete_type_witness %SubWith.type.306 [concrete]
// CHECK:STDOUT:   %SubWith.facet.dd9: %SubWith.type.306 = facet_value %i16, (%custom_witness.a7b) [concrete]
// CHECK:STDOUT:   %SubWith.WithSelf.Op.type.44d: type = fn_type @SubWith.WithSelf.Op, @SubWith.WithSelf(%Int16, %SubWith.facet.dd9) [concrete]
// CHECK:STDOUT:   %.acf: type = fn_type_with_self_type %SubWith.WithSelf.Op.type.44d, %SubWith.facet.dd9 [concrete]
// CHECK:STDOUT:   %complete_type.4d2: <witness> = complete_type_witness %MulWith.type.0cd [concrete]
// CHECK:STDOUT:   %MulWith.facet.2a7: %MulWith.type.0cd = facet_value %i16, (%custom_witness.e40) [concrete]
// CHECK:STDOUT:   %MulWith.WithSelf.Op.type.504: type = fn_type @MulWith.WithSelf.Op, @MulWith.WithSelf(%Int16, %MulWith.facet.2a7) [concrete]
// CHECK:STDOUT:   %.20d: type = fn_type_with_self_type %MulWith.WithSelf.Op.type.504, %MulWith.facet.2a7 [concrete]
// CHECK:STDOUT:   %complete_type.e42: <witness> = complete_type_witness %DivWith.type.4c2 [concrete]
// CHECK:STDOUT:   %DivWith.facet.c60: %DivWith.type.4c2 = facet_value %i16, (%custom_witness.fb3) [concrete]
// CHECK:STDOUT:   %DivWith.WithSelf.Op.type.7eb: type = fn_type @DivWith.WithSelf.Op, @DivWith.WithSelf(%Int16, %DivWith.facet.c60) [concrete]
// CHECK:STDOUT:   %.411: type = fn_type_with_self_type %DivWith.WithSelf.Op.type.7eb, %DivWith.facet.c60 [concrete]
// CHECK:STDOUT:   %complete_type.e9f: <witness> = complete_type_witness %ModWith.type.009 [concrete]
// CHECK:STDOUT:   %ModWith.facet.0e3: %ModWith.type.009 = facet_value %i16, (%custom_witness.69e) [concrete]
// CHECK:STDOUT:   %ModWith.WithSelf.Op.type.a02: type = fn_type @ModWith.WithSelf.Op, @ModWith.WithSelf(%Int16, %ModWith.facet.0e3) [concrete]
// CHECK:STDOUT:   %.a0e: type = fn_type_with_self_type %ModWith.WithSelf.Op.type.a02, %ModWith.facet.0e3 [concrete]
// CHECK:STDOUT:   %complete_type.e1b: <witness> = complete_type_witness %AddWith.type.a18 [concrete]
// CHECK:STDOUT:   %AddWith.facet.c96: %AddWith.type.a18 = facet_value %Int16, (%custom_witness.096) [concrete]
// CHECK:STDOUT:   %AddWith.WithSelf.Op.type.b73: type = fn_type @AddWith.WithSelf.Op, @AddWith.WithSelf(%i16, %AddWith.facet.c96) [concrete]
// CHECK:STDOUT:   %.f80: type = fn_type_with_self_type %AddWith.WithSelf.Op.type.b73, %AddWith.facet.c96 [concrete]
// CHECK:STDOUT:   %complete_type.6de: <witness> = complete_type_witness %SubWith.type.a1b [concrete]
// CHECK:STDOUT:   %SubWith.facet.3c2: %SubWith.type.a1b = facet_value %Int16, (%custom_witness.ced) [concrete]
// CHECK:STDOUT:   %SubWith.WithSelf.Op.type.9bf: type = fn_type @SubWith.WithSelf.Op, @SubWith.WithSelf(%i16, %SubWith.facet.3c2) [concrete]
// CHECK:STDOUT:   %.90d: type = fn_type_with_self_type %SubWith.WithSelf.Op.type.9bf, %SubWith.facet.3c2 [concrete]
// CHECK:STDOUT:   %complete_type.8b34: <witness> = complete_type_witness %MulWith.type.ee9 [concrete]
// CHECK:STDOUT:   %MulWith.facet.0a1: %MulWith.type.ee9 = facet_value %Int16, (%custom_witness.270) [concrete]
// CHECK:STDOUT:   %MulWith.WithSelf.Op.type.a45: type = fn_type @MulWith.WithSelf.Op, @MulWith.WithSelf(%i16, %MulWith.facet.0a1) [concrete]
// CHECK:STDOUT:   %.70d: type = fn_type_with_self_type %MulWith.WithSelf.Op.type.a45, %MulWith.facet.0a1 [concrete]
// CHECK:STDOUT:   %complete_type.fc5: <witness> = complete_type_witness %DivWith.type.9ae [concrete]
// CHECK:STDOUT:   %DivWith.facet.915: %DivWith.type.9ae = facet_value %Int16, (%custom_witness.462) [concrete]
// CHECK:STDOUT:   %DivWith.WithSelf.Op.type.e9d: type = fn_type @DivWith.WithSelf.Op, @DivWith.WithSelf(%i16, %DivWith.facet.915) [concrete]
// CHECK:STDOUT:   %.d3f: type = fn_type_with_self_type %DivWith.WithSelf.Op.type.e9d, %DivWith.facet.915 [concrete]
// CHECK:STDOUT:   %complete_type.78b: <witness> = complete_type_witness %ModWith.type.406 [concrete]
// CHECK:STDOUT:   %ModWith.facet.6ed: %ModWith.type.406 = facet_value %Int16, (%custom_witness.84d2) [concrete]
// CHECK:STDOUT:   %ModWith.WithSelf.Op.type.a5a: type = fn_type @ModWith.WithSelf.Op, @ModWith.WithSelf(%i16, %ModWith.facet.6ed) [concrete]
// CHECK:STDOUT:   %.a4e: type = fn_type_with_self_type %ModWith.WithSelf.Op.type.a5a, %ModWith.facet.6ed [concrete]
// CHECK:STDOUT:   %complete_type.03d: <witness> = complete_type_witness %AddWith.type.e9e [concrete]
// CHECK:STDOUT:   %AddWith.facet.abd: %AddWith.type.e9e = facet_value %Int16, (%custom_witness.200) [concrete]
// CHECK:STDOUT:   %AddWith.WithSelf.Op.type.708: type = fn_type @AddWith.WithSelf.Op, @AddWith.WithSelf(%Int32, %AddWith.facet.abd) [concrete]
// CHECK:STDOUT:   %.0ff: type = fn_type_with_self_type %AddWith.WithSelf.Op.type.708, %AddWith.facet.abd [concrete]
// CHECK:STDOUT:   %complete_type.9bf: <witness> = complete_type_witness %SubWith.type.84c [concrete]
// CHECK:STDOUT:   %SubWith.facet.696: %SubWith.type.84c = facet_value %Int16, (%custom_witness.103) [concrete]
// CHECK:STDOUT:   %SubWith.WithSelf.Op.type.e30: type = fn_type @SubWith.WithSelf.Op, @SubWith.WithSelf(%Int32, %SubWith.facet.696) [concrete]
// CHECK:STDOUT:   %.a54: type = fn_type_with_self_type %SubWith.WithSelf.Op.type.e30, %SubWith.facet.696 [concrete]
// CHECK:STDOUT:   %complete_type.a1f: <witness> = complete_type_witness %MulWith.type.d1b [concrete]
// CHECK:STDOUT:   %MulWith.facet.6d1: %MulWith.type.d1b = facet_value %Int16, (%custom_witness.a6d) [concrete]
// CHECK:STDOUT:   %MulWith.WithSelf.Op.type.61d: type = fn_type @MulWith.WithSelf.Op, @MulWith.WithSelf(%Int32, %MulWith.facet.6d1) [concrete]
// CHECK:STDOUT:   %.3a1: type = fn_type_with_self_type %MulWith.WithSelf.Op.type.61d, %MulWith.facet.6d1 [concrete]
// CHECK:STDOUT:   %complete_type.e49: <witness> = complete_type_witness %DivWith.type.26e [concrete]
// CHECK:STDOUT:   %DivWith.facet.246: %DivWith.type.26e = facet_value %Int16, (%custom_witness.510) [concrete]
// CHECK:STDOUT:   %DivWith.WithSelf.Op.type.623: type = fn_type @DivWith.WithSelf.Op, @DivWith.WithSelf(%Int32, %DivWith.facet.246) [concrete]
// CHECK:STDOUT:   %.e6b: type = fn_type_with_self_type %DivWith.WithSelf.Op.type.623, %DivWith.facet.246 [concrete]
// CHECK:STDOUT:   %complete_type.3ed: <witness> = complete_type_witness %ModWith.type.5c4 [concrete]
// CHECK:STDOUT:   %ModWith.facet.76b: %ModWith.type.5c4 = facet_value %Int16, (%custom_witness.a62) [concrete]
// CHECK:STDOUT:   %ModWith.WithSelf.Op.type.cfc: type = fn_type @ModWith.WithSelf.Op, @ModWith.WithSelf(%Int32, %ModWith.facet.76b) [concrete]
// CHECK:STDOUT:   %.132: type = fn_type_with_self_type %ModWith.WithSelf.Op.type.cfc, %ModWith.facet.76b [concrete]
// CHECK:STDOUT:   %AddWith.facet.fce: %AddWith.type.344 = facet_value %Int32, (%custom_witness.116) [concrete]
// CHECK:STDOUT:   %AddWith.WithSelf.Op.type.523: type = fn_type @AddWith.WithSelf.Op, @AddWith.WithSelf(%Int16, %AddWith.facet.fce) [concrete]
// CHECK:STDOUT:   %.a28: type = fn_type_with_self_type %AddWith.WithSelf.Op.type.523, %AddWith.facet.fce [concrete]
// CHECK:STDOUT:   %SubWith.facet.099: %SubWith.type.306 = facet_value %Int32, (%custom_witness.904) [concrete]
// CHECK:STDOUT:   %SubWith.WithSelf.Op.type.5d6: type = fn_type @SubWith.WithSelf.Op, @SubWith.WithSelf(%Int16, %SubWith.facet.099) [concrete]
// CHECK:STDOUT:   %.43e: type = fn_type_with_self_type %SubWith.WithSelf.Op.type.5d6, %SubWith.facet.099 [concrete]
// CHECK:STDOUT:   %MulWith.facet.aea: %MulWith.type.0cd = facet_value %Int32, (%custom_witness.04f) [concrete]
// CHECK:STDOUT:   %MulWith.WithSelf.Op.type.f3c: type = fn_type @MulWith.WithSelf.Op, @MulWith.WithSelf(%Int16, %MulWith.facet.aea) [concrete]
// CHECK:STDOUT:   %.a30: type = fn_type_with_self_type %MulWith.WithSelf.Op.type.f3c, %MulWith.facet.aea [concrete]
// CHECK:STDOUT:   %DivWith.facet.687: %DivWith.type.4c2 = facet_value %Int32, (%custom_witness.53b) [concrete]
// CHECK:STDOUT:   %DivWith.WithSelf.Op.type.b6a: type = fn_type @DivWith.WithSelf.Op, @DivWith.WithSelf(%Int16, %DivWith.facet.687) [concrete]
// CHECK:STDOUT:   %.c6a: type = fn_type_with_self_type %DivWith.WithSelf.Op.type.b6a, %DivWith.facet.687 [concrete]
// CHECK:STDOUT:   %ModWith.facet.7e6: %ModWith.type.009 = facet_value %Int32, (%custom_witness.895) [concrete]
// CHECK:STDOUT:   %ModWith.WithSelf.Op.type.55e: type = fn_type @ModWith.WithSelf.Op, @ModWith.WithSelf(%Int16, %ModWith.facet.7e6) [concrete]
// CHECK:STDOUT:   %.274: type = fn_type_with_self_type %ModWith.WithSelf.Op.type.55e, %ModWith.facet.7e6 [concrete]
// CHECK:STDOUT:   %AddWith.facet.e5b: %AddWith.type.e9e = facet_value %Int32, (%custom_witness.78d) [concrete]
// CHECK:STDOUT:   %AddWith.WithSelf.Op.type.13e: type = fn_type @AddWith.WithSelf.Op, @AddWith.WithSelf(%Int32, %AddWith.facet.e5b) [concrete]
// CHECK:STDOUT:   %.56e: type = fn_type_with_self_type %AddWith.WithSelf.Op.type.13e, %AddWith.facet.e5b [concrete]
// CHECK:STDOUT:   %SubWith.facet.e26: %SubWith.type.84c = facet_value %Int32, (%custom_witness.64d) [concrete]
// CHECK:STDOUT:   %SubWith.WithSelf.Op.type.d8a: type = fn_type @SubWith.WithSelf.Op, @SubWith.WithSelf(%Int32, %SubWith.facet.e26) [concrete]
// CHECK:STDOUT:   %.cb1: type = fn_type_with_self_type %SubWith.WithSelf.Op.type.d8a, %SubWith.facet.e26 [concrete]
// CHECK:STDOUT:   %MulWith.facet.b74: %MulWith.type.d1b = facet_value %Int32, (%custom_witness.7b6d) [concrete]
// CHECK:STDOUT:   %MulWith.WithSelf.Op.type.25b: type = fn_type @MulWith.WithSelf.Op, @MulWith.WithSelf(%Int32, %MulWith.facet.b74) [concrete]
// CHECK:STDOUT:   %.e42: type = fn_type_with_self_type %MulWith.WithSelf.Op.type.25b, %MulWith.facet.b74 [concrete]
// CHECK:STDOUT:   %DivWith.facet.8b3: %DivWith.type.26e = facet_value %Int32, (%custom_witness.db6) [concrete]
// CHECK:STDOUT:   %DivWith.WithSelf.Op.type.f6d: type = fn_type @DivWith.WithSelf.Op, @DivWith.WithSelf(%Int32, %DivWith.facet.8b3) [concrete]
// CHECK:STDOUT:   %.c6f: type = fn_type_with_self_type %DivWith.WithSelf.Op.type.f6d, %DivWith.facet.8b3 [concrete]
// CHECK:STDOUT:   %ModWith.facet.c6b: %ModWith.type.5c4 = facet_value %Int32, (%custom_witness.e01) [concrete]
// CHECK:STDOUT:   %ModWith.WithSelf.Op.type.b72: type = fn_type @ModWith.WithSelf.Op, @ModWith.WithSelf(%Int32, %ModWith.facet.c6b) [concrete]
// CHECK:STDOUT:   %.a97: type = fn_type_with_self_type %ModWith.WithSelf.Op.type.b72, %ModWith.facet.c6b [concrete]
// CHECK:STDOUT:   %complete_type.d2d: <witness> = complete_type_witness %AddWith.type.ee5 [concrete]
// CHECK:STDOUT:   %AddWith.facet.484: %AddWith.type.ee5 = facet_value %Int32, (%custom_witness.1b7) [concrete]
// CHECK:STDOUT:   %AddWith.WithSelf.Op.type.24f: type = fn_type @AddWith.WithSelf.Op, @AddWith.WithSelf(%Int64, %AddWith.facet.484) [concrete]
// CHECK:STDOUT:   %.442: type = fn_type_with_self_type %AddWith.WithSelf.Op.type.24f, %AddWith.facet.484 [concrete]
// CHECK:STDOUT:   %complete_type.c98: <witness> = complete_type_witness %SubWith.type.b41 [concrete]
// CHECK:STDOUT:   %SubWith.facet.80e: %SubWith.type.b41 = facet_value %Int32, (%custom_witness.cab) [concrete]
// CHECK:STDOUT:   %SubWith.WithSelf.Op.type.c99: type = fn_type @SubWith.WithSelf.Op, @SubWith.WithSelf(%Int64, %SubWith.facet.80e) [concrete]
// CHECK:STDOUT:   %.9c9: type = fn_type_with_self_type %SubWith.WithSelf.Op.type.c99, %SubWith.facet.80e [concrete]
// CHECK:STDOUT:   %complete_type.b09: <witness> = complete_type_witness %MulWith.type.166 [concrete]
// CHECK:STDOUT:   %MulWith.facet.0d5: %MulWith.type.166 = facet_value %Int32, (%custom_witness.f53) [concrete]
// CHECK:STDOUT:   %MulWith.WithSelf.Op.type.97b: type = fn_type @MulWith.WithSelf.Op, @MulWith.WithSelf(%Int64, %MulWith.facet.0d5) [concrete]
// CHECK:STDOUT:   %.07b: type = fn_type_with_self_type %MulWith.WithSelf.Op.type.97b, %MulWith.facet.0d5 [concrete]
// CHECK:STDOUT:   %complete_type.09b: <witness> = complete_type_witness %DivWith.type.70f [concrete]
// CHECK:STDOUT:   %DivWith.facet.22c: %DivWith.type.70f = facet_value %Int32, (%custom_witness.bca) [concrete]
// CHECK:STDOUT:   %DivWith.WithSelf.Op.type.565: type = fn_type @DivWith.WithSelf.Op, @DivWith.WithSelf(%Int64, %DivWith.facet.22c) [concrete]
// CHECK:STDOUT:   %.b72: type = fn_type_with_self_type %DivWith.WithSelf.Op.type.565, %DivWith.facet.22c [concrete]
// CHECK:STDOUT:   %complete_type.6fa: <witness> = complete_type_witness %ModWith.type.a92 [concrete]
// CHECK:STDOUT:   %ModWith.facet.570: %ModWith.type.a92 = facet_value %Int32, (%custom_witness.1b0) [concrete]
// CHECK:STDOUT:   %ModWith.WithSelf.Op.type.2b2: type = fn_type @ModWith.WithSelf.Op, @ModWith.WithSelf(%Int64, %ModWith.facet.570) [concrete]
// CHECK:STDOUT:   %.16c: type = fn_type_with_self_type %ModWith.WithSelf.Op.type.2b2, %ModWith.facet.570 [concrete]
// CHECK:STDOUT:   %AddWith.facet.66c: %AddWith.type.e9e = facet_value %Int64, (%custom_witness.fc1) [concrete]
// CHECK:STDOUT:   %AddWith.WithSelf.Op.type.e16: type = fn_type @AddWith.WithSelf.Op, @AddWith.WithSelf(%Int32, %AddWith.facet.66c) [concrete]
// CHECK:STDOUT:   %.41b: type = fn_type_with_self_type %AddWith.WithSelf.Op.type.e16, %AddWith.facet.66c [concrete]
// CHECK:STDOUT:   %SubWith.facet.494: %SubWith.type.84c = facet_value %Int64, (%custom_witness.31e) [concrete]
// CHECK:STDOUT:   %SubWith.WithSelf.Op.type.0e3: type = fn_type @SubWith.WithSelf.Op, @SubWith.WithSelf(%Int32, %SubWith.facet.494) [concrete]
// CHECK:STDOUT:   %.936: type = fn_type_with_self_type %SubWith.WithSelf.Op.type.0e3, %SubWith.facet.494 [concrete]
// CHECK:STDOUT:   %MulWith.facet.dab: %MulWith.type.d1b = facet_value %Int64, (%custom_witness.709) [concrete]
// CHECK:STDOUT:   %MulWith.WithSelf.Op.type.549: type = fn_type @MulWith.WithSelf.Op, @MulWith.WithSelf(%Int32, %MulWith.facet.dab) [concrete]
// CHECK:STDOUT:   %.dc0: type = fn_type_with_self_type %MulWith.WithSelf.Op.type.549, %MulWith.facet.dab [concrete]
// CHECK:STDOUT:   %DivWith.facet.9ad: %DivWith.type.26e = facet_value %Int64, (%custom_witness.47b) [concrete]
// CHECK:STDOUT:   %DivWith.WithSelf.Op.type.596: type = fn_type @DivWith.WithSelf.Op, @DivWith.WithSelf(%Int32, %DivWith.facet.9ad) [concrete]
// CHECK:STDOUT:   %.def: type = fn_type_with_self_type %DivWith.WithSelf.Op.type.596, %DivWith.facet.9ad [concrete]
// CHECK:STDOUT:   %ModWith.facet.325: %ModWith.type.5c4 = facet_value %Int64, (%custom_witness.ad6) [concrete]
// CHECK:STDOUT:   %ModWith.WithSelf.Op.type.efe: type = fn_type @ModWith.WithSelf.Op, @ModWith.WithSelf(%Int32, %ModWith.facet.325) [concrete]
// CHECK:STDOUT:   %.ef4: type = fn_type_with_self_type %ModWith.WithSelf.Op.type.efe, %ModWith.facet.325 [concrete]
// CHECK:STDOUT:   %complete_type.901: <witness> = complete_type_witness %AddAssignWith.type.704 [concrete]
// CHECK:STDOUT:   %AddAssignWith.WithSelf.Op.type.2c7: type = fn_type @AddAssignWith.WithSelf.Op, @AddAssignWith.WithSelf(%i16, %AddAssignWith.facet.23f) [concrete]
// CHECK:STDOUT:   %.122: type = fn_type_with_self_type %AddAssignWith.WithSelf.Op.type.2c7, %AddAssignWith.facet.23f [concrete]
// CHECK:STDOUT:   %complete_type.f45: <witness> = complete_type_witness %SubAssignWith.type.fed [concrete]
// CHECK:STDOUT:   %SubAssignWith.WithSelf.Op.type.71a: type = fn_type @SubAssignWith.WithSelf.Op, @SubAssignWith.WithSelf(%i16, %SubAssignWith.facet.3e7) [concrete]
// CHECK:STDOUT:   %.697: type = fn_type_with_self_type %SubAssignWith.WithSelf.Op.type.71a, %SubAssignWith.facet.3e7 [concrete]
// CHECK:STDOUT:   %complete_type.a71: <witness> = complete_type_witness %MulAssignWith.type.6c0 [concrete]
// CHECK:STDOUT:   %MulAssignWith.WithSelf.Op.type.dca: type = fn_type @MulAssignWith.WithSelf.Op, @MulAssignWith.WithSelf(%i16, %MulAssignWith.facet.78f) [concrete]
// CHECK:STDOUT:   %.bb2: type = fn_type_with_self_type %MulAssignWith.WithSelf.Op.type.dca, %MulAssignWith.facet.78f [concrete]
// CHECK:STDOUT:   %complete_type.620: <witness> = complete_type_witness %DivAssignWith.type.527 [concrete]
// CHECK:STDOUT:   %DivAssignWith.WithSelf.Op.type.76f: type = fn_type @DivAssignWith.WithSelf.Op, @DivAssignWith.WithSelf(%i16, %DivAssignWith.facet.b0d) [concrete]
// CHECK:STDOUT:   %.32e: type = fn_type_with_self_type %DivAssignWith.WithSelf.Op.type.76f, %DivAssignWith.facet.b0d [concrete]
// CHECK:STDOUT:   %complete_type.422: <witness> = complete_type_witness %ModAssignWith.type.ead [concrete]
// CHECK:STDOUT:   %ModAssignWith.WithSelf.Op.type.bea: type = fn_type @ModAssignWith.WithSelf.Op, @ModAssignWith.WithSelf(%i16, %ModAssignWith.facet.ada) [concrete]
// CHECK:STDOUT:   %.51a: type = fn_type_with_self_type %ModAssignWith.WithSelf.Op.type.bea, %ModAssignWith.facet.ada [concrete]
// CHECK:STDOUT:   %complete_type.2ec: <witness> = complete_type_witness %AddAssignWith.type.a71 [concrete]
// CHECK:STDOUT:   %AddAssignWith.WithSelf.Op.type.181: type = fn_type @AddAssignWith.WithSelf.Op, @AddAssignWith.WithSelf(%Int32, %AddAssignWith.facet.d85) [concrete]
// CHECK:STDOUT:   %.e44: type = fn_type_with_self_type %AddAssignWith.WithSelf.Op.type.181, %AddAssignWith.facet.d85 [concrete]
// CHECK:STDOUT:   %complete_type.1ca: <witness> = complete_type_witness %SubAssignWith.type.35f [concrete]
// CHECK:STDOUT:   %SubAssignWith.WithSelf.Op.type.e51: type = fn_type @SubAssignWith.WithSelf.Op, @SubAssignWith.WithSelf(%Int32, %SubAssignWith.facet.187) [concrete]
// CHECK:STDOUT:   %.34b: type = fn_type_with_self_type %SubAssignWith.WithSelf.Op.type.e51, %SubAssignWith.facet.187 [concrete]
// CHECK:STDOUT:   %complete_type.a19: <witness> = complete_type_witness %MulAssignWith.type.dd2 [concrete]
// CHECK:STDOUT:   %MulAssignWith.WithSelf.Op.type.89d: type = fn_type @MulAssignWith.WithSelf.Op, @MulAssignWith.WithSelf(%Int32, %MulAssignWith.facet.775) [concrete]
// CHECK:STDOUT:   %.263: type = fn_type_with_self_type %MulAssignWith.WithSelf.Op.type.89d, %MulAssignWith.facet.775 [concrete]
// CHECK:STDOUT:   %complete_type.136: <witness> = complete_type_witness %DivAssignWith.type.a38 [concrete]
// CHECK:STDOUT:   %DivAssignWith.WithSelf.Op.type.593: type = fn_type @DivAssignWith.WithSelf.Op, @DivAssignWith.WithSelf(%Int32, %DivAssignWith.facet.f5b) [concrete]
// CHECK:STDOUT:   %.c99: type = fn_type_with_self_type %DivAssignWith.WithSelf.Op.type.593, %DivAssignWith.facet.f5b [concrete]
// CHECK:STDOUT:   %complete_type.ca8: <witness> = complete_type_witness %ModAssignWith.type.36f [concrete]
// CHECK:STDOUT:   %ModAssignWith.WithSelf.Op.type.f83: type = fn_type @ModAssignWith.WithSelf.Op, @ModAssignWith.WithSelf(%Int32, %ModAssignWith.facet.b82) [concrete]
// CHECK:STDOUT:   %.146: type = fn_type_with_self_type %ModAssignWith.WithSelf.Op.type.f83, %ModAssignWith.facet.b82 [concrete]
// CHECK:STDOUT:   %AddAssignWith.WithSelf.Op.type.750: type = fn_type @AddAssignWith.WithSelf.Op, @AddAssignWith.WithSelf(%Int32, %AddAssignWith.facet.b70) [concrete]
// CHECK:STDOUT:   %.5d8: type = fn_type_with_self_type %AddAssignWith.WithSelf.Op.type.750, %AddAssignWith.facet.b70 [concrete]
// CHECK:STDOUT:   %SubAssignWith.WithSelf.Op.type.792: type = fn_type @SubAssignWith.WithSelf.Op, @SubAssignWith.WithSelf(%Int32, %SubAssignWith.facet.bd0) [concrete]
// CHECK:STDOUT:   %.827: type = fn_type_with_self_type %SubAssignWith.WithSelf.Op.type.792, %SubAssignWith.facet.bd0 [concrete]
// CHECK:STDOUT:   %MulAssignWith.WithSelf.Op.type.788: type = fn_type @MulAssignWith.WithSelf.Op, @MulAssignWith.WithSelf(%Int32, %MulAssignWith.facet.982) [concrete]
// CHECK:STDOUT:   %.4a3: type = fn_type_with_self_type %MulAssignWith.WithSelf.Op.type.788, %MulAssignWith.facet.982 [concrete]
// CHECK:STDOUT:   %DivAssignWith.WithSelf.Op.type.646: type = fn_type @DivAssignWith.WithSelf.Op, @DivAssignWith.WithSelf(%Int32, %DivAssignWith.facet.d3b) [concrete]
// CHECK:STDOUT:   %.384: type = fn_type_with_self_type %DivAssignWith.WithSelf.Op.type.646, %DivAssignWith.facet.d3b [concrete]
// CHECK:STDOUT:   %ModAssignWith.WithSelf.Op.type.a34: type = fn_type @ModAssignWith.WithSelf.Op, @ModAssignWith.WithSelf(%Int32, %ModAssignWith.facet.1c6) [concrete]
// CHECK:STDOUT:   %.ee0: type = fn_type_with_self_type %ModAssignWith.WithSelf.Op.type.a34, %ModAssignWith.facet.1c6 [concrete]
// CHECK:STDOUT:   %Inc.WithSelf.Op.type.eaa: type = fn_type @Inc.WithSelf.Op, @Inc.WithSelf(%Inc.facet.390) [concrete]
// CHECK:STDOUT:   %.696: type = fn_type_with_self_type %Inc.WithSelf.Op.type.eaa, %Inc.facet.390 [concrete]
// CHECK:STDOUT:   %Dec.WithSelf.Op.type.6fb: type = fn_type @Dec.WithSelf.Op, @Dec.WithSelf(%Dec.facet.961) [concrete]
// CHECK:STDOUT:   %.cdd: type = fn_type_with_self_type %Dec.WithSelf.Op.type.6fb, %Dec.facet.961 [concrete]
// CHECK:STDOUT:   %Negate.facet.0a5: %Negate.type = facet_value %Int16, (%custom_witness.e58) [concrete]
// CHECK:STDOUT:   %Negate.WithSelf.Op.type.d7d: type = fn_type @Negate.WithSelf.Op, @Negate.WithSelf(%Negate.facet.0a5) [concrete]
// CHECK:STDOUT:   %.3e7: type = fn_type_with_self_type %Negate.WithSelf.Op.type.d7d, %Negate.facet.0a5 [concrete]
// CHECK:STDOUT:   %Negate.facet.3c9: %Negate.type = facet_value %Int32, (%custom_witness.e8a) [concrete]
// CHECK:STDOUT:   %Negate.WithSelf.Op.type.1bc: type = fn_type @Negate.WithSelf.Op, @Negate.WithSelf(%Negate.facet.3c9) [concrete]
// CHECK:STDOUT:   %.19b: type = fn_type_with_self_type %Negate.WithSelf.Op.type.1bc, %Negate.facet.3c9 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.037: @AddWith.WithSelf.%AddWith.assoc_type (%AddWith.assoc_type.b6a) = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [symbolic = @AddWith.WithSelf.%assoc1 (constants.%assoc1.4d4)]
// CHECK:STDOUT:   %Core.import_ref.1b1: @AddWith.WithSelf.%AddWith.WithSelf.Op.type (%AddWith.WithSelf.Op.type.08f) = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [symbolic = @AddWith.WithSelf.%AddWith.WithSelf.Op (constants.%AddWith.WithSelf.Op.fb8)]
// CHECK:STDOUT:   %Core.import_ref.6e9: type = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [concrete = %Result.559]
// CHECK:STDOUT:   %Result.559: type = assoc_const_decl @Result.1 [concrete] {}
// CHECK:STDOUT:   %Core.import_ref.a84 = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.413: @SubWith.WithSelf.%SubWith.assoc_type (%SubWith.assoc_type.645) = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [symbolic = @SubWith.WithSelf.%assoc1 (constants.%assoc1.f09)]
// CHECK:STDOUT:   %Core.import_ref.abb: @SubWith.WithSelf.%SubWith.WithSelf.Op.type (%SubWith.WithSelf.Op.type.0bf) = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [symbolic = @SubWith.WithSelf.%SubWith.WithSelf.Op (constants.%SubWith.WithSelf.Op.bd9)]
// CHECK:STDOUT:   %Core.import_ref.7e5: type = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [concrete = %Result.757]
// CHECK:STDOUT:   %Result.757: type = assoc_const_decl @Result.2 [concrete] {}
// CHECK:STDOUT:   %Core.import_ref.e83 = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.a1b: @MulWith.WithSelf.%MulWith.assoc_type (%MulWith.assoc_type.b5d) = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [symbolic = @MulWith.WithSelf.%assoc1 (constants.%assoc1.ccf)]
// CHECK:STDOUT:   %Core.import_ref.abc: @MulWith.WithSelf.%MulWith.WithSelf.Op.type (%MulWith.WithSelf.Op.type.00c) = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [symbolic = @MulWith.WithSelf.%MulWith.WithSelf.Op (constants.%MulWith.WithSelf.Op.07a)]
// CHECK:STDOUT:   %Core.import_ref.d9b: type = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [concrete = %Result.4ff]
// CHECK:STDOUT:   %Result.4ff: type = assoc_const_decl @Result.3 [concrete] {}
// CHECK:STDOUT:   %Core.import_ref.d5a = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.beb: @DivWith.WithSelf.%DivWith.assoc_type (%DivWith.assoc_type.16c) = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [symbolic = @DivWith.WithSelf.%assoc1 (constants.%assoc1.b83)]
// CHECK:STDOUT:   %Core.import_ref.bb2c: @DivWith.WithSelf.%DivWith.WithSelf.Op.type (%DivWith.WithSelf.Op.type.8a1) = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [symbolic = @DivWith.WithSelf.%DivWith.WithSelf.Op (constants.%DivWith.WithSelf.Op.0a9)]
// CHECK:STDOUT:   %Core.import_ref.14e: type = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [concrete = %Result.e26]
// CHECK:STDOUT:   %Result.e26: type = assoc_const_decl @Result.4 [concrete] {}
// CHECK:STDOUT:   %Core.import_ref.4c4 = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.fcf: @ModWith.WithSelf.%ModWith.assoc_type (%ModWith.assoc_type.e6f) = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [symbolic = @ModWith.WithSelf.%assoc1 (constants.%assoc1.e77)]
// CHECK:STDOUT:   %Core.import_ref.c05: @ModWith.WithSelf.%ModWith.WithSelf.Op.type (%ModWith.WithSelf.Op.type.f61) = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [symbolic = @ModWith.WithSelf.%ModWith.WithSelf.Op (constants.%ModWith.WithSelf.Op.3ac)]
// CHECK:STDOUT:   %Core.import_ref.a0d: type = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [concrete = %Result.044]
// CHECK:STDOUT:   %Result.044: type = assoc_const_decl @Result.5 [concrete] {}
// CHECK:STDOUT:   %Core.import_ref.7b7 = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.016: @AddAssignWith.WithSelf.%AddAssignWith.assoc_type (%AddAssignWith.assoc_type.ca2a80.1) = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [symbolic = @AddAssignWith.WithSelf.%assoc0 (constants.%assoc0.5d2)]
// CHECK:STDOUT:   %Core.import_ref.7e1: @AddAssignWith.WithSelf.%AddAssignWith.WithSelf.Op.type (%AddAssignWith.WithSelf.Op.type.df0722.1) = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [symbolic = @AddAssignWith.WithSelf.%AddAssignWith.WithSelf.Op (constants.%AddAssignWith.WithSelf.Op.b33928.1)]
// CHECK:STDOUT:   %Core.import_ref.f18 = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.405: @SubAssignWith.WithSelf.%SubAssignWith.assoc_type (%SubAssignWith.assoc_type.4cb9a1.1) = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [symbolic = @SubAssignWith.WithSelf.%assoc0 (constants.%assoc0.fd1)]
// CHECK:STDOUT:   %Core.import_ref.5b4: @SubAssignWith.WithSelf.%SubAssignWith.WithSelf.Op.type (%SubAssignWith.WithSelf.Op.type.317c02.1) = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [symbolic = @SubAssignWith.WithSelf.%SubAssignWith.WithSelf.Op (constants.%SubAssignWith.WithSelf.Op.26e13f.1)]
// CHECK:STDOUT:   %Core.import_ref.a26 = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.257: @MulAssignWith.WithSelf.%MulAssignWith.assoc_type (%MulAssignWith.assoc_type.595561.1) = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [symbolic = @MulAssignWith.WithSelf.%assoc0 (constants.%assoc0.fc1)]
// CHECK:STDOUT:   %Core.import_ref.69b: @MulAssignWith.WithSelf.%MulAssignWith.WithSelf.Op.type (%MulAssignWith.WithSelf.Op.type.31cf70.1) = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [symbolic = @MulAssignWith.WithSelf.%MulAssignWith.WithSelf.Op (constants.%MulAssignWith.WithSelf.Op.cdf083.1)]
// CHECK:STDOUT:   %Core.import_ref.2e6 = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.b1f: @DivAssignWith.WithSelf.%DivAssignWith.assoc_type (%DivAssignWith.assoc_type.8b4b5b.1) = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [symbolic = @DivAssignWith.WithSelf.%assoc0 (constants.%assoc0.b19)]
// CHECK:STDOUT:   %Core.import_ref.799: @DivAssignWith.WithSelf.%DivAssignWith.WithSelf.Op.type (%DivAssignWith.WithSelf.Op.type.ded5d7.1) = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [symbolic = @DivAssignWith.WithSelf.%DivAssignWith.WithSelf.Op (constants.%DivAssignWith.WithSelf.Op.6f2821.1)]
// CHECK:STDOUT:   %Core.import_ref.7f9 = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.b81: @ModAssignWith.WithSelf.%ModAssignWith.assoc_type (%ModAssignWith.assoc_type.1462b9.1) = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [symbolic = @ModAssignWith.WithSelf.%assoc0 (constants.%assoc0.0e6)]
// CHECK:STDOUT:   %Core.import_ref.51c: @ModAssignWith.WithSelf.%ModAssignWith.WithSelf.Op.type (%ModAssignWith.WithSelf.Op.type.5452e9.1) = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [symbolic = @ModAssignWith.WithSelf.%ModAssignWith.WithSelf.Op (constants.%ModAssignWith.WithSelf.Op.c5cc9b.1)]
// CHECK:STDOUT:   %Core.import_ref.fa5 = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @AddWith.loc124(%Result.loc121_9.2: %Destroy.type, %U.loc122_4.2: %Destroy.type, %T.loc123_4.2: @AddWith.loc124.%facet_type.loc123_38 (%facet_type.024)) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %require_complete.loc126: <witness> = require_complete_type %AddWith.type.loc123_36.1 [symbolic = %require_complete.loc126 (constants.%require_complete.8bd)]
// CHECK:STDOUT:   %assoc1: @AddWith.loc124.%AddWith.assoc_type (%AddWith.assoc_type.8c2) = assoc_entity element1, imports.%Core.import_ref.1b1 [symbolic = %assoc1 (constants.%assoc1.0f4)]
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.loc126: <witness> = lookup_impl_witness %T.loc123_4.1, @AddWith.1, @AddWith.1(%U.as_type.loc123_36.1) [symbolic = %AddWith.lookup_impl_witness.loc126 (constants.%AddWith.lookup_impl_witness.3c7)]
// CHECK:STDOUT:   %AddWith.facet: @AddWith.loc124.%AddWith.type.loc123_36.1 (%AddWith.type.07d) = facet_value %T.as_type.loc124_29.1, (%AddWith.lookup_impl_witness.loc126) [symbolic = %AddWith.facet (constants.%AddWith.facet.570)]
// CHECK:STDOUT:   %AddWith.WithSelf.Op.type: type = fn_type @AddWith.WithSelf.Op, @AddWith.WithSelf(%U.as_type.loc123_36.1, %AddWith.facet) [symbolic = %AddWith.WithSelf.Op.type (constants.%AddWith.WithSelf.Op.type.82f)]
// CHECK:STDOUT:   %.loc126_5.4: type = fn_type_with_self_type %AddWith.WithSelf.Op.type, %AddWith.facet [symbolic = %.loc126_5.4 (constants.%.97d)]
// CHECK:STDOUT:   %impl.elem1.loc126_5.2: @AddWith.loc124.%.loc126_5.4 (%.97d) = impl_witness_access %AddWith.lookup_impl_witness.loc126, element1 [symbolic = %impl.elem1.loc126_5.2 (constants.%impl.elem1.07a)]
// CHECK:STDOUT:   %specific_impl_fn.loc126_5.3: <specific function> = specific_impl_function %impl.elem1.loc126_5.2, @AddWith.WithSelf.Op(%U.as_type.loc123_36.1, %AddWith.facet) [symbolic = %specific_impl_fn.loc126_5.3 (constants.%specific_impl_fn.dbc)]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Result.loc121_9.1) [symbolic = %Destroy.WithSelf.Op.type (constants.%Destroy.WithSelf.Op.type.cb2e47.2)]
// CHECK:STDOUT:   %.loc126_5.5: type = fn_type_with_self_type %Destroy.WithSelf.Op.type, %Result.loc121_9.1 [symbolic = %.loc126_5.5 (constants.%.d5f)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %Result.loc121_9.1, @Destroy [symbolic = %Destroy.lookup_impl_witness (constants.%Destroy.lookup_impl_witness.d86)]
// CHECK:STDOUT:   %impl.elem0.loc126_5.2: @AddWith.loc124.%.loc126_5.5 (%.d5f) = impl_witness_access %Destroy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc126_5.2 (constants.%impl.elem0.a6b)]
// CHECK:STDOUT:   %specific_impl_fn.loc126_5.4: <specific function> = specific_impl_function %impl.elem0.loc126_5.2, @Destroy.WithSelf.Op(%Result.loc121_9.1) [symbolic = %specific_impl_fn.loc126_5.4 (constants.%specific_impl_fn.761)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%result.param: @AddWith.loc124.%Result.as_type.loc123_54.1 (%Result.as_type), %x.param: @AddWith.loc124.%T.as_type.loc124_29.1 (%T.as_type.ab9), %y.param: @AddWith.loc124.%U.as_type.loc123_36.1 (%U.as_type.2b8)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: @AddWith.loc124.%T.as_type.loc124_29.1 (%T.as_type.ab9) = name_ref x, %x
// CHECK:STDOUT:     %y.ref: @AddWith.loc124.%U.as_type.loc123_36.1 (%U.as_type.2b8) = name_ref y, %y
// CHECK:STDOUT:     %AddWith.type.loc126: type = facet_type <@AddWith.1, @AddWith.1(constants.%U.as_type.2b8)> [symbolic = %AddWith.type.loc123_36.1 (constants.%AddWith.type.07d)]
// CHECK:STDOUT:     %.loc126_5.1: @AddWith.loc124.%AddWith.assoc_type (%AddWith.assoc_type.8c2) = specific_constant imports.%Core.import_ref.037, @AddWith.WithSelf(constants.%U.as_type.2b8, constants.%Self.b7c) [symbolic = %assoc1 (constants.%assoc1.0f4)]
// CHECK:STDOUT:     %Op.ref: @AddWith.loc124.%AddWith.assoc_type (%AddWith.assoc_type.8c2) = name_ref Op, %.loc126_5.1 [symbolic = %assoc1 (constants.%assoc1.0f4)]
// CHECK:STDOUT:     %impl.elem1.loc126_5.1: @AddWith.loc124.%.loc126_5.4 (%.97d) = impl_witness_access constants.%AddWith.lookup_impl_witness.3c7, element1 [symbolic = %impl.elem1.loc126_5.2 (constants.%impl.elem1.07a)]
// CHECK:STDOUT:     %bound_method.loc126_5.1: <bound method> = bound_method %x.ref, %impl.elem1.loc126_5.1
// CHECK:STDOUT:     %specific_impl_fn.loc126_5.1: <specific function> = specific_impl_function %impl.elem1.loc126_5.1, @AddWith.WithSelf.Op(constants.%U.as_type.2b8, constants.%AddWith.facet.570) [symbolic = %specific_impl_fn.loc126_5.3 (constants.%specific_impl_fn.dbc)]
// CHECK:STDOUT:     %bound_method.loc126_5.2: <bound method> = bound_method %x.ref, %specific_impl_fn.loc126_5.1
// CHECK:STDOUT:     %.loc126_5.2: ref @AddWith.loc124.%Result.as_type.loc123_54.1 (%Result.as_type) = temporary_storage
// CHECK:STDOUT:     %AddWith.WithSelf.Op.call: init @AddWith.loc124.%Result.as_type.loc123_54.1 (%Result.as_type) to %.loc126_5.2 = call %bound_method.loc126_5.2(%x.ref, %y.ref)
// CHECK:STDOUT:     %.loc126_5.3: ref @AddWith.loc124.%Result.as_type.loc123_54.1 (%Result.as_type) = temporary %.loc126_5.2, %AddWith.WithSelf.Op.call
// CHECK:STDOUT:     %impl.elem0.loc126_5.1: @AddWith.loc124.%.loc126_5.5 (%.d5f) = impl_witness_access constants.%Destroy.lookup_impl_witness.d86, element0 [symbolic = %impl.elem0.loc126_5.2 (constants.%impl.elem0.a6b)]
// CHECK:STDOUT:     %bound_method.loc126_5.3: <bound method> = bound_method %.loc126_5.3, %impl.elem0.loc126_5.1
// CHECK:STDOUT:     %specific_impl_fn.loc126_5.2: <specific function> = specific_impl_function %impl.elem0.loc126_5.1, @Destroy.WithSelf.Op(constants.%Result) [symbolic = %specific_impl_fn.loc126_5.4 (constants.%specific_impl_fn.761)]
// CHECK:STDOUT:     %bound_method.loc126_5.4: <bound method> = bound_method %.loc126_5.3, %specific_impl_fn.loc126_5.2
// CHECK:STDOUT:     %Destroy.WithSelf.Op.call: init %empty_tuple.type = call %bound_method.loc126_5.4(%.loc126_5.3)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @SubWith.loc135(%Result.loc132_9.2: %Destroy.type, %U.loc133_4.2: %Destroy.type, %T.loc134_4.2: @SubWith.loc135.%facet_type.loc134_38 (%facet_type.bea)) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %require_complete.loc137: <witness> = require_complete_type %SubWith.type.loc134_36.1 [symbolic = %require_complete.loc137 (constants.%require_complete.ae1)]
// CHECK:STDOUT:   %assoc1: @SubWith.loc135.%SubWith.assoc_type (%SubWith.assoc_type.065) = assoc_entity element1, imports.%Core.import_ref.abb [symbolic = %assoc1 (constants.%assoc1.173)]
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.loc137: <witness> = lookup_impl_witness %T.loc134_4.1, @SubWith.1, @SubWith.1(%U.as_type.loc134_36.1) [symbolic = %SubWith.lookup_impl_witness.loc137 (constants.%SubWith.lookup_impl_witness.db9)]
// CHECK:STDOUT:   %SubWith.facet: @SubWith.loc135.%SubWith.type.loc134_36.1 (%SubWith.type.593) = facet_value %T.as_type.loc135_29.1, (%SubWith.lookup_impl_witness.loc137) [symbolic = %SubWith.facet (constants.%SubWith.facet.7f1)]
// CHECK:STDOUT:   %SubWith.WithSelf.Op.type: type = fn_type @SubWith.WithSelf.Op, @SubWith.WithSelf(%U.as_type.loc134_36.1, %SubWith.facet) [symbolic = %SubWith.WithSelf.Op.type (constants.%SubWith.WithSelf.Op.type.e59)]
// CHECK:STDOUT:   %.loc137_5.4: type = fn_type_with_self_type %SubWith.WithSelf.Op.type, %SubWith.facet [symbolic = %.loc137_5.4 (constants.%.4b6)]
// CHECK:STDOUT:   %impl.elem1.loc137_5.2: @SubWith.loc135.%.loc137_5.4 (%.4b6) = impl_witness_access %SubWith.lookup_impl_witness.loc137, element1 [symbolic = %impl.elem1.loc137_5.2 (constants.%impl.elem1.6e4)]
// CHECK:STDOUT:   %specific_impl_fn.loc137_5.3: <specific function> = specific_impl_function %impl.elem1.loc137_5.2, @SubWith.WithSelf.Op(%U.as_type.loc134_36.1, %SubWith.facet) [symbolic = %specific_impl_fn.loc137_5.3 (constants.%specific_impl_fn.0d3)]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Result.loc132_9.1) [symbolic = %Destroy.WithSelf.Op.type (constants.%Destroy.WithSelf.Op.type.cb2e47.2)]
// CHECK:STDOUT:   %.loc137_5.5: type = fn_type_with_self_type %Destroy.WithSelf.Op.type, %Result.loc132_9.1 [symbolic = %.loc137_5.5 (constants.%.d5f)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %Result.loc132_9.1, @Destroy [symbolic = %Destroy.lookup_impl_witness (constants.%Destroy.lookup_impl_witness.d86)]
// CHECK:STDOUT:   %impl.elem0.loc137_5.2: @SubWith.loc135.%.loc137_5.5 (%.d5f) = impl_witness_access %Destroy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc137_5.2 (constants.%impl.elem0.a6b)]
// CHECK:STDOUT:   %specific_impl_fn.loc137_5.4: <specific function> = specific_impl_function %impl.elem0.loc137_5.2, @Destroy.WithSelf.Op(%Result.loc132_9.1) [symbolic = %specific_impl_fn.loc137_5.4 (constants.%specific_impl_fn.761)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%result.param: @SubWith.loc135.%Result.as_type.loc134_54.1 (%Result.as_type), %x.param: @SubWith.loc135.%T.as_type.loc135_29.1 (%T.as_type.474), %y.param: @SubWith.loc135.%U.as_type.loc134_36.1 (%U.as_type.2b8)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: @SubWith.loc135.%T.as_type.loc135_29.1 (%T.as_type.474) = name_ref x, %x
// CHECK:STDOUT:     %y.ref: @SubWith.loc135.%U.as_type.loc134_36.1 (%U.as_type.2b8) = name_ref y, %y
// CHECK:STDOUT:     %SubWith.type.loc137: type = facet_type <@SubWith.1, @SubWith.1(constants.%U.as_type.2b8)> [symbolic = %SubWith.type.loc134_36.1 (constants.%SubWith.type.593)]
// CHECK:STDOUT:     %.loc137_5.1: @SubWith.loc135.%SubWith.assoc_type (%SubWith.assoc_type.065) = specific_constant imports.%Core.import_ref.413, @SubWith.WithSelf(constants.%U.as_type.2b8, constants.%Self.ddf) [symbolic = %assoc1 (constants.%assoc1.173)]
// CHECK:STDOUT:     %Op.ref: @SubWith.loc135.%SubWith.assoc_type (%SubWith.assoc_type.065) = name_ref Op, %.loc137_5.1 [symbolic = %assoc1 (constants.%assoc1.173)]
// CHECK:STDOUT:     %impl.elem1.loc137_5.1: @SubWith.loc135.%.loc137_5.4 (%.4b6) = impl_witness_access constants.%SubWith.lookup_impl_witness.db9, element1 [symbolic = %impl.elem1.loc137_5.2 (constants.%impl.elem1.6e4)]
// CHECK:STDOUT:     %bound_method.loc137_5.1: <bound method> = bound_method %x.ref, %impl.elem1.loc137_5.1
// CHECK:STDOUT:     %specific_impl_fn.loc137_5.1: <specific function> = specific_impl_function %impl.elem1.loc137_5.1, @SubWith.WithSelf.Op(constants.%U.as_type.2b8, constants.%SubWith.facet.7f1) [symbolic = %specific_impl_fn.loc137_5.3 (constants.%specific_impl_fn.0d3)]
// CHECK:STDOUT:     %bound_method.loc137_5.2: <bound method> = bound_method %x.ref, %specific_impl_fn.loc137_5.1
// CHECK:STDOUT:     %.loc137_5.2: ref @SubWith.loc135.%Result.as_type.loc134_54.1 (%Result.as_type) = temporary_storage
// CHECK:STDOUT:     %SubWith.WithSelf.Op.call: init @SubWith.loc135.%Result.as_type.loc134_54.1 (%Result.as_type) to %.loc137_5.2 = call %bound_method.loc137_5.2(%x.ref, %y.ref)
// CHECK:STDOUT:     %.loc137_5.3: ref @SubWith.loc135.%Result.as_type.loc134_54.1 (%Result.as_type) = temporary %.loc137_5.2, %SubWith.WithSelf.Op.call
// CHECK:STDOUT:     %impl.elem0.loc137_5.1: @SubWith.loc135.%.loc137_5.5 (%.d5f) = impl_witness_access constants.%Destroy.lookup_impl_witness.d86, element0 [symbolic = %impl.elem0.loc137_5.2 (constants.%impl.elem0.a6b)]
// CHECK:STDOUT:     %bound_method.loc137_5.3: <bound method> = bound_method %.loc137_5.3, %impl.elem0.loc137_5.1
// CHECK:STDOUT:     %specific_impl_fn.loc137_5.2: <specific function> = specific_impl_function %impl.elem0.loc137_5.1, @Destroy.WithSelf.Op(constants.%Result) [symbolic = %specific_impl_fn.loc137_5.4 (constants.%specific_impl_fn.761)]
// CHECK:STDOUT:     %bound_method.loc137_5.4: <bound method> = bound_method %.loc137_5.3, %specific_impl_fn.loc137_5.2
// CHECK:STDOUT:     %Destroy.WithSelf.Op.call: init %empty_tuple.type = call %bound_method.loc137_5.4(%.loc137_5.3)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @MulWith.loc146(%Result.loc143_9.2: %Destroy.type, %U.loc144_4.2: %Destroy.type, %T.loc145_4.2: @MulWith.loc146.%facet_type.loc145_38 (%facet_type.91e)) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %require_complete.loc148: <witness> = require_complete_type %MulWith.type.loc145_36.1 [symbolic = %require_complete.loc148 (constants.%require_complete.d9d)]
// CHECK:STDOUT:   %assoc1: @MulWith.loc146.%MulWith.assoc_type (%MulWith.assoc_type.97c) = assoc_entity element1, imports.%Core.import_ref.abc [symbolic = %assoc1 (constants.%assoc1.496)]
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.loc148: <witness> = lookup_impl_witness %T.loc145_4.1, @MulWith.1, @MulWith.1(%U.as_type.loc145_36.1) [symbolic = %MulWith.lookup_impl_witness.loc148 (constants.%MulWith.lookup_impl_witness.009)]
// CHECK:STDOUT:   %MulWith.facet: @MulWith.loc146.%MulWith.type.loc145_36.1 (%MulWith.type.9532) = facet_value %T.as_type.loc146_29.1, (%MulWith.lookup_impl_witness.loc148) [symbolic = %MulWith.facet (constants.%MulWith.facet.2d0)]
// CHECK:STDOUT:   %MulWith.WithSelf.Op.type: type = fn_type @MulWith.WithSelf.Op, @MulWith.WithSelf(%U.as_type.loc145_36.1, %MulWith.facet) [symbolic = %MulWith.WithSelf.Op.type (constants.%MulWith.WithSelf.Op.type.283)]
// CHECK:STDOUT:   %.loc148_5.4: type = fn_type_with_self_type %MulWith.WithSelf.Op.type, %MulWith.facet [symbolic = %.loc148_5.4 (constants.%.978)]
// CHECK:STDOUT:   %impl.elem1.loc148_5.2: @MulWith.loc146.%.loc148_5.4 (%.978) = impl_witness_access %MulWith.lookup_impl_witness.loc148, element1 [symbolic = %impl.elem1.loc148_5.2 (constants.%impl.elem1.80d)]
// CHECK:STDOUT:   %specific_impl_fn.loc148_5.3: <specific function> = specific_impl_function %impl.elem1.loc148_5.2, @MulWith.WithSelf.Op(%U.as_type.loc145_36.1, %MulWith.facet) [symbolic = %specific_impl_fn.loc148_5.3 (constants.%specific_impl_fn.889)]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Result.loc143_9.1) [symbolic = %Destroy.WithSelf.Op.type (constants.%Destroy.WithSelf.Op.type.cb2e47.2)]
// CHECK:STDOUT:   %.loc148_5.5: type = fn_type_with_self_type %Destroy.WithSelf.Op.type, %Result.loc143_9.1 [symbolic = %.loc148_5.5 (constants.%.d5f)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %Result.loc143_9.1, @Destroy [symbolic = %Destroy.lookup_impl_witness (constants.%Destroy.lookup_impl_witness.d86)]
// CHECK:STDOUT:   %impl.elem0.loc148_5.2: @MulWith.loc146.%.loc148_5.5 (%.d5f) = impl_witness_access %Destroy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc148_5.2 (constants.%impl.elem0.a6b)]
// CHECK:STDOUT:   %specific_impl_fn.loc148_5.4: <specific function> = specific_impl_function %impl.elem0.loc148_5.2, @Destroy.WithSelf.Op(%Result.loc143_9.1) [symbolic = %specific_impl_fn.loc148_5.4 (constants.%specific_impl_fn.761)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%result.param: @MulWith.loc146.%Result.as_type.loc145_54.1 (%Result.as_type), %x.param: @MulWith.loc146.%T.as_type.loc146_29.1 (%T.as_type.68f), %y.param: @MulWith.loc146.%U.as_type.loc145_36.1 (%U.as_type.2b8)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: @MulWith.loc146.%T.as_type.loc146_29.1 (%T.as_type.68f) = name_ref x, %x
// CHECK:STDOUT:     %y.ref: @MulWith.loc146.%U.as_type.loc145_36.1 (%U.as_type.2b8) = name_ref y, %y
// CHECK:STDOUT:     %MulWith.type.loc148: type = facet_type <@MulWith.1, @MulWith.1(constants.%U.as_type.2b8)> [symbolic = %MulWith.type.loc145_36.1 (constants.%MulWith.type.9532)]
// CHECK:STDOUT:     %.loc148_5.1: @MulWith.loc146.%MulWith.assoc_type (%MulWith.assoc_type.97c) = specific_constant imports.%Core.import_ref.a1b, @MulWith.WithSelf(constants.%U.as_type.2b8, constants.%Self.9c4) [symbolic = %assoc1 (constants.%assoc1.496)]
// CHECK:STDOUT:     %Op.ref: @MulWith.loc146.%MulWith.assoc_type (%MulWith.assoc_type.97c) = name_ref Op, %.loc148_5.1 [symbolic = %assoc1 (constants.%assoc1.496)]
// CHECK:STDOUT:     %impl.elem1.loc148_5.1: @MulWith.loc146.%.loc148_5.4 (%.978) = impl_witness_access constants.%MulWith.lookup_impl_witness.009, element1 [symbolic = %impl.elem1.loc148_5.2 (constants.%impl.elem1.80d)]
// CHECK:STDOUT:     %bound_method.loc148_5.1: <bound method> = bound_method %x.ref, %impl.elem1.loc148_5.1
// CHECK:STDOUT:     %specific_impl_fn.loc148_5.1: <specific function> = specific_impl_function %impl.elem1.loc148_5.1, @MulWith.WithSelf.Op(constants.%U.as_type.2b8, constants.%MulWith.facet.2d0) [symbolic = %specific_impl_fn.loc148_5.3 (constants.%specific_impl_fn.889)]
// CHECK:STDOUT:     %bound_method.loc148_5.2: <bound method> = bound_method %x.ref, %specific_impl_fn.loc148_5.1
// CHECK:STDOUT:     %.loc148_5.2: ref @MulWith.loc146.%Result.as_type.loc145_54.1 (%Result.as_type) = temporary_storage
// CHECK:STDOUT:     %MulWith.WithSelf.Op.call: init @MulWith.loc146.%Result.as_type.loc145_54.1 (%Result.as_type) to %.loc148_5.2 = call %bound_method.loc148_5.2(%x.ref, %y.ref)
// CHECK:STDOUT:     %.loc148_5.3: ref @MulWith.loc146.%Result.as_type.loc145_54.1 (%Result.as_type) = temporary %.loc148_5.2, %MulWith.WithSelf.Op.call
// CHECK:STDOUT:     %impl.elem0.loc148_5.1: @MulWith.loc146.%.loc148_5.5 (%.d5f) = impl_witness_access constants.%Destroy.lookup_impl_witness.d86, element0 [symbolic = %impl.elem0.loc148_5.2 (constants.%impl.elem0.a6b)]
// CHECK:STDOUT:     %bound_method.loc148_5.3: <bound method> = bound_method %.loc148_5.3, %impl.elem0.loc148_5.1
// CHECK:STDOUT:     %specific_impl_fn.loc148_5.2: <specific function> = specific_impl_function %impl.elem0.loc148_5.1, @Destroy.WithSelf.Op(constants.%Result) [symbolic = %specific_impl_fn.loc148_5.4 (constants.%specific_impl_fn.761)]
// CHECK:STDOUT:     %bound_method.loc148_5.4: <bound method> = bound_method %.loc148_5.3, %specific_impl_fn.loc148_5.2
// CHECK:STDOUT:     %Destroy.WithSelf.Op.call: init %empty_tuple.type = call %bound_method.loc148_5.4(%.loc148_5.3)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @DivWith.loc157(%Result.loc154_9.2: %Destroy.type, %U.loc155_4.2: %Destroy.type, %T.loc156_4.2: @DivWith.loc157.%facet_type.loc156_38 (%facet_type.6e6)) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %require_complete.loc159: <witness> = require_complete_type %DivWith.type.loc156_36.1 [symbolic = %require_complete.loc159 (constants.%require_complete.9a1)]
// CHECK:STDOUT:   %assoc1: @DivWith.loc157.%DivWith.assoc_type (%DivWith.assoc_type.06d) = assoc_entity element1, imports.%Core.import_ref.bb2c [symbolic = %assoc1 (constants.%assoc1.49b)]
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.loc159: <witness> = lookup_impl_witness %T.loc156_4.1, @DivWith.1, @DivWith.1(%U.as_type.loc156_36.1) [symbolic = %DivWith.lookup_impl_witness.loc159 (constants.%DivWith.lookup_impl_witness.b69)]
// CHECK:STDOUT:   %DivWith.facet: @DivWith.loc157.%DivWith.type.loc156_36.1 (%DivWith.type.0e2) = facet_value %T.as_type.loc157_29.1, (%DivWith.lookup_impl_witness.loc159) [symbolic = %DivWith.facet (constants.%DivWith.facet.490)]
// CHECK:STDOUT:   %DivWith.WithSelf.Op.type: type = fn_type @DivWith.WithSelf.Op, @DivWith.WithSelf(%U.as_type.loc156_36.1, %DivWith.facet) [symbolic = %DivWith.WithSelf.Op.type (constants.%DivWith.WithSelf.Op.type.2ad)]
// CHECK:STDOUT:   %.loc159_5.4: type = fn_type_with_self_type %DivWith.WithSelf.Op.type, %DivWith.facet [symbolic = %.loc159_5.4 (constants.%.0bf)]
// CHECK:STDOUT:   %impl.elem1.loc159_5.2: @DivWith.loc157.%.loc159_5.4 (%.0bf) = impl_witness_access %DivWith.lookup_impl_witness.loc159, element1 [symbolic = %impl.elem1.loc159_5.2 (constants.%impl.elem1.b07)]
// CHECK:STDOUT:   %specific_impl_fn.loc159_5.3: <specific function> = specific_impl_function %impl.elem1.loc159_5.2, @DivWith.WithSelf.Op(%U.as_type.loc156_36.1, %DivWith.facet) [symbolic = %specific_impl_fn.loc159_5.3 (constants.%specific_impl_fn.dbd)]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Result.loc154_9.1) [symbolic = %Destroy.WithSelf.Op.type (constants.%Destroy.WithSelf.Op.type.cb2e47.2)]
// CHECK:STDOUT:   %.loc159_5.5: type = fn_type_with_self_type %Destroy.WithSelf.Op.type, %Result.loc154_9.1 [symbolic = %.loc159_5.5 (constants.%.d5f)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %Result.loc154_9.1, @Destroy [symbolic = %Destroy.lookup_impl_witness (constants.%Destroy.lookup_impl_witness.d86)]
// CHECK:STDOUT:   %impl.elem0.loc159_5.2: @DivWith.loc157.%.loc159_5.5 (%.d5f) = impl_witness_access %Destroy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc159_5.2 (constants.%impl.elem0.a6b)]
// CHECK:STDOUT:   %specific_impl_fn.loc159_5.4: <specific function> = specific_impl_function %impl.elem0.loc159_5.2, @Destroy.WithSelf.Op(%Result.loc154_9.1) [symbolic = %specific_impl_fn.loc159_5.4 (constants.%specific_impl_fn.761)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%result.param: @DivWith.loc157.%Result.as_type.loc156_54.1 (%Result.as_type), %x.param: @DivWith.loc157.%T.as_type.loc157_29.1 (%T.as_type.cfe), %y.param: @DivWith.loc157.%U.as_type.loc156_36.1 (%U.as_type.2b8)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: @DivWith.loc157.%T.as_type.loc157_29.1 (%T.as_type.cfe) = name_ref x, %x
// CHECK:STDOUT:     %y.ref: @DivWith.loc157.%U.as_type.loc156_36.1 (%U.as_type.2b8) = name_ref y, %y
// CHECK:STDOUT:     %DivWith.type.loc159: type = facet_type <@DivWith.1, @DivWith.1(constants.%U.as_type.2b8)> [symbolic = %DivWith.type.loc156_36.1 (constants.%DivWith.type.0e2)]
// CHECK:STDOUT:     %.loc159_5.1: @DivWith.loc157.%DivWith.assoc_type (%DivWith.assoc_type.06d) = specific_constant imports.%Core.import_ref.beb, @DivWith.WithSelf(constants.%U.as_type.2b8, constants.%Self.d71) [symbolic = %assoc1 (constants.%assoc1.49b)]
// CHECK:STDOUT:     %Op.ref: @DivWith.loc157.%DivWith.assoc_type (%DivWith.assoc_type.06d) = name_ref Op, %.loc159_5.1 [symbolic = %assoc1 (constants.%assoc1.49b)]
// CHECK:STDOUT:     %impl.elem1.loc159_5.1: @DivWith.loc157.%.loc159_5.4 (%.0bf) = impl_witness_access constants.%DivWith.lookup_impl_witness.b69, element1 [symbolic = %impl.elem1.loc159_5.2 (constants.%impl.elem1.b07)]
// CHECK:STDOUT:     %bound_method.loc159_5.1: <bound method> = bound_method %x.ref, %impl.elem1.loc159_5.1
// CHECK:STDOUT:     %specific_impl_fn.loc159_5.1: <specific function> = specific_impl_function %impl.elem1.loc159_5.1, @DivWith.WithSelf.Op(constants.%U.as_type.2b8, constants.%DivWith.facet.490) [symbolic = %specific_impl_fn.loc159_5.3 (constants.%specific_impl_fn.dbd)]
// CHECK:STDOUT:     %bound_method.loc159_5.2: <bound method> = bound_method %x.ref, %specific_impl_fn.loc159_5.1
// CHECK:STDOUT:     %.loc159_5.2: ref @DivWith.loc157.%Result.as_type.loc156_54.1 (%Result.as_type) = temporary_storage
// CHECK:STDOUT:     %DivWith.WithSelf.Op.call: init @DivWith.loc157.%Result.as_type.loc156_54.1 (%Result.as_type) to %.loc159_5.2 = call %bound_method.loc159_5.2(%x.ref, %y.ref)
// CHECK:STDOUT:     %.loc159_5.3: ref @DivWith.loc157.%Result.as_type.loc156_54.1 (%Result.as_type) = temporary %.loc159_5.2, %DivWith.WithSelf.Op.call
// CHECK:STDOUT:     %impl.elem0.loc159_5.1: @DivWith.loc157.%.loc159_5.5 (%.d5f) = impl_witness_access constants.%Destroy.lookup_impl_witness.d86, element0 [symbolic = %impl.elem0.loc159_5.2 (constants.%impl.elem0.a6b)]
// CHECK:STDOUT:     %bound_method.loc159_5.3: <bound method> = bound_method %.loc159_5.3, %impl.elem0.loc159_5.1
// CHECK:STDOUT:     %specific_impl_fn.loc159_5.2: <specific function> = specific_impl_function %impl.elem0.loc159_5.1, @Destroy.WithSelf.Op(constants.%Result) [symbolic = %specific_impl_fn.loc159_5.4 (constants.%specific_impl_fn.761)]
// CHECK:STDOUT:     %bound_method.loc159_5.4: <bound method> = bound_method %.loc159_5.3, %specific_impl_fn.loc159_5.2
// CHECK:STDOUT:     %Destroy.WithSelf.Op.call: init %empty_tuple.type = call %bound_method.loc159_5.4(%.loc159_5.3)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @ModWith.loc168(%Result.loc165_9.2: %Destroy.type, %U.loc166_4.2: %Destroy.type, %T.loc167_4.2: @ModWith.loc168.%facet_type.loc167_38 (%facet_type.3a6)) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %require_complete.loc170: <witness> = require_complete_type %ModWith.type.loc167_36.1 [symbolic = %require_complete.loc170 (constants.%require_complete.9b1)]
// CHECK:STDOUT:   %assoc1: @ModWith.loc168.%ModWith.assoc_type (%ModWith.assoc_type.66e) = assoc_entity element1, imports.%Core.import_ref.c05 [symbolic = %assoc1 (constants.%assoc1.3da)]
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.loc170: <witness> = lookup_impl_witness %T.loc167_4.1, @ModWith.1, @ModWith.1(%U.as_type.loc167_36.1) [symbolic = %ModWith.lookup_impl_witness.loc170 (constants.%ModWith.lookup_impl_witness.e3f)]
// CHECK:STDOUT:   %ModWith.facet: @ModWith.loc168.%ModWith.type.loc167_36.1 (%ModWith.type.c6f) = facet_value %T.as_type.loc168_29.1, (%ModWith.lookup_impl_witness.loc170) [symbolic = %ModWith.facet (constants.%ModWith.facet.430)]
// CHECK:STDOUT:   %ModWith.WithSelf.Op.type: type = fn_type @ModWith.WithSelf.Op, @ModWith.WithSelf(%U.as_type.loc167_36.1, %ModWith.facet) [symbolic = %ModWith.WithSelf.Op.type (constants.%ModWith.WithSelf.Op.type.5b5)]
// CHECK:STDOUT:   %.loc170_5.4: type = fn_type_with_self_type %ModWith.WithSelf.Op.type, %ModWith.facet [symbolic = %.loc170_5.4 (constants.%.feb)]
// CHECK:STDOUT:   %impl.elem1.loc170_5.2: @ModWith.loc168.%.loc170_5.4 (%.feb) = impl_witness_access %ModWith.lookup_impl_witness.loc170, element1 [symbolic = %impl.elem1.loc170_5.2 (constants.%impl.elem1.1af)]
// CHECK:STDOUT:   %specific_impl_fn.loc170_5.3: <specific function> = specific_impl_function %impl.elem1.loc170_5.2, @ModWith.WithSelf.Op(%U.as_type.loc167_36.1, %ModWith.facet) [symbolic = %specific_impl_fn.loc170_5.3 (constants.%specific_impl_fn.772)]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Result.loc165_9.1) [symbolic = %Destroy.WithSelf.Op.type (constants.%Destroy.WithSelf.Op.type.cb2e47.2)]
// CHECK:STDOUT:   %.loc170_5.5: type = fn_type_with_self_type %Destroy.WithSelf.Op.type, %Result.loc165_9.1 [symbolic = %.loc170_5.5 (constants.%.d5f)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %Result.loc165_9.1, @Destroy [symbolic = %Destroy.lookup_impl_witness (constants.%Destroy.lookup_impl_witness.d86)]
// CHECK:STDOUT:   %impl.elem0.loc170_5.2: @ModWith.loc168.%.loc170_5.5 (%.d5f) = impl_witness_access %Destroy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc170_5.2 (constants.%impl.elem0.a6b)]
// CHECK:STDOUT:   %specific_impl_fn.loc170_5.4: <specific function> = specific_impl_function %impl.elem0.loc170_5.2, @Destroy.WithSelf.Op(%Result.loc165_9.1) [symbolic = %specific_impl_fn.loc170_5.4 (constants.%specific_impl_fn.761)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%result.param: @ModWith.loc168.%Result.as_type.loc167_54.1 (%Result.as_type), %x.param: @ModWith.loc168.%T.as_type.loc168_29.1 (%T.as_type.10f), %y.param: @ModWith.loc168.%U.as_type.loc167_36.1 (%U.as_type.2b8)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: @ModWith.loc168.%T.as_type.loc168_29.1 (%T.as_type.10f) = name_ref x, %x
// CHECK:STDOUT:     %y.ref: @ModWith.loc168.%U.as_type.loc167_36.1 (%U.as_type.2b8) = name_ref y, %y
// CHECK:STDOUT:     %ModWith.type.loc170: type = facet_type <@ModWith.1, @ModWith.1(constants.%U.as_type.2b8)> [symbolic = %ModWith.type.loc167_36.1 (constants.%ModWith.type.c6f)]
// CHECK:STDOUT:     %.loc170_5.1: @ModWith.loc168.%ModWith.assoc_type (%ModWith.assoc_type.66e) = specific_constant imports.%Core.import_ref.fcf, @ModWith.WithSelf(constants.%U.as_type.2b8, constants.%Self.797) [symbolic = %assoc1 (constants.%assoc1.3da)]
// CHECK:STDOUT:     %Op.ref: @ModWith.loc168.%ModWith.assoc_type (%ModWith.assoc_type.66e) = name_ref Op, %.loc170_5.1 [symbolic = %assoc1 (constants.%assoc1.3da)]
// CHECK:STDOUT:     %impl.elem1.loc170_5.1: @ModWith.loc168.%.loc170_5.4 (%.feb) = impl_witness_access constants.%ModWith.lookup_impl_witness.e3f, element1 [symbolic = %impl.elem1.loc170_5.2 (constants.%impl.elem1.1af)]
// CHECK:STDOUT:     %bound_method.loc170_5.1: <bound method> = bound_method %x.ref, %impl.elem1.loc170_5.1
// CHECK:STDOUT:     %specific_impl_fn.loc170_5.1: <specific function> = specific_impl_function %impl.elem1.loc170_5.1, @ModWith.WithSelf.Op(constants.%U.as_type.2b8, constants.%ModWith.facet.430) [symbolic = %specific_impl_fn.loc170_5.3 (constants.%specific_impl_fn.772)]
// CHECK:STDOUT:     %bound_method.loc170_5.2: <bound method> = bound_method %x.ref, %specific_impl_fn.loc170_5.1
// CHECK:STDOUT:     %.loc170_5.2: ref @ModWith.loc168.%Result.as_type.loc167_54.1 (%Result.as_type) = temporary_storage
// CHECK:STDOUT:     %ModWith.WithSelf.Op.call: init @ModWith.loc168.%Result.as_type.loc167_54.1 (%Result.as_type) to %.loc170_5.2 = call %bound_method.loc170_5.2(%x.ref, %y.ref)
// CHECK:STDOUT:     %.loc170_5.3: ref @ModWith.loc168.%Result.as_type.loc167_54.1 (%Result.as_type) = temporary %.loc170_5.2, %ModWith.WithSelf.Op.call
// CHECK:STDOUT:     %impl.elem0.loc170_5.1: @ModWith.loc168.%.loc170_5.5 (%.d5f) = impl_witness_access constants.%Destroy.lookup_impl_witness.d86, element0 [symbolic = %impl.elem0.loc170_5.2 (constants.%impl.elem0.a6b)]
// CHECK:STDOUT:     %bound_method.loc170_5.3: <bound method> = bound_method %.loc170_5.3, %impl.elem0.loc170_5.1
// CHECK:STDOUT:     %specific_impl_fn.loc170_5.2: <specific function> = specific_impl_function %impl.elem0.loc170_5.1, @Destroy.WithSelf.Op(constants.%Result) [symbolic = %specific_impl_fn.loc170_5.4 (constants.%specific_impl_fn.761)]
// CHECK:STDOUT:     %bound_method.loc170_5.4: <bound method> = bound_method %.loc170_5.3, %specific_impl_fn.loc170_5.2
// CHECK:STDOUT:     %Destroy.WithSelf.Op.call: init %empty_tuple.type = call %bound_method.loc170_5.4(%.loc170_5.3)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @AddAssignWith.loc225(%U.loc225_19.2: type, %T.loc225_29.2: @AddAssignWith.loc225.%AddAssignWith.type.loc225_52.1 (%AddAssignWith.type.4d3b38.2)) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %require_complete.loc227: <witness> = require_complete_type %AddAssignWith.type.loc225_52.1 [symbolic = %require_complete.loc227 (constants.%require_complete.cb0)]
// CHECK:STDOUT:   %AddAssignWith.assoc_type: type = assoc_entity_type @AddAssignWith.1, @AddAssignWith.1(%U.loc225_19.1) [symbolic = %AddAssignWith.assoc_type (constants.%AddAssignWith.assoc_type.ca2a80.2)]
// CHECK:STDOUT:   %assoc0: @AddAssignWith.loc225.%AddAssignWith.assoc_type (%AddAssignWith.assoc_type.ca2a80.2) = assoc_entity element0, imports.%Core.import_ref.7e1 [symbolic = %assoc0 (constants.%assoc0.f6c0ab.2)]
// CHECK:STDOUT:   %AddAssignWith.WithSelf.Op.type: type = fn_type @AddAssignWith.WithSelf.Op, @AddAssignWith.WithSelf(%U.loc225_19.1, %T.loc225_29.1) [symbolic = %AddAssignWith.WithSelf.Op.type (constants.%AddAssignWith.WithSelf.Op.type.df0722.3)]
// CHECK:STDOUT:   %.loc227_5.2: type = fn_type_with_self_type %AddAssignWith.WithSelf.Op.type, %T.loc225_29.1 [symbolic = %.loc227_5.2 (constants.%.8f6)]
// CHECK:STDOUT:   %AddAssignWith.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc225_29.1, @AddAssignWith.1, @AddAssignWith.1(%U.loc225_19.1) [symbolic = %AddAssignWith.lookup_impl_witness (constants.%AddAssignWith.lookup_impl_witness)]
// CHECK:STDOUT:   %impl.elem0.loc227_5.2: @AddAssignWith.loc225.%.loc227_5.2 (%.8f6) = impl_witness_access %AddAssignWith.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc227_5.2 (constants.%impl.elem0.c7b)]
// CHECK:STDOUT:   %specific_impl_fn.loc227_5.2: <specific function> = specific_impl_function %impl.elem0.loc227_5.2, @AddAssignWith.WithSelf.Op(%U.loc225_19.1, %T.loc225_29.1) [symbolic = %specific_impl_fn.loc227_5.2 (constants.%specific_impl_fn.fca)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: ref @AddAssignWith.loc225.%T.as_type.loc225_62.1 (%T.as_type.191), %y.param: @AddAssignWith.loc225.%U.loc225_19.1 (%U.67d)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: ref @AddAssignWith.loc225.%T.as_type.loc225_62.1 (%T.as_type.191) = name_ref x, %x
// CHECK:STDOUT:     %y.ref: @AddAssignWith.loc225.%U.loc225_19.1 (%U.67d) = name_ref y, %y
// CHECK:STDOUT:     %AddAssignWith.type.loc227: type = facet_type <@AddAssignWith.1, @AddAssignWith.1(constants.%U.67d)> [symbolic = %AddAssignWith.type.loc225_52.1 (constants.%AddAssignWith.type.4d3b38.2)]
// CHECK:STDOUT:     %.loc227_5.1: @AddAssignWith.loc225.%AddAssignWith.assoc_type (%AddAssignWith.assoc_type.ca2a80.2) = specific_constant imports.%Core.import_ref.016, @AddAssignWith.WithSelf(constants.%U.67d, constants.%Self.a00362.1) [symbolic = %assoc0 (constants.%assoc0.f6c0ab.2)]
// CHECK:STDOUT:     %Op.ref: @AddAssignWith.loc225.%AddAssignWith.assoc_type (%AddAssignWith.assoc_type.ca2a80.2) = name_ref Op, %.loc227_5.1 [symbolic = %assoc0 (constants.%assoc0.f6c0ab.2)]
// CHECK:STDOUT:     %impl.elem0.loc227_5.1: @AddAssignWith.loc225.%.loc227_5.2 (%.8f6) = impl_witness_access constants.%AddAssignWith.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc227_5.2 (constants.%impl.elem0.c7b)]
// CHECK:STDOUT:     %bound_method.loc227_5.1: <bound method> = bound_method %x.ref, %impl.elem0.loc227_5.1
// CHECK:STDOUT:     %specific_impl_fn.loc227_5.1: <specific function> = specific_impl_function %impl.elem0.loc227_5.1, @AddAssignWith.WithSelf.Op(constants.%U.67d, constants.%T.a00) [symbolic = %specific_impl_fn.loc227_5.2 (constants.%specific_impl_fn.fca)]
// CHECK:STDOUT:     %bound_method.loc227_5.2: <bound method> = bound_method %x.ref, %specific_impl_fn.loc227_5.1
// CHECK:STDOUT:     %AddAssignWith.WithSelf.Op.call: init %empty_tuple.type = call %bound_method.loc227_5.2(%x.ref, %y.ref)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @SubAssignWith.loc231(%U.loc231_19.2: type, %T.loc231_29.2: @SubAssignWith.loc231.%SubAssignWith.type.loc231_52.1 (%SubAssignWith.type.969cbb.2)) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %require_complete.loc233: <witness> = require_complete_type %SubAssignWith.type.loc231_52.1 [symbolic = %require_complete.loc233 (constants.%require_complete.22d3)]
// CHECK:STDOUT:   %SubAssignWith.assoc_type: type = assoc_entity_type @SubAssignWith.1, @SubAssignWith.1(%U.loc231_19.1) [symbolic = %SubAssignWith.assoc_type (constants.%SubAssignWith.assoc_type.4cb9a1.2)]
// CHECK:STDOUT:   %assoc0: @SubAssignWith.loc231.%SubAssignWith.assoc_type (%SubAssignWith.assoc_type.4cb9a1.2) = assoc_entity element0, imports.%Core.import_ref.5b4 [symbolic = %assoc0 (constants.%assoc0.6115cc.2)]
// CHECK:STDOUT:   %SubAssignWith.WithSelf.Op.type: type = fn_type @SubAssignWith.WithSelf.Op, @SubAssignWith.WithSelf(%U.loc231_19.1, %T.loc231_29.1) [symbolic = %SubAssignWith.WithSelf.Op.type (constants.%SubAssignWith.WithSelf.Op.type.317c02.3)]
// CHECK:STDOUT:   %.loc233_5.2: type = fn_type_with_self_type %SubAssignWith.WithSelf.Op.type, %T.loc231_29.1 [symbolic = %.loc233_5.2 (constants.%.307)]
// CHECK:STDOUT:   %SubAssignWith.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc231_29.1, @SubAssignWith.1, @SubAssignWith.1(%U.loc231_19.1) [symbolic = %SubAssignWith.lookup_impl_witness (constants.%SubAssignWith.lookup_impl_witness)]
// CHECK:STDOUT:   %impl.elem0.loc233_5.2: @SubAssignWith.loc231.%.loc233_5.2 (%.307) = impl_witness_access %SubAssignWith.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc233_5.2 (constants.%impl.elem0.e10)]
// CHECK:STDOUT:   %specific_impl_fn.loc233_5.2: <specific function> = specific_impl_function %impl.elem0.loc233_5.2, @SubAssignWith.WithSelf.Op(%U.loc231_19.1, %T.loc231_29.1) [symbolic = %specific_impl_fn.loc233_5.2 (constants.%specific_impl_fn.c8d)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: ref @SubAssignWith.loc231.%T.as_type.loc231_62.1 (%T.as_type.b8f), %y.param: @SubAssignWith.loc231.%U.loc231_19.1 (%U.67d)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: ref @SubAssignWith.loc231.%T.as_type.loc231_62.1 (%T.as_type.b8f) = name_ref x, %x
// CHECK:STDOUT:     %y.ref: @SubAssignWith.loc231.%U.loc231_19.1 (%U.67d) = name_ref y, %y
// CHECK:STDOUT:     %SubAssignWith.type.loc233: type = facet_type <@SubAssignWith.1, @SubAssignWith.1(constants.%U.67d)> [symbolic = %SubAssignWith.type.loc231_52.1 (constants.%SubAssignWith.type.969cbb.2)]
// CHECK:STDOUT:     %.loc233_5.1: @SubAssignWith.loc231.%SubAssignWith.assoc_type (%SubAssignWith.assoc_type.4cb9a1.2) = specific_constant imports.%Core.import_ref.405, @SubAssignWith.WithSelf(constants.%U.67d, constants.%Self.1de093.1) [symbolic = %assoc0 (constants.%assoc0.6115cc.2)]
// CHECK:STDOUT:     %Op.ref: @SubAssignWith.loc231.%SubAssignWith.assoc_type (%SubAssignWith.assoc_type.4cb9a1.2) = name_ref Op, %.loc233_5.1 [symbolic = %assoc0 (constants.%assoc0.6115cc.2)]
// CHECK:STDOUT:     %impl.elem0.loc233_5.1: @SubAssignWith.loc231.%.loc233_5.2 (%.307) = impl_witness_access constants.%SubAssignWith.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc233_5.2 (constants.%impl.elem0.e10)]
// CHECK:STDOUT:     %bound_method.loc233_5.1: <bound method> = bound_method %x.ref, %impl.elem0.loc233_5.1
// CHECK:STDOUT:     %specific_impl_fn.loc233_5.1: <specific function> = specific_impl_function %impl.elem0.loc233_5.1, @SubAssignWith.WithSelf.Op(constants.%U.67d, constants.%T.1de) [symbolic = %specific_impl_fn.loc233_5.2 (constants.%specific_impl_fn.c8d)]
// CHECK:STDOUT:     %bound_method.loc233_5.2: <bound method> = bound_method %x.ref, %specific_impl_fn.loc233_5.1
// CHECK:STDOUT:     %SubAssignWith.WithSelf.Op.call: init %empty_tuple.type = call %bound_method.loc233_5.2(%x.ref, %y.ref)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @MulAssignWith.loc237(%U.loc237_19.2: type, %T.loc237_29.2: @MulAssignWith.loc237.%MulAssignWith.type.loc237_52.1 (%MulAssignWith.type.ccc9e0.2)) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %require_complete.loc239: <witness> = require_complete_type %MulAssignWith.type.loc237_52.1 [symbolic = %require_complete.loc239 (constants.%require_complete.6dc)]
// CHECK:STDOUT:   %MulAssignWith.assoc_type: type = assoc_entity_type @MulAssignWith.1, @MulAssignWith.1(%U.loc237_19.1) [symbolic = %MulAssignWith.assoc_type (constants.%MulAssignWith.assoc_type.595561.2)]
// CHECK:STDOUT:   %assoc0: @MulAssignWith.loc237.%MulAssignWith.assoc_type (%MulAssignWith.assoc_type.595561.2) = assoc_entity element0, imports.%Core.import_ref.69b [symbolic = %assoc0 (constants.%assoc0.71e6fb.2)]
// CHECK:STDOUT:   %MulAssignWith.WithSelf.Op.type: type = fn_type @MulAssignWith.WithSelf.Op, @MulAssignWith.WithSelf(%U.loc237_19.1, %T.loc237_29.1) [symbolic = %MulAssignWith.WithSelf.Op.type (constants.%MulAssignWith.WithSelf.Op.type.31cf70.3)]
// CHECK:STDOUT:   %.loc239_5.2: type = fn_type_with_self_type %MulAssignWith.WithSelf.Op.type, %T.loc237_29.1 [symbolic = %.loc239_5.2 (constants.%.1b8)]
// CHECK:STDOUT:   %MulAssignWith.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc237_29.1, @MulAssignWith.1, @MulAssignWith.1(%U.loc237_19.1) [symbolic = %MulAssignWith.lookup_impl_witness (constants.%MulAssignWith.lookup_impl_witness)]
// CHECK:STDOUT:   %impl.elem0.loc239_5.2: @MulAssignWith.loc237.%.loc239_5.2 (%.1b8) = impl_witness_access %MulAssignWith.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc239_5.2 (constants.%impl.elem0.cd0)]
// CHECK:STDOUT:   %specific_impl_fn.loc239_5.2: <specific function> = specific_impl_function %impl.elem0.loc239_5.2, @MulAssignWith.WithSelf.Op(%U.loc237_19.1, %T.loc237_29.1) [symbolic = %specific_impl_fn.loc239_5.2 (constants.%specific_impl_fn.397)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: ref @MulAssignWith.loc237.%T.as_type.loc237_62.1 (%T.as_type.6a0), %y.param: @MulAssignWith.loc237.%U.loc237_19.1 (%U.67d)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: ref @MulAssignWith.loc237.%T.as_type.loc237_62.1 (%T.as_type.6a0) = name_ref x, %x
// CHECK:STDOUT:     %y.ref: @MulAssignWith.loc237.%U.loc237_19.1 (%U.67d) = name_ref y, %y
// CHECK:STDOUT:     %MulAssignWith.type.loc239: type = facet_type <@MulAssignWith.1, @MulAssignWith.1(constants.%U.67d)> [symbolic = %MulAssignWith.type.loc237_52.1 (constants.%MulAssignWith.type.ccc9e0.2)]
// CHECK:STDOUT:     %.loc239_5.1: @MulAssignWith.loc237.%MulAssignWith.assoc_type (%MulAssignWith.assoc_type.595561.2) = specific_constant imports.%Core.import_ref.257, @MulAssignWith.WithSelf(constants.%U.67d, constants.%Self.059e93.1) [symbolic = %assoc0 (constants.%assoc0.71e6fb.2)]
// CHECK:STDOUT:     %Op.ref: @MulAssignWith.loc237.%MulAssignWith.assoc_type (%MulAssignWith.assoc_type.595561.2) = name_ref Op, %.loc239_5.1 [symbolic = %assoc0 (constants.%assoc0.71e6fb.2)]
// CHECK:STDOUT:     %impl.elem0.loc239_5.1: @MulAssignWith.loc237.%.loc239_5.2 (%.1b8) = impl_witness_access constants.%MulAssignWith.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc239_5.2 (constants.%impl.elem0.cd0)]
// CHECK:STDOUT:     %bound_method.loc239_5.1: <bound method> = bound_method %x.ref, %impl.elem0.loc239_5.1
// CHECK:STDOUT:     %specific_impl_fn.loc239_5.1: <specific function> = specific_impl_function %impl.elem0.loc239_5.1, @MulAssignWith.WithSelf.Op(constants.%U.67d, constants.%T.059) [symbolic = %specific_impl_fn.loc239_5.2 (constants.%specific_impl_fn.397)]
// CHECK:STDOUT:     %bound_method.loc239_5.2: <bound method> = bound_method %x.ref, %specific_impl_fn.loc239_5.1
// CHECK:STDOUT:     %MulAssignWith.WithSelf.Op.call: init %empty_tuple.type = call %bound_method.loc239_5.2(%x.ref, %y.ref)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @DivAssignWith.loc243(%U.loc243_19.2: type, %T.loc243_29.2: @DivAssignWith.loc243.%DivAssignWith.type.loc243_52.1 (%DivAssignWith.type.be08dd.2)) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %require_complete.loc245: <witness> = require_complete_type %DivAssignWith.type.loc243_52.1 [symbolic = %require_complete.loc245 (constants.%require_complete.542)]
// CHECK:STDOUT:   %DivAssignWith.assoc_type: type = assoc_entity_type @DivAssignWith.1, @DivAssignWith.1(%U.loc243_19.1) [symbolic = %DivAssignWith.assoc_type (constants.%DivAssignWith.assoc_type.8b4b5b.2)]
// CHECK:STDOUT:   %assoc0: @DivAssignWith.loc243.%DivAssignWith.assoc_type (%DivAssignWith.assoc_type.8b4b5b.2) = assoc_entity element0, imports.%Core.import_ref.799 [symbolic = %assoc0 (constants.%assoc0.399004.2)]
// CHECK:STDOUT:   %DivAssignWith.WithSelf.Op.type: type = fn_type @DivAssignWith.WithSelf.Op, @DivAssignWith.WithSelf(%U.loc243_19.1, %T.loc243_29.1) [symbolic = %DivAssignWith.WithSelf.Op.type (constants.%DivAssignWith.WithSelf.Op.type.ded5d7.3)]
// CHECK:STDOUT:   %.loc245_5.2: type = fn_type_with_self_type %DivAssignWith.WithSelf.Op.type, %T.loc243_29.1 [symbolic = %.loc245_5.2 (constants.%.607)]
// CHECK:STDOUT:   %DivAssignWith.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc243_29.1, @DivAssignWith.1, @DivAssignWith.1(%U.loc243_19.1) [symbolic = %DivAssignWith.lookup_impl_witness (constants.%DivAssignWith.lookup_impl_witness)]
// CHECK:STDOUT:   %impl.elem0.loc245_5.2: @DivAssignWith.loc243.%.loc245_5.2 (%.607) = impl_witness_access %DivAssignWith.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc245_5.2 (constants.%impl.elem0.46f)]
// CHECK:STDOUT:   %specific_impl_fn.loc245_5.2: <specific function> = specific_impl_function %impl.elem0.loc245_5.2, @DivAssignWith.WithSelf.Op(%U.loc243_19.1, %T.loc243_29.1) [symbolic = %specific_impl_fn.loc245_5.2 (constants.%specific_impl_fn.666)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: ref @DivAssignWith.loc243.%T.as_type.loc243_62.1 (%T.as_type.efd), %y.param: @DivAssignWith.loc243.%U.loc243_19.1 (%U.67d)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: ref @DivAssignWith.loc243.%T.as_type.loc243_62.1 (%T.as_type.efd) = name_ref x, %x
// CHECK:STDOUT:     %y.ref: @DivAssignWith.loc243.%U.loc243_19.1 (%U.67d) = name_ref y, %y
// CHECK:STDOUT:     %DivAssignWith.type.loc245: type = facet_type <@DivAssignWith.1, @DivAssignWith.1(constants.%U.67d)> [symbolic = %DivAssignWith.type.loc243_52.1 (constants.%DivAssignWith.type.be08dd.2)]
// CHECK:STDOUT:     %.loc245_5.1: @DivAssignWith.loc243.%DivAssignWith.assoc_type (%DivAssignWith.assoc_type.8b4b5b.2) = specific_constant imports.%Core.import_ref.b1f, @DivAssignWith.WithSelf(constants.%U.67d, constants.%Self.aa8835.1) [symbolic = %assoc0 (constants.%assoc0.399004.2)]
// CHECK:STDOUT:     %Op.ref: @DivAssignWith.loc243.%DivAssignWith.assoc_type (%DivAssignWith.assoc_type.8b4b5b.2) = name_ref Op, %.loc245_5.1 [symbolic = %assoc0 (constants.%assoc0.399004.2)]
// CHECK:STDOUT:     %impl.elem0.loc245_5.1: @DivAssignWith.loc243.%.loc245_5.2 (%.607) = impl_witness_access constants.%DivAssignWith.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc245_5.2 (constants.%impl.elem0.46f)]
// CHECK:STDOUT:     %bound_method.loc245_5.1: <bound method> = bound_method %x.ref, %impl.elem0.loc245_5.1
// CHECK:STDOUT:     %specific_impl_fn.loc245_5.1: <specific function> = specific_impl_function %impl.elem0.loc245_5.1, @DivAssignWith.WithSelf.Op(constants.%U.67d, constants.%T.aa8) [symbolic = %specific_impl_fn.loc245_5.2 (constants.%specific_impl_fn.666)]
// CHECK:STDOUT:     %bound_method.loc245_5.2: <bound method> = bound_method %x.ref, %specific_impl_fn.loc245_5.1
// CHECK:STDOUT:     %DivAssignWith.WithSelf.Op.call: init %empty_tuple.type = call %bound_method.loc245_5.2(%x.ref, %y.ref)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @ModAssignWith.loc249(%U.loc249_19.2: type, %T.loc249_29.2: @ModAssignWith.loc249.%ModAssignWith.type.loc249_52.1 (%ModAssignWith.type.c65b77.2)) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %require_complete.loc251: <witness> = require_complete_type %ModAssignWith.type.loc249_52.1 [symbolic = %require_complete.loc251 (constants.%require_complete.8bf)]
// CHECK:STDOUT:   %ModAssignWith.assoc_type: type = assoc_entity_type @ModAssignWith.1, @ModAssignWith.1(%U.loc249_19.1) [symbolic = %ModAssignWith.assoc_type (constants.%ModAssignWith.assoc_type.1462b9.2)]
// CHECK:STDOUT:   %assoc0: @ModAssignWith.loc249.%ModAssignWith.assoc_type (%ModAssignWith.assoc_type.1462b9.2) = assoc_entity element0, imports.%Core.import_ref.51c [symbolic = %assoc0 (constants.%assoc0.378445.2)]
// CHECK:STDOUT:   %ModAssignWith.WithSelf.Op.type: type = fn_type @ModAssignWith.WithSelf.Op, @ModAssignWith.WithSelf(%U.loc249_19.1, %T.loc249_29.1) [symbolic = %ModAssignWith.WithSelf.Op.type (constants.%ModAssignWith.WithSelf.Op.type.5452e9.3)]
// CHECK:STDOUT:   %.loc251_5.2: type = fn_type_with_self_type %ModAssignWith.WithSelf.Op.type, %T.loc249_29.1 [symbolic = %.loc251_5.2 (constants.%.dfa)]
// CHECK:STDOUT:   %ModAssignWith.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc249_29.1, @ModAssignWith.1, @ModAssignWith.1(%U.loc249_19.1) [symbolic = %ModAssignWith.lookup_impl_witness (constants.%ModAssignWith.lookup_impl_witness)]
// CHECK:STDOUT:   %impl.elem0.loc251_5.2: @ModAssignWith.loc249.%.loc251_5.2 (%.dfa) = impl_witness_access %ModAssignWith.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc251_5.2 (constants.%impl.elem0.18c)]
// CHECK:STDOUT:   %specific_impl_fn.loc251_5.2: <specific function> = specific_impl_function %impl.elem0.loc251_5.2, @ModAssignWith.WithSelf.Op(%U.loc249_19.1, %T.loc249_29.1) [symbolic = %specific_impl_fn.loc251_5.2 (constants.%specific_impl_fn.28f)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: ref @ModAssignWith.loc249.%T.as_type.loc249_62.1 (%T.as_type.4c2), %y.param: @ModAssignWith.loc249.%U.loc249_19.1 (%U.67d)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: ref @ModAssignWith.loc249.%T.as_type.loc249_62.1 (%T.as_type.4c2) = name_ref x, %x
// CHECK:STDOUT:     %y.ref: @ModAssignWith.loc249.%U.loc249_19.1 (%U.67d) = name_ref y, %y
// CHECK:STDOUT:     %ModAssignWith.type.loc251: type = facet_type <@ModAssignWith.1, @ModAssignWith.1(constants.%U.67d)> [symbolic = %ModAssignWith.type.loc249_52.1 (constants.%ModAssignWith.type.c65b77.2)]
// CHECK:STDOUT:     %.loc251_5.1: @ModAssignWith.loc249.%ModAssignWith.assoc_type (%ModAssignWith.assoc_type.1462b9.2) = specific_constant imports.%Core.import_ref.b81, @ModAssignWith.WithSelf(constants.%U.67d, constants.%Self.d35c46.1) [symbolic = %assoc0 (constants.%assoc0.378445.2)]
// CHECK:STDOUT:     %Op.ref: @ModAssignWith.loc249.%ModAssignWith.assoc_type (%ModAssignWith.assoc_type.1462b9.2) = name_ref Op, %.loc251_5.1 [symbolic = %assoc0 (constants.%assoc0.378445.2)]
// CHECK:STDOUT:     %impl.elem0.loc251_5.1: @ModAssignWith.loc249.%.loc251_5.2 (%.dfa) = impl_witness_access constants.%ModAssignWith.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc251_5.2 (constants.%impl.elem0.18c)]
// CHECK:STDOUT:     %bound_method.loc251_5.1: <bound method> = bound_method %x.ref, %impl.elem0.loc251_5.1
// CHECK:STDOUT:     %specific_impl_fn.loc251_5.1: <specific function> = specific_impl_function %impl.elem0.loc251_5.1, @ModAssignWith.WithSelf.Op(constants.%U.67d, constants.%T.d35) [symbolic = %specific_impl_fn.loc251_5.2 (constants.%specific_impl_fn.28f)]
// CHECK:STDOUT:     %bound_method.loc251_5.2: <bound method> = bound_method %x.ref, %specific_impl_fn.loc251_5.1
// CHECK:STDOUT:     %ModAssignWith.WithSelf.Op.call: init %empty_tuple.type = call %bound_method.loc251_5.2(%x.ref, %y.ref)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @TestInc(%T.loc278_13.2: %Inc.type) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Inc.WithSelf.Op.type: type = fn_type @Inc.WithSelf.Op, @Inc.WithSelf(%T.loc278_13.1) [symbolic = %Inc.WithSelf.Op.type (constants.%Inc.WithSelf.Op.type.623399.2)]
// CHECK:STDOUT:   %.loc280: type = fn_type_with_self_type %Inc.WithSelf.Op.type, %T.loc278_13.1 [symbolic = %.loc280 (constants.%.ad4)]
// CHECK:STDOUT:   %Inc.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc278_13.1, @Inc [symbolic = %Inc.lookup_impl_witness (constants.%Inc.lookup_impl_witness)]
// CHECK:STDOUT:   %impl.elem0.loc280_3.2: @TestInc.%.loc280 (%.ad4) = impl_witness_access %Inc.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc280_3.2 (constants.%impl.elem0.489)]
// CHECK:STDOUT:   %specific_impl_fn.loc280_3.2: <specific function> = specific_impl_function %impl.elem0.loc280_3.2, @Inc.WithSelf.Op(%T.loc278_13.1) [symbolic = %specific_impl_fn.loc280_3.2 (constants.%specific_impl_fn.71c)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: ref @TestInc.%T.as_type.loc278_33.1 (%T.as_type.3f5)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: ref @TestInc.%T.as_type.loc278_33.1 (%T.as_type.3f5) = name_ref x, %x
// CHECK:STDOUT:     %impl.elem0.loc280_3.1: @TestInc.%.loc280 (%.ad4) = impl_witness_access constants.%Inc.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc280_3.2 (constants.%impl.elem0.489)]
// CHECK:STDOUT:     %bound_method.loc280_3.1: <bound method> = bound_method %x.ref, %impl.elem0.loc280_3.1
// CHECK:STDOUT:     %specific_impl_fn.loc280_3.1: <specific function> = specific_impl_function %impl.elem0.loc280_3.1, @Inc.WithSelf.Op(constants.%T.b7e) [symbolic = %specific_impl_fn.loc280_3.2 (constants.%specific_impl_fn.71c)]
// CHECK:STDOUT:     %bound_method.loc280_3.2: <bound method> = bound_method %x.ref, %specific_impl_fn.loc280_3.1
// CHECK:STDOUT:     %Inc.WithSelf.Op.call: init %empty_tuple.type = call %bound_method.loc280_3.2(%x.ref)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @TestDec(%T.loc284_13.2: %Dec.type) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Dec.WithSelf.Op.type: type = fn_type @Dec.WithSelf.Op, @Dec.WithSelf(%T.loc284_13.1) [symbolic = %Dec.WithSelf.Op.type (constants.%Dec.WithSelf.Op.type.41adba.2)]
// CHECK:STDOUT:   %.loc286: type = fn_type_with_self_type %Dec.WithSelf.Op.type, %T.loc284_13.1 [symbolic = %.loc286 (constants.%.41c)]
// CHECK:STDOUT:   %Dec.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc284_13.1, @Dec [symbolic = %Dec.lookup_impl_witness (constants.%Dec.lookup_impl_witness)]
// CHECK:STDOUT:   %impl.elem0.loc286_3.2: @TestDec.%.loc286 (%.41c) = impl_witness_access %Dec.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc286_3.2 (constants.%impl.elem0.64b)]
// CHECK:STDOUT:   %specific_impl_fn.loc286_3.2: <specific function> = specific_impl_function %impl.elem0.loc286_3.2, @Dec.WithSelf.Op(%T.loc284_13.1) [symbolic = %specific_impl_fn.loc286_3.2 (constants.%specific_impl_fn.91c)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: ref @TestDec.%T.as_type.loc284_33.1 (%T.as_type.ead)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: ref @TestDec.%T.as_type.loc284_33.1 (%T.as_type.ead) = name_ref x, %x
// CHECK:STDOUT:     %impl.elem0.loc286_3.1: @TestDec.%.loc286 (%.41c) = impl_witness_access constants.%Dec.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc286_3.2 (constants.%impl.elem0.64b)]
// CHECK:STDOUT:     %bound_method.loc286_3.1: <bound method> = bound_method %x.ref, %impl.elem0.loc286_3.1
// CHECK:STDOUT:     %specific_impl_fn.loc286_3.1: <specific function> = specific_impl_function %impl.elem0.loc286_3.1, @Dec.WithSelf.Op(constants.%T.ce1) [symbolic = %specific_impl_fn.loc286_3.2 (constants.%specific_impl_fn.91c)]
// CHECK:STDOUT:     %bound_method.loc286_3.2: <bound method> = bound_method %x.ref, %specific_impl_fn.loc286_3.1
// CHECK:STDOUT:     %Dec.WithSelf.Op.call: init %empty_tuple.type = call %bound_method.loc286_3.2(%x.ref)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @TestNegate(%Result.loc292_9.2: %Destroy.type, %T.loc293_4.2: @TestNegate.%facet_type (%facet_type.0fc)) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Negate.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc293_4.1, @Negate [symbolic = %Negate.lookup_impl_witness (constants.%Negate.lookup_impl_witness.783)]
// CHECK:STDOUT:   %Negate.facet: %Negate.type = facet_value %T.as_type.loc294_24.1, (%Negate.lookup_impl_witness) [symbolic = %Negate.facet (constants.%Negate.facet.758)]
// CHECK:STDOUT:   %Negate.WithSelf.Op.type: type = fn_type @Negate.WithSelf.Op, @Negate.WithSelf(%Negate.facet) [symbolic = %Negate.WithSelf.Op.type (constants.%Negate.WithSelf.Op.type.899)]
// CHECK:STDOUT:   %.loc296_3.3: type = fn_type_with_self_type %Negate.WithSelf.Op.type, %Negate.facet [symbolic = %.loc296_3.3 (constants.%.0d7)]
// CHECK:STDOUT:   %impl.elem1.loc296_3.2: @TestNegate.%.loc296_3.3 (%.0d7) = impl_witness_access %Negate.lookup_impl_witness, element1 [symbolic = %impl.elem1.loc296_3.2 (constants.%impl.elem1.ec8)]
// CHECK:STDOUT:   %specific_impl_fn.loc296_3.3: <specific function> = specific_impl_function %impl.elem1.loc296_3.2, @Negate.WithSelf.Op(%Negate.facet) [symbolic = %specific_impl_fn.loc296_3.3 (constants.%specific_impl_fn.f22)]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Result.loc292_9.1) [symbolic = %Destroy.WithSelf.Op.type (constants.%Destroy.WithSelf.Op.type.cb2e47.2)]
// CHECK:STDOUT:   %.loc296_3.4: type = fn_type_with_self_type %Destroy.WithSelf.Op.type, %Result.loc292_9.1 [symbolic = %.loc296_3.4 (constants.%.d5f)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %Result.loc292_9.1, @Destroy [symbolic = %Destroy.lookup_impl_witness (constants.%Destroy.lookup_impl_witness.d86)]
// CHECK:STDOUT:   %impl.elem0.loc296_3.2: @TestNegate.%.loc296_3.4 (%.d5f) = impl_witness_access %Destroy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc296_3.2 (constants.%impl.elem0.a6b)]
// CHECK:STDOUT:   %specific_impl_fn.loc296_3.4: <specific function> = specific_impl_function %impl.elem0.loc296_3.2, @Destroy.WithSelf.Op(%Result.loc292_9.1) [symbolic = %specific_impl_fn.loc296_3.4 (constants.%specific_impl_fn.761)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%y.param: @TestNegate.%Result.as_type.loc293_50.1 (%Result.as_type), %x.param: @TestNegate.%T.as_type.loc294_24.1 (%T.as_type.c82)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: @TestNegate.%T.as_type.loc294_24.1 (%T.as_type.c82) = name_ref x, %x
// CHECK:STDOUT:     %impl.elem1.loc296_3.1: @TestNegate.%.loc296_3.3 (%.0d7) = impl_witness_access constants.%Negate.lookup_impl_witness.783, element1 [symbolic = %impl.elem1.loc296_3.2 (constants.%impl.elem1.ec8)]
// CHECK:STDOUT:     %bound_method.loc296_3.1: <bound method> = bound_method %x.ref, %impl.elem1.loc296_3.1
// CHECK:STDOUT:     %specific_impl_fn.loc296_3.1: <specific function> = specific_impl_function %impl.elem1.loc296_3.1, @Negate.WithSelf.Op(constants.%Negate.facet.758) [symbolic = %specific_impl_fn.loc296_3.3 (constants.%specific_impl_fn.f22)]
// CHECK:STDOUT:     %bound_method.loc296_3.2: <bound method> = bound_method %x.ref, %specific_impl_fn.loc296_3.1
// CHECK:STDOUT:     %.loc296_3.1: ref @TestNegate.%Result.as_type.loc293_50.1 (%Result.as_type) = temporary_storage
// CHECK:STDOUT:     %Negate.WithSelf.Op.call: init @TestNegate.%Result.as_type.loc293_50.1 (%Result.as_type) to %.loc296_3.1 = call %bound_method.loc296_3.2(%x.ref)
// CHECK:STDOUT:     %.loc296_3.2: ref @TestNegate.%Result.as_type.loc293_50.1 (%Result.as_type) = temporary %.loc296_3.1, %Negate.WithSelf.Op.call
// CHECK:STDOUT:     %impl.elem0.loc296_3.1: @TestNegate.%.loc296_3.4 (%.d5f) = impl_witness_access constants.%Destroy.lookup_impl_witness.d86, element0 [symbolic = %impl.elem0.loc296_3.2 (constants.%impl.elem0.a6b)]
// CHECK:STDOUT:     %bound_method.loc296_3.3: <bound method> = bound_method %.loc296_3.2, %impl.elem0.loc296_3.1
// CHECK:STDOUT:     %specific_impl_fn.loc296_3.2: <specific function> = specific_impl_function %impl.elem0.loc296_3.1, @Destroy.WithSelf.Op(constants.%Result) [symbolic = %specific_impl_fn.loc296_3.4 (constants.%specific_impl_fn.761)]
// CHECK:STDOUT:     %bound_method.loc296_3.4: <bound method> = bound_method %.loc296_3.2, %specific_impl_fn.loc296_3.2
// CHECK:STDOUT:     %Destroy.WithSelf.Op.call: init %empty_tuple.type = call %bound_method.loc296_3.4(%.loc296_3.2)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AddWith.loc124(constants.%Result, constants.%U.0bd, constants.%T.25f) {
// CHECK:STDOUT:   %Result.loc121_9.1 => constants.%Result
// CHECK:STDOUT:   %U.loc122_4.1 => constants.%U.0bd
// CHECK:STDOUT:   %U.as_type.loc123_36.1 => constants.%U.as_type.2b8
// CHECK:STDOUT:   %AddWith.type.loc123_36.1 => constants.%AddWith.type.07d
// CHECK:STDOUT:   %facet_type.loc123_20 => constants.%facet_type.74d
// CHECK:STDOUT:   %.Self.1 => constants.%.Self.5c0
// CHECK:STDOUT:   %require_complete.loc123 => constants.%require_complete.311
// CHECK:STDOUT:   %.Self.as_type.loc123_44.1 => constants.%.Self.as_type.f21
// CHECK:STDOUT:   %AddWith.assoc_type => constants.%AddWith.assoc_type.8c2
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.7d3
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.loc123 => constants.%AddWith.lookup_impl_witness.5a0
// CHECK:STDOUT:   %impl.elem0.loc123_44.1 => constants.%impl.elem0.08a
// CHECK:STDOUT:   %Result.as_type.loc123_54.1 => constants.%Result.as_type
// CHECK:STDOUT:   %facet_type.loc123_38 => constants.%facet_type.024
// CHECK:STDOUT:   %T.loc123_4.1 => constants.%T.25f
// CHECK:STDOUT:   %pattern_type.loc123 => constants.%pattern_type.077a
// CHECK:STDOUT:   %pattern_type.loc124_16 => constants.%pattern_type.eca0e4.2
// CHECK:STDOUT:   %T.as_type.loc124_29.1 => constants.%T.as_type.ab9
// CHECK:STDOUT:   %pattern_type.loc124_27 => constants.%pattern_type.7bb
// CHECK:STDOUT:   %pattern_type.loc124_33 => constants.%pattern_type.711
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @SubWith.loc135(constants.%Result, constants.%U.0bd, constants.%T.970) {
// CHECK:STDOUT:   %Result.loc132_9.1 => constants.%Result
// CHECK:STDOUT:   %U.loc133_4.1 => constants.%U.0bd
// CHECK:STDOUT:   %U.as_type.loc134_36.1 => constants.%U.as_type.2b8
// CHECK:STDOUT:   %SubWith.type.loc134_36.1 => constants.%SubWith.type.593
// CHECK:STDOUT:   %facet_type.loc134_20 => constants.%facet_type.db3
// CHECK:STDOUT:   %.Self.1 => constants.%.Self.db3
// CHECK:STDOUT:   %require_complete.loc134 => constants.%require_complete.b71
// CHECK:STDOUT:   %.Self.as_type.loc134_44.1 => constants.%.Self.as_type.2a6
// CHECK:STDOUT:   %SubWith.assoc_type => constants.%SubWith.assoc_type.065
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.55d
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.loc134 => constants.%SubWith.lookup_impl_witness.98d
// CHECK:STDOUT:   %impl.elem0.loc134_44.1 => constants.%impl.elem0.bbb
// CHECK:STDOUT:   %Result.as_type.loc134_54.1 => constants.%Result.as_type
// CHECK:STDOUT:   %facet_type.loc134_38 => constants.%facet_type.bea
// CHECK:STDOUT:   %T.loc134_4.1 => constants.%T.970
// CHECK:STDOUT:   %pattern_type.loc134 => constants.%pattern_type.265
// CHECK:STDOUT:   %pattern_type.loc135_16 => constants.%pattern_type.eca0e4.2
// CHECK:STDOUT:   %T.as_type.loc135_29.1 => constants.%T.as_type.474
// CHECK:STDOUT:   %pattern_type.loc135_27 => constants.%pattern_type.b6a
// CHECK:STDOUT:   %pattern_type.loc135_33 => constants.%pattern_type.711
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @MulWith.loc146(constants.%Result, constants.%U.0bd, constants.%T.438) {
// CHECK:STDOUT:   %Result.loc143_9.1 => constants.%Result
// CHECK:STDOUT:   %U.loc144_4.1 => constants.%U.0bd
// CHECK:STDOUT:   %U.as_type.loc145_36.1 => constants.%U.as_type.2b8
// CHECK:STDOUT:   %MulWith.type.loc145_36.1 => constants.%MulWith.type.9532
// CHECK:STDOUT:   %facet_type.loc145_20 => constants.%facet_type.17c
// CHECK:STDOUT:   %.Self.1 => constants.%.Self.afa
// CHECK:STDOUT:   %require_complete.loc145 => constants.%require_complete.665
// CHECK:STDOUT:   %.Self.as_type.loc145_44.1 => constants.%.Self.as_type.9cb
// CHECK:STDOUT:   %MulWith.assoc_type => constants.%MulWith.assoc_type.97c
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.1354
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.loc145 => constants.%MulWith.lookup_impl_witness.05e
// CHECK:STDOUT:   %impl.elem0.loc145_44.1 => constants.%impl.elem0.972
// CHECK:STDOUT:   %Result.as_type.loc145_54.1 => constants.%Result.as_type
// CHECK:STDOUT:   %facet_type.loc145_38 => constants.%facet_type.91e
// CHECK:STDOUT:   %T.loc145_4.1 => constants.%T.438
// CHECK:STDOUT:   %pattern_type.loc145 => constants.%pattern_type.b93
// CHECK:STDOUT:   %pattern_type.loc146_16 => constants.%pattern_type.eca0e4.2
// CHECK:STDOUT:   %T.as_type.loc146_29.1 => constants.%T.as_type.68f
// CHECK:STDOUT:   %pattern_type.loc146_27 => constants.%pattern_type.b74
// CHECK:STDOUT:   %pattern_type.loc146_33 => constants.%pattern_type.711
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @DivWith.loc157(constants.%Result, constants.%U.0bd, constants.%T.30e) {
// CHECK:STDOUT:   %Result.loc154_9.1 => constants.%Result
// CHECK:STDOUT:   %U.loc155_4.1 => constants.%U.0bd
// CHECK:STDOUT:   %U.as_type.loc156_36.1 => constants.%U.as_type.2b8
// CHECK:STDOUT:   %DivWith.type.loc156_36.1 => constants.%DivWith.type.0e2
// CHECK:STDOUT:   %facet_type.loc156_20 => constants.%facet_type.94d
// CHECK:STDOUT:   %.Self.1 => constants.%.Self.5a5
// CHECK:STDOUT:   %require_complete.loc156 => constants.%require_complete.3a4
// CHECK:STDOUT:   %.Self.as_type.loc156_44.1 => constants.%.Self.as_type.dad
// CHECK:STDOUT:   %DivWith.assoc_type => constants.%DivWith.assoc_type.06d
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.1db
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.loc156 => constants.%DivWith.lookup_impl_witness.fbf
// CHECK:STDOUT:   %impl.elem0.loc156_44.1 => constants.%impl.elem0.409
// CHECK:STDOUT:   %Result.as_type.loc156_54.1 => constants.%Result.as_type
// CHECK:STDOUT:   %facet_type.loc156_38 => constants.%facet_type.6e6
// CHECK:STDOUT:   %T.loc156_4.1 => constants.%T.30e
// CHECK:STDOUT:   %pattern_type.loc156 => constants.%pattern_type.4ef
// CHECK:STDOUT:   %pattern_type.loc157_16 => constants.%pattern_type.eca0e4.2
// CHECK:STDOUT:   %T.as_type.loc157_29.1 => constants.%T.as_type.cfe
// CHECK:STDOUT:   %pattern_type.loc157_27 => constants.%pattern_type.daa
// CHECK:STDOUT:   %pattern_type.loc157_33 => constants.%pattern_type.711
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ModWith.loc168(constants.%Result, constants.%U.0bd, constants.%T.a77) {
// CHECK:STDOUT:   %Result.loc165_9.1 => constants.%Result
// CHECK:STDOUT:   %U.loc166_4.1 => constants.%U.0bd
// CHECK:STDOUT:   %U.as_type.loc167_36.1 => constants.%U.as_type.2b8
// CHECK:STDOUT:   %ModWith.type.loc167_36.1 => constants.%ModWith.type.c6f
// CHECK:STDOUT:   %facet_type.loc167_20 => constants.%facet_type.428
// CHECK:STDOUT:   %.Self.1 => constants.%.Self.2d4
// CHECK:STDOUT:   %require_complete.loc167 => constants.%require_complete.9441
// CHECK:STDOUT:   %.Self.as_type.loc167_44.1 => constants.%.Self.as_type.12a6
// CHECK:STDOUT:   %ModWith.assoc_type => constants.%ModWith.assoc_type.66e
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.fd6
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.loc167 => constants.%ModWith.lookup_impl_witness.afa
// CHECK:STDOUT:   %impl.elem0.loc167_44.1 => constants.%impl.elem0.44b
// CHECK:STDOUT:   %Result.as_type.loc167_54.1 => constants.%Result.as_type
// CHECK:STDOUT:   %facet_type.loc167_38 => constants.%facet_type.3a6
// CHECK:STDOUT:   %T.loc167_4.1 => constants.%T.a77
// CHECK:STDOUT:   %pattern_type.loc167 => constants.%pattern_type.50d
// CHECK:STDOUT:   %pattern_type.loc168_16 => constants.%pattern_type.eca0e4.2
// CHECK:STDOUT:   %T.as_type.loc168_29.1 => constants.%T.as_type.10f
// CHECK:STDOUT:   %pattern_type.loc168_27 => constants.%pattern_type.8db
// CHECK:STDOUT:   %pattern_type.loc168_33 => constants.%pattern_type.711
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AddWith.loc124(constants.%Destroy.facet.e06, constants.%Destroy.facet.e06, constants.%facet_value.c79) {
// CHECK:STDOUT:   %Result.loc121_9.1 => constants.%Destroy.facet.e06
// CHECK:STDOUT:   %U.loc122_4.1 => constants.%Destroy.facet.e06
// CHECK:STDOUT:   %U.as_type.loc123_36.1 => constants.%Int16
// CHECK:STDOUT:   %AddWith.type.loc123_36.1 => constants.%AddWith.type.344
// CHECK:STDOUT:   %facet_type.loc123_20 => constants.%facet_type.a81
// CHECK:STDOUT:   %.Self.1 => constants.%.Self.0ae
// CHECK:STDOUT:   %require_complete.loc123 => constants.%complete_type.f61
// CHECK:STDOUT:   %.Self.as_type.loc123_44.1 => constants.%.Self.as_type.12a0
// CHECK:STDOUT:   %AddWith.assoc_type => constants.%AddWith.assoc_type.f75
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.dc9
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.loc123 => constants.%AddWith.lookup_impl_witness.db4
// CHECK:STDOUT:   %impl.elem0.loc123_44.1 => constants.%impl.elem0.cdb
// CHECK:STDOUT:   %Result.as_type.loc123_54.1 => constants.%Int16
// CHECK:STDOUT:   %facet_type.loc123_38 => constants.%facet_type.e97
// CHECK:STDOUT:   %T.loc123_4.1 => constants.%facet_value.c79
// CHECK:STDOUT:   %pattern_type.loc123 => constants.%pattern_type.690
// CHECK:STDOUT:   %pattern_type.loc124_16 => constants.%pattern_type.9a4
// CHECK:STDOUT:   %T.as_type.loc124_29.1 => constants.%i16
// CHECK:STDOUT:   %pattern_type.loc124_27 => constants.%pattern_type.2f8
// CHECK:STDOUT:   %pattern_type.loc124_33 => constants.%pattern_type.9a4
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc124_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc124_27 => constants.%complete_type.2da
// CHECK:STDOUT:   %require_complete.loc124_33 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc126 => constants.%complete_type.f1c
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.646
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.loc126 => constants.%custom_witness.47f
// CHECK:STDOUT:   %AddWith.facet => constants.%AddWith.facet.ffe
// CHECK:STDOUT:   %AddWith.WithSelf.Op.type => constants.%AddWith.WithSelf.Op.type.894
// CHECK:STDOUT:   %.loc126_5.4 => constants.%.b10
// CHECK:STDOUT:   %impl.elem1.loc126_5.2 => constants.%Op.8fb557.1
// CHECK:STDOUT:   %specific_impl_fn.loc126_5.3 => constants.%Op.8fb557.1
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.833
// CHECK:STDOUT:   %.loc126_5.5 => constants.%.368
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.e5b
// CHECK:STDOUT:   %impl.elem0.loc126_5.2 => constants.%Int16.Op.c3e49c.1
// CHECK:STDOUT:   %specific_impl_fn.loc126_5.4 => constants.%Int16.Op.c3e49c.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @SubWith.loc135(constants.%Destroy.facet.e06, constants.%Destroy.facet.e06, constants.%facet_value.501) {
// CHECK:STDOUT:   %Result.loc132_9.1 => constants.%Destroy.facet.e06
// CHECK:STDOUT:   %U.loc133_4.1 => constants.%Destroy.facet.e06
// CHECK:STDOUT:   %U.as_type.loc134_36.1 => constants.%Int16
// CHECK:STDOUT:   %SubWith.type.loc134_36.1 => constants.%SubWith.type.306
// CHECK:STDOUT:   %facet_type.loc134_20 => constants.%facet_type.b37
// CHECK:STDOUT:   %.Self.1 => constants.%.Self.107
// CHECK:STDOUT:   %require_complete.loc134 => constants.%complete_type.5cf
// CHECK:STDOUT:   %.Self.as_type.loc134_44.1 => constants.%.Self.as_type.e34
// CHECK:STDOUT:   %SubWith.assoc_type => constants.%SubWith.assoc_type.7a2
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.f42
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.loc134 => constants.%SubWith.lookup_impl_witness.a35
// CHECK:STDOUT:   %impl.elem0.loc134_44.1 => constants.%impl.elem0.345
// CHECK:STDOUT:   %Result.as_type.loc134_54.1 => constants.%Int16
// CHECK:STDOUT:   %facet_type.loc134_38 => constants.%facet_type.281
// CHECK:STDOUT:   %T.loc134_4.1 => constants.%facet_value.501
// CHECK:STDOUT:   %pattern_type.loc134 => constants.%pattern_type.926
// CHECK:STDOUT:   %pattern_type.loc135_16 => constants.%pattern_type.9a4
// CHECK:STDOUT:   %T.as_type.loc135_29.1 => constants.%i16
// CHECK:STDOUT:   %pattern_type.loc135_27 => constants.%pattern_type.2f8
// CHECK:STDOUT:   %pattern_type.loc135_33 => constants.%pattern_type.9a4
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc135_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc135_27 => constants.%complete_type.2da
// CHECK:STDOUT:   %require_complete.loc135_33 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc137 => constants.%complete_type.9b3
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.2c3
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.loc137 => constants.%custom_witness.a7b
// CHECK:STDOUT:   %SubWith.facet => constants.%SubWith.facet.dd9
// CHECK:STDOUT:   %SubWith.WithSelf.Op.type => constants.%SubWith.WithSelf.Op.type.44d
// CHECK:STDOUT:   %.loc137_5.4 => constants.%.acf
// CHECK:STDOUT:   %impl.elem1.loc137_5.2 => constants.%Op.8fb557.2
// CHECK:STDOUT:   %specific_impl_fn.loc137_5.3 => constants.%Op.8fb557.2
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.833
// CHECK:STDOUT:   %.loc137_5.5 => constants.%.368
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.e5b
// CHECK:STDOUT:   %impl.elem0.loc137_5.2 => constants.%Int16.Op.c3e49c.1
// CHECK:STDOUT:   %specific_impl_fn.loc137_5.4 => constants.%Int16.Op.c3e49c.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @MulWith.loc146(constants.%Destroy.facet.e06, constants.%Destroy.facet.e06, constants.%facet_value.86a) {
// CHECK:STDOUT:   %Result.loc143_9.1 => constants.%Destroy.facet.e06
// CHECK:STDOUT:   %U.loc144_4.1 => constants.%Destroy.facet.e06
// CHECK:STDOUT:   %U.as_type.loc145_36.1 => constants.%Int16
// CHECK:STDOUT:   %MulWith.type.loc145_36.1 => constants.%MulWith.type.0cd
// CHECK:STDOUT:   %facet_type.loc145_20 => constants.%facet_type.2dd
// CHECK:STDOUT:   %.Self.1 => constants.%.Self.594
// CHECK:STDOUT:   %require_complete.loc145 => constants.%complete_type.b32
// CHECK:STDOUT:   %.Self.as_type.loc145_44.1 => constants.%.Self.as_type.674
// CHECK:STDOUT:   %MulWith.assoc_type => constants.%MulWith.assoc_type.f1b
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.3ee
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.loc145 => constants.%MulWith.lookup_impl_witness.834
// CHECK:STDOUT:   %impl.elem0.loc145_44.1 => constants.%impl.elem0.6fb
// CHECK:STDOUT:   %Result.as_type.loc145_54.1 => constants.%Int16
// CHECK:STDOUT:   %facet_type.loc145_38 => constants.%facet_type.b7c
// CHECK:STDOUT:   %T.loc145_4.1 => constants.%facet_value.86a
// CHECK:STDOUT:   %pattern_type.loc145 => constants.%pattern_type.401
// CHECK:STDOUT:   %pattern_type.loc146_16 => constants.%pattern_type.9a4
// CHECK:STDOUT:   %T.as_type.loc146_29.1 => constants.%i16
// CHECK:STDOUT:   %pattern_type.loc146_27 => constants.%pattern_type.2f8
// CHECK:STDOUT:   %pattern_type.loc146_33 => constants.%pattern_type.9a4
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc146_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc146_27 => constants.%complete_type.2da
// CHECK:STDOUT:   %require_complete.loc146_33 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc148 => constants.%complete_type.4d2
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.988
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.loc148 => constants.%custom_witness.e40
// CHECK:STDOUT:   %MulWith.facet => constants.%MulWith.facet.2a7
// CHECK:STDOUT:   %MulWith.WithSelf.Op.type => constants.%MulWith.WithSelf.Op.type.504
// CHECK:STDOUT:   %.loc148_5.4 => constants.%.20d
// CHECK:STDOUT:   %impl.elem1.loc148_5.2 => constants.%Op.8fb557.3
// CHECK:STDOUT:   %specific_impl_fn.loc148_5.3 => constants.%Op.8fb557.3
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.833
// CHECK:STDOUT:   %.loc148_5.5 => constants.%.368
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.e5b
// CHECK:STDOUT:   %impl.elem0.loc148_5.2 => constants.%Int16.Op.c3e49c.1
// CHECK:STDOUT:   %specific_impl_fn.loc148_5.4 => constants.%Int16.Op.c3e49c.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @DivWith.loc157(constants.%Destroy.facet.e06, constants.%Destroy.facet.e06, constants.%facet_value.018) {
// CHECK:STDOUT:   %Result.loc154_9.1 => constants.%Destroy.facet.e06
// CHECK:STDOUT:   %U.loc155_4.1 => constants.%Destroy.facet.e06
// CHECK:STDOUT:   %U.as_type.loc156_36.1 => constants.%Int16
// CHECK:STDOUT:   %DivWith.type.loc156_36.1 => constants.%DivWith.type.4c2
// CHECK:STDOUT:   %facet_type.loc156_20 => constants.%facet_type.e82
// CHECK:STDOUT:   %.Self.1 => constants.%.Self.c70
// CHECK:STDOUT:   %require_complete.loc156 => constants.%complete_type.8b32
// CHECK:STDOUT:   %.Self.as_type.loc156_44.1 => constants.%.Self.as_type.e0e
// CHECK:STDOUT:   %DivWith.assoc_type => constants.%DivWith.assoc_type.9fa
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.cf5
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.loc156 => constants.%DivWith.lookup_impl_witness.dbb
// CHECK:STDOUT:   %impl.elem0.loc156_44.1 => constants.%impl.elem0.e43
// CHECK:STDOUT:   %Result.as_type.loc156_54.1 => constants.%Int16
// CHECK:STDOUT:   %facet_type.loc156_38 => constants.%facet_type.a08
// CHECK:STDOUT:   %T.loc156_4.1 => constants.%facet_value.018
// CHECK:STDOUT:   %pattern_type.loc156 => constants.%pattern_type.6c0
// CHECK:STDOUT:   %pattern_type.loc157_16 => constants.%pattern_type.9a4
// CHECK:STDOUT:   %T.as_type.loc157_29.1 => constants.%i16
// CHECK:STDOUT:   %pattern_type.loc157_27 => constants.%pattern_type.2f8
// CHECK:STDOUT:   %pattern_type.loc157_33 => constants.%pattern_type.9a4
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc157_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc157_27 => constants.%complete_type.2da
// CHECK:STDOUT:   %require_complete.loc157_33 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc159 => constants.%complete_type.e42
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.cda
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.loc159 => constants.%custom_witness.fb3
// CHECK:STDOUT:   %DivWith.facet => constants.%DivWith.facet.c60
// CHECK:STDOUT:   %DivWith.WithSelf.Op.type => constants.%DivWith.WithSelf.Op.type.7eb
// CHECK:STDOUT:   %.loc159_5.4 => constants.%.411
// CHECK:STDOUT:   %impl.elem1.loc159_5.2 => constants.%Op.8fb557.4
// CHECK:STDOUT:   %specific_impl_fn.loc159_5.3 => constants.%Op.8fb557.4
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.833
// CHECK:STDOUT:   %.loc159_5.5 => constants.%.368
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.e5b
// CHECK:STDOUT:   %impl.elem0.loc159_5.2 => constants.%Int16.Op.c3e49c.1
// CHECK:STDOUT:   %specific_impl_fn.loc159_5.4 => constants.%Int16.Op.c3e49c.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ModWith.loc168(constants.%Destroy.facet.e06, constants.%Destroy.facet.e06, constants.%facet_value.ee6) {
// CHECK:STDOUT:   %Result.loc165_9.1 => constants.%Destroy.facet.e06
// CHECK:STDOUT:   %U.loc166_4.1 => constants.%Destroy.facet.e06
// CHECK:STDOUT:   %U.as_type.loc167_36.1 => constants.%Int16
// CHECK:STDOUT:   %ModWith.type.loc167_36.1 => constants.%ModWith.type.009
// CHECK:STDOUT:   %facet_type.loc167_20 => constants.%facet_type.e1e
// CHECK:STDOUT:   %.Self.1 => constants.%.Self.76e
// CHECK:STDOUT:   %require_complete.loc167 => constants.%complete_type.5b6
// CHECK:STDOUT:   %.Self.as_type.loc167_44.1 => constants.%.Self.as_type.242
// CHECK:STDOUT:   %ModWith.assoc_type => constants.%ModWith.assoc_type.355
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.0f7
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.loc167 => constants.%ModWith.lookup_impl_witness.c6b
// CHECK:STDOUT:   %impl.elem0.loc167_44.1 => constants.%impl.elem0.cba
// CHECK:STDOUT:   %Result.as_type.loc167_54.1 => constants.%Int16
// CHECK:STDOUT:   %facet_type.loc167_38 => constants.%facet_type.469
// CHECK:STDOUT:   %T.loc167_4.1 => constants.%facet_value.ee6
// CHECK:STDOUT:   %pattern_type.loc167 => constants.%pattern_type.13c
// CHECK:STDOUT:   %pattern_type.loc168_16 => constants.%pattern_type.9a4
// CHECK:STDOUT:   %T.as_type.loc168_29.1 => constants.%i16
// CHECK:STDOUT:   %pattern_type.loc168_27 => constants.%pattern_type.2f8
// CHECK:STDOUT:   %pattern_type.loc168_33 => constants.%pattern_type.9a4
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc168_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc168_27 => constants.%complete_type.2da
// CHECK:STDOUT:   %require_complete.loc168_33 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc170 => constants.%complete_type.e9f
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.c82
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.loc170 => constants.%custom_witness.69e
// CHECK:STDOUT:   %ModWith.facet => constants.%ModWith.facet.0e3
// CHECK:STDOUT:   %ModWith.WithSelf.Op.type => constants.%ModWith.WithSelf.Op.type.a02
// CHECK:STDOUT:   %.loc170_5.4 => constants.%.a0e
// CHECK:STDOUT:   %impl.elem1.loc170_5.2 => constants.%Op.8fb557.5
// CHECK:STDOUT:   %specific_impl_fn.loc170_5.3 => constants.%Op.8fb557.5
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.833
// CHECK:STDOUT:   %.loc170_5.5 => constants.%.368
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.e5b
// CHECK:STDOUT:   %impl.elem0.loc170_5.2 => constants.%Int16.Op.c3e49c.1
// CHECK:STDOUT:   %specific_impl_fn.loc170_5.4 => constants.%Int16.Op.c3e49c.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AddWith.loc124(constants.%Destroy.facet.e06, constants.%Destroy.facet.77e, constants.%facet_value.dd6) {
// CHECK:STDOUT:   %Result.loc121_9.1 => constants.%Destroy.facet.e06
// CHECK:STDOUT:   %U.loc122_4.1 => constants.%Destroy.facet.77e
// CHECK:STDOUT:   %U.as_type.loc123_36.1 => constants.%i16
// CHECK:STDOUT:   %AddWith.type.loc123_36.1 => constants.%AddWith.type.a18
// CHECK:STDOUT:   %facet_type.loc123_20 => constants.%facet_type.8b1
// CHECK:STDOUT:   %.Self.1 => constants.%.Self.ba7
// CHECK:STDOUT:   %require_complete.loc123 => constants.%complete_type.59d
// CHECK:STDOUT:   %.Self.as_type.loc123_44.1 => constants.%.Self.as_type.f28
// CHECK:STDOUT:   %AddWith.assoc_type => constants.%AddWith.assoc_type.4f0
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.ba6
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.loc123 => constants.%AddWith.lookup_impl_witness.d8b
// CHECK:STDOUT:   %impl.elem0.loc123_44.1 => constants.%impl.elem0.265
// CHECK:STDOUT:   %Result.as_type.loc123_54.1 => constants.%Int16
// CHECK:STDOUT:   %facet_type.loc123_38 => constants.%facet_type.731
// CHECK:STDOUT:   %T.loc123_4.1 => constants.%facet_value.dd6
// CHECK:STDOUT:   %pattern_type.loc123 => constants.%pattern_type.336
// CHECK:STDOUT:   %pattern_type.loc124_16 => constants.%pattern_type.9a4
// CHECK:STDOUT:   %T.as_type.loc124_29.1 => constants.%Int16
// CHECK:STDOUT:   %pattern_type.loc124_27 => constants.%pattern_type.9a4
// CHECK:STDOUT:   %pattern_type.loc124_33 => constants.%pattern_type.2f8
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc124_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc124_27 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc124_33 => constants.%complete_type.2da
// CHECK:STDOUT:   %require_complete.loc126 => constants.%complete_type.e1b
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.88d
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.loc126 => constants.%custom_witness.096
// CHECK:STDOUT:   %AddWith.facet => constants.%AddWith.facet.c96
// CHECK:STDOUT:   %AddWith.WithSelf.Op.type => constants.%AddWith.WithSelf.Op.type.b73
// CHECK:STDOUT:   %.loc126_5.4 => constants.%.f80
// CHECK:STDOUT:   %impl.elem1.loc126_5.2 => constants.%Int16.Op.c3e49c.3
// CHECK:STDOUT:   %specific_impl_fn.loc126_5.3 => constants.%Int16.Op.c3e49c.3
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.833
// CHECK:STDOUT:   %.loc126_5.5 => constants.%.368
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.e5b
// CHECK:STDOUT:   %impl.elem0.loc126_5.2 => constants.%Int16.Op.c3e49c.1
// CHECK:STDOUT:   %specific_impl_fn.loc126_5.4 => constants.%Int16.Op.c3e49c.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @SubWith.loc135(constants.%Destroy.facet.e06, constants.%Destroy.facet.77e, constants.%facet_value.3f5) {
// CHECK:STDOUT:   %Result.loc132_9.1 => constants.%Destroy.facet.e06
// CHECK:STDOUT:   %U.loc133_4.1 => constants.%Destroy.facet.77e
// CHECK:STDOUT:   %U.as_type.loc134_36.1 => constants.%i16
// CHECK:STDOUT:   %SubWith.type.loc134_36.1 => constants.%SubWith.type.a1b
// CHECK:STDOUT:   %facet_type.loc134_20 => constants.%facet_type.bd5
// CHECK:STDOUT:   %.Self.1 => constants.%.Self.d04
// CHECK:STDOUT:   %require_complete.loc134 => constants.%complete_type.d4c
// CHECK:STDOUT:   %.Self.as_type.loc134_44.1 => constants.%.Self.as_type.d0c7
// CHECK:STDOUT:   %SubWith.assoc_type => constants.%SubWith.assoc_type.108
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.053
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.loc134 => constants.%SubWith.lookup_impl_witness.029
// CHECK:STDOUT:   %impl.elem0.loc134_44.1 => constants.%impl.elem0.e59
// CHECK:STDOUT:   %Result.as_type.loc134_54.1 => constants.%Int16
// CHECK:STDOUT:   %facet_type.loc134_38 => constants.%facet_type.257
// CHECK:STDOUT:   %T.loc134_4.1 => constants.%facet_value.3f5
// CHECK:STDOUT:   %pattern_type.loc134 => constants.%pattern_type.021
// CHECK:STDOUT:   %pattern_type.loc135_16 => constants.%pattern_type.9a4
// CHECK:STDOUT:   %T.as_type.loc135_29.1 => constants.%Int16
// CHECK:STDOUT:   %pattern_type.loc135_27 => constants.%pattern_type.9a4
// CHECK:STDOUT:   %pattern_type.loc135_33 => constants.%pattern_type.2f8
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc135_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc135_27 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc135_33 => constants.%complete_type.2da
// CHECK:STDOUT:   %require_complete.loc137 => constants.%complete_type.6de
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.ef4
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.loc137 => constants.%custom_witness.ced
// CHECK:STDOUT:   %SubWith.facet => constants.%SubWith.facet.3c2
// CHECK:STDOUT:   %SubWith.WithSelf.Op.type => constants.%SubWith.WithSelf.Op.type.9bf
// CHECK:STDOUT:   %.loc137_5.4 => constants.%.90d
// CHECK:STDOUT:   %impl.elem1.loc137_5.2 => constants.%Int16.Op.c3e49c.4
// CHECK:STDOUT:   %specific_impl_fn.loc137_5.3 => constants.%Int16.Op.c3e49c.4
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.833
// CHECK:STDOUT:   %.loc137_5.5 => constants.%.368
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.e5b
// CHECK:STDOUT:   %impl.elem0.loc137_5.2 => constants.%Int16.Op.c3e49c.1
// CHECK:STDOUT:   %specific_impl_fn.loc137_5.4 => constants.%Int16.Op.c3e49c.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @MulWith.loc146(constants.%Destroy.facet.e06, constants.%Destroy.facet.77e, constants.%facet_value.4ca) {
// CHECK:STDOUT:   %Result.loc143_9.1 => constants.%Destroy.facet.e06
// CHECK:STDOUT:   %U.loc144_4.1 => constants.%Destroy.facet.77e
// CHECK:STDOUT:   %U.as_type.loc145_36.1 => constants.%i16
// CHECK:STDOUT:   %MulWith.type.loc145_36.1 => constants.%MulWith.type.ee9
// CHECK:STDOUT:   %facet_type.loc145_20 => constants.%facet_type.5ca
// CHECK:STDOUT:   %.Self.1 => constants.%.Self.3cf
// CHECK:STDOUT:   %require_complete.loc145 => constants.%complete_type.d8f
// CHECK:STDOUT:   %.Self.as_type.loc145_44.1 => constants.%.Self.as_type.4dc
// CHECK:STDOUT:   %MulWith.assoc_type => constants.%MulWith.assoc_type.aae
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.4d95
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.loc145 => constants.%MulWith.lookup_impl_witness.aee
// CHECK:STDOUT:   %impl.elem0.loc145_44.1 => constants.%impl.elem0.5d1
// CHECK:STDOUT:   %Result.as_type.loc145_54.1 => constants.%Int16
// CHECK:STDOUT:   %facet_type.loc145_38 => constants.%facet_type.865
// CHECK:STDOUT:   %T.loc145_4.1 => constants.%facet_value.4ca
// CHECK:STDOUT:   %pattern_type.loc145 => constants.%pattern_type.038
// CHECK:STDOUT:   %pattern_type.loc146_16 => constants.%pattern_type.9a4
// CHECK:STDOUT:   %T.as_type.loc146_29.1 => constants.%Int16
// CHECK:STDOUT:   %pattern_type.loc146_27 => constants.%pattern_type.9a4
// CHECK:STDOUT:   %pattern_type.loc146_33 => constants.%pattern_type.2f8
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc146_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc146_27 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc146_33 => constants.%complete_type.2da
// CHECK:STDOUT:   %require_complete.loc148 => constants.%complete_type.8b34
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.587
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.loc148 => constants.%custom_witness.270
// CHECK:STDOUT:   %MulWith.facet => constants.%MulWith.facet.0a1
// CHECK:STDOUT:   %MulWith.WithSelf.Op.type => constants.%MulWith.WithSelf.Op.type.a45
// CHECK:STDOUT:   %.loc148_5.4 => constants.%.70d
// CHECK:STDOUT:   %impl.elem1.loc148_5.2 => constants.%Int16.Op.c3e49c.5
// CHECK:STDOUT:   %specific_impl_fn.loc148_5.3 => constants.%Int16.Op.c3e49c.5
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.833
// CHECK:STDOUT:   %.loc148_5.5 => constants.%.368
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.e5b
// CHECK:STDOUT:   %impl.elem0.loc148_5.2 => constants.%Int16.Op.c3e49c.1
// CHECK:STDOUT:   %specific_impl_fn.loc148_5.4 => constants.%Int16.Op.c3e49c.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @DivWith.loc157(constants.%Destroy.facet.e06, constants.%Destroy.facet.77e, constants.%facet_value.ff9) {
// CHECK:STDOUT:   %Result.loc154_9.1 => constants.%Destroy.facet.e06
// CHECK:STDOUT:   %U.loc155_4.1 => constants.%Destroy.facet.77e
// CHECK:STDOUT:   %U.as_type.loc156_36.1 => constants.%i16
// CHECK:STDOUT:   %DivWith.type.loc156_36.1 => constants.%DivWith.type.9ae
// CHECK:STDOUT:   %facet_type.loc156_20 => constants.%facet_type.014
// CHECK:STDOUT:   %.Self.1 => constants.%.Self.197
// CHECK:STDOUT:   %require_complete.loc156 => constants.%complete_type.b92
// CHECK:STDOUT:   %.Self.as_type.loc156_44.1 => constants.%.Self.as_type.2e1
// CHECK:STDOUT:   %DivWith.assoc_type => constants.%DivWith.assoc_type.e80
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.75e
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.loc156 => constants.%DivWith.lookup_impl_witness.834
// CHECK:STDOUT:   %impl.elem0.loc156_44.1 => constants.%impl.elem0.146
// CHECK:STDOUT:   %Result.as_type.loc156_54.1 => constants.%Int16
// CHECK:STDOUT:   %facet_type.loc156_38 => constants.%facet_type.c20
// CHECK:STDOUT:   %T.loc156_4.1 => constants.%facet_value.ff9
// CHECK:STDOUT:   %pattern_type.loc156 => constants.%pattern_type.8c5
// CHECK:STDOUT:   %pattern_type.loc157_16 => constants.%pattern_type.9a4
// CHECK:STDOUT:   %T.as_type.loc157_29.1 => constants.%Int16
// CHECK:STDOUT:   %pattern_type.loc157_27 => constants.%pattern_type.9a4
// CHECK:STDOUT:   %pattern_type.loc157_33 => constants.%pattern_type.2f8
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc157_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc157_27 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc157_33 => constants.%complete_type.2da
// CHECK:STDOUT:   %require_complete.loc159 => constants.%complete_type.fc5
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.9ae
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.loc159 => constants.%custom_witness.462
// CHECK:STDOUT:   %DivWith.facet => constants.%DivWith.facet.915
// CHECK:STDOUT:   %DivWith.WithSelf.Op.type => constants.%DivWith.WithSelf.Op.type.e9d
// CHECK:STDOUT:   %.loc159_5.4 => constants.%.d3f
// CHECK:STDOUT:   %impl.elem1.loc159_5.2 => constants.%Int16.Op.c3e49c.6
// CHECK:STDOUT:   %specific_impl_fn.loc159_5.3 => constants.%Int16.Op.c3e49c.6
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.833
// CHECK:STDOUT:   %.loc159_5.5 => constants.%.368
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.e5b
// CHECK:STDOUT:   %impl.elem0.loc159_5.2 => constants.%Int16.Op.c3e49c.1
// CHECK:STDOUT:   %specific_impl_fn.loc159_5.4 => constants.%Int16.Op.c3e49c.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ModWith.loc168(constants.%Destroy.facet.e06, constants.%Destroy.facet.77e, constants.%facet_value.f56) {
// CHECK:STDOUT:   %Result.loc165_9.1 => constants.%Destroy.facet.e06
// CHECK:STDOUT:   %U.loc166_4.1 => constants.%Destroy.facet.77e
// CHECK:STDOUT:   %U.as_type.loc167_36.1 => constants.%i16
// CHECK:STDOUT:   %ModWith.type.loc167_36.1 => constants.%ModWith.type.406
// CHECK:STDOUT:   %facet_type.loc167_20 => constants.%facet_type.fe7
// CHECK:STDOUT:   %.Self.1 => constants.%.Self.7d0
// CHECK:STDOUT:   %require_complete.loc167 => constants.%complete_type.dfc
// CHECK:STDOUT:   %.Self.as_type.loc167_44.1 => constants.%.Self.as_type.81b
// CHECK:STDOUT:   %ModWith.assoc_type => constants.%ModWith.assoc_type.ffe
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.b49
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.loc167 => constants.%ModWith.lookup_impl_witness.ff6
// CHECK:STDOUT:   %impl.elem0.loc167_44.1 => constants.%impl.elem0.f78
// CHECK:STDOUT:   %Result.as_type.loc167_54.1 => constants.%Int16
// CHECK:STDOUT:   %facet_type.loc167_38 => constants.%facet_type.a93
// CHECK:STDOUT:   %T.loc167_4.1 => constants.%facet_value.f56
// CHECK:STDOUT:   %pattern_type.loc167 => constants.%pattern_type.0e2
// CHECK:STDOUT:   %pattern_type.loc168_16 => constants.%pattern_type.9a4
// CHECK:STDOUT:   %T.as_type.loc168_29.1 => constants.%Int16
// CHECK:STDOUT:   %pattern_type.loc168_27 => constants.%pattern_type.9a4
// CHECK:STDOUT:   %pattern_type.loc168_33 => constants.%pattern_type.2f8
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc168_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc168_27 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc168_33 => constants.%complete_type.2da
// CHECK:STDOUT:   %require_complete.loc170 => constants.%complete_type.78b
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.d40
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.loc170 => constants.%custom_witness.84d2
// CHECK:STDOUT:   %ModWith.facet => constants.%ModWith.facet.6ed
// CHECK:STDOUT:   %ModWith.WithSelf.Op.type => constants.%ModWith.WithSelf.Op.type.a5a
// CHECK:STDOUT:   %.loc170_5.4 => constants.%.a4e
// CHECK:STDOUT:   %impl.elem1.loc170_5.2 => constants.%Int16.Op.c3e49c.7
// CHECK:STDOUT:   %specific_impl_fn.loc170_5.3 => constants.%Int16.Op.c3e49c.7
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.833
// CHECK:STDOUT:   %.loc170_5.5 => constants.%.368
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.e5b
// CHECK:STDOUT:   %impl.elem0.loc170_5.2 => constants.%Int16.Op.c3e49c.1
// CHECK:STDOUT:   %specific_impl_fn.loc170_5.4 => constants.%Int16.Op.c3e49c.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AddWith.loc124(constants.%Destroy.facet.c06, constants.%Destroy.facet.c06, constants.%facet_value.2b7) {
// CHECK:STDOUT:   %Result.loc121_9.1 => constants.%Destroy.facet.c06
// CHECK:STDOUT:   %U.loc122_4.1 => constants.%Destroy.facet.c06
// CHECK:STDOUT:   %U.as_type.loc123_36.1 => constants.%Int32
// CHECK:STDOUT:   %AddWith.type.loc123_36.1 => constants.%AddWith.type.e9e
// CHECK:STDOUT:   %facet_type.loc123_20 => constants.%facet_type.4a7
// CHECK:STDOUT:   %.Self.1 => constants.%.Self.3d9
// CHECK:STDOUT:   %require_complete.loc123 => constants.%complete_type.0d9
// CHECK:STDOUT:   %.Self.as_type.loc123_44.1 => constants.%.Self.as_type.7ba
// CHECK:STDOUT:   %AddWith.assoc_type => constants.%AddWith.assoc_type.29a
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.701
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.loc123 => constants.%AddWith.lookup_impl_witness.632
// CHECK:STDOUT:   %impl.elem0.loc123_44.1 => constants.%impl.elem0.cee
// CHECK:STDOUT:   %Result.as_type.loc123_54.1 => constants.%Int32
// CHECK:STDOUT:   %facet_type.loc123_38 => constants.%facet_type.14d
// CHECK:STDOUT:   %T.loc123_4.1 => constants.%facet_value.2b7
// CHECK:STDOUT:   %pattern_type.loc123 => constants.%pattern_type.3f9
// CHECK:STDOUT:   %pattern_type.loc124_16 => constants.%pattern_type.9c9
// CHECK:STDOUT:   %T.as_type.loc124_29.1 => constants.%Int16
// CHECK:STDOUT:   %pattern_type.loc124_27 => constants.%pattern_type.9a4
// CHECK:STDOUT:   %pattern_type.loc124_33 => constants.%pattern_type.9c9
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc124_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc124_27 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc124_33 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc126 => constants.%complete_type.03d
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.38b
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.loc126 => constants.%custom_witness.200
// CHECK:STDOUT:   %AddWith.facet => constants.%AddWith.facet.abd
// CHECK:STDOUT:   %AddWith.WithSelf.Op.type => constants.%AddWith.WithSelf.Op.type.708
// CHECK:STDOUT:   %.loc126_5.4 => constants.%.0ff
// CHECK:STDOUT:   %impl.elem1.loc126_5.2 => constants.%Int16.Op.c3e49c.8
// CHECK:STDOUT:   %specific_impl_fn.loc126_5.3 => constants.%Int16.Op.c3e49c.8
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.2e1
// CHECK:STDOUT:   %.loc126_5.5 => constants.%.e6c
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.ce1
// CHECK:STDOUT:   %impl.elem0.loc126_5.2 => constants.%Int32.Op.6afff1.1
// CHECK:STDOUT:   %specific_impl_fn.loc126_5.4 => constants.%Int32.Op.6afff1.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @SubWith.loc135(constants.%Destroy.facet.c06, constants.%Destroy.facet.c06, constants.%facet_value.7d7) {
// CHECK:STDOUT:   %Result.loc132_9.1 => constants.%Destroy.facet.c06
// CHECK:STDOUT:   %U.loc133_4.1 => constants.%Destroy.facet.c06
// CHECK:STDOUT:   %U.as_type.loc134_36.1 => constants.%Int32
// CHECK:STDOUT:   %SubWith.type.loc134_36.1 => constants.%SubWith.type.84c
// CHECK:STDOUT:   %facet_type.loc134_20 => constants.%facet_type.e7a
// CHECK:STDOUT:   %.Self.1 => constants.%.Self.384
// CHECK:STDOUT:   %require_complete.loc134 => constants.%complete_type.a34
// CHECK:STDOUT:   %.Self.as_type.loc134_44.1 => constants.%.Self.as_type.45d
// CHECK:STDOUT:   %SubWith.assoc_type => constants.%SubWith.assoc_type.0f9
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.1e9
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.loc134 => constants.%SubWith.lookup_impl_witness.b61
// CHECK:STDOUT:   %impl.elem0.loc134_44.1 => constants.%impl.elem0.edd
// CHECK:STDOUT:   %Result.as_type.loc134_54.1 => constants.%Int32
// CHECK:STDOUT:   %facet_type.loc134_38 => constants.%facet_type.f15
// CHECK:STDOUT:   %T.loc134_4.1 => constants.%facet_value.7d7
// CHECK:STDOUT:   %pattern_type.loc134 => constants.%pattern_type.56e
// CHECK:STDOUT:   %pattern_type.loc135_16 => constants.%pattern_type.9c9
// CHECK:STDOUT:   %T.as_type.loc135_29.1 => constants.%Int16
// CHECK:STDOUT:   %pattern_type.loc135_27 => constants.%pattern_type.9a4
// CHECK:STDOUT:   %pattern_type.loc135_33 => constants.%pattern_type.9c9
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc135_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc135_27 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc135_33 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc137 => constants.%complete_type.9bf
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.714
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.loc137 => constants.%custom_witness.103
// CHECK:STDOUT:   %SubWith.facet => constants.%SubWith.facet.696
// CHECK:STDOUT:   %SubWith.WithSelf.Op.type => constants.%SubWith.WithSelf.Op.type.e30
// CHECK:STDOUT:   %.loc137_5.4 => constants.%.a54
// CHECK:STDOUT:   %impl.elem1.loc137_5.2 => constants.%Int16.Op.c3e49c.9
// CHECK:STDOUT:   %specific_impl_fn.loc137_5.3 => constants.%Int16.Op.c3e49c.9
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.2e1
// CHECK:STDOUT:   %.loc137_5.5 => constants.%.e6c
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.ce1
// CHECK:STDOUT:   %impl.elem0.loc137_5.2 => constants.%Int32.Op.6afff1.1
// CHECK:STDOUT:   %specific_impl_fn.loc137_5.4 => constants.%Int32.Op.6afff1.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @MulWith.loc146(constants.%Destroy.facet.c06, constants.%Destroy.facet.c06, constants.%facet_value.75c) {
// CHECK:STDOUT:   %Result.loc143_9.1 => constants.%Destroy.facet.c06
// CHECK:STDOUT:   %U.loc144_4.1 => constants.%Destroy.facet.c06
// CHECK:STDOUT:   %U.as_type.loc145_36.1 => constants.%Int32
// CHECK:STDOUT:   %MulWith.type.loc145_36.1 => constants.%MulWith.type.d1b
// CHECK:STDOUT:   %facet_type.loc145_20 => constants.%facet_type.2e6
// CHECK:STDOUT:   %.Self.1 => constants.%.Self.84b
// CHECK:STDOUT:   %require_complete.loc145 => constants.%complete_type.043
// CHECK:STDOUT:   %.Self.as_type.loc145_44.1 => constants.%.Self.as_type.d0cd
// CHECK:STDOUT:   %MulWith.assoc_type => constants.%MulWith.assoc_type.8c6
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.a8f
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.loc145 => constants.%MulWith.lookup_impl_witness.539
// CHECK:STDOUT:   %impl.elem0.loc145_44.1 => constants.%impl.elem0.e63
// CHECK:STDOUT:   %Result.as_type.loc145_54.1 => constants.%Int32
// CHECK:STDOUT:   %facet_type.loc145_38 => constants.%facet_type.c22
// CHECK:STDOUT:   %T.loc145_4.1 => constants.%facet_value.75c
// CHECK:STDOUT:   %pattern_type.loc145 => constants.%pattern_type.965
// CHECK:STDOUT:   %pattern_type.loc146_16 => constants.%pattern_type.9c9
// CHECK:STDOUT:   %T.as_type.loc146_29.1 => constants.%Int16
// CHECK:STDOUT:   %pattern_type.loc146_27 => constants.%pattern_type.9a4
// CHECK:STDOUT:   %pattern_type.loc146_33 => constants.%pattern_type.9c9
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc146_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc146_27 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc146_33 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc148 => constants.%complete_type.a1f
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.6bc
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.loc148 => constants.%custom_witness.a6d
// CHECK:STDOUT:   %MulWith.facet => constants.%MulWith.facet.6d1
// CHECK:STDOUT:   %MulWith.WithSelf.Op.type => constants.%MulWith.WithSelf.Op.type.61d
// CHECK:STDOUT:   %.loc148_5.4 => constants.%.3a1
// CHECK:STDOUT:   %impl.elem1.loc148_5.2 => constants.%Int16.Op.c3e49c.10
// CHECK:STDOUT:   %specific_impl_fn.loc148_5.3 => constants.%Int16.Op.c3e49c.10
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.2e1
// CHECK:STDOUT:   %.loc148_5.5 => constants.%.e6c
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.ce1
// CHECK:STDOUT:   %impl.elem0.loc148_5.2 => constants.%Int32.Op.6afff1.1
// CHECK:STDOUT:   %specific_impl_fn.loc148_5.4 => constants.%Int32.Op.6afff1.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @DivWith.loc157(constants.%Destroy.facet.c06, constants.%Destroy.facet.c06, constants.%facet_value.85a) {
// CHECK:STDOUT:   %Result.loc154_9.1 => constants.%Destroy.facet.c06
// CHECK:STDOUT:   %U.loc155_4.1 => constants.%Destroy.facet.c06
// CHECK:STDOUT:   %U.as_type.loc156_36.1 => constants.%Int32
// CHECK:STDOUT:   %DivWith.type.loc156_36.1 => constants.%DivWith.type.26e
// CHECK:STDOUT:   %facet_type.loc156_20 => constants.%facet_type.eb5
// CHECK:STDOUT:   %.Self.1 => constants.%.Self.203
// CHECK:STDOUT:   %require_complete.loc156 => constants.%complete_type.d76
// CHECK:STDOUT:   %.Self.as_type.loc156_44.1 => constants.%.Self.as_type.f8e
// CHECK:STDOUT:   %DivWith.assoc_type => constants.%DivWith.assoc_type.109
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.d76
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.loc156 => constants.%DivWith.lookup_impl_witness.512
// CHECK:STDOUT:   %impl.elem0.loc156_44.1 => constants.%impl.elem0.232
// CHECK:STDOUT:   %Result.as_type.loc156_54.1 => constants.%Int32
// CHECK:STDOUT:   %facet_type.loc156_38 => constants.%facet_type.529
// CHECK:STDOUT:   %T.loc156_4.1 => constants.%facet_value.85a
// CHECK:STDOUT:   %pattern_type.loc156 => constants.%pattern_type.04c
// CHECK:STDOUT:   %pattern_type.loc157_16 => constants.%pattern_type.9c9
// CHECK:STDOUT:   %T.as_type.loc157_29.1 => constants.%Int16
// CHECK:STDOUT:   %pattern_type.loc157_27 => constants.%pattern_type.9a4
// CHECK:STDOUT:   %pattern_type.loc157_33 => constants.%pattern_type.9c9
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc157_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc157_27 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc157_33 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc159 => constants.%complete_type.e49
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.4a0
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.loc159 => constants.%custom_witness.510
// CHECK:STDOUT:   %DivWith.facet => constants.%DivWith.facet.246
// CHECK:STDOUT:   %DivWith.WithSelf.Op.type => constants.%DivWith.WithSelf.Op.type.623
// CHECK:STDOUT:   %.loc159_5.4 => constants.%.e6b
// CHECK:STDOUT:   %impl.elem1.loc159_5.2 => constants.%Int16.Op.c3e49c.11
// CHECK:STDOUT:   %specific_impl_fn.loc159_5.3 => constants.%Int16.Op.c3e49c.11
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.2e1
// CHECK:STDOUT:   %.loc159_5.5 => constants.%.e6c
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.ce1
// CHECK:STDOUT:   %impl.elem0.loc159_5.2 => constants.%Int32.Op.6afff1.1
// CHECK:STDOUT:   %specific_impl_fn.loc159_5.4 => constants.%Int32.Op.6afff1.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ModWith.loc168(constants.%Destroy.facet.c06, constants.%Destroy.facet.c06, constants.%facet_value.dd3) {
// CHECK:STDOUT:   %Result.loc165_9.1 => constants.%Destroy.facet.c06
// CHECK:STDOUT:   %U.loc166_4.1 => constants.%Destroy.facet.c06
// CHECK:STDOUT:   %U.as_type.loc167_36.1 => constants.%Int32
// CHECK:STDOUT:   %ModWith.type.loc167_36.1 => constants.%ModWith.type.5c4
// CHECK:STDOUT:   %facet_type.loc167_20 => constants.%facet_type.1e6
// CHECK:STDOUT:   %.Self.1 => constants.%.Self.0c5
// CHECK:STDOUT:   %require_complete.loc167 => constants.%complete_type.1bb
// CHECK:STDOUT:   %.Self.as_type.loc167_44.1 => constants.%.Self.as_type.ef1
// CHECK:STDOUT:   %ModWith.assoc_type => constants.%ModWith.assoc_type.1db
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.8e2
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.loc167 => constants.%ModWith.lookup_impl_witness.9f7
// CHECK:STDOUT:   %impl.elem0.loc167_44.1 => constants.%impl.elem0.919
// CHECK:STDOUT:   %Result.as_type.loc167_54.1 => constants.%Int32
// CHECK:STDOUT:   %facet_type.loc167_38 => constants.%facet_type.422
// CHECK:STDOUT:   %T.loc167_4.1 => constants.%facet_value.dd3
// CHECK:STDOUT:   %pattern_type.loc167 => constants.%pattern_type.4d3
// CHECK:STDOUT:   %pattern_type.loc168_16 => constants.%pattern_type.9c9
// CHECK:STDOUT:   %T.as_type.loc168_29.1 => constants.%Int16
// CHECK:STDOUT:   %pattern_type.loc168_27 => constants.%pattern_type.9a4
// CHECK:STDOUT:   %pattern_type.loc168_33 => constants.%pattern_type.9c9
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc168_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc168_27 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc168_33 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc170 => constants.%complete_type.3ed
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.826
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.loc170 => constants.%custom_witness.a62
// CHECK:STDOUT:   %ModWith.facet => constants.%ModWith.facet.76b
// CHECK:STDOUT:   %ModWith.WithSelf.Op.type => constants.%ModWith.WithSelf.Op.type.cfc
// CHECK:STDOUT:   %.loc170_5.4 => constants.%.132
// CHECK:STDOUT:   %impl.elem1.loc170_5.2 => constants.%Int16.Op.c3e49c.12
// CHECK:STDOUT:   %specific_impl_fn.loc170_5.3 => constants.%Int16.Op.c3e49c.12
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.2e1
// CHECK:STDOUT:   %.loc170_5.5 => constants.%.e6c
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.ce1
// CHECK:STDOUT:   %impl.elem0.loc170_5.2 => constants.%Int32.Op.6afff1.1
// CHECK:STDOUT:   %specific_impl_fn.loc170_5.4 => constants.%Int32.Op.6afff1.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AddWith.loc124(constants.%Destroy.facet.c06, constants.%Destroy.facet.e06, constants.%facet_value.741) {
// CHECK:STDOUT:   %Result.loc121_9.1 => constants.%Destroy.facet.c06
// CHECK:STDOUT:   %U.loc122_4.1 => constants.%Destroy.facet.e06
// CHECK:STDOUT:   %U.as_type.loc123_36.1 => constants.%Int16
// CHECK:STDOUT:   %AddWith.type.loc123_36.1 => constants.%AddWith.type.344
// CHECK:STDOUT:   %facet_type.loc123_20 => constants.%facet_type.a81
// CHECK:STDOUT:   %.Self.1 => constants.%.Self.0ae
// CHECK:STDOUT:   %require_complete.loc123 => constants.%complete_type.f61
// CHECK:STDOUT:   %.Self.as_type.loc123_44.1 => constants.%.Self.as_type.12a0
// CHECK:STDOUT:   %AddWith.assoc_type => constants.%AddWith.assoc_type.f75
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.dc9
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.loc123 => constants.%AddWith.lookup_impl_witness.db4
// CHECK:STDOUT:   %impl.elem0.loc123_44.1 => constants.%impl.elem0.cdb
// CHECK:STDOUT:   %Result.as_type.loc123_54.1 => constants.%Int32
// CHECK:STDOUT:   %facet_type.loc123_38 => constants.%facet_type.b0a
// CHECK:STDOUT:   %T.loc123_4.1 => constants.%facet_value.741
// CHECK:STDOUT:   %pattern_type.loc123 => constants.%pattern_type.e21
// CHECK:STDOUT:   %pattern_type.loc124_16 => constants.%pattern_type.9c9
// CHECK:STDOUT:   %T.as_type.loc124_29.1 => constants.%Int32
// CHECK:STDOUT:   %pattern_type.loc124_27 => constants.%pattern_type.9c9
// CHECK:STDOUT:   %pattern_type.loc124_33 => constants.%pattern_type.9a4
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc124_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc124_27 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc124_33 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc126 => constants.%complete_type.f1c
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.646
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.loc126 => constants.%custom_witness.116
// CHECK:STDOUT:   %AddWith.facet => constants.%AddWith.facet.fce
// CHECK:STDOUT:   %AddWith.WithSelf.Op.type => constants.%AddWith.WithSelf.Op.type.523
// CHECK:STDOUT:   %.loc126_5.4 => constants.%.a28
// CHECK:STDOUT:   %impl.elem1.loc126_5.2 => constants.%Int32.Op.6afff1.2
// CHECK:STDOUT:   %specific_impl_fn.loc126_5.3 => constants.%Int32.Op.6afff1.2
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.2e1
// CHECK:STDOUT:   %.loc126_5.5 => constants.%.e6c
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.ce1
// CHECK:STDOUT:   %impl.elem0.loc126_5.2 => constants.%Int32.Op.6afff1.1
// CHECK:STDOUT:   %specific_impl_fn.loc126_5.4 => constants.%Int32.Op.6afff1.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @SubWith.loc135(constants.%Destroy.facet.c06, constants.%Destroy.facet.e06, constants.%facet_value.f31) {
// CHECK:STDOUT:   %Result.loc132_9.1 => constants.%Destroy.facet.c06
// CHECK:STDOUT:   %U.loc133_4.1 => constants.%Destroy.facet.e06
// CHECK:STDOUT:   %U.as_type.loc134_36.1 => constants.%Int16
// CHECK:STDOUT:   %SubWith.type.loc134_36.1 => constants.%SubWith.type.306
// CHECK:STDOUT:   %facet_type.loc134_20 => constants.%facet_type.b37
// CHECK:STDOUT:   %.Self.1 => constants.%.Self.107
// CHECK:STDOUT:   %require_complete.loc134 => constants.%complete_type.5cf
// CHECK:STDOUT:   %.Self.as_type.loc134_44.1 => constants.%.Self.as_type.e34
// CHECK:STDOUT:   %SubWith.assoc_type => constants.%SubWith.assoc_type.7a2
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.f42
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.loc134 => constants.%SubWith.lookup_impl_witness.a35
// CHECK:STDOUT:   %impl.elem0.loc134_44.1 => constants.%impl.elem0.345
// CHECK:STDOUT:   %Result.as_type.loc134_54.1 => constants.%Int32
// CHECK:STDOUT:   %facet_type.loc134_38 => constants.%facet_type.afb
// CHECK:STDOUT:   %T.loc134_4.1 => constants.%facet_value.f31
// CHECK:STDOUT:   %pattern_type.loc134 => constants.%pattern_type.6ad
// CHECK:STDOUT:   %pattern_type.loc135_16 => constants.%pattern_type.9c9
// CHECK:STDOUT:   %T.as_type.loc135_29.1 => constants.%Int32
// CHECK:STDOUT:   %pattern_type.loc135_27 => constants.%pattern_type.9c9
// CHECK:STDOUT:   %pattern_type.loc135_33 => constants.%pattern_type.9a4
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc135_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc135_27 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc135_33 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc137 => constants.%complete_type.9b3
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.2c3
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.loc137 => constants.%custom_witness.904
// CHECK:STDOUT:   %SubWith.facet => constants.%SubWith.facet.099
// CHECK:STDOUT:   %SubWith.WithSelf.Op.type => constants.%SubWith.WithSelf.Op.type.5d6
// CHECK:STDOUT:   %.loc137_5.4 => constants.%.43e
// CHECK:STDOUT:   %impl.elem1.loc137_5.2 => constants.%Int32.Op.6afff1.3
// CHECK:STDOUT:   %specific_impl_fn.loc137_5.3 => constants.%Int32.Op.6afff1.3
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.2e1
// CHECK:STDOUT:   %.loc137_5.5 => constants.%.e6c
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.ce1
// CHECK:STDOUT:   %impl.elem0.loc137_5.2 => constants.%Int32.Op.6afff1.1
// CHECK:STDOUT:   %specific_impl_fn.loc137_5.4 => constants.%Int32.Op.6afff1.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @MulWith.loc146(constants.%Destroy.facet.c06, constants.%Destroy.facet.e06, constants.%facet_value.daa) {
// CHECK:STDOUT:   %Result.loc143_9.1 => constants.%Destroy.facet.c06
// CHECK:STDOUT:   %U.loc144_4.1 => constants.%Destroy.facet.e06
// CHECK:STDOUT:   %U.as_type.loc145_36.1 => constants.%Int16
// CHECK:STDOUT:   %MulWith.type.loc145_36.1 => constants.%MulWith.type.0cd
// CHECK:STDOUT:   %facet_type.loc145_20 => constants.%facet_type.2dd
// CHECK:STDOUT:   %.Self.1 => constants.%.Self.594
// CHECK:STDOUT:   %require_complete.loc145 => constants.%complete_type.b32
// CHECK:STDOUT:   %.Self.as_type.loc145_44.1 => constants.%.Self.as_type.674
// CHECK:STDOUT:   %MulWith.assoc_type => constants.%MulWith.assoc_type.f1b
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.3ee
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.loc145 => constants.%MulWith.lookup_impl_witness.834
// CHECK:STDOUT:   %impl.elem0.loc145_44.1 => constants.%impl.elem0.6fb
// CHECK:STDOUT:   %Result.as_type.loc145_54.1 => constants.%Int32
// CHECK:STDOUT:   %facet_type.loc145_38 => constants.%facet_type.bfe
// CHECK:STDOUT:   %T.loc145_4.1 => constants.%facet_value.daa
// CHECK:STDOUT:   %pattern_type.loc145 => constants.%pattern_type.8a2
// CHECK:STDOUT:   %pattern_type.loc146_16 => constants.%pattern_type.9c9
// CHECK:STDOUT:   %T.as_type.loc146_29.1 => constants.%Int32
// CHECK:STDOUT:   %pattern_type.loc146_27 => constants.%pattern_type.9c9
// CHECK:STDOUT:   %pattern_type.loc146_33 => constants.%pattern_type.9a4
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc146_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc146_27 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc146_33 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc148 => constants.%complete_type.4d2
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.988
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.loc148 => constants.%custom_witness.04f
// CHECK:STDOUT:   %MulWith.facet => constants.%MulWith.facet.aea
// CHECK:STDOUT:   %MulWith.WithSelf.Op.type => constants.%MulWith.WithSelf.Op.type.f3c
// CHECK:STDOUT:   %.loc148_5.4 => constants.%.a30
// CHECK:STDOUT:   %impl.elem1.loc148_5.2 => constants.%Int32.Op.6afff1.4
// CHECK:STDOUT:   %specific_impl_fn.loc148_5.3 => constants.%Int32.Op.6afff1.4
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.2e1
// CHECK:STDOUT:   %.loc148_5.5 => constants.%.e6c
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.ce1
// CHECK:STDOUT:   %impl.elem0.loc148_5.2 => constants.%Int32.Op.6afff1.1
// CHECK:STDOUT:   %specific_impl_fn.loc148_5.4 => constants.%Int32.Op.6afff1.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @DivWith.loc157(constants.%Destroy.facet.c06, constants.%Destroy.facet.e06, constants.%facet_value.437) {
// CHECK:STDOUT:   %Result.loc154_9.1 => constants.%Destroy.facet.c06
// CHECK:STDOUT:   %U.loc155_4.1 => constants.%Destroy.facet.e06
// CHECK:STDOUT:   %U.as_type.loc156_36.1 => constants.%Int16
// CHECK:STDOUT:   %DivWith.type.loc156_36.1 => constants.%DivWith.type.4c2
// CHECK:STDOUT:   %facet_type.loc156_20 => constants.%facet_type.e82
// CHECK:STDOUT:   %.Self.1 => constants.%.Self.c70
// CHECK:STDOUT:   %require_complete.loc156 => constants.%complete_type.8b32
// CHECK:STDOUT:   %.Self.as_type.loc156_44.1 => constants.%.Self.as_type.e0e
// CHECK:STDOUT:   %DivWith.assoc_type => constants.%DivWith.assoc_type.9fa
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.cf5
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.loc156 => constants.%DivWith.lookup_impl_witness.dbb
// CHECK:STDOUT:   %impl.elem0.loc156_44.1 => constants.%impl.elem0.e43
// CHECK:STDOUT:   %Result.as_type.loc156_54.1 => constants.%Int32
// CHECK:STDOUT:   %facet_type.loc156_38 => constants.%facet_type.917
// CHECK:STDOUT:   %T.loc156_4.1 => constants.%facet_value.437
// CHECK:STDOUT:   %pattern_type.loc156 => constants.%pattern_type.b7f
// CHECK:STDOUT:   %pattern_type.loc157_16 => constants.%pattern_type.9c9
// CHECK:STDOUT:   %T.as_type.loc157_29.1 => constants.%Int32
// CHECK:STDOUT:   %pattern_type.loc157_27 => constants.%pattern_type.9c9
// CHECK:STDOUT:   %pattern_type.loc157_33 => constants.%pattern_type.9a4
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc157_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc157_27 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc157_33 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc159 => constants.%complete_type.e42
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.cda
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.loc159 => constants.%custom_witness.53b
// CHECK:STDOUT:   %DivWith.facet => constants.%DivWith.facet.687
// CHECK:STDOUT:   %DivWith.WithSelf.Op.type => constants.%DivWith.WithSelf.Op.type.b6a
// CHECK:STDOUT:   %.loc159_5.4 => constants.%.c6a
// CHECK:STDOUT:   %impl.elem1.loc159_5.2 => constants.%Int32.Op.6afff1.5
// CHECK:STDOUT:   %specific_impl_fn.loc159_5.3 => constants.%Int32.Op.6afff1.5
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.2e1
// CHECK:STDOUT:   %.loc159_5.5 => constants.%.e6c
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.ce1
// CHECK:STDOUT:   %impl.elem0.loc159_5.2 => constants.%Int32.Op.6afff1.1
// CHECK:STDOUT:   %specific_impl_fn.loc159_5.4 => constants.%Int32.Op.6afff1.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ModWith.loc168(constants.%Destroy.facet.c06, constants.%Destroy.facet.e06, constants.%facet_value.9f1) {
// CHECK:STDOUT:   %Result.loc165_9.1 => constants.%Destroy.facet.c06
// CHECK:STDOUT:   %U.loc166_4.1 => constants.%Destroy.facet.e06
// CHECK:STDOUT:   %U.as_type.loc167_36.1 => constants.%Int16
// CHECK:STDOUT:   %ModWith.type.loc167_36.1 => constants.%ModWith.type.009
// CHECK:STDOUT:   %facet_type.loc167_20 => constants.%facet_type.e1e
// CHECK:STDOUT:   %.Self.1 => constants.%.Self.76e
// CHECK:STDOUT:   %require_complete.loc167 => constants.%complete_type.5b6
// CHECK:STDOUT:   %.Self.as_type.loc167_44.1 => constants.%.Self.as_type.242
// CHECK:STDOUT:   %ModWith.assoc_type => constants.%ModWith.assoc_type.355
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.0f7
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.loc167 => constants.%ModWith.lookup_impl_witness.c6b
// CHECK:STDOUT:   %impl.elem0.loc167_44.1 => constants.%impl.elem0.cba
// CHECK:STDOUT:   %Result.as_type.loc167_54.1 => constants.%Int32
// CHECK:STDOUT:   %facet_type.loc167_38 => constants.%facet_type.d08
// CHECK:STDOUT:   %T.loc167_4.1 => constants.%facet_value.9f1
// CHECK:STDOUT:   %pattern_type.loc167 => constants.%pattern_type.a04
// CHECK:STDOUT:   %pattern_type.loc168_16 => constants.%pattern_type.9c9
// CHECK:STDOUT:   %T.as_type.loc168_29.1 => constants.%Int32
// CHECK:STDOUT:   %pattern_type.loc168_27 => constants.%pattern_type.9c9
// CHECK:STDOUT:   %pattern_type.loc168_33 => constants.%pattern_type.9a4
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc168_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc168_27 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc168_33 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc170 => constants.%complete_type.e9f
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.c82
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.loc170 => constants.%custom_witness.895
// CHECK:STDOUT:   %ModWith.facet => constants.%ModWith.facet.7e6
// CHECK:STDOUT:   %ModWith.WithSelf.Op.type => constants.%ModWith.WithSelf.Op.type.55e
// CHECK:STDOUT:   %.loc170_5.4 => constants.%.274
// CHECK:STDOUT:   %impl.elem1.loc170_5.2 => constants.%Int32.Op.6afff1.6
// CHECK:STDOUT:   %specific_impl_fn.loc170_5.3 => constants.%Int32.Op.6afff1.6
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.2e1
// CHECK:STDOUT:   %.loc170_5.5 => constants.%.e6c
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.ce1
// CHECK:STDOUT:   %impl.elem0.loc170_5.2 => constants.%Int32.Op.6afff1.1
// CHECK:STDOUT:   %specific_impl_fn.loc170_5.4 => constants.%Int32.Op.6afff1.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AddWith.loc124(constants.%Destroy.facet.c06, constants.%Destroy.facet.c06, constants.%facet_value.fca) {
// CHECK:STDOUT:   %Result.loc121_9.1 => constants.%Destroy.facet.c06
// CHECK:STDOUT:   %U.loc122_4.1 => constants.%Destroy.facet.c06
// CHECK:STDOUT:   %U.as_type.loc123_36.1 => constants.%Int32
// CHECK:STDOUT:   %AddWith.type.loc123_36.1 => constants.%AddWith.type.e9e
// CHECK:STDOUT:   %facet_type.loc123_20 => constants.%facet_type.4a7
// CHECK:STDOUT:   %.Self.1 => constants.%.Self.3d9
// CHECK:STDOUT:   %require_complete.loc123 => constants.%complete_type.0d9
// CHECK:STDOUT:   %.Self.as_type.loc123_44.1 => constants.%.Self.as_type.7ba
// CHECK:STDOUT:   %AddWith.assoc_type => constants.%AddWith.assoc_type.29a
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.701
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.loc123 => constants.%AddWith.lookup_impl_witness.632
// CHECK:STDOUT:   %impl.elem0.loc123_44.1 => constants.%impl.elem0.cee
// CHECK:STDOUT:   %Result.as_type.loc123_54.1 => constants.%Int32
// CHECK:STDOUT:   %facet_type.loc123_38 => constants.%facet_type.14d
// CHECK:STDOUT:   %T.loc123_4.1 => constants.%facet_value.fca
// CHECK:STDOUT:   %pattern_type.loc123 => constants.%pattern_type.3f9
// CHECK:STDOUT:   %pattern_type.loc124_16 => constants.%pattern_type.9c9
// CHECK:STDOUT:   %T.as_type.loc124_29.1 => constants.%Int32
// CHECK:STDOUT:   %pattern_type.loc124_27 => constants.%pattern_type.9c9
// CHECK:STDOUT:   %pattern_type.loc124_33 => constants.%pattern_type.9c9
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc124_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc124_27 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc124_33 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc126 => constants.%complete_type.03d
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.38b
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.loc126 => constants.%custom_witness.78d
// CHECK:STDOUT:   %AddWith.facet => constants.%AddWith.facet.e5b
// CHECK:STDOUT:   %AddWith.WithSelf.Op.type => constants.%AddWith.WithSelf.Op.type.13e
// CHECK:STDOUT:   %.loc126_5.4 => constants.%.56e
// CHECK:STDOUT:   %impl.elem1.loc126_5.2 => constants.%Int32.Op.6afff1.7
// CHECK:STDOUT:   %specific_impl_fn.loc126_5.3 => constants.%Int32.Op.6afff1.7
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.2e1
// CHECK:STDOUT:   %.loc126_5.5 => constants.%.e6c
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.ce1
// CHECK:STDOUT:   %impl.elem0.loc126_5.2 => constants.%Int32.Op.6afff1.1
// CHECK:STDOUT:   %specific_impl_fn.loc126_5.4 => constants.%Int32.Op.6afff1.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @SubWith.loc135(constants.%Destroy.facet.c06, constants.%Destroy.facet.c06, constants.%facet_value.9a9) {
// CHECK:STDOUT:   %Result.loc132_9.1 => constants.%Destroy.facet.c06
// CHECK:STDOUT:   %U.loc133_4.1 => constants.%Destroy.facet.c06
// CHECK:STDOUT:   %U.as_type.loc134_36.1 => constants.%Int32
// CHECK:STDOUT:   %SubWith.type.loc134_36.1 => constants.%SubWith.type.84c
// CHECK:STDOUT:   %facet_type.loc134_20 => constants.%facet_type.e7a
// CHECK:STDOUT:   %.Self.1 => constants.%.Self.384
// CHECK:STDOUT:   %require_complete.loc134 => constants.%complete_type.a34
// CHECK:STDOUT:   %.Self.as_type.loc134_44.1 => constants.%.Self.as_type.45d
// CHECK:STDOUT:   %SubWith.assoc_type => constants.%SubWith.assoc_type.0f9
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.1e9
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.loc134 => constants.%SubWith.lookup_impl_witness.b61
// CHECK:STDOUT:   %impl.elem0.loc134_44.1 => constants.%impl.elem0.edd
// CHECK:STDOUT:   %Result.as_type.loc134_54.1 => constants.%Int32
// CHECK:STDOUT:   %facet_type.loc134_38 => constants.%facet_type.f15
// CHECK:STDOUT:   %T.loc134_4.1 => constants.%facet_value.9a9
// CHECK:STDOUT:   %pattern_type.loc134 => constants.%pattern_type.56e
// CHECK:STDOUT:   %pattern_type.loc135_16 => constants.%pattern_type.9c9
// CHECK:STDOUT:   %T.as_type.loc135_29.1 => constants.%Int32
// CHECK:STDOUT:   %pattern_type.loc135_27 => constants.%pattern_type.9c9
// CHECK:STDOUT:   %pattern_type.loc135_33 => constants.%pattern_type.9c9
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc135_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc135_27 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc135_33 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc137 => constants.%complete_type.9bf
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.714
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.loc137 => constants.%custom_witness.64d
// CHECK:STDOUT:   %SubWith.facet => constants.%SubWith.facet.e26
// CHECK:STDOUT:   %SubWith.WithSelf.Op.type => constants.%SubWith.WithSelf.Op.type.d8a
// CHECK:STDOUT:   %.loc137_5.4 => constants.%.cb1
// CHECK:STDOUT:   %impl.elem1.loc137_5.2 => constants.%Int32.Op.6afff1.8
// CHECK:STDOUT:   %specific_impl_fn.loc137_5.3 => constants.%Int32.Op.6afff1.8
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.2e1
// CHECK:STDOUT:   %.loc137_5.5 => constants.%.e6c
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.ce1
// CHECK:STDOUT:   %impl.elem0.loc137_5.2 => constants.%Int32.Op.6afff1.1
// CHECK:STDOUT:   %specific_impl_fn.loc137_5.4 => constants.%Int32.Op.6afff1.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @MulWith.loc146(constants.%Destroy.facet.c06, constants.%Destroy.facet.c06, constants.%facet_value.03b) {
// CHECK:STDOUT:   %Result.loc143_9.1 => constants.%Destroy.facet.c06
// CHECK:STDOUT:   %U.loc144_4.1 => constants.%Destroy.facet.c06
// CHECK:STDOUT:   %U.as_type.loc145_36.1 => constants.%Int32
// CHECK:STDOUT:   %MulWith.type.loc145_36.1 => constants.%MulWith.type.d1b
// CHECK:STDOUT:   %facet_type.loc145_20 => constants.%facet_type.2e6
// CHECK:STDOUT:   %.Self.1 => constants.%.Self.84b
// CHECK:STDOUT:   %require_complete.loc145 => constants.%complete_type.043
// CHECK:STDOUT:   %.Self.as_type.loc145_44.1 => constants.%.Self.as_type.d0cd
// CHECK:STDOUT:   %MulWith.assoc_type => constants.%MulWith.assoc_type.8c6
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.a8f
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.loc145 => constants.%MulWith.lookup_impl_witness.539
// CHECK:STDOUT:   %impl.elem0.loc145_44.1 => constants.%impl.elem0.e63
// CHECK:STDOUT:   %Result.as_type.loc145_54.1 => constants.%Int32
// CHECK:STDOUT:   %facet_type.loc145_38 => constants.%facet_type.c22
// CHECK:STDOUT:   %T.loc145_4.1 => constants.%facet_value.03b
// CHECK:STDOUT:   %pattern_type.loc145 => constants.%pattern_type.965
// CHECK:STDOUT:   %pattern_type.loc146_16 => constants.%pattern_type.9c9
// CHECK:STDOUT:   %T.as_type.loc146_29.1 => constants.%Int32
// CHECK:STDOUT:   %pattern_type.loc146_27 => constants.%pattern_type.9c9
// CHECK:STDOUT:   %pattern_type.loc146_33 => constants.%pattern_type.9c9
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc146_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc146_27 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc146_33 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc148 => constants.%complete_type.a1f
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.6bc
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.loc148 => constants.%custom_witness.7b6d
// CHECK:STDOUT:   %MulWith.facet => constants.%MulWith.facet.b74
// CHECK:STDOUT:   %MulWith.WithSelf.Op.type => constants.%MulWith.WithSelf.Op.type.25b
// CHECK:STDOUT:   %.loc148_5.4 => constants.%.e42
// CHECK:STDOUT:   %impl.elem1.loc148_5.2 => constants.%Int32.Op.6afff1.9
// CHECK:STDOUT:   %specific_impl_fn.loc148_5.3 => constants.%Int32.Op.6afff1.9
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.2e1
// CHECK:STDOUT:   %.loc148_5.5 => constants.%.e6c
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.ce1
// CHECK:STDOUT:   %impl.elem0.loc148_5.2 => constants.%Int32.Op.6afff1.1
// CHECK:STDOUT:   %specific_impl_fn.loc148_5.4 => constants.%Int32.Op.6afff1.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @DivWith.loc157(constants.%Destroy.facet.c06, constants.%Destroy.facet.c06, constants.%facet_value.a2b) {
// CHECK:STDOUT:   %Result.loc154_9.1 => constants.%Destroy.facet.c06
// CHECK:STDOUT:   %U.loc155_4.1 => constants.%Destroy.facet.c06
// CHECK:STDOUT:   %U.as_type.loc156_36.1 => constants.%Int32
// CHECK:STDOUT:   %DivWith.type.loc156_36.1 => constants.%DivWith.type.26e
// CHECK:STDOUT:   %facet_type.loc156_20 => constants.%facet_type.eb5
// CHECK:STDOUT:   %.Self.1 => constants.%.Self.203
// CHECK:STDOUT:   %require_complete.loc156 => constants.%complete_type.d76
// CHECK:STDOUT:   %.Self.as_type.loc156_44.1 => constants.%.Self.as_type.f8e
// CHECK:STDOUT:   %DivWith.assoc_type => constants.%DivWith.assoc_type.109
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.d76
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.loc156 => constants.%DivWith.lookup_impl_witness.512
// CHECK:STDOUT:   %impl.elem0.loc156_44.1 => constants.%impl.elem0.232
// CHECK:STDOUT:   %Result.as_type.loc156_54.1 => constants.%Int32
// CHECK:STDOUT:   %facet_type.loc156_38 => constants.%facet_type.529
// CHECK:STDOUT:   %T.loc156_4.1 => constants.%facet_value.a2b
// CHECK:STDOUT:   %pattern_type.loc156 => constants.%pattern_type.04c
// CHECK:STDOUT:   %pattern_type.loc157_16 => constants.%pattern_type.9c9
// CHECK:STDOUT:   %T.as_type.loc157_29.1 => constants.%Int32
// CHECK:STDOUT:   %pattern_type.loc157_27 => constants.%pattern_type.9c9
// CHECK:STDOUT:   %pattern_type.loc157_33 => constants.%pattern_type.9c9
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc157_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc157_27 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc157_33 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc159 => constants.%complete_type.e49
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.4a0
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.loc159 => constants.%custom_witness.db6
// CHECK:STDOUT:   %DivWith.facet => constants.%DivWith.facet.8b3
// CHECK:STDOUT:   %DivWith.WithSelf.Op.type => constants.%DivWith.WithSelf.Op.type.f6d
// CHECK:STDOUT:   %.loc159_5.4 => constants.%.c6f
// CHECK:STDOUT:   %impl.elem1.loc159_5.2 => constants.%Int32.Op.6afff1.10
// CHECK:STDOUT:   %specific_impl_fn.loc159_5.3 => constants.%Int32.Op.6afff1.10
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.2e1
// CHECK:STDOUT:   %.loc159_5.5 => constants.%.e6c
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.ce1
// CHECK:STDOUT:   %impl.elem0.loc159_5.2 => constants.%Int32.Op.6afff1.1
// CHECK:STDOUT:   %specific_impl_fn.loc159_5.4 => constants.%Int32.Op.6afff1.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ModWith.loc168(constants.%Destroy.facet.c06, constants.%Destroy.facet.c06, constants.%facet_value.a4c) {
// CHECK:STDOUT:   %Result.loc165_9.1 => constants.%Destroy.facet.c06
// CHECK:STDOUT:   %U.loc166_4.1 => constants.%Destroy.facet.c06
// CHECK:STDOUT:   %U.as_type.loc167_36.1 => constants.%Int32
// CHECK:STDOUT:   %ModWith.type.loc167_36.1 => constants.%ModWith.type.5c4
// CHECK:STDOUT:   %facet_type.loc167_20 => constants.%facet_type.1e6
// CHECK:STDOUT:   %.Self.1 => constants.%.Self.0c5
// CHECK:STDOUT:   %require_complete.loc167 => constants.%complete_type.1bb
// CHECK:STDOUT:   %.Self.as_type.loc167_44.1 => constants.%.Self.as_type.ef1
// CHECK:STDOUT:   %ModWith.assoc_type => constants.%ModWith.assoc_type.1db
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.8e2
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.loc167 => constants.%ModWith.lookup_impl_witness.9f7
// CHECK:STDOUT:   %impl.elem0.loc167_44.1 => constants.%impl.elem0.919
// CHECK:STDOUT:   %Result.as_type.loc167_54.1 => constants.%Int32
// CHECK:STDOUT:   %facet_type.loc167_38 => constants.%facet_type.422
// CHECK:STDOUT:   %T.loc167_4.1 => constants.%facet_value.a4c
// CHECK:STDOUT:   %pattern_type.loc167 => constants.%pattern_type.4d3
// CHECK:STDOUT:   %pattern_type.loc168_16 => constants.%pattern_type.9c9
// CHECK:STDOUT:   %T.as_type.loc168_29.1 => constants.%Int32
// CHECK:STDOUT:   %pattern_type.loc168_27 => constants.%pattern_type.9c9
// CHECK:STDOUT:   %pattern_type.loc168_33 => constants.%pattern_type.9c9
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc168_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc168_27 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc168_33 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc170 => constants.%complete_type.3ed
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.826
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.loc170 => constants.%custom_witness.e01
// CHECK:STDOUT:   %ModWith.facet => constants.%ModWith.facet.c6b
// CHECK:STDOUT:   %ModWith.WithSelf.Op.type => constants.%ModWith.WithSelf.Op.type.b72
// CHECK:STDOUT:   %.loc170_5.4 => constants.%.a97
// CHECK:STDOUT:   %impl.elem1.loc170_5.2 => constants.%Int32.Op.6afff1.11
// CHECK:STDOUT:   %specific_impl_fn.loc170_5.3 => constants.%Int32.Op.6afff1.11
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.2e1
// CHECK:STDOUT:   %.loc170_5.5 => constants.%.e6c
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.ce1
// CHECK:STDOUT:   %impl.elem0.loc170_5.2 => constants.%Int32.Op.6afff1.1
// CHECK:STDOUT:   %specific_impl_fn.loc170_5.4 => constants.%Int32.Op.6afff1.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AddWith.loc124(constants.%Destroy.facet.b0a, constants.%Destroy.facet.b0a, constants.%facet_value.054) {
// CHECK:STDOUT:   %Result.loc121_9.1 => constants.%Destroy.facet.b0a
// CHECK:STDOUT:   %U.loc122_4.1 => constants.%Destroy.facet.b0a
// CHECK:STDOUT:   %U.as_type.loc123_36.1 => constants.%Int64
// CHECK:STDOUT:   %AddWith.type.loc123_36.1 => constants.%AddWith.type.ee5
// CHECK:STDOUT:   %facet_type.loc123_20 => constants.%facet_type.221
// CHECK:STDOUT:   %.Self.1 => constants.%.Self.954
// CHECK:STDOUT:   %require_complete.loc123 => constants.%complete_type.7fa
// CHECK:STDOUT:   %.Self.as_type.loc123_44.1 => constants.%.Self.as_type.083
// CHECK:STDOUT:   %AddWith.assoc_type => constants.%AddWith.assoc_type.330
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.1fe
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.loc123 => constants.%AddWith.lookup_impl_witness.076
// CHECK:STDOUT:   %impl.elem0.loc123_44.1 => constants.%impl.elem0.d09
// CHECK:STDOUT:   %Result.as_type.loc123_54.1 => constants.%Int64
// CHECK:STDOUT:   %facet_type.loc123_38 => constants.%facet_type.7b1
// CHECK:STDOUT:   %T.loc123_4.1 => constants.%facet_value.054
// CHECK:STDOUT:   %pattern_type.loc123 => constants.%pattern_type.aa1
// CHECK:STDOUT:   %pattern_type.loc124_16 => constants.%pattern_type.f42
// CHECK:STDOUT:   %T.as_type.loc124_29.1 => constants.%Int32
// CHECK:STDOUT:   %pattern_type.loc124_27 => constants.%pattern_type.9c9
// CHECK:STDOUT:   %pattern_type.loc124_33 => constants.%pattern_type.f42
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc124_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc124_27 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc124_33 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc126 => constants.%complete_type.d2d
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.cd6
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.loc126 => constants.%custom_witness.1b7
// CHECK:STDOUT:   %AddWith.facet => constants.%AddWith.facet.484
// CHECK:STDOUT:   %AddWith.WithSelf.Op.type => constants.%AddWith.WithSelf.Op.type.24f
// CHECK:STDOUT:   %.loc126_5.4 => constants.%.442
// CHECK:STDOUT:   %impl.elem1.loc126_5.2 => constants.%Op.8fb557.6
// CHECK:STDOUT:   %specific_impl_fn.loc126_5.3 => constants.%Op.8fb557.6
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.f16
// CHECK:STDOUT:   %.loc126_5.5 => constants.%.db1
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.d7e
// CHECK:STDOUT:   %impl.elem0.loc126_5.2 => constants.%Int64.Op.b941da.1
// CHECK:STDOUT:   %specific_impl_fn.loc126_5.4 => constants.%Int64.Op.b941da.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @SubWith.loc135(constants.%Destroy.facet.b0a, constants.%Destroy.facet.b0a, constants.%facet_value.f7f) {
// CHECK:STDOUT:   %Result.loc132_9.1 => constants.%Destroy.facet.b0a
// CHECK:STDOUT:   %U.loc133_4.1 => constants.%Destroy.facet.b0a
// CHECK:STDOUT:   %U.as_type.loc134_36.1 => constants.%Int64
// CHECK:STDOUT:   %SubWith.type.loc134_36.1 => constants.%SubWith.type.b41
// CHECK:STDOUT:   %facet_type.loc134_20 => constants.%facet_type.660
// CHECK:STDOUT:   %.Self.1 => constants.%.Self.ea7
// CHECK:STDOUT:   %require_complete.loc134 => constants.%complete_type.6f3
// CHECK:STDOUT:   %.Self.as_type.loc134_44.1 => constants.%.Self.as_type.d4d
// CHECK:STDOUT:   %SubWith.assoc_type => constants.%SubWith.assoc_type.b7b
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.979
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.loc134 => constants.%SubWith.lookup_impl_witness.894
// CHECK:STDOUT:   %impl.elem0.loc134_44.1 => constants.%impl.elem0.9dd
// CHECK:STDOUT:   %Result.as_type.loc134_54.1 => constants.%Int64
// CHECK:STDOUT:   %facet_type.loc134_38 => constants.%facet_type.17a
// CHECK:STDOUT:   %T.loc134_4.1 => constants.%facet_value.f7f
// CHECK:STDOUT:   %pattern_type.loc134 => constants.%pattern_type.30e
// CHECK:STDOUT:   %pattern_type.loc135_16 => constants.%pattern_type.f42
// CHECK:STDOUT:   %T.as_type.loc135_29.1 => constants.%Int32
// CHECK:STDOUT:   %pattern_type.loc135_27 => constants.%pattern_type.9c9
// CHECK:STDOUT:   %pattern_type.loc135_33 => constants.%pattern_type.f42
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc135_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc135_27 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc135_33 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc137 => constants.%complete_type.c98
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.80d
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.loc137 => constants.%custom_witness.cab
// CHECK:STDOUT:   %SubWith.facet => constants.%SubWith.facet.80e
// CHECK:STDOUT:   %SubWith.WithSelf.Op.type => constants.%SubWith.WithSelf.Op.type.c99
// CHECK:STDOUT:   %.loc137_5.4 => constants.%.9c9
// CHECK:STDOUT:   %impl.elem1.loc137_5.2 => constants.%Op.8fb557.7
// CHECK:STDOUT:   %specific_impl_fn.loc137_5.3 => constants.%Op.8fb557.7
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.f16
// CHECK:STDOUT:   %.loc137_5.5 => constants.%.db1
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.d7e
// CHECK:STDOUT:   %impl.elem0.loc137_5.2 => constants.%Int64.Op.b941da.1
// CHECK:STDOUT:   %specific_impl_fn.loc137_5.4 => constants.%Int64.Op.b941da.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @MulWith.loc146(constants.%Destroy.facet.b0a, constants.%Destroy.facet.b0a, constants.%facet_value.823) {
// CHECK:STDOUT:   %Result.loc143_9.1 => constants.%Destroy.facet.b0a
// CHECK:STDOUT:   %U.loc144_4.1 => constants.%Destroy.facet.b0a
// CHECK:STDOUT:   %U.as_type.loc145_36.1 => constants.%Int64
// CHECK:STDOUT:   %MulWith.type.loc145_36.1 => constants.%MulWith.type.166
// CHECK:STDOUT:   %facet_type.loc145_20 => constants.%facet_type.60a
// CHECK:STDOUT:   %.Self.1 => constants.%.Self.39e
// CHECK:STDOUT:   %require_complete.loc145 => constants.%complete_type.7a4
// CHECK:STDOUT:   %.Self.as_type.loc145_44.1 => constants.%.Self.as_type.604
// CHECK:STDOUT:   %MulWith.assoc_type => constants.%MulWith.assoc_type.cbc
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.c58
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.loc145 => constants.%MulWith.lookup_impl_witness.a30
// CHECK:STDOUT:   %impl.elem0.loc145_44.1 => constants.%impl.elem0.3f6
// CHECK:STDOUT:   %Result.as_type.loc145_54.1 => constants.%Int64
// CHECK:STDOUT:   %facet_type.loc145_38 => constants.%facet_type.9d2
// CHECK:STDOUT:   %T.loc145_4.1 => constants.%facet_value.823
// CHECK:STDOUT:   %pattern_type.loc145 => constants.%pattern_type.eb5
// CHECK:STDOUT:   %pattern_type.loc146_16 => constants.%pattern_type.f42
// CHECK:STDOUT:   %T.as_type.loc146_29.1 => constants.%Int32
// CHECK:STDOUT:   %pattern_type.loc146_27 => constants.%pattern_type.9c9
// CHECK:STDOUT:   %pattern_type.loc146_33 => constants.%pattern_type.f42
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc146_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc146_27 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc146_33 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc148 => constants.%complete_type.b09
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.279
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.loc148 => constants.%custom_witness.f53
// CHECK:STDOUT:   %MulWith.facet => constants.%MulWith.facet.0d5
// CHECK:STDOUT:   %MulWith.WithSelf.Op.type => constants.%MulWith.WithSelf.Op.type.97b
// CHECK:STDOUT:   %.loc148_5.4 => constants.%.07b
// CHECK:STDOUT:   %impl.elem1.loc148_5.2 => constants.%Op.8fb557.8
// CHECK:STDOUT:   %specific_impl_fn.loc148_5.3 => constants.%Op.8fb557.8
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.f16
// CHECK:STDOUT:   %.loc148_5.5 => constants.%.db1
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.d7e
// CHECK:STDOUT:   %impl.elem0.loc148_5.2 => constants.%Int64.Op.b941da.1
// CHECK:STDOUT:   %specific_impl_fn.loc148_5.4 => constants.%Int64.Op.b941da.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @DivWith.loc157(constants.%Destroy.facet.b0a, constants.%Destroy.facet.b0a, constants.%facet_value.cba) {
// CHECK:STDOUT:   %Result.loc154_9.1 => constants.%Destroy.facet.b0a
// CHECK:STDOUT:   %U.loc155_4.1 => constants.%Destroy.facet.b0a
// CHECK:STDOUT:   %U.as_type.loc156_36.1 => constants.%Int64
// CHECK:STDOUT:   %DivWith.type.loc156_36.1 => constants.%DivWith.type.70f
// CHECK:STDOUT:   %facet_type.loc156_20 => constants.%facet_type.f02
// CHECK:STDOUT:   %.Self.1 => constants.%.Self.ad5
// CHECK:STDOUT:   %require_complete.loc156 => constants.%complete_type.e31
// CHECK:STDOUT:   %.Self.as_type.loc156_44.1 => constants.%.Self.as_type.245
// CHECK:STDOUT:   %DivWith.assoc_type => constants.%DivWith.assoc_type.03e
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.a25
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.loc156 => constants.%DivWith.lookup_impl_witness.cc2
// CHECK:STDOUT:   %impl.elem0.loc156_44.1 => constants.%impl.elem0.2d2
// CHECK:STDOUT:   %Result.as_type.loc156_54.1 => constants.%Int64
// CHECK:STDOUT:   %facet_type.loc156_38 => constants.%facet_type.6aa
// CHECK:STDOUT:   %T.loc156_4.1 => constants.%facet_value.cba
// CHECK:STDOUT:   %pattern_type.loc156 => constants.%pattern_type.c59
// CHECK:STDOUT:   %pattern_type.loc157_16 => constants.%pattern_type.f42
// CHECK:STDOUT:   %T.as_type.loc157_29.1 => constants.%Int32
// CHECK:STDOUT:   %pattern_type.loc157_27 => constants.%pattern_type.9c9
// CHECK:STDOUT:   %pattern_type.loc157_33 => constants.%pattern_type.f42
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc157_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc157_27 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc157_33 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc159 => constants.%complete_type.09b
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.f42
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.loc159 => constants.%custom_witness.bca
// CHECK:STDOUT:   %DivWith.facet => constants.%DivWith.facet.22c
// CHECK:STDOUT:   %DivWith.WithSelf.Op.type => constants.%DivWith.WithSelf.Op.type.565
// CHECK:STDOUT:   %.loc159_5.4 => constants.%.b72
// CHECK:STDOUT:   %impl.elem1.loc159_5.2 => constants.%Op.8fb557.9
// CHECK:STDOUT:   %specific_impl_fn.loc159_5.3 => constants.%Op.8fb557.9
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.f16
// CHECK:STDOUT:   %.loc159_5.5 => constants.%.db1
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.d7e
// CHECK:STDOUT:   %impl.elem0.loc159_5.2 => constants.%Int64.Op.b941da.1
// CHECK:STDOUT:   %specific_impl_fn.loc159_5.4 => constants.%Int64.Op.b941da.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ModWith.loc168(constants.%Destroy.facet.b0a, constants.%Destroy.facet.b0a, constants.%facet_value.403) {
// CHECK:STDOUT:   %Result.loc165_9.1 => constants.%Destroy.facet.b0a
// CHECK:STDOUT:   %U.loc166_4.1 => constants.%Destroy.facet.b0a
// CHECK:STDOUT:   %U.as_type.loc167_36.1 => constants.%Int64
// CHECK:STDOUT:   %ModWith.type.loc167_36.1 => constants.%ModWith.type.a92
// CHECK:STDOUT:   %facet_type.loc167_20 => constants.%facet_type.b13
// CHECK:STDOUT:   %.Self.1 => constants.%.Self.856
// CHECK:STDOUT:   %require_complete.loc167 => constants.%complete_type.780
// CHECK:STDOUT:   %.Self.as_type.loc167_44.1 => constants.%.Self.as_type.6bd
// CHECK:STDOUT:   %ModWith.assoc_type => constants.%ModWith.assoc_type.f50
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.c89
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.loc167 => constants.%ModWith.lookup_impl_witness.81f
// CHECK:STDOUT:   %impl.elem0.loc167_44.1 => constants.%impl.elem0.369
// CHECK:STDOUT:   %Result.as_type.loc167_54.1 => constants.%Int64
// CHECK:STDOUT:   %facet_type.loc167_38 => constants.%facet_type.41c
// CHECK:STDOUT:   %T.loc167_4.1 => constants.%facet_value.403
// CHECK:STDOUT:   %pattern_type.loc167 => constants.%pattern_type.4ab
// CHECK:STDOUT:   %pattern_type.loc168_16 => constants.%pattern_type.f42
// CHECK:STDOUT:   %T.as_type.loc168_29.1 => constants.%Int32
// CHECK:STDOUT:   %pattern_type.loc168_27 => constants.%pattern_type.9c9
// CHECK:STDOUT:   %pattern_type.loc168_33 => constants.%pattern_type.f42
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc168_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc168_27 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc168_33 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc170 => constants.%complete_type.6fa
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.a80
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.loc170 => constants.%custom_witness.1b0
// CHECK:STDOUT:   %ModWith.facet => constants.%ModWith.facet.570
// CHECK:STDOUT:   %ModWith.WithSelf.Op.type => constants.%ModWith.WithSelf.Op.type.2b2
// CHECK:STDOUT:   %.loc170_5.4 => constants.%.16c
// CHECK:STDOUT:   %impl.elem1.loc170_5.2 => constants.%Op.8fb557.10
// CHECK:STDOUT:   %specific_impl_fn.loc170_5.3 => constants.%Op.8fb557.10
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.f16
// CHECK:STDOUT:   %.loc170_5.5 => constants.%.db1
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.d7e
// CHECK:STDOUT:   %impl.elem0.loc170_5.2 => constants.%Int64.Op.b941da.1
// CHECK:STDOUT:   %specific_impl_fn.loc170_5.4 => constants.%Int64.Op.b941da.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AddWith.loc124(constants.%Destroy.facet.b0a, constants.%Destroy.facet.c06, constants.%facet_value.8dc) {
// CHECK:STDOUT:   %Result.loc121_9.1 => constants.%Destroy.facet.b0a
// CHECK:STDOUT:   %U.loc122_4.1 => constants.%Destroy.facet.c06
// CHECK:STDOUT:   %U.as_type.loc123_36.1 => constants.%Int32
// CHECK:STDOUT:   %AddWith.type.loc123_36.1 => constants.%AddWith.type.e9e
// CHECK:STDOUT:   %facet_type.loc123_20 => constants.%facet_type.4a7
// CHECK:STDOUT:   %.Self.1 => constants.%.Self.3d9
// CHECK:STDOUT:   %require_complete.loc123 => constants.%complete_type.0d9
// CHECK:STDOUT:   %.Self.as_type.loc123_44.1 => constants.%.Self.as_type.7ba
// CHECK:STDOUT:   %AddWith.assoc_type => constants.%AddWith.assoc_type.29a
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.701
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.loc123 => constants.%AddWith.lookup_impl_witness.632
// CHECK:STDOUT:   %impl.elem0.loc123_44.1 => constants.%impl.elem0.cee
// CHECK:STDOUT:   %Result.as_type.loc123_54.1 => constants.%Int64
// CHECK:STDOUT:   %facet_type.loc123_38 => constants.%facet_type.a53
// CHECK:STDOUT:   %T.loc123_4.1 => constants.%facet_value.8dc
// CHECK:STDOUT:   %pattern_type.loc123 => constants.%pattern_type.7f1
// CHECK:STDOUT:   %pattern_type.loc124_16 => constants.%pattern_type.f42
// CHECK:STDOUT:   %T.as_type.loc124_29.1 => constants.%Int64
// CHECK:STDOUT:   %pattern_type.loc124_27 => constants.%pattern_type.f42
// CHECK:STDOUT:   %pattern_type.loc124_33 => constants.%pattern_type.9c9
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc124_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc124_27 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc124_33 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc126 => constants.%complete_type.03d
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.38b
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.loc126 => constants.%custom_witness.fc1
// CHECK:STDOUT:   %AddWith.facet => constants.%AddWith.facet.66c
// CHECK:STDOUT:   %AddWith.WithSelf.Op.type => constants.%AddWith.WithSelf.Op.type.e16
// CHECK:STDOUT:   %.loc126_5.4 => constants.%.41b
// CHECK:STDOUT:   %impl.elem1.loc126_5.2 => constants.%Op.8fb557.11
// CHECK:STDOUT:   %specific_impl_fn.loc126_5.3 => constants.%Op.8fb557.11
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.f16
// CHECK:STDOUT:   %.loc126_5.5 => constants.%.db1
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.d7e
// CHECK:STDOUT:   %impl.elem0.loc126_5.2 => constants.%Int64.Op.b941da.1
// CHECK:STDOUT:   %specific_impl_fn.loc126_5.4 => constants.%Int64.Op.b941da.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @SubWith.loc135(constants.%Destroy.facet.b0a, constants.%Destroy.facet.c06, constants.%facet_value.c40) {
// CHECK:STDOUT:   %Result.loc132_9.1 => constants.%Destroy.facet.b0a
// CHECK:STDOUT:   %U.loc133_4.1 => constants.%Destroy.facet.c06
// CHECK:STDOUT:   %U.as_type.loc134_36.1 => constants.%Int32
// CHECK:STDOUT:   %SubWith.type.loc134_36.1 => constants.%SubWith.type.84c
// CHECK:STDOUT:   %facet_type.loc134_20 => constants.%facet_type.e7a
// CHECK:STDOUT:   %.Self.1 => constants.%.Self.384
// CHECK:STDOUT:   %require_complete.loc134 => constants.%complete_type.a34
// CHECK:STDOUT:   %.Self.as_type.loc134_44.1 => constants.%.Self.as_type.45d
// CHECK:STDOUT:   %SubWith.assoc_type => constants.%SubWith.assoc_type.0f9
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.1e9
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.loc134 => constants.%SubWith.lookup_impl_witness.b61
// CHECK:STDOUT:   %impl.elem0.loc134_44.1 => constants.%impl.elem0.edd
// CHECK:STDOUT:   %Result.as_type.loc134_54.1 => constants.%Int64
// CHECK:STDOUT:   %facet_type.loc134_38 => constants.%facet_type.364
// CHECK:STDOUT:   %T.loc134_4.1 => constants.%facet_value.c40
// CHECK:STDOUT:   %pattern_type.loc134 => constants.%pattern_type.c45
// CHECK:STDOUT:   %pattern_type.loc135_16 => constants.%pattern_type.f42
// CHECK:STDOUT:   %T.as_type.loc135_29.1 => constants.%Int64
// CHECK:STDOUT:   %pattern_type.loc135_27 => constants.%pattern_type.f42
// CHECK:STDOUT:   %pattern_type.loc135_33 => constants.%pattern_type.9c9
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc135_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc135_27 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc135_33 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc137 => constants.%complete_type.9bf
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.714
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.loc137 => constants.%custom_witness.31e
// CHECK:STDOUT:   %SubWith.facet => constants.%SubWith.facet.494
// CHECK:STDOUT:   %SubWith.WithSelf.Op.type => constants.%SubWith.WithSelf.Op.type.0e3
// CHECK:STDOUT:   %.loc137_5.4 => constants.%.936
// CHECK:STDOUT:   %impl.elem1.loc137_5.2 => constants.%Op.8fb557.12
// CHECK:STDOUT:   %specific_impl_fn.loc137_5.3 => constants.%Op.8fb557.12
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.f16
// CHECK:STDOUT:   %.loc137_5.5 => constants.%.db1
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.d7e
// CHECK:STDOUT:   %impl.elem0.loc137_5.2 => constants.%Int64.Op.b941da.1
// CHECK:STDOUT:   %specific_impl_fn.loc137_5.4 => constants.%Int64.Op.b941da.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @MulWith.loc146(constants.%Destroy.facet.b0a, constants.%Destroy.facet.c06, constants.%facet_value.9d5) {
// CHECK:STDOUT:   %Result.loc143_9.1 => constants.%Destroy.facet.b0a
// CHECK:STDOUT:   %U.loc144_4.1 => constants.%Destroy.facet.c06
// CHECK:STDOUT:   %U.as_type.loc145_36.1 => constants.%Int32
// CHECK:STDOUT:   %MulWith.type.loc145_36.1 => constants.%MulWith.type.d1b
// CHECK:STDOUT:   %facet_type.loc145_20 => constants.%facet_type.2e6
// CHECK:STDOUT:   %.Self.1 => constants.%.Self.84b
// CHECK:STDOUT:   %require_complete.loc145 => constants.%complete_type.043
// CHECK:STDOUT:   %.Self.as_type.loc145_44.1 => constants.%.Self.as_type.d0cd
// CHECK:STDOUT:   %MulWith.assoc_type => constants.%MulWith.assoc_type.8c6
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.a8f
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.loc145 => constants.%MulWith.lookup_impl_witness.539
// CHECK:STDOUT:   %impl.elem0.loc145_44.1 => constants.%impl.elem0.e63
// CHECK:STDOUT:   %Result.as_type.loc145_54.1 => constants.%Int64
// CHECK:STDOUT:   %facet_type.loc145_38 => constants.%facet_type.d7d
// CHECK:STDOUT:   %T.loc145_4.1 => constants.%facet_value.9d5
// CHECK:STDOUT:   %pattern_type.loc145 => constants.%pattern_type.450
// CHECK:STDOUT:   %pattern_type.loc146_16 => constants.%pattern_type.f42
// CHECK:STDOUT:   %T.as_type.loc146_29.1 => constants.%Int64
// CHECK:STDOUT:   %pattern_type.loc146_27 => constants.%pattern_type.f42
// CHECK:STDOUT:   %pattern_type.loc146_33 => constants.%pattern_type.9c9
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc146_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc146_27 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc146_33 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc148 => constants.%complete_type.a1f
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.6bc
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.loc148 => constants.%custom_witness.709
// CHECK:STDOUT:   %MulWith.facet => constants.%MulWith.facet.dab
// CHECK:STDOUT:   %MulWith.WithSelf.Op.type => constants.%MulWith.WithSelf.Op.type.549
// CHECK:STDOUT:   %.loc148_5.4 => constants.%.dc0
// CHECK:STDOUT:   %impl.elem1.loc148_5.2 => constants.%Op.8fb557.13
// CHECK:STDOUT:   %specific_impl_fn.loc148_5.3 => constants.%Op.8fb557.13
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.f16
// CHECK:STDOUT:   %.loc148_5.5 => constants.%.db1
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.d7e
// CHECK:STDOUT:   %impl.elem0.loc148_5.2 => constants.%Int64.Op.b941da.1
// CHECK:STDOUT:   %specific_impl_fn.loc148_5.4 => constants.%Int64.Op.b941da.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @DivWith.loc157(constants.%Destroy.facet.b0a, constants.%Destroy.facet.c06, constants.%facet_value.c21) {
// CHECK:STDOUT:   %Result.loc154_9.1 => constants.%Destroy.facet.b0a
// CHECK:STDOUT:   %U.loc155_4.1 => constants.%Destroy.facet.c06
// CHECK:STDOUT:   %U.as_type.loc156_36.1 => constants.%Int32
// CHECK:STDOUT:   %DivWith.type.loc156_36.1 => constants.%DivWith.type.26e
// CHECK:STDOUT:   %facet_type.loc156_20 => constants.%facet_type.eb5
// CHECK:STDOUT:   %.Self.1 => constants.%.Self.203
// CHECK:STDOUT:   %require_complete.loc156 => constants.%complete_type.d76
// CHECK:STDOUT:   %.Self.as_type.loc156_44.1 => constants.%.Self.as_type.f8e
// CHECK:STDOUT:   %DivWith.assoc_type => constants.%DivWith.assoc_type.109
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.d76
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.loc156 => constants.%DivWith.lookup_impl_witness.512
// CHECK:STDOUT:   %impl.elem0.loc156_44.1 => constants.%impl.elem0.232
// CHECK:STDOUT:   %Result.as_type.loc156_54.1 => constants.%Int64
// CHECK:STDOUT:   %facet_type.loc156_38 => constants.%facet_type.a9f
// CHECK:STDOUT:   %T.loc156_4.1 => constants.%facet_value.c21
// CHECK:STDOUT:   %pattern_type.loc156 => constants.%pattern_type.0b0
// CHECK:STDOUT:   %pattern_type.loc157_16 => constants.%pattern_type.f42
// CHECK:STDOUT:   %T.as_type.loc157_29.1 => constants.%Int64
// CHECK:STDOUT:   %pattern_type.loc157_27 => constants.%pattern_type.f42
// CHECK:STDOUT:   %pattern_type.loc157_33 => constants.%pattern_type.9c9
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc157_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc157_27 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc157_33 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc159 => constants.%complete_type.e49
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.4a0
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.loc159 => constants.%custom_witness.47b
// CHECK:STDOUT:   %DivWith.facet => constants.%DivWith.facet.9ad
// CHECK:STDOUT:   %DivWith.WithSelf.Op.type => constants.%DivWith.WithSelf.Op.type.596
// CHECK:STDOUT:   %.loc159_5.4 => constants.%.def
// CHECK:STDOUT:   %impl.elem1.loc159_5.2 => constants.%Op.8fb557.14
// CHECK:STDOUT:   %specific_impl_fn.loc159_5.3 => constants.%Op.8fb557.14
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.f16
// CHECK:STDOUT:   %.loc159_5.5 => constants.%.db1
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.d7e
// CHECK:STDOUT:   %impl.elem0.loc159_5.2 => constants.%Int64.Op.b941da.1
// CHECK:STDOUT:   %specific_impl_fn.loc159_5.4 => constants.%Int64.Op.b941da.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ModWith.loc168(constants.%Destroy.facet.b0a, constants.%Destroy.facet.c06, constants.%facet_value.766) {
// CHECK:STDOUT:   %Result.loc165_9.1 => constants.%Destroy.facet.b0a
// CHECK:STDOUT:   %U.loc166_4.1 => constants.%Destroy.facet.c06
// CHECK:STDOUT:   %U.as_type.loc167_36.1 => constants.%Int32
// CHECK:STDOUT:   %ModWith.type.loc167_36.1 => constants.%ModWith.type.5c4
// CHECK:STDOUT:   %facet_type.loc167_20 => constants.%facet_type.1e6
// CHECK:STDOUT:   %.Self.1 => constants.%.Self.0c5
// CHECK:STDOUT:   %require_complete.loc167 => constants.%complete_type.1bb
// CHECK:STDOUT:   %.Self.as_type.loc167_44.1 => constants.%.Self.as_type.ef1
// CHECK:STDOUT:   %ModWith.assoc_type => constants.%ModWith.assoc_type.1db
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.8e2
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.loc167 => constants.%ModWith.lookup_impl_witness.9f7
// CHECK:STDOUT:   %impl.elem0.loc167_44.1 => constants.%impl.elem0.919
// CHECK:STDOUT:   %Result.as_type.loc167_54.1 => constants.%Int64
// CHECK:STDOUT:   %facet_type.loc167_38 => constants.%facet_type.f0f
// CHECK:STDOUT:   %T.loc167_4.1 => constants.%facet_value.766
// CHECK:STDOUT:   %pattern_type.loc167 => constants.%pattern_type.bd2
// CHECK:STDOUT:   %pattern_type.loc168_16 => constants.%pattern_type.f42
// CHECK:STDOUT:   %T.as_type.loc168_29.1 => constants.%Int64
// CHECK:STDOUT:   %pattern_type.loc168_27 => constants.%pattern_type.f42
// CHECK:STDOUT:   %pattern_type.loc168_33 => constants.%pattern_type.9c9
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc168_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc168_27 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc168_33 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc170 => constants.%complete_type.3ed
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.826
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.loc170 => constants.%custom_witness.ad6
// CHECK:STDOUT:   %ModWith.facet => constants.%ModWith.facet.325
// CHECK:STDOUT:   %ModWith.WithSelf.Op.type => constants.%ModWith.WithSelf.Op.type.efe
// CHECK:STDOUT:   %.loc170_5.4 => constants.%.ef4
// CHECK:STDOUT:   %impl.elem1.loc170_5.2 => constants.%Op.8fb557.15
// CHECK:STDOUT:   %specific_impl_fn.loc170_5.3 => constants.%Op.8fb557.15
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.f16
// CHECK:STDOUT:   %.loc170_5.5 => constants.%.db1
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.d7e
// CHECK:STDOUT:   %impl.elem0.loc170_5.2 => constants.%Int64.Op.b941da.1
// CHECK:STDOUT:   %specific_impl_fn.loc170_5.4 => constants.%Int64.Op.b941da.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AddAssignWith.loc225(constants.%U.67d, constants.%T.a00) {
// CHECK:STDOUT:   %U.loc225_19.1 => constants.%U.67d
// CHECK:STDOUT:   %AddAssignWith.type.loc225_52.1 => constants.%AddAssignWith.type.4d3b38.2
// CHECK:STDOUT:   %T.loc225_29.1 => constants.%T.a00
// CHECK:STDOUT:   %pattern_type.loc225_29 => constants.%pattern_type.092
// CHECK:STDOUT:   %T.as_type.loc225_62.1 => constants.%T.as_type.191
// CHECK:STDOUT:   %pattern_type.loc225_60 => constants.%pattern_type.03496c.2
// CHECK:STDOUT:   %pattern_type.loc225_66 => constants.%pattern_type.51d1c4.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @SubAssignWith.loc231(constants.%U.67d, constants.%T.1de) {
// CHECK:STDOUT:   %U.loc231_19.1 => constants.%U.67d
// CHECK:STDOUT:   %SubAssignWith.type.loc231_52.1 => constants.%SubAssignWith.type.969cbb.2
// CHECK:STDOUT:   %T.loc231_29.1 => constants.%T.1de
// CHECK:STDOUT:   %pattern_type.loc231_29 => constants.%pattern_type.abb
// CHECK:STDOUT:   %T.as_type.loc231_62.1 => constants.%T.as_type.b8f
// CHECK:STDOUT:   %pattern_type.loc231_60 => constants.%pattern_type.2f1979.2
// CHECK:STDOUT:   %pattern_type.loc231_66 => constants.%pattern_type.51d1c4.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @MulAssignWith.loc237(constants.%U.67d, constants.%T.059) {
// CHECK:STDOUT:   %U.loc237_19.1 => constants.%U.67d
// CHECK:STDOUT:   %MulAssignWith.type.loc237_52.1 => constants.%MulAssignWith.type.ccc9e0.2
// CHECK:STDOUT:   %T.loc237_29.1 => constants.%T.059
// CHECK:STDOUT:   %pattern_type.loc237_29 => constants.%pattern_type.043
// CHECK:STDOUT:   %T.as_type.loc237_62.1 => constants.%T.as_type.6a0
// CHECK:STDOUT:   %pattern_type.loc237_60 => constants.%pattern_type.587662.2
// CHECK:STDOUT:   %pattern_type.loc237_66 => constants.%pattern_type.51d1c4.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @DivAssignWith.loc243(constants.%U.67d, constants.%T.aa8) {
// CHECK:STDOUT:   %U.loc243_19.1 => constants.%U.67d
// CHECK:STDOUT:   %DivAssignWith.type.loc243_52.1 => constants.%DivAssignWith.type.be08dd.2
// CHECK:STDOUT:   %T.loc243_29.1 => constants.%T.aa8
// CHECK:STDOUT:   %pattern_type.loc243_29 => constants.%pattern_type.8e3
// CHECK:STDOUT:   %T.as_type.loc243_62.1 => constants.%T.as_type.efd
// CHECK:STDOUT:   %pattern_type.loc243_60 => constants.%pattern_type.593823.2
// CHECK:STDOUT:   %pattern_type.loc243_66 => constants.%pattern_type.51d1c4.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ModAssignWith.loc249(constants.%U.67d, constants.%T.d35) {
// CHECK:STDOUT:   %U.loc249_19.1 => constants.%U.67d
// CHECK:STDOUT:   %ModAssignWith.type.loc249_52.1 => constants.%ModAssignWith.type.c65b77.2
// CHECK:STDOUT:   %T.loc249_29.1 => constants.%T.d35
// CHECK:STDOUT:   %pattern_type.loc249_29 => constants.%pattern_type.26a
// CHECK:STDOUT:   %T.as_type.loc249_62.1 => constants.%T.as_type.4c2
// CHECK:STDOUT:   %pattern_type.loc249_60 => constants.%pattern_type.6ec6ce.2
// CHECK:STDOUT:   %pattern_type.loc249_66 => constants.%pattern_type.51d1c4.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AddAssignWith.loc225(constants.%i16, constants.%AddAssignWith.facet.23f) {
// CHECK:STDOUT:   %U.loc225_19.1 => constants.%i16
// CHECK:STDOUT:   %AddAssignWith.type.loc225_52.1 => constants.%AddAssignWith.type.704
// CHECK:STDOUT:   %T.loc225_29.1 => constants.%AddAssignWith.facet.23f
// CHECK:STDOUT:   %pattern_type.loc225_29 => constants.%pattern_type.139
// CHECK:STDOUT:   %T.as_type.loc225_62.1 => constants.%Int16
// CHECK:STDOUT:   %pattern_type.loc225_60 => constants.%pattern_type.9a4
// CHECK:STDOUT:   %pattern_type.loc225_66 => constants.%pattern_type.2f8
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc225_55 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc225_66 => constants.%complete_type.2da
// CHECK:STDOUT:   %require_complete.loc227 => constants.%complete_type.901
// CHECK:STDOUT:   %AddAssignWith.assoc_type => constants.%AddAssignWith.assoc_type.23b
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.08d
// CHECK:STDOUT:   %AddAssignWith.WithSelf.Op.type => constants.%AddAssignWith.WithSelf.Op.type.2c7
// CHECK:STDOUT:   %.loc227_5.2 => constants.%.122
// CHECK:STDOUT:   %AddAssignWith.lookup_impl_witness => constants.%custom_witness.6da
// CHECK:STDOUT:   %impl.elem0.loc227_5.2 => constants.%Int16.Op.c3e49c.13
// CHECK:STDOUT:   %specific_impl_fn.loc227_5.2 => constants.%Int16.Op.c3e49c.13
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @SubAssignWith.loc231(constants.%i16, constants.%SubAssignWith.facet.3e7) {
// CHECK:STDOUT:   %U.loc231_19.1 => constants.%i16
// CHECK:STDOUT:   %SubAssignWith.type.loc231_52.1 => constants.%SubAssignWith.type.fed
// CHECK:STDOUT:   %T.loc231_29.1 => constants.%SubAssignWith.facet.3e7
// CHECK:STDOUT:   %pattern_type.loc231_29 => constants.%pattern_type.912
// CHECK:STDOUT:   %T.as_type.loc231_62.1 => constants.%Int16
// CHECK:STDOUT:   %pattern_type.loc231_60 => constants.%pattern_type.9a4
// CHECK:STDOUT:   %pattern_type.loc231_66 => constants.%pattern_type.2f8
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc231_55 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc231_66 => constants.%complete_type.2da
// CHECK:STDOUT:   %require_complete.loc233 => constants.%complete_type.f45
// CHECK:STDOUT:   %SubAssignWith.assoc_type => constants.%SubAssignWith.assoc_type.81a
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.0a4
// CHECK:STDOUT:   %SubAssignWith.WithSelf.Op.type => constants.%SubAssignWith.WithSelf.Op.type.71a
// CHECK:STDOUT:   %.loc233_5.2 => constants.%.697
// CHECK:STDOUT:   %SubAssignWith.lookup_impl_witness => constants.%custom_witness.509
// CHECK:STDOUT:   %impl.elem0.loc233_5.2 => constants.%Int16.Op.c3e49c.14
// CHECK:STDOUT:   %specific_impl_fn.loc233_5.2 => constants.%Int16.Op.c3e49c.14
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @MulAssignWith.loc237(constants.%i16, constants.%MulAssignWith.facet.78f) {
// CHECK:STDOUT:   %U.loc237_19.1 => constants.%i16
// CHECK:STDOUT:   %MulAssignWith.type.loc237_52.1 => constants.%MulAssignWith.type.6c0
// CHECK:STDOUT:   %T.loc237_29.1 => constants.%MulAssignWith.facet.78f
// CHECK:STDOUT:   %pattern_type.loc237_29 => constants.%pattern_type.578
// CHECK:STDOUT:   %T.as_type.loc237_62.1 => constants.%Int16
// CHECK:STDOUT:   %pattern_type.loc237_60 => constants.%pattern_type.9a4
// CHECK:STDOUT:   %pattern_type.loc237_66 => constants.%pattern_type.2f8
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc237_55 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc237_66 => constants.%complete_type.2da
// CHECK:STDOUT:   %require_complete.loc239 => constants.%complete_type.a71
// CHECK:STDOUT:   %MulAssignWith.assoc_type => constants.%MulAssignWith.assoc_type.af2
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.e50
// CHECK:STDOUT:   %MulAssignWith.WithSelf.Op.type => constants.%MulAssignWith.WithSelf.Op.type.dca
// CHECK:STDOUT:   %.loc239_5.2 => constants.%.bb2
// CHECK:STDOUT:   %MulAssignWith.lookup_impl_witness => constants.%custom_witness.a1f
// CHECK:STDOUT:   %impl.elem0.loc239_5.2 => constants.%Int16.Op.c3e49c.15
// CHECK:STDOUT:   %specific_impl_fn.loc239_5.2 => constants.%Int16.Op.c3e49c.15
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @DivAssignWith.loc243(constants.%i16, constants.%DivAssignWith.facet.b0d) {
// CHECK:STDOUT:   %U.loc243_19.1 => constants.%i16
// CHECK:STDOUT:   %DivAssignWith.type.loc243_52.1 => constants.%DivAssignWith.type.527
// CHECK:STDOUT:   %T.loc243_29.1 => constants.%DivAssignWith.facet.b0d
// CHECK:STDOUT:   %pattern_type.loc243_29 => constants.%pattern_type.5f7
// CHECK:STDOUT:   %T.as_type.loc243_62.1 => constants.%Int16
// CHECK:STDOUT:   %pattern_type.loc243_60 => constants.%pattern_type.9a4
// CHECK:STDOUT:   %pattern_type.loc243_66 => constants.%pattern_type.2f8
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc243_55 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc243_66 => constants.%complete_type.2da
// CHECK:STDOUT:   %require_complete.loc245 => constants.%complete_type.620
// CHECK:STDOUT:   %DivAssignWith.assoc_type => constants.%DivAssignWith.assoc_type.5b8
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.ced
// CHECK:STDOUT:   %DivAssignWith.WithSelf.Op.type => constants.%DivAssignWith.WithSelf.Op.type.76f
// CHECK:STDOUT:   %.loc245_5.2 => constants.%.32e
// CHECK:STDOUT:   %DivAssignWith.lookup_impl_witness => constants.%custom_witness.deb
// CHECK:STDOUT:   %impl.elem0.loc245_5.2 => constants.%Int16.Op.c3e49c.16
// CHECK:STDOUT:   %specific_impl_fn.loc245_5.2 => constants.%Int16.Op.c3e49c.16
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ModAssignWith.loc249(constants.%i16, constants.%ModAssignWith.facet.ada) {
// CHECK:STDOUT:   %U.loc249_19.1 => constants.%i16
// CHECK:STDOUT:   %ModAssignWith.type.loc249_52.1 => constants.%ModAssignWith.type.ead
// CHECK:STDOUT:   %T.loc249_29.1 => constants.%ModAssignWith.facet.ada
// CHECK:STDOUT:   %pattern_type.loc249_29 => constants.%pattern_type.bf6
// CHECK:STDOUT:   %T.as_type.loc249_62.1 => constants.%Int16
// CHECK:STDOUT:   %pattern_type.loc249_60 => constants.%pattern_type.9a4
// CHECK:STDOUT:   %pattern_type.loc249_66 => constants.%pattern_type.2f8
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc249_55 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc249_66 => constants.%complete_type.2da
// CHECK:STDOUT:   %require_complete.loc251 => constants.%complete_type.422
// CHECK:STDOUT:   %ModAssignWith.assoc_type => constants.%ModAssignWith.assoc_type.556
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.b4c
// CHECK:STDOUT:   %ModAssignWith.WithSelf.Op.type => constants.%ModAssignWith.WithSelf.Op.type.bea
// CHECK:STDOUT:   %.loc251_5.2 => constants.%.51a
// CHECK:STDOUT:   %ModAssignWith.lookup_impl_witness => constants.%custom_witness.c26
// CHECK:STDOUT:   %impl.elem0.loc251_5.2 => constants.%Int16.Op.c3e49c.17
// CHECK:STDOUT:   %specific_impl_fn.loc251_5.2 => constants.%Int16.Op.c3e49c.17
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AddAssignWith.loc225(constants.%Int32, constants.%AddAssignWith.facet.d85) {
// CHECK:STDOUT:   %U.loc225_19.1 => constants.%Int32
// CHECK:STDOUT:   %AddAssignWith.type.loc225_52.1 => constants.%AddAssignWith.type.a71
// CHECK:STDOUT:   %T.loc225_29.1 => constants.%AddAssignWith.facet.d85
// CHECK:STDOUT:   %pattern_type.loc225_29 => constants.%pattern_type.736
// CHECK:STDOUT:   %T.as_type.loc225_62.1 => constants.%Int32
// CHECK:STDOUT:   %pattern_type.loc225_60 => constants.%pattern_type.9c9
// CHECK:STDOUT:   %pattern_type.loc225_66 => constants.%pattern_type.9c9
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc225_55 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc225_66 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc227 => constants.%complete_type.2ec
// CHECK:STDOUT:   %AddAssignWith.assoc_type => constants.%AddAssignWith.assoc_type.b4b
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.e09
// CHECK:STDOUT:   %AddAssignWith.WithSelf.Op.type => constants.%AddAssignWith.WithSelf.Op.type.181
// CHECK:STDOUT:   %.loc227_5.2 => constants.%.e44
// CHECK:STDOUT:   %AddAssignWith.lookup_impl_witness => constants.%custom_witness.c02
// CHECK:STDOUT:   %impl.elem0.loc227_5.2 => constants.%Int32.Op.6afff1.12
// CHECK:STDOUT:   %specific_impl_fn.loc227_5.2 => constants.%Int32.Op.6afff1.12
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @SubAssignWith.loc231(constants.%Int32, constants.%SubAssignWith.facet.187) {
// CHECK:STDOUT:   %U.loc231_19.1 => constants.%Int32
// CHECK:STDOUT:   %SubAssignWith.type.loc231_52.1 => constants.%SubAssignWith.type.35f
// CHECK:STDOUT:   %T.loc231_29.1 => constants.%SubAssignWith.facet.187
// CHECK:STDOUT:   %pattern_type.loc231_29 => constants.%pattern_type.870
// CHECK:STDOUT:   %T.as_type.loc231_62.1 => constants.%Int32
// CHECK:STDOUT:   %pattern_type.loc231_60 => constants.%pattern_type.9c9
// CHECK:STDOUT:   %pattern_type.loc231_66 => constants.%pattern_type.9c9
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc231_55 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc231_66 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc233 => constants.%complete_type.1ca
// CHECK:STDOUT:   %SubAssignWith.assoc_type => constants.%SubAssignWith.assoc_type.867
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.aa5
// CHECK:STDOUT:   %SubAssignWith.WithSelf.Op.type => constants.%SubAssignWith.WithSelf.Op.type.e51
// CHECK:STDOUT:   %.loc233_5.2 => constants.%.34b
// CHECK:STDOUT:   %SubAssignWith.lookup_impl_witness => constants.%custom_witness.062
// CHECK:STDOUT:   %impl.elem0.loc233_5.2 => constants.%Int32.Op.6afff1.14
// CHECK:STDOUT:   %specific_impl_fn.loc233_5.2 => constants.%Int32.Op.6afff1.14
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @MulAssignWith.loc237(constants.%Int32, constants.%MulAssignWith.facet.775) {
// CHECK:STDOUT:   %U.loc237_19.1 => constants.%Int32
// CHECK:STDOUT:   %MulAssignWith.type.loc237_52.1 => constants.%MulAssignWith.type.dd2
// CHECK:STDOUT:   %T.loc237_29.1 => constants.%MulAssignWith.facet.775
// CHECK:STDOUT:   %pattern_type.loc237_29 => constants.%pattern_type.e94
// CHECK:STDOUT:   %T.as_type.loc237_62.1 => constants.%Int32
// CHECK:STDOUT:   %pattern_type.loc237_60 => constants.%pattern_type.9c9
// CHECK:STDOUT:   %pattern_type.loc237_66 => constants.%pattern_type.9c9
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc237_55 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc237_66 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc239 => constants.%complete_type.a19
// CHECK:STDOUT:   %MulAssignWith.assoc_type => constants.%MulAssignWith.assoc_type.5c3
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.6d0
// CHECK:STDOUT:   %MulAssignWith.WithSelf.Op.type => constants.%MulAssignWith.WithSelf.Op.type.89d
// CHECK:STDOUT:   %.loc239_5.2 => constants.%.263
// CHECK:STDOUT:   %MulAssignWith.lookup_impl_witness => constants.%custom_witness.03af
// CHECK:STDOUT:   %impl.elem0.loc239_5.2 => constants.%Int32.Op.6afff1.15
// CHECK:STDOUT:   %specific_impl_fn.loc239_5.2 => constants.%Int32.Op.6afff1.15
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @DivAssignWith.loc243(constants.%Int32, constants.%DivAssignWith.facet.f5b) {
// CHECK:STDOUT:   %U.loc243_19.1 => constants.%Int32
// CHECK:STDOUT:   %DivAssignWith.type.loc243_52.1 => constants.%DivAssignWith.type.a38
// CHECK:STDOUT:   %T.loc243_29.1 => constants.%DivAssignWith.facet.f5b
// CHECK:STDOUT:   %pattern_type.loc243_29 => constants.%pattern_type.689
// CHECK:STDOUT:   %T.as_type.loc243_62.1 => constants.%Int32
// CHECK:STDOUT:   %pattern_type.loc243_60 => constants.%pattern_type.9c9
// CHECK:STDOUT:   %pattern_type.loc243_66 => constants.%pattern_type.9c9
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc243_55 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc243_66 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc245 => constants.%complete_type.136
// CHECK:STDOUT:   %DivAssignWith.assoc_type => constants.%DivAssignWith.assoc_type.86d
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.b6e
// CHECK:STDOUT:   %DivAssignWith.WithSelf.Op.type => constants.%DivAssignWith.WithSelf.Op.type.593
// CHECK:STDOUT:   %.loc245_5.2 => constants.%.c99
// CHECK:STDOUT:   %DivAssignWith.lookup_impl_witness => constants.%custom_witness.be6
// CHECK:STDOUT:   %impl.elem0.loc245_5.2 => constants.%Int32.Op.6afff1.16
// CHECK:STDOUT:   %specific_impl_fn.loc245_5.2 => constants.%Int32.Op.6afff1.16
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ModAssignWith.loc249(constants.%Int32, constants.%ModAssignWith.facet.b82) {
// CHECK:STDOUT:   %U.loc249_19.1 => constants.%Int32
// CHECK:STDOUT:   %ModAssignWith.type.loc249_52.1 => constants.%ModAssignWith.type.36f
// CHECK:STDOUT:   %T.loc249_29.1 => constants.%ModAssignWith.facet.b82
// CHECK:STDOUT:   %pattern_type.loc249_29 => constants.%pattern_type.ebe
// CHECK:STDOUT:   %T.as_type.loc249_62.1 => constants.%Int32
// CHECK:STDOUT:   %pattern_type.loc249_60 => constants.%pattern_type.9c9
// CHECK:STDOUT:   %pattern_type.loc249_66 => constants.%pattern_type.9c9
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc249_55 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc249_66 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc251 => constants.%complete_type.ca8
// CHECK:STDOUT:   %ModAssignWith.assoc_type => constants.%ModAssignWith.assoc_type.1a5
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.609
// CHECK:STDOUT:   %ModAssignWith.WithSelf.Op.type => constants.%ModAssignWith.WithSelf.Op.type.f83
// CHECK:STDOUT:   %.loc251_5.2 => constants.%.146
// CHECK:STDOUT:   %ModAssignWith.lookup_impl_witness => constants.%custom_witness.203
// CHECK:STDOUT:   %impl.elem0.loc251_5.2 => constants.%Int32.Op.6afff1.17
// CHECK:STDOUT:   %specific_impl_fn.loc251_5.2 => constants.%Int32.Op.6afff1.17
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AddAssignWith.loc225(constants.%Int32, constants.%AddAssignWith.facet.b70) {
// CHECK:STDOUT:   %U.loc225_19.1 => constants.%Int32
// CHECK:STDOUT:   %AddAssignWith.type.loc225_52.1 => constants.%AddAssignWith.type.a71
// CHECK:STDOUT:   %T.loc225_29.1 => constants.%AddAssignWith.facet.b70
// CHECK:STDOUT:   %pattern_type.loc225_29 => constants.%pattern_type.736
// CHECK:STDOUT:   %T.as_type.loc225_62.1 => constants.%Int64
// CHECK:STDOUT:   %pattern_type.loc225_60 => constants.%pattern_type.f42
// CHECK:STDOUT:   %pattern_type.loc225_66 => constants.%pattern_type.9c9
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc225_55 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc225_66 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc227 => constants.%complete_type.2ec
// CHECK:STDOUT:   %AddAssignWith.assoc_type => constants.%AddAssignWith.assoc_type.b4b
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.e09
// CHECK:STDOUT:   %AddAssignWith.WithSelf.Op.type => constants.%AddAssignWith.WithSelf.Op.type.750
// CHECK:STDOUT:   %.loc227_5.2 => constants.%.5d8
// CHECK:STDOUT:   %AddAssignWith.lookup_impl_witness => constants.%custom_witness.3a4
// CHECK:STDOUT:   %impl.elem0.loc227_5.2 => constants.%Int64.Op.b941da.2
// CHECK:STDOUT:   %specific_impl_fn.loc227_5.2 => constants.%Int64.Op.b941da.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @SubAssignWith.loc231(constants.%Int32, constants.%SubAssignWith.facet.bd0) {
// CHECK:STDOUT:   %U.loc231_19.1 => constants.%Int32
// CHECK:STDOUT:   %SubAssignWith.type.loc231_52.1 => constants.%SubAssignWith.type.35f
// CHECK:STDOUT:   %T.loc231_29.1 => constants.%SubAssignWith.facet.bd0
// CHECK:STDOUT:   %pattern_type.loc231_29 => constants.%pattern_type.870
// CHECK:STDOUT:   %T.as_type.loc231_62.1 => constants.%Int64
// CHECK:STDOUT:   %pattern_type.loc231_60 => constants.%pattern_type.f42
// CHECK:STDOUT:   %pattern_type.loc231_66 => constants.%pattern_type.9c9
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc231_55 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc231_66 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc233 => constants.%complete_type.1ca
// CHECK:STDOUT:   %SubAssignWith.assoc_type => constants.%SubAssignWith.assoc_type.867
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.aa5
// CHECK:STDOUT:   %SubAssignWith.WithSelf.Op.type => constants.%SubAssignWith.WithSelf.Op.type.792
// CHECK:STDOUT:   %.loc233_5.2 => constants.%.827
// CHECK:STDOUT:   %SubAssignWith.lookup_impl_witness => constants.%custom_witness.de3
// CHECK:STDOUT:   %impl.elem0.loc233_5.2 => constants.%Int64.Op.b941da.4
// CHECK:STDOUT:   %specific_impl_fn.loc233_5.2 => constants.%Int64.Op.b941da.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @MulAssignWith.loc237(constants.%Int32, constants.%MulAssignWith.facet.982) {
// CHECK:STDOUT:   %U.loc237_19.1 => constants.%Int32
// CHECK:STDOUT:   %MulAssignWith.type.loc237_52.1 => constants.%MulAssignWith.type.dd2
// CHECK:STDOUT:   %T.loc237_29.1 => constants.%MulAssignWith.facet.982
// CHECK:STDOUT:   %pattern_type.loc237_29 => constants.%pattern_type.e94
// CHECK:STDOUT:   %T.as_type.loc237_62.1 => constants.%Int64
// CHECK:STDOUT:   %pattern_type.loc237_60 => constants.%pattern_type.f42
// CHECK:STDOUT:   %pattern_type.loc237_66 => constants.%pattern_type.9c9
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc237_55 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc237_66 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc239 => constants.%complete_type.a19
// CHECK:STDOUT:   %MulAssignWith.assoc_type => constants.%MulAssignWith.assoc_type.5c3
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.6d0
// CHECK:STDOUT:   %MulAssignWith.WithSelf.Op.type => constants.%MulAssignWith.WithSelf.Op.type.788
// CHECK:STDOUT:   %.loc239_5.2 => constants.%.4a3
// CHECK:STDOUT:   %MulAssignWith.lookup_impl_witness => constants.%custom_witness.917
// CHECK:STDOUT:   %impl.elem0.loc239_5.2 => constants.%Int64.Op.b941da.5
// CHECK:STDOUT:   %specific_impl_fn.loc239_5.2 => constants.%Int64.Op.b941da.5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @DivAssignWith.loc243(constants.%Int32, constants.%DivAssignWith.facet.d3b) {
// CHECK:STDOUT:   %U.loc243_19.1 => constants.%Int32
// CHECK:STDOUT:   %DivAssignWith.type.loc243_52.1 => constants.%DivAssignWith.type.a38
// CHECK:STDOUT:   %T.loc243_29.1 => constants.%DivAssignWith.facet.d3b
// CHECK:STDOUT:   %pattern_type.loc243_29 => constants.%pattern_type.689
// CHECK:STDOUT:   %T.as_type.loc243_62.1 => constants.%Int64
// CHECK:STDOUT:   %pattern_type.loc243_60 => constants.%pattern_type.f42
// CHECK:STDOUT:   %pattern_type.loc243_66 => constants.%pattern_type.9c9
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc243_55 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc243_66 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc245 => constants.%complete_type.136
// CHECK:STDOUT:   %DivAssignWith.assoc_type => constants.%DivAssignWith.assoc_type.86d
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.b6e
// CHECK:STDOUT:   %DivAssignWith.WithSelf.Op.type => constants.%DivAssignWith.WithSelf.Op.type.646
// CHECK:STDOUT:   %.loc245_5.2 => constants.%.384
// CHECK:STDOUT:   %DivAssignWith.lookup_impl_witness => constants.%custom_witness.f47
// CHECK:STDOUT:   %impl.elem0.loc245_5.2 => constants.%Int64.Op.b941da.6
// CHECK:STDOUT:   %specific_impl_fn.loc245_5.2 => constants.%Int64.Op.b941da.6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ModAssignWith.loc249(constants.%Int32, constants.%ModAssignWith.facet.1c6) {
// CHECK:STDOUT:   %U.loc249_19.1 => constants.%Int32
// CHECK:STDOUT:   %ModAssignWith.type.loc249_52.1 => constants.%ModAssignWith.type.36f
// CHECK:STDOUT:   %T.loc249_29.1 => constants.%ModAssignWith.facet.1c6
// CHECK:STDOUT:   %pattern_type.loc249_29 => constants.%pattern_type.ebe
// CHECK:STDOUT:   %T.as_type.loc249_62.1 => constants.%Int64
// CHECK:STDOUT:   %pattern_type.loc249_60 => constants.%pattern_type.f42
// CHECK:STDOUT:   %pattern_type.loc249_66 => constants.%pattern_type.9c9
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc249_55 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc249_66 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc251 => constants.%complete_type.ca8
// CHECK:STDOUT:   %ModAssignWith.assoc_type => constants.%ModAssignWith.assoc_type.1a5
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.609
// CHECK:STDOUT:   %ModAssignWith.WithSelf.Op.type => constants.%ModAssignWith.WithSelf.Op.type.a34
// CHECK:STDOUT:   %.loc251_5.2 => constants.%.ee0
// CHECK:STDOUT:   %ModAssignWith.lookup_impl_witness => constants.%custom_witness.ad7
// CHECK:STDOUT:   %impl.elem0.loc251_5.2 => constants.%Int64.Op.b941da.7
// CHECK:STDOUT:   %specific_impl_fn.loc251_5.2 => constants.%Int64.Op.b941da.7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestInc(constants.%T.b7e) {
// CHECK:STDOUT:   %T.loc278_13.1 => constants.%T.b7e
// CHECK:STDOUT:   %T.as_type.loc278_33.1 => constants.%T.as_type.3f5
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.fb5eac.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestDec(constants.%T.ce1) {
// CHECK:STDOUT:   %T.loc284_13.1 => constants.%T.ce1
// CHECK:STDOUT:   %T.as_type.loc284_33.1 => constants.%T.as_type.ead
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.f51ad8.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestNegate(constants.%Result, constants.%T.abc) {
// CHECK:STDOUT:   %Result.loc292_9.1 => constants.%Result
// CHECK:STDOUT:   %Result.as_type.loc293_50.1 => constants.%Result.as_type
// CHECK:STDOUT:   %facet_type => constants.%facet_type.0fc
// CHECK:STDOUT:   %T.loc293_4.1 => constants.%T.abc
// CHECK:STDOUT:   %pattern_type.loc293 => constants.%pattern_type.5c4
// CHECK:STDOUT:   %pattern_type.loc294_11 => constants.%pattern_type.eca0e4.2
// CHECK:STDOUT:   %T.as_type.loc294_24.1 => constants.%T.as_type.c82
// CHECK:STDOUT:   %pattern_type.loc294_22 => constants.%pattern_type.ece
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestInc(constants.%Inc.facet.390) {
// CHECK:STDOUT:   %T.loc278_13.1 => constants.%Inc.facet.390
// CHECK:STDOUT:   %T.as_type.loc278_33.1 => constants.%Int32
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.9c9
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.357
// CHECK:STDOUT:   %Inc.WithSelf.Op.type => constants.%Inc.WithSelf.Op.type.eaa
// CHECK:STDOUT:   %.loc280 => constants.%.696
// CHECK:STDOUT:   %Inc.lookup_impl_witness => constants.%custom_witness.934
// CHECK:STDOUT:   %impl.elem0.loc280_3.2 => constants.%Int32.cpp_operator.0010d3.16
// CHECK:STDOUT:   %specific_impl_fn.loc280_3.2 => constants.%Int32.cpp_operator.0010d3.16
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestDec(constants.%Dec.facet.961) {
// CHECK:STDOUT:   %T.loc284_13.1 => constants.%Dec.facet.961
// CHECK:STDOUT:   %T.as_type.loc284_33.1 => constants.%Int32
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.9c9
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.357
// CHECK:STDOUT:   %Dec.WithSelf.Op.type => constants.%Dec.WithSelf.Op.type.6fb
// CHECK:STDOUT:   %.loc286 => constants.%.cdd
// CHECK:STDOUT:   %Dec.lookup_impl_witness => constants.%custom_witness.99f
// CHECK:STDOUT:   %impl.elem0.loc286_3.2 => constants.%Int32.cpp_operator.0010d3.17
// CHECK:STDOUT:   %specific_impl_fn.loc286_3.2 => constants.%Int32.cpp_operator.0010d3.17
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestNegate(constants.%Destroy.facet.b0a, constants.%facet_value.70a) {
// CHECK:STDOUT:   %Result.loc292_9.1 => constants.%Destroy.facet.b0a
// CHECK:STDOUT:   %Result.as_type.loc293_50.1 => constants.%Int64
// CHECK:STDOUT:   %facet_type => constants.%facet_type.89a
// CHECK:STDOUT:   %T.loc293_4.1 => constants.%facet_value.70a
// CHECK:STDOUT:   %pattern_type.loc293 => constants.%pattern_type.c49
// CHECK:STDOUT:   %pattern_type.loc294_11 => constants.%pattern_type.f42
// CHECK:STDOUT:   %T.as_type.loc294_24.1 => constants.%Int16
// CHECK:STDOUT:   %pattern_type.loc294_22 => constants.%pattern_type.9a4
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc294_11 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc294_22 => constants.%complete_type.357
// CHECK:STDOUT:   %Negate.lookup_impl_witness => constants.%custom_witness.e58
// CHECK:STDOUT:   %Negate.facet => constants.%Negate.facet.0a5
// CHECK:STDOUT:   %Negate.WithSelf.Op.type => constants.%Negate.WithSelf.Op.type.d7d
// CHECK:STDOUT:   %.loc296_3.3 => constants.%.3e7
// CHECK:STDOUT:   %impl.elem1.loc296_3.2 => constants.%Int16.Op.c3e49c.18
// CHECK:STDOUT:   %specific_impl_fn.loc296_3.3 => constants.%Int16.Op.c3e49c.18
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.f16
// CHECK:STDOUT:   %.loc296_3.4 => constants.%.db1
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.d7e
// CHECK:STDOUT:   %impl.elem0.loc296_3.2 => constants.%Int64.Op.b941da.1
// CHECK:STDOUT:   %specific_impl_fn.loc296_3.4 => constants.%Int64.Op.b941da.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestNegate(constants.%Destroy.facet.c06, constants.%facet_value.396) {
// CHECK:STDOUT:   %Result.loc292_9.1 => constants.%Destroy.facet.c06
// CHECK:STDOUT:   %Result.as_type.loc293_50.1 => constants.%Int32
// CHECK:STDOUT:   %facet_type => constants.%facet_type.d5d
// CHECK:STDOUT:   %T.loc293_4.1 => constants.%facet_value.396
// CHECK:STDOUT:   %pattern_type.loc293 => constants.%pattern_type.0ec
// CHECK:STDOUT:   %pattern_type.loc294_11 => constants.%pattern_type.9c9
// CHECK:STDOUT:   %T.as_type.loc294_24.1 => constants.%Int32
// CHECK:STDOUT:   %pattern_type.loc294_22 => constants.%pattern_type.9c9
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc294_11 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc294_22 => constants.%complete_type.357
// CHECK:STDOUT:   %Negate.lookup_impl_witness => constants.%custom_witness.e8a
// CHECK:STDOUT:   %Negate.facet => constants.%Negate.facet.3c9
// CHECK:STDOUT:   %Negate.WithSelf.Op.type => constants.%Negate.WithSelf.Op.type.1bc
// CHECK:STDOUT:   %.loc296_3.3 => constants.%.19b
// CHECK:STDOUT:   %impl.elem1.loc296_3.2 => constants.%Int32.Op.6afff1.18
// CHECK:STDOUT:   %specific_impl_fn.loc296_3.3 => constants.%Int32.Op.6afff1.18
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.2e1
// CHECK:STDOUT:   %.loc296_3.4 => constants.%.e6c
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.ce1
// CHECK:STDOUT:   %impl.elem0.loc296_3.2 => constants.%Int32.Op.6afff1.1
// CHECK:STDOUT:   %specific_impl_fn.loc296_3.4 => constants.%Int32.Op.6afff1.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
