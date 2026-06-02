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
// CHECK:STDOUT:   %U.99e: %Destroy.type = symbolic_binding U, 1 [symbolic]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Other: type = symbolic_binding Other, 0 [symbolic]
// CHECK:STDOUT:   %AddWith.type.893: type = facet_type <@AddWith.1, @AddWith.1(%Other)> [symbolic]
// CHECK:STDOUT:   %Self.66b: %AddWith.type.893 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %AddWith.assoc_type.5da: type = assoc_entity_type @AddWith.1, @AddWith.1(%Other) [symbolic]
// CHECK:STDOUT:   %AddWith.WithSelf.Op.type.c45: type = fn_type @AddWith.WithSelf.Op, @AddWith.WithSelf(%Other, %Self.66b) [symbolic]
// CHECK:STDOUT:   %AddWith.WithSelf.Op.2f3: %AddWith.WithSelf.Op.type.c45 = struct_value () [symbolic]
// CHECK:STDOUT:   %U.as_type.f7f: type = facet_access_type %U.99e [symbolic]
// CHECK:STDOUT:   %AddWith.type.ad2: type = facet_type <@AddWith.1, @AddWith.1(%U.as_type.f7f)> [symbolic]
// CHECK:STDOUT:   %facet_type.4ec: type = facet_type <@Destroy & @AddWith.1, @AddWith.1(%U.as_type.f7f)> [symbolic]
// CHECK:STDOUT:   %.Self.53a: %facet_type.4ec = symbolic_binding .Self [symbolic]
// CHECK:STDOUT:   %AddWith.assoc_type.c6c: type = assoc_entity_type @AddWith.1, @AddWith.1(%U.as_type.f7f) [symbolic]
// CHECK:STDOUT:   %assoc0.35c: %AddWith.assoc_type.c6c = assoc_entity element0, imports.%Core.import_ref.1dd [symbolic]
// CHECK:STDOUT:   %assoc1.d93: %AddWith.assoc_type.c6c = assoc_entity element1, imports.%Core.import_ref.29ff [symbolic]
// CHECK:STDOUT:   %require_complete.3ac: <witness> = require_complete_type %facet_type.4ec [symbolic]
// CHECK:STDOUT:   %.Self.as_type.626: type = facet_access_type %.Self.53a [symbolic]
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.46e: <witness> = lookup_impl_witness %.Self.53a, @AddWith.1, @AddWith.1(%U.as_type.f7f) [symbolic]
// CHECK:STDOUT:   %impl.elem0.851: type = impl_witness_access %AddWith.lookup_impl_witness.46e, element0 [symbolic]
// CHECK:STDOUT:   %Result.as_type: type = facet_access_type %Result [symbolic]
// CHECK:STDOUT:   %facet_type.cac: type = facet_type <@Destroy & @AddWith.1, @AddWith.1(%U.as_type.f7f) where %impl.elem0.851 = %Result.as_type> [symbolic]
// CHECK:STDOUT:   %pattern_type.bcc: type = pattern_type %facet_type.cac [symbolic]
// CHECK:STDOUT:   %T.23b: %facet_type.cac = symbolic_binding T, 2 [symbolic]
// CHECK:STDOUT:   %pattern_type.227a41.2: type = pattern_type %Result.as_type [symbolic]
// CHECK:STDOUT:   %T.as_type.46a: type = facet_access_type %T.23b [symbolic]
// CHECK:STDOUT:   %pattern_type.d37: type = pattern_type %T.as_type.46a [symbolic]
// CHECK:STDOUT:   %pattern_type.87c: type = pattern_type %U.as_type.f7f [symbolic]
// CHECK:STDOUT:   %require_complete.3dc: <witness> = require_complete_type %AddWith.type.ad2 [symbolic]
// CHECK:STDOUT:   %assoc1.fdf: %AddWith.assoc_type.5da = assoc_entity element1, imports.%Core.import_ref.a1e [symbolic]
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.e7e: <witness> = lookup_impl_witness %T.23b, @AddWith.1, @AddWith.1(%U.as_type.f7f) [symbolic]
// CHECK:STDOUT:   %AddWith.facet.c8e: %AddWith.type.ad2 = facet_value %T.as_type.46a, (%AddWith.lookup_impl_witness.e7e) [symbolic]
// CHECK:STDOUT:   %AddWith.WithSelf.Op.type.12d: type = fn_type @AddWith.WithSelf.Op, @AddWith.WithSelf(%U.as_type.f7f, %AddWith.facet.c8e) [symbolic]
// CHECK:STDOUT:   %.ac7: type = fn_type_with_self_type %AddWith.WithSelf.Op.type.12d, %AddWith.facet.c8e [symbolic]
// CHECK:STDOUT:   %impl.elem1.f9d: %.ac7 = impl_witness_access %AddWith.lookup_impl_witness.e7e, element1 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.8f2: <specific function> = specific_impl_function %impl.elem1.f9d, @AddWith.WithSelf.Op(%U.as_type.f7f, %AddWith.facet.c8e) [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %Result, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.6304ea.2: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Result) [symbolic]
// CHECK:STDOUT:   %.543: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.6304ea.2, %Result [symbolic]
// CHECK:STDOUT:   %impl.elem0.531: %.543 = impl_witness_access %Destroy.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.010: <specific function> = specific_impl_function %impl.elem0.531, @Destroy.WithSelf.Op(%Result) [symbolic]
// CHECK:STDOUT:   %SubWith.type.92e: type = facet_type <@SubWith.1, @SubWith.1(%Other)> [symbolic]
// CHECK:STDOUT:   %Self.cc1: %SubWith.type.92e = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %SubWith.assoc_type.72b: type = assoc_entity_type @SubWith.1, @SubWith.1(%Other) [symbolic]
// CHECK:STDOUT:   %SubWith.WithSelf.Op.type.7da: type = fn_type @SubWith.WithSelf.Op, @SubWith.WithSelf(%Other, %Self.cc1) [symbolic]
// CHECK:STDOUT:   %SubWith.WithSelf.Op.bd5: %SubWith.WithSelf.Op.type.7da = struct_value () [symbolic]
// CHECK:STDOUT:   %SubWith.type.394: type = facet_type <@SubWith.1, @SubWith.1(%U.as_type.f7f)> [symbolic]
// CHECK:STDOUT:   %facet_type.c40: type = facet_type <@Destroy & @SubWith.1, @SubWith.1(%U.as_type.f7f)> [symbolic]
// CHECK:STDOUT:   %.Self.f02: %facet_type.c40 = symbolic_binding .Self [symbolic]
// CHECK:STDOUT:   %SubWith.assoc_type.767: type = assoc_entity_type @SubWith.1, @SubWith.1(%U.as_type.f7f) [symbolic]
// CHECK:STDOUT:   %assoc0.705: %SubWith.assoc_type.767 = assoc_entity element0, imports.%Core.import_ref.391 [symbolic]
// CHECK:STDOUT:   %assoc1.edb: %SubWith.assoc_type.767 = assoc_entity element1, imports.%Core.import_ref.940 [symbolic]
// CHECK:STDOUT:   %require_complete.f69: <witness> = require_complete_type %facet_type.c40 [symbolic]
// CHECK:STDOUT:   %.Self.as_type.dc7: type = facet_access_type %.Self.f02 [symbolic]
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.54d: <witness> = lookup_impl_witness %.Self.f02, @SubWith.1, @SubWith.1(%U.as_type.f7f) [symbolic]
// CHECK:STDOUT:   %impl.elem0.67e: type = impl_witness_access %SubWith.lookup_impl_witness.54d, element0 [symbolic]
// CHECK:STDOUT:   %facet_type.98a: type = facet_type <@Destroy & @SubWith.1, @SubWith.1(%U.as_type.f7f) where %impl.elem0.67e = %Result.as_type> [symbolic]
// CHECK:STDOUT:   %pattern_type.054: type = pattern_type %facet_type.98a [symbolic]
// CHECK:STDOUT:   %T.4a2: %facet_type.98a = symbolic_binding T, 2 [symbolic]
// CHECK:STDOUT:   %T.as_type.330: type = facet_access_type %T.4a2 [symbolic]
// CHECK:STDOUT:   %pattern_type.2b7: type = pattern_type %T.as_type.330 [symbolic]
// CHECK:STDOUT:   %require_complete.30c: <witness> = require_complete_type %SubWith.type.394 [symbolic]
// CHECK:STDOUT:   %assoc1.e6a: %SubWith.assoc_type.72b = assoc_entity element1, imports.%Core.import_ref.e2a [symbolic]
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.031: <witness> = lookup_impl_witness %T.4a2, @SubWith.1, @SubWith.1(%U.as_type.f7f) [symbolic]
// CHECK:STDOUT:   %SubWith.facet.2d1: %SubWith.type.394 = facet_value %T.as_type.330, (%SubWith.lookup_impl_witness.031) [symbolic]
// CHECK:STDOUT:   %SubWith.WithSelf.Op.type.496: type = fn_type @SubWith.WithSelf.Op, @SubWith.WithSelf(%U.as_type.f7f, %SubWith.facet.2d1) [symbolic]
// CHECK:STDOUT:   %.754: type = fn_type_with_self_type %SubWith.WithSelf.Op.type.496, %SubWith.facet.2d1 [symbolic]
// CHECK:STDOUT:   %impl.elem1.a03: %.754 = impl_witness_access %SubWith.lookup_impl_witness.031, element1 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.069: <specific function> = specific_impl_function %impl.elem1.a03, @SubWith.WithSelf.Op(%U.as_type.f7f, %SubWith.facet.2d1) [symbolic]
// CHECK:STDOUT:   %MulWith.type.98b: type = facet_type <@MulWith.1, @MulWith.1(%Other)> [symbolic]
// CHECK:STDOUT:   %Self.0e2: %MulWith.type.98b = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %MulWith.assoc_type.634: type = assoc_entity_type @MulWith.1, @MulWith.1(%Other) [symbolic]
// CHECK:STDOUT:   %MulWith.WithSelf.Op.type.309: type = fn_type @MulWith.WithSelf.Op, @MulWith.WithSelf(%Other, %Self.0e2) [symbolic]
// CHECK:STDOUT:   %MulWith.WithSelf.Op.56c: %MulWith.WithSelf.Op.type.309 = struct_value () [symbolic]
// CHECK:STDOUT:   %MulWith.type.195: type = facet_type <@MulWith.1, @MulWith.1(%U.as_type.f7f)> [symbolic]
// CHECK:STDOUT:   %facet_type.e45: type = facet_type <@Destroy & @MulWith.1, @MulWith.1(%U.as_type.f7f)> [symbolic]
// CHECK:STDOUT:   %.Self.117: %facet_type.e45 = symbolic_binding .Self [symbolic]
// CHECK:STDOUT:   %MulWith.assoc_type.fff: type = assoc_entity_type @MulWith.1, @MulWith.1(%U.as_type.f7f) [symbolic]
// CHECK:STDOUT:   %assoc0.dfc: %MulWith.assoc_type.fff = assoc_entity element0, imports.%Core.import_ref.41e [symbolic]
// CHECK:STDOUT:   %assoc1.bde: %MulWith.assoc_type.fff = assoc_entity element1, imports.%Core.import_ref.d23 [symbolic]
// CHECK:STDOUT:   %require_complete.aad: <witness> = require_complete_type %facet_type.e45 [symbolic]
// CHECK:STDOUT:   %.Self.as_type.91e: type = facet_access_type %.Self.117 [symbolic]
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.bd7: <witness> = lookup_impl_witness %.Self.117, @MulWith.1, @MulWith.1(%U.as_type.f7f) [symbolic]
// CHECK:STDOUT:   %impl.elem0.e5b: type = impl_witness_access %MulWith.lookup_impl_witness.bd7, element0 [symbolic]
// CHECK:STDOUT:   %facet_type.a59: type = facet_type <@Destroy & @MulWith.1, @MulWith.1(%U.as_type.f7f) where %impl.elem0.e5b = %Result.as_type> [symbolic]
// CHECK:STDOUT:   %pattern_type.4b6: type = pattern_type %facet_type.a59 [symbolic]
// CHECK:STDOUT:   %T.2c2: %facet_type.a59 = symbolic_binding T, 2 [symbolic]
// CHECK:STDOUT:   %T.as_type.198: type = facet_access_type %T.2c2 [symbolic]
// CHECK:STDOUT:   %pattern_type.2a1: type = pattern_type %T.as_type.198 [symbolic]
// CHECK:STDOUT:   %require_complete.01c: <witness> = require_complete_type %MulWith.type.195 [symbolic]
// CHECK:STDOUT:   %assoc1.14f: %MulWith.assoc_type.634 = assoc_entity element1, imports.%Core.import_ref.087 [symbolic]
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.e5c: <witness> = lookup_impl_witness %T.2c2, @MulWith.1, @MulWith.1(%U.as_type.f7f) [symbolic]
// CHECK:STDOUT:   %MulWith.facet.eb2: %MulWith.type.195 = facet_value %T.as_type.198, (%MulWith.lookup_impl_witness.e5c) [symbolic]
// CHECK:STDOUT:   %MulWith.WithSelf.Op.type.288: type = fn_type @MulWith.WithSelf.Op, @MulWith.WithSelf(%U.as_type.f7f, %MulWith.facet.eb2) [symbolic]
// CHECK:STDOUT:   %.a85: type = fn_type_with_self_type %MulWith.WithSelf.Op.type.288, %MulWith.facet.eb2 [symbolic]
// CHECK:STDOUT:   %impl.elem1.447: %.a85 = impl_witness_access %MulWith.lookup_impl_witness.e5c, element1 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.32c: <specific function> = specific_impl_function %impl.elem1.447, @MulWith.WithSelf.Op(%U.as_type.f7f, %MulWith.facet.eb2) [symbolic]
// CHECK:STDOUT:   %DivWith.type.177: type = facet_type <@DivWith.1, @DivWith.1(%Other)> [symbolic]
// CHECK:STDOUT:   %Self.7db: %DivWith.type.177 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %DivWith.assoc_type.5b4: type = assoc_entity_type @DivWith.1, @DivWith.1(%Other) [symbolic]
// CHECK:STDOUT:   %DivWith.WithSelf.Op.type.e6f: type = fn_type @DivWith.WithSelf.Op, @DivWith.WithSelf(%Other, %Self.7db) [symbolic]
// CHECK:STDOUT:   %DivWith.WithSelf.Op.908: %DivWith.WithSelf.Op.type.e6f = struct_value () [symbolic]
// CHECK:STDOUT:   %DivWith.type.63d: type = facet_type <@DivWith.1, @DivWith.1(%U.as_type.f7f)> [symbolic]
// CHECK:STDOUT:   %facet_type.511: type = facet_type <@Destroy & @DivWith.1, @DivWith.1(%U.as_type.f7f)> [symbolic]
// CHECK:STDOUT:   %.Self.10f: %facet_type.511 = symbolic_binding .Self [symbolic]
// CHECK:STDOUT:   %DivWith.assoc_type.e27: type = assoc_entity_type @DivWith.1, @DivWith.1(%U.as_type.f7f) [symbolic]
// CHECK:STDOUT:   %assoc0.46e: %DivWith.assoc_type.e27 = assoc_entity element0, imports.%Core.import_ref.e71d [symbolic]
// CHECK:STDOUT:   %assoc1.7f9: %DivWith.assoc_type.e27 = assoc_entity element1, imports.%Core.import_ref.e712 [symbolic]
// CHECK:STDOUT:   %require_complete.8fe: <witness> = require_complete_type %facet_type.511 [symbolic]
// CHECK:STDOUT:   %.Self.as_type.151: type = facet_access_type %.Self.10f [symbolic]
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.0c0: <witness> = lookup_impl_witness %.Self.10f, @DivWith.1, @DivWith.1(%U.as_type.f7f) [symbolic]
// CHECK:STDOUT:   %impl.elem0.2d7: type = impl_witness_access %DivWith.lookup_impl_witness.0c0, element0 [symbolic]
// CHECK:STDOUT:   %facet_type.5cc: type = facet_type <@Destroy & @DivWith.1, @DivWith.1(%U.as_type.f7f) where %impl.elem0.2d7 = %Result.as_type> [symbolic]
// CHECK:STDOUT:   %pattern_type.fb6: type = pattern_type %facet_type.5cc [symbolic]
// CHECK:STDOUT:   %T.2b9: %facet_type.5cc = symbolic_binding T, 2 [symbolic]
// CHECK:STDOUT:   %T.as_type.e49: type = facet_access_type %T.2b9 [symbolic]
// CHECK:STDOUT:   %pattern_type.673: type = pattern_type %T.as_type.e49 [symbolic]
// CHECK:STDOUT:   %require_complete.be3: <witness> = require_complete_type %DivWith.type.63d [symbolic]
// CHECK:STDOUT:   %assoc1.9374: %DivWith.assoc_type.5b4 = assoc_entity element1, imports.%Core.import_ref.993 [symbolic]
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.2e1: <witness> = lookup_impl_witness %T.2b9, @DivWith.1, @DivWith.1(%U.as_type.f7f) [symbolic]
// CHECK:STDOUT:   %DivWith.facet.b8f: %DivWith.type.63d = facet_value %T.as_type.e49, (%DivWith.lookup_impl_witness.2e1) [symbolic]
// CHECK:STDOUT:   %DivWith.WithSelf.Op.type.be6: type = fn_type @DivWith.WithSelf.Op, @DivWith.WithSelf(%U.as_type.f7f, %DivWith.facet.b8f) [symbolic]
// CHECK:STDOUT:   %.42b: type = fn_type_with_self_type %DivWith.WithSelf.Op.type.be6, %DivWith.facet.b8f [symbolic]
// CHECK:STDOUT:   %impl.elem1.6f6: %.42b = impl_witness_access %DivWith.lookup_impl_witness.2e1, element1 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.263: <specific function> = specific_impl_function %impl.elem1.6f6, @DivWith.WithSelf.Op(%U.as_type.f7f, %DivWith.facet.b8f) [symbolic]
// CHECK:STDOUT:   %ModWith.type.8cc: type = facet_type <@ModWith.1, @ModWith.1(%Other)> [symbolic]
// CHECK:STDOUT:   %Self.310: %ModWith.type.8cc = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %ModWith.assoc_type.4a7: type = assoc_entity_type @ModWith.1, @ModWith.1(%Other) [symbolic]
// CHECK:STDOUT:   %ModWith.WithSelf.Op.type.f6b: type = fn_type @ModWith.WithSelf.Op, @ModWith.WithSelf(%Other, %Self.310) [symbolic]
// CHECK:STDOUT:   %ModWith.WithSelf.Op.397: %ModWith.WithSelf.Op.type.f6b = struct_value () [symbolic]
// CHECK:STDOUT:   %ModWith.type.125: type = facet_type <@ModWith.1, @ModWith.1(%U.as_type.f7f)> [symbolic]
// CHECK:STDOUT:   %facet_type.978: type = facet_type <@Destroy & @ModWith.1, @ModWith.1(%U.as_type.f7f)> [symbolic]
// CHECK:STDOUT:   %.Self.0e2: %facet_type.978 = symbolic_binding .Self [symbolic]
// CHECK:STDOUT:   %ModWith.assoc_type.3d0: type = assoc_entity_type @ModWith.1, @ModWith.1(%U.as_type.f7f) [symbolic]
// CHECK:STDOUT:   %assoc0.649: %ModWith.assoc_type.3d0 = assoc_entity element0, imports.%Core.import_ref.327 [symbolic]
// CHECK:STDOUT:   %assoc1.3aa: %ModWith.assoc_type.3d0 = assoc_entity element1, imports.%Core.import_ref.104 [symbolic]
// CHECK:STDOUT:   %require_complete.4ae: <witness> = require_complete_type %facet_type.978 [symbolic]
// CHECK:STDOUT:   %.Self.as_type.d0f: type = facet_access_type %.Self.0e2 [symbolic]
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.ac5: <witness> = lookup_impl_witness %.Self.0e2, @ModWith.1, @ModWith.1(%U.as_type.f7f) [symbolic]
// CHECK:STDOUT:   %impl.elem0.50a: type = impl_witness_access %ModWith.lookup_impl_witness.ac5, element0 [symbolic]
// CHECK:STDOUT:   %facet_type.c32: type = facet_type <@Destroy & @ModWith.1, @ModWith.1(%U.as_type.f7f) where %impl.elem0.50a = %Result.as_type> [symbolic]
// CHECK:STDOUT:   %pattern_type.cc0: type = pattern_type %facet_type.c32 [symbolic]
// CHECK:STDOUT:   %T.173: %facet_type.c32 = symbolic_binding T, 2 [symbolic]
// CHECK:STDOUT:   %T.as_type.105: type = facet_access_type %T.173 [symbolic]
// CHECK:STDOUT:   %pattern_type.9c5: type = pattern_type %T.as_type.105 [symbolic]
// CHECK:STDOUT:   %require_complete.a64: <witness> = require_complete_type %ModWith.type.125 [symbolic]
// CHECK:STDOUT:   %assoc1.232: %ModWith.assoc_type.4a7 = assoc_entity element1, imports.%Core.import_ref.bbd [symbolic]
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.35d: <witness> = lookup_impl_witness %T.173, @ModWith.1, @ModWith.1(%U.as_type.f7f) [symbolic]
// CHECK:STDOUT:   %ModWith.facet.132: %ModWith.type.125 = facet_value %T.as_type.105, (%ModWith.lookup_impl_witness.35d) [symbolic]
// CHECK:STDOUT:   %ModWith.WithSelf.Op.type.942: type = fn_type @ModWith.WithSelf.Op, @ModWith.WithSelf(%U.as_type.f7f, %ModWith.facet.132) [symbolic]
// CHECK:STDOUT:   %.1a3: type = fn_type_with_self_type %ModWith.WithSelf.Op.type.942, %ModWith.facet.132 [symbolic]
// CHECK:STDOUT:   %impl.elem1.4de: %.1a3 = impl_witness_access %ModWith.lookup_impl_witness.35d, element1 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.5d3: <specific function> = specific_impl_function %impl.elem1.4de, @ModWith.WithSelf.Op(%U.as_type.f7f, %ModWith.facet.132) [symbolic]
// CHECK:STDOUT:   %Int16: type = class_type @Int16 [concrete]
// CHECK:STDOUT:   %pattern_type.802: type = pattern_type %Int16 [concrete]
// CHECK:STDOUT:   %Int32: type = class_type @Int32 [concrete]
// CHECK:STDOUT:   %pattern_type.e9d: type = pattern_type %Int32 [concrete]
// CHECK:STDOUT:   %Int64: type = class_type @Int64 [concrete]
// CHECK:STDOUT:   %pattern_type.7b1: type = pattern_type %Int64 [concrete]
// CHECK:STDOUT:   %int_16: Core.IntLiteral = int_value 16 [concrete]
// CHECK:STDOUT:   %i16: type = class_type @Int, @Int(%int_16) [concrete]
// CHECK:STDOUT:   %pattern_type.e1b: type = pattern_type %i16 [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %i16.builtin: type = int_type signed, %int_16 [concrete]
// CHECK:STDOUT:   %complete_type.2da: <witness> = complete_type_witness %i16.builtin [concrete]
// CHECK:STDOUT:   %Int16.Op.type.43d658.1: type = fn_type @Int16.Op.1 [concrete]
// CHECK:STDOUT:   %Int16.Op.cc70d5.1: %Int16.Op.type.43d658.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.a98c63.1: <witness> = custom_witness (%Int16.Op.cc70d5.1), @Destroy [concrete]
// CHECK:STDOUT:   %Destroy.facet.1b8: %Destroy.type = facet_value %Int16, (%custom_witness.a98c63.1) [concrete]
// CHECK:STDOUT:   %facet_type.f17: type = facet_type <@Destroy & @AddWith.1, @AddWith.1(%Int16)> [concrete]
// CHECK:STDOUT:   %.Self.863: %facet_type.f17 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.896: <witness> = lookup_impl_witness %.Self.863, @AddWith.1, @AddWith.1(%Int16) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.2d1: type = impl_witness_access %AddWith.lookup_impl_witness.896, element0 [symbolic_self]
// CHECK:STDOUT:   %facet_type.753: type = facet_type <@Destroy & @AddWith.1, @AddWith.1(%Int16) where %impl.elem0.2d1 = %Int16> [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.790b3b.2: type = fn_type @Destroy.Op.loc176_34.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.2dd605.2: %Destroy.Op.type.790b3b.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.e20f1f.2: <witness> = custom_witness (%Destroy.Op.2dd605.2), @Destroy [concrete]
// CHECK:STDOUT:   %AddWith.type.6f4: type = facet_type <@AddWith.1, @AddWith.1(%Int16)> [concrete]
// CHECK:STDOUT:   %AddWith.assoc_type.e37: type = assoc_entity_type @AddWith.1, @AddWith.1(%Int16) [concrete]
// CHECK:STDOUT:   %assoc0.4ec: %AddWith.assoc_type.e37 = assoc_entity element0, imports.%Core.import_ref.1dd [concrete]
// CHECK:STDOUT:   %assoc1.c66: %AddWith.assoc_type.e37 = assoc_entity element1, imports.%Core.import_ref.29ff [concrete]
// CHECK:STDOUT:   %Op.type.0b8387.1: type = fn_type @Op.1 [concrete]
// CHECK:STDOUT:   %Op.b9c47d.1: %Op.type.0b8387.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.facet.181: %Destroy.type = facet_value %i16, (%custom_witness.e20f1f.2) [concrete]
// CHECK:STDOUT:   %custom_witness.604: <witness> = custom_witness (%Int16, %Op.b9c47d.1), @AddWith.1, @AddWith.1(%Int16) [concrete]
// CHECK:STDOUT:   %facet_value.fdd: %facet_type.753 = facet_value %i16, (%custom_witness.e20f1f.2, %custom_witness.604) [concrete]
// CHECK:STDOUT:   %complete_type.211: <witness> = complete_type_witness %facet_type.f17 [concrete]
// CHECK:STDOUT:   %.Self.as_type.db9: type = facet_access_type %.Self.863 [symbolic_self]
// CHECK:STDOUT:   %pattern_type.bc4: type = pattern_type %facet_type.753 [concrete]
// CHECK:STDOUT:   %facet_type.c80: type = facet_type <@Destroy & @SubWith.1, @SubWith.1(%Int16)> [concrete]
// CHECK:STDOUT:   %.Self.700: %facet_type.c80 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.e8f: <witness> = lookup_impl_witness %.Self.700, @SubWith.1, @SubWith.1(%Int16) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.0b0: type = impl_witness_access %SubWith.lookup_impl_witness.e8f, element0 [symbolic_self]
// CHECK:STDOUT:   %facet_type.5f5: type = facet_type <@Destroy & @SubWith.1, @SubWith.1(%Int16) where %impl.elem0.0b0 = %Int16> [concrete]
// CHECK:STDOUT:   %SubWith.type.8bd: type = facet_type <@SubWith.1, @SubWith.1(%Int16)> [concrete]
// CHECK:STDOUT:   %SubWith.assoc_type.eb8: type = assoc_entity_type @SubWith.1, @SubWith.1(%Int16) [concrete]
// CHECK:STDOUT:   %assoc0.8a1: %SubWith.assoc_type.eb8 = assoc_entity element0, imports.%Core.import_ref.391 [concrete]
// CHECK:STDOUT:   %assoc1.937f: %SubWith.assoc_type.eb8 = assoc_entity element1, imports.%Core.import_ref.940 [concrete]
// CHECK:STDOUT:   %Op.type.0b8387.2: type = fn_type @Op.2 [concrete]
// CHECK:STDOUT:   %Op.b9c47d.2: %Op.type.0b8387.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.4c1: <witness> = custom_witness (%Int16, %Op.b9c47d.2), @SubWith.1, @SubWith.1(%Int16) [concrete]
// CHECK:STDOUT:   %facet_value.578: %facet_type.5f5 = facet_value %i16, (%custom_witness.e20f1f.2, %custom_witness.4c1) [concrete]
// CHECK:STDOUT:   %complete_type.a74: <witness> = complete_type_witness %facet_type.c80 [concrete]
// CHECK:STDOUT:   %.Self.as_type.c4d: type = facet_access_type %.Self.700 [symbolic_self]
// CHECK:STDOUT:   %pattern_type.4d6: type = pattern_type %facet_type.5f5 [concrete]
// CHECK:STDOUT:   %facet_type.d27: type = facet_type <@Destroy & @MulWith.1, @MulWith.1(%Int16)> [concrete]
// CHECK:STDOUT:   %.Self.e39: %facet_type.d27 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.5e1: <witness> = lookup_impl_witness %.Self.e39, @MulWith.1, @MulWith.1(%Int16) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.e54: type = impl_witness_access %MulWith.lookup_impl_witness.5e1, element0 [symbolic_self]
// CHECK:STDOUT:   %facet_type.9da: type = facet_type <@Destroy & @MulWith.1, @MulWith.1(%Int16) where %impl.elem0.e54 = %Int16> [concrete]
// CHECK:STDOUT:   %MulWith.type.c25: type = facet_type <@MulWith.1, @MulWith.1(%Int16)> [concrete]
// CHECK:STDOUT:   %MulWith.assoc_type.543: type = assoc_entity_type @MulWith.1, @MulWith.1(%Int16) [concrete]
// CHECK:STDOUT:   %assoc0.81e: %MulWith.assoc_type.543 = assoc_entity element0, imports.%Core.import_ref.41e [concrete]
// CHECK:STDOUT:   %assoc1.911: %MulWith.assoc_type.543 = assoc_entity element1, imports.%Core.import_ref.d23 [concrete]
// CHECK:STDOUT:   %Op.type.0b8387.3: type = fn_type @Op.3 [concrete]
// CHECK:STDOUT:   %Op.b9c47d.3: %Op.type.0b8387.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.72a: <witness> = custom_witness (%Int16, %Op.b9c47d.3), @MulWith.1, @MulWith.1(%Int16) [concrete]
// CHECK:STDOUT:   %facet_value.1ee: %facet_type.9da = facet_value %i16, (%custom_witness.e20f1f.2, %custom_witness.72a) [concrete]
// CHECK:STDOUT:   %complete_type.aa7: <witness> = complete_type_witness %facet_type.d27 [concrete]
// CHECK:STDOUT:   %.Self.as_type.87e: type = facet_access_type %.Self.e39 [symbolic_self]
// CHECK:STDOUT:   %pattern_type.d31: type = pattern_type %facet_type.9da [concrete]
// CHECK:STDOUT:   %facet_type.8eb: type = facet_type <@Destroy & @DivWith.1, @DivWith.1(%Int16)> [concrete]
// CHECK:STDOUT:   %.Self.467: %facet_type.8eb = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.9bc: <witness> = lookup_impl_witness %.Self.467, @DivWith.1, @DivWith.1(%Int16) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.5c6: type = impl_witness_access %DivWith.lookup_impl_witness.9bc, element0 [symbolic_self]
// CHECK:STDOUT:   %facet_type.7f5: type = facet_type <@Destroy & @DivWith.1, @DivWith.1(%Int16) where %impl.elem0.5c6 = %Int16> [concrete]
// CHECK:STDOUT:   %DivWith.type.053: type = facet_type <@DivWith.1, @DivWith.1(%Int16)> [concrete]
// CHECK:STDOUT:   %DivWith.assoc_type.73e: type = assoc_entity_type @DivWith.1, @DivWith.1(%Int16) [concrete]
// CHECK:STDOUT:   %assoc0.e96: %DivWith.assoc_type.73e = assoc_entity element0, imports.%Core.import_ref.e71d [concrete]
// CHECK:STDOUT:   %assoc1.10b: %DivWith.assoc_type.73e = assoc_entity element1, imports.%Core.import_ref.e712 [concrete]
// CHECK:STDOUT:   %Op.type.0b8387.4: type = fn_type @Op.4 [concrete]
// CHECK:STDOUT:   %Op.b9c47d.4: %Op.type.0b8387.4 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.6ea: <witness> = custom_witness (%Int16, %Op.b9c47d.4), @DivWith.1, @DivWith.1(%Int16) [concrete]
// CHECK:STDOUT:   %facet_value.6b2: %facet_type.7f5 = facet_value %i16, (%custom_witness.e20f1f.2, %custom_witness.6ea) [concrete]
// CHECK:STDOUT:   %complete_type.d563: <witness> = complete_type_witness %facet_type.8eb [concrete]
// CHECK:STDOUT:   %.Self.as_type.3cd: type = facet_access_type %.Self.467 [symbolic_self]
// CHECK:STDOUT:   %pattern_type.2be: type = pattern_type %facet_type.7f5 [concrete]
// CHECK:STDOUT:   %facet_type.dc7: type = facet_type <@Destroy & @ModWith.1, @ModWith.1(%Int16)> [concrete]
// CHECK:STDOUT:   %.Self.703: %facet_type.dc7 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.c7a: <witness> = lookup_impl_witness %.Self.703, @ModWith.1, @ModWith.1(%Int16) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.094: type = impl_witness_access %ModWith.lookup_impl_witness.c7a, element0 [symbolic_self]
// CHECK:STDOUT:   %facet_type.fde: type = facet_type <@Destroy & @ModWith.1, @ModWith.1(%Int16) where %impl.elem0.094 = %Int16> [concrete]
// CHECK:STDOUT:   %ModWith.type.a83: type = facet_type <@ModWith.1, @ModWith.1(%Int16)> [concrete]
// CHECK:STDOUT:   %ModWith.assoc_type.55b: type = assoc_entity_type @ModWith.1, @ModWith.1(%Int16) [concrete]
// CHECK:STDOUT:   %assoc0.8a3: %ModWith.assoc_type.55b = assoc_entity element0, imports.%Core.import_ref.327 [concrete]
// CHECK:STDOUT:   %assoc1.7d4: %ModWith.assoc_type.55b = assoc_entity element1, imports.%Core.import_ref.104 [concrete]
// CHECK:STDOUT:   %Op.type.0b8387.5: type = fn_type @Op.5 [concrete]
// CHECK:STDOUT:   %Op.b9c47d.5: %Op.type.0b8387.5 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b81: <witness> = custom_witness (%Int16, %Op.b9c47d.5), @ModWith.1, @ModWith.1(%Int16) [concrete]
// CHECK:STDOUT:   %facet_value.07d: %facet_type.fde = facet_value %i16, (%custom_witness.e20f1f.2, %custom_witness.b81) [concrete]
// CHECK:STDOUT:   %complete_type.65d: <witness> = complete_type_witness %facet_type.dc7 [concrete]
// CHECK:STDOUT:   %.Self.as_type.c72: type = facet_access_type %.Self.703 [symbolic_self]
// CHECK:STDOUT:   %pattern_type.268: type = pattern_type %facet_type.fde [concrete]
// CHECK:STDOUT:   %facet_type.bf5: type = facet_type <@Destroy & @AddWith.1, @AddWith.1(%i16)> [concrete]
// CHECK:STDOUT:   %.Self.b0b: %facet_type.bf5 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.2ab: <witness> = lookup_impl_witness %.Self.b0b, @AddWith.1, @AddWith.1(%i16) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.b1c: type = impl_witness_access %AddWith.lookup_impl_witness.2ab, element0 [symbolic_self]
// CHECK:STDOUT:   %facet_type.c90: type = facet_type <@Destroy & @AddWith.1, @AddWith.1(%i16) where %impl.elem0.b1c = %Int16> [concrete]
// CHECK:STDOUT:   %AddWith.type.63e: type = facet_type <@AddWith.1, @AddWith.1(%i16)> [concrete]
// CHECK:STDOUT:   %AddWith.assoc_type.fdd: type = assoc_entity_type @AddWith.1, @AddWith.1(%i16) [concrete]
// CHECK:STDOUT:   %assoc0.c67: %AddWith.assoc_type.fdd = assoc_entity element0, imports.%Core.import_ref.1dd [concrete]
// CHECK:STDOUT:   %assoc1.0e4: %AddWith.assoc_type.fdd = assoc_entity element1, imports.%Core.import_ref.29ff [concrete]
// CHECK:STDOUT:   %Int16.Op.type.43d658.3: type = fn_type @Int16.Op.3 [concrete]
// CHECK:STDOUT:   %Int16.Op.cc70d5.3: %Int16.Op.type.43d658.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.9f2: <witness> = custom_witness (%Int16, %Int16.Op.cc70d5.3), @AddWith.1, @AddWith.1(%i16) [concrete]
// CHECK:STDOUT:   %facet_value.b21: %facet_type.c90 = facet_value %Int16, (%custom_witness.a98c63.1, %custom_witness.9f2) [concrete]
// CHECK:STDOUT:   %complete_type.b92: <witness> = complete_type_witness %facet_type.bf5 [concrete]
// CHECK:STDOUT:   %.Self.as_type.ed9: type = facet_access_type %.Self.b0b [symbolic_self]
// CHECK:STDOUT:   %pattern_type.a15: type = pattern_type %facet_type.c90 [concrete]
// CHECK:STDOUT:   %facet_type.5e7: type = facet_type <@Destroy & @SubWith.1, @SubWith.1(%i16)> [concrete]
// CHECK:STDOUT:   %.Self.bea: %facet_type.5e7 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.7b4: <witness> = lookup_impl_witness %.Self.bea, @SubWith.1, @SubWith.1(%i16) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.61d: type = impl_witness_access %SubWith.lookup_impl_witness.7b4, element0 [symbolic_self]
// CHECK:STDOUT:   %facet_type.80f: type = facet_type <@Destroy & @SubWith.1, @SubWith.1(%i16) where %impl.elem0.61d = %Int16> [concrete]
// CHECK:STDOUT:   %SubWith.type.1db: type = facet_type <@SubWith.1, @SubWith.1(%i16)> [concrete]
// CHECK:STDOUT:   %SubWith.assoc_type.5df: type = assoc_entity_type @SubWith.1, @SubWith.1(%i16) [concrete]
// CHECK:STDOUT:   %assoc0.870: %SubWith.assoc_type.5df = assoc_entity element0, imports.%Core.import_ref.391 [concrete]
// CHECK:STDOUT:   %assoc1.b6b: %SubWith.assoc_type.5df = assoc_entity element1, imports.%Core.import_ref.940 [concrete]
// CHECK:STDOUT:   %Int16.Op.type.43d658.4: type = fn_type @Int16.Op.4 [concrete]
// CHECK:STDOUT:   %Int16.Op.cc70d5.4: %Int16.Op.type.43d658.4 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.6a5: <witness> = custom_witness (%Int16, %Int16.Op.cc70d5.4), @SubWith.1, @SubWith.1(%i16) [concrete]
// CHECK:STDOUT:   %facet_value.2b4: %facet_type.80f = facet_value %Int16, (%custom_witness.a98c63.1, %custom_witness.6a5) [concrete]
// CHECK:STDOUT:   %complete_type.9fc: <witness> = complete_type_witness %facet_type.5e7 [concrete]
// CHECK:STDOUT:   %.Self.as_type.b77: type = facet_access_type %.Self.bea [symbolic_self]
// CHECK:STDOUT:   %pattern_type.7a0: type = pattern_type %facet_type.80f [concrete]
// CHECK:STDOUT:   %facet_type.183: type = facet_type <@Destroy & @MulWith.1, @MulWith.1(%i16)> [concrete]
// CHECK:STDOUT:   %.Self.0ef: %facet_type.183 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.b3c: <witness> = lookup_impl_witness %.Self.0ef, @MulWith.1, @MulWith.1(%i16) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.4a2: type = impl_witness_access %MulWith.lookup_impl_witness.b3c, element0 [symbolic_self]
// CHECK:STDOUT:   %facet_type.6bb: type = facet_type <@Destroy & @MulWith.1, @MulWith.1(%i16) where %impl.elem0.4a2 = %Int16> [concrete]
// CHECK:STDOUT:   %MulWith.type.c35: type = facet_type <@MulWith.1, @MulWith.1(%i16)> [concrete]
// CHECK:STDOUT:   %MulWith.assoc_type.378: type = assoc_entity_type @MulWith.1, @MulWith.1(%i16) [concrete]
// CHECK:STDOUT:   %assoc0.7cb: %MulWith.assoc_type.378 = assoc_entity element0, imports.%Core.import_ref.41e [concrete]
// CHECK:STDOUT:   %assoc1.9b3: %MulWith.assoc_type.378 = assoc_entity element1, imports.%Core.import_ref.d23 [concrete]
// CHECK:STDOUT:   %Int16.Op.type.43d658.5: type = fn_type @Int16.Op.5 [concrete]
// CHECK:STDOUT:   %Int16.Op.cc70d5.5: %Int16.Op.type.43d658.5 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.35c: <witness> = custom_witness (%Int16, %Int16.Op.cc70d5.5), @MulWith.1, @MulWith.1(%i16) [concrete]
// CHECK:STDOUT:   %facet_value.88a: %facet_type.6bb = facet_value %Int16, (%custom_witness.a98c63.1, %custom_witness.35c) [concrete]
// CHECK:STDOUT:   %complete_type.309: <witness> = complete_type_witness %facet_type.183 [concrete]
// CHECK:STDOUT:   %.Self.as_type.751: type = facet_access_type %.Self.0ef [symbolic_self]
// CHECK:STDOUT:   %pattern_type.fff: type = pattern_type %facet_type.6bb [concrete]
// CHECK:STDOUT:   %facet_type.ab4: type = facet_type <@Destroy & @DivWith.1, @DivWith.1(%i16)> [concrete]
// CHECK:STDOUT:   %.Self.627: %facet_type.ab4 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.3d7: <witness> = lookup_impl_witness %.Self.627, @DivWith.1, @DivWith.1(%i16) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.ce6: type = impl_witness_access %DivWith.lookup_impl_witness.3d7, element0 [symbolic_self]
// CHECK:STDOUT:   %facet_type.6b1: type = facet_type <@Destroy & @DivWith.1, @DivWith.1(%i16) where %impl.elem0.ce6 = %Int16> [concrete]
// CHECK:STDOUT:   %DivWith.type.086: type = facet_type <@DivWith.1, @DivWith.1(%i16)> [concrete]
// CHECK:STDOUT:   %DivWith.assoc_type.d3a: type = assoc_entity_type @DivWith.1, @DivWith.1(%i16) [concrete]
// CHECK:STDOUT:   %assoc0.8b8: %DivWith.assoc_type.d3a = assoc_entity element0, imports.%Core.import_ref.e71d [concrete]
// CHECK:STDOUT:   %assoc1.acb: %DivWith.assoc_type.d3a = assoc_entity element1, imports.%Core.import_ref.e712 [concrete]
// CHECK:STDOUT:   %Int16.Op.type.43d658.6: type = fn_type @Int16.Op.6 [concrete]
// CHECK:STDOUT:   %Int16.Op.cc70d5.6: %Int16.Op.type.43d658.6 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.52d: <witness> = custom_witness (%Int16, %Int16.Op.cc70d5.6), @DivWith.1, @DivWith.1(%i16) [concrete]
// CHECK:STDOUT:   %facet_value.473: %facet_type.6b1 = facet_value %Int16, (%custom_witness.a98c63.1, %custom_witness.52d) [concrete]
// CHECK:STDOUT:   %complete_type.d56e: <witness> = complete_type_witness %facet_type.ab4 [concrete]
// CHECK:STDOUT:   %.Self.as_type.4c8: type = facet_access_type %.Self.627 [symbolic_self]
// CHECK:STDOUT:   %pattern_type.92a: type = pattern_type %facet_type.6b1 [concrete]
// CHECK:STDOUT:   %facet_type.4ac: type = facet_type <@Destroy & @ModWith.1, @ModWith.1(%i16)> [concrete]
// CHECK:STDOUT:   %.Self.b05: %facet_type.4ac = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.ae1: <witness> = lookup_impl_witness %.Self.b05, @ModWith.1, @ModWith.1(%i16) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.9ea: type = impl_witness_access %ModWith.lookup_impl_witness.ae1, element0 [symbolic_self]
// CHECK:STDOUT:   %facet_type.b54: type = facet_type <@Destroy & @ModWith.1, @ModWith.1(%i16) where %impl.elem0.9ea = %Int16> [concrete]
// CHECK:STDOUT:   %ModWith.type.59b: type = facet_type <@ModWith.1, @ModWith.1(%i16)> [concrete]
// CHECK:STDOUT:   %ModWith.assoc_type.95c: type = assoc_entity_type @ModWith.1, @ModWith.1(%i16) [concrete]
// CHECK:STDOUT:   %assoc0.954: %ModWith.assoc_type.95c = assoc_entity element0, imports.%Core.import_ref.327 [concrete]
// CHECK:STDOUT:   %assoc1.503: %ModWith.assoc_type.95c = assoc_entity element1, imports.%Core.import_ref.104 [concrete]
// CHECK:STDOUT:   %Int16.Op.type.43d658.7: type = fn_type @Int16.Op.7 [concrete]
// CHECK:STDOUT:   %Int16.Op.cc70d5.7: %Int16.Op.type.43d658.7 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.887: <witness> = custom_witness (%Int16, %Int16.Op.cc70d5.7), @ModWith.1, @ModWith.1(%i16) [concrete]
// CHECK:STDOUT:   %facet_value.c3b: %facet_type.b54 = facet_value %Int16, (%custom_witness.a98c63.1, %custom_witness.887) [concrete]
// CHECK:STDOUT:   %complete_type.602: <witness> = complete_type_witness %facet_type.4ac [concrete]
// CHECK:STDOUT:   %.Self.as_type.c10: type = facet_access_type %.Self.b05 [symbolic_self]
// CHECK:STDOUT:   %pattern_type.87b: type = pattern_type %facet_type.b54 [concrete]
// CHECK:STDOUT:   %Int32.Op.type.43d658.1: type = fn_type @Int32.Op.1 [concrete]
// CHECK:STDOUT:   %Int32.Op.cc70d5.1: %Int32.Op.type.43d658.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.a98c63.2: <witness> = custom_witness (%Int32.Op.cc70d5.1), @Destroy [concrete]
// CHECK:STDOUT:   %Destroy.facet.d87: %Destroy.type = facet_value %Int32, (%custom_witness.a98c63.2) [concrete]
// CHECK:STDOUT:   %facet_type.9c9: type = facet_type <@Destroy & @AddWith.1, @AddWith.1(%Int32)> [concrete]
// CHECK:STDOUT:   %.Self.495: %facet_type.9c9 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.4d9: <witness> = lookup_impl_witness %.Self.495, @AddWith.1, @AddWith.1(%Int32) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.53a: type = impl_witness_access %AddWith.lookup_impl_witness.4d9, element0 [symbolic_self]
// CHECK:STDOUT:   %facet_type.8fd: type = facet_type <@Destroy & @AddWith.1, @AddWith.1(%Int32) where %impl.elem0.53a = %Int32> [concrete]
// CHECK:STDOUT:   %AddWith.type.78c: type = facet_type <@AddWith.1, @AddWith.1(%Int32)> [concrete]
// CHECK:STDOUT:   %AddWith.assoc_type.882: type = assoc_entity_type @AddWith.1, @AddWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %assoc0.5b9: %AddWith.assoc_type.882 = assoc_entity element0, imports.%Core.import_ref.1dd [concrete]
// CHECK:STDOUT:   %assoc1.892: %AddWith.assoc_type.882 = assoc_entity element1, imports.%Core.import_ref.29ff [concrete]
// CHECK:STDOUT:   %Int16.Op.type.43d658.8: type = fn_type @Int16.Op.8 [concrete]
// CHECK:STDOUT:   %Int16.Op.cc70d5.8: %Int16.Op.type.43d658.8 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.e46c75.1: <witness> = custom_witness (%Int32, %Int16.Op.cc70d5.8), @AddWith.1, @AddWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %facet_value.464: %facet_type.8fd = facet_value %Int16, (%custom_witness.a98c63.1, %custom_witness.e46c75.1) [concrete]
// CHECK:STDOUT:   %complete_type.f26: <witness> = complete_type_witness %facet_type.9c9 [concrete]
// CHECK:STDOUT:   %.Self.as_type.154: type = facet_access_type %.Self.495 [symbolic_self]
// CHECK:STDOUT:   %pattern_type.15c: type = pattern_type %facet_type.8fd [concrete]
// CHECK:STDOUT:   %facet_type.6c8: type = facet_type <@Destroy & @SubWith.1, @SubWith.1(%Int32)> [concrete]
// CHECK:STDOUT:   %.Self.cbb: %facet_type.6c8 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.733: <witness> = lookup_impl_witness %.Self.cbb, @SubWith.1, @SubWith.1(%Int32) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.e47: type = impl_witness_access %SubWith.lookup_impl_witness.733, element0 [symbolic_self]
// CHECK:STDOUT:   %facet_type.02a: type = facet_type <@Destroy & @SubWith.1, @SubWith.1(%Int32) where %impl.elem0.e47 = %Int32> [concrete]
// CHECK:STDOUT:   %SubWith.type.629: type = facet_type <@SubWith.1, @SubWith.1(%Int32)> [concrete]
// CHECK:STDOUT:   %SubWith.assoc_type.a03: type = assoc_entity_type @SubWith.1, @SubWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %assoc0.199: %SubWith.assoc_type.a03 = assoc_entity element0, imports.%Core.import_ref.391 [concrete]
// CHECK:STDOUT:   %assoc1.bff: %SubWith.assoc_type.a03 = assoc_entity element1, imports.%Core.import_ref.940 [concrete]
// CHECK:STDOUT:   %Int16.Op.type.43d658.9: type = fn_type @Int16.Op.9 [concrete]
// CHECK:STDOUT:   %Int16.Op.cc70d5.9: %Int16.Op.type.43d658.9 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.7ffeb1.1: <witness> = custom_witness (%Int32, %Int16.Op.cc70d5.9), @SubWith.1, @SubWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %facet_value.197: %facet_type.02a = facet_value %Int16, (%custom_witness.a98c63.1, %custom_witness.7ffeb1.1) [concrete]
// CHECK:STDOUT:   %complete_type.dae: <witness> = complete_type_witness %facet_type.6c8 [concrete]
// CHECK:STDOUT:   %.Self.as_type.11e: type = facet_access_type %.Self.cbb [symbolic_self]
// CHECK:STDOUT:   %pattern_type.553: type = pattern_type %facet_type.02a [concrete]
// CHECK:STDOUT:   %facet_type.299: type = facet_type <@Destroy & @MulWith.1, @MulWith.1(%Int32)> [concrete]
// CHECK:STDOUT:   %.Self.1c7: %facet_type.299 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.5ca: <witness> = lookup_impl_witness %.Self.1c7, @MulWith.1, @MulWith.1(%Int32) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.cab: type = impl_witness_access %MulWith.lookup_impl_witness.5ca, element0 [symbolic_self]
// CHECK:STDOUT:   %facet_type.a5c: type = facet_type <@Destroy & @MulWith.1, @MulWith.1(%Int32) where %impl.elem0.cab = %Int32> [concrete]
// CHECK:STDOUT:   %MulWith.type.3ee: type = facet_type <@MulWith.1, @MulWith.1(%Int32)> [concrete]
// CHECK:STDOUT:   %MulWith.assoc_type.564: type = assoc_entity_type @MulWith.1, @MulWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %assoc0.6fa: %MulWith.assoc_type.564 = assoc_entity element0, imports.%Core.import_ref.41e [concrete]
// CHECK:STDOUT:   %assoc1.00d: %MulWith.assoc_type.564 = assoc_entity element1, imports.%Core.import_ref.d23 [concrete]
// CHECK:STDOUT:   %Int16.Op.type.43d658.10: type = fn_type @Int16.Op.10 [concrete]
// CHECK:STDOUT:   %Int16.Op.cc70d5.10: %Int16.Op.type.43d658.10 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.68a6cf.1: <witness> = custom_witness (%Int32, %Int16.Op.cc70d5.10), @MulWith.1, @MulWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %facet_value.5b3: %facet_type.a5c = facet_value %Int16, (%custom_witness.a98c63.1, %custom_witness.68a6cf.1) [concrete]
// CHECK:STDOUT:   %complete_type.0cf: <witness> = complete_type_witness %facet_type.299 [concrete]
// CHECK:STDOUT:   %.Self.as_type.ba5: type = facet_access_type %.Self.1c7 [symbolic_self]
// CHECK:STDOUT:   %pattern_type.11c: type = pattern_type %facet_type.a5c [concrete]
// CHECK:STDOUT:   %facet_type.c6e: type = facet_type <@Destroy & @DivWith.1, @DivWith.1(%Int32)> [concrete]
// CHECK:STDOUT:   %.Self.988: %facet_type.c6e = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.c00: <witness> = lookup_impl_witness %.Self.988, @DivWith.1, @DivWith.1(%Int32) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.7a0: type = impl_witness_access %DivWith.lookup_impl_witness.c00, element0 [symbolic_self]
// CHECK:STDOUT:   %facet_type.b40: type = facet_type <@Destroy & @DivWith.1, @DivWith.1(%Int32) where %impl.elem0.7a0 = %Int32> [concrete]
// CHECK:STDOUT:   %DivWith.type.09b: type = facet_type <@DivWith.1, @DivWith.1(%Int32)> [concrete]
// CHECK:STDOUT:   %DivWith.assoc_type.a72: type = assoc_entity_type @DivWith.1, @DivWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %assoc0.d0a: %DivWith.assoc_type.a72 = assoc_entity element0, imports.%Core.import_ref.e71d [concrete]
// CHECK:STDOUT:   %assoc1.489: %DivWith.assoc_type.a72 = assoc_entity element1, imports.%Core.import_ref.e712 [concrete]
// CHECK:STDOUT:   %Int16.Op.type.43d658.11: type = fn_type @Int16.Op.11 [concrete]
// CHECK:STDOUT:   %Int16.Op.cc70d5.11: %Int16.Op.type.43d658.11 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.2ccbdb.1: <witness> = custom_witness (%Int32, %Int16.Op.cc70d5.11), @DivWith.1, @DivWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %facet_value.6fd: %facet_type.b40 = facet_value %Int16, (%custom_witness.a98c63.1, %custom_witness.2ccbdb.1) [concrete]
// CHECK:STDOUT:   %complete_type.36c: <witness> = complete_type_witness %facet_type.c6e [concrete]
// CHECK:STDOUT:   %.Self.as_type.f38: type = facet_access_type %.Self.988 [symbolic_self]
// CHECK:STDOUT:   %pattern_type.6b3: type = pattern_type %facet_type.b40 [concrete]
// CHECK:STDOUT:   %facet_type.22f: type = facet_type <@Destroy & @ModWith.1, @ModWith.1(%Int32)> [concrete]
// CHECK:STDOUT:   %.Self.572: %facet_type.22f = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.18e: <witness> = lookup_impl_witness %.Self.572, @ModWith.1, @ModWith.1(%Int32) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.288: type = impl_witness_access %ModWith.lookup_impl_witness.18e, element0 [symbolic_self]
// CHECK:STDOUT:   %facet_type.88b: type = facet_type <@Destroy & @ModWith.1, @ModWith.1(%Int32) where %impl.elem0.288 = %Int32> [concrete]
// CHECK:STDOUT:   %ModWith.type.6f6: type = facet_type <@ModWith.1, @ModWith.1(%Int32)> [concrete]
// CHECK:STDOUT:   %ModWith.assoc_type.d09: type = assoc_entity_type @ModWith.1, @ModWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %assoc0.fb6: %ModWith.assoc_type.d09 = assoc_entity element0, imports.%Core.import_ref.327 [concrete]
// CHECK:STDOUT:   %assoc1.56e: %ModWith.assoc_type.d09 = assoc_entity element1, imports.%Core.import_ref.104 [concrete]
// CHECK:STDOUT:   %Int16.Op.type.43d658.12: type = fn_type @Int16.Op.12 [concrete]
// CHECK:STDOUT:   %Int16.Op.cc70d5.12: %Int16.Op.type.43d658.12 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.596ded.1: <witness> = custom_witness (%Int32, %Int16.Op.cc70d5.12), @ModWith.1, @ModWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %facet_value.592: %facet_type.88b = facet_value %Int16, (%custom_witness.a98c63.1, %custom_witness.596ded.1) [concrete]
// CHECK:STDOUT:   %complete_type.e22: <witness> = complete_type_witness %facet_type.22f [concrete]
// CHECK:STDOUT:   %.Self.as_type.aac: type = facet_access_type %.Self.572 [symbolic_self]
// CHECK:STDOUT:   %pattern_type.dda: type = pattern_type %facet_type.88b [concrete]
// CHECK:STDOUT:   %facet_type.558: type = facet_type <@Destroy & @AddWith.1, @AddWith.1(%Int16) where %impl.elem0.2d1 = %Int32> [concrete]
// CHECK:STDOUT:   %Int32.Op.type.43d658.2: type = fn_type @Int32.Op.2 [concrete]
// CHECK:STDOUT:   %Int32.Op.cc70d5.2: %Int32.Op.type.43d658.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.e32: <witness> = custom_witness (%Int32, %Int32.Op.cc70d5.2), @AddWith.1, @AddWith.1(%Int16) [concrete]
// CHECK:STDOUT:   %facet_value.f2b: %facet_type.558 = facet_value %Int32, (%custom_witness.a98c63.2, %custom_witness.e32) [concrete]
// CHECK:STDOUT:   %pattern_type.715: type = pattern_type %facet_type.558 [concrete]
// CHECK:STDOUT:   %facet_type.064: type = facet_type <@Destroy & @SubWith.1, @SubWith.1(%Int16) where %impl.elem0.0b0 = %Int32> [concrete]
// CHECK:STDOUT:   %Int32.Op.type.43d658.3: type = fn_type @Int32.Op.3 [concrete]
// CHECK:STDOUT:   %Int32.Op.cc70d5.3: %Int32.Op.type.43d658.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.395: <witness> = custom_witness (%Int32, %Int32.Op.cc70d5.3), @SubWith.1, @SubWith.1(%Int16) [concrete]
// CHECK:STDOUT:   %facet_value.dc0: %facet_type.064 = facet_value %Int32, (%custom_witness.a98c63.2, %custom_witness.395) [concrete]
// CHECK:STDOUT:   %pattern_type.b06: type = pattern_type %facet_type.064 [concrete]
// CHECK:STDOUT:   %facet_type.d97: type = facet_type <@Destroy & @MulWith.1, @MulWith.1(%Int16) where %impl.elem0.e54 = %Int32> [concrete]
// CHECK:STDOUT:   %Int32.Op.type.43d658.4: type = fn_type @Int32.Op.4 [concrete]
// CHECK:STDOUT:   %Int32.Op.cc70d5.4: %Int32.Op.type.43d658.4 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.678: <witness> = custom_witness (%Int32, %Int32.Op.cc70d5.4), @MulWith.1, @MulWith.1(%Int16) [concrete]
// CHECK:STDOUT:   %facet_value.ff0: %facet_type.d97 = facet_value %Int32, (%custom_witness.a98c63.2, %custom_witness.678) [concrete]
// CHECK:STDOUT:   %pattern_type.3ec: type = pattern_type %facet_type.d97 [concrete]
// CHECK:STDOUT:   %facet_type.aa6: type = facet_type <@Destroy & @DivWith.1, @DivWith.1(%Int16) where %impl.elem0.5c6 = %Int32> [concrete]
// CHECK:STDOUT:   %Int32.Op.type.43d658.5: type = fn_type @Int32.Op.5 [concrete]
// CHECK:STDOUT:   %Int32.Op.cc70d5.5: %Int32.Op.type.43d658.5 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.8ad: <witness> = custom_witness (%Int32, %Int32.Op.cc70d5.5), @DivWith.1, @DivWith.1(%Int16) [concrete]
// CHECK:STDOUT:   %facet_value.42a: %facet_type.aa6 = facet_value %Int32, (%custom_witness.a98c63.2, %custom_witness.8ad) [concrete]
// CHECK:STDOUT:   %pattern_type.814: type = pattern_type %facet_type.aa6 [concrete]
// CHECK:STDOUT:   %facet_type.b16: type = facet_type <@Destroy & @ModWith.1, @ModWith.1(%Int16) where %impl.elem0.094 = %Int32> [concrete]
// CHECK:STDOUT:   %Int32.Op.type.43d658.6: type = fn_type @Int32.Op.6 [concrete]
// CHECK:STDOUT:   %Int32.Op.cc70d5.6: %Int32.Op.type.43d658.6 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.f63: <witness> = custom_witness (%Int32, %Int32.Op.cc70d5.6), @ModWith.1, @ModWith.1(%Int16) [concrete]
// CHECK:STDOUT:   %facet_value.7ad: %facet_type.b16 = facet_value %Int32, (%custom_witness.a98c63.2, %custom_witness.f63) [concrete]
// CHECK:STDOUT:   %pattern_type.7fa: type = pattern_type %facet_type.b16 [concrete]
// CHECK:STDOUT:   %Int32.Op.type.43d658.7: type = fn_type @Int32.Op.7 [concrete]
// CHECK:STDOUT:   %Int32.Op.cc70d5.7: %Int32.Op.type.43d658.7 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.e46c75.2: <witness> = custom_witness (%Int32, %Int32.Op.cc70d5.7), @AddWith.1, @AddWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %facet_value.f56: %facet_type.8fd = facet_value %Int32, (%custom_witness.a98c63.2, %custom_witness.e46c75.2) [concrete]
// CHECK:STDOUT:   %Int32.Op.type.43d658.8: type = fn_type @Int32.Op.8 [concrete]
// CHECK:STDOUT:   %Int32.Op.cc70d5.8: %Int32.Op.type.43d658.8 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.7ffeb1.2: <witness> = custom_witness (%Int32, %Int32.Op.cc70d5.8), @SubWith.1, @SubWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %facet_value.8a6: %facet_type.02a = facet_value %Int32, (%custom_witness.a98c63.2, %custom_witness.7ffeb1.2) [concrete]
// CHECK:STDOUT:   %Int32.Op.type.43d658.9: type = fn_type @Int32.Op.9 [concrete]
// CHECK:STDOUT:   %Int32.Op.cc70d5.9: %Int32.Op.type.43d658.9 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.68a6cf.2: <witness> = custom_witness (%Int32, %Int32.Op.cc70d5.9), @MulWith.1, @MulWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %facet_value.e09: %facet_type.a5c = facet_value %Int32, (%custom_witness.a98c63.2, %custom_witness.68a6cf.2) [concrete]
// CHECK:STDOUT:   %Int32.Op.type.43d658.10: type = fn_type @Int32.Op.10 [concrete]
// CHECK:STDOUT:   %Int32.Op.cc70d5.10: %Int32.Op.type.43d658.10 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.2ccbdb.2: <witness> = custom_witness (%Int32, %Int32.Op.cc70d5.10), @DivWith.1, @DivWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %facet_value.a3b: %facet_type.b40 = facet_value %Int32, (%custom_witness.a98c63.2, %custom_witness.2ccbdb.2) [concrete]
// CHECK:STDOUT:   %Int32.Op.type.43d658.11: type = fn_type @Int32.Op.11 [concrete]
// CHECK:STDOUT:   %Int32.Op.cc70d5.11: %Int32.Op.type.43d658.11 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.596ded.2: <witness> = custom_witness (%Int32, %Int32.Op.cc70d5.11), @ModWith.1, @ModWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %facet_value.40c: %facet_type.88b = facet_value %Int32, (%custom_witness.a98c63.2, %custom_witness.596ded.2) [concrete]
// CHECK:STDOUT:   %Int64.Op.type.43d658.1: type = fn_type @Int64.Op.1 [concrete]
// CHECK:STDOUT:   %Int64.Op.cc70d5.1: %Int64.Op.type.43d658.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.a98c63.3: <witness> = custom_witness (%Int64.Op.cc70d5.1), @Destroy [concrete]
// CHECK:STDOUT:   %Destroy.facet.3de: %Destroy.type = facet_value %Int64, (%custom_witness.a98c63.3) [concrete]
// CHECK:STDOUT:   %facet_type.fb6: type = facet_type <@Destroy & @AddWith.1, @AddWith.1(%Int64)> [concrete]
// CHECK:STDOUT:   %.Self.d3e: %facet_type.fb6 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.708: <witness> = lookup_impl_witness %.Self.d3e, @AddWith.1, @AddWith.1(%Int64) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.86a: type = impl_witness_access %AddWith.lookup_impl_witness.708, element0 [symbolic_self]
// CHECK:STDOUT:   %facet_type.376: type = facet_type <@Destroy & @AddWith.1, @AddWith.1(%Int64) where %impl.elem0.86a = %Int64> [concrete]
// CHECK:STDOUT:   %AddWith.type.01b: type = facet_type <@AddWith.1, @AddWith.1(%Int64)> [concrete]
// CHECK:STDOUT:   %AddWith.assoc_type.7c5: type = assoc_entity_type @AddWith.1, @AddWith.1(%Int64) [concrete]
// CHECK:STDOUT:   %assoc0.f19: %AddWith.assoc_type.7c5 = assoc_entity element0, imports.%Core.import_ref.1dd [concrete]
// CHECK:STDOUT:   %assoc1.3e0: %AddWith.assoc_type.7c5 = assoc_entity element1, imports.%Core.import_ref.29ff [concrete]
// CHECK:STDOUT:   %Op.type.0b8387.6: type = fn_type @Op.6 [concrete]
// CHECK:STDOUT:   %Op.b9c47d.6: %Op.type.0b8387.6 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.e1e: <witness> = custom_witness (%Int64, %Op.b9c47d.6), @AddWith.1, @AddWith.1(%Int64) [concrete]
// CHECK:STDOUT:   %facet_value.5d2: %facet_type.376 = facet_value %Int32, (%custom_witness.a98c63.2, %custom_witness.e1e) [concrete]
// CHECK:STDOUT:   %complete_type.900: <witness> = complete_type_witness %facet_type.fb6 [concrete]
// CHECK:STDOUT:   %.Self.as_type.1db: type = facet_access_type %.Self.d3e [symbolic_self]
// CHECK:STDOUT:   %pattern_type.e29: type = pattern_type %facet_type.376 [concrete]
// CHECK:STDOUT:   %facet_type.038: type = facet_type <@Destroy & @SubWith.1, @SubWith.1(%Int64)> [concrete]
// CHECK:STDOUT:   %.Self.869: %facet_type.038 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.c9f: <witness> = lookup_impl_witness %.Self.869, @SubWith.1, @SubWith.1(%Int64) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.6f7: type = impl_witness_access %SubWith.lookup_impl_witness.c9f, element0 [symbolic_self]
// CHECK:STDOUT:   %facet_type.4a1: type = facet_type <@Destroy & @SubWith.1, @SubWith.1(%Int64) where %impl.elem0.6f7 = %Int64> [concrete]
// CHECK:STDOUT:   %SubWith.type.82c: type = facet_type <@SubWith.1, @SubWith.1(%Int64)> [concrete]
// CHECK:STDOUT:   %SubWith.assoc_type.f62: type = assoc_entity_type @SubWith.1, @SubWith.1(%Int64) [concrete]
// CHECK:STDOUT:   %assoc0.018: %SubWith.assoc_type.f62 = assoc_entity element0, imports.%Core.import_ref.391 [concrete]
// CHECK:STDOUT:   %assoc1.1cf: %SubWith.assoc_type.f62 = assoc_entity element1, imports.%Core.import_ref.940 [concrete]
// CHECK:STDOUT:   %Op.type.0b8387.7: type = fn_type @Op.7 [concrete]
// CHECK:STDOUT:   %Op.b9c47d.7: %Op.type.0b8387.7 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.053: <witness> = custom_witness (%Int64, %Op.b9c47d.7), @SubWith.1, @SubWith.1(%Int64) [concrete]
// CHECK:STDOUT:   %facet_value.916: %facet_type.4a1 = facet_value %Int32, (%custom_witness.a98c63.2, %custom_witness.053) [concrete]
// CHECK:STDOUT:   %complete_type.e31: <witness> = complete_type_witness %facet_type.038 [concrete]
// CHECK:STDOUT:   %.Self.as_type.824: type = facet_access_type %.Self.869 [symbolic_self]
// CHECK:STDOUT:   %pattern_type.58c: type = pattern_type %facet_type.4a1 [concrete]
// CHECK:STDOUT:   %facet_type.e03: type = facet_type <@Destroy & @MulWith.1, @MulWith.1(%Int64)> [concrete]
// CHECK:STDOUT:   %.Self.d17: %facet_type.e03 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.13a: <witness> = lookup_impl_witness %.Self.d17, @MulWith.1, @MulWith.1(%Int64) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.7f7: type = impl_witness_access %MulWith.lookup_impl_witness.13a, element0 [symbolic_self]
// CHECK:STDOUT:   %facet_type.88c: type = facet_type <@Destroy & @MulWith.1, @MulWith.1(%Int64) where %impl.elem0.7f7 = %Int64> [concrete]
// CHECK:STDOUT:   %MulWith.type.9ed: type = facet_type <@MulWith.1, @MulWith.1(%Int64)> [concrete]
// CHECK:STDOUT:   %MulWith.assoc_type.8d1: type = assoc_entity_type @MulWith.1, @MulWith.1(%Int64) [concrete]
// CHECK:STDOUT:   %assoc0.c01: %MulWith.assoc_type.8d1 = assoc_entity element0, imports.%Core.import_ref.41e [concrete]
// CHECK:STDOUT:   %assoc1.eef: %MulWith.assoc_type.8d1 = assoc_entity element1, imports.%Core.import_ref.d23 [concrete]
// CHECK:STDOUT:   %Op.type.0b8387.8: type = fn_type @Op.8 [concrete]
// CHECK:STDOUT:   %Op.b9c47d.8: %Op.type.0b8387.8 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.f29: <witness> = custom_witness (%Int64, %Op.b9c47d.8), @MulWith.1, @MulWith.1(%Int64) [concrete]
// CHECK:STDOUT:   %facet_value.7ba: %facet_type.88c = facet_value %Int32, (%custom_witness.a98c63.2, %custom_witness.f29) [concrete]
// CHECK:STDOUT:   %complete_type.25d: <witness> = complete_type_witness %facet_type.e03 [concrete]
// CHECK:STDOUT:   %.Self.as_type.a7c: type = facet_access_type %.Self.d17 [symbolic_self]
// CHECK:STDOUT:   %pattern_type.c54: type = pattern_type %facet_type.88c [concrete]
// CHECK:STDOUT:   %facet_type.d41: type = facet_type <@Destroy & @DivWith.1, @DivWith.1(%Int64)> [concrete]
// CHECK:STDOUT:   %.Self.73b: %facet_type.d41 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.12e: <witness> = lookup_impl_witness %.Self.73b, @DivWith.1, @DivWith.1(%Int64) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.2db: type = impl_witness_access %DivWith.lookup_impl_witness.12e, element0 [symbolic_self]
// CHECK:STDOUT:   %facet_type.604: type = facet_type <@Destroy & @DivWith.1, @DivWith.1(%Int64) where %impl.elem0.2db = %Int64> [concrete]
// CHECK:STDOUT:   %DivWith.type.7ac: type = facet_type <@DivWith.1, @DivWith.1(%Int64)> [concrete]
// CHECK:STDOUT:   %DivWith.assoc_type.bfe: type = assoc_entity_type @DivWith.1, @DivWith.1(%Int64) [concrete]
// CHECK:STDOUT:   %assoc0.578: %DivWith.assoc_type.bfe = assoc_entity element0, imports.%Core.import_ref.e71d [concrete]
// CHECK:STDOUT:   %assoc1.c32: %DivWith.assoc_type.bfe = assoc_entity element1, imports.%Core.import_ref.e712 [concrete]
// CHECK:STDOUT:   %Op.type.0b8387.9: type = fn_type @Op.9 [concrete]
// CHECK:STDOUT:   %Op.b9c47d.9: %Op.type.0b8387.9 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.863: <witness> = custom_witness (%Int64, %Op.b9c47d.9), @DivWith.1, @DivWith.1(%Int64) [concrete]
// CHECK:STDOUT:   %facet_value.969: %facet_type.604 = facet_value %Int32, (%custom_witness.a98c63.2, %custom_witness.863) [concrete]
// CHECK:STDOUT:   %complete_type.67e: <witness> = complete_type_witness %facet_type.d41 [concrete]
// CHECK:STDOUT:   %.Self.as_type.e27: type = facet_access_type %.Self.73b [symbolic_self]
// CHECK:STDOUT:   %pattern_type.171: type = pattern_type %facet_type.604 [concrete]
// CHECK:STDOUT:   %facet_type.dfd: type = facet_type <@Destroy & @ModWith.1, @ModWith.1(%Int64)> [concrete]
// CHECK:STDOUT:   %.Self.5bc: %facet_type.dfd = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.45e: <witness> = lookup_impl_witness %.Self.5bc, @ModWith.1, @ModWith.1(%Int64) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.f40: type = impl_witness_access %ModWith.lookup_impl_witness.45e, element0 [symbolic_self]
// CHECK:STDOUT:   %facet_type.d93: type = facet_type <@Destroy & @ModWith.1, @ModWith.1(%Int64) where %impl.elem0.f40 = %Int64> [concrete]
// CHECK:STDOUT:   %ModWith.type.6b0: type = facet_type <@ModWith.1, @ModWith.1(%Int64)> [concrete]
// CHECK:STDOUT:   %ModWith.assoc_type.eb7: type = assoc_entity_type @ModWith.1, @ModWith.1(%Int64) [concrete]
// CHECK:STDOUT:   %assoc0.8a2: %ModWith.assoc_type.eb7 = assoc_entity element0, imports.%Core.import_ref.327 [concrete]
// CHECK:STDOUT:   %assoc1.fa5: %ModWith.assoc_type.eb7 = assoc_entity element1, imports.%Core.import_ref.104 [concrete]
// CHECK:STDOUT:   %Op.type.0b8387.10: type = fn_type @Op.10 [concrete]
// CHECK:STDOUT:   %Op.b9c47d.10: %Op.type.0b8387.10 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.eb8: <witness> = custom_witness (%Int64, %Op.b9c47d.10), @ModWith.1, @ModWith.1(%Int64) [concrete]
// CHECK:STDOUT:   %facet_value.0ea: %facet_type.d93 = facet_value %Int32, (%custom_witness.a98c63.2, %custom_witness.eb8) [concrete]
// CHECK:STDOUT:   %complete_type.76b: <witness> = complete_type_witness %facet_type.dfd [concrete]
// CHECK:STDOUT:   %.Self.as_type.97b: type = facet_access_type %.Self.5bc [symbolic_self]
// CHECK:STDOUT:   %pattern_type.5a9: type = pattern_type %facet_type.d93 [concrete]
// CHECK:STDOUT:   %facet_type.002: type = facet_type <@Destroy & @AddWith.1, @AddWith.1(%Int32) where %impl.elem0.53a = %Int64> [concrete]
// CHECK:STDOUT:   %Op.type.0b8387.11: type = fn_type @Op.11 [concrete]
// CHECK:STDOUT:   %Op.b9c47d.11: %Op.type.0b8387.11 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.483: <witness> = custom_witness (%Int64, %Op.b9c47d.11), @AddWith.1, @AddWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %facet_value.706: %facet_type.002 = facet_value %Int64, (%custom_witness.a98c63.3, %custom_witness.483) [concrete]
// CHECK:STDOUT:   %pattern_type.429: type = pattern_type %facet_type.002 [concrete]
// CHECK:STDOUT:   %facet_type.a40: type = facet_type <@Destroy & @SubWith.1, @SubWith.1(%Int32) where %impl.elem0.e47 = %Int64> [concrete]
// CHECK:STDOUT:   %Op.type.0b8387.12: type = fn_type @Op.12 [concrete]
// CHECK:STDOUT:   %Op.b9c47d.12: %Op.type.0b8387.12 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.095: <witness> = custom_witness (%Int64, %Op.b9c47d.12), @SubWith.1, @SubWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %facet_value.1d6: %facet_type.a40 = facet_value %Int64, (%custom_witness.a98c63.3, %custom_witness.095) [concrete]
// CHECK:STDOUT:   %pattern_type.59c: type = pattern_type %facet_type.a40 [concrete]
// CHECK:STDOUT:   %facet_type.ac8: type = facet_type <@Destroy & @MulWith.1, @MulWith.1(%Int32) where %impl.elem0.cab = %Int64> [concrete]
// CHECK:STDOUT:   %Op.type.0b8387.13: type = fn_type @Op.13 [concrete]
// CHECK:STDOUT:   %Op.b9c47d.13: %Op.type.0b8387.13 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.f6e: <witness> = custom_witness (%Int64, %Op.b9c47d.13), @MulWith.1, @MulWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %facet_value.3ed: %facet_type.ac8 = facet_value %Int64, (%custom_witness.a98c63.3, %custom_witness.f6e) [concrete]
// CHECK:STDOUT:   %pattern_type.5de: type = pattern_type %facet_type.ac8 [concrete]
// CHECK:STDOUT:   %facet_type.b8e: type = facet_type <@Destroy & @DivWith.1, @DivWith.1(%Int32) where %impl.elem0.7a0 = %Int64> [concrete]
// CHECK:STDOUT:   %Op.type.0b8387.14: type = fn_type @Op.14 [concrete]
// CHECK:STDOUT:   %Op.b9c47d.14: %Op.type.0b8387.14 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.4e5: <witness> = custom_witness (%Int64, %Op.b9c47d.14), @DivWith.1, @DivWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %facet_value.4d6: %facet_type.b8e = facet_value %Int64, (%custom_witness.a98c63.3, %custom_witness.4e5) [concrete]
// CHECK:STDOUT:   %pattern_type.435: type = pattern_type %facet_type.b8e [concrete]
// CHECK:STDOUT:   %facet_type.dbb: type = facet_type <@Destroy & @ModWith.1, @ModWith.1(%Int32) where %impl.elem0.288 = %Int64> [concrete]
// CHECK:STDOUT:   %Op.type.0b8387.15: type = fn_type @Op.15 [concrete]
// CHECK:STDOUT:   %Op.b9c47d.15: %Op.type.0b8387.15 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.99b: <witness> = custom_witness (%Int64, %Op.b9c47d.15), @ModWith.1, @ModWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %facet_value.8fa: %facet_type.dbb = facet_value %Int64, (%custom_witness.a98c63.3, %custom_witness.99b) [concrete]
// CHECK:STDOUT:   %pattern_type.4c3: type = pattern_type %facet_type.dbb [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.2bb: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.3de) [concrete]
// CHECK:STDOUT:   %.dda: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.2bb, %Destroy.facet.3de [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.a62: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.d87) [concrete]
// CHECK:STDOUT:   %.e4f: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.a62, %Destroy.facet.d87 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.e80: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.1b8) [concrete]
// CHECK:STDOUT:   %.154: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.e80, %Destroy.facet.1b8 [concrete]
// CHECK:STDOUT:   %U.67d: type = symbolic_binding U, 0 [symbolic]
// CHECK:STDOUT:   %AddAssignWith.type.f7e94b.1: type = facet_type <@AddAssignWith.1, @AddAssignWith.1(%Other)> [symbolic]
// CHECK:STDOUT:   %Self.c9ea37.1: %AddAssignWith.type.f7e94b.1 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %AddAssignWith.assoc_type.d9654f.1: type = assoc_entity_type @AddAssignWith.1, @AddAssignWith.1(%Other) [symbolic]
// CHECK:STDOUT:   %AddAssignWith.WithSelf.Op.type.86ac33.1: type = fn_type @AddAssignWith.WithSelf.Op, @AddAssignWith.WithSelf(%Other, %Self.c9ea37.1) [symbolic]
// CHECK:STDOUT:   %AddAssignWith.WithSelf.Op.31f15f.1: %AddAssignWith.WithSelf.Op.type.86ac33.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %AddAssignWith.type.f7e94b.2: type = facet_type <@AddAssignWith.1, @AddAssignWith.1(%U.67d)> [symbolic]
// CHECK:STDOUT:   %pattern_type.323: type = pattern_type %AddAssignWith.type.f7e94b.2 [symbolic]
// CHECK:STDOUT:   %T.c9e: %AddAssignWith.type.f7e94b.2 = symbolic_binding T, 1 [symbolic]
// CHECK:STDOUT:   %AddAssignWith.assoc_type.d9654f.2: type = assoc_entity_type @AddAssignWith.1, @AddAssignWith.1(%U.67d) [symbolic]
// CHECK:STDOUT:   %assoc0.beca96.2: %AddAssignWith.assoc_type.d9654f.2 = assoc_entity element0, imports.%Core.import_ref.62c [symbolic]
// CHECK:STDOUT:   %T.as_type.82b: type = facet_access_type %T.c9e [symbolic]
// CHECK:STDOUT:   %pattern_type.d713dc.2: type = pattern_type %T.as_type.82b [symbolic]
// CHECK:STDOUT:   %pattern_type.51d1c4.4: type = pattern_type %U.67d [symbolic]
// CHECK:STDOUT:   %require_complete.e50: <witness> = require_complete_type %AddAssignWith.type.f7e94b.2 [symbolic]
// CHECK:STDOUT:   %assoc0.1e6: %AddAssignWith.assoc_type.d9654f.1 = assoc_entity element0, imports.%Core.import_ref.970 [symbolic]
// CHECK:STDOUT:   %AddAssignWith.lookup_impl_witness: <witness> = lookup_impl_witness %T.c9e, @AddAssignWith.1, @AddAssignWith.1(%U.67d) [symbolic]
// CHECK:STDOUT:   %AddAssignWith.WithSelf.Op.type.86ac33.3: type = fn_type @AddAssignWith.WithSelf.Op, @AddAssignWith.WithSelf(%U.67d, %T.c9e) [symbolic]
// CHECK:STDOUT:   %.1da: type = fn_type_with_self_type %AddAssignWith.WithSelf.Op.type.86ac33.3, %T.c9e [symbolic]
// CHECK:STDOUT:   %impl.elem0.28f: %.1da = impl_witness_access %AddAssignWith.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.c19: <specific function> = specific_impl_function %impl.elem0.28f, @AddAssignWith.WithSelf.Op(%U.67d, %T.c9e) [symbolic]
// CHECK:STDOUT:   %SubAssignWith.type.591a5f.1: type = facet_type <@SubAssignWith.1, @SubAssignWith.1(%Other)> [symbolic]
// CHECK:STDOUT:   %Self.a96693.1: %SubAssignWith.type.591a5f.1 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %SubAssignWith.assoc_type.efddf3.1: type = assoc_entity_type @SubAssignWith.1, @SubAssignWith.1(%Other) [symbolic]
// CHECK:STDOUT:   %SubAssignWith.WithSelf.Op.type.839113.1: type = fn_type @SubAssignWith.WithSelf.Op, @SubAssignWith.WithSelf(%Other, %Self.a96693.1) [symbolic]
// CHECK:STDOUT:   %SubAssignWith.WithSelf.Op.2286e8.1: %SubAssignWith.WithSelf.Op.type.839113.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %SubAssignWith.type.591a5f.2: type = facet_type <@SubAssignWith.1, @SubAssignWith.1(%U.67d)> [symbolic]
// CHECK:STDOUT:   %pattern_type.46c: type = pattern_type %SubAssignWith.type.591a5f.2 [symbolic]
// CHECK:STDOUT:   %T.a96: %SubAssignWith.type.591a5f.2 = symbolic_binding T, 1 [symbolic]
// CHECK:STDOUT:   %SubAssignWith.assoc_type.efddf3.2: type = assoc_entity_type @SubAssignWith.1, @SubAssignWith.1(%U.67d) [symbolic]
// CHECK:STDOUT:   %assoc0.382bd4.2: %SubAssignWith.assoc_type.efddf3.2 = assoc_entity element0, imports.%Core.import_ref.e50 [symbolic]
// CHECK:STDOUT:   %T.as_type.765: type = facet_access_type %T.a96 [symbolic]
// CHECK:STDOUT:   %pattern_type.f40fa3.2: type = pattern_type %T.as_type.765 [symbolic]
// CHECK:STDOUT:   %require_complete.b28: <witness> = require_complete_type %SubAssignWith.type.591a5f.2 [symbolic]
// CHECK:STDOUT:   %assoc0.936: %SubAssignWith.assoc_type.efddf3.1 = assoc_entity element0, imports.%Core.import_ref.7b0 [symbolic]
// CHECK:STDOUT:   %SubAssignWith.lookup_impl_witness: <witness> = lookup_impl_witness %T.a96, @SubAssignWith.1, @SubAssignWith.1(%U.67d) [symbolic]
// CHECK:STDOUT:   %SubAssignWith.WithSelf.Op.type.839113.3: type = fn_type @SubAssignWith.WithSelf.Op, @SubAssignWith.WithSelf(%U.67d, %T.a96) [symbolic]
// CHECK:STDOUT:   %.3df: type = fn_type_with_self_type %SubAssignWith.WithSelf.Op.type.839113.3, %T.a96 [symbolic]
// CHECK:STDOUT:   %impl.elem0.dac: %.3df = impl_witness_access %SubAssignWith.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.2f2: <specific function> = specific_impl_function %impl.elem0.dac, @SubAssignWith.WithSelf.Op(%U.67d, %T.a96) [symbolic]
// CHECK:STDOUT:   %MulAssignWith.type.b9aa5c.1: type = facet_type <@MulAssignWith.1, @MulAssignWith.1(%Other)> [symbolic]
// CHECK:STDOUT:   %Self.1ab5ec.1: %MulAssignWith.type.b9aa5c.1 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %MulAssignWith.assoc_type.153441.1: type = assoc_entity_type @MulAssignWith.1, @MulAssignWith.1(%Other) [symbolic]
// CHECK:STDOUT:   %MulAssignWith.WithSelf.Op.type.44adac.1: type = fn_type @MulAssignWith.WithSelf.Op, @MulAssignWith.WithSelf(%Other, %Self.1ab5ec.1) [symbolic]
// CHECK:STDOUT:   %MulAssignWith.WithSelf.Op.f87570.1: %MulAssignWith.WithSelf.Op.type.44adac.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %MulAssignWith.type.b9aa5c.2: type = facet_type <@MulAssignWith.1, @MulAssignWith.1(%U.67d)> [symbolic]
// CHECK:STDOUT:   %pattern_type.a30: type = pattern_type %MulAssignWith.type.b9aa5c.2 [symbolic]
// CHECK:STDOUT:   %T.1ab: %MulAssignWith.type.b9aa5c.2 = symbolic_binding T, 1 [symbolic]
// CHECK:STDOUT:   %MulAssignWith.assoc_type.153441.2: type = assoc_entity_type @MulAssignWith.1, @MulAssignWith.1(%U.67d) [symbolic]
// CHECK:STDOUT:   %assoc0.54a880.2: %MulAssignWith.assoc_type.153441.2 = assoc_entity element0, imports.%Core.import_ref.763 [symbolic]
// CHECK:STDOUT:   %T.as_type.e75: type = facet_access_type %T.1ab [symbolic]
// CHECK:STDOUT:   %pattern_type.07a4e0.2: type = pattern_type %T.as_type.e75 [symbolic]
// CHECK:STDOUT:   %require_complete.ea2: <witness> = require_complete_type %MulAssignWith.type.b9aa5c.2 [symbolic]
// CHECK:STDOUT:   %assoc0.862: %MulAssignWith.assoc_type.153441.1 = assoc_entity element0, imports.%Core.import_ref.634 [symbolic]
// CHECK:STDOUT:   %MulAssignWith.lookup_impl_witness: <witness> = lookup_impl_witness %T.1ab, @MulAssignWith.1, @MulAssignWith.1(%U.67d) [symbolic]
// CHECK:STDOUT:   %MulAssignWith.WithSelf.Op.type.44adac.3: type = fn_type @MulAssignWith.WithSelf.Op, @MulAssignWith.WithSelf(%U.67d, %T.1ab) [symbolic]
// CHECK:STDOUT:   %.42e: type = fn_type_with_self_type %MulAssignWith.WithSelf.Op.type.44adac.3, %T.1ab [symbolic]
// CHECK:STDOUT:   %impl.elem0.822: %.42e = impl_witness_access %MulAssignWith.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.cf3: <specific function> = specific_impl_function %impl.elem0.822, @MulAssignWith.WithSelf.Op(%U.67d, %T.1ab) [symbolic]
// CHECK:STDOUT:   %DivAssignWith.type.c97ac9.1: type = facet_type <@DivAssignWith.1, @DivAssignWith.1(%Other)> [symbolic]
// CHECK:STDOUT:   %Self.43908b.1: %DivAssignWith.type.c97ac9.1 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %DivAssignWith.assoc_type.c30cb0.1: type = assoc_entity_type @DivAssignWith.1, @DivAssignWith.1(%Other) [symbolic]
// CHECK:STDOUT:   %DivAssignWith.WithSelf.Op.type.8e0740.1: type = fn_type @DivAssignWith.WithSelf.Op, @DivAssignWith.WithSelf(%Other, %Self.43908b.1) [symbolic]
// CHECK:STDOUT:   %DivAssignWith.WithSelf.Op.0bd7e1.1: %DivAssignWith.WithSelf.Op.type.8e0740.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %DivAssignWith.type.c97ac9.2: type = facet_type <@DivAssignWith.1, @DivAssignWith.1(%U.67d)> [symbolic]
// CHECK:STDOUT:   %pattern_type.fa6: type = pattern_type %DivAssignWith.type.c97ac9.2 [symbolic]
// CHECK:STDOUT:   %T.439: %DivAssignWith.type.c97ac9.2 = symbolic_binding T, 1 [symbolic]
// CHECK:STDOUT:   %DivAssignWith.assoc_type.c30cb0.2: type = assoc_entity_type @DivAssignWith.1, @DivAssignWith.1(%U.67d) [symbolic]
// CHECK:STDOUT:   %assoc0.bd6a2d.2: %DivAssignWith.assoc_type.c30cb0.2 = assoc_entity element0, imports.%Core.import_ref.d8a [symbolic]
// CHECK:STDOUT:   %T.as_type.5d7: type = facet_access_type %T.439 [symbolic]
// CHECK:STDOUT:   %pattern_type.12d907.2: type = pattern_type %T.as_type.5d7 [symbolic]
// CHECK:STDOUT:   %require_complete.3f4: <witness> = require_complete_type %DivAssignWith.type.c97ac9.2 [symbolic]
// CHECK:STDOUT:   %assoc0.086: %DivAssignWith.assoc_type.c30cb0.1 = assoc_entity element0, imports.%Core.import_ref.b97 [symbolic]
// CHECK:STDOUT:   %DivAssignWith.lookup_impl_witness: <witness> = lookup_impl_witness %T.439, @DivAssignWith.1, @DivAssignWith.1(%U.67d) [symbolic]
// CHECK:STDOUT:   %DivAssignWith.WithSelf.Op.type.8e0740.3: type = fn_type @DivAssignWith.WithSelf.Op, @DivAssignWith.WithSelf(%U.67d, %T.439) [symbolic]
// CHECK:STDOUT:   %.28f: type = fn_type_with_self_type %DivAssignWith.WithSelf.Op.type.8e0740.3, %T.439 [symbolic]
// CHECK:STDOUT:   %impl.elem0.f6e: %.28f = impl_witness_access %DivAssignWith.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.5bc: <specific function> = specific_impl_function %impl.elem0.f6e, @DivAssignWith.WithSelf.Op(%U.67d, %T.439) [symbolic]
// CHECK:STDOUT:   %ModAssignWith.type.8faa47.1: type = facet_type <@ModAssignWith.1, @ModAssignWith.1(%Other)> [symbolic]
// CHECK:STDOUT:   %Self.da03f6.1: %ModAssignWith.type.8faa47.1 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %ModAssignWith.assoc_type.0ff0e5.1: type = assoc_entity_type @ModAssignWith.1, @ModAssignWith.1(%Other) [symbolic]
// CHECK:STDOUT:   %ModAssignWith.WithSelf.Op.type.292942.1: type = fn_type @ModAssignWith.WithSelf.Op, @ModAssignWith.WithSelf(%Other, %Self.da03f6.1) [symbolic]
// CHECK:STDOUT:   %ModAssignWith.WithSelf.Op.84faf3.1: %ModAssignWith.WithSelf.Op.type.292942.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %ModAssignWith.type.8faa47.2: type = facet_type <@ModAssignWith.1, @ModAssignWith.1(%U.67d)> [symbolic]
// CHECK:STDOUT:   %pattern_type.383: type = pattern_type %ModAssignWith.type.8faa47.2 [symbolic]
// CHECK:STDOUT:   %T.da0: %ModAssignWith.type.8faa47.2 = symbolic_binding T, 1 [symbolic]
// CHECK:STDOUT:   %ModAssignWith.assoc_type.0ff0e5.2: type = assoc_entity_type @ModAssignWith.1, @ModAssignWith.1(%U.67d) [symbolic]
// CHECK:STDOUT:   %assoc0.3824c9.2: %ModAssignWith.assoc_type.0ff0e5.2 = assoc_entity element0, imports.%Core.import_ref.aee3 [symbolic]
// CHECK:STDOUT:   %T.as_type.4cb: type = facet_access_type %T.da0 [symbolic]
// CHECK:STDOUT:   %pattern_type.7b701e.2: type = pattern_type %T.as_type.4cb [symbolic]
// CHECK:STDOUT:   %require_complete.9bf: <witness> = require_complete_type %ModAssignWith.type.8faa47.2 [symbolic]
// CHECK:STDOUT:   %assoc0.715: %ModAssignWith.assoc_type.0ff0e5.1 = assoc_entity element0, imports.%Core.import_ref.248 [symbolic]
// CHECK:STDOUT:   %ModAssignWith.lookup_impl_witness: <witness> = lookup_impl_witness %T.da0, @ModAssignWith.1, @ModAssignWith.1(%U.67d) [symbolic]
// CHECK:STDOUT:   %ModAssignWith.WithSelf.Op.type.292942.3: type = fn_type @ModAssignWith.WithSelf.Op, @ModAssignWith.WithSelf(%U.67d, %T.da0) [symbolic]
// CHECK:STDOUT:   %.acb: type = fn_type_with_self_type %ModAssignWith.WithSelf.Op.type.292942.3, %T.da0 [symbolic]
// CHECK:STDOUT:   %impl.elem0.f9d: %.acb = impl_witness_access %ModAssignWith.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.802: <specific function> = specific_impl_function %impl.elem0.f9d, @ModAssignWith.WithSelf.Op(%U.67d, %T.da0) [symbolic]
// CHECK:STDOUT:   %AddAssignWith.type.35f: type = facet_type <@AddAssignWith.1, @AddAssignWith.1(%i16)> [concrete]
// CHECK:STDOUT:   %AddAssignWith.assoc_type.3c9: type = assoc_entity_type @AddAssignWith.1, @AddAssignWith.1(%i16) [concrete]
// CHECK:STDOUT:   %assoc0.d69: %AddAssignWith.assoc_type.3c9 = assoc_entity element0, imports.%Core.import_ref.62c [concrete]
// CHECK:STDOUT:   %Int16.Op.type.43d658.13: type = fn_type @Int16.Op.13 [concrete]
// CHECK:STDOUT:   %Int16.Op.cc70d5.13: %Int16.Op.type.43d658.13 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.364: <witness> = custom_witness (%Int16.Op.cc70d5.13), @AddAssignWith.1, @AddAssignWith.1(%i16) [concrete]
// CHECK:STDOUT:   %AddAssignWith.facet.1b3: %AddAssignWith.type.35f = facet_value %Int16, (%custom_witness.364) [concrete]
// CHECK:STDOUT:   %pattern_type.25b: type = pattern_type %AddAssignWith.type.35f [concrete]
// CHECK:STDOUT:   %SubAssignWith.type.121: type = facet_type <@SubAssignWith.1, @SubAssignWith.1(%i16)> [concrete]
// CHECK:STDOUT:   %SubAssignWith.assoc_type.5f8: type = assoc_entity_type @SubAssignWith.1, @SubAssignWith.1(%i16) [concrete]
// CHECK:STDOUT:   %assoc0.ce1: %SubAssignWith.assoc_type.5f8 = assoc_entity element0, imports.%Core.import_ref.e50 [concrete]
// CHECK:STDOUT:   %Int16.Op.type.43d658.14: type = fn_type @Int16.Op.14 [concrete]
// CHECK:STDOUT:   %Int16.Op.cc70d5.14: %Int16.Op.type.43d658.14 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.4e8: <witness> = custom_witness (%Int16.Op.cc70d5.14), @SubAssignWith.1, @SubAssignWith.1(%i16) [concrete]
// CHECK:STDOUT:   %SubAssignWith.facet.45a: %SubAssignWith.type.121 = facet_value %Int16, (%custom_witness.4e8) [concrete]
// CHECK:STDOUT:   %pattern_type.1c3: type = pattern_type %SubAssignWith.type.121 [concrete]
// CHECK:STDOUT:   %MulAssignWith.type.b7e: type = facet_type <@MulAssignWith.1, @MulAssignWith.1(%i16)> [concrete]
// CHECK:STDOUT:   %MulAssignWith.assoc_type.551: type = assoc_entity_type @MulAssignWith.1, @MulAssignWith.1(%i16) [concrete]
// CHECK:STDOUT:   %assoc0.75e: %MulAssignWith.assoc_type.551 = assoc_entity element0, imports.%Core.import_ref.763 [concrete]
// CHECK:STDOUT:   %Int16.Op.type.43d658.15: type = fn_type @Int16.Op.15 [concrete]
// CHECK:STDOUT:   %Int16.Op.cc70d5.15: %Int16.Op.type.43d658.15 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b74: <witness> = custom_witness (%Int16.Op.cc70d5.15), @MulAssignWith.1, @MulAssignWith.1(%i16) [concrete]
// CHECK:STDOUT:   %MulAssignWith.facet.805: %MulAssignWith.type.b7e = facet_value %Int16, (%custom_witness.b74) [concrete]
// CHECK:STDOUT:   %pattern_type.ebd: type = pattern_type %MulAssignWith.type.b7e [concrete]
// CHECK:STDOUT:   %DivAssignWith.type.0d6: type = facet_type <@DivAssignWith.1, @DivAssignWith.1(%i16)> [concrete]
// CHECK:STDOUT:   %DivAssignWith.assoc_type.01d: type = assoc_entity_type @DivAssignWith.1, @DivAssignWith.1(%i16) [concrete]
// CHECK:STDOUT:   %assoc0.f56: %DivAssignWith.assoc_type.01d = assoc_entity element0, imports.%Core.import_ref.d8a [concrete]
// CHECK:STDOUT:   %Int16.Op.type.43d658.16: type = fn_type @Int16.Op.16 [concrete]
// CHECK:STDOUT:   %Int16.Op.cc70d5.16: %Int16.Op.type.43d658.16 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.8b6: <witness> = custom_witness (%Int16.Op.cc70d5.16), @DivAssignWith.1, @DivAssignWith.1(%i16) [concrete]
// CHECK:STDOUT:   %DivAssignWith.facet.ac9: %DivAssignWith.type.0d6 = facet_value %Int16, (%custom_witness.8b6) [concrete]
// CHECK:STDOUT:   %pattern_type.1f58: type = pattern_type %DivAssignWith.type.0d6 [concrete]
// CHECK:STDOUT:   %ModAssignWith.type.d0f: type = facet_type <@ModAssignWith.1, @ModAssignWith.1(%i16)> [concrete]
// CHECK:STDOUT:   %ModAssignWith.assoc_type.9c3: type = assoc_entity_type @ModAssignWith.1, @ModAssignWith.1(%i16) [concrete]
// CHECK:STDOUT:   %assoc0.e07: %ModAssignWith.assoc_type.9c3 = assoc_entity element0, imports.%Core.import_ref.aee3 [concrete]
// CHECK:STDOUT:   %Int16.Op.type.43d658.17: type = fn_type @Int16.Op.17 [concrete]
// CHECK:STDOUT:   %Int16.Op.cc70d5.17: %Int16.Op.type.43d658.17 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.e6d: <witness> = custom_witness (%Int16.Op.cc70d5.17), @ModAssignWith.1, @ModAssignWith.1(%i16) [concrete]
// CHECK:STDOUT:   %ModAssignWith.facet.350: %ModAssignWith.type.d0f = facet_value %Int16, (%custom_witness.e6d) [concrete]
// CHECK:STDOUT:   %pattern_type.e7a: type = pattern_type %ModAssignWith.type.d0f [concrete]
// CHECK:STDOUT:   %AddAssignWith.type.3e8: type = facet_type <@AddAssignWith.1, @AddAssignWith.1(%Int32)> [concrete]
// CHECK:STDOUT:   %AddAssignWith.assoc_type.115: type = assoc_entity_type @AddAssignWith.1, @AddAssignWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %assoc0.8b4: %AddAssignWith.assoc_type.115 = assoc_entity element0, imports.%Core.import_ref.62c [concrete]
// CHECK:STDOUT:   %Int32.Op.type.43d658.12: type = fn_type @Int32.Op.12 [concrete]
// CHECK:STDOUT:   %Int32.Op.cc70d5.12: %Int32.Op.type.43d658.12 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.ec4e63.1: <witness> = custom_witness (%Int32.Op.cc70d5.12), @AddAssignWith.1, @AddAssignWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %AddAssignWith.facet.610: %AddAssignWith.type.3e8 = facet_value %Int32, (%custom_witness.ec4e63.1) [concrete]
// CHECK:STDOUT:   %pattern_type.ae6: type = pattern_type %AddAssignWith.type.3e8 [concrete]
// CHECK:STDOUT:   %SubAssignWith.type.678: type = facet_type <@SubAssignWith.1, @SubAssignWith.1(%Int32)> [concrete]
// CHECK:STDOUT:   %SubAssignWith.assoc_type.53a: type = assoc_entity_type @SubAssignWith.1, @SubAssignWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %assoc0.60e: %SubAssignWith.assoc_type.53a = assoc_entity element0, imports.%Core.import_ref.e50 [concrete]
// CHECK:STDOUT:   %Int32.Op.type.43d658.14: type = fn_type @Int32.Op.14 [concrete]
// CHECK:STDOUT:   %Int32.Op.cc70d5.14: %Int32.Op.type.43d658.14 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.75ca2a.1: <witness> = custom_witness (%Int32.Op.cc70d5.14), @SubAssignWith.1, @SubAssignWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %SubAssignWith.facet.fb2: %SubAssignWith.type.678 = facet_value %Int32, (%custom_witness.75ca2a.1) [concrete]
// CHECK:STDOUT:   %pattern_type.470: type = pattern_type %SubAssignWith.type.678 [concrete]
// CHECK:STDOUT:   %MulAssignWith.type.011: type = facet_type <@MulAssignWith.1, @MulAssignWith.1(%Int32)> [concrete]
// CHECK:STDOUT:   %MulAssignWith.assoc_type.f6b: type = assoc_entity_type @MulAssignWith.1, @MulAssignWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %assoc0.1a2: %MulAssignWith.assoc_type.f6b = assoc_entity element0, imports.%Core.import_ref.763 [concrete]
// CHECK:STDOUT:   %Int32.Op.type.43d658.15: type = fn_type @Int32.Op.15 [concrete]
// CHECK:STDOUT:   %Int32.Op.cc70d5.15: %Int32.Op.type.43d658.15 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.7d92f3.1: <witness> = custom_witness (%Int32.Op.cc70d5.15), @MulAssignWith.1, @MulAssignWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %MulAssignWith.facet.8f5: %MulAssignWith.type.011 = facet_value %Int32, (%custom_witness.7d92f3.1) [concrete]
// CHECK:STDOUT:   %pattern_type.c27: type = pattern_type %MulAssignWith.type.011 [concrete]
// CHECK:STDOUT:   %DivAssignWith.type.487: type = facet_type <@DivAssignWith.1, @DivAssignWith.1(%Int32)> [concrete]
// CHECK:STDOUT:   %DivAssignWith.assoc_type.047: type = assoc_entity_type @DivAssignWith.1, @DivAssignWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %assoc0.cd7: %DivAssignWith.assoc_type.047 = assoc_entity element0, imports.%Core.import_ref.d8a [concrete]
// CHECK:STDOUT:   %Int32.Op.type.43d658.16: type = fn_type @Int32.Op.16 [concrete]
// CHECK:STDOUT:   %Int32.Op.cc70d5.16: %Int32.Op.type.43d658.16 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.cdc44b.1: <witness> = custom_witness (%Int32.Op.cc70d5.16), @DivAssignWith.1, @DivAssignWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %DivAssignWith.facet.3bc: %DivAssignWith.type.487 = facet_value %Int32, (%custom_witness.cdc44b.1) [concrete]
// CHECK:STDOUT:   %pattern_type.492: type = pattern_type %DivAssignWith.type.487 [concrete]
// CHECK:STDOUT:   %ModAssignWith.type.499: type = facet_type <@ModAssignWith.1, @ModAssignWith.1(%Int32)> [concrete]
// CHECK:STDOUT:   %ModAssignWith.assoc_type.e0e: type = assoc_entity_type @ModAssignWith.1, @ModAssignWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %assoc0.928: %ModAssignWith.assoc_type.e0e = assoc_entity element0, imports.%Core.import_ref.aee3 [concrete]
// CHECK:STDOUT:   %Int32.Op.type.43d658.17: type = fn_type @Int32.Op.17 [concrete]
// CHECK:STDOUT:   %Int32.Op.cc70d5.17: %Int32.Op.type.43d658.17 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.ec99b7.1: <witness> = custom_witness (%Int32.Op.cc70d5.17), @ModAssignWith.1, @ModAssignWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %ModAssignWith.facet.b73: %ModAssignWith.type.499 = facet_value %Int32, (%custom_witness.ec99b7.1) [concrete]
// CHECK:STDOUT:   %pattern_type.f79: type = pattern_type %ModAssignWith.type.499 [concrete]
// CHECK:STDOUT:   %Int64.Op.type.43d658.2: type = fn_type @Int64.Op.2 [concrete]
// CHECK:STDOUT:   %Int64.Op.cc70d5.2: %Int64.Op.type.43d658.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.ec4e63.2: <witness> = custom_witness (%Int64.Op.cc70d5.2), @AddAssignWith.1, @AddAssignWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %AddAssignWith.facet.080: %AddAssignWith.type.3e8 = facet_value %Int64, (%custom_witness.ec4e63.2) [concrete]
// CHECK:STDOUT:   %Int64.Op.type.43d658.4: type = fn_type @Int64.Op.4 [concrete]
// CHECK:STDOUT:   %Int64.Op.cc70d5.4: %Int64.Op.type.43d658.4 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.75ca2a.2: <witness> = custom_witness (%Int64.Op.cc70d5.4), @SubAssignWith.1, @SubAssignWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %SubAssignWith.facet.8bc: %SubAssignWith.type.678 = facet_value %Int64, (%custom_witness.75ca2a.2) [concrete]
// CHECK:STDOUT:   %Int64.Op.type.43d658.5: type = fn_type @Int64.Op.5 [concrete]
// CHECK:STDOUT:   %Int64.Op.cc70d5.5: %Int64.Op.type.43d658.5 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.7d92f3.2: <witness> = custom_witness (%Int64.Op.cc70d5.5), @MulAssignWith.1, @MulAssignWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %MulAssignWith.facet.990: %MulAssignWith.type.011 = facet_value %Int64, (%custom_witness.7d92f3.2) [concrete]
// CHECK:STDOUT:   %Int64.Op.type.43d658.6: type = fn_type @Int64.Op.6 [concrete]
// CHECK:STDOUT:   %Int64.Op.cc70d5.6: %Int64.Op.type.43d658.6 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.cdc44b.2: <witness> = custom_witness (%Int64.Op.cc70d5.6), @DivAssignWith.1, @DivAssignWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %DivAssignWith.facet.3eb: %DivAssignWith.type.487 = facet_value %Int64, (%custom_witness.cdc44b.2) [concrete]
// CHECK:STDOUT:   %Int64.Op.type.43d658.7: type = fn_type @Int64.Op.7 [concrete]
// CHECK:STDOUT:   %Int64.Op.cc70d5.7: %Int64.Op.type.43d658.7 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.ec99b7.2: <witness> = custom_witness (%Int64.Op.cc70d5.7), @ModAssignWith.1, @ModAssignWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %ModAssignWith.facet.2d7: %ModAssignWith.type.499 = facet_value %Int64, (%custom_witness.ec99b7.2) [concrete]
// CHECK:STDOUT:   %Inc.type: type = facet_type <@Inc> [concrete]
// CHECK:STDOUT:   %T.c06: %Inc.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.4cc: type = facet_access_type %T.c06 [symbolic]
// CHECK:STDOUT:   %pattern_type.55fce5.2: type = pattern_type %T.as_type.4cc [symbolic]
// CHECK:STDOUT:   %Inc.lookup_impl_witness: <witness> = lookup_impl_witness %T.c06, @Inc [symbolic]
// CHECK:STDOUT:   %Inc.WithSelf.Op.type.1a5b76.2: type = fn_type @Inc.WithSelf.Op, @Inc.WithSelf(%T.c06) [symbolic]
// CHECK:STDOUT:   %.bdc: type = fn_type_with_self_type %Inc.WithSelf.Op.type.1a5b76.2, %T.c06 [symbolic]
// CHECK:STDOUT:   %impl.elem0.a96: %.bdc = impl_witness_access %Inc.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.d8c: <specific function> = specific_impl_function %impl.elem0.a96, @Inc.WithSelf.Op(%T.c06) [symbolic]
// CHECK:STDOUT:   %Dec.type: type = facet_type <@Dec> [concrete]
// CHECK:STDOUT:   %T.d9e: %Dec.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.915: type = facet_access_type %T.d9e [symbolic]
// CHECK:STDOUT:   %pattern_type.317946.2: type = pattern_type %T.as_type.915 [symbolic]
// CHECK:STDOUT:   %Dec.lookup_impl_witness: <witness> = lookup_impl_witness %T.d9e, @Dec [symbolic]
// CHECK:STDOUT:   %Dec.WithSelf.Op.type.c8aa8e.2: type = fn_type @Dec.WithSelf.Op, @Dec.WithSelf(%T.d9e) [symbolic]
// CHECK:STDOUT:   %.842: type = fn_type_with_self_type %Dec.WithSelf.Op.type.c8aa8e.2, %T.d9e [symbolic]
// CHECK:STDOUT:   %impl.elem0.e17: %.842 = impl_witness_access %Dec.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.56c: <specific function> = specific_impl_function %impl.elem0.e17, @Dec.WithSelf.Op(%T.d9e) [symbolic]
// CHECK:STDOUT:   %Negate.type: type = facet_type <@Negate> [concrete]
// CHECK:STDOUT:   %facet_type.d7a: type = facet_type <@Destroy & @Negate> [concrete]
// CHECK:STDOUT:   %.Self.d26: %facet_type.d7a = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %Negate.lookup_impl_witness.53f: <witness> = lookup_impl_witness %.Self.d26, @Negate [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.b2f: type = impl_witness_access %Negate.lookup_impl_witness.53f, element0 [symbolic_self]
// CHECK:STDOUT:   %facet_type.fb9: type = facet_type <@Destroy & @Negate where %impl.elem0.b2f = %Result.as_type> [symbolic]
// CHECK:STDOUT:   %pattern_type.301: type = pattern_type %facet_type.fb9 [symbolic]
// CHECK:STDOUT:   %T.0e9: %facet_type.fb9 = symbolic_binding T, 1 [symbolic]
// CHECK:STDOUT:   %T.as_type.b01: type = facet_access_type %T.0e9 [symbolic]
// CHECK:STDOUT:   %pattern_type.5e6: type = pattern_type %T.as_type.b01 [symbolic]
// CHECK:STDOUT:   %Negate.lookup_impl_witness.5d0: <witness> = lookup_impl_witness %T.0e9, @Negate [symbolic]
// CHECK:STDOUT:   %Negate.facet.7e7: %Negate.type = facet_value %T.as_type.b01, (%Negate.lookup_impl_witness.5d0) [symbolic]
// CHECK:STDOUT:   %Negate.WithSelf.Op.type.26a: type = fn_type @Negate.WithSelf.Op, @Negate.WithSelf(%Negate.facet.7e7) [symbolic]
// CHECK:STDOUT:   %.0fd: type = fn_type_with_self_type %Negate.WithSelf.Op.type.26a, %Negate.facet.7e7 [symbolic]
// CHECK:STDOUT:   %impl.elem1.b4a: %.0fd = impl_witness_access %Negate.lookup_impl_witness.5d0, element1 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.037: <specific function> = specific_impl_function %impl.elem1.b4a, @Negate.WithSelf.Op(%Negate.facet.7e7) [symbolic]
// CHECK:STDOUT:   %Int32.Op.type.43d658.18: type = fn_type @Int32.Op.18 [concrete]
// CHECK:STDOUT:   %Int32.Op.cc70d5.18: %Int32.Op.type.43d658.18 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.ef4: <witness> = custom_witness (%Int32.Op.cc70d5.18), @Inc [concrete]
// CHECK:STDOUT:   %Inc.facet.ace: %Inc.type = facet_value %Int32, (%custom_witness.ef4) [concrete]
// CHECK:STDOUT:   %Int32.Op.type.43d658.19: type = fn_type @Int32.Op.19 [concrete]
// CHECK:STDOUT:   %Int32.Op.cc70d5.19: %Int32.Op.type.43d658.19 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.7c5: <witness> = custom_witness (%Int32.Op.cc70d5.19), @Dec [concrete]
// CHECK:STDOUT:   %Dec.facet.be4: %Dec.type = facet_value %Int32, (%custom_witness.7c5) [concrete]
// CHECK:STDOUT:   %facet_type.999: type = facet_type <@Destroy & @Negate where %impl.elem0.b2f = %Int64> [concrete]
// CHECK:STDOUT:   %Int16.Op.type.43d658.18: type = fn_type @Int16.Op.18 [concrete]
// CHECK:STDOUT:   %Int16.Op.cc70d5.18: %Int16.Op.type.43d658.18 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.71b: <witness> = custom_witness (%Int64, %Int16.Op.cc70d5.18), @Negate [concrete]
// CHECK:STDOUT:   %facet_value.e43: %facet_type.999 = facet_value %Int16, (%custom_witness.a98c63.1, %custom_witness.71b) [concrete]
// CHECK:STDOUT:   %pattern_type.04d: type = pattern_type %facet_type.999 [concrete]
// CHECK:STDOUT:   %facet_type.363: type = facet_type <@Destroy & @Negate where %impl.elem0.b2f = %Int32> [concrete]
// CHECK:STDOUT:   %Int32.Op.type.43d658.20: type = fn_type @Int32.Op.20 [concrete]
// CHECK:STDOUT:   %Int32.Op.cc70d5.20: %Int32.Op.type.43d658.20 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.45a: <witness> = custom_witness (%Int32, %Int32.Op.cc70d5.20), @Negate [concrete]
// CHECK:STDOUT:   %facet_value.b3a: %facet_type.363 = facet_value %Int32, (%custom_witness.a98c63.2, %custom_witness.45a) [concrete]
// CHECK:STDOUT:   %pattern_type.4d0: type = pattern_type %facet_type.363 [concrete]
// CHECK:STDOUT:   %complete_type.386: <witness> = complete_type_witness %AddWith.type.6f4 [concrete]
// CHECK:STDOUT:   %AddWith.facet.85f: %AddWith.type.6f4 = facet_value %i16, (%custom_witness.604) [concrete]
// CHECK:STDOUT:   %AddWith.WithSelf.Op.type.09b: type = fn_type @AddWith.WithSelf.Op, @AddWith.WithSelf(%Int16, %AddWith.facet.85f) [concrete]
// CHECK:STDOUT:   %.b5b: type = fn_type_with_self_type %AddWith.WithSelf.Op.type.09b, %AddWith.facet.85f [concrete]
// CHECK:STDOUT:   %complete_type.41c: <witness> = complete_type_witness %SubWith.type.8bd [concrete]
// CHECK:STDOUT:   %SubWith.facet.1de: %SubWith.type.8bd = facet_value %i16, (%custom_witness.4c1) [concrete]
// CHECK:STDOUT:   %SubWith.WithSelf.Op.type.fc2: type = fn_type @SubWith.WithSelf.Op, @SubWith.WithSelf(%Int16, %SubWith.facet.1de) [concrete]
// CHECK:STDOUT:   %.ec9: type = fn_type_with_self_type %SubWith.WithSelf.Op.type.fc2, %SubWith.facet.1de [concrete]
// CHECK:STDOUT:   %complete_type.b16: <witness> = complete_type_witness %MulWith.type.c25 [concrete]
// CHECK:STDOUT:   %MulWith.facet.d75: %MulWith.type.c25 = facet_value %i16, (%custom_witness.72a) [concrete]
// CHECK:STDOUT:   %MulWith.WithSelf.Op.type.53a: type = fn_type @MulWith.WithSelf.Op, @MulWith.WithSelf(%Int16, %MulWith.facet.d75) [concrete]
// CHECK:STDOUT:   %.f80: type = fn_type_with_self_type %MulWith.WithSelf.Op.type.53a, %MulWith.facet.d75 [concrete]
// CHECK:STDOUT:   %complete_type.7c5: <witness> = complete_type_witness %DivWith.type.053 [concrete]
// CHECK:STDOUT:   %DivWith.facet.2be: %DivWith.type.053 = facet_value %i16, (%custom_witness.6ea) [concrete]
// CHECK:STDOUT:   %DivWith.WithSelf.Op.type.88f: type = fn_type @DivWith.WithSelf.Op, @DivWith.WithSelf(%Int16, %DivWith.facet.2be) [concrete]
// CHECK:STDOUT:   %.a93: type = fn_type_with_self_type %DivWith.WithSelf.Op.type.88f, %DivWith.facet.2be [concrete]
// CHECK:STDOUT:   %complete_type.2f8: <witness> = complete_type_witness %ModWith.type.a83 [concrete]
// CHECK:STDOUT:   %ModWith.facet.ea4: %ModWith.type.a83 = facet_value %i16, (%custom_witness.b81) [concrete]
// CHECK:STDOUT:   %ModWith.WithSelf.Op.type.6fc: type = fn_type @ModWith.WithSelf.Op, @ModWith.WithSelf(%Int16, %ModWith.facet.ea4) [concrete]
// CHECK:STDOUT:   %.3cd: type = fn_type_with_self_type %ModWith.WithSelf.Op.type.6fc, %ModWith.facet.ea4 [concrete]
// CHECK:STDOUT:   %complete_type.b45: <witness> = complete_type_witness %AddWith.type.63e [concrete]
// CHECK:STDOUT:   %AddWith.facet.cea: %AddWith.type.63e = facet_value %Int16, (%custom_witness.9f2) [concrete]
// CHECK:STDOUT:   %AddWith.WithSelf.Op.type.18e: type = fn_type @AddWith.WithSelf.Op, @AddWith.WithSelf(%i16, %AddWith.facet.cea) [concrete]
// CHECK:STDOUT:   %.1c0: type = fn_type_with_self_type %AddWith.WithSelf.Op.type.18e, %AddWith.facet.cea [concrete]
// CHECK:STDOUT:   %complete_type.edc: <witness> = complete_type_witness %SubWith.type.1db [concrete]
// CHECK:STDOUT:   %SubWith.facet.3dd: %SubWith.type.1db = facet_value %Int16, (%custom_witness.6a5) [concrete]
// CHECK:STDOUT:   %SubWith.WithSelf.Op.type.1ca: type = fn_type @SubWith.WithSelf.Op, @SubWith.WithSelf(%i16, %SubWith.facet.3dd) [concrete]
// CHECK:STDOUT:   %.f6d: type = fn_type_with_self_type %SubWith.WithSelf.Op.type.1ca, %SubWith.facet.3dd [concrete]
// CHECK:STDOUT:   %complete_type.355: <witness> = complete_type_witness %MulWith.type.c35 [concrete]
// CHECK:STDOUT:   %MulWith.facet.3b5: %MulWith.type.c35 = facet_value %Int16, (%custom_witness.35c) [concrete]
// CHECK:STDOUT:   %MulWith.WithSelf.Op.type.896: type = fn_type @MulWith.WithSelf.Op, @MulWith.WithSelf(%i16, %MulWith.facet.3b5) [concrete]
// CHECK:STDOUT:   %.ca0: type = fn_type_with_self_type %MulWith.WithSelf.Op.type.896, %MulWith.facet.3b5 [concrete]
// CHECK:STDOUT:   %complete_type.db2: <witness> = complete_type_witness %DivWith.type.086 [concrete]
// CHECK:STDOUT:   %DivWith.facet.8d2: %DivWith.type.086 = facet_value %Int16, (%custom_witness.52d) [concrete]
// CHECK:STDOUT:   %DivWith.WithSelf.Op.type.2d4: type = fn_type @DivWith.WithSelf.Op, @DivWith.WithSelf(%i16, %DivWith.facet.8d2) [concrete]
// CHECK:STDOUT:   %.cb6e: type = fn_type_with_self_type %DivWith.WithSelf.Op.type.2d4, %DivWith.facet.8d2 [concrete]
// CHECK:STDOUT:   %complete_type.b14: <witness> = complete_type_witness %ModWith.type.59b [concrete]
// CHECK:STDOUT:   %ModWith.facet.8ae: %ModWith.type.59b = facet_value %Int16, (%custom_witness.887) [concrete]
// CHECK:STDOUT:   %ModWith.WithSelf.Op.type.5f0: type = fn_type @ModWith.WithSelf.Op, @ModWith.WithSelf(%i16, %ModWith.facet.8ae) [concrete]
// CHECK:STDOUT:   %.851: type = fn_type_with_self_type %ModWith.WithSelf.Op.type.5f0, %ModWith.facet.8ae [concrete]
// CHECK:STDOUT:   %complete_type.c57: <witness> = complete_type_witness %AddWith.type.78c [concrete]
// CHECK:STDOUT:   %AddWith.facet.b3b: %AddWith.type.78c = facet_value %Int16, (%custom_witness.e46c75.1) [concrete]
// CHECK:STDOUT:   %AddWith.WithSelf.Op.type.675: type = fn_type @AddWith.WithSelf.Op, @AddWith.WithSelf(%Int32, %AddWith.facet.b3b) [concrete]
// CHECK:STDOUT:   %.615: type = fn_type_with_self_type %AddWith.WithSelf.Op.type.675, %AddWith.facet.b3b [concrete]
// CHECK:STDOUT:   %complete_type.48a: <witness> = complete_type_witness %SubWith.type.629 [concrete]
// CHECK:STDOUT:   %SubWith.facet.da1: %SubWith.type.629 = facet_value %Int16, (%custom_witness.7ffeb1.1) [concrete]
// CHECK:STDOUT:   %SubWith.WithSelf.Op.type.a0e: type = fn_type @SubWith.WithSelf.Op, @SubWith.WithSelf(%Int32, %SubWith.facet.da1) [concrete]
// CHECK:STDOUT:   %.1a2: type = fn_type_with_self_type %SubWith.WithSelf.Op.type.a0e, %SubWith.facet.da1 [concrete]
// CHECK:STDOUT:   %complete_type.6e9: <witness> = complete_type_witness %MulWith.type.3ee [concrete]
// CHECK:STDOUT:   %MulWith.facet.c1a: %MulWith.type.3ee = facet_value %Int16, (%custom_witness.68a6cf.1) [concrete]
// CHECK:STDOUT:   %MulWith.WithSelf.Op.type.b9a: type = fn_type @MulWith.WithSelf.Op, @MulWith.WithSelf(%Int32, %MulWith.facet.c1a) [concrete]
// CHECK:STDOUT:   %.f10: type = fn_type_with_self_type %MulWith.WithSelf.Op.type.b9a, %MulWith.facet.c1a [concrete]
// CHECK:STDOUT:   %complete_type.627: <witness> = complete_type_witness %DivWith.type.09b [concrete]
// CHECK:STDOUT:   %DivWith.facet.b53: %DivWith.type.09b = facet_value %Int16, (%custom_witness.2ccbdb.1) [concrete]
// CHECK:STDOUT:   %DivWith.WithSelf.Op.type.214: type = fn_type @DivWith.WithSelf.Op, @DivWith.WithSelf(%Int32, %DivWith.facet.b53) [concrete]
// CHECK:STDOUT:   %.eff: type = fn_type_with_self_type %DivWith.WithSelf.Op.type.214, %DivWith.facet.b53 [concrete]
// CHECK:STDOUT:   %complete_type.1cf: <witness> = complete_type_witness %ModWith.type.6f6 [concrete]
// CHECK:STDOUT:   %ModWith.facet.142: %ModWith.type.6f6 = facet_value %Int16, (%custom_witness.596ded.1) [concrete]
// CHECK:STDOUT:   %ModWith.WithSelf.Op.type.513: type = fn_type @ModWith.WithSelf.Op, @ModWith.WithSelf(%Int32, %ModWith.facet.142) [concrete]
// CHECK:STDOUT:   %.699: type = fn_type_with_self_type %ModWith.WithSelf.Op.type.513, %ModWith.facet.142 [concrete]
// CHECK:STDOUT:   %AddWith.facet.aef: %AddWith.type.6f4 = facet_value %Int32, (%custom_witness.e32) [concrete]
// CHECK:STDOUT:   %AddWith.WithSelf.Op.type.0aa: type = fn_type @AddWith.WithSelf.Op, @AddWith.WithSelf(%Int16, %AddWith.facet.aef) [concrete]
// CHECK:STDOUT:   %.211: type = fn_type_with_self_type %AddWith.WithSelf.Op.type.0aa, %AddWith.facet.aef [concrete]
// CHECK:STDOUT:   %SubWith.facet.9f2: %SubWith.type.8bd = facet_value %Int32, (%custom_witness.395) [concrete]
// CHECK:STDOUT:   %SubWith.WithSelf.Op.type.bba: type = fn_type @SubWith.WithSelf.Op, @SubWith.WithSelf(%Int16, %SubWith.facet.9f2) [concrete]
// CHECK:STDOUT:   %.2b2: type = fn_type_with_self_type %SubWith.WithSelf.Op.type.bba, %SubWith.facet.9f2 [concrete]
// CHECK:STDOUT:   %MulWith.facet.9cb: %MulWith.type.c25 = facet_value %Int32, (%custom_witness.678) [concrete]
// CHECK:STDOUT:   %MulWith.WithSelf.Op.type.15f: type = fn_type @MulWith.WithSelf.Op, @MulWith.WithSelf(%Int16, %MulWith.facet.9cb) [concrete]
// CHECK:STDOUT:   %.dd6: type = fn_type_with_self_type %MulWith.WithSelf.Op.type.15f, %MulWith.facet.9cb [concrete]
// CHECK:STDOUT:   %DivWith.facet.53f: %DivWith.type.053 = facet_value %Int32, (%custom_witness.8ad) [concrete]
// CHECK:STDOUT:   %DivWith.WithSelf.Op.type.ba3: type = fn_type @DivWith.WithSelf.Op, @DivWith.WithSelf(%Int16, %DivWith.facet.53f) [concrete]
// CHECK:STDOUT:   %.540: type = fn_type_with_self_type %DivWith.WithSelf.Op.type.ba3, %DivWith.facet.53f [concrete]
// CHECK:STDOUT:   %ModWith.facet.0ea: %ModWith.type.a83 = facet_value %Int32, (%custom_witness.f63) [concrete]
// CHECK:STDOUT:   %ModWith.WithSelf.Op.type.aeb: type = fn_type @ModWith.WithSelf.Op, @ModWith.WithSelf(%Int16, %ModWith.facet.0ea) [concrete]
// CHECK:STDOUT:   %.b1a: type = fn_type_with_self_type %ModWith.WithSelf.Op.type.aeb, %ModWith.facet.0ea [concrete]
// CHECK:STDOUT:   %AddWith.facet.616: %AddWith.type.78c = facet_value %Int32, (%custom_witness.e46c75.2) [concrete]
// CHECK:STDOUT:   %AddWith.WithSelf.Op.type.9e3: type = fn_type @AddWith.WithSelf.Op, @AddWith.WithSelf(%Int32, %AddWith.facet.616) [concrete]
// CHECK:STDOUT:   %.d9d: type = fn_type_with_self_type %AddWith.WithSelf.Op.type.9e3, %AddWith.facet.616 [concrete]
// CHECK:STDOUT:   %SubWith.facet.36d: %SubWith.type.629 = facet_value %Int32, (%custom_witness.7ffeb1.2) [concrete]
// CHECK:STDOUT:   %SubWith.WithSelf.Op.type.1db: type = fn_type @SubWith.WithSelf.Op, @SubWith.WithSelf(%Int32, %SubWith.facet.36d) [concrete]
// CHECK:STDOUT:   %.613: type = fn_type_with_self_type %SubWith.WithSelf.Op.type.1db, %SubWith.facet.36d [concrete]
// CHECK:STDOUT:   %MulWith.facet.dd3: %MulWith.type.3ee = facet_value %Int32, (%custom_witness.68a6cf.2) [concrete]
// CHECK:STDOUT:   %MulWith.WithSelf.Op.type.854: type = fn_type @MulWith.WithSelf.Op, @MulWith.WithSelf(%Int32, %MulWith.facet.dd3) [concrete]
// CHECK:STDOUT:   %.ec7: type = fn_type_with_self_type %MulWith.WithSelf.Op.type.854, %MulWith.facet.dd3 [concrete]
// CHECK:STDOUT:   %DivWith.facet.c86: %DivWith.type.09b = facet_value %Int32, (%custom_witness.2ccbdb.2) [concrete]
// CHECK:STDOUT:   %DivWith.WithSelf.Op.type.e4e: type = fn_type @DivWith.WithSelf.Op, @DivWith.WithSelf(%Int32, %DivWith.facet.c86) [concrete]
// CHECK:STDOUT:   %.865: type = fn_type_with_self_type %DivWith.WithSelf.Op.type.e4e, %DivWith.facet.c86 [concrete]
// CHECK:STDOUT:   %ModWith.facet.c80: %ModWith.type.6f6 = facet_value %Int32, (%custom_witness.596ded.2) [concrete]
// CHECK:STDOUT:   %ModWith.WithSelf.Op.type.e0b: type = fn_type @ModWith.WithSelf.Op, @ModWith.WithSelf(%Int32, %ModWith.facet.c80) [concrete]
// CHECK:STDOUT:   %.d2e: type = fn_type_with_self_type %ModWith.WithSelf.Op.type.e0b, %ModWith.facet.c80 [concrete]
// CHECK:STDOUT:   %complete_type.495: <witness> = complete_type_witness %AddWith.type.01b [concrete]
// CHECK:STDOUT:   %AddWith.facet.31a: %AddWith.type.01b = facet_value %Int32, (%custom_witness.e1e) [concrete]
// CHECK:STDOUT:   %AddWith.WithSelf.Op.type.c0a: type = fn_type @AddWith.WithSelf.Op, @AddWith.WithSelf(%Int64, %AddWith.facet.31a) [concrete]
// CHECK:STDOUT:   %.2ba: type = fn_type_with_self_type %AddWith.WithSelf.Op.type.c0a, %AddWith.facet.31a [concrete]
// CHECK:STDOUT:   %complete_type.cc7: <witness> = complete_type_witness %SubWith.type.82c [concrete]
// CHECK:STDOUT:   %SubWith.facet.9cc: %SubWith.type.82c = facet_value %Int32, (%custom_witness.053) [concrete]
// CHECK:STDOUT:   %SubWith.WithSelf.Op.type.a06: type = fn_type @SubWith.WithSelf.Op, @SubWith.WithSelf(%Int64, %SubWith.facet.9cc) [concrete]
// CHECK:STDOUT:   %.086: type = fn_type_with_self_type %SubWith.WithSelf.Op.type.a06, %SubWith.facet.9cc [concrete]
// CHECK:STDOUT:   %complete_type.da6: <witness> = complete_type_witness %MulWith.type.9ed [concrete]
// CHECK:STDOUT:   %MulWith.facet.83a: %MulWith.type.9ed = facet_value %Int32, (%custom_witness.f29) [concrete]
// CHECK:STDOUT:   %MulWith.WithSelf.Op.type.bc6: type = fn_type @MulWith.WithSelf.Op, @MulWith.WithSelf(%Int64, %MulWith.facet.83a) [concrete]
// CHECK:STDOUT:   %.ca6: type = fn_type_with_self_type %MulWith.WithSelf.Op.type.bc6, %MulWith.facet.83a [concrete]
// CHECK:STDOUT:   %complete_type.dba: <witness> = complete_type_witness %DivWith.type.7ac [concrete]
// CHECK:STDOUT:   %DivWith.facet.62d: %DivWith.type.7ac = facet_value %Int32, (%custom_witness.863) [concrete]
// CHECK:STDOUT:   %DivWith.WithSelf.Op.type.90b: type = fn_type @DivWith.WithSelf.Op, @DivWith.WithSelf(%Int64, %DivWith.facet.62d) [concrete]
// CHECK:STDOUT:   %.cb62: type = fn_type_with_self_type %DivWith.WithSelf.Op.type.90b, %DivWith.facet.62d [concrete]
// CHECK:STDOUT:   %complete_type.f9c: <witness> = complete_type_witness %ModWith.type.6b0 [concrete]
// CHECK:STDOUT:   %ModWith.facet.02b: %ModWith.type.6b0 = facet_value %Int32, (%custom_witness.eb8) [concrete]
// CHECK:STDOUT:   %ModWith.WithSelf.Op.type.17d: type = fn_type @ModWith.WithSelf.Op, @ModWith.WithSelf(%Int64, %ModWith.facet.02b) [concrete]
// CHECK:STDOUT:   %.0ca: type = fn_type_with_self_type %ModWith.WithSelf.Op.type.17d, %ModWith.facet.02b [concrete]
// CHECK:STDOUT:   %AddWith.facet.6a1: %AddWith.type.78c = facet_value %Int64, (%custom_witness.483) [concrete]
// CHECK:STDOUT:   %AddWith.WithSelf.Op.type.5ee: type = fn_type @AddWith.WithSelf.Op, @AddWith.WithSelf(%Int32, %AddWith.facet.6a1) [concrete]
// CHECK:STDOUT:   %.11a: type = fn_type_with_self_type %AddWith.WithSelf.Op.type.5ee, %AddWith.facet.6a1 [concrete]
// CHECK:STDOUT:   %SubWith.facet.8cb: %SubWith.type.629 = facet_value %Int64, (%custom_witness.095) [concrete]
// CHECK:STDOUT:   %SubWith.WithSelf.Op.type.1b8: type = fn_type @SubWith.WithSelf.Op, @SubWith.WithSelf(%Int32, %SubWith.facet.8cb) [concrete]
// CHECK:STDOUT:   %.51b: type = fn_type_with_self_type %SubWith.WithSelf.Op.type.1b8, %SubWith.facet.8cb [concrete]
// CHECK:STDOUT:   %MulWith.facet.e61: %MulWith.type.3ee = facet_value %Int64, (%custom_witness.f6e) [concrete]
// CHECK:STDOUT:   %MulWith.WithSelf.Op.type.d31: type = fn_type @MulWith.WithSelf.Op, @MulWith.WithSelf(%Int32, %MulWith.facet.e61) [concrete]
// CHECK:STDOUT:   %.2e4: type = fn_type_with_self_type %MulWith.WithSelf.Op.type.d31, %MulWith.facet.e61 [concrete]
// CHECK:STDOUT:   %DivWith.facet.b24: %DivWith.type.09b = facet_value %Int64, (%custom_witness.4e5) [concrete]
// CHECK:STDOUT:   %DivWith.WithSelf.Op.type.58d: type = fn_type @DivWith.WithSelf.Op, @DivWith.WithSelf(%Int32, %DivWith.facet.b24) [concrete]
// CHECK:STDOUT:   %.f2d: type = fn_type_with_self_type %DivWith.WithSelf.Op.type.58d, %DivWith.facet.b24 [concrete]
// CHECK:STDOUT:   %ModWith.facet.4c1: %ModWith.type.6f6 = facet_value %Int64, (%custom_witness.99b) [concrete]
// CHECK:STDOUT:   %ModWith.WithSelf.Op.type.fb3: type = fn_type @ModWith.WithSelf.Op, @ModWith.WithSelf(%Int32, %ModWith.facet.4c1) [concrete]
// CHECK:STDOUT:   %.7b9: type = fn_type_with_self_type %ModWith.WithSelf.Op.type.fb3, %ModWith.facet.4c1 [concrete]
// CHECK:STDOUT:   %complete_type.f45: <witness> = complete_type_witness %AddAssignWith.type.35f [concrete]
// CHECK:STDOUT:   %AddAssignWith.WithSelf.Op.type.f5d: type = fn_type @AddAssignWith.WithSelf.Op, @AddAssignWith.WithSelf(%i16, %AddAssignWith.facet.1b3) [concrete]
// CHECK:STDOUT:   %.c71: type = fn_type_with_self_type %AddAssignWith.WithSelf.Op.type.f5d, %AddAssignWith.facet.1b3 [concrete]
// CHECK:STDOUT:   %complete_type.c74: <witness> = complete_type_witness %SubAssignWith.type.121 [concrete]
// CHECK:STDOUT:   %SubAssignWith.WithSelf.Op.type.224: type = fn_type @SubAssignWith.WithSelf.Op, @SubAssignWith.WithSelf(%i16, %SubAssignWith.facet.45a) [concrete]
// CHECK:STDOUT:   %.9c3: type = fn_type_with_self_type %SubAssignWith.WithSelf.Op.type.224, %SubAssignWith.facet.45a [concrete]
// CHECK:STDOUT:   %complete_type.f93: <witness> = complete_type_witness %MulAssignWith.type.b7e [concrete]
// CHECK:STDOUT:   %MulAssignWith.WithSelf.Op.type.354: type = fn_type @MulAssignWith.WithSelf.Op, @MulAssignWith.WithSelf(%i16, %MulAssignWith.facet.805) [concrete]
// CHECK:STDOUT:   %.cbb: type = fn_type_with_self_type %MulAssignWith.WithSelf.Op.type.354, %MulAssignWith.facet.805 [concrete]
// CHECK:STDOUT:   %complete_type.913: <witness> = complete_type_witness %DivAssignWith.type.0d6 [concrete]
// CHECK:STDOUT:   %DivAssignWith.WithSelf.Op.type.510: type = fn_type @DivAssignWith.WithSelf.Op, @DivAssignWith.WithSelf(%i16, %DivAssignWith.facet.ac9) [concrete]
// CHECK:STDOUT:   %.b82: type = fn_type_with_self_type %DivAssignWith.WithSelf.Op.type.510, %DivAssignWith.facet.ac9 [concrete]
// CHECK:STDOUT:   %complete_type.597: <witness> = complete_type_witness %ModAssignWith.type.d0f [concrete]
// CHECK:STDOUT:   %ModAssignWith.WithSelf.Op.type.ad4: type = fn_type @ModAssignWith.WithSelf.Op, @ModAssignWith.WithSelf(%i16, %ModAssignWith.facet.350) [concrete]
// CHECK:STDOUT:   %.003: type = fn_type_with_self_type %ModAssignWith.WithSelf.Op.type.ad4, %ModAssignWith.facet.350 [concrete]
// CHECK:STDOUT:   %complete_type.d4d: <witness> = complete_type_witness %AddAssignWith.type.3e8 [concrete]
// CHECK:STDOUT:   %AddAssignWith.WithSelf.Op.type.b28: type = fn_type @AddAssignWith.WithSelf.Op, @AddAssignWith.WithSelf(%Int32, %AddAssignWith.facet.610) [concrete]
// CHECK:STDOUT:   %.05e: type = fn_type_with_self_type %AddAssignWith.WithSelf.Op.type.b28, %AddAssignWith.facet.610 [concrete]
// CHECK:STDOUT:   %complete_type.838: <witness> = complete_type_witness %SubAssignWith.type.678 [concrete]
// CHECK:STDOUT:   %SubAssignWith.WithSelf.Op.type.7bd: type = fn_type @SubAssignWith.WithSelf.Op, @SubAssignWith.WithSelf(%Int32, %SubAssignWith.facet.fb2) [concrete]
// CHECK:STDOUT:   %.cd8: type = fn_type_with_self_type %SubAssignWith.WithSelf.Op.type.7bd, %SubAssignWith.facet.fb2 [concrete]
// CHECK:STDOUT:   %complete_type.908: <witness> = complete_type_witness %MulAssignWith.type.011 [concrete]
// CHECK:STDOUT:   %MulAssignWith.WithSelf.Op.type.9a6: type = fn_type @MulAssignWith.WithSelf.Op, @MulAssignWith.WithSelf(%Int32, %MulAssignWith.facet.8f5) [concrete]
// CHECK:STDOUT:   %.300: type = fn_type_with_self_type %MulAssignWith.WithSelf.Op.type.9a6, %MulAssignWith.facet.8f5 [concrete]
// CHECK:STDOUT:   %complete_type.054: <witness> = complete_type_witness %DivAssignWith.type.487 [concrete]
// CHECK:STDOUT:   %DivAssignWith.WithSelf.Op.type.1ec: type = fn_type @DivAssignWith.WithSelf.Op, @DivAssignWith.WithSelf(%Int32, %DivAssignWith.facet.3bc) [concrete]
// CHECK:STDOUT:   %.d63: type = fn_type_with_self_type %DivAssignWith.WithSelf.Op.type.1ec, %DivAssignWith.facet.3bc [concrete]
// CHECK:STDOUT:   %complete_type.022: <witness> = complete_type_witness %ModAssignWith.type.499 [concrete]
// CHECK:STDOUT:   %ModAssignWith.WithSelf.Op.type.1a7: type = fn_type @ModAssignWith.WithSelf.Op, @ModAssignWith.WithSelf(%Int32, %ModAssignWith.facet.b73) [concrete]
// CHECK:STDOUT:   %.029: type = fn_type_with_self_type %ModAssignWith.WithSelf.Op.type.1a7, %ModAssignWith.facet.b73 [concrete]
// CHECK:STDOUT:   %AddAssignWith.WithSelf.Op.type.272: type = fn_type @AddAssignWith.WithSelf.Op, @AddAssignWith.WithSelf(%Int32, %AddAssignWith.facet.080) [concrete]
// CHECK:STDOUT:   %.257: type = fn_type_with_self_type %AddAssignWith.WithSelf.Op.type.272, %AddAssignWith.facet.080 [concrete]
// CHECK:STDOUT:   %SubAssignWith.WithSelf.Op.type.461: type = fn_type @SubAssignWith.WithSelf.Op, @SubAssignWith.WithSelf(%Int32, %SubAssignWith.facet.8bc) [concrete]
// CHECK:STDOUT:   %.e92: type = fn_type_with_self_type %SubAssignWith.WithSelf.Op.type.461, %SubAssignWith.facet.8bc [concrete]
// CHECK:STDOUT:   %MulAssignWith.WithSelf.Op.type.152: type = fn_type @MulAssignWith.WithSelf.Op, @MulAssignWith.WithSelf(%Int32, %MulAssignWith.facet.990) [concrete]
// CHECK:STDOUT:   %.694: type = fn_type_with_self_type %MulAssignWith.WithSelf.Op.type.152, %MulAssignWith.facet.990 [concrete]
// CHECK:STDOUT:   %DivAssignWith.WithSelf.Op.type.53f: type = fn_type @DivAssignWith.WithSelf.Op, @DivAssignWith.WithSelf(%Int32, %DivAssignWith.facet.3eb) [concrete]
// CHECK:STDOUT:   %.09f: type = fn_type_with_self_type %DivAssignWith.WithSelf.Op.type.53f, %DivAssignWith.facet.3eb [concrete]
// CHECK:STDOUT:   %ModAssignWith.WithSelf.Op.type.30a: type = fn_type @ModAssignWith.WithSelf.Op, @ModAssignWith.WithSelf(%Int32, %ModAssignWith.facet.2d7) [concrete]
// CHECK:STDOUT:   %.e8b: type = fn_type_with_self_type %ModAssignWith.WithSelf.Op.type.30a, %ModAssignWith.facet.2d7 [concrete]
// CHECK:STDOUT:   %Inc.WithSelf.Op.type.f8b: type = fn_type @Inc.WithSelf.Op, @Inc.WithSelf(%Inc.facet.ace) [concrete]
// CHECK:STDOUT:   %.b6c: type = fn_type_with_self_type %Inc.WithSelf.Op.type.f8b, %Inc.facet.ace [concrete]
// CHECK:STDOUT:   %Dec.WithSelf.Op.type.479: type = fn_type @Dec.WithSelf.Op, @Dec.WithSelf(%Dec.facet.be4) [concrete]
// CHECK:STDOUT:   %.a3b: type = fn_type_with_self_type %Dec.WithSelf.Op.type.479, %Dec.facet.be4 [concrete]
// CHECK:STDOUT:   %Negate.facet.070: %Negate.type = facet_value %Int16, (%custom_witness.71b) [concrete]
// CHECK:STDOUT:   %Negate.WithSelf.Op.type.cde: type = fn_type @Negate.WithSelf.Op, @Negate.WithSelf(%Negate.facet.070) [concrete]
// CHECK:STDOUT:   %.e93: type = fn_type_with_self_type %Negate.WithSelf.Op.type.cde, %Negate.facet.070 [concrete]
// CHECK:STDOUT:   %Negate.facet.670: %Negate.type = facet_value %Int32, (%custom_witness.45a) [concrete]
// CHECK:STDOUT:   %Negate.WithSelf.Op.type.20b: type = fn_type @Negate.WithSelf.Op, @Negate.WithSelf(%Negate.facet.670) [concrete]
// CHECK:STDOUT:   %.76e: type = fn_type_with_self_type %Negate.WithSelf.Op.type.20b, %Negate.facet.670 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.ada: @AddWith.WithSelf.%AddWith.assoc_type (%AddWith.assoc_type.5da) = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [symbolic = @AddWith.WithSelf.%assoc1 (constants.%assoc1.fdf)]
// CHECK:STDOUT:   %Core.import_ref.29ff: @AddWith.WithSelf.%AddWith.WithSelf.Op.type (%AddWith.WithSelf.Op.type.c45) = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [symbolic = @AddWith.WithSelf.%AddWith.WithSelf.Op (constants.%AddWith.WithSelf.Op.2f3)]
// CHECK:STDOUT:   %Core.import_ref.1dd: type = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [concrete = %Result.a0e]
// CHECK:STDOUT:   %Result.a0e: type = assoc_const_decl @Result.1 [concrete] {}
// CHECK:STDOUT:   %Core.import_ref.a1e = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.cd2: @SubWith.WithSelf.%SubWith.assoc_type (%SubWith.assoc_type.72b) = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [symbolic = @SubWith.WithSelf.%assoc1 (constants.%assoc1.e6a)]
// CHECK:STDOUT:   %Core.import_ref.940: @SubWith.WithSelf.%SubWith.WithSelf.Op.type (%SubWith.WithSelf.Op.type.7da) = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [symbolic = @SubWith.WithSelf.%SubWith.WithSelf.Op (constants.%SubWith.WithSelf.Op.bd5)]
// CHECK:STDOUT:   %Core.import_ref.391: type = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [concrete = %Result.982]
// CHECK:STDOUT:   %Result.982: type = assoc_const_decl @Result.2 [concrete] {}
// CHECK:STDOUT:   %Core.import_ref.e2a = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.537: @MulWith.WithSelf.%MulWith.assoc_type (%MulWith.assoc_type.634) = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [symbolic = @MulWith.WithSelf.%assoc1 (constants.%assoc1.14f)]
// CHECK:STDOUT:   %Core.import_ref.d23: @MulWith.WithSelf.%MulWith.WithSelf.Op.type (%MulWith.WithSelf.Op.type.309) = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [symbolic = @MulWith.WithSelf.%MulWith.WithSelf.Op (constants.%MulWith.WithSelf.Op.56c)]
// CHECK:STDOUT:   %Core.import_ref.41e: type = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [concrete = %Result.518]
// CHECK:STDOUT:   %Result.518: type = assoc_const_decl @Result.3 [concrete] {}
// CHECK:STDOUT:   %Core.import_ref.087 = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.973: @DivWith.WithSelf.%DivWith.assoc_type (%DivWith.assoc_type.5b4) = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [symbolic = @DivWith.WithSelf.%assoc1 (constants.%assoc1.9374)]
// CHECK:STDOUT:   %Core.import_ref.e712: @DivWith.WithSelf.%DivWith.WithSelf.Op.type (%DivWith.WithSelf.Op.type.e6f) = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [symbolic = @DivWith.WithSelf.%DivWith.WithSelf.Op (constants.%DivWith.WithSelf.Op.908)]
// CHECK:STDOUT:   %Core.import_ref.e71d: type = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [concrete = %Result.397]
// CHECK:STDOUT:   %Result.397: type = assoc_const_decl @Result.4 [concrete] {}
// CHECK:STDOUT:   %Core.import_ref.993 = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.fc2: @ModWith.WithSelf.%ModWith.assoc_type (%ModWith.assoc_type.4a7) = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [symbolic = @ModWith.WithSelf.%assoc1 (constants.%assoc1.232)]
// CHECK:STDOUT:   %Core.import_ref.104: @ModWith.WithSelf.%ModWith.WithSelf.Op.type (%ModWith.WithSelf.Op.type.f6b) = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [symbolic = @ModWith.WithSelf.%ModWith.WithSelf.Op (constants.%ModWith.WithSelf.Op.397)]
// CHECK:STDOUT:   %Core.import_ref.327: type = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [concrete = %Result.22a]
// CHECK:STDOUT:   %Result.22a: type = assoc_const_decl @Result.5 [concrete] {}
// CHECK:STDOUT:   %Core.import_ref.bbd = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.3bc: @AddAssignWith.WithSelf.%AddAssignWith.assoc_type (%AddAssignWith.assoc_type.d9654f.1) = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [symbolic = @AddAssignWith.WithSelf.%assoc0 (constants.%assoc0.1e6)]
// CHECK:STDOUT:   %Core.import_ref.62c: @AddAssignWith.WithSelf.%AddAssignWith.WithSelf.Op.type (%AddAssignWith.WithSelf.Op.type.86ac33.1) = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [symbolic = @AddAssignWith.WithSelf.%AddAssignWith.WithSelf.Op (constants.%AddAssignWith.WithSelf.Op.31f15f.1)]
// CHECK:STDOUT:   %Core.import_ref.970 = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.639: @SubAssignWith.WithSelf.%SubAssignWith.assoc_type (%SubAssignWith.assoc_type.efddf3.1) = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [symbolic = @SubAssignWith.WithSelf.%assoc0 (constants.%assoc0.936)]
// CHECK:STDOUT:   %Core.import_ref.e50: @SubAssignWith.WithSelf.%SubAssignWith.WithSelf.Op.type (%SubAssignWith.WithSelf.Op.type.839113.1) = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [symbolic = @SubAssignWith.WithSelf.%SubAssignWith.WithSelf.Op (constants.%SubAssignWith.WithSelf.Op.2286e8.1)]
// CHECK:STDOUT:   %Core.import_ref.7b0 = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.5c1: @MulAssignWith.WithSelf.%MulAssignWith.assoc_type (%MulAssignWith.assoc_type.153441.1) = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [symbolic = @MulAssignWith.WithSelf.%assoc0 (constants.%assoc0.862)]
// CHECK:STDOUT:   %Core.import_ref.763: @MulAssignWith.WithSelf.%MulAssignWith.WithSelf.Op.type (%MulAssignWith.WithSelf.Op.type.44adac.1) = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [symbolic = @MulAssignWith.WithSelf.%MulAssignWith.WithSelf.Op (constants.%MulAssignWith.WithSelf.Op.f87570.1)]
// CHECK:STDOUT:   %Core.import_ref.634 = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.cbe: @DivAssignWith.WithSelf.%DivAssignWith.assoc_type (%DivAssignWith.assoc_type.c30cb0.1) = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [symbolic = @DivAssignWith.WithSelf.%assoc0 (constants.%assoc0.086)]
// CHECK:STDOUT:   %Core.import_ref.d8a: @DivAssignWith.WithSelf.%DivAssignWith.WithSelf.Op.type (%DivAssignWith.WithSelf.Op.type.8e0740.1) = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [symbolic = @DivAssignWith.WithSelf.%DivAssignWith.WithSelf.Op (constants.%DivAssignWith.WithSelf.Op.0bd7e1.1)]
// CHECK:STDOUT:   %Core.import_ref.b97 = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.447: @ModAssignWith.WithSelf.%ModAssignWith.assoc_type (%ModAssignWith.assoc_type.0ff0e5.1) = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [symbolic = @ModAssignWith.WithSelf.%assoc0 (constants.%assoc0.715)]
// CHECK:STDOUT:   %Core.import_ref.aee3: @ModAssignWith.WithSelf.%ModAssignWith.WithSelf.Op.type (%ModAssignWith.WithSelf.Op.type.292942.1) = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [symbolic = @ModAssignWith.WithSelf.%ModAssignWith.WithSelf.Op (constants.%ModAssignWith.WithSelf.Op.84faf3.1)]
// CHECK:STDOUT:   %Core.import_ref.248 = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @AddWith.loc124(%Result.loc121_9.2: %Destroy.type, %U.loc122_4.2: %Destroy.type, %T.loc123_4.2: @AddWith.loc124.%facet_type.loc123_38 (%facet_type.cac)) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %require_complete.loc126: <witness> = require_complete_type %AddWith.type.loc123_36.1 [symbolic = %require_complete.loc126 (constants.%require_complete.3dc)]
// CHECK:STDOUT:   %assoc1: @AddWith.loc124.%AddWith.assoc_type (%AddWith.assoc_type.c6c) = assoc_entity element1, imports.%Core.import_ref.29ff [symbolic = %assoc1 (constants.%assoc1.d93)]
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.loc126: <witness> = lookup_impl_witness %T.loc123_4.1, @AddWith.1, @AddWith.1(%U.as_type.loc123_36.1) [symbolic = %AddWith.lookup_impl_witness.loc126 (constants.%AddWith.lookup_impl_witness.e7e)]
// CHECK:STDOUT:   %AddWith.facet: @AddWith.loc124.%AddWith.type.loc123_36.1 (%AddWith.type.ad2) = facet_value %T.as_type.loc124_29.1, (%AddWith.lookup_impl_witness.loc126) [symbolic = %AddWith.facet (constants.%AddWith.facet.c8e)]
// CHECK:STDOUT:   %AddWith.WithSelf.Op.type: type = fn_type @AddWith.WithSelf.Op, @AddWith.WithSelf(%U.as_type.loc123_36.1, %AddWith.facet) [symbolic = %AddWith.WithSelf.Op.type (constants.%AddWith.WithSelf.Op.type.12d)]
// CHECK:STDOUT:   %.loc126_5.4: type = fn_type_with_self_type %AddWith.WithSelf.Op.type, %AddWith.facet [symbolic = %.loc126_5.4 (constants.%.ac7)]
// CHECK:STDOUT:   %impl.elem1.loc126_5.2: @AddWith.loc124.%.loc126_5.4 (%.ac7) = impl_witness_access %AddWith.lookup_impl_witness.loc126, element1 [symbolic = %impl.elem1.loc126_5.2 (constants.%impl.elem1.f9d)]
// CHECK:STDOUT:   %specific_impl_fn.loc126_5.3: <specific function> = specific_impl_function %impl.elem1.loc126_5.2, @AddWith.WithSelf.Op(%U.as_type.loc123_36.1, %AddWith.facet) [symbolic = %specific_impl_fn.loc126_5.3 (constants.%specific_impl_fn.8f2)]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Result.loc121_9.1) [symbolic = %Destroy.WithSelf.Op.type (constants.%Destroy.WithSelf.Op.type.6304ea.2)]
// CHECK:STDOUT:   %.loc126_5.5: type = fn_type_with_self_type %Destroy.WithSelf.Op.type, %Result.loc121_9.1 [symbolic = %.loc126_5.5 (constants.%.543)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %Result.loc121_9.1, @Destroy [symbolic = %Destroy.lookup_impl_witness (constants.%Destroy.lookup_impl_witness)]
// CHECK:STDOUT:   %impl.elem0.loc126_5.2: @AddWith.loc124.%.loc126_5.5 (%.543) = impl_witness_access %Destroy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc126_5.2 (constants.%impl.elem0.531)]
// CHECK:STDOUT:   %specific_impl_fn.loc126_5.4: <specific function> = specific_impl_function %impl.elem0.loc126_5.2, @Destroy.WithSelf.Op(%Result.loc121_9.1) [symbolic = %specific_impl_fn.loc126_5.4 (constants.%specific_impl_fn.010)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%result.param: @AddWith.loc124.%Result.as_type.loc123_54.1 (%Result.as_type), %x.param: @AddWith.loc124.%T.as_type.loc124_29.1 (%T.as_type.46a), %y.param: @AddWith.loc124.%U.as_type.loc123_36.1 (%U.as_type.f7f)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: @AddWith.loc124.%T.as_type.loc124_29.1 (%T.as_type.46a) = name_ref x, %x
// CHECK:STDOUT:     %y.ref: @AddWith.loc124.%U.as_type.loc123_36.1 (%U.as_type.f7f) = name_ref y, %y
// CHECK:STDOUT:     %AddWith.type.loc126: type = facet_type <@AddWith.1, @AddWith.1(constants.%U.as_type.f7f)> [symbolic = %AddWith.type.loc123_36.1 (constants.%AddWith.type.ad2)]
// CHECK:STDOUT:     %.loc126_5.1: @AddWith.loc124.%AddWith.assoc_type (%AddWith.assoc_type.c6c) = specific_constant imports.%Core.import_ref.ada, @AddWith.WithSelf(constants.%U.as_type.f7f, constants.%Self.66b) [symbolic = %assoc1 (constants.%assoc1.d93)]
// CHECK:STDOUT:     %Op.ref: @AddWith.loc124.%AddWith.assoc_type (%AddWith.assoc_type.c6c) = name_ref Op, %.loc126_5.1 [symbolic = %assoc1 (constants.%assoc1.d93)]
// CHECK:STDOUT:     %impl.elem1.loc126_5.1: @AddWith.loc124.%.loc126_5.4 (%.ac7) = impl_witness_access constants.%AddWith.lookup_impl_witness.e7e, element1 [symbolic = %impl.elem1.loc126_5.2 (constants.%impl.elem1.f9d)]
// CHECK:STDOUT:     %bound_method.loc126_5.1: <bound method> = bound_method %x.ref, %impl.elem1.loc126_5.1
// CHECK:STDOUT:     %specific_impl_fn.loc126_5.1: <specific function> = specific_impl_function %impl.elem1.loc126_5.1, @AddWith.WithSelf.Op(constants.%U.as_type.f7f, constants.%AddWith.facet.c8e) [symbolic = %specific_impl_fn.loc126_5.3 (constants.%specific_impl_fn.8f2)]
// CHECK:STDOUT:     %bound_method.loc126_5.2: <bound method> = bound_method %x.ref, %specific_impl_fn.loc126_5.1
// CHECK:STDOUT:     %.loc126_5.2: ref @AddWith.loc124.%Result.as_type.loc123_54.1 (%Result.as_type) = temporary_storage
// CHECK:STDOUT:     %AddWith.WithSelf.Op.call: init @AddWith.loc124.%Result.as_type.loc123_54.1 (%Result.as_type) to %.loc126_5.2 = call %bound_method.loc126_5.2(%x.ref, %y.ref)
// CHECK:STDOUT:     %.loc126_5.3: ref @AddWith.loc124.%Result.as_type.loc123_54.1 (%Result.as_type) = temporary %.loc126_5.2, %AddWith.WithSelf.Op.call
// CHECK:STDOUT:     %impl.elem0.loc126_5.1: @AddWith.loc124.%.loc126_5.5 (%.543) = impl_witness_access constants.%Destroy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc126_5.2 (constants.%impl.elem0.531)]
// CHECK:STDOUT:     %bound_method.loc126_5.3: <bound method> = bound_method %.loc126_5.3, %impl.elem0.loc126_5.1
// CHECK:STDOUT:     %specific_impl_fn.loc126_5.2: <specific function> = specific_impl_function %impl.elem0.loc126_5.1, @Destroy.WithSelf.Op(constants.%Result) [symbolic = %specific_impl_fn.loc126_5.4 (constants.%specific_impl_fn.010)]
// CHECK:STDOUT:     %bound_method.loc126_5.4: <bound method> = bound_method %.loc126_5.3, %specific_impl_fn.loc126_5.2
// CHECK:STDOUT:     %Destroy.WithSelf.Op.call: init %empty_tuple.type = call %bound_method.loc126_5.4(%.loc126_5.3)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @SubWith.loc135(%Result.loc132_9.2: %Destroy.type, %U.loc133_4.2: %Destroy.type, %T.loc134_4.2: @SubWith.loc135.%facet_type.loc134_38 (%facet_type.98a)) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %require_complete.loc137: <witness> = require_complete_type %SubWith.type.loc134_36.1 [symbolic = %require_complete.loc137 (constants.%require_complete.30c)]
// CHECK:STDOUT:   %assoc1: @SubWith.loc135.%SubWith.assoc_type (%SubWith.assoc_type.767) = assoc_entity element1, imports.%Core.import_ref.940 [symbolic = %assoc1 (constants.%assoc1.edb)]
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.loc137: <witness> = lookup_impl_witness %T.loc134_4.1, @SubWith.1, @SubWith.1(%U.as_type.loc134_36.1) [symbolic = %SubWith.lookup_impl_witness.loc137 (constants.%SubWith.lookup_impl_witness.031)]
// CHECK:STDOUT:   %SubWith.facet: @SubWith.loc135.%SubWith.type.loc134_36.1 (%SubWith.type.394) = facet_value %T.as_type.loc135_29.1, (%SubWith.lookup_impl_witness.loc137) [symbolic = %SubWith.facet (constants.%SubWith.facet.2d1)]
// CHECK:STDOUT:   %SubWith.WithSelf.Op.type: type = fn_type @SubWith.WithSelf.Op, @SubWith.WithSelf(%U.as_type.loc134_36.1, %SubWith.facet) [symbolic = %SubWith.WithSelf.Op.type (constants.%SubWith.WithSelf.Op.type.496)]
// CHECK:STDOUT:   %.loc137_5.4: type = fn_type_with_self_type %SubWith.WithSelf.Op.type, %SubWith.facet [symbolic = %.loc137_5.4 (constants.%.754)]
// CHECK:STDOUT:   %impl.elem1.loc137_5.2: @SubWith.loc135.%.loc137_5.4 (%.754) = impl_witness_access %SubWith.lookup_impl_witness.loc137, element1 [symbolic = %impl.elem1.loc137_5.2 (constants.%impl.elem1.a03)]
// CHECK:STDOUT:   %specific_impl_fn.loc137_5.3: <specific function> = specific_impl_function %impl.elem1.loc137_5.2, @SubWith.WithSelf.Op(%U.as_type.loc134_36.1, %SubWith.facet) [symbolic = %specific_impl_fn.loc137_5.3 (constants.%specific_impl_fn.069)]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Result.loc132_9.1) [symbolic = %Destroy.WithSelf.Op.type (constants.%Destroy.WithSelf.Op.type.6304ea.2)]
// CHECK:STDOUT:   %.loc137_5.5: type = fn_type_with_self_type %Destroy.WithSelf.Op.type, %Result.loc132_9.1 [symbolic = %.loc137_5.5 (constants.%.543)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %Result.loc132_9.1, @Destroy [symbolic = %Destroy.lookup_impl_witness (constants.%Destroy.lookup_impl_witness)]
// CHECK:STDOUT:   %impl.elem0.loc137_5.2: @SubWith.loc135.%.loc137_5.5 (%.543) = impl_witness_access %Destroy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc137_5.2 (constants.%impl.elem0.531)]
// CHECK:STDOUT:   %specific_impl_fn.loc137_5.4: <specific function> = specific_impl_function %impl.elem0.loc137_5.2, @Destroy.WithSelf.Op(%Result.loc132_9.1) [symbolic = %specific_impl_fn.loc137_5.4 (constants.%specific_impl_fn.010)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%result.param: @SubWith.loc135.%Result.as_type.loc134_54.1 (%Result.as_type), %x.param: @SubWith.loc135.%T.as_type.loc135_29.1 (%T.as_type.330), %y.param: @SubWith.loc135.%U.as_type.loc134_36.1 (%U.as_type.f7f)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: @SubWith.loc135.%T.as_type.loc135_29.1 (%T.as_type.330) = name_ref x, %x
// CHECK:STDOUT:     %y.ref: @SubWith.loc135.%U.as_type.loc134_36.1 (%U.as_type.f7f) = name_ref y, %y
// CHECK:STDOUT:     %SubWith.type.loc137: type = facet_type <@SubWith.1, @SubWith.1(constants.%U.as_type.f7f)> [symbolic = %SubWith.type.loc134_36.1 (constants.%SubWith.type.394)]
// CHECK:STDOUT:     %.loc137_5.1: @SubWith.loc135.%SubWith.assoc_type (%SubWith.assoc_type.767) = specific_constant imports.%Core.import_ref.cd2, @SubWith.WithSelf(constants.%U.as_type.f7f, constants.%Self.cc1) [symbolic = %assoc1 (constants.%assoc1.edb)]
// CHECK:STDOUT:     %Op.ref: @SubWith.loc135.%SubWith.assoc_type (%SubWith.assoc_type.767) = name_ref Op, %.loc137_5.1 [symbolic = %assoc1 (constants.%assoc1.edb)]
// CHECK:STDOUT:     %impl.elem1.loc137_5.1: @SubWith.loc135.%.loc137_5.4 (%.754) = impl_witness_access constants.%SubWith.lookup_impl_witness.031, element1 [symbolic = %impl.elem1.loc137_5.2 (constants.%impl.elem1.a03)]
// CHECK:STDOUT:     %bound_method.loc137_5.1: <bound method> = bound_method %x.ref, %impl.elem1.loc137_5.1
// CHECK:STDOUT:     %specific_impl_fn.loc137_5.1: <specific function> = specific_impl_function %impl.elem1.loc137_5.1, @SubWith.WithSelf.Op(constants.%U.as_type.f7f, constants.%SubWith.facet.2d1) [symbolic = %specific_impl_fn.loc137_5.3 (constants.%specific_impl_fn.069)]
// CHECK:STDOUT:     %bound_method.loc137_5.2: <bound method> = bound_method %x.ref, %specific_impl_fn.loc137_5.1
// CHECK:STDOUT:     %.loc137_5.2: ref @SubWith.loc135.%Result.as_type.loc134_54.1 (%Result.as_type) = temporary_storage
// CHECK:STDOUT:     %SubWith.WithSelf.Op.call: init @SubWith.loc135.%Result.as_type.loc134_54.1 (%Result.as_type) to %.loc137_5.2 = call %bound_method.loc137_5.2(%x.ref, %y.ref)
// CHECK:STDOUT:     %.loc137_5.3: ref @SubWith.loc135.%Result.as_type.loc134_54.1 (%Result.as_type) = temporary %.loc137_5.2, %SubWith.WithSelf.Op.call
// CHECK:STDOUT:     %impl.elem0.loc137_5.1: @SubWith.loc135.%.loc137_5.5 (%.543) = impl_witness_access constants.%Destroy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc137_5.2 (constants.%impl.elem0.531)]
// CHECK:STDOUT:     %bound_method.loc137_5.3: <bound method> = bound_method %.loc137_5.3, %impl.elem0.loc137_5.1
// CHECK:STDOUT:     %specific_impl_fn.loc137_5.2: <specific function> = specific_impl_function %impl.elem0.loc137_5.1, @Destroy.WithSelf.Op(constants.%Result) [symbolic = %specific_impl_fn.loc137_5.4 (constants.%specific_impl_fn.010)]
// CHECK:STDOUT:     %bound_method.loc137_5.4: <bound method> = bound_method %.loc137_5.3, %specific_impl_fn.loc137_5.2
// CHECK:STDOUT:     %Destroy.WithSelf.Op.call: init %empty_tuple.type = call %bound_method.loc137_5.4(%.loc137_5.3)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @MulWith.loc146(%Result.loc143_9.2: %Destroy.type, %U.loc144_4.2: %Destroy.type, %T.loc145_4.2: @MulWith.loc146.%facet_type.loc145_38 (%facet_type.a59)) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %require_complete.loc148: <witness> = require_complete_type %MulWith.type.loc145_36.1 [symbolic = %require_complete.loc148 (constants.%require_complete.01c)]
// CHECK:STDOUT:   %assoc1: @MulWith.loc146.%MulWith.assoc_type (%MulWith.assoc_type.fff) = assoc_entity element1, imports.%Core.import_ref.d23 [symbolic = %assoc1 (constants.%assoc1.bde)]
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.loc148: <witness> = lookup_impl_witness %T.loc145_4.1, @MulWith.1, @MulWith.1(%U.as_type.loc145_36.1) [symbolic = %MulWith.lookup_impl_witness.loc148 (constants.%MulWith.lookup_impl_witness.e5c)]
// CHECK:STDOUT:   %MulWith.facet: @MulWith.loc146.%MulWith.type.loc145_36.1 (%MulWith.type.195) = facet_value %T.as_type.loc146_29.1, (%MulWith.lookup_impl_witness.loc148) [symbolic = %MulWith.facet (constants.%MulWith.facet.eb2)]
// CHECK:STDOUT:   %MulWith.WithSelf.Op.type: type = fn_type @MulWith.WithSelf.Op, @MulWith.WithSelf(%U.as_type.loc145_36.1, %MulWith.facet) [symbolic = %MulWith.WithSelf.Op.type (constants.%MulWith.WithSelf.Op.type.288)]
// CHECK:STDOUT:   %.loc148_5.4: type = fn_type_with_self_type %MulWith.WithSelf.Op.type, %MulWith.facet [symbolic = %.loc148_5.4 (constants.%.a85)]
// CHECK:STDOUT:   %impl.elem1.loc148_5.2: @MulWith.loc146.%.loc148_5.4 (%.a85) = impl_witness_access %MulWith.lookup_impl_witness.loc148, element1 [symbolic = %impl.elem1.loc148_5.2 (constants.%impl.elem1.447)]
// CHECK:STDOUT:   %specific_impl_fn.loc148_5.3: <specific function> = specific_impl_function %impl.elem1.loc148_5.2, @MulWith.WithSelf.Op(%U.as_type.loc145_36.1, %MulWith.facet) [symbolic = %specific_impl_fn.loc148_5.3 (constants.%specific_impl_fn.32c)]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Result.loc143_9.1) [symbolic = %Destroy.WithSelf.Op.type (constants.%Destroy.WithSelf.Op.type.6304ea.2)]
// CHECK:STDOUT:   %.loc148_5.5: type = fn_type_with_self_type %Destroy.WithSelf.Op.type, %Result.loc143_9.1 [symbolic = %.loc148_5.5 (constants.%.543)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %Result.loc143_9.1, @Destroy [symbolic = %Destroy.lookup_impl_witness (constants.%Destroy.lookup_impl_witness)]
// CHECK:STDOUT:   %impl.elem0.loc148_5.2: @MulWith.loc146.%.loc148_5.5 (%.543) = impl_witness_access %Destroy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc148_5.2 (constants.%impl.elem0.531)]
// CHECK:STDOUT:   %specific_impl_fn.loc148_5.4: <specific function> = specific_impl_function %impl.elem0.loc148_5.2, @Destroy.WithSelf.Op(%Result.loc143_9.1) [symbolic = %specific_impl_fn.loc148_5.4 (constants.%specific_impl_fn.010)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%result.param: @MulWith.loc146.%Result.as_type.loc145_54.1 (%Result.as_type), %x.param: @MulWith.loc146.%T.as_type.loc146_29.1 (%T.as_type.198), %y.param: @MulWith.loc146.%U.as_type.loc145_36.1 (%U.as_type.f7f)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: @MulWith.loc146.%T.as_type.loc146_29.1 (%T.as_type.198) = name_ref x, %x
// CHECK:STDOUT:     %y.ref: @MulWith.loc146.%U.as_type.loc145_36.1 (%U.as_type.f7f) = name_ref y, %y
// CHECK:STDOUT:     %MulWith.type.loc148: type = facet_type <@MulWith.1, @MulWith.1(constants.%U.as_type.f7f)> [symbolic = %MulWith.type.loc145_36.1 (constants.%MulWith.type.195)]
// CHECK:STDOUT:     %.loc148_5.1: @MulWith.loc146.%MulWith.assoc_type (%MulWith.assoc_type.fff) = specific_constant imports.%Core.import_ref.537, @MulWith.WithSelf(constants.%U.as_type.f7f, constants.%Self.0e2) [symbolic = %assoc1 (constants.%assoc1.bde)]
// CHECK:STDOUT:     %Op.ref: @MulWith.loc146.%MulWith.assoc_type (%MulWith.assoc_type.fff) = name_ref Op, %.loc148_5.1 [symbolic = %assoc1 (constants.%assoc1.bde)]
// CHECK:STDOUT:     %impl.elem1.loc148_5.1: @MulWith.loc146.%.loc148_5.4 (%.a85) = impl_witness_access constants.%MulWith.lookup_impl_witness.e5c, element1 [symbolic = %impl.elem1.loc148_5.2 (constants.%impl.elem1.447)]
// CHECK:STDOUT:     %bound_method.loc148_5.1: <bound method> = bound_method %x.ref, %impl.elem1.loc148_5.1
// CHECK:STDOUT:     %specific_impl_fn.loc148_5.1: <specific function> = specific_impl_function %impl.elem1.loc148_5.1, @MulWith.WithSelf.Op(constants.%U.as_type.f7f, constants.%MulWith.facet.eb2) [symbolic = %specific_impl_fn.loc148_5.3 (constants.%specific_impl_fn.32c)]
// CHECK:STDOUT:     %bound_method.loc148_5.2: <bound method> = bound_method %x.ref, %specific_impl_fn.loc148_5.1
// CHECK:STDOUT:     %.loc148_5.2: ref @MulWith.loc146.%Result.as_type.loc145_54.1 (%Result.as_type) = temporary_storage
// CHECK:STDOUT:     %MulWith.WithSelf.Op.call: init @MulWith.loc146.%Result.as_type.loc145_54.1 (%Result.as_type) to %.loc148_5.2 = call %bound_method.loc148_5.2(%x.ref, %y.ref)
// CHECK:STDOUT:     %.loc148_5.3: ref @MulWith.loc146.%Result.as_type.loc145_54.1 (%Result.as_type) = temporary %.loc148_5.2, %MulWith.WithSelf.Op.call
// CHECK:STDOUT:     %impl.elem0.loc148_5.1: @MulWith.loc146.%.loc148_5.5 (%.543) = impl_witness_access constants.%Destroy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc148_5.2 (constants.%impl.elem0.531)]
// CHECK:STDOUT:     %bound_method.loc148_5.3: <bound method> = bound_method %.loc148_5.3, %impl.elem0.loc148_5.1
// CHECK:STDOUT:     %specific_impl_fn.loc148_5.2: <specific function> = specific_impl_function %impl.elem0.loc148_5.1, @Destroy.WithSelf.Op(constants.%Result) [symbolic = %specific_impl_fn.loc148_5.4 (constants.%specific_impl_fn.010)]
// CHECK:STDOUT:     %bound_method.loc148_5.4: <bound method> = bound_method %.loc148_5.3, %specific_impl_fn.loc148_5.2
// CHECK:STDOUT:     %Destroy.WithSelf.Op.call: init %empty_tuple.type = call %bound_method.loc148_5.4(%.loc148_5.3)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @DivWith.loc157(%Result.loc154_9.2: %Destroy.type, %U.loc155_4.2: %Destroy.type, %T.loc156_4.2: @DivWith.loc157.%facet_type.loc156_38 (%facet_type.5cc)) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %require_complete.loc159: <witness> = require_complete_type %DivWith.type.loc156_36.1 [symbolic = %require_complete.loc159 (constants.%require_complete.be3)]
// CHECK:STDOUT:   %assoc1: @DivWith.loc157.%DivWith.assoc_type (%DivWith.assoc_type.e27) = assoc_entity element1, imports.%Core.import_ref.e712 [symbolic = %assoc1 (constants.%assoc1.7f9)]
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.loc159: <witness> = lookup_impl_witness %T.loc156_4.1, @DivWith.1, @DivWith.1(%U.as_type.loc156_36.1) [symbolic = %DivWith.lookup_impl_witness.loc159 (constants.%DivWith.lookup_impl_witness.2e1)]
// CHECK:STDOUT:   %DivWith.facet: @DivWith.loc157.%DivWith.type.loc156_36.1 (%DivWith.type.63d) = facet_value %T.as_type.loc157_29.1, (%DivWith.lookup_impl_witness.loc159) [symbolic = %DivWith.facet (constants.%DivWith.facet.b8f)]
// CHECK:STDOUT:   %DivWith.WithSelf.Op.type: type = fn_type @DivWith.WithSelf.Op, @DivWith.WithSelf(%U.as_type.loc156_36.1, %DivWith.facet) [symbolic = %DivWith.WithSelf.Op.type (constants.%DivWith.WithSelf.Op.type.be6)]
// CHECK:STDOUT:   %.loc159_5.4: type = fn_type_with_self_type %DivWith.WithSelf.Op.type, %DivWith.facet [symbolic = %.loc159_5.4 (constants.%.42b)]
// CHECK:STDOUT:   %impl.elem1.loc159_5.2: @DivWith.loc157.%.loc159_5.4 (%.42b) = impl_witness_access %DivWith.lookup_impl_witness.loc159, element1 [symbolic = %impl.elem1.loc159_5.2 (constants.%impl.elem1.6f6)]
// CHECK:STDOUT:   %specific_impl_fn.loc159_5.3: <specific function> = specific_impl_function %impl.elem1.loc159_5.2, @DivWith.WithSelf.Op(%U.as_type.loc156_36.1, %DivWith.facet) [symbolic = %specific_impl_fn.loc159_5.3 (constants.%specific_impl_fn.263)]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Result.loc154_9.1) [symbolic = %Destroy.WithSelf.Op.type (constants.%Destroy.WithSelf.Op.type.6304ea.2)]
// CHECK:STDOUT:   %.loc159_5.5: type = fn_type_with_self_type %Destroy.WithSelf.Op.type, %Result.loc154_9.1 [symbolic = %.loc159_5.5 (constants.%.543)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %Result.loc154_9.1, @Destroy [symbolic = %Destroy.lookup_impl_witness (constants.%Destroy.lookup_impl_witness)]
// CHECK:STDOUT:   %impl.elem0.loc159_5.2: @DivWith.loc157.%.loc159_5.5 (%.543) = impl_witness_access %Destroy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc159_5.2 (constants.%impl.elem0.531)]
// CHECK:STDOUT:   %specific_impl_fn.loc159_5.4: <specific function> = specific_impl_function %impl.elem0.loc159_5.2, @Destroy.WithSelf.Op(%Result.loc154_9.1) [symbolic = %specific_impl_fn.loc159_5.4 (constants.%specific_impl_fn.010)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%result.param: @DivWith.loc157.%Result.as_type.loc156_54.1 (%Result.as_type), %x.param: @DivWith.loc157.%T.as_type.loc157_29.1 (%T.as_type.e49), %y.param: @DivWith.loc157.%U.as_type.loc156_36.1 (%U.as_type.f7f)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: @DivWith.loc157.%T.as_type.loc157_29.1 (%T.as_type.e49) = name_ref x, %x
// CHECK:STDOUT:     %y.ref: @DivWith.loc157.%U.as_type.loc156_36.1 (%U.as_type.f7f) = name_ref y, %y
// CHECK:STDOUT:     %DivWith.type.loc159: type = facet_type <@DivWith.1, @DivWith.1(constants.%U.as_type.f7f)> [symbolic = %DivWith.type.loc156_36.1 (constants.%DivWith.type.63d)]
// CHECK:STDOUT:     %.loc159_5.1: @DivWith.loc157.%DivWith.assoc_type (%DivWith.assoc_type.e27) = specific_constant imports.%Core.import_ref.973, @DivWith.WithSelf(constants.%U.as_type.f7f, constants.%Self.7db) [symbolic = %assoc1 (constants.%assoc1.7f9)]
// CHECK:STDOUT:     %Op.ref: @DivWith.loc157.%DivWith.assoc_type (%DivWith.assoc_type.e27) = name_ref Op, %.loc159_5.1 [symbolic = %assoc1 (constants.%assoc1.7f9)]
// CHECK:STDOUT:     %impl.elem1.loc159_5.1: @DivWith.loc157.%.loc159_5.4 (%.42b) = impl_witness_access constants.%DivWith.lookup_impl_witness.2e1, element1 [symbolic = %impl.elem1.loc159_5.2 (constants.%impl.elem1.6f6)]
// CHECK:STDOUT:     %bound_method.loc159_5.1: <bound method> = bound_method %x.ref, %impl.elem1.loc159_5.1
// CHECK:STDOUT:     %specific_impl_fn.loc159_5.1: <specific function> = specific_impl_function %impl.elem1.loc159_5.1, @DivWith.WithSelf.Op(constants.%U.as_type.f7f, constants.%DivWith.facet.b8f) [symbolic = %specific_impl_fn.loc159_5.3 (constants.%specific_impl_fn.263)]
// CHECK:STDOUT:     %bound_method.loc159_5.2: <bound method> = bound_method %x.ref, %specific_impl_fn.loc159_5.1
// CHECK:STDOUT:     %.loc159_5.2: ref @DivWith.loc157.%Result.as_type.loc156_54.1 (%Result.as_type) = temporary_storage
// CHECK:STDOUT:     %DivWith.WithSelf.Op.call: init @DivWith.loc157.%Result.as_type.loc156_54.1 (%Result.as_type) to %.loc159_5.2 = call %bound_method.loc159_5.2(%x.ref, %y.ref)
// CHECK:STDOUT:     %.loc159_5.3: ref @DivWith.loc157.%Result.as_type.loc156_54.1 (%Result.as_type) = temporary %.loc159_5.2, %DivWith.WithSelf.Op.call
// CHECK:STDOUT:     %impl.elem0.loc159_5.1: @DivWith.loc157.%.loc159_5.5 (%.543) = impl_witness_access constants.%Destroy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc159_5.2 (constants.%impl.elem0.531)]
// CHECK:STDOUT:     %bound_method.loc159_5.3: <bound method> = bound_method %.loc159_5.3, %impl.elem0.loc159_5.1
// CHECK:STDOUT:     %specific_impl_fn.loc159_5.2: <specific function> = specific_impl_function %impl.elem0.loc159_5.1, @Destroy.WithSelf.Op(constants.%Result) [symbolic = %specific_impl_fn.loc159_5.4 (constants.%specific_impl_fn.010)]
// CHECK:STDOUT:     %bound_method.loc159_5.4: <bound method> = bound_method %.loc159_5.3, %specific_impl_fn.loc159_5.2
// CHECK:STDOUT:     %Destroy.WithSelf.Op.call: init %empty_tuple.type = call %bound_method.loc159_5.4(%.loc159_5.3)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @ModWith.loc168(%Result.loc165_9.2: %Destroy.type, %U.loc166_4.2: %Destroy.type, %T.loc167_4.2: @ModWith.loc168.%facet_type.loc167_38 (%facet_type.c32)) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %require_complete.loc170: <witness> = require_complete_type %ModWith.type.loc167_36.1 [symbolic = %require_complete.loc170 (constants.%require_complete.a64)]
// CHECK:STDOUT:   %assoc1: @ModWith.loc168.%ModWith.assoc_type (%ModWith.assoc_type.3d0) = assoc_entity element1, imports.%Core.import_ref.104 [symbolic = %assoc1 (constants.%assoc1.3aa)]
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.loc170: <witness> = lookup_impl_witness %T.loc167_4.1, @ModWith.1, @ModWith.1(%U.as_type.loc167_36.1) [symbolic = %ModWith.lookup_impl_witness.loc170 (constants.%ModWith.lookup_impl_witness.35d)]
// CHECK:STDOUT:   %ModWith.facet: @ModWith.loc168.%ModWith.type.loc167_36.1 (%ModWith.type.125) = facet_value %T.as_type.loc168_29.1, (%ModWith.lookup_impl_witness.loc170) [symbolic = %ModWith.facet (constants.%ModWith.facet.132)]
// CHECK:STDOUT:   %ModWith.WithSelf.Op.type: type = fn_type @ModWith.WithSelf.Op, @ModWith.WithSelf(%U.as_type.loc167_36.1, %ModWith.facet) [symbolic = %ModWith.WithSelf.Op.type (constants.%ModWith.WithSelf.Op.type.942)]
// CHECK:STDOUT:   %.loc170_5.4: type = fn_type_with_self_type %ModWith.WithSelf.Op.type, %ModWith.facet [symbolic = %.loc170_5.4 (constants.%.1a3)]
// CHECK:STDOUT:   %impl.elem1.loc170_5.2: @ModWith.loc168.%.loc170_5.4 (%.1a3) = impl_witness_access %ModWith.lookup_impl_witness.loc170, element1 [symbolic = %impl.elem1.loc170_5.2 (constants.%impl.elem1.4de)]
// CHECK:STDOUT:   %specific_impl_fn.loc170_5.3: <specific function> = specific_impl_function %impl.elem1.loc170_5.2, @ModWith.WithSelf.Op(%U.as_type.loc167_36.1, %ModWith.facet) [symbolic = %specific_impl_fn.loc170_5.3 (constants.%specific_impl_fn.5d3)]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Result.loc165_9.1) [symbolic = %Destroy.WithSelf.Op.type (constants.%Destroy.WithSelf.Op.type.6304ea.2)]
// CHECK:STDOUT:   %.loc170_5.5: type = fn_type_with_self_type %Destroy.WithSelf.Op.type, %Result.loc165_9.1 [symbolic = %.loc170_5.5 (constants.%.543)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %Result.loc165_9.1, @Destroy [symbolic = %Destroy.lookup_impl_witness (constants.%Destroy.lookup_impl_witness)]
// CHECK:STDOUT:   %impl.elem0.loc170_5.2: @ModWith.loc168.%.loc170_5.5 (%.543) = impl_witness_access %Destroy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc170_5.2 (constants.%impl.elem0.531)]
// CHECK:STDOUT:   %specific_impl_fn.loc170_5.4: <specific function> = specific_impl_function %impl.elem0.loc170_5.2, @Destroy.WithSelf.Op(%Result.loc165_9.1) [symbolic = %specific_impl_fn.loc170_5.4 (constants.%specific_impl_fn.010)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%result.param: @ModWith.loc168.%Result.as_type.loc167_54.1 (%Result.as_type), %x.param: @ModWith.loc168.%T.as_type.loc168_29.1 (%T.as_type.105), %y.param: @ModWith.loc168.%U.as_type.loc167_36.1 (%U.as_type.f7f)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: @ModWith.loc168.%T.as_type.loc168_29.1 (%T.as_type.105) = name_ref x, %x
// CHECK:STDOUT:     %y.ref: @ModWith.loc168.%U.as_type.loc167_36.1 (%U.as_type.f7f) = name_ref y, %y
// CHECK:STDOUT:     %ModWith.type.loc170: type = facet_type <@ModWith.1, @ModWith.1(constants.%U.as_type.f7f)> [symbolic = %ModWith.type.loc167_36.1 (constants.%ModWith.type.125)]
// CHECK:STDOUT:     %.loc170_5.1: @ModWith.loc168.%ModWith.assoc_type (%ModWith.assoc_type.3d0) = specific_constant imports.%Core.import_ref.fc2, @ModWith.WithSelf(constants.%U.as_type.f7f, constants.%Self.310) [symbolic = %assoc1 (constants.%assoc1.3aa)]
// CHECK:STDOUT:     %Op.ref: @ModWith.loc168.%ModWith.assoc_type (%ModWith.assoc_type.3d0) = name_ref Op, %.loc170_5.1 [symbolic = %assoc1 (constants.%assoc1.3aa)]
// CHECK:STDOUT:     %impl.elem1.loc170_5.1: @ModWith.loc168.%.loc170_5.4 (%.1a3) = impl_witness_access constants.%ModWith.lookup_impl_witness.35d, element1 [symbolic = %impl.elem1.loc170_5.2 (constants.%impl.elem1.4de)]
// CHECK:STDOUT:     %bound_method.loc170_5.1: <bound method> = bound_method %x.ref, %impl.elem1.loc170_5.1
// CHECK:STDOUT:     %specific_impl_fn.loc170_5.1: <specific function> = specific_impl_function %impl.elem1.loc170_5.1, @ModWith.WithSelf.Op(constants.%U.as_type.f7f, constants.%ModWith.facet.132) [symbolic = %specific_impl_fn.loc170_5.3 (constants.%specific_impl_fn.5d3)]
// CHECK:STDOUT:     %bound_method.loc170_5.2: <bound method> = bound_method %x.ref, %specific_impl_fn.loc170_5.1
// CHECK:STDOUT:     %.loc170_5.2: ref @ModWith.loc168.%Result.as_type.loc167_54.1 (%Result.as_type) = temporary_storage
// CHECK:STDOUT:     %ModWith.WithSelf.Op.call: init @ModWith.loc168.%Result.as_type.loc167_54.1 (%Result.as_type) to %.loc170_5.2 = call %bound_method.loc170_5.2(%x.ref, %y.ref)
// CHECK:STDOUT:     %.loc170_5.3: ref @ModWith.loc168.%Result.as_type.loc167_54.1 (%Result.as_type) = temporary %.loc170_5.2, %ModWith.WithSelf.Op.call
// CHECK:STDOUT:     %impl.elem0.loc170_5.1: @ModWith.loc168.%.loc170_5.5 (%.543) = impl_witness_access constants.%Destroy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc170_5.2 (constants.%impl.elem0.531)]
// CHECK:STDOUT:     %bound_method.loc170_5.3: <bound method> = bound_method %.loc170_5.3, %impl.elem0.loc170_5.1
// CHECK:STDOUT:     %specific_impl_fn.loc170_5.2: <specific function> = specific_impl_function %impl.elem0.loc170_5.1, @Destroy.WithSelf.Op(constants.%Result) [symbolic = %specific_impl_fn.loc170_5.4 (constants.%specific_impl_fn.010)]
// CHECK:STDOUT:     %bound_method.loc170_5.4: <bound method> = bound_method %.loc170_5.3, %specific_impl_fn.loc170_5.2
// CHECK:STDOUT:     %Destroy.WithSelf.Op.call: init %empty_tuple.type = call %bound_method.loc170_5.4(%.loc170_5.3)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @AddAssignWith.loc225(%U.loc225_19.2: type, %T.loc225_29.2: @AddAssignWith.loc225.%AddAssignWith.type.loc225_52.1 (%AddAssignWith.type.f7e94b.2)) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %require_complete.loc227: <witness> = require_complete_type %AddAssignWith.type.loc225_52.1 [symbolic = %require_complete.loc227 (constants.%require_complete.e50)]
// CHECK:STDOUT:   %AddAssignWith.assoc_type: type = assoc_entity_type @AddAssignWith.1, @AddAssignWith.1(%U.loc225_19.1) [symbolic = %AddAssignWith.assoc_type (constants.%AddAssignWith.assoc_type.d9654f.2)]
// CHECK:STDOUT:   %assoc0: @AddAssignWith.loc225.%AddAssignWith.assoc_type (%AddAssignWith.assoc_type.d9654f.2) = assoc_entity element0, imports.%Core.import_ref.62c [symbolic = %assoc0 (constants.%assoc0.beca96.2)]
// CHECK:STDOUT:   %AddAssignWith.WithSelf.Op.type: type = fn_type @AddAssignWith.WithSelf.Op, @AddAssignWith.WithSelf(%U.loc225_19.1, %T.loc225_29.1) [symbolic = %AddAssignWith.WithSelf.Op.type (constants.%AddAssignWith.WithSelf.Op.type.86ac33.3)]
// CHECK:STDOUT:   %.loc227_5.2: type = fn_type_with_self_type %AddAssignWith.WithSelf.Op.type, %T.loc225_29.1 [symbolic = %.loc227_5.2 (constants.%.1da)]
// CHECK:STDOUT:   %AddAssignWith.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc225_29.1, @AddAssignWith.1, @AddAssignWith.1(%U.loc225_19.1) [symbolic = %AddAssignWith.lookup_impl_witness (constants.%AddAssignWith.lookup_impl_witness)]
// CHECK:STDOUT:   %impl.elem0.loc227_5.2: @AddAssignWith.loc225.%.loc227_5.2 (%.1da) = impl_witness_access %AddAssignWith.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc227_5.2 (constants.%impl.elem0.28f)]
// CHECK:STDOUT:   %specific_impl_fn.loc227_5.2: <specific function> = specific_impl_function %impl.elem0.loc227_5.2, @AddAssignWith.WithSelf.Op(%U.loc225_19.1, %T.loc225_29.1) [symbolic = %specific_impl_fn.loc227_5.2 (constants.%specific_impl_fn.c19)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: ref @AddAssignWith.loc225.%T.as_type.loc225_62.1 (%T.as_type.82b), %y.param: @AddAssignWith.loc225.%U.loc225_19.1 (%U.67d)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: ref @AddAssignWith.loc225.%T.as_type.loc225_62.1 (%T.as_type.82b) = name_ref x, %x
// CHECK:STDOUT:     %y.ref: @AddAssignWith.loc225.%U.loc225_19.1 (%U.67d) = name_ref y, %y
// CHECK:STDOUT:     %AddAssignWith.type.loc227: type = facet_type <@AddAssignWith.1, @AddAssignWith.1(constants.%U.67d)> [symbolic = %AddAssignWith.type.loc225_52.1 (constants.%AddAssignWith.type.f7e94b.2)]
// CHECK:STDOUT:     %.loc227_5.1: @AddAssignWith.loc225.%AddAssignWith.assoc_type (%AddAssignWith.assoc_type.d9654f.2) = specific_constant imports.%Core.import_ref.3bc, @AddAssignWith.WithSelf(constants.%U.67d, constants.%Self.c9ea37.1) [symbolic = %assoc0 (constants.%assoc0.beca96.2)]
// CHECK:STDOUT:     %Op.ref: @AddAssignWith.loc225.%AddAssignWith.assoc_type (%AddAssignWith.assoc_type.d9654f.2) = name_ref Op, %.loc227_5.1 [symbolic = %assoc0 (constants.%assoc0.beca96.2)]
// CHECK:STDOUT:     %impl.elem0.loc227_5.1: @AddAssignWith.loc225.%.loc227_5.2 (%.1da) = impl_witness_access constants.%AddAssignWith.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc227_5.2 (constants.%impl.elem0.28f)]
// CHECK:STDOUT:     %bound_method.loc227_5.1: <bound method> = bound_method %x.ref, %impl.elem0.loc227_5.1
// CHECK:STDOUT:     %specific_impl_fn.loc227_5.1: <specific function> = specific_impl_function %impl.elem0.loc227_5.1, @AddAssignWith.WithSelf.Op(constants.%U.67d, constants.%T.c9e) [symbolic = %specific_impl_fn.loc227_5.2 (constants.%specific_impl_fn.c19)]
// CHECK:STDOUT:     %bound_method.loc227_5.2: <bound method> = bound_method %x.ref, %specific_impl_fn.loc227_5.1
// CHECK:STDOUT:     %AddAssignWith.WithSelf.Op.call: init %empty_tuple.type = call %bound_method.loc227_5.2(%x.ref, %y.ref)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @SubAssignWith.loc231(%U.loc231_19.2: type, %T.loc231_29.2: @SubAssignWith.loc231.%SubAssignWith.type.loc231_52.1 (%SubAssignWith.type.591a5f.2)) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %require_complete.loc233: <witness> = require_complete_type %SubAssignWith.type.loc231_52.1 [symbolic = %require_complete.loc233 (constants.%require_complete.b28)]
// CHECK:STDOUT:   %SubAssignWith.assoc_type: type = assoc_entity_type @SubAssignWith.1, @SubAssignWith.1(%U.loc231_19.1) [symbolic = %SubAssignWith.assoc_type (constants.%SubAssignWith.assoc_type.efddf3.2)]
// CHECK:STDOUT:   %assoc0: @SubAssignWith.loc231.%SubAssignWith.assoc_type (%SubAssignWith.assoc_type.efddf3.2) = assoc_entity element0, imports.%Core.import_ref.e50 [symbolic = %assoc0 (constants.%assoc0.382bd4.2)]
// CHECK:STDOUT:   %SubAssignWith.WithSelf.Op.type: type = fn_type @SubAssignWith.WithSelf.Op, @SubAssignWith.WithSelf(%U.loc231_19.1, %T.loc231_29.1) [symbolic = %SubAssignWith.WithSelf.Op.type (constants.%SubAssignWith.WithSelf.Op.type.839113.3)]
// CHECK:STDOUT:   %.loc233_5.2: type = fn_type_with_self_type %SubAssignWith.WithSelf.Op.type, %T.loc231_29.1 [symbolic = %.loc233_5.2 (constants.%.3df)]
// CHECK:STDOUT:   %SubAssignWith.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc231_29.1, @SubAssignWith.1, @SubAssignWith.1(%U.loc231_19.1) [symbolic = %SubAssignWith.lookup_impl_witness (constants.%SubAssignWith.lookup_impl_witness)]
// CHECK:STDOUT:   %impl.elem0.loc233_5.2: @SubAssignWith.loc231.%.loc233_5.2 (%.3df) = impl_witness_access %SubAssignWith.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc233_5.2 (constants.%impl.elem0.dac)]
// CHECK:STDOUT:   %specific_impl_fn.loc233_5.2: <specific function> = specific_impl_function %impl.elem0.loc233_5.2, @SubAssignWith.WithSelf.Op(%U.loc231_19.1, %T.loc231_29.1) [symbolic = %specific_impl_fn.loc233_5.2 (constants.%specific_impl_fn.2f2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: ref @SubAssignWith.loc231.%T.as_type.loc231_62.1 (%T.as_type.765), %y.param: @SubAssignWith.loc231.%U.loc231_19.1 (%U.67d)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: ref @SubAssignWith.loc231.%T.as_type.loc231_62.1 (%T.as_type.765) = name_ref x, %x
// CHECK:STDOUT:     %y.ref: @SubAssignWith.loc231.%U.loc231_19.1 (%U.67d) = name_ref y, %y
// CHECK:STDOUT:     %SubAssignWith.type.loc233: type = facet_type <@SubAssignWith.1, @SubAssignWith.1(constants.%U.67d)> [symbolic = %SubAssignWith.type.loc231_52.1 (constants.%SubAssignWith.type.591a5f.2)]
// CHECK:STDOUT:     %.loc233_5.1: @SubAssignWith.loc231.%SubAssignWith.assoc_type (%SubAssignWith.assoc_type.efddf3.2) = specific_constant imports.%Core.import_ref.639, @SubAssignWith.WithSelf(constants.%U.67d, constants.%Self.a96693.1) [symbolic = %assoc0 (constants.%assoc0.382bd4.2)]
// CHECK:STDOUT:     %Op.ref: @SubAssignWith.loc231.%SubAssignWith.assoc_type (%SubAssignWith.assoc_type.efddf3.2) = name_ref Op, %.loc233_5.1 [symbolic = %assoc0 (constants.%assoc0.382bd4.2)]
// CHECK:STDOUT:     %impl.elem0.loc233_5.1: @SubAssignWith.loc231.%.loc233_5.2 (%.3df) = impl_witness_access constants.%SubAssignWith.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc233_5.2 (constants.%impl.elem0.dac)]
// CHECK:STDOUT:     %bound_method.loc233_5.1: <bound method> = bound_method %x.ref, %impl.elem0.loc233_5.1
// CHECK:STDOUT:     %specific_impl_fn.loc233_5.1: <specific function> = specific_impl_function %impl.elem0.loc233_5.1, @SubAssignWith.WithSelf.Op(constants.%U.67d, constants.%T.a96) [symbolic = %specific_impl_fn.loc233_5.2 (constants.%specific_impl_fn.2f2)]
// CHECK:STDOUT:     %bound_method.loc233_5.2: <bound method> = bound_method %x.ref, %specific_impl_fn.loc233_5.1
// CHECK:STDOUT:     %SubAssignWith.WithSelf.Op.call: init %empty_tuple.type = call %bound_method.loc233_5.2(%x.ref, %y.ref)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @MulAssignWith.loc237(%U.loc237_19.2: type, %T.loc237_29.2: @MulAssignWith.loc237.%MulAssignWith.type.loc237_52.1 (%MulAssignWith.type.b9aa5c.2)) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %require_complete.loc239: <witness> = require_complete_type %MulAssignWith.type.loc237_52.1 [symbolic = %require_complete.loc239 (constants.%require_complete.ea2)]
// CHECK:STDOUT:   %MulAssignWith.assoc_type: type = assoc_entity_type @MulAssignWith.1, @MulAssignWith.1(%U.loc237_19.1) [symbolic = %MulAssignWith.assoc_type (constants.%MulAssignWith.assoc_type.153441.2)]
// CHECK:STDOUT:   %assoc0: @MulAssignWith.loc237.%MulAssignWith.assoc_type (%MulAssignWith.assoc_type.153441.2) = assoc_entity element0, imports.%Core.import_ref.763 [symbolic = %assoc0 (constants.%assoc0.54a880.2)]
// CHECK:STDOUT:   %MulAssignWith.WithSelf.Op.type: type = fn_type @MulAssignWith.WithSelf.Op, @MulAssignWith.WithSelf(%U.loc237_19.1, %T.loc237_29.1) [symbolic = %MulAssignWith.WithSelf.Op.type (constants.%MulAssignWith.WithSelf.Op.type.44adac.3)]
// CHECK:STDOUT:   %.loc239_5.2: type = fn_type_with_self_type %MulAssignWith.WithSelf.Op.type, %T.loc237_29.1 [symbolic = %.loc239_5.2 (constants.%.42e)]
// CHECK:STDOUT:   %MulAssignWith.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc237_29.1, @MulAssignWith.1, @MulAssignWith.1(%U.loc237_19.1) [symbolic = %MulAssignWith.lookup_impl_witness (constants.%MulAssignWith.lookup_impl_witness)]
// CHECK:STDOUT:   %impl.elem0.loc239_5.2: @MulAssignWith.loc237.%.loc239_5.2 (%.42e) = impl_witness_access %MulAssignWith.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc239_5.2 (constants.%impl.elem0.822)]
// CHECK:STDOUT:   %specific_impl_fn.loc239_5.2: <specific function> = specific_impl_function %impl.elem0.loc239_5.2, @MulAssignWith.WithSelf.Op(%U.loc237_19.1, %T.loc237_29.1) [symbolic = %specific_impl_fn.loc239_5.2 (constants.%specific_impl_fn.cf3)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: ref @MulAssignWith.loc237.%T.as_type.loc237_62.1 (%T.as_type.e75), %y.param: @MulAssignWith.loc237.%U.loc237_19.1 (%U.67d)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: ref @MulAssignWith.loc237.%T.as_type.loc237_62.1 (%T.as_type.e75) = name_ref x, %x
// CHECK:STDOUT:     %y.ref: @MulAssignWith.loc237.%U.loc237_19.1 (%U.67d) = name_ref y, %y
// CHECK:STDOUT:     %MulAssignWith.type.loc239: type = facet_type <@MulAssignWith.1, @MulAssignWith.1(constants.%U.67d)> [symbolic = %MulAssignWith.type.loc237_52.1 (constants.%MulAssignWith.type.b9aa5c.2)]
// CHECK:STDOUT:     %.loc239_5.1: @MulAssignWith.loc237.%MulAssignWith.assoc_type (%MulAssignWith.assoc_type.153441.2) = specific_constant imports.%Core.import_ref.5c1, @MulAssignWith.WithSelf(constants.%U.67d, constants.%Self.1ab5ec.1) [symbolic = %assoc0 (constants.%assoc0.54a880.2)]
// CHECK:STDOUT:     %Op.ref: @MulAssignWith.loc237.%MulAssignWith.assoc_type (%MulAssignWith.assoc_type.153441.2) = name_ref Op, %.loc239_5.1 [symbolic = %assoc0 (constants.%assoc0.54a880.2)]
// CHECK:STDOUT:     %impl.elem0.loc239_5.1: @MulAssignWith.loc237.%.loc239_5.2 (%.42e) = impl_witness_access constants.%MulAssignWith.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc239_5.2 (constants.%impl.elem0.822)]
// CHECK:STDOUT:     %bound_method.loc239_5.1: <bound method> = bound_method %x.ref, %impl.elem0.loc239_5.1
// CHECK:STDOUT:     %specific_impl_fn.loc239_5.1: <specific function> = specific_impl_function %impl.elem0.loc239_5.1, @MulAssignWith.WithSelf.Op(constants.%U.67d, constants.%T.1ab) [symbolic = %specific_impl_fn.loc239_5.2 (constants.%specific_impl_fn.cf3)]
// CHECK:STDOUT:     %bound_method.loc239_5.2: <bound method> = bound_method %x.ref, %specific_impl_fn.loc239_5.1
// CHECK:STDOUT:     %MulAssignWith.WithSelf.Op.call: init %empty_tuple.type = call %bound_method.loc239_5.2(%x.ref, %y.ref)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @DivAssignWith.loc243(%U.loc243_19.2: type, %T.loc243_29.2: @DivAssignWith.loc243.%DivAssignWith.type.loc243_52.1 (%DivAssignWith.type.c97ac9.2)) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %require_complete.loc245: <witness> = require_complete_type %DivAssignWith.type.loc243_52.1 [symbolic = %require_complete.loc245 (constants.%require_complete.3f4)]
// CHECK:STDOUT:   %DivAssignWith.assoc_type: type = assoc_entity_type @DivAssignWith.1, @DivAssignWith.1(%U.loc243_19.1) [symbolic = %DivAssignWith.assoc_type (constants.%DivAssignWith.assoc_type.c30cb0.2)]
// CHECK:STDOUT:   %assoc0: @DivAssignWith.loc243.%DivAssignWith.assoc_type (%DivAssignWith.assoc_type.c30cb0.2) = assoc_entity element0, imports.%Core.import_ref.d8a [symbolic = %assoc0 (constants.%assoc0.bd6a2d.2)]
// CHECK:STDOUT:   %DivAssignWith.WithSelf.Op.type: type = fn_type @DivAssignWith.WithSelf.Op, @DivAssignWith.WithSelf(%U.loc243_19.1, %T.loc243_29.1) [symbolic = %DivAssignWith.WithSelf.Op.type (constants.%DivAssignWith.WithSelf.Op.type.8e0740.3)]
// CHECK:STDOUT:   %.loc245_5.2: type = fn_type_with_self_type %DivAssignWith.WithSelf.Op.type, %T.loc243_29.1 [symbolic = %.loc245_5.2 (constants.%.28f)]
// CHECK:STDOUT:   %DivAssignWith.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc243_29.1, @DivAssignWith.1, @DivAssignWith.1(%U.loc243_19.1) [symbolic = %DivAssignWith.lookup_impl_witness (constants.%DivAssignWith.lookup_impl_witness)]
// CHECK:STDOUT:   %impl.elem0.loc245_5.2: @DivAssignWith.loc243.%.loc245_5.2 (%.28f) = impl_witness_access %DivAssignWith.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc245_5.2 (constants.%impl.elem0.f6e)]
// CHECK:STDOUT:   %specific_impl_fn.loc245_5.2: <specific function> = specific_impl_function %impl.elem0.loc245_5.2, @DivAssignWith.WithSelf.Op(%U.loc243_19.1, %T.loc243_29.1) [symbolic = %specific_impl_fn.loc245_5.2 (constants.%specific_impl_fn.5bc)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: ref @DivAssignWith.loc243.%T.as_type.loc243_62.1 (%T.as_type.5d7), %y.param: @DivAssignWith.loc243.%U.loc243_19.1 (%U.67d)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: ref @DivAssignWith.loc243.%T.as_type.loc243_62.1 (%T.as_type.5d7) = name_ref x, %x
// CHECK:STDOUT:     %y.ref: @DivAssignWith.loc243.%U.loc243_19.1 (%U.67d) = name_ref y, %y
// CHECK:STDOUT:     %DivAssignWith.type.loc245: type = facet_type <@DivAssignWith.1, @DivAssignWith.1(constants.%U.67d)> [symbolic = %DivAssignWith.type.loc243_52.1 (constants.%DivAssignWith.type.c97ac9.2)]
// CHECK:STDOUT:     %.loc245_5.1: @DivAssignWith.loc243.%DivAssignWith.assoc_type (%DivAssignWith.assoc_type.c30cb0.2) = specific_constant imports.%Core.import_ref.cbe, @DivAssignWith.WithSelf(constants.%U.67d, constants.%Self.43908b.1) [symbolic = %assoc0 (constants.%assoc0.bd6a2d.2)]
// CHECK:STDOUT:     %Op.ref: @DivAssignWith.loc243.%DivAssignWith.assoc_type (%DivAssignWith.assoc_type.c30cb0.2) = name_ref Op, %.loc245_5.1 [symbolic = %assoc0 (constants.%assoc0.bd6a2d.2)]
// CHECK:STDOUT:     %impl.elem0.loc245_5.1: @DivAssignWith.loc243.%.loc245_5.2 (%.28f) = impl_witness_access constants.%DivAssignWith.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc245_5.2 (constants.%impl.elem0.f6e)]
// CHECK:STDOUT:     %bound_method.loc245_5.1: <bound method> = bound_method %x.ref, %impl.elem0.loc245_5.1
// CHECK:STDOUT:     %specific_impl_fn.loc245_5.1: <specific function> = specific_impl_function %impl.elem0.loc245_5.1, @DivAssignWith.WithSelf.Op(constants.%U.67d, constants.%T.439) [symbolic = %specific_impl_fn.loc245_5.2 (constants.%specific_impl_fn.5bc)]
// CHECK:STDOUT:     %bound_method.loc245_5.2: <bound method> = bound_method %x.ref, %specific_impl_fn.loc245_5.1
// CHECK:STDOUT:     %DivAssignWith.WithSelf.Op.call: init %empty_tuple.type = call %bound_method.loc245_5.2(%x.ref, %y.ref)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @ModAssignWith.loc249(%U.loc249_19.2: type, %T.loc249_29.2: @ModAssignWith.loc249.%ModAssignWith.type.loc249_52.1 (%ModAssignWith.type.8faa47.2)) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %require_complete.loc251: <witness> = require_complete_type %ModAssignWith.type.loc249_52.1 [symbolic = %require_complete.loc251 (constants.%require_complete.9bf)]
// CHECK:STDOUT:   %ModAssignWith.assoc_type: type = assoc_entity_type @ModAssignWith.1, @ModAssignWith.1(%U.loc249_19.1) [symbolic = %ModAssignWith.assoc_type (constants.%ModAssignWith.assoc_type.0ff0e5.2)]
// CHECK:STDOUT:   %assoc0: @ModAssignWith.loc249.%ModAssignWith.assoc_type (%ModAssignWith.assoc_type.0ff0e5.2) = assoc_entity element0, imports.%Core.import_ref.aee3 [symbolic = %assoc0 (constants.%assoc0.3824c9.2)]
// CHECK:STDOUT:   %ModAssignWith.WithSelf.Op.type: type = fn_type @ModAssignWith.WithSelf.Op, @ModAssignWith.WithSelf(%U.loc249_19.1, %T.loc249_29.1) [symbolic = %ModAssignWith.WithSelf.Op.type (constants.%ModAssignWith.WithSelf.Op.type.292942.3)]
// CHECK:STDOUT:   %.loc251_5.2: type = fn_type_with_self_type %ModAssignWith.WithSelf.Op.type, %T.loc249_29.1 [symbolic = %.loc251_5.2 (constants.%.acb)]
// CHECK:STDOUT:   %ModAssignWith.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc249_29.1, @ModAssignWith.1, @ModAssignWith.1(%U.loc249_19.1) [symbolic = %ModAssignWith.lookup_impl_witness (constants.%ModAssignWith.lookup_impl_witness)]
// CHECK:STDOUT:   %impl.elem0.loc251_5.2: @ModAssignWith.loc249.%.loc251_5.2 (%.acb) = impl_witness_access %ModAssignWith.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc251_5.2 (constants.%impl.elem0.f9d)]
// CHECK:STDOUT:   %specific_impl_fn.loc251_5.2: <specific function> = specific_impl_function %impl.elem0.loc251_5.2, @ModAssignWith.WithSelf.Op(%U.loc249_19.1, %T.loc249_29.1) [symbolic = %specific_impl_fn.loc251_5.2 (constants.%specific_impl_fn.802)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: ref @ModAssignWith.loc249.%T.as_type.loc249_62.1 (%T.as_type.4cb), %y.param: @ModAssignWith.loc249.%U.loc249_19.1 (%U.67d)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: ref @ModAssignWith.loc249.%T.as_type.loc249_62.1 (%T.as_type.4cb) = name_ref x, %x
// CHECK:STDOUT:     %y.ref: @ModAssignWith.loc249.%U.loc249_19.1 (%U.67d) = name_ref y, %y
// CHECK:STDOUT:     %ModAssignWith.type.loc251: type = facet_type <@ModAssignWith.1, @ModAssignWith.1(constants.%U.67d)> [symbolic = %ModAssignWith.type.loc249_52.1 (constants.%ModAssignWith.type.8faa47.2)]
// CHECK:STDOUT:     %.loc251_5.1: @ModAssignWith.loc249.%ModAssignWith.assoc_type (%ModAssignWith.assoc_type.0ff0e5.2) = specific_constant imports.%Core.import_ref.447, @ModAssignWith.WithSelf(constants.%U.67d, constants.%Self.da03f6.1) [symbolic = %assoc0 (constants.%assoc0.3824c9.2)]
// CHECK:STDOUT:     %Op.ref: @ModAssignWith.loc249.%ModAssignWith.assoc_type (%ModAssignWith.assoc_type.0ff0e5.2) = name_ref Op, %.loc251_5.1 [symbolic = %assoc0 (constants.%assoc0.3824c9.2)]
// CHECK:STDOUT:     %impl.elem0.loc251_5.1: @ModAssignWith.loc249.%.loc251_5.2 (%.acb) = impl_witness_access constants.%ModAssignWith.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc251_5.2 (constants.%impl.elem0.f9d)]
// CHECK:STDOUT:     %bound_method.loc251_5.1: <bound method> = bound_method %x.ref, %impl.elem0.loc251_5.1
// CHECK:STDOUT:     %specific_impl_fn.loc251_5.1: <specific function> = specific_impl_function %impl.elem0.loc251_5.1, @ModAssignWith.WithSelf.Op(constants.%U.67d, constants.%T.da0) [symbolic = %specific_impl_fn.loc251_5.2 (constants.%specific_impl_fn.802)]
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
// CHECK:STDOUT:   %Inc.WithSelf.Op.type: type = fn_type @Inc.WithSelf.Op, @Inc.WithSelf(%T.loc278_13.1) [symbolic = %Inc.WithSelf.Op.type (constants.%Inc.WithSelf.Op.type.1a5b76.2)]
// CHECK:STDOUT:   %.loc280: type = fn_type_with_self_type %Inc.WithSelf.Op.type, %T.loc278_13.1 [symbolic = %.loc280 (constants.%.bdc)]
// CHECK:STDOUT:   %Inc.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc278_13.1, @Inc [symbolic = %Inc.lookup_impl_witness (constants.%Inc.lookup_impl_witness)]
// CHECK:STDOUT:   %impl.elem0.loc280_3.2: @TestInc.%.loc280 (%.bdc) = impl_witness_access %Inc.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc280_3.2 (constants.%impl.elem0.a96)]
// CHECK:STDOUT:   %specific_impl_fn.loc280_3.2: <specific function> = specific_impl_function %impl.elem0.loc280_3.2, @Inc.WithSelf.Op(%T.loc278_13.1) [symbolic = %specific_impl_fn.loc280_3.2 (constants.%specific_impl_fn.d8c)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: ref @TestInc.%T.as_type.loc278_33.1 (%T.as_type.4cc)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: ref @TestInc.%T.as_type.loc278_33.1 (%T.as_type.4cc) = name_ref x, %x
// CHECK:STDOUT:     %impl.elem0.loc280_3.1: @TestInc.%.loc280 (%.bdc) = impl_witness_access constants.%Inc.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc280_3.2 (constants.%impl.elem0.a96)]
// CHECK:STDOUT:     %bound_method.loc280_3.1: <bound method> = bound_method %x.ref, %impl.elem0.loc280_3.1
// CHECK:STDOUT:     %specific_impl_fn.loc280_3.1: <specific function> = specific_impl_function %impl.elem0.loc280_3.1, @Inc.WithSelf.Op(constants.%T.c06) [symbolic = %specific_impl_fn.loc280_3.2 (constants.%specific_impl_fn.d8c)]
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
// CHECK:STDOUT:   %Dec.WithSelf.Op.type: type = fn_type @Dec.WithSelf.Op, @Dec.WithSelf(%T.loc284_13.1) [symbolic = %Dec.WithSelf.Op.type (constants.%Dec.WithSelf.Op.type.c8aa8e.2)]
// CHECK:STDOUT:   %.loc286: type = fn_type_with_self_type %Dec.WithSelf.Op.type, %T.loc284_13.1 [symbolic = %.loc286 (constants.%.842)]
// CHECK:STDOUT:   %Dec.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc284_13.1, @Dec [symbolic = %Dec.lookup_impl_witness (constants.%Dec.lookup_impl_witness)]
// CHECK:STDOUT:   %impl.elem0.loc286_3.2: @TestDec.%.loc286 (%.842) = impl_witness_access %Dec.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc286_3.2 (constants.%impl.elem0.e17)]
// CHECK:STDOUT:   %specific_impl_fn.loc286_3.2: <specific function> = specific_impl_function %impl.elem0.loc286_3.2, @Dec.WithSelf.Op(%T.loc284_13.1) [symbolic = %specific_impl_fn.loc286_3.2 (constants.%specific_impl_fn.56c)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: ref @TestDec.%T.as_type.loc284_33.1 (%T.as_type.915)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: ref @TestDec.%T.as_type.loc284_33.1 (%T.as_type.915) = name_ref x, %x
// CHECK:STDOUT:     %impl.elem0.loc286_3.1: @TestDec.%.loc286 (%.842) = impl_witness_access constants.%Dec.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc286_3.2 (constants.%impl.elem0.e17)]
// CHECK:STDOUT:     %bound_method.loc286_3.1: <bound method> = bound_method %x.ref, %impl.elem0.loc286_3.1
// CHECK:STDOUT:     %specific_impl_fn.loc286_3.1: <specific function> = specific_impl_function %impl.elem0.loc286_3.1, @Dec.WithSelf.Op(constants.%T.d9e) [symbolic = %specific_impl_fn.loc286_3.2 (constants.%specific_impl_fn.56c)]
// CHECK:STDOUT:     %bound_method.loc286_3.2: <bound method> = bound_method %x.ref, %specific_impl_fn.loc286_3.1
// CHECK:STDOUT:     %Dec.WithSelf.Op.call: init %empty_tuple.type = call %bound_method.loc286_3.2(%x.ref)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @TestNegate(%Result.loc292_9.2: %Destroy.type, %T.loc293_4.2: @TestNegate.%facet_type (%facet_type.fb9)) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Negate.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc293_4.1, @Negate [symbolic = %Negate.lookup_impl_witness (constants.%Negate.lookup_impl_witness.5d0)]
// CHECK:STDOUT:   %Negate.facet: %Negate.type = facet_value %T.as_type.loc294_24.1, (%Negate.lookup_impl_witness) [symbolic = %Negate.facet (constants.%Negate.facet.7e7)]
// CHECK:STDOUT:   %Negate.WithSelf.Op.type: type = fn_type @Negate.WithSelf.Op, @Negate.WithSelf(%Negate.facet) [symbolic = %Negate.WithSelf.Op.type (constants.%Negate.WithSelf.Op.type.26a)]
// CHECK:STDOUT:   %.loc296_3.3: type = fn_type_with_self_type %Negate.WithSelf.Op.type, %Negate.facet [symbolic = %.loc296_3.3 (constants.%.0fd)]
// CHECK:STDOUT:   %impl.elem1.loc296_3.2: @TestNegate.%.loc296_3.3 (%.0fd) = impl_witness_access %Negate.lookup_impl_witness, element1 [symbolic = %impl.elem1.loc296_3.2 (constants.%impl.elem1.b4a)]
// CHECK:STDOUT:   %specific_impl_fn.loc296_3.3: <specific function> = specific_impl_function %impl.elem1.loc296_3.2, @Negate.WithSelf.Op(%Negate.facet) [symbolic = %specific_impl_fn.loc296_3.3 (constants.%specific_impl_fn.037)]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Result.loc292_9.1) [symbolic = %Destroy.WithSelf.Op.type (constants.%Destroy.WithSelf.Op.type.6304ea.2)]
// CHECK:STDOUT:   %.loc296_3.4: type = fn_type_with_self_type %Destroy.WithSelf.Op.type, %Result.loc292_9.1 [symbolic = %.loc296_3.4 (constants.%.543)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %Result.loc292_9.1, @Destroy [symbolic = %Destroy.lookup_impl_witness (constants.%Destroy.lookup_impl_witness)]
// CHECK:STDOUT:   %impl.elem0.loc296_3.2: @TestNegate.%.loc296_3.4 (%.543) = impl_witness_access %Destroy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc296_3.2 (constants.%impl.elem0.531)]
// CHECK:STDOUT:   %specific_impl_fn.loc296_3.4: <specific function> = specific_impl_function %impl.elem0.loc296_3.2, @Destroy.WithSelf.Op(%Result.loc292_9.1) [symbolic = %specific_impl_fn.loc296_3.4 (constants.%specific_impl_fn.010)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%y.param: @TestNegate.%Result.as_type.loc293_50.1 (%Result.as_type), %x.param: @TestNegate.%T.as_type.loc294_24.1 (%T.as_type.b01)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: @TestNegate.%T.as_type.loc294_24.1 (%T.as_type.b01) = name_ref x, %x
// CHECK:STDOUT:     %impl.elem1.loc296_3.1: @TestNegate.%.loc296_3.3 (%.0fd) = impl_witness_access constants.%Negate.lookup_impl_witness.5d0, element1 [symbolic = %impl.elem1.loc296_3.2 (constants.%impl.elem1.b4a)]
// CHECK:STDOUT:     %bound_method.loc296_3.1: <bound method> = bound_method %x.ref, %impl.elem1.loc296_3.1
// CHECK:STDOUT:     %specific_impl_fn.loc296_3.1: <specific function> = specific_impl_function %impl.elem1.loc296_3.1, @Negate.WithSelf.Op(constants.%Negate.facet.7e7) [symbolic = %specific_impl_fn.loc296_3.3 (constants.%specific_impl_fn.037)]
// CHECK:STDOUT:     %bound_method.loc296_3.2: <bound method> = bound_method %x.ref, %specific_impl_fn.loc296_3.1
// CHECK:STDOUT:     %.loc296_3.1: ref @TestNegate.%Result.as_type.loc293_50.1 (%Result.as_type) = temporary_storage
// CHECK:STDOUT:     %Negate.WithSelf.Op.call: init @TestNegate.%Result.as_type.loc293_50.1 (%Result.as_type) to %.loc296_3.1 = call %bound_method.loc296_3.2(%x.ref)
// CHECK:STDOUT:     %.loc296_3.2: ref @TestNegate.%Result.as_type.loc293_50.1 (%Result.as_type) = temporary %.loc296_3.1, %Negate.WithSelf.Op.call
// CHECK:STDOUT:     %impl.elem0.loc296_3.1: @TestNegate.%.loc296_3.4 (%.543) = impl_witness_access constants.%Destroy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc296_3.2 (constants.%impl.elem0.531)]
// CHECK:STDOUT:     %bound_method.loc296_3.3: <bound method> = bound_method %.loc296_3.2, %impl.elem0.loc296_3.1
// CHECK:STDOUT:     %specific_impl_fn.loc296_3.2: <specific function> = specific_impl_function %impl.elem0.loc296_3.1, @Destroy.WithSelf.Op(constants.%Result) [symbolic = %specific_impl_fn.loc296_3.4 (constants.%specific_impl_fn.010)]
// CHECK:STDOUT:     %bound_method.loc296_3.4: <bound method> = bound_method %.loc296_3.2, %specific_impl_fn.loc296_3.2
// CHECK:STDOUT:     %Destroy.WithSelf.Op.call: init %empty_tuple.type = call %bound_method.loc296_3.4(%.loc296_3.2)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AddWith.loc124(constants.%Result, constants.%U.99e, constants.%T.23b) {
// CHECK:STDOUT:   %Result.loc121_9.1 => constants.%Result
// CHECK:STDOUT:   %U.loc122_4.1 => constants.%U.99e
// CHECK:STDOUT:   %U.as_type.loc123_36.1 => constants.%U.as_type.f7f
// CHECK:STDOUT:   %AddWith.type.loc123_36.1 => constants.%AddWith.type.ad2
// CHECK:STDOUT:   %facet_type.loc123_20 => constants.%facet_type.4ec
// CHECK:STDOUT:   %.Self.loc123_38.1 => constants.%.Self.53a
// CHECK:STDOUT:   %require_complete.loc123 => constants.%require_complete.3ac
// CHECK:STDOUT:   %.Self.as_type.loc123_44.1 => constants.%.Self.as_type.626
// CHECK:STDOUT:   %AddWith.assoc_type => constants.%AddWith.assoc_type.c6c
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.35c
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.loc123 => constants.%AddWith.lookup_impl_witness.46e
// CHECK:STDOUT:   %impl.elem0.loc123_44.1 => constants.%impl.elem0.851
// CHECK:STDOUT:   %Result.as_type.loc123_54.1 => constants.%Result.as_type
// CHECK:STDOUT:   %facet_type.loc123_38 => constants.%facet_type.cac
// CHECK:STDOUT:   %T.loc123_4.1 => constants.%T.23b
// CHECK:STDOUT:   %pattern_type.loc123 => constants.%pattern_type.bcc
// CHECK:STDOUT:   %pattern_type.loc124_16 => constants.%pattern_type.227a41.2
// CHECK:STDOUT:   %T.as_type.loc124_29.1 => constants.%T.as_type.46a
// CHECK:STDOUT:   %pattern_type.loc124_27 => constants.%pattern_type.d37
// CHECK:STDOUT:   %pattern_type.loc124_33 => constants.%pattern_type.87c
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @SubWith.loc135(constants.%Result, constants.%U.99e, constants.%T.4a2) {
// CHECK:STDOUT:   %Result.loc132_9.1 => constants.%Result
// CHECK:STDOUT:   %U.loc133_4.1 => constants.%U.99e
// CHECK:STDOUT:   %U.as_type.loc134_36.1 => constants.%U.as_type.f7f
// CHECK:STDOUT:   %SubWith.type.loc134_36.1 => constants.%SubWith.type.394
// CHECK:STDOUT:   %facet_type.loc134_20 => constants.%facet_type.c40
// CHECK:STDOUT:   %.Self.loc134_38.1 => constants.%.Self.f02
// CHECK:STDOUT:   %require_complete.loc134 => constants.%require_complete.f69
// CHECK:STDOUT:   %.Self.as_type.loc134_44.1 => constants.%.Self.as_type.dc7
// CHECK:STDOUT:   %SubWith.assoc_type => constants.%SubWith.assoc_type.767
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.705
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.loc134 => constants.%SubWith.lookup_impl_witness.54d
// CHECK:STDOUT:   %impl.elem0.loc134_44.1 => constants.%impl.elem0.67e
// CHECK:STDOUT:   %Result.as_type.loc134_54.1 => constants.%Result.as_type
// CHECK:STDOUT:   %facet_type.loc134_38 => constants.%facet_type.98a
// CHECK:STDOUT:   %T.loc134_4.1 => constants.%T.4a2
// CHECK:STDOUT:   %pattern_type.loc134 => constants.%pattern_type.054
// CHECK:STDOUT:   %pattern_type.loc135_16 => constants.%pattern_type.227a41.2
// CHECK:STDOUT:   %T.as_type.loc135_29.1 => constants.%T.as_type.330
// CHECK:STDOUT:   %pattern_type.loc135_27 => constants.%pattern_type.2b7
// CHECK:STDOUT:   %pattern_type.loc135_33 => constants.%pattern_type.87c
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @MulWith.loc146(constants.%Result, constants.%U.99e, constants.%T.2c2) {
// CHECK:STDOUT:   %Result.loc143_9.1 => constants.%Result
// CHECK:STDOUT:   %U.loc144_4.1 => constants.%U.99e
// CHECK:STDOUT:   %U.as_type.loc145_36.1 => constants.%U.as_type.f7f
// CHECK:STDOUT:   %MulWith.type.loc145_36.1 => constants.%MulWith.type.195
// CHECK:STDOUT:   %facet_type.loc145_20 => constants.%facet_type.e45
// CHECK:STDOUT:   %.Self.loc145_38.1 => constants.%.Self.117
// CHECK:STDOUT:   %require_complete.loc145 => constants.%require_complete.aad
// CHECK:STDOUT:   %.Self.as_type.loc145_44.1 => constants.%.Self.as_type.91e
// CHECK:STDOUT:   %MulWith.assoc_type => constants.%MulWith.assoc_type.fff
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.dfc
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.loc145 => constants.%MulWith.lookup_impl_witness.bd7
// CHECK:STDOUT:   %impl.elem0.loc145_44.1 => constants.%impl.elem0.e5b
// CHECK:STDOUT:   %Result.as_type.loc145_54.1 => constants.%Result.as_type
// CHECK:STDOUT:   %facet_type.loc145_38 => constants.%facet_type.a59
// CHECK:STDOUT:   %T.loc145_4.1 => constants.%T.2c2
// CHECK:STDOUT:   %pattern_type.loc145 => constants.%pattern_type.4b6
// CHECK:STDOUT:   %pattern_type.loc146_16 => constants.%pattern_type.227a41.2
// CHECK:STDOUT:   %T.as_type.loc146_29.1 => constants.%T.as_type.198
// CHECK:STDOUT:   %pattern_type.loc146_27 => constants.%pattern_type.2a1
// CHECK:STDOUT:   %pattern_type.loc146_33 => constants.%pattern_type.87c
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @DivWith.loc157(constants.%Result, constants.%U.99e, constants.%T.2b9) {
// CHECK:STDOUT:   %Result.loc154_9.1 => constants.%Result
// CHECK:STDOUT:   %U.loc155_4.1 => constants.%U.99e
// CHECK:STDOUT:   %U.as_type.loc156_36.1 => constants.%U.as_type.f7f
// CHECK:STDOUT:   %DivWith.type.loc156_36.1 => constants.%DivWith.type.63d
// CHECK:STDOUT:   %facet_type.loc156_20 => constants.%facet_type.511
// CHECK:STDOUT:   %.Self.loc156_38.1 => constants.%.Self.10f
// CHECK:STDOUT:   %require_complete.loc156 => constants.%require_complete.8fe
// CHECK:STDOUT:   %.Self.as_type.loc156_44.1 => constants.%.Self.as_type.151
// CHECK:STDOUT:   %DivWith.assoc_type => constants.%DivWith.assoc_type.e27
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.46e
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.loc156 => constants.%DivWith.lookup_impl_witness.0c0
// CHECK:STDOUT:   %impl.elem0.loc156_44.1 => constants.%impl.elem0.2d7
// CHECK:STDOUT:   %Result.as_type.loc156_54.1 => constants.%Result.as_type
// CHECK:STDOUT:   %facet_type.loc156_38 => constants.%facet_type.5cc
// CHECK:STDOUT:   %T.loc156_4.1 => constants.%T.2b9
// CHECK:STDOUT:   %pattern_type.loc156 => constants.%pattern_type.fb6
// CHECK:STDOUT:   %pattern_type.loc157_16 => constants.%pattern_type.227a41.2
// CHECK:STDOUT:   %T.as_type.loc157_29.1 => constants.%T.as_type.e49
// CHECK:STDOUT:   %pattern_type.loc157_27 => constants.%pattern_type.673
// CHECK:STDOUT:   %pattern_type.loc157_33 => constants.%pattern_type.87c
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ModWith.loc168(constants.%Result, constants.%U.99e, constants.%T.173) {
// CHECK:STDOUT:   %Result.loc165_9.1 => constants.%Result
// CHECK:STDOUT:   %U.loc166_4.1 => constants.%U.99e
// CHECK:STDOUT:   %U.as_type.loc167_36.1 => constants.%U.as_type.f7f
// CHECK:STDOUT:   %ModWith.type.loc167_36.1 => constants.%ModWith.type.125
// CHECK:STDOUT:   %facet_type.loc167_20 => constants.%facet_type.978
// CHECK:STDOUT:   %.Self.loc167_38.1 => constants.%.Self.0e2
// CHECK:STDOUT:   %require_complete.loc167 => constants.%require_complete.4ae
// CHECK:STDOUT:   %.Self.as_type.loc167_44.1 => constants.%.Self.as_type.d0f
// CHECK:STDOUT:   %ModWith.assoc_type => constants.%ModWith.assoc_type.3d0
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.649
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.loc167 => constants.%ModWith.lookup_impl_witness.ac5
// CHECK:STDOUT:   %impl.elem0.loc167_44.1 => constants.%impl.elem0.50a
// CHECK:STDOUT:   %Result.as_type.loc167_54.1 => constants.%Result.as_type
// CHECK:STDOUT:   %facet_type.loc167_38 => constants.%facet_type.c32
// CHECK:STDOUT:   %T.loc167_4.1 => constants.%T.173
// CHECK:STDOUT:   %pattern_type.loc167 => constants.%pattern_type.cc0
// CHECK:STDOUT:   %pattern_type.loc168_16 => constants.%pattern_type.227a41.2
// CHECK:STDOUT:   %T.as_type.loc168_29.1 => constants.%T.as_type.105
// CHECK:STDOUT:   %pattern_type.loc168_27 => constants.%pattern_type.9c5
// CHECK:STDOUT:   %pattern_type.loc168_33 => constants.%pattern_type.87c
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AddWith.loc124(constants.%Destroy.facet.1b8, constants.%Destroy.facet.1b8, constants.%facet_value.fdd) {
// CHECK:STDOUT:   %Result.loc121_9.1 => constants.%Destroy.facet.1b8
// CHECK:STDOUT:   %U.loc122_4.1 => constants.%Destroy.facet.1b8
// CHECK:STDOUT:   %U.as_type.loc123_36.1 => constants.%Int16
// CHECK:STDOUT:   %AddWith.type.loc123_36.1 => constants.%AddWith.type.6f4
// CHECK:STDOUT:   %facet_type.loc123_20 => constants.%facet_type.f17
// CHECK:STDOUT:   %.Self.loc123_38.1 => constants.%.Self.863
// CHECK:STDOUT:   %require_complete.loc123 => constants.%complete_type.211
// CHECK:STDOUT:   %.Self.as_type.loc123_44.1 => constants.%.Self.as_type.db9
// CHECK:STDOUT:   %AddWith.assoc_type => constants.%AddWith.assoc_type.e37
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.4ec
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.loc123 => constants.%AddWith.lookup_impl_witness.896
// CHECK:STDOUT:   %impl.elem0.loc123_44.1 => constants.%impl.elem0.2d1
// CHECK:STDOUT:   %Result.as_type.loc123_54.1 => constants.%Int16
// CHECK:STDOUT:   %facet_type.loc123_38 => constants.%facet_type.753
// CHECK:STDOUT:   %T.loc123_4.1 => constants.%facet_value.fdd
// CHECK:STDOUT:   %pattern_type.loc123 => constants.%pattern_type.bc4
// CHECK:STDOUT:   %pattern_type.loc124_16 => constants.%pattern_type.802
// CHECK:STDOUT:   %T.as_type.loc124_29.1 => constants.%i16
// CHECK:STDOUT:   %pattern_type.loc124_27 => constants.%pattern_type.e1b
// CHECK:STDOUT:   %pattern_type.loc124_33 => constants.%pattern_type.802
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc124_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc124_27 => constants.%complete_type.2da
// CHECK:STDOUT:   %require_complete.loc124_33 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc126 => constants.%complete_type.386
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.c66
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.loc126 => constants.%custom_witness.604
// CHECK:STDOUT:   %AddWith.facet => constants.%AddWith.facet.85f
// CHECK:STDOUT:   %AddWith.WithSelf.Op.type => constants.%AddWith.WithSelf.Op.type.09b
// CHECK:STDOUT:   %.loc126_5.4 => constants.%.b5b
// CHECK:STDOUT:   %impl.elem1.loc126_5.2 => constants.%Op.b9c47d.1
// CHECK:STDOUT:   %specific_impl_fn.loc126_5.3 => constants.%Op.b9c47d.1
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.e80
// CHECK:STDOUT:   %.loc126_5.5 => constants.%.154
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.a98c63.1
// CHECK:STDOUT:   %impl.elem0.loc126_5.2 => constants.%Int16.Op.cc70d5.1
// CHECK:STDOUT:   %specific_impl_fn.loc126_5.4 => constants.%Int16.Op.cc70d5.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @SubWith.loc135(constants.%Destroy.facet.1b8, constants.%Destroy.facet.1b8, constants.%facet_value.578) {
// CHECK:STDOUT:   %Result.loc132_9.1 => constants.%Destroy.facet.1b8
// CHECK:STDOUT:   %U.loc133_4.1 => constants.%Destroy.facet.1b8
// CHECK:STDOUT:   %U.as_type.loc134_36.1 => constants.%Int16
// CHECK:STDOUT:   %SubWith.type.loc134_36.1 => constants.%SubWith.type.8bd
// CHECK:STDOUT:   %facet_type.loc134_20 => constants.%facet_type.c80
// CHECK:STDOUT:   %.Self.loc134_38.1 => constants.%.Self.700
// CHECK:STDOUT:   %require_complete.loc134 => constants.%complete_type.a74
// CHECK:STDOUT:   %.Self.as_type.loc134_44.1 => constants.%.Self.as_type.c4d
// CHECK:STDOUT:   %SubWith.assoc_type => constants.%SubWith.assoc_type.eb8
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.8a1
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.loc134 => constants.%SubWith.lookup_impl_witness.e8f
// CHECK:STDOUT:   %impl.elem0.loc134_44.1 => constants.%impl.elem0.0b0
// CHECK:STDOUT:   %Result.as_type.loc134_54.1 => constants.%Int16
// CHECK:STDOUT:   %facet_type.loc134_38 => constants.%facet_type.5f5
// CHECK:STDOUT:   %T.loc134_4.1 => constants.%facet_value.578
// CHECK:STDOUT:   %pattern_type.loc134 => constants.%pattern_type.4d6
// CHECK:STDOUT:   %pattern_type.loc135_16 => constants.%pattern_type.802
// CHECK:STDOUT:   %T.as_type.loc135_29.1 => constants.%i16
// CHECK:STDOUT:   %pattern_type.loc135_27 => constants.%pattern_type.e1b
// CHECK:STDOUT:   %pattern_type.loc135_33 => constants.%pattern_type.802
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc135_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc135_27 => constants.%complete_type.2da
// CHECK:STDOUT:   %require_complete.loc135_33 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc137 => constants.%complete_type.41c
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.937f
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.loc137 => constants.%custom_witness.4c1
// CHECK:STDOUT:   %SubWith.facet => constants.%SubWith.facet.1de
// CHECK:STDOUT:   %SubWith.WithSelf.Op.type => constants.%SubWith.WithSelf.Op.type.fc2
// CHECK:STDOUT:   %.loc137_5.4 => constants.%.ec9
// CHECK:STDOUT:   %impl.elem1.loc137_5.2 => constants.%Op.b9c47d.2
// CHECK:STDOUT:   %specific_impl_fn.loc137_5.3 => constants.%Op.b9c47d.2
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.e80
// CHECK:STDOUT:   %.loc137_5.5 => constants.%.154
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.a98c63.1
// CHECK:STDOUT:   %impl.elem0.loc137_5.2 => constants.%Int16.Op.cc70d5.1
// CHECK:STDOUT:   %specific_impl_fn.loc137_5.4 => constants.%Int16.Op.cc70d5.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @MulWith.loc146(constants.%Destroy.facet.1b8, constants.%Destroy.facet.1b8, constants.%facet_value.1ee) {
// CHECK:STDOUT:   %Result.loc143_9.1 => constants.%Destroy.facet.1b8
// CHECK:STDOUT:   %U.loc144_4.1 => constants.%Destroy.facet.1b8
// CHECK:STDOUT:   %U.as_type.loc145_36.1 => constants.%Int16
// CHECK:STDOUT:   %MulWith.type.loc145_36.1 => constants.%MulWith.type.c25
// CHECK:STDOUT:   %facet_type.loc145_20 => constants.%facet_type.d27
// CHECK:STDOUT:   %.Self.loc145_38.1 => constants.%.Self.e39
// CHECK:STDOUT:   %require_complete.loc145 => constants.%complete_type.aa7
// CHECK:STDOUT:   %.Self.as_type.loc145_44.1 => constants.%.Self.as_type.87e
// CHECK:STDOUT:   %MulWith.assoc_type => constants.%MulWith.assoc_type.543
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.81e
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.loc145 => constants.%MulWith.lookup_impl_witness.5e1
// CHECK:STDOUT:   %impl.elem0.loc145_44.1 => constants.%impl.elem0.e54
// CHECK:STDOUT:   %Result.as_type.loc145_54.1 => constants.%Int16
// CHECK:STDOUT:   %facet_type.loc145_38 => constants.%facet_type.9da
// CHECK:STDOUT:   %T.loc145_4.1 => constants.%facet_value.1ee
// CHECK:STDOUT:   %pattern_type.loc145 => constants.%pattern_type.d31
// CHECK:STDOUT:   %pattern_type.loc146_16 => constants.%pattern_type.802
// CHECK:STDOUT:   %T.as_type.loc146_29.1 => constants.%i16
// CHECK:STDOUT:   %pattern_type.loc146_27 => constants.%pattern_type.e1b
// CHECK:STDOUT:   %pattern_type.loc146_33 => constants.%pattern_type.802
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc146_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc146_27 => constants.%complete_type.2da
// CHECK:STDOUT:   %require_complete.loc146_33 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc148 => constants.%complete_type.b16
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.911
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.loc148 => constants.%custom_witness.72a
// CHECK:STDOUT:   %MulWith.facet => constants.%MulWith.facet.d75
// CHECK:STDOUT:   %MulWith.WithSelf.Op.type => constants.%MulWith.WithSelf.Op.type.53a
// CHECK:STDOUT:   %.loc148_5.4 => constants.%.f80
// CHECK:STDOUT:   %impl.elem1.loc148_5.2 => constants.%Op.b9c47d.3
// CHECK:STDOUT:   %specific_impl_fn.loc148_5.3 => constants.%Op.b9c47d.3
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.e80
// CHECK:STDOUT:   %.loc148_5.5 => constants.%.154
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.a98c63.1
// CHECK:STDOUT:   %impl.elem0.loc148_5.2 => constants.%Int16.Op.cc70d5.1
// CHECK:STDOUT:   %specific_impl_fn.loc148_5.4 => constants.%Int16.Op.cc70d5.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @DivWith.loc157(constants.%Destroy.facet.1b8, constants.%Destroy.facet.1b8, constants.%facet_value.6b2) {
// CHECK:STDOUT:   %Result.loc154_9.1 => constants.%Destroy.facet.1b8
// CHECK:STDOUT:   %U.loc155_4.1 => constants.%Destroy.facet.1b8
// CHECK:STDOUT:   %U.as_type.loc156_36.1 => constants.%Int16
// CHECK:STDOUT:   %DivWith.type.loc156_36.1 => constants.%DivWith.type.053
// CHECK:STDOUT:   %facet_type.loc156_20 => constants.%facet_type.8eb
// CHECK:STDOUT:   %.Self.loc156_38.1 => constants.%.Self.467
// CHECK:STDOUT:   %require_complete.loc156 => constants.%complete_type.d563
// CHECK:STDOUT:   %.Self.as_type.loc156_44.1 => constants.%.Self.as_type.3cd
// CHECK:STDOUT:   %DivWith.assoc_type => constants.%DivWith.assoc_type.73e
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.e96
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.loc156 => constants.%DivWith.lookup_impl_witness.9bc
// CHECK:STDOUT:   %impl.elem0.loc156_44.1 => constants.%impl.elem0.5c6
// CHECK:STDOUT:   %Result.as_type.loc156_54.1 => constants.%Int16
// CHECK:STDOUT:   %facet_type.loc156_38 => constants.%facet_type.7f5
// CHECK:STDOUT:   %T.loc156_4.1 => constants.%facet_value.6b2
// CHECK:STDOUT:   %pattern_type.loc156 => constants.%pattern_type.2be
// CHECK:STDOUT:   %pattern_type.loc157_16 => constants.%pattern_type.802
// CHECK:STDOUT:   %T.as_type.loc157_29.1 => constants.%i16
// CHECK:STDOUT:   %pattern_type.loc157_27 => constants.%pattern_type.e1b
// CHECK:STDOUT:   %pattern_type.loc157_33 => constants.%pattern_type.802
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc157_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc157_27 => constants.%complete_type.2da
// CHECK:STDOUT:   %require_complete.loc157_33 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc159 => constants.%complete_type.7c5
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.10b
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.loc159 => constants.%custom_witness.6ea
// CHECK:STDOUT:   %DivWith.facet => constants.%DivWith.facet.2be
// CHECK:STDOUT:   %DivWith.WithSelf.Op.type => constants.%DivWith.WithSelf.Op.type.88f
// CHECK:STDOUT:   %.loc159_5.4 => constants.%.a93
// CHECK:STDOUT:   %impl.elem1.loc159_5.2 => constants.%Op.b9c47d.4
// CHECK:STDOUT:   %specific_impl_fn.loc159_5.3 => constants.%Op.b9c47d.4
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.e80
// CHECK:STDOUT:   %.loc159_5.5 => constants.%.154
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.a98c63.1
// CHECK:STDOUT:   %impl.elem0.loc159_5.2 => constants.%Int16.Op.cc70d5.1
// CHECK:STDOUT:   %specific_impl_fn.loc159_5.4 => constants.%Int16.Op.cc70d5.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ModWith.loc168(constants.%Destroy.facet.1b8, constants.%Destroy.facet.1b8, constants.%facet_value.07d) {
// CHECK:STDOUT:   %Result.loc165_9.1 => constants.%Destroy.facet.1b8
// CHECK:STDOUT:   %U.loc166_4.1 => constants.%Destroy.facet.1b8
// CHECK:STDOUT:   %U.as_type.loc167_36.1 => constants.%Int16
// CHECK:STDOUT:   %ModWith.type.loc167_36.1 => constants.%ModWith.type.a83
// CHECK:STDOUT:   %facet_type.loc167_20 => constants.%facet_type.dc7
// CHECK:STDOUT:   %.Self.loc167_38.1 => constants.%.Self.703
// CHECK:STDOUT:   %require_complete.loc167 => constants.%complete_type.65d
// CHECK:STDOUT:   %.Self.as_type.loc167_44.1 => constants.%.Self.as_type.c72
// CHECK:STDOUT:   %ModWith.assoc_type => constants.%ModWith.assoc_type.55b
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.8a3
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.loc167 => constants.%ModWith.lookup_impl_witness.c7a
// CHECK:STDOUT:   %impl.elem0.loc167_44.1 => constants.%impl.elem0.094
// CHECK:STDOUT:   %Result.as_type.loc167_54.1 => constants.%Int16
// CHECK:STDOUT:   %facet_type.loc167_38 => constants.%facet_type.fde
// CHECK:STDOUT:   %T.loc167_4.1 => constants.%facet_value.07d
// CHECK:STDOUT:   %pattern_type.loc167 => constants.%pattern_type.268
// CHECK:STDOUT:   %pattern_type.loc168_16 => constants.%pattern_type.802
// CHECK:STDOUT:   %T.as_type.loc168_29.1 => constants.%i16
// CHECK:STDOUT:   %pattern_type.loc168_27 => constants.%pattern_type.e1b
// CHECK:STDOUT:   %pattern_type.loc168_33 => constants.%pattern_type.802
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc168_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc168_27 => constants.%complete_type.2da
// CHECK:STDOUT:   %require_complete.loc168_33 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc170 => constants.%complete_type.2f8
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.7d4
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.loc170 => constants.%custom_witness.b81
// CHECK:STDOUT:   %ModWith.facet => constants.%ModWith.facet.ea4
// CHECK:STDOUT:   %ModWith.WithSelf.Op.type => constants.%ModWith.WithSelf.Op.type.6fc
// CHECK:STDOUT:   %.loc170_5.4 => constants.%.3cd
// CHECK:STDOUT:   %impl.elem1.loc170_5.2 => constants.%Op.b9c47d.5
// CHECK:STDOUT:   %specific_impl_fn.loc170_5.3 => constants.%Op.b9c47d.5
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.e80
// CHECK:STDOUT:   %.loc170_5.5 => constants.%.154
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.a98c63.1
// CHECK:STDOUT:   %impl.elem0.loc170_5.2 => constants.%Int16.Op.cc70d5.1
// CHECK:STDOUT:   %specific_impl_fn.loc170_5.4 => constants.%Int16.Op.cc70d5.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AddWith.loc124(constants.%Destroy.facet.1b8, constants.%Destroy.facet.181, constants.%facet_value.b21) {
// CHECK:STDOUT:   %Result.loc121_9.1 => constants.%Destroy.facet.1b8
// CHECK:STDOUT:   %U.loc122_4.1 => constants.%Destroy.facet.181
// CHECK:STDOUT:   %U.as_type.loc123_36.1 => constants.%i16
// CHECK:STDOUT:   %AddWith.type.loc123_36.1 => constants.%AddWith.type.63e
// CHECK:STDOUT:   %facet_type.loc123_20 => constants.%facet_type.bf5
// CHECK:STDOUT:   %.Self.loc123_38.1 => constants.%.Self.b0b
// CHECK:STDOUT:   %require_complete.loc123 => constants.%complete_type.b92
// CHECK:STDOUT:   %.Self.as_type.loc123_44.1 => constants.%.Self.as_type.ed9
// CHECK:STDOUT:   %AddWith.assoc_type => constants.%AddWith.assoc_type.fdd
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.c67
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.loc123 => constants.%AddWith.lookup_impl_witness.2ab
// CHECK:STDOUT:   %impl.elem0.loc123_44.1 => constants.%impl.elem0.b1c
// CHECK:STDOUT:   %Result.as_type.loc123_54.1 => constants.%Int16
// CHECK:STDOUT:   %facet_type.loc123_38 => constants.%facet_type.c90
// CHECK:STDOUT:   %T.loc123_4.1 => constants.%facet_value.b21
// CHECK:STDOUT:   %pattern_type.loc123 => constants.%pattern_type.a15
// CHECK:STDOUT:   %pattern_type.loc124_16 => constants.%pattern_type.802
// CHECK:STDOUT:   %T.as_type.loc124_29.1 => constants.%Int16
// CHECK:STDOUT:   %pattern_type.loc124_27 => constants.%pattern_type.802
// CHECK:STDOUT:   %pattern_type.loc124_33 => constants.%pattern_type.e1b
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc124_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc124_27 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc124_33 => constants.%complete_type.2da
// CHECK:STDOUT:   %require_complete.loc126 => constants.%complete_type.b45
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.0e4
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.loc126 => constants.%custom_witness.9f2
// CHECK:STDOUT:   %AddWith.facet => constants.%AddWith.facet.cea
// CHECK:STDOUT:   %AddWith.WithSelf.Op.type => constants.%AddWith.WithSelf.Op.type.18e
// CHECK:STDOUT:   %.loc126_5.4 => constants.%.1c0
// CHECK:STDOUT:   %impl.elem1.loc126_5.2 => constants.%Int16.Op.cc70d5.3
// CHECK:STDOUT:   %specific_impl_fn.loc126_5.3 => constants.%Int16.Op.cc70d5.3
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.e80
// CHECK:STDOUT:   %.loc126_5.5 => constants.%.154
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.a98c63.1
// CHECK:STDOUT:   %impl.elem0.loc126_5.2 => constants.%Int16.Op.cc70d5.1
// CHECK:STDOUT:   %specific_impl_fn.loc126_5.4 => constants.%Int16.Op.cc70d5.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @SubWith.loc135(constants.%Destroy.facet.1b8, constants.%Destroy.facet.181, constants.%facet_value.2b4) {
// CHECK:STDOUT:   %Result.loc132_9.1 => constants.%Destroy.facet.1b8
// CHECK:STDOUT:   %U.loc133_4.1 => constants.%Destroy.facet.181
// CHECK:STDOUT:   %U.as_type.loc134_36.1 => constants.%i16
// CHECK:STDOUT:   %SubWith.type.loc134_36.1 => constants.%SubWith.type.1db
// CHECK:STDOUT:   %facet_type.loc134_20 => constants.%facet_type.5e7
// CHECK:STDOUT:   %.Self.loc134_38.1 => constants.%.Self.bea
// CHECK:STDOUT:   %require_complete.loc134 => constants.%complete_type.9fc
// CHECK:STDOUT:   %.Self.as_type.loc134_44.1 => constants.%.Self.as_type.b77
// CHECK:STDOUT:   %SubWith.assoc_type => constants.%SubWith.assoc_type.5df
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.870
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.loc134 => constants.%SubWith.lookup_impl_witness.7b4
// CHECK:STDOUT:   %impl.elem0.loc134_44.1 => constants.%impl.elem0.61d
// CHECK:STDOUT:   %Result.as_type.loc134_54.1 => constants.%Int16
// CHECK:STDOUT:   %facet_type.loc134_38 => constants.%facet_type.80f
// CHECK:STDOUT:   %T.loc134_4.1 => constants.%facet_value.2b4
// CHECK:STDOUT:   %pattern_type.loc134 => constants.%pattern_type.7a0
// CHECK:STDOUT:   %pattern_type.loc135_16 => constants.%pattern_type.802
// CHECK:STDOUT:   %T.as_type.loc135_29.1 => constants.%Int16
// CHECK:STDOUT:   %pattern_type.loc135_27 => constants.%pattern_type.802
// CHECK:STDOUT:   %pattern_type.loc135_33 => constants.%pattern_type.e1b
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc135_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc135_27 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc135_33 => constants.%complete_type.2da
// CHECK:STDOUT:   %require_complete.loc137 => constants.%complete_type.edc
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.b6b
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.loc137 => constants.%custom_witness.6a5
// CHECK:STDOUT:   %SubWith.facet => constants.%SubWith.facet.3dd
// CHECK:STDOUT:   %SubWith.WithSelf.Op.type => constants.%SubWith.WithSelf.Op.type.1ca
// CHECK:STDOUT:   %.loc137_5.4 => constants.%.f6d
// CHECK:STDOUT:   %impl.elem1.loc137_5.2 => constants.%Int16.Op.cc70d5.4
// CHECK:STDOUT:   %specific_impl_fn.loc137_5.3 => constants.%Int16.Op.cc70d5.4
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.e80
// CHECK:STDOUT:   %.loc137_5.5 => constants.%.154
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.a98c63.1
// CHECK:STDOUT:   %impl.elem0.loc137_5.2 => constants.%Int16.Op.cc70d5.1
// CHECK:STDOUT:   %specific_impl_fn.loc137_5.4 => constants.%Int16.Op.cc70d5.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @MulWith.loc146(constants.%Destroy.facet.1b8, constants.%Destroy.facet.181, constants.%facet_value.88a) {
// CHECK:STDOUT:   %Result.loc143_9.1 => constants.%Destroy.facet.1b8
// CHECK:STDOUT:   %U.loc144_4.1 => constants.%Destroy.facet.181
// CHECK:STDOUT:   %U.as_type.loc145_36.1 => constants.%i16
// CHECK:STDOUT:   %MulWith.type.loc145_36.1 => constants.%MulWith.type.c35
// CHECK:STDOUT:   %facet_type.loc145_20 => constants.%facet_type.183
// CHECK:STDOUT:   %.Self.loc145_38.1 => constants.%.Self.0ef
// CHECK:STDOUT:   %require_complete.loc145 => constants.%complete_type.309
// CHECK:STDOUT:   %.Self.as_type.loc145_44.1 => constants.%.Self.as_type.751
// CHECK:STDOUT:   %MulWith.assoc_type => constants.%MulWith.assoc_type.378
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.7cb
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.loc145 => constants.%MulWith.lookup_impl_witness.b3c
// CHECK:STDOUT:   %impl.elem0.loc145_44.1 => constants.%impl.elem0.4a2
// CHECK:STDOUT:   %Result.as_type.loc145_54.1 => constants.%Int16
// CHECK:STDOUT:   %facet_type.loc145_38 => constants.%facet_type.6bb
// CHECK:STDOUT:   %T.loc145_4.1 => constants.%facet_value.88a
// CHECK:STDOUT:   %pattern_type.loc145 => constants.%pattern_type.fff
// CHECK:STDOUT:   %pattern_type.loc146_16 => constants.%pattern_type.802
// CHECK:STDOUT:   %T.as_type.loc146_29.1 => constants.%Int16
// CHECK:STDOUT:   %pattern_type.loc146_27 => constants.%pattern_type.802
// CHECK:STDOUT:   %pattern_type.loc146_33 => constants.%pattern_type.e1b
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc146_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc146_27 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc146_33 => constants.%complete_type.2da
// CHECK:STDOUT:   %require_complete.loc148 => constants.%complete_type.355
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.9b3
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.loc148 => constants.%custom_witness.35c
// CHECK:STDOUT:   %MulWith.facet => constants.%MulWith.facet.3b5
// CHECK:STDOUT:   %MulWith.WithSelf.Op.type => constants.%MulWith.WithSelf.Op.type.896
// CHECK:STDOUT:   %.loc148_5.4 => constants.%.ca0
// CHECK:STDOUT:   %impl.elem1.loc148_5.2 => constants.%Int16.Op.cc70d5.5
// CHECK:STDOUT:   %specific_impl_fn.loc148_5.3 => constants.%Int16.Op.cc70d5.5
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.e80
// CHECK:STDOUT:   %.loc148_5.5 => constants.%.154
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.a98c63.1
// CHECK:STDOUT:   %impl.elem0.loc148_5.2 => constants.%Int16.Op.cc70d5.1
// CHECK:STDOUT:   %specific_impl_fn.loc148_5.4 => constants.%Int16.Op.cc70d5.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @DivWith.loc157(constants.%Destroy.facet.1b8, constants.%Destroy.facet.181, constants.%facet_value.473) {
// CHECK:STDOUT:   %Result.loc154_9.1 => constants.%Destroy.facet.1b8
// CHECK:STDOUT:   %U.loc155_4.1 => constants.%Destroy.facet.181
// CHECK:STDOUT:   %U.as_type.loc156_36.1 => constants.%i16
// CHECK:STDOUT:   %DivWith.type.loc156_36.1 => constants.%DivWith.type.086
// CHECK:STDOUT:   %facet_type.loc156_20 => constants.%facet_type.ab4
// CHECK:STDOUT:   %.Self.loc156_38.1 => constants.%.Self.627
// CHECK:STDOUT:   %require_complete.loc156 => constants.%complete_type.d56e
// CHECK:STDOUT:   %.Self.as_type.loc156_44.1 => constants.%.Self.as_type.4c8
// CHECK:STDOUT:   %DivWith.assoc_type => constants.%DivWith.assoc_type.d3a
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.8b8
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.loc156 => constants.%DivWith.lookup_impl_witness.3d7
// CHECK:STDOUT:   %impl.elem0.loc156_44.1 => constants.%impl.elem0.ce6
// CHECK:STDOUT:   %Result.as_type.loc156_54.1 => constants.%Int16
// CHECK:STDOUT:   %facet_type.loc156_38 => constants.%facet_type.6b1
// CHECK:STDOUT:   %T.loc156_4.1 => constants.%facet_value.473
// CHECK:STDOUT:   %pattern_type.loc156 => constants.%pattern_type.92a
// CHECK:STDOUT:   %pattern_type.loc157_16 => constants.%pattern_type.802
// CHECK:STDOUT:   %T.as_type.loc157_29.1 => constants.%Int16
// CHECK:STDOUT:   %pattern_type.loc157_27 => constants.%pattern_type.802
// CHECK:STDOUT:   %pattern_type.loc157_33 => constants.%pattern_type.e1b
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc157_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc157_27 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc157_33 => constants.%complete_type.2da
// CHECK:STDOUT:   %require_complete.loc159 => constants.%complete_type.db2
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.acb
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.loc159 => constants.%custom_witness.52d
// CHECK:STDOUT:   %DivWith.facet => constants.%DivWith.facet.8d2
// CHECK:STDOUT:   %DivWith.WithSelf.Op.type => constants.%DivWith.WithSelf.Op.type.2d4
// CHECK:STDOUT:   %.loc159_5.4 => constants.%.cb6e
// CHECK:STDOUT:   %impl.elem1.loc159_5.2 => constants.%Int16.Op.cc70d5.6
// CHECK:STDOUT:   %specific_impl_fn.loc159_5.3 => constants.%Int16.Op.cc70d5.6
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.e80
// CHECK:STDOUT:   %.loc159_5.5 => constants.%.154
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.a98c63.1
// CHECK:STDOUT:   %impl.elem0.loc159_5.2 => constants.%Int16.Op.cc70d5.1
// CHECK:STDOUT:   %specific_impl_fn.loc159_5.4 => constants.%Int16.Op.cc70d5.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ModWith.loc168(constants.%Destroy.facet.1b8, constants.%Destroy.facet.181, constants.%facet_value.c3b) {
// CHECK:STDOUT:   %Result.loc165_9.1 => constants.%Destroy.facet.1b8
// CHECK:STDOUT:   %U.loc166_4.1 => constants.%Destroy.facet.181
// CHECK:STDOUT:   %U.as_type.loc167_36.1 => constants.%i16
// CHECK:STDOUT:   %ModWith.type.loc167_36.1 => constants.%ModWith.type.59b
// CHECK:STDOUT:   %facet_type.loc167_20 => constants.%facet_type.4ac
// CHECK:STDOUT:   %.Self.loc167_38.1 => constants.%.Self.b05
// CHECK:STDOUT:   %require_complete.loc167 => constants.%complete_type.602
// CHECK:STDOUT:   %.Self.as_type.loc167_44.1 => constants.%.Self.as_type.c10
// CHECK:STDOUT:   %ModWith.assoc_type => constants.%ModWith.assoc_type.95c
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.954
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.loc167 => constants.%ModWith.lookup_impl_witness.ae1
// CHECK:STDOUT:   %impl.elem0.loc167_44.1 => constants.%impl.elem0.9ea
// CHECK:STDOUT:   %Result.as_type.loc167_54.1 => constants.%Int16
// CHECK:STDOUT:   %facet_type.loc167_38 => constants.%facet_type.b54
// CHECK:STDOUT:   %T.loc167_4.1 => constants.%facet_value.c3b
// CHECK:STDOUT:   %pattern_type.loc167 => constants.%pattern_type.87b
// CHECK:STDOUT:   %pattern_type.loc168_16 => constants.%pattern_type.802
// CHECK:STDOUT:   %T.as_type.loc168_29.1 => constants.%Int16
// CHECK:STDOUT:   %pattern_type.loc168_27 => constants.%pattern_type.802
// CHECK:STDOUT:   %pattern_type.loc168_33 => constants.%pattern_type.e1b
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc168_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc168_27 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc168_33 => constants.%complete_type.2da
// CHECK:STDOUT:   %require_complete.loc170 => constants.%complete_type.b14
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.503
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.loc170 => constants.%custom_witness.887
// CHECK:STDOUT:   %ModWith.facet => constants.%ModWith.facet.8ae
// CHECK:STDOUT:   %ModWith.WithSelf.Op.type => constants.%ModWith.WithSelf.Op.type.5f0
// CHECK:STDOUT:   %.loc170_5.4 => constants.%.851
// CHECK:STDOUT:   %impl.elem1.loc170_5.2 => constants.%Int16.Op.cc70d5.7
// CHECK:STDOUT:   %specific_impl_fn.loc170_5.3 => constants.%Int16.Op.cc70d5.7
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.e80
// CHECK:STDOUT:   %.loc170_5.5 => constants.%.154
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.a98c63.1
// CHECK:STDOUT:   %impl.elem0.loc170_5.2 => constants.%Int16.Op.cc70d5.1
// CHECK:STDOUT:   %specific_impl_fn.loc170_5.4 => constants.%Int16.Op.cc70d5.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AddWith.loc124(constants.%Destroy.facet.d87, constants.%Destroy.facet.d87, constants.%facet_value.464) {
// CHECK:STDOUT:   %Result.loc121_9.1 => constants.%Destroy.facet.d87
// CHECK:STDOUT:   %U.loc122_4.1 => constants.%Destroy.facet.d87
// CHECK:STDOUT:   %U.as_type.loc123_36.1 => constants.%Int32
// CHECK:STDOUT:   %AddWith.type.loc123_36.1 => constants.%AddWith.type.78c
// CHECK:STDOUT:   %facet_type.loc123_20 => constants.%facet_type.9c9
// CHECK:STDOUT:   %.Self.loc123_38.1 => constants.%.Self.495
// CHECK:STDOUT:   %require_complete.loc123 => constants.%complete_type.f26
// CHECK:STDOUT:   %.Self.as_type.loc123_44.1 => constants.%.Self.as_type.154
// CHECK:STDOUT:   %AddWith.assoc_type => constants.%AddWith.assoc_type.882
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.5b9
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.loc123 => constants.%AddWith.lookup_impl_witness.4d9
// CHECK:STDOUT:   %impl.elem0.loc123_44.1 => constants.%impl.elem0.53a
// CHECK:STDOUT:   %Result.as_type.loc123_54.1 => constants.%Int32
// CHECK:STDOUT:   %facet_type.loc123_38 => constants.%facet_type.8fd
// CHECK:STDOUT:   %T.loc123_4.1 => constants.%facet_value.464
// CHECK:STDOUT:   %pattern_type.loc123 => constants.%pattern_type.15c
// CHECK:STDOUT:   %pattern_type.loc124_16 => constants.%pattern_type.e9d
// CHECK:STDOUT:   %T.as_type.loc124_29.1 => constants.%Int16
// CHECK:STDOUT:   %pattern_type.loc124_27 => constants.%pattern_type.802
// CHECK:STDOUT:   %pattern_type.loc124_33 => constants.%pattern_type.e9d
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc124_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc124_27 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc124_33 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc126 => constants.%complete_type.c57
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.892
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.loc126 => constants.%custom_witness.e46c75.1
// CHECK:STDOUT:   %AddWith.facet => constants.%AddWith.facet.b3b
// CHECK:STDOUT:   %AddWith.WithSelf.Op.type => constants.%AddWith.WithSelf.Op.type.675
// CHECK:STDOUT:   %.loc126_5.4 => constants.%.615
// CHECK:STDOUT:   %impl.elem1.loc126_5.2 => constants.%Int16.Op.cc70d5.8
// CHECK:STDOUT:   %specific_impl_fn.loc126_5.3 => constants.%Int16.Op.cc70d5.8
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.a62
// CHECK:STDOUT:   %.loc126_5.5 => constants.%.e4f
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.a98c63.2
// CHECK:STDOUT:   %impl.elem0.loc126_5.2 => constants.%Int32.Op.cc70d5.1
// CHECK:STDOUT:   %specific_impl_fn.loc126_5.4 => constants.%Int32.Op.cc70d5.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @SubWith.loc135(constants.%Destroy.facet.d87, constants.%Destroy.facet.d87, constants.%facet_value.197) {
// CHECK:STDOUT:   %Result.loc132_9.1 => constants.%Destroy.facet.d87
// CHECK:STDOUT:   %U.loc133_4.1 => constants.%Destroy.facet.d87
// CHECK:STDOUT:   %U.as_type.loc134_36.1 => constants.%Int32
// CHECK:STDOUT:   %SubWith.type.loc134_36.1 => constants.%SubWith.type.629
// CHECK:STDOUT:   %facet_type.loc134_20 => constants.%facet_type.6c8
// CHECK:STDOUT:   %.Self.loc134_38.1 => constants.%.Self.cbb
// CHECK:STDOUT:   %require_complete.loc134 => constants.%complete_type.dae
// CHECK:STDOUT:   %.Self.as_type.loc134_44.1 => constants.%.Self.as_type.11e
// CHECK:STDOUT:   %SubWith.assoc_type => constants.%SubWith.assoc_type.a03
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.199
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.loc134 => constants.%SubWith.lookup_impl_witness.733
// CHECK:STDOUT:   %impl.elem0.loc134_44.1 => constants.%impl.elem0.e47
// CHECK:STDOUT:   %Result.as_type.loc134_54.1 => constants.%Int32
// CHECK:STDOUT:   %facet_type.loc134_38 => constants.%facet_type.02a
// CHECK:STDOUT:   %T.loc134_4.1 => constants.%facet_value.197
// CHECK:STDOUT:   %pattern_type.loc134 => constants.%pattern_type.553
// CHECK:STDOUT:   %pattern_type.loc135_16 => constants.%pattern_type.e9d
// CHECK:STDOUT:   %T.as_type.loc135_29.1 => constants.%Int16
// CHECK:STDOUT:   %pattern_type.loc135_27 => constants.%pattern_type.802
// CHECK:STDOUT:   %pattern_type.loc135_33 => constants.%pattern_type.e9d
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc135_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc135_27 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc135_33 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc137 => constants.%complete_type.48a
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.bff
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.loc137 => constants.%custom_witness.7ffeb1.1
// CHECK:STDOUT:   %SubWith.facet => constants.%SubWith.facet.da1
// CHECK:STDOUT:   %SubWith.WithSelf.Op.type => constants.%SubWith.WithSelf.Op.type.a0e
// CHECK:STDOUT:   %.loc137_5.4 => constants.%.1a2
// CHECK:STDOUT:   %impl.elem1.loc137_5.2 => constants.%Int16.Op.cc70d5.9
// CHECK:STDOUT:   %specific_impl_fn.loc137_5.3 => constants.%Int16.Op.cc70d5.9
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.a62
// CHECK:STDOUT:   %.loc137_5.5 => constants.%.e4f
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.a98c63.2
// CHECK:STDOUT:   %impl.elem0.loc137_5.2 => constants.%Int32.Op.cc70d5.1
// CHECK:STDOUT:   %specific_impl_fn.loc137_5.4 => constants.%Int32.Op.cc70d5.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @MulWith.loc146(constants.%Destroy.facet.d87, constants.%Destroy.facet.d87, constants.%facet_value.5b3) {
// CHECK:STDOUT:   %Result.loc143_9.1 => constants.%Destroy.facet.d87
// CHECK:STDOUT:   %U.loc144_4.1 => constants.%Destroy.facet.d87
// CHECK:STDOUT:   %U.as_type.loc145_36.1 => constants.%Int32
// CHECK:STDOUT:   %MulWith.type.loc145_36.1 => constants.%MulWith.type.3ee
// CHECK:STDOUT:   %facet_type.loc145_20 => constants.%facet_type.299
// CHECK:STDOUT:   %.Self.loc145_38.1 => constants.%.Self.1c7
// CHECK:STDOUT:   %require_complete.loc145 => constants.%complete_type.0cf
// CHECK:STDOUT:   %.Self.as_type.loc145_44.1 => constants.%.Self.as_type.ba5
// CHECK:STDOUT:   %MulWith.assoc_type => constants.%MulWith.assoc_type.564
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.6fa
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.loc145 => constants.%MulWith.lookup_impl_witness.5ca
// CHECK:STDOUT:   %impl.elem0.loc145_44.1 => constants.%impl.elem0.cab
// CHECK:STDOUT:   %Result.as_type.loc145_54.1 => constants.%Int32
// CHECK:STDOUT:   %facet_type.loc145_38 => constants.%facet_type.a5c
// CHECK:STDOUT:   %T.loc145_4.1 => constants.%facet_value.5b3
// CHECK:STDOUT:   %pattern_type.loc145 => constants.%pattern_type.11c
// CHECK:STDOUT:   %pattern_type.loc146_16 => constants.%pattern_type.e9d
// CHECK:STDOUT:   %T.as_type.loc146_29.1 => constants.%Int16
// CHECK:STDOUT:   %pattern_type.loc146_27 => constants.%pattern_type.802
// CHECK:STDOUT:   %pattern_type.loc146_33 => constants.%pattern_type.e9d
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc146_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc146_27 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc146_33 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc148 => constants.%complete_type.6e9
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.00d
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.loc148 => constants.%custom_witness.68a6cf.1
// CHECK:STDOUT:   %MulWith.facet => constants.%MulWith.facet.c1a
// CHECK:STDOUT:   %MulWith.WithSelf.Op.type => constants.%MulWith.WithSelf.Op.type.b9a
// CHECK:STDOUT:   %.loc148_5.4 => constants.%.f10
// CHECK:STDOUT:   %impl.elem1.loc148_5.2 => constants.%Int16.Op.cc70d5.10
// CHECK:STDOUT:   %specific_impl_fn.loc148_5.3 => constants.%Int16.Op.cc70d5.10
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.a62
// CHECK:STDOUT:   %.loc148_5.5 => constants.%.e4f
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.a98c63.2
// CHECK:STDOUT:   %impl.elem0.loc148_5.2 => constants.%Int32.Op.cc70d5.1
// CHECK:STDOUT:   %specific_impl_fn.loc148_5.4 => constants.%Int32.Op.cc70d5.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @DivWith.loc157(constants.%Destroy.facet.d87, constants.%Destroy.facet.d87, constants.%facet_value.6fd) {
// CHECK:STDOUT:   %Result.loc154_9.1 => constants.%Destroy.facet.d87
// CHECK:STDOUT:   %U.loc155_4.1 => constants.%Destroy.facet.d87
// CHECK:STDOUT:   %U.as_type.loc156_36.1 => constants.%Int32
// CHECK:STDOUT:   %DivWith.type.loc156_36.1 => constants.%DivWith.type.09b
// CHECK:STDOUT:   %facet_type.loc156_20 => constants.%facet_type.c6e
// CHECK:STDOUT:   %.Self.loc156_38.1 => constants.%.Self.988
// CHECK:STDOUT:   %require_complete.loc156 => constants.%complete_type.36c
// CHECK:STDOUT:   %.Self.as_type.loc156_44.1 => constants.%.Self.as_type.f38
// CHECK:STDOUT:   %DivWith.assoc_type => constants.%DivWith.assoc_type.a72
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.d0a
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.loc156 => constants.%DivWith.lookup_impl_witness.c00
// CHECK:STDOUT:   %impl.elem0.loc156_44.1 => constants.%impl.elem0.7a0
// CHECK:STDOUT:   %Result.as_type.loc156_54.1 => constants.%Int32
// CHECK:STDOUT:   %facet_type.loc156_38 => constants.%facet_type.b40
// CHECK:STDOUT:   %T.loc156_4.1 => constants.%facet_value.6fd
// CHECK:STDOUT:   %pattern_type.loc156 => constants.%pattern_type.6b3
// CHECK:STDOUT:   %pattern_type.loc157_16 => constants.%pattern_type.e9d
// CHECK:STDOUT:   %T.as_type.loc157_29.1 => constants.%Int16
// CHECK:STDOUT:   %pattern_type.loc157_27 => constants.%pattern_type.802
// CHECK:STDOUT:   %pattern_type.loc157_33 => constants.%pattern_type.e9d
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc157_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc157_27 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc157_33 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc159 => constants.%complete_type.627
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.489
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.loc159 => constants.%custom_witness.2ccbdb.1
// CHECK:STDOUT:   %DivWith.facet => constants.%DivWith.facet.b53
// CHECK:STDOUT:   %DivWith.WithSelf.Op.type => constants.%DivWith.WithSelf.Op.type.214
// CHECK:STDOUT:   %.loc159_5.4 => constants.%.eff
// CHECK:STDOUT:   %impl.elem1.loc159_5.2 => constants.%Int16.Op.cc70d5.11
// CHECK:STDOUT:   %specific_impl_fn.loc159_5.3 => constants.%Int16.Op.cc70d5.11
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.a62
// CHECK:STDOUT:   %.loc159_5.5 => constants.%.e4f
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.a98c63.2
// CHECK:STDOUT:   %impl.elem0.loc159_5.2 => constants.%Int32.Op.cc70d5.1
// CHECK:STDOUT:   %specific_impl_fn.loc159_5.4 => constants.%Int32.Op.cc70d5.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ModWith.loc168(constants.%Destroy.facet.d87, constants.%Destroy.facet.d87, constants.%facet_value.592) {
// CHECK:STDOUT:   %Result.loc165_9.1 => constants.%Destroy.facet.d87
// CHECK:STDOUT:   %U.loc166_4.1 => constants.%Destroy.facet.d87
// CHECK:STDOUT:   %U.as_type.loc167_36.1 => constants.%Int32
// CHECK:STDOUT:   %ModWith.type.loc167_36.1 => constants.%ModWith.type.6f6
// CHECK:STDOUT:   %facet_type.loc167_20 => constants.%facet_type.22f
// CHECK:STDOUT:   %.Self.loc167_38.1 => constants.%.Self.572
// CHECK:STDOUT:   %require_complete.loc167 => constants.%complete_type.e22
// CHECK:STDOUT:   %.Self.as_type.loc167_44.1 => constants.%.Self.as_type.aac
// CHECK:STDOUT:   %ModWith.assoc_type => constants.%ModWith.assoc_type.d09
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.fb6
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.loc167 => constants.%ModWith.lookup_impl_witness.18e
// CHECK:STDOUT:   %impl.elem0.loc167_44.1 => constants.%impl.elem0.288
// CHECK:STDOUT:   %Result.as_type.loc167_54.1 => constants.%Int32
// CHECK:STDOUT:   %facet_type.loc167_38 => constants.%facet_type.88b
// CHECK:STDOUT:   %T.loc167_4.1 => constants.%facet_value.592
// CHECK:STDOUT:   %pattern_type.loc167 => constants.%pattern_type.dda
// CHECK:STDOUT:   %pattern_type.loc168_16 => constants.%pattern_type.e9d
// CHECK:STDOUT:   %T.as_type.loc168_29.1 => constants.%Int16
// CHECK:STDOUT:   %pattern_type.loc168_27 => constants.%pattern_type.802
// CHECK:STDOUT:   %pattern_type.loc168_33 => constants.%pattern_type.e9d
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc168_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc168_27 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc168_33 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc170 => constants.%complete_type.1cf
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.56e
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.loc170 => constants.%custom_witness.596ded.1
// CHECK:STDOUT:   %ModWith.facet => constants.%ModWith.facet.142
// CHECK:STDOUT:   %ModWith.WithSelf.Op.type => constants.%ModWith.WithSelf.Op.type.513
// CHECK:STDOUT:   %.loc170_5.4 => constants.%.699
// CHECK:STDOUT:   %impl.elem1.loc170_5.2 => constants.%Int16.Op.cc70d5.12
// CHECK:STDOUT:   %specific_impl_fn.loc170_5.3 => constants.%Int16.Op.cc70d5.12
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.a62
// CHECK:STDOUT:   %.loc170_5.5 => constants.%.e4f
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.a98c63.2
// CHECK:STDOUT:   %impl.elem0.loc170_5.2 => constants.%Int32.Op.cc70d5.1
// CHECK:STDOUT:   %specific_impl_fn.loc170_5.4 => constants.%Int32.Op.cc70d5.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AddWith.loc124(constants.%Destroy.facet.d87, constants.%Destroy.facet.1b8, constants.%facet_value.f2b) {
// CHECK:STDOUT:   %Result.loc121_9.1 => constants.%Destroy.facet.d87
// CHECK:STDOUT:   %U.loc122_4.1 => constants.%Destroy.facet.1b8
// CHECK:STDOUT:   %U.as_type.loc123_36.1 => constants.%Int16
// CHECK:STDOUT:   %AddWith.type.loc123_36.1 => constants.%AddWith.type.6f4
// CHECK:STDOUT:   %facet_type.loc123_20 => constants.%facet_type.f17
// CHECK:STDOUT:   %.Self.loc123_38.1 => constants.%.Self.863
// CHECK:STDOUT:   %require_complete.loc123 => constants.%complete_type.211
// CHECK:STDOUT:   %.Self.as_type.loc123_44.1 => constants.%.Self.as_type.db9
// CHECK:STDOUT:   %AddWith.assoc_type => constants.%AddWith.assoc_type.e37
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.4ec
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.loc123 => constants.%AddWith.lookup_impl_witness.896
// CHECK:STDOUT:   %impl.elem0.loc123_44.1 => constants.%impl.elem0.2d1
// CHECK:STDOUT:   %Result.as_type.loc123_54.1 => constants.%Int32
// CHECK:STDOUT:   %facet_type.loc123_38 => constants.%facet_type.558
// CHECK:STDOUT:   %T.loc123_4.1 => constants.%facet_value.f2b
// CHECK:STDOUT:   %pattern_type.loc123 => constants.%pattern_type.715
// CHECK:STDOUT:   %pattern_type.loc124_16 => constants.%pattern_type.e9d
// CHECK:STDOUT:   %T.as_type.loc124_29.1 => constants.%Int32
// CHECK:STDOUT:   %pattern_type.loc124_27 => constants.%pattern_type.e9d
// CHECK:STDOUT:   %pattern_type.loc124_33 => constants.%pattern_type.802
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc124_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc124_27 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc124_33 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc126 => constants.%complete_type.386
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.c66
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.loc126 => constants.%custom_witness.e32
// CHECK:STDOUT:   %AddWith.facet => constants.%AddWith.facet.aef
// CHECK:STDOUT:   %AddWith.WithSelf.Op.type => constants.%AddWith.WithSelf.Op.type.0aa
// CHECK:STDOUT:   %.loc126_5.4 => constants.%.211
// CHECK:STDOUT:   %impl.elem1.loc126_5.2 => constants.%Int32.Op.cc70d5.2
// CHECK:STDOUT:   %specific_impl_fn.loc126_5.3 => constants.%Int32.Op.cc70d5.2
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.a62
// CHECK:STDOUT:   %.loc126_5.5 => constants.%.e4f
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.a98c63.2
// CHECK:STDOUT:   %impl.elem0.loc126_5.2 => constants.%Int32.Op.cc70d5.1
// CHECK:STDOUT:   %specific_impl_fn.loc126_5.4 => constants.%Int32.Op.cc70d5.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @SubWith.loc135(constants.%Destroy.facet.d87, constants.%Destroy.facet.1b8, constants.%facet_value.dc0) {
// CHECK:STDOUT:   %Result.loc132_9.1 => constants.%Destroy.facet.d87
// CHECK:STDOUT:   %U.loc133_4.1 => constants.%Destroy.facet.1b8
// CHECK:STDOUT:   %U.as_type.loc134_36.1 => constants.%Int16
// CHECK:STDOUT:   %SubWith.type.loc134_36.1 => constants.%SubWith.type.8bd
// CHECK:STDOUT:   %facet_type.loc134_20 => constants.%facet_type.c80
// CHECK:STDOUT:   %.Self.loc134_38.1 => constants.%.Self.700
// CHECK:STDOUT:   %require_complete.loc134 => constants.%complete_type.a74
// CHECK:STDOUT:   %.Self.as_type.loc134_44.1 => constants.%.Self.as_type.c4d
// CHECK:STDOUT:   %SubWith.assoc_type => constants.%SubWith.assoc_type.eb8
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.8a1
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.loc134 => constants.%SubWith.lookup_impl_witness.e8f
// CHECK:STDOUT:   %impl.elem0.loc134_44.1 => constants.%impl.elem0.0b0
// CHECK:STDOUT:   %Result.as_type.loc134_54.1 => constants.%Int32
// CHECK:STDOUT:   %facet_type.loc134_38 => constants.%facet_type.064
// CHECK:STDOUT:   %T.loc134_4.1 => constants.%facet_value.dc0
// CHECK:STDOUT:   %pattern_type.loc134 => constants.%pattern_type.b06
// CHECK:STDOUT:   %pattern_type.loc135_16 => constants.%pattern_type.e9d
// CHECK:STDOUT:   %T.as_type.loc135_29.1 => constants.%Int32
// CHECK:STDOUT:   %pattern_type.loc135_27 => constants.%pattern_type.e9d
// CHECK:STDOUT:   %pattern_type.loc135_33 => constants.%pattern_type.802
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc135_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc135_27 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc135_33 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc137 => constants.%complete_type.41c
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.937f
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.loc137 => constants.%custom_witness.395
// CHECK:STDOUT:   %SubWith.facet => constants.%SubWith.facet.9f2
// CHECK:STDOUT:   %SubWith.WithSelf.Op.type => constants.%SubWith.WithSelf.Op.type.bba
// CHECK:STDOUT:   %.loc137_5.4 => constants.%.2b2
// CHECK:STDOUT:   %impl.elem1.loc137_5.2 => constants.%Int32.Op.cc70d5.3
// CHECK:STDOUT:   %specific_impl_fn.loc137_5.3 => constants.%Int32.Op.cc70d5.3
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.a62
// CHECK:STDOUT:   %.loc137_5.5 => constants.%.e4f
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.a98c63.2
// CHECK:STDOUT:   %impl.elem0.loc137_5.2 => constants.%Int32.Op.cc70d5.1
// CHECK:STDOUT:   %specific_impl_fn.loc137_5.4 => constants.%Int32.Op.cc70d5.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @MulWith.loc146(constants.%Destroy.facet.d87, constants.%Destroy.facet.1b8, constants.%facet_value.ff0) {
// CHECK:STDOUT:   %Result.loc143_9.1 => constants.%Destroy.facet.d87
// CHECK:STDOUT:   %U.loc144_4.1 => constants.%Destroy.facet.1b8
// CHECK:STDOUT:   %U.as_type.loc145_36.1 => constants.%Int16
// CHECK:STDOUT:   %MulWith.type.loc145_36.1 => constants.%MulWith.type.c25
// CHECK:STDOUT:   %facet_type.loc145_20 => constants.%facet_type.d27
// CHECK:STDOUT:   %.Self.loc145_38.1 => constants.%.Self.e39
// CHECK:STDOUT:   %require_complete.loc145 => constants.%complete_type.aa7
// CHECK:STDOUT:   %.Self.as_type.loc145_44.1 => constants.%.Self.as_type.87e
// CHECK:STDOUT:   %MulWith.assoc_type => constants.%MulWith.assoc_type.543
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.81e
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.loc145 => constants.%MulWith.lookup_impl_witness.5e1
// CHECK:STDOUT:   %impl.elem0.loc145_44.1 => constants.%impl.elem0.e54
// CHECK:STDOUT:   %Result.as_type.loc145_54.1 => constants.%Int32
// CHECK:STDOUT:   %facet_type.loc145_38 => constants.%facet_type.d97
// CHECK:STDOUT:   %T.loc145_4.1 => constants.%facet_value.ff0
// CHECK:STDOUT:   %pattern_type.loc145 => constants.%pattern_type.3ec
// CHECK:STDOUT:   %pattern_type.loc146_16 => constants.%pattern_type.e9d
// CHECK:STDOUT:   %T.as_type.loc146_29.1 => constants.%Int32
// CHECK:STDOUT:   %pattern_type.loc146_27 => constants.%pattern_type.e9d
// CHECK:STDOUT:   %pattern_type.loc146_33 => constants.%pattern_type.802
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc146_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc146_27 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc146_33 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc148 => constants.%complete_type.b16
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.911
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.loc148 => constants.%custom_witness.678
// CHECK:STDOUT:   %MulWith.facet => constants.%MulWith.facet.9cb
// CHECK:STDOUT:   %MulWith.WithSelf.Op.type => constants.%MulWith.WithSelf.Op.type.15f
// CHECK:STDOUT:   %.loc148_5.4 => constants.%.dd6
// CHECK:STDOUT:   %impl.elem1.loc148_5.2 => constants.%Int32.Op.cc70d5.4
// CHECK:STDOUT:   %specific_impl_fn.loc148_5.3 => constants.%Int32.Op.cc70d5.4
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.a62
// CHECK:STDOUT:   %.loc148_5.5 => constants.%.e4f
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.a98c63.2
// CHECK:STDOUT:   %impl.elem0.loc148_5.2 => constants.%Int32.Op.cc70d5.1
// CHECK:STDOUT:   %specific_impl_fn.loc148_5.4 => constants.%Int32.Op.cc70d5.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @DivWith.loc157(constants.%Destroy.facet.d87, constants.%Destroy.facet.1b8, constants.%facet_value.42a) {
// CHECK:STDOUT:   %Result.loc154_9.1 => constants.%Destroy.facet.d87
// CHECK:STDOUT:   %U.loc155_4.1 => constants.%Destroy.facet.1b8
// CHECK:STDOUT:   %U.as_type.loc156_36.1 => constants.%Int16
// CHECK:STDOUT:   %DivWith.type.loc156_36.1 => constants.%DivWith.type.053
// CHECK:STDOUT:   %facet_type.loc156_20 => constants.%facet_type.8eb
// CHECK:STDOUT:   %.Self.loc156_38.1 => constants.%.Self.467
// CHECK:STDOUT:   %require_complete.loc156 => constants.%complete_type.d563
// CHECK:STDOUT:   %.Self.as_type.loc156_44.1 => constants.%.Self.as_type.3cd
// CHECK:STDOUT:   %DivWith.assoc_type => constants.%DivWith.assoc_type.73e
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.e96
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.loc156 => constants.%DivWith.lookup_impl_witness.9bc
// CHECK:STDOUT:   %impl.elem0.loc156_44.1 => constants.%impl.elem0.5c6
// CHECK:STDOUT:   %Result.as_type.loc156_54.1 => constants.%Int32
// CHECK:STDOUT:   %facet_type.loc156_38 => constants.%facet_type.aa6
// CHECK:STDOUT:   %T.loc156_4.1 => constants.%facet_value.42a
// CHECK:STDOUT:   %pattern_type.loc156 => constants.%pattern_type.814
// CHECK:STDOUT:   %pattern_type.loc157_16 => constants.%pattern_type.e9d
// CHECK:STDOUT:   %T.as_type.loc157_29.1 => constants.%Int32
// CHECK:STDOUT:   %pattern_type.loc157_27 => constants.%pattern_type.e9d
// CHECK:STDOUT:   %pattern_type.loc157_33 => constants.%pattern_type.802
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc157_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc157_27 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc157_33 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc159 => constants.%complete_type.7c5
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.10b
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.loc159 => constants.%custom_witness.8ad
// CHECK:STDOUT:   %DivWith.facet => constants.%DivWith.facet.53f
// CHECK:STDOUT:   %DivWith.WithSelf.Op.type => constants.%DivWith.WithSelf.Op.type.ba3
// CHECK:STDOUT:   %.loc159_5.4 => constants.%.540
// CHECK:STDOUT:   %impl.elem1.loc159_5.2 => constants.%Int32.Op.cc70d5.5
// CHECK:STDOUT:   %specific_impl_fn.loc159_5.3 => constants.%Int32.Op.cc70d5.5
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.a62
// CHECK:STDOUT:   %.loc159_5.5 => constants.%.e4f
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.a98c63.2
// CHECK:STDOUT:   %impl.elem0.loc159_5.2 => constants.%Int32.Op.cc70d5.1
// CHECK:STDOUT:   %specific_impl_fn.loc159_5.4 => constants.%Int32.Op.cc70d5.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ModWith.loc168(constants.%Destroy.facet.d87, constants.%Destroy.facet.1b8, constants.%facet_value.7ad) {
// CHECK:STDOUT:   %Result.loc165_9.1 => constants.%Destroy.facet.d87
// CHECK:STDOUT:   %U.loc166_4.1 => constants.%Destroy.facet.1b8
// CHECK:STDOUT:   %U.as_type.loc167_36.1 => constants.%Int16
// CHECK:STDOUT:   %ModWith.type.loc167_36.1 => constants.%ModWith.type.a83
// CHECK:STDOUT:   %facet_type.loc167_20 => constants.%facet_type.dc7
// CHECK:STDOUT:   %.Self.loc167_38.1 => constants.%.Self.703
// CHECK:STDOUT:   %require_complete.loc167 => constants.%complete_type.65d
// CHECK:STDOUT:   %.Self.as_type.loc167_44.1 => constants.%.Self.as_type.c72
// CHECK:STDOUT:   %ModWith.assoc_type => constants.%ModWith.assoc_type.55b
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.8a3
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.loc167 => constants.%ModWith.lookup_impl_witness.c7a
// CHECK:STDOUT:   %impl.elem0.loc167_44.1 => constants.%impl.elem0.094
// CHECK:STDOUT:   %Result.as_type.loc167_54.1 => constants.%Int32
// CHECK:STDOUT:   %facet_type.loc167_38 => constants.%facet_type.b16
// CHECK:STDOUT:   %T.loc167_4.1 => constants.%facet_value.7ad
// CHECK:STDOUT:   %pattern_type.loc167 => constants.%pattern_type.7fa
// CHECK:STDOUT:   %pattern_type.loc168_16 => constants.%pattern_type.e9d
// CHECK:STDOUT:   %T.as_type.loc168_29.1 => constants.%Int32
// CHECK:STDOUT:   %pattern_type.loc168_27 => constants.%pattern_type.e9d
// CHECK:STDOUT:   %pattern_type.loc168_33 => constants.%pattern_type.802
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc168_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc168_27 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc168_33 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc170 => constants.%complete_type.2f8
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.7d4
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.loc170 => constants.%custom_witness.f63
// CHECK:STDOUT:   %ModWith.facet => constants.%ModWith.facet.0ea
// CHECK:STDOUT:   %ModWith.WithSelf.Op.type => constants.%ModWith.WithSelf.Op.type.aeb
// CHECK:STDOUT:   %.loc170_5.4 => constants.%.b1a
// CHECK:STDOUT:   %impl.elem1.loc170_5.2 => constants.%Int32.Op.cc70d5.6
// CHECK:STDOUT:   %specific_impl_fn.loc170_5.3 => constants.%Int32.Op.cc70d5.6
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.a62
// CHECK:STDOUT:   %.loc170_5.5 => constants.%.e4f
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.a98c63.2
// CHECK:STDOUT:   %impl.elem0.loc170_5.2 => constants.%Int32.Op.cc70d5.1
// CHECK:STDOUT:   %specific_impl_fn.loc170_5.4 => constants.%Int32.Op.cc70d5.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AddWith.loc124(constants.%Destroy.facet.d87, constants.%Destroy.facet.d87, constants.%facet_value.f56) {
// CHECK:STDOUT:   %Result.loc121_9.1 => constants.%Destroy.facet.d87
// CHECK:STDOUT:   %U.loc122_4.1 => constants.%Destroy.facet.d87
// CHECK:STDOUT:   %U.as_type.loc123_36.1 => constants.%Int32
// CHECK:STDOUT:   %AddWith.type.loc123_36.1 => constants.%AddWith.type.78c
// CHECK:STDOUT:   %facet_type.loc123_20 => constants.%facet_type.9c9
// CHECK:STDOUT:   %.Self.loc123_38.1 => constants.%.Self.495
// CHECK:STDOUT:   %require_complete.loc123 => constants.%complete_type.f26
// CHECK:STDOUT:   %.Self.as_type.loc123_44.1 => constants.%.Self.as_type.154
// CHECK:STDOUT:   %AddWith.assoc_type => constants.%AddWith.assoc_type.882
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.5b9
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.loc123 => constants.%AddWith.lookup_impl_witness.4d9
// CHECK:STDOUT:   %impl.elem0.loc123_44.1 => constants.%impl.elem0.53a
// CHECK:STDOUT:   %Result.as_type.loc123_54.1 => constants.%Int32
// CHECK:STDOUT:   %facet_type.loc123_38 => constants.%facet_type.8fd
// CHECK:STDOUT:   %T.loc123_4.1 => constants.%facet_value.f56
// CHECK:STDOUT:   %pattern_type.loc123 => constants.%pattern_type.15c
// CHECK:STDOUT:   %pattern_type.loc124_16 => constants.%pattern_type.e9d
// CHECK:STDOUT:   %T.as_type.loc124_29.1 => constants.%Int32
// CHECK:STDOUT:   %pattern_type.loc124_27 => constants.%pattern_type.e9d
// CHECK:STDOUT:   %pattern_type.loc124_33 => constants.%pattern_type.e9d
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc124_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc124_27 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc124_33 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc126 => constants.%complete_type.c57
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.892
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.loc126 => constants.%custom_witness.e46c75.2
// CHECK:STDOUT:   %AddWith.facet => constants.%AddWith.facet.616
// CHECK:STDOUT:   %AddWith.WithSelf.Op.type => constants.%AddWith.WithSelf.Op.type.9e3
// CHECK:STDOUT:   %.loc126_5.4 => constants.%.d9d
// CHECK:STDOUT:   %impl.elem1.loc126_5.2 => constants.%Int32.Op.cc70d5.7
// CHECK:STDOUT:   %specific_impl_fn.loc126_5.3 => constants.%Int32.Op.cc70d5.7
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.a62
// CHECK:STDOUT:   %.loc126_5.5 => constants.%.e4f
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.a98c63.2
// CHECK:STDOUT:   %impl.elem0.loc126_5.2 => constants.%Int32.Op.cc70d5.1
// CHECK:STDOUT:   %specific_impl_fn.loc126_5.4 => constants.%Int32.Op.cc70d5.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @SubWith.loc135(constants.%Destroy.facet.d87, constants.%Destroy.facet.d87, constants.%facet_value.8a6) {
// CHECK:STDOUT:   %Result.loc132_9.1 => constants.%Destroy.facet.d87
// CHECK:STDOUT:   %U.loc133_4.1 => constants.%Destroy.facet.d87
// CHECK:STDOUT:   %U.as_type.loc134_36.1 => constants.%Int32
// CHECK:STDOUT:   %SubWith.type.loc134_36.1 => constants.%SubWith.type.629
// CHECK:STDOUT:   %facet_type.loc134_20 => constants.%facet_type.6c8
// CHECK:STDOUT:   %.Self.loc134_38.1 => constants.%.Self.cbb
// CHECK:STDOUT:   %require_complete.loc134 => constants.%complete_type.dae
// CHECK:STDOUT:   %.Self.as_type.loc134_44.1 => constants.%.Self.as_type.11e
// CHECK:STDOUT:   %SubWith.assoc_type => constants.%SubWith.assoc_type.a03
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.199
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.loc134 => constants.%SubWith.lookup_impl_witness.733
// CHECK:STDOUT:   %impl.elem0.loc134_44.1 => constants.%impl.elem0.e47
// CHECK:STDOUT:   %Result.as_type.loc134_54.1 => constants.%Int32
// CHECK:STDOUT:   %facet_type.loc134_38 => constants.%facet_type.02a
// CHECK:STDOUT:   %T.loc134_4.1 => constants.%facet_value.8a6
// CHECK:STDOUT:   %pattern_type.loc134 => constants.%pattern_type.553
// CHECK:STDOUT:   %pattern_type.loc135_16 => constants.%pattern_type.e9d
// CHECK:STDOUT:   %T.as_type.loc135_29.1 => constants.%Int32
// CHECK:STDOUT:   %pattern_type.loc135_27 => constants.%pattern_type.e9d
// CHECK:STDOUT:   %pattern_type.loc135_33 => constants.%pattern_type.e9d
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc135_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc135_27 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc135_33 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc137 => constants.%complete_type.48a
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.bff
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.loc137 => constants.%custom_witness.7ffeb1.2
// CHECK:STDOUT:   %SubWith.facet => constants.%SubWith.facet.36d
// CHECK:STDOUT:   %SubWith.WithSelf.Op.type => constants.%SubWith.WithSelf.Op.type.1db
// CHECK:STDOUT:   %.loc137_5.4 => constants.%.613
// CHECK:STDOUT:   %impl.elem1.loc137_5.2 => constants.%Int32.Op.cc70d5.8
// CHECK:STDOUT:   %specific_impl_fn.loc137_5.3 => constants.%Int32.Op.cc70d5.8
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.a62
// CHECK:STDOUT:   %.loc137_5.5 => constants.%.e4f
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.a98c63.2
// CHECK:STDOUT:   %impl.elem0.loc137_5.2 => constants.%Int32.Op.cc70d5.1
// CHECK:STDOUT:   %specific_impl_fn.loc137_5.4 => constants.%Int32.Op.cc70d5.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @MulWith.loc146(constants.%Destroy.facet.d87, constants.%Destroy.facet.d87, constants.%facet_value.e09) {
// CHECK:STDOUT:   %Result.loc143_9.1 => constants.%Destroy.facet.d87
// CHECK:STDOUT:   %U.loc144_4.1 => constants.%Destroy.facet.d87
// CHECK:STDOUT:   %U.as_type.loc145_36.1 => constants.%Int32
// CHECK:STDOUT:   %MulWith.type.loc145_36.1 => constants.%MulWith.type.3ee
// CHECK:STDOUT:   %facet_type.loc145_20 => constants.%facet_type.299
// CHECK:STDOUT:   %.Self.loc145_38.1 => constants.%.Self.1c7
// CHECK:STDOUT:   %require_complete.loc145 => constants.%complete_type.0cf
// CHECK:STDOUT:   %.Self.as_type.loc145_44.1 => constants.%.Self.as_type.ba5
// CHECK:STDOUT:   %MulWith.assoc_type => constants.%MulWith.assoc_type.564
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.6fa
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.loc145 => constants.%MulWith.lookup_impl_witness.5ca
// CHECK:STDOUT:   %impl.elem0.loc145_44.1 => constants.%impl.elem0.cab
// CHECK:STDOUT:   %Result.as_type.loc145_54.1 => constants.%Int32
// CHECK:STDOUT:   %facet_type.loc145_38 => constants.%facet_type.a5c
// CHECK:STDOUT:   %T.loc145_4.1 => constants.%facet_value.e09
// CHECK:STDOUT:   %pattern_type.loc145 => constants.%pattern_type.11c
// CHECK:STDOUT:   %pattern_type.loc146_16 => constants.%pattern_type.e9d
// CHECK:STDOUT:   %T.as_type.loc146_29.1 => constants.%Int32
// CHECK:STDOUT:   %pattern_type.loc146_27 => constants.%pattern_type.e9d
// CHECK:STDOUT:   %pattern_type.loc146_33 => constants.%pattern_type.e9d
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc146_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc146_27 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc146_33 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc148 => constants.%complete_type.6e9
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.00d
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.loc148 => constants.%custom_witness.68a6cf.2
// CHECK:STDOUT:   %MulWith.facet => constants.%MulWith.facet.dd3
// CHECK:STDOUT:   %MulWith.WithSelf.Op.type => constants.%MulWith.WithSelf.Op.type.854
// CHECK:STDOUT:   %.loc148_5.4 => constants.%.ec7
// CHECK:STDOUT:   %impl.elem1.loc148_5.2 => constants.%Int32.Op.cc70d5.9
// CHECK:STDOUT:   %specific_impl_fn.loc148_5.3 => constants.%Int32.Op.cc70d5.9
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.a62
// CHECK:STDOUT:   %.loc148_5.5 => constants.%.e4f
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.a98c63.2
// CHECK:STDOUT:   %impl.elem0.loc148_5.2 => constants.%Int32.Op.cc70d5.1
// CHECK:STDOUT:   %specific_impl_fn.loc148_5.4 => constants.%Int32.Op.cc70d5.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @DivWith.loc157(constants.%Destroy.facet.d87, constants.%Destroy.facet.d87, constants.%facet_value.a3b) {
// CHECK:STDOUT:   %Result.loc154_9.1 => constants.%Destroy.facet.d87
// CHECK:STDOUT:   %U.loc155_4.1 => constants.%Destroy.facet.d87
// CHECK:STDOUT:   %U.as_type.loc156_36.1 => constants.%Int32
// CHECK:STDOUT:   %DivWith.type.loc156_36.1 => constants.%DivWith.type.09b
// CHECK:STDOUT:   %facet_type.loc156_20 => constants.%facet_type.c6e
// CHECK:STDOUT:   %.Self.loc156_38.1 => constants.%.Self.988
// CHECK:STDOUT:   %require_complete.loc156 => constants.%complete_type.36c
// CHECK:STDOUT:   %.Self.as_type.loc156_44.1 => constants.%.Self.as_type.f38
// CHECK:STDOUT:   %DivWith.assoc_type => constants.%DivWith.assoc_type.a72
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.d0a
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.loc156 => constants.%DivWith.lookup_impl_witness.c00
// CHECK:STDOUT:   %impl.elem0.loc156_44.1 => constants.%impl.elem0.7a0
// CHECK:STDOUT:   %Result.as_type.loc156_54.1 => constants.%Int32
// CHECK:STDOUT:   %facet_type.loc156_38 => constants.%facet_type.b40
// CHECK:STDOUT:   %T.loc156_4.1 => constants.%facet_value.a3b
// CHECK:STDOUT:   %pattern_type.loc156 => constants.%pattern_type.6b3
// CHECK:STDOUT:   %pattern_type.loc157_16 => constants.%pattern_type.e9d
// CHECK:STDOUT:   %T.as_type.loc157_29.1 => constants.%Int32
// CHECK:STDOUT:   %pattern_type.loc157_27 => constants.%pattern_type.e9d
// CHECK:STDOUT:   %pattern_type.loc157_33 => constants.%pattern_type.e9d
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc157_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc157_27 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc157_33 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc159 => constants.%complete_type.627
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.489
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.loc159 => constants.%custom_witness.2ccbdb.2
// CHECK:STDOUT:   %DivWith.facet => constants.%DivWith.facet.c86
// CHECK:STDOUT:   %DivWith.WithSelf.Op.type => constants.%DivWith.WithSelf.Op.type.e4e
// CHECK:STDOUT:   %.loc159_5.4 => constants.%.865
// CHECK:STDOUT:   %impl.elem1.loc159_5.2 => constants.%Int32.Op.cc70d5.10
// CHECK:STDOUT:   %specific_impl_fn.loc159_5.3 => constants.%Int32.Op.cc70d5.10
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.a62
// CHECK:STDOUT:   %.loc159_5.5 => constants.%.e4f
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.a98c63.2
// CHECK:STDOUT:   %impl.elem0.loc159_5.2 => constants.%Int32.Op.cc70d5.1
// CHECK:STDOUT:   %specific_impl_fn.loc159_5.4 => constants.%Int32.Op.cc70d5.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ModWith.loc168(constants.%Destroy.facet.d87, constants.%Destroy.facet.d87, constants.%facet_value.40c) {
// CHECK:STDOUT:   %Result.loc165_9.1 => constants.%Destroy.facet.d87
// CHECK:STDOUT:   %U.loc166_4.1 => constants.%Destroy.facet.d87
// CHECK:STDOUT:   %U.as_type.loc167_36.1 => constants.%Int32
// CHECK:STDOUT:   %ModWith.type.loc167_36.1 => constants.%ModWith.type.6f6
// CHECK:STDOUT:   %facet_type.loc167_20 => constants.%facet_type.22f
// CHECK:STDOUT:   %.Self.loc167_38.1 => constants.%.Self.572
// CHECK:STDOUT:   %require_complete.loc167 => constants.%complete_type.e22
// CHECK:STDOUT:   %.Self.as_type.loc167_44.1 => constants.%.Self.as_type.aac
// CHECK:STDOUT:   %ModWith.assoc_type => constants.%ModWith.assoc_type.d09
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.fb6
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.loc167 => constants.%ModWith.lookup_impl_witness.18e
// CHECK:STDOUT:   %impl.elem0.loc167_44.1 => constants.%impl.elem0.288
// CHECK:STDOUT:   %Result.as_type.loc167_54.1 => constants.%Int32
// CHECK:STDOUT:   %facet_type.loc167_38 => constants.%facet_type.88b
// CHECK:STDOUT:   %T.loc167_4.1 => constants.%facet_value.40c
// CHECK:STDOUT:   %pattern_type.loc167 => constants.%pattern_type.dda
// CHECK:STDOUT:   %pattern_type.loc168_16 => constants.%pattern_type.e9d
// CHECK:STDOUT:   %T.as_type.loc168_29.1 => constants.%Int32
// CHECK:STDOUT:   %pattern_type.loc168_27 => constants.%pattern_type.e9d
// CHECK:STDOUT:   %pattern_type.loc168_33 => constants.%pattern_type.e9d
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc168_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc168_27 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc168_33 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc170 => constants.%complete_type.1cf
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.56e
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.loc170 => constants.%custom_witness.596ded.2
// CHECK:STDOUT:   %ModWith.facet => constants.%ModWith.facet.c80
// CHECK:STDOUT:   %ModWith.WithSelf.Op.type => constants.%ModWith.WithSelf.Op.type.e0b
// CHECK:STDOUT:   %.loc170_5.4 => constants.%.d2e
// CHECK:STDOUT:   %impl.elem1.loc170_5.2 => constants.%Int32.Op.cc70d5.11
// CHECK:STDOUT:   %specific_impl_fn.loc170_5.3 => constants.%Int32.Op.cc70d5.11
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.a62
// CHECK:STDOUT:   %.loc170_5.5 => constants.%.e4f
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.a98c63.2
// CHECK:STDOUT:   %impl.elem0.loc170_5.2 => constants.%Int32.Op.cc70d5.1
// CHECK:STDOUT:   %specific_impl_fn.loc170_5.4 => constants.%Int32.Op.cc70d5.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AddWith.loc124(constants.%Destroy.facet.3de, constants.%Destroy.facet.3de, constants.%facet_value.5d2) {
// CHECK:STDOUT:   %Result.loc121_9.1 => constants.%Destroy.facet.3de
// CHECK:STDOUT:   %U.loc122_4.1 => constants.%Destroy.facet.3de
// CHECK:STDOUT:   %U.as_type.loc123_36.1 => constants.%Int64
// CHECK:STDOUT:   %AddWith.type.loc123_36.1 => constants.%AddWith.type.01b
// CHECK:STDOUT:   %facet_type.loc123_20 => constants.%facet_type.fb6
// CHECK:STDOUT:   %.Self.loc123_38.1 => constants.%.Self.d3e
// CHECK:STDOUT:   %require_complete.loc123 => constants.%complete_type.900
// CHECK:STDOUT:   %.Self.as_type.loc123_44.1 => constants.%.Self.as_type.1db
// CHECK:STDOUT:   %AddWith.assoc_type => constants.%AddWith.assoc_type.7c5
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.f19
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.loc123 => constants.%AddWith.lookup_impl_witness.708
// CHECK:STDOUT:   %impl.elem0.loc123_44.1 => constants.%impl.elem0.86a
// CHECK:STDOUT:   %Result.as_type.loc123_54.1 => constants.%Int64
// CHECK:STDOUT:   %facet_type.loc123_38 => constants.%facet_type.376
// CHECK:STDOUT:   %T.loc123_4.1 => constants.%facet_value.5d2
// CHECK:STDOUT:   %pattern_type.loc123 => constants.%pattern_type.e29
// CHECK:STDOUT:   %pattern_type.loc124_16 => constants.%pattern_type.7b1
// CHECK:STDOUT:   %T.as_type.loc124_29.1 => constants.%Int32
// CHECK:STDOUT:   %pattern_type.loc124_27 => constants.%pattern_type.e9d
// CHECK:STDOUT:   %pattern_type.loc124_33 => constants.%pattern_type.7b1
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc124_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc124_27 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc124_33 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc126 => constants.%complete_type.495
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.3e0
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.loc126 => constants.%custom_witness.e1e
// CHECK:STDOUT:   %AddWith.facet => constants.%AddWith.facet.31a
// CHECK:STDOUT:   %AddWith.WithSelf.Op.type => constants.%AddWith.WithSelf.Op.type.c0a
// CHECK:STDOUT:   %.loc126_5.4 => constants.%.2ba
// CHECK:STDOUT:   %impl.elem1.loc126_5.2 => constants.%Op.b9c47d.6
// CHECK:STDOUT:   %specific_impl_fn.loc126_5.3 => constants.%Op.b9c47d.6
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.2bb
// CHECK:STDOUT:   %.loc126_5.5 => constants.%.dda
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.a98c63.3
// CHECK:STDOUT:   %impl.elem0.loc126_5.2 => constants.%Int64.Op.cc70d5.1
// CHECK:STDOUT:   %specific_impl_fn.loc126_5.4 => constants.%Int64.Op.cc70d5.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @SubWith.loc135(constants.%Destroy.facet.3de, constants.%Destroy.facet.3de, constants.%facet_value.916) {
// CHECK:STDOUT:   %Result.loc132_9.1 => constants.%Destroy.facet.3de
// CHECK:STDOUT:   %U.loc133_4.1 => constants.%Destroy.facet.3de
// CHECK:STDOUT:   %U.as_type.loc134_36.1 => constants.%Int64
// CHECK:STDOUT:   %SubWith.type.loc134_36.1 => constants.%SubWith.type.82c
// CHECK:STDOUT:   %facet_type.loc134_20 => constants.%facet_type.038
// CHECK:STDOUT:   %.Self.loc134_38.1 => constants.%.Self.869
// CHECK:STDOUT:   %require_complete.loc134 => constants.%complete_type.e31
// CHECK:STDOUT:   %.Self.as_type.loc134_44.1 => constants.%.Self.as_type.824
// CHECK:STDOUT:   %SubWith.assoc_type => constants.%SubWith.assoc_type.f62
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.018
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.loc134 => constants.%SubWith.lookup_impl_witness.c9f
// CHECK:STDOUT:   %impl.elem0.loc134_44.1 => constants.%impl.elem0.6f7
// CHECK:STDOUT:   %Result.as_type.loc134_54.1 => constants.%Int64
// CHECK:STDOUT:   %facet_type.loc134_38 => constants.%facet_type.4a1
// CHECK:STDOUT:   %T.loc134_4.1 => constants.%facet_value.916
// CHECK:STDOUT:   %pattern_type.loc134 => constants.%pattern_type.58c
// CHECK:STDOUT:   %pattern_type.loc135_16 => constants.%pattern_type.7b1
// CHECK:STDOUT:   %T.as_type.loc135_29.1 => constants.%Int32
// CHECK:STDOUT:   %pattern_type.loc135_27 => constants.%pattern_type.e9d
// CHECK:STDOUT:   %pattern_type.loc135_33 => constants.%pattern_type.7b1
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc135_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc135_27 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc135_33 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc137 => constants.%complete_type.cc7
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.1cf
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.loc137 => constants.%custom_witness.053
// CHECK:STDOUT:   %SubWith.facet => constants.%SubWith.facet.9cc
// CHECK:STDOUT:   %SubWith.WithSelf.Op.type => constants.%SubWith.WithSelf.Op.type.a06
// CHECK:STDOUT:   %.loc137_5.4 => constants.%.086
// CHECK:STDOUT:   %impl.elem1.loc137_5.2 => constants.%Op.b9c47d.7
// CHECK:STDOUT:   %specific_impl_fn.loc137_5.3 => constants.%Op.b9c47d.7
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.2bb
// CHECK:STDOUT:   %.loc137_5.5 => constants.%.dda
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.a98c63.3
// CHECK:STDOUT:   %impl.elem0.loc137_5.2 => constants.%Int64.Op.cc70d5.1
// CHECK:STDOUT:   %specific_impl_fn.loc137_5.4 => constants.%Int64.Op.cc70d5.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @MulWith.loc146(constants.%Destroy.facet.3de, constants.%Destroy.facet.3de, constants.%facet_value.7ba) {
// CHECK:STDOUT:   %Result.loc143_9.1 => constants.%Destroy.facet.3de
// CHECK:STDOUT:   %U.loc144_4.1 => constants.%Destroy.facet.3de
// CHECK:STDOUT:   %U.as_type.loc145_36.1 => constants.%Int64
// CHECK:STDOUT:   %MulWith.type.loc145_36.1 => constants.%MulWith.type.9ed
// CHECK:STDOUT:   %facet_type.loc145_20 => constants.%facet_type.e03
// CHECK:STDOUT:   %.Self.loc145_38.1 => constants.%.Self.d17
// CHECK:STDOUT:   %require_complete.loc145 => constants.%complete_type.25d
// CHECK:STDOUT:   %.Self.as_type.loc145_44.1 => constants.%.Self.as_type.a7c
// CHECK:STDOUT:   %MulWith.assoc_type => constants.%MulWith.assoc_type.8d1
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.c01
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.loc145 => constants.%MulWith.lookup_impl_witness.13a
// CHECK:STDOUT:   %impl.elem0.loc145_44.1 => constants.%impl.elem0.7f7
// CHECK:STDOUT:   %Result.as_type.loc145_54.1 => constants.%Int64
// CHECK:STDOUT:   %facet_type.loc145_38 => constants.%facet_type.88c
// CHECK:STDOUT:   %T.loc145_4.1 => constants.%facet_value.7ba
// CHECK:STDOUT:   %pattern_type.loc145 => constants.%pattern_type.c54
// CHECK:STDOUT:   %pattern_type.loc146_16 => constants.%pattern_type.7b1
// CHECK:STDOUT:   %T.as_type.loc146_29.1 => constants.%Int32
// CHECK:STDOUT:   %pattern_type.loc146_27 => constants.%pattern_type.e9d
// CHECK:STDOUT:   %pattern_type.loc146_33 => constants.%pattern_type.7b1
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc146_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc146_27 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc146_33 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc148 => constants.%complete_type.da6
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.eef
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.loc148 => constants.%custom_witness.f29
// CHECK:STDOUT:   %MulWith.facet => constants.%MulWith.facet.83a
// CHECK:STDOUT:   %MulWith.WithSelf.Op.type => constants.%MulWith.WithSelf.Op.type.bc6
// CHECK:STDOUT:   %.loc148_5.4 => constants.%.ca6
// CHECK:STDOUT:   %impl.elem1.loc148_5.2 => constants.%Op.b9c47d.8
// CHECK:STDOUT:   %specific_impl_fn.loc148_5.3 => constants.%Op.b9c47d.8
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.2bb
// CHECK:STDOUT:   %.loc148_5.5 => constants.%.dda
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.a98c63.3
// CHECK:STDOUT:   %impl.elem0.loc148_5.2 => constants.%Int64.Op.cc70d5.1
// CHECK:STDOUT:   %specific_impl_fn.loc148_5.4 => constants.%Int64.Op.cc70d5.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @DivWith.loc157(constants.%Destroy.facet.3de, constants.%Destroy.facet.3de, constants.%facet_value.969) {
// CHECK:STDOUT:   %Result.loc154_9.1 => constants.%Destroy.facet.3de
// CHECK:STDOUT:   %U.loc155_4.1 => constants.%Destroy.facet.3de
// CHECK:STDOUT:   %U.as_type.loc156_36.1 => constants.%Int64
// CHECK:STDOUT:   %DivWith.type.loc156_36.1 => constants.%DivWith.type.7ac
// CHECK:STDOUT:   %facet_type.loc156_20 => constants.%facet_type.d41
// CHECK:STDOUT:   %.Self.loc156_38.1 => constants.%.Self.73b
// CHECK:STDOUT:   %require_complete.loc156 => constants.%complete_type.67e
// CHECK:STDOUT:   %.Self.as_type.loc156_44.1 => constants.%.Self.as_type.e27
// CHECK:STDOUT:   %DivWith.assoc_type => constants.%DivWith.assoc_type.bfe
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.578
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.loc156 => constants.%DivWith.lookup_impl_witness.12e
// CHECK:STDOUT:   %impl.elem0.loc156_44.1 => constants.%impl.elem0.2db
// CHECK:STDOUT:   %Result.as_type.loc156_54.1 => constants.%Int64
// CHECK:STDOUT:   %facet_type.loc156_38 => constants.%facet_type.604
// CHECK:STDOUT:   %T.loc156_4.1 => constants.%facet_value.969
// CHECK:STDOUT:   %pattern_type.loc156 => constants.%pattern_type.171
// CHECK:STDOUT:   %pattern_type.loc157_16 => constants.%pattern_type.7b1
// CHECK:STDOUT:   %T.as_type.loc157_29.1 => constants.%Int32
// CHECK:STDOUT:   %pattern_type.loc157_27 => constants.%pattern_type.e9d
// CHECK:STDOUT:   %pattern_type.loc157_33 => constants.%pattern_type.7b1
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc157_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc157_27 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc157_33 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc159 => constants.%complete_type.dba
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.c32
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.loc159 => constants.%custom_witness.863
// CHECK:STDOUT:   %DivWith.facet => constants.%DivWith.facet.62d
// CHECK:STDOUT:   %DivWith.WithSelf.Op.type => constants.%DivWith.WithSelf.Op.type.90b
// CHECK:STDOUT:   %.loc159_5.4 => constants.%.cb62
// CHECK:STDOUT:   %impl.elem1.loc159_5.2 => constants.%Op.b9c47d.9
// CHECK:STDOUT:   %specific_impl_fn.loc159_5.3 => constants.%Op.b9c47d.9
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.2bb
// CHECK:STDOUT:   %.loc159_5.5 => constants.%.dda
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.a98c63.3
// CHECK:STDOUT:   %impl.elem0.loc159_5.2 => constants.%Int64.Op.cc70d5.1
// CHECK:STDOUT:   %specific_impl_fn.loc159_5.4 => constants.%Int64.Op.cc70d5.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ModWith.loc168(constants.%Destroy.facet.3de, constants.%Destroy.facet.3de, constants.%facet_value.0ea) {
// CHECK:STDOUT:   %Result.loc165_9.1 => constants.%Destroy.facet.3de
// CHECK:STDOUT:   %U.loc166_4.1 => constants.%Destroy.facet.3de
// CHECK:STDOUT:   %U.as_type.loc167_36.1 => constants.%Int64
// CHECK:STDOUT:   %ModWith.type.loc167_36.1 => constants.%ModWith.type.6b0
// CHECK:STDOUT:   %facet_type.loc167_20 => constants.%facet_type.dfd
// CHECK:STDOUT:   %.Self.loc167_38.1 => constants.%.Self.5bc
// CHECK:STDOUT:   %require_complete.loc167 => constants.%complete_type.76b
// CHECK:STDOUT:   %.Self.as_type.loc167_44.1 => constants.%.Self.as_type.97b
// CHECK:STDOUT:   %ModWith.assoc_type => constants.%ModWith.assoc_type.eb7
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.8a2
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.loc167 => constants.%ModWith.lookup_impl_witness.45e
// CHECK:STDOUT:   %impl.elem0.loc167_44.1 => constants.%impl.elem0.f40
// CHECK:STDOUT:   %Result.as_type.loc167_54.1 => constants.%Int64
// CHECK:STDOUT:   %facet_type.loc167_38 => constants.%facet_type.d93
// CHECK:STDOUT:   %T.loc167_4.1 => constants.%facet_value.0ea
// CHECK:STDOUT:   %pattern_type.loc167 => constants.%pattern_type.5a9
// CHECK:STDOUT:   %pattern_type.loc168_16 => constants.%pattern_type.7b1
// CHECK:STDOUT:   %T.as_type.loc168_29.1 => constants.%Int32
// CHECK:STDOUT:   %pattern_type.loc168_27 => constants.%pattern_type.e9d
// CHECK:STDOUT:   %pattern_type.loc168_33 => constants.%pattern_type.7b1
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc168_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc168_27 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc168_33 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc170 => constants.%complete_type.f9c
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.fa5
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.loc170 => constants.%custom_witness.eb8
// CHECK:STDOUT:   %ModWith.facet => constants.%ModWith.facet.02b
// CHECK:STDOUT:   %ModWith.WithSelf.Op.type => constants.%ModWith.WithSelf.Op.type.17d
// CHECK:STDOUT:   %.loc170_5.4 => constants.%.0ca
// CHECK:STDOUT:   %impl.elem1.loc170_5.2 => constants.%Op.b9c47d.10
// CHECK:STDOUT:   %specific_impl_fn.loc170_5.3 => constants.%Op.b9c47d.10
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.2bb
// CHECK:STDOUT:   %.loc170_5.5 => constants.%.dda
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.a98c63.3
// CHECK:STDOUT:   %impl.elem0.loc170_5.2 => constants.%Int64.Op.cc70d5.1
// CHECK:STDOUT:   %specific_impl_fn.loc170_5.4 => constants.%Int64.Op.cc70d5.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AddWith.loc124(constants.%Destroy.facet.3de, constants.%Destroy.facet.d87, constants.%facet_value.706) {
// CHECK:STDOUT:   %Result.loc121_9.1 => constants.%Destroy.facet.3de
// CHECK:STDOUT:   %U.loc122_4.1 => constants.%Destroy.facet.d87
// CHECK:STDOUT:   %U.as_type.loc123_36.1 => constants.%Int32
// CHECK:STDOUT:   %AddWith.type.loc123_36.1 => constants.%AddWith.type.78c
// CHECK:STDOUT:   %facet_type.loc123_20 => constants.%facet_type.9c9
// CHECK:STDOUT:   %.Self.loc123_38.1 => constants.%.Self.495
// CHECK:STDOUT:   %require_complete.loc123 => constants.%complete_type.f26
// CHECK:STDOUT:   %.Self.as_type.loc123_44.1 => constants.%.Self.as_type.154
// CHECK:STDOUT:   %AddWith.assoc_type => constants.%AddWith.assoc_type.882
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.5b9
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.loc123 => constants.%AddWith.lookup_impl_witness.4d9
// CHECK:STDOUT:   %impl.elem0.loc123_44.1 => constants.%impl.elem0.53a
// CHECK:STDOUT:   %Result.as_type.loc123_54.1 => constants.%Int64
// CHECK:STDOUT:   %facet_type.loc123_38 => constants.%facet_type.002
// CHECK:STDOUT:   %T.loc123_4.1 => constants.%facet_value.706
// CHECK:STDOUT:   %pattern_type.loc123 => constants.%pattern_type.429
// CHECK:STDOUT:   %pattern_type.loc124_16 => constants.%pattern_type.7b1
// CHECK:STDOUT:   %T.as_type.loc124_29.1 => constants.%Int64
// CHECK:STDOUT:   %pattern_type.loc124_27 => constants.%pattern_type.7b1
// CHECK:STDOUT:   %pattern_type.loc124_33 => constants.%pattern_type.e9d
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc124_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc124_27 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc124_33 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc126 => constants.%complete_type.c57
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.892
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.loc126 => constants.%custom_witness.483
// CHECK:STDOUT:   %AddWith.facet => constants.%AddWith.facet.6a1
// CHECK:STDOUT:   %AddWith.WithSelf.Op.type => constants.%AddWith.WithSelf.Op.type.5ee
// CHECK:STDOUT:   %.loc126_5.4 => constants.%.11a
// CHECK:STDOUT:   %impl.elem1.loc126_5.2 => constants.%Op.b9c47d.11
// CHECK:STDOUT:   %specific_impl_fn.loc126_5.3 => constants.%Op.b9c47d.11
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.2bb
// CHECK:STDOUT:   %.loc126_5.5 => constants.%.dda
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.a98c63.3
// CHECK:STDOUT:   %impl.elem0.loc126_5.2 => constants.%Int64.Op.cc70d5.1
// CHECK:STDOUT:   %specific_impl_fn.loc126_5.4 => constants.%Int64.Op.cc70d5.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @SubWith.loc135(constants.%Destroy.facet.3de, constants.%Destroy.facet.d87, constants.%facet_value.1d6) {
// CHECK:STDOUT:   %Result.loc132_9.1 => constants.%Destroy.facet.3de
// CHECK:STDOUT:   %U.loc133_4.1 => constants.%Destroy.facet.d87
// CHECK:STDOUT:   %U.as_type.loc134_36.1 => constants.%Int32
// CHECK:STDOUT:   %SubWith.type.loc134_36.1 => constants.%SubWith.type.629
// CHECK:STDOUT:   %facet_type.loc134_20 => constants.%facet_type.6c8
// CHECK:STDOUT:   %.Self.loc134_38.1 => constants.%.Self.cbb
// CHECK:STDOUT:   %require_complete.loc134 => constants.%complete_type.dae
// CHECK:STDOUT:   %.Self.as_type.loc134_44.1 => constants.%.Self.as_type.11e
// CHECK:STDOUT:   %SubWith.assoc_type => constants.%SubWith.assoc_type.a03
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.199
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.loc134 => constants.%SubWith.lookup_impl_witness.733
// CHECK:STDOUT:   %impl.elem0.loc134_44.1 => constants.%impl.elem0.e47
// CHECK:STDOUT:   %Result.as_type.loc134_54.1 => constants.%Int64
// CHECK:STDOUT:   %facet_type.loc134_38 => constants.%facet_type.a40
// CHECK:STDOUT:   %T.loc134_4.1 => constants.%facet_value.1d6
// CHECK:STDOUT:   %pattern_type.loc134 => constants.%pattern_type.59c
// CHECK:STDOUT:   %pattern_type.loc135_16 => constants.%pattern_type.7b1
// CHECK:STDOUT:   %T.as_type.loc135_29.1 => constants.%Int64
// CHECK:STDOUT:   %pattern_type.loc135_27 => constants.%pattern_type.7b1
// CHECK:STDOUT:   %pattern_type.loc135_33 => constants.%pattern_type.e9d
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc135_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc135_27 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc135_33 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc137 => constants.%complete_type.48a
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.bff
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.loc137 => constants.%custom_witness.095
// CHECK:STDOUT:   %SubWith.facet => constants.%SubWith.facet.8cb
// CHECK:STDOUT:   %SubWith.WithSelf.Op.type => constants.%SubWith.WithSelf.Op.type.1b8
// CHECK:STDOUT:   %.loc137_5.4 => constants.%.51b
// CHECK:STDOUT:   %impl.elem1.loc137_5.2 => constants.%Op.b9c47d.12
// CHECK:STDOUT:   %specific_impl_fn.loc137_5.3 => constants.%Op.b9c47d.12
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.2bb
// CHECK:STDOUT:   %.loc137_5.5 => constants.%.dda
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.a98c63.3
// CHECK:STDOUT:   %impl.elem0.loc137_5.2 => constants.%Int64.Op.cc70d5.1
// CHECK:STDOUT:   %specific_impl_fn.loc137_5.4 => constants.%Int64.Op.cc70d5.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @MulWith.loc146(constants.%Destroy.facet.3de, constants.%Destroy.facet.d87, constants.%facet_value.3ed) {
// CHECK:STDOUT:   %Result.loc143_9.1 => constants.%Destroy.facet.3de
// CHECK:STDOUT:   %U.loc144_4.1 => constants.%Destroy.facet.d87
// CHECK:STDOUT:   %U.as_type.loc145_36.1 => constants.%Int32
// CHECK:STDOUT:   %MulWith.type.loc145_36.1 => constants.%MulWith.type.3ee
// CHECK:STDOUT:   %facet_type.loc145_20 => constants.%facet_type.299
// CHECK:STDOUT:   %.Self.loc145_38.1 => constants.%.Self.1c7
// CHECK:STDOUT:   %require_complete.loc145 => constants.%complete_type.0cf
// CHECK:STDOUT:   %.Self.as_type.loc145_44.1 => constants.%.Self.as_type.ba5
// CHECK:STDOUT:   %MulWith.assoc_type => constants.%MulWith.assoc_type.564
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.6fa
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.loc145 => constants.%MulWith.lookup_impl_witness.5ca
// CHECK:STDOUT:   %impl.elem0.loc145_44.1 => constants.%impl.elem0.cab
// CHECK:STDOUT:   %Result.as_type.loc145_54.1 => constants.%Int64
// CHECK:STDOUT:   %facet_type.loc145_38 => constants.%facet_type.ac8
// CHECK:STDOUT:   %T.loc145_4.1 => constants.%facet_value.3ed
// CHECK:STDOUT:   %pattern_type.loc145 => constants.%pattern_type.5de
// CHECK:STDOUT:   %pattern_type.loc146_16 => constants.%pattern_type.7b1
// CHECK:STDOUT:   %T.as_type.loc146_29.1 => constants.%Int64
// CHECK:STDOUT:   %pattern_type.loc146_27 => constants.%pattern_type.7b1
// CHECK:STDOUT:   %pattern_type.loc146_33 => constants.%pattern_type.e9d
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc146_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc146_27 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc146_33 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc148 => constants.%complete_type.6e9
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.00d
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.loc148 => constants.%custom_witness.f6e
// CHECK:STDOUT:   %MulWith.facet => constants.%MulWith.facet.e61
// CHECK:STDOUT:   %MulWith.WithSelf.Op.type => constants.%MulWith.WithSelf.Op.type.d31
// CHECK:STDOUT:   %.loc148_5.4 => constants.%.2e4
// CHECK:STDOUT:   %impl.elem1.loc148_5.2 => constants.%Op.b9c47d.13
// CHECK:STDOUT:   %specific_impl_fn.loc148_5.3 => constants.%Op.b9c47d.13
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.2bb
// CHECK:STDOUT:   %.loc148_5.5 => constants.%.dda
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.a98c63.3
// CHECK:STDOUT:   %impl.elem0.loc148_5.2 => constants.%Int64.Op.cc70d5.1
// CHECK:STDOUT:   %specific_impl_fn.loc148_5.4 => constants.%Int64.Op.cc70d5.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @DivWith.loc157(constants.%Destroy.facet.3de, constants.%Destroy.facet.d87, constants.%facet_value.4d6) {
// CHECK:STDOUT:   %Result.loc154_9.1 => constants.%Destroy.facet.3de
// CHECK:STDOUT:   %U.loc155_4.1 => constants.%Destroy.facet.d87
// CHECK:STDOUT:   %U.as_type.loc156_36.1 => constants.%Int32
// CHECK:STDOUT:   %DivWith.type.loc156_36.1 => constants.%DivWith.type.09b
// CHECK:STDOUT:   %facet_type.loc156_20 => constants.%facet_type.c6e
// CHECK:STDOUT:   %.Self.loc156_38.1 => constants.%.Self.988
// CHECK:STDOUT:   %require_complete.loc156 => constants.%complete_type.36c
// CHECK:STDOUT:   %.Self.as_type.loc156_44.1 => constants.%.Self.as_type.f38
// CHECK:STDOUT:   %DivWith.assoc_type => constants.%DivWith.assoc_type.a72
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.d0a
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.loc156 => constants.%DivWith.lookup_impl_witness.c00
// CHECK:STDOUT:   %impl.elem0.loc156_44.1 => constants.%impl.elem0.7a0
// CHECK:STDOUT:   %Result.as_type.loc156_54.1 => constants.%Int64
// CHECK:STDOUT:   %facet_type.loc156_38 => constants.%facet_type.b8e
// CHECK:STDOUT:   %T.loc156_4.1 => constants.%facet_value.4d6
// CHECK:STDOUT:   %pattern_type.loc156 => constants.%pattern_type.435
// CHECK:STDOUT:   %pattern_type.loc157_16 => constants.%pattern_type.7b1
// CHECK:STDOUT:   %T.as_type.loc157_29.1 => constants.%Int64
// CHECK:STDOUT:   %pattern_type.loc157_27 => constants.%pattern_type.7b1
// CHECK:STDOUT:   %pattern_type.loc157_33 => constants.%pattern_type.e9d
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc157_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc157_27 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc157_33 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc159 => constants.%complete_type.627
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.489
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.loc159 => constants.%custom_witness.4e5
// CHECK:STDOUT:   %DivWith.facet => constants.%DivWith.facet.b24
// CHECK:STDOUT:   %DivWith.WithSelf.Op.type => constants.%DivWith.WithSelf.Op.type.58d
// CHECK:STDOUT:   %.loc159_5.4 => constants.%.f2d
// CHECK:STDOUT:   %impl.elem1.loc159_5.2 => constants.%Op.b9c47d.14
// CHECK:STDOUT:   %specific_impl_fn.loc159_5.3 => constants.%Op.b9c47d.14
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.2bb
// CHECK:STDOUT:   %.loc159_5.5 => constants.%.dda
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.a98c63.3
// CHECK:STDOUT:   %impl.elem0.loc159_5.2 => constants.%Int64.Op.cc70d5.1
// CHECK:STDOUT:   %specific_impl_fn.loc159_5.4 => constants.%Int64.Op.cc70d5.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ModWith.loc168(constants.%Destroy.facet.3de, constants.%Destroy.facet.d87, constants.%facet_value.8fa) {
// CHECK:STDOUT:   %Result.loc165_9.1 => constants.%Destroy.facet.3de
// CHECK:STDOUT:   %U.loc166_4.1 => constants.%Destroy.facet.d87
// CHECK:STDOUT:   %U.as_type.loc167_36.1 => constants.%Int32
// CHECK:STDOUT:   %ModWith.type.loc167_36.1 => constants.%ModWith.type.6f6
// CHECK:STDOUT:   %facet_type.loc167_20 => constants.%facet_type.22f
// CHECK:STDOUT:   %.Self.loc167_38.1 => constants.%.Self.572
// CHECK:STDOUT:   %require_complete.loc167 => constants.%complete_type.e22
// CHECK:STDOUT:   %.Self.as_type.loc167_44.1 => constants.%.Self.as_type.aac
// CHECK:STDOUT:   %ModWith.assoc_type => constants.%ModWith.assoc_type.d09
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.fb6
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.loc167 => constants.%ModWith.lookup_impl_witness.18e
// CHECK:STDOUT:   %impl.elem0.loc167_44.1 => constants.%impl.elem0.288
// CHECK:STDOUT:   %Result.as_type.loc167_54.1 => constants.%Int64
// CHECK:STDOUT:   %facet_type.loc167_38 => constants.%facet_type.dbb
// CHECK:STDOUT:   %T.loc167_4.1 => constants.%facet_value.8fa
// CHECK:STDOUT:   %pattern_type.loc167 => constants.%pattern_type.4c3
// CHECK:STDOUT:   %pattern_type.loc168_16 => constants.%pattern_type.7b1
// CHECK:STDOUT:   %T.as_type.loc168_29.1 => constants.%Int64
// CHECK:STDOUT:   %pattern_type.loc168_27 => constants.%pattern_type.7b1
// CHECK:STDOUT:   %pattern_type.loc168_33 => constants.%pattern_type.e9d
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc168_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc168_27 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc168_33 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc170 => constants.%complete_type.1cf
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.56e
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.loc170 => constants.%custom_witness.99b
// CHECK:STDOUT:   %ModWith.facet => constants.%ModWith.facet.4c1
// CHECK:STDOUT:   %ModWith.WithSelf.Op.type => constants.%ModWith.WithSelf.Op.type.fb3
// CHECK:STDOUT:   %.loc170_5.4 => constants.%.7b9
// CHECK:STDOUT:   %impl.elem1.loc170_5.2 => constants.%Op.b9c47d.15
// CHECK:STDOUT:   %specific_impl_fn.loc170_5.3 => constants.%Op.b9c47d.15
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.2bb
// CHECK:STDOUT:   %.loc170_5.5 => constants.%.dda
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.a98c63.3
// CHECK:STDOUT:   %impl.elem0.loc170_5.2 => constants.%Int64.Op.cc70d5.1
// CHECK:STDOUT:   %specific_impl_fn.loc170_5.4 => constants.%Int64.Op.cc70d5.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AddAssignWith.loc225(constants.%U.67d, constants.%T.c9e) {
// CHECK:STDOUT:   %U.loc225_19.1 => constants.%U.67d
// CHECK:STDOUT:   %AddAssignWith.type.loc225_52.1 => constants.%AddAssignWith.type.f7e94b.2
// CHECK:STDOUT:   %T.loc225_29.1 => constants.%T.c9e
// CHECK:STDOUT:   %pattern_type.loc225_29 => constants.%pattern_type.323
// CHECK:STDOUT:   %T.as_type.loc225_62.1 => constants.%T.as_type.82b
// CHECK:STDOUT:   %pattern_type.loc225_60 => constants.%pattern_type.d713dc.2
// CHECK:STDOUT:   %pattern_type.loc225_66 => constants.%pattern_type.51d1c4.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @SubAssignWith.loc231(constants.%U.67d, constants.%T.a96) {
// CHECK:STDOUT:   %U.loc231_19.1 => constants.%U.67d
// CHECK:STDOUT:   %SubAssignWith.type.loc231_52.1 => constants.%SubAssignWith.type.591a5f.2
// CHECK:STDOUT:   %T.loc231_29.1 => constants.%T.a96
// CHECK:STDOUT:   %pattern_type.loc231_29 => constants.%pattern_type.46c
// CHECK:STDOUT:   %T.as_type.loc231_62.1 => constants.%T.as_type.765
// CHECK:STDOUT:   %pattern_type.loc231_60 => constants.%pattern_type.f40fa3.2
// CHECK:STDOUT:   %pattern_type.loc231_66 => constants.%pattern_type.51d1c4.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @MulAssignWith.loc237(constants.%U.67d, constants.%T.1ab) {
// CHECK:STDOUT:   %U.loc237_19.1 => constants.%U.67d
// CHECK:STDOUT:   %MulAssignWith.type.loc237_52.1 => constants.%MulAssignWith.type.b9aa5c.2
// CHECK:STDOUT:   %T.loc237_29.1 => constants.%T.1ab
// CHECK:STDOUT:   %pattern_type.loc237_29 => constants.%pattern_type.a30
// CHECK:STDOUT:   %T.as_type.loc237_62.1 => constants.%T.as_type.e75
// CHECK:STDOUT:   %pattern_type.loc237_60 => constants.%pattern_type.07a4e0.2
// CHECK:STDOUT:   %pattern_type.loc237_66 => constants.%pattern_type.51d1c4.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @DivAssignWith.loc243(constants.%U.67d, constants.%T.439) {
// CHECK:STDOUT:   %U.loc243_19.1 => constants.%U.67d
// CHECK:STDOUT:   %DivAssignWith.type.loc243_52.1 => constants.%DivAssignWith.type.c97ac9.2
// CHECK:STDOUT:   %T.loc243_29.1 => constants.%T.439
// CHECK:STDOUT:   %pattern_type.loc243_29 => constants.%pattern_type.fa6
// CHECK:STDOUT:   %T.as_type.loc243_62.1 => constants.%T.as_type.5d7
// CHECK:STDOUT:   %pattern_type.loc243_60 => constants.%pattern_type.12d907.2
// CHECK:STDOUT:   %pattern_type.loc243_66 => constants.%pattern_type.51d1c4.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ModAssignWith.loc249(constants.%U.67d, constants.%T.da0) {
// CHECK:STDOUT:   %U.loc249_19.1 => constants.%U.67d
// CHECK:STDOUT:   %ModAssignWith.type.loc249_52.1 => constants.%ModAssignWith.type.8faa47.2
// CHECK:STDOUT:   %T.loc249_29.1 => constants.%T.da0
// CHECK:STDOUT:   %pattern_type.loc249_29 => constants.%pattern_type.383
// CHECK:STDOUT:   %T.as_type.loc249_62.1 => constants.%T.as_type.4cb
// CHECK:STDOUT:   %pattern_type.loc249_60 => constants.%pattern_type.7b701e.2
// CHECK:STDOUT:   %pattern_type.loc249_66 => constants.%pattern_type.51d1c4.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AddAssignWith.loc225(constants.%i16, constants.%AddAssignWith.facet.1b3) {
// CHECK:STDOUT:   %U.loc225_19.1 => constants.%i16
// CHECK:STDOUT:   %AddAssignWith.type.loc225_52.1 => constants.%AddAssignWith.type.35f
// CHECK:STDOUT:   %T.loc225_29.1 => constants.%AddAssignWith.facet.1b3
// CHECK:STDOUT:   %pattern_type.loc225_29 => constants.%pattern_type.25b
// CHECK:STDOUT:   %T.as_type.loc225_62.1 => constants.%Int16
// CHECK:STDOUT:   %pattern_type.loc225_60 => constants.%pattern_type.802
// CHECK:STDOUT:   %pattern_type.loc225_66 => constants.%pattern_type.e1b
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc225_55 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc225_66 => constants.%complete_type.2da
// CHECK:STDOUT:   %require_complete.loc227 => constants.%complete_type.f45
// CHECK:STDOUT:   %AddAssignWith.assoc_type => constants.%AddAssignWith.assoc_type.3c9
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.d69
// CHECK:STDOUT:   %AddAssignWith.WithSelf.Op.type => constants.%AddAssignWith.WithSelf.Op.type.f5d
// CHECK:STDOUT:   %.loc227_5.2 => constants.%.c71
// CHECK:STDOUT:   %AddAssignWith.lookup_impl_witness => constants.%custom_witness.364
// CHECK:STDOUT:   %impl.elem0.loc227_5.2 => constants.%Int16.Op.cc70d5.13
// CHECK:STDOUT:   %specific_impl_fn.loc227_5.2 => constants.%Int16.Op.cc70d5.13
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @SubAssignWith.loc231(constants.%i16, constants.%SubAssignWith.facet.45a) {
// CHECK:STDOUT:   %U.loc231_19.1 => constants.%i16
// CHECK:STDOUT:   %SubAssignWith.type.loc231_52.1 => constants.%SubAssignWith.type.121
// CHECK:STDOUT:   %T.loc231_29.1 => constants.%SubAssignWith.facet.45a
// CHECK:STDOUT:   %pattern_type.loc231_29 => constants.%pattern_type.1c3
// CHECK:STDOUT:   %T.as_type.loc231_62.1 => constants.%Int16
// CHECK:STDOUT:   %pattern_type.loc231_60 => constants.%pattern_type.802
// CHECK:STDOUT:   %pattern_type.loc231_66 => constants.%pattern_type.e1b
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc231_55 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc231_66 => constants.%complete_type.2da
// CHECK:STDOUT:   %require_complete.loc233 => constants.%complete_type.c74
// CHECK:STDOUT:   %SubAssignWith.assoc_type => constants.%SubAssignWith.assoc_type.5f8
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.ce1
// CHECK:STDOUT:   %SubAssignWith.WithSelf.Op.type => constants.%SubAssignWith.WithSelf.Op.type.224
// CHECK:STDOUT:   %.loc233_5.2 => constants.%.9c3
// CHECK:STDOUT:   %SubAssignWith.lookup_impl_witness => constants.%custom_witness.4e8
// CHECK:STDOUT:   %impl.elem0.loc233_5.2 => constants.%Int16.Op.cc70d5.14
// CHECK:STDOUT:   %specific_impl_fn.loc233_5.2 => constants.%Int16.Op.cc70d5.14
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @MulAssignWith.loc237(constants.%i16, constants.%MulAssignWith.facet.805) {
// CHECK:STDOUT:   %U.loc237_19.1 => constants.%i16
// CHECK:STDOUT:   %MulAssignWith.type.loc237_52.1 => constants.%MulAssignWith.type.b7e
// CHECK:STDOUT:   %T.loc237_29.1 => constants.%MulAssignWith.facet.805
// CHECK:STDOUT:   %pattern_type.loc237_29 => constants.%pattern_type.ebd
// CHECK:STDOUT:   %T.as_type.loc237_62.1 => constants.%Int16
// CHECK:STDOUT:   %pattern_type.loc237_60 => constants.%pattern_type.802
// CHECK:STDOUT:   %pattern_type.loc237_66 => constants.%pattern_type.e1b
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc237_55 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc237_66 => constants.%complete_type.2da
// CHECK:STDOUT:   %require_complete.loc239 => constants.%complete_type.f93
// CHECK:STDOUT:   %MulAssignWith.assoc_type => constants.%MulAssignWith.assoc_type.551
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.75e
// CHECK:STDOUT:   %MulAssignWith.WithSelf.Op.type => constants.%MulAssignWith.WithSelf.Op.type.354
// CHECK:STDOUT:   %.loc239_5.2 => constants.%.cbb
// CHECK:STDOUT:   %MulAssignWith.lookup_impl_witness => constants.%custom_witness.b74
// CHECK:STDOUT:   %impl.elem0.loc239_5.2 => constants.%Int16.Op.cc70d5.15
// CHECK:STDOUT:   %specific_impl_fn.loc239_5.2 => constants.%Int16.Op.cc70d5.15
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @DivAssignWith.loc243(constants.%i16, constants.%DivAssignWith.facet.ac9) {
// CHECK:STDOUT:   %U.loc243_19.1 => constants.%i16
// CHECK:STDOUT:   %DivAssignWith.type.loc243_52.1 => constants.%DivAssignWith.type.0d6
// CHECK:STDOUT:   %T.loc243_29.1 => constants.%DivAssignWith.facet.ac9
// CHECK:STDOUT:   %pattern_type.loc243_29 => constants.%pattern_type.1f58
// CHECK:STDOUT:   %T.as_type.loc243_62.1 => constants.%Int16
// CHECK:STDOUT:   %pattern_type.loc243_60 => constants.%pattern_type.802
// CHECK:STDOUT:   %pattern_type.loc243_66 => constants.%pattern_type.e1b
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc243_55 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc243_66 => constants.%complete_type.2da
// CHECK:STDOUT:   %require_complete.loc245 => constants.%complete_type.913
// CHECK:STDOUT:   %DivAssignWith.assoc_type => constants.%DivAssignWith.assoc_type.01d
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.f56
// CHECK:STDOUT:   %DivAssignWith.WithSelf.Op.type => constants.%DivAssignWith.WithSelf.Op.type.510
// CHECK:STDOUT:   %.loc245_5.2 => constants.%.b82
// CHECK:STDOUT:   %DivAssignWith.lookup_impl_witness => constants.%custom_witness.8b6
// CHECK:STDOUT:   %impl.elem0.loc245_5.2 => constants.%Int16.Op.cc70d5.16
// CHECK:STDOUT:   %specific_impl_fn.loc245_5.2 => constants.%Int16.Op.cc70d5.16
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ModAssignWith.loc249(constants.%i16, constants.%ModAssignWith.facet.350) {
// CHECK:STDOUT:   %U.loc249_19.1 => constants.%i16
// CHECK:STDOUT:   %ModAssignWith.type.loc249_52.1 => constants.%ModAssignWith.type.d0f
// CHECK:STDOUT:   %T.loc249_29.1 => constants.%ModAssignWith.facet.350
// CHECK:STDOUT:   %pattern_type.loc249_29 => constants.%pattern_type.e7a
// CHECK:STDOUT:   %T.as_type.loc249_62.1 => constants.%Int16
// CHECK:STDOUT:   %pattern_type.loc249_60 => constants.%pattern_type.802
// CHECK:STDOUT:   %pattern_type.loc249_66 => constants.%pattern_type.e1b
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc249_55 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc249_66 => constants.%complete_type.2da
// CHECK:STDOUT:   %require_complete.loc251 => constants.%complete_type.597
// CHECK:STDOUT:   %ModAssignWith.assoc_type => constants.%ModAssignWith.assoc_type.9c3
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.e07
// CHECK:STDOUT:   %ModAssignWith.WithSelf.Op.type => constants.%ModAssignWith.WithSelf.Op.type.ad4
// CHECK:STDOUT:   %.loc251_5.2 => constants.%.003
// CHECK:STDOUT:   %ModAssignWith.lookup_impl_witness => constants.%custom_witness.e6d
// CHECK:STDOUT:   %impl.elem0.loc251_5.2 => constants.%Int16.Op.cc70d5.17
// CHECK:STDOUT:   %specific_impl_fn.loc251_5.2 => constants.%Int16.Op.cc70d5.17
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AddAssignWith.loc225(constants.%Int32, constants.%AddAssignWith.facet.610) {
// CHECK:STDOUT:   %U.loc225_19.1 => constants.%Int32
// CHECK:STDOUT:   %AddAssignWith.type.loc225_52.1 => constants.%AddAssignWith.type.3e8
// CHECK:STDOUT:   %T.loc225_29.1 => constants.%AddAssignWith.facet.610
// CHECK:STDOUT:   %pattern_type.loc225_29 => constants.%pattern_type.ae6
// CHECK:STDOUT:   %T.as_type.loc225_62.1 => constants.%Int32
// CHECK:STDOUT:   %pattern_type.loc225_60 => constants.%pattern_type.e9d
// CHECK:STDOUT:   %pattern_type.loc225_66 => constants.%pattern_type.e9d
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc225_55 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc225_66 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc227 => constants.%complete_type.d4d
// CHECK:STDOUT:   %AddAssignWith.assoc_type => constants.%AddAssignWith.assoc_type.115
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.8b4
// CHECK:STDOUT:   %AddAssignWith.WithSelf.Op.type => constants.%AddAssignWith.WithSelf.Op.type.b28
// CHECK:STDOUT:   %.loc227_5.2 => constants.%.05e
// CHECK:STDOUT:   %AddAssignWith.lookup_impl_witness => constants.%custom_witness.ec4e63.1
// CHECK:STDOUT:   %impl.elem0.loc227_5.2 => constants.%Int32.Op.cc70d5.12
// CHECK:STDOUT:   %specific_impl_fn.loc227_5.2 => constants.%Int32.Op.cc70d5.12
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @SubAssignWith.loc231(constants.%Int32, constants.%SubAssignWith.facet.fb2) {
// CHECK:STDOUT:   %U.loc231_19.1 => constants.%Int32
// CHECK:STDOUT:   %SubAssignWith.type.loc231_52.1 => constants.%SubAssignWith.type.678
// CHECK:STDOUT:   %T.loc231_29.1 => constants.%SubAssignWith.facet.fb2
// CHECK:STDOUT:   %pattern_type.loc231_29 => constants.%pattern_type.470
// CHECK:STDOUT:   %T.as_type.loc231_62.1 => constants.%Int32
// CHECK:STDOUT:   %pattern_type.loc231_60 => constants.%pattern_type.e9d
// CHECK:STDOUT:   %pattern_type.loc231_66 => constants.%pattern_type.e9d
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc231_55 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc231_66 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc233 => constants.%complete_type.838
// CHECK:STDOUT:   %SubAssignWith.assoc_type => constants.%SubAssignWith.assoc_type.53a
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.60e
// CHECK:STDOUT:   %SubAssignWith.WithSelf.Op.type => constants.%SubAssignWith.WithSelf.Op.type.7bd
// CHECK:STDOUT:   %.loc233_5.2 => constants.%.cd8
// CHECK:STDOUT:   %SubAssignWith.lookup_impl_witness => constants.%custom_witness.75ca2a.1
// CHECK:STDOUT:   %impl.elem0.loc233_5.2 => constants.%Int32.Op.cc70d5.14
// CHECK:STDOUT:   %specific_impl_fn.loc233_5.2 => constants.%Int32.Op.cc70d5.14
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @MulAssignWith.loc237(constants.%Int32, constants.%MulAssignWith.facet.8f5) {
// CHECK:STDOUT:   %U.loc237_19.1 => constants.%Int32
// CHECK:STDOUT:   %MulAssignWith.type.loc237_52.1 => constants.%MulAssignWith.type.011
// CHECK:STDOUT:   %T.loc237_29.1 => constants.%MulAssignWith.facet.8f5
// CHECK:STDOUT:   %pattern_type.loc237_29 => constants.%pattern_type.c27
// CHECK:STDOUT:   %T.as_type.loc237_62.1 => constants.%Int32
// CHECK:STDOUT:   %pattern_type.loc237_60 => constants.%pattern_type.e9d
// CHECK:STDOUT:   %pattern_type.loc237_66 => constants.%pattern_type.e9d
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc237_55 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc237_66 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc239 => constants.%complete_type.908
// CHECK:STDOUT:   %MulAssignWith.assoc_type => constants.%MulAssignWith.assoc_type.f6b
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.1a2
// CHECK:STDOUT:   %MulAssignWith.WithSelf.Op.type => constants.%MulAssignWith.WithSelf.Op.type.9a6
// CHECK:STDOUT:   %.loc239_5.2 => constants.%.300
// CHECK:STDOUT:   %MulAssignWith.lookup_impl_witness => constants.%custom_witness.7d92f3.1
// CHECK:STDOUT:   %impl.elem0.loc239_5.2 => constants.%Int32.Op.cc70d5.15
// CHECK:STDOUT:   %specific_impl_fn.loc239_5.2 => constants.%Int32.Op.cc70d5.15
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @DivAssignWith.loc243(constants.%Int32, constants.%DivAssignWith.facet.3bc) {
// CHECK:STDOUT:   %U.loc243_19.1 => constants.%Int32
// CHECK:STDOUT:   %DivAssignWith.type.loc243_52.1 => constants.%DivAssignWith.type.487
// CHECK:STDOUT:   %T.loc243_29.1 => constants.%DivAssignWith.facet.3bc
// CHECK:STDOUT:   %pattern_type.loc243_29 => constants.%pattern_type.492
// CHECK:STDOUT:   %T.as_type.loc243_62.1 => constants.%Int32
// CHECK:STDOUT:   %pattern_type.loc243_60 => constants.%pattern_type.e9d
// CHECK:STDOUT:   %pattern_type.loc243_66 => constants.%pattern_type.e9d
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc243_55 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc243_66 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc245 => constants.%complete_type.054
// CHECK:STDOUT:   %DivAssignWith.assoc_type => constants.%DivAssignWith.assoc_type.047
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.cd7
// CHECK:STDOUT:   %DivAssignWith.WithSelf.Op.type => constants.%DivAssignWith.WithSelf.Op.type.1ec
// CHECK:STDOUT:   %.loc245_5.2 => constants.%.d63
// CHECK:STDOUT:   %DivAssignWith.lookup_impl_witness => constants.%custom_witness.cdc44b.1
// CHECK:STDOUT:   %impl.elem0.loc245_5.2 => constants.%Int32.Op.cc70d5.16
// CHECK:STDOUT:   %specific_impl_fn.loc245_5.2 => constants.%Int32.Op.cc70d5.16
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ModAssignWith.loc249(constants.%Int32, constants.%ModAssignWith.facet.b73) {
// CHECK:STDOUT:   %U.loc249_19.1 => constants.%Int32
// CHECK:STDOUT:   %ModAssignWith.type.loc249_52.1 => constants.%ModAssignWith.type.499
// CHECK:STDOUT:   %T.loc249_29.1 => constants.%ModAssignWith.facet.b73
// CHECK:STDOUT:   %pattern_type.loc249_29 => constants.%pattern_type.f79
// CHECK:STDOUT:   %T.as_type.loc249_62.1 => constants.%Int32
// CHECK:STDOUT:   %pattern_type.loc249_60 => constants.%pattern_type.e9d
// CHECK:STDOUT:   %pattern_type.loc249_66 => constants.%pattern_type.e9d
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc249_55 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc249_66 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc251 => constants.%complete_type.022
// CHECK:STDOUT:   %ModAssignWith.assoc_type => constants.%ModAssignWith.assoc_type.e0e
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.928
// CHECK:STDOUT:   %ModAssignWith.WithSelf.Op.type => constants.%ModAssignWith.WithSelf.Op.type.1a7
// CHECK:STDOUT:   %.loc251_5.2 => constants.%.029
// CHECK:STDOUT:   %ModAssignWith.lookup_impl_witness => constants.%custom_witness.ec99b7.1
// CHECK:STDOUT:   %impl.elem0.loc251_5.2 => constants.%Int32.Op.cc70d5.17
// CHECK:STDOUT:   %specific_impl_fn.loc251_5.2 => constants.%Int32.Op.cc70d5.17
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AddAssignWith.loc225(constants.%Int32, constants.%AddAssignWith.facet.080) {
// CHECK:STDOUT:   %U.loc225_19.1 => constants.%Int32
// CHECK:STDOUT:   %AddAssignWith.type.loc225_52.1 => constants.%AddAssignWith.type.3e8
// CHECK:STDOUT:   %T.loc225_29.1 => constants.%AddAssignWith.facet.080
// CHECK:STDOUT:   %pattern_type.loc225_29 => constants.%pattern_type.ae6
// CHECK:STDOUT:   %T.as_type.loc225_62.1 => constants.%Int64
// CHECK:STDOUT:   %pattern_type.loc225_60 => constants.%pattern_type.7b1
// CHECK:STDOUT:   %pattern_type.loc225_66 => constants.%pattern_type.e9d
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc225_55 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc225_66 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc227 => constants.%complete_type.d4d
// CHECK:STDOUT:   %AddAssignWith.assoc_type => constants.%AddAssignWith.assoc_type.115
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.8b4
// CHECK:STDOUT:   %AddAssignWith.WithSelf.Op.type => constants.%AddAssignWith.WithSelf.Op.type.272
// CHECK:STDOUT:   %.loc227_5.2 => constants.%.257
// CHECK:STDOUT:   %AddAssignWith.lookup_impl_witness => constants.%custom_witness.ec4e63.2
// CHECK:STDOUT:   %impl.elem0.loc227_5.2 => constants.%Int64.Op.cc70d5.2
// CHECK:STDOUT:   %specific_impl_fn.loc227_5.2 => constants.%Int64.Op.cc70d5.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @SubAssignWith.loc231(constants.%Int32, constants.%SubAssignWith.facet.8bc) {
// CHECK:STDOUT:   %U.loc231_19.1 => constants.%Int32
// CHECK:STDOUT:   %SubAssignWith.type.loc231_52.1 => constants.%SubAssignWith.type.678
// CHECK:STDOUT:   %T.loc231_29.1 => constants.%SubAssignWith.facet.8bc
// CHECK:STDOUT:   %pattern_type.loc231_29 => constants.%pattern_type.470
// CHECK:STDOUT:   %T.as_type.loc231_62.1 => constants.%Int64
// CHECK:STDOUT:   %pattern_type.loc231_60 => constants.%pattern_type.7b1
// CHECK:STDOUT:   %pattern_type.loc231_66 => constants.%pattern_type.e9d
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc231_55 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc231_66 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc233 => constants.%complete_type.838
// CHECK:STDOUT:   %SubAssignWith.assoc_type => constants.%SubAssignWith.assoc_type.53a
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.60e
// CHECK:STDOUT:   %SubAssignWith.WithSelf.Op.type => constants.%SubAssignWith.WithSelf.Op.type.461
// CHECK:STDOUT:   %.loc233_5.2 => constants.%.e92
// CHECK:STDOUT:   %SubAssignWith.lookup_impl_witness => constants.%custom_witness.75ca2a.2
// CHECK:STDOUT:   %impl.elem0.loc233_5.2 => constants.%Int64.Op.cc70d5.4
// CHECK:STDOUT:   %specific_impl_fn.loc233_5.2 => constants.%Int64.Op.cc70d5.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @MulAssignWith.loc237(constants.%Int32, constants.%MulAssignWith.facet.990) {
// CHECK:STDOUT:   %U.loc237_19.1 => constants.%Int32
// CHECK:STDOUT:   %MulAssignWith.type.loc237_52.1 => constants.%MulAssignWith.type.011
// CHECK:STDOUT:   %T.loc237_29.1 => constants.%MulAssignWith.facet.990
// CHECK:STDOUT:   %pattern_type.loc237_29 => constants.%pattern_type.c27
// CHECK:STDOUT:   %T.as_type.loc237_62.1 => constants.%Int64
// CHECK:STDOUT:   %pattern_type.loc237_60 => constants.%pattern_type.7b1
// CHECK:STDOUT:   %pattern_type.loc237_66 => constants.%pattern_type.e9d
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc237_55 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc237_66 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc239 => constants.%complete_type.908
// CHECK:STDOUT:   %MulAssignWith.assoc_type => constants.%MulAssignWith.assoc_type.f6b
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.1a2
// CHECK:STDOUT:   %MulAssignWith.WithSelf.Op.type => constants.%MulAssignWith.WithSelf.Op.type.152
// CHECK:STDOUT:   %.loc239_5.2 => constants.%.694
// CHECK:STDOUT:   %MulAssignWith.lookup_impl_witness => constants.%custom_witness.7d92f3.2
// CHECK:STDOUT:   %impl.elem0.loc239_5.2 => constants.%Int64.Op.cc70d5.5
// CHECK:STDOUT:   %specific_impl_fn.loc239_5.2 => constants.%Int64.Op.cc70d5.5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @DivAssignWith.loc243(constants.%Int32, constants.%DivAssignWith.facet.3eb) {
// CHECK:STDOUT:   %U.loc243_19.1 => constants.%Int32
// CHECK:STDOUT:   %DivAssignWith.type.loc243_52.1 => constants.%DivAssignWith.type.487
// CHECK:STDOUT:   %T.loc243_29.1 => constants.%DivAssignWith.facet.3eb
// CHECK:STDOUT:   %pattern_type.loc243_29 => constants.%pattern_type.492
// CHECK:STDOUT:   %T.as_type.loc243_62.1 => constants.%Int64
// CHECK:STDOUT:   %pattern_type.loc243_60 => constants.%pattern_type.7b1
// CHECK:STDOUT:   %pattern_type.loc243_66 => constants.%pattern_type.e9d
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc243_55 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc243_66 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc245 => constants.%complete_type.054
// CHECK:STDOUT:   %DivAssignWith.assoc_type => constants.%DivAssignWith.assoc_type.047
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.cd7
// CHECK:STDOUT:   %DivAssignWith.WithSelf.Op.type => constants.%DivAssignWith.WithSelf.Op.type.53f
// CHECK:STDOUT:   %.loc245_5.2 => constants.%.09f
// CHECK:STDOUT:   %DivAssignWith.lookup_impl_witness => constants.%custom_witness.cdc44b.2
// CHECK:STDOUT:   %impl.elem0.loc245_5.2 => constants.%Int64.Op.cc70d5.6
// CHECK:STDOUT:   %specific_impl_fn.loc245_5.2 => constants.%Int64.Op.cc70d5.6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ModAssignWith.loc249(constants.%Int32, constants.%ModAssignWith.facet.2d7) {
// CHECK:STDOUT:   %U.loc249_19.1 => constants.%Int32
// CHECK:STDOUT:   %ModAssignWith.type.loc249_52.1 => constants.%ModAssignWith.type.499
// CHECK:STDOUT:   %T.loc249_29.1 => constants.%ModAssignWith.facet.2d7
// CHECK:STDOUT:   %pattern_type.loc249_29 => constants.%pattern_type.f79
// CHECK:STDOUT:   %T.as_type.loc249_62.1 => constants.%Int64
// CHECK:STDOUT:   %pattern_type.loc249_60 => constants.%pattern_type.7b1
// CHECK:STDOUT:   %pattern_type.loc249_66 => constants.%pattern_type.e9d
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc249_55 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc249_66 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc251 => constants.%complete_type.022
// CHECK:STDOUT:   %ModAssignWith.assoc_type => constants.%ModAssignWith.assoc_type.e0e
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.928
// CHECK:STDOUT:   %ModAssignWith.WithSelf.Op.type => constants.%ModAssignWith.WithSelf.Op.type.30a
// CHECK:STDOUT:   %.loc251_5.2 => constants.%.e8b
// CHECK:STDOUT:   %ModAssignWith.lookup_impl_witness => constants.%custom_witness.ec99b7.2
// CHECK:STDOUT:   %impl.elem0.loc251_5.2 => constants.%Int64.Op.cc70d5.7
// CHECK:STDOUT:   %specific_impl_fn.loc251_5.2 => constants.%Int64.Op.cc70d5.7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestInc(constants.%T.c06) {
// CHECK:STDOUT:   %T.loc278_13.1 => constants.%T.c06
// CHECK:STDOUT:   %T.as_type.loc278_33.1 => constants.%T.as_type.4cc
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.55fce5.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestDec(constants.%T.d9e) {
// CHECK:STDOUT:   %T.loc284_13.1 => constants.%T.d9e
// CHECK:STDOUT:   %T.as_type.loc284_33.1 => constants.%T.as_type.915
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.317946.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestNegate(constants.%Result, constants.%T.0e9) {
// CHECK:STDOUT:   %Result.loc292_9.1 => constants.%Result
// CHECK:STDOUT:   %Result.as_type.loc293_50.1 => constants.%Result.as_type
// CHECK:STDOUT:   %facet_type => constants.%facet_type.fb9
// CHECK:STDOUT:   %T.loc293_4.1 => constants.%T.0e9
// CHECK:STDOUT:   %pattern_type.loc293 => constants.%pattern_type.301
// CHECK:STDOUT:   %pattern_type.loc294_11 => constants.%pattern_type.227a41.2
// CHECK:STDOUT:   %T.as_type.loc294_24.1 => constants.%T.as_type.b01
// CHECK:STDOUT:   %pattern_type.loc294_22 => constants.%pattern_type.5e6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestInc(constants.%Inc.facet.ace) {
// CHECK:STDOUT:   %T.loc278_13.1 => constants.%Inc.facet.ace
// CHECK:STDOUT:   %T.as_type.loc278_33.1 => constants.%Int32
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.e9d
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.357
// CHECK:STDOUT:   %Inc.WithSelf.Op.type => constants.%Inc.WithSelf.Op.type.f8b
// CHECK:STDOUT:   %.loc280 => constants.%.b6c
// CHECK:STDOUT:   %Inc.lookup_impl_witness => constants.%custom_witness.ef4
// CHECK:STDOUT:   %impl.elem0.loc280_3.2 => constants.%Int32.Op.cc70d5.18
// CHECK:STDOUT:   %specific_impl_fn.loc280_3.2 => constants.%Int32.Op.cc70d5.18
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestDec(constants.%Dec.facet.be4) {
// CHECK:STDOUT:   %T.loc284_13.1 => constants.%Dec.facet.be4
// CHECK:STDOUT:   %T.as_type.loc284_33.1 => constants.%Int32
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.e9d
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.357
// CHECK:STDOUT:   %Dec.WithSelf.Op.type => constants.%Dec.WithSelf.Op.type.479
// CHECK:STDOUT:   %.loc286 => constants.%.a3b
// CHECK:STDOUT:   %Dec.lookup_impl_witness => constants.%custom_witness.7c5
// CHECK:STDOUT:   %impl.elem0.loc286_3.2 => constants.%Int32.Op.cc70d5.19
// CHECK:STDOUT:   %specific_impl_fn.loc286_3.2 => constants.%Int32.Op.cc70d5.19
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestNegate(constants.%Destroy.facet.3de, constants.%facet_value.e43) {
// CHECK:STDOUT:   %Result.loc292_9.1 => constants.%Destroy.facet.3de
// CHECK:STDOUT:   %Result.as_type.loc293_50.1 => constants.%Int64
// CHECK:STDOUT:   %facet_type => constants.%facet_type.999
// CHECK:STDOUT:   %T.loc293_4.1 => constants.%facet_value.e43
// CHECK:STDOUT:   %pattern_type.loc293 => constants.%pattern_type.04d
// CHECK:STDOUT:   %pattern_type.loc294_11 => constants.%pattern_type.7b1
// CHECK:STDOUT:   %T.as_type.loc294_24.1 => constants.%Int16
// CHECK:STDOUT:   %pattern_type.loc294_22 => constants.%pattern_type.802
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc294_11 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc294_22 => constants.%complete_type.357
// CHECK:STDOUT:   %Negate.lookup_impl_witness => constants.%custom_witness.71b
// CHECK:STDOUT:   %Negate.facet => constants.%Negate.facet.070
// CHECK:STDOUT:   %Negate.WithSelf.Op.type => constants.%Negate.WithSelf.Op.type.cde
// CHECK:STDOUT:   %.loc296_3.3 => constants.%.e93
// CHECK:STDOUT:   %impl.elem1.loc296_3.2 => constants.%Int16.Op.cc70d5.18
// CHECK:STDOUT:   %specific_impl_fn.loc296_3.3 => constants.%Int16.Op.cc70d5.18
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.2bb
// CHECK:STDOUT:   %.loc296_3.4 => constants.%.dda
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.a98c63.3
// CHECK:STDOUT:   %impl.elem0.loc296_3.2 => constants.%Int64.Op.cc70d5.1
// CHECK:STDOUT:   %specific_impl_fn.loc296_3.4 => constants.%Int64.Op.cc70d5.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestNegate(constants.%Destroy.facet.d87, constants.%facet_value.b3a) {
// CHECK:STDOUT:   %Result.loc292_9.1 => constants.%Destroy.facet.d87
// CHECK:STDOUT:   %Result.as_type.loc293_50.1 => constants.%Int32
// CHECK:STDOUT:   %facet_type => constants.%facet_type.363
// CHECK:STDOUT:   %T.loc293_4.1 => constants.%facet_value.b3a
// CHECK:STDOUT:   %pattern_type.loc293 => constants.%pattern_type.4d0
// CHECK:STDOUT:   %pattern_type.loc294_11 => constants.%pattern_type.e9d
// CHECK:STDOUT:   %T.as_type.loc294_24.1 => constants.%Int32
// CHECK:STDOUT:   %pattern_type.loc294_22 => constants.%pattern_type.e9d
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc294_11 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc294_22 => constants.%complete_type.357
// CHECK:STDOUT:   %Negate.lookup_impl_witness => constants.%custom_witness.45a
// CHECK:STDOUT:   %Negate.facet => constants.%Negate.facet.670
// CHECK:STDOUT:   %Negate.WithSelf.Op.type => constants.%Negate.WithSelf.Op.type.20b
// CHECK:STDOUT:   %.loc296_3.3 => constants.%.76e
// CHECK:STDOUT:   %impl.elem1.loc296_3.2 => constants.%Int32.Op.cc70d5.20
// CHECK:STDOUT:   %specific_impl_fn.loc296_3.3 => constants.%Int32.Op.cc70d5.20
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.a62
// CHECK:STDOUT:   %.loc296_3.4 => constants.%.e4f
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.a98c63.2
// CHECK:STDOUT:   %impl.elem0.loc296_3.2 => constants.%Int32.Op.cc70d5.1
// CHECK:STDOUT:   %specific_impl_fn.loc296_3.4 => constants.%Int32.Op.cc70d5.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
