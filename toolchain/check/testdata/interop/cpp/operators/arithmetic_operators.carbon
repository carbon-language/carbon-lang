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
// CHECK:STDOUT:   %U.866: %Destroy.type = symbolic_binding U, 1 [symbolic]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Other: type = symbolic_binding Other, 0 [symbolic]
// CHECK:STDOUT:   %AddWith.type.be3: type = facet_type <@AddWith.1, @AddWith.1(%Other)> [symbolic]
// CHECK:STDOUT:   %Self.39b: %AddWith.type.be3 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %AddWith.assoc_type.dfc: type = assoc_entity_type @AddWith.1, @AddWith.1(%Other) [symbolic]
// CHECK:STDOUT:   %AddWith.WithSelf.Op.type.fc0: type = fn_type @AddWith.WithSelf.Op, @AddWith.WithSelf(%Other, %Self.39b) [symbolic]
// CHECK:STDOUT:   %AddWith.WithSelf.Op.056: %AddWith.WithSelf.Op.type.fc0 = struct_value () [symbolic]
// CHECK:STDOUT:   %U.as_type.0fd: type = facet_access_type %U.866 [symbolic]
// CHECK:STDOUT:   %AddWith.type.68f: type = facet_type <@AddWith.1, @AddWith.1(%U.as_type.0fd)> [symbolic]
// CHECK:STDOUT:   %facet_type.5e7: type = facet_type <@Destroy & @AddWith.1, @AddWith.1(%U.as_type.0fd)> [symbolic]
// CHECK:STDOUT:   %.Self.11b: %facet_type.5e7 = symbolic_binding .Self [symbolic]
// CHECK:STDOUT:   %AddWith.assoc_type.f48: type = assoc_entity_type @AddWith.1, @AddWith.1(%U.as_type.0fd) [symbolic]
// CHECK:STDOUT:   %assoc0.5e7: %AddWith.assoc_type.f48 = assoc_entity element0, imports.%Core.import_ref.b37 [symbolic]
// CHECK:STDOUT:   %assoc1.bb6: %AddWith.assoc_type.f48 = assoc_entity element1, imports.%Core.import_ref.935 [symbolic]
// CHECK:STDOUT:   %require_complete.d10: <witness> = require_complete_type %facet_type.5e7 [symbolic]
// CHECK:STDOUT:   %.Self.as_type.8a9: type = facet_access_type %.Self.11b [symbolic]
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.2ac: <witness> = lookup_impl_witness %.Self.11b, @AddWith.1, @AddWith.1(%U.as_type.0fd) [symbolic]
// CHECK:STDOUT:   %impl.elem0.e3f: type = impl_witness_access %AddWith.lookup_impl_witness.2ac, element0 [symbolic]
// CHECK:STDOUT:   %Result.as_type: type = facet_access_type %Result [symbolic]
// CHECK:STDOUT:   %facet_type.d00: type = facet_type <@Destroy & @AddWith.1, @AddWith.1(%U.as_type.0fd) where %impl.elem0.e3f = %Result.as_type> [symbolic]
// CHECK:STDOUT:   %pattern_type.768: type = pattern_type %facet_type.d00 [symbolic]
// CHECK:STDOUT:   %T.9be: %facet_type.d00 = symbolic_binding T, 2 [symbolic]
// CHECK:STDOUT:   %pattern_type.5eec30.2: type = pattern_type %Result.as_type [symbolic]
// CHECK:STDOUT:   %T.as_type.6fa: type = facet_access_type %T.9be [symbolic]
// CHECK:STDOUT:   %pattern_type.ae2: type = pattern_type %T.as_type.6fa [symbolic]
// CHECK:STDOUT:   %pattern_type.a04: type = pattern_type %U.as_type.0fd [symbolic]
// CHECK:STDOUT:   %require_complete.672: <witness> = require_complete_type %AddWith.type.68f [symbolic]
// CHECK:STDOUT:   %assoc1.593: %AddWith.assoc_type.dfc = assoc_entity element1, imports.%Core.import_ref.31c [symbolic]
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.211: <witness> = lookup_impl_witness %T.9be, @AddWith.1, @AddWith.1(%U.as_type.0fd) [symbolic]
// CHECK:STDOUT:   %AddWith.facet.57d: %AddWith.type.68f = facet_value %T.as_type.6fa, (%AddWith.lookup_impl_witness.211) [symbolic]
// CHECK:STDOUT:   %AddWith.WithSelf.Op.type.b30a: type = fn_type @AddWith.WithSelf.Op, @AddWith.WithSelf(%U.as_type.0fd, %AddWith.facet.57d) [symbolic]
// CHECK:STDOUT:   %.09b: type = fn_type_with_self_type %AddWith.WithSelf.Op.type.b30a, %AddWith.facet.57d [symbolic]
// CHECK:STDOUT:   %impl.elem1.4bc: %.09b = impl_witness_access %AddWith.lookup_impl_witness.211, element1 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.31d: <specific function> = specific_impl_function %impl.elem1.4bc, @AddWith.WithSelf.Op(%U.as_type.0fd, %AddWith.facet.57d) [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %Result, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.12b49e.2: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Result) [symbolic]
// CHECK:STDOUT:   %.9c3: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.12b49e.2, %Result [symbolic]
// CHECK:STDOUT:   %impl.elem0.8bd: %.9c3 = impl_witness_access %Destroy.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.8dd: <specific function> = specific_impl_function %impl.elem0.8bd, @Destroy.WithSelf.Op(%Result) [symbolic]
// CHECK:STDOUT:   %SubWith.type.381: type = facet_type <@SubWith.1, @SubWith.1(%Other)> [symbolic]
// CHECK:STDOUT:   %Self.631: %SubWith.type.381 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %SubWith.assoc_type.6b2: type = assoc_entity_type @SubWith.1, @SubWith.1(%Other) [symbolic]
// CHECK:STDOUT:   %SubWith.WithSelf.Op.type.2ca: type = fn_type @SubWith.WithSelf.Op, @SubWith.WithSelf(%Other, %Self.631) [symbolic]
// CHECK:STDOUT:   %SubWith.WithSelf.Op.918: %SubWith.WithSelf.Op.type.2ca = struct_value () [symbolic]
// CHECK:STDOUT:   %SubWith.type.3e0: type = facet_type <@SubWith.1, @SubWith.1(%U.as_type.0fd)> [symbolic]
// CHECK:STDOUT:   %facet_type.b45: type = facet_type <@Destroy & @SubWith.1, @SubWith.1(%U.as_type.0fd)> [symbolic]
// CHECK:STDOUT:   %.Self.ecd: %facet_type.b45 = symbolic_binding .Self [symbolic]
// CHECK:STDOUT:   %SubWith.assoc_type.f11: type = assoc_entity_type @SubWith.1, @SubWith.1(%U.as_type.0fd) [symbolic]
// CHECK:STDOUT:   %assoc0.ccf: %SubWith.assoc_type.f11 = assoc_entity element0, imports.%Core.import_ref.bb2 [symbolic]
// CHECK:STDOUT:   %assoc1.8f3: %SubWith.assoc_type.f11 = assoc_entity element1, imports.%Core.import_ref.480 [symbolic]
// CHECK:STDOUT:   %require_complete.029: <witness> = require_complete_type %facet_type.b45 [symbolic]
// CHECK:STDOUT:   %.Self.as_type.8c5: type = facet_access_type %.Self.ecd [symbolic]
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.69c: <witness> = lookup_impl_witness %.Self.ecd, @SubWith.1, @SubWith.1(%U.as_type.0fd) [symbolic]
// CHECK:STDOUT:   %impl.elem0.35c: type = impl_witness_access %SubWith.lookup_impl_witness.69c, element0 [symbolic]
// CHECK:STDOUT:   %facet_type.de3: type = facet_type <@Destroy & @SubWith.1, @SubWith.1(%U.as_type.0fd) where %impl.elem0.35c = %Result.as_type> [symbolic]
// CHECK:STDOUT:   %pattern_type.6e8: type = pattern_type %facet_type.de3 [symbolic]
// CHECK:STDOUT:   %T.248: %facet_type.de3 = symbolic_binding T, 2 [symbolic]
// CHECK:STDOUT:   %T.as_type.3b2: type = facet_access_type %T.248 [symbolic]
// CHECK:STDOUT:   %pattern_type.eb4: type = pattern_type %T.as_type.3b2 [symbolic]
// CHECK:STDOUT:   %require_complete.f2d: <witness> = require_complete_type %SubWith.type.3e0 [symbolic]
// CHECK:STDOUT:   %assoc1.8b9: %SubWith.assoc_type.6b2 = assoc_entity element1, imports.%Core.import_ref.507 [symbolic]
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.9fd: <witness> = lookup_impl_witness %T.248, @SubWith.1, @SubWith.1(%U.as_type.0fd) [symbolic]
// CHECK:STDOUT:   %SubWith.facet.1b3: %SubWith.type.3e0 = facet_value %T.as_type.3b2, (%SubWith.lookup_impl_witness.9fd) [symbolic]
// CHECK:STDOUT:   %SubWith.WithSelf.Op.type.649: type = fn_type @SubWith.WithSelf.Op, @SubWith.WithSelf(%U.as_type.0fd, %SubWith.facet.1b3) [symbolic]
// CHECK:STDOUT:   %.e46: type = fn_type_with_self_type %SubWith.WithSelf.Op.type.649, %SubWith.facet.1b3 [symbolic]
// CHECK:STDOUT:   %impl.elem1.d3b: %.e46 = impl_witness_access %SubWith.lookup_impl_witness.9fd, element1 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.d50: <specific function> = specific_impl_function %impl.elem1.d3b, @SubWith.WithSelf.Op(%U.as_type.0fd, %SubWith.facet.1b3) [symbolic]
// CHECK:STDOUT:   %MulWith.type.9db: type = facet_type <@MulWith.1, @MulWith.1(%Other)> [symbolic]
// CHECK:STDOUT:   %Self.019: %MulWith.type.9db = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %MulWith.assoc_type.03e: type = assoc_entity_type @MulWith.1, @MulWith.1(%Other) [symbolic]
// CHECK:STDOUT:   %MulWith.WithSelf.Op.type.e29: type = fn_type @MulWith.WithSelf.Op, @MulWith.WithSelf(%Other, %Self.019) [symbolic]
// CHECK:STDOUT:   %MulWith.WithSelf.Op.45b: %MulWith.WithSelf.Op.type.e29 = struct_value () [symbolic]
// CHECK:STDOUT:   %MulWith.type.b5e: type = facet_type <@MulWith.1, @MulWith.1(%U.as_type.0fd)> [symbolic]
// CHECK:STDOUT:   %facet_type.7d9: type = facet_type <@Destroy & @MulWith.1, @MulWith.1(%U.as_type.0fd)> [symbolic]
// CHECK:STDOUT:   %.Self.e6f: %facet_type.7d9 = symbolic_binding .Self [symbolic]
// CHECK:STDOUT:   %MulWith.assoc_type.4f3: type = assoc_entity_type @MulWith.1, @MulWith.1(%U.as_type.0fd) [symbolic]
// CHECK:STDOUT:   %assoc0.0e6: %MulWith.assoc_type.4f3 = assoc_entity element0, imports.%Core.import_ref.a1a [symbolic]
// CHECK:STDOUT:   %assoc1.4a5: %MulWith.assoc_type.4f3 = assoc_entity element1, imports.%Core.import_ref.7ed [symbolic]
// CHECK:STDOUT:   %require_complete.38c: <witness> = require_complete_type %facet_type.7d9 [symbolic]
// CHECK:STDOUT:   %.Self.as_type.2c6: type = facet_access_type %.Self.e6f [symbolic]
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.4c7: <witness> = lookup_impl_witness %.Self.e6f, @MulWith.1, @MulWith.1(%U.as_type.0fd) [symbolic]
// CHECK:STDOUT:   %impl.elem0.0fa: type = impl_witness_access %MulWith.lookup_impl_witness.4c7, element0 [symbolic]
// CHECK:STDOUT:   %facet_type.779: type = facet_type <@Destroy & @MulWith.1, @MulWith.1(%U.as_type.0fd) where %impl.elem0.0fa = %Result.as_type> [symbolic]
// CHECK:STDOUT:   %pattern_type.2d4: type = pattern_type %facet_type.779 [symbolic]
// CHECK:STDOUT:   %T.ce0: %facet_type.779 = symbolic_binding T, 2 [symbolic]
// CHECK:STDOUT:   %T.as_type.380: type = facet_access_type %T.ce0 [symbolic]
// CHECK:STDOUT:   %pattern_type.4b9: type = pattern_type %T.as_type.380 [symbolic]
// CHECK:STDOUT:   %require_complete.be1: <witness> = require_complete_type %MulWith.type.b5e [symbolic]
// CHECK:STDOUT:   %assoc1.56c: %MulWith.assoc_type.03e = assoc_entity element1, imports.%Core.import_ref.f63 [symbolic]
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.69d: <witness> = lookup_impl_witness %T.ce0, @MulWith.1, @MulWith.1(%U.as_type.0fd) [symbolic]
// CHECK:STDOUT:   %MulWith.facet.181: %MulWith.type.b5e = facet_value %T.as_type.380, (%MulWith.lookup_impl_witness.69d) [symbolic]
// CHECK:STDOUT:   %MulWith.WithSelf.Op.type.ea9: type = fn_type @MulWith.WithSelf.Op, @MulWith.WithSelf(%U.as_type.0fd, %MulWith.facet.181) [symbolic]
// CHECK:STDOUT:   %.efc: type = fn_type_with_self_type %MulWith.WithSelf.Op.type.ea9, %MulWith.facet.181 [symbolic]
// CHECK:STDOUT:   %impl.elem1.002: %.efc = impl_witness_access %MulWith.lookup_impl_witness.69d, element1 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.168: <specific function> = specific_impl_function %impl.elem1.002, @MulWith.WithSelf.Op(%U.as_type.0fd, %MulWith.facet.181) [symbolic]
// CHECK:STDOUT:   %DivWith.type.43b: type = facet_type <@DivWith.1, @DivWith.1(%Other)> [symbolic]
// CHECK:STDOUT:   %Self.555: %DivWith.type.43b = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %DivWith.assoc_type.1ff: type = assoc_entity_type @DivWith.1, @DivWith.1(%Other) [symbolic]
// CHECK:STDOUT:   %DivWith.WithSelf.Op.type.388: type = fn_type @DivWith.WithSelf.Op, @DivWith.WithSelf(%Other, %Self.555) [symbolic]
// CHECK:STDOUT:   %DivWith.WithSelf.Op.0ce: %DivWith.WithSelf.Op.type.388 = struct_value () [symbolic]
// CHECK:STDOUT:   %DivWith.type.c0c: type = facet_type <@DivWith.1, @DivWith.1(%U.as_type.0fd)> [symbolic]
// CHECK:STDOUT:   %facet_type.9b5: type = facet_type <@Destroy & @DivWith.1, @DivWith.1(%U.as_type.0fd)> [symbolic]
// CHECK:STDOUT:   %.Self.3de: %facet_type.9b5 = symbolic_binding .Self [symbolic]
// CHECK:STDOUT:   %DivWith.assoc_type.cd1: type = assoc_entity_type @DivWith.1, @DivWith.1(%U.as_type.0fd) [symbolic]
// CHECK:STDOUT:   %assoc0.453: %DivWith.assoc_type.cd1 = assoc_entity element0, imports.%Core.import_ref.b8f [symbolic]
// CHECK:STDOUT:   %assoc1.5f3: %DivWith.assoc_type.cd1 = assoc_entity element1, imports.%Core.import_ref.44d [symbolic]
// CHECK:STDOUT:   %require_complete.b55: <witness> = require_complete_type %facet_type.9b5 [symbolic]
// CHECK:STDOUT:   %.Self.as_type.d38: type = facet_access_type %.Self.3de [symbolic]
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.405: <witness> = lookup_impl_witness %.Self.3de, @DivWith.1, @DivWith.1(%U.as_type.0fd) [symbolic]
// CHECK:STDOUT:   %impl.elem0.e26: type = impl_witness_access %DivWith.lookup_impl_witness.405, element0 [symbolic]
// CHECK:STDOUT:   %facet_type.c18: type = facet_type <@Destroy & @DivWith.1, @DivWith.1(%U.as_type.0fd) where %impl.elem0.e26 = %Result.as_type> [symbolic]
// CHECK:STDOUT:   %pattern_type.7f8: type = pattern_type %facet_type.c18 [symbolic]
// CHECK:STDOUT:   %T.ff8: %facet_type.c18 = symbolic_binding T, 2 [symbolic]
// CHECK:STDOUT:   %T.as_type.7d7: type = facet_access_type %T.ff8 [symbolic]
// CHECK:STDOUT:   %pattern_type.4a4: type = pattern_type %T.as_type.7d7 [symbolic]
// CHECK:STDOUT:   %require_complete.df0: <witness> = require_complete_type %DivWith.type.c0c [symbolic]
// CHECK:STDOUT:   %assoc1.71c: %DivWith.assoc_type.1ff = assoc_entity element1, imports.%Core.import_ref.4ea [symbolic]
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.62a: <witness> = lookup_impl_witness %T.ff8, @DivWith.1, @DivWith.1(%U.as_type.0fd) [symbolic]
// CHECK:STDOUT:   %DivWith.facet.08b: %DivWith.type.c0c = facet_value %T.as_type.7d7, (%DivWith.lookup_impl_witness.62a) [symbolic]
// CHECK:STDOUT:   %DivWith.WithSelf.Op.type.ed1: type = fn_type @DivWith.WithSelf.Op, @DivWith.WithSelf(%U.as_type.0fd, %DivWith.facet.08b) [symbolic]
// CHECK:STDOUT:   %.612: type = fn_type_with_self_type %DivWith.WithSelf.Op.type.ed1, %DivWith.facet.08b [symbolic]
// CHECK:STDOUT:   %impl.elem1.536: %.612 = impl_witness_access %DivWith.lookup_impl_witness.62a, element1 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.706: <specific function> = specific_impl_function %impl.elem1.536, @DivWith.WithSelf.Op(%U.as_type.0fd, %DivWith.facet.08b) [symbolic]
// CHECK:STDOUT:   %ModWith.type.92f: type = facet_type <@ModWith.1, @ModWith.1(%Other)> [symbolic]
// CHECK:STDOUT:   %Self.b83: %ModWith.type.92f = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %ModWith.assoc_type.7ff: type = assoc_entity_type @ModWith.1, @ModWith.1(%Other) [symbolic]
// CHECK:STDOUT:   %ModWith.WithSelf.Op.type.f7c: type = fn_type @ModWith.WithSelf.Op, @ModWith.WithSelf(%Other, %Self.b83) [symbolic]
// CHECK:STDOUT:   %ModWith.WithSelf.Op.a28: %ModWith.WithSelf.Op.type.f7c = struct_value () [symbolic]
// CHECK:STDOUT:   %ModWith.type.fba: type = facet_type <@ModWith.1, @ModWith.1(%U.as_type.0fd)> [symbolic]
// CHECK:STDOUT:   %facet_type.1e9: type = facet_type <@Destroy & @ModWith.1, @ModWith.1(%U.as_type.0fd)> [symbolic]
// CHECK:STDOUT:   %.Self.ade: %facet_type.1e9 = symbolic_binding .Self [symbolic]
// CHECK:STDOUT:   %ModWith.assoc_type.e42: type = assoc_entity_type @ModWith.1, @ModWith.1(%U.as_type.0fd) [symbolic]
// CHECK:STDOUT:   %assoc0.4ea: %ModWith.assoc_type.e42 = assoc_entity element0, imports.%Core.import_ref.f19 [symbolic]
// CHECK:STDOUT:   %assoc1.516: %ModWith.assoc_type.e42 = assoc_entity element1, imports.%Core.import_ref.556 [symbolic]
// CHECK:STDOUT:   %require_complete.797: <witness> = require_complete_type %facet_type.1e9 [symbolic]
// CHECK:STDOUT:   %.Self.as_type.142: type = facet_access_type %.Self.ade [symbolic]
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.2ca: <witness> = lookup_impl_witness %.Self.ade, @ModWith.1, @ModWith.1(%U.as_type.0fd) [symbolic]
// CHECK:STDOUT:   %impl.elem0.0c4: type = impl_witness_access %ModWith.lookup_impl_witness.2ca, element0 [symbolic]
// CHECK:STDOUT:   %facet_type.ae7: type = facet_type <@Destroy & @ModWith.1, @ModWith.1(%U.as_type.0fd) where %impl.elem0.0c4 = %Result.as_type> [symbolic]
// CHECK:STDOUT:   %pattern_type.d24: type = pattern_type %facet_type.ae7 [symbolic]
// CHECK:STDOUT:   %T.5f9: %facet_type.ae7 = symbolic_binding T, 2 [symbolic]
// CHECK:STDOUT:   %T.as_type.c82: type = facet_access_type %T.5f9 [symbolic]
// CHECK:STDOUT:   %pattern_type.b5f: type = pattern_type %T.as_type.c82 [symbolic]
// CHECK:STDOUT:   %require_complete.ed2: <witness> = require_complete_type %ModWith.type.fba [symbolic]
// CHECK:STDOUT:   %assoc1.e05: %ModWith.assoc_type.7ff = assoc_entity element1, imports.%Core.import_ref.f10 [symbolic]
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.e49: <witness> = lookup_impl_witness %T.5f9, @ModWith.1, @ModWith.1(%U.as_type.0fd) [symbolic]
// CHECK:STDOUT:   %ModWith.facet.6df: %ModWith.type.fba = facet_value %T.as_type.c82, (%ModWith.lookup_impl_witness.e49) [symbolic]
// CHECK:STDOUT:   %ModWith.WithSelf.Op.type.a89: type = fn_type @ModWith.WithSelf.Op, @ModWith.WithSelf(%U.as_type.0fd, %ModWith.facet.6df) [symbolic]
// CHECK:STDOUT:   %.977: type = fn_type_with_self_type %ModWith.WithSelf.Op.type.a89, %ModWith.facet.6df [symbolic]
// CHECK:STDOUT:   %impl.elem1.34f: %.977 = impl_witness_access %ModWith.lookup_impl_witness.e49, element1 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.6e5: <specific function> = specific_impl_function %impl.elem1.34f, @ModWith.WithSelf.Op(%U.as_type.0fd, %ModWith.facet.6df) [symbolic]
// CHECK:STDOUT:   %Int16: type = class_type @Int16 [concrete]
// CHECK:STDOUT:   %pattern_type.004: type = pattern_type %Int16 [concrete]
// CHECK:STDOUT:   %Int32: type = class_type @Int32 [concrete]
// CHECK:STDOUT:   %pattern_type.3b9: type = pattern_type %Int32 [concrete]
// CHECK:STDOUT:   %Int64: type = class_type @Int64 [concrete]
// CHECK:STDOUT:   %pattern_type.41c: type = pattern_type %Int64 [concrete]
// CHECK:STDOUT:   %int_16: Core.IntLiteral = int_value 16 [concrete]
// CHECK:STDOUT:   %i16: type = class_type @Int, @Int(%int_16) [concrete]
// CHECK:STDOUT:   %pattern_type.0cb: type = pattern_type %i16 [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %i16.builtin: type = int_type signed, %int_16 [concrete]
// CHECK:STDOUT:   %complete_type.2da: <witness> = complete_type_witness %i16.builtin [concrete]
// CHECK:STDOUT:   %Int16.Op.type.d5d3d3.1: type = fn_type @Int16.Op.1 [concrete]
// CHECK:STDOUT:   %Int16.Op.ecb1a5.1: %Int16.Op.type.d5d3d3.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.f83583.1: <witness> = custom_witness (%Int16.Op.ecb1a5.1), @Destroy [concrete]
// CHECK:STDOUT:   %Destroy.facet.738: %Destroy.type = facet_value %Int16, (%custom_witness.f83583.1) [concrete]
// CHECK:STDOUT:   %facet_type.aa6: type = facet_type <@Destroy & @AddWith.1, @AddWith.1(%Int16)> [concrete]
// CHECK:STDOUT:   %.Self.cf1: %facet_type.aa6 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.d17: <witness> = lookup_impl_witness %.Self.cf1, @AddWith.1, @AddWith.1(%Int16) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.9d9: type = impl_witness_access %AddWith.lookup_impl_witness.d17, element0 [symbolic_self]
// CHECK:STDOUT:   %facet_type.f01: type = facet_type <@Destroy & @AddWith.1, @AddWith.1(%Int16) where %impl.elem0.9d9 = %Int16> [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.60afa5.2: type = fn_type @Destroy.Op.loc176_34.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.b63e49.2: %Destroy.Op.type.60afa5.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.275260.2: <witness> = custom_witness (%Destroy.Op.b63e49.2), @Destroy [concrete]
// CHECK:STDOUT:   %AddWith.type.116: type = facet_type <@AddWith.1, @AddWith.1(%Int16)> [concrete]
// CHECK:STDOUT:   %AddWith.assoc_type.dda: type = assoc_entity_type @AddWith.1, @AddWith.1(%Int16) [concrete]
// CHECK:STDOUT:   %assoc0.c76: %AddWith.assoc_type.dda = assoc_entity element0, imports.%Core.import_ref.b37 [concrete]
// CHECK:STDOUT:   %assoc1.83e: %AddWith.assoc_type.dda = assoc_entity element1, imports.%Core.import_ref.935 [concrete]
// CHECK:STDOUT:   %Op.type.0b8387.1: type = fn_type @Op.1 [concrete]
// CHECK:STDOUT:   %Op.b9c47d.1: %Op.type.0b8387.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.facet.6e0: %Destroy.type = facet_value %i16, (%custom_witness.275260.2) [concrete]
// CHECK:STDOUT:   %custom_witness.91a: <witness> = custom_witness (%Int16, %Op.b9c47d.1), @AddWith.1, @AddWith.1(%Int16) [concrete]
// CHECK:STDOUT:   %facet_value.70e: %facet_type.f01 = facet_value %i16, (%custom_witness.275260.2, %custom_witness.91a) [concrete]
// CHECK:STDOUT:   %complete_type.a93: <witness> = complete_type_witness %facet_type.aa6 [concrete]
// CHECK:STDOUT:   %.Self.as_type.70cc: type = facet_access_type %.Self.cf1 [symbolic_self]
// CHECK:STDOUT:   %pattern_type.616: type = pattern_type %facet_type.f01 [concrete]
// CHECK:STDOUT:   %facet_type.4d6: type = facet_type <@Destroy & @SubWith.1, @SubWith.1(%Int16)> [concrete]
// CHECK:STDOUT:   %.Self.c3b: %facet_type.4d6 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.72f: <witness> = lookup_impl_witness %.Self.c3b, @SubWith.1, @SubWith.1(%Int16) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.b83: type = impl_witness_access %SubWith.lookup_impl_witness.72f, element0 [symbolic_self]
// CHECK:STDOUT:   %facet_type.d9b: type = facet_type <@Destroy & @SubWith.1, @SubWith.1(%Int16) where %impl.elem0.b83 = %Int16> [concrete]
// CHECK:STDOUT:   %SubWith.type.49a: type = facet_type <@SubWith.1, @SubWith.1(%Int16)> [concrete]
// CHECK:STDOUT:   %SubWith.assoc_type.445: type = assoc_entity_type @SubWith.1, @SubWith.1(%Int16) [concrete]
// CHECK:STDOUT:   %assoc0.c71: %SubWith.assoc_type.445 = assoc_entity element0, imports.%Core.import_ref.bb2 [concrete]
// CHECK:STDOUT:   %assoc1.a3f: %SubWith.assoc_type.445 = assoc_entity element1, imports.%Core.import_ref.480 [concrete]
// CHECK:STDOUT:   %Op.type.0b8387.2: type = fn_type @Op.2 [concrete]
// CHECK:STDOUT:   %Op.b9c47d.2: %Op.type.0b8387.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.7ee: <witness> = custom_witness (%Int16, %Op.b9c47d.2), @SubWith.1, @SubWith.1(%Int16) [concrete]
// CHECK:STDOUT:   %facet_value.677: %facet_type.d9b = facet_value %i16, (%custom_witness.275260.2, %custom_witness.7ee) [concrete]
// CHECK:STDOUT:   %complete_type.4fe: <witness> = complete_type_witness %facet_type.4d6 [concrete]
// CHECK:STDOUT:   %.Self.as_type.42f: type = facet_access_type %.Self.c3b [symbolic_self]
// CHECK:STDOUT:   %pattern_type.ff0: type = pattern_type %facet_type.d9b [concrete]
// CHECK:STDOUT:   %facet_type.624: type = facet_type <@Destroy & @MulWith.1, @MulWith.1(%Int16)> [concrete]
// CHECK:STDOUT:   %.Self.164: %facet_type.624 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.588: <witness> = lookup_impl_witness %.Self.164, @MulWith.1, @MulWith.1(%Int16) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.66a: type = impl_witness_access %MulWith.lookup_impl_witness.588, element0 [symbolic_self]
// CHECK:STDOUT:   %facet_type.cc5: type = facet_type <@Destroy & @MulWith.1, @MulWith.1(%Int16) where %impl.elem0.66a = %Int16> [concrete]
// CHECK:STDOUT:   %MulWith.type.779: type = facet_type <@MulWith.1, @MulWith.1(%Int16)> [concrete]
// CHECK:STDOUT:   %MulWith.assoc_type.3a9: type = assoc_entity_type @MulWith.1, @MulWith.1(%Int16) [concrete]
// CHECK:STDOUT:   %assoc0.97a: %MulWith.assoc_type.3a9 = assoc_entity element0, imports.%Core.import_ref.a1a [concrete]
// CHECK:STDOUT:   %assoc1.a46: %MulWith.assoc_type.3a9 = assoc_entity element1, imports.%Core.import_ref.7ed [concrete]
// CHECK:STDOUT:   %Op.type.0b8387.3: type = fn_type @Op.3 [concrete]
// CHECK:STDOUT:   %Op.b9c47d.3: %Op.type.0b8387.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.448: <witness> = custom_witness (%Int16, %Op.b9c47d.3), @MulWith.1, @MulWith.1(%Int16) [concrete]
// CHECK:STDOUT:   %facet_value.661: %facet_type.cc5 = facet_value %i16, (%custom_witness.275260.2, %custom_witness.448) [concrete]
// CHECK:STDOUT:   %complete_type.ec6: <witness> = complete_type_witness %facet_type.624 [concrete]
// CHECK:STDOUT:   %.Self.as_type.49c: type = facet_access_type %.Self.164 [symbolic_self]
// CHECK:STDOUT:   %pattern_type.da6: type = pattern_type %facet_type.cc5 [concrete]
// CHECK:STDOUT:   %facet_type.b9b: type = facet_type <@Destroy & @DivWith.1, @DivWith.1(%Int16)> [concrete]
// CHECK:STDOUT:   %.Self.7ba: %facet_type.b9b = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.836: <witness> = lookup_impl_witness %.Self.7ba, @DivWith.1, @DivWith.1(%Int16) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.6cb: type = impl_witness_access %DivWith.lookup_impl_witness.836, element0 [symbolic_self]
// CHECK:STDOUT:   %facet_type.1b5: type = facet_type <@Destroy & @DivWith.1, @DivWith.1(%Int16) where %impl.elem0.6cb = %Int16> [concrete]
// CHECK:STDOUT:   %DivWith.type.35b: type = facet_type <@DivWith.1, @DivWith.1(%Int16)> [concrete]
// CHECK:STDOUT:   %DivWith.assoc_type.6ae: type = assoc_entity_type @DivWith.1, @DivWith.1(%Int16) [concrete]
// CHECK:STDOUT:   %assoc0.1bc: %DivWith.assoc_type.6ae = assoc_entity element0, imports.%Core.import_ref.b8f [concrete]
// CHECK:STDOUT:   %assoc1.5a7: %DivWith.assoc_type.6ae = assoc_entity element1, imports.%Core.import_ref.44d [concrete]
// CHECK:STDOUT:   %Op.type.0b8387.4: type = fn_type @Op.4 [concrete]
// CHECK:STDOUT:   %Op.b9c47d.4: %Op.type.0b8387.4 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.3be: <witness> = custom_witness (%Int16, %Op.b9c47d.4), @DivWith.1, @DivWith.1(%Int16) [concrete]
// CHECK:STDOUT:   %facet_value.cf3: %facet_type.1b5 = facet_value %i16, (%custom_witness.275260.2, %custom_witness.3be) [concrete]
// CHECK:STDOUT:   %complete_type.7df: <witness> = complete_type_witness %facet_type.b9b [concrete]
// CHECK:STDOUT:   %.Self.as_type.31a: type = facet_access_type %.Self.7ba [symbolic_self]
// CHECK:STDOUT:   %pattern_type.ec4: type = pattern_type %facet_type.1b5 [concrete]
// CHECK:STDOUT:   %facet_type.ca3: type = facet_type <@Destroy & @ModWith.1, @ModWith.1(%Int16)> [concrete]
// CHECK:STDOUT:   %.Self.cfe: %facet_type.ca3 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.305: <witness> = lookup_impl_witness %.Self.cfe, @ModWith.1, @ModWith.1(%Int16) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.2b9: type = impl_witness_access %ModWith.lookup_impl_witness.305, element0 [symbolic_self]
// CHECK:STDOUT:   %facet_type.9a8: type = facet_type <@Destroy & @ModWith.1, @ModWith.1(%Int16) where %impl.elem0.2b9 = %Int16> [concrete]
// CHECK:STDOUT:   %ModWith.type.3ef: type = facet_type <@ModWith.1, @ModWith.1(%Int16)> [concrete]
// CHECK:STDOUT:   %ModWith.assoc_type.744: type = assoc_entity_type @ModWith.1, @ModWith.1(%Int16) [concrete]
// CHECK:STDOUT:   %assoc0.727: %ModWith.assoc_type.744 = assoc_entity element0, imports.%Core.import_ref.f19 [concrete]
// CHECK:STDOUT:   %assoc1.ba4: %ModWith.assoc_type.744 = assoc_entity element1, imports.%Core.import_ref.556 [concrete]
// CHECK:STDOUT:   %Op.type.0b8387.5: type = fn_type @Op.5 [concrete]
// CHECK:STDOUT:   %Op.b9c47d.5: %Op.type.0b8387.5 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.dcf: <witness> = custom_witness (%Int16, %Op.b9c47d.5), @ModWith.1, @ModWith.1(%Int16) [concrete]
// CHECK:STDOUT:   %facet_value.ef3: %facet_type.9a8 = facet_value %i16, (%custom_witness.275260.2, %custom_witness.dcf) [concrete]
// CHECK:STDOUT:   %complete_type.1cf: <witness> = complete_type_witness %facet_type.ca3 [concrete]
// CHECK:STDOUT:   %.Self.as_type.da7: type = facet_access_type %.Self.cfe [symbolic_self]
// CHECK:STDOUT:   %pattern_type.a3b: type = pattern_type %facet_type.9a8 [concrete]
// CHECK:STDOUT:   %facet_type.074: type = facet_type <@Destroy & @AddWith.1, @AddWith.1(%i16)> [concrete]
// CHECK:STDOUT:   %.Self.a3f: %facet_type.074 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.e89: <witness> = lookup_impl_witness %.Self.a3f, @AddWith.1, @AddWith.1(%i16) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.4db: type = impl_witness_access %AddWith.lookup_impl_witness.e89, element0 [symbolic_self]
// CHECK:STDOUT:   %facet_type.854: type = facet_type <@Destroy & @AddWith.1, @AddWith.1(%i16) where %impl.elem0.4db = %Int16> [concrete]
// CHECK:STDOUT:   %AddWith.type.a6e: type = facet_type <@AddWith.1, @AddWith.1(%i16)> [concrete]
// CHECK:STDOUT:   %AddWith.assoc_type.6b4: type = assoc_entity_type @AddWith.1, @AddWith.1(%i16) [concrete]
// CHECK:STDOUT:   %assoc0.8ca: %AddWith.assoc_type.6b4 = assoc_entity element0, imports.%Core.import_ref.b37 [concrete]
// CHECK:STDOUT:   %assoc1.864: %AddWith.assoc_type.6b4 = assoc_entity element1, imports.%Core.import_ref.935 [concrete]
// CHECK:STDOUT:   %Int16.Op.type.d5d3d3.3: type = fn_type @Int16.Op.3 [concrete]
// CHECK:STDOUT:   %Int16.Op.ecb1a5.3: %Int16.Op.type.d5d3d3.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.0db: <witness> = custom_witness (%Int16, %Int16.Op.ecb1a5.3), @AddWith.1, @AddWith.1(%i16) [concrete]
// CHECK:STDOUT:   %facet_value.ea8: %facet_type.854 = facet_value %Int16, (%custom_witness.f83583.1, %custom_witness.0db) [concrete]
// CHECK:STDOUT:   %complete_type.fbb: <witness> = complete_type_witness %facet_type.074 [concrete]
// CHECK:STDOUT:   %.Self.as_type.1c2: type = facet_access_type %.Self.a3f [symbolic_self]
// CHECK:STDOUT:   %pattern_type.7902: type = pattern_type %facet_type.854 [concrete]
// CHECK:STDOUT:   %facet_type.352: type = facet_type <@Destroy & @SubWith.1, @SubWith.1(%i16)> [concrete]
// CHECK:STDOUT:   %.Self.982: %facet_type.352 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.7e3: <witness> = lookup_impl_witness %.Self.982, @SubWith.1, @SubWith.1(%i16) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.aa0: type = impl_witness_access %SubWith.lookup_impl_witness.7e3, element0 [symbolic_self]
// CHECK:STDOUT:   %facet_type.712: type = facet_type <@Destroy & @SubWith.1, @SubWith.1(%i16) where %impl.elem0.aa0 = %Int16> [concrete]
// CHECK:STDOUT:   %SubWith.type.2f5: type = facet_type <@SubWith.1, @SubWith.1(%i16)> [concrete]
// CHECK:STDOUT:   %SubWith.assoc_type.70d: type = assoc_entity_type @SubWith.1, @SubWith.1(%i16) [concrete]
// CHECK:STDOUT:   %assoc0.367: %SubWith.assoc_type.70d = assoc_entity element0, imports.%Core.import_ref.bb2 [concrete]
// CHECK:STDOUT:   %assoc1.6ca: %SubWith.assoc_type.70d = assoc_entity element1, imports.%Core.import_ref.480 [concrete]
// CHECK:STDOUT:   %Int16.Op.type.d5d3d3.4: type = fn_type @Int16.Op.4 [concrete]
// CHECK:STDOUT:   %Int16.Op.ecb1a5.4: %Int16.Op.type.d5d3d3.4 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.2f4: <witness> = custom_witness (%Int16, %Int16.Op.ecb1a5.4), @SubWith.1, @SubWith.1(%i16) [concrete]
// CHECK:STDOUT:   %facet_value.62c: %facet_type.712 = facet_value %Int16, (%custom_witness.f83583.1, %custom_witness.2f4) [concrete]
// CHECK:STDOUT:   %complete_type.757: <witness> = complete_type_witness %facet_type.352 [concrete]
// CHECK:STDOUT:   %.Self.as_type.311: type = facet_access_type %.Self.982 [symbolic_self]
// CHECK:STDOUT:   %pattern_type.f30: type = pattern_type %facet_type.712 [concrete]
// CHECK:STDOUT:   %facet_type.bff: type = facet_type <@Destroy & @MulWith.1, @MulWith.1(%i16)> [concrete]
// CHECK:STDOUT:   %.Self.92d: %facet_type.bff = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.fd5: <witness> = lookup_impl_witness %.Self.92d, @MulWith.1, @MulWith.1(%i16) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.004: type = impl_witness_access %MulWith.lookup_impl_witness.fd5, element0 [symbolic_self]
// CHECK:STDOUT:   %facet_type.e32: type = facet_type <@Destroy & @MulWith.1, @MulWith.1(%i16) where %impl.elem0.004 = %Int16> [concrete]
// CHECK:STDOUT:   %MulWith.type.278: type = facet_type <@MulWith.1, @MulWith.1(%i16)> [concrete]
// CHECK:STDOUT:   %MulWith.assoc_type.8cf: type = assoc_entity_type @MulWith.1, @MulWith.1(%i16) [concrete]
// CHECK:STDOUT:   %assoc0.de2: %MulWith.assoc_type.8cf = assoc_entity element0, imports.%Core.import_ref.a1a [concrete]
// CHECK:STDOUT:   %assoc1.c99: %MulWith.assoc_type.8cf = assoc_entity element1, imports.%Core.import_ref.7ed [concrete]
// CHECK:STDOUT:   %Int16.Op.type.d5d3d3.5: type = fn_type @Int16.Op.5 [concrete]
// CHECK:STDOUT:   %Int16.Op.ecb1a5.5: %Int16.Op.type.d5d3d3.5 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.d98: <witness> = custom_witness (%Int16, %Int16.Op.ecb1a5.5), @MulWith.1, @MulWith.1(%i16) [concrete]
// CHECK:STDOUT:   %facet_value.ce7: %facet_type.e32 = facet_value %Int16, (%custom_witness.f83583.1, %custom_witness.d98) [concrete]
// CHECK:STDOUT:   %complete_type.4fc: <witness> = complete_type_witness %facet_type.bff [concrete]
// CHECK:STDOUT:   %.Self.as_type.f34: type = facet_access_type %.Self.92d [symbolic_self]
// CHECK:STDOUT:   %pattern_type.cfd: type = pattern_type %facet_type.e32 [concrete]
// CHECK:STDOUT:   %facet_type.223: type = facet_type <@Destroy & @DivWith.1, @DivWith.1(%i16)> [concrete]
// CHECK:STDOUT:   %.Self.867: %facet_type.223 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.7f7: <witness> = lookup_impl_witness %.Self.867, @DivWith.1, @DivWith.1(%i16) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.e0d: type = impl_witness_access %DivWith.lookup_impl_witness.7f7, element0 [symbolic_self]
// CHECK:STDOUT:   %facet_type.fd7: type = facet_type <@Destroy & @DivWith.1, @DivWith.1(%i16) where %impl.elem0.e0d = %Int16> [concrete]
// CHECK:STDOUT:   %DivWith.type.9b1: type = facet_type <@DivWith.1, @DivWith.1(%i16)> [concrete]
// CHECK:STDOUT:   %DivWith.assoc_type.746: type = assoc_entity_type @DivWith.1, @DivWith.1(%i16) [concrete]
// CHECK:STDOUT:   %assoc0.13d: %DivWith.assoc_type.746 = assoc_entity element0, imports.%Core.import_ref.b8f [concrete]
// CHECK:STDOUT:   %assoc1.84d: %DivWith.assoc_type.746 = assoc_entity element1, imports.%Core.import_ref.44d [concrete]
// CHECK:STDOUT:   %Int16.Op.type.d5d3d3.6: type = fn_type @Int16.Op.6 [concrete]
// CHECK:STDOUT:   %Int16.Op.ecb1a5.6: %Int16.Op.type.d5d3d3.6 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.006: <witness> = custom_witness (%Int16, %Int16.Op.ecb1a5.6), @DivWith.1, @DivWith.1(%i16) [concrete]
// CHECK:STDOUT:   %facet_value.1c8: %facet_type.fd7 = facet_value %Int16, (%custom_witness.f83583.1, %custom_witness.006) [concrete]
// CHECK:STDOUT:   %complete_type.ddd: <witness> = complete_type_witness %facet_type.223 [concrete]
// CHECK:STDOUT:   %.Self.as_type.3fc: type = facet_access_type %.Self.867 [symbolic_self]
// CHECK:STDOUT:   %pattern_type.5f9: type = pattern_type %facet_type.fd7 [concrete]
// CHECK:STDOUT:   %facet_type.c0c: type = facet_type <@Destroy & @ModWith.1, @ModWith.1(%i16)> [concrete]
// CHECK:STDOUT:   %.Self.762: %facet_type.c0c = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.cfe: <witness> = lookup_impl_witness %.Self.762, @ModWith.1, @ModWith.1(%i16) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.4d9: type = impl_witness_access %ModWith.lookup_impl_witness.cfe, element0 [symbolic_self]
// CHECK:STDOUT:   %facet_type.edf: type = facet_type <@Destroy & @ModWith.1, @ModWith.1(%i16) where %impl.elem0.4d9 = %Int16> [concrete]
// CHECK:STDOUT:   %ModWith.type.59b: type = facet_type <@ModWith.1, @ModWith.1(%i16)> [concrete]
// CHECK:STDOUT:   %ModWith.assoc_type.3a6: type = assoc_entity_type @ModWith.1, @ModWith.1(%i16) [concrete]
// CHECK:STDOUT:   %assoc0.2b8: %ModWith.assoc_type.3a6 = assoc_entity element0, imports.%Core.import_ref.f19 [concrete]
// CHECK:STDOUT:   %assoc1.a21: %ModWith.assoc_type.3a6 = assoc_entity element1, imports.%Core.import_ref.556 [concrete]
// CHECK:STDOUT:   %Int16.Op.type.d5d3d3.7: type = fn_type @Int16.Op.7 [concrete]
// CHECK:STDOUT:   %Int16.Op.ecb1a5.7: %Int16.Op.type.d5d3d3.7 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.e08: <witness> = custom_witness (%Int16, %Int16.Op.ecb1a5.7), @ModWith.1, @ModWith.1(%i16) [concrete]
// CHECK:STDOUT:   %facet_value.322: %facet_type.edf = facet_value %Int16, (%custom_witness.f83583.1, %custom_witness.e08) [concrete]
// CHECK:STDOUT:   %complete_type.e02: <witness> = complete_type_witness %facet_type.c0c [concrete]
// CHECK:STDOUT:   %.Self.as_type.68f: type = facet_access_type %.Self.762 [symbolic_self]
// CHECK:STDOUT:   %pattern_type.83e: type = pattern_type %facet_type.edf [concrete]
// CHECK:STDOUT:   %Int32.Op.type.d5d3d3.1: type = fn_type @Int32.Op.1 [concrete]
// CHECK:STDOUT:   %Int32.Op.ecb1a5.1: %Int32.Op.type.d5d3d3.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.f83583.2: <witness> = custom_witness (%Int32.Op.ecb1a5.1), @Destroy [concrete]
// CHECK:STDOUT:   %Destroy.facet.700: %Destroy.type = facet_value %Int32, (%custom_witness.f83583.2) [concrete]
// CHECK:STDOUT:   %facet_type.f56: type = facet_type <@Destroy & @AddWith.1, @AddWith.1(%Int32)> [concrete]
// CHECK:STDOUT:   %.Self.977: %facet_type.f56 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.883: <witness> = lookup_impl_witness %.Self.977, @AddWith.1, @AddWith.1(%Int32) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.2e9: type = impl_witness_access %AddWith.lookup_impl_witness.883, element0 [symbolic_self]
// CHECK:STDOUT:   %facet_type.36e: type = facet_type <@Destroy & @AddWith.1, @AddWith.1(%Int32) where %impl.elem0.2e9 = %Int32> [concrete]
// CHECK:STDOUT:   %AddWith.type.c13: type = facet_type <@AddWith.1, @AddWith.1(%Int32)> [concrete]
// CHECK:STDOUT:   %AddWith.assoc_type.f6f: type = assoc_entity_type @AddWith.1, @AddWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %assoc0.41e: %AddWith.assoc_type.f6f = assoc_entity element0, imports.%Core.import_ref.b37 [concrete]
// CHECK:STDOUT:   %assoc1.519: %AddWith.assoc_type.f6f = assoc_entity element1, imports.%Core.import_ref.935 [concrete]
// CHECK:STDOUT:   %Int16.Op.type.d5d3d3.8: type = fn_type @Int16.Op.8 [concrete]
// CHECK:STDOUT:   %Int16.Op.ecb1a5.8: %Int16.Op.type.d5d3d3.8 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.27d802.1: <witness> = custom_witness (%Int32, %Int16.Op.ecb1a5.8), @AddWith.1, @AddWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %facet_value.228: %facet_type.36e = facet_value %Int16, (%custom_witness.f83583.1, %custom_witness.27d802.1) [concrete]
// CHECK:STDOUT:   %complete_type.39f: <witness> = complete_type_witness %facet_type.f56 [concrete]
// CHECK:STDOUT:   %.Self.as_type.70c2: type = facet_access_type %.Self.977 [symbolic_self]
// CHECK:STDOUT:   %pattern_type.204: type = pattern_type %facet_type.36e [concrete]
// CHECK:STDOUT:   %facet_type.eb0: type = facet_type <@Destroy & @SubWith.1, @SubWith.1(%Int32)> [concrete]
// CHECK:STDOUT:   %.Self.604: %facet_type.eb0 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.8be: <witness> = lookup_impl_witness %.Self.604, @SubWith.1, @SubWith.1(%Int32) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.7af: type = impl_witness_access %SubWith.lookup_impl_witness.8be, element0 [symbolic_self]
// CHECK:STDOUT:   %facet_type.dfd: type = facet_type <@Destroy & @SubWith.1, @SubWith.1(%Int32) where %impl.elem0.7af = %Int32> [concrete]
// CHECK:STDOUT:   %SubWith.type.554: type = facet_type <@SubWith.1, @SubWith.1(%Int32)> [concrete]
// CHECK:STDOUT:   %SubWith.assoc_type.6db: type = assoc_entity_type @SubWith.1, @SubWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %assoc0.053: %SubWith.assoc_type.6db = assoc_entity element0, imports.%Core.import_ref.bb2 [concrete]
// CHECK:STDOUT:   %assoc1.3e8: %SubWith.assoc_type.6db = assoc_entity element1, imports.%Core.import_ref.480 [concrete]
// CHECK:STDOUT:   %Int16.Op.type.d5d3d3.9: type = fn_type @Int16.Op.9 [concrete]
// CHECK:STDOUT:   %Int16.Op.ecb1a5.9: %Int16.Op.type.d5d3d3.9 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.f4a1b8.1: <witness> = custom_witness (%Int32, %Int16.Op.ecb1a5.9), @SubWith.1, @SubWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %facet_value.f72: %facet_type.dfd = facet_value %Int16, (%custom_witness.f83583.1, %custom_witness.f4a1b8.1) [concrete]
// CHECK:STDOUT:   %complete_type.872: <witness> = complete_type_witness %facet_type.eb0 [concrete]
// CHECK:STDOUT:   %.Self.as_type.d34: type = facet_access_type %.Self.604 [symbolic_self]
// CHECK:STDOUT:   %pattern_type.b5c: type = pattern_type %facet_type.dfd [concrete]
// CHECK:STDOUT:   %facet_type.87a: type = facet_type <@Destroy & @MulWith.1, @MulWith.1(%Int32)> [concrete]
// CHECK:STDOUT:   %.Self.485: %facet_type.87a = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.99b: <witness> = lookup_impl_witness %.Self.485, @MulWith.1, @MulWith.1(%Int32) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.a71: type = impl_witness_access %MulWith.lookup_impl_witness.99b, element0 [symbolic_self]
// CHECK:STDOUT:   %facet_type.4d5: type = facet_type <@Destroy & @MulWith.1, @MulWith.1(%Int32) where %impl.elem0.a71 = %Int32> [concrete]
// CHECK:STDOUT:   %MulWith.type.d6e: type = facet_type <@MulWith.1, @MulWith.1(%Int32)> [concrete]
// CHECK:STDOUT:   %MulWith.assoc_type.e4f: type = assoc_entity_type @MulWith.1, @MulWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %assoc0.879: %MulWith.assoc_type.e4f = assoc_entity element0, imports.%Core.import_ref.a1a [concrete]
// CHECK:STDOUT:   %assoc1.aaa: %MulWith.assoc_type.e4f = assoc_entity element1, imports.%Core.import_ref.7ed [concrete]
// CHECK:STDOUT:   %Int16.Op.type.d5d3d3.10: type = fn_type @Int16.Op.10 [concrete]
// CHECK:STDOUT:   %Int16.Op.ecb1a5.10: %Int16.Op.type.d5d3d3.10 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.cbf765.1: <witness> = custom_witness (%Int32, %Int16.Op.ecb1a5.10), @MulWith.1, @MulWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %facet_value.3ab: %facet_type.4d5 = facet_value %Int16, (%custom_witness.f83583.1, %custom_witness.cbf765.1) [concrete]
// CHECK:STDOUT:   %complete_type.e5c: <witness> = complete_type_witness %facet_type.87a [concrete]
// CHECK:STDOUT:   %.Self.as_type.963: type = facet_access_type %.Self.485 [symbolic_self]
// CHECK:STDOUT:   %pattern_type.459: type = pattern_type %facet_type.4d5 [concrete]
// CHECK:STDOUT:   %facet_type.81b: type = facet_type <@Destroy & @DivWith.1, @DivWith.1(%Int32)> [concrete]
// CHECK:STDOUT:   %.Self.ec5: %facet_type.81b = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.073: <witness> = lookup_impl_witness %.Self.ec5, @DivWith.1, @DivWith.1(%Int32) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.920: type = impl_witness_access %DivWith.lookup_impl_witness.073, element0 [symbolic_self]
// CHECK:STDOUT:   %facet_type.6bd: type = facet_type <@Destroy & @DivWith.1, @DivWith.1(%Int32) where %impl.elem0.920 = %Int32> [concrete]
// CHECK:STDOUT:   %DivWith.type.0f5: type = facet_type <@DivWith.1, @DivWith.1(%Int32)> [concrete]
// CHECK:STDOUT:   %DivWith.assoc_type.205: type = assoc_entity_type @DivWith.1, @DivWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %assoc0.6dd: %DivWith.assoc_type.205 = assoc_entity element0, imports.%Core.import_ref.b8f [concrete]
// CHECK:STDOUT:   %assoc1.87f: %DivWith.assoc_type.205 = assoc_entity element1, imports.%Core.import_ref.44d [concrete]
// CHECK:STDOUT:   %Int16.Op.type.d5d3d3.11: type = fn_type @Int16.Op.11 [concrete]
// CHECK:STDOUT:   %Int16.Op.ecb1a5.11: %Int16.Op.type.d5d3d3.11 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.ec9d25.1: <witness> = custom_witness (%Int32, %Int16.Op.ecb1a5.11), @DivWith.1, @DivWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %facet_value.697: %facet_type.6bd = facet_value %Int16, (%custom_witness.f83583.1, %custom_witness.ec9d25.1) [concrete]
// CHECK:STDOUT:   %complete_type.a26: <witness> = complete_type_witness %facet_type.81b [concrete]
// CHECK:STDOUT:   %.Self.as_type.0ba: type = facet_access_type %.Self.ec5 [symbolic_self]
// CHECK:STDOUT:   %pattern_type.790c: type = pattern_type %facet_type.6bd [concrete]
// CHECK:STDOUT:   %facet_type.fed: type = facet_type <@Destroy & @ModWith.1, @ModWith.1(%Int32)> [concrete]
// CHECK:STDOUT:   %.Self.6f8: %facet_type.fed = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.ba6: <witness> = lookup_impl_witness %.Self.6f8, @ModWith.1, @ModWith.1(%Int32) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.1ad: type = impl_witness_access %ModWith.lookup_impl_witness.ba6, element0 [symbolic_self]
// CHECK:STDOUT:   %facet_type.8ee: type = facet_type <@Destroy & @ModWith.1, @ModWith.1(%Int32) where %impl.elem0.1ad = %Int32> [concrete]
// CHECK:STDOUT:   %ModWith.type.e59: type = facet_type <@ModWith.1, @ModWith.1(%Int32)> [concrete]
// CHECK:STDOUT:   %ModWith.assoc_type.e6b: type = assoc_entity_type @ModWith.1, @ModWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %assoc0.bf2: %ModWith.assoc_type.e6b = assoc_entity element0, imports.%Core.import_ref.f19 [concrete]
// CHECK:STDOUT:   %assoc1.adf: %ModWith.assoc_type.e6b = assoc_entity element1, imports.%Core.import_ref.556 [concrete]
// CHECK:STDOUT:   %Int16.Op.type.d5d3d3.12: type = fn_type @Int16.Op.12 [concrete]
// CHECK:STDOUT:   %Int16.Op.ecb1a5.12: %Int16.Op.type.d5d3d3.12 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.998647.1: <witness> = custom_witness (%Int32, %Int16.Op.ecb1a5.12), @ModWith.1, @ModWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %facet_value.7c5: %facet_type.8ee = facet_value %Int16, (%custom_witness.f83583.1, %custom_witness.998647.1) [concrete]
// CHECK:STDOUT:   %complete_type.9d2: <witness> = complete_type_witness %facet_type.fed [concrete]
// CHECK:STDOUT:   %.Self.as_type.0bc: type = facet_access_type %.Self.6f8 [symbolic_self]
// CHECK:STDOUT:   %pattern_type.239: type = pattern_type %facet_type.8ee [concrete]
// CHECK:STDOUT:   %facet_type.2f8: type = facet_type <@Destroy & @AddWith.1, @AddWith.1(%Int16) where %impl.elem0.9d9 = %Int32> [concrete]
// CHECK:STDOUT:   %Int32.Op.type.d5d3d3.2: type = fn_type @Int32.Op.2 [concrete]
// CHECK:STDOUT:   %Int32.Op.ecb1a5.2: %Int32.Op.type.d5d3d3.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.6f6: <witness> = custom_witness (%Int32, %Int32.Op.ecb1a5.2), @AddWith.1, @AddWith.1(%Int16) [concrete]
// CHECK:STDOUT:   %facet_value.2f2: %facet_type.2f8 = facet_value %Int32, (%custom_witness.f83583.2, %custom_witness.6f6) [concrete]
// CHECK:STDOUT:   %pattern_type.8af: type = pattern_type %facet_type.2f8 [concrete]
// CHECK:STDOUT:   %facet_type.e67: type = facet_type <@Destroy & @SubWith.1, @SubWith.1(%Int16) where %impl.elem0.b83 = %Int32> [concrete]
// CHECK:STDOUT:   %Int32.Op.type.d5d3d3.3: type = fn_type @Int32.Op.3 [concrete]
// CHECK:STDOUT:   %Int32.Op.ecb1a5.3: %Int32.Op.type.d5d3d3.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.97b: <witness> = custom_witness (%Int32, %Int32.Op.ecb1a5.3), @SubWith.1, @SubWith.1(%Int16) [concrete]
// CHECK:STDOUT:   %facet_value.c1a: %facet_type.e67 = facet_value %Int32, (%custom_witness.f83583.2, %custom_witness.97b) [concrete]
// CHECK:STDOUT:   %pattern_type.f19: type = pattern_type %facet_type.e67 [concrete]
// CHECK:STDOUT:   %facet_type.4ee: type = facet_type <@Destroy & @MulWith.1, @MulWith.1(%Int16) where %impl.elem0.66a = %Int32> [concrete]
// CHECK:STDOUT:   %Int32.Op.type.d5d3d3.4: type = fn_type @Int32.Op.4 [concrete]
// CHECK:STDOUT:   %Int32.Op.ecb1a5.4: %Int32.Op.type.d5d3d3.4 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.ac6: <witness> = custom_witness (%Int32, %Int32.Op.ecb1a5.4), @MulWith.1, @MulWith.1(%Int16) [concrete]
// CHECK:STDOUT:   %facet_value.46e: %facet_type.4ee = facet_value %Int32, (%custom_witness.f83583.2, %custom_witness.ac6) [concrete]
// CHECK:STDOUT:   %pattern_type.5fc: type = pattern_type %facet_type.4ee [concrete]
// CHECK:STDOUT:   %facet_type.f88: type = facet_type <@Destroy & @DivWith.1, @DivWith.1(%Int16) where %impl.elem0.6cb = %Int32> [concrete]
// CHECK:STDOUT:   %Int32.Op.type.d5d3d3.5: type = fn_type @Int32.Op.5 [concrete]
// CHECK:STDOUT:   %Int32.Op.ecb1a5.5: %Int32.Op.type.d5d3d3.5 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.387: <witness> = custom_witness (%Int32, %Int32.Op.ecb1a5.5), @DivWith.1, @DivWith.1(%Int16) [concrete]
// CHECK:STDOUT:   %facet_value.270: %facet_type.f88 = facet_value %Int32, (%custom_witness.f83583.2, %custom_witness.387) [concrete]
// CHECK:STDOUT:   %pattern_type.871: type = pattern_type %facet_type.f88 [concrete]
// CHECK:STDOUT:   %facet_type.994: type = facet_type <@Destroy & @ModWith.1, @ModWith.1(%Int16) where %impl.elem0.2b9 = %Int32> [concrete]
// CHECK:STDOUT:   %Int32.Op.type.d5d3d3.6: type = fn_type @Int32.Op.6 [concrete]
// CHECK:STDOUT:   %Int32.Op.ecb1a5.6: %Int32.Op.type.d5d3d3.6 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.5cd: <witness> = custom_witness (%Int32, %Int32.Op.ecb1a5.6), @ModWith.1, @ModWith.1(%Int16) [concrete]
// CHECK:STDOUT:   %facet_value.64f: %facet_type.994 = facet_value %Int32, (%custom_witness.f83583.2, %custom_witness.5cd) [concrete]
// CHECK:STDOUT:   %pattern_type.269: type = pattern_type %facet_type.994 [concrete]
// CHECK:STDOUT:   %Int32.Op.type.d5d3d3.7: type = fn_type @Int32.Op.7 [concrete]
// CHECK:STDOUT:   %Int32.Op.ecb1a5.7: %Int32.Op.type.d5d3d3.7 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.27d802.2: <witness> = custom_witness (%Int32, %Int32.Op.ecb1a5.7), @AddWith.1, @AddWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %facet_value.f1e: %facet_type.36e = facet_value %Int32, (%custom_witness.f83583.2, %custom_witness.27d802.2) [concrete]
// CHECK:STDOUT:   %Int32.Op.type.d5d3d3.8: type = fn_type @Int32.Op.8 [concrete]
// CHECK:STDOUT:   %Int32.Op.ecb1a5.8: %Int32.Op.type.d5d3d3.8 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.f4a1b8.2: <witness> = custom_witness (%Int32, %Int32.Op.ecb1a5.8), @SubWith.1, @SubWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %facet_value.b8e: %facet_type.dfd = facet_value %Int32, (%custom_witness.f83583.2, %custom_witness.f4a1b8.2) [concrete]
// CHECK:STDOUT:   %Int32.Op.type.d5d3d3.9: type = fn_type @Int32.Op.9 [concrete]
// CHECK:STDOUT:   %Int32.Op.ecb1a5.9: %Int32.Op.type.d5d3d3.9 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.cbf765.2: <witness> = custom_witness (%Int32, %Int32.Op.ecb1a5.9), @MulWith.1, @MulWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %facet_value.3cc: %facet_type.4d5 = facet_value %Int32, (%custom_witness.f83583.2, %custom_witness.cbf765.2) [concrete]
// CHECK:STDOUT:   %Int32.Op.type.d5d3d3.10: type = fn_type @Int32.Op.10 [concrete]
// CHECK:STDOUT:   %Int32.Op.ecb1a5.10: %Int32.Op.type.d5d3d3.10 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.ec9d25.2: <witness> = custom_witness (%Int32, %Int32.Op.ecb1a5.10), @DivWith.1, @DivWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %facet_value.64a: %facet_type.6bd = facet_value %Int32, (%custom_witness.f83583.2, %custom_witness.ec9d25.2) [concrete]
// CHECK:STDOUT:   %Int32.Op.type.d5d3d3.11: type = fn_type @Int32.Op.11 [concrete]
// CHECK:STDOUT:   %Int32.Op.ecb1a5.11: %Int32.Op.type.d5d3d3.11 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.998647.2: <witness> = custom_witness (%Int32, %Int32.Op.ecb1a5.11), @ModWith.1, @ModWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %facet_value.c6d: %facet_type.8ee = facet_value %Int32, (%custom_witness.f83583.2, %custom_witness.998647.2) [concrete]
// CHECK:STDOUT:   %Int64.Op.type.d5d3d3.1: type = fn_type @Int64.Op.1 [concrete]
// CHECK:STDOUT:   %Int64.Op.ecb1a5.1: %Int64.Op.type.d5d3d3.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.f83583.3: <witness> = custom_witness (%Int64.Op.ecb1a5.1), @Destroy [concrete]
// CHECK:STDOUT:   %Destroy.facet.a4a: %Destroy.type = facet_value %Int64, (%custom_witness.f83583.3) [concrete]
// CHECK:STDOUT:   %facet_type.b51: type = facet_type <@Destroy & @AddWith.1, @AddWith.1(%Int64)> [concrete]
// CHECK:STDOUT:   %.Self.b56: %facet_type.b51 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.71c: <witness> = lookup_impl_witness %.Self.b56, @AddWith.1, @AddWith.1(%Int64) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.cdf: type = impl_witness_access %AddWith.lookup_impl_witness.71c, element0 [symbolic_self]
// CHECK:STDOUT:   %facet_type.b34: type = facet_type <@Destroy & @AddWith.1, @AddWith.1(%Int64) where %impl.elem0.cdf = %Int64> [concrete]
// CHECK:STDOUT:   %AddWith.type.b5c: type = facet_type <@AddWith.1, @AddWith.1(%Int64)> [concrete]
// CHECK:STDOUT:   %AddWith.assoc_type.ea8: type = assoc_entity_type @AddWith.1, @AddWith.1(%Int64) [concrete]
// CHECK:STDOUT:   %assoc0.c63: %AddWith.assoc_type.ea8 = assoc_entity element0, imports.%Core.import_ref.b37 [concrete]
// CHECK:STDOUT:   %assoc1.69c: %AddWith.assoc_type.ea8 = assoc_entity element1, imports.%Core.import_ref.935 [concrete]
// CHECK:STDOUT:   %Op.type.0b8387.6: type = fn_type @Op.6 [concrete]
// CHECK:STDOUT:   %Op.b9c47d.6: %Op.type.0b8387.6 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.e14: <witness> = custom_witness (%Int64, %Op.b9c47d.6), @AddWith.1, @AddWith.1(%Int64) [concrete]
// CHECK:STDOUT:   %facet_value.9f2: %facet_type.b34 = facet_value %Int32, (%custom_witness.f83583.2, %custom_witness.e14) [concrete]
// CHECK:STDOUT:   %complete_type.245: <witness> = complete_type_witness %facet_type.b51 [concrete]
// CHECK:STDOUT:   %.Self.as_type.f5c: type = facet_access_type %.Self.b56 [symbolic_self]
// CHECK:STDOUT:   %pattern_type.1c9: type = pattern_type %facet_type.b34 [concrete]
// CHECK:STDOUT:   %facet_type.8dc: type = facet_type <@Destroy & @SubWith.1, @SubWith.1(%Int64)> [concrete]
// CHECK:STDOUT:   %.Self.504: %facet_type.8dc = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.960: <witness> = lookup_impl_witness %.Self.504, @SubWith.1, @SubWith.1(%Int64) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.8ab: type = impl_witness_access %SubWith.lookup_impl_witness.960, element0 [symbolic_self]
// CHECK:STDOUT:   %facet_type.b2b: type = facet_type <@Destroy & @SubWith.1, @SubWith.1(%Int64) where %impl.elem0.8ab = %Int64> [concrete]
// CHECK:STDOUT:   %SubWith.type.4c4: type = facet_type <@SubWith.1, @SubWith.1(%Int64)> [concrete]
// CHECK:STDOUT:   %SubWith.assoc_type.25c: type = assoc_entity_type @SubWith.1, @SubWith.1(%Int64) [concrete]
// CHECK:STDOUT:   %assoc0.2a2: %SubWith.assoc_type.25c = assoc_entity element0, imports.%Core.import_ref.bb2 [concrete]
// CHECK:STDOUT:   %assoc1.4e8: %SubWith.assoc_type.25c = assoc_entity element1, imports.%Core.import_ref.480 [concrete]
// CHECK:STDOUT:   %Op.type.0b8387.7: type = fn_type @Op.7 [concrete]
// CHECK:STDOUT:   %Op.b9c47d.7: %Op.type.0b8387.7 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.d0e: <witness> = custom_witness (%Int64, %Op.b9c47d.7), @SubWith.1, @SubWith.1(%Int64) [concrete]
// CHECK:STDOUT:   %facet_value.b33: %facet_type.b2b = facet_value %Int32, (%custom_witness.f83583.2, %custom_witness.d0e) [concrete]
// CHECK:STDOUT:   %complete_type.8ff: <witness> = complete_type_witness %facet_type.8dc [concrete]
// CHECK:STDOUT:   %.Self.as_type.e9c: type = facet_access_type %.Self.504 [symbolic_self]
// CHECK:STDOUT:   %pattern_type.e40: type = pattern_type %facet_type.b2b [concrete]
// CHECK:STDOUT:   %facet_type.4f0: type = facet_type <@Destroy & @MulWith.1, @MulWith.1(%Int64)> [concrete]
// CHECK:STDOUT:   %.Self.bc5: %facet_type.4f0 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.3f1: <witness> = lookup_impl_witness %.Self.bc5, @MulWith.1, @MulWith.1(%Int64) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.a20: type = impl_witness_access %MulWith.lookup_impl_witness.3f1, element0 [symbolic_self]
// CHECK:STDOUT:   %facet_type.b1b: type = facet_type <@Destroy & @MulWith.1, @MulWith.1(%Int64) where %impl.elem0.a20 = %Int64> [concrete]
// CHECK:STDOUT:   %MulWith.type.6fc: type = facet_type <@MulWith.1, @MulWith.1(%Int64)> [concrete]
// CHECK:STDOUT:   %MulWith.assoc_type.23b: type = assoc_entity_type @MulWith.1, @MulWith.1(%Int64) [concrete]
// CHECK:STDOUT:   %assoc0.5d0: %MulWith.assoc_type.23b = assoc_entity element0, imports.%Core.import_ref.a1a [concrete]
// CHECK:STDOUT:   %assoc1.78a: %MulWith.assoc_type.23b = assoc_entity element1, imports.%Core.import_ref.7ed [concrete]
// CHECK:STDOUT:   %Op.type.0b8387.8: type = fn_type @Op.8 [concrete]
// CHECK:STDOUT:   %Op.b9c47d.8: %Op.type.0b8387.8 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.784: <witness> = custom_witness (%Int64, %Op.b9c47d.8), @MulWith.1, @MulWith.1(%Int64) [concrete]
// CHECK:STDOUT:   %facet_value.1e7: %facet_type.b1b = facet_value %Int32, (%custom_witness.f83583.2, %custom_witness.784) [concrete]
// CHECK:STDOUT:   %complete_type.609: <witness> = complete_type_witness %facet_type.4f0 [concrete]
// CHECK:STDOUT:   %.Self.as_type.ec4: type = facet_access_type %.Self.bc5 [symbolic_self]
// CHECK:STDOUT:   %pattern_type.c0b: type = pattern_type %facet_type.b1b [concrete]
// CHECK:STDOUT:   %facet_type.d54: type = facet_type <@Destroy & @DivWith.1, @DivWith.1(%Int64)> [concrete]
// CHECK:STDOUT:   %.Self.987: %facet_type.d54 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.d5e: <witness> = lookup_impl_witness %.Self.987, @DivWith.1, @DivWith.1(%Int64) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.cba: type = impl_witness_access %DivWith.lookup_impl_witness.d5e, element0 [symbolic_self]
// CHECK:STDOUT:   %facet_type.0c6: type = facet_type <@Destroy & @DivWith.1, @DivWith.1(%Int64) where %impl.elem0.cba = %Int64> [concrete]
// CHECK:STDOUT:   %DivWith.type.46b: type = facet_type <@DivWith.1, @DivWith.1(%Int64)> [concrete]
// CHECK:STDOUT:   %DivWith.assoc_type.d66: type = assoc_entity_type @DivWith.1, @DivWith.1(%Int64) [concrete]
// CHECK:STDOUT:   %assoc0.d4a: %DivWith.assoc_type.d66 = assoc_entity element0, imports.%Core.import_ref.b8f [concrete]
// CHECK:STDOUT:   %assoc1.e3e: %DivWith.assoc_type.d66 = assoc_entity element1, imports.%Core.import_ref.44d [concrete]
// CHECK:STDOUT:   %Op.type.0b8387.9: type = fn_type @Op.9 [concrete]
// CHECK:STDOUT:   %Op.b9c47d.9: %Op.type.0b8387.9 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.a1d: <witness> = custom_witness (%Int64, %Op.b9c47d.9), @DivWith.1, @DivWith.1(%Int64) [concrete]
// CHECK:STDOUT:   %facet_value.23f: %facet_type.0c6 = facet_value %Int32, (%custom_witness.f83583.2, %custom_witness.a1d) [concrete]
// CHECK:STDOUT:   %complete_type.d68: <witness> = complete_type_witness %facet_type.d54 [concrete]
// CHECK:STDOUT:   %.Self.as_type.fad: type = facet_access_type %.Self.987 [symbolic_self]
// CHECK:STDOUT:   %pattern_type.4e3: type = pattern_type %facet_type.0c6 [concrete]
// CHECK:STDOUT:   %facet_type.ad6: type = facet_type <@Destroy & @ModWith.1, @ModWith.1(%Int64)> [concrete]
// CHECK:STDOUT:   %.Self.eba: %facet_type.ad6 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.d93: <witness> = lookup_impl_witness %.Self.eba, @ModWith.1, @ModWith.1(%Int64) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.036: type = impl_witness_access %ModWith.lookup_impl_witness.d93, element0 [symbolic_self]
// CHECK:STDOUT:   %facet_type.892: type = facet_type <@Destroy & @ModWith.1, @ModWith.1(%Int64) where %impl.elem0.036 = %Int64> [concrete]
// CHECK:STDOUT:   %ModWith.type.610: type = facet_type <@ModWith.1, @ModWith.1(%Int64)> [concrete]
// CHECK:STDOUT:   %ModWith.assoc_type.08d: type = assoc_entity_type @ModWith.1, @ModWith.1(%Int64) [concrete]
// CHECK:STDOUT:   %assoc0.558: %ModWith.assoc_type.08d = assoc_entity element0, imports.%Core.import_ref.f19 [concrete]
// CHECK:STDOUT:   %assoc1.ad0: %ModWith.assoc_type.08d = assoc_entity element1, imports.%Core.import_ref.556 [concrete]
// CHECK:STDOUT:   %Op.type.0b8387.10: type = fn_type @Op.10 [concrete]
// CHECK:STDOUT:   %Op.b9c47d.10: %Op.type.0b8387.10 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.1f5: <witness> = custom_witness (%Int64, %Op.b9c47d.10), @ModWith.1, @ModWith.1(%Int64) [concrete]
// CHECK:STDOUT:   %facet_value.b6d: %facet_type.892 = facet_value %Int32, (%custom_witness.f83583.2, %custom_witness.1f5) [concrete]
// CHECK:STDOUT:   %complete_type.1c0: <witness> = complete_type_witness %facet_type.ad6 [concrete]
// CHECK:STDOUT:   %.Self.as_type.0bf: type = facet_access_type %.Self.eba [symbolic_self]
// CHECK:STDOUT:   %pattern_type.a7b8: type = pattern_type %facet_type.892 [concrete]
// CHECK:STDOUT:   %facet_type.167: type = facet_type <@Destroy & @AddWith.1, @AddWith.1(%Int32) where %impl.elem0.2e9 = %Int64> [concrete]
// CHECK:STDOUT:   %Op.type.0b8387.11: type = fn_type @Op.11 [concrete]
// CHECK:STDOUT:   %Op.b9c47d.11: %Op.type.0b8387.11 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.8f1: <witness> = custom_witness (%Int64, %Op.b9c47d.11), @AddWith.1, @AddWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %facet_value.ac6: %facet_type.167 = facet_value %Int64, (%custom_witness.f83583.3, %custom_witness.8f1) [concrete]
// CHECK:STDOUT:   %pattern_type.109: type = pattern_type %facet_type.167 [concrete]
// CHECK:STDOUT:   %facet_type.39e: type = facet_type <@Destroy & @SubWith.1, @SubWith.1(%Int32) where %impl.elem0.7af = %Int64> [concrete]
// CHECK:STDOUT:   %Op.type.0b8387.12: type = fn_type @Op.12 [concrete]
// CHECK:STDOUT:   %Op.b9c47d.12: %Op.type.0b8387.12 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.148: <witness> = custom_witness (%Int64, %Op.b9c47d.12), @SubWith.1, @SubWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %facet_value.fce: %facet_type.39e = facet_value %Int64, (%custom_witness.f83583.3, %custom_witness.148) [concrete]
// CHECK:STDOUT:   %pattern_type.a19: type = pattern_type %facet_type.39e [concrete]
// CHECK:STDOUT:   %facet_type.7cf: type = facet_type <@Destroy & @MulWith.1, @MulWith.1(%Int32) where %impl.elem0.a71 = %Int64> [concrete]
// CHECK:STDOUT:   %Op.type.0b8387.13: type = fn_type @Op.13 [concrete]
// CHECK:STDOUT:   %Op.b9c47d.13: %Op.type.0b8387.13 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.d3e: <witness> = custom_witness (%Int64, %Op.b9c47d.13), @MulWith.1, @MulWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %facet_value.e5d: %facet_type.7cf = facet_value %Int64, (%custom_witness.f83583.3, %custom_witness.d3e) [concrete]
// CHECK:STDOUT:   %pattern_type.2b4: type = pattern_type %facet_type.7cf [concrete]
// CHECK:STDOUT:   %facet_type.e1a: type = facet_type <@Destroy & @DivWith.1, @DivWith.1(%Int32) where %impl.elem0.920 = %Int64> [concrete]
// CHECK:STDOUT:   %Op.type.0b8387.14: type = fn_type @Op.14 [concrete]
// CHECK:STDOUT:   %Op.b9c47d.14: %Op.type.0b8387.14 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.847: <witness> = custom_witness (%Int64, %Op.b9c47d.14), @DivWith.1, @DivWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %facet_value.ab0: %facet_type.e1a = facet_value %Int64, (%custom_witness.f83583.3, %custom_witness.847) [concrete]
// CHECK:STDOUT:   %pattern_type.d89: type = pattern_type %facet_type.e1a [concrete]
// CHECK:STDOUT:   %facet_type.349: type = facet_type <@Destroy & @ModWith.1, @ModWith.1(%Int32) where %impl.elem0.1ad = %Int64> [concrete]
// CHECK:STDOUT:   %Op.type.0b8387.15: type = fn_type @Op.15 [concrete]
// CHECK:STDOUT:   %Op.b9c47d.15: %Op.type.0b8387.15 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.4aa: <witness> = custom_witness (%Int64, %Op.b9c47d.15), @ModWith.1, @ModWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %facet_value.32f: %facet_type.349 = facet_value %Int64, (%custom_witness.f83583.3, %custom_witness.4aa) [concrete]
// CHECK:STDOUT:   %pattern_type.a62: type = pattern_type %facet_type.349 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.2a2: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.a4a) [concrete]
// CHECK:STDOUT:   %.08b: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.2a2, %Destroy.facet.a4a [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.7bc: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.700) [concrete]
// CHECK:STDOUT:   %.362: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.7bc, %Destroy.facet.700 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.7e2: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.738) [concrete]
// CHECK:STDOUT:   %.d2d: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.7e2, %Destroy.facet.738 [concrete]
// CHECK:STDOUT:   %U.67d: type = symbolic_binding U, 0 [symbolic]
// CHECK:STDOUT:   %AddAssignWith.type.dc8ed5.1: type = facet_type <@AddAssignWith.1, @AddAssignWith.1(%Other)> [symbolic]
// CHECK:STDOUT:   %Self.689a47.1: %AddAssignWith.type.dc8ed5.1 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %AddAssignWith.assoc_type.a25160.1: type = assoc_entity_type @AddAssignWith.1, @AddAssignWith.1(%Other) [symbolic]
// CHECK:STDOUT:   %AddAssignWith.WithSelf.Op.type.8b23d4.1: type = fn_type @AddAssignWith.WithSelf.Op, @AddAssignWith.WithSelf(%Other, %Self.689a47.1) [symbolic]
// CHECK:STDOUT:   %AddAssignWith.WithSelf.Op.4cbad4.1: %AddAssignWith.WithSelf.Op.type.8b23d4.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %AddAssignWith.type.dc8ed5.2: type = facet_type <@AddAssignWith.1, @AddAssignWith.1(%U.67d)> [symbolic]
// CHECK:STDOUT:   %pattern_type.f87: type = pattern_type %AddAssignWith.type.dc8ed5.2 [symbolic]
// CHECK:STDOUT:   %T.689: %AddAssignWith.type.dc8ed5.2 = symbolic_binding T, 1 [symbolic]
// CHECK:STDOUT:   %AddAssignWith.assoc_type.a25160.2: type = assoc_entity_type @AddAssignWith.1, @AddAssignWith.1(%U.67d) [symbolic]
// CHECK:STDOUT:   %assoc0.9fb771.2: %AddAssignWith.assoc_type.a25160.2 = assoc_entity element0, imports.%Core.import_ref.4eb [symbolic]
// CHECK:STDOUT:   %T.as_type.7d6: type = facet_access_type %T.689 [symbolic]
// CHECK:STDOUT:   %pattern_type.657c0b.2: type = pattern_type %T.as_type.7d6 [symbolic]
// CHECK:STDOUT:   %pattern_type.51d1c4.4: type = pattern_type %U.67d [symbolic]
// CHECK:STDOUT:   %require_complete.44c: <witness> = require_complete_type %AddAssignWith.type.dc8ed5.2 [symbolic]
// CHECK:STDOUT:   %assoc0.c1f: %AddAssignWith.assoc_type.a25160.1 = assoc_entity element0, imports.%Core.import_ref.763 [symbolic]
// CHECK:STDOUT:   %AddAssignWith.lookup_impl_witness: <witness> = lookup_impl_witness %T.689, @AddAssignWith.1, @AddAssignWith.1(%U.67d) [symbolic]
// CHECK:STDOUT:   %AddAssignWith.WithSelf.Op.type.8b23d4.3: type = fn_type @AddAssignWith.WithSelf.Op, @AddAssignWith.WithSelf(%U.67d, %T.689) [symbolic]
// CHECK:STDOUT:   %.ef8: type = fn_type_with_self_type %AddAssignWith.WithSelf.Op.type.8b23d4.3, %T.689 [symbolic]
// CHECK:STDOUT:   %impl.elem0.89f: %.ef8 = impl_witness_access %AddAssignWith.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.7ee: <specific function> = specific_impl_function %impl.elem0.89f, @AddAssignWith.WithSelf.Op(%U.67d, %T.689) [symbolic]
// CHECK:STDOUT:   %SubAssignWith.type.a95ddd.1: type = facet_type <@SubAssignWith.1, @SubAssignWith.1(%Other)> [symbolic]
// CHECK:STDOUT:   %Self.af3029.1: %SubAssignWith.type.a95ddd.1 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %SubAssignWith.assoc_type.0440e4.1: type = assoc_entity_type @SubAssignWith.1, @SubAssignWith.1(%Other) [symbolic]
// CHECK:STDOUT:   %SubAssignWith.WithSelf.Op.type.ac317f.1: type = fn_type @SubAssignWith.WithSelf.Op, @SubAssignWith.WithSelf(%Other, %Self.af3029.1) [symbolic]
// CHECK:STDOUT:   %SubAssignWith.WithSelf.Op.b6cd67.1: %SubAssignWith.WithSelf.Op.type.ac317f.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %SubAssignWith.type.a95ddd.2: type = facet_type <@SubAssignWith.1, @SubAssignWith.1(%U.67d)> [symbolic]
// CHECK:STDOUT:   %pattern_type.3cd: type = pattern_type %SubAssignWith.type.a95ddd.2 [symbolic]
// CHECK:STDOUT:   %T.af3: %SubAssignWith.type.a95ddd.2 = symbolic_binding T, 1 [symbolic]
// CHECK:STDOUT:   %SubAssignWith.assoc_type.0440e4.2: type = assoc_entity_type @SubAssignWith.1, @SubAssignWith.1(%U.67d) [symbolic]
// CHECK:STDOUT:   %assoc0.180b38.2: %SubAssignWith.assoc_type.0440e4.2 = assoc_entity element0, imports.%Core.import_ref.a2d0 [symbolic]
// CHECK:STDOUT:   %T.as_type.a39: type = facet_access_type %T.af3 [symbolic]
// CHECK:STDOUT:   %pattern_type.5c82ed.2: type = pattern_type %T.as_type.a39 [symbolic]
// CHECK:STDOUT:   %require_complete.f21: <witness> = require_complete_type %SubAssignWith.type.a95ddd.2 [symbolic]
// CHECK:STDOUT:   %assoc0.5f4: %SubAssignWith.assoc_type.0440e4.1 = assoc_entity element0, imports.%Core.import_ref.3c0 [symbolic]
// CHECK:STDOUT:   %SubAssignWith.lookup_impl_witness: <witness> = lookup_impl_witness %T.af3, @SubAssignWith.1, @SubAssignWith.1(%U.67d) [symbolic]
// CHECK:STDOUT:   %SubAssignWith.WithSelf.Op.type.ac317f.3: type = fn_type @SubAssignWith.WithSelf.Op, @SubAssignWith.WithSelf(%U.67d, %T.af3) [symbolic]
// CHECK:STDOUT:   %.6d8a: type = fn_type_with_self_type %SubAssignWith.WithSelf.Op.type.ac317f.3, %T.af3 [symbolic]
// CHECK:STDOUT:   %impl.elem0.ab4: %.6d8a = impl_witness_access %SubAssignWith.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.579: <specific function> = specific_impl_function %impl.elem0.ab4, @SubAssignWith.WithSelf.Op(%U.67d, %T.af3) [symbolic]
// CHECK:STDOUT:   %MulAssignWith.type.c08ab6.1: type = facet_type <@MulAssignWith.1, @MulAssignWith.1(%Other)> [symbolic]
// CHECK:STDOUT:   %Self.c8f88d.1: %MulAssignWith.type.c08ab6.1 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %MulAssignWith.assoc_type.dd8648.1: type = assoc_entity_type @MulAssignWith.1, @MulAssignWith.1(%Other) [symbolic]
// CHECK:STDOUT:   %MulAssignWith.WithSelf.Op.type.dff752.1: type = fn_type @MulAssignWith.WithSelf.Op, @MulAssignWith.WithSelf(%Other, %Self.c8f88d.1) [symbolic]
// CHECK:STDOUT:   %MulAssignWith.WithSelf.Op.28a4dd.1: %MulAssignWith.WithSelf.Op.type.dff752.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %MulAssignWith.type.c08ab6.2: type = facet_type <@MulAssignWith.1, @MulAssignWith.1(%U.67d)> [symbolic]
// CHECK:STDOUT:   %pattern_type.78c: type = pattern_type %MulAssignWith.type.c08ab6.2 [symbolic]
// CHECK:STDOUT:   %T.c8f: %MulAssignWith.type.c08ab6.2 = symbolic_binding T, 1 [symbolic]
// CHECK:STDOUT:   %MulAssignWith.assoc_type.dd8648.2: type = assoc_entity_type @MulAssignWith.1, @MulAssignWith.1(%U.67d) [symbolic]
// CHECK:STDOUT:   %assoc0.bb9cfe.2: %MulAssignWith.assoc_type.dd8648.2 = assoc_entity element0, imports.%Core.import_ref.475 [symbolic]
// CHECK:STDOUT:   %T.as_type.26e: type = facet_access_type %T.c8f [symbolic]
// CHECK:STDOUT:   %pattern_type.3695a8.2: type = pattern_type %T.as_type.26e [symbolic]
// CHECK:STDOUT:   %require_complete.635: <witness> = require_complete_type %MulAssignWith.type.c08ab6.2 [symbolic]
// CHECK:STDOUT:   %assoc0.181: %MulAssignWith.assoc_type.dd8648.1 = assoc_entity element0, imports.%Core.import_ref.768 [symbolic]
// CHECK:STDOUT:   %MulAssignWith.lookup_impl_witness: <witness> = lookup_impl_witness %T.c8f, @MulAssignWith.1, @MulAssignWith.1(%U.67d) [symbolic]
// CHECK:STDOUT:   %MulAssignWith.WithSelf.Op.type.dff752.3: type = fn_type @MulAssignWith.WithSelf.Op, @MulAssignWith.WithSelf(%U.67d, %T.c8f) [symbolic]
// CHECK:STDOUT:   %.9a8d: type = fn_type_with_self_type %MulAssignWith.WithSelf.Op.type.dff752.3, %T.c8f [symbolic]
// CHECK:STDOUT:   %impl.elem0.653: %.9a8d = impl_witness_access %MulAssignWith.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.5ff: <specific function> = specific_impl_function %impl.elem0.653, @MulAssignWith.WithSelf.Op(%U.67d, %T.c8f) [symbolic]
// CHECK:STDOUT:   %DivAssignWith.type.d7b8dd.1: type = facet_type <@DivAssignWith.1, @DivAssignWith.1(%Other)> [symbolic]
// CHECK:STDOUT:   %Self.f22977.1: %DivAssignWith.type.d7b8dd.1 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %DivAssignWith.assoc_type.738ee6.1: type = assoc_entity_type @DivAssignWith.1, @DivAssignWith.1(%Other) [symbolic]
// CHECK:STDOUT:   %DivAssignWith.WithSelf.Op.type.3077dd.1: type = fn_type @DivAssignWith.WithSelf.Op, @DivAssignWith.WithSelf(%Other, %Self.f22977.1) [symbolic]
// CHECK:STDOUT:   %DivAssignWith.WithSelf.Op.710329.1: %DivAssignWith.WithSelf.Op.type.3077dd.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %DivAssignWith.type.d7b8dd.2: type = facet_type <@DivAssignWith.1, @DivAssignWith.1(%U.67d)> [symbolic]
// CHECK:STDOUT:   %pattern_type.a67: type = pattern_type %DivAssignWith.type.d7b8dd.2 [symbolic]
// CHECK:STDOUT:   %T.f22: %DivAssignWith.type.d7b8dd.2 = symbolic_binding T, 1 [symbolic]
// CHECK:STDOUT:   %DivAssignWith.assoc_type.738ee6.2: type = assoc_entity_type @DivAssignWith.1, @DivAssignWith.1(%U.67d) [symbolic]
// CHECK:STDOUT:   %assoc0.d42d61.2: %DivAssignWith.assoc_type.738ee6.2 = assoc_entity element0, imports.%Core.import_ref.12d [symbolic]
// CHECK:STDOUT:   %T.as_type.dc5: type = facet_access_type %T.f22 [symbolic]
// CHECK:STDOUT:   %pattern_type.ef84c8.2: type = pattern_type %T.as_type.dc5 [symbolic]
// CHECK:STDOUT:   %require_complete.3cf: <witness> = require_complete_type %DivAssignWith.type.d7b8dd.2 [symbolic]
// CHECK:STDOUT:   %assoc0.f6f: %DivAssignWith.assoc_type.738ee6.1 = assoc_entity element0, imports.%Core.import_ref.9c0 [symbolic]
// CHECK:STDOUT:   %DivAssignWith.lookup_impl_witness: <witness> = lookup_impl_witness %T.f22, @DivAssignWith.1, @DivAssignWith.1(%U.67d) [symbolic]
// CHECK:STDOUT:   %DivAssignWith.WithSelf.Op.type.3077dd.3: type = fn_type @DivAssignWith.WithSelf.Op, @DivAssignWith.WithSelf(%U.67d, %T.f22) [symbolic]
// CHECK:STDOUT:   %.d5f: type = fn_type_with_self_type %DivAssignWith.WithSelf.Op.type.3077dd.3, %T.f22 [symbolic]
// CHECK:STDOUT:   %impl.elem0.faa: %.d5f = impl_witness_access %DivAssignWith.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.63f: <specific function> = specific_impl_function %impl.elem0.faa, @DivAssignWith.WithSelf.Op(%U.67d, %T.f22) [symbolic]
// CHECK:STDOUT:   %ModAssignWith.type.958f4d.1: type = facet_type <@ModAssignWith.1, @ModAssignWith.1(%Other)> [symbolic]
// CHECK:STDOUT:   %Self.37b6d9.1: %ModAssignWith.type.958f4d.1 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %ModAssignWith.assoc_type.9ba07c.1: type = assoc_entity_type @ModAssignWith.1, @ModAssignWith.1(%Other) [symbolic]
// CHECK:STDOUT:   %ModAssignWith.WithSelf.Op.type.3abf50.1: type = fn_type @ModAssignWith.WithSelf.Op, @ModAssignWith.WithSelf(%Other, %Self.37b6d9.1) [symbolic]
// CHECK:STDOUT:   %ModAssignWith.WithSelf.Op.42aedd.1: %ModAssignWith.WithSelf.Op.type.3abf50.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %ModAssignWith.type.958f4d.2: type = facet_type <@ModAssignWith.1, @ModAssignWith.1(%U.67d)> [symbolic]
// CHECK:STDOUT:   %pattern_type.91c: type = pattern_type %ModAssignWith.type.958f4d.2 [symbolic]
// CHECK:STDOUT:   %T.37b: %ModAssignWith.type.958f4d.2 = symbolic_binding T, 1 [symbolic]
// CHECK:STDOUT:   %ModAssignWith.assoc_type.9ba07c.2: type = assoc_entity_type @ModAssignWith.1, @ModAssignWith.1(%U.67d) [symbolic]
// CHECK:STDOUT:   %assoc0.fd162a.2: %ModAssignWith.assoc_type.9ba07c.2 = assoc_entity element0, imports.%Core.import_ref.c884 [symbolic]
// CHECK:STDOUT:   %T.as_type.5ba: type = facet_access_type %T.37b [symbolic]
// CHECK:STDOUT:   %pattern_type.22dddb.2: type = pattern_type %T.as_type.5ba [symbolic]
// CHECK:STDOUT:   %require_complete.479: <witness> = require_complete_type %ModAssignWith.type.958f4d.2 [symbolic]
// CHECK:STDOUT:   %assoc0.26c: %ModAssignWith.assoc_type.9ba07c.1 = assoc_entity element0, imports.%Core.import_ref.d41 [symbolic]
// CHECK:STDOUT:   %ModAssignWith.lookup_impl_witness: <witness> = lookup_impl_witness %T.37b, @ModAssignWith.1, @ModAssignWith.1(%U.67d) [symbolic]
// CHECK:STDOUT:   %ModAssignWith.WithSelf.Op.type.3abf50.3: type = fn_type @ModAssignWith.WithSelf.Op, @ModAssignWith.WithSelf(%U.67d, %T.37b) [symbolic]
// CHECK:STDOUT:   %.759: type = fn_type_with_self_type %ModAssignWith.WithSelf.Op.type.3abf50.3, %T.37b [symbolic]
// CHECK:STDOUT:   %impl.elem0.289: %.759 = impl_witness_access %ModAssignWith.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.af2: <specific function> = specific_impl_function %impl.elem0.289, @ModAssignWith.WithSelf.Op(%U.67d, %T.37b) [symbolic]
// CHECK:STDOUT:   %AddAssignWith.type.5a5: type = facet_type <@AddAssignWith.1, @AddAssignWith.1(%i16)> [concrete]
// CHECK:STDOUT:   %AddAssignWith.assoc_type.12d: type = assoc_entity_type @AddAssignWith.1, @AddAssignWith.1(%i16) [concrete]
// CHECK:STDOUT:   %assoc0.6ce: %AddAssignWith.assoc_type.12d = assoc_entity element0, imports.%Core.import_ref.4eb [concrete]
// CHECK:STDOUT:   %Int16.Op.type.d5d3d3.13: type = fn_type @Int16.Op.13 [concrete]
// CHECK:STDOUT:   %Int16.Op.ecb1a5.13: %Int16.Op.type.d5d3d3.13 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.1f9: <witness> = custom_witness (%Int16.Op.ecb1a5.13), @AddAssignWith.1, @AddAssignWith.1(%i16) [concrete]
// CHECK:STDOUT:   %AddAssignWith.facet.b45: %AddAssignWith.type.5a5 = facet_value %Int16, (%custom_witness.1f9) [concrete]
// CHECK:STDOUT:   %pattern_type.307: type = pattern_type %AddAssignWith.type.5a5 [concrete]
// CHECK:STDOUT:   %SubAssignWith.type.2da: type = facet_type <@SubAssignWith.1, @SubAssignWith.1(%i16)> [concrete]
// CHECK:STDOUT:   %SubAssignWith.assoc_type.f3f: type = assoc_entity_type @SubAssignWith.1, @SubAssignWith.1(%i16) [concrete]
// CHECK:STDOUT:   %assoc0.1c8: %SubAssignWith.assoc_type.f3f = assoc_entity element0, imports.%Core.import_ref.a2d0 [concrete]
// CHECK:STDOUT:   %Int16.Op.type.d5d3d3.14: type = fn_type @Int16.Op.14 [concrete]
// CHECK:STDOUT:   %Int16.Op.ecb1a5.14: %Int16.Op.type.d5d3d3.14 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.922: <witness> = custom_witness (%Int16.Op.ecb1a5.14), @SubAssignWith.1, @SubAssignWith.1(%i16) [concrete]
// CHECK:STDOUT:   %SubAssignWith.facet.351: %SubAssignWith.type.2da = facet_value %Int16, (%custom_witness.922) [concrete]
// CHECK:STDOUT:   %pattern_type.2c7: type = pattern_type %SubAssignWith.type.2da [concrete]
// CHECK:STDOUT:   %MulAssignWith.type.a70: type = facet_type <@MulAssignWith.1, @MulAssignWith.1(%i16)> [concrete]
// CHECK:STDOUT:   %MulAssignWith.assoc_type.f71: type = assoc_entity_type @MulAssignWith.1, @MulAssignWith.1(%i16) [concrete]
// CHECK:STDOUT:   %assoc0.037: %MulAssignWith.assoc_type.f71 = assoc_entity element0, imports.%Core.import_ref.475 [concrete]
// CHECK:STDOUT:   %Int16.Op.type.d5d3d3.15: type = fn_type @Int16.Op.15 [concrete]
// CHECK:STDOUT:   %Int16.Op.ecb1a5.15: %Int16.Op.type.d5d3d3.15 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.c9b: <witness> = custom_witness (%Int16.Op.ecb1a5.15), @MulAssignWith.1, @MulAssignWith.1(%i16) [concrete]
// CHECK:STDOUT:   %MulAssignWith.facet.2aa: %MulAssignWith.type.a70 = facet_value %Int16, (%custom_witness.c9b) [concrete]
// CHECK:STDOUT:   %pattern_type.0b2: type = pattern_type %MulAssignWith.type.a70 [concrete]
// CHECK:STDOUT:   %DivAssignWith.type.7f8: type = facet_type <@DivAssignWith.1, @DivAssignWith.1(%i16)> [concrete]
// CHECK:STDOUT:   %DivAssignWith.assoc_type.e83: type = assoc_entity_type @DivAssignWith.1, @DivAssignWith.1(%i16) [concrete]
// CHECK:STDOUT:   %assoc0.065: %DivAssignWith.assoc_type.e83 = assoc_entity element0, imports.%Core.import_ref.12d [concrete]
// CHECK:STDOUT:   %Int16.Op.type.d5d3d3.16: type = fn_type @Int16.Op.16 [concrete]
// CHECK:STDOUT:   %Int16.Op.ecb1a5.16: %Int16.Op.type.d5d3d3.16 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.719: <witness> = custom_witness (%Int16.Op.ecb1a5.16), @DivAssignWith.1, @DivAssignWith.1(%i16) [concrete]
// CHECK:STDOUT:   %DivAssignWith.facet.253: %DivAssignWith.type.7f8 = facet_value %Int16, (%custom_witness.719) [concrete]
// CHECK:STDOUT:   %pattern_type.455: type = pattern_type %DivAssignWith.type.7f8 [concrete]
// CHECK:STDOUT:   %ModAssignWith.type.eee: type = facet_type <@ModAssignWith.1, @ModAssignWith.1(%i16)> [concrete]
// CHECK:STDOUT:   %ModAssignWith.assoc_type.08a: type = assoc_entity_type @ModAssignWith.1, @ModAssignWith.1(%i16) [concrete]
// CHECK:STDOUT:   %assoc0.c9f: %ModAssignWith.assoc_type.08a = assoc_entity element0, imports.%Core.import_ref.c884 [concrete]
// CHECK:STDOUT:   %Int16.Op.type.d5d3d3.17: type = fn_type @Int16.Op.17 [concrete]
// CHECK:STDOUT:   %Int16.Op.ecb1a5.17: %Int16.Op.type.d5d3d3.17 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.827: <witness> = custom_witness (%Int16.Op.ecb1a5.17), @ModAssignWith.1, @ModAssignWith.1(%i16) [concrete]
// CHECK:STDOUT:   %ModAssignWith.facet.c21: %ModAssignWith.type.eee = facet_value %Int16, (%custom_witness.827) [concrete]
// CHECK:STDOUT:   %pattern_type.df1: type = pattern_type %ModAssignWith.type.eee [concrete]
// CHECK:STDOUT:   %AddAssignWith.type.22f: type = facet_type <@AddAssignWith.1, @AddAssignWith.1(%Int32)> [concrete]
// CHECK:STDOUT:   %AddAssignWith.assoc_type.6a5: type = assoc_entity_type @AddAssignWith.1, @AddAssignWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %assoc0.20f: %AddAssignWith.assoc_type.6a5 = assoc_entity element0, imports.%Core.import_ref.4eb [concrete]
// CHECK:STDOUT:   %Int32.Op.type.d5d3d3.12: type = fn_type @Int32.Op.12 [concrete]
// CHECK:STDOUT:   %Int32.Op.ecb1a5.12: %Int32.Op.type.d5d3d3.12 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.167b9d.1: <witness> = custom_witness (%Int32.Op.ecb1a5.12), @AddAssignWith.1, @AddAssignWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %AddAssignWith.facet.e5f: %AddAssignWith.type.22f = facet_value %Int32, (%custom_witness.167b9d.1) [concrete]
// CHECK:STDOUT:   %pattern_type.2d7: type = pattern_type %AddAssignWith.type.22f [concrete]
// CHECK:STDOUT:   %SubAssignWith.type.c0c: type = facet_type <@SubAssignWith.1, @SubAssignWith.1(%Int32)> [concrete]
// CHECK:STDOUT:   %SubAssignWith.assoc_type.5d1: type = assoc_entity_type @SubAssignWith.1, @SubAssignWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %assoc0.5e1: %SubAssignWith.assoc_type.5d1 = assoc_entity element0, imports.%Core.import_ref.a2d0 [concrete]
// CHECK:STDOUT:   %Int32.Op.type.d5d3d3.14: type = fn_type @Int32.Op.14 [concrete]
// CHECK:STDOUT:   %Int32.Op.ecb1a5.14: %Int32.Op.type.d5d3d3.14 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.0d70d6.1: <witness> = custom_witness (%Int32.Op.ecb1a5.14), @SubAssignWith.1, @SubAssignWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %SubAssignWith.facet.c12: %SubAssignWith.type.c0c = facet_value %Int32, (%custom_witness.0d70d6.1) [concrete]
// CHECK:STDOUT:   %pattern_type.0ae: type = pattern_type %SubAssignWith.type.c0c [concrete]
// CHECK:STDOUT:   %MulAssignWith.type.f1b: type = facet_type <@MulAssignWith.1, @MulAssignWith.1(%Int32)> [concrete]
// CHECK:STDOUT:   %MulAssignWith.assoc_type.38d: type = assoc_entity_type @MulAssignWith.1, @MulAssignWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %assoc0.bb5: %MulAssignWith.assoc_type.38d = assoc_entity element0, imports.%Core.import_ref.475 [concrete]
// CHECK:STDOUT:   %Int32.Op.type.d5d3d3.15: type = fn_type @Int32.Op.15 [concrete]
// CHECK:STDOUT:   %Int32.Op.ecb1a5.15: %Int32.Op.type.d5d3d3.15 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.027973.1: <witness> = custom_witness (%Int32.Op.ecb1a5.15), @MulAssignWith.1, @MulAssignWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %MulAssignWith.facet.7cb: %MulAssignWith.type.f1b = facet_value %Int32, (%custom_witness.027973.1) [concrete]
// CHECK:STDOUT:   %pattern_type.b7f: type = pattern_type %MulAssignWith.type.f1b [concrete]
// CHECK:STDOUT:   %DivAssignWith.type.453: type = facet_type <@DivAssignWith.1, @DivAssignWith.1(%Int32)> [concrete]
// CHECK:STDOUT:   %DivAssignWith.assoc_type.693: type = assoc_entity_type @DivAssignWith.1, @DivAssignWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %assoc0.50a: %DivAssignWith.assoc_type.693 = assoc_entity element0, imports.%Core.import_ref.12d [concrete]
// CHECK:STDOUT:   %Int32.Op.type.d5d3d3.16: type = fn_type @Int32.Op.16 [concrete]
// CHECK:STDOUT:   %Int32.Op.ecb1a5.16: %Int32.Op.type.d5d3d3.16 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.30b4a6.1: <witness> = custom_witness (%Int32.Op.ecb1a5.16), @DivAssignWith.1, @DivAssignWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %DivAssignWith.facet.ffe: %DivAssignWith.type.453 = facet_value %Int32, (%custom_witness.30b4a6.1) [concrete]
// CHECK:STDOUT:   %pattern_type.964: type = pattern_type %DivAssignWith.type.453 [concrete]
// CHECK:STDOUT:   %ModAssignWith.type.390: type = facet_type <@ModAssignWith.1, @ModAssignWith.1(%Int32)> [concrete]
// CHECK:STDOUT:   %ModAssignWith.assoc_type.e54: type = assoc_entity_type @ModAssignWith.1, @ModAssignWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %assoc0.ec3: %ModAssignWith.assoc_type.e54 = assoc_entity element0, imports.%Core.import_ref.c884 [concrete]
// CHECK:STDOUT:   %Int32.Op.type.d5d3d3.17: type = fn_type @Int32.Op.17 [concrete]
// CHECK:STDOUT:   %Int32.Op.ecb1a5.17: %Int32.Op.type.d5d3d3.17 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.bbc21d.1: <witness> = custom_witness (%Int32.Op.ecb1a5.17), @ModAssignWith.1, @ModAssignWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %ModAssignWith.facet.1bc: %ModAssignWith.type.390 = facet_value %Int32, (%custom_witness.bbc21d.1) [concrete]
// CHECK:STDOUT:   %pattern_type.dd3: type = pattern_type %ModAssignWith.type.390 [concrete]
// CHECK:STDOUT:   %Int64.Op.type.d5d3d3.2: type = fn_type @Int64.Op.2 [concrete]
// CHECK:STDOUT:   %Int64.Op.ecb1a5.2: %Int64.Op.type.d5d3d3.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.167b9d.2: <witness> = custom_witness (%Int64.Op.ecb1a5.2), @AddAssignWith.1, @AddAssignWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %AddAssignWith.facet.62e: %AddAssignWith.type.22f = facet_value %Int64, (%custom_witness.167b9d.2) [concrete]
// CHECK:STDOUT:   %Int64.Op.type.d5d3d3.4: type = fn_type @Int64.Op.4 [concrete]
// CHECK:STDOUT:   %Int64.Op.ecb1a5.4: %Int64.Op.type.d5d3d3.4 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.0d70d6.2: <witness> = custom_witness (%Int64.Op.ecb1a5.4), @SubAssignWith.1, @SubAssignWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %SubAssignWith.facet.1f9: %SubAssignWith.type.c0c = facet_value %Int64, (%custom_witness.0d70d6.2) [concrete]
// CHECK:STDOUT:   %Int64.Op.type.d5d3d3.5: type = fn_type @Int64.Op.5 [concrete]
// CHECK:STDOUT:   %Int64.Op.ecb1a5.5: %Int64.Op.type.d5d3d3.5 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.027973.2: <witness> = custom_witness (%Int64.Op.ecb1a5.5), @MulAssignWith.1, @MulAssignWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %MulAssignWith.facet.b47: %MulAssignWith.type.f1b = facet_value %Int64, (%custom_witness.027973.2) [concrete]
// CHECK:STDOUT:   %Int64.Op.type.d5d3d3.6: type = fn_type @Int64.Op.6 [concrete]
// CHECK:STDOUT:   %Int64.Op.ecb1a5.6: %Int64.Op.type.d5d3d3.6 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.30b4a6.2: <witness> = custom_witness (%Int64.Op.ecb1a5.6), @DivAssignWith.1, @DivAssignWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %DivAssignWith.facet.f27: %DivAssignWith.type.453 = facet_value %Int64, (%custom_witness.30b4a6.2) [concrete]
// CHECK:STDOUT:   %Int64.Op.type.d5d3d3.7: type = fn_type @Int64.Op.7 [concrete]
// CHECK:STDOUT:   %Int64.Op.ecb1a5.7: %Int64.Op.type.d5d3d3.7 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.bbc21d.2: <witness> = custom_witness (%Int64.Op.ecb1a5.7), @ModAssignWith.1, @ModAssignWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %ModAssignWith.facet.a21: %ModAssignWith.type.390 = facet_value %Int64, (%custom_witness.bbc21d.2) [concrete]
// CHECK:STDOUT:   %Inc.type: type = facet_type <@Inc> [concrete]
// CHECK:STDOUT:   %T.fed: %Inc.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.1d1: type = facet_access_type %T.fed [symbolic]
// CHECK:STDOUT:   %pattern_type.8ba822.2: type = pattern_type %T.as_type.1d1 [symbolic]
// CHECK:STDOUT:   %Inc.lookup_impl_witness: <witness> = lookup_impl_witness %T.fed, @Inc [symbolic]
// CHECK:STDOUT:   %Inc.WithSelf.Op.type.0fa0d9.2: type = fn_type @Inc.WithSelf.Op, @Inc.WithSelf(%T.fed) [symbolic]
// CHECK:STDOUT:   %.8af: type = fn_type_with_self_type %Inc.WithSelf.Op.type.0fa0d9.2, %T.fed [symbolic]
// CHECK:STDOUT:   %impl.elem0.95b: %.8af = impl_witness_access %Inc.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.979: <specific function> = specific_impl_function %impl.elem0.95b, @Inc.WithSelf.Op(%T.fed) [symbolic]
// CHECK:STDOUT:   %Dec.type: type = facet_type <@Dec> [concrete]
// CHECK:STDOUT:   %T.c61: %Dec.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.5ae: type = facet_access_type %T.c61 [symbolic]
// CHECK:STDOUT:   %pattern_type.a489aa.2: type = pattern_type %T.as_type.5ae [symbolic]
// CHECK:STDOUT:   %Dec.lookup_impl_witness: <witness> = lookup_impl_witness %T.c61, @Dec [symbolic]
// CHECK:STDOUT:   %Dec.WithSelf.Op.type.a100f5.2: type = fn_type @Dec.WithSelf.Op, @Dec.WithSelf(%T.c61) [symbolic]
// CHECK:STDOUT:   %.e8e: type = fn_type_with_self_type %Dec.WithSelf.Op.type.a100f5.2, %T.c61 [symbolic]
// CHECK:STDOUT:   %impl.elem0.e49: %.e8e = impl_witness_access %Dec.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.38c: <specific function> = specific_impl_function %impl.elem0.e49, @Dec.WithSelf.Op(%T.c61) [symbolic]
// CHECK:STDOUT:   %Negate.type: type = facet_type <@Negate> [concrete]
// CHECK:STDOUT:   %facet_type.99f: type = facet_type <@Destroy & @Negate> [concrete]
// CHECK:STDOUT:   %.Self.81e: %facet_type.99f = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %Negate.lookup_impl_witness.21e: <witness> = lookup_impl_witness %.Self.81e, @Negate [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.d91: type = impl_witness_access %Negate.lookup_impl_witness.21e, element0 [symbolic_self]
// CHECK:STDOUT:   %facet_type.4bf: type = facet_type <@Destroy & @Negate where %impl.elem0.d91 = %Result.as_type> [symbolic]
// CHECK:STDOUT:   %pattern_type.ad3: type = pattern_type %facet_type.4bf [symbolic]
// CHECK:STDOUT:   %T.3f8: %facet_type.4bf = symbolic_binding T, 1 [symbolic]
// CHECK:STDOUT:   %T.as_type.013: type = facet_access_type %T.3f8 [symbolic]
// CHECK:STDOUT:   %pattern_type.48b: type = pattern_type %T.as_type.013 [symbolic]
// CHECK:STDOUT:   %Negate.lookup_impl_witness.755: <witness> = lookup_impl_witness %T.3f8, @Negate [symbolic]
// CHECK:STDOUT:   %Negate.facet.ad2: %Negate.type = facet_value %T.as_type.013, (%Negate.lookup_impl_witness.755) [symbolic]
// CHECK:STDOUT:   %Negate.WithSelf.Op.type.039: type = fn_type @Negate.WithSelf.Op, @Negate.WithSelf(%Negate.facet.ad2) [symbolic]
// CHECK:STDOUT:   %.487: type = fn_type_with_self_type %Negate.WithSelf.Op.type.039, %Negate.facet.ad2 [symbolic]
// CHECK:STDOUT:   %impl.elem1.026: %.487 = impl_witness_access %Negate.lookup_impl_witness.755, element1 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.ee1: <specific function> = specific_impl_function %impl.elem1.026, @Negate.WithSelf.Op(%Negate.facet.ad2) [symbolic]
// CHECK:STDOUT:   %Int32.Op.type.d5d3d3.18: type = fn_type @Int32.Op.18 [concrete]
// CHECK:STDOUT:   %Int32.Op.ecb1a5.18: %Int32.Op.type.d5d3d3.18 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.104: <witness> = custom_witness (%Int32.Op.ecb1a5.18), @Inc [concrete]
// CHECK:STDOUT:   %Inc.facet.338: %Inc.type = facet_value %Int32, (%custom_witness.104) [concrete]
// CHECK:STDOUT:   %Int32.Op.type.d5d3d3.19: type = fn_type @Int32.Op.19 [concrete]
// CHECK:STDOUT:   %Int32.Op.ecb1a5.19: %Int32.Op.type.d5d3d3.19 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.a2f: <witness> = custom_witness (%Int32.Op.ecb1a5.19), @Dec [concrete]
// CHECK:STDOUT:   %Dec.facet.b97: %Dec.type = facet_value %Int32, (%custom_witness.a2f) [concrete]
// CHECK:STDOUT:   %facet_type.67d: type = facet_type <@Destroy & @Negate where %impl.elem0.d91 = %Int64> [concrete]
// CHECK:STDOUT:   %Int16.Op.type.d5d3d3.18: type = fn_type @Int16.Op.18 [concrete]
// CHECK:STDOUT:   %Int16.Op.ecb1a5.18: %Int16.Op.type.d5d3d3.18 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.243: <witness> = custom_witness (%Int64, %Int16.Op.ecb1a5.18), @Negate [concrete]
// CHECK:STDOUT:   %facet_value.658: %facet_type.67d = facet_value %Int16, (%custom_witness.f83583.1, %custom_witness.243) [concrete]
// CHECK:STDOUT:   %pattern_type.100: type = pattern_type %facet_type.67d [concrete]
// CHECK:STDOUT:   %facet_type.016: type = facet_type <@Destroy & @Negate where %impl.elem0.d91 = %Int32> [concrete]
// CHECK:STDOUT:   %Int32.Op.type.d5d3d3.20: type = fn_type @Int32.Op.20 [concrete]
// CHECK:STDOUT:   %Int32.Op.ecb1a5.20: %Int32.Op.type.d5d3d3.20 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.9ae: <witness> = custom_witness (%Int32, %Int32.Op.ecb1a5.20), @Negate [concrete]
// CHECK:STDOUT:   %facet_value.cc4: %facet_type.016 = facet_value %Int32, (%custom_witness.f83583.2, %custom_witness.9ae) [concrete]
// CHECK:STDOUT:   %pattern_type.17e: type = pattern_type %facet_type.016 [concrete]
// CHECK:STDOUT:   %complete_type.eb1: <witness> = complete_type_witness %AddWith.type.116 [concrete]
// CHECK:STDOUT:   %AddWith.facet.b43: %AddWith.type.116 = facet_value %i16, (%custom_witness.91a) [concrete]
// CHECK:STDOUT:   %AddWith.WithSelf.Op.type.af0: type = fn_type @AddWith.WithSelf.Op, @AddWith.WithSelf(%Int16, %AddWith.facet.b43) [concrete]
// CHECK:STDOUT:   %.6ba: type = fn_type_with_self_type %AddWith.WithSelf.Op.type.af0, %AddWith.facet.b43 [concrete]
// CHECK:STDOUT:   %complete_type.99b: <witness> = complete_type_witness %SubWith.type.49a [concrete]
// CHECK:STDOUT:   %SubWith.facet.428: %SubWith.type.49a = facet_value %i16, (%custom_witness.7ee) [concrete]
// CHECK:STDOUT:   %SubWith.WithSelf.Op.type.196: type = fn_type @SubWith.WithSelf.Op, @SubWith.WithSelf(%Int16, %SubWith.facet.428) [concrete]
// CHECK:STDOUT:   %.ee9: type = fn_type_with_self_type %SubWith.WithSelf.Op.type.196, %SubWith.facet.428 [concrete]
// CHECK:STDOUT:   %complete_type.255: <witness> = complete_type_witness %MulWith.type.779 [concrete]
// CHECK:STDOUT:   %MulWith.facet.48c: %MulWith.type.779 = facet_value %i16, (%custom_witness.448) [concrete]
// CHECK:STDOUT:   %MulWith.WithSelf.Op.type.1d0: type = fn_type @MulWith.WithSelf.Op, @MulWith.WithSelf(%Int16, %MulWith.facet.48c) [concrete]
// CHECK:STDOUT:   %.03b: type = fn_type_with_self_type %MulWith.WithSelf.Op.type.1d0, %MulWith.facet.48c [concrete]
// CHECK:STDOUT:   %complete_type.c6d: <witness> = complete_type_witness %DivWith.type.35b [concrete]
// CHECK:STDOUT:   %DivWith.facet.b79: %DivWith.type.35b = facet_value %i16, (%custom_witness.3be) [concrete]
// CHECK:STDOUT:   %DivWith.WithSelf.Op.type.b57: type = fn_type @DivWith.WithSelf.Op, @DivWith.WithSelf(%Int16, %DivWith.facet.b79) [concrete]
// CHECK:STDOUT:   %.c89: type = fn_type_with_self_type %DivWith.WithSelf.Op.type.b57, %DivWith.facet.b79 [concrete]
// CHECK:STDOUT:   %complete_type.55f: <witness> = complete_type_witness %ModWith.type.3ef [concrete]
// CHECK:STDOUT:   %ModWith.facet.8a9: %ModWith.type.3ef = facet_value %i16, (%custom_witness.dcf) [concrete]
// CHECK:STDOUT:   %ModWith.WithSelf.Op.type.06b: type = fn_type @ModWith.WithSelf.Op, @ModWith.WithSelf(%Int16, %ModWith.facet.8a9) [concrete]
// CHECK:STDOUT:   %.d15: type = fn_type_with_self_type %ModWith.WithSelf.Op.type.06b, %ModWith.facet.8a9 [concrete]
// CHECK:STDOUT:   %complete_type.def: <witness> = complete_type_witness %AddWith.type.a6e [concrete]
// CHECK:STDOUT:   %AddWith.facet.ea1: %AddWith.type.a6e = facet_value %Int16, (%custom_witness.0db) [concrete]
// CHECK:STDOUT:   %AddWith.WithSelf.Op.type.c4d: type = fn_type @AddWith.WithSelf.Op, @AddWith.WithSelf(%i16, %AddWith.facet.ea1) [concrete]
// CHECK:STDOUT:   %.5e8: type = fn_type_with_self_type %AddWith.WithSelf.Op.type.c4d, %AddWith.facet.ea1 [concrete]
// CHECK:STDOUT:   %complete_type.8c1: <witness> = complete_type_witness %SubWith.type.2f5 [concrete]
// CHECK:STDOUT:   %SubWith.facet.b1e: %SubWith.type.2f5 = facet_value %Int16, (%custom_witness.2f4) [concrete]
// CHECK:STDOUT:   %SubWith.WithSelf.Op.type.fa8: type = fn_type @SubWith.WithSelf.Op, @SubWith.WithSelf(%i16, %SubWith.facet.b1e) [concrete]
// CHECK:STDOUT:   %.26c: type = fn_type_with_self_type %SubWith.WithSelf.Op.type.fa8, %SubWith.facet.b1e [concrete]
// CHECK:STDOUT:   %complete_type.514: <witness> = complete_type_witness %MulWith.type.278 [concrete]
// CHECK:STDOUT:   %MulWith.facet.391: %MulWith.type.278 = facet_value %Int16, (%custom_witness.d98) [concrete]
// CHECK:STDOUT:   %MulWith.WithSelf.Op.type.f65: type = fn_type @MulWith.WithSelf.Op, @MulWith.WithSelf(%i16, %MulWith.facet.391) [concrete]
// CHECK:STDOUT:   %.796: type = fn_type_with_self_type %MulWith.WithSelf.Op.type.f65, %MulWith.facet.391 [concrete]
// CHECK:STDOUT:   %complete_type.51d: <witness> = complete_type_witness %DivWith.type.9b1 [concrete]
// CHECK:STDOUT:   %DivWith.facet.b5b: %DivWith.type.9b1 = facet_value %Int16, (%custom_witness.006) [concrete]
// CHECK:STDOUT:   %DivWith.WithSelf.Op.type.fb7: type = fn_type @DivWith.WithSelf.Op, @DivWith.WithSelf(%i16, %DivWith.facet.b5b) [concrete]
// CHECK:STDOUT:   %.14a: type = fn_type_with_self_type %DivWith.WithSelf.Op.type.fb7, %DivWith.facet.b5b [concrete]
// CHECK:STDOUT:   %complete_type.155: <witness> = complete_type_witness %ModWith.type.59b [concrete]
// CHECK:STDOUT:   %ModWith.facet.cde: %ModWith.type.59b = facet_value %Int16, (%custom_witness.e08) [concrete]
// CHECK:STDOUT:   %ModWith.WithSelf.Op.type.f06: type = fn_type @ModWith.WithSelf.Op, @ModWith.WithSelf(%i16, %ModWith.facet.cde) [concrete]
// CHECK:STDOUT:   %.7a7: type = fn_type_with_self_type %ModWith.WithSelf.Op.type.f06, %ModWith.facet.cde [concrete]
// CHECK:STDOUT:   %complete_type.f05: <witness> = complete_type_witness %AddWith.type.c13 [concrete]
// CHECK:STDOUT:   %AddWith.facet.013: %AddWith.type.c13 = facet_value %Int16, (%custom_witness.27d802.1) [concrete]
// CHECK:STDOUT:   %AddWith.WithSelf.Op.type.c77: type = fn_type @AddWith.WithSelf.Op, @AddWith.WithSelf(%Int32, %AddWith.facet.013) [concrete]
// CHECK:STDOUT:   %.396: type = fn_type_with_self_type %AddWith.WithSelf.Op.type.c77, %AddWith.facet.013 [concrete]
// CHECK:STDOUT:   %complete_type.60a: <witness> = complete_type_witness %SubWith.type.554 [concrete]
// CHECK:STDOUT:   %SubWith.facet.cc1: %SubWith.type.554 = facet_value %Int16, (%custom_witness.f4a1b8.1) [concrete]
// CHECK:STDOUT:   %SubWith.WithSelf.Op.type.3ad: type = fn_type @SubWith.WithSelf.Op, @SubWith.WithSelf(%Int32, %SubWith.facet.cc1) [concrete]
// CHECK:STDOUT:   %.ed8: type = fn_type_with_self_type %SubWith.WithSelf.Op.type.3ad, %SubWith.facet.cc1 [concrete]
// CHECK:STDOUT:   %complete_type.e35: <witness> = complete_type_witness %MulWith.type.d6e [concrete]
// CHECK:STDOUT:   %MulWith.facet.ad3: %MulWith.type.d6e = facet_value %Int16, (%custom_witness.cbf765.1) [concrete]
// CHECK:STDOUT:   %MulWith.WithSelf.Op.type.39c: type = fn_type @MulWith.WithSelf.Op, @MulWith.WithSelf(%Int32, %MulWith.facet.ad3) [concrete]
// CHECK:STDOUT:   %.71b3: type = fn_type_with_self_type %MulWith.WithSelf.Op.type.39c, %MulWith.facet.ad3 [concrete]
// CHECK:STDOUT:   %complete_type.fbd: <witness> = complete_type_witness %DivWith.type.0f5 [concrete]
// CHECK:STDOUT:   %DivWith.facet.e46: %DivWith.type.0f5 = facet_value %Int16, (%custom_witness.ec9d25.1) [concrete]
// CHECK:STDOUT:   %DivWith.WithSelf.Op.type.fa5: type = fn_type @DivWith.WithSelf.Op, @DivWith.WithSelf(%Int32, %DivWith.facet.e46) [concrete]
// CHECK:STDOUT:   %.755: type = fn_type_with_self_type %DivWith.WithSelf.Op.type.fa5, %DivWith.facet.e46 [concrete]
// CHECK:STDOUT:   %complete_type.965: <witness> = complete_type_witness %ModWith.type.e59 [concrete]
// CHECK:STDOUT:   %ModWith.facet.c8c: %ModWith.type.e59 = facet_value %Int16, (%custom_witness.998647.1) [concrete]
// CHECK:STDOUT:   %ModWith.WithSelf.Op.type.5ca: type = fn_type @ModWith.WithSelf.Op, @ModWith.WithSelf(%Int32, %ModWith.facet.c8c) [concrete]
// CHECK:STDOUT:   %.98c: type = fn_type_with_self_type %ModWith.WithSelf.Op.type.5ca, %ModWith.facet.c8c [concrete]
// CHECK:STDOUT:   %AddWith.facet.438: %AddWith.type.116 = facet_value %Int32, (%custom_witness.6f6) [concrete]
// CHECK:STDOUT:   %AddWith.WithSelf.Op.type.f56: type = fn_type @AddWith.WithSelf.Op, @AddWith.WithSelf(%Int16, %AddWith.facet.438) [concrete]
// CHECK:STDOUT:   %.dcc: type = fn_type_with_self_type %AddWith.WithSelf.Op.type.f56, %AddWith.facet.438 [concrete]
// CHECK:STDOUT:   %SubWith.facet.0d0: %SubWith.type.49a = facet_value %Int32, (%custom_witness.97b) [concrete]
// CHECK:STDOUT:   %SubWith.WithSelf.Op.type.093: type = fn_type @SubWith.WithSelf.Op, @SubWith.WithSelf(%Int16, %SubWith.facet.0d0) [concrete]
// CHECK:STDOUT:   %.ef68: type = fn_type_with_self_type %SubWith.WithSelf.Op.type.093, %SubWith.facet.0d0 [concrete]
// CHECK:STDOUT:   %MulWith.facet.b0d: %MulWith.type.779 = facet_value %Int32, (%custom_witness.ac6) [concrete]
// CHECK:STDOUT:   %MulWith.WithSelf.Op.type.27b: type = fn_type @MulWith.WithSelf.Op, @MulWith.WithSelf(%Int16, %MulWith.facet.b0d) [concrete]
// CHECK:STDOUT:   %.5ae: type = fn_type_with_self_type %MulWith.WithSelf.Op.type.27b, %MulWith.facet.b0d [concrete]
// CHECK:STDOUT:   %DivWith.facet.0ba: %DivWith.type.35b = facet_value %Int32, (%custom_witness.387) [concrete]
// CHECK:STDOUT:   %DivWith.WithSelf.Op.type.798: type = fn_type @DivWith.WithSelf.Op, @DivWith.WithSelf(%Int16, %DivWith.facet.0ba) [concrete]
// CHECK:STDOUT:   %.314: type = fn_type_with_self_type %DivWith.WithSelf.Op.type.798, %DivWith.facet.0ba [concrete]
// CHECK:STDOUT:   %ModWith.facet.f7c: %ModWith.type.3ef = facet_value %Int32, (%custom_witness.5cd) [concrete]
// CHECK:STDOUT:   %ModWith.WithSelf.Op.type.e0b: type = fn_type @ModWith.WithSelf.Op, @ModWith.WithSelf(%Int16, %ModWith.facet.f7c) [concrete]
// CHECK:STDOUT:   %.ed5: type = fn_type_with_self_type %ModWith.WithSelf.Op.type.e0b, %ModWith.facet.f7c [concrete]
// CHECK:STDOUT:   %AddWith.facet.524: %AddWith.type.c13 = facet_value %Int32, (%custom_witness.27d802.2) [concrete]
// CHECK:STDOUT:   %AddWith.WithSelf.Op.type.56f: type = fn_type @AddWith.WithSelf.Op, @AddWith.WithSelf(%Int32, %AddWith.facet.524) [concrete]
// CHECK:STDOUT:   %.57b: type = fn_type_with_self_type %AddWith.WithSelf.Op.type.56f, %AddWith.facet.524 [concrete]
// CHECK:STDOUT:   %SubWith.facet.304: %SubWith.type.554 = facet_value %Int32, (%custom_witness.f4a1b8.2) [concrete]
// CHECK:STDOUT:   %SubWith.WithSelf.Op.type.dc6: type = fn_type @SubWith.WithSelf.Op, @SubWith.WithSelf(%Int32, %SubWith.facet.304) [concrete]
// CHECK:STDOUT:   %.f43: type = fn_type_with_self_type %SubWith.WithSelf.Op.type.dc6, %SubWith.facet.304 [concrete]
// CHECK:STDOUT:   %MulWith.facet.4d2: %MulWith.type.d6e = facet_value %Int32, (%custom_witness.cbf765.2) [concrete]
// CHECK:STDOUT:   %MulWith.WithSelf.Op.type.6e0: type = fn_type @MulWith.WithSelf.Op, @MulWith.WithSelf(%Int32, %MulWith.facet.4d2) [concrete]
// CHECK:STDOUT:   %.c8a: type = fn_type_with_self_type %MulWith.WithSelf.Op.type.6e0, %MulWith.facet.4d2 [concrete]
// CHECK:STDOUT:   %DivWith.facet.47e: %DivWith.type.0f5 = facet_value %Int32, (%custom_witness.ec9d25.2) [concrete]
// CHECK:STDOUT:   %DivWith.WithSelf.Op.type.489: type = fn_type @DivWith.WithSelf.Op, @DivWith.WithSelf(%Int32, %DivWith.facet.47e) [concrete]
// CHECK:STDOUT:   %.9129: type = fn_type_with_self_type %DivWith.WithSelf.Op.type.489, %DivWith.facet.47e [concrete]
// CHECK:STDOUT:   %ModWith.facet.217: %ModWith.type.e59 = facet_value %Int32, (%custom_witness.998647.2) [concrete]
// CHECK:STDOUT:   %ModWith.WithSelf.Op.type.c61: type = fn_type @ModWith.WithSelf.Op, @ModWith.WithSelf(%Int32, %ModWith.facet.217) [concrete]
// CHECK:STDOUT:   %.512: type = fn_type_with_self_type %ModWith.WithSelf.Op.type.c61, %ModWith.facet.217 [concrete]
// CHECK:STDOUT:   %complete_type.905: <witness> = complete_type_witness %AddWith.type.b5c [concrete]
// CHECK:STDOUT:   %AddWith.facet.d6a: %AddWith.type.b5c = facet_value %Int32, (%custom_witness.e14) [concrete]
// CHECK:STDOUT:   %AddWith.WithSelf.Op.type.815: type = fn_type @AddWith.WithSelf.Op, @AddWith.WithSelf(%Int64, %AddWith.facet.d6a) [concrete]
// CHECK:STDOUT:   %.c1e: type = fn_type_with_self_type %AddWith.WithSelf.Op.type.815, %AddWith.facet.d6a [concrete]
// CHECK:STDOUT:   %complete_type.7e8: <witness> = complete_type_witness %SubWith.type.4c4 [concrete]
// CHECK:STDOUT:   %SubWith.facet.bd2: %SubWith.type.4c4 = facet_value %Int32, (%custom_witness.d0e) [concrete]
// CHECK:STDOUT:   %SubWith.WithSelf.Op.type.6af: type = fn_type @SubWith.WithSelf.Op, @SubWith.WithSelf(%Int64, %SubWith.facet.bd2) [concrete]
// CHECK:STDOUT:   %.b8f: type = fn_type_with_self_type %SubWith.WithSelf.Op.type.6af, %SubWith.facet.bd2 [concrete]
// CHECK:STDOUT:   %complete_type.3a2: <witness> = complete_type_witness %MulWith.type.6fc [concrete]
// CHECK:STDOUT:   %MulWith.facet.806: %MulWith.type.6fc = facet_value %Int32, (%custom_witness.784) [concrete]
// CHECK:STDOUT:   %MulWith.WithSelf.Op.type.96e: type = fn_type @MulWith.WithSelf.Op, @MulWith.WithSelf(%Int64, %MulWith.facet.806) [concrete]
// CHECK:STDOUT:   %.a56: type = fn_type_with_self_type %MulWith.WithSelf.Op.type.96e, %MulWith.facet.806 [concrete]
// CHECK:STDOUT:   %complete_type.504: <witness> = complete_type_witness %DivWith.type.46b [concrete]
// CHECK:STDOUT:   %DivWith.facet.3c3: %DivWith.type.46b = facet_value %Int32, (%custom_witness.a1d) [concrete]
// CHECK:STDOUT:   %DivWith.WithSelf.Op.type.bde: type = fn_type @DivWith.WithSelf.Op, @DivWith.WithSelf(%Int64, %DivWith.facet.3c3) [concrete]
// CHECK:STDOUT:   %.27f: type = fn_type_with_self_type %DivWith.WithSelf.Op.type.bde, %DivWith.facet.3c3 [concrete]
// CHECK:STDOUT:   %complete_type.52f: <witness> = complete_type_witness %ModWith.type.610 [concrete]
// CHECK:STDOUT:   %ModWith.facet.e04: %ModWith.type.610 = facet_value %Int32, (%custom_witness.1f5) [concrete]
// CHECK:STDOUT:   %ModWith.WithSelf.Op.type.abf: type = fn_type @ModWith.WithSelf.Op, @ModWith.WithSelf(%Int64, %ModWith.facet.e04) [concrete]
// CHECK:STDOUT:   %.22f: type = fn_type_with_self_type %ModWith.WithSelf.Op.type.abf, %ModWith.facet.e04 [concrete]
// CHECK:STDOUT:   %AddWith.facet.fd0: %AddWith.type.c13 = facet_value %Int64, (%custom_witness.8f1) [concrete]
// CHECK:STDOUT:   %AddWith.WithSelf.Op.type.3e0: type = fn_type @AddWith.WithSelf.Op, @AddWith.WithSelf(%Int32, %AddWith.facet.fd0) [concrete]
// CHECK:STDOUT:   %.800: type = fn_type_with_self_type %AddWith.WithSelf.Op.type.3e0, %AddWith.facet.fd0 [concrete]
// CHECK:STDOUT:   %SubWith.facet.09f: %SubWith.type.554 = facet_value %Int64, (%custom_witness.148) [concrete]
// CHECK:STDOUT:   %SubWith.WithSelf.Op.type.4fe: type = fn_type @SubWith.WithSelf.Op, @SubWith.WithSelf(%Int32, %SubWith.facet.09f) [concrete]
// CHECK:STDOUT:   %.593: type = fn_type_with_self_type %SubWith.WithSelf.Op.type.4fe, %SubWith.facet.09f [concrete]
// CHECK:STDOUT:   %MulWith.facet.f0b: %MulWith.type.d6e = facet_value %Int64, (%custom_witness.d3e) [concrete]
// CHECK:STDOUT:   %MulWith.WithSelf.Op.type.bf9: type = fn_type @MulWith.WithSelf.Op, @MulWith.WithSelf(%Int32, %MulWith.facet.f0b) [concrete]
// CHECK:STDOUT:   %.f35: type = fn_type_with_self_type %MulWith.WithSelf.Op.type.bf9, %MulWith.facet.f0b [concrete]
// CHECK:STDOUT:   %DivWith.facet.756: %DivWith.type.0f5 = facet_value %Int64, (%custom_witness.847) [concrete]
// CHECK:STDOUT:   %DivWith.WithSelf.Op.type.4be: type = fn_type @DivWith.WithSelf.Op, @DivWith.WithSelf(%Int32, %DivWith.facet.756) [concrete]
// CHECK:STDOUT:   %.eae: type = fn_type_with_self_type %DivWith.WithSelf.Op.type.4be, %DivWith.facet.756 [concrete]
// CHECK:STDOUT:   %ModWith.facet.735: %ModWith.type.e59 = facet_value %Int64, (%custom_witness.4aa) [concrete]
// CHECK:STDOUT:   %ModWith.WithSelf.Op.type.e0f: type = fn_type @ModWith.WithSelf.Op, @ModWith.WithSelf(%Int32, %ModWith.facet.735) [concrete]
// CHECK:STDOUT:   %.0ab: type = fn_type_with_self_type %ModWith.WithSelf.Op.type.e0f, %ModWith.facet.735 [concrete]
// CHECK:STDOUT:   %complete_type.c3b: <witness> = complete_type_witness %AddAssignWith.type.5a5 [concrete]
// CHECK:STDOUT:   %AddAssignWith.WithSelf.Op.type.0f1: type = fn_type @AddAssignWith.WithSelf.Op, @AddAssignWith.WithSelf(%i16, %AddAssignWith.facet.b45) [concrete]
// CHECK:STDOUT:   %.dd3: type = fn_type_with_self_type %AddAssignWith.WithSelf.Op.type.0f1, %AddAssignWith.facet.b45 [concrete]
// CHECK:STDOUT:   %complete_type.289: <witness> = complete_type_witness %SubAssignWith.type.2da [concrete]
// CHECK:STDOUT:   %SubAssignWith.WithSelf.Op.type.286: type = fn_type @SubAssignWith.WithSelf.Op, @SubAssignWith.WithSelf(%i16, %SubAssignWith.facet.351) [concrete]
// CHECK:STDOUT:   %.d9d: type = fn_type_with_self_type %SubAssignWith.WithSelf.Op.type.286, %SubAssignWith.facet.351 [concrete]
// CHECK:STDOUT:   %complete_type.9b6: <witness> = complete_type_witness %MulAssignWith.type.a70 [concrete]
// CHECK:STDOUT:   %MulAssignWith.WithSelf.Op.type.29b: type = fn_type @MulAssignWith.WithSelf.Op, @MulAssignWith.WithSelf(%i16, %MulAssignWith.facet.2aa) [concrete]
// CHECK:STDOUT:   %.a28: type = fn_type_with_self_type %MulAssignWith.WithSelf.Op.type.29b, %MulAssignWith.facet.2aa [concrete]
// CHECK:STDOUT:   %complete_type.41f: <witness> = complete_type_witness %DivAssignWith.type.7f8 [concrete]
// CHECK:STDOUT:   %DivAssignWith.WithSelf.Op.type.604: type = fn_type @DivAssignWith.WithSelf.Op, @DivAssignWith.WithSelf(%i16, %DivAssignWith.facet.253) [concrete]
// CHECK:STDOUT:   %.0f0: type = fn_type_with_self_type %DivAssignWith.WithSelf.Op.type.604, %DivAssignWith.facet.253 [concrete]
// CHECK:STDOUT:   %complete_type.955: <witness> = complete_type_witness %ModAssignWith.type.eee [concrete]
// CHECK:STDOUT:   %ModAssignWith.WithSelf.Op.type.436: type = fn_type @ModAssignWith.WithSelf.Op, @ModAssignWith.WithSelf(%i16, %ModAssignWith.facet.c21) [concrete]
// CHECK:STDOUT:   %.8a6: type = fn_type_with_self_type %ModAssignWith.WithSelf.Op.type.436, %ModAssignWith.facet.c21 [concrete]
// CHECK:STDOUT:   %complete_type.547: <witness> = complete_type_witness %AddAssignWith.type.22f [concrete]
// CHECK:STDOUT:   %AddAssignWith.WithSelf.Op.type.417: type = fn_type @AddAssignWith.WithSelf.Op, @AddAssignWith.WithSelf(%Int32, %AddAssignWith.facet.e5f) [concrete]
// CHECK:STDOUT:   %.c71: type = fn_type_with_self_type %AddAssignWith.WithSelf.Op.type.417, %AddAssignWith.facet.e5f [concrete]
// CHECK:STDOUT:   %complete_type.d8e: <witness> = complete_type_witness %SubAssignWith.type.c0c [concrete]
// CHECK:STDOUT:   %SubAssignWith.WithSelf.Op.type.d08: type = fn_type @SubAssignWith.WithSelf.Op, @SubAssignWith.WithSelf(%Int32, %SubAssignWith.facet.c12) [concrete]
// CHECK:STDOUT:   %.8b4: type = fn_type_with_self_type %SubAssignWith.WithSelf.Op.type.d08, %SubAssignWith.facet.c12 [concrete]
// CHECK:STDOUT:   %complete_type.4e2: <witness> = complete_type_witness %MulAssignWith.type.f1b [concrete]
// CHECK:STDOUT:   %MulAssignWith.WithSelf.Op.type.e8b: type = fn_type @MulAssignWith.WithSelf.Op, @MulAssignWith.WithSelf(%Int32, %MulAssignWith.facet.7cb) [concrete]
// CHECK:STDOUT:   %.d1f: type = fn_type_with_self_type %MulAssignWith.WithSelf.Op.type.e8b, %MulAssignWith.facet.7cb [concrete]
// CHECK:STDOUT:   %complete_type.463: <witness> = complete_type_witness %DivAssignWith.type.453 [concrete]
// CHECK:STDOUT:   %DivAssignWith.WithSelf.Op.type.f70: type = fn_type @DivAssignWith.WithSelf.Op, @DivAssignWith.WithSelf(%Int32, %DivAssignWith.facet.ffe) [concrete]
// CHECK:STDOUT:   %.2b1: type = fn_type_with_self_type %DivAssignWith.WithSelf.Op.type.f70, %DivAssignWith.facet.ffe [concrete]
// CHECK:STDOUT:   %complete_type.e8b: <witness> = complete_type_witness %ModAssignWith.type.390 [concrete]
// CHECK:STDOUT:   %ModAssignWith.WithSelf.Op.type.ac2: type = fn_type @ModAssignWith.WithSelf.Op, @ModAssignWith.WithSelf(%Int32, %ModAssignWith.facet.1bc) [concrete]
// CHECK:STDOUT:   %.e93: type = fn_type_with_self_type %ModAssignWith.WithSelf.Op.type.ac2, %ModAssignWith.facet.1bc [concrete]
// CHECK:STDOUT:   %AddAssignWith.WithSelf.Op.type.eaa: type = fn_type @AddAssignWith.WithSelf.Op, @AddAssignWith.WithSelf(%Int32, %AddAssignWith.facet.62e) [concrete]
// CHECK:STDOUT:   %.869: type = fn_type_with_self_type %AddAssignWith.WithSelf.Op.type.eaa, %AddAssignWith.facet.62e [concrete]
// CHECK:STDOUT:   %SubAssignWith.WithSelf.Op.type.6f7: type = fn_type @SubAssignWith.WithSelf.Op, @SubAssignWith.WithSelf(%Int32, %SubAssignWith.facet.1f9) [concrete]
// CHECK:STDOUT:   %.792: type = fn_type_with_self_type %SubAssignWith.WithSelf.Op.type.6f7, %SubAssignWith.facet.1f9 [concrete]
// CHECK:STDOUT:   %MulAssignWith.WithSelf.Op.type.830: type = fn_type @MulAssignWith.WithSelf.Op, @MulAssignWith.WithSelf(%Int32, %MulAssignWith.facet.b47) [concrete]
// CHECK:STDOUT:   %.e4c: type = fn_type_with_self_type %MulAssignWith.WithSelf.Op.type.830, %MulAssignWith.facet.b47 [concrete]
// CHECK:STDOUT:   %DivAssignWith.WithSelf.Op.type.494: type = fn_type @DivAssignWith.WithSelf.Op, @DivAssignWith.WithSelf(%Int32, %DivAssignWith.facet.f27) [concrete]
// CHECK:STDOUT:   %.55b: type = fn_type_with_self_type %DivAssignWith.WithSelf.Op.type.494, %DivAssignWith.facet.f27 [concrete]
// CHECK:STDOUT:   %ModAssignWith.WithSelf.Op.type.bed: type = fn_type @ModAssignWith.WithSelf.Op, @ModAssignWith.WithSelf(%Int32, %ModAssignWith.facet.a21) [concrete]
// CHECK:STDOUT:   %.074: type = fn_type_with_self_type %ModAssignWith.WithSelf.Op.type.bed, %ModAssignWith.facet.a21 [concrete]
// CHECK:STDOUT:   %Inc.WithSelf.Op.type.f93: type = fn_type @Inc.WithSelf.Op, @Inc.WithSelf(%Inc.facet.338) [concrete]
// CHECK:STDOUT:   %.fb7: type = fn_type_with_self_type %Inc.WithSelf.Op.type.f93, %Inc.facet.338 [concrete]
// CHECK:STDOUT:   %Dec.WithSelf.Op.type.221: type = fn_type @Dec.WithSelf.Op, @Dec.WithSelf(%Dec.facet.b97) [concrete]
// CHECK:STDOUT:   %.2d8: type = fn_type_with_self_type %Dec.WithSelf.Op.type.221, %Dec.facet.b97 [concrete]
// CHECK:STDOUT:   %Negate.facet.7e2: %Negate.type = facet_value %Int16, (%custom_witness.243) [concrete]
// CHECK:STDOUT:   %Negate.WithSelf.Op.type.394: type = fn_type @Negate.WithSelf.Op, @Negate.WithSelf(%Negate.facet.7e2) [concrete]
// CHECK:STDOUT:   %.187: type = fn_type_with_self_type %Negate.WithSelf.Op.type.394, %Negate.facet.7e2 [concrete]
// CHECK:STDOUT:   %Negate.facet.41a: %Negate.type = facet_value %Int32, (%custom_witness.9ae) [concrete]
// CHECK:STDOUT:   %Negate.WithSelf.Op.type.81f: type = fn_type @Negate.WithSelf.Op, @Negate.WithSelf(%Negate.facet.41a) [concrete]
// CHECK:STDOUT:   %.71b4: type = fn_type_with_self_type %Negate.WithSelf.Op.type.81f, %Negate.facet.41a [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.010: @AddWith.WithSelf.%AddWith.assoc_type (%AddWith.assoc_type.dfc) = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [symbolic = @AddWith.WithSelf.%assoc1 (constants.%assoc1.593)]
// CHECK:STDOUT:   %Core.import_ref.935: @AddWith.WithSelf.%AddWith.WithSelf.Op.type (%AddWith.WithSelf.Op.type.fc0) = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [symbolic = @AddWith.WithSelf.%AddWith.WithSelf.Op (constants.%AddWith.WithSelf.Op.056)]
// CHECK:STDOUT:   %Core.import_ref.b37: type = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [concrete = %Result.c43]
// CHECK:STDOUT:   %Result.c43: type = assoc_const_decl @Result.1 [concrete] {}
// CHECK:STDOUT:   %Core.import_ref.31c = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.cbb: @SubWith.WithSelf.%SubWith.assoc_type (%SubWith.assoc_type.6b2) = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [symbolic = @SubWith.WithSelf.%assoc1 (constants.%assoc1.8b9)]
// CHECK:STDOUT:   %Core.import_ref.480: @SubWith.WithSelf.%SubWith.WithSelf.Op.type (%SubWith.WithSelf.Op.type.2ca) = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [symbolic = @SubWith.WithSelf.%SubWith.WithSelf.Op (constants.%SubWith.WithSelf.Op.918)]
// CHECK:STDOUT:   %Core.import_ref.bb2: type = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [concrete = %Result.d9b]
// CHECK:STDOUT:   %Result.d9b: type = assoc_const_decl @Result.2 [concrete] {}
// CHECK:STDOUT:   %Core.import_ref.507 = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.1d4: @MulWith.WithSelf.%MulWith.assoc_type (%MulWith.assoc_type.03e) = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [symbolic = @MulWith.WithSelf.%assoc1 (constants.%assoc1.56c)]
// CHECK:STDOUT:   %Core.import_ref.7ed: @MulWith.WithSelf.%MulWith.WithSelf.Op.type (%MulWith.WithSelf.Op.type.e29) = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [symbolic = @MulWith.WithSelf.%MulWith.WithSelf.Op (constants.%MulWith.WithSelf.Op.45b)]
// CHECK:STDOUT:   %Core.import_ref.a1a: type = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [concrete = %Result.f7f]
// CHECK:STDOUT:   %Result.f7f: type = assoc_const_decl @Result.3 [concrete] {}
// CHECK:STDOUT:   %Core.import_ref.f63 = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.d16: @DivWith.WithSelf.%DivWith.assoc_type (%DivWith.assoc_type.1ff) = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [symbolic = @DivWith.WithSelf.%assoc1 (constants.%assoc1.71c)]
// CHECK:STDOUT:   %Core.import_ref.44d: @DivWith.WithSelf.%DivWith.WithSelf.Op.type (%DivWith.WithSelf.Op.type.388) = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [symbolic = @DivWith.WithSelf.%DivWith.WithSelf.Op (constants.%DivWith.WithSelf.Op.0ce)]
// CHECK:STDOUT:   %Core.import_ref.b8f: type = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [concrete = %Result.10f]
// CHECK:STDOUT:   %Result.10f: type = assoc_const_decl @Result.4 [concrete] {}
// CHECK:STDOUT:   %Core.import_ref.4ea = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.e51: @ModWith.WithSelf.%ModWith.assoc_type (%ModWith.assoc_type.7ff) = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [symbolic = @ModWith.WithSelf.%assoc1 (constants.%assoc1.e05)]
// CHECK:STDOUT:   %Core.import_ref.556: @ModWith.WithSelf.%ModWith.WithSelf.Op.type (%ModWith.WithSelf.Op.type.f7c) = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [symbolic = @ModWith.WithSelf.%ModWith.WithSelf.Op (constants.%ModWith.WithSelf.Op.a28)]
// CHECK:STDOUT:   %Core.import_ref.f19: type = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [concrete = %Result.a8f]
// CHECK:STDOUT:   %Result.a8f: type = assoc_const_decl @Result.5 [concrete] {}
// CHECK:STDOUT:   %Core.import_ref.f10 = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.eb9: @AddAssignWith.WithSelf.%AddAssignWith.assoc_type (%AddAssignWith.assoc_type.a25160.1) = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [symbolic = @AddAssignWith.WithSelf.%assoc0 (constants.%assoc0.c1f)]
// CHECK:STDOUT:   %Core.import_ref.4eb: @AddAssignWith.WithSelf.%AddAssignWith.WithSelf.Op.type (%AddAssignWith.WithSelf.Op.type.8b23d4.1) = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [symbolic = @AddAssignWith.WithSelf.%AddAssignWith.WithSelf.Op (constants.%AddAssignWith.WithSelf.Op.4cbad4.1)]
// CHECK:STDOUT:   %Core.import_ref.763 = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.7fa: @SubAssignWith.WithSelf.%SubAssignWith.assoc_type (%SubAssignWith.assoc_type.0440e4.1) = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [symbolic = @SubAssignWith.WithSelf.%assoc0 (constants.%assoc0.5f4)]
// CHECK:STDOUT:   %Core.import_ref.a2d0: @SubAssignWith.WithSelf.%SubAssignWith.WithSelf.Op.type (%SubAssignWith.WithSelf.Op.type.ac317f.1) = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [symbolic = @SubAssignWith.WithSelf.%SubAssignWith.WithSelf.Op (constants.%SubAssignWith.WithSelf.Op.b6cd67.1)]
// CHECK:STDOUT:   %Core.import_ref.3c0 = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.dd3: @MulAssignWith.WithSelf.%MulAssignWith.assoc_type (%MulAssignWith.assoc_type.dd8648.1) = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [symbolic = @MulAssignWith.WithSelf.%assoc0 (constants.%assoc0.181)]
// CHECK:STDOUT:   %Core.import_ref.475: @MulAssignWith.WithSelf.%MulAssignWith.WithSelf.Op.type (%MulAssignWith.WithSelf.Op.type.dff752.1) = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [symbolic = @MulAssignWith.WithSelf.%MulAssignWith.WithSelf.Op (constants.%MulAssignWith.WithSelf.Op.28a4dd.1)]
// CHECK:STDOUT:   %Core.import_ref.768 = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.927: @DivAssignWith.WithSelf.%DivAssignWith.assoc_type (%DivAssignWith.assoc_type.738ee6.1) = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [symbolic = @DivAssignWith.WithSelf.%assoc0 (constants.%assoc0.f6f)]
// CHECK:STDOUT:   %Core.import_ref.12d: @DivAssignWith.WithSelf.%DivAssignWith.WithSelf.Op.type (%DivAssignWith.WithSelf.Op.type.3077dd.1) = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [symbolic = @DivAssignWith.WithSelf.%DivAssignWith.WithSelf.Op (constants.%DivAssignWith.WithSelf.Op.710329.1)]
// CHECK:STDOUT:   %Core.import_ref.9c0 = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.ab7: @ModAssignWith.WithSelf.%ModAssignWith.assoc_type (%ModAssignWith.assoc_type.9ba07c.1) = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [symbolic = @ModAssignWith.WithSelf.%assoc0 (constants.%assoc0.26c)]
// CHECK:STDOUT:   %Core.import_ref.c884: @ModAssignWith.WithSelf.%ModAssignWith.WithSelf.Op.type (%ModAssignWith.WithSelf.Op.type.3abf50.1) = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [symbolic = @ModAssignWith.WithSelf.%ModAssignWith.WithSelf.Op (constants.%ModAssignWith.WithSelf.Op.42aedd.1)]
// CHECK:STDOUT:   %Core.import_ref.d41 = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @AddWith.loc124(%Result.loc121_9.2: %Destroy.type, %U.loc122_4.2: %Destroy.type, %T.loc123_4.2: @AddWith.loc124.%facet_type.loc123_38 (%facet_type.d00)) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %require_complete.loc126: <witness> = require_complete_type %AddWith.type.loc123_36.1 [symbolic = %require_complete.loc126 (constants.%require_complete.672)]
// CHECK:STDOUT:   %assoc1: @AddWith.loc124.%AddWith.assoc_type (%AddWith.assoc_type.f48) = assoc_entity element1, imports.%Core.import_ref.935 [symbolic = %assoc1 (constants.%assoc1.bb6)]
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.loc126: <witness> = lookup_impl_witness %T.loc123_4.1, @AddWith.1, @AddWith.1(%U.as_type.loc123_36.1) [symbolic = %AddWith.lookup_impl_witness.loc126 (constants.%AddWith.lookup_impl_witness.211)]
// CHECK:STDOUT:   %AddWith.facet: @AddWith.loc124.%AddWith.type.loc123_36.1 (%AddWith.type.68f) = facet_value %T.as_type.loc124_29.1, (%AddWith.lookup_impl_witness.loc126) [symbolic = %AddWith.facet (constants.%AddWith.facet.57d)]
// CHECK:STDOUT:   %AddWith.WithSelf.Op.type: type = fn_type @AddWith.WithSelf.Op, @AddWith.WithSelf(%U.as_type.loc123_36.1, %AddWith.facet) [symbolic = %AddWith.WithSelf.Op.type (constants.%AddWith.WithSelf.Op.type.b30a)]
// CHECK:STDOUT:   %.loc126_5.4: type = fn_type_with_self_type %AddWith.WithSelf.Op.type, %AddWith.facet [symbolic = %.loc126_5.4 (constants.%.09b)]
// CHECK:STDOUT:   %impl.elem1.loc126_5.2: @AddWith.loc124.%.loc126_5.4 (%.09b) = impl_witness_access %AddWith.lookup_impl_witness.loc126, element1 [symbolic = %impl.elem1.loc126_5.2 (constants.%impl.elem1.4bc)]
// CHECK:STDOUT:   %specific_impl_fn.loc126_5.3: <specific function> = specific_impl_function %impl.elem1.loc126_5.2, @AddWith.WithSelf.Op(%U.as_type.loc123_36.1, %AddWith.facet) [symbolic = %specific_impl_fn.loc126_5.3 (constants.%specific_impl_fn.31d)]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Result.loc121_9.1) [symbolic = %Destroy.WithSelf.Op.type (constants.%Destroy.WithSelf.Op.type.12b49e.2)]
// CHECK:STDOUT:   %.loc126_5.5: type = fn_type_with_self_type %Destroy.WithSelf.Op.type, %Result.loc121_9.1 [symbolic = %.loc126_5.5 (constants.%.9c3)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %Result.loc121_9.1, @Destroy [symbolic = %Destroy.lookup_impl_witness (constants.%Destroy.lookup_impl_witness)]
// CHECK:STDOUT:   %impl.elem0.loc126_5.2: @AddWith.loc124.%.loc126_5.5 (%.9c3) = impl_witness_access %Destroy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc126_5.2 (constants.%impl.elem0.8bd)]
// CHECK:STDOUT:   %specific_impl_fn.loc126_5.4: <specific function> = specific_impl_function %impl.elem0.loc126_5.2, @Destroy.WithSelf.Op(%Result.loc121_9.1) [symbolic = %specific_impl_fn.loc126_5.4 (constants.%specific_impl_fn.8dd)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%result.param: @AddWith.loc124.%Result.as_type.loc123_54.1 (%Result.as_type), %x.param: @AddWith.loc124.%T.as_type.loc124_29.1 (%T.as_type.6fa), %y.param: @AddWith.loc124.%U.as_type.loc123_36.1 (%U.as_type.0fd)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: @AddWith.loc124.%T.as_type.loc124_29.1 (%T.as_type.6fa) = name_ref x, %x
// CHECK:STDOUT:     %y.ref: @AddWith.loc124.%U.as_type.loc123_36.1 (%U.as_type.0fd) = name_ref y, %y
// CHECK:STDOUT:     %AddWith.type.loc126: type = facet_type <@AddWith.1, @AddWith.1(constants.%U.as_type.0fd)> [symbolic = %AddWith.type.loc123_36.1 (constants.%AddWith.type.68f)]
// CHECK:STDOUT:     %.loc126_5.1: @AddWith.loc124.%AddWith.assoc_type (%AddWith.assoc_type.f48) = specific_constant imports.%Core.import_ref.010, @AddWith.WithSelf(constants.%U.as_type.0fd, constants.%Self.39b) [symbolic = %assoc1 (constants.%assoc1.bb6)]
// CHECK:STDOUT:     %Op.ref: @AddWith.loc124.%AddWith.assoc_type (%AddWith.assoc_type.f48) = name_ref Op, %.loc126_5.1 [symbolic = %assoc1 (constants.%assoc1.bb6)]
// CHECK:STDOUT:     %impl.elem1.loc126_5.1: @AddWith.loc124.%.loc126_5.4 (%.09b) = impl_witness_access constants.%AddWith.lookup_impl_witness.211, element1 [symbolic = %impl.elem1.loc126_5.2 (constants.%impl.elem1.4bc)]
// CHECK:STDOUT:     %bound_method.loc126_5.1: <bound method> = bound_method %x.ref, %impl.elem1.loc126_5.1
// CHECK:STDOUT:     %specific_impl_fn.loc126_5.1: <specific function> = specific_impl_function %impl.elem1.loc126_5.1, @AddWith.WithSelf.Op(constants.%U.as_type.0fd, constants.%AddWith.facet.57d) [symbolic = %specific_impl_fn.loc126_5.3 (constants.%specific_impl_fn.31d)]
// CHECK:STDOUT:     %bound_method.loc126_5.2: <bound method> = bound_method %x.ref, %specific_impl_fn.loc126_5.1
// CHECK:STDOUT:     %.loc126_5.2: ref @AddWith.loc124.%Result.as_type.loc123_54.1 (%Result.as_type) = temporary_storage
// CHECK:STDOUT:     %AddWith.WithSelf.Op.call: init @AddWith.loc124.%Result.as_type.loc123_54.1 (%Result.as_type) to %.loc126_5.2 = call %bound_method.loc126_5.2(%x.ref, %y.ref)
// CHECK:STDOUT:     %.loc126_5.3: ref @AddWith.loc124.%Result.as_type.loc123_54.1 (%Result.as_type) = temporary %.loc126_5.2, %AddWith.WithSelf.Op.call
// CHECK:STDOUT:     %impl.elem0.loc126_5.1: @AddWith.loc124.%.loc126_5.5 (%.9c3) = impl_witness_access constants.%Destroy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc126_5.2 (constants.%impl.elem0.8bd)]
// CHECK:STDOUT:     %bound_method.loc126_5.3: <bound method> = bound_method %.loc126_5.3, %impl.elem0.loc126_5.1
// CHECK:STDOUT:     %specific_impl_fn.loc126_5.2: <specific function> = specific_impl_function %impl.elem0.loc126_5.1, @Destroy.WithSelf.Op(constants.%Result) [symbolic = %specific_impl_fn.loc126_5.4 (constants.%specific_impl_fn.8dd)]
// CHECK:STDOUT:     %bound_method.loc126_5.4: <bound method> = bound_method %.loc126_5.3, %specific_impl_fn.loc126_5.2
// CHECK:STDOUT:     %Destroy.WithSelf.Op.call: init %empty_tuple.type = call %bound_method.loc126_5.4(%.loc126_5.3)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @SubWith.loc135(%Result.loc132_9.2: %Destroy.type, %U.loc133_4.2: %Destroy.type, %T.loc134_4.2: @SubWith.loc135.%facet_type.loc134_38 (%facet_type.de3)) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %require_complete.loc137: <witness> = require_complete_type %SubWith.type.loc134_36.1 [symbolic = %require_complete.loc137 (constants.%require_complete.f2d)]
// CHECK:STDOUT:   %assoc1: @SubWith.loc135.%SubWith.assoc_type (%SubWith.assoc_type.f11) = assoc_entity element1, imports.%Core.import_ref.480 [symbolic = %assoc1 (constants.%assoc1.8f3)]
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.loc137: <witness> = lookup_impl_witness %T.loc134_4.1, @SubWith.1, @SubWith.1(%U.as_type.loc134_36.1) [symbolic = %SubWith.lookup_impl_witness.loc137 (constants.%SubWith.lookup_impl_witness.9fd)]
// CHECK:STDOUT:   %SubWith.facet: @SubWith.loc135.%SubWith.type.loc134_36.1 (%SubWith.type.3e0) = facet_value %T.as_type.loc135_29.1, (%SubWith.lookup_impl_witness.loc137) [symbolic = %SubWith.facet (constants.%SubWith.facet.1b3)]
// CHECK:STDOUT:   %SubWith.WithSelf.Op.type: type = fn_type @SubWith.WithSelf.Op, @SubWith.WithSelf(%U.as_type.loc134_36.1, %SubWith.facet) [symbolic = %SubWith.WithSelf.Op.type (constants.%SubWith.WithSelf.Op.type.649)]
// CHECK:STDOUT:   %.loc137_5.4: type = fn_type_with_self_type %SubWith.WithSelf.Op.type, %SubWith.facet [symbolic = %.loc137_5.4 (constants.%.e46)]
// CHECK:STDOUT:   %impl.elem1.loc137_5.2: @SubWith.loc135.%.loc137_5.4 (%.e46) = impl_witness_access %SubWith.lookup_impl_witness.loc137, element1 [symbolic = %impl.elem1.loc137_5.2 (constants.%impl.elem1.d3b)]
// CHECK:STDOUT:   %specific_impl_fn.loc137_5.3: <specific function> = specific_impl_function %impl.elem1.loc137_5.2, @SubWith.WithSelf.Op(%U.as_type.loc134_36.1, %SubWith.facet) [symbolic = %specific_impl_fn.loc137_5.3 (constants.%specific_impl_fn.d50)]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Result.loc132_9.1) [symbolic = %Destroy.WithSelf.Op.type (constants.%Destroy.WithSelf.Op.type.12b49e.2)]
// CHECK:STDOUT:   %.loc137_5.5: type = fn_type_with_self_type %Destroy.WithSelf.Op.type, %Result.loc132_9.1 [symbolic = %.loc137_5.5 (constants.%.9c3)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %Result.loc132_9.1, @Destroy [symbolic = %Destroy.lookup_impl_witness (constants.%Destroy.lookup_impl_witness)]
// CHECK:STDOUT:   %impl.elem0.loc137_5.2: @SubWith.loc135.%.loc137_5.5 (%.9c3) = impl_witness_access %Destroy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc137_5.2 (constants.%impl.elem0.8bd)]
// CHECK:STDOUT:   %specific_impl_fn.loc137_5.4: <specific function> = specific_impl_function %impl.elem0.loc137_5.2, @Destroy.WithSelf.Op(%Result.loc132_9.1) [symbolic = %specific_impl_fn.loc137_5.4 (constants.%specific_impl_fn.8dd)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%result.param: @SubWith.loc135.%Result.as_type.loc134_54.1 (%Result.as_type), %x.param: @SubWith.loc135.%T.as_type.loc135_29.1 (%T.as_type.3b2), %y.param: @SubWith.loc135.%U.as_type.loc134_36.1 (%U.as_type.0fd)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: @SubWith.loc135.%T.as_type.loc135_29.1 (%T.as_type.3b2) = name_ref x, %x
// CHECK:STDOUT:     %y.ref: @SubWith.loc135.%U.as_type.loc134_36.1 (%U.as_type.0fd) = name_ref y, %y
// CHECK:STDOUT:     %SubWith.type.loc137: type = facet_type <@SubWith.1, @SubWith.1(constants.%U.as_type.0fd)> [symbolic = %SubWith.type.loc134_36.1 (constants.%SubWith.type.3e0)]
// CHECK:STDOUT:     %.loc137_5.1: @SubWith.loc135.%SubWith.assoc_type (%SubWith.assoc_type.f11) = specific_constant imports.%Core.import_ref.cbb, @SubWith.WithSelf(constants.%U.as_type.0fd, constants.%Self.631) [symbolic = %assoc1 (constants.%assoc1.8f3)]
// CHECK:STDOUT:     %Op.ref: @SubWith.loc135.%SubWith.assoc_type (%SubWith.assoc_type.f11) = name_ref Op, %.loc137_5.1 [symbolic = %assoc1 (constants.%assoc1.8f3)]
// CHECK:STDOUT:     %impl.elem1.loc137_5.1: @SubWith.loc135.%.loc137_5.4 (%.e46) = impl_witness_access constants.%SubWith.lookup_impl_witness.9fd, element1 [symbolic = %impl.elem1.loc137_5.2 (constants.%impl.elem1.d3b)]
// CHECK:STDOUT:     %bound_method.loc137_5.1: <bound method> = bound_method %x.ref, %impl.elem1.loc137_5.1
// CHECK:STDOUT:     %specific_impl_fn.loc137_5.1: <specific function> = specific_impl_function %impl.elem1.loc137_5.1, @SubWith.WithSelf.Op(constants.%U.as_type.0fd, constants.%SubWith.facet.1b3) [symbolic = %specific_impl_fn.loc137_5.3 (constants.%specific_impl_fn.d50)]
// CHECK:STDOUT:     %bound_method.loc137_5.2: <bound method> = bound_method %x.ref, %specific_impl_fn.loc137_5.1
// CHECK:STDOUT:     %.loc137_5.2: ref @SubWith.loc135.%Result.as_type.loc134_54.1 (%Result.as_type) = temporary_storage
// CHECK:STDOUT:     %SubWith.WithSelf.Op.call: init @SubWith.loc135.%Result.as_type.loc134_54.1 (%Result.as_type) to %.loc137_5.2 = call %bound_method.loc137_5.2(%x.ref, %y.ref)
// CHECK:STDOUT:     %.loc137_5.3: ref @SubWith.loc135.%Result.as_type.loc134_54.1 (%Result.as_type) = temporary %.loc137_5.2, %SubWith.WithSelf.Op.call
// CHECK:STDOUT:     %impl.elem0.loc137_5.1: @SubWith.loc135.%.loc137_5.5 (%.9c3) = impl_witness_access constants.%Destroy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc137_5.2 (constants.%impl.elem0.8bd)]
// CHECK:STDOUT:     %bound_method.loc137_5.3: <bound method> = bound_method %.loc137_5.3, %impl.elem0.loc137_5.1
// CHECK:STDOUT:     %specific_impl_fn.loc137_5.2: <specific function> = specific_impl_function %impl.elem0.loc137_5.1, @Destroy.WithSelf.Op(constants.%Result) [symbolic = %specific_impl_fn.loc137_5.4 (constants.%specific_impl_fn.8dd)]
// CHECK:STDOUT:     %bound_method.loc137_5.4: <bound method> = bound_method %.loc137_5.3, %specific_impl_fn.loc137_5.2
// CHECK:STDOUT:     %Destroy.WithSelf.Op.call: init %empty_tuple.type = call %bound_method.loc137_5.4(%.loc137_5.3)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @MulWith.loc146(%Result.loc143_9.2: %Destroy.type, %U.loc144_4.2: %Destroy.type, %T.loc145_4.2: @MulWith.loc146.%facet_type.loc145_38 (%facet_type.779)) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %require_complete.loc148: <witness> = require_complete_type %MulWith.type.loc145_36.1 [symbolic = %require_complete.loc148 (constants.%require_complete.be1)]
// CHECK:STDOUT:   %assoc1: @MulWith.loc146.%MulWith.assoc_type (%MulWith.assoc_type.4f3) = assoc_entity element1, imports.%Core.import_ref.7ed [symbolic = %assoc1 (constants.%assoc1.4a5)]
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.loc148: <witness> = lookup_impl_witness %T.loc145_4.1, @MulWith.1, @MulWith.1(%U.as_type.loc145_36.1) [symbolic = %MulWith.lookup_impl_witness.loc148 (constants.%MulWith.lookup_impl_witness.69d)]
// CHECK:STDOUT:   %MulWith.facet: @MulWith.loc146.%MulWith.type.loc145_36.1 (%MulWith.type.b5e) = facet_value %T.as_type.loc146_29.1, (%MulWith.lookup_impl_witness.loc148) [symbolic = %MulWith.facet (constants.%MulWith.facet.181)]
// CHECK:STDOUT:   %MulWith.WithSelf.Op.type: type = fn_type @MulWith.WithSelf.Op, @MulWith.WithSelf(%U.as_type.loc145_36.1, %MulWith.facet) [symbolic = %MulWith.WithSelf.Op.type (constants.%MulWith.WithSelf.Op.type.ea9)]
// CHECK:STDOUT:   %.loc148_5.4: type = fn_type_with_self_type %MulWith.WithSelf.Op.type, %MulWith.facet [symbolic = %.loc148_5.4 (constants.%.efc)]
// CHECK:STDOUT:   %impl.elem1.loc148_5.2: @MulWith.loc146.%.loc148_5.4 (%.efc) = impl_witness_access %MulWith.lookup_impl_witness.loc148, element1 [symbolic = %impl.elem1.loc148_5.2 (constants.%impl.elem1.002)]
// CHECK:STDOUT:   %specific_impl_fn.loc148_5.3: <specific function> = specific_impl_function %impl.elem1.loc148_5.2, @MulWith.WithSelf.Op(%U.as_type.loc145_36.1, %MulWith.facet) [symbolic = %specific_impl_fn.loc148_5.3 (constants.%specific_impl_fn.168)]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Result.loc143_9.1) [symbolic = %Destroy.WithSelf.Op.type (constants.%Destroy.WithSelf.Op.type.12b49e.2)]
// CHECK:STDOUT:   %.loc148_5.5: type = fn_type_with_self_type %Destroy.WithSelf.Op.type, %Result.loc143_9.1 [symbolic = %.loc148_5.5 (constants.%.9c3)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %Result.loc143_9.1, @Destroy [symbolic = %Destroy.lookup_impl_witness (constants.%Destroy.lookup_impl_witness)]
// CHECK:STDOUT:   %impl.elem0.loc148_5.2: @MulWith.loc146.%.loc148_5.5 (%.9c3) = impl_witness_access %Destroy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc148_5.2 (constants.%impl.elem0.8bd)]
// CHECK:STDOUT:   %specific_impl_fn.loc148_5.4: <specific function> = specific_impl_function %impl.elem0.loc148_5.2, @Destroy.WithSelf.Op(%Result.loc143_9.1) [symbolic = %specific_impl_fn.loc148_5.4 (constants.%specific_impl_fn.8dd)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%result.param: @MulWith.loc146.%Result.as_type.loc145_54.1 (%Result.as_type), %x.param: @MulWith.loc146.%T.as_type.loc146_29.1 (%T.as_type.380), %y.param: @MulWith.loc146.%U.as_type.loc145_36.1 (%U.as_type.0fd)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: @MulWith.loc146.%T.as_type.loc146_29.1 (%T.as_type.380) = name_ref x, %x
// CHECK:STDOUT:     %y.ref: @MulWith.loc146.%U.as_type.loc145_36.1 (%U.as_type.0fd) = name_ref y, %y
// CHECK:STDOUT:     %MulWith.type.loc148: type = facet_type <@MulWith.1, @MulWith.1(constants.%U.as_type.0fd)> [symbolic = %MulWith.type.loc145_36.1 (constants.%MulWith.type.b5e)]
// CHECK:STDOUT:     %.loc148_5.1: @MulWith.loc146.%MulWith.assoc_type (%MulWith.assoc_type.4f3) = specific_constant imports.%Core.import_ref.1d4, @MulWith.WithSelf(constants.%U.as_type.0fd, constants.%Self.019) [symbolic = %assoc1 (constants.%assoc1.4a5)]
// CHECK:STDOUT:     %Op.ref: @MulWith.loc146.%MulWith.assoc_type (%MulWith.assoc_type.4f3) = name_ref Op, %.loc148_5.1 [symbolic = %assoc1 (constants.%assoc1.4a5)]
// CHECK:STDOUT:     %impl.elem1.loc148_5.1: @MulWith.loc146.%.loc148_5.4 (%.efc) = impl_witness_access constants.%MulWith.lookup_impl_witness.69d, element1 [symbolic = %impl.elem1.loc148_5.2 (constants.%impl.elem1.002)]
// CHECK:STDOUT:     %bound_method.loc148_5.1: <bound method> = bound_method %x.ref, %impl.elem1.loc148_5.1
// CHECK:STDOUT:     %specific_impl_fn.loc148_5.1: <specific function> = specific_impl_function %impl.elem1.loc148_5.1, @MulWith.WithSelf.Op(constants.%U.as_type.0fd, constants.%MulWith.facet.181) [symbolic = %specific_impl_fn.loc148_5.3 (constants.%specific_impl_fn.168)]
// CHECK:STDOUT:     %bound_method.loc148_5.2: <bound method> = bound_method %x.ref, %specific_impl_fn.loc148_5.1
// CHECK:STDOUT:     %.loc148_5.2: ref @MulWith.loc146.%Result.as_type.loc145_54.1 (%Result.as_type) = temporary_storage
// CHECK:STDOUT:     %MulWith.WithSelf.Op.call: init @MulWith.loc146.%Result.as_type.loc145_54.1 (%Result.as_type) to %.loc148_5.2 = call %bound_method.loc148_5.2(%x.ref, %y.ref)
// CHECK:STDOUT:     %.loc148_5.3: ref @MulWith.loc146.%Result.as_type.loc145_54.1 (%Result.as_type) = temporary %.loc148_5.2, %MulWith.WithSelf.Op.call
// CHECK:STDOUT:     %impl.elem0.loc148_5.1: @MulWith.loc146.%.loc148_5.5 (%.9c3) = impl_witness_access constants.%Destroy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc148_5.2 (constants.%impl.elem0.8bd)]
// CHECK:STDOUT:     %bound_method.loc148_5.3: <bound method> = bound_method %.loc148_5.3, %impl.elem0.loc148_5.1
// CHECK:STDOUT:     %specific_impl_fn.loc148_5.2: <specific function> = specific_impl_function %impl.elem0.loc148_5.1, @Destroy.WithSelf.Op(constants.%Result) [symbolic = %specific_impl_fn.loc148_5.4 (constants.%specific_impl_fn.8dd)]
// CHECK:STDOUT:     %bound_method.loc148_5.4: <bound method> = bound_method %.loc148_5.3, %specific_impl_fn.loc148_5.2
// CHECK:STDOUT:     %Destroy.WithSelf.Op.call: init %empty_tuple.type = call %bound_method.loc148_5.4(%.loc148_5.3)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @DivWith.loc157(%Result.loc154_9.2: %Destroy.type, %U.loc155_4.2: %Destroy.type, %T.loc156_4.2: @DivWith.loc157.%facet_type.loc156_38 (%facet_type.c18)) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %require_complete.loc159: <witness> = require_complete_type %DivWith.type.loc156_36.1 [symbolic = %require_complete.loc159 (constants.%require_complete.df0)]
// CHECK:STDOUT:   %assoc1: @DivWith.loc157.%DivWith.assoc_type (%DivWith.assoc_type.cd1) = assoc_entity element1, imports.%Core.import_ref.44d [symbolic = %assoc1 (constants.%assoc1.5f3)]
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.loc159: <witness> = lookup_impl_witness %T.loc156_4.1, @DivWith.1, @DivWith.1(%U.as_type.loc156_36.1) [symbolic = %DivWith.lookup_impl_witness.loc159 (constants.%DivWith.lookup_impl_witness.62a)]
// CHECK:STDOUT:   %DivWith.facet: @DivWith.loc157.%DivWith.type.loc156_36.1 (%DivWith.type.c0c) = facet_value %T.as_type.loc157_29.1, (%DivWith.lookup_impl_witness.loc159) [symbolic = %DivWith.facet (constants.%DivWith.facet.08b)]
// CHECK:STDOUT:   %DivWith.WithSelf.Op.type: type = fn_type @DivWith.WithSelf.Op, @DivWith.WithSelf(%U.as_type.loc156_36.1, %DivWith.facet) [symbolic = %DivWith.WithSelf.Op.type (constants.%DivWith.WithSelf.Op.type.ed1)]
// CHECK:STDOUT:   %.loc159_5.4: type = fn_type_with_self_type %DivWith.WithSelf.Op.type, %DivWith.facet [symbolic = %.loc159_5.4 (constants.%.612)]
// CHECK:STDOUT:   %impl.elem1.loc159_5.2: @DivWith.loc157.%.loc159_5.4 (%.612) = impl_witness_access %DivWith.lookup_impl_witness.loc159, element1 [symbolic = %impl.elem1.loc159_5.2 (constants.%impl.elem1.536)]
// CHECK:STDOUT:   %specific_impl_fn.loc159_5.3: <specific function> = specific_impl_function %impl.elem1.loc159_5.2, @DivWith.WithSelf.Op(%U.as_type.loc156_36.1, %DivWith.facet) [symbolic = %specific_impl_fn.loc159_5.3 (constants.%specific_impl_fn.706)]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Result.loc154_9.1) [symbolic = %Destroy.WithSelf.Op.type (constants.%Destroy.WithSelf.Op.type.12b49e.2)]
// CHECK:STDOUT:   %.loc159_5.5: type = fn_type_with_self_type %Destroy.WithSelf.Op.type, %Result.loc154_9.1 [symbolic = %.loc159_5.5 (constants.%.9c3)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %Result.loc154_9.1, @Destroy [symbolic = %Destroy.lookup_impl_witness (constants.%Destroy.lookup_impl_witness)]
// CHECK:STDOUT:   %impl.elem0.loc159_5.2: @DivWith.loc157.%.loc159_5.5 (%.9c3) = impl_witness_access %Destroy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc159_5.2 (constants.%impl.elem0.8bd)]
// CHECK:STDOUT:   %specific_impl_fn.loc159_5.4: <specific function> = specific_impl_function %impl.elem0.loc159_5.2, @Destroy.WithSelf.Op(%Result.loc154_9.1) [symbolic = %specific_impl_fn.loc159_5.4 (constants.%specific_impl_fn.8dd)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%result.param: @DivWith.loc157.%Result.as_type.loc156_54.1 (%Result.as_type), %x.param: @DivWith.loc157.%T.as_type.loc157_29.1 (%T.as_type.7d7), %y.param: @DivWith.loc157.%U.as_type.loc156_36.1 (%U.as_type.0fd)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: @DivWith.loc157.%T.as_type.loc157_29.1 (%T.as_type.7d7) = name_ref x, %x
// CHECK:STDOUT:     %y.ref: @DivWith.loc157.%U.as_type.loc156_36.1 (%U.as_type.0fd) = name_ref y, %y
// CHECK:STDOUT:     %DivWith.type.loc159: type = facet_type <@DivWith.1, @DivWith.1(constants.%U.as_type.0fd)> [symbolic = %DivWith.type.loc156_36.1 (constants.%DivWith.type.c0c)]
// CHECK:STDOUT:     %.loc159_5.1: @DivWith.loc157.%DivWith.assoc_type (%DivWith.assoc_type.cd1) = specific_constant imports.%Core.import_ref.d16, @DivWith.WithSelf(constants.%U.as_type.0fd, constants.%Self.555) [symbolic = %assoc1 (constants.%assoc1.5f3)]
// CHECK:STDOUT:     %Op.ref: @DivWith.loc157.%DivWith.assoc_type (%DivWith.assoc_type.cd1) = name_ref Op, %.loc159_5.1 [symbolic = %assoc1 (constants.%assoc1.5f3)]
// CHECK:STDOUT:     %impl.elem1.loc159_5.1: @DivWith.loc157.%.loc159_5.4 (%.612) = impl_witness_access constants.%DivWith.lookup_impl_witness.62a, element1 [symbolic = %impl.elem1.loc159_5.2 (constants.%impl.elem1.536)]
// CHECK:STDOUT:     %bound_method.loc159_5.1: <bound method> = bound_method %x.ref, %impl.elem1.loc159_5.1
// CHECK:STDOUT:     %specific_impl_fn.loc159_5.1: <specific function> = specific_impl_function %impl.elem1.loc159_5.1, @DivWith.WithSelf.Op(constants.%U.as_type.0fd, constants.%DivWith.facet.08b) [symbolic = %specific_impl_fn.loc159_5.3 (constants.%specific_impl_fn.706)]
// CHECK:STDOUT:     %bound_method.loc159_5.2: <bound method> = bound_method %x.ref, %specific_impl_fn.loc159_5.1
// CHECK:STDOUT:     %.loc159_5.2: ref @DivWith.loc157.%Result.as_type.loc156_54.1 (%Result.as_type) = temporary_storage
// CHECK:STDOUT:     %DivWith.WithSelf.Op.call: init @DivWith.loc157.%Result.as_type.loc156_54.1 (%Result.as_type) to %.loc159_5.2 = call %bound_method.loc159_5.2(%x.ref, %y.ref)
// CHECK:STDOUT:     %.loc159_5.3: ref @DivWith.loc157.%Result.as_type.loc156_54.1 (%Result.as_type) = temporary %.loc159_5.2, %DivWith.WithSelf.Op.call
// CHECK:STDOUT:     %impl.elem0.loc159_5.1: @DivWith.loc157.%.loc159_5.5 (%.9c3) = impl_witness_access constants.%Destroy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc159_5.2 (constants.%impl.elem0.8bd)]
// CHECK:STDOUT:     %bound_method.loc159_5.3: <bound method> = bound_method %.loc159_5.3, %impl.elem0.loc159_5.1
// CHECK:STDOUT:     %specific_impl_fn.loc159_5.2: <specific function> = specific_impl_function %impl.elem0.loc159_5.1, @Destroy.WithSelf.Op(constants.%Result) [symbolic = %specific_impl_fn.loc159_5.4 (constants.%specific_impl_fn.8dd)]
// CHECK:STDOUT:     %bound_method.loc159_5.4: <bound method> = bound_method %.loc159_5.3, %specific_impl_fn.loc159_5.2
// CHECK:STDOUT:     %Destroy.WithSelf.Op.call: init %empty_tuple.type = call %bound_method.loc159_5.4(%.loc159_5.3)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @ModWith.loc168(%Result.loc165_9.2: %Destroy.type, %U.loc166_4.2: %Destroy.type, %T.loc167_4.2: @ModWith.loc168.%facet_type.loc167_38 (%facet_type.ae7)) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %require_complete.loc170: <witness> = require_complete_type %ModWith.type.loc167_36.1 [symbolic = %require_complete.loc170 (constants.%require_complete.ed2)]
// CHECK:STDOUT:   %assoc1: @ModWith.loc168.%ModWith.assoc_type (%ModWith.assoc_type.e42) = assoc_entity element1, imports.%Core.import_ref.556 [symbolic = %assoc1 (constants.%assoc1.516)]
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.loc170: <witness> = lookup_impl_witness %T.loc167_4.1, @ModWith.1, @ModWith.1(%U.as_type.loc167_36.1) [symbolic = %ModWith.lookup_impl_witness.loc170 (constants.%ModWith.lookup_impl_witness.e49)]
// CHECK:STDOUT:   %ModWith.facet: @ModWith.loc168.%ModWith.type.loc167_36.1 (%ModWith.type.fba) = facet_value %T.as_type.loc168_29.1, (%ModWith.lookup_impl_witness.loc170) [symbolic = %ModWith.facet (constants.%ModWith.facet.6df)]
// CHECK:STDOUT:   %ModWith.WithSelf.Op.type: type = fn_type @ModWith.WithSelf.Op, @ModWith.WithSelf(%U.as_type.loc167_36.1, %ModWith.facet) [symbolic = %ModWith.WithSelf.Op.type (constants.%ModWith.WithSelf.Op.type.a89)]
// CHECK:STDOUT:   %.loc170_5.4: type = fn_type_with_self_type %ModWith.WithSelf.Op.type, %ModWith.facet [symbolic = %.loc170_5.4 (constants.%.977)]
// CHECK:STDOUT:   %impl.elem1.loc170_5.2: @ModWith.loc168.%.loc170_5.4 (%.977) = impl_witness_access %ModWith.lookup_impl_witness.loc170, element1 [symbolic = %impl.elem1.loc170_5.2 (constants.%impl.elem1.34f)]
// CHECK:STDOUT:   %specific_impl_fn.loc170_5.3: <specific function> = specific_impl_function %impl.elem1.loc170_5.2, @ModWith.WithSelf.Op(%U.as_type.loc167_36.1, %ModWith.facet) [symbolic = %specific_impl_fn.loc170_5.3 (constants.%specific_impl_fn.6e5)]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Result.loc165_9.1) [symbolic = %Destroy.WithSelf.Op.type (constants.%Destroy.WithSelf.Op.type.12b49e.2)]
// CHECK:STDOUT:   %.loc170_5.5: type = fn_type_with_self_type %Destroy.WithSelf.Op.type, %Result.loc165_9.1 [symbolic = %.loc170_5.5 (constants.%.9c3)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %Result.loc165_9.1, @Destroy [symbolic = %Destroy.lookup_impl_witness (constants.%Destroy.lookup_impl_witness)]
// CHECK:STDOUT:   %impl.elem0.loc170_5.2: @ModWith.loc168.%.loc170_5.5 (%.9c3) = impl_witness_access %Destroy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc170_5.2 (constants.%impl.elem0.8bd)]
// CHECK:STDOUT:   %specific_impl_fn.loc170_5.4: <specific function> = specific_impl_function %impl.elem0.loc170_5.2, @Destroy.WithSelf.Op(%Result.loc165_9.1) [symbolic = %specific_impl_fn.loc170_5.4 (constants.%specific_impl_fn.8dd)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%result.param: @ModWith.loc168.%Result.as_type.loc167_54.1 (%Result.as_type), %x.param: @ModWith.loc168.%T.as_type.loc168_29.1 (%T.as_type.c82), %y.param: @ModWith.loc168.%U.as_type.loc167_36.1 (%U.as_type.0fd)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: @ModWith.loc168.%T.as_type.loc168_29.1 (%T.as_type.c82) = name_ref x, %x
// CHECK:STDOUT:     %y.ref: @ModWith.loc168.%U.as_type.loc167_36.1 (%U.as_type.0fd) = name_ref y, %y
// CHECK:STDOUT:     %ModWith.type.loc170: type = facet_type <@ModWith.1, @ModWith.1(constants.%U.as_type.0fd)> [symbolic = %ModWith.type.loc167_36.1 (constants.%ModWith.type.fba)]
// CHECK:STDOUT:     %.loc170_5.1: @ModWith.loc168.%ModWith.assoc_type (%ModWith.assoc_type.e42) = specific_constant imports.%Core.import_ref.e51, @ModWith.WithSelf(constants.%U.as_type.0fd, constants.%Self.b83) [symbolic = %assoc1 (constants.%assoc1.516)]
// CHECK:STDOUT:     %Op.ref: @ModWith.loc168.%ModWith.assoc_type (%ModWith.assoc_type.e42) = name_ref Op, %.loc170_5.1 [symbolic = %assoc1 (constants.%assoc1.516)]
// CHECK:STDOUT:     %impl.elem1.loc170_5.1: @ModWith.loc168.%.loc170_5.4 (%.977) = impl_witness_access constants.%ModWith.lookup_impl_witness.e49, element1 [symbolic = %impl.elem1.loc170_5.2 (constants.%impl.elem1.34f)]
// CHECK:STDOUT:     %bound_method.loc170_5.1: <bound method> = bound_method %x.ref, %impl.elem1.loc170_5.1
// CHECK:STDOUT:     %specific_impl_fn.loc170_5.1: <specific function> = specific_impl_function %impl.elem1.loc170_5.1, @ModWith.WithSelf.Op(constants.%U.as_type.0fd, constants.%ModWith.facet.6df) [symbolic = %specific_impl_fn.loc170_5.3 (constants.%specific_impl_fn.6e5)]
// CHECK:STDOUT:     %bound_method.loc170_5.2: <bound method> = bound_method %x.ref, %specific_impl_fn.loc170_5.1
// CHECK:STDOUT:     %.loc170_5.2: ref @ModWith.loc168.%Result.as_type.loc167_54.1 (%Result.as_type) = temporary_storage
// CHECK:STDOUT:     %ModWith.WithSelf.Op.call: init @ModWith.loc168.%Result.as_type.loc167_54.1 (%Result.as_type) to %.loc170_5.2 = call %bound_method.loc170_5.2(%x.ref, %y.ref)
// CHECK:STDOUT:     %.loc170_5.3: ref @ModWith.loc168.%Result.as_type.loc167_54.1 (%Result.as_type) = temporary %.loc170_5.2, %ModWith.WithSelf.Op.call
// CHECK:STDOUT:     %impl.elem0.loc170_5.1: @ModWith.loc168.%.loc170_5.5 (%.9c3) = impl_witness_access constants.%Destroy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc170_5.2 (constants.%impl.elem0.8bd)]
// CHECK:STDOUT:     %bound_method.loc170_5.3: <bound method> = bound_method %.loc170_5.3, %impl.elem0.loc170_5.1
// CHECK:STDOUT:     %specific_impl_fn.loc170_5.2: <specific function> = specific_impl_function %impl.elem0.loc170_5.1, @Destroy.WithSelf.Op(constants.%Result) [symbolic = %specific_impl_fn.loc170_5.4 (constants.%specific_impl_fn.8dd)]
// CHECK:STDOUT:     %bound_method.loc170_5.4: <bound method> = bound_method %.loc170_5.3, %specific_impl_fn.loc170_5.2
// CHECK:STDOUT:     %Destroy.WithSelf.Op.call: init %empty_tuple.type = call %bound_method.loc170_5.4(%.loc170_5.3)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @AddAssignWith.loc225(%U.loc225_19.2: type, %T.loc225_29.2: @AddAssignWith.loc225.%AddAssignWith.type.loc225_52.1 (%AddAssignWith.type.dc8ed5.2)) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %require_complete.loc227: <witness> = require_complete_type %AddAssignWith.type.loc225_52.1 [symbolic = %require_complete.loc227 (constants.%require_complete.44c)]
// CHECK:STDOUT:   %AddAssignWith.assoc_type: type = assoc_entity_type @AddAssignWith.1, @AddAssignWith.1(%U.loc225_19.1) [symbolic = %AddAssignWith.assoc_type (constants.%AddAssignWith.assoc_type.a25160.2)]
// CHECK:STDOUT:   %assoc0: @AddAssignWith.loc225.%AddAssignWith.assoc_type (%AddAssignWith.assoc_type.a25160.2) = assoc_entity element0, imports.%Core.import_ref.4eb [symbolic = %assoc0 (constants.%assoc0.9fb771.2)]
// CHECK:STDOUT:   %AddAssignWith.WithSelf.Op.type: type = fn_type @AddAssignWith.WithSelf.Op, @AddAssignWith.WithSelf(%U.loc225_19.1, %T.loc225_29.1) [symbolic = %AddAssignWith.WithSelf.Op.type (constants.%AddAssignWith.WithSelf.Op.type.8b23d4.3)]
// CHECK:STDOUT:   %.loc227_5.2: type = fn_type_with_self_type %AddAssignWith.WithSelf.Op.type, %T.loc225_29.1 [symbolic = %.loc227_5.2 (constants.%.ef8)]
// CHECK:STDOUT:   %AddAssignWith.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc225_29.1, @AddAssignWith.1, @AddAssignWith.1(%U.loc225_19.1) [symbolic = %AddAssignWith.lookup_impl_witness (constants.%AddAssignWith.lookup_impl_witness)]
// CHECK:STDOUT:   %impl.elem0.loc227_5.2: @AddAssignWith.loc225.%.loc227_5.2 (%.ef8) = impl_witness_access %AddAssignWith.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc227_5.2 (constants.%impl.elem0.89f)]
// CHECK:STDOUT:   %specific_impl_fn.loc227_5.2: <specific function> = specific_impl_function %impl.elem0.loc227_5.2, @AddAssignWith.WithSelf.Op(%U.loc225_19.1, %T.loc225_29.1) [symbolic = %specific_impl_fn.loc227_5.2 (constants.%specific_impl_fn.7ee)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: ref @AddAssignWith.loc225.%T.as_type.loc225_62.1 (%T.as_type.7d6), %y.param: @AddAssignWith.loc225.%U.loc225_19.1 (%U.67d)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: ref @AddAssignWith.loc225.%T.as_type.loc225_62.1 (%T.as_type.7d6) = name_ref x, %x
// CHECK:STDOUT:     %y.ref: @AddAssignWith.loc225.%U.loc225_19.1 (%U.67d) = name_ref y, %y
// CHECK:STDOUT:     %AddAssignWith.type.loc227: type = facet_type <@AddAssignWith.1, @AddAssignWith.1(constants.%U.67d)> [symbolic = %AddAssignWith.type.loc225_52.1 (constants.%AddAssignWith.type.dc8ed5.2)]
// CHECK:STDOUT:     %.loc227_5.1: @AddAssignWith.loc225.%AddAssignWith.assoc_type (%AddAssignWith.assoc_type.a25160.2) = specific_constant imports.%Core.import_ref.eb9, @AddAssignWith.WithSelf(constants.%U.67d, constants.%Self.689a47.1) [symbolic = %assoc0 (constants.%assoc0.9fb771.2)]
// CHECK:STDOUT:     %Op.ref: @AddAssignWith.loc225.%AddAssignWith.assoc_type (%AddAssignWith.assoc_type.a25160.2) = name_ref Op, %.loc227_5.1 [symbolic = %assoc0 (constants.%assoc0.9fb771.2)]
// CHECK:STDOUT:     %impl.elem0.loc227_5.1: @AddAssignWith.loc225.%.loc227_5.2 (%.ef8) = impl_witness_access constants.%AddAssignWith.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc227_5.2 (constants.%impl.elem0.89f)]
// CHECK:STDOUT:     %bound_method.loc227_5.1: <bound method> = bound_method %x.ref, %impl.elem0.loc227_5.1
// CHECK:STDOUT:     %specific_impl_fn.loc227_5.1: <specific function> = specific_impl_function %impl.elem0.loc227_5.1, @AddAssignWith.WithSelf.Op(constants.%U.67d, constants.%T.689) [symbolic = %specific_impl_fn.loc227_5.2 (constants.%specific_impl_fn.7ee)]
// CHECK:STDOUT:     %bound_method.loc227_5.2: <bound method> = bound_method %x.ref, %specific_impl_fn.loc227_5.1
// CHECK:STDOUT:     %AddAssignWith.WithSelf.Op.call: init %empty_tuple.type = call %bound_method.loc227_5.2(%x.ref, %y.ref)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @SubAssignWith.loc231(%U.loc231_19.2: type, %T.loc231_29.2: @SubAssignWith.loc231.%SubAssignWith.type.loc231_52.1 (%SubAssignWith.type.a95ddd.2)) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %require_complete.loc233: <witness> = require_complete_type %SubAssignWith.type.loc231_52.1 [symbolic = %require_complete.loc233 (constants.%require_complete.f21)]
// CHECK:STDOUT:   %SubAssignWith.assoc_type: type = assoc_entity_type @SubAssignWith.1, @SubAssignWith.1(%U.loc231_19.1) [symbolic = %SubAssignWith.assoc_type (constants.%SubAssignWith.assoc_type.0440e4.2)]
// CHECK:STDOUT:   %assoc0: @SubAssignWith.loc231.%SubAssignWith.assoc_type (%SubAssignWith.assoc_type.0440e4.2) = assoc_entity element0, imports.%Core.import_ref.a2d0 [symbolic = %assoc0 (constants.%assoc0.180b38.2)]
// CHECK:STDOUT:   %SubAssignWith.WithSelf.Op.type: type = fn_type @SubAssignWith.WithSelf.Op, @SubAssignWith.WithSelf(%U.loc231_19.1, %T.loc231_29.1) [symbolic = %SubAssignWith.WithSelf.Op.type (constants.%SubAssignWith.WithSelf.Op.type.ac317f.3)]
// CHECK:STDOUT:   %.loc233_5.2: type = fn_type_with_self_type %SubAssignWith.WithSelf.Op.type, %T.loc231_29.1 [symbolic = %.loc233_5.2 (constants.%.6d8a)]
// CHECK:STDOUT:   %SubAssignWith.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc231_29.1, @SubAssignWith.1, @SubAssignWith.1(%U.loc231_19.1) [symbolic = %SubAssignWith.lookup_impl_witness (constants.%SubAssignWith.lookup_impl_witness)]
// CHECK:STDOUT:   %impl.elem0.loc233_5.2: @SubAssignWith.loc231.%.loc233_5.2 (%.6d8a) = impl_witness_access %SubAssignWith.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc233_5.2 (constants.%impl.elem0.ab4)]
// CHECK:STDOUT:   %specific_impl_fn.loc233_5.2: <specific function> = specific_impl_function %impl.elem0.loc233_5.2, @SubAssignWith.WithSelf.Op(%U.loc231_19.1, %T.loc231_29.1) [symbolic = %specific_impl_fn.loc233_5.2 (constants.%specific_impl_fn.579)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: ref @SubAssignWith.loc231.%T.as_type.loc231_62.1 (%T.as_type.a39), %y.param: @SubAssignWith.loc231.%U.loc231_19.1 (%U.67d)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: ref @SubAssignWith.loc231.%T.as_type.loc231_62.1 (%T.as_type.a39) = name_ref x, %x
// CHECK:STDOUT:     %y.ref: @SubAssignWith.loc231.%U.loc231_19.1 (%U.67d) = name_ref y, %y
// CHECK:STDOUT:     %SubAssignWith.type.loc233: type = facet_type <@SubAssignWith.1, @SubAssignWith.1(constants.%U.67d)> [symbolic = %SubAssignWith.type.loc231_52.1 (constants.%SubAssignWith.type.a95ddd.2)]
// CHECK:STDOUT:     %.loc233_5.1: @SubAssignWith.loc231.%SubAssignWith.assoc_type (%SubAssignWith.assoc_type.0440e4.2) = specific_constant imports.%Core.import_ref.7fa, @SubAssignWith.WithSelf(constants.%U.67d, constants.%Self.af3029.1) [symbolic = %assoc0 (constants.%assoc0.180b38.2)]
// CHECK:STDOUT:     %Op.ref: @SubAssignWith.loc231.%SubAssignWith.assoc_type (%SubAssignWith.assoc_type.0440e4.2) = name_ref Op, %.loc233_5.1 [symbolic = %assoc0 (constants.%assoc0.180b38.2)]
// CHECK:STDOUT:     %impl.elem0.loc233_5.1: @SubAssignWith.loc231.%.loc233_5.2 (%.6d8a) = impl_witness_access constants.%SubAssignWith.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc233_5.2 (constants.%impl.elem0.ab4)]
// CHECK:STDOUT:     %bound_method.loc233_5.1: <bound method> = bound_method %x.ref, %impl.elem0.loc233_5.1
// CHECK:STDOUT:     %specific_impl_fn.loc233_5.1: <specific function> = specific_impl_function %impl.elem0.loc233_5.1, @SubAssignWith.WithSelf.Op(constants.%U.67d, constants.%T.af3) [symbolic = %specific_impl_fn.loc233_5.2 (constants.%specific_impl_fn.579)]
// CHECK:STDOUT:     %bound_method.loc233_5.2: <bound method> = bound_method %x.ref, %specific_impl_fn.loc233_5.1
// CHECK:STDOUT:     %SubAssignWith.WithSelf.Op.call: init %empty_tuple.type = call %bound_method.loc233_5.2(%x.ref, %y.ref)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @MulAssignWith.loc237(%U.loc237_19.2: type, %T.loc237_29.2: @MulAssignWith.loc237.%MulAssignWith.type.loc237_52.1 (%MulAssignWith.type.c08ab6.2)) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %require_complete.loc239: <witness> = require_complete_type %MulAssignWith.type.loc237_52.1 [symbolic = %require_complete.loc239 (constants.%require_complete.635)]
// CHECK:STDOUT:   %MulAssignWith.assoc_type: type = assoc_entity_type @MulAssignWith.1, @MulAssignWith.1(%U.loc237_19.1) [symbolic = %MulAssignWith.assoc_type (constants.%MulAssignWith.assoc_type.dd8648.2)]
// CHECK:STDOUT:   %assoc0: @MulAssignWith.loc237.%MulAssignWith.assoc_type (%MulAssignWith.assoc_type.dd8648.2) = assoc_entity element0, imports.%Core.import_ref.475 [symbolic = %assoc0 (constants.%assoc0.bb9cfe.2)]
// CHECK:STDOUT:   %MulAssignWith.WithSelf.Op.type: type = fn_type @MulAssignWith.WithSelf.Op, @MulAssignWith.WithSelf(%U.loc237_19.1, %T.loc237_29.1) [symbolic = %MulAssignWith.WithSelf.Op.type (constants.%MulAssignWith.WithSelf.Op.type.dff752.3)]
// CHECK:STDOUT:   %.loc239_5.2: type = fn_type_with_self_type %MulAssignWith.WithSelf.Op.type, %T.loc237_29.1 [symbolic = %.loc239_5.2 (constants.%.9a8d)]
// CHECK:STDOUT:   %MulAssignWith.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc237_29.1, @MulAssignWith.1, @MulAssignWith.1(%U.loc237_19.1) [symbolic = %MulAssignWith.lookup_impl_witness (constants.%MulAssignWith.lookup_impl_witness)]
// CHECK:STDOUT:   %impl.elem0.loc239_5.2: @MulAssignWith.loc237.%.loc239_5.2 (%.9a8d) = impl_witness_access %MulAssignWith.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc239_5.2 (constants.%impl.elem0.653)]
// CHECK:STDOUT:   %specific_impl_fn.loc239_5.2: <specific function> = specific_impl_function %impl.elem0.loc239_5.2, @MulAssignWith.WithSelf.Op(%U.loc237_19.1, %T.loc237_29.1) [symbolic = %specific_impl_fn.loc239_5.2 (constants.%specific_impl_fn.5ff)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: ref @MulAssignWith.loc237.%T.as_type.loc237_62.1 (%T.as_type.26e), %y.param: @MulAssignWith.loc237.%U.loc237_19.1 (%U.67d)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: ref @MulAssignWith.loc237.%T.as_type.loc237_62.1 (%T.as_type.26e) = name_ref x, %x
// CHECK:STDOUT:     %y.ref: @MulAssignWith.loc237.%U.loc237_19.1 (%U.67d) = name_ref y, %y
// CHECK:STDOUT:     %MulAssignWith.type.loc239: type = facet_type <@MulAssignWith.1, @MulAssignWith.1(constants.%U.67d)> [symbolic = %MulAssignWith.type.loc237_52.1 (constants.%MulAssignWith.type.c08ab6.2)]
// CHECK:STDOUT:     %.loc239_5.1: @MulAssignWith.loc237.%MulAssignWith.assoc_type (%MulAssignWith.assoc_type.dd8648.2) = specific_constant imports.%Core.import_ref.dd3, @MulAssignWith.WithSelf(constants.%U.67d, constants.%Self.c8f88d.1) [symbolic = %assoc0 (constants.%assoc0.bb9cfe.2)]
// CHECK:STDOUT:     %Op.ref: @MulAssignWith.loc237.%MulAssignWith.assoc_type (%MulAssignWith.assoc_type.dd8648.2) = name_ref Op, %.loc239_5.1 [symbolic = %assoc0 (constants.%assoc0.bb9cfe.2)]
// CHECK:STDOUT:     %impl.elem0.loc239_5.1: @MulAssignWith.loc237.%.loc239_5.2 (%.9a8d) = impl_witness_access constants.%MulAssignWith.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc239_5.2 (constants.%impl.elem0.653)]
// CHECK:STDOUT:     %bound_method.loc239_5.1: <bound method> = bound_method %x.ref, %impl.elem0.loc239_5.1
// CHECK:STDOUT:     %specific_impl_fn.loc239_5.1: <specific function> = specific_impl_function %impl.elem0.loc239_5.1, @MulAssignWith.WithSelf.Op(constants.%U.67d, constants.%T.c8f) [symbolic = %specific_impl_fn.loc239_5.2 (constants.%specific_impl_fn.5ff)]
// CHECK:STDOUT:     %bound_method.loc239_5.2: <bound method> = bound_method %x.ref, %specific_impl_fn.loc239_5.1
// CHECK:STDOUT:     %MulAssignWith.WithSelf.Op.call: init %empty_tuple.type = call %bound_method.loc239_5.2(%x.ref, %y.ref)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @DivAssignWith.loc243(%U.loc243_19.2: type, %T.loc243_29.2: @DivAssignWith.loc243.%DivAssignWith.type.loc243_52.1 (%DivAssignWith.type.d7b8dd.2)) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %require_complete.loc245: <witness> = require_complete_type %DivAssignWith.type.loc243_52.1 [symbolic = %require_complete.loc245 (constants.%require_complete.3cf)]
// CHECK:STDOUT:   %DivAssignWith.assoc_type: type = assoc_entity_type @DivAssignWith.1, @DivAssignWith.1(%U.loc243_19.1) [symbolic = %DivAssignWith.assoc_type (constants.%DivAssignWith.assoc_type.738ee6.2)]
// CHECK:STDOUT:   %assoc0: @DivAssignWith.loc243.%DivAssignWith.assoc_type (%DivAssignWith.assoc_type.738ee6.2) = assoc_entity element0, imports.%Core.import_ref.12d [symbolic = %assoc0 (constants.%assoc0.d42d61.2)]
// CHECK:STDOUT:   %DivAssignWith.WithSelf.Op.type: type = fn_type @DivAssignWith.WithSelf.Op, @DivAssignWith.WithSelf(%U.loc243_19.1, %T.loc243_29.1) [symbolic = %DivAssignWith.WithSelf.Op.type (constants.%DivAssignWith.WithSelf.Op.type.3077dd.3)]
// CHECK:STDOUT:   %.loc245_5.2: type = fn_type_with_self_type %DivAssignWith.WithSelf.Op.type, %T.loc243_29.1 [symbolic = %.loc245_5.2 (constants.%.d5f)]
// CHECK:STDOUT:   %DivAssignWith.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc243_29.1, @DivAssignWith.1, @DivAssignWith.1(%U.loc243_19.1) [symbolic = %DivAssignWith.lookup_impl_witness (constants.%DivAssignWith.lookup_impl_witness)]
// CHECK:STDOUT:   %impl.elem0.loc245_5.2: @DivAssignWith.loc243.%.loc245_5.2 (%.d5f) = impl_witness_access %DivAssignWith.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc245_5.2 (constants.%impl.elem0.faa)]
// CHECK:STDOUT:   %specific_impl_fn.loc245_5.2: <specific function> = specific_impl_function %impl.elem0.loc245_5.2, @DivAssignWith.WithSelf.Op(%U.loc243_19.1, %T.loc243_29.1) [symbolic = %specific_impl_fn.loc245_5.2 (constants.%specific_impl_fn.63f)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: ref @DivAssignWith.loc243.%T.as_type.loc243_62.1 (%T.as_type.dc5), %y.param: @DivAssignWith.loc243.%U.loc243_19.1 (%U.67d)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: ref @DivAssignWith.loc243.%T.as_type.loc243_62.1 (%T.as_type.dc5) = name_ref x, %x
// CHECK:STDOUT:     %y.ref: @DivAssignWith.loc243.%U.loc243_19.1 (%U.67d) = name_ref y, %y
// CHECK:STDOUT:     %DivAssignWith.type.loc245: type = facet_type <@DivAssignWith.1, @DivAssignWith.1(constants.%U.67d)> [symbolic = %DivAssignWith.type.loc243_52.1 (constants.%DivAssignWith.type.d7b8dd.2)]
// CHECK:STDOUT:     %.loc245_5.1: @DivAssignWith.loc243.%DivAssignWith.assoc_type (%DivAssignWith.assoc_type.738ee6.2) = specific_constant imports.%Core.import_ref.927, @DivAssignWith.WithSelf(constants.%U.67d, constants.%Self.f22977.1) [symbolic = %assoc0 (constants.%assoc0.d42d61.2)]
// CHECK:STDOUT:     %Op.ref: @DivAssignWith.loc243.%DivAssignWith.assoc_type (%DivAssignWith.assoc_type.738ee6.2) = name_ref Op, %.loc245_5.1 [symbolic = %assoc0 (constants.%assoc0.d42d61.2)]
// CHECK:STDOUT:     %impl.elem0.loc245_5.1: @DivAssignWith.loc243.%.loc245_5.2 (%.d5f) = impl_witness_access constants.%DivAssignWith.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc245_5.2 (constants.%impl.elem0.faa)]
// CHECK:STDOUT:     %bound_method.loc245_5.1: <bound method> = bound_method %x.ref, %impl.elem0.loc245_5.1
// CHECK:STDOUT:     %specific_impl_fn.loc245_5.1: <specific function> = specific_impl_function %impl.elem0.loc245_5.1, @DivAssignWith.WithSelf.Op(constants.%U.67d, constants.%T.f22) [symbolic = %specific_impl_fn.loc245_5.2 (constants.%specific_impl_fn.63f)]
// CHECK:STDOUT:     %bound_method.loc245_5.2: <bound method> = bound_method %x.ref, %specific_impl_fn.loc245_5.1
// CHECK:STDOUT:     %DivAssignWith.WithSelf.Op.call: init %empty_tuple.type = call %bound_method.loc245_5.2(%x.ref, %y.ref)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @ModAssignWith.loc249(%U.loc249_19.2: type, %T.loc249_29.2: @ModAssignWith.loc249.%ModAssignWith.type.loc249_52.1 (%ModAssignWith.type.958f4d.2)) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %require_complete.loc251: <witness> = require_complete_type %ModAssignWith.type.loc249_52.1 [symbolic = %require_complete.loc251 (constants.%require_complete.479)]
// CHECK:STDOUT:   %ModAssignWith.assoc_type: type = assoc_entity_type @ModAssignWith.1, @ModAssignWith.1(%U.loc249_19.1) [symbolic = %ModAssignWith.assoc_type (constants.%ModAssignWith.assoc_type.9ba07c.2)]
// CHECK:STDOUT:   %assoc0: @ModAssignWith.loc249.%ModAssignWith.assoc_type (%ModAssignWith.assoc_type.9ba07c.2) = assoc_entity element0, imports.%Core.import_ref.c884 [symbolic = %assoc0 (constants.%assoc0.fd162a.2)]
// CHECK:STDOUT:   %ModAssignWith.WithSelf.Op.type: type = fn_type @ModAssignWith.WithSelf.Op, @ModAssignWith.WithSelf(%U.loc249_19.1, %T.loc249_29.1) [symbolic = %ModAssignWith.WithSelf.Op.type (constants.%ModAssignWith.WithSelf.Op.type.3abf50.3)]
// CHECK:STDOUT:   %.loc251_5.2: type = fn_type_with_self_type %ModAssignWith.WithSelf.Op.type, %T.loc249_29.1 [symbolic = %.loc251_5.2 (constants.%.759)]
// CHECK:STDOUT:   %ModAssignWith.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc249_29.1, @ModAssignWith.1, @ModAssignWith.1(%U.loc249_19.1) [symbolic = %ModAssignWith.lookup_impl_witness (constants.%ModAssignWith.lookup_impl_witness)]
// CHECK:STDOUT:   %impl.elem0.loc251_5.2: @ModAssignWith.loc249.%.loc251_5.2 (%.759) = impl_witness_access %ModAssignWith.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc251_5.2 (constants.%impl.elem0.289)]
// CHECK:STDOUT:   %specific_impl_fn.loc251_5.2: <specific function> = specific_impl_function %impl.elem0.loc251_5.2, @ModAssignWith.WithSelf.Op(%U.loc249_19.1, %T.loc249_29.1) [symbolic = %specific_impl_fn.loc251_5.2 (constants.%specific_impl_fn.af2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: ref @ModAssignWith.loc249.%T.as_type.loc249_62.1 (%T.as_type.5ba), %y.param: @ModAssignWith.loc249.%U.loc249_19.1 (%U.67d)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: ref @ModAssignWith.loc249.%T.as_type.loc249_62.1 (%T.as_type.5ba) = name_ref x, %x
// CHECK:STDOUT:     %y.ref: @ModAssignWith.loc249.%U.loc249_19.1 (%U.67d) = name_ref y, %y
// CHECK:STDOUT:     %ModAssignWith.type.loc251: type = facet_type <@ModAssignWith.1, @ModAssignWith.1(constants.%U.67d)> [symbolic = %ModAssignWith.type.loc249_52.1 (constants.%ModAssignWith.type.958f4d.2)]
// CHECK:STDOUT:     %.loc251_5.1: @ModAssignWith.loc249.%ModAssignWith.assoc_type (%ModAssignWith.assoc_type.9ba07c.2) = specific_constant imports.%Core.import_ref.ab7, @ModAssignWith.WithSelf(constants.%U.67d, constants.%Self.37b6d9.1) [symbolic = %assoc0 (constants.%assoc0.fd162a.2)]
// CHECK:STDOUT:     %Op.ref: @ModAssignWith.loc249.%ModAssignWith.assoc_type (%ModAssignWith.assoc_type.9ba07c.2) = name_ref Op, %.loc251_5.1 [symbolic = %assoc0 (constants.%assoc0.fd162a.2)]
// CHECK:STDOUT:     %impl.elem0.loc251_5.1: @ModAssignWith.loc249.%.loc251_5.2 (%.759) = impl_witness_access constants.%ModAssignWith.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc251_5.2 (constants.%impl.elem0.289)]
// CHECK:STDOUT:     %bound_method.loc251_5.1: <bound method> = bound_method %x.ref, %impl.elem0.loc251_5.1
// CHECK:STDOUT:     %specific_impl_fn.loc251_5.1: <specific function> = specific_impl_function %impl.elem0.loc251_5.1, @ModAssignWith.WithSelf.Op(constants.%U.67d, constants.%T.37b) [symbolic = %specific_impl_fn.loc251_5.2 (constants.%specific_impl_fn.af2)]
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
// CHECK:STDOUT:   %Inc.WithSelf.Op.type: type = fn_type @Inc.WithSelf.Op, @Inc.WithSelf(%T.loc278_13.1) [symbolic = %Inc.WithSelf.Op.type (constants.%Inc.WithSelf.Op.type.0fa0d9.2)]
// CHECK:STDOUT:   %.loc280: type = fn_type_with_self_type %Inc.WithSelf.Op.type, %T.loc278_13.1 [symbolic = %.loc280 (constants.%.8af)]
// CHECK:STDOUT:   %Inc.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc278_13.1, @Inc [symbolic = %Inc.lookup_impl_witness (constants.%Inc.lookup_impl_witness)]
// CHECK:STDOUT:   %impl.elem0.loc280_3.2: @TestInc.%.loc280 (%.8af) = impl_witness_access %Inc.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc280_3.2 (constants.%impl.elem0.95b)]
// CHECK:STDOUT:   %specific_impl_fn.loc280_3.2: <specific function> = specific_impl_function %impl.elem0.loc280_3.2, @Inc.WithSelf.Op(%T.loc278_13.1) [symbolic = %specific_impl_fn.loc280_3.2 (constants.%specific_impl_fn.979)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: ref @TestInc.%T.as_type.loc278_33.1 (%T.as_type.1d1)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: ref @TestInc.%T.as_type.loc278_33.1 (%T.as_type.1d1) = name_ref x, %x
// CHECK:STDOUT:     %impl.elem0.loc280_3.1: @TestInc.%.loc280 (%.8af) = impl_witness_access constants.%Inc.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc280_3.2 (constants.%impl.elem0.95b)]
// CHECK:STDOUT:     %bound_method.loc280_3.1: <bound method> = bound_method %x.ref, %impl.elem0.loc280_3.1
// CHECK:STDOUT:     %specific_impl_fn.loc280_3.1: <specific function> = specific_impl_function %impl.elem0.loc280_3.1, @Inc.WithSelf.Op(constants.%T.fed) [symbolic = %specific_impl_fn.loc280_3.2 (constants.%specific_impl_fn.979)]
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
// CHECK:STDOUT:   %Dec.WithSelf.Op.type: type = fn_type @Dec.WithSelf.Op, @Dec.WithSelf(%T.loc284_13.1) [symbolic = %Dec.WithSelf.Op.type (constants.%Dec.WithSelf.Op.type.a100f5.2)]
// CHECK:STDOUT:   %.loc286: type = fn_type_with_self_type %Dec.WithSelf.Op.type, %T.loc284_13.1 [symbolic = %.loc286 (constants.%.e8e)]
// CHECK:STDOUT:   %Dec.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc284_13.1, @Dec [symbolic = %Dec.lookup_impl_witness (constants.%Dec.lookup_impl_witness)]
// CHECK:STDOUT:   %impl.elem0.loc286_3.2: @TestDec.%.loc286 (%.e8e) = impl_witness_access %Dec.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc286_3.2 (constants.%impl.elem0.e49)]
// CHECK:STDOUT:   %specific_impl_fn.loc286_3.2: <specific function> = specific_impl_function %impl.elem0.loc286_3.2, @Dec.WithSelf.Op(%T.loc284_13.1) [symbolic = %specific_impl_fn.loc286_3.2 (constants.%specific_impl_fn.38c)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: ref @TestDec.%T.as_type.loc284_33.1 (%T.as_type.5ae)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: ref @TestDec.%T.as_type.loc284_33.1 (%T.as_type.5ae) = name_ref x, %x
// CHECK:STDOUT:     %impl.elem0.loc286_3.1: @TestDec.%.loc286 (%.e8e) = impl_witness_access constants.%Dec.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc286_3.2 (constants.%impl.elem0.e49)]
// CHECK:STDOUT:     %bound_method.loc286_3.1: <bound method> = bound_method %x.ref, %impl.elem0.loc286_3.1
// CHECK:STDOUT:     %specific_impl_fn.loc286_3.1: <specific function> = specific_impl_function %impl.elem0.loc286_3.1, @Dec.WithSelf.Op(constants.%T.c61) [symbolic = %specific_impl_fn.loc286_3.2 (constants.%specific_impl_fn.38c)]
// CHECK:STDOUT:     %bound_method.loc286_3.2: <bound method> = bound_method %x.ref, %specific_impl_fn.loc286_3.1
// CHECK:STDOUT:     %Dec.WithSelf.Op.call: init %empty_tuple.type = call %bound_method.loc286_3.2(%x.ref)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @TestNegate(%Result.loc292_9.2: %Destroy.type, %T.loc293_4.2: @TestNegate.%facet_type (%facet_type.4bf)) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Negate.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc293_4.1, @Negate [symbolic = %Negate.lookup_impl_witness (constants.%Negate.lookup_impl_witness.755)]
// CHECK:STDOUT:   %Negate.facet: %Negate.type = facet_value %T.as_type.loc294_24.1, (%Negate.lookup_impl_witness) [symbolic = %Negate.facet (constants.%Negate.facet.ad2)]
// CHECK:STDOUT:   %Negate.WithSelf.Op.type: type = fn_type @Negate.WithSelf.Op, @Negate.WithSelf(%Negate.facet) [symbolic = %Negate.WithSelf.Op.type (constants.%Negate.WithSelf.Op.type.039)]
// CHECK:STDOUT:   %.loc296_3.3: type = fn_type_with_self_type %Negate.WithSelf.Op.type, %Negate.facet [symbolic = %.loc296_3.3 (constants.%.487)]
// CHECK:STDOUT:   %impl.elem1.loc296_3.2: @TestNegate.%.loc296_3.3 (%.487) = impl_witness_access %Negate.lookup_impl_witness, element1 [symbolic = %impl.elem1.loc296_3.2 (constants.%impl.elem1.026)]
// CHECK:STDOUT:   %specific_impl_fn.loc296_3.3: <specific function> = specific_impl_function %impl.elem1.loc296_3.2, @Negate.WithSelf.Op(%Negate.facet) [symbolic = %specific_impl_fn.loc296_3.3 (constants.%specific_impl_fn.ee1)]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Result.loc292_9.1) [symbolic = %Destroy.WithSelf.Op.type (constants.%Destroy.WithSelf.Op.type.12b49e.2)]
// CHECK:STDOUT:   %.loc296_3.4: type = fn_type_with_self_type %Destroy.WithSelf.Op.type, %Result.loc292_9.1 [symbolic = %.loc296_3.4 (constants.%.9c3)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %Result.loc292_9.1, @Destroy [symbolic = %Destroy.lookup_impl_witness (constants.%Destroy.lookup_impl_witness)]
// CHECK:STDOUT:   %impl.elem0.loc296_3.2: @TestNegate.%.loc296_3.4 (%.9c3) = impl_witness_access %Destroy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc296_3.2 (constants.%impl.elem0.8bd)]
// CHECK:STDOUT:   %specific_impl_fn.loc296_3.4: <specific function> = specific_impl_function %impl.elem0.loc296_3.2, @Destroy.WithSelf.Op(%Result.loc292_9.1) [symbolic = %specific_impl_fn.loc296_3.4 (constants.%specific_impl_fn.8dd)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%y.param: @TestNegate.%Result.as_type.loc293_50.1 (%Result.as_type), %x.param: @TestNegate.%T.as_type.loc294_24.1 (%T.as_type.013)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: @TestNegate.%T.as_type.loc294_24.1 (%T.as_type.013) = name_ref x, %x
// CHECK:STDOUT:     %impl.elem1.loc296_3.1: @TestNegate.%.loc296_3.3 (%.487) = impl_witness_access constants.%Negate.lookup_impl_witness.755, element1 [symbolic = %impl.elem1.loc296_3.2 (constants.%impl.elem1.026)]
// CHECK:STDOUT:     %bound_method.loc296_3.1: <bound method> = bound_method %x.ref, %impl.elem1.loc296_3.1
// CHECK:STDOUT:     %specific_impl_fn.loc296_3.1: <specific function> = specific_impl_function %impl.elem1.loc296_3.1, @Negate.WithSelf.Op(constants.%Negate.facet.ad2) [symbolic = %specific_impl_fn.loc296_3.3 (constants.%specific_impl_fn.ee1)]
// CHECK:STDOUT:     %bound_method.loc296_3.2: <bound method> = bound_method %x.ref, %specific_impl_fn.loc296_3.1
// CHECK:STDOUT:     %.loc296_3.1: ref @TestNegate.%Result.as_type.loc293_50.1 (%Result.as_type) = temporary_storage
// CHECK:STDOUT:     %Negate.WithSelf.Op.call: init @TestNegate.%Result.as_type.loc293_50.1 (%Result.as_type) to %.loc296_3.1 = call %bound_method.loc296_3.2(%x.ref)
// CHECK:STDOUT:     %.loc296_3.2: ref @TestNegate.%Result.as_type.loc293_50.1 (%Result.as_type) = temporary %.loc296_3.1, %Negate.WithSelf.Op.call
// CHECK:STDOUT:     %impl.elem0.loc296_3.1: @TestNegate.%.loc296_3.4 (%.9c3) = impl_witness_access constants.%Destroy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc296_3.2 (constants.%impl.elem0.8bd)]
// CHECK:STDOUT:     %bound_method.loc296_3.3: <bound method> = bound_method %.loc296_3.2, %impl.elem0.loc296_3.1
// CHECK:STDOUT:     %specific_impl_fn.loc296_3.2: <specific function> = specific_impl_function %impl.elem0.loc296_3.1, @Destroy.WithSelf.Op(constants.%Result) [symbolic = %specific_impl_fn.loc296_3.4 (constants.%specific_impl_fn.8dd)]
// CHECK:STDOUT:     %bound_method.loc296_3.4: <bound method> = bound_method %.loc296_3.2, %specific_impl_fn.loc296_3.2
// CHECK:STDOUT:     %Destroy.WithSelf.Op.call: init %empty_tuple.type = call %bound_method.loc296_3.4(%.loc296_3.2)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AddWith.loc124(constants.%Result, constants.%U.866, constants.%T.9be) {
// CHECK:STDOUT:   %Result.loc121_9.1 => constants.%Result
// CHECK:STDOUT:   %U.loc122_4.1 => constants.%U.866
// CHECK:STDOUT:   %U.as_type.loc123_36.1 => constants.%U.as_type.0fd
// CHECK:STDOUT:   %AddWith.type.loc123_36.1 => constants.%AddWith.type.68f
// CHECK:STDOUT:   %facet_type.loc123_20 => constants.%facet_type.5e7
// CHECK:STDOUT:   %.Self.loc123_38.1 => constants.%.Self.11b
// CHECK:STDOUT:   %require_complete.loc123 => constants.%require_complete.d10
// CHECK:STDOUT:   %.Self.as_type.loc123_44.1 => constants.%.Self.as_type.8a9
// CHECK:STDOUT:   %AddWith.assoc_type => constants.%AddWith.assoc_type.f48
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.5e7
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.loc123 => constants.%AddWith.lookup_impl_witness.2ac
// CHECK:STDOUT:   %impl.elem0.loc123_44.1 => constants.%impl.elem0.e3f
// CHECK:STDOUT:   %Result.as_type.loc123_54.1 => constants.%Result.as_type
// CHECK:STDOUT:   %facet_type.loc123_38 => constants.%facet_type.d00
// CHECK:STDOUT:   %T.loc123_4.1 => constants.%T.9be
// CHECK:STDOUT:   %pattern_type.loc123 => constants.%pattern_type.768
// CHECK:STDOUT:   %pattern_type.loc124_16 => constants.%pattern_type.5eec30.2
// CHECK:STDOUT:   %T.as_type.loc124_29.1 => constants.%T.as_type.6fa
// CHECK:STDOUT:   %pattern_type.loc124_27 => constants.%pattern_type.ae2
// CHECK:STDOUT:   %pattern_type.loc124_33 => constants.%pattern_type.a04
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @SubWith.loc135(constants.%Result, constants.%U.866, constants.%T.248) {
// CHECK:STDOUT:   %Result.loc132_9.1 => constants.%Result
// CHECK:STDOUT:   %U.loc133_4.1 => constants.%U.866
// CHECK:STDOUT:   %U.as_type.loc134_36.1 => constants.%U.as_type.0fd
// CHECK:STDOUT:   %SubWith.type.loc134_36.1 => constants.%SubWith.type.3e0
// CHECK:STDOUT:   %facet_type.loc134_20 => constants.%facet_type.b45
// CHECK:STDOUT:   %.Self.loc134_38.1 => constants.%.Self.ecd
// CHECK:STDOUT:   %require_complete.loc134 => constants.%require_complete.029
// CHECK:STDOUT:   %.Self.as_type.loc134_44.1 => constants.%.Self.as_type.8c5
// CHECK:STDOUT:   %SubWith.assoc_type => constants.%SubWith.assoc_type.f11
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.ccf
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.loc134 => constants.%SubWith.lookup_impl_witness.69c
// CHECK:STDOUT:   %impl.elem0.loc134_44.1 => constants.%impl.elem0.35c
// CHECK:STDOUT:   %Result.as_type.loc134_54.1 => constants.%Result.as_type
// CHECK:STDOUT:   %facet_type.loc134_38 => constants.%facet_type.de3
// CHECK:STDOUT:   %T.loc134_4.1 => constants.%T.248
// CHECK:STDOUT:   %pattern_type.loc134 => constants.%pattern_type.6e8
// CHECK:STDOUT:   %pattern_type.loc135_16 => constants.%pattern_type.5eec30.2
// CHECK:STDOUT:   %T.as_type.loc135_29.1 => constants.%T.as_type.3b2
// CHECK:STDOUT:   %pattern_type.loc135_27 => constants.%pattern_type.eb4
// CHECK:STDOUT:   %pattern_type.loc135_33 => constants.%pattern_type.a04
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @MulWith.loc146(constants.%Result, constants.%U.866, constants.%T.ce0) {
// CHECK:STDOUT:   %Result.loc143_9.1 => constants.%Result
// CHECK:STDOUT:   %U.loc144_4.1 => constants.%U.866
// CHECK:STDOUT:   %U.as_type.loc145_36.1 => constants.%U.as_type.0fd
// CHECK:STDOUT:   %MulWith.type.loc145_36.1 => constants.%MulWith.type.b5e
// CHECK:STDOUT:   %facet_type.loc145_20 => constants.%facet_type.7d9
// CHECK:STDOUT:   %.Self.loc145_38.1 => constants.%.Self.e6f
// CHECK:STDOUT:   %require_complete.loc145 => constants.%require_complete.38c
// CHECK:STDOUT:   %.Self.as_type.loc145_44.1 => constants.%.Self.as_type.2c6
// CHECK:STDOUT:   %MulWith.assoc_type => constants.%MulWith.assoc_type.4f3
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.0e6
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.loc145 => constants.%MulWith.lookup_impl_witness.4c7
// CHECK:STDOUT:   %impl.elem0.loc145_44.1 => constants.%impl.elem0.0fa
// CHECK:STDOUT:   %Result.as_type.loc145_54.1 => constants.%Result.as_type
// CHECK:STDOUT:   %facet_type.loc145_38 => constants.%facet_type.779
// CHECK:STDOUT:   %T.loc145_4.1 => constants.%T.ce0
// CHECK:STDOUT:   %pattern_type.loc145 => constants.%pattern_type.2d4
// CHECK:STDOUT:   %pattern_type.loc146_16 => constants.%pattern_type.5eec30.2
// CHECK:STDOUT:   %T.as_type.loc146_29.1 => constants.%T.as_type.380
// CHECK:STDOUT:   %pattern_type.loc146_27 => constants.%pattern_type.4b9
// CHECK:STDOUT:   %pattern_type.loc146_33 => constants.%pattern_type.a04
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @DivWith.loc157(constants.%Result, constants.%U.866, constants.%T.ff8) {
// CHECK:STDOUT:   %Result.loc154_9.1 => constants.%Result
// CHECK:STDOUT:   %U.loc155_4.1 => constants.%U.866
// CHECK:STDOUT:   %U.as_type.loc156_36.1 => constants.%U.as_type.0fd
// CHECK:STDOUT:   %DivWith.type.loc156_36.1 => constants.%DivWith.type.c0c
// CHECK:STDOUT:   %facet_type.loc156_20 => constants.%facet_type.9b5
// CHECK:STDOUT:   %.Self.loc156_38.1 => constants.%.Self.3de
// CHECK:STDOUT:   %require_complete.loc156 => constants.%require_complete.b55
// CHECK:STDOUT:   %.Self.as_type.loc156_44.1 => constants.%.Self.as_type.d38
// CHECK:STDOUT:   %DivWith.assoc_type => constants.%DivWith.assoc_type.cd1
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.453
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.loc156 => constants.%DivWith.lookup_impl_witness.405
// CHECK:STDOUT:   %impl.elem0.loc156_44.1 => constants.%impl.elem0.e26
// CHECK:STDOUT:   %Result.as_type.loc156_54.1 => constants.%Result.as_type
// CHECK:STDOUT:   %facet_type.loc156_38 => constants.%facet_type.c18
// CHECK:STDOUT:   %T.loc156_4.1 => constants.%T.ff8
// CHECK:STDOUT:   %pattern_type.loc156 => constants.%pattern_type.7f8
// CHECK:STDOUT:   %pattern_type.loc157_16 => constants.%pattern_type.5eec30.2
// CHECK:STDOUT:   %T.as_type.loc157_29.1 => constants.%T.as_type.7d7
// CHECK:STDOUT:   %pattern_type.loc157_27 => constants.%pattern_type.4a4
// CHECK:STDOUT:   %pattern_type.loc157_33 => constants.%pattern_type.a04
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ModWith.loc168(constants.%Result, constants.%U.866, constants.%T.5f9) {
// CHECK:STDOUT:   %Result.loc165_9.1 => constants.%Result
// CHECK:STDOUT:   %U.loc166_4.1 => constants.%U.866
// CHECK:STDOUT:   %U.as_type.loc167_36.1 => constants.%U.as_type.0fd
// CHECK:STDOUT:   %ModWith.type.loc167_36.1 => constants.%ModWith.type.fba
// CHECK:STDOUT:   %facet_type.loc167_20 => constants.%facet_type.1e9
// CHECK:STDOUT:   %.Self.loc167_38.1 => constants.%.Self.ade
// CHECK:STDOUT:   %require_complete.loc167 => constants.%require_complete.797
// CHECK:STDOUT:   %.Self.as_type.loc167_44.1 => constants.%.Self.as_type.142
// CHECK:STDOUT:   %ModWith.assoc_type => constants.%ModWith.assoc_type.e42
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.4ea
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.loc167 => constants.%ModWith.lookup_impl_witness.2ca
// CHECK:STDOUT:   %impl.elem0.loc167_44.1 => constants.%impl.elem0.0c4
// CHECK:STDOUT:   %Result.as_type.loc167_54.1 => constants.%Result.as_type
// CHECK:STDOUT:   %facet_type.loc167_38 => constants.%facet_type.ae7
// CHECK:STDOUT:   %T.loc167_4.1 => constants.%T.5f9
// CHECK:STDOUT:   %pattern_type.loc167 => constants.%pattern_type.d24
// CHECK:STDOUT:   %pattern_type.loc168_16 => constants.%pattern_type.5eec30.2
// CHECK:STDOUT:   %T.as_type.loc168_29.1 => constants.%T.as_type.c82
// CHECK:STDOUT:   %pattern_type.loc168_27 => constants.%pattern_type.b5f
// CHECK:STDOUT:   %pattern_type.loc168_33 => constants.%pattern_type.a04
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AddWith.loc124(constants.%Destroy.facet.738, constants.%Destroy.facet.738, constants.%facet_value.70e) {
// CHECK:STDOUT:   %Result.loc121_9.1 => constants.%Destroy.facet.738
// CHECK:STDOUT:   %U.loc122_4.1 => constants.%Destroy.facet.738
// CHECK:STDOUT:   %U.as_type.loc123_36.1 => constants.%Int16
// CHECK:STDOUT:   %AddWith.type.loc123_36.1 => constants.%AddWith.type.116
// CHECK:STDOUT:   %facet_type.loc123_20 => constants.%facet_type.aa6
// CHECK:STDOUT:   %.Self.loc123_38.1 => constants.%.Self.cf1
// CHECK:STDOUT:   %require_complete.loc123 => constants.%complete_type.a93
// CHECK:STDOUT:   %.Self.as_type.loc123_44.1 => constants.%.Self.as_type.70cc
// CHECK:STDOUT:   %AddWith.assoc_type => constants.%AddWith.assoc_type.dda
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.c76
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.loc123 => constants.%AddWith.lookup_impl_witness.d17
// CHECK:STDOUT:   %impl.elem0.loc123_44.1 => constants.%impl.elem0.9d9
// CHECK:STDOUT:   %Result.as_type.loc123_54.1 => constants.%Int16
// CHECK:STDOUT:   %facet_type.loc123_38 => constants.%facet_type.f01
// CHECK:STDOUT:   %T.loc123_4.1 => constants.%facet_value.70e
// CHECK:STDOUT:   %pattern_type.loc123 => constants.%pattern_type.616
// CHECK:STDOUT:   %pattern_type.loc124_16 => constants.%pattern_type.004
// CHECK:STDOUT:   %T.as_type.loc124_29.1 => constants.%i16
// CHECK:STDOUT:   %pattern_type.loc124_27 => constants.%pattern_type.0cb
// CHECK:STDOUT:   %pattern_type.loc124_33 => constants.%pattern_type.004
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc124_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc124_27 => constants.%complete_type.2da
// CHECK:STDOUT:   %require_complete.loc124_33 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc126 => constants.%complete_type.eb1
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.83e
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.loc126 => constants.%custom_witness.91a
// CHECK:STDOUT:   %AddWith.facet => constants.%AddWith.facet.b43
// CHECK:STDOUT:   %AddWith.WithSelf.Op.type => constants.%AddWith.WithSelf.Op.type.af0
// CHECK:STDOUT:   %.loc126_5.4 => constants.%.6ba
// CHECK:STDOUT:   %impl.elem1.loc126_5.2 => constants.%Op.b9c47d.1
// CHECK:STDOUT:   %specific_impl_fn.loc126_5.3 => constants.%Op.b9c47d.1
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.7e2
// CHECK:STDOUT:   %.loc126_5.5 => constants.%.d2d
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.f83583.1
// CHECK:STDOUT:   %impl.elem0.loc126_5.2 => constants.%Int16.Op.ecb1a5.1
// CHECK:STDOUT:   %specific_impl_fn.loc126_5.4 => constants.%Int16.Op.ecb1a5.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @SubWith.loc135(constants.%Destroy.facet.738, constants.%Destroy.facet.738, constants.%facet_value.677) {
// CHECK:STDOUT:   %Result.loc132_9.1 => constants.%Destroy.facet.738
// CHECK:STDOUT:   %U.loc133_4.1 => constants.%Destroy.facet.738
// CHECK:STDOUT:   %U.as_type.loc134_36.1 => constants.%Int16
// CHECK:STDOUT:   %SubWith.type.loc134_36.1 => constants.%SubWith.type.49a
// CHECK:STDOUT:   %facet_type.loc134_20 => constants.%facet_type.4d6
// CHECK:STDOUT:   %.Self.loc134_38.1 => constants.%.Self.c3b
// CHECK:STDOUT:   %require_complete.loc134 => constants.%complete_type.4fe
// CHECK:STDOUT:   %.Self.as_type.loc134_44.1 => constants.%.Self.as_type.42f
// CHECK:STDOUT:   %SubWith.assoc_type => constants.%SubWith.assoc_type.445
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.c71
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.loc134 => constants.%SubWith.lookup_impl_witness.72f
// CHECK:STDOUT:   %impl.elem0.loc134_44.1 => constants.%impl.elem0.b83
// CHECK:STDOUT:   %Result.as_type.loc134_54.1 => constants.%Int16
// CHECK:STDOUT:   %facet_type.loc134_38 => constants.%facet_type.d9b
// CHECK:STDOUT:   %T.loc134_4.1 => constants.%facet_value.677
// CHECK:STDOUT:   %pattern_type.loc134 => constants.%pattern_type.ff0
// CHECK:STDOUT:   %pattern_type.loc135_16 => constants.%pattern_type.004
// CHECK:STDOUT:   %T.as_type.loc135_29.1 => constants.%i16
// CHECK:STDOUT:   %pattern_type.loc135_27 => constants.%pattern_type.0cb
// CHECK:STDOUT:   %pattern_type.loc135_33 => constants.%pattern_type.004
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc135_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc135_27 => constants.%complete_type.2da
// CHECK:STDOUT:   %require_complete.loc135_33 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc137 => constants.%complete_type.99b
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.a3f
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.loc137 => constants.%custom_witness.7ee
// CHECK:STDOUT:   %SubWith.facet => constants.%SubWith.facet.428
// CHECK:STDOUT:   %SubWith.WithSelf.Op.type => constants.%SubWith.WithSelf.Op.type.196
// CHECK:STDOUT:   %.loc137_5.4 => constants.%.ee9
// CHECK:STDOUT:   %impl.elem1.loc137_5.2 => constants.%Op.b9c47d.2
// CHECK:STDOUT:   %specific_impl_fn.loc137_5.3 => constants.%Op.b9c47d.2
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.7e2
// CHECK:STDOUT:   %.loc137_5.5 => constants.%.d2d
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.f83583.1
// CHECK:STDOUT:   %impl.elem0.loc137_5.2 => constants.%Int16.Op.ecb1a5.1
// CHECK:STDOUT:   %specific_impl_fn.loc137_5.4 => constants.%Int16.Op.ecb1a5.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @MulWith.loc146(constants.%Destroy.facet.738, constants.%Destroy.facet.738, constants.%facet_value.661) {
// CHECK:STDOUT:   %Result.loc143_9.1 => constants.%Destroy.facet.738
// CHECK:STDOUT:   %U.loc144_4.1 => constants.%Destroy.facet.738
// CHECK:STDOUT:   %U.as_type.loc145_36.1 => constants.%Int16
// CHECK:STDOUT:   %MulWith.type.loc145_36.1 => constants.%MulWith.type.779
// CHECK:STDOUT:   %facet_type.loc145_20 => constants.%facet_type.624
// CHECK:STDOUT:   %.Self.loc145_38.1 => constants.%.Self.164
// CHECK:STDOUT:   %require_complete.loc145 => constants.%complete_type.ec6
// CHECK:STDOUT:   %.Self.as_type.loc145_44.1 => constants.%.Self.as_type.49c
// CHECK:STDOUT:   %MulWith.assoc_type => constants.%MulWith.assoc_type.3a9
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.97a
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.loc145 => constants.%MulWith.lookup_impl_witness.588
// CHECK:STDOUT:   %impl.elem0.loc145_44.1 => constants.%impl.elem0.66a
// CHECK:STDOUT:   %Result.as_type.loc145_54.1 => constants.%Int16
// CHECK:STDOUT:   %facet_type.loc145_38 => constants.%facet_type.cc5
// CHECK:STDOUT:   %T.loc145_4.1 => constants.%facet_value.661
// CHECK:STDOUT:   %pattern_type.loc145 => constants.%pattern_type.da6
// CHECK:STDOUT:   %pattern_type.loc146_16 => constants.%pattern_type.004
// CHECK:STDOUT:   %T.as_type.loc146_29.1 => constants.%i16
// CHECK:STDOUT:   %pattern_type.loc146_27 => constants.%pattern_type.0cb
// CHECK:STDOUT:   %pattern_type.loc146_33 => constants.%pattern_type.004
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc146_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc146_27 => constants.%complete_type.2da
// CHECK:STDOUT:   %require_complete.loc146_33 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc148 => constants.%complete_type.255
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.a46
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.loc148 => constants.%custom_witness.448
// CHECK:STDOUT:   %MulWith.facet => constants.%MulWith.facet.48c
// CHECK:STDOUT:   %MulWith.WithSelf.Op.type => constants.%MulWith.WithSelf.Op.type.1d0
// CHECK:STDOUT:   %.loc148_5.4 => constants.%.03b
// CHECK:STDOUT:   %impl.elem1.loc148_5.2 => constants.%Op.b9c47d.3
// CHECK:STDOUT:   %specific_impl_fn.loc148_5.3 => constants.%Op.b9c47d.3
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.7e2
// CHECK:STDOUT:   %.loc148_5.5 => constants.%.d2d
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.f83583.1
// CHECK:STDOUT:   %impl.elem0.loc148_5.2 => constants.%Int16.Op.ecb1a5.1
// CHECK:STDOUT:   %specific_impl_fn.loc148_5.4 => constants.%Int16.Op.ecb1a5.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @DivWith.loc157(constants.%Destroy.facet.738, constants.%Destroy.facet.738, constants.%facet_value.cf3) {
// CHECK:STDOUT:   %Result.loc154_9.1 => constants.%Destroy.facet.738
// CHECK:STDOUT:   %U.loc155_4.1 => constants.%Destroy.facet.738
// CHECK:STDOUT:   %U.as_type.loc156_36.1 => constants.%Int16
// CHECK:STDOUT:   %DivWith.type.loc156_36.1 => constants.%DivWith.type.35b
// CHECK:STDOUT:   %facet_type.loc156_20 => constants.%facet_type.b9b
// CHECK:STDOUT:   %.Self.loc156_38.1 => constants.%.Self.7ba
// CHECK:STDOUT:   %require_complete.loc156 => constants.%complete_type.7df
// CHECK:STDOUT:   %.Self.as_type.loc156_44.1 => constants.%.Self.as_type.31a
// CHECK:STDOUT:   %DivWith.assoc_type => constants.%DivWith.assoc_type.6ae
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.1bc
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.loc156 => constants.%DivWith.lookup_impl_witness.836
// CHECK:STDOUT:   %impl.elem0.loc156_44.1 => constants.%impl.elem0.6cb
// CHECK:STDOUT:   %Result.as_type.loc156_54.1 => constants.%Int16
// CHECK:STDOUT:   %facet_type.loc156_38 => constants.%facet_type.1b5
// CHECK:STDOUT:   %T.loc156_4.1 => constants.%facet_value.cf3
// CHECK:STDOUT:   %pattern_type.loc156 => constants.%pattern_type.ec4
// CHECK:STDOUT:   %pattern_type.loc157_16 => constants.%pattern_type.004
// CHECK:STDOUT:   %T.as_type.loc157_29.1 => constants.%i16
// CHECK:STDOUT:   %pattern_type.loc157_27 => constants.%pattern_type.0cb
// CHECK:STDOUT:   %pattern_type.loc157_33 => constants.%pattern_type.004
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc157_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc157_27 => constants.%complete_type.2da
// CHECK:STDOUT:   %require_complete.loc157_33 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc159 => constants.%complete_type.c6d
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.5a7
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.loc159 => constants.%custom_witness.3be
// CHECK:STDOUT:   %DivWith.facet => constants.%DivWith.facet.b79
// CHECK:STDOUT:   %DivWith.WithSelf.Op.type => constants.%DivWith.WithSelf.Op.type.b57
// CHECK:STDOUT:   %.loc159_5.4 => constants.%.c89
// CHECK:STDOUT:   %impl.elem1.loc159_5.2 => constants.%Op.b9c47d.4
// CHECK:STDOUT:   %specific_impl_fn.loc159_5.3 => constants.%Op.b9c47d.4
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.7e2
// CHECK:STDOUT:   %.loc159_5.5 => constants.%.d2d
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.f83583.1
// CHECK:STDOUT:   %impl.elem0.loc159_5.2 => constants.%Int16.Op.ecb1a5.1
// CHECK:STDOUT:   %specific_impl_fn.loc159_5.4 => constants.%Int16.Op.ecb1a5.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ModWith.loc168(constants.%Destroy.facet.738, constants.%Destroy.facet.738, constants.%facet_value.ef3) {
// CHECK:STDOUT:   %Result.loc165_9.1 => constants.%Destroy.facet.738
// CHECK:STDOUT:   %U.loc166_4.1 => constants.%Destroy.facet.738
// CHECK:STDOUT:   %U.as_type.loc167_36.1 => constants.%Int16
// CHECK:STDOUT:   %ModWith.type.loc167_36.1 => constants.%ModWith.type.3ef
// CHECK:STDOUT:   %facet_type.loc167_20 => constants.%facet_type.ca3
// CHECK:STDOUT:   %.Self.loc167_38.1 => constants.%.Self.cfe
// CHECK:STDOUT:   %require_complete.loc167 => constants.%complete_type.1cf
// CHECK:STDOUT:   %.Self.as_type.loc167_44.1 => constants.%.Self.as_type.da7
// CHECK:STDOUT:   %ModWith.assoc_type => constants.%ModWith.assoc_type.744
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.727
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.loc167 => constants.%ModWith.lookup_impl_witness.305
// CHECK:STDOUT:   %impl.elem0.loc167_44.1 => constants.%impl.elem0.2b9
// CHECK:STDOUT:   %Result.as_type.loc167_54.1 => constants.%Int16
// CHECK:STDOUT:   %facet_type.loc167_38 => constants.%facet_type.9a8
// CHECK:STDOUT:   %T.loc167_4.1 => constants.%facet_value.ef3
// CHECK:STDOUT:   %pattern_type.loc167 => constants.%pattern_type.a3b
// CHECK:STDOUT:   %pattern_type.loc168_16 => constants.%pattern_type.004
// CHECK:STDOUT:   %T.as_type.loc168_29.1 => constants.%i16
// CHECK:STDOUT:   %pattern_type.loc168_27 => constants.%pattern_type.0cb
// CHECK:STDOUT:   %pattern_type.loc168_33 => constants.%pattern_type.004
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc168_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc168_27 => constants.%complete_type.2da
// CHECK:STDOUT:   %require_complete.loc168_33 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc170 => constants.%complete_type.55f
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.ba4
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.loc170 => constants.%custom_witness.dcf
// CHECK:STDOUT:   %ModWith.facet => constants.%ModWith.facet.8a9
// CHECK:STDOUT:   %ModWith.WithSelf.Op.type => constants.%ModWith.WithSelf.Op.type.06b
// CHECK:STDOUT:   %.loc170_5.4 => constants.%.d15
// CHECK:STDOUT:   %impl.elem1.loc170_5.2 => constants.%Op.b9c47d.5
// CHECK:STDOUT:   %specific_impl_fn.loc170_5.3 => constants.%Op.b9c47d.5
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.7e2
// CHECK:STDOUT:   %.loc170_5.5 => constants.%.d2d
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.f83583.1
// CHECK:STDOUT:   %impl.elem0.loc170_5.2 => constants.%Int16.Op.ecb1a5.1
// CHECK:STDOUT:   %specific_impl_fn.loc170_5.4 => constants.%Int16.Op.ecb1a5.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AddWith.loc124(constants.%Destroy.facet.738, constants.%Destroy.facet.6e0, constants.%facet_value.ea8) {
// CHECK:STDOUT:   %Result.loc121_9.1 => constants.%Destroy.facet.738
// CHECK:STDOUT:   %U.loc122_4.1 => constants.%Destroy.facet.6e0
// CHECK:STDOUT:   %U.as_type.loc123_36.1 => constants.%i16
// CHECK:STDOUT:   %AddWith.type.loc123_36.1 => constants.%AddWith.type.a6e
// CHECK:STDOUT:   %facet_type.loc123_20 => constants.%facet_type.074
// CHECK:STDOUT:   %.Self.loc123_38.1 => constants.%.Self.a3f
// CHECK:STDOUT:   %require_complete.loc123 => constants.%complete_type.fbb
// CHECK:STDOUT:   %.Self.as_type.loc123_44.1 => constants.%.Self.as_type.1c2
// CHECK:STDOUT:   %AddWith.assoc_type => constants.%AddWith.assoc_type.6b4
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.8ca
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.loc123 => constants.%AddWith.lookup_impl_witness.e89
// CHECK:STDOUT:   %impl.elem0.loc123_44.1 => constants.%impl.elem0.4db
// CHECK:STDOUT:   %Result.as_type.loc123_54.1 => constants.%Int16
// CHECK:STDOUT:   %facet_type.loc123_38 => constants.%facet_type.854
// CHECK:STDOUT:   %T.loc123_4.1 => constants.%facet_value.ea8
// CHECK:STDOUT:   %pattern_type.loc123 => constants.%pattern_type.7902
// CHECK:STDOUT:   %pattern_type.loc124_16 => constants.%pattern_type.004
// CHECK:STDOUT:   %T.as_type.loc124_29.1 => constants.%Int16
// CHECK:STDOUT:   %pattern_type.loc124_27 => constants.%pattern_type.004
// CHECK:STDOUT:   %pattern_type.loc124_33 => constants.%pattern_type.0cb
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc124_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc124_27 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc124_33 => constants.%complete_type.2da
// CHECK:STDOUT:   %require_complete.loc126 => constants.%complete_type.def
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.864
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.loc126 => constants.%custom_witness.0db
// CHECK:STDOUT:   %AddWith.facet => constants.%AddWith.facet.ea1
// CHECK:STDOUT:   %AddWith.WithSelf.Op.type => constants.%AddWith.WithSelf.Op.type.c4d
// CHECK:STDOUT:   %.loc126_5.4 => constants.%.5e8
// CHECK:STDOUT:   %impl.elem1.loc126_5.2 => constants.%Int16.Op.ecb1a5.3
// CHECK:STDOUT:   %specific_impl_fn.loc126_5.3 => constants.%Int16.Op.ecb1a5.3
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.7e2
// CHECK:STDOUT:   %.loc126_5.5 => constants.%.d2d
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.f83583.1
// CHECK:STDOUT:   %impl.elem0.loc126_5.2 => constants.%Int16.Op.ecb1a5.1
// CHECK:STDOUT:   %specific_impl_fn.loc126_5.4 => constants.%Int16.Op.ecb1a5.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @SubWith.loc135(constants.%Destroy.facet.738, constants.%Destroy.facet.6e0, constants.%facet_value.62c) {
// CHECK:STDOUT:   %Result.loc132_9.1 => constants.%Destroy.facet.738
// CHECK:STDOUT:   %U.loc133_4.1 => constants.%Destroy.facet.6e0
// CHECK:STDOUT:   %U.as_type.loc134_36.1 => constants.%i16
// CHECK:STDOUT:   %SubWith.type.loc134_36.1 => constants.%SubWith.type.2f5
// CHECK:STDOUT:   %facet_type.loc134_20 => constants.%facet_type.352
// CHECK:STDOUT:   %.Self.loc134_38.1 => constants.%.Self.982
// CHECK:STDOUT:   %require_complete.loc134 => constants.%complete_type.757
// CHECK:STDOUT:   %.Self.as_type.loc134_44.1 => constants.%.Self.as_type.311
// CHECK:STDOUT:   %SubWith.assoc_type => constants.%SubWith.assoc_type.70d
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.367
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.loc134 => constants.%SubWith.lookup_impl_witness.7e3
// CHECK:STDOUT:   %impl.elem0.loc134_44.1 => constants.%impl.elem0.aa0
// CHECK:STDOUT:   %Result.as_type.loc134_54.1 => constants.%Int16
// CHECK:STDOUT:   %facet_type.loc134_38 => constants.%facet_type.712
// CHECK:STDOUT:   %T.loc134_4.1 => constants.%facet_value.62c
// CHECK:STDOUT:   %pattern_type.loc134 => constants.%pattern_type.f30
// CHECK:STDOUT:   %pattern_type.loc135_16 => constants.%pattern_type.004
// CHECK:STDOUT:   %T.as_type.loc135_29.1 => constants.%Int16
// CHECK:STDOUT:   %pattern_type.loc135_27 => constants.%pattern_type.004
// CHECK:STDOUT:   %pattern_type.loc135_33 => constants.%pattern_type.0cb
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc135_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc135_27 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc135_33 => constants.%complete_type.2da
// CHECK:STDOUT:   %require_complete.loc137 => constants.%complete_type.8c1
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.6ca
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.loc137 => constants.%custom_witness.2f4
// CHECK:STDOUT:   %SubWith.facet => constants.%SubWith.facet.b1e
// CHECK:STDOUT:   %SubWith.WithSelf.Op.type => constants.%SubWith.WithSelf.Op.type.fa8
// CHECK:STDOUT:   %.loc137_5.4 => constants.%.26c
// CHECK:STDOUT:   %impl.elem1.loc137_5.2 => constants.%Int16.Op.ecb1a5.4
// CHECK:STDOUT:   %specific_impl_fn.loc137_5.3 => constants.%Int16.Op.ecb1a5.4
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.7e2
// CHECK:STDOUT:   %.loc137_5.5 => constants.%.d2d
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.f83583.1
// CHECK:STDOUT:   %impl.elem0.loc137_5.2 => constants.%Int16.Op.ecb1a5.1
// CHECK:STDOUT:   %specific_impl_fn.loc137_5.4 => constants.%Int16.Op.ecb1a5.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @MulWith.loc146(constants.%Destroy.facet.738, constants.%Destroy.facet.6e0, constants.%facet_value.ce7) {
// CHECK:STDOUT:   %Result.loc143_9.1 => constants.%Destroy.facet.738
// CHECK:STDOUT:   %U.loc144_4.1 => constants.%Destroy.facet.6e0
// CHECK:STDOUT:   %U.as_type.loc145_36.1 => constants.%i16
// CHECK:STDOUT:   %MulWith.type.loc145_36.1 => constants.%MulWith.type.278
// CHECK:STDOUT:   %facet_type.loc145_20 => constants.%facet_type.bff
// CHECK:STDOUT:   %.Self.loc145_38.1 => constants.%.Self.92d
// CHECK:STDOUT:   %require_complete.loc145 => constants.%complete_type.4fc
// CHECK:STDOUT:   %.Self.as_type.loc145_44.1 => constants.%.Self.as_type.f34
// CHECK:STDOUT:   %MulWith.assoc_type => constants.%MulWith.assoc_type.8cf
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.de2
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.loc145 => constants.%MulWith.lookup_impl_witness.fd5
// CHECK:STDOUT:   %impl.elem0.loc145_44.1 => constants.%impl.elem0.004
// CHECK:STDOUT:   %Result.as_type.loc145_54.1 => constants.%Int16
// CHECK:STDOUT:   %facet_type.loc145_38 => constants.%facet_type.e32
// CHECK:STDOUT:   %T.loc145_4.1 => constants.%facet_value.ce7
// CHECK:STDOUT:   %pattern_type.loc145 => constants.%pattern_type.cfd
// CHECK:STDOUT:   %pattern_type.loc146_16 => constants.%pattern_type.004
// CHECK:STDOUT:   %T.as_type.loc146_29.1 => constants.%Int16
// CHECK:STDOUT:   %pattern_type.loc146_27 => constants.%pattern_type.004
// CHECK:STDOUT:   %pattern_type.loc146_33 => constants.%pattern_type.0cb
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc146_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc146_27 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc146_33 => constants.%complete_type.2da
// CHECK:STDOUT:   %require_complete.loc148 => constants.%complete_type.514
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.c99
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.loc148 => constants.%custom_witness.d98
// CHECK:STDOUT:   %MulWith.facet => constants.%MulWith.facet.391
// CHECK:STDOUT:   %MulWith.WithSelf.Op.type => constants.%MulWith.WithSelf.Op.type.f65
// CHECK:STDOUT:   %.loc148_5.4 => constants.%.796
// CHECK:STDOUT:   %impl.elem1.loc148_5.2 => constants.%Int16.Op.ecb1a5.5
// CHECK:STDOUT:   %specific_impl_fn.loc148_5.3 => constants.%Int16.Op.ecb1a5.5
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.7e2
// CHECK:STDOUT:   %.loc148_5.5 => constants.%.d2d
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.f83583.1
// CHECK:STDOUT:   %impl.elem0.loc148_5.2 => constants.%Int16.Op.ecb1a5.1
// CHECK:STDOUT:   %specific_impl_fn.loc148_5.4 => constants.%Int16.Op.ecb1a5.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @DivWith.loc157(constants.%Destroy.facet.738, constants.%Destroy.facet.6e0, constants.%facet_value.1c8) {
// CHECK:STDOUT:   %Result.loc154_9.1 => constants.%Destroy.facet.738
// CHECK:STDOUT:   %U.loc155_4.1 => constants.%Destroy.facet.6e0
// CHECK:STDOUT:   %U.as_type.loc156_36.1 => constants.%i16
// CHECK:STDOUT:   %DivWith.type.loc156_36.1 => constants.%DivWith.type.9b1
// CHECK:STDOUT:   %facet_type.loc156_20 => constants.%facet_type.223
// CHECK:STDOUT:   %.Self.loc156_38.1 => constants.%.Self.867
// CHECK:STDOUT:   %require_complete.loc156 => constants.%complete_type.ddd
// CHECK:STDOUT:   %.Self.as_type.loc156_44.1 => constants.%.Self.as_type.3fc
// CHECK:STDOUT:   %DivWith.assoc_type => constants.%DivWith.assoc_type.746
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.13d
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.loc156 => constants.%DivWith.lookup_impl_witness.7f7
// CHECK:STDOUT:   %impl.elem0.loc156_44.1 => constants.%impl.elem0.e0d
// CHECK:STDOUT:   %Result.as_type.loc156_54.1 => constants.%Int16
// CHECK:STDOUT:   %facet_type.loc156_38 => constants.%facet_type.fd7
// CHECK:STDOUT:   %T.loc156_4.1 => constants.%facet_value.1c8
// CHECK:STDOUT:   %pattern_type.loc156 => constants.%pattern_type.5f9
// CHECK:STDOUT:   %pattern_type.loc157_16 => constants.%pattern_type.004
// CHECK:STDOUT:   %T.as_type.loc157_29.1 => constants.%Int16
// CHECK:STDOUT:   %pattern_type.loc157_27 => constants.%pattern_type.004
// CHECK:STDOUT:   %pattern_type.loc157_33 => constants.%pattern_type.0cb
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc157_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc157_27 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc157_33 => constants.%complete_type.2da
// CHECK:STDOUT:   %require_complete.loc159 => constants.%complete_type.51d
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.84d
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.loc159 => constants.%custom_witness.006
// CHECK:STDOUT:   %DivWith.facet => constants.%DivWith.facet.b5b
// CHECK:STDOUT:   %DivWith.WithSelf.Op.type => constants.%DivWith.WithSelf.Op.type.fb7
// CHECK:STDOUT:   %.loc159_5.4 => constants.%.14a
// CHECK:STDOUT:   %impl.elem1.loc159_5.2 => constants.%Int16.Op.ecb1a5.6
// CHECK:STDOUT:   %specific_impl_fn.loc159_5.3 => constants.%Int16.Op.ecb1a5.6
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.7e2
// CHECK:STDOUT:   %.loc159_5.5 => constants.%.d2d
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.f83583.1
// CHECK:STDOUT:   %impl.elem0.loc159_5.2 => constants.%Int16.Op.ecb1a5.1
// CHECK:STDOUT:   %specific_impl_fn.loc159_5.4 => constants.%Int16.Op.ecb1a5.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ModWith.loc168(constants.%Destroy.facet.738, constants.%Destroy.facet.6e0, constants.%facet_value.322) {
// CHECK:STDOUT:   %Result.loc165_9.1 => constants.%Destroy.facet.738
// CHECK:STDOUT:   %U.loc166_4.1 => constants.%Destroy.facet.6e0
// CHECK:STDOUT:   %U.as_type.loc167_36.1 => constants.%i16
// CHECK:STDOUT:   %ModWith.type.loc167_36.1 => constants.%ModWith.type.59b
// CHECK:STDOUT:   %facet_type.loc167_20 => constants.%facet_type.c0c
// CHECK:STDOUT:   %.Self.loc167_38.1 => constants.%.Self.762
// CHECK:STDOUT:   %require_complete.loc167 => constants.%complete_type.e02
// CHECK:STDOUT:   %.Self.as_type.loc167_44.1 => constants.%.Self.as_type.68f
// CHECK:STDOUT:   %ModWith.assoc_type => constants.%ModWith.assoc_type.3a6
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.2b8
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.loc167 => constants.%ModWith.lookup_impl_witness.cfe
// CHECK:STDOUT:   %impl.elem0.loc167_44.1 => constants.%impl.elem0.4d9
// CHECK:STDOUT:   %Result.as_type.loc167_54.1 => constants.%Int16
// CHECK:STDOUT:   %facet_type.loc167_38 => constants.%facet_type.edf
// CHECK:STDOUT:   %T.loc167_4.1 => constants.%facet_value.322
// CHECK:STDOUT:   %pattern_type.loc167 => constants.%pattern_type.83e
// CHECK:STDOUT:   %pattern_type.loc168_16 => constants.%pattern_type.004
// CHECK:STDOUT:   %T.as_type.loc168_29.1 => constants.%Int16
// CHECK:STDOUT:   %pattern_type.loc168_27 => constants.%pattern_type.004
// CHECK:STDOUT:   %pattern_type.loc168_33 => constants.%pattern_type.0cb
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc168_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc168_27 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc168_33 => constants.%complete_type.2da
// CHECK:STDOUT:   %require_complete.loc170 => constants.%complete_type.155
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.a21
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.loc170 => constants.%custom_witness.e08
// CHECK:STDOUT:   %ModWith.facet => constants.%ModWith.facet.cde
// CHECK:STDOUT:   %ModWith.WithSelf.Op.type => constants.%ModWith.WithSelf.Op.type.f06
// CHECK:STDOUT:   %.loc170_5.4 => constants.%.7a7
// CHECK:STDOUT:   %impl.elem1.loc170_5.2 => constants.%Int16.Op.ecb1a5.7
// CHECK:STDOUT:   %specific_impl_fn.loc170_5.3 => constants.%Int16.Op.ecb1a5.7
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.7e2
// CHECK:STDOUT:   %.loc170_5.5 => constants.%.d2d
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.f83583.1
// CHECK:STDOUT:   %impl.elem0.loc170_5.2 => constants.%Int16.Op.ecb1a5.1
// CHECK:STDOUT:   %specific_impl_fn.loc170_5.4 => constants.%Int16.Op.ecb1a5.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AddWith.loc124(constants.%Destroy.facet.700, constants.%Destroy.facet.700, constants.%facet_value.228) {
// CHECK:STDOUT:   %Result.loc121_9.1 => constants.%Destroy.facet.700
// CHECK:STDOUT:   %U.loc122_4.1 => constants.%Destroy.facet.700
// CHECK:STDOUT:   %U.as_type.loc123_36.1 => constants.%Int32
// CHECK:STDOUT:   %AddWith.type.loc123_36.1 => constants.%AddWith.type.c13
// CHECK:STDOUT:   %facet_type.loc123_20 => constants.%facet_type.f56
// CHECK:STDOUT:   %.Self.loc123_38.1 => constants.%.Self.977
// CHECK:STDOUT:   %require_complete.loc123 => constants.%complete_type.39f
// CHECK:STDOUT:   %.Self.as_type.loc123_44.1 => constants.%.Self.as_type.70c2
// CHECK:STDOUT:   %AddWith.assoc_type => constants.%AddWith.assoc_type.f6f
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.41e
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.loc123 => constants.%AddWith.lookup_impl_witness.883
// CHECK:STDOUT:   %impl.elem0.loc123_44.1 => constants.%impl.elem0.2e9
// CHECK:STDOUT:   %Result.as_type.loc123_54.1 => constants.%Int32
// CHECK:STDOUT:   %facet_type.loc123_38 => constants.%facet_type.36e
// CHECK:STDOUT:   %T.loc123_4.1 => constants.%facet_value.228
// CHECK:STDOUT:   %pattern_type.loc123 => constants.%pattern_type.204
// CHECK:STDOUT:   %pattern_type.loc124_16 => constants.%pattern_type.3b9
// CHECK:STDOUT:   %T.as_type.loc124_29.1 => constants.%Int16
// CHECK:STDOUT:   %pattern_type.loc124_27 => constants.%pattern_type.004
// CHECK:STDOUT:   %pattern_type.loc124_33 => constants.%pattern_type.3b9
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc124_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc124_27 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc124_33 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc126 => constants.%complete_type.f05
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.519
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.loc126 => constants.%custom_witness.27d802.1
// CHECK:STDOUT:   %AddWith.facet => constants.%AddWith.facet.013
// CHECK:STDOUT:   %AddWith.WithSelf.Op.type => constants.%AddWith.WithSelf.Op.type.c77
// CHECK:STDOUT:   %.loc126_5.4 => constants.%.396
// CHECK:STDOUT:   %impl.elem1.loc126_5.2 => constants.%Int16.Op.ecb1a5.8
// CHECK:STDOUT:   %specific_impl_fn.loc126_5.3 => constants.%Int16.Op.ecb1a5.8
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.7bc
// CHECK:STDOUT:   %.loc126_5.5 => constants.%.362
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.f83583.2
// CHECK:STDOUT:   %impl.elem0.loc126_5.2 => constants.%Int32.Op.ecb1a5.1
// CHECK:STDOUT:   %specific_impl_fn.loc126_5.4 => constants.%Int32.Op.ecb1a5.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @SubWith.loc135(constants.%Destroy.facet.700, constants.%Destroy.facet.700, constants.%facet_value.f72) {
// CHECK:STDOUT:   %Result.loc132_9.1 => constants.%Destroy.facet.700
// CHECK:STDOUT:   %U.loc133_4.1 => constants.%Destroy.facet.700
// CHECK:STDOUT:   %U.as_type.loc134_36.1 => constants.%Int32
// CHECK:STDOUT:   %SubWith.type.loc134_36.1 => constants.%SubWith.type.554
// CHECK:STDOUT:   %facet_type.loc134_20 => constants.%facet_type.eb0
// CHECK:STDOUT:   %.Self.loc134_38.1 => constants.%.Self.604
// CHECK:STDOUT:   %require_complete.loc134 => constants.%complete_type.872
// CHECK:STDOUT:   %.Self.as_type.loc134_44.1 => constants.%.Self.as_type.d34
// CHECK:STDOUT:   %SubWith.assoc_type => constants.%SubWith.assoc_type.6db
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.053
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.loc134 => constants.%SubWith.lookup_impl_witness.8be
// CHECK:STDOUT:   %impl.elem0.loc134_44.1 => constants.%impl.elem0.7af
// CHECK:STDOUT:   %Result.as_type.loc134_54.1 => constants.%Int32
// CHECK:STDOUT:   %facet_type.loc134_38 => constants.%facet_type.dfd
// CHECK:STDOUT:   %T.loc134_4.1 => constants.%facet_value.f72
// CHECK:STDOUT:   %pattern_type.loc134 => constants.%pattern_type.b5c
// CHECK:STDOUT:   %pattern_type.loc135_16 => constants.%pattern_type.3b9
// CHECK:STDOUT:   %T.as_type.loc135_29.1 => constants.%Int16
// CHECK:STDOUT:   %pattern_type.loc135_27 => constants.%pattern_type.004
// CHECK:STDOUT:   %pattern_type.loc135_33 => constants.%pattern_type.3b9
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc135_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc135_27 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc135_33 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc137 => constants.%complete_type.60a
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.3e8
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.loc137 => constants.%custom_witness.f4a1b8.1
// CHECK:STDOUT:   %SubWith.facet => constants.%SubWith.facet.cc1
// CHECK:STDOUT:   %SubWith.WithSelf.Op.type => constants.%SubWith.WithSelf.Op.type.3ad
// CHECK:STDOUT:   %.loc137_5.4 => constants.%.ed8
// CHECK:STDOUT:   %impl.elem1.loc137_5.2 => constants.%Int16.Op.ecb1a5.9
// CHECK:STDOUT:   %specific_impl_fn.loc137_5.3 => constants.%Int16.Op.ecb1a5.9
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.7bc
// CHECK:STDOUT:   %.loc137_5.5 => constants.%.362
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.f83583.2
// CHECK:STDOUT:   %impl.elem0.loc137_5.2 => constants.%Int32.Op.ecb1a5.1
// CHECK:STDOUT:   %specific_impl_fn.loc137_5.4 => constants.%Int32.Op.ecb1a5.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @MulWith.loc146(constants.%Destroy.facet.700, constants.%Destroy.facet.700, constants.%facet_value.3ab) {
// CHECK:STDOUT:   %Result.loc143_9.1 => constants.%Destroy.facet.700
// CHECK:STDOUT:   %U.loc144_4.1 => constants.%Destroy.facet.700
// CHECK:STDOUT:   %U.as_type.loc145_36.1 => constants.%Int32
// CHECK:STDOUT:   %MulWith.type.loc145_36.1 => constants.%MulWith.type.d6e
// CHECK:STDOUT:   %facet_type.loc145_20 => constants.%facet_type.87a
// CHECK:STDOUT:   %.Self.loc145_38.1 => constants.%.Self.485
// CHECK:STDOUT:   %require_complete.loc145 => constants.%complete_type.e5c
// CHECK:STDOUT:   %.Self.as_type.loc145_44.1 => constants.%.Self.as_type.963
// CHECK:STDOUT:   %MulWith.assoc_type => constants.%MulWith.assoc_type.e4f
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.879
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.loc145 => constants.%MulWith.lookup_impl_witness.99b
// CHECK:STDOUT:   %impl.elem0.loc145_44.1 => constants.%impl.elem0.a71
// CHECK:STDOUT:   %Result.as_type.loc145_54.1 => constants.%Int32
// CHECK:STDOUT:   %facet_type.loc145_38 => constants.%facet_type.4d5
// CHECK:STDOUT:   %T.loc145_4.1 => constants.%facet_value.3ab
// CHECK:STDOUT:   %pattern_type.loc145 => constants.%pattern_type.459
// CHECK:STDOUT:   %pattern_type.loc146_16 => constants.%pattern_type.3b9
// CHECK:STDOUT:   %T.as_type.loc146_29.1 => constants.%Int16
// CHECK:STDOUT:   %pattern_type.loc146_27 => constants.%pattern_type.004
// CHECK:STDOUT:   %pattern_type.loc146_33 => constants.%pattern_type.3b9
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc146_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc146_27 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc146_33 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc148 => constants.%complete_type.e35
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.aaa
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.loc148 => constants.%custom_witness.cbf765.1
// CHECK:STDOUT:   %MulWith.facet => constants.%MulWith.facet.ad3
// CHECK:STDOUT:   %MulWith.WithSelf.Op.type => constants.%MulWith.WithSelf.Op.type.39c
// CHECK:STDOUT:   %.loc148_5.4 => constants.%.71b3
// CHECK:STDOUT:   %impl.elem1.loc148_5.2 => constants.%Int16.Op.ecb1a5.10
// CHECK:STDOUT:   %specific_impl_fn.loc148_5.3 => constants.%Int16.Op.ecb1a5.10
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.7bc
// CHECK:STDOUT:   %.loc148_5.5 => constants.%.362
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.f83583.2
// CHECK:STDOUT:   %impl.elem0.loc148_5.2 => constants.%Int32.Op.ecb1a5.1
// CHECK:STDOUT:   %specific_impl_fn.loc148_5.4 => constants.%Int32.Op.ecb1a5.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @DivWith.loc157(constants.%Destroy.facet.700, constants.%Destroy.facet.700, constants.%facet_value.697) {
// CHECK:STDOUT:   %Result.loc154_9.1 => constants.%Destroy.facet.700
// CHECK:STDOUT:   %U.loc155_4.1 => constants.%Destroy.facet.700
// CHECK:STDOUT:   %U.as_type.loc156_36.1 => constants.%Int32
// CHECK:STDOUT:   %DivWith.type.loc156_36.1 => constants.%DivWith.type.0f5
// CHECK:STDOUT:   %facet_type.loc156_20 => constants.%facet_type.81b
// CHECK:STDOUT:   %.Self.loc156_38.1 => constants.%.Self.ec5
// CHECK:STDOUT:   %require_complete.loc156 => constants.%complete_type.a26
// CHECK:STDOUT:   %.Self.as_type.loc156_44.1 => constants.%.Self.as_type.0ba
// CHECK:STDOUT:   %DivWith.assoc_type => constants.%DivWith.assoc_type.205
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.6dd
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.loc156 => constants.%DivWith.lookup_impl_witness.073
// CHECK:STDOUT:   %impl.elem0.loc156_44.1 => constants.%impl.elem0.920
// CHECK:STDOUT:   %Result.as_type.loc156_54.1 => constants.%Int32
// CHECK:STDOUT:   %facet_type.loc156_38 => constants.%facet_type.6bd
// CHECK:STDOUT:   %T.loc156_4.1 => constants.%facet_value.697
// CHECK:STDOUT:   %pattern_type.loc156 => constants.%pattern_type.790c
// CHECK:STDOUT:   %pattern_type.loc157_16 => constants.%pattern_type.3b9
// CHECK:STDOUT:   %T.as_type.loc157_29.1 => constants.%Int16
// CHECK:STDOUT:   %pattern_type.loc157_27 => constants.%pattern_type.004
// CHECK:STDOUT:   %pattern_type.loc157_33 => constants.%pattern_type.3b9
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc157_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc157_27 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc157_33 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc159 => constants.%complete_type.fbd
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.87f
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.loc159 => constants.%custom_witness.ec9d25.1
// CHECK:STDOUT:   %DivWith.facet => constants.%DivWith.facet.e46
// CHECK:STDOUT:   %DivWith.WithSelf.Op.type => constants.%DivWith.WithSelf.Op.type.fa5
// CHECK:STDOUT:   %.loc159_5.4 => constants.%.755
// CHECK:STDOUT:   %impl.elem1.loc159_5.2 => constants.%Int16.Op.ecb1a5.11
// CHECK:STDOUT:   %specific_impl_fn.loc159_5.3 => constants.%Int16.Op.ecb1a5.11
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.7bc
// CHECK:STDOUT:   %.loc159_5.5 => constants.%.362
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.f83583.2
// CHECK:STDOUT:   %impl.elem0.loc159_5.2 => constants.%Int32.Op.ecb1a5.1
// CHECK:STDOUT:   %specific_impl_fn.loc159_5.4 => constants.%Int32.Op.ecb1a5.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ModWith.loc168(constants.%Destroy.facet.700, constants.%Destroy.facet.700, constants.%facet_value.7c5) {
// CHECK:STDOUT:   %Result.loc165_9.1 => constants.%Destroy.facet.700
// CHECK:STDOUT:   %U.loc166_4.1 => constants.%Destroy.facet.700
// CHECK:STDOUT:   %U.as_type.loc167_36.1 => constants.%Int32
// CHECK:STDOUT:   %ModWith.type.loc167_36.1 => constants.%ModWith.type.e59
// CHECK:STDOUT:   %facet_type.loc167_20 => constants.%facet_type.fed
// CHECK:STDOUT:   %.Self.loc167_38.1 => constants.%.Self.6f8
// CHECK:STDOUT:   %require_complete.loc167 => constants.%complete_type.9d2
// CHECK:STDOUT:   %.Self.as_type.loc167_44.1 => constants.%.Self.as_type.0bc
// CHECK:STDOUT:   %ModWith.assoc_type => constants.%ModWith.assoc_type.e6b
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.bf2
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.loc167 => constants.%ModWith.lookup_impl_witness.ba6
// CHECK:STDOUT:   %impl.elem0.loc167_44.1 => constants.%impl.elem0.1ad
// CHECK:STDOUT:   %Result.as_type.loc167_54.1 => constants.%Int32
// CHECK:STDOUT:   %facet_type.loc167_38 => constants.%facet_type.8ee
// CHECK:STDOUT:   %T.loc167_4.1 => constants.%facet_value.7c5
// CHECK:STDOUT:   %pattern_type.loc167 => constants.%pattern_type.239
// CHECK:STDOUT:   %pattern_type.loc168_16 => constants.%pattern_type.3b9
// CHECK:STDOUT:   %T.as_type.loc168_29.1 => constants.%Int16
// CHECK:STDOUT:   %pattern_type.loc168_27 => constants.%pattern_type.004
// CHECK:STDOUT:   %pattern_type.loc168_33 => constants.%pattern_type.3b9
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc168_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc168_27 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc168_33 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc170 => constants.%complete_type.965
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.adf
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.loc170 => constants.%custom_witness.998647.1
// CHECK:STDOUT:   %ModWith.facet => constants.%ModWith.facet.c8c
// CHECK:STDOUT:   %ModWith.WithSelf.Op.type => constants.%ModWith.WithSelf.Op.type.5ca
// CHECK:STDOUT:   %.loc170_5.4 => constants.%.98c
// CHECK:STDOUT:   %impl.elem1.loc170_5.2 => constants.%Int16.Op.ecb1a5.12
// CHECK:STDOUT:   %specific_impl_fn.loc170_5.3 => constants.%Int16.Op.ecb1a5.12
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.7bc
// CHECK:STDOUT:   %.loc170_5.5 => constants.%.362
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.f83583.2
// CHECK:STDOUT:   %impl.elem0.loc170_5.2 => constants.%Int32.Op.ecb1a5.1
// CHECK:STDOUT:   %specific_impl_fn.loc170_5.4 => constants.%Int32.Op.ecb1a5.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AddWith.loc124(constants.%Destroy.facet.700, constants.%Destroy.facet.738, constants.%facet_value.2f2) {
// CHECK:STDOUT:   %Result.loc121_9.1 => constants.%Destroy.facet.700
// CHECK:STDOUT:   %U.loc122_4.1 => constants.%Destroy.facet.738
// CHECK:STDOUT:   %U.as_type.loc123_36.1 => constants.%Int16
// CHECK:STDOUT:   %AddWith.type.loc123_36.1 => constants.%AddWith.type.116
// CHECK:STDOUT:   %facet_type.loc123_20 => constants.%facet_type.aa6
// CHECK:STDOUT:   %.Self.loc123_38.1 => constants.%.Self.cf1
// CHECK:STDOUT:   %require_complete.loc123 => constants.%complete_type.a93
// CHECK:STDOUT:   %.Self.as_type.loc123_44.1 => constants.%.Self.as_type.70cc
// CHECK:STDOUT:   %AddWith.assoc_type => constants.%AddWith.assoc_type.dda
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.c76
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.loc123 => constants.%AddWith.lookup_impl_witness.d17
// CHECK:STDOUT:   %impl.elem0.loc123_44.1 => constants.%impl.elem0.9d9
// CHECK:STDOUT:   %Result.as_type.loc123_54.1 => constants.%Int32
// CHECK:STDOUT:   %facet_type.loc123_38 => constants.%facet_type.2f8
// CHECK:STDOUT:   %T.loc123_4.1 => constants.%facet_value.2f2
// CHECK:STDOUT:   %pattern_type.loc123 => constants.%pattern_type.8af
// CHECK:STDOUT:   %pattern_type.loc124_16 => constants.%pattern_type.3b9
// CHECK:STDOUT:   %T.as_type.loc124_29.1 => constants.%Int32
// CHECK:STDOUT:   %pattern_type.loc124_27 => constants.%pattern_type.3b9
// CHECK:STDOUT:   %pattern_type.loc124_33 => constants.%pattern_type.004
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc124_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc124_27 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc124_33 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc126 => constants.%complete_type.eb1
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.83e
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.loc126 => constants.%custom_witness.6f6
// CHECK:STDOUT:   %AddWith.facet => constants.%AddWith.facet.438
// CHECK:STDOUT:   %AddWith.WithSelf.Op.type => constants.%AddWith.WithSelf.Op.type.f56
// CHECK:STDOUT:   %.loc126_5.4 => constants.%.dcc
// CHECK:STDOUT:   %impl.elem1.loc126_5.2 => constants.%Int32.Op.ecb1a5.2
// CHECK:STDOUT:   %specific_impl_fn.loc126_5.3 => constants.%Int32.Op.ecb1a5.2
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.7bc
// CHECK:STDOUT:   %.loc126_5.5 => constants.%.362
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.f83583.2
// CHECK:STDOUT:   %impl.elem0.loc126_5.2 => constants.%Int32.Op.ecb1a5.1
// CHECK:STDOUT:   %specific_impl_fn.loc126_5.4 => constants.%Int32.Op.ecb1a5.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @SubWith.loc135(constants.%Destroy.facet.700, constants.%Destroy.facet.738, constants.%facet_value.c1a) {
// CHECK:STDOUT:   %Result.loc132_9.1 => constants.%Destroy.facet.700
// CHECK:STDOUT:   %U.loc133_4.1 => constants.%Destroy.facet.738
// CHECK:STDOUT:   %U.as_type.loc134_36.1 => constants.%Int16
// CHECK:STDOUT:   %SubWith.type.loc134_36.1 => constants.%SubWith.type.49a
// CHECK:STDOUT:   %facet_type.loc134_20 => constants.%facet_type.4d6
// CHECK:STDOUT:   %.Self.loc134_38.1 => constants.%.Self.c3b
// CHECK:STDOUT:   %require_complete.loc134 => constants.%complete_type.4fe
// CHECK:STDOUT:   %.Self.as_type.loc134_44.1 => constants.%.Self.as_type.42f
// CHECK:STDOUT:   %SubWith.assoc_type => constants.%SubWith.assoc_type.445
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.c71
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.loc134 => constants.%SubWith.lookup_impl_witness.72f
// CHECK:STDOUT:   %impl.elem0.loc134_44.1 => constants.%impl.elem0.b83
// CHECK:STDOUT:   %Result.as_type.loc134_54.1 => constants.%Int32
// CHECK:STDOUT:   %facet_type.loc134_38 => constants.%facet_type.e67
// CHECK:STDOUT:   %T.loc134_4.1 => constants.%facet_value.c1a
// CHECK:STDOUT:   %pattern_type.loc134 => constants.%pattern_type.f19
// CHECK:STDOUT:   %pattern_type.loc135_16 => constants.%pattern_type.3b9
// CHECK:STDOUT:   %T.as_type.loc135_29.1 => constants.%Int32
// CHECK:STDOUT:   %pattern_type.loc135_27 => constants.%pattern_type.3b9
// CHECK:STDOUT:   %pattern_type.loc135_33 => constants.%pattern_type.004
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc135_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc135_27 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc135_33 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc137 => constants.%complete_type.99b
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.a3f
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.loc137 => constants.%custom_witness.97b
// CHECK:STDOUT:   %SubWith.facet => constants.%SubWith.facet.0d0
// CHECK:STDOUT:   %SubWith.WithSelf.Op.type => constants.%SubWith.WithSelf.Op.type.093
// CHECK:STDOUT:   %.loc137_5.4 => constants.%.ef68
// CHECK:STDOUT:   %impl.elem1.loc137_5.2 => constants.%Int32.Op.ecb1a5.3
// CHECK:STDOUT:   %specific_impl_fn.loc137_5.3 => constants.%Int32.Op.ecb1a5.3
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.7bc
// CHECK:STDOUT:   %.loc137_5.5 => constants.%.362
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.f83583.2
// CHECK:STDOUT:   %impl.elem0.loc137_5.2 => constants.%Int32.Op.ecb1a5.1
// CHECK:STDOUT:   %specific_impl_fn.loc137_5.4 => constants.%Int32.Op.ecb1a5.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @MulWith.loc146(constants.%Destroy.facet.700, constants.%Destroy.facet.738, constants.%facet_value.46e) {
// CHECK:STDOUT:   %Result.loc143_9.1 => constants.%Destroy.facet.700
// CHECK:STDOUT:   %U.loc144_4.1 => constants.%Destroy.facet.738
// CHECK:STDOUT:   %U.as_type.loc145_36.1 => constants.%Int16
// CHECK:STDOUT:   %MulWith.type.loc145_36.1 => constants.%MulWith.type.779
// CHECK:STDOUT:   %facet_type.loc145_20 => constants.%facet_type.624
// CHECK:STDOUT:   %.Self.loc145_38.1 => constants.%.Self.164
// CHECK:STDOUT:   %require_complete.loc145 => constants.%complete_type.ec6
// CHECK:STDOUT:   %.Self.as_type.loc145_44.1 => constants.%.Self.as_type.49c
// CHECK:STDOUT:   %MulWith.assoc_type => constants.%MulWith.assoc_type.3a9
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.97a
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.loc145 => constants.%MulWith.lookup_impl_witness.588
// CHECK:STDOUT:   %impl.elem0.loc145_44.1 => constants.%impl.elem0.66a
// CHECK:STDOUT:   %Result.as_type.loc145_54.1 => constants.%Int32
// CHECK:STDOUT:   %facet_type.loc145_38 => constants.%facet_type.4ee
// CHECK:STDOUT:   %T.loc145_4.1 => constants.%facet_value.46e
// CHECK:STDOUT:   %pattern_type.loc145 => constants.%pattern_type.5fc
// CHECK:STDOUT:   %pattern_type.loc146_16 => constants.%pattern_type.3b9
// CHECK:STDOUT:   %T.as_type.loc146_29.1 => constants.%Int32
// CHECK:STDOUT:   %pattern_type.loc146_27 => constants.%pattern_type.3b9
// CHECK:STDOUT:   %pattern_type.loc146_33 => constants.%pattern_type.004
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc146_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc146_27 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc146_33 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc148 => constants.%complete_type.255
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.a46
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.loc148 => constants.%custom_witness.ac6
// CHECK:STDOUT:   %MulWith.facet => constants.%MulWith.facet.b0d
// CHECK:STDOUT:   %MulWith.WithSelf.Op.type => constants.%MulWith.WithSelf.Op.type.27b
// CHECK:STDOUT:   %.loc148_5.4 => constants.%.5ae
// CHECK:STDOUT:   %impl.elem1.loc148_5.2 => constants.%Int32.Op.ecb1a5.4
// CHECK:STDOUT:   %specific_impl_fn.loc148_5.3 => constants.%Int32.Op.ecb1a5.4
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.7bc
// CHECK:STDOUT:   %.loc148_5.5 => constants.%.362
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.f83583.2
// CHECK:STDOUT:   %impl.elem0.loc148_5.2 => constants.%Int32.Op.ecb1a5.1
// CHECK:STDOUT:   %specific_impl_fn.loc148_5.4 => constants.%Int32.Op.ecb1a5.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @DivWith.loc157(constants.%Destroy.facet.700, constants.%Destroy.facet.738, constants.%facet_value.270) {
// CHECK:STDOUT:   %Result.loc154_9.1 => constants.%Destroy.facet.700
// CHECK:STDOUT:   %U.loc155_4.1 => constants.%Destroy.facet.738
// CHECK:STDOUT:   %U.as_type.loc156_36.1 => constants.%Int16
// CHECK:STDOUT:   %DivWith.type.loc156_36.1 => constants.%DivWith.type.35b
// CHECK:STDOUT:   %facet_type.loc156_20 => constants.%facet_type.b9b
// CHECK:STDOUT:   %.Self.loc156_38.1 => constants.%.Self.7ba
// CHECK:STDOUT:   %require_complete.loc156 => constants.%complete_type.7df
// CHECK:STDOUT:   %.Self.as_type.loc156_44.1 => constants.%.Self.as_type.31a
// CHECK:STDOUT:   %DivWith.assoc_type => constants.%DivWith.assoc_type.6ae
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.1bc
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.loc156 => constants.%DivWith.lookup_impl_witness.836
// CHECK:STDOUT:   %impl.elem0.loc156_44.1 => constants.%impl.elem0.6cb
// CHECK:STDOUT:   %Result.as_type.loc156_54.1 => constants.%Int32
// CHECK:STDOUT:   %facet_type.loc156_38 => constants.%facet_type.f88
// CHECK:STDOUT:   %T.loc156_4.1 => constants.%facet_value.270
// CHECK:STDOUT:   %pattern_type.loc156 => constants.%pattern_type.871
// CHECK:STDOUT:   %pattern_type.loc157_16 => constants.%pattern_type.3b9
// CHECK:STDOUT:   %T.as_type.loc157_29.1 => constants.%Int32
// CHECK:STDOUT:   %pattern_type.loc157_27 => constants.%pattern_type.3b9
// CHECK:STDOUT:   %pattern_type.loc157_33 => constants.%pattern_type.004
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc157_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc157_27 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc157_33 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc159 => constants.%complete_type.c6d
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.5a7
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.loc159 => constants.%custom_witness.387
// CHECK:STDOUT:   %DivWith.facet => constants.%DivWith.facet.0ba
// CHECK:STDOUT:   %DivWith.WithSelf.Op.type => constants.%DivWith.WithSelf.Op.type.798
// CHECK:STDOUT:   %.loc159_5.4 => constants.%.314
// CHECK:STDOUT:   %impl.elem1.loc159_5.2 => constants.%Int32.Op.ecb1a5.5
// CHECK:STDOUT:   %specific_impl_fn.loc159_5.3 => constants.%Int32.Op.ecb1a5.5
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.7bc
// CHECK:STDOUT:   %.loc159_5.5 => constants.%.362
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.f83583.2
// CHECK:STDOUT:   %impl.elem0.loc159_5.2 => constants.%Int32.Op.ecb1a5.1
// CHECK:STDOUT:   %specific_impl_fn.loc159_5.4 => constants.%Int32.Op.ecb1a5.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ModWith.loc168(constants.%Destroy.facet.700, constants.%Destroy.facet.738, constants.%facet_value.64f) {
// CHECK:STDOUT:   %Result.loc165_9.1 => constants.%Destroy.facet.700
// CHECK:STDOUT:   %U.loc166_4.1 => constants.%Destroy.facet.738
// CHECK:STDOUT:   %U.as_type.loc167_36.1 => constants.%Int16
// CHECK:STDOUT:   %ModWith.type.loc167_36.1 => constants.%ModWith.type.3ef
// CHECK:STDOUT:   %facet_type.loc167_20 => constants.%facet_type.ca3
// CHECK:STDOUT:   %.Self.loc167_38.1 => constants.%.Self.cfe
// CHECK:STDOUT:   %require_complete.loc167 => constants.%complete_type.1cf
// CHECK:STDOUT:   %.Self.as_type.loc167_44.1 => constants.%.Self.as_type.da7
// CHECK:STDOUT:   %ModWith.assoc_type => constants.%ModWith.assoc_type.744
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.727
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.loc167 => constants.%ModWith.lookup_impl_witness.305
// CHECK:STDOUT:   %impl.elem0.loc167_44.1 => constants.%impl.elem0.2b9
// CHECK:STDOUT:   %Result.as_type.loc167_54.1 => constants.%Int32
// CHECK:STDOUT:   %facet_type.loc167_38 => constants.%facet_type.994
// CHECK:STDOUT:   %T.loc167_4.1 => constants.%facet_value.64f
// CHECK:STDOUT:   %pattern_type.loc167 => constants.%pattern_type.269
// CHECK:STDOUT:   %pattern_type.loc168_16 => constants.%pattern_type.3b9
// CHECK:STDOUT:   %T.as_type.loc168_29.1 => constants.%Int32
// CHECK:STDOUT:   %pattern_type.loc168_27 => constants.%pattern_type.3b9
// CHECK:STDOUT:   %pattern_type.loc168_33 => constants.%pattern_type.004
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc168_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc168_27 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc168_33 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc170 => constants.%complete_type.55f
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.ba4
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.loc170 => constants.%custom_witness.5cd
// CHECK:STDOUT:   %ModWith.facet => constants.%ModWith.facet.f7c
// CHECK:STDOUT:   %ModWith.WithSelf.Op.type => constants.%ModWith.WithSelf.Op.type.e0b
// CHECK:STDOUT:   %.loc170_5.4 => constants.%.ed5
// CHECK:STDOUT:   %impl.elem1.loc170_5.2 => constants.%Int32.Op.ecb1a5.6
// CHECK:STDOUT:   %specific_impl_fn.loc170_5.3 => constants.%Int32.Op.ecb1a5.6
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.7bc
// CHECK:STDOUT:   %.loc170_5.5 => constants.%.362
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.f83583.2
// CHECK:STDOUT:   %impl.elem0.loc170_5.2 => constants.%Int32.Op.ecb1a5.1
// CHECK:STDOUT:   %specific_impl_fn.loc170_5.4 => constants.%Int32.Op.ecb1a5.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AddWith.loc124(constants.%Destroy.facet.700, constants.%Destroy.facet.700, constants.%facet_value.f1e) {
// CHECK:STDOUT:   %Result.loc121_9.1 => constants.%Destroy.facet.700
// CHECK:STDOUT:   %U.loc122_4.1 => constants.%Destroy.facet.700
// CHECK:STDOUT:   %U.as_type.loc123_36.1 => constants.%Int32
// CHECK:STDOUT:   %AddWith.type.loc123_36.1 => constants.%AddWith.type.c13
// CHECK:STDOUT:   %facet_type.loc123_20 => constants.%facet_type.f56
// CHECK:STDOUT:   %.Self.loc123_38.1 => constants.%.Self.977
// CHECK:STDOUT:   %require_complete.loc123 => constants.%complete_type.39f
// CHECK:STDOUT:   %.Self.as_type.loc123_44.1 => constants.%.Self.as_type.70c2
// CHECK:STDOUT:   %AddWith.assoc_type => constants.%AddWith.assoc_type.f6f
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.41e
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.loc123 => constants.%AddWith.lookup_impl_witness.883
// CHECK:STDOUT:   %impl.elem0.loc123_44.1 => constants.%impl.elem0.2e9
// CHECK:STDOUT:   %Result.as_type.loc123_54.1 => constants.%Int32
// CHECK:STDOUT:   %facet_type.loc123_38 => constants.%facet_type.36e
// CHECK:STDOUT:   %T.loc123_4.1 => constants.%facet_value.f1e
// CHECK:STDOUT:   %pattern_type.loc123 => constants.%pattern_type.204
// CHECK:STDOUT:   %pattern_type.loc124_16 => constants.%pattern_type.3b9
// CHECK:STDOUT:   %T.as_type.loc124_29.1 => constants.%Int32
// CHECK:STDOUT:   %pattern_type.loc124_27 => constants.%pattern_type.3b9
// CHECK:STDOUT:   %pattern_type.loc124_33 => constants.%pattern_type.3b9
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc124_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc124_27 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc124_33 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc126 => constants.%complete_type.f05
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.519
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.loc126 => constants.%custom_witness.27d802.2
// CHECK:STDOUT:   %AddWith.facet => constants.%AddWith.facet.524
// CHECK:STDOUT:   %AddWith.WithSelf.Op.type => constants.%AddWith.WithSelf.Op.type.56f
// CHECK:STDOUT:   %.loc126_5.4 => constants.%.57b
// CHECK:STDOUT:   %impl.elem1.loc126_5.2 => constants.%Int32.Op.ecb1a5.7
// CHECK:STDOUT:   %specific_impl_fn.loc126_5.3 => constants.%Int32.Op.ecb1a5.7
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.7bc
// CHECK:STDOUT:   %.loc126_5.5 => constants.%.362
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.f83583.2
// CHECK:STDOUT:   %impl.elem0.loc126_5.2 => constants.%Int32.Op.ecb1a5.1
// CHECK:STDOUT:   %specific_impl_fn.loc126_5.4 => constants.%Int32.Op.ecb1a5.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @SubWith.loc135(constants.%Destroy.facet.700, constants.%Destroy.facet.700, constants.%facet_value.b8e) {
// CHECK:STDOUT:   %Result.loc132_9.1 => constants.%Destroy.facet.700
// CHECK:STDOUT:   %U.loc133_4.1 => constants.%Destroy.facet.700
// CHECK:STDOUT:   %U.as_type.loc134_36.1 => constants.%Int32
// CHECK:STDOUT:   %SubWith.type.loc134_36.1 => constants.%SubWith.type.554
// CHECK:STDOUT:   %facet_type.loc134_20 => constants.%facet_type.eb0
// CHECK:STDOUT:   %.Self.loc134_38.1 => constants.%.Self.604
// CHECK:STDOUT:   %require_complete.loc134 => constants.%complete_type.872
// CHECK:STDOUT:   %.Self.as_type.loc134_44.1 => constants.%.Self.as_type.d34
// CHECK:STDOUT:   %SubWith.assoc_type => constants.%SubWith.assoc_type.6db
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.053
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.loc134 => constants.%SubWith.lookup_impl_witness.8be
// CHECK:STDOUT:   %impl.elem0.loc134_44.1 => constants.%impl.elem0.7af
// CHECK:STDOUT:   %Result.as_type.loc134_54.1 => constants.%Int32
// CHECK:STDOUT:   %facet_type.loc134_38 => constants.%facet_type.dfd
// CHECK:STDOUT:   %T.loc134_4.1 => constants.%facet_value.b8e
// CHECK:STDOUT:   %pattern_type.loc134 => constants.%pattern_type.b5c
// CHECK:STDOUT:   %pattern_type.loc135_16 => constants.%pattern_type.3b9
// CHECK:STDOUT:   %T.as_type.loc135_29.1 => constants.%Int32
// CHECK:STDOUT:   %pattern_type.loc135_27 => constants.%pattern_type.3b9
// CHECK:STDOUT:   %pattern_type.loc135_33 => constants.%pattern_type.3b9
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc135_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc135_27 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc135_33 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc137 => constants.%complete_type.60a
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.3e8
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.loc137 => constants.%custom_witness.f4a1b8.2
// CHECK:STDOUT:   %SubWith.facet => constants.%SubWith.facet.304
// CHECK:STDOUT:   %SubWith.WithSelf.Op.type => constants.%SubWith.WithSelf.Op.type.dc6
// CHECK:STDOUT:   %.loc137_5.4 => constants.%.f43
// CHECK:STDOUT:   %impl.elem1.loc137_5.2 => constants.%Int32.Op.ecb1a5.8
// CHECK:STDOUT:   %specific_impl_fn.loc137_5.3 => constants.%Int32.Op.ecb1a5.8
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.7bc
// CHECK:STDOUT:   %.loc137_5.5 => constants.%.362
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.f83583.2
// CHECK:STDOUT:   %impl.elem0.loc137_5.2 => constants.%Int32.Op.ecb1a5.1
// CHECK:STDOUT:   %specific_impl_fn.loc137_5.4 => constants.%Int32.Op.ecb1a5.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @MulWith.loc146(constants.%Destroy.facet.700, constants.%Destroy.facet.700, constants.%facet_value.3cc) {
// CHECK:STDOUT:   %Result.loc143_9.1 => constants.%Destroy.facet.700
// CHECK:STDOUT:   %U.loc144_4.1 => constants.%Destroy.facet.700
// CHECK:STDOUT:   %U.as_type.loc145_36.1 => constants.%Int32
// CHECK:STDOUT:   %MulWith.type.loc145_36.1 => constants.%MulWith.type.d6e
// CHECK:STDOUT:   %facet_type.loc145_20 => constants.%facet_type.87a
// CHECK:STDOUT:   %.Self.loc145_38.1 => constants.%.Self.485
// CHECK:STDOUT:   %require_complete.loc145 => constants.%complete_type.e5c
// CHECK:STDOUT:   %.Self.as_type.loc145_44.1 => constants.%.Self.as_type.963
// CHECK:STDOUT:   %MulWith.assoc_type => constants.%MulWith.assoc_type.e4f
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.879
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.loc145 => constants.%MulWith.lookup_impl_witness.99b
// CHECK:STDOUT:   %impl.elem0.loc145_44.1 => constants.%impl.elem0.a71
// CHECK:STDOUT:   %Result.as_type.loc145_54.1 => constants.%Int32
// CHECK:STDOUT:   %facet_type.loc145_38 => constants.%facet_type.4d5
// CHECK:STDOUT:   %T.loc145_4.1 => constants.%facet_value.3cc
// CHECK:STDOUT:   %pattern_type.loc145 => constants.%pattern_type.459
// CHECK:STDOUT:   %pattern_type.loc146_16 => constants.%pattern_type.3b9
// CHECK:STDOUT:   %T.as_type.loc146_29.1 => constants.%Int32
// CHECK:STDOUT:   %pattern_type.loc146_27 => constants.%pattern_type.3b9
// CHECK:STDOUT:   %pattern_type.loc146_33 => constants.%pattern_type.3b9
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc146_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc146_27 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc146_33 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc148 => constants.%complete_type.e35
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.aaa
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.loc148 => constants.%custom_witness.cbf765.2
// CHECK:STDOUT:   %MulWith.facet => constants.%MulWith.facet.4d2
// CHECK:STDOUT:   %MulWith.WithSelf.Op.type => constants.%MulWith.WithSelf.Op.type.6e0
// CHECK:STDOUT:   %.loc148_5.4 => constants.%.c8a
// CHECK:STDOUT:   %impl.elem1.loc148_5.2 => constants.%Int32.Op.ecb1a5.9
// CHECK:STDOUT:   %specific_impl_fn.loc148_5.3 => constants.%Int32.Op.ecb1a5.9
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.7bc
// CHECK:STDOUT:   %.loc148_5.5 => constants.%.362
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.f83583.2
// CHECK:STDOUT:   %impl.elem0.loc148_5.2 => constants.%Int32.Op.ecb1a5.1
// CHECK:STDOUT:   %specific_impl_fn.loc148_5.4 => constants.%Int32.Op.ecb1a5.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @DivWith.loc157(constants.%Destroy.facet.700, constants.%Destroy.facet.700, constants.%facet_value.64a) {
// CHECK:STDOUT:   %Result.loc154_9.1 => constants.%Destroy.facet.700
// CHECK:STDOUT:   %U.loc155_4.1 => constants.%Destroy.facet.700
// CHECK:STDOUT:   %U.as_type.loc156_36.1 => constants.%Int32
// CHECK:STDOUT:   %DivWith.type.loc156_36.1 => constants.%DivWith.type.0f5
// CHECK:STDOUT:   %facet_type.loc156_20 => constants.%facet_type.81b
// CHECK:STDOUT:   %.Self.loc156_38.1 => constants.%.Self.ec5
// CHECK:STDOUT:   %require_complete.loc156 => constants.%complete_type.a26
// CHECK:STDOUT:   %.Self.as_type.loc156_44.1 => constants.%.Self.as_type.0ba
// CHECK:STDOUT:   %DivWith.assoc_type => constants.%DivWith.assoc_type.205
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.6dd
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.loc156 => constants.%DivWith.lookup_impl_witness.073
// CHECK:STDOUT:   %impl.elem0.loc156_44.1 => constants.%impl.elem0.920
// CHECK:STDOUT:   %Result.as_type.loc156_54.1 => constants.%Int32
// CHECK:STDOUT:   %facet_type.loc156_38 => constants.%facet_type.6bd
// CHECK:STDOUT:   %T.loc156_4.1 => constants.%facet_value.64a
// CHECK:STDOUT:   %pattern_type.loc156 => constants.%pattern_type.790c
// CHECK:STDOUT:   %pattern_type.loc157_16 => constants.%pattern_type.3b9
// CHECK:STDOUT:   %T.as_type.loc157_29.1 => constants.%Int32
// CHECK:STDOUT:   %pattern_type.loc157_27 => constants.%pattern_type.3b9
// CHECK:STDOUT:   %pattern_type.loc157_33 => constants.%pattern_type.3b9
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc157_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc157_27 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc157_33 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc159 => constants.%complete_type.fbd
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.87f
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.loc159 => constants.%custom_witness.ec9d25.2
// CHECK:STDOUT:   %DivWith.facet => constants.%DivWith.facet.47e
// CHECK:STDOUT:   %DivWith.WithSelf.Op.type => constants.%DivWith.WithSelf.Op.type.489
// CHECK:STDOUT:   %.loc159_5.4 => constants.%.9129
// CHECK:STDOUT:   %impl.elem1.loc159_5.2 => constants.%Int32.Op.ecb1a5.10
// CHECK:STDOUT:   %specific_impl_fn.loc159_5.3 => constants.%Int32.Op.ecb1a5.10
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.7bc
// CHECK:STDOUT:   %.loc159_5.5 => constants.%.362
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.f83583.2
// CHECK:STDOUT:   %impl.elem0.loc159_5.2 => constants.%Int32.Op.ecb1a5.1
// CHECK:STDOUT:   %specific_impl_fn.loc159_5.4 => constants.%Int32.Op.ecb1a5.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ModWith.loc168(constants.%Destroy.facet.700, constants.%Destroy.facet.700, constants.%facet_value.c6d) {
// CHECK:STDOUT:   %Result.loc165_9.1 => constants.%Destroy.facet.700
// CHECK:STDOUT:   %U.loc166_4.1 => constants.%Destroy.facet.700
// CHECK:STDOUT:   %U.as_type.loc167_36.1 => constants.%Int32
// CHECK:STDOUT:   %ModWith.type.loc167_36.1 => constants.%ModWith.type.e59
// CHECK:STDOUT:   %facet_type.loc167_20 => constants.%facet_type.fed
// CHECK:STDOUT:   %.Self.loc167_38.1 => constants.%.Self.6f8
// CHECK:STDOUT:   %require_complete.loc167 => constants.%complete_type.9d2
// CHECK:STDOUT:   %.Self.as_type.loc167_44.1 => constants.%.Self.as_type.0bc
// CHECK:STDOUT:   %ModWith.assoc_type => constants.%ModWith.assoc_type.e6b
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.bf2
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.loc167 => constants.%ModWith.lookup_impl_witness.ba6
// CHECK:STDOUT:   %impl.elem0.loc167_44.1 => constants.%impl.elem0.1ad
// CHECK:STDOUT:   %Result.as_type.loc167_54.1 => constants.%Int32
// CHECK:STDOUT:   %facet_type.loc167_38 => constants.%facet_type.8ee
// CHECK:STDOUT:   %T.loc167_4.1 => constants.%facet_value.c6d
// CHECK:STDOUT:   %pattern_type.loc167 => constants.%pattern_type.239
// CHECK:STDOUT:   %pattern_type.loc168_16 => constants.%pattern_type.3b9
// CHECK:STDOUT:   %T.as_type.loc168_29.1 => constants.%Int32
// CHECK:STDOUT:   %pattern_type.loc168_27 => constants.%pattern_type.3b9
// CHECK:STDOUT:   %pattern_type.loc168_33 => constants.%pattern_type.3b9
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc168_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc168_27 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc168_33 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc170 => constants.%complete_type.965
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.adf
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.loc170 => constants.%custom_witness.998647.2
// CHECK:STDOUT:   %ModWith.facet => constants.%ModWith.facet.217
// CHECK:STDOUT:   %ModWith.WithSelf.Op.type => constants.%ModWith.WithSelf.Op.type.c61
// CHECK:STDOUT:   %.loc170_5.4 => constants.%.512
// CHECK:STDOUT:   %impl.elem1.loc170_5.2 => constants.%Int32.Op.ecb1a5.11
// CHECK:STDOUT:   %specific_impl_fn.loc170_5.3 => constants.%Int32.Op.ecb1a5.11
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.7bc
// CHECK:STDOUT:   %.loc170_5.5 => constants.%.362
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.f83583.2
// CHECK:STDOUT:   %impl.elem0.loc170_5.2 => constants.%Int32.Op.ecb1a5.1
// CHECK:STDOUT:   %specific_impl_fn.loc170_5.4 => constants.%Int32.Op.ecb1a5.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AddWith.loc124(constants.%Destroy.facet.a4a, constants.%Destroy.facet.a4a, constants.%facet_value.9f2) {
// CHECK:STDOUT:   %Result.loc121_9.1 => constants.%Destroy.facet.a4a
// CHECK:STDOUT:   %U.loc122_4.1 => constants.%Destroy.facet.a4a
// CHECK:STDOUT:   %U.as_type.loc123_36.1 => constants.%Int64
// CHECK:STDOUT:   %AddWith.type.loc123_36.1 => constants.%AddWith.type.b5c
// CHECK:STDOUT:   %facet_type.loc123_20 => constants.%facet_type.b51
// CHECK:STDOUT:   %.Self.loc123_38.1 => constants.%.Self.b56
// CHECK:STDOUT:   %require_complete.loc123 => constants.%complete_type.245
// CHECK:STDOUT:   %.Self.as_type.loc123_44.1 => constants.%.Self.as_type.f5c
// CHECK:STDOUT:   %AddWith.assoc_type => constants.%AddWith.assoc_type.ea8
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.c63
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.loc123 => constants.%AddWith.lookup_impl_witness.71c
// CHECK:STDOUT:   %impl.elem0.loc123_44.1 => constants.%impl.elem0.cdf
// CHECK:STDOUT:   %Result.as_type.loc123_54.1 => constants.%Int64
// CHECK:STDOUT:   %facet_type.loc123_38 => constants.%facet_type.b34
// CHECK:STDOUT:   %T.loc123_4.1 => constants.%facet_value.9f2
// CHECK:STDOUT:   %pattern_type.loc123 => constants.%pattern_type.1c9
// CHECK:STDOUT:   %pattern_type.loc124_16 => constants.%pattern_type.41c
// CHECK:STDOUT:   %T.as_type.loc124_29.1 => constants.%Int32
// CHECK:STDOUT:   %pattern_type.loc124_27 => constants.%pattern_type.3b9
// CHECK:STDOUT:   %pattern_type.loc124_33 => constants.%pattern_type.41c
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc124_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc124_27 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc124_33 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc126 => constants.%complete_type.905
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.69c
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.loc126 => constants.%custom_witness.e14
// CHECK:STDOUT:   %AddWith.facet => constants.%AddWith.facet.d6a
// CHECK:STDOUT:   %AddWith.WithSelf.Op.type => constants.%AddWith.WithSelf.Op.type.815
// CHECK:STDOUT:   %.loc126_5.4 => constants.%.c1e
// CHECK:STDOUT:   %impl.elem1.loc126_5.2 => constants.%Op.b9c47d.6
// CHECK:STDOUT:   %specific_impl_fn.loc126_5.3 => constants.%Op.b9c47d.6
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.2a2
// CHECK:STDOUT:   %.loc126_5.5 => constants.%.08b
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.f83583.3
// CHECK:STDOUT:   %impl.elem0.loc126_5.2 => constants.%Int64.Op.ecb1a5.1
// CHECK:STDOUT:   %specific_impl_fn.loc126_5.4 => constants.%Int64.Op.ecb1a5.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @SubWith.loc135(constants.%Destroy.facet.a4a, constants.%Destroy.facet.a4a, constants.%facet_value.b33) {
// CHECK:STDOUT:   %Result.loc132_9.1 => constants.%Destroy.facet.a4a
// CHECK:STDOUT:   %U.loc133_4.1 => constants.%Destroy.facet.a4a
// CHECK:STDOUT:   %U.as_type.loc134_36.1 => constants.%Int64
// CHECK:STDOUT:   %SubWith.type.loc134_36.1 => constants.%SubWith.type.4c4
// CHECK:STDOUT:   %facet_type.loc134_20 => constants.%facet_type.8dc
// CHECK:STDOUT:   %.Self.loc134_38.1 => constants.%.Self.504
// CHECK:STDOUT:   %require_complete.loc134 => constants.%complete_type.8ff
// CHECK:STDOUT:   %.Self.as_type.loc134_44.1 => constants.%.Self.as_type.e9c
// CHECK:STDOUT:   %SubWith.assoc_type => constants.%SubWith.assoc_type.25c
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.2a2
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.loc134 => constants.%SubWith.lookup_impl_witness.960
// CHECK:STDOUT:   %impl.elem0.loc134_44.1 => constants.%impl.elem0.8ab
// CHECK:STDOUT:   %Result.as_type.loc134_54.1 => constants.%Int64
// CHECK:STDOUT:   %facet_type.loc134_38 => constants.%facet_type.b2b
// CHECK:STDOUT:   %T.loc134_4.1 => constants.%facet_value.b33
// CHECK:STDOUT:   %pattern_type.loc134 => constants.%pattern_type.e40
// CHECK:STDOUT:   %pattern_type.loc135_16 => constants.%pattern_type.41c
// CHECK:STDOUT:   %T.as_type.loc135_29.1 => constants.%Int32
// CHECK:STDOUT:   %pattern_type.loc135_27 => constants.%pattern_type.3b9
// CHECK:STDOUT:   %pattern_type.loc135_33 => constants.%pattern_type.41c
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc135_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc135_27 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc135_33 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc137 => constants.%complete_type.7e8
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.4e8
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.loc137 => constants.%custom_witness.d0e
// CHECK:STDOUT:   %SubWith.facet => constants.%SubWith.facet.bd2
// CHECK:STDOUT:   %SubWith.WithSelf.Op.type => constants.%SubWith.WithSelf.Op.type.6af
// CHECK:STDOUT:   %.loc137_5.4 => constants.%.b8f
// CHECK:STDOUT:   %impl.elem1.loc137_5.2 => constants.%Op.b9c47d.7
// CHECK:STDOUT:   %specific_impl_fn.loc137_5.3 => constants.%Op.b9c47d.7
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.2a2
// CHECK:STDOUT:   %.loc137_5.5 => constants.%.08b
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.f83583.3
// CHECK:STDOUT:   %impl.elem0.loc137_5.2 => constants.%Int64.Op.ecb1a5.1
// CHECK:STDOUT:   %specific_impl_fn.loc137_5.4 => constants.%Int64.Op.ecb1a5.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @MulWith.loc146(constants.%Destroy.facet.a4a, constants.%Destroy.facet.a4a, constants.%facet_value.1e7) {
// CHECK:STDOUT:   %Result.loc143_9.1 => constants.%Destroy.facet.a4a
// CHECK:STDOUT:   %U.loc144_4.1 => constants.%Destroy.facet.a4a
// CHECK:STDOUT:   %U.as_type.loc145_36.1 => constants.%Int64
// CHECK:STDOUT:   %MulWith.type.loc145_36.1 => constants.%MulWith.type.6fc
// CHECK:STDOUT:   %facet_type.loc145_20 => constants.%facet_type.4f0
// CHECK:STDOUT:   %.Self.loc145_38.1 => constants.%.Self.bc5
// CHECK:STDOUT:   %require_complete.loc145 => constants.%complete_type.609
// CHECK:STDOUT:   %.Self.as_type.loc145_44.1 => constants.%.Self.as_type.ec4
// CHECK:STDOUT:   %MulWith.assoc_type => constants.%MulWith.assoc_type.23b
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.5d0
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.loc145 => constants.%MulWith.lookup_impl_witness.3f1
// CHECK:STDOUT:   %impl.elem0.loc145_44.1 => constants.%impl.elem0.a20
// CHECK:STDOUT:   %Result.as_type.loc145_54.1 => constants.%Int64
// CHECK:STDOUT:   %facet_type.loc145_38 => constants.%facet_type.b1b
// CHECK:STDOUT:   %T.loc145_4.1 => constants.%facet_value.1e7
// CHECK:STDOUT:   %pattern_type.loc145 => constants.%pattern_type.c0b
// CHECK:STDOUT:   %pattern_type.loc146_16 => constants.%pattern_type.41c
// CHECK:STDOUT:   %T.as_type.loc146_29.1 => constants.%Int32
// CHECK:STDOUT:   %pattern_type.loc146_27 => constants.%pattern_type.3b9
// CHECK:STDOUT:   %pattern_type.loc146_33 => constants.%pattern_type.41c
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc146_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc146_27 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc146_33 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc148 => constants.%complete_type.3a2
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.78a
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.loc148 => constants.%custom_witness.784
// CHECK:STDOUT:   %MulWith.facet => constants.%MulWith.facet.806
// CHECK:STDOUT:   %MulWith.WithSelf.Op.type => constants.%MulWith.WithSelf.Op.type.96e
// CHECK:STDOUT:   %.loc148_5.4 => constants.%.a56
// CHECK:STDOUT:   %impl.elem1.loc148_5.2 => constants.%Op.b9c47d.8
// CHECK:STDOUT:   %specific_impl_fn.loc148_5.3 => constants.%Op.b9c47d.8
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.2a2
// CHECK:STDOUT:   %.loc148_5.5 => constants.%.08b
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.f83583.3
// CHECK:STDOUT:   %impl.elem0.loc148_5.2 => constants.%Int64.Op.ecb1a5.1
// CHECK:STDOUT:   %specific_impl_fn.loc148_5.4 => constants.%Int64.Op.ecb1a5.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @DivWith.loc157(constants.%Destroy.facet.a4a, constants.%Destroy.facet.a4a, constants.%facet_value.23f) {
// CHECK:STDOUT:   %Result.loc154_9.1 => constants.%Destroy.facet.a4a
// CHECK:STDOUT:   %U.loc155_4.1 => constants.%Destroy.facet.a4a
// CHECK:STDOUT:   %U.as_type.loc156_36.1 => constants.%Int64
// CHECK:STDOUT:   %DivWith.type.loc156_36.1 => constants.%DivWith.type.46b
// CHECK:STDOUT:   %facet_type.loc156_20 => constants.%facet_type.d54
// CHECK:STDOUT:   %.Self.loc156_38.1 => constants.%.Self.987
// CHECK:STDOUT:   %require_complete.loc156 => constants.%complete_type.d68
// CHECK:STDOUT:   %.Self.as_type.loc156_44.1 => constants.%.Self.as_type.fad
// CHECK:STDOUT:   %DivWith.assoc_type => constants.%DivWith.assoc_type.d66
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.d4a
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.loc156 => constants.%DivWith.lookup_impl_witness.d5e
// CHECK:STDOUT:   %impl.elem0.loc156_44.1 => constants.%impl.elem0.cba
// CHECK:STDOUT:   %Result.as_type.loc156_54.1 => constants.%Int64
// CHECK:STDOUT:   %facet_type.loc156_38 => constants.%facet_type.0c6
// CHECK:STDOUT:   %T.loc156_4.1 => constants.%facet_value.23f
// CHECK:STDOUT:   %pattern_type.loc156 => constants.%pattern_type.4e3
// CHECK:STDOUT:   %pattern_type.loc157_16 => constants.%pattern_type.41c
// CHECK:STDOUT:   %T.as_type.loc157_29.1 => constants.%Int32
// CHECK:STDOUT:   %pattern_type.loc157_27 => constants.%pattern_type.3b9
// CHECK:STDOUT:   %pattern_type.loc157_33 => constants.%pattern_type.41c
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc157_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc157_27 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc157_33 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc159 => constants.%complete_type.504
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.e3e
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.loc159 => constants.%custom_witness.a1d
// CHECK:STDOUT:   %DivWith.facet => constants.%DivWith.facet.3c3
// CHECK:STDOUT:   %DivWith.WithSelf.Op.type => constants.%DivWith.WithSelf.Op.type.bde
// CHECK:STDOUT:   %.loc159_5.4 => constants.%.27f
// CHECK:STDOUT:   %impl.elem1.loc159_5.2 => constants.%Op.b9c47d.9
// CHECK:STDOUT:   %specific_impl_fn.loc159_5.3 => constants.%Op.b9c47d.9
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.2a2
// CHECK:STDOUT:   %.loc159_5.5 => constants.%.08b
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.f83583.3
// CHECK:STDOUT:   %impl.elem0.loc159_5.2 => constants.%Int64.Op.ecb1a5.1
// CHECK:STDOUT:   %specific_impl_fn.loc159_5.4 => constants.%Int64.Op.ecb1a5.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ModWith.loc168(constants.%Destroy.facet.a4a, constants.%Destroy.facet.a4a, constants.%facet_value.b6d) {
// CHECK:STDOUT:   %Result.loc165_9.1 => constants.%Destroy.facet.a4a
// CHECK:STDOUT:   %U.loc166_4.1 => constants.%Destroy.facet.a4a
// CHECK:STDOUT:   %U.as_type.loc167_36.1 => constants.%Int64
// CHECK:STDOUT:   %ModWith.type.loc167_36.1 => constants.%ModWith.type.610
// CHECK:STDOUT:   %facet_type.loc167_20 => constants.%facet_type.ad6
// CHECK:STDOUT:   %.Self.loc167_38.1 => constants.%.Self.eba
// CHECK:STDOUT:   %require_complete.loc167 => constants.%complete_type.1c0
// CHECK:STDOUT:   %.Self.as_type.loc167_44.1 => constants.%.Self.as_type.0bf
// CHECK:STDOUT:   %ModWith.assoc_type => constants.%ModWith.assoc_type.08d
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.558
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.loc167 => constants.%ModWith.lookup_impl_witness.d93
// CHECK:STDOUT:   %impl.elem0.loc167_44.1 => constants.%impl.elem0.036
// CHECK:STDOUT:   %Result.as_type.loc167_54.1 => constants.%Int64
// CHECK:STDOUT:   %facet_type.loc167_38 => constants.%facet_type.892
// CHECK:STDOUT:   %T.loc167_4.1 => constants.%facet_value.b6d
// CHECK:STDOUT:   %pattern_type.loc167 => constants.%pattern_type.a7b8
// CHECK:STDOUT:   %pattern_type.loc168_16 => constants.%pattern_type.41c
// CHECK:STDOUT:   %T.as_type.loc168_29.1 => constants.%Int32
// CHECK:STDOUT:   %pattern_type.loc168_27 => constants.%pattern_type.3b9
// CHECK:STDOUT:   %pattern_type.loc168_33 => constants.%pattern_type.41c
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc168_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc168_27 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc168_33 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc170 => constants.%complete_type.52f
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.ad0
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.loc170 => constants.%custom_witness.1f5
// CHECK:STDOUT:   %ModWith.facet => constants.%ModWith.facet.e04
// CHECK:STDOUT:   %ModWith.WithSelf.Op.type => constants.%ModWith.WithSelf.Op.type.abf
// CHECK:STDOUT:   %.loc170_5.4 => constants.%.22f
// CHECK:STDOUT:   %impl.elem1.loc170_5.2 => constants.%Op.b9c47d.10
// CHECK:STDOUT:   %specific_impl_fn.loc170_5.3 => constants.%Op.b9c47d.10
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.2a2
// CHECK:STDOUT:   %.loc170_5.5 => constants.%.08b
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.f83583.3
// CHECK:STDOUT:   %impl.elem0.loc170_5.2 => constants.%Int64.Op.ecb1a5.1
// CHECK:STDOUT:   %specific_impl_fn.loc170_5.4 => constants.%Int64.Op.ecb1a5.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AddWith.loc124(constants.%Destroy.facet.a4a, constants.%Destroy.facet.700, constants.%facet_value.ac6) {
// CHECK:STDOUT:   %Result.loc121_9.1 => constants.%Destroy.facet.a4a
// CHECK:STDOUT:   %U.loc122_4.1 => constants.%Destroy.facet.700
// CHECK:STDOUT:   %U.as_type.loc123_36.1 => constants.%Int32
// CHECK:STDOUT:   %AddWith.type.loc123_36.1 => constants.%AddWith.type.c13
// CHECK:STDOUT:   %facet_type.loc123_20 => constants.%facet_type.f56
// CHECK:STDOUT:   %.Self.loc123_38.1 => constants.%.Self.977
// CHECK:STDOUT:   %require_complete.loc123 => constants.%complete_type.39f
// CHECK:STDOUT:   %.Self.as_type.loc123_44.1 => constants.%.Self.as_type.70c2
// CHECK:STDOUT:   %AddWith.assoc_type => constants.%AddWith.assoc_type.f6f
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.41e
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.loc123 => constants.%AddWith.lookup_impl_witness.883
// CHECK:STDOUT:   %impl.elem0.loc123_44.1 => constants.%impl.elem0.2e9
// CHECK:STDOUT:   %Result.as_type.loc123_54.1 => constants.%Int64
// CHECK:STDOUT:   %facet_type.loc123_38 => constants.%facet_type.167
// CHECK:STDOUT:   %T.loc123_4.1 => constants.%facet_value.ac6
// CHECK:STDOUT:   %pattern_type.loc123 => constants.%pattern_type.109
// CHECK:STDOUT:   %pattern_type.loc124_16 => constants.%pattern_type.41c
// CHECK:STDOUT:   %T.as_type.loc124_29.1 => constants.%Int64
// CHECK:STDOUT:   %pattern_type.loc124_27 => constants.%pattern_type.41c
// CHECK:STDOUT:   %pattern_type.loc124_33 => constants.%pattern_type.3b9
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc124_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc124_27 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc124_33 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc126 => constants.%complete_type.f05
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.519
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.loc126 => constants.%custom_witness.8f1
// CHECK:STDOUT:   %AddWith.facet => constants.%AddWith.facet.fd0
// CHECK:STDOUT:   %AddWith.WithSelf.Op.type => constants.%AddWith.WithSelf.Op.type.3e0
// CHECK:STDOUT:   %.loc126_5.4 => constants.%.800
// CHECK:STDOUT:   %impl.elem1.loc126_5.2 => constants.%Op.b9c47d.11
// CHECK:STDOUT:   %specific_impl_fn.loc126_5.3 => constants.%Op.b9c47d.11
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.2a2
// CHECK:STDOUT:   %.loc126_5.5 => constants.%.08b
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.f83583.3
// CHECK:STDOUT:   %impl.elem0.loc126_5.2 => constants.%Int64.Op.ecb1a5.1
// CHECK:STDOUT:   %specific_impl_fn.loc126_5.4 => constants.%Int64.Op.ecb1a5.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @SubWith.loc135(constants.%Destroy.facet.a4a, constants.%Destroy.facet.700, constants.%facet_value.fce) {
// CHECK:STDOUT:   %Result.loc132_9.1 => constants.%Destroy.facet.a4a
// CHECK:STDOUT:   %U.loc133_4.1 => constants.%Destroy.facet.700
// CHECK:STDOUT:   %U.as_type.loc134_36.1 => constants.%Int32
// CHECK:STDOUT:   %SubWith.type.loc134_36.1 => constants.%SubWith.type.554
// CHECK:STDOUT:   %facet_type.loc134_20 => constants.%facet_type.eb0
// CHECK:STDOUT:   %.Self.loc134_38.1 => constants.%.Self.604
// CHECK:STDOUT:   %require_complete.loc134 => constants.%complete_type.872
// CHECK:STDOUT:   %.Self.as_type.loc134_44.1 => constants.%.Self.as_type.d34
// CHECK:STDOUT:   %SubWith.assoc_type => constants.%SubWith.assoc_type.6db
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.053
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.loc134 => constants.%SubWith.lookup_impl_witness.8be
// CHECK:STDOUT:   %impl.elem0.loc134_44.1 => constants.%impl.elem0.7af
// CHECK:STDOUT:   %Result.as_type.loc134_54.1 => constants.%Int64
// CHECK:STDOUT:   %facet_type.loc134_38 => constants.%facet_type.39e
// CHECK:STDOUT:   %T.loc134_4.1 => constants.%facet_value.fce
// CHECK:STDOUT:   %pattern_type.loc134 => constants.%pattern_type.a19
// CHECK:STDOUT:   %pattern_type.loc135_16 => constants.%pattern_type.41c
// CHECK:STDOUT:   %T.as_type.loc135_29.1 => constants.%Int64
// CHECK:STDOUT:   %pattern_type.loc135_27 => constants.%pattern_type.41c
// CHECK:STDOUT:   %pattern_type.loc135_33 => constants.%pattern_type.3b9
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc135_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc135_27 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc135_33 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc137 => constants.%complete_type.60a
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.3e8
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.loc137 => constants.%custom_witness.148
// CHECK:STDOUT:   %SubWith.facet => constants.%SubWith.facet.09f
// CHECK:STDOUT:   %SubWith.WithSelf.Op.type => constants.%SubWith.WithSelf.Op.type.4fe
// CHECK:STDOUT:   %.loc137_5.4 => constants.%.593
// CHECK:STDOUT:   %impl.elem1.loc137_5.2 => constants.%Op.b9c47d.12
// CHECK:STDOUT:   %specific_impl_fn.loc137_5.3 => constants.%Op.b9c47d.12
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.2a2
// CHECK:STDOUT:   %.loc137_5.5 => constants.%.08b
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.f83583.3
// CHECK:STDOUT:   %impl.elem0.loc137_5.2 => constants.%Int64.Op.ecb1a5.1
// CHECK:STDOUT:   %specific_impl_fn.loc137_5.4 => constants.%Int64.Op.ecb1a5.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @MulWith.loc146(constants.%Destroy.facet.a4a, constants.%Destroy.facet.700, constants.%facet_value.e5d) {
// CHECK:STDOUT:   %Result.loc143_9.1 => constants.%Destroy.facet.a4a
// CHECK:STDOUT:   %U.loc144_4.1 => constants.%Destroy.facet.700
// CHECK:STDOUT:   %U.as_type.loc145_36.1 => constants.%Int32
// CHECK:STDOUT:   %MulWith.type.loc145_36.1 => constants.%MulWith.type.d6e
// CHECK:STDOUT:   %facet_type.loc145_20 => constants.%facet_type.87a
// CHECK:STDOUT:   %.Self.loc145_38.1 => constants.%.Self.485
// CHECK:STDOUT:   %require_complete.loc145 => constants.%complete_type.e5c
// CHECK:STDOUT:   %.Self.as_type.loc145_44.1 => constants.%.Self.as_type.963
// CHECK:STDOUT:   %MulWith.assoc_type => constants.%MulWith.assoc_type.e4f
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.879
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.loc145 => constants.%MulWith.lookup_impl_witness.99b
// CHECK:STDOUT:   %impl.elem0.loc145_44.1 => constants.%impl.elem0.a71
// CHECK:STDOUT:   %Result.as_type.loc145_54.1 => constants.%Int64
// CHECK:STDOUT:   %facet_type.loc145_38 => constants.%facet_type.7cf
// CHECK:STDOUT:   %T.loc145_4.1 => constants.%facet_value.e5d
// CHECK:STDOUT:   %pattern_type.loc145 => constants.%pattern_type.2b4
// CHECK:STDOUT:   %pattern_type.loc146_16 => constants.%pattern_type.41c
// CHECK:STDOUT:   %T.as_type.loc146_29.1 => constants.%Int64
// CHECK:STDOUT:   %pattern_type.loc146_27 => constants.%pattern_type.41c
// CHECK:STDOUT:   %pattern_type.loc146_33 => constants.%pattern_type.3b9
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc146_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc146_27 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc146_33 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc148 => constants.%complete_type.e35
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.aaa
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.loc148 => constants.%custom_witness.d3e
// CHECK:STDOUT:   %MulWith.facet => constants.%MulWith.facet.f0b
// CHECK:STDOUT:   %MulWith.WithSelf.Op.type => constants.%MulWith.WithSelf.Op.type.bf9
// CHECK:STDOUT:   %.loc148_5.4 => constants.%.f35
// CHECK:STDOUT:   %impl.elem1.loc148_5.2 => constants.%Op.b9c47d.13
// CHECK:STDOUT:   %specific_impl_fn.loc148_5.3 => constants.%Op.b9c47d.13
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.2a2
// CHECK:STDOUT:   %.loc148_5.5 => constants.%.08b
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.f83583.3
// CHECK:STDOUT:   %impl.elem0.loc148_5.2 => constants.%Int64.Op.ecb1a5.1
// CHECK:STDOUT:   %specific_impl_fn.loc148_5.4 => constants.%Int64.Op.ecb1a5.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @DivWith.loc157(constants.%Destroy.facet.a4a, constants.%Destroy.facet.700, constants.%facet_value.ab0) {
// CHECK:STDOUT:   %Result.loc154_9.1 => constants.%Destroy.facet.a4a
// CHECK:STDOUT:   %U.loc155_4.1 => constants.%Destroy.facet.700
// CHECK:STDOUT:   %U.as_type.loc156_36.1 => constants.%Int32
// CHECK:STDOUT:   %DivWith.type.loc156_36.1 => constants.%DivWith.type.0f5
// CHECK:STDOUT:   %facet_type.loc156_20 => constants.%facet_type.81b
// CHECK:STDOUT:   %.Self.loc156_38.1 => constants.%.Self.ec5
// CHECK:STDOUT:   %require_complete.loc156 => constants.%complete_type.a26
// CHECK:STDOUT:   %.Self.as_type.loc156_44.1 => constants.%.Self.as_type.0ba
// CHECK:STDOUT:   %DivWith.assoc_type => constants.%DivWith.assoc_type.205
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.6dd
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.loc156 => constants.%DivWith.lookup_impl_witness.073
// CHECK:STDOUT:   %impl.elem0.loc156_44.1 => constants.%impl.elem0.920
// CHECK:STDOUT:   %Result.as_type.loc156_54.1 => constants.%Int64
// CHECK:STDOUT:   %facet_type.loc156_38 => constants.%facet_type.e1a
// CHECK:STDOUT:   %T.loc156_4.1 => constants.%facet_value.ab0
// CHECK:STDOUT:   %pattern_type.loc156 => constants.%pattern_type.d89
// CHECK:STDOUT:   %pattern_type.loc157_16 => constants.%pattern_type.41c
// CHECK:STDOUT:   %T.as_type.loc157_29.1 => constants.%Int64
// CHECK:STDOUT:   %pattern_type.loc157_27 => constants.%pattern_type.41c
// CHECK:STDOUT:   %pattern_type.loc157_33 => constants.%pattern_type.3b9
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc157_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc157_27 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc157_33 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc159 => constants.%complete_type.fbd
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.87f
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.loc159 => constants.%custom_witness.847
// CHECK:STDOUT:   %DivWith.facet => constants.%DivWith.facet.756
// CHECK:STDOUT:   %DivWith.WithSelf.Op.type => constants.%DivWith.WithSelf.Op.type.4be
// CHECK:STDOUT:   %.loc159_5.4 => constants.%.eae
// CHECK:STDOUT:   %impl.elem1.loc159_5.2 => constants.%Op.b9c47d.14
// CHECK:STDOUT:   %specific_impl_fn.loc159_5.3 => constants.%Op.b9c47d.14
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.2a2
// CHECK:STDOUT:   %.loc159_5.5 => constants.%.08b
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.f83583.3
// CHECK:STDOUT:   %impl.elem0.loc159_5.2 => constants.%Int64.Op.ecb1a5.1
// CHECK:STDOUT:   %specific_impl_fn.loc159_5.4 => constants.%Int64.Op.ecb1a5.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ModWith.loc168(constants.%Destroy.facet.a4a, constants.%Destroy.facet.700, constants.%facet_value.32f) {
// CHECK:STDOUT:   %Result.loc165_9.1 => constants.%Destroy.facet.a4a
// CHECK:STDOUT:   %U.loc166_4.1 => constants.%Destroy.facet.700
// CHECK:STDOUT:   %U.as_type.loc167_36.1 => constants.%Int32
// CHECK:STDOUT:   %ModWith.type.loc167_36.1 => constants.%ModWith.type.e59
// CHECK:STDOUT:   %facet_type.loc167_20 => constants.%facet_type.fed
// CHECK:STDOUT:   %.Self.loc167_38.1 => constants.%.Self.6f8
// CHECK:STDOUT:   %require_complete.loc167 => constants.%complete_type.9d2
// CHECK:STDOUT:   %.Self.as_type.loc167_44.1 => constants.%.Self.as_type.0bc
// CHECK:STDOUT:   %ModWith.assoc_type => constants.%ModWith.assoc_type.e6b
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.bf2
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.loc167 => constants.%ModWith.lookup_impl_witness.ba6
// CHECK:STDOUT:   %impl.elem0.loc167_44.1 => constants.%impl.elem0.1ad
// CHECK:STDOUT:   %Result.as_type.loc167_54.1 => constants.%Int64
// CHECK:STDOUT:   %facet_type.loc167_38 => constants.%facet_type.349
// CHECK:STDOUT:   %T.loc167_4.1 => constants.%facet_value.32f
// CHECK:STDOUT:   %pattern_type.loc167 => constants.%pattern_type.a62
// CHECK:STDOUT:   %pattern_type.loc168_16 => constants.%pattern_type.41c
// CHECK:STDOUT:   %T.as_type.loc168_29.1 => constants.%Int64
// CHECK:STDOUT:   %pattern_type.loc168_27 => constants.%pattern_type.41c
// CHECK:STDOUT:   %pattern_type.loc168_33 => constants.%pattern_type.3b9
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc168_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc168_27 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc168_33 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc170 => constants.%complete_type.965
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.adf
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.loc170 => constants.%custom_witness.4aa
// CHECK:STDOUT:   %ModWith.facet => constants.%ModWith.facet.735
// CHECK:STDOUT:   %ModWith.WithSelf.Op.type => constants.%ModWith.WithSelf.Op.type.e0f
// CHECK:STDOUT:   %.loc170_5.4 => constants.%.0ab
// CHECK:STDOUT:   %impl.elem1.loc170_5.2 => constants.%Op.b9c47d.15
// CHECK:STDOUT:   %specific_impl_fn.loc170_5.3 => constants.%Op.b9c47d.15
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.2a2
// CHECK:STDOUT:   %.loc170_5.5 => constants.%.08b
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.f83583.3
// CHECK:STDOUT:   %impl.elem0.loc170_5.2 => constants.%Int64.Op.ecb1a5.1
// CHECK:STDOUT:   %specific_impl_fn.loc170_5.4 => constants.%Int64.Op.ecb1a5.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AddAssignWith.loc225(constants.%U.67d, constants.%T.689) {
// CHECK:STDOUT:   %U.loc225_19.1 => constants.%U.67d
// CHECK:STDOUT:   %AddAssignWith.type.loc225_52.1 => constants.%AddAssignWith.type.dc8ed5.2
// CHECK:STDOUT:   %T.loc225_29.1 => constants.%T.689
// CHECK:STDOUT:   %pattern_type.loc225_29 => constants.%pattern_type.f87
// CHECK:STDOUT:   %T.as_type.loc225_62.1 => constants.%T.as_type.7d6
// CHECK:STDOUT:   %pattern_type.loc225_60 => constants.%pattern_type.657c0b.2
// CHECK:STDOUT:   %pattern_type.loc225_66 => constants.%pattern_type.51d1c4.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @SubAssignWith.loc231(constants.%U.67d, constants.%T.af3) {
// CHECK:STDOUT:   %U.loc231_19.1 => constants.%U.67d
// CHECK:STDOUT:   %SubAssignWith.type.loc231_52.1 => constants.%SubAssignWith.type.a95ddd.2
// CHECK:STDOUT:   %T.loc231_29.1 => constants.%T.af3
// CHECK:STDOUT:   %pattern_type.loc231_29 => constants.%pattern_type.3cd
// CHECK:STDOUT:   %T.as_type.loc231_62.1 => constants.%T.as_type.a39
// CHECK:STDOUT:   %pattern_type.loc231_60 => constants.%pattern_type.5c82ed.2
// CHECK:STDOUT:   %pattern_type.loc231_66 => constants.%pattern_type.51d1c4.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @MulAssignWith.loc237(constants.%U.67d, constants.%T.c8f) {
// CHECK:STDOUT:   %U.loc237_19.1 => constants.%U.67d
// CHECK:STDOUT:   %MulAssignWith.type.loc237_52.1 => constants.%MulAssignWith.type.c08ab6.2
// CHECK:STDOUT:   %T.loc237_29.1 => constants.%T.c8f
// CHECK:STDOUT:   %pattern_type.loc237_29 => constants.%pattern_type.78c
// CHECK:STDOUT:   %T.as_type.loc237_62.1 => constants.%T.as_type.26e
// CHECK:STDOUT:   %pattern_type.loc237_60 => constants.%pattern_type.3695a8.2
// CHECK:STDOUT:   %pattern_type.loc237_66 => constants.%pattern_type.51d1c4.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @DivAssignWith.loc243(constants.%U.67d, constants.%T.f22) {
// CHECK:STDOUT:   %U.loc243_19.1 => constants.%U.67d
// CHECK:STDOUT:   %DivAssignWith.type.loc243_52.1 => constants.%DivAssignWith.type.d7b8dd.2
// CHECK:STDOUT:   %T.loc243_29.1 => constants.%T.f22
// CHECK:STDOUT:   %pattern_type.loc243_29 => constants.%pattern_type.a67
// CHECK:STDOUT:   %T.as_type.loc243_62.1 => constants.%T.as_type.dc5
// CHECK:STDOUT:   %pattern_type.loc243_60 => constants.%pattern_type.ef84c8.2
// CHECK:STDOUT:   %pattern_type.loc243_66 => constants.%pattern_type.51d1c4.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ModAssignWith.loc249(constants.%U.67d, constants.%T.37b) {
// CHECK:STDOUT:   %U.loc249_19.1 => constants.%U.67d
// CHECK:STDOUT:   %ModAssignWith.type.loc249_52.1 => constants.%ModAssignWith.type.958f4d.2
// CHECK:STDOUT:   %T.loc249_29.1 => constants.%T.37b
// CHECK:STDOUT:   %pattern_type.loc249_29 => constants.%pattern_type.91c
// CHECK:STDOUT:   %T.as_type.loc249_62.1 => constants.%T.as_type.5ba
// CHECK:STDOUT:   %pattern_type.loc249_60 => constants.%pattern_type.22dddb.2
// CHECK:STDOUT:   %pattern_type.loc249_66 => constants.%pattern_type.51d1c4.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AddAssignWith.loc225(constants.%i16, constants.%AddAssignWith.facet.b45) {
// CHECK:STDOUT:   %U.loc225_19.1 => constants.%i16
// CHECK:STDOUT:   %AddAssignWith.type.loc225_52.1 => constants.%AddAssignWith.type.5a5
// CHECK:STDOUT:   %T.loc225_29.1 => constants.%AddAssignWith.facet.b45
// CHECK:STDOUT:   %pattern_type.loc225_29 => constants.%pattern_type.307
// CHECK:STDOUT:   %T.as_type.loc225_62.1 => constants.%Int16
// CHECK:STDOUT:   %pattern_type.loc225_60 => constants.%pattern_type.004
// CHECK:STDOUT:   %pattern_type.loc225_66 => constants.%pattern_type.0cb
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc225_55 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc225_66 => constants.%complete_type.2da
// CHECK:STDOUT:   %require_complete.loc227 => constants.%complete_type.c3b
// CHECK:STDOUT:   %AddAssignWith.assoc_type => constants.%AddAssignWith.assoc_type.12d
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.6ce
// CHECK:STDOUT:   %AddAssignWith.WithSelf.Op.type => constants.%AddAssignWith.WithSelf.Op.type.0f1
// CHECK:STDOUT:   %.loc227_5.2 => constants.%.dd3
// CHECK:STDOUT:   %AddAssignWith.lookup_impl_witness => constants.%custom_witness.1f9
// CHECK:STDOUT:   %impl.elem0.loc227_5.2 => constants.%Int16.Op.ecb1a5.13
// CHECK:STDOUT:   %specific_impl_fn.loc227_5.2 => constants.%Int16.Op.ecb1a5.13
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @SubAssignWith.loc231(constants.%i16, constants.%SubAssignWith.facet.351) {
// CHECK:STDOUT:   %U.loc231_19.1 => constants.%i16
// CHECK:STDOUT:   %SubAssignWith.type.loc231_52.1 => constants.%SubAssignWith.type.2da
// CHECK:STDOUT:   %T.loc231_29.1 => constants.%SubAssignWith.facet.351
// CHECK:STDOUT:   %pattern_type.loc231_29 => constants.%pattern_type.2c7
// CHECK:STDOUT:   %T.as_type.loc231_62.1 => constants.%Int16
// CHECK:STDOUT:   %pattern_type.loc231_60 => constants.%pattern_type.004
// CHECK:STDOUT:   %pattern_type.loc231_66 => constants.%pattern_type.0cb
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc231_55 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc231_66 => constants.%complete_type.2da
// CHECK:STDOUT:   %require_complete.loc233 => constants.%complete_type.289
// CHECK:STDOUT:   %SubAssignWith.assoc_type => constants.%SubAssignWith.assoc_type.f3f
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.1c8
// CHECK:STDOUT:   %SubAssignWith.WithSelf.Op.type => constants.%SubAssignWith.WithSelf.Op.type.286
// CHECK:STDOUT:   %.loc233_5.2 => constants.%.d9d
// CHECK:STDOUT:   %SubAssignWith.lookup_impl_witness => constants.%custom_witness.922
// CHECK:STDOUT:   %impl.elem0.loc233_5.2 => constants.%Int16.Op.ecb1a5.14
// CHECK:STDOUT:   %specific_impl_fn.loc233_5.2 => constants.%Int16.Op.ecb1a5.14
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @MulAssignWith.loc237(constants.%i16, constants.%MulAssignWith.facet.2aa) {
// CHECK:STDOUT:   %U.loc237_19.1 => constants.%i16
// CHECK:STDOUT:   %MulAssignWith.type.loc237_52.1 => constants.%MulAssignWith.type.a70
// CHECK:STDOUT:   %T.loc237_29.1 => constants.%MulAssignWith.facet.2aa
// CHECK:STDOUT:   %pattern_type.loc237_29 => constants.%pattern_type.0b2
// CHECK:STDOUT:   %T.as_type.loc237_62.1 => constants.%Int16
// CHECK:STDOUT:   %pattern_type.loc237_60 => constants.%pattern_type.004
// CHECK:STDOUT:   %pattern_type.loc237_66 => constants.%pattern_type.0cb
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc237_55 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc237_66 => constants.%complete_type.2da
// CHECK:STDOUT:   %require_complete.loc239 => constants.%complete_type.9b6
// CHECK:STDOUT:   %MulAssignWith.assoc_type => constants.%MulAssignWith.assoc_type.f71
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.037
// CHECK:STDOUT:   %MulAssignWith.WithSelf.Op.type => constants.%MulAssignWith.WithSelf.Op.type.29b
// CHECK:STDOUT:   %.loc239_5.2 => constants.%.a28
// CHECK:STDOUT:   %MulAssignWith.lookup_impl_witness => constants.%custom_witness.c9b
// CHECK:STDOUT:   %impl.elem0.loc239_5.2 => constants.%Int16.Op.ecb1a5.15
// CHECK:STDOUT:   %specific_impl_fn.loc239_5.2 => constants.%Int16.Op.ecb1a5.15
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @DivAssignWith.loc243(constants.%i16, constants.%DivAssignWith.facet.253) {
// CHECK:STDOUT:   %U.loc243_19.1 => constants.%i16
// CHECK:STDOUT:   %DivAssignWith.type.loc243_52.1 => constants.%DivAssignWith.type.7f8
// CHECK:STDOUT:   %T.loc243_29.1 => constants.%DivAssignWith.facet.253
// CHECK:STDOUT:   %pattern_type.loc243_29 => constants.%pattern_type.455
// CHECK:STDOUT:   %T.as_type.loc243_62.1 => constants.%Int16
// CHECK:STDOUT:   %pattern_type.loc243_60 => constants.%pattern_type.004
// CHECK:STDOUT:   %pattern_type.loc243_66 => constants.%pattern_type.0cb
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc243_55 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc243_66 => constants.%complete_type.2da
// CHECK:STDOUT:   %require_complete.loc245 => constants.%complete_type.41f
// CHECK:STDOUT:   %DivAssignWith.assoc_type => constants.%DivAssignWith.assoc_type.e83
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.065
// CHECK:STDOUT:   %DivAssignWith.WithSelf.Op.type => constants.%DivAssignWith.WithSelf.Op.type.604
// CHECK:STDOUT:   %.loc245_5.2 => constants.%.0f0
// CHECK:STDOUT:   %DivAssignWith.lookup_impl_witness => constants.%custom_witness.719
// CHECK:STDOUT:   %impl.elem0.loc245_5.2 => constants.%Int16.Op.ecb1a5.16
// CHECK:STDOUT:   %specific_impl_fn.loc245_5.2 => constants.%Int16.Op.ecb1a5.16
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ModAssignWith.loc249(constants.%i16, constants.%ModAssignWith.facet.c21) {
// CHECK:STDOUT:   %U.loc249_19.1 => constants.%i16
// CHECK:STDOUT:   %ModAssignWith.type.loc249_52.1 => constants.%ModAssignWith.type.eee
// CHECK:STDOUT:   %T.loc249_29.1 => constants.%ModAssignWith.facet.c21
// CHECK:STDOUT:   %pattern_type.loc249_29 => constants.%pattern_type.df1
// CHECK:STDOUT:   %T.as_type.loc249_62.1 => constants.%Int16
// CHECK:STDOUT:   %pattern_type.loc249_60 => constants.%pattern_type.004
// CHECK:STDOUT:   %pattern_type.loc249_66 => constants.%pattern_type.0cb
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc249_55 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc249_66 => constants.%complete_type.2da
// CHECK:STDOUT:   %require_complete.loc251 => constants.%complete_type.955
// CHECK:STDOUT:   %ModAssignWith.assoc_type => constants.%ModAssignWith.assoc_type.08a
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.c9f
// CHECK:STDOUT:   %ModAssignWith.WithSelf.Op.type => constants.%ModAssignWith.WithSelf.Op.type.436
// CHECK:STDOUT:   %.loc251_5.2 => constants.%.8a6
// CHECK:STDOUT:   %ModAssignWith.lookup_impl_witness => constants.%custom_witness.827
// CHECK:STDOUT:   %impl.elem0.loc251_5.2 => constants.%Int16.Op.ecb1a5.17
// CHECK:STDOUT:   %specific_impl_fn.loc251_5.2 => constants.%Int16.Op.ecb1a5.17
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AddAssignWith.loc225(constants.%Int32, constants.%AddAssignWith.facet.e5f) {
// CHECK:STDOUT:   %U.loc225_19.1 => constants.%Int32
// CHECK:STDOUT:   %AddAssignWith.type.loc225_52.1 => constants.%AddAssignWith.type.22f
// CHECK:STDOUT:   %T.loc225_29.1 => constants.%AddAssignWith.facet.e5f
// CHECK:STDOUT:   %pattern_type.loc225_29 => constants.%pattern_type.2d7
// CHECK:STDOUT:   %T.as_type.loc225_62.1 => constants.%Int32
// CHECK:STDOUT:   %pattern_type.loc225_60 => constants.%pattern_type.3b9
// CHECK:STDOUT:   %pattern_type.loc225_66 => constants.%pattern_type.3b9
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc225_55 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc225_66 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc227 => constants.%complete_type.547
// CHECK:STDOUT:   %AddAssignWith.assoc_type => constants.%AddAssignWith.assoc_type.6a5
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.20f
// CHECK:STDOUT:   %AddAssignWith.WithSelf.Op.type => constants.%AddAssignWith.WithSelf.Op.type.417
// CHECK:STDOUT:   %.loc227_5.2 => constants.%.c71
// CHECK:STDOUT:   %AddAssignWith.lookup_impl_witness => constants.%custom_witness.167b9d.1
// CHECK:STDOUT:   %impl.elem0.loc227_5.2 => constants.%Int32.Op.ecb1a5.12
// CHECK:STDOUT:   %specific_impl_fn.loc227_5.2 => constants.%Int32.Op.ecb1a5.12
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @SubAssignWith.loc231(constants.%Int32, constants.%SubAssignWith.facet.c12) {
// CHECK:STDOUT:   %U.loc231_19.1 => constants.%Int32
// CHECK:STDOUT:   %SubAssignWith.type.loc231_52.1 => constants.%SubAssignWith.type.c0c
// CHECK:STDOUT:   %T.loc231_29.1 => constants.%SubAssignWith.facet.c12
// CHECK:STDOUT:   %pattern_type.loc231_29 => constants.%pattern_type.0ae
// CHECK:STDOUT:   %T.as_type.loc231_62.1 => constants.%Int32
// CHECK:STDOUT:   %pattern_type.loc231_60 => constants.%pattern_type.3b9
// CHECK:STDOUT:   %pattern_type.loc231_66 => constants.%pattern_type.3b9
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc231_55 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc231_66 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc233 => constants.%complete_type.d8e
// CHECK:STDOUT:   %SubAssignWith.assoc_type => constants.%SubAssignWith.assoc_type.5d1
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.5e1
// CHECK:STDOUT:   %SubAssignWith.WithSelf.Op.type => constants.%SubAssignWith.WithSelf.Op.type.d08
// CHECK:STDOUT:   %.loc233_5.2 => constants.%.8b4
// CHECK:STDOUT:   %SubAssignWith.lookup_impl_witness => constants.%custom_witness.0d70d6.1
// CHECK:STDOUT:   %impl.elem0.loc233_5.2 => constants.%Int32.Op.ecb1a5.14
// CHECK:STDOUT:   %specific_impl_fn.loc233_5.2 => constants.%Int32.Op.ecb1a5.14
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @MulAssignWith.loc237(constants.%Int32, constants.%MulAssignWith.facet.7cb) {
// CHECK:STDOUT:   %U.loc237_19.1 => constants.%Int32
// CHECK:STDOUT:   %MulAssignWith.type.loc237_52.1 => constants.%MulAssignWith.type.f1b
// CHECK:STDOUT:   %T.loc237_29.1 => constants.%MulAssignWith.facet.7cb
// CHECK:STDOUT:   %pattern_type.loc237_29 => constants.%pattern_type.b7f
// CHECK:STDOUT:   %T.as_type.loc237_62.1 => constants.%Int32
// CHECK:STDOUT:   %pattern_type.loc237_60 => constants.%pattern_type.3b9
// CHECK:STDOUT:   %pattern_type.loc237_66 => constants.%pattern_type.3b9
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc237_55 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc237_66 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc239 => constants.%complete_type.4e2
// CHECK:STDOUT:   %MulAssignWith.assoc_type => constants.%MulAssignWith.assoc_type.38d
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.bb5
// CHECK:STDOUT:   %MulAssignWith.WithSelf.Op.type => constants.%MulAssignWith.WithSelf.Op.type.e8b
// CHECK:STDOUT:   %.loc239_5.2 => constants.%.d1f
// CHECK:STDOUT:   %MulAssignWith.lookup_impl_witness => constants.%custom_witness.027973.1
// CHECK:STDOUT:   %impl.elem0.loc239_5.2 => constants.%Int32.Op.ecb1a5.15
// CHECK:STDOUT:   %specific_impl_fn.loc239_5.2 => constants.%Int32.Op.ecb1a5.15
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @DivAssignWith.loc243(constants.%Int32, constants.%DivAssignWith.facet.ffe) {
// CHECK:STDOUT:   %U.loc243_19.1 => constants.%Int32
// CHECK:STDOUT:   %DivAssignWith.type.loc243_52.1 => constants.%DivAssignWith.type.453
// CHECK:STDOUT:   %T.loc243_29.1 => constants.%DivAssignWith.facet.ffe
// CHECK:STDOUT:   %pattern_type.loc243_29 => constants.%pattern_type.964
// CHECK:STDOUT:   %T.as_type.loc243_62.1 => constants.%Int32
// CHECK:STDOUT:   %pattern_type.loc243_60 => constants.%pattern_type.3b9
// CHECK:STDOUT:   %pattern_type.loc243_66 => constants.%pattern_type.3b9
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc243_55 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc243_66 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc245 => constants.%complete_type.463
// CHECK:STDOUT:   %DivAssignWith.assoc_type => constants.%DivAssignWith.assoc_type.693
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.50a
// CHECK:STDOUT:   %DivAssignWith.WithSelf.Op.type => constants.%DivAssignWith.WithSelf.Op.type.f70
// CHECK:STDOUT:   %.loc245_5.2 => constants.%.2b1
// CHECK:STDOUT:   %DivAssignWith.lookup_impl_witness => constants.%custom_witness.30b4a6.1
// CHECK:STDOUT:   %impl.elem0.loc245_5.2 => constants.%Int32.Op.ecb1a5.16
// CHECK:STDOUT:   %specific_impl_fn.loc245_5.2 => constants.%Int32.Op.ecb1a5.16
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ModAssignWith.loc249(constants.%Int32, constants.%ModAssignWith.facet.1bc) {
// CHECK:STDOUT:   %U.loc249_19.1 => constants.%Int32
// CHECK:STDOUT:   %ModAssignWith.type.loc249_52.1 => constants.%ModAssignWith.type.390
// CHECK:STDOUT:   %T.loc249_29.1 => constants.%ModAssignWith.facet.1bc
// CHECK:STDOUT:   %pattern_type.loc249_29 => constants.%pattern_type.dd3
// CHECK:STDOUT:   %T.as_type.loc249_62.1 => constants.%Int32
// CHECK:STDOUT:   %pattern_type.loc249_60 => constants.%pattern_type.3b9
// CHECK:STDOUT:   %pattern_type.loc249_66 => constants.%pattern_type.3b9
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc249_55 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc249_66 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc251 => constants.%complete_type.e8b
// CHECK:STDOUT:   %ModAssignWith.assoc_type => constants.%ModAssignWith.assoc_type.e54
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.ec3
// CHECK:STDOUT:   %ModAssignWith.WithSelf.Op.type => constants.%ModAssignWith.WithSelf.Op.type.ac2
// CHECK:STDOUT:   %.loc251_5.2 => constants.%.e93
// CHECK:STDOUT:   %ModAssignWith.lookup_impl_witness => constants.%custom_witness.bbc21d.1
// CHECK:STDOUT:   %impl.elem0.loc251_5.2 => constants.%Int32.Op.ecb1a5.17
// CHECK:STDOUT:   %specific_impl_fn.loc251_5.2 => constants.%Int32.Op.ecb1a5.17
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AddAssignWith.loc225(constants.%Int32, constants.%AddAssignWith.facet.62e) {
// CHECK:STDOUT:   %U.loc225_19.1 => constants.%Int32
// CHECK:STDOUT:   %AddAssignWith.type.loc225_52.1 => constants.%AddAssignWith.type.22f
// CHECK:STDOUT:   %T.loc225_29.1 => constants.%AddAssignWith.facet.62e
// CHECK:STDOUT:   %pattern_type.loc225_29 => constants.%pattern_type.2d7
// CHECK:STDOUT:   %T.as_type.loc225_62.1 => constants.%Int64
// CHECK:STDOUT:   %pattern_type.loc225_60 => constants.%pattern_type.41c
// CHECK:STDOUT:   %pattern_type.loc225_66 => constants.%pattern_type.3b9
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc225_55 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc225_66 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc227 => constants.%complete_type.547
// CHECK:STDOUT:   %AddAssignWith.assoc_type => constants.%AddAssignWith.assoc_type.6a5
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.20f
// CHECK:STDOUT:   %AddAssignWith.WithSelf.Op.type => constants.%AddAssignWith.WithSelf.Op.type.eaa
// CHECK:STDOUT:   %.loc227_5.2 => constants.%.869
// CHECK:STDOUT:   %AddAssignWith.lookup_impl_witness => constants.%custom_witness.167b9d.2
// CHECK:STDOUT:   %impl.elem0.loc227_5.2 => constants.%Int64.Op.ecb1a5.2
// CHECK:STDOUT:   %specific_impl_fn.loc227_5.2 => constants.%Int64.Op.ecb1a5.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @SubAssignWith.loc231(constants.%Int32, constants.%SubAssignWith.facet.1f9) {
// CHECK:STDOUT:   %U.loc231_19.1 => constants.%Int32
// CHECK:STDOUT:   %SubAssignWith.type.loc231_52.1 => constants.%SubAssignWith.type.c0c
// CHECK:STDOUT:   %T.loc231_29.1 => constants.%SubAssignWith.facet.1f9
// CHECK:STDOUT:   %pattern_type.loc231_29 => constants.%pattern_type.0ae
// CHECK:STDOUT:   %T.as_type.loc231_62.1 => constants.%Int64
// CHECK:STDOUT:   %pattern_type.loc231_60 => constants.%pattern_type.41c
// CHECK:STDOUT:   %pattern_type.loc231_66 => constants.%pattern_type.3b9
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc231_55 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc231_66 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc233 => constants.%complete_type.d8e
// CHECK:STDOUT:   %SubAssignWith.assoc_type => constants.%SubAssignWith.assoc_type.5d1
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.5e1
// CHECK:STDOUT:   %SubAssignWith.WithSelf.Op.type => constants.%SubAssignWith.WithSelf.Op.type.6f7
// CHECK:STDOUT:   %.loc233_5.2 => constants.%.792
// CHECK:STDOUT:   %SubAssignWith.lookup_impl_witness => constants.%custom_witness.0d70d6.2
// CHECK:STDOUT:   %impl.elem0.loc233_5.2 => constants.%Int64.Op.ecb1a5.4
// CHECK:STDOUT:   %specific_impl_fn.loc233_5.2 => constants.%Int64.Op.ecb1a5.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @MulAssignWith.loc237(constants.%Int32, constants.%MulAssignWith.facet.b47) {
// CHECK:STDOUT:   %U.loc237_19.1 => constants.%Int32
// CHECK:STDOUT:   %MulAssignWith.type.loc237_52.1 => constants.%MulAssignWith.type.f1b
// CHECK:STDOUT:   %T.loc237_29.1 => constants.%MulAssignWith.facet.b47
// CHECK:STDOUT:   %pattern_type.loc237_29 => constants.%pattern_type.b7f
// CHECK:STDOUT:   %T.as_type.loc237_62.1 => constants.%Int64
// CHECK:STDOUT:   %pattern_type.loc237_60 => constants.%pattern_type.41c
// CHECK:STDOUT:   %pattern_type.loc237_66 => constants.%pattern_type.3b9
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc237_55 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc237_66 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc239 => constants.%complete_type.4e2
// CHECK:STDOUT:   %MulAssignWith.assoc_type => constants.%MulAssignWith.assoc_type.38d
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.bb5
// CHECK:STDOUT:   %MulAssignWith.WithSelf.Op.type => constants.%MulAssignWith.WithSelf.Op.type.830
// CHECK:STDOUT:   %.loc239_5.2 => constants.%.e4c
// CHECK:STDOUT:   %MulAssignWith.lookup_impl_witness => constants.%custom_witness.027973.2
// CHECK:STDOUT:   %impl.elem0.loc239_5.2 => constants.%Int64.Op.ecb1a5.5
// CHECK:STDOUT:   %specific_impl_fn.loc239_5.2 => constants.%Int64.Op.ecb1a5.5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @DivAssignWith.loc243(constants.%Int32, constants.%DivAssignWith.facet.f27) {
// CHECK:STDOUT:   %U.loc243_19.1 => constants.%Int32
// CHECK:STDOUT:   %DivAssignWith.type.loc243_52.1 => constants.%DivAssignWith.type.453
// CHECK:STDOUT:   %T.loc243_29.1 => constants.%DivAssignWith.facet.f27
// CHECK:STDOUT:   %pattern_type.loc243_29 => constants.%pattern_type.964
// CHECK:STDOUT:   %T.as_type.loc243_62.1 => constants.%Int64
// CHECK:STDOUT:   %pattern_type.loc243_60 => constants.%pattern_type.41c
// CHECK:STDOUT:   %pattern_type.loc243_66 => constants.%pattern_type.3b9
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc243_55 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc243_66 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc245 => constants.%complete_type.463
// CHECK:STDOUT:   %DivAssignWith.assoc_type => constants.%DivAssignWith.assoc_type.693
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.50a
// CHECK:STDOUT:   %DivAssignWith.WithSelf.Op.type => constants.%DivAssignWith.WithSelf.Op.type.494
// CHECK:STDOUT:   %.loc245_5.2 => constants.%.55b
// CHECK:STDOUT:   %DivAssignWith.lookup_impl_witness => constants.%custom_witness.30b4a6.2
// CHECK:STDOUT:   %impl.elem0.loc245_5.2 => constants.%Int64.Op.ecb1a5.6
// CHECK:STDOUT:   %specific_impl_fn.loc245_5.2 => constants.%Int64.Op.ecb1a5.6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ModAssignWith.loc249(constants.%Int32, constants.%ModAssignWith.facet.a21) {
// CHECK:STDOUT:   %U.loc249_19.1 => constants.%Int32
// CHECK:STDOUT:   %ModAssignWith.type.loc249_52.1 => constants.%ModAssignWith.type.390
// CHECK:STDOUT:   %T.loc249_29.1 => constants.%ModAssignWith.facet.a21
// CHECK:STDOUT:   %pattern_type.loc249_29 => constants.%pattern_type.dd3
// CHECK:STDOUT:   %T.as_type.loc249_62.1 => constants.%Int64
// CHECK:STDOUT:   %pattern_type.loc249_60 => constants.%pattern_type.41c
// CHECK:STDOUT:   %pattern_type.loc249_66 => constants.%pattern_type.3b9
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc249_55 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc249_66 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc251 => constants.%complete_type.e8b
// CHECK:STDOUT:   %ModAssignWith.assoc_type => constants.%ModAssignWith.assoc_type.e54
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.ec3
// CHECK:STDOUT:   %ModAssignWith.WithSelf.Op.type => constants.%ModAssignWith.WithSelf.Op.type.bed
// CHECK:STDOUT:   %.loc251_5.2 => constants.%.074
// CHECK:STDOUT:   %ModAssignWith.lookup_impl_witness => constants.%custom_witness.bbc21d.2
// CHECK:STDOUT:   %impl.elem0.loc251_5.2 => constants.%Int64.Op.ecb1a5.7
// CHECK:STDOUT:   %specific_impl_fn.loc251_5.2 => constants.%Int64.Op.ecb1a5.7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestInc(constants.%T.fed) {
// CHECK:STDOUT:   %T.loc278_13.1 => constants.%T.fed
// CHECK:STDOUT:   %T.as_type.loc278_33.1 => constants.%T.as_type.1d1
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.8ba822.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestDec(constants.%T.c61) {
// CHECK:STDOUT:   %T.loc284_13.1 => constants.%T.c61
// CHECK:STDOUT:   %T.as_type.loc284_33.1 => constants.%T.as_type.5ae
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.a489aa.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestNegate(constants.%Result, constants.%T.3f8) {
// CHECK:STDOUT:   %Result.loc292_9.1 => constants.%Result
// CHECK:STDOUT:   %Result.as_type.loc293_50.1 => constants.%Result.as_type
// CHECK:STDOUT:   %facet_type => constants.%facet_type.4bf
// CHECK:STDOUT:   %T.loc293_4.1 => constants.%T.3f8
// CHECK:STDOUT:   %pattern_type.loc293 => constants.%pattern_type.ad3
// CHECK:STDOUT:   %pattern_type.loc294_11 => constants.%pattern_type.5eec30.2
// CHECK:STDOUT:   %T.as_type.loc294_24.1 => constants.%T.as_type.013
// CHECK:STDOUT:   %pattern_type.loc294_22 => constants.%pattern_type.48b
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestInc(constants.%Inc.facet.338) {
// CHECK:STDOUT:   %T.loc278_13.1 => constants.%Inc.facet.338
// CHECK:STDOUT:   %T.as_type.loc278_33.1 => constants.%Int32
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.3b9
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.357
// CHECK:STDOUT:   %Inc.WithSelf.Op.type => constants.%Inc.WithSelf.Op.type.f93
// CHECK:STDOUT:   %.loc280 => constants.%.fb7
// CHECK:STDOUT:   %Inc.lookup_impl_witness => constants.%custom_witness.104
// CHECK:STDOUT:   %impl.elem0.loc280_3.2 => constants.%Int32.Op.ecb1a5.18
// CHECK:STDOUT:   %specific_impl_fn.loc280_3.2 => constants.%Int32.Op.ecb1a5.18
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestDec(constants.%Dec.facet.b97) {
// CHECK:STDOUT:   %T.loc284_13.1 => constants.%Dec.facet.b97
// CHECK:STDOUT:   %T.as_type.loc284_33.1 => constants.%Int32
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.3b9
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.357
// CHECK:STDOUT:   %Dec.WithSelf.Op.type => constants.%Dec.WithSelf.Op.type.221
// CHECK:STDOUT:   %.loc286 => constants.%.2d8
// CHECK:STDOUT:   %Dec.lookup_impl_witness => constants.%custom_witness.a2f
// CHECK:STDOUT:   %impl.elem0.loc286_3.2 => constants.%Int32.Op.ecb1a5.19
// CHECK:STDOUT:   %specific_impl_fn.loc286_3.2 => constants.%Int32.Op.ecb1a5.19
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestNegate(constants.%Destroy.facet.a4a, constants.%facet_value.658) {
// CHECK:STDOUT:   %Result.loc292_9.1 => constants.%Destroy.facet.a4a
// CHECK:STDOUT:   %Result.as_type.loc293_50.1 => constants.%Int64
// CHECK:STDOUT:   %facet_type => constants.%facet_type.67d
// CHECK:STDOUT:   %T.loc293_4.1 => constants.%facet_value.658
// CHECK:STDOUT:   %pattern_type.loc293 => constants.%pattern_type.100
// CHECK:STDOUT:   %pattern_type.loc294_11 => constants.%pattern_type.41c
// CHECK:STDOUT:   %T.as_type.loc294_24.1 => constants.%Int16
// CHECK:STDOUT:   %pattern_type.loc294_22 => constants.%pattern_type.004
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc294_11 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc294_22 => constants.%complete_type.357
// CHECK:STDOUT:   %Negate.lookup_impl_witness => constants.%custom_witness.243
// CHECK:STDOUT:   %Negate.facet => constants.%Negate.facet.7e2
// CHECK:STDOUT:   %Negate.WithSelf.Op.type => constants.%Negate.WithSelf.Op.type.394
// CHECK:STDOUT:   %.loc296_3.3 => constants.%.187
// CHECK:STDOUT:   %impl.elem1.loc296_3.2 => constants.%Int16.Op.ecb1a5.18
// CHECK:STDOUT:   %specific_impl_fn.loc296_3.3 => constants.%Int16.Op.ecb1a5.18
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.2a2
// CHECK:STDOUT:   %.loc296_3.4 => constants.%.08b
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.f83583.3
// CHECK:STDOUT:   %impl.elem0.loc296_3.2 => constants.%Int64.Op.ecb1a5.1
// CHECK:STDOUT:   %specific_impl_fn.loc296_3.4 => constants.%Int64.Op.ecb1a5.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestNegate(constants.%Destroy.facet.700, constants.%facet_value.cc4) {
// CHECK:STDOUT:   %Result.loc292_9.1 => constants.%Destroy.facet.700
// CHECK:STDOUT:   %Result.as_type.loc293_50.1 => constants.%Int32
// CHECK:STDOUT:   %facet_type => constants.%facet_type.016
// CHECK:STDOUT:   %T.loc293_4.1 => constants.%facet_value.cc4
// CHECK:STDOUT:   %pattern_type.loc293 => constants.%pattern_type.17e
// CHECK:STDOUT:   %pattern_type.loc294_11 => constants.%pattern_type.3b9
// CHECK:STDOUT:   %T.as_type.loc294_24.1 => constants.%Int32
// CHECK:STDOUT:   %pattern_type.loc294_22 => constants.%pattern_type.3b9
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc294_11 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc294_22 => constants.%complete_type.357
// CHECK:STDOUT:   %Negate.lookup_impl_witness => constants.%custom_witness.9ae
// CHECK:STDOUT:   %Negate.facet => constants.%Negate.facet.41a
// CHECK:STDOUT:   %Negate.WithSelf.Op.type => constants.%Negate.WithSelf.Op.type.81f
// CHECK:STDOUT:   %.loc296_3.3 => constants.%.71b4
// CHECK:STDOUT:   %impl.elem1.loc296_3.2 => constants.%Int32.Op.ecb1a5.20
// CHECK:STDOUT:   %specific_impl_fn.loc296_3.3 => constants.%Int32.Op.ecb1a5.20
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.7bc
// CHECK:STDOUT:   %.loc296_3.4 => constants.%.362
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.f83583.2
// CHECK:STDOUT:   %impl.elem0.loc296_3.2 => constants.%Int32.Op.ecb1a5.1
// CHECK:STDOUT:   %specific_impl_fn.loc296_3.4 => constants.%Int32.Op.ecb1a5.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
