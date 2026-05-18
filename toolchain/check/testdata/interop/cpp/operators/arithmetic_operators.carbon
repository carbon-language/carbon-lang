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
// CHECK:STDOUT:   %U.ac3: %Destroy.type = symbolic_binding U, 1 [symbolic]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Other: type = symbolic_binding Other, 0 [symbolic]
// CHECK:STDOUT:   %AddWith.type.adb: type = facet_type <@AddWith.1, @AddWith.1(%Other)> [symbolic]
// CHECK:STDOUT:   %Self.e74: %AddWith.type.adb = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %AddWith.assoc_type.329: type = assoc_entity_type @AddWith.1, @AddWith.1(%Other) [symbolic]
// CHECK:STDOUT:   %AddWith.WithSelf.Op.type.171: type = fn_type @AddWith.WithSelf.Op, @AddWith.WithSelf(%Other, %Self.e74) [symbolic]
// CHECK:STDOUT:   %AddWith.WithSelf.Op.3fe: %AddWith.WithSelf.Op.type.171 = struct_value () [symbolic]
// CHECK:STDOUT:   %U.as_type.04c: type = facet_access_type %U.ac3 [symbolic]
// CHECK:STDOUT:   %AddWith.type.e1b: type = facet_type <@AddWith.1, @AddWith.1(%U.as_type.04c)> [symbolic]
// CHECK:STDOUT:   %facet_type.e9c: type = facet_type <@Destroy & @AddWith.1, @AddWith.1(%U.as_type.04c)> [symbolic]
// CHECK:STDOUT:   %.Self.33f: %facet_type.e9c = symbolic_binding .Self [symbolic]
// CHECK:STDOUT:   %AddWith.assoc_type.91f: type = assoc_entity_type @AddWith.1, @AddWith.1(%U.as_type.04c) [symbolic]
// CHECK:STDOUT:   %assoc0.535: %AddWith.assoc_type.91f = assoc_entity element0, imports.%Core.import_ref.3ad [symbolic]
// CHECK:STDOUT:   %assoc1.85e: %AddWith.assoc_type.91f = assoc_entity element1, imports.%Core.import_ref.2b2 [symbolic]
// CHECK:STDOUT:   %require_complete.b5e: <witness> = require_complete_type %facet_type.e9c [symbolic]
// CHECK:STDOUT:   %.Self.as_type.bf0: type = facet_access_type %.Self.33f [symbolic]
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.82b: <witness> = lookup_impl_witness %.Self.33f, @AddWith.1, @AddWith.1(%U.as_type.04c) [symbolic]
// CHECK:STDOUT:   %impl.elem0.f47: type = impl_witness_access %AddWith.lookup_impl_witness.82b, element0 [symbolic]
// CHECK:STDOUT:   %Result.as_type: type = facet_access_type %Result [symbolic]
// CHECK:STDOUT:   %facet_type.f91: type = facet_type <@Destroy & @AddWith.1, @AddWith.1(%U.as_type.04c) where %impl.elem0.f47 = %Result.as_type> [symbolic]
// CHECK:STDOUT:   %pattern_type.ed1: type = pattern_type %facet_type.f91 [symbolic]
// CHECK:STDOUT:   %T.db8: %facet_type.f91 = symbolic_binding T, 2 [symbolic]
// CHECK:STDOUT:   %pattern_type.96123d.2: type = pattern_type %Result.as_type [symbolic]
// CHECK:STDOUT:   %T.as_type.94c: type = facet_access_type %T.db8 [symbolic]
// CHECK:STDOUT:   %pattern_type.add: type = pattern_type %T.as_type.94c [symbolic]
// CHECK:STDOUT:   %pattern_type.c6d: type = pattern_type %U.as_type.04c [symbolic]
// CHECK:STDOUT:   %require_complete.911: <witness> = require_complete_type %AddWith.type.e1b [symbolic]
// CHECK:STDOUT:   %assoc1.a2f: %AddWith.assoc_type.329 = assoc_entity element1, imports.%Core.import_ref.8be [symbolic]
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.1f7: <witness> = lookup_impl_witness %T.db8, @AddWith.1, @AddWith.1(%U.as_type.04c) [symbolic]
// CHECK:STDOUT:   %AddWith.facet.4a7: %AddWith.type.e1b = facet_value %T.as_type.94c, (%AddWith.lookup_impl_witness.1f7) [symbolic]
// CHECK:STDOUT:   %AddWith.WithSelf.Op.type.185: type = fn_type @AddWith.WithSelf.Op, @AddWith.WithSelf(%U.as_type.04c, %AddWith.facet.4a7) [symbolic]
// CHECK:STDOUT:   %.993: type = fn_type_with_self_type %AddWith.WithSelf.Op.type.185, %AddWith.facet.4a7 [symbolic]
// CHECK:STDOUT:   %impl.elem1.ad4: %.993 = impl_witness_access %AddWith.lookup_impl_witness.1f7, element1 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.d63: <specific function> = specific_impl_function %impl.elem1.ad4, @AddWith.WithSelf.Op(%U.as_type.04c, %AddWith.facet.4a7) [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %Result, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.b9d4b3.2: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Result) [symbolic]
// CHECK:STDOUT:   %.387: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.b9d4b3.2, %Result [symbolic]
// CHECK:STDOUT:   %impl.elem0.2e6: %.387 = impl_witness_access %Destroy.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.62a: <specific function> = specific_impl_function %impl.elem0.2e6, @Destroy.WithSelf.Op(%Result) [symbolic]
// CHECK:STDOUT:   %SubWith.type.c59: type = facet_type <@SubWith.1, @SubWith.1(%Other)> [symbolic]
// CHECK:STDOUT:   %Self.97e: %SubWith.type.c59 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %SubWith.assoc_type.02f: type = assoc_entity_type @SubWith.1, @SubWith.1(%Other) [symbolic]
// CHECK:STDOUT:   %SubWith.WithSelf.Op.type.353: type = fn_type @SubWith.WithSelf.Op, @SubWith.WithSelf(%Other, %Self.97e) [symbolic]
// CHECK:STDOUT:   %SubWith.WithSelf.Op.197: %SubWith.WithSelf.Op.type.353 = struct_value () [symbolic]
// CHECK:STDOUT:   %SubWith.type.2e8: type = facet_type <@SubWith.1, @SubWith.1(%U.as_type.04c)> [symbolic]
// CHECK:STDOUT:   %facet_type.241: type = facet_type <@Destroy & @SubWith.1, @SubWith.1(%U.as_type.04c)> [symbolic]
// CHECK:STDOUT:   %.Self.a38: %facet_type.241 = symbolic_binding .Self [symbolic]
// CHECK:STDOUT:   %SubWith.assoc_type.760: type = assoc_entity_type @SubWith.1, @SubWith.1(%U.as_type.04c) [symbolic]
// CHECK:STDOUT:   %assoc0.d73: %SubWith.assoc_type.760 = assoc_entity element0, imports.%Core.import_ref.afe [symbolic]
// CHECK:STDOUT:   %assoc1.972: %SubWith.assoc_type.760 = assoc_entity element1, imports.%Core.import_ref.bc0 [symbolic]
// CHECK:STDOUT:   %require_complete.250: <witness> = require_complete_type %facet_type.241 [symbolic]
// CHECK:STDOUT:   %.Self.as_type.2cd: type = facet_access_type %.Self.a38 [symbolic]
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.cb4: <witness> = lookup_impl_witness %.Self.a38, @SubWith.1, @SubWith.1(%U.as_type.04c) [symbolic]
// CHECK:STDOUT:   %impl.elem0.b99: type = impl_witness_access %SubWith.lookup_impl_witness.cb4, element0 [symbolic]
// CHECK:STDOUT:   %facet_type.a29: type = facet_type <@Destroy & @SubWith.1, @SubWith.1(%U.as_type.04c) where %impl.elem0.b99 = %Result.as_type> [symbolic]
// CHECK:STDOUT:   %pattern_type.095: type = pattern_type %facet_type.a29 [symbolic]
// CHECK:STDOUT:   %T.813: %facet_type.a29 = symbolic_binding T, 2 [symbolic]
// CHECK:STDOUT:   %T.as_type.803: type = facet_access_type %T.813 [symbolic]
// CHECK:STDOUT:   %pattern_type.d56: type = pattern_type %T.as_type.803 [symbolic]
// CHECK:STDOUT:   %require_complete.a67: <witness> = require_complete_type %SubWith.type.2e8 [symbolic]
// CHECK:STDOUT:   %assoc1.076: %SubWith.assoc_type.02f = assoc_entity element1, imports.%Core.import_ref.efd [symbolic]
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.ebb: <witness> = lookup_impl_witness %T.813, @SubWith.1, @SubWith.1(%U.as_type.04c) [symbolic]
// CHECK:STDOUT:   %SubWith.facet.e7a: %SubWith.type.2e8 = facet_value %T.as_type.803, (%SubWith.lookup_impl_witness.ebb) [symbolic]
// CHECK:STDOUT:   %SubWith.WithSelf.Op.type.0dc: type = fn_type @SubWith.WithSelf.Op, @SubWith.WithSelf(%U.as_type.04c, %SubWith.facet.e7a) [symbolic]
// CHECK:STDOUT:   %.4bc: type = fn_type_with_self_type %SubWith.WithSelf.Op.type.0dc, %SubWith.facet.e7a [symbolic]
// CHECK:STDOUT:   %impl.elem1.a3d: %.4bc = impl_witness_access %SubWith.lookup_impl_witness.ebb, element1 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.542: <specific function> = specific_impl_function %impl.elem1.a3d, @SubWith.WithSelf.Op(%U.as_type.04c, %SubWith.facet.e7a) [symbolic]
// CHECK:STDOUT:   %MulWith.type.fe5: type = facet_type <@MulWith.1, @MulWith.1(%Other)> [symbolic]
// CHECK:STDOUT:   %Self.11d: %MulWith.type.fe5 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %MulWith.assoc_type.ae7: type = assoc_entity_type @MulWith.1, @MulWith.1(%Other) [symbolic]
// CHECK:STDOUT:   %MulWith.WithSelf.Op.type.896: type = fn_type @MulWith.WithSelf.Op, @MulWith.WithSelf(%Other, %Self.11d) [symbolic]
// CHECK:STDOUT:   %MulWith.WithSelf.Op.9fb: %MulWith.WithSelf.Op.type.896 = struct_value () [symbolic]
// CHECK:STDOUT:   %MulWith.type.293: type = facet_type <@MulWith.1, @MulWith.1(%U.as_type.04c)> [symbolic]
// CHECK:STDOUT:   %facet_type.3d2: type = facet_type <@Destroy & @MulWith.1, @MulWith.1(%U.as_type.04c)> [symbolic]
// CHECK:STDOUT:   %.Self.b95: %facet_type.3d2 = symbolic_binding .Self [symbolic]
// CHECK:STDOUT:   %MulWith.assoc_type.8c9: type = assoc_entity_type @MulWith.1, @MulWith.1(%U.as_type.04c) [symbolic]
// CHECK:STDOUT:   %assoc0.007: %MulWith.assoc_type.8c9 = assoc_entity element0, imports.%Core.import_ref.648 [symbolic]
// CHECK:STDOUT:   %assoc1.8e8: %MulWith.assoc_type.8c9 = assoc_entity element1, imports.%Core.import_ref.32c [symbolic]
// CHECK:STDOUT:   %require_complete.ac4: <witness> = require_complete_type %facet_type.3d2 [symbolic]
// CHECK:STDOUT:   %.Self.as_type.3e1: type = facet_access_type %.Self.b95 [symbolic]
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.ca1: <witness> = lookup_impl_witness %.Self.b95, @MulWith.1, @MulWith.1(%U.as_type.04c) [symbolic]
// CHECK:STDOUT:   %impl.elem0.975: type = impl_witness_access %MulWith.lookup_impl_witness.ca1, element0 [symbolic]
// CHECK:STDOUT:   %facet_type.ec5: type = facet_type <@Destroy & @MulWith.1, @MulWith.1(%U.as_type.04c) where %impl.elem0.975 = %Result.as_type> [symbolic]
// CHECK:STDOUT:   %pattern_type.91b: type = pattern_type %facet_type.ec5 [symbolic]
// CHECK:STDOUT:   %T.c84: %facet_type.ec5 = symbolic_binding T, 2 [symbolic]
// CHECK:STDOUT:   %T.as_type.0c1: type = facet_access_type %T.c84 [symbolic]
// CHECK:STDOUT:   %pattern_type.ac7: type = pattern_type %T.as_type.0c1 [symbolic]
// CHECK:STDOUT:   %require_complete.50e: <witness> = require_complete_type %MulWith.type.293 [symbolic]
// CHECK:STDOUT:   %assoc1.d52: %MulWith.assoc_type.ae7 = assoc_entity element1, imports.%Core.import_ref.d54 [symbolic]
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.38f: <witness> = lookup_impl_witness %T.c84, @MulWith.1, @MulWith.1(%U.as_type.04c) [symbolic]
// CHECK:STDOUT:   %MulWith.facet.69f: %MulWith.type.293 = facet_value %T.as_type.0c1, (%MulWith.lookup_impl_witness.38f) [symbolic]
// CHECK:STDOUT:   %MulWith.WithSelf.Op.type.929: type = fn_type @MulWith.WithSelf.Op, @MulWith.WithSelf(%U.as_type.04c, %MulWith.facet.69f) [symbolic]
// CHECK:STDOUT:   %.b06: type = fn_type_with_self_type %MulWith.WithSelf.Op.type.929, %MulWith.facet.69f [symbolic]
// CHECK:STDOUT:   %impl.elem1.a81: %.b06 = impl_witness_access %MulWith.lookup_impl_witness.38f, element1 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.551: <specific function> = specific_impl_function %impl.elem1.a81, @MulWith.WithSelf.Op(%U.as_type.04c, %MulWith.facet.69f) [symbolic]
// CHECK:STDOUT:   %DivWith.type.74f: type = facet_type <@DivWith.1, @DivWith.1(%Other)> [symbolic]
// CHECK:STDOUT:   %Self.c0b: %DivWith.type.74f = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %DivWith.assoc_type.115: type = assoc_entity_type @DivWith.1, @DivWith.1(%Other) [symbolic]
// CHECK:STDOUT:   %DivWith.WithSelf.Op.type.993: type = fn_type @DivWith.WithSelf.Op, @DivWith.WithSelf(%Other, %Self.c0b) [symbolic]
// CHECK:STDOUT:   %DivWith.WithSelf.Op.e1e: %DivWith.WithSelf.Op.type.993 = struct_value () [symbolic]
// CHECK:STDOUT:   %DivWith.type.25a: type = facet_type <@DivWith.1, @DivWith.1(%U.as_type.04c)> [symbolic]
// CHECK:STDOUT:   %facet_type.0d2: type = facet_type <@Destroy & @DivWith.1, @DivWith.1(%U.as_type.04c)> [symbolic]
// CHECK:STDOUT:   %.Self.db2: %facet_type.0d2 = symbolic_binding .Self [symbolic]
// CHECK:STDOUT:   %DivWith.assoc_type.b3b: type = assoc_entity_type @DivWith.1, @DivWith.1(%U.as_type.04c) [symbolic]
// CHECK:STDOUT:   %assoc0.f1d: %DivWith.assoc_type.b3b = assoc_entity element0, imports.%Core.import_ref.db1 [symbolic]
// CHECK:STDOUT:   %assoc1.d84: %DivWith.assoc_type.b3b = assoc_entity element1, imports.%Core.import_ref.7dc [symbolic]
// CHECK:STDOUT:   %require_complete.770: <witness> = require_complete_type %facet_type.0d2 [symbolic]
// CHECK:STDOUT:   %.Self.as_type.67d: type = facet_access_type %.Self.db2 [symbolic]
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.a3e: <witness> = lookup_impl_witness %.Self.db2, @DivWith.1, @DivWith.1(%U.as_type.04c) [symbolic]
// CHECK:STDOUT:   %impl.elem0.f72: type = impl_witness_access %DivWith.lookup_impl_witness.a3e, element0 [symbolic]
// CHECK:STDOUT:   %facet_type.b8a: type = facet_type <@Destroy & @DivWith.1, @DivWith.1(%U.as_type.04c) where %impl.elem0.f72 = %Result.as_type> [symbolic]
// CHECK:STDOUT:   %pattern_type.e40: type = pattern_type %facet_type.b8a [symbolic]
// CHECK:STDOUT:   %T.81b: %facet_type.b8a = symbolic_binding T, 2 [symbolic]
// CHECK:STDOUT:   %T.as_type.2d1: type = facet_access_type %T.81b [symbolic]
// CHECK:STDOUT:   %pattern_type.4c5: type = pattern_type %T.as_type.2d1 [symbolic]
// CHECK:STDOUT:   %require_complete.2bc: <witness> = require_complete_type %DivWith.type.25a [symbolic]
// CHECK:STDOUT:   %assoc1.868: %DivWith.assoc_type.115 = assoc_entity element1, imports.%Core.import_ref.33d [symbolic]
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.b03: <witness> = lookup_impl_witness %T.81b, @DivWith.1, @DivWith.1(%U.as_type.04c) [symbolic]
// CHECK:STDOUT:   %DivWith.facet.924: %DivWith.type.25a = facet_value %T.as_type.2d1, (%DivWith.lookup_impl_witness.b03) [symbolic]
// CHECK:STDOUT:   %DivWith.WithSelf.Op.type.ddd: type = fn_type @DivWith.WithSelf.Op, @DivWith.WithSelf(%U.as_type.04c, %DivWith.facet.924) [symbolic]
// CHECK:STDOUT:   %.4f8: type = fn_type_with_self_type %DivWith.WithSelf.Op.type.ddd, %DivWith.facet.924 [symbolic]
// CHECK:STDOUT:   %impl.elem1.3cc: %.4f8 = impl_witness_access %DivWith.lookup_impl_witness.b03, element1 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.7c7: <specific function> = specific_impl_function %impl.elem1.3cc, @DivWith.WithSelf.Op(%U.as_type.04c, %DivWith.facet.924) [symbolic]
// CHECK:STDOUT:   %ModWith.type.4d3: type = facet_type <@ModWith.1, @ModWith.1(%Other)> [symbolic]
// CHECK:STDOUT:   %Self.852: %ModWith.type.4d3 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %ModWith.assoc_type.8a9: type = assoc_entity_type @ModWith.1, @ModWith.1(%Other) [symbolic]
// CHECK:STDOUT:   %ModWith.WithSelf.Op.type.c98: type = fn_type @ModWith.WithSelf.Op, @ModWith.WithSelf(%Other, %Self.852) [symbolic]
// CHECK:STDOUT:   %ModWith.WithSelf.Op.ac5: %ModWith.WithSelf.Op.type.c98 = struct_value () [symbolic]
// CHECK:STDOUT:   %ModWith.type.3cd: type = facet_type <@ModWith.1, @ModWith.1(%U.as_type.04c)> [symbolic]
// CHECK:STDOUT:   %facet_type.8d7: type = facet_type <@Destroy & @ModWith.1, @ModWith.1(%U.as_type.04c)> [symbolic]
// CHECK:STDOUT:   %.Self.b9d: %facet_type.8d7 = symbolic_binding .Self [symbolic]
// CHECK:STDOUT:   %ModWith.assoc_type.84f: type = assoc_entity_type @ModWith.1, @ModWith.1(%U.as_type.04c) [symbolic]
// CHECK:STDOUT:   %assoc0.bf7: %ModWith.assoc_type.84f = assoc_entity element0, imports.%Core.import_ref.78b [symbolic]
// CHECK:STDOUT:   %assoc1.ea2: %ModWith.assoc_type.84f = assoc_entity element1, imports.%Core.import_ref.9e2 [symbolic]
// CHECK:STDOUT:   %require_complete.ba5: <witness> = require_complete_type %facet_type.8d7 [symbolic]
// CHECK:STDOUT:   %.Self.as_type.25e: type = facet_access_type %.Self.b9d [symbolic]
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.271: <witness> = lookup_impl_witness %.Self.b9d, @ModWith.1, @ModWith.1(%U.as_type.04c) [symbolic]
// CHECK:STDOUT:   %impl.elem0.4a9: type = impl_witness_access %ModWith.lookup_impl_witness.271, element0 [symbolic]
// CHECK:STDOUT:   %facet_type.c9c: type = facet_type <@Destroy & @ModWith.1, @ModWith.1(%U.as_type.04c) where %impl.elem0.4a9 = %Result.as_type> [symbolic]
// CHECK:STDOUT:   %pattern_type.cf5: type = pattern_type %facet_type.c9c [symbolic]
// CHECK:STDOUT:   %T.ddd: %facet_type.c9c = symbolic_binding T, 2 [symbolic]
// CHECK:STDOUT:   %T.as_type.2fa: type = facet_access_type %T.ddd [symbolic]
// CHECK:STDOUT:   %pattern_type.e09: type = pattern_type %T.as_type.2fa [symbolic]
// CHECK:STDOUT:   %require_complete.ccd: <witness> = require_complete_type %ModWith.type.3cd [symbolic]
// CHECK:STDOUT:   %assoc1.7ba: %ModWith.assoc_type.8a9 = assoc_entity element1, imports.%Core.import_ref.3af [symbolic]
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.962: <witness> = lookup_impl_witness %T.ddd, @ModWith.1, @ModWith.1(%U.as_type.04c) [symbolic]
// CHECK:STDOUT:   %ModWith.facet.bc2: %ModWith.type.3cd = facet_value %T.as_type.2fa, (%ModWith.lookup_impl_witness.962) [symbolic]
// CHECK:STDOUT:   %ModWith.WithSelf.Op.type.fed: type = fn_type @ModWith.WithSelf.Op, @ModWith.WithSelf(%U.as_type.04c, %ModWith.facet.bc2) [symbolic]
// CHECK:STDOUT:   %.ed2: type = fn_type_with_self_type %ModWith.WithSelf.Op.type.fed, %ModWith.facet.bc2 [symbolic]
// CHECK:STDOUT:   %impl.elem1.260: %.ed2 = impl_witness_access %ModWith.lookup_impl_witness.962, element1 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.4e2: <specific function> = specific_impl_function %impl.elem1.260, @ModWith.WithSelf.Op(%U.as_type.04c, %ModWith.facet.bc2) [symbolic]
// CHECK:STDOUT:   %Int16: type = class_type @Int16 [concrete]
// CHECK:STDOUT:   %pattern_type.b06: type = pattern_type %Int16 [concrete]
// CHECK:STDOUT:   %Int32: type = class_type @Int32 [concrete]
// CHECK:STDOUT:   %pattern_type.4b8: type = pattern_type %Int32 [concrete]
// CHECK:STDOUT:   %Int64: type = class_type @Int64 [concrete]
// CHECK:STDOUT:   %pattern_type.c76: type = pattern_type %Int64 [concrete]
// CHECK:STDOUT:   %int_16: Core.IntLiteral = int_value 16 [concrete]
// CHECK:STDOUT:   %i16: type = class_type @Int, @Int(%int_16) [concrete]
// CHECK:STDOUT:   %pattern_type.0e3: type = pattern_type %i16 [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %i16.builtin: type = int_type signed, %int_16 [concrete]
// CHECK:STDOUT:   %complete_type.2da: <witness> = complete_type_witness %i16.builtin [concrete]
// CHECK:STDOUT:   %Int16.Op.type.da5d51.1: type = fn_type @Int16.Op.1 [concrete]
// CHECK:STDOUT:   %Int16.Op.7ce432.1: %Int16.Op.type.da5d51.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.ecd0a5.1: <witness> = custom_witness (%Int16.Op.7ce432.1), @Destroy [concrete]
// CHECK:STDOUT:   %Destroy.facet.551: %Destroy.type = facet_value %Int16, (%custom_witness.ecd0a5.1) [concrete]
// CHECK:STDOUT:   %facet_type.b25: type = facet_type <@Destroy & @AddWith.1, @AddWith.1(%Int16)> [concrete]
// CHECK:STDOUT:   %.Self.cee: %facet_type.b25 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.ef3: <witness> = lookup_impl_witness %.Self.cee, @AddWith.1, @AddWith.1(%Int16) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.1ef: type = impl_witness_access %AddWith.lookup_impl_witness.ef3, element0 [symbolic_self]
// CHECK:STDOUT:   %facet_type.eb1: type = facet_type <@Destroy & @AddWith.1, @AddWith.1(%Int16) where %impl.elem0.1ef = %Int16> [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.3f5f78.2: type = fn_type @Destroy.Op.loc176_34.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.f0ba5f.2: %Destroy.Op.type.3f5f78.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.1367bd.2: <witness> = custom_witness (%Destroy.Op.f0ba5f.2), @Destroy [concrete]
// CHECK:STDOUT:   %AddWith.type.41f: type = facet_type <@AddWith.1, @AddWith.1(%Int16)> [concrete]
// CHECK:STDOUT:   %AddWith.assoc_type.5ee: type = assoc_entity_type @AddWith.1, @AddWith.1(%Int16) [concrete]
// CHECK:STDOUT:   %assoc0.95b: %AddWith.assoc_type.5ee = assoc_entity element0, imports.%Core.import_ref.3ad [concrete]
// CHECK:STDOUT:   %assoc1.41e: %AddWith.assoc_type.5ee = assoc_entity element1, imports.%Core.import_ref.2b2 [concrete]
// CHECK:STDOUT:   %Op.type.175699.1: type = fn_type @Op.1 [concrete]
// CHECK:STDOUT:   %Op.848e8c.1: %Op.type.175699.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.facet.0e0: %Destroy.type = facet_value %i16, (%custom_witness.1367bd.2) [concrete]
// CHECK:STDOUT:   %custom_witness.069: <witness> = custom_witness (%Int16, %Op.848e8c.1), @AddWith.1, @AddWith.1(%Int16) [concrete]
// CHECK:STDOUT:   %facet_value.227e: %facet_type.eb1 = facet_value %i16, (%custom_witness.1367bd.2, %custom_witness.069) [concrete]
// CHECK:STDOUT:   %complete_type.08f: <witness> = complete_type_witness %facet_type.b25 [concrete]
// CHECK:STDOUT:   %.Self.as_type.38a: type = facet_access_type %.Self.cee [symbolic_self]
// CHECK:STDOUT:   %pattern_type.748: type = pattern_type %facet_type.eb1 [concrete]
// CHECK:STDOUT:   %facet_type.5bd: type = facet_type <@Destroy & @SubWith.1, @SubWith.1(%Int16)> [concrete]
// CHECK:STDOUT:   %.Self.0bf: %facet_type.5bd = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.9d6: <witness> = lookup_impl_witness %.Self.0bf, @SubWith.1, @SubWith.1(%Int16) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.214: type = impl_witness_access %SubWith.lookup_impl_witness.9d6, element0 [symbolic_self]
// CHECK:STDOUT:   %facet_type.7bc: type = facet_type <@Destroy & @SubWith.1, @SubWith.1(%Int16) where %impl.elem0.214 = %Int16> [concrete]
// CHECK:STDOUT:   %SubWith.type.f7a: type = facet_type <@SubWith.1, @SubWith.1(%Int16)> [concrete]
// CHECK:STDOUT:   %SubWith.assoc_type.e4b: type = assoc_entity_type @SubWith.1, @SubWith.1(%Int16) [concrete]
// CHECK:STDOUT:   %assoc0.f15: %SubWith.assoc_type.e4b = assoc_entity element0, imports.%Core.import_ref.afe [concrete]
// CHECK:STDOUT:   %assoc1.8a0: %SubWith.assoc_type.e4b = assoc_entity element1, imports.%Core.import_ref.bc0 [concrete]
// CHECK:STDOUT:   %Op.type.175699.2: type = fn_type @Op.2 [concrete]
// CHECK:STDOUT:   %Op.848e8c.2: %Op.type.175699.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.2c8: <witness> = custom_witness (%Int16, %Op.848e8c.2), @SubWith.1, @SubWith.1(%Int16) [concrete]
// CHECK:STDOUT:   %facet_value.d1c: %facet_type.7bc = facet_value %i16, (%custom_witness.1367bd.2, %custom_witness.2c8) [concrete]
// CHECK:STDOUT:   %complete_type.4b4: <witness> = complete_type_witness %facet_type.5bd [concrete]
// CHECK:STDOUT:   %.Self.as_type.752: type = facet_access_type %.Self.0bf [symbolic_self]
// CHECK:STDOUT:   %pattern_type.ca1: type = pattern_type %facet_type.7bc [concrete]
// CHECK:STDOUT:   %facet_type.b8b: type = facet_type <@Destroy & @MulWith.1, @MulWith.1(%Int16)> [concrete]
// CHECK:STDOUT:   %.Self.117: %facet_type.b8b = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.485: <witness> = lookup_impl_witness %.Self.117, @MulWith.1, @MulWith.1(%Int16) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.851: type = impl_witness_access %MulWith.lookup_impl_witness.485, element0 [symbolic_self]
// CHECK:STDOUT:   %facet_type.166: type = facet_type <@Destroy & @MulWith.1, @MulWith.1(%Int16) where %impl.elem0.851 = %Int16> [concrete]
// CHECK:STDOUT:   %MulWith.type.ebe: type = facet_type <@MulWith.1, @MulWith.1(%Int16)> [concrete]
// CHECK:STDOUT:   %MulWith.assoc_type.286: type = assoc_entity_type @MulWith.1, @MulWith.1(%Int16) [concrete]
// CHECK:STDOUT:   %assoc0.52c: %MulWith.assoc_type.286 = assoc_entity element0, imports.%Core.import_ref.648 [concrete]
// CHECK:STDOUT:   %assoc1.0bf: %MulWith.assoc_type.286 = assoc_entity element1, imports.%Core.import_ref.32c [concrete]
// CHECK:STDOUT:   %Op.type.175699.3: type = fn_type @Op.3 [concrete]
// CHECK:STDOUT:   %Op.848e8c.3: %Op.type.175699.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.23bf: <witness> = custom_witness (%Int16, %Op.848e8c.3), @MulWith.1, @MulWith.1(%Int16) [concrete]
// CHECK:STDOUT:   %facet_value.b66: %facet_type.166 = facet_value %i16, (%custom_witness.1367bd.2, %custom_witness.23bf) [concrete]
// CHECK:STDOUT:   %complete_type.25f: <witness> = complete_type_witness %facet_type.b8b [concrete]
// CHECK:STDOUT:   %.Self.as_type.ad8: type = facet_access_type %.Self.117 [symbolic_self]
// CHECK:STDOUT:   %pattern_type.380: type = pattern_type %facet_type.166 [concrete]
// CHECK:STDOUT:   %facet_type.078: type = facet_type <@Destroy & @DivWith.1, @DivWith.1(%Int16)> [concrete]
// CHECK:STDOUT:   %.Self.f7a: %facet_type.078 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.922: <witness> = lookup_impl_witness %.Self.f7a, @DivWith.1, @DivWith.1(%Int16) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.1ae: type = impl_witness_access %DivWith.lookup_impl_witness.922, element0 [symbolic_self]
// CHECK:STDOUT:   %facet_type.9a8: type = facet_type <@Destroy & @DivWith.1, @DivWith.1(%Int16) where %impl.elem0.1ae = %Int16> [concrete]
// CHECK:STDOUT:   %DivWith.type.996: type = facet_type <@DivWith.1, @DivWith.1(%Int16)> [concrete]
// CHECK:STDOUT:   %DivWith.assoc_type.ae8: type = assoc_entity_type @DivWith.1, @DivWith.1(%Int16) [concrete]
// CHECK:STDOUT:   %assoc0.7b6: %DivWith.assoc_type.ae8 = assoc_entity element0, imports.%Core.import_ref.db1 [concrete]
// CHECK:STDOUT:   %assoc1.f9f: %DivWith.assoc_type.ae8 = assoc_entity element1, imports.%Core.import_ref.7dc [concrete]
// CHECK:STDOUT:   %Op.type.175699.4: type = fn_type @Op.4 [concrete]
// CHECK:STDOUT:   %Op.848e8c.4: %Op.type.175699.4 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.ff6: <witness> = custom_witness (%Int16, %Op.848e8c.4), @DivWith.1, @DivWith.1(%Int16) [concrete]
// CHECK:STDOUT:   %facet_value.f9d: %facet_type.9a8 = facet_value %i16, (%custom_witness.1367bd.2, %custom_witness.ff6) [concrete]
// CHECK:STDOUT:   %complete_type.4f9: <witness> = complete_type_witness %facet_type.078 [concrete]
// CHECK:STDOUT:   %.Self.as_type.91d: type = facet_access_type %.Self.f7a [symbolic_self]
// CHECK:STDOUT:   %pattern_type.515: type = pattern_type %facet_type.9a8 [concrete]
// CHECK:STDOUT:   %facet_type.dc0: type = facet_type <@Destroy & @ModWith.1, @ModWith.1(%Int16)> [concrete]
// CHECK:STDOUT:   %.Self.c9e: %facet_type.dc0 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.9ea: <witness> = lookup_impl_witness %.Self.c9e, @ModWith.1, @ModWith.1(%Int16) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.7c3: type = impl_witness_access %ModWith.lookup_impl_witness.9ea, element0 [symbolic_self]
// CHECK:STDOUT:   %facet_type.3d1: type = facet_type <@Destroy & @ModWith.1, @ModWith.1(%Int16) where %impl.elem0.7c3 = %Int16> [concrete]
// CHECK:STDOUT:   %ModWith.type.d4b: type = facet_type <@ModWith.1, @ModWith.1(%Int16)> [concrete]
// CHECK:STDOUT:   %ModWith.assoc_type.8c2: type = assoc_entity_type @ModWith.1, @ModWith.1(%Int16) [concrete]
// CHECK:STDOUT:   %assoc0.262: %ModWith.assoc_type.8c2 = assoc_entity element0, imports.%Core.import_ref.78b [concrete]
// CHECK:STDOUT:   %assoc1.63a: %ModWith.assoc_type.8c2 = assoc_entity element1, imports.%Core.import_ref.9e2 [concrete]
// CHECK:STDOUT:   %Op.type.175699.5: type = fn_type @Op.5 [concrete]
// CHECK:STDOUT:   %Op.848e8c.5: %Op.type.175699.5 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.74e: <witness> = custom_witness (%Int16, %Op.848e8c.5), @ModWith.1, @ModWith.1(%Int16) [concrete]
// CHECK:STDOUT:   %facet_value.eed: %facet_type.3d1 = facet_value %i16, (%custom_witness.1367bd.2, %custom_witness.74e) [concrete]
// CHECK:STDOUT:   %complete_type.10f: <witness> = complete_type_witness %facet_type.dc0 [concrete]
// CHECK:STDOUT:   %.Self.as_type.fc8: type = facet_access_type %.Self.c9e [symbolic_self]
// CHECK:STDOUT:   %pattern_type.9dd: type = pattern_type %facet_type.3d1 [concrete]
// CHECK:STDOUT:   %facet_type.d65: type = facet_type <@Destroy & @AddWith.1, @AddWith.1(%i16)> [concrete]
// CHECK:STDOUT:   %.Self.0d7: %facet_type.d65 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.750: <witness> = lookup_impl_witness %.Self.0d7, @AddWith.1, @AddWith.1(%i16) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.ed3: type = impl_witness_access %AddWith.lookup_impl_witness.750, element0 [symbolic_self]
// CHECK:STDOUT:   %facet_type.9d0: type = facet_type <@Destroy & @AddWith.1, @AddWith.1(%i16) where %impl.elem0.ed3 = %Int16> [concrete]
// CHECK:STDOUT:   %AddWith.type.c9a: type = facet_type <@AddWith.1, @AddWith.1(%i16)> [concrete]
// CHECK:STDOUT:   %AddWith.assoc_type.67c: type = assoc_entity_type @AddWith.1, @AddWith.1(%i16) [concrete]
// CHECK:STDOUT:   %assoc0.91b: %AddWith.assoc_type.67c = assoc_entity element0, imports.%Core.import_ref.3ad [concrete]
// CHECK:STDOUT:   %assoc1.e96: %AddWith.assoc_type.67c = assoc_entity element1, imports.%Core.import_ref.2b2 [concrete]
// CHECK:STDOUT:   %Int16.Op.type.da5d51.3: type = fn_type @Int16.Op.3 [concrete]
// CHECK:STDOUT:   %Int16.Op.7ce432.3: %Int16.Op.type.da5d51.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.067: <witness> = custom_witness (%Int16, %Int16.Op.7ce432.3), @AddWith.1, @AddWith.1(%i16) [concrete]
// CHECK:STDOUT:   %facet_value.511: %facet_type.9d0 = facet_value %Int16, (%custom_witness.ecd0a5.1, %custom_witness.067) [concrete]
// CHECK:STDOUT:   %complete_type.aae1: <witness> = complete_type_witness %facet_type.d65 [concrete]
// CHECK:STDOUT:   %.Self.as_type.578: type = facet_access_type %.Self.0d7 [symbolic_self]
// CHECK:STDOUT:   %pattern_type.366: type = pattern_type %facet_type.9d0 [concrete]
// CHECK:STDOUT:   %facet_type.21e: type = facet_type <@Destroy & @SubWith.1, @SubWith.1(%i16)> [concrete]
// CHECK:STDOUT:   %.Self.8c7: %facet_type.21e = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.2bb: <witness> = lookup_impl_witness %.Self.8c7, @SubWith.1, @SubWith.1(%i16) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.759: type = impl_witness_access %SubWith.lookup_impl_witness.2bb, element0 [symbolic_self]
// CHECK:STDOUT:   %facet_type.54a: type = facet_type <@Destroy & @SubWith.1, @SubWith.1(%i16) where %impl.elem0.759 = %Int16> [concrete]
// CHECK:STDOUT:   %SubWith.type.191: type = facet_type <@SubWith.1, @SubWith.1(%i16)> [concrete]
// CHECK:STDOUT:   %SubWith.assoc_type.b4a: type = assoc_entity_type @SubWith.1, @SubWith.1(%i16) [concrete]
// CHECK:STDOUT:   %assoc0.ecf: %SubWith.assoc_type.b4a = assoc_entity element0, imports.%Core.import_ref.afe [concrete]
// CHECK:STDOUT:   %assoc1.708: %SubWith.assoc_type.b4a = assoc_entity element1, imports.%Core.import_ref.bc0 [concrete]
// CHECK:STDOUT:   %Int16.Op.type.da5d51.4: type = fn_type @Int16.Op.4 [concrete]
// CHECK:STDOUT:   %Int16.Op.7ce432.4: %Int16.Op.type.da5d51.4 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.1e0: <witness> = custom_witness (%Int16, %Int16.Op.7ce432.4), @SubWith.1, @SubWith.1(%i16) [concrete]
// CHECK:STDOUT:   %facet_value.70a: %facet_type.54a = facet_value %Int16, (%custom_witness.ecd0a5.1, %custom_witness.1e0) [concrete]
// CHECK:STDOUT:   %complete_type.3a7: <witness> = complete_type_witness %facet_type.21e [concrete]
// CHECK:STDOUT:   %.Self.as_type.ef8: type = facet_access_type %.Self.8c7 [symbolic_self]
// CHECK:STDOUT:   %pattern_type.a0a: type = pattern_type %facet_type.54a [concrete]
// CHECK:STDOUT:   %facet_type.fd8: type = facet_type <@Destroy & @MulWith.1, @MulWith.1(%i16)> [concrete]
// CHECK:STDOUT:   %.Self.615: %facet_type.fd8 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.b02: <witness> = lookup_impl_witness %.Self.615, @MulWith.1, @MulWith.1(%i16) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.d68: type = impl_witness_access %MulWith.lookup_impl_witness.b02, element0 [symbolic_self]
// CHECK:STDOUT:   %facet_type.0ff: type = facet_type <@Destroy & @MulWith.1, @MulWith.1(%i16) where %impl.elem0.d68 = %Int16> [concrete]
// CHECK:STDOUT:   %MulWith.type.df1: type = facet_type <@MulWith.1, @MulWith.1(%i16)> [concrete]
// CHECK:STDOUT:   %MulWith.assoc_type.693: type = assoc_entity_type @MulWith.1, @MulWith.1(%i16) [concrete]
// CHECK:STDOUT:   %assoc0.037: %MulWith.assoc_type.693 = assoc_entity element0, imports.%Core.import_ref.648 [concrete]
// CHECK:STDOUT:   %assoc1.7fb: %MulWith.assoc_type.693 = assoc_entity element1, imports.%Core.import_ref.32c [concrete]
// CHECK:STDOUT:   %Int16.Op.type.da5d51.5: type = fn_type @Int16.Op.5 [concrete]
// CHECK:STDOUT:   %Int16.Op.7ce432.5: %Int16.Op.type.da5d51.5 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.631: <witness> = custom_witness (%Int16, %Int16.Op.7ce432.5), @MulWith.1, @MulWith.1(%i16) [concrete]
// CHECK:STDOUT:   %facet_value.60b: %facet_type.0ff = facet_value %Int16, (%custom_witness.ecd0a5.1, %custom_witness.631) [concrete]
// CHECK:STDOUT:   %complete_type.050: <witness> = complete_type_witness %facet_type.fd8 [concrete]
// CHECK:STDOUT:   %.Self.as_type.434: type = facet_access_type %.Self.615 [symbolic_self]
// CHECK:STDOUT:   %pattern_type.370: type = pattern_type %facet_type.0ff [concrete]
// CHECK:STDOUT:   %facet_type.e1e: type = facet_type <@Destroy & @DivWith.1, @DivWith.1(%i16)> [concrete]
// CHECK:STDOUT:   %.Self.319: %facet_type.e1e = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.493: <witness> = lookup_impl_witness %.Self.319, @DivWith.1, @DivWith.1(%i16) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.ed9: type = impl_witness_access %DivWith.lookup_impl_witness.493, element0 [symbolic_self]
// CHECK:STDOUT:   %facet_type.721: type = facet_type <@Destroy & @DivWith.1, @DivWith.1(%i16) where %impl.elem0.ed9 = %Int16> [concrete]
// CHECK:STDOUT:   %DivWith.type.957: type = facet_type <@DivWith.1, @DivWith.1(%i16)> [concrete]
// CHECK:STDOUT:   %DivWith.assoc_type.c56: type = assoc_entity_type @DivWith.1, @DivWith.1(%i16) [concrete]
// CHECK:STDOUT:   %assoc0.e06: %DivWith.assoc_type.c56 = assoc_entity element0, imports.%Core.import_ref.db1 [concrete]
// CHECK:STDOUT:   %assoc1.59d: %DivWith.assoc_type.c56 = assoc_entity element1, imports.%Core.import_ref.7dc [concrete]
// CHECK:STDOUT:   %Int16.Op.type.da5d51.6: type = fn_type @Int16.Op.6 [concrete]
// CHECK:STDOUT:   %Int16.Op.7ce432.6: %Int16.Op.type.da5d51.6 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.64e: <witness> = custom_witness (%Int16, %Int16.Op.7ce432.6), @DivWith.1, @DivWith.1(%i16) [concrete]
// CHECK:STDOUT:   %facet_value.4bd: %facet_type.721 = facet_value %Int16, (%custom_witness.ecd0a5.1, %custom_witness.64e) [concrete]
// CHECK:STDOUT:   %complete_type.509: <witness> = complete_type_witness %facet_type.e1e [concrete]
// CHECK:STDOUT:   %.Self.as_type.619: type = facet_access_type %.Self.319 [symbolic_self]
// CHECK:STDOUT:   %pattern_type.2b6: type = pattern_type %facet_type.721 [concrete]
// CHECK:STDOUT:   %facet_type.dca: type = facet_type <@Destroy & @ModWith.1, @ModWith.1(%i16)> [concrete]
// CHECK:STDOUT:   %.Self.041: %facet_type.dca = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.dfa: <witness> = lookup_impl_witness %.Self.041, @ModWith.1, @ModWith.1(%i16) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.025: type = impl_witness_access %ModWith.lookup_impl_witness.dfa, element0 [symbolic_self]
// CHECK:STDOUT:   %facet_type.6dc: type = facet_type <@Destroy & @ModWith.1, @ModWith.1(%i16) where %impl.elem0.025 = %Int16> [concrete]
// CHECK:STDOUT:   %ModWith.type.0ea: type = facet_type <@ModWith.1, @ModWith.1(%i16)> [concrete]
// CHECK:STDOUT:   %ModWith.assoc_type.327: type = assoc_entity_type @ModWith.1, @ModWith.1(%i16) [concrete]
// CHECK:STDOUT:   %assoc0.2be: %ModWith.assoc_type.327 = assoc_entity element0, imports.%Core.import_ref.78b [concrete]
// CHECK:STDOUT:   %assoc1.423: %ModWith.assoc_type.327 = assoc_entity element1, imports.%Core.import_ref.9e2 [concrete]
// CHECK:STDOUT:   %Int16.Op.type.da5d51.7: type = fn_type @Int16.Op.7 [concrete]
// CHECK:STDOUT:   %Int16.Op.7ce432.7: %Int16.Op.type.da5d51.7 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.394: <witness> = custom_witness (%Int16, %Int16.Op.7ce432.7), @ModWith.1, @ModWith.1(%i16) [concrete]
// CHECK:STDOUT:   %facet_value.dd2: %facet_type.6dc = facet_value %Int16, (%custom_witness.ecd0a5.1, %custom_witness.394) [concrete]
// CHECK:STDOUT:   %complete_type.65a: <witness> = complete_type_witness %facet_type.dca [concrete]
// CHECK:STDOUT:   %.Self.as_type.535: type = facet_access_type %.Self.041 [symbolic_self]
// CHECK:STDOUT:   %pattern_type.49d: type = pattern_type %facet_type.6dc [concrete]
// CHECK:STDOUT:   %Int32.Op.type.da5d51.1: type = fn_type @Int32.Op.1 [concrete]
// CHECK:STDOUT:   %Int32.Op.7ce432.1: %Int32.Op.type.da5d51.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.ecd0a5.2: <witness> = custom_witness (%Int32.Op.7ce432.1), @Destroy [concrete]
// CHECK:STDOUT:   %Destroy.facet.184: %Destroy.type = facet_value %Int32, (%custom_witness.ecd0a5.2) [concrete]
// CHECK:STDOUT:   %facet_type.35e: type = facet_type <@Destroy & @AddWith.1, @AddWith.1(%Int32)> [concrete]
// CHECK:STDOUT:   %.Self.ea4: %facet_type.35e = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.587: <witness> = lookup_impl_witness %.Self.ea4, @AddWith.1, @AddWith.1(%Int32) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.4f1: type = impl_witness_access %AddWith.lookup_impl_witness.587, element0 [symbolic_self]
// CHECK:STDOUT:   %facet_type.809: type = facet_type <@Destroy & @AddWith.1, @AddWith.1(%Int32) where %impl.elem0.4f1 = %Int32> [concrete]
// CHECK:STDOUT:   %AddWith.type.4f1: type = facet_type <@AddWith.1, @AddWith.1(%Int32)> [concrete]
// CHECK:STDOUT:   %AddWith.assoc_type.5fc: type = assoc_entity_type @AddWith.1, @AddWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %assoc0.836: %AddWith.assoc_type.5fc = assoc_entity element0, imports.%Core.import_ref.3ad [concrete]
// CHECK:STDOUT:   %assoc1.96b: %AddWith.assoc_type.5fc = assoc_entity element1, imports.%Core.import_ref.2b2 [concrete]
// CHECK:STDOUT:   %Int16.Op.type.da5d51.8: type = fn_type @Int16.Op.8 [concrete]
// CHECK:STDOUT:   %Int16.Op.7ce432.8: %Int16.Op.type.da5d51.8 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.bb0e2d.1: <witness> = custom_witness (%Int32, %Int16.Op.7ce432.8), @AddWith.1, @AddWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %facet_value.a60: %facet_type.809 = facet_value %Int16, (%custom_witness.ecd0a5.1, %custom_witness.bb0e2d.1) [concrete]
// CHECK:STDOUT:   %complete_type.bc1: <witness> = complete_type_witness %facet_type.35e [concrete]
// CHECK:STDOUT:   %.Self.as_type.1e3: type = facet_access_type %.Self.ea4 [symbolic_self]
// CHECK:STDOUT:   %pattern_type.a7b: type = pattern_type %facet_type.809 [concrete]
// CHECK:STDOUT:   %facet_type.cbc: type = facet_type <@Destroy & @SubWith.1, @SubWith.1(%Int32)> [concrete]
// CHECK:STDOUT:   %.Self.ae4: %facet_type.cbc = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.d98: <witness> = lookup_impl_witness %.Self.ae4, @SubWith.1, @SubWith.1(%Int32) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.f09: type = impl_witness_access %SubWith.lookup_impl_witness.d98, element0 [symbolic_self]
// CHECK:STDOUT:   %facet_type.445: type = facet_type <@Destroy & @SubWith.1, @SubWith.1(%Int32) where %impl.elem0.f09 = %Int32> [concrete]
// CHECK:STDOUT:   %SubWith.type.40d: type = facet_type <@SubWith.1, @SubWith.1(%Int32)> [concrete]
// CHECK:STDOUT:   %SubWith.assoc_type.73f: type = assoc_entity_type @SubWith.1, @SubWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %assoc0.c3d: %SubWith.assoc_type.73f = assoc_entity element0, imports.%Core.import_ref.afe [concrete]
// CHECK:STDOUT:   %assoc1.827: %SubWith.assoc_type.73f = assoc_entity element1, imports.%Core.import_ref.bc0 [concrete]
// CHECK:STDOUT:   %Int16.Op.type.da5d51.9: type = fn_type @Int16.Op.9 [concrete]
// CHECK:STDOUT:   %Int16.Op.7ce432.9: %Int16.Op.type.da5d51.9 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.dee291.1: <witness> = custom_witness (%Int32, %Int16.Op.7ce432.9), @SubWith.1, @SubWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %facet_value.88f: %facet_type.445 = facet_value %Int16, (%custom_witness.ecd0a5.1, %custom_witness.dee291.1) [concrete]
// CHECK:STDOUT:   %complete_type.44f: <witness> = complete_type_witness %facet_type.cbc [concrete]
// CHECK:STDOUT:   %.Self.as_type.484: type = facet_access_type %.Self.ae4 [symbolic_self]
// CHECK:STDOUT:   %pattern_type.766: type = pattern_type %facet_type.445 [concrete]
// CHECK:STDOUT:   %facet_type.655: type = facet_type <@Destroy & @MulWith.1, @MulWith.1(%Int32)> [concrete]
// CHECK:STDOUT:   %.Self.e02: %facet_type.655 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.5f7: <witness> = lookup_impl_witness %.Self.e02, @MulWith.1, @MulWith.1(%Int32) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.0bb: type = impl_witness_access %MulWith.lookup_impl_witness.5f7, element0 [symbolic_self]
// CHECK:STDOUT:   %facet_type.8a8: type = facet_type <@Destroy & @MulWith.1, @MulWith.1(%Int32) where %impl.elem0.0bb = %Int32> [concrete]
// CHECK:STDOUT:   %MulWith.type.8d1: type = facet_type <@MulWith.1, @MulWith.1(%Int32)> [concrete]
// CHECK:STDOUT:   %MulWith.assoc_type.d17: type = assoc_entity_type @MulWith.1, @MulWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %assoc0.225: %MulWith.assoc_type.d17 = assoc_entity element0, imports.%Core.import_ref.648 [concrete]
// CHECK:STDOUT:   %assoc1.a92: %MulWith.assoc_type.d17 = assoc_entity element1, imports.%Core.import_ref.32c [concrete]
// CHECK:STDOUT:   %Int16.Op.type.da5d51.10: type = fn_type @Int16.Op.10 [concrete]
// CHECK:STDOUT:   %Int16.Op.7ce432.10: %Int16.Op.type.da5d51.10 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.db8ceb.1: <witness> = custom_witness (%Int32, %Int16.Op.7ce432.10), @MulWith.1, @MulWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %facet_value.668: %facet_type.8a8 = facet_value %Int16, (%custom_witness.ecd0a5.1, %custom_witness.db8ceb.1) [concrete]
// CHECK:STDOUT:   %complete_type.aab: <witness> = complete_type_witness %facet_type.655 [concrete]
// CHECK:STDOUT:   %.Self.as_type.efe: type = facet_access_type %.Self.e02 [symbolic_self]
// CHECK:STDOUT:   %pattern_type.bf1: type = pattern_type %facet_type.8a8 [concrete]
// CHECK:STDOUT:   %facet_type.9ed: type = facet_type <@Destroy & @DivWith.1, @DivWith.1(%Int32)> [concrete]
// CHECK:STDOUT:   %.Self.536: %facet_type.9ed = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.3ec: <witness> = lookup_impl_witness %.Self.536, @DivWith.1, @DivWith.1(%Int32) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.1dc3: type = impl_witness_access %DivWith.lookup_impl_witness.3ec, element0 [symbolic_self]
// CHECK:STDOUT:   %facet_type.a43: type = facet_type <@Destroy & @DivWith.1, @DivWith.1(%Int32) where %impl.elem0.1dc3 = %Int32> [concrete]
// CHECK:STDOUT:   %DivWith.type.a2b: type = facet_type <@DivWith.1, @DivWith.1(%Int32)> [concrete]
// CHECK:STDOUT:   %DivWith.assoc_type.185: type = assoc_entity_type @DivWith.1, @DivWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %assoc0.2d7: %DivWith.assoc_type.185 = assoc_entity element0, imports.%Core.import_ref.db1 [concrete]
// CHECK:STDOUT:   %assoc1.9b2: %DivWith.assoc_type.185 = assoc_entity element1, imports.%Core.import_ref.7dc [concrete]
// CHECK:STDOUT:   %Int16.Op.type.da5d51.11: type = fn_type @Int16.Op.11 [concrete]
// CHECK:STDOUT:   %Int16.Op.7ce432.11: %Int16.Op.type.da5d51.11 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.deca3a.1: <witness> = custom_witness (%Int32, %Int16.Op.7ce432.11), @DivWith.1, @DivWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %facet_value.c22: %facet_type.a43 = facet_value %Int16, (%custom_witness.ecd0a5.1, %custom_witness.deca3a.1) [concrete]
// CHECK:STDOUT:   %complete_type.bfd: <witness> = complete_type_witness %facet_type.9ed [concrete]
// CHECK:STDOUT:   %.Self.as_type.a54: type = facet_access_type %.Self.536 [symbolic_self]
// CHECK:STDOUT:   %pattern_type.3fb: type = pattern_type %facet_type.a43 [concrete]
// CHECK:STDOUT:   %facet_type.cb7: type = facet_type <@Destroy & @ModWith.1, @ModWith.1(%Int32)> [concrete]
// CHECK:STDOUT:   %.Self.6c7: %facet_type.cb7 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.f61: <witness> = lookup_impl_witness %.Self.6c7, @ModWith.1, @ModWith.1(%Int32) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.e0b: type = impl_witness_access %ModWith.lookup_impl_witness.f61, element0 [symbolic_self]
// CHECK:STDOUT:   %facet_type.e5f: type = facet_type <@Destroy & @ModWith.1, @ModWith.1(%Int32) where %impl.elem0.e0b = %Int32> [concrete]
// CHECK:STDOUT:   %ModWith.type.cff: type = facet_type <@ModWith.1, @ModWith.1(%Int32)> [concrete]
// CHECK:STDOUT:   %ModWith.assoc_type.557: type = assoc_entity_type @ModWith.1, @ModWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %assoc0.beb: %ModWith.assoc_type.557 = assoc_entity element0, imports.%Core.import_ref.78b [concrete]
// CHECK:STDOUT:   %assoc1.b6a: %ModWith.assoc_type.557 = assoc_entity element1, imports.%Core.import_ref.9e2 [concrete]
// CHECK:STDOUT:   %Int16.Op.type.da5d51.12: type = fn_type @Int16.Op.12 [concrete]
// CHECK:STDOUT:   %Int16.Op.7ce432.12: %Int16.Op.type.da5d51.12 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.c72db9.1: <witness> = custom_witness (%Int32, %Int16.Op.7ce432.12), @ModWith.1, @ModWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %facet_value.194: %facet_type.e5f = facet_value %Int16, (%custom_witness.ecd0a5.1, %custom_witness.c72db9.1) [concrete]
// CHECK:STDOUT:   %complete_type.e06: <witness> = complete_type_witness %facet_type.cb7 [concrete]
// CHECK:STDOUT:   %.Self.as_type.2f0: type = facet_access_type %.Self.6c7 [symbolic_self]
// CHECK:STDOUT:   %pattern_type.011: type = pattern_type %facet_type.e5f [concrete]
// CHECK:STDOUT:   %facet_type.e93: type = facet_type <@Destroy & @AddWith.1, @AddWith.1(%Int16) where %impl.elem0.1ef = %Int32> [concrete]
// CHECK:STDOUT:   %Int32.Op.type.da5d51.2: type = fn_type @Int32.Op.2 [concrete]
// CHECK:STDOUT:   %Int32.Op.7ce432.2: %Int32.Op.type.da5d51.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.8d5: <witness> = custom_witness (%Int32, %Int32.Op.7ce432.2), @AddWith.1, @AddWith.1(%Int16) [concrete]
// CHECK:STDOUT:   %facet_value.5cb: %facet_type.e93 = facet_value %Int32, (%custom_witness.ecd0a5.2, %custom_witness.8d5) [concrete]
// CHECK:STDOUT:   %pattern_type.05d: type = pattern_type %facet_type.e93 [concrete]
// CHECK:STDOUT:   %facet_type.7be: type = facet_type <@Destroy & @SubWith.1, @SubWith.1(%Int16) where %impl.elem0.214 = %Int32> [concrete]
// CHECK:STDOUT:   %Int32.Op.type.da5d51.3: type = fn_type @Int32.Op.3 [concrete]
// CHECK:STDOUT:   %Int32.Op.7ce432.3: %Int32.Op.type.da5d51.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.14c: <witness> = custom_witness (%Int32, %Int32.Op.7ce432.3), @SubWith.1, @SubWith.1(%Int16) [concrete]
// CHECK:STDOUT:   %facet_value.801: %facet_type.7be = facet_value %Int32, (%custom_witness.ecd0a5.2, %custom_witness.14c) [concrete]
// CHECK:STDOUT:   %pattern_type.452: type = pattern_type %facet_type.7be [concrete]
// CHECK:STDOUT:   %facet_type.460: type = facet_type <@Destroy & @MulWith.1, @MulWith.1(%Int16) where %impl.elem0.851 = %Int32> [concrete]
// CHECK:STDOUT:   %Int32.Op.type.da5d51.4: type = fn_type @Int32.Op.4 [concrete]
// CHECK:STDOUT:   %Int32.Op.7ce432.4: %Int32.Op.type.da5d51.4 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.fd0: <witness> = custom_witness (%Int32, %Int32.Op.7ce432.4), @MulWith.1, @MulWith.1(%Int16) [concrete]
// CHECK:STDOUT:   %facet_value.ac1: %facet_type.460 = facet_value %Int32, (%custom_witness.ecd0a5.2, %custom_witness.fd0) [concrete]
// CHECK:STDOUT:   %pattern_type.cfe: type = pattern_type %facet_type.460 [concrete]
// CHECK:STDOUT:   %facet_type.072: type = facet_type <@Destroy & @DivWith.1, @DivWith.1(%Int16) where %impl.elem0.1ae = %Int32> [concrete]
// CHECK:STDOUT:   %Int32.Op.type.da5d51.5: type = fn_type @Int32.Op.5 [concrete]
// CHECK:STDOUT:   %Int32.Op.7ce432.5: %Int32.Op.type.da5d51.5 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.3a3: <witness> = custom_witness (%Int32, %Int32.Op.7ce432.5), @DivWith.1, @DivWith.1(%Int16) [concrete]
// CHECK:STDOUT:   %facet_value.c58: %facet_type.072 = facet_value %Int32, (%custom_witness.ecd0a5.2, %custom_witness.3a3) [concrete]
// CHECK:STDOUT:   %pattern_type.7e3: type = pattern_type %facet_type.072 [concrete]
// CHECK:STDOUT:   %facet_type.07e: type = facet_type <@Destroy & @ModWith.1, @ModWith.1(%Int16) where %impl.elem0.7c3 = %Int32> [concrete]
// CHECK:STDOUT:   %Int32.Op.type.da5d51.6: type = fn_type @Int32.Op.6 [concrete]
// CHECK:STDOUT:   %Int32.Op.7ce432.6: %Int32.Op.type.da5d51.6 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.e15: <witness> = custom_witness (%Int32, %Int32.Op.7ce432.6), @ModWith.1, @ModWith.1(%Int16) [concrete]
// CHECK:STDOUT:   %facet_value.f21: %facet_type.07e = facet_value %Int32, (%custom_witness.ecd0a5.2, %custom_witness.e15) [concrete]
// CHECK:STDOUT:   %pattern_type.83e: type = pattern_type %facet_type.07e [concrete]
// CHECK:STDOUT:   %Int32.Op.type.da5d51.7: type = fn_type @Int32.Op.7 [concrete]
// CHECK:STDOUT:   %Int32.Op.7ce432.7: %Int32.Op.type.da5d51.7 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.bb0e2d.2: <witness> = custom_witness (%Int32, %Int32.Op.7ce432.7), @AddWith.1, @AddWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %facet_value.3a9: %facet_type.809 = facet_value %Int32, (%custom_witness.ecd0a5.2, %custom_witness.bb0e2d.2) [concrete]
// CHECK:STDOUT:   %Int32.Op.type.da5d51.8: type = fn_type @Int32.Op.8 [concrete]
// CHECK:STDOUT:   %Int32.Op.7ce432.8: %Int32.Op.type.da5d51.8 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.dee291.2: <witness> = custom_witness (%Int32, %Int32.Op.7ce432.8), @SubWith.1, @SubWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %facet_value.785: %facet_type.445 = facet_value %Int32, (%custom_witness.ecd0a5.2, %custom_witness.dee291.2) [concrete]
// CHECK:STDOUT:   %Int32.Op.type.da5d51.9: type = fn_type @Int32.Op.9 [concrete]
// CHECK:STDOUT:   %Int32.Op.7ce432.9: %Int32.Op.type.da5d51.9 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.db8ceb.2: <witness> = custom_witness (%Int32, %Int32.Op.7ce432.9), @MulWith.1, @MulWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %facet_value.0dc: %facet_type.8a8 = facet_value %Int32, (%custom_witness.ecd0a5.2, %custom_witness.db8ceb.2) [concrete]
// CHECK:STDOUT:   %Int32.Op.type.da5d51.10: type = fn_type @Int32.Op.10 [concrete]
// CHECK:STDOUT:   %Int32.Op.7ce432.10: %Int32.Op.type.da5d51.10 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.deca3a.2: <witness> = custom_witness (%Int32, %Int32.Op.7ce432.10), @DivWith.1, @DivWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %facet_value.a53: %facet_type.a43 = facet_value %Int32, (%custom_witness.ecd0a5.2, %custom_witness.deca3a.2) [concrete]
// CHECK:STDOUT:   %Int32.Op.type.da5d51.11: type = fn_type @Int32.Op.11 [concrete]
// CHECK:STDOUT:   %Int32.Op.7ce432.11: %Int32.Op.type.da5d51.11 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.c72db9.2: <witness> = custom_witness (%Int32, %Int32.Op.7ce432.11), @ModWith.1, @ModWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %facet_value.821: %facet_type.e5f = facet_value %Int32, (%custom_witness.ecd0a5.2, %custom_witness.c72db9.2) [concrete]
// CHECK:STDOUT:   %Int64.Op.type.da5d51.1: type = fn_type @Int64.Op.1 [concrete]
// CHECK:STDOUT:   %Int64.Op.7ce432.1: %Int64.Op.type.da5d51.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.ecd0a5.3: <witness> = custom_witness (%Int64.Op.7ce432.1), @Destroy [concrete]
// CHECK:STDOUT:   %Destroy.facet.b0d: %Destroy.type = facet_value %Int64, (%custom_witness.ecd0a5.3) [concrete]
// CHECK:STDOUT:   %facet_type.bb9: type = facet_type <@Destroy & @AddWith.1, @AddWith.1(%Int64)> [concrete]
// CHECK:STDOUT:   %.Self.945: %facet_type.bb9 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.3da: <witness> = lookup_impl_witness %.Self.945, @AddWith.1, @AddWith.1(%Int64) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.e09: type = impl_witness_access %AddWith.lookup_impl_witness.3da, element0 [symbolic_self]
// CHECK:STDOUT:   %facet_type.afb: type = facet_type <@Destroy & @AddWith.1, @AddWith.1(%Int64) where %impl.elem0.e09 = %Int64> [concrete]
// CHECK:STDOUT:   %AddWith.type.fef: type = facet_type <@AddWith.1, @AddWith.1(%Int64)> [concrete]
// CHECK:STDOUT:   %AddWith.assoc_type.c7a: type = assoc_entity_type @AddWith.1, @AddWith.1(%Int64) [concrete]
// CHECK:STDOUT:   %assoc0.57b: %AddWith.assoc_type.c7a = assoc_entity element0, imports.%Core.import_ref.3ad [concrete]
// CHECK:STDOUT:   %assoc1.7dd: %AddWith.assoc_type.c7a = assoc_entity element1, imports.%Core.import_ref.2b2 [concrete]
// CHECK:STDOUT:   %Op.type.175699.6: type = fn_type @Op.6 [concrete]
// CHECK:STDOUT:   %Op.848e8c.6: %Op.type.175699.6 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.3f7: <witness> = custom_witness (%Int64, %Op.848e8c.6), @AddWith.1, @AddWith.1(%Int64) [concrete]
// CHECK:STDOUT:   %facet_value.527: %facet_type.afb = facet_value %Int32, (%custom_witness.ecd0a5.2, %custom_witness.3f7) [concrete]
// CHECK:STDOUT:   %complete_type.fdd: <witness> = complete_type_witness %facet_type.bb9 [concrete]
// CHECK:STDOUT:   %.Self.as_type.833: type = facet_access_type %.Self.945 [symbolic_self]
// CHECK:STDOUT:   %pattern_type.8eb: type = pattern_type %facet_type.afb [concrete]
// CHECK:STDOUT:   %facet_type.4c4: type = facet_type <@Destroy & @SubWith.1, @SubWith.1(%Int64)> [concrete]
// CHECK:STDOUT:   %.Self.89d: %facet_type.4c4 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.f3c: <witness> = lookup_impl_witness %.Self.89d, @SubWith.1, @SubWith.1(%Int64) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.1dcc: type = impl_witness_access %SubWith.lookup_impl_witness.f3c, element0 [symbolic_self]
// CHECK:STDOUT:   %facet_type.019: type = facet_type <@Destroy & @SubWith.1, @SubWith.1(%Int64) where %impl.elem0.1dcc = %Int64> [concrete]
// CHECK:STDOUT:   %SubWith.type.c5d: type = facet_type <@SubWith.1, @SubWith.1(%Int64)> [concrete]
// CHECK:STDOUT:   %SubWith.assoc_type.6bb: type = assoc_entity_type @SubWith.1, @SubWith.1(%Int64) [concrete]
// CHECK:STDOUT:   %assoc0.2092: %SubWith.assoc_type.6bb = assoc_entity element0, imports.%Core.import_ref.afe [concrete]
// CHECK:STDOUT:   %assoc1.83a: %SubWith.assoc_type.6bb = assoc_entity element1, imports.%Core.import_ref.bc0 [concrete]
// CHECK:STDOUT:   %Op.type.175699.7: type = fn_type @Op.7 [concrete]
// CHECK:STDOUT:   %Op.848e8c.7: %Op.type.175699.7 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.aad: <witness> = custom_witness (%Int64, %Op.848e8c.7), @SubWith.1, @SubWith.1(%Int64) [concrete]
// CHECK:STDOUT:   %facet_value.93f: %facet_type.019 = facet_value %Int32, (%custom_witness.ecd0a5.2, %custom_witness.aad) [concrete]
// CHECK:STDOUT:   %complete_type.bee: <witness> = complete_type_witness %facet_type.4c4 [concrete]
// CHECK:STDOUT:   %.Self.as_type.8357: type = facet_access_type %.Self.89d [symbolic_self]
// CHECK:STDOUT:   %pattern_type.37c: type = pattern_type %facet_type.019 [concrete]
// CHECK:STDOUT:   %facet_type.5f9: type = facet_type <@Destroy & @MulWith.1, @MulWith.1(%Int64)> [concrete]
// CHECK:STDOUT:   %.Self.baa: %facet_type.5f9 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.998: <witness> = lookup_impl_witness %.Self.baa, @MulWith.1, @MulWith.1(%Int64) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.106: type = impl_witness_access %MulWith.lookup_impl_witness.998, element0 [symbolic_self]
// CHECK:STDOUT:   %facet_type.0c8: type = facet_type <@Destroy & @MulWith.1, @MulWith.1(%Int64) where %impl.elem0.106 = %Int64> [concrete]
// CHECK:STDOUT:   %MulWith.type.22c: type = facet_type <@MulWith.1, @MulWith.1(%Int64)> [concrete]
// CHECK:STDOUT:   %MulWith.assoc_type.d02: type = assoc_entity_type @MulWith.1, @MulWith.1(%Int64) [concrete]
// CHECK:STDOUT:   %assoc0.1fd: %MulWith.assoc_type.d02 = assoc_entity element0, imports.%Core.import_ref.648 [concrete]
// CHECK:STDOUT:   %assoc1.fab: %MulWith.assoc_type.d02 = assoc_entity element1, imports.%Core.import_ref.32c [concrete]
// CHECK:STDOUT:   %Op.type.175699.8: type = fn_type @Op.8 [concrete]
// CHECK:STDOUT:   %Op.848e8c.8: %Op.type.175699.8 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.1d4: <witness> = custom_witness (%Int64, %Op.848e8c.8), @MulWith.1, @MulWith.1(%Int64) [concrete]
// CHECK:STDOUT:   %facet_value.450: %facet_type.0c8 = facet_value %Int32, (%custom_witness.ecd0a5.2, %custom_witness.1d4) [concrete]
// CHECK:STDOUT:   %complete_type.4a7: <witness> = complete_type_witness %facet_type.5f9 [concrete]
// CHECK:STDOUT:   %.Self.as_type.374: type = facet_access_type %.Self.baa [symbolic_self]
// CHECK:STDOUT:   %pattern_type.be2: type = pattern_type %facet_type.0c8 [concrete]
// CHECK:STDOUT:   %facet_type.6ae: type = facet_type <@Destroy & @DivWith.1, @DivWith.1(%Int64)> [concrete]
// CHECK:STDOUT:   %.Self.dac: %facet_type.6ae = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.94a: <witness> = lookup_impl_witness %.Self.dac, @DivWith.1, @DivWith.1(%Int64) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.d13: type = impl_witness_access %DivWith.lookup_impl_witness.94a, element0 [symbolic_self]
// CHECK:STDOUT:   %facet_type.c0d: type = facet_type <@Destroy & @DivWith.1, @DivWith.1(%Int64) where %impl.elem0.d13 = %Int64> [concrete]
// CHECK:STDOUT:   %DivWith.type.95f: type = facet_type <@DivWith.1, @DivWith.1(%Int64)> [concrete]
// CHECK:STDOUT:   %DivWith.assoc_type.7db: type = assoc_entity_type @DivWith.1, @DivWith.1(%Int64) [concrete]
// CHECK:STDOUT:   %assoc0.f64: %DivWith.assoc_type.7db = assoc_entity element0, imports.%Core.import_ref.db1 [concrete]
// CHECK:STDOUT:   %assoc1.0b4: %DivWith.assoc_type.7db = assoc_entity element1, imports.%Core.import_ref.7dc [concrete]
// CHECK:STDOUT:   %Op.type.175699.9: type = fn_type @Op.9 [concrete]
// CHECK:STDOUT:   %Op.848e8c.9: %Op.type.175699.9 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.515: <witness> = custom_witness (%Int64, %Op.848e8c.9), @DivWith.1, @DivWith.1(%Int64) [concrete]
// CHECK:STDOUT:   %facet_value.2d8: %facet_type.c0d = facet_value %Int32, (%custom_witness.ecd0a5.2, %custom_witness.515) [concrete]
// CHECK:STDOUT:   %complete_type.b1e: <witness> = complete_type_witness %facet_type.6ae [concrete]
// CHECK:STDOUT:   %.Self.as_type.771: type = facet_access_type %.Self.dac [symbolic_self]
// CHECK:STDOUT:   %pattern_type.841: type = pattern_type %facet_type.c0d [concrete]
// CHECK:STDOUT:   %facet_type.c80: type = facet_type <@Destroy & @ModWith.1, @ModWith.1(%Int64)> [concrete]
// CHECK:STDOUT:   %.Self.a93: %facet_type.c80 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.324: <witness> = lookup_impl_witness %.Self.a93, @ModWith.1, @ModWith.1(%Int64) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.96f: type = impl_witness_access %ModWith.lookup_impl_witness.324, element0 [symbolic_self]
// CHECK:STDOUT:   %facet_type.b48: type = facet_type <@Destroy & @ModWith.1, @ModWith.1(%Int64) where %impl.elem0.96f = %Int64> [concrete]
// CHECK:STDOUT:   %ModWith.type.e6e: type = facet_type <@ModWith.1, @ModWith.1(%Int64)> [concrete]
// CHECK:STDOUT:   %ModWith.assoc_type.ac1: type = assoc_entity_type @ModWith.1, @ModWith.1(%Int64) [concrete]
// CHECK:STDOUT:   %assoc0.97e: %ModWith.assoc_type.ac1 = assoc_entity element0, imports.%Core.import_ref.78b [concrete]
// CHECK:STDOUT:   %assoc1.6f9: %ModWith.assoc_type.ac1 = assoc_entity element1, imports.%Core.import_ref.9e2 [concrete]
// CHECK:STDOUT:   %Op.type.175699.10: type = fn_type @Op.10 [concrete]
// CHECK:STDOUT:   %Op.848e8c.10: %Op.type.175699.10 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.1af: <witness> = custom_witness (%Int64, %Op.848e8c.10), @ModWith.1, @ModWith.1(%Int64) [concrete]
// CHECK:STDOUT:   %facet_value.bab: %facet_type.b48 = facet_value %Int32, (%custom_witness.ecd0a5.2, %custom_witness.1af) [concrete]
// CHECK:STDOUT:   %complete_type.320: <witness> = complete_type_witness %facet_type.c80 [concrete]
// CHECK:STDOUT:   %.Self.as_type.d20: type = facet_access_type %.Self.a93 [symbolic_self]
// CHECK:STDOUT:   %pattern_type.250: type = pattern_type %facet_type.b48 [concrete]
// CHECK:STDOUT:   %facet_type.8f4: type = facet_type <@Destroy & @AddWith.1, @AddWith.1(%Int32) where %impl.elem0.4f1 = %Int64> [concrete]
// CHECK:STDOUT:   %Op.type.175699.11: type = fn_type @Op.11 [concrete]
// CHECK:STDOUT:   %Op.848e8c.11: %Op.type.175699.11 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.dbe: <witness> = custom_witness (%Int64, %Op.848e8c.11), @AddWith.1, @AddWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %facet_value.461: %facet_type.8f4 = facet_value %Int64, (%custom_witness.ecd0a5.3, %custom_witness.dbe) [concrete]
// CHECK:STDOUT:   %pattern_type.ff2: type = pattern_type %facet_type.8f4 [concrete]
// CHECK:STDOUT:   %facet_type.458: type = facet_type <@Destroy & @SubWith.1, @SubWith.1(%Int32) where %impl.elem0.f09 = %Int64> [concrete]
// CHECK:STDOUT:   %Op.type.175699.12: type = fn_type @Op.12 [concrete]
// CHECK:STDOUT:   %Op.848e8c.12: %Op.type.175699.12 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.af3: <witness> = custom_witness (%Int64, %Op.848e8c.12), @SubWith.1, @SubWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %facet_value.ec6: %facet_type.458 = facet_value %Int64, (%custom_witness.ecd0a5.3, %custom_witness.af3) [concrete]
// CHECK:STDOUT:   %pattern_type.679: type = pattern_type %facet_type.458 [concrete]
// CHECK:STDOUT:   %facet_type.fe5: type = facet_type <@Destroy & @MulWith.1, @MulWith.1(%Int32) where %impl.elem0.0bb = %Int64> [concrete]
// CHECK:STDOUT:   %Op.type.175699.13: type = fn_type @Op.13 [concrete]
// CHECK:STDOUT:   %Op.848e8c.13: %Op.type.175699.13 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.d10: <witness> = custom_witness (%Int64, %Op.848e8c.13), @MulWith.1, @MulWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %facet_value.7d6: %facet_type.fe5 = facet_value %Int64, (%custom_witness.ecd0a5.3, %custom_witness.d10) [concrete]
// CHECK:STDOUT:   %pattern_type.66a: type = pattern_type %facet_type.fe5 [concrete]
// CHECK:STDOUT:   %facet_type.b92: type = facet_type <@Destroy & @DivWith.1, @DivWith.1(%Int32) where %impl.elem0.1dc3 = %Int64> [concrete]
// CHECK:STDOUT:   %Op.type.175699.14: type = fn_type @Op.14 [concrete]
// CHECK:STDOUT:   %Op.848e8c.14: %Op.type.175699.14 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.438: <witness> = custom_witness (%Int64, %Op.848e8c.14), @DivWith.1, @DivWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %facet_value.798: %facet_type.b92 = facet_value %Int64, (%custom_witness.ecd0a5.3, %custom_witness.438) [concrete]
// CHECK:STDOUT:   %pattern_type.624: type = pattern_type %facet_type.b92 [concrete]
// CHECK:STDOUT:   %facet_type.235: type = facet_type <@Destroy & @ModWith.1, @ModWith.1(%Int32) where %impl.elem0.e0b = %Int64> [concrete]
// CHECK:STDOUT:   %Op.type.175699.15: type = fn_type @Op.15 [concrete]
// CHECK:STDOUT:   %Op.848e8c.15: %Op.type.175699.15 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.bf8: <witness> = custom_witness (%Int64, %Op.848e8c.15), @ModWith.1, @ModWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %facet_value.dd8: %facet_type.235 = facet_value %Int64, (%custom_witness.ecd0a5.3, %custom_witness.bf8) [concrete]
// CHECK:STDOUT:   %pattern_type.0fa: type = pattern_type %facet_type.235 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.1c2: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.b0d) [concrete]
// CHECK:STDOUT:   %.4de: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.1c2, %Destroy.facet.b0d [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.257: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.184) [concrete]
// CHECK:STDOUT:   %.d3f: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.257, %Destroy.facet.184 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.9f9: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.551) [concrete]
// CHECK:STDOUT:   %.16b: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.9f9, %Destroy.facet.551 [concrete]
// CHECK:STDOUT:   %U.67d: type = symbolic_binding U, 0 [symbolic]
// CHECK:STDOUT:   %AddAssignWith.type.d3e5c5.1: type = facet_type <@AddAssignWith.1, @AddAssignWith.1(%Other)> [symbolic]
// CHECK:STDOUT:   %Self.c28b6d.1: %AddAssignWith.type.d3e5c5.1 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %AddAssignWith.assoc_type.2656aa.1: type = assoc_entity_type @AddAssignWith.1, @AddAssignWith.1(%Other) [symbolic]
// CHECK:STDOUT:   %AddAssignWith.WithSelf.Op.type.646415.1: type = fn_type @AddAssignWith.WithSelf.Op, @AddAssignWith.WithSelf(%Other, %Self.c28b6d.1) [symbolic]
// CHECK:STDOUT:   %AddAssignWith.WithSelf.Op.f5fdb8.1: %AddAssignWith.WithSelf.Op.type.646415.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %AddAssignWith.type.d3e5c5.2: type = facet_type <@AddAssignWith.1, @AddAssignWith.1(%U.67d)> [symbolic]
// CHECK:STDOUT:   %pattern_type.b54: type = pattern_type %AddAssignWith.type.d3e5c5.2 [symbolic]
// CHECK:STDOUT:   %T.c28: %AddAssignWith.type.d3e5c5.2 = symbolic_binding T, 1 [symbolic]
// CHECK:STDOUT:   %AddAssignWith.assoc_type.2656aa.2: type = assoc_entity_type @AddAssignWith.1, @AddAssignWith.1(%U.67d) [symbolic]
// CHECK:STDOUT:   %assoc0.d3a3f1.2: %AddAssignWith.assoc_type.2656aa.2 = assoc_entity element0, imports.%Core.import_ref.073 [symbolic]
// CHECK:STDOUT:   %T.as_type.984: type = facet_access_type %T.c28 [symbolic]
// CHECK:STDOUT:   %pattern_type.7bda36.2: type = pattern_type %T.as_type.984 [symbolic]
// CHECK:STDOUT:   %pattern_type.51d1c4.4: type = pattern_type %U.67d [symbolic]
// CHECK:STDOUT:   %require_complete.b1a: <witness> = require_complete_type %AddAssignWith.type.d3e5c5.2 [symbolic]
// CHECK:STDOUT:   %assoc0.595: %AddAssignWith.assoc_type.2656aa.1 = assoc_entity element0, imports.%Core.import_ref.0da [symbolic]
// CHECK:STDOUT:   %AddAssignWith.lookup_impl_witness: <witness> = lookup_impl_witness %T.c28, @AddAssignWith.1, @AddAssignWith.1(%U.67d) [symbolic]
// CHECK:STDOUT:   %AddAssignWith.WithSelf.Op.type.646415.3: type = fn_type @AddAssignWith.WithSelf.Op, @AddAssignWith.WithSelf(%U.67d, %T.c28) [symbolic]
// CHECK:STDOUT:   %.8a0: type = fn_type_with_self_type %AddAssignWith.WithSelf.Op.type.646415.3, %T.c28 [symbolic]
// CHECK:STDOUT:   %impl.elem0.1e5: %.8a0 = impl_witness_access %AddAssignWith.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.97d: <specific function> = specific_impl_function %impl.elem0.1e5, @AddAssignWith.WithSelf.Op(%U.67d, %T.c28) [symbolic]
// CHECK:STDOUT:   %SubAssignWith.type.267e69.1: type = facet_type <@SubAssignWith.1, @SubAssignWith.1(%Other)> [symbolic]
// CHECK:STDOUT:   %Self.bd9eb5.1: %SubAssignWith.type.267e69.1 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %SubAssignWith.assoc_type.f4a281.1: type = assoc_entity_type @SubAssignWith.1, @SubAssignWith.1(%Other) [symbolic]
// CHECK:STDOUT:   %SubAssignWith.WithSelf.Op.type.1e97d0.1: type = fn_type @SubAssignWith.WithSelf.Op, @SubAssignWith.WithSelf(%Other, %Self.bd9eb5.1) [symbolic]
// CHECK:STDOUT:   %SubAssignWith.WithSelf.Op.0e2d98.1: %SubAssignWith.WithSelf.Op.type.1e97d0.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %SubAssignWith.type.267e69.2: type = facet_type <@SubAssignWith.1, @SubAssignWith.1(%U.67d)> [symbolic]
// CHECK:STDOUT:   %pattern_type.c3f: type = pattern_type %SubAssignWith.type.267e69.2 [symbolic]
// CHECK:STDOUT:   %T.bd9: %SubAssignWith.type.267e69.2 = symbolic_binding T, 1 [symbolic]
// CHECK:STDOUT:   %SubAssignWith.assoc_type.f4a281.2: type = assoc_entity_type @SubAssignWith.1, @SubAssignWith.1(%U.67d) [symbolic]
// CHECK:STDOUT:   %assoc0.556a96.2: %SubAssignWith.assoc_type.f4a281.2 = assoc_entity element0, imports.%Core.import_ref.fc8 [symbolic]
// CHECK:STDOUT:   %T.as_type.b79: type = facet_access_type %T.bd9 [symbolic]
// CHECK:STDOUT:   %pattern_type.944ba1.2: type = pattern_type %T.as_type.b79 [symbolic]
// CHECK:STDOUT:   %require_complete.401: <witness> = require_complete_type %SubAssignWith.type.267e69.2 [symbolic]
// CHECK:STDOUT:   %assoc0.5d8: %SubAssignWith.assoc_type.f4a281.1 = assoc_entity element0, imports.%Core.import_ref.4e6 [symbolic]
// CHECK:STDOUT:   %SubAssignWith.lookup_impl_witness: <witness> = lookup_impl_witness %T.bd9, @SubAssignWith.1, @SubAssignWith.1(%U.67d) [symbolic]
// CHECK:STDOUT:   %SubAssignWith.WithSelf.Op.type.1e97d0.3: type = fn_type @SubAssignWith.WithSelf.Op, @SubAssignWith.WithSelf(%U.67d, %T.bd9) [symbolic]
// CHECK:STDOUT:   %.b89: type = fn_type_with_self_type %SubAssignWith.WithSelf.Op.type.1e97d0.3, %T.bd9 [symbolic]
// CHECK:STDOUT:   %impl.elem0.f58: %.b89 = impl_witness_access %SubAssignWith.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.340: <specific function> = specific_impl_function %impl.elem0.f58, @SubAssignWith.WithSelf.Op(%U.67d, %T.bd9) [symbolic]
// CHECK:STDOUT:   %MulAssignWith.type.802f2e.1: type = facet_type <@MulAssignWith.1, @MulAssignWith.1(%Other)> [symbolic]
// CHECK:STDOUT:   %Self.b81696.1: %MulAssignWith.type.802f2e.1 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %MulAssignWith.assoc_type.fda5e1.1: type = assoc_entity_type @MulAssignWith.1, @MulAssignWith.1(%Other) [symbolic]
// CHECK:STDOUT:   %MulAssignWith.WithSelf.Op.type.35745f.1: type = fn_type @MulAssignWith.WithSelf.Op, @MulAssignWith.WithSelf(%Other, %Self.b81696.1) [symbolic]
// CHECK:STDOUT:   %MulAssignWith.WithSelf.Op.352c74.1: %MulAssignWith.WithSelf.Op.type.35745f.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %MulAssignWith.type.802f2e.2: type = facet_type <@MulAssignWith.1, @MulAssignWith.1(%U.67d)> [symbolic]
// CHECK:STDOUT:   %pattern_type.306: type = pattern_type %MulAssignWith.type.802f2e.2 [symbolic]
// CHECK:STDOUT:   %T.b81: %MulAssignWith.type.802f2e.2 = symbolic_binding T, 1 [symbolic]
// CHECK:STDOUT:   %MulAssignWith.assoc_type.fda5e1.2: type = assoc_entity_type @MulAssignWith.1, @MulAssignWith.1(%U.67d) [symbolic]
// CHECK:STDOUT:   %assoc0.da1d40.2: %MulAssignWith.assoc_type.fda5e1.2 = assoc_entity element0, imports.%Core.import_ref.303 [symbolic]
// CHECK:STDOUT:   %T.as_type.2ad: type = facet_access_type %T.b81 [symbolic]
// CHECK:STDOUT:   %pattern_type.cd4e6a.2: type = pattern_type %T.as_type.2ad [symbolic]
// CHECK:STDOUT:   %require_complete.e52: <witness> = require_complete_type %MulAssignWith.type.802f2e.2 [symbolic]
// CHECK:STDOUT:   %assoc0.40c: %MulAssignWith.assoc_type.fda5e1.1 = assoc_entity element0, imports.%Core.import_ref.c93 [symbolic]
// CHECK:STDOUT:   %MulAssignWith.lookup_impl_witness: <witness> = lookup_impl_witness %T.b81, @MulAssignWith.1, @MulAssignWith.1(%U.67d) [symbolic]
// CHECK:STDOUT:   %MulAssignWith.WithSelf.Op.type.35745f.3: type = fn_type @MulAssignWith.WithSelf.Op, @MulAssignWith.WithSelf(%U.67d, %T.b81) [symbolic]
// CHECK:STDOUT:   %.683: type = fn_type_with_self_type %MulAssignWith.WithSelf.Op.type.35745f.3, %T.b81 [symbolic]
// CHECK:STDOUT:   %impl.elem0.936: %.683 = impl_witness_access %MulAssignWith.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.7bb: <specific function> = specific_impl_function %impl.elem0.936, @MulAssignWith.WithSelf.Op(%U.67d, %T.b81) [symbolic]
// CHECK:STDOUT:   %DivAssignWith.type.2e6641.1: type = facet_type <@DivAssignWith.1, @DivAssignWith.1(%Other)> [symbolic]
// CHECK:STDOUT:   %Self.e57cd6.1: %DivAssignWith.type.2e6641.1 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %DivAssignWith.assoc_type.c34e7f.1: type = assoc_entity_type @DivAssignWith.1, @DivAssignWith.1(%Other) [symbolic]
// CHECK:STDOUT:   %DivAssignWith.WithSelf.Op.type.8f43bb.1: type = fn_type @DivAssignWith.WithSelf.Op, @DivAssignWith.WithSelf(%Other, %Self.e57cd6.1) [symbolic]
// CHECK:STDOUT:   %DivAssignWith.WithSelf.Op.773fdb.1: %DivAssignWith.WithSelf.Op.type.8f43bb.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %DivAssignWith.type.2e6641.2: type = facet_type <@DivAssignWith.1, @DivAssignWith.1(%U.67d)> [symbolic]
// CHECK:STDOUT:   %pattern_type.72f: type = pattern_type %DivAssignWith.type.2e6641.2 [symbolic]
// CHECK:STDOUT:   %T.e57: %DivAssignWith.type.2e6641.2 = symbolic_binding T, 1 [symbolic]
// CHECK:STDOUT:   %DivAssignWith.assoc_type.c34e7f.2: type = assoc_entity_type @DivAssignWith.1, @DivAssignWith.1(%U.67d) [symbolic]
// CHECK:STDOUT:   %assoc0.857e90.2: %DivAssignWith.assoc_type.c34e7f.2 = assoc_entity element0, imports.%Core.import_ref.52b [symbolic]
// CHECK:STDOUT:   %T.as_type.905: type = facet_access_type %T.e57 [symbolic]
// CHECK:STDOUT:   %pattern_type.5d02d2.2: type = pattern_type %T.as_type.905 [symbolic]
// CHECK:STDOUT:   %require_complete.b86: <witness> = require_complete_type %DivAssignWith.type.2e6641.2 [symbolic]
// CHECK:STDOUT:   %assoc0.420: %DivAssignWith.assoc_type.c34e7f.1 = assoc_entity element0, imports.%Core.import_ref.a22 [symbolic]
// CHECK:STDOUT:   %DivAssignWith.lookup_impl_witness: <witness> = lookup_impl_witness %T.e57, @DivAssignWith.1, @DivAssignWith.1(%U.67d) [symbolic]
// CHECK:STDOUT:   %DivAssignWith.WithSelf.Op.type.8f43bb.3: type = fn_type @DivAssignWith.WithSelf.Op, @DivAssignWith.WithSelf(%U.67d, %T.e57) [symbolic]
// CHECK:STDOUT:   %.c07: type = fn_type_with_self_type %DivAssignWith.WithSelf.Op.type.8f43bb.3, %T.e57 [symbolic]
// CHECK:STDOUT:   %impl.elem0.537: %.c07 = impl_witness_access %DivAssignWith.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.f69: <specific function> = specific_impl_function %impl.elem0.537, @DivAssignWith.WithSelf.Op(%U.67d, %T.e57) [symbolic]
// CHECK:STDOUT:   %ModAssignWith.type.6d648f.1: type = facet_type <@ModAssignWith.1, @ModAssignWith.1(%Other)> [symbolic]
// CHECK:STDOUT:   %Self.aac87d.1: %ModAssignWith.type.6d648f.1 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %ModAssignWith.assoc_type.0beb3b.1: type = assoc_entity_type @ModAssignWith.1, @ModAssignWith.1(%Other) [symbolic]
// CHECK:STDOUT:   %ModAssignWith.WithSelf.Op.type.d01472.1: type = fn_type @ModAssignWith.WithSelf.Op, @ModAssignWith.WithSelf(%Other, %Self.aac87d.1) [symbolic]
// CHECK:STDOUT:   %ModAssignWith.WithSelf.Op.6c6076.1: %ModAssignWith.WithSelf.Op.type.d01472.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %ModAssignWith.type.6d648f.2: type = facet_type <@ModAssignWith.1, @ModAssignWith.1(%U.67d)> [symbolic]
// CHECK:STDOUT:   %pattern_type.9b9: type = pattern_type %ModAssignWith.type.6d648f.2 [symbolic]
// CHECK:STDOUT:   %T.aac: %ModAssignWith.type.6d648f.2 = symbolic_binding T, 1 [symbolic]
// CHECK:STDOUT:   %ModAssignWith.assoc_type.0beb3b.2: type = assoc_entity_type @ModAssignWith.1, @ModAssignWith.1(%U.67d) [symbolic]
// CHECK:STDOUT:   %assoc0.6957e8.2: %ModAssignWith.assoc_type.0beb3b.2 = assoc_entity element0, imports.%Core.import_ref.a9d [symbolic]
// CHECK:STDOUT:   %T.as_type.ab6: type = facet_access_type %T.aac [symbolic]
// CHECK:STDOUT:   %pattern_type.102a46.2: type = pattern_type %T.as_type.ab6 [symbolic]
// CHECK:STDOUT:   %require_complete.869: <witness> = require_complete_type %ModAssignWith.type.6d648f.2 [symbolic]
// CHECK:STDOUT:   %assoc0.bd1: %ModAssignWith.assoc_type.0beb3b.1 = assoc_entity element0, imports.%Core.import_ref.3c9 [symbolic]
// CHECK:STDOUT:   %ModAssignWith.lookup_impl_witness: <witness> = lookup_impl_witness %T.aac, @ModAssignWith.1, @ModAssignWith.1(%U.67d) [symbolic]
// CHECK:STDOUT:   %ModAssignWith.WithSelf.Op.type.d01472.3: type = fn_type @ModAssignWith.WithSelf.Op, @ModAssignWith.WithSelf(%U.67d, %T.aac) [symbolic]
// CHECK:STDOUT:   %.0dc: type = fn_type_with_self_type %ModAssignWith.WithSelf.Op.type.d01472.3, %T.aac [symbolic]
// CHECK:STDOUT:   %impl.elem0.74f: %.0dc = impl_witness_access %ModAssignWith.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.875: <specific function> = specific_impl_function %impl.elem0.74f, @ModAssignWith.WithSelf.Op(%U.67d, %T.aac) [symbolic]
// CHECK:STDOUT:   %AddAssignWith.type.002: type = facet_type <@AddAssignWith.1, @AddAssignWith.1(%i16)> [concrete]
// CHECK:STDOUT:   %AddAssignWith.assoc_type.a10: type = assoc_entity_type @AddAssignWith.1, @AddAssignWith.1(%i16) [concrete]
// CHECK:STDOUT:   %assoc0.e25: %AddAssignWith.assoc_type.a10 = assoc_entity element0, imports.%Core.import_ref.073 [concrete]
// CHECK:STDOUT:   %Int16.Op.type.da5d51.13: type = fn_type @Int16.Op.13 [concrete]
// CHECK:STDOUT:   %Int16.Op.7ce432.13: %Int16.Op.type.da5d51.13 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.9ba: <witness> = custom_witness (%Int16.Op.7ce432.13), @AddAssignWith.1, @AddAssignWith.1(%i16) [concrete]
// CHECK:STDOUT:   %AddAssignWith.facet.52e: %AddAssignWith.type.002 = facet_value %Int16, (%custom_witness.9ba) [concrete]
// CHECK:STDOUT:   %pattern_type.866: type = pattern_type %AddAssignWith.type.002 [concrete]
// CHECK:STDOUT:   %SubAssignWith.type.08c: type = facet_type <@SubAssignWith.1, @SubAssignWith.1(%i16)> [concrete]
// CHECK:STDOUT:   %SubAssignWith.assoc_type.57e: type = assoc_entity_type @SubAssignWith.1, @SubAssignWith.1(%i16) [concrete]
// CHECK:STDOUT:   %assoc0.acf: %SubAssignWith.assoc_type.57e = assoc_entity element0, imports.%Core.import_ref.fc8 [concrete]
// CHECK:STDOUT:   %Int16.Op.type.da5d51.14: type = fn_type @Int16.Op.14 [concrete]
// CHECK:STDOUT:   %Int16.Op.7ce432.14: %Int16.Op.type.da5d51.14 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.6ff: <witness> = custom_witness (%Int16.Op.7ce432.14), @SubAssignWith.1, @SubAssignWith.1(%i16) [concrete]
// CHECK:STDOUT:   %SubAssignWith.facet.a61: %SubAssignWith.type.08c = facet_value %Int16, (%custom_witness.6ff) [concrete]
// CHECK:STDOUT:   %pattern_type.a60: type = pattern_type %SubAssignWith.type.08c [concrete]
// CHECK:STDOUT:   %MulAssignWith.type.767: type = facet_type <@MulAssignWith.1, @MulAssignWith.1(%i16)> [concrete]
// CHECK:STDOUT:   %MulAssignWith.assoc_type.c88: type = assoc_entity_type @MulAssignWith.1, @MulAssignWith.1(%i16) [concrete]
// CHECK:STDOUT:   %assoc0.bb1: %MulAssignWith.assoc_type.c88 = assoc_entity element0, imports.%Core.import_ref.303 [concrete]
// CHECK:STDOUT:   %Int16.Op.type.da5d51.15: type = fn_type @Int16.Op.15 [concrete]
// CHECK:STDOUT:   %Int16.Op.7ce432.15: %Int16.Op.type.da5d51.15 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.1dc: <witness> = custom_witness (%Int16.Op.7ce432.15), @MulAssignWith.1, @MulAssignWith.1(%i16) [concrete]
// CHECK:STDOUT:   %MulAssignWith.facet.253: %MulAssignWith.type.767 = facet_value %Int16, (%custom_witness.1dc) [concrete]
// CHECK:STDOUT:   %pattern_type.eb7: type = pattern_type %MulAssignWith.type.767 [concrete]
// CHECK:STDOUT:   %DivAssignWith.type.61f: type = facet_type <@DivAssignWith.1, @DivAssignWith.1(%i16)> [concrete]
// CHECK:STDOUT:   %DivAssignWith.assoc_type.73f: type = assoc_entity_type @DivAssignWith.1, @DivAssignWith.1(%i16) [concrete]
// CHECK:STDOUT:   %assoc0.e4b: %DivAssignWith.assoc_type.73f = assoc_entity element0, imports.%Core.import_ref.52b [concrete]
// CHECK:STDOUT:   %Int16.Op.type.da5d51.16: type = fn_type @Int16.Op.16 [concrete]
// CHECK:STDOUT:   %Int16.Op.7ce432.16: %Int16.Op.type.da5d51.16 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.780: <witness> = custom_witness (%Int16.Op.7ce432.16), @DivAssignWith.1, @DivAssignWith.1(%i16) [concrete]
// CHECK:STDOUT:   %DivAssignWith.facet.14d: %DivAssignWith.type.61f = facet_value %Int16, (%custom_witness.780) [concrete]
// CHECK:STDOUT:   %pattern_type.bfb: type = pattern_type %DivAssignWith.type.61f [concrete]
// CHECK:STDOUT:   %ModAssignWith.type.48f: type = facet_type <@ModAssignWith.1, @ModAssignWith.1(%i16)> [concrete]
// CHECK:STDOUT:   %ModAssignWith.assoc_type.c84: type = assoc_entity_type @ModAssignWith.1, @ModAssignWith.1(%i16) [concrete]
// CHECK:STDOUT:   %assoc0.2c9: %ModAssignWith.assoc_type.c84 = assoc_entity element0, imports.%Core.import_ref.a9d [concrete]
// CHECK:STDOUT:   %Int16.Op.type.da5d51.17: type = fn_type @Int16.Op.17 [concrete]
// CHECK:STDOUT:   %Int16.Op.7ce432.17: %Int16.Op.type.da5d51.17 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.4e8: <witness> = custom_witness (%Int16.Op.7ce432.17), @ModAssignWith.1, @ModAssignWith.1(%i16) [concrete]
// CHECK:STDOUT:   %ModAssignWith.facet.8a8: %ModAssignWith.type.48f = facet_value %Int16, (%custom_witness.4e8) [concrete]
// CHECK:STDOUT:   %pattern_type.dbf: type = pattern_type %ModAssignWith.type.48f [concrete]
// CHECK:STDOUT:   %AddAssignWith.type.eee: type = facet_type <@AddAssignWith.1, @AddAssignWith.1(%Int32)> [concrete]
// CHECK:STDOUT:   %AddAssignWith.assoc_type.7f0: type = assoc_entity_type @AddAssignWith.1, @AddAssignWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %assoc0.b73: %AddAssignWith.assoc_type.7f0 = assoc_entity element0, imports.%Core.import_ref.073 [concrete]
// CHECK:STDOUT:   %Int32.Op.type.da5d51.12: type = fn_type @Int32.Op.12 [concrete]
// CHECK:STDOUT:   %Int32.Op.7ce432.12: %Int32.Op.type.da5d51.12 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.2be94e.1: <witness> = custom_witness (%Int32.Op.7ce432.12), @AddAssignWith.1, @AddAssignWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %AddAssignWith.facet.d99: %AddAssignWith.type.eee = facet_value %Int32, (%custom_witness.2be94e.1) [concrete]
// CHECK:STDOUT:   %pattern_type.e31: type = pattern_type %AddAssignWith.type.eee [concrete]
// CHECK:STDOUT:   %SubAssignWith.type.eed: type = facet_type <@SubAssignWith.1, @SubAssignWith.1(%Int32)> [concrete]
// CHECK:STDOUT:   %SubAssignWith.assoc_type.0dd: type = assoc_entity_type @SubAssignWith.1, @SubAssignWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %assoc0.d09: %SubAssignWith.assoc_type.0dd = assoc_entity element0, imports.%Core.import_ref.fc8 [concrete]
// CHECK:STDOUT:   %Int32.Op.type.da5d51.14: type = fn_type @Int32.Op.14 [concrete]
// CHECK:STDOUT:   %Int32.Op.7ce432.14: %Int32.Op.type.da5d51.14 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.3ff7e3.1: <witness> = custom_witness (%Int32.Op.7ce432.14), @SubAssignWith.1, @SubAssignWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %SubAssignWith.facet.8ee: %SubAssignWith.type.eed = facet_value %Int32, (%custom_witness.3ff7e3.1) [concrete]
// CHECK:STDOUT:   %pattern_type.1aa: type = pattern_type %SubAssignWith.type.eed [concrete]
// CHECK:STDOUT:   %MulAssignWith.type.8e6: type = facet_type <@MulAssignWith.1, @MulAssignWith.1(%Int32)> [concrete]
// CHECK:STDOUT:   %MulAssignWith.assoc_type.4c9: type = assoc_entity_type @MulAssignWith.1, @MulAssignWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %assoc0.4fa: %MulAssignWith.assoc_type.4c9 = assoc_entity element0, imports.%Core.import_ref.303 [concrete]
// CHECK:STDOUT:   %Int32.Op.type.da5d51.15: type = fn_type @Int32.Op.15 [concrete]
// CHECK:STDOUT:   %Int32.Op.7ce432.15: %Int32.Op.type.da5d51.15 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.a36f66.1: <witness> = custom_witness (%Int32.Op.7ce432.15), @MulAssignWith.1, @MulAssignWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %MulAssignWith.facet.7ec: %MulAssignWith.type.8e6 = facet_value %Int32, (%custom_witness.a36f66.1) [concrete]
// CHECK:STDOUT:   %pattern_type.a97: type = pattern_type %MulAssignWith.type.8e6 [concrete]
// CHECK:STDOUT:   %DivAssignWith.type.f58: type = facet_type <@DivAssignWith.1, @DivAssignWith.1(%Int32)> [concrete]
// CHECK:STDOUT:   %DivAssignWith.assoc_type.1f3: type = assoc_entity_type @DivAssignWith.1, @DivAssignWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %assoc0.550: %DivAssignWith.assoc_type.1f3 = assoc_entity element0, imports.%Core.import_ref.52b [concrete]
// CHECK:STDOUT:   %Int32.Op.type.da5d51.16: type = fn_type @Int32.Op.16 [concrete]
// CHECK:STDOUT:   %Int32.Op.7ce432.16: %Int32.Op.type.da5d51.16 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.8bd5d4.1: <witness> = custom_witness (%Int32.Op.7ce432.16), @DivAssignWith.1, @DivAssignWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %DivAssignWith.facet.a84: %DivAssignWith.type.f58 = facet_value %Int32, (%custom_witness.8bd5d4.1) [concrete]
// CHECK:STDOUT:   %pattern_type.2fa: type = pattern_type %DivAssignWith.type.f58 [concrete]
// CHECK:STDOUT:   %ModAssignWith.type.9a3: type = facet_type <@ModAssignWith.1, @ModAssignWith.1(%Int32)> [concrete]
// CHECK:STDOUT:   %ModAssignWith.assoc_type.11d: type = assoc_entity_type @ModAssignWith.1, @ModAssignWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %assoc0.5df: %ModAssignWith.assoc_type.11d = assoc_entity element0, imports.%Core.import_ref.a9d [concrete]
// CHECK:STDOUT:   %Int32.Op.type.da5d51.17: type = fn_type @Int32.Op.17 [concrete]
// CHECK:STDOUT:   %Int32.Op.7ce432.17: %Int32.Op.type.da5d51.17 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.8c8a80.1: <witness> = custom_witness (%Int32.Op.7ce432.17), @ModAssignWith.1, @ModAssignWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %ModAssignWith.facet.0d2: %ModAssignWith.type.9a3 = facet_value %Int32, (%custom_witness.8c8a80.1) [concrete]
// CHECK:STDOUT:   %pattern_type.a2a: type = pattern_type %ModAssignWith.type.9a3 [concrete]
// CHECK:STDOUT:   %Int64.Op.type.da5d51.2: type = fn_type @Int64.Op.2 [concrete]
// CHECK:STDOUT:   %Int64.Op.7ce432.2: %Int64.Op.type.da5d51.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.2be94e.2: <witness> = custom_witness (%Int64.Op.7ce432.2), @AddAssignWith.1, @AddAssignWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %AddAssignWith.facet.2a2: %AddAssignWith.type.eee = facet_value %Int64, (%custom_witness.2be94e.2) [concrete]
// CHECK:STDOUT:   %Int64.Op.type.da5d51.4: type = fn_type @Int64.Op.4 [concrete]
// CHECK:STDOUT:   %Int64.Op.7ce432.4: %Int64.Op.type.da5d51.4 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.3ff7e3.2: <witness> = custom_witness (%Int64.Op.7ce432.4), @SubAssignWith.1, @SubAssignWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %SubAssignWith.facet.839: %SubAssignWith.type.eed = facet_value %Int64, (%custom_witness.3ff7e3.2) [concrete]
// CHECK:STDOUT:   %Int64.Op.type.da5d51.5: type = fn_type @Int64.Op.5 [concrete]
// CHECK:STDOUT:   %Int64.Op.7ce432.5: %Int64.Op.type.da5d51.5 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.a36f66.2: <witness> = custom_witness (%Int64.Op.7ce432.5), @MulAssignWith.1, @MulAssignWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %MulAssignWith.facet.19a: %MulAssignWith.type.8e6 = facet_value %Int64, (%custom_witness.a36f66.2) [concrete]
// CHECK:STDOUT:   %Int64.Op.type.da5d51.6: type = fn_type @Int64.Op.6 [concrete]
// CHECK:STDOUT:   %Int64.Op.7ce432.6: %Int64.Op.type.da5d51.6 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.8bd5d4.2: <witness> = custom_witness (%Int64.Op.7ce432.6), @DivAssignWith.1, @DivAssignWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %DivAssignWith.facet.d95: %DivAssignWith.type.f58 = facet_value %Int64, (%custom_witness.8bd5d4.2) [concrete]
// CHECK:STDOUT:   %Int64.Op.type.da5d51.7: type = fn_type @Int64.Op.7 [concrete]
// CHECK:STDOUT:   %Int64.Op.7ce432.7: %Int64.Op.type.da5d51.7 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.8c8a80.2: <witness> = custom_witness (%Int64.Op.7ce432.7), @ModAssignWith.1, @ModAssignWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %ModAssignWith.facet.a8a: %ModAssignWith.type.9a3 = facet_value %Int64, (%custom_witness.8c8a80.2) [concrete]
// CHECK:STDOUT:   %Inc.type: type = facet_type <@Inc> [concrete]
// CHECK:STDOUT:   %T.3d6: %Inc.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.ec5: type = facet_access_type %T.3d6 [symbolic]
// CHECK:STDOUT:   %pattern_type.657b3b.2: type = pattern_type %T.as_type.ec5 [symbolic]
// CHECK:STDOUT:   %Inc.lookup_impl_witness: <witness> = lookup_impl_witness %T.3d6, @Inc [symbolic]
// CHECK:STDOUT:   %Inc.WithSelf.Op.type.667000.2: type = fn_type @Inc.WithSelf.Op, @Inc.WithSelf(%T.3d6) [symbolic]
// CHECK:STDOUT:   %.9f3: type = fn_type_with_self_type %Inc.WithSelf.Op.type.667000.2, %T.3d6 [symbolic]
// CHECK:STDOUT:   %impl.elem0.5bb: %.9f3 = impl_witness_access %Inc.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.4ac: <specific function> = specific_impl_function %impl.elem0.5bb, @Inc.WithSelf.Op(%T.3d6) [symbolic]
// CHECK:STDOUT:   %Dec.type: type = facet_type <@Dec> [concrete]
// CHECK:STDOUT:   %T.cee: %Dec.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.b80: type = facet_access_type %T.cee [symbolic]
// CHECK:STDOUT:   %pattern_type.a28bf9.2: type = pattern_type %T.as_type.b80 [symbolic]
// CHECK:STDOUT:   %Dec.lookup_impl_witness: <witness> = lookup_impl_witness %T.cee, @Dec [symbolic]
// CHECK:STDOUT:   %Dec.WithSelf.Op.type.97f359.2: type = fn_type @Dec.WithSelf.Op, @Dec.WithSelf(%T.cee) [symbolic]
// CHECK:STDOUT:   %.369: type = fn_type_with_self_type %Dec.WithSelf.Op.type.97f359.2, %T.cee [symbolic]
// CHECK:STDOUT:   %impl.elem0.b37: %.369 = impl_witness_access %Dec.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.afd: <specific function> = specific_impl_function %impl.elem0.b37, @Dec.WithSelf.Op(%T.cee) [symbolic]
// CHECK:STDOUT:   %Negate.type: type = facet_type <@Negate> [concrete]
// CHECK:STDOUT:   %facet_type.9f5: type = facet_type <@Destroy & @Negate> [concrete]
// CHECK:STDOUT:   %.Self.28d: %facet_type.9f5 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %Negate.lookup_impl_witness.04a: <witness> = lookup_impl_witness %.Self.28d, @Negate [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.0c6: type = impl_witness_access %Negate.lookup_impl_witness.04a, element0 [symbolic_self]
// CHECK:STDOUT:   %facet_type.5a6: type = facet_type <@Destroy & @Negate where %impl.elem0.0c6 = %Result.as_type> [symbolic]
// CHECK:STDOUT:   %pattern_type.d0c: type = pattern_type %facet_type.5a6 [symbolic]
// CHECK:STDOUT:   %T.ec8: %facet_type.5a6 = symbolic_binding T, 1 [symbolic]
// CHECK:STDOUT:   %T.as_type.55b: type = facet_access_type %T.ec8 [symbolic]
// CHECK:STDOUT:   %pattern_type.5a7: type = pattern_type %T.as_type.55b [symbolic]
// CHECK:STDOUT:   %Negate.lookup_impl_witness.139: <witness> = lookup_impl_witness %T.ec8, @Negate [symbolic]
// CHECK:STDOUT:   %Negate.facet.6d8: %Negate.type = facet_value %T.as_type.55b, (%Negate.lookup_impl_witness.139) [symbolic]
// CHECK:STDOUT:   %Negate.WithSelf.Op.type.bcd: type = fn_type @Negate.WithSelf.Op, @Negate.WithSelf(%Negate.facet.6d8) [symbolic]
// CHECK:STDOUT:   %.7a2: type = fn_type_with_self_type %Negate.WithSelf.Op.type.bcd, %Negate.facet.6d8 [symbolic]
// CHECK:STDOUT:   %impl.elem1.2c5: %.7a2 = impl_witness_access %Negate.lookup_impl_witness.139, element1 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.cfb: <specific function> = specific_impl_function %impl.elem1.2c5, @Negate.WithSelf.Op(%Negate.facet.6d8) [symbolic]
// CHECK:STDOUT:   %Int32.Op.type.da5d51.18: type = fn_type @Int32.Op.18 [concrete]
// CHECK:STDOUT:   %Int32.Op.7ce432.18: %Int32.Op.type.da5d51.18 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.696: <witness> = custom_witness (%Int32.Op.7ce432.18), @Inc [concrete]
// CHECK:STDOUT:   %Inc.facet.353: %Inc.type = facet_value %Int32, (%custom_witness.696) [concrete]
// CHECK:STDOUT:   %Int32.Op.type.da5d51.19: type = fn_type @Int32.Op.19 [concrete]
// CHECK:STDOUT:   %Int32.Op.7ce432.19: %Int32.Op.type.da5d51.19 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.ff1: <witness> = custom_witness (%Int32.Op.7ce432.19), @Dec [concrete]
// CHECK:STDOUT:   %Dec.facet.b75: %Dec.type = facet_value %Int32, (%custom_witness.ff1) [concrete]
// CHECK:STDOUT:   %facet_type.064: type = facet_type <@Destroy & @Negate where %impl.elem0.0c6 = %Int64> [concrete]
// CHECK:STDOUT:   %Int16.Op.type.da5d51.18: type = fn_type @Int16.Op.18 [concrete]
// CHECK:STDOUT:   %Int16.Op.7ce432.18: %Int16.Op.type.da5d51.18 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.1a0: <witness> = custom_witness (%Int64, %Int16.Op.7ce432.18), @Negate [concrete]
// CHECK:STDOUT:   %facet_value.9ca: %facet_type.064 = facet_value %Int16, (%custom_witness.ecd0a5.1, %custom_witness.1a0) [concrete]
// CHECK:STDOUT:   %pattern_type.328: type = pattern_type %facet_type.064 [concrete]
// CHECK:STDOUT:   %facet_type.ff4: type = facet_type <@Destroy & @Negate where %impl.elem0.0c6 = %Int32> [concrete]
// CHECK:STDOUT:   %Int32.Op.type.da5d51.20: type = fn_type @Int32.Op.20 [concrete]
// CHECK:STDOUT:   %Int32.Op.7ce432.20: %Int32.Op.type.da5d51.20 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.3db: <witness> = custom_witness (%Int32, %Int32.Op.7ce432.20), @Negate [concrete]
// CHECK:STDOUT:   %facet_value.620: %facet_type.ff4 = facet_value %Int32, (%custom_witness.ecd0a5.2, %custom_witness.3db) [concrete]
// CHECK:STDOUT:   %pattern_type.2af: type = pattern_type %facet_type.ff4 [concrete]
// CHECK:STDOUT:   %complete_type.f1a: <witness> = complete_type_witness %AddWith.type.41f [concrete]
// CHECK:STDOUT:   %AddWith.facet.02b: %AddWith.type.41f = facet_value %i16, (%custom_witness.069) [concrete]
// CHECK:STDOUT:   %AddWith.WithSelf.Op.type.bba: type = fn_type @AddWith.WithSelf.Op, @AddWith.WithSelf(%Int16, %AddWith.facet.02b) [concrete]
// CHECK:STDOUT:   %.130: type = fn_type_with_self_type %AddWith.WithSelf.Op.type.bba, %AddWith.facet.02b [concrete]
// CHECK:STDOUT:   %complete_type.c17: <witness> = complete_type_witness %SubWith.type.f7a [concrete]
// CHECK:STDOUT:   %SubWith.facet.f1c: %SubWith.type.f7a = facet_value %i16, (%custom_witness.2c8) [concrete]
// CHECK:STDOUT:   %SubWith.WithSelf.Op.type.4dd: type = fn_type @SubWith.WithSelf.Op, @SubWith.WithSelf(%Int16, %SubWith.facet.f1c) [concrete]
// CHECK:STDOUT:   %.859: type = fn_type_with_self_type %SubWith.WithSelf.Op.type.4dd, %SubWith.facet.f1c [concrete]
// CHECK:STDOUT:   %complete_type.b33: <witness> = complete_type_witness %MulWith.type.ebe [concrete]
// CHECK:STDOUT:   %MulWith.facet.f7a: %MulWith.type.ebe = facet_value %i16, (%custom_witness.23bf) [concrete]
// CHECK:STDOUT:   %MulWith.WithSelf.Op.type.358: type = fn_type @MulWith.WithSelf.Op, @MulWith.WithSelf(%Int16, %MulWith.facet.f7a) [concrete]
// CHECK:STDOUT:   %.833: type = fn_type_with_self_type %MulWith.WithSelf.Op.type.358, %MulWith.facet.f7a [concrete]
// CHECK:STDOUT:   %complete_type.691: <witness> = complete_type_witness %DivWith.type.996 [concrete]
// CHECK:STDOUT:   %DivWith.facet.d55: %DivWith.type.996 = facet_value %i16, (%custom_witness.ff6) [concrete]
// CHECK:STDOUT:   %DivWith.WithSelf.Op.type.822: type = fn_type @DivWith.WithSelf.Op, @DivWith.WithSelf(%Int16, %DivWith.facet.d55) [concrete]
// CHECK:STDOUT:   %.65a: type = fn_type_with_self_type %DivWith.WithSelf.Op.type.822, %DivWith.facet.d55 [concrete]
// CHECK:STDOUT:   %complete_type.3a6: <witness> = complete_type_witness %ModWith.type.d4b [concrete]
// CHECK:STDOUT:   %ModWith.facet.bab: %ModWith.type.d4b = facet_value %i16, (%custom_witness.74e) [concrete]
// CHECK:STDOUT:   %ModWith.WithSelf.Op.type.f9f2: type = fn_type @ModWith.WithSelf.Op, @ModWith.WithSelf(%Int16, %ModWith.facet.bab) [concrete]
// CHECK:STDOUT:   %.ea5: type = fn_type_with_self_type %ModWith.WithSelf.Op.type.f9f2, %ModWith.facet.bab [concrete]
// CHECK:STDOUT:   %complete_type.aae5: <witness> = complete_type_witness %AddWith.type.c9a [concrete]
// CHECK:STDOUT:   %AddWith.facet.9a7: %AddWith.type.c9a = facet_value %Int16, (%custom_witness.067) [concrete]
// CHECK:STDOUT:   %AddWith.WithSelf.Op.type.d41: type = fn_type @AddWith.WithSelf.Op, @AddWith.WithSelf(%i16, %AddWith.facet.9a7) [concrete]
// CHECK:STDOUT:   %.a6b: type = fn_type_with_self_type %AddWith.WithSelf.Op.type.d41, %AddWith.facet.9a7 [concrete]
// CHECK:STDOUT:   %complete_type.79d: <witness> = complete_type_witness %SubWith.type.191 [concrete]
// CHECK:STDOUT:   %SubWith.facet.e26: %SubWith.type.191 = facet_value %Int16, (%custom_witness.1e0) [concrete]
// CHECK:STDOUT:   %SubWith.WithSelf.Op.type.098: type = fn_type @SubWith.WithSelf.Op, @SubWith.WithSelf(%i16, %SubWith.facet.e26) [concrete]
// CHECK:STDOUT:   %.ee1: type = fn_type_with_self_type %SubWith.WithSelf.Op.type.098, %SubWith.facet.e26 [concrete]
// CHECK:STDOUT:   %complete_type.4fe: <witness> = complete_type_witness %MulWith.type.df1 [concrete]
// CHECK:STDOUT:   %MulWith.facet.df1: %MulWith.type.df1 = facet_value %Int16, (%custom_witness.631) [concrete]
// CHECK:STDOUT:   %MulWith.WithSelf.Op.type.5a5: type = fn_type @MulWith.WithSelf.Op, @MulWith.WithSelf(%i16, %MulWith.facet.df1) [concrete]
// CHECK:STDOUT:   %.0a1: type = fn_type_with_self_type %MulWith.WithSelf.Op.type.5a5, %MulWith.facet.df1 [concrete]
// CHECK:STDOUT:   %complete_type.377: <witness> = complete_type_witness %DivWith.type.957 [concrete]
// CHECK:STDOUT:   %DivWith.facet.243: %DivWith.type.957 = facet_value %Int16, (%custom_witness.64e) [concrete]
// CHECK:STDOUT:   %DivWith.WithSelf.Op.type.644: type = fn_type @DivWith.WithSelf.Op, @DivWith.WithSelf(%i16, %DivWith.facet.243) [concrete]
// CHECK:STDOUT:   %.66b: type = fn_type_with_self_type %DivWith.WithSelf.Op.type.644, %DivWith.facet.243 [concrete]
// CHECK:STDOUT:   %complete_type.662: <witness> = complete_type_witness %ModWith.type.0ea [concrete]
// CHECK:STDOUT:   %ModWith.facet.37f: %ModWith.type.0ea = facet_value %Int16, (%custom_witness.394) [concrete]
// CHECK:STDOUT:   %ModWith.WithSelf.Op.type.2f8: type = fn_type @ModWith.WithSelf.Op, @ModWith.WithSelf(%i16, %ModWith.facet.37f) [concrete]
// CHECK:STDOUT:   %.1bc: type = fn_type_with_self_type %ModWith.WithSelf.Op.type.2f8, %ModWith.facet.37f [concrete]
// CHECK:STDOUT:   %complete_type.e69: <witness> = complete_type_witness %AddWith.type.4f1 [concrete]
// CHECK:STDOUT:   %AddWith.facet.cea: %AddWith.type.4f1 = facet_value %Int16, (%custom_witness.bb0e2d.1) [concrete]
// CHECK:STDOUT:   %AddWith.WithSelf.Op.type.f93: type = fn_type @AddWith.WithSelf.Op, @AddWith.WithSelf(%Int32, %AddWith.facet.cea) [concrete]
// CHECK:STDOUT:   %.886: type = fn_type_with_self_type %AddWith.WithSelf.Op.type.f93, %AddWith.facet.cea [concrete]
// CHECK:STDOUT:   %complete_type.c40: <witness> = complete_type_witness %SubWith.type.40d [concrete]
// CHECK:STDOUT:   %SubWith.facet.152: %SubWith.type.40d = facet_value %Int16, (%custom_witness.dee291.1) [concrete]
// CHECK:STDOUT:   %SubWith.WithSelf.Op.type.15f: type = fn_type @SubWith.WithSelf.Op, @SubWith.WithSelf(%Int32, %SubWith.facet.152) [concrete]
// CHECK:STDOUT:   %.18a: type = fn_type_with_self_type %SubWith.WithSelf.Op.type.15f, %SubWith.facet.152 [concrete]
// CHECK:STDOUT:   %complete_type.ab9: <witness> = complete_type_witness %MulWith.type.8d1 [concrete]
// CHECK:STDOUT:   %MulWith.facet.e4c: %MulWith.type.8d1 = facet_value %Int16, (%custom_witness.db8ceb.1) [concrete]
// CHECK:STDOUT:   %MulWith.WithSelf.Op.type.ac5: type = fn_type @MulWith.WithSelf.Op, @MulWith.WithSelf(%Int32, %MulWith.facet.e4c) [concrete]
// CHECK:STDOUT:   %.d07: type = fn_type_with_self_type %MulWith.WithSelf.Op.type.ac5, %MulWith.facet.e4c [concrete]
// CHECK:STDOUT:   %complete_type.0f3: <witness> = complete_type_witness %DivWith.type.a2b [concrete]
// CHECK:STDOUT:   %DivWith.facet.37d: %DivWith.type.a2b = facet_value %Int16, (%custom_witness.deca3a.1) [concrete]
// CHECK:STDOUT:   %DivWith.WithSelf.Op.type.811: type = fn_type @DivWith.WithSelf.Op, @DivWith.WithSelf(%Int32, %DivWith.facet.37d) [concrete]
// CHECK:STDOUT:   %.d58: type = fn_type_with_self_type %DivWith.WithSelf.Op.type.811, %DivWith.facet.37d [concrete]
// CHECK:STDOUT:   %complete_type.fae: <witness> = complete_type_witness %ModWith.type.cff [concrete]
// CHECK:STDOUT:   %ModWith.facet.ac1: %ModWith.type.cff = facet_value %Int16, (%custom_witness.c72db9.1) [concrete]
// CHECK:STDOUT:   %ModWith.WithSelf.Op.type.f9f5: type = fn_type @ModWith.WithSelf.Op, @ModWith.WithSelf(%Int32, %ModWith.facet.ac1) [concrete]
// CHECK:STDOUT:   %.37d: type = fn_type_with_self_type %ModWith.WithSelf.Op.type.f9f5, %ModWith.facet.ac1 [concrete]
// CHECK:STDOUT:   %AddWith.facet.82f: %AddWith.type.41f = facet_value %Int32, (%custom_witness.8d5) [concrete]
// CHECK:STDOUT:   %AddWith.WithSelf.Op.type.28d: type = fn_type @AddWith.WithSelf.Op, @AddWith.WithSelf(%Int16, %AddWith.facet.82f) [concrete]
// CHECK:STDOUT:   %.e93: type = fn_type_with_self_type %AddWith.WithSelf.Op.type.28d, %AddWith.facet.82f [concrete]
// CHECK:STDOUT:   %SubWith.facet.dfe: %SubWith.type.f7a = facet_value %Int32, (%custom_witness.14c) [concrete]
// CHECK:STDOUT:   %SubWith.WithSelf.Op.type.685: type = fn_type @SubWith.WithSelf.Op, @SubWith.WithSelf(%Int16, %SubWith.facet.dfe) [concrete]
// CHECK:STDOUT:   %.d09: type = fn_type_with_self_type %SubWith.WithSelf.Op.type.685, %SubWith.facet.dfe [concrete]
// CHECK:STDOUT:   %MulWith.facet.7de: %MulWith.type.ebe = facet_value %Int32, (%custom_witness.fd0) [concrete]
// CHECK:STDOUT:   %MulWith.WithSelf.Op.type.5bf: type = fn_type @MulWith.WithSelf.Op, @MulWith.WithSelf(%Int16, %MulWith.facet.7de) [concrete]
// CHECK:STDOUT:   %.0d2: type = fn_type_with_self_type %MulWith.WithSelf.Op.type.5bf, %MulWith.facet.7de [concrete]
// CHECK:STDOUT:   %DivWith.facet.021: %DivWith.type.996 = facet_value %Int32, (%custom_witness.3a3) [concrete]
// CHECK:STDOUT:   %DivWith.WithSelf.Op.type.8bc: type = fn_type @DivWith.WithSelf.Op, @DivWith.WithSelf(%Int16, %DivWith.facet.021) [concrete]
// CHECK:STDOUT:   %.715: type = fn_type_with_self_type %DivWith.WithSelf.Op.type.8bc, %DivWith.facet.021 [concrete]
// CHECK:STDOUT:   %ModWith.facet.481: %ModWith.type.d4b = facet_value %Int32, (%custom_witness.e15) [concrete]
// CHECK:STDOUT:   %ModWith.WithSelf.Op.type.efe: type = fn_type @ModWith.WithSelf.Op, @ModWith.WithSelf(%Int16, %ModWith.facet.481) [concrete]
// CHECK:STDOUT:   %.942: type = fn_type_with_self_type %ModWith.WithSelf.Op.type.efe, %ModWith.facet.481 [concrete]
// CHECK:STDOUT:   %AddWith.facet.5a2: %AddWith.type.4f1 = facet_value %Int32, (%custom_witness.bb0e2d.2) [concrete]
// CHECK:STDOUT:   %AddWith.WithSelf.Op.type.b30: type = fn_type @AddWith.WithSelf.Op, @AddWith.WithSelf(%Int32, %AddWith.facet.5a2) [concrete]
// CHECK:STDOUT:   %.82a: type = fn_type_with_self_type %AddWith.WithSelf.Op.type.b30, %AddWith.facet.5a2 [concrete]
// CHECK:STDOUT:   %SubWith.facet.a57: %SubWith.type.40d = facet_value %Int32, (%custom_witness.dee291.2) [concrete]
// CHECK:STDOUT:   %SubWith.WithSelf.Op.type.cb0: type = fn_type @SubWith.WithSelf.Op, @SubWith.WithSelf(%Int32, %SubWith.facet.a57) [concrete]
// CHECK:STDOUT:   %.634: type = fn_type_with_self_type %SubWith.WithSelf.Op.type.cb0, %SubWith.facet.a57 [concrete]
// CHECK:STDOUT:   %MulWith.facet.3b9: %MulWith.type.8d1 = facet_value %Int32, (%custom_witness.db8ceb.2) [concrete]
// CHECK:STDOUT:   %MulWith.WithSelf.Op.type.c67: type = fn_type @MulWith.WithSelf.Op, @MulWith.WithSelf(%Int32, %MulWith.facet.3b9) [concrete]
// CHECK:STDOUT:   %.e9a: type = fn_type_with_self_type %MulWith.WithSelf.Op.type.c67, %MulWith.facet.3b9 [concrete]
// CHECK:STDOUT:   %DivWith.facet.ca0: %DivWith.type.a2b = facet_value %Int32, (%custom_witness.deca3a.2) [concrete]
// CHECK:STDOUT:   %DivWith.WithSelf.Op.type.971: type = fn_type @DivWith.WithSelf.Op, @DivWith.WithSelf(%Int32, %DivWith.facet.ca0) [concrete]
// CHECK:STDOUT:   %.4dc: type = fn_type_with_self_type %DivWith.WithSelf.Op.type.971, %DivWith.facet.ca0 [concrete]
// CHECK:STDOUT:   %ModWith.facet.97e: %ModWith.type.cff = facet_value %Int32, (%custom_witness.c72db9.2) [concrete]
// CHECK:STDOUT:   %ModWith.WithSelf.Op.type.927: type = fn_type @ModWith.WithSelf.Op, @ModWith.WithSelf(%Int32, %ModWith.facet.97e) [concrete]
// CHECK:STDOUT:   %.156: type = fn_type_with_self_type %ModWith.WithSelf.Op.type.927, %ModWith.facet.97e [concrete]
// CHECK:STDOUT:   %complete_type.477: <witness> = complete_type_witness %AddWith.type.fef [concrete]
// CHECK:STDOUT:   %AddWith.facet.19e: %AddWith.type.fef = facet_value %Int32, (%custom_witness.3f7) [concrete]
// CHECK:STDOUT:   %AddWith.WithSelf.Op.type.377: type = fn_type @AddWith.WithSelf.Op, @AddWith.WithSelf(%Int64, %AddWith.facet.19e) [concrete]
// CHECK:STDOUT:   %.16f: type = fn_type_with_self_type %AddWith.WithSelf.Op.type.377, %AddWith.facet.19e [concrete]
// CHECK:STDOUT:   %complete_type.2df: <witness> = complete_type_witness %SubWith.type.c5d [concrete]
// CHECK:STDOUT:   %SubWith.facet.9f0: %SubWith.type.c5d = facet_value %Int32, (%custom_witness.aad) [concrete]
// CHECK:STDOUT:   %SubWith.WithSelf.Op.type.492: type = fn_type @SubWith.WithSelf.Op, @SubWith.WithSelf(%Int64, %SubWith.facet.9f0) [concrete]
// CHECK:STDOUT:   %.d94: type = fn_type_with_self_type %SubWith.WithSelf.Op.type.492, %SubWith.facet.9f0 [concrete]
// CHECK:STDOUT:   %complete_type.5ad: <witness> = complete_type_witness %MulWith.type.22c [concrete]
// CHECK:STDOUT:   %MulWith.facet.af5: %MulWith.type.22c = facet_value %Int32, (%custom_witness.1d4) [concrete]
// CHECK:STDOUT:   %MulWith.WithSelf.Op.type.6cd: type = fn_type @MulWith.WithSelf.Op, @MulWith.WithSelf(%Int64, %MulWith.facet.af5) [concrete]
// CHECK:STDOUT:   %.a61: type = fn_type_with_self_type %MulWith.WithSelf.Op.type.6cd, %MulWith.facet.af5 [concrete]
// CHECK:STDOUT:   %complete_type.5d2: <witness> = complete_type_witness %DivWith.type.95f [concrete]
// CHECK:STDOUT:   %DivWith.facet.731: %DivWith.type.95f = facet_value %Int32, (%custom_witness.515) [concrete]
// CHECK:STDOUT:   %DivWith.WithSelf.Op.type.614: type = fn_type @DivWith.WithSelf.Op, @DivWith.WithSelf(%Int64, %DivWith.facet.731) [concrete]
// CHECK:STDOUT:   %.448: type = fn_type_with_self_type %DivWith.WithSelf.Op.type.614, %DivWith.facet.731 [concrete]
// CHECK:STDOUT:   %complete_type.d30: <witness> = complete_type_witness %ModWith.type.e6e [concrete]
// CHECK:STDOUT:   %ModWith.facet.fff: %ModWith.type.e6e = facet_value %Int32, (%custom_witness.1af) [concrete]
// CHECK:STDOUT:   %ModWith.WithSelf.Op.type.de3: type = fn_type @ModWith.WithSelf.Op, @ModWith.WithSelf(%Int64, %ModWith.facet.fff) [concrete]
// CHECK:STDOUT:   %.401: type = fn_type_with_self_type %ModWith.WithSelf.Op.type.de3, %ModWith.facet.fff [concrete]
// CHECK:STDOUT:   %AddWith.facet.1a3: %AddWith.type.4f1 = facet_value %Int64, (%custom_witness.dbe) [concrete]
// CHECK:STDOUT:   %AddWith.WithSelf.Op.type.bc9: type = fn_type @AddWith.WithSelf.Op, @AddWith.WithSelf(%Int32, %AddWith.facet.1a3) [concrete]
// CHECK:STDOUT:   %.cec: type = fn_type_with_self_type %AddWith.WithSelf.Op.type.bc9, %AddWith.facet.1a3 [concrete]
// CHECK:STDOUT:   %SubWith.facet.8e2: %SubWith.type.40d = facet_value %Int64, (%custom_witness.af3) [concrete]
// CHECK:STDOUT:   %SubWith.WithSelf.Op.type.634: type = fn_type @SubWith.WithSelf.Op, @SubWith.WithSelf(%Int32, %SubWith.facet.8e2) [concrete]
// CHECK:STDOUT:   %.72e: type = fn_type_with_self_type %SubWith.WithSelf.Op.type.634, %SubWith.facet.8e2 [concrete]
// CHECK:STDOUT:   %MulWith.facet.d5b: %MulWith.type.8d1 = facet_value %Int64, (%custom_witness.d10) [concrete]
// CHECK:STDOUT:   %MulWith.WithSelf.Op.type.21f: type = fn_type @MulWith.WithSelf.Op, @MulWith.WithSelf(%Int32, %MulWith.facet.d5b) [concrete]
// CHECK:STDOUT:   %.ff7: type = fn_type_with_self_type %MulWith.WithSelf.Op.type.21f, %MulWith.facet.d5b [concrete]
// CHECK:STDOUT:   %DivWith.facet.efb: %DivWith.type.a2b = facet_value %Int64, (%custom_witness.438) [concrete]
// CHECK:STDOUT:   %DivWith.WithSelf.Op.type.0cc: type = fn_type @DivWith.WithSelf.Op, @DivWith.WithSelf(%Int32, %DivWith.facet.efb) [concrete]
// CHECK:STDOUT:   %.291: type = fn_type_with_self_type %DivWith.WithSelf.Op.type.0cc, %DivWith.facet.efb [concrete]
// CHECK:STDOUT:   %ModWith.facet.f6a: %ModWith.type.cff = facet_value %Int64, (%custom_witness.bf8) [concrete]
// CHECK:STDOUT:   %ModWith.WithSelf.Op.type.da4: type = fn_type @ModWith.WithSelf.Op, @ModWith.WithSelf(%Int32, %ModWith.facet.f6a) [concrete]
// CHECK:STDOUT:   %.7a88: type = fn_type_with_self_type %ModWith.WithSelf.Op.type.da4, %ModWith.facet.f6a [concrete]
// CHECK:STDOUT:   %complete_type.95c: <witness> = complete_type_witness %AddAssignWith.type.002 [concrete]
// CHECK:STDOUT:   %AddAssignWith.WithSelf.Op.type.616: type = fn_type @AddAssignWith.WithSelf.Op, @AddAssignWith.WithSelf(%i16, %AddAssignWith.facet.52e) [concrete]
// CHECK:STDOUT:   %.5b6: type = fn_type_with_self_type %AddAssignWith.WithSelf.Op.type.616, %AddAssignWith.facet.52e [concrete]
// CHECK:STDOUT:   %complete_type.f27: <witness> = complete_type_witness %SubAssignWith.type.08c [concrete]
// CHECK:STDOUT:   %SubAssignWith.WithSelf.Op.type.93e: type = fn_type @SubAssignWith.WithSelf.Op, @SubAssignWith.WithSelf(%i16, %SubAssignWith.facet.a61) [concrete]
// CHECK:STDOUT:   %.49f: type = fn_type_with_self_type %SubAssignWith.WithSelf.Op.type.93e, %SubAssignWith.facet.a61 [concrete]
// CHECK:STDOUT:   %complete_type.cbb: <witness> = complete_type_witness %MulAssignWith.type.767 [concrete]
// CHECK:STDOUT:   %MulAssignWith.WithSelf.Op.type.bc0: type = fn_type @MulAssignWith.WithSelf.Op, @MulAssignWith.WithSelf(%i16, %MulAssignWith.facet.253) [concrete]
// CHECK:STDOUT:   %.c0a: type = fn_type_with_self_type %MulAssignWith.WithSelf.Op.type.bc0, %MulAssignWith.facet.253 [concrete]
// CHECK:STDOUT:   %complete_type.e4b: <witness> = complete_type_witness %DivAssignWith.type.61f [concrete]
// CHECK:STDOUT:   %DivAssignWith.WithSelf.Op.type.42f: type = fn_type @DivAssignWith.WithSelf.Op, @DivAssignWith.WithSelf(%i16, %DivAssignWith.facet.14d) [concrete]
// CHECK:STDOUT:   %.506: type = fn_type_with_self_type %DivAssignWith.WithSelf.Op.type.42f, %DivAssignWith.facet.14d [concrete]
// CHECK:STDOUT:   %complete_type.9a2: <witness> = complete_type_witness %ModAssignWith.type.48f [concrete]
// CHECK:STDOUT:   %ModAssignWith.WithSelf.Op.type.3a5: type = fn_type @ModAssignWith.WithSelf.Op, @ModAssignWith.WithSelf(%i16, %ModAssignWith.facet.8a8) [concrete]
// CHECK:STDOUT:   %.dd5: type = fn_type_with_self_type %ModAssignWith.WithSelf.Op.type.3a5, %ModAssignWith.facet.8a8 [concrete]
// CHECK:STDOUT:   %complete_type.987: <witness> = complete_type_witness %AddAssignWith.type.eee [concrete]
// CHECK:STDOUT:   %AddAssignWith.WithSelf.Op.type.e5e: type = fn_type @AddAssignWith.WithSelf.Op, @AddAssignWith.WithSelf(%Int32, %AddAssignWith.facet.d99) [concrete]
// CHECK:STDOUT:   %.353: type = fn_type_with_self_type %AddAssignWith.WithSelf.Op.type.e5e, %AddAssignWith.facet.d99 [concrete]
// CHECK:STDOUT:   %complete_type.cb4: <witness> = complete_type_witness %SubAssignWith.type.eed [concrete]
// CHECK:STDOUT:   %SubAssignWith.WithSelf.Op.type.446: type = fn_type @SubAssignWith.WithSelf.Op, @SubAssignWith.WithSelf(%Int32, %SubAssignWith.facet.8ee) [concrete]
// CHECK:STDOUT:   %.4fb: type = fn_type_with_self_type %SubAssignWith.WithSelf.Op.type.446, %SubAssignWith.facet.8ee [concrete]
// CHECK:STDOUT:   %complete_type.5a8: <witness> = complete_type_witness %MulAssignWith.type.8e6 [concrete]
// CHECK:STDOUT:   %MulAssignWith.WithSelf.Op.type.8e5: type = fn_type @MulAssignWith.WithSelf.Op, @MulAssignWith.WithSelf(%Int32, %MulAssignWith.facet.7ec) [concrete]
// CHECK:STDOUT:   %.fef: type = fn_type_with_self_type %MulAssignWith.WithSelf.Op.type.8e5, %MulAssignWith.facet.7ec [concrete]
// CHECK:STDOUT:   %complete_type.3ca: <witness> = complete_type_witness %DivAssignWith.type.f58 [concrete]
// CHECK:STDOUT:   %DivAssignWith.WithSelf.Op.type.ca3: type = fn_type @DivAssignWith.WithSelf.Op, @DivAssignWith.WithSelf(%Int32, %DivAssignWith.facet.a84) [concrete]
// CHECK:STDOUT:   %.9aa: type = fn_type_with_self_type %DivAssignWith.WithSelf.Op.type.ca3, %DivAssignWith.facet.a84 [concrete]
// CHECK:STDOUT:   %complete_type.0bd: <witness> = complete_type_witness %ModAssignWith.type.9a3 [concrete]
// CHECK:STDOUT:   %ModAssignWith.WithSelf.Op.type.39d: type = fn_type @ModAssignWith.WithSelf.Op, @ModAssignWith.WithSelf(%Int32, %ModAssignWith.facet.0d2) [concrete]
// CHECK:STDOUT:   %.35d: type = fn_type_with_self_type %ModAssignWith.WithSelf.Op.type.39d, %ModAssignWith.facet.0d2 [concrete]
// CHECK:STDOUT:   %AddAssignWith.WithSelf.Op.type.5b1: type = fn_type @AddAssignWith.WithSelf.Op, @AddAssignWith.WithSelf(%Int32, %AddAssignWith.facet.2a2) [concrete]
// CHECK:STDOUT:   %.632: type = fn_type_with_self_type %AddAssignWith.WithSelf.Op.type.5b1, %AddAssignWith.facet.2a2 [concrete]
// CHECK:STDOUT:   %SubAssignWith.WithSelf.Op.type.9fa: type = fn_type @SubAssignWith.WithSelf.Op, @SubAssignWith.WithSelf(%Int32, %SubAssignWith.facet.839) [concrete]
// CHECK:STDOUT:   %.de8: type = fn_type_with_self_type %SubAssignWith.WithSelf.Op.type.9fa, %SubAssignWith.facet.839 [concrete]
// CHECK:STDOUT:   %MulAssignWith.WithSelf.Op.type.2e7: type = fn_type @MulAssignWith.WithSelf.Op, @MulAssignWith.WithSelf(%Int32, %MulAssignWith.facet.19a) [concrete]
// CHECK:STDOUT:   %.1d5: type = fn_type_with_self_type %MulAssignWith.WithSelf.Op.type.2e7, %MulAssignWith.facet.19a [concrete]
// CHECK:STDOUT:   %DivAssignWith.WithSelf.Op.type.354: type = fn_type @DivAssignWith.WithSelf.Op, @DivAssignWith.WithSelf(%Int32, %DivAssignWith.facet.d95) [concrete]
// CHECK:STDOUT:   %.f9d: type = fn_type_with_self_type %DivAssignWith.WithSelf.Op.type.354, %DivAssignWith.facet.d95 [concrete]
// CHECK:STDOUT:   %ModAssignWith.WithSelf.Op.type.569: type = fn_type @ModAssignWith.WithSelf.Op, @ModAssignWith.WithSelf(%Int32, %ModAssignWith.facet.a8a) [concrete]
// CHECK:STDOUT:   %.fbc: type = fn_type_with_self_type %ModAssignWith.WithSelf.Op.type.569, %ModAssignWith.facet.a8a [concrete]
// CHECK:STDOUT:   %Inc.WithSelf.Op.type.ab7: type = fn_type @Inc.WithSelf.Op, @Inc.WithSelf(%Inc.facet.353) [concrete]
// CHECK:STDOUT:   %.f5b: type = fn_type_with_self_type %Inc.WithSelf.Op.type.ab7, %Inc.facet.353 [concrete]
// CHECK:STDOUT:   %Dec.WithSelf.Op.type.b72: type = fn_type @Dec.WithSelf.Op, @Dec.WithSelf(%Dec.facet.b75) [concrete]
// CHECK:STDOUT:   %.2d9: type = fn_type_with_self_type %Dec.WithSelf.Op.type.b72, %Dec.facet.b75 [concrete]
// CHECK:STDOUT:   %Negate.facet.086: %Negate.type = facet_value %Int16, (%custom_witness.1a0) [concrete]
// CHECK:STDOUT:   %Negate.WithSelf.Op.type.dff: type = fn_type @Negate.WithSelf.Op, @Negate.WithSelf(%Negate.facet.086) [concrete]
// CHECK:STDOUT:   %.db4: type = fn_type_with_self_type %Negate.WithSelf.Op.type.dff, %Negate.facet.086 [concrete]
// CHECK:STDOUT:   %Negate.facet.e21: %Negate.type = facet_value %Int32, (%custom_witness.3db) [concrete]
// CHECK:STDOUT:   %Negate.WithSelf.Op.type.384: type = fn_type @Negate.WithSelf.Op, @Negate.WithSelf(%Negate.facet.e21) [concrete]
// CHECK:STDOUT:   %.08c: type = fn_type_with_self_type %Negate.WithSelf.Op.type.384, %Negate.facet.e21 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.c92d: @AddWith.WithSelf.%AddWith.assoc_type (%AddWith.assoc_type.329) = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [symbolic = @AddWith.WithSelf.%assoc1 (constants.%assoc1.a2f)]
// CHECK:STDOUT:   %Core.import_ref.2b2: @AddWith.WithSelf.%AddWith.WithSelf.Op.type (%AddWith.WithSelf.Op.type.171) = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [symbolic = @AddWith.WithSelf.%AddWith.WithSelf.Op (constants.%AddWith.WithSelf.Op.3fe)]
// CHECK:STDOUT:   %Core.import_ref.3ad: type = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [concrete = %Result.ee3]
// CHECK:STDOUT:   %Result.ee3: type = assoc_const_decl @Result.1 [concrete] {}
// CHECK:STDOUT:   %Core.import_ref.8be = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.d04: @SubWith.WithSelf.%SubWith.assoc_type (%SubWith.assoc_type.02f) = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [symbolic = @SubWith.WithSelf.%assoc1 (constants.%assoc1.076)]
// CHECK:STDOUT:   %Core.import_ref.bc0: @SubWith.WithSelf.%SubWith.WithSelf.Op.type (%SubWith.WithSelf.Op.type.353) = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [symbolic = @SubWith.WithSelf.%SubWith.WithSelf.Op (constants.%SubWith.WithSelf.Op.197)]
// CHECK:STDOUT:   %Core.import_ref.afe: type = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [concrete = %Result.36c]
// CHECK:STDOUT:   %Result.36c: type = assoc_const_decl @Result.2 [concrete] {}
// CHECK:STDOUT:   %Core.import_ref.efd = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.4a3: @MulWith.WithSelf.%MulWith.assoc_type (%MulWith.assoc_type.ae7) = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [symbolic = @MulWith.WithSelf.%assoc1 (constants.%assoc1.d52)]
// CHECK:STDOUT:   %Core.import_ref.32c: @MulWith.WithSelf.%MulWith.WithSelf.Op.type (%MulWith.WithSelf.Op.type.896) = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [symbolic = @MulWith.WithSelf.%MulWith.WithSelf.Op (constants.%MulWith.WithSelf.Op.9fb)]
// CHECK:STDOUT:   %Core.import_ref.648: type = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [concrete = %Result.5ae]
// CHECK:STDOUT:   %Result.5ae: type = assoc_const_decl @Result.3 [concrete] {}
// CHECK:STDOUT:   %Core.import_ref.d54 = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.9631f: @DivWith.WithSelf.%DivWith.assoc_type (%DivWith.assoc_type.115) = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [symbolic = @DivWith.WithSelf.%assoc1 (constants.%assoc1.868)]
// CHECK:STDOUT:   %Core.import_ref.7dc: @DivWith.WithSelf.%DivWith.WithSelf.Op.type (%DivWith.WithSelf.Op.type.993) = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [symbolic = @DivWith.WithSelf.%DivWith.WithSelf.Op (constants.%DivWith.WithSelf.Op.e1e)]
// CHECK:STDOUT:   %Core.import_ref.db1: type = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [concrete = %Result.dba]
// CHECK:STDOUT:   %Result.dba: type = assoc_const_decl @Result.4 [concrete] {}
// CHECK:STDOUT:   %Core.import_ref.33d = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.db2: @ModWith.WithSelf.%ModWith.assoc_type (%ModWith.assoc_type.8a9) = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [symbolic = @ModWith.WithSelf.%assoc1 (constants.%assoc1.7ba)]
// CHECK:STDOUT:   %Core.import_ref.9e2: @ModWith.WithSelf.%ModWith.WithSelf.Op.type (%ModWith.WithSelf.Op.type.c98) = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [symbolic = @ModWith.WithSelf.%ModWith.WithSelf.Op (constants.%ModWith.WithSelf.Op.ac5)]
// CHECK:STDOUT:   %Core.import_ref.78b: type = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [concrete = %Result.63c]
// CHECK:STDOUT:   %Result.63c: type = assoc_const_decl @Result.5 [concrete] {}
// CHECK:STDOUT:   %Core.import_ref.3af = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.ab0: @AddAssignWith.WithSelf.%AddAssignWith.assoc_type (%AddAssignWith.assoc_type.2656aa.1) = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [symbolic = @AddAssignWith.WithSelf.%assoc0 (constants.%assoc0.595)]
// CHECK:STDOUT:   %Core.import_ref.073: @AddAssignWith.WithSelf.%AddAssignWith.WithSelf.Op.type (%AddAssignWith.WithSelf.Op.type.646415.1) = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [symbolic = @AddAssignWith.WithSelf.%AddAssignWith.WithSelf.Op (constants.%AddAssignWith.WithSelf.Op.f5fdb8.1)]
// CHECK:STDOUT:   %Core.import_ref.0da = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.3a2: @SubAssignWith.WithSelf.%SubAssignWith.assoc_type (%SubAssignWith.assoc_type.f4a281.1) = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [symbolic = @SubAssignWith.WithSelf.%assoc0 (constants.%assoc0.5d8)]
// CHECK:STDOUT:   %Core.import_ref.fc8: @SubAssignWith.WithSelf.%SubAssignWith.WithSelf.Op.type (%SubAssignWith.WithSelf.Op.type.1e97d0.1) = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [symbolic = @SubAssignWith.WithSelf.%SubAssignWith.WithSelf.Op (constants.%SubAssignWith.WithSelf.Op.0e2d98.1)]
// CHECK:STDOUT:   %Core.import_ref.4e6 = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.8ca: @MulAssignWith.WithSelf.%MulAssignWith.assoc_type (%MulAssignWith.assoc_type.fda5e1.1) = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [symbolic = @MulAssignWith.WithSelf.%assoc0 (constants.%assoc0.40c)]
// CHECK:STDOUT:   %Core.import_ref.303: @MulAssignWith.WithSelf.%MulAssignWith.WithSelf.Op.type (%MulAssignWith.WithSelf.Op.type.35745f.1) = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [symbolic = @MulAssignWith.WithSelf.%MulAssignWith.WithSelf.Op (constants.%MulAssignWith.WithSelf.Op.352c74.1)]
// CHECK:STDOUT:   %Core.import_ref.c93 = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.f7c: @DivAssignWith.WithSelf.%DivAssignWith.assoc_type (%DivAssignWith.assoc_type.c34e7f.1) = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [symbolic = @DivAssignWith.WithSelf.%assoc0 (constants.%assoc0.420)]
// CHECK:STDOUT:   %Core.import_ref.52b: @DivAssignWith.WithSelf.%DivAssignWith.WithSelf.Op.type (%DivAssignWith.WithSelf.Op.type.8f43bb.1) = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [symbolic = @DivAssignWith.WithSelf.%DivAssignWith.WithSelf.Op (constants.%DivAssignWith.WithSelf.Op.773fdb.1)]
// CHECK:STDOUT:   %Core.import_ref.a22 = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.042: @ModAssignWith.WithSelf.%ModAssignWith.assoc_type (%ModAssignWith.assoc_type.0beb3b.1) = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [symbolic = @ModAssignWith.WithSelf.%assoc0 (constants.%assoc0.bd1)]
// CHECK:STDOUT:   %Core.import_ref.a9d: @ModAssignWith.WithSelf.%ModAssignWith.WithSelf.Op.type (%ModAssignWith.WithSelf.Op.type.d01472.1) = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [symbolic = @ModAssignWith.WithSelf.%ModAssignWith.WithSelf.Op (constants.%ModAssignWith.WithSelf.Op.6c6076.1)]
// CHECK:STDOUT:   %Core.import_ref.3c9 = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @AddWith.loc124(%Result.loc121_9.2: %Destroy.type, %U.loc122_4.2: %Destroy.type, %T.loc123_4.2: @AddWith.loc124.%facet_type.loc123_38 (%facet_type.f91)) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %require_complete.loc126: <witness> = require_complete_type %AddWith.type.loc123_36.1 [symbolic = %require_complete.loc126 (constants.%require_complete.911)]
// CHECK:STDOUT:   %assoc1: @AddWith.loc124.%AddWith.assoc_type (%AddWith.assoc_type.91f) = assoc_entity element1, imports.%Core.import_ref.2b2 [symbolic = %assoc1 (constants.%assoc1.85e)]
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.loc126: <witness> = lookup_impl_witness %T.loc123_4.1, @AddWith.1, @AddWith.1(%U.as_type.loc123_36.1) [symbolic = %AddWith.lookup_impl_witness.loc126 (constants.%AddWith.lookup_impl_witness.1f7)]
// CHECK:STDOUT:   %AddWith.facet: @AddWith.loc124.%AddWith.type.loc123_36.1 (%AddWith.type.e1b) = facet_value %T.as_type.loc124_29.1, (%AddWith.lookup_impl_witness.loc126) [symbolic = %AddWith.facet (constants.%AddWith.facet.4a7)]
// CHECK:STDOUT:   %AddWith.WithSelf.Op.type: type = fn_type @AddWith.WithSelf.Op, @AddWith.WithSelf(%U.as_type.loc123_36.1, %AddWith.facet) [symbolic = %AddWith.WithSelf.Op.type (constants.%AddWith.WithSelf.Op.type.185)]
// CHECK:STDOUT:   %.loc126_5.4: type = fn_type_with_self_type %AddWith.WithSelf.Op.type, %AddWith.facet [symbolic = %.loc126_5.4 (constants.%.993)]
// CHECK:STDOUT:   %impl.elem1.loc126_5.2: @AddWith.loc124.%.loc126_5.4 (%.993) = impl_witness_access %AddWith.lookup_impl_witness.loc126, element1 [symbolic = %impl.elem1.loc126_5.2 (constants.%impl.elem1.ad4)]
// CHECK:STDOUT:   %specific_impl_fn.loc126_5.3: <specific function> = specific_impl_function %impl.elem1.loc126_5.2, @AddWith.WithSelf.Op(%U.as_type.loc123_36.1, %AddWith.facet) [symbolic = %specific_impl_fn.loc126_5.3 (constants.%specific_impl_fn.d63)]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Result.loc121_9.1) [symbolic = %Destroy.WithSelf.Op.type (constants.%Destroy.WithSelf.Op.type.b9d4b3.2)]
// CHECK:STDOUT:   %.loc126_5.5: type = fn_type_with_self_type %Destroy.WithSelf.Op.type, %Result.loc121_9.1 [symbolic = %.loc126_5.5 (constants.%.387)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %Result.loc121_9.1, @Destroy [symbolic = %Destroy.lookup_impl_witness (constants.%Destroy.lookup_impl_witness)]
// CHECK:STDOUT:   %impl.elem0.loc126_5.2: @AddWith.loc124.%.loc126_5.5 (%.387) = impl_witness_access %Destroy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc126_5.2 (constants.%impl.elem0.2e6)]
// CHECK:STDOUT:   %specific_impl_fn.loc126_5.4: <specific function> = specific_impl_function %impl.elem0.loc126_5.2, @Destroy.WithSelf.Op(%Result.loc121_9.1) [symbolic = %specific_impl_fn.loc126_5.4 (constants.%specific_impl_fn.62a)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%result.param: @AddWith.loc124.%Result.as_type.loc123_54.1 (%Result.as_type), %x.param: @AddWith.loc124.%T.as_type.loc124_29.1 (%T.as_type.94c), %y.param: @AddWith.loc124.%U.as_type.loc123_36.1 (%U.as_type.04c)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: @AddWith.loc124.%T.as_type.loc124_29.1 (%T.as_type.94c) = name_ref x, %x
// CHECK:STDOUT:     %y.ref: @AddWith.loc124.%U.as_type.loc123_36.1 (%U.as_type.04c) = name_ref y, %y
// CHECK:STDOUT:     %AddWith.type.loc126: type = facet_type <@AddWith.1, @AddWith.1(constants.%U.as_type.04c)> [symbolic = %AddWith.type.loc123_36.1 (constants.%AddWith.type.e1b)]
// CHECK:STDOUT:     %.loc126_5.1: @AddWith.loc124.%AddWith.assoc_type (%AddWith.assoc_type.91f) = specific_constant imports.%Core.import_ref.c92d, @AddWith.WithSelf(constants.%U.as_type.04c, constants.%Self.e74) [symbolic = %assoc1 (constants.%assoc1.85e)]
// CHECK:STDOUT:     %Op.ref: @AddWith.loc124.%AddWith.assoc_type (%AddWith.assoc_type.91f) = name_ref Op, %.loc126_5.1 [symbolic = %assoc1 (constants.%assoc1.85e)]
// CHECK:STDOUT:     %impl.elem1.loc126_5.1: @AddWith.loc124.%.loc126_5.4 (%.993) = impl_witness_access constants.%AddWith.lookup_impl_witness.1f7, element1 [symbolic = %impl.elem1.loc126_5.2 (constants.%impl.elem1.ad4)]
// CHECK:STDOUT:     %bound_method.loc126_5.1: <bound method> = bound_method %x.ref, %impl.elem1.loc126_5.1
// CHECK:STDOUT:     %specific_impl_fn.loc126_5.1: <specific function> = specific_impl_function %impl.elem1.loc126_5.1, @AddWith.WithSelf.Op(constants.%U.as_type.04c, constants.%AddWith.facet.4a7) [symbolic = %specific_impl_fn.loc126_5.3 (constants.%specific_impl_fn.d63)]
// CHECK:STDOUT:     %bound_method.loc126_5.2: <bound method> = bound_method %x.ref, %specific_impl_fn.loc126_5.1
// CHECK:STDOUT:     %.loc126_5.2: ref @AddWith.loc124.%Result.as_type.loc123_54.1 (%Result.as_type) = temporary_storage
// CHECK:STDOUT:     %AddWith.WithSelf.Op.call: init @AddWith.loc124.%Result.as_type.loc123_54.1 (%Result.as_type) to %.loc126_5.2 = call %bound_method.loc126_5.2(%x.ref, %y.ref)
// CHECK:STDOUT:     %.loc126_5.3: ref @AddWith.loc124.%Result.as_type.loc123_54.1 (%Result.as_type) = temporary %.loc126_5.2, %AddWith.WithSelf.Op.call
// CHECK:STDOUT:     %impl.elem0.loc126_5.1: @AddWith.loc124.%.loc126_5.5 (%.387) = impl_witness_access constants.%Destroy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc126_5.2 (constants.%impl.elem0.2e6)]
// CHECK:STDOUT:     %bound_method.loc126_5.3: <bound method> = bound_method %.loc126_5.3, %impl.elem0.loc126_5.1
// CHECK:STDOUT:     %specific_impl_fn.loc126_5.2: <specific function> = specific_impl_function %impl.elem0.loc126_5.1, @Destroy.WithSelf.Op(constants.%Result) [symbolic = %specific_impl_fn.loc126_5.4 (constants.%specific_impl_fn.62a)]
// CHECK:STDOUT:     %bound_method.loc126_5.4: <bound method> = bound_method %.loc126_5.3, %specific_impl_fn.loc126_5.2
// CHECK:STDOUT:     %Destroy.WithSelf.Op.call: init %empty_tuple.type = call %bound_method.loc126_5.4(%.loc126_5.3)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @SubWith.loc135(%Result.loc132_9.2: %Destroy.type, %U.loc133_4.2: %Destroy.type, %T.loc134_4.2: @SubWith.loc135.%facet_type.loc134_38 (%facet_type.a29)) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %require_complete.loc137: <witness> = require_complete_type %SubWith.type.loc134_36.1 [symbolic = %require_complete.loc137 (constants.%require_complete.a67)]
// CHECK:STDOUT:   %assoc1: @SubWith.loc135.%SubWith.assoc_type (%SubWith.assoc_type.760) = assoc_entity element1, imports.%Core.import_ref.bc0 [symbolic = %assoc1 (constants.%assoc1.972)]
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.loc137: <witness> = lookup_impl_witness %T.loc134_4.1, @SubWith.1, @SubWith.1(%U.as_type.loc134_36.1) [symbolic = %SubWith.lookup_impl_witness.loc137 (constants.%SubWith.lookup_impl_witness.ebb)]
// CHECK:STDOUT:   %SubWith.facet: @SubWith.loc135.%SubWith.type.loc134_36.1 (%SubWith.type.2e8) = facet_value %T.as_type.loc135_29.1, (%SubWith.lookup_impl_witness.loc137) [symbolic = %SubWith.facet (constants.%SubWith.facet.e7a)]
// CHECK:STDOUT:   %SubWith.WithSelf.Op.type: type = fn_type @SubWith.WithSelf.Op, @SubWith.WithSelf(%U.as_type.loc134_36.1, %SubWith.facet) [symbolic = %SubWith.WithSelf.Op.type (constants.%SubWith.WithSelf.Op.type.0dc)]
// CHECK:STDOUT:   %.loc137_5.4: type = fn_type_with_self_type %SubWith.WithSelf.Op.type, %SubWith.facet [symbolic = %.loc137_5.4 (constants.%.4bc)]
// CHECK:STDOUT:   %impl.elem1.loc137_5.2: @SubWith.loc135.%.loc137_5.4 (%.4bc) = impl_witness_access %SubWith.lookup_impl_witness.loc137, element1 [symbolic = %impl.elem1.loc137_5.2 (constants.%impl.elem1.a3d)]
// CHECK:STDOUT:   %specific_impl_fn.loc137_5.3: <specific function> = specific_impl_function %impl.elem1.loc137_5.2, @SubWith.WithSelf.Op(%U.as_type.loc134_36.1, %SubWith.facet) [symbolic = %specific_impl_fn.loc137_5.3 (constants.%specific_impl_fn.542)]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Result.loc132_9.1) [symbolic = %Destroy.WithSelf.Op.type (constants.%Destroy.WithSelf.Op.type.b9d4b3.2)]
// CHECK:STDOUT:   %.loc137_5.5: type = fn_type_with_self_type %Destroy.WithSelf.Op.type, %Result.loc132_9.1 [symbolic = %.loc137_5.5 (constants.%.387)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %Result.loc132_9.1, @Destroy [symbolic = %Destroy.lookup_impl_witness (constants.%Destroy.lookup_impl_witness)]
// CHECK:STDOUT:   %impl.elem0.loc137_5.2: @SubWith.loc135.%.loc137_5.5 (%.387) = impl_witness_access %Destroy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc137_5.2 (constants.%impl.elem0.2e6)]
// CHECK:STDOUT:   %specific_impl_fn.loc137_5.4: <specific function> = specific_impl_function %impl.elem0.loc137_5.2, @Destroy.WithSelf.Op(%Result.loc132_9.1) [symbolic = %specific_impl_fn.loc137_5.4 (constants.%specific_impl_fn.62a)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%result.param: @SubWith.loc135.%Result.as_type.loc134_54.1 (%Result.as_type), %x.param: @SubWith.loc135.%T.as_type.loc135_29.1 (%T.as_type.803), %y.param: @SubWith.loc135.%U.as_type.loc134_36.1 (%U.as_type.04c)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: @SubWith.loc135.%T.as_type.loc135_29.1 (%T.as_type.803) = name_ref x, %x
// CHECK:STDOUT:     %y.ref: @SubWith.loc135.%U.as_type.loc134_36.1 (%U.as_type.04c) = name_ref y, %y
// CHECK:STDOUT:     %SubWith.type.loc137: type = facet_type <@SubWith.1, @SubWith.1(constants.%U.as_type.04c)> [symbolic = %SubWith.type.loc134_36.1 (constants.%SubWith.type.2e8)]
// CHECK:STDOUT:     %.loc137_5.1: @SubWith.loc135.%SubWith.assoc_type (%SubWith.assoc_type.760) = specific_constant imports.%Core.import_ref.d04, @SubWith.WithSelf(constants.%U.as_type.04c, constants.%Self.97e) [symbolic = %assoc1 (constants.%assoc1.972)]
// CHECK:STDOUT:     %Op.ref: @SubWith.loc135.%SubWith.assoc_type (%SubWith.assoc_type.760) = name_ref Op, %.loc137_5.1 [symbolic = %assoc1 (constants.%assoc1.972)]
// CHECK:STDOUT:     %impl.elem1.loc137_5.1: @SubWith.loc135.%.loc137_5.4 (%.4bc) = impl_witness_access constants.%SubWith.lookup_impl_witness.ebb, element1 [symbolic = %impl.elem1.loc137_5.2 (constants.%impl.elem1.a3d)]
// CHECK:STDOUT:     %bound_method.loc137_5.1: <bound method> = bound_method %x.ref, %impl.elem1.loc137_5.1
// CHECK:STDOUT:     %specific_impl_fn.loc137_5.1: <specific function> = specific_impl_function %impl.elem1.loc137_5.1, @SubWith.WithSelf.Op(constants.%U.as_type.04c, constants.%SubWith.facet.e7a) [symbolic = %specific_impl_fn.loc137_5.3 (constants.%specific_impl_fn.542)]
// CHECK:STDOUT:     %bound_method.loc137_5.2: <bound method> = bound_method %x.ref, %specific_impl_fn.loc137_5.1
// CHECK:STDOUT:     %.loc137_5.2: ref @SubWith.loc135.%Result.as_type.loc134_54.1 (%Result.as_type) = temporary_storage
// CHECK:STDOUT:     %SubWith.WithSelf.Op.call: init @SubWith.loc135.%Result.as_type.loc134_54.1 (%Result.as_type) to %.loc137_5.2 = call %bound_method.loc137_5.2(%x.ref, %y.ref)
// CHECK:STDOUT:     %.loc137_5.3: ref @SubWith.loc135.%Result.as_type.loc134_54.1 (%Result.as_type) = temporary %.loc137_5.2, %SubWith.WithSelf.Op.call
// CHECK:STDOUT:     %impl.elem0.loc137_5.1: @SubWith.loc135.%.loc137_5.5 (%.387) = impl_witness_access constants.%Destroy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc137_5.2 (constants.%impl.elem0.2e6)]
// CHECK:STDOUT:     %bound_method.loc137_5.3: <bound method> = bound_method %.loc137_5.3, %impl.elem0.loc137_5.1
// CHECK:STDOUT:     %specific_impl_fn.loc137_5.2: <specific function> = specific_impl_function %impl.elem0.loc137_5.1, @Destroy.WithSelf.Op(constants.%Result) [symbolic = %specific_impl_fn.loc137_5.4 (constants.%specific_impl_fn.62a)]
// CHECK:STDOUT:     %bound_method.loc137_5.4: <bound method> = bound_method %.loc137_5.3, %specific_impl_fn.loc137_5.2
// CHECK:STDOUT:     %Destroy.WithSelf.Op.call: init %empty_tuple.type = call %bound_method.loc137_5.4(%.loc137_5.3)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @MulWith.loc146(%Result.loc143_9.2: %Destroy.type, %U.loc144_4.2: %Destroy.type, %T.loc145_4.2: @MulWith.loc146.%facet_type.loc145_38 (%facet_type.ec5)) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %require_complete.loc148: <witness> = require_complete_type %MulWith.type.loc145_36.1 [symbolic = %require_complete.loc148 (constants.%require_complete.50e)]
// CHECK:STDOUT:   %assoc1: @MulWith.loc146.%MulWith.assoc_type (%MulWith.assoc_type.8c9) = assoc_entity element1, imports.%Core.import_ref.32c [symbolic = %assoc1 (constants.%assoc1.8e8)]
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.loc148: <witness> = lookup_impl_witness %T.loc145_4.1, @MulWith.1, @MulWith.1(%U.as_type.loc145_36.1) [symbolic = %MulWith.lookup_impl_witness.loc148 (constants.%MulWith.lookup_impl_witness.38f)]
// CHECK:STDOUT:   %MulWith.facet: @MulWith.loc146.%MulWith.type.loc145_36.1 (%MulWith.type.293) = facet_value %T.as_type.loc146_29.1, (%MulWith.lookup_impl_witness.loc148) [symbolic = %MulWith.facet (constants.%MulWith.facet.69f)]
// CHECK:STDOUT:   %MulWith.WithSelf.Op.type: type = fn_type @MulWith.WithSelf.Op, @MulWith.WithSelf(%U.as_type.loc145_36.1, %MulWith.facet) [symbolic = %MulWith.WithSelf.Op.type (constants.%MulWith.WithSelf.Op.type.929)]
// CHECK:STDOUT:   %.loc148_5.4: type = fn_type_with_self_type %MulWith.WithSelf.Op.type, %MulWith.facet [symbolic = %.loc148_5.4 (constants.%.b06)]
// CHECK:STDOUT:   %impl.elem1.loc148_5.2: @MulWith.loc146.%.loc148_5.4 (%.b06) = impl_witness_access %MulWith.lookup_impl_witness.loc148, element1 [symbolic = %impl.elem1.loc148_5.2 (constants.%impl.elem1.a81)]
// CHECK:STDOUT:   %specific_impl_fn.loc148_5.3: <specific function> = specific_impl_function %impl.elem1.loc148_5.2, @MulWith.WithSelf.Op(%U.as_type.loc145_36.1, %MulWith.facet) [symbolic = %specific_impl_fn.loc148_5.3 (constants.%specific_impl_fn.551)]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Result.loc143_9.1) [symbolic = %Destroy.WithSelf.Op.type (constants.%Destroy.WithSelf.Op.type.b9d4b3.2)]
// CHECK:STDOUT:   %.loc148_5.5: type = fn_type_with_self_type %Destroy.WithSelf.Op.type, %Result.loc143_9.1 [symbolic = %.loc148_5.5 (constants.%.387)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %Result.loc143_9.1, @Destroy [symbolic = %Destroy.lookup_impl_witness (constants.%Destroy.lookup_impl_witness)]
// CHECK:STDOUT:   %impl.elem0.loc148_5.2: @MulWith.loc146.%.loc148_5.5 (%.387) = impl_witness_access %Destroy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc148_5.2 (constants.%impl.elem0.2e6)]
// CHECK:STDOUT:   %specific_impl_fn.loc148_5.4: <specific function> = specific_impl_function %impl.elem0.loc148_5.2, @Destroy.WithSelf.Op(%Result.loc143_9.1) [symbolic = %specific_impl_fn.loc148_5.4 (constants.%specific_impl_fn.62a)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%result.param: @MulWith.loc146.%Result.as_type.loc145_54.1 (%Result.as_type), %x.param: @MulWith.loc146.%T.as_type.loc146_29.1 (%T.as_type.0c1), %y.param: @MulWith.loc146.%U.as_type.loc145_36.1 (%U.as_type.04c)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: @MulWith.loc146.%T.as_type.loc146_29.1 (%T.as_type.0c1) = name_ref x, %x
// CHECK:STDOUT:     %y.ref: @MulWith.loc146.%U.as_type.loc145_36.1 (%U.as_type.04c) = name_ref y, %y
// CHECK:STDOUT:     %MulWith.type.loc148: type = facet_type <@MulWith.1, @MulWith.1(constants.%U.as_type.04c)> [symbolic = %MulWith.type.loc145_36.1 (constants.%MulWith.type.293)]
// CHECK:STDOUT:     %.loc148_5.1: @MulWith.loc146.%MulWith.assoc_type (%MulWith.assoc_type.8c9) = specific_constant imports.%Core.import_ref.4a3, @MulWith.WithSelf(constants.%U.as_type.04c, constants.%Self.11d) [symbolic = %assoc1 (constants.%assoc1.8e8)]
// CHECK:STDOUT:     %Op.ref: @MulWith.loc146.%MulWith.assoc_type (%MulWith.assoc_type.8c9) = name_ref Op, %.loc148_5.1 [symbolic = %assoc1 (constants.%assoc1.8e8)]
// CHECK:STDOUT:     %impl.elem1.loc148_5.1: @MulWith.loc146.%.loc148_5.4 (%.b06) = impl_witness_access constants.%MulWith.lookup_impl_witness.38f, element1 [symbolic = %impl.elem1.loc148_5.2 (constants.%impl.elem1.a81)]
// CHECK:STDOUT:     %bound_method.loc148_5.1: <bound method> = bound_method %x.ref, %impl.elem1.loc148_5.1
// CHECK:STDOUT:     %specific_impl_fn.loc148_5.1: <specific function> = specific_impl_function %impl.elem1.loc148_5.1, @MulWith.WithSelf.Op(constants.%U.as_type.04c, constants.%MulWith.facet.69f) [symbolic = %specific_impl_fn.loc148_5.3 (constants.%specific_impl_fn.551)]
// CHECK:STDOUT:     %bound_method.loc148_5.2: <bound method> = bound_method %x.ref, %specific_impl_fn.loc148_5.1
// CHECK:STDOUT:     %.loc148_5.2: ref @MulWith.loc146.%Result.as_type.loc145_54.1 (%Result.as_type) = temporary_storage
// CHECK:STDOUT:     %MulWith.WithSelf.Op.call: init @MulWith.loc146.%Result.as_type.loc145_54.1 (%Result.as_type) to %.loc148_5.2 = call %bound_method.loc148_5.2(%x.ref, %y.ref)
// CHECK:STDOUT:     %.loc148_5.3: ref @MulWith.loc146.%Result.as_type.loc145_54.1 (%Result.as_type) = temporary %.loc148_5.2, %MulWith.WithSelf.Op.call
// CHECK:STDOUT:     %impl.elem0.loc148_5.1: @MulWith.loc146.%.loc148_5.5 (%.387) = impl_witness_access constants.%Destroy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc148_5.2 (constants.%impl.elem0.2e6)]
// CHECK:STDOUT:     %bound_method.loc148_5.3: <bound method> = bound_method %.loc148_5.3, %impl.elem0.loc148_5.1
// CHECK:STDOUT:     %specific_impl_fn.loc148_5.2: <specific function> = specific_impl_function %impl.elem0.loc148_5.1, @Destroy.WithSelf.Op(constants.%Result) [symbolic = %specific_impl_fn.loc148_5.4 (constants.%specific_impl_fn.62a)]
// CHECK:STDOUT:     %bound_method.loc148_5.4: <bound method> = bound_method %.loc148_5.3, %specific_impl_fn.loc148_5.2
// CHECK:STDOUT:     %Destroy.WithSelf.Op.call: init %empty_tuple.type = call %bound_method.loc148_5.4(%.loc148_5.3)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @DivWith.loc157(%Result.loc154_9.2: %Destroy.type, %U.loc155_4.2: %Destroy.type, %T.loc156_4.2: @DivWith.loc157.%facet_type.loc156_38 (%facet_type.b8a)) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %require_complete.loc159: <witness> = require_complete_type %DivWith.type.loc156_36.1 [symbolic = %require_complete.loc159 (constants.%require_complete.2bc)]
// CHECK:STDOUT:   %assoc1: @DivWith.loc157.%DivWith.assoc_type (%DivWith.assoc_type.b3b) = assoc_entity element1, imports.%Core.import_ref.7dc [symbolic = %assoc1 (constants.%assoc1.d84)]
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.loc159: <witness> = lookup_impl_witness %T.loc156_4.1, @DivWith.1, @DivWith.1(%U.as_type.loc156_36.1) [symbolic = %DivWith.lookup_impl_witness.loc159 (constants.%DivWith.lookup_impl_witness.b03)]
// CHECK:STDOUT:   %DivWith.facet: @DivWith.loc157.%DivWith.type.loc156_36.1 (%DivWith.type.25a) = facet_value %T.as_type.loc157_29.1, (%DivWith.lookup_impl_witness.loc159) [symbolic = %DivWith.facet (constants.%DivWith.facet.924)]
// CHECK:STDOUT:   %DivWith.WithSelf.Op.type: type = fn_type @DivWith.WithSelf.Op, @DivWith.WithSelf(%U.as_type.loc156_36.1, %DivWith.facet) [symbolic = %DivWith.WithSelf.Op.type (constants.%DivWith.WithSelf.Op.type.ddd)]
// CHECK:STDOUT:   %.loc159_5.4: type = fn_type_with_self_type %DivWith.WithSelf.Op.type, %DivWith.facet [symbolic = %.loc159_5.4 (constants.%.4f8)]
// CHECK:STDOUT:   %impl.elem1.loc159_5.2: @DivWith.loc157.%.loc159_5.4 (%.4f8) = impl_witness_access %DivWith.lookup_impl_witness.loc159, element1 [symbolic = %impl.elem1.loc159_5.2 (constants.%impl.elem1.3cc)]
// CHECK:STDOUT:   %specific_impl_fn.loc159_5.3: <specific function> = specific_impl_function %impl.elem1.loc159_5.2, @DivWith.WithSelf.Op(%U.as_type.loc156_36.1, %DivWith.facet) [symbolic = %specific_impl_fn.loc159_5.3 (constants.%specific_impl_fn.7c7)]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Result.loc154_9.1) [symbolic = %Destroy.WithSelf.Op.type (constants.%Destroy.WithSelf.Op.type.b9d4b3.2)]
// CHECK:STDOUT:   %.loc159_5.5: type = fn_type_with_self_type %Destroy.WithSelf.Op.type, %Result.loc154_9.1 [symbolic = %.loc159_5.5 (constants.%.387)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %Result.loc154_9.1, @Destroy [symbolic = %Destroy.lookup_impl_witness (constants.%Destroy.lookup_impl_witness)]
// CHECK:STDOUT:   %impl.elem0.loc159_5.2: @DivWith.loc157.%.loc159_5.5 (%.387) = impl_witness_access %Destroy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc159_5.2 (constants.%impl.elem0.2e6)]
// CHECK:STDOUT:   %specific_impl_fn.loc159_5.4: <specific function> = specific_impl_function %impl.elem0.loc159_5.2, @Destroy.WithSelf.Op(%Result.loc154_9.1) [symbolic = %specific_impl_fn.loc159_5.4 (constants.%specific_impl_fn.62a)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%result.param: @DivWith.loc157.%Result.as_type.loc156_54.1 (%Result.as_type), %x.param: @DivWith.loc157.%T.as_type.loc157_29.1 (%T.as_type.2d1), %y.param: @DivWith.loc157.%U.as_type.loc156_36.1 (%U.as_type.04c)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: @DivWith.loc157.%T.as_type.loc157_29.1 (%T.as_type.2d1) = name_ref x, %x
// CHECK:STDOUT:     %y.ref: @DivWith.loc157.%U.as_type.loc156_36.1 (%U.as_type.04c) = name_ref y, %y
// CHECK:STDOUT:     %DivWith.type.loc159: type = facet_type <@DivWith.1, @DivWith.1(constants.%U.as_type.04c)> [symbolic = %DivWith.type.loc156_36.1 (constants.%DivWith.type.25a)]
// CHECK:STDOUT:     %.loc159_5.1: @DivWith.loc157.%DivWith.assoc_type (%DivWith.assoc_type.b3b) = specific_constant imports.%Core.import_ref.9631f, @DivWith.WithSelf(constants.%U.as_type.04c, constants.%Self.c0b) [symbolic = %assoc1 (constants.%assoc1.d84)]
// CHECK:STDOUT:     %Op.ref: @DivWith.loc157.%DivWith.assoc_type (%DivWith.assoc_type.b3b) = name_ref Op, %.loc159_5.1 [symbolic = %assoc1 (constants.%assoc1.d84)]
// CHECK:STDOUT:     %impl.elem1.loc159_5.1: @DivWith.loc157.%.loc159_5.4 (%.4f8) = impl_witness_access constants.%DivWith.lookup_impl_witness.b03, element1 [symbolic = %impl.elem1.loc159_5.2 (constants.%impl.elem1.3cc)]
// CHECK:STDOUT:     %bound_method.loc159_5.1: <bound method> = bound_method %x.ref, %impl.elem1.loc159_5.1
// CHECK:STDOUT:     %specific_impl_fn.loc159_5.1: <specific function> = specific_impl_function %impl.elem1.loc159_5.1, @DivWith.WithSelf.Op(constants.%U.as_type.04c, constants.%DivWith.facet.924) [symbolic = %specific_impl_fn.loc159_5.3 (constants.%specific_impl_fn.7c7)]
// CHECK:STDOUT:     %bound_method.loc159_5.2: <bound method> = bound_method %x.ref, %specific_impl_fn.loc159_5.1
// CHECK:STDOUT:     %.loc159_5.2: ref @DivWith.loc157.%Result.as_type.loc156_54.1 (%Result.as_type) = temporary_storage
// CHECK:STDOUT:     %DivWith.WithSelf.Op.call: init @DivWith.loc157.%Result.as_type.loc156_54.1 (%Result.as_type) to %.loc159_5.2 = call %bound_method.loc159_5.2(%x.ref, %y.ref)
// CHECK:STDOUT:     %.loc159_5.3: ref @DivWith.loc157.%Result.as_type.loc156_54.1 (%Result.as_type) = temporary %.loc159_5.2, %DivWith.WithSelf.Op.call
// CHECK:STDOUT:     %impl.elem0.loc159_5.1: @DivWith.loc157.%.loc159_5.5 (%.387) = impl_witness_access constants.%Destroy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc159_5.2 (constants.%impl.elem0.2e6)]
// CHECK:STDOUT:     %bound_method.loc159_5.3: <bound method> = bound_method %.loc159_5.3, %impl.elem0.loc159_5.1
// CHECK:STDOUT:     %specific_impl_fn.loc159_5.2: <specific function> = specific_impl_function %impl.elem0.loc159_5.1, @Destroy.WithSelf.Op(constants.%Result) [symbolic = %specific_impl_fn.loc159_5.4 (constants.%specific_impl_fn.62a)]
// CHECK:STDOUT:     %bound_method.loc159_5.4: <bound method> = bound_method %.loc159_5.3, %specific_impl_fn.loc159_5.2
// CHECK:STDOUT:     %Destroy.WithSelf.Op.call: init %empty_tuple.type = call %bound_method.loc159_5.4(%.loc159_5.3)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @ModWith.loc168(%Result.loc165_9.2: %Destroy.type, %U.loc166_4.2: %Destroy.type, %T.loc167_4.2: @ModWith.loc168.%facet_type.loc167_38 (%facet_type.c9c)) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %require_complete.loc170: <witness> = require_complete_type %ModWith.type.loc167_36.1 [symbolic = %require_complete.loc170 (constants.%require_complete.ccd)]
// CHECK:STDOUT:   %assoc1: @ModWith.loc168.%ModWith.assoc_type (%ModWith.assoc_type.84f) = assoc_entity element1, imports.%Core.import_ref.9e2 [symbolic = %assoc1 (constants.%assoc1.ea2)]
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.loc170: <witness> = lookup_impl_witness %T.loc167_4.1, @ModWith.1, @ModWith.1(%U.as_type.loc167_36.1) [symbolic = %ModWith.lookup_impl_witness.loc170 (constants.%ModWith.lookup_impl_witness.962)]
// CHECK:STDOUT:   %ModWith.facet: @ModWith.loc168.%ModWith.type.loc167_36.1 (%ModWith.type.3cd) = facet_value %T.as_type.loc168_29.1, (%ModWith.lookup_impl_witness.loc170) [symbolic = %ModWith.facet (constants.%ModWith.facet.bc2)]
// CHECK:STDOUT:   %ModWith.WithSelf.Op.type: type = fn_type @ModWith.WithSelf.Op, @ModWith.WithSelf(%U.as_type.loc167_36.1, %ModWith.facet) [symbolic = %ModWith.WithSelf.Op.type (constants.%ModWith.WithSelf.Op.type.fed)]
// CHECK:STDOUT:   %.loc170_5.4: type = fn_type_with_self_type %ModWith.WithSelf.Op.type, %ModWith.facet [symbolic = %.loc170_5.4 (constants.%.ed2)]
// CHECK:STDOUT:   %impl.elem1.loc170_5.2: @ModWith.loc168.%.loc170_5.4 (%.ed2) = impl_witness_access %ModWith.lookup_impl_witness.loc170, element1 [symbolic = %impl.elem1.loc170_5.2 (constants.%impl.elem1.260)]
// CHECK:STDOUT:   %specific_impl_fn.loc170_5.3: <specific function> = specific_impl_function %impl.elem1.loc170_5.2, @ModWith.WithSelf.Op(%U.as_type.loc167_36.1, %ModWith.facet) [symbolic = %specific_impl_fn.loc170_5.3 (constants.%specific_impl_fn.4e2)]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Result.loc165_9.1) [symbolic = %Destroy.WithSelf.Op.type (constants.%Destroy.WithSelf.Op.type.b9d4b3.2)]
// CHECK:STDOUT:   %.loc170_5.5: type = fn_type_with_self_type %Destroy.WithSelf.Op.type, %Result.loc165_9.1 [symbolic = %.loc170_5.5 (constants.%.387)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %Result.loc165_9.1, @Destroy [symbolic = %Destroy.lookup_impl_witness (constants.%Destroy.lookup_impl_witness)]
// CHECK:STDOUT:   %impl.elem0.loc170_5.2: @ModWith.loc168.%.loc170_5.5 (%.387) = impl_witness_access %Destroy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc170_5.2 (constants.%impl.elem0.2e6)]
// CHECK:STDOUT:   %specific_impl_fn.loc170_5.4: <specific function> = specific_impl_function %impl.elem0.loc170_5.2, @Destroy.WithSelf.Op(%Result.loc165_9.1) [symbolic = %specific_impl_fn.loc170_5.4 (constants.%specific_impl_fn.62a)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%result.param: @ModWith.loc168.%Result.as_type.loc167_54.1 (%Result.as_type), %x.param: @ModWith.loc168.%T.as_type.loc168_29.1 (%T.as_type.2fa), %y.param: @ModWith.loc168.%U.as_type.loc167_36.1 (%U.as_type.04c)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: @ModWith.loc168.%T.as_type.loc168_29.1 (%T.as_type.2fa) = name_ref x, %x
// CHECK:STDOUT:     %y.ref: @ModWith.loc168.%U.as_type.loc167_36.1 (%U.as_type.04c) = name_ref y, %y
// CHECK:STDOUT:     %ModWith.type.loc170: type = facet_type <@ModWith.1, @ModWith.1(constants.%U.as_type.04c)> [symbolic = %ModWith.type.loc167_36.1 (constants.%ModWith.type.3cd)]
// CHECK:STDOUT:     %.loc170_5.1: @ModWith.loc168.%ModWith.assoc_type (%ModWith.assoc_type.84f) = specific_constant imports.%Core.import_ref.db2, @ModWith.WithSelf(constants.%U.as_type.04c, constants.%Self.852) [symbolic = %assoc1 (constants.%assoc1.ea2)]
// CHECK:STDOUT:     %Op.ref: @ModWith.loc168.%ModWith.assoc_type (%ModWith.assoc_type.84f) = name_ref Op, %.loc170_5.1 [symbolic = %assoc1 (constants.%assoc1.ea2)]
// CHECK:STDOUT:     %impl.elem1.loc170_5.1: @ModWith.loc168.%.loc170_5.4 (%.ed2) = impl_witness_access constants.%ModWith.lookup_impl_witness.962, element1 [symbolic = %impl.elem1.loc170_5.2 (constants.%impl.elem1.260)]
// CHECK:STDOUT:     %bound_method.loc170_5.1: <bound method> = bound_method %x.ref, %impl.elem1.loc170_5.1
// CHECK:STDOUT:     %specific_impl_fn.loc170_5.1: <specific function> = specific_impl_function %impl.elem1.loc170_5.1, @ModWith.WithSelf.Op(constants.%U.as_type.04c, constants.%ModWith.facet.bc2) [symbolic = %specific_impl_fn.loc170_5.3 (constants.%specific_impl_fn.4e2)]
// CHECK:STDOUT:     %bound_method.loc170_5.2: <bound method> = bound_method %x.ref, %specific_impl_fn.loc170_5.1
// CHECK:STDOUT:     %.loc170_5.2: ref @ModWith.loc168.%Result.as_type.loc167_54.1 (%Result.as_type) = temporary_storage
// CHECK:STDOUT:     %ModWith.WithSelf.Op.call: init @ModWith.loc168.%Result.as_type.loc167_54.1 (%Result.as_type) to %.loc170_5.2 = call %bound_method.loc170_5.2(%x.ref, %y.ref)
// CHECK:STDOUT:     %.loc170_5.3: ref @ModWith.loc168.%Result.as_type.loc167_54.1 (%Result.as_type) = temporary %.loc170_5.2, %ModWith.WithSelf.Op.call
// CHECK:STDOUT:     %impl.elem0.loc170_5.1: @ModWith.loc168.%.loc170_5.5 (%.387) = impl_witness_access constants.%Destroy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc170_5.2 (constants.%impl.elem0.2e6)]
// CHECK:STDOUT:     %bound_method.loc170_5.3: <bound method> = bound_method %.loc170_5.3, %impl.elem0.loc170_5.1
// CHECK:STDOUT:     %specific_impl_fn.loc170_5.2: <specific function> = specific_impl_function %impl.elem0.loc170_5.1, @Destroy.WithSelf.Op(constants.%Result) [symbolic = %specific_impl_fn.loc170_5.4 (constants.%specific_impl_fn.62a)]
// CHECK:STDOUT:     %bound_method.loc170_5.4: <bound method> = bound_method %.loc170_5.3, %specific_impl_fn.loc170_5.2
// CHECK:STDOUT:     %Destroy.WithSelf.Op.call: init %empty_tuple.type = call %bound_method.loc170_5.4(%.loc170_5.3)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @AddAssignWith.loc225(%U.loc225_19.2: type, %T.loc225_29.2: @AddAssignWith.loc225.%AddAssignWith.type.loc225_52.1 (%AddAssignWith.type.d3e5c5.2)) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %require_complete.loc227: <witness> = require_complete_type %AddAssignWith.type.loc225_52.1 [symbolic = %require_complete.loc227 (constants.%require_complete.b1a)]
// CHECK:STDOUT:   %AddAssignWith.assoc_type: type = assoc_entity_type @AddAssignWith.1, @AddAssignWith.1(%U.loc225_19.1) [symbolic = %AddAssignWith.assoc_type (constants.%AddAssignWith.assoc_type.2656aa.2)]
// CHECK:STDOUT:   %assoc0: @AddAssignWith.loc225.%AddAssignWith.assoc_type (%AddAssignWith.assoc_type.2656aa.2) = assoc_entity element0, imports.%Core.import_ref.073 [symbolic = %assoc0 (constants.%assoc0.d3a3f1.2)]
// CHECK:STDOUT:   %AddAssignWith.WithSelf.Op.type: type = fn_type @AddAssignWith.WithSelf.Op, @AddAssignWith.WithSelf(%U.loc225_19.1, %T.loc225_29.1) [symbolic = %AddAssignWith.WithSelf.Op.type (constants.%AddAssignWith.WithSelf.Op.type.646415.3)]
// CHECK:STDOUT:   %.loc227_5.2: type = fn_type_with_self_type %AddAssignWith.WithSelf.Op.type, %T.loc225_29.1 [symbolic = %.loc227_5.2 (constants.%.8a0)]
// CHECK:STDOUT:   %AddAssignWith.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc225_29.1, @AddAssignWith.1, @AddAssignWith.1(%U.loc225_19.1) [symbolic = %AddAssignWith.lookup_impl_witness (constants.%AddAssignWith.lookup_impl_witness)]
// CHECK:STDOUT:   %impl.elem0.loc227_5.2: @AddAssignWith.loc225.%.loc227_5.2 (%.8a0) = impl_witness_access %AddAssignWith.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc227_5.2 (constants.%impl.elem0.1e5)]
// CHECK:STDOUT:   %specific_impl_fn.loc227_5.2: <specific function> = specific_impl_function %impl.elem0.loc227_5.2, @AddAssignWith.WithSelf.Op(%U.loc225_19.1, %T.loc225_29.1) [symbolic = %specific_impl_fn.loc227_5.2 (constants.%specific_impl_fn.97d)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: ref @AddAssignWith.loc225.%T.as_type.loc225_62.1 (%T.as_type.984), %y.param: @AddAssignWith.loc225.%U.loc225_19.1 (%U.67d)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: ref @AddAssignWith.loc225.%T.as_type.loc225_62.1 (%T.as_type.984) = name_ref x, %x
// CHECK:STDOUT:     %y.ref: @AddAssignWith.loc225.%U.loc225_19.1 (%U.67d) = name_ref y, %y
// CHECK:STDOUT:     %AddAssignWith.type.loc227: type = facet_type <@AddAssignWith.1, @AddAssignWith.1(constants.%U.67d)> [symbolic = %AddAssignWith.type.loc225_52.1 (constants.%AddAssignWith.type.d3e5c5.2)]
// CHECK:STDOUT:     %.loc227_5.1: @AddAssignWith.loc225.%AddAssignWith.assoc_type (%AddAssignWith.assoc_type.2656aa.2) = specific_constant imports.%Core.import_ref.ab0, @AddAssignWith.WithSelf(constants.%U.67d, constants.%Self.c28b6d.1) [symbolic = %assoc0 (constants.%assoc0.d3a3f1.2)]
// CHECK:STDOUT:     %Op.ref: @AddAssignWith.loc225.%AddAssignWith.assoc_type (%AddAssignWith.assoc_type.2656aa.2) = name_ref Op, %.loc227_5.1 [symbolic = %assoc0 (constants.%assoc0.d3a3f1.2)]
// CHECK:STDOUT:     %impl.elem0.loc227_5.1: @AddAssignWith.loc225.%.loc227_5.2 (%.8a0) = impl_witness_access constants.%AddAssignWith.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc227_5.2 (constants.%impl.elem0.1e5)]
// CHECK:STDOUT:     %bound_method.loc227_5.1: <bound method> = bound_method %x.ref, %impl.elem0.loc227_5.1
// CHECK:STDOUT:     %specific_impl_fn.loc227_5.1: <specific function> = specific_impl_function %impl.elem0.loc227_5.1, @AddAssignWith.WithSelf.Op(constants.%U.67d, constants.%T.c28) [symbolic = %specific_impl_fn.loc227_5.2 (constants.%specific_impl_fn.97d)]
// CHECK:STDOUT:     %bound_method.loc227_5.2: <bound method> = bound_method %x.ref, %specific_impl_fn.loc227_5.1
// CHECK:STDOUT:     %AddAssignWith.WithSelf.Op.call: init %empty_tuple.type = call %bound_method.loc227_5.2(%x.ref, %y.ref)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @SubAssignWith.loc231(%U.loc231_19.2: type, %T.loc231_29.2: @SubAssignWith.loc231.%SubAssignWith.type.loc231_52.1 (%SubAssignWith.type.267e69.2)) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %require_complete.loc233: <witness> = require_complete_type %SubAssignWith.type.loc231_52.1 [symbolic = %require_complete.loc233 (constants.%require_complete.401)]
// CHECK:STDOUT:   %SubAssignWith.assoc_type: type = assoc_entity_type @SubAssignWith.1, @SubAssignWith.1(%U.loc231_19.1) [symbolic = %SubAssignWith.assoc_type (constants.%SubAssignWith.assoc_type.f4a281.2)]
// CHECK:STDOUT:   %assoc0: @SubAssignWith.loc231.%SubAssignWith.assoc_type (%SubAssignWith.assoc_type.f4a281.2) = assoc_entity element0, imports.%Core.import_ref.fc8 [symbolic = %assoc0 (constants.%assoc0.556a96.2)]
// CHECK:STDOUT:   %SubAssignWith.WithSelf.Op.type: type = fn_type @SubAssignWith.WithSelf.Op, @SubAssignWith.WithSelf(%U.loc231_19.1, %T.loc231_29.1) [symbolic = %SubAssignWith.WithSelf.Op.type (constants.%SubAssignWith.WithSelf.Op.type.1e97d0.3)]
// CHECK:STDOUT:   %.loc233_5.2: type = fn_type_with_self_type %SubAssignWith.WithSelf.Op.type, %T.loc231_29.1 [symbolic = %.loc233_5.2 (constants.%.b89)]
// CHECK:STDOUT:   %SubAssignWith.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc231_29.1, @SubAssignWith.1, @SubAssignWith.1(%U.loc231_19.1) [symbolic = %SubAssignWith.lookup_impl_witness (constants.%SubAssignWith.lookup_impl_witness)]
// CHECK:STDOUT:   %impl.elem0.loc233_5.2: @SubAssignWith.loc231.%.loc233_5.2 (%.b89) = impl_witness_access %SubAssignWith.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc233_5.2 (constants.%impl.elem0.f58)]
// CHECK:STDOUT:   %specific_impl_fn.loc233_5.2: <specific function> = specific_impl_function %impl.elem0.loc233_5.2, @SubAssignWith.WithSelf.Op(%U.loc231_19.1, %T.loc231_29.1) [symbolic = %specific_impl_fn.loc233_5.2 (constants.%specific_impl_fn.340)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: ref @SubAssignWith.loc231.%T.as_type.loc231_62.1 (%T.as_type.b79), %y.param: @SubAssignWith.loc231.%U.loc231_19.1 (%U.67d)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: ref @SubAssignWith.loc231.%T.as_type.loc231_62.1 (%T.as_type.b79) = name_ref x, %x
// CHECK:STDOUT:     %y.ref: @SubAssignWith.loc231.%U.loc231_19.1 (%U.67d) = name_ref y, %y
// CHECK:STDOUT:     %SubAssignWith.type.loc233: type = facet_type <@SubAssignWith.1, @SubAssignWith.1(constants.%U.67d)> [symbolic = %SubAssignWith.type.loc231_52.1 (constants.%SubAssignWith.type.267e69.2)]
// CHECK:STDOUT:     %.loc233_5.1: @SubAssignWith.loc231.%SubAssignWith.assoc_type (%SubAssignWith.assoc_type.f4a281.2) = specific_constant imports.%Core.import_ref.3a2, @SubAssignWith.WithSelf(constants.%U.67d, constants.%Self.bd9eb5.1) [symbolic = %assoc0 (constants.%assoc0.556a96.2)]
// CHECK:STDOUT:     %Op.ref: @SubAssignWith.loc231.%SubAssignWith.assoc_type (%SubAssignWith.assoc_type.f4a281.2) = name_ref Op, %.loc233_5.1 [symbolic = %assoc0 (constants.%assoc0.556a96.2)]
// CHECK:STDOUT:     %impl.elem0.loc233_5.1: @SubAssignWith.loc231.%.loc233_5.2 (%.b89) = impl_witness_access constants.%SubAssignWith.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc233_5.2 (constants.%impl.elem0.f58)]
// CHECK:STDOUT:     %bound_method.loc233_5.1: <bound method> = bound_method %x.ref, %impl.elem0.loc233_5.1
// CHECK:STDOUT:     %specific_impl_fn.loc233_5.1: <specific function> = specific_impl_function %impl.elem0.loc233_5.1, @SubAssignWith.WithSelf.Op(constants.%U.67d, constants.%T.bd9) [symbolic = %specific_impl_fn.loc233_5.2 (constants.%specific_impl_fn.340)]
// CHECK:STDOUT:     %bound_method.loc233_5.2: <bound method> = bound_method %x.ref, %specific_impl_fn.loc233_5.1
// CHECK:STDOUT:     %SubAssignWith.WithSelf.Op.call: init %empty_tuple.type = call %bound_method.loc233_5.2(%x.ref, %y.ref)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @MulAssignWith.loc237(%U.loc237_19.2: type, %T.loc237_29.2: @MulAssignWith.loc237.%MulAssignWith.type.loc237_52.1 (%MulAssignWith.type.802f2e.2)) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %require_complete.loc239: <witness> = require_complete_type %MulAssignWith.type.loc237_52.1 [symbolic = %require_complete.loc239 (constants.%require_complete.e52)]
// CHECK:STDOUT:   %MulAssignWith.assoc_type: type = assoc_entity_type @MulAssignWith.1, @MulAssignWith.1(%U.loc237_19.1) [symbolic = %MulAssignWith.assoc_type (constants.%MulAssignWith.assoc_type.fda5e1.2)]
// CHECK:STDOUT:   %assoc0: @MulAssignWith.loc237.%MulAssignWith.assoc_type (%MulAssignWith.assoc_type.fda5e1.2) = assoc_entity element0, imports.%Core.import_ref.303 [symbolic = %assoc0 (constants.%assoc0.da1d40.2)]
// CHECK:STDOUT:   %MulAssignWith.WithSelf.Op.type: type = fn_type @MulAssignWith.WithSelf.Op, @MulAssignWith.WithSelf(%U.loc237_19.1, %T.loc237_29.1) [symbolic = %MulAssignWith.WithSelf.Op.type (constants.%MulAssignWith.WithSelf.Op.type.35745f.3)]
// CHECK:STDOUT:   %.loc239_5.2: type = fn_type_with_self_type %MulAssignWith.WithSelf.Op.type, %T.loc237_29.1 [symbolic = %.loc239_5.2 (constants.%.683)]
// CHECK:STDOUT:   %MulAssignWith.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc237_29.1, @MulAssignWith.1, @MulAssignWith.1(%U.loc237_19.1) [symbolic = %MulAssignWith.lookup_impl_witness (constants.%MulAssignWith.lookup_impl_witness)]
// CHECK:STDOUT:   %impl.elem0.loc239_5.2: @MulAssignWith.loc237.%.loc239_5.2 (%.683) = impl_witness_access %MulAssignWith.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc239_5.2 (constants.%impl.elem0.936)]
// CHECK:STDOUT:   %specific_impl_fn.loc239_5.2: <specific function> = specific_impl_function %impl.elem0.loc239_5.2, @MulAssignWith.WithSelf.Op(%U.loc237_19.1, %T.loc237_29.1) [symbolic = %specific_impl_fn.loc239_5.2 (constants.%specific_impl_fn.7bb)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: ref @MulAssignWith.loc237.%T.as_type.loc237_62.1 (%T.as_type.2ad), %y.param: @MulAssignWith.loc237.%U.loc237_19.1 (%U.67d)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: ref @MulAssignWith.loc237.%T.as_type.loc237_62.1 (%T.as_type.2ad) = name_ref x, %x
// CHECK:STDOUT:     %y.ref: @MulAssignWith.loc237.%U.loc237_19.1 (%U.67d) = name_ref y, %y
// CHECK:STDOUT:     %MulAssignWith.type.loc239: type = facet_type <@MulAssignWith.1, @MulAssignWith.1(constants.%U.67d)> [symbolic = %MulAssignWith.type.loc237_52.1 (constants.%MulAssignWith.type.802f2e.2)]
// CHECK:STDOUT:     %.loc239_5.1: @MulAssignWith.loc237.%MulAssignWith.assoc_type (%MulAssignWith.assoc_type.fda5e1.2) = specific_constant imports.%Core.import_ref.8ca, @MulAssignWith.WithSelf(constants.%U.67d, constants.%Self.b81696.1) [symbolic = %assoc0 (constants.%assoc0.da1d40.2)]
// CHECK:STDOUT:     %Op.ref: @MulAssignWith.loc237.%MulAssignWith.assoc_type (%MulAssignWith.assoc_type.fda5e1.2) = name_ref Op, %.loc239_5.1 [symbolic = %assoc0 (constants.%assoc0.da1d40.2)]
// CHECK:STDOUT:     %impl.elem0.loc239_5.1: @MulAssignWith.loc237.%.loc239_5.2 (%.683) = impl_witness_access constants.%MulAssignWith.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc239_5.2 (constants.%impl.elem0.936)]
// CHECK:STDOUT:     %bound_method.loc239_5.1: <bound method> = bound_method %x.ref, %impl.elem0.loc239_5.1
// CHECK:STDOUT:     %specific_impl_fn.loc239_5.1: <specific function> = specific_impl_function %impl.elem0.loc239_5.1, @MulAssignWith.WithSelf.Op(constants.%U.67d, constants.%T.b81) [symbolic = %specific_impl_fn.loc239_5.2 (constants.%specific_impl_fn.7bb)]
// CHECK:STDOUT:     %bound_method.loc239_5.2: <bound method> = bound_method %x.ref, %specific_impl_fn.loc239_5.1
// CHECK:STDOUT:     %MulAssignWith.WithSelf.Op.call: init %empty_tuple.type = call %bound_method.loc239_5.2(%x.ref, %y.ref)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @DivAssignWith.loc243(%U.loc243_19.2: type, %T.loc243_29.2: @DivAssignWith.loc243.%DivAssignWith.type.loc243_52.1 (%DivAssignWith.type.2e6641.2)) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %require_complete.loc245: <witness> = require_complete_type %DivAssignWith.type.loc243_52.1 [symbolic = %require_complete.loc245 (constants.%require_complete.b86)]
// CHECK:STDOUT:   %DivAssignWith.assoc_type: type = assoc_entity_type @DivAssignWith.1, @DivAssignWith.1(%U.loc243_19.1) [symbolic = %DivAssignWith.assoc_type (constants.%DivAssignWith.assoc_type.c34e7f.2)]
// CHECK:STDOUT:   %assoc0: @DivAssignWith.loc243.%DivAssignWith.assoc_type (%DivAssignWith.assoc_type.c34e7f.2) = assoc_entity element0, imports.%Core.import_ref.52b [symbolic = %assoc0 (constants.%assoc0.857e90.2)]
// CHECK:STDOUT:   %DivAssignWith.WithSelf.Op.type: type = fn_type @DivAssignWith.WithSelf.Op, @DivAssignWith.WithSelf(%U.loc243_19.1, %T.loc243_29.1) [symbolic = %DivAssignWith.WithSelf.Op.type (constants.%DivAssignWith.WithSelf.Op.type.8f43bb.3)]
// CHECK:STDOUT:   %.loc245_5.2: type = fn_type_with_self_type %DivAssignWith.WithSelf.Op.type, %T.loc243_29.1 [symbolic = %.loc245_5.2 (constants.%.c07)]
// CHECK:STDOUT:   %DivAssignWith.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc243_29.1, @DivAssignWith.1, @DivAssignWith.1(%U.loc243_19.1) [symbolic = %DivAssignWith.lookup_impl_witness (constants.%DivAssignWith.lookup_impl_witness)]
// CHECK:STDOUT:   %impl.elem0.loc245_5.2: @DivAssignWith.loc243.%.loc245_5.2 (%.c07) = impl_witness_access %DivAssignWith.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc245_5.2 (constants.%impl.elem0.537)]
// CHECK:STDOUT:   %specific_impl_fn.loc245_5.2: <specific function> = specific_impl_function %impl.elem0.loc245_5.2, @DivAssignWith.WithSelf.Op(%U.loc243_19.1, %T.loc243_29.1) [symbolic = %specific_impl_fn.loc245_5.2 (constants.%specific_impl_fn.f69)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: ref @DivAssignWith.loc243.%T.as_type.loc243_62.1 (%T.as_type.905), %y.param: @DivAssignWith.loc243.%U.loc243_19.1 (%U.67d)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: ref @DivAssignWith.loc243.%T.as_type.loc243_62.1 (%T.as_type.905) = name_ref x, %x
// CHECK:STDOUT:     %y.ref: @DivAssignWith.loc243.%U.loc243_19.1 (%U.67d) = name_ref y, %y
// CHECK:STDOUT:     %DivAssignWith.type.loc245: type = facet_type <@DivAssignWith.1, @DivAssignWith.1(constants.%U.67d)> [symbolic = %DivAssignWith.type.loc243_52.1 (constants.%DivAssignWith.type.2e6641.2)]
// CHECK:STDOUT:     %.loc245_5.1: @DivAssignWith.loc243.%DivAssignWith.assoc_type (%DivAssignWith.assoc_type.c34e7f.2) = specific_constant imports.%Core.import_ref.f7c, @DivAssignWith.WithSelf(constants.%U.67d, constants.%Self.e57cd6.1) [symbolic = %assoc0 (constants.%assoc0.857e90.2)]
// CHECK:STDOUT:     %Op.ref: @DivAssignWith.loc243.%DivAssignWith.assoc_type (%DivAssignWith.assoc_type.c34e7f.2) = name_ref Op, %.loc245_5.1 [symbolic = %assoc0 (constants.%assoc0.857e90.2)]
// CHECK:STDOUT:     %impl.elem0.loc245_5.1: @DivAssignWith.loc243.%.loc245_5.2 (%.c07) = impl_witness_access constants.%DivAssignWith.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc245_5.2 (constants.%impl.elem0.537)]
// CHECK:STDOUT:     %bound_method.loc245_5.1: <bound method> = bound_method %x.ref, %impl.elem0.loc245_5.1
// CHECK:STDOUT:     %specific_impl_fn.loc245_5.1: <specific function> = specific_impl_function %impl.elem0.loc245_5.1, @DivAssignWith.WithSelf.Op(constants.%U.67d, constants.%T.e57) [symbolic = %specific_impl_fn.loc245_5.2 (constants.%specific_impl_fn.f69)]
// CHECK:STDOUT:     %bound_method.loc245_5.2: <bound method> = bound_method %x.ref, %specific_impl_fn.loc245_5.1
// CHECK:STDOUT:     %DivAssignWith.WithSelf.Op.call: init %empty_tuple.type = call %bound_method.loc245_5.2(%x.ref, %y.ref)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @ModAssignWith.loc249(%U.loc249_19.2: type, %T.loc249_29.2: @ModAssignWith.loc249.%ModAssignWith.type.loc249_52.1 (%ModAssignWith.type.6d648f.2)) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %require_complete.loc251: <witness> = require_complete_type %ModAssignWith.type.loc249_52.1 [symbolic = %require_complete.loc251 (constants.%require_complete.869)]
// CHECK:STDOUT:   %ModAssignWith.assoc_type: type = assoc_entity_type @ModAssignWith.1, @ModAssignWith.1(%U.loc249_19.1) [symbolic = %ModAssignWith.assoc_type (constants.%ModAssignWith.assoc_type.0beb3b.2)]
// CHECK:STDOUT:   %assoc0: @ModAssignWith.loc249.%ModAssignWith.assoc_type (%ModAssignWith.assoc_type.0beb3b.2) = assoc_entity element0, imports.%Core.import_ref.a9d [symbolic = %assoc0 (constants.%assoc0.6957e8.2)]
// CHECK:STDOUT:   %ModAssignWith.WithSelf.Op.type: type = fn_type @ModAssignWith.WithSelf.Op, @ModAssignWith.WithSelf(%U.loc249_19.1, %T.loc249_29.1) [symbolic = %ModAssignWith.WithSelf.Op.type (constants.%ModAssignWith.WithSelf.Op.type.d01472.3)]
// CHECK:STDOUT:   %.loc251_5.2: type = fn_type_with_self_type %ModAssignWith.WithSelf.Op.type, %T.loc249_29.1 [symbolic = %.loc251_5.2 (constants.%.0dc)]
// CHECK:STDOUT:   %ModAssignWith.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc249_29.1, @ModAssignWith.1, @ModAssignWith.1(%U.loc249_19.1) [symbolic = %ModAssignWith.lookup_impl_witness (constants.%ModAssignWith.lookup_impl_witness)]
// CHECK:STDOUT:   %impl.elem0.loc251_5.2: @ModAssignWith.loc249.%.loc251_5.2 (%.0dc) = impl_witness_access %ModAssignWith.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc251_5.2 (constants.%impl.elem0.74f)]
// CHECK:STDOUT:   %specific_impl_fn.loc251_5.2: <specific function> = specific_impl_function %impl.elem0.loc251_5.2, @ModAssignWith.WithSelf.Op(%U.loc249_19.1, %T.loc249_29.1) [symbolic = %specific_impl_fn.loc251_5.2 (constants.%specific_impl_fn.875)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: ref @ModAssignWith.loc249.%T.as_type.loc249_62.1 (%T.as_type.ab6), %y.param: @ModAssignWith.loc249.%U.loc249_19.1 (%U.67d)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: ref @ModAssignWith.loc249.%T.as_type.loc249_62.1 (%T.as_type.ab6) = name_ref x, %x
// CHECK:STDOUT:     %y.ref: @ModAssignWith.loc249.%U.loc249_19.1 (%U.67d) = name_ref y, %y
// CHECK:STDOUT:     %ModAssignWith.type.loc251: type = facet_type <@ModAssignWith.1, @ModAssignWith.1(constants.%U.67d)> [symbolic = %ModAssignWith.type.loc249_52.1 (constants.%ModAssignWith.type.6d648f.2)]
// CHECK:STDOUT:     %.loc251_5.1: @ModAssignWith.loc249.%ModAssignWith.assoc_type (%ModAssignWith.assoc_type.0beb3b.2) = specific_constant imports.%Core.import_ref.042, @ModAssignWith.WithSelf(constants.%U.67d, constants.%Self.aac87d.1) [symbolic = %assoc0 (constants.%assoc0.6957e8.2)]
// CHECK:STDOUT:     %Op.ref: @ModAssignWith.loc249.%ModAssignWith.assoc_type (%ModAssignWith.assoc_type.0beb3b.2) = name_ref Op, %.loc251_5.1 [symbolic = %assoc0 (constants.%assoc0.6957e8.2)]
// CHECK:STDOUT:     %impl.elem0.loc251_5.1: @ModAssignWith.loc249.%.loc251_5.2 (%.0dc) = impl_witness_access constants.%ModAssignWith.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc251_5.2 (constants.%impl.elem0.74f)]
// CHECK:STDOUT:     %bound_method.loc251_5.1: <bound method> = bound_method %x.ref, %impl.elem0.loc251_5.1
// CHECK:STDOUT:     %specific_impl_fn.loc251_5.1: <specific function> = specific_impl_function %impl.elem0.loc251_5.1, @ModAssignWith.WithSelf.Op(constants.%U.67d, constants.%T.aac) [symbolic = %specific_impl_fn.loc251_5.2 (constants.%specific_impl_fn.875)]
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
// CHECK:STDOUT:   %Inc.WithSelf.Op.type: type = fn_type @Inc.WithSelf.Op, @Inc.WithSelf(%T.loc278_13.1) [symbolic = %Inc.WithSelf.Op.type (constants.%Inc.WithSelf.Op.type.667000.2)]
// CHECK:STDOUT:   %.loc280: type = fn_type_with_self_type %Inc.WithSelf.Op.type, %T.loc278_13.1 [symbolic = %.loc280 (constants.%.9f3)]
// CHECK:STDOUT:   %Inc.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc278_13.1, @Inc [symbolic = %Inc.lookup_impl_witness (constants.%Inc.lookup_impl_witness)]
// CHECK:STDOUT:   %impl.elem0.loc280_3.2: @TestInc.%.loc280 (%.9f3) = impl_witness_access %Inc.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc280_3.2 (constants.%impl.elem0.5bb)]
// CHECK:STDOUT:   %specific_impl_fn.loc280_3.2: <specific function> = specific_impl_function %impl.elem0.loc280_3.2, @Inc.WithSelf.Op(%T.loc278_13.1) [symbolic = %specific_impl_fn.loc280_3.2 (constants.%specific_impl_fn.4ac)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: ref @TestInc.%T.as_type.loc278_33.1 (%T.as_type.ec5)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: ref @TestInc.%T.as_type.loc278_33.1 (%T.as_type.ec5) = name_ref x, %x
// CHECK:STDOUT:     %impl.elem0.loc280_3.1: @TestInc.%.loc280 (%.9f3) = impl_witness_access constants.%Inc.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc280_3.2 (constants.%impl.elem0.5bb)]
// CHECK:STDOUT:     %bound_method.loc280_3.1: <bound method> = bound_method %x.ref, %impl.elem0.loc280_3.1
// CHECK:STDOUT:     %specific_impl_fn.loc280_3.1: <specific function> = specific_impl_function %impl.elem0.loc280_3.1, @Inc.WithSelf.Op(constants.%T.3d6) [symbolic = %specific_impl_fn.loc280_3.2 (constants.%specific_impl_fn.4ac)]
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
// CHECK:STDOUT:   %Dec.WithSelf.Op.type: type = fn_type @Dec.WithSelf.Op, @Dec.WithSelf(%T.loc284_13.1) [symbolic = %Dec.WithSelf.Op.type (constants.%Dec.WithSelf.Op.type.97f359.2)]
// CHECK:STDOUT:   %.loc286: type = fn_type_with_self_type %Dec.WithSelf.Op.type, %T.loc284_13.1 [symbolic = %.loc286 (constants.%.369)]
// CHECK:STDOUT:   %Dec.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc284_13.1, @Dec [symbolic = %Dec.lookup_impl_witness (constants.%Dec.lookup_impl_witness)]
// CHECK:STDOUT:   %impl.elem0.loc286_3.2: @TestDec.%.loc286 (%.369) = impl_witness_access %Dec.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc286_3.2 (constants.%impl.elem0.b37)]
// CHECK:STDOUT:   %specific_impl_fn.loc286_3.2: <specific function> = specific_impl_function %impl.elem0.loc286_3.2, @Dec.WithSelf.Op(%T.loc284_13.1) [symbolic = %specific_impl_fn.loc286_3.2 (constants.%specific_impl_fn.afd)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: ref @TestDec.%T.as_type.loc284_33.1 (%T.as_type.b80)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: ref @TestDec.%T.as_type.loc284_33.1 (%T.as_type.b80) = name_ref x, %x
// CHECK:STDOUT:     %impl.elem0.loc286_3.1: @TestDec.%.loc286 (%.369) = impl_witness_access constants.%Dec.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc286_3.2 (constants.%impl.elem0.b37)]
// CHECK:STDOUT:     %bound_method.loc286_3.1: <bound method> = bound_method %x.ref, %impl.elem0.loc286_3.1
// CHECK:STDOUT:     %specific_impl_fn.loc286_3.1: <specific function> = specific_impl_function %impl.elem0.loc286_3.1, @Dec.WithSelf.Op(constants.%T.cee) [symbolic = %specific_impl_fn.loc286_3.2 (constants.%specific_impl_fn.afd)]
// CHECK:STDOUT:     %bound_method.loc286_3.2: <bound method> = bound_method %x.ref, %specific_impl_fn.loc286_3.1
// CHECK:STDOUT:     %Dec.WithSelf.Op.call: init %empty_tuple.type = call %bound_method.loc286_3.2(%x.ref)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @TestNegate(%Result.loc292_9.2: %Destroy.type, %T.loc293_4.2: @TestNegate.%facet_type (%facet_type.5a6)) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Negate.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc293_4.1, @Negate [symbolic = %Negate.lookup_impl_witness (constants.%Negate.lookup_impl_witness.139)]
// CHECK:STDOUT:   %Negate.facet: %Negate.type = facet_value %T.as_type.loc294_24.1, (%Negate.lookup_impl_witness) [symbolic = %Negate.facet (constants.%Negate.facet.6d8)]
// CHECK:STDOUT:   %Negate.WithSelf.Op.type: type = fn_type @Negate.WithSelf.Op, @Negate.WithSelf(%Negate.facet) [symbolic = %Negate.WithSelf.Op.type (constants.%Negate.WithSelf.Op.type.bcd)]
// CHECK:STDOUT:   %.loc296_3.3: type = fn_type_with_self_type %Negate.WithSelf.Op.type, %Negate.facet [symbolic = %.loc296_3.3 (constants.%.7a2)]
// CHECK:STDOUT:   %impl.elem1.loc296_3.2: @TestNegate.%.loc296_3.3 (%.7a2) = impl_witness_access %Negate.lookup_impl_witness, element1 [symbolic = %impl.elem1.loc296_3.2 (constants.%impl.elem1.2c5)]
// CHECK:STDOUT:   %specific_impl_fn.loc296_3.3: <specific function> = specific_impl_function %impl.elem1.loc296_3.2, @Negate.WithSelf.Op(%Negate.facet) [symbolic = %specific_impl_fn.loc296_3.3 (constants.%specific_impl_fn.cfb)]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Result.loc292_9.1) [symbolic = %Destroy.WithSelf.Op.type (constants.%Destroy.WithSelf.Op.type.b9d4b3.2)]
// CHECK:STDOUT:   %.loc296_3.4: type = fn_type_with_self_type %Destroy.WithSelf.Op.type, %Result.loc292_9.1 [symbolic = %.loc296_3.4 (constants.%.387)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %Result.loc292_9.1, @Destroy [symbolic = %Destroy.lookup_impl_witness (constants.%Destroy.lookup_impl_witness)]
// CHECK:STDOUT:   %impl.elem0.loc296_3.2: @TestNegate.%.loc296_3.4 (%.387) = impl_witness_access %Destroy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc296_3.2 (constants.%impl.elem0.2e6)]
// CHECK:STDOUT:   %specific_impl_fn.loc296_3.4: <specific function> = specific_impl_function %impl.elem0.loc296_3.2, @Destroy.WithSelf.Op(%Result.loc292_9.1) [symbolic = %specific_impl_fn.loc296_3.4 (constants.%specific_impl_fn.62a)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%y.param: @TestNegate.%Result.as_type.loc293_50.1 (%Result.as_type), %x.param: @TestNegate.%T.as_type.loc294_24.1 (%T.as_type.55b)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: @TestNegate.%T.as_type.loc294_24.1 (%T.as_type.55b) = name_ref x, %x
// CHECK:STDOUT:     %impl.elem1.loc296_3.1: @TestNegate.%.loc296_3.3 (%.7a2) = impl_witness_access constants.%Negate.lookup_impl_witness.139, element1 [symbolic = %impl.elem1.loc296_3.2 (constants.%impl.elem1.2c5)]
// CHECK:STDOUT:     %bound_method.loc296_3.1: <bound method> = bound_method %x.ref, %impl.elem1.loc296_3.1
// CHECK:STDOUT:     %specific_impl_fn.loc296_3.1: <specific function> = specific_impl_function %impl.elem1.loc296_3.1, @Negate.WithSelf.Op(constants.%Negate.facet.6d8) [symbolic = %specific_impl_fn.loc296_3.3 (constants.%specific_impl_fn.cfb)]
// CHECK:STDOUT:     %bound_method.loc296_3.2: <bound method> = bound_method %x.ref, %specific_impl_fn.loc296_3.1
// CHECK:STDOUT:     %.loc296_3.1: ref @TestNegate.%Result.as_type.loc293_50.1 (%Result.as_type) = temporary_storage
// CHECK:STDOUT:     %Negate.WithSelf.Op.call: init @TestNegate.%Result.as_type.loc293_50.1 (%Result.as_type) to %.loc296_3.1 = call %bound_method.loc296_3.2(%x.ref)
// CHECK:STDOUT:     %.loc296_3.2: ref @TestNegate.%Result.as_type.loc293_50.1 (%Result.as_type) = temporary %.loc296_3.1, %Negate.WithSelf.Op.call
// CHECK:STDOUT:     %impl.elem0.loc296_3.1: @TestNegate.%.loc296_3.4 (%.387) = impl_witness_access constants.%Destroy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc296_3.2 (constants.%impl.elem0.2e6)]
// CHECK:STDOUT:     %bound_method.loc296_3.3: <bound method> = bound_method %.loc296_3.2, %impl.elem0.loc296_3.1
// CHECK:STDOUT:     %specific_impl_fn.loc296_3.2: <specific function> = specific_impl_function %impl.elem0.loc296_3.1, @Destroy.WithSelf.Op(constants.%Result) [symbolic = %specific_impl_fn.loc296_3.4 (constants.%specific_impl_fn.62a)]
// CHECK:STDOUT:     %bound_method.loc296_3.4: <bound method> = bound_method %.loc296_3.2, %specific_impl_fn.loc296_3.2
// CHECK:STDOUT:     %Destroy.WithSelf.Op.call: init %empty_tuple.type = call %bound_method.loc296_3.4(%.loc296_3.2)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AddWith.loc124(constants.%Result, constants.%U.ac3, constants.%T.db8) {
// CHECK:STDOUT:   %Result.loc121_9.1 => constants.%Result
// CHECK:STDOUT:   %U.loc122_4.1 => constants.%U.ac3
// CHECK:STDOUT:   %U.as_type.loc123_36.1 => constants.%U.as_type.04c
// CHECK:STDOUT:   %AddWith.type.loc123_36.1 => constants.%AddWith.type.e1b
// CHECK:STDOUT:   %facet_type.loc123_20 => constants.%facet_type.e9c
// CHECK:STDOUT:   %.Self.loc123_38.1 => constants.%.Self.33f
// CHECK:STDOUT:   %require_complete.loc123 => constants.%require_complete.b5e
// CHECK:STDOUT:   %.Self.as_type.loc123_44.1 => constants.%.Self.as_type.bf0
// CHECK:STDOUT:   %AddWith.assoc_type => constants.%AddWith.assoc_type.91f
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.535
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.loc123 => constants.%AddWith.lookup_impl_witness.82b
// CHECK:STDOUT:   %impl.elem0.loc123_44.1 => constants.%impl.elem0.f47
// CHECK:STDOUT:   %Result.as_type.loc123_54.1 => constants.%Result.as_type
// CHECK:STDOUT:   %facet_type.loc123_38 => constants.%facet_type.f91
// CHECK:STDOUT:   %T.loc123_4.1 => constants.%T.db8
// CHECK:STDOUT:   %pattern_type.loc123 => constants.%pattern_type.ed1
// CHECK:STDOUT:   %pattern_type.loc124_16 => constants.%pattern_type.96123d.2
// CHECK:STDOUT:   %T.as_type.loc124_29.1 => constants.%T.as_type.94c
// CHECK:STDOUT:   %pattern_type.loc124_27 => constants.%pattern_type.add
// CHECK:STDOUT:   %pattern_type.loc124_33 => constants.%pattern_type.c6d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @SubWith.loc135(constants.%Result, constants.%U.ac3, constants.%T.813) {
// CHECK:STDOUT:   %Result.loc132_9.1 => constants.%Result
// CHECK:STDOUT:   %U.loc133_4.1 => constants.%U.ac3
// CHECK:STDOUT:   %U.as_type.loc134_36.1 => constants.%U.as_type.04c
// CHECK:STDOUT:   %SubWith.type.loc134_36.1 => constants.%SubWith.type.2e8
// CHECK:STDOUT:   %facet_type.loc134_20 => constants.%facet_type.241
// CHECK:STDOUT:   %.Self.loc134_38.1 => constants.%.Self.a38
// CHECK:STDOUT:   %require_complete.loc134 => constants.%require_complete.250
// CHECK:STDOUT:   %.Self.as_type.loc134_44.1 => constants.%.Self.as_type.2cd
// CHECK:STDOUT:   %SubWith.assoc_type => constants.%SubWith.assoc_type.760
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.d73
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.loc134 => constants.%SubWith.lookup_impl_witness.cb4
// CHECK:STDOUT:   %impl.elem0.loc134_44.1 => constants.%impl.elem0.b99
// CHECK:STDOUT:   %Result.as_type.loc134_54.1 => constants.%Result.as_type
// CHECK:STDOUT:   %facet_type.loc134_38 => constants.%facet_type.a29
// CHECK:STDOUT:   %T.loc134_4.1 => constants.%T.813
// CHECK:STDOUT:   %pattern_type.loc134 => constants.%pattern_type.095
// CHECK:STDOUT:   %pattern_type.loc135_16 => constants.%pattern_type.96123d.2
// CHECK:STDOUT:   %T.as_type.loc135_29.1 => constants.%T.as_type.803
// CHECK:STDOUT:   %pattern_type.loc135_27 => constants.%pattern_type.d56
// CHECK:STDOUT:   %pattern_type.loc135_33 => constants.%pattern_type.c6d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @MulWith.loc146(constants.%Result, constants.%U.ac3, constants.%T.c84) {
// CHECK:STDOUT:   %Result.loc143_9.1 => constants.%Result
// CHECK:STDOUT:   %U.loc144_4.1 => constants.%U.ac3
// CHECK:STDOUT:   %U.as_type.loc145_36.1 => constants.%U.as_type.04c
// CHECK:STDOUT:   %MulWith.type.loc145_36.1 => constants.%MulWith.type.293
// CHECK:STDOUT:   %facet_type.loc145_20 => constants.%facet_type.3d2
// CHECK:STDOUT:   %.Self.loc145_38.1 => constants.%.Self.b95
// CHECK:STDOUT:   %require_complete.loc145 => constants.%require_complete.ac4
// CHECK:STDOUT:   %.Self.as_type.loc145_44.1 => constants.%.Self.as_type.3e1
// CHECK:STDOUT:   %MulWith.assoc_type => constants.%MulWith.assoc_type.8c9
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.007
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.loc145 => constants.%MulWith.lookup_impl_witness.ca1
// CHECK:STDOUT:   %impl.elem0.loc145_44.1 => constants.%impl.elem0.975
// CHECK:STDOUT:   %Result.as_type.loc145_54.1 => constants.%Result.as_type
// CHECK:STDOUT:   %facet_type.loc145_38 => constants.%facet_type.ec5
// CHECK:STDOUT:   %T.loc145_4.1 => constants.%T.c84
// CHECK:STDOUT:   %pattern_type.loc145 => constants.%pattern_type.91b
// CHECK:STDOUT:   %pattern_type.loc146_16 => constants.%pattern_type.96123d.2
// CHECK:STDOUT:   %T.as_type.loc146_29.1 => constants.%T.as_type.0c1
// CHECK:STDOUT:   %pattern_type.loc146_27 => constants.%pattern_type.ac7
// CHECK:STDOUT:   %pattern_type.loc146_33 => constants.%pattern_type.c6d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @DivWith.loc157(constants.%Result, constants.%U.ac3, constants.%T.81b) {
// CHECK:STDOUT:   %Result.loc154_9.1 => constants.%Result
// CHECK:STDOUT:   %U.loc155_4.1 => constants.%U.ac3
// CHECK:STDOUT:   %U.as_type.loc156_36.1 => constants.%U.as_type.04c
// CHECK:STDOUT:   %DivWith.type.loc156_36.1 => constants.%DivWith.type.25a
// CHECK:STDOUT:   %facet_type.loc156_20 => constants.%facet_type.0d2
// CHECK:STDOUT:   %.Self.loc156_38.1 => constants.%.Self.db2
// CHECK:STDOUT:   %require_complete.loc156 => constants.%require_complete.770
// CHECK:STDOUT:   %.Self.as_type.loc156_44.1 => constants.%.Self.as_type.67d
// CHECK:STDOUT:   %DivWith.assoc_type => constants.%DivWith.assoc_type.b3b
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.f1d
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.loc156 => constants.%DivWith.lookup_impl_witness.a3e
// CHECK:STDOUT:   %impl.elem0.loc156_44.1 => constants.%impl.elem0.f72
// CHECK:STDOUT:   %Result.as_type.loc156_54.1 => constants.%Result.as_type
// CHECK:STDOUT:   %facet_type.loc156_38 => constants.%facet_type.b8a
// CHECK:STDOUT:   %T.loc156_4.1 => constants.%T.81b
// CHECK:STDOUT:   %pattern_type.loc156 => constants.%pattern_type.e40
// CHECK:STDOUT:   %pattern_type.loc157_16 => constants.%pattern_type.96123d.2
// CHECK:STDOUT:   %T.as_type.loc157_29.1 => constants.%T.as_type.2d1
// CHECK:STDOUT:   %pattern_type.loc157_27 => constants.%pattern_type.4c5
// CHECK:STDOUT:   %pattern_type.loc157_33 => constants.%pattern_type.c6d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ModWith.loc168(constants.%Result, constants.%U.ac3, constants.%T.ddd) {
// CHECK:STDOUT:   %Result.loc165_9.1 => constants.%Result
// CHECK:STDOUT:   %U.loc166_4.1 => constants.%U.ac3
// CHECK:STDOUT:   %U.as_type.loc167_36.1 => constants.%U.as_type.04c
// CHECK:STDOUT:   %ModWith.type.loc167_36.1 => constants.%ModWith.type.3cd
// CHECK:STDOUT:   %facet_type.loc167_20 => constants.%facet_type.8d7
// CHECK:STDOUT:   %.Self.loc167_38.1 => constants.%.Self.b9d
// CHECK:STDOUT:   %require_complete.loc167 => constants.%require_complete.ba5
// CHECK:STDOUT:   %.Self.as_type.loc167_44.1 => constants.%.Self.as_type.25e
// CHECK:STDOUT:   %ModWith.assoc_type => constants.%ModWith.assoc_type.84f
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.bf7
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.loc167 => constants.%ModWith.lookup_impl_witness.271
// CHECK:STDOUT:   %impl.elem0.loc167_44.1 => constants.%impl.elem0.4a9
// CHECK:STDOUT:   %Result.as_type.loc167_54.1 => constants.%Result.as_type
// CHECK:STDOUT:   %facet_type.loc167_38 => constants.%facet_type.c9c
// CHECK:STDOUT:   %T.loc167_4.1 => constants.%T.ddd
// CHECK:STDOUT:   %pattern_type.loc167 => constants.%pattern_type.cf5
// CHECK:STDOUT:   %pattern_type.loc168_16 => constants.%pattern_type.96123d.2
// CHECK:STDOUT:   %T.as_type.loc168_29.1 => constants.%T.as_type.2fa
// CHECK:STDOUT:   %pattern_type.loc168_27 => constants.%pattern_type.e09
// CHECK:STDOUT:   %pattern_type.loc168_33 => constants.%pattern_type.c6d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AddWith.loc124(constants.%Destroy.facet.551, constants.%Destroy.facet.551, constants.%facet_value.227e) {
// CHECK:STDOUT:   %Result.loc121_9.1 => constants.%Destroy.facet.551
// CHECK:STDOUT:   %U.loc122_4.1 => constants.%Destroy.facet.551
// CHECK:STDOUT:   %U.as_type.loc123_36.1 => constants.%Int16
// CHECK:STDOUT:   %AddWith.type.loc123_36.1 => constants.%AddWith.type.41f
// CHECK:STDOUT:   %facet_type.loc123_20 => constants.%facet_type.b25
// CHECK:STDOUT:   %.Self.loc123_38.1 => constants.%.Self.cee
// CHECK:STDOUT:   %require_complete.loc123 => constants.%complete_type.08f
// CHECK:STDOUT:   %.Self.as_type.loc123_44.1 => constants.%.Self.as_type.38a
// CHECK:STDOUT:   %AddWith.assoc_type => constants.%AddWith.assoc_type.5ee
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.95b
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.loc123 => constants.%AddWith.lookup_impl_witness.ef3
// CHECK:STDOUT:   %impl.elem0.loc123_44.1 => constants.%impl.elem0.1ef
// CHECK:STDOUT:   %Result.as_type.loc123_54.1 => constants.%Int16
// CHECK:STDOUT:   %facet_type.loc123_38 => constants.%facet_type.eb1
// CHECK:STDOUT:   %T.loc123_4.1 => constants.%facet_value.227e
// CHECK:STDOUT:   %pattern_type.loc123 => constants.%pattern_type.748
// CHECK:STDOUT:   %pattern_type.loc124_16 => constants.%pattern_type.b06
// CHECK:STDOUT:   %T.as_type.loc124_29.1 => constants.%i16
// CHECK:STDOUT:   %pattern_type.loc124_27 => constants.%pattern_type.0e3
// CHECK:STDOUT:   %pattern_type.loc124_33 => constants.%pattern_type.b06
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc124_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc124_27 => constants.%complete_type.2da
// CHECK:STDOUT:   %require_complete.loc124_33 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc126 => constants.%complete_type.f1a
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.41e
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.loc126 => constants.%custom_witness.069
// CHECK:STDOUT:   %AddWith.facet => constants.%AddWith.facet.02b
// CHECK:STDOUT:   %AddWith.WithSelf.Op.type => constants.%AddWith.WithSelf.Op.type.bba
// CHECK:STDOUT:   %.loc126_5.4 => constants.%.130
// CHECK:STDOUT:   %impl.elem1.loc126_5.2 => constants.%Op.848e8c.1
// CHECK:STDOUT:   %specific_impl_fn.loc126_5.3 => constants.%Op.848e8c.1
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.9f9
// CHECK:STDOUT:   %.loc126_5.5 => constants.%.16b
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.ecd0a5.1
// CHECK:STDOUT:   %impl.elem0.loc126_5.2 => constants.%Int16.Op.7ce432.1
// CHECK:STDOUT:   %specific_impl_fn.loc126_5.4 => constants.%Int16.Op.7ce432.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @SubWith.loc135(constants.%Destroy.facet.551, constants.%Destroy.facet.551, constants.%facet_value.d1c) {
// CHECK:STDOUT:   %Result.loc132_9.1 => constants.%Destroy.facet.551
// CHECK:STDOUT:   %U.loc133_4.1 => constants.%Destroy.facet.551
// CHECK:STDOUT:   %U.as_type.loc134_36.1 => constants.%Int16
// CHECK:STDOUT:   %SubWith.type.loc134_36.1 => constants.%SubWith.type.f7a
// CHECK:STDOUT:   %facet_type.loc134_20 => constants.%facet_type.5bd
// CHECK:STDOUT:   %.Self.loc134_38.1 => constants.%.Self.0bf
// CHECK:STDOUT:   %require_complete.loc134 => constants.%complete_type.4b4
// CHECK:STDOUT:   %.Self.as_type.loc134_44.1 => constants.%.Self.as_type.752
// CHECK:STDOUT:   %SubWith.assoc_type => constants.%SubWith.assoc_type.e4b
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.f15
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.loc134 => constants.%SubWith.lookup_impl_witness.9d6
// CHECK:STDOUT:   %impl.elem0.loc134_44.1 => constants.%impl.elem0.214
// CHECK:STDOUT:   %Result.as_type.loc134_54.1 => constants.%Int16
// CHECK:STDOUT:   %facet_type.loc134_38 => constants.%facet_type.7bc
// CHECK:STDOUT:   %T.loc134_4.1 => constants.%facet_value.d1c
// CHECK:STDOUT:   %pattern_type.loc134 => constants.%pattern_type.ca1
// CHECK:STDOUT:   %pattern_type.loc135_16 => constants.%pattern_type.b06
// CHECK:STDOUT:   %T.as_type.loc135_29.1 => constants.%i16
// CHECK:STDOUT:   %pattern_type.loc135_27 => constants.%pattern_type.0e3
// CHECK:STDOUT:   %pattern_type.loc135_33 => constants.%pattern_type.b06
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc135_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc135_27 => constants.%complete_type.2da
// CHECK:STDOUT:   %require_complete.loc135_33 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc137 => constants.%complete_type.c17
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.8a0
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.loc137 => constants.%custom_witness.2c8
// CHECK:STDOUT:   %SubWith.facet => constants.%SubWith.facet.f1c
// CHECK:STDOUT:   %SubWith.WithSelf.Op.type => constants.%SubWith.WithSelf.Op.type.4dd
// CHECK:STDOUT:   %.loc137_5.4 => constants.%.859
// CHECK:STDOUT:   %impl.elem1.loc137_5.2 => constants.%Op.848e8c.2
// CHECK:STDOUT:   %specific_impl_fn.loc137_5.3 => constants.%Op.848e8c.2
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.9f9
// CHECK:STDOUT:   %.loc137_5.5 => constants.%.16b
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.ecd0a5.1
// CHECK:STDOUT:   %impl.elem0.loc137_5.2 => constants.%Int16.Op.7ce432.1
// CHECK:STDOUT:   %specific_impl_fn.loc137_5.4 => constants.%Int16.Op.7ce432.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @MulWith.loc146(constants.%Destroy.facet.551, constants.%Destroy.facet.551, constants.%facet_value.b66) {
// CHECK:STDOUT:   %Result.loc143_9.1 => constants.%Destroy.facet.551
// CHECK:STDOUT:   %U.loc144_4.1 => constants.%Destroy.facet.551
// CHECK:STDOUT:   %U.as_type.loc145_36.1 => constants.%Int16
// CHECK:STDOUT:   %MulWith.type.loc145_36.1 => constants.%MulWith.type.ebe
// CHECK:STDOUT:   %facet_type.loc145_20 => constants.%facet_type.b8b
// CHECK:STDOUT:   %.Self.loc145_38.1 => constants.%.Self.117
// CHECK:STDOUT:   %require_complete.loc145 => constants.%complete_type.25f
// CHECK:STDOUT:   %.Self.as_type.loc145_44.1 => constants.%.Self.as_type.ad8
// CHECK:STDOUT:   %MulWith.assoc_type => constants.%MulWith.assoc_type.286
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.52c
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.loc145 => constants.%MulWith.lookup_impl_witness.485
// CHECK:STDOUT:   %impl.elem0.loc145_44.1 => constants.%impl.elem0.851
// CHECK:STDOUT:   %Result.as_type.loc145_54.1 => constants.%Int16
// CHECK:STDOUT:   %facet_type.loc145_38 => constants.%facet_type.166
// CHECK:STDOUT:   %T.loc145_4.1 => constants.%facet_value.b66
// CHECK:STDOUT:   %pattern_type.loc145 => constants.%pattern_type.380
// CHECK:STDOUT:   %pattern_type.loc146_16 => constants.%pattern_type.b06
// CHECK:STDOUT:   %T.as_type.loc146_29.1 => constants.%i16
// CHECK:STDOUT:   %pattern_type.loc146_27 => constants.%pattern_type.0e3
// CHECK:STDOUT:   %pattern_type.loc146_33 => constants.%pattern_type.b06
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc146_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc146_27 => constants.%complete_type.2da
// CHECK:STDOUT:   %require_complete.loc146_33 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc148 => constants.%complete_type.b33
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.0bf
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.loc148 => constants.%custom_witness.23bf
// CHECK:STDOUT:   %MulWith.facet => constants.%MulWith.facet.f7a
// CHECK:STDOUT:   %MulWith.WithSelf.Op.type => constants.%MulWith.WithSelf.Op.type.358
// CHECK:STDOUT:   %.loc148_5.4 => constants.%.833
// CHECK:STDOUT:   %impl.elem1.loc148_5.2 => constants.%Op.848e8c.3
// CHECK:STDOUT:   %specific_impl_fn.loc148_5.3 => constants.%Op.848e8c.3
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.9f9
// CHECK:STDOUT:   %.loc148_5.5 => constants.%.16b
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.ecd0a5.1
// CHECK:STDOUT:   %impl.elem0.loc148_5.2 => constants.%Int16.Op.7ce432.1
// CHECK:STDOUT:   %specific_impl_fn.loc148_5.4 => constants.%Int16.Op.7ce432.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @DivWith.loc157(constants.%Destroy.facet.551, constants.%Destroy.facet.551, constants.%facet_value.f9d) {
// CHECK:STDOUT:   %Result.loc154_9.1 => constants.%Destroy.facet.551
// CHECK:STDOUT:   %U.loc155_4.1 => constants.%Destroy.facet.551
// CHECK:STDOUT:   %U.as_type.loc156_36.1 => constants.%Int16
// CHECK:STDOUT:   %DivWith.type.loc156_36.1 => constants.%DivWith.type.996
// CHECK:STDOUT:   %facet_type.loc156_20 => constants.%facet_type.078
// CHECK:STDOUT:   %.Self.loc156_38.1 => constants.%.Self.f7a
// CHECK:STDOUT:   %require_complete.loc156 => constants.%complete_type.4f9
// CHECK:STDOUT:   %.Self.as_type.loc156_44.1 => constants.%.Self.as_type.91d
// CHECK:STDOUT:   %DivWith.assoc_type => constants.%DivWith.assoc_type.ae8
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.7b6
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.loc156 => constants.%DivWith.lookup_impl_witness.922
// CHECK:STDOUT:   %impl.elem0.loc156_44.1 => constants.%impl.elem0.1ae
// CHECK:STDOUT:   %Result.as_type.loc156_54.1 => constants.%Int16
// CHECK:STDOUT:   %facet_type.loc156_38 => constants.%facet_type.9a8
// CHECK:STDOUT:   %T.loc156_4.1 => constants.%facet_value.f9d
// CHECK:STDOUT:   %pattern_type.loc156 => constants.%pattern_type.515
// CHECK:STDOUT:   %pattern_type.loc157_16 => constants.%pattern_type.b06
// CHECK:STDOUT:   %T.as_type.loc157_29.1 => constants.%i16
// CHECK:STDOUT:   %pattern_type.loc157_27 => constants.%pattern_type.0e3
// CHECK:STDOUT:   %pattern_type.loc157_33 => constants.%pattern_type.b06
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc157_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc157_27 => constants.%complete_type.2da
// CHECK:STDOUT:   %require_complete.loc157_33 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc159 => constants.%complete_type.691
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.f9f
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.loc159 => constants.%custom_witness.ff6
// CHECK:STDOUT:   %DivWith.facet => constants.%DivWith.facet.d55
// CHECK:STDOUT:   %DivWith.WithSelf.Op.type => constants.%DivWith.WithSelf.Op.type.822
// CHECK:STDOUT:   %.loc159_5.4 => constants.%.65a
// CHECK:STDOUT:   %impl.elem1.loc159_5.2 => constants.%Op.848e8c.4
// CHECK:STDOUT:   %specific_impl_fn.loc159_5.3 => constants.%Op.848e8c.4
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.9f9
// CHECK:STDOUT:   %.loc159_5.5 => constants.%.16b
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.ecd0a5.1
// CHECK:STDOUT:   %impl.elem0.loc159_5.2 => constants.%Int16.Op.7ce432.1
// CHECK:STDOUT:   %specific_impl_fn.loc159_5.4 => constants.%Int16.Op.7ce432.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ModWith.loc168(constants.%Destroy.facet.551, constants.%Destroy.facet.551, constants.%facet_value.eed) {
// CHECK:STDOUT:   %Result.loc165_9.1 => constants.%Destroy.facet.551
// CHECK:STDOUT:   %U.loc166_4.1 => constants.%Destroy.facet.551
// CHECK:STDOUT:   %U.as_type.loc167_36.1 => constants.%Int16
// CHECK:STDOUT:   %ModWith.type.loc167_36.1 => constants.%ModWith.type.d4b
// CHECK:STDOUT:   %facet_type.loc167_20 => constants.%facet_type.dc0
// CHECK:STDOUT:   %.Self.loc167_38.1 => constants.%.Self.c9e
// CHECK:STDOUT:   %require_complete.loc167 => constants.%complete_type.10f
// CHECK:STDOUT:   %.Self.as_type.loc167_44.1 => constants.%.Self.as_type.fc8
// CHECK:STDOUT:   %ModWith.assoc_type => constants.%ModWith.assoc_type.8c2
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.262
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.loc167 => constants.%ModWith.lookup_impl_witness.9ea
// CHECK:STDOUT:   %impl.elem0.loc167_44.1 => constants.%impl.elem0.7c3
// CHECK:STDOUT:   %Result.as_type.loc167_54.1 => constants.%Int16
// CHECK:STDOUT:   %facet_type.loc167_38 => constants.%facet_type.3d1
// CHECK:STDOUT:   %T.loc167_4.1 => constants.%facet_value.eed
// CHECK:STDOUT:   %pattern_type.loc167 => constants.%pattern_type.9dd
// CHECK:STDOUT:   %pattern_type.loc168_16 => constants.%pattern_type.b06
// CHECK:STDOUT:   %T.as_type.loc168_29.1 => constants.%i16
// CHECK:STDOUT:   %pattern_type.loc168_27 => constants.%pattern_type.0e3
// CHECK:STDOUT:   %pattern_type.loc168_33 => constants.%pattern_type.b06
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc168_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc168_27 => constants.%complete_type.2da
// CHECK:STDOUT:   %require_complete.loc168_33 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc170 => constants.%complete_type.3a6
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.63a
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.loc170 => constants.%custom_witness.74e
// CHECK:STDOUT:   %ModWith.facet => constants.%ModWith.facet.bab
// CHECK:STDOUT:   %ModWith.WithSelf.Op.type => constants.%ModWith.WithSelf.Op.type.f9f2
// CHECK:STDOUT:   %.loc170_5.4 => constants.%.ea5
// CHECK:STDOUT:   %impl.elem1.loc170_5.2 => constants.%Op.848e8c.5
// CHECK:STDOUT:   %specific_impl_fn.loc170_5.3 => constants.%Op.848e8c.5
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.9f9
// CHECK:STDOUT:   %.loc170_5.5 => constants.%.16b
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.ecd0a5.1
// CHECK:STDOUT:   %impl.elem0.loc170_5.2 => constants.%Int16.Op.7ce432.1
// CHECK:STDOUT:   %specific_impl_fn.loc170_5.4 => constants.%Int16.Op.7ce432.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AddWith.loc124(constants.%Destroy.facet.551, constants.%Destroy.facet.0e0, constants.%facet_value.511) {
// CHECK:STDOUT:   %Result.loc121_9.1 => constants.%Destroy.facet.551
// CHECK:STDOUT:   %U.loc122_4.1 => constants.%Destroy.facet.0e0
// CHECK:STDOUT:   %U.as_type.loc123_36.1 => constants.%i16
// CHECK:STDOUT:   %AddWith.type.loc123_36.1 => constants.%AddWith.type.c9a
// CHECK:STDOUT:   %facet_type.loc123_20 => constants.%facet_type.d65
// CHECK:STDOUT:   %.Self.loc123_38.1 => constants.%.Self.0d7
// CHECK:STDOUT:   %require_complete.loc123 => constants.%complete_type.aae1
// CHECK:STDOUT:   %.Self.as_type.loc123_44.1 => constants.%.Self.as_type.578
// CHECK:STDOUT:   %AddWith.assoc_type => constants.%AddWith.assoc_type.67c
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.91b
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.loc123 => constants.%AddWith.lookup_impl_witness.750
// CHECK:STDOUT:   %impl.elem0.loc123_44.1 => constants.%impl.elem0.ed3
// CHECK:STDOUT:   %Result.as_type.loc123_54.1 => constants.%Int16
// CHECK:STDOUT:   %facet_type.loc123_38 => constants.%facet_type.9d0
// CHECK:STDOUT:   %T.loc123_4.1 => constants.%facet_value.511
// CHECK:STDOUT:   %pattern_type.loc123 => constants.%pattern_type.366
// CHECK:STDOUT:   %pattern_type.loc124_16 => constants.%pattern_type.b06
// CHECK:STDOUT:   %T.as_type.loc124_29.1 => constants.%Int16
// CHECK:STDOUT:   %pattern_type.loc124_27 => constants.%pattern_type.b06
// CHECK:STDOUT:   %pattern_type.loc124_33 => constants.%pattern_type.0e3
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc124_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc124_27 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc124_33 => constants.%complete_type.2da
// CHECK:STDOUT:   %require_complete.loc126 => constants.%complete_type.aae5
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.e96
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.loc126 => constants.%custom_witness.067
// CHECK:STDOUT:   %AddWith.facet => constants.%AddWith.facet.9a7
// CHECK:STDOUT:   %AddWith.WithSelf.Op.type => constants.%AddWith.WithSelf.Op.type.d41
// CHECK:STDOUT:   %.loc126_5.4 => constants.%.a6b
// CHECK:STDOUT:   %impl.elem1.loc126_5.2 => constants.%Int16.Op.7ce432.3
// CHECK:STDOUT:   %specific_impl_fn.loc126_5.3 => constants.%Int16.Op.7ce432.3
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.9f9
// CHECK:STDOUT:   %.loc126_5.5 => constants.%.16b
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.ecd0a5.1
// CHECK:STDOUT:   %impl.elem0.loc126_5.2 => constants.%Int16.Op.7ce432.1
// CHECK:STDOUT:   %specific_impl_fn.loc126_5.4 => constants.%Int16.Op.7ce432.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @SubWith.loc135(constants.%Destroy.facet.551, constants.%Destroy.facet.0e0, constants.%facet_value.70a) {
// CHECK:STDOUT:   %Result.loc132_9.1 => constants.%Destroy.facet.551
// CHECK:STDOUT:   %U.loc133_4.1 => constants.%Destroy.facet.0e0
// CHECK:STDOUT:   %U.as_type.loc134_36.1 => constants.%i16
// CHECK:STDOUT:   %SubWith.type.loc134_36.1 => constants.%SubWith.type.191
// CHECK:STDOUT:   %facet_type.loc134_20 => constants.%facet_type.21e
// CHECK:STDOUT:   %.Self.loc134_38.1 => constants.%.Self.8c7
// CHECK:STDOUT:   %require_complete.loc134 => constants.%complete_type.3a7
// CHECK:STDOUT:   %.Self.as_type.loc134_44.1 => constants.%.Self.as_type.ef8
// CHECK:STDOUT:   %SubWith.assoc_type => constants.%SubWith.assoc_type.b4a
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.ecf
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.loc134 => constants.%SubWith.lookup_impl_witness.2bb
// CHECK:STDOUT:   %impl.elem0.loc134_44.1 => constants.%impl.elem0.759
// CHECK:STDOUT:   %Result.as_type.loc134_54.1 => constants.%Int16
// CHECK:STDOUT:   %facet_type.loc134_38 => constants.%facet_type.54a
// CHECK:STDOUT:   %T.loc134_4.1 => constants.%facet_value.70a
// CHECK:STDOUT:   %pattern_type.loc134 => constants.%pattern_type.a0a
// CHECK:STDOUT:   %pattern_type.loc135_16 => constants.%pattern_type.b06
// CHECK:STDOUT:   %T.as_type.loc135_29.1 => constants.%Int16
// CHECK:STDOUT:   %pattern_type.loc135_27 => constants.%pattern_type.b06
// CHECK:STDOUT:   %pattern_type.loc135_33 => constants.%pattern_type.0e3
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc135_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc135_27 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc135_33 => constants.%complete_type.2da
// CHECK:STDOUT:   %require_complete.loc137 => constants.%complete_type.79d
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.708
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.loc137 => constants.%custom_witness.1e0
// CHECK:STDOUT:   %SubWith.facet => constants.%SubWith.facet.e26
// CHECK:STDOUT:   %SubWith.WithSelf.Op.type => constants.%SubWith.WithSelf.Op.type.098
// CHECK:STDOUT:   %.loc137_5.4 => constants.%.ee1
// CHECK:STDOUT:   %impl.elem1.loc137_5.2 => constants.%Int16.Op.7ce432.4
// CHECK:STDOUT:   %specific_impl_fn.loc137_5.3 => constants.%Int16.Op.7ce432.4
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.9f9
// CHECK:STDOUT:   %.loc137_5.5 => constants.%.16b
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.ecd0a5.1
// CHECK:STDOUT:   %impl.elem0.loc137_5.2 => constants.%Int16.Op.7ce432.1
// CHECK:STDOUT:   %specific_impl_fn.loc137_5.4 => constants.%Int16.Op.7ce432.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @MulWith.loc146(constants.%Destroy.facet.551, constants.%Destroy.facet.0e0, constants.%facet_value.60b) {
// CHECK:STDOUT:   %Result.loc143_9.1 => constants.%Destroy.facet.551
// CHECK:STDOUT:   %U.loc144_4.1 => constants.%Destroy.facet.0e0
// CHECK:STDOUT:   %U.as_type.loc145_36.1 => constants.%i16
// CHECK:STDOUT:   %MulWith.type.loc145_36.1 => constants.%MulWith.type.df1
// CHECK:STDOUT:   %facet_type.loc145_20 => constants.%facet_type.fd8
// CHECK:STDOUT:   %.Self.loc145_38.1 => constants.%.Self.615
// CHECK:STDOUT:   %require_complete.loc145 => constants.%complete_type.050
// CHECK:STDOUT:   %.Self.as_type.loc145_44.1 => constants.%.Self.as_type.434
// CHECK:STDOUT:   %MulWith.assoc_type => constants.%MulWith.assoc_type.693
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.037
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.loc145 => constants.%MulWith.lookup_impl_witness.b02
// CHECK:STDOUT:   %impl.elem0.loc145_44.1 => constants.%impl.elem0.d68
// CHECK:STDOUT:   %Result.as_type.loc145_54.1 => constants.%Int16
// CHECK:STDOUT:   %facet_type.loc145_38 => constants.%facet_type.0ff
// CHECK:STDOUT:   %T.loc145_4.1 => constants.%facet_value.60b
// CHECK:STDOUT:   %pattern_type.loc145 => constants.%pattern_type.370
// CHECK:STDOUT:   %pattern_type.loc146_16 => constants.%pattern_type.b06
// CHECK:STDOUT:   %T.as_type.loc146_29.1 => constants.%Int16
// CHECK:STDOUT:   %pattern_type.loc146_27 => constants.%pattern_type.b06
// CHECK:STDOUT:   %pattern_type.loc146_33 => constants.%pattern_type.0e3
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc146_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc146_27 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc146_33 => constants.%complete_type.2da
// CHECK:STDOUT:   %require_complete.loc148 => constants.%complete_type.4fe
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.7fb
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.loc148 => constants.%custom_witness.631
// CHECK:STDOUT:   %MulWith.facet => constants.%MulWith.facet.df1
// CHECK:STDOUT:   %MulWith.WithSelf.Op.type => constants.%MulWith.WithSelf.Op.type.5a5
// CHECK:STDOUT:   %.loc148_5.4 => constants.%.0a1
// CHECK:STDOUT:   %impl.elem1.loc148_5.2 => constants.%Int16.Op.7ce432.5
// CHECK:STDOUT:   %specific_impl_fn.loc148_5.3 => constants.%Int16.Op.7ce432.5
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.9f9
// CHECK:STDOUT:   %.loc148_5.5 => constants.%.16b
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.ecd0a5.1
// CHECK:STDOUT:   %impl.elem0.loc148_5.2 => constants.%Int16.Op.7ce432.1
// CHECK:STDOUT:   %specific_impl_fn.loc148_5.4 => constants.%Int16.Op.7ce432.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @DivWith.loc157(constants.%Destroy.facet.551, constants.%Destroy.facet.0e0, constants.%facet_value.4bd) {
// CHECK:STDOUT:   %Result.loc154_9.1 => constants.%Destroy.facet.551
// CHECK:STDOUT:   %U.loc155_4.1 => constants.%Destroy.facet.0e0
// CHECK:STDOUT:   %U.as_type.loc156_36.1 => constants.%i16
// CHECK:STDOUT:   %DivWith.type.loc156_36.1 => constants.%DivWith.type.957
// CHECK:STDOUT:   %facet_type.loc156_20 => constants.%facet_type.e1e
// CHECK:STDOUT:   %.Self.loc156_38.1 => constants.%.Self.319
// CHECK:STDOUT:   %require_complete.loc156 => constants.%complete_type.509
// CHECK:STDOUT:   %.Self.as_type.loc156_44.1 => constants.%.Self.as_type.619
// CHECK:STDOUT:   %DivWith.assoc_type => constants.%DivWith.assoc_type.c56
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.e06
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.loc156 => constants.%DivWith.lookup_impl_witness.493
// CHECK:STDOUT:   %impl.elem0.loc156_44.1 => constants.%impl.elem0.ed9
// CHECK:STDOUT:   %Result.as_type.loc156_54.1 => constants.%Int16
// CHECK:STDOUT:   %facet_type.loc156_38 => constants.%facet_type.721
// CHECK:STDOUT:   %T.loc156_4.1 => constants.%facet_value.4bd
// CHECK:STDOUT:   %pattern_type.loc156 => constants.%pattern_type.2b6
// CHECK:STDOUT:   %pattern_type.loc157_16 => constants.%pattern_type.b06
// CHECK:STDOUT:   %T.as_type.loc157_29.1 => constants.%Int16
// CHECK:STDOUT:   %pattern_type.loc157_27 => constants.%pattern_type.b06
// CHECK:STDOUT:   %pattern_type.loc157_33 => constants.%pattern_type.0e3
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc157_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc157_27 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc157_33 => constants.%complete_type.2da
// CHECK:STDOUT:   %require_complete.loc159 => constants.%complete_type.377
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.59d
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.loc159 => constants.%custom_witness.64e
// CHECK:STDOUT:   %DivWith.facet => constants.%DivWith.facet.243
// CHECK:STDOUT:   %DivWith.WithSelf.Op.type => constants.%DivWith.WithSelf.Op.type.644
// CHECK:STDOUT:   %.loc159_5.4 => constants.%.66b
// CHECK:STDOUT:   %impl.elem1.loc159_5.2 => constants.%Int16.Op.7ce432.6
// CHECK:STDOUT:   %specific_impl_fn.loc159_5.3 => constants.%Int16.Op.7ce432.6
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.9f9
// CHECK:STDOUT:   %.loc159_5.5 => constants.%.16b
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.ecd0a5.1
// CHECK:STDOUT:   %impl.elem0.loc159_5.2 => constants.%Int16.Op.7ce432.1
// CHECK:STDOUT:   %specific_impl_fn.loc159_5.4 => constants.%Int16.Op.7ce432.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ModWith.loc168(constants.%Destroy.facet.551, constants.%Destroy.facet.0e0, constants.%facet_value.dd2) {
// CHECK:STDOUT:   %Result.loc165_9.1 => constants.%Destroy.facet.551
// CHECK:STDOUT:   %U.loc166_4.1 => constants.%Destroy.facet.0e0
// CHECK:STDOUT:   %U.as_type.loc167_36.1 => constants.%i16
// CHECK:STDOUT:   %ModWith.type.loc167_36.1 => constants.%ModWith.type.0ea
// CHECK:STDOUT:   %facet_type.loc167_20 => constants.%facet_type.dca
// CHECK:STDOUT:   %.Self.loc167_38.1 => constants.%.Self.041
// CHECK:STDOUT:   %require_complete.loc167 => constants.%complete_type.65a
// CHECK:STDOUT:   %.Self.as_type.loc167_44.1 => constants.%.Self.as_type.535
// CHECK:STDOUT:   %ModWith.assoc_type => constants.%ModWith.assoc_type.327
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.2be
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.loc167 => constants.%ModWith.lookup_impl_witness.dfa
// CHECK:STDOUT:   %impl.elem0.loc167_44.1 => constants.%impl.elem0.025
// CHECK:STDOUT:   %Result.as_type.loc167_54.1 => constants.%Int16
// CHECK:STDOUT:   %facet_type.loc167_38 => constants.%facet_type.6dc
// CHECK:STDOUT:   %T.loc167_4.1 => constants.%facet_value.dd2
// CHECK:STDOUT:   %pattern_type.loc167 => constants.%pattern_type.49d
// CHECK:STDOUT:   %pattern_type.loc168_16 => constants.%pattern_type.b06
// CHECK:STDOUT:   %T.as_type.loc168_29.1 => constants.%Int16
// CHECK:STDOUT:   %pattern_type.loc168_27 => constants.%pattern_type.b06
// CHECK:STDOUT:   %pattern_type.loc168_33 => constants.%pattern_type.0e3
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc168_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc168_27 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc168_33 => constants.%complete_type.2da
// CHECK:STDOUT:   %require_complete.loc170 => constants.%complete_type.662
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.423
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.loc170 => constants.%custom_witness.394
// CHECK:STDOUT:   %ModWith.facet => constants.%ModWith.facet.37f
// CHECK:STDOUT:   %ModWith.WithSelf.Op.type => constants.%ModWith.WithSelf.Op.type.2f8
// CHECK:STDOUT:   %.loc170_5.4 => constants.%.1bc
// CHECK:STDOUT:   %impl.elem1.loc170_5.2 => constants.%Int16.Op.7ce432.7
// CHECK:STDOUT:   %specific_impl_fn.loc170_5.3 => constants.%Int16.Op.7ce432.7
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.9f9
// CHECK:STDOUT:   %.loc170_5.5 => constants.%.16b
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.ecd0a5.1
// CHECK:STDOUT:   %impl.elem0.loc170_5.2 => constants.%Int16.Op.7ce432.1
// CHECK:STDOUT:   %specific_impl_fn.loc170_5.4 => constants.%Int16.Op.7ce432.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AddWith.loc124(constants.%Destroy.facet.184, constants.%Destroy.facet.184, constants.%facet_value.a60) {
// CHECK:STDOUT:   %Result.loc121_9.1 => constants.%Destroy.facet.184
// CHECK:STDOUT:   %U.loc122_4.1 => constants.%Destroy.facet.184
// CHECK:STDOUT:   %U.as_type.loc123_36.1 => constants.%Int32
// CHECK:STDOUT:   %AddWith.type.loc123_36.1 => constants.%AddWith.type.4f1
// CHECK:STDOUT:   %facet_type.loc123_20 => constants.%facet_type.35e
// CHECK:STDOUT:   %.Self.loc123_38.1 => constants.%.Self.ea4
// CHECK:STDOUT:   %require_complete.loc123 => constants.%complete_type.bc1
// CHECK:STDOUT:   %.Self.as_type.loc123_44.1 => constants.%.Self.as_type.1e3
// CHECK:STDOUT:   %AddWith.assoc_type => constants.%AddWith.assoc_type.5fc
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.836
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.loc123 => constants.%AddWith.lookup_impl_witness.587
// CHECK:STDOUT:   %impl.elem0.loc123_44.1 => constants.%impl.elem0.4f1
// CHECK:STDOUT:   %Result.as_type.loc123_54.1 => constants.%Int32
// CHECK:STDOUT:   %facet_type.loc123_38 => constants.%facet_type.809
// CHECK:STDOUT:   %T.loc123_4.1 => constants.%facet_value.a60
// CHECK:STDOUT:   %pattern_type.loc123 => constants.%pattern_type.a7b
// CHECK:STDOUT:   %pattern_type.loc124_16 => constants.%pattern_type.4b8
// CHECK:STDOUT:   %T.as_type.loc124_29.1 => constants.%Int16
// CHECK:STDOUT:   %pattern_type.loc124_27 => constants.%pattern_type.b06
// CHECK:STDOUT:   %pattern_type.loc124_33 => constants.%pattern_type.4b8
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc124_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc124_27 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc124_33 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc126 => constants.%complete_type.e69
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.96b
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.loc126 => constants.%custom_witness.bb0e2d.1
// CHECK:STDOUT:   %AddWith.facet => constants.%AddWith.facet.cea
// CHECK:STDOUT:   %AddWith.WithSelf.Op.type => constants.%AddWith.WithSelf.Op.type.f93
// CHECK:STDOUT:   %.loc126_5.4 => constants.%.886
// CHECK:STDOUT:   %impl.elem1.loc126_5.2 => constants.%Int16.Op.7ce432.8
// CHECK:STDOUT:   %specific_impl_fn.loc126_5.3 => constants.%Int16.Op.7ce432.8
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.257
// CHECK:STDOUT:   %.loc126_5.5 => constants.%.d3f
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.ecd0a5.2
// CHECK:STDOUT:   %impl.elem0.loc126_5.2 => constants.%Int32.Op.7ce432.1
// CHECK:STDOUT:   %specific_impl_fn.loc126_5.4 => constants.%Int32.Op.7ce432.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @SubWith.loc135(constants.%Destroy.facet.184, constants.%Destroy.facet.184, constants.%facet_value.88f) {
// CHECK:STDOUT:   %Result.loc132_9.1 => constants.%Destroy.facet.184
// CHECK:STDOUT:   %U.loc133_4.1 => constants.%Destroy.facet.184
// CHECK:STDOUT:   %U.as_type.loc134_36.1 => constants.%Int32
// CHECK:STDOUT:   %SubWith.type.loc134_36.1 => constants.%SubWith.type.40d
// CHECK:STDOUT:   %facet_type.loc134_20 => constants.%facet_type.cbc
// CHECK:STDOUT:   %.Self.loc134_38.1 => constants.%.Self.ae4
// CHECK:STDOUT:   %require_complete.loc134 => constants.%complete_type.44f
// CHECK:STDOUT:   %.Self.as_type.loc134_44.1 => constants.%.Self.as_type.484
// CHECK:STDOUT:   %SubWith.assoc_type => constants.%SubWith.assoc_type.73f
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.c3d
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.loc134 => constants.%SubWith.lookup_impl_witness.d98
// CHECK:STDOUT:   %impl.elem0.loc134_44.1 => constants.%impl.elem0.f09
// CHECK:STDOUT:   %Result.as_type.loc134_54.1 => constants.%Int32
// CHECK:STDOUT:   %facet_type.loc134_38 => constants.%facet_type.445
// CHECK:STDOUT:   %T.loc134_4.1 => constants.%facet_value.88f
// CHECK:STDOUT:   %pattern_type.loc134 => constants.%pattern_type.766
// CHECK:STDOUT:   %pattern_type.loc135_16 => constants.%pattern_type.4b8
// CHECK:STDOUT:   %T.as_type.loc135_29.1 => constants.%Int16
// CHECK:STDOUT:   %pattern_type.loc135_27 => constants.%pattern_type.b06
// CHECK:STDOUT:   %pattern_type.loc135_33 => constants.%pattern_type.4b8
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc135_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc135_27 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc135_33 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc137 => constants.%complete_type.c40
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.827
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.loc137 => constants.%custom_witness.dee291.1
// CHECK:STDOUT:   %SubWith.facet => constants.%SubWith.facet.152
// CHECK:STDOUT:   %SubWith.WithSelf.Op.type => constants.%SubWith.WithSelf.Op.type.15f
// CHECK:STDOUT:   %.loc137_5.4 => constants.%.18a
// CHECK:STDOUT:   %impl.elem1.loc137_5.2 => constants.%Int16.Op.7ce432.9
// CHECK:STDOUT:   %specific_impl_fn.loc137_5.3 => constants.%Int16.Op.7ce432.9
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.257
// CHECK:STDOUT:   %.loc137_5.5 => constants.%.d3f
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.ecd0a5.2
// CHECK:STDOUT:   %impl.elem0.loc137_5.2 => constants.%Int32.Op.7ce432.1
// CHECK:STDOUT:   %specific_impl_fn.loc137_5.4 => constants.%Int32.Op.7ce432.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @MulWith.loc146(constants.%Destroy.facet.184, constants.%Destroy.facet.184, constants.%facet_value.668) {
// CHECK:STDOUT:   %Result.loc143_9.1 => constants.%Destroy.facet.184
// CHECK:STDOUT:   %U.loc144_4.1 => constants.%Destroy.facet.184
// CHECK:STDOUT:   %U.as_type.loc145_36.1 => constants.%Int32
// CHECK:STDOUT:   %MulWith.type.loc145_36.1 => constants.%MulWith.type.8d1
// CHECK:STDOUT:   %facet_type.loc145_20 => constants.%facet_type.655
// CHECK:STDOUT:   %.Self.loc145_38.1 => constants.%.Self.e02
// CHECK:STDOUT:   %require_complete.loc145 => constants.%complete_type.aab
// CHECK:STDOUT:   %.Self.as_type.loc145_44.1 => constants.%.Self.as_type.efe
// CHECK:STDOUT:   %MulWith.assoc_type => constants.%MulWith.assoc_type.d17
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.225
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.loc145 => constants.%MulWith.lookup_impl_witness.5f7
// CHECK:STDOUT:   %impl.elem0.loc145_44.1 => constants.%impl.elem0.0bb
// CHECK:STDOUT:   %Result.as_type.loc145_54.1 => constants.%Int32
// CHECK:STDOUT:   %facet_type.loc145_38 => constants.%facet_type.8a8
// CHECK:STDOUT:   %T.loc145_4.1 => constants.%facet_value.668
// CHECK:STDOUT:   %pattern_type.loc145 => constants.%pattern_type.bf1
// CHECK:STDOUT:   %pattern_type.loc146_16 => constants.%pattern_type.4b8
// CHECK:STDOUT:   %T.as_type.loc146_29.1 => constants.%Int16
// CHECK:STDOUT:   %pattern_type.loc146_27 => constants.%pattern_type.b06
// CHECK:STDOUT:   %pattern_type.loc146_33 => constants.%pattern_type.4b8
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc146_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc146_27 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc146_33 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc148 => constants.%complete_type.ab9
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.a92
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.loc148 => constants.%custom_witness.db8ceb.1
// CHECK:STDOUT:   %MulWith.facet => constants.%MulWith.facet.e4c
// CHECK:STDOUT:   %MulWith.WithSelf.Op.type => constants.%MulWith.WithSelf.Op.type.ac5
// CHECK:STDOUT:   %.loc148_5.4 => constants.%.d07
// CHECK:STDOUT:   %impl.elem1.loc148_5.2 => constants.%Int16.Op.7ce432.10
// CHECK:STDOUT:   %specific_impl_fn.loc148_5.3 => constants.%Int16.Op.7ce432.10
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.257
// CHECK:STDOUT:   %.loc148_5.5 => constants.%.d3f
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.ecd0a5.2
// CHECK:STDOUT:   %impl.elem0.loc148_5.2 => constants.%Int32.Op.7ce432.1
// CHECK:STDOUT:   %specific_impl_fn.loc148_5.4 => constants.%Int32.Op.7ce432.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @DivWith.loc157(constants.%Destroy.facet.184, constants.%Destroy.facet.184, constants.%facet_value.c22) {
// CHECK:STDOUT:   %Result.loc154_9.1 => constants.%Destroy.facet.184
// CHECK:STDOUT:   %U.loc155_4.1 => constants.%Destroy.facet.184
// CHECK:STDOUT:   %U.as_type.loc156_36.1 => constants.%Int32
// CHECK:STDOUT:   %DivWith.type.loc156_36.1 => constants.%DivWith.type.a2b
// CHECK:STDOUT:   %facet_type.loc156_20 => constants.%facet_type.9ed
// CHECK:STDOUT:   %.Self.loc156_38.1 => constants.%.Self.536
// CHECK:STDOUT:   %require_complete.loc156 => constants.%complete_type.bfd
// CHECK:STDOUT:   %.Self.as_type.loc156_44.1 => constants.%.Self.as_type.a54
// CHECK:STDOUT:   %DivWith.assoc_type => constants.%DivWith.assoc_type.185
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.2d7
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.loc156 => constants.%DivWith.lookup_impl_witness.3ec
// CHECK:STDOUT:   %impl.elem0.loc156_44.1 => constants.%impl.elem0.1dc3
// CHECK:STDOUT:   %Result.as_type.loc156_54.1 => constants.%Int32
// CHECK:STDOUT:   %facet_type.loc156_38 => constants.%facet_type.a43
// CHECK:STDOUT:   %T.loc156_4.1 => constants.%facet_value.c22
// CHECK:STDOUT:   %pattern_type.loc156 => constants.%pattern_type.3fb
// CHECK:STDOUT:   %pattern_type.loc157_16 => constants.%pattern_type.4b8
// CHECK:STDOUT:   %T.as_type.loc157_29.1 => constants.%Int16
// CHECK:STDOUT:   %pattern_type.loc157_27 => constants.%pattern_type.b06
// CHECK:STDOUT:   %pattern_type.loc157_33 => constants.%pattern_type.4b8
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc157_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc157_27 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc157_33 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc159 => constants.%complete_type.0f3
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.9b2
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.loc159 => constants.%custom_witness.deca3a.1
// CHECK:STDOUT:   %DivWith.facet => constants.%DivWith.facet.37d
// CHECK:STDOUT:   %DivWith.WithSelf.Op.type => constants.%DivWith.WithSelf.Op.type.811
// CHECK:STDOUT:   %.loc159_5.4 => constants.%.d58
// CHECK:STDOUT:   %impl.elem1.loc159_5.2 => constants.%Int16.Op.7ce432.11
// CHECK:STDOUT:   %specific_impl_fn.loc159_5.3 => constants.%Int16.Op.7ce432.11
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.257
// CHECK:STDOUT:   %.loc159_5.5 => constants.%.d3f
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.ecd0a5.2
// CHECK:STDOUT:   %impl.elem0.loc159_5.2 => constants.%Int32.Op.7ce432.1
// CHECK:STDOUT:   %specific_impl_fn.loc159_5.4 => constants.%Int32.Op.7ce432.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ModWith.loc168(constants.%Destroy.facet.184, constants.%Destroy.facet.184, constants.%facet_value.194) {
// CHECK:STDOUT:   %Result.loc165_9.1 => constants.%Destroy.facet.184
// CHECK:STDOUT:   %U.loc166_4.1 => constants.%Destroy.facet.184
// CHECK:STDOUT:   %U.as_type.loc167_36.1 => constants.%Int32
// CHECK:STDOUT:   %ModWith.type.loc167_36.1 => constants.%ModWith.type.cff
// CHECK:STDOUT:   %facet_type.loc167_20 => constants.%facet_type.cb7
// CHECK:STDOUT:   %.Self.loc167_38.1 => constants.%.Self.6c7
// CHECK:STDOUT:   %require_complete.loc167 => constants.%complete_type.e06
// CHECK:STDOUT:   %.Self.as_type.loc167_44.1 => constants.%.Self.as_type.2f0
// CHECK:STDOUT:   %ModWith.assoc_type => constants.%ModWith.assoc_type.557
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.beb
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.loc167 => constants.%ModWith.lookup_impl_witness.f61
// CHECK:STDOUT:   %impl.elem0.loc167_44.1 => constants.%impl.elem0.e0b
// CHECK:STDOUT:   %Result.as_type.loc167_54.1 => constants.%Int32
// CHECK:STDOUT:   %facet_type.loc167_38 => constants.%facet_type.e5f
// CHECK:STDOUT:   %T.loc167_4.1 => constants.%facet_value.194
// CHECK:STDOUT:   %pattern_type.loc167 => constants.%pattern_type.011
// CHECK:STDOUT:   %pattern_type.loc168_16 => constants.%pattern_type.4b8
// CHECK:STDOUT:   %T.as_type.loc168_29.1 => constants.%Int16
// CHECK:STDOUT:   %pattern_type.loc168_27 => constants.%pattern_type.b06
// CHECK:STDOUT:   %pattern_type.loc168_33 => constants.%pattern_type.4b8
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc168_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc168_27 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc168_33 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc170 => constants.%complete_type.fae
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.b6a
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.loc170 => constants.%custom_witness.c72db9.1
// CHECK:STDOUT:   %ModWith.facet => constants.%ModWith.facet.ac1
// CHECK:STDOUT:   %ModWith.WithSelf.Op.type => constants.%ModWith.WithSelf.Op.type.f9f5
// CHECK:STDOUT:   %.loc170_5.4 => constants.%.37d
// CHECK:STDOUT:   %impl.elem1.loc170_5.2 => constants.%Int16.Op.7ce432.12
// CHECK:STDOUT:   %specific_impl_fn.loc170_5.3 => constants.%Int16.Op.7ce432.12
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.257
// CHECK:STDOUT:   %.loc170_5.5 => constants.%.d3f
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.ecd0a5.2
// CHECK:STDOUT:   %impl.elem0.loc170_5.2 => constants.%Int32.Op.7ce432.1
// CHECK:STDOUT:   %specific_impl_fn.loc170_5.4 => constants.%Int32.Op.7ce432.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AddWith.loc124(constants.%Destroy.facet.184, constants.%Destroy.facet.551, constants.%facet_value.5cb) {
// CHECK:STDOUT:   %Result.loc121_9.1 => constants.%Destroy.facet.184
// CHECK:STDOUT:   %U.loc122_4.1 => constants.%Destroy.facet.551
// CHECK:STDOUT:   %U.as_type.loc123_36.1 => constants.%Int16
// CHECK:STDOUT:   %AddWith.type.loc123_36.1 => constants.%AddWith.type.41f
// CHECK:STDOUT:   %facet_type.loc123_20 => constants.%facet_type.b25
// CHECK:STDOUT:   %.Self.loc123_38.1 => constants.%.Self.cee
// CHECK:STDOUT:   %require_complete.loc123 => constants.%complete_type.08f
// CHECK:STDOUT:   %.Self.as_type.loc123_44.1 => constants.%.Self.as_type.38a
// CHECK:STDOUT:   %AddWith.assoc_type => constants.%AddWith.assoc_type.5ee
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.95b
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.loc123 => constants.%AddWith.lookup_impl_witness.ef3
// CHECK:STDOUT:   %impl.elem0.loc123_44.1 => constants.%impl.elem0.1ef
// CHECK:STDOUT:   %Result.as_type.loc123_54.1 => constants.%Int32
// CHECK:STDOUT:   %facet_type.loc123_38 => constants.%facet_type.e93
// CHECK:STDOUT:   %T.loc123_4.1 => constants.%facet_value.5cb
// CHECK:STDOUT:   %pattern_type.loc123 => constants.%pattern_type.05d
// CHECK:STDOUT:   %pattern_type.loc124_16 => constants.%pattern_type.4b8
// CHECK:STDOUT:   %T.as_type.loc124_29.1 => constants.%Int32
// CHECK:STDOUT:   %pattern_type.loc124_27 => constants.%pattern_type.4b8
// CHECK:STDOUT:   %pattern_type.loc124_33 => constants.%pattern_type.b06
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc124_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc124_27 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc124_33 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc126 => constants.%complete_type.f1a
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.41e
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.loc126 => constants.%custom_witness.8d5
// CHECK:STDOUT:   %AddWith.facet => constants.%AddWith.facet.82f
// CHECK:STDOUT:   %AddWith.WithSelf.Op.type => constants.%AddWith.WithSelf.Op.type.28d
// CHECK:STDOUT:   %.loc126_5.4 => constants.%.e93
// CHECK:STDOUT:   %impl.elem1.loc126_5.2 => constants.%Int32.Op.7ce432.2
// CHECK:STDOUT:   %specific_impl_fn.loc126_5.3 => constants.%Int32.Op.7ce432.2
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.257
// CHECK:STDOUT:   %.loc126_5.5 => constants.%.d3f
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.ecd0a5.2
// CHECK:STDOUT:   %impl.elem0.loc126_5.2 => constants.%Int32.Op.7ce432.1
// CHECK:STDOUT:   %specific_impl_fn.loc126_5.4 => constants.%Int32.Op.7ce432.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @SubWith.loc135(constants.%Destroy.facet.184, constants.%Destroy.facet.551, constants.%facet_value.801) {
// CHECK:STDOUT:   %Result.loc132_9.1 => constants.%Destroy.facet.184
// CHECK:STDOUT:   %U.loc133_4.1 => constants.%Destroy.facet.551
// CHECK:STDOUT:   %U.as_type.loc134_36.1 => constants.%Int16
// CHECK:STDOUT:   %SubWith.type.loc134_36.1 => constants.%SubWith.type.f7a
// CHECK:STDOUT:   %facet_type.loc134_20 => constants.%facet_type.5bd
// CHECK:STDOUT:   %.Self.loc134_38.1 => constants.%.Self.0bf
// CHECK:STDOUT:   %require_complete.loc134 => constants.%complete_type.4b4
// CHECK:STDOUT:   %.Self.as_type.loc134_44.1 => constants.%.Self.as_type.752
// CHECK:STDOUT:   %SubWith.assoc_type => constants.%SubWith.assoc_type.e4b
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.f15
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.loc134 => constants.%SubWith.lookup_impl_witness.9d6
// CHECK:STDOUT:   %impl.elem0.loc134_44.1 => constants.%impl.elem0.214
// CHECK:STDOUT:   %Result.as_type.loc134_54.1 => constants.%Int32
// CHECK:STDOUT:   %facet_type.loc134_38 => constants.%facet_type.7be
// CHECK:STDOUT:   %T.loc134_4.1 => constants.%facet_value.801
// CHECK:STDOUT:   %pattern_type.loc134 => constants.%pattern_type.452
// CHECK:STDOUT:   %pattern_type.loc135_16 => constants.%pattern_type.4b8
// CHECK:STDOUT:   %T.as_type.loc135_29.1 => constants.%Int32
// CHECK:STDOUT:   %pattern_type.loc135_27 => constants.%pattern_type.4b8
// CHECK:STDOUT:   %pattern_type.loc135_33 => constants.%pattern_type.b06
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc135_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc135_27 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc135_33 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc137 => constants.%complete_type.c17
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.8a0
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.loc137 => constants.%custom_witness.14c
// CHECK:STDOUT:   %SubWith.facet => constants.%SubWith.facet.dfe
// CHECK:STDOUT:   %SubWith.WithSelf.Op.type => constants.%SubWith.WithSelf.Op.type.685
// CHECK:STDOUT:   %.loc137_5.4 => constants.%.d09
// CHECK:STDOUT:   %impl.elem1.loc137_5.2 => constants.%Int32.Op.7ce432.3
// CHECK:STDOUT:   %specific_impl_fn.loc137_5.3 => constants.%Int32.Op.7ce432.3
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.257
// CHECK:STDOUT:   %.loc137_5.5 => constants.%.d3f
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.ecd0a5.2
// CHECK:STDOUT:   %impl.elem0.loc137_5.2 => constants.%Int32.Op.7ce432.1
// CHECK:STDOUT:   %specific_impl_fn.loc137_5.4 => constants.%Int32.Op.7ce432.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @MulWith.loc146(constants.%Destroy.facet.184, constants.%Destroy.facet.551, constants.%facet_value.ac1) {
// CHECK:STDOUT:   %Result.loc143_9.1 => constants.%Destroy.facet.184
// CHECK:STDOUT:   %U.loc144_4.1 => constants.%Destroy.facet.551
// CHECK:STDOUT:   %U.as_type.loc145_36.1 => constants.%Int16
// CHECK:STDOUT:   %MulWith.type.loc145_36.1 => constants.%MulWith.type.ebe
// CHECK:STDOUT:   %facet_type.loc145_20 => constants.%facet_type.b8b
// CHECK:STDOUT:   %.Self.loc145_38.1 => constants.%.Self.117
// CHECK:STDOUT:   %require_complete.loc145 => constants.%complete_type.25f
// CHECK:STDOUT:   %.Self.as_type.loc145_44.1 => constants.%.Self.as_type.ad8
// CHECK:STDOUT:   %MulWith.assoc_type => constants.%MulWith.assoc_type.286
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.52c
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.loc145 => constants.%MulWith.lookup_impl_witness.485
// CHECK:STDOUT:   %impl.elem0.loc145_44.1 => constants.%impl.elem0.851
// CHECK:STDOUT:   %Result.as_type.loc145_54.1 => constants.%Int32
// CHECK:STDOUT:   %facet_type.loc145_38 => constants.%facet_type.460
// CHECK:STDOUT:   %T.loc145_4.1 => constants.%facet_value.ac1
// CHECK:STDOUT:   %pattern_type.loc145 => constants.%pattern_type.cfe
// CHECK:STDOUT:   %pattern_type.loc146_16 => constants.%pattern_type.4b8
// CHECK:STDOUT:   %T.as_type.loc146_29.1 => constants.%Int32
// CHECK:STDOUT:   %pattern_type.loc146_27 => constants.%pattern_type.4b8
// CHECK:STDOUT:   %pattern_type.loc146_33 => constants.%pattern_type.b06
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc146_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc146_27 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc146_33 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc148 => constants.%complete_type.b33
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.0bf
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.loc148 => constants.%custom_witness.fd0
// CHECK:STDOUT:   %MulWith.facet => constants.%MulWith.facet.7de
// CHECK:STDOUT:   %MulWith.WithSelf.Op.type => constants.%MulWith.WithSelf.Op.type.5bf
// CHECK:STDOUT:   %.loc148_5.4 => constants.%.0d2
// CHECK:STDOUT:   %impl.elem1.loc148_5.2 => constants.%Int32.Op.7ce432.4
// CHECK:STDOUT:   %specific_impl_fn.loc148_5.3 => constants.%Int32.Op.7ce432.4
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.257
// CHECK:STDOUT:   %.loc148_5.5 => constants.%.d3f
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.ecd0a5.2
// CHECK:STDOUT:   %impl.elem0.loc148_5.2 => constants.%Int32.Op.7ce432.1
// CHECK:STDOUT:   %specific_impl_fn.loc148_5.4 => constants.%Int32.Op.7ce432.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @DivWith.loc157(constants.%Destroy.facet.184, constants.%Destroy.facet.551, constants.%facet_value.c58) {
// CHECK:STDOUT:   %Result.loc154_9.1 => constants.%Destroy.facet.184
// CHECK:STDOUT:   %U.loc155_4.1 => constants.%Destroy.facet.551
// CHECK:STDOUT:   %U.as_type.loc156_36.1 => constants.%Int16
// CHECK:STDOUT:   %DivWith.type.loc156_36.1 => constants.%DivWith.type.996
// CHECK:STDOUT:   %facet_type.loc156_20 => constants.%facet_type.078
// CHECK:STDOUT:   %.Self.loc156_38.1 => constants.%.Self.f7a
// CHECK:STDOUT:   %require_complete.loc156 => constants.%complete_type.4f9
// CHECK:STDOUT:   %.Self.as_type.loc156_44.1 => constants.%.Self.as_type.91d
// CHECK:STDOUT:   %DivWith.assoc_type => constants.%DivWith.assoc_type.ae8
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.7b6
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.loc156 => constants.%DivWith.lookup_impl_witness.922
// CHECK:STDOUT:   %impl.elem0.loc156_44.1 => constants.%impl.elem0.1ae
// CHECK:STDOUT:   %Result.as_type.loc156_54.1 => constants.%Int32
// CHECK:STDOUT:   %facet_type.loc156_38 => constants.%facet_type.072
// CHECK:STDOUT:   %T.loc156_4.1 => constants.%facet_value.c58
// CHECK:STDOUT:   %pattern_type.loc156 => constants.%pattern_type.7e3
// CHECK:STDOUT:   %pattern_type.loc157_16 => constants.%pattern_type.4b8
// CHECK:STDOUT:   %T.as_type.loc157_29.1 => constants.%Int32
// CHECK:STDOUT:   %pattern_type.loc157_27 => constants.%pattern_type.4b8
// CHECK:STDOUT:   %pattern_type.loc157_33 => constants.%pattern_type.b06
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc157_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc157_27 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc157_33 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc159 => constants.%complete_type.691
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.f9f
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.loc159 => constants.%custom_witness.3a3
// CHECK:STDOUT:   %DivWith.facet => constants.%DivWith.facet.021
// CHECK:STDOUT:   %DivWith.WithSelf.Op.type => constants.%DivWith.WithSelf.Op.type.8bc
// CHECK:STDOUT:   %.loc159_5.4 => constants.%.715
// CHECK:STDOUT:   %impl.elem1.loc159_5.2 => constants.%Int32.Op.7ce432.5
// CHECK:STDOUT:   %specific_impl_fn.loc159_5.3 => constants.%Int32.Op.7ce432.5
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.257
// CHECK:STDOUT:   %.loc159_5.5 => constants.%.d3f
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.ecd0a5.2
// CHECK:STDOUT:   %impl.elem0.loc159_5.2 => constants.%Int32.Op.7ce432.1
// CHECK:STDOUT:   %specific_impl_fn.loc159_5.4 => constants.%Int32.Op.7ce432.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ModWith.loc168(constants.%Destroy.facet.184, constants.%Destroy.facet.551, constants.%facet_value.f21) {
// CHECK:STDOUT:   %Result.loc165_9.1 => constants.%Destroy.facet.184
// CHECK:STDOUT:   %U.loc166_4.1 => constants.%Destroy.facet.551
// CHECK:STDOUT:   %U.as_type.loc167_36.1 => constants.%Int16
// CHECK:STDOUT:   %ModWith.type.loc167_36.1 => constants.%ModWith.type.d4b
// CHECK:STDOUT:   %facet_type.loc167_20 => constants.%facet_type.dc0
// CHECK:STDOUT:   %.Self.loc167_38.1 => constants.%.Self.c9e
// CHECK:STDOUT:   %require_complete.loc167 => constants.%complete_type.10f
// CHECK:STDOUT:   %.Self.as_type.loc167_44.1 => constants.%.Self.as_type.fc8
// CHECK:STDOUT:   %ModWith.assoc_type => constants.%ModWith.assoc_type.8c2
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.262
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.loc167 => constants.%ModWith.lookup_impl_witness.9ea
// CHECK:STDOUT:   %impl.elem0.loc167_44.1 => constants.%impl.elem0.7c3
// CHECK:STDOUT:   %Result.as_type.loc167_54.1 => constants.%Int32
// CHECK:STDOUT:   %facet_type.loc167_38 => constants.%facet_type.07e
// CHECK:STDOUT:   %T.loc167_4.1 => constants.%facet_value.f21
// CHECK:STDOUT:   %pattern_type.loc167 => constants.%pattern_type.83e
// CHECK:STDOUT:   %pattern_type.loc168_16 => constants.%pattern_type.4b8
// CHECK:STDOUT:   %T.as_type.loc168_29.1 => constants.%Int32
// CHECK:STDOUT:   %pattern_type.loc168_27 => constants.%pattern_type.4b8
// CHECK:STDOUT:   %pattern_type.loc168_33 => constants.%pattern_type.b06
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc168_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc168_27 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc168_33 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc170 => constants.%complete_type.3a6
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.63a
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.loc170 => constants.%custom_witness.e15
// CHECK:STDOUT:   %ModWith.facet => constants.%ModWith.facet.481
// CHECK:STDOUT:   %ModWith.WithSelf.Op.type => constants.%ModWith.WithSelf.Op.type.efe
// CHECK:STDOUT:   %.loc170_5.4 => constants.%.942
// CHECK:STDOUT:   %impl.elem1.loc170_5.2 => constants.%Int32.Op.7ce432.6
// CHECK:STDOUT:   %specific_impl_fn.loc170_5.3 => constants.%Int32.Op.7ce432.6
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.257
// CHECK:STDOUT:   %.loc170_5.5 => constants.%.d3f
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.ecd0a5.2
// CHECK:STDOUT:   %impl.elem0.loc170_5.2 => constants.%Int32.Op.7ce432.1
// CHECK:STDOUT:   %specific_impl_fn.loc170_5.4 => constants.%Int32.Op.7ce432.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AddWith.loc124(constants.%Destroy.facet.184, constants.%Destroy.facet.184, constants.%facet_value.3a9) {
// CHECK:STDOUT:   %Result.loc121_9.1 => constants.%Destroy.facet.184
// CHECK:STDOUT:   %U.loc122_4.1 => constants.%Destroy.facet.184
// CHECK:STDOUT:   %U.as_type.loc123_36.1 => constants.%Int32
// CHECK:STDOUT:   %AddWith.type.loc123_36.1 => constants.%AddWith.type.4f1
// CHECK:STDOUT:   %facet_type.loc123_20 => constants.%facet_type.35e
// CHECK:STDOUT:   %.Self.loc123_38.1 => constants.%.Self.ea4
// CHECK:STDOUT:   %require_complete.loc123 => constants.%complete_type.bc1
// CHECK:STDOUT:   %.Self.as_type.loc123_44.1 => constants.%.Self.as_type.1e3
// CHECK:STDOUT:   %AddWith.assoc_type => constants.%AddWith.assoc_type.5fc
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.836
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.loc123 => constants.%AddWith.lookup_impl_witness.587
// CHECK:STDOUT:   %impl.elem0.loc123_44.1 => constants.%impl.elem0.4f1
// CHECK:STDOUT:   %Result.as_type.loc123_54.1 => constants.%Int32
// CHECK:STDOUT:   %facet_type.loc123_38 => constants.%facet_type.809
// CHECK:STDOUT:   %T.loc123_4.1 => constants.%facet_value.3a9
// CHECK:STDOUT:   %pattern_type.loc123 => constants.%pattern_type.a7b
// CHECK:STDOUT:   %pattern_type.loc124_16 => constants.%pattern_type.4b8
// CHECK:STDOUT:   %T.as_type.loc124_29.1 => constants.%Int32
// CHECK:STDOUT:   %pattern_type.loc124_27 => constants.%pattern_type.4b8
// CHECK:STDOUT:   %pattern_type.loc124_33 => constants.%pattern_type.4b8
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc124_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc124_27 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc124_33 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc126 => constants.%complete_type.e69
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.96b
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.loc126 => constants.%custom_witness.bb0e2d.2
// CHECK:STDOUT:   %AddWith.facet => constants.%AddWith.facet.5a2
// CHECK:STDOUT:   %AddWith.WithSelf.Op.type => constants.%AddWith.WithSelf.Op.type.b30
// CHECK:STDOUT:   %.loc126_5.4 => constants.%.82a
// CHECK:STDOUT:   %impl.elem1.loc126_5.2 => constants.%Int32.Op.7ce432.7
// CHECK:STDOUT:   %specific_impl_fn.loc126_5.3 => constants.%Int32.Op.7ce432.7
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.257
// CHECK:STDOUT:   %.loc126_5.5 => constants.%.d3f
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.ecd0a5.2
// CHECK:STDOUT:   %impl.elem0.loc126_5.2 => constants.%Int32.Op.7ce432.1
// CHECK:STDOUT:   %specific_impl_fn.loc126_5.4 => constants.%Int32.Op.7ce432.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @SubWith.loc135(constants.%Destroy.facet.184, constants.%Destroy.facet.184, constants.%facet_value.785) {
// CHECK:STDOUT:   %Result.loc132_9.1 => constants.%Destroy.facet.184
// CHECK:STDOUT:   %U.loc133_4.1 => constants.%Destroy.facet.184
// CHECK:STDOUT:   %U.as_type.loc134_36.1 => constants.%Int32
// CHECK:STDOUT:   %SubWith.type.loc134_36.1 => constants.%SubWith.type.40d
// CHECK:STDOUT:   %facet_type.loc134_20 => constants.%facet_type.cbc
// CHECK:STDOUT:   %.Self.loc134_38.1 => constants.%.Self.ae4
// CHECK:STDOUT:   %require_complete.loc134 => constants.%complete_type.44f
// CHECK:STDOUT:   %.Self.as_type.loc134_44.1 => constants.%.Self.as_type.484
// CHECK:STDOUT:   %SubWith.assoc_type => constants.%SubWith.assoc_type.73f
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.c3d
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.loc134 => constants.%SubWith.lookup_impl_witness.d98
// CHECK:STDOUT:   %impl.elem0.loc134_44.1 => constants.%impl.elem0.f09
// CHECK:STDOUT:   %Result.as_type.loc134_54.1 => constants.%Int32
// CHECK:STDOUT:   %facet_type.loc134_38 => constants.%facet_type.445
// CHECK:STDOUT:   %T.loc134_4.1 => constants.%facet_value.785
// CHECK:STDOUT:   %pattern_type.loc134 => constants.%pattern_type.766
// CHECK:STDOUT:   %pattern_type.loc135_16 => constants.%pattern_type.4b8
// CHECK:STDOUT:   %T.as_type.loc135_29.1 => constants.%Int32
// CHECK:STDOUT:   %pattern_type.loc135_27 => constants.%pattern_type.4b8
// CHECK:STDOUT:   %pattern_type.loc135_33 => constants.%pattern_type.4b8
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc135_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc135_27 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc135_33 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc137 => constants.%complete_type.c40
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.827
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.loc137 => constants.%custom_witness.dee291.2
// CHECK:STDOUT:   %SubWith.facet => constants.%SubWith.facet.a57
// CHECK:STDOUT:   %SubWith.WithSelf.Op.type => constants.%SubWith.WithSelf.Op.type.cb0
// CHECK:STDOUT:   %.loc137_5.4 => constants.%.634
// CHECK:STDOUT:   %impl.elem1.loc137_5.2 => constants.%Int32.Op.7ce432.8
// CHECK:STDOUT:   %specific_impl_fn.loc137_5.3 => constants.%Int32.Op.7ce432.8
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.257
// CHECK:STDOUT:   %.loc137_5.5 => constants.%.d3f
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.ecd0a5.2
// CHECK:STDOUT:   %impl.elem0.loc137_5.2 => constants.%Int32.Op.7ce432.1
// CHECK:STDOUT:   %specific_impl_fn.loc137_5.4 => constants.%Int32.Op.7ce432.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @MulWith.loc146(constants.%Destroy.facet.184, constants.%Destroy.facet.184, constants.%facet_value.0dc) {
// CHECK:STDOUT:   %Result.loc143_9.1 => constants.%Destroy.facet.184
// CHECK:STDOUT:   %U.loc144_4.1 => constants.%Destroy.facet.184
// CHECK:STDOUT:   %U.as_type.loc145_36.1 => constants.%Int32
// CHECK:STDOUT:   %MulWith.type.loc145_36.1 => constants.%MulWith.type.8d1
// CHECK:STDOUT:   %facet_type.loc145_20 => constants.%facet_type.655
// CHECK:STDOUT:   %.Self.loc145_38.1 => constants.%.Self.e02
// CHECK:STDOUT:   %require_complete.loc145 => constants.%complete_type.aab
// CHECK:STDOUT:   %.Self.as_type.loc145_44.1 => constants.%.Self.as_type.efe
// CHECK:STDOUT:   %MulWith.assoc_type => constants.%MulWith.assoc_type.d17
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.225
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.loc145 => constants.%MulWith.lookup_impl_witness.5f7
// CHECK:STDOUT:   %impl.elem0.loc145_44.1 => constants.%impl.elem0.0bb
// CHECK:STDOUT:   %Result.as_type.loc145_54.1 => constants.%Int32
// CHECK:STDOUT:   %facet_type.loc145_38 => constants.%facet_type.8a8
// CHECK:STDOUT:   %T.loc145_4.1 => constants.%facet_value.0dc
// CHECK:STDOUT:   %pattern_type.loc145 => constants.%pattern_type.bf1
// CHECK:STDOUT:   %pattern_type.loc146_16 => constants.%pattern_type.4b8
// CHECK:STDOUT:   %T.as_type.loc146_29.1 => constants.%Int32
// CHECK:STDOUT:   %pattern_type.loc146_27 => constants.%pattern_type.4b8
// CHECK:STDOUT:   %pattern_type.loc146_33 => constants.%pattern_type.4b8
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc146_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc146_27 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc146_33 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc148 => constants.%complete_type.ab9
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.a92
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.loc148 => constants.%custom_witness.db8ceb.2
// CHECK:STDOUT:   %MulWith.facet => constants.%MulWith.facet.3b9
// CHECK:STDOUT:   %MulWith.WithSelf.Op.type => constants.%MulWith.WithSelf.Op.type.c67
// CHECK:STDOUT:   %.loc148_5.4 => constants.%.e9a
// CHECK:STDOUT:   %impl.elem1.loc148_5.2 => constants.%Int32.Op.7ce432.9
// CHECK:STDOUT:   %specific_impl_fn.loc148_5.3 => constants.%Int32.Op.7ce432.9
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.257
// CHECK:STDOUT:   %.loc148_5.5 => constants.%.d3f
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.ecd0a5.2
// CHECK:STDOUT:   %impl.elem0.loc148_5.2 => constants.%Int32.Op.7ce432.1
// CHECK:STDOUT:   %specific_impl_fn.loc148_5.4 => constants.%Int32.Op.7ce432.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @DivWith.loc157(constants.%Destroy.facet.184, constants.%Destroy.facet.184, constants.%facet_value.a53) {
// CHECK:STDOUT:   %Result.loc154_9.1 => constants.%Destroy.facet.184
// CHECK:STDOUT:   %U.loc155_4.1 => constants.%Destroy.facet.184
// CHECK:STDOUT:   %U.as_type.loc156_36.1 => constants.%Int32
// CHECK:STDOUT:   %DivWith.type.loc156_36.1 => constants.%DivWith.type.a2b
// CHECK:STDOUT:   %facet_type.loc156_20 => constants.%facet_type.9ed
// CHECK:STDOUT:   %.Self.loc156_38.1 => constants.%.Self.536
// CHECK:STDOUT:   %require_complete.loc156 => constants.%complete_type.bfd
// CHECK:STDOUT:   %.Self.as_type.loc156_44.1 => constants.%.Self.as_type.a54
// CHECK:STDOUT:   %DivWith.assoc_type => constants.%DivWith.assoc_type.185
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.2d7
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.loc156 => constants.%DivWith.lookup_impl_witness.3ec
// CHECK:STDOUT:   %impl.elem0.loc156_44.1 => constants.%impl.elem0.1dc3
// CHECK:STDOUT:   %Result.as_type.loc156_54.1 => constants.%Int32
// CHECK:STDOUT:   %facet_type.loc156_38 => constants.%facet_type.a43
// CHECK:STDOUT:   %T.loc156_4.1 => constants.%facet_value.a53
// CHECK:STDOUT:   %pattern_type.loc156 => constants.%pattern_type.3fb
// CHECK:STDOUT:   %pattern_type.loc157_16 => constants.%pattern_type.4b8
// CHECK:STDOUT:   %T.as_type.loc157_29.1 => constants.%Int32
// CHECK:STDOUT:   %pattern_type.loc157_27 => constants.%pattern_type.4b8
// CHECK:STDOUT:   %pattern_type.loc157_33 => constants.%pattern_type.4b8
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc157_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc157_27 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc157_33 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc159 => constants.%complete_type.0f3
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.9b2
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.loc159 => constants.%custom_witness.deca3a.2
// CHECK:STDOUT:   %DivWith.facet => constants.%DivWith.facet.ca0
// CHECK:STDOUT:   %DivWith.WithSelf.Op.type => constants.%DivWith.WithSelf.Op.type.971
// CHECK:STDOUT:   %.loc159_5.4 => constants.%.4dc
// CHECK:STDOUT:   %impl.elem1.loc159_5.2 => constants.%Int32.Op.7ce432.10
// CHECK:STDOUT:   %specific_impl_fn.loc159_5.3 => constants.%Int32.Op.7ce432.10
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.257
// CHECK:STDOUT:   %.loc159_5.5 => constants.%.d3f
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.ecd0a5.2
// CHECK:STDOUT:   %impl.elem0.loc159_5.2 => constants.%Int32.Op.7ce432.1
// CHECK:STDOUT:   %specific_impl_fn.loc159_5.4 => constants.%Int32.Op.7ce432.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ModWith.loc168(constants.%Destroy.facet.184, constants.%Destroy.facet.184, constants.%facet_value.821) {
// CHECK:STDOUT:   %Result.loc165_9.1 => constants.%Destroy.facet.184
// CHECK:STDOUT:   %U.loc166_4.1 => constants.%Destroy.facet.184
// CHECK:STDOUT:   %U.as_type.loc167_36.1 => constants.%Int32
// CHECK:STDOUT:   %ModWith.type.loc167_36.1 => constants.%ModWith.type.cff
// CHECK:STDOUT:   %facet_type.loc167_20 => constants.%facet_type.cb7
// CHECK:STDOUT:   %.Self.loc167_38.1 => constants.%.Self.6c7
// CHECK:STDOUT:   %require_complete.loc167 => constants.%complete_type.e06
// CHECK:STDOUT:   %.Self.as_type.loc167_44.1 => constants.%.Self.as_type.2f0
// CHECK:STDOUT:   %ModWith.assoc_type => constants.%ModWith.assoc_type.557
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.beb
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.loc167 => constants.%ModWith.lookup_impl_witness.f61
// CHECK:STDOUT:   %impl.elem0.loc167_44.1 => constants.%impl.elem0.e0b
// CHECK:STDOUT:   %Result.as_type.loc167_54.1 => constants.%Int32
// CHECK:STDOUT:   %facet_type.loc167_38 => constants.%facet_type.e5f
// CHECK:STDOUT:   %T.loc167_4.1 => constants.%facet_value.821
// CHECK:STDOUT:   %pattern_type.loc167 => constants.%pattern_type.011
// CHECK:STDOUT:   %pattern_type.loc168_16 => constants.%pattern_type.4b8
// CHECK:STDOUT:   %T.as_type.loc168_29.1 => constants.%Int32
// CHECK:STDOUT:   %pattern_type.loc168_27 => constants.%pattern_type.4b8
// CHECK:STDOUT:   %pattern_type.loc168_33 => constants.%pattern_type.4b8
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc168_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc168_27 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc168_33 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc170 => constants.%complete_type.fae
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.b6a
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.loc170 => constants.%custom_witness.c72db9.2
// CHECK:STDOUT:   %ModWith.facet => constants.%ModWith.facet.97e
// CHECK:STDOUT:   %ModWith.WithSelf.Op.type => constants.%ModWith.WithSelf.Op.type.927
// CHECK:STDOUT:   %.loc170_5.4 => constants.%.156
// CHECK:STDOUT:   %impl.elem1.loc170_5.2 => constants.%Int32.Op.7ce432.11
// CHECK:STDOUT:   %specific_impl_fn.loc170_5.3 => constants.%Int32.Op.7ce432.11
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.257
// CHECK:STDOUT:   %.loc170_5.5 => constants.%.d3f
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.ecd0a5.2
// CHECK:STDOUT:   %impl.elem0.loc170_5.2 => constants.%Int32.Op.7ce432.1
// CHECK:STDOUT:   %specific_impl_fn.loc170_5.4 => constants.%Int32.Op.7ce432.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AddWith.loc124(constants.%Destroy.facet.b0d, constants.%Destroy.facet.b0d, constants.%facet_value.527) {
// CHECK:STDOUT:   %Result.loc121_9.1 => constants.%Destroy.facet.b0d
// CHECK:STDOUT:   %U.loc122_4.1 => constants.%Destroy.facet.b0d
// CHECK:STDOUT:   %U.as_type.loc123_36.1 => constants.%Int64
// CHECK:STDOUT:   %AddWith.type.loc123_36.1 => constants.%AddWith.type.fef
// CHECK:STDOUT:   %facet_type.loc123_20 => constants.%facet_type.bb9
// CHECK:STDOUT:   %.Self.loc123_38.1 => constants.%.Self.945
// CHECK:STDOUT:   %require_complete.loc123 => constants.%complete_type.fdd
// CHECK:STDOUT:   %.Self.as_type.loc123_44.1 => constants.%.Self.as_type.833
// CHECK:STDOUT:   %AddWith.assoc_type => constants.%AddWith.assoc_type.c7a
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.57b
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.loc123 => constants.%AddWith.lookup_impl_witness.3da
// CHECK:STDOUT:   %impl.elem0.loc123_44.1 => constants.%impl.elem0.e09
// CHECK:STDOUT:   %Result.as_type.loc123_54.1 => constants.%Int64
// CHECK:STDOUT:   %facet_type.loc123_38 => constants.%facet_type.afb
// CHECK:STDOUT:   %T.loc123_4.1 => constants.%facet_value.527
// CHECK:STDOUT:   %pattern_type.loc123 => constants.%pattern_type.8eb
// CHECK:STDOUT:   %pattern_type.loc124_16 => constants.%pattern_type.c76
// CHECK:STDOUT:   %T.as_type.loc124_29.1 => constants.%Int32
// CHECK:STDOUT:   %pattern_type.loc124_27 => constants.%pattern_type.4b8
// CHECK:STDOUT:   %pattern_type.loc124_33 => constants.%pattern_type.c76
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc124_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc124_27 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc124_33 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc126 => constants.%complete_type.477
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.7dd
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.loc126 => constants.%custom_witness.3f7
// CHECK:STDOUT:   %AddWith.facet => constants.%AddWith.facet.19e
// CHECK:STDOUT:   %AddWith.WithSelf.Op.type => constants.%AddWith.WithSelf.Op.type.377
// CHECK:STDOUT:   %.loc126_5.4 => constants.%.16f
// CHECK:STDOUT:   %impl.elem1.loc126_5.2 => constants.%Op.848e8c.6
// CHECK:STDOUT:   %specific_impl_fn.loc126_5.3 => constants.%Op.848e8c.6
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.1c2
// CHECK:STDOUT:   %.loc126_5.5 => constants.%.4de
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.ecd0a5.3
// CHECK:STDOUT:   %impl.elem0.loc126_5.2 => constants.%Int64.Op.7ce432.1
// CHECK:STDOUT:   %specific_impl_fn.loc126_5.4 => constants.%Int64.Op.7ce432.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @SubWith.loc135(constants.%Destroy.facet.b0d, constants.%Destroy.facet.b0d, constants.%facet_value.93f) {
// CHECK:STDOUT:   %Result.loc132_9.1 => constants.%Destroy.facet.b0d
// CHECK:STDOUT:   %U.loc133_4.1 => constants.%Destroy.facet.b0d
// CHECK:STDOUT:   %U.as_type.loc134_36.1 => constants.%Int64
// CHECK:STDOUT:   %SubWith.type.loc134_36.1 => constants.%SubWith.type.c5d
// CHECK:STDOUT:   %facet_type.loc134_20 => constants.%facet_type.4c4
// CHECK:STDOUT:   %.Self.loc134_38.1 => constants.%.Self.89d
// CHECK:STDOUT:   %require_complete.loc134 => constants.%complete_type.bee
// CHECK:STDOUT:   %.Self.as_type.loc134_44.1 => constants.%.Self.as_type.8357
// CHECK:STDOUT:   %SubWith.assoc_type => constants.%SubWith.assoc_type.6bb
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.2092
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.loc134 => constants.%SubWith.lookup_impl_witness.f3c
// CHECK:STDOUT:   %impl.elem0.loc134_44.1 => constants.%impl.elem0.1dcc
// CHECK:STDOUT:   %Result.as_type.loc134_54.1 => constants.%Int64
// CHECK:STDOUT:   %facet_type.loc134_38 => constants.%facet_type.019
// CHECK:STDOUT:   %T.loc134_4.1 => constants.%facet_value.93f
// CHECK:STDOUT:   %pattern_type.loc134 => constants.%pattern_type.37c
// CHECK:STDOUT:   %pattern_type.loc135_16 => constants.%pattern_type.c76
// CHECK:STDOUT:   %T.as_type.loc135_29.1 => constants.%Int32
// CHECK:STDOUT:   %pattern_type.loc135_27 => constants.%pattern_type.4b8
// CHECK:STDOUT:   %pattern_type.loc135_33 => constants.%pattern_type.c76
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc135_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc135_27 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc135_33 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc137 => constants.%complete_type.2df
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.83a
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.loc137 => constants.%custom_witness.aad
// CHECK:STDOUT:   %SubWith.facet => constants.%SubWith.facet.9f0
// CHECK:STDOUT:   %SubWith.WithSelf.Op.type => constants.%SubWith.WithSelf.Op.type.492
// CHECK:STDOUT:   %.loc137_5.4 => constants.%.d94
// CHECK:STDOUT:   %impl.elem1.loc137_5.2 => constants.%Op.848e8c.7
// CHECK:STDOUT:   %specific_impl_fn.loc137_5.3 => constants.%Op.848e8c.7
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.1c2
// CHECK:STDOUT:   %.loc137_5.5 => constants.%.4de
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.ecd0a5.3
// CHECK:STDOUT:   %impl.elem0.loc137_5.2 => constants.%Int64.Op.7ce432.1
// CHECK:STDOUT:   %specific_impl_fn.loc137_5.4 => constants.%Int64.Op.7ce432.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @MulWith.loc146(constants.%Destroy.facet.b0d, constants.%Destroy.facet.b0d, constants.%facet_value.450) {
// CHECK:STDOUT:   %Result.loc143_9.1 => constants.%Destroy.facet.b0d
// CHECK:STDOUT:   %U.loc144_4.1 => constants.%Destroy.facet.b0d
// CHECK:STDOUT:   %U.as_type.loc145_36.1 => constants.%Int64
// CHECK:STDOUT:   %MulWith.type.loc145_36.1 => constants.%MulWith.type.22c
// CHECK:STDOUT:   %facet_type.loc145_20 => constants.%facet_type.5f9
// CHECK:STDOUT:   %.Self.loc145_38.1 => constants.%.Self.baa
// CHECK:STDOUT:   %require_complete.loc145 => constants.%complete_type.4a7
// CHECK:STDOUT:   %.Self.as_type.loc145_44.1 => constants.%.Self.as_type.374
// CHECK:STDOUT:   %MulWith.assoc_type => constants.%MulWith.assoc_type.d02
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.1fd
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.loc145 => constants.%MulWith.lookup_impl_witness.998
// CHECK:STDOUT:   %impl.elem0.loc145_44.1 => constants.%impl.elem0.106
// CHECK:STDOUT:   %Result.as_type.loc145_54.1 => constants.%Int64
// CHECK:STDOUT:   %facet_type.loc145_38 => constants.%facet_type.0c8
// CHECK:STDOUT:   %T.loc145_4.1 => constants.%facet_value.450
// CHECK:STDOUT:   %pattern_type.loc145 => constants.%pattern_type.be2
// CHECK:STDOUT:   %pattern_type.loc146_16 => constants.%pattern_type.c76
// CHECK:STDOUT:   %T.as_type.loc146_29.1 => constants.%Int32
// CHECK:STDOUT:   %pattern_type.loc146_27 => constants.%pattern_type.4b8
// CHECK:STDOUT:   %pattern_type.loc146_33 => constants.%pattern_type.c76
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc146_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc146_27 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc146_33 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc148 => constants.%complete_type.5ad
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.fab
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.loc148 => constants.%custom_witness.1d4
// CHECK:STDOUT:   %MulWith.facet => constants.%MulWith.facet.af5
// CHECK:STDOUT:   %MulWith.WithSelf.Op.type => constants.%MulWith.WithSelf.Op.type.6cd
// CHECK:STDOUT:   %.loc148_5.4 => constants.%.a61
// CHECK:STDOUT:   %impl.elem1.loc148_5.2 => constants.%Op.848e8c.8
// CHECK:STDOUT:   %specific_impl_fn.loc148_5.3 => constants.%Op.848e8c.8
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.1c2
// CHECK:STDOUT:   %.loc148_5.5 => constants.%.4de
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.ecd0a5.3
// CHECK:STDOUT:   %impl.elem0.loc148_5.2 => constants.%Int64.Op.7ce432.1
// CHECK:STDOUT:   %specific_impl_fn.loc148_5.4 => constants.%Int64.Op.7ce432.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @DivWith.loc157(constants.%Destroy.facet.b0d, constants.%Destroy.facet.b0d, constants.%facet_value.2d8) {
// CHECK:STDOUT:   %Result.loc154_9.1 => constants.%Destroy.facet.b0d
// CHECK:STDOUT:   %U.loc155_4.1 => constants.%Destroy.facet.b0d
// CHECK:STDOUT:   %U.as_type.loc156_36.1 => constants.%Int64
// CHECK:STDOUT:   %DivWith.type.loc156_36.1 => constants.%DivWith.type.95f
// CHECK:STDOUT:   %facet_type.loc156_20 => constants.%facet_type.6ae
// CHECK:STDOUT:   %.Self.loc156_38.1 => constants.%.Self.dac
// CHECK:STDOUT:   %require_complete.loc156 => constants.%complete_type.b1e
// CHECK:STDOUT:   %.Self.as_type.loc156_44.1 => constants.%.Self.as_type.771
// CHECK:STDOUT:   %DivWith.assoc_type => constants.%DivWith.assoc_type.7db
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.f64
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.loc156 => constants.%DivWith.lookup_impl_witness.94a
// CHECK:STDOUT:   %impl.elem0.loc156_44.1 => constants.%impl.elem0.d13
// CHECK:STDOUT:   %Result.as_type.loc156_54.1 => constants.%Int64
// CHECK:STDOUT:   %facet_type.loc156_38 => constants.%facet_type.c0d
// CHECK:STDOUT:   %T.loc156_4.1 => constants.%facet_value.2d8
// CHECK:STDOUT:   %pattern_type.loc156 => constants.%pattern_type.841
// CHECK:STDOUT:   %pattern_type.loc157_16 => constants.%pattern_type.c76
// CHECK:STDOUT:   %T.as_type.loc157_29.1 => constants.%Int32
// CHECK:STDOUT:   %pattern_type.loc157_27 => constants.%pattern_type.4b8
// CHECK:STDOUT:   %pattern_type.loc157_33 => constants.%pattern_type.c76
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc157_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc157_27 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc157_33 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc159 => constants.%complete_type.5d2
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.0b4
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.loc159 => constants.%custom_witness.515
// CHECK:STDOUT:   %DivWith.facet => constants.%DivWith.facet.731
// CHECK:STDOUT:   %DivWith.WithSelf.Op.type => constants.%DivWith.WithSelf.Op.type.614
// CHECK:STDOUT:   %.loc159_5.4 => constants.%.448
// CHECK:STDOUT:   %impl.elem1.loc159_5.2 => constants.%Op.848e8c.9
// CHECK:STDOUT:   %specific_impl_fn.loc159_5.3 => constants.%Op.848e8c.9
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.1c2
// CHECK:STDOUT:   %.loc159_5.5 => constants.%.4de
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.ecd0a5.3
// CHECK:STDOUT:   %impl.elem0.loc159_5.2 => constants.%Int64.Op.7ce432.1
// CHECK:STDOUT:   %specific_impl_fn.loc159_5.4 => constants.%Int64.Op.7ce432.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ModWith.loc168(constants.%Destroy.facet.b0d, constants.%Destroy.facet.b0d, constants.%facet_value.bab) {
// CHECK:STDOUT:   %Result.loc165_9.1 => constants.%Destroy.facet.b0d
// CHECK:STDOUT:   %U.loc166_4.1 => constants.%Destroy.facet.b0d
// CHECK:STDOUT:   %U.as_type.loc167_36.1 => constants.%Int64
// CHECK:STDOUT:   %ModWith.type.loc167_36.1 => constants.%ModWith.type.e6e
// CHECK:STDOUT:   %facet_type.loc167_20 => constants.%facet_type.c80
// CHECK:STDOUT:   %.Self.loc167_38.1 => constants.%.Self.a93
// CHECK:STDOUT:   %require_complete.loc167 => constants.%complete_type.320
// CHECK:STDOUT:   %.Self.as_type.loc167_44.1 => constants.%.Self.as_type.d20
// CHECK:STDOUT:   %ModWith.assoc_type => constants.%ModWith.assoc_type.ac1
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.97e
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.loc167 => constants.%ModWith.lookup_impl_witness.324
// CHECK:STDOUT:   %impl.elem0.loc167_44.1 => constants.%impl.elem0.96f
// CHECK:STDOUT:   %Result.as_type.loc167_54.1 => constants.%Int64
// CHECK:STDOUT:   %facet_type.loc167_38 => constants.%facet_type.b48
// CHECK:STDOUT:   %T.loc167_4.1 => constants.%facet_value.bab
// CHECK:STDOUT:   %pattern_type.loc167 => constants.%pattern_type.250
// CHECK:STDOUT:   %pattern_type.loc168_16 => constants.%pattern_type.c76
// CHECK:STDOUT:   %T.as_type.loc168_29.1 => constants.%Int32
// CHECK:STDOUT:   %pattern_type.loc168_27 => constants.%pattern_type.4b8
// CHECK:STDOUT:   %pattern_type.loc168_33 => constants.%pattern_type.c76
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc168_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc168_27 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc168_33 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc170 => constants.%complete_type.d30
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.6f9
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.loc170 => constants.%custom_witness.1af
// CHECK:STDOUT:   %ModWith.facet => constants.%ModWith.facet.fff
// CHECK:STDOUT:   %ModWith.WithSelf.Op.type => constants.%ModWith.WithSelf.Op.type.de3
// CHECK:STDOUT:   %.loc170_5.4 => constants.%.401
// CHECK:STDOUT:   %impl.elem1.loc170_5.2 => constants.%Op.848e8c.10
// CHECK:STDOUT:   %specific_impl_fn.loc170_5.3 => constants.%Op.848e8c.10
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.1c2
// CHECK:STDOUT:   %.loc170_5.5 => constants.%.4de
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.ecd0a5.3
// CHECK:STDOUT:   %impl.elem0.loc170_5.2 => constants.%Int64.Op.7ce432.1
// CHECK:STDOUT:   %specific_impl_fn.loc170_5.4 => constants.%Int64.Op.7ce432.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AddWith.loc124(constants.%Destroy.facet.b0d, constants.%Destroy.facet.184, constants.%facet_value.461) {
// CHECK:STDOUT:   %Result.loc121_9.1 => constants.%Destroy.facet.b0d
// CHECK:STDOUT:   %U.loc122_4.1 => constants.%Destroy.facet.184
// CHECK:STDOUT:   %U.as_type.loc123_36.1 => constants.%Int32
// CHECK:STDOUT:   %AddWith.type.loc123_36.1 => constants.%AddWith.type.4f1
// CHECK:STDOUT:   %facet_type.loc123_20 => constants.%facet_type.35e
// CHECK:STDOUT:   %.Self.loc123_38.1 => constants.%.Self.ea4
// CHECK:STDOUT:   %require_complete.loc123 => constants.%complete_type.bc1
// CHECK:STDOUT:   %.Self.as_type.loc123_44.1 => constants.%.Self.as_type.1e3
// CHECK:STDOUT:   %AddWith.assoc_type => constants.%AddWith.assoc_type.5fc
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.836
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.loc123 => constants.%AddWith.lookup_impl_witness.587
// CHECK:STDOUT:   %impl.elem0.loc123_44.1 => constants.%impl.elem0.4f1
// CHECK:STDOUT:   %Result.as_type.loc123_54.1 => constants.%Int64
// CHECK:STDOUT:   %facet_type.loc123_38 => constants.%facet_type.8f4
// CHECK:STDOUT:   %T.loc123_4.1 => constants.%facet_value.461
// CHECK:STDOUT:   %pattern_type.loc123 => constants.%pattern_type.ff2
// CHECK:STDOUT:   %pattern_type.loc124_16 => constants.%pattern_type.c76
// CHECK:STDOUT:   %T.as_type.loc124_29.1 => constants.%Int64
// CHECK:STDOUT:   %pattern_type.loc124_27 => constants.%pattern_type.c76
// CHECK:STDOUT:   %pattern_type.loc124_33 => constants.%pattern_type.4b8
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc124_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc124_27 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc124_33 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc126 => constants.%complete_type.e69
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.96b
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.loc126 => constants.%custom_witness.dbe
// CHECK:STDOUT:   %AddWith.facet => constants.%AddWith.facet.1a3
// CHECK:STDOUT:   %AddWith.WithSelf.Op.type => constants.%AddWith.WithSelf.Op.type.bc9
// CHECK:STDOUT:   %.loc126_5.4 => constants.%.cec
// CHECK:STDOUT:   %impl.elem1.loc126_5.2 => constants.%Op.848e8c.11
// CHECK:STDOUT:   %specific_impl_fn.loc126_5.3 => constants.%Op.848e8c.11
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.1c2
// CHECK:STDOUT:   %.loc126_5.5 => constants.%.4de
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.ecd0a5.3
// CHECK:STDOUT:   %impl.elem0.loc126_5.2 => constants.%Int64.Op.7ce432.1
// CHECK:STDOUT:   %specific_impl_fn.loc126_5.4 => constants.%Int64.Op.7ce432.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @SubWith.loc135(constants.%Destroy.facet.b0d, constants.%Destroy.facet.184, constants.%facet_value.ec6) {
// CHECK:STDOUT:   %Result.loc132_9.1 => constants.%Destroy.facet.b0d
// CHECK:STDOUT:   %U.loc133_4.1 => constants.%Destroy.facet.184
// CHECK:STDOUT:   %U.as_type.loc134_36.1 => constants.%Int32
// CHECK:STDOUT:   %SubWith.type.loc134_36.1 => constants.%SubWith.type.40d
// CHECK:STDOUT:   %facet_type.loc134_20 => constants.%facet_type.cbc
// CHECK:STDOUT:   %.Self.loc134_38.1 => constants.%.Self.ae4
// CHECK:STDOUT:   %require_complete.loc134 => constants.%complete_type.44f
// CHECK:STDOUT:   %.Self.as_type.loc134_44.1 => constants.%.Self.as_type.484
// CHECK:STDOUT:   %SubWith.assoc_type => constants.%SubWith.assoc_type.73f
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.c3d
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.loc134 => constants.%SubWith.lookup_impl_witness.d98
// CHECK:STDOUT:   %impl.elem0.loc134_44.1 => constants.%impl.elem0.f09
// CHECK:STDOUT:   %Result.as_type.loc134_54.1 => constants.%Int64
// CHECK:STDOUT:   %facet_type.loc134_38 => constants.%facet_type.458
// CHECK:STDOUT:   %T.loc134_4.1 => constants.%facet_value.ec6
// CHECK:STDOUT:   %pattern_type.loc134 => constants.%pattern_type.679
// CHECK:STDOUT:   %pattern_type.loc135_16 => constants.%pattern_type.c76
// CHECK:STDOUT:   %T.as_type.loc135_29.1 => constants.%Int64
// CHECK:STDOUT:   %pattern_type.loc135_27 => constants.%pattern_type.c76
// CHECK:STDOUT:   %pattern_type.loc135_33 => constants.%pattern_type.4b8
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc135_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc135_27 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc135_33 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc137 => constants.%complete_type.c40
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.827
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.loc137 => constants.%custom_witness.af3
// CHECK:STDOUT:   %SubWith.facet => constants.%SubWith.facet.8e2
// CHECK:STDOUT:   %SubWith.WithSelf.Op.type => constants.%SubWith.WithSelf.Op.type.634
// CHECK:STDOUT:   %.loc137_5.4 => constants.%.72e
// CHECK:STDOUT:   %impl.elem1.loc137_5.2 => constants.%Op.848e8c.12
// CHECK:STDOUT:   %specific_impl_fn.loc137_5.3 => constants.%Op.848e8c.12
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.1c2
// CHECK:STDOUT:   %.loc137_5.5 => constants.%.4de
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.ecd0a5.3
// CHECK:STDOUT:   %impl.elem0.loc137_5.2 => constants.%Int64.Op.7ce432.1
// CHECK:STDOUT:   %specific_impl_fn.loc137_5.4 => constants.%Int64.Op.7ce432.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @MulWith.loc146(constants.%Destroy.facet.b0d, constants.%Destroy.facet.184, constants.%facet_value.7d6) {
// CHECK:STDOUT:   %Result.loc143_9.1 => constants.%Destroy.facet.b0d
// CHECK:STDOUT:   %U.loc144_4.1 => constants.%Destroy.facet.184
// CHECK:STDOUT:   %U.as_type.loc145_36.1 => constants.%Int32
// CHECK:STDOUT:   %MulWith.type.loc145_36.1 => constants.%MulWith.type.8d1
// CHECK:STDOUT:   %facet_type.loc145_20 => constants.%facet_type.655
// CHECK:STDOUT:   %.Self.loc145_38.1 => constants.%.Self.e02
// CHECK:STDOUT:   %require_complete.loc145 => constants.%complete_type.aab
// CHECK:STDOUT:   %.Self.as_type.loc145_44.1 => constants.%.Self.as_type.efe
// CHECK:STDOUT:   %MulWith.assoc_type => constants.%MulWith.assoc_type.d17
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.225
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.loc145 => constants.%MulWith.lookup_impl_witness.5f7
// CHECK:STDOUT:   %impl.elem0.loc145_44.1 => constants.%impl.elem0.0bb
// CHECK:STDOUT:   %Result.as_type.loc145_54.1 => constants.%Int64
// CHECK:STDOUT:   %facet_type.loc145_38 => constants.%facet_type.fe5
// CHECK:STDOUT:   %T.loc145_4.1 => constants.%facet_value.7d6
// CHECK:STDOUT:   %pattern_type.loc145 => constants.%pattern_type.66a
// CHECK:STDOUT:   %pattern_type.loc146_16 => constants.%pattern_type.c76
// CHECK:STDOUT:   %T.as_type.loc146_29.1 => constants.%Int64
// CHECK:STDOUT:   %pattern_type.loc146_27 => constants.%pattern_type.c76
// CHECK:STDOUT:   %pattern_type.loc146_33 => constants.%pattern_type.4b8
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc146_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc146_27 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc146_33 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc148 => constants.%complete_type.ab9
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.a92
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.loc148 => constants.%custom_witness.d10
// CHECK:STDOUT:   %MulWith.facet => constants.%MulWith.facet.d5b
// CHECK:STDOUT:   %MulWith.WithSelf.Op.type => constants.%MulWith.WithSelf.Op.type.21f
// CHECK:STDOUT:   %.loc148_5.4 => constants.%.ff7
// CHECK:STDOUT:   %impl.elem1.loc148_5.2 => constants.%Op.848e8c.13
// CHECK:STDOUT:   %specific_impl_fn.loc148_5.3 => constants.%Op.848e8c.13
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.1c2
// CHECK:STDOUT:   %.loc148_5.5 => constants.%.4de
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.ecd0a5.3
// CHECK:STDOUT:   %impl.elem0.loc148_5.2 => constants.%Int64.Op.7ce432.1
// CHECK:STDOUT:   %specific_impl_fn.loc148_5.4 => constants.%Int64.Op.7ce432.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @DivWith.loc157(constants.%Destroy.facet.b0d, constants.%Destroy.facet.184, constants.%facet_value.798) {
// CHECK:STDOUT:   %Result.loc154_9.1 => constants.%Destroy.facet.b0d
// CHECK:STDOUT:   %U.loc155_4.1 => constants.%Destroy.facet.184
// CHECK:STDOUT:   %U.as_type.loc156_36.1 => constants.%Int32
// CHECK:STDOUT:   %DivWith.type.loc156_36.1 => constants.%DivWith.type.a2b
// CHECK:STDOUT:   %facet_type.loc156_20 => constants.%facet_type.9ed
// CHECK:STDOUT:   %.Self.loc156_38.1 => constants.%.Self.536
// CHECK:STDOUT:   %require_complete.loc156 => constants.%complete_type.bfd
// CHECK:STDOUT:   %.Self.as_type.loc156_44.1 => constants.%.Self.as_type.a54
// CHECK:STDOUT:   %DivWith.assoc_type => constants.%DivWith.assoc_type.185
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.2d7
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.loc156 => constants.%DivWith.lookup_impl_witness.3ec
// CHECK:STDOUT:   %impl.elem0.loc156_44.1 => constants.%impl.elem0.1dc3
// CHECK:STDOUT:   %Result.as_type.loc156_54.1 => constants.%Int64
// CHECK:STDOUT:   %facet_type.loc156_38 => constants.%facet_type.b92
// CHECK:STDOUT:   %T.loc156_4.1 => constants.%facet_value.798
// CHECK:STDOUT:   %pattern_type.loc156 => constants.%pattern_type.624
// CHECK:STDOUT:   %pattern_type.loc157_16 => constants.%pattern_type.c76
// CHECK:STDOUT:   %T.as_type.loc157_29.1 => constants.%Int64
// CHECK:STDOUT:   %pattern_type.loc157_27 => constants.%pattern_type.c76
// CHECK:STDOUT:   %pattern_type.loc157_33 => constants.%pattern_type.4b8
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc157_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc157_27 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc157_33 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc159 => constants.%complete_type.0f3
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.9b2
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.loc159 => constants.%custom_witness.438
// CHECK:STDOUT:   %DivWith.facet => constants.%DivWith.facet.efb
// CHECK:STDOUT:   %DivWith.WithSelf.Op.type => constants.%DivWith.WithSelf.Op.type.0cc
// CHECK:STDOUT:   %.loc159_5.4 => constants.%.291
// CHECK:STDOUT:   %impl.elem1.loc159_5.2 => constants.%Op.848e8c.14
// CHECK:STDOUT:   %specific_impl_fn.loc159_5.3 => constants.%Op.848e8c.14
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.1c2
// CHECK:STDOUT:   %.loc159_5.5 => constants.%.4de
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.ecd0a5.3
// CHECK:STDOUT:   %impl.elem0.loc159_5.2 => constants.%Int64.Op.7ce432.1
// CHECK:STDOUT:   %specific_impl_fn.loc159_5.4 => constants.%Int64.Op.7ce432.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ModWith.loc168(constants.%Destroy.facet.b0d, constants.%Destroy.facet.184, constants.%facet_value.dd8) {
// CHECK:STDOUT:   %Result.loc165_9.1 => constants.%Destroy.facet.b0d
// CHECK:STDOUT:   %U.loc166_4.1 => constants.%Destroy.facet.184
// CHECK:STDOUT:   %U.as_type.loc167_36.1 => constants.%Int32
// CHECK:STDOUT:   %ModWith.type.loc167_36.1 => constants.%ModWith.type.cff
// CHECK:STDOUT:   %facet_type.loc167_20 => constants.%facet_type.cb7
// CHECK:STDOUT:   %.Self.loc167_38.1 => constants.%.Self.6c7
// CHECK:STDOUT:   %require_complete.loc167 => constants.%complete_type.e06
// CHECK:STDOUT:   %.Self.as_type.loc167_44.1 => constants.%.Self.as_type.2f0
// CHECK:STDOUT:   %ModWith.assoc_type => constants.%ModWith.assoc_type.557
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.beb
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.loc167 => constants.%ModWith.lookup_impl_witness.f61
// CHECK:STDOUT:   %impl.elem0.loc167_44.1 => constants.%impl.elem0.e0b
// CHECK:STDOUT:   %Result.as_type.loc167_54.1 => constants.%Int64
// CHECK:STDOUT:   %facet_type.loc167_38 => constants.%facet_type.235
// CHECK:STDOUT:   %T.loc167_4.1 => constants.%facet_value.dd8
// CHECK:STDOUT:   %pattern_type.loc167 => constants.%pattern_type.0fa
// CHECK:STDOUT:   %pattern_type.loc168_16 => constants.%pattern_type.c76
// CHECK:STDOUT:   %T.as_type.loc168_29.1 => constants.%Int64
// CHECK:STDOUT:   %pattern_type.loc168_27 => constants.%pattern_type.c76
// CHECK:STDOUT:   %pattern_type.loc168_33 => constants.%pattern_type.4b8
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc168_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc168_27 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc168_33 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc170 => constants.%complete_type.fae
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.b6a
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.loc170 => constants.%custom_witness.bf8
// CHECK:STDOUT:   %ModWith.facet => constants.%ModWith.facet.f6a
// CHECK:STDOUT:   %ModWith.WithSelf.Op.type => constants.%ModWith.WithSelf.Op.type.da4
// CHECK:STDOUT:   %.loc170_5.4 => constants.%.7a88
// CHECK:STDOUT:   %impl.elem1.loc170_5.2 => constants.%Op.848e8c.15
// CHECK:STDOUT:   %specific_impl_fn.loc170_5.3 => constants.%Op.848e8c.15
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.1c2
// CHECK:STDOUT:   %.loc170_5.5 => constants.%.4de
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.ecd0a5.3
// CHECK:STDOUT:   %impl.elem0.loc170_5.2 => constants.%Int64.Op.7ce432.1
// CHECK:STDOUT:   %specific_impl_fn.loc170_5.4 => constants.%Int64.Op.7ce432.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AddAssignWith.loc225(constants.%U.67d, constants.%T.c28) {
// CHECK:STDOUT:   %U.loc225_19.1 => constants.%U.67d
// CHECK:STDOUT:   %AddAssignWith.type.loc225_52.1 => constants.%AddAssignWith.type.d3e5c5.2
// CHECK:STDOUT:   %T.loc225_29.1 => constants.%T.c28
// CHECK:STDOUT:   %pattern_type.loc225_29 => constants.%pattern_type.b54
// CHECK:STDOUT:   %T.as_type.loc225_62.1 => constants.%T.as_type.984
// CHECK:STDOUT:   %pattern_type.loc225_60 => constants.%pattern_type.7bda36.2
// CHECK:STDOUT:   %pattern_type.loc225_66 => constants.%pattern_type.51d1c4.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @SubAssignWith.loc231(constants.%U.67d, constants.%T.bd9) {
// CHECK:STDOUT:   %U.loc231_19.1 => constants.%U.67d
// CHECK:STDOUT:   %SubAssignWith.type.loc231_52.1 => constants.%SubAssignWith.type.267e69.2
// CHECK:STDOUT:   %T.loc231_29.1 => constants.%T.bd9
// CHECK:STDOUT:   %pattern_type.loc231_29 => constants.%pattern_type.c3f
// CHECK:STDOUT:   %T.as_type.loc231_62.1 => constants.%T.as_type.b79
// CHECK:STDOUT:   %pattern_type.loc231_60 => constants.%pattern_type.944ba1.2
// CHECK:STDOUT:   %pattern_type.loc231_66 => constants.%pattern_type.51d1c4.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @MulAssignWith.loc237(constants.%U.67d, constants.%T.b81) {
// CHECK:STDOUT:   %U.loc237_19.1 => constants.%U.67d
// CHECK:STDOUT:   %MulAssignWith.type.loc237_52.1 => constants.%MulAssignWith.type.802f2e.2
// CHECK:STDOUT:   %T.loc237_29.1 => constants.%T.b81
// CHECK:STDOUT:   %pattern_type.loc237_29 => constants.%pattern_type.306
// CHECK:STDOUT:   %T.as_type.loc237_62.1 => constants.%T.as_type.2ad
// CHECK:STDOUT:   %pattern_type.loc237_60 => constants.%pattern_type.cd4e6a.2
// CHECK:STDOUT:   %pattern_type.loc237_66 => constants.%pattern_type.51d1c4.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @DivAssignWith.loc243(constants.%U.67d, constants.%T.e57) {
// CHECK:STDOUT:   %U.loc243_19.1 => constants.%U.67d
// CHECK:STDOUT:   %DivAssignWith.type.loc243_52.1 => constants.%DivAssignWith.type.2e6641.2
// CHECK:STDOUT:   %T.loc243_29.1 => constants.%T.e57
// CHECK:STDOUT:   %pattern_type.loc243_29 => constants.%pattern_type.72f
// CHECK:STDOUT:   %T.as_type.loc243_62.1 => constants.%T.as_type.905
// CHECK:STDOUT:   %pattern_type.loc243_60 => constants.%pattern_type.5d02d2.2
// CHECK:STDOUT:   %pattern_type.loc243_66 => constants.%pattern_type.51d1c4.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ModAssignWith.loc249(constants.%U.67d, constants.%T.aac) {
// CHECK:STDOUT:   %U.loc249_19.1 => constants.%U.67d
// CHECK:STDOUT:   %ModAssignWith.type.loc249_52.1 => constants.%ModAssignWith.type.6d648f.2
// CHECK:STDOUT:   %T.loc249_29.1 => constants.%T.aac
// CHECK:STDOUT:   %pattern_type.loc249_29 => constants.%pattern_type.9b9
// CHECK:STDOUT:   %T.as_type.loc249_62.1 => constants.%T.as_type.ab6
// CHECK:STDOUT:   %pattern_type.loc249_60 => constants.%pattern_type.102a46.2
// CHECK:STDOUT:   %pattern_type.loc249_66 => constants.%pattern_type.51d1c4.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AddAssignWith.loc225(constants.%i16, constants.%AddAssignWith.facet.52e) {
// CHECK:STDOUT:   %U.loc225_19.1 => constants.%i16
// CHECK:STDOUT:   %AddAssignWith.type.loc225_52.1 => constants.%AddAssignWith.type.002
// CHECK:STDOUT:   %T.loc225_29.1 => constants.%AddAssignWith.facet.52e
// CHECK:STDOUT:   %pattern_type.loc225_29 => constants.%pattern_type.866
// CHECK:STDOUT:   %T.as_type.loc225_62.1 => constants.%Int16
// CHECK:STDOUT:   %pattern_type.loc225_60 => constants.%pattern_type.b06
// CHECK:STDOUT:   %pattern_type.loc225_66 => constants.%pattern_type.0e3
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc225_55 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc225_66 => constants.%complete_type.2da
// CHECK:STDOUT:   %require_complete.loc227 => constants.%complete_type.95c
// CHECK:STDOUT:   %AddAssignWith.assoc_type => constants.%AddAssignWith.assoc_type.a10
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.e25
// CHECK:STDOUT:   %AddAssignWith.WithSelf.Op.type => constants.%AddAssignWith.WithSelf.Op.type.616
// CHECK:STDOUT:   %.loc227_5.2 => constants.%.5b6
// CHECK:STDOUT:   %AddAssignWith.lookup_impl_witness => constants.%custom_witness.9ba
// CHECK:STDOUT:   %impl.elem0.loc227_5.2 => constants.%Int16.Op.7ce432.13
// CHECK:STDOUT:   %specific_impl_fn.loc227_5.2 => constants.%Int16.Op.7ce432.13
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @SubAssignWith.loc231(constants.%i16, constants.%SubAssignWith.facet.a61) {
// CHECK:STDOUT:   %U.loc231_19.1 => constants.%i16
// CHECK:STDOUT:   %SubAssignWith.type.loc231_52.1 => constants.%SubAssignWith.type.08c
// CHECK:STDOUT:   %T.loc231_29.1 => constants.%SubAssignWith.facet.a61
// CHECK:STDOUT:   %pattern_type.loc231_29 => constants.%pattern_type.a60
// CHECK:STDOUT:   %T.as_type.loc231_62.1 => constants.%Int16
// CHECK:STDOUT:   %pattern_type.loc231_60 => constants.%pattern_type.b06
// CHECK:STDOUT:   %pattern_type.loc231_66 => constants.%pattern_type.0e3
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc231_55 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc231_66 => constants.%complete_type.2da
// CHECK:STDOUT:   %require_complete.loc233 => constants.%complete_type.f27
// CHECK:STDOUT:   %SubAssignWith.assoc_type => constants.%SubAssignWith.assoc_type.57e
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.acf
// CHECK:STDOUT:   %SubAssignWith.WithSelf.Op.type => constants.%SubAssignWith.WithSelf.Op.type.93e
// CHECK:STDOUT:   %.loc233_5.2 => constants.%.49f
// CHECK:STDOUT:   %SubAssignWith.lookup_impl_witness => constants.%custom_witness.6ff
// CHECK:STDOUT:   %impl.elem0.loc233_5.2 => constants.%Int16.Op.7ce432.14
// CHECK:STDOUT:   %specific_impl_fn.loc233_5.2 => constants.%Int16.Op.7ce432.14
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @MulAssignWith.loc237(constants.%i16, constants.%MulAssignWith.facet.253) {
// CHECK:STDOUT:   %U.loc237_19.1 => constants.%i16
// CHECK:STDOUT:   %MulAssignWith.type.loc237_52.1 => constants.%MulAssignWith.type.767
// CHECK:STDOUT:   %T.loc237_29.1 => constants.%MulAssignWith.facet.253
// CHECK:STDOUT:   %pattern_type.loc237_29 => constants.%pattern_type.eb7
// CHECK:STDOUT:   %T.as_type.loc237_62.1 => constants.%Int16
// CHECK:STDOUT:   %pattern_type.loc237_60 => constants.%pattern_type.b06
// CHECK:STDOUT:   %pattern_type.loc237_66 => constants.%pattern_type.0e3
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc237_55 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc237_66 => constants.%complete_type.2da
// CHECK:STDOUT:   %require_complete.loc239 => constants.%complete_type.cbb
// CHECK:STDOUT:   %MulAssignWith.assoc_type => constants.%MulAssignWith.assoc_type.c88
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.bb1
// CHECK:STDOUT:   %MulAssignWith.WithSelf.Op.type => constants.%MulAssignWith.WithSelf.Op.type.bc0
// CHECK:STDOUT:   %.loc239_5.2 => constants.%.c0a
// CHECK:STDOUT:   %MulAssignWith.lookup_impl_witness => constants.%custom_witness.1dc
// CHECK:STDOUT:   %impl.elem0.loc239_5.2 => constants.%Int16.Op.7ce432.15
// CHECK:STDOUT:   %specific_impl_fn.loc239_5.2 => constants.%Int16.Op.7ce432.15
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @DivAssignWith.loc243(constants.%i16, constants.%DivAssignWith.facet.14d) {
// CHECK:STDOUT:   %U.loc243_19.1 => constants.%i16
// CHECK:STDOUT:   %DivAssignWith.type.loc243_52.1 => constants.%DivAssignWith.type.61f
// CHECK:STDOUT:   %T.loc243_29.1 => constants.%DivAssignWith.facet.14d
// CHECK:STDOUT:   %pattern_type.loc243_29 => constants.%pattern_type.bfb
// CHECK:STDOUT:   %T.as_type.loc243_62.1 => constants.%Int16
// CHECK:STDOUT:   %pattern_type.loc243_60 => constants.%pattern_type.b06
// CHECK:STDOUT:   %pattern_type.loc243_66 => constants.%pattern_type.0e3
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc243_55 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc243_66 => constants.%complete_type.2da
// CHECK:STDOUT:   %require_complete.loc245 => constants.%complete_type.e4b
// CHECK:STDOUT:   %DivAssignWith.assoc_type => constants.%DivAssignWith.assoc_type.73f
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.e4b
// CHECK:STDOUT:   %DivAssignWith.WithSelf.Op.type => constants.%DivAssignWith.WithSelf.Op.type.42f
// CHECK:STDOUT:   %.loc245_5.2 => constants.%.506
// CHECK:STDOUT:   %DivAssignWith.lookup_impl_witness => constants.%custom_witness.780
// CHECK:STDOUT:   %impl.elem0.loc245_5.2 => constants.%Int16.Op.7ce432.16
// CHECK:STDOUT:   %specific_impl_fn.loc245_5.2 => constants.%Int16.Op.7ce432.16
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ModAssignWith.loc249(constants.%i16, constants.%ModAssignWith.facet.8a8) {
// CHECK:STDOUT:   %U.loc249_19.1 => constants.%i16
// CHECK:STDOUT:   %ModAssignWith.type.loc249_52.1 => constants.%ModAssignWith.type.48f
// CHECK:STDOUT:   %T.loc249_29.1 => constants.%ModAssignWith.facet.8a8
// CHECK:STDOUT:   %pattern_type.loc249_29 => constants.%pattern_type.dbf
// CHECK:STDOUT:   %T.as_type.loc249_62.1 => constants.%Int16
// CHECK:STDOUT:   %pattern_type.loc249_60 => constants.%pattern_type.b06
// CHECK:STDOUT:   %pattern_type.loc249_66 => constants.%pattern_type.0e3
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc249_55 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc249_66 => constants.%complete_type.2da
// CHECK:STDOUT:   %require_complete.loc251 => constants.%complete_type.9a2
// CHECK:STDOUT:   %ModAssignWith.assoc_type => constants.%ModAssignWith.assoc_type.c84
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.2c9
// CHECK:STDOUT:   %ModAssignWith.WithSelf.Op.type => constants.%ModAssignWith.WithSelf.Op.type.3a5
// CHECK:STDOUT:   %.loc251_5.2 => constants.%.dd5
// CHECK:STDOUT:   %ModAssignWith.lookup_impl_witness => constants.%custom_witness.4e8
// CHECK:STDOUT:   %impl.elem0.loc251_5.2 => constants.%Int16.Op.7ce432.17
// CHECK:STDOUT:   %specific_impl_fn.loc251_5.2 => constants.%Int16.Op.7ce432.17
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AddAssignWith.loc225(constants.%Int32, constants.%AddAssignWith.facet.d99) {
// CHECK:STDOUT:   %U.loc225_19.1 => constants.%Int32
// CHECK:STDOUT:   %AddAssignWith.type.loc225_52.1 => constants.%AddAssignWith.type.eee
// CHECK:STDOUT:   %T.loc225_29.1 => constants.%AddAssignWith.facet.d99
// CHECK:STDOUT:   %pattern_type.loc225_29 => constants.%pattern_type.e31
// CHECK:STDOUT:   %T.as_type.loc225_62.1 => constants.%Int32
// CHECK:STDOUT:   %pattern_type.loc225_60 => constants.%pattern_type.4b8
// CHECK:STDOUT:   %pattern_type.loc225_66 => constants.%pattern_type.4b8
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc225_55 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc225_66 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc227 => constants.%complete_type.987
// CHECK:STDOUT:   %AddAssignWith.assoc_type => constants.%AddAssignWith.assoc_type.7f0
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.b73
// CHECK:STDOUT:   %AddAssignWith.WithSelf.Op.type => constants.%AddAssignWith.WithSelf.Op.type.e5e
// CHECK:STDOUT:   %.loc227_5.2 => constants.%.353
// CHECK:STDOUT:   %AddAssignWith.lookup_impl_witness => constants.%custom_witness.2be94e.1
// CHECK:STDOUT:   %impl.elem0.loc227_5.2 => constants.%Int32.Op.7ce432.12
// CHECK:STDOUT:   %specific_impl_fn.loc227_5.2 => constants.%Int32.Op.7ce432.12
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @SubAssignWith.loc231(constants.%Int32, constants.%SubAssignWith.facet.8ee) {
// CHECK:STDOUT:   %U.loc231_19.1 => constants.%Int32
// CHECK:STDOUT:   %SubAssignWith.type.loc231_52.1 => constants.%SubAssignWith.type.eed
// CHECK:STDOUT:   %T.loc231_29.1 => constants.%SubAssignWith.facet.8ee
// CHECK:STDOUT:   %pattern_type.loc231_29 => constants.%pattern_type.1aa
// CHECK:STDOUT:   %T.as_type.loc231_62.1 => constants.%Int32
// CHECK:STDOUT:   %pattern_type.loc231_60 => constants.%pattern_type.4b8
// CHECK:STDOUT:   %pattern_type.loc231_66 => constants.%pattern_type.4b8
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc231_55 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc231_66 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc233 => constants.%complete_type.cb4
// CHECK:STDOUT:   %SubAssignWith.assoc_type => constants.%SubAssignWith.assoc_type.0dd
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.d09
// CHECK:STDOUT:   %SubAssignWith.WithSelf.Op.type => constants.%SubAssignWith.WithSelf.Op.type.446
// CHECK:STDOUT:   %.loc233_5.2 => constants.%.4fb
// CHECK:STDOUT:   %SubAssignWith.lookup_impl_witness => constants.%custom_witness.3ff7e3.1
// CHECK:STDOUT:   %impl.elem0.loc233_5.2 => constants.%Int32.Op.7ce432.14
// CHECK:STDOUT:   %specific_impl_fn.loc233_5.2 => constants.%Int32.Op.7ce432.14
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @MulAssignWith.loc237(constants.%Int32, constants.%MulAssignWith.facet.7ec) {
// CHECK:STDOUT:   %U.loc237_19.1 => constants.%Int32
// CHECK:STDOUT:   %MulAssignWith.type.loc237_52.1 => constants.%MulAssignWith.type.8e6
// CHECK:STDOUT:   %T.loc237_29.1 => constants.%MulAssignWith.facet.7ec
// CHECK:STDOUT:   %pattern_type.loc237_29 => constants.%pattern_type.a97
// CHECK:STDOUT:   %T.as_type.loc237_62.1 => constants.%Int32
// CHECK:STDOUT:   %pattern_type.loc237_60 => constants.%pattern_type.4b8
// CHECK:STDOUT:   %pattern_type.loc237_66 => constants.%pattern_type.4b8
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc237_55 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc237_66 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc239 => constants.%complete_type.5a8
// CHECK:STDOUT:   %MulAssignWith.assoc_type => constants.%MulAssignWith.assoc_type.4c9
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.4fa
// CHECK:STDOUT:   %MulAssignWith.WithSelf.Op.type => constants.%MulAssignWith.WithSelf.Op.type.8e5
// CHECK:STDOUT:   %.loc239_5.2 => constants.%.fef
// CHECK:STDOUT:   %MulAssignWith.lookup_impl_witness => constants.%custom_witness.a36f66.1
// CHECK:STDOUT:   %impl.elem0.loc239_5.2 => constants.%Int32.Op.7ce432.15
// CHECK:STDOUT:   %specific_impl_fn.loc239_5.2 => constants.%Int32.Op.7ce432.15
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @DivAssignWith.loc243(constants.%Int32, constants.%DivAssignWith.facet.a84) {
// CHECK:STDOUT:   %U.loc243_19.1 => constants.%Int32
// CHECK:STDOUT:   %DivAssignWith.type.loc243_52.1 => constants.%DivAssignWith.type.f58
// CHECK:STDOUT:   %T.loc243_29.1 => constants.%DivAssignWith.facet.a84
// CHECK:STDOUT:   %pattern_type.loc243_29 => constants.%pattern_type.2fa
// CHECK:STDOUT:   %T.as_type.loc243_62.1 => constants.%Int32
// CHECK:STDOUT:   %pattern_type.loc243_60 => constants.%pattern_type.4b8
// CHECK:STDOUT:   %pattern_type.loc243_66 => constants.%pattern_type.4b8
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc243_55 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc243_66 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc245 => constants.%complete_type.3ca
// CHECK:STDOUT:   %DivAssignWith.assoc_type => constants.%DivAssignWith.assoc_type.1f3
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.550
// CHECK:STDOUT:   %DivAssignWith.WithSelf.Op.type => constants.%DivAssignWith.WithSelf.Op.type.ca3
// CHECK:STDOUT:   %.loc245_5.2 => constants.%.9aa
// CHECK:STDOUT:   %DivAssignWith.lookup_impl_witness => constants.%custom_witness.8bd5d4.1
// CHECK:STDOUT:   %impl.elem0.loc245_5.2 => constants.%Int32.Op.7ce432.16
// CHECK:STDOUT:   %specific_impl_fn.loc245_5.2 => constants.%Int32.Op.7ce432.16
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ModAssignWith.loc249(constants.%Int32, constants.%ModAssignWith.facet.0d2) {
// CHECK:STDOUT:   %U.loc249_19.1 => constants.%Int32
// CHECK:STDOUT:   %ModAssignWith.type.loc249_52.1 => constants.%ModAssignWith.type.9a3
// CHECK:STDOUT:   %T.loc249_29.1 => constants.%ModAssignWith.facet.0d2
// CHECK:STDOUT:   %pattern_type.loc249_29 => constants.%pattern_type.a2a
// CHECK:STDOUT:   %T.as_type.loc249_62.1 => constants.%Int32
// CHECK:STDOUT:   %pattern_type.loc249_60 => constants.%pattern_type.4b8
// CHECK:STDOUT:   %pattern_type.loc249_66 => constants.%pattern_type.4b8
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc249_55 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc249_66 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc251 => constants.%complete_type.0bd
// CHECK:STDOUT:   %ModAssignWith.assoc_type => constants.%ModAssignWith.assoc_type.11d
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.5df
// CHECK:STDOUT:   %ModAssignWith.WithSelf.Op.type => constants.%ModAssignWith.WithSelf.Op.type.39d
// CHECK:STDOUT:   %.loc251_5.2 => constants.%.35d
// CHECK:STDOUT:   %ModAssignWith.lookup_impl_witness => constants.%custom_witness.8c8a80.1
// CHECK:STDOUT:   %impl.elem0.loc251_5.2 => constants.%Int32.Op.7ce432.17
// CHECK:STDOUT:   %specific_impl_fn.loc251_5.2 => constants.%Int32.Op.7ce432.17
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AddAssignWith.loc225(constants.%Int32, constants.%AddAssignWith.facet.2a2) {
// CHECK:STDOUT:   %U.loc225_19.1 => constants.%Int32
// CHECK:STDOUT:   %AddAssignWith.type.loc225_52.1 => constants.%AddAssignWith.type.eee
// CHECK:STDOUT:   %T.loc225_29.1 => constants.%AddAssignWith.facet.2a2
// CHECK:STDOUT:   %pattern_type.loc225_29 => constants.%pattern_type.e31
// CHECK:STDOUT:   %T.as_type.loc225_62.1 => constants.%Int64
// CHECK:STDOUT:   %pattern_type.loc225_60 => constants.%pattern_type.c76
// CHECK:STDOUT:   %pattern_type.loc225_66 => constants.%pattern_type.4b8
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc225_55 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc225_66 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc227 => constants.%complete_type.987
// CHECK:STDOUT:   %AddAssignWith.assoc_type => constants.%AddAssignWith.assoc_type.7f0
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.b73
// CHECK:STDOUT:   %AddAssignWith.WithSelf.Op.type => constants.%AddAssignWith.WithSelf.Op.type.5b1
// CHECK:STDOUT:   %.loc227_5.2 => constants.%.632
// CHECK:STDOUT:   %AddAssignWith.lookup_impl_witness => constants.%custom_witness.2be94e.2
// CHECK:STDOUT:   %impl.elem0.loc227_5.2 => constants.%Int64.Op.7ce432.2
// CHECK:STDOUT:   %specific_impl_fn.loc227_5.2 => constants.%Int64.Op.7ce432.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @SubAssignWith.loc231(constants.%Int32, constants.%SubAssignWith.facet.839) {
// CHECK:STDOUT:   %U.loc231_19.1 => constants.%Int32
// CHECK:STDOUT:   %SubAssignWith.type.loc231_52.1 => constants.%SubAssignWith.type.eed
// CHECK:STDOUT:   %T.loc231_29.1 => constants.%SubAssignWith.facet.839
// CHECK:STDOUT:   %pattern_type.loc231_29 => constants.%pattern_type.1aa
// CHECK:STDOUT:   %T.as_type.loc231_62.1 => constants.%Int64
// CHECK:STDOUT:   %pattern_type.loc231_60 => constants.%pattern_type.c76
// CHECK:STDOUT:   %pattern_type.loc231_66 => constants.%pattern_type.4b8
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc231_55 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc231_66 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc233 => constants.%complete_type.cb4
// CHECK:STDOUT:   %SubAssignWith.assoc_type => constants.%SubAssignWith.assoc_type.0dd
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.d09
// CHECK:STDOUT:   %SubAssignWith.WithSelf.Op.type => constants.%SubAssignWith.WithSelf.Op.type.9fa
// CHECK:STDOUT:   %.loc233_5.2 => constants.%.de8
// CHECK:STDOUT:   %SubAssignWith.lookup_impl_witness => constants.%custom_witness.3ff7e3.2
// CHECK:STDOUT:   %impl.elem0.loc233_5.2 => constants.%Int64.Op.7ce432.4
// CHECK:STDOUT:   %specific_impl_fn.loc233_5.2 => constants.%Int64.Op.7ce432.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @MulAssignWith.loc237(constants.%Int32, constants.%MulAssignWith.facet.19a) {
// CHECK:STDOUT:   %U.loc237_19.1 => constants.%Int32
// CHECK:STDOUT:   %MulAssignWith.type.loc237_52.1 => constants.%MulAssignWith.type.8e6
// CHECK:STDOUT:   %T.loc237_29.1 => constants.%MulAssignWith.facet.19a
// CHECK:STDOUT:   %pattern_type.loc237_29 => constants.%pattern_type.a97
// CHECK:STDOUT:   %T.as_type.loc237_62.1 => constants.%Int64
// CHECK:STDOUT:   %pattern_type.loc237_60 => constants.%pattern_type.c76
// CHECK:STDOUT:   %pattern_type.loc237_66 => constants.%pattern_type.4b8
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc237_55 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc237_66 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc239 => constants.%complete_type.5a8
// CHECK:STDOUT:   %MulAssignWith.assoc_type => constants.%MulAssignWith.assoc_type.4c9
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.4fa
// CHECK:STDOUT:   %MulAssignWith.WithSelf.Op.type => constants.%MulAssignWith.WithSelf.Op.type.2e7
// CHECK:STDOUT:   %.loc239_5.2 => constants.%.1d5
// CHECK:STDOUT:   %MulAssignWith.lookup_impl_witness => constants.%custom_witness.a36f66.2
// CHECK:STDOUT:   %impl.elem0.loc239_5.2 => constants.%Int64.Op.7ce432.5
// CHECK:STDOUT:   %specific_impl_fn.loc239_5.2 => constants.%Int64.Op.7ce432.5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @DivAssignWith.loc243(constants.%Int32, constants.%DivAssignWith.facet.d95) {
// CHECK:STDOUT:   %U.loc243_19.1 => constants.%Int32
// CHECK:STDOUT:   %DivAssignWith.type.loc243_52.1 => constants.%DivAssignWith.type.f58
// CHECK:STDOUT:   %T.loc243_29.1 => constants.%DivAssignWith.facet.d95
// CHECK:STDOUT:   %pattern_type.loc243_29 => constants.%pattern_type.2fa
// CHECK:STDOUT:   %T.as_type.loc243_62.1 => constants.%Int64
// CHECK:STDOUT:   %pattern_type.loc243_60 => constants.%pattern_type.c76
// CHECK:STDOUT:   %pattern_type.loc243_66 => constants.%pattern_type.4b8
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc243_55 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc243_66 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc245 => constants.%complete_type.3ca
// CHECK:STDOUT:   %DivAssignWith.assoc_type => constants.%DivAssignWith.assoc_type.1f3
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.550
// CHECK:STDOUT:   %DivAssignWith.WithSelf.Op.type => constants.%DivAssignWith.WithSelf.Op.type.354
// CHECK:STDOUT:   %.loc245_5.2 => constants.%.f9d
// CHECK:STDOUT:   %DivAssignWith.lookup_impl_witness => constants.%custom_witness.8bd5d4.2
// CHECK:STDOUT:   %impl.elem0.loc245_5.2 => constants.%Int64.Op.7ce432.6
// CHECK:STDOUT:   %specific_impl_fn.loc245_5.2 => constants.%Int64.Op.7ce432.6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ModAssignWith.loc249(constants.%Int32, constants.%ModAssignWith.facet.a8a) {
// CHECK:STDOUT:   %U.loc249_19.1 => constants.%Int32
// CHECK:STDOUT:   %ModAssignWith.type.loc249_52.1 => constants.%ModAssignWith.type.9a3
// CHECK:STDOUT:   %T.loc249_29.1 => constants.%ModAssignWith.facet.a8a
// CHECK:STDOUT:   %pattern_type.loc249_29 => constants.%pattern_type.a2a
// CHECK:STDOUT:   %T.as_type.loc249_62.1 => constants.%Int64
// CHECK:STDOUT:   %pattern_type.loc249_60 => constants.%pattern_type.c76
// CHECK:STDOUT:   %pattern_type.loc249_66 => constants.%pattern_type.4b8
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc249_55 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc249_66 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc251 => constants.%complete_type.0bd
// CHECK:STDOUT:   %ModAssignWith.assoc_type => constants.%ModAssignWith.assoc_type.11d
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.5df
// CHECK:STDOUT:   %ModAssignWith.WithSelf.Op.type => constants.%ModAssignWith.WithSelf.Op.type.569
// CHECK:STDOUT:   %.loc251_5.2 => constants.%.fbc
// CHECK:STDOUT:   %ModAssignWith.lookup_impl_witness => constants.%custom_witness.8c8a80.2
// CHECK:STDOUT:   %impl.elem0.loc251_5.2 => constants.%Int64.Op.7ce432.7
// CHECK:STDOUT:   %specific_impl_fn.loc251_5.2 => constants.%Int64.Op.7ce432.7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestInc(constants.%T.3d6) {
// CHECK:STDOUT:   %T.loc278_13.1 => constants.%T.3d6
// CHECK:STDOUT:   %T.as_type.loc278_33.1 => constants.%T.as_type.ec5
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.657b3b.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestDec(constants.%T.cee) {
// CHECK:STDOUT:   %T.loc284_13.1 => constants.%T.cee
// CHECK:STDOUT:   %T.as_type.loc284_33.1 => constants.%T.as_type.b80
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.a28bf9.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestNegate(constants.%Result, constants.%T.ec8) {
// CHECK:STDOUT:   %Result.loc292_9.1 => constants.%Result
// CHECK:STDOUT:   %Result.as_type.loc293_50.1 => constants.%Result.as_type
// CHECK:STDOUT:   %facet_type => constants.%facet_type.5a6
// CHECK:STDOUT:   %T.loc293_4.1 => constants.%T.ec8
// CHECK:STDOUT:   %pattern_type.loc293 => constants.%pattern_type.d0c
// CHECK:STDOUT:   %pattern_type.loc294_11 => constants.%pattern_type.96123d.2
// CHECK:STDOUT:   %T.as_type.loc294_24.1 => constants.%T.as_type.55b
// CHECK:STDOUT:   %pattern_type.loc294_22 => constants.%pattern_type.5a7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestInc(constants.%Inc.facet.353) {
// CHECK:STDOUT:   %T.loc278_13.1 => constants.%Inc.facet.353
// CHECK:STDOUT:   %T.as_type.loc278_33.1 => constants.%Int32
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.4b8
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.357
// CHECK:STDOUT:   %Inc.WithSelf.Op.type => constants.%Inc.WithSelf.Op.type.ab7
// CHECK:STDOUT:   %.loc280 => constants.%.f5b
// CHECK:STDOUT:   %Inc.lookup_impl_witness => constants.%custom_witness.696
// CHECK:STDOUT:   %impl.elem0.loc280_3.2 => constants.%Int32.Op.7ce432.18
// CHECK:STDOUT:   %specific_impl_fn.loc280_3.2 => constants.%Int32.Op.7ce432.18
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestDec(constants.%Dec.facet.b75) {
// CHECK:STDOUT:   %T.loc284_13.1 => constants.%Dec.facet.b75
// CHECK:STDOUT:   %T.as_type.loc284_33.1 => constants.%Int32
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.4b8
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.357
// CHECK:STDOUT:   %Dec.WithSelf.Op.type => constants.%Dec.WithSelf.Op.type.b72
// CHECK:STDOUT:   %.loc286 => constants.%.2d9
// CHECK:STDOUT:   %Dec.lookup_impl_witness => constants.%custom_witness.ff1
// CHECK:STDOUT:   %impl.elem0.loc286_3.2 => constants.%Int32.Op.7ce432.19
// CHECK:STDOUT:   %specific_impl_fn.loc286_3.2 => constants.%Int32.Op.7ce432.19
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestNegate(constants.%Destroy.facet.b0d, constants.%facet_value.9ca) {
// CHECK:STDOUT:   %Result.loc292_9.1 => constants.%Destroy.facet.b0d
// CHECK:STDOUT:   %Result.as_type.loc293_50.1 => constants.%Int64
// CHECK:STDOUT:   %facet_type => constants.%facet_type.064
// CHECK:STDOUT:   %T.loc293_4.1 => constants.%facet_value.9ca
// CHECK:STDOUT:   %pattern_type.loc293 => constants.%pattern_type.328
// CHECK:STDOUT:   %pattern_type.loc294_11 => constants.%pattern_type.c76
// CHECK:STDOUT:   %T.as_type.loc294_24.1 => constants.%Int16
// CHECK:STDOUT:   %pattern_type.loc294_22 => constants.%pattern_type.b06
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc294_11 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc294_22 => constants.%complete_type.357
// CHECK:STDOUT:   %Negate.lookup_impl_witness => constants.%custom_witness.1a0
// CHECK:STDOUT:   %Negate.facet => constants.%Negate.facet.086
// CHECK:STDOUT:   %Negate.WithSelf.Op.type => constants.%Negate.WithSelf.Op.type.dff
// CHECK:STDOUT:   %.loc296_3.3 => constants.%.db4
// CHECK:STDOUT:   %impl.elem1.loc296_3.2 => constants.%Int16.Op.7ce432.18
// CHECK:STDOUT:   %specific_impl_fn.loc296_3.3 => constants.%Int16.Op.7ce432.18
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.1c2
// CHECK:STDOUT:   %.loc296_3.4 => constants.%.4de
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.ecd0a5.3
// CHECK:STDOUT:   %impl.elem0.loc296_3.2 => constants.%Int64.Op.7ce432.1
// CHECK:STDOUT:   %specific_impl_fn.loc296_3.4 => constants.%Int64.Op.7ce432.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestNegate(constants.%Destroy.facet.184, constants.%facet_value.620) {
// CHECK:STDOUT:   %Result.loc292_9.1 => constants.%Destroy.facet.184
// CHECK:STDOUT:   %Result.as_type.loc293_50.1 => constants.%Int32
// CHECK:STDOUT:   %facet_type => constants.%facet_type.ff4
// CHECK:STDOUT:   %T.loc293_4.1 => constants.%facet_value.620
// CHECK:STDOUT:   %pattern_type.loc293 => constants.%pattern_type.2af
// CHECK:STDOUT:   %pattern_type.loc294_11 => constants.%pattern_type.4b8
// CHECK:STDOUT:   %T.as_type.loc294_24.1 => constants.%Int32
// CHECK:STDOUT:   %pattern_type.loc294_22 => constants.%pattern_type.4b8
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc294_11 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc294_22 => constants.%complete_type.357
// CHECK:STDOUT:   %Negate.lookup_impl_witness => constants.%custom_witness.3db
// CHECK:STDOUT:   %Negate.facet => constants.%Negate.facet.e21
// CHECK:STDOUT:   %Negate.WithSelf.Op.type => constants.%Negate.WithSelf.Op.type.384
// CHECK:STDOUT:   %.loc296_3.3 => constants.%.08c
// CHECK:STDOUT:   %impl.elem1.loc296_3.2 => constants.%Int32.Op.7ce432.20
// CHECK:STDOUT:   %specific_impl_fn.loc296_3.3 => constants.%Int32.Op.7ce432.20
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.257
// CHECK:STDOUT:   %.loc296_3.4 => constants.%.d3f
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.ecd0a5.2
// CHECK:STDOUT:   %impl.elem0.loc296_3.2 => constants.%Int32.Op.7ce432.1
// CHECK:STDOUT:   %specific_impl_fn.loc296_3.4 => constants.%Int32.Op.7ce432.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
