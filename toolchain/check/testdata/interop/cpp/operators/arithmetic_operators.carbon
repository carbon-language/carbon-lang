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
// AddWith[U: type, T: Core.AddWith(U) where .Result impls Core.Destroy](x: T, y: T)
// ```
fn AddWith[
  Result: Core.Destroy,
  U: Core.Destroy,
  T: Core.Destroy & Core.AddWith(U) where .Result = Result
](unused result: Result, x: T, y: U) {
  //@dump-sem-ir-begin
  x + y;
  //@dump-sem-ir-end
}

// TODO: Refactor when `.Result impls Core.Destroy` is possible.
fn SubWith[
  Result: Core.Destroy,
  U: Core.Destroy,
  T: Core.Destroy & Core.SubWith(U) where .Result = Result
](unused result: Result, x: T, y: U) {
  //@dump-sem-ir-begin
  x - y;
  //@dump-sem-ir-end
}

// TODO: Refactor when `.Result impls Core.Destroy` is possible.
fn MulWith[
  Result: Core.Destroy,
  U: Core.Destroy,
  T: Core.Destroy & Core.MulWith(U) where .Result = Result
](unused result: Result, x: T, y: U) {
  //@dump-sem-ir-begin
  x * y;
  //@dump-sem-ir-end
}

// TODO: Refactor when `.Result impls Core.Destroy` is possible.
fn DivWith[
  Result: Core.Destroy,
  U: Core.Destroy,
  T: Core.Destroy & Core.DivWith(U) where .Result = Result
](unused result: Result, x: T, y: U) {
  //@dump-sem-ir-begin
  x / y;
  //@dump-sem-ir-end
}

// TODO: Refactor when `.Result impls Core.Destroy` is possible.
fn ModWith[
  Result: Core.Destroy,
  U: Core.Destroy,
  T: Core.Destroy & Core.ModWith(U) where .Result = Result
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

fn AddAssignWith[U: type, T: Core.AddAssignWith(U)](var x: T, y: U) {
  //@dump-sem-ir-begin
  x += y;
  //@dump-sem-ir-end
}

fn SubAssignWith[U: type, T: Core.SubAssignWith(U)](var x: T, y: U) {
  //@dump-sem-ir-begin
  x -= y;
  //@dump-sem-ir-end
}

fn MulAssignWith[U: type, T: Core.MulAssignWith(U)](var x: T, y: U) {
  //@dump-sem-ir-begin
  x *= y;
  //@dump-sem-ir-end
}

fn DivAssignWith[U: type, T: Core.DivAssignWith(U)](var x: T, y: U) {
  //@dump-sem-ir-begin
  x /= y;
  //@dump-sem-ir-end
}

fn ModAssignWith[U: type, T: Core.ModAssignWith(U)](var x: T, y: U) {
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

fn TestInc[T: Core.Inc](var x: T) {
  //@dump-sem-ir-begin
  ++x;
  //@dump-sem-ir-end
}

fn TestDec[T: Core.Dec](var x: T) {
  //@dump-sem-ir-begin
  --x;
  //@dump-sem-ir-end
}

// TODO: Refactor when `.Result impls Core.Destroy` is possible.
fn TestNegate[
  Result: Core.Destroy,
  T: Core.Destroy & Core.Negate where .Result = Result
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
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %pattern_type.9a587c.1: type = pattern_type %Destroy.type [concrete]
// CHECK:STDOUT:   %Result.patt: %pattern_type.9a587c.1 = symbolic_binding_pattern Result, 0 [symbolic]
// CHECK:STDOUT:   %Result: %Destroy.type = symbolic_binding Result, 0 [symbolic]
// CHECK:STDOUT:   %U.patt.cb8: %pattern_type.9a587c.1 = symbolic_binding_pattern U, 1 [symbolic]
// CHECK:STDOUT:   %U.e1f: %Destroy.type = symbolic_binding U, 1 [symbolic]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Other: type = symbolic_binding Other, 0 [symbolic]
// CHECK:STDOUT:   %AddWith.type.872: type = facet_type <@AddWith.1, @AddWith.1(%Other)> [symbolic]
// CHECK:STDOUT:   %pattern_type.9a587c.2: type = pattern_type type [concrete]
// CHECK:STDOUT:   %Self.958: %AddWith.type.872 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %AddWith.assoc_type.1d5: type = assoc_entity_type @AddWith.1, @AddWith.1(%Other) [symbolic]
// CHECK:STDOUT:   %AddWith.WithSelf.Op.type.70b: type = fn_type @AddWith.WithSelf.Op, @AddWith.WithSelf(%Other, %Self.958) [symbolic]
// CHECK:STDOUT:   %AddWith.WithSelf.Op.bb1: %AddWith.WithSelf.Op.type.70b = struct_value () [symbolic]
// CHECK:STDOUT:   %U.as_type.0bd: type = facet_access_type %U.e1f [symbolic]
// CHECK:STDOUT:   %AddWith.type.4f5: type = facet_type <@AddWith.1, @AddWith.1(%U.as_type.0bd)> [symbolic]
// CHECK:STDOUT:   %facet_type.4f5: type = facet_type <@AddWith.1, @AddWith.1(%U.as_type.0bd) & @Destroy> [symbolic]
// CHECK:STDOUT:   %.Self.frozen.136: %facet_type.4f5 = symbolic_binding .Self [symbolic]
// CHECK:STDOUT:   %AddWith.assoc_type.71f: type = assoc_entity_type @AddWith.1, @AddWith.1(%U.as_type.0bd) [symbolic]
// CHECK:STDOUT:   %assoc0.155: %AddWith.assoc_type.71f = assoc_entity element0, imports.%Core.import_ref.91f [symbolic]
// CHECK:STDOUT:   %assoc1.f40: %AddWith.assoc_type.71f = assoc_entity element1, imports.%Core.import_ref.8ad [symbolic]
// CHECK:STDOUT:   %require_complete.528878.1: <witness> = require_complete_type %facet_type.4f5 [symbolic]
// CHECK:STDOUT:   %.Self.frozen.as_type.034: type = facet_access_type %.Self.frozen.136 [symbolic]
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.0f3: <witness> = lookup_impl_witness %.Self.frozen.136, @AddWith.1, @AddWith.1(%U.as_type.0bd) [symbolic]
// CHECK:STDOUT:   %impl.elem0.1cd: type = impl_witness_access %AddWith.lookup_impl_witness.0f3, element0 [symbolic]
// CHECK:STDOUT:   %Result.as_type: type = facet_access_type %Result [symbolic]
// CHECK:STDOUT:   %.Self.354: %facet_type.4f5 = symbolic_binding .Self [symbolic]
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.ab5: <witness> = lookup_impl_witness %.Self.354, @AddWith.1, @AddWith.1(%U.as_type.0bd) [symbolic]
// CHECK:STDOUT:   %impl.elem0.2c2: type = impl_witness_access %AddWith.lookup_impl_witness.ab5, element0 [symbolic]
// CHECK:STDOUT:   %facet_type.c17: type = facet_type <@AddWith.1, @AddWith.1(%U.as_type.0bd) & @Destroy where %impl.elem0.2c2 = %Result.as_type> [symbolic]
// CHECK:STDOUT:   %pattern_type.fef: type = pattern_type %facet_type.c17 [symbolic]
// CHECK:STDOUT:   %T.patt.910: %pattern_type.fef = symbolic_binding_pattern T, 2 [symbolic]
// CHECK:STDOUT:   %T.21f: %facet_type.c17 = symbolic_binding T, 2 [symbolic]
// CHECK:STDOUT:   %pattern_type.86a9db.1: type = pattern_type %Result.as_type [symbolic]
// CHECK:STDOUT:   %result.param_patt.f86: %pattern_type.86a9db.1 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %result.patt.a71562.1: %pattern_type.86a9db.1 = wrapper_binding_pattern result, %result.param_patt.f86 [symbolic]
// CHECK:STDOUT:   %T.as_type.d2e: type = facet_access_type %T.21f [symbolic]
// CHECK:STDOUT:   %pattern_type.fab: type = pattern_type %T.as_type.d2e [symbolic]
// CHECK:STDOUT:   %x.param_patt.947: %pattern_type.fab = value_param_pattern [symbolic]
// CHECK:STDOUT:   %x.patt.edd: %pattern_type.fab = wrapper_binding_pattern x, %x.param_patt.947 [symbolic]
// CHECK:STDOUT:   %pattern_type.eff: type = pattern_type %U.as_type.0bd [symbolic]
// CHECK:STDOUT:   %y.param_patt.8c7: %pattern_type.eff = value_param_pattern [symbolic]
// CHECK:STDOUT:   %y.patt.b5a: %pattern_type.eff = wrapper_binding_pattern y, %y.param_patt.8c7 [symbolic]
// CHECK:STDOUT:   %require_complete.528878.2: <witness> = require_complete_type %AddWith.type.4f5 [symbolic]
// CHECK:STDOUT:   %assoc1.84a: %AddWith.assoc_type.1d5 = assoc_entity element1, imports.%Core.import_ref.e9a [symbolic]
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.206: <witness> = lookup_impl_witness %T.21f, @AddWith.1, @AddWith.1(%U.as_type.0bd) [symbolic]
// CHECK:STDOUT:   %AddWith.facet.880: %AddWith.type.4f5 = facet_value %T.as_type.d2e, (%AddWith.lookup_impl_witness.206) [symbolic]
// CHECK:STDOUT:   %AddWith.WithSelf.Op.type.c00: type = fn_type @AddWith.WithSelf.Op, @AddWith.WithSelf(%U.as_type.0bd, %AddWith.facet.880) [symbolic]
// CHECK:STDOUT:   %.a11: type = fn_type_with_self_type %AddWith.WithSelf.Op.type.c00, %AddWith.facet.880 [symbolic]
// CHECK:STDOUT:   %impl.elem1.fe2: %.a11 = impl_witness_access %AddWith.lookup_impl_witness.206, element1 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.896: <specific function> = specific_impl_function %impl.elem1.fe2, @AddWith.WithSelf.Op(%U.as_type.0bd, %AddWith.facet.880) [symbolic]
// CHECK:STDOUT:   %SelfDestruct.type: type = fn_type @SelfDestruct [concrete]
// CHECK:STDOUT:   %SelfDestruct: %SelfDestruct.type = struct_value () [concrete]
// CHECK:STDOUT:   %SubWith.type.73b: type = facet_type <@SubWith.1, @SubWith.1(%Other)> [symbolic]
// CHECK:STDOUT:   %Self.18a: %SubWith.type.73b = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %SubWith.assoc_type.300: type = assoc_entity_type @SubWith.1, @SubWith.1(%Other) [symbolic]
// CHECK:STDOUT:   %SubWith.WithSelf.Op.type.028: type = fn_type @SubWith.WithSelf.Op, @SubWith.WithSelf(%Other, %Self.18a) [symbolic]
// CHECK:STDOUT:   %SubWith.WithSelf.Op.e84: %SubWith.WithSelf.Op.type.028 = struct_value () [symbolic]
// CHECK:STDOUT:   %SubWith.type.769: type = facet_type <@SubWith.1, @SubWith.1(%U.as_type.0bd)> [symbolic]
// CHECK:STDOUT:   %facet_type.769: type = facet_type <@SubWith.1, @SubWith.1(%U.as_type.0bd) & @Destroy> [symbolic]
// CHECK:STDOUT:   %.Self.frozen.b13: %facet_type.769 = symbolic_binding .Self [symbolic]
// CHECK:STDOUT:   %SubWith.assoc_type.5f0: type = assoc_entity_type @SubWith.1, @SubWith.1(%U.as_type.0bd) [symbolic]
// CHECK:STDOUT:   %assoc0.a5d: %SubWith.assoc_type.5f0 = assoc_entity element0, imports.%Core.import_ref.190 [symbolic]
// CHECK:STDOUT:   %assoc1.a86: %SubWith.assoc_type.5f0 = assoc_entity element1, imports.%Core.import_ref.2f4 [symbolic]
// CHECK:STDOUT:   %require_complete.49f499.1: <witness> = require_complete_type %facet_type.769 [symbolic]
// CHECK:STDOUT:   %.Self.frozen.as_type.d1e: type = facet_access_type %.Self.frozen.b13 [symbolic]
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.fe8: <witness> = lookup_impl_witness %.Self.frozen.b13, @SubWith.1, @SubWith.1(%U.as_type.0bd) [symbolic]
// CHECK:STDOUT:   %impl.elem0.7da: type = impl_witness_access %SubWith.lookup_impl_witness.fe8, element0 [symbolic]
// CHECK:STDOUT:   %.Self.4de: %facet_type.769 = symbolic_binding .Self [symbolic]
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.354: <witness> = lookup_impl_witness %.Self.4de, @SubWith.1, @SubWith.1(%U.as_type.0bd) [symbolic]
// CHECK:STDOUT:   %impl.elem0.3dc: type = impl_witness_access %SubWith.lookup_impl_witness.354, element0 [symbolic]
// CHECK:STDOUT:   %facet_type.0a9: type = facet_type <@SubWith.1, @SubWith.1(%U.as_type.0bd) & @Destroy where %impl.elem0.3dc = %Result.as_type> [symbolic]
// CHECK:STDOUT:   %pattern_type.a14: type = pattern_type %facet_type.0a9 [symbolic]
// CHECK:STDOUT:   %T.patt.51b: %pattern_type.a14 = symbolic_binding_pattern T, 2 [symbolic]
// CHECK:STDOUT:   %T.3c5: %facet_type.0a9 = symbolic_binding T, 2 [symbolic]
// CHECK:STDOUT:   %T.as_type.4b7: type = facet_access_type %T.3c5 [symbolic]
// CHECK:STDOUT:   %pattern_type.ca3: type = pattern_type %T.as_type.4b7 [symbolic]
// CHECK:STDOUT:   %x.param_patt.10b: %pattern_type.ca3 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %x.patt.7cd: %pattern_type.ca3 = wrapper_binding_pattern x, %x.param_patt.10b [symbolic]
// CHECK:STDOUT:   %require_complete.49f499.2: <witness> = require_complete_type %SubWith.type.769 [symbolic]
// CHECK:STDOUT:   %assoc1.b58: %SubWith.assoc_type.300 = assoc_entity element1, imports.%Core.import_ref.124 [symbolic]
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.725: <witness> = lookup_impl_witness %T.3c5, @SubWith.1, @SubWith.1(%U.as_type.0bd) [symbolic]
// CHECK:STDOUT:   %SubWith.facet.910: %SubWith.type.769 = facet_value %T.as_type.4b7, (%SubWith.lookup_impl_witness.725) [symbolic]
// CHECK:STDOUT:   %SubWith.WithSelf.Op.type.772: type = fn_type @SubWith.WithSelf.Op, @SubWith.WithSelf(%U.as_type.0bd, %SubWith.facet.910) [symbolic]
// CHECK:STDOUT:   %.47a6: type = fn_type_with_self_type %SubWith.WithSelf.Op.type.772, %SubWith.facet.910 [symbolic]
// CHECK:STDOUT:   %impl.elem1.415: %.47a6 = impl_witness_access %SubWith.lookup_impl_witness.725, element1 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.a72: <specific function> = specific_impl_function %impl.elem1.415, @SubWith.WithSelf.Op(%U.as_type.0bd, %SubWith.facet.910) [symbolic]
// CHECK:STDOUT:   %MulWith.type.616: type = facet_type <@MulWith.1, @MulWith.1(%Other)> [symbolic]
// CHECK:STDOUT:   %Self.742: %MulWith.type.616 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %MulWith.assoc_type.c81: type = assoc_entity_type @MulWith.1, @MulWith.1(%Other) [symbolic]
// CHECK:STDOUT:   %MulWith.WithSelf.Op.type.dad: type = fn_type @MulWith.WithSelf.Op, @MulWith.WithSelf(%Other, %Self.742) [symbolic]
// CHECK:STDOUT:   %MulWith.WithSelf.Op.507: %MulWith.WithSelf.Op.type.dad = struct_value () [symbolic]
// CHECK:STDOUT:   %MulWith.type.b0b: type = facet_type <@MulWith.1, @MulWith.1(%U.as_type.0bd)> [symbolic]
// CHECK:STDOUT:   %facet_type.b0b: type = facet_type <@MulWith.1, @MulWith.1(%U.as_type.0bd) & @Destroy> [symbolic]
// CHECK:STDOUT:   %.Self.frozen.a62: %facet_type.b0b = symbolic_binding .Self [symbolic]
// CHECK:STDOUT:   %MulWith.assoc_type.2e5: type = assoc_entity_type @MulWith.1, @MulWith.1(%U.as_type.0bd) [symbolic]
// CHECK:STDOUT:   %assoc0.e89: %MulWith.assoc_type.2e5 = assoc_entity element0, imports.%Core.import_ref.59c [symbolic]
// CHECK:STDOUT:   %assoc1.4eb: %MulWith.assoc_type.2e5 = assoc_entity element1, imports.%Core.import_ref.fc2 [symbolic]
// CHECK:STDOUT:   %require_complete.8389d3.1: <witness> = require_complete_type %facet_type.b0b [symbolic]
// CHECK:STDOUT:   %.Self.frozen.as_type.bd3: type = facet_access_type %.Self.frozen.a62 [symbolic]
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.136: <witness> = lookup_impl_witness %.Self.frozen.a62, @MulWith.1, @MulWith.1(%U.as_type.0bd) [symbolic]
// CHECK:STDOUT:   %impl.elem0.16c: type = impl_witness_access %MulWith.lookup_impl_witness.136, element0 [symbolic]
// CHECK:STDOUT:   %.Self.a44: %facet_type.b0b = symbolic_binding .Self [symbolic]
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.ced: <witness> = lookup_impl_witness %.Self.a44, @MulWith.1, @MulWith.1(%U.as_type.0bd) [symbolic]
// CHECK:STDOUT:   %impl.elem0.0e8: type = impl_witness_access %MulWith.lookup_impl_witness.ced, element0 [symbolic]
// CHECK:STDOUT:   %facet_type.a5f: type = facet_type <@MulWith.1, @MulWith.1(%U.as_type.0bd) & @Destroy where %impl.elem0.0e8 = %Result.as_type> [symbolic]
// CHECK:STDOUT:   %pattern_type.3ab: type = pattern_type %facet_type.a5f [symbolic]
// CHECK:STDOUT:   %T.patt.170: %pattern_type.3ab = symbolic_binding_pattern T, 2 [symbolic]
// CHECK:STDOUT:   %T.d03: %facet_type.a5f = symbolic_binding T, 2 [symbolic]
// CHECK:STDOUT:   %T.as_type.639: type = facet_access_type %T.d03 [symbolic]
// CHECK:STDOUT:   %pattern_type.d7a: type = pattern_type %T.as_type.639 [symbolic]
// CHECK:STDOUT:   %x.param_patt.1c9: %pattern_type.d7a = value_param_pattern [symbolic]
// CHECK:STDOUT:   %x.patt.080: %pattern_type.d7a = wrapper_binding_pattern x, %x.param_patt.1c9 [symbolic]
// CHECK:STDOUT:   %require_complete.8389d3.2: <witness> = require_complete_type %MulWith.type.b0b [symbolic]
// CHECK:STDOUT:   %assoc1.d9e: %MulWith.assoc_type.c81 = assoc_entity element1, imports.%Core.import_ref.29d [symbolic]
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.50b: <witness> = lookup_impl_witness %T.d03, @MulWith.1, @MulWith.1(%U.as_type.0bd) [symbolic]
// CHECK:STDOUT:   %MulWith.facet.756: %MulWith.type.b0b = facet_value %T.as_type.639, (%MulWith.lookup_impl_witness.50b) [symbolic]
// CHECK:STDOUT:   %MulWith.WithSelf.Op.type.689: type = fn_type @MulWith.WithSelf.Op, @MulWith.WithSelf(%U.as_type.0bd, %MulWith.facet.756) [symbolic]
// CHECK:STDOUT:   %.7f0: type = fn_type_with_self_type %MulWith.WithSelf.Op.type.689, %MulWith.facet.756 [symbolic]
// CHECK:STDOUT:   %impl.elem1.334: %.7f0 = impl_witness_access %MulWith.lookup_impl_witness.50b, element1 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.2bd: <specific function> = specific_impl_function %impl.elem1.334, @MulWith.WithSelf.Op(%U.as_type.0bd, %MulWith.facet.756) [symbolic]
// CHECK:STDOUT:   %DivWith.type.472: type = facet_type <@DivWith.1, @DivWith.1(%Other)> [symbolic]
// CHECK:STDOUT:   %Self.5b3: %DivWith.type.472 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %DivWith.assoc_type.e7d: type = assoc_entity_type @DivWith.1, @DivWith.1(%Other) [symbolic]
// CHECK:STDOUT:   %DivWith.WithSelf.Op.type.207: type = fn_type @DivWith.WithSelf.Op, @DivWith.WithSelf(%Other, %Self.5b3) [symbolic]
// CHECK:STDOUT:   %DivWith.WithSelf.Op.4f1: %DivWith.WithSelf.Op.type.207 = struct_value () [symbolic]
// CHECK:STDOUT:   %DivWith.type.71d: type = facet_type <@DivWith.1, @DivWith.1(%U.as_type.0bd)> [symbolic]
// CHECK:STDOUT:   %facet_type.71d: type = facet_type <@DivWith.1, @DivWith.1(%U.as_type.0bd) & @Destroy> [symbolic]
// CHECK:STDOUT:   %.Self.frozen.575: %facet_type.71d = symbolic_binding .Self [symbolic]
// CHECK:STDOUT:   %DivWith.assoc_type.e65: type = assoc_entity_type @DivWith.1, @DivWith.1(%U.as_type.0bd) [symbolic]
// CHECK:STDOUT:   %assoc0.48a: %DivWith.assoc_type.e65 = assoc_entity element0, imports.%Core.import_ref.dee [symbolic]
// CHECK:STDOUT:   %assoc1.ed1: %DivWith.assoc_type.e65 = assoc_entity element1, imports.%Core.import_ref.f2f [symbolic]
// CHECK:STDOUT:   %require_complete.c4013f.1: <witness> = require_complete_type %facet_type.71d [symbolic]
// CHECK:STDOUT:   %.Self.frozen.as_type.a77: type = facet_access_type %.Self.frozen.575 [symbolic]
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.d5d: <witness> = lookup_impl_witness %.Self.frozen.575, @DivWith.1, @DivWith.1(%U.as_type.0bd) [symbolic]
// CHECK:STDOUT:   %impl.elem0.e77: type = impl_witness_access %DivWith.lookup_impl_witness.d5d, element0 [symbolic]
// CHECK:STDOUT:   %.Self.c58: %facet_type.71d = symbolic_binding .Self [symbolic]
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.f1d: <witness> = lookup_impl_witness %.Self.c58, @DivWith.1, @DivWith.1(%U.as_type.0bd) [symbolic]
// CHECK:STDOUT:   %impl.elem0.af2: type = impl_witness_access %DivWith.lookup_impl_witness.f1d, element0 [symbolic]
// CHECK:STDOUT:   %facet_type.351: type = facet_type <@DivWith.1, @DivWith.1(%U.as_type.0bd) & @Destroy where %impl.elem0.af2 = %Result.as_type> [symbolic]
// CHECK:STDOUT:   %pattern_type.ae7: type = pattern_type %facet_type.351 [symbolic]
// CHECK:STDOUT:   %T.patt.32b: %pattern_type.ae7 = symbolic_binding_pattern T, 2 [symbolic]
// CHECK:STDOUT:   %T.942: %facet_type.351 = symbolic_binding T, 2 [symbolic]
// CHECK:STDOUT:   %T.as_type.6cf: type = facet_access_type %T.942 [symbolic]
// CHECK:STDOUT:   %pattern_type.c8a: type = pattern_type %T.as_type.6cf [symbolic]
// CHECK:STDOUT:   %x.param_patt.795: %pattern_type.c8a = value_param_pattern [symbolic]
// CHECK:STDOUT:   %x.patt.818: %pattern_type.c8a = wrapper_binding_pattern x, %x.param_patt.795 [symbolic]
// CHECK:STDOUT:   %require_complete.c4013f.2: <witness> = require_complete_type %DivWith.type.71d [symbolic]
// CHECK:STDOUT:   %assoc1.2b8: %DivWith.assoc_type.e7d = assoc_entity element1, imports.%Core.import_ref.94d [symbolic]
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.e1e: <witness> = lookup_impl_witness %T.942, @DivWith.1, @DivWith.1(%U.as_type.0bd) [symbolic]
// CHECK:STDOUT:   %DivWith.facet.538: %DivWith.type.71d = facet_value %T.as_type.6cf, (%DivWith.lookup_impl_witness.e1e) [symbolic]
// CHECK:STDOUT:   %DivWith.WithSelf.Op.type.663: type = fn_type @DivWith.WithSelf.Op, @DivWith.WithSelf(%U.as_type.0bd, %DivWith.facet.538) [symbolic]
// CHECK:STDOUT:   %.0f9: type = fn_type_with_self_type %DivWith.WithSelf.Op.type.663, %DivWith.facet.538 [symbolic]
// CHECK:STDOUT:   %impl.elem1.44e: %.0f9 = impl_witness_access %DivWith.lookup_impl_witness.e1e, element1 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.ea9: <specific function> = specific_impl_function %impl.elem1.44e, @DivWith.WithSelf.Op(%U.as_type.0bd, %DivWith.facet.538) [symbolic]
// CHECK:STDOUT:   %ModWith.type.11e: type = facet_type <@ModWith.1, @ModWith.1(%Other)> [symbolic]
// CHECK:STDOUT:   %Self.c18: %ModWith.type.11e = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %ModWith.assoc_type.1f1: type = assoc_entity_type @ModWith.1, @ModWith.1(%Other) [symbolic]
// CHECK:STDOUT:   %ModWith.WithSelf.Op.type.04e: type = fn_type @ModWith.WithSelf.Op, @ModWith.WithSelf(%Other, %Self.c18) [symbolic]
// CHECK:STDOUT:   %ModWith.WithSelf.Op.6fb: %ModWith.WithSelf.Op.type.04e = struct_value () [symbolic]
// CHECK:STDOUT:   %ModWith.type.68a: type = facet_type <@ModWith.1, @ModWith.1(%U.as_type.0bd)> [symbolic]
// CHECK:STDOUT:   %facet_type.68a: type = facet_type <@ModWith.1, @ModWith.1(%U.as_type.0bd) & @Destroy> [symbolic]
// CHECK:STDOUT:   %.Self.frozen.e0d: %facet_type.68a = symbolic_binding .Self [symbolic]
// CHECK:STDOUT:   %ModWith.assoc_type.9c3: type = assoc_entity_type @ModWith.1, @ModWith.1(%U.as_type.0bd) [symbolic]
// CHECK:STDOUT:   %assoc0.cb9: %ModWith.assoc_type.9c3 = assoc_entity element0, imports.%Core.import_ref.61a [symbolic]
// CHECK:STDOUT:   %assoc1.3d6: %ModWith.assoc_type.9c3 = assoc_entity element1, imports.%Core.import_ref.9f8f [symbolic]
// CHECK:STDOUT:   %require_complete.312803.1: <witness> = require_complete_type %facet_type.68a [symbolic]
// CHECK:STDOUT:   %.Self.frozen.as_type.01e: type = facet_access_type %.Self.frozen.e0d [symbolic]
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.4ec: <witness> = lookup_impl_witness %.Self.frozen.e0d, @ModWith.1, @ModWith.1(%U.as_type.0bd) [symbolic]
// CHECK:STDOUT:   %impl.elem0.671: type = impl_witness_access %ModWith.lookup_impl_witness.4ec, element0 [symbolic]
// CHECK:STDOUT:   %.Self.2c1: %facet_type.68a = symbolic_binding .Self [symbolic]
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.930: <witness> = lookup_impl_witness %.Self.2c1, @ModWith.1, @ModWith.1(%U.as_type.0bd) [symbolic]
// CHECK:STDOUT:   %impl.elem0.e6d: type = impl_witness_access %ModWith.lookup_impl_witness.930, element0 [symbolic]
// CHECK:STDOUT:   %facet_type.938: type = facet_type <@ModWith.1, @ModWith.1(%U.as_type.0bd) & @Destroy where %impl.elem0.e6d = %Result.as_type> [symbolic]
// CHECK:STDOUT:   %pattern_type.b1e: type = pattern_type %facet_type.938 [symbolic]
// CHECK:STDOUT:   %T.patt.721: %pattern_type.b1e = symbolic_binding_pattern T, 2 [symbolic]
// CHECK:STDOUT:   %T.fdb: %facet_type.938 = symbolic_binding T, 2 [symbolic]
// CHECK:STDOUT:   %T.as_type.73e: type = facet_access_type %T.fdb [symbolic]
// CHECK:STDOUT:   %pattern_type.70e: type = pattern_type %T.as_type.73e [symbolic]
// CHECK:STDOUT:   %x.param_patt.375: %pattern_type.70e = value_param_pattern [symbolic]
// CHECK:STDOUT:   %x.patt.f07: %pattern_type.70e = wrapper_binding_pattern x, %x.param_patt.375 [symbolic]
// CHECK:STDOUT:   %require_complete.312803.2: <witness> = require_complete_type %ModWith.type.68a [symbolic]
// CHECK:STDOUT:   %assoc1.f4b: %ModWith.assoc_type.1f1 = assoc_entity element1, imports.%Core.import_ref.be00 [symbolic]
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.c4b: <witness> = lookup_impl_witness %T.fdb, @ModWith.1, @ModWith.1(%U.as_type.0bd) [symbolic]
// CHECK:STDOUT:   %ModWith.facet.297: %ModWith.type.68a = facet_value %T.as_type.73e, (%ModWith.lookup_impl_witness.c4b) [symbolic]
// CHECK:STDOUT:   %ModWith.WithSelf.Op.type.639: type = fn_type @ModWith.WithSelf.Op, @ModWith.WithSelf(%U.as_type.0bd, %ModWith.facet.297) [symbolic]
// CHECK:STDOUT:   %.100: type = fn_type_with_self_type %ModWith.WithSelf.Op.type.639, %ModWith.facet.297 [symbolic]
// CHECK:STDOUT:   %impl.elem1.b51: %.100 = impl_witness_access %ModWith.lookup_impl_witness.c4b, element1 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.6e9: <specific function> = specific_impl_function %impl.elem1.b51, @ModWith.WithSelf.Op(%U.as_type.0bd, %ModWith.facet.297) [symbolic]
// CHECK:STDOUT:   %Int16: type = class_type @Int16 [concrete]
// CHECK:STDOUT:   %pattern_type.785: type = pattern_type %Int16 [concrete]
// CHECK:STDOUT:   %Int32: type = class_type @Int32 [concrete]
// CHECK:STDOUT:   %pattern_type.8b4: type = pattern_type %Int32 [concrete]
// CHECK:STDOUT:   %Int64: type = class_type @Int64 [concrete]
// CHECK:STDOUT:   %pattern_type.4aa: type = pattern_type %Int64 [concrete]
// CHECK:STDOUT:   %int_16: Core.IntLiteral = int_value 16 [concrete]
// CHECK:STDOUT:   %i16: type = class_type @Int, @Int(%int_16) [concrete]
// CHECK:STDOUT:   %pattern_type.9bc: type = pattern_type %i16 [concrete]
// CHECK:STDOUT:   %x.param_patt.77a: %pattern_type.9bc = value_param_pattern [concrete]
// CHECK:STDOUT:   %x.patt.9b0: %pattern_type.9bc = wrapper_binding_pattern x, %x.param_patt.77a [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %i16.builtin: type = int_type signed, %int_16 [concrete]
// CHECK:STDOUT:   %complete_type.2da: <witness> = complete_type_witness %i16.builtin [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.1: type = fn_type @SubobjectDestroy.WithSelf.Op.loc176_34.1 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.1: %SubobjectDestroy.WithSelf.Op.type.6870fd.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.1: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.1), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.78d: %SubobjectDestroy.type = facet_value %Int16, (%custom_witness.b80ef6.1) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.891: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.78d) [concrete]
// CHECK:STDOUT:   %Destroy.facet.c1d: %Destroy.type = facet_value %Int16, (%DestroyImpl.impl_witness.891, %custom_witness.b80ef6.1) [concrete]
// CHECK:STDOUT:   %facet_type.e4b: type = facet_type <@AddWith.1, @AddWith.1(%Int16) & @Destroy> [concrete]
// CHECK:STDOUT:   %.Self.033: %facet_type.e4b = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.73e: <witness> = lookup_impl_witness %.Self.033, @AddWith.1, @AddWith.1(%Int16) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.08e: type = impl_witness_access %AddWith.lookup_impl_witness.73e, element0 [symbolic_self]
// CHECK:STDOUT:   %facet_type.5b8: type = facet_type <@AddWith.1, @AddWith.1(%Int16) & @Destroy where %impl.elem0.08e = %Int16> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.3: type = fn_type @SubobjectDestroy.WithSelf.Op.loc176_34.3 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.3: %SubobjectDestroy.WithSelf.Op.type.6870fd.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.3: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.3), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.c5e: %SubobjectDestroy.type = facet_value %i16, (%custom_witness.b80ef6.3) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.dff: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.c5e) [concrete]
// CHECK:STDOUT:   %AddWith.type.e4b: type = facet_type <@AddWith.1, @AddWith.1(%Int16)> [concrete]
// CHECK:STDOUT:   %AddWith.assoc_type.86e: type = assoc_entity_type @AddWith.1, @AddWith.1(%Int16) [concrete]
// CHECK:STDOUT:   %assoc0.30c: %AddWith.assoc_type.86e = assoc_entity element0, imports.%Core.import_ref.91f [concrete]
// CHECK:STDOUT:   %assoc1.9bf: %AddWith.assoc_type.86e = assoc_entity element1, imports.%Core.import_ref.8ad [concrete]
// CHECK:STDOUT:   %Op.type.442b68.1: type = fn_type @Op.1 [concrete]
// CHECK:STDOUT:   %Op.72717c.1: %Op.type.442b68.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.facet.1b4: %Destroy.type = facet_value %i16, (%DestroyImpl.impl_witness.dff, %custom_witness.b80ef6.3) [concrete]
// CHECK:STDOUT:   %custom_witness.cba: <witness> = custom_witness (%Int16, %Op.72717c.1), @AddWith.1, @AddWith.1(%Int16) [concrete]
// CHECK:STDOUT:   %facet_value.b07: %facet_type.5b8 = facet_value %i16, (%DestroyImpl.impl_witness.dff, %custom_witness.b80ef6.3, %custom_witness.cba) [concrete]
// CHECK:STDOUT:   %.Self.frozen.ea9: %facet_type.e4b = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %complete_type.f37bc8.1: <witness> = complete_type_witness %facet_type.e4b [concrete]
// CHECK:STDOUT:   %.Self.frozen.as_type.ceb: type = facet_access_type %.Self.frozen.ea9 [symbolic_self]
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.e18: <witness> = lookup_impl_witness %.Self.frozen.ea9, @AddWith.1, @AddWith.1(%Int16) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.de3: type = impl_witness_access %AddWith.lookup_impl_witness.e18, element0 [symbolic_self]
// CHECK:STDOUT:   %pattern_type.f83: type = pattern_type %facet_type.5b8 [concrete]
// CHECK:STDOUT:   %T.patt.056: %pattern_type.f83 = symbolic_binding_pattern T, 2 [symbolic]
// CHECK:STDOUT:   %result.param_patt.be6: %pattern_type.785 = value_param_pattern [concrete]
// CHECK:STDOUT:   %result.patt.a31: %pattern_type.785 = wrapper_binding_pattern result, %result.param_patt.be6 [concrete]
// CHECK:STDOUT:   %y.param_patt.418: %pattern_type.785 = value_param_pattern [concrete]
// CHECK:STDOUT:   %y.patt.890: %pattern_type.785 = wrapper_binding_pattern y, %y.param_patt.418 [concrete]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.581: <specific function> = specific_function %SelfDestruct, @SelfDestruct(%Destroy.facet.c1d) [concrete]
// CHECK:STDOUT:   %facet_type.68d: type = facet_type <@SubWith.1, @SubWith.1(%Int16) & @Destroy> [concrete]
// CHECK:STDOUT:   %.Self.a10: %facet_type.68d = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.c4d: <witness> = lookup_impl_witness %.Self.a10, @SubWith.1, @SubWith.1(%Int16) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.22a: type = impl_witness_access %SubWith.lookup_impl_witness.c4d, element0 [symbolic_self]
// CHECK:STDOUT:   %facet_type.5db: type = facet_type <@SubWith.1, @SubWith.1(%Int16) & @Destroy where %impl.elem0.22a = %Int16> [concrete]
// CHECK:STDOUT:   %SubWith.type.68d: type = facet_type <@SubWith.1, @SubWith.1(%Int16)> [concrete]
// CHECK:STDOUT:   %SubWith.assoc_type.dd6: type = assoc_entity_type @SubWith.1, @SubWith.1(%Int16) [concrete]
// CHECK:STDOUT:   %assoc0.dfb: %SubWith.assoc_type.dd6 = assoc_entity element0, imports.%Core.import_ref.190 [concrete]
// CHECK:STDOUT:   %assoc1.ada: %SubWith.assoc_type.dd6 = assoc_entity element1, imports.%Core.import_ref.2f4 [concrete]
// CHECK:STDOUT:   %Op.type.442b68.2: type = fn_type @Op.2 [concrete]
// CHECK:STDOUT:   %Op.72717c.2: %Op.type.442b68.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.354: <witness> = custom_witness (%Int16, %Op.72717c.2), @SubWith.1, @SubWith.1(%Int16) [concrete]
// CHECK:STDOUT:   %facet_value.ca4: %facet_type.5db = facet_value %i16, (%DestroyImpl.impl_witness.dff, %custom_witness.b80ef6.3, %custom_witness.354) [concrete]
// CHECK:STDOUT:   %.Self.frozen.0c9: %facet_type.68d = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %complete_type.8a2f63.1: <witness> = complete_type_witness %facet_type.68d [concrete]
// CHECK:STDOUT:   %.Self.frozen.as_type.471: type = facet_access_type %.Self.frozen.0c9 [symbolic_self]
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.379: <witness> = lookup_impl_witness %.Self.frozen.0c9, @SubWith.1, @SubWith.1(%Int16) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.e78: type = impl_witness_access %SubWith.lookup_impl_witness.379, element0 [symbolic_self]
// CHECK:STDOUT:   %pattern_type.7bf: type = pattern_type %facet_type.5db [concrete]
// CHECK:STDOUT:   %T.patt.f08: %pattern_type.7bf = symbolic_binding_pattern T, 2 [symbolic]
// CHECK:STDOUT:   %facet_type.f65: type = facet_type <@MulWith.1, @MulWith.1(%Int16) & @Destroy> [concrete]
// CHECK:STDOUT:   %.Self.1e8: %facet_type.f65 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.b4a: <witness> = lookup_impl_witness %.Self.1e8, @MulWith.1, @MulWith.1(%Int16) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.088: type = impl_witness_access %MulWith.lookup_impl_witness.b4a, element0 [symbolic_self]
// CHECK:STDOUT:   %facet_type.1c9: type = facet_type <@MulWith.1, @MulWith.1(%Int16) & @Destroy where %impl.elem0.088 = %Int16> [concrete]
// CHECK:STDOUT:   %MulWith.type.f65: type = facet_type <@MulWith.1, @MulWith.1(%Int16)> [concrete]
// CHECK:STDOUT:   %MulWith.assoc_type.4c3: type = assoc_entity_type @MulWith.1, @MulWith.1(%Int16) [concrete]
// CHECK:STDOUT:   %assoc0.6ec: %MulWith.assoc_type.4c3 = assoc_entity element0, imports.%Core.import_ref.59c [concrete]
// CHECK:STDOUT:   %assoc1.b14: %MulWith.assoc_type.4c3 = assoc_entity element1, imports.%Core.import_ref.fc2 [concrete]
// CHECK:STDOUT:   %Op.type.442b68.3: type = fn_type @Op.3 [concrete]
// CHECK:STDOUT:   %Op.72717c.3: %Op.type.442b68.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.3c3: <witness> = custom_witness (%Int16, %Op.72717c.3), @MulWith.1, @MulWith.1(%Int16) [concrete]
// CHECK:STDOUT:   %facet_value.3e5: %facet_type.1c9 = facet_value %i16, (%DestroyImpl.impl_witness.dff, %custom_witness.b80ef6.3, %custom_witness.3c3) [concrete]
// CHECK:STDOUT:   %.Self.frozen.aec: %facet_type.f65 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %complete_type.8c670c.1: <witness> = complete_type_witness %facet_type.f65 [concrete]
// CHECK:STDOUT:   %.Self.frozen.as_type.34d: type = facet_access_type %.Self.frozen.aec [symbolic_self]
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.94b: <witness> = lookup_impl_witness %.Self.frozen.aec, @MulWith.1, @MulWith.1(%Int16) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.212: type = impl_witness_access %MulWith.lookup_impl_witness.94b, element0 [symbolic_self]
// CHECK:STDOUT:   %pattern_type.2f6: type = pattern_type %facet_type.1c9 [concrete]
// CHECK:STDOUT:   %T.patt.e8a: %pattern_type.2f6 = symbolic_binding_pattern T, 2 [symbolic]
// CHECK:STDOUT:   %facet_type.b8a: type = facet_type <@DivWith.1, @DivWith.1(%Int16) & @Destroy> [concrete]
// CHECK:STDOUT:   %.Self.74d: %facet_type.b8a = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.afb: <witness> = lookup_impl_witness %.Self.74d, @DivWith.1, @DivWith.1(%Int16) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.6cf: type = impl_witness_access %DivWith.lookup_impl_witness.afb, element0 [symbolic_self]
// CHECK:STDOUT:   %facet_type.ee9: type = facet_type <@DivWith.1, @DivWith.1(%Int16) & @Destroy where %impl.elem0.6cf = %Int16> [concrete]
// CHECK:STDOUT:   %DivWith.type.b8a: type = facet_type <@DivWith.1, @DivWith.1(%Int16)> [concrete]
// CHECK:STDOUT:   %DivWith.assoc_type.712: type = assoc_entity_type @DivWith.1, @DivWith.1(%Int16) [concrete]
// CHECK:STDOUT:   %assoc0.99c: %DivWith.assoc_type.712 = assoc_entity element0, imports.%Core.import_ref.dee [concrete]
// CHECK:STDOUT:   %assoc1.fb6: %DivWith.assoc_type.712 = assoc_entity element1, imports.%Core.import_ref.f2f [concrete]
// CHECK:STDOUT:   %Op.type.442b68.4: type = fn_type @Op.4 [concrete]
// CHECK:STDOUT:   %Op.72717c.4: %Op.type.442b68.4 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.8d0: <witness> = custom_witness (%Int16, %Op.72717c.4), @DivWith.1, @DivWith.1(%Int16) [concrete]
// CHECK:STDOUT:   %facet_value.318: %facet_type.ee9 = facet_value %i16, (%DestroyImpl.impl_witness.dff, %custom_witness.b80ef6.3, %custom_witness.8d0) [concrete]
// CHECK:STDOUT:   %.Self.frozen.ff8: %facet_type.b8a = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %complete_type.d99d66.1: <witness> = complete_type_witness %facet_type.b8a [concrete]
// CHECK:STDOUT:   %.Self.frozen.as_type.2d9: type = facet_access_type %.Self.frozen.ff8 [symbolic_self]
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.483: <witness> = lookup_impl_witness %.Self.frozen.ff8, @DivWith.1, @DivWith.1(%Int16) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.f9d: type = impl_witness_access %DivWith.lookup_impl_witness.483, element0 [symbolic_self]
// CHECK:STDOUT:   %pattern_type.575: type = pattern_type %facet_type.ee9 [concrete]
// CHECK:STDOUT:   %T.patt.257: %pattern_type.575 = symbolic_binding_pattern T, 2 [symbolic]
// CHECK:STDOUT:   %facet_type.c67: type = facet_type <@ModWith.1, @ModWith.1(%Int16) & @Destroy> [concrete]
// CHECK:STDOUT:   %.Self.71c: %facet_type.c67 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.c2a: <witness> = lookup_impl_witness %.Self.71c, @ModWith.1, @ModWith.1(%Int16) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.14c: type = impl_witness_access %ModWith.lookup_impl_witness.c2a, element0 [symbolic_self]
// CHECK:STDOUT:   %facet_type.fe4: type = facet_type <@ModWith.1, @ModWith.1(%Int16) & @Destroy where %impl.elem0.14c = %Int16> [concrete]
// CHECK:STDOUT:   %ModWith.type.c67: type = facet_type <@ModWith.1, @ModWith.1(%Int16)> [concrete]
// CHECK:STDOUT:   %ModWith.assoc_type.5f6: type = assoc_entity_type @ModWith.1, @ModWith.1(%Int16) [concrete]
// CHECK:STDOUT:   %assoc0.f04: %ModWith.assoc_type.5f6 = assoc_entity element0, imports.%Core.import_ref.61a [concrete]
// CHECK:STDOUT:   %assoc1.d19: %ModWith.assoc_type.5f6 = assoc_entity element1, imports.%Core.import_ref.9f8f [concrete]
// CHECK:STDOUT:   %Op.type.442b68.5: type = fn_type @Op.5 [concrete]
// CHECK:STDOUT:   %Op.72717c.5: %Op.type.442b68.5 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.169: <witness> = custom_witness (%Int16, %Op.72717c.5), @ModWith.1, @ModWith.1(%Int16) [concrete]
// CHECK:STDOUT:   %facet_value.425: %facet_type.fe4 = facet_value %i16, (%DestroyImpl.impl_witness.dff, %custom_witness.b80ef6.3, %custom_witness.169) [concrete]
// CHECK:STDOUT:   %.Self.frozen.13f: %facet_type.c67 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %complete_type.7d80e1.1: <witness> = complete_type_witness %facet_type.c67 [concrete]
// CHECK:STDOUT:   %.Self.frozen.as_type.cb4: type = facet_access_type %.Self.frozen.13f [symbolic_self]
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.0af: <witness> = lookup_impl_witness %.Self.frozen.13f, @ModWith.1, @ModWith.1(%Int16) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.d91: type = impl_witness_access %ModWith.lookup_impl_witness.0af, element0 [symbolic_self]
// CHECK:STDOUT:   %pattern_type.330: type = pattern_type %facet_type.fe4 [concrete]
// CHECK:STDOUT:   %T.patt.5c0: %pattern_type.330 = symbolic_binding_pattern T, 2 [symbolic]
// CHECK:STDOUT:   %facet_type.2ed: type = facet_type <@AddWith.1, @AddWith.1(%i16) & @Destroy> [concrete]
// CHECK:STDOUT:   %.Self.737: %facet_type.2ed = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.de0: <witness> = lookup_impl_witness %.Self.737, @AddWith.1, @AddWith.1(%i16) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.025: type = impl_witness_access %AddWith.lookup_impl_witness.de0, element0 [symbolic_self]
// CHECK:STDOUT:   %facet_type.9dd: type = facet_type <@AddWith.1, @AddWith.1(%i16) & @Destroy where %impl.elem0.025 = %Int16> [concrete]
// CHECK:STDOUT:   %AddWith.type.2ed: type = facet_type <@AddWith.1, @AddWith.1(%i16)> [concrete]
// CHECK:STDOUT:   %AddWith.assoc_type.18f: type = assoc_entity_type @AddWith.1, @AddWith.1(%i16) [concrete]
// CHECK:STDOUT:   %assoc0.b7d: %AddWith.assoc_type.18f = assoc_entity element0, imports.%Core.import_ref.91f [concrete]
// CHECK:STDOUT:   %assoc1.e08: %AddWith.assoc_type.18f = assoc_entity element1, imports.%Core.import_ref.8ad [concrete]
// CHECK:STDOUT:   %Int16.Op.type.83af6e.2: type = fn_type @Int16.Op.2 [concrete]
// CHECK:STDOUT:   %Int16.Op.4dca31.2: %Int16.Op.type.83af6e.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b79: <witness> = custom_witness (%Int16, %Int16.Op.4dca31.2), @AddWith.1, @AddWith.1(%i16) [concrete]
// CHECK:STDOUT:   %facet_value.7e4: %facet_type.9dd = facet_value %Int16, (%DestroyImpl.impl_witness.891, %custom_witness.b80ef6.1, %custom_witness.b79) [concrete]
// CHECK:STDOUT:   %.Self.frozen.17a: %facet_type.2ed = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %complete_type.32de8b.1: <witness> = complete_type_witness %facet_type.2ed [concrete]
// CHECK:STDOUT:   %.Self.frozen.as_type.8c1: type = facet_access_type %.Self.frozen.17a [symbolic_self]
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.03c: <witness> = lookup_impl_witness %.Self.frozen.17a, @AddWith.1, @AddWith.1(%i16) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.719: type = impl_witness_access %AddWith.lookup_impl_witness.03c, element0 [symbolic_self]
// CHECK:STDOUT:   %pattern_type.b6f: type = pattern_type %facet_type.9dd [concrete]
// CHECK:STDOUT:   %T.patt.b9b: %pattern_type.b6f = symbolic_binding_pattern T, 2 [symbolic]
// CHECK:STDOUT:   %x.param_patt.3b0: %pattern_type.785 = value_param_pattern [concrete]
// CHECK:STDOUT:   %x.patt.869: %pattern_type.785 = wrapper_binding_pattern x, %x.param_patt.3b0 [concrete]
// CHECK:STDOUT:   %y.param_patt.322: %pattern_type.9bc = value_param_pattern [concrete]
// CHECK:STDOUT:   %y.patt.a3a: %pattern_type.9bc = wrapper_binding_pattern y, %y.param_patt.322 [concrete]
// CHECK:STDOUT:   %facet_type.606: type = facet_type <@SubWith.1, @SubWith.1(%i16) & @Destroy> [concrete]
// CHECK:STDOUT:   %.Self.220: %facet_type.606 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.ebb: <witness> = lookup_impl_witness %.Self.220, @SubWith.1, @SubWith.1(%i16) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.1b2: type = impl_witness_access %SubWith.lookup_impl_witness.ebb, element0 [symbolic_self]
// CHECK:STDOUT:   %facet_type.1a3: type = facet_type <@SubWith.1, @SubWith.1(%i16) & @Destroy where %impl.elem0.1b2 = %Int16> [concrete]
// CHECK:STDOUT:   %SubWith.type.606: type = facet_type <@SubWith.1, @SubWith.1(%i16)> [concrete]
// CHECK:STDOUT:   %SubWith.assoc_type.9c3: type = assoc_entity_type @SubWith.1, @SubWith.1(%i16) [concrete]
// CHECK:STDOUT:   %assoc0.f66: %SubWith.assoc_type.9c3 = assoc_entity element0, imports.%Core.import_ref.190 [concrete]
// CHECK:STDOUT:   %assoc1.144: %SubWith.assoc_type.9c3 = assoc_entity element1, imports.%Core.import_ref.2f4 [concrete]
// CHECK:STDOUT:   %Int16.Op.type.83af6e.3: type = fn_type @Int16.Op.3 [concrete]
// CHECK:STDOUT:   %Int16.Op.4dca31.3: %Int16.Op.type.83af6e.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.1e8: <witness> = custom_witness (%Int16, %Int16.Op.4dca31.3), @SubWith.1, @SubWith.1(%i16) [concrete]
// CHECK:STDOUT:   %facet_value.dd3: %facet_type.1a3 = facet_value %Int16, (%DestroyImpl.impl_witness.891, %custom_witness.b80ef6.1, %custom_witness.1e8) [concrete]
// CHECK:STDOUT:   %.Self.frozen.6eb: %facet_type.606 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %complete_type.75c188.1: <witness> = complete_type_witness %facet_type.606 [concrete]
// CHECK:STDOUT:   %.Self.frozen.as_type.fe6: type = facet_access_type %.Self.frozen.6eb [symbolic_self]
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.183: <witness> = lookup_impl_witness %.Self.frozen.6eb, @SubWith.1, @SubWith.1(%i16) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.cc9: type = impl_witness_access %SubWith.lookup_impl_witness.183, element0 [symbolic_self]
// CHECK:STDOUT:   %pattern_type.8ce: type = pattern_type %facet_type.1a3 [concrete]
// CHECK:STDOUT:   %T.patt.88b: %pattern_type.8ce = symbolic_binding_pattern T, 2 [symbolic]
// CHECK:STDOUT:   %facet_type.1e2: type = facet_type <@MulWith.1, @MulWith.1(%i16) & @Destroy> [concrete]
// CHECK:STDOUT:   %.Self.c91: %facet_type.1e2 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.04e: <witness> = lookup_impl_witness %.Self.c91, @MulWith.1, @MulWith.1(%i16) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.0df: type = impl_witness_access %MulWith.lookup_impl_witness.04e, element0 [symbolic_self]
// CHECK:STDOUT:   %facet_type.26b: type = facet_type <@MulWith.1, @MulWith.1(%i16) & @Destroy where %impl.elem0.0df = %Int16> [concrete]
// CHECK:STDOUT:   %MulWith.type.1e2: type = facet_type <@MulWith.1, @MulWith.1(%i16)> [concrete]
// CHECK:STDOUT:   %MulWith.assoc_type.34c: type = assoc_entity_type @MulWith.1, @MulWith.1(%i16) [concrete]
// CHECK:STDOUT:   %assoc0.2d4: %MulWith.assoc_type.34c = assoc_entity element0, imports.%Core.import_ref.59c [concrete]
// CHECK:STDOUT:   %assoc1.635: %MulWith.assoc_type.34c = assoc_entity element1, imports.%Core.import_ref.fc2 [concrete]
// CHECK:STDOUT:   %Int16.Op.type.83af6e.4: type = fn_type @Int16.Op.4 [concrete]
// CHECK:STDOUT:   %Int16.Op.4dca31.4: %Int16.Op.type.83af6e.4 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.e2b: <witness> = custom_witness (%Int16, %Int16.Op.4dca31.4), @MulWith.1, @MulWith.1(%i16) [concrete]
// CHECK:STDOUT:   %facet_value.3d9: %facet_type.26b = facet_value %Int16, (%DestroyImpl.impl_witness.891, %custom_witness.b80ef6.1, %custom_witness.e2b) [concrete]
// CHECK:STDOUT:   %.Self.frozen.a99: %facet_type.1e2 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %complete_type.a7adcd.1: <witness> = complete_type_witness %facet_type.1e2 [concrete]
// CHECK:STDOUT:   %.Self.frozen.as_type.6f1: type = facet_access_type %.Self.frozen.a99 [symbolic_self]
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.7ca: <witness> = lookup_impl_witness %.Self.frozen.a99, @MulWith.1, @MulWith.1(%i16) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.c79: type = impl_witness_access %MulWith.lookup_impl_witness.7ca, element0 [symbolic_self]
// CHECK:STDOUT:   %pattern_type.2a0: type = pattern_type %facet_type.26b [concrete]
// CHECK:STDOUT:   %T.patt.c4b: %pattern_type.2a0 = symbolic_binding_pattern T, 2 [symbolic]
// CHECK:STDOUT:   %facet_type.45d: type = facet_type <@DivWith.1, @DivWith.1(%i16) & @Destroy> [concrete]
// CHECK:STDOUT:   %.Self.327: %facet_type.45d = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.851: <witness> = lookup_impl_witness %.Self.327, @DivWith.1, @DivWith.1(%i16) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.833: type = impl_witness_access %DivWith.lookup_impl_witness.851, element0 [symbolic_self]
// CHECK:STDOUT:   %facet_type.cb9: type = facet_type <@DivWith.1, @DivWith.1(%i16) & @Destroy where %impl.elem0.833 = %Int16> [concrete]
// CHECK:STDOUT:   %DivWith.type.45d: type = facet_type <@DivWith.1, @DivWith.1(%i16)> [concrete]
// CHECK:STDOUT:   %DivWith.assoc_type.5a4: type = assoc_entity_type @DivWith.1, @DivWith.1(%i16) [concrete]
// CHECK:STDOUT:   %assoc0.156: %DivWith.assoc_type.5a4 = assoc_entity element0, imports.%Core.import_ref.dee [concrete]
// CHECK:STDOUT:   %assoc1.444: %DivWith.assoc_type.5a4 = assoc_entity element1, imports.%Core.import_ref.f2f [concrete]
// CHECK:STDOUT:   %Int16.Op.type.83af6e.5: type = fn_type @Int16.Op.5 [concrete]
// CHECK:STDOUT:   %Int16.Op.4dca31.5: %Int16.Op.type.83af6e.5 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.21db: <witness> = custom_witness (%Int16, %Int16.Op.4dca31.5), @DivWith.1, @DivWith.1(%i16) [concrete]
// CHECK:STDOUT:   %facet_value.4fa: %facet_type.cb9 = facet_value %Int16, (%DestroyImpl.impl_witness.891, %custom_witness.b80ef6.1, %custom_witness.21db) [concrete]
// CHECK:STDOUT:   %.Self.frozen.52f: %facet_type.45d = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %complete_type.d57eaf.1: <witness> = complete_type_witness %facet_type.45d [concrete]
// CHECK:STDOUT:   %.Self.frozen.as_type.8c0: type = facet_access_type %.Self.frozen.52f [symbolic_self]
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.5df: <witness> = lookup_impl_witness %.Self.frozen.52f, @DivWith.1, @DivWith.1(%i16) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.967: type = impl_witness_access %DivWith.lookup_impl_witness.5df, element0 [symbolic_self]
// CHECK:STDOUT:   %pattern_type.264: type = pattern_type %facet_type.cb9 [concrete]
// CHECK:STDOUT:   %T.patt.19d: %pattern_type.264 = symbolic_binding_pattern T, 2 [symbolic]
// CHECK:STDOUT:   %facet_type.104: type = facet_type <@ModWith.1, @ModWith.1(%i16) & @Destroy> [concrete]
// CHECK:STDOUT:   %.Self.6a7: %facet_type.104 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.177: <witness> = lookup_impl_witness %.Self.6a7, @ModWith.1, @ModWith.1(%i16) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.d83: type = impl_witness_access %ModWith.lookup_impl_witness.177, element0 [symbolic_self]
// CHECK:STDOUT:   %facet_type.70f: type = facet_type <@ModWith.1, @ModWith.1(%i16) & @Destroy where %impl.elem0.d83 = %Int16> [concrete]
// CHECK:STDOUT:   %ModWith.type.104: type = facet_type <@ModWith.1, @ModWith.1(%i16)> [concrete]
// CHECK:STDOUT:   %ModWith.assoc_type.0e9: type = assoc_entity_type @ModWith.1, @ModWith.1(%i16) [concrete]
// CHECK:STDOUT:   %assoc0.92c: %ModWith.assoc_type.0e9 = assoc_entity element0, imports.%Core.import_ref.61a [concrete]
// CHECK:STDOUT:   %assoc1.870: %ModWith.assoc_type.0e9 = assoc_entity element1, imports.%Core.import_ref.9f8f [concrete]
// CHECK:STDOUT:   %Int16.Op.type.83af6e.6: type = fn_type @Int16.Op.6 [concrete]
// CHECK:STDOUT:   %Int16.Op.4dca31.6: %Int16.Op.type.83af6e.6 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.6dd: <witness> = custom_witness (%Int16, %Int16.Op.4dca31.6), @ModWith.1, @ModWith.1(%i16) [concrete]
// CHECK:STDOUT:   %facet_value.45d: %facet_type.70f = facet_value %Int16, (%DestroyImpl.impl_witness.891, %custom_witness.b80ef6.1, %custom_witness.6dd) [concrete]
// CHECK:STDOUT:   %.Self.frozen.b7c: %facet_type.104 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %complete_type.4d792a.1: <witness> = complete_type_witness %facet_type.104 [concrete]
// CHECK:STDOUT:   %.Self.frozen.as_type.e20: type = facet_access_type %.Self.frozen.b7c [symbolic_self]
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.a99: <witness> = lookup_impl_witness %.Self.frozen.b7c, @ModWith.1, @ModWith.1(%i16) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.b94: type = impl_witness_access %ModWith.lookup_impl_witness.a99, element0 [symbolic_self]
// CHECK:STDOUT:   %pattern_type.4c3: type = pattern_type %facet_type.70f [concrete]
// CHECK:STDOUT:   %T.patt.cf6: %pattern_type.4c3 = symbolic_binding_pattern T, 2 [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.4: type = fn_type @SubobjectDestroy.WithSelf.Op.loc190 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.4: %SubobjectDestroy.WithSelf.Op.type.6870fd.4 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.4: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.4), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.b43: %SubobjectDestroy.type = facet_value %Int32, (%custom_witness.b80ef6.4) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.716: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.b43) [concrete]
// CHECK:STDOUT:   %Destroy.facet.806: %Destroy.type = facet_value %Int32, (%DestroyImpl.impl_witness.716, %custom_witness.b80ef6.4) [concrete]
// CHECK:STDOUT:   %facet_type.db7: type = facet_type <@AddWith.1, @AddWith.1(%Int32) & @Destroy> [concrete]
// CHECK:STDOUT:   %.Self.6a9: %facet_type.db7 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.f6c: <witness> = lookup_impl_witness %.Self.6a9, @AddWith.1, @AddWith.1(%Int32) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.a74: type = impl_witness_access %AddWith.lookup_impl_witness.f6c, element0 [symbolic_self]
// CHECK:STDOUT:   %facet_type.bf5: type = facet_type <@AddWith.1, @AddWith.1(%Int32) & @Destroy where %impl.elem0.a74 = %Int32> [concrete]
// CHECK:STDOUT:   %AddWith.type.db7: type = facet_type <@AddWith.1, @AddWith.1(%Int32)> [concrete]
// CHECK:STDOUT:   %AddWith.assoc_type.be8: type = assoc_entity_type @AddWith.1, @AddWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %assoc0.31c: %AddWith.assoc_type.be8 = assoc_entity element0, imports.%Core.import_ref.91f [concrete]
// CHECK:STDOUT:   %assoc1.86c: %AddWith.assoc_type.be8 = assoc_entity element1, imports.%Core.import_ref.8ad [concrete]
// CHECK:STDOUT:   %Int16.Op.type.83af6e.7: type = fn_type @Int16.Op.7 [concrete]
// CHECK:STDOUT:   %Int16.Op.4dca31.7: %Int16.Op.type.83af6e.7 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.513476.1: <witness> = custom_witness (%Int32, %Int16.Op.4dca31.7), @AddWith.1, @AddWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %facet_value.790: %facet_type.bf5 = facet_value %Int16, (%DestroyImpl.impl_witness.891, %custom_witness.b80ef6.1, %custom_witness.513476.1) [concrete]
// CHECK:STDOUT:   %.Self.frozen.bc4: %facet_type.db7 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %complete_type.758ea8.1: <witness> = complete_type_witness %facet_type.db7 [concrete]
// CHECK:STDOUT:   %.Self.frozen.as_type.447: type = facet_access_type %.Self.frozen.bc4 [symbolic_self]
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.a9e: <witness> = lookup_impl_witness %.Self.frozen.bc4, @AddWith.1, @AddWith.1(%Int32) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.517: type = impl_witness_access %AddWith.lookup_impl_witness.a9e, element0 [symbolic_self]
// CHECK:STDOUT:   %pattern_type.eea: type = pattern_type %facet_type.bf5 [concrete]
// CHECK:STDOUT:   %T.patt.79b: %pattern_type.eea = symbolic_binding_pattern T, 2 [symbolic]
// CHECK:STDOUT:   %result.param_patt.e81: %pattern_type.8b4 = value_param_pattern [concrete]
// CHECK:STDOUT:   %result.patt.47f: %pattern_type.8b4 = wrapper_binding_pattern result, %result.param_patt.e81 [concrete]
// CHECK:STDOUT:   %y.param_patt.e0c: %pattern_type.8b4 = value_param_pattern [concrete]
// CHECK:STDOUT:   %y.patt.9e1: %pattern_type.8b4 = wrapper_binding_pattern y, %y.param_patt.e0c [concrete]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.13e: <specific function> = specific_function %SelfDestruct, @SelfDestruct(%Destroy.facet.806) [concrete]
// CHECK:STDOUT:   %facet_type.120: type = facet_type <@SubWith.1, @SubWith.1(%Int32) & @Destroy> [concrete]
// CHECK:STDOUT:   %.Self.861: %facet_type.120 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.4cf: <witness> = lookup_impl_witness %.Self.861, @SubWith.1, @SubWith.1(%Int32) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.94c: type = impl_witness_access %SubWith.lookup_impl_witness.4cf, element0 [symbolic_self]
// CHECK:STDOUT:   %facet_type.b03: type = facet_type <@SubWith.1, @SubWith.1(%Int32) & @Destroy where %impl.elem0.94c = %Int32> [concrete]
// CHECK:STDOUT:   %SubWith.type.120: type = facet_type <@SubWith.1, @SubWith.1(%Int32)> [concrete]
// CHECK:STDOUT:   %SubWith.assoc_type.dae: type = assoc_entity_type @SubWith.1, @SubWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %assoc0.fac: %SubWith.assoc_type.dae = assoc_entity element0, imports.%Core.import_ref.190 [concrete]
// CHECK:STDOUT:   %assoc1.147: %SubWith.assoc_type.dae = assoc_entity element1, imports.%Core.import_ref.2f4 [concrete]
// CHECK:STDOUT:   %Int16.Op.type.83af6e.8: type = fn_type @Int16.Op.8 [concrete]
// CHECK:STDOUT:   %Int16.Op.4dca31.8: %Int16.Op.type.83af6e.8 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.32cef1.1: <witness> = custom_witness (%Int32, %Int16.Op.4dca31.8), @SubWith.1, @SubWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %facet_value.2de: %facet_type.b03 = facet_value %Int16, (%DestroyImpl.impl_witness.891, %custom_witness.b80ef6.1, %custom_witness.32cef1.1) [concrete]
// CHECK:STDOUT:   %.Self.frozen.faf: %facet_type.120 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %complete_type.46ea3c.1: <witness> = complete_type_witness %facet_type.120 [concrete]
// CHECK:STDOUT:   %.Self.frozen.as_type.ea6: type = facet_access_type %.Self.frozen.faf [symbolic_self]
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.c4f: <witness> = lookup_impl_witness %.Self.frozen.faf, @SubWith.1, @SubWith.1(%Int32) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.b5a: type = impl_witness_access %SubWith.lookup_impl_witness.c4f, element0 [symbolic_self]
// CHECK:STDOUT:   %pattern_type.110: type = pattern_type %facet_type.b03 [concrete]
// CHECK:STDOUT:   %T.patt.e31: %pattern_type.110 = symbolic_binding_pattern T, 2 [symbolic]
// CHECK:STDOUT:   %facet_type.d67b: type = facet_type <@MulWith.1, @MulWith.1(%Int32) & @Destroy> [concrete]
// CHECK:STDOUT:   %.Self.cb9: %facet_type.d67b = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.4ca: <witness> = lookup_impl_witness %.Self.cb9, @MulWith.1, @MulWith.1(%Int32) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.311: type = impl_witness_access %MulWith.lookup_impl_witness.4ca, element0 [symbolic_self]
// CHECK:STDOUT:   %facet_type.b52: type = facet_type <@MulWith.1, @MulWith.1(%Int32) & @Destroy where %impl.elem0.311 = %Int32> [concrete]
// CHECK:STDOUT:   %MulWith.type.d67: type = facet_type <@MulWith.1, @MulWith.1(%Int32)> [concrete]
// CHECK:STDOUT:   %MulWith.assoc_type.ab1: type = assoc_entity_type @MulWith.1, @MulWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %assoc0.d0a: %MulWith.assoc_type.ab1 = assoc_entity element0, imports.%Core.import_ref.59c [concrete]
// CHECK:STDOUT:   %assoc1.349: %MulWith.assoc_type.ab1 = assoc_entity element1, imports.%Core.import_ref.fc2 [concrete]
// CHECK:STDOUT:   %Int16.Op.type.83af6e.9: type = fn_type @Int16.Op.9 [concrete]
// CHECK:STDOUT:   %Int16.Op.4dca31.9: %Int16.Op.type.83af6e.9 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.3a0675.1: <witness> = custom_witness (%Int32, %Int16.Op.4dca31.9), @MulWith.1, @MulWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %facet_value.1fb: %facet_type.b52 = facet_value %Int16, (%DestroyImpl.impl_witness.891, %custom_witness.b80ef6.1, %custom_witness.3a0675.1) [concrete]
// CHECK:STDOUT:   %.Self.frozen.c7a: %facet_type.d67b = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %complete_type.920626.1: <witness> = complete_type_witness %facet_type.d67b [concrete]
// CHECK:STDOUT:   %.Self.frozen.as_type.c47: type = facet_access_type %.Self.frozen.c7a [symbolic_self]
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.ab5: <witness> = lookup_impl_witness %.Self.frozen.c7a, @MulWith.1, @MulWith.1(%Int32) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.6ef: type = impl_witness_access %MulWith.lookup_impl_witness.ab5, element0 [symbolic_self]
// CHECK:STDOUT:   %pattern_type.f70: type = pattern_type %facet_type.b52 [concrete]
// CHECK:STDOUT:   %T.patt.040: %pattern_type.f70 = symbolic_binding_pattern T, 2 [symbolic]
// CHECK:STDOUT:   %facet_type.5a1: type = facet_type <@DivWith.1, @DivWith.1(%Int32) & @Destroy> [concrete]
// CHECK:STDOUT:   %.Self.208: %facet_type.5a1 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.0ac: <witness> = lookup_impl_witness %.Self.208, @DivWith.1, @DivWith.1(%Int32) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.437: type = impl_witness_access %DivWith.lookup_impl_witness.0ac, element0 [symbolic_self]
// CHECK:STDOUT:   %facet_type.f29: type = facet_type <@DivWith.1, @DivWith.1(%Int32) & @Destroy where %impl.elem0.437 = %Int32> [concrete]
// CHECK:STDOUT:   %DivWith.type.5a1: type = facet_type <@DivWith.1, @DivWith.1(%Int32)> [concrete]
// CHECK:STDOUT:   %DivWith.assoc_type.22f: type = assoc_entity_type @DivWith.1, @DivWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %assoc0.a9d: %DivWith.assoc_type.22f = assoc_entity element0, imports.%Core.import_ref.dee [concrete]
// CHECK:STDOUT:   %assoc1.004: %DivWith.assoc_type.22f = assoc_entity element1, imports.%Core.import_ref.f2f [concrete]
// CHECK:STDOUT:   %Int16.Op.type.83af6e.10: type = fn_type @Int16.Op.10 [concrete]
// CHECK:STDOUT:   %Int16.Op.4dca31.10: %Int16.Op.type.83af6e.10 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.c32685.1: <witness> = custom_witness (%Int32, %Int16.Op.4dca31.10), @DivWith.1, @DivWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %facet_value.807: %facet_type.f29 = facet_value %Int16, (%DestroyImpl.impl_witness.891, %custom_witness.b80ef6.1, %custom_witness.c32685.1) [concrete]
// CHECK:STDOUT:   %.Self.frozen.057: %facet_type.5a1 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %complete_type.348fba.1: <witness> = complete_type_witness %facet_type.5a1 [concrete]
// CHECK:STDOUT:   %.Self.frozen.as_type.cc3: type = facet_access_type %.Self.frozen.057 [symbolic_self]
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.ca2: <witness> = lookup_impl_witness %.Self.frozen.057, @DivWith.1, @DivWith.1(%Int32) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.14b: type = impl_witness_access %DivWith.lookup_impl_witness.ca2, element0 [symbolic_self]
// CHECK:STDOUT:   %pattern_type.3d3: type = pattern_type %facet_type.f29 [concrete]
// CHECK:STDOUT:   %T.patt.db7: %pattern_type.3d3 = symbolic_binding_pattern T, 2 [symbolic]
// CHECK:STDOUT:   %facet_type.794: type = facet_type <@ModWith.1, @ModWith.1(%Int32) & @Destroy> [concrete]
// CHECK:STDOUT:   %.Self.618: %facet_type.794 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.01c: <witness> = lookup_impl_witness %.Self.618, @ModWith.1, @ModWith.1(%Int32) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.3bb: type = impl_witness_access %ModWith.lookup_impl_witness.01c, element0 [symbolic_self]
// CHECK:STDOUT:   %facet_type.3d9: type = facet_type <@ModWith.1, @ModWith.1(%Int32) & @Destroy where %impl.elem0.3bb = %Int32> [concrete]
// CHECK:STDOUT:   %ModWith.type.794: type = facet_type <@ModWith.1, @ModWith.1(%Int32)> [concrete]
// CHECK:STDOUT:   %ModWith.assoc_type.021: type = assoc_entity_type @ModWith.1, @ModWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %assoc0.310: %ModWith.assoc_type.021 = assoc_entity element0, imports.%Core.import_ref.61a [concrete]
// CHECK:STDOUT:   %assoc1.2eb: %ModWith.assoc_type.021 = assoc_entity element1, imports.%Core.import_ref.9f8f [concrete]
// CHECK:STDOUT:   %Int16.Op.type.83af6e.11: type = fn_type @Int16.Op.11 [concrete]
// CHECK:STDOUT:   %Int16.Op.4dca31.11: %Int16.Op.type.83af6e.11 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.994c75.1: <witness> = custom_witness (%Int32, %Int16.Op.4dca31.11), @ModWith.1, @ModWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %facet_value.c9f: %facet_type.3d9 = facet_value %Int16, (%DestroyImpl.impl_witness.891, %custom_witness.b80ef6.1, %custom_witness.994c75.1) [concrete]
// CHECK:STDOUT:   %.Self.frozen.813: %facet_type.794 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %complete_type.cc95ce.1: <witness> = complete_type_witness %facet_type.794 [concrete]
// CHECK:STDOUT:   %.Self.frozen.as_type.f4a: type = facet_access_type %.Self.frozen.813 [symbolic_self]
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.efb: <witness> = lookup_impl_witness %.Self.frozen.813, @ModWith.1, @ModWith.1(%Int32) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.e83: type = impl_witness_access %ModWith.lookup_impl_witness.efb, element0 [symbolic_self]
// CHECK:STDOUT:   %pattern_type.f5f: type = pattern_type %facet_type.3d9 [concrete]
// CHECK:STDOUT:   %T.patt.27e: %pattern_type.f5f = symbolic_binding_pattern T, 2 [symbolic]
// CHECK:STDOUT:   %facet_type.177: type = facet_type <@AddWith.1, @AddWith.1(%Int16) & @Destroy where %impl.elem0.08e = %Int32> [concrete]
// CHECK:STDOUT:   %Int32.Op.type.83af6e.1: type = fn_type @Int32.Op.1 [concrete]
// CHECK:STDOUT:   %Int32.Op.4dca31.1: %Int32.Op.type.83af6e.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.ea1: <witness> = custom_witness (%Int32, %Int32.Op.4dca31.1), @AddWith.1, @AddWith.1(%Int16) [concrete]
// CHECK:STDOUT:   %facet_value.0f6: %facet_type.177 = facet_value %Int32, (%DestroyImpl.impl_witness.716, %custom_witness.b80ef6.4, %custom_witness.ea1) [concrete]
// CHECK:STDOUT:   %pattern_type.f16: type = pattern_type %facet_type.177 [concrete]
// CHECK:STDOUT:   %T.patt.4c6: %pattern_type.f16 = symbolic_binding_pattern T, 2 [symbolic]
// CHECK:STDOUT:   %x.param_patt.27c: %pattern_type.8b4 = value_param_pattern [concrete]
// CHECK:STDOUT:   %x.patt.740: %pattern_type.8b4 = wrapper_binding_pattern x, %x.param_patt.27c [concrete]
// CHECK:STDOUT:   %facet_type.760: type = facet_type <@SubWith.1, @SubWith.1(%Int16) & @Destroy where %impl.elem0.22a = %Int32> [concrete]
// CHECK:STDOUT:   %Int32.Op.type.83af6e.2: type = fn_type @Int32.Op.2 [concrete]
// CHECK:STDOUT:   %Int32.Op.4dca31.2: %Int32.Op.type.83af6e.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.576: <witness> = custom_witness (%Int32, %Int32.Op.4dca31.2), @SubWith.1, @SubWith.1(%Int16) [concrete]
// CHECK:STDOUT:   %facet_value.50b: %facet_type.760 = facet_value %Int32, (%DestroyImpl.impl_witness.716, %custom_witness.b80ef6.4, %custom_witness.576) [concrete]
// CHECK:STDOUT:   %pattern_type.524: type = pattern_type %facet_type.760 [concrete]
// CHECK:STDOUT:   %T.patt.83d: %pattern_type.524 = symbolic_binding_pattern T, 2 [symbolic]
// CHECK:STDOUT:   %facet_type.1cc: type = facet_type <@MulWith.1, @MulWith.1(%Int16) & @Destroy where %impl.elem0.088 = %Int32> [concrete]
// CHECK:STDOUT:   %Int32.Op.type.83af6e.3: type = fn_type @Int32.Op.3 [concrete]
// CHECK:STDOUT:   %Int32.Op.4dca31.3: %Int32.Op.type.83af6e.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.751: <witness> = custom_witness (%Int32, %Int32.Op.4dca31.3), @MulWith.1, @MulWith.1(%Int16) [concrete]
// CHECK:STDOUT:   %facet_value.457: %facet_type.1cc = facet_value %Int32, (%DestroyImpl.impl_witness.716, %custom_witness.b80ef6.4, %custom_witness.751) [concrete]
// CHECK:STDOUT:   %pattern_type.89d: type = pattern_type %facet_type.1cc [concrete]
// CHECK:STDOUT:   %T.patt.b89: %pattern_type.89d = symbolic_binding_pattern T, 2 [symbolic]
// CHECK:STDOUT:   %facet_type.8ff: type = facet_type <@DivWith.1, @DivWith.1(%Int16) & @Destroy where %impl.elem0.6cf = %Int32> [concrete]
// CHECK:STDOUT:   %Int32.Op.type.83af6e.4: type = fn_type @Int32.Op.4 [concrete]
// CHECK:STDOUT:   %Int32.Op.4dca31.4: %Int32.Op.type.83af6e.4 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.9ad: <witness> = custom_witness (%Int32, %Int32.Op.4dca31.4), @DivWith.1, @DivWith.1(%Int16) [concrete]
// CHECK:STDOUT:   %facet_value.134: %facet_type.8ff = facet_value %Int32, (%DestroyImpl.impl_witness.716, %custom_witness.b80ef6.4, %custom_witness.9ad) [concrete]
// CHECK:STDOUT:   %pattern_type.d01: type = pattern_type %facet_type.8ff [concrete]
// CHECK:STDOUT:   %T.patt.832: %pattern_type.d01 = symbolic_binding_pattern T, 2 [symbolic]
// CHECK:STDOUT:   %facet_type.960: type = facet_type <@ModWith.1, @ModWith.1(%Int16) & @Destroy where %impl.elem0.14c = %Int32> [concrete]
// CHECK:STDOUT:   %Int32.Op.type.83af6e.5: type = fn_type @Int32.Op.5 [concrete]
// CHECK:STDOUT:   %Int32.Op.4dca31.5: %Int32.Op.type.83af6e.5 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.8a2: <witness> = custom_witness (%Int32, %Int32.Op.4dca31.5), @ModWith.1, @ModWith.1(%Int16) [concrete]
// CHECK:STDOUT:   %facet_value.158: %facet_type.960 = facet_value %Int32, (%DestroyImpl.impl_witness.716, %custom_witness.b80ef6.4, %custom_witness.8a2) [concrete]
// CHECK:STDOUT:   %pattern_type.3be: type = pattern_type %facet_type.960 [concrete]
// CHECK:STDOUT:   %T.patt.c9a: %pattern_type.3be = symbolic_binding_pattern T, 2 [symbolic]
// CHECK:STDOUT:   %Int32.Op.type.83af6e.6: type = fn_type @Int32.Op.6 [concrete]
// CHECK:STDOUT:   %Int32.Op.4dca31.6: %Int32.Op.type.83af6e.6 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.513476.2: <witness> = custom_witness (%Int32, %Int32.Op.4dca31.6), @AddWith.1, @AddWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %facet_value.b57: %facet_type.bf5 = facet_value %Int32, (%DestroyImpl.impl_witness.716, %custom_witness.b80ef6.4, %custom_witness.513476.2) [concrete]
// CHECK:STDOUT:   %Int32.Op.type.83af6e.7: type = fn_type @Int32.Op.7 [concrete]
// CHECK:STDOUT:   %Int32.Op.4dca31.7: %Int32.Op.type.83af6e.7 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.32cef1.2: <witness> = custom_witness (%Int32, %Int32.Op.4dca31.7), @SubWith.1, @SubWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %facet_value.748: %facet_type.b03 = facet_value %Int32, (%DestroyImpl.impl_witness.716, %custom_witness.b80ef6.4, %custom_witness.32cef1.2) [concrete]
// CHECK:STDOUT:   %Int32.Op.type.83af6e.8: type = fn_type @Int32.Op.8 [concrete]
// CHECK:STDOUT:   %Int32.Op.4dca31.8: %Int32.Op.type.83af6e.8 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.3a0675.2: <witness> = custom_witness (%Int32, %Int32.Op.4dca31.8), @MulWith.1, @MulWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %facet_value.b23: %facet_type.b52 = facet_value %Int32, (%DestroyImpl.impl_witness.716, %custom_witness.b80ef6.4, %custom_witness.3a0675.2) [concrete]
// CHECK:STDOUT:   %Int32.Op.type.83af6e.9: type = fn_type @Int32.Op.9 [concrete]
// CHECK:STDOUT:   %Int32.Op.4dca31.9: %Int32.Op.type.83af6e.9 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.c32685.2: <witness> = custom_witness (%Int32, %Int32.Op.4dca31.9), @DivWith.1, @DivWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %facet_value.958: %facet_type.f29 = facet_value %Int32, (%DestroyImpl.impl_witness.716, %custom_witness.b80ef6.4, %custom_witness.c32685.2) [concrete]
// CHECK:STDOUT:   %Int32.Op.type.83af6e.10: type = fn_type @Int32.Op.10 [concrete]
// CHECK:STDOUT:   %Int32.Op.4dca31.10: %Int32.Op.type.83af6e.10 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.994c75.2: <witness> = custom_witness (%Int32, %Int32.Op.4dca31.10), @ModWith.1, @ModWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %facet_value.046: %facet_type.3d9 = facet_value %Int32, (%DestroyImpl.impl_witness.716, %custom_witness.b80ef6.4, %custom_witness.994c75.2) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.5: type = fn_type @SubobjectDestroy.WithSelf.Op.loc211 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.5: %SubobjectDestroy.WithSelf.Op.type.6870fd.5 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.5: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.5), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.c36: %SubobjectDestroy.type = facet_value %Int64, (%custom_witness.b80ef6.5) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.67f: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.c36) [concrete]
// CHECK:STDOUT:   %Destroy.facet.c11: %Destroy.type = facet_value %Int64, (%DestroyImpl.impl_witness.67f, %custom_witness.b80ef6.5) [concrete]
// CHECK:STDOUT:   %facet_type.1ff: type = facet_type <@AddWith.1, @AddWith.1(%Int64) & @Destroy> [concrete]
// CHECK:STDOUT:   %.Self.f92: %facet_type.1ff = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.96d: <witness> = lookup_impl_witness %.Self.f92, @AddWith.1, @AddWith.1(%Int64) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.516: type = impl_witness_access %AddWith.lookup_impl_witness.96d, element0 [symbolic_self]
// CHECK:STDOUT:   %facet_type.dab: type = facet_type <@AddWith.1, @AddWith.1(%Int64) & @Destroy where %impl.elem0.516 = %Int64> [concrete]
// CHECK:STDOUT:   %y.param_patt.a08: %pattern_type.4aa = value_param_pattern [concrete]
// CHECK:STDOUT:   %y.patt.069: %pattern_type.4aa = wrapper_binding_pattern y, %y.param_patt.a08 [concrete]
// CHECK:STDOUT:   %AddWith.type.1ff: type = facet_type <@AddWith.1, @AddWith.1(%Int64)> [concrete]
// CHECK:STDOUT:   %AddWith.assoc_type.97f: type = assoc_entity_type @AddWith.1, @AddWith.1(%Int64) [concrete]
// CHECK:STDOUT:   %assoc0.f02: %AddWith.assoc_type.97f = assoc_entity element0, imports.%Core.import_ref.91f [concrete]
// CHECK:STDOUT:   %assoc1.a75: %AddWith.assoc_type.97f = assoc_entity element1, imports.%Core.import_ref.8ad [concrete]
// CHECK:STDOUT:   %Op.type.442b68.6: type = fn_type @Op.6 [concrete]
// CHECK:STDOUT:   %Op.72717c.6: %Op.type.442b68.6 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.f2f: <witness> = custom_witness (%Int64, %Op.72717c.6), @AddWith.1, @AddWith.1(%Int64) [concrete]
// CHECK:STDOUT:   %facet_value.5f6: %facet_type.dab = facet_value %Int32, (%DestroyImpl.impl_witness.716, %custom_witness.b80ef6.4, %custom_witness.f2f) [concrete]
// CHECK:STDOUT:   %.Self.frozen.91a: %facet_type.1ff = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %complete_type.57be9d.1: <witness> = complete_type_witness %facet_type.1ff [concrete]
// CHECK:STDOUT:   %.Self.frozen.as_type.a1b: type = facet_access_type %.Self.frozen.91a [symbolic_self]
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.30a: <witness> = lookup_impl_witness %.Self.frozen.91a, @AddWith.1, @AddWith.1(%Int64) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.b7d: type = impl_witness_access %AddWith.lookup_impl_witness.30a, element0 [symbolic_self]
// CHECK:STDOUT:   %pattern_type.652: type = pattern_type %facet_type.dab [concrete]
// CHECK:STDOUT:   %T.patt.cb1: %pattern_type.652 = symbolic_binding_pattern T, 2 [symbolic]
// CHECK:STDOUT:   %result.param_patt.1e1: %pattern_type.4aa = value_param_pattern [concrete]
// CHECK:STDOUT:   %result.patt.93b: %pattern_type.4aa = wrapper_binding_pattern result, %result.param_patt.1e1 [concrete]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.fb0: <specific function> = specific_function %SelfDestruct, @SelfDestruct(%Destroy.facet.c11) [concrete]
// CHECK:STDOUT:   %facet_type.31c: type = facet_type <@SubWith.1, @SubWith.1(%Int64) & @Destroy> [concrete]
// CHECK:STDOUT:   %.Self.23f: %facet_type.31c = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.445: <witness> = lookup_impl_witness %.Self.23f, @SubWith.1, @SubWith.1(%Int64) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.ee1: type = impl_witness_access %SubWith.lookup_impl_witness.445, element0 [symbolic_self]
// CHECK:STDOUT:   %facet_type.d68: type = facet_type <@SubWith.1, @SubWith.1(%Int64) & @Destroy where %impl.elem0.ee1 = %Int64> [concrete]
// CHECK:STDOUT:   %SubWith.type.31c: type = facet_type <@SubWith.1, @SubWith.1(%Int64)> [concrete]
// CHECK:STDOUT:   %SubWith.assoc_type.4f8: type = assoc_entity_type @SubWith.1, @SubWith.1(%Int64) [concrete]
// CHECK:STDOUT:   %assoc0.13b: %SubWith.assoc_type.4f8 = assoc_entity element0, imports.%Core.import_ref.190 [concrete]
// CHECK:STDOUT:   %assoc1.e70: %SubWith.assoc_type.4f8 = assoc_entity element1, imports.%Core.import_ref.2f4 [concrete]
// CHECK:STDOUT:   %Op.type.442b68.7: type = fn_type @Op.7 [concrete]
// CHECK:STDOUT:   %Op.72717c.7: %Op.type.442b68.7 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.276: <witness> = custom_witness (%Int64, %Op.72717c.7), @SubWith.1, @SubWith.1(%Int64) [concrete]
// CHECK:STDOUT:   %facet_value.a96: %facet_type.d68 = facet_value %Int32, (%DestroyImpl.impl_witness.716, %custom_witness.b80ef6.4, %custom_witness.276) [concrete]
// CHECK:STDOUT:   %.Self.frozen.c93: %facet_type.31c = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %complete_type.cb55bb.1: <witness> = complete_type_witness %facet_type.31c [concrete]
// CHECK:STDOUT:   %.Self.frozen.as_type.7cf: type = facet_access_type %.Self.frozen.c93 [symbolic_self]
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.52f: <witness> = lookup_impl_witness %.Self.frozen.c93, @SubWith.1, @SubWith.1(%Int64) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.5f9: type = impl_witness_access %SubWith.lookup_impl_witness.52f, element0 [symbolic_self]
// CHECK:STDOUT:   %pattern_type.ea4: type = pattern_type %facet_type.d68 [concrete]
// CHECK:STDOUT:   %T.patt.25e: %pattern_type.ea4 = symbolic_binding_pattern T, 2 [symbolic]
// CHECK:STDOUT:   %facet_type.661: type = facet_type <@MulWith.1, @MulWith.1(%Int64) & @Destroy> [concrete]
// CHECK:STDOUT:   %.Self.6d9: %facet_type.661 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.74c: <witness> = lookup_impl_witness %.Self.6d9, @MulWith.1, @MulWith.1(%Int64) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.4c1: type = impl_witness_access %MulWith.lookup_impl_witness.74c, element0 [symbolic_self]
// CHECK:STDOUT:   %facet_type.a2e: type = facet_type <@MulWith.1, @MulWith.1(%Int64) & @Destroy where %impl.elem0.4c1 = %Int64> [concrete]
// CHECK:STDOUT:   %MulWith.type.661: type = facet_type <@MulWith.1, @MulWith.1(%Int64)> [concrete]
// CHECK:STDOUT:   %MulWith.assoc_type.940: type = assoc_entity_type @MulWith.1, @MulWith.1(%Int64) [concrete]
// CHECK:STDOUT:   %assoc0.9d9: %MulWith.assoc_type.940 = assoc_entity element0, imports.%Core.import_ref.59c [concrete]
// CHECK:STDOUT:   %assoc1.b22: %MulWith.assoc_type.940 = assoc_entity element1, imports.%Core.import_ref.fc2 [concrete]
// CHECK:STDOUT:   %Op.type.442b68.8: type = fn_type @Op.8 [concrete]
// CHECK:STDOUT:   %Op.72717c.8: %Op.type.442b68.8 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.8e4: <witness> = custom_witness (%Int64, %Op.72717c.8), @MulWith.1, @MulWith.1(%Int64) [concrete]
// CHECK:STDOUT:   %facet_value.7ed: %facet_type.a2e = facet_value %Int32, (%DestroyImpl.impl_witness.716, %custom_witness.b80ef6.4, %custom_witness.8e4) [concrete]
// CHECK:STDOUT:   %.Self.frozen.5fe: %facet_type.661 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %complete_type.0a6803.1: <witness> = complete_type_witness %facet_type.661 [concrete]
// CHECK:STDOUT:   %.Self.frozen.as_type.e09: type = facet_access_type %.Self.frozen.5fe [symbolic_self]
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.6c6: <witness> = lookup_impl_witness %.Self.frozen.5fe, @MulWith.1, @MulWith.1(%Int64) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.731: type = impl_witness_access %MulWith.lookup_impl_witness.6c6, element0 [symbolic_self]
// CHECK:STDOUT:   %pattern_type.216: type = pattern_type %facet_type.a2e [concrete]
// CHECK:STDOUT:   %T.patt.ec6: %pattern_type.216 = symbolic_binding_pattern T, 2 [symbolic]
// CHECK:STDOUT:   %facet_type.dd8: type = facet_type <@DivWith.1, @DivWith.1(%Int64) & @Destroy> [concrete]
// CHECK:STDOUT:   %.Self.2e9: %facet_type.dd8 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.14c: <witness> = lookup_impl_witness %.Self.2e9, @DivWith.1, @DivWith.1(%Int64) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.d16: type = impl_witness_access %DivWith.lookup_impl_witness.14c, element0 [symbolic_self]
// CHECK:STDOUT:   %facet_type.89f: type = facet_type <@DivWith.1, @DivWith.1(%Int64) & @Destroy where %impl.elem0.d16 = %Int64> [concrete]
// CHECK:STDOUT:   %DivWith.type.dd8: type = facet_type <@DivWith.1, @DivWith.1(%Int64)> [concrete]
// CHECK:STDOUT:   %DivWith.assoc_type.2dc: type = assoc_entity_type @DivWith.1, @DivWith.1(%Int64) [concrete]
// CHECK:STDOUT:   %assoc0.e8b: %DivWith.assoc_type.2dc = assoc_entity element0, imports.%Core.import_ref.dee [concrete]
// CHECK:STDOUT:   %assoc1.8b9: %DivWith.assoc_type.2dc = assoc_entity element1, imports.%Core.import_ref.f2f [concrete]
// CHECK:STDOUT:   %Op.type.442b68.9: type = fn_type @Op.9 [concrete]
// CHECK:STDOUT:   %Op.72717c.9: %Op.type.442b68.9 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.099: <witness> = custom_witness (%Int64, %Op.72717c.9), @DivWith.1, @DivWith.1(%Int64) [concrete]
// CHECK:STDOUT:   %facet_value.5d8: %facet_type.89f = facet_value %Int32, (%DestroyImpl.impl_witness.716, %custom_witness.b80ef6.4, %custom_witness.099) [concrete]
// CHECK:STDOUT:   %.Self.frozen.d10: %facet_type.dd8 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %complete_type.4732f8.1: <witness> = complete_type_witness %facet_type.dd8 [concrete]
// CHECK:STDOUT:   %.Self.frozen.as_type.483: type = facet_access_type %.Self.frozen.d10 [symbolic_self]
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.531: <witness> = lookup_impl_witness %.Self.frozen.d10, @DivWith.1, @DivWith.1(%Int64) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.872: type = impl_witness_access %DivWith.lookup_impl_witness.531, element0 [symbolic_self]
// CHECK:STDOUT:   %pattern_type.fcf: type = pattern_type %facet_type.89f [concrete]
// CHECK:STDOUT:   %T.patt.5ce: %pattern_type.fcf = symbolic_binding_pattern T, 2 [symbolic]
// CHECK:STDOUT:   %facet_type.c72: type = facet_type <@ModWith.1, @ModWith.1(%Int64) & @Destroy> [concrete]
// CHECK:STDOUT:   %.Self.dc4: %facet_type.c72 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.305: <witness> = lookup_impl_witness %.Self.dc4, @ModWith.1, @ModWith.1(%Int64) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.9c8: type = impl_witness_access %ModWith.lookup_impl_witness.305, element0 [symbolic_self]
// CHECK:STDOUT:   %facet_type.573: type = facet_type <@ModWith.1, @ModWith.1(%Int64) & @Destroy where %impl.elem0.9c8 = %Int64> [concrete]
// CHECK:STDOUT:   %ModWith.type.c72: type = facet_type <@ModWith.1, @ModWith.1(%Int64)> [concrete]
// CHECK:STDOUT:   %ModWith.assoc_type.ad2: type = assoc_entity_type @ModWith.1, @ModWith.1(%Int64) [concrete]
// CHECK:STDOUT:   %assoc0.c8a: %ModWith.assoc_type.ad2 = assoc_entity element0, imports.%Core.import_ref.61a [concrete]
// CHECK:STDOUT:   %assoc1.eed: %ModWith.assoc_type.ad2 = assoc_entity element1, imports.%Core.import_ref.9f8f [concrete]
// CHECK:STDOUT:   %Op.type.442b68.10: type = fn_type @Op.10 [concrete]
// CHECK:STDOUT:   %Op.72717c.10: %Op.type.442b68.10 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.129: <witness> = custom_witness (%Int64, %Op.72717c.10), @ModWith.1, @ModWith.1(%Int64) [concrete]
// CHECK:STDOUT:   %facet_value.762: %facet_type.573 = facet_value %Int32, (%DestroyImpl.impl_witness.716, %custom_witness.b80ef6.4, %custom_witness.129) [concrete]
// CHECK:STDOUT:   %.Self.frozen.c84: %facet_type.c72 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %complete_type.4d1b1b.1: <witness> = complete_type_witness %facet_type.c72 [concrete]
// CHECK:STDOUT:   %.Self.frozen.as_type.40c: type = facet_access_type %.Self.frozen.c84 [symbolic_self]
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.06d: <witness> = lookup_impl_witness %.Self.frozen.c84, @ModWith.1, @ModWith.1(%Int64) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.968: type = impl_witness_access %ModWith.lookup_impl_witness.06d, element0 [symbolic_self]
// CHECK:STDOUT:   %pattern_type.6af: type = pattern_type %facet_type.573 [concrete]
// CHECK:STDOUT:   %T.patt.6c9: %pattern_type.6af = symbolic_binding_pattern T, 2 [symbolic]
// CHECK:STDOUT:   %facet_type.279: type = facet_type <@AddWith.1, @AddWith.1(%Int32) & @Destroy where %impl.elem0.a74 = %Int64> [concrete]
// CHECK:STDOUT:   %Op.type.442b68.11: type = fn_type @Op.11 [concrete]
// CHECK:STDOUT:   %Op.72717c.11: %Op.type.442b68.11 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.16b: <witness> = custom_witness (%Int64, %Op.72717c.11), @AddWith.1, @AddWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %facet_value.acf: %facet_type.279 = facet_value %Int64, (%DestroyImpl.impl_witness.67f, %custom_witness.b80ef6.5, %custom_witness.16b) [concrete]
// CHECK:STDOUT:   %pattern_type.95b: type = pattern_type %facet_type.279 [concrete]
// CHECK:STDOUT:   %T.patt.cc5: %pattern_type.95b = symbolic_binding_pattern T, 2 [symbolic]
// CHECK:STDOUT:   %x.param_patt.e68: %pattern_type.4aa = value_param_pattern [concrete]
// CHECK:STDOUT:   %x.patt.7aa: %pattern_type.4aa = wrapper_binding_pattern x, %x.param_patt.e68 [concrete]
// CHECK:STDOUT:   %facet_type.1c5: type = facet_type <@SubWith.1, @SubWith.1(%Int32) & @Destroy where %impl.elem0.94c = %Int64> [concrete]
// CHECK:STDOUT:   %Op.type.442b68.12: type = fn_type @Op.12 [concrete]
// CHECK:STDOUT:   %Op.72717c.12: %Op.type.442b68.12 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.21a: <witness> = custom_witness (%Int64, %Op.72717c.12), @SubWith.1, @SubWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %facet_value.957: %facet_type.1c5 = facet_value %Int64, (%DestroyImpl.impl_witness.67f, %custom_witness.b80ef6.5, %custom_witness.21a) [concrete]
// CHECK:STDOUT:   %pattern_type.33d: type = pattern_type %facet_type.1c5 [concrete]
// CHECK:STDOUT:   %T.patt.203: %pattern_type.33d = symbolic_binding_pattern T, 2 [symbolic]
// CHECK:STDOUT:   %facet_type.ec7: type = facet_type <@MulWith.1, @MulWith.1(%Int32) & @Destroy where %impl.elem0.311 = %Int64> [concrete]
// CHECK:STDOUT:   %Op.type.442b68.13: type = fn_type @Op.13 [concrete]
// CHECK:STDOUT:   %Op.72717c.13: %Op.type.442b68.13 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.798: <witness> = custom_witness (%Int64, %Op.72717c.13), @MulWith.1, @MulWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %facet_value.43e: %facet_type.ec7 = facet_value %Int64, (%DestroyImpl.impl_witness.67f, %custom_witness.b80ef6.5, %custom_witness.798) [concrete]
// CHECK:STDOUT:   %pattern_type.537: type = pattern_type %facet_type.ec7 [concrete]
// CHECK:STDOUT:   %T.patt.f45: %pattern_type.537 = symbolic_binding_pattern T, 2 [symbolic]
// CHECK:STDOUT:   %facet_type.da8: type = facet_type <@DivWith.1, @DivWith.1(%Int32) & @Destroy where %impl.elem0.437 = %Int64> [concrete]
// CHECK:STDOUT:   %Op.type.442b68.14: type = fn_type @Op.14 [concrete]
// CHECK:STDOUT:   %Op.72717c.14: %Op.type.442b68.14 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.37f: <witness> = custom_witness (%Int64, %Op.72717c.14), @DivWith.1, @DivWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %facet_value.7ce: %facet_type.da8 = facet_value %Int64, (%DestroyImpl.impl_witness.67f, %custom_witness.b80ef6.5, %custom_witness.37f) [concrete]
// CHECK:STDOUT:   %pattern_type.612: type = pattern_type %facet_type.da8 [concrete]
// CHECK:STDOUT:   %T.patt.bc3: %pattern_type.612 = symbolic_binding_pattern T, 2 [symbolic]
// CHECK:STDOUT:   %facet_type.005: type = facet_type <@ModWith.1, @ModWith.1(%Int32) & @Destroy where %impl.elem0.3bb = %Int64> [concrete]
// CHECK:STDOUT:   %Op.type.442b68.15: type = fn_type @Op.15 [concrete]
// CHECK:STDOUT:   %Op.72717c.15: %Op.type.442b68.15 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.981: <witness> = custom_witness (%Int64, %Op.72717c.15), @ModWith.1, @ModWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %facet_value.5a6: %facet_type.005 = facet_value %Int64, (%DestroyImpl.impl_witness.67f, %custom_witness.b80ef6.5, %custom_witness.981) [concrete]
// CHECK:STDOUT:   %pattern_type.b4f: type = pattern_type %facet_type.005 [concrete]
// CHECK:STDOUT:   %T.patt.0b5: %pattern_type.b4f = symbolic_binding_pattern T, 2 [symbolic]
// CHECK:STDOUT:   %U.patt.3a0: %pattern_type.9a587c.2 = symbolic_binding_pattern U, 0 [symbolic]
// CHECK:STDOUT:   %U.67d: type = symbolic_binding U, 0 [symbolic]
// CHECK:STDOUT:   %AddAssignWith.type.3a470e.1: type = facet_type <@AddAssignWith.1, @AddAssignWith.1(%Other)> [symbolic]
// CHECK:STDOUT:   %Self.73bb96.1: %AddAssignWith.type.3a470e.1 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %AddAssignWith.assoc_type.08473b.1: type = assoc_entity_type @AddAssignWith.1, @AddAssignWith.1(%Other) [symbolic]
// CHECK:STDOUT:   %AddAssignWith.WithSelf.Op.type.9de7cb.1: type = fn_type @AddAssignWith.WithSelf.Op, @AddAssignWith.WithSelf(%Other, %Self.73bb96.1) [symbolic]
// CHECK:STDOUT:   %AddAssignWith.WithSelf.Op.cfca3c.1: %AddAssignWith.WithSelf.Op.type.9de7cb.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %AddAssignWith.type.3a470e.2: type = facet_type <@AddAssignWith.1, @AddAssignWith.1(%U.67d)> [symbolic]
// CHECK:STDOUT:   %pattern_type.00e: type = pattern_type %AddAssignWith.type.3a470e.2 [symbolic]
// CHECK:STDOUT:   %T.patt.86c: %pattern_type.00e = symbolic_binding_pattern T, 1 [symbolic]
// CHECK:STDOUT:   %T.73b: %AddAssignWith.type.3a470e.2 = symbolic_binding T, 1 [symbolic]
// CHECK:STDOUT:   %T.as_type.415: type = facet_access_type %T.73b [symbolic]
// CHECK:STDOUT:   %pattern_type.a93410.2: type = pattern_type %T.as_type.415 [symbolic]
// CHECK:STDOUT:   %x.patt.988: %pattern_type.a93410.2 = ref_binding_pattern x [symbolic]
// CHECK:STDOUT:   %x.param_patt.2ee: %pattern_type.a93410.2 = var_param_pattern %x.patt.988 [symbolic]
// CHECK:STDOUT:   %pattern_type.51d1c4.4: type = pattern_type %U.67d [symbolic]
// CHECK:STDOUT:   %y.param_patt.be3: %pattern_type.51d1c4.4 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %y.patt.678: %pattern_type.51d1c4.4 = wrapper_binding_pattern y, %y.param_patt.be3 [symbolic]
// CHECK:STDOUT:   %AddAssignWith.assoc_type.08473b.2: type = assoc_entity_type @AddAssignWith.1, @AddAssignWith.1(%U.67d) [symbolic]
// CHECK:STDOUT:   %assoc0.0fc488.2: %AddAssignWith.assoc_type.08473b.2 = assoc_entity element0, imports.%Core.import_ref.90f [symbolic]
// CHECK:STDOUT:   %require_complete.a80: <witness> = require_complete_type %AddAssignWith.type.3a470e.2 [symbolic]
// CHECK:STDOUT:   %assoc0.3b5: %AddAssignWith.assoc_type.08473b.1 = assoc_entity element0, imports.%Core.import_ref.2783 [symbolic]
// CHECK:STDOUT:   %AddAssignWith.lookup_impl_witness: <witness> = lookup_impl_witness %T.73b, @AddAssignWith.1, @AddAssignWith.1(%U.67d) [symbolic]
// CHECK:STDOUT:   %AddAssignWith.WithSelf.Op.type.9de7cb.3: type = fn_type @AddAssignWith.WithSelf.Op, @AddAssignWith.WithSelf(%U.67d, %T.73b) [symbolic]
// CHECK:STDOUT:   %.643: type = fn_type_with_self_type %AddAssignWith.WithSelf.Op.type.9de7cb.3, %T.73b [symbolic]
// CHECK:STDOUT:   %impl.elem0.20b: %.643 = impl_witness_access %AddAssignWith.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.03f: <specific function> = specific_impl_function %impl.elem0.20b, @AddAssignWith.WithSelf.Op(%U.67d, %T.73b) [symbolic]
// CHECK:STDOUT:   %SubAssignWith.type.0c61b3.1: type = facet_type <@SubAssignWith.1, @SubAssignWith.1(%Other)> [symbolic]
// CHECK:STDOUT:   %Self.0dd047.1: %SubAssignWith.type.0c61b3.1 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %SubAssignWith.assoc_type.8211d1.1: type = assoc_entity_type @SubAssignWith.1, @SubAssignWith.1(%Other) [symbolic]
// CHECK:STDOUT:   %SubAssignWith.WithSelf.Op.type.7e787a.1: type = fn_type @SubAssignWith.WithSelf.Op, @SubAssignWith.WithSelf(%Other, %Self.0dd047.1) [symbolic]
// CHECK:STDOUT:   %SubAssignWith.WithSelf.Op.002979.1: %SubAssignWith.WithSelf.Op.type.7e787a.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %SubAssignWith.type.0c61b3.2: type = facet_type <@SubAssignWith.1, @SubAssignWith.1(%U.67d)> [symbolic]
// CHECK:STDOUT:   %pattern_type.c56: type = pattern_type %SubAssignWith.type.0c61b3.2 [symbolic]
// CHECK:STDOUT:   %T.patt.253: %pattern_type.c56 = symbolic_binding_pattern T, 1 [symbolic]
// CHECK:STDOUT:   %T.0dd: %SubAssignWith.type.0c61b3.2 = symbolic_binding T, 1 [symbolic]
// CHECK:STDOUT:   %T.as_type.540: type = facet_access_type %T.0dd [symbolic]
// CHECK:STDOUT:   %pattern_type.e85cd1.2: type = pattern_type %T.as_type.540 [symbolic]
// CHECK:STDOUT:   %x.patt.fed: %pattern_type.e85cd1.2 = ref_binding_pattern x [symbolic]
// CHECK:STDOUT:   %x.param_patt.f54: %pattern_type.e85cd1.2 = var_param_pattern %x.patt.fed [symbolic]
// CHECK:STDOUT:   %SubAssignWith.assoc_type.8211d1.2: type = assoc_entity_type @SubAssignWith.1, @SubAssignWith.1(%U.67d) [symbolic]
// CHECK:STDOUT:   %assoc0.0ba1dc.2: %SubAssignWith.assoc_type.8211d1.2 = assoc_entity element0, imports.%Core.import_ref.8e5 [symbolic]
// CHECK:STDOUT:   %require_complete.b01: <witness> = require_complete_type %SubAssignWith.type.0c61b3.2 [symbolic]
// CHECK:STDOUT:   %assoc0.467: %SubAssignWith.assoc_type.8211d1.1 = assoc_entity element0, imports.%Core.import_ref.91a [symbolic]
// CHECK:STDOUT:   %SubAssignWith.lookup_impl_witness: <witness> = lookup_impl_witness %T.0dd, @SubAssignWith.1, @SubAssignWith.1(%U.67d) [symbolic]
// CHECK:STDOUT:   %SubAssignWith.WithSelf.Op.type.7e787a.3: type = fn_type @SubAssignWith.WithSelf.Op, @SubAssignWith.WithSelf(%U.67d, %T.0dd) [symbolic]
// CHECK:STDOUT:   %.dcf: type = fn_type_with_self_type %SubAssignWith.WithSelf.Op.type.7e787a.3, %T.0dd [symbolic]
// CHECK:STDOUT:   %impl.elem0.8f6: %.dcf = impl_witness_access %SubAssignWith.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.b79: <specific function> = specific_impl_function %impl.elem0.8f6, @SubAssignWith.WithSelf.Op(%U.67d, %T.0dd) [symbolic]
// CHECK:STDOUT:   %MulAssignWith.type.2339fd.1: type = facet_type <@MulAssignWith.1, @MulAssignWith.1(%Other)> [symbolic]
// CHECK:STDOUT:   %Self.9e408c.1: %MulAssignWith.type.2339fd.1 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %MulAssignWith.assoc_type.5adc6f.1: type = assoc_entity_type @MulAssignWith.1, @MulAssignWith.1(%Other) [symbolic]
// CHECK:STDOUT:   %MulAssignWith.WithSelf.Op.type.c4075c.1: type = fn_type @MulAssignWith.WithSelf.Op, @MulAssignWith.WithSelf(%Other, %Self.9e408c.1) [symbolic]
// CHECK:STDOUT:   %MulAssignWith.WithSelf.Op.b36d17.1: %MulAssignWith.WithSelf.Op.type.c4075c.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %MulAssignWith.type.2339fd.2: type = facet_type <@MulAssignWith.1, @MulAssignWith.1(%U.67d)> [symbolic]
// CHECK:STDOUT:   %pattern_type.4a0: type = pattern_type %MulAssignWith.type.2339fd.2 [symbolic]
// CHECK:STDOUT:   %T.patt.34b: %pattern_type.4a0 = symbolic_binding_pattern T, 1 [symbolic]
// CHECK:STDOUT:   %T.9e4: %MulAssignWith.type.2339fd.2 = symbolic_binding T, 1 [symbolic]
// CHECK:STDOUT:   %T.as_type.3e8: type = facet_access_type %T.9e4 [symbolic]
// CHECK:STDOUT:   %pattern_type.9fac4e.2: type = pattern_type %T.as_type.3e8 [symbolic]
// CHECK:STDOUT:   %x.patt.5dd: %pattern_type.9fac4e.2 = ref_binding_pattern x [symbolic]
// CHECK:STDOUT:   %x.param_patt.f0f: %pattern_type.9fac4e.2 = var_param_pattern %x.patt.5dd [symbolic]
// CHECK:STDOUT:   %MulAssignWith.assoc_type.5adc6f.2: type = assoc_entity_type @MulAssignWith.1, @MulAssignWith.1(%U.67d) [symbolic]
// CHECK:STDOUT:   %assoc0.42a576.2: %MulAssignWith.assoc_type.5adc6f.2 = assoc_entity element0, imports.%Core.import_ref.c116 [symbolic]
// CHECK:STDOUT:   %require_complete.422: <witness> = require_complete_type %MulAssignWith.type.2339fd.2 [symbolic]
// CHECK:STDOUT:   %assoc0.911: %MulAssignWith.assoc_type.5adc6f.1 = assoc_entity element0, imports.%Core.import_ref.d9a [symbolic]
// CHECK:STDOUT:   %MulAssignWith.lookup_impl_witness: <witness> = lookup_impl_witness %T.9e4, @MulAssignWith.1, @MulAssignWith.1(%U.67d) [symbolic]
// CHECK:STDOUT:   %MulAssignWith.WithSelf.Op.type.c4075c.3: type = fn_type @MulAssignWith.WithSelf.Op, @MulAssignWith.WithSelf(%U.67d, %T.9e4) [symbolic]
// CHECK:STDOUT:   %.66e: type = fn_type_with_self_type %MulAssignWith.WithSelf.Op.type.c4075c.3, %T.9e4 [symbolic]
// CHECK:STDOUT:   %impl.elem0.1f5: %.66e = impl_witness_access %MulAssignWith.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.817: <specific function> = specific_impl_function %impl.elem0.1f5, @MulAssignWith.WithSelf.Op(%U.67d, %T.9e4) [symbolic]
// CHECK:STDOUT:   %DivAssignWith.type.4b7503.1: type = facet_type <@DivAssignWith.1, @DivAssignWith.1(%Other)> [symbolic]
// CHECK:STDOUT:   %Self.5bbd7a.1: %DivAssignWith.type.4b7503.1 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %DivAssignWith.assoc_type.4e8a63.1: type = assoc_entity_type @DivAssignWith.1, @DivAssignWith.1(%Other) [symbolic]
// CHECK:STDOUT:   %DivAssignWith.WithSelf.Op.type.42bffc.1: type = fn_type @DivAssignWith.WithSelf.Op, @DivAssignWith.WithSelf(%Other, %Self.5bbd7a.1) [symbolic]
// CHECK:STDOUT:   %DivAssignWith.WithSelf.Op.2980e7.1: %DivAssignWith.WithSelf.Op.type.42bffc.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %DivAssignWith.type.4b7503.2: type = facet_type <@DivAssignWith.1, @DivAssignWith.1(%U.67d)> [symbolic]
// CHECK:STDOUT:   %pattern_type.91f: type = pattern_type %DivAssignWith.type.4b7503.2 [symbolic]
// CHECK:STDOUT:   %T.patt.2c1: %pattern_type.91f = symbolic_binding_pattern T, 1 [symbolic]
// CHECK:STDOUT:   %T.5bb: %DivAssignWith.type.4b7503.2 = symbolic_binding T, 1 [symbolic]
// CHECK:STDOUT:   %T.as_type.cf7: type = facet_access_type %T.5bb [symbolic]
// CHECK:STDOUT:   %pattern_type.ce26b8.2: type = pattern_type %T.as_type.cf7 [symbolic]
// CHECK:STDOUT:   %x.patt.df3: %pattern_type.ce26b8.2 = ref_binding_pattern x [symbolic]
// CHECK:STDOUT:   %x.param_patt.d43: %pattern_type.ce26b8.2 = var_param_pattern %x.patt.df3 [symbolic]
// CHECK:STDOUT:   %DivAssignWith.assoc_type.4e8a63.2: type = assoc_entity_type @DivAssignWith.1, @DivAssignWith.1(%U.67d) [symbolic]
// CHECK:STDOUT:   %assoc0.ce2908.2: %DivAssignWith.assoc_type.4e8a63.2 = assoc_entity element0, imports.%Core.import_ref.ee5 [symbolic]
// CHECK:STDOUT:   %require_complete.5fe: <witness> = require_complete_type %DivAssignWith.type.4b7503.2 [symbolic]
// CHECK:STDOUT:   %assoc0.45a: %DivAssignWith.assoc_type.4e8a63.1 = assoc_entity element0, imports.%Core.import_ref.50b [symbolic]
// CHECK:STDOUT:   %DivAssignWith.lookup_impl_witness: <witness> = lookup_impl_witness %T.5bb, @DivAssignWith.1, @DivAssignWith.1(%U.67d) [symbolic]
// CHECK:STDOUT:   %DivAssignWith.WithSelf.Op.type.42bffc.3: type = fn_type @DivAssignWith.WithSelf.Op, @DivAssignWith.WithSelf(%U.67d, %T.5bb) [symbolic]
// CHECK:STDOUT:   %.caf: type = fn_type_with_self_type %DivAssignWith.WithSelf.Op.type.42bffc.3, %T.5bb [symbolic]
// CHECK:STDOUT:   %impl.elem0.c50: %.caf = impl_witness_access %DivAssignWith.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.5a8: <specific function> = specific_impl_function %impl.elem0.c50, @DivAssignWith.WithSelf.Op(%U.67d, %T.5bb) [symbolic]
// CHECK:STDOUT:   %ModAssignWith.type.3a0f29.1: type = facet_type <@ModAssignWith.1, @ModAssignWith.1(%Other)> [symbolic]
// CHECK:STDOUT:   %Self.cc0e96.1: %ModAssignWith.type.3a0f29.1 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %ModAssignWith.assoc_type.01d13e.1: type = assoc_entity_type @ModAssignWith.1, @ModAssignWith.1(%Other) [symbolic]
// CHECK:STDOUT:   %ModAssignWith.WithSelf.Op.type.7f247d.1: type = fn_type @ModAssignWith.WithSelf.Op, @ModAssignWith.WithSelf(%Other, %Self.cc0e96.1) [symbolic]
// CHECK:STDOUT:   %ModAssignWith.WithSelf.Op.9b8575.1: %ModAssignWith.WithSelf.Op.type.7f247d.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %ModAssignWith.type.3a0f29.2: type = facet_type <@ModAssignWith.1, @ModAssignWith.1(%U.67d)> [symbolic]
// CHECK:STDOUT:   %pattern_type.685: type = pattern_type %ModAssignWith.type.3a0f29.2 [symbolic]
// CHECK:STDOUT:   %T.patt.1bf: %pattern_type.685 = symbolic_binding_pattern T, 1 [symbolic]
// CHECK:STDOUT:   %T.cc0: %ModAssignWith.type.3a0f29.2 = symbolic_binding T, 1 [symbolic]
// CHECK:STDOUT:   %T.as_type.739: type = facet_access_type %T.cc0 [symbolic]
// CHECK:STDOUT:   %pattern_type.443202.2: type = pattern_type %T.as_type.739 [symbolic]
// CHECK:STDOUT:   %x.patt.14d: %pattern_type.443202.2 = ref_binding_pattern x [symbolic]
// CHECK:STDOUT:   %x.param_patt.605: %pattern_type.443202.2 = var_param_pattern %x.patt.14d [symbolic]
// CHECK:STDOUT:   %ModAssignWith.assoc_type.01d13e.2: type = assoc_entity_type @ModAssignWith.1, @ModAssignWith.1(%U.67d) [symbolic]
// CHECK:STDOUT:   %assoc0.a11270.2: %ModAssignWith.assoc_type.01d13e.2 = assoc_entity element0, imports.%Core.import_ref.97f [symbolic]
// CHECK:STDOUT:   %require_complete.a87: <witness> = require_complete_type %ModAssignWith.type.3a0f29.2 [symbolic]
// CHECK:STDOUT:   %assoc0.3a8: %ModAssignWith.assoc_type.01d13e.1 = assoc_entity element0, imports.%Core.import_ref.ffb [symbolic]
// CHECK:STDOUT:   %ModAssignWith.lookup_impl_witness: <witness> = lookup_impl_witness %T.cc0, @ModAssignWith.1, @ModAssignWith.1(%U.67d) [symbolic]
// CHECK:STDOUT:   %ModAssignWith.WithSelf.Op.type.7f247d.3: type = fn_type @ModAssignWith.WithSelf.Op, @ModAssignWith.WithSelf(%U.67d, %T.cc0) [symbolic]
// CHECK:STDOUT:   %.99f: type = fn_type_with_self_type %ModAssignWith.WithSelf.Op.type.7f247d.3, %T.cc0 [symbolic]
// CHECK:STDOUT:   %impl.elem0.46a: %.99f = impl_witness_access %ModAssignWith.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.caf: <specific function> = specific_impl_function %impl.elem0.46a, @ModAssignWith.WithSelf.Op(%U.67d, %T.cc0) [symbolic]
// CHECK:STDOUT:   %AddAssignWith.type.803: type = facet_type <@AddAssignWith.1, @AddAssignWith.1(%i16)> [concrete]
// CHECK:STDOUT:   %AddAssignWith.assoc_type.82a: type = assoc_entity_type @AddAssignWith.1, @AddAssignWith.1(%i16) [concrete]
// CHECK:STDOUT:   %assoc0.7ab: %AddAssignWith.assoc_type.82a = assoc_entity element0, imports.%Core.import_ref.90f [concrete]
// CHECK:STDOUT:   %Int16.Op.type.83af6e.12: type = fn_type @Int16.Op.12 [concrete]
// CHECK:STDOUT:   %Int16.Op.4dca31.12: %Int16.Op.type.83af6e.12 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.2fc: <witness> = custom_witness (%Int16.Op.4dca31.12), @AddAssignWith.1, @AddAssignWith.1(%i16) [concrete]
// CHECK:STDOUT:   %AddAssignWith.facet.213: %AddAssignWith.type.803 = facet_value %Int16, (%custom_witness.2fc) [concrete]
// CHECK:STDOUT:   %pattern_type.7df: type = pattern_type %AddAssignWith.type.803 [concrete]
// CHECK:STDOUT:   %T.patt.ec8: %pattern_type.7df = symbolic_binding_pattern T, 1 [symbolic]
// CHECK:STDOUT:   %x.patt.49e: %pattern_type.785 = ref_binding_pattern x [concrete]
// CHECK:STDOUT:   %x.param_patt.69a: %pattern_type.785 = var_param_pattern %x.patt.49e [concrete]
// CHECK:STDOUT:   %SubAssignWith.type.929: type = facet_type <@SubAssignWith.1, @SubAssignWith.1(%i16)> [concrete]
// CHECK:STDOUT:   %SubAssignWith.assoc_type.ecb: type = assoc_entity_type @SubAssignWith.1, @SubAssignWith.1(%i16) [concrete]
// CHECK:STDOUT:   %assoc0.ecb: %SubAssignWith.assoc_type.ecb = assoc_entity element0, imports.%Core.import_ref.8e5 [concrete]
// CHECK:STDOUT:   %Int16.Op.type.83af6e.13: type = fn_type @Int16.Op.13 [concrete]
// CHECK:STDOUT:   %Int16.Op.4dca31.13: %Int16.Op.type.83af6e.13 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.21df: <witness> = custom_witness (%Int16.Op.4dca31.13), @SubAssignWith.1, @SubAssignWith.1(%i16) [concrete]
// CHECK:STDOUT:   %SubAssignWith.facet.474: %SubAssignWith.type.929 = facet_value %Int16, (%custom_witness.21df) [concrete]
// CHECK:STDOUT:   %pattern_type.23a: type = pattern_type %SubAssignWith.type.929 [concrete]
// CHECK:STDOUT:   %T.patt.467: %pattern_type.23a = symbolic_binding_pattern T, 1 [symbolic]
// CHECK:STDOUT:   %MulAssignWith.type.ba0: type = facet_type <@MulAssignWith.1, @MulAssignWith.1(%i16)> [concrete]
// CHECK:STDOUT:   %MulAssignWith.assoc_type.89a: type = assoc_entity_type @MulAssignWith.1, @MulAssignWith.1(%i16) [concrete]
// CHECK:STDOUT:   %assoc0.144: %MulAssignWith.assoc_type.89a = assoc_entity element0, imports.%Core.import_ref.c116 [concrete]
// CHECK:STDOUT:   %Int16.Op.type.83af6e.14: type = fn_type @Int16.Op.14 [concrete]
// CHECK:STDOUT:   %Int16.Op.4dca31.14: %Int16.Op.type.83af6e.14 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.81e: <witness> = custom_witness (%Int16.Op.4dca31.14), @MulAssignWith.1, @MulAssignWith.1(%i16) [concrete]
// CHECK:STDOUT:   %MulAssignWith.facet.9d6: %MulAssignWith.type.ba0 = facet_value %Int16, (%custom_witness.81e) [concrete]
// CHECK:STDOUT:   %pattern_type.d7c: type = pattern_type %MulAssignWith.type.ba0 [concrete]
// CHECK:STDOUT:   %T.patt.32f: %pattern_type.d7c = symbolic_binding_pattern T, 1 [symbolic]
// CHECK:STDOUT:   %DivAssignWith.type.6b5: type = facet_type <@DivAssignWith.1, @DivAssignWith.1(%i16)> [concrete]
// CHECK:STDOUT:   %DivAssignWith.assoc_type.fae: type = assoc_entity_type @DivAssignWith.1, @DivAssignWith.1(%i16) [concrete]
// CHECK:STDOUT:   %assoc0.1d8: %DivAssignWith.assoc_type.fae = assoc_entity element0, imports.%Core.import_ref.ee5 [concrete]
// CHECK:STDOUT:   %Int16.Op.type.83af6e.15: type = fn_type @Int16.Op.15 [concrete]
// CHECK:STDOUT:   %Int16.Op.4dca31.15: %Int16.Op.type.83af6e.15 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.787: <witness> = custom_witness (%Int16.Op.4dca31.15), @DivAssignWith.1, @DivAssignWith.1(%i16) [concrete]
// CHECK:STDOUT:   %DivAssignWith.facet.bed: %DivAssignWith.type.6b5 = facet_value %Int16, (%custom_witness.787) [concrete]
// CHECK:STDOUT:   %pattern_type.7b9: type = pattern_type %DivAssignWith.type.6b5 [concrete]
// CHECK:STDOUT:   %T.patt.db5: %pattern_type.7b9 = symbolic_binding_pattern T, 1 [symbolic]
// CHECK:STDOUT:   %ModAssignWith.type.ddb: type = facet_type <@ModAssignWith.1, @ModAssignWith.1(%i16)> [concrete]
// CHECK:STDOUT:   %ModAssignWith.assoc_type.725: type = assoc_entity_type @ModAssignWith.1, @ModAssignWith.1(%i16) [concrete]
// CHECK:STDOUT:   %assoc0.701: %ModAssignWith.assoc_type.725 = assoc_entity element0, imports.%Core.import_ref.97f [concrete]
// CHECK:STDOUT:   %Int16.Op.type.83af6e.16: type = fn_type @Int16.Op.16 [concrete]
// CHECK:STDOUT:   %Int16.Op.4dca31.16: %Int16.Op.type.83af6e.16 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.c25: <witness> = custom_witness (%Int16.Op.4dca31.16), @ModAssignWith.1, @ModAssignWith.1(%i16) [concrete]
// CHECK:STDOUT:   %ModAssignWith.facet.6d2: %ModAssignWith.type.ddb = facet_value %Int16, (%custom_witness.c25) [concrete]
// CHECK:STDOUT:   %pattern_type.83d: type = pattern_type %ModAssignWith.type.ddb [concrete]
// CHECK:STDOUT:   %T.patt.0ba: %pattern_type.83d = symbolic_binding_pattern T, 1 [symbolic]
// CHECK:STDOUT:   %AddAssignWith.type.eef: type = facet_type <@AddAssignWith.1, @AddAssignWith.1(%Int32)> [concrete]
// CHECK:STDOUT:   %AddAssignWith.assoc_type.c4c: type = assoc_entity_type @AddAssignWith.1, @AddAssignWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %assoc0.ce4: %AddAssignWith.assoc_type.c4c = assoc_entity element0, imports.%Core.import_ref.90f [concrete]
// CHECK:STDOUT:   %Int32.Op.type.83af6e.11: type = fn_type @Int32.Op.11 [concrete]
// CHECK:STDOUT:   %Int32.Op.4dca31.11: %Int32.Op.type.83af6e.11 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.17487e.1: <witness> = custom_witness (%Int32.Op.4dca31.11), @AddAssignWith.1, @AddAssignWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %AddAssignWith.facet.e80: %AddAssignWith.type.eef = facet_value %Int32, (%custom_witness.17487e.1) [concrete]
// CHECK:STDOUT:   %pattern_type.1ab: type = pattern_type %AddAssignWith.type.eef [concrete]
// CHECK:STDOUT:   %T.patt.034: %pattern_type.1ab = symbolic_binding_pattern T, 1 [symbolic]
// CHECK:STDOUT:   %x.patt.d79: %pattern_type.8b4 = ref_binding_pattern x [concrete]
// CHECK:STDOUT:   %x.param_patt.0c7: %pattern_type.8b4 = var_param_pattern %x.patt.d79 [concrete]
// CHECK:STDOUT:   %SubAssignWith.type.1c4: type = facet_type <@SubAssignWith.1, @SubAssignWith.1(%Int32)> [concrete]
// CHECK:STDOUT:   %SubAssignWith.assoc_type.2e9: type = assoc_entity_type @SubAssignWith.1, @SubAssignWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %assoc0.b0e: %SubAssignWith.assoc_type.2e9 = assoc_entity element0, imports.%Core.import_ref.8e5 [concrete]
// CHECK:STDOUT:   %Int32.Op.type.83af6e.13: type = fn_type @Int32.Op.13 [concrete]
// CHECK:STDOUT:   %Int32.Op.4dca31.13: %Int32.Op.type.83af6e.13 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.ebd70f.1: <witness> = custom_witness (%Int32.Op.4dca31.13), @SubAssignWith.1, @SubAssignWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %SubAssignWith.facet.143: %SubAssignWith.type.1c4 = facet_value %Int32, (%custom_witness.ebd70f.1) [concrete]
// CHECK:STDOUT:   %pattern_type.303: type = pattern_type %SubAssignWith.type.1c4 [concrete]
// CHECK:STDOUT:   %T.patt.dc8: %pattern_type.303 = symbolic_binding_pattern T, 1 [symbolic]
// CHECK:STDOUT:   %MulAssignWith.type.ec7: type = facet_type <@MulAssignWith.1, @MulAssignWith.1(%Int32)> [concrete]
// CHECK:STDOUT:   %MulAssignWith.assoc_type.cbc: type = assoc_entity_type @MulAssignWith.1, @MulAssignWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %assoc0.e00: %MulAssignWith.assoc_type.cbc = assoc_entity element0, imports.%Core.import_ref.c116 [concrete]
// CHECK:STDOUT:   %Int32.Op.type.83af6e.14: type = fn_type @Int32.Op.14 [concrete]
// CHECK:STDOUT:   %Int32.Op.4dca31.14: %Int32.Op.type.83af6e.14 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.636b34.1: <witness> = custom_witness (%Int32.Op.4dca31.14), @MulAssignWith.1, @MulAssignWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %MulAssignWith.facet.1e6: %MulAssignWith.type.ec7 = facet_value %Int32, (%custom_witness.636b34.1) [concrete]
// CHECK:STDOUT:   %pattern_type.b21: type = pattern_type %MulAssignWith.type.ec7 [concrete]
// CHECK:STDOUT:   %T.patt.59f: %pattern_type.b21 = symbolic_binding_pattern T, 1 [symbolic]
// CHECK:STDOUT:   %DivAssignWith.type.12e: type = facet_type <@DivAssignWith.1, @DivAssignWith.1(%Int32)> [concrete]
// CHECK:STDOUT:   %DivAssignWith.assoc_type.776: type = assoc_entity_type @DivAssignWith.1, @DivAssignWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %assoc0.14c: %DivAssignWith.assoc_type.776 = assoc_entity element0, imports.%Core.import_ref.ee5 [concrete]
// CHECK:STDOUT:   %Int32.Op.type.83af6e.15: type = fn_type @Int32.Op.15 [concrete]
// CHECK:STDOUT:   %Int32.Op.4dca31.15: %Int32.Op.type.83af6e.15 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.45756b.1: <witness> = custom_witness (%Int32.Op.4dca31.15), @DivAssignWith.1, @DivAssignWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %DivAssignWith.facet.1ca: %DivAssignWith.type.12e = facet_value %Int32, (%custom_witness.45756b.1) [concrete]
// CHECK:STDOUT:   %pattern_type.9bf: type = pattern_type %DivAssignWith.type.12e [concrete]
// CHECK:STDOUT:   %T.patt.b33: %pattern_type.9bf = symbolic_binding_pattern T, 1 [symbolic]
// CHECK:STDOUT:   %ModAssignWith.type.c90: type = facet_type <@ModAssignWith.1, @ModAssignWith.1(%Int32)> [concrete]
// CHECK:STDOUT:   %ModAssignWith.assoc_type.7d0: type = assoc_entity_type @ModAssignWith.1, @ModAssignWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %assoc0.f20: %ModAssignWith.assoc_type.7d0 = assoc_entity element0, imports.%Core.import_ref.97f [concrete]
// CHECK:STDOUT:   %Int32.Op.type.83af6e.16: type = fn_type @Int32.Op.16 [concrete]
// CHECK:STDOUT:   %Int32.Op.4dca31.16: %Int32.Op.type.83af6e.16 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.773f24.1: <witness> = custom_witness (%Int32.Op.4dca31.16), @ModAssignWith.1, @ModAssignWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %ModAssignWith.facet.715: %ModAssignWith.type.c90 = facet_value %Int32, (%custom_witness.773f24.1) [concrete]
// CHECK:STDOUT:   %pattern_type.065: type = pattern_type %ModAssignWith.type.c90 [concrete]
// CHECK:STDOUT:   %T.patt.707: %pattern_type.065 = symbolic_binding_pattern T, 1 [symbolic]
// CHECK:STDOUT:   %Int64.Op.type.83af6e.1: type = fn_type @Int64.Op.1 [concrete]
// CHECK:STDOUT:   %Int64.Op.4dca31.1: %Int64.Op.type.83af6e.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.17487e.2: <witness> = custom_witness (%Int64.Op.4dca31.1), @AddAssignWith.1, @AddAssignWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %AddAssignWith.facet.c6d: %AddAssignWith.type.eef = facet_value %Int64, (%custom_witness.17487e.2) [concrete]
// CHECK:STDOUT:   %x.patt.ace: %pattern_type.4aa = ref_binding_pattern x [concrete]
// CHECK:STDOUT:   %x.param_patt.b33: %pattern_type.4aa = var_param_pattern %x.patt.ace [concrete]
// CHECK:STDOUT:   %Int64.Op.type.83af6e.3: type = fn_type @Int64.Op.3 [concrete]
// CHECK:STDOUT:   %Int64.Op.4dca31.3: %Int64.Op.type.83af6e.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.ebd70f.2: <witness> = custom_witness (%Int64.Op.4dca31.3), @SubAssignWith.1, @SubAssignWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %SubAssignWith.facet.74a: %SubAssignWith.type.1c4 = facet_value %Int64, (%custom_witness.ebd70f.2) [concrete]
// CHECK:STDOUT:   %Int64.Op.type.83af6e.4: type = fn_type @Int64.Op.4 [concrete]
// CHECK:STDOUT:   %Int64.Op.4dca31.4: %Int64.Op.type.83af6e.4 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.636b34.2: <witness> = custom_witness (%Int64.Op.4dca31.4), @MulAssignWith.1, @MulAssignWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %MulAssignWith.facet.d5c: %MulAssignWith.type.ec7 = facet_value %Int64, (%custom_witness.636b34.2) [concrete]
// CHECK:STDOUT:   %Int64.Op.type.83af6e.5: type = fn_type @Int64.Op.5 [concrete]
// CHECK:STDOUT:   %Int64.Op.4dca31.5: %Int64.Op.type.83af6e.5 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.45756b.2: <witness> = custom_witness (%Int64.Op.4dca31.5), @DivAssignWith.1, @DivAssignWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %DivAssignWith.facet.c0b: %DivAssignWith.type.12e = facet_value %Int64, (%custom_witness.45756b.2) [concrete]
// CHECK:STDOUT:   %Int64.Op.type.83af6e.6: type = fn_type @Int64.Op.6 [concrete]
// CHECK:STDOUT:   %Int64.Op.4dca31.6: %Int64.Op.type.83af6e.6 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.773f24.2: <witness> = custom_witness (%Int64.Op.4dca31.6), @ModAssignWith.1, @ModAssignWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %ModAssignWith.facet.cf2: %ModAssignWith.type.c90 = facet_value %Int64, (%custom_witness.773f24.2) [concrete]
// CHECK:STDOUT:   %Inc.type: type = facet_type <@Inc> [concrete]
// CHECK:STDOUT:   %pattern_type.8c1: type = pattern_type %Inc.type [concrete]
// CHECK:STDOUT:   %T.patt.eab: %pattern_type.8c1 = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T.8f3: %Inc.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.232: type = facet_access_type %T.8f3 [symbolic]
// CHECK:STDOUT:   %pattern_type.427bfc.2: type = pattern_type %T.as_type.232 [symbolic]
// CHECK:STDOUT:   %x.patt.a75: %pattern_type.427bfc.2 = ref_binding_pattern x [symbolic]
// CHECK:STDOUT:   %x.param_patt.5da: %pattern_type.427bfc.2 = var_param_pattern %x.patt.a75 [symbolic]
// CHECK:STDOUT:   %Inc.lookup_impl_witness: <witness> = lookup_impl_witness %T.8f3, @Inc [symbolic]
// CHECK:STDOUT:   %Inc.WithSelf.Op.type.877857.2: type = fn_type @Inc.WithSelf.Op, @Inc.WithSelf(%T.8f3) [symbolic]
// CHECK:STDOUT:   %.5a2b: type = fn_type_with_self_type %Inc.WithSelf.Op.type.877857.2, %T.8f3 [symbolic]
// CHECK:STDOUT:   %impl.elem0.ec8: %.5a2b = impl_witness_access %Inc.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.790: <specific function> = specific_impl_function %impl.elem0.ec8, @Inc.WithSelf.Op(%T.8f3) [symbolic]
// CHECK:STDOUT:   %Dec.type: type = facet_type <@Dec> [concrete]
// CHECK:STDOUT:   %pattern_type.72a: type = pattern_type %Dec.type [concrete]
// CHECK:STDOUT:   %T.patt.de9: %pattern_type.72a = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T.e8e: %Dec.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.c32: type = facet_access_type %T.e8e [symbolic]
// CHECK:STDOUT:   %pattern_type.2754f0.2: type = pattern_type %T.as_type.c32 [symbolic]
// CHECK:STDOUT:   %x.patt.08b: %pattern_type.2754f0.2 = ref_binding_pattern x [symbolic]
// CHECK:STDOUT:   %x.param_patt.678: %pattern_type.2754f0.2 = var_param_pattern %x.patt.08b [symbolic]
// CHECK:STDOUT:   %Dec.lookup_impl_witness: <witness> = lookup_impl_witness %T.e8e, @Dec [symbolic]
// CHECK:STDOUT:   %Dec.WithSelf.Op.type.dbc4d2.2: type = fn_type @Dec.WithSelf.Op, @Dec.WithSelf(%T.e8e) [symbolic]
// CHECK:STDOUT:   %.528: type = fn_type_with_self_type %Dec.WithSelf.Op.type.dbc4d2.2, %T.e8e [symbolic]
// CHECK:STDOUT:   %impl.elem0.0b1: %.528 = impl_witness_access %Dec.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.e97: <specific function> = specific_impl_function %impl.elem0.0b1, @Dec.WithSelf.Op(%T.e8e) [symbolic]
// CHECK:STDOUT:   %Negate.type: type = facet_type <@Negate> [concrete]
// CHECK:STDOUT:   %facet_type.871: type = facet_type <@Negate & @Destroy> [concrete]
// CHECK:STDOUT:   %.Self.c98: %facet_type.871 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %Negate.lookup_impl_witness.659: <witness> = lookup_impl_witness %.Self.c98, @Negate [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.2e4: type = impl_witness_access %Negate.lookup_impl_witness.659, element0 [symbolic_self]
// CHECK:STDOUT:   %facet_type.86f: type = facet_type <@Negate & @Destroy where %impl.elem0.2e4 = %Result.as_type> [symbolic]
// CHECK:STDOUT:   %pattern_type.114: type = pattern_type %facet_type.86f [symbolic]
// CHECK:STDOUT:   %T.patt.986: %pattern_type.114 = symbolic_binding_pattern T, 1 [symbolic]
// CHECK:STDOUT:   %T.d5e: %facet_type.86f = symbolic_binding T, 1 [symbolic]
// CHECK:STDOUT:   %y.param_patt.5b5: %pattern_type.86a9db.1 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %y.patt.1d3: %pattern_type.86a9db.1 = wrapper_binding_pattern y, %y.param_patt.5b5 [symbolic]
// CHECK:STDOUT:   %T.as_type.735: type = facet_access_type %T.d5e [symbolic]
// CHECK:STDOUT:   %pattern_type.66b2: type = pattern_type %T.as_type.735 [symbolic]
// CHECK:STDOUT:   %x.param_patt.0a8: %pattern_type.66b2 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %x.patt.424: %pattern_type.66b2 = wrapper_binding_pattern x, %x.param_patt.0a8 [symbolic]
// CHECK:STDOUT:   %Negate.lookup_impl_witness.75a: <witness> = lookup_impl_witness %T.d5e, @Negate [symbolic]
// CHECK:STDOUT:   %Negate.facet.b7e: %Negate.type = facet_value %T.as_type.735, (%Negate.lookup_impl_witness.75a) [symbolic]
// CHECK:STDOUT:   %Negate.WithSelf.Op.type.4d1: type = fn_type @Negate.WithSelf.Op, @Negate.WithSelf(%Negate.facet.b7e) [symbolic]
// CHECK:STDOUT:   %.70d: type = fn_type_with_self_type %Negate.WithSelf.Op.type.4d1, %Negate.facet.b7e [symbolic]
// CHECK:STDOUT:   %impl.elem1.466: %.70d = impl_witness_access %Negate.lookup_impl_witness.75a, element1 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.9ba: <specific function> = specific_impl_function %impl.elem1.466, @Negate.WithSelf.Op(%Negate.facet.b7e) [symbolic]
// CHECK:STDOUT:   %Int32.Op.type.83af6e.17: type = fn_type @Int32.Op.17 [concrete]
// CHECK:STDOUT:   %Int32.Op.4dca31.17: %Int32.Op.type.83af6e.17 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.f3f8: <witness> = custom_witness (%Int32.Op.4dca31.17), @Inc [concrete]
// CHECK:STDOUT:   %Inc.facet.dd2: %Inc.type = facet_value %Int32, (%custom_witness.f3f8) [concrete]
// CHECK:STDOUT:   %Int32.Op.type.83af6e.18: type = fn_type @Int32.Op.18 [concrete]
// CHECK:STDOUT:   %Int32.Op.4dca31.18: %Int32.Op.type.83af6e.18 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b24: <witness> = custom_witness (%Int32.Op.4dca31.18), @Dec [concrete]
// CHECK:STDOUT:   %Dec.facet.e1b: %Dec.type = facet_value %Int32, (%custom_witness.b24) [concrete]
// CHECK:STDOUT:   %facet_type.8cf: type = facet_type <@Negate & @Destroy where %impl.elem0.2e4 = %Int64> [concrete]
// CHECK:STDOUT:   %Int16.Op.type.83af6e.17: type = fn_type @Int16.Op.17 [concrete]
// CHECK:STDOUT:   %Int16.Op.4dca31.17: %Int16.Op.type.83af6e.17 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.100: <witness> = custom_witness (%Int64, %Int16.Op.4dca31.17), @Negate [concrete]
// CHECK:STDOUT:   %facet_value.d2e: %facet_type.8cf = facet_value %Int16, (%DestroyImpl.impl_witness.891, %custom_witness.b80ef6.1, %custom_witness.100) [concrete]
// CHECK:STDOUT:   %pattern_type.4c7: type = pattern_type %facet_type.8cf [concrete]
// CHECK:STDOUT:   %T.patt.0b0: %pattern_type.4c7 = symbolic_binding_pattern T, 1 [symbolic]
// CHECK:STDOUT:   %facet_type.517: type = facet_type <@Negate & @Destroy where %impl.elem0.2e4 = %Int32> [concrete]
// CHECK:STDOUT:   %Int32.Op.type.83af6e.19: type = fn_type @Int32.Op.19 [concrete]
// CHECK:STDOUT:   %Int32.Op.4dca31.19: %Int32.Op.type.83af6e.19 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.a85: <witness> = custom_witness (%Int32, %Int32.Op.4dca31.19), @Negate [concrete]
// CHECK:STDOUT:   %facet_value.8a2: %facet_type.517 = facet_value %Int32, (%DestroyImpl.impl_witness.716, %custom_witness.b80ef6.4, %custom_witness.a85) [concrete]
// CHECK:STDOUT:   %pattern_type.eaa: type = pattern_type %facet_type.517 [concrete]
// CHECK:STDOUT:   %T.patt.532: %pattern_type.eaa = symbolic_binding_pattern T, 1 [symbolic]
// CHECK:STDOUT:   %complete_type.f37bc8.2: <witness> = complete_type_witness %AddWith.type.e4b [concrete]
// CHECK:STDOUT:   %AddWith.facet.2ce: %AddWith.type.e4b = facet_value %i16, (%custom_witness.cba) [concrete]
// CHECK:STDOUT:   %AddWith.WithSelf.Op.type.4a0: type = fn_type @AddWith.WithSelf.Op, @AddWith.WithSelf(%Int16, %AddWith.facet.2ce) [concrete]
// CHECK:STDOUT:   %.7bc: type = fn_type_with_self_type %AddWith.WithSelf.Op.type.4a0, %AddWith.facet.2ce [concrete]
// CHECK:STDOUT:   %complete_type.8a2f63.2: <witness> = complete_type_witness %SubWith.type.68d [concrete]
// CHECK:STDOUT:   %SubWith.facet.1ef: %SubWith.type.68d = facet_value %i16, (%custom_witness.354) [concrete]
// CHECK:STDOUT:   %SubWith.WithSelf.Op.type.23f: type = fn_type @SubWith.WithSelf.Op, @SubWith.WithSelf(%Int16, %SubWith.facet.1ef) [concrete]
// CHECK:STDOUT:   %.eed: type = fn_type_with_self_type %SubWith.WithSelf.Op.type.23f, %SubWith.facet.1ef [concrete]
// CHECK:STDOUT:   %complete_type.8c670c.2: <witness> = complete_type_witness %MulWith.type.f65 [concrete]
// CHECK:STDOUT:   %MulWith.facet.d84: %MulWith.type.f65 = facet_value %i16, (%custom_witness.3c3) [concrete]
// CHECK:STDOUT:   %MulWith.WithSelf.Op.type.131: type = fn_type @MulWith.WithSelf.Op, @MulWith.WithSelf(%Int16, %MulWith.facet.d84) [concrete]
// CHECK:STDOUT:   %.010: type = fn_type_with_self_type %MulWith.WithSelf.Op.type.131, %MulWith.facet.d84 [concrete]
// CHECK:STDOUT:   %complete_type.d99d66.2: <witness> = complete_type_witness %DivWith.type.b8a [concrete]
// CHECK:STDOUT:   %DivWith.facet.4ad: %DivWith.type.b8a = facet_value %i16, (%custom_witness.8d0) [concrete]
// CHECK:STDOUT:   %DivWith.WithSelf.Op.type.edb: type = fn_type @DivWith.WithSelf.Op, @DivWith.WithSelf(%Int16, %DivWith.facet.4ad) [concrete]
// CHECK:STDOUT:   %.b5e: type = fn_type_with_self_type %DivWith.WithSelf.Op.type.edb, %DivWith.facet.4ad [concrete]
// CHECK:STDOUT:   %complete_type.7d80e1.2: <witness> = complete_type_witness %ModWith.type.c67 [concrete]
// CHECK:STDOUT:   %ModWith.facet.ab1: %ModWith.type.c67 = facet_value %i16, (%custom_witness.169) [concrete]
// CHECK:STDOUT:   %ModWith.WithSelf.Op.type.e29: type = fn_type @ModWith.WithSelf.Op, @ModWith.WithSelf(%Int16, %ModWith.facet.ab1) [concrete]
// CHECK:STDOUT:   %.d62: type = fn_type_with_self_type %ModWith.WithSelf.Op.type.e29, %ModWith.facet.ab1 [concrete]
// CHECK:STDOUT:   %complete_type.32de8b.2: <witness> = complete_type_witness %AddWith.type.2ed [concrete]
// CHECK:STDOUT:   %AddWith.facet.11e: %AddWith.type.2ed = facet_value %Int16, (%custom_witness.b79) [concrete]
// CHECK:STDOUT:   %AddWith.WithSelf.Op.type.eb3: type = fn_type @AddWith.WithSelf.Op, @AddWith.WithSelf(%i16, %AddWith.facet.11e) [concrete]
// CHECK:STDOUT:   %.da8: type = fn_type_with_self_type %AddWith.WithSelf.Op.type.eb3, %AddWith.facet.11e [concrete]
// CHECK:STDOUT:   %complete_type.75c188.2: <witness> = complete_type_witness %SubWith.type.606 [concrete]
// CHECK:STDOUT:   %SubWith.facet.6fa: %SubWith.type.606 = facet_value %Int16, (%custom_witness.1e8) [concrete]
// CHECK:STDOUT:   %SubWith.WithSelf.Op.type.bb5: type = fn_type @SubWith.WithSelf.Op, @SubWith.WithSelf(%i16, %SubWith.facet.6fa) [concrete]
// CHECK:STDOUT:   %.42a: type = fn_type_with_self_type %SubWith.WithSelf.Op.type.bb5, %SubWith.facet.6fa [concrete]
// CHECK:STDOUT:   %complete_type.a7adcd.2: <witness> = complete_type_witness %MulWith.type.1e2 [concrete]
// CHECK:STDOUT:   %MulWith.facet.8b1: %MulWith.type.1e2 = facet_value %Int16, (%custom_witness.e2b) [concrete]
// CHECK:STDOUT:   %MulWith.WithSelf.Op.type.8f5: type = fn_type @MulWith.WithSelf.Op, @MulWith.WithSelf(%i16, %MulWith.facet.8b1) [concrete]
// CHECK:STDOUT:   %.ecc: type = fn_type_with_self_type %MulWith.WithSelf.Op.type.8f5, %MulWith.facet.8b1 [concrete]
// CHECK:STDOUT:   %complete_type.d57eaf.2: <witness> = complete_type_witness %DivWith.type.45d [concrete]
// CHECK:STDOUT:   %DivWith.facet.57c: %DivWith.type.45d = facet_value %Int16, (%custom_witness.21db) [concrete]
// CHECK:STDOUT:   %DivWith.WithSelf.Op.type.37a: type = fn_type @DivWith.WithSelf.Op, @DivWith.WithSelf(%i16, %DivWith.facet.57c) [concrete]
// CHECK:STDOUT:   %.5a3: type = fn_type_with_self_type %DivWith.WithSelf.Op.type.37a, %DivWith.facet.57c [concrete]
// CHECK:STDOUT:   %complete_type.4d792a.2: <witness> = complete_type_witness %ModWith.type.104 [concrete]
// CHECK:STDOUT:   %ModWith.facet.f5a5: %ModWith.type.104 = facet_value %Int16, (%custom_witness.6dd) [concrete]
// CHECK:STDOUT:   %ModWith.WithSelf.Op.type.374: type = fn_type @ModWith.WithSelf.Op, @ModWith.WithSelf(%i16, %ModWith.facet.f5a5) [concrete]
// CHECK:STDOUT:   %.52c1: type = fn_type_with_self_type %ModWith.WithSelf.Op.type.374, %ModWith.facet.f5a5 [concrete]
// CHECK:STDOUT:   %complete_type.758ea8.2: <witness> = complete_type_witness %AddWith.type.db7 [concrete]
// CHECK:STDOUT:   %AddWith.facet.8f8: %AddWith.type.db7 = facet_value %Int16, (%custom_witness.513476.1) [concrete]
// CHECK:STDOUT:   %AddWith.WithSelf.Op.type.4cf: type = fn_type @AddWith.WithSelf.Op, @AddWith.WithSelf(%Int32, %AddWith.facet.8f8) [concrete]
// CHECK:STDOUT:   %.feb: type = fn_type_with_self_type %AddWith.WithSelf.Op.type.4cf, %AddWith.facet.8f8 [concrete]
// CHECK:STDOUT:   %complete_type.46ea3c.2: <witness> = complete_type_witness %SubWith.type.120 [concrete]
// CHECK:STDOUT:   %SubWith.facet.0b0: %SubWith.type.120 = facet_value %Int16, (%custom_witness.32cef1.1) [concrete]
// CHECK:STDOUT:   %SubWith.WithSelf.Op.type.e1e: type = fn_type @SubWith.WithSelf.Op, @SubWith.WithSelf(%Int32, %SubWith.facet.0b0) [concrete]
// CHECK:STDOUT:   %.988: type = fn_type_with_self_type %SubWith.WithSelf.Op.type.e1e, %SubWith.facet.0b0 [concrete]
// CHECK:STDOUT:   %complete_type.920626.2: <witness> = complete_type_witness %MulWith.type.d67 [concrete]
// CHECK:STDOUT:   %MulWith.facet.4b9: %MulWith.type.d67 = facet_value %Int16, (%custom_witness.3a0675.1) [concrete]
// CHECK:STDOUT:   %MulWith.WithSelf.Op.type.8f2: type = fn_type @MulWith.WithSelf.Op, @MulWith.WithSelf(%Int32, %MulWith.facet.4b9) [concrete]
// CHECK:STDOUT:   %.47a1: type = fn_type_with_self_type %MulWith.WithSelf.Op.type.8f2, %MulWith.facet.4b9 [concrete]
// CHECK:STDOUT:   %complete_type.348fba.2: <witness> = complete_type_witness %DivWith.type.5a1 [concrete]
// CHECK:STDOUT:   %DivWith.facet.5c5: %DivWith.type.5a1 = facet_value %Int16, (%custom_witness.c32685.1) [concrete]
// CHECK:STDOUT:   %DivWith.WithSelf.Op.type.af3: type = fn_type @DivWith.WithSelf.Op, @DivWith.WithSelf(%Int32, %DivWith.facet.5c5) [concrete]
// CHECK:STDOUT:   %.836: type = fn_type_with_self_type %DivWith.WithSelf.Op.type.af3, %DivWith.facet.5c5 [concrete]
// CHECK:STDOUT:   %complete_type.cc95ce.2: <witness> = complete_type_witness %ModWith.type.794 [concrete]
// CHECK:STDOUT:   %ModWith.facet.cc5: %ModWith.type.794 = facet_value %Int16, (%custom_witness.994c75.1) [concrete]
// CHECK:STDOUT:   %ModWith.WithSelf.Op.type.ab7: type = fn_type @ModWith.WithSelf.Op, @ModWith.WithSelf(%Int32, %ModWith.facet.cc5) [concrete]
// CHECK:STDOUT:   %.2a3: type = fn_type_with_self_type %ModWith.WithSelf.Op.type.ab7, %ModWith.facet.cc5 [concrete]
// CHECK:STDOUT:   %AddWith.facet.b31: %AddWith.type.e4b = facet_value %Int32, (%custom_witness.ea1) [concrete]
// CHECK:STDOUT:   %AddWith.WithSelf.Op.type.faf: type = fn_type @AddWith.WithSelf.Op, @AddWith.WithSelf(%Int16, %AddWith.facet.b31) [concrete]
// CHECK:STDOUT:   %.fd6: type = fn_type_with_self_type %AddWith.WithSelf.Op.type.faf, %AddWith.facet.b31 [concrete]
// CHECK:STDOUT:   %SubWith.facet.d30: %SubWith.type.68d = facet_value %Int32, (%custom_witness.576) [concrete]
// CHECK:STDOUT:   %SubWith.WithSelf.Op.type.6e5: type = fn_type @SubWith.WithSelf.Op, @SubWith.WithSelf(%Int16, %SubWith.facet.d30) [concrete]
// CHECK:STDOUT:   %.d1a: type = fn_type_with_self_type %SubWith.WithSelf.Op.type.6e5, %SubWith.facet.d30 [concrete]
// CHECK:STDOUT:   %MulWith.facet.cd8: %MulWith.type.f65 = facet_value %Int32, (%custom_witness.751) [concrete]
// CHECK:STDOUT:   %MulWith.WithSelf.Op.type.31d: type = fn_type @MulWith.WithSelf.Op, @MulWith.WithSelf(%Int16, %MulWith.facet.cd8) [concrete]
// CHECK:STDOUT:   %.ba7: type = fn_type_with_self_type %MulWith.WithSelf.Op.type.31d, %MulWith.facet.cd8 [concrete]
// CHECK:STDOUT:   %DivWith.facet.d6b: %DivWith.type.b8a = facet_value %Int32, (%custom_witness.9ad) [concrete]
// CHECK:STDOUT:   %DivWith.WithSelf.Op.type.b35: type = fn_type @DivWith.WithSelf.Op, @DivWith.WithSelf(%Int16, %DivWith.facet.d6b) [concrete]
// CHECK:STDOUT:   %.636: type = fn_type_with_self_type %DivWith.WithSelf.Op.type.b35, %DivWith.facet.d6b [concrete]
// CHECK:STDOUT:   %ModWith.facet.9d1: %ModWith.type.c67 = facet_value %Int32, (%custom_witness.8a2) [concrete]
// CHECK:STDOUT:   %ModWith.WithSelf.Op.type.e61: type = fn_type @ModWith.WithSelf.Op, @ModWith.WithSelf(%Int16, %ModWith.facet.9d1) [concrete]
// CHECK:STDOUT:   %.25b: type = fn_type_with_self_type %ModWith.WithSelf.Op.type.e61, %ModWith.facet.9d1 [concrete]
// CHECK:STDOUT:   %AddWith.facet.684: %AddWith.type.db7 = facet_value %Int32, (%custom_witness.513476.2) [concrete]
// CHECK:STDOUT:   %AddWith.WithSelf.Op.type.0af: type = fn_type @AddWith.WithSelf.Op, @AddWith.WithSelf(%Int32, %AddWith.facet.684) [concrete]
// CHECK:STDOUT:   %.e92: type = fn_type_with_self_type %AddWith.WithSelf.Op.type.0af, %AddWith.facet.684 [concrete]
// CHECK:STDOUT:   %SubWith.facet.a46: %SubWith.type.120 = facet_value %Int32, (%custom_witness.32cef1.2) [concrete]
// CHECK:STDOUT:   %SubWith.WithSelf.Op.type.1a5: type = fn_type @SubWith.WithSelf.Op, @SubWith.WithSelf(%Int32, %SubWith.facet.a46) [concrete]
// CHECK:STDOUT:   %.457: type = fn_type_with_self_type %SubWith.WithSelf.Op.type.1a5, %SubWith.facet.a46 [concrete]
// CHECK:STDOUT:   %MulWith.facet.3ab: %MulWith.type.d67 = facet_value %Int32, (%custom_witness.3a0675.2) [concrete]
// CHECK:STDOUT:   %MulWith.WithSelf.Op.type.2ed: type = fn_type @MulWith.WithSelf.Op, @MulWith.WithSelf(%Int32, %MulWith.facet.3ab) [concrete]
// CHECK:STDOUT:   %.9d6: type = fn_type_with_self_type %MulWith.WithSelf.Op.type.2ed, %MulWith.facet.3ab [concrete]
// CHECK:STDOUT:   %DivWith.facet.b4a: %DivWith.type.5a1 = facet_value %Int32, (%custom_witness.c32685.2) [concrete]
// CHECK:STDOUT:   %DivWith.WithSelf.Op.type.085: type = fn_type @DivWith.WithSelf.Op, @DivWith.WithSelf(%Int32, %DivWith.facet.b4a) [concrete]
// CHECK:STDOUT:   %.10c: type = fn_type_with_self_type %DivWith.WithSelf.Op.type.085, %DivWith.facet.b4a [concrete]
// CHECK:STDOUT:   %ModWith.facet.3dc: %ModWith.type.794 = facet_value %Int32, (%custom_witness.994c75.2) [concrete]
// CHECK:STDOUT:   %ModWith.WithSelf.Op.type.a2a: type = fn_type @ModWith.WithSelf.Op, @ModWith.WithSelf(%Int32, %ModWith.facet.3dc) [concrete]
// CHECK:STDOUT:   %.b6d: type = fn_type_with_self_type %ModWith.WithSelf.Op.type.a2a, %ModWith.facet.3dc [concrete]
// CHECK:STDOUT:   %complete_type.57be9d.2: <witness> = complete_type_witness %AddWith.type.1ff [concrete]
// CHECK:STDOUT:   %AddWith.facet.0d8: %AddWith.type.1ff = facet_value %Int32, (%custom_witness.f2f) [concrete]
// CHECK:STDOUT:   %AddWith.WithSelf.Op.type.a40: type = fn_type @AddWith.WithSelf.Op, @AddWith.WithSelf(%Int64, %AddWith.facet.0d8) [concrete]
// CHECK:STDOUT:   %.51d: type = fn_type_with_self_type %AddWith.WithSelf.Op.type.a40, %AddWith.facet.0d8 [concrete]
// CHECK:STDOUT:   %complete_type.cb55bb.2: <witness> = complete_type_witness %SubWith.type.31c [concrete]
// CHECK:STDOUT:   %SubWith.facet.d62: %SubWith.type.31c = facet_value %Int32, (%custom_witness.276) [concrete]
// CHECK:STDOUT:   %SubWith.WithSelf.Op.type.d39: type = fn_type @SubWith.WithSelf.Op, @SubWith.WithSelf(%Int64, %SubWith.facet.d62) [concrete]
// CHECK:STDOUT:   %.e39: type = fn_type_with_self_type %SubWith.WithSelf.Op.type.d39, %SubWith.facet.d62 [concrete]
// CHECK:STDOUT:   %complete_type.0a6803.2: <witness> = complete_type_witness %MulWith.type.661 [concrete]
// CHECK:STDOUT:   %MulWith.facet.4a0: %MulWith.type.661 = facet_value %Int32, (%custom_witness.8e4) [concrete]
// CHECK:STDOUT:   %MulWith.WithSelf.Op.type.489: type = fn_type @MulWith.WithSelf.Op, @MulWith.WithSelf(%Int64, %MulWith.facet.4a0) [concrete]
// CHECK:STDOUT:   %.2e6: type = fn_type_with_self_type %MulWith.WithSelf.Op.type.489, %MulWith.facet.4a0 [concrete]
// CHECK:STDOUT:   %complete_type.4732f8.2: <witness> = complete_type_witness %DivWith.type.dd8 [concrete]
// CHECK:STDOUT:   %DivWith.facet.b29: %DivWith.type.dd8 = facet_value %Int32, (%custom_witness.099) [concrete]
// CHECK:STDOUT:   %DivWith.WithSelf.Op.type.9fa: type = fn_type @DivWith.WithSelf.Op, @DivWith.WithSelf(%Int64, %DivWith.facet.b29) [concrete]
// CHECK:STDOUT:   %.bc2: type = fn_type_with_self_type %DivWith.WithSelf.Op.type.9fa, %DivWith.facet.b29 [concrete]
// CHECK:STDOUT:   %complete_type.4d1b1b.2: <witness> = complete_type_witness %ModWith.type.c72 [concrete]
// CHECK:STDOUT:   %ModWith.facet.b10: %ModWith.type.c72 = facet_value %Int32, (%custom_witness.129) [concrete]
// CHECK:STDOUT:   %ModWith.WithSelf.Op.type.1ecc: type = fn_type @ModWith.WithSelf.Op, @ModWith.WithSelf(%Int64, %ModWith.facet.b10) [concrete]
// CHECK:STDOUT:   %.bf4: type = fn_type_with_self_type %ModWith.WithSelf.Op.type.1ecc, %ModWith.facet.b10 [concrete]
// CHECK:STDOUT:   %AddWith.facet.f65: %AddWith.type.db7 = facet_value %Int64, (%custom_witness.16b) [concrete]
// CHECK:STDOUT:   %AddWith.WithSelf.Op.type.8b6: type = fn_type @AddWith.WithSelf.Op, @AddWith.WithSelf(%Int32, %AddWith.facet.f65) [concrete]
// CHECK:STDOUT:   %.cf7: type = fn_type_with_self_type %AddWith.WithSelf.Op.type.8b6, %AddWith.facet.f65 [concrete]
// CHECK:STDOUT:   %SubWith.facet.245: %SubWith.type.120 = facet_value %Int64, (%custom_witness.21a) [concrete]
// CHECK:STDOUT:   %SubWith.WithSelf.Op.type.2cf: type = fn_type @SubWith.WithSelf.Op, @SubWith.WithSelf(%Int32, %SubWith.facet.245) [concrete]
// CHECK:STDOUT:   %.4e8: type = fn_type_with_self_type %SubWith.WithSelf.Op.type.2cf, %SubWith.facet.245 [concrete]
// CHECK:STDOUT:   %MulWith.facet.394: %MulWith.type.d67 = facet_value %Int64, (%custom_witness.798) [concrete]
// CHECK:STDOUT:   %MulWith.WithSelf.Op.type.0ab: type = fn_type @MulWith.WithSelf.Op, @MulWith.WithSelf(%Int32, %MulWith.facet.394) [concrete]
// CHECK:STDOUT:   %.22d: type = fn_type_with_self_type %MulWith.WithSelf.Op.type.0ab, %MulWith.facet.394 [concrete]
// CHECK:STDOUT:   %DivWith.facet.85e: %DivWith.type.5a1 = facet_value %Int64, (%custom_witness.37f) [concrete]
// CHECK:STDOUT:   %DivWith.WithSelf.Op.type.7b8: type = fn_type @DivWith.WithSelf.Op, @DivWith.WithSelf(%Int32, %DivWith.facet.85e) [concrete]
// CHECK:STDOUT:   %.cde: type = fn_type_with_self_type %DivWith.WithSelf.Op.type.7b8, %DivWith.facet.85e [concrete]
// CHECK:STDOUT:   %ModWith.facet.f5a2: %ModWith.type.794 = facet_value %Int64, (%custom_witness.981) [concrete]
// CHECK:STDOUT:   %ModWith.WithSelf.Op.type.d27: type = fn_type @ModWith.WithSelf.Op, @ModWith.WithSelf(%Int32, %ModWith.facet.f5a2) [concrete]
// CHECK:STDOUT:   %.3b0: type = fn_type_with_self_type %ModWith.WithSelf.Op.type.d27, %ModWith.facet.f5a2 [concrete]
// CHECK:STDOUT:   %complete_type.5fd: <witness> = complete_type_witness %AddAssignWith.type.803 [concrete]
// CHECK:STDOUT:   %AddAssignWith.WithSelf.Op.type.893: type = fn_type @AddAssignWith.WithSelf.Op, @AddAssignWith.WithSelf(%i16, %AddAssignWith.facet.213) [concrete]
// CHECK:STDOUT:   %.5a24: type = fn_type_with_self_type %AddAssignWith.WithSelf.Op.type.893, %AddAssignWith.facet.213 [concrete]
// CHECK:STDOUT:   %complete_type.aff: <witness> = complete_type_witness %SubAssignWith.type.929 [concrete]
// CHECK:STDOUT:   %SubAssignWith.WithSelf.Op.type.0df: type = fn_type @SubAssignWith.WithSelf.Op, @SubAssignWith.WithSelf(%i16, %SubAssignWith.facet.474) [concrete]
// CHECK:STDOUT:   %.cfa: type = fn_type_with_self_type %SubAssignWith.WithSelf.Op.type.0df, %SubAssignWith.facet.474 [concrete]
// CHECK:STDOUT:   %complete_type.deb: <witness> = complete_type_witness %MulAssignWith.type.ba0 [concrete]
// CHECK:STDOUT:   %MulAssignWith.WithSelf.Op.type.726: type = fn_type @MulAssignWith.WithSelf.Op, @MulAssignWith.WithSelf(%i16, %MulAssignWith.facet.9d6) [concrete]
// CHECK:STDOUT:   %.9a5: type = fn_type_with_self_type %MulAssignWith.WithSelf.Op.type.726, %MulAssignWith.facet.9d6 [concrete]
// CHECK:STDOUT:   %complete_type.383: <witness> = complete_type_witness %DivAssignWith.type.6b5 [concrete]
// CHECK:STDOUT:   %DivAssignWith.WithSelf.Op.type.204: type = fn_type @DivAssignWith.WithSelf.Op, @DivAssignWith.WithSelf(%i16, %DivAssignWith.facet.bed) [concrete]
// CHECK:STDOUT:   %.f1e: type = fn_type_with_self_type %DivAssignWith.WithSelf.Op.type.204, %DivAssignWith.facet.bed [concrete]
// CHECK:STDOUT:   %complete_type.3ce: <witness> = complete_type_witness %ModAssignWith.type.ddb [concrete]
// CHECK:STDOUT:   %ModAssignWith.WithSelf.Op.type.4c4: type = fn_type @ModAssignWith.WithSelf.Op, @ModAssignWith.WithSelf(%i16, %ModAssignWith.facet.6d2) [concrete]
// CHECK:STDOUT:   %.37f: type = fn_type_with_self_type %ModAssignWith.WithSelf.Op.type.4c4, %ModAssignWith.facet.6d2 [concrete]
// CHECK:STDOUT:   %complete_type.c41: <witness> = complete_type_witness %AddAssignWith.type.eef [concrete]
// CHECK:STDOUT:   %AddAssignWith.WithSelf.Op.type.b93: type = fn_type @AddAssignWith.WithSelf.Op, @AddAssignWith.WithSelf(%Int32, %AddAssignWith.facet.e80) [concrete]
// CHECK:STDOUT:   %.05e: type = fn_type_with_self_type %AddAssignWith.WithSelf.Op.type.b93, %AddAssignWith.facet.e80 [concrete]
// CHECK:STDOUT:   %complete_type.b00: <witness> = complete_type_witness %SubAssignWith.type.1c4 [concrete]
// CHECK:STDOUT:   %SubAssignWith.WithSelf.Op.type.0d2: type = fn_type @SubAssignWith.WithSelf.Op, @SubAssignWith.WithSelf(%Int32, %SubAssignWith.facet.143) [concrete]
// CHECK:STDOUT:   %.68c: type = fn_type_with_self_type %SubAssignWith.WithSelf.Op.type.0d2, %SubAssignWith.facet.143 [concrete]
// CHECK:STDOUT:   %complete_type.14d: <witness> = complete_type_witness %MulAssignWith.type.ec7 [concrete]
// CHECK:STDOUT:   %MulAssignWith.WithSelf.Op.type.e90: type = fn_type @MulAssignWith.WithSelf.Op, @MulAssignWith.WithSelf(%Int32, %MulAssignWith.facet.1e6) [concrete]
// CHECK:STDOUT:   %.a68: type = fn_type_with_self_type %MulAssignWith.WithSelf.Op.type.e90, %MulAssignWith.facet.1e6 [concrete]
// CHECK:STDOUT:   %complete_type.77b: <witness> = complete_type_witness %DivAssignWith.type.12e [concrete]
// CHECK:STDOUT:   %DivAssignWith.WithSelf.Op.type.78a: type = fn_type @DivAssignWith.WithSelf.Op, @DivAssignWith.WithSelf(%Int32, %DivAssignWith.facet.1ca) [concrete]
// CHECK:STDOUT:   %.aa9: type = fn_type_with_self_type %DivAssignWith.WithSelf.Op.type.78a, %DivAssignWith.facet.1ca [concrete]
// CHECK:STDOUT:   %complete_type.65d: <witness> = complete_type_witness %ModAssignWith.type.c90 [concrete]
// CHECK:STDOUT:   %ModAssignWith.WithSelf.Op.type.81c: type = fn_type @ModAssignWith.WithSelf.Op, @ModAssignWith.WithSelf(%Int32, %ModAssignWith.facet.715) [concrete]
// CHECK:STDOUT:   %.913: type = fn_type_with_self_type %ModAssignWith.WithSelf.Op.type.81c, %ModAssignWith.facet.715 [concrete]
// CHECK:STDOUT:   %AddAssignWith.WithSelf.Op.type.e31: type = fn_type @AddAssignWith.WithSelf.Op, @AddAssignWith.WithSelf(%Int32, %AddAssignWith.facet.c6d) [concrete]
// CHECK:STDOUT:   %.454: type = fn_type_with_self_type %AddAssignWith.WithSelf.Op.type.e31, %AddAssignWith.facet.c6d [concrete]
// CHECK:STDOUT:   %SubAssignWith.WithSelf.Op.type.e54: type = fn_type @SubAssignWith.WithSelf.Op, @SubAssignWith.WithSelf(%Int32, %SubAssignWith.facet.74a) [concrete]
// CHECK:STDOUT:   %.d58: type = fn_type_with_self_type %SubAssignWith.WithSelf.Op.type.e54, %SubAssignWith.facet.74a [concrete]
// CHECK:STDOUT:   %MulAssignWith.WithSelf.Op.type.2e6: type = fn_type @MulAssignWith.WithSelf.Op, @MulAssignWith.WithSelf(%Int32, %MulAssignWith.facet.d5c) [concrete]
// CHECK:STDOUT:   %.536: type = fn_type_with_self_type %MulAssignWith.WithSelf.Op.type.2e6, %MulAssignWith.facet.d5c [concrete]
// CHECK:STDOUT:   %DivAssignWith.WithSelf.Op.type.f46: type = fn_type @DivAssignWith.WithSelf.Op, @DivAssignWith.WithSelf(%Int32, %DivAssignWith.facet.c0b) [concrete]
// CHECK:STDOUT:   %.58e: type = fn_type_with_self_type %DivAssignWith.WithSelf.Op.type.f46, %DivAssignWith.facet.c0b [concrete]
// CHECK:STDOUT:   %ModAssignWith.WithSelf.Op.type.aab: type = fn_type @ModAssignWith.WithSelf.Op, @ModAssignWith.WithSelf(%Int32, %ModAssignWith.facet.cf2) [concrete]
// CHECK:STDOUT:   %.253: type = fn_type_with_self_type %ModAssignWith.WithSelf.Op.type.aab, %ModAssignWith.facet.cf2 [concrete]
// CHECK:STDOUT:   %Inc.WithSelf.Op.type.13b: type = fn_type @Inc.WithSelf.Op, @Inc.WithSelf(%Inc.facet.dd2) [concrete]
// CHECK:STDOUT:   %.214: type = fn_type_with_self_type %Inc.WithSelf.Op.type.13b, %Inc.facet.dd2 [concrete]
// CHECK:STDOUT:   %Dec.WithSelf.Op.type.0a4: type = fn_type @Dec.WithSelf.Op, @Dec.WithSelf(%Dec.facet.e1b) [concrete]
// CHECK:STDOUT:   %.55c: type = fn_type_with_self_type %Dec.WithSelf.Op.type.0a4, %Dec.facet.e1b [concrete]
// CHECK:STDOUT:   %Negate.facet.8f4: %Negate.type = facet_value %Int16, (%custom_witness.100) [concrete]
// CHECK:STDOUT:   %Negate.WithSelf.Op.type.34a: type = fn_type @Negate.WithSelf.Op, @Negate.WithSelf(%Negate.facet.8f4) [concrete]
// CHECK:STDOUT:   %.789: type = fn_type_with_self_type %Negate.WithSelf.Op.type.34a, %Negate.facet.8f4 [concrete]
// CHECK:STDOUT:   %Negate.facet.aa9: %Negate.type = facet_value %Int32, (%custom_witness.a85) [concrete]
// CHECK:STDOUT:   %Negate.WithSelf.Op.type.25a: type = fn_type @Negate.WithSelf.Op, @Negate.WithSelf(%Negate.facet.aa9) [concrete]
// CHECK:STDOUT:   %.dcd: type = fn_type_with_self_type %Negate.WithSelf.Op.type.25a, %Negate.facet.aa9 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.5c64: @AddWith.WithSelf.%AddWith.assoc_type (%AddWith.assoc_type.1d5) = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [symbolic = @AddWith.WithSelf.%assoc1 (constants.%assoc1.84a)]
// CHECK:STDOUT:   %Core.import_ref.8ad: @AddWith.WithSelf.%AddWith.WithSelf.Op.type (%AddWith.WithSelf.Op.type.70b) = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [symbolic = @AddWith.WithSelf.%AddWith.WithSelf.Op (constants.%AddWith.WithSelf.Op.bb1)]
// CHECK:STDOUT:   %Core.import_ref.91f: type = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [concrete = %Result.9a4]
// CHECK:STDOUT:   %Result.9a4: type = assoc_const_decl @Result.1 [concrete] {}
// CHECK:STDOUT:   %Core.import_ref.e9a = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.5fe: @SubWith.WithSelf.%SubWith.assoc_type (%SubWith.assoc_type.300) = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [symbolic = @SubWith.WithSelf.%assoc1 (constants.%assoc1.b58)]
// CHECK:STDOUT:   %Core.import_ref.2f4: @SubWith.WithSelf.%SubWith.WithSelf.Op.type (%SubWith.WithSelf.Op.type.028) = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [symbolic = @SubWith.WithSelf.%SubWith.WithSelf.Op (constants.%SubWith.WithSelf.Op.e84)]
// CHECK:STDOUT:   %Core.import_ref.190: type = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [concrete = %Result.7bd]
// CHECK:STDOUT:   %Result.7bd: type = assoc_const_decl @Result.2 [concrete] {}
// CHECK:STDOUT:   %Core.import_ref.124 = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.0a8: @MulWith.WithSelf.%MulWith.assoc_type (%MulWith.assoc_type.c81) = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [symbolic = @MulWith.WithSelf.%assoc1 (constants.%assoc1.d9e)]
// CHECK:STDOUT:   %Core.import_ref.fc2: @MulWith.WithSelf.%MulWith.WithSelf.Op.type (%MulWith.WithSelf.Op.type.dad) = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [symbolic = @MulWith.WithSelf.%MulWith.WithSelf.Op (constants.%MulWith.WithSelf.Op.507)]
// CHECK:STDOUT:   %Core.import_ref.59c: type = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [concrete = %Result.a64]
// CHECK:STDOUT:   %Result.a64: type = assoc_const_decl @Result.3 [concrete] {}
// CHECK:STDOUT:   %Core.import_ref.29d = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.972: @DivWith.WithSelf.%DivWith.assoc_type (%DivWith.assoc_type.e7d) = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [symbolic = @DivWith.WithSelf.%assoc1 (constants.%assoc1.2b8)]
// CHECK:STDOUT:   %Core.import_ref.f2f: @DivWith.WithSelf.%DivWith.WithSelf.Op.type (%DivWith.WithSelf.Op.type.207) = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [symbolic = @DivWith.WithSelf.%DivWith.WithSelf.Op (constants.%DivWith.WithSelf.Op.4f1)]
// CHECK:STDOUT:   %Core.import_ref.dee: type = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [concrete = %Result.cb0]
// CHECK:STDOUT:   %Result.cb0: type = assoc_const_decl @Result.4 [concrete] {}
// CHECK:STDOUT:   %Core.import_ref.94d = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.ea5e3: @ModWith.WithSelf.%ModWith.assoc_type (%ModWith.assoc_type.1f1) = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [symbolic = @ModWith.WithSelf.%assoc1 (constants.%assoc1.f4b)]
// CHECK:STDOUT:   %Core.import_ref.9f8f: @ModWith.WithSelf.%ModWith.WithSelf.Op.type (%ModWith.WithSelf.Op.type.04e) = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [symbolic = @ModWith.WithSelf.%ModWith.WithSelf.Op (constants.%ModWith.WithSelf.Op.6fb)]
// CHECK:STDOUT:   %Core.import_ref.61a: type = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [concrete = %Result.803]
// CHECK:STDOUT:   %Result.803: type = assoc_const_decl @Result.5 [concrete] {}
// CHECK:STDOUT:   %Core.import_ref.be00 = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.36d: @AddAssignWith.WithSelf.%AddAssignWith.assoc_type (%AddAssignWith.assoc_type.08473b.1) = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [symbolic = @AddAssignWith.WithSelf.%assoc0 (constants.%assoc0.3b5)]
// CHECK:STDOUT:   %Core.import_ref.90f: @AddAssignWith.WithSelf.%AddAssignWith.WithSelf.Op.type (%AddAssignWith.WithSelf.Op.type.9de7cb.1) = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [symbolic = @AddAssignWith.WithSelf.%AddAssignWith.WithSelf.Op (constants.%AddAssignWith.WithSelf.Op.cfca3c.1)]
// CHECK:STDOUT:   %Core.import_ref.2783 = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.1f6: @SubAssignWith.WithSelf.%SubAssignWith.assoc_type (%SubAssignWith.assoc_type.8211d1.1) = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [symbolic = @SubAssignWith.WithSelf.%assoc0 (constants.%assoc0.467)]
// CHECK:STDOUT:   %Core.import_ref.8e5: @SubAssignWith.WithSelf.%SubAssignWith.WithSelf.Op.type (%SubAssignWith.WithSelf.Op.type.7e787a.1) = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [symbolic = @SubAssignWith.WithSelf.%SubAssignWith.WithSelf.Op (constants.%SubAssignWith.WithSelf.Op.002979.1)]
// CHECK:STDOUT:   %Core.import_ref.91a = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.575: @MulAssignWith.WithSelf.%MulAssignWith.assoc_type (%MulAssignWith.assoc_type.5adc6f.1) = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [symbolic = @MulAssignWith.WithSelf.%assoc0 (constants.%assoc0.911)]
// CHECK:STDOUT:   %Core.import_ref.c116: @MulAssignWith.WithSelf.%MulAssignWith.WithSelf.Op.type (%MulAssignWith.WithSelf.Op.type.c4075c.1) = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [symbolic = @MulAssignWith.WithSelf.%MulAssignWith.WithSelf.Op (constants.%MulAssignWith.WithSelf.Op.b36d17.1)]
// CHECK:STDOUT:   %Core.import_ref.d9a = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.5dd: @DivAssignWith.WithSelf.%DivAssignWith.assoc_type (%DivAssignWith.assoc_type.4e8a63.1) = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [symbolic = @DivAssignWith.WithSelf.%assoc0 (constants.%assoc0.45a)]
// CHECK:STDOUT:   %Core.import_ref.ee5: @DivAssignWith.WithSelf.%DivAssignWith.WithSelf.Op.type (%DivAssignWith.WithSelf.Op.type.42bffc.1) = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [symbolic = @DivAssignWith.WithSelf.%DivAssignWith.WithSelf.Op (constants.%DivAssignWith.WithSelf.Op.2980e7.1)]
// CHECK:STDOUT:   %Core.import_ref.50b = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.1a2: @ModAssignWith.WithSelf.%ModAssignWith.assoc_type (%ModAssignWith.assoc_type.01d13e.1) = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [symbolic = @ModAssignWith.WithSelf.%assoc0 (constants.%assoc0.3a8)]
// CHECK:STDOUT:   %Core.import_ref.97f: @ModAssignWith.WithSelf.%ModAssignWith.WithSelf.Op.type (%ModAssignWith.WithSelf.Op.type.7f247d.1) = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [symbolic = @ModAssignWith.WithSelf.%ModAssignWith.WithSelf.Op (constants.%ModAssignWith.WithSelf.Op.9b8575.1)]
// CHECK:STDOUT:   %Core.import_ref.ffb = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @AddWith.loc124(%Result.loc121_9.2: %Destroy.type, %U.loc122_4.2: %Destroy.type, %T.loc123_4.2: @AddWith.loc124.%facet_type.loc123_37 (%facet_type.c17)) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %require_complete.loc126: <witness> = require_complete_type %AddWith.type.loc123_35.1 [symbolic = %require_complete.loc126 (constants.%require_complete.528878.2)]
// CHECK:STDOUT:   %assoc1: @AddWith.loc124.%AddWith.assoc_type (%AddWith.assoc_type.71f) = assoc_entity element1, imports.%Core.import_ref.8ad [symbolic = %assoc1 (constants.%assoc1.f40)]
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.loc126: <witness> = lookup_impl_witness %T.loc123_4.1, @AddWith.1, @AddWith.1(%U.as_type.loc123_35.1) [symbolic = %AddWith.lookup_impl_witness.loc126 (constants.%AddWith.lookup_impl_witness.206)]
// CHECK:STDOUT:   %AddWith.facet: @AddWith.loc124.%AddWith.type.loc123_35.1 (%AddWith.type.4f5) = facet_value %T.as_type.loc124_29.1, (%AddWith.lookup_impl_witness.loc126) [symbolic = %AddWith.facet (constants.%AddWith.facet.880)]
// CHECK:STDOUT:   %AddWith.WithSelf.Op.type: type = fn_type @AddWith.WithSelf.Op, @AddWith.WithSelf(%U.as_type.loc123_35.1, %AddWith.facet) [symbolic = %AddWith.WithSelf.Op.type (constants.%AddWith.WithSelf.Op.type.c00)]
// CHECK:STDOUT:   %.loc126_5.6: type = fn_type_with_self_type %AddWith.WithSelf.Op.type, %AddWith.facet [symbolic = %.loc126_5.6 (constants.%.a11)]
// CHECK:STDOUT:   %impl.elem1.loc126_5.2: @AddWith.loc124.%.loc126_5.6 (%.a11) = impl_witness_access %AddWith.lookup_impl_witness.loc126, element1 [symbolic = %impl.elem1.loc126_5.2 (constants.%impl.elem1.fe2)]
// CHECK:STDOUT:   %specific_impl_fn.loc126_5.2: <specific function> = specific_impl_function %impl.elem1.loc126_5.2, @AddWith.WithSelf.Op(%U.as_type.loc123_35.1, %AddWith.facet) [symbolic = %specific_impl_fn.loc126_5.2 (constants.%specific_impl_fn.896)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%result.param: @AddWith.loc124.%Result.as_type.loc123_53.1 (%Result.as_type), %x.param: @AddWith.loc124.%T.as_type.loc124_29.1 (%T.as_type.d2e), %y.param: @AddWith.loc124.%U.as_type.loc123_35.1 (%U.as_type.0bd)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: @AddWith.loc124.%T.as_type.loc124_29.1 (%T.as_type.d2e) = name_ref x, %x
// CHECK:STDOUT:     %y.ref: @AddWith.loc124.%U.as_type.loc123_35.1 (%U.as_type.0bd) = name_ref y, %y
// CHECK:STDOUT:     %AddWith.type.loc126: type = facet_type <@AddWith.1, @AddWith.1(constants.%U.as_type.0bd)> [symbolic = %AddWith.type.loc123_35.1 (constants.%AddWith.type.4f5)]
// CHECK:STDOUT:     %.loc126_5.1: @AddWith.loc124.%AddWith.assoc_type (%AddWith.assoc_type.71f) = specific_constant imports.%Core.import_ref.5c64, @AddWith.WithSelf(constants.%U.as_type.0bd, constants.%Self.958) [symbolic = %assoc1 (constants.%assoc1.f40)]
// CHECK:STDOUT:     %Op.ref: @AddWith.loc124.%AddWith.assoc_type (%AddWith.assoc_type.71f) = name_ref Op, %.loc126_5.1 [symbolic = %assoc1 (constants.%assoc1.f40)]
// CHECK:STDOUT:     %impl.elem1.loc126_5.1: @AddWith.loc124.%.loc126_5.6 (%.a11) = impl_witness_access constants.%AddWith.lookup_impl_witness.206, element1 [symbolic = %impl.elem1.loc126_5.2 (constants.%impl.elem1.fe2)]
// CHECK:STDOUT:     %bound_method.loc126_5.1: <bound method> = bound_method %x.ref, %impl.elem1.loc126_5.1
// CHECK:STDOUT:     %specific_impl_fn.loc126_5.1: <specific function> = specific_impl_function %impl.elem1.loc126_5.1, @AddWith.WithSelf.Op(constants.%U.as_type.0bd, constants.%AddWith.facet.880) [symbolic = %specific_impl_fn.loc126_5.2 (constants.%specific_impl_fn.896)]
// CHECK:STDOUT:     %bound_method.loc126_5.2: <bound method> = bound_method %x.ref, %specific_impl_fn.loc126_5.1
// CHECK:STDOUT:     %.loc126_5.2: ref @AddWith.loc124.%Result.as_type.loc123_53.1 (%Result.as_type) = temporary_storage
// CHECK:STDOUT:     %AddWith.WithSelf.Op.call: init @AddWith.loc124.%Result.as_type.loc123_53.1 (%Result.as_type) to %.loc126_5.2 = call %bound_method.loc126_5.2(%x.ref, %y.ref)
// CHECK:STDOUT:     %.loc126_5.3: ref @AddWith.loc124.%Result.as_type.loc123_53.1 (%Result.as_type) = temporary %.loc126_5.2, %AddWith.WithSelf.Op.call
// CHECK:STDOUT:     %.loc126_5.4: %Destroy.type = converted constants.%Result.as_type, constants.%Result [symbolic = %Result.loc121_9.1 (constants.%Result)]
// CHECK:STDOUT:     %.loc126_5.5: %Destroy.type = converted constants.%Result.as_type, constants.%Result [symbolic = %Result.loc121_9.1 (constants.%Result)]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc126_5.3)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @SubWith.loc135(%Result.loc132_9.2: %Destroy.type, %U.loc133_4.2: %Destroy.type, %T.loc134_4.2: @SubWith.loc135.%facet_type.loc134_37 (%facet_type.0a9)) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %require_complete.loc137: <witness> = require_complete_type %SubWith.type.loc134_35.1 [symbolic = %require_complete.loc137 (constants.%require_complete.49f499.2)]
// CHECK:STDOUT:   %assoc1: @SubWith.loc135.%SubWith.assoc_type (%SubWith.assoc_type.5f0) = assoc_entity element1, imports.%Core.import_ref.2f4 [symbolic = %assoc1 (constants.%assoc1.a86)]
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.loc137: <witness> = lookup_impl_witness %T.loc134_4.1, @SubWith.1, @SubWith.1(%U.as_type.loc134_35.1) [symbolic = %SubWith.lookup_impl_witness.loc137 (constants.%SubWith.lookup_impl_witness.725)]
// CHECK:STDOUT:   %SubWith.facet: @SubWith.loc135.%SubWith.type.loc134_35.1 (%SubWith.type.769) = facet_value %T.as_type.loc135_29.1, (%SubWith.lookup_impl_witness.loc137) [symbolic = %SubWith.facet (constants.%SubWith.facet.910)]
// CHECK:STDOUT:   %SubWith.WithSelf.Op.type: type = fn_type @SubWith.WithSelf.Op, @SubWith.WithSelf(%U.as_type.loc134_35.1, %SubWith.facet) [symbolic = %SubWith.WithSelf.Op.type (constants.%SubWith.WithSelf.Op.type.772)]
// CHECK:STDOUT:   %.loc137_5.6: type = fn_type_with_self_type %SubWith.WithSelf.Op.type, %SubWith.facet [symbolic = %.loc137_5.6 (constants.%.47a6)]
// CHECK:STDOUT:   %impl.elem1.loc137_5.2: @SubWith.loc135.%.loc137_5.6 (%.47a6) = impl_witness_access %SubWith.lookup_impl_witness.loc137, element1 [symbolic = %impl.elem1.loc137_5.2 (constants.%impl.elem1.415)]
// CHECK:STDOUT:   %specific_impl_fn.loc137_5.2: <specific function> = specific_impl_function %impl.elem1.loc137_5.2, @SubWith.WithSelf.Op(%U.as_type.loc134_35.1, %SubWith.facet) [symbolic = %specific_impl_fn.loc137_5.2 (constants.%specific_impl_fn.a72)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%result.param: @SubWith.loc135.%Result.as_type.loc134_53.1 (%Result.as_type), %x.param: @SubWith.loc135.%T.as_type.loc135_29.1 (%T.as_type.4b7), %y.param: @SubWith.loc135.%U.as_type.loc134_35.1 (%U.as_type.0bd)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: @SubWith.loc135.%T.as_type.loc135_29.1 (%T.as_type.4b7) = name_ref x, %x
// CHECK:STDOUT:     %y.ref: @SubWith.loc135.%U.as_type.loc134_35.1 (%U.as_type.0bd) = name_ref y, %y
// CHECK:STDOUT:     %SubWith.type.loc137: type = facet_type <@SubWith.1, @SubWith.1(constants.%U.as_type.0bd)> [symbolic = %SubWith.type.loc134_35.1 (constants.%SubWith.type.769)]
// CHECK:STDOUT:     %.loc137_5.1: @SubWith.loc135.%SubWith.assoc_type (%SubWith.assoc_type.5f0) = specific_constant imports.%Core.import_ref.5fe, @SubWith.WithSelf(constants.%U.as_type.0bd, constants.%Self.18a) [symbolic = %assoc1 (constants.%assoc1.a86)]
// CHECK:STDOUT:     %Op.ref: @SubWith.loc135.%SubWith.assoc_type (%SubWith.assoc_type.5f0) = name_ref Op, %.loc137_5.1 [symbolic = %assoc1 (constants.%assoc1.a86)]
// CHECK:STDOUT:     %impl.elem1.loc137_5.1: @SubWith.loc135.%.loc137_5.6 (%.47a6) = impl_witness_access constants.%SubWith.lookup_impl_witness.725, element1 [symbolic = %impl.elem1.loc137_5.2 (constants.%impl.elem1.415)]
// CHECK:STDOUT:     %bound_method.loc137_5.1: <bound method> = bound_method %x.ref, %impl.elem1.loc137_5.1
// CHECK:STDOUT:     %specific_impl_fn.loc137_5.1: <specific function> = specific_impl_function %impl.elem1.loc137_5.1, @SubWith.WithSelf.Op(constants.%U.as_type.0bd, constants.%SubWith.facet.910) [symbolic = %specific_impl_fn.loc137_5.2 (constants.%specific_impl_fn.a72)]
// CHECK:STDOUT:     %bound_method.loc137_5.2: <bound method> = bound_method %x.ref, %specific_impl_fn.loc137_5.1
// CHECK:STDOUT:     %.loc137_5.2: ref @SubWith.loc135.%Result.as_type.loc134_53.1 (%Result.as_type) = temporary_storage
// CHECK:STDOUT:     %SubWith.WithSelf.Op.call: init @SubWith.loc135.%Result.as_type.loc134_53.1 (%Result.as_type) to %.loc137_5.2 = call %bound_method.loc137_5.2(%x.ref, %y.ref)
// CHECK:STDOUT:     %.loc137_5.3: ref @SubWith.loc135.%Result.as_type.loc134_53.1 (%Result.as_type) = temporary %.loc137_5.2, %SubWith.WithSelf.Op.call
// CHECK:STDOUT:     %.loc137_5.4: %Destroy.type = converted constants.%Result.as_type, constants.%Result [symbolic = %Result.loc132_9.1 (constants.%Result)]
// CHECK:STDOUT:     %.loc137_5.5: %Destroy.type = converted constants.%Result.as_type, constants.%Result [symbolic = %Result.loc132_9.1 (constants.%Result)]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc137_5.3)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @MulWith.loc146(%Result.loc143_9.2: %Destroy.type, %U.loc144_4.2: %Destroy.type, %T.loc145_4.2: @MulWith.loc146.%facet_type.loc145_37 (%facet_type.a5f)) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %require_complete.loc148: <witness> = require_complete_type %MulWith.type.loc145_35.1 [symbolic = %require_complete.loc148 (constants.%require_complete.8389d3.2)]
// CHECK:STDOUT:   %assoc1: @MulWith.loc146.%MulWith.assoc_type (%MulWith.assoc_type.2e5) = assoc_entity element1, imports.%Core.import_ref.fc2 [symbolic = %assoc1 (constants.%assoc1.4eb)]
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.loc148: <witness> = lookup_impl_witness %T.loc145_4.1, @MulWith.1, @MulWith.1(%U.as_type.loc145_35.1) [symbolic = %MulWith.lookup_impl_witness.loc148 (constants.%MulWith.lookup_impl_witness.50b)]
// CHECK:STDOUT:   %MulWith.facet: @MulWith.loc146.%MulWith.type.loc145_35.1 (%MulWith.type.b0b) = facet_value %T.as_type.loc146_29.1, (%MulWith.lookup_impl_witness.loc148) [symbolic = %MulWith.facet (constants.%MulWith.facet.756)]
// CHECK:STDOUT:   %MulWith.WithSelf.Op.type: type = fn_type @MulWith.WithSelf.Op, @MulWith.WithSelf(%U.as_type.loc145_35.1, %MulWith.facet) [symbolic = %MulWith.WithSelf.Op.type (constants.%MulWith.WithSelf.Op.type.689)]
// CHECK:STDOUT:   %.loc148_5.6: type = fn_type_with_self_type %MulWith.WithSelf.Op.type, %MulWith.facet [symbolic = %.loc148_5.6 (constants.%.7f0)]
// CHECK:STDOUT:   %impl.elem1.loc148_5.2: @MulWith.loc146.%.loc148_5.6 (%.7f0) = impl_witness_access %MulWith.lookup_impl_witness.loc148, element1 [symbolic = %impl.elem1.loc148_5.2 (constants.%impl.elem1.334)]
// CHECK:STDOUT:   %specific_impl_fn.loc148_5.2: <specific function> = specific_impl_function %impl.elem1.loc148_5.2, @MulWith.WithSelf.Op(%U.as_type.loc145_35.1, %MulWith.facet) [symbolic = %specific_impl_fn.loc148_5.2 (constants.%specific_impl_fn.2bd)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%result.param: @MulWith.loc146.%Result.as_type.loc145_53.1 (%Result.as_type), %x.param: @MulWith.loc146.%T.as_type.loc146_29.1 (%T.as_type.639), %y.param: @MulWith.loc146.%U.as_type.loc145_35.1 (%U.as_type.0bd)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: @MulWith.loc146.%T.as_type.loc146_29.1 (%T.as_type.639) = name_ref x, %x
// CHECK:STDOUT:     %y.ref: @MulWith.loc146.%U.as_type.loc145_35.1 (%U.as_type.0bd) = name_ref y, %y
// CHECK:STDOUT:     %MulWith.type.loc148: type = facet_type <@MulWith.1, @MulWith.1(constants.%U.as_type.0bd)> [symbolic = %MulWith.type.loc145_35.1 (constants.%MulWith.type.b0b)]
// CHECK:STDOUT:     %.loc148_5.1: @MulWith.loc146.%MulWith.assoc_type (%MulWith.assoc_type.2e5) = specific_constant imports.%Core.import_ref.0a8, @MulWith.WithSelf(constants.%U.as_type.0bd, constants.%Self.742) [symbolic = %assoc1 (constants.%assoc1.4eb)]
// CHECK:STDOUT:     %Op.ref: @MulWith.loc146.%MulWith.assoc_type (%MulWith.assoc_type.2e5) = name_ref Op, %.loc148_5.1 [symbolic = %assoc1 (constants.%assoc1.4eb)]
// CHECK:STDOUT:     %impl.elem1.loc148_5.1: @MulWith.loc146.%.loc148_5.6 (%.7f0) = impl_witness_access constants.%MulWith.lookup_impl_witness.50b, element1 [symbolic = %impl.elem1.loc148_5.2 (constants.%impl.elem1.334)]
// CHECK:STDOUT:     %bound_method.loc148_5.1: <bound method> = bound_method %x.ref, %impl.elem1.loc148_5.1
// CHECK:STDOUT:     %specific_impl_fn.loc148_5.1: <specific function> = specific_impl_function %impl.elem1.loc148_5.1, @MulWith.WithSelf.Op(constants.%U.as_type.0bd, constants.%MulWith.facet.756) [symbolic = %specific_impl_fn.loc148_5.2 (constants.%specific_impl_fn.2bd)]
// CHECK:STDOUT:     %bound_method.loc148_5.2: <bound method> = bound_method %x.ref, %specific_impl_fn.loc148_5.1
// CHECK:STDOUT:     %.loc148_5.2: ref @MulWith.loc146.%Result.as_type.loc145_53.1 (%Result.as_type) = temporary_storage
// CHECK:STDOUT:     %MulWith.WithSelf.Op.call: init @MulWith.loc146.%Result.as_type.loc145_53.1 (%Result.as_type) to %.loc148_5.2 = call %bound_method.loc148_5.2(%x.ref, %y.ref)
// CHECK:STDOUT:     %.loc148_5.3: ref @MulWith.loc146.%Result.as_type.loc145_53.1 (%Result.as_type) = temporary %.loc148_5.2, %MulWith.WithSelf.Op.call
// CHECK:STDOUT:     %.loc148_5.4: %Destroy.type = converted constants.%Result.as_type, constants.%Result [symbolic = %Result.loc143_9.1 (constants.%Result)]
// CHECK:STDOUT:     %.loc148_5.5: %Destroy.type = converted constants.%Result.as_type, constants.%Result [symbolic = %Result.loc143_9.1 (constants.%Result)]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc148_5.3)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @DivWith.loc157(%Result.loc154_9.2: %Destroy.type, %U.loc155_4.2: %Destroy.type, %T.loc156_4.2: @DivWith.loc157.%facet_type.loc156_37 (%facet_type.351)) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %require_complete.loc159: <witness> = require_complete_type %DivWith.type.loc156_35.1 [symbolic = %require_complete.loc159 (constants.%require_complete.c4013f.2)]
// CHECK:STDOUT:   %assoc1: @DivWith.loc157.%DivWith.assoc_type (%DivWith.assoc_type.e65) = assoc_entity element1, imports.%Core.import_ref.f2f [symbolic = %assoc1 (constants.%assoc1.ed1)]
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.loc159: <witness> = lookup_impl_witness %T.loc156_4.1, @DivWith.1, @DivWith.1(%U.as_type.loc156_35.1) [symbolic = %DivWith.lookup_impl_witness.loc159 (constants.%DivWith.lookup_impl_witness.e1e)]
// CHECK:STDOUT:   %DivWith.facet: @DivWith.loc157.%DivWith.type.loc156_35.1 (%DivWith.type.71d) = facet_value %T.as_type.loc157_29.1, (%DivWith.lookup_impl_witness.loc159) [symbolic = %DivWith.facet (constants.%DivWith.facet.538)]
// CHECK:STDOUT:   %DivWith.WithSelf.Op.type: type = fn_type @DivWith.WithSelf.Op, @DivWith.WithSelf(%U.as_type.loc156_35.1, %DivWith.facet) [symbolic = %DivWith.WithSelf.Op.type (constants.%DivWith.WithSelf.Op.type.663)]
// CHECK:STDOUT:   %.loc159_5.6: type = fn_type_with_self_type %DivWith.WithSelf.Op.type, %DivWith.facet [symbolic = %.loc159_5.6 (constants.%.0f9)]
// CHECK:STDOUT:   %impl.elem1.loc159_5.2: @DivWith.loc157.%.loc159_5.6 (%.0f9) = impl_witness_access %DivWith.lookup_impl_witness.loc159, element1 [symbolic = %impl.elem1.loc159_5.2 (constants.%impl.elem1.44e)]
// CHECK:STDOUT:   %specific_impl_fn.loc159_5.2: <specific function> = specific_impl_function %impl.elem1.loc159_5.2, @DivWith.WithSelf.Op(%U.as_type.loc156_35.1, %DivWith.facet) [symbolic = %specific_impl_fn.loc159_5.2 (constants.%specific_impl_fn.ea9)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%result.param: @DivWith.loc157.%Result.as_type.loc156_53.1 (%Result.as_type), %x.param: @DivWith.loc157.%T.as_type.loc157_29.1 (%T.as_type.6cf), %y.param: @DivWith.loc157.%U.as_type.loc156_35.1 (%U.as_type.0bd)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: @DivWith.loc157.%T.as_type.loc157_29.1 (%T.as_type.6cf) = name_ref x, %x
// CHECK:STDOUT:     %y.ref: @DivWith.loc157.%U.as_type.loc156_35.1 (%U.as_type.0bd) = name_ref y, %y
// CHECK:STDOUT:     %DivWith.type.loc159: type = facet_type <@DivWith.1, @DivWith.1(constants.%U.as_type.0bd)> [symbolic = %DivWith.type.loc156_35.1 (constants.%DivWith.type.71d)]
// CHECK:STDOUT:     %.loc159_5.1: @DivWith.loc157.%DivWith.assoc_type (%DivWith.assoc_type.e65) = specific_constant imports.%Core.import_ref.972, @DivWith.WithSelf(constants.%U.as_type.0bd, constants.%Self.5b3) [symbolic = %assoc1 (constants.%assoc1.ed1)]
// CHECK:STDOUT:     %Op.ref: @DivWith.loc157.%DivWith.assoc_type (%DivWith.assoc_type.e65) = name_ref Op, %.loc159_5.1 [symbolic = %assoc1 (constants.%assoc1.ed1)]
// CHECK:STDOUT:     %impl.elem1.loc159_5.1: @DivWith.loc157.%.loc159_5.6 (%.0f9) = impl_witness_access constants.%DivWith.lookup_impl_witness.e1e, element1 [symbolic = %impl.elem1.loc159_5.2 (constants.%impl.elem1.44e)]
// CHECK:STDOUT:     %bound_method.loc159_5.1: <bound method> = bound_method %x.ref, %impl.elem1.loc159_5.1
// CHECK:STDOUT:     %specific_impl_fn.loc159_5.1: <specific function> = specific_impl_function %impl.elem1.loc159_5.1, @DivWith.WithSelf.Op(constants.%U.as_type.0bd, constants.%DivWith.facet.538) [symbolic = %specific_impl_fn.loc159_5.2 (constants.%specific_impl_fn.ea9)]
// CHECK:STDOUT:     %bound_method.loc159_5.2: <bound method> = bound_method %x.ref, %specific_impl_fn.loc159_5.1
// CHECK:STDOUT:     %.loc159_5.2: ref @DivWith.loc157.%Result.as_type.loc156_53.1 (%Result.as_type) = temporary_storage
// CHECK:STDOUT:     %DivWith.WithSelf.Op.call: init @DivWith.loc157.%Result.as_type.loc156_53.1 (%Result.as_type) to %.loc159_5.2 = call %bound_method.loc159_5.2(%x.ref, %y.ref)
// CHECK:STDOUT:     %.loc159_5.3: ref @DivWith.loc157.%Result.as_type.loc156_53.1 (%Result.as_type) = temporary %.loc159_5.2, %DivWith.WithSelf.Op.call
// CHECK:STDOUT:     %.loc159_5.4: %Destroy.type = converted constants.%Result.as_type, constants.%Result [symbolic = %Result.loc154_9.1 (constants.%Result)]
// CHECK:STDOUT:     %.loc159_5.5: %Destroy.type = converted constants.%Result.as_type, constants.%Result [symbolic = %Result.loc154_9.1 (constants.%Result)]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc159_5.3)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @ModWith.loc168(%Result.loc165_9.2: %Destroy.type, %U.loc166_4.2: %Destroy.type, %T.loc167_4.2: @ModWith.loc168.%facet_type.loc167_37 (%facet_type.938)) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %require_complete.loc170: <witness> = require_complete_type %ModWith.type.loc167_35.1 [symbolic = %require_complete.loc170 (constants.%require_complete.312803.2)]
// CHECK:STDOUT:   %assoc1: @ModWith.loc168.%ModWith.assoc_type (%ModWith.assoc_type.9c3) = assoc_entity element1, imports.%Core.import_ref.9f8f [symbolic = %assoc1 (constants.%assoc1.3d6)]
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.loc170: <witness> = lookup_impl_witness %T.loc167_4.1, @ModWith.1, @ModWith.1(%U.as_type.loc167_35.1) [symbolic = %ModWith.lookup_impl_witness.loc170 (constants.%ModWith.lookup_impl_witness.c4b)]
// CHECK:STDOUT:   %ModWith.facet: @ModWith.loc168.%ModWith.type.loc167_35.1 (%ModWith.type.68a) = facet_value %T.as_type.loc168_29.1, (%ModWith.lookup_impl_witness.loc170) [symbolic = %ModWith.facet (constants.%ModWith.facet.297)]
// CHECK:STDOUT:   %ModWith.WithSelf.Op.type: type = fn_type @ModWith.WithSelf.Op, @ModWith.WithSelf(%U.as_type.loc167_35.1, %ModWith.facet) [symbolic = %ModWith.WithSelf.Op.type (constants.%ModWith.WithSelf.Op.type.639)]
// CHECK:STDOUT:   %.loc170_5.6: type = fn_type_with_self_type %ModWith.WithSelf.Op.type, %ModWith.facet [symbolic = %.loc170_5.6 (constants.%.100)]
// CHECK:STDOUT:   %impl.elem1.loc170_5.2: @ModWith.loc168.%.loc170_5.6 (%.100) = impl_witness_access %ModWith.lookup_impl_witness.loc170, element1 [symbolic = %impl.elem1.loc170_5.2 (constants.%impl.elem1.b51)]
// CHECK:STDOUT:   %specific_impl_fn.loc170_5.2: <specific function> = specific_impl_function %impl.elem1.loc170_5.2, @ModWith.WithSelf.Op(%U.as_type.loc167_35.1, %ModWith.facet) [symbolic = %specific_impl_fn.loc170_5.2 (constants.%specific_impl_fn.6e9)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%result.param: @ModWith.loc168.%Result.as_type.loc167_53.1 (%Result.as_type), %x.param: @ModWith.loc168.%T.as_type.loc168_29.1 (%T.as_type.73e), %y.param: @ModWith.loc168.%U.as_type.loc167_35.1 (%U.as_type.0bd)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: @ModWith.loc168.%T.as_type.loc168_29.1 (%T.as_type.73e) = name_ref x, %x
// CHECK:STDOUT:     %y.ref: @ModWith.loc168.%U.as_type.loc167_35.1 (%U.as_type.0bd) = name_ref y, %y
// CHECK:STDOUT:     %ModWith.type.loc170: type = facet_type <@ModWith.1, @ModWith.1(constants.%U.as_type.0bd)> [symbolic = %ModWith.type.loc167_35.1 (constants.%ModWith.type.68a)]
// CHECK:STDOUT:     %.loc170_5.1: @ModWith.loc168.%ModWith.assoc_type (%ModWith.assoc_type.9c3) = specific_constant imports.%Core.import_ref.ea5e3, @ModWith.WithSelf(constants.%U.as_type.0bd, constants.%Self.c18) [symbolic = %assoc1 (constants.%assoc1.3d6)]
// CHECK:STDOUT:     %Op.ref: @ModWith.loc168.%ModWith.assoc_type (%ModWith.assoc_type.9c3) = name_ref Op, %.loc170_5.1 [symbolic = %assoc1 (constants.%assoc1.3d6)]
// CHECK:STDOUT:     %impl.elem1.loc170_5.1: @ModWith.loc168.%.loc170_5.6 (%.100) = impl_witness_access constants.%ModWith.lookup_impl_witness.c4b, element1 [symbolic = %impl.elem1.loc170_5.2 (constants.%impl.elem1.b51)]
// CHECK:STDOUT:     %bound_method.loc170_5.1: <bound method> = bound_method %x.ref, %impl.elem1.loc170_5.1
// CHECK:STDOUT:     %specific_impl_fn.loc170_5.1: <specific function> = specific_impl_function %impl.elem1.loc170_5.1, @ModWith.WithSelf.Op(constants.%U.as_type.0bd, constants.%ModWith.facet.297) [symbolic = %specific_impl_fn.loc170_5.2 (constants.%specific_impl_fn.6e9)]
// CHECK:STDOUT:     %bound_method.loc170_5.2: <bound method> = bound_method %x.ref, %specific_impl_fn.loc170_5.1
// CHECK:STDOUT:     %.loc170_5.2: ref @ModWith.loc168.%Result.as_type.loc167_53.1 (%Result.as_type) = temporary_storage
// CHECK:STDOUT:     %ModWith.WithSelf.Op.call: init @ModWith.loc168.%Result.as_type.loc167_53.1 (%Result.as_type) to %.loc170_5.2 = call %bound_method.loc170_5.2(%x.ref, %y.ref)
// CHECK:STDOUT:     %.loc170_5.3: ref @ModWith.loc168.%Result.as_type.loc167_53.1 (%Result.as_type) = temporary %.loc170_5.2, %ModWith.WithSelf.Op.call
// CHECK:STDOUT:     %.loc170_5.4: %Destroy.type = converted constants.%Result.as_type, constants.%Result [symbolic = %Result.loc165_9.1 (constants.%Result)]
// CHECK:STDOUT:     %.loc170_5.5: %Destroy.type = converted constants.%Result.as_type, constants.%Result [symbolic = %Result.loc165_9.1 (constants.%Result)]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc170_5.3)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @AddAssignWith.loc225(%U.loc225_19.2: type, %T.loc225_28.2: @AddAssignWith.loc225.%AddAssignWith.type.loc225_50.1 (%AddAssignWith.type.3a470e.2)) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %require_complete.loc227: <witness> = require_complete_type %AddAssignWith.type.loc225_50.1 [symbolic = %require_complete.loc227 (constants.%require_complete.a80)]
// CHECK:STDOUT:   %AddAssignWith.assoc_type: type = assoc_entity_type @AddAssignWith.1, @AddAssignWith.1(%U.loc225_19.1) [symbolic = %AddAssignWith.assoc_type (constants.%AddAssignWith.assoc_type.08473b.2)]
// CHECK:STDOUT:   %assoc0: @AddAssignWith.loc225.%AddAssignWith.assoc_type (%AddAssignWith.assoc_type.08473b.2) = assoc_entity element0, imports.%Core.import_ref.90f [symbolic = %assoc0 (constants.%assoc0.0fc488.2)]
// CHECK:STDOUT:   %AddAssignWith.WithSelf.Op.type: type = fn_type @AddAssignWith.WithSelf.Op, @AddAssignWith.WithSelf(%U.loc225_19.1, %T.loc225_28.1) [symbolic = %AddAssignWith.WithSelf.Op.type (constants.%AddAssignWith.WithSelf.Op.type.9de7cb.3)]
// CHECK:STDOUT:   %.loc227_5.2: type = fn_type_with_self_type %AddAssignWith.WithSelf.Op.type, %T.loc225_28.1 [symbolic = %.loc227_5.2 (constants.%.643)]
// CHECK:STDOUT:   %AddAssignWith.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc225_28.1, @AddAssignWith.1, @AddAssignWith.1(%U.loc225_19.1) [symbolic = %AddAssignWith.lookup_impl_witness (constants.%AddAssignWith.lookup_impl_witness)]
// CHECK:STDOUT:   %impl.elem0.loc227_5.2: @AddAssignWith.loc225.%.loc227_5.2 (%.643) = impl_witness_access %AddAssignWith.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc227_5.2 (constants.%impl.elem0.20b)]
// CHECK:STDOUT:   %specific_impl_fn.loc227_5.2: <specific function> = specific_impl_function %impl.elem0.loc227_5.2, @AddAssignWith.WithSelf.Op(%U.loc225_19.1, %T.loc225_28.1) [symbolic = %specific_impl_fn.loc227_5.2 (constants.%specific_impl_fn.03f)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: ref @AddAssignWith.loc225.%T.as_type.loc225_60.1 (%T.as_type.415), %y.param: @AddAssignWith.loc225.%U.loc225_19.1 (%U.67d)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: ref @AddAssignWith.loc225.%T.as_type.loc225_60.1 (%T.as_type.415) = name_ref x, %x
// CHECK:STDOUT:     %y.ref: @AddAssignWith.loc225.%U.loc225_19.1 (%U.67d) = name_ref y, %y
// CHECK:STDOUT:     %AddAssignWith.type.loc227: type = facet_type <@AddAssignWith.1, @AddAssignWith.1(constants.%U.67d)> [symbolic = %AddAssignWith.type.loc225_50.1 (constants.%AddAssignWith.type.3a470e.2)]
// CHECK:STDOUT:     %.loc227_5.1: @AddAssignWith.loc225.%AddAssignWith.assoc_type (%AddAssignWith.assoc_type.08473b.2) = specific_constant imports.%Core.import_ref.36d, @AddAssignWith.WithSelf(constants.%U.67d, constants.%Self.73bb96.1) [symbolic = %assoc0 (constants.%assoc0.0fc488.2)]
// CHECK:STDOUT:     %Op.ref: @AddAssignWith.loc225.%AddAssignWith.assoc_type (%AddAssignWith.assoc_type.08473b.2) = name_ref Op, %.loc227_5.1 [symbolic = %assoc0 (constants.%assoc0.0fc488.2)]
// CHECK:STDOUT:     %impl.elem0.loc227_5.1: @AddAssignWith.loc225.%.loc227_5.2 (%.643) = impl_witness_access constants.%AddAssignWith.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc227_5.2 (constants.%impl.elem0.20b)]
// CHECK:STDOUT:     %bound_method.loc227_5.1: <bound method> = bound_method %x.ref, %impl.elem0.loc227_5.1
// CHECK:STDOUT:     %specific_impl_fn.loc227_5.1: <specific function> = specific_impl_function %impl.elem0.loc227_5.1, @AddAssignWith.WithSelf.Op(constants.%U.67d, constants.%T.73b) [symbolic = %specific_impl_fn.loc227_5.2 (constants.%specific_impl_fn.03f)]
// CHECK:STDOUT:     %bound_method.loc227_5.2: <bound method> = bound_method %x.ref, %specific_impl_fn.loc227_5.1
// CHECK:STDOUT:     %AddAssignWith.WithSelf.Op.call: init %empty_tuple.type = call %bound_method.loc227_5.2(%x.ref, %y.ref)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @SubAssignWith.loc231(%U.loc231_19.2: type, %T.loc231_28.2: @SubAssignWith.loc231.%SubAssignWith.type.loc231_50.1 (%SubAssignWith.type.0c61b3.2)) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %require_complete.loc233: <witness> = require_complete_type %SubAssignWith.type.loc231_50.1 [symbolic = %require_complete.loc233 (constants.%require_complete.b01)]
// CHECK:STDOUT:   %SubAssignWith.assoc_type: type = assoc_entity_type @SubAssignWith.1, @SubAssignWith.1(%U.loc231_19.1) [symbolic = %SubAssignWith.assoc_type (constants.%SubAssignWith.assoc_type.8211d1.2)]
// CHECK:STDOUT:   %assoc0: @SubAssignWith.loc231.%SubAssignWith.assoc_type (%SubAssignWith.assoc_type.8211d1.2) = assoc_entity element0, imports.%Core.import_ref.8e5 [symbolic = %assoc0 (constants.%assoc0.0ba1dc.2)]
// CHECK:STDOUT:   %SubAssignWith.WithSelf.Op.type: type = fn_type @SubAssignWith.WithSelf.Op, @SubAssignWith.WithSelf(%U.loc231_19.1, %T.loc231_28.1) [symbolic = %SubAssignWith.WithSelf.Op.type (constants.%SubAssignWith.WithSelf.Op.type.7e787a.3)]
// CHECK:STDOUT:   %.loc233_5.2: type = fn_type_with_self_type %SubAssignWith.WithSelf.Op.type, %T.loc231_28.1 [symbolic = %.loc233_5.2 (constants.%.dcf)]
// CHECK:STDOUT:   %SubAssignWith.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc231_28.1, @SubAssignWith.1, @SubAssignWith.1(%U.loc231_19.1) [symbolic = %SubAssignWith.lookup_impl_witness (constants.%SubAssignWith.lookup_impl_witness)]
// CHECK:STDOUT:   %impl.elem0.loc233_5.2: @SubAssignWith.loc231.%.loc233_5.2 (%.dcf) = impl_witness_access %SubAssignWith.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc233_5.2 (constants.%impl.elem0.8f6)]
// CHECK:STDOUT:   %specific_impl_fn.loc233_5.2: <specific function> = specific_impl_function %impl.elem0.loc233_5.2, @SubAssignWith.WithSelf.Op(%U.loc231_19.1, %T.loc231_28.1) [symbolic = %specific_impl_fn.loc233_5.2 (constants.%specific_impl_fn.b79)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: ref @SubAssignWith.loc231.%T.as_type.loc231_60.1 (%T.as_type.540), %y.param: @SubAssignWith.loc231.%U.loc231_19.1 (%U.67d)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: ref @SubAssignWith.loc231.%T.as_type.loc231_60.1 (%T.as_type.540) = name_ref x, %x
// CHECK:STDOUT:     %y.ref: @SubAssignWith.loc231.%U.loc231_19.1 (%U.67d) = name_ref y, %y
// CHECK:STDOUT:     %SubAssignWith.type.loc233: type = facet_type <@SubAssignWith.1, @SubAssignWith.1(constants.%U.67d)> [symbolic = %SubAssignWith.type.loc231_50.1 (constants.%SubAssignWith.type.0c61b3.2)]
// CHECK:STDOUT:     %.loc233_5.1: @SubAssignWith.loc231.%SubAssignWith.assoc_type (%SubAssignWith.assoc_type.8211d1.2) = specific_constant imports.%Core.import_ref.1f6, @SubAssignWith.WithSelf(constants.%U.67d, constants.%Self.0dd047.1) [symbolic = %assoc0 (constants.%assoc0.0ba1dc.2)]
// CHECK:STDOUT:     %Op.ref: @SubAssignWith.loc231.%SubAssignWith.assoc_type (%SubAssignWith.assoc_type.8211d1.2) = name_ref Op, %.loc233_5.1 [symbolic = %assoc0 (constants.%assoc0.0ba1dc.2)]
// CHECK:STDOUT:     %impl.elem0.loc233_5.1: @SubAssignWith.loc231.%.loc233_5.2 (%.dcf) = impl_witness_access constants.%SubAssignWith.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc233_5.2 (constants.%impl.elem0.8f6)]
// CHECK:STDOUT:     %bound_method.loc233_5.1: <bound method> = bound_method %x.ref, %impl.elem0.loc233_5.1
// CHECK:STDOUT:     %specific_impl_fn.loc233_5.1: <specific function> = specific_impl_function %impl.elem0.loc233_5.1, @SubAssignWith.WithSelf.Op(constants.%U.67d, constants.%T.0dd) [symbolic = %specific_impl_fn.loc233_5.2 (constants.%specific_impl_fn.b79)]
// CHECK:STDOUT:     %bound_method.loc233_5.2: <bound method> = bound_method %x.ref, %specific_impl_fn.loc233_5.1
// CHECK:STDOUT:     %SubAssignWith.WithSelf.Op.call: init %empty_tuple.type = call %bound_method.loc233_5.2(%x.ref, %y.ref)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @MulAssignWith.loc237(%U.loc237_19.2: type, %T.loc237_28.2: @MulAssignWith.loc237.%MulAssignWith.type.loc237_50.1 (%MulAssignWith.type.2339fd.2)) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %require_complete.loc239: <witness> = require_complete_type %MulAssignWith.type.loc237_50.1 [symbolic = %require_complete.loc239 (constants.%require_complete.422)]
// CHECK:STDOUT:   %MulAssignWith.assoc_type: type = assoc_entity_type @MulAssignWith.1, @MulAssignWith.1(%U.loc237_19.1) [symbolic = %MulAssignWith.assoc_type (constants.%MulAssignWith.assoc_type.5adc6f.2)]
// CHECK:STDOUT:   %assoc0: @MulAssignWith.loc237.%MulAssignWith.assoc_type (%MulAssignWith.assoc_type.5adc6f.2) = assoc_entity element0, imports.%Core.import_ref.c116 [symbolic = %assoc0 (constants.%assoc0.42a576.2)]
// CHECK:STDOUT:   %MulAssignWith.WithSelf.Op.type: type = fn_type @MulAssignWith.WithSelf.Op, @MulAssignWith.WithSelf(%U.loc237_19.1, %T.loc237_28.1) [symbolic = %MulAssignWith.WithSelf.Op.type (constants.%MulAssignWith.WithSelf.Op.type.c4075c.3)]
// CHECK:STDOUT:   %.loc239_5.2: type = fn_type_with_self_type %MulAssignWith.WithSelf.Op.type, %T.loc237_28.1 [symbolic = %.loc239_5.2 (constants.%.66e)]
// CHECK:STDOUT:   %MulAssignWith.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc237_28.1, @MulAssignWith.1, @MulAssignWith.1(%U.loc237_19.1) [symbolic = %MulAssignWith.lookup_impl_witness (constants.%MulAssignWith.lookup_impl_witness)]
// CHECK:STDOUT:   %impl.elem0.loc239_5.2: @MulAssignWith.loc237.%.loc239_5.2 (%.66e) = impl_witness_access %MulAssignWith.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc239_5.2 (constants.%impl.elem0.1f5)]
// CHECK:STDOUT:   %specific_impl_fn.loc239_5.2: <specific function> = specific_impl_function %impl.elem0.loc239_5.2, @MulAssignWith.WithSelf.Op(%U.loc237_19.1, %T.loc237_28.1) [symbolic = %specific_impl_fn.loc239_5.2 (constants.%specific_impl_fn.817)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: ref @MulAssignWith.loc237.%T.as_type.loc237_60.1 (%T.as_type.3e8), %y.param: @MulAssignWith.loc237.%U.loc237_19.1 (%U.67d)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: ref @MulAssignWith.loc237.%T.as_type.loc237_60.1 (%T.as_type.3e8) = name_ref x, %x
// CHECK:STDOUT:     %y.ref: @MulAssignWith.loc237.%U.loc237_19.1 (%U.67d) = name_ref y, %y
// CHECK:STDOUT:     %MulAssignWith.type.loc239: type = facet_type <@MulAssignWith.1, @MulAssignWith.1(constants.%U.67d)> [symbolic = %MulAssignWith.type.loc237_50.1 (constants.%MulAssignWith.type.2339fd.2)]
// CHECK:STDOUT:     %.loc239_5.1: @MulAssignWith.loc237.%MulAssignWith.assoc_type (%MulAssignWith.assoc_type.5adc6f.2) = specific_constant imports.%Core.import_ref.575, @MulAssignWith.WithSelf(constants.%U.67d, constants.%Self.9e408c.1) [symbolic = %assoc0 (constants.%assoc0.42a576.2)]
// CHECK:STDOUT:     %Op.ref: @MulAssignWith.loc237.%MulAssignWith.assoc_type (%MulAssignWith.assoc_type.5adc6f.2) = name_ref Op, %.loc239_5.1 [symbolic = %assoc0 (constants.%assoc0.42a576.2)]
// CHECK:STDOUT:     %impl.elem0.loc239_5.1: @MulAssignWith.loc237.%.loc239_5.2 (%.66e) = impl_witness_access constants.%MulAssignWith.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc239_5.2 (constants.%impl.elem0.1f5)]
// CHECK:STDOUT:     %bound_method.loc239_5.1: <bound method> = bound_method %x.ref, %impl.elem0.loc239_5.1
// CHECK:STDOUT:     %specific_impl_fn.loc239_5.1: <specific function> = specific_impl_function %impl.elem0.loc239_5.1, @MulAssignWith.WithSelf.Op(constants.%U.67d, constants.%T.9e4) [symbolic = %specific_impl_fn.loc239_5.2 (constants.%specific_impl_fn.817)]
// CHECK:STDOUT:     %bound_method.loc239_5.2: <bound method> = bound_method %x.ref, %specific_impl_fn.loc239_5.1
// CHECK:STDOUT:     %MulAssignWith.WithSelf.Op.call: init %empty_tuple.type = call %bound_method.loc239_5.2(%x.ref, %y.ref)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @DivAssignWith.loc243(%U.loc243_19.2: type, %T.loc243_28.2: @DivAssignWith.loc243.%DivAssignWith.type.loc243_50.1 (%DivAssignWith.type.4b7503.2)) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %require_complete.loc245: <witness> = require_complete_type %DivAssignWith.type.loc243_50.1 [symbolic = %require_complete.loc245 (constants.%require_complete.5fe)]
// CHECK:STDOUT:   %DivAssignWith.assoc_type: type = assoc_entity_type @DivAssignWith.1, @DivAssignWith.1(%U.loc243_19.1) [symbolic = %DivAssignWith.assoc_type (constants.%DivAssignWith.assoc_type.4e8a63.2)]
// CHECK:STDOUT:   %assoc0: @DivAssignWith.loc243.%DivAssignWith.assoc_type (%DivAssignWith.assoc_type.4e8a63.2) = assoc_entity element0, imports.%Core.import_ref.ee5 [symbolic = %assoc0 (constants.%assoc0.ce2908.2)]
// CHECK:STDOUT:   %DivAssignWith.WithSelf.Op.type: type = fn_type @DivAssignWith.WithSelf.Op, @DivAssignWith.WithSelf(%U.loc243_19.1, %T.loc243_28.1) [symbolic = %DivAssignWith.WithSelf.Op.type (constants.%DivAssignWith.WithSelf.Op.type.42bffc.3)]
// CHECK:STDOUT:   %.loc245_5.2: type = fn_type_with_self_type %DivAssignWith.WithSelf.Op.type, %T.loc243_28.1 [symbolic = %.loc245_5.2 (constants.%.caf)]
// CHECK:STDOUT:   %DivAssignWith.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc243_28.1, @DivAssignWith.1, @DivAssignWith.1(%U.loc243_19.1) [symbolic = %DivAssignWith.lookup_impl_witness (constants.%DivAssignWith.lookup_impl_witness)]
// CHECK:STDOUT:   %impl.elem0.loc245_5.2: @DivAssignWith.loc243.%.loc245_5.2 (%.caf) = impl_witness_access %DivAssignWith.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc245_5.2 (constants.%impl.elem0.c50)]
// CHECK:STDOUT:   %specific_impl_fn.loc245_5.2: <specific function> = specific_impl_function %impl.elem0.loc245_5.2, @DivAssignWith.WithSelf.Op(%U.loc243_19.1, %T.loc243_28.1) [symbolic = %specific_impl_fn.loc245_5.2 (constants.%specific_impl_fn.5a8)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: ref @DivAssignWith.loc243.%T.as_type.loc243_60.1 (%T.as_type.cf7), %y.param: @DivAssignWith.loc243.%U.loc243_19.1 (%U.67d)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: ref @DivAssignWith.loc243.%T.as_type.loc243_60.1 (%T.as_type.cf7) = name_ref x, %x
// CHECK:STDOUT:     %y.ref: @DivAssignWith.loc243.%U.loc243_19.1 (%U.67d) = name_ref y, %y
// CHECK:STDOUT:     %DivAssignWith.type.loc245: type = facet_type <@DivAssignWith.1, @DivAssignWith.1(constants.%U.67d)> [symbolic = %DivAssignWith.type.loc243_50.1 (constants.%DivAssignWith.type.4b7503.2)]
// CHECK:STDOUT:     %.loc245_5.1: @DivAssignWith.loc243.%DivAssignWith.assoc_type (%DivAssignWith.assoc_type.4e8a63.2) = specific_constant imports.%Core.import_ref.5dd, @DivAssignWith.WithSelf(constants.%U.67d, constants.%Self.5bbd7a.1) [symbolic = %assoc0 (constants.%assoc0.ce2908.2)]
// CHECK:STDOUT:     %Op.ref: @DivAssignWith.loc243.%DivAssignWith.assoc_type (%DivAssignWith.assoc_type.4e8a63.2) = name_ref Op, %.loc245_5.1 [symbolic = %assoc0 (constants.%assoc0.ce2908.2)]
// CHECK:STDOUT:     %impl.elem0.loc245_5.1: @DivAssignWith.loc243.%.loc245_5.2 (%.caf) = impl_witness_access constants.%DivAssignWith.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc245_5.2 (constants.%impl.elem0.c50)]
// CHECK:STDOUT:     %bound_method.loc245_5.1: <bound method> = bound_method %x.ref, %impl.elem0.loc245_5.1
// CHECK:STDOUT:     %specific_impl_fn.loc245_5.1: <specific function> = specific_impl_function %impl.elem0.loc245_5.1, @DivAssignWith.WithSelf.Op(constants.%U.67d, constants.%T.5bb) [symbolic = %specific_impl_fn.loc245_5.2 (constants.%specific_impl_fn.5a8)]
// CHECK:STDOUT:     %bound_method.loc245_5.2: <bound method> = bound_method %x.ref, %specific_impl_fn.loc245_5.1
// CHECK:STDOUT:     %DivAssignWith.WithSelf.Op.call: init %empty_tuple.type = call %bound_method.loc245_5.2(%x.ref, %y.ref)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @ModAssignWith.loc249(%U.loc249_19.2: type, %T.loc249_28.2: @ModAssignWith.loc249.%ModAssignWith.type.loc249_50.1 (%ModAssignWith.type.3a0f29.2)) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %require_complete.loc251: <witness> = require_complete_type %ModAssignWith.type.loc249_50.1 [symbolic = %require_complete.loc251 (constants.%require_complete.a87)]
// CHECK:STDOUT:   %ModAssignWith.assoc_type: type = assoc_entity_type @ModAssignWith.1, @ModAssignWith.1(%U.loc249_19.1) [symbolic = %ModAssignWith.assoc_type (constants.%ModAssignWith.assoc_type.01d13e.2)]
// CHECK:STDOUT:   %assoc0: @ModAssignWith.loc249.%ModAssignWith.assoc_type (%ModAssignWith.assoc_type.01d13e.2) = assoc_entity element0, imports.%Core.import_ref.97f [symbolic = %assoc0 (constants.%assoc0.a11270.2)]
// CHECK:STDOUT:   %ModAssignWith.WithSelf.Op.type: type = fn_type @ModAssignWith.WithSelf.Op, @ModAssignWith.WithSelf(%U.loc249_19.1, %T.loc249_28.1) [symbolic = %ModAssignWith.WithSelf.Op.type (constants.%ModAssignWith.WithSelf.Op.type.7f247d.3)]
// CHECK:STDOUT:   %.loc251_5.2: type = fn_type_with_self_type %ModAssignWith.WithSelf.Op.type, %T.loc249_28.1 [symbolic = %.loc251_5.2 (constants.%.99f)]
// CHECK:STDOUT:   %ModAssignWith.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc249_28.1, @ModAssignWith.1, @ModAssignWith.1(%U.loc249_19.1) [symbolic = %ModAssignWith.lookup_impl_witness (constants.%ModAssignWith.lookup_impl_witness)]
// CHECK:STDOUT:   %impl.elem0.loc251_5.2: @ModAssignWith.loc249.%.loc251_5.2 (%.99f) = impl_witness_access %ModAssignWith.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc251_5.2 (constants.%impl.elem0.46a)]
// CHECK:STDOUT:   %specific_impl_fn.loc251_5.2: <specific function> = specific_impl_function %impl.elem0.loc251_5.2, @ModAssignWith.WithSelf.Op(%U.loc249_19.1, %T.loc249_28.1) [symbolic = %specific_impl_fn.loc251_5.2 (constants.%specific_impl_fn.caf)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: ref @ModAssignWith.loc249.%T.as_type.loc249_60.1 (%T.as_type.739), %y.param: @ModAssignWith.loc249.%U.loc249_19.1 (%U.67d)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: ref @ModAssignWith.loc249.%T.as_type.loc249_60.1 (%T.as_type.739) = name_ref x, %x
// CHECK:STDOUT:     %y.ref: @ModAssignWith.loc249.%U.loc249_19.1 (%U.67d) = name_ref y, %y
// CHECK:STDOUT:     %ModAssignWith.type.loc251: type = facet_type <@ModAssignWith.1, @ModAssignWith.1(constants.%U.67d)> [symbolic = %ModAssignWith.type.loc249_50.1 (constants.%ModAssignWith.type.3a0f29.2)]
// CHECK:STDOUT:     %.loc251_5.1: @ModAssignWith.loc249.%ModAssignWith.assoc_type (%ModAssignWith.assoc_type.01d13e.2) = specific_constant imports.%Core.import_ref.1a2, @ModAssignWith.WithSelf(constants.%U.67d, constants.%Self.cc0e96.1) [symbolic = %assoc0 (constants.%assoc0.a11270.2)]
// CHECK:STDOUT:     %Op.ref: @ModAssignWith.loc249.%ModAssignWith.assoc_type (%ModAssignWith.assoc_type.01d13e.2) = name_ref Op, %.loc251_5.1 [symbolic = %assoc0 (constants.%assoc0.a11270.2)]
// CHECK:STDOUT:     %impl.elem0.loc251_5.1: @ModAssignWith.loc249.%.loc251_5.2 (%.99f) = impl_witness_access constants.%ModAssignWith.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc251_5.2 (constants.%impl.elem0.46a)]
// CHECK:STDOUT:     %bound_method.loc251_5.1: <bound method> = bound_method %x.ref, %impl.elem0.loc251_5.1
// CHECK:STDOUT:     %specific_impl_fn.loc251_5.1: <specific function> = specific_impl_function %impl.elem0.loc251_5.1, @ModAssignWith.WithSelf.Op(constants.%U.67d, constants.%T.cc0) [symbolic = %specific_impl_fn.loc251_5.2 (constants.%specific_impl_fn.caf)]
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
// CHECK:STDOUT:   %Inc.WithSelf.Op.type: type = fn_type @Inc.WithSelf.Op, @Inc.WithSelf(%T.loc278_13.1) [symbolic = %Inc.WithSelf.Op.type (constants.%Inc.WithSelf.Op.type.877857.2)]
// CHECK:STDOUT:   %.loc280_3.3: type = fn_type_with_self_type %Inc.WithSelf.Op.type, %T.loc278_13.1 [symbolic = %.loc280_3.3 (constants.%.5a2b)]
// CHECK:STDOUT:   %Inc.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc278_13.1, @Inc [symbolic = %Inc.lookup_impl_witness (constants.%Inc.lookup_impl_witness)]
// CHECK:STDOUT:   %impl.elem0.loc280_3.2: @TestInc.%.loc280_3.3 (%.5a2b) = impl_witness_access %Inc.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc280_3.2 (constants.%impl.elem0.ec8)]
// CHECK:STDOUT:   %specific_impl_fn.loc280_3.2: <specific function> = specific_impl_function %impl.elem0.loc280_3.2, @Inc.WithSelf.Op(%T.loc278_13.1) [symbolic = %specific_impl_fn.loc280_3.2 (constants.%specific_impl_fn.790)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: ref @TestInc.%T.as_type.loc278_32.1 (%T.as_type.232)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: ref @TestInc.%T.as_type.loc278_32.1 (%T.as_type.232) = name_ref x, %x
// CHECK:STDOUT:     %impl.elem0.loc280_3.1: @TestInc.%.loc280_3.3 (%.5a2b) = impl_witness_access constants.%Inc.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc280_3.2 (constants.%impl.elem0.ec8)]
// CHECK:STDOUT:     %bound_method.loc280_3.1: <bound method> = bound_method %x.ref, %impl.elem0.loc280_3.1
// CHECK:STDOUT:     %.loc280_3.1: %Inc.type = converted constants.%T.as_type.232, constants.%T.8f3 [symbolic = %T.loc278_13.1 (constants.%T.8f3)]
// CHECK:STDOUT:     %.loc280_3.2: %Inc.type = converted constants.%T.as_type.232, constants.%T.8f3 [symbolic = %T.loc278_13.1 (constants.%T.8f3)]
// CHECK:STDOUT:     %specific_impl_fn.loc280_3.1: <specific function> = specific_impl_function %impl.elem0.loc280_3.1, @Inc.WithSelf.Op(constants.%T.8f3) [symbolic = %specific_impl_fn.loc280_3.2 (constants.%specific_impl_fn.790)]
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
// CHECK:STDOUT:   %Dec.WithSelf.Op.type: type = fn_type @Dec.WithSelf.Op, @Dec.WithSelf(%T.loc284_13.1) [symbolic = %Dec.WithSelf.Op.type (constants.%Dec.WithSelf.Op.type.dbc4d2.2)]
// CHECK:STDOUT:   %.loc286_3.3: type = fn_type_with_self_type %Dec.WithSelf.Op.type, %T.loc284_13.1 [symbolic = %.loc286_3.3 (constants.%.528)]
// CHECK:STDOUT:   %Dec.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc284_13.1, @Dec [symbolic = %Dec.lookup_impl_witness (constants.%Dec.lookup_impl_witness)]
// CHECK:STDOUT:   %impl.elem0.loc286_3.2: @TestDec.%.loc286_3.3 (%.528) = impl_witness_access %Dec.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc286_3.2 (constants.%impl.elem0.0b1)]
// CHECK:STDOUT:   %specific_impl_fn.loc286_3.2: <specific function> = specific_impl_function %impl.elem0.loc286_3.2, @Dec.WithSelf.Op(%T.loc284_13.1) [symbolic = %specific_impl_fn.loc286_3.2 (constants.%specific_impl_fn.e97)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: ref @TestDec.%T.as_type.loc284_32.1 (%T.as_type.c32)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: ref @TestDec.%T.as_type.loc284_32.1 (%T.as_type.c32) = name_ref x, %x
// CHECK:STDOUT:     %impl.elem0.loc286_3.1: @TestDec.%.loc286_3.3 (%.528) = impl_witness_access constants.%Dec.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc286_3.2 (constants.%impl.elem0.0b1)]
// CHECK:STDOUT:     %bound_method.loc286_3.1: <bound method> = bound_method %x.ref, %impl.elem0.loc286_3.1
// CHECK:STDOUT:     %.loc286_3.1: %Dec.type = converted constants.%T.as_type.c32, constants.%T.e8e [symbolic = %T.loc284_13.1 (constants.%T.e8e)]
// CHECK:STDOUT:     %.loc286_3.2: %Dec.type = converted constants.%T.as_type.c32, constants.%T.e8e [symbolic = %T.loc284_13.1 (constants.%T.e8e)]
// CHECK:STDOUT:     %specific_impl_fn.loc286_3.1: <specific function> = specific_impl_function %impl.elem0.loc286_3.1, @Dec.WithSelf.Op(constants.%T.e8e) [symbolic = %specific_impl_fn.loc286_3.2 (constants.%specific_impl_fn.e97)]
// CHECK:STDOUT:     %bound_method.loc286_3.2: <bound method> = bound_method %x.ref, %specific_impl_fn.loc286_3.1
// CHECK:STDOUT:     %Dec.WithSelf.Op.call: init %empty_tuple.type = call %bound_method.loc286_3.2(%x.ref)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @TestNegate(%Result.loc292_9.2: %Destroy.type, %T.loc293_4.2: @TestNegate.%facet_type (%facet_type.86f)) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Negate.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc293_4.1, @Negate [symbolic = %Negate.lookup_impl_witness (constants.%Negate.lookup_impl_witness.75a)]
// CHECK:STDOUT:   %Negate.facet.loc296_3.3: %Negate.type = facet_value %T.as_type.loc294_24.1, (%Negate.lookup_impl_witness) [symbolic = %Negate.facet.loc296_3.3 (constants.%Negate.facet.b7e)]
// CHECK:STDOUT:   %Negate.WithSelf.Op.type: type = fn_type @Negate.WithSelf.Op, @Negate.WithSelf(%Negate.facet.loc296_3.3) [symbolic = %Negate.WithSelf.Op.type (constants.%Negate.WithSelf.Op.type.4d1)]
// CHECK:STDOUT:   %.loc296_3.7: type = fn_type_with_self_type %Negate.WithSelf.Op.type, %Negate.facet.loc296_3.3 [symbolic = %.loc296_3.7 (constants.%.70d)]
// CHECK:STDOUT:   %impl.elem1.loc296_3.2: @TestNegate.%.loc296_3.7 (%.70d) = impl_witness_access %Negate.lookup_impl_witness, element1 [symbolic = %impl.elem1.loc296_3.2 (constants.%impl.elem1.466)]
// CHECK:STDOUT:   %specific_impl_fn.loc296_3.2: <specific function> = specific_impl_function %impl.elem1.loc296_3.2, @Negate.WithSelf.Op(%Negate.facet.loc296_3.3) [symbolic = %specific_impl_fn.loc296_3.2 (constants.%specific_impl_fn.9ba)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%y.param: @TestNegate.%Result.as_type.loc293_49.1 (%Result.as_type), %x.param: @TestNegate.%T.as_type.loc294_24.1 (%T.as_type.735)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: @TestNegate.%T.as_type.loc294_24.1 (%T.as_type.735) = name_ref x, %x
// CHECK:STDOUT:     %impl.elem1.loc296_3.1: @TestNegate.%.loc296_3.7 (%.70d) = impl_witness_access constants.%Negate.lookup_impl_witness.75a, element1 [symbolic = %impl.elem1.loc296_3.2 (constants.%impl.elem1.466)]
// CHECK:STDOUT:     %bound_method.loc296_3.1: <bound method> = bound_method %x.ref, %impl.elem1.loc296_3.1
// CHECK:STDOUT:     %Negate.facet.loc296_3.1: %Negate.type = facet_value constants.%T.as_type.735, (constants.%Negate.lookup_impl_witness.75a) [symbolic = %Negate.facet.loc296_3.3 (constants.%Negate.facet.b7e)]
// CHECK:STDOUT:     %.loc296_3.1: %Negate.type = converted constants.%T.as_type.735, %Negate.facet.loc296_3.1 [symbolic = %Negate.facet.loc296_3.3 (constants.%Negate.facet.b7e)]
// CHECK:STDOUT:     %Negate.facet.loc296_3.2: %Negate.type = facet_value constants.%T.as_type.735, (constants.%Negate.lookup_impl_witness.75a) [symbolic = %Negate.facet.loc296_3.3 (constants.%Negate.facet.b7e)]
// CHECK:STDOUT:     %.loc296_3.2: %Negate.type = converted constants.%T.as_type.735, %Negate.facet.loc296_3.2 [symbolic = %Negate.facet.loc296_3.3 (constants.%Negate.facet.b7e)]
// CHECK:STDOUT:     %specific_impl_fn.loc296_3.1: <specific function> = specific_impl_function %impl.elem1.loc296_3.1, @Negate.WithSelf.Op(constants.%Negate.facet.b7e) [symbolic = %specific_impl_fn.loc296_3.2 (constants.%specific_impl_fn.9ba)]
// CHECK:STDOUT:     %bound_method.loc296_3.2: <bound method> = bound_method %x.ref, %specific_impl_fn.loc296_3.1
// CHECK:STDOUT:     %.loc296_3.3: ref @TestNegate.%Result.as_type.loc293_49.1 (%Result.as_type) = temporary_storage
// CHECK:STDOUT:     %Negate.WithSelf.Op.call: init @TestNegate.%Result.as_type.loc293_49.1 (%Result.as_type) to %.loc296_3.3 = call %bound_method.loc296_3.2(%x.ref)
// CHECK:STDOUT:     %.loc296_3.4: ref @TestNegate.%Result.as_type.loc293_49.1 (%Result.as_type) = temporary %.loc296_3.3, %Negate.WithSelf.Op.call
// CHECK:STDOUT:     %.loc296_3.5: %Destroy.type = converted constants.%Result.as_type, constants.%Result [symbolic = %Result.loc292_9.1 (constants.%Result)]
// CHECK:STDOUT:     %.loc296_3.6: %Destroy.type = converted constants.%Result.as_type, constants.%Result [symbolic = %Result.loc292_9.1 (constants.%Result)]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc296_3.4)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AddWith.loc124(constants.%Result, constants.%U.e1f, constants.%T.21f) {
// CHECK:STDOUT:   %Result.patt.loc121_9.2 => constants.%Result.patt
// CHECK:STDOUT:   %Result.loc121_9.1 => constants.%Result
// CHECK:STDOUT:   %U.patt.loc122_4.2 => constants.%U.patt.cb8
// CHECK:STDOUT:   %U.loc122_4.1 => constants.%U.e1f
// CHECK:STDOUT:   %U.as_type.loc123_35.1 => constants.%U.as_type.0bd
// CHECK:STDOUT:   %AddWith.type.loc123_35.1 => constants.%AddWith.type.4f5
// CHECK:STDOUT:   %facet_type.loc123_19 => constants.%facet_type.4f5
// CHECK:STDOUT:   %.Self.frozen.loc123_37.1 => constants.%.Self.frozen.136
// CHECK:STDOUT:   %require_complete.loc123 => constants.%require_complete.528878.1
// CHECK:STDOUT:   %.Self.frozen.as_type => constants.%.Self.frozen.as_type.034
// CHECK:STDOUT:   %AddWith.assoc_type => constants.%AddWith.assoc_type.71f
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.155
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.loc123_43.1 => constants.%AddWith.lookup_impl_witness.0f3
// CHECK:STDOUT:   %impl.elem0.loc123_43.1 => constants.%impl.elem0.1cd
// CHECK:STDOUT:   %Result.as_type.loc123_53.1 => constants.%Result.as_type
// CHECK:STDOUT:   %.Self => constants.%.Self.354
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.loc123_43.2 => constants.%AddWith.lookup_impl_witness.ab5
// CHECK:STDOUT:   %impl.elem0.loc123_43.2 => constants.%impl.elem0.2c2
// CHECK:STDOUT:   %facet_type.loc123_37 => constants.%facet_type.c17
// CHECK:STDOUT:   %pattern_type.loc123 => constants.%pattern_type.fef
// CHECK:STDOUT:   %T.patt.loc123_4.2 => constants.%T.patt.910
// CHECK:STDOUT:   %T.loc123_4.1 => constants.%T.21f
// CHECK:STDOUT:   %pattern_type.loc124_16 => constants.%pattern_type.86a9db.1
// CHECK:STDOUT:   %result.param_patt.loc124_16.2 => constants.%result.param_patt.f86
// CHECK:STDOUT:   %result.patt.loc124_16.2 => constants.%result.patt.a71562.1
// CHECK:STDOUT:   %T.as_type.loc124_29.1 => constants.%T.as_type.d2e
// CHECK:STDOUT:   %pattern_type.loc124_27 => constants.%pattern_type.fab
// CHECK:STDOUT:   %x.param_patt.loc124_27.2 => constants.%x.param_patt.947
// CHECK:STDOUT:   %x.patt.loc124_27.2 => constants.%x.patt.edd
// CHECK:STDOUT:   %pattern_type.loc124_33 => constants.%pattern_type.eff
// CHECK:STDOUT:   %y.param_patt.loc124_33.2 => constants.%y.param_patt.8c7
// CHECK:STDOUT:   %y.patt.loc124_33.2 => constants.%y.patt.b5a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @SubWith.loc135(constants.%Result, constants.%U.e1f, constants.%T.3c5) {
// CHECK:STDOUT:   %Result.patt.loc132_9.2 => constants.%Result.patt
// CHECK:STDOUT:   %Result.loc132_9.1 => constants.%Result
// CHECK:STDOUT:   %U.patt.loc133_4.2 => constants.%U.patt.cb8
// CHECK:STDOUT:   %U.loc133_4.1 => constants.%U.e1f
// CHECK:STDOUT:   %U.as_type.loc134_35.1 => constants.%U.as_type.0bd
// CHECK:STDOUT:   %SubWith.type.loc134_35.1 => constants.%SubWith.type.769
// CHECK:STDOUT:   %facet_type.loc134_19 => constants.%facet_type.769
// CHECK:STDOUT:   %.Self.frozen.loc134_37.1 => constants.%.Self.frozen.b13
// CHECK:STDOUT:   %require_complete.loc134 => constants.%require_complete.49f499.1
// CHECK:STDOUT:   %.Self.frozen.as_type => constants.%.Self.frozen.as_type.d1e
// CHECK:STDOUT:   %SubWith.assoc_type => constants.%SubWith.assoc_type.5f0
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.a5d
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.loc134_43.1 => constants.%SubWith.lookup_impl_witness.fe8
// CHECK:STDOUT:   %impl.elem0.loc134_43.1 => constants.%impl.elem0.7da
// CHECK:STDOUT:   %Result.as_type.loc134_53.1 => constants.%Result.as_type
// CHECK:STDOUT:   %.Self => constants.%.Self.4de
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.loc134_43.2 => constants.%SubWith.lookup_impl_witness.354
// CHECK:STDOUT:   %impl.elem0.loc134_43.2 => constants.%impl.elem0.3dc
// CHECK:STDOUT:   %facet_type.loc134_37 => constants.%facet_type.0a9
// CHECK:STDOUT:   %pattern_type.loc134 => constants.%pattern_type.a14
// CHECK:STDOUT:   %T.patt.loc134_4.2 => constants.%T.patt.51b
// CHECK:STDOUT:   %T.loc134_4.1 => constants.%T.3c5
// CHECK:STDOUT:   %pattern_type.loc135_16 => constants.%pattern_type.86a9db.1
// CHECK:STDOUT:   %result.param_patt.loc135_16.2 => constants.%result.param_patt.f86
// CHECK:STDOUT:   %result.patt.loc135_16.2 => constants.%result.patt.a71562.1
// CHECK:STDOUT:   %T.as_type.loc135_29.1 => constants.%T.as_type.4b7
// CHECK:STDOUT:   %pattern_type.loc135_27 => constants.%pattern_type.ca3
// CHECK:STDOUT:   %x.param_patt.loc135_27.2 => constants.%x.param_patt.10b
// CHECK:STDOUT:   %x.patt.loc135_27.2 => constants.%x.patt.7cd
// CHECK:STDOUT:   %pattern_type.loc135_33 => constants.%pattern_type.eff
// CHECK:STDOUT:   %y.param_patt.loc135_33.2 => constants.%y.param_patt.8c7
// CHECK:STDOUT:   %y.patt.loc135_33.2 => constants.%y.patt.b5a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @MulWith.loc146(constants.%Result, constants.%U.e1f, constants.%T.d03) {
// CHECK:STDOUT:   %Result.patt.loc143_9.2 => constants.%Result.patt
// CHECK:STDOUT:   %Result.loc143_9.1 => constants.%Result
// CHECK:STDOUT:   %U.patt.loc144_4.2 => constants.%U.patt.cb8
// CHECK:STDOUT:   %U.loc144_4.1 => constants.%U.e1f
// CHECK:STDOUT:   %U.as_type.loc145_35.1 => constants.%U.as_type.0bd
// CHECK:STDOUT:   %MulWith.type.loc145_35.1 => constants.%MulWith.type.b0b
// CHECK:STDOUT:   %facet_type.loc145_19 => constants.%facet_type.b0b
// CHECK:STDOUT:   %.Self.frozen.loc145_37.1 => constants.%.Self.frozen.a62
// CHECK:STDOUT:   %require_complete.loc145 => constants.%require_complete.8389d3.1
// CHECK:STDOUT:   %.Self.frozen.as_type => constants.%.Self.frozen.as_type.bd3
// CHECK:STDOUT:   %MulWith.assoc_type => constants.%MulWith.assoc_type.2e5
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.e89
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.loc145_43.1 => constants.%MulWith.lookup_impl_witness.136
// CHECK:STDOUT:   %impl.elem0.loc145_43.1 => constants.%impl.elem0.16c
// CHECK:STDOUT:   %Result.as_type.loc145_53.1 => constants.%Result.as_type
// CHECK:STDOUT:   %.Self => constants.%.Self.a44
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.loc145_43.2 => constants.%MulWith.lookup_impl_witness.ced
// CHECK:STDOUT:   %impl.elem0.loc145_43.2 => constants.%impl.elem0.0e8
// CHECK:STDOUT:   %facet_type.loc145_37 => constants.%facet_type.a5f
// CHECK:STDOUT:   %pattern_type.loc145 => constants.%pattern_type.3ab
// CHECK:STDOUT:   %T.patt.loc145_4.2 => constants.%T.patt.170
// CHECK:STDOUT:   %T.loc145_4.1 => constants.%T.d03
// CHECK:STDOUT:   %pattern_type.loc146_16 => constants.%pattern_type.86a9db.1
// CHECK:STDOUT:   %result.param_patt.loc146_16.2 => constants.%result.param_patt.f86
// CHECK:STDOUT:   %result.patt.loc146_16.2 => constants.%result.patt.a71562.1
// CHECK:STDOUT:   %T.as_type.loc146_29.1 => constants.%T.as_type.639
// CHECK:STDOUT:   %pattern_type.loc146_27 => constants.%pattern_type.d7a
// CHECK:STDOUT:   %x.param_patt.loc146_27.2 => constants.%x.param_patt.1c9
// CHECK:STDOUT:   %x.patt.loc146_27.2 => constants.%x.patt.080
// CHECK:STDOUT:   %pattern_type.loc146_33 => constants.%pattern_type.eff
// CHECK:STDOUT:   %y.param_patt.loc146_33.2 => constants.%y.param_patt.8c7
// CHECK:STDOUT:   %y.patt.loc146_33.2 => constants.%y.patt.b5a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @DivWith.loc157(constants.%Result, constants.%U.e1f, constants.%T.942) {
// CHECK:STDOUT:   %Result.patt.loc154_9.2 => constants.%Result.patt
// CHECK:STDOUT:   %Result.loc154_9.1 => constants.%Result
// CHECK:STDOUT:   %U.patt.loc155_4.2 => constants.%U.patt.cb8
// CHECK:STDOUT:   %U.loc155_4.1 => constants.%U.e1f
// CHECK:STDOUT:   %U.as_type.loc156_35.1 => constants.%U.as_type.0bd
// CHECK:STDOUT:   %DivWith.type.loc156_35.1 => constants.%DivWith.type.71d
// CHECK:STDOUT:   %facet_type.loc156_19 => constants.%facet_type.71d
// CHECK:STDOUT:   %.Self.frozen.loc156_37.1 => constants.%.Self.frozen.575
// CHECK:STDOUT:   %require_complete.loc156 => constants.%require_complete.c4013f.1
// CHECK:STDOUT:   %.Self.frozen.as_type => constants.%.Self.frozen.as_type.a77
// CHECK:STDOUT:   %DivWith.assoc_type => constants.%DivWith.assoc_type.e65
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.48a
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.loc156_43.1 => constants.%DivWith.lookup_impl_witness.d5d
// CHECK:STDOUT:   %impl.elem0.loc156_43.1 => constants.%impl.elem0.e77
// CHECK:STDOUT:   %Result.as_type.loc156_53.1 => constants.%Result.as_type
// CHECK:STDOUT:   %.Self => constants.%.Self.c58
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.loc156_43.2 => constants.%DivWith.lookup_impl_witness.f1d
// CHECK:STDOUT:   %impl.elem0.loc156_43.2 => constants.%impl.elem0.af2
// CHECK:STDOUT:   %facet_type.loc156_37 => constants.%facet_type.351
// CHECK:STDOUT:   %pattern_type.loc156 => constants.%pattern_type.ae7
// CHECK:STDOUT:   %T.patt.loc156_4.2 => constants.%T.patt.32b
// CHECK:STDOUT:   %T.loc156_4.1 => constants.%T.942
// CHECK:STDOUT:   %pattern_type.loc157_16 => constants.%pattern_type.86a9db.1
// CHECK:STDOUT:   %result.param_patt.loc157_16.2 => constants.%result.param_patt.f86
// CHECK:STDOUT:   %result.patt.loc157_16.2 => constants.%result.patt.a71562.1
// CHECK:STDOUT:   %T.as_type.loc157_29.1 => constants.%T.as_type.6cf
// CHECK:STDOUT:   %pattern_type.loc157_27 => constants.%pattern_type.c8a
// CHECK:STDOUT:   %x.param_patt.loc157_27.2 => constants.%x.param_patt.795
// CHECK:STDOUT:   %x.patt.loc157_27.2 => constants.%x.patt.818
// CHECK:STDOUT:   %pattern_type.loc157_33 => constants.%pattern_type.eff
// CHECK:STDOUT:   %y.param_patt.loc157_33.2 => constants.%y.param_patt.8c7
// CHECK:STDOUT:   %y.patt.loc157_33.2 => constants.%y.patt.b5a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ModWith.loc168(constants.%Result, constants.%U.e1f, constants.%T.fdb) {
// CHECK:STDOUT:   %Result.patt.loc165_9.2 => constants.%Result.patt
// CHECK:STDOUT:   %Result.loc165_9.1 => constants.%Result
// CHECK:STDOUT:   %U.patt.loc166_4.2 => constants.%U.patt.cb8
// CHECK:STDOUT:   %U.loc166_4.1 => constants.%U.e1f
// CHECK:STDOUT:   %U.as_type.loc167_35.1 => constants.%U.as_type.0bd
// CHECK:STDOUT:   %ModWith.type.loc167_35.1 => constants.%ModWith.type.68a
// CHECK:STDOUT:   %facet_type.loc167_19 => constants.%facet_type.68a
// CHECK:STDOUT:   %.Self.frozen.loc167_37.1 => constants.%.Self.frozen.e0d
// CHECK:STDOUT:   %require_complete.loc167 => constants.%require_complete.312803.1
// CHECK:STDOUT:   %.Self.frozen.as_type => constants.%.Self.frozen.as_type.01e
// CHECK:STDOUT:   %ModWith.assoc_type => constants.%ModWith.assoc_type.9c3
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.cb9
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.loc167_43.1 => constants.%ModWith.lookup_impl_witness.4ec
// CHECK:STDOUT:   %impl.elem0.loc167_43.1 => constants.%impl.elem0.671
// CHECK:STDOUT:   %Result.as_type.loc167_53.1 => constants.%Result.as_type
// CHECK:STDOUT:   %.Self => constants.%.Self.2c1
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.loc167_43.2 => constants.%ModWith.lookup_impl_witness.930
// CHECK:STDOUT:   %impl.elem0.loc167_43.2 => constants.%impl.elem0.e6d
// CHECK:STDOUT:   %facet_type.loc167_37 => constants.%facet_type.938
// CHECK:STDOUT:   %pattern_type.loc167 => constants.%pattern_type.b1e
// CHECK:STDOUT:   %T.patt.loc167_4.2 => constants.%T.patt.721
// CHECK:STDOUT:   %T.loc167_4.1 => constants.%T.fdb
// CHECK:STDOUT:   %pattern_type.loc168_16 => constants.%pattern_type.86a9db.1
// CHECK:STDOUT:   %result.param_patt.loc168_16.2 => constants.%result.param_patt.f86
// CHECK:STDOUT:   %result.patt.loc168_16.2 => constants.%result.patt.a71562.1
// CHECK:STDOUT:   %T.as_type.loc168_29.1 => constants.%T.as_type.73e
// CHECK:STDOUT:   %pattern_type.loc168_27 => constants.%pattern_type.70e
// CHECK:STDOUT:   %x.param_patt.loc168_27.2 => constants.%x.param_patt.375
// CHECK:STDOUT:   %x.patt.loc168_27.2 => constants.%x.patt.f07
// CHECK:STDOUT:   %pattern_type.loc168_33 => constants.%pattern_type.eff
// CHECK:STDOUT:   %y.param_patt.loc168_33.2 => constants.%y.param_patt.8c7
// CHECK:STDOUT:   %y.patt.loc168_33.2 => constants.%y.patt.b5a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AddWith.loc124(constants.%Destroy.facet.c1d, constants.%Destroy.facet.c1d, constants.%facet_value.b07) {
// CHECK:STDOUT:   %Result.patt.loc121_9.2 => constants.%Result.patt
// CHECK:STDOUT:   %Result.loc121_9.1 => constants.%Destroy.facet.c1d
// CHECK:STDOUT:   %U.patt.loc122_4.2 => constants.%U.patt.cb8
// CHECK:STDOUT:   %U.loc122_4.1 => constants.%Destroy.facet.c1d
// CHECK:STDOUT:   %U.as_type.loc123_35.1 => constants.%Int16
// CHECK:STDOUT:   %AddWith.type.loc123_35.1 => constants.%AddWith.type.e4b
// CHECK:STDOUT:   %facet_type.loc123_19 => constants.%facet_type.e4b
// CHECK:STDOUT:   %.Self.frozen.loc123_37.1 => constants.%.Self.frozen.ea9
// CHECK:STDOUT:   %require_complete.loc123 => constants.%complete_type.f37bc8.1
// CHECK:STDOUT:   %.Self.frozen.as_type => constants.%.Self.frozen.as_type.ceb
// CHECK:STDOUT:   %AddWith.assoc_type => constants.%AddWith.assoc_type.86e
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.30c
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.loc123_43.1 => constants.%AddWith.lookup_impl_witness.e18
// CHECK:STDOUT:   %impl.elem0.loc123_43.1 => constants.%impl.elem0.de3
// CHECK:STDOUT:   %Result.as_type.loc123_53.1 => constants.%Int16
// CHECK:STDOUT:   %.Self => constants.%.Self.033
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.loc123_43.2 => constants.%AddWith.lookup_impl_witness.73e
// CHECK:STDOUT:   %impl.elem0.loc123_43.2 => constants.%impl.elem0.08e
// CHECK:STDOUT:   %facet_type.loc123_37 => constants.%facet_type.5b8
// CHECK:STDOUT:   %pattern_type.loc123 => constants.%pattern_type.f83
// CHECK:STDOUT:   %T.patt.loc123_4.2 => constants.%T.patt.056
// CHECK:STDOUT:   %T.loc123_4.1 => constants.%facet_value.b07
// CHECK:STDOUT:   %pattern_type.loc124_16 => constants.%pattern_type.785
// CHECK:STDOUT:   %result.param_patt.loc124_16.2 => constants.%result.param_patt.be6
// CHECK:STDOUT:   %result.patt.loc124_16.2 => constants.%result.patt.a31
// CHECK:STDOUT:   %T.as_type.loc124_29.1 => constants.%i16
// CHECK:STDOUT:   %pattern_type.loc124_27 => constants.%pattern_type.9bc
// CHECK:STDOUT:   %x.param_patt.loc124_27.2 => constants.%x.param_patt.77a
// CHECK:STDOUT:   %x.patt.loc124_27.2 => constants.%x.patt.9b0
// CHECK:STDOUT:   %pattern_type.loc124_33 => constants.%pattern_type.785
// CHECK:STDOUT:   %y.param_patt.loc124_33.2 => constants.%y.param_patt.418
// CHECK:STDOUT:   %y.patt.loc124_33.2 => constants.%y.patt.890
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc124_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc124_27 => constants.%complete_type.2da
// CHECK:STDOUT:   %require_complete.loc124_33 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc126 => constants.%complete_type.f37bc8.2
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.9bf
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.loc126 => constants.%custom_witness.cba
// CHECK:STDOUT:   %AddWith.facet => constants.%AddWith.facet.2ce
// CHECK:STDOUT:   %AddWith.WithSelf.Op.type => constants.%AddWith.WithSelf.Op.type.4a0
// CHECK:STDOUT:   %.loc126_5.6 => constants.%.7bc
// CHECK:STDOUT:   %impl.elem1.loc126_5.2 => constants.%Op.72717c.1
// CHECK:STDOUT:   %specific_impl_fn.loc126_5.2 => constants.%Op.72717c.1
// CHECK:STDOUT:   %SelfDestruct.specific_fn.2 => constants.%SelfDestruct.specific_fn.581
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @SubWith.loc135(constants.%Destroy.facet.c1d, constants.%Destroy.facet.c1d, constants.%facet_value.ca4) {
// CHECK:STDOUT:   %Result.patt.loc132_9.2 => constants.%Result.patt
// CHECK:STDOUT:   %Result.loc132_9.1 => constants.%Destroy.facet.c1d
// CHECK:STDOUT:   %U.patt.loc133_4.2 => constants.%U.patt.cb8
// CHECK:STDOUT:   %U.loc133_4.1 => constants.%Destroy.facet.c1d
// CHECK:STDOUT:   %U.as_type.loc134_35.1 => constants.%Int16
// CHECK:STDOUT:   %SubWith.type.loc134_35.1 => constants.%SubWith.type.68d
// CHECK:STDOUT:   %facet_type.loc134_19 => constants.%facet_type.68d
// CHECK:STDOUT:   %.Self.frozen.loc134_37.1 => constants.%.Self.frozen.0c9
// CHECK:STDOUT:   %require_complete.loc134 => constants.%complete_type.8a2f63.1
// CHECK:STDOUT:   %.Self.frozen.as_type => constants.%.Self.frozen.as_type.471
// CHECK:STDOUT:   %SubWith.assoc_type => constants.%SubWith.assoc_type.dd6
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.dfb
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.loc134_43.1 => constants.%SubWith.lookup_impl_witness.379
// CHECK:STDOUT:   %impl.elem0.loc134_43.1 => constants.%impl.elem0.e78
// CHECK:STDOUT:   %Result.as_type.loc134_53.1 => constants.%Int16
// CHECK:STDOUT:   %.Self => constants.%.Self.a10
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.loc134_43.2 => constants.%SubWith.lookup_impl_witness.c4d
// CHECK:STDOUT:   %impl.elem0.loc134_43.2 => constants.%impl.elem0.22a
// CHECK:STDOUT:   %facet_type.loc134_37 => constants.%facet_type.5db
// CHECK:STDOUT:   %pattern_type.loc134 => constants.%pattern_type.7bf
// CHECK:STDOUT:   %T.patt.loc134_4.2 => constants.%T.patt.f08
// CHECK:STDOUT:   %T.loc134_4.1 => constants.%facet_value.ca4
// CHECK:STDOUT:   %pattern_type.loc135_16 => constants.%pattern_type.785
// CHECK:STDOUT:   %result.param_patt.loc135_16.2 => constants.%result.param_patt.be6
// CHECK:STDOUT:   %result.patt.loc135_16.2 => constants.%result.patt.a31
// CHECK:STDOUT:   %T.as_type.loc135_29.1 => constants.%i16
// CHECK:STDOUT:   %pattern_type.loc135_27 => constants.%pattern_type.9bc
// CHECK:STDOUT:   %x.param_patt.loc135_27.2 => constants.%x.param_patt.77a
// CHECK:STDOUT:   %x.patt.loc135_27.2 => constants.%x.patt.9b0
// CHECK:STDOUT:   %pattern_type.loc135_33 => constants.%pattern_type.785
// CHECK:STDOUT:   %y.param_patt.loc135_33.2 => constants.%y.param_patt.418
// CHECK:STDOUT:   %y.patt.loc135_33.2 => constants.%y.patt.890
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc135_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc135_27 => constants.%complete_type.2da
// CHECK:STDOUT:   %require_complete.loc135_33 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc137 => constants.%complete_type.8a2f63.2
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.ada
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.loc137 => constants.%custom_witness.354
// CHECK:STDOUT:   %SubWith.facet => constants.%SubWith.facet.1ef
// CHECK:STDOUT:   %SubWith.WithSelf.Op.type => constants.%SubWith.WithSelf.Op.type.23f
// CHECK:STDOUT:   %.loc137_5.6 => constants.%.eed
// CHECK:STDOUT:   %impl.elem1.loc137_5.2 => constants.%Op.72717c.2
// CHECK:STDOUT:   %specific_impl_fn.loc137_5.2 => constants.%Op.72717c.2
// CHECK:STDOUT:   %SelfDestruct.specific_fn.2 => constants.%SelfDestruct.specific_fn.581
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @MulWith.loc146(constants.%Destroy.facet.c1d, constants.%Destroy.facet.c1d, constants.%facet_value.3e5) {
// CHECK:STDOUT:   %Result.patt.loc143_9.2 => constants.%Result.patt
// CHECK:STDOUT:   %Result.loc143_9.1 => constants.%Destroy.facet.c1d
// CHECK:STDOUT:   %U.patt.loc144_4.2 => constants.%U.patt.cb8
// CHECK:STDOUT:   %U.loc144_4.1 => constants.%Destroy.facet.c1d
// CHECK:STDOUT:   %U.as_type.loc145_35.1 => constants.%Int16
// CHECK:STDOUT:   %MulWith.type.loc145_35.1 => constants.%MulWith.type.f65
// CHECK:STDOUT:   %facet_type.loc145_19 => constants.%facet_type.f65
// CHECK:STDOUT:   %.Self.frozen.loc145_37.1 => constants.%.Self.frozen.aec
// CHECK:STDOUT:   %require_complete.loc145 => constants.%complete_type.8c670c.1
// CHECK:STDOUT:   %.Self.frozen.as_type => constants.%.Self.frozen.as_type.34d
// CHECK:STDOUT:   %MulWith.assoc_type => constants.%MulWith.assoc_type.4c3
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.6ec
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.loc145_43.1 => constants.%MulWith.lookup_impl_witness.94b
// CHECK:STDOUT:   %impl.elem0.loc145_43.1 => constants.%impl.elem0.212
// CHECK:STDOUT:   %Result.as_type.loc145_53.1 => constants.%Int16
// CHECK:STDOUT:   %.Self => constants.%.Self.1e8
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.loc145_43.2 => constants.%MulWith.lookup_impl_witness.b4a
// CHECK:STDOUT:   %impl.elem0.loc145_43.2 => constants.%impl.elem0.088
// CHECK:STDOUT:   %facet_type.loc145_37 => constants.%facet_type.1c9
// CHECK:STDOUT:   %pattern_type.loc145 => constants.%pattern_type.2f6
// CHECK:STDOUT:   %T.patt.loc145_4.2 => constants.%T.patt.e8a
// CHECK:STDOUT:   %T.loc145_4.1 => constants.%facet_value.3e5
// CHECK:STDOUT:   %pattern_type.loc146_16 => constants.%pattern_type.785
// CHECK:STDOUT:   %result.param_patt.loc146_16.2 => constants.%result.param_patt.be6
// CHECK:STDOUT:   %result.patt.loc146_16.2 => constants.%result.patt.a31
// CHECK:STDOUT:   %T.as_type.loc146_29.1 => constants.%i16
// CHECK:STDOUT:   %pattern_type.loc146_27 => constants.%pattern_type.9bc
// CHECK:STDOUT:   %x.param_patt.loc146_27.2 => constants.%x.param_patt.77a
// CHECK:STDOUT:   %x.patt.loc146_27.2 => constants.%x.patt.9b0
// CHECK:STDOUT:   %pattern_type.loc146_33 => constants.%pattern_type.785
// CHECK:STDOUT:   %y.param_patt.loc146_33.2 => constants.%y.param_patt.418
// CHECK:STDOUT:   %y.patt.loc146_33.2 => constants.%y.patt.890
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc146_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc146_27 => constants.%complete_type.2da
// CHECK:STDOUT:   %require_complete.loc146_33 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc148 => constants.%complete_type.8c670c.2
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.b14
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.loc148 => constants.%custom_witness.3c3
// CHECK:STDOUT:   %MulWith.facet => constants.%MulWith.facet.d84
// CHECK:STDOUT:   %MulWith.WithSelf.Op.type => constants.%MulWith.WithSelf.Op.type.131
// CHECK:STDOUT:   %.loc148_5.6 => constants.%.010
// CHECK:STDOUT:   %impl.elem1.loc148_5.2 => constants.%Op.72717c.3
// CHECK:STDOUT:   %specific_impl_fn.loc148_5.2 => constants.%Op.72717c.3
// CHECK:STDOUT:   %SelfDestruct.specific_fn.2 => constants.%SelfDestruct.specific_fn.581
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @DivWith.loc157(constants.%Destroy.facet.c1d, constants.%Destroy.facet.c1d, constants.%facet_value.318) {
// CHECK:STDOUT:   %Result.patt.loc154_9.2 => constants.%Result.patt
// CHECK:STDOUT:   %Result.loc154_9.1 => constants.%Destroy.facet.c1d
// CHECK:STDOUT:   %U.patt.loc155_4.2 => constants.%U.patt.cb8
// CHECK:STDOUT:   %U.loc155_4.1 => constants.%Destroy.facet.c1d
// CHECK:STDOUT:   %U.as_type.loc156_35.1 => constants.%Int16
// CHECK:STDOUT:   %DivWith.type.loc156_35.1 => constants.%DivWith.type.b8a
// CHECK:STDOUT:   %facet_type.loc156_19 => constants.%facet_type.b8a
// CHECK:STDOUT:   %.Self.frozen.loc156_37.1 => constants.%.Self.frozen.ff8
// CHECK:STDOUT:   %require_complete.loc156 => constants.%complete_type.d99d66.1
// CHECK:STDOUT:   %.Self.frozen.as_type => constants.%.Self.frozen.as_type.2d9
// CHECK:STDOUT:   %DivWith.assoc_type => constants.%DivWith.assoc_type.712
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.99c
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.loc156_43.1 => constants.%DivWith.lookup_impl_witness.483
// CHECK:STDOUT:   %impl.elem0.loc156_43.1 => constants.%impl.elem0.f9d
// CHECK:STDOUT:   %Result.as_type.loc156_53.1 => constants.%Int16
// CHECK:STDOUT:   %.Self => constants.%.Self.74d
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.loc156_43.2 => constants.%DivWith.lookup_impl_witness.afb
// CHECK:STDOUT:   %impl.elem0.loc156_43.2 => constants.%impl.elem0.6cf
// CHECK:STDOUT:   %facet_type.loc156_37 => constants.%facet_type.ee9
// CHECK:STDOUT:   %pattern_type.loc156 => constants.%pattern_type.575
// CHECK:STDOUT:   %T.patt.loc156_4.2 => constants.%T.patt.257
// CHECK:STDOUT:   %T.loc156_4.1 => constants.%facet_value.318
// CHECK:STDOUT:   %pattern_type.loc157_16 => constants.%pattern_type.785
// CHECK:STDOUT:   %result.param_patt.loc157_16.2 => constants.%result.param_patt.be6
// CHECK:STDOUT:   %result.patt.loc157_16.2 => constants.%result.patt.a31
// CHECK:STDOUT:   %T.as_type.loc157_29.1 => constants.%i16
// CHECK:STDOUT:   %pattern_type.loc157_27 => constants.%pattern_type.9bc
// CHECK:STDOUT:   %x.param_patt.loc157_27.2 => constants.%x.param_patt.77a
// CHECK:STDOUT:   %x.patt.loc157_27.2 => constants.%x.patt.9b0
// CHECK:STDOUT:   %pattern_type.loc157_33 => constants.%pattern_type.785
// CHECK:STDOUT:   %y.param_patt.loc157_33.2 => constants.%y.param_patt.418
// CHECK:STDOUT:   %y.patt.loc157_33.2 => constants.%y.patt.890
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc157_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc157_27 => constants.%complete_type.2da
// CHECK:STDOUT:   %require_complete.loc157_33 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc159 => constants.%complete_type.d99d66.2
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.fb6
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.loc159 => constants.%custom_witness.8d0
// CHECK:STDOUT:   %DivWith.facet => constants.%DivWith.facet.4ad
// CHECK:STDOUT:   %DivWith.WithSelf.Op.type => constants.%DivWith.WithSelf.Op.type.edb
// CHECK:STDOUT:   %.loc159_5.6 => constants.%.b5e
// CHECK:STDOUT:   %impl.elem1.loc159_5.2 => constants.%Op.72717c.4
// CHECK:STDOUT:   %specific_impl_fn.loc159_5.2 => constants.%Op.72717c.4
// CHECK:STDOUT:   %SelfDestruct.specific_fn.2 => constants.%SelfDestruct.specific_fn.581
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ModWith.loc168(constants.%Destroy.facet.c1d, constants.%Destroy.facet.c1d, constants.%facet_value.425) {
// CHECK:STDOUT:   %Result.patt.loc165_9.2 => constants.%Result.patt
// CHECK:STDOUT:   %Result.loc165_9.1 => constants.%Destroy.facet.c1d
// CHECK:STDOUT:   %U.patt.loc166_4.2 => constants.%U.patt.cb8
// CHECK:STDOUT:   %U.loc166_4.1 => constants.%Destroy.facet.c1d
// CHECK:STDOUT:   %U.as_type.loc167_35.1 => constants.%Int16
// CHECK:STDOUT:   %ModWith.type.loc167_35.1 => constants.%ModWith.type.c67
// CHECK:STDOUT:   %facet_type.loc167_19 => constants.%facet_type.c67
// CHECK:STDOUT:   %.Self.frozen.loc167_37.1 => constants.%.Self.frozen.13f
// CHECK:STDOUT:   %require_complete.loc167 => constants.%complete_type.7d80e1.1
// CHECK:STDOUT:   %.Self.frozen.as_type => constants.%.Self.frozen.as_type.cb4
// CHECK:STDOUT:   %ModWith.assoc_type => constants.%ModWith.assoc_type.5f6
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.f04
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.loc167_43.1 => constants.%ModWith.lookup_impl_witness.0af
// CHECK:STDOUT:   %impl.elem0.loc167_43.1 => constants.%impl.elem0.d91
// CHECK:STDOUT:   %Result.as_type.loc167_53.1 => constants.%Int16
// CHECK:STDOUT:   %.Self => constants.%.Self.71c
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.loc167_43.2 => constants.%ModWith.lookup_impl_witness.c2a
// CHECK:STDOUT:   %impl.elem0.loc167_43.2 => constants.%impl.elem0.14c
// CHECK:STDOUT:   %facet_type.loc167_37 => constants.%facet_type.fe4
// CHECK:STDOUT:   %pattern_type.loc167 => constants.%pattern_type.330
// CHECK:STDOUT:   %T.patt.loc167_4.2 => constants.%T.patt.5c0
// CHECK:STDOUT:   %T.loc167_4.1 => constants.%facet_value.425
// CHECK:STDOUT:   %pattern_type.loc168_16 => constants.%pattern_type.785
// CHECK:STDOUT:   %result.param_patt.loc168_16.2 => constants.%result.param_patt.be6
// CHECK:STDOUT:   %result.patt.loc168_16.2 => constants.%result.patt.a31
// CHECK:STDOUT:   %T.as_type.loc168_29.1 => constants.%i16
// CHECK:STDOUT:   %pattern_type.loc168_27 => constants.%pattern_type.9bc
// CHECK:STDOUT:   %x.param_patt.loc168_27.2 => constants.%x.param_patt.77a
// CHECK:STDOUT:   %x.patt.loc168_27.2 => constants.%x.patt.9b0
// CHECK:STDOUT:   %pattern_type.loc168_33 => constants.%pattern_type.785
// CHECK:STDOUT:   %y.param_patt.loc168_33.2 => constants.%y.param_patt.418
// CHECK:STDOUT:   %y.patt.loc168_33.2 => constants.%y.patt.890
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc168_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc168_27 => constants.%complete_type.2da
// CHECK:STDOUT:   %require_complete.loc168_33 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc170 => constants.%complete_type.7d80e1.2
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.d19
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.loc170 => constants.%custom_witness.169
// CHECK:STDOUT:   %ModWith.facet => constants.%ModWith.facet.ab1
// CHECK:STDOUT:   %ModWith.WithSelf.Op.type => constants.%ModWith.WithSelf.Op.type.e29
// CHECK:STDOUT:   %.loc170_5.6 => constants.%.d62
// CHECK:STDOUT:   %impl.elem1.loc170_5.2 => constants.%Op.72717c.5
// CHECK:STDOUT:   %specific_impl_fn.loc170_5.2 => constants.%Op.72717c.5
// CHECK:STDOUT:   %SelfDestruct.specific_fn.2 => constants.%SelfDestruct.specific_fn.581
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AddWith.loc124(constants.%Destroy.facet.c1d, constants.%Destroy.facet.1b4, constants.%facet_value.7e4) {
// CHECK:STDOUT:   %Result.patt.loc121_9.2 => constants.%Result.patt
// CHECK:STDOUT:   %Result.loc121_9.1 => constants.%Destroy.facet.c1d
// CHECK:STDOUT:   %U.patt.loc122_4.2 => constants.%U.patt.cb8
// CHECK:STDOUT:   %U.loc122_4.1 => constants.%Destroy.facet.1b4
// CHECK:STDOUT:   %U.as_type.loc123_35.1 => constants.%i16
// CHECK:STDOUT:   %AddWith.type.loc123_35.1 => constants.%AddWith.type.2ed
// CHECK:STDOUT:   %facet_type.loc123_19 => constants.%facet_type.2ed
// CHECK:STDOUT:   %.Self.frozen.loc123_37.1 => constants.%.Self.frozen.17a
// CHECK:STDOUT:   %require_complete.loc123 => constants.%complete_type.32de8b.1
// CHECK:STDOUT:   %.Self.frozen.as_type => constants.%.Self.frozen.as_type.8c1
// CHECK:STDOUT:   %AddWith.assoc_type => constants.%AddWith.assoc_type.18f
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.b7d
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.loc123_43.1 => constants.%AddWith.lookup_impl_witness.03c
// CHECK:STDOUT:   %impl.elem0.loc123_43.1 => constants.%impl.elem0.719
// CHECK:STDOUT:   %Result.as_type.loc123_53.1 => constants.%Int16
// CHECK:STDOUT:   %.Self => constants.%.Self.737
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.loc123_43.2 => constants.%AddWith.lookup_impl_witness.de0
// CHECK:STDOUT:   %impl.elem0.loc123_43.2 => constants.%impl.elem0.025
// CHECK:STDOUT:   %facet_type.loc123_37 => constants.%facet_type.9dd
// CHECK:STDOUT:   %pattern_type.loc123 => constants.%pattern_type.b6f
// CHECK:STDOUT:   %T.patt.loc123_4.2 => constants.%T.patt.b9b
// CHECK:STDOUT:   %T.loc123_4.1 => constants.%facet_value.7e4
// CHECK:STDOUT:   %pattern_type.loc124_16 => constants.%pattern_type.785
// CHECK:STDOUT:   %result.param_patt.loc124_16.2 => constants.%result.param_patt.be6
// CHECK:STDOUT:   %result.patt.loc124_16.2 => constants.%result.patt.a31
// CHECK:STDOUT:   %T.as_type.loc124_29.1 => constants.%Int16
// CHECK:STDOUT:   %pattern_type.loc124_27 => constants.%pattern_type.785
// CHECK:STDOUT:   %x.param_patt.loc124_27.2 => constants.%x.param_patt.3b0
// CHECK:STDOUT:   %x.patt.loc124_27.2 => constants.%x.patt.869
// CHECK:STDOUT:   %pattern_type.loc124_33 => constants.%pattern_type.9bc
// CHECK:STDOUT:   %y.param_patt.loc124_33.2 => constants.%y.param_patt.322
// CHECK:STDOUT:   %y.patt.loc124_33.2 => constants.%y.patt.a3a
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc124_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc124_27 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc124_33 => constants.%complete_type.2da
// CHECK:STDOUT:   %require_complete.loc126 => constants.%complete_type.32de8b.2
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.e08
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.loc126 => constants.%custom_witness.b79
// CHECK:STDOUT:   %AddWith.facet => constants.%AddWith.facet.11e
// CHECK:STDOUT:   %AddWith.WithSelf.Op.type => constants.%AddWith.WithSelf.Op.type.eb3
// CHECK:STDOUT:   %.loc126_5.6 => constants.%.da8
// CHECK:STDOUT:   %impl.elem1.loc126_5.2 => constants.%Int16.Op.4dca31.2
// CHECK:STDOUT:   %specific_impl_fn.loc126_5.2 => constants.%Int16.Op.4dca31.2
// CHECK:STDOUT:   %SelfDestruct.specific_fn.2 => constants.%SelfDestruct.specific_fn.581
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @SubWith.loc135(constants.%Destroy.facet.c1d, constants.%Destroy.facet.1b4, constants.%facet_value.dd3) {
// CHECK:STDOUT:   %Result.patt.loc132_9.2 => constants.%Result.patt
// CHECK:STDOUT:   %Result.loc132_9.1 => constants.%Destroy.facet.c1d
// CHECK:STDOUT:   %U.patt.loc133_4.2 => constants.%U.patt.cb8
// CHECK:STDOUT:   %U.loc133_4.1 => constants.%Destroy.facet.1b4
// CHECK:STDOUT:   %U.as_type.loc134_35.1 => constants.%i16
// CHECK:STDOUT:   %SubWith.type.loc134_35.1 => constants.%SubWith.type.606
// CHECK:STDOUT:   %facet_type.loc134_19 => constants.%facet_type.606
// CHECK:STDOUT:   %.Self.frozen.loc134_37.1 => constants.%.Self.frozen.6eb
// CHECK:STDOUT:   %require_complete.loc134 => constants.%complete_type.75c188.1
// CHECK:STDOUT:   %.Self.frozen.as_type => constants.%.Self.frozen.as_type.fe6
// CHECK:STDOUT:   %SubWith.assoc_type => constants.%SubWith.assoc_type.9c3
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.f66
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.loc134_43.1 => constants.%SubWith.lookup_impl_witness.183
// CHECK:STDOUT:   %impl.elem0.loc134_43.1 => constants.%impl.elem0.cc9
// CHECK:STDOUT:   %Result.as_type.loc134_53.1 => constants.%Int16
// CHECK:STDOUT:   %.Self => constants.%.Self.220
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.loc134_43.2 => constants.%SubWith.lookup_impl_witness.ebb
// CHECK:STDOUT:   %impl.elem0.loc134_43.2 => constants.%impl.elem0.1b2
// CHECK:STDOUT:   %facet_type.loc134_37 => constants.%facet_type.1a3
// CHECK:STDOUT:   %pattern_type.loc134 => constants.%pattern_type.8ce
// CHECK:STDOUT:   %T.patt.loc134_4.2 => constants.%T.patt.88b
// CHECK:STDOUT:   %T.loc134_4.1 => constants.%facet_value.dd3
// CHECK:STDOUT:   %pattern_type.loc135_16 => constants.%pattern_type.785
// CHECK:STDOUT:   %result.param_patt.loc135_16.2 => constants.%result.param_patt.be6
// CHECK:STDOUT:   %result.patt.loc135_16.2 => constants.%result.patt.a31
// CHECK:STDOUT:   %T.as_type.loc135_29.1 => constants.%Int16
// CHECK:STDOUT:   %pattern_type.loc135_27 => constants.%pattern_type.785
// CHECK:STDOUT:   %x.param_patt.loc135_27.2 => constants.%x.param_patt.3b0
// CHECK:STDOUT:   %x.patt.loc135_27.2 => constants.%x.patt.869
// CHECK:STDOUT:   %pattern_type.loc135_33 => constants.%pattern_type.9bc
// CHECK:STDOUT:   %y.param_patt.loc135_33.2 => constants.%y.param_patt.322
// CHECK:STDOUT:   %y.patt.loc135_33.2 => constants.%y.patt.a3a
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc135_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc135_27 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc135_33 => constants.%complete_type.2da
// CHECK:STDOUT:   %require_complete.loc137 => constants.%complete_type.75c188.2
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.144
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.loc137 => constants.%custom_witness.1e8
// CHECK:STDOUT:   %SubWith.facet => constants.%SubWith.facet.6fa
// CHECK:STDOUT:   %SubWith.WithSelf.Op.type => constants.%SubWith.WithSelf.Op.type.bb5
// CHECK:STDOUT:   %.loc137_5.6 => constants.%.42a
// CHECK:STDOUT:   %impl.elem1.loc137_5.2 => constants.%Int16.Op.4dca31.3
// CHECK:STDOUT:   %specific_impl_fn.loc137_5.2 => constants.%Int16.Op.4dca31.3
// CHECK:STDOUT:   %SelfDestruct.specific_fn.2 => constants.%SelfDestruct.specific_fn.581
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @MulWith.loc146(constants.%Destroy.facet.c1d, constants.%Destroy.facet.1b4, constants.%facet_value.3d9) {
// CHECK:STDOUT:   %Result.patt.loc143_9.2 => constants.%Result.patt
// CHECK:STDOUT:   %Result.loc143_9.1 => constants.%Destroy.facet.c1d
// CHECK:STDOUT:   %U.patt.loc144_4.2 => constants.%U.patt.cb8
// CHECK:STDOUT:   %U.loc144_4.1 => constants.%Destroy.facet.1b4
// CHECK:STDOUT:   %U.as_type.loc145_35.1 => constants.%i16
// CHECK:STDOUT:   %MulWith.type.loc145_35.1 => constants.%MulWith.type.1e2
// CHECK:STDOUT:   %facet_type.loc145_19 => constants.%facet_type.1e2
// CHECK:STDOUT:   %.Self.frozen.loc145_37.1 => constants.%.Self.frozen.a99
// CHECK:STDOUT:   %require_complete.loc145 => constants.%complete_type.a7adcd.1
// CHECK:STDOUT:   %.Self.frozen.as_type => constants.%.Self.frozen.as_type.6f1
// CHECK:STDOUT:   %MulWith.assoc_type => constants.%MulWith.assoc_type.34c
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.2d4
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.loc145_43.1 => constants.%MulWith.lookup_impl_witness.7ca
// CHECK:STDOUT:   %impl.elem0.loc145_43.1 => constants.%impl.elem0.c79
// CHECK:STDOUT:   %Result.as_type.loc145_53.1 => constants.%Int16
// CHECK:STDOUT:   %.Self => constants.%.Self.c91
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.loc145_43.2 => constants.%MulWith.lookup_impl_witness.04e
// CHECK:STDOUT:   %impl.elem0.loc145_43.2 => constants.%impl.elem0.0df
// CHECK:STDOUT:   %facet_type.loc145_37 => constants.%facet_type.26b
// CHECK:STDOUT:   %pattern_type.loc145 => constants.%pattern_type.2a0
// CHECK:STDOUT:   %T.patt.loc145_4.2 => constants.%T.patt.c4b
// CHECK:STDOUT:   %T.loc145_4.1 => constants.%facet_value.3d9
// CHECK:STDOUT:   %pattern_type.loc146_16 => constants.%pattern_type.785
// CHECK:STDOUT:   %result.param_patt.loc146_16.2 => constants.%result.param_patt.be6
// CHECK:STDOUT:   %result.patt.loc146_16.2 => constants.%result.patt.a31
// CHECK:STDOUT:   %T.as_type.loc146_29.1 => constants.%Int16
// CHECK:STDOUT:   %pattern_type.loc146_27 => constants.%pattern_type.785
// CHECK:STDOUT:   %x.param_patt.loc146_27.2 => constants.%x.param_patt.3b0
// CHECK:STDOUT:   %x.patt.loc146_27.2 => constants.%x.patt.869
// CHECK:STDOUT:   %pattern_type.loc146_33 => constants.%pattern_type.9bc
// CHECK:STDOUT:   %y.param_patt.loc146_33.2 => constants.%y.param_patt.322
// CHECK:STDOUT:   %y.patt.loc146_33.2 => constants.%y.patt.a3a
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc146_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc146_27 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc146_33 => constants.%complete_type.2da
// CHECK:STDOUT:   %require_complete.loc148 => constants.%complete_type.a7adcd.2
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.635
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.loc148 => constants.%custom_witness.e2b
// CHECK:STDOUT:   %MulWith.facet => constants.%MulWith.facet.8b1
// CHECK:STDOUT:   %MulWith.WithSelf.Op.type => constants.%MulWith.WithSelf.Op.type.8f5
// CHECK:STDOUT:   %.loc148_5.6 => constants.%.ecc
// CHECK:STDOUT:   %impl.elem1.loc148_5.2 => constants.%Int16.Op.4dca31.4
// CHECK:STDOUT:   %specific_impl_fn.loc148_5.2 => constants.%Int16.Op.4dca31.4
// CHECK:STDOUT:   %SelfDestruct.specific_fn.2 => constants.%SelfDestruct.specific_fn.581
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @DivWith.loc157(constants.%Destroy.facet.c1d, constants.%Destroy.facet.1b4, constants.%facet_value.4fa) {
// CHECK:STDOUT:   %Result.patt.loc154_9.2 => constants.%Result.patt
// CHECK:STDOUT:   %Result.loc154_9.1 => constants.%Destroy.facet.c1d
// CHECK:STDOUT:   %U.patt.loc155_4.2 => constants.%U.patt.cb8
// CHECK:STDOUT:   %U.loc155_4.1 => constants.%Destroy.facet.1b4
// CHECK:STDOUT:   %U.as_type.loc156_35.1 => constants.%i16
// CHECK:STDOUT:   %DivWith.type.loc156_35.1 => constants.%DivWith.type.45d
// CHECK:STDOUT:   %facet_type.loc156_19 => constants.%facet_type.45d
// CHECK:STDOUT:   %.Self.frozen.loc156_37.1 => constants.%.Self.frozen.52f
// CHECK:STDOUT:   %require_complete.loc156 => constants.%complete_type.d57eaf.1
// CHECK:STDOUT:   %.Self.frozen.as_type => constants.%.Self.frozen.as_type.8c0
// CHECK:STDOUT:   %DivWith.assoc_type => constants.%DivWith.assoc_type.5a4
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.156
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.loc156_43.1 => constants.%DivWith.lookup_impl_witness.5df
// CHECK:STDOUT:   %impl.elem0.loc156_43.1 => constants.%impl.elem0.967
// CHECK:STDOUT:   %Result.as_type.loc156_53.1 => constants.%Int16
// CHECK:STDOUT:   %.Self => constants.%.Self.327
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.loc156_43.2 => constants.%DivWith.lookup_impl_witness.851
// CHECK:STDOUT:   %impl.elem0.loc156_43.2 => constants.%impl.elem0.833
// CHECK:STDOUT:   %facet_type.loc156_37 => constants.%facet_type.cb9
// CHECK:STDOUT:   %pattern_type.loc156 => constants.%pattern_type.264
// CHECK:STDOUT:   %T.patt.loc156_4.2 => constants.%T.patt.19d
// CHECK:STDOUT:   %T.loc156_4.1 => constants.%facet_value.4fa
// CHECK:STDOUT:   %pattern_type.loc157_16 => constants.%pattern_type.785
// CHECK:STDOUT:   %result.param_patt.loc157_16.2 => constants.%result.param_patt.be6
// CHECK:STDOUT:   %result.patt.loc157_16.2 => constants.%result.patt.a31
// CHECK:STDOUT:   %T.as_type.loc157_29.1 => constants.%Int16
// CHECK:STDOUT:   %pattern_type.loc157_27 => constants.%pattern_type.785
// CHECK:STDOUT:   %x.param_patt.loc157_27.2 => constants.%x.param_patt.3b0
// CHECK:STDOUT:   %x.patt.loc157_27.2 => constants.%x.patt.869
// CHECK:STDOUT:   %pattern_type.loc157_33 => constants.%pattern_type.9bc
// CHECK:STDOUT:   %y.param_patt.loc157_33.2 => constants.%y.param_patt.322
// CHECK:STDOUT:   %y.patt.loc157_33.2 => constants.%y.patt.a3a
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc157_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc157_27 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc157_33 => constants.%complete_type.2da
// CHECK:STDOUT:   %require_complete.loc159 => constants.%complete_type.d57eaf.2
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.444
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.loc159 => constants.%custom_witness.21db
// CHECK:STDOUT:   %DivWith.facet => constants.%DivWith.facet.57c
// CHECK:STDOUT:   %DivWith.WithSelf.Op.type => constants.%DivWith.WithSelf.Op.type.37a
// CHECK:STDOUT:   %.loc159_5.6 => constants.%.5a3
// CHECK:STDOUT:   %impl.elem1.loc159_5.2 => constants.%Int16.Op.4dca31.5
// CHECK:STDOUT:   %specific_impl_fn.loc159_5.2 => constants.%Int16.Op.4dca31.5
// CHECK:STDOUT:   %SelfDestruct.specific_fn.2 => constants.%SelfDestruct.specific_fn.581
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ModWith.loc168(constants.%Destroy.facet.c1d, constants.%Destroy.facet.1b4, constants.%facet_value.45d) {
// CHECK:STDOUT:   %Result.patt.loc165_9.2 => constants.%Result.patt
// CHECK:STDOUT:   %Result.loc165_9.1 => constants.%Destroy.facet.c1d
// CHECK:STDOUT:   %U.patt.loc166_4.2 => constants.%U.patt.cb8
// CHECK:STDOUT:   %U.loc166_4.1 => constants.%Destroy.facet.1b4
// CHECK:STDOUT:   %U.as_type.loc167_35.1 => constants.%i16
// CHECK:STDOUT:   %ModWith.type.loc167_35.1 => constants.%ModWith.type.104
// CHECK:STDOUT:   %facet_type.loc167_19 => constants.%facet_type.104
// CHECK:STDOUT:   %.Self.frozen.loc167_37.1 => constants.%.Self.frozen.b7c
// CHECK:STDOUT:   %require_complete.loc167 => constants.%complete_type.4d792a.1
// CHECK:STDOUT:   %.Self.frozen.as_type => constants.%.Self.frozen.as_type.e20
// CHECK:STDOUT:   %ModWith.assoc_type => constants.%ModWith.assoc_type.0e9
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.92c
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.loc167_43.1 => constants.%ModWith.lookup_impl_witness.a99
// CHECK:STDOUT:   %impl.elem0.loc167_43.1 => constants.%impl.elem0.b94
// CHECK:STDOUT:   %Result.as_type.loc167_53.1 => constants.%Int16
// CHECK:STDOUT:   %.Self => constants.%.Self.6a7
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.loc167_43.2 => constants.%ModWith.lookup_impl_witness.177
// CHECK:STDOUT:   %impl.elem0.loc167_43.2 => constants.%impl.elem0.d83
// CHECK:STDOUT:   %facet_type.loc167_37 => constants.%facet_type.70f
// CHECK:STDOUT:   %pattern_type.loc167 => constants.%pattern_type.4c3
// CHECK:STDOUT:   %T.patt.loc167_4.2 => constants.%T.patt.cf6
// CHECK:STDOUT:   %T.loc167_4.1 => constants.%facet_value.45d
// CHECK:STDOUT:   %pattern_type.loc168_16 => constants.%pattern_type.785
// CHECK:STDOUT:   %result.param_patt.loc168_16.2 => constants.%result.param_patt.be6
// CHECK:STDOUT:   %result.patt.loc168_16.2 => constants.%result.patt.a31
// CHECK:STDOUT:   %T.as_type.loc168_29.1 => constants.%Int16
// CHECK:STDOUT:   %pattern_type.loc168_27 => constants.%pattern_type.785
// CHECK:STDOUT:   %x.param_patt.loc168_27.2 => constants.%x.param_patt.3b0
// CHECK:STDOUT:   %x.patt.loc168_27.2 => constants.%x.patt.869
// CHECK:STDOUT:   %pattern_type.loc168_33 => constants.%pattern_type.9bc
// CHECK:STDOUT:   %y.param_patt.loc168_33.2 => constants.%y.param_patt.322
// CHECK:STDOUT:   %y.patt.loc168_33.2 => constants.%y.patt.a3a
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc168_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc168_27 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc168_33 => constants.%complete_type.2da
// CHECK:STDOUT:   %require_complete.loc170 => constants.%complete_type.4d792a.2
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.870
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.loc170 => constants.%custom_witness.6dd
// CHECK:STDOUT:   %ModWith.facet => constants.%ModWith.facet.f5a5
// CHECK:STDOUT:   %ModWith.WithSelf.Op.type => constants.%ModWith.WithSelf.Op.type.374
// CHECK:STDOUT:   %.loc170_5.6 => constants.%.52c1
// CHECK:STDOUT:   %impl.elem1.loc170_5.2 => constants.%Int16.Op.4dca31.6
// CHECK:STDOUT:   %specific_impl_fn.loc170_5.2 => constants.%Int16.Op.4dca31.6
// CHECK:STDOUT:   %SelfDestruct.specific_fn.2 => constants.%SelfDestruct.specific_fn.581
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AddWith.loc124(constants.%Destroy.facet.806, constants.%Destroy.facet.806, constants.%facet_value.790) {
// CHECK:STDOUT:   %Result.patt.loc121_9.2 => constants.%Result.patt
// CHECK:STDOUT:   %Result.loc121_9.1 => constants.%Destroy.facet.806
// CHECK:STDOUT:   %U.patt.loc122_4.2 => constants.%U.patt.cb8
// CHECK:STDOUT:   %U.loc122_4.1 => constants.%Destroy.facet.806
// CHECK:STDOUT:   %U.as_type.loc123_35.1 => constants.%Int32
// CHECK:STDOUT:   %AddWith.type.loc123_35.1 => constants.%AddWith.type.db7
// CHECK:STDOUT:   %facet_type.loc123_19 => constants.%facet_type.db7
// CHECK:STDOUT:   %.Self.frozen.loc123_37.1 => constants.%.Self.frozen.bc4
// CHECK:STDOUT:   %require_complete.loc123 => constants.%complete_type.758ea8.1
// CHECK:STDOUT:   %.Self.frozen.as_type => constants.%.Self.frozen.as_type.447
// CHECK:STDOUT:   %AddWith.assoc_type => constants.%AddWith.assoc_type.be8
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.31c
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.loc123_43.1 => constants.%AddWith.lookup_impl_witness.a9e
// CHECK:STDOUT:   %impl.elem0.loc123_43.1 => constants.%impl.elem0.517
// CHECK:STDOUT:   %Result.as_type.loc123_53.1 => constants.%Int32
// CHECK:STDOUT:   %.Self => constants.%.Self.6a9
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.loc123_43.2 => constants.%AddWith.lookup_impl_witness.f6c
// CHECK:STDOUT:   %impl.elem0.loc123_43.2 => constants.%impl.elem0.a74
// CHECK:STDOUT:   %facet_type.loc123_37 => constants.%facet_type.bf5
// CHECK:STDOUT:   %pattern_type.loc123 => constants.%pattern_type.eea
// CHECK:STDOUT:   %T.patt.loc123_4.2 => constants.%T.patt.79b
// CHECK:STDOUT:   %T.loc123_4.1 => constants.%facet_value.790
// CHECK:STDOUT:   %pattern_type.loc124_16 => constants.%pattern_type.8b4
// CHECK:STDOUT:   %result.param_patt.loc124_16.2 => constants.%result.param_patt.e81
// CHECK:STDOUT:   %result.patt.loc124_16.2 => constants.%result.patt.47f
// CHECK:STDOUT:   %T.as_type.loc124_29.1 => constants.%Int16
// CHECK:STDOUT:   %pattern_type.loc124_27 => constants.%pattern_type.785
// CHECK:STDOUT:   %x.param_patt.loc124_27.2 => constants.%x.param_patt.3b0
// CHECK:STDOUT:   %x.patt.loc124_27.2 => constants.%x.patt.869
// CHECK:STDOUT:   %pattern_type.loc124_33 => constants.%pattern_type.8b4
// CHECK:STDOUT:   %y.param_patt.loc124_33.2 => constants.%y.param_patt.e0c
// CHECK:STDOUT:   %y.patt.loc124_33.2 => constants.%y.patt.9e1
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc124_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc124_27 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc124_33 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc126 => constants.%complete_type.758ea8.2
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.86c
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.loc126 => constants.%custom_witness.513476.1
// CHECK:STDOUT:   %AddWith.facet => constants.%AddWith.facet.8f8
// CHECK:STDOUT:   %AddWith.WithSelf.Op.type => constants.%AddWith.WithSelf.Op.type.4cf
// CHECK:STDOUT:   %.loc126_5.6 => constants.%.feb
// CHECK:STDOUT:   %impl.elem1.loc126_5.2 => constants.%Int16.Op.4dca31.7
// CHECK:STDOUT:   %specific_impl_fn.loc126_5.2 => constants.%Int16.Op.4dca31.7
// CHECK:STDOUT:   %SelfDestruct.specific_fn.2 => constants.%SelfDestruct.specific_fn.13e
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @SubWith.loc135(constants.%Destroy.facet.806, constants.%Destroy.facet.806, constants.%facet_value.2de) {
// CHECK:STDOUT:   %Result.patt.loc132_9.2 => constants.%Result.patt
// CHECK:STDOUT:   %Result.loc132_9.1 => constants.%Destroy.facet.806
// CHECK:STDOUT:   %U.patt.loc133_4.2 => constants.%U.patt.cb8
// CHECK:STDOUT:   %U.loc133_4.1 => constants.%Destroy.facet.806
// CHECK:STDOUT:   %U.as_type.loc134_35.1 => constants.%Int32
// CHECK:STDOUT:   %SubWith.type.loc134_35.1 => constants.%SubWith.type.120
// CHECK:STDOUT:   %facet_type.loc134_19 => constants.%facet_type.120
// CHECK:STDOUT:   %.Self.frozen.loc134_37.1 => constants.%.Self.frozen.faf
// CHECK:STDOUT:   %require_complete.loc134 => constants.%complete_type.46ea3c.1
// CHECK:STDOUT:   %.Self.frozen.as_type => constants.%.Self.frozen.as_type.ea6
// CHECK:STDOUT:   %SubWith.assoc_type => constants.%SubWith.assoc_type.dae
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.fac
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.loc134_43.1 => constants.%SubWith.lookup_impl_witness.c4f
// CHECK:STDOUT:   %impl.elem0.loc134_43.1 => constants.%impl.elem0.b5a
// CHECK:STDOUT:   %Result.as_type.loc134_53.1 => constants.%Int32
// CHECK:STDOUT:   %.Self => constants.%.Self.861
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.loc134_43.2 => constants.%SubWith.lookup_impl_witness.4cf
// CHECK:STDOUT:   %impl.elem0.loc134_43.2 => constants.%impl.elem0.94c
// CHECK:STDOUT:   %facet_type.loc134_37 => constants.%facet_type.b03
// CHECK:STDOUT:   %pattern_type.loc134 => constants.%pattern_type.110
// CHECK:STDOUT:   %T.patt.loc134_4.2 => constants.%T.patt.e31
// CHECK:STDOUT:   %T.loc134_4.1 => constants.%facet_value.2de
// CHECK:STDOUT:   %pattern_type.loc135_16 => constants.%pattern_type.8b4
// CHECK:STDOUT:   %result.param_patt.loc135_16.2 => constants.%result.param_patt.e81
// CHECK:STDOUT:   %result.patt.loc135_16.2 => constants.%result.patt.47f
// CHECK:STDOUT:   %T.as_type.loc135_29.1 => constants.%Int16
// CHECK:STDOUT:   %pattern_type.loc135_27 => constants.%pattern_type.785
// CHECK:STDOUT:   %x.param_patt.loc135_27.2 => constants.%x.param_patt.3b0
// CHECK:STDOUT:   %x.patt.loc135_27.2 => constants.%x.patt.869
// CHECK:STDOUT:   %pattern_type.loc135_33 => constants.%pattern_type.8b4
// CHECK:STDOUT:   %y.param_patt.loc135_33.2 => constants.%y.param_patt.e0c
// CHECK:STDOUT:   %y.patt.loc135_33.2 => constants.%y.patt.9e1
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc135_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc135_27 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc135_33 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc137 => constants.%complete_type.46ea3c.2
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.147
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.loc137 => constants.%custom_witness.32cef1.1
// CHECK:STDOUT:   %SubWith.facet => constants.%SubWith.facet.0b0
// CHECK:STDOUT:   %SubWith.WithSelf.Op.type => constants.%SubWith.WithSelf.Op.type.e1e
// CHECK:STDOUT:   %.loc137_5.6 => constants.%.988
// CHECK:STDOUT:   %impl.elem1.loc137_5.2 => constants.%Int16.Op.4dca31.8
// CHECK:STDOUT:   %specific_impl_fn.loc137_5.2 => constants.%Int16.Op.4dca31.8
// CHECK:STDOUT:   %SelfDestruct.specific_fn.2 => constants.%SelfDestruct.specific_fn.13e
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @MulWith.loc146(constants.%Destroy.facet.806, constants.%Destroy.facet.806, constants.%facet_value.1fb) {
// CHECK:STDOUT:   %Result.patt.loc143_9.2 => constants.%Result.patt
// CHECK:STDOUT:   %Result.loc143_9.1 => constants.%Destroy.facet.806
// CHECK:STDOUT:   %U.patt.loc144_4.2 => constants.%U.patt.cb8
// CHECK:STDOUT:   %U.loc144_4.1 => constants.%Destroy.facet.806
// CHECK:STDOUT:   %U.as_type.loc145_35.1 => constants.%Int32
// CHECK:STDOUT:   %MulWith.type.loc145_35.1 => constants.%MulWith.type.d67
// CHECK:STDOUT:   %facet_type.loc145_19 => constants.%facet_type.d67b
// CHECK:STDOUT:   %.Self.frozen.loc145_37.1 => constants.%.Self.frozen.c7a
// CHECK:STDOUT:   %require_complete.loc145 => constants.%complete_type.920626.1
// CHECK:STDOUT:   %.Self.frozen.as_type => constants.%.Self.frozen.as_type.c47
// CHECK:STDOUT:   %MulWith.assoc_type => constants.%MulWith.assoc_type.ab1
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.d0a
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.loc145_43.1 => constants.%MulWith.lookup_impl_witness.ab5
// CHECK:STDOUT:   %impl.elem0.loc145_43.1 => constants.%impl.elem0.6ef
// CHECK:STDOUT:   %Result.as_type.loc145_53.1 => constants.%Int32
// CHECK:STDOUT:   %.Self => constants.%.Self.cb9
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.loc145_43.2 => constants.%MulWith.lookup_impl_witness.4ca
// CHECK:STDOUT:   %impl.elem0.loc145_43.2 => constants.%impl.elem0.311
// CHECK:STDOUT:   %facet_type.loc145_37 => constants.%facet_type.b52
// CHECK:STDOUT:   %pattern_type.loc145 => constants.%pattern_type.f70
// CHECK:STDOUT:   %T.patt.loc145_4.2 => constants.%T.patt.040
// CHECK:STDOUT:   %T.loc145_4.1 => constants.%facet_value.1fb
// CHECK:STDOUT:   %pattern_type.loc146_16 => constants.%pattern_type.8b4
// CHECK:STDOUT:   %result.param_patt.loc146_16.2 => constants.%result.param_patt.e81
// CHECK:STDOUT:   %result.patt.loc146_16.2 => constants.%result.patt.47f
// CHECK:STDOUT:   %T.as_type.loc146_29.1 => constants.%Int16
// CHECK:STDOUT:   %pattern_type.loc146_27 => constants.%pattern_type.785
// CHECK:STDOUT:   %x.param_patt.loc146_27.2 => constants.%x.param_patt.3b0
// CHECK:STDOUT:   %x.patt.loc146_27.2 => constants.%x.patt.869
// CHECK:STDOUT:   %pattern_type.loc146_33 => constants.%pattern_type.8b4
// CHECK:STDOUT:   %y.param_patt.loc146_33.2 => constants.%y.param_patt.e0c
// CHECK:STDOUT:   %y.patt.loc146_33.2 => constants.%y.patt.9e1
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc146_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc146_27 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc146_33 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc148 => constants.%complete_type.920626.2
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.349
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.loc148 => constants.%custom_witness.3a0675.1
// CHECK:STDOUT:   %MulWith.facet => constants.%MulWith.facet.4b9
// CHECK:STDOUT:   %MulWith.WithSelf.Op.type => constants.%MulWith.WithSelf.Op.type.8f2
// CHECK:STDOUT:   %.loc148_5.6 => constants.%.47a1
// CHECK:STDOUT:   %impl.elem1.loc148_5.2 => constants.%Int16.Op.4dca31.9
// CHECK:STDOUT:   %specific_impl_fn.loc148_5.2 => constants.%Int16.Op.4dca31.9
// CHECK:STDOUT:   %SelfDestruct.specific_fn.2 => constants.%SelfDestruct.specific_fn.13e
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @DivWith.loc157(constants.%Destroy.facet.806, constants.%Destroy.facet.806, constants.%facet_value.807) {
// CHECK:STDOUT:   %Result.patt.loc154_9.2 => constants.%Result.patt
// CHECK:STDOUT:   %Result.loc154_9.1 => constants.%Destroy.facet.806
// CHECK:STDOUT:   %U.patt.loc155_4.2 => constants.%U.patt.cb8
// CHECK:STDOUT:   %U.loc155_4.1 => constants.%Destroy.facet.806
// CHECK:STDOUT:   %U.as_type.loc156_35.1 => constants.%Int32
// CHECK:STDOUT:   %DivWith.type.loc156_35.1 => constants.%DivWith.type.5a1
// CHECK:STDOUT:   %facet_type.loc156_19 => constants.%facet_type.5a1
// CHECK:STDOUT:   %.Self.frozen.loc156_37.1 => constants.%.Self.frozen.057
// CHECK:STDOUT:   %require_complete.loc156 => constants.%complete_type.348fba.1
// CHECK:STDOUT:   %.Self.frozen.as_type => constants.%.Self.frozen.as_type.cc3
// CHECK:STDOUT:   %DivWith.assoc_type => constants.%DivWith.assoc_type.22f
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.a9d
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.loc156_43.1 => constants.%DivWith.lookup_impl_witness.ca2
// CHECK:STDOUT:   %impl.elem0.loc156_43.1 => constants.%impl.elem0.14b
// CHECK:STDOUT:   %Result.as_type.loc156_53.1 => constants.%Int32
// CHECK:STDOUT:   %.Self => constants.%.Self.208
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.loc156_43.2 => constants.%DivWith.lookup_impl_witness.0ac
// CHECK:STDOUT:   %impl.elem0.loc156_43.2 => constants.%impl.elem0.437
// CHECK:STDOUT:   %facet_type.loc156_37 => constants.%facet_type.f29
// CHECK:STDOUT:   %pattern_type.loc156 => constants.%pattern_type.3d3
// CHECK:STDOUT:   %T.patt.loc156_4.2 => constants.%T.patt.db7
// CHECK:STDOUT:   %T.loc156_4.1 => constants.%facet_value.807
// CHECK:STDOUT:   %pattern_type.loc157_16 => constants.%pattern_type.8b4
// CHECK:STDOUT:   %result.param_patt.loc157_16.2 => constants.%result.param_patt.e81
// CHECK:STDOUT:   %result.patt.loc157_16.2 => constants.%result.patt.47f
// CHECK:STDOUT:   %T.as_type.loc157_29.1 => constants.%Int16
// CHECK:STDOUT:   %pattern_type.loc157_27 => constants.%pattern_type.785
// CHECK:STDOUT:   %x.param_patt.loc157_27.2 => constants.%x.param_patt.3b0
// CHECK:STDOUT:   %x.patt.loc157_27.2 => constants.%x.patt.869
// CHECK:STDOUT:   %pattern_type.loc157_33 => constants.%pattern_type.8b4
// CHECK:STDOUT:   %y.param_patt.loc157_33.2 => constants.%y.param_patt.e0c
// CHECK:STDOUT:   %y.patt.loc157_33.2 => constants.%y.patt.9e1
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc157_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc157_27 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc157_33 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc159 => constants.%complete_type.348fba.2
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.004
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.loc159 => constants.%custom_witness.c32685.1
// CHECK:STDOUT:   %DivWith.facet => constants.%DivWith.facet.5c5
// CHECK:STDOUT:   %DivWith.WithSelf.Op.type => constants.%DivWith.WithSelf.Op.type.af3
// CHECK:STDOUT:   %.loc159_5.6 => constants.%.836
// CHECK:STDOUT:   %impl.elem1.loc159_5.2 => constants.%Int16.Op.4dca31.10
// CHECK:STDOUT:   %specific_impl_fn.loc159_5.2 => constants.%Int16.Op.4dca31.10
// CHECK:STDOUT:   %SelfDestruct.specific_fn.2 => constants.%SelfDestruct.specific_fn.13e
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ModWith.loc168(constants.%Destroy.facet.806, constants.%Destroy.facet.806, constants.%facet_value.c9f) {
// CHECK:STDOUT:   %Result.patt.loc165_9.2 => constants.%Result.patt
// CHECK:STDOUT:   %Result.loc165_9.1 => constants.%Destroy.facet.806
// CHECK:STDOUT:   %U.patt.loc166_4.2 => constants.%U.patt.cb8
// CHECK:STDOUT:   %U.loc166_4.1 => constants.%Destroy.facet.806
// CHECK:STDOUT:   %U.as_type.loc167_35.1 => constants.%Int32
// CHECK:STDOUT:   %ModWith.type.loc167_35.1 => constants.%ModWith.type.794
// CHECK:STDOUT:   %facet_type.loc167_19 => constants.%facet_type.794
// CHECK:STDOUT:   %.Self.frozen.loc167_37.1 => constants.%.Self.frozen.813
// CHECK:STDOUT:   %require_complete.loc167 => constants.%complete_type.cc95ce.1
// CHECK:STDOUT:   %.Self.frozen.as_type => constants.%.Self.frozen.as_type.f4a
// CHECK:STDOUT:   %ModWith.assoc_type => constants.%ModWith.assoc_type.021
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.310
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.loc167_43.1 => constants.%ModWith.lookup_impl_witness.efb
// CHECK:STDOUT:   %impl.elem0.loc167_43.1 => constants.%impl.elem0.e83
// CHECK:STDOUT:   %Result.as_type.loc167_53.1 => constants.%Int32
// CHECK:STDOUT:   %.Self => constants.%.Self.618
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.loc167_43.2 => constants.%ModWith.lookup_impl_witness.01c
// CHECK:STDOUT:   %impl.elem0.loc167_43.2 => constants.%impl.elem0.3bb
// CHECK:STDOUT:   %facet_type.loc167_37 => constants.%facet_type.3d9
// CHECK:STDOUT:   %pattern_type.loc167 => constants.%pattern_type.f5f
// CHECK:STDOUT:   %T.patt.loc167_4.2 => constants.%T.patt.27e
// CHECK:STDOUT:   %T.loc167_4.1 => constants.%facet_value.c9f
// CHECK:STDOUT:   %pattern_type.loc168_16 => constants.%pattern_type.8b4
// CHECK:STDOUT:   %result.param_patt.loc168_16.2 => constants.%result.param_patt.e81
// CHECK:STDOUT:   %result.patt.loc168_16.2 => constants.%result.patt.47f
// CHECK:STDOUT:   %T.as_type.loc168_29.1 => constants.%Int16
// CHECK:STDOUT:   %pattern_type.loc168_27 => constants.%pattern_type.785
// CHECK:STDOUT:   %x.param_patt.loc168_27.2 => constants.%x.param_patt.3b0
// CHECK:STDOUT:   %x.patt.loc168_27.2 => constants.%x.patt.869
// CHECK:STDOUT:   %pattern_type.loc168_33 => constants.%pattern_type.8b4
// CHECK:STDOUT:   %y.param_patt.loc168_33.2 => constants.%y.param_patt.e0c
// CHECK:STDOUT:   %y.patt.loc168_33.2 => constants.%y.patt.9e1
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc168_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc168_27 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc168_33 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc170 => constants.%complete_type.cc95ce.2
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.2eb
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.loc170 => constants.%custom_witness.994c75.1
// CHECK:STDOUT:   %ModWith.facet => constants.%ModWith.facet.cc5
// CHECK:STDOUT:   %ModWith.WithSelf.Op.type => constants.%ModWith.WithSelf.Op.type.ab7
// CHECK:STDOUT:   %.loc170_5.6 => constants.%.2a3
// CHECK:STDOUT:   %impl.elem1.loc170_5.2 => constants.%Int16.Op.4dca31.11
// CHECK:STDOUT:   %specific_impl_fn.loc170_5.2 => constants.%Int16.Op.4dca31.11
// CHECK:STDOUT:   %SelfDestruct.specific_fn.2 => constants.%SelfDestruct.specific_fn.13e
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AddWith.loc124(constants.%Destroy.facet.806, constants.%Destroy.facet.c1d, constants.%facet_value.0f6) {
// CHECK:STDOUT:   %Result.patt.loc121_9.2 => constants.%Result.patt
// CHECK:STDOUT:   %Result.loc121_9.1 => constants.%Destroy.facet.806
// CHECK:STDOUT:   %U.patt.loc122_4.2 => constants.%U.patt.cb8
// CHECK:STDOUT:   %U.loc122_4.1 => constants.%Destroy.facet.c1d
// CHECK:STDOUT:   %U.as_type.loc123_35.1 => constants.%Int16
// CHECK:STDOUT:   %AddWith.type.loc123_35.1 => constants.%AddWith.type.e4b
// CHECK:STDOUT:   %facet_type.loc123_19 => constants.%facet_type.e4b
// CHECK:STDOUT:   %.Self.frozen.loc123_37.1 => constants.%.Self.frozen.ea9
// CHECK:STDOUT:   %require_complete.loc123 => constants.%complete_type.f37bc8.1
// CHECK:STDOUT:   %.Self.frozen.as_type => constants.%.Self.frozen.as_type.ceb
// CHECK:STDOUT:   %AddWith.assoc_type => constants.%AddWith.assoc_type.86e
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.30c
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.loc123_43.1 => constants.%AddWith.lookup_impl_witness.e18
// CHECK:STDOUT:   %impl.elem0.loc123_43.1 => constants.%impl.elem0.de3
// CHECK:STDOUT:   %Result.as_type.loc123_53.1 => constants.%Int32
// CHECK:STDOUT:   %.Self => constants.%.Self.033
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.loc123_43.2 => constants.%AddWith.lookup_impl_witness.73e
// CHECK:STDOUT:   %impl.elem0.loc123_43.2 => constants.%impl.elem0.08e
// CHECK:STDOUT:   %facet_type.loc123_37 => constants.%facet_type.177
// CHECK:STDOUT:   %pattern_type.loc123 => constants.%pattern_type.f16
// CHECK:STDOUT:   %T.patt.loc123_4.2 => constants.%T.patt.4c6
// CHECK:STDOUT:   %T.loc123_4.1 => constants.%facet_value.0f6
// CHECK:STDOUT:   %pattern_type.loc124_16 => constants.%pattern_type.8b4
// CHECK:STDOUT:   %result.param_patt.loc124_16.2 => constants.%result.param_patt.e81
// CHECK:STDOUT:   %result.patt.loc124_16.2 => constants.%result.patt.47f
// CHECK:STDOUT:   %T.as_type.loc124_29.1 => constants.%Int32
// CHECK:STDOUT:   %pattern_type.loc124_27 => constants.%pattern_type.8b4
// CHECK:STDOUT:   %x.param_patt.loc124_27.2 => constants.%x.param_patt.27c
// CHECK:STDOUT:   %x.patt.loc124_27.2 => constants.%x.patt.740
// CHECK:STDOUT:   %pattern_type.loc124_33 => constants.%pattern_type.785
// CHECK:STDOUT:   %y.param_patt.loc124_33.2 => constants.%y.param_patt.418
// CHECK:STDOUT:   %y.patt.loc124_33.2 => constants.%y.patt.890
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc124_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc124_27 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc124_33 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc126 => constants.%complete_type.f37bc8.2
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.9bf
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.loc126 => constants.%custom_witness.ea1
// CHECK:STDOUT:   %AddWith.facet => constants.%AddWith.facet.b31
// CHECK:STDOUT:   %AddWith.WithSelf.Op.type => constants.%AddWith.WithSelf.Op.type.faf
// CHECK:STDOUT:   %.loc126_5.6 => constants.%.fd6
// CHECK:STDOUT:   %impl.elem1.loc126_5.2 => constants.%Int32.Op.4dca31.1
// CHECK:STDOUT:   %specific_impl_fn.loc126_5.2 => constants.%Int32.Op.4dca31.1
// CHECK:STDOUT:   %SelfDestruct.specific_fn.2 => constants.%SelfDestruct.specific_fn.13e
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @SubWith.loc135(constants.%Destroy.facet.806, constants.%Destroy.facet.c1d, constants.%facet_value.50b) {
// CHECK:STDOUT:   %Result.patt.loc132_9.2 => constants.%Result.patt
// CHECK:STDOUT:   %Result.loc132_9.1 => constants.%Destroy.facet.806
// CHECK:STDOUT:   %U.patt.loc133_4.2 => constants.%U.patt.cb8
// CHECK:STDOUT:   %U.loc133_4.1 => constants.%Destroy.facet.c1d
// CHECK:STDOUT:   %U.as_type.loc134_35.1 => constants.%Int16
// CHECK:STDOUT:   %SubWith.type.loc134_35.1 => constants.%SubWith.type.68d
// CHECK:STDOUT:   %facet_type.loc134_19 => constants.%facet_type.68d
// CHECK:STDOUT:   %.Self.frozen.loc134_37.1 => constants.%.Self.frozen.0c9
// CHECK:STDOUT:   %require_complete.loc134 => constants.%complete_type.8a2f63.1
// CHECK:STDOUT:   %.Self.frozen.as_type => constants.%.Self.frozen.as_type.471
// CHECK:STDOUT:   %SubWith.assoc_type => constants.%SubWith.assoc_type.dd6
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.dfb
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.loc134_43.1 => constants.%SubWith.lookup_impl_witness.379
// CHECK:STDOUT:   %impl.elem0.loc134_43.1 => constants.%impl.elem0.e78
// CHECK:STDOUT:   %Result.as_type.loc134_53.1 => constants.%Int32
// CHECK:STDOUT:   %.Self => constants.%.Self.a10
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.loc134_43.2 => constants.%SubWith.lookup_impl_witness.c4d
// CHECK:STDOUT:   %impl.elem0.loc134_43.2 => constants.%impl.elem0.22a
// CHECK:STDOUT:   %facet_type.loc134_37 => constants.%facet_type.760
// CHECK:STDOUT:   %pattern_type.loc134 => constants.%pattern_type.524
// CHECK:STDOUT:   %T.patt.loc134_4.2 => constants.%T.patt.83d
// CHECK:STDOUT:   %T.loc134_4.1 => constants.%facet_value.50b
// CHECK:STDOUT:   %pattern_type.loc135_16 => constants.%pattern_type.8b4
// CHECK:STDOUT:   %result.param_patt.loc135_16.2 => constants.%result.param_patt.e81
// CHECK:STDOUT:   %result.patt.loc135_16.2 => constants.%result.patt.47f
// CHECK:STDOUT:   %T.as_type.loc135_29.1 => constants.%Int32
// CHECK:STDOUT:   %pattern_type.loc135_27 => constants.%pattern_type.8b4
// CHECK:STDOUT:   %x.param_patt.loc135_27.2 => constants.%x.param_patt.27c
// CHECK:STDOUT:   %x.patt.loc135_27.2 => constants.%x.patt.740
// CHECK:STDOUT:   %pattern_type.loc135_33 => constants.%pattern_type.785
// CHECK:STDOUT:   %y.param_patt.loc135_33.2 => constants.%y.param_patt.418
// CHECK:STDOUT:   %y.patt.loc135_33.2 => constants.%y.patt.890
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc135_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc135_27 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc135_33 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc137 => constants.%complete_type.8a2f63.2
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.ada
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.loc137 => constants.%custom_witness.576
// CHECK:STDOUT:   %SubWith.facet => constants.%SubWith.facet.d30
// CHECK:STDOUT:   %SubWith.WithSelf.Op.type => constants.%SubWith.WithSelf.Op.type.6e5
// CHECK:STDOUT:   %.loc137_5.6 => constants.%.d1a
// CHECK:STDOUT:   %impl.elem1.loc137_5.2 => constants.%Int32.Op.4dca31.2
// CHECK:STDOUT:   %specific_impl_fn.loc137_5.2 => constants.%Int32.Op.4dca31.2
// CHECK:STDOUT:   %SelfDestruct.specific_fn.2 => constants.%SelfDestruct.specific_fn.13e
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @MulWith.loc146(constants.%Destroy.facet.806, constants.%Destroy.facet.c1d, constants.%facet_value.457) {
// CHECK:STDOUT:   %Result.patt.loc143_9.2 => constants.%Result.patt
// CHECK:STDOUT:   %Result.loc143_9.1 => constants.%Destroy.facet.806
// CHECK:STDOUT:   %U.patt.loc144_4.2 => constants.%U.patt.cb8
// CHECK:STDOUT:   %U.loc144_4.1 => constants.%Destroy.facet.c1d
// CHECK:STDOUT:   %U.as_type.loc145_35.1 => constants.%Int16
// CHECK:STDOUT:   %MulWith.type.loc145_35.1 => constants.%MulWith.type.f65
// CHECK:STDOUT:   %facet_type.loc145_19 => constants.%facet_type.f65
// CHECK:STDOUT:   %.Self.frozen.loc145_37.1 => constants.%.Self.frozen.aec
// CHECK:STDOUT:   %require_complete.loc145 => constants.%complete_type.8c670c.1
// CHECK:STDOUT:   %.Self.frozen.as_type => constants.%.Self.frozen.as_type.34d
// CHECK:STDOUT:   %MulWith.assoc_type => constants.%MulWith.assoc_type.4c3
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.6ec
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.loc145_43.1 => constants.%MulWith.lookup_impl_witness.94b
// CHECK:STDOUT:   %impl.elem0.loc145_43.1 => constants.%impl.elem0.212
// CHECK:STDOUT:   %Result.as_type.loc145_53.1 => constants.%Int32
// CHECK:STDOUT:   %.Self => constants.%.Self.1e8
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.loc145_43.2 => constants.%MulWith.lookup_impl_witness.b4a
// CHECK:STDOUT:   %impl.elem0.loc145_43.2 => constants.%impl.elem0.088
// CHECK:STDOUT:   %facet_type.loc145_37 => constants.%facet_type.1cc
// CHECK:STDOUT:   %pattern_type.loc145 => constants.%pattern_type.89d
// CHECK:STDOUT:   %T.patt.loc145_4.2 => constants.%T.patt.b89
// CHECK:STDOUT:   %T.loc145_4.1 => constants.%facet_value.457
// CHECK:STDOUT:   %pattern_type.loc146_16 => constants.%pattern_type.8b4
// CHECK:STDOUT:   %result.param_patt.loc146_16.2 => constants.%result.param_patt.e81
// CHECK:STDOUT:   %result.patt.loc146_16.2 => constants.%result.patt.47f
// CHECK:STDOUT:   %T.as_type.loc146_29.1 => constants.%Int32
// CHECK:STDOUT:   %pattern_type.loc146_27 => constants.%pattern_type.8b4
// CHECK:STDOUT:   %x.param_patt.loc146_27.2 => constants.%x.param_patt.27c
// CHECK:STDOUT:   %x.patt.loc146_27.2 => constants.%x.patt.740
// CHECK:STDOUT:   %pattern_type.loc146_33 => constants.%pattern_type.785
// CHECK:STDOUT:   %y.param_patt.loc146_33.2 => constants.%y.param_patt.418
// CHECK:STDOUT:   %y.patt.loc146_33.2 => constants.%y.patt.890
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc146_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc146_27 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc146_33 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc148 => constants.%complete_type.8c670c.2
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.b14
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.loc148 => constants.%custom_witness.751
// CHECK:STDOUT:   %MulWith.facet => constants.%MulWith.facet.cd8
// CHECK:STDOUT:   %MulWith.WithSelf.Op.type => constants.%MulWith.WithSelf.Op.type.31d
// CHECK:STDOUT:   %.loc148_5.6 => constants.%.ba7
// CHECK:STDOUT:   %impl.elem1.loc148_5.2 => constants.%Int32.Op.4dca31.3
// CHECK:STDOUT:   %specific_impl_fn.loc148_5.2 => constants.%Int32.Op.4dca31.3
// CHECK:STDOUT:   %SelfDestruct.specific_fn.2 => constants.%SelfDestruct.specific_fn.13e
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @DivWith.loc157(constants.%Destroy.facet.806, constants.%Destroy.facet.c1d, constants.%facet_value.134) {
// CHECK:STDOUT:   %Result.patt.loc154_9.2 => constants.%Result.patt
// CHECK:STDOUT:   %Result.loc154_9.1 => constants.%Destroy.facet.806
// CHECK:STDOUT:   %U.patt.loc155_4.2 => constants.%U.patt.cb8
// CHECK:STDOUT:   %U.loc155_4.1 => constants.%Destroy.facet.c1d
// CHECK:STDOUT:   %U.as_type.loc156_35.1 => constants.%Int16
// CHECK:STDOUT:   %DivWith.type.loc156_35.1 => constants.%DivWith.type.b8a
// CHECK:STDOUT:   %facet_type.loc156_19 => constants.%facet_type.b8a
// CHECK:STDOUT:   %.Self.frozen.loc156_37.1 => constants.%.Self.frozen.ff8
// CHECK:STDOUT:   %require_complete.loc156 => constants.%complete_type.d99d66.1
// CHECK:STDOUT:   %.Self.frozen.as_type => constants.%.Self.frozen.as_type.2d9
// CHECK:STDOUT:   %DivWith.assoc_type => constants.%DivWith.assoc_type.712
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.99c
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.loc156_43.1 => constants.%DivWith.lookup_impl_witness.483
// CHECK:STDOUT:   %impl.elem0.loc156_43.1 => constants.%impl.elem0.f9d
// CHECK:STDOUT:   %Result.as_type.loc156_53.1 => constants.%Int32
// CHECK:STDOUT:   %.Self => constants.%.Self.74d
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.loc156_43.2 => constants.%DivWith.lookup_impl_witness.afb
// CHECK:STDOUT:   %impl.elem0.loc156_43.2 => constants.%impl.elem0.6cf
// CHECK:STDOUT:   %facet_type.loc156_37 => constants.%facet_type.8ff
// CHECK:STDOUT:   %pattern_type.loc156 => constants.%pattern_type.d01
// CHECK:STDOUT:   %T.patt.loc156_4.2 => constants.%T.patt.832
// CHECK:STDOUT:   %T.loc156_4.1 => constants.%facet_value.134
// CHECK:STDOUT:   %pattern_type.loc157_16 => constants.%pattern_type.8b4
// CHECK:STDOUT:   %result.param_patt.loc157_16.2 => constants.%result.param_patt.e81
// CHECK:STDOUT:   %result.patt.loc157_16.2 => constants.%result.patt.47f
// CHECK:STDOUT:   %T.as_type.loc157_29.1 => constants.%Int32
// CHECK:STDOUT:   %pattern_type.loc157_27 => constants.%pattern_type.8b4
// CHECK:STDOUT:   %x.param_patt.loc157_27.2 => constants.%x.param_patt.27c
// CHECK:STDOUT:   %x.patt.loc157_27.2 => constants.%x.patt.740
// CHECK:STDOUT:   %pattern_type.loc157_33 => constants.%pattern_type.785
// CHECK:STDOUT:   %y.param_patt.loc157_33.2 => constants.%y.param_patt.418
// CHECK:STDOUT:   %y.patt.loc157_33.2 => constants.%y.patt.890
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc157_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc157_27 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc157_33 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc159 => constants.%complete_type.d99d66.2
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.fb6
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.loc159 => constants.%custom_witness.9ad
// CHECK:STDOUT:   %DivWith.facet => constants.%DivWith.facet.d6b
// CHECK:STDOUT:   %DivWith.WithSelf.Op.type => constants.%DivWith.WithSelf.Op.type.b35
// CHECK:STDOUT:   %.loc159_5.6 => constants.%.636
// CHECK:STDOUT:   %impl.elem1.loc159_5.2 => constants.%Int32.Op.4dca31.4
// CHECK:STDOUT:   %specific_impl_fn.loc159_5.2 => constants.%Int32.Op.4dca31.4
// CHECK:STDOUT:   %SelfDestruct.specific_fn.2 => constants.%SelfDestruct.specific_fn.13e
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ModWith.loc168(constants.%Destroy.facet.806, constants.%Destroy.facet.c1d, constants.%facet_value.158) {
// CHECK:STDOUT:   %Result.patt.loc165_9.2 => constants.%Result.patt
// CHECK:STDOUT:   %Result.loc165_9.1 => constants.%Destroy.facet.806
// CHECK:STDOUT:   %U.patt.loc166_4.2 => constants.%U.patt.cb8
// CHECK:STDOUT:   %U.loc166_4.1 => constants.%Destroy.facet.c1d
// CHECK:STDOUT:   %U.as_type.loc167_35.1 => constants.%Int16
// CHECK:STDOUT:   %ModWith.type.loc167_35.1 => constants.%ModWith.type.c67
// CHECK:STDOUT:   %facet_type.loc167_19 => constants.%facet_type.c67
// CHECK:STDOUT:   %.Self.frozen.loc167_37.1 => constants.%.Self.frozen.13f
// CHECK:STDOUT:   %require_complete.loc167 => constants.%complete_type.7d80e1.1
// CHECK:STDOUT:   %.Self.frozen.as_type => constants.%.Self.frozen.as_type.cb4
// CHECK:STDOUT:   %ModWith.assoc_type => constants.%ModWith.assoc_type.5f6
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.f04
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.loc167_43.1 => constants.%ModWith.lookup_impl_witness.0af
// CHECK:STDOUT:   %impl.elem0.loc167_43.1 => constants.%impl.elem0.d91
// CHECK:STDOUT:   %Result.as_type.loc167_53.1 => constants.%Int32
// CHECK:STDOUT:   %.Self => constants.%.Self.71c
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.loc167_43.2 => constants.%ModWith.lookup_impl_witness.c2a
// CHECK:STDOUT:   %impl.elem0.loc167_43.2 => constants.%impl.elem0.14c
// CHECK:STDOUT:   %facet_type.loc167_37 => constants.%facet_type.960
// CHECK:STDOUT:   %pattern_type.loc167 => constants.%pattern_type.3be
// CHECK:STDOUT:   %T.patt.loc167_4.2 => constants.%T.patt.c9a
// CHECK:STDOUT:   %T.loc167_4.1 => constants.%facet_value.158
// CHECK:STDOUT:   %pattern_type.loc168_16 => constants.%pattern_type.8b4
// CHECK:STDOUT:   %result.param_patt.loc168_16.2 => constants.%result.param_patt.e81
// CHECK:STDOUT:   %result.patt.loc168_16.2 => constants.%result.patt.47f
// CHECK:STDOUT:   %T.as_type.loc168_29.1 => constants.%Int32
// CHECK:STDOUT:   %pattern_type.loc168_27 => constants.%pattern_type.8b4
// CHECK:STDOUT:   %x.param_patt.loc168_27.2 => constants.%x.param_patt.27c
// CHECK:STDOUT:   %x.patt.loc168_27.2 => constants.%x.patt.740
// CHECK:STDOUT:   %pattern_type.loc168_33 => constants.%pattern_type.785
// CHECK:STDOUT:   %y.param_patt.loc168_33.2 => constants.%y.param_patt.418
// CHECK:STDOUT:   %y.patt.loc168_33.2 => constants.%y.patt.890
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc168_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc168_27 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc168_33 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc170 => constants.%complete_type.7d80e1.2
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.d19
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.loc170 => constants.%custom_witness.8a2
// CHECK:STDOUT:   %ModWith.facet => constants.%ModWith.facet.9d1
// CHECK:STDOUT:   %ModWith.WithSelf.Op.type => constants.%ModWith.WithSelf.Op.type.e61
// CHECK:STDOUT:   %.loc170_5.6 => constants.%.25b
// CHECK:STDOUT:   %impl.elem1.loc170_5.2 => constants.%Int32.Op.4dca31.5
// CHECK:STDOUT:   %specific_impl_fn.loc170_5.2 => constants.%Int32.Op.4dca31.5
// CHECK:STDOUT:   %SelfDestruct.specific_fn.2 => constants.%SelfDestruct.specific_fn.13e
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AddWith.loc124(constants.%Destroy.facet.806, constants.%Destroy.facet.806, constants.%facet_value.b57) {
// CHECK:STDOUT:   %Result.patt.loc121_9.2 => constants.%Result.patt
// CHECK:STDOUT:   %Result.loc121_9.1 => constants.%Destroy.facet.806
// CHECK:STDOUT:   %U.patt.loc122_4.2 => constants.%U.patt.cb8
// CHECK:STDOUT:   %U.loc122_4.1 => constants.%Destroy.facet.806
// CHECK:STDOUT:   %U.as_type.loc123_35.1 => constants.%Int32
// CHECK:STDOUT:   %AddWith.type.loc123_35.1 => constants.%AddWith.type.db7
// CHECK:STDOUT:   %facet_type.loc123_19 => constants.%facet_type.db7
// CHECK:STDOUT:   %.Self.frozen.loc123_37.1 => constants.%.Self.frozen.bc4
// CHECK:STDOUT:   %require_complete.loc123 => constants.%complete_type.758ea8.1
// CHECK:STDOUT:   %.Self.frozen.as_type => constants.%.Self.frozen.as_type.447
// CHECK:STDOUT:   %AddWith.assoc_type => constants.%AddWith.assoc_type.be8
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.31c
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.loc123_43.1 => constants.%AddWith.lookup_impl_witness.a9e
// CHECK:STDOUT:   %impl.elem0.loc123_43.1 => constants.%impl.elem0.517
// CHECK:STDOUT:   %Result.as_type.loc123_53.1 => constants.%Int32
// CHECK:STDOUT:   %.Self => constants.%.Self.6a9
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.loc123_43.2 => constants.%AddWith.lookup_impl_witness.f6c
// CHECK:STDOUT:   %impl.elem0.loc123_43.2 => constants.%impl.elem0.a74
// CHECK:STDOUT:   %facet_type.loc123_37 => constants.%facet_type.bf5
// CHECK:STDOUT:   %pattern_type.loc123 => constants.%pattern_type.eea
// CHECK:STDOUT:   %T.patt.loc123_4.2 => constants.%T.patt.79b
// CHECK:STDOUT:   %T.loc123_4.1 => constants.%facet_value.b57
// CHECK:STDOUT:   %pattern_type.loc124_16 => constants.%pattern_type.8b4
// CHECK:STDOUT:   %result.param_patt.loc124_16.2 => constants.%result.param_patt.e81
// CHECK:STDOUT:   %result.patt.loc124_16.2 => constants.%result.patt.47f
// CHECK:STDOUT:   %T.as_type.loc124_29.1 => constants.%Int32
// CHECK:STDOUT:   %pattern_type.loc124_27 => constants.%pattern_type.8b4
// CHECK:STDOUT:   %x.param_patt.loc124_27.2 => constants.%x.param_patt.27c
// CHECK:STDOUT:   %x.patt.loc124_27.2 => constants.%x.patt.740
// CHECK:STDOUT:   %pattern_type.loc124_33 => constants.%pattern_type.8b4
// CHECK:STDOUT:   %y.param_patt.loc124_33.2 => constants.%y.param_patt.e0c
// CHECK:STDOUT:   %y.patt.loc124_33.2 => constants.%y.patt.9e1
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc124_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc124_27 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc124_33 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc126 => constants.%complete_type.758ea8.2
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.86c
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.loc126 => constants.%custom_witness.513476.2
// CHECK:STDOUT:   %AddWith.facet => constants.%AddWith.facet.684
// CHECK:STDOUT:   %AddWith.WithSelf.Op.type => constants.%AddWith.WithSelf.Op.type.0af
// CHECK:STDOUT:   %.loc126_5.6 => constants.%.e92
// CHECK:STDOUT:   %impl.elem1.loc126_5.2 => constants.%Int32.Op.4dca31.6
// CHECK:STDOUT:   %specific_impl_fn.loc126_5.2 => constants.%Int32.Op.4dca31.6
// CHECK:STDOUT:   %SelfDestruct.specific_fn.2 => constants.%SelfDestruct.specific_fn.13e
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @SubWith.loc135(constants.%Destroy.facet.806, constants.%Destroy.facet.806, constants.%facet_value.748) {
// CHECK:STDOUT:   %Result.patt.loc132_9.2 => constants.%Result.patt
// CHECK:STDOUT:   %Result.loc132_9.1 => constants.%Destroy.facet.806
// CHECK:STDOUT:   %U.patt.loc133_4.2 => constants.%U.patt.cb8
// CHECK:STDOUT:   %U.loc133_4.1 => constants.%Destroy.facet.806
// CHECK:STDOUT:   %U.as_type.loc134_35.1 => constants.%Int32
// CHECK:STDOUT:   %SubWith.type.loc134_35.1 => constants.%SubWith.type.120
// CHECK:STDOUT:   %facet_type.loc134_19 => constants.%facet_type.120
// CHECK:STDOUT:   %.Self.frozen.loc134_37.1 => constants.%.Self.frozen.faf
// CHECK:STDOUT:   %require_complete.loc134 => constants.%complete_type.46ea3c.1
// CHECK:STDOUT:   %.Self.frozen.as_type => constants.%.Self.frozen.as_type.ea6
// CHECK:STDOUT:   %SubWith.assoc_type => constants.%SubWith.assoc_type.dae
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.fac
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.loc134_43.1 => constants.%SubWith.lookup_impl_witness.c4f
// CHECK:STDOUT:   %impl.elem0.loc134_43.1 => constants.%impl.elem0.b5a
// CHECK:STDOUT:   %Result.as_type.loc134_53.1 => constants.%Int32
// CHECK:STDOUT:   %.Self => constants.%.Self.861
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.loc134_43.2 => constants.%SubWith.lookup_impl_witness.4cf
// CHECK:STDOUT:   %impl.elem0.loc134_43.2 => constants.%impl.elem0.94c
// CHECK:STDOUT:   %facet_type.loc134_37 => constants.%facet_type.b03
// CHECK:STDOUT:   %pattern_type.loc134 => constants.%pattern_type.110
// CHECK:STDOUT:   %T.patt.loc134_4.2 => constants.%T.patt.e31
// CHECK:STDOUT:   %T.loc134_4.1 => constants.%facet_value.748
// CHECK:STDOUT:   %pattern_type.loc135_16 => constants.%pattern_type.8b4
// CHECK:STDOUT:   %result.param_patt.loc135_16.2 => constants.%result.param_patt.e81
// CHECK:STDOUT:   %result.patt.loc135_16.2 => constants.%result.patt.47f
// CHECK:STDOUT:   %T.as_type.loc135_29.1 => constants.%Int32
// CHECK:STDOUT:   %pattern_type.loc135_27 => constants.%pattern_type.8b4
// CHECK:STDOUT:   %x.param_patt.loc135_27.2 => constants.%x.param_patt.27c
// CHECK:STDOUT:   %x.patt.loc135_27.2 => constants.%x.patt.740
// CHECK:STDOUT:   %pattern_type.loc135_33 => constants.%pattern_type.8b4
// CHECK:STDOUT:   %y.param_patt.loc135_33.2 => constants.%y.param_patt.e0c
// CHECK:STDOUT:   %y.patt.loc135_33.2 => constants.%y.patt.9e1
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc135_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc135_27 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc135_33 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc137 => constants.%complete_type.46ea3c.2
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.147
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.loc137 => constants.%custom_witness.32cef1.2
// CHECK:STDOUT:   %SubWith.facet => constants.%SubWith.facet.a46
// CHECK:STDOUT:   %SubWith.WithSelf.Op.type => constants.%SubWith.WithSelf.Op.type.1a5
// CHECK:STDOUT:   %.loc137_5.6 => constants.%.457
// CHECK:STDOUT:   %impl.elem1.loc137_5.2 => constants.%Int32.Op.4dca31.7
// CHECK:STDOUT:   %specific_impl_fn.loc137_5.2 => constants.%Int32.Op.4dca31.7
// CHECK:STDOUT:   %SelfDestruct.specific_fn.2 => constants.%SelfDestruct.specific_fn.13e
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @MulWith.loc146(constants.%Destroy.facet.806, constants.%Destroy.facet.806, constants.%facet_value.b23) {
// CHECK:STDOUT:   %Result.patt.loc143_9.2 => constants.%Result.patt
// CHECK:STDOUT:   %Result.loc143_9.1 => constants.%Destroy.facet.806
// CHECK:STDOUT:   %U.patt.loc144_4.2 => constants.%U.patt.cb8
// CHECK:STDOUT:   %U.loc144_4.1 => constants.%Destroy.facet.806
// CHECK:STDOUT:   %U.as_type.loc145_35.1 => constants.%Int32
// CHECK:STDOUT:   %MulWith.type.loc145_35.1 => constants.%MulWith.type.d67
// CHECK:STDOUT:   %facet_type.loc145_19 => constants.%facet_type.d67b
// CHECK:STDOUT:   %.Self.frozen.loc145_37.1 => constants.%.Self.frozen.c7a
// CHECK:STDOUT:   %require_complete.loc145 => constants.%complete_type.920626.1
// CHECK:STDOUT:   %.Self.frozen.as_type => constants.%.Self.frozen.as_type.c47
// CHECK:STDOUT:   %MulWith.assoc_type => constants.%MulWith.assoc_type.ab1
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.d0a
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.loc145_43.1 => constants.%MulWith.lookup_impl_witness.ab5
// CHECK:STDOUT:   %impl.elem0.loc145_43.1 => constants.%impl.elem0.6ef
// CHECK:STDOUT:   %Result.as_type.loc145_53.1 => constants.%Int32
// CHECK:STDOUT:   %.Self => constants.%.Self.cb9
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.loc145_43.2 => constants.%MulWith.lookup_impl_witness.4ca
// CHECK:STDOUT:   %impl.elem0.loc145_43.2 => constants.%impl.elem0.311
// CHECK:STDOUT:   %facet_type.loc145_37 => constants.%facet_type.b52
// CHECK:STDOUT:   %pattern_type.loc145 => constants.%pattern_type.f70
// CHECK:STDOUT:   %T.patt.loc145_4.2 => constants.%T.patt.040
// CHECK:STDOUT:   %T.loc145_4.1 => constants.%facet_value.b23
// CHECK:STDOUT:   %pattern_type.loc146_16 => constants.%pattern_type.8b4
// CHECK:STDOUT:   %result.param_patt.loc146_16.2 => constants.%result.param_patt.e81
// CHECK:STDOUT:   %result.patt.loc146_16.2 => constants.%result.patt.47f
// CHECK:STDOUT:   %T.as_type.loc146_29.1 => constants.%Int32
// CHECK:STDOUT:   %pattern_type.loc146_27 => constants.%pattern_type.8b4
// CHECK:STDOUT:   %x.param_patt.loc146_27.2 => constants.%x.param_patt.27c
// CHECK:STDOUT:   %x.patt.loc146_27.2 => constants.%x.patt.740
// CHECK:STDOUT:   %pattern_type.loc146_33 => constants.%pattern_type.8b4
// CHECK:STDOUT:   %y.param_patt.loc146_33.2 => constants.%y.param_patt.e0c
// CHECK:STDOUT:   %y.patt.loc146_33.2 => constants.%y.patt.9e1
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc146_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc146_27 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc146_33 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc148 => constants.%complete_type.920626.2
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.349
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.loc148 => constants.%custom_witness.3a0675.2
// CHECK:STDOUT:   %MulWith.facet => constants.%MulWith.facet.3ab
// CHECK:STDOUT:   %MulWith.WithSelf.Op.type => constants.%MulWith.WithSelf.Op.type.2ed
// CHECK:STDOUT:   %.loc148_5.6 => constants.%.9d6
// CHECK:STDOUT:   %impl.elem1.loc148_5.2 => constants.%Int32.Op.4dca31.8
// CHECK:STDOUT:   %specific_impl_fn.loc148_5.2 => constants.%Int32.Op.4dca31.8
// CHECK:STDOUT:   %SelfDestruct.specific_fn.2 => constants.%SelfDestruct.specific_fn.13e
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @DivWith.loc157(constants.%Destroy.facet.806, constants.%Destroy.facet.806, constants.%facet_value.958) {
// CHECK:STDOUT:   %Result.patt.loc154_9.2 => constants.%Result.patt
// CHECK:STDOUT:   %Result.loc154_9.1 => constants.%Destroy.facet.806
// CHECK:STDOUT:   %U.patt.loc155_4.2 => constants.%U.patt.cb8
// CHECK:STDOUT:   %U.loc155_4.1 => constants.%Destroy.facet.806
// CHECK:STDOUT:   %U.as_type.loc156_35.1 => constants.%Int32
// CHECK:STDOUT:   %DivWith.type.loc156_35.1 => constants.%DivWith.type.5a1
// CHECK:STDOUT:   %facet_type.loc156_19 => constants.%facet_type.5a1
// CHECK:STDOUT:   %.Self.frozen.loc156_37.1 => constants.%.Self.frozen.057
// CHECK:STDOUT:   %require_complete.loc156 => constants.%complete_type.348fba.1
// CHECK:STDOUT:   %.Self.frozen.as_type => constants.%.Self.frozen.as_type.cc3
// CHECK:STDOUT:   %DivWith.assoc_type => constants.%DivWith.assoc_type.22f
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.a9d
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.loc156_43.1 => constants.%DivWith.lookup_impl_witness.ca2
// CHECK:STDOUT:   %impl.elem0.loc156_43.1 => constants.%impl.elem0.14b
// CHECK:STDOUT:   %Result.as_type.loc156_53.1 => constants.%Int32
// CHECK:STDOUT:   %.Self => constants.%.Self.208
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.loc156_43.2 => constants.%DivWith.lookup_impl_witness.0ac
// CHECK:STDOUT:   %impl.elem0.loc156_43.2 => constants.%impl.elem0.437
// CHECK:STDOUT:   %facet_type.loc156_37 => constants.%facet_type.f29
// CHECK:STDOUT:   %pattern_type.loc156 => constants.%pattern_type.3d3
// CHECK:STDOUT:   %T.patt.loc156_4.2 => constants.%T.patt.db7
// CHECK:STDOUT:   %T.loc156_4.1 => constants.%facet_value.958
// CHECK:STDOUT:   %pattern_type.loc157_16 => constants.%pattern_type.8b4
// CHECK:STDOUT:   %result.param_patt.loc157_16.2 => constants.%result.param_patt.e81
// CHECK:STDOUT:   %result.patt.loc157_16.2 => constants.%result.patt.47f
// CHECK:STDOUT:   %T.as_type.loc157_29.1 => constants.%Int32
// CHECK:STDOUT:   %pattern_type.loc157_27 => constants.%pattern_type.8b4
// CHECK:STDOUT:   %x.param_patt.loc157_27.2 => constants.%x.param_patt.27c
// CHECK:STDOUT:   %x.patt.loc157_27.2 => constants.%x.patt.740
// CHECK:STDOUT:   %pattern_type.loc157_33 => constants.%pattern_type.8b4
// CHECK:STDOUT:   %y.param_patt.loc157_33.2 => constants.%y.param_patt.e0c
// CHECK:STDOUT:   %y.patt.loc157_33.2 => constants.%y.patt.9e1
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc157_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc157_27 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc157_33 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc159 => constants.%complete_type.348fba.2
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.004
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.loc159 => constants.%custom_witness.c32685.2
// CHECK:STDOUT:   %DivWith.facet => constants.%DivWith.facet.b4a
// CHECK:STDOUT:   %DivWith.WithSelf.Op.type => constants.%DivWith.WithSelf.Op.type.085
// CHECK:STDOUT:   %.loc159_5.6 => constants.%.10c
// CHECK:STDOUT:   %impl.elem1.loc159_5.2 => constants.%Int32.Op.4dca31.9
// CHECK:STDOUT:   %specific_impl_fn.loc159_5.2 => constants.%Int32.Op.4dca31.9
// CHECK:STDOUT:   %SelfDestruct.specific_fn.2 => constants.%SelfDestruct.specific_fn.13e
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ModWith.loc168(constants.%Destroy.facet.806, constants.%Destroy.facet.806, constants.%facet_value.046) {
// CHECK:STDOUT:   %Result.patt.loc165_9.2 => constants.%Result.patt
// CHECK:STDOUT:   %Result.loc165_9.1 => constants.%Destroy.facet.806
// CHECK:STDOUT:   %U.patt.loc166_4.2 => constants.%U.patt.cb8
// CHECK:STDOUT:   %U.loc166_4.1 => constants.%Destroy.facet.806
// CHECK:STDOUT:   %U.as_type.loc167_35.1 => constants.%Int32
// CHECK:STDOUT:   %ModWith.type.loc167_35.1 => constants.%ModWith.type.794
// CHECK:STDOUT:   %facet_type.loc167_19 => constants.%facet_type.794
// CHECK:STDOUT:   %.Self.frozen.loc167_37.1 => constants.%.Self.frozen.813
// CHECK:STDOUT:   %require_complete.loc167 => constants.%complete_type.cc95ce.1
// CHECK:STDOUT:   %.Self.frozen.as_type => constants.%.Self.frozen.as_type.f4a
// CHECK:STDOUT:   %ModWith.assoc_type => constants.%ModWith.assoc_type.021
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.310
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.loc167_43.1 => constants.%ModWith.lookup_impl_witness.efb
// CHECK:STDOUT:   %impl.elem0.loc167_43.1 => constants.%impl.elem0.e83
// CHECK:STDOUT:   %Result.as_type.loc167_53.1 => constants.%Int32
// CHECK:STDOUT:   %.Self => constants.%.Self.618
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.loc167_43.2 => constants.%ModWith.lookup_impl_witness.01c
// CHECK:STDOUT:   %impl.elem0.loc167_43.2 => constants.%impl.elem0.3bb
// CHECK:STDOUT:   %facet_type.loc167_37 => constants.%facet_type.3d9
// CHECK:STDOUT:   %pattern_type.loc167 => constants.%pattern_type.f5f
// CHECK:STDOUT:   %T.patt.loc167_4.2 => constants.%T.patt.27e
// CHECK:STDOUT:   %T.loc167_4.1 => constants.%facet_value.046
// CHECK:STDOUT:   %pattern_type.loc168_16 => constants.%pattern_type.8b4
// CHECK:STDOUT:   %result.param_patt.loc168_16.2 => constants.%result.param_patt.e81
// CHECK:STDOUT:   %result.patt.loc168_16.2 => constants.%result.patt.47f
// CHECK:STDOUT:   %T.as_type.loc168_29.1 => constants.%Int32
// CHECK:STDOUT:   %pattern_type.loc168_27 => constants.%pattern_type.8b4
// CHECK:STDOUT:   %x.param_patt.loc168_27.2 => constants.%x.param_patt.27c
// CHECK:STDOUT:   %x.patt.loc168_27.2 => constants.%x.patt.740
// CHECK:STDOUT:   %pattern_type.loc168_33 => constants.%pattern_type.8b4
// CHECK:STDOUT:   %y.param_patt.loc168_33.2 => constants.%y.param_patt.e0c
// CHECK:STDOUT:   %y.patt.loc168_33.2 => constants.%y.patt.9e1
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc168_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc168_27 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc168_33 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc170 => constants.%complete_type.cc95ce.2
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.2eb
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.loc170 => constants.%custom_witness.994c75.2
// CHECK:STDOUT:   %ModWith.facet => constants.%ModWith.facet.3dc
// CHECK:STDOUT:   %ModWith.WithSelf.Op.type => constants.%ModWith.WithSelf.Op.type.a2a
// CHECK:STDOUT:   %.loc170_5.6 => constants.%.b6d
// CHECK:STDOUT:   %impl.elem1.loc170_5.2 => constants.%Int32.Op.4dca31.10
// CHECK:STDOUT:   %specific_impl_fn.loc170_5.2 => constants.%Int32.Op.4dca31.10
// CHECK:STDOUT:   %SelfDestruct.specific_fn.2 => constants.%SelfDestruct.specific_fn.13e
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AddWith.loc124(constants.%Destroy.facet.c11, constants.%Destroy.facet.c11, constants.%facet_value.5f6) {
// CHECK:STDOUT:   %Result.patt.loc121_9.2 => constants.%Result.patt
// CHECK:STDOUT:   %Result.loc121_9.1 => constants.%Destroy.facet.c11
// CHECK:STDOUT:   %U.patt.loc122_4.2 => constants.%U.patt.cb8
// CHECK:STDOUT:   %U.loc122_4.1 => constants.%Destroy.facet.c11
// CHECK:STDOUT:   %U.as_type.loc123_35.1 => constants.%Int64
// CHECK:STDOUT:   %AddWith.type.loc123_35.1 => constants.%AddWith.type.1ff
// CHECK:STDOUT:   %facet_type.loc123_19 => constants.%facet_type.1ff
// CHECK:STDOUT:   %.Self.frozen.loc123_37.1 => constants.%.Self.frozen.91a
// CHECK:STDOUT:   %require_complete.loc123 => constants.%complete_type.57be9d.1
// CHECK:STDOUT:   %.Self.frozen.as_type => constants.%.Self.frozen.as_type.a1b
// CHECK:STDOUT:   %AddWith.assoc_type => constants.%AddWith.assoc_type.97f
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.f02
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.loc123_43.1 => constants.%AddWith.lookup_impl_witness.30a
// CHECK:STDOUT:   %impl.elem0.loc123_43.1 => constants.%impl.elem0.b7d
// CHECK:STDOUT:   %Result.as_type.loc123_53.1 => constants.%Int64
// CHECK:STDOUT:   %.Self => constants.%.Self.f92
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.loc123_43.2 => constants.%AddWith.lookup_impl_witness.96d
// CHECK:STDOUT:   %impl.elem0.loc123_43.2 => constants.%impl.elem0.516
// CHECK:STDOUT:   %facet_type.loc123_37 => constants.%facet_type.dab
// CHECK:STDOUT:   %pattern_type.loc123 => constants.%pattern_type.652
// CHECK:STDOUT:   %T.patt.loc123_4.2 => constants.%T.patt.cb1
// CHECK:STDOUT:   %T.loc123_4.1 => constants.%facet_value.5f6
// CHECK:STDOUT:   %pattern_type.loc124_16 => constants.%pattern_type.4aa
// CHECK:STDOUT:   %result.param_patt.loc124_16.2 => constants.%result.param_patt.1e1
// CHECK:STDOUT:   %result.patt.loc124_16.2 => constants.%result.patt.93b
// CHECK:STDOUT:   %T.as_type.loc124_29.1 => constants.%Int32
// CHECK:STDOUT:   %pattern_type.loc124_27 => constants.%pattern_type.8b4
// CHECK:STDOUT:   %x.param_patt.loc124_27.2 => constants.%x.param_patt.27c
// CHECK:STDOUT:   %x.patt.loc124_27.2 => constants.%x.patt.740
// CHECK:STDOUT:   %pattern_type.loc124_33 => constants.%pattern_type.4aa
// CHECK:STDOUT:   %y.param_patt.loc124_33.2 => constants.%y.param_patt.a08
// CHECK:STDOUT:   %y.patt.loc124_33.2 => constants.%y.patt.069
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc124_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc124_27 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc124_33 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc126 => constants.%complete_type.57be9d.2
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.a75
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.loc126 => constants.%custom_witness.f2f
// CHECK:STDOUT:   %AddWith.facet => constants.%AddWith.facet.0d8
// CHECK:STDOUT:   %AddWith.WithSelf.Op.type => constants.%AddWith.WithSelf.Op.type.a40
// CHECK:STDOUT:   %.loc126_5.6 => constants.%.51d
// CHECK:STDOUT:   %impl.elem1.loc126_5.2 => constants.%Op.72717c.6
// CHECK:STDOUT:   %specific_impl_fn.loc126_5.2 => constants.%Op.72717c.6
// CHECK:STDOUT:   %SelfDestruct.specific_fn.2 => constants.%SelfDestruct.specific_fn.fb0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @SubWith.loc135(constants.%Destroy.facet.c11, constants.%Destroy.facet.c11, constants.%facet_value.a96) {
// CHECK:STDOUT:   %Result.patt.loc132_9.2 => constants.%Result.patt
// CHECK:STDOUT:   %Result.loc132_9.1 => constants.%Destroy.facet.c11
// CHECK:STDOUT:   %U.patt.loc133_4.2 => constants.%U.patt.cb8
// CHECK:STDOUT:   %U.loc133_4.1 => constants.%Destroy.facet.c11
// CHECK:STDOUT:   %U.as_type.loc134_35.1 => constants.%Int64
// CHECK:STDOUT:   %SubWith.type.loc134_35.1 => constants.%SubWith.type.31c
// CHECK:STDOUT:   %facet_type.loc134_19 => constants.%facet_type.31c
// CHECK:STDOUT:   %.Self.frozen.loc134_37.1 => constants.%.Self.frozen.c93
// CHECK:STDOUT:   %require_complete.loc134 => constants.%complete_type.cb55bb.1
// CHECK:STDOUT:   %.Self.frozen.as_type => constants.%.Self.frozen.as_type.7cf
// CHECK:STDOUT:   %SubWith.assoc_type => constants.%SubWith.assoc_type.4f8
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.13b
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.loc134_43.1 => constants.%SubWith.lookup_impl_witness.52f
// CHECK:STDOUT:   %impl.elem0.loc134_43.1 => constants.%impl.elem0.5f9
// CHECK:STDOUT:   %Result.as_type.loc134_53.1 => constants.%Int64
// CHECK:STDOUT:   %.Self => constants.%.Self.23f
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.loc134_43.2 => constants.%SubWith.lookup_impl_witness.445
// CHECK:STDOUT:   %impl.elem0.loc134_43.2 => constants.%impl.elem0.ee1
// CHECK:STDOUT:   %facet_type.loc134_37 => constants.%facet_type.d68
// CHECK:STDOUT:   %pattern_type.loc134 => constants.%pattern_type.ea4
// CHECK:STDOUT:   %T.patt.loc134_4.2 => constants.%T.patt.25e
// CHECK:STDOUT:   %T.loc134_4.1 => constants.%facet_value.a96
// CHECK:STDOUT:   %pattern_type.loc135_16 => constants.%pattern_type.4aa
// CHECK:STDOUT:   %result.param_patt.loc135_16.2 => constants.%result.param_patt.1e1
// CHECK:STDOUT:   %result.patt.loc135_16.2 => constants.%result.patt.93b
// CHECK:STDOUT:   %T.as_type.loc135_29.1 => constants.%Int32
// CHECK:STDOUT:   %pattern_type.loc135_27 => constants.%pattern_type.8b4
// CHECK:STDOUT:   %x.param_patt.loc135_27.2 => constants.%x.param_patt.27c
// CHECK:STDOUT:   %x.patt.loc135_27.2 => constants.%x.patt.740
// CHECK:STDOUT:   %pattern_type.loc135_33 => constants.%pattern_type.4aa
// CHECK:STDOUT:   %y.param_patt.loc135_33.2 => constants.%y.param_patt.a08
// CHECK:STDOUT:   %y.patt.loc135_33.2 => constants.%y.patt.069
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc135_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc135_27 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc135_33 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc137 => constants.%complete_type.cb55bb.2
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.e70
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.loc137 => constants.%custom_witness.276
// CHECK:STDOUT:   %SubWith.facet => constants.%SubWith.facet.d62
// CHECK:STDOUT:   %SubWith.WithSelf.Op.type => constants.%SubWith.WithSelf.Op.type.d39
// CHECK:STDOUT:   %.loc137_5.6 => constants.%.e39
// CHECK:STDOUT:   %impl.elem1.loc137_5.2 => constants.%Op.72717c.7
// CHECK:STDOUT:   %specific_impl_fn.loc137_5.2 => constants.%Op.72717c.7
// CHECK:STDOUT:   %SelfDestruct.specific_fn.2 => constants.%SelfDestruct.specific_fn.fb0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @MulWith.loc146(constants.%Destroy.facet.c11, constants.%Destroy.facet.c11, constants.%facet_value.7ed) {
// CHECK:STDOUT:   %Result.patt.loc143_9.2 => constants.%Result.patt
// CHECK:STDOUT:   %Result.loc143_9.1 => constants.%Destroy.facet.c11
// CHECK:STDOUT:   %U.patt.loc144_4.2 => constants.%U.patt.cb8
// CHECK:STDOUT:   %U.loc144_4.1 => constants.%Destroy.facet.c11
// CHECK:STDOUT:   %U.as_type.loc145_35.1 => constants.%Int64
// CHECK:STDOUT:   %MulWith.type.loc145_35.1 => constants.%MulWith.type.661
// CHECK:STDOUT:   %facet_type.loc145_19 => constants.%facet_type.661
// CHECK:STDOUT:   %.Self.frozen.loc145_37.1 => constants.%.Self.frozen.5fe
// CHECK:STDOUT:   %require_complete.loc145 => constants.%complete_type.0a6803.1
// CHECK:STDOUT:   %.Self.frozen.as_type => constants.%.Self.frozen.as_type.e09
// CHECK:STDOUT:   %MulWith.assoc_type => constants.%MulWith.assoc_type.940
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.9d9
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.loc145_43.1 => constants.%MulWith.lookup_impl_witness.6c6
// CHECK:STDOUT:   %impl.elem0.loc145_43.1 => constants.%impl.elem0.731
// CHECK:STDOUT:   %Result.as_type.loc145_53.1 => constants.%Int64
// CHECK:STDOUT:   %.Self => constants.%.Self.6d9
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.loc145_43.2 => constants.%MulWith.lookup_impl_witness.74c
// CHECK:STDOUT:   %impl.elem0.loc145_43.2 => constants.%impl.elem0.4c1
// CHECK:STDOUT:   %facet_type.loc145_37 => constants.%facet_type.a2e
// CHECK:STDOUT:   %pattern_type.loc145 => constants.%pattern_type.216
// CHECK:STDOUT:   %T.patt.loc145_4.2 => constants.%T.patt.ec6
// CHECK:STDOUT:   %T.loc145_4.1 => constants.%facet_value.7ed
// CHECK:STDOUT:   %pattern_type.loc146_16 => constants.%pattern_type.4aa
// CHECK:STDOUT:   %result.param_patt.loc146_16.2 => constants.%result.param_patt.1e1
// CHECK:STDOUT:   %result.patt.loc146_16.2 => constants.%result.patt.93b
// CHECK:STDOUT:   %T.as_type.loc146_29.1 => constants.%Int32
// CHECK:STDOUT:   %pattern_type.loc146_27 => constants.%pattern_type.8b4
// CHECK:STDOUT:   %x.param_patt.loc146_27.2 => constants.%x.param_patt.27c
// CHECK:STDOUT:   %x.patt.loc146_27.2 => constants.%x.patt.740
// CHECK:STDOUT:   %pattern_type.loc146_33 => constants.%pattern_type.4aa
// CHECK:STDOUT:   %y.param_patt.loc146_33.2 => constants.%y.param_patt.a08
// CHECK:STDOUT:   %y.patt.loc146_33.2 => constants.%y.patt.069
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc146_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc146_27 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc146_33 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc148 => constants.%complete_type.0a6803.2
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.b22
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.loc148 => constants.%custom_witness.8e4
// CHECK:STDOUT:   %MulWith.facet => constants.%MulWith.facet.4a0
// CHECK:STDOUT:   %MulWith.WithSelf.Op.type => constants.%MulWith.WithSelf.Op.type.489
// CHECK:STDOUT:   %.loc148_5.6 => constants.%.2e6
// CHECK:STDOUT:   %impl.elem1.loc148_5.2 => constants.%Op.72717c.8
// CHECK:STDOUT:   %specific_impl_fn.loc148_5.2 => constants.%Op.72717c.8
// CHECK:STDOUT:   %SelfDestruct.specific_fn.2 => constants.%SelfDestruct.specific_fn.fb0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @DivWith.loc157(constants.%Destroy.facet.c11, constants.%Destroy.facet.c11, constants.%facet_value.5d8) {
// CHECK:STDOUT:   %Result.patt.loc154_9.2 => constants.%Result.patt
// CHECK:STDOUT:   %Result.loc154_9.1 => constants.%Destroy.facet.c11
// CHECK:STDOUT:   %U.patt.loc155_4.2 => constants.%U.patt.cb8
// CHECK:STDOUT:   %U.loc155_4.1 => constants.%Destroy.facet.c11
// CHECK:STDOUT:   %U.as_type.loc156_35.1 => constants.%Int64
// CHECK:STDOUT:   %DivWith.type.loc156_35.1 => constants.%DivWith.type.dd8
// CHECK:STDOUT:   %facet_type.loc156_19 => constants.%facet_type.dd8
// CHECK:STDOUT:   %.Self.frozen.loc156_37.1 => constants.%.Self.frozen.d10
// CHECK:STDOUT:   %require_complete.loc156 => constants.%complete_type.4732f8.1
// CHECK:STDOUT:   %.Self.frozen.as_type => constants.%.Self.frozen.as_type.483
// CHECK:STDOUT:   %DivWith.assoc_type => constants.%DivWith.assoc_type.2dc
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.e8b
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.loc156_43.1 => constants.%DivWith.lookup_impl_witness.531
// CHECK:STDOUT:   %impl.elem0.loc156_43.1 => constants.%impl.elem0.872
// CHECK:STDOUT:   %Result.as_type.loc156_53.1 => constants.%Int64
// CHECK:STDOUT:   %.Self => constants.%.Self.2e9
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.loc156_43.2 => constants.%DivWith.lookup_impl_witness.14c
// CHECK:STDOUT:   %impl.elem0.loc156_43.2 => constants.%impl.elem0.d16
// CHECK:STDOUT:   %facet_type.loc156_37 => constants.%facet_type.89f
// CHECK:STDOUT:   %pattern_type.loc156 => constants.%pattern_type.fcf
// CHECK:STDOUT:   %T.patt.loc156_4.2 => constants.%T.patt.5ce
// CHECK:STDOUT:   %T.loc156_4.1 => constants.%facet_value.5d8
// CHECK:STDOUT:   %pattern_type.loc157_16 => constants.%pattern_type.4aa
// CHECK:STDOUT:   %result.param_patt.loc157_16.2 => constants.%result.param_patt.1e1
// CHECK:STDOUT:   %result.patt.loc157_16.2 => constants.%result.patt.93b
// CHECK:STDOUT:   %T.as_type.loc157_29.1 => constants.%Int32
// CHECK:STDOUT:   %pattern_type.loc157_27 => constants.%pattern_type.8b4
// CHECK:STDOUT:   %x.param_patt.loc157_27.2 => constants.%x.param_patt.27c
// CHECK:STDOUT:   %x.patt.loc157_27.2 => constants.%x.patt.740
// CHECK:STDOUT:   %pattern_type.loc157_33 => constants.%pattern_type.4aa
// CHECK:STDOUT:   %y.param_patt.loc157_33.2 => constants.%y.param_patt.a08
// CHECK:STDOUT:   %y.patt.loc157_33.2 => constants.%y.patt.069
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc157_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc157_27 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc157_33 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc159 => constants.%complete_type.4732f8.2
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.8b9
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.loc159 => constants.%custom_witness.099
// CHECK:STDOUT:   %DivWith.facet => constants.%DivWith.facet.b29
// CHECK:STDOUT:   %DivWith.WithSelf.Op.type => constants.%DivWith.WithSelf.Op.type.9fa
// CHECK:STDOUT:   %.loc159_5.6 => constants.%.bc2
// CHECK:STDOUT:   %impl.elem1.loc159_5.2 => constants.%Op.72717c.9
// CHECK:STDOUT:   %specific_impl_fn.loc159_5.2 => constants.%Op.72717c.9
// CHECK:STDOUT:   %SelfDestruct.specific_fn.2 => constants.%SelfDestruct.specific_fn.fb0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ModWith.loc168(constants.%Destroy.facet.c11, constants.%Destroy.facet.c11, constants.%facet_value.762) {
// CHECK:STDOUT:   %Result.patt.loc165_9.2 => constants.%Result.patt
// CHECK:STDOUT:   %Result.loc165_9.1 => constants.%Destroy.facet.c11
// CHECK:STDOUT:   %U.patt.loc166_4.2 => constants.%U.patt.cb8
// CHECK:STDOUT:   %U.loc166_4.1 => constants.%Destroy.facet.c11
// CHECK:STDOUT:   %U.as_type.loc167_35.1 => constants.%Int64
// CHECK:STDOUT:   %ModWith.type.loc167_35.1 => constants.%ModWith.type.c72
// CHECK:STDOUT:   %facet_type.loc167_19 => constants.%facet_type.c72
// CHECK:STDOUT:   %.Self.frozen.loc167_37.1 => constants.%.Self.frozen.c84
// CHECK:STDOUT:   %require_complete.loc167 => constants.%complete_type.4d1b1b.1
// CHECK:STDOUT:   %.Self.frozen.as_type => constants.%.Self.frozen.as_type.40c
// CHECK:STDOUT:   %ModWith.assoc_type => constants.%ModWith.assoc_type.ad2
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.c8a
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.loc167_43.1 => constants.%ModWith.lookup_impl_witness.06d
// CHECK:STDOUT:   %impl.elem0.loc167_43.1 => constants.%impl.elem0.968
// CHECK:STDOUT:   %Result.as_type.loc167_53.1 => constants.%Int64
// CHECK:STDOUT:   %.Self => constants.%.Self.dc4
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.loc167_43.2 => constants.%ModWith.lookup_impl_witness.305
// CHECK:STDOUT:   %impl.elem0.loc167_43.2 => constants.%impl.elem0.9c8
// CHECK:STDOUT:   %facet_type.loc167_37 => constants.%facet_type.573
// CHECK:STDOUT:   %pattern_type.loc167 => constants.%pattern_type.6af
// CHECK:STDOUT:   %T.patt.loc167_4.2 => constants.%T.patt.6c9
// CHECK:STDOUT:   %T.loc167_4.1 => constants.%facet_value.762
// CHECK:STDOUT:   %pattern_type.loc168_16 => constants.%pattern_type.4aa
// CHECK:STDOUT:   %result.param_patt.loc168_16.2 => constants.%result.param_patt.1e1
// CHECK:STDOUT:   %result.patt.loc168_16.2 => constants.%result.patt.93b
// CHECK:STDOUT:   %T.as_type.loc168_29.1 => constants.%Int32
// CHECK:STDOUT:   %pattern_type.loc168_27 => constants.%pattern_type.8b4
// CHECK:STDOUT:   %x.param_patt.loc168_27.2 => constants.%x.param_patt.27c
// CHECK:STDOUT:   %x.patt.loc168_27.2 => constants.%x.patt.740
// CHECK:STDOUT:   %pattern_type.loc168_33 => constants.%pattern_type.4aa
// CHECK:STDOUT:   %y.param_patt.loc168_33.2 => constants.%y.param_patt.a08
// CHECK:STDOUT:   %y.patt.loc168_33.2 => constants.%y.patt.069
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc168_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc168_27 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc168_33 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc170 => constants.%complete_type.4d1b1b.2
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.eed
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.loc170 => constants.%custom_witness.129
// CHECK:STDOUT:   %ModWith.facet => constants.%ModWith.facet.b10
// CHECK:STDOUT:   %ModWith.WithSelf.Op.type => constants.%ModWith.WithSelf.Op.type.1ecc
// CHECK:STDOUT:   %.loc170_5.6 => constants.%.bf4
// CHECK:STDOUT:   %impl.elem1.loc170_5.2 => constants.%Op.72717c.10
// CHECK:STDOUT:   %specific_impl_fn.loc170_5.2 => constants.%Op.72717c.10
// CHECK:STDOUT:   %SelfDestruct.specific_fn.2 => constants.%SelfDestruct.specific_fn.fb0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AddWith.loc124(constants.%Destroy.facet.c11, constants.%Destroy.facet.806, constants.%facet_value.acf) {
// CHECK:STDOUT:   %Result.patt.loc121_9.2 => constants.%Result.patt
// CHECK:STDOUT:   %Result.loc121_9.1 => constants.%Destroy.facet.c11
// CHECK:STDOUT:   %U.patt.loc122_4.2 => constants.%U.patt.cb8
// CHECK:STDOUT:   %U.loc122_4.1 => constants.%Destroy.facet.806
// CHECK:STDOUT:   %U.as_type.loc123_35.1 => constants.%Int32
// CHECK:STDOUT:   %AddWith.type.loc123_35.1 => constants.%AddWith.type.db7
// CHECK:STDOUT:   %facet_type.loc123_19 => constants.%facet_type.db7
// CHECK:STDOUT:   %.Self.frozen.loc123_37.1 => constants.%.Self.frozen.bc4
// CHECK:STDOUT:   %require_complete.loc123 => constants.%complete_type.758ea8.1
// CHECK:STDOUT:   %.Self.frozen.as_type => constants.%.Self.frozen.as_type.447
// CHECK:STDOUT:   %AddWith.assoc_type => constants.%AddWith.assoc_type.be8
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.31c
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.loc123_43.1 => constants.%AddWith.lookup_impl_witness.a9e
// CHECK:STDOUT:   %impl.elem0.loc123_43.1 => constants.%impl.elem0.517
// CHECK:STDOUT:   %Result.as_type.loc123_53.1 => constants.%Int64
// CHECK:STDOUT:   %.Self => constants.%.Self.6a9
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.loc123_43.2 => constants.%AddWith.lookup_impl_witness.f6c
// CHECK:STDOUT:   %impl.elem0.loc123_43.2 => constants.%impl.elem0.a74
// CHECK:STDOUT:   %facet_type.loc123_37 => constants.%facet_type.279
// CHECK:STDOUT:   %pattern_type.loc123 => constants.%pattern_type.95b
// CHECK:STDOUT:   %T.patt.loc123_4.2 => constants.%T.patt.cc5
// CHECK:STDOUT:   %T.loc123_4.1 => constants.%facet_value.acf
// CHECK:STDOUT:   %pattern_type.loc124_16 => constants.%pattern_type.4aa
// CHECK:STDOUT:   %result.param_patt.loc124_16.2 => constants.%result.param_patt.1e1
// CHECK:STDOUT:   %result.patt.loc124_16.2 => constants.%result.patt.93b
// CHECK:STDOUT:   %T.as_type.loc124_29.1 => constants.%Int64
// CHECK:STDOUT:   %pattern_type.loc124_27 => constants.%pattern_type.4aa
// CHECK:STDOUT:   %x.param_patt.loc124_27.2 => constants.%x.param_patt.e68
// CHECK:STDOUT:   %x.patt.loc124_27.2 => constants.%x.patt.7aa
// CHECK:STDOUT:   %pattern_type.loc124_33 => constants.%pattern_type.8b4
// CHECK:STDOUT:   %y.param_patt.loc124_33.2 => constants.%y.param_patt.e0c
// CHECK:STDOUT:   %y.patt.loc124_33.2 => constants.%y.patt.9e1
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc124_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc124_27 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc124_33 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc126 => constants.%complete_type.758ea8.2
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.86c
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.loc126 => constants.%custom_witness.16b
// CHECK:STDOUT:   %AddWith.facet => constants.%AddWith.facet.f65
// CHECK:STDOUT:   %AddWith.WithSelf.Op.type => constants.%AddWith.WithSelf.Op.type.8b6
// CHECK:STDOUT:   %.loc126_5.6 => constants.%.cf7
// CHECK:STDOUT:   %impl.elem1.loc126_5.2 => constants.%Op.72717c.11
// CHECK:STDOUT:   %specific_impl_fn.loc126_5.2 => constants.%Op.72717c.11
// CHECK:STDOUT:   %SelfDestruct.specific_fn.2 => constants.%SelfDestruct.specific_fn.fb0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @SubWith.loc135(constants.%Destroy.facet.c11, constants.%Destroy.facet.806, constants.%facet_value.957) {
// CHECK:STDOUT:   %Result.patt.loc132_9.2 => constants.%Result.patt
// CHECK:STDOUT:   %Result.loc132_9.1 => constants.%Destroy.facet.c11
// CHECK:STDOUT:   %U.patt.loc133_4.2 => constants.%U.patt.cb8
// CHECK:STDOUT:   %U.loc133_4.1 => constants.%Destroy.facet.806
// CHECK:STDOUT:   %U.as_type.loc134_35.1 => constants.%Int32
// CHECK:STDOUT:   %SubWith.type.loc134_35.1 => constants.%SubWith.type.120
// CHECK:STDOUT:   %facet_type.loc134_19 => constants.%facet_type.120
// CHECK:STDOUT:   %.Self.frozen.loc134_37.1 => constants.%.Self.frozen.faf
// CHECK:STDOUT:   %require_complete.loc134 => constants.%complete_type.46ea3c.1
// CHECK:STDOUT:   %.Self.frozen.as_type => constants.%.Self.frozen.as_type.ea6
// CHECK:STDOUT:   %SubWith.assoc_type => constants.%SubWith.assoc_type.dae
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.fac
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.loc134_43.1 => constants.%SubWith.lookup_impl_witness.c4f
// CHECK:STDOUT:   %impl.elem0.loc134_43.1 => constants.%impl.elem0.b5a
// CHECK:STDOUT:   %Result.as_type.loc134_53.1 => constants.%Int64
// CHECK:STDOUT:   %.Self => constants.%.Self.861
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.loc134_43.2 => constants.%SubWith.lookup_impl_witness.4cf
// CHECK:STDOUT:   %impl.elem0.loc134_43.2 => constants.%impl.elem0.94c
// CHECK:STDOUT:   %facet_type.loc134_37 => constants.%facet_type.1c5
// CHECK:STDOUT:   %pattern_type.loc134 => constants.%pattern_type.33d
// CHECK:STDOUT:   %T.patt.loc134_4.2 => constants.%T.patt.203
// CHECK:STDOUT:   %T.loc134_4.1 => constants.%facet_value.957
// CHECK:STDOUT:   %pattern_type.loc135_16 => constants.%pattern_type.4aa
// CHECK:STDOUT:   %result.param_patt.loc135_16.2 => constants.%result.param_patt.1e1
// CHECK:STDOUT:   %result.patt.loc135_16.2 => constants.%result.patt.93b
// CHECK:STDOUT:   %T.as_type.loc135_29.1 => constants.%Int64
// CHECK:STDOUT:   %pattern_type.loc135_27 => constants.%pattern_type.4aa
// CHECK:STDOUT:   %x.param_patt.loc135_27.2 => constants.%x.param_patt.e68
// CHECK:STDOUT:   %x.patt.loc135_27.2 => constants.%x.patt.7aa
// CHECK:STDOUT:   %pattern_type.loc135_33 => constants.%pattern_type.8b4
// CHECK:STDOUT:   %y.param_patt.loc135_33.2 => constants.%y.param_patt.e0c
// CHECK:STDOUT:   %y.patt.loc135_33.2 => constants.%y.patt.9e1
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc135_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc135_27 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc135_33 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc137 => constants.%complete_type.46ea3c.2
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.147
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.loc137 => constants.%custom_witness.21a
// CHECK:STDOUT:   %SubWith.facet => constants.%SubWith.facet.245
// CHECK:STDOUT:   %SubWith.WithSelf.Op.type => constants.%SubWith.WithSelf.Op.type.2cf
// CHECK:STDOUT:   %.loc137_5.6 => constants.%.4e8
// CHECK:STDOUT:   %impl.elem1.loc137_5.2 => constants.%Op.72717c.12
// CHECK:STDOUT:   %specific_impl_fn.loc137_5.2 => constants.%Op.72717c.12
// CHECK:STDOUT:   %SelfDestruct.specific_fn.2 => constants.%SelfDestruct.specific_fn.fb0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @MulWith.loc146(constants.%Destroy.facet.c11, constants.%Destroy.facet.806, constants.%facet_value.43e) {
// CHECK:STDOUT:   %Result.patt.loc143_9.2 => constants.%Result.patt
// CHECK:STDOUT:   %Result.loc143_9.1 => constants.%Destroy.facet.c11
// CHECK:STDOUT:   %U.patt.loc144_4.2 => constants.%U.patt.cb8
// CHECK:STDOUT:   %U.loc144_4.1 => constants.%Destroy.facet.806
// CHECK:STDOUT:   %U.as_type.loc145_35.1 => constants.%Int32
// CHECK:STDOUT:   %MulWith.type.loc145_35.1 => constants.%MulWith.type.d67
// CHECK:STDOUT:   %facet_type.loc145_19 => constants.%facet_type.d67b
// CHECK:STDOUT:   %.Self.frozen.loc145_37.1 => constants.%.Self.frozen.c7a
// CHECK:STDOUT:   %require_complete.loc145 => constants.%complete_type.920626.1
// CHECK:STDOUT:   %.Self.frozen.as_type => constants.%.Self.frozen.as_type.c47
// CHECK:STDOUT:   %MulWith.assoc_type => constants.%MulWith.assoc_type.ab1
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.d0a
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.loc145_43.1 => constants.%MulWith.lookup_impl_witness.ab5
// CHECK:STDOUT:   %impl.elem0.loc145_43.1 => constants.%impl.elem0.6ef
// CHECK:STDOUT:   %Result.as_type.loc145_53.1 => constants.%Int64
// CHECK:STDOUT:   %.Self => constants.%.Self.cb9
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.loc145_43.2 => constants.%MulWith.lookup_impl_witness.4ca
// CHECK:STDOUT:   %impl.elem0.loc145_43.2 => constants.%impl.elem0.311
// CHECK:STDOUT:   %facet_type.loc145_37 => constants.%facet_type.ec7
// CHECK:STDOUT:   %pattern_type.loc145 => constants.%pattern_type.537
// CHECK:STDOUT:   %T.patt.loc145_4.2 => constants.%T.patt.f45
// CHECK:STDOUT:   %T.loc145_4.1 => constants.%facet_value.43e
// CHECK:STDOUT:   %pattern_type.loc146_16 => constants.%pattern_type.4aa
// CHECK:STDOUT:   %result.param_patt.loc146_16.2 => constants.%result.param_patt.1e1
// CHECK:STDOUT:   %result.patt.loc146_16.2 => constants.%result.patt.93b
// CHECK:STDOUT:   %T.as_type.loc146_29.1 => constants.%Int64
// CHECK:STDOUT:   %pattern_type.loc146_27 => constants.%pattern_type.4aa
// CHECK:STDOUT:   %x.param_patt.loc146_27.2 => constants.%x.param_patt.e68
// CHECK:STDOUT:   %x.patt.loc146_27.2 => constants.%x.patt.7aa
// CHECK:STDOUT:   %pattern_type.loc146_33 => constants.%pattern_type.8b4
// CHECK:STDOUT:   %y.param_patt.loc146_33.2 => constants.%y.param_patt.e0c
// CHECK:STDOUT:   %y.patt.loc146_33.2 => constants.%y.patt.9e1
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc146_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc146_27 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc146_33 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc148 => constants.%complete_type.920626.2
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.349
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.loc148 => constants.%custom_witness.798
// CHECK:STDOUT:   %MulWith.facet => constants.%MulWith.facet.394
// CHECK:STDOUT:   %MulWith.WithSelf.Op.type => constants.%MulWith.WithSelf.Op.type.0ab
// CHECK:STDOUT:   %.loc148_5.6 => constants.%.22d
// CHECK:STDOUT:   %impl.elem1.loc148_5.2 => constants.%Op.72717c.13
// CHECK:STDOUT:   %specific_impl_fn.loc148_5.2 => constants.%Op.72717c.13
// CHECK:STDOUT:   %SelfDestruct.specific_fn.2 => constants.%SelfDestruct.specific_fn.fb0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @DivWith.loc157(constants.%Destroy.facet.c11, constants.%Destroy.facet.806, constants.%facet_value.7ce) {
// CHECK:STDOUT:   %Result.patt.loc154_9.2 => constants.%Result.patt
// CHECK:STDOUT:   %Result.loc154_9.1 => constants.%Destroy.facet.c11
// CHECK:STDOUT:   %U.patt.loc155_4.2 => constants.%U.patt.cb8
// CHECK:STDOUT:   %U.loc155_4.1 => constants.%Destroy.facet.806
// CHECK:STDOUT:   %U.as_type.loc156_35.1 => constants.%Int32
// CHECK:STDOUT:   %DivWith.type.loc156_35.1 => constants.%DivWith.type.5a1
// CHECK:STDOUT:   %facet_type.loc156_19 => constants.%facet_type.5a1
// CHECK:STDOUT:   %.Self.frozen.loc156_37.1 => constants.%.Self.frozen.057
// CHECK:STDOUT:   %require_complete.loc156 => constants.%complete_type.348fba.1
// CHECK:STDOUT:   %.Self.frozen.as_type => constants.%.Self.frozen.as_type.cc3
// CHECK:STDOUT:   %DivWith.assoc_type => constants.%DivWith.assoc_type.22f
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.a9d
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.loc156_43.1 => constants.%DivWith.lookup_impl_witness.ca2
// CHECK:STDOUT:   %impl.elem0.loc156_43.1 => constants.%impl.elem0.14b
// CHECK:STDOUT:   %Result.as_type.loc156_53.1 => constants.%Int64
// CHECK:STDOUT:   %.Self => constants.%.Self.208
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.loc156_43.2 => constants.%DivWith.lookup_impl_witness.0ac
// CHECK:STDOUT:   %impl.elem0.loc156_43.2 => constants.%impl.elem0.437
// CHECK:STDOUT:   %facet_type.loc156_37 => constants.%facet_type.da8
// CHECK:STDOUT:   %pattern_type.loc156 => constants.%pattern_type.612
// CHECK:STDOUT:   %T.patt.loc156_4.2 => constants.%T.patt.bc3
// CHECK:STDOUT:   %T.loc156_4.1 => constants.%facet_value.7ce
// CHECK:STDOUT:   %pattern_type.loc157_16 => constants.%pattern_type.4aa
// CHECK:STDOUT:   %result.param_patt.loc157_16.2 => constants.%result.param_patt.1e1
// CHECK:STDOUT:   %result.patt.loc157_16.2 => constants.%result.patt.93b
// CHECK:STDOUT:   %T.as_type.loc157_29.1 => constants.%Int64
// CHECK:STDOUT:   %pattern_type.loc157_27 => constants.%pattern_type.4aa
// CHECK:STDOUT:   %x.param_patt.loc157_27.2 => constants.%x.param_patt.e68
// CHECK:STDOUT:   %x.patt.loc157_27.2 => constants.%x.patt.7aa
// CHECK:STDOUT:   %pattern_type.loc157_33 => constants.%pattern_type.8b4
// CHECK:STDOUT:   %y.param_patt.loc157_33.2 => constants.%y.param_patt.e0c
// CHECK:STDOUT:   %y.patt.loc157_33.2 => constants.%y.patt.9e1
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc157_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc157_27 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc157_33 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc159 => constants.%complete_type.348fba.2
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.004
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.loc159 => constants.%custom_witness.37f
// CHECK:STDOUT:   %DivWith.facet => constants.%DivWith.facet.85e
// CHECK:STDOUT:   %DivWith.WithSelf.Op.type => constants.%DivWith.WithSelf.Op.type.7b8
// CHECK:STDOUT:   %.loc159_5.6 => constants.%.cde
// CHECK:STDOUT:   %impl.elem1.loc159_5.2 => constants.%Op.72717c.14
// CHECK:STDOUT:   %specific_impl_fn.loc159_5.2 => constants.%Op.72717c.14
// CHECK:STDOUT:   %SelfDestruct.specific_fn.2 => constants.%SelfDestruct.specific_fn.fb0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ModWith.loc168(constants.%Destroy.facet.c11, constants.%Destroy.facet.806, constants.%facet_value.5a6) {
// CHECK:STDOUT:   %Result.patt.loc165_9.2 => constants.%Result.patt
// CHECK:STDOUT:   %Result.loc165_9.1 => constants.%Destroy.facet.c11
// CHECK:STDOUT:   %U.patt.loc166_4.2 => constants.%U.patt.cb8
// CHECK:STDOUT:   %U.loc166_4.1 => constants.%Destroy.facet.806
// CHECK:STDOUT:   %U.as_type.loc167_35.1 => constants.%Int32
// CHECK:STDOUT:   %ModWith.type.loc167_35.1 => constants.%ModWith.type.794
// CHECK:STDOUT:   %facet_type.loc167_19 => constants.%facet_type.794
// CHECK:STDOUT:   %.Self.frozen.loc167_37.1 => constants.%.Self.frozen.813
// CHECK:STDOUT:   %require_complete.loc167 => constants.%complete_type.cc95ce.1
// CHECK:STDOUT:   %.Self.frozen.as_type => constants.%.Self.frozen.as_type.f4a
// CHECK:STDOUT:   %ModWith.assoc_type => constants.%ModWith.assoc_type.021
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.310
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.loc167_43.1 => constants.%ModWith.lookup_impl_witness.efb
// CHECK:STDOUT:   %impl.elem0.loc167_43.1 => constants.%impl.elem0.e83
// CHECK:STDOUT:   %Result.as_type.loc167_53.1 => constants.%Int64
// CHECK:STDOUT:   %.Self => constants.%.Self.618
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.loc167_43.2 => constants.%ModWith.lookup_impl_witness.01c
// CHECK:STDOUT:   %impl.elem0.loc167_43.2 => constants.%impl.elem0.3bb
// CHECK:STDOUT:   %facet_type.loc167_37 => constants.%facet_type.005
// CHECK:STDOUT:   %pattern_type.loc167 => constants.%pattern_type.b4f
// CHECK:STDOUT:   %T.patt.loc167_4.2 => constants.%T.patt.0b5
// CHECK:STDOUT:   %T.loc167_4.1 => constants.%facet_value.5a6
// CHECK:STDOUT:   %pattern_type.loc168_16 => constants.%pattern_type.4aa
// CHECK:STDOUT:   %result.param_patt.loc168_16.2 => constants.%result.param_patt.1e1
// CHECK:STDOUT:   %result.patt.loc168_16.2 => constants.%result.patt.93b
// CHECK:STDOUT:   %T.as_type.loc168_29.1 => constants.%Int64
// CHECK:STDOUT:   %pattern_type.loc168_27 => constants.%pattern_type.4aa
// CHECK:STDOUT:   %x.param_patt.loc168_27.2 => constants.%x.param_patt.e68
// CHECK:STDOUT:   %x.patt.loc168_27.2 => constants.%x.patt.7aa
// CHECK:STDOUT:   %pattern_type.loc168_33 => constants.%pattern_type.8b4
// CHECK:STDOUT:   %y.param_patt.loc168_33.2 => constants.%y.param_patt.e0c
// CHECK:STDOUT:   %y.patt.loc168_33.2 => constants.%y.patt.9e1
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc168_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc168_27 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc168_33 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc170 => constants.%complete_type.cc95ce.2
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.2eb
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.loc170 => constants.%custom_witness.981
// CHECK:STDOUT:   %ModWith.facet => constants.%ModWith.facet.f5a2
// CHECK:STDOUT:   %ModWith.WithSelf.Op.type => constants.%ModWith.WithSelf.Op.type.d27
// CHECK:STDOUT:   %.loc170_5.6 => constants.%.3b0
// CHECK:STDOUT:   %impl.elem1.loc170_5.2 => constants.%Op.72717c.15
// CHECK:STDOUT:   %specific_impl_fn.loc170_5.2 => constants.%Op.72717c.15
// CHECK:STDOUT:   %SelfDestruct.specific_fn.2 => constants.%SelfDestruct.specific_fn.fb0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AddAssignWith.loc225(constants.%U.67d, constants.%T.73b) {
// CHECK:STDOUT:   %U.patt.loc225_19.2 => constants.%U.patt.3a0
// CHECK:STDOUT:   %U.loc225_19.1 => constants.%U.67d
// CHECK:STDOUT:   %AddAssignWith.type.loc225_50.1 => constants.%AddAssignWith.type.3a470e.2
// CHECK:STDOUT:   %pattern_type.loc225_28 => constants.%pattern_type.00e
// CHECK:STDOUT:   %T.patt.loc225_28.2 => constants.%T.patt.86c
// CHECK:STDOUT:   %T.loc225_28.1 => constants.%T.73b
// CHECK:STDOUT:   %T.as_type.loc225_60.1 => constants.%T.as_type.415
// CHECK:STDOUT:   %pattern_type.loc225_58 => constants.%pattern_type.a93410.2
// CHECK:STDOUT:   %x.patt.loc225_58.2 => constants.%x.patt.988
// CHECK:STDOUT:   %x.param_patt.loc225_53.2 => constants.%x.param_patt.2ee
// CHECK:STDOUT:   %pattern_type.loc225_64 => constants.%pattern_type.51d1c4.4
// CHECK:STDOUT:   %y.param_patt.loc225_64.2 => constants.%y.param_patt.be3
// CHECK:STDOUT:   %y.patt.loc225_64.2 => constants.%y.patt.678
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @SubAssignWith.loc231(constants.%U.67d, constants.%T.0dd) {
// CHECK:STDOUT:   %U.patt.loc231_19.2 => constants.%U.patt.3a0
// CHECK:STDOUT:   %U.loc231_19.1 => constants.%U.67d
// CHECK:STDOUT:   %SubAssignWith.type.loc231_50.1 => constants.%SubAssignWith.type.0c61b3.2
// CHECK:STDOUT:   %pattern_type.loc231_28 => constants.%pattern_type.c56
// CHECK:STDOUT:   %T.patt.loc231_28.2 => constants.%T.patt.253
// CHECK:STDOUT:   %T.loc231_28.1 => constants.%T.0dd
// CHECK:STDOUT:   %T.as_type.loc231_60.1 => constants.%T.as_type.540
// CHECK:STDOUT:   %pattern_type.loc231_58 => constants.%pattern_type.e85cd1.2
// CHECK:STDOUT:   %x.patt.loc231_58.2 => constants.%x.patt.fed
// CHECK:STDOUT:   %x.param_patt.loc231_53.2 => constants.%x.param_patt.f54
// CHECK:STDOUT:   %pattern_type.loc231_64 => constants.%pattern_type.51d1c4.4
// CHECK:STDOUT:   %y.param_patt.loc231_64.2 => constants.%y.param_patt.be3
// CHECK:STDOUT:   %y.patt.loc231_64.2 => constants.%y.patt.678
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @MulAssignWith.loc237(constants.%U.67d, constants.%T.9e4) {
// CHECK:STDOUT:   %U.patt.loc237_19.2 => constants.%U.patt.3a0
// CHECK:STDOUT:   %U.loc237_19.1 => constants.%U.67d
// CHECK:STDOUT:   %MulAssignWith.type.loc237_50.1 => constants.%MulAssignWith.type.2339fd.2
// CHECK:STDOUT:   %pattern_type.loc237_28 => constants.%pattern_type.4a0
// CHECK:STDOUT:   %T.patt.loc237_28.2 => constants.%T.patt.34b
// CHECK:STDOUT:   %T.loc237_28.1 => constants.%T.9e4
// CHECK:STDOUT:   %T.as_type.loc237_60.1 => constants.%T.as_type.3e8
// CHECK:STDOUT:   %pattern_type.loc237_58 => constants.%pattern_type.9fac4e.2
// CHECK:STDOUT:   %x.patt.loc237_58.2 => constants.%x.patt.5dd
// CHECK:STDOUT:   %x.param_patt.loc237_53.2 => constants.%x.param_patt.f0f
// CHECK:STDOUT:   %pattern_type.loc237_64 => constants.%pattern_type.51d1c4.4
// CHECK:STDOUT:   %y.param_patt.loc237_64.2 => constants.%y.param_patt.be3
// CHECK:STDOUT:   %y.patt.loc237_64.2 => constants.%y.patt.678
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @DivAssignWith.loc243(constants.%U.67d, constants.%T.5bb) {
// CHECK:STDOUT:   %U.patt.loc243_19.2 => constants.%U.patt.3a0
// CHECK:STDOUT:   %U.loc243_19.1 => constants.%U.67d
// CHECK:STDOUT:   %DivAssignWith.type.loc243_50.1 => constants.%DivAssignWith.type.4b7503.2
// CHECK:STDOUT:   %pattern_type.loc243_28 => constants.%pattern_type.91f
// CHECK:STDOUT:   %T.patt.loc243_28.2 => constants.%T.patt.2c1
// CHECK:STDOUT:   %T.loc243_28.1 => constants.%T.5bb
// CHECK:STDOUT:   %T.as_type.loc243_60.1 => constants.%T.as_type.cf7
// CHECK:STDOUT:   %pattern_type.loc243_58 => constants.%pattern_type.ce26b8.2
// CHECK:STDOUT:   %x.patt.loc243_58.2 => constants.%x.patt.df3
// CHECK:STDOUT:   %x.param_patt.loc243_53.2 => constants.%x.param_patt.d43
// CHECK:STDOUT:   %pattern_type.loc243_64 => constants.%pattern_type.51d1c4.4
// CHECK:STDOUT:   %y.param_patt.loc243_64.2 => constants.%y.param_patt.be3
// CHECK:STDOUT:   %y.patt.loc243_64.2 => constants.%y.patt.678
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ModAssignWith.loc249(constants.%U.67d, constants.%T.cc0) {
// CHECK:STDOUT:   %U.patt.loc249_19.2 => constants.%U.patt.3a0
// CHECK:STDOUT:   %U.loc249_19.1 => constants.%U.67d
// CHECK:STDOUT:   %ModAssignWith.type.loc249_50.1 => constants.%ModAssignWith.type.3a0f29.2
// CHECK:STDOUT:   %pattern_type.loc249_28 => constants.%pattern_type.685
// CHECK:STDOUT:   %T.patt.loc249_28.2 => constants.%T.patt.1bf
// CHECK:STDOUT:   %T.loc249_28.1 => constants.%T.cc0
// CHECK:STDOUT:   %T.as_type.loc249_60.1 => constants.%T.as_type.739
// CHECK:STDOUT:   %pattern_type.loc249_58 => constants.%pattern_type.443202.2
// CHECK:STDOUT:   %x.patt.loc249_58.2 => constants.%x.patt.14d
// CHECK:STDOUT:   %x.param_patt.loc249_53.2 => constants.%x.param_patt.605
// CHECK:STDOUT:   %pattern_type.loc249_64 => constants.%pattern_type.51d1c4.4
// CHECK:STDOUT:   %y.param_patt.loc249_64.2 => constants.%y.param_patt.be3
// CHECK:STDOUT:   %y.patt.loc249_64.2 => constants.%y.patt.678
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AddAssignWith.loc225(constants.%i16, constants.%AddAssignWith.facet.213) {
// CHECK:STDOUT:   %U.patt.loc225_19.2 => constants.%U.patt.3a0
// CHECK:STDOUT:   %U.loc225_19.1 => constants.%i16
// CHECK:STDOUT:   %AddAssignWith.type.loc225_50.1 => constants.%AddAssignWith.type.803
// CHECK:STDOUT:   %pattern_type.loc225_28 => constants.%pattern_type.7df
// CHECK:STDOUT:   %T.patt.loc225_28.2 => constants.%T.patt.ec8
// CHECK:STDOUT:   %T.loc225_28.1 => constants.%AddAssignWith.facet.213
// CHECK:STDOUT:   %T.as_type.loc225_60.1 => constants.%Int16
// CHECK:STDOUT:   %pattern_type.loc225_58 => constants.%pattern_type.785
// CHECK:STDOUT:   %x.patt.loc225_58.2 => constants.%x.patt.49e
// CHECK:STDOUT:   %x.param_patt.loc225_53.2 => constants.%x.param_patt.69a
// CHECK:STDOUT:   %pattern_type.loc225_64 => constants.%pattern_type.9bc
// CHECK:STDOUT:   %y.param_patt.loc225_64.2 => constants.%y.param_patt.322
// CHECK:STDOUT:   %y.patt.loc225_64.2 => constants.%y.patt.a3a
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc225_53 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc225_64 => constants.%complete_type.2da
// CHECK:STDOUT:   %require_complete.loc227 => constants.%complete_type.5fd
// CHECK:STDOUT:   %AddAssignWith.assoc_type => constants.%AddAssignWith.assoc_type.82a
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.7ab
// CHECK:STDOUT:   %AddAssignWith.WithSelf.Op.type => constants.%AddAssignWith.WithSelf.Op.type.893
// CHECK:STDOUT:   %.loc227_5.2 => constants.%.5a24
// CHECK:STDOUT:   %AddAssignWith.lookup_impl_witness => constants.%custom_witness.2fc
// CHECK:STDOUT:   %impl.elem0.loc227_5.2 => constants.%Int16.Op.4dca31.12
// CHECK:STDOUT:   %specific_impl_fn.loc227_5.2 => constants.%Int16.Op.4dca31.12
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @SubAssignWith.loc231(constants.%i16, constants.%SubAssignWith.facet.474) {
// CHECK:STDOUT:   %U.patt.loc231_19.2 => constants.%U.patt.3a0
// CHECK:STDOUT:   %U.loc231_19.1 => constants.%i16
// CHECK:STDOUT:   %SubAssignWith.type.loc231_50.1 => constants.%SubAssignWith.type.929
// CHECK:STDOUT:   %pattern_type.loc231_28 => constants.%pattern_type.23a
// CHECK:STDOUT:   %T.patt.loc231_28.2 => constants.%T.patt.467
// CHECK:STDOUT:   %T.loc231_28.1 => constants.%SubAssignWith.facet.474
// CHECK:STDOUT:   %T.as_type.loc231_60.1 => constants.%Int16
// CHECK:STDOUT:   %pattern_type.loc231_58 => constants.%pattern_type.785
// CHECK:STDOUT:   %x.patt.loc231_58.2 => constants.%x.patt.49e
// CHECK:STDOUT:   %x.param_patt.loc231_53.2 => constants.%x.param_patt.69a
// CHECK:STDOUT:   %pattern_type.loc231_64 => constants.%pattern_type.9bc
// CHECK:STDOUT:   %y.param_patt.loc231_64.2 => constants.%y.param_patt.322
// CHECK:STDOUT:   %y.patt.loc231_64.2 => constants.%y.patt.a3a
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc231_53 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc231_64 => constants.%complete_type.2da
// CHECK:STDOUT:   %require_complete.loc233 => constants.%complete_type.aff
// CHECK:STDOUT:   %SubAssignWith.assoc_type => constants.%SubAssignWith.assoc_type.ecb
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.ecb
// CHECK:STDOUT:   %SubAssignWith.WithSelf.Op.type => constants.%SubAssignWith.WithSelf.Op.type.0df
// CHECK:STDOUT:   %.loc233_5.2 => constants.%.cfa
// CHECK:STDOUT:   %SubAssignWith.lookup_impl_witness => constants.%custom_witness.21df
// CHECK:STDOUT:   %impl.elem0.loc233_5.2 => constants.%Int16.Op.4dca31.13
// CHECK:STDOUT:   %specific_impl_fn.loc233_5.2 => constants.%Int16.Op.4dca31.13
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @MulAssignWith.loc237(constants.%i16, constants.%MulAssignWith.facet.9d6) {
// CHECK:STDOUT:   %U.patt.loc237_19.2 => constants.%U.patt.3a0
// CHECK:STDOUT:   %U.loc237_19.1 => constants.%i16
// CHECK:STDOUT:   %MulAssignWith.type.loc237_50.1 => constants.%MulAssignWith.type.ba0
// CHECK:STDOUT:   %pattern_type.loc237_28 => constants.%pattern_type.d7c
// CHECK:STDOUT:   %T.patt.loc237_28.2 => constants.%T.patt.32f
// CHECK:STDOUT:   %T.loc237_28.1 => constants.%MulAssignWith.facet.9d6
// CHECK:STDOUT:   %T.as_type.loc237_60.1 => constants.%Int16
// CHECK:STDOUT:   %pattern_type.loc237_58 => constants.%pattern_type.785
// CHECK:STDOUT:   %x.patt.loc237_58.2 => constants.%x.patt.49e
// CHECK:STDOUT:   %x.param_patt.loc237_53.2 => constants.%x.param_patt.69a
// CHECK:STDOUT:   %pattern_type.loc237_64 => constants.%pattern_type.9bc
// CHECK:STDOUT:   %y.param_patt.loc237_64.2 => constants.%y.param_patt.322
// CHECK:STDOUT:   %y.patt.loc237_64.2 => constants.%y.patt.a3a
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc237_53 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc237_64 => constants.%complete_type.2da
// CHECK:STDOUT:   %require_complete.loc239 => constants.%complete_type.deb
// CHECK:STDOUT:   %MulAssignWith.assoc_type => constants.%MulAssignWith.assoc_type.89a
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.144
// CHECK:STDOUT:   %MulAssignWith.WithSelf.Op.type => constants.%MulAssignWith.WithSelf.Op.type.726
// CHECK:STDOUT:   %.loc239_5.2 => constants.%.9a5
// CHECK:STDOUT:   %MulAssignWith.lookup_impl_witness => constants.%custom_witness.81e
// CHECK:STDOUT:   %impl.elem0.loc239_5.2 => constants.%Int16.Op.4dca31.14
// CHECK:STDOUT:   %specific_impl_fn.loc239_5.2 => constants.%Int16.Op.4dca31.14
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @DivAssignWith.loc243(constants.%i16, constants.%DivAssignWith.facet.bed) {
// CHECK:STDOUT:   %U.patt.loc243_19.2 => constants.%U.patt.3a0
// CHECK:STDOUT:   %U.loc243_19.1 => constants.%i16
// CHECK:STDOUT:   %DivAssignWith.type.loc243_50.1 => constants.%DivAssignWith.type.6b5
// CHECK:STDOUT:   %pattern_type.loc243_28 => constants.%pattern_type.7b9
// CHECK:STDOUT:   %T.patt.loc243_28.2 => constants.%T.patt.db5
// CHECK:STDOUT:   %T.loc243_28.1 => constants.%DivAssignWith.facet.bed
// CHECK:STDOUT:   %T.as_type.loc243_60.1 => constants.%Int16
// CHECK:STDOUT:   %pattern_type.loc243_58 => constants.%pattern_type.785
// CHECK:STDOUT:   %x.patt.loc243_58.2 => constants.%x.patt.49e
// CHECK:STDOUT:   %x.param_patt.loc243_53.2 => constants.%x.param_patt.69a
// CHECK:STDOUT:   %pattern_type.loc243_64 => constants.%pattern_type.9bc
// CHECK:STDOUT:   %y.param_patt.loc243_64.2 => constants.%y.param_patt.322
// CHECK:STDOUT:   %y.patt.loc243_64.2 => constants.%y.patt.a3a
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc243_53 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc243_64 => constants.%complete_type.2da
// CHECK:STDOUT:   %require_complete.loc245 => constants.%complete_type.383
// CHECK:STDOUT:   %DivAssignWith.assoc_type => constants.%DivAssignWith.assoc_type.fae
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.1d8
// CHECK:STDOUT:   %DivAssignWith.WithSelf.Op.type => constants.%DivAssignWith.WithSelf.Op.type.204
// CHECK:STDOUT:   %.loc245_5.2 => constants.%.f1e
// CHECK:STDOUT:   %DivAssignWith.lookup_impl_witness => constants.%custom_witness.787
// CHECK:STDOUT:   %impl.elem0.loc245_5.2 => constants.%Int16.Op.4dca31.15
// CHECK:STDOUT:   %specific_impl_fn.loc245_5.2 => constants.%Int16.Op.4dca31.15
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ModAssignWith.loc249(constants.%i16, constants.%ModAssignWith.facet.6d2) {
// CHECK:STDOUT:   %U.patt.loc249_19.2 => constants.%U.patt.3a0
// CHECK:STDOUT:   %U.loc249_19.1 => constants.%i16
// CHECK:STDOUT:   %ModAssignWith.type.loc249_50.1 => constants.%ModAssignWith.type.ddb
// CHECK:STDOUT:   %pattern_type.loc249_28 => constants.%pattern_type.83d
// CHECK:STDOUT:   %T.patt.loc249_28.2 => constants.%T.patt.0ba
// CHECK:STDOUT:   %T.loc249_28.1 => constants.%ModAssignWith.facet.6d2
// CHECK:STDOUT:   %T.as_type.loc249_60.1 => constants.%Int16
// CHECK:STDOUT:   %pattern_type.loc249_58 => constants.%pattern_type.785
// CHECK:STDOUT:   %x.patt.loc249_58.2 => constants.%x.patt.49e
// CHECK:STDOUT:   %x.param_patt.loc249_53.2 => constants.%x.param_patt.69a
// CHECK:STDOUT:   %pattern_type.loc249_64 => constants.%pattern_type.9bc
// CHECK:STDOUT:   %y.param_patt.loc249_64.2 => constants.%y.param_patt.322
// CHECK:STDOUT:   %y.patt.loc249_64.2 => constants.%y.patt.a3a
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc249_53 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc249_64 => constants.%complete_type.2da
// CHECK:STDOUT:   %require_complete.loc251 => constants.%complete_type.3ce
// CHECK:STDOUT:   %ModAssignWith.assoc_type => constants.%ModAssignWith.assoc_type.725
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.701
// CHECK:STDOUT:   %ModAssignWith.WithSelf.Op.type => constants.%ModAssignWith.WithSelf.Op.type.4c4
// CHECK:STDOUT:   %.loc251_5.2 => constants.%.37f
// CHECK:STDOUT:   %ModAssignWith.lookup_impl_witness => constants.%custom_witness.c25
// CHECK:STDOUT:   %impl.elem0.loc251_5.2 => constants.%Int16.Op.4dca31.16
// CHECK:STDOUT:   %specific_impl_fn.loc251_5.2 => constants.%Int16.Op.4dca31.16
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AddAssignWith.loc225(constants.%Int32, constants.%AddAssignWith.facet.e80) {
// CHECK:STDOUT:   %U.patt.loc225_19.2 => constants.%U.patt.3a0
// CHECK:STDOUT:   %U.loc225_19.1 => constants.%Int32
// CHECK:STDOUT:   %AddAssignWith.type.loc225_50.1 => constants.%AddAssignWith.type.eef
// CHECK:STDOUT:   %pattern_type.loc225_28 => constants.%pattern_type.1ab
// CHECK:STDOUT:   %T.patt.loc225_28.2 => constants.%T.patt.034
// CHECK:STDOUT:   %T.loc225_28.1 => constants.%AddAssignWith.facet.e80
// CHECK:STDOUT:   %T.as_type.loc225_60.1 => constants.%Int32
// CHECK:STDOUT:   %pattern_type.loc225_58 => constants.%pattern_type.8b4
// CHECK:STDOUT:   %x.patt.loc225_58.2 => constants.%x.patt.d79
// CHECK:STDOUT:   %x.param_patt.loc225_53.2 => constants.%x.param_patt.0c7
// CHECK:STDOUT:   %pattern_type.loc225_64 => constants.%pattern_type.8b4
// CHECK:STDOUT:   %y.param_patt.loc225_64.2 => constants.%y.param_patt.e0c
// CHECK:STDOUT:   %y.patt.loc225_64.2 => constants.%y.patt.9e1
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc225_53 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc225_64 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc227 => constants.%complete_type.c41
// CHECK:STDOUT:   %AddAssignWith.assoc_type => constants.%AddAssignWith.assoc_type.c4c
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.ce4
// CHECK:STDOUT:   %AddAssignWith.WithSelf.Op.type => constants.%AddAssignWith.WithSelf.Op.type.b93
// CHECK:STDOUT:   %.loc227_5.2 => constants.%.05e
// CHECK:STDOUT:   %AddAssignWith.lookup_impl_witness => constants.%custom_witness.17487e.1
// CHECK:STDOUT:   %impl.elem0.loc227_5.2 => constants.%Int32.Op.4dca31.11
// CHECK:STDOUT:   %specific_impl_fn.loc227_5.2 => constants.%Int32.Op.4dca31.11
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @SubAssignWith.loc231(constants.%Int32, constants.%SubAssignWith.facet.143) {
// CHECK:STDOUT:   %U.patt.loc231_19.2 => constants.%U.patt.3a0
// CHECK:STDOUT:   %U.loc231_19.1 => constants.%Int32
// CHECK:STDOUT:   %SubAssignWith.type.loc231_50.1 => constants.%SubAssignWith.type.1c4
// CHECK:STDOUT:   %pattern_type.loc231_28 => constants.%pattern_type.303
// CHECK:STDOUT:   %T.patt.loc231_28.2 => constants.%T.patt.dc8
// CHECK:STDOUT:   %T.loc231_28.1 => constants.%SubAssignWith.facet.143
// CHECK:STDOUT:   %T.as_type.loc231_60.1 => constants.%Int32
// CHECK:STDOUT:   %pattern_type.loc231_58 => constants.%pattern_type.8b4
// CHECK:STDOUT:   %x.patt.loc231_58.2 => constants.%x.patt.d79
// CHECK:STDOUT:   %x.param_patt.loc231_53.2 => constants.%x.param_patt.0c7
// CHECK:STDOUT:   %pattern_type.loc231_64 => constants.%pattern_type.8b4
// CHECK:STDOUT:   %y.param_patt.loc231_64.2 => constants.%y.param_patt.e0c
// CHECK:STDOUT:   %y.patt.loc231_64.2 => constants.%y.patt.9e1
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc231_53 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc231_64 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc233 => constants.%complete_type.b00
// CHECK:STDOUT:   %SubAssignWith.assoc_type => constants.%SubAssignWith.assoc_type.2e9
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.b0e
// CHECK:STDOUT:   %SubAssignWith.WithSelf.Op.type => constants.%SubAssignWith.WithSelf.Op.type.0d2
// CHECK:STDOUT:   %.loc233_5.2 => constants.%.68c
// CHECK:STDOUT:   %SubAssignWith.lookup_impl_witness => constants.%custom_witness.ebd70f.1
// CHECK:STDOUT:   %impl.elem0.loc233_5.2 => constants.%Int32.Op.4dca31.13
// CHECK:STDOUT:   %specific_impl_fn.loc233_5.2 => constants.%Int32.Op.4dca31.13
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @MulAssignWith.loc237(constants.%Int32, constants.%MulAssignWith.facet.1e6) {
// CHECK:STDOUT:   %U.patt.loc237_19.2 => constants.%U.patt.3a0
// CHECK:STDOUT:   %U.loc237_19.1 => constants.%Int32
// CHECK:STDOUT:   %MulAssignWith.type.loc237_50.1 => constants.%MulAssignWith.type.ec7
// CHECK:STDOUT:   %pattern_type.loc237_28 => constants.%pattern_type.b21
// CHECK:STDOUT:   %T.patt.loc237_28.2 => constants.%T.patt.59f
// CHECK:STDOUT:   %T.loc237_28.1 => constants.%MulAssignWith.facet.1e6
// CHECK:STDOUT:   %T.as_type.loc237_60.1 => constants.%Int32
// CHECK:STDOUT:   %pattern_type.loc237_58 => constants.%pattern_type.8b4
// CHECK:STDOUT:   %x.patt.loc237_58.2 => constants.%x.patt.d79
// CHECK:STDOUT:   %x.param_patt.loc237_53.2 => constants.%x.param_patt.0c7
// CHECK:STDOUT:   %pattern_type.loc237_64 => constants.%pattern_type.8b4
// CHECK:STDOUT:   %y.param_patt.loc237_64.2 => constants.%y.param_patt.e0c
// CHECK:STDOUT:   %y.patt.loc237_64.2 => constants.%y.patt.9e1
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc237_53 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc237_64 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc239 => constants.%complete_type.14d
// CHECK:STDOUT:   %MulAssignWith.assoc_type => constants.%MulAssignWith.assoc_type.cbc
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.e00
// CHECK:STDOUT:   %MulAssignWith.WithSelf.Op.type => constants.%MulAssignWith.WithSelf.Op.type.e90
// CHECK:STDOUT:   %.loc239_5.2 => constants.%.a68
// CHECK:STDOUT:   %MulAssignWith.lookup_impl_witness => constants.%custom_witness.636b34.1
// CHECK:STDOUT:   %impl.elem0.loc239_5.2 => constants.%Int32.Op.4dca31.14
// CHECK:STDOUT:   %specific_impl_fn.loc239_5.2 => constants.%Int32.Op.4dca31.14
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @DivAssignWith.loc243(constants.%Int32, constants.%DivAssignWith.facet.1ca) {
// CHECK:STDOUT:   %U.patt.loc243_19.2 => constants.%U.patt.3a0
// CHECK:STDOUT:   %U.loc243_19.1 => constants.%Int32
// CHECK:STDOUT:   %DivAssignWith.type.loc243_50.1 => constants.%DivAssignWith.type.12e
// CHECK:STDOUT:   %pattern_type.loc243_28 => constants.%pattern_type.9bf
// CHECK:STDOUT:   %T.patt.loc243_28.2 => constants.%T.patt.b33
// CHECK:STDOUT:   %T.loc243_28.1 => constants.%DivAssignWith.facet.1ca
// CHECK:STDOUT:   %T.as_type.loc243_60.1 => constants.%Int32
// CHECK:STDOUT:   %pattern_type.loc243_58 => constants.%pattern_type.8b4
// CHECK:STDOUT:   %x.patt.loc243_58.2 => constants.%x.patt.d79
// CHECK:STDOUT:   %x.param_patt.loc243_53.2 => constants.%x.param_patt.0c7
// CHECK:STDOUT:   %pattern_type.loc243_64 => constants.%pattern_type.8b4
// CHECK:STDOUT:   %y.param_patt.loc243_64.2 => constants.%y.param_patt.e0c
// CHECK:STDOUT:   %y.patt.loc243_64.2 => constants.%y.patt.9e1
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc243_53 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc243_64 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc245 => constants.%complete_type.77b
// CHECK:STDOUT:   %DivAssignWith.assoc_type => constants.%DivAssignWith.assoc_type.776
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.14c
// CHECK:STDOUT:   %DivAssignWith.WithSelf.Op.type => constants.%DivAssignWith.WithSelf.Op.type.78a
// CHECK:STDOUT:   %.loc245_5.2 => constants.%.aa9
// CHECK:STDOUT:   %DivAssignWith.lookup_impl_witness => constants.%custom_witness.45756b.1
// CHECK:STDOUT:   %impl.elem0.loc245_5.2 => constants.%Int32.Op.4dca31.15
// CHECK:STDOUT:   %specific_impl_fn.loc245_5.2 => constants.%Int32.Op.4dca31.15
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ModAssignWith.loc249(constants.%Int32, constants.%ModAssignWith.facet.715) {
// CHECK:STDOUT:   %U.patt.loc249_19.2 => constants.%U.patt.3a0
// CHECK:STDOUT:   %U.loc249_19.1 => constants.%Int32
// CHECK:STDOUT:   %ModAssignWith.type.loc249_50.1 => constants.%ModAssignWith.type.c90
// CHECK:STDOUT:   %pattern_type.loc249_28 => constants.%pattern_type.065
// CHECK:STDOUT:   %T.patt.loc249_28.2 => constants.%T.patt.707
// CHECK:STDOUT:   %T.loc249_28.1 => constants.%ModAssignWith.facet.715
// CHECK:STDOUT:   %T.as_type.loc249_60.1 => constants.%Int32
// CHECK:STDOUT:   %pattern_type.loc249_58 => constants.%pattern_type.8b4
// CHECK:STDOUT:   %x.patt.loc249_58.2 => constants.%x.patt.d79
// CHECK:STDOUT:   %x.param_patt.loc249_53.2 => constants.%x.param_patt.0c7
// CHECK:STDOUT:   %pattern_type.loc249_64 => constants.%pattern_type.8b4
// CHECK:STDOUT:   %y.param_patt.loc249_64.2 => constants.%y.param_patt.e0c
// CHECK:STDOUT:   %y.patt.loc249_64.2 => constants.%y.patt.9e1
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc249_53 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc249_64 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc251 => constants.%complete_type.65d
// CHECK:STDOUT:   %ModAssignWith.assoc_type => constants.%ModAssignWith.assoc_type.7d0
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.f20
// CHECK:STDOUT:   %ModAssignWith.WithSelf.Op.type => constants.%ModAssignWith.WithSelf.Op.type.81c
// CHECK:STDOUT:   %.loc251_5.2 => constants.%.913
// CHECK:STDOUT:   %ModAssignWith.lookup_impl_witness => constants.%custom_witness.773f24.1
// CHECK:STDOUT:   %impl.elem0.loc251_5.2 => constants.%Int32.Op.4dca31.16
// CHECK:STDOUT:   %specific_impl_fn.loc251_5.2 => constants.%Int32.Op.4dca31.16
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AddAssignWith.loc225(constants.%Int32, constants.%AddAssignWith.facet.c6d) {
// CHECK:STDOUT:   %U.patt.loc225_19.2 => constants.%U.patt.3a0
// CHECK:STDOUT:   %U.loc225_19.1 => constants.%Int32
// CHECK:STDOUT:   %AddAssignWith.type.loc225_50.1 => constants.%AddAssignWith.type.eef
// CHECK:STDOUT:   %pattern_type.loc225_28 => constants.%pattern_type.1ab
// CHECK:STDOUT:   %T.patt.loc225_28.2 => constants.%T.patt.034
// CHECK:STDOUT:   %T.loc225_28.1 => constants.%AddAssignWith.facet.c6d
// CHECK:STDOUT:   %T.as_type.loc225_60.1 => constants.%Int64
// CHECK:STDOUT:   %pattern_type.loc225_58 => constants.%pattern_type.4aa
// CHECK:STDOUT:   %x.patt.loc225_58.2 => constants.%x.patt.ace
// CHECK:STDOUT:   %x.param_patt.loc225_53.2 => constants.%x.param_patt.b33
// CHECK:STDOUT:   %pattern_type.loc225_64 => constants.%pattern_type.8b4
// CHECK:STDOUT:   %y.param_patt.loc225_64.2 => constants.%y.param_patt.e0c
// CHECK:STDOUT:   %y.patt.loc225_64.2 => constants.%y.patt.9e1
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc225_53 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc225_64 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc227 => constants.%complete_type.c41
// CHECK:STDOUT:   %AddAssignWith.assoc_type => constants.%AddAssignWith.assoc_type.c4c
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.ce4
// CHECK:STDOUT:   %AddAssignWith.WithSelf.Op.type => constants.%AddAssignWith.WithSelf.Op.type.e31
// CHECK:STDOUT:   %.loc227_5.2 => constants.%.454
// CHECK:STDOUT:   %AddAssignWith.lookup_impl_witness => constants.%custom_witness.17487e.2
// CHECK:STDOUT:   %impl.elem0.loc227_5.2 => constants.%Int64.Op.4dca31.1
// CHECK:STDOUT:   %specific_impl_fn.loc227_5.2 => constants.%Int64.Op.4dca31.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @SubAssignWith.loc231(constants.%Int32, constants.%SubAssignWith.facet.74a) {
// CHECK:STDOUT:   %U.patt.loc231_19.2 => constants.%U.patt.3a0
// CHECK:STDOUT:   %U.loc231_19.1 => constants.%Int32
// CHECK:STDOUT:   %SubAssignWith.type.loc231_50.1 => constants.%SubAssignWith.type.1c4
// CHECK:STDOUT:   %pattern_type.loc231_28 => constants.%pattern_type.303
// CHECK:STDOUT:   %T.patt.loc231_28.2 => constants.%T.patt.dc8
// CHECK:STDOUT:   %T.loc231_28.1 => constants.%SubAssignWith.facet.74a
// CHECK:STDOUT:   %T.as_type.loc231_60.1 => constants.%Int64
// CHECK:STDOUT:   %pattern_type.loc231_58 => constants.%pattern_type.4aa
// CHECK:STDOUT:   %x.patt.loc231_58.2 => constants.%x.patt.ace
// CHECK:STDOUT:   %x.param_patt.loc231_53.2 => constants.%x.param_patt.b33
// CHECK:STDOUT:   %pattern_type.loc231_64 => constants.%pattern_type.8b4
// CHECK:STDOUT:   %y.param_patt.loc231_64.2 => constants.%y.param_patt.e0c
// CHECK:STDOUT:   %y.patt.loc231_64.2 => constants.%y.patt.9e1
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc231_53 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc231_64 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc233 => constants.%complete_type.b00
// CHECK:STDOUT:   %SubAssignWith.assoc_type => constants.%SubAssignWith.assoc_type.2e9
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.b0e
// CHECK:STDOUT:   %SubAssignWith.WithSelf.Op.type => constants.%SubAssignWith.WithSelf.Op.type.e54
// CHECK:STDOUT:   %.loc233_5.2 => constants.%.d58
// CHECK:STDOUT:   %SubAssignWith.lookup_impl_witness => constants.%custom_witness.ebd70f.2
// CHECK:STDOUT:   %impl.elem0.loc233_5.2 => constants.%Int64.Op.4dca31.3
// CHECK:STDOUT:   %specific_impl_fn.loc233_5.2 => constants.%Int64.Op.4dca31.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @MulAssignWith.loc237(constants.%Int32, constants.%MulAssignWith.facet.d5c) {
// CHECK:STDOUT:   %U.patt.loc237_19.2 => constants.%U.patt.3a0
// CHECK:STDOUT:   %U.loc237_19.1 => constants.%Int32
// CHECK:STDOUT:   %MulAssignWith.type.loc237_50.1 => constants.%MulAssignWith.type.ec7
// CHECK:STDOUT:   %pattern_type.loc237_28 => constants.%pattern_type.b21
// CHECK:STDOUT:   %T.patt.loc237_28.2 => constants.%T.patt.59f
// CHECK:STDOUT:   %T.loc237_28.1 => constants.%MulAssignWith.facet.d5c
// CHECK:STDOUT:   %T.as_type.loc237_60.1 => constants.%Int64
// CHECK:STDOUT:   %pattern_type.loc237_58 => constants.%pattern_type.4aa
// CHECK:STDOUT:   %x.patt.loc237_58.2 => constants.%x.patt.ace
// CHECK:STDOUT:   %x.param_patt.loc237_53.2 => constants.%x.param_patt.b33
// CHECK:STDOUT:   %pattern_type.loc237_64 => constants.%pattern_type.8b4
// CHECK:STDOUT:   %y.param_patt.loc237_64.2 => constants.%y.param_patt.e0c
// CHECK:STDOUT:   %y.patt.loc237_64.2 => constants.%y.patt.9e1
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc237_53 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc237_64 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc239 => constants.%complete_type.14d
// CHECK:STDOUT:   %MulAssignWith.assoc_type => constants.%MulAssignWith.assoc_type.cbc
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.e00
// CHECK:STDOUT:   %MulAssignWith.WithSelf.Op.type => constants.%MulAssignWith.WithSelf.Op.type.2e6
// CHECK:STDOUT:   %.loc239_5.2 => constants.%.536
// CHECK:STDOUT:   %MulAssignWith.lookup_impl_witness => constants.%custom_witness.636b34.2
// CHECK:STDOUT:   %impl.elem0.loc239_5.2 => constants.%Int64.Op.4dca31.4
// CHECK:STDOUT:   %specific_impl_fn.loc239_5.2 => constants.%Int64.Op.4dca31.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @DivAssignWith.loc243(constants.%Int32, constants.%DivAssignWith.facet.c0b) {
// CHECK:STDOUT:   %U.patt.loc243_19.2 => constants.%U.patt.3a0
// CHECK:STDOUT:   %U.loc243_19.1 => constants.%Int32
// CHECK:STDOUT:   %DivAssignWith.type.loc243_50.1 => constants.%DivAssignWith.type.12e
// CHECK:STDOUT:   %pattern_type.loc243_28 => constants.%pattern_type.9bf
// CHECK:STDOUT:   %T.patt.loc243_28.2 => constants.%T.patt.b33
// CHECK:STDOUT:   %T.loc243_28.1 => constants.%DivAssignWith.facet.c0b
// CHECK:STDOUT:   %T.as_type.loc243_60.1 => constants.%Int64
// CHECK:STDOUT:   %pattern_type.loc243_58 => constants.%pattern_type.4aa
// CHECK:STDOUT:   %x.patt.loc243_58.2 => constants.%x.patt.ace
// CHECK:STDOUT:   %x.param_patt.loc243_53.2 => constants.%x.param_patt.b33
// CHECK:STDOUT:   %pattern_type.loc243_64 => constants.%pattern_type.8b4
// CHECK:STDOUT:   %y.param_patt.loc243_64.2 => constants.%y.param_patt.e0c
// CHECK:STDOUT:   %y.patt.loc243_64.2 => constants.%y.patt.9e1
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc243_53 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc243_64 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc245 => constants.%complete_type.77b
// CHECK:STDOUT:   %DivAssignWith.assoc_type => constants.%DivAssignWith.assoc_type.776
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.14c
// CHECK:STDOUT:   %DivAssignWith.WithSelf.Op.type => constants.%DivAssignWith.WithSelf.Op.type.f46
// CHECK:STDOUT:   %.loc245_5.2 => constants.%.58e
// CHECK:STDOUT:   %DivAssignWith.lookup_impl_witness => constants.%custom_witness.45756b.2
// CHECK:STDOUT:   %impl.elem0.loc245_5.2 => constants.%Int64.Op.4dca31.5
// CHECK:STDOUT:   %specific_impl_fn.loc245_5.2 => constants.%Int64.Op.4dca31.5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ModAssignWith.loc249(constants.%Int32, constants.%ModAssignWith.facet.cf2) {
// CHECK:STDOUT:   %U.patt.loc249_19.2 => constants.%U.patt.3a0
// CHECK:STDOUT:   %U.loc249_19.1 => constants.%Int32
// CHECK:STDOUT:   %ModAssignWith.type.loc249_50.1 => constants.%ModAssignWith.type.c90
// CHECK:STDOUT:   %pattern_type.loc249_28 => constants.%pattern_type.065
// CHECK:STDOUT:   %T.patt.loc249_28.2 => constants.%T.patt.707
// CHECK:STDOUT:   %T.loc249_28.1 => constants.%ModAssignWith.facet.cf2
// CHECK:STDOUT:   %T.as_type.loc249_60.1 => constants.%Int64
// CHECK:STDOUT:   %pattern_type.loc249_58 => constants.%pattern_type.4aa
// CHECK:STDOUT:   %x.patt.loc249_58.2 => constants.%x.patt.ace
// CHECK:STDOUT:   %x.param_patt.loc249_53.2 => constants.%x.param_patt.b33
// CHECK:STDOUT:   %pattern_type.loc249_64 => constants.%pattern_type.8b4
// CHECK:STDOUT:   %y.param_patt.loc249_64.2 => constants.%y.param_patt.e0c
// CHECK:STDOUT:   %y.patt.loc249_64.2 => constants.%y.patt.9e1
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc249_53 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc249_64 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc251 => constants.%complete_type.65d
// CHECK:STDOUT:   %ModAssignWith.assoc_type => constants.%ModAssignWith.assoc_type.7d0
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.f20
// CHECK:STDOUT:   %ModAssignWith.WithSelf.Op.type => constants.%ModAssignWith.WithSelf.Op.type.aab
// CHECK:STDOUT:   %.loc251_5.2 => constants.%.253
// CHECK:STDOUT:   %ModAssignWith.lookup_impl_witness => constants.%custom_witness.773f24.2
// CHECK:STDOUT:   %impl.elem0.loc251_5.2 => constants.%Int64.Op.4dca31.6
// CHECK:STDOUT:   %specific_impl_fn.loc251_5.2 => constants.%Int64.Op.4dca31.6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestInc(constants.%T.8f3) {
// CHECK:STDOUT:   %T.patt.loc278_13.2 => constants.%T.patt.eab
// CHECK:STDOUT:   %T.loc278_13.1 => constants.%T.8f3
// CHECK:STDOUT:   %T.as_type.loc278_32.1 => constants.%T.as_type.232
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.427bfc.2
// CHECK:STDOUT:   %x.patt.loc278_30.2 => constants.%x.patt.a75
// CHECK:STDOUT:   %x.param_patt.loc278_25.2 => constants.%x.param_patt.5da
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestDec(constants.%T.e8e) {
// CHECK:STDOUT:   %T.patt.loc284_13.2 => constants.%T.patt.de9
// CHECK:STDOUT:   %T.loc284_13.1 => constants.%T.e8e
// CHECK:STDOUT:   %T.as_type.loc284_32.1 => constants.%T.as_type.c32
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.2754f0.2
// CHECK:STDOUT:   %x.patt.loc284_30.2 => constants.%x.patt.08b
// CHECK:STDOUT:   %x.param_patt.loc284_25.2 => constants.%x.param_patt.678
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestNegate(constants.%Result, constants.%T.d5e) {
// CHECK:STDOUT:   %Result.patt.loc292_9.2 => constants.%Result.patt
// CHECK:STDOUT:   %Result.loc292_9.1 => constants.%Result
// CHECK:STDOUT:   %Result.as_type.loc293_49.1 => constants.%Result.as_type
// CHECK:STDOUT:   %facet_type => constants.%facet_type.86f
// CHECK:STDOUT:   %pattern_type.loc293 => constants.%pattern_type.114
// CHECK:STDOUT:   %T.patt.loc293_4.2 => constants.%T.patt.986
// CHECK:STDOUT:   %T.loc293_4.1 => constants.%T.d5e
// CHECK:STDOUT:   %pattern_type.loc294_11 => constants.%pattern_type.86a9db.1
// CHECK:STDOUT:   %y.param_patt.loc294_11.2 => constants.%y.param_patt.5b5
// CHECK:STDOUT:   %y.patt.loc294_11.2 => constants.%y.patt.1d3
// CHECK:STDOUT:   %T.as_type.loc294_24.1 => constants.%T.as_type.735
// CHECK:STDOUT:   %pattern_type.loc294_22 => constants.%pattern_type.66b2
// CHECK:STDOUT:   %x.param_patt.loc294_22.2 => constants.%x.param_patt.0a8
// CHECK:STDOUT:   %x.patt.loc294_22.2 => constants.%x.patt.424
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestInc(constants.%Inc.facet.dd2) {
// CHECK:STDOUT:   %T.patt.loc278_13.2 => constants.%T.patt.eab
// CHECK:STDOUT:   %T.loc278_13.1 => constants.%Inc.facet.dd2
// CHECK:STDOUT:   %T.as_type.loc278_32.1 => constants.%Int32
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.8b4
// CHECK:STDOUT:   %x.patt.loc278_30.2 => constants.%x.patt.d79
// CHECK:STDOUT:   %x.param_patt.loc278_25.2 => constants.%x.param_patt.0c7
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.357
// CHECK:STDOUT:   %Inc.WithSelf.Op.type => constants.%Inc.WithSelf.Op.type.13b
// CHECK:STDOUT:   %.loc280_3.3 => constants.%.214
// CHECK:STDOUT:   %Inc.lookup_impl_witness => constants.%custom_witness.f3f8
// CHECK:STDOUT:   %impl.elem0.loc280_3.2 => constants.%Int32.Op.4dca31.17
// CHECK:STDOUT:   %specific_impl_fn.loc280_3.2 => constants.%Int32.Op.4dca31.17
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestDec(constants.%Dec.facet.e1b) {
// CHECK:STDOUT:   %T.patt.loc284_13.2 => constants.%T.patt.de9
// CHECK:STDOUT:   %T.loc284_13.1 => constants.%Dec.facet.e1b
// CHECK:STDOUT:   %T.as_type.loc284_32.1 => constants.%Int32
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.8b4
// CHECK:STDOUT:   %x.patt.loc284_30.2 => constants.%x.patt.d79
// CHECK:STDOUT:   %x.param_patt.loc284_25.2 => constants.%x.param_patt.0c7
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.357
// CHECK:STDOUT:   %Dec.WithSelf.Op.type => constants.%Dec.WithSelf.Op.type.0a4
// CHECK:STDOUT:   %.loc286_3.3 => constants.%.55c
// CHECK:STDOUT:   %Dec.lookup_impl_witness => constants.%custom_witness.b24
// CHECK:STDOUT:   %impl.elem0.loc286_3.2 => constants.%Int32.Op.4dca31.18
// CHECK:STDOUT:   %specific_impl_fn.loc286_3.2 => constants.%Int32.Op.4dca31.18
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestNegate(constants.%Destroy.facet.c11, constants.%facet_value.d2e) {
// CHECK:STDOUT:   %Result.patt.loc292_9.2 => constants.%Result.patt
// CHECK:STDOUT:   %Result.loc292_9.1 => constants.%Destroy.facet.c11
// CHECK:STDOUT:   %Result.as_type.loc293_49.1 => constants.%Int64
// CHECK:STDOUT:   %facet_type => constants.%facet_type.8cf
// CHECK:STDOUT:   %pattern_type.loc293 => constants.%pattern_type.4c7
// CHECK:STDOUT:   %T.patt.loc293_4.2 => constants.%T.patt.0b0
// CHECK:STDOUT:   %T.loc293_4.1 => constants.%facet_value.d2e
// CHECK:STDOUT:   %pattern_type.loc294_11 => constants.%pattern_type.4aa
// CHECK:STDOUT:   %y.param_patt.loc294_11.2 => constants.%y.param_patt.a08
// CHECK:STDOUT:   %y.patt.loc294_11.2 => constants.%y.patt.069
// CHECK:STDOUT:   %T.as_type.loc294_24.1 => constants.%Int16
// CHECK:STDOUT:   %pattern_type.loc294_22 => constants.%pattern_type.785
// CHECK:STDOUT:   %x.param_patt.loc294_22.2 => constants.%x.param_patt.3b0
// CHECK:STDOUT:   %x.patt.loc294_22.2 => constants.%x.patt.869
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc294_11 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc294_22 => constants.%complete_type.357
// CHECK:STDOUT:   %Negate.lookup_impl_witness => constants.%custom_witness.100
// CHECK:STDOUT:   %Negate.facet.loc296_3.3 => constants.%Negate.facet.8f4
// CHECK:STDOUT:   %Negate.WithSelf.Op.type => constants.%Negate.WithSelf.Op.type.34a
// CHECK:STDOUT:   %.loc296_3.7 => constants.%.789
// CHECK:STDOUT:   %impl.elem1.loc296_3.2 => constants.%Int16.Op.4dca31.17
// CHECK:STDOUT:   %specific_impl_fn.loc296_3.2 => constants.%Int16.Op.4dca31.17
// CHECK:STDOUT:   %SelfDestruct.specific_fn.2 => constants.%SelfDestruct.specific_fn.fb0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestNegate(constants.%Destroy.facet.806, constants.%facet_value.8a2) {
// CHECK:STDOUT:   %Result.patt.loc292_9.2 => constants.%Result.patt
// CHECK:STDOUT:   %Result.loc292_9.1 => constants.%Destroy.facet.806
// CHECK:STDOUT:   %Result.as_type.loc293_49.1 => constants.%Int32
// CHECK:STDOUT:   %facet_type => constants.%facet_type.517
// CHECK:STDOUT:   %pattern_type.loc293 => constants.%pattern_type.eaa
// CHECK:STDOUT:   %T.patt.loc293_4.2 => constants.%T.patt.532
// CHECK:STDOUT:   %T.loc293_4.1 => constants.%facet_value.8a2
// CHECK:STDOUT:   %pattern_type.loc294_11 => constants.%pattern_type.8b4
// CHECK:STDOUT:   %y.param_patt.loc294_11.2 => constants.%y.param_patt.e0c
// CHECK:STDOUT:   %y.patt.loc294_11.2 => constants.%y.patt.9e1
// CHECK:STDOUT:   %T.as_type.loc294_24.1 => constants.%Int32
// CHECK:STDOUT:   %pattern_type.loc294_22 => constants.%pattern_type.8b4
// CHECK:STDOUT:   %x.param_patt.loc294_22.2 => constants.%x.param_patt.27c
// CHECK:STDOUT:   %x.patt.loc294_22.2 => constants.%x.patt.740
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc294_11 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc294_22 => constants.%complete_type.357
// CHECK:STDOUT:   %Negate.lookup_impl_witness => constants.%custom_witness.a85
// CHECK:STDOUT:   %Negate.facet.loc296_3.3 => constants.%Negate.facet.aa9
// CHECK:STDOUT:   %Negate.WithSelf.Op.type => constants.%Negate.WithSelf.Op.type.25a
// CHECK:STDOUT:   %.loc296_3.7 => constants.%.dcd
// CHECK:STDOUT:   %impl.elem1.loc296_3.2 => constants.%Int32.Op.4dca31.19
// CHECK:STDOUT:   %specific_impl_fn.loc296_3.2 => constants.%Int32.Op.4dca31.19
// CHECK:STDOUT:   %SelfDestruct.specific_fn.2 => constants.%SelfDestruct.specific_fn.13e
// CHECK:STDOUT: }
// CHECK:STDOUT:
