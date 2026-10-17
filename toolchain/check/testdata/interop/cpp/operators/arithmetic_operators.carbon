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
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %facet_type.ece: type = facet_type <@DestroyImpl & @SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %pattern_type.735: type = pattern_type %facet_type.ece [concrete]
// CHECK:STDOUT:   %Result.patt: %pattern_type.735 = symbolic_binding_pattern Result, 0 [symbolic]
// CHECK:STDOUT:   %Result: %facet_type.ece = symbolic_binding Result, 0 [symbolic]
// CHECK:STDOUT:   %U.patt.4c9: %pattern_type.735 = symbolic_binding_pattern U, 1 [symbolic]
// CHECK:STDOUT:   %U.84b: %facet_type.ece = symbolic_binding U, 1 [symbolic]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Other: type = symbolic_binding Other, 0 [symbolic]
// CHECK:STDOUT:   %AddWith.type.872: type = facet_type <@AddWith.1, @AddWith.1(%Other)> [symbolic]
// CHECK:STDOUT:   %pattern_type.9a5: type = pattern_type type [concrete]
// CHECK:STDOUT:   %Self.958: %AddWith.type.872 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %AddWith.assoc_type.1d5: type = assoc_entity_type @AddWith.1, @AddWith.1(%Other) [symbolic]
// CHECK:STDOUT:   %AddWith.WithSelf.Op.type.70b: type = fn_type @AddWith.WithSelf.Op, @AddWith.WithSelf(%Other, %Self.958) [symbolic]
// CHECK:STDOUT:   %AddWith.WithSelf.Op.bb1: %AddWith.WithSelf.Op.type.70b = struct_value () [symbolic]
// CHECK:STDOUT:   %U.as_type.15c: type = facet_access_type %U.84b [symbolic]
// CHECK:STDOUT:   %AddWith.type.55e: type = facet_type <@AddWith.1, @AddWith.1(%U.as_type.15c)> [symbolic]
// CHECK:STDOUT:   %facet_type.39b: type = facet_type <@DestroyImpl & @SubobjectDestroy & @AddWith.1, @AddWith.1(%U.as_type.15c)> [symbolic]
// CHECK:STDOUT:   %.Self.frozen.c40: %facet_type.39b = symbolic_binding .Self [symbolic]
// CHECK:STDOUT:   %AddWith.assoc_type.cc7: type = assoc_entity_type @AddWith.1, @AddWith.1(%U.as_type.15c) [symbolic]
// CHECK:STDOUT:   %assoc0.920: %AddWith.assoc_type.cc7 = assoc_entity element0, imports.%Core.import_ref.91f [symbolic]
// CHECK:STDOUT:   %assoc1.6cf: %AddWith.assoc_type.cc7 = assoc_entity element1, imports.%Core.import_ref.8ad [symbolic]
// CHECK:STDOUT:   %require_complete.1b3: <witness> = require_complete_type %facet_type.39b [symbolic]
// CHECK:STDOUT:   %.Self.frozen.as_type.459: type = facet_access_type %.Self.frozen.c40 [symbolic]
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.2e9: <witness> = lookup_impl_witness %.Self.frozen.c40, @AddWith.1, @AddWith.1(%U.as_type.15c) [symbolic]
// CHECK:STDOUT:   %impl.elem0.58a: type = impl_witness_access %AddWith.lookup_impl_witness.2e9, element0 [symbolic]
// CHECK:STDOUT:   %Result.as_type: type = facet_access_type %Result [symbolic]
// CHECK:STDOUT:   %.Self.878: %facet_type.39b = symbolic_binding .Self [symbolic]
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.cf2: <witness> = lookup_impl_witness %.Self.878, @AddWith.1, @AddWith.1(%U.as_type.15c) [symbolic]
// CHECK:STDOUT:   %impl.elem0.f49: type = impl_witness_access %AddWith.lookup_impl_witness.cf2, element0 [symbolic]
// CHECK:STDOUT:   %facet_type.0bd: type = facet_type <@DestroyImpl & @SubobjectDestroy & @AddWith.1, @AddWith.1(%U.as_type.15c) where %impl.elem0.f49 = %Result.as_type> [symbolic]
// CHECK:STDOUT:   %pattern_type.e9f: type = pattern_type %facet_type.0bd [symbolic]
// CHECK:STDOUT:   %T.patt.35d: %pattern_type.e9f = symbolic_binding_pattern T, 2 [symbolic]
// CHECK:STDOUT:   %T.d3f: %facet_type.0bd = symbolic_binding T, 2 [symbolic]
// CHECK:STDOUT:   %pattern_type.e4c132.1: type = pattern_type %Result.as_type [symbolic]
// CHECK:STDOUT:   %result.param_patt.b19: %pattern_type.e4c132.1 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %result.patt.583fb4.1: %pattern_type.e4c132.1 = wrapper_binding_pattern result, %result.param_patt.b19 [symbolic]
// CHECK:STDOUT:   %T.as_type.d99: type = facet_access_type %T.d3f [symbolic]
// CHECK:STDOUT:   %pattern_type.734: type = pattern_type %T.as_type.d99 [symbolic]
// CHECK:STDOUT:   %x.param_patt.17a: %pattern_type.734 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %x.patt.982: %pattern_type.734 = wrapper_binding_pattern x, %x.param_patt.17a [symbolic]
// CHECK:STDOUT:   %pattern_type.d0c4: type = pattern_type %U.as_type.15c [symbolic]
// CHECK:STDOUT:   %y.param_patt.14c: %pattern_type.d0c4 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %y.patt.efa: %pattern_type.d0c4 = wrapper_binding_pattern y, %y.param_patt.14c [symbolic]
// CHECK:STDOUT:   %require_complete.2d4: <witness> = require_complete_type %AddWith.type.55e [symbolic]
// CHECK:STDOUT:   %assoc1.84a: %AddWith.assoc_type.1d5 = assoc_entity element1, imports.%Core.import_ref.e9a [symbolic]
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.f7f: <witness> = lookup_impl_witness %T.d3f, @AddWith.1, @AddWith.1(%U.as_type.15c) [symbolic]
// CHECK:STDOUT:   %AddWith.facet.a2a: %AddWith.type.55e = facet_value %T.as_type.d99, (%AddWith.lookup_impl_witness.f7f) [symbolic]
// CHECK:STDOUT:   %AddWith.WithSelf.Op.type.b84: type = fn_type @AddWith.WithSelf.Op, @AddWith.WithSelf(%U.as_type.15c, %AddWith.facet.a2a) [symbolic]
// CHECK:STDOUT:   %.28a: type = fn_type_with_self_type %AddWith.WithSelf.Op.type.b84, %AddWith.facet.a2a [symbolic]
// CHECK:STDOUT:   %impl.elem1.5d2: %.28a = impl_witness_access %AddWith.lookup_impl_witness.f7f, element1 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.140: <specific function> = specific_impl_function %impl.elem1.5d2, @AddWith.WithSelf.Op(%U.as_type.15c, %AddWith.facet.a2a) [symbolic]
// CHECK:STDOUT:   %SelfDestruct.type: type = fn_type @SelfDestruct [concrete]
// CHECK:STDOUT:   %SelfDestruct: %SelfDestruct.type = struct_value () [concrete]
// CHECK:STDOUT:   %SubWith.type.73b: type = facet_type <@SubWith.1, @SubWith.1(%Other)> [symbolic]
// CHECK:STDOUT:   %Self.18a: %SubWith.type.73b = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %SubWith.assoc_type.300: type = assoc_entity_type @SubWith.1, @SubWith.1(%Other) [symbolic]
// CHECK:STDOUT:   %SubWith.WithSelf.Op.type.028: type = fn_type @SubWith.WithSelf.Op, @SubWith.WithSelf(%Other, %Self.18a) [symbolic]
// CHECK:STDOUT:   %SubWith.WithSelf.Op.e84: %SubWith.WithSelf.Op.type.028 = struct_value () [symbolic]
// CHECK:STDOUT:   %SubWith.type.776: type = facet_type <@SubWith.1, @SubWith.1(%U.as_type.15c)> [symbolic]
// CHECK:STDOUT:   %facet_type.8f9: type = facet_type <@DestroyImpl & @SubobjectDestroy & @SubWith.1, @SubWith.1(%U.as_type.15c)> [symbolic]
// CHECK:STDOUT:   %.Self.frozen.ca0: %facet_type.8f9 = symbolic_binding .Self [symbolic]
// CHECK:STDOUT:   %SubWith.assoc_type.cf4: type = assoc_entity_type @SubWith.1, @SubWith.1(%U.as_type.15c) [symbolic]
// CHECK:STDOUT:   %assoc0.963: %SubWith.assoc_type.cf4 = assoc_entity element0, imports.%Core.import_ref.190 [symbolic]
// CHECK:STDOUT:   %assoc1.543: %SubWith.assoc_type.cf4 = assoc_entity element1, imports.%Core.import_ref.2f4 [symbolic]
// CHECK:STDOUT:   %require_complete.b4d: <witness> = require_complete_type %facet_type.8f9 [symbolic]
// CHECK:STDOUT:   %.Self.frozen.as_type.ff3: type = facet_access_type %.Self.frozen.ca0 [symbolic]
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.909: <witness> = lookup_impl_witness %.Self.frozen.ca0, @SubWith.1, @SubWith.1(%U.as_type.15c) [symbolic]
// CHECK:STDOUT:   %impl.elem0.92e: type = impl_witness_access %SubWith.lookup_impl_witness.909, element0 [symbolic]
// CHECK:STDOUT:   %.Self.355: %facet_type.8f9 = symbolic_binding .Self [symbolic]
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.9ec: <witness> = lookup_impl_witness %.Self.355, @SubWith.1, @SubWith.1(%U.as_type.15c) [symbolic]
// CHECK:STDOUT:   %impl.elem0.f58: type = impl_witness_access %SubWith.lookup_impl_witness.9ec, element0 [symbolic]
// CHECK:STDOUT:   %facet_type.952: type = facet_type <@DestroyImpl & @SubobjectDestroy & @SubWith.1, @SubWith.1(%U.as_type.15c) where %impl.elem0.f58 = %Result.as_type> [symbolic]
// CHECK:STDOUT:   %pattern_type.70e: type = pattern_type %facet_type.952 [symbolic]
// CHECK:STDOUT:   %T.patt.b38: %pattern_type.70e = symbolic_binding_pattern T, 2 [symbolic]
// CHECK:STDOUT:   %T.b27: %facet_type.952 = symbolic_binding T, 2 [symbolic]
// CHECK:STDOUT:   %T.as_type.137: type = facet_access_type %T.b27 [symbolic]
// CHECK:STDOUT:   %pattern_type.6b7: type = pattern_type %T.as_type.137 [symbolic]
// CHECK:STDOUT:   %x.param_patt.1ce: %pattern_type.6b7 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %x.patt.ab5: %pattern_type.6b7 = wrapper_binding_pattern x, %x.param_patt.1ce [symbolic]
// CHECK:STDOUT:   %require_complete.6c6: <witness> = require_complete_type %SubWith.type.776 [symbolic]
// CHECK:STDOUT:   %assoc1.b58: %SubWith.assoc_type.300 = assoc_entity element1, imports.%Core.import_ref.124 [symbolic]
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.c46: <witness> = lookup_impl_witness %T.b27, @SubWith.1, @SubWith.1(%U.as_type.15c) [symbolic]
// CHECK:STDOUT:   %SubWith.facet.26d: %SubWith.type.776 = facet_value %T.as_type.137, (%SubWith.lookup_impl_witness.c46) [symbolic]
// CHECK:STDOUT:   %SubWith.WithSelf.Op.type.0e6: type = fn_type @SubWith.WithSelf.Op, @SubWith.WithSelf(%U.as_type.15c, %SubWith.facet.26d) [symbolic]
// CHECK:STDOUT:   %.3a1: type = fn_type_with_self_type %SubWith.WithSelf.Op.type.0e6, %SubWith.facet.26d [symbolic]
// CHECK:STDOUT:   %impl.elem1.1b8: %.3a1 = impl_witness_access %SubWith.lookup_impl_witness.c46, element1 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.0da: <specific function> = specific_impl_function %impl.elem1.1b8, @SubWith.WithSelf.Op(%U.as_type.15c, %SubWith.facet.26d) [symbolic]
// CHECK:STDOUT:   %MulWith.type.616: type = facet_type <@MulWith.1, @MulWith.1(%Other)> [symbolic]
// CHECK:STDOUT:   %Self.742: %MulWith.type.616 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %MulWith.assoc_type.c81: type = assoc_entity_type @MulWith.1, @MulWith.1(%Other) [symbolic]
// CHECK:STDOUT:   %MulWith.WithSelf.Op.type.dad: type = fn_type @MulWith.WithSelf.Op, @MulWith.WithSelf(%Other, %Self.742) [symbolic]
// CHECK:STDOUT:   %MulWith.WithSelf.Op.507: %MulWith.WithSelf.Op.type.dad = struct_value () [symbolic]
// CHECK:STDOUT:   %MulWith.type.068: type = facet_type <@MulWith.1, @MulWith.1(%U.as_type.15c)> [symbolic]
// CHECK:STDOUT:   %facet_type.a35: type = facet_type <@DestroyImpl & @SubobjectDestroy & @MulWith.1, @MulWith.1(%U.as_type.15c)> [symbolic]
// CHECK:STDOUT:   %.Self.frozen.87a: %facet_type.a35 = symbolic_binding .Self [symbolic]
// CHECK:STDOUT:   %MulWith.assoc_type.083: type = assoc_entity_type @MulWith.1, @MulWith.1(%U.as_type.15c) [symbolic]
// CHECK:STDOUT:   %assoc0.882: %MulWith.assoc_type.083 = assoc_entity element0, imports.%Core.import_ref.59c [symbolic]
// CHECK:STDOUT:   %assoc1.2ea: %MulWith.assoc_type.083 = assoc_entity element1, imports.%Core.import_ref.fc2 [symbolic]
// CHECK:STDOUT:   %require_complete.8d9: <witness> = require_complete_type %facet_type.a35 [symbolic]
// CHECK:STDOUT:   %.Self.frozen.as_type.1de: type = facet_access_type %.Self.frozen.87a [symbolic]
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.b96: <witness> = lookup_impl_witness %.Self.frozen.87a, @MulWith.1, @MulWith.1(%U.as_type.15c) [symbolic]
// CHECK:STDOUT:   %impl.elem0.25a: type = impl_witness_access %MulWith.lookup_impl_witness.b96, element0 [symbolic]
// CHECK:STDOUT:   %.Self.1a7: %facet_type.a35 = symbolic_binding .Self [symbolic]
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.7c7: <witness> = lookup_impl_witness %.Self.1a7, @MulWith.1, @MulWith.1(%U.as_type.15c) [symbolic]
// CHECK:STDOUT:   %impl.elem0.155: type = impl_witness_access %MulWith.lookup_impl_witness.7c7, element0 [symbolic]
// CHECK:STDOUT:   %facet_type.6e8: type = facet_type <@DestroyImpl & @SubobjectDestroy & @MulWith.1, @MulWith.1(%U.as_type.15c) where %impl.elem0.155 = %Result.as_type> [symbolic]
// CHECK:STDOUT:   %pattern_type.8df: type = pattern_type %facet_type.6e8 [symbolic]
// CHECK:STDOUT:   %T.patt.35a: %pattern_type.8df = symbolic_binding_pattern T, 2 [symbolic]
// CHECK:STDOUT:   %T.470: %facet_type.6e8 = symbolic_binding T, 2 [symbolic]
// CHECK:STDOUT:   %T.as_type.d42: type = facet_access_type %T.470 [symbolic]
// CHECK:STDOUT:   %pattern_type.328: type = pattern_type %T.as_type.d42 [symbolic]
// CHECK:STDOUT:   %x.param_patt.066: %pattern_type.328 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %x.patt.f71: %pattern_type.328 = wrapper_binding_pattern x, %x.param_patt.066 [symbolic]
// CHECK:STDOUT:   %require_complete.91f: <witness> = require_complete_type %MulWith.type.068 [symbolic]
// CHECK:STDOUT:   %assoc1.d9e: %MulWith.assoc_type.c81 = assoc_entity element1, imports.%Core.import_ref.29d [symbolic]
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.10d: <witness> = lookup_impl_witness %T.470, @MulWith.1, @MulWith.1(%U.as_type.15c) [symbolic]
// CHECK:STDOUT:   %MulWith.facet.aa0: %MulWith.type.068 = facet_value %T.as_type.d42, (%MulWith.lookup_impl_witness.10d) [symbolic]
// CHECK:STDOUT:   %MulWith.WithSelf.Op.type.e1e: type = fn_type @MulWith.WithSelf.Op, @MulWith.WithSelf(%U.as_type.15c, %MulWith.facet.aa0) [symbolic]
// CHECK:STDOUT:   %.7cf: type = fn_type_with_self_type %MulWith.WithSelf.Op.type.e1e, %MulWith.facet.aa0 [symbolic]
// CHECK:STDOUT:   %impl.elem1.0d2: %.7cf = impl_witness_access %MulWith.lookup_impl_witness.10d, element1 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.e0c: <specific function> = specific_impl_function %impl.elem1.0d2, @MulWith.WithSelf.Op(%U.as_type.15c, %MulWith.facet.aa0) [symbolic]
// CHECK:STDOUT:   %DivWith.type.472: type = facet_type <@DivWith.1, @DivWith.1(%Other)> [symbolic]
// CHECK:STDOUT:   %Self.5b3: %DivWith.type.472 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %DivWith.assoc_type.e7d: type = assoc_entity_type @DivWith.1, @DivWith.1(%Other) [symbolic]
// CHECK:STDOUT:   %DivWith.WithSelf.Op.type.207: type = fn_type @DivWith.WithSelf.Op, @DivWith.WithSelf(%Other, %Self.5b3) [symbolic]
// CHECK:STDOUT:   %DivWith.WithSelf.Op.4f1: %DivWith.WithSelf.Op.type.207 = struct_value () [symbolic]
// CHECK:STDOUT:   %DivWith.type.687: type = facet_type <@DivWith.1, @DivWith.1(%U.as_type.15c)> [symbolic]
// CHECK:STDOUT:   %facet_type.519: type = facet_type <@DestroyImpl & @SubobjectDestroy & @DivWith.1, @DivWith.1(%U.as_type.15c)> [symbolic]
// CHECK:STDOUT:   %.Self.frozen.2fc: %facet_type.519 = symbolic_binding .Self [symbolic]
// CHECK:STDOUT:   %DivWith.assoc_type.9a3: type = assoc_entity_type @DivWith.1, @DivWith.1(%U.as_type.15c) [symbolic]
// CHECK:STDOUT:   %assoc0.913: %DivWith.assoc_type.9a3 = assoc_entity element0, imports.%Core.import_ref.dee [symbolic]
// CHECK:STDOUT:   %assoc1.1bc: %DivWith.assoc_type.9a3 = assoc_entity element1, imports.%Core.import_ref.f2f [symbolic]
// CHECK:STDOUT:   %require_complete.660: <witness> = require_complete_type %facet_type.519 [symbolic]
// CHECK:STDOUT:   %.Self.frozen.as_type.11b: type = facet_access_type %.Self.frozen.2fc [symbolic]
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.354: <witness> = lookup_impl_witness %.Self.frozen.2fc, @DivWith.1, @DivWith.1(%U.as_type.15c) [symbolic]
// CHECK:STDOUT:   %impl.elem0.dfa: type = impl_witness_access %DivWith.lookup_impl_witness.354, element0 [symbolic]
// CHECK:STDOUT:   %.Self.310: %facet_type.519 = symbolic_binding .Self [symbolic]
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.3de: <witness> = lookup_impl_witness %.Self.310, @DivWith.1, @DivWith.1(%U.as_type.15c) [symbolic]
// CHECK:STDOUT:   %impl.elem0.e74: type = impl_witness_access %DivWith.lookup_impl_witness.3de, element0 [symbolic]
// CHECK:STDOUT:   %facet_type.344: type = facet_type <@DestroyImpl & @SubobjectDestroy & @DivWith.1, @DivWith.1(%U.as_type.15c) where %impl.elem0.e74 = %Result.as_type> [symbolic]
// CHECK:STDOUT:   %pattern_type.a7c: type = pattern_type %facet_type.344 [symbolic]
// CHECK:STDOUT:   %T.patt.436: %pattern_type.a7c = symbolic_binding_pattern T, 2 [symbolic]
// CHECK:STDOUT:   %T.9b4: %facet_type.344 = symbolic_binding T, 2 [symbolic]
// CHECK:STDOUT:   %T.as_type.6cf: type = facet_access_type %T.9b4 [symbolic]
// CHECK:STDOUT:   %pattern_type.980: type = pattern_type %T.as_type.6cf [symbolic]
// CHECK:STDOUT:   %x.param_patt.cd2: %pattern_type.980 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %x.patt.d99: %pattern_type.980 = wrapper_binding_pattern x, %x.param_patt.cd2 [symbolic]
// CHECK:STDOUT:   %require_complete.a62: <witness> = require_complete_type %DivWith.type.687 [symbolic]
// CHECK:STDOUT:   %assoc1.2b8: %DivWith.assoc_type.e7d = assoc_entity element1, imports.%Core.import_ref.94d [symbolic]
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.51a: <witness> = lookup_impl_witness %T.9b4, @DivWith.1, @DivWith.1(%U.as_type.15c) [symbolic]
// CHECK:STDOUT:   %DivWith.facet.901: %DivWith.type.687 = facet_value %T.as_type.6cf, (%DivWith.lookup_impl_witness.51a) [symbolic]
// CHECK:STDOUT:   %DivWith.WithSelf.Op.type.396: type = fn_type @DivWith.WithSelf.Op, @DivWith.WithSelf(%U.as_type.15c, %DivWith.facet.901) [symbolic]
// CHECK:STDOUT:   %.422: type = fn_type_with_self_type %DivWith.WithSelf.Op.type.396, %DivWith.facet.901 [symbolic]
// CHECK:STDOUT:   %impl.elem1.338: %.422 = impl_witness_access %DivWith.lookup_impl_witness.51a, element1 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.8b9: <specific function> = specific_impl_function %impl.elem1.338, @DivWith.WithSelf.Op(%U.as_type.15c, %DivWith.facet.901) [symbolic]
// CHECK:STDOUT:   %ModWith.type.11e: type = facet_type <@ModWith.1, @ModWith.1(%Other)> [symbolic]
// CHECK:STDOUT:   %Self.c18: %ModWith.type.11e = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %ModWith.assoc_type.1f1: type = assoc_entity_type @ModWith.1, @ModWith.1(%Other) [symbolic]
// CHECK:STDOUT:   %ModWith.WithSelf.Op.type.04e: type = fn_type @ModWith.WithSelf.Op, @ModWith.WithSelf(%Other, %Self.c18) [symbolic]
// CHECK:STDOUT:   %ModWith.WithSelf.Op.6fb: %ModWith.WithSelf.Op.type.04e = struct_value () [symbolic]
// CHECK:STDOUT:   %ModWith.type.584: type = facet_type <@ModWith.1, @ModWith.1(%U.as_type.15c)> [symbolic]
// CHECK:STDOUT:   %facet_type.baa: type = facet_type <@DestroyImpl & @SubobjectDestroy & @ModWith.1, @ModWith.1(%U.as_type.15c)> [symbolic]
// CHECK:STDOUT:   %.Self.frozen.d89: %facet_type.baa = symbolic_binding .Self [symbolic]
// CHECK:STDOUT:   %ModWith.assoc_type.279: type = assoc_entity_type @ModWith.1, @ModWith.1(%U.as_type.15c) [symbolic]
// CHECK:STDOUT:   %assoc0.431: %ModWith.assoc_type.279 = assoc_entity element0, imports.%Core.import_ref.61a [symbolic]
// CHECK:STDOUT:   %assoc1.bb3: %ModWith.assoc_type.279 = assoc_entity element1, imports.%Core.import_ref.9f8f [symbolic]
// CHECK:STDOUT:   %require_complete.1f1: <witness> = require_complete_type %facet_type.baa [symbolic]
// CHECK:STDOUT:   %.Self.frozen.as_type.49a: type = facet_access_type %.Self.frozen.d89 [symbolic]
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.966: <witness> = lookup_impl_witness %.Self.frozen.d89, @ModWith.1, @ModWith.1(%U.as_type.15c) [symbolic]
// CHECK:STDOUT:   %impl.elem0.c4e: type = impl_witness_access %ModWith.lookup_impl_witness.966, element0 [symbolic]
// CHECK:STDOUT:   %.Self.055: %facet_type.baa = symbolic_binding .Self [symbolic]
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.041: <witness> = lookup_impl_witness %.Self.055, @ModWith.1, @ModWith.1(%U.as_type.15c) [symbolic]
// CHECK:STDOUT:   %impl.elem0.269: type = impl_witness_access %ModWith.lookup_impl_witness.041, element0 [symbolic]
// CHECK:STDOUT:   %facet_type.668: type = facet_type <@DestroyImpl & @SubobjectDestroy & @ModWith.1, @ModWith.1(%U.as_type.15c) where %impl.elem0.269 = %Result.as_type> [symbolic]
// CHECK:STDOUT:   %pattern_type.3c9: type = pattern_type %facet_type.668 [symbolic]
// CHECK:STDOUT:   %T.patt.d3a: %pattern_type.3c9 = symbolic_binding_pattern T, 2 [symbolic]
// CHECK:STDOUT:   %T.af4: %facet_type.668 = symbolic_binding T, 2 [symbolic]
// CHECK:STDOUT:   %T.as_type.a02: type = facet_access_type %T.af4 [symbolic]
// CHECK:STDOUT:   %pattern_type.72e: type = pattern_type %T.as_type.a02 [symbolic]
// CHECK:STDOUT:   %x.param_patt.dc7: %pattern_type.72e = value_param_pattern [symbolic]
// CHECK:STDOUT:   %x.patt.a0b: %pattern_type.72e = wrapper_binding_pattern x, %x.param_patt.dc7 [symbolic]
// CHECK:STDOUT:   %require_complete.714: <witness> = require_complete_type %ModWith.type.584 [symbolic]
// CHECK:STDOUT:   %assoc1.f4b: %ModWith.assoc_type.1f1 = assoc_entity element1, imports.%Core.import_ref.be00 [symbolic]
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.03e: <witness> = lookup_impl_witness %T.af4, @ModWith.1, @ModWith.1(%U.as_type.15c) [symbolic]
// CHECK:STDOUT:   %ModWith.facet.b48: %ModWith.type.584 = facet_value %T.as_type.a02, (%ModWith.lookup_impl_witness.03e) [symbolic]
// CHECK:STDOUT:   %ModWith.WithSelf.Op.type.bef: type = fn_type @ModWith.WithSelf.Op, @ModWith.WithSelf(%U.as_type.15c, %ModWith.facet.b48) [symbolic]
// CHECK:STDOUT:   %.5369: type = fn_type_with_self_type %ModWith.WithSelf.Op.type.bef, %ModWith.facet.b48 [symbolic]
// CHECK:STDOUT:   %impl.elem1.275: %.5369 = impl_witness_access %ModWith.lookup_impl_witness.03e, element1 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.bba: <specific function> = specific_impl_function %impl.elem1.275, @ModWith.WithSelf.Op(%U.as_type.15c, %ModWith.facet.b48) [symbolic]
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
// CHECK:STDOUT:   %facet_value.a87: %facet_type.ece = facet_value %Int16, (%DestroyImpl.impl_witness.891, %custom_witness.b80ef6.1) [concrete]
// CHECK:STDOUT:   %facet_type.299: type = facet_type <@DestroyImpl & @SubobjectDestroy & @AddWith.1, @AddWith.1(%Int16)> [concrete]
// CHECK:STDOUT:   %.Self.32a: %facet_type.299 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.da9: <witness> = lookup_impl_witness %.Self.32a, @AddWith.1, @AddWith.1(%Int16) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.4ef: type = impl_witness_access %AddWith.lookup_impl_witness.da9, element0 [symbolic_self]
// CHECK:STDOUT:   %facet_type.3a6: type = facet_type <@DestroyImpl & @SubobjectDestroy & @AddWith.1, @AddWith.1(%Int16) where %impl.elem0.4ef = %Int16> [concrete]
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
// CHECK:STDOUT:   %facet_value.e48: %facet_type.ece = facet_value %i16, (%DestroyImpl.impl_witness.dff, %custom_witness.b80ef6.3) [concrete]
// CHECK:STDOUT:   %custom_witness.cba: <witness> = custom_witness (%Int16, %Op.72717c.1), @AddWith.1, @AddWith.1(%Int16) [concrete]
// CHECK:STDOUT:   %facet_value.d2f: %facet_type.3a6 = facet_value %i16, (%DestroyImpl.impl_witness.dff, %custom_witness.b80ef6.3, %custom_witness.cba) [concrete]
// CHECK:STDOUT:   %.Self.frozen.f66: %facet_type.299 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %complete_type.6e0: <witness> = complete_type_witness %facet_type.299 [concrete]
// CHECK:STDOUT:   %.Self.frozen.as_type.ea2: type = facet_access_type %.Self.frozen.f66 [symbolic_self]
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.451: <witness> = lookup_impl_witness %.Self.frozen.f66, @AddWith.1, @AddWith.1(%Int16) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.f96: type = impl_witness_access %AddWith.lookup_impl_witness.451, element0 [symbolic_self]
// CHECK:STDOUT:   %pattern_type.26d: type = pattern_type %facet_type.3a6 [concrete]
// CHECK:STDOUT:   %T.patt.a9b: %pattern_type.26d = symbolic_binding_pattern T, 2 [symbolic]
// CHECK:STDOUT:   %result.param_patt.be6: %pattern_type.785 = value_param_pattern [concrete]
// CHECK:STDOUT:   %result.patt.a31: %pattern_type.785 = wrapper_binding_pattern result, %result.param_patt.be6 [concrete]
// CHECK:STDOUT:   %y.param_patt.418: %pattern_type.785 = value_param_pattern [concrete]
// CHECK:STDOUT:   %y.patt.890: %pattern_type.785 = wrapper_binding_pattern y, %y.param_patt.418 [concrete]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.65d: <specific function> = specific_function %SelfDestruct, @SelfDestruct(%facet_value.a87) [concrete]
// CHECK:STDOUT:   %facet_type.eed: type = facet_type <@DestroyImpl & @SubobjectDestroy & @SubWith.1, @SubWith.1(%Int16)> [concrete]
// CHECK:STDOUT:   %.Self.3d7: %facet_type.eed = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.dde: <witness> = lookup_impl_witness %.Self.3d7, @SubWith.1, @SubWith.1(%Int16) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.cb3: type = impl_witness_access %SubWith.lookup_impl_witness.dde, element0 [symbolic_self]
// CHECK:STDOUT:   %facet_type.129: type = facet_type <@DestroyImpl & @SubobjectDestroy & @SubWith.1, @SubWith.1(%Int16) where %impl.elem0.cb3 = %Int16> [concrete]
// CHECK:STDOUT:   %SubWith.type.68d: type = facet_type <@SubWith.1, @SubWith.1(%Int16)> [concrete]
// CHECK:STDOUT:   %SubWith.assoc_type.dd6: type = assoc_entity_type @SubWith.1, @SubWith.1(%Int16) [concrete]
// CHECK:STDOUT:   %assoc0.dfb: %SubWith.assoc_type.dd6 = assoc_entity element0, imports.%Core.import_ref.190 [concrete]
// CHECK:STDOUT:   %assoc1.ada: %SubWith.assoc_type.dd6 = assoc_entity element1, imports.%Core.import_ref.2f4 [concrete]
// CHECK:STDOUT:   %Op.type.442b68.2: type = fn_type @Op.2 [concrete]
// CHECK:STDOUT:   %Op.72717c.2: %Op.type.442b68.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.354: <witness> = custom_witness (%Int16, %Op.72717c.2), @SubWith.1, @SubWith.1(%Int16) [concrete]
// CHECK:STDOUT:   %facet_value.4fe: %facet_type.129 = facet_value %i16, (%DestroyImpl.impl_witness.dff, %custom_witness.b80ef6.3, %custom_witness.354) [concrete]
// CHECK:STDOUT:   %.Self.frozen.669: %facet_type.eed = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %complete_type.5eb: <witness> = complete_type_witness %facet_type.eed [concrete]
// CHECK:STDOUT:   %.Self.frozen.as_type.aff: type = facet_access_type %.Self.frozen.669 [symbolic_self]
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.46e: <witness> = lookup_impl_witness %.Self.frozen.669, @SubWith.1, @SubWith.1(%Int16) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.168: type = impl_witness_access %SubWith.lookup_impl_witness.46e, element0 [symbolic_self]
// CHECK:STDOUT:   %pattern_type.d0c7: type = pattern_type %facet_type.129 [concrete]
// CHECK:STDOUT:   %T.patt.67c: %pattern_type.d0c7 = symbolic_binding_pattern T, 2 [symbolic]
// CHECK:STDOUT:   %facet_type.eda: type = facet_type <@DestroyImpl & @SubobjectDestroy & @MulWith.1, @MulWith.1(%Int16)> [concrete]
// CHECK:STDOUT:   %.Self.a47: %facet_type.eda = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.923: <witness> = lookup_impl_witness %.Self.a47, @MulWith.1, @MulWith.1(%Int16) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.567: type = impl_witness_access %MulWith.lookup_impl_witness.923, element0 [symbolic_self]
// CHECK:STDOUT:   %facet_type.2c7: type = facet_type <@DestroyImpl & @SubobjectDestroy & @MulWith.1, @MulWith.1(%Int16) where %impl.elem0.567 = %Int16> [concrete]
// CHECK:STDOUT:   %MulWith.type.f65: type = facet_type <@MulWith.1, @MulWith.1(%Int16)> [concrete]
// CHECK:STDOUT:   %MulWith.assoc_type.4c3: type = assoc_entity_type @MulWith.1, @MulWith.1(%Int16) [concrete]
// CHECK:STDOUT:   %assoc0.6ec: %MulWith.assoc_type.4c3 = assoc_entity element0, imports.%Core.import_ref.59c [concrete]
// CHECK:STDOUT:   %assoc1.b14: %MulWith.assoc_type.4c3 = assoc_entity element1, imports.%Core.import_ref.fc2 [concrete]
// CHECK:STDOUT:   %Op.type.442b68.3: type = fn_type @Op.3 [concrete]
// CHECK:STDOUT:   %Op.72717c.3: %Op.type.442b68.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.3c3: <witness> = custom_witness (%Int16, %Op.72717c.3), @MulWith.1, @MulWith.1(%Int16) [concrete]
// CHECK:STDOUT:   %facet_value.31a: %facet_type.2c7 = facet_value %i16, (%DestroyImpl.impl_witness.dff, %custom_witness.b80ef6.3, %custom_witness.3c3) [concrete]
// CHECK:STDOUT:   %.Self.frozen.8ef: %facet_type.eda = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %complete_type.57a: <witness> = complete_type_witness %facet_type.eda [concrete]
// CHECK:STDOUT:   %.Self.frozen.as_type.29d: type = facet_access_type %.Self.frozen.8ef [symbolic_self]
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.8f6: <witness> = lookup_impl_witness %.Self.frozen.8ef, @MulWith.1, @MulWith.1(%Int16) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.0af: type = impl_witness_access %MulWith.lookup_impl_witness.8f6, element0 [symbolic_self]
// CHECK:STDOUT:   %pattern_type.fcc: type = pattern_type %facet_type.2c7 [concrete]
// CHECK:STDOUT:   %T.patt.14a: %pattern_type.fcc = symbolic_binding_pattern T, 2 [symbolic]
// CHECK:STDOUT:   %facet_type.d6d: type = facet_type <@DestroyImpl & @SubobjectDestroy & @DivWith.1, @DivWith.1(%Int16)> [concrete]
// CHECK:STDOUT:   %.Self.1ea: %facet_type.d6d = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.5ff: <witness> = lookup_impl_witness %.Self.1ea, @DivWith.1, @DivWith.1(%Int16) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.416: type = impl_witness_access %DivWith.lookup_impl_witness.5ff, element0 [symbolic_self]
// CHECK:STDOUT:   %facet_type.403: type = facet_type <@DestroyImpl & @SubobjectDestroy & @DivWith.1, @DivWith.1(%Int16) where %impl.elem0.416 = %Int16> [concrete]
// CHECK:STDOUT:   %DivWith.type.b8a: type = facet_type <@DivWith.1, @DivWith.1(%Int16)> [concrete]
// CHECK:STDOUT:   %DivWith.assoc_type.712: type = assoc_entity_type @DivWith.1, @DivWith.1(%Int16) [concrete]
// CHECK:STDOUT:   %assoc0.99c: %DivWith.assoc_type.712 = assoc_entity element0, imports.%Core.import_ref.dee [concrete]
// CHECK:STDOUT:   %assoc1.fb6: %DivWith.assoc_type.712 = assoc_entity element1, imports.%Core.import_ref.f2f [concrete]
// CHECK:STDOUT:   %Op.type.442b68.4: type = fn_type @Op.4 [concrete]
// CHECK:STDOUT:   %Op.72717c.4: %Op.type.442b68.4 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.8d0: <witness> = custom_witness (%Int16, %Op.72717c.4), @DivWith.1, @DivWith.1(%Int16) [concrete]
// CHECK:STDOUT:   %facet_value.b4a: %facet_type.403 = facet_value %i16, (%DestroyImpl.impl_witness.dff, %custom_witness.b80ef6.3, %custom_witness.8d0) [concrete]
// CHECK:STDOUT:   %.Self.frozen.df7: %facet_type.d6d = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %complete_type.e4f: <witness> = complete_type_witness %facet_type.d6d [concrete]
// CHECK:STDOUT:   %.Self.frozen.as_type.cfb: type = facet_access_type %.Self.frozen.df7 [symbolic_self]
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.1bb: <witness> = lookup_impl_witness %.Self.frozen.df7, @DivWith.1, @DivWith.1(%Int16) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.729: type = impl_witness_access %DivWith.lookup_impl_witness.1bb, element0 [symbolic_self]
// CHECK:STDOUT:   %pattern_type.245b: type = pattern_type %facet_type.403 [concrete]
// CHECK:STDOUT:   %T.patt.415: %pattern_type.245b = symbolic_binding_pattern T, 2 [symbolic]
// CHECK:STDOUT:   %facet_type.8eb: type = facet_type <@DestroyImpl & @SubobjectDestroy & @ModWith.1, @ModWith.1(%Int16)> [concrete]
// CHECK:STDOUT:   %.Self.806: %facet_type.8eb = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.f9a: <witness> = lookup_impl_witness %.Self.806, @ModWith.1, @ModWith.1(%Int16) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.507: type = impl_witness_access %ModWith.lookup_impl_witness.f9a, element0 [symbolic_self]
// CHECK:STDOUT:   %facet_type.399: type = facet_type <@DestroyImpl & @SubobjectDestroy & @ModWith.1, @ModWith.1(%Int16) where %impl.elem0.507 = %Int16> [concrete]
// CHECK:STDOUT:   %ModWith.type.c67: type = facet_type <@ModWith.1, @ModWith.1(%Int16)> [concrete]
// CHECK:STDOUT:   %ModWith.assoc_type.5f6: type = assoc_entity_type @ModWith.1, @ModWith.1(%Int16) [concrete]
// CHECK:STDOUT:   %assoc0.f04: %ModWith.assoc_type.5f6 = assoc_entity element0, imports.%Core.import_ref.61a [concrete]
// CHECK:STDOUT:   %assoc1.d19: %ModWith.assoc_type.5f6 = assoc_entity element1, imports.%Core.import_ref.9f8f [concrete]
// CHECK:STDOUT:   %Op.type.442b68.5: type = fn_type @Op.5 [concrete]
// CHECK:STDOUT:   %Op.72717c.5: %Op.type.442b68.5 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.169: <witness> = custom_witness (%Int16, %Op.72717c.5), @ModWith.1, @ModWith.1(%Int16) [concrete]
// CHECK:STDOUT:   %facet_value.bff: %facet_type.399 = facet_value %i16, (%DestroyImpl.impl_witness.dff, %custom_witness.b80ef6.3, %custom_witness.169) [concrete]
// CHECK:STDOUT:   %.Self.frozen.fb4: %facet_type.8eb = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %complete_type.94f: <witness> = complete_type_witness %facet_type.8eb [concrete]
// CHECK:STDOUT:   %.Self.frozen.as_type.260: type = facet_access_type %.Self.frozen.fb4 [symbolic_self]
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.679: <witness> = lookup_impl_witness %.Self.frozen.fb4, @ModWith.1, @ModWith.1(%Int16) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.64e: type = impl_witness_access %ModWith.lookup_impl_witness.679, element0 [symbolic_self]
// CHECK:STDOUT:   %pattern_type.8c9: type = pattern_type %facet_type.399 [concrete]
// CHECK:STDOUT:   %T.patt.f92: %pattern_type.8c9 = symbolic_binding_pattern T, 2 [symbolic]
// CHECK:STDOUT:   %facet_type.b35: type = facet_type <@DestroyImpl & @SubobjectDestroy & @AddWith.1, @AddWith.1(%i16)> [concrete]
// CHECK:STDOUT:   %.Self.a3e: %facet_type.b35 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.05d: <witness> = lookup_impl_witness %.Self.a3e, @AddWith.1, @AddWith.1(%i16) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.01e: type = impl_witness_access %AddWith.lookup_impl_witness.05d, element0 [symbolic_self]
// CHECK:STDOUT:   %facet_type.676: type = facet_type <@DestroyImpl & @SubobjectDestroy & @AddWith.1, @AddWith.1(%i16) where %impl.elem0.01e = %Int16> [concrete]
// CHECK:STDOUT:   %AddWith.type.2ed: type = facet_type <@AddWith.1, @AddWith.1(%i16)> [concrete]
// CHECK:STDOUT:   %AddWith.assoc_type.18f: type = assoc_entity_type @AddWith.1, @AddWith.1(%i16) [concrete]
// CHECK:STDOUT:   %assoc0.b7d: %AddWith.assoc_type.18f = assoc_entity element0, imports.%Core.import_ref.91f [concrete]
// CHECK:STDOUT:   %assoc1.e08: %AddWith.assoc_type.18f = assoc_entity element1, imports.%Core.import_ref.8ad [concrete]
// CHECK:STDOUT:   %Int16.Op.type.83af6e.2: type = fn_type @Int16.Op.2 [concrete]
// CHECK:STDOUT:   %Int16.Op.4dca31.2: %Int16.Op.type.83af6e.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b79: <witness> = custom_witness (%Int16, %Int16.Op.4dca31.2), @AddWith.1, @AddWith.1(%i16) [concrete]
// CHECK:STDOUT:   %facet_value.39b: %facet_type.676 = facet_value %Int16, (%DestroyImpl.impl_witness.891, %custom_witness.b80ef6.1, %custom_witness.b79) [concrete]
// CHECK:STDOUT:   %.Self.frozen.aef: %facet_type.b35 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %complete_type.aee: <witness> = complete_type_witness %facet_type.b35 [concrete]
// CHECK:STDOUT:   %.Self.frozen.as_type.a48: type = facet_access_type %.Self.frozen.aef [symbolic_self]
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.5f1: <witness> = lookup_impl_witness %.Self.frozen.aef, @AddWith.1, @AddWith.1(%i16) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.877: type = impl_witness_access %AddWith.lookup_impl_witness.5f1, element0 [symbolic_self]
// CHECK:STDOUT:   %pattern_type.c34: type = pattern_type %facet_type.676 [concrete]
// CHECK:STDOUT:   %T.patt.715: %pattern_type.c34 = symbolic_binding_pattern T, 2 [symbolic]
// CHECK:STDOUT:   %x.param_patt.3b0: %pattern_type.785 = value_param_pattern [concrete]
// CHECK:STDOUT:   %x.patt.869: %pattern_type.785 = wrapper_binding_pattern x, %x.param_patt.3b0 [concrete]
// CHECK:STDOUT:   %y.param_patt.322: %pattern_type.9bc = value_param_pattern [concrete]
// CHECK:STDOUT:   %y.patt.a3a: %pattern_type.9bc = wrapper_binding_pattern y, %y.param_patt.322 [concrete]
// CHECK:STDOUT:   %facet_type.8ae: type = facet_type <@DestroyImpl & @SubobjectDestroy & @SubWith.1, @SubWith.1(%i16)> [concrete]
// CHECK:STDOUT:   %.Self.60a: %facet_type.8ae = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.71f: <witness> = lookup_impl_witness %.Self.60a, @SubWith.1, @SubWith.1(%i16) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.499: type = impl_witness_access %SubWith.lookup_impl_witness.71f, element0 [symbolic_self]
// CHECK:STDOUT:   %facet_type.f60: type = facet_type <@DestroyImpl & @SubobjectDestroy & @SubWith.1, @SubWith.1(%i16) where %impl.elem0.499 = %Int16> [concrete]
// CHECK:STDOUT:   %SubWith.type.606: type = facet_type <@SubWith.1, @SubWith.1(%i16)> [concrete]
// CHECK:STDOUT:   %SubWith.assoc_type.9c3: type = assoc_entity_type @SubWith.1, @SubWith.1(%i16) [concrete]
// CHECK:STDOUT:   %assoc0.f66: %SubWith.assoc_type.9c3 = assoc_entity element0, imports.%Core.import_ref.190 [concrete]
// CHECK:STDOUT:   %assoc1.144: %SubWith.assoc_type.9c3 = assoc_entity element1, imports.%Core.import_ref.2f4 [concrete]
// CHECK:STDOUT:   %Int16.Op.type.83af6e.3: type = fn_type @Int16.Op.3 [concrete]
// CHECK:STDOUT:   %Int16.Op.4dca31.3: %Int16.Op.type.83af6e.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.1e8: <witness> = custom_witness (%Int16, %Int16.Op.4dca31.3), @SubWith.1, @SubWith.1(%i16) [concrete]
// CHECK:STDOUT:   %facet_value.666: %facet_type.f60 = facet_value %Int16, (%DestroyImpl.impl_witness.891, %custom_witness.b80ef6.1, %custom_witness.1e8) [concrete]
// CHECK:STDOUT:   %.Self.frozen.d2d: %facet_type.8ae = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %complete_type.a80: <witness> = complete_type_witness %facet_type.8ae [concrete]
// CHECK:STDOUT:   %.Self.frozen.as_type.d5b: type = facet_access_type %.Self.frozen.d2d [symbolic_self]
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.19e: <witness> = lookup_impl_witness %.Self.frozen.d2d, @SubWith.1, @SubWith.1(%i16) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.275: type = impl_witness_access %SubWith.lookup_impl_witness.19e, element0 [symbolic_self]
// CHECK:STDOUT:   %pattern_type.2454: type = pattern_type %facet_type.f60 [concrete]
// CHECK:STDOUT:   %T.patt.e9d: %pattern_type.2454 = symbolic_binding_pattern T, 2 [symbolic]
// CHECK:STDOUT:   %facet_type.69d: type = facet_type <@DestroyImpl & @SubobjectDestroy & @MulWith.1, @MulWith.1(%i16)> [concrete]
// CHECK:STDOUT:   %.Self.0b6: %facet_type.69d = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.58f: <witness> = lookup_impl_witness %.Self.0b6, @MulWith.1, @MulWith.1(%i16) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.8b8: type = impl_witness_access %MulWith.lookup_impl_witness.58f, element0 [symbolic_self]
// CHECK:STDOUT:   %facet_type.63d: type = facet_type <@DestroyImpl & @SubobjectDestroy & @MulWith.1, @MulWith.1(%i16) where %impl.elem0.8b8 = %Int16> [concrete]
// CHECK:STDOUT:   %MulWith.type.1e2: type = facet_type <@MulWith.1, @MulWith.1(%i16)> [concrete]
// CHECK:STDOUT:   %MulWith.assoc_type.34c: type = assoc_entity_type @MulWith.1, @MulWith.1(%i16) [concrete]
// CHECK:STDOUT:   %assoc0.2d4: %MulWith.assoc_type.34c = assoc_entity element0, imports.%Core.import_ref.59c [concrete]
// CHECK:STDOUT:   %assoc1.635: %MulWith.assoc_type.34c = assoc_entity element1, imports.%Core.import_ref.fc2 [concrete]
// CHECK:STDOUT:   %Int16.Op.type.83af6e.4: type = fn_type @Int16.Op.4 [concrete]
// CHECK:STDOUT:   %Int16.Op.4dca31.4: %Int16.Op.type.83af6e.4 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.e2b: <witness> = custom_witness (%Int16, %Int16.Op.4dca31.4), @MulWith.1, @MulWith.1(%i16) [concrete]
// CHECK:STDOUT:   %facet_value.7b6: %facet_type.63d = facet_value %Int16, (%DestroyImpl.impl_witness.891, %custom_witness.b80ef6.1, %custom_witness.e2b) [concrete]
// CHECK:STDOUT:   %.Self.frozen.c18: %facet_type.69d = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %complete_type.d63: <witness> = complete_type_witness %facet_type.69d [concrete]
// CHECK:STDOUT:   %.Self.frozen.as_type.d64: type = facet_access_type %.Self.frozen.c18 [symbolic_self]
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.367: <witness> = lookup_impl_witness %.Self.frozen.c18, @MulWith.1, @MulWith.1(%i16) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.5f2: type = impl_witness_access %MulWith.lookup_impl_witness.367, element0 [symbolic_self]
// CHECK:STDOUT:   %pattern_type.563: type = pattern_type %facet_type.63d [concrete]
// CHECK:STDOUT:   %T.patt.b7e: %pattern_type.563 = symbolic_binding_pattern T, 2 [symbolic]
// CHECK:STDOUT:   %facet_type.e3e: type = facet_type <@DestroyImpl & @SubobjectDestroy & @DivWith.1, @DivWith.1(%i16)> [concrete]
// CHECK:STDOUT:   %.Self.d5c: %facet_type.e3e = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.510: <witness> = lookup_impl_witness %.Self.d5c, @DivWith.1, @DivWith.1(%i16) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.d69: type = impl_witness_access %DivWith.lookup_impl_witness.510, element0 [symbolic_self]
// CHECK:STDOUT:   %facet_type.38f: type = facet_type <@DestroyImpl & @SubobjectDestroy & @DivWith.1, @DivWith.1(%i16) where %impl.elem0.d69 = %Int16> [concrete]
// CHECK:STDOUT:   %DivWith.type.45d: type = facet_type <@DivWith.1, @DivWith.1(%i16)> [concrete]
// CHECK:STDOUT:   %DivWith.assoc_type.5a4: type = assoc_entity_type @DivWith.1, @DivWith.1(%i16) [concrete]
// CHECK:STDOUT:   %assoc0.156: %DivWith.assoc_type.5a4 = assoc_entity element0, imports.%Core.import_ref.dee [concrete]
// CHECK:STDOUT:   %assoc1.444: %DivWith.assoc_type.5a4 = assoc_entity element1, imports.%Core.import_ref.f2f [concrete]
// CHECK:STDOUT:   %Int16.Op.type.83af6e.5: type = fn_type @Int16.Op.5 [concrete]
// CHECK:STDOUT:   %Int16.Op.4dca31.5: %Int16.Op.type.83af6e.5 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.21db: <witness> = custom_witness (%Int16, %Int16.Op.4dca31.5), @DivWith.1, @DivWith.1(%i16) [concrete]
// CHECK:STDOUT:   %facet_value.410: %facet_type.38f = facet_value %Int16, (%DestroyImpl.impl_witness.891, %custom_witness.b80ef6.1, %custom_witness.21db) [concrete]
// CHECK:STDOUT:   %.Self.frozen.513: %facet_type.e3e = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %complete_type.97d: <witness> = complete_type_witness %facet_type.e3e [concrete]
// CHECK:STDOUT:   %.Self.frozen.as_type.9c4: type = facet_access_type %.Self.frozen.513 [symbolic_self]
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.3e1: <witness> = lookup_impl_witness %.Self.frozen.513, @DivWith.1, @DivWith.1(%i16) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.3d6: type = impl_witness_access %DivWith.lookup_impl_witness.3e1, element0 [symbolic_self]
// CHECK:STDOUT:   %pattern_type.5d8: type = pattern_type %facet_type.38f [concrete]
// CHECK:STDOUT:   %T.patt.50f: %pattern_type.5d8 = symbolic_binding_pattern T, 2 [symbolic]
// CHECK:STDOUT:   %facet_type.a81: type = facet_type <@DestroyImpl & @SubobjectDestroy & @ModWith.1, @ModWith.1(%i16)> [concrete]
// CHECK:STDOUT:   %.Self.860: %facet_type.a81 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.e90: <witness> = lookup_impl_witness %.Self.860, @ModWith.1, @ModWith.1(%i16) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.740: type = impl_witness_access %ModWith.lookup_impl_witness.e90, element0 [symbolic_self]
// CHECK:STDOUT:   %facet_type.283: type = facet_type <@DestroyImpl & @SubobjectDestroy & @ModWith.1, @ModWith.1(%i16) where %impl.elem0.740 = %Int16> [concrete]
// CHECK:STDOUT:   %ModWith.type.104: type = facet_type <@ModWith.1, @ModWith.1(%i16)> [concrete]
// CHECK:STDOUT:   %ModWith.assoc_type.0e9: type = assoc_entity_type @ModWith.1, @ModWith.1(%i16) [concrete]
// CHECK:STDOUT:   %assoc0.92c: %ModWith.assoc_type.0e9 = assoc_entity element0, imports.%Core.import_ref.61a [concrete]
// CHECK:STDOUT:   %assoc1.870: %ModWith.assoc_type.0e9 = assoc_entity element1, imports.%Core.import_ref.9f8f [concrete]
// CHECK:STDOUT:   %Int16.Op.type.83af6e.6: type = fn_type @Int16.Op.6 [concrete]
// CHECK:STDOUT:   %Int16.Op.4dca31.6: %Int16.Op.type.83af6e.6 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.6dd: <witness> = custom_witness (%Int16, %Int16.Op.4dca31.6), @ModWith.1, @ModWith.1(%i16) [concrete]
// CHECK:STDOUT:   %facet_value.47a: %facet_type.283 = facet_value %Int16, (%DestroyImpl.impl_witness.891, %custom_witness.b80ef6.1, %custom_witness.6dd) [concrete]
// CHECK:STDOUT:   %.Self.frozen.68c: %facet_type.a81 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %complete_type.cbe: <witness> = complete_type_witness %facet_type.a81 [concrete]
// CHECK:STDOUT:   %.Self.frozen.as_type.d41: type = facet_access_type %.Self.frozen.68c [symbolic_self]
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.d2e: <witness> = lookup_impl_witness %.Self.frozen.68c, @ModWith.1, @ModWith.1(%i16) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.6f4: type = impl_witness_access %ModWith.lookup_impl_witness.d2e, element0 [symbolic_self]
// CHECK:STDOUT:   %pattern_type.2a1: type = pattern_type %facet_type.283 [concrete]
// CHECK:STDOUT:   %T.patt.4c7: %pattern_type.2a1 = symbolic_binding_pattern T, 2 [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.4: type = fn_type @SubobjectDestroy.WithSelf.Op.loc190 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.4: %SubobjectDestroy.WithSelf.Op.type.6870fd.4 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.4: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.4), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.b43: %SubobjectDestroy.type = facet_value %Int32, (%custom_witness.b80ef6.4) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.716: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.b43) [concrete]
// CHECK:STDOUT:   %facet_value.cb7: %facet_type.ece = facet_value %Int32, (%DestroyImpl.impl_witness.716, %custom_witness.b80ef6.4) [concrete]
// CHECK:STDOUT:   %facet_type.f00: type = facet_type <@DestroyImpl & @SubobjectDestroy & @AddWith.1, @AddWith.1(%Int32)> [concrete]
// CHECK:STDOUT:   %.Self.29d: %facet_type.f00 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.c08: <witness> = lookup_impl_witness %.Self.29d, @AddWith.1, @AddWith.1(%Int32) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.49b: type = impl_witness_access %AddWith.lookup_impl_witness.c08, element0 [symbolic_self]
// CHECK:STDOUT:   %facet_type.8ac: type = facet_type <@DestroyImpl & @SubobjectDestroy & @AddWith.1, @AddWith.1(%Int32) where %impl.elem0.49b = %Int32> [concrete]
// CHECK:STDOUT:   %AddWith.type.db7: type = facet_type <@AddWith.1, @AddWith.1(%Int32)> [concrete]
// CHECK:STDOUT:   %AddWith.assoc_type.be8: type = assoc_entity_type @AddWith.1, @AddWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %assoc0.31c: %AddWith.assoc_type.be8 = assoc_entity element0, imports.%Core.import_ref.91f [concrete]
// CHECK:STDOUT:   %assoc1.86c: %AddWith.assoc_type.be8 = assoc_entity element1, imports.%Core.import_ref.8ad [concrete]
// CHECK:STDOUT:   %Int16.Op.type.83af6e.7: type = fn_type @Int16.Op.7 [concrete]
// CHECK:STDOUT:   %Int16.Op.4dca31.7: %Int16.Op.type.83af6e.7 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.513476.1: <witness> = custom_witness (%Int32, %Int16.Op.4dca31.7), @AddWith.1, @AddWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %facet_value.312: %facet_type.8ac = facet_value %Int16, (%DestroyImpl.impl_witness.891, %custom_witness.b80ef6.1, %custom_witness.513476.1) [concrete]
// CHECK:STDOUT:   %.Self.frozen.5ba: %facet_type.f00 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %complete_type.3c7: <witness> = complete_type_witness %facet_type.f00 [concrete]
// CHECK:STDOUT:   %.Self.frozen.as_type.cb7: type = facet_access_type %.Self.frozen.5ba [symbolic_self]
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.337: <witness> = lookup_impl_witness %.Self.frozen.5ba, @AddWith.1, @AddWith.1(%Int32) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.14f: type = impl_witness_access %AddWith.lookup_impl_witness.337, element0 [symbolic_self]
// CHECK:STDOUT:   %pattern_type.ff4: type = pattern_type %facet_type.8ac [concrete]
// CHECK:STDOUT:   %T.patt.fe3: %pattern_type.ff4 = symbolic_binding_pattern T, 2 [symbolic]
// CHECK:STDOUT:   %result.param_patt.e81: %pattern_type.8b4 = value_param_pattern [concrete]
// CHECK:STDOUT:   %result.patt.47f: %pattern_type.8b4 = wrapper_binding_pattern result, %result.param_patt.e81 [concrete]
// CHECK:STDOUT:   %y.param_patt.e0c: %pattern_type.8b4 = value_param_pattern [concrete]
// CHECK:STDOUT:   %y.patt.9e1: %pattern_type.8b4 = wrapper_binding_pattern y, %y.param_patt.e0c [concrete]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.861: <specific function> = specific_function %SelfDestruct, @SelfDestruct(%facet_value.cb7) [concrete]
// CHECK:STDOUT:   %facet_type.4e1: type = facet_type <@DestroyImpl & @SubobjectDestroy & @SubWith.1, @SubWith.1(%Int32)> [concrete]
// CHECK:STDOUT:   %.Self.763: %facet_type.4e1 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.08b: <witness> = lookup_impl_witness %.Self.763, @SubWith.1, @SubWith.1(%Int32) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.496: type = impl_witness_access %SubWith.lookup_impl_witness.08b, element0 [symbolic_self]
// CHECK:STDOUT:   %facet_type.541: type = facet_type <@DestroyImpl & @SubobjectDestroy & @SubWith.1, @SubWith.1(%Int32) where %impl.elem0.496 = %Int32> [concrete]
// CHECK:STDOUT:   %SubWith.type.120: type = facet_type <@SubWith.1, @SubWith.1(%Int32)> [concrete]
// CHECK:STDOUT:   %SubWith.assoc_type.dae: type = assoc_entity_type @SubWith.1, @SubWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %assoc0.fac: %SubWith.assoc_type.dae = assoc_entity element0, imports.%Core.import_ref.190 [concrete]
// CHECK:STDOUT:   %assoc1.147: %SubWith.assoc_type.dae = assoc_entity element1, imports.%Core.import_ref.2f4 [concrete]
// CHECK:STDOUT:   %Int16.Op.type.83af6e.8: type = fn_type @Int16.Op.8 [concrete]
// CHECK:STDOUT:   %Int16.Op.4dca31.8: %Int16.Op.type.83af6e.8 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.32cef1.1: <witness> = custom_witness (%Int32, %Int16.Op.4dca31.8), @SubWith.1, @SubWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %facet_value.1ab: %facet_type.541 = facet_value %Int16, (%DestroyImpl.impl_witness.891, %custom_witness.b80ef6.1, %custom_witness.32cef1.1) [concrete]
// CHECK:STDOUT:   %.Self.frozen.0f1: %facet_type.4e1 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %complete_type.ce7: <witness> = complete_type_witness %facet_type.4e1 [concrete]
// CHECK:STDOUT:   %.Self.frozen.as_type.815: type = facet_access_type %.Self.frozen.0f1 [symbolic_self]
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.5d6: <witness> = lookup_impl_witness %.Self.frozen.0f1, @SubWith.1, @SubWith.1(%Int32) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.1ff: type = impl_witness_access %SubWith.lookup_impl_witness.5d6, element0 [symbolic_self]
// CHECK:STDOUT:   %pattern_type.183: type = pattern_type %facet_type.541 [concrete]
// CHECK:STDOUT:   %T.patt.a86: %pattern_type.183 = symbolic_binding_pattern T, 2 [symbolic]
// CHECK:STDOUT:   %facet_type.89e: type = facet_type <@DestroyImpl & @SubobjectDestroy & @MulWith.1, @MulWith.1(%Int32)> [concrete]
// CHECK:STDOUT:   %.Self.4cb: %facet_type.89e = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.1b8: <witness> = lookup_impl_witness %.Self.4cb, @MulWith.1, @MulWith.1(%Int32) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.fa7: type = impl_witness_access %MulWith.lookup_impl_witness.1b8, element0 [symbolic_self]
// CHECK:STDOUT:   %facet_type.3b1: type = facet_type <@DestroyImpl & @SubobjectDestroy & @MulWith.1, @MulWith.1(%Int32) where %impl.elem0.fa7 = %Int32> [concrete]
// CHECK:STDOUT:   %MulWith.type.d67: type = facet_type <@MulWith.1, @MulWith.1(%Int32)> [concrete]
// CHECK:STDOUT:   %MulWith.assoc_type.ab1: type = assoc_entity_type @MulWith.1, @MulWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %assoc0.d0a: %MulWith.assoc_type.ab1 = assoc_entity element0, imports.%Core.import_ref.59c [concrete]
// CHECK:STDOUT:   %assoc1.349: %MulWith.assoc_type.ab1 = assoc_entity element1, imports.%Core.import_ref.fc2 [concrete]
// CHECK:STDOUT:   %Int16.Op.type.83af6e.9: type = fn_type @Int16.Op.9 [concrete]
// CHECK:STDOUT:   %Int16.Op.4dca31.9: %Int16.Op.type.83af6e.9 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.3a0675.1: <witness> = custom_witness (%Int32, %Int16.Op.4dca31.9), @MulWith.1, @MulWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %facet_value.56c: %facet_type.3b1 = facet_value %Int16, (%DestroyImpl.impl_witness.891, %custom_witness.b80ef6.1, %custom_witness.3a0675.1) [concrete]
// CHECK:STDOUT:   %.Self.frozen.709: %facet_type.89e = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %complete_type.a47: <witness> = complete_type_witness %facet_type.89e [concrete]
// CHECK:STDOUT:   %.Self.frozen.as_type.93b: type = facet_access_type %.Self.frozen.709 [symbolic_self]
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.5dc: <witness> = lookup_impl_witness %.Self.frozen.709, @MulWith.1, @MulWith.1(%Int32) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.a0b: type = impl_witness_access %MulWith.lookup_impl_witness.5dc, element0 [symbolic_self]
// CHECK:STDOUT:   %pattern_type.7e1: type = pattern_type %facet_type.3b1 [concrete]
// CHECK:STDOUT:   %T.patt.de6: %pattern_type.7e1 = symbolic_binding_pattern T, 2 [symbolic]
// CHECK:STDOUT:   %facet_type.2ab: type = facet_type <@DestroyImpl & @SubobjectDestroy & @DivWith.1, @DivWith.1(%Int32)> [concrete]
// CHECK:STDOUT:   %.Self.b0f: %facet_type.2ab = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.38e: <witness> = lookup_impl_witness %.Self.b0f, @DivWith.1, @DivWith.1(%Int32) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.183: type = impl_witness_access %DivWith.lookup_impl_witness.38e, element0 [symbolic_self]
// CHECK:STDOUT:   %facet_type.50c: type = facet_type <@DestroyImpl & @SubobjectDestroy & @DivWith.1, @DivWith.1(%Int32) where %impl.elem0.183 = %Int32> [concrete]
// CHECK:STDOUT:   %DivWith.type.5a1: type = facet_type <@DivWith.1, @DivWith.1(%Int32)> [concrete]
// CHECK:STDOUT:   %DivWith.assoc_type.22f: type = assoc_entity_type @DivWith.1, @DivWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %assoc0.a9d: %DivWith.assoc_type.22f = assoc_entity element0, imports.%Core.import_ref.dee [concrete]
// CHECK:STDOUT:   %assoc1.004: %DivWith.assoc_type.22f = assoc_entity element1, imports.%Core.import_ref.f2f [concrete]
// CHECK:STDOUT:   %Int16.Op.type.83af6e.10: type = fn_type @Int16.Op.10 [concrete]
// CHECK:STDOUT:   %Int16.Op.4dca31.10: %Int16.Op.type.83af6e.10 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.c32685.1: <witness> = custom_witness (%Int32, %Int16.Op.4dca31.10), @DivWith.1, @DivWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %facet_value.d5f: %facet_type.50c = facet_value %Int16, (%DestroyImpl.impl_witness.891, %custom_witness.b80ef6.1, %custom_witness.c32685.1) [concrete]
// CHECK:STDOUT:   %.Self.frozen.ee7: %facet_type.2ab = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %complete_type.f97: <witness> = complete_type_witness %facet_type.2ab [concrete]
// CHECK:STDOUT:   %.Self.frozen.as_type.7f5: type = facet_access_type %.Self.frozen.ee7 [symbolic_self]
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.b80: <witness> = lookup_impl_witness %.Self.frozen.ee7, @DivWith.1, @DivWith.1(%Int32) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.16df: type = impl_witness_access %DivWith.lookup_impl_witness.b80, element0 [symbolic_self]
// CHECK:STDOUT:   %pattern_type.df6: type = pattern_type %facet_type.50c [concrete]
// CHECK:STDOUT:   %T.patt.c47: %pattern_type.df6 = symbolic_binding_pattern T, 2 [symbolic]
// CHECK:STDOUT:   %facet_type.fc7: type = facet_type <@DestroyImpl & @SubobjectDestroy & @ModWith.1, @ModWith.1(%Int32)> [concrete]
// CHECK:STDOUT:   %.Self.797: %facet_type.fc7 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.126: <witness> = lookup_impl_witness %.Self.797, @ModWith.1, @ModWith.1(%Int32) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.6bf: type = impl_witness_access %ModWith.lookup_impl_witness.126, element0 [symbolic_self]
// CHECK:STDOUT:   %facet_type.54d: type = facet_type <@DestroyImpl & @SubobjectDestroy & @ModWith.1, @ModWith.1(%Int32) where %impl.elem0.6bf = %Int32> [concrete]
// CHECK:STDOUT:   %ModWith.type.794: type = facet_type <@ModWith.1, @ModWith.1(%Int32)> [concrete]
// CHECK:STDOUT:   %ModWith.assoc_type.021: type = assoc_entity_type @ModWith.1, @ModWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %assoc0.310: %ModWith.assoc_type.021 = assoc_entity element0, imports.%Core.import_ref.61a [concrete]
// CHECK:STDOUT:   %assoc1.2eb: %ModWith.assoc_type.021 = assoc_entity element1, imports.%Core.import_ref.9f8f [concrete]
// CHECK:STDOUT:   %Int16.Op.type.83af6e.11: type = fn_type @Int16.Op.11 [concrete]
// CHECK:STDOUT:   %Int16.Op.4dca31.11: %Int16.Op.type.83af6e.11 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.994c75.1: <witness> = custom_witness (%Int32, %Int16.Op.4dca31.11), @ModWith.1, @ModWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %facet_value.272: %facet_type.54d = facet_value %Int16, (%DestroyImpl.impl_witness.891, %custom_witness.b80ef6.1, %custom_witness.994c75.1) [concrete]
// CHECK:STDOUT:   %.Self.frozen.f41: %facet_type.fc7 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %complete_type.0f1: <witness> = complete_type_witness %facet_type.fc7 [concrete]
// CHECK:STDOUT:   %.Self.frozen.as_type.7dc: type = facet_access_type %.Self.frozen.f41 [symbolic_self]
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.f4e: <witness> = lookup_impl_witness %.Self.frozen.f41, @ModWith.1, @ModWith.1(%Int32) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.3c4: type = impl_witness_access %ModWith.lookup_impl_witness.f4e, element0 [symbolic_self]
// CHECK:STDOUT:   %pattern_type.dae: type = pattern_type %facet_type.54d [concrete]
// CHECK:STDOUT:   %T.patt.4d5: %pattern_type.dae = symbolic_binding_pattern T, 2 [symbolic]
// CHECK:STDOUT:   %facet_type.ead: type = facet_type <@DestroyImpl & @SubobjectDestroy & @AddWith.1, @AddWith.1(%Int16) where %impl.elem0.4ef = %Int32> [concrete]
// CHECK:STDOUT:   %Int32.Op.type.83af6e.1: type = fn_type @Int32.Op.1 [concrete]
// CHECK:STDOUT:   %Int32.Op.4dca31.1: %Int32.Op.type.83af6e.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.ea1: <witness> = custom_witness (%Int32, %Int32.Op.4dca31.1), @AddWith.1, @AddWith.1(%Int16) [concrete]
// CHECK:STDOUT:   %facet_value.145: %facet_type.ead = facet_value %Int32, (%DestroyImpl.impl_witness.716, %custom_witness.b80ef6.4, %custom_witness.ea1) [concrete]
// CHECK:STDOUT:   %pattern_type.f8e: type = pattern_type %facet_type.ead [concrete]
// CHECK:STDOUT:   %T.patt.f20: %pattern_type.f8e = symbolic_binding_pattern T, 2 [symbolic]
// CHECK:STDOUT:   %x.param_patt.27c: %pattern_type.8b4 = value_param_pattern [concrete]
// CHECK:STDOUT:   %x.patt.740: %pattern_type.8b4 = wrapper_binding_pattern x, %x.param_patt.27c [concrete]
// CHECK:STDOUT:   %facet_type.bfb: type = facet_type <@DestroyImpl & @SubobjectDestroy & @SubWith.1, @SubWith.1(%Int16) where %impl.elem0.cb3 = %Int32> [concrete]
// CHECK:STDOUT:   %Int32.Op.type.83af6e.2: type = fn_type @Int32.Op.2 [concrete]
// CHECK:STDOUT:   %Int32.Op.4dca31.2: %Int32.Op.type.83af6e.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.576: <witness> = custom_witness (%Int32, %Int32.Op.4dca31.2), @SubWith.1, @SubWith.1(%Int16) [concrete]
// CHECK:STDOUT:   %facet_value.645: %facet_type.bfb = facet_value %Int32, (%DestroyImpl.impl_witness.716, %custom_witness.b80ef6.4, %custom_witness.576) [concrete]
// CHECK:STDOUT:   %pattern_type.edc: type = pattern_type %facet_type.bfb [concrete]
// CHECK:STDOUT:   %T.patt.07b: %pattern_type.edc = symbolic_binding_pattern T, 2 [symbolic]
// CHECK:STDOUT:   %facet_type.d1d: type = facet_type <@DestroyImpl & @SubobjectDestroy & @MulWith.1, @MulWith.1(%Int16) where %impl.elem0.567 = %Int32> [concrete]
// CHECK:STDOUT:   %Int32.Op.type.83af6e.3: type = fn_type @Int32.Op.3 [concrete]
// CHECK:STDOUT:   %Int32.Op.4dca31.3: %Int32.Op.type.83af6e.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.751: <witness> = custom_witness (%Int32, %Int32.Op.4dca31.3), @MulWith.1, @MulWith.1(%Int16) [concrete]
// CHECK:STDOUT:   %facet_value.a15: %facet_type.d1d = facet_value %Int32, (%DestroyImpl.impl_witness.716, %custom_witness.b80ef6.4, %custom_witness.751) [concrete]
// CHECK:STDOUT:   %pattern_type.f82: type = pattern_type %facet_type.d1d [concrete]
// CHECK:STDOUT:   %T.patt.983: %pattern_type.f82 = symbolic_binding_pattern T, 2 [symbolic]
// CHECK:STDOUT:   %facet_type.835: type = facet_type <@DestroyImpl & @SubobjectDestroy & @DivWith.1, @DivWith.1(%Int16) where %impl.elem0.416 = %Int32> [concrete]
// CHECK:STDOUT:   %Int32.Op.type.83af6e.4: type = fn_type @Int32.Op.4 [concrete]
// CHECK:STDOUT:   %Int32.Op.4dca31.4: %Int32.Op.type.83af6e.4 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.9ad: <witness> = custom_witness (%Int32, %Int32.Op.4dca31.4), @DivWith.1, @DivWith.1(%Int16) [concrete]
// CHECK:STDOUT:   %facet_value.d17: %facet_type.835 = facet_value %Int32, (%DestroyImpl.impl_witness.716, %custom_witness.b80ef6.4, %custom_witness.9ad) [concrete]
// CHECK:STDOUT:   %pattern_type.5a9: type = pattern_type %facet_type.835 [concrete]
// CHECK:STDOUT:   %T.patt.7bf: %pattern_type.5a9 = symbolic_binding_pattern T, 2 [symbolic]
// CHECK:STDOUT:   %facet_type.9e1: type = facet_type <@DestroyImpl & @SubobjectDestroy & @ModWith.1, @ModWith.1(%Int16) where %impl.elem0.507 = %Int32> [concrete]
// CHECK:STDOUT:   %Int32.Op.type.83af6e.5: type = fn_type @Int32.Op.5 [concrete]
// CHECK:STDOUT:   %Int32.Op.4dca31.5: %Int32.Op.type.83af6e.5 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.8a2: <witness> = custom_witness (%Int32, %Int32.Op.4dca31.5), @ModWith.1, @ModWith.1(%Int16) [concrete]
// CHECK:STDOUT:   %facet_value.a16: %facet_type.9e1 = facet_value %Int32, (%DestroyImpl.impl_witness.716, %custom_witness.b80ef6.4, %custom_witness.8a2) [concrete]
// CHECK:STDOUT:   %pattern_type.f91: type = pattern_type %facet_type.9e1 [concrete]
// CHECK:STDOUT:   %T.patt.e03: %pattern_type.f91 = symbolic_binding_pattern T, 2 [symbolic]
// CHECK:STDOUT:   %Int32.Op.type.83af6e.6: type = fn_type @Int32.Op.6 [concrete]
// CHECK:STDOUT:   %Int32.Op.4dca31.6: %Int32.Op.type.83af6e.6 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.513476.2: <witness> = custom_witness (%Int32, %Int32.Op.4dca31.6), @AddWith.1, @AddWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %facet_value.c03: %facet_type.8ac = facet_value %Int32, (%DestroyImpl.impl_witness.716, %custom_witness.b80ef6.4, %custom_witness.513476.2) [concrete]
// CHECK:STDOUT:   %Int32.Op.type.83af6e.7: type = fn_type @Int32.Op.7 [concrete]
// CHECK:STDOUT:   %Int32.Op.4dca31.7: %Int32.Op.type.83af6e.7 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.32cef1.2: <witness> = custom_witness (%Int32, %Int32.Op.4dca31.7), @SubWith.1, @SubWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %facet_value.789: %facet_type.541 = facet_value %Int32, (%DestroyImpl.impl_witness.716, %custom_witness.b80ef6.4, %custom_witness.32cef1.2) [concrete]
// CHECK:STDOUT:   %Int32.Op.type.83af6e.8: type = fn_type @Int32.Op.8 [concrete]
// CHECK:STDOUT:   %Int32.Op.4dca31.8: %Int32.Op.type.83af6e.8 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.3a0675.2: <witness> = custom_witness (%Int32, %Int32.Op.4dca31.8), @MulWith.1, @MulWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %facet_value.9fe: %facet_type.3b1 = facet_value %Int32, (%DestroyImpl.impl_witness.716, %custom_witness.b80ef6.4, %custom_witness.3a0675.2) [concrete]
// CHECK:STDOUT:   %Int32.Op.type.83af6e.9: type = fn_type @Int32.Op.9 [concrete]
// CHECK:STDOUT:   %Int32.Op.4dca31.9: %Int32.Op.type.83af6e.9 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.c32685.2: <witness> = custom_witness (%Int32, %Int32.Op.4dca31.9), @DivWith.1, @DivWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %facet_value.079: %facet_type.50c = facet_value %Int32, (%DestroyImpl.impl_witness.716, %custom_witness.b80ef6.4, %custom_witness.c32685.2) [concrete]
// CHECK:STDOUT:   %Int32.Op.type.83af6e.10: type = fn_type @Int32.Op.10 [concrete]
// CHECK:STDOUT:   %Int32.Op.4dca31.10: %Int32.Op.type.83af6e.10 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.994c75.2: <witness> = custom_witness (%Int32, %Int32.Op.4dca31.10), @ModWith.1, @ModWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %facet_value.72c: %facet_type.54d = facet_value %Int32, (%DestroyImpl.impl_witness.716, %custom_witness.b80ef6.4, %custom_witness.994c75.2) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.5: type = fn_type @SubobjectDestroy.WithSelf.Op.loc211 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.5: %SubobjectDestroy.WithSelf.Op.type.6870fd.5 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.5: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.5), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.c36: %SubobjectDestroy.type = facet_value %Int64, (%custom_witness.b80ef6.5) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.67f: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.c36) [concrete]
// CHECK:STDOUT:   %facet_value.ca8: %facet_type.ece = facet_value %Int64, (%DestroyImpl.impl_witness.67f, %custom_witness.b80ef6.5) [concrete]
// CHECK:STDOUT:   %facet_type.884: type = facet_type <@DestroyImpl & @SubobjectDestroy & @AddWith.1, @AddWith.1(%Int64)> [concrete]
// CHECK:STDOUT:   %.Self.f99: %facet_type.884 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.6d7: <witness> = lookup_impl_witness %.Self.f99, @AddWith.1, @AddWith.1(%Int64) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.e6e: type = impl_witness_access %AddWith.lookup_impl_witness.6d7, element0 [symbolic_self]
// CHECK:STDOUT:   %facet_type.226: type = facet_type <@DestroyImpl & @SubobjectDestroy & @AddWith.1, @AddWith.1(%Int64) where %impl.elem0.e6e = %Int64> [concrete]
// CHECK:STDOUT:   %y.param_patt.a08: %pattern_type.4aa = value_param_pattern [concrete]
// CHECK:STDOUT:   %y.patt.069: %pattern_type.4aa = wrapper_binding_pattern y, %y.param_patt.a08 [concrete]
// CHECK:STDOUT:   %AddWith.type.1ff: type = facet_type <@AddWith.1, @AddWith.1(%Int64)> [concrete]
// CHECK:STDOUT:   %AddWith.assoc_type.97f: type = assoc_entity_type @AddWith.1, @AddWith.1(%Int64) [concrete]
// CHECK:STDOUT:   %assoc0.f02: %AddWith.assoc_type.97f = assoc_entity element0, imports.%Core.import_ref.91f [concrete]
// CHECK:STDOUT:   %assoc1.a75: %AddWith.assoc_type.97f = assoc_entity element1, imports.%Core.import_ref.8ad [concrete]
// CHECK:STDOUT:   %Op.type.442b68.6: type = fn_type @Op.6 [concrete]
// CHECK:STDOUT:   %Op.72717c.6: %Op.type.442b68.6 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.f2f: <witness> = custom_witness (%Int64, %Op.72717c.6), @AddWith.1, @AddWith.1(%Int64) [concrete]
// CHECK:STDOUT:   %facet_value.797: %facet_type.226 = facet_value %Int32, (%DestroyImpl.impl_witness.716, %custom_witness.b80ef6.4, %custom_witness.f2f) [concrete]
// CHECK:STDOUT:   %.Self.frozen.266: %facet_type.884 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %complete_type.2dc: <witness> = complete_type_witness %facet_type.884 [concrete]
// CHECK:STDOUT:   %.Self.frozen.as_type.021: type = facet_access_type %.Self.frozen.266 [symbolic_self]
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.a15: <witness> = lookup_impl_witness %.Self.frozen.266, @AddWith.1, @AddWith.1(%Int64) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.f79: type = impl_witness_access %AddWith.lookup_impl_witness.a15, element0 [symbolic_self]
// CHECK:STDOUT:   %pattern_type.a52: type = pattern_type %facet_type.226 [concrete]
// CHECK:STDOUT:   %T.patt.e68: %pattern_type.a52 = symbolic_binding_pattern T, 2 [symbolic]
// CHECK:STDOUT:   %result.param_patt.1e1: %pattern_type.4aa = value_param_pattern [concrete]
// CHECK:STDOUT:   %result.patt.93b: %pattern_type.4aa = wrapper_binding_pattern result, %result.param_patt.1e1 [concrete]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.d50: <specific function> = specific_function %SelfDestruct, @SelfDestruct(%facet_value.ca8) [concrete]
// CHECK:STDOUT:   %facet_type.511: type = facet_type <@DestroyImpl & @SubobjectDestroy & @SubWith.1, @SubWith.1(%Int64)> [concrete]
// CHECK:STDOUT:   %.Self.f66: %facet_type.511 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.05e: <witness> = lookup_impl_witness %.Self.f66, @SubWith.1, @SubWith.1(%Int64) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.bc3: type = impl_witness_access %SubWith.lookup_impl_witness.05e, element0 [symbolic_self]
// CHECK:STDOUT:   %facet_type.ad6: type = facet_type <@DestroyImpl & @SubobjectDestroy & @SubWith.1, @SubWith.1(%Int64) where %impl.elem0.bc3 = %Int64> [concrete]
// CHECK:STDOUT:   %SubWith.type.31c: type = facet_type <@SubWith.1, @SubWith.1(%Int64)> [concrete]
// CHECK:STDOUT:   %SubWith.assoc_type.4f8: type = assoc_entity_type @SubWith.1, @SubWith.1(%Int64) [concrete]
// CHECK:STDOUT:   %assoc0.13b: %SubWith.assoc_type.4f8 = assoc_entity element0, imports.%Core.import_ref.190 [concrete]
// CHECK:STDOUT:   %assoc1.e70: %SubWith.assoc_type.4f8 = assoc_entity element1, imports.%Core.import_ref.2f4 [concrete]
// CHECK:STDOUT:   %Op.type.442b68.7: type = fn_type @Op.7 [concrete]
// CHECK:STDOUT:   %Op.72717c.7: %Op.type.442b68.7 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.276: <witness> = custom_witness (%Int64, %Op.72717c.7), @SubWith.1, @SubWith.1(%Int64) [concrete]
// CHECK:STDOUT:   %facet_value.e83: %facet_type.ad6 = facet_value %Int32, (%DestroyImpl.impl_witness.716, %custom_witness.b80ef6.4, %custom_witness.276) [concrete]
// CHECK:STDOUT:   %.Self.frozen.0f4: %facet_type.511 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %complete_type.5d9: <witness> = complete_type_witness %facet_type.511 [concrete]
// CHECK:STDOUT:   %.Self.frozen.as_type.a25: type = facet_access_type %.Self.frozen.0f4 [symbolic_self]
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.266: <witness> = lookup_impl_witness %.Self.frozen.0f4, @SubWith.1, @SubWith.1(%Int64) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.097: type = impl_witness_access %SubWith.lookup_impl_witness.266, element0 [symbolic_self]
// CHECK:STDOUT:   %pattern_type.647: type = pattern_type %facet_type.ad6 [concrete]
// CHECK:STDOUT:   %T.patt.b82: %pattern_type.647 = symbolic_binding_pattern T, 2 [symbolic]
// CHECK:STDOUT:   %facet_type.58d: type = facet_type <@DestroyImpl & @SubobjectDestroy & @MulWith.1, @MulWith.1(%Int64)> [concrete]
// CHECK:STDOUT:   %.Self.f7f: %facet_type.58d = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.53a: <witness> = lookup_impl_witness %.Self.f7f, @MulWith.1, @MulWith.1(%Int64) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.5d5: type = impl_witness_access %MulWith.lookup_impl_witness.53a, element0 [symbolic_self]
// CHECK:STDOUT:   %facet_type.f3f: type = facet_type <@DestroyImpl & @SubobjectDestroy & @MulWith.1, @MulWith.1(%Int64) where %impl.elem0.5d5 = %Int64> [concrete]
// CHECK:STDOUT:   %MulWith.type.661: type = facet_type <@MulWith.1, @MulWith.1(%Int64)> [concrete]
// CHECK:STDOUT:   %MulWith.assoc_type.940: type = assoc_entity_type @MulWith.1, @MulWith.1(%Int64) [concrete]
// CHECK:STDOUT:   %assoc0.9d9: %MulWith.assoc_type.940 = assoc_entity element0, imports.%Core.import_ref.59c [concrete]
// CHECK:STDOUT:   %assoc1.b22: %MulWith.assoc_type.940 = assoc_entity element1, imports.%Core.import_ref.fc2 [concrete]
// CHECK:STDOUT:   %Op.type.442b68.8: type = fn_type @Op.8 [concrete]
// CHECK:STDOUT:   %Op.72717c.8: %Op.type.442b68.8 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.8e4: <witness> = custom_witness (%Int64, %Op.72717c.8), @MulWith.1, @MulWith.1(%Int64) [concrete]
// CHECK:STDOUT:   %facet_value.055: %facet_type.f3f = facet_value %Int32, (%DestroyImpl.impl_witness.716, %custom_witness.b80ef6.4, %custom_witness.8e4) [concrete]
// CHECK:STDOUT:   %.Self.frozen.edb: %facet_type.58d = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %complete_type.c4e: <witness> = complete_type_witness %facet_type.58d [concrete]
// CHECK:STDOUT:   %.Self.frozen.as_type.63f: type = facet_access_type %.Self.frozen.edb [symbolic_self]
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.ddf: <witness> = lookup_impl_witness %.Self.frozen.edb, @MulWith.1, @MulWith.1(%Int64) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.1e2: type = impl_witness_access %MulWith.lookup_impl_witness.ddf, element0 [symbolic_self]
// CHECK:STDOUT:   %pattern_type.761: type = pattern_type %facet_type.f3f [concrete]
// CHECK:STDOUT:   %T.patt.460: %pattern_type.761 = symbolic_binding_pattern T, 2 [symbolic]
// CHECK:STDOUT:   %facet_type.6e2: type = facet_type <@DestroyImpl & @SubobjectDestroy & @DivWith.1, @DivWith.1(%Int64)> [concrete]
// CHECK:STDOUT:   %.Self.92e: %facet_type.6e2 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.1c8: <witness> = lookup_impl_witness %.Self.92e, @DivWith.1, @DivWith.1(%Int64) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.39a: type = impl_witness_access %DivWith.lookup_impl_witness.1c8, element0 [symbolic_self]
// CHECK:STDOUT:   %facet_type.708: type = facet_type <@DestroyImpl & @SubobjectDestroy & @DivWith.1, @DivWith.1(%Int64) where %impl.elem0.39a = %Int64> [concrete]
// CHECK:STDOUT:   %DivWith.type.dd8: type = facet_type <@DivWith.1, @DivWith.1(%Int64)> [concrete]
// CHECK:STDOUT:   %DivWith.assoc_type.2dc: type = assoc_entity_type @DivWith.1, @DivWith.1(%Int64) [concrete]
// CHECK:STDOUT:   %assoc0.e8b: %DivWith.assoc_type.2dc = assoc_entity element0, imports.%Core.import_ref.dee [concrete]
// CHECK:STDOUT:   %assoc1.8b9: %DivWith.assoc_type.2dc = assoc_entity element1, imports.%Core.import_ref.f2f [concrete]
// CHECK:STDOUT:   %Op.type.442b68.9: type = fn_type @Op.9 [concrete]
// CHECK:STDOUT:   %Op.72717c.9: %Op.type.442b68.9 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.099: <witness> = custom_witness (%Int64, %Op.72717c.9), @DivWith.1, @DivWith.1(%Int64) [concrete]
// CHECK:STDOUT:   %facet_value.260: %facet_type.708 = facet_value %Int32, (%DestroyImpl.impl_witness.716, %custom_witness.b80ef6.4, %custom_witness.099) [concrete]
// CHECK:STDOUT:   %.Self.frozen.a2d: %facet_type.6e2 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %complete_type.f3e: <witness> = complete_type_witness %facet_type.6e2 [concrete]
// CHECK:STDOUT:   %.Self.frozen.as_type.342: type = facet_access_type %.Self.frozen.a2d [symbolic_self]
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.b6a: <witness> = lookup_impl_witness %.Self.frozen.a2d, @DivWith.1, @DivWith.1(%Int64) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.935: type = impl_witness_access %DivWith.lookup_impl_witness.b6a, element0 [symbolic_self]
// CHECK:STDOUT:   %pattern_type.f93: type = pattern_type %facet_type.708 [concrete]
// CHECK:STDOUT:   %T.patt.0cc: %pattern_type.f93 = symbolic_binding_pattern T, 2 [symbolic]
// CHECK:STDOUT:   %facet_type.a05: type = facet_type <@DestroyImpl & @SubobjectDestroy & @ModWith.1, @ModWith.1(%Int64)> [concrete]
// CHECK:STDOUT:   %.Self.445: %facet_type.a05 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.ee1: <witness> = lookup_impl_witness %.Self.445, @ModWith.1, @ModWith.1(%Int64) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.1ca: type = impl_witness_access %ModWith.lookup_impl_witness.ee1, element0 [symbolic_self]
// CHECK:STDOUT:   %facet_type.e35: type = facet_type <@DestroyImpl & @SubobjectDestroy & @ModWith.1, @ModWith.1(%Int64) where %impl.elem0.1ca = %Int64> [concrete]
// CHECK:STDOUT:   %ModWith.type.c72: type = facet_type <@ModWith.1, @ModWith.1(%Int64)> [concrete]
// CHECK:STDOUT:   %ModWith.assoc_type.ad2: type = assoc_entity_type @ModWith.1, @ModWith.1(%Int64) [concrete]
// CHECK:STDOUT:   %assoc0.c8a: %ModWith.assoc_type.ad2 = assoc_entity element0, imports.%Core.import_ref.61a [concrete]
// CHECK:STDOUT:   %assoc1.eed: %ModWith.assoc_type.ad2 = assoc_entity element1, imports.%Core.import_ref.9f8f [concrete]
// CHECK:STDOUT:   %Op.type.442b68.10: type = fn_type @Op.10 [concrete]
// CHECK:STDOUT:   %Op.72717c.10: %Op.type.442b68.10 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.129: <witness> = custom_witness (%Int64, %Op.72717c.10), @ModWith.1, @ModWith.1(%Int64) [concrete]
// CHECK:STDOUT:   %facet_value.f07: %facet_type.e35 = facet_value %Int32, (%DestroyImpl.impl_witness.716, %custom_witness.b80ef6.4, %custom_witness.129) [concrete]
// CHECK:STDOUT:   %.Self.frozen.d87: %facet_type.a05 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %complete_type.93c: <witness> = complete_type_witness %facet_type.a05 [concrete]
// CHECK:STDOUT:   %.Self.frozen.as_type.c60: type = facet_access_type %.Self.frozen.d87 [symbolic_self]
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.174: <witness> = lookup_impl_witness %.Self.frozen.d87, @ModWith.1, @ModWith.1(%Int64) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.020: type = impl_witness_access %ModWith.lookup_impl_witness.174, element0 [symbolic_self]
// CHECK:STDOUT:   %pattern_type.1e1: type = pattern_type %facet_type.e35 [concrete]
// CHECK:STDOUT:   %T.patt.07e: %pattern_type.1e1 = symbolic_binding_pattern T, 2 [symbolic]
// CHECK:STDOUT:   %facet_type.ec8: type = facet_type <@DestroyImpl & @SubobjectDestroy & @AddWith.1, @AddWith.1(%Int32) where %impl.elem0.49b = %Int64> [concrete]
// CHECK:STDOUT:   %Op.type.442b68.11: type = fn_type @Op.11 [concrete]
// CHECK:STDOUT:   %Op.72717c.11: %Op.type.442b68.11 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.16b: <witness> = custom_witness (%Int64, %Op.72717c.11), @AddWith.1, @AddWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %facet_value.252: %facet_type.ec8 = facet_value %Int64, (%DestroyImpl.impl_witness.67f, %custom_witness.b80ef6.5, %custom_witness.16b) [concrete]
// CHECK:STDOUT:   %pattern_type.d1c: type = pattern_type %facet_type.ec8 [concrete]
// CHECK:STDOUT:   %T.patt.e18: %pattern_type.d1c = symbolic_binding_pattern T, 2 [symbolic]
// CHECK:STDOUT:   %x.param_patt.e68: %pattern_type.4aa = value_param_pattern [concrete]
// CHECK:STDOUT:   %x.patt.7aa: %pattern_type.4aa = wrapper_binding_pattern x, %x.param_patt.e68 [concrete]
// CHECK:STDOUT:   %facet_type.1f3: type = facet_type <@DestroyImpl & @SubobjectDestroy & @SubWith.1, @SubWith.1(%Int32) where %impl.elem0.496 = %Int64> [concrete]
// CHECK:STDOUT:   %Op.type.442b68.12: type = fn_type @Op.12 [concrete]
// CHECK:STDOUT:   %Op.72717c.12: %Op.type.442b68.12 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.21a: <witness> = custom_witness (%Int64, %Op.72717c.12), @SubWith.1, @SubWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %facet_value.cfe: %facet_type.1f3 = facet_value %Int64, (%DestroyImpl.impl_witness.67f, %custom_witness.b80ef6.5, %custom_witness.21a) [concrete]
// CHECK:STDOUT:   %pattern_type.e00: type = pattern_type %facet_type.1f3 [concrete]
// CHECK:STDOUT:   %T.patt.0e9: %pattern_type.e00 = symbolic_binding_pattern T, 2 [symbolic]
// CHECK:STDOUT:   %facet_type.9ad: type = facet_type <@DestroyImpl & @SubobjectDestroy & @MulWith.1, @MulWith.1(%Int32) where %impl.elem0.fa7 = %Int64> [concrete]
// CHECK:STDOUT:   %Op.type.442b68.13: type = fn_type @Op.13 [concrete]
// CHECK:STDOUT:   %Op.72717c.13: %Op.type.442b68.13 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.798: <witness> = custom_witness (%Int64, %Op.72717c.13), @MulWith.1, @MulWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %facet_value.f28: %facet_type.9ad = facet_value %Int64, (%DestroyImpl.impl_witness.67f, %custom_witness.b80ef6.5, %custom_witness.798) [concrete]
// CHECK:STDOUT:   %pattern_type.47d: type = pattern_type %facet_type.9ad [concrete]
// CHECK:STDOUT:   %T.patt.24f: %pattern_type.47d = symbolic_binding_pattern T, 2 [symbolic]
// CHECK:STDOUT:   %facet_type.c2a: type = facet_type <@DestroyImpl & @SubobjectDestroy & @DivWith.1, @DivWith.1(%Int32) where %impl.elem0.183 = %Int64> [concrete]
// CHECK:STDOUT:   %Op.type.442b68.14: type = fn_type @Op.14 [concrete]
// CHECK:STDOUT:   %Op.72717c.14: %Op.type.442b68.14 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.37f: <witness> = custom_witness (%Int64, %Op.72717c.14), @DivWith.1, @DivWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %facet_value.725: %facet_type.c2a = facet_value %Int64, (%DestroyImpl.impl_witness.67f, %custom_witness.b80ef6.5, %custom_witness.37f) [concrete]
// CHECK:STDOUT:   %pattern_type.8cc: type = pattern_type %facet_type.c2a [concrete]
// CHECK:STDOUT:   %T.patt.790: %pattern_type.8cc = symbolic_binding_pattern T, 2 [symbolic]
// CHECK:STDOUT:   %facet_type.26a: type = facet_type <@DestroyImpl & @SubobjectDestroy & @ModWith.1, @ModWith.1(%Int32) where %impl.elem0.6bf = %Int64> [concrete]
// CHECK:STDOUT:   %Op.type.442b68.15: type = fn_type @Op.15 [concrete]
// CHECK:STDOUT:   %Op.72717c.15: %Op.type.442b68.15 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.981: <witness> = custom_witness (%Int64, %Op.72717c.15), @ModWith.1, @ModWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %facet_value.f68: %facet_type.26a = facet_value %Int64, (%DestroyImpl.impl_witness.67f, %custom_witness.b80ef6.5, %custom_witness.981) [concrete]
// CHECK:STDOUT:   %pattern_type.5b0: type = pattern_type %facet_type.26a [concrete]
// CHECK:STDOUT:   %T.patt.730: %pattern_type.5b0 = symbolic_binding_pattern T, 2 [symbolic]
// CHECK:STDOUT:   %U.patt.3a0: %pattern_type.9a5 = symbolic_binding_pattern U, 0 [symbolic]
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
// CHECK:STDOUT:   %facet_type.559: type = facet_type <@DestroyImpl & @SubobjectDestroy & @Negate> [concrete]
// CHECK:STDOUT:   %.Self.e76: %facet_type.559 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %Negate.lookup_impl_witness.cf4: <witness> = lookup_impl_witness %.Self.e76, @Negate [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.c9c: type = impl_witness_access %Negate.lookup_impl_witness.cf4, element0 [symbolic_self]
// CHECK:STDOUT:   %facet_type.5ed: type = facet_type <@DestroyImpl & @SubobjectDestroy & @Negate where %impl.elem0.c9c = %Result.as_type> [symbolic]
// CHECK:STDOUT:   %pattern_type.6d4: type = pattern_type %facet_type.5ed [symbolic]
// CHECK:STDOUT:   %T.patt.7ae: %pattern_type.6d4 = symbolic_binding_pattern T, 1 [symbolic]
// CHECK:STDOUT:   %T.810: %facet_type.5ed = symbolic_binding T, 1 [symbolic]
// CHECK:STDOUT:   %y.param_patt.068: %pattern_type.e4c132.1 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %y.patt.e33: %pattern_type.e4c132.1 = wrapper_binding_pattern y, %y.param_patt.068 [symbolic]
// CHECK:STDOUT:   %T.as_type.a42: type = facet_access_type %T.810 [symbolic]
// CHECK:STDOUT:   %pattern_type.e29: type = pattern_type %T.as_type.a42 [symbolic]
// CHECK:STDOUT:   %x.param_patt.7c9: %pattern_type.e29 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %x.patt.efa: %pattern_type.e29 = wrapper_binding_pattern x, %x.param_patt.7c9 [symbolic]
// CHECK:STDOUT:   %Negate.lookup_impl_witness.eaf: <witness> = lookup_impl_witness %T.810, @Negate [symbolic]
// CHECK:STDOUT:   %Negate.facet.473: %Negate.type = facet_value %T.as_type.a42, (%Negate.lookup_impl_witness.eaf) [symbolic]
// CHECK:STDOUT:   %Negate.WithSelf.Op.type.96d: type = fn_type @Negate.WithSelf.Op, @Negate.WithSelf(%Negate.facet.473) [symbolic]
// CHECK:STDOUT:   %.1b4: type = fn_type_with_self_type %Negate.WithSelf.Op.type.96d, %Negate.facet.473 [symbolic]
// CHECK:STDOUT:   %impl.elem1.322: %.1b4 = impl_witness_access %Negate.lookup_impl_witness.eaf, element1 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.511: <specific function> = specific_impl_function %impl.elem1.322, @Negate.WithSelf.Op(%Negate.facet.473) [symbolic]
// CHECK:STDOUT:   %Int32.Op.type.83af6e.17: type = fn_type @Int32.Op.17 [concrete]
// CHECK:STDOUT:   %Int32.Op.4dca31.17: %Int32.Op.type.83af6e.17 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.f3f8: <witness> = custom_witness (%Int32.Op.4dca31.17), @Inc [concrete]
// CHECK:STDOUT:   %Inc.facet.dd2: %Inc.type = facet_value %Int32, (%custom_witness.f3f8) [concrete]
// CHECK:STDOUT:   %Int32.Op.type.83af6e.18: type = fn_type @Int32.Op.18 [concrete]
// CHECK:STDOUT:   %Int32.Op.4dca31.18: %Int32.Op.type.83af6e.18 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b24: <witness> = custom_witness (%Int32.Op.4dca31.18), @Dec [concrete]
// CHECK:STDOUT:   %Dec.facet.e1b: %Dec.type = facet_value %Int32, (%custom_witness.b24) [concrete]
// CHECK:STDOUT:   %facet_type.b1a: type = facet_type <@DestroyImpl & @SubobjectDestroy & @Negate where %impl.elem0.c9c = %Int64> [concrete]
// CHECK:STDOUT:   %Int16.Op.type.83af6e.17: type = fn_type @Int16.Op.17 [concrete]
// CHECK:STDOUT:   %Int16.Op.4dca31.17: %Int16.Op.type.83af6e.17 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.100: <witness> = custom_witness (%Int64, %Int16.Op.4dca31.17), @Negate [concrete]
// CHECK:STDOUT:   %facet_value.901: %facet_type.b1a = facet_value %Int16, (%DestroyImpl.impl_witness.891, %custom_witness.b80ef6.1, %custom_witness.100) [concrete]
// CHECK:STDOUT:   %pattern_type.06e: type = pattern_type %facet_type.b1a [concrete]
// CHECK:STDOUT:   %T.patt.315: %pattern_type.06e = symbolic_binding_pattern T, 1 [symbolic]
// CHECK:STDOUT:   %facet_type.76a: type = facet_type <@DestroyImpl & @SubobjectDestroy & @Negate where %impl.elem0.c9c = %Int32> [concrete]
// CHECK:STDOUT:   %Int32.Op.type.83af6e.19: type = fn_type @Int32.Op.19 [concrete]
// CHECK:STDOUT:   %Int32.Op.4dca31.19: %Int32.Op.type.83af6e.19 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.a85: <witness> = custom_witness (%Int32, %Int32.Op.4dca31.19), @Negate [concrete]
// CHECK:STDOUT:   %facet_value.1d9: %facet_type.76a = facet_value %Int32, (%DestroyImpl.impl_witness.716, %custom_witness.b80ef6.4, %custom_witness.a85) [concrete]
// CHECK:STDOUT:   %pattern_type.839: type = pattern_type %facet_type.76a [concrete]
// CHECK:STDOUT:   %T.patt.845: %pattern_type.839 = symbolic_binding_pattern T, 1 [symbolic]
// CHECK:STDOUT:   %complete_type.f37: <witness> = complete_type_witness %AddWith.type.e4b [concrete]
// CHECK:STDOUT:   %AddWith.facet.2ce: %AddWith.type.e4b = facet_value %i16, (%custom_witness.cba) [concrete]
// CHECK:STDOUT:   %AddWith.WithSelf.Op.type.4a0: type = fn_type @AddWith.WithSelf.Op, @AddWith.WithSelf(%Int16, %AddWith.facet.2ce) [concrete]
// CHECK:STDOUT:   %.7bc: type = fn_type_with_self_type %AddWith.WithSelf.Op.type.4a0, %AddWith.facet.2ce [concrete]
// CHECK:STDOUT:   %complete_type.8a2: <witness> = complete_type_witness %SubWith.type.68d [concrete]
// CHECK:STDOUT:   %SubWith.facet.1ef: %SubWith.type.68d = facet_value %i16, (%custom_witness.354) [concrete]
// CHECK:STDOUT:   %SubWith.WithSelf.Op.type.23f: type = fn_type @SubWith.WithSelf.Op, @SubWith.WithSelf(%Int16, %SubWith.facet.1ef) [concrete]
// CHECK:STDOUT:   %.eed: type = fn_type_with_self_type %SubWith.WithSelf.Op.type.23f, %SubWith.facet.1ef [concrete]
// CHECK:STDOUT:   %complete_type.8c6: <witness> = complete_type_witness %MulWith.type.f65 [concrete]
// CHECK:STDOUT:   %MulWith.facet.d84: %MulWith.type.f65 = facet_value %i16, (%custom_witness.3c3) [concrete]
// CHECK:STDOUT:   %MulWith.WithSelf.Op.type.131: type = fn_type @MulWith.WithSelf.Op, @MulWith.WithSelf(%Int16, %MulWith.facet.d84) [concrete]
// CHECK:STDOUT:   %.010: type = fn_type_with_self_type %MulWith.WithSelf.Op.type.131, %MulWith.facet.d84 [concrete]
// CHECK:STDOUT:   %complete_type.d99: <witness> = complete_type_witness %DivWith.type.b8a [concrete]
// CHECK:STDOUT:   %DivWith.facet.4ad: %DivWith.type.b8a = facet_value %i16, (%custom_witness.8d0) [concrete]
// CHECK:STDOUT:   %DivWith.WithSelf.Op.type.edb: type = fn_type @DivWith.WithSelf.Op, @DivWith.WithSelf(%Int16, %DivWith.facet.4ad) [concrete]
// CHECK:STDOUT:   %.b5e: type = fn_type_with_self_type %DivWith.WithSelf.Op.type.edb, %DivWith.facet.4ad [concrete]
// CHECK:STDOUT:   %complete_type.7d8: <witness> = complete_type_witness %ModWith.type.c67 [concrete]
// CHECK:STDOUT:   %ModWith.facet.ab1: %ModWith.type.c67 = facet_value %i16, (%custom_witness.169) [concrete]
// CHECK:STDOUT:   %ModWith.WithSelf.Op.type.e29: type = fn_type @ModWith.WithSelf.Op, @ModWith.WithSelf(%Int16, %ModWith.facet.ab1) [concrete]
// CHECK:STDOUT:   %.d62: type = fn_type_with_self_type %ModWith.WithSelf.Op.type.e29, %ModWith.facet.ab1 [concrete]
// CHECK:STDOUT:   %complete_type.32d: <witness> = complete_type_witness %AddWith.type.2ed [concrete]
// CHECK:STDOUT:   %AddWith.facet.11e: %AddWith.type.2ed = facet_value %Int16, (%custom_witness.b79) [concrete]
// CHECK:STDOUT:   %AddWith.WithSelf.Op.type.eb3: type = fn_type @AddWith.WithSelf.Op, @AddWith.WithSelf(%i16, %AddWith.facet.11e) [concrete]
// CHECK:STDOUT:   %.da8: type = fn_type_with_self_type %AddWith.WithSelf.Op.type.eb3, %AddWith.facet.11e [concrete]
// CHECK:STDOUT:   %complete_type.75c: <witness> = complete_type_witness %SubWith.type.606 [concrete]
// CHECK:STDOUT:   %SubWith.facet.6fa: %SubWith.type.606 = facet_value %Int16, (%custom_witness.1e8) [concrete]
// CHECK:STDOUT:   %SubWith.WithSelf.Op.type.bb5: type = fn_type @SubWith.WithSelf.Op, @SubWith.WithSelf(%i16, %SubWith.facet.6fa) [concrete]
// CHECK:STDOUT:   %.42a: type = fn_type_with_self_type %SubWith.WithSelf.Op.type.bb5, %SubWith.facet.6fa [concrete]
// CHECK:STDOUT:   %complete_type.a7a: <witness> = complete_type_witness %MulWith.type.1e2 [concrete]
// CHECK:STDOUT:   %MulWith.facet.8b1: %MulWith.type.1e2 = facet_value %Int16, (%custom_witness.e2b) [concrete]
// CHECK:STDOUT:   %MulWith.WithSelf.Op.type.8f5: type = fn_type @MulWith.WithSelf.Op, @MulWith.WithSelf(%i16, %MulWith.facet.8b1) [concrete]
// CHECK:STDOUT:   %.ecc: type = fn_type_with_self_type %MulWith.WithSelf.Op.type.8f5, %MulWith.facet.8b1 [concrete]
// CHECK:STDOUT:   %complete_type.d57: <witness> = complete_type_witness %DivWith.type.45d [concrete]
// CHECK:STDOUT:   %DivWith.facet.57c: %DivWith.type.45d = facet_value %Int16, (%custom_witness.21db) [concrete]
// CHECK:STDOUT:   %DivWith.WithSelf.Op.type.37a: type = fn_type @DivWith.WithSelf.Op, @DivWith.WithSelf(%i16, %DivWith.facet.57c) [concrete]
// CHECK:STDOUT:   %.5a3: type = fn_type_with_self_type %DivWith.WithSelf.Op.type.37a, %DivWith.facet.57c [concrete]
// CHECK:STDOUT:   %complete_type.4d7: <witness> = complete_type_witness %ModWith.type.104 [concrete]
// CHECK:STDOUT:   %ModWith.facet.f5a5: %ModWith.type.104 = facet_value %Int16, (%custom_witness.6dd) [concrete]
// CHECK:STDOUT:   %ModWith.WithSelf.Op.type.374: type = fn_type @ModWith.WithSelf.Op, @ModWith.WithSelf(%i16, %ModWith.facet.f5a5) [concrete]
// CHECK:STDOUT:   %.52c1: type = fn_type_with_self_type %ModWith.WithSelf.Op.type.374, %ModWith.facet.f5a5 [concrete]
// CHECK:STDOUT:   %complete_type.758: <witness> = complete_type_witness %AddWith.type.db7 [concrete]
// CHECK:STDOUT:   %AddWith.facet.8f8: %AddWith.type.db7 = facet_value %Int16, (%custom_witness.513476.1) [concrete]
// CHECK:STDOUT:   %AddWith.WithSelf.Op.type.4cf: type = fn_type @AddWith.WithSelf.Op, @AddWith.WithSelf(%Int32, %AddWith.facet.8f8) [concrete]
// CHECK:STDOUT:   %.feb: type = fn_type_with_self_type %AddWith.WithSelf.Op.type.4cf, %AddWith.facet.8f8 [concrete]
// CHECK:STDOUT:   %complete_type.46e: <witness> = complete_type_witness %SubWith.type.120 [concrete]
// CHECK:STDOUT:   %SubWith.facet.0b0: %SubWith.type.120 = facet_value %Int16, (%custom_witness.32cef1.1) [concrete]
// CHECK:STDOUT:   %SubWith.WithSelf.Op.type.e1e: type = fn_type @SubWith.WithSelf.Op, @SubWith.WithSelf(%Int32, %SubWith.facet.0b0) [concrete]
// CHECK:STDOUT:   %.988: type = fn_type_with_self_type %SubWith.WithSelf.Op.type.e1e, %SubWith.facet.0b0 [concrete]
// CHECK:STDOUT:   %complete_type.920: <witness> = complete_type_witness %MulWith.type.d67 [concrete]
// CHECK:STDOUT:   %MulWith.facet.4b9: %MulWith.type.d67 = facet_value %Int16, (%custom_witness.3a0675.1) [concrete]
// CHECK:STDOUT:   %MulWith.WithSelf.Op.type.8f2: type = fn_type @MulWith.WithSelf.Op, @MulWith.WithSelf(%Int32, %MulWith.facet.4b9) [concrete]
// CHECK:STDOUT:   %.47a: type = fn_type_with_self_type %MulWith.WithSelf.Op.type.8f2, %MulWith.facet.4b9 [concrete]
// CHECK:STDOUT:   %complete_type.348: <witness> = complete_type_witness %DivWith.type.5a1 [concrete]
// CHECK:STDOUT:   %DivWith.facet.5c5: %DivWith.type.5a1 = facet_value %Int16, (%custom_witness.c32685.1) [concrete]
// CHECK:STDOUT:   %DivWith.WithSelf.Op.type.af3: type = fn_type @DivWith.WithSelf.Op, @DivWith.WithSelf(%Int32, %DivWith.facet.5c5) [concrete]
// CHECK:STDOUT:   %.836: type = fn_type_with_self_type %DivWith.WithSelf.Op.type.af3, %DivWith.facet.5c5 [concrete]
// CHECK:STDOUT:   %complete_type.cc9: <witness> = complete_type_witness %ModWith.type.794 [concrete]
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
// CHECK:STDOUT:   %complete_type.57b: <witness> = complete_type_witness %AddWith.type.1ff [concrete]
// CHECK:STDOUT:   %AddWith.facet.0d8: %AddWith.type.1ff = facet_value %Int32, (%custom_witness.f2f) [concrete]
// CHECK:STDOUT:   %AddWith.WithSelf.Op.type.a40: type = fn_type @AddWith.WithSelf.Op, @AddWith.WithSelf(%Int64, %AddWith.facet.0d8) [concrete]
// CHECK:STDOUT:   %.51d: type = fn_type_with_self_type %AddWith.WithSelf.Op.type.a40, %AddWith.facet.0d8 [concrete]
// CHECK:STDOUT:   %complete_type.cb5: <witness> = complete_type_witness %SubWith.type.31c [concrete]
// CHECK:STDOUT:   %SubWith.facet.d62: %SubWith.type.31c = facet_value %Int32, (%custom_witness.276) [concrete]
// CHECK:STDOUT:   %SubWith.WithSelf.Op.type.d39: type = fn_type @SubWith.WithSelf.Op, @SubWith.WithSelf(%Int64, %SubWith.facet.d62) [concrete]
// CHECK:STDOUT:   %.e39: type = fn_type_with_self_type %SubWith.WithSelf.Op.type.d39, %SubWith.facet.d62 [concrete]
// CHECK:STDOUT:   %complete_type.0a6: <witness> = complete_type_witness %MulWith.type.661 [concrete]
// CHECK:STDOUT:   %MulWith.facet.4a0: %MulWith.type.661 = facet_value %Int32, (%custom_witness.8e4) [concrete]
// CHECK:STDOUT:   %MulWith.WithSelf.Op.type.489: type = fn_type @MulWith.WithSelf.Op, @MulWith.WithSelf(%Int64, %MulWith.facet.4a0) [concrete]
// CHECK:STDOUT:   %.2e6: type = fn_type_with_self_type %MulWith.WithSelf.Op.type.489, %MulWith.facet.4a0 [concrete]
// CHECK:STDOUT:   %complete_type.473: <witness> = complete_type_witness %DivWith.type.dd8 [concrete]
// CHECK:STDOUT:   %DivWith.facet.b29: %DivWith.type.dd8 = facet_value %Int32, (%custom_witness.099) [concrete]
// CHECK:STDOUT:   %DivWith.WithSelf.Op.type.9fa: type = fn_type @DivWith.WithSelf.Op, @DivWith.WithSelf(%Int64, %DivWith.facet.b29) [concrete]
// CHECK:STDOUT:   %.bc2: type = fn_type_with_self_type %DivWith.WithSelf.Op.type.9fa, %DivWith.facet.b29 [concrete]
// CHECK:STDOUT:   %complete_type.4d1: <witness> = complete_type_witness %ModWith.type.c72 [concrete]
// CHECK:STDOUT:   %ModWith.facet.b10: %ModWith.type.c72 = facet_value %Int32, (%custom_witness.129) [concrete]
// CHECK:STDOUT:   %ModWith.WithSelf.Op.type.1ecc: type = fn_type @ModWith.WithSelf.Op, @ModWith.WithSelf(%Int64, %ModWith.facet.b10) [concrete]
// CHECK:STDOUT:   %.bf4: type = fn_type_with_self_type %ModWith.WithSelf.Op.type.1ecc, %ModWith.facet.b10 [concrete]
// CHECK:STDOUT:   %AddWith.facet.f65: %AddWith.type.db7 = facet_value %Int64, (%custom_witness.16b) [concrete]
// CHECK:STDOUT:   %AddWith.WithSelf.Op.type.8b6: type = fn_type @AddWith.WithSelf.Op, @AddWith.WithSelf(%Int32, %AddWith.facet.f65) [concrete]
// CHECK:STDOUT:   %.cf7: type = fn_type_with_self_type %AddWith.WithSelf.Op.type.8b6, %AddWith.facet.f65 [concrete]
// CHECK:STDOUT:   %SubWith.facet.245: %SubWith.type.120 = facet_value %Int64, (%custom_witness.21a) [concrete]
// CHECK:STDOUT:   %SubWith.WithSelf.Op.type.2cf1: type = fn_type @SubWith.WithSelf.Op, @SubWith.WithSelf(%Int32, %SubWith.facet.245) [concrete]
// CHECK:STDOUT:   %.4e8: type = fn_type_with_self_type %SubWith.WithSelf.Op.type.2cf1, %SubWith.facet.245 [concrete]
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
// CHECK:STDOUT:   %.5363: type = fn_type_with_self_type %MulAssignWith.WithSelf.Op.type.2e6, %MulAssignWith.facet.d5c [concrete]
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
// CHECK:STDOUT: generic fn @AddWith.loc124(%Result.loc121_9.2: %facet_type.ece, %U.loc122_4.2: %facet_type.ece, %T.loc123_4.2: @AddWith.loc124.%facet_type.loc123_37 (%facet_type.0bd)) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %require_complete.loc126: <witness> = require_complete_type %AddWith.type.loc123_35.1 [symbolic = %require_complete.loc126 (constants.%require_complete.2d4)]
// CHECK:STDOUT:   %assoc1: @AddWith.loc124.%AddWith.assoc_type (%AddWith.assoc_type.cc7) = assoc_entity element1, imports.%Core.import_ref.8ad [symbolic = %assoc1 (constants.%assoc1.6cf)]
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.loc126: <witness> = lookup_impl_witness %T.loc123_4.1, @AddWith.1, @AddWith.1(%U.as_type.loc123_35.1) [symbolic = %AddWith.lookup_impl_witness.loc126 (constants.%AddWith.lookup_impl_witness.f7f)]
// CHECK:STDOUT:   %AddWith.facet: @AddWith.loc124.%AddWith.type.loc123_35.1 (%AddWith.type.55e) = facet_value %T.as_type.loc124_29.1, (%AddWith.lookup_impl_witness.loc126) [symbolic = %AddWith.facet (constants.%AddWith.facet.a2a)]
// CHECK:STDOUT:   %AddWith.WithSelf.Op.type: type = fn_type @AddWith.WithSelf.Op, @AddWith.WithSelf(%U.as_type.loc123_35.1, %AddWith.facet) [symbolic = %AddWith.WithSelf.Op.type (constants.%AddWith.WithSelf.Op.type.b84)]
// CHECK:STDOUT:   %.loc126_5.6: type = fn_type_with_self_type %AddWith.WithSelf.Op.type, %AddWith.facet [symbolic = %.loc126_5.6 (constants.%.28a)]
// CHECK:STDOUT:   %impl.elem1.loc126_5.2: @AddWith.loc124.%.loc126_5.6 (%.28a) = impl_witness_access %AddWith.lookup_impl_witness.loc126, element1 [symbolic = %impl.elem1.loc126_5.2 (constants.%impl.elem1.5d2)]
// CHECK:STDOUT:   %specific_impl_fn.loc126_5.2: <specific function> = specific_impl_function %impl.elem1.loc126_5.2, @AddWith.WithSelf.Op(%U.as_type.loc123_35.1, %AddWith.facet) [symbolic = %specific_impl_fn.loc126_5.2 (constants.%specific_impl_fn.140)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%result.param: @AddWith.loc124.%Result.as_type.loc123_53.1 (%Result.as_type), %x.param: @AddWith.loc124.%T.as_type.loc124_29.1 (%T.as_type.d99), %y.param: @AddWith.loc124.%U.as_type.loc123_35.1 (%U.as_type.15c)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: @AddWith.loc124.%T.as_type.loc124_29.1 (%T.as_type.d99) = name_ref x, %x
// CHECK:STDOUT:     %y.ref: @AddWith.loc124.%U.as_type.loc123_35.1 (%U.as_type.15c) = name_ref y, %y
// CHECK:STDOUT:     %AddWith.type.loc126: type = facet_type <@AddWith.1, @AddWith.1(constants.%U.as_type.15c)> [symbolic = %AddWith.type.loc123_35.1 (constants.%AddWith.type.55e)]
// CHECK:STDOUT:     %.loc126_5.1: @AddWith.loc124.%AddWith.assoc_type (%AddWith.assoc_type.cc7) = specific_constant imports.%Core.import_ref.5c64, @AddWith.WithSelf(constants.%U.as_type.15c, constants.%Self.958) [symbolic = %assoc1 (constants.%assoc1.6cf)]
// CHECK:STDOUT:     %Op.ref: @AddWith.loc124.%AddWith.assoc_type (%AddWith.assoc_type.cc7) = name_ref Op, %.loc126_5.1 [symbolic = %assoc1 (constants.%assoc1.6cf)]
// CHECK:STDOUT:     %impl.elem1.loc126_5.1: @AddWith.loc124.%.loc126_5.6 (%.28a) = impl_witness_access constants.%AddWith.lookup_impl_witness.f7f, element1 [symbolic = %impl.elem1.loc126_5.2 (constants.%impl.elem1.5d2)]
// CHECK:STDOUT:     %bound_method.loc126_5.1: <bound method> = bound_method %x.ref, %impl.elem1.loc126_5.1
// CHECK:STDOUT:     %specific_impl_fn.loc126_5.1: <specific function> = specific_impl_function %impl.elem1.loc126_5.1, @AddWith.WithSelf.Op(constants.%U.as_type.15c, constants.%AddWith.facet.a2a) [symbolic = %specific_impl_fn.loc126_5.2 (constants.%specific_impl_fn.140)]
// CHECK:STDOUT:     %bound_method.loc126_5.2: <bound method> = bound_method %x.ref, %specific_impl_fn.loc126_5.1
// CHECK:STDOUT:     %.loc126_5.2: ref @AddWith.loc124.%Result.as_type.loc123_53.1 (%Result.as_type) = temporary_storage
// CHECK:STDOUT:     %AddWith.WithSelf.Op.call: init @AddWith.loc124.%Result.as_type.loc123_53.1 (%Result.as_type) to %.loc126_5.2 = call %bound_method.loc126_5.2(%x.ref, %y.ref)
// CHECK:STDOUT:     %.loc126_5.3: ref @AddWith.loc124.%Result.as_type.loc123_53.1 (%Result.as_type) = temporary %.loc126_5.2, %AddWith.WithSelf.Op.call
// CHECK:STDOUT:     %.loc126_5.4: %facet_type.ece = converted constants.%Result.as_type, constants.%Result [symbolic = %Result.loc121_9.1 (constants.%Result)]
// CHECK:STDOUT:     %.loc126_5.5: %facet_type.ece = converted constants.%Result.as_type, constants.%Result [symbolic = %Result.loc121_9.1 (constants.%Result)]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc126_5.3)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @SubWith.loc135(%Result.loc132_9.2: %facet_type.ece, %U.loc133_4.2: %facet_type.ece, %T.loc134_4.2: @SubWith.loc135.%facet_type.loc134_37 (%facet_type.952)) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %require_complete.loc137: <witness> = require_complete_type %SubWith.type.loc134_35.1 [symbolic = %require_complete.loc137 (constants.%require_complete.6c6)]
// CHECK:STDOUT:   %assoc1: @SubWith.loc135.%SubWith.assoc_type (%SubWith.assoc_type.cf4) = assoc_entity element1, imports.%Core.import_ref.2f4 [symbolic = %assoc1 (constants.%assoc1.543)]
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.loc137: <witness> = lookup_impl_witness %T.loc134_4.1, @SubWith.1, @SubWith.1(%U.as_type.loc134_35.1) [symbolic = %SubWith.lookup_impl_witness.loc137 (constants.%SubWith.lookup_impl_witness.c46)]
// CHECK:STDOUT:   %SubWith.facet: @SubWith.loc135.%SubWith.type.loc134_35.1 (%SubWith.type.776) = facet_value %T.as_type.loc135_29.1, (%SubWith.lookup_impl_witness.loc137) [symbolic = %SubWith.facet (constants.%SubWith.facet.26d)]
// CHECK:STDOUT:   %SubWith.WithSelf.Op.type: type = fn_type @SubWith.WithSelf.Op, @SubWith.WithSelf(%U.as_type.loc134_35.1, %SubWith.facet) [symbolic = %SubWith.WithSelf.Op.type (constants.%SubWith.WithSelf.Op.type.0e6)]
// CHECK:STDOUT:   %.loc137_5.6: type = fn_type_with_self_type %SubWith.WithSelf.Op.type, %SubWith.facet [symbolic = %.loc137_5.6 (constants.%.3a1)]
// CHECK:STDOUT:   %impl.elem1.loc137_5.2: @SubWith.loc135.%.loc137_5.6 (%.3a1) = impl_witness_access %SubWith.lookup_impl_witness.loc137, element1 [symbolic = %impl.elem1.loc137_5.2 (constants.%impl.elem1.1b8)]
// CHECK:STDOUT:   %specific_impl_fn.loc137_5.2: <specific function> = specific_impl_function %impl.elem1.loc137_5.2, @SubWith.WithSelf.Op(%U.as_type.loc134_35.1, %SubWith.facet) [symbolic = %specific_impl_fn.loc137_5.2 (constants.%specific_impl_fn.0da)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%result.param: @SubWith.loc135.%Result.as_type.loc134_53.1 (%Result.as_type), %x.param: @SubWith.loc135.%T.as_type.loc135_29.1 (%T.as_type.137), %y.param: @SubWith.loc135.%U.as_type.loc134_35.1 (%U.as_type.15c)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: @SubWith.loc135.%T.as_type.loc135_29.1 (%T.as_type.137) = name_ref x, %x
// CHECK:STDOUT:     %y.ref: @SubWith.loc135.%U.as_type.loc134_35.1 (%U.as_type.15c) = name_ref y, %y
// CHECK:STDOUT:     %SubWith.type.loc137: type = facet_type <@SubWith.1, @SubWith.1(constants.%U.as_type.15c)> [symbolic = %SubWith.type.loc134_35.1 (constants.%SubWith.type.776)]
// CHECK:STDOUT:     %.loc137_5.1: @SubWith.loc135.%SubWith.assoc_type (%SubWith.assoc_type.cf4) = specific_constant imports.%Core.import_ref.5fe, @SubWith.WithSelf(constants.%U.as_type.15c, constants.%Self.18a) [symbolic = %assoc1 (constants.%assoc1.543)]
// CHECK:STDOUT:     %Op.ref: @SubWith.loc135.%SubWith.assoc_type (%SubWith.assoc_type.cf4) = name_ref Op, %.loc137_5.1 [symbolic = %assoc1 (constants.%assoc1.543)]
// CHECK:STDOUT:     %impl.elem1.loc137_5.1: @SubWith.loc135.%.loc137_5.6 (%.3a1) = impl_witness_access constants.%SubWith.lookup_impl_witness.c46, element1 [symbolic = %impl.elem1.loc137_5.2 (constants.%impl.elem1.1b8)]
// CHECK:STDOUT:     %bound_method.loc137_5.1: <bound method> = bound_method %x.ref, %impl.elem1.loc137_5.1
// CHECK:STDOUT:     %specific_impl_fn.loc137_5.1: <specific function> = specific_impl_function %impl.elem1.loc137_5.1, @SubWith.WithSelf.Op(constants.%U.as_type.15c, constants.%SubWith.facet.26d) [symbolic = %specific_impl_fn.loc137_5.2 (constants.%specific_impl_fn.0da)]
// CHECK:STDOUT:     %bound_method.loc137_5.2: <bound method> = bound_method %x.ref, %specific_impl_fn.loc137_5.1
// CHECK:STDOUT:     %.loc137_5.2: ref @SubWith.loc135.%Result.as_type.loc134_53.1 (%Result.as_type) = temporary_storage
// CHECK:STDOUT:     %SubWith.WithSelf.Op.call: init @SubWith.loc135.%Result.as_type.loc134_53.1 (%Result.as_type) to %.loc137_5.2 = call %bound_method.loc137_5.2(%x.ref, %y.ref)
// CHECK:STDOUT:     %.loc137_5.3: ref @SubWith.loc135.%Result.as_type.loc134_53.1 (%Result.as_type) = temporary %.loc137_5.2, %SubWith.WithSelf.Op.call
// CHECK:STDOUT:     %.loc137_5.4: %facet_type.ece = converted constants.%Result.as_type, constants.%Result [symbolic = %Result.loc132_9.1 (constants.%Result)]
// CHECK:STDOUT:     %.loc137_5.5: %facet_type.ece = converted constants.%Result.as_type, constants.%Result [symbolic = %Result.loc132_9.1 (constants.%Result)]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc137_5.3)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @MulWith.loc146(%Result.loc143_9.2: %facet_type.ece, %U.loc144_4.2: %facet_type.ece, %T.loc145_4.2: @MulWith.loc146.%facet_type.loc145_37 (%facet_type.6e8)) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %require_complete.loc148: <witness> = require_complete_type %MulWith.type.loc145_35.1 [symbolic = %require_complete.loc148 (constants.%require_complete.91f)]
// CHECK:STDOUT:   %assoc1: @MulWith.loc146.%MulWith.assoc_type (%MulWith.assoc_type.083) = assoc_entity element1, imports.%Core.import_ref.fc2 [symbolic = %assoc1 (constants.%assoc1.2ea)]
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.loc148: <witness> = lookup_impl_witness %T.loc145_4.1, @MulWith.1, @MulWith.1(%U.as_type.loc145_35.1) [symbolic = %MulWith.lookup_impl_witness.loc148 (constants.%MulWith.lookup_impl_witness.10d)]
// CHECK:STDOUT:   %MulWith.facet: @MulWith.loc146.%MulWith.type.loc145_35.1 (%MulWith.type.068) = facet_value %T.as_type.loc146_29.1, (%MulWith.lookup_impl_witness.loc148) [symbolic = %MulWith.facet (constants.%MulWith.facet.aa0)]
// CHECK:STDOUT:   %MulWith.WithSelf.Op.type: type = fn_type @MulWith.WithSelf.Op, @MulWith.WithSelf(%U.as_type.loc145_35.1, %MulWith.facet) [symbolic = %MulWith.WithSelf.Op.type (constants.%MulWith.WithSelf.Op.type.e1e)]
// CHECK:STDOUT:   %.loc148_5.6: type = fn_type_with_self_type %MulWith.WithSelf.Op.type, %MulWith.facet [symbolic = %.loc148_5.6 (constants.%.7cf)]
// CHECK:STDOUT:   %impl.elem1.loc148_5.2: @MulWith.loc146.%.loc148_5.6 (%.7cf) = impl_witness_access %MulWith.lookup_impl_witness.loc148, element1 [symbolic = %impl.elem1.loc148_5.2 (constants.%impl.elem1.0d2)]
// CHECK:STDOUT:   %specific_impl_fn.loc148_5.2: <specific function> = specific_impl_function %impl.elem1.loc148_5.2, @MulWith.WithSelf.Op(%U.as_type.loc145_35.1, %MulWith.facet) [symbolic = %specific_impl_fn.loc148_5.2 (constants.%specific_impl_fn.e0c)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%result.param: @MulWith.loc146.%Result.as_type.loc145_53.1 (%Result.as_type), %x.param: @MulWith.loc146.%T.as_type.loc146_29.1 (%T.as_type.d42), %y.param: @MulWith.loc146.%U.as_type.loc145_35.1 (%U.as_type.15c)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: @MulWith.loc146.%T.as_type.loc146_29.1 (%T.as_type.d42) = name_ref x, %x
// CHECK:STDOUT:     %y.ref: @MulWith.loc146.%U.as_type.loc145_35.1 (%U.as_type.15c) = name_ref y, %y
// CHECK:STDOUT:     %MulWith.type.loc148: type = facet_type <@MulWith.1, @MulWith.1(constants.%U.as_type.15c)> [symbolic = %MulWith.type.loc145_35.1 (constants.%MulWith.type.068)]
// CHECK:STDOUT:     %.loc148_5.1: @MulWith.loc146.%MulWith.assoc_type (%MulWith.assoc_type.083) = specific_constant imports.%Core.import_ref.0a8, @MulWith.WithSelf(constants.%U.as_type.15c, constants.%Self.742) [symbolic = %assoc1 (constants.%assoc1.2ea)]
// CHECK:STDOUT:     %Op.ref: @MulWith.loc146.%MulWith.assoc_type (%MulWith.assoc_type.083) = name_ref Op, %.loc148_5.1 [symbolic = %assoc1 (constants.%assoc1.2ea)]
// CHECK:STDOUT:     %impl.elem1.loc148_5.1: @MulWith.loc146.%.loc148_5.6 (%.7cf) = impl_witness_access constants.%MulWith.lookup_impl_witness.10d, element1 [symbolic = %impl.elem1.loc148_5.2 (constants.%impl.elem1.0d2)]
// CHECK:STDOUT:     %bound_method.loc148_5.1: <bound method> = bound_method %x.ref, %impl.elem1.loc148_5.1
// CHECK:STDOUT:     %specific_impl_fn.loc148_5.1: <specific function> = specific_impl_function %impl.elem1.loc148_5.1, @MulWith.WithSelf.Op(constants.%U.as_type.15c, constants.%MulWith.facet.aa0) [symbolic = %specific_impl_fn.loc148_5.2 (constants.%specific_impl_fn.e0c)]
// CHECK:STDOUT:     %bound_method.loc148_5.2: <bound method> = bound_method %x.ref, %specific_impl_fn.loc148_5.1
// CHECK:STDOUT:     %.loc148_5.2: ref @MulWith.loc146.%Result.as_type.loc145_53.1 (%Result.as_type) = temporary_storage
// CHECK:STDOUT:     %MulWith.WithSelf.Op.call: init @MulWith.loc146.%Result.as_type.loc145_53.1 (%Result.as_type) to %.loc148_5.2 = call %bound_method.loc148_5.2(%x.ref, %y.ref)
// CHECK:STDOUT:     %.loc148_5.3: ref @MulWith.loc146.%Result.as_type.loc145_53.1 (%Result.as_type) = temporary %.loc148_5.2, %MulWith.WithSelf.Op.call
// CHECK:STDOUT:     %.loc148_5.4: %facet_type.ece = converted constants.%Result.as_type, constants.%Result [symbolic = %Result.loc143_9.1 (constants.%Result)]
// CHECK:STDOUT:     %.loc148_5.5: %facet_type.ece = converted constants.%Result.as_type, constants.%Result [symbolic = %Result.loc143_9.1 (constants.%Result)]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc148_5.3)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @DivWith.loc157(%Result.loc154_9.2: %facet_type.ece, %U.loc155_4.2: %facet_type.ece, %T.loc156_4.2: @DivWith.loc157.%facet_type.loc156_37 (%facet_type.344)) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %require_complete.loc159: <witness> = require_complete_type %DivWith.type.loc156_35.1 [symbolic = %require_complete.loc159 (constants.%require_complete.a62)]
// CHECK:STDOUT:   %assoc1: @DivWith.loc157.%DivWith.assoc_type (%DivWith.assoc_type.9a3) = assoc_entity element1, imports.%Core.import_ref.f2f [symbolic = %assoc1 (constants.%assoc1.1bc)]
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.loc159: <witness> = lookup_impl_witness %T.loc156_4.1, @DivWith.1, @DivWith.1(%U.as_type.loc156_35.1) [symbolic = %DivWith.lookup_impl_witness.loc159 (constants.%DivWith.lookup_impl_witness.51a)]
// CHECK:STDOUT:   %DivWith.facet: @DivWith.loc157.%DivWith.type.loc156_35.1 (%DivWith.type.687) = facet_value %T.as_type.loc157_29.1, (%DivWith.lookup_impl_witness.loc159) [symbolic = %DivWith.facet (constants.%DivWith.facet.901)]
// CHECK:STDOUT:   %DivWith.WithSelf.Op.type: type = fn_type @DivWith.WithSelf.Op, @DivWith.WithSelf(%U.as_type.loc156_35.1, %DivWith.facet) [symbolic = %DivWith.WithSelf.Op.type (constants.%DivWith.WithSelf.Op.type.396)]
// CHECK:STDOUT:   %.loc159_5.6: type = fn_type_with_self_type %DivWith.WithSelf.Op.type, %DivWith.facet [symbolic = %.loc159_5.6 (constants.%.422)]
// CHECK:STDOUT:   %impl.elem1.loc159_5.2: @DivWith.loc157.%.loc159_5.6 (%.422) = impl_witness_access %DivWith.lookup_impl_witness.loc159, element1 [symbolic = %impl.elem1.loc159_5.2 (constants.%impl.elem1.338)]
// CHECK:STDOUT:   %specific_impl_fn.loc159_5.2: <specific function> = specific_impl_function %impl.elem1.loc159_5.2, @DivWith.WithSelf.Op(%U.as_type.loc156_35.1, %DivWith.facet) [symbolic = %specific_impl_fn.loc159_5.2 (constants.%specific_impl_fn.8b9)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%result.param: @DivWith.loc157.%Result.as_type.loc156_53.1 (%Result.as_type), %x.param: @DivWith.loc157.%T.as_type.loc157_29.1 (%T.as_type.6cf), %y.param: @DivWith.loc157.%U.as_type.loc156_35.1 (%U.as_type.15c)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: @DivWith.loc157.%T.as_type.loc157_29.1 (%T.as_type.6cf) = name_ref x, %x
// CHECK:STDOUT:     %y.ref: @DivWith.loc157.%U.as_type.loc156_35.1 (%U.as_type.15c) = name_ref y, %y
// CHECK:STDOUT:     %DivWith.type.loc159: type = facet_type <@DivWith.1, @DivWith.1(constants.%U.as_type.15c)> [symbolic = %DivWith.type.loc156_35.1 (constants.%DivWith.type.687)]
// CHECK:STDOUT:     %.loc159_5.1: @DivWith.loc157.%DivWith.assoc_type (%DivWith.assoc_type.9a3) = specific_constant imports.%Core.import_ref.972, @DivWith.WithSelf(constants.%U.as_type.15c, constants.%Self.5b3) [symbolic = %assoc1 (constants.%assoc1.1bc)]
// CHECK:STDOUT:     %Op.ref: @DivWith.loc157.%DivWith.assoc_type (%DivWith.assoc_type.9a3) = name_ref Op, %.loc159_5.1 [symbolic = %assoc1 (constants.%assoc1.1bc)]
// CHECK:STDOUT:     %impl.elem1.loc159_5.1: @DivWith.loc157.%.loc159_5.6 (%.422) = impl_witness_access constants.%DivWith.lookup_impl_witness.51a, element1 [symbolic = %impl.elem1.loc159_5.2 (constants.%impl.elem1.338)]
// CHECK:STDOUT:     %bound_method.loc159_5.1: <bound method> = bound_method %x.ref, %impl.elem1.loc159_5.1
// CHECK:STDOUT:     %specific_impl_fn.loc159_5.1: <specific function> = specific_impl_function %impl.elem1.loc159_5.1, @DivWith.WithSelf.Op(constants.%U.as_type.15c, constants.%DivWith.facet.901) [symbolic = %specific_impl_fn.loc159_5.2 (constants.%specific_impl_fn.8b9)]
// CHECK:STDOUT:     %bound_method.loc159_5.2: <bound method> = bound_method %x.ref, %specific_impl_fn.loc159_5.1
// CHECK:STDOUT:     %.loc159_5.2: ref @DivWith.loc157.%Result.as_type.loc156_53.1 (%Result.as_type) = temporary_storage
// CHECK:STDOUT:     %DivWith.WithSelf.Op.call: init @DivWith.loc157.%Result.as_type.loc156_53.1 (%Result.as_type) to %.loc159_5.2 = call %bound_method.loc159_5.2(%x.ref, %y.ref)
// CHECK:STDOUT:     %.loc159_5.3: ref @DivWith.loc157.%Result.as_type.loc156_53.1 (%Result.as_type) = temporary %.loc159_5.2, %DivWith.WithSelf.Op.call
// CHECK:STDOUT:     %.loc159_5.4: %facet_type.ece = converted constants.%Result.as_type, constants.%Result [symbolic = %Result.loc154_9.1 (constants.%Result)]
// CHECK:STDOUT:     %.loc159_5.5: %facet_type.ece = converted constants.%Result.as_type, constants.%Result [symbolic = %Result.loc154_9.1 (constants.%Result)]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc159_5.3)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @ModWith.loc168(%Result.loc165_9.2: %facet_type.ece, %U.loc166_4.2: %facet_type.ece, %T.loc167_4.2: @ModWith.loc168.%facet_type.loc167_37 (%facet_type.668)) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %require_complete.loc170: <witness> = require_complete_type %ModWith.type.loc167_35.1 [symbolic = %require_complete.loc170 (constants.%require_complete.714)]
// CHECK:STDOUT:   %assoc1: @ModWith.loc168.%ModWith.assoc_type (%ModWith.assoc_type.279) = assoc_entity element1, imports.%Core.import_ref.9f8f [symbolic = %assoc1 (constants.%assoc1.bb3)]
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.loc170: <witness> = lookup_impl_witness %T.loc167_4.1, @ModWith.1, @ModWith.1(%U.as_type.loc167_35.1) [symbolic = %ModWith.lookup_impl_witness.loc170 (constants.%ModWith.lookup_impl_witness.03e)]
// CHECK:STDOUT:   %ModWith.facet: @ModWith.loc168.%ModWith.type.loc167_35.1 (%ModWith.type.584) = facet_value %T.as_type.loc168_29.1, (%ModWith.lookup_impl_witness.loc170) [symbolic = %ModWith.facet (constants.%ModWith.facet.b48)]
// CHECK:STDOUT:   %ModWith.WithSelf.Op.type: type = fn_type @ModWith.WithSelf.Op, @ModWith.WithSelf(%U.as_type.loc167_35.1, %ModWith.facet) [symbolic = %ModWith.WithSelf.Op.type (constants.%ModWith.WithSelf.Op.type.bef)]
// CHECK:STDOUT:   %.loc170_5.6: type = fn_type_with_self_type %ModWith.WithSelf.Op.type, %ModWith.facet [symbolic = %.loc170_5.6 (constants.%.5369)]
// CHECK:STDOUT:   %impl.elem1.loc170_5.2: @ModWith.loc168.%.loc170_5.6 (%.5369) = impl_witness_access %ModWith.lookup_impl_witness.loc170, element1 [symbolic = %impl.elem1.loc170_5.2 (constants.%impl.elem1.275)]
// CHECK:STDOUT:   %specific_impl_fn.loc170_5.2: <specific function> = specific_impl_function %impl.elem1.loc170_5.2, @ModWith.WithSelf.Op(%U.as_type.loc167_35.1, %ModWith.facet) [symbolic = %specific_impl_fn.loc170_5.2 (constants.%specific_impl_fn.bba)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%result.param: @ModWith.loc168.%Result.as_type.loc167_53.1 (%Result.as_type), %x.param: @ModWith.loc168.%T.as_type.loc168_29.1 (%T.as_type.a02), %y.param: @ModWith.loc168.%U.as_type.loc167_35.1 (%U.as_type.15c)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: @ModWith.loc168.%T.as_type.loc168_29.1 (%T.as_type.a02) = name_ref x, %x
// CHECK:STDOUT:     %y.ref: @ModWith.loc168.%U.as_type.loc167_35.1 (%U.as_type.15c) = name_ref y, %y
// CHECK:STDOUT:     %ModWith.type.loc170: type = facet_type <@ModWith.1, @ModWith.1(constants.%U.as_type.15c)> [symbolic = %ModWith.type.loc167_35.1 (constants.%ModWith.type.584)]
// CHECK:STDOUT:     %.loc170_5.1: @ModWith.loc168.%ModWith.assoc_type (%ModWith.assoc_type.279) = specific_constant imports.%Core.import_ref.ea5e3, @ModWith.WithSelf(constants.%U.as_type.15c, constants.%Self.c18) [symbolic = %assoc1 (constants.%assoc1.bb3)]
// CHECK:STDOUT:     %Op.ref: @ModWith.loc168.%ModWith.assoc_type (%ModWith.assoc_type.279) = name_ref Op, %.loc170_5.1 [symbolic = %assoc1 (constants.%assoc1.bb3)]
// CHECK:STDOUT:     %impl.elem1.loc170_5.1: @ModWith.loc168.%.loc170_5.6 (%.5369) = impl_witness_access constants.%ModWith.lookup_impl_witness.03e, element1 [symbolic = %impl.elem1.loc170_5.2 (constants.%impl.elem1.275)]
// CHECK:STDOUT:     %bound_method.loc170_5.1: <bound method> = bound_method %x.ref, %impl.elem1.loc170_5.1
// CHECK:STDOUT:     %specific_impl_fn.loc170_5.1: <specific function> = specific_impl_function %impl.elem1.loc170_5.1, @ModWith.WithSelf.Op(constants.%U.as_type.15c, constants.%ModWith.facet.b48) [symbolic = %specific_impl_fn.loc170_5.2 (constants.%specific_impl_fn.bba)]
// CHECK:STDOUT:     %bound_method.loc170_5.2: <bound method> = bound_method %x.ref, %specific_impl_fn.loc170_5.1
// CHECK:STDOUT:     %.loc170_5.2: ref @ModWith.loc168.%Result.as_type.loc167_53.1 (%Result.as_type) = temporary_storage
// CHECK:STDOUT:     %ModWith.WithSelf.Op.call: init @ModWith.loc168.%Result.as_type.loc167_53.1 (%Result.as_type) to %.loc170_5.2 = call %bound_method.loc170_5.2(%x.ref, %y.ref)
// CHECK:STDOUT:     %.loc170_5.3: ref @ModWith.loc168.%Result.as_type.loc167_53.1 (%Result.as_type) = temporary %.loc170_5.2, %ModWith.WithSelf.Op.call
// CHECK:STDOUT:     %.loc170_5.4: %facet_type.ece = converted constants.%Result.as_type, constants.%Result [symbolic = %Result.loc165_9.1 (constants.%Result)]
// CHECK:STDOUT:     %.loc170_5.5: %facet_type.ece = converted constants.%Result.as_type, constants.%Result [symbolic = %Result.loc165_9.1 (constants.%Result)]
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
// CHECK:STDOUT: generic fn @TestNegate(%Result.loc292_9.2: %facet_type.ece, %T.loc293_4.2: @TestNegate.%facet_type (%facet_type.5ed)) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Negate.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc293_4.1, @Negate [symbolic = %Negate.lookup_impl_witness (constants.%Negate.lookup_impl_witness.eaf)]
// CHECK:STDOUT:   %Negate.facet.loc296_3.3: %Negate.type = facet_value %T.as_type.loc294_24.1, (%Negate.lookup_impl_witness) [symbolic = %Negate.facet.loc296_3.3 (constants.%Negate.facet.473)]
// CHECK:STDOUT:   %Negate.WithSelf.Op.type: type = fn_type @Negate.WithSelf.Op, @Negate.WithSelf(%Negate.facet.loc296_3.3) [symbolic = %Negate.WithSelf.Op.type (constants.%Negate.WithSelf.Op.type.96d)]
// CHECK:STDOUT:   %.loc296_3.7: type = fn_type_with_self_type %Negate.WithSelf.Op.type, %Negate.facet.loc296_3.3 [symbolic = %.loc296_3.7 (constants.%.1b4)]
// CHECK:STDOUT:   %impl.elem1.loc296_3.2: @TestNegate.%.loc296_3.7 (%.1b4) = impl_witness_access %Negate.lookup_impl_witness, element1 [symbolic = %impl.elem1.loc296_3.2 (constants.%impl.elem1.322)]
// CHECK:STDOUT:   %specific_impl_fn.loc296_3.2: <specific function> = specific_impl_function %impl.elem1.loc296_3.2, @Negate.WithSelf.Op(%Negate.facet.loc296_3.3) [symbolic = %specific_impl_fn.loc296_3.2 (constants.%specific_impl_fn.511)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%y.param: @TestNegate.%Result.as_type.loc293_49.1 (%Result.as_type), %x.param: @TestNegate.%T.as_type.loc294_24.1 (%T.as_type.a42)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: @TestNegate.%T.as_type.loc294_24.1 (%T.as_type.a42) = name_ref x, %x
// CHECK:STDOUT:     %impl.elem1.loc296_3.1: @TestNegate.%.loc296_3.7 (%.1b4) = impl_witness_access constants.%Negate.lookup_impl_witness.eaf, element1 [symbolic = %impl.elem1.loc296_3.2 (constants.%impl.elem1.322)]
// CHECK:STDOUT:     %bound_method.loc296_3.1: <bound method> = bound_method %x.ref, %impl.elem1.loc296_3.1
// CHECK:STDOUT:     %Negate.facet.loc296_3.1: %Negate.type = facet_value constants.%T.as_type.a42, (constants.%Negate.lookup_impl_witness.eaf) [symbolic = %Negate.facet.loc296_3.3 (constants.%Negate.facet.473)]
// CHECK:STDOUT:     %.loc296_3.1: %Negate.type = converted constants.%T.as_type.a42, %Negate.facet.loc296_3.1 [symbolic = %Negate.facet.loc296_3.3 (constants.%Negate.facet.473)]
// CHECK:STDOUT:     %Negate.facet.loc296_3.2: %Negate.type = facet_value constants.%T.as_type.a42, (constants.%Negate.lookup_impl_witness.eaf) [symbolic = %Negate.facet.loc296_3.3 (constants.%Negate.facet.473)]
// CHECK:STDOUT:     %.loc296_3.2: %Negate.type = converted constants.%T.as_type.a42, %Negate.facet.loc296_3.2 [symbolic = %Negate.facet.loc296_3.3 (constants.%Negate.facet.473)]
// CHECK:STDOUT:     %specific_impl_fn.loc296_3.1: <specific function> = specific_impl_function %impl.elem1.loc296_3.1, @Negate.WithSelf.Op(constants.%Negate.facet.473) [symbolic = %specific_impl_fn.loc296_3.2 (constants.%specific_impl_fn.511)]
// CHECK:STDOUT:     %bound_method.loc296_3.2: <bound method> = bound_method %x.ref, %specific_impl_fn.loc296_3.1
// CHECK:STDOUT:     %.loc296_3.3: ref @TestNegate.%Result.as_type.loc293_49.1 (%Result.as_type) = temporary_storage
// CHECK:STDOUT:     %Negate.WithSelf.Op.call: init @TestNegate.%Result.as_type.loc293_49.1 (%Result.as_type) to %.loc296_3.3 = call %bound_method.loc296_3.2(%x.ref)
// CHECK:STDOUT:     %.loc296_3.4: ref @TestNegate.%Result.as_type.loc293_49.1 (%Result.as_type) = temporary %.loc296_3.3, %Negate.WithSelf.Op.call
// CHECK:STDOUT:     %.loc296_3.5: %facet_type.ece = converted constants.%Result.as_type, constants.%Result [symbolic = %Result.loc292_9.1 (constants.%Result)]
// CHECK:STDOUT:     %.loc296_3.6: %facet_type.ece = converted constants.%Result.as_type, constants.%Result [symbolic = %Result.loc292_9.1 (constants.%Result)]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc296_3.4)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AddWith.loc124(constants.%Result, constants.%U.84b, constants.%T.d3f) {
// CHECK:STDOUT:   %Result.patt.loc121_9.2 => constants.%Result.patt
// CHECK:STDOUT:   %Result.loc121_9.1 => constants.%Result
// CHECK:STDOUT:   %U.patt.loc122_4.2 => constants.%U.patt.4c9
// CHECK:STDOUT:   %U.loc122_4.1 => constants.%U.84b
// CHECK:STDOUT:   %U.as_type.loc123_35.1 => constants.%U.as_type.15c
// CHECK:STDOUT:   %AddWith.type.loc123_35.1 => constants.%AddWith.type.55e
// CHECK:STDOUT:   %facet_type.loc123_19 => constants.%facet_type.39b
// CHECK:STDOUT:   %.Self.frozen.loc123_37.1 => constants.%.Self.frozen.c40
// CHECK:STDOUT:   %require_complete.loc123 => constants.%require_complete.1b3
// CHECK:STDOUT:   %.Self.frozen.as_type => constants.%.Self.frozen.as_type.459
// CHECK:STDOUT:   %AddWith.assoc_type => constants.%AddWith.assoc_type.cc7
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.920
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.loc123_43.1 => constants.%AddWith.lookup_impl_witness.2e9
// CHECK:STDOUT:   %impl.elem0.loc123_43.1 => constants.%impl.elem0.58a
// CHECK:STDOUT:   %Result.as_type.loc123_53.1 => constants.%Result.as_type
// CHECK:STDOUT:   %.Self => constants.%.Self.878
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.loc123_43.2 => constants.%AddWith.lookup_impl_witness.cf2
// CHECK:STDOUT:   %impl.elem0.loc123_43.2 => constants.%impl.elem0.f49
// CHECK:STDOUT:   %facet_type.loc123_37 => constants.%facet_type.0bd
// CHECK:STDOUT:   %pattern_type.loc123 => constants.%pattern_type.e9f
// CHECK:STDOUT:   %T.patt.loc123_4.2 => constants.%T.patt.35d
// CHECK:STDOUT:   %T.loc123_4.1 => constants.%T.d3f
// CHECK:STDOUT:   %pattern_type.loc124_16 => constants.%pattern_type.e4c132.1
// CHECK:STDOUT:   %result.param_patt.loc124_16.2 => constants.%result.param_patt.b19
// CHECK:STDOUT:   %result.patt.loc124_16.2 => constants.%result.patt.583fb4.1
// CHECK:STDOUT:   %T.as_type.loc124_29.1 => constants.%T.as_type.d99
// CHECK:STDOUT:   %pattern_type.loc124_27 => constants.%pattern_type.734
// CHECK:STDOUT:   %x.param_patt.loc124_27.2 => constants.%x.param_patt.17a
// CHECK:STDOUT:   %x.patt.loc124_27.2 => constants.%x.patt.982
// CHECK:STDOUT:   %pattern_type.loc124_33 => constants.%pattern_type.d0c4
// CHECK:STDOUT:   %y.param_patt.loc124_33.2 => constants.%y.param_patt.14c
// CHECK:STDOUT:   %y.patt.loc124_33.2 => constants.%y.patt.efa
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @SubWith.loc135(constants.%Result, constants.%U.84b, constants.%T.b27) {
// CHECK:STDOUT:   %Result.patt.loc132_9.2 => constants.%Result.patt
// CHECK:STDOUT:   %Result.loc132_9.1 => constants.%Result
// CHECK:STDOUT:   %U.patt.loc133_4.2 => constants.%U.patt.4c9
// CHECK:STDOUT:   %U.loc133_4.1 => constants.%U.84b
// CHECK:STDOUT:   %U.as_type.loc134_35.1 => constants.%U.as_type.15c
// CHECK:STDOUT:   %SubWith.type.loc134_35.1 => constants.%SubWith.type.776
// CHECK:STDOUT:   %facet_type.loc134_19 => constants.%facet_type.8f9
// CHECK:STDOUT:   %.Self.frozen.loc134_37.1 => constants.%.Self.frozen.ca0
// CHECK:STDOUT:   %require_complete.loc134 => constants.%require_complete.b4d
// CHECK:STDOUT:   %.Self.frozen.as_type => constants.%.Self.frozen.as_type.ff3
// CHECK:STDOUT:   %SubWith.assoc_type => constants.%SubWith.assoc_type.cf4
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.963
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.loc134_43.1 => constants.%SubWith.lookup_impl_witness.909
// CHECK:STDOUT:   %impl.elem0.loc134_43.1 => constants.%impl.elem0.92e
// CHECK:STDOUT:   %Result.as_type.loc134_53.1 => constants.%Result.as_type
// CHECK:STDOUT:   %.Self => constants.%.Self.355
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.loc134_43.2 => constants.%SubWith.lookup_impl_witness.9ec
// CHECK:STDOUT:   %impl.elem0.loc134_43.2 => constants.%impl.elem0.f58
// CHECK:STDOUT:   %facet_type.loc134_37 => constants.%facet_type.952
// CHECK:STDOUT:   %pattern_type.loc134 => constants.%pattern_type.70e
// CHECK:STDOUT:   %T.patt.loc134_4.2 => constants.%T.patt.b38
// CHECK:STDOUT:   %T.loc134_4.1 => constants.%T.b27
// CHECK:STDOUT:   %pattern_type.loc135_16 => constants.%pattern_type.e4c132.1
// CHECK:STDOUT:   %result.param_patt.loc135_16.2 => constants.%result.param_patt.b19
// CHECK:STDOUT:   %result.patt.loc135_16.2 => constants.%result.patt.583fb4.1
// CHECK:STDOUT:   %T.as_type.loc135_29.1 => constants.%T.as_type.137
// CHECK:STDOUT:   %pattern_type.loc135_27 => constants.%pattern_type.6b7
// CHECK:STDOUT:   %x.param_patt.loc135_27.2 => constants.%x.param_patt.1ce
// CHECK:STDOUT:   %x.patt.loc135_27.2 => constants.%x.patt.ab5
// CHECK:STDOUT:   %pattern_type.loc135_33 => constants.%pattern_type.d0c4
// CHECK:STDOUT:   %y.param_patt.loc135_33.2 => constants.%y.param_patt.14c
// CHECK:STDOUT:   %y.patt.loc135_33.2 => constants.%y.patt.efa
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @MulWith.loc146(constants.%Result, constants.%U.84b, constants.%T.470) {
// CHECK:STDOUT:   %Result.patt.loc143_9.2 => constants.%Result.patt
// CHECK:STDOUT:   %Result.loc143_9.1 => constants.%Result
// CHECK:STDOUT:   %U.patt.loc144_4.2 => constants.%U.patt.4c9
// CHECK:STDOUT:   %U.loc144_4.1 => constants.%U.84b
// CHECK:STDOUT:   %U.as_type.loc145_35.1 => constants.%U.as_type.15c
// CHECK:STDOUT:   %MulWith.type.loc145_35.1 => constants.%MulWith.type.068
// CHECK:STDOUT:   %facet_type.loc145_19 => constants.%facet_type.a35
// CHECK:STDOUT:   %.Self.frozen.loc145_37.1 => constants.%.Self.frozen.87a
// CHECK:STDOUT:   %require_complete.loc145 => constants.%require_complete.8d9
// CHECK:STDOUT:   %.Self.frozen.as_type => constants.%.Self.frozen.as_type.1de
// CHECK:STDOUT:   %MulWith.assoc_type => constants.%MulWith.assoc_type.083
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.882
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.loc145_43.1 => constants.%MulWith.lookup_impl_witness.b96
// CHECK:STDOUT:   %impl.elem0.loc145_43.1 => constants.%impl.elem0.25a
// CHECK:STDOUT:   %Result.as_type.loc145_53.1 => constants.%Result.as_type
// CHECK:STDOUT:   %.Self => constants.%.Self.1a7
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.loc145_43.2 => constants.%MulWith.lookup_impl_witness.7c7
// CHECK:STDOUT:   %impl.elem0.loc145_43.2 => constants.%impl.elem0.155
// CHECK:STDOUT:   %facet_type.loc145_37 => constants.%facet_type.6e8
// CHECK:STDOUT:   %pattern_type.loc145 => constants.%pattern_type.8df
// CHECK:STDOUT:   %T.patt.loc145_4.2 => constants.%T.patt.35a
// CHECK:STDOUT:   %T.loc145_4.1 => constants.%T.470
// CHECK:STDOUT:   %pattern_type.loc146_16 => constants.%pattern_type.e4c132.1
// CHECK:STDOUT:   %result.param_patt.loc146_16.2 => constants.%result.param_patt.b19
// CHECK:STDOUT:   %result.patt.loc146_16.2 => constants.%result.patt.583fb4.1
// CHECK:STDOUT:   %T.as_type.loc146_29.1 => constants.%T.as_type.d42
// CHECK:STDOUT:   %pattern_type.loc146_27 => constants.%pattern_type.328
// CHECK:STDOUT:   %x.param_patt.loc146_27.2 => constants.%x.param_patt.066
// CHECK:STDOUT:   %x.patt.loc146_27.2 => constants.%x.patt.f71
// CHECK:STDOUT:   %pattern_type.loc146_33 => constants.%pattern_type.d0c4
// CHECK:STDOUT:   %y.param_patt.loc146_33.2 => constants.%y.param_patt.14c
// CHECK:STDOUT:   %y.patt.loc146_33.2 => constants.%y.patt.efa
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @DivWith.loc157(constants.%Result, constants.%U.84b, constants.%T.9b4) {
// CHECK:STDOUT:   %Result.patt.loc154_9.2 => constants.%Result.patt
// CHECK:STDOUT:   %Result.loc154_9.1 => constants.%Result
// CHECK:STDOUT:   %U.patt.loc155_4.2 => constants.%U.patt.4c9
// CHECK:STDOUT:   %U.loc155_4.1 => constants.%U.84b
// CHECK:STDOUT:   %U.as_type.loc156_35.1 => constants.%U.as_type.15c
// CHECK:STDOUT:   %DivWith.type.loc156_35.1 => constants.%DivWith.type.687
// CHECK:STDOUT:   %facet_type.loc156_19 => constants.%facet_type.519
// CHECK:STDOUT:   %.Self.frozen.loc156_37.1 => constants.%.Self.frozen.2fc
// CHECK:STDOUT:   %require_complete.loc156 => constants.%require_complete.660
// CHECK:STDOUT:   %.Self.frozen.as_type => constants.%.Self.frozen.as_type.11b
// CHECK:STDOUT:   %DivWith.assoc_type => constants.%DivWith.assoc_type.9a3
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.913
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.loc156_43.1 => constants.%DivWith.lookup_impl_witness.354
// CHECK:STDOUT:   %impl.elem0.loc156_43.1 => constants.%impl.elem0.dfa
// CHECK:STDOUT:   %Result.as_type.loc156_53.1 => constants.%Result.as_type
// CHECK:STDOUT:   %.Self => constants.%.Self.310
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.loc156_43.2 => constants.%DivWith.lookup_impl_witness.3de
// CHECK:STDOUT:   %impl.elem0.loc156_43.2 => constants.%impl.elem0.e74
// CHECK:STDOUT:   %facet_type.loc156_37 => constants.%facet_type.344
// CHECK:STDOUT:   %pattern_type.loc156 => constants.%pattern_type.a7c
// CHECK:STDOUT:   %T.patt.loc156_4.2 => constants.%T.patt.436
// CHECK:STDOUT:   %T.loc156_4.1 => constants.%T.9b4
// CHECK:STDOUT:   %pattern_type.loc157_16 => constants.%pattern_type.e4c132.1
// CHECK:STDOUT:   %result.param_patt.loc157_16.2 => constants.%result.param_patt.b19
// CHECK:STDOUT:   %result.patt.loc157_16.2 => constants.%result.patt.583fb4.1
// CHECK:STDOUT:   %T.as_type.loc157_29.1 => constants.%T.as_type.6cf
// CHECK:STDOUT:   %pattern_type.loc157_27 => constants.%pattern_type.980
// CHECK:STDOUT:   %x.param_patt.loc157_27.2 => constants.%x.param_patt.cd2
// CHECK:STDOUT:   %x.patt.loc157_27.2 => constants.%x.patt.d99
// CHECK:STDOUT:   %pattern_type.loc157_33 => constants.%pattern_type.d0c4
// CHECK:STDOUT:   %y.param_patt.loc157_33.2 => constants.%y.param_patt.14c
// CHECK:STDOUT:   %y.patt.loc157_33.2 => constants.%y.patt.efa
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ModWith.loc168(constants.%Result, constants.%U.84b, constants.%T.af4) {
// CHECK:STDOUT:   %Result.patt.loc165_9.2 => constants.%Result.patt
// CHECK:STDOUT:   %Result.loc165_9.1 => constants.%Result
// CHECK:STDOUT:   %U.patt.loc166_4.2 => constants.%U.patt.4c9
// CHECK:STDOUT:   %U.loc166_4.1 => constants.%U.84b
// CHECK:STDOUT:   %U.as_type.loc167_35.1 => constants.%U.as_type.15c
// CHECK:STDOUT:   %ModWith.type.loc167_35.1 => constants.%ModWith.type.584
// CHECK:STDOUT:   %facet_type.loc167_19 => constants.%facet_type.baa
// CHECK:STDOUT:   %.Self.frozen.loc167_37.1 => constants.%.Self.frozen.d89
// CHECK:STDOUT:   %require_complete.loc167 => constants.%require_complete.1f1
// CHECK:STDOUT:   %.Self.frozen.as_type => constants.%.Self.frozen.as_type.49a
// CHECK:STDOUT:   %ModWith.assoc_type => constants.%ModWith.assoc_type.279
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.431
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.loc167_43.1 => constants.%ModWith.lookup_impl_witness.966
// CHECK:STDOUT:   %impl.elem0.loc167_43.1 => constants.%impl.elem0.c4e
// CHECK:STDOUT:   %Result.as_type.loc167_53.1 => constants.%Result.as_type
// CHECK:STDOUT:   %.Self => constants.%.Self.055
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.loc167_43.2 => constants.%ModWith.lookup_impl_witness.041
// CHECK:STDOUT:   %impl.elem0.loc167_43.2 => constants.%impl.elem0.269
// CHECK:STDOUT:   %facet_type.loc167_37 => constants.%facet_type.668
// CHECK:STDOUT:   %pattern_type.loc167 => constants.%pattern_type.3c9
// CHECK:STDOUT:   %T.patt.loc167_4.2 => constants.%T.patt.d3a
// CHECK:STDOUT:   %T.loc167_4.1 => constants.%T.af4
// CHECK:STDOUT:   %pattern_type.loc168_16 => constants.%pattern_type.e4c132.1
// CHECK:STDOUT:   %result.param_patt.loc168_16.2 => constants.%result.param_patt.b19
// CHECK:STDOUT:   %result.patt.loc168_16.2 => constants.%result.patt.583fb4.1
// CHECK:STDOUT:   %T.as_type.loc168_29.1 => constants.%T.as_type.a02
// CHECK:STDOUT:   %pattern_type.loc168_27 => constants.%pattern_type.72e
// CHECK:STDOUT:   %x.param_patt.loc168_27.2 => constants.%x.param_patt.dc7
// CHECK:STDOUT:   %x.patt.loc168_27.2 => constants.%x.patt.a0b
// CHECK:STDOUT:   %pattern_type.loc168_33 => constants.%pattern_type.d0c4
// CHECK:STDOUT:   %y.param_patt.loc168_33.2 => constants.%y.param_patt.14c
// CHECK:STDOUT:   %y.patt.loc168_33.2 => constants.%y.patt.efa
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AddWith.loc124(constants.%facet_value.a87, constants.%facet_value.a87, constants.%facet_value.d2f) {
// CHECK:STDOUT:   %Result.patt.loc121_9.2 => constants.%Result.patt
// CHECK:STDOUT:   %Result.loc121_9.1 => constants.%facet_value.a87
// CHECK:STDOUT:   %U.patt.loc122_4.2 => constants.%U.patt.4c9
// CHECK:STDOUT:   %U.loc122_4.1 => constants.%facet_value.a87
// CHECK:STDOUT:   %U.as_type.loc123_35.1 => constants.%Int16
// CHECK:STDOUT:   %AddWith.type.loc123_35.1 => constants.%AddWith.type.e4b
// CHECK:STDOUT:   %facet_type.loc123_19 => constants.%facet_type.299
// CHECK:STDOUT:   %.Self.frozen.loc123_37.1 => constants.%.Self.frozen.f66
// CHECK:STDOUT:   %require_complete.loc123 => constants.%complete_type.6e0
// CHECK:STDOUT:   %.Self.frozen.as_type => constants.%.Self.frozen.as_type.ea2
// CHECK:STDOUT:   %AddWith.assoc_type => constants.%AddWith.assoc_type.86e
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.30c
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.loc123_43.1 => constants.%AddWith.lookup_impl_witness.451
// CHECK:STDOUT:   %impl.elem0.loc123_43.1 => constants.%impl.elem0.f96
// CHECK:STDOUT:   %Result.as_type.loc123_53.1 => constants.%Int16
// CHECK:STDOUT:   %.Self => constants.%.Self.32a
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.loc123_43.2 => constants.%AddWith.lookup_impl_witness.da9
// CHECK:STDOUT:   %impl.elem0.loc123_43.2 => constants.%impl.elem0.4ef
// CHECK:STDOUT:   %facet_type.loc123_37 => constants.%facet_type.3a6
// CHECK:STDOUT:   %pattern_type.loc123 => constants.%pattern_type.26d
// CHECK:STDOUT:   %T.patt.loc123_4.2 => constants.%T.patt.a9b
// CHECK:STDOUT:   %T.loc123_4.1 => constants.%facet_value.d2f
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
// CHECK:STDOUT:   %require_complete.loc126 => constants.%complete_type.f37
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.9bf
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.loc126 => constants.%custom_witness.cba
// CHECK:STDOUT:   %AddWith.facet => constants.%AddWith.facet.2ce
// CHECK:STDOUT:   %AddWith.WithSelf.Op.type => constants.%AddWith.WithSelf.Op.type.4a0
// CHECK:STDOUT:   %.loc126_5.6 => constants.%.7bc
// CHECK:STDOUT:   %impl.elem1.loc126_5.2 => constants.%Op.72717c.1
// CHECK:STDOUT:   %specific_impl_fn.loc126_5.2 => constants.%Op.72717c.1
// CHECK:STDOUT:   %SelfDestruct.specific_fn.2 => constants.%SelfDestruct.specific_fn.65d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @SubWith.loc135(constants.%facet_value.a87, constants.%facet_value.a87, constants.%facet_value.4fe) {
// CHECK:STDOUT:   %Result.patt.loc132_9.2 => constants.%Result.patt
// CHECK:STDOUT:   %Result.loc132_9.1 => constants.%facet_value.a87
// CHECK:STDOUT:   %U.patt.loc133_4.2 => constants.%U.patt.4c9
// CHECK:STDOUT:   %U.loc133_4.1 => constants.%facet_value.a87
// CHECK:STDOUT:   %U.as_type.loc134_35.1 => constants.%Int16
// CHECK:STDOUT:   %SubWith.type.loc134_35.1 => constants.%SubWith.type.68d
// CHECK:STDOUT:   %facet_type.loc134_19 => constants.%facet_type.eed
// CHECK:STDOUT:   %.Self.frozen.loc134_37.1 => constants.%.Self.frozen.669
// CHECK:STDOUT:   %require_complete.loc134 => constants.%complete_type.5eb
// CHECK:STDOUT:   %.Self.frozen.as_type => constants.%.Self.frozen.as_type.aff
// CHECK:STDOUT:   %SubWith.assoc_type => constants.%SubWith.assoc_type.dd6
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.dfb
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.loc134_43.1 => constants.%SubWith.lookup_impl_witness.46e
// CHECK:STDOUT:   %impl.elem0.loc134_43.1 => constants.%impl.elem0.168
// CHECK:STDOUT:   %Result.as_type.loc134_53.1 => constants.%Int16
// CHECK:STDOUT:   %.Self => constants.%.Self.3d7
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.loc134_43.2 => constants.%SubWith.lookup_impl_witness.dde
// CHECK:STDOUT:   %impl.elem0.loc134_43.2 => constants.%impl.elem0.cb3
// CHECK:STDOUT:   %facet_type.loc134_37 => constants.%facet_type.129
// CHECK:STDOUT:   %pattern_type.loc134 => constants.%pattern_type.d0c7
// CHECK:STDOUT:   %T.patt.loc134_4.2 => constants.%T.patt.67c
// CHECK:STDOUT:   %T.loc134_4.1 => constants.%facet_value.4fe
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
// CHECK:STDOUT:   %require_complete.loc137 => constants.%complete_type.8a2
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.ada
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.loc137 => constants.%custom_witness.354
// CHECK:STDOUT:   %SubWith.facet => constants.%SubWith.facet.1ef
// CHECK:STDOUT:   %SubWith.WithSelf.Op.type => constants.%SubWith.WithSelf.Op.type.23f
// CHECK:STDOUT:   %.loc137_5.6 => constants.%.eed
// CHECK:STDOUT:   %impl.elem1.loc137_5.2 => constants.%Op.72717c.2
// CHECK:STDOUT:   %specific_impl_fn.loc137_5.2 => constants.%Op.72717c.2
// CHECK:STDOUT:   %SelfDestruct.specific_fn.2 => constants.%SelfDestruct.specific_fn.65d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @MulWith.loc146(constants.%facet_value.a87, constants.%facet_value.a87, constants.%facet_value.31a) {
// CHECK:STDOUT:   %Result.patt.loc143_9.2 => constants.%Result.patt
// CHECK:STDOUT:   %Result.loc143_9.1 => constants.%facet_value.a87
// CHECK:STDOUT:   %U.patt.loc144_4.2 => constants.%U.patt.4c9
// CHECK:STDOUT:   %U.loc144_4.1 => constants.%facet_value.a87
// CHECK:STDOUT:   %U.as_type.loc145_35.1 => constants.%Int16
// CHECK:STDOUT:   %MulWith.type.loc145_35.1 => constants.%MulWith.type.f65
// CHECK:STDOUT:   %facet_type.loc145_19 => constants.%facet_type.eda
// CHECK:STDOUT:   %.Self.frozen.loc145_37.1 => constants.%.Self.frozen.8ef
// CHECK:STDOUT:   %require_complete.loc145 => constants.%complete_type.57a
// CHECK:STDOUT:   %.Self.frozen.as_type => constants.%.Self.frozen.as_type.29d
// CHECK:STDOUT:   %MulWith.assoc_type => constants.%MulWith.assoc_type.4c3
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.6ec
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.loc145_43.1 => constants.%MulWith.lookup_impl_witness.8f6
// CHECK:STDOUT:   %impl.elem0.loc145_43.1 => constants.%impl.elem0.0af
// CHECK:STDOUT:   %Result.as_type.loc145_53.1 => constants.%Int16
// CHECK:STDOUT:   %.Self => constants.%.Self.a47
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.loc145_43.2 => constants.%MulWith.lookup_impl_witness.923
// CHECK:STDOUT:   %impl.elem0.loc145_43.2 => constants.%impl.elem0.567
// CHECK:STDOUT:   %facet_type.loc145_37 => constants.%facet_type.2c7
// CHECK:STDOUT:   %pattern_type.loc145 => constants.%pattern_type.fcc
// CHECK:STDOUT:   %T.patt.loc145_4.2 => constants.%T.patt.14a
// CHECK:STDOUT:   %T.loc145_4.1 => constants.%facet_value.31a
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
// CHECK:STDOUT:   %require_complete.loc148 => constants.%complete_type.8c6
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.b14
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.loc148 => constants.%custom_witness.3c3
// CHECK:STDOUT:   %MulWith.facet => constants.%MulWith.facet.d84
// CHECK:STDOUT:   %MulWith.WithSelf.Op.type => constants.%MulWith.WithSelf.Op.type.131
// CHECK:STDOUT:   %.loc148_5.6 => constants.%.010
// CHECK:STDOUT:   %impl.elem1.loc148_5.2 => constants.%Op.72717c.3
// CHECK:STDOUT:   %specific_impl_fn.loc148_5.2 => constants.%Op.72717c.3
// CHECK:STDOUT:   %SelfDestruct.specific_fn.2 => constants.%SelfDestruct.specific_fn.65d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @DivWith.loc157(constants.%facet_value.a87, constants.%facet_value.a87, constants.%facet_value.b4a) {
// CHECK:STDOUT:   %Result.patt.loc154_9.2 => constants.%Result.patt
// CHECK:STDOUT:   %Result.loc154_9.1 => constants.%facet_value.a87
// CHECK:STDOUT:   %U.patt.loc155_4.2 => constants.%U.patt.4c9
// CHECK:STDOUT:   %U.loc155_4.1 => constants.%facet_value.a87
// CHECK:STDOUT:   %U.as_type.loc156_35.1 => constants.%Int16
// CHECK:STDOUT:   %DivWith.type.loc156_35.1 => constants.%DivWith.type.b8a
// CHECK:STDOUT:   %facet_type.loc156_19 => constants.%facet_type.d6d
// CHECK:STDOUT:   %.Self.frozen.loc156_37.1 => constants.%.Self.frozen.df7
// CHECK:STDOUT:   %require_complete.loc156 => constants.%complete_type.e4f
// CHECK:STDOUT:   %.Self.frozen.as_type => constants.%.Self.frozen.as_type.cfb
// CHECK:STDOUT:   %DivWith.assoc_type => constants.%DivWith.assoc_type.712
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.99c
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.loc156_43.1 => constants.%DivWith.lookup_impl_witness.1bb
// CHECK:STDOUT:   %impl.elem0.loc156_43.1 => constants.%impl.elem0.729
// CHECK:STDOUT:   %Result.as_type.loc156_53.1 => constants.%Int16
// CHECK:STDOUT:   %.Self => constants.%.Self.1ea
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.loc156_43.2 => constants.%DivWith.lookup_impl_witness.5ff
// CHECK:STDOUT:   %impl.elem0.loc156_43.2 => constants.%impl.elem0.416
// CHECK:STDOUT:   %facet_type.loc156_37 => constants.%facet_type.403
// CHECK:STDOUT:   %pattern_type.loc156 => constants.%pattern_type.245b
// CHECK:STDOUT:   %T.patt.loc156_4.2 => constants.%T.patt.415
// CHECK:STDOUT:   %T.loc156_4.1 => constants.%facet_value.b4a
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
// CHECK:STDOUT:   %require_complete.loc159 => constants.%complete_type.d99
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.fb6
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.loc159 => constants.%custom_witness.8d0
// CHECK:STDOUT:   %DivWith.facet => constants.%DivWith.facet.4ad
// CHECK:STDOUT:   %DivWith.WithSelf.Op.type => constants.%DivWith.WithSelf.Op.type.edb
// CHECK:STDOUT:   %.loc159_5.6 => constants.%.b5e
// CHECK:STDOUT:   %impl.elem1.loc159_5.2 => constants.%Op.72717c.4
// CHECK:STDOUT:   %specific_impl_fn.loc159_5.2 => constants.%Op.72717c.4
// CHECK:STDOUT:   %SelfDestruct.specific_fn.2 => constants.%SelfDestruct.specific_fn.65d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ModWith.loc168(constants.%facet_value.a87, constants.%facet_value.a87, constants.%facet_value.bff) {
// CHECK:STDOUT:   %Result.patt.loc165_9.2 => constants.%Result.patt
// CHECK:STDOUT:   %Result.loc165_9.1 => constants.%facet_value.a87
// CHECK:STDOUT:   %U.patt.loc166_4.2 => constants.%U.patt.4c9
// CHECK:STDOUT:   %U.loc166_4.1 => constants.%facet_value.a87
// CHECK:STDOUT:   %U.as_type.loc167_35.1 => constants.%Int16
// CHECK:STDOUT:   %ModWith.type.loc167_35.1 => constants.%ModWith.type.c67
// CHECK:STDOUT:   %facet_type.loc167_19 => constants.%facet_type.8eb
// CHECK:STDOUT:   %.Self.frozen.loc167_37.1 => constants.%.Self.frozen.fb4
// CHECK:STDOUT:   %require_complete.loc167 => constants.%complete_type.94f
// CHECK:STDOUT:   %.Self.frozen.as_type => constants.%.Self.frozen.as_type.260
// CHECK:STDOUT:   %ModWith.assoc_type => constants.%ModWith.assoc_type.5f6
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.f04
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.loc167_43.1 => constants.%ModWith.lookup_impl_witness.679
// CHECK:STDOUT:   %impl.elem0.loc167_43.1 => constants.%impl.elem0.64e
// CHECK:STDOUT:   %Result.as_type.loc167_53.1 => constants.%Int16
// CHECK:STDOUT:   %.Self => constants.%.Self.806
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.loc167_43.2 => constants.%ModWith.lookup_impl_witness.f9a
// CHECK:STDOUT:   %impl.elem0.loc167_43.2 => constants.%impl.elem0.507
// CHECK:STDOUT:   %facet_type.loc167_37 => constants.%facet_type.399
// CHECK:STDOUT:   %pattern_type.loc167 => constants.%pattern_type.8c9
// CHECK:STDOUT:   %T.patt.loc167_4.2 => constants.%T.patt.f92
// CHECK:STDOUT:   %T.loc167_4.1 => constants.%facet_value.bff
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
// CHECK:STDOUT:   %require_complete.loc170 => constants.%complete_type.7d8
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.d19
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.loc170 => constants.%custom_witness.169
// CHECK:STDOUT:   %ModWith.facet => constants.%ModWith.facet.ab1
// CHECK:STDOUT:   %ModWith.WithSelf.Op.type => constants.%ModWith.WithSelf.Op.type.e29
// CHECK:STDOUT:   %.loc170_5.6 => constants.%.d62
// CHECK:STDOUT:   %impl.elem1.loc170_5.2 => constants.%Op.72717c.5
// CHECK:STDOUT:   %specific_impl_fn.loc170_5.2 => constants.%Op.72717c.5
// CHECK:STDOUT:   %SelfDestruct.specific_fn.2 => constants.%SelfDestruct.specific_fn.65d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AddWith.loc124(constants.%facet_value.a87, constants.%facet_value.e48, constants.%facet_value.39b) {
// CHECK:STDOUT:   %Result.patt.loc121_9.2 => constants.%Result.patt
// CHECK:STDOUT:   %Result.loc121_9.1 => constants.%facet_value.a87
// CHECK:STDOUT:   %U.patt.loc122_4.2 => constants.%U.patt.4c9
// CHECK:STDOUT:   %U.loc122_4.1 => constants.%facet_value.e48
// CHECK:STDOUT:   %U.as_type.loc123_35.1 => constants.%i16
// CHECK:STDOUT:   %AddWith.type.loc123_35.1 => constants.%AddWith.type.2ed
// CHECK:STDOUT:   %facet_type.loc123_19 => constants.%facet_type.b35
// CHECK:STDOUT:   %.Self.frozen.loc123_37.1 => constants.%.Self.frozen.aef
// CHECK:STDOUT:   %require_complete.loc123 => constants.%complete_type.aee
// CHECK:STDOUT:   %.Self.frozen.as_type => constants.%.Self.frozen.as_type.a48
// CHECK:STDOUT:   %AddWith.assoc_type => constants.%AddWith.assoc_type.18f
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.b7d
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.loc123_43.1 => constants.%AddWith.lookup_impl_witness.5f1
// CHECK:STDOUT:   %impl.elem0.loc123_43.1 => constants.%impl.elem0.877
// CHECK:STDOUT:   %Result.as_type.loc123_53.1 => constants.%Int16
// CHECK:STDOUT:   %.Self => constants.%.Self.a3e
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.loc123_43.2 => constants.%AddWith.lookup_impl_witness.05d
// CHECK:STDOUT:   %impl.elem0.loc123_43.2 => constants.%impl.elem0.01e
// CHECK:STDOUT:   %facet_type.loc123_37 => constants.%facet_type.676
// CHECK:STDOUT:   %pattern_type.loc123 => constants.%pattern_type.c34
// CHECK:STDOUT:   %T.patt.loc123_4.2 => constants.%T.patt.715
// CHECK:STDOUT:   %T.loc123_4.1 => constants.%facet_value.39b
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
// CHECK:STDOUT:   %require_complete.loc126 => constants.%complete_type.32d
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.e08
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.loc126 => constants.%custom_witness.b79
// CHECK:STDOUT:   %AddWith.facet => constants.%AddWith.facet.11e
// CHECK:STDOUT:   %AddWith.WithSelf.Op.type => constants.%AddWith.WithSelf.Op.type.eb3
// CHECK:STDOUT:   %.loc126_5.6 => constants.%.da8
// CHECK:STDOUT:   %impl.elem1.loc126_5.2 => constants.%Int16.Op.4dca31.2
// CHECK:STDOUT:   %specific_impl_fn.loc126_5.2 => constants.%Int16.Op.4dca31.2
// CHECK:STDOUT:   %SelfDestruct.specific_fn.2 => constants.%SelfDestruct.specific_fn.65d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @SubWith.loc135(constants.%facet_value.a87, constants.%facet_value.e48, constants.%facet_value.666) {
// CHECK:STDOUT:   %Result.patt.loc132_9.2 => constants.%Result.patt
// CHECK:STDOUT:   %Result.loc132_9.1 => constants.%facet_value.a87
// CHECK:STDOUT:   %U.patt.loc133_4.2 => constants.%U.patt.4c9
// CHECK:STDOUT:   %U.loc133_4.1 => constants.%facet_value.e48
// CHECK:STDOUT:   %U.as_type.loc134_35.1 => constants.%i16
// CHECK:STDOUT:   %SubWith.type.loc134_35.1 => constants.%SubWith.type.606
// CHECK:STDOUT:   %facet_type.loc134_19 => constants.%facet_type.8ae
// CHECK:STDOUT:   %.Self.frozen.loc134_37.1 => constants.%.Self.frozen.d2d
// CHECK:STDOUT:   %require_complete.loc134 => constants.%complete_type.a80
// CHECK:STDOUT:   %.Self.frozen.as_type => constants.%.Self.frozen.as_type.d5b
// CHECK:STDOUT:   %SubWith.assoc_type => constants.%SubWith.assoc_type.9c3
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.f66
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.loc134_43.1 => constants.%SubWith.lookup_impl_witness.19e
// CHECK:STDOUT:   %impl.elem0.loc134_43.1 => constants.%impl.elem0.275
// CHECK:STDOUT:   %Result.as_type.loc134_53.1 => constants.%Int16
// CHECK:STDOUT:   %.Self => constants.%.Self.60a
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.loc134_43.2 => constants.%SubWith.lookup_impl_witness.71f
// CHECK:STDOUT:   %impl.elem0.loc134_43.2 => constants.%impl.elem0.499
// CHECK:STDOUT:   %facet_type.loc134_37 => constants.%facet_type.f60
// CHECK:STDOUT:   %pattern_type.loc134 => constants.%pattern_type.2454
// CHECK:STDOUT:   %T.patt.loc134_4.2 => constants.%T.patt.e9d
// CHECK:STDOUT:   %T.loc134_4.1 => constants.%facet_value.666
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
// CHECK:STDOUT:   %require_complete.loc137 => constants.%complete_type.75c
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.144
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.loc137 => constants.%custom_witness.1e8
// CHECK:STDOUT:   %SubWith.facet => constants.%SubWith.facet.6fa
// CHECK:STDOUT:   %SubWith.WithSelf.Op.type => constants.%SubWith.WithSelf.Op.type.bb5
// CHECK:STDOUT:   %.loc137_5.6 => constants.%.42a
// CHECK:STDOUT:   %impl.elem1.loc137_5.2 => constants.%Int16.Op.4dca31.3
// CHECK:STDOUT:   %specific_impl_fn.loc137_5.2 => constants.%Int16.Op.4dca31.3
// CHECK:STDOUT:   %SelfDestruct.specific_fn.2 => constants.%SelfDestruct.specific_fn.65d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @MulWith.loc146(constants.%facet_value.a87, constants.%facet_value.e48, constants.%facet_value.7b6) {
// CHECK:STDOUT:   %Result.patt.loc143_9.2 => constants.%Result.patt
// CHECK:STDOUT:   %Result.loc143_9.1 => constants.%facet_value.a87
// CHECK:STDOUT:   %U.patt.loc144_4.2 => constants.%U.patt.4c9
// CHECK:STDOUT:   %U.loc144_4.1 => constants.%facet_value.e48
// CHECK:STDOUT:   %U.as_type.loc145_35.1 => constants.%i16
// CHECK:STDOUT:   %MulWith.type.loc145_35.1 => constants.%MulWith.type.1e2
// CHECK:STDOUT:   %facet_type.loc145_19 => constants.%facet_type.69d
// CHECK:STDOUT:   %.Self.frozen.loc145_37.1 => constants.%.Self.frozen.c18
// CHECK:STDOUT:   %require_complete.loc145 => constants.%complete_type.d63
// CHECK:STDOUT:   %.Self.frozen.as_type => constants.%.Self.frozen.as_type.d64
// CHECK:STDOUT:   %MulWith.assoc_type => constants.%MulWith.assoc_type.34c
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.2d4
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.loc145_43.1 => constants.%MulWith.lookup_impl_witness.367
// CHECK:STDOUT:   %impl.elem0.loc145_43.1 => constants.%impl.elem0.5f2
// CHECK:STDOUT:   %Result.as_type.loc145_53.1 => constants.%Int16
// CHECK:STDOUT:   %.Self => constants.%.Self.0b6
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.loc145_43.2 => constants.%MulWith.lookup_impl_witness.58f
// CHECK:STDOUT:   %impl.elem0.loc145_43.2 => constants.%impl.elem0.8b8
// CHECK:STDOUT:   %facet_type.loc145_37 => constants.%facet_type.63d
// CHECK:STDOUT:   %pattern_type.loc145 => constants.%pattern_type.563
// CHECK:STDOUT:   %T.patt.loc145_4.2 => constants.%T.patt.b7e
// CHECK:STDOUT:   %T.loc145_4.1 => constants.%facet_value.7b6
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
// CHECK:STDOUT:   %require_complete.loc148 => constants.%complete_type.a7a
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.635
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.loc148 => constants.%custom_witness.e2b
// CHECK:STDOUT:   %MulWith.facet => constants.%MulWith.facet.8b1
// CHECK:STDOUT:   %MulWith.WithSelf.Op.type => constants.%MulWith.WithSelf.Op.type.8f5
// CHECK:STDOUT:   %.loc148_5.6 => constants.%.ecc
// CHECK:STDOUT:   %impl.elem1.loc148_5.2 => constants.%Int16.Op.4dca31.4
// CHECK:STDOUT:   %specific_impl_fn.loc148_5.2 => constants.%Int16.Op.4dca31.4
// CHECK:STDOUT:   %SelfDestruct.specific_fn.2 => constants.%SelfDestruct.specific_fn.65d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @DivWith.loc157(constants.%facet_value.a87, constants.%facet_value.e48, constants.%facet_value.410) {
// CHECK:STDOUT:   %Result.patt.loc154_9.2 => constants.%Result.patt
// CHECK:STDOUT:   %Result.loc154_9.1 => constants.%facet_value.a87
// CHECK:STDOUT:   %U.patt.loc155_4.2 => constants.%U.patt.4c9
// CHECK:STDOUT:   %U.loc155_4.1 => constants.%facet_value.e48
// CHECK:STDOUT:   %U.as_type.loc156_35.1 => constants.%i16
// CHECK:STDOUT:   %DivWith.type.loc156_35.1 => constants.%DivWith.type.45d
// CHECK:STDOUT:   %facet_type.loc156_19 => constants.%facet_type.e3e
// CHECK:STDOUT:   %.Self.frozen.loc156_37.1 => constants.%.Self.frozen.513
// CHECK:STDOUT:   %require_complete.loc156 => constants.%complete_type.97d
// CHECK:STDOUT:   %.Self.frozen.as_type => constants.%.Self.frozen.as_type.9c4
// CHECK:STDOUT:   %DivWith.assoc_type => constants.%DivWith.assoc_type.5a4
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.156
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.loc156_43.1 => constants.%DivWith.lookup_impl_witness.3e1
// CHECK:STDOUT:   %impl.elem0.loc156_43.1 => constants.%impl.elem0.3d6
// CHECK:STDOUT:   %Result.as_type.loc156_53.1 => constants.%Int16
// CHECK:STDOUT:   %.Self => constants.%.Self.d5c
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.loc156_43.2 => constants.%DivWith.lookup_impl_witness.510
// CHECK:STDOUT:   %impl.elem0.loc156_43.2 => constants.%impl.elem0.d69
// CHECK:STDOUT:   %facet_type.loc156_37 => constants.%facet_type.38f
// CHECK:STDOUT:   %pattern_type.loc156 => constants.%pattern_type.5d8
// CHECK:STDOUT:   %T.patt.loc156_4.2 => constants.%T.patt.50f
// CHECK:STDOUT:   %T.loc156_4.1 => constants.%facet_value.410
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
// CHECK:STDOUT:   %require_complete.loc159 => constants.%complete_type.d57
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.444
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.loc159 => constants.%custom_witness.21db
// CHECK:STDOUT:   %DivWith.facet => constants.%DivWith.facet.57c
// CHECK:STDOUT:   %DivWith.WithSelf.Op.type => constants.%DivWith.WithSelf.Op.type.37a
// CHECK:STDOUT:   %.loc159_5.6 => constants.%.5a3
// CHECK:STDOUT:   %impl.elem1.loc159_5.2 => constants.%Int16.Op.4dca31.5
// CHECK:STDOUT:   %specific_impl_fn.loc159_5.2 => constants.%Int16.Op.4dca31.5
// CHECK:STDOUT:   %SelfDestruct.specific_fn.2 => constants.%SelfDestruct.specific_fn.65d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ModWith.loc168(constants.%facet_value.a87, constants.%facet_value.e48, constants.%facet_value.47a) {
// CHECK:STDOUT:   %Result.patt.loc165_9.2 => constants.%Result.patt
// CHECK:STDOUT:   %Result.loc165_9.1 => constants.%facet_value.a87
// CHECK:STDOUT:   %U.patt.loc166_4.2 => constants.%U.patt.4c9
// CHECK:STDOUT:   %U.loc166_4.1 => constants.%facet_value.e48
// CHECK:STDOUT:   %U.as_type.loc167_35.1 => constants.%i16
// CHECK:STDOUT:   %ModWith.type.loc167_35.1 => constants.%ModWith.type.104
// CHECK:STDOUT:   %facet_type.loc167_19 => constants.%facet_type.a81
// CHECK:STDOUT:   %.Self.frozen.loc167_37.1 => constants.%.Self.frozen.68c
// CHECK:STDOUT:   %require_complete.loc167 => constants.%complete_type.cbe
// CHECK:STDOUT:   %.Self.frozen.as_type => constants.%.Self.frozen.as_type.d41
// CHECK:STDOUT:   %ModWith.assoc_type => constants.%ModWith.assoc_type.0e9
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.92c
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.loc167_43.1 => constants.%ModWith.lookup_impl_witness.d2e
// CHECK:STDOUT:   %impl.elem0.loc167_43.1 => constants.%impl.elem0.6f4
// CHECK:STDOUT:   %Result.as_type.loc167_53.1 => constants.%Int16
// CHECK:STDOUT:   %.Self => constants.%.Self.860
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.loc167_43.2 => constants.%ModWith.lookup_impl_witness.e90
// CHECK:STDOUT:   %impl.elem0.loc167_43.2 => constants.%impl.elem0.740
// CHECK:STDOUT:   %facet_type.loc167_37 => constants.%facet_type.283
// CHECK:STDOUT:   %pattern_type.loc167 => constants.%pattern_type.2a1
// CHECK:STDOUT:   %T.patt.loc167_4.2 => constants.%T.patt.4c7
// CHECK:STDOUT:   %T.loc167_4.1 => constants.%facet_value.47a
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
// CHECK:STDOUT:   %require_complete.loc170 => constants.%complete_type.4d7
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.870
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.loc170 => constants.%custom_witness.6dd
// CHECK:STDOUT:   %ModWith.facet => constants.%ModWith.facet.f5a5
// CHECK:STDOUT:   %ModWith.WithSelf.Op.type => constants.%ModWith.WithSelf.Op.type.374
// CHECK:STDOUT:   %.loc170_5.6 => constants.%.52c1
// CHECK:STDOUT:   %impl.elem1.loc170_5.2 => constants.%Int16.Op.4dca31.6
// CHECK:STDOUT:   %specific_impl_fn.loc170_5.2 => constants.%Int16.Op.4dca31.6
// CHECK:STDOUT:   %SelfDestruct.specific_fn.2 => constants.%SelfDestruct.specific_fn.65d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AddWith.loc124(constants.%facet_value.cb7, constants.%facet_value.cb7, constants.%facet_value.312) {
// CHECK:STDOUT:   %Result.patt.loc121_9.2 => constants.%Result.patt
// CHECK:STDOUT:   %Result.loc121_9.1 => constants.%facet_value.cb7
// CHECK:STDOUT:   %U.patt.loc122_4.2 => constants.%U.patt.4c9
// CHECK:STDOUT:   %U.loc122_4.1 => constants.%facet_value.cb7
// CHECK:STDOUT:   %U.as_type.loc123_35.1 => constants.%Int32
// CHECK:STDOUT:   %AddWith.type.loc123_35.1 => constants.%AddWith.type.db7
// CHECK:STDOUT:   %facet_type.loc123_19 => constants.%facet_type.f00
// CHECK:STDOUT:   %.Self.frozen.loc123_37.1 => constants.%.Self.frozen.5ba
// CHECK:STDOUT:   %require_complete.loc123 => constants.%complete_type.3c7
// CHECK:STDOUT:   %.Self.frozen.as_type => constants.%.Self.frozen.as_type.cb7
// CHECK:STDOUT:   %AddWith.assoc_type => constants.%AddWith.assoc_type.be8
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.31c
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.loc123_43.1 => constants.%AddWith.lookup_impl_witness.337
// CHECK:STDOUT:   %impl.elem0.loc123_43.1 => constants.%impl.elem0.14f
// CHECK:STDOUT:   %Result.as_type.loc123_53.1 => constants.%Int32
// CHECK:STDOUT:   %.Self => constants.%.Self.29d
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.loc123_43.2 => constants.%AddWith.lookup_impl_witness.c08
// CHECK:STDOUT:   %impl.elem0.loc123_43.2 => constants.%impl.elem0.49b
// CHECK:STDOUT:   %facet_type.loc123_37 => constants.%facet_type.8ac
// CHECK:STDOUT:   %pattern_type.loc123 => constants.%pattern_type.ff4
// CHECK:STDOUT:   %T.patt.loc123_4.2 => constants.%T.patt.fe3
// CHECK:STDOUT:   %T.loc123_4.1 => constants.%facet_value.312
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
// CHECK:STDOUT:   %require_complete.loc126 => constants.%complete_type.758
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.86c
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.loc126 => constants.%custom_witness.513476.1
// CHECK:STDOUT:   %AddWith.facet => constants.%AddWith.facet.8f8
// CHECK:STDOUT:   %AddWith.WithSelf.Op.type => constants.%AddWith.WithSelf.Op.type.4cf
// CHECK:STDOUT:   %.loc126_5.6 => constants.%.feb
// CHECK:STDOUT:   %impl.elem1.loc126_5.2 => constants.%Int16.Op.4dca31.7
// CHECK:STDOUT:   %specific_impl_fn.loc126_5.2 => constants.%Int16.Op.4dca31.7
// CHECK:STDOUT:   %SelfDestruct.specific_fn.2 => constants.%SelfDestruct.specific_fn.861
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @SubWith.loc135(constants.%facet_value.cb7, constants.%facet_value.cb7, constants.%facet_value.1ab) {
// CHECK:STDOUT:   %Result.patt.loc132_9.2 => constants.%Result.patt
// CHECK:STDOUT:   %Result.loc132_9.1 => constants.%facet_value.cb7
// CHECK:STDOUT:   %U.patt.loc133_4.2 => constants.%U.patt.4c9
// CHECK:STDOUT:   %U.loc133_4.1 => constants.%facet_value.cb7
// CHECK:STDOUT:   %U.as_type.loc134_35.1 => constants.%Int32
// CHECK:STDOUT:   %SubWith.type.loc134_35.1 => constants.%SubWith.type.120
// CHECK:STDOUT:   %facet_type.loc134_19 => constants.%facet_type.4e1
// CHECK:STDOUT:   %.Self.frozen.loc134_37.1 => constants.%.Self.frozen.0f1
// CHECK:STDOUT:   %require_complete.loc134 => constants.%complete_type.ce7
// CHECK:STDOUT:   %.Self.frozen.as_type => constants.%.Self.frozen.as_type.815
// CHECK:STDOUT:   %SubWith.assoc_type => constants.%SubWith.assoc_type.dae
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.fac
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.loc134_43.1 => constants.%SubWith.lookup_impl_witness.5d6
// CHECK:STDOUT:   %impl.elem0.loc134_43.1 => constants.%impl.elem0.1ff
// CHECK:STDOUT:   %Result.as_type.loc134_53.1 => constants.%Int32
// CHECK:STDOUT:   %.Self => constants.%.Self.763
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.loc134_43.2 => constants.%SubWith.lookup_impl_witness.08b
// CHECK:STDOUT:   %impl.elem0.loc134_43.2 => constants.%impl.elem0.496
// CHECK:STDOUT:   %facet_type.loc134_37 => constants.%facet_type.541
// CHECK:STDOUT:   %pattern_type.loc134 => constants.%pattern_type.183
// CHECK:STDOUT:   %T.patt.loc134_4.2 => constants.%T.patt.a86
// CHECK:STDOUT:   %T.loc134_4.1 => constants.%facet_value.1ab
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
// CHECK:STDOUT:   %require_complete.loc137 => constants.%complete_type.46e
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.147
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.loc137 => constants.%custom_witness.32cef1.1
// CHECK:STDOUT:   %SubWith.facet => constants.%SubWith.facet.0b0
// CHECK:STDOUT:   %SubWith.WithSelf.Op.type => constants.%SubWith.WithSelf.Op.type.e1e
// CHECK:STDOUT:   %.loc137_5.6 => constants.%.988
// CHECK:STDOUT:   %impl.elem1.loc137_5.2 => constants.%Int16.Op.4dca31.8
// CHECK:STDOUT:   %specific_impl_fn.loc137_5.2 => constants.%Int16.Op.4dca31.8
// CHECK:STDOUT:   %SelfDestruct.specific_fn.2 => constants.%SelfDestruct.specific_fn.861
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @MulWith.loc146(constants.%facet_value.cb7, constants.%facet_value.cb7, constants.%facet_value.56c) {
// CHECK:STDOUT:   %Result.patt.loc143_9.2 => constants.%Result.patt
// CHECK:STDOUT:   %Result.loc143_9.1 => constants.%facet_value.cb7
// CHECK:STDOUT:   %U.patt.loc144_4.2 => constants.%U.patt.4c9
// CHECK:STDOUT:   %U.loc144_4.1 => constants.%facet_value.cb7
// CHECK:STDOUT:   %U.as_type.loc145_35.1 => constants.%Int32
// CHECK:STDOUT:   %MulWith.type.loc145_35.1 => constants.%MulWith.type.d67
// CHECK:STDOUT:   %facet_type.loc145_19 => constants.%facet_type.89e
// CHECK:STDOUT:   %.Self.frozen.loc145_37.1 => constants.%.Self.frozen.709
// CHECK:STDOUT:   %require_complete.loc145 => constants.%complete_type.a47
// CHECK:STDOUT:   %.Self.frozen.as_type => constants.%.Self.frozen.as_type.93b
// CHECK:STDOUT:   %MulWith.assoc_type => constants.%MulWith.assoc_type.ab1
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.d0a
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.loc145_43.1 => constants.%MulWith.lookup_impl_witness.5dc
// CHECK:STDOUT:   %impl.elem0.loc145_43.1 => constants.%impl.elem0.a0b
// CHECK:STDOUT:   %Result.as_type.loc145_53.1 => constants.%Int32
// CHECK:STDOUT:   %.Self => constants.%.Self.4cb
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.loc145_43.2 => constants.%MulWith.lookup_impl_witness.1b8
// CHECK:STDOUT:   %impl.elem0.loc145_43.2 => constants.%impl.elem0.fa7
// CHECK:STDOUT:   %facet_type.loc145_37 => constants.%facet_type.3b1
// CHECK:STDOUT:   %pattern_type.loc145 => constants.%pattern_type.7e1
// CHECK:STDOUT:   %T.patt.loc145_4.2 => constants.%T.patt.de6
// CHECK:STDOUT:   %T.loc145_4.1 => constants.%facet_value.56c
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
// CHECK:STDOUT:   %require_complete.loc148 => constants.%complete_type.920
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.349
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.loc148 => constants.%custom_witness.3a0675.1
// CHECK:STDOUT:   %MulWith.facet => constants.%MulWith.facet.4b9
// CHECK:STDOUT:   %MulWith.WithSelf.Op.type => constants.%MulWith.WithSelf.Op.type.8f2
// CHECK:STDOUT:   %.loc148_5.6 => constants.%.47a
// CHECK:STDOUT:   %impl.elem1.loc148_5.2 => constants.%Int16.Op.4dca31.9
// CHECK:STDOUT:   %specific_impl_fn.loc148_5.2 => constants.%Int16.Op.4dca31.9
// CHECK:STDOUT:   %SelfDestruct.specific_fn.2 => constants.%SelfDestruct.specific_fn.861
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @DivWith.loc157(constants.%facet_value.cb7, constants.%facet_value.cb7, constants.%facet_value.d5f) {
// CHECK:STDOUT:   %Result.patt.loc154_9.2 => constants.%Result.patt
// CHECK:STDOUT:   %Result.loc154_9.1 => constants.%facet_value.cb7
// CHECK:STDOUT:   %U.patt.loc155_4.2 => constants.%U.patt.4c9
// CHECK:STDOUT:   %U.loc155_4.1 => constants.%facet_value.cb7
// CHECK:STDOUT:   %U.as_type.loc156_35.1 => constants.%Int32
// CHECK:STDOUT:   %DivWith.type.loc156_35.1 => constants.%DivWith.type.5a1
// CHECK:STDOUT:   %facet_type.loc156_19 => constants.%facet_type.2ab
// CHECK:STDOUT:   %.Self.frozen.loc156_37.1 => constants.%.Self.frozen.ee7
// CHECK:STDOUT:   %require_complete.loc156 => constants.%complete_type.f97
// CHECK:STDOUT:   %.Self.frozen.as_type => constants.%.Self.frozen.as_type.7f5
// CHECK:STDOUT:   %DivWith.assoc_type => constants.%DivWith.assoc_type.22f
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.a9d
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.loc156_43.1 => constants.%DivWith.lookup_impl_witness.b80
// CHECK:STDOUT:   %impl.elem0.loc156_43.1 => constants.%impl.elem0.16df
// CHECK:STDOUT:   %Result.as_type.loc156_53.1 => constants.%Int32
// CHECK:STDOUT:   %.Self => constants.%.Self.b0f
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.loc156_43.2 => constants.%DivWith.lookup_impl_witness.38e
// CHECK:STDOUT:   %impl.elem0.loc156_43.2 => constants.%impl.elem0.183
// CHECK:STDOUT:   %facet_type.loc156_37 => constants.%facet_type.50c
// CHECK:STDOUT:   %pattern_type.loc156 => constants.%pattern_type.df6
// CHECK:STDOUT:   %T.patt.loc156_4.2 => constants.%T.patt.c47
// CHECK:STDOUT:   %T.loc156_4.1 => constants.%facet_value.d5f
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
// CHECK:STDOUT:   %require_complete.loc159 => constants.%complete_type.348
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.004
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.loc159 => constants.%custom_witness.c32685.1
// CHECK:STDOUT:   %DivWith.facet => constants.%DivWith.facet.5c5
// CHECK:STDOUT:   %DivWith.WithSelf.Op.type => constants.%DivWith.WithSelf.Op.type.af3
// CHECK:STDOUT:   %.loc159_5.6 => constants.%.836
// CHECK:STDOUT:   %impl.elem1.loc159_5.2 => constants.%Int16.Op.4dca31.10
// CHECK:STDOUT:   %specific_impl_fn.loc159_5.2 => constants.%Int16.Op.4dca31.10
// CHECK:STDOUT:   %SelfDestruct.specific_fn.2 => constants.%SelfDestruct.specific_fn.861
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ModWith.loc168(constants.%facet_value.cb7, constants.%facet_value.cb7, constants.%facet_value.272) {
// CHECK:STDOUT:   %Result.patt.loc165_9.2 => constants.%Result.patt
// CHECK:STDOUT:   %Result.loc165_9.1 => constants.%facet_value.cb7
// CHECK:STDOUT:   %U.patt.loc166_4.2 => constants.%U.patt.4c9
// CHECK:STDOUT:   %U.loc166_4.1 => constants.%facet_value.cb7
// CHECK:STDOUT:   %U.as_type.loc167_35.1 => constants.%Int32
// CHECK:STDOUT:   %ModWith.type.loc167_35.1 => constants.%ModWith.type.794
// CHECK:STDOUT:   %facet_type.loc167_19 => constants.%facet_type.fc7
// CHECK:STDOUT:   %.Self.frozen.loc167_37.1 => constants.%.Self.frozen.f41
// CHECK:STDOUT:   %require_complete.loc167 => constants.%complete_type.0f1
// CHECK:STDOUT:   %.Self.frozen.as_type => constants.%.Self.frozen.as_type.7dc
// CHECK:STDOUT:   %ModWith.assoc_type => constants.%ModWith.assoc_type.021
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.310
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.loc167_43.1 => constants.%ModWith.lookup_impl_witness.f4e
// CHECK:STDOUT:   %impl.elem0.loc167_43.1 => constants.%impl.elem0.3c4
// CHECK:STDOUT:   %Result.as_type.loc167_53.1 => constants.%Int32
// CHECK:STDOUT:   %.Self => constants.%.Self.797
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.loc167_43.2 => constants.%ModWith.lookup_impl_witness.126
// CHECK:STDOUT:   %impl.elem0.loc167_43.2 => constants.%impl.elem0.6bf
// CHECK:STDOUT:   %facet_type.loc167_37 => constants.%facet_type.54d
// CHECK:STDOUT:   %pattern_type.loc167 => constants.%pattern_type.dae
// CHECK:STDOUT:   %T.patt.loc167_4.2 => constants.%T.patt.4d5
// CHECK:STDOUT:   %T.loc167_4.1 => constants.%facet_value.272
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
// CHECK:STDOUT:   %require_complete.loc170 => constants.%complete_type.cc9
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.2eb
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.loc170 => constants.%custom_witness.994c75.1
// CHECK:STDOUT:   %ModWith.facet => constants.%ModWith.facet.cc5
// CHECK:STDOUT:   %ModWith.WithSelf.Op.type => constants.%ModWith.WithSelf.Op.type.ab7
// CHECK:STDOUT:   %.loc170_5.6 => constants.%.2a3
// CHECK:STDOUT:   %impl.elem1.loc170_5.2 => constants.%Int16.Op.4dca31.11
// CHECK:STDOUT:   %specific_impl_fn.loc170_5.2 => constants.%Int16.Op.4dca31.11
// CHECK:STDOUT:   %SelfDestruct.specific_fn.2 => constants.%SelfDestruct.specific_fn.861
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AddWith.loc124(constants.%facet_value.cb7, constants.%facet_value.a87, constants.%facet_value.145) {
// CHECK:STDOUT:   %Result.patt.loc121_9.2 => constants.%Result.patt
// CHECK:STDOUT:   %Result.loc121_9.1 => constants.%facet_value.cb7
// CHECK:STDOUT:   %U.patt.loc122_4.2 => constants.%U.patt.4c9
// CHECK:STDOUT:   %U.loc122_4.1 => constants.%facet_value.a87
// CHECK:STDOUT:   %U.as_type.loc123_35.1 => constants.%Int16
// CHECK:STDOUT:   %AddWith.type.loc123_35.1 => constants.%AddWith.type.e4b
// CHECK:STDOUT:   %facet_type.loc123_19 => constants.%facet_type.299
// CHECK:STDOUT:   %.Self.frozen.loc123_37.1 => constants.%.Self.frozen.f66
// CHECK:STDOUT:   %require_complete.loc123 => constants.%complete_type.6e0
// CHECK:STDOUT:   %.Self.frozen.as_type => constants.%.Self.frozen.as_type.ea2
// CHECK:STDOUT:   %AddWith.assoc_type => constants.%AddWith.assoc_type.86e
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.30c
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.loc123_43.1 => constants.%AddWith.lookup_impl_witness.451
// CHECK:STDOUT:   %impl.elem0.loc123_43.1 => constants.%impl.elem0.f96
// CHECK:STDOUT:   %Result.as_type.loc123_53.1 => constants.%Int32
// CHECK:STDOUT:   %.Self => constants.%.Self.32a
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.loc123_43.2 => constants.%AddWith.lookup_impl_witness.da9
// CHECK:STDOUT:   %impl.elem0.loc123_43.2 => constants.%impl.elem0.4ef
// CHECK:STDOUT:   %facet_type.loc123_37 => constants.%facet_type.ead
// CHECK:STDOUT:   %pattern_type.loc123 => constants.%pattern_type.f8e
// CHECK:STDOUT:   %T.patt.loc123_4.2 => constants.%T.patt.f20
// CHECK:STDOUT:   %T.loc123_4.1 => constants.%facet_value.145
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
// CHECK:STDOUT:   %require_complete.loc126 => constants.%complete_type.f37
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.9bf
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.loc126 => constants.%custom_witness.ea1
// CHECK:STDOUT:   %AddWith.facet => constants.%AddWith.facet.b31
// CHECK:STDOUT:   %AddWith.WithSelf.Op.type => constants.%AddWith.WithSelf.Op.type.faf
// CHECK:STDOUT:   %.loc126_5.6 => constants.%.fd6
// CHECK:STDOUT:   %impl.elem1.loc126_5.2 => constants.%Int32.Op.4dca31.1
// CHECK:STDOUT:   %specific_impl_fn.loc126_5.2 => constants.%Int32.Op.4dca31.1
// CHECK:STDOUT:   %SelfDestruct.specific_fn.2 => constants.%SelfDestruct.specific_fn.861
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @SubWith.loc135(constants.%facet_value.cb7, constants.%facet_value.a87, constants.%facet_value.645) {
// CHECK:STDOUT:   %Result.patt.loc132_9.2 => constants.%Result.patt
// CHECK:STDOUT:   %Result.loc132_9.1 => constants.%facet_value.cb7
// CHECK:STDOUT:   %U.patt.loc133_4.2 => constants.%U.patt.4c9
// CHECK:STDOUT:   %U.loc133_4.1 => constants.%facet_value.a87
// CHECK:STDOUT:   %U.as_type.loc134_35.1 => constants.%Int16
// CHECK:STDOUT:   %SubWith.type.loc134_35.1 => constants.%SubWith.type.68d
// CHECK:STDOUT:   %facet_type.loc134_19 => constants.%facet_type.eed
// CHECK:STDOUT:   %.Self.frozen.loc134_37.1 => constants.%.Self.frozen.669
// CHECK:STDOUT:   %require_complete.loc134 => constants.%complete_type.5eb
// CHECK:STDOUT:   %.Self.frozen.as_type => constants.%.Self.frozen.as_type.aff
// CHECK:STDOUT:   %SubWith.assoc_type => constants.%SubWith.assoc_type.dd6
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.dfb
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.loc134_43.1 => constants.%SubWith.lookup_impl_witness.46e
// CHECK:STDOUT:   %impl.elem0.loc134_43.1 => constants.%impl.elem0.168
// CHECK:STDOUT:   %Result.as_type.loc134_53.1 => constants.%Int32
// CHECK:STDOUT:   %.Self => constants.%.Self.3d7
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.loc134_43.2 => constants.%SubWith.lookup_impl_witness.dde
// CHECK:STDOUT:   %impl.elem0.loc134_43.2 => constants.%impl.elem0.cb3
// CHECK:STDOUT:   %facet_type.loc134_37 => constants.%facet_type.bfb
// CHECK:STDOUT:   %pattern_type.loc134 => constants.%pattern_type.edc
// CHECK:STDOUT:   %T.patt.loc134_4.2 => constants.%T.patt.07b
// CHECK:STDOUT:   %T.loc134_4.1 => constants.%facet_value.645
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
// CHECK:STDOUT:   %require_complete.loc137 => constants.%complete_type.8a2
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.ada
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.loc137 => constants.%custom_witness.576
// CHECK:STDOUT:   %SubWith.facet => constants.%SubWith.facet.d30
// CHECK:STDOUT:   %SubWith.WithSelf.Op.type => constants.%SubWith.WithSelf.Op.type.6e5
// CHECK:STDOUT:   %.loc137_5.6 => constants.%.d1a
// CHECK:STDOUT:   %impl.elem1.loc137_5.2 => constants.%Int32.Op.4dca31.2
// CHECK:STDOUT:   %specific_impl_fn.loc137_5.2 => constants.%Int32.Op.4dca31.2
// CHECK:STDOUT:   %SelfDestruct.specific_fn.2 => constants.%SelfDestruct.specific_fn.861
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @MulWith.loc146(constants.%facet_value.cb7, constants.%facet_value.a87, constants.%facet_value.a15) {
// CHECK:STDOUT:   %Result.patt.loc143_9.2 => constants.%Result.patt
// CHECK:STDOUT:   %Result.loc143_9.1 => constants.%facet_value.cb7
// CHECK:STDOUT:   %U.patt.loc144_4.2 => constants.%U.patt.4c9
// CHECK:STDOUT:   %U.loc144_4.1 => constants.%facet_value.a87
// CHECK:STDOUT:   %U.as_type.loc145_35.1 => constants.%Int16
// CHECK:STDOUT:   %MulWith.type.loc145_35.1 => constants.%MulWith.type.f65
// CHECK:STDOUT:   %facet_type.loc145_19 => constants.%facet_type.eda
// CHECK:STDOUT:   %.Self.frozen.loc145_37.1 => constants.%.Self.frozen.8ef
// CHECK:STDOUT:   %require_complete.loc145 => constants.%complete_type.57a
// CHECK:STDOUT:   %.Self.frozen.as_type => constants.%.Self.frozen.as_type.29d
// CHECK:STDOUT:   %MulWith.assoc_type => constants.%MulWith.assoc_type.4c3
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.6ec
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.loc145_43.1 => constants.%MulWith.lookup_impl_witness.8f6
// CHECK:STDOUT:   %impl.elem0.loc145_43.1 => constants.%impl.elem0.0af
// CHECK:STDOUT:   %Result.as_type.loc145_53.1 => constants.%Int32
// CHECK:STDOUT:   %.Self => constants.%.Self.a47
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.loc145_43.2 => constants.%MulWith.lookup_impl_witness.923
// CHECK:STDOUT:   %impl.elem0.loc145_43.2 => constants.%impl.elem0.567
// CHECK:STDOUT:   %facet_type.loc145_37 => constants.%facet_type.d1d
// CHECK:STDOUT:   %pattern_type.loc145 => constants.%pattern_type.f82
// CHECK:STDOUT:   %T.patt.loc145_4.2 => constants.%T.patt.983
// CHECK:STDOUT:   %T.loc145_4.1 => constants.%facet_value.a15
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
// CHECK:STDOUT:   %require_complete.loc148 => constants.%complete_type.8c6
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.b14
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.loc148 => constants.%custom_witness.751
// CHECK:STDOUT:   %MulWith.facet => constants.%MulWith.facet.cd8
// CHECK:STDOUT:   %MulWith.WithSelf.Op.type => constants.%MulWith.WithSelf.Op.type.31d
// CHECK:STDOUT:   %.loc148_5.6 => constants.%.ba7
// CHECK:STDOUT:   %impl.elem1.loc148_5.2 => constants.%Int32.Op.4dca31.3
// CHECK:STDOUT:   %specific_impl_fn.loc148_5.2 => constants.%Int32.Op.4dca31.3
// CHECK:STDOUT:   %SelfDestruct.specific_fn.2 => constants.%SelfDestruct.specific_fn.861
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @DivWith.loc157(constants.%facet_value.cb7, constants.%facet_value.a87, constants.%facet_value.d17) {
// CHECK:STDOUT:   %Result.patt.loc154_9.2 => constants.%Result.patt
// CHECK:STDOUT:   %Result.loc154_9.1 => constants.%facet_value.cb7
// CHECK:STDOUT:   %U.patt.loc155_4.2 => constants.%U.patt.4c9
// CHECK:STDOUT:   %U.loc155_4.1 => constants.%facet_value.a87
// CHECK:STDOUT:   %U.as_type.loc156_35.1 => constants.%Int16
// CHECK:STDOUT:   %DivWith.type.loc156_35.1 => constants.%DivWith.type.b8a
// CHECK:STDOUT:   %facet_type.loc156_19 => constants.%facet_type.d6d
// CHECK:STDOUT:   %.Self.frozen.loc156_37.1 => constants.%.Self.frozen.df7
// CHECK:STDOUT:   %require_complete.loc156 => constants.%complete_type.e4f
// CHECK:STDOUT:   %.Self.frozen.as_type => constants.%.Self.frozen.as_type.cfb
// CHECK:STDOUT:   %DivWith.assoc_type => constants.%DivWith.assoc_type.712
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.99c
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.loc156_43.1 => constants.%DivWith.lookup_impl_witness.1bb
// CHECK:STDOUT:   %impl.elem0.loc156_43.1 => constants.%impl.elem0.729
// CHECK:STDOUT:   %Result.as_type.loc156_53.1 => constants.%Int32
// CHECK:STDOUT:   %.Self => constants.%.Self.1ea
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.loc156_43.2 => constants.%DivWith.lookup_impl_witness.5ff
// CHECK:STDOUT:   %impl.elem0.loc156_43.2 => constants.%impl.elem0.416
// CHECK:STDOUT:   %facet_type.loc156_37 => constants.%facet_type.835
// CHECK:STDOUT:   %pattern_type.loc156 => constants.%pattern_type.5a9
// CHECK:STDOUT:   %T.patt.loc156_4.2 => constants.%T.patt.7bf
// CHECK:STDOUT:   %T.loc156_4.1 => constants.%facet_value.d17
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
// CHECK:STDOUT:   %require_complete.loc159 => constants.%complete_type.d99
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.fb6
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.loc159 => constants.%custom_witness.9ad
// CHECK:STDOUT:   %DivWith.facet => constants.%DivWith.facet.d6b
// CHECK:STDOUT:   %DivWith.WithSelf.Op.type => constants.%DivWith.WithSelf.Op.type.b35
// CHECK:STDOUT:   %.loc159_5.6 => constants.%.636
// CHECK:STDOUT:   %impl.elem1.loc159_5.2 => constants.%Int32.Op.4dca31.4
// CHECK:STDOUT:   %specific_impl_fn.loc159_5.2 => constants.%Int32.Op.4dca31.4
// CHECK:STDOUT:   %SelfDestruct.specific_fn.2 => constants.%SelfDestruct.specific_fn.861
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ModWith.loc168(constants.%facet_value.cb7, constants.%facet_value.a87, constants.%facet_value.a16) {
// CHECK:STDOUT:   %Result.patt.loc165_9.2 => constants.%Result.patt
// CHECK:STDOUT:   %Result.loc165_9.1 => constants.%facet_value.cb7
// CHECK:STDOUT:   %U.patt.loc166_4.2 => constants.%U.patt.4c9
// CHECK:STDOUT:   %U.loc166_4.1 => constants.%facet_value.a87
// CHECK:STDOUT:   %U.as_type.loc167_35.1 => constants.%Int16
// CHECK:STDOUT:   %ModWith.type.loc167_35.1 => constants.%ModWith.type.c67
// CHECK:STDOUT:   %facet_type.loc167_19 => constants.%facet_type.8eb
// CHECK:STDOUT:   %.Self.frozen.loc167_37.1 => constants.%.Self.frozen.fb4
// CHECK:STDOUT:   %require_complete.loc167 => constants.%complete_type.94f
// CHECK:STDOUT:   %.Self.frozen.as_type => constants.%.Self.frozen.as_type.260
// CHECK:STDOUT:   %ModWith.assoc_type => constants.%ModWith.assoc_type.5f6
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.f04
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.loc167_43.1 => constants.%ModWith.lookup_impl_witness.679
// CHECK:STDOUT:   %impl.elem0.loc167_43.1 => constants.%impl.elem0.64e
// CHECK:STDOUT:   %Result.as_type.loc167_53.1 => constants.%Int32
// CHECK:STDOUT:   %.Self => constants.%.Self.806
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.loc167_43.2 => constants.%ModWith.lookup_impl_witness.f9a
// CHECK:STDOUT:   %impl.elem0.loc167_43.2 => constants.%impl.elem0.507
// CHECK:STDOUT:   %facet_type.loc167_37 => constants.%facet_type.9e1
// CHECK:STDOUT:   %pattern_type.loc167 => constants.%pattern_type.f91
// CHECK:STDOUT:   %T.patt.loc167_4.2 => constants.%T.patt.e03
// CHECK:STDOUT:   %T.loc167_4.1 => constants.%facet_value.a16
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
// CHECK:STDOUT:   %require_complete.loc170 => constants.%complete_type.7d8
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.d19
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.loc170 => constants.%custom_witness.8a2
// CHECK:STDOUT:   %ModWith.facet => constants.%ModWith.facet.9d1
// CHECK:STDOUT:   %ModWith.WithSelf.Op.type => constants.%ModWith.WithSelf.Op.type.e61
// CHECK:STDOUT:   %.loc170_5.6 => constants.%.25b
// CHECK:STDOUT:   %impl.elem1.loc170_5.2 => constants.%Int32.Op.4dca31.5
// CHECK:STDOUT:   %specific_impl_fn.loc170_5.2 => constants.%Int32.Op.4dca31.5
// CHECK:STDOUT:   %SelfDestruct.specific_fn.2 => constants.%SelfDestruct.specific_fn.861
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AddWith.loc124(constants.%facet_value.cb7, constants.%facet_value.cb7, constants.%facet_value.c03) {
// CHECK:STDOUT:   %Result.patt.loc121_9.2 => constants.%Result.patt
// CHECK:STDOUT:   %Result.loc121_9.1 => constants.%facet_value.cb7
// CHECK:STDOUT:   %U.patt.loc122_4.2 => constants.%U.patt.4c9
// CHECK:STDOUT:   %U.loc122_4.1 => constants.%facet_value.cb7
// CHECK:STDOUT:   %U.as_type.loc123_35.1 => constants.%Int32
// CHECK:STDOUT:   %AddWith.type.loc123_35.1 => constants.%AddWith.type.db7
// CHECK:STDOUT:   %facet_type.loc123_19 => constants.%facet_type.f00
// CHECK:STDOUT:   %.Self.frozen.loc123_37.1 => constants.%.Self.frozen.5ba
// CHECK:STDOUT:   %require_complete.loc123 => constants.%complete_type.3c7
// CHECK:STDOUT:   %.Self.frozen.as_type => constants.%.Self.frozen.as_type.cb7
// CHECK:STDOUT:   %AddWith.assoc_type => constants.%AddWith.assoc_type.be8
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.31c
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.loc123_43.1 => constants.%AddWith.lookup_impl_witness.337
// CHECK:STDOUT:   %impl.elem0.loc123_43.1 => constants.%impl.elem0.14f
// CHECK:STDOUT:   %Result.as_type.loc123_53.1 => constants.%Int32
// CHECK:STDOUT:   %.Self => constants.%.Self.29d
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.loc123_43.2 => constants.%AddWith.lookup_impl_witness.c08
// CHECK:STDOUT:   %impl.elem0.loc123_43.2 => constants.%impl.elem0.49b
// CHECK:STDOUT:   %facet_type.loc123_37 => constants.%facet_type.8ac
// CHECK:STDOUT:   %pattern_type.loc123 => constants.%pattern_type.ff4
// CHECK:STDOUT:   %T.patt.loc123_4.2 => constants.%T.patt.fe3
// CHECK:STDOUT:   %T.loc123_4.1 => constants.%facet_value.c03
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
// CHECK:STDOUT:   %require_complete.loc126 => constants.%complete_type.758
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.86c
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.loc126 => constants.%custom_witness.513476.2
// CHECK:STDOUT:   %AddWith.facet => constants.%AddWith.facet.684
// CHECK:STDOUT:   %AddWith.WithSelf.Op.type => constants.%AddWith.WithSelf.Op.type.0af
// CHECK:STDOUT:   %.loc126_5.6 => constants.%.e92
// CHECK:STDOUT:   %impl.elem1.loc126_5.2 => constants.%Int32.Op.4dca31.6
// CHECK:STDOUT:   %specific_impl_fn.loc126_5.2 => constants.%Int32.Op.4dca31.6
// CHECK:STDOUT:   %SelfDestruct.specific_fn.2 => constants.%SelfDestruct.specific_fn.861
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @SubWith.loc135(constants.%facet_value.cb7, constants.%facet_value.cb7, constants.%facet_value.789) {
// CHECK:STDOUT:   %Result.patt.loc132_9.2 => constants.%Result.patt
// CHECK:STDOUT:   %Result.loc132_9.1 => constants.%facet_value.cb7
// CHECK:STDOUT:   %U.patt.loc133_4.2 => constants.%U.patt.4c9
// CHECK:STDOUT:   %U.loc133_4.1 => constants.%facet_value.cb7
// CHECK:STDOUT:   %U.as_type.loc134_35.1 => constants.%Int32
// CHECK:STDOUT:   %SubWith.type.loc134_35.1 => constants.%SubWith.type.120
// CHECK:STDOUT:   %facet_type.loc134_19 => constants.%facet_type.4e1
// CHECK:STDOUT:   %.Self.frozen.loc134_37.1 => constants.%.Self.frozen.0f1
// CHECK:STDOUT:   %require_complete.loc134 => constants.%complete_type.ce7
// CHECK:STDOUT:   %.Self.frozen.as_type => constants.%.Self.frozen.as_type.815
// CHECK:STDOUT:   %SubWith.assoc_type => constants.%SubWith.assoc_type.dae
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.fac
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.loc134_43.1 => constants.%SubWith.lookup_impl_witness.5d6
// CHECK:STDOUT:   %impl.elem0.loc134_43.1 => constants.%impl.elem0.1ff
// CHECK:STDOUT:   %Result.as_type.loc134_53.1 => constants.%Int32
// CHECK:STDOUT:   %.Self => constants.%.Self.763
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.loc134_43.2 => constants.%SubWith.lookup_impl_witness.08b
// CHECK:STDOUT:   %impl.elem0.loc134_43.2 => constants.%impl.elem0.496
// CHECK:STDOUT:   %facet_type.loc134_37 => constants.%facet_type.541
// CHECK:STDOUT:   %pattern_type.loc134 => constants.%pattern_type.183
// CHECK:STDOUT:   %T.patt.loc134_4.2 => constants.%T.patt.a86
// CHECK:STDOUT:   %T.loc134_4.1 => constants.%facet_value.789
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
// CHECK:STDOUT:   %require_complete.loc137 => constants.%complete_type.46e
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.147
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.loc137 => constants.%custom_witness.32cef1.2
// CHECK:STDOUT:   %SubWith.facet => constants.%SubWith.facet.a46
// CHECK:STDOUT:   %SubWith.WithSelf.Op.type => constants.%SubWith.WithSelf.Op.type.1a5
// CHECK:STDOUT:   %.loc137_5.6 => constants.%.457
// CHECK:STDOUT:   %impl.elem1.loc137_5.2 => constants.%Int32.Op.4dca31.7
// CHECK:STDOUT:   %specific_impl_fn.loc137_5.2 => constants.%Int32.Op.4dca31.7
// CHECK:STDOUT:   %SelfDestruct.specific_fn.2 => constants.%SelfDestruct.specific_fn.861
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @MulWith.loc146(constants.%facet_value.cb7, constants.%facet_value.cb7, constants.%facet_value.9fe) {
// CHECK:STDOUT:   %Result.patt.loc143_9.2 => constants.%Result.patt
// CHECK:STDOUT:   %Result.loc143_9.1 => constants.%facet_value.cb7
// CHECK:STDOUT:   %U.patt.loc144_4.2 => constants.%U.patt.4c9
// CHECK:STDOUT:   %U.loc144_4.1 => constants.%facet_value.cb7
// CHECK:STDOUT:   %U.as_type.loc145_35.1 => constants.%Int32
// CHECK:STDOUT:   %MulWith.type.loc145_35.1 => constants.%MulWith.type.d67
// CHECK:STDOUT:   %facet_type.loc145_19 => constants.%facet_type.89e
// CHECK:STDOUT:   %.Self.frozen.loc145_37.1 => constants.%.Self.frozen.709
// CHECK:STDOUT:   %require_complete.loc145 => constants.%complete_type.a47
// CHECK:STDOUT:   %.Self.frozen.as_type => constants.%.Self.frozen.as_type.93b
// CHECK:STDOUT:   %MulWith.assoc_type => constants.%MulWith.assoc_type.ab1
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.d0a
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.loc145_43.1 => constants.%MulWith.lookup_impl_witness.5dc
// CHECK:STDOUT:   %impl.elem0.loc145_43.1 => constants.%impl.elem0.a0b
// CHECK:STDOUT:   %Result.as_type.loc145_53.1 => constants.%Int32
// CHECK:STDOUT:   %.Self => constants.%.Self.4cb
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.loc145_43.2 => constants.%MulWith.lookup_impl_witness.1b8
// CHECK:STDOUT:   %impl.elem0.loc145_43.2 => constants.%impl.elem0.fa7
// CHECK:STDOUT:   %facet_type.loc145_37 => constants.%facet_type.3b1
// CHECK:STDOUT:   %pattern_type.loc145 => constants.%pattern_type.7e1
// CHECK:STDOUT:   %T.patt.loc145_4.2 => constants.%T.patt.de6
// CHECK:STDOUT:   %T.loc145_4.1 => constants.%facet_value.9fe
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
// CHECK:STDOUT:   %require_complete.loc148 => constants.%complete_type.920
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.349
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.loc148 => constants.%custom_witness.3a0675.2
// CHECK:STDOUT:   %MulWith.facet => constants.%MulWith.facet.3ab
// CHECK:STDOUT:   %MulWith.WithSelf.Op.type => constants.%MulWith.WithSelf.Op.type.2ed
// CHECK:STDOUT:   %.loc148_5.6 => constants.%.9d6
// CHECK:STDOUT:   %impl.elem1.loc148_5.2 => constants.%Int32.Op.4dca31.8
// CHECK:STDOUT:   %specific_impl_fn.loc148_5.2 => constants.%Int32.Op.4dca31.8
// CHECK:STDOUT:   %SelfDestruct.specific_fn.2 => constants.%SelfDestruct.specific_fn.861
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @DivWith.loc157(constants.%facet_value.cb7, constants.%facet_value.cb7, constants.%facet_value.079) {
// CHECK:STDOUT:   %Result.patt.loc154_9.2 => constants.%Result.patt
// CHECK:STDOUT:   %Result.loc154_9.1 => constants.%facet_value.cb7
// CHECK:STDOUT:   %U.patt.loc155_4.2 => constants.%U.patt.4c9
// CHECK:STDOUT:   %U.loc155_4.1 => constants.%facet_value.cb7
// CHECK:STDOUT:   %U.as_type.loc156_35.1 => constants.%Int32
// CHECK:STDOUT:   %DivWith.type.loc156_35.1 => constants.%DivWith.type.5a1
// CHECK:STDOUT:   %facet_type.loc156_19 => constants.%facet_type.2ab
// CHECK:STDOUT:   %.Self.frozen.loc156_37.1 => constants.%.Self.frozen.ee7
// CHECK:STDOUT:   %require_complete.loc156 => constants.%complete_type.f97
// CHECK:STDOUT:   %.Self.frozen.as_type => constants.%.Self.frozen.as_type.7f5
// CHECK:STDOUT:   %DivWith.assoc_type => constants.%DivWith.assoc_type.22f
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.a9d
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.loc156_43.1 => constants.%DivWith.lookup_impl_witness.b80
// CHECK:STDOUT:   %impl.elem0.loc156_43.1 => constants.%impl.elem0.16df
// CHECK:STDOUT:   %Result.as_type.loc156_53.1 => constants.%Int32
// CHECK:STDOUT:   %.Self => constants.%.Self.b0f
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.loc156_43.2 => constants.%DivWith.lookup_impl_witness.38e
// CHECK:STDOUT:   %impl.elem0.loc156_43.2 => constants.%impl.elem0.183
// CHECK:STDOUT:   %facet_type.loc156_37 => constants.%facet_type.50c
// CHECK:STDOUT:   %pattern_type.loc156 => constants.%pattern_type.df6
// CHECK:STDOUT:   %T.patt.loc156_4.2 => constants.%T.patt.c47
// CHECK:STDOUT:   %T.loc156_4.1 => constants.%facet_value.079
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
// CHECK:STDOUT:   %require_complete.loc159 => constants.%complete_type.348
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.004
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.loc159 => constants.%custom_witness.c32685.2
// CHECK:STDOUT:   %DivWith.facet => constants.%DivWith.facet.b4a
// CHECK:STDOUT:   %DivWith.WithSelf.Op.type => constants.%DivWith.WithSelf.Op.type.085
// CHECK:STDOUT:   %.loc159_5.6 => constants.%.10c
// CHECK:STDOUT:   %impl.elem1.loc159_5.2 => constants.%Int32.Op.4dca31.9
// CHECK:STDOUT:   %specific_impl_fn.loc159_5.2 => constants.%Int32.Op.4dca31.9
// CHECK:STDOUT:   %SelfDestruct.specific_fn.2 => constants.%SelfDestruct.specific_fn.861
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ModWith.loc168(constants.%facet_value.cb7, constants.%facet_value.cb7, constants.%facet_value.72c) {
// CHECK:STDOUT:   %Result.patt.loc165_9.2 => constants.%Result.patt
// CHECK:STDOUT:   %Result.loc165_9.1 => constants.%facet_value.cb7
// CHECK:STDOUT:   %U.patt.loc166_4.2 => constants.%U.patt.4c9
// CHECK:STDOUT:   %U.loc166_4.1 => constants.%facet_value.cb7
// CHECK:STDOUT:   %U.as_type.loc167_35.1 => constants.%Int32
// CHECK:STDOUT:   %ModWith.type.loc167_35.1 => constants.%ModWith.type.794
// CHECK:STDOUT:   %facet_type.loc167_19 => constants.%facet_type.fc7
// CHECK:STDOUT:   %.Self.frozen.loc167_37.1 => constants.%.Self.frozen.f41
// CHECK:STDOUT:   %require_complete.loc167 => constants.%complete_type.0f1
// CHECK:STDOUT:   %.Self.frozen.as_type => constants.%.Self.frozen.as_type.7dc
// CHECK:STDOUT:   %ModWith.assoc_type => constants.%ModWith.assoc_type.021
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.310
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.loc167_43.1 => constants.%ModWith.lookup_impl_witness.f4e
// CHECK:STDOUT:   %impl.elem0.loc167_43.1 => constants.%impl.elem0.3c4
// CHECK:STDOUT:   %Result.as_type.loc167_53.1 => constants.%Int32
// CHECK:STDOUT:   %.Self => constants.%.Self.797
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.loc167_43.2 => constants.%ModWith.lookup_impl_witness.126
// CHECK:STDOUT:   %impl.elem0.loc167_43.2 => constants.%impl.elem0.6bf
// CHECK:STDOUT:   %facet_type.loc167_37 => constants.%facet_type.54d
// CHECK:STDOUT:   %pattern_type.loc167 => constants.%pattern_type.dae
// CHECK:STDOUT:   %T.patt.loc167_4.2 => constants.%T.patt.4d5
// CHECK:STDOUT:   %T.loc167_4.1 => constants.%facet_value.72c
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
// CHECK:STDOUT:   %require_complete.loc170 => constants.%complete_type.cc9
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.2eb
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.loc170 => constants.%custom_witness.994c75.2
// CHECK:STDOUT:   %ModWith.facet => constants.%ModWith.facet.3dc
// CHECK:STDOUT:   %ModWith.WithSelf.Op.type => constants.%ModWith.WithSelf.Op.type.a2a
// CHECK:STDOUT:   %.loc170_5.6 => constants.%.b6d
// CHECK:STDOUT:   %impl.elem1.loc170_5.2 => constants.%Int32.Op.4dca31.10
// CHECK:STDOUT:   %specific_impl_fn.loc170_5.2 => constants.%Int32.Op.4dca31.10
// CHECK:STDOUT:   %SelfDestruct.specific_fn.2 => constants.%SelfDestruct.specific_fn.861
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AddWith.loc124(constants.%facet_value.ca8, constants.%facet_value.ca8, constants.%facet_value.797) {
// CHECK:STDOUT:   %Result.patt.loc121_9.2 => constants.%Result.patt
// CHECK:STDOUT:   %Result.loc121_9.1 => constants.%facet_value.ca8
// CHECK:STDOUT:   %U.patt.loc122_4.2 => constants.%U.patt.4c9
// CHECK:STDOUT:   %U.loc122_4.1 => constants.%facet_value.ca8
// CHECK:STDOUT:   %U.as_type.loc123_35.1 => constants.%Int64
// CHECK:STDOUT:   %AddWith.type.loc123_35.1 => constants.%AddWith.type.1ff
// CHECK:STDOUT:   %facet_type.loc123_19 => constants.%facet_type.884
// CHECK:STDOUT:   %.Self.frozen.loc123_37.1 => constants.%.Self.frozen.266
// CHECK:STDOUT:   %require_complete.loc123 => constants.%complete_type.2dc
// CHECK:STDOUT:   %.Self.frozen.as_type => constants.%.Self.frozen.as_type.021
// CHECK:STDOUT:   %AddWith.assoc_type => constants.%AddWith.assoc_type.97f
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.f02
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.loc123_43.1 => constants.%AddWith.lookup_impl_witness.a15
// CHECK:STDOUT:   %impl.elem0.loc123_43.1 => constants.%impl.elem0.f79
// CHECK:STDOUT:   %Result.as_type.loc123_53.1 => constants.%Int64
// CHECK:STDOUT:   %.Self => constants.%.Self.f99
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.loc123_43.2 => constants.%AddWith.lookup_impl_witness.6d7
// CHECK:STDOUT:   %impl.elem0.loc123_43.2 => constants.%impl.elem0.e6e
// CHECK:STDOUT:   %facet_type.loc123_37 => constants.%facet_type.226
// CHECK:STDOUT:   %pattern_type.loc123 => constants.%pattern_type.a52
// CHECK:STDOUT:   %T.patt.loc123_4.2 => constants.%T.patt.e68
// CHECK:STDOUT:   %T.loc123_4.1 => constants.%facet_value.797
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
// CHECK:STDOUT:   %require_complete.loc126 => constants.%complete_type.57b
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.a75
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.loc126 => constants.%custom_witness.f2f
// CHECK:STDOUT:   %AddWith.facet => constants.%AddWith.facet.0d8
// CHECK:STDOUT:   %AddWith.WithSelf.Op.type => constants.%AddWith.WithSelf.Op.type.a40
// CHECK:STDOUT:   %.loc126_5.6 => constants.%.51d
// CHECK:STDOUT:   %impl.elem1.loc126_5.2 => constants.%Op.72717c.6
// CHECK:STDOUT:   %specific_impl_fn.loc126_5.2 => constants.%Op.72717c.6
// CHECK:STDOUT:   %SelfDestruct.specific_fn.2 => constants.%SelfDestruct.specific_fn.d50
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @SubWith.loc135(constants.%facet_value.ca8, constants.%facet_value.ca8, constants.%facet_value.e83) {
// CHECK:STDOUT:   %Result.patt.loc132_9.2 => constants.%Result.patt
// CHECK:STDOUT:   %Result.loc132_9.1 => constants.%facet_value.ca8
// CHECK:STDOUT:   %U.patt.loc133_4.2 => constants.%U.patt.4c9
// CHECK:STDOUT:   %U.loc133_4.1 => constants.%facet_value.ca8
// CHECK:STDOUT:   %U.as_type.loc134_35.1 => constants.%Int64
// CHECK:STDOUT:   %SubWith.type.loc134_35.1 => constants.%SubWith.type.31c
// CHECK:STDOUT:   %facet_type.loc134_19 => constants.%facet_type.511
// CHECK:STDOUT:   %.Self.frozen.loc134_37.1 => constants.%.Self.frozen.0f4
// CHECK:STDOUT:   %require_complete.loc134 => constants.%complete_type.5d9
// CHECK:STDOUT:   %.Self.frozen.as_type => constants.%.Self.frozen.as_type.a25
// CHECK:STDOUT:   %SubWith.assoc_type => constants.%SubWith.assoc_type.4f8
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.13b
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.loc134_43.1 => constants.%SubWith.lookup_impl_witness.266
// CHECK:STDOUT:   %impl.elem0.loc134_43.1 => constants.%impl.elem0.097
// CHECK:STDOUT:   %Result.as_type.loc134_53.1 => constants.%Int64
// CHECK:STDOUT:   %.Self => constants.%.Self.f66
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.loc134_43.2 => constants.%SubWith.lookup_impl_witness.05e
// CHECK:STDOUT:   %impl.elem0.loc134_43.2 => constants.%impl.elem0.bc3
// CHECK:STDOUT:   %facet_type.loc134_37 => constants.%facet_type.ad6
// CHECK:STDOUT:   %pattern_type.loc134 => constants.%pattern_type.647
// CHECK:STDOUT:   %T.patt.loc134_4.2 => constants.%T.patt.b82
// CHECK:STDOUT:   %T.loc134_4.1 => constants.%facet_value.e83
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
// CHECK:STDOUT:   %require_complete.loc137 => constants.%complete_type.cb5
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.e70
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.loc137 => constants.%custom_witness.276
// CHECK:STDOUT:   %SubWith.facet => constants.%SubWith.facet.d62
// CHECK:STDOUT:   %SubWith.WithSelf.Op.type => constants.%SubWith.WithSelf.Op.type.d39
// CHECK:STDOUT:   %.loc137_5.6 => constants.%.e39
// CHECK:STDOUT:   %impl.elem1.loc137_5.2 => constants.%Op.72717c.7
// CHECK:STDOUT:   %specific_impl_fn.loc137_5.2 => constants.%Op.72717c.7
// CHECK:STDOUT:   %SelfDestruct.specific_fn.2 => constants.%SelfDestruct.specific_fn.d50
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @MulWith.loc146(constants.%facet_value.ca8, constants.%facet_value.ca8, constants.%facet_value.055) {
// CHECK:STDOUT:   %Result.patt.loc143_9.2 => constants.%Result.patt
// CHECK:STDOUT:   %Result.loc143_9.1 => constants.%facet_value.ca8
// CHECK:STDOUT:   %U.patt.loc144_4.2 => constants.%U.patt.4c9
// CHECK:STDOUT:   %U.loc144_4.1 => constants.%facet_value.ca8
// CHECK:STDOUT:   %U.as_type.loc145_35.1 => constants.%Int64
// CHECK:STDOUT:   %MulWith.type.loc145_35.1 => constants.%MulWith.type.661
// CHECK:STDOUT:   %facet_type.loc145_19 => constants.%facet_type.58d
// CHECK:STDOUT:   %.Self.frozen.loc145_37.1 => constants.%.Self.frozen.edb
// CHECK:STDOUT:   %require_complete.loc145 => constants.%complete_type.c4e
// CHECK:STDOUT:   %.Self.frozen.as_type => constants.%.Self.frozen.as_type.63f
// CHECK:STDOUT:   %MulWith.assoc_type => constants.%MulWith.assoc_type.940
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.9d9
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.loc145_43.1 => constants.%MulWith.lookup_impl_witness.ddf
// CHECK:STDOUT:   %impl.elem0.loc145_43.1 => constants.%impl.elem0.1e2
// CHECK:STDOUT:   %Result.as_type.loc145_53.1 => constants.%Int64
// CHECK:STDOUT:   %.Self => constants.%.Self.f7f
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.loc145_43.2 => constants.%MulWith.lookup_impl_witness.53a
// CHECK:STDOUT:   %impl.elem0.loc145_43.2 => constants.%impl.elem0.5d5
// CHECK:STDOUT:   %facet_type.loc145_37 => constants.%facet_type.f3f
// CHECK:STDOUT:   %pattern_type.loc145 => constants.%pattern_type.761
// CHECK:STDOUT:   %T.patt.loc145_4.2 => constants.%T.patt.460
// CHECK:STDOUT:   %T.loc145_4.1 => constants.%facet_value.055
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
// CHECK:STDOUT:   %require_complete.loc148 => constants.%complete_type.0a6
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.b22
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.loc148 => constants.%custom_witness.8e4
// CHECK:STDOUT:   %MulWith.facet => constants.%MulWith.facet.4a0
// CHECK:STDOUT:   %MulWith.WithSelf.Op.type => constants.%MulWith.WithSelf.Op.type.489
// CHECK:STDOUT:   %.loc148_5.6 => constants.%.2e6
// CHECK:STDOUT:   %impl.elem1.loc148_5.2 => constants.%Op.72717c.8
// CHECK:STDOUT:   %specific_impl_fn.loc148_5.2 => constants.%Op.72717c.8
// CHECK:STDOUT:   %SelfDestruct.specific_fn.2 => constants.%SelfDestruct.specific_fn.d50
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @DivWith.loc157(constants.%facet_value.ca8, constants.%facet_value.ca8, constants.%facet_value.260) {
// CHECK:STDOUT:   %Result.patt.loc154_9.2 => constants.%Result.patt
// CHECK:STDOUT:   %Result.loc154_9.1 => constants.%facet_value.ca8
// CHECK:STDOUT:   %U.patt.loc155_4.2 => constants.%U.patt.4c9
// CHECK:STDOUT:   %U.loc155_4.1 => constants.%facet_value.ca8
// CHECK:STDOUT:   %U.as_type.loc156_35.1 => constants.%Int64
// CHECK:STDOUT:   %DivWith.type.loc156_35.1 => constants.%DivWith.type.dd8
// CHECK:STDOUT:   %facet_type.loc156_19 => constants.%facet_type.6e2
// CHECK:STDOUT:   %.Self.frozen.loc156_37.1 => constants.%.Self.frozen.a2d
// CHECK:STDOUT:   %require_complete.loc156 => constants.%complete_type.f3e
// CHECK:STDOUT:   %.Self.frozen.as_type => constants.%.Self.frozen.as_type.342
// CHECK:STDOUT:   %DivWith.assoc_type => constants.%DivWith.assoc_type.2dc
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.e8b
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.loc156_43.1 => constants.%DivWith.lookup_impl_witness.b6a
// CHECK:STDOUT:   %impl.elem0.loc156_43.1 => constants.%impl.elem0.935
// CHECK:STDOUT:   %Result.as_type.loc156_53.1 => constants.%Int64
// CHECK:STDOUT:   %.Self => constants.%.Self.92e
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.loc156_43.2 => constants.%DivWith.lookup_impl_witness.1c8
// CHECK:STDOUT:   %impl.elem0.loc156_43.2 => constants.%impl.elem0.39a
// CHECK:STDOUT:   %facet_type.loc156_37 => constants.%facet_type.708
// CHECK:STDOUT:   %pattern_type.loc156 => constants.%pattern_type.f93
// CHECK:STDOUT:   %T.patt.loc156_4.2 => constants.%T.patt.0cc
// CHECK:STDOUT:   %T.loc156_4.1 => constants.%facet_value.260
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
// CHECK:STDOUT:   %require_complete.loc159 => constants.%complete_type.473
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.8b9
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.loc159 => constants.%custom_witness.099
// CHECK:STDOUT:   %DivWith.facet => constants.%DivWith.facet.b29
// CHECK:STDOUT:   %DivWith.WithSelf.Op.type => constants.%DivWith.WithSelf.Op.type.9fa
// CHECK:STDOUT:   %.loc159_5.6 => constants.%.bc2
// CHECK:STDOUT:   %impl.elem1.loc159_5.2 => constants.%Op.72717c.9
// CHECK:STDOUT:   %specific_impl_fn.loc159_5.2 => constants.%Op.72717c.9
// CHECK:STDOUT:   %SelfDestruct.specific_fn.2 => constants.%SelfDestruct.specific_fn.d50
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ModWith.loc168(constants.%facet_value.ca8, constants.%facet_value.ca8, constants.%facet_value.f07) {
// CHECK:STDOUT:   %Result.patt.loc165_9.2 => constants.%Result.patt
// CHECK:STDOUT:   %Result.loc165_9.1 => constants.%facet_value.ca8
// CHECK:STDOUT:   %U.patt.loc166_4.2 => constants.%U.patt.4c9
// CHECK:STDOUT:   %U.loc166_4.1 => constants.%facet_value.ca8
// CHECK:STDOUT:   %U.as_type.loc167_35.1 => constants.%Int64
// CHECK:STDOUT:   %ModWith.type.loc167_35.1 => constants.%ModWith.type.c72
// CHECK:STDOUT:   %facet_type.loc167_19 => constants.%facet_type.a05
// CHECK:STDOUT:   %.Self.frozen.loc167_37.1 => constants.%.Self.frozen.d87
// CHECK:STDOUT:   %require_complete.loc167 => constants.%complete_type.93c
// CHECK:STDOUT:   %.Self.frozen.as_type => constants.%.Self.frozen.as_type.c60
// CHECK:STDOUT:   %ModWith.assoc_type => constants.%ModWith.assoc_type.ad2
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.c8a
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.loc167_43.1 => constants.%ModWith.lookup_impl_witness.174
// CHECK:STDOUT:   %impl.elem0.loc167_43.1 => constants.%impl.elem0.020
// CHECK:STDOUT:   %Result.as_type.loc167_53.1 => constants.%Int64
// CHECK:STDOUT:   %.Self => constants.%.Self.445
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.loc167_43.2 => constants.%ModWith.lookup_impl_witness.ee1
// CHECK:STDOUT:   %impl.elem0.loc167_43.2 => constants.%impl.elem0.1ca
// CHECK:STDOUT:   %facet_type.loc167_37 => constants.%facet_type.e35
// CHECK:STDOUT:   %pattern_type.loc167 => constants.%pattern_type.1e1
// CHECK:STDOUT:   %T.patt.loc167_4.2 => constants.%T.patt.07e
// CHECK:STDOUT:   %T.loc167_4.1 => constants.%facet_value.f07
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
// CHECK:STDOUT:   %require_complete.loc170 => constants.%complete_type.4d1
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.eed
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.loc170 => constants.%custom_witness.129
// CHECK:STDOUT:   %ModWith.facet => constants.%ModWith.facet.b10
// CHECK:STDOUT:   %ModWith.WithSelf.Op.type => constants.%ModWith.WithSelf.Op.type.1ecc
// CHECK:STDOUT:   %.loc170_5.6 => constants.%.bf4
// CHECK:STDOUT:   %impl.elem1.loc170_5.2 => constants.%Op.72717c.10
// CHECK:STDOUT:   %specific_impl_fn.loc170_5.2 => constants.%Op.72717c.10
// CHECK:STDOUT:   %SelfDestruct.specific_fn.2 => constants.%SelfDestruct.specific_fn.d50
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AddWith.loc124(constants.%facet_value.ca8, constants.%facet_value.cb7, constants.%facet_value.252) {
// CHECK:STDOUT:   %Result.patt.loc121_9.2 => constants.%Result.patt
// CHECK:STDOUT:   %Result.loc121_9.1 => constants.%facet_value.ca8
// CHECK:STDOUT:   %U.patt.loc122_4.2 => constants.%U.patt.4c9
// CHECK:STDOUT:   %U.loc122_4.1 => constants.%facet_value.cb7
// CHECK:STDOUT:   %U.as_type.loc123_35.1 => constants.%Int32
// CHECK:STDOUT:   %AddWith.type.loc123_35.1 => constants.%AddWith.type.db7
// CHECK:STDOUT:   %facet_type.loc123_19 => constants.%facet_type.f00
// CHECK:STDOUT:   %.Self.frozen.loc123_37.1 => constants.%.Self.frozen.5ba
// CHECK:STDOUT:   %require_complete.loc123 => constants.%complete_type.3c7
// CHECK:STDOUT:   %.Self.frozen.as_type => constants.%.Self.frozen.as_type.cb7
// CHECK:STDOUT:   %AddWith.assoc_type => constants.%AddWith.assoc_type.be8
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.31c
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.loc123_43.1 => constants.%AddWith.lookup_impl_witness.337
// CHECK:STDOUT:   %impl.elem0.loc123_43.1 => constants.%impl.elem0.14f
// CHECK:STDOUT:   %Result.as_type.loc123_53.1 => constants.%Int64
// CHECK:STDOUT:   %.Self => constants.%.Self.29d
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.loc123_43.2 => constants.%AddWith.lookup_impl_witness.c08
// CHECK:STDOUT:   %impl.elem0.loc123_43.2 => constants.%impl.elem0.49b
// CHECK:STDOUT:   %facet_type.loc123_37 => constants.%facet_type.ec8
// CHECK:STDOUT:   %pattern_type.loc123 => constants.%pattern_type.d1c
// CHECK:STDOUT:   %T.patt.loc123_4.2 => constants.%T.patt.e18
// CHECK:STDOUT:   %T.loc123_4.1 => constants.%facet_value.252
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
// CHECK:STDOUT:   %require_complete.loc126 => constants.%complete_type.758
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.86c
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.loc126 => constants.%custom_witness.16b
// CHECK:STDOUT:   %AddWith.facet => constants.%AddWith.facet.f65
// CHECK:STDOUT:   %AddWith.WithSelf.Op.type => constants.%AddWith.WithSelf.Op.type.8b6
// CHECK:STDOUT:   %.loc126_5.6 => constants.%.cf7
// CHECK:STDOUT:   %impl.elem1.loc126_5.2 => constants.%Op.72717c.11
// CHECK:STDOUT:   %specific_impl_fn.loc126_5.2 => constants.%Op.72717c.11
// CHECK:STDOUT:   %SelfDestruct.specific_fn.2 => constants.%SelfDestruct.specific_fn.d50
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @SubWith.loc135(constants.%facet_value.ca8, constants.%facet_value.cb7, constants.%facet_value.cfe) {
// CHECK:STDOUT:   %Result.patt.loc132_9.2 => constants.%Result.patt
// CHECK:STDOUT:   %Result.loc132_9.1 => constants.%facet_value.ca8
// CHECK:STDOUT:   %U.patt.loc133_4.2 => constants.%U.patt.4c9
// CHECK:STDOUT:   %U.loc133_4.1 => constants.%facet_value.cb7
// CHECK:STDOUT:   %U.as_type.loc134_35.1 => constants.%Int32
// CHECK:STDOUT:   %SubWith.type.loc134_35.1 => constants.%SubWith.type.120
// CHECK:STDOUT:   %facet_type.loc134_19 => constants.%facet_type.4e1
// CHECK:STDOUT:   %.Self.frozen.loc134_37.1 => constants.%.Self.frozen.0f1
// CHECK:STDOUT:   %require_complete.loc134 => constants.%complete_type.ce7
// CHECK:STDOUT:   %.Self.frozen.as_type => constants.%.Self.frozen.as_type.815
// CHECK:STDOUT:   %SubWith.assoc_type => constants.%SubWith.assoc_type.dae
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.fac
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.loc134_43.1 => constants.%SubWith.lookup_impl_witness.5d6
// CHECK:STDOUT:   %impl.elem0.loc134_43.1 => constants.%impl.elem0.1ff
// CHECK:STDOUT:   %Result.as_type.loc134_53.1 => constants.%Int64
// CHECK:STDOUT:   %.Self => constants.%.Self.763
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.loc134_43.2 => constants.%SubWith.lookup_impl_witness.08b
// CHECK:STDOUT:   %impl.elem0.loc134_43.2 => constants.%impl.elem0.496
// CHECK:STDOUT:   %facet_type.loc134_37 => constants.%facet_type.1f3
// CHECK:STDOUT:   %pattern_type.loc134 => constants.%pattern_type.e00
// CHECK:STDOUT:   %T.patt.loc134_4.2 => constants.%T.patt.0e9
// CHECK:STDOUT:   %T.loc134_4.1 => constants.%facet_value.cfe
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
// CHECK:STDOUT:   %require_complete.loc137 => constants.%complete_type.46e
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.147
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.loc137 => constants.%custom_witness.21a
// CHECK:STDOUT:   %SubWith.facet => constants.%SubWith.facet.245
// CHECK:STDOUT:   %SubWith.WithSelf.Op.type => constants.%SubWith.WithSelf.Op.type.2cf1
// CHECK:STDOUT:   %.loc137_5.6 => constants.%.4e8
// CHECK:STDOUT:   %impl.elem1.loc137_5.2 => constants.%Op.72717c.12
// CHECK:STDOUT:   %specific_impl_fn.loc137_5.2 => constants.%Op.72717c.12
// CHECK:STDOUT:   %SelfDestruct.specific_fn.2 => constants.%SelfDestruct.specific_fn.d50
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @MulWith.loc146(constants.%facet_value.ca8, constants.%facet_value.cb7, constants.%facet_value.f28) {
// CHECK:STDOUT:   %Result.patt.loc143_9.2 => constants.%Result.patt
// CHECK:STDOUT:   %Result.loc143_9.1 => constants.%facet_value.ca8
// CHECK:STDOUT:   %U.patt.loc144_4.2 => constants.%U.patt.4c9
// CHECK:STDOUT:   %U.loc144_4.1 => constants.%facet_value.cb7
// CHECK:STDOUT:   %U.as_type.loc145_35.1 => constants.%Int32
// CHECK:STDOUT:   %MulWith.type.loc145_35.1 => constants.%MulWith.type.d67
// CHECK:STDOUT:   %facet_type.loc145_19 => constants.%facet_type.89e
// CHECK:STDOUT:   %.Self.frozen.loc145_37.1 => constants.%.Self.frozen.709
// CHECK:STDOUT:   %require_complete.loc145 => constants.%complete_type.a47
// CHECK:STDOUT:   %.Self.frozen.as_type => constants.%.Self.frozen.as_type.93b
// CHECK:STDOUT:   %MulWith.assoc_type => constants.%MulWith.assoc_type.ab1
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.d0a
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.loc145_43.1 => constants.%MulWith.lookup_impl_witness.5dc
// CHECK:STDOUT:   %impl.elem0.loc145_43.1 => constants.%impl.elem0.a0b
// CHECK:STDOUT:   %Result.as_type.loc145_53.1 => constants.%Int64
// CHECK:STDOUT:   %.Self => constants.%.Self.4cb
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.loc145_43.2 => constants.%MulWith.lookup_impl_witness.1b8
// CHECK:STDOUT:   %impl.elem0.loc145_43.2 => constants.%impl.elem0.fa7
// CHECK:STDOUT:   %facet_type.loc145_37 => constants.%facet_type.9ad
// CHECK:STDOUT:   %pattern_type.loc145 => constants.%pattern_type.47d
// CHECK:STDOUT:   %T.patt.loc145_4.2 => constants.%T.patt.24f
// CHECK:STDOUT:   %T.loc145_4.1 => constants.%facet_value.f28
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
// CHECK:STDOUT:   %require_complete.loc148 => constants.%complete_type.920
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.349
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.loc148 => constants.%custom_witness.798
// CHECK:STDOUT:   %MulWith.facet => constants.%MulWith.facet.394
// CHECK:STDOUT:   %MulWith.WithSelf.Op.type => constants.%MulWith.WithSelf.Op.type.0ab
// CHECK:STDOUT:   %.loc148_5.6 => constants.%.22d
// CHECK:STDOUT:   %impl.elem1.loc148_5.2 => constants.%Op.72717c.13
// CHECK:STDOUT:   %specific_impl_fn.loc148_5.2 => constants.%Op.72717c.13
// CHECK:STDOUT:   %SelfDestruct.specific_fn.2 => constants.%SelfDestruct.specific_fn.d50
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @DivWith.loc157(constants.%facet_value.ca8, constants.%facet_value.cb7, constants.%facet_value.725) {
// CHECK:STDOUT:   %Result.patt.loc154_9.2 => constants.%Result.patt
// CHECK:STDOUT:   %Result.loc154_9.1 => constants.%facet_value.ca8
// CHECK:STDOUT:   %U.patt.loc155_4.2 => constants.%U.patt.4c9
// CHECK:STDOUT:   %U.loc155_4.1 => constants.%facet_value.cb7
// CHECK:STDOUT:   %U.as_type.loc156_35.1 => constants.%Int32
// CHECK:STDOUT:   %DivWith.type.loc156_35.1 => constants.%DivWith.type.5a1
// CHECK:STDOUT:   %facet_type.loc156_19 => constants.%facet_type.2ab
// CHECK:STDOUT:   %.Self.frozen.loc156_37.1 => constants.%.Self.frozen.ee7
// CHECK:STDOUT:   %require_complete.loc156 => constants.%complete_type.f97
// CHECK:STDOUT:   %.Self.frozen.as_type => constants.%.Self.frozen.as_type.7f5
// CHECK:STDOUT:   %DivWith.assoc_type => constants.%DivWith.assoc_type.22f
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.a9d
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.loc156_43.1 => constants.%DivWith.lookup_impl_witness.b80
// CHECK:STDOUT:   %impl.elem0.loc156_43.1 => constants.%impl.elem0.16df
// CHECK:STDOUT:   %Result.as_type.loc156_53.1 => constants.%Int64
// CHECK:STDOUT:   %.Self => constants.%.Self.b0f
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.loc156_43.2 => constants.%DivWith.lookup_impl_witness.38e
// CHECK:STDOUT:   %impl.elem0.loc156_43.2 => constants.%impl.elem0.183
// CHECK:STDOUT:   %facet_type.loc156_37 => constants.%facet_type.c2a
// CHECK:STDOUT:   %pattern_type.loc156 => constants.%pattern_type.8cc
// CHECK:STDOUT:   %T.patt.loc156_4.2 => constants.%T.patt.790
// CHECK:STDOUT:   %T.loc156_4.1 => constants.%facet_value.725
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
// CHECK:STDOUT:   %require_complete.loc159 => constants.%complete_type.348
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.004
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.loc159 => constants.%custom_witness.37f
// CHECK:STDOUT:   %DivWith.facet => constants.%DivWith.facet.85e
// CHECK:STDOUT:   %DivWith.WithSelf.Op.type => constants.%DivWith.WithSelf.Op.type.7b8
// CHECK:STDOUT:   %.loc159_5.6 => constants.%.cde
// CHECK:STDOUT:   %impl.elem1.loc159_5.2 => constants.%Op.72717c.14
// CHECK:STDOUT:   %specific_impl_fn.loc159_5.2 => constants.%Op.72717c.14
// CHECK:STDOUT:   %SelfDestruct.specific_fn.2 => constants.%SelfDestruct.specific_fn.d50
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ModWith.loc168(constants.%facet_value.ca8, constants.%facet_value.cb7, constants.%facet_value.f68) {
// CHECK:STDOUT:   %Result.patt.loc165_9.2 => constants.%Result.patt
// CHECK:STDOUT:   %Result.loc165_9.1 => constants.%facet_value.ca8
// CHECK:STDOUT:   %U.patt.loc166_4.2 => constants.%U.patt.4c9
// CHECK:STDOUT:   %U.loc166_4.1 => constants.%facet_value.cb7
// CHECK:STDOUT:   %U.as_type.loc167_35.1 => constants.%Int32
// CHECK:STDOUT:   %ModWith.type.loc167_35.1 => constants.%ModWith.type.794
// CHECK:STDOUT:   %facet_type.loc167_19 => constants.%facet_type.fc7
// CHECK:STDOUT:   %.Self.frozen.loc167_37.1 => constants.%.Self.frozen.f41
// CHECK:STDOUT:   %require_complete.loc167 => constants.%complete_type.0f1
// CHECK:STDOUT:   %.Self.frozen.as_type => constants.%.Self.frozen.as_type.7dc
// CHECK:STDOUT:   %ModWith.assoc_type => constants.%ModWith.assoc_type.021
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.310
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.loc167_43.1 => constants.%ModWith.lookup_impl_witness.f4e
// CHECK:STDOUT:   %impl.elem0.loc167_43.1 => constants.%impl.elem0.3c4
// CHECK:STDOUT:   %Result.as_type.loc167_53.1 => constants.%Int64
// CHECK:STDOUT:   %.Self => constants.%.Self.797
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.loc167_43.2 => constants.%ModWith.lookup_impl_witness.126
// CHECK:STDOUT:   %impl.elem0.loc167_43.2 => constants.%impl.elem0.6bf
// CHECK:STDOUT:   %facet_type.loc167_37 => constants.%facet_type.26a
// CHECK:STDOUT:   %pattern_type.loc167 => constants.%pattern_type.5b0
// CHECK:STDOUT:   %T.patt.loc167_4.2 => constants.%T.patt.730
// CHECK:STDOUT:   %T.loc167_4.1 => constants.%facet_value.f68
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
// CHECK:STDOUT:   %require_complete.loc170 => constants.%complete_type.cc9
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.2eb
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.loc170 => constants.%custom_witness.981
// CHECK:STDOUT:   %ModWith.facet => constants.%ModWith.facet.f5a2
// CHECK:STDOUT:   %ModWith.WithSelf.Op.type => constants.%ModWith.WithSelf.Op.type.d27
// CHECK:STDOUT:   %.loc170_5.6 => constants.%.3b0
// CHECK:STDOUT:   %impl.elem1.loc170_5.2 => constants.%Op.72717c.15
// CHECK:STDOUT:   %specific_impl_fn.loc170_5.2 => constants.%Op.72717c.15
// CHECK:STDOUT:   %SelfDestruct.specific_fn.2 => constants.%SelfDestruct.specific_fn.d50
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
// CHECK:STDOUT:   %.loc239_5.2 => constants.%.5363
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
// CHECK:STDOUT: specific @TestNegate(constants.%Result, constants.%T.810) {
// CHECK:STDOUT:   %Result.patt.loc292_9.2 => constants.%Result.patt
// CHECK:STDOUT:   %Result.loc292_9.1 => constants.%Result
// CHECK:STDOUT:   %Result.as_type.loc293_49.1 => constants.%Result.as_type
// CHECK:STDOUT:   %facet_type => constants.%facet_type.5ed
// CHECK:STDOUT:   %pattern_type.loc293 => constants.%pattern_type.6d4
// CHECK:STDOUT:   %T.patt.loc293_4.2 => constants.%T.patt.7ae
// CHECK:STDOUT:   %T.loc293_4.1 => constants.%T.810
// CHECK:STDOUT:   %pattern_type.loc294_11 => constants.%pattern_type.e4c132.1
// CHECK:STDOUT:   %y.param_patt.loc294_11.2 => constants.%y.param_patt.068
// CHECK:STDOUT:   %y.patt.loc294_11.2 => constants.%y.patt.e33
// CHECK:STDOUT:   %T.as_type.loc294_24.1 => constants.%T.as_type.a42
// CHECK:STDOUT:   %pattern_type.loc294_22 => constants.%pattern_type.e29
// CHECK:STDOUT:   %x.param_patt.loc294_22.2 => constants.%x.param_patt.7c9
// CHECK:STDOUT:   %x.patt.loc294_22.2 => constants.%x.patt.efa
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
// CHECK:STDOUT: specific @TestNegate(constants.%facet_value.ca8, constants.%facet_value.901) {
// CHECK:STDOUT:   %Result.patt.loc292_9.2 => constants.%Result.patt
// CHECK:STDOUT:   %Result.loc292_9.1 => constants.%facet_value.ca8
// CHECK:STDOUT:   %Result.as_type.loc293_49.1 => constants.%Int64
// CHECK:STDOUT:   %facet_type => constants.%facet_type.b1a
// CHECK:STDOUT:   %pattern_type.loc293 => constants.%pattern_type.06e
// CHECK:STDOUT:   %T.patt.loc293_4.2 => constants.%T.patt.315
// CHECK:STDOUT:   %T.loc293_4.1 => constants.%facet_value.901
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
// CHECK:STDOUT:   %SelfDestruct.specific_fn.2 => constants.%SelfDestruct.specific_fn.d50
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestNegate(constants.%facet_value.cb7, constants.%facet_value.1d9) {
// CHECK:STDOUT:   %Result.patt.loc292_9.2 => constants.%Result.patt
// CHECK:STDOUT:   %Result.loc292_9.1 => constants.%facet_value.cb7
// CHECK:STDOUT:   %Result.as_type.loc293_49.1 => constants.%Int32
// CHECK:STDOUT:   %facet_type => constants.%facet_type.76a
// CHECK:STDOUT:   %pattern_type.loc293 => constants.%pattern_type.839
// CHECK:STDOUT:   %T.patt.loc293_4.2 => constants.%T.patt.845
// CHECK:STDOUT:   %T.loc293_4.1 => constants.%facet_value.1d9
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
// CHECK:STDOUT:   %SelfDestruct.specific_fn.2 => constants.%SelfDestruct.specific_fn.861
// CHECK:STDOUT: }
// CHECK:STDOUT:
