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
// CHECK:STDOUT:   %pattern_type.429: type = pattern_type %Destroy.type [concrete]
// CHECK:STDOUT:   %Result.patt: %pattern_type.429 = symbolic_binding_pattern Result, 0 [symbolic]
// CHECK:STDOUT:   %Result: %Destroy.type = symbolic_binding Result, 0 [symbolic]
// CHECK:STDOUT:   %U.patt.7a2: %pattern_type.429 = symbolic_binding_pattern U, 1 [symbolic]
// CHECK:STDOUT:   %U.f40: %Destroy.type = symbolic_binding U, 1 [symbolic]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Other: type = symbolic_binding Other, 0 [symbolic]
// CHECK:STDOUT:   %AddWith.type.872: type = facet_type <@AddWith.1, @AddWith.1(%Other)> [symbolic]
// CHECK:STDOUT:   %pattern_type.9a5: type = pattern_type type [concrete]
// CHECK:STDOUT:   %Self.958: %AddWith.type.872 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %AddWith.assoc_type.1d5: type = assoc_entity_type @AddWith.1, @AddWith.1(%Other) [symbolic]
// CHECK:STDOUT:   %AddWith.WithSelf.Op.type.70b: type = fn_type @AddWith.WithSelf.Op, @AddWith.WithSelf(%Other, %Self.958) [symbolic]
// CHECK:STDOUT:   %AddWith.WithSelf.Op.bb1: %AddWith.WithSelf.Op.type.70b = struct_value () [symbolic]
// CHECK:STDOUT:   %U.as_type.b61: type = facet_access_type %U.f40 [symbolic]
// CHECK:STDOUT:   %AddWith.type.d58: type = facet_type <@AddWith.1, @AddWith.1(%U.as_type.b61)> [symbolic]
// CHECK:STDOUT:   %facet_type.f36: type = facet_type <@Destroy & @AddWith.1, @AddWith.1(%U.as_type.b61)> [symbolic]
// CHECK:STDOUT:   %.Self.frozen.03f: %facet_type.f36 = symbolic_binding .Self [symbolic]
// CHECK:STDOUT:   %AddWith.assoc_type.964: type = assoc_entity_type @AddWith.1, @AddWith.1(%U.as_type.b61) [symbolic]
// CHECK:STDOUT:   %assoc0.a1a: %AddWith.assoc_type.964 = assoc_entity element0, imports.%Core.import_ref.91f [symbolic]
// CHECK:STDOUT:   %assoc1.261: %AddWith.assoc_type.964 = assoc_entity element1, imports.%Core.import_ref.8ad [symbolic]
// CHECK:STDOUT:   %require_complete.621: <witness> = require_complete_type %facet_type.f36 [symbolic]
// CHECK:STDOUT:   %.Self.frozen.as_type.464: type = facet_access_type %.Self.frozen.03f [symbolic]
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.99b: <witness> = lookup_impl_witness %.Self.frozen.03f, @AddWith.1, @AddWith.1(%U.as_type.b61) [symbolic]
// CHECK:STDOUT:   %impl.elem0.e85: type = impl_witness_access %AddWith.lookup_impl_witness.99b, element0 [symbolic]
// CHECK:STDOUT:   %Result.as_type: type = facet_access_type %Result [symbolic]
// CHECK:STDOUT:   %.Self.787: %facet_type.f36 = symbolic_binding .Self [symbolic]
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.405: <witness> = lookup_impl_witness %.Self.787, @AddWith.1, @AddWith.1(%U.as_type.b61) [symbolic]
// CHECK:STDOUT:   %impl.elem0.f14: type = impl_witness_access %AddWith.lookup_impl_witness.405, element0 [symbolic]
// CHECK:STDOUT:   %facet_type.660: type = facet_type <@Destroy & @AddWith.1, @AddWith.1(%U.as_type.b61) where %impl.elem0.f14 = %Result.as_type> [symbolic]
// CHECK:STDOUT:   %pattern_type.6b8: type = pattern_type %facet_type.660 [symbolic]
// CHECK:STDOUT:   %T.patt.0a9: %pattern_type.6b8 = symbolic_binding_pattern T, 2 [symbolic]
// CHECK:STDOUT:   %T.dd6: %facet_type.660 = symbolic_binding T, 2 [symbolic]
// CHECK:STDOUT:   %pattern_type.f4e485.2: type = pattern_type %Result.as_type [symbolic]
// CHECK:STDOUT:   %result.param_patt.6ba: %pattern_type.f4e485.2 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %result.patt.c73b2c.1: %pattern_type.f4e485.2 = wrapper_binding_pattern result, %result.param_patt.6ba [symbolic]
// CHECK:STDOUT:   %T.as_type.fe6: type = facet_access_type %T.dd6 [symbolic]
// CHECK:STDOUT:   %pattern_type.0df: type = pattern_type %T.as_type.fe6 [symbolic]
// CHECK:STDOUT:   %x.param_patt.018: %pattern_type.0df = value_param_pattern [symbolic]
// CHECK:STDOUT:   %x.patt.3b5: %pattern_type.0df = wrapper_binding_pattern x, %x.param_patt.018 [symbolic]
// CHECK:STDOUT:   %pattern_type.753: type = pattern_type %U.as_type.b61 [symbolic]
// CHECK:STDOUT:   %y.param_patt.740: %pattern_type.753 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %y.patt.603: %pattern_type.753 = wrapper_binding_pattern y, %y.param_patt.740 [symbolic]
// CHECK:STDOUT:   %require_complete.5dc: <witness> = require_complete_type %AddWith.type.d58 [symbolic]
// CHECK:STDOUT:   %assoc1.84a: %AddWith.assoc_type.1d5 = assoc_entity element1, imports.%Core.import_ref.e9a [symbolic]
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.aab: <witness> = lookup_impl_witness %T.dd6, @AddWith.1, @AddWith.1(%U.as_type.b61) [symbolic]
// CHECK:STDOUT:   %AddWith.facet.dad: %AddWith.type.d58 = facet_value %T.as_type.fe6, (%AddWith.lookup_impl_witness.aab) [symbolic]
// CHECK:STDOUT:   %AddWith.WithSelf.Op.type.081: type = fn_type @AddWith.WithSelf.Op, @AddWith.WithSelf(%U.as_type.b61, %AddWith.facet.dad) [symbolic]
// CHECK:STDOUT:   %.612: type = fn_type_with_self_type %AddWith.WithSelf.Op.type.081, %AddWith.facet.dad [symbolic]
// CHECK:STDOUT:   %impl.elem1.9e8: %.612 = impl_witness_access %AddWith.lookup_impl_witness.aab, element1 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.666: <specific function> = specific_impl_function %impl.elem1.9e8, @AddWith.WithSelf.Op(%U.as_type.b61, %AddWith.facet.dad) [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.dfb4c4.2: <witness> = lookup_impl_witness %Result, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.869b78.2: type = fn_type @Destroy.WithSelf.SelfDestruct.1, @Destroy.WithSelf(%Result) [symbolic]
// CHECK:STDOUT:   %.f0c: type = fn_type_with_self_type %Destroy.WithSelf.SelfDestruct.type.869b78.2, %Result [symbolic]
// CHECK:STDOUT:   %impl.elem2: %.f0c = impl_witness_access %Destroy.lookup_impl_witness.dfb4c4.2, element2 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.6f9: <specific function> = specific_impl_function %impl.elem2, @Destroy.WithSelf.SelfDestruct.1(%Result) [symbolic]
// CHECK:STDOUT:   %SubWith.type.73b: type = facet_type <@SubWith.1, @SubWith.1(%Other)> [symbolic]
// CHECK:STDOUT:   %Self.18a: %SubWith.type.73b = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %SubWith.assoc_type.300: type = assoc_entity_type @SubWith.1, @SubWith.1(%Other) [symbolic]
// CHECK:STDOUT:   %SubWith.WithSelf.Op.type.028: type = fn_type @SubWith.WithSelf.Op, @SubWith.WithSelf(%Other, %Self.18a) [symbolic]
// CHECK:STDOUT:   %SubWith.WithSelf.Op.e84: %SubWith.WithSelf.Op.type.028 = struct_value () [symbolic]
// CHECK:STDOUT:   %SubWith.type.d62: type = facet_type <@SubWith.1, @SubWith.1(%U.as_type.b61)> [symbolic]
// CHECK:STDOUT:   %facet_type.e74: type = facet_type <@Destroy & @SubWith.1, @SubWith.1(%U.as_type.b61)> [symbolic]
// CHECK:STDOUT:   %.Self.frozen.d95: %facet_type.e74 = symbolic_binding .Self [symbolic]
// CHECK:STDOUT:   %SubWith.assoc_type.378: type = assoc_entity_type @SubWith.1, @SubWith.1(%U.as_type.b61) [symbolic]
// CHECK:STDOUT:   %assoc0.7c6: %SubWith.assoc_type.378 = assoc_entity element0, imports.%Core.import_ref.190 [symbolic]
// CHECK:STDOUT:   %assoc1.0a7: %SubWith.assoc_type.378 = assoc_entity element1, imports.%Core.import_ref.2f4 [symbolic]
// CHECK:STDOUT:   %require_complete.a5f: <witness> = require_complete_type %facet_type.e74 [symbolic]
// CHECK:STDOUT:   %.Self.frozen.as_type.319: type = facet_access_type %.Self.frozen.d95 [symbolic]
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.d59: <witness> = lookup_impl_witness %.Self.frozen.d95, @SubWith.1, @SubWith.1(%U.as_type.b61) [symbolic]
// CHECK:STDOUT:   %impl.elem0.68a: type = impl_witness_access %SubWith.lookup_impl_witness.d59, element0 [symbolic]
// CHECK:STDOUT:   %.Self.f1c: %facet_type.e74 = symbolic_binding .Self [symbolic]
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.e7f: <witness> = lookup_impl_witness %.Self.f1c, @SubWith.1, @SubWith.1(%U.as_type.b61) [symbolic]
// CHECK:STDOUT:   %impl.elem0.a1a: type = impl_witness_access %SubWith.lookup_impl_witness.e7f, element0 [symbolic]
// CHECK:STDOUT:   %facet_type.ba3: type = facet_type <@Destroy & @SubWith.1, @SubWith.1(%U.as_type.b61) where %impl.elem0.a1a = %Result.as_type> [symbolic]
// CHECK:STDOUT:   %pattern_type.500: type = pattern_type %facet_type.ba3 [symbolic]
// CHECK:STDOUT:   %T.patt.3ed: %pattern_type.500 = symbolic_binding_pattern T, 2 [symbolic]
// CHECK:STDOUT:   %T.34f: %facet_type.ba3 = symbolic_binding T, 2 [symbolic]
// CHECK:STDOUT:   %T.as_type.dfc: type = facet_access_type %T.34f [symbolic]
// CHECK:STDOUT:   %pattern_type.c9e: type = pattern_type %T.as_type.dfc [symbolic]
// CHECK:STDOUT:   %x.param_patt.bd1: %pattern_type.c9e = value_param_pattern [symbolic]
// CHECK:STDOUT:   %x.patt.684: %pattern_type.c9e = wrapper_binding_pattern x, %x.param_patt.bd1 [symbolic]
// CHECK:STDOUT:   %require_complete.c6e: <witness> = require_complete_type %SubWith.type.d62 [symbolic]
// CHECK:STDOUT:   %assoc1.b58: %SubWith.assoc_type.300 = assoc_entity element1, imports.%Core.import_ref.124 [symbolic]
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.1fa: <witness> = lookup_impl_witness %T.34f, @SubWith.1, @SubWith.1(%U.as_type.b61) [symbolic]
// CHECK:STDOUT:   %SubWith.facet.eca: %SubWith.type.d62 = facet_value %T.as_type.dfc, (%SubWith.lookup_impl_witness.1fa) [symbolic]
// CHECK:STDOUT:   %SubWith.WithSelf.Op.type.90e: type = fn_type @SubWith.WithSelf.Op, @SubWith.WithSelf(%U.as_type.b61, %SubWith.facet.eca) [symbolic]
// CHECK:STDOUT:   %.844: type = fn_type_with_self_type %SubWith.WithSelf.Op.type.90e, %SubWith.facet.eca [symbolic]
// CHECK:STDOUT:   %impl.elem1.fa7: %.844 = impl_witness_access %SubWith.lookup_impl_witness.1fa, element1 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.2de: <specific function> = specific_impl_function %impl.elem1.fa7, @SubWith.WithSelf.Op(%U.as_type.b61, %SubWith.facet.eca) [symbolic]
// CHECK:STDOUT:   %MulWith.type.616: type = facet_type <@MulWith.1, @MulWith.1(%Other)> [symbolic]
// CHECK:STDOUT:   %Self.742: %MulWith.type.616 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %MulWith.assoc_type.c81: type = assoc_entity_type @MulWith.1, @MulWith.1(%Other) [symbolic]
// CHECK:STDOUT:   %MulWith.WithSelf.Op.type.dad: type = fn_type @MulWith.WithSelf.Op, @MulWith.WithSelf(%Other, %Self.742) [symbolic]
// CHECK:STDOUT:   %MulWith.WithSelf.Op.507: %MulWith.WithSelf.Op.type.dad = struct_value () [symbolic]
// CHECK:STDOUT:   %MulWith.type.5c9: type = facet_type <@MulWith.1, @MulWith.1(%U.as_type.b61)> [symbolic]
// CHECK:STDOUT:   %facet_type.945: type = facet_type <@Destroy & @MulWith.1, @MulWith.1(%U.as_type.b61)> [symbolic]
// CHECK:STDOUT:   %.Self.frozen.10d: %facet_type.945 = symbolic_binding .Self [symbolic]
// CHECK:STDOUT:   %MulWith.assoc_type.1d7: type = assoc_entity_type @MulWith.1, @MulWith.1(%U.as_type.b61) [symbolic]
// CHECK:STDOUT:   %assoc0.78b: %MulWith.assoc_type.1d7 = assoc_entity element0, imports.%Core.import_ref.59c [symbolic]
// CHECK:STDOUT:   %assoc1.2be: %MulWith.assoc_type.1d7 = assoc_entity element1, imports.%Core.import_ref.fc2 [symbolic]
// CHECK:STDOUT:   %require_complete.264: <witness> = require_complete_type %facet_type.945 [symbolic]
// CHECK:STDOUT:   %.Self.frozen.as_type.f69: type = facet_access_type %.Self.frozen.10d [symbolic]
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.fa1: <witness> = lookup_impl_witness %.Self.frozen.10d, @MulWith.1, @MulWith.1(%U.as_type.b61) [symbolic]
// CHECK:STDOUT:   %impl.elem0.095: type = impl_witness_access %MulWith.lookup_impl_witness.fa1, element0 [symbolic]
// CHECK:STDOUT:   %.Self.cb0: %facet_type.945 = symbolic_binding .Self [symbolic]
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.1ff: <witness> = lookup_impl_witness %.Self.cb0, @MulWith.1, @MulWith.1(%U.as_type.b61) [symbolic]
// CHECK:STDOUT:   %impl.elem0.ad0: type = impl_witness_access %MulWith.lookup_impl_witness.1ff, element0 [symbolic]
// CHECK:STDOUT:   %facet_type.9c7: type = facet_type <@Destroy & @MulWith.1, @MulWith.1(%U.as_type.b61) where %impl.elem0.ad0 = %Result.as_type> [symbolic]
// CHECK:STDOUT:   %pattern_type.112: type = pattern_type %facet_type.9c7 [symbolic]
// CHECK:STDOUT:   %T.patt.9e1: %pattern_type.112 = symbolic_binding_pattern T, 2 [symbolic]
// CHECK:STDOUT:   %T.961: %facet_type.9c7 = symbolic_binding T, 2 [symbolic]
// CHECK:STDOUT:   %T.as_type.521: type = facet_access_type %T.961 [symbolic]
// CHECK:STDOUT:   %pattern_type.3f7: type = pattern_type %T.as_type.521 [symbolic]
// CHECK:STDOUT:   %x.param_patt.a5b: %pattern_type.3f7 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %x.patt.eae: %pattern_type.3f7 = wrapper_binding_pattern x, %x.param_patt.a5b [symbolic]
// CHECK:STDOUT:   %require_complete.dc1: <witness> = require_complete_type %MulWith.type.5c9 [symbolic]
// CHECK:STDOUT:   %assoc1.d9e: %MulWith.assoc_type.c81 = assoc_entity element1, imports.%Core.import_ref.29d [symbolic]
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.ac6: <witness> = lookup_impl_witness %T.961, @MulWith.1, @MulWith.1(%U.as_type.b61) [symbolic]
// CHECK:STDOUT:   %MulWith.facet.cae: %MulWith.type.5c9 = facet_value %T.as_type.521, (%MulWith.lookup_impl_witness.ac6) [symbolic]
// CHECK:STDOUT:   %MulWith.WithSelf.Op.type.6df: type = fn_type @MulWith.WithSelf.Op, @MulWith.WithSelf(%U.as_type.b61, %MulWith.facet.cae) [symbolic]
// CHECK:STDOUT:   %.d85: type = fn_type_with_self_type %MulWith.WithSelf.Op.type.6df, %MulWith.facet.cae [symbolic]
// CHECK:STDOUT:   %impl.elem1.5cc: %.d85 = impl_witness_access %MulWith.lookup_impl_witness.ac6, element1 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.2cd: <specific function> = specific_impl_function %impl.elem1.5cc, @MulWith.WithSelf.Op(%U.as_type.b61, %MulWith.facet.cae) [symbolic]
// CHECK:STDOUT:   %DivWith.type.472: type = facet_type <@DivWith.1, @DivWith.1(%Other)> [symbolic]
// CHECK:STDOUT:   %Self.5b3: %DivWith.type.472 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %DivWith.assoc_type.e7d: type = assoc_entity_type @DivWith.1, @DivWith.1(%Other) [symbolic]
// CHECK:STDOUT:   %DivWith.WithSelf.Op.type.207: type = fn_type @DivWith.WithSelf.Op, @DivWith.WithSelf(%Other, %Self.5b3) [symbolic]
// CHECK:STDOUT:   %DivWith.WithSelf.Op.4f1: %DivWith.WithSelf.Op.type.207 = struct_value () [symbolic]
// CHECK:STDOUT:   %DivWith.type.e82: type = facet_type <@DivWith.1, @DivWith.1(%U.as_type.b61)> [symbolic]
// CHECK:STDOUT:   %facet_type.084: type = facet_type <@Destroy & @DivWith.1, @DivWith.1(%U.as_type.b61)> [symbolic]
// CHECK:STDOUT:   %.Self.frozen.8e5: %facet_type.084 = symbolic_binding .Self [symbolic]
// CHECK:STDOUT:   %DivWith.assoc_type.248: type = assoc_entity_type @DivWith.1, @DivWith.1(%U.as_type.b61) [symbolic]
// CHECK:STDOUT:   %assoc0.b46: %DivWith.assoc_type.248 = assoc_entity element0, imports.%Core.import_ref.dee [symbolic]
// CHECK:STDOUT:   %assoc1.6e8: %DivWith.assoc_type.248 = assoc_entity element1, imports.%Core.import_ref.f2f [symbolic]
// CHECK:STDOUT:   %require_complete.1ee: <witness> = require_complete_type %facet_type.084 [symbolic]
// CHECK:STDOUT:   %.Self.frozen.as_type.379: type = facet_access_type %.Self.frozen.8e5 [symbolic]
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.cd9: <witness> = lookup_impl_witness %.Self.frozen.8e5, @DivWith.1, @DivWith.1(%U.as_type.b61) [symbolic]
// CHECK:STDOUT:   %impl.elem0.76b: type = impl_witness_access %DivWith.lookup_impl_witness.cd9, element0 [symbolic]
// CHECK:STDOUT:   %.Self.f43: %facet_type.084 = symbolic_binding .Self [symbolic]
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.f1e: <witness> = lookup_impl_witness %.Self.f43, @DivWith.1, @DivWith.1(%U.as_type.b61) [symbolic]
// CHECK:STDOUT:   %impl.elem0.bba: type = impl_witness_access %DivWith.lookup_impl_witness.f1e, element0 [symbolic]
// CHECK:STDOUT:   %facet_type.89a: type = facet_type <@Destroy & @DivWith.1, @DivWith.1(%U.as_type.b61) where %impl.elem0.bba = %Result.as_type> [symbolic]
// CHECK:STDOUT:   %pattern_type.658: type = pattern_type %facet_type.89a [symbolic]
// CHECK:STDOUT:   %T.patt.506: %pattern_type.658 = symbolic_binding_pattern T, 2 [symbolic]
// CHECK:STDOUT:   %T.d59: %facet_type.89a = symbolic_binding T, 2 [symbolic]
// CHECK:STDOUT:   %T.as_type.881: type = facet_access_type %T.d59 [symbolic]
// CHECK:STDOUT:   %pattern_type.a65: type = pattern_type %T.as_type.881 [symbolic]
// CHECK:STDOUT:   %x.param_patt.8e8: %pattern_type.a65 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %x.patt.d40: %pattern_type.a65 = wrapper_binding_pattern x, %x.param_patt.8e8 [symbolic]
// CHECK:STDOUT:   %require_complete.4b7: <witness> = require_complete_type %DivWith.type.e82 [symbolic]
// CHECK:STDOUT:   %assoc1.2b8: %DivWith.assoc_type.e7d = assoc_entity element1, imports.%Core.import_ref.94d [symbolic]
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.c25: <witness> = lookup_impl_witness %T.d59, @DivWith.1, @DivWith.1(%U.as_type.b61) [symbolic]
// CHECK:STDOUT:   %DivWith.facet.7d7: %DivWith.type.e82 = facet_value %T.as_type.881, (%DivWith.lookup_impl_witness.c25) [symbolic]
// CHECK:STDOUT:   %DivWith.WithSelf.Op.type.0d7: type = fn_type @DivWith.WithSelf.Op, @DivWith.WithSelf(%U.as_type.b61, %DivWith.facet.7d7) [symbolic]
// CHECK:STDOUT:   %.ba0: type = fn_type_with_self_type %DivWith.WithSelf.Op.type.0d7, %DivWith.facet.7d7 [symbolic]
// CHECK:STDOUT:   %impl.elem1.610: %.ba0 = impl_witness_access %DivWith.lookup_impl_witness.c25, element1 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.7d6: <specific function> = specific_impl_function %impl.elem1.610, @DivWith.WithSelf.Op(%U.as_type.b61, %DivWith.facet.7d7) [symbolic]
// CHECK:STDOUT:   %ModWith.type.11e: type = facet_type <@ModWith.1, @ModWith.1(%Other)> [symbolic]
// CHECK:STDOUT:   %Self.c18: %ModWith.type.11e = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %ModWith.assoc_type.1f1: type = assoc_entity_type @ModWith.1, @ModWith.1(%Other) [symbolic]
// CHECK:STDOUT:   %ModWith.WithSelf.Op.type.04e: type = fn_type @ModWith.WithSelf.Op, @ModWith.WithSelf(%Other, %Self.c18) [symbolic]
// CHECK:STDOUT:   %ModWith.WithSelf.Op.6fb: %ModWith.WithSelf.Op.type.04e = struct_value () [symbolic]
// CHECK:STDOUT:   %ModWith.type.475: type = facet_type <@ModWith.1, @ModWith.1(%U.as_type.b61)> [symbolic]
// CHECK:STDOUT:   %facet_type.fd4: type = facet_type <@Destroy & @ModWith.1, @ModWith.1(%U.as_type.b61)> [symbolic]
// CHECK:STDOUT:   %.Self.frozen.fe0: %facet_type.fd4 = symbolic_binding .Self [symbolic]
// CHECK:STDOUT:   %ModWith.assoc_type.85f: type = assoc_entity_type @ModWith.1, @ModWith.1(%U.as_type.b61) [symbolic]
// CHECK:STDOUT:   %assoc0.bf1: %ModWith.assoc_type.85f = assoc_entity element0, imports.%Core.import_ref.61a [symbolic]
// CHECK:STDOUT:   %assoc1.a6b: %ModWith.assoc_type.85f = assoc_entity element1, imports.%Core.import_ref.9f8f [symbolic]
// CHECK:STDOUT:   %require_complete.8f0: <witness> = require_complete_type %facet_type.fd4 [symbolic]
// CHECK:STDOUT:   %.Self.frozen.as_type.78c: type = facet_access_type %.Self.frozen.fe0 [symbolic]
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.448: <witness> = lookup_impl_witness %.Self.frozen.fe0, @ModWith.1, @ModWith.1(%U.as_type.b61) [symbolic]
// CHECK:STDOUT:   %impl.elem0.043: type = impl_witness_access %ModWith.lookup_impl_witness.448, element0 [symbolic]
// CHECK:STDOUT:   %.Self.dd1: %facet_type.fd4 = symbolic_binding .Self [symbolic]
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.4d5: <witness> = lookup_impl_witness %.Self.dd1, @ModWith.1, @ModWith.1(%U.as_type.b61) [symbolic]
// CHECK:STDOUT:   %impl.elem0.cb3: type = impl_witness_access %ModWith.lookup_impl_witness.4d5, element0 [symbolic]
// CHECK:STDOUT:   %facet_type.247: type = facet_type <@Destroy & @ModWith.1, @ModWith.1(%U.as_type.b61) where %impl.elem0.cb3 = %Result.as_type> [symbolic]
// CHECK:STDOUT:   %pattern_type.c70: type = pattern_type %facet_type.247 [symbolic]
// CHECK:STDOUT:   %T.patt.7fe: %pattern_type.c70 = symbolic_binding_pattern T, 2 [symbolic]
// CHECK:STDOUT:   %T.be8: %facet_type.247 = symbolic_binding T, 2 [symbolic]
// CHECK:STDOUT:   %T.as_type.8ce: type = facet_access_type %T.be8 [symbolic]
// CHECK:STDOUT:   %pattern_type.5e2: type = pattern_type %T.as_type.8ce [symbolic]
// CHECK:STDOUT:   %x.param_patt.41c: %pattern_type.5e2 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %x.patt.2de: %pattern_type.5e2 = wrapper_binding_pattern x, %x.param_patt.41c [symbolic]
// CHECK:STDOUT:   %require_complete.421: <witness> = require_complete_type %ModWith.type.475 [symbolic]
// CHECK:STDOUT:   %assoc1.f4b: %ModWith.assoc_type.1f1 = assoc_entity element1, imports.%Core.import_ref.be0 [symbolic]
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.8e1: <witness> = lookup_impl_witness %T.be8, @ModWith.1, @ModWith.1(%U.as_type.b61) [symbolic]
// CHECK:STDOUT:   %ModWith.facet.204: %ModWith.type.475 = facet_value %T.as_type.8ce, (%ModWith.lookup_impl_witness.8e1) [symbolic]
// CHECK:STDOUT:   %ModWith.WithSelf.Op.type.340: type = fn_type @ModWith.WithSelf.Op, @ModWith.WithSelf(%U.as_type.b61, %ModWith.facet.204) [symbolic]
// CHECK:STDOUT:   %.c4c: type = fn_type_with_self_type %ModWith.WithSelf.Op.type.340, %ModWith.facet.204 [symbolic]
// CHECK:STDOUT:   %impl.elem1.789: %.c4c = impl_witness_access %ModWith.lookup_impl_witness.8e1, element1 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.87e: <specific function> = specific_impl_function %impl.elem1.789, @ModWith.WithSelf.Op(%U.as_type.b61, %ModWith.facet.204) [symbolic]
// CHECK:STDOUT:   %Int16: type = class_type @Int16 [concrete]
// CHECK:STDOUT:   %pattern_type.785: type = pattern_type %Int16 [concrete]
// CHECK:STDOUT:   %Int32: type = class_type @Int32 [concrete]
// CHECK:STDOUT:   %pattern_type.8b4: type = pattern_type %Int32 [concrete]
// CHECK:STDOUT:   %Int64: type = class_type @Int64 [concrete]
// CHECK:STDOUT:   %pattern_type.4aa: type = pattern_type %Int64 [concrete]
// CHECK:STDOUT:   %int_16: Core.IntLiteral = int_value 16 [concrete]
// CHECK:STDOUT:   %i16: type = class_type @Int, @Int(%int_16) [concrete]
// CHECK:STDOUT:   %pattern_type.cbf: type = pattern_type %i16 [concrete]
// CHECK:STDOUT:   %x.param_patt.6cd: %pattern_type.cbf = value_param_pattern [concrete]
// CHECK:STDOUT:   %x.patt.580: %pattern_type.cbf = wrapper_binding_pattern x, %x.param_patt.6cd [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %i16.builtin: type = int_type signed, %int_16 [concrete]
// CHECK:STDOUT:   %complete_type.2da: <witness> = complete_type_witness %i16.builtin [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.1: type = fn_type @Destroy.WithSelf.Op.loc176_34.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.1: %Destroy.WithSelf.Op.type.ef016f.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.1: type = fn_type @Destroy.WithSelf.SelfDestruct.loc176_34.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.1: %Destroy.WithSelf.SelfDestruct.type.fbceb5.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Int16.SubobjectDestroy.type: type = fn_type @Int16.SubobjectDestroy [concrete]
// CHECK:STDOUT:   %Int16.SubobjectDestroy: %Int16.SubobjectDestroy.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.1bfa2a.1: <witness> = custom_witness (%Destroy.WithSelf.Op.403171.1, %Int16.SubobjectDestroy, %Destroy.WithSelf.SelfDestruct.db3fdb.1), @Destroy [concrete]
// CHECK:STDOUT:   %Destroy.facet.f58: %Destroy.type = facet_value %Int16, (%custom_witness.1bfa2a.1) [concrete]
// CHECK:STDOUT:   %facet_type.067: type = facet_type <@Destroy & @AddWith.1, @AddWith.1(%Int16)> [concrete]
// CHECK:STDOUT:   %.Self.4c6: %facet_type.067 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.5df: <witness> = lookup_impl_witness %.Self.4c6, @AddWith.1, @AddWith.1(%Int16) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.2cc: type = impl_witness_access %AddWith.lookup_impl_witness.5df, element0 [symbolic_self]
// CHECK:STDOUT:   %facet_type.754: type = facet_type <@Destroy & @AddWith.1, @AddWith.1(%Int16) where %impl.elem0.2cc = %Int16> [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfc: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc176 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01: %Destroy.WithSelf.SubobjectDestroy.type.dfc = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.2: type = fn_type @Destroy.WithSelf.Op.loc176_34.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.2: %Destroy.WithSelf.Op.type.ef016f.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.2: type = fn_type @Destroy.WithSelf.SelfDestruct.loc176_34.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.2: %Destroy.WithSelf.SelfDestruct.type.fbceb5.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.f8f: <witness> = custom_witness (%Destroy.WithSelf.Op.403171.2, %Destroy.WithSelf.SubobjectDestroy.d01, %Destroy.WithSelf.SelfDestruct.db3fdb.2), @Destroy [concrete]
// CHECK:STDOUT:   %AddWith.type.e4b: type = facet_type <@AddWith.1, @AddWith.1(%Int16)> [concrete]
// CHECK:STDOUT:   %AddWith.assoc_type.86e: type = assoc_entity_type @AddWith.1, @AddWith.1(%Int16) [concrete]
// CHECK:STDOUT:   %assoc0.30c: %AddWith.assoc_type.86e = assoc_entity element0, imports.%Core.import_ref.91f [concrete]
// CHECK:STDOUT:   %assoc1.9bf: %AddWith.assoc_type.86e = assoc_entity element1, imports.%Core.import_ref.8ad [concrete]
// CHECK:STDOUT:   %Op.type.442b68.1: type = fn_type @Op.1 [concrete]
// CHECK:STDOUT:   %Op.72717c.1: %Op.type.442b68.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.facet.091: %Destroy.type = facet_value %i16, (%custom_witness.f8f) [concrete]
// CHECK:STDOUT:   %custom_witness.cba: <witness> = custom_witness (%Int16, %Op.72717c.1), @AddWith.1, @AddWith.1(%Int16) [concrete]
// CHECK:STDOUT:   %facet_value.5e0: %facet_type.754 = facet_value %i16, (%custom_witness.f8f, %custom_witness.cba) [concrete]
// CHECK:STDOUT:   %.Self.frozen.30b: %facet_type.067 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %complete_type.cd3: <witness> = complete_type_witness %facet_type.067 [concrete]
// CHECK:STDOUT:   %.Self.frozen.as_type.bb1: type = facet_access_type %.Self.frozen.30b [symbolic_self]
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.b71: <witness> = lookup_impl_witness %.Self.frozen.30b, @AddWith.1, @AddWith.1(%Int16) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.0bf: type = impl_witness_access %AddWith.lookup_impl_witness.b71, element0 [symbolic_self]
// CHECK:STDOUT:   %pattern_type.9c4: type = pattern_type %facet_type.754 [concrete]
// CHECK:STDOUT:   %T.patt.96c: %pattern_type.9c4 = symbolic_binding_pattern T, 2 [symbolic]
// CHECK:STDOUT:   %result.param_patt.be6: %pattern_type.785 = value_param_pattern [concrete]
// CHECK:STDOUT:   %result.patt.a31: %pattern_type.785 = wrapper_binding_pattern result, %result.param_patt.be6 [concrete]
// CHECK:STDOUT:   %y.param_patt.418: %pattern_type.785 = value_param_pattern [concrete]
// CHECK:STDOUT:   %y.patt.890: %pattern_type.785 = wrapper_binding_pattern y, %y.param_patt.418 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.610: type = fn_type @Destroy.WithSelf.SelfDestruct.1, @Destroy.WithSelf(%Destroy.facet.f58) [concrete]
// CHECK:STDOUT:   %.348: type = fn_type_with_self_type %Destroy.WithSelf.SelfDestruct.type.610, %Destroy.facet.f58 [concrete]
// CHECK:STDOUT:   %facet_type.90f: type = facet_type <@Destroy & @SubWith.1, @SubWith.1(%Int16)> [concrete]
// CHECK:STDOUT:   %.Self.565: %facet_type.90f = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.dd3: <witness> = lookup_impl_witness %.Self.565, @SubWith.1, @SubWith.1(%Int16) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.f92: type = impl_witness_access %SubWith.lookup_impl_witness.dd3, element0 [symbolic_self]
// CHECK:STDOUT:   %facet_type.7bc: type = facet_type <@Destroy & @SubWith.1, @SubWith.1(%Int16) where %impl.elem0.f92 = %Int16> [concrete]
// CHECK:STDOUT:   %SubWith.type.68d: type = facet_type <@SubWith.1, @SubWith.1(%Int16)> [concrete]
// CHECK:STDOUT:   %SubWith.assoc_type.dd6: type = assoc_entity_type @SubWith.1, @SubWith.1(%Int16) [concrete]
// CHECK:STDOUT:   %assoc0.dfb: %SubWith.assoc_type.dd6 = assoc_entity element0, imports.%Core.import_ref.190 [concrete]
// CHECK:STDOUT:   %assoc1.ada: %SubWith.assoc_type.dd6 = assoc_entity element1, imports.%Core.import_ref.2f4 [concrete]
// CHECK:STDOUT:   %Op.type.442b68.2: type = fn_type @Op.2 [concrete]
// CHECK:STDOUT:   %Op.72717c.2: %Op.type.442b68.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.354: <witness> = custom_witness (%Int16, %Op.72717c.2), @SubWith.1, @SubWith.1(%Int16) [concrete]
// CHECK:STDOUT:   %facet_value.ec5: %facet_type.7bc = facet_value %i16, (%custom_witness.f8f, %custom_witness.354) [concrete]
// CHECK:STDOUT:   %.Self.frozen.b1a: %facet_type.90f = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %complete_type.067: <witness> = complete_type_witness %facet_type.90f [concrete]
// CHECK:STDOUT:   %.Self.frozen.as_type.917: type = facet_access_type %.Self.frozen.b1a [symbolic_self]
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.135: <witness> = lookup_impl_witness %.Self.frozen.b1a, @SubWith.1, @SubWith.1(%Int16) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.1d6: type = impl_witness_access %SubWith.lookup_impl_witness.135, element0 [symbolic_self]
// CHECK:STDOUT:   %pattern_type.f9a: type = pattern_type %facet_type.7bc [concrete]
// CHECK:STDOUT:   %T.patt.edb: %pattern_type.f9a = symbolic_binding_pattern T, 2 [symbolic]
// CHECK:STDOUT:   %facet_type.ab1: type = facet_type <@Destroy & @MulWith.1, @MulWith.1(%Int16)> [concrete]
// CHECK:STDOUT:   %.Self.1e1: %facet_type.ab1 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.d39: <witness> = lookup_impl_witness %.Self.1e1, @MulWith.1, @MulWith.1(%Int16) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.979: type = impl_witness_access %MulWith.lookup_impl_witness.d39, element0 [symbolic_self]
// CHECK:STDOUT:   %facet_type.e2b: type = facet_type <@Destroy & @MulWith.1, @MulWith.1(%Int16) where %impl.elem0.979 = %Int16> [concrete]
// CHECK:STDOUT:   %MulWith.type.f65: type = facet_type <@MulWith.1, @MulWith.1(%Int16)> [concrete]
// CHECK:STDOUT:   %MulWith.assoc_type.4c3: type = assoc_entity_type @MulWith.1, @MulWith.1(%Int16) [concrete]
// CHECK:STDOUT:   %assoc0.6ec: %MulWith.assoc_type.4c3 = assoc_entity element0, imports.%Core.import_ref.59c [concrete]
// CHECK:STDOUT:   %assoc1.b14: %MulWith.assoc_type.4c3 = assoc_entity element1, imports.%Core.import_ref.fc2 [concrete]
// CHECK:STDOUT:   %Op.type.442b68.3: type = fn_type @Op.3 [concrete]
// CHECK:STDOUT:   %Op.72717c.3: %Op.type.442b68.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.3c3: <witness> = custom_witness (%Int16, %Op.72717c.3), @MulWith.1, @MulWith.1(%Int16) [concrete]
// CHECK:STDOUT:   %facet_value.7b7: %facet_type.e2b = facet_value %i16, (%custom_witness.f8f, %custom_witness.3c3) [concrete]
// CHECK:STDOUT:   %.Self.frozen.752: %facet_type.ab1 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %complete_type.797: <witness> = complete_type_witness %facet_type.ab1 [concrete]
// CHECK:STDOUT:   %.Self.frozen.as_type.3c5: type = facet_access_type %.Self.frozen.752 [symbolic_self]
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.25c: <witness> = lookup_impl_witness %.Self.frozen.752, @MulWith.1, @MulWith.1(%Int16) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.c8f: type = impl_witness_access %MulWith.lookup_impl_witness.25c, element0 [symbolic_self]
// CHECK:STDOUT:   %pattern_type.fbc: type = pattern_type %facet_type.e2b [concrete]
// CHECK:STDOUT:   %T.patt.d6f: %pattern_type.fbc = symbolic_binding_pattern T, 2 [symbolic]
// CHECK:STDOUT:   %facet_type.b64: type = facet_type <@Destroy & @DivWith.1, @DivWith.1(%Int16)> [concrete]
// CHECK:STDOUT:   %.Self.315: %facet_type.b64 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.147: <witness> = lookup_impl_witness %.Self.315, @DivWith.1, @DivWith.1(%Int16) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.17b: type = impl_witness_access %DivWith.lookup_impl_witness.147, element0 [symbolic_self]
// CHECK:STDOUT:   %facet_type.f21: type = facet_type <@Destroy & @DivWith.1, @DivWith.1(%Int16) where %impl.elem0.17b = %Int16> [concrete]
// CHECK:STDOUT:   %DivWith.type.b8a: type = facet_type <@DivWith.1, @DivWith.1(%Int16)> [concrete]
// CHECK:STDOUT:   %DivWith.assoc_type.712: type = assoc_entity_type @DivWith.1, @DivWith.1(%Int16) [concrete]
// CHECK:STDOUT:   %assoc0.99c: %DivWith.assoc_type.712 = assoc_entity element0, imports.%Core.import_ref.dee [concrete]
// CHECK:STDOUT:   %assoc1.fb6: %DivWith.assoc_type.712 = assoc_entity element1, imports.%Core.import_ref.f2f [concrete]
// CHECK:STDOUT:   %Op.type.442b68.4: type = fn_type @Op.4 [concrete]
// CHECK:STDOUT:   %Op.72717c.4: %Op.type.442b68.4 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.8d0: <witness> = custom_witness (%Int16, %Op.72717c.4), @DivWith.1, @DivWith.1(%Int16) [concrete]
// CHECK:STDOUT:   %facet_value.8d4: %facet_type.f21 = facet_value %i16, (%custom_witness.f8f, %custom_witness.8d0) [concrete]
// CHECK:STDOUT:   %.Self.frozen.797: %facet_type.b64 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %complete_type.ca5: <witness> = complete_type_witness %facet_type.b64 [concrete]
// CHECK:STDOUT:   %.Self.frozen.as_type.b76: type = facet_access_type %.Self.frozen.797 [symbolic_self]
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.60a: <witness> = lookup_impl_witness %.Self.frozen.797, @DivWith.1, @DivWith.1(%Int16) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.431: type = impl_witness_access %DivWith.lookup_impl_witness.60a, element0 [symbolic_self]
// CHECK:STDOUT:   %pattern_type.7ca: type = pattern_type %facet_type.f21 [concrete]
// CHECK:STDOUT:   %T.patt.905: %pattern_type.7ca = symbolic_binding_pattern T, 2 [symbolic]
// CHECK:STDOUT:   %facet_type.2c9: type = facet_type <@Destroy & @ModWith.1, @ModWith.1(%Int16)> [concrete]
// CHECK:STDOUT:   %.Self.7d4: %facet_type.2c9 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.c30: <witness> = lookup_impl_witness %.Self.7d4, @ModWith.1, @ModWith.1(%Int16) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.e30: type = impl_witness_access %ModWith.lookup_impl_witness.c30, element0 [symbolic_self]
// CHECK:STDOUT:   %facet_type.76c: type = facet_type <@Destroy & @ModWith.1, @ModWith.1(%Int16) where %impl.elem0.e30 = %Int16> [concrete]
// CHECK:STDOUT:   %ModWith.type.c67: type = facet_type <@ModWith.1, @ModWith.1(%Int16)> [concrete]
// CHECK:STDOUT:   %ModWith.assoc_type.5f6: type = assoc_entity_type @ModWith.1, @ModWith.1(%Int16) [concrete]
// CHECK:STDOUT:   %assoc0.f04: %ModWith.assoc_type.5f6 = assoc_entity element0, imports.%Core.import_ref.61a [concrete]
// CHECK:STDOUT:   %assoc1.d19: %ModWith.assoc_type.5f6 = assoc_entity element1, imports.%Core.import_ref.9f8f [concrete]
// CHECK:STDOUT:   %Op.type.442b68.5: type = fn_type @Op.5 [concrete]
// CHECK:STDOUT:   %Op.72717c.5: %Op.type.442b68.5 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.169: <witness> = custom_witness (%Int16, %Op.72717c.5), @ModWith.1, @ModWith.1(%Int16) [concrete]
// CHECK:STDOUT:   %facet_value.a68: %facet_type.76c = facet_value %i16, (%custom_witness.f8f, %custom_witness.169) [concrete]
// CHECK:STDOUT:   %.Self.frozen.7c6: %facet_type.2c9 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %complete_type.300: <witness> = complete_type_witness %facet_type.2c9 [concrete]
// CHECK:STDOUT:   %.Self.frozen.as_type.c18: type = facet_access_type %.Self.frozen.7c6 [symbolic_self]
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.49e: <witness> = lookup_impl_witness %.Self.frozen.7c6, @ModWith.1, @ModWith.1(%Int16) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.042: type = impl_witness_access %ModWith.lookup_impl_witness.49e, element0 [symbolic_self]
// CHECK:STDOUT:   %pattern_type.976: type = pattern_type %facet_type.76c [concrete]
// CHECK:STDOUT:   %T.patt.2a9: %pattern_type.976 = symbolic_binding_pattern T, 2 [symbolic]
// CHECK:STDOUT:   %facet_type.c24: type = facet_type <@Destroy & @AddWith.1, @AddWith.1(%i16)> [concrete]
// CHECK:STDOUT:   %.Self.7e7: %facet_type.c24 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.a15: <witness> = lookup_impl_witness %.Self.7e7, @AddWith.1, @AddWith.1(%i16) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.198: type = impl_witness_access %AddWith.lookup_impl_witness.a15, element0 [symbolic_self]
// CHECK:STDOUT:   %facet_type.877: type = facet_type <@Destroy & @AddWith.1, @AddWith.1(%i16) where %impl.elem0.198 = %Int16> [concrete]
// CHECK:STDOUT:   %AddWith.type.631: type = facet_type <@AddWith.1, @AddWith.1(%i16)> [concrete]
// CHECK:STDOUT:   %AddWith.assoc_type.c35: type = assoc_entity_type @AddWith.1, @AddWith.1(%i16) [concrete]
// CHECK:STDOUT:   %assoc0.9a7: %AddWith.assoc_type.c35 = assoc_entity element0, imports.%Core.import_ref.91f [concrete]
// CHECK:STDOUT:   %assoc1.fb4: %AddWith.assoc_type.c35 = assoc_entity element1, imports.%Core.import_ref.8ad [concrete]
// CHECK:STDOUT:   %Int16.Op.type.83af6e.2: type = fn_type @Int16.Op.2 [concrete]
// CHECK:STDOUT:   %Int16.Op.4dca31.2: %Int16.Op.type.83af6e.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.917: <witness> = custom_witness (%Int16, %Int16.Op.4dca31.2), @AddWith.1, @AddWith.1(%i16) [concrete]
// CHECK:STDOUT:   %facet_value.b63: %facet_type.877 = facet_value %Int16, (%custom_witness.1bfa2a.1, %custom_witness.917) [concrete]
// CHECK:STDOUT:   %.Self.frozen.1fa: %facet_type.c24 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %complete_type.448: <witness> = complete_type_witness %facet_type.c24 [concrete]
// CHECK:STDOUT:   %.Self.frozen.as_type.019: type = facet_access_type %.Self.frozen.1fa [symbolic_self]
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.d9d: <witness> = lookup_impl_witness %.Self.frozen.1fa, @AddWith.1, @AddWith.1(%i16) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.1e3: type = impl_witness_access %AddWith.lookup_impl_witness.d9d, element0 [symbolic_self]
// CHECK:STDOUT:   %pattern_type.1bb: type = pattern_type %facet_type.877 [concrete]
// CHECK:STDOUT:   %T.patt.c2e: %pattern_type.1bb = symbolic_binding_pattern T, 2 [symbolic]
// CHECK:STDOUT:   %x.param_patt.3b0: %pattern_type.785 = value_param_pattern [concrete]
// CHECK:STDOUT:   %x.patt.869: %pattern_type.785 = wrapper_binding_pattern x, %x.param_patt.3b0 [concrete]
// CHECK:STDOUT:   %y.param_patt.40e: %pattern_type.cbf = value_param_pattern [concrete]
// CHECK:STDOUT:   %y.patt.b86: %pattern_type.cbf = wrapper_binding_pattern y, %y.param_patt.40e [concrete]
// CHECK:STDOUT:   %facet_type.75c: type = facet_type <@Destroy & @SubWith.1, @SubWith.1(%i16)> [concrete]
// CHECK:STDOUT:   %.Self.2d8: %facet_type.75c = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.80f: <witness> = lookup_impl_witness %.Self.2d8, @SubWith.1, @SubWith.1(%i16) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.bd6: type = impl_witness_access %SubWith.lookup_impl_witness.80f, element0 [symbolic_self]
// CHECK:STDOUT:   %facet_type.d1b: type = facet_type <@Destroy & @SubWith.1, @SubWith.1(%i16) where %impl.elem0.bd6 = %Int16> [concrete]
// CHECK:STDOUT:   %SubWith.type.971: type = facet_type <@SubWith.1, @SubWith.1(%i16)> [concrete]
// CHECK:STDOUT:   %SubWith.assoc_type.6b7: type = assoc_entity_type @SubWith.1, @SubWith.1(%i16) [concrete]
// CHECK:STDOUT:   %assoc0.8aa: %SubWith.assoc_type.6b7 = assoc_entity element0, imports.%Core.import_ref.190 [concrete]
// CHECK:STDOUT:   %assoc1.b62: %SubWith.assoc_type.6b7 = assoc_entity element1, imports.%Core.import_ref.2f4 [concrete]
// CHECK:STDOUT:   %Int16.Op.type.83af6e.3: type = fn_type @Int16.Op.3 [concrete]
// CHECK:STDOUT:   %Int16.Op.4dca31.3: %Int16.Op.type.83af6e.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.1d8: <witness> = custom_witness (%Int16, %Int16.Op.4dca31.3), @SubWith.1, @SubWith.1(%i16) [concrete]
// CHECK:STDOUT:   %facet_value.ab3: %facet_type.d1b = facet_value %Int16, (%custom_witness.1bfa2a.1, %custom_witness.1d8) [concrete]
// CHECK:STDOUT:   %.Self.frozen.2164: %facet_type.75c = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %complete_type.916: <witness> = complete_type_witness %facet_type.75c [concrete]
// CHECK:STDOUT:   %.Self.frozen.as_type.1b5: type = facet_access_type %.Self.frozen.2164 [symbolic_self]
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.f08: <witness> = lookup_impl_witness %.Self.frozen.2164, @SubWith.1, @SubWith.1(%i16) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.ead: type = impl_witness_access %SubWith.lookup_impl_witness.f08, element0 [symbolic_self]
// CHECK:STDOUT:   %pattern_type.5cb: type = pattern_type %facet_type.d1b [concrete]
// CHECK:STDOUT:   %T.patt.83e: %pattern_type.5cb = symbolic_binding_pattern T, 2 [symbolic]
// CHECK:STDOUT:   %facet_type.35c: type = facet_type <@Destroy & @MulWith.1, @MulWith.1(%i16)> [concrete]
// CHECK:STDOUT:   %.Self.6d3: %facet_type.35c = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.eb9: <witness> = lookup_impl_witness %.Self.6d3, @MulWith.1, @MulWith.1(%i16) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.3ca: type = impl_witness_access %MulWith.lookup_impl_witness.eb9, element0 [symbolic_self]
// CHECK:STDOUT:   %facet_type.d3b: type = facet_type <@Destroy & @MulWith.1, @MulWith.1(%i16) where %impl.elem0.3ca = %Int16> [concrete]
// CHECK:STDOUT:   %MulWith.type.f13: type = facet_type <@MulWith.1, @MulWith.1(%i16)> [concrete]
// CHECK:STDOUT:   %MulWith.assoc_type.4df: type = assoc_entity_type @MulWith.1, @MulWith.1(%i16) [concrete]
// CHECK:STDOUT:   %assoc0.958: %MulWith.assoc_type.4df = assoc_entity element0, imports.%Core.import_ref.59c [concrete]
// CHECK:STDOUT:   %assoc1.403: %MulWith.assoc_type.4df = assoc_entity element1, imports.%Core.import_ref.fc2 [concrete]
// CHECK:STDOUT:   %Int16.Op.type.83af6e.4: type = fn_type @Int16.Op.4 [concrete]
// CHECK:STDOUT:   %Int16.Op.4dca31.4: %Int16.Op.type.83af6e.4 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.36e: <witness> = custom_witness (%Int16, %Int16.Op.4dca31.4), @MulWith.1, @MulWith.1(%i16) [concrete]
// CHECK:STDOUT:   %facet_value.19b: %facet_type.d3b = facet_value %Int16, (%custom_witness.1bfa2a.1, %custom_witness.36e) [concrete]
// CHECK:STDOUT:   %.Self.frozen.532: %facet_type.35c = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %complete_type.ea3: <witness> = complete_type_witness %facet_type.35c [concrete]
// CHECK:STDOUT:   %.Self.frozen.as_type.1ed: type = facet_access_type %.Self.frozen.532 [symbolic_self]
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.0d6: <witness> = lookup_impl_witness %.Self.frozen.532, @MulWith.1, @MulWith.1(%i16) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.ea8: type = impl_witness_access %MulWith.lookup_impl_witness.0d6, element0 [symbolic_self]
// CHECK:STDOUT:   %pattern_type.36b: type = pattern_type %facet_type.d3b [concrete]
// CHECK:STDOUT:   %T.patt.173: %pattern_type.36b = symbolic_binding_pattern T, 2 [symbolic]
// CHECK:STDOUT:   %facet_type.39f: type = facet_type <@Destroy & @DivWith.1, @DivWith.1(%i16)> [concrete]
// CHECK:STDOUT:   %.Self.eef: %facet_type.39f = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.25a: <witness> = lookup_impl_witness %.Self.eef, @DivWith.1, @DivWith.1(%i16) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.1c6: type = impl_witness_access %DivWith.lookup_impl_witness.25a, element0 [symbolic_self]
// CHECK:STDOUT:   %facet_type.f5f: type = facet_type <@Destroy & @DivWith.1, @DivWith.1(%i16) where %impl.elem0.1c6 = %Int16> [concrete]
// CHECK:STDOUT:   %DivWith.type.f46: type = facet_type <@DivWith.1, @DivWith.1(%i16)> [concrete]
// CHECK:STDOUT:   %DivWith.assoc_type.7cc: type = assoc_entity_type @DivWith.1, @DivWith.1(%i16) [concrete]
// CHECK:STDOUT:   %assoc0.0cd: %DivWith.assoc_type.7cc = assoc_entity element0, imports.%Core.import_ref.dee [concrete]
// CHECK:STDOUT:   %assoc1.c4f: %DivWith.assoc_type.7cc = assoc_entity element1, imports.%Core.import_ref.f2f [concrete]
// CHECK:STDOUT:   %Int16.Op.type.83af6e.5: type = fn_type @Int16.Op.5 [concrete]
// CHECK:STDOUT:   %Int16.Op.4dca31.5: %Int16.Op.type.83af6e.5 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.5f8: <witness> = custom_witness (%Int16, %Int16.Op.4dca31.5), @DivWith.1, @DivWith.1(%i16) [concrete]
// CHECK:STDOUT:   %facet_value.a28: %facet_type.f5f = facet_value %Int16, (%custom_witness.1bfa2a.1, %custom_witness.5f8) [concrete]
// CHECK:STDOUT:   %.Self.frozen.dd6: %facet_type.39f = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %complete_type.370: <witness> = complete_type_witness %facet_type.39f [concrete]
// CHECK:STDOUT:   %.Self.frozen.as_type.f55: type = facet_access_type %.Self.frozen.dd6 [symbolic_self]
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.804: <witness> = lookup_impl_witness %.Self.frozen.dd6, @DivWith.1, @DivWith.1(%i16) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.358: type = impl_witness_access %DivWith.lookup_impl_witness.804, element0 [symbolic_self]
// CHECK:STDOUT:   %pattern_type.2b9: type = pattern_type %facet_type.f5f [concrete]
// CHECK:STDOUT:   %T.patt.bd4: %pattern_type.2b9 = symbolic_binding_pattern T, 2 [symbolic]
// CHECK:STDOUT:   %facet_type.ce0: type = facet_type <@Destroy & @ModWith.1, @ModWith.1(%i16)> [concrete]
// CHECK:STDOUT:   %.Self.acc: %facet_type.ce0 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.2d5: <witness> = lookup_impl_witness %.Self.acc, @ModWith.1, @ModWith.1(%i16) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.e17: type = impl_witness_access %ModWith.lookup_impl_witness.2d5, element0 [symbolic_self]
// CHECK:STDOUT:   %facet_type.b68: type = facet_type <@Destroy & @ModWith.1, @ModWith.1(%i16) where %impl.elem0.e17 = %Int16> [concrete]
// CHECK:STDOUT:   %ModWith.type.065: type = facet_type <@ModWith.1, @ModWith.1(%i16)> [concrete]
// CHECK:STDOUT:   %ModWith.assoc_type.57d: type = assoc_entity_type @ModWith.1, @ModWith.1(%i16) [concrete]
// CHECK:STDOUT:   %assoc0.529: %ModWith.assoc_type.57d = assoc_entity element0, imports.%Core.import_ref.61a [concrete]
// CHECK:STDOUT:   %assoc1.929: %ModWith.assoc_type.57d = assoc_entity element1, imports.%Core.import_ref.9f8f [concrete]
// CHECK:STDOUT:   %Int16.Op.type.83af6e.6: type = fn_type @Int16.Op.6 [concrete]
// CHECK:STDOUT:   %Int16.Op.4dca31.6: %Int16.Op.type.83af6e.6 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.127: <witness> = custom_witness (%Int16, %Int16.Op.4dca31.6), @ModWith.1, @ModWith.1(%i16) [concrete]
// CHECK:STDOUT:   %facet_value.09b: %facet_type.b68 = facet_value %Int16, (%custom_witness.1bfa2a.1, %custom_witness.127) [concrete]
// CHECK:STDOUT:   %.Self.frozen.817: %facet_type.ce0 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %complete_type.c87: <witness> = complete_type_witness %facet_type.ce0 [concrete]
// CHECK:STDOUT:   %.Self.frozen.as_type.251: type = facet_access_type %.Self.frozen.817 [symbolic_self]
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.7f8: <witness> = lookup_impl_witness %.Self.frozen.817, @ModWith.1, @ModWith.1(%i16) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.6d3: type = impl_witness_access %ModWith.lookup_impl_witness.7f8, element0 [symbolic_self]
// CHECK:STDOUT:   %pattern_type.587: type = pattern_type %facet_type.b68 [concrete]
// CHECK:STDOUT:   %T.patt.e39: %pattern_type.587 = symbolic_binding_pattern T, 2 [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.3: type = fn_type @Destroy.WithSelf.Op.loc190 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.3: %Destroy.WithSelf.Op.type.ef016f.3 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.3: type = fn_type @Destroy.WithSelf.SelfDestruct.loc190 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.3: %Destroy.WithSelf.SelfDestruct.type.fbceb5.3 = struct_value () [concrete]
// CHECK:STDOUT:   %Int32.SubobjectDestroy.type: type = fn_type @Int32.SubobjectDestroy [concrete]
// CHECK:STDOUT:   %Int32.SubobjectDestroy: %Int32.SubobjectDestroy.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.1bfa2a.2: <witness> = custom_witness (%Destroy.WithSelf.Op.403171.3, %Int32.SubobjectDestroy, %Destroy.WithSelf.SelfDestruct.db3fdb.3), @Destroy [concrete]
// CHECK:STDOUT:   %Destroy.facet.84d: %Destroy.type = facet_value %Int32, (%custom_witness.1bfa2a.2) [concrete]
// CHECK:STDOUT:   %facet_type.5cd: type = facet_type <@Destroy & @AddWith.1, @AddWith.1(%Int32)> [concrete]
// CHECK:STDOUT:   %.Self.726: %facet_type.5cd = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.bcf: <witness> = lookup_impl_witness %.Self.726, @AddWith.1, @AddWith.1(%Int32) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.953: type = impl_witness_access %AddWith.lookup_impl_witness.bcf, element0 [symbolic_self]
// CHECK:STDOUT:   %facet_type.030: type = facet_type <@Destroy & @AddWith.1, @AddWith.1(%Int32) where %impl.elem0.953 = %Int32> [concrete]
// CHECK:STDOUT:   %AddWith.type.db7: type = facet_type <@AddWith.1, @AddWith.1(%Int32)> [concrete]
// CHECK:STDOUT:   %AddWith.assoc_type.be8: type = assoc_entity_type @AddWith.1, @AddWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %assoc0.31c: %AddWith.assoc_type.be8 = assoc_entity element0, imports.%Core.import_ref.91f [concrete]
// CHECK:STDOUT:   %assoc1.86c: %AddWith.assoc_type.be8 = assoc_entity element1, imports.%Core.import_ref.8ad [concrete]
// CHECK:STDOUT:   %Int16.Op.type.83af6e.7: type = fn_type @Int16.Op.7 [concrete]
// CHECK:STDOUT:   %Int16.Op.4dca31.7: %Int16.Op.type.83af6e.7 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.513476.1: <witness> = custom_witness (%Int32, %Int16.Op.4dca31.7), @AddWith.1, @AddWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %facet_value.bf4: %facet_type.030 = facet_value %Int16, (%custom_witness.1bfa2a.1, %custom_witness.513476.1) [concrete]
// CHECK:STDOUT:   %.Self.frozen.52a: %facet_type.5cd = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %complete_type.18d: <witness> = complete_type_witness %facet_type.5cd [concrete]
// CHECK:STDOUT:   %.Self.frozen.as_type.988: type = facet_access_type %.Self.frozen.52a [symbolic_self]
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.3ac: <witness> = lookup_impl_witness %.Self.frozen.52a, @AddWith.1, @AddWith.1(%Int32) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.958: type = impl_witness_access %AddWith.lookup_impl_witness.3ac, element0 [symbolic_self]
// CHECK:STDOUT:   %pattern_type.c1b: type = pattern_type %facet_type.030 [concrete]
// CHECK:STDOUT:   %T.patt.16ff: %pattern_type.c1b = symbolic_binding_pattern T, 2 [symbolic]
// CHECK:STDOUT:   %result.param_patt.e81: %pattern_type.8b4 = value_param_pattern [concrete]
// CHECK:STDOUT:   %result.patt.47f: %pattern_type.8b4 = wrapper_binding_pattern result, %result.param_patt.e81 [concrete]
// CHECK:STDOUT:   %y.param_patt.e0c: %pattern_type.8b4 = value_param_pattern [concrete]
// CHECK:STDOUT:   %y.patt.9e1: %pattern_type.8b4 = wrapper_binding_pattern y, %y.param_patt.e0c [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.53c: type = fn_type @Destroy.WithSelf.SelfDestruct.1, @Destroy.WithSelf(%Destroy.facet.84d) [concrete]
// CHECK:STDOUT:   %.77c: type = fn_type_with_self_type %Destroy.WithSelf.SelfDestruct.type.53c, %Destroy.facet.84d [concrete]
// CHECK:STDOUT:   %facet_type.393: type = facet_type <@Destroy & @SubWith.1, @SubWith.1(%Int32)> [concrete]
// CHECK:STDOUT:   %.Self.eac: %facet_type.393 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.0dc: <witness> = lookup_impl_witness %.Self.eac, @SubWith.1, @SubWith.1(%Int32) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.bb5a: type = impl_witness_access %SubWith.lookup_impl_witness.0dc, element0 [symbolic_self]
// CHECK:STDOUT:   %facet_type.767: type = facet_type <@Destroy & @SubWith.1, @SubWith.1(%Int32) where %impl.elem0.bb5a = %Int32> [concrete]
// CHECK:STDOUT:   %SubWith.type.120: type = facet_type <@SubWith.1, @SubWith.1(%Int32)> [concrete]
// CHECK:STDOUT:   %SubWith.assoc_type.dae: type = assoc_entity_type @SubWith.1, @SubWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %assoc0.fac: %SubWith.assoc_type.dae = assoc_entity element0, imports.%Core.import_ref.190 [concrete]
// CHECK:STDOUT:   %assoc1.147: %SubWith.assoc_type.dae = assoc_entity element1, imports.%Core.import_ref.2f4 [concrete]
// CHECK:STDOUT:   %Int16.Op.type.83af6e.8: type = fn_type @Int16.Op.8 [concrete]
// CHECK:STDOUT:   %Int16.Op.4dca31.8: %Int16.Op.type.83af6e.8 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.32cef1.1: <witness> = custom_witness (%Int32, %Int16.Op.4dca31.8), @SubWith.1, @SubWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %facet_value.436: %facet_type.767 = facet_value %Int16, (%custom_witness.1bfa2a.1, %custom_witness.32cef1.1) [concrete]
// CHECK:STDOUT:   %.Self.frozen.36d: %facet_type.393 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %complete_type.8a6: <witness> = complete_type_witness %facet_type.393 [concrete]
// CHECK:STDOUT:   %.Self.frozen.as_type.88b: type = facet_access_type %.Self.frozen.36d [symbolic_self]
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.10e: <witness> = lookup_impl_witness %.Self.frozen.36d, @SubWith.1, @SubWith.1(%Int32) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.d88: type = impl_witness_access %SubWith.lookup_impl_witness.10e, element0 [symbolic_self]
// CHECK:STDOUT:   %pattern_type.974: type = pattern_type %facet_type.767 [concrete]
// CHECK:STDOUT:   %T.patt.3c7: %pattern_type.974 = symbolic_binding_pattern T, 2 [symbolic]
// CHECK:STDOUT:   %facet_type.ea6: type = facet_type <@Destroy & @MulWith.1, @MulWith.1(%Int32)> [concrete]
// CHECK:STDOUT:   %.Self.44f: %facet_type.ea6 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.527: <witness> = lookup_impl_witness %.Self.44f, @MulWith.1, @MulWith.1(%Int32) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.054: type = impl_witness_access %MulWith.lookup_impl_witness.527, element0 [symbolic_self]
// CHECK:STDOUT:   %facet_type.9d8: type = facet_type <@Destroy & @MulWith.1, @MulWith.1(%Int32) where %impl.elem0.054 = %Int32> [concrete]
// CHECK:STDOUT:   %MulWith.type.d67: type = facet_type <@MulWith.1, @MulWith.1(%Int32)> [concrete]
// CHECK:STDOUT:   %MulWith.assoc_type.ab1: type = assoc_entity_type @MulWith.1, @MulWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %assoc0.d0a: %MulWith.assoc_type.ab1 = assoc_entity element0, imports.%Core.import_ref.59c [concrete]
// CHECK:STDOUT:   %assoc1.349: %MulWith.assoc_type.ab1 = assoc_entity element1, imports.%Core.import_ref.fc2 [concrete]
// CHECK:STDOUT:   %Int16.Op.type.83af6e.9: type = fn_type @Int16.Op.9 [concrete]
// CHECK:STDOUT:   %Int16.Op.4dca31.9: %Int16.Op.type.83af6e.9 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.3a0675.1: <witness> = custom_witness (%Int32, %Int16.Op.4dca31.9), @MulWith.1, @MulWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %facet_value.1b5: %facet_type.9d8 = facet_value %Int16, (%custom_witness.1bfa2a.1, %custom_witness.3a0675.1) [concrete]
// CHECK:STDOUT:   %.Self.frozen.95b: %facet_type.ea6 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %complete_type.8fe: <witness> = complete_type_witness %facet_type.ea6 [concrete]
// CHECK:STDOUT:   %.Self.frozen.as_type.f47: type = facet_access_type %.Self.frozen.95b [symbolic_self]
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.675: <witness> = lookup_impl_witness %.Self.frozen.95b, @MulWith.1, @MulWith.1(%Int32) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.1de: type = impl_witness_access %MulWith.lookup_impl_witness.675, element0 [symbolic_self]
// CHECK:STDOUT:   %pattern_type.a89: type = pattern_type %facet_type.9d8 [concrete]
// CHECK:STDOUT:   %T.patt.144: %pattern_type.a89 = symbolic_binding_pattern T, 2 [symbolic]
// CHECK:STDOUT:   %facet_type.821: type = facet_type <@Destroy & @DivWith.1, @DivWith.1(%Int32)> [concrete]
// CHECK:STDOUT:   %.Self.37d: %facet_type.821 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.7ae: <witness> = lookup_impl_witness %.Self.37d, @DivWith.1, @DivWith.1(%Int32) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.6c4: type = impl_witness_access %DivWith.lookup_impl_witness.7ae, element0 [symbolic_self]
// CHECK:STDOUT:   %facet_type.f48: type = facet_type <@Destroy & @DivWith.1, @DivWith.1(%Int32) where %impl.elem0.6c4 = %Int32> [concrete]
// CHECK:STDOUT:   %DivWith.type.5a1: type = facet_type <@DivWith.1, @DivWith.1(%Int32)> [concrete]
// CHECK:STDOUT:   %DivWith.assoc_type.22f: type = assoc_entity_type @DivWith.1, @DivWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %assoc0.a9d: %DivWith.assoc_type.22f = assoc_entity element0, imports.%Core.import_ref.dee [concrete]
// CHECK:STDOUT:   %assoc1.004: %DivWith.assoc_type.22f = assoc_entity element1, imports.%Core.import_ref.f2f [concrete]
// CHECK:STDOUT:   %Int16.Op.type.83af6e.10: type = fn_type @Int16.Op.10 [concrete]
// CHECK:STDOUT:   %Int16.Op.4dca31.10: %Int16.Op.type.83af6e.10 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.c32685.1: <witness> = custom_witness (%Int32, %Int16.Op.4dca31.10), @DivWith.1, @DivWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %facet_value.f68: %facet_type.f48 = facet_value %Int16, (%custom_witness.1bfa2a.1, %custom_witness.c32685.1) [concrete]
// CHECK:STDOUT:   %.Self.frozen.3ed: %facet_type.821 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %complete_type.5bd: <witness> = complete_type_witness %facet_type.821 [concrete]
// CHECK:STDOUT:   %.Self.frozen.as_type.553: type = facet_access_type %.Self.frozen.3ed [symbolic_self]
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.1af: <witness> = lookup_impl_witness %.Self.frozen.3ed, @DivWith.1, @DivWith.1(%Int32) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.a46: type = impl_witness_access %DivWith.lookup_impl_witness.1af, element0 [symbolic_self]
// CHECK:STDOUT:   %pattern_type.4e6: type = pattern_type %facet_type.f48 [concrete]
// CHECK:STDOUT:   %T.patt.def: %pattern_type.4e6 = symbolic_binding_pattern T, 2 [symbolic]
// CHECK:STDOUT:   %facet_type.9c3: type = facet_type <@Destroy & @ModWith.1, @ModWith.1(%Int32)> [concrete]
// CHECK:STDOUT:   %.Self.f60: %facet_type.9c3 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.9a9: <witness> = lookup_impl_witness %.Self.f60, @ModWith.1, @ModWith.1(%Int32) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.522: type = impl_witness_access %ModWith.lookup_impl_witness.9a9, element0 [symbolic_self]
// CHECK:STDOUT:   %facet_type.115: type = facet_type <@Destroy & @ModWith.1, @ModWith.1(%Int32) where %impl.elem0.522 = %Int32> [concrete]
// CHECK:STDOUT:   %ModWith.type.794: type = facet_type <@ModWith.1, @ModWith.1(%Int32)> [concrete]
// CHECK:STDOUT:   %ModWith.assoc_type.021: type = assoc_entity_type @ModWith.1, @ModWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %assoc0.310: %ModWith.assoc_type.021 = assoc_entity element0, imports.%Core.import_ref.61a [concrete]
// CHECK:STDOUT:   %assoc1.2eb: %ModWith.assoc_type.021 = assoc_entity element1, imports.%Core.import_ref.9f8f [concrete]
// CHECK:STDOUT:   %Int16.Op.type.83af6e.11: type = fn_type @Int16.Op.11 [concrete]
// CHECK:STDOUT:   %Int16.Op.4dca31.11: %Int16.Op.type.83af6e.11 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.994c75.1: <witness> = custom_witness (%Int32, %Int16.Op.4dca31.11), @ModWith.1, @ModWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %facet_value.46b: %facet_type.115 = facet_value %Int16, (%custom_witness.1bfa2a.1, %custom_witness.994c75.1) [concrete]
// CHECK:STDOUT:   %.Self.frozen.b381: %facet_type.9c3 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %complete_type.0b7: <witness> = complete_type_witness %facet_type.9c3 [concrete]
// CHECK:STDOUT:   %.Self.frozen.as_type.f01: type = facet_access_type %.Self.frozen.b381 [symbolic_self]
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.ac5: <witness> = lookup_impl_witness %.Self.frozen.b381, @ModWith.1, @ModWith.1(%Int32) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.914: type = impl_witness_access %ModWith.lookup_impl_witness.ac5, element0 [symbolic_self]
// CHECK:STDOUT:   %pattern_type.393: type = pattern_type %facet_type.115 [concrete]
// CHECK:STDOUT:   %T.patt.be9: %pattern_type.393 = symbolic_binding_pattern T, 2 [symbolic]
// CHECK:STDOUT:   %facet_type.788: type = facet_type <@Destroy & @AddWith.1, @AddWith.1(%Int16) where %impl.elem0.2cc = %Int32> [concrete]
// CHECK:STDOUT:   %Int32.Op.type.83af6e.1: type = fn_type @Int32.Op.1 [concrete]
// CHECK:STDOUT:   %Int32.Op.4dca31.1: %Int32.Op.type.83af6e.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.ea1: <witness> = custom_witness (%Int32, %Int32.Op.4dca31.1), @AddWith.1, @AddWith.1(%Int16) [concrete]
// CHECK:STDOUT:   %facet_value.e86: %facet_type.788 = facet_value %Int32, (%custom_witness.1bfa2a.2, %custom_witness.ea1) [concrete]
// CHECK:STDOUT:   %pattern_type.dc6: type = pattern_type %facet_type.788 [concrete]
// CHECK:STDOUT:   %T.patt.0c1: %pattern_type.dc6 = symbolic_binding_pattern T, 2 [symbolic]
// CHECK:STDOUT:   %x.param_patt.27c: %pattern_type.8b4 = value_param_pattern [concrete]
// CHECK:STDOUT:   %x.patt.740: %pattern_type.8b4 = wrapper_binding_pattern x, %x.param_patt.27c [concrete]
// CHECK:STDOUT:   %facet_type.1b0: type = facet_type <@Destroy & @SubWith.1, @SubWith.1(%Int16) where %impl.elem0.f92 = %Int32> [concrete]
// CHECK:STDOUT:   %Int32.Op.type.83af6e.2: type = fn_type @Int32.Op.2 [concrete]
// CHECK:STDOUT:   %Int32.Op.4dca31.2: %Int32.Op.type.83af6e.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.576: <witness> = custom_witness (%Int32, %Int32.Op.4dca31.2), @SubWith.1, @SubWith.1(%Int16) [concrete]
// CHECK:STDOUT:   %facet_value.b78: %facet_type.1b0 = facet_value %Int32, (%custom_witness.1bfa2a.2, %custom_witness.576) [concrete]
// CHECK:STDOUT:   %pattern_type.6d5: type = pattern_type %facet_type.1b0 [concrete]
// CHECK:STDOUT:   %T.patt.282: %pattern_type.6d5 = symbolic_binding_pattern T, 2 [symbolic]
// CHECK:STDOUT:   %facet_type.ccf: type = facet_type <@Destroy & @MulWith.1, @MulWith.1(%Int16) where %impl.elem0.979 = %Int32> [concrete]
// CHECK:STDOUT:   %Int32.Op.type.83af6e.3: type = fn_type @Int32.Op.3 [concrete]
// CHECK:STDOUT:   %Int32.Op.4dca31.3: %Int32.Op.type.83af6e.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.751: <witness> = custom_witness (%Int32, %Int32.Op.4dca31.3), @MulWith.1, @MulWith.1(%Int16) [concrete]
// CHECK:STDOUT:   %facet_value.191: %facet_type.ccf = facet_value %Int32, (%custom_witness.1bfa2a.2, %custom_witness.751) [concrete]
// CHECK:STDOUT:   %pattern_type.d9e: type = pattern_type %facet_type.ccf [concrete]
// CHECK:STDOUT:   %T.patt.879: %pattern_type.d9e = symbolic_binding_pattern T, 2 [symbolic]
// CHECK:STDOUT:   %facet_type.64f: type = facet_type <@Destroy & @DivWith.1, @DivWith.1(%Int16) where %impl.elem0.17b = %Int32> [concrete]
// CHECK:STDOUT:   %Int32.Op.type.83af6e.4: type = fn_type @Int32.Op.4 [concrete]
// CHECK:STDOUT:   %Int32.Op.4dca31.4: %Int32.Op.type.83af6e.4 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.9ad: <witness> = custom_witness (%Int32, %Int32.Op.4dca31.4), @DivWith.1, @DivWith.1(%Int16) [concrete]
// CHECK:STDOUT:   %facet_value.211: %facet_type.64f = facet_value %Int32, (%custom_witness.1bfa2a.2, %custom_witness.9ad) [concrete]
// CHECK:STDOUT:   %pattern_type.b6c: type = pattern_type %facet_type.64f [concrete]
// CHECK:STDOUT:   %T.patt.731: %pattern_type.b6c = symbolic_binding_pattern T, 2 [symbolic]
// CHECK:STDOUT:   %facet_type.b31: type = facet_type <@Destroy & @ModWith.1, @ModWith.1(%Int16) where %impl.elem0.e30 = %Int32> [concrete]
// CHECK:STDOUT:   %Int32.Op.type.83af6e.5: type = fn_type @Int32.Op.5 [concrete]
// CHECK:STDOUT:   %Int32.Op.4dca31.5: %Int32.Op.type.83af6e.5 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.8a2: <witness> = custom_witness (%Int32, %Int32.Op.4dca31.5), @ModWith.1, @ModWith.1(%Int16) [concrete]
// CHECK:STDOUT:   %facet_value.09f: %facet_type.b31 = facet_value %Int32, (%custom_witness.1bfa2a.2, %custom_witness.8a2) [concrete]
// CHECK:STDOUT:   %pattern_type.df6: type = pattern_type %facet_type.b31 [concrete]
// CHECK:STDOUT:   %T.patt.336: %pattern_type.df6 = symbolic_binding_pattern T, 2 [symbolic]
// CHECK:STDOUT:   %Int32.Op.type.83af6e.6: type = fn_type @Int32.Op.6 [concrete]
// CHECK:STDOUT:   %Int32.Op.4dca31.6: %Int32.Op.type.83af6e.6 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.513476.2: <witness> = custom_witness (%Int32, %Int32.Op.4dca31.6), @AddWith.1, @AddWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %facet_value.5ed: %facet_type.030 = facet_value %Int32, (%custom_witness.1bfa2a.2, %custom_witness.513476.2) [concrete]
// CHECK:STDOUT:   %Int32.Op.type.83af6e.7: type = fn_type @Int32.Op.7 [concrete]
// CHECK:STDOUT:   %Int32.Op.4dca31.7: %Int32.Op.type.83af6e.7 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.32cef1.2: <witness> = custom_witness (%Int32, %Int32.Op.4dca31.7), @SubWith.1, @SubWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %facet_value.889: %facet_type.767 = facet_value %Int32, (%custom_witness.1bfa2a.2, %custom_witness.32cef1.2) [concrete]
// CHECK:STDOUT:   %Int32.Op.type.83af6e.8: type = fn_type @Int32.Op.8 [concrete]
// CHECK:STDOUT:   %Int32.Op.4dca31.8: %Int32.Op.type.83af6e.8 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.3a0675.2: <witness> = custom_witness (%Int32, %Int32.Op.4dca31.8), @MulWith.1, @MulWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %facet_value.f7d: %facet_type.9d8 = facet_value %Int32, (%custom_witness.1bfa2a.2, %custom_witness.3a0675.2) [concrete]
// CHECK:STDOUT:   %Int32.Op.type.83af6e.9: type = fn_type @Int32.Op.9 [concrete]
// CHECK:STDOUT:   %Int32.Op.4dca31.9: %Int32.Op.type.83af6e.9 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.c32685.2: <witness> = custom_witness (%Int32, %Int32.Op.4dca31.9), @DivWith.1, @DivWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %facet_value.403: %facet_type.f48 = facet_value %Int32, (%custom_witness.1bfa2a.2, %custom_witness.c32685.2) [concrete]
// CHECK:STDOUT:   %Int32.Op.type.83af6e.10: type = fn_type @Int32.Op.10 [concrete]
// CHECK:STDOUT:   %Int32.Op.4dca31.10: %Int32.Op.type.83af6e.10 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.994c75.2: <witness> = custom_witness (%Int32, %Int32.Op.4dca31.10), @ModWith.1, @ModWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %facet_value.559: %facet_type.115 = facet_value %Int32, (%custom_witness.1bfa2a.2, %custom_witness.994c75.2) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.4: type = fn_type @Destroy.WithSelf.Op.loc211 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.4: %Destroy.WithSelf.Op.type.ef016f.4 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.4: type = fn_type @Destroy.WithSelf.SelfDestruct.loc211 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.4: %Destroy.WithSelf.SelfDestruct.type.fbceb5.4 = struct_value () [concrete]
// CHECK:STDOUT:   %Int64.SubobjectDestroy.type: type = fn_type @Int64.SubobjectDestroy [concrete]
// CHECK:STDOUT:   %Int64.SubobjectDestroy: %Int64.SubobjectDestroy.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.1bfa2a.3: <witness> = custom_witness (%Destroy.WithSelf.Op.403171.4, %Int64.SubobjectDestroy, %Destroy.WithSelf.SelfDestruct.db3fdb.4), @Destroy [concrete]
// CHECK:STDOUT:   %Destroy.facet.0b4: %Destroy.type = facet_value %Int64, (%custom_witness.1bfa2a.3) [concrete]
// CHECK:STDOUT:   %facet_type.e1d: type = facet_type <@Destroy & @AddWith.1, @AddWith.1(%Int64)> [concrete]
// CHECK:STDOUT:   %.Self.ad4: %facet_type.e1d = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.a60: <witness> = lookup_impl_witness %.Self.ad4, @AddWith.1, @AddWith.1(%Int64) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.7fa: type = impl_witness_access %AddWith.lookup_impl_witness.a60, element0 [symbolic_self]
// CHECK:STDOUT:   %facet_type.37a: type = facet_type <@Destroy & @AddWith.1, @AddWith.1(%Int64) where %impl.elem0.7fa = %Int64> [concrete]
// CHECK:STDOUT:   %y.param_patt.a08: %pattern_type.4aa = value_param_pattern [concrete]
// CHECK:STDOUT:   %y.patt.069: %pattern_type.4aa = wrapper_binding_pattern y, %y.param_patt.a08 [concrete]
// CHECK:STDOUT:   %AddWith.type.1ff: type = facet_type <@AddWith.1, @AddWith.1(%Int64)> [concrete]
// CHECK:STDOUT:   %AddWith.assoc_type.97f: type = assoc_entity_type @AddWith.1, @AddWith.1(%Int64) [concrete]
// CHECK:STDOUT:   %assoc0.f02: %AddWith.assoc_type.97f = assoc_entity element0, imports.%Core.import_ref.91f [concrete]
// CHECK:STDOUT:   %assoc1.a75: %AddWith.assoc_type.97f = assoc_entity element1, imports.%Core.import_ref.8ad [concrete]
// CHECK:STDOUT:   %Op.type.442b68.6: type = fn_type @Op.6 [concrete]
// CHECK:STDOUT:   %Op.72717c.6: %Op.type.442b68.6 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.f2f: <witness> = custom_witness (%Int64, %Op.72717c.6), @AddWith.1, @AddWith.1(%Int64) [concrete]
// CHECK:STDOUT:   %facet_value.806: %facet_type.37a = facet_value %Int32, (%custom_witness.1bfa2a.2, %custom_witness.f2f) [concrete]
// CHECK:STDOUT:   %.Self.frozen.d58: %facet_type.e1d = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %complete_type.545: <witness> = complete_type_witness %facet_type.e1d [concrete]
// CHECK:STDOUT:   %.Self.frozen.as_type.f6a: type = facet_access_type %.Self.frozen.d58 [symbolic_self]
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.bc0: <witness> = lookup_impl_witness %.Self.frozen.d58, @AddWith.1, @AddWith.1(%Int64) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.1e6: type = impl_witness_access %AddWith.lookup_impl_witness.bc0, element0 [symbolic_self]
// CHECK:STDOUT:   %pattern_type.ec3: type = pattern_type %facet_type.37a [concrete]
// CHECK:STDOUT:   %T.patt.af0: %pattern_type.ec3 = symbolic_binding_pattern T, 2 [symbolic]
// CHECK:STDOUT:   %result.param_patt.1e1: %pattern_type.4aa = value_param_pattern [concrete]
// CHECK:STDOUT:   %result.patt.93b: %pattern_type.4aa = wrapper_binding_pattern result, %result.param_patt.1e1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.36f: type = fn_type @Destroy.WithSelf.SelfDestruct.1, @Destroy.WithSelf(%Destroy.facet.0b4) [concrete]
// CHECK:STDOUT:   %.7aa: type = fn_type_with_self_type %Destroy.WithSelf.SelfDestruct.type.36f, %Destroy.facet.0b4 [concrete]
// CHECK:STDOUT:   %facet_type.6db: type = facet_type <@Destroy & @SubWith.1, @SubWith.1(%Int64)> [concrete]
// CHECK:STDOUT:   %.Self.b71: %facet_type.6db = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.4b8: <witness> = lookup_impl_witness %.Self.b71, @SubWith.1, @SubWith.1(%Int64) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.5e41: type = impl_witness_access %SubWith.lookup_impl_witness.4b8, element0 [symbolic_self]
// CHECK:STDOUT:   %facet_type.cfa: type = facet_type <@Destroy & @SubWith.1, @SubWith.1(%Int64) where %impl.elem0.5e41 = %Int64> [concrete]
// CHECK:STDOUT:   %SubWith.type.31c: type = facet_type <@SubWith.1, @SubWith.1(%Int64)> [concrete]
// CHECK:STDOUT:   %SubWith.assoc_type.4f8: type = assoc_entity_type @SubWith.1, @SubWith.1(%Int64) [concrete]
// CHECK:STDOUT:   %assoc0.13b: %SubWith.assoc_type.4f8 = assoc_entity element0, imports.%Core.import_ref.190 [concrete]
// CHECK:STDOUT:   %assoc1.e70: %SubWith.assoc_type.4f8 = assoc_entity element1, imports.%Core.import_ref.2f4 [concrete]
// CHECK:STDOUT:   %Op.type.442b68.7: type = fn_type @Op.7 [concrete]
// CHECK:STDOUT:   %Op.72717c.7: %Op.type.442b68.7 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.276: <witness> = custom_witness (%Int64, %Op.72717c.7), @SubWith.1, @SubWith.1(%Int64) [concrete]
// CHECK:STDOUT:   %facet_value.442: %facet_type.cfa = facet_value %Int32, (%custom_witness.1bfa2a.2, %custom_witness.276) [concrete]
// CHECK:STDOUT:   %.Self.frozen.a3d: %facet_type.6db = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %complete_type.b70: <witness> = complete_type_witness %facet_type.6db [concrete]
// CHECK:STDOUT:   %.Self.frozen.as_type.f8c: type = facet_access_type %.Self.frozen.a3d [symbolic_self]
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.117: <witness> = lookup_impl_witness %.Self.frozen.a3d, @SubWith.1, @SubWith.1(%Int64) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.79c: type = impl_witness_access %SubWith.lookup_impl_witness.117, element0 [symbolic_self]
// CHECK:STDOUT:   %pattern_type.16a: type = pattern_type %facet_type.cfa [concrete]
// CHECK:STDOUT:   %T.patt.18c: %pattern_type.16a = symbolic_binding_pattern T, 2 [symbolic]
// CHECK:STDOUT:   %facet_type.dc5: type = facet_type <@Destroy & @MulWith.1, @MulWith.1(%Int64)> [concrete]
// CHECK:STDOUT:   %.Self.6e4: %facet_type.dc5 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.977: <witness> = lookup_impl_witness %.Self.6e4, @MulWith.1, @MulWith.1(%Int64) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.a52: type = impl_witness_access %MulWith.lookup_impl_witness.977, element0 [symbolic_self]
// CHECK:STDOUT:   %facet_type.48a: type = facet_type <@Destroy & @MulWith.1, @MulWith.1(%Int64) where %impl.elem0.a52 = %Int64> [concrete]
// CHECK:STDOUT:   %MulWith.type.661: type = facet_type <@MulWith.1, @MulWith.1(%Int64)> [concrete]
// CHECK:STDOUT:   %MulWith.assoc_type.940: type = assoc_entity_type @MulWith.1, @MulWith.1(%Int64) [concrete]
// CHECK:STDOUT:   %assoc0.9d9: %MulWith.assoc_type.940 = assoc_entity element0, imports.%Core.import_ref.59c [concrete]
// CHECK:STDOUT:   %assoc1.b22: %MulWith.assoc_type.940 = assoc_entity element1, imports.%Core.import_ref.fc2 [concrete]
// CHECK:STDOUT:   %Op.type.442b68.8: type = fn_type @Op.8 [concrete]
// CHECK:STDOUT:   %Op.72717c.8: %Op.type.442b68.8 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.8e4: <witness> = custom_witness (%Int64, %Op.72717c.8), @MulWith.1, @MulWith.1(%Int64) [concrete]
// CHECK:STDOUT:   %facet_value.97e: %facet_type.48a = facet_value %Int32, (%custom_witness.1bfa2a.2, %custom_witness.8e4) [concrete]
// CHECK:STDOUT:   %.Self.frozen.1e0: %facet_type.dc5 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %complete_type.757: <witness> = complete_type_witness %facet_type.dc5 [concrete]
// CHECK:STDOUT:   %.Self.frozen.as_type.5a6: type = facet_access_type %.Self.frozen.1e0 [symbolic_self]
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.741: <witness> = lookup_impl_witness %.Self.frozen.1e0, @MulWith.1, @MulWith.1(%Int64) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.dca: type = impl_witness_access %MulWith.lookup_impl_witness.741, element0 [symbolic_self]
// CHECK:STDOUT:   %pattern_type.307: type = pattern_type %facet_type.48a [concrete]
// CHECK:STDOUT:   %T.patt.23a: %pattern_type.307 = symbolic_binding_pattern T, 2 [symbolic]
// CHECK:STDOUT:   %facet_type.7ad: type = facet_type <@Destroy & @DivWith.1, @DivWith.1(%Int64)> [concrete]
// CHECK:STDOUT:   %.Self.7b8: %facet_type.7ad = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.055: <witness> = lookup_impl_witness %.Self.7b8, @DivWith.1, @DivWith.1(%Int64) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.212: type = impl_witness_access %DivWith.lookup_impl_witness.055, element0 [symbolic_self]
// CHECK:STDOUT:   %facet_type.011: type = facet_type <@Destroy & @DivWith.1, @DivWith.1(%Int64) where %impl.elem0.212 = %Int64> [concrete]
// CHECK:STDOUT:   %DivWith.type.dd8: type = facet_type <@DivWith.1, @DivWith.1(%Int64)> [concrete]
// CHECK:STDOUT:   %DivWith.assoc_type.2dc: type = assoc_entity_type @DivWith.1, @DivWith.1(%Int64) [concrete]
// CHECK:STDOUT:   %assoc0.e8b: %DivWith.assoc_type.2dc = assoc_entity element0, imports.%Core.import_ref.dee [concrete]
// CHECK:STDOUT:   %assoc1.8b9: %DivWith.assoc_type.2dc = assoc_entity element1, imports.%Core.import_ref.f2f [concrete]
// CHECK:STDOUT:   %Op.type.442b68.9: type = fn_type @Op.9 [concrete]
// CHECK:STDOUT:   %Op.72717c.9: %Op.type.442b68.9 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.099: <witness> = custom_witness (%Int64, %Op.72717c.9), @DivWith.1, @DivWith.1(%Int64) [concrete]
// CHECK:STDOUT:   %facet_value.585: %facet_type.011 = facet_value %Int32, (%custom_witness.1bfa2a.2, %custom_witness.099) [concrete]
// CHECK:STDOUT:   %.Self.frozen.2167: %facet_type.7ad = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %complete_type.c23: <witness> = complete_type_witness %facet_type.7ad [concrete]
// CHECK:STDOUT:   %.Self.frozen.as_type.c77: type = facet_access_type %.Self.frozen.2167 [symbolic_self]
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.958: <witness> = lookup_impl_witness %.Self.frozen.2167, @DivWith.1, @DivWith.1(%Int64) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.a26: type = impl_witness_access %DivWith.lookup_impl_witness.958, element0 [symbolic_self]
// CHECK:STDOUT:   %pattern_type.5eb: type = pattern_type %facet_type.011 [concrete]
// CHECK:STDOUT:   %T.patt.518: %pattern_type.5eb = symbolic_binding_pattern T, 2 [symbolic]
// CHECK:STDOUT:   %facet_type.830: type = facet_type <@Destroy & @ModWith.1, @ModWith.1(%Int64)> [concrete]
// CHECK:STDOUT:   %.Self.23a: %facet_type.830 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.4a5: <witness> = lookup_impl_witness %.Self.23a, @ModWith.1, @ModWith.1(%Int64) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.007: type = impl_witness_access %ModWith.lookup_impl_witness.4a5, element0 [symbolic_self]
// CHECK:STDOUT:   %facet_type.2db: type = facet_type <@Destroy & @ModWith.1, @ModWith.1(%Int64) where %impl.elem0.007 = %Int64> [concrete]
// CHECK:STDOUT:   %ModWith.type.c72: type = facet_type <@ModWith.1, @ModWith.1(%Int64)> [concrete]
// CHECK:STDOUT:   %ModWith.assoc_type.ad2: type = assoc_entity_type @ModWith.1, @ModWith.1(%Int64) [concrete]
// CHECK:STDOUT:   %assoc0.c8a: %ModWith.assoc_type.ad2 = assoc_entity element0, imports.%Core.import_ref.61a [concrete]
// CHECK:STDOUT:   %assoc1.eed: %ModWith.assoc_type.ad2 = assoc_entity element1, imports.%Core.import_ref.9f8f [concrete]
// CHECK:STDOUT:   %Op.type.442b68.10: type = fn_type @Op.10 [concrete]
// CHECK:STDOUT:   %Op.72717c.10: %Op.type.442b68.10 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.129: <witness> = custom_witness (%Int64, %Op.72717c.10), @ModWith.1, @ModWith.1(%Int64) [concrete]
// CHECK:STDOUT:   %facet_value.b2a: %facet_type.2db = facet_value %Int32, (%custom_witness.1bfa2a.2, %custom_witness.129) [concrete]
// CHECK:STDOUT:   %.Self.frozen.5a8: %facet_type.830 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %complete_type.ecd: <witness> = complete_type_witness %facet_type.830 [concrete]
// CHECK:STDOUT:   %.Self.frozen.as_type.7bd: type = facet_access_type %.Self.frozen.5a8 [symbolic_self]
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.601: <witness> = lookup_impl_witness %.Self.frozen.5a8, @ModWith.1, @ModWith.1(%Int64) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.001: type = impl_witness_access %ModWith.lookup_impl_witness.601, element0 [symbolic_self]
// CHECK:STDOUT:   %pattern_type.3f3: type = pattern_type %facet_type.2db [concrete]
// CHECK:STDOUT:   %T.patt.3de: %pattern_type.3f3 = symbolic_binding_pattern T, 2 [symbolic]
// CHECK:STDOUT:   %facet_type.223: type = facet_type <@Destroy & @AddWith.1, @AddWith.1(%Int32) where %impl.elem0.953 = %Int64> [concrete]
// CHECK:STDOUT:   %Op.type.442b68.11: type = fn_type @Op.11 [concrete]
// CHECK:STDOUT:   %Op.72717c.11: %Op.type.442b68.11 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.16b: <witness> = custom_witness (%Int64, %Op.72717c.11), @AddWith.1, @AddWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %facet_value.26c: %facet_type.223 = facet_value %Int64, (%custom_witness.1bfa2a.3, %custom_witness.16b) [concrete]
// CHECK:STDOUT:   %pattern_type.2c7: type = pattern_type %facet_type.223 [concrete]
// CHECK:STDOUT:   %T.patt.e6f: %pattern_type.2c7 = symbolic_binding_pattern T, 2 [symbolic]
// CHECK:STDOUT:   %x.param_patt.e68: %pattern_type.4aa = value_param_pattern [concrete]
// CHECK:STDOUT:   %x.patt.7aa: %pattern_type.4aa = wrapper_binding_pattern x, %x.param_patt.e68 [concrete]
// CHECK:STDOUT:   %facet_type.ecd: type = facet_type <@Destroy & @SubWith.1, @SubWith.1(%Int32) where %impl.elem0.bb5a = %Int64> [concrete]
// CHECK:STDOUT:   %Op.type.442b68.12: type = fn_type @Op.12 [concrete]
// CHECK:STDOUT:   %Op.72717c.12: %Op.type.442b68.12 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.21a: <witness> = custom_witness (%Int64, %Op.72717c.12), @SubWith.1, @SubWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %facet_value.d8b: %facet_type.ecd = facet_value %Int64, (%custom_witness.1bfa2a.3, %custom_witness.21a) [concrete]
// CHECK:STDOUT:   %pattern_type.ab2: type = pattern_type %facet_type.ecd [concrete]
// CHECK:STDOUT:   %T.patt.7e5: %pattern_type.ab2 = symbolic_binding_pattern T, 2 [symbolic]
// CHECK:STDOUT:   %facet_type.486: type = facet_type <@Destroy & @MulWith.1, @MulWith.1(%Int32) where %impl.elem0.054 = %Int64> [concrete]
// CHECK:STDOUT:   %Op.type.442b68.13: type = fn_type @Op.13 [concrete]
// CHECK:STDOUT:   %Op.72717c.13: %Op.type.442b68.13 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.798: <witness> = custom_witness (%Int64, %Op.72717c.13), @MulWith.1, @MulWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %facet_value.72f: %facet_type.486 = facet_value %Int64, (%custom_witness.1bfa2a.3, %custom_witness.798) [concrete]
// CHECK:STDOUT:   %pattern_type.92d: type = pattern_type %facet_type.486 [concrete]
// CHECK:STDOUT:   %T.patt.2f0: %pattern_type.92d = symbolic_binding_pattern T, 2 [symbolic]
// CHECK:STDOUT:   %facet_type.752: type = facet_type <@Destroy & @DivWith.1, @DivWith.1(%Int32) where %impl.elem0.6c4 = %Int64> [concrete]
// CHECK:STDOUT:   %Op.type.442b68.14: type = fn_type @Op.14 [concrete]
// CHECK:STDOUT:   %Op.72717c.14: %Op.type.442b68.14 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.37f: <witness> = custom_witness (%Int64, %Op.72717c.14), @DivWith.1, @DivWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %facet_value.c74: %facet_type.752 = facet_value %Int64, (%custom_witness.1bfa2a.3, %custom_witness.37f) [concrete]
// CHECK:STDOUT:   %pattern_type.d51: type = pattern_type %facet_type.752 [concrete]
// CHECK:STDOUT:   %T.patt.c8a: %pattern_type.d51 = symbolic_binding_pattern T, 2 [symbolic]
// CHECK:STDOUT:   %facet_type.088: type = facet_type <@Destroy & @ModWith.1, @ModWith.1(%Int32) where %impl.elem0.522 = %Int64> [concrete]
// CHECK:STDOUT:   %Op.type.442b68.15: type = fn_type @Op.15 [concrete]
// CHECK:STDOUT:   %Op.72717c.15: %Op.type.442b68.15 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.981: <witness> = custom_witness (%Int64, %Op.72717c.15), @ModWith.1, @ModWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %facet_value.b3f: %facet_type.088 = facet_value %Int64, (%custom_witness.1bfa2a.3, %custom_witness.981) [concrete]
// CHECK:STDOUT:   %pattern_type.fff: type = pattern_type %facet_type.088 [concrete]
// CHECK:STDOUT:   %T.patt.1fc: %pattern_type.fff = symbolic_binding_pattern T, 2 [symbolic]
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
// CHECK:STDOUT:   %assoc0.a11270.2: %ModAssignWith.assoc_type.01d13e.2 = assoc_entity element0, imports.%Core.import_ref.97f5 [symbolic]
// CHECK:STDOUT:   %require_complete.a87: <witness> = require_complete_type %ModAssignWith.type.3a0f29.2 [symbolic]
// CHECK:STDOUT:   %assoc0.3a8: %ModAssignWith.assoc_type.01d13e.1 = assoc_entity element0, imports.%Core.import_ref.ffb [symbolic]
// CHECK:STDOUT:   %ModAssignWith.lookup_impl_witness: <witness> = lookup_impl_witness %T.cc0, @ModAssignWith.1, @ModAssignWith.1(%U.67d) [symbolic]
// CHECK:STDOUT:   %ModAssignWith.WithSelf.Op.type.7f247d.3: type = fn_type @ModAssignWith.WithSelf.Op, @ModAssignWith.WithSelf(%U.67d, %T.cc0) [symbolic]
// CHECK:STDOUT:   %.99f: type = fn_type_with_self_type %ModAssignWith.WithSelf.Op.type.7f247d.3, %T.cc0 [symbolic]
// CHECK:STDOUT:   %impl.elem0.46a: %.99f = impl_witness_access %ModAssignWith.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.caf: <specific function> = specific_impl_function %impl.elem0.46a, @ModAssignWith.WithSelf.Op(%U.67d, %T.cc0) [symbolic]
// CHECK:STDOUT:   %AddAssignWith.type.b50: type = facet_type <@AddAssignWith.1, @AddAssignWith.1(%i16)> [concrete]
// CHECK:STDOUT:   %AddAssignWith.assoc_type.75b: type = assoc_entity_type @AddAssignWith.1, @AddAssignWith.1(%i16) [concrete]
// CHECK:STDOUT:   %assoc0.017: %AddAssignWith.assoc_type.75b = assoc_entity element0, imports.%Core.import_ref.90f [concrete]
// CHECK:STDOUT:   %Int16.Op.type.83af6e.12: type = fn_type @Int16.Op.12 [concrete]
// CHECK:STDOUT:   %Int16.Op.4dca31.12: %Int16.Op.type.83af6e.12 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.af6: <witness> = custom_witness (%Int16.Op.4dca31.12), @AddAssignWith.1, @AddAssignWith.1(%i16) [concrete]
// CHECK:STDOUT:   %AddAssignWith.facet.510: %AddAssignWith.type.b50 = facet_value %Int16, (%custom_witness.af6) [concrete]
// CHECK:STDOUT:   %pattern_type.010: type = pattern_type %AddAssignWith.type.b50 [concrete]
// CHECK:STDOUT:   %T.patt.0f8: %pattern_type.010 = symbolic_binding_pattern T, 1 [symbolic]
// CHECK:STDOUT:   %x.patt.49e: %pattern_type.785 = ref_binding_pattern x [concrete]
// CHECK:STDOUT:   %x.param_patt.69a: %pattern_type.785 = var_param_pattern %x.patt.49e [concrete]
// CHECK:STDOUT:   %SubAssignWith.type.5c1: type = facet_type <@SubAssignWith.1, @SubAssignWith.1(%i16)> [concrete]
// CHECK:STDOUT:   %SubAssignWith.assoc_type.2e5: type = assoc_entity_type @SubAssignWith.1, @SubAssignWith.1(%i16) [concrete]
// CHECK:STDOUT:   %assoc0.7e9: %SubAssignWith.assoc_type.2e5 = assoc_entity element0, imports.%Core.import_ref.8e5 [concrete]
// CHECK:STDOUT:   %Int16.Op.type.83af6e.13: type = fn_type @Int16.Op.13 [concrete]
// CHECK:STDOUT:   %Int16.Op.4dca31.13: %Int16.Op.type.83af6e.13 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.ba9: <witness> = custom_witness (%Int16.Op.4dca31.13), @SubAssignWith.1, @SubAssignWith.1(%i16) [concrete]
// CHECK:STDOUT:   %SubAssignWith.facet.6c0: %SubAssignWith.type.5c1 = facet_value %Int16, (%custom_witness.ba9) [concrete]
// CHECK:STDOUT:   %pattern_type.7e6: type = pattern_type %SubAssignWith.type.5c1 [concrete]
// CHECK:STDOUT:   %T.patt.a81: %pattern_type.7e6 = symbolic_binding_pattern T, 1 [symbolic]
// CHECK:STDOUT:   %MulAssignWith.type.8be: type = facet_type <@MulAssignWith.1, @MulAssignWith.1(%i16)> [concrete]
// CHECK:STDOUT:   %MulAssignWith.assoc_type.3cc: type = assoc_entity_type @MulAssignWith.1, @MulAssignWith.1(%i16) [concrete]
// CHECK:STDOUT:   %assoc0.2ab: %MulAssignWith.assoc_type.3cc = assoc_entity element0, imports.%Core.import_ref.c116 [concrete]
// CHECK:STDOUT:   %Int16.Op.type.83af6e.14: type = fn_type @Int16.Op.14 [concrete]
// CHECK:STDOUT:   %Int16.Op.4dca31.14: %Int16.Op.type.83af6e.14 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.87d: <witness> = custom_witness (%Int16.Op.4dca31.14), @MulAssignWith.1, @MulAssignWith.1(%i16) [concrete]
// CHECK:STDOUT:   %MulAssignWith.facet.7b9: %MulAssignWith.type.8be = facet_value %Int16, (%custom_witness.87d) [concrete]
// CHECK:STDOUT:   %pattern_type.186: type = pattern_type %MulAssignWith.type.8be [concrete]
// CHECK:STDOUT:   %T.patt.a55: %pattern_type.186 = symbolic_binding_pattern T, 1 [symbolic]
// CHECK:STDOUT:   %DivAssignWith.type.572: type = facet_type <@DivAssignWith.1, @DivAssignWith.1(%i16)> [concrete]
// CHECK:STDOUT:   %DivAssignWith.assoc_type.753: type = assoc_entity_type @DivAssignWith.1, @DivAssignWith.1(%i16) [concrete]
// CHECK:STDOUT:   %assoc0.e4a0: %DivAssignWith.assoc_type.753 = assoc_entity element0, imports.%Core.import_ref.ee5 [concrete]
// CHECK:STDOUT:   %Int16.Op.type.83af6e.15: type = fn_type @Int16.Op.15 [concrete]
// CHECK:STDOUT:   %Int16.Op.4dca31.15: %Int16.Op.type.83af6e.15 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.8a1: <witness> = custom_witness (%Int16.Op.4dca31.15), @DivAssignWith.1, @DivAssignWith.1(%i16) [concrete]
// CHECK:STDOUT:   %DivAssignWith.facet.1ff: %DivAssignWith.type.572 = facet_value %Int16, (%custom_witness.8a1) [concrete]
// CHECK:STDOUT:   %pattern_type.54f: type = pattern_type %DivAssignWith.type.572 [concrete]
// CHECK:STDOUT:   %T.patt.998: %pattern_type.54f = symbolic_binding_pattern T, 1 [symbolic]
// CHECK:STDOUT:   %ModAssignWith.type.444: type = facet_type <@ModAssignWith.1, @ModAssignWith.1(%i16)> [concrete]
// CHECK:STDOUT:   %ModAssignWith.assoc_type.a2c: type = assoc_entity_type @ModAssignWith.1, @ModAssignWith.1(%i16) [concrete]
// CHECK:STDOUT:   %assoc0.0b1: %ModAssignWith.assoc_type.a2c = assoc_entity element0, imports.%Core.import_ref.97f5 [concrete]
// CHECK:STDOUT:   %Int16.Op.type.83af6e.16: type = fn_type @Int16.Op.16 [concrete]
// CHECK:STDOUT:   %Int16.Op.4dca31.16: %Int16.Op.type.83af6e.16 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b91: <witness> = custom_witness (%Int16.Op.4dca31.16), @ModAssignWith.1, @ModAssignWith.1(%i16) [concrete]
// CHECK:STDOUT:   %ModAssignWith.facet.9bc: %ModAssignWith.type.444 = facet_value %Int16, (%custom_witness.b91) [concrete]
// CHECK:STDOUT:   %pattern_type.35b: type = pattern_type %ModAssignWith.type.444 [concrete]
// CHECK:STDOUT:   %T.patt.191: %pattern_type.35b = symbolic_binding_pattern T, 1 [symbolic]
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
// CHECK:STDOUT:   %assoc0.f20: %ModAssignWith.assoc_type.7d0 = assoc_entity element0, imports.%Core.import_ref.97f5 [concrete]
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
// CHECK:STDOUT:   %.5a2: type = fn_type_with_self_type %Inc.WithSelf.Op.type.877857.2, %T.8f3 [symbolic]
// CHECK:STDOUT:   %impl.elem0.ec8: %.5a2 = impl_witness_access %Inc.lookup_impl_witness, element0 [symbolic]
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
// CHECK:STDOUT:   %facet_type.68f: type = facet_type <@Destroy & @Negate> [concrete]
// CHECK:STDOUT:   %.Self.718: %facet_type.68f = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %Negate.lookup_impl_witness.cdc: <witness> = lookup_impl_witness %.Self.718, @Negate [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.e29: type = impl_witness_access %Negate.lookup_impl_witness.cdc, element0 [symbolic_self]
// CHECK:STDOUT:   %facet_type.352: type = facet_type <@Destroy & @Negate where %impl.elem0.e29 = %Result.as_type> [symbolic]
// CHECK:STDOUT:   %pattern_type.4c2: type = pattern_type %facet_type.352 [symbolic]
// CHECK:STDOUT:   %T.patt.c5b: %pattern_type.4c2 = symbolic_binding_pattern T, 1 [symbolic]
// CHECK:STDOUT:   %T.e01: %facet_type.352 = symbolic_binding T, 1 [symbolic]
// CHECK:STDOUT:   %y.param_patt.023: %pattern_type.f4e485.2 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %y.patt.33f: %pattern_type.f4e485.2 = wrapper_binding_pattern y, %y.param_patt.023 [symbolic]
// CHECK:STDOUT:   %T.as_type.684: type = facet_access_type %T.e01 [symbolic]
// CHECK:STDOUT:   %pattern_type.2dd: type = pattern_type %T.as_type.684 [symbolic]
// CHECK:STDOUT:   %x.param_patt.cbc: %pattern_type.2dd = value_param_pattern [symbolic]
// CHECK:STDOUT:   %x.patt.ef5: %pattern_type.2dd = wrapper_binding_pattern x, %x.param_patt.cbc [symbolic]
// CHECK:STDOUT:   %Negate.lookup_impl_witness.218: <witness> = lookup_impl_witness %T.e01, @Negate [symbolic]
// CHECK:STDOUT:   %Negate.facet.5e5: %Negate.type = facet_value %T.as_type.684, (%Negate.lookup_impl_witness.218) [symbolic]
// CHECK:STDOUT:   %Negate.WithSelf.Op.type.d06: type = fn_type @Negate.WithSelf.Op, @Negate.WithSelf(%Negate.facet.5e5) [symbolic]
// CHECK:STDOUT:   %.5a7: type = fn_type_with_self_type %Negate.WithSelf.Op.type.d06, %Negate.facet.5e5 [symbolic]
// CHECK:STDOUT:   %impl.elem1.e2e: %.5a7 = impl_witness_access %Negate.lookup_impl_witness.218, element1 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.0c8: <specific function> = specific_impl_function %impl.elem1.e2e, @Negate.WithSelf.Op(%Negate.facet.5e5) [symbolic]
// CHECK:STDOUT:   %Int32.Op.type.83af6e.17: type = fn_type @Int32.Op.17 [concrete]
// CHECK:STDOUT:   %Int32.Op.4dca31.17: %Int32.Op.type.83af6e.17 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.f3f8: <witness> = custom_witness (%Int32.Op.4dca31.17), @Inc [concrete]
// CHECK:STDOUT:   %Inc.facet.dd2: %Inc.type = facet_value %Int32, (%custom_witness.f3f8) [concrete]
// CHECK:STDOUT:   %Int32.Op.type.83af6e.18: type = fn_type @Int32.Op.18 [concrete]
// CHECK:STDOUT:   %Int32.Op.4dca31.18: %Int32.Op.type.83af6e.18 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b24: <witness> = custom_witness (%Int32.Op.4dca31.18), @Dec [concrete]
// CHECK:STDOUT:   %Dec.facet.e1b: %Dec.type = facet_value %Int32, (%custom_witness.b24) [concrete]
// CHECK:STDOUT:   %facet_type.d8c: type = facet_type <@Destroy & @Negate where %impl.elem0.e29 = %Int64> [concrete]
// CHECK:STDOUT:   %Int16.Op.type.83af6e.17: type = fn_type @Int16.Op.17 [concrete]
// CHECK:STDOUT:   %Int16.Op.4dca31.17: %Int16.Op.type.83af6e.17 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.100: <witness> = custom_witness (%Int64, %Int16.Op.4dca31.17), @Negate [concrete]
// CHECK:STDOUT:   %facet_value.a67: %facet_type.d8c = facet_value %Int16, (%custom_witness.1bfa2a.1, %custom_witness.100) [concrete]
// CHECK:STDOUT:   %pattern_type.54c: type = pattern_type %facet_type.d8c [concrete]
// CHECK:STDOUT:   %T.patt.16f3: %pattern_type.54c = symbolic_binding_pattern T, 1 [symbolic]
// CHECK:STDOUT:   %facet_type.021: type = facet_type <@Destroy & @Negate where %impl.elem0.e29 = %Int32> [concrete]
// CHECK:STDOUT:   %Int32.Op.type.83af6e.19: type = fn_type @Int32.Op.19 [concrete]
// CHECK:STDOUT:   %Int32.Op.4dca31.19: %Int32.Op.type.83af6e.19 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.a85: <witness> = custom_witness (%Int32, %Int32.Op.4dca31.19), @Negate [concrete]
// CHECK:STDOUT:   %facet_value.73e: %facet_type.021 = facet_value %Int32, (%custom_witness.1bfa2a.2, %custom_witness.a85) [concrete]
// CHECK:STDOUT:   %pattern_type.4cd: type = pattern_type %facet_type.021 [concrete]
// CHECK:STDOUT:   %T.patt.d74: %pattern_type.4cd = symbolic_binding_pattern T, 1 [symbolic]
// CHECK:STDOUT:   %complete_type.f37: <witness> = complete_type_witness %AddWith.type.e4b [concrete]
// CHECK:STDOUT:   %AddWith.facet.41e: %AddWith.type.e4b = facet_value %i16, (%custom_witness.cba) [concrete]
// CHECK:STDOUT:   %AddWith.WithSelf.Op.type.348: type = fn_type @AddWith.WithSelf.Op, @AddWith.WithSelf(%Int16, %AddWith.facet.41e) [concrete]
// CHECK:STDOUT:   %.30b: type = fn_type_with_self_type %AddWith.WithSelf.Op.type.348, %AddWith.facet.41e [concrete]
// CHECK:STDOUT:   %complete_type.8a2: <witness> = complete_type_witness %SubWith.type.68d [concrete]
// CHECK:STDOUT:   %SubWith.facet.46f: %SubWith.type.68d = facet_value %i16, (%custom_witness.354) [concrete]
// CHECK:STDOUT:   %SubWith.WithSelf.Op.type.d0c: type = fn_type @SubWith.WithSelf.Op, @SubWith.WithSelf(%Int16, %SubWith.facet.46f) [concrete]
// CHECK:STDOUT:   %.4e84: type = fn_type_with_self_type %SubWith.WithSelf.Op.type.d0c, %SubWith.facet.46f [concrete]
// CHECK:STDOUT:   %complete_type.8c6: <witness> = complete_type_witness %MulWith.type.f65 [concrete]
// CHECK:STDOUT:   %MulWith.facet.808: %MulWith.type.f65 = facet_value %i16, (%custom_witness.3c3) [concrete]
// CHECK:STDOUT:   %MulWith.WithSelf.Op.type.263: type = fn_type @MulWith.WithSelf.Op, @MulWith.WithSelf(%Int16, %MulWith.facet.808) [concrete]
// CHECK:STDOUT:   %.3b1: type = fn_type_with_self_type %MulWith.WithSelf.Op.type.263, %MulWith.facet.808 [concrete]
// CHECK:STDOUT:   %complete_type.d99: <witness> = complete_type_witness %DivWith.type.b8a [concrete]
// CHECK:STDOUT:   %DivWith.facet.f0e: %DivWith.type.b8a = facet_value %i16, (%custom_witness.8d0) [concrete]
// CHECK:STDOUT:   %DivWith.WithSelf.Op.type.833: type = fn_type @DivWith.WithSelf.Op, @DivWith.WithSelf(%Int16, %DivWith.facet.f0e) [concrete]
// CHECK:STDOUT:   %.6c7: type = fn_type_with_self_type %DivWith.WithSelf.Op.type.833, %DivWith.facet.f0e [concrete]
// CHECK:STDOUT:   %complete_type.7d8: <witness> = complete_type_witness %ModWith.type.c67 [concrete]
// CHECK:STDOUT:   %ModWith.facet.139: %ModWith.type.c67 = facet_value %i16, (%custom_witness.169) [concrete]
// CHECK:STDOUT:   %ModWith.WithSelf.Op.type.f47: type = fn_type @ModWith.WithSelf.Op, @ModWith.WithSelf(%Int16, %ModWith.facet.139) [concrete]
// CHECK:STDOUT:   %.1cf: type = fn_type_with_self_type %ModWith.WithSelf.Op.type.f47, %ModWith.facet.139 [concrete]
// CHECK:STDOUT:   %complete_type.4db: <witness> = complete_type_witness %AddWith.type.631 [concrete]
// CHECK:STDOUT:   %AddWith.facet.b20: %AddWith.type.631 = facet_value %Int16, (%custom_witness.917) [concrete]
// CHECK:STDOUT:   %AddWith.WithSelf.Op.type.aba: type = fn_type @AddWith.WithSelf.Op, @AddWith.WithSelf(%i16, %AddWith.facet.b20) [concrete]
// CHECK:STDOUT:   %.689: type = fn_type_with_self_type %AddWith.WithSelf.Op.type.aba, %AddWith.facet.b20 [concrete]
// CHECK:STDOUT:   %complete_type.c36: <witness> = complete_type_witness %SubWith.type.971 [concrete]
// CHECK:STDOUT:   %SubWith.facet.b6f: %SubWith.type.971 = facet_value %Int16, (%custom_witness.1d8) [concrete]
// CHECK:STDOUT:   %SubWith.WithSelf.Op.type.90f: type = fn_type @SubWith.WithSelf.Op, @SubWith.WithSelf(%i16, %SubWith.facet.b6f) [concrete]
// CHECK:STDOUT:   %.6a5: type = fn_type_with_self_type %SubWith.WithSelf.Op.type.90f, %SubWith.facet.b6f [concrete]
// CHECK:STDOUT:   %complete_type.864: <witness> = complete_type_witness %MulWith.type.f13 [concrete]
// CHECK:STDOUT:   %MulWith.facet.6f9: %MulWith.type.f13 = facet_value %Int16, (%custom_witness.36e) [concrete]
// CHECK:STDOUT:   %MulWith.WithSelf.Op.type.f0e: type = fn_type @MulWith.WithSelf.Op, @MulWith.WithSelf(%i16, %MulWith.facet.6f9) [concrete]
// CHECK:STDOUT:   %.b9b: type = fn_type_with_self_type %MulWith.WithSelf.Op.type.f0e, %MulWith.facet.6f9 [concrete]
// CHECK:STDOUT:   %complete_type.730: <witness> = complete_type_witness %DivWith.type.f46 [concrete]
// CHECK:STDOUT:   %DivWith.facet.b7c: %DivWith.type.f46 = facet_value %Int16, (%custom_witness.5f8) [concrete]
// CHECK:STDOUT:   %DivWith.WithSelf.Op.type.b11: type = fn_type @DivWith.WithSelf.Op, @DivWith.WithSelf(%i16, %DivWith.facet.b7c) [concrete]
// CHECK:STDOUT:   %.20e: type = fn_type_with_self_type %DivWith.WithSelf.Op.type.b11, %DivWith.facet.b7c [concrete]
// CHECK:STDOUT:   %complete_type.b59: <witness> = complete_type_witness %ModWith.type.065 [concrete]
// CHECK:STDOUT:   %ModWith.facet.45c: %ModWith.type.065 = facet_value %Int16, (%custom_witness.127) [concrete]
// CHECK:STDOUT:   %ModWith.WithSelf.Op.type.377: type = fn_type @ModWith.WithSelf.Op, @ModWith.WithSelf(%i16, %ModWith.facet.45c) [concrete]
// CHECK:STDOUT:   %.448: type = fn_type_with_self_type %ModWith.WithSelf.Op.type.377, %ModWith.facet.45c [concrete]
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
// CHECK:STDOUT:   %SubWith.WithSelf.Op.type.2cf: type = fn_type @SubWith.WithSelf.Op, @SubWith.WithSelf(%Int32, %SubWith.facet.245) [concrete]
// CHECK:STDOUT:   %.4e86: type = fn_type_with_self_type %SubWith.WithSelf.Op.type.2cf, %SubWith.facet.245 [concrete]
// CHECK:STDOUT:   %MulWith.facet.394: %MulWith.type.d67 = facet_value %Int64, (%custom_witness.798) [concrete]
// CHECK:STDOUT:   %MulWith.WithSelf.Op.type.0ab: type = fn_type @MulWith.WithSelf.Op, @MulWith.WithSelf(%Int32, %MulWith.facet.394) [concrete]
// CHECK:STDOUT:   %.22d: type = fn_type_with_self_type %MulWith.WithSelf.Op.type.0ab, %MulWith.facet.394 [concrete]
// CHECK:STDOUT:   %DivWith.facet.85e: %DivWith.type.5a1 = facet_value %Int64, (%custom_witness.37f) [concrete]
// CHECK:STDOUT:   %DivWith.WithSelf.Op.type.7b8: type = fn_type @DivWith.WithSelf.Op, @DivWith.WithSelf(%Int32, %DivWith.facet.85e) [concrete]
// CHECK:STDOUT:   %.cde: type = fn_type_with_self_type %DivWith.WithSelf.Op.type.7b8, %DivWith.facet.85e [concrete]
// CHECK:STDOUT:   %ModWith.facet.f5a: %ModWith.type.794 = facet_value %Int64, (%custom_witness.981) [concrete]
// CHECK:STDOUT:   %ModWith.WithSelf.Op.type.d27: type = fn_type @ModWith.WithSelf.Op, @ModWith.WithSelf(%Int32, %ModWith.facet.f5a) [concrete]
// CHECK:STDOUT:   %.3b0: type = fn_type_with_self_type %ModWith.WithSelf.Op.type.d27, %ModWith.facet.f5a [concrete]
// CHECK:STDOUT:   %complete_type.1ab: <witness> = complete_type_witness %AddAssignWith.type.b50 [concrete]
// CHECK:STDOUT:   %AddAssignWith.WithSelf.Op.type.ccb: type = fn_type @AddAssignWith.WithSelf.Op, @AddAssignWith.WithSelf(%i16, %AddAssignWith.facet.510) [concrete]
// CHECK:STDOUT:   %.0ed: type = fn_type_with_self_type %AddAssignWith.WithSelf.Op.type.ccb, %AddAssignWith.facet.510 [concrete]
// CHECK:STDOUT:   %complete_type.767: <witness> = complete_type_witness %SubAssignWith.type.5c1 [concrete]
// CHECK:STDOUT:   %SubAssignWith.WithSelf.Op.type.311: type = fn_type @SubAssignWith.WithSelf.Op, @SubAssignWith.WithSelf(%i16, %SubAssignWith.facet.6c0) [concrete]
// CHECK:STDOUT:   %.472: type = fn_type_with_self_type %SubAssignWith.WithSelf.Op.type.311, %SubAssignWith.facet.6c0 [concrete]
// CHECK:STDOUT:   %complete_type.512: <witness> = complete_type_witness %MulAssignWith.type.8be [concrete]
// CHECK:STDOUT:   %MulAssignWith.WithSelf.Op.type.697: type = fn_type @MulAssignWith.WithSelf.Op, @MulAssignWith.WithSelf(%i16, %MulAssignWith.facet.7b9) [concrete]
// CHECK:STDOUT:   %.1c8: type = fn_type_with_self_type %MulAssignWith.WithSelf.Op.type.697, %MulAssignWith.facet.7b9 [concrete]
// CHECK:STDOUT:   %complete_type.23e: <witness> = complete_type_witness %DivAssignWith.type.572 [concrete]
// CHECK:STDOUT:   %DivAssignWith.WithSelf.Op.type.d0e: type = fn_type @DivAssignWith.WithSelf.Op, @DivAssignWith.WithSelf(%i16, %DivAssignWith.facet.1ff) [concrete]
// CHECK:STDOUT:   %.f42: type = fn_type_with_self_type %DivAssignWith.WithSelf.Op.type.d0e, %DivAssignWith.facet.1ff [concrete]
// CHECK:STDOUT:   %complete_type.4dc: <witness> = complete_type_witness %ModAssignWith.type.444 [concrete]
// CHECK:STDOUT:   %ModAssignWith.WithSelf.Op.type.ab6: type = fn_type @ModAssignWith.WithSelf.Op, @ModAssignWith.WithSelf(%i16, %ModAssignWith.facet.9bc) [concrete]
// CHECK:STDOUT:   %.182: type = fn_type_with_self_type %ModAssignWith.WithSelf.Op.type.ab6, %ModAssignWith.facet.9bc [concrete]
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
// CHECK:STDOUT:   %Core.import_ref.ea5e: @ModWith.WithSelf.%ModWith.assoc_type (%ModWith.assoc_type.1f1) = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [symbolic = @ModWith.WithSelf.%assoc1 (constants.%assoc1.f4b)]
// CHECK:STDOUT:   %Core.import_ref.9f8f: @ModWith.WithSelf.%ModWith.WithSelf.Op.type (%ModWith.WithSelf.Op.type.04e) = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [symbolic = @ModWith.WithSelf.%ModWith.WithSelf.Op (constants.%ModWith.WithSelf.Op.6fb)]
// CHECK:STDOUT:   %Core.import_ref.61a: type = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [concrete = %Result.803]
// CHECK:STDOUT:   %Result.803: type = assoc_const_decl @Result.5 [concrete] {}
// CHECK:STDOUT:   %Core.import_ref.be0 = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, unloaded
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
// CHECK:STDOUT:   %Core.import_ref.97f5: @ModAssignWith.WithSelf.%ModAssignWith.WithSelf.Op.type (%ModAssignWith.WithSelf.Op.type.7f247d.1) = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [symbolic = @ModAssignWith.WithSelf.%ModAssignWith.WithSelf.Op (constants.%ModAssignWith.WithSelf.Op.9b8575.1)]
// CHECK:STDOUT:   %Core.import_ref.ffb = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @AddWith.loc124(%Result.loc121_9.2: %Destroy.type, %U.loc122_4.2: %Destroy.type, %T.loc123_4.2: @AddWith.loc124.%facet_type.loc123_37 (%facet_type.660)) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %require_complete.loc126: <witness> = require_complete_type %AddWith.type.loc123_35.1 [symbolic = %require_complete.loc126 (constants.%require_complete.5dc)]
// CHECK:STDOUT:   %assoc1: @AddWith.loc124.%AddWith.assoc_type (%AddWith.assoc_type.964) = assoc_entity element1, imports.%Core.import_ref.8ad [symbolic = %assoc1 (constants.%assoc1.261)]
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.loc126: <witness> = lookup_impl_witness %T.loc123_4.1, @AddWith.1, @AddWith.1(%U.as_type.loc123_35.1) [symbolic = %AddWith.lookup_impl_witness.loc126 (constants.%AddWith.lookup_impl_witness.aab)]
// CHECK:STDOUT:   %AddWith.facet: @AddWith.loc124.%AddWith.type.loc123_35.1 (%AddWith.type.d58) = facet_value %T.as_type.loc124_29.1, (%AddWith.lookup_impl_witness.loc126) [symbolic = %AddWith.facet (constants.%AddWith.facet.dad)]
// CHECK:STDOUT:   %AddWith.WithSelf.Op.type: type = fn_type @AddWith.WithSelf.Op, @AddWith.WithSelf(%U.as_type.loc123_35.1, %AddWith.facet) [symbolic = %AddWith.WithSelf.Op.type (constants.%AddWith.WithSelf.Op.type.081)]
// CHECK:STDOUT:   %.loc126_5.6: type = fn_type_with_self_type %AddWith.WithSelf.Op.type, %AddWith.facet [symbolic = %.loc126_5.6 (constants.%.612)]
// CHECK:STDOUT:   %impl.elem1.loc126_5.2: @AddWith.loc124.%.loc126_5.6 (%.612) = impl_witness_access %AddWith.lookup_impl_witness.loc126, element1 [symbolic = %impl.elem1.loc126_5.2 (constants.%impl.elem1.9e8)]
// CHECK:STDOUT:   %specific_impl_fn.loc126_5.3: <specific function> = specific_impl_function %impl.elem1.loc126_5.2, @AddWith.WithSelf.Op(%U.as_type.loc123_35.1, %AddWith.facet) [symbolic = %specific_impl_fn.loc126_5.3 (constants.%specific_impl_fn.666)]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type: type = fn_type @Destroy.WithSelf.SelfDestruct.1, @Destroy.WithSelf(%Result.loc121_9.1) [symbolic = %Destroy.WithSelf.SelfDestruct.type (constants.%Destroy.WithSelf.SelfDestruct.type.869b78.2)]
// CHECK:STDOUT:   %.loc126_5.7: type = fn_type_with_self_type %Destroy.WithSelf.SelfDestruct.type, %Result.loc121_9.1 [symbolic = %.loc126_5.7 (constants.%.f0c)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %Result.loc121_9.1, @Destroy [symbolic = %Destroy.lookup_impl_witness (constants.%Destroy.lookup_impl_witness.dfb4c4.2)]
// CHECK:STDOUT:   %impl.elem2.loc126_5.2: @AddWith.loc124.%.loc126_5.7 (%.f0c) = impl_witness_access %Destroy.lookup_impl_witness, element2 [symbolic = %impl.elem2.loc126_5.2 (constants.%impl.elem2)]
// CHECK:STDOUT:   %specific_impl_fn.loc126_5.4: <specific function> = specific_impl_function %impl.elem2.loc126_5.2, @Destroy.WithSelf.SelfDestruct.1(%Result.loc121_9.1) [symbolic = %specific_impl_fn.loc126_5.4 (constants.%specific_impl_fn.6f9)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%result.param: @AddWith.loc124.%Result.as_type.loc123_53.1 (%Result.as_type), %x.param: @AddWith.loc124.%T.as_type.loc124_29.1 (%T.as_type.fe6), %y.param: @AddWith.loc124.%U.as_type.loc123_35.1 (%U.as_type.b61)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: @AddWith.loc124.%T.as_type.loc124_29.1 (%T.as_type.fe6) = name_ref x, %x
// CHECK:STDOUT:     %y.ref: @AddWith.loc124.%U.as_type.loc123_35.1 (%U.as_type.b61) = name_ref y, %y
// CHECK:STDOUT:     %AddWith.type.loc126: type = facet_type <@AddWith.1, @AddWith.1(constants.%U.as_type.b61)> [symbolic = %AddWith.type.loc123_35.1 (constants.%AddWith.type.d58)]
// CHECK:STDOUT:     %.loc126_5.1: @AddWith.loc124.%AddWith.assoc_type (%AddWith.assoc_type.964) = specific_constant imports.%Core.import_ref.5c64, @AddWith.WithSelf(constants.%U.as_type.b61, constants.%Self.958) [symbolic = %assoc1 (constants.%assoc1.261)]
// CHECK:STDOUT:     %Op.ref: @AddWith.loc124.%AddWith.assoc_type (%AddWith.assoc_type.964) = name_ref Op, %.loc126_5.1 [symbolic = %assoc1 (constants.%assoc1.261)]
// CHECK:STDOUT:     %impl.elem1.loc126_5.1: @AddWith.loc124.%.loc126_5.6 (%.612) = impl_witness_access constants.%AddWith.lookup_impl_witness.aab, element1 [symbolic = %impl.elem1.loc126_5.2 (constants.%impl.elem1.9e8)]
// CHECK:STDOUT:     %bound_method.loc126_5.1: <bound method> = bound_method %x.ref, %impl.elem1.loc126_5.1
// CHECK:STDOUT:     %specific_impl_fn.loc126_5.1: <specific function> = specific_impl_function %impl.elem1.loc126_5.1, @AddWith.WithSelf.Op(constants.%U.as_type.b61, constants.%AddWith.facet.dad) [symbolic = %specific_impl_fn.loc126_5.3 (constants.%specific_impl_fn.666)]
// CHECK:STDOUT:     %bound_method.loc126_5.2: <bound method> = bound_method %x.ref, %specific_impl_fn.loc126_5.1
// CHECK:STDOUT:     %.loc126_5.2: ref @AddWith.loc124.%Result.as_type.loc123_53.1 (%Result.as_type) = temporary_storage
// CHECK:STDOUT:     %AddWith.WithSelf.Op.call: init @AddWith.loc124.%Result.as_type.loc123_53.1 (%Result.as_type) to %.loc126_5.2 = call %bound_method.loc126_5.2(%x.ref, %y.ref)
// CHECK:STDOUT:     %.loc126_5.3: ref @AddWith.loc124.%Result.as_type.loc123_53.1 (%Result.as_type) = temporary %.loc126_5.2, %AddWith.WithSelf.Op.call
// CHECK:STDOUT:     %impl.elem2.loc126_5.1: @AddWith.loc124.%.loc126_5.7 (%.f0c) = impl_witness_access constants.%Destroy.lookup_impl_witness.dfb4c4.2, element2 [symbolic = %impl.elem2.loc126_5.2 (constants.%impl.elem2)]
// CHECK:STDOUT:     %bound_method.loc126_5.3: <bound method> = bound_method %.loc126_5.3, %impl.elem2.loc126_5.1
// CHECK:STDOUT:     %.loc126_5.4: %Destroy.type = converted constants.%Result.as_type, constants.%Result [symbolic = %Result.loc121_9.1 (constants.%Result)]
// CHECK:STDOUT:     %.loc126_5.5: %Destroy.type = converted constants.%Result.as_type, constants.%Result [symbolic = %Result.loc121_9.1 (constants.%Result)]
// CHECK:STDOUT:     %specific_impl_fn.loc126_5.2: <specific function> = specific_impl_function %impl.elem2.loc126_5.1, @Destroy.WithSelf.SelfDestruct.1(constants.%Result) [symbolic = %specific_impl_fn.loc126_5.4 (constants.%specific_impl_fn.6f9)]
// CHECK:STDOUT:     %bound_method.loc126_5.4: <bound method> = bound_method %.loc126_5.3, %specific_impl_fn.loc126_5.2
// CHECK:STDOUT:     %Destroy.WithSelf.SelfDestruct.call: init %empty_tuple.type = call %bound_method.loc126_5.4(%.loc126_5.3)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @SubWith.loc135(%Result.loc132_9.2: %Destroy.type, %U.loc133_4.2: %Destroy.type, %T.loc134_4.2: @SubWith.loc135.%facet_type.loc134_37 (%facet_type.ba3)) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %require_complete.loc137: <witness> = require_complete_type %SubWith.type.loc134_35.1 [symbolic = %require_complete.loc137 (constants.%require_complete.c6e)]
// CHECK:STDOUT:   %assoc1: @SubWith.loc135.%SubWith.assoc_type (%SubWith.assoc_type.378) = assoc_entity element1, imports.%Core.import_ref.2f4 [symbolic = %assoc1 (constants.%assoc1.0a7)]
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.loc137: <witness> = lookup_impl_witness %T.loc134_4.1, @SubWith.1, @SubWith.1(%U.as_type.loc134_35.1) [symbolic = %SubWith.lookup_impl_witness.loc137 (constants.%SubWith.lookup_impl_witness.1fa)]
// CHECK:STDOUT:   %SubWith.facet: @SubWith.loc135.%SubWith.type.loc134_35.1 (%SubWith.type.d62) = facet_value %T.as_type.loc135_29.1, (%SubWith.lookup_impl_witness.loc137) [symbolic = %SubWith.facet (constants.%SubWith.facet.eca)]
// CHECK:STDOUT:   %SubWith.WithSelf.Op.type: type = fn_type @SubWith.WithSelf.Op, @SubWith.WithSelf(%U.as_type.loc134_35.1, %SubWith.facet) [symbolic = %SubWith.WithSelf.Op.type (constants.%SubWith.WithSelf.Op.type.90e)]
// CHECK:STDOUT:   %.loc137_5.6: type = fn_type_with_self_type %SubWith.WithSelf.Op.type, %SubWith.facet [symbolic = %.loc137_5.6 (constants.%.844)]
// CHECK:STDOUT:   %impl.elem1.loc137_5.2: @SubWith.loc135.%.loc137_5.6 (%.844) = impl_witness_access %SubWith.lookup_impl_witness.loc137, element1 [symbolic = %impl.elem1.loc137_5.2 (constants.%impl.elem1.fa7)]
// CHECK:STDOUT:   %specific_impl_fn.loc137_5.3: <specific function> = specific_impl_function %impl.elem1.loc137_5.2, @SubWith.WithSelf.Op(%U.as_type.loc134_35.1, %SubWith.facet) [symbolic = %specific_impl_fn.loc137_5.3 (constants.%specific_impl_fn.2de)]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type: type = fn_type @Destroy.WithSelf.SelfDestruct.1, @Destroy.WithSelf(%Result.loc132_9.1) [symbolic = %Destroy.WithSelf.SelfDestruct.type (constants.%Destroy.WithSelf.SelfDestruct.type.869b78.2)]
// CHECK:STDOUT:   %.loc137_5.7: type = fn_type_with_self_type %Destroy.WithSelf.SelfDestruct.type, %Result.loc132_9.1 [symbolic = %.loc137_5.7 (constants.%.f0c)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %Result.loc132_9.1, @Destroy [symbolic = %Destroy.lookup_impl_witness (constants.%Destroy.lookup_impl_witness.dfb4c4.2)]
// CHECK:STDOUT:   %impl.elem2.loc137_5.2: @SubWith.loc135.%.loc137_5.7 (%.f0c) = impl_witness_access %Destroy.lookup_impl_witness, element2 [symbolic = %impl.elem2.loc137_5.2 (constants.%impl.elem2)]
// CHECK:STDOUT:   %specific_impl_fn.loc137_5.4: <specific function> = specific_impl_function %impl.elem2.loc137_5.2, @Destroy.WithSelf.SelfDestruct.1(%Result.loc132_9.1) [symbolic = %specific_impl_fn.loc137_5.4 (constants.%specific_impl_fn.6f9)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%result.param: @SubWith.loc135.%Result.as_type.loc134_53.1 (%Result.as_type), %x.param: @SubWith.loc135.%T.as_type.loc135_29.1 (%T.as_type.dfc), %y.param: @SubWith.loc135.%U.as_type.loc134_35.1 (%U.as_type.b61)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: @SubWith.loc135.%T.as_type.loc135_29.1 (%T.as_type.dfc) = name_ref x, %x
// CHECK:STDOUT:     %y.ref: @SubWith.loc135.%U.as_type.loc134_35.1 (%U.as_type.b61) = name_ref y, %y
// CHECK:STDOUT:     %SubWith.type.loc137: type = facet_type <@SubWith.1, @SubWith.1(constants.%U.as_type.b61)> [symbolic = %SubWith.type.loc134_35.1 (constants.%SubWith.type.d62)]
// CHECK:STDOUT:     %.loc137_5.1: @SubWith.loc135.%SubWith.assoc_type (%SubWith.assoc_type.378) = specific_constant imports.%Core.import_ref.5fe, @SubWith.WithSelf(constants.%U.as_type.b61, constants.%Self.18a) [symbolic = %assoc1 (constants.%assoc1.0a7)]
// CHECK:STDOUT:     %Op.ref: @SubWith.loc135.%SubWith.assoc_type (%SubWith.assoc_type.378) = name_ref Op, %.loc137_5.1 [symbolic = %assoc1 (constants.%assoc1.0a7)]
// CHECK:STDOUT:     %impl.elem1.loc137_5.1: @SubWith.loc135.%.loc137_5.6 (%.844) = impl_witness_access constants.%SubWith.lookup_impl_witness.1fa, element1 [symbolic = %impl.elem1.loc137_5.2 (constants.%impl.elem1.fa7)]
// CHECK:STDOUT:     %bound_method.loc137_5.1: <bound method> = bound_method %x.ref, %impl.elem1.loc137_5.1
// CHECK:STDOUT:     %specific_impl_fn.loc137_5.1: <specific function> = specific_impl_function %impl.elem1.loc137_5.1, @SubWith.WithSelf.Op(constants.%U.as_type.b61, constants.%SubWith.facet.eca) [symbolic = %specific_impl_fn.loc137_5.3 (constants.%specific_impl_fn.2de)]
// CHECK:STDOUT:     %bound_method.loc137_5.2: <bound method> = bound_method %x.ref, %specific_impl_fn.loc137_5.1
// CHECK:STDOUT:     %.loc137_5.2: ref @SubWith.loc135.%Result.as_type.loc134_53.1 (%Result.as_type) = temporary_storage
// CHECK:STDOUT:     %SubWith.WithSelf.Op.call: init @SubWith.loc135.%Result.as_type.loc134_53.1 (%Result.as_type) to %.loc137_5.2 = call %bound_method.loc137_5.2(%x.ref, %y.ref)
// CHECK:STDOUT:     %.loc137_5.3: ref @SubWith.loc135.%Result.as_type.loc134_53.1 (%Result.as_type) = temporary %.loc137_5.2, %SubWith.WithSelf.Op.call
// CHECK:STDOUT:     %impl.elem2.loc137_5.1: @SubWith.loc135.%.loc137_5.7 (%.f0c) = impl_witness_access constants.%Destroy.lookup_impl_witness.dfb4c4.2, element2 [symbolic = %impl.elem2.loc137_5.2 (constants.%impl.elem2)]
// CHECK:STDOUT:     %bound_method.loc137_5.3: <bound method> = bound_method %.loc137_5.3, %impl.elem2.loc137_5.1
// CHECK:STDOUT:     %.loc137_5.4: %Destroy.type = converted constants.%Result.as_type, constants.%Result [symbolic = %Result.loc132_9.1 (constants.%Result)]
// CHECK:STDOUT:     %.loc137_5.5: %Destroy.type = converted constants.%Result.as_type, constants.%Result [symbolic = %Result.loc132_9.1 (constants.%Result)]
// CHECK:STDOUT:     %specific_impl_fn.loc137_5.2: <specific function> = specific_impl_function %impl.elem2.loc137_5.1, @Destroy.WithSelf.SelfDestruct.1(constants.%Result) [symbolic = %specific_impl_fn.loc137_5.4 (constants.%specific_impl_fn.6f9)]
// CHECK:STDOUT:     %bound_method.loc137_5.4: <bound method> = bound_method %.loc137_5.3, %specific_impl_fn.loc137_5.2
// CHECK:STDOUT:     %Destroy.WithSelf.SelfDestruct.call: init %empty_tuple.type = call %bound_method.loc137_5.4(%.loc137_5.3)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @MulWith.loc146(%Result.loc143_9.2: %Destroy.type, %U.loc144_4.2: %Destroy.type, %T.loc145_4.2: @MulWith.loc146.%facet_type.loc145_37 (%facet_type.9c7)) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %require_complete.loc148: <witness> = require_complete_type %MulWith.type.loc145_35.1 [symbolic = %require_complete.loc148 (constants.%require_complete.dc1)]
// CHECK:STDOUT:   %assoc1: @MulWith.loc146.%MulWith.assoc_type (%MulWith.assoc_type.1d7) = assoc_entity element1, imports.%Core.import_ref.fc2 [symbolic = %assoc1 (constants.%assoc1.2be)]
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.loc148: <witness> = lookup_impl_witness %T.loc145_4.1, @MulWith.1, @MulWith.1(%U.as_type.loc145_35.1) [symbolic = %MulWith.lookup_impl_witness.loc148 (constants.%MulWith.lookup_impl_witness.ac6)]
// CHECK:STDOUT:   %MulWith.facet: @MulWith.loc146.%MulWith.type.loc145_35.1 (%MulWith.type.5c9) = facet_value %T.as_type.loc146_29.1, (%MulWith.lookup_impl_witness.loc148) [symbolic = %MulWith.facet (constants.%MulWith.facet.cae)]
// CHECK:STDOUT:   %MulWith.WithSelf.Op.type: type = fn_type @MulWith.WithSelf.Op, @MulWith.WithSelf(%U.as_type.loc145_35.1, %MulWith.facet) [symbolic = %MulWith.WithSelf.Op.type (constants.%MulWith.WithSelf.Op.type.6df)]
// CHECK:STDOUT:   %.loc148_5.6: type = fn_type_with_self_type %MulWith.WithSelf.Op.type, %MulWith.facet [symbolic = %.loc148_5.6 (constants.%.d85)]
// CHECK:STDOUT:   %impl.elem1.loc148_5.2: @MulWith.loc146.%.loc148_5.6 (%.d85) = impl_witness_access %MulWith.lookup_impl_witness.loc148, element1 [symbolic = %impl.elem1.loc148_5.2 (constants.%impl.elem1.5cc)]
// CHECK:STDOUT:   %specific_impl_fn.loc148_5.3: <specific function> = specific_impl_function %impl.elem1.loc148_5.2, @MulWith.WithSelf.Op(%U.as_type.loc145_35.1, %MulWith.facet) [symbolic = %specific_impl_fn.loc148_5.3 (constants.%specific_impl_fn.2cd)]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type: type = fn_type @Destroy.WithSelf.SelfDestruct.1, @Destroy.WithSelf(%Result.loc143_9.1) [symbolic = %Destroy.WithSelf.SelfDestruct.type (constants.%Destroy.WithSelf.SelfDestruct.type.869b78.2)]
// CHECK:STDOUT:   %.loc148_5.7: type = fn_type_with_self_type %Destroy.WithSelf.SelfDestruct.type, %Result.loc143_9.1 [symbolic = %.loc148_5.7 (constants.%.f0c)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %Result.loc143_9.1, @Destroy [symbolic = %Destroy.lookup_impl_witness (constants.%Destroy.lookup_impl_witness.dfb4c4.2)]
// CHECK:STDOUT:   %impl.elem2.loc148_5.2: @MulWith.loc146.%.loc148_5.7 (%.f0c) = impl_witness_access %Destroy.lookup_impl_witness, element2 [symbolic = %impl.elem2.loc148_5.2 (constants.%impl.elem2)]
// CHECK:STDOUT:   %specific_impl_fn.loc148_5.4: <specific function> = specific_impl_function %impl.elem2.loc148_5.2, @Destroy.WithSelf.SelfDestruct.1(%Result.loc143_9.1) [symbolic = %specific_impl_fn.loc148_5.4 (constants.%specific_impl_fn.6f9)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%result.param: @MulWith.loc146.%Result.as_type.loc145_53.1 (%Result.as_type), %x.param: @MulWith.loc146.%T.as_type.loc146_29.1 (%T.as_type.521), %y.param: @MulWith.loc146.%U.as_type.loc145_35.1 (%U.as_type.b61)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: @MulWith.loc146.%T.as_type.loc146_29.1 (%T.as_type.521) = name_ref x, %x
// CHECK:STDOUT:     %y.ref: @MulWith.loc146.%U.as_type.loc145_35.1 (%U.as_type.b61) = name_ref y, %y
// CHECK:STDOUT:     %MulWith.type.loc148: type = facet_type <@MulWith.1, @MulWith.1(constants.%U.as_type.b61)> [symbolic = %MulWith.type.loc145_35.1 (constants.%MulWith.type.5c9)]
// CHECK:STDOUT:     %.loc148_5.1: @MulWith.loc146.%MulWith.assoc_type (%MulWith.assoc_type.1d7) = specific_constant imports.%Core.import_ref.0a8, @MulWith.WithSelf(constants.%U.as_type.b61, constants.%Self.742) [symbolic = %assoc1 (constants.%assoc1.2be)]
// CHECK:STDOUT:     %Op.ref: @MulWith.loc146.%MulWith.assoc_type (%MulWith.assoc_type.1d7) = name_ref Op, %.loc148_5.1 [symbolic = %assoc1 (constants.%assoc1.2be)]
// CHECK:STDOUT:     %impl.elem1.loc148_5.1: @MulWith.loc146.%.loc148_5.6 (%.d85) = impl_witness_access constants.%MulWith.lookup_impl_witness.ac6, element1 [symbolic = %impl.elem1.loc148_5.2 (constants.%impl.elem1.5cc)]
// CHECK:STDOUT:     %bound_method.loc148_5.1: <bound method> = bound_method %x.ref, %impl.elem1.loc148_5.1
// CHECK:STDOUT:     %specific_impl_fn.loc148_5.1: <specific function> = specific_impl_function %impl.elem1.loc148_5.1, @MulWith.WithSelf.Op(constants.%U.as_type.b61, constants.%MulWith.facet.cae) [symbolic = %specific_impl_fn.loc148_5.3 (constants.%specific_impl_fn.2cd)]
// CHECK:STDOUT:     %bound_method.loc148_5.2: <bound method> = bound_method %x.ref, %specific_impl_fn.loc148_5.1
// CHECK:STDOUT:     %.loc148_5.2: ref @MulWith.loc146.%Result.as_type.loc145_53.1 (%Result.as_type) = temporary_storage
// CHECK:STDOUT:     %MulWith.WithSelf.Op.call: init @MulWith.loc146.%Result.as_type.loc145_53.1 (%Result.as_type) to %.loc148_5.2 = call %bound_method.loc148_5.2(%x.ref, %y.ref)
// CHECK:STDOUT:     %.loc148_5.3: ref @MulWith.loc146.%Result.as_type.loc145_53.1 (%Result.as_type) = temporary %.loc148_5.2, %MulWith.WithSelf.Op.call
// CHECK:STDOUT:     %impl.elem2.loc148_5.1: @MulWith.loc146.%.loc148_5.7 (%.f0c) = impl_witness_access constants.%Destroy.lookup_impl_witness.dfb4c4.2, element2 [symbolic = %impl.elem2.loc148_5.2 (constants.%impl.elem2)]
// CHECK:STDOUT:     %bound_method.loc148_5.3: <bound method> = bound_method %.loc148_5.3, %impl.elem2.loc148_5.1
// CHECK:STDOUT:     %.loc148_5.4: %Destroy.type = converted constants.%Result.as_type, constants.%Result [symbolic = %Result.loc143_9.1 (constants.%Result)]
// CHECK:STDOUT:     %.loc148_5.5: %Destroy.type = converted constants.%Result.as_type, constants.%Result [symbolic = %Result.loc143_9.1 (constants.%Result)]
// CHECK:STDOUT:     %specific_impl_fn.loc148_5.2: <specific function> = specific_impl_function %impl.elem2.loc148_5.1, @Destroy.WithSelf.SelfDestruct.1(constants.%Result) [symbolic = %specific_impl_fn.loc148_5.4 (constants.%specific_impl_fn.6f9)]
// CHECK:STDOUT:     %bound_method.loc148_5.4: <bound method> = bound_method %.loc148_5.3, %specific_impl_fn.loc148_5.2
// CHECK:STDOUT:     %Destroy.WithSelf.SelfDestruct.call: init %empty_tuple.type = call %bound_method.loc148_5.4(%.loc148_5.3)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @DivWith.loc157(%Result.loc154_9.2: %Destroy.type, %U.loc155_4.2: %Destroy.type, %T.loc156_4.2: @DivWith.loc157.%facet_type.loc156_37 (%facet_type.89a)) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %require_complete.loc159: <witness> = require_complete_type %DivWith.type.loc156_35.1 [symbolic = %require_complete.loc159 (constants.%require_complete.4b7)]
// CHECK:STDOUT:   %assoc1: @DivWith.loc157.%DivWith.assoc_type (%DivWith.assoc_type.248) = assoc_entity element1, imports.%Core.import_ref.f2f [symbolic = %assoc1 (constants.%assoc1.6e8)]
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.loc159: <witness> = lookup_impl_witness %T.loc156_4.1, @DivWith.1, @DivWith.1(%U.as_type.loc156_35.1) [symbolic = %DivWith.lookup_impl_witness.loc159 (constants.%DivWith.lookup_impl_witness.c25)]
// CHECK:STDOUT:   %DivWith.facet: @DivWith.loc157.%DivWith.type.loc156_35.1 (%DivWith.type.e82) = facet_value %T.as_type.loc157_29.1, (%DivWith.lookup_impl_witness.loc159) [symbolic = %DivWith.facet (constants.%DivWith.facet.7d7)]
// CHECK:STDOUT:   %DivWith.WithSelf.Op.type: type = fn_type @DivWith.WithSelf.Op, @DivWith.WithSelf(%U.as_type.loc156_35.1, %DivWith.facet) [symbolic = %DivWith.WithSelf.Op.type (constants.%DivWith.WithSelf.Op.type.0d7)]
// CHECK:STDOUT:   %.loc159_5.6: type = fn_type_with_self_type %DivWith.WithSelf.Op.type, %DivWith.facet [symbolic = %.loc159_5.6 (constants.%.ba0)]
// CHECK:STDOUT:   %impl.elem1.loc159_5.2: @DivWith.loc157.%.loc159_5.6 (%.ba0) = impl_witness_access %DivWith.lookup_impl_witness.loc159, element1 [symbolic = %impl.elem1.loc159_5.2 (constants.%impl.elem1.610)]
// CHECK:STDOUT:   %specific_impl_fn.loc159_5.3: <specific function> = specific_impl_function %impl.elem1.loc159_5.2, @DivWith.WithSelf.Op(%U.as_type.loc156_35.1, %DivWith.facet) [symbolic = %specific_impl_fn.loc159_5.3 (constants.%specific_impl_fn.7d6)]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type: type = fn_type @Destroy.WithSelf.SelfDestruct.1, @Destroy.WithSelf(%Result.loc154_9.1) [symbolic = %Destroy.WithSelf.SelfDestruct.type (constants.%Destroy.WithSelf.SelfDestruct.type.869b78.2)]
// CHECK:STDOUT:   %.loc159_5.7: type = fn_type_with_self_type %Destroy.WithSelf.SelfDestruct.type, %Result.loc154_9.1 [symbolic = %.loc159_5.7 (constants.%.f0c)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %Result.loc154_9.1, @Destroy [symbolic = %Destroy.lookup_impl_witness (constants.%Destroy.lookup_impl_witness.dfb4c4.2)]
// CHECK:STDOUT:   %impl.elem2.loc159_5.2: @DivWith.loc157.%.loc159_5.7 (%.f0c) = impl_witness_access %Destroy.lookup_impl_witness, element2 [symbolic = %impl.elem2.loc159_5.2 (constants.%impl.elem2)]
// CHECK:STDOUT:   %specific_impl_fn.loc159_5.4: <specific function> = specific_impl_function %impl.elem2.loc159_5.2, @Destroy.WithSelf.SelfDestruct.1(%Result.loc154_9.1) [symbolic = %specific_impl_fn.loc159_5.4 (constants.%specific_impl_fn.6f9)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%result.param: @DivWith.loc157.%Result.as_type.loc156_53.1 (%Result.as_type), %x.param: @DivWith.loc157.%T.as_type.loc157_29.1 (%T.as_type.881), %y.param: @DivWith.loc157.%U.as_type.loc156_35.1 (%U.as_type.b61)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: @DivWith.loc157.%T.as_type.loc157_29.1 (%T.as_type.881) = name_ref x, %x
// CHECK:STDOUT:     %y.ref: @DivWith.loc157.%U.as_type.loc156_35.1 (%U.as_type.b61) = name_ref y, %y
// CHECK:STDOUT:     %DivWith.type.loc159: type = facet_type <@DivWith.1, @DivWith.1(constants.%U.as_type.b61)> [symbolic = %DivWith.type.loc156_35.1 (constants.%DivWith.type.e82)]
// CHECK:STDOUT:     %.loc159_5.1: @DivWith.loc157.%DivWith.assoc_type (%DivWith.assoc_type.248) = specific_constant imports.%Core.import_ref.972, @DivWith.WithSelf(constants.%U.as_type.b61, constants.%Self.5b3) [symbolic = %assoc1 (constants.%assoc1.6e8)]
// CHECK:STDOUT:     %Op.ref: @DivWith.loc157.%DivWith.assoc_type (%DivWith.assoc_type.248) = name_ref Op, %.loc159_5.1 [symbolic = %assoc1 (constants.%assoc1.6e8)]
// CHECK:STDOUT:     %impl.elem1.loc159_5.1: @DivWith.loc157.%.loc159_5.6 (%.ba0) = impl_witness_access constants.%DivWith.lookup_impl_witness.c25, element1 [symbolic = %impl.elem1.loc159_5.2 (constants.%impl.elem1.610)]
// CHECK:STDOUT:     %bound_method.loc159_5.1: <bound method> = bound_method %x.ref, %impl.elem1.loc159_5.1
// CHECK:STDOUT:     %specific_impl_fn.loc159_5.1: <specific function> = specific_impl_function %impl.elem1.loc159_5.1, @DivWith.WithSelf.Op(constants.%U.as_type.b61, constants.%DivWith.facet.7d7) [symbolic = %specific_impl_fn.loc159_5.3 (constants.%specific_impl_fn.7d6)]
// CHECK:STDOUT:     %bound_method.loc159_5.2: <bound method> = bound_method %x.ref, %specific_impl_fn.loc159_5.1
// CHECK:STDOUT:     %.loc159_5.2: ref @DivWith.loc157.%Result.as_type.loc156_53.1 (%Result.as_type) = temporary_storage
// CHECK:STDOUT:     %DivWith.WithSelf.Op.call: init @DivWith.loc157.%Result.as_type.loc156_53.1 (%Result.as_type) to %.loc159_5.2 = call %bound_method.loc159_5.2(%x.ref, %y.ref)
// CHECK:STDOUT:     %.loc159_5.3: ref @DivWith.loc157.%Result.as_type.loc156_53.1 (%Result.as_type) = temporary %.loc159_5.2, %DivWith.WithSelf.Op.call
// CHECK:STDOUT:     %impl.elem2.loc159_5.1: @DivWith.loc157.%.loc159_5.7 (%.f0c) = impl_witness_access constants.%Destroy.lookup_impl_witness.dfb4c4.2, element2 [symbolic = %impl.elem2.loc159_5.2 (constants.%impl.elem2)]
// CHECK:STDOUT:     %bound_method.loc159_5.3: <bound method> = bound_method %.loc159_5.3, %impl.elem2.loc159_5.1
// CHECK:STDOUT:     %.loc159_5.4: %Destroy.type = converted constants.%Result.as_type, constants.%Result [symbolic = %Result.loc154_9.1 (constants.%Result)]
// CHECK:STDOUT:     %.loc159_5.5: %Destroy.type = converted constants.%Result.as_type, constants.%Result [symbolic = %Result.loc154_9.1 (constants.%Result)]
// CHECK:STDOUT:     %specific_impl_fn.loc159_5.2: <specific function> = specific_impl_function %impl.elem2.loc159_5.1, @Destroy.WithSelf.SelfDestruct.1(constants.%Result) [symbolic = %specific_impl_fn.loc159_5.4 (constants.%specific_impl_fn.6f9)]
// CHECK:STDOUT:     %bound_method.loc159_5.4: <bound method> = bound_method %.loc159_5.3, %specific_impl_fn.loc159_5.2
// CHECK:STDOUT:     %Destroy.WithSelf.SelfDestruct.call: init %empty_tuple.type = call %bound_method.loc159_5.4(%.loc159_5.3)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @ModWith.loc168(%Result.loc165_9.2: %Destroy.type, %U.loc166_4.2: %Destroy.type, %T.loc167_4.2: @ModWith.loc168.%facet_type.loc167_37 (%facet_type.247)) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %require_complete.loc170: <witness> = require_complete_type %ModWith.type.loc167_35.1 [symbolic = %require_complete.loc170 (constants.%require_complete.421)]
// CHECK:STDOUT:   %assoc1: @ModWith.loc168.%ModWith.assoc_type (%ModWith.assoc_type.85f) = assoc_entity element1, imports.%Core.import_ref.9f8f [symbolic = %assoc1 (constants.%assoc1.a6b)]
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.loc170: <witness> = lookup_impl_witness %T.loc167_4.1, @ModWith.1, @ModWith.1(%U.as_type.loc167_35.1) [symbolic = %ModWith.lookup_impl_witness.loc170 (constants.%ModWith.lookup_impl_witness.8e1)]
// CHECK:STDOUT:   %ModWith.facet: @ModWith.loc168.%ModWith.type.loc167_35.1 (%ModWith.type.475) = facet_value %T.as_type.loc168_29.1, (%ModWith.lookup_impl_witness.loc170) [symbolic = %ModWith.facet (constants.%ModWith.facet.204)]
// CHECK:STDOUT:   %ModWith.WithSelf.Op.type: type = fn_type @ModWith.WithSelf.Op, @ModWith.WithSelf(%U.as_type.loc167_35.1, %ModWith.facet) [symbolic = %ModWith.WithSelf.Op.type (constants.%ModWith.WithSelf.Op.type.340)]
// CHECK:STDOUT:   %.loc170_5.6: type = fn_type_with_self_type %ModWith.WithSelf.Op.type, %ModWith.facet [symbolic = %.loc170_5.6 (constants.%.c4c)]
// CHECK:STDOUT:   %impl.elem1.loc170_5.2: @ModWith.loc168.%.loc170_5.6 (%.c4c) = impl_witness_access %ModWith.lookup_impl_witness.loc170, element1 [symbolic = %impl.elem1.loc170_5.2 (constants.%impl.elem1.789)]
// CHECK:STDOUT:   %specific_impl_fn.loc170_5.3: <specific function> = specific_impl_function %impl.elem1.loc170_5.2, @ModWith.WithSelf.Op(%U.as_type.loc167_35.1, %ModWith.facet) [symbolic = %specific_impl_fn.loc170_5.3 (constants.%specific_impl_fn.87e)]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type: type = fn_type @Destroy.WithSelf.SelfDestruct.1, @Destroy.WithSelf(%Result.loc165_9.1) [symbolic = %Destroy.WithSelf.SelfDestruct.type (constants.%Destroy.WithSelf.SelfDestruct.type.869b78.2)]
// CHECK:STDOUT:   %.loc170_5.7: type = fn_type_with_self_type %Destroy.WithSelf.SelfDestruct.type, %Result.loc165_9.1 [symbolic = %.loc170_5.7 (constants.%.f0c)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %Result.loc165_9.1, @Destroy [symbolic = %Destroy.lookup_impl_witness (constants.%Destroy.lookup_impl_witness.dfb4c4.2)]
// CHECK:STDOUT:   %impl.elem2.loc170_5.2: @ModWith.loc168.%.loc170_5.7 (%.f0c) = impl_witness_access %Destroy.lookup_impl_witness, element2 [symbolic = %impl.elem2.loc170_5.2 (constants.%impl.elem2)]
// CHECK:STDOUT:   %specific_impl_fn.loc170_5.4: <specific function> = specific_impl_function %impl.elem2.loc170_5.2, @Destroy.WithSelf.SelfDestruct.1(%Result.loc165_9.1) [symbolic = %specific_impl_fn.loc170_5.4 (constants.%specific_impl_fn.6f9)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%result.param: @ModWith.loc168.%Result.as_type.loc167_53.1 (%Result.as_type), %x.param: @ModWith.loc168.%T.as_type.loc168_29.1 (%T.as_type.8ce), %y.param: @ModWith.loc168.%U.as_type.loc167_35.1 (%U.as_type.b61)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: @ModWith.loc168.%T.as_type.loc168_29.1 (%T.as_type.8ce) = name_ref x, %x
// CHECK:STDOUT:     %y.ref: @ModWith.loc168.%U.as_type.loc167_35.1 (%U.as_type.b61) = name_ref y, %y
// CHECK:STDOUT:     %ModWith.type.loc170: type = facet_type <@ModWith.1, @ModWith.1(constants.%U.as_type.b61)> [symbolic = %ModWith.type.loc167_35.1 (constants.%ModWith.type.475)]
// CHECK:STDOUT:     %.loc170_5.1: @ModWith.loc168.%ModWith.assoc_type (%ModWith.assoc_type.85f) = specific_constant imports.%Core.import_ref.ea5e, @ModWith.WithSelf(constants.%U.as_type.b61, constants.%Self.c18) [symbolic = %assoc1 (constants.%assoc1.a6b)]
// CHECK:STDOUT:     %Op.ref: @ModWith.loc168.%ModWith.assoc_type (%ModWith.assoc_type.85f) = name_ref Op, %.loc170_5.1 [symbolic = %assoc1 (constants.%assoc1.a6b)]
// CHECK:STDOUT:     %impl.elem1.loc170_5.1: @ModWith.loc168.%.loc170_5.6 (%.c4c) = impl_witness_access constants.%ModWith.lookup_impl_witness.8e1, element1 [symbolic = %impl.elem1.loc170_5.2 (constants.%impl.elem1.789)]
// CHECK:STDOUT:     %bound_method.loc170_5.1: <bound method> = bound_method %x.ref, %impl.elem1.loc170_5.1
// CHECK:STDOUT:     %specific_impl_fn.loc170_5.1: <specific function> = specific_impl_function %impl.elem1.loc170_5.1, @ModWith.WithSelf.Op(constants.%U.as_type.b61, constants.%ModWith.facet.204) [symbolic = %specific_impl_fn.loc170_5.3 (constants.%specific_impl_fn.87e)]
// CHECK:STDOUT:     %bound_method.loc170_5.2: <bound method> = bound_method %x.ref, %specific_impl_fn.loc170_5.1
// CHECK:STDOUT:     %.loc170_5.2: ref @ModWith.loc168.%Result.as_type.loc167_53.1 (%Result.as_type) = temporary_storage
// CHECK:STDOUT:     %ModWith.WithSelf.Op.call: init @ModWith.loc168.%Result.as_type.loc167_53.1 (%Result.as_type) to %.loc170_5.2 = call %bound_method.loc170_5.2(%x.ref, %y.ref)
// CHECK:STDOUT:     %.loc170_5.3: ref @ModWith.loc168.%Result.as_type.loc167_53.1 (%Result.as_type) = temporary %.loc170_5.2, %ModWith.WithSelf.Op.call
// CHECK:STDOUT:     %impl.elem2.loc170_5.1: @ModWith.loc168.%.loc170_5.7 (%.f0c) = impl_witness_access constants.%Destroy.lookup_impl_witness.dfb4c4.2, element2 [symbolic = %impl.elem2.loc170_5.2 (constants.%impl.elem2)]
// CHECK:STDOUT:     %bound_method.loc170_5.3: <bound method> = bound_method %.loc170_5.3, %impl.elem2.loc170_5.1
// CHECK:STDOUT:     %.loc170_5.4: %Destroy.type = converted constants.%Result.as_type, constants.%Result [symbolic = %Result.loc165_9.1 (constants.%Result)]
// CHECK:STDOUT:     %.loc170_5.5: %Destroy.type = converted constants.%Result.as_type, constants.%Result [symbolic = %Result.loc165_9.1 (constants.%Result)]
// CHECK:STDOUT:     %specific_impl_fn.loc170_5.2: <specific function> = specific_impl_function %impl.elem2.loc170_5.1, @Destroy.WithSelf.SelfDestruct.1(constants.%Result) [symbolic = %specific_impl_fn.loc170_5.4 (constants.%specific_impl_fn.6f9)]
// CHECK:STDOUT:     %bound_method.loc170_5.4: <bound method> = bound_method %.loc170_5.3, %specific_impl_fn.loc170_5.2
// CHECK:STDOUT:     %Destroy.WithSelf.SelfDestruct.call: init %empty_tuple.type = call %bound_method.loc170_5.4(%.loc170_5.3)
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
// CHECK:STDOUT:   %assoc0: @ModAssignWith.loc249.%ModAssignWith.assoc_type (%ModAssignWith.assoc_type.01d13e.2) = assoc_entity element0, imports.%Core.import_ref.97f5 [symbolic = %assoc0 (constants.%assoc0.a11270.2)]
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
// CHECK:STDOUT:   %.loc280_3.3: type = fn_type_with_self_type %Inc.WithSelf.Op.type, %T.loc278_13.1 [symbolic = %.loc280_3.3 (constants.%.5a2)]
// CHECK:STDOUT:   %Inc.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc278_13.1, @Inc [symbolic = %Inc.lookup_impl_witness (constants.%Inc.lookup_impl_witness)]
// CHECK:STDOUT:   %impl.elem0.loc280_3.2: @TestInc.%.loc280_3.3 (%.5a2) = impl_witness_access %Inc.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc280_3.2 (constants.%impl.elem0.ec8)]
// CHECK:STDOUT:   %specific_impl_fn.loc280_3.2: <specific function> = specific_impl_function %impl.elem0.loc280_3.2, @Inc.WithSelf.Op(%T.loc278_13.1) [symbolic = %specific_impl_fn.loc280_3.2 (constants.%specific_impl_fn.790)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: ref @TestInc.%T.as_type.loc278_32.1 (%T.as_type.232)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: ref @TestInc.%T.as_type.loc278_32.1 (%T.as_type.232) = name_ref x, %x
// CHECK:STDOUT:     %impl.elem0.loc280_3.1: @TestInc.%.loc280_3.3 (%.5a2) = impl_witness_access constants.%Inc.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc280_3.2 (constants.%impl.elem0.ec8)]
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
// CHECK:STDOUT: generic fn @TestNegate(%Result.loc292_9.2: %Destroy.type, %T.loc293_4.2: @TestNegate.%facet_type (%facet_type.352)) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Negate.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc293_4.1, @Negate [symbolic = %Negate.lookup_impl_witness (constants.%Negate.lookup_impl_witness.218)]
// CHECK:STDOUT:   %Negate.facet.loc296_3.3: %Negate.type = facet_value %T.as_type.loc294_24.1, (%Negate.lookup_impl_witness) [symbolic = %Negate.facet.loc296_3.3 (constants.%Negate.facet.5e5)]
// CHECK:STDOUT:   %Negate.WithSelf.Op.type: type = fn_type @Negate.WithSelf.Op, @Negate.WithSelf(%Negate.facet.loc296_3.3) [symbolic = %Negate.WithSelf.Op.type (constants.%Negate.WithSelf.Op.type.d06)]
// CHECK:STDOUT:   %.loc296_3.7: type = fn_type_with_self_type %Negate.WithSelf.Op.type, %Negate.facet.loc296_3.3 [symbolic = %.loc296_3.7 (constants.%.5a7)]
// CHECK:STDOUT:   %impl.elem1.loc296_3.2: @TestNegate.%.loc296_3.7 (%.5a7) = impl_witness_access %Negate.lookup_impl_witness, element1 [symbolic = %impl.elem1.loc296_3.2 (constants.%impl.elem1.e2e)]
// CHECK:STDOUT:   %specific_impl_fn.loc296_3.3: <specific function> = specific_impl_function %impl.elem1.loc296_3.2, @Negate.WithSelf.Op(%Negate.facet.loc296_3.3) [symbolic = %specific_impl_fn.loc296_3.3 (constants.%specific_impl_fn.0c8)]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type: type = fn_type @Destroy.WithSelf.SelfDestruct.1, @Destroy.WithSelf(%Result.loc292_9.1) [symbolic = %Destroy.WithSelf.SelfDestruct.type (constants.%Destroy.WithSelf.SelfDestruct.type.869b78.2)]
// CHECK:STDOUT:   %.loc296_3.8: type = fn_type_with_self_type %Destroy.WithSelf.SelfDestruct.type, %Result.loc292_9.1 [symbolic = %.loc296_3.8 (constants.%.f0c)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %Result.loc292_9.1, @Destroy [symbolic = %Destroy.lookup_impl_witness (constants.%Destroy.lookup_impl_witness.dfb4c4.2)]
// CHECK:STDOUT:   %impl.elem2.loc296_3.2: @TestNegate.%.loc296_3.8 (%.f0c) = impl_witness_access %Destroy.lookup_impl_witness, element2 [symbolic = %impl.elem2.loc296_3.2 (constants.%impl.elem2)]
// CHECK:STDOUT:   %specific_impl_fn.loc296_3.4: <specific function> = specific_impl_function %impl.elem2.loc296_3.2, @Destroy.WithSelf.SelfDestruct.1(%Result.loc292_9.1) [symbolic = %specific_impl_fn.loc296_3.4 (constants.%specific_impl_fn.6f9)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%y.param: @TestNegate.%Result.as_type.loc293_49.1 (%Result.as_type), %x.param: @TestNegate.%T.as_type.loc294_24.1 (%T.as_type.684)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: @TestNegate.%T.as_type.loc294_24.1 (%T.as_type.684) = name_ref x, %x
// CHECK:STDOUT:     %impl.elem1.loc296_3.1: @TestNegate.%.loc296_3.7 (%.5a7) = impl_witness_access constants.%Negate.lookup_impl_witness.218, element1 [symbolic = %impl.elem1.loc296_3.2 (constants.%impl.elem1.e2e)]
// CHECK:STDOUT:     %bound_method.loc296_3.1: <bound method> = bound_method %x.ref, %impl.elem1.loc296_3.1
// CHECK:STDOUT:     %Negate.facet.loc296_3.1: %Negate.type = facet_value constants.%T.as_type.684, (constants.%Negate.lookup_impl_witness.218) [symbolic = %Negate.facet.loc296_3.3 (constants.%Negate.facet.5e5)]
// CHECK:STDOUT:     %.loc296_3.1: %Negate.type = converted constants.%T.as_type.684, %Negate.facet.loc296_3.1 [symbolic = %Negate.facet.loc296_3.3 (constants.%Negate.facet.5e5)]
// CHECK:STDOUT:     %Negate.facet.loc296_3.2: %Negate.type = facet_value constants.%T.as_type.684, (constants.%Negate.lookup_impl_witness.218) [symbolic = %Negate.facet.loc296_3.3 (constants.%Negate.facet.5e5)]
// CHECK:STDOUT:     %.loc296_3.2: %Negate.type = converted constants.%T.as_type.684, %Negate.facet.loc296_3.2 [symbolic = %Negate.facet.loc296_3.3 (constants.%Negate.facet.5e5)]
// CHECK:STDOUT:     %specific_impl_fn.loc296_3.1: <specific function> = specific_impl_function %impl.elem1.loc296_3.1, @Negate.WithSelf.Op(constants.%Negate.facet.5e5) [symbolic = %specific_impl_fn.loc296_3.3 (constants.%specific_impl_fn.0c8)]
// CHECK:STDOUT:     %bound_method.loc296_3.2: <bound method> = bound_method %x.ref, %specific_impl_fn.loc296_3.1
// CHECK:STDOUT:     %.loc296_3.3: ref @TestNegate.%Result.as_type.loc293_49.1 (%Result.as_type) = temporary_storage
// CHECK:STDOUT:     %Negate.WithSelf.Op.call: init @TestNegate.%Result.as_type.loc293_49.1 (%Result.as_type) to %.loc296_3.3 = call %bound_method.loc296_3.2(%x.ref)
// CHECK:STDOUT:     %.loc296_3.4: ref @TestNegate.%Result.as_type.loc293_49.1 (%Result.as_type) = temporary %.loc296_3.3, %Negate.WithSelf.Op.call
// CHECK:STDOUT:     %impl.elem2.loc296_3.1: @TestNegate.%.loc296_3.8 (%.f0c) = impl_witness_access constants.%Destroy.lookup_impl_witness.dfb4c4.2, element2 [symbolic = %impl.elem2.loc296_3.2 (constants.%impl.elem2)]
// CHECK:STDOUT:     %bound_method.loc296_3.3: <bound method> = bound_method %.loc296_3.4, %impl.elem2.loc296_3.1
// CHECK:STDOUT:     %.loc296_3.5: %Destroy.type = converted constants.%Result.as_type, constants.%Result [symbolic = %Result.loc292_9.1 (constants.%Result)]
// CHECK:STDOUT:     %.loc296_3.6: %Destroy.type = converted constants.%Result.as_type, constants.%Result [symbolic = %Result.loc292_9.1 (constants.%Result)]
// CHECK:STDOUT:     %specific_impl_fn.loc296_3.2: <specific function> = specific_impl_function %impl.elem2.loc296_3.1, @Destroy.WithSelf.SelfDestruct.1(constants.%Result) [symbolic = %specific_impl_fn.loc296_3.4 (constants.%specific_impl_fn.6f9)]
// CHECK:STDOUT:     %bound_method.loc296_3.4: <bound method> = bound_method %.loc296_3.4, %specific_impl_fn.loc296_3.2
// CHECK:STDOUT:     %Destroy.WithSelf.SelfDestruct.call: init %empty_tuple.type = call %bound_method.loc296_3.4(%.loc296_3.4)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AddWith.loc124(constants.%Result, constants.%U.f40, constants.%T.dd6) {
// CHECK:STDOUT:   %Result.patt.loc121_9.2 => constants.%Result.patt
// CHECK:STDOUT:   %Result.loc121_9.1 => constants.%Result
// CHECK:STDOUT:   %U.patt.loc122_4.2 => constants.%U.patt.7a2
// CHECK:STDOUT:   %U.loc122_4.1 => constants.%U.f40
// CHECK:STDOUT:   %U.as_type.loc123_35.1 => constants.%U.as_type.b61
// CHECK:STDOUT:   %AddWith.type.loc123_35.1 => constants.%AddWith.type.d58
// CHECK:STDOUT:   %facet_type.loc123_19 => constants.%facet_type.f36
// CHECK:STDOUT:   %.Self.frozen.loc123_37.1 => constants.%.Self.frozen.03f
// CHECK:STDOUT:   %require_complete.loc123 => constants.%require_complete.621
// CHECK:STDOUT:   %.Self.frozen.as_type => constants.%.Self.frozen.as_type.464
// CHECK:STDOUT:   %AddWith.assoc_type => constants.%AddWith.assoc_type.964
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.a1a
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.loc123_43.1 => constants.%AddWith.lookup_impl_witness.99b
// CHECK:STDOUT:   %impl.elem0.loc123_43.1 => constants.%impl.elem0.e85
// CHECK:STDOUT:   %Result.as_type.loc123_53.1 => constants.%Result.as_type
// CHECK:STDOUT:   %.Self => constants.%.Self.787
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.loc123_43.2 => constants.%AddWith.lookup_impl_witness.405
// CHECK:STDOUT:   %impl.elem0.loc123_43.2 => constants.%impl.elem0.f14
// CHECK:STDOUT:   %facet_type.loc123_37 => constants.%facet_type.660
// CHECK:STDOUT:   %pattern_type.loc123 => constants.%pattern_type.6b8
// CHECK:STDOUT:   %T.patt.loc123_4.2 => constants.%T.patt.0a9
// CHECK:STDOUT:   %T.loc123_4.1 => constants.%T.dd6
// CHECK:STDOUT:   %pattern_type.loc124_16 => constants.%pattern_type.f4e485.2
// CHECK:STDOUT:   %result.param_patt.loc124_16.2 => constants.%result.param_patt.6ba
// CHECK:STDOUT:   %result.patt.loc124_16.2 => constants.%result.patt.c73b2c.1
// CHECK:STDOUT:   %T.as_type.loc124_29.1 => constants.%T.as_type.fe6
// CHECK:STDOUT:   %pattern_type.loc124_27 => constants.%pattern_type.0df
// CHECK:STDOUT:   %x.param_patt.loc124_27.2 => constants.%x.param_patt.018
// CHECK:STDOUT:   %x.patt.loc124_27.2 => constants.%x.patt.3b5
// CHECK:STDOUT:   %pattern_type.loc124_33 => constants.%pattern_type.753
// CHECK:STDOUT:   %y.param_patt.loc124_33.2 => constants.%y.param_patt.740
// CHECK:STDOUT:   %y.patt.loc124_33.2 => constants.%y.patt.603
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @SubWith.loc135(constants.%Result, constants.%U.f40, constants.%T.34f) {
// CHECK:STDOUT:   %Result.patt.loc132_9.2 => constants.%Result.patt
// CHECK:STDOUT:   %Result.loc132_9.1 => constants.%Result
// CHECK:STDOUT:   %U.patt.loc133_4.2 => constants.%U.patt.7a2
// CHECK:STDOUT:   %U.loc133_4.1 => constants.%U.f40
// CHECK:STDOUT:   %U.as_type.loc134_35.1 => constants.%U.as_type.b61
// CHECK:STDOUT:   %SubWith.type.loc134_35.1 => constants.%SubWith.type.d62
// CHECK:STDOUT:   %facet_type.loc134_19 => constants.%facet_type.e74
// CHECK:STDOUT:   %.Self.frozen.loc134_37.1 => constants.%.Self.frozen.d95
// CHECK:STDOUT:   %require_complete.loc134 => constants.%require_complete.a5f
// CHECK:STDOUT:   %.Self.frozen.as_type => constants.%.Self.frozen.as_type.319
// CHECK:STDOUT:   %SubWith.assoc_type => constants.%SubWith.assoc_type.378
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.7c6
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.loc134_43.1 => constants.%SubWith.lookup_impl_witness.d59
// CHECK:STDOUT:   %impl.elem0.loc134_43.1 => constants.%impl.elem0.68a
// CHECK:STDOUT:   %Result.as_type.loc134_53.1 => constants.%Result.as_type
// CHECK:STDOUT:   %.Self => constants.%.Self.f1c
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.loc134_43.2 => constants.%SubWith.lookup_impl_witness.e7f
// CHECK:STDOUT:   %impl.elem0.loc134_43.2 => constants.%impl.elem0.a1a
// CHECK:STDOUT:   %facet_type.loc134_37 => constants.%facet_type.ba3
// CHECK:STDOUT:   %pattern_type.loc134 => constants.%pattern_type.500
// CHECK:STDOUT:   %T.patt.loc134_4.2 => constants.%T.patt.3ed
// CHECK:STDOUT:   %T.loc134_4.1 => constants.%T.34f
// CHECK:STDOUT:   %pattern_type.loc135_16 => constants.%pattern_type.f4e485.2
// CHECK:STDOUT:   %result.param_patt.loc135_16.2 => constants.%result.param_patt.6ba
// CHECK:STDOUT:   %result.patt.loc135_16.2 => constants.%result.patt.c73b2c.1
// CHECK:STDOUT:   %T.as_type.loc135_29.1 => constants.%T.as_type.dfc
// CHECK:STDOUT:   %pattern_type.loc135_27 => constants.%pattern_type.c9e
// CHECK:STDOUT:   %x.param_patt.loc135_27.2 => constants.%x.param_patt.bd1
// CHECK:STDOUT:   %x.patt.loc135_27.2 => constants.%x.patt.684
// CHECK:STDOUT:   %pattern_type.loc135_33 => constants.%pattern_type.753
// CHECK:STDOUT:   %y.param_patt.loc135_33.2 => constants.%y.param_patt.740
// CHECK:STDOUT:   %y.patt.loc135_33.2 => constants.%y.patt.603
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @MulWith.loc146(constants.%Result, constants.%U.f40, constants.%T.961) {
// CHECK:STDOUT:   %Result.patt.loc143_9.2 => constants.%Result.patt
// CHECK:STDOUT:   %Result.loc143_9.1 => constants.%Result
// CHECK:STDOUT:   %U.patt.loc144_4.2 => constants.%U.patt.7a2
// CHECK:STDOUT:   %U.loc144_4.1 => constants.%U.f40
// CHECK:STDOUT:   %U.as_type.loc145_35.1 => constants.%U.as_type.b61
// CHECK:STDOUT:   %MulWith.type.loc145_35.1 => constants.%MulWith.type.5c9
// CHECK:STDOUT:   %facet_type.loc145_19 => constants.%facet_type.945
// CHECK:STDOUT:   %.Self.frozen.loc145_37.1 => constants.%.Self.frozen.10d
// CHECK:STDOUT:   %require_complete.loc145 => constants.%require_complete.264
// CHECK:STDOUT:   %.Self.frozen.as_type => constants.%.Self.frozen.as_type.f69
// CHECK:STDOUT:   %MulWith.assoc_type => constants.%MulWith.assoc_type.1d7
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.78b
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.loc145_43.1 => constants.%MulWith.lookup_impl_witness.fa1
// CHECK:STDOUT:   %impl.elem0.loc145_43.1 => constants.%impl.elem0.095
// CHECK:STDOUT:   %Result.as_type.loc145_53.1 => constants.%Result.as_type
// CHECK:STDOUT:   %.Self => constants.%.Self.cb0
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.loc145_43.2 => constants.%MulWith.lookup_impl_witness.1ff
// CHECK:STDOUT:   %impl.elem0.loc145_43.2 => constants.%impl.elem0.ad0
// CHECK:STDOUT:   %facet_type.loc145_37 => constants.%facet_type.9c7
// CHECK:STDOUT:   %pattern_type.loc145 => constants.%pattern_type.112
// CHECK:STDOUT:   %T.patt.loc145_4.2 => constants.%T.patt.9e1
// CHECK:STDOUT:   %T.loc145_4.1 => constants.%T.961
// CHECK:STDOUT:   %pattern_type.loc146_16 => constants.%pattern_type.f4e485.2
// CHECK:STDOUT:   %result.param_patt.loc146_16.2 => constants.%result.param_patt.6ba
// CHECK:STDOUT:   %result.patt.loc146_16.2 => constants.%result.patt.c73b2c.1
// CHECK:STDOUT:   %T.as_type.loc146_29.1 => constants.%T.as_type.521
// CHECK:STDOUT:   %pattern_type.loc146_27 => constants.%pattern_type.3f7
// CHECK:STDOUT:   %x.param_patt.loc146_27.2 => constants.%x.param_patt.a5b
// CHECK:STDOUT:   %x.patt.loc146_27.2 => constants.%x.patt.eae
// CHECK:STDOUT:   %pattern_type.loc146_33 => constants.%pattern_type.753
// CHECK:STDOUT:   %y.param_patt.loc146_33.2 => constants.%y.param_patt.740
// CHECK:STDOUT:   %y.patt.loc146_33.2 => constants.%y.patt.603
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @DivWith.loc157(constants.%Result, constants.%U.f40, constants.%T.d59) {
// CHECK:STDOUT:   %Result.patt.loc154_9.2 => constants.%Result.patt
// CHECK:STDOUT:   %Result.loc154_9.1 => constants.%Result
// CHECK:STDOUT:   %U.patt.loc155_4.2 => constants.%U.patt.7a2
// CHECK:STDOUT:   %U.loc155_4.1 => constants.%U.f40
// CHECK:STDOUT:   %U.as_type.loc156_35.1 => constants.%U.as_type.b61
// CHECK:STDOUT:   %DivWith.type.loc156_35.1 => constants.%DivWith.type.e82
// CHECK:STDOUT:   %facet_type.loc156_19 => constants.%facet_type.084
// CHECK:STDOUT:   %.Self.frozen.loc156_37.1 => constants.%.Self.frozen.8e5
// CHECK:STDOUT:   %require_complete.loc156 => constants.%require_complete.1ee
// CHECK:STDOUT:   %.Self.frozen.as_type => constants.%.Self.frozen.as_type.379
// CHECK:STDOUT:   %DivWith.assoc_type => constants.%DivWith.assoc_type.248
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.b46
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.loc156_43.1 => constants.%DivWith.lookup_impl_witness.cd9
// CHECK:STDOUT:   %impl.elem0.loc156_43.1 => constants.%impl.elem0.76b
// CHECK:STDOUT:   %Result.as_type.loc156_53.1 => constants.%Result.as_type
// CHECK:STDOUT:   %.Self => constants.%.Self.f43
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.loc156_43.2 => constants.%DivWith.lookup_impl_witness.f1e
// CHECK:STDOUT:   %impl.elem0.loc156_43.2 => constants.%impl.elem0.bba
// CHECK:STDOUT:   %facet_type.loc156_37 => constants.%facet_type.89a
// CHECK:STDOUT:   %pattern_type.loc156 => constants.%pattern_type.658
// CHECK:STDOUT:   %T.patt.loc156_4.2 => constants.%T.patt.506
// CHECK:STDOUT:   %T.loc156_4.1 => constants.%T.d59
// CHECK:STDOUT:   %pattern_type.loc157_16 => constants.%pattern_type.f4e485.2
// CHECK:STDOUT:   %result.param_patt.loc157_16.2 => constants.%result.param_patt.6ba
// CHECK:STDOUT:   %result.patt.loc157_16.2 => constants.%result.patt.c73b2c.1
// CHECK:STDOUT:   %T.as_type.loc157_29.1 => constants.%T.as_type.881
// CHECK:STDOUT:   %pattern_type.loc157_27 => constants.%pattern_type.a65
// CHECK:STDOUT:   %x.param_patt.loc157_27.2 => constants.%x.param_patt.8e8
// CHECK:STDOUT:   %x.patt.loc157_27.2 => constants.%x.patt.d40
// CHECK:STDOUT:   %pattern_type.loc157_33 => constants.%pattern_type.753
// CHECK:STDOUT:   %y.param_patt.loc157_33.2 => constants.%y.param_patt.740
// CHECK:STDOUT:   %y.patt.loc157_33.2 => constants.%y.patt.603
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ModWith.loc168(constants.%Result, constants.%U.f40, constants.%T.be8) {
// CHECK:STDOUT:   %Result.patt.loc165_9.2 => constants.%Result.patt
// CHECK:STDOUT:   %Result.loc165_9.1 => constants.%Result
// CHECK:STDOUT:   %U.patt.loc166_4.2 => constants.%U.patt.7a2
// CHECK:STDOUT:   %U.loc166_4.1 => constants.%U.f40
// CHECK:STDOUT:   %U.as_type.loc167_35.1 => constants.%U.as_type.b61
// CHECK:STDOUT:   %ModWith.type.loc167_35.1 => constants.%ModWith.type.475
// CHECK:STDOUT:   %facet_type.loc167_19 => constants.%facet_type.fd4
// CHECK:STDOUT:   %.Self.frozen.loc167_37.1 => constants.%.Self.frozen.fe0
// CHECK:STDOUT:   %require_complete.loc167 => constants.%require_complete.8f0
// CHECK:STDOUT:   %.Self.frozen.as_type => constants.%.Self.frozen.as_type.78c
// CHECK:STDOUT:   %ModWith.assoc_type => constants.%ModWith.assoc_type.85f
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.bf1
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.loc167_43.1 => constants.%ModWith.lookup_impl_witness.448
// CHECK:STDOUT:   %impl.elem0.loc167_43.1 => constants.%impl.elem0.043
// CHECK:STDOUT:   %Result.as_type.loc167_53.1 => constants.%Result.as_type
// CHECK:STDOUT:   %.Self => constants.%.Self.dd1
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.loc167_43.2 => constants.%ModWith.lookup_impl_witness.4d5
// CHECK:STDOUT:   %impl.elem0.loc167_43.2 => constants.%impl.elem0.cb3
// CHECK:STDOUT:   %facet_type.loc167_37 => constants.%facet_type.247
// CHECK:STDOUT:   %pattern_type.loc167 => constants.%pattern_type.c70
// CHECK:STDOUT:   %T.patt.loc167_4.2 => constants.%T.patt.7fe
// CHECK:STDOUT:   %T.loc167_4.1 => constants.%T.be8
// CHECK:STDOUT:   %pattern_type.loc168_16 => constants.%pattern_type.f4e485.2
// CHECK:STDOUT:   %result.param_patt.loc168_16.2 => constants.%result.param_patt.6ba
// CHECK:STDOUT:   %result.patt.loc168_16.2 => constants.%result.patt.c73b2c.1
// CHECK:STDOUT:   %T.as_type.loc168_29.1 => constants.%T.as_type.8ce
// CHECK:STDOUT:   %pattern_type.loc168_27 => constants.%pattern_type.5e2
// CHECK:STDOUT:   %x.param_patt.loc168_27.2 => constants.%x.param_patt.41c
// CHECK:STDOUT:   %x.patt.loc168_27.2 => constants.%x.patt.2de
// CHECK:STDOUT:   %pattern_type.loc168_33 => constants.%pattern_type.753
// CHECK:STDOUT:   %y.param_patt.loc168_33.2 => constants.%y.param_patt.740
// CHECK:STDOUT:   %y.patt.loc168_33.2 => constants.%y.patt.603
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AddWith.loc124(constants.%Destroy.facet.f58, constants.%Destroy.facet.f58, constants.%facet_value.5e0) {
// CHECK:STDOUT:   %Result.patt.loc121_9.2 => constants.%Result.patt
// CHECK:STDOUT:   %Result.loc121_9.1 => constants.%Destroy.facet.f58
// CHECK:STDOUT:   %U.patt.loc122_4.2 => constants.%U.patt.7a2
// CHECK:STDOUT:   %U.loc122_4.1 => constants.%Destroy.facet.f58
// CHECK:STDOUT:   %U.as_type.loc123_35.1 => constants.%Int16
// CHECK:STDOUT:   %AddWith.type.loc123_35.1 => constants.%AddWith.type.e4b
// CHECK:STDOUT:   %facet_type.loc123_19 => constants.%facet_type.067
// CHECK:STDOUT:   %.Self.frozen.loc123_37.1 => constants.%.Self.frozen.30b
// CHECK:STDOUT:   %require_complete.loc123 => constants.%complete_type.cd3
// CHECK:STDOUT:   %.Self.frozen.as_type => constants.%.Self.frozen.as_type.bb1
// CHECK:STDOUT:   %AddWith.assoc_type => constants.%AddWith.assoc_type.86e
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.30c
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.loc123_43.1 => constants.%AddWith.lookup_impl_witness.b71
// CHECK:STDOUT:   %impl.elem0.loc123_43.1 => constants.%impl.elem0.0bf
// CHECK:STDOUT:   %Result.as_type.loc123_53.1 => constants.%Int16
// CHECK:STDOUT:   %.Self => constants.%.Self.4c6
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.loc123_43.2 => constants.%AddWith.lookup_impl_witness.5df
// CHECK:STDOUT:   %impl.elem0.loc123_43.2 => constants.%impl.elem0.2cc
// CHECK:STDOUT:   %facet_type.loc123_37 => constants.%facet_type.754
// CHECK:STDOUT:   %pattern_type.loc123 => constants.%pattern_type.9c4
// CHECK:STDOUT:   %T.patt.loc123_4.2 => constants.%T.patt.96c
// CHECK:STDOUT:   %T.loc123_4.1 => constants.%facet_value.5e0
// CHECK:STDOUT:   %pattern_type.loc124_16 => constants.%pattern_type.785
// CHECK:STDOUT:   %result.param_patt.loc124_16.2 => constants.%result.param_patt.be6
// CHECK:STDOUT:   %result.patt.loc124_16.2 => constants.%result.patt.a31
// CHECK:STDOUT:   %T.as_type.loc124_29.1 => constants.%i16
// CHECK:STDOUT:   %pattern_type.loc124_27 => constants.%pattern_type.cbf
// CHECK:STDOUT:   %x.param_patt.loc124_27.2 => constants.%x.param_patt.6cd
// CHECK:STDOUT:   %x.patt.loc124_27.2 => constants.%x.patt.580
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
// CHECK:STDOUT:   %AddWith.facet => constants.%AddWith.facet.41e
// CHECK:STDOUT:   %AddWith.WithSelf.Op.type => constants.%AddWith.WithSelf.Op.type.348
// CHECK:STDOUT:   %.loc126_5.6 => constants.%.30b
// CHECK:STDOUT:   %impl.elem1.loc126_5.2 => constants.%Op.72717c.1
// CHECK:STDOUT:   %specific_impl_fn.loc126_5.3 => constants.%Op.72717c.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type => constants.%Destroy.WithSelf.SelfDestruct.type.610
// CHECK:STDOUT:   %.loc126_5.7 => constants.%.348
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.1bfa2a.1
// CHECK:STDOUT:   %impl.elem2.loc126_5.2 => constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %specific_impl_fn.loc126_5.4 => constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @SubWith.loc135(constants.%Destroy.facet.f58, constants.%Destroy.facet.f58, constants.%facet_value.ec5) {
// CHECK:STDOUT:   %Result.patt.loc132_9.2 => constants.%Result.patt
// CHECK:STDOUT:   %Result.loc132_9.1 => constants.%Destroy.facet.f58
// CHECK:STDOUT:   %U.patt.loc133_4.2 => constants.%U.patt.7a2
// CHECK:STDOUT:   %U.loc133_4.1 => constants.%Destroy.facet.f58
// CHECK:STDOUT:   %U.as_type.loc134_35.1 => constants.%Int16
// CHECK:STDOUT:   %SubWith.type.loc134_35.1 => constants.%SubWith.type.68d
// CHECK:STDOUT:   %facet_type.loc134_19 => constants.%facet_type.90f
// CHECK:STDOUT:   %.Self.frozen.loc134_37.1 => constants.%.Self.frozen.b1a
// CHECK:STDOUT:   %require_complete.loc134 => constants.%complete_type.067
// CHECK:STDOUT:   %.Self.frozen.as_type => constants.%.Self.frozen.as_type.917
// CHECK:STDOUT:   %SubWith.assoc_type => constants.%SubWith.assoc_type.dd6
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.dfb
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.loc134_43.1 => constants.%SubWith.lookup_impl_witness.135
// CHECK:STDOUT:   %impl.elem0.loc134_43.1 => constants.%impl.elem0.1d6
// CHECK:STDOUT:   %Result.as_type.loc134_53.1 => constants.%Int16
// CHECK:STDOUT:   %.Self => constants.%.Self.565
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.loc134_43.2 => constants.%SubWith.lookup_impl_witness.dd3
// CHECK:STDOUT:   %impl.elem0.loc134_43.2 => constants.%impl.elem0.f92
// CHECK:STDOUT:   %facet_type.loc134_37 => constants.%facet_type.7bc
// CHECK:STDOUT:   %pattern_type.loc134 => constants.%pattern_type.f9a
// CHECK:STDOUT:   %T.patt.loc134_4.2 => constants.%T.patt.edb
// CHECK:STDOUT:   %T.loc134_4.1 => constants.%facet_value.ec5
// CHECK:STDOUT:   %pattern_type.loc135_16 => constants.%pattern_type.785
// CHECK:STDOUT:   %result.param_patt.loc135_16.2 => constants.%result.param_patt.be6
// CHECK:STDOUT:   %result.patt.loc135_16.2 => constants.%result.patt.a31
// CHECK:STDOUT:   %T.as_type.loc135_29.1 => constants.%i16
// CHECK:STDOUT:   %pattern_type.loc135_27 => constants.%pattern_type.cbf
// CHECK:STDOUT:   %x.param_patt.loc135_27.2 => constants.%x.param_patt.6cd
// CHECK:STDOUT:   %x.patt.loc135_27.2 => constants.%x.patt.580
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
// CHECK:STDOUT:   %SubWith.facet => constants.%SubWith.facet.46f
// CHECK:STDOUT:   %SubWith.WithSelf.Op.type => constants.%SubWith.WithSelf.Op.type.d0c
// CHECK:STDOUT:   %.loc137_5.6 => constants.%.4e84
// CHECK:STDOUT:   %impl.elem1.loc137_5.2 => constants.%Op.72717c.2
// CHECK:STDOUT:   %specific_impl_fn.loc137_5.3 => constants.%Op.72717c.2
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type => constants.%Destroy.WithSelf.SelfDestruct.type.610
// CHECK:STDOUT:   %.loc137_5.7 => constants.%.348
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.1bfa2a.1
// CHECK:STDOUT:   %impl.elem2.loc137_5.2 => constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %specific_impl_fn.loc137_5.4 => constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @MulWith.loc146(constants.%Destroy.facet.f58, constants.%Destroy.facet.f58, constants.%facet_value.7b7) {
// CHECK:STDOUT:   %Result.patt.loc143_9.2 => constants.%Result.patt
// CHECK:STDOUT:   %Result.loc143_9.1 => constants.%Destroy.facet.f58
// CHECK:STDOUT:   %U.patt.loc144_4.2 => constants.%U.patt.7a2
// CHECK:STDOUT:   %U.loc144_4.1 => constants.%Destroy.facet.f58
// CHECK:STDOUT:   %U.as_type.loc145_35.1 => constants.%Int16
// CHECK:STDOUT:   %MulWith.type.loc145_35.1 => constants.%MulWith.type.f65
// CHECK:STDOUT:   %facet_type.loc145_19 => constants.%facet_type.ab1
// CHECK:STDOUT:   %.Self.frozen.loc145_37.1 => constants.%.Self.frozen.752
// CHECK:STDOUT:   %require_complete.loc145 => constants.%complete_type.797
// CHECK:STDOUT:   %.Self.frozen.as_type => constants.%.Self.frozen.as_type.3c5
// CHECK:STDOUT:   %MulWith.assoc_type => constants.%MulWith.assoc_type.4c3
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.6ec
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.loc145_43.1 => constants.%MulWith.lookup_impl_witness.25c
// CHECK:STDOUT:   %impl.elem0.loc145_43.1 => constants.%impl.elem0.c8f
// CHECK:STDOUT:   %Result.as_type.loc145_53.1 => constants.%Int16
// CHECK:STDOUT:   %.Self => constants.%.Self.1e1
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.loc145_43.2 => constants.%MulWith.lookup_impl_witness.d39
// CHECK:STDOUT:   %impl.elem0.loc145_43.2 => constants.%impl.elem0.979
// CHECK:STDOUT:   %facet_type.loc145_37 => constants.%facet_type.e2b
// CHECK:STDOUT:   %pattern_type.loc145 => constants.%pattern_type.fbc
// CHECK:STDOUT:   %T.patt.loc145_4.2 => constants.%T.patt.d6f
// CHECK:STDOUT:   %T.loc145_4.1 => constants.%facet_value.7b7
// CHECK:STDOUT:   %pattern_type.loc146_16 => constants.%pattern_type.785
// CHECK:STDOUT:   %result.param_patt.loc146_16.2 => constants.%result.param_patt.be6
// CHECK:STDOUT:   %result.patt.loc146_16.2 => constants.%result.patt.a31
// CHECK:STDOUT:   %T.as_type.loc146_29.1 => constants.%i16
// CHECK:STDOUT:   %pattern_type.loc146_27 => constants.%pattern_type.cbf
// CHECK:STDOUT:   %x.param_patt.loc146_27.2 => constants.%x.param_patt.6cd
// CHECK:STDOUT:   %x.patt.loc146_27.2 => constants.%x.patt.580
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
// CHECK:STDOUT:   %MulWith.facet => constants.%MulWith.facet.808
// CHECK:STDOUT:   %MulWith.WithSelf.Op.type => constants.%MulWith.WithSelf.Op.type.263
// CHECK:STDOUT:   %.loc148_5.6 => constants.%.3b1
// CHECK:STDOUT:   %impl.elem1.loc148_5.2 => constants.%Op.72717c.3
// CHECK:STDOUT:   %specific_impl_fn.loc148_5.3 => constants.%Op.72717c.3
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type => constants.%Destroy.WithSelf.SelfDestruct.type.610
// CHECK:STDOUT:   %.loc148_5.7 => constants.%.348
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.1bfa2a.1
// CHECK:STDOUT:   %impl.elem2.loc148_5.2 => constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %specific_impl_fn.loc148_5.4 => constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @DivWith.loc157(constants.%Destroy.facet.f58, constants.%Destroy.facet.f58, constants.%facet_value.8d4) {
// CHECK:STDOUT:   %Result.patt.loc154_9.2 => constants.%Result.patt
// CHECK:STDOUT:   %Result.loc154_9.1 => constants.%Destroy.facet.f58
// CHECK:STDOUT:   %U.patt.loc155_4.2 => constants.%U.patt.7a2
// CHECK:STDOUT:   %U.loc155_4.1 => constants.%Destroy.facet.f58
// CHECK:STDOUT:   %U.as_type.loc156_35.1 => constants.%Int16
// CHECK:STDOUT:   %DivWith.type.loc156_35.1 => constants.%DivWith.type.b8a
// CHECK:STDOUT:   %facet_type.loc156_19 => constants.%facet_type.b64
// CHECK:STDOUT:   %.Self.frozen.loc156_37.1 => constants.%.Self.frozen.797
// CHECK:STDOUT:   %require_complete.loc156 => constants.%complete_type.ca5
// CHECK:STDOUT:   %.Self.frozen.as_type => constants.%.Self.frozen.as_type.b76
// CHECK:STDOUT:   %DivWith.assoc_type => constants.%DivWith.assoc_type.712
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.99c
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.loc156_43.1 => constants.%DivWith.lookup_impl_witness.60a
// CHECK:STDOUT:   %impl.elem0.loc156_43.1 => constants.%impl.elem0.431
// CHECK:STDOUT:   %Result.as_type.loc156_53.1 => constants.%Int16
// CHECK:STDOUT:   %.Self => constants.%.Self.315
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.loc156_43.2 => constants.%DivWith.lookup_impl_witness.147
// CHECK:STDOUT:   %impl.elem0.loc156_43.2 => constants.%impl.elem0.17b
// CHECK:STDOUT:   %facet_type.loc156_37 => constants.%facet_type.f21
// CHECK:STDOUT:   %pattern_type.loc156 => constants.%pattern_type.7ca
// CHECK:STDOUT:   %T.patt.loc156_4.2 => constants.%T.patt.905
// CHECK:STDOUT:   %T.loc156_4.1 => constants.%facet_value.8d4
// CHECK:STDOUT:   %pattern_type.loc157_16 => constants.%pattern_type.785
// CHECK:STDOUT:   %result.param_patt.loc157_16.2 => constants.%result.param_patt.be6
// CHECK:STDOUT:   %result.patt.loc157_16.2 => constants.%result.patt.a31
// CHECK:STDOUT:   %T.as_type.loc157_29.1 => constants.%i16
// CHECK:STDOUT:   %pattern_type.loc157_27 => constants.%pattern_type.cbf
// CHECK:STDOUT:   %x.param_patt.loc157_27.2 => constants.%x.param_patt.6cd
// CHECK:STDOUT:   %x.patt.loc157_27.2 => constants.%x.patt.580
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
// CHECK:STDOUT:   %DivWith.facet => constants.%DivWith.facet.f0e
// CHECK:STDOUT:   %DivWith.WithSelf.Op.type => constants.%DivWith.WithSelf.Op.type.833
// CHECK:STDOUT:   %.loc159_5.6 => constants.%.6c7
// CHECK:STDOUT:   %impl.elem1.loc159_5.2 => constants.%Op.72717c.4
// CHECK:STDOUT:   %specific_impl_fn.loc159_5.3 => constants.%Op.72717c.4
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type => constants.%Destroy.WithSelf.SelfDestruct.type.610
// CHECK:STDOUT:   %.loc159_5.7 => constants.%.348
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.1bfa2a.1
// CHECK:STDOUT:   %impl.elem2.loc159_5.2 => constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %specific_impl_fn.loc159_5.4 => constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ModWith.loc168(constants.%Destroy.facet.f58, constants.%Destroy.facet.f58, constants.%facet_value.a68) {
// CHECK:STDOUT:   %Result.patt.loc165_9.2 => constants.%Result.patt
// CHECK:STDOUT:   %Result.loc165_9.1 => constants.%Destroy.facet.f58
// CHECK:STDOUT:   %U.patt.loc166_4.2 => constants.%U.patt.7a2
// CHECK:STDOUT:   %U.loc166_4.1 => constants.%Destroy.facet.f58
// CHECK:STDOUT:   %U.as_type.loc167_35.1 => constants.%Int16
// CHECK:STDOUT:   %ModWith.type.loc167_35.1 => constants.%ModWith.type.c67
// CHECK:STDOUT:   %facet_type.loc167_19 => constants.%facet_type.2c9
// CHECK:STDOUT:   %.Self.frozen.loc167_37.1 => constants.%.Self.frozen.7c6
// CHECK:STDOUT:   %require_complete.loc167 => constants.%complete_type.300
// CHECK:STDOUT:   %.Self.frozen.as_type => constants.%.Self.frozen.as_type.c18
// CHECK:STDOUT:   %ModWith.assoc_type => constants.%ModWith.assoc_type.5f6
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.f04
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.loc167_43.1 => constants.%ModWith.lookup_impl_witness.49e
// CHECK:STDOUT:   %impl.elem0.loc167_43.1 => constants.%impl.elem0.042
// CHECK:STDOUT:   %Result.as_type.loc167_53.1 => constants.%Int16
// CHECK:STDOUT:   %.Self => constants.%.Self.7d4
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.loc167_43.2 => constants.%ModWith.lookup_impl_witness.c30
// CHECK:STDOUT:   %impl.elem0.loc167_43.2 => constants.%impl.elem0.e30
// CHECK:STDOUT:   %facet_type.loc167_37 => constants.%facet_type.76c
// CHECK:STDOUT:   %pattern_type.loc167 => constants.%pattern_type.976
// CHECK:STDOUT:   %T.patt.loc167_4.2 => constants.%T.patt.2a9
// CHECK:STDOUT:   %T.loc167_4.1 => constants.%facet_value.a68
// CHECK:STDOUT:   %pattern_type.loc168_16 => constants.%pattern_type.785
// CHECK:STDOUT:   %result.param_patt.loc168_16.2 => constants.%result.param_patt.be6
// CHECK:STDOUT:   %result.patt.loc168_16.2 => constants.%result.patt.a31
// CHECK:STDOUT:   %T.as_type.loc168_29.1 => constants.%i16
// CHECK:STDOUT:   %pattern_type.loc168_27 => constants.%pattern_type.cbf
// CHECK:STDOUT:   %x.param_patt.loc168_27.2 => constants.%x.param_patt.6cd
// CHECK:STDOUT:   %x.patt.loc168_27.2 => constants.%x.patt.580
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
// CHECK:STDOUT:   %ModWith.facet => constants.%ModWith.facet.139
// CHECK:STDOUT:   %ModWith.WithSelf.Op.type => constants.%ModWith.WithSelf.Op.type.f47
// CHECK:STDOUT:   %.loc170_5.6 => constants.%.1cf
// CHECK:STDOUT:   %impl.elem1.loc170_5.2 => constants.%Op.72717c.5
// CHECK:STDOUT:   %specific_impl_fn.loc170_5.3 => constants.%Op.72717c.5
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type => constants.%Destroy.WithSelf.SelfDestruct.type.610
// CHECK:STDOUT:   %.loc170_5.7 => constants.%.348
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.1bfa2a.1
// CHECK:STDOUT:   %impl.elem2.loc170_5.2 => constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %specific_impl_fn.loc170_5.4 => constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AddWith.loc124(constants.%Destroy.facet.f58, constants.%Destroy.facet.091, constants.%facet_value.b63) {
// CHECK:STDOUT:   %Result.patt.loc121_9.2 => constants.%Result.patt
// CHECK:STDOUT:   %Result.loc121_9.1 => constants.%Destroy.facet.f58
// CHECK:STDOUT:   %U.patt.loc122_4.2 => constants.%U.patt.7a2
// CHECK:STDOUT:   %U.loc122_4.1 => constants.%Destroy.facet.091
// CHECK:STDOUT:   %U.as_type.loc123_35.1 => constants.%i16
// CHECK:STDOUT:   %AddWith.type.loc123_35.1 => constants.%AddWith.type.631
// CHECK:STDOUT:   %facet_type.loc123_19 => constants.%facet_type.c24
// CHECK:STDOUT:   %.Self.frozen.loc123_37.1 => constants.%.Self.frozen.1fa
// CHECK:STDOUT:   %require_complete.loc123 => constants.%complete_type.448
// CHECK:STDOUT:   %.Self.frozen.as_type => constants.%.Self.frozen.as_type.019
// CHECK:STDOUT:   %AddWith.assoc_type => constants.%AddWith.assoc_type.c35
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.9a7
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.loc123_43.1 => constants.%AddWith.lookup_impl_witness.d9d
// CHECK:STDOUT:   %impl.elem0.loc123_43.1 => constants.%impl.elem0.1e3
// CHECK:STDOUT:   %Result.as_type.loc123_53.1 => constants.%Int16
// CHECK:STDOUT:   %.Self => constants.%.Self.7e7
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.loc123_43.2 => constants.%AddWith.lookup_impl_witness.a15
// CHECK:STDOUT:   %impl.elem0.loc123_43.2 => constants.%impl.elem0.198
// CHECK:STDOUT:   %facet_type.loc123_37 => constants.%facet_type.877
// CHECK:STDOUT:   %pattern_type.loc123 => constants.%pattern_type.1bb
// CHECK:STDOUT:   %T.patt.loc123_4.2 => constants.%T.patt.c2e
// CHECK:STDOUT:   %T.loc123_4.1 => constants.%facet_value.b63
// CHECK:STDOUT:   %pattern_type.loc124_16 => constants.%pattern_type.785
// CHECK:STDOUT:   %result.param_patt.loc124_16.2 => constants.%result.param_patt.be6
// CHECK:STDOUT:   %result.patt.loc124_16.2 => constants.%result.patt.a31
// CHECK:STDOUT:   %T.as_type.loc124_29.1 => constants.%Int16
// CHECK:STDOUT:   %pattern_type.loc124_27 => constants.%pattern_type.785
// CHECK:STDOUT:   %x.param_patt.loc124_27.2 => constants.%x.param_patt.3b0
// CHECK:STDOUT:   %x.patt.loc124_27.2 => constants.%x.patt.869
// CHECK:STDOUT:   %pattern_type.loc124_33 => constants.%pattern_type.cbf
// CHECK:STDOUT:   %y.param_patt.loc124_33.2 => constants.%y.param_patt.40e
// CHECK:STDOUT:   %y.patt.loc124_33.2 => constants.%y.patt.b86
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc124_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc124_27 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc124_33 => constants.%complete_type.2da
// CHECK:STDOUT:   %require_complete.loc126 => constants.%complete_type.4db
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.fb4
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.loc126 => constants.%custom_witness.917
// CHECK:STDOUT:   %AddWith.facet => constants.%AddWith.facet.b20
// CHECK:STDOUT:   %AddWith.WithSelf.Op.type => constants.%AddWith.WithSelf.Op.type.aba
// CHECK:STDOUT:   %.loc126_5.6 => constants.%.689
// CHECK:STDOUT:   %impl.elem1.loc126_5.2 => constants.%Int16.Op.4dca31.2
// CHECK:STDOUT:   %specific_impl_fn.loc126_5.3 => constants.%Int16.Op.4dca31.2
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type => constants.%Destroy.WithSelf.SelfDestruct.type.610
// CHECK:STDOUT:   %.loc126_5.7 => constants.%.348
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.1bfa2a.1
// CHECK:STDOUT:   %impl.elem2.loc126_5.2 => constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %specific_impl_fn.loc126_5.4 => constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @SubWith.loc135(constants.%Destroy.facet.f58, constants.%Destroy.facet.091, constants.%facet_value.ab3) {
// CHECK:STDOUT:   %Result.patt.loc132_9.2 => constants.%Result.patt
// CHECK:STDOUT:   %Result.loc132_9.1 => constants.%Destroy.facet.f58
// CHECK:STDOUT:   %U.patt.loc133_4.2 => constants.%U.patt.7a2
// CHECK:STDOUT:   %U.loc133_4.1 => constants.%Destroy.facet.091
// CHECK:STDOUT:   %U.as_type.loc134_35.1 => constants.%i16
// CHECK:STDOUT:   %SubWith.type.loc134_35.1 => constants.%SubWith.type.971
// CHECK:STDOUT:   %facet_type.loc134_19 => constants.%facet_type.75c
// CHECK:STDOUT:   %.Self.frozen.loc134_37.1 => constants.%.Self.frozen.2164
// CHECK:STDOUT:   %require_complete.loc134 => constants.%complete_type.916
// CHECK:STDOUT:   %.Self.frozen.as_type => constants.%.Self.frozen.as_type.1b5
// CHECK:STDOUT:   %SubWith.assoc_type => constants.%SubWith.assoc_type.6b7
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.8aa
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.loc134_43.1 => constants.%SubWith.lookup_impl_witness.f08
// CHECK:STDOUT:   %impl.elem0.loc134_43.1 => constants.%impl.elem0.ead
// CHECK:STDOUT:   %Result.as_type.loc134_53.1 => constants.%Int16
// CHECK:STDOUT:   %.Self => constants.%.Self.2d8
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.loc134_43.2 => constants.%SubWith.lookup_impl_witness.80f
// CHECK:STDOUT:   %impl.elem0.loc134_43.2 => constants.%impl.elem0.bd6
// CHECK:STDOUT:   %facet_type.loc134_37 => constants.%facet_type.d1b
// CHECK:STDOUT:   %pattern_type.loc134 => constants.%pattern_type.5cb
// CHECK:STDOUT:   %T.patt.loc134_4.2 => constants.%T.patt.83e
// CHECK:STDOUT:   %T.loc134_4.1 => constants.%facet_value.ab3
// CHECK:STDOUT:   %pattern_type.loc135_16 => constants.%pattern_type.785
// CHECK:STDOUT:   %result.param_patt.loc135_16.2 => constants.%result.param_patt.be6
// CHECK:STDOUT:   %result.patt.loc135_16.2 => constants.%result.patt.a31
// CHECK:STDOUT:   %T.as_type.loc135_29.1 => constants.%Int16
// CHECK:STDOUT:   %pattern_type.loc135_27 => constants.%pattern_type.785
// CHECK:STDOUT:   %x.param_patt.loc135_27.2 => constants.%x.param_patt.3b0
// CHECK:STDOUT:   %x.patt.loc135_27.2 => constants.%x.patt.869
// CHECK:STDOUT:   %pattern_type.loc135_33 => constants.%pattern_type.cbf
// CHECK:STDOUT:   %y.param_patt.loc135_33.2 => constants.%y.param_patt.40e
// CHECK:STDOUT:   %y.patt.loc135_33.2 => constants.%y.patt.b86
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc135_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc135_27 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc135_33 => constants.%complete_type.2da
// CHECK:STDOUT:   %require_complete.loc137 => constants.%complete_type.c36
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.b62
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.loc137 => constants.%custom_witness.1d8
// CHECK:STDOUT:   %SubWith.facet => constants.%SubWith.facet.b6f
// CHECK:STDOUT:   %SubWith.WithSelf.Op.type => constants.%SubWith.WithSelf.Op.type.90f
// CHECK:STDOUT:   %.loc137_5.6 => constants.%.6a5
// CHECK:STDOUT:   %impl.elem1.loc137_5.2 => constants.%Int16.Op.4dca31.3
// CHECK:STDOUT:   %specific_impl_fn.loc137_5.3 => constants.%Int16.Op.4dca31.3
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type => constants.%Destroy.WithSelf.SelfDestruct.type.610
// CHECK:STDOUT:   %.loc137_5.7 => constants.%.348
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.1bfa2a.1
// CHECK:STDOUT:   %impl.elem2.loc137_5.2 => constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %specific_impl_fn.loc137_5.4 => constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @MulWith.loc146(constants.%Destroy.facet.f58, constants.%Destroy.facet.091, constants.%facet_value.19b) {
// CHECK:STDOUT:   %Result.patt.loc143_9.2 => constants.%Result.patt
// CHECK:STDOUT:   %Result.loc143_9.1 => constants.%Destroy.facet.f58
// CHECK:STDOUT:   %U.patt.loc144_4.2 => constants.%U.patt.7a2
// CHECK:STDOUT:   %U.loc144_4.1 => constants.%Destroy.facet.091
// CHECK:STDOUT:   %U.as_type.loc145_35.1 => constants.%i16
// CHECK:STDOUT:   %MulWith.type.loc145_35.1 => constants.%MulWith.type.f13
// CHECK:STDOUT:   %facet_type.loc145_19 => constants.%facet_type.35c
// CHECK:STDOUT:   %.Self.frozen.loc145_37.1 => constants.%.Self.frozen.532
// CHECK:STDOUT:   %require_complete.loc145 => constants.%complete_type.ea3
// CHECK:STDOUT:   %.Self.frozen.as_type => constants.%.Self.frozen.as_type.1ed
// CHECK:STDOUT:   %MulWith.assoc_type => constants.%MulWith.assoc_type.4df
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.958
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.loc145_43.1 => constants.%MulWith.lookup_impl_witness.0d6
// CHECK:STDOUT:   %impl.elem0.loc145_43.1 => constants.%impl.elem0.ea8
// CHECK:STDOUT:   %Result.as_type.loc145_53.1 => constants.%Int16
// CHECK:STDOUT:   %.Self => constants.%.Self.6d3
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.loc145_43.2 => constants.%MulWith.lookup_impl_witness.eb9
// CHECK:STDOUT:   %impl.elem0.loc145_43.2 => constants.%impl.elem0.3ca
// CHECK:STDOUT:   %facet_type.loc145_37 => constants.%facet_type.d3b
// CHECK:STDOUT:   %pattern_type.loc145 => constants.%pattern_type.36b
// CHECK:STDOUT:   %T.patt.loc145_4.2 => constants.%T.patt.173
// CHECK:STDOUT:   %T.loc145_4.1 => constants.%facet_value.19b
// CHECK:STDOUT:   %pattern_type.loc146_16 => constants.%pattern_type.785
// CHECK:STDOUT:   %result.param_patt.loc146_16.2 => constants.%result.param_patt.be6
// CHECK:STDOUT:   %result.patt.loc146_16.2 => constants.%result.patt.a31
// CHECK:STDOUT:   %T.as_type.loc146_29.1 => constants.%Int16
// CHECK:STDOUT:   %pattern_type.loc146_27 => constants.%pattern_type.785
// CHECK:STDOUT:   %x.param_patt.loc146_27.2 => constants.%x.param_patt.3b0
// CHECK:STDOUT:   %x.patt.loc146_27.2 => constants.%x.patt.869
// CHECK:STDOUT:   %pattern_type.loc146_33 => constants.%pattern_type.cbf
// CHECK:STDOUT:   %y.param_patt.loc146_33.2 => constants.%y.param_patt.40e
// CHECK:STDOUT:   %y.patt.loc146_33.2 => constants.%y.patt.b86
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc146_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc146_27 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc146_33 => constants.%complete_type.2da
// CHECK:STDOUT:   %require_complete.loc148 => constants.%complete_type.864
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.403
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.loc148 => constants.%custom_witness.36e
// CHECK:STDOUT:   %MulWith.facet => constants.%MulWith.facet.6f9
// CHECK:STDOUT:   %MulWith.WithSelf.Op.type => constants.%MulWith.WithSelf.Op.type.f0e
// CHECK:STDOUT:   %.loc148_5.6 => constants.%.b9b
// CHECK:STDOUT:   %impl.elem1.loc148_5.2 => constants.%Int16.Op.4dca31.4
// CHECK:STDOUT:   %specific_impl_fn.loc148_5.3 => constants.%Int16.Op.4dca31.4
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type => constants.%Destroy.WithSelf.SelfDestruct.type.610
// CHECK:STDOUT:   %.loc148_5.7 => constants.%.348
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.1bfa2a.1
// CHECK:STDOUT:   %impl.elem2.loc148_5.2 => constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %specific_impl_fn.loc148_5.4 => constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @DivWith.loc157(constants.%Destroy.facet.f58, constants.%Destroy.facet.091, constants.%facet_value.a28) {
// CHECK:STDOUT:   %Result.patt.loc154_9.2 => constants.%Result.patt
// CHECK:STDOUT:   %Result.loc154_9.1 => constants.%Destroy.facet.f58
// CHECK:STDOUT:   %U.patt.loc155_4.2 => constants.%U.patt.7a2
// CHECK:STDOUT:   %U.loc155_4.1 => constants.%Destroy.facet.091
// CHECK:STDOUT:   %U.as_type.loc156_35.1 => constants.%i16
// CHECK:STDOUT:   %DivWith.type.loc156_35.1 => constants.%DivWith.type.f46
// CHECK:STDOUT:   %facet_type.loc156_19 => constants.%facet_type.39f
// CHECK:STDOUT:   %.Self.frozen.loc156_37.1 => constants.%.Self.frozen.dd6
// CHECK:STDOUT:   %require_complete.loc156 => constants.%complete_type.370
// CHECK:STDOUT:   %.Self.frozen.as_type => constants.%.Self.frozen.as_type.f55
// CHECK:STDOUT:   %DivWith.assoc_type => constants.%DivWith.assoc_type.7cc
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.0cd
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.loc156_43.1 => constants.%DivWith.lookup_impl_witness.804
// CHECK:STDOUT:   %impl.elem0.loc156_43.1 => constants.%impl.elem0.358
// CHECK:STDOUT:   %Result.as_type.loc156_53.1 => constants.%Int16
// CHECK:STDOUT:   %.Self => constants.%.Self.eef
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.loc156_43.2 => constants.%DivWith.lookup_impl_witness.25a
// CHECK:STDOUT:   %impl.elem0.loc156_43.2 => constants.%impl.elem0.1c6
// CHECK:STDOUT:   %facet_type.loc156_37 => constants.%facet_type.f5f
// CHECK:STDOUT:   %pattern_type.loc156 => constants.%pattern_type.2b9
// CHECK:STDOUT:   %T.patt.loc156_4.2 => constants.%T.patt.bd4
// CHECK:STDOUT:   %T.loc156_4.1 => constants.%facet_value.a28
// CHECK:STDOUT:   %pattern_type.loc157_16 => constants.%pattern_type.785
// CHECK:STDOUT:   %result.param_patt.loc157_16.2 => constants.%result.param_patt.be6
// CHECK:STDOUT:   %result.patt.loc157_16.2 => constants.%result.patt.a31
// CHECK:STDOUT:   %T.as_type.loc157_29.1 => constants.%Int16
// CHECK:STDOUT:   %pattern_type.loc157_27 => constants.%pattern_type.785
// CHECK:STDOUT:   %x.param_patt.loc157_27.2 => constants.%x.param_patt.3b0
// CHECK:STDOUT:   %x.patt.loc157_27.2 => constants.%x.patt.869
// CHECK:STDOUT:   %pattern_type.loc157_33 => constants.%pattern_type.cbf
// CHECK:STDOUT:   %y.param_patt.loc157_33.2 => constants.%y.param_patt.40e
// CHECK:STDOUT:   %y.patt.loc157_33.2 => constants.%y.patt.b86
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc157_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc157_27 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc157_33 => constants.%complete_type.2da
// CHECK:STDOUT:   %require_complete.loc159 => constants.%complete_type.730
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.c4f
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.loc159 => constants.%custom_witness.5f8
// CHECK:STDOUT:   %DivWith.facet => constants.%DivWith.facet.b7c
// CHECK:STDOUT:   %DivWith.WithSelf.Op.type => constants.%DivWith.WithSelf.Op.type.b11
// CHECK:STDOUT:   %.loc159_5.6 => constants.%.20e
// CHECK:STDOUT:   %impl.elem1.loc159_5.2 => constants.%Int16.Op.4dca31.5
// CHECK:STDOUT:   %specific_impl_fn.loc159_5.3 => constants.%Int16.Op.4dca31.5
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type => constants.%Destroy.WithSelf.SelfDestruct.type.610
// CHECK:STDOUT:   %.loc159_5.7 => constants.%.348
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.1bfa2a.1
// CHECK:STDOUT:   %impl.elem2.loc159_5.2 => constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %specific_impl_fn.loc159_5.4 => constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ModWith.loc168(constants.%Destroy.facet.f58, constants.%Destroy.facet.091, constants.%facet_value.09b) {
// CHECK:STDOUT:   %Result.patt.loc165_9.2 => constants.%Result.patt
// CHECK:STDOUT:   %Result.loc165_9.1 => constants.%Destroy.facet.f58
// CHECK:STDOUT:   %U.patt.loc166_4.2 => constants.%U.patt.7a2
// CHECK:STDOUT:   %U.loc166_4.1 => constants.%Destroy.facet.091
// CHECK:STDOUT:   %U.as_type.loc167_35.1 => constants.%i16
// CHECK:STDOUT:   %ModWith.type.loc167_35.1 => constants.%ModWith.type.065
// CHECK:STDOUT:   %facet_type.loc167_19 => constants.%facet_type.ce0
// CHECK:STDOUT:   %.Self.frozen.loc167_37.1 => constants.%.Self.frozen.817
// CHECK:STDOUT:   %require_complete.loc167 => constants.%complete_type.c87
// CHECK:STDOUT:   %.Self.frozen.as_type => constants.%.Self.frozen.as_type.251
// CHECK:STDOUT:   %ModWith.assoc_type => constants.%ModWith.assoc_type.57d
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.529
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.loc167_43.1 => constants.%ModWith.lookup_impl_witness.7f8
// CHECK:STDOUT:   %impl.elem0.loc167_43.1 => constants.%impl.elem0.6d3
// CHECK:STDOUT:   %Result.as_type.loc167_53.1 => constants.%Int16
// CHECK:STDOUT:   %.Self => constants.%.Self.acc
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.loc167_43.2 => constants.%ModWith.lookup_impl_witness.2d5
// CHECK:STDOUT:   %impl.elem0.loc167_43.2 => constants.%impl.elem0.e17
// CHECK:STDOUT:   %facet_type.loc167_37 => constants.%facet_type.b68
// CHECK:STDOUT:   %pattern_type.loc167 => constants.%pattern_type.587
// CHECK:STDOUT:   %T.patt.loc167_4.2 => constants.%T.patt.e39
// CHECK:STDOUT:   %T.loc167_4.1 => constants.%facet_value.09b
// CHECK:STDOUT:   %pattern_type.loc168_16 => constants.%pattern_type.785
// CHECK:STDOUT:   %result.param_patt.loc168_16.2 => constants.%result.param_patt.be6
// CHECK:STDOUT:   %result.patt.loc168_16.2 => constants.%result.patt.a31
// CHECK:STDOUT:   %T.as_type.loc168_29.1 => constants.%Int16
// CHECK:STDOUT:   %pattern_type.loc168_27 => constants.%pattern_type.785
// CHECK:STDOUT:   %x.param_patt.loc168_27.2 => constants.%x.param_patt.3b0
// CHECK:STDOUT:   %x.patt.loc168_27.2 => constants.%x.patt.869
// CHECK:STDOUT:   %pattern_type.loc168_33 => constants.%pattern_type.cbf
// CHECK:STDOUT:   %y.param_patt.loc168_33.2 => constants.%y.param_patt.40e
// CHECK:STDOUT:   %y.patt.loc168_33.2 => constants.%y.patt.b86
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc168_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc168_27 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc168_33 => constants.%complete_type.2da
// CHECK:STDOUT:   %require_complete.loc170 => constants.%complete_type.b59
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.929
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.loc170 => constants.%custom_witness.127
// CHECK:STDOUT:   %ModWith.facet => constants.%ModWith.facet.45c
// CHECK:STDOUT:   %ModWith.WithSelf.Op.type => constants.%ModWith.WithSelf.Op.type.377
// CHECK:STDOUT:   %.loc170_5.6 => constants.%.448
// CHECK:STDOUT:   %impl.elem1.loc170_5.2 => constants.%Int16.Op.4dca31.6
// CHECK:STDOUT:   %specific_impl_fn.loc170_5.3 => constants.%Int16.Op.4dca31.6
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type => constants.%Destroy.WithSelf.SelfDestruct.type.610
// CHECK:STDOUT:   %.loc170_5.7 => constants.%.348
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.1bfa2a.1
// CHECK:STDOUT:   %impl.elem2.loc170_5.2 => constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %specific_impl_fn.loc170_5.4 => constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AddWith.loc124(constants.%Destroy.facet.84d, constants.%Destroy.facet.84d, constants.%facet_value.bf4) {
// CHECK:STDOUT:   %Result.patt.loc121_9.2 => constants.%Result.patt
// CHECK:STDOUT:   %Result.loc121_9.1 => constants.%Destroy.facet.84d
// CHECK:STDOUT:   %U.patt.loc122_4.2 => constants.%U.patt.7a2
// CHECK:STDOUT:   %U.loc122_4.1 => constants.%Destroy.facet.84d
// CHECK:STDOUT:   %U.as_type.loc123_35.1 => constants.%Int32
// CHECK:STDOUT:   %AddWith.type.loc123_35.1 => constants.%AddWith.type.db7
// CHECK:STDOUT:   %facet_type.loc123_19 => constants.%facet_type.5cd
// CHECK:STDOUT:   %.Self.frozen.loc123_37.1 => constants.%.Self.frozen.52a
// CHECK:STDOUT:   %require_complete.loc123 => constants.%complete_type.18d
// CHECK:STDOUT:   %.Self.frozen.as_type => constants.%.Self.frozen.as_type.988
// CHECK:STDOUT:   %AddWith.assoc_type => constants.%AddWith.assoc_type.be8
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.31c
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.loc123_43.1 => constants.%AddWith.lookup_impl_witness.3ac
// CHECK:STDOUT:   %impl.elem0.loc123_43.1 => constants.%impl.elem0.958
// CHECK:STDOUT:   %Result.as_type.loc123_53.1 => constants.%Int32
// CHECK:STDOUT:   %.Self => constants.%.Self.726
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.loc123_43.2 => constants.%AddWith.lookup_impl_witness.bcf
// CHECK:STDOUT:   %impl.elem0.loc123_43.2 => constants.%impl.elem0.953
// CHECK:STDOUT:   %facet_type.loc123_37 => constants.%facet_type.030
// CHECK:STDOUT:   %pattern_type.loc123 => constants.%pattern_type.c1b
// CHECK:STDOUT:   %T.patt.loc123_4.2 => constants.%T.patt.16ff
// CHECK:STDOUT:   %T.loc123_4.1 => constants.%facet_value.bf4
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
// CHECK:STDOUT:   %specific_impl_fn.loc126_5.3 => constants.%Int16.Op.4dca31.7
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type => constants.%Destroy.WithSelf.SelfDestruct.type.53c
// CHECK:STDOUT:   %.loc126_5.7 => constants.%.77c
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.1bfa2a.2
// CHECK:STDOUT:   %impl.elem2.loc126_5.2 => constants.%Destroy.WithSelf.SelfDestruct.db3fdb.3
// CHECK:STDOUT:   %specific_impl_fn.loc126_5.4 => constants.%Destroy.WithSelf.SelfDestruct.db3fdb.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @SubWith.loc135(constants.%Destroy.facet.84d, constants.%Destroy.facet.84d, constants.%facet_value.436) {
// CHECK:STDOUT:   %Result.patt.loc132_9.2 => constants.%Result.patt
// CHECK:STDOUT:   %Result.loc132_9.1 => constants.%Destroy.facet.84d
// CHECK:STDOUT:   %U.patt.loc133_4.2 => constants.%U.patt.7a2
// CHECK:STDOUT:   %U.loc133_4.1 => constants.%Destroy.facet.84d
// CHECK:STDOUT:   %U.as_type.loc134_35.1 => constants.%Int32
// CHECK:STDOUT:   %SubWith.type.loc134_35.1 => constants.%SubWith.type.120
// CHECK:STDOUT:   %facet_type.loc134_19 => constants.%facet_type.393
// CHECK:STDOUT:   %.Self.frozen.loc134_37.1 => constants.%.Self.frozen.36d
// CHECK:STDOUT:   %require_complete.loc134 => constants.%complete_type.8a6
// CHECK:STDOUT:   %.Self.frozen.as_type => constants.%.Self.frozen.as_type.88b
// CHECK:STDOUT:   %SubWith.assoc_type => constants.%SubWith.assoc_type.dae
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.fac
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.loc134_43.1 => constants.%SubWith.lookup_impl_witness.10e
// CHECK:STDOUT:   %impl.elem0.loc134_43.1 => constants.%impl.elem0.d88
// CHECK:STDOUT:   %Result.as_type.loc134_53.1 => constants.%Int32
// CHECK:STDOUT:   %.Self => constants.%.Self.eac
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.loc134_43.2 => constants.%SubWith.lookup_impl_witness.0dc
// CHECK:STDOUT:   %impl.elem0.loc134_43.2 => constants.%impl.elem0.bb5a
// CHECK:STDOUT:   %facet_type.loc134_37 => constants.%facet_type.767
// CHECK:STDOUT:   %pattern_type.loc134 => constants.%pattern_type.974
// CHECK:STDOUT:   %T.patt.loc134_4.2 => constants.%T.patt.3c7
// CHECK:STDOUT:   %T.loc134_4.1 => constants.%facet_value.436
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
// CHECK:STDOUT:   %specific_impl_fn.loc137_5.3 => constants.%Int16.Op.4dca31.8
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type => constants.%Destroy.WithSelf.SelfDestruct.type.53c
// CHECK:STDOUT:   %.loc137_5.7 => constants.%.77c
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.1bfa2a.2
// CHECK:STDOUT:   %impl.elem2.loc137_5.2 => constants.%Destroy.WithSelf.SelfDestruct.db3fdb.3
// CHECK:STDOUT:   %specific_impl_fn.loc137_5.4 => constants.%Destroy.WithSelf.SelfDestruct.db3fdb.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @MulWith.loc146(constants.%Destroy.facet.84d, constants.%Destroy.facet.84d, constants.%facet_value.1b5) {
// CHECK:STDOUT:   %Result.patt.loc143_9.2 => constants.%Result.patt
// CHECK:STDOUT:   %Result.loc143_9.1 => constants.%Destroy.facet.84d
// CHECK:STDOUT:   %U.patt.loc144_4.2 => constants.%U.patt.7a2
// CHECK:STDOUT:   %U.loc144_4.1 => constants.%Destroy.facet.84d
// CHECK:STDOUT:   %U.as_type.loc145_35.1 => constants.%Int32
// CHECK:STDOUT:   %MulWith.type.loc145_35.1 => constants.%MulWith.type.d67
// CHECK:STDOUT:   %facet_type.loc145_19 => constants.%facet_type.ea6
// CHECK:STDOUT:   %.Self.frozen.loc145_37.1 => constants.%.Self.frozen.95b
// CHECK:STDOUT:   %require_complete.loc145 => constants.%complete_type.8fe
// CHECK:STDOUT:   %.Self.frozen.as_type => constants.%.Self.frozen.as_type.f47
// CHECK:STDOUT:   %MulWith.assoc_type => constants.%MulWith.assoc_type.ab1
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.d0a
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.loc145_43.1 => constants.%MulWith.lookup_impl_witness.675
// CHECK:STDOUT:   %impl.elem0.loc145_43.1 => constants.%impl.elem0.1de
// CHECK:STDOUT:   %Result.as_type.loc145_53.1 => constants.%Int32
// CHECK:STDOUT:   %.Self => constants.%.Self.44f
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.loc145_43.2 => constants.%MulWith.lookup_impl_witness.527
// CHECK:STDOUT:   %impl.elem0.loc145_43.2 => constants.%impl.elem0.054
// CHECK:STDOUT:   %facet_type.loc145_37 => constants.%facet_type.9d8
// CHECK:STDOUT:   %pattern_type.loc145 => constants.%pattern_type.a89
// CHECK:STDOUT:   %T.patt.loc145_4.2 => constants.%T.patt.144
// CHECK:STDOUT:   %T.loc145_4.1 => constants.%facet_value.1b5
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
// CHECK:STDOUT:   %specific_impl_fn.loc148_5.3 => constants.%Int16.Op.4dca31.9
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type => constants.%Destroy.WithSelf.SelfDestruct.type.53c
// CHECK:STDOUT:   %.loc148_5.7 => constants.%.77c
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.1bfa2a.2
// CHECK:STDOUT:   %impl.elem2.loc148_5.2 => constants.%Destroy.WithSelf.SelfDestruct.db3fdb.3
// CHECK:STDOUT:   %specific_impl_fn.loc148_5.4 => constants.%Destroy.WithSelf.SelfDestruct.db3fdb.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @DivWith.loc157(constants.%Destroy.facet.84d, constants.%Destroy.facet.84d, constants.%facet_value.f68) {
// CHECK:STDOUT:   %Result.patt.loc154_9.2 => constants.%Result.patt
// CHECK:STDOUT:   %Result.loc154_9.1 => constants.%Destroy.facet.84d
// CHECK:STDOUT:   %U.patt.loc155_4.2 => constants.%U.patt.7a2
// CHECK:STDOUT:   %U.loc155_4.1 => constants.%Destroy.facet.84d
// CHECK:STDOUT:   %U.as_type.loc156_35.1 => constants.%Int32
// CHECK:STDOUT:   %DivWith.type.loc156_35.1 => constants.%DivWith.type.5a1
// CHECK:STDOUT:   %facet_type.loc156_19 => constants.%facet_type.821
// CHECK:STDOUT:   %.Self.frozen.loc156_37.1 => constants.%.Self.frozen.3ed
// CHECK:STDOUT:   %require_complete.loc156 => constants.%complete_type.5bd
// CHECK:STDOUT:   %.Self.frozen.as_type => constants.%.Self.frozen.as_type.553
// CHECK:STDOUT:   %DivWith.assoc_type => constants.%DivWith.assoc_type.22f
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.a9d
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.loc156_43.1 => constants.%DivWith.lookup_impl_witness.1af
// CHECK:STDOUT:   %impl.elem0.loc156_43.1 => constants.%impl.elem0.a46
// CHECK:STDOUT:   %Result.as_type.loc156_53.1 => constants.%Int32
// CHECK:STDOUT:   %.Self => constants.%.Self.37d
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.loc156_43.2 => constants.%DivWith.lookup_impl_witness.7ae
// CHECK:STDOUT:   %impl.elem0.loc156_43.2 => constants.%impl.elem0.6c4
// CHECK:STDOUT:   %facet_type.loc156_37 => constants.%facet_type.f48
// CHECK:STDOUT:   %pattern_type.loc156 => constants.%pattern_type.4e6
// CHECK:STDOUT:   %T.patt.loc156_4.2 => constants.%T.patt.def
// CHECK:STDOUT:   %T.loc156_4.1 => constants.%facet_value.f68
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
// CHECK:STDOUT:   %specific_impl_fn.loc159_5.3 => constants.%Int16.Op.4dca31.10
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type => constants.%Destroy.WithSelf.SelfDestruct.type.53c
// CHECK:STDOUT:   %.loc159_5.7 => constants.%.77c
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.1bfa2a.2
// CHECK:STDOUT:   %impl.elem2.loc159_5.2 => constants.%Destroy.WithSelf.SelfDestruct.db3fdb.3
// CHECK:STDOUT:   %specific_impl_fn.loc159_5.4 => constants.%Destroy.WithSelf.SelfDestruct.db3fdb.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ModWith.loc168(constants.%Destroy.facet.84d, constants.%Destroy.facet.84d, constants.%facet_value.46b) {
// CHECK:STDOUT:   %Result.patt.loc165_9.2 => constants.%Result.patt
// CHECK:STDOUT:   %Result.loc165_9.1 => constants.%Destroy.facet.84d
// CHECK:STDOUT:   %U.patt.loc166_4.2 => constants.%U.patt.7a2
// CHECK:STDOUT:   %U.loc166_4.1 => constants.%Destroy.facet.84d
// CHECK:STDOUT:   %U.as_type.loc167_35.1 => constants.%Int32
// CHECK:STDOUT:   %ModWith.type.loc167_35.1 => constants.%ModWith.type.794
// CHECK:STDOUT:   %facet_type.loc167_19 => constants.%facet_type.9c3
// CHECK:STDOUT:   %.Self.frozen.loc167_37.1 => constants.%.Self.frozen.b381
// CHECK:STDOUT:   %require_complete.loc167 => constants.%complete_type.0b7
// CHECK:STDOUT:   %.Self.frozen.as_type => constants.%.Self.frozen.as_type.f01
// CHECK:STDOUT:   %ModWith.assoc_type => constants.%ModWith.assoc_type.021
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.310
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.loc167_43.1 => constants.%ModWith.lookup_impl_witness.ac5
// CHECK:STDOUT:   %impl.elem0.loc167_43.1 => constants.%impl.elem0.914
// CHECK:STDOUT:   %Result.as_type.loc167_53.1 => constants.%Int32
// CHECK:STDOUT:   %.Self => constants.%.Self.f60
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.loc167_43.2 => constants.%ModWith.lookup_impl_witness.9a9
// CHECK:STDOUT:   %impl.elem0.loc167_43.2 => constants.%impl.elem0.522
// CHECK:STDOUT:   %facet_type.loc167_37 => constants.%facet_type.115
// CHECK:STDOUT:   %pattern_type.loc167 => constants.%pattern_type.393
// CHECK:STDOUT:   %T.patt.loc167_4.2 => constants.%T.patt.be9
// CHECK:STDOUT:   %T.loc167_4.1 => constants.%facet_value.46b
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
// CHECK:STDOUT:   %specific_impl_fn.loc170_5.3 => constants.%Int16.Op.4dca31.11
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type => constants.%Destroy.WithSelf.SelfDestruct.type.53c
// CHECK:STDOUT:   %.loc170_5.7 => constants.%.77c
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.1bfa2a.2
// CHECK:STDOUT:   %impl.elem2.loc170_5.2 => constants.%Destroy.WithSelf.SelfDestruct.db3fdb.3
// CHECK:STDOUT:   %specific_impl_fn.loc170_5.4 => constants.%Destroy.WithSelf.SelfDestruct.db3fdb.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AddWith.loc124(constants.%Destroy.facet.84d, constants.%Destroy.facet.f58, constants.%facet_value.e86) {
// CHECK:STDOUT:   %Result.patt.loc121_9.2 => constants.%Result.patt
// CHECK:STDOUT:   %Result.loc121_9.1 => constants.%Destroy.facet.84d
// CHECK:STDOUT:   %U.patt.loc122_4.2 => constants.%U.patt.7a2
// CHECK:STDOUT:   %U.loc122_4.1 => constants.%Destroy.facet.f58
// CHECK:STDOUT:   %U.as_type.loc123_35.1 => constants.%Int16
// CHECK:STDOUT:   %AddWith.type.loc123_35.1 => constants.%AddWith.type.e4b
// CHECK:STDOUT:   %facet_type.loc123_19 => constants.%facet_type.067
// CHECK:STDOUT:   %.Self.frozen.loc123_37.1 => constants.%.Self.frozen.30b
// CHECK:STDOUT:   %require_complete.loc123 => constants.%complete_type.cd3
// CHECK:STDOUT:   %.Self.frozen.as_type => constants.%.Self.frozen.as_type.bb1
// CHECK:STDOUT:   %AddWith.assoc_type => constants.%AddWith.assoc_type.86e
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.30c
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.loc123_43.1 => constants.%AddWith.lookup_impl_witness.b71
// CHECK:STDOUT:   %impl.elem0.loc123_43.1 => constants.%impl.elem0.0bf
// CHECK:STDOUT:   %Result.as_type.loc123_53.1 => constants.%Int32
// CHECK:STDOUT:   %.Self => constants.%.Self.4c6
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.loc123_43.2 => constants.%AddWith.lookup_impl_witness.5df
// CHECK:STDOUT:   %impl.elem0.loc123_43.2 => constants.%impl.elem0.2cc
// CHECK:STDOUT:   %facet_type.loc123_37 => constants.%facet_type.788
// CHECK:STDOUT:   %pattern_type.loc123 => constants.%pattern_type.dc6
// CHECK:STDOUT:   %T.patt.loc123_4.2 => constants.%T.patt.0c1
// CHECK:STDOUT:   %T.loc123_4.1 => constants.%facet_value.e86
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
// CHECK:STDOUT:   %specific_impl_fn.loc126_5.3 => constants.%Int32.Op.4dca31.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type => constants.%Destroy.WithSelf.SelfDestruct.type.53c
// CHECK:STDOUT:   %.loc126_5.7 => constants.%.77c
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.1bfa2a.2
// CHECK:STDOUT:   %impl.elem2.loc126_5.2 => constants.%Destroy.WithSelf.SelfDestruct.db3fdb.3
// CHECK:STDOUT:   %specific_impl_fn.loc126_5.4 => constants.%Destroy.WithSelf.SelfDestruct.db3fdb.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @SubWith.loc135(constants.%Destroy.facet.84d, constants.%Destroy.facet.f58, constants.%facet_value.b78) {
// CHECK:STDOUT:   %Result.patt.loc132_9.2 => constants.%Result.patt
// CHECK:STDOUT:   %Result.loc132_9.1 => constants.%Destroy.facet.84d
// CHECK:STDOUT:   %U.patt.loc133_4.2 => constants.%U.patt.7a2
// CHECK:STDOUT:   %U.loc133_4.1 => constants.%Destroy.facet.f58
// CHECK:STDOUT:   %U.as_type.loc134_35.1 => constants.%Int16
// CHECK:STDOUT:   %SubWith.type.loc134_35.1 => constants.%SubWith.type.68d
// CHECK:STDOUT:   %facet_type.loc134_19 => constants.%facet_type.90f
// CHECK:STDOUT:   %.Self.frozen.loc134_37.1 => constants.%.Self.frozen.b1a
// CHECK:STDOUT:   %require_complete.loc134 => constants.%complete_type.067
// CHECK:STDOUT:   %.Self.frozen.as_type => constants.%.Self.frozen.as_type.917
// CHECK:STDOUT:   %SubWith.assoc_type => constants.%SubWith.assoc_type.dd6
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.dfb
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.loc134_43.1 => constants.%SubWith.lookup_impl_witness.135
// CHECK:STDOUT:   %impl.elem0.loc134_43.1 => constants.%impl.elem0.1d6
// CHECK:STDOUT:   %Result.as_type.loc134_53.1 => constants.%Int32
// CHECK:STDOUT:   %.Self => constants.%.Self.565
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.loc134_43.2 => constants.%SubWith.lookup_impl_witness.dd3
// CHECK:STDOUT:   %impl.elem0.loc134_43.2 => constants.%impl.elem0.f92
// CHECK:STDOUT:   %facet_type.loc134_37 => constants.%facet_type.1b0
// CHECK:STDOUT:   %pattern_type.loc134 => constants.%pattern_type.6d5
// CHECK:STDOUT:   %T.patt.loc134_4.2 => constants.%T.patt.282
// CHECK:STDOUT:   %T.loc134_4.1 => constants.%facet_value.b78
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
// CHECK:STDOUT:   %specific_impl_fn.loc137_5.3 => constants.%Int32.Op.4dca31.2
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type => constants.%Destroy.WithSelf.SelfDestruct.type.53c
// CHECK:STDOUT:   %.loc137_5.7 => constants.%.77c
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.1bfa2a.2
// CHECK:STDOUT:   %impl.elem2.loc137_5.2 => constants.%Destroy.WithSelf.SelfDestruct.db3fdb.3
// CHECK:STDOUT:   %specific_impl_fn.loc137_5.4 => constants.%Destroy.WithSelf.SelfDestruct.db3fdb.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @MulWith.loc146(constants.%Destroy.facet.84d, constants.%Destroy.facet.f58, constants.%facet_value.191) {
// CHECK:STDOUT:   %Result.patt.loc143_9.2 => constants.%Result.patt
// CHECK:STDOUT:   %Result.loc143_9.1 => constants.%Destroy.facet.84d
// CHECK:STDOUT:   %U.patt.loc144_4.2 => constants.%U.patt.7a2
// CHECK:STDOUT:   %U.loc144_4.1 => constants.%Destroy.facet.f58
// CHECK:STDOUT:   %U.as_type.loc145_35.1 => constants.%Int16
// CHECK:STDOUT:   %MulWith.type.loc145_35.1 => constants.%MulWith.type.f65
// CHECK:STDOUT:   %facet_type.loc145_19 => constants.%facet_type.ab1
// CHECK:STDOUT:   %.Self.frozen.loc145_37.1 => constants.%.Self.frozen.752
// CHECK:STDOUT:   %require_complete.loc145 => constants.%complete_type.797
// CHECK:STDOUT:   %.Self.frozen.as_type => constants.%.Self.frozen.as_type.3c5
// CHECK:STDOUT:   %MulWith.assoc_type => constants.%MulWith.assoc_type.4c3
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.6ec
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.loc145_43.1 => constants.%MulWith.lookup_impl_witness.25c
// CHECK:STDOUT:   %impl.elem0.loc145_43.1 => constants.%impl.elem0.c8f
// CHECK:STDOUT:   %Result.as_type.loc145_53.1 => constants.%Int32
// CHECK:STDOUT:   %.Self => constants.%.Self.1e1
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.loc145_43.2 => constants.%MulWith.lookup_impl_witness.d39
// CHECK:STDOUT:   %impl.elem0.loc145_43.2 => constants.%impl.elem0.979
// CHECK:STDOUT:   %facet_type.loc145_37 => constants.%facet_type.ccf
// CHECK:STDOUT:   %pattern_type.loc145 => constants.%pattern_type.d9e
// CHECK:STDOUT:   %T.patt.loc145_4.2 => constants.%T.patt.879
// CHECK:STDOUT:   %T.loc145_4.1 => constants.%facet_value.191
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
// CHECK:STDOUT:   %specific_impl_fn.loc148_5.3 => constants.%Int32.Op.4dca31.3
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type => constants.%Destroy.WithSelf.SelfDestruct.type.53c
// CHECK:STDOUT:   %.loc148_5.7 => constants.%.77c
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.1bfa2a.2
// CHECK:STDOUT:   %impl.elem2.loc148_5.2 => constants.%Destroy.WithSelf.SelfDestruct.db3fdb.3
// CHECK:STDOUT:   %specific_impl_fn.loc148_5.4 => constants.%Destroy.WithSelf.SelfDestruct.db3fdb.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @DivWith.loc157(constants.%Destroy.facet.84d, constants.%Destroy.facet.f58, constants.%facet_value.211) {
// CHECK:STDOUT:   %Result.patt.loc154_9.2 => constants.%Result.patt
// CHECK:STDOUT:   %Result.loc154_9.1 => constants.%Destroy.facet.84d
// CHECK:STDOUT:   %U.patt.loc155_4.2 => constants.%U.patt.7a2
// CHECK:STDOUT:   %U.loc155_4.1 => constants.%Destroy.facet.f58
// CHECK:STDOUT:   %U.as_type.loc156_35.1 => constants.%Int16
// CHECK:STDOUT:   %DivWith.type.loc156_35.1 => constants.%DivWith.type.b8a
// CHECK:STDOUT:   %facet_type.loc156_19 => constants.%facet_type.b64
// CHECK:STDOUT:   %.Self.frozen.loc156_37.1 => constants.%.Self.frozen.797
// CHECK:STDOUT:   %require_complete.loc156 => constants.%complete_type.ca5
// CHECK:STDOUT:   %.Self.frozen.as_type => constants.%.Self.frozen.as_type.b76
// CHECK:STDOUT:   %DivWith.assoc_type => constants.%DivWith.assoc_type.712
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.99c
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.loc156_43.1 => constants.%DivWith.lookup_impl_witness.60a
// CHECK:STDOUT:   %impl.elem0.loc156_43.1 => constants.%impl.elem0.431
// CHECK:STDOUT:   %Result.as_type.loc156_53.1 => constants.%Int32
// CHECK:STDOUT:   %.Self => constants.%.Self.315
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.loc156_43.2 => constants.%DivWith.lookup_impl_witness.147
// CHECK:STDOUT:   %impl.elem0.loc156_43.2 => constants.%impl.elem0.17b
// CHECK:STDOUT:   %facet_type.loc156_37 => constants.%facet_type.64f
// CHECK:STDOUT:   %pattern_type.loc156 => constants.%pattern_type.b6c
// CHECK:STDOUT:   %T.patt.loc156_4.2 => constants.%T.patt.731
// CHECK:STDOUT:   %T.loc156_4.1 => constants.%facet_value.211
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
// CHECK:STDOUT:   %specific_impl_fn.loc159_5.3 => constants.%Int32.Op.4dca31.4
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type => constants.%Destroy.WithSelf.SelfDestruct.type.53c
// CHECK:STDOUT:   %.loc159_5.7 => constants.%.77c
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.1bfa2a.2
// CHECK:STDOUT:   %impl.elem2.loc159_5.2 => constants.%Destroy.WithSelf.SelfDestruct.db3fdb.3
// CHECK:STDOUT:   %specific_impl_fn.loc159_5.4 => constants.%Destroy.WithSelf.SelfDestruct.db3fdb.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ModWith.loc168(constants.%Destroy.facet.84d, constants.%Destroy.facet.f58, constants.%facet_value.09f) {
// CHECK:STDOUT:   %Result.patt.loc165_9.2 => constants.%Result.patt
// CHECK:STDOUT:   %Result.loc165_9.1 => constants.%Destroy.facet.84d
// CHECK:STDOUT:   %U.patt.loc166_4.2 => constants.%U.patt.7a2
// CHECK:STDOUT:   %U.loc166_4.1 => constants.%Destroy.facet.f58
// CHECK:STDOUT:   %U.as_type.loc167_35.1 => constants.%Int16
// CHECK:STDOUT:   %ModWith.type.loc167_35.1 => constants.%ModWith.type.c67
// CHECK:STDOUT:   %facet_type.loc167_19 => constants.%facet_type.2c9
// CHECK:STDOUT:   %.Self.frozen.loc167_37.1 => constants.%.Self.frozen.7c6
// CHECK:STDOUT:   %require_complete.loc167 => constants.%complete_type.300
// CHECK:STDOUT:   %.Self.frozen.as_type => constants.%.Self.frozen.as_type.c18
// CHECK:STDOUT:   %ModWith.assoc_type => constants.%ModWith.assoc_type.5f6
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.f04
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.loc167_43.1 => constants.%ModWith.lookup_impl_witness.49e
// CHECK:STDOUT:   %impl.elem0.loc167_43.1 => constants.%impl.elem0.042
// CHECK:STDOUT:   %Result.as_type.loc167_53.1 => constants.%Int32
// CHECK:STDOUT:   %.Self => constants.%.Self.7d4
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.loc167_43.2 => constants.%ModWith.lookup_impl_witness.c30
// CHECK:STDOUT:   %impl.elem0.loc167_43.2 => constants.%impl.elem0.e30
// CHECK:STDOUT:   %facet_type.loc167_37 => constants.%facet_type.b31
// CHECK:STDOUT:   %pattern_type.loc167 => constants.%pattern_type.df6
// CHECK:STDOUT:   %T.patt.loc167_4.2 => constants.%T.patt.336
// CHECK:STDOUT:   %T.loc167_4.1 => constants.%facet_value.09f
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
// CHECK:STDOUT:   %specific_impl_fn.loc170_5.3 => constants.%Int32.Op.4dca31.5
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type => constants.%Destroy.WithSelf.SelfDestruct.type.53c
// CHECK:STDOUT:   %.loc170_5.7 => constants.%.77c
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.1bfa2a.2
// CHECK:STDOUT:   %impl.elem2.loc170_5.2 => constants.%Destroy.WithSelf.SelfDestruct.db3fdb.3
// CHECK:STDOUT:   %specific_impl_fn.loc170_5.4 => constants.%Destroy.WithSelf.SelfDestruct.db3fdb.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AddWith.loc124(constants.%Destroy.facet.84d, constants.%Destroy.facet.84d, constants.%facet_value.5ed) {
// CHECK:STDOUT:   %Result.patt.loc121_9.2 => constants.%Result.patt
// CHECK:STDOUT:   %Result.loc121_9.1 => constants.%Destroy.facet.84d
// CHECK:STDOUT:   %U.patt.loc122_4.2 => constants.%U.patt.7a2
// CHECK:STDOUT:   %U.loc122_4.1 => constants.%Destroy.facet.84d
// CHECK:STDOUT:   %U.as_type.loc123_35.1 => constants.%Int32
// CHECK:STDOUT:   %AddWith.type.loc123_35.1 => constants.%AddWith.type.db7
// CHECK:STDOUT:   %facet_type.loc123_19 => constants.%facet_type.5cd
// CHECK:STDOUT:   %.Self.frozen.loc123_37.1 => constants.%.Self.frozen.52a
// CHECK:STDOUT:   %require_complete.loc123 => constants.%complete_type.18d
// CHECK:STDOUT:   %.Self.frozen.as_type => constants.%.Self.frozen.as_type.988
// CHECK:STDOUT:   %AddWith.assoc_type => constants.%AddWith.assoc_type.be8
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.31c
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.loc123_43.1 => constants.%AddWith.lookup_impl_witness.3ac
// CHECK:STDOUT:   %impl.elem0.loc123_43.1 => constants.%impl.elem0.958
// CHECK:STDOUT:   %Result.as_type.loc123_53.1 => constants.%Int32
// CHECK:STDOUT:   %.Self => constants.%.Self.726
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.loc123_43.2 => constants.%AddWith.lookup_impl_witness.bcf
// CHECK:STDOUT:   %impl.elem0.loc123_43.2 => constants.%impl.elem0.953
// CHECK:STDOUT:   %facet_type.loc123_37 => constants.%facet_type.030
// CHECK:STDOUT:   %pattern_type.loc123 => constants.%pattern_type.c1b
// CHECK:STDOUT:   %T.patt.loc123_4.2 => constants.%T.patt.16ff
// CHECK:STDOUT:   %T.loc123_4.1 => constants.%facet_value.5ed
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
// CHECK:STDOUT:   %specific_impl_fn.loc126_5.3 => constants.%Int32.Op.4dca31.6
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type => constants.%Destroy.WithSelf.SelfDestruct.type.53c
// CHECK:STDOUT:   %.loc126_5.7 => constants.%.77c
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.1bfa2a.2
// CHECK:STDOUT:   %impl.elem2.loc126_5.2 => constants.%Destroy.WithSelf.SelfDestruct.db3fdb.3
// CHECK:STDOUT:   %specific_impl_fn.loc126_5.4 => constants.%Destroy.WithSelf.SelfDestruct.db3fdb.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @SubWith.loc135(constants.%Destroy.facet.84d, constants.%Destroy.facet.84d, constants.%facet_value.889) {
// CHECK:STDOUT:   %Result.patt.loc132_9.2 => constants.%Result.patt
// CHECK:STDOUT:   %Result.loc132_9.1 => constants.%Destroy.facet.84d
// CHECK:STDOUT:   %U.patt.loc133_4.2 => constants.%U.patt.7a2
// CHECK:STDOUT:   %U.loc133_4.1 => constants.%Destroy.facet.84d
// CHECK:STDOUT:   %U.as_type.loc134_35.1 => constants.%Int32
// CHECK:STDOUT:   %SubWith.type.loc134_35.1 => constants.%SubWith.type.120
// CHECK:STDOUT:   %facet_type.loc134_19 => constants.%facet_type.393
// CHECK:STDOUT:   %.Self.frozen.loc134_37.1 => constants.%.Self.frozen.36d
// CHECK:STDOUT:   %require_complete.loc134 => constants.%complete_type.8a6
// CHECK:STDOUT:   %.Self.frozen.as_type => constants.%.Self.frozen.as_type.88b
// CHECK:STDOUT:   %SubWith.assoc_type => constants.%SubWith.assoc_type.dae
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.fac
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.loc134_43.1 => constants.%SubWith.lookup_impl_witness.10e
// CHECK:STDOUT:   %impl.elem0.loc134_43.1 => constants.%impl.elem0.d88
// CHECK:STDOUT:   %Result.as_type.loc134_53.1 => constants.%Int32
// CHECK:STDOUT:   %.Self => constants.%.Self.eac
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.loc134_43.2 => constants.%SubWith.lookup_impl_witness.0dc
// CHECK:STDOUT:   %impl.elem0.loc134_43.2 => constants.%impl.elem0.bb5a
// CHECK:STDOUT:   %facet_type.loc134_37 => constants.%facet_type.767
// CHECK:STDOUT:   %pattern_type.loc134 => constants.%pattern_type.974
// CHECK:STDOUT:   %T.patt.loc134_4.2 => constants.%T.patt.3c7
// CHECK:STDOUT:   %T.loc134_4.1 => constants.%facet_value.889
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
// CHECK:STDOUT:   %specific_impl_fn.loc137_5.3 => constants.%Int32.Op.4dca31.7
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type => constants.%Destroy.WithSelf.SelfDestruct.type.53c
// CHECK:STDOUT:   %.loc137_5.7 => constants.%.77c
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.1bfa2a.2
// CHECK:STDOUT:   %impl.elem2.loc137_5.2 => constants.%Destroy.WithSelf.SelfDestruct.db3fdb.3
// CHECK:STDOUT:   %specific_impl_fn.loc137_5.4 => constants.%Destroy.WithSelf.SelfDestruct.db3fdb.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @MulWith.loc146(constants.%Destroy.facet.84d, constants.%Destroy.facet.84d, constants.%facet_value.f7d) {
// CHECK:STDOUT:   %Result.patt.loc143_9.2 => constants.%Result.patt
// CHECK:STDOUT:   %Result.loc143_9.1 => constants.%Destroy.facet.84d
// CHECK:STDOUT:   %U.patt.loc144_4.2 => constants.%U.patt.7a2
// CHECK:STDOUT:   %U.loc144_4.1 => constants.%Destroy.facet.84d
// CHECK:STDOUT:   %U.as_type.loc145_35.1 => constants.%Int32
// CHECK:STDOUT:   %MulWith.type.loc145_35.1 => constants.%MulWith.type.d67
// CHECK:STDOUT:   %facet_type.loc145_19 => constants.%facet_type.ea6
// CHECK:STDOUT:   %.Self.frozen.loc145_37.1 => constants.%.Self.frozen.95b
// CHECK:STDOUT:   %require_complete.loc145 => constants.%complete_type.8fe
// CHECK:STDOUT:   %.Self.frozen.as_type => constants.%.Self.frozen.as_type.f47
// CHECK:STDOUT:   %MulWith.assoc_type => constants.%MulWith.assoc_type.ab1
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.d0a
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.loc145_43.1 => constants.%MulWith.lookup_impl_witness.675
// CHECK:STDOUT:   %impl.elem0.loc145_43.1 => constants.%impl.elem0.1de
// CHECK:STDOUT:   %Result.as_type.loc145_53.1 => constants.%Int32
// CHECK:STDOUT:   %.Self => constants.%.Self.44f
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.loc145_43.2 => constants.%MulWith.lookup_impl_witness.527
// CHECK:STDOUT:   %impl.elem0.loc145_43.2 => constants.%impl.elem0.054
// CHECK:STDOUT:   %facet_type.loc145_37 => constants.%facet_type.9d8
// CHECK:STDOUT:   %pattern_type.loc145 => constants.%pattern_type.a89
// CHECK:STDOUT:   %T.patt.loc145_4.2 => constants.%T.patt.144
// CHECK:STDOUT:   %T.loc145_4.1 => constants.%facet_value.f7d
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
// CHECK:STDOUT:   %specific_impl_fn.loc148_5.3 => constants.%Int32.Op.4dca31.8
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type => constants.%Destroy.WithSelf.SelfDestruct.type.53c
// CHECK:STDOUT:   %.loc148_5.7 => constants.%.77c
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.1bfa2a.2
// CHECK:STDOUT:   %impl.elem2.loc148_5.2 => constants.%Destroy.WithSelf.SelfDestruct.db3fdb.3
// CHECK:STDOUT:   %specific_impl_fn.loc148_5.4 => constants.%Destroy.WithSelf.SelfDestruct.db3fdb.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @DivWith.loc157(constants.%Destroy.facet.84d, constants.%Destroy.facet.84d, constants.%facet_value.403) {
// CHECK:STDOUT:   %Result.patt.loc154_9.2 => constants.%Result.patt
// CHECK:STDOUT:   %Result.loc154_9.1 => constants.%Destroy.facet.84d
// CHECK:STDOUT:   %U.patt.loc155_4.2 => constants.%U.patt.7a2
// CHECK:STDOUT:   %U.loc155_4.1 => constants.%Destroy.facet.84d
// CHECK:STDOUT:   %U.as_type.loc156_35.1 => constants.%Int32
// CHECK:STDOUT:   %DivWith.type.loc156_35.1 => constants.%DivWith.type.5a1
// CHECK:STDOUT:   %facet_type.loc156_19 => constants.%facet_type.821
// CHECK:STDOUT:   %.Self.frozen.loc156_37.1 => constants.%.Self.frozen.3ed
// CHECK:STDOUT:   %require_complete.loc156 => constants.%complete_type.5bd
// CHECK:STDOUT:   %.Self.frozen.as_type => constants.%.Self.frozen.as_type.553
// CHECK:STDOUT:   %DivWith.assoc_type => constants.%DivWith.assoc_type.22f
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.a9d
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.loc156_43.1 => constants.%DivWith.lookup_impl_witness.1af
// CHECK:STDOUT:   %impl.elem0.loc156_43.1 => constants.%impl.elem0.a46
// CHECK:STDOUT:   %Result.as_type.loc156_53.1 => constants.%Int32
// CHECK:STDOUT:   %.Self => constants.%.Self.37d
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.loc156_43.2 => constants.%DivWith.lookup_impl_witness.7ae
// CHECK:STDOUT:   %impl.elem0.loc156_43.2 => constants.%impl.elem0.6c4
// CHECK:STDOUT:   %facet_type.loc156_37 => constants.%facet_type.f48
// CHECK:STDOUT:   %pattern_type.loc156 => constants.%pattern_type.4e6
// CHECK:STDOUT:   %T.patt.loc156_4.2 => constants.%T.patt.def
// CHECK:STDOUT:   %T.loc156_4.1 => constants.%facet_value.403
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
// CHECK:STDOUT:   %specific_impl_fn.loc159_5.3 => constants.%Int32.Op.4dca31.9
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type => constants.%Destroy.WithSelf.SelfDestruct.type.53c
// CHECK:STDOUT:   %.loc159_5.7 => constants.%.77c
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.1bfa2a.2
// CHECK:STDOUT:   %impl.elem2.loc159_5.2 => constants.%Destroy.WithSelf.SelfDestruct.db3fdb.3
// CHECK:STDOUT:   %specific_impl_fn.loc159_5.4 => constants.%Destroy.WithSelf.SelfDestruct.db3fdb.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ModWith.loc168(constants.%Destroy.facet.84d, constants.%Destroy.facet.84d, constants.%facet_value.559) {
// CHECK:STDOUT:   %Result.patt.loc165_9.2 => constants.%Result.patt
// CHECK:STDOUT:   %Result.loc165_9.1 => constants.%Destroy.facet.84d
// CHECK:STDOUT:   %U.patt.loc166_4.2 => constants.%U.patt.7a2
// CHECK:STDOUT:   %U.loc166_4.1 => constants.%Destroy.facet.84d
// CHECK:STDOUT:   %U.as_type.loc167_35.1 => constants.%Int32
// CHECK:STDOUT:   %ModWith.type.loc167_35.1 => constants.%ModWith.type.794
// CHECK:STDOUT:   %facet_type.loc167_19 => constants.%facet_type.9c3
// CHECK:STDOUT:   %.Self.frozen.loc167_37.1 => constants.%.Self.frozen.b381
// CHECK:STDOUT:   %require_complete.loc167 => constants.%complete_type.0b7
// CHECK:STDOUT:   %.Self.frozen.as_type => constants.%.Self.frozen.as_type.f01
// CHECK:STDOUT:   %ModWith.assoc_type => constants.%ModWith.assoc_type.021
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.310
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.loc167_43.1 => constants.%ModWith.lookup_impl_witness.ac5
// CHECK:STDOUT:   %impl.elem0.loc167_43.1 => constants.%impl.elem0.914
// CHECK:STDOUT:   %Result.as_type.loc167_53.1 => constants.%Int32
// CHECK:STDOUT:   %.Self => constants.%.Self.f60
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.loc167_43.2 => constants.%ModWith.lookup_impl_witness.9a9
// CHECK:STDOUT:   %impl.elem0.loc167_43.2 => constants.%impl.elem0.522
// CHECK:STDOUT:   %facet_type.loc167_37 => constants.%facet_type.115
// CHECK:STDOUT:   %pattern_type.loc167 => constants.%pattern_type.393
// CHECK:STDOUT:   %T.patt.loc167_4.2 => constants.%T.patt.be9
// CHECK:STDOUT:   %T.loc167_4.1 => constants.%facet_value.559
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
// CHECK:STDOUT:   %specific_impl_fn.loc170_5.3 => constants.%Int32.Op.4dca31.10
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type => constants.%Destroy.WithSelf.SelfDestruct.type.53c
// CHECK:STDOUT:   %.loc170_5.7 => constants.%.77c
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.1bfa2a.2
// CHECK:STDOUT:   %impl.elem2.loc170_5.2 => constants.%Destroy.WithSelf.SelfDestruct.db3fdb.3
// CHECK:STDOUT:   %specific_impl_fn.loc170_5.4 => constants.%Destroy.WithSelf.SelfDestruct.db3fdb.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AddWith.loc124(constants.%Destroy.facet.0b4, constants.%Destroy.facet.0b4, constants.%facet_value.806) {
// CHECK:STDOUT:   %Result.patt.loc121_9.2 => constants.%Result.patt
// CHECK:STDOUT:   %Result.loc121_9.1 => constants.%Destroy.facet.0b4
// CHECK:STDOUT:   %U.patt.loc122_4.2 => constants.%U.patt.7a2
// CHECK:STDOUT:   %U.loc122_4.1 => constants.%Destroy.facet.0b4
// CHECK:STDOUT:   %U.as_type.loc123_35.1 => constants.%Int64
// CHECK:STDOUT:   %AddWith.type.loc123_35.1 => constants.%AddWith.type.1ff
// CHECK:STDOUT:   %facet_type.loc123_19 => constants.%facet_type.e1d
// CHECK:STDOUT:   %.Self.frozen.loc123_37.1 => constants.%.Self.frozen.d58
// CHECK:STDOUT:   %require_complete.loc123 => constants.%complete_type.545
// CHECK:STDOUT:   %.Self.frozen.as_type => constants.%.Self.frozen.as_type.f6a
// CHECK:STDOUT:   %AddWith.assoc_type => constants.%AddWith.assoc_type.97f
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.f02
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.loc123_43.1 => constants.%AddWith.lookup_impl_witness.bc0
// CHECK:STDOUT:   %impl.elem0.loc123_43.1 => constants.%impl.elem0.1e6
// CHECK:STDOUT:   %Result.as_type.loc123_53.1 => constants.%Int64
// CHECK:STDOUT:   %.Self => constants.%.Self.ad4
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.loc123_43.2 => constants.%AddWith.lookup_impl_witness.a60
// CHECK:STDOUT:   %impl.elem0.loc123_43.2 => constants.%impl.elem0.7fa
// CHECK:STDOUT:   %facet_type.loc123_37 => constants.%facet_type.37a
// CHECK:STDOUT:   %pattern_type.loc123 => constants.%pattern_type.ec3
// CHECK:STDOUT:   %T.patt.loc123_4.2 => constants.%T.patt.af0
// CHECK:STDOUT:   %T.loc123_4.1 => constants.%facet_value.806
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
// CHECK:STDOUT:   %specific_impl_fn.loc126_5.3 => constants.%Op.72717c.6
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type => constants.%Destroy.WithSelf.SelfDestruct.type.36f
// CHECK:STDOUT:   %.loc126_5.7 => constants.%.7aa
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.1bfa2a.3
// CHECK:STDOUT:   %impl.elem2.loc126_5.2 => constants.%Destroy.WithSelf.SelfDestruct.db3fdb.4
// CHECK:STDOUT:   %specific_impl_fn.loc126_5.4 => constants.%Destroy.WithSelf.SelfDestruct.db3fdb.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @SubWith.loc135(constants.%Destroy.facet.0b4, constants.%Destroy.facet.0b4, constants.%facet_value.442) {
// CHECK:STDOUT:   %Result.patt.loc132_9.2 => constants.%Result.patt
// CHECK:STDOUT:   %Result.loc132_9.1 => constants.%Destroy.facet.0b4
// CHECK:STDOUT:   %U.patt.loc133_4.2 => constants.%U.patt.7a2
// CHECK:STDOUT:   %U.loc133_4.1 => constants.%Destroy.facet.0b4
// CHECK:STDOUT:   %U.as_type.loc134_35.1 => constants.%Int64
// CHECK:STDOUT:   %SubWith.type.loc134_35.1 => constants.%SubWith.type.31c
// CHECK:STDOUT:   %facet_type.loc134_19 => constants.%facet_type.6db
// CHECK:STDOUT:   %.Self.frozen.loc134_37.1 => constants.%.Self.frozen.a3d
// CHECK:STDOUT:   %require_complete.loc134 => constants.%complete_type.b70
// CHECK:STDOUT:   %.Self.frozen.as_type => constants.%.Self.frozen.as_type.f8c
// CHECK:STDOUT:   %SubWith.assoc_type => constants.%SubWith.assoc_type.4f8
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.13b
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.loc134_43.1 => constants.%SubWith.lookup_impl_witness.117
// CHECK:STDOUT:   %impl.elem0.loc134_43.1 => constants.%impl.elem0.79c
// CHECK:STDOUT:   %Result.as_type.loc134_53.1 => constants.%Int64
// CHECK:STDOUT:   %.Self => constants.%.Self.b71
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.loc134_43.2 => constants.%SubWith.lookup_impl_witness.4b8
// CHECK:STDOUT:   %impl.elem0.loc134_43.2 => constants.%impl.elem0.5e41
// CHECK:STDOUT:   %facet_type.loc134_37 => constants.%facet_type.cfa
// CHECK:STDOUT:   %pattern_type.loc134 => constants.%pattern_type.16a
// CHECK:STDOUT:   %T.patt.loc134_4.2 => constants.%T.patt.18c
// CHECK:STDOUT:   %T.loc134_4.1 => constants.%facet_value.442
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
// CHECK:STDOUT:   %specific_impl_fn.loc137_5.3 => constants.%Op.72717c.7
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type => constants.%Destroy.WithSelf.SelfDestruct.type.36f
// CHECK:STDOUT:   %.loc137_5.7 => constants.%.7aa
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.1bfa2a.3
// CHECK:STDOUT:   %impl.elem2.loc137_5.2 => constants.%Destroy.WithSelf.SelfDestruct.db3fdb.4
// CHECK:STDOUT:   %specific_impl_fn.loc137_5.4 => constants.%Destroy.WithSelf.SelfDestruct.db3fdb.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @MulWith.loc146(constants.%Destroy.facet.0b4, constants.%Destroy.facet.0b4, constants.%facet_value.97e) {
// CHECK:STDOUT:   %Result.patt.loc143_9.2 => constants.%Result.patt
// CHECK:STDOUT:   %Result.loc143_9.1 => constants.%Destroy.facet.0b4
// CHECK:STDOUT:   %U.patt.loc144_4.2 => constants.%U.patt.7a2
// CHECK:STDOUT:   %U.loc144_4.1 => constants.%Destroy.facet.0b4
// CHECK:STDOUT:   %U.as_type.loc145_35.1 => constants.%Int64
// CHECK:STDOUT:   %MulWith.type.loc145_35.1 => constants.%MulWith.type.661
// CHECK:STDOUT:   %facet_type.loc145_19 => constants.%facet_type.dc5
// CHECK:STDOUT:   %.Self.frozen.loc145_37.1 => constants.%.Self.frozen.1e0
// CHECK:STDOUT:   %require_complete.loc145 => constants.%complete_type.757
// CHECK:STDOUT:   %.Self.frozen.as_type => constants.%.Self.frozen.as_type.5a6
// CHECK:STDOUT:   %MulWith.assoc_type => constants.%MulWith.assoc_type.940
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.9d9
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.loc145_43.1 => constants.%MulWith.lookup_impl_witness.741
// CHECK:STDOUT:   %impl.elem0.loc145_43.1 => constants.%impl.elem0.dca
// CHECK:STDOUT:   %Result.as_type.loc145_53.1 => constants.%Int64
// CHECK:STDOUT:   %.Self => constants.%.Self.6e4
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.loc145_43.2 => constants.%MulWith.lookup_impl_witness.977
// CHECK:STDOUT:   %impl.elem0.loc145_43.2 => constants.%impl.elem0.a52
// CHECK:STDOUT:   %facet_type.loc145_37 => constants.%facet_type.48a
// CHECK:STDOUT:   %pattern_type.loc145 => constants.%pattern_type.307
// CHECK:STDOUT:   %T.patt.loc145_4.2 => constants.%T.patt.23a
// CHECK:STDOUT:   %T.loc145_4.1 => constants.%facet_value.97e
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
// CHECK:STDOUT:   %specific_impl_fn.loc148_5.3 => constants.%Op.72717c.8
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type => constants.%Destroy.WithSelf.SelfDestruct.type.36f
// CHECK:STDOUT:   %.loc148_5.7 => constants.%.7aa
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.1bfa2a.3
// CHECK:STDOUT:   %impl.elem2.loc148_5.2 => constants.%Destroy.WithSelf.SelfDestruct.db3fdb.4
// CHECK:STDOUT:   %specific_impl_fn.loc148_5.4 => constants.%Destroy.WithSelf.SelfDestruct.db3fdb.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @DivWith.loc157(constants.%Destroy.facet.0b4, constants.%Destroy.facet.0b4, constants.%facet_value.585) {
// CHECK:STDOUT:   %Result.patt.loc154_9.2 => constants.%Result.patt
// CHECK:STDOUT:   %Result.loc154_9.1 => constants.%Destroy.facet.0b4
// CHECK:STDOUT:   %U.patt.loc155_4.2 => constants.%U.patt.7a2
// CHECK:STDOUT:   %U.loc155_4.1 => constants.%Destroy.facet.0b4
// CHECK:STDOUT:   %U.as_type.loc156_35.1 => constants.%Int64
// CHECK:STDOUT:   %DivWith.type.loc156_35.1 => constants.%DivWith.type.dd8
// CHECK:STDOUT:   %facet_type.loc156_19 => constants.%facet_type.7ad
// CHECK:STDOUT:   %.Self.frozen.loc156_37.1 => constants.%.Self.frozen.2167
// CHECK:STDOUT:   %require_complete.loc156 => constants.%complete_type.c23
// CHECK:STDOUT:   %.Self.frozen.as_type => constants.%.Self.frozen.as_type.c77
// CHECK:STDOUT:   %DivWith.assoc_type => constants.%DivWith.assoc_type.2dc
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.e8b
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.loc156_43.1 => constants.%DivWith.lookup_impl_witness.958
// CHECK:STDOUT:   %impl.elem0.loc156_43.1 => constants.%impl.elem0.a26
// CHECK:STDOUT:   %Result.as_type.loc156_53.1 => constants.%Int64
// CHECK:STDOUT:   %.Self => constants.%.Self.7b8
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.loc156_43.2 => constants.%DivWith.lookup_impl_witness.055
// CHECK:STDOUT:   %impl.elem0.loc156_43.2 => constants.%impl.elem0.212
// CHECK:STDOUT:   %facet_type.loc156_37 => constants.%facet_type.011
// CHECK:STDOUT:   %pattern_type.loc156 => constants.%pattern_type.5eb
// CHECK:STDOUT:   %T.patt.loc156_4.2 => constants.%T.patt.518
// CHECK:STDOUT:   %T.loc156_4.1 => constants.%facet_value.585
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
// CHECK:STDOUT:   %specific_impl_fn.loc159_5.3 => constants.%Op.72717c.9
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type => constants.%Destroy.WithSelf.SelfDestruct.type.36f
// CHECK:STDOUT:   %.loc159_5.7 => constants.%.7aa
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.1bfa2a.3
// CHECK:STDOUT:   %impl.elem2.loc159_5.2 => constants.%Destroy.WithSelf.SelfDestruct.db3fdb.4
// CHECK:STDOUT:   %specific_impl_fn.loc159_5.4 => constants.%Destroy.WithSelf.SelfDestruct.db3fdb.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ModWith.loc168(constants.%Destroy.facet.0b4, constants.%Destroy.facet.0b4, constants.%facet_value.b2a) {
// CHECK:STDOUT:   %Result.patt.loc165_9.2 => constants.%Result.patt
// CHECK:STDOUT:   %Result.loc165_9.1 => constants.%Destroy.facet.0b4
// CHECK:STDOUT:   %U.patt.loc166_4.2 => constants.%U.patt.7a2
// CHECK:STDOUT:   %U.loc166_4.1 => constants.%Destroy.facet.0b4
// CHECK:STDOUT:   %U.as_type.loc167_35.1 => constants.%Int64
// CHECK:STDOUT:   %ModWith.type.loc167_35.1 => constants.%ModWith.type.c72
// CHECK:STDOUT:   %facet_type.loc167_19 => constants.%facet_type.830
// CHECK:STDOUT:   %.Self.frozen.loc167_37.1 => constants.%.Self.frozen.5a8
// CHECK:STDOUT:   %require_complete.loc167 => constants.%complete_type.ecd
// CHECK:STDOUT:   %.Self.frozen.as_type => constants.%.Self.frozen.as_type.7bd
// CHECK:STDOUT:   %ModWith.assoc_type => constants.%ModWith.assoc_type.ad2
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.c8a
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.loc167_43.1 => constants.%ModWith.lookup_impl_witness.601
// CHECK:STDOUT:   %impl.elem0.loc167_43.1 => constants.%impl.elem0.001
// CHECK:STDOUT:   %Result.as_type.loc167_53.1 => constants.%Int64
// CHECK:STDOUT:   %.Self => constants.%.Self.23a
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.loc167_43.2 => constants.%ModWith.lookup_impl_witness.4a5
// CHECK:STDOUT:   %impl.elem0.loc167_43.2 => constants.%impl.elem0.007
// CHECK:STDOUT:   %facet_type.loc167_37 => constants.%facet_type.2db
// CHECK:STDOUT:   %pattern_type.loc167 => constants.%pattern_type.3f3
// CHECK:STDOUT:   %T.patt.loc167_4.2 => constants.%T.patt.3de
// CHECK:STDOUT:   %T.loc167_4.1 => constants.%facet_value.b2a
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
// CHECK:STDOUT:   %specific_impl_fn.loc170_5.3 => constants.%Op.72717c.10
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type => constants.%Destroy.WithSelf.SelfDestruct.type.36f
// CHECK:STDOUT:   %.loc170_5.7 => constants.%.7aa
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.1bfa2a.3
// CHECK:STDOUT:   %impl.elem2.loc170_5.2 => constants.%Destroy.WithSelf.SelfDestruct.db3fdb.4
// CHECK:STDOUT:   %specific_impl_fn.loc170_5.4 => constants.%Destroy.WithSelf.SelfDestruct.db3fdb.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AddWith.loc124(constants.%Destroy.facet.0b4, constants.%Destroy.facet.84d, constants.%facet_value.26c) {
// CHECK:STDOUT:   %Result.patt.loc121_9.2 => constants.%Result.patt
// CHECK:STDOUT:   %Result.loc121_9.1 => constants.%Destroy.facet.0b4
// CHECK:STDOUT:   %U.patt.loc122_4.2 => constants.%U.patt.7a2
// CHECK:STDOUT:   %U.loc122_4.1 => constants.%Destroy.facet.84d
// CHECK:STDOUT:   %U.as_type.loc123_35.1 => constants.%Int32
// CHECK:STDOUT:   %AddWith.type.loc123_35.1 => constants.%AddWith.type.db7
// CHECK:STDOUT:   %facet_type.loc123_19 => constants.%facet_type.5cd
// CHECK:STDOUT:   %.Self.frozen.loc123_37.1 => constants.%.Self.frozen.52a
// CHECK:STDOUT:   %require_complete.loc123 => constants.%complete_type.18d
// CHECK:STDOUT:   %.Self.frozen.as_type => constants.%.Self.frozen.as_type.988
// CHECK:STDOUT:   %AddWith.assoc_type => constants.%AddWith.assoc_type.be8
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.31c
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.loc123_43.1 => constants.%AddWith.lookup_impl_witness.3ac
// CHECK:STDOUT:   %impl.elem0.loc123_43.1 => constants.%impl.elem0.958
// CHECK:STDOUT:   %Result.as_type.loc123_53.1 => constants.%Int64
// CHECK:STDOUT:   %.Self => constants.%.Self.726
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.loc123_43.2 => constants.%AddWith.lookup_impl_witness.bcf
// CHECK:STDOUT:   %impl.elem0.loc123_43.2 => constants.%impl.elem0.953
// CHECK:STDOUT:   %facet_type.loc123_37 => constants.%facet_type.223
// CHECK:STDOUT:   %pattern_type.loc123 => constants.%pattern_type.2c7
// CHECK:STDOUT:   %T.patt.loc123_4.2 => constants.%T.patt.e6f
// CHECK:STDOUT:   %T.loc123_4.1 => constants.%facet_value.26c
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
// CHECK:STDOUT:   %specific_impl_fn.loc126_5.3 => constants.%Op.72717c.11
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type => constants.%Destroy.WithSelf.SelfDestruct.type.36f
// CHECK:STDOUT:   %.loc126_5.7 => constants.%.7aa
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.1bfa2a.3
// CHECK:STDOUT:   %impl.elem2.loc126_5.2 => constants.%Destroy.WithSelf.SelfDestruct.db3fdb.4
// CHECK:STDOUT:   %specific_impl_fn.loc126_5.4 => constants.%Destroy.WithSelf.SelfDestruct.db3fdb.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @SubWith.loc135(constants.%Destroy.facet.0b4, constants.%Destroy.facet.84d, constants.%facet_value.d8b) {
// CHECK:STDOUT:   %Result.patt.loc132_9.2 => constants.%Result.patt
// CHECK:STDOUT:   %Result.loc132_9.1 => constants.%Destroy.facet.0b4
// CHECK:STDOUT:   %U.patt.loc133_4.2 => constants.%U.patt.7a2
// CHECK:STDOUT:   %U.loc133_4.1 => constants.%Destroy.facet.84d
// CHECK:STDOUT:   %U.as_type.loc134_35.1 => constants.%Int32
// CHECK:STDOUT:   %SubWith.type.loc134_35.1 => constants.%SubWith.type.120
// CHECK:STDOUT:   %facet_type.loc134_19 => constants.%facet_type.393
// CHECK:STDOUT:   %.Self.frozen.loc134_37.1 => constants.%.Self.frozen.36d
// CHECK:STDOUT:   %require_complete.loc134 => constants.%complete_type.8a6
// CHECK:STDOUT:   %.Self.frozen.as_type => constants.%.Self.frozen.as_type.88b
// CHECK:STDOUT:   %SubWith.assoc_type => constants.%SubWith.assoc_type.dae
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.fac
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.loc134_43.1 => constants.%SubWith.lookup_impl_witness.10e
// CHECK:STDOUT:   %impl.elem0.loc134_43.1 => constants.%impl.elem0.d88
// CHECK:STDOUT:   %Result.as_type.loc134_53.1 => constants.%Int64
// CHECK:STDOUT:   %.Self => constants.%.Self.eac
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.loc134_43.2 => constants.%SubWith.lookup_impl_witness.0dc
// CHECK:STDOUT:   %impl.elem0.loc134_43.2 => constants.%impl.elem0.bb5a
// CHECK:STDOUT:   %facet_type.loc134_37 => constants.%facet_type.ecd
// CHECK:STDOUT:   %pattern_type.loc134 => constants.%pattern_type.ab2
// CHECK:STDOUT:   %T.patt.loc134_4.2 => constants.%T.patt.7e5
// CHECK:STDOUT:   %T.loc134_4.1 => constants.%facet_value.d8b
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
// CHECK:STDOUT:   %SubWith.WithSelf.Op.type => constants.%SubWith.WithSelf.Op.type.2cf
// CHECK:STDOUT:   %.loc137_5.6 => constants.%.4e86
// CHECK:STDOUT:   %impl.elem1.loc137_5.2 => constants.%Op.72717c.12
// CHECK:STDOUT:   %specific_impl_fn.loc137_5.3 => constants.%Op.72717c.12
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type => constants.%Destroy.WithSelf.SelfDestruct.type.36f
// CHECK:STDOUT:   %.loc137_5.7 => constants.%.7aa
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.1bfa2a.3
// CHECK:STDOUT:   %impl.elem2.loc137_5.2 => constants.%Destroy.WithSelf.SelfDestruct.db3fdb.4
// CHECK:STDOUT:   %specific_impl_fn.loc137_5.4 => constants.%Destroy.WithSelf.SelfDestruct.db3fdb.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @MulWith.loc146(constants.%Destroy.facet.0b4, constants.%Destroy.facet.84d, constants.%facet_value.72f) {
// CHECK:STDOUT:   %Result.patt.loc143_9.2 => constants.%Result.patt
// CHECK:STDOUT:   %Result.loc143_9.1 => constants.%Destroy.facet.0b4
// CHECK:STDOUT:   %U.patt.loc144_4.2 => constants.%U.patt.7a2
// CHECK:STDOUT:   %U.loc144_4.1 => constants.%Destroy.facet.84d
// CHECK:STDOUT:   %U.as_type.loc145_35.1 => constants.%Int32
// CHECK:STDOUT:   %MulWith.type.loc145_35.1 => constants.%MulWith.type.d67
// CHECK:STDOUT:   %facet_type.loc145_19 => constants.%facet_type.ea6
// CHECK:STDOUT:   %.Self.frozen.loc145_37.1 => constants.%.Self.frozen.95b
// CHECK:STDOUT:   %require_complete.loc145 => constants.%complete_type.8fe
// CHECK:STDOUT:   %.Self.frozen.as_type => constants.%.Self.frozen.as_type.f47
// CHECK:STDOUT:   %MulWith.assoc_type => constants.%MulWith.assoc_type.ab1
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.d0a
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.loc145_43.1 => constants.%MulWith.lookup_impl_witness.675
// CHECK:STDOUT:   %impl.elem0.loc145_43.1 => constants.%impl.elem0.1de
// CHECK:STDOUT:   %Result.as_type.loc145_53.1 => constants.%Int64
// CHECK:STDOUT:   %.Self => constants.%.Self.44f
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.loc145_43.2 => constants.%MulWith.lookup_impl_witness.527
// CHECK:STDOUT:   %impl.elem0.loc145_43.2 => constants.%impl.elem0.054
// CHECK:STDOUT:   %facet_type.loc145_37 => constants.%facet_type.486
// CHECK:STDOUT:   %pattern_type.loc145 => constants.%pattern_type.92d
// CHECK:STDOUT:   %T.patt.loc145_4.2 => constants.%T.patt.2f0
// CHECK:STDOUT:   %T.loc145_4.1 => constants.%facet_value.72f
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
// CHECK:STDOUT:   %specific_impl_fn.loc148_5.3 => constants.%Op.72717c.13
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type => constants.%Destroy.WithSelf.SelfDestruct.type.36f
// CHECK:STDOUT:   %.loc148_5.7 => constants.%.7aa
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.1bfa2a.3
// CHECK:STDOUT:   %impl.elem2.loc148_5.2 => constants.%Destroy.WithSelf.SelfDestruct.db3fdb.4
// CHECK:STDOUT:   %specific_impl_fn.loc148_5.4 => constants.%Destroy.WithSelf.SelfDestruct.db3fdb.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @DivWith.loc157(constants.%Destroy.facet.0b4, constants.%Destroy.facet.84d, constants.%facet_value.c74) {
// CHECK:STDOUT:   %Result.patt.loc154_9.2 => constants.%Result.patt
// CHECK:STDOUT:   %Result.loc154_9.1 => constants.%Destroy.facet.0b4
// CHECK:STDOUT:   %U.patt.loc155_4.2 => constants.%U.patt.7a2
// CHECK:STDOUT:   %U.loc155_4.1 => constants.%Destroy.facet.84d
// CHECK:STDOUT:   %U.as_type.loc156_35.1 => constants.%Int32
// CHECK:STDOUT:   %DivWith.type.loc156_35.1 => constants.%DivWith.type.5a1
// CHECK:STDOUT:   %facet_type.loc156_19 => constants.%facet_type.821
// CHECK:STDOUT:   %.Self.frozen.loc156_37.1 => constants.%.Self.frozen.3ed
// CHECK:STDOUT:   %require_complete.loc156 => constants.%complete_type.5bd
// CHECK:STDOUT:   %.Self.frozen.as_type => constants.%.Self.frozen.as_type.553
// CHECK:STDOUT:   %DivWith.assoc_type => constants.%DivWith.assoc_type.22f
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.a9d
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.loc156_43.1 => constants.%DivWith.lookup_impl_witness.1af
// CHECK:STDOUT:   %impl.elem0.loc156_43.1 => constants.%impl.elem0.a46
// CHECK:STDOUT:   %Result.as_type.loc156_53.1 => constants.%Int64
// CHECK:STDOUT:   %.Self => constants.%.Self.37d
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.loc156_43.2 => constants.%DivWith.lookup_impl_witness.7ae
// CHECK:STDOUT:   %impl.elem0.loc156_43.2 => constants.%impl.elem0.6c4
// CHECK:STDOUT:   %facet_type.loc156_37 => constants.%facet_type.752
// CHECK:STDOUT:   %pattern_type.loc156 => constants.%pattern_type.d51
// CHECK:STDOUT:   %T.patt.loc156_4.2 => constants.%T.patt.c8a
// CHECK:STDOUT:   %T.loc156_4.1 => constants.%facet_value.c74
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
// CHECK:STDOUT:   %specific_impl_fn.loc159_5.3 => constants.%Op.72717c.14
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type => constants.%Destroy.WithSelf.SelfDestruct.type.36f
// CHECK:STDOUT:   %.loc159_5.7 => constants.%.7aa
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.1bfa2a.3
// CHECK:STDOUT:   %impl.elem2.loc159_5.2 => constants.%Destroy.WithSelf.SelfDestruct.db3fdb.4
// CHECK:STDOUT:   %specific_impl_fn.loc159_5.4 => constants.%Destroy.WithSelf.SelfDestruct.db3fdb.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ModWith.loc168(constants.%Destroy.facet.0b4, constants.%Destroy.facet.84d, constants.%facet_value.b3f) {
// CHECK:STDOUT:   %Result.patt.loc165_9.2 => constants.%Result.patt
// CHECK:STDOUT:   %Result.loc165_9.1 => constants.%Destroy.facet.0b4
// CHECK:STDOUT:   %U.patt.loc166_4.2 => constants.%U.patt.7a2
// CHECK:STDOUT:   %U.loc166_4.1 => constants.%Destroy.facet.84d
// CHECK:STDOUT:   %U.as_type.loc167_35.1 => constants.%Int32
// CHECK:STDOUT:   %ModWith.type.loc167_35.1 => constants.%ModWith.type.794
// CHECK:STDOUT:   %facet_type.loc167_19 => constants.%facet_type.9c3
// CHECK:STDOUT:   %.Self.frozen.loc167_37.1 => constants.%.Self.frozen.b381
// CHECK:STDOUT:   %require_complete.loc167 => constants.%complete_type.0b7
// CHECK:STDOUT:   %.Self.frozen.as_type => constants.%.Self.frozen.as_type.f01
// CHECK:STDOUT:   %ModWith.assoc_type => constants.%ModWith.assoc_type.021
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.310
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.loc167_43.1 => constants.%ModWith.lookup_impl_witness.ac5
// CHECK:STDOUT:   %impl.elem0.loc167_43.1 => constants.%impl.elem0.914
// CHECK:STDOUT:   %Result.as_type.loc167_53.1 => constants.%Int64
// CHECK:STDOUT:   %.Self => constants.%.Self.f60
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.loc167_43.2 => constants.%ModWith.lookup_impl_witness.9a9
// CHECK:STDOUT:   %impl.elem0.loc167_43.2 => constants.%impl.elem0.522
// CHECK:STDOUT:   %facet_type.loc167_37 => constants.%facet_type.088
// CHECK:STDOUT:   %pattern_type.loc167 => constants.%pattern_type.fff
// CHECK:STDOUT:   %T.patt.loc167_4.2 => constants.%T.patt.1fc
// CHECK:STDOUT:   %T.loc167_4.1 => constants.%facet_value.b3f
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
// CHECK:STDOUT:   %ModWith.facet => constants.%ModWith.facet.f5a
// CHECK:STDOUT:   %ModWith.WithSelf.Op.type => constants.%ModWith.WithSelf.Op.type.d27
// CHECK:STDOUT:   %.loc170_5.6 => constants.%.3b0
// CHECK:STDOUT:   %impl.elem1.loc170_5.2 => constants.%Op.72717c.15
// CHECK:STDOUT:   %specific_impl_fn.loc170_5.3 => constants.%Op.72717c.15
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type => constants.%Destroy.WithSelf.SelfDestruct.type.36f
// CHECK:STDOUT:   %.loc170_5.7 => constants.%.7aa
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.1bfa2a.3
// CHECK:STDOUT:   %impl.elem2.loc170_5.2 => constants.%Destroy.WithSelf.SelfDestruct.db3fdb.4
// CHECK:STDOUT:   %specific_impl_fn.loc170_5.4 => constants.%Destroy.WithSelf.SelfDestruct.db3fdb.4
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
// CHECK:STDOUT: specific @AddAssignWith.loc225(constants.%i16, constants.%AddAssignWith.facet.510) {
// CHECK:STDOUT:   %U.patt.loc225_19.2 => constants.%U.patt.3a0
// CHECK:STDOUT:   %U.loc225_19.1 => constants.%i16
// CHECK:STDOUT:   %AddAssignWith.type.loc225_50.1 => constants.%AddAssignWith.type.b50
// CHECK:STDOUT:   %pattern_type.loc225_28 => constants.%pattern_type.010
// CHECK:STDOUT:   %T.patt.loc225_28.2 => constants.%T.patt.0f8
// CHECK:STDOUT:   %T.loc225_28.1 => constants.%AddAssignWith.facet.510
// CHECK:STDOUT:   %T.as_type.loc225_60.1 => constants.%Int16
// CHECK:STDOUT:   %pattern_type.loc225_58 => constants.%pattern_type.785
// CHECK:STDOUT:   %x.patt.loc225_58.2 => constants.%x.patt.49e
// CHECK:STDOUT:   %x.param_patt.loc225_53.2 => constants.%x.param_patt.69a
// CHECK:STDOUT:   %pattern_type.loc225_64 => constants.%pattern_type.cbf
// CHECK:STDOUT:   %y.param_patt.loc225_64.2 => constants.%y.param_patt.40e
// CHECK:STDOUT:   %y.patt.loc225_64.2 => constants.%y.patt.b86
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc225_53 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc225_64 => constants.%complete_type.2da
// CHECK:STDOUT:   %require_complete.loc227 => constants.%complete_type.1ab
// CHECK:STDOUT:   %AddAssignWith.assoc_type => constants.%AddAssignWith.assoc_type.75b
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.017
// CHECK:STDOUT:   %AddAssignWith.WithSelf.Op.type => constants.%AddAssignWith.WithSelf.Op.type.ccb
// CHECK:STDOUT:   %.loc227_5.2 => constants.%.0ed
// CHECK:STDOUT:   %AddAssignWith.lookup_impl_witness => constants.%custom_witness.af6
// CHECK:STDOUT:   %impl.elem0.loc227_5.2 => constants.%Int16.Op.4dca31.12
// CHECK:STDOUT:   %specific_impl_fn.loc227_5.2 => constants.%Int16.Op.4dca31.12
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @SubAssignWith.loc231(constants.%i16, constants.%SubAssignWith.facet.6c0) {
// CHECK:STDOUT:   %U.patt.loc231_19.2 => constants.%U.patt.3a0
// CHECK:STDOUT:   %U.loc231_19.1 => constants.%i16
// CHECK:STDOUT:   %SubAssignWith.type.loc231_50.1 => constants.%SubAssignWith.type.5c1
// CHECK:STDOUT:   %pattern_type.loc231_28 => constants.%pattern_type.7e6
// CHECK:STDOUT:   %T.patt.loc231_28.2 => constants.%T.patt.a81
// CHECK:STDOUT:   %T.loc231_28.1 => constants.%SubAssignWith.facet.6c0
// CHECK:STDOUT:   %T.as_type.loc231_60.1 => constants.%Int16
// CHECK:STDOUT:   %pattern_type.loc231_58 => constants.%pattern_type.785
// CHECK:STDOUT:   %x.patt.loc231_58.2 => constants.%x.patt.49e
// CHECK:STDOUT:   %x.param_patt.loc231_53.2 => constants.%x.param_patt.69a
// CHECK:STDOUT:   %pattern_type.loc231_64 => constants.%pattern_type.cbf
// CHECK:STDOUT:   %y.param_patt.loc231_64.2 => constants.%y.param_patt.40e
// CHECK:STDOUT:   %y.patt.loc231_64.2 => constants.%y.patt.b86
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc231_53 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc231_64 => constants.%complete_type.2da
// CHECK:STDOUT:   %require_complete.loc233 => constants.%complete_type.767
// CHECK:STDOUT:   %SubAssignWith.assoc_type => constants.%SubAssignWith.assoc_type.2e5
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.7e9
// CHECK:STDOUT:   %SubAssignWith.WithSelf.Op.type => constants.%SubAssignWith.WithSelf.Op.type.311
// CHECK:STDOUT:   %.loc233_5.2 => constants.%.472
// CHECK:STDOUT:   %SubAssignWith.lookup_impl_witness => constants.%custom_witness.ba9
// CHECK:STDOUT:   %impl.elem0.loc233_5.2 => constants.%Int16.Op.4dca31.13
// CHECK:STDOUT:   %specific_impl_fn.loc233_5.2 => constants.%Int16.Op.4dca31.13
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @MulAssignWith.loc237(constants.%i16, constants.%MulAssignWith.facet.7b9) {
// CHECK:STDOUT:   %U.patt.loc237_19.2 => constants.%U.patt.3a0
// CHECK:STDOUT:   %U.loc237_19.1 => constants.%i16
// CHECK:STDOUT:   %MulAssignWith.type.loc237_50.1 => constants.%MulAssignWith.type.8be
// CHECK:STDOUT:   %pattern_type.loc237_28 => constants.%pattern_type.186
// CHECK:STDOUT:   %T.patt.loc237_28.2 => constants.%T.patt.a55
// CHECK:STDOUT:   %T.loc237_28.1 => constants.%MulAssignWith.facet.7b9
// CHECK:STDOUT:   %T.as_type.loc237_60.1 => constants.%Int16
// CHECK:STDOUT:   %pattern_type.loc237_58 => constants.%pattern_type.785
// CHECK:STDOUT:   %x.patt.loc237_58.2 => constants.%x.patt.49e
// CHECK:STDOUT:   %x.param_patt.loc237_53.2 => constants.%x.param_patt.69a
// CHECK:STDOUT:   %pattern_type.loc237_64 => constants.%pattern_type.cbf
// CHECK:STDOUT:   %y.param_patt.loc237_64.2 => constants.%y.param_patt.40e
// CHECK:STDOUT:   %y.patt.loc237_64.2 => constants.%y.patt.b86
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc237_53 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc237_64 => constants.%complete_type.2da
// CHECK:STDOUT:   %require_complete.loc239 => constants.%complete_type.512
// CHECK:STDOUT:   %MulAssignWith.assoc_type => constants.%MulAssignWith.assoc_type.3cc
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.2ab
// CHECK:STDOUT:   %MulAssignWith.WithSelf.Op.type => constants.%MulAssignWith.WithSelf.Op.type.697
// CHECK:STDOUT:   %.loc239_5.2 => constants.%.1c8
// CHECK:STDOUT:   %MulAssignWith.lookup_impl_witness => constants.%custom_witness.87d
// CHECK:STDOUT:   %impl.elem0.loc239_5.2 => constants.%Int16.Op.4dca31.14
// CHECK:STDOUT:   %specific_impl_fn.loc239_5.2 => constants.%Int16.Op.4dca31.14
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @DivAssignWith.loc243(constants.%i16, constants.%DivAssignWith.facet.1ff) {
// CHECK:STDOUT:   %U.patt.loc243_19.2 => constants.%U.patt.3a0
// CHECK:STDOUT:   %U.loc243_19.1 => constants.%i16
// CHECK:STDOUT:   %DivAssignWith.type.loc243_50.1 => constants.%DivAssignWith.type.572
// CHECK:STDOUT:   %pattern_type.loc243_28 => constants.%pattern_type.54f
// CHECK:STDOUT:   %T.patt.loc243_28.2 => constants.%T.patt.998
// CHECK:STDOUT:   %T.loc243_28.1 => constants.%DivAssignWith.facet.1ff
// CHECK:STDOUT:   %T.as_type.loc243_60.1 => constants.%Int16
// CHECK:STDOUT:   %pattern_type.loc243_58 => constants.%pattern_type.785
// CHECK:STDOUT:   %x.patt.loc243_58.2 => constants.%x.patt.49e
// CHECK:STDOUT:   %x.param_patt.loc243_53.2 => constants.%x.param_patt.69a
// CHECK:STDOUT:   %pattern_type.loc243_64 => constants.%pattern_type.cbf
// CHECK:STDOUT:   %y.param_patt.loc243_64.2 => constants.%y.param_patt.40e
// CHECK:STDOUT:   %y.patt.loc243_64.2 => constants.%y.patt.b86
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc243_53 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc243_64 => constants.%complete_type.2da
// CHECK:STDOUT:   %require_complete.loc245 => constants.%complete_type.23e
// CHECK:STDOUT:   %DivAssignWith.assoc_type => constants.%DivAssignWith.assoc_type.753
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.e4a0
// CHECK:STDOUT:   %DivAssignWith.WithSelf.Op.type => constants.%DivAssignWith.WithSelf.Op.type.d0e
// CHECK:STDOUT:   %.loc245_5.2 => constants.%.f42
// CHECK:STDOUT:   %DivAssignWith.lookup_impl_witness => constants.%custom_witness.8a1
// CHECK:STDOUT:   %impl.elem0.loc245_5.2 => constants.%Int16.Op.4dca31.15
// CHECK:STDOUT:   %specific_impl_fn.loc245_5.2 => constants.%Int16.Op.4dca31.15
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ModAssignWith.loc249(constants.%i16, constants.%ModAssignWith.facet.9bc) {
// CHECK:STDOUT:   %U.patt.loc249_19.2 => constants.%U.patt.3a0
// CHECK:STDOUT:   %U.loc249_19.1 => constants.%i16
// CHECK:STDOUT:   %ModAssignWith.type.loc249_50.1 => constants.%ModAssignWith.type.444
// CHECK:STDOUT:   %pattern_type.loc249_28 => constants.%pattern_type.35b
// CHECK:STDOUT:   %T.patt.loc249_28.2 => constants.%T.patt.191
// CHECK:STDOUT:   %T.loc249_28.1 => constants.%ModAssignWith.facet.9bc
// CHECK:STDOUT:   %T.as_type.loc249_60.1 => constants.%Int16
// CHECK:STDOUT:   %pattern_type.loc249_58 => constants.%pattern_type.785
// CHECK:STDOUT:   %x.patt.loc249_58.2 => constants.%x.patt.49e
// CHECK:STDOUT:   %x.param_patt.loc249_53.2 => constants.%x.param_patt.69a
// CHECK:STDOUT:   %pattern_type.loc249_64 => constants.%pattern_type.cbf
// CHECK:STDOUT:   %y.param_patt.loc249_64.2 => constants.%y.param_patt.40e
// CHECK:STDOUT:   %y.patt.loc249_64.2 => constants.%y.patt.b86
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc249_53 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc249_64 => constants.%complete_type.2da
// CHECK:STDOUT:   %require_complete.loc251 => constants.%complete_type.4dc
// CHECK:STDOUT:   %ModAssignWith.assoc_type => constants.%ModAssignWith.assoc_type.a2c
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.0b1
// CHECK:STDOUT:   %ModAssignWith.WithSelf.Op.type => constants.%ModAssignWith.WithSelf.Op.type.ab6
// CHECK:STDOUT:   %.loc251_5.2 => constants.%.182
// CHECK:STDOUT:   %ModAssignWith.lookup_impl_witness => constants.%custom_witness.b91
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
// CHECK:STDOUT: specific @TestNegate(constants.%Result, constants.%T.e01) {
// CHECK:STDOUT:   %Result.patt.loc292_9.2 => constants.%Result.patt
// CHECK:STDOUT:   %Result.loc292_9.1 => constants.%Result
// CHECK:STDOUT:   %Result.as_type.loc293_49.1 => constants.%Result.as_type
// CHECK:STDOUT:   %facet_type => constants.%facet_type.352
// CHECK:STDOUT:   %pattern_type.loc293 => constants.%pattern_type.4c2
// CHECK:STDOUT:   %T.patt.loc293_4.2 => constants.%T.patt.c5b
// CHECK:STDOUT:   %T.loc293_4.1 => constants.%T.e01
// CHECK:STDOUT:   %pattern_type.loc294_11 => constants.%pattern_type.f4e485.2
// CHECK:STDOUT:   %y.param_patt.loc294_11.2 => constants.%y.param_patt.023
// CHECK:STDOUT:   %y.patt.loc294_11.2 => constants.%y.patt.33f
// CHECK:STDOUT:   %T.as_type.loc294_24.1 => constants.%T.as_type.684
// CHECK:STDOUT:   %pattern_type.loc294_22 => constants.%pattern_type.2dd
// CHECK:STDOUT:   %x.param_patt.loc294_22.2 => constants.%x.param_patt.cbc
// CHECK:STDOUT:   %x.patt.loc294_22.2 => constants.%x.patt.ef5
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
// CHECK:STDOUT: specific @TestNegate(constants.%Destroy.facet.0b4, constants.%facet_value.a67) {
// CHECK:STDOUT:   %Result.patt.loc292_9.2 => constants.%Result.patt
// CHECK:STDOUT:   %Result.loc292_9.1 => constants.%Destroy.facet.0b4
// CHECK:STDOUT:   %Result.as_type.loc293_49.1 => constants.%Int64
// CHECK:STDOUT:   %facet_type => constants.%facet_type.d8c
// CHECK:STDOUT:   %pattern_type.loc293 => constants.%pattern_type.54c
// CHECK:STDOUT:   %T.patt.loc293_4.2 => constants.%T.patt.16f3
// CHECK:STDOUT:   %T.loc293_4.1 => constants.%facet_value.a67
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
// CHECK:STDOUT:   %specific_impl_fn.loc296_3.3 => constants.%Int16.Op.4dca31.17
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type => constants.%Destroy.WithSelf.SelfDestruct.type.36f
// CHECK:STDOUT:   %.loc296_3.8 => constants.%.7aa
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.1bfa2a.3
// CHECK:STDOUT:   %impl.elem2.loc296_3.2 => constants.%Destroy.WithSelf.SelfDestruct.db3fdb.4
// CHECK:STDOUT:   %specific_impl_fn.loc296_3.4 => constants.%Destroy.WithSelf.SelfDestruct.db3fdb.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestNegate(constants.%Destroy.facet.84d, constants.%facet_value.73e) {
// CHECK:STDOUT:   %Result.patt.loc292_9.2 => constants.%Result.patt
// CHECK:STDOUT:   %Result.loc292_9.1 => constants.%Destroy.facet.84d
// CHECK:STDOUT:   %Result.as_type.loc293_49.1 => constants.%Int32
// CHECK:STDOUT:   %facet_type => constants.%facet_type.021
// CHECK:STDOUT:   %pattern_type.loc293 => constants.%pattern_type.4cd
// CHECK:STDOUT:   %T.patt.loc293_4.2 => constants.%T.patt.d74
// CHECK:STDOUT:   %T.loc293_4.1 => constants.%facet_value.73e
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
// CHECK:STDOUT:   %specific_impl_fn.loc296_3.3 => constants.%Int32.Op.4dca31.19
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type => constants.%Destroy.WithSelf.SelfDestruct.type.53c
// CHECK:STDOUT:   %.loc296_3.8 => constants.%.77c
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.1bfa2a.2
// CHECK:STDOUT:   %impl.elem2.loc296_3.2 => constants.%Destroy.WithSelf.SelfDestruct.db3fdb.3
// CHECK:STDOUT:   %specific_impl_fn.loc296_3.4 => constants.%Destroy.WithSelf.SelfDestruct.db3fdb.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
