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
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %pattern_type.429: type = pattern_type %Destroy.type [concrete]
// CHECK:STDOUT:   %Result.patt: %pattern_type.429 = symbolic_binding_pattern Result, 0 [symbolic]
// CHECK:STDOUT:   %Result: %Destroy.type = symbolic_binding Result, 0 [symbolic]
// CHECK:STDOUT:   %U.patt.7a2: %pattern_type.429 = symbolic_binding_pattern U, 1 [symbolic]
// CHECK:STDOUT:   %U.f40: %Destroy.type = symbolic_binding U, 1 [symbolic]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Other: type = symbolic_binding Other, 0 [symbolic]
// CHECK:STDOUT:   %AddWith.type.7e9: type = facet_type <@AddWith.1, @AddWith.1(%Other)> [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %Self.96a: %AddWith.type.7e9 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %AddWith.assoc_type.588: type = assoc_entity_type @AddWith.1, @AddWith.1(%Other) [symbolic]
// CHECK:STDOUT:   %AddWith.WithSelf.Op.type.28e: type = fn_type @AddWith.WithSelf.Op, @AddWith.WithSelf(%Other, %Self.96a) [symbolic]
// CHECK:STDOUT:   %AddWith.WithSelf.Op.eab: %AddWith.WithSelf.Op.type.28e = struct_value () [symbolic]
// CHECK:STDOUT:   %U.as_type.b61: type = facet_access_type %U.f40 [symbolic]
// CHECK:STDOUT:   %AddWith.type.9da: type = facet_type <@AddWith.1, @AddWith.1(%U.as_type.b61)> [symbolic]
// CHECK:STDOUT:   %facet_type.b1e: type = facet_type <@Destroy & @AddWith.1, @AddWith.1(%U.as_type.b61)> [symbolic]
// CHECK:STDOUT:   %.Self.frozen.1ea: %facet_type.b1e = symbolic_binding .Self [symbolic]
// CHECK:STDOUT:   %AddWith.assoc_type.bc8: type = assoc_entity_type @AddWith.1, @AddWith.1(%U.as_type.b61) [symbolic]
// CHECK:STDOUT:   %assoc0.21f: %AddWith.assoc_type.bc8 = assoc_entity element0, imports.%Core.import_ref.91f8 [symbolic]
// CHECK:STDOUT:   %assoc1.13e: %AddWith.assoc_type.bc8 = assoc_entity element1, imports.%Core.import_ref.3cf [symbolic]
// CHECK:STDOUT:   %require_complete.c7d: <witness> = require_complete_type %facet_type.b1e [symbolic]
// CHECK:STDOUT:   %.Self.frozen.as_type.015: type = facet_access_type %.Self.frozen.1ea [symbolic]
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.2f6: <witness> = lookup_impl_witness %.Self.frozen.1ea, @AddWith.1, @AddWith.1(%U.as_type.b61) [symbolic]
// CHECK:STDOUT:   %impl.elem0.1c6: type = impl_witness_access %AddWith.lookup_impl_witness.2f6, element0 [symbolic]
// CHECK:STDOUT:   %Result.as_type: type = facet_access_type %Result [symbolic]
// CHECK:STDOUT:   %.Self.014: %facet_type.b1e = symbolic_binding .Self [symbolic]
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.0d5: <witness> = lookup_impl_witness %.Self.014, @AddWith.1, @AddWith.1(%U.as_type.b61) [symbolic]
// CHECK:STDOUT:   %impl.elem0.ea5: type = impl_witness_access %AddWith.lookup_impl_witness.0d5, element0 [symbolic]
// CHECK:STDOUT:   %facet_type.c2e: type = facet_type <@Destroy & @AddWith.1, @AddWith.1(%U.as_type.b61) where %impl.elem0.ea5 = %Result.as_type> [symbolic]
// CHECK:STDOUT:   %pattern_type.0a9: type = pattern_type %facet_type.c2e [symbolic]
// CHECK:STDOUT:   %T.patt.69e: %pattern_type.0a9 = symbolic_binding_pattern T, 2 [symbolic]
// CHECK:STDOUT:   %T.ceb: %facet_type.c2e = symbolic_binding T, 2 [symbolic]
// CHECK:STDOUT:   %pattern_type.f4e485.2: type = pattern_type %Result.as_type [symbolic]
// CHECK:STDOUT:   %result.param_patt.6ba: %pattern_type.f4e485.2 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %result.patt.748373.1: %pattern_type.f4e485.2 = at_binding_pattern result, %result.param_patt.6ba [symbolic]
// CHECK:STDOUT:   %T.as_type.6fc: type = facet_access_type %T.ceb [symbolic]
// CHECK:STDOUT:   %pattern_type.786: type = pattern_type %T.as_type.6fc [symbolic]
// CHECK:STDOUT:   %x.param_patt.0c5: %pattern_type.786 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %x.patt.4db: %pattern_type.786 = at_binding_pattern x, %x.param_patt.0c5 [symbolic]
// CHECK:STDOUT:   %pattern_type.753: type = pattern_type %U.as_type.b61 [symbolic]
// CHECK:STDOUT:   %y.param_patt.740: %pattern_type.753 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %y.patt.4ae: %pattern_type.753 = at_binding_pattern y, %y.param_patt.740 [symbolic]
// CHECK:STDOUT:   %require_complete.15e: <witness> = require_complete_type %AddWith.type.9da [symbolic]
// CHECK:STDOUT:   %assoc1.57d: %AddWith.assoc_type.588 = assoc_entity element1, imports.%Core.import_ref.9768 [symbolic]
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.2b1: <witness> = lookup_impl_witness %T.ceb, @AddWith.1, @AddWith.1(%U.as_type.b61) [symbolic]
// CHECK:STDOUT:   %AddWith.facet.fea: %AddWith.type.9da = facet_value %T.as_type.6fc, (%AddWith.lookup_impl_witness.2b1) [symbolic]
// CHECK:STDOUT:   %AddWith.WithSelf.Op.type.977: type = fn_type @AddWith.WithSelf.Op, @AddWith.WithSelf(%U.as_type.b61, %AddWith.facet.fea) [symbolic]
// CHECK:STDOUT:   %.a30: type = fn_type_with_self_type %AddWith.WithSelf.Op.type.977, %AddWith.facet.fea [symbolic]
// CHECK:STDOUT:   %impl.elem1.b05: %.a30 = impl_witness_access %AddWith.lookup_impl_witness.2b1, element1 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.78d: <specific function> = specific_impl_function %impl.elem1.b05, @AddWith.WithSelf.Op(%U.as_type.b61, %AddWith.facet.fea) [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %Result, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.d3ece9.2: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Result) [symbolic]
// CHECK:STDOUT:   %.53a: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.d3ece9.2, %Result [symbolic]
// CHECK:STDOUT:   %impl.elem0.0f9: %.53a = impl_witness_access %Destroy.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.afc: <specific function> = specific_impl_function %impl.elem0.0f9, @Destroy.WithSelf.Op(%Result) [symbolic]
// CHECK:STDOUT:   %SubWith.type.ddc: type = facet_type <@SubWith.1, @SubWith.1(%Other)> [symbolic]
// CHECK:STDOUT:   %Self.d6b: %SubWith.type.ddc = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %SubWith.assoc_type.b0b: type = assoc_entity_type @SubWith.1, @SubWith.1(%Other) [symbolic]
// CHECK:STDOUT:   %SubWith.WithSelf.Op.type.68d: type = fn_type @SubWith.WithSelf.Op, @SubWith.WithSelf(%Other, %Self.d6b) [symbolic]
// CHECK:STDOUT:   %SubWith.WithSelf.Op.97a: %SubWith.WithSelf.Op.type.68d = struct_value () [symbolic]
// CHECK:STDOUT:   %SubWith.type.b5d: type = facet_type <@SubWith.1, @SubWith.1(%U.as_type.b61)> [symbolic]
// CHECK:STDOUT:   %facet_type.d46: type = facet_type <@Destroy & @SubWith.1, @SubWith.1(%U.as_type.b61)> [symbolic]
// CHECK:STDOUT:   %.Self.frozen.efa: %facet_type.d46 = symbolic_binding .Self [symbolic]
// CHECK:STDOUT:   %SubWith.assoc_type.3a1: type = assoc_entity_type @SubWith.1, @SubWith.1(%U.as_type.b61) [symbolic]
// CHECK:STDOUT:   %assoc0.729: %SubWith.assoc_type.3a1 = assoc_entity element0, imports.%Core.import_ref.190e [symbolic]
// CHECK:STDOUT:   %assoc1.8fd: %SubWith.assoc_type.3a1 = assoc_entity element1, imports.%Core.import_ref.3a4 [symbolic]
// CHECK:STDOUT:   %require_complete.cda: <witness> = require_complete_type %facet_type.d46 [symbolic]
// CHECK:STDOUT:   %.Self.frozen.as_type.79f: type = facet_access_type %.Self.frozen.efa [symbolic]
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.acf: <witness> = lookup_impl_witness %.Self.frozen.efa, @SubWith.1, @SubWith.1(%U.as_type.b61) [symbolic]
// CHECK:STDOUT:   %impl.elem0.050: type = impl_witness_access %SubWith.lookup_impl_witness.acf, element0 [symbolic]
// CHECK:STDOUT:   %.Self.d13: %facet_type.d46 = symbolic_binding .Self [symbolic]
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.5d0: <witness> = lookup_impl_witness %.Self.d13, @SubWith.1, @SubWith.1(%U.as_type.b61) [symbolic]
// CHECK:STDOUT:   %impl.elem0.e4e: type = impl_witness_access %SubWith.lookup_impl_witness.5d0, element0 [symbolic]
// CHECK:STDOUT:   %facet_type.eeb: type = facet_type <@Destroy & @SubWith.1, @SubWith.1(%U.as_type.b61) where %impl.elem0.e4e = %Result.as_type> [symbolic]
// CHECK:STDOUT:   %pattern_type.502: type = pattern_type %facet_type.eeb [symbolic]
// CHECK:STDOUT:   %T.patt.cd4: %pattern_type.502 = symbolic_binding_pattern T, 2 [symbolic]
// CHECK:STDOUT:   %T.bf6: %facet_type.eeb = symbolic_binding T, 2 [symbolic]
// CHECK:STDOUT:   %T.as_type.58f: type = facet_access_type %T.bf6 [symbolic]
// CHECK:STDOUT:   %pattern_type.a10: type = pattern_type %T.as_type.58f [symbolic]
// CHECK:STDOUT:   %x.param_patt.02e: %pattern_type.a10 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %x.patt.eb5: %pattern_type.a10 = at_binding_pattern x, %x.param_patt.02e [symbolic]
// CHECK:STDOUT:   %require_complete.6ff: <witness> = require_complete_type %SubWith.type.b5d [symbolic]
// CHECK:STDOUT:   %assoc1.178: %SubWith.assoc_type.b0b = assoc_entity element1, imports.%Core.import_ref.b59 [symbolic]
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.ce5: <witness> = lookup_impl_witness %T.bf6, @SubWith.1, @SubWith.1(%U.as_type.b61) [symbolic]
// CHECK:STDOUT:   %SubWith.facet.f6a: %SubWith.type.b5d = facet_value %T.as_type.58f, (%SubWith.lookup_impl_witness.ce5) [symbolic]
// CHECK:STDOUT:   %SubWith.WithSelf.Op.type.4fe: type = fn_type @SubWith.WithSelf.Op, @SubWith.WithSelf(%U.as_type.b61, %SubWith.facet.f6a) [symbolic]
// CHECK:STDOUT:   %.a4a: type = fn_type_with_self_type %SubWith.WithSelf.Op.type.4fe, %SubWith.facet.f6a [symbolic]
// CHECK:STDOUT:   %impl.elem1.e4f: %.a4a = impl_witness_access %SubWith.lookup_impl_witness.ce5, element1 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.46d: <specific function> = specific_impl_function %impl.elem1.e4f, @SubWith.WithSelf.Op(%U.as_type.b61, %SubWith.facet.f6a) [symbolic]
// CHECK:STDOUT:   %MulWith.type.f84: type = facet_type <@MulWith.1, @MulWith.1(%Other)> [symbolic]
// CHECK:STDOUT:   %Self.941: %MulWith.type.f84 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %MulWith.assoc_type.7b2: type = assoc_entity_type @MulWith.1, @MulWith.1(%Other) [symbolic]
// CHECK:STDOUT:   %MulWith.WithSelf.Op.type.176: type = fn_type @MulWith.WithSelf.Op, @MulWith.WithSelf(%Other, %Self.941) [symbolic]
// CHECK:STDOUT:   %MulWith.WithSelf.Op.a75: %MulWith.WithSelf.Op.type.176 = struct_value () [symbolic]
// CHECK:STDOUT:   %MulWith.type.f02: type = facet_type <@MulWith.1, @MulWith.1(%U.as_type.b61)> [symbolic]
// CHECK:STDOUT:   %facet_type.f87: type = facet_type <@Destroy & @MulWith.1, @MulWith.1(%U.as_type.b61)> [symbolic]
// CHECK:STDOUT:   %.Self.frozen.463: %facet_type.f87 = symbolic_binding .Self [symbolic]
// CHECK:STDOUT:   %MulWith.assoc_type.a12: type = assoc_entity_type @MulWith.1, @MulWith.1(%U.as_type.b61) [symbolic]
// CHECK:STDOUT:   %assoc0.116: %MulWith.assoc_type.a12 = assoc_entity element0, imports.%Core.import_ref.59c [symbolic]
// CHECK:STDOUT:   %assoc1.a0f: %MulWith.assoc_type.a12 = assoc_entity element1, imports.%Core.import_ref.5a0 [symbolic]
// CHECK:STDOUT:   %require_complete.0a3: <witness> = require_complete_type %facet_type.f87 [symbolic]
// CHECK:STDOUT:   %.Self.frozen.as_type.e91: type = facet_access_type %.Self.frozen.463 [symbolic]
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.637: <witness> = lookup_impl_witness %.Self.frozen.463, @MulWith.1, @MulWith.1(%U.as_type.b61) [symbolic]
// CHECK:STDOUT:   %impl.elem0.132: type = impl_witness_access %MulWith.lookup_impl_witness.637, element0 [symbolic]
// CHECK:STDOUT:   %.Self.588: %facet_type.f87 = symbolic_binding .Self [symbolic]
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.d4f: <witness> = lookup_impl_witness %.Self.588, @MulWith.1, @MulWith.1(%U.as_type.b61) [symbolic]
// CHECK:STDOUT:   %impl.elem0.62f: type = impl_witness_access %MulWith.lookup_impl_witness.d4f, element0 [symbolic]
// CHECK:STDOUT:   %facet_type.0e5: type = facet_type <@Destroy & @MulWith.1, @MulWith.1(%U.as_type.b61) where %impl.elem0.62f = %Result.as_type> [symbolic]
// CHECK:STDOUT:   %pattern_type.e8a: type = pattern_type %facet_type.0e5 [symbolic]
// CHECK:STDOUT:   %T.patt.dec: %pattern_type.e8a = symbolic_binding_pattern T, 2 [symbolic]
// CHECK:STDOUT:   %T.9a2: %facet_type.0e5 = symbolic_binding T, 2 [symbolic]
// CHECK:STDOUT:   %T.as_type.83f: type = facet_access_type %T.9a2 [symbolic]
// CHECK:STDOUT:   %pattern_type.fff: type = pattern_type %T.as_type.83f [symbolic]
// CHECK:STDOUT:   %x.param_patt.f8f: %pattern_type.fff = value_param_pattern [symbolic]
// CHECK:STDOUT:   %x.patt.909: %pattern_type.fff = at_binding_pattern x, %x.param_patt.f8f [symbolic]
// CHECK:STDOUT:   %require_complete.861: <witness> = require_complete_type %MulWith.type.f02 [symbolic]
// CHECK:STDOUT:   %assoc1.8f7: %MulWith.assoc_type.7b2 = assoc_entity element1, imports.%Core.import_ref.955 [symbolic]
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.444: <witness> = lookup_impl_witness %T.9a2, @MulWith.1, @MulWith.1(%U.as_type.b61) [symbolic]
// CHECK:STDOUT:   %MulWith.facet.b99: %MulWith.type.f02 = facet_value %T.as_type.83f, (%MulWith.lookup_impl_witness.444) [symbolic]
// CHECK:STDOUT:   %MulWith.WithSelf.Op.type.c95: type = fn_type @MulWith.WithSelf.Op, @MulWith.WithSelf(%U.as_type.b61, %MulWith.facet.b99) [symbolic]
// CHECK:STDOUT:   %.a49: type = fn_type_with_self_type %MulWith.WithSelf.Op.type.c95, %MulWith.facet.b99 [symbolic]
// CHECK:STDOUT:   %impl.elem1.a7b: %.a49 = impl_witness_access %MulWith.lookup_impl_witness.444, element1 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.eab: <specific function> = specific_impl_function %impl.elem1.a7b, @MulWith.WithSelf.Op(%U.as_type.b61, %MulWith.facet.b99) [symbolic]
// CHECK:STDOUT:   %DivWith.type.13f: type = facet_type <@DivWith.1, @DivWith.1(%Other)> [symbolic]
// CHECK:STDOUT:   %Self.16a: %DivWith.type.13f = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %DivWith.assoc_type.d8b: type = assoc_entity_type @DivWith.1, @DivWith.1(%Other) [symbolic]
// CHECK:STDOUT:   %DivWith.WithSelf.Op.type.0f0: type = fn_type @DivWith.WithSelf.Op, @DivWith.WithSelf(%Other, %Self.16a) [symbolic]
// CHECK:STDOUT:   %DivWith.WithSelf.Op.51e: %DivWith.WithSelf.Op.type.0f0 = struct_value () [symbolic]
// CHECK:STDOUT:   %DivWith.type.523: type = facet_type <@DivWith.1, @DivWith.1(%U.as_type.b61)> [symbolic]
// CHECK:STDOUT:   %facet_type.86cd: type = facet_type <@Destroy & @DivWith.1, @DivWith.1(%U.as_type.b61)> [symbolic]
// CHECK:STDOUT:   %.Self.frozen.1f9: %facet_type.86cd = symbolic_binding .Self [symbolic]
// CHECK:STDOUT:   %DivWith.assoc_type.cf4: type = assoc_entity_type @DivWith.1, @DivWith.1(%U.as_type.b61) [symbolic]
// CHECK:STDOUT:   %assoc0.c57: %DivWith.assoc_type.cf4 = assoc_entity element0, imports.%Core.import_ref.dee [symbolic]
// CHECK:STDOUT:   %assoc1.4eb: %DivWith.assoc_type.cf4 = assoc_entity element1, imports.%Core.import_ref.7845 [symbolic]
// CHECK:STDOUT:   %require_complete.4d7: <witness> = require_complete_type %facet_type.86cd [symbolic]
// CHECK:STDOUT:   %.Self.frozen.as_type.8ca: type = facet_access_type %.Self.frozen.1f9 [symbolic]
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.154: <witness> = lookup_impl_witness %.Self.frozen.1f9, @DivWith.1, @DivWith.1(%U.as_type.b61) [symbolic]
// CHECK:STDOUT:   %impl.elem0.252: type = impl_witness_access %DivWith.lookup_impl_witness.154, element0 [symbolic]
// CHECK:STDOUT:   %.Self.862: %facet_type.86cd = symbolic_binding .Self [symbolic]
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.d02: <witness> = lookup_impl_witness %.Self.862, @DivWith.1, @DivWith.1(%U.as_type.b61) [symbolic]
// CHECK:STDOUT:   %impl.elem0.606: type = impl_witness_access %DivWith.lookup_impl_witness.d02, element0 [symbolic]
// CHECK:STDOUT:   %facet_type.ed9: type = facet_type <@Destroy & @DivWith.1, @DivWith.1(%U.as_type.b61) where %impl.elem0.606 = %Result.as_type> [symbolic]
// CHECK:STDOUT:   %pattern_type.07c: type = pattern_type %facet_type.ed9 [symbolic]
// CHECK:STDOUT:   %T.patt.869: %pattern_type.07c = symbolic_binding_pattern T, 2 [symbolic]
// CHECK:STDOUT:   %T.4bb: %facet_type.ed9 = symbolic_binding T, 2 [symbolic]
// CHECK:STDOUT:   %T.as_type.a74: type = facet_access_type %T.4bb [symbolic]
// CHECK:STDOUT:   %pattern_type.b83: type = pattern_type %T.as_type.a74 [symbolic]
// CHECK:STDOUT:   %x.param_patt.21f: %pattern_type.b83 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %x.patt.46c: %pattern_type.b83 = at_binding_pattern x, %x.param_patt.21f [symbolic]
// CHECK:STDOUT:   %require_complete.026: <witness> = require_complete_type %DivWith.type.523 [symbolic]
// CHECK:STDOUT:   %assoc1.7b1: %DivWith.assoc_type.d8b = assoc_entity element1, imports.%Core.import_ref.8b6 [symbolic]
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.831: <witness> = lookup_impl_witness %T.4bb, @DivWith.1, @DivWith.1(%U.as_type.b61) [symbolic]
// CHECK:STDOUT:   %DivWith.facet.5f7: %DivWith.type.523 = facet_value %T.as_type.a74, (%DivWith.lookup_impl_witness.831) [symbolic]
// CHECK:STDOUT:   %DivWith.WithSelf.Op.type.1b2: type = fn_type @DivWith.WithSelf.Op, @DivWith.WithSelf(%U.as_type.b61, %DivWith.facet.5f7) [symbolic]
// CHECK:STDOUT:   %.6b1: type = fn_type_with_self_type %DivWith.WithSelf.Op.type.1b2, %DivWith.facet.5f7 [symbolic]
// CHECK:STDOUT:   %impl.elem1.bfe: %.6b1 = impl_witness_access %DivWith.lookup_impl_witness.831, element1 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.e98: <specific function> = specific_impl_function %impl.elem1.bfe, @DivWith.WithSelf.Op(%U.as_type.b61, %DivWith.facet.5f7) [symbolic]
// CHECK:STDOUT:   %ModWith.type.2e8: type = facet_type <@ModWith.1, @ModWith.1(%Other)> [symbolic]
// CHECK:STDOUT:   %Self.0ef: %ModWith.type.2e8 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %ModWith.assoc_type.316: type = assoc_entity_type @ModWith.1, @ModWith.1(%Other) [symbolic]
// CHECK:STDOUT:   %ModWith.WithSelf.Op.type.d14: type = fn_type @ModWith.WithSelf.Op, @ModWith.WithSelf(%Other, %Self.0ef) [symbolic]
// CHECK:STDOUT:   %ModWith.WithSelf.Op.9e4: %ModWith.WithSelf.Op.type.d14 = struct_value () [symbolic]
// CHECK:STDOUT:   %ModWith.type.937: type = facet_type <@ModWith.1, @ModWith.1(%U.as_type.b61)> [symbolic]
// CHECK:STDOUT:   %facet_type.b18: type = facet_type <@Destroy & @ModWith.1, @ModWith.1(%U.as_type.b61)> [symbolic]
// CHECK:STDOUT:   %.Self.frozen.f70: %facet_type.b18 = symbolic_binding .Self [symbolic]
// CHECK:STDOUT:   %ModWith.assoc_type.626: type = assoc_entity_type @ModWith.1, @ModWith.1(%U.as_type.b61) [symbolic]
// CHECK:STDOUT:   %assoc0.56e: %ModWith.assoc_type.626 = assoc_entity element0, imports.%Core.import_ref.61a [symbolic]
// CHECK:STDOUT:   %assoc1.6d1: %ModWith.assoc_type.626 = assoc_entity element1, imports.%Core.import_ref.b56 [symbolic]
// CHECK:STDOUT:   %require_complete.708: <witness> = require_complete_type %facet_type.b18 [symbolic]
// CHECK:STDOUT:   %.Self.frozen.as_type.447: type = facet_access_type %.Self.frozen.f70 [symbolic]
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.80e: <witness> = lookup_impl_witness %.Self.frozen.f70, @ModWith.1, @ModWith.1(%U.as_type.b61) [symbolic]
// CHECK:STDOUT:   %impl.elem0.523: type = impl_witness_access %ModWith.lookup_impl_witness.80e, element0 [symbolic]
// CHECK:STDOUT:   %.Self.a68: %facet_type.b18 = symbolic_binding .Self [symbolic]
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.569: <witness> = lookup_impl_witness %.Self.a68, @ModWith.1, @ModWith.1(%U.as_type.b61) [symbolic]
// CHECK:STDOUT:   %impl.elem0.c51: type = impl_witness_access %ModWith.lookup_impl_witness.569, element0 [symbolic]
// CHECK:STDOUT:   %facet_type.639: type = facet_type <@Destroy & @ModWith.1, @ModWith.1(%U.as_type.b61) where %impl.elem0.c51 = %Result.as_type> [symbolic]
// CHECK:STDOUT:   %pattern_type.d85: type = pattern_type %facet_type.639 [symbolic]
// CHECK:STDOUT:   %T.patt.49b: %pattern_type.d85 = symbolic_binding_pattern T, 2 [symbolic]
// CHECK:STDOUT:   %T.999: %facet_type.639 = symbolic_binding T, 2 [symbolic]
// CHECK:STDOUT:   %T.as_type.1c2: type = facet_access_type %T.999 [symbolic]
// CHECK:STDOUT:   %pattern_type.01b: type = pattern_type %T.as_type.1c2 [symbolic]
// CHECK:STDOUT:   %x.param_patt.f25: %pattern_type.01b = value_param_pattern [symbolic]
// CHECK:STDOUT:   %x.patt.804: %pattern_type.01b = at_binding_pattern x, %x.param_patt.f25 [symbolic]
// CHECK:STDOUT:   %require_complete.a4a: <witness> = require_complete_type %ModWith.type.937 [symbolic]
// CHECK:STDOUT:   %assoc1.132: %ModWith.assoc_type.316 = assoc_entity element1, imports.%Core.import_ref.33c [symbolic]
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.e17: <witness> = lookup_impl_witness %T.999, @ModWith.1, @ModWith.1(%U.as_type.b61) [symbolic]
// CHECK:STDOUT:   %ModWith.facet.874: %ModWith.type.937 = facet_value %T.as_type.1c2, (%ModWith.lookup_impl_witness.e17) [symbolic]
// CHECK:STDOUT:   %ModWith.WithSelf.Op.type.314: type = fn_type @ModWith.WithSelf.Op, @ModWith.WithSelf(%U.as_type.b61, %ModWith.facet.874) [symbolic]
// CHECK:STDOUT:   %.195: type = fn_type_with_self_type %ModWith.WithSelf.Op.type.314, %ModWith.facet.874 [symbolic]
// CHECK:STDOUT:   %impl.elem1.715: %.195 = impl_witness_access %ModWith.lookup_impl_witness.e17, element1 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.0b2: <specific function> = specific_impl_function %impl.elem1.715, @ModWith.WithSelf.Op(%U.as_type.b61, %ModWith.facet.874) [symbolic]
// CHECK:STDOUT:   %Int16: type = class_type @Int16 [concrete]
// CHECK:STDOUT:   %pattern_type.785: type = pattern_type %Int16 [concrete]
// CHECK:STDOUT:   %Int32: type = class_type @Int32 [concrete]
// CHECK:STDOUT:   %pattern_type.8b4: type = pattern_type %Int32 [concrete]
// CHECK:STDOUT:   %Int64: type = class_type @Int64 [concrete]
// CHECK:STDOUT:   %pattern_type.4aa: type = pattern_type %Int64 [concrete]
// CHECK:STDOUT:   %int_16: Core.IntLiteral = int_value 16 [concrete]
// CHECK:STDOUT:   %i16: type = class_type @Int, @Int(%int_16) [concrete]
// CHECK:STDOUT:   %pattern_type.203: type = pattern_type %i16 [concrete]
// CHECK:STDOUT:   %x.param_patt.676: %pattern_type.203 = value_param_pattern [concrete]
// CHECK:STDOUT:   %x.patt.26e: %pattern_type.203 = at_binding_pattern x, %x.param_patt.676 [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %i16.builtin: type = int_type signed, %int_16 [concrete]
// CHECK:STDOUT:   %complete_type.2da: <witness> = complete_type_witness %i16.builtin [concrete]
// CHECK:STDOUT:   %Int16.Op.type.83af6e.1: type = fn_type @Int16.Op.1 [concrete]
// CHECK:STDOUT:   %Int16.Op.4dca31.1: %Int16.Op.type.83af6e.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.e7a31c.1: <witness> = custom_witness (%Int16.Op.4dca31.1), @Destroy [concrete]
// CHECK:STDOUT:   %Destroy.facet.2cc: %Destroy.type = facet_value %Int16, (%custom_witness.e7a31c.1) [concrete]
// CHECK:STDOUT:   %facet_type.297: type = facet_type <@Destroy & @AddWith.1, @AddWith.1(%Int16)> [concrete]
// CHECK:STDOUT:   %.Self.45a: %facet_type.297 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.131: <witness> = lookup_impl_witness %.Self.45a, @AddWith.1, @AddWith.1(%Int16) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.28c: type = impl_witness_access %AddWith.lookup_impl_witness.131, element0 [symbolic_self]
// CHECK:STDOUT:   %facet_type.225: type = facet_type <@Destroy & @AddWith.1, @AddWith.1(%Int16) where %impl.elem0.28c = %Int16> [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.2: type = fn_type @Destroy.Op.loc176_34.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.2: %Destroy.Op.type.1d8f74.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.df9cc1.2: <witness> = custom_witness (%Destroy.Op.1a2547.2), @Destroy [concrete]
// CHECK:STDOUT:   %AddWith.type.b62: type = facet_type <@AddWith.1, @AddWith.1(%Int16)> [concrete]
// CHECK:STDOUT:   %AddWith.assoc_type.c72: type = assoc_entity_type @AddWith.1, @AddWith.1(%Int16) [concrete]
// CHECK:STDOUT:   %assoc0.261: %AddWith.assoc_type.c72 = assoc_entity element0, imports.%Core.import_ref.91f8 [concrete]
// CHECK:STDOUT:   %assoc1.006: %AddWith.assoc_type.c72 = assoc_entity element1, imports.%Core.import_ref.3cf [concrete]
// CHECK:STDOUT:   %Op.type.442b68.1: type = fn_type @Op.1 [concrete]
// CHECK:STDOUT:   %Op.72717c.1: %Op.type.442b68.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.facet.e5c: %Destroy.type = facet_value %i16, (%custom_witness.df9cc1.2) [concrete]
// CHECK:STDOUT:   %custom_witness.248: <witness> = custom_witness (%Int16, %Op.72717c.1), @AddWith.1, @AddWith.1(%Int16) [concrete]
// CHECK:STDOUT:   %facet_value.696: %facet_type.225 = facet_value %i16, (%custom_witness.df9cc1.2, %custom_witness.248) [concrete]
// CHECK:STDOUT:   %.Self.frozen.7e3: %facet_type.297 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %complete_type.a82: <witness> = complete_type_witness %facet_type.297 [concrete]
// CHECK:STDOUT:   %.Self.frozen.as_type.85e: type = facet_access_type %.Self.frozen.7e3 [symbolic_self]
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.5b7: <witness> = lookup_impl_witness %.Self.frozen.7e3, @AddWith.1, @AddWith.1(%Int16) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.d80: type = impl_witness_access %AddWith.lookup_impl_witness.5b7, element0 [symbolic_self]
// CHECK:STDOUT:   %pattern_type.34a: type = pattern_type %facet_type.225 [concrete]
// CHECK:STDOUT:   %T.patt.3ec: %pattern_type.34a = symbolic_binding_pattern T, 2 [symbolic]
// CHECK:STDOUT:   %result.param_patt.be6: %pattern_type.785 = value_param_pattern [concrete]
// CHECK:STDOUT:   %result.patt.488: %pattern_type.785 = at_binding_pattern result, %result.param_patt.be6 [concrete]
// CHECK:STDOUT:   %y.param_patt.418: %pattern_type.785 = value_param_pattern [concrete]
// CHECK:STDOUT:   %y.patt.4e6: %pattern_type.785 = at_binding_pattern y, %y.param_patt.418 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.9d5: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.2cc) [concrete]
// CHECK:STDOUT:   %.6da: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.9d5, %Destroy.facet.2cc [concrete]
// CHECK:STDOUT:   %facet_type.b15: type = facet_type <@Destroy & @SubWith.1, @SubWith.1(%Int16)> [concrete]
// CHECK:STDOUT:   %.Self.415: %facet_type.b15 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.266: <witness> = lookup_impl_witness %.Self.415, @SubWith.1, @SubWith.1(%Int16) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.4a0: type = impl_witness_access %SubWith.lookup_impl_witness.266, element0 [symbolic_self]
// CHECK:STDOUT:   %facet_type.b71: type = facet_type <@Destroy & @SubWith.1, @SubWith.1(%Int16) where %impl.elem0.4a0 = %Int16> [concrete]
// CHECK:STDOUT:   %SubWith.type.b12: type = facet_type <@SubWith.1, @SubWith.1(%Int16)> [concrete]
// CHECK:STDOUT:   %SubWith.assoc_type.eb8: type = assoc_entity_type @SubWith.1, @SubWith.1(%Int16) [concrete]
// CHECK:STDOUT:   %assoc0.a90: %SubWith.assoc_type.eb8 = assoc_entity element0, imports.%Core.import_ref.190e [concrete]
// CHECK:STDOUT:   %assoc1.2cf: %SubWith.assoc_type.eb8 = assoc_entity element1, imports.%Core.import_ref.3a4 [concrete]
// CHECK:STDOUT:   %Op.type.442b68.2: type = fn_type @Op.2 [concrete]
// CHECK:STDOUT:   %Op.72717c.2: %Op.type.442b68.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.f14: <witness> = custom_witness (%Int16, %Op.72717c.2), @SubWith.1, @SubWith.1(%Int16) [concrete]
// CHECK:STDOUT:   %facet_value.031: %facet_type.b71 = facet_value %i16, (%custom_witness.df9cc1.2, %custom_witness.f14) [concrete]
// CHECK:STDOUT:   %.Self.frozen.581: %facet_type.b15 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %complete_type.647: <witness> = complete_type_witness %facet_type.b15 [concrete]
// CHECK:STDOUT:   %.Self.frozen.as_type.092: type = facet_access_type %.Self.frozen.581 [symbolic_self]
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.b30: <witness> = lookup_impl_witness %.Self.frozen.581, @SubWith.1, @SubWith.1(%Int16) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.748: type = impl_witness_access %SubWith.lookup_impl_witness.b30, element0 [symbolic_self]
// CHECK:STDOUT:   %pattern_type.f54: type = pattern_type %facet_type.b71 [concrete]
// CHECK:STDOUT:   %T.patt.842: %pattern_type.f54 = symbolic_binding_pattern T, 2 [symbolic]
// CHECK:STDOUT:   %facet_type.a64: type = facet_type <@Destroy & @MulWith.1, @MulWith.1(%Int16)> [concrete]
// CHECK:STDOUT:   %.Self.00f: %facet_type.a64 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.782: <witness> = lookup_impl_witness %.Self.00f, @MulWith.1, @MulWith.1(%Int16) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.8de: type = impl_witness_access %MulWith.lookup_impl_witness.782, element0 [symbolic_self]
// CHECK:STDOUT:   %facet_type.662: type = facet_type <@Destroy & @MulWith.1, @MulWith.1(%Int16) where %impl.elem0.8de = %Int16> [concrete]
// CHECK:STDOUT:   %MulWith.type.c18: type = facet_type <@MulWith.1, @MulWith.1(%Int16)> [concrete]
// CHECK:STDOUT:   %MulWith.assoc_type.f15: type = assoc_entity_type @MulWith.1, @MulWith.1(%Int16) [concrete]
// CHECK:STDOUT:   %assoc0.f8b: %MulWith.assoc_type.f15 = assoc_entity element0, imports.%Core.import_ref.59c [concrete]
// CHECK:STDOUT:   %assoc1.eb5: %MulWith.assoc_type.f15 = assoc_entity element1, imports.%Core.import_ref.5a0 [concrete]
// CHECK:STDOUT:   %Op.type.442b68.3: type = fn_type @Op.3 [concrete]
// CHECK:STDOUT:   %Op.72717c.3: %Op.type.442b68.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.6bd: <witness> = custom_witness (%Int16, %Op.72717c.3), @MulWith.1, @MulWith.1(%Int16) [concrete]
// CHECK:STDOUT:   %facet_value.920: %facet_type.662 = facet_value %i16, (%custom_witness.df9cc1.2, %custom_witness.6bd) [concrete]
// CHECK:STDOUT:   %.Self.frozen.05e: %facet_type.a64 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %complete_type.4d3: <witness> = complete_type_witness %facet_type.a64 [concrete]
// CHECK:STDOUT:   %.Self.frozen.as_type.2d1: type = facet_access_type %.Self.frozen.05e [symbolic_self]
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.701: <witness> = lookup_impl_witness %.Self.frozen.05e, @MulWith.1, @MulWith.1(%Int16) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.381: type = impl_witness_access %MulWith.lookup_impl_witness.701, element0 [symbolic_self]
// CHECK:STDOUT:   %pattern_type.90f: type = pattern_type %facet_type.662 [concrete]
// CHECK:STDOUT:   %T.patt.8ad: %pattern_type.90f = symbolic_binding_pattern T, 2 [symbolic]
// CHECK:STDOUT:   %facet_type.0b4: type = facet_type <@Destroy & @DivWith.1, @DivWith.1(%Int16)> [concrete]
// CHECK:STDOUT:   %.Self.9b5: %facet_type.0b4 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.648: <witness> = lookup_impl_witness %.Self.9b5, @DivWith.1, @DivWith.1(%Int16) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.36a: type = impl_witness_access %DivWith.lookup_impl_witness.648, element0 [symbolic_self]
// CHECK:STDOUT:   %facet_type.bd6: type = facet_type <@Destroy & @DivWith.1, @DivWith.1(%Int16) where %impl.elem0.36a = %Int16> [concrete]
// CHECK:STDOUT:   %DivWith.type.6f9: type = facet_type <@DivWith.1, @DivWith.1(%Int16)> [concrete]
// CHECK:STDOUT:   %DivWith.assoc_type.333: type = assoc_entity_type @DivWith.1, @DivWith.1(%Int16) [concrete]
// CHECK:STDOUT:   %assoc0.962: %DivWith.assoc_type.333 = assoc_entity element0, imports.%Core.import_ref.dee [concrete]
// CHECK:STDOUT:   %assoc1.8b5: %DivWith.assoc_type.333 = assoc_entity element1, imports.%Core.import_ref.7845 [concrete]
// CHECK:STDOUT:   %Op.type.442b68.4: type = fn_type @Op.4 [concrete]
// CHECK:STDOUT:   %Op.72717c.4: %Op.type.442b68.4 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.22c: <witness> = custom_witness (%Int16, %Op.72717c.4), @DivWith.1, @DivWith.1(%Int16) [concrete]
// CHECK:STDOUT:   %facet_value.ce6: %facet_type.bd6 = facet_value %i16, (%custom_witness.df9cc1.2, %custom_witness.22c) [concrete]
// CHECK:STDOUT:   %.Self.frozen.ecc: %facet_type.0b4 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %complete_type.acc: <witness> = complete_type_witness %facet_type.0b4 [concrete]
// CHECK:STDOUT:   %.Self.frozen.as_type.851: type = facet_access_type %.Self.frozen.ecc [symbolic_self]
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.6fa: <witness> = lookup_impl_witness %.Self.frozen.ecc, @DivWith.1, @DivWith.1(%Int16) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.e0f: type = impl_witness_access %DivWith.lookup_impl_witness.6fa, element0 [symbolic_self]
// CHECK:STDOUT:   %pattern_type.692: type = pattern_type %facet_type.bd6 [concrete]
// CHECK:STDOUT:   %T.patt.00c: %pattern_type.692 = symbolic_binding_pattern T, 2 [symbolic]
// CHECK:STDOUT:   %facet_type.ee8: type = facet_type <@Destroy & @ModWith.1, @ModWith.1(%Int16)> [concrete]
// CHECK:STDOUT:   %.Self.67f: %facet_type.ee8 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.af3: <witness> = lookup_impl_witness %.Self.67f, @ModWith.1, @ModWith.1(%Int16) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.436: type = impl_witness_access %ModWith.lookup_impl_witness.af3, element0 [symbolic_self]
// CHECK:STDOUT:   %facet_type.a2b: type = facet_type <@Destroy & @ModWith.1, @ModWith.1(%Int16) where %impl.elem0.436 = %Int16> [concrete]
// CHECK:STDOUT:   %ModWith.type.763: type = facet_type <@ModWith.1, @ModWith.1(%Int16)> [concrete]
// CHECK:STDOUT:   %ModWith.assoc_type.e6a: type = assoc_entity_type @ModWith.1, @ModWith.1(%Int16) [concrete]
// CHECK:STDOUT:   %assoc0.2cc: %ModWith.assoc_type.e6a = assoc_entity element0, imports.%Core.import_ref.61a [concrete]
// CHECK:STDOUT:   %assoc1.3dd: %ModWith.assoc_type.e6a = assoc_entity element1, imports.%Core.import_ref.b56 [concrete]
// CHECK:STDOUT:   %Op.type.442b68.5: type = fn_type @Op.5 [concrete]
// CHECK:STDOUT:   %Op.72717c.5: %Op.type.442b68.5 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.9e6: <witness> = custom_witness (%Int16, %Op.72717c.5), @ModWith.1, @ModWith.1(%Int16) [concrete]
// CHECK:STDOUT:   %facet_value.cde: %facet_type.a2b = facet_value %i16, (%custom_witness.df9cc1.2, %custom_witness.9e6) [concrete]
// CHECK:STDOUT:   %.Self.frozen.9a7: %facet_type.ee8 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %complete_type.8fc: <witness> = complete_type_witness %facet_type.ee8 [concrete]
// CHECK:STDOUT:   %.Self.frozen.as_type.954: type = facet_access_type %.Self.frozen.9a7 [symbolic_self]
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.689: <witness> = lookup_impl_witness %.Self.frozen.9a7, @ModWith.1, @ModWith.1(%Int16) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.42c: type = impl_witness_access %ModWith.lookup_impl_witness.689, element0 [symbolic_self]
// CHECK:STDOUT:   %pattern_type.61d: type = pattern_type %facet_type.a2b [concrete]
// CHECK:STDOUT:   %T.patt.a00: %pattern_type.61d = symbolic_binding_pattern T, 2 [symbolic]
// CHECK:STDOUT:   %facet_type.1cc: type = facet_type <@Destroy & @AddWith.1, @AddWith.1(%i16)> [concrete]
// CHECK:STDOUT:   %.Self.a82: %facet_type.1cc = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.e27: <witness> = lookup_impl_witness %.Self.a82, @AddWith.1, @AddWith.1(%i16) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.115: type = impl_witness_access %AddWith.lookup_impl_witness.e27, element0 [symbolic_self]
// CHECK:STDOUT:   %facet_type.b7b: type = facet_type <@Destroy & @AddWith.1, @AddWith.1(%i16) where %impl.elem0.115 = %Int16> [concrete]
// CHECK:STDOUT:   %AddWith.type.a36: type = facet_type <@AddWith.1, @AddWith.1(%i16)> [concrete]
// CHECK:STDOUT:   %AddWith.assoc_type.6bf: type = assoc_entity_type @AddWith.1, @AddWith.1(%i16) [concrete]
// CHECK:STDOUT:   %assoc0.e6b: %AddWith.assoc_type.6bf = assoc_entity element0, imports.%Core.import_ref.91f8 [concrete]
// CHECK:STDOUT:   %assoc1.80e: %AddWith.assoc_type.6bf = assoc_entity element1, imports.%Core.import_ref.3cf [concrete]
// CHECK:STDOUT:   %Int16.Op.type.83af6e.3: type = fn_type @Int16.Op.3 [concrete]
// CHECK:STDOUT:   %Int16.Op.4dca31.3: %Int16.Op.type.83af6e.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b17: <witness> = custom_witness (%Int16, %Int16.Op.4dca31.3), @AddWith.1, @AddWith.1(%i16) [concrete]
// CHECK:STDOUT:   %facet_value.0c7: %facet_type.b7b = facet_value %Int16, (%custom_witness.e7a31c.1, %custom_witness.b17) [concrete]
// CHECK:STDOUT:   %.Self.frozen.e5b: %facet_type.1cc = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %complete_type.5a6: <witness> = complete_type_witness %facet_type.1cc [concrete]
// CHECK:STDOUT:   %.Self.frozen.as_type.b7c: type = facet_access_type %.Self.frozen.e5b [symbolic_self]
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.047: <witness> = lookup_impl_witness %.Self.frozen.e5b, @AddWith.1, @AddWith.1(%i16) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.f6a: type = impl_witness_access %AddWith.lookup_impl_witness.047, element0 [symbolic_self]
// CHECK:STDOUT:   %pattern_type.c51: type = pattern_type %facet_type.b7b [concrete]
// CHECK:STDOUT:   %T.patt.5cb: %pattern_type.c51 = symbolic_binding_pattern T, 2 [symbolic]
// CHECK:STDOUT:   %x.param_patt.3b0: %pattern_type.785 = value_param_pattern [concrete]
// CHECK:STDOUT:   %x.patt.b76: %pattern_type.785 = at_binding_pattern x, %x.param_patt.3b0 [concrete]
// CHECK:STDOUT:   %y.param_patt.4b4: %pattern_type.203 = value_param_pattern [concrete]
// CHECK:STDOUT:   %y.patt.dd3: %pattern_type.203 = at_binding_pattern y, %y.param_patt.4b4 [concrete]
// CHECK:STDOUT:   %facet_type.cb2: type = facet_type <@Destroy & @SubWith.1, @SubWith.1(%i16)> [concrete]
// CHECK:STDOUT:   %.Self.a84: %facet_type.cb2 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.581: <witness> = lookup_impl_witness %.Self.a84, @SubWith.1, @SubWith.1(%i16) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.a77: type = impl_witness_access %SubWith.lookup_impl_witness.581, element0 [symbolic_self]
// CHECK:STDOUT:   %facet_type.a67: type = facet_type <@Destroy & @SubWith.1, @SubWith.1(%i16) where %impl.elem0.a77 = %Int16> [concrete]
// CHECK:STDOUT:   %SubWith.type.08b: type = facet_type <@SubWith.1, @SubWith.1(%i16)> [concrete]
// CHECK:STDOUT:   %SubWith.assoc_type.077: type = assoc_entity_type @SubWith.1, @SubWith.1(%i16) [concrete]
// CHECK:STDOUT:   %assoc0.465: %SubWith.assoc_type.077 = assoc_entity element0, imports.%Core.import_ref.190e [concrete]
// CHECK:STDOUT:   %assoc1.9f7: %SubWith.assoc_type.077 = assoc_entity element1, imports.%Core.import_ref.3a4 [concrete]
// CHECK:STDOUT:   %Int16.Op.type.83af6e.4: type = fn_type @Int16.Op.4 [concrete]
// CHECK:STDOUT:   %Int16.Op.4dca31.4: %Int16.Op.type.83af6e.4 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b07: <witness> = custom_witness (%Int16, %Int16.Op.4dca31.4), @SubWith.1, @SubWith.1(%i16) [concrete]
// CHECK:STDOUT:   %facet_value.33e: %facet_type.a67 = facet_value %Int16, (%custom_witness.e7a31c.1, %custom_witness.b07) [concrete]
// CHECK:STDOUT:   %.Self.frozen.bbc: %facet_type.cb2 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %complete_type.090: <witness> = complete_type_witness %facet_type.cb2 [concrete]
// CHECK:STDOUT:   %.Self.frozen.as_type.ae6: type = facet_access_type %.Self.frozen.bbc [symbolic_self]
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.596: <witness> = lookup_impl_witness %.Self.frozen.bbc, @SubWith.1, @SubWith.1(%i16) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.733: type = impl_witness_access %SubWith.lookup_impl_witness.596, element0 [symbolic_self]
// CHECK:STDOUT:   %pattern_type.486: type = pattern_type %facet_type.a67 [concrete]
// CHECK:STDOUT:   %T.patt.eb2: %pattern_type.486 = symbolic_binding_pattern T, 2 [symbolic]
// CHECK:STDOUT:   %facet_type.67c: type = facet_type <@Destroy & @MulWith.1, @MulWith.1(%i16)> [concrete]
// CHECK:STDOUT:   %.Self.5dd: %facet_type.67c = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.092: <witness> = lookup_impl_witness %.Self.5dd, @MulWith.1, @MulWith.1(%i16) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.163: type = impl_witness_access %MulWith.lookup_impl_witness.092, element0 [symbolic_self]
// CHECK:STDOUT:   %facet_type.d40: type = facet_type <@Destroy & @MulWith.1, @MulWith.1(%i16) where %impl.elem0.163 = %Int16> [concrete]
// CHECK:STDOUT:   %MulWith.type.55d: type = facet_type <@MulWith.1, @MulWith.1(%i16)> [concrete]
// CHECK:STDOUT:   %MulWith.assoc_type.212: type = assoc_entity_type @MulWith.1, @MulWith.1(%i16) [concrete]
// CHECK:STDOUT:   %assoc0.455: %MulWith.assoc_type.212 = assoc_entity element0, imports.%Core.import_ref.59c [concrete]
// CHECK:STDOUT:   %assoc1.2d0: %MulWith.assoc_type.212 = assoc_entity element1, imports.%Core.import_ref.5a0 [concrete]
// CHECK:STDOUT:   %Int16.Op.type.83af6e.5: type = fn_type @Int16.Op.5 [concrete]
// CHECK:STDOUT:   %Int16.Op.4dca31.5: %Int16.Op.type.83af6e.5 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.8bd: <witness> = custom_witness (%Int16, %Int16.Op.4dca31.5), @MulWith.1, @MulWith.1(%i16) [concrete]
// CHECK:STDOUT:   %facet_value.ba4: %facet_type.d40 = facet_value %Int16, (%custom_witness.e7a31c.1, %custom_witness.8bd) [concrete]
// CHECK:STDOUT:   %.Self.frozen.c15: %facet_type.67c = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %complete_type.a8b: <witness> = complete_type_witness %facet_type.67c [concrete]
// CHECK:STDOUT:   %.Self.frozen.as_type.225: type = facet_access_type %.Self.frozen.c15 [symbolic_self]
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.481: <witness> = lookup_impl_witness %.Self.frozen.c15, @MulWith.1, @MulWith.1(%i16) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.574: type = impl_witness_access %MulWith.lookup_impl_witness.481, element0 [symbolic_self]
// CHECK:STDOUT:   %pattern_type.f6f: type = pattern_type %facet_type.d40 [concrete]
// CHECK:STDOUT:   %T.patt.dc7: %pattern_type.f6f = symbolic_binding_pattern T, 2 [symbolic]
// CHECK:STDOUT:   %facet_type.86c8: type = facet_type <@Destroy & @DivWith.1, @DivWith.1(%i16)> [concrete]
// CHECK:STDOUT:   %.Self.d04: %facet_type.86c8 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.6ec: <witness> = lookup_impl_witness %.Self.d04, @DivWith.1, @DivWith.1(%i16) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.58b: type = impl_witness_access %DivWith.lookup_impl_witness.6ec, element0 [symbolic_self]
// CHECK:STDOUT:   %facet_type.541: type = facet_type <@Destroy & @DivWith.1, @DivWith.1(%i16) where %impl.elem0.58b = %Int16> [concrete]
// CHECK:STDOUT:   %DivWith.type.ab0: type = facet_type <@DivWith.1, @DivWith.1(%i16)> [concrete]
// CHECK:STDOUT:   %DivWith.assoc_type.1f2: type = assoc_entity_type @DivWith.1, @DivWith.1(%i16) [concrete]
// CHECK:STDOUT:   %assoc0.be2: %DivWith.assoc_type.1f2 = assoc_entity element0, imports.%Core.import_ref.dee [concrete]
// CHECK:STDOUT:   %assoc1.885: %DivWith.assoc_type.1f2 = assoc_entity element1, imports.%Core.import_ref.7845 [concrete]
// CHECK:STDOUT:   %Int16.Op.type.83af6e.6: type = fn_type @Int16.Op.6 [concrete]
// CHECK:STDOUT:   %Int16.Op.4dca31.6: %Int16.Op.type.83af6e.6 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.963: <witness> = custom_witness (%Int16, %Int16.Op.4dca31.6), @DivWith.1, @DivWith.1(%i16) [concrete]
// CHECK:STDOUT:   %facet_value.521: %facet_type.541 = facet_value %Int16, (%custom_witness.e7a31c.1, %custom_witness.963) [concrete]
// CHECK:STDOUT:   %.Self.frozen.128: %facet_type.86c8 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %complete_type.1cb: <witness> = complete_type_witness %facet_type.86c8 [concrete]
// CHECK:STDOUT:   %.Self.frozen.as_type.ac3: type = facet_access_type %.Self.frozen.128 [symbolic_self]
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.a5b: <witness> = lookup_impl_witness %.Self.frozen.128, @DivWith.1, @DivWith.1(%i16) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.4e9: type = impl_witness_access %DivWith.lookup_impl_witness.a5b, element0 [symbolic_self]
// CHECK:STDOUT:   %pattern_type.80a: type = pattern_type %facet_type.541 [concrete]
// CHECK:STDOUT:   %T.patt.c06: %pattern_type.80a = symbolic_binding_pattern T, 2 [symbolic]
// CHECK:STDOUT:   %facet_type.afdb: type = facet_type <@Destroy & @ModWith.1, @ModWith.1(%i16)> [concrete]
// CHECK:STDOUT:   %.Self.69d: %facet_type.afdb = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.b3b: <witness> = lookup_impl_witness %.Self.69d, @ModWith.1, @ModWith.1(%i16) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.0f8: type = impl_witness_access %ModWith.lookup_impl_witness.b3b, element0 [symbolic_self]
// CHECK:STDOUT:   %facet_type.ee6: type = facet_type <@Destroy & @ModWith.1, @ModWith.1(%i16) where %impl.elem0.0f8 = %Int16> [concrete]
// CHECK:STDOUT:   %ModWith.type.0b2: type = facet_type <@ModWith.1, @ModWith.1(%i16)> [concrete]
// CHECK:STDOUT:   %ModWith.assoc_type.e06: type = assoc_entity_type @ModWith.1, @ModWith.1(%i16) [concrete]
// CHECK:STDOUT:   %assoc0.b5a: %ModWith.assoc_type.e06 = assoc_entity element0, imports.%Core.import_ref.61a [concrete]
// CHECK:STDOUT:   %assoc1.972: %ModWith.assoc_type.e06 = assoc_entity element1, imports.%Core.import_ref.b56 [concrete]
// CHECK:STDOUT:   %Int16.Op.type.83af6e.7: type = fn_type @Int16.Op.7 [concrete]
// CHECK:STDOUT:   %Int16.Op.4dca31.7: %Int16.Op.type.83af6e.7 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.8f4: <witness> = custom_witness (%Int16, %Int16.Op.4dca31.7), @ModWith.1, @ModWith.1(%i16) [concrete]
// CHECK:STDOUT:   %facet_value.ce1: %facet_type.ee6 = facet_value %Int16, (%custom_witness.e7a31c.1, %custom_witness.8f4) [concrete]
// CHECK:STDOUT:   %.Self.frozen.46c: %facet_type.afdb = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %complete_type.1f9: <witness> = complete_type_witness %facet_type.afdb [concrete]
// CHECK:STDOUT:   %.Self.frozen.as_type.12f: type = facet_access_type %.Self.frozen.46c [symbolic_self]
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.5ab: <witness> = lookup_impl_witness %.Self.frozen.46c, @ModWith.1, @ModWith.1(%i16) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.172: type = impl_witness_access %ModWith.lookup_impl_witness.5ab, element0 [symbolic_self]
// CHECK:STDOUT:   %pattern_type.4d8: type = pattern_type %facet_type.ee6 [concrete]
// CHECK:STDOUT:   %T.patt.a7e: %pattern_type.4d8 = symbolic_binding_pattern T, 2 [symbolic]
// CHECK:STDOUT:   %Int32.Op.type.83af6e.1: type = fn_type @Int32.Op.1 [concrete]
// CHECK:STDOUT:   %Int32.Op.4dca31.1: %Int32.Op.type.83af6e.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.e7a31c.2: <witness> = custom_witness (%Int32.Op.4dca31.1), @Destroy [concrete]
// CHECK:STDOUT:   %Destroy.facet.099: %Destroy.type = facet_value %Int32, (%custom_witness.e7a31c.2) [concrete]
// CHECK:STDOUT:   %facet_type.698: type = facet_type <@Destroy & @AddWith.1, @AddWith.1(%Int32)> [concrete]
// CHECK:STDOUT:   %.Self.591: %facet_type.698 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.47a: <witness> = lookup_impl_witness %.Self.591, @AddWith.1, @AddWith.1(%Int32) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.6e6: type = impl_witness_access %AddWith.lookup_impl_witness.47a, element0 [symbolic_self]
// CHECK:STDOUT:   %facet_type.ecf: type = facet_type <@Destroy & @AddWith.1, @AddWith.1(%Int32) where %impl.elem0.6e6 = %Int32> [concrete]
// CHECK:STDOUT:   %AddWith.type.c71: type = facet_type <@AddWith.1, @AddWith.1(%Int32)> [concrete]
// CHECK:STDOUT:   %AddWith.assoc_type.72d: type = assoc_entity_type @AddWith.1, @AddWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %assoc0.c2b: %AddWith.assoc_type.72d = assoc_entity element0, imports.%Core.import_ref.91f8 [concrete]
// CHECK:STDOUT:   %assoc1.0fc: %AddWith.assoc_type.72d = assoc_entity element1, imports.%Core.import_ref.3cf [concrete]
// CHECK:STDOUT:   %Int16.Op.type.83af6e.8: type = fn_type @Int16.Op.8 [concrete]
// CHECK:STDOUT:   %Int16.Op.4dca31.8: %Int16.Op.type.83af6e.8 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.e02e0c.1: <witness> = custom_witness (%Int32, %Int16.Op.4dca31.8), @AddWith.1, @AddWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %facet_value.17e: %facet_type.ecf = facet_value %Int16, (%custom_witness.e7a31c.1, %custom_witness.e02e0c.1) [concrete]
// CHECK:STDOUT:   %.Self.frozen.f1e: %facet_type.698 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %complete_type.224: <witness> = complete_type_witness %facet_type.698 [concrete]
// CHECK:STDOUT:   %.Self.frozen.as_type.440: type = facet_access_type %.Self.frozen.f1e [symbolic_self]
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.7f6: <witness> = lookup_impl_witness %.Self.frozen.f1e, @AddWith.1, @AddWith.1(%Int32) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.7a4: type = impl_witness_access %AddWith.lookup_impl_witness.7f6, element0 [symbolic_self]
// CHECK:STDOUT:   %pattern_type.c976: type = pattern_type %facet_type.ecf [concrete]
// CHECK:STDOUT:   %T.patt.0a8: %pattern_type.c976 = symbolic_binding_pattern T, 2 [symbolic]
// CHECK:STDOUT:   %result.param_patt.e81: %pattern_type.8b4 = value_param_pattern [concrete]
// CHECK:STDOUT:   %result.patt.830: %pattern_type.8b4 = at_binding_pattern result, %result.param_patt.e81 [concrete]
// CHECK:STDOUT:   %y.param_patt.e0c: %pattern_type.8b4 = value_param_pattern [concrete]
// CHECK:STDOUT:   %y.patt.0a3: %pattern_type.8b4 = at_binding_pattern y, %y.param_patt.e0c [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.b76: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.099) [concrete]
// CHECK:STDOUT:   %.043: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.b76, %Destroy.facet.099 [concrete]
// CHECK:STDOUT:   %facet_type.44e: type = facet_type <@Destroy & @SubWith.1, @SubWith.1(%Int32)> [concrete]
// CHECK:STDOUT:   %.Self.8ac: %facet_type.44e = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.f41: <witness> = lookup_impl_witness %.Self.8ac, @SubWith.1, @SubWith.1(%Int32) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.e31: type = impl_witness_access %SubWith.lookup_impl_witness.f41, element0 [symbolic_self]
// CHECK:STDOUT:   %facet_type.630: type = facet_type <@Destroy & @SubWith.1, @SubWith.1(%Int32) where %impl.elem0.e31 = %Int32> [concrete]
// CHECK:STDOUT:   %SubWith.type.c81: type = facet_type <@SubWith.1, @SubWith.1(%Int32)> [concrete]
// CHECK:STDOUT:   %SubWith.assoc_type.68f: type = assoc_entity_type @SubWith.1, @SubWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %assoc0.f11: %SubWith.assoc_type.68f = assoc_entity element0, imports.%Core.import_ref.190e [concrete]
// CHECK:STDOUT:   %assoc1.799: %SubWith.assoc_type.68f = assoc_entity element1, imports.%Core.import_ref.3a4 [concrete]
// CHECK:STDOUT:   %Int16.Op.type.83af6e.9: type = fn_type @Int16.Op.9 [concrete]
// CHECK:STDOUT:   %Int16.Op.4dca31.9: %Int16.Op.type.83af6e.9 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.4dea15.1: <witness> = custom_witness (%Int32, %Int16.Op.4dca31.9), @SubWith.1, @SubWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %facet_value.9eb: %facet_type.630 = facet_value %Int16, (%custom_witness.e7a31c.1, %custom_witness.4dea15.1) [concrete]
// CHECK:STDOUT:   %.Self.frozen.db8: %facet_type.44e = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %complete_type.63f: <witness> = complete_type_witness %facet_type.44e [concrete]
// CHECK:STDOUT:   %.Self.frozen.as_type.068: type = facet_access_type %.Self.frozen.db8 [symbolic_self]
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.32d: <witness> = lookup_impl_witness %.Self.frozen.db8, @SubWith.1, @SubWith.1(%Int32) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.821: type = impl_witness_access %SubWith.lookup_impl_witness.32d, element0 [symbolic_self]
// CHECK:STDOUT:   %pattern_type.c58: type = pattern_type %facet_type.630 [concrete]
// CHECK:STDOUT:   %T.patt.315: %pattern_type.c58 = symbolic_binding_pattern T, 2 [symbolic]
// CHECK:STDOUT:   %facet_type.ff6: type = facet_type <@Destroy & @MulWith.1, @MulWith.1(%Int32)> [concrete]
// CHECK:STDOUT:   %.Self.d4f: %facet_type.ff6 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.07b: <witness> = lookup_impl_witness %.Self.d4f, @MulWith.1, @MulWith.1(%Int32) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.0ea: type = impl_witness_access %MulWith.lookup_impl_witness.07b, element0 [symbolic_self]
// CHECK:STDOUT:   %facet_type.26b: type = facet_type <@Destroy & @MulWith.1, @MulWith.1(%Int32) where %impl.elem0.0ea = %Int32> [concrete]
// CHECK:STDOUT:   %MulWith.type.eeb: type = facet_type <@MulWith.1, @MulWith.1(%Int32)> [concrete]
// CHECK:STDOUT:   %MulWith.assoc_type.083: type = assoc_entity_type @MulWith.1, @MulWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %assoc0.fc0: %MulWith.assoc_type.083 = assoc_entity element0, imports.%Core.import_ref.59c [concrete]
// CHECK:STDOUT:   %assoc1.019: %MulWith.assoc_type.083 = assoc_entity element1, imports.%Core.import_ref.5a0 [concrete]
// CHECK:STDOUT:   %Int16.Op.type.83af6e.10: type = fn_type @Int16.Op.10 [concrete]
// CHECK:STDOUT:   %Int16.Op.4dca31.10: %Int16.Op.type.83af6e.10 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.337c88.1: <witness> = custom_witness (%Int32, %Int16.Op.4dca31.10), @MulWith.1, @MulWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %facet_value.9ae: %facet_type.26b = facet_value %Int16, (%custom_witness.e7a31c.1, %custom_witness.337c88.1) [concrete]
// CHECK:STDOUT:   %.Self.frozen.be6: %facet_type.ff6 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %complete_type.3fe: <witness> = complete_type_witness %facet_type.ff6 [concrete]
// CHECK:STDOUT:   %.Self.frozen.as_type.aaf: type = facet_access_type %.Self.frozen.be6 [symbolic_self]
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.4cf: <witness> = lookup_impl_witness %.Self.frozen.be6, @MulWith.1, @MulWith.1(%Int32) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.167: type = impl_witness_access %MulWith.lookup_impl_witness.4cf, element0 [symbolic_self]
// CHECK:STDOUT:   %pattern_type.354: type = pattern_type %facet_type.26b [concrete]
// CHECK:STDOUT:   %T.patt.4b7: %pattern_type.354 = symbolic_binding_pattern T, 2 [symbolic]
// CHECK:STDOUT:   %facet_type.d41: type = facet_type <@Destroy & @DivWith.1, @DivWith.1(%Int32)> [concrete]
// CHECK:STDOUT:   %.Self.13f: %facet_type.d41 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.abd: <witness> = lookup_impl_witness %.Self.13f, @DivWith.1, @DivWith.1(%Int32) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.1a6: type = impl_witness_access %DivWith.lookup_impl_witness.abd, element0 [symbolic_self]
// CHECK:STDOUT:   %facet_type.1a8: type = facet_type <@Destroy & @DivWith.1, @DivWith.1(%Int32) where %impl.elem0.1a6 = %Int32> [concrete]
// CHECK:STDOUT:   %DivWith.type.fb1: type = facet_type <@DivWith.1, @DivWith.1(%Int32)> [concrete]
// CHECK:STDOUT:   %DivWith.assoc_type.725: type = assoc_entity_type @DivWith.1, @DivWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %assoc0.2da: %DivWith.assoc_type.725 = assoc_entity element0, imports.%Core.import_ref.dee [concrete]
// CHECK:STDOUT:   %assoc1.580: %DivWith.assoc_type.725 = assoc_entity element1, imports.%Core.import_ref.7845 [concrete]
// CHECK:STDOUT:   %Int16.Op.type.83af6e.11: type = fn_type @Int16.Op.11 [concrete]
// CHECK:STDOUT:   %Int16.Op.4dca31.11: %Int16.Op.type.83af6e.11 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.3eb227.1: <witness> = custom_witness (%Int32, %Int16.Op.4dca31.11), @DivWith.1, @DivWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %facet_value.38c: %facet_type.1a8 = facet_value %Int16, (%custom_witness.e7a31c.1, %custom_witness.3eb227.1) [concrete]
// CHECK:STDOUT:   %.Self.frozen.be1: %facet_type.d41 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %complete_type.11b: <witness> = complete_type_witness %facet_type.d41 [concrete]
// CHECK:STDOUT:   %.Self.frozen.as_type.88a: type = facet_access_type %.Self.frozen.be1 [symbolic_self]
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.385: <witness> = lookup_impl_witness %.Self.frozen.be1, @DivWith.1, @DivWith.1(%Int32) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.b42: type = impl_witness_access %DivWith.lookup_impl_witness.385, element0 [symbolic_self]
// CHECK:STDOUT:   %pattern_type.761: type = pattern_type %facet_type.1a8 [concrete]
// CHECK:STDOUT:   %T.patt.f94: %pattern_type.761 = symbolic_binding_pattern T, 2 [symbolic]
// CHECK:STDOUT:   %facet_type.ea8: type = facet_type <@Destroy & @ModWith.1, @ModWith.1(%Int32)> [concrete]
// CHECK:STDOUT:   %.Self.008: %facet_type.ea8 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.557: <witness> = lookup_impl_witness %.Self.008, @ModWith.1, @ModWith.1(%Int32) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.cdb: type = impl_witness_access %ModWith.lookup_impl_witness.557, element0 [symbolic_self]
// CHECK:STDOUT:   %facet_type.c8c: type = facet_type <@Destroy & @ModWith.1, @ModWith.1(%Int32) where %impl.elem0.cdb = %Int32> [concrete]
// CHECK:STDOUT:   %ModWith.type.253: type = facet_type <@ModWith.1, @ModWith.1(%Int32)> [concrete]
// CHECK:STDOUT:   %ModWith.assoc_type.47d: type = assoc_entity_type @ModWith.1, @ModWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %assoc0.45c: %ModWith.assoc_type.47d = assoc_entity element0, imports.%Core.import_ref.61a [concrete]
// CHECK:STDOUT:   %assoc1.878: %ModWith.assoc_type.47d = assoc_entity element1, imports.%Core.import_ref.b56 [concrete]
// CHECK:STDOUT:   %Int16.Op.type.83af6e.12: type = fn_type @Int16.Op.12 [concrete]
// CHECK:STDOUT:   %Int16.Op.4dca31.12: %Int16.Op.type.83af6e.12 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.501c6d.1: <witness> = custom_witness (%Int32, %Int16.Op.4dca31.12), @ModWith.1, @ModWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %facet_value.68d: %facet_type.c8c = facet_value %Int16, (%custom_witness.e7a31c.1, %custom_witness.501c6d.1) [concrete]
// CHECK:STDOUT:   %.Self.frozen.4b6: %facet_type.ea8 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %complete_type.2aa: <witness> = complete_type_witness %facet_type.ea8 [concrete]
// CHECK:STDOUT:   %.Self.frozen.as_type.080: type = facet_access_type %.Self.frozen.4b6 [symbolic_self]
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.fe2: <witness> = lookup_impl_witness %.Self.frozen.4b6, @ModWith.1, @ModWith.1(%Int32) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.e60: type = impl_witness_access %ModWith.lookup_impl_witness.fe2, element0 [symbolic_self]
// CHECK:STDOUT:   %pattern_type.c9a: type = pattern_type %facet_type.c8c [concrete]
// CHECK:STDOUT:   %T.patt.b4d: %pattern_type.c9a = symbolic_binding_pattern T, 2 [symbolic]
// CHECK:STDOUT:   %facet_type.37e: type = facet_type <@Destroy & @AddWith.1, @AddWith.1(%Int16) where %impl.elem0.28c = %Int32> [concrete]
// CHECK:STDOUT:   %Int32.Op.type.83af6e.2: type = fn_type @Int32.Op.2 [concrete]
// CHECK:STDOUT:   %Int32.Op.4dca31.2: %Int32.Op.type.83af6e.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.15f: <witness> = custom_witness (%Int32, %Int32.Op.4dca31.2), @AddWith.1, @AddWith.1(%Int16) [concrete]
// CHECK:STDOUT:   %facet_value.6ea: %facet_type.37e = facet_value %Int32, (%custom_witness.e7a31c.2, %custom_witness.15f) [concrete]
// CHECK:STDOUT:   %pattern_type.363: type = pattern_type %facet_type.37e [concrete]
// CHECK:STDOUT:   %T.patt.ccd: %pattern_type.363 = symbolic_binding_pattern T, 2 [symbolic]
// CHECK:STDOUT:   %x.param_patt.27c: %pattern_type.8b4 = value_param_pattern [concrete]
// CHECK:STDOUT:   %x.patt.4c9: %pattern_type.8b4 = at_binding_pattern x, %x.param_patt.27c [concrete]
// CHECK:STDOUT:   %facet_type.983: type = facet_type <@Destroy & @SubWith.1, @SubWith.1(%Int16) where %impl.elem0.4a0 = %Int32> [concrete]
// CHECK:STDOUT:   %Int32.Op.type.83af6e.3: type = fn_type @Int32.Op.3 [concrete]
// CHECK:STDOUT:   %Int32.Op.4dca31.3: %Int32.Op.type.83af6e.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.bb4: <witness> = custom_witness (%Int32, %Int32.Op.4dca31.3), @SubWith.1, @SubWith.1(%Int16) [concrete]
// CHECK:STDOUT:   %facet_value.bcc: %facet_type.983 = facet_value %Int32, (%custom_witness.e7a31c.2, %custom_witness.bb4) [concrete]
// CHECK:STDOUT:   %pattern_type.af3: type = pattern_type %facet_type.983 [concrete]
// CHECK:STDOUT:   %T.patt.14c: %pattern_type.af3 = symbolic_binding_pattern T, 2 [symbolic]
// CHECK:STDOUT:   %facet_type.f48: type = facet_type <@Destroy & @MulWith.1, @MulWith.1(%Int16) where %impl.elem0.8de = %Int32> [concrete]
// CHECK:STDOUT:   %Int32.Op.type.83af6e.4: type = fn_type @Int32.Op.4 [concrete]
// CHECK:STDOUT:   %Int32.Op.4dca31.4: %Int32.Op.type.83af6e.4 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.0ab: <witness> = custom_witness (%Int32, %Int32.Op.4dca31.4), @MulWith.1, @MulWith.1(%Int16) [concrete]
// CHECK:STDOUT:   %facet_value.795: %facet_type.f48 = facet_value %Int32, (%custom_witness.e7a31c.2, %custom_witness.0ab) [concrete]
// CHECK:STDOUT:   %pattern_type.02f: type = pattern_type %facet_type.f48 [concrete]
// CHECK:STDOUT:   %T.patt.a9e: %pattern_type.02f = symbolic_binding_pattern T, 2 [symbolic]
// CHECK:STDOUT:   %facet_type.50d: type = facet_type <@Destroy & @DivWith.1, @DivWith.1(%Int16) where %impl.elem0.36a = %Int32> [concrete]
// CHECK:STDOUT:   %Int32.Op.type.83af6e.5: type = fn_type @Int32.Op.5 [concrete]
// CHECK:STDOUT:   %Int32.Op.4dca31.5: %Int32.Op.type.83af6e.5 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.3ec: <witness> = custom_witness (%Int32, %Int32.Op.4dca31.5), @DivWith.1, @DivWith.1(%Int16) [concrete]
// CHECK:STDOUT:   %facet_value.c8d: %facet_type.50d = facet_value %Int32, (%custom_witness.e7a31c.2, %custom_witness.3ec) [concrete]
// CHECK:STDOUT:   %pattern_type.9af: type = pattern_type %facet_type.50d [concrete]
// CHECK:STDOUT:   %T.patt.dfa: %pattern_type.9af = symbolic_binding_pattern T, 2 [symbolic]
// CHECK:STDOUT:   %facet_type.b37: type = facet_type <@Destroy & @ModWith.1, @ModWith.1(%Int16) where %impl.elem0.436 = %Int32> [concrete]
// CHECK:STDOUT:   %Int32.Op.type.83af6e.6: type = fn_type @Int32.Op.6 [concrete]
// CHECK:STDOUT:   %Int32.Op.4dca31.6: %Int32.Op.type.83af6e.6 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.de0: <witness> = custom_witness (%Int32, %Int32.Op.4dca31.6), @ModWith.1, @ModWith.1(%Int16) [concrete]
// CHECK:STDOUT:   %facet_value.da8: %facet_type.b37 = facet_value %Int32, (%custom_witness.e7a31c.2, %custom_witness.de0) [concrete]
// CHECK:STDOUT:   %pattern_type.448: type = pattern_type %facet_type.b37 [concrete]
// CHECK:STDOUT:   %T.patt.528: %pattern_type.448 = symbolic_binding_pattern T, 2 [symbolic]
// CHECK:STDOUT:   %Int32.Op.type.83af6e.7: type = fn_type @Int32.Op.7 [concrete]
// CHECK:STDOUT:   %Int32.Op.4dca31.7: %Int32.Op.type.83af6e.7 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.e02e0c.2: <witness> = custom_witness (%Int32, %Int32.Op.4dca31.7), @AddWith.1, @AddWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %facet_value.f1a: %facet_type.ecf = facet_value %Int32, (%custom_witness.e7a31c.2, %custom_witness.e02e0c.2) [concrete]
// CHECK:STDOUT:   %Int32.Op.type.83af6e.8: type = fn_type @Int32.Op.8 [concrete]
// CHECK:STDOUT:   %Int32.Op.4dca31.8: %Int32.Op.type.83af6e.8 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.4dea15.2: <witness> = custom_witness (%Int32, %Int32.Op.4dca31.8), @SubWith.1, @SubWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %facet_value.6fc: %facet_type.630 = facet_value %Int32, (%custom_witness.e7a31c.2, %custom_witness.4dea15.2) [concrete]
// CHECK:STDOUT:   %Int32.Op.type.83af6e.9: type = fn_type @Int32.Op.9 [concrete]
// CHECK:STDOUT:   %Int32.Op.4dca31.9: %Int32.Op.type.83af6e.9 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.337c88.2: <witness> = custom_witness (%Int32, %Int32.Op.4dca31.9), @MulWith.1, @MulWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %facet_value.c80: %facet_type.26b = facet_value %Int32, (%custom_witness.e7a31c.2, %custom_witness.337c88.2) [concrete]
// CHECK:STDOUT:   %Int32.Op.type.83af6e.10: type = fn_type @Int32.Op.10 [concrete]
// CHECK:STDOUT:   %Int32.Op.4dca31.10: %Int32.Op.type.83af6e.10 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.3eb227.2: <witness> = custom_witness (%Int32, %Int32.Op.4dca31.10), @DivWith.1, @DivWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %facet_value.b2c: %facet_type.1a8 = facet_value %Int32, (%custom_witness.e7a31c.2, %custom_witness.3eb227.2) [concrete]
// CHECK:STDOUT:   %Int32.Op.type.83af6e.11: type = fn_type @Int32.Op.11 [concrete]
// CHECK:STDOUT:   %Int32.Op.4dca31.11: %Int32.Op.type.83af6e.11 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.501c6d.2: <witness> = custom_witness (%Int32, %Int32.Op.4dca31.11), @ModWith.1, @ModWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %facet_value.7ec: %facet_type.c8c = facet_value %Int32, (%custom_witness.e7a31c.2, %custom_witness.501c6d.2) [concrete]
// CHECK:STDOUT:   %Int64.Op.type.83af6e.1: type = fn_type @Int64.Op.1 [concrete]
// CHECK:STDOUT:   %Int64.Op.4dca31.1: %Int64.Op.type.83af6e.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.e7a31c.3: <witness> = custom_witness (%Int64.Op.4dca31.1), @Destroy [concrete]
// CHECK:STDOUT:   %Destroy.facet.191: %Destroy.type = facet_value %Int64, (%custom_witness.e7a31c.3) [concrete]
// CHECK:STDOUT:   %facet_type.afd1: type = facet_type <@Destroy & @AddWith.1, @AddWith.1(%Int64)> [concrete]
// CHECK:STDOUT:   %.Self.d7b: %facet_type.afd1 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.085: <witness> = lookup_impl_witness %.Self.d7b, @AddWith.1, @AddWith.1(%Int64) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.7e8: type = impl_witness_access %AddWith.lookup_impl_witness.085, element0 [symbolic_self]
// CHECK:STDOUT:   %facet_type.522: type = facet_type <@Destroy & @AddWith.1, @AddWith.1(%Int64) where %impl.elem0.7e8 = %Int64> [concrete]
// CHECK:STDOUT:   %y.param_patt.a08: %pattern_type.4aa = value_param_pattern [concrete]
// CHECK:STDOUT:   %y.patt.63b: %pattern_type.4aa = at_binding_pattern y, %y.param_patt.a08 [concrete]
// CHECK:STDOUT:   %AddWith.type.b5b: type = facet_type <@AddWith.1, @AddWith.1(%Int64)> [concrete]
// CHECK:STDOUT:   %AddWith.assoc_type.0c5: type = assoc_entity_type @AddWith.1, @AddWith.1(%Int64) [concrete]
// CHECK:STDOUT:   %assoc0.947: %AddWith.assoc_type.0c5 = assoc_entity element0, imports.%Core.import_ref.91f8 [concrete]
// CHECK:STDOUT:   %assoc1.18c: %AddWith.assoc_type.0c5 = assoc_entity element1, imports.%Core.import_ref.3cf [concrete]
// CHECK:STDOUT:   %Op.type.442b68.6: type = fn_type @Op.6 [concrete]
// CHECK:STDOUT:   %Op.72717c.6: %Op.type.442b68.6 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.68d: <witness> = custom_witness (%Int64, %Op.72717c.6), @AddWith.1, @AddWith.1(%Int64) [concrete]
// CHECK:STDOUT:   %facet_value.178: %facet_type.522 = facet_value %Int32, (%custom_witness.e7a31c.2, %custom_witness.68d) [concrete]
// CHECK:STDOUT:   %.Self.frozen.2e1: %facet_type.afd1 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %complete_type.412: <witness> = complete_type_witness %facet_type.afd1 [concrete]
// CHECK:STDOUT:   %.Self.frozen.as_type.d9c: type = facet_access_type %.Self.frozen.2e1 [symbolic_self]
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.bc7: <witness> = lookup_impl_witness %.Self.frozen.2e1, @AddWith.1, @AddWith.1(%Int64) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.a41: type = impl_witness_access %AddWith.lookup_impl_witness.bc7, element0 [symbolic_self]
// CHECK:STDOUT:   %pattern_type.613: type = pattern_type %facet_type.522 [concrete]
// CHECK:STDOUT:   %T.patt.f5c: %pattern_type.613 = symbolic_binding_pattern T, 2 [symbolic]
// CHECK:STDOUT:   %result.param_patt.1e1: %pattern_type.4aa = value_param_pattern [concrete]
// CHECK:STDOUT:   %result.patt.41f: %pattern_type.4aa = at_binding_pattern result, %result.param_patt.1e1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.c2c: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.191) [concrete]
// CHECK:STDOUT:   %.c2e: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.c2c, %Destroy.facet.191 [concrete]
// CHECK:STDOUT:   %facet_type.502: type = facet_type <@Destroy & @SubWith.1, @SubWith.1(%Int64)> [concrete]
// CHECK:STDOUT:   %.Self.074: %facet_type.502 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.0c4: <witness> = lookup_impl_witness %.Self.074, @SubWith.1, @SubWith.1(%Int64) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.68d: type = impl_witness_access %SubWith.lookup_impl_witness.0c4, element0 [symbolic_self]
// CHECK:STDOUT:   %facet_type.e5e: type = facet_type <@Destroy & @SubWith.1, @SubWith.1(%Int64) where %impl.elem0.68d = %Int64> [concrete]
// CHECK:STDOUT:   %SubWith.type.a46: type = facet_type <@SubWith.1, @SubWith.1(%Int64)> [concrete]
// CHECK:STDOUT:   %SubWith.assoc_type.e16: type = assoc_entity_type @SubWith.1, @SubWith.1(%Int64) [concrete]
// CHECK:STDOUT:   %assoc0.88a: %SubWith.assoc_type.e16 = assoc_entity element0, imports.%Core.import_ref.190e [concrete]
// CHECK:STDOUT:   %assoc1.3b1: %SubWith.assoc_type.e16 = assoc_entity element1, imports.%Core.import_ref.3a4 [concrete]
// CHECK:STDOUT:   %Op.type.442b68.7: type = fn_type @Op.7 [concrete]
// CHECK:STDOUT:   %Op.72717c.7: %Op.type.442b68.7 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.1fa: <witness> = custom_witness (%Int64, %Op.72717c.7), @SubWith.1, @SubWith.1(%Int64) [concrete]
// CHECK:STDOUT:   %facet_value.c78: %facet_type.e5e = facet_value %Int32, (%custom_witness.e7a31c.2, %custom_witness.1fa) [concrete]
// CHECK:STDOUT:   %.Self.frozen.da7: %facet_type.502 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %complete_type.181: <witness> = complete_type_witness %facet_type.502 [concrete]
// CHECK:STDOUT:   %.Self.frozen.as_type.436: type = facet_access_type %.Self.frozen.da7 [symbolic_self]
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.362: <witness> = lookup_impl_witness %.Self.frozen.da7, @SubWith.1, @SubWith.1(%Int64) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.f3a: type = impl_witness_access %SubWith.lookup_impl_witness.362, element0 [symbolic_self]
// CHECK:STDOUT:   %pattern_type.41a: type = pattern_type %facet_type.e5e [concrete]
// CHECK:STDOUT:   %T.patt.282: %pattern_type.41a = symbolic_binding_pattern T, 2 [symbolic]
// CHECK:STDOUT:   %facet_type.770: type = facet_type <@Destroy & @MulWith.1, @MulWith.1(%Int64)> [concrete]
// CHECK:STDOUT:   %.Self.4e1: %facet_type.770 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.55a: <witness> = lookup_impl_witness %.Self.4e1, @MulWith.1, @MulWith.1(%Int64) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.641: type = impl_witness_access %MulWith.lookup_impl_witness.55a, element0 [symbolic_self]
// CHECK:STDOUT:   %facet_type.9a3: type = facet_type <@Destroy & @MulWith.1, @MulWith.1(%Int64) where %impl.elem0.641 = %Int64> [concrete]
// CHECK:STDOUT:   %MulWith.type.76f: type = facet_type <@MulWith.1, @MulWith.1(%Int64)> [concrete]
// CHECK:STDOUT:   %MulWith.assoc_type.a37: type = assoc_entity_type @MulWith.1, @MulWith.1(%Int64) [concrete]
// CHECK:STDOUT:   %assoc0.210: %MulWith.assoc_type.a37 = assoc_entity element0, imports.%Core.import_ref.59c [concrete]
// CHECK:STDOUT:   %assoc1.cd1: %MulWith.assoc_type.a37 = assoc_entity element1, imports.%Core.import_ref.5a0 [concrete]
// CHECK:STDOUT:   %Op.type.442b68.8: type = fn_type @Op.8 [concrete]
// CHECK:STDOUT:   %Op.72717c.8: %Op.type.442b68.8 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.99d: <witness> = custom_witness (%Int64, %Op.72717c.8), @MulWith.1, @MulWith.1(%Int64) [concrete]
// CHECK:STDOUT:   %facet_value.34a: %facet_type.9a3 = facet_value %Int32, (%custom_witness.e7a31c.2, %custom_witness.99d) [concrete]
// CHECK:STDOUT:   %.Self.frozen.f85: %facet_type.770 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %complete_type.16e: <witness> = complete_type_witness %facet_type.770 [concrete]
// CHECK:STDOUT:   %.Self.frozen.as_type.f77: type = facet_access_type %.Self.frozen.f85 [symbolic_self]
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.472: <witness> = lookup_impl_witness %.Self.frozen.f85, @MulWith.1, @MulWith.1(%Int64) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.c8c: type = impl_witness_access %MulWith.lookup_impl_witness.472, element0 [symbolic_self]
// CHECK:STDOUT:   %pattern_type.346: type = pattern_type %facet_type.9a3 [concrete]
// CHECK:STDOUT:   %T.patt.6f8: %pattern_type.346 = symbolic_binding_pattern T, 2 [symbolic]
// CHECK:STDOUT:   %facet_type.c98: type = facet_type <@Destroy & @DivWith.1, @DivWith.1(%Int64)> [concrete]
// CHECK:STDOUT:   %.Self.f70: %facet_type.c98 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.4c3: <witness> = lookup_impl_witness %.Self.f70, @DivWith.1, @DivWith.1(%Int64) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.622: type = impl_witness_access %DivWith.lookup_impl_witness.4c3, element0 [symbolic_self]
// CHECK:STDOUT:   %facet_type.641: type = facet_type <@Destroy & @DivWith.1, @DivWith.1(%Int64) where %impl.elem0.622 = %Int64> [concrete]
// CHECK:STDOUT:   %DivWith.type.095: type = facet_type <@DivWith.1, @DivWith.1(%Int64)> [concrete]
// CHECK:STDOUT:   %DivWith.assoc_type.8a4: type = assoc_entity_type @DivWith.1, @DivWith.1(%Int64) [concrete]
// CHECK:STDOUT:   %assoc0.982: %DivWith.assoc_type.8a4 = assoc_entity element0, imports.%Core.import_ref.dee [concrete]
// CHECK:STDOUT:   %assoc1.c76: %DivWith.assoc_type.8a4 = assoc_entity element1, imports.%Core.import_ref.7845 [concrete]
// CHECK:STDOUT:   %Op.type.442b68.9: type = fn_type @Op.9 [concrete]
// CHECK:STDOUT:   %Op.72717c.9: %Op.type.442b68.9 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.a8f: <witness> = custom_witness (%Int64, %Op.72717c.9), @DivWith.1, @DivWith.1(%Int64) [concrete]
// CHECK:STDOUT:   %facet_value.dd9: %facet_type.641 = facet_value %Int32, (%custom_witness.e7a31c.2, %custom_witness.a8f) [concrete]
// CHECK:STDOUT:   %.Self.frozen.8cb: %facet_type.c98 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %complete_type.621: <witness> = complete_type_witness %facet_type.c98 [concrete]
// CHECK:STDOUT:   %.Self.frozen.as_type.bdb: type = facet_access_type %.Self.frozen.8cb [symbolic_self]
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.a82: <witness> = lookup_impl_witness %.Self.frozen.8cb, @DivWith.1, @DivWith.1(%Int64) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.709: type = impl_witness_access %DivWith.lookup_impl_witness.a82, element0 [symbolic_self]
// CHECK:STDOUT:   %pattern_type.719: type = pattern_type %facet_type.641 [concrete]
// CHECK:STDOUT:   %T.patt.e0b: %pattern_type.719 = symbolic_binding_pattern T, 2 [symbolic]
// CHECK:STDOUT:   %facet_type.f26: type = facet_type <@Destroy & @ModWith.1, @ModWith.1(%Int64)> [concrete]
// CHECK:STDOUT:   %.Self.494: %facet_type.f26 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.103: <witness> = lookup_impl_witness %.Self.494, @ModWith.1, @ModWith.1(%Int64) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.ffb: type = impl_witness_access %ModWith.lookup_impl_witness.103, element0 [symbolic_self]
// CHECK:STDOUT:   %facet_type.096: type = facet_type <@Destroy & @ModWith.1, @ModWith.1(%Int64) where %impl.elem0.ffb = %Int64> [concrete]
// CHECK:STDOUT:   %ModWith.type.204: type = facet_type <@ModWith.1, @ModWith.1(%Int64)> [concrete]
// CHECK:STDOUT:   %ModWith.assoc_type.1f7: type = assoc_entity_type @ModWith.1, @ModWith.1(%Int64) [concrete]
// CHECK:STDOUT:   %assoc0.0a6: %ModWith.assoc_type.1f7 = assoc_entity element0, imports.%Core.import_ref.61a [concrete]
// CHECK:STDOUT:   %assoc1.ee5: %ModWith.assoc_type.1f7 = assoc_entity element1, imports.%Core.import_ref.b56 [concrete]
// CHECK:STDOUT:   %Op.type.442b68.10: type = fn_type @Op.10 [concrete]
// CHECK:STDOUT:   %Op.72717c.10: %Op.type.442b68.10 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.972: <witness> = custom_witness (%Int64, %Op.72717c.10), @ModWith.1, @ModWith.1(%Int64) [concrete]
// CHECK:STDOUT:   %facet_value.ea3: %facet_type.096 = facet_value %Int32, (%custom_witness.e7a31c.2, %custom_witness.972) [concrete]
// CHECK:STDOUT:   %.Self.frozen.ac6: %facet_type.f26 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %complete_type.359: <witness> = complete_type_witness %facet_type.f26 [concrete]
// CHECK:STDOUT:   %.Self.frozen.as_type.17a: type = facet_access_type %.Self.frozen.ac6 [symbolic_self]
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.072: <witness> = lookup_impl_witness %.Self.frozen.ac6, @ModWith.1, @ModWith.1(%Int64) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.220f: type = impl_witness_access %ModWith.lookup_impl_witness.072, element0 [symbolic_self]
// CHECK:STDOUT:   %pattern_type.df3: type = pattern_type %facet_type.096 [concrete]
// CHECK:STDOUT:   %T.patt.c87: %pattern_type.df3 = symbolic_binding_pattern T, 2 [symbolic]
// CHECK:STDOUT:   %facet_type.840: type = facet_type <@Destroy & @AddWith.1, @AddWith.1(%Int32) where %impl.elem0.6e6 = %Int64> [concrete]
// CHECK:STDOUT:   %Op.type.442b68.11: type = fn_type @Op.11 [concrete]
// CHECK:STDOUT:   %Op.72717c.11: %Op.type.442b68.11 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.98f: <witness> = custom_witness (%Int64, %Op.72717c.11), @AddWith.1, @AddWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %facet_value.393: %facet_type.840 = facet_value %Int64, (%custom_witness.e7a31c.3, %custom_witness.98f) [concrete]
// CHECK:STDOUT:   %pattern_type.46d: type = pattern_type %facet_type.840 [concrete]
// CHECK:STDOUT:   %T.patt.211: %pattern_type.46d = symbolic_binding_pattern T, 2 [symbolic]
// CHECK:STDOUT:   %x.param_patt.e68: %pattern_type.4aa = value_param_pattern [concrete]
// CHECK:STDOUT:   %x.patt.52a: %pattern_type.4aa = at_binding_pattern x, %x.param_patt.e68 [concrete]
// CHECK:STDOUT:   %facet_type.99d: type = facet_type <@Destroy & @SubWith.1, @SubWith.1(%Int32) where %impl.elem0.e31 = %Int64> [concrete]
// CHECK:STDOUT:   %Op.type.442b68.12: type = fn_type @Op.12 [concrete]
// CHECK:STDOUT:   %Op.72717c.12: %Op.type.442b68.12 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.5c3: <witness> = custom_witness (%Int64, %Op.72717c.12), @SubWith.1, @SubWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %facet_value.e02: %facet_type.99d = facet_value %Int64, (%custom_witness.e7a31c.3, %custom_witness.5c3) [concrete]
// CHECK:STDOUT:   %pattern_type.b19: type = pattern_type %facet_type.99d [concrete]
// CHECK:STDOUT:   %T.patt.e50: %pattern_type.b19 = symbolic_binding_pattern T, 2 [symbolic]
// CHECK:STDOUT:   %facet_type.5e8: type = facet_type <@Destroy & @MulWith.1, @MulWith.1(%Int32) where %impl.elem0.0ea = %Int64> [concrete]
// CHECK:STDOUT:   %Op.type.442b68.13: type = fn_type @Op.13 [concrete]
// CHECK:STDOUT:   %Op.72717c.13: %Op.type.442b68.13 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.ac0: <witness> = custom_witness (%Int64, %Op.72717c.13), @MulWith.1, @MulWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %facet_value.dda: %facet_type.5e8 = facet_value %Int64, (%custom_witness.e7a31c.3, %custom_witness.ac0) [concrete]
// CHECK:STDOUT:   %pattern_type.d24: type = pattern_type %facet_type.5e8 [concrete]
// CHECK:STDOUT:   %T.patt.54a: %pattern_type.d24 = symbolic_binding_pattern T, 2 [symbolic]
// CHECK:STDOUT:   %facet_type.97b: type = facet_type <@Destroy & @DivWith.1, @DivWith.1(%Int32) where %impl.elem0.1a6 = %Int64> [concrete]
// CHECK:STDOUT:   %Op.type.442b68.14: type = fn_type @Op.14 [concrete]
// CHECK:STDOUT:   %Op.72717c.14: %Op.type.442b68.14 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.c98: <witness> = custom_witness (%Int64, %Op.72717c.14), @DivWith.1, @DivWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %facet_value.886: %facet_type.97b = facet_value %Int64, (%custom_witness.e7a31c.3, %custom_witness.c98) [concrete]
// CHECK:STDOUT:   %pattern_type.6b7: type = pattern_type %facet_type.97b [concrete]
// CHECK:STDOUT:   %T.patt.0b0: %pattern_type.6b7 = symbolic_binding_pattern T, 2 [symbolic]
// CHECK:STDOUT:   %facet_type.546: type = facet_type <@Destroy & @ModWith.1, @ModWith.1(%Int32) where %impl.elem0.cdb = %Int64> [concrete]
// CHECK:STDOUT:   %Op.type.442b68.15: type = fn_type @Op.15 [concrete]
// CHECK:STDOUT:   %Op.72717c.15: %Op.type.442b68.15 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.e66: <witness> = custom_witness (%Int64, %Op.72717c.15), @ModWith.1, @ModWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %facet_value.20a: %facet_type.546 = facet_value %Int64, (%custom_witness.e7a31c.3, %custom_witness.e66) [concrete]
// CHECK:STDOUT:   %pattern_type.5db: type = pattern_type %facet_type.546 [concrete]
// CHECK:STDOUT:   %T.patt.ed0: %pattern_type.5db = symbolic_binding_pattern T, 2 [symbolic]
// CHECK:STDOUT:   %U.patt.d47: %pattern_type.98f = symbolic_binding_pattern U, 0 [symbolic]
// CHECK:STDOUT:   %U.67d: type = symbolic_binding U, 0 [symbolic]
// CHECK:STDOUT:   %AddAssignWith.type.2c8890.1: type = facet_type <@AddAssignWith.1, @AddAssignWith.1(%Other)> [symbolic]
// CHECK:STDOUT:   %Self.792732.1: %AddAssignWith.type.2c8890.1 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %AddAssignWith.assoc_type.8ddaf4.1: type = assoc_entity_type @AddAssignWith.1, @AddAssignWith.1(%Other) [symbolic]
// CHECK:STDOUT:   %AddAssignWith.WithSelf.Op.type.cd574e.1: type = fn_type @AddAssignWith.WithSelf.Op, @AddAssignWith.WithSelf(%Other, %Self.792732.1) [symbolic]
// CHECK:STDOUT:   %AddAssignWith.WithSelf.Op.106889.1: %AddAssignWith.WithSelf.Op.type.cd574e.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %AddAssignWith.type.2c8890.2: type = facet_type <@AddAssignWith.1, @AddAssignWith.1(%U.67d)> [symbolic]
// CHECK:STDOUT:   %pattern_type.82d: type = pattern_type %AddAssignWith.type.2c8890.2 [symbolic]
// CHECK:STDOUT:   %T.patt.a22: %pattern_type.82d = symbolic_binding_pattern T, 1 [symbolic]
// CHECK:STDOUT:   %T.792: %AddAssignWith.type.2c8890.2 = symbolic_binding T, 1 [symbolic]
// CHECK:STDOUT:   %AddAssignWith.assoc_type.8ddaf4.2: type = assoc_entity_type @AddAssignWith.1, @AddAssignWith.1(%U.67d) [symbolic]
// CHECK:STDOUT:   %assoc0.8c5739.2: %AddAssignWith.assoc_type.8ddaf4.2 = assoc_entity element0, imports.%Core.import_ref.3b2 [symbolic]
// CHECK:STDOUT:   %T.as_type.50c: type = facet_access_type %T.792 [symbolic]
// CHECK:STDOUT:   %pattern_type.643306.2: type = pattern_type %T.as_type.50c [symbolic]
// CHECK:STDOUT:   %x.patt.227: %pattern_type.643306.2 = ref_binding_pattern x [symbolic]
// CHECK:STDOUT:   %x.param_patt.343: %pattern_type.643306.2 = var_param_pattern %x.patt.227 [symbolic]
// CHECK:STDOUT:   %pattern_type.51d1c4.4: type = pattern_type %U.67d [symbolic]
// CHECK:STDOUT:   %y.param_patt.be3: %pattern_type.51d1c4.4 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %y.patt.b85: %pattern_type.51d1c4.4 = at_binding_pattern y, %y.param_patt.be3 [symbolic]
// CHECK:STDOUT:   %require_complete.2a8: <witness> = require_complete_type %AddAssignWith.type.2c8890.2 [symbolic]
// CHECK:STDOUT:   %assoc0.86d: %AddAssignWith.assoc_type.8ddaf4.1 = assoc_entity element0, imports.%Core.import_ref.87e [symbolic]
// CHECK:STDOUT:   %AddAssignWith.lookup_impl_witness: <witness> = lookup_impl_witness %T.792, @AddAssignWith.1, @AddAssignWith.1(%U.67d) [symbolic]
// CHECK:STDOUT:   %AddAssignWith.WithSelf.Op.type.cd574e.3: type = fn_type @AddAssignWith.WithSelf.Op, @AddAssignWith.WithSelf(%U.67d, %T.792) [symbolic]
// CHECK:STDOUT:   %.9b1: type = fn_type_with_self_type %AddAssignWith.WithSelf.Op.type.cd574e.3, %T.792 [symbolic]
// CHECK:STDOUT:   %impl.elem0.985: %.9b1 = impl_witness_access %AddAssignWith.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.d36: <specific function> = specific_impl_function %impl.elem0.985, @AddAssignWith.WithSelf.Op(%U.67d, %T.792) [symbolic]
// CHECK:STDOUT:   %SubAssignWith.type.7e53cf.1: type = facet_type <@SubAssignWith.1, @SubAssignWith.1(%Other)> [symbolic]
// CHECK:STDOUT:   %Self.f6c9aa.1: %SubAssignWith.type.7e53cf.1 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %SubAssignWith.assoc_type.6e3877.1: type = assoc_entity_type @SubAssignWith.1, @SubAssignWith.1(%Other) [symbolic]
// CHECK:STDOUT:   %SubAssignWith.WithSelf.Op.type.42f7cc.1: type = fn_type @SubAssignWith.WithSelf.Op, @SubAssignWith.WithSelf(%Other, %Self.f6c9aa.1) [symbolic]
// CHECK:STDOUT:   %SubAssignWith.WithSelf.Op.25af21.1: %SubAssignWith.WithSelf.Op.type.42f7cc.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %SubAssignWith.type.7e53cf.2: type = facet_type <@SubAssignWith.1, @SubAssignWith.1(%U.67d)> [symbolic]
// CHECK:STDOUT:   %pattern_type.260: type = pattern_type %SubAssignWith.type.7e53cf.2 [symbolic]
// CHECK:STDOUT:   %T.patt.f45: %pattern_type.260 = symbolic_binding_pattern T, 1 [symbolic]
// CHECK:STDOUT:   %T.f6c: %SubAssignWith.type.7e53cf.2 = symbolic_binding T, 1 [symbolic]
// CHECK:STDOUT:   %SubAssignWith.assoc_type.6e3877.2: type = assoc_entity_type @SubAssignWith.1, @SubAssignWith.1(%U.67d) [symbolic]
// CHECK:STDOUT:   %assoc0.701e5f.2: %SubAssignWith.assoc_type.6e3877.2 = assoc_entity element0, imports.%Core.import_ref.8f2 [symbolic]
// CHECK:STDOUT:   %T.as_type.3ff: type = facet_access_type %T.f6c [symbolic]
// CHECK:STDOUT:   %pattern_type.f08ed5.2: type = pattern_type %T.as_type.3ff [symbolic]
// CHECK:STDOUT:   %x.patt.95a: %pattern_type.f08ed5.2 = ref_binding_pattern x [symbolic]
// CHECK:STDOUT:   %x.param_patt.a45: %pattern_type.f08ed5.2 = var_param_pattern %x.patt.95a [symbolic]
// CHECK:STDOUT:   %require_complete.862: <witness> = require_complete_type %SubAssignWith.type.7e53cf.2 [symbolic]
// CHECK:STDOUT:   %assoc0.b8e: %SubAssignWith.assoc_type.6e3877.1 = assoc_entity element0, imports.%Core.import_ref.e7f [symbolic]
// CHECK:STDOUT:   %SubAssignWith.lookup_impl_witness: <witness> = lookup_impl_witness %T.f6c, @SubAssignWith.1, @SubAssignWith.1(%U.67d) [symbolic]
// CHECK:STDOUT:   %SubAssignWith.WithSelf.Op.type.42f7cc.3: type = fn_type @SubAssignWith.WithSelf.Op, @SubAssignWith.WithSelf(%U.67d, %T.f6c) [symbolic]
// CHECK:STDOUT:   %.e56: type = fn_type_with_self_type %SubAssignWith.WithSelf.Op.type.42f7cc.3, %T.f6c [symbolic]
// CHECK:STDOUT:   %impl.elem0.dd3: %.e56 = impl_witness_access %SubAssignWith.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.5da: <specific function> = specific_impl_function %impl.elem0.dd3, @SubAssignWith.WithSelf.Op(%U.67d, %T.f6c) [symbolic]
// CHECK:STDOUT:   %MulAssignWith.type.3a0e4d.1: type = facet_type <@MulAssignWith.1, @MulAssignWith.1(%Other)> [symbolic]
// CHECK:STDOUT:   %Self.64dcb0.1: %MulAssignWith.type.3a0e4d.1 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %MulAssignWith.assoc_type.9aa18f.1: type = assoc_entity_type @MulAssignWith.1, @MulAssignWith.1(%Other) [symbolic]
// CHECK:STDOUT:   %MulAssignWith.WithSelf.Op.type.5df5bd.1: type = fn_type @MulAssignWith.WithSelf.Op, @MulAssignWith.WithSelf(%Other, %Self.64dcb0.1) [symbolic]
// CHECK:STDOUT:   %MulAssignWith.WithSelf.Op.ee0eda.1: %MulAssignWith.WithSelf.Op.type.5df5bd.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %MulAssignWith.type.3a0e4d.2: type = facet_type <@MulAssignWith.1, @MulAssignWith.1(%U.67d)> [symbolic]
// CHECK:STDOUT:   %pattern_type.8c2: type = pattern_type %MulAssignWith.type.3a0e4d.2 [symbolic]
// CHECK:STDOUT:   %T.patt.db5: %pattern_type.8c2 = symbolic_binding_pattern T, 1 [symbolic]
// CHECK:STDOUT:   %T.64d: %MulAssignWith.type.3a0e4d.2 = symbolic_binding T, 1 [symbolic]
// CHECK:STDOUT:   %MulAssignWith.assoc_type.9aa18f.2: type = assoc_entity_type @MulAssignWith.1, @MulAssignWith.1(%U.67d) [symbolic]
// CHECK:STDOUT:   %assoc0.22264a.2: %MulAssignWith.assoc_type.9aa18f.2 = assoc_entity element0, imports.%Core.import_ref.ba3 [symbolic]
// CHECK:STDOUT:   %T.as_type.1de: type = facet_access_type %T.64d [symbolic]
// CHECK:STDOUT:   %pattern_type.8bf366.2: type = pattern_type %T.as_type.1de [symbolic]
// CHECK:STDOUT:   %x.patt.4e9: %pattern_type.8bf366.2 = ref_binding_pattern x [symbolic]
// CHECK:STDOUT:   %x.param_patt.6c8: %pattern_type.8bf366.2 = var_param_pattern %x.patt.4e9 [symbolic]
// CHECK:STDOUT:   %require_complete.6b1: <witness> = require_complete_type %MulAssignWith.type.3a0e4d.2 [symbolic]
// CHECK:STDOUT:   %assoc0.2ec: %MulAssignWith.assoc_type.9aa18f.1 = assoc_entity element0, imports.%Core.import_ref.408 [symbolic]
// CHECK:STDOUT:   %MulAssignWith.lookup_impl_witness: <witness> = lookup_impl_witness %T.64d, @MulAssignWith.1, @MulAssignWith.1(%U.67d) [symbolic]
// CHECK:STDOUT:   %MulAssignWith.WithSelf.Op.type.5df5bd.3: type = fn_type @MulAssignWith.WithSelf.Op, @MulAssignWith.WithSelf(%U.67d, %T.64d) [symbolic]
// CHECK:STDOUT:   %.2db: type = fn_type_with_self_type %MulAssignWith.WithSelf.Op.type.5df5bd.3, %T.64d [symbolic]
// CHECK:STDOUT:   %impl.elem0.c0f: %.2db = impl_witness_access %MulAssignWith.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.54d: <specific function> = specific_impl_function %impl.elem0.c0f, @MulAssignWith.WithSelf.Op(%U.67d, %T.64d) [symbolic]
// CHECK:STDOUT:   %DivAssignWith.type.7e58ca.1: type = facet_type <@DivAssignWith.1, @DivAssignWith.1(%Other)> [symbolic]
// CHECK:STDOUT:   %Self.c33c81.1: %DivAssignWith.type.7e58ca.1 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %DivAssignWith.assoc_type.bb68ec.1: type = assoc_entity_type @DivAssignWith.1, @DivAssignWith.1(%Other) [symbolic]
// CHECK:STDOUT:   %DivAssignWith.WithSelf.Op.type.ff5775.1: type = fn_type @DivAssignWith.WithSelf.Op, @DivAssignWith.WithSelf(%Other, %Self.c33c81.1) [symbolic]
// CHECK:STDOUT:   %DivAssignWith.WithSelf.Op.91e0ff.1: %DivAssignWith.WithSelf.Op.type.ff5775.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %DivAssignWith.type.7e58ca.2: type = facet_type <@DivAssignWith.1, @DivAssignWith.1(%U.67d)> [symbolic]
// CHECK:STDOUT:   %pattern_type.f74: type = pattern_type %DivAssignWith.type.7e58ca.2 [symbolic]
// CHECK:STDOUT:   %T.patt.9a5: %pattern_type.f74 = symbolic_binding_pattern T, 1 [symbolic]
// CHECK:STDOUT:   %T.c33: %DivAssignWith.type.7e58ca.2 = symbolic_binding T, 1 [symbolic]
// CHECK:STDOUT:   %DivAssignWith.assoc_type.bb68ec.2: type = assoc_entity_type @DivAssignWith.1, @DivAssignWith.1(%U.67d) [symbolic]
// CHECK:STDOUT:   %assoc0.32f100.2: %DivAssignWith.assoc_type.bb68ec.2 = assoc_entity element0, imports.%Core.import_ref.a5d [symbolic]
// CHECK:STDOUT:   %T.as_type.e57: type = facet_access_type %T.c33 [symbolic]
// CHECK:STDOUT:   %pattern_type.da303b.2: type = pattern_type %T.as_type.e57 [symbolic]
// CHECK:STDOUT:   %x.patt.cf6: %pattern_type.da303b.2 = ref_binding_pattern x [symbolic]
// CHECK:STDOUT:   %x.param_patt.660: %pattern_type.da303b.2 = var_param_pattern %x.patt.cf6 [symbolic]
// CHECK:STDOUT:   %require_complete.c5c: <witness> = require_complete_type %DivAssignWith.type.7e58ca.2 [symbolic]
// CHECK:STDOUT:   %assoc0.de1: %DivAssignWith.assoc_type.bb68ec.1 = assoc_entity element0, imports.%Core.import_ref.df2 [symbolic]
// CHECK:STDOUT:   %DivAssignWith.lookup_impl_witness: <witness> = lookup_impl_witness %T.c33, @DivAssignWith.1, @DivAssignWith.1(%U.67d) [symbolic]
// CHECK:STDOUT:   %DivAssignWith.WithSelf.Op.type.ff5775.3: type = fn_type @DivAssignWith.WithSelf.Op, @DivAssignWith.WithSelf(%U.67d, %T.c33) [symbolic]
// CHECK:STDOUT:   %.754: type = fn_type_with_self_type %DivAssignWith.WithSelf.Op.type.ff5775.3, %T.c33 [symbolic]
// CHECK:STDOUT:   %impl.elem0.0c4: %.754 = impl_witness_access %DivAssignWith.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.011: <specific function> = specific_impl_function %impl.elem0.0c4, @DivAssignWith.WithSelf.Op(%U.67d, %T.c33) [symbolic]
// CHECK:STDOUT:   %ModAssignWith.type.778d68.1: type = facet_type <@ModAssignWith.1, @ModAssignWith.1(%Other)> [symbolic]
// CHECK:STDOUT:   %Self.bca00b.1: %ModAssignWith.type.778d68.1 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %ModAssignWith.assoc_type.bc79ec.1: type = assoc_entity_type @ModAssignWith.1, @ModAssignWith.1(%Other) [symbolic]
// CHECK:STDOUT:   %ModAssignWith.WithSelf.Op.type.ec4109.1: type = fn_type @ModAssignWith.WithSelf.Op, @ModAssignWith.WithSelf(%Other, %Self.bca00b.1) [symbolic]
// CHECK:STDOUT:   %ModAssignWith.WithSelf.Op.ac45fc.1: %ModAssignWith.WithSelf.Op.type.ec4109.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %ModAssignWith.type.778d68.2: type = facet_type <@ModAssignWith.1, @ModAssignWith.1(%U.67d)> [symbolic]
// CHECK:STDOUT:   %pattern_type.fb8: type = pattern_type %ModAssignWith.type.778d68.2 [symbolic]
// CHECK:STDOUT:   %T.patt.c29: %pattern_type.fb8 = symbolic_binding_pattern T, 1 [symbolic]
// CHECK:STDOUT:   %T.bca: %ModAssignWith.type.778d68.2 = symbolic_binding T, 1 [symbolic]
// CHECK:STDOUT:   %ModAssignWith.assoc_type.bc79ec.2: type = assoc_entity_type @ModAssignWith.1, @ModAssignWith.1(%U.67d) [symbolic]
// CHECK:STDOUT:   %assoc0.8c0719.2: %ModAssignWith.assoc_type.bc79ec.2 = assoc_entity element0, imports.%Core.import_ref.cf6 [symbolic]
// CHECK:STDOUT:   %T.as_type.ce6: type = facet_access_type %T.bca [symbolic]
// CHECK:STDOUT:   %pattern_type.1d5f30.2: type = pattern_type %T.as_type.ce6 [symbolic]
// CHECK:STDOUT:   %x.patt.fa4: %pattern_type.1d5f30.2 = ref_binding_pattern x [symbolic]
// CHECK:STDOUT:   %x.param_patt.749: %pattern_type.1d5f30.2 = var_param_pattern %x.patt.fa4 [symbolic]
// CHECK:STDOUT:   %require_complete.680: <witness> = require_complete_type %ModAssignWith.type.778d68.2 [symbolic]
// CHECK:STDOUT:   %assoc0.0f3: %ModAssignWith.assoc_type.bc79ec.1 = assoc_entity element0, imports.%Core.import_ref.d29 [symbolic]
// CHECK:STDOUT:   %ModAssignWith.lookup_impl_witness: <witness> = lookup_impl_witness %T.bca, @ModAssignWith.1, @ModAssignWith.1(%U.67d) [symbolic]
// CHECK:STDOUT:   %ModAssignWith.WithSelf.Op.type.ec4109.3: type = fn_type @ModAssignWith.WithSelf.Op, @ModAssignWith.WithSelf(%U.67d, %T.bca) [symbolic]
// CHECK:STDOUT:   %.965: type = fn_type_with_self_type %ModAssignWith.WithSelf.Op.type.ec4109.3, %T.bca [symbolic]
// CHECK:STDOUT:   %impl.elem0.ec7: %.965 = impl_witness_access %ModAssignWith.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.cf7: <specific function> = specific_impl_function %impl.elem0.ec7, @ModAssignWith.WithSelf.Op(%U.67d, %T.bca) [symbolic]
// CHECK:STDOUT:   %AddAssignWith.type.2a0: type = facet_type <@AddAssignWith.1, @AddAssignWith.1(%i16)> [concrete]
// CHECK:STDOUT:   %AddAssignWith.assoc_type.1bb: type = assoc_entity_type @AddAssignWith.1, @AddAssignWith.1(%i16) [concrete]
// CHECK:STDOUT:   %assoc0.24b: %AddAssignWith.assoc_type.1bb = assoc_entity element0, imports.%Core.import_ref.3b2 [concrete]
// CHECK:STDOUT:   %Int16.Op.type.83af6e.13: type = fn_type @Int16.Op.13 [concrete]
// CHECK:STDOUT:   %Int16.Op.4dca31.13: %Int16.Op.type.83af6e.13 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.6a0: <witness> = custom_witness (%Int16.Op.4dca31.13), @AddAssignWith.1, @AddAssignWith.1(%i16) [concrete]
// CHECK:STDOUT:   %AddAssignWith.facet.e34: %AddAssignWith.type.2a0 = facet_value %Int16, (%custom_witness.6a0) [concrete]
// CHECK:STDOUT:   %pattern_type.b256: type = pattern_type %AddAssignWith.type.2a0 [concrete]
// CHECK:STDOUT:   %T.patt.d2a: %pattern_type.b256 = symbolic_binding_pattern T, 1 [symbolic]
// CHECK:STDOUT:   %x.patt.49e: %pattern_type.785 = ref_binding_pattern x [concrete]
// CHECK:STDOUT:   %x.param_patt.69a: %pattern_type.785 = var_param_pattern %x.patt.49e [concrete]
// CHECK:STDOUT:   %SubAssignWith.type.e8b: type = facet_type <@SubAssignWith.1, @SubAssignWith.1(%i16)> [concrete]
// CHECK:STDOUT:   %SubAssignWith.assoc_type.e13: type = assoc_entity_type @SubAssignWith.1, @SubAssignWith.1(%i16) [concrete]
// CHECK:STDOUT:   %assoc0.47c: %SubAssignWith.assoc_type.e13 = assoc_entity element0, imports.%Core.import_ref.8f2 [concrete]
// CHECK:STDOUT:   %Int16.Op.type.83af6e.14: type = fn_type @Int16.Op.14 [concrete]
// CHECK:STDOUT:   %Int16.Op.4dca31.14: %Int16.Op.type.83af6e.14 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.722: <witness> = custom_witness (%Int16.Op.4dca31.14), @SubAssignWith.1, @SubAssignWith.1(%i16) [concrete]
// CHECK:STDOUT:   %SubAssignWith.facet.a52: %SubAssignWith.type.e8b = facet_value %Int16, (%custom_witness.722) [concrete]
// CHECK:STDOUT:   %pattern_type.c22: type = pattern_type %SubAssignWith.type.e8b [concrete]
// CHECK:STDOUT:   %T.patt.0f1: %pattern_type.c22 = symbolic_binding_pattern T, 1 [symbolic]
// CHECK:STDOUT:   %MulAssignWith.type.7d4: type = facet_type <@MulAssignWith.1, @MulAssignWith.1(%i16)> [concrete]
// CHECK:STDOUT:   %MulAssignWith.assoc_type.aa8: type = assoc_entity_type @MulAssignWith.1, @MulAssignWith.1(%i16) [concrete]
// CHECK:STDOUT:   %assoc0.137: %MulAssignWith.assoc_type.aa8 = assoc_entity element0, imports.%Core.import_ref.ba3 [concrete]
// CHECK:STDOUT:   %Int16.Op.type.83af6e.15: type = fn_type @Int16.Op.15 [concrete]
// CHECK:STDOUT:   %Int16.Op.4dca31.15: %Int16.Op.type.83af6e.15 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.c67: <witness> = custom_witness (%Int16.Op.4dca31.15), @MulAssignWith.1, @MulAssignWith.1(%i16) [concrete]
// CHECK:STDOUT:   %MulAssignWith.facet.201: %MulAssignWith.type.7d4 = facet_value %Int16, (%custom_witness.c67) [concrete]
// CHECK:STDOUT:   %pattern_type.a24: type = pattern_type %MulAssignWith.type.7d4 [concrete]
// CHECK:STDOUT:   %T.patt.f71: %pattern_type.a24 = symbolic_binding_pattern T, 1 [symbolic]
// CHECK:STDOUT:   %DivAssignWith.type.081: type = facet_type <@DivAssignWith.1, @DivAssignWith.1(%i16)> [concrete]
// CHECK:STDOUT:   %DivAssignWith.assoc_type.791: type = assoc_entity_type @DivAssignWith.1, @DivAssignWith.1(%i16) [concrete]
// CHECK:STDOUT:   %assoc0.a3b: %DivAssignWith.assoc_type.791 = assoc_entity element0, imports.%Core.import_ref.a5d [concrete]
// CHECK:STDOUT:   %Int16.Op.type.83af6e.16: type = fn_type @Int16.Op.16 [concrete]
// CHECK:STDOUT:   %Int16.Op.4dca31.16: %Int16.Op.type.83af6e.16 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.91d: <witness> = custom_witness (%Int16.Op.4dca31.16), @DivAssignWith.1, @DivAssignWith.1(%i16) [concrete]
// CHECK:STDOUT:   %DivAssignWith.facet.a72: %DivAssignWith.type.081 = facet_value %Int16, (%custom_witness.91d) [concrete]
// CHECK:STDOUT:   %pattern_type.c970: type = pattern_type %DivAssignWith.type.081 [concrete]
// CHECK:STDOUT:   %T.patt.35e: %pattern_type.c970 = symbolic_binding_pattern T, 1 [symbolic]
// CHECK:STDOUT:   %ModAssignWith.type.1ea: type = facet_type <@ModAssignWith.1, @ModAssignWith.1(%i16)> [concrete]
// CHECK:STDOUT:   %ModAssignWith.assoc_type.91d: type = assoc_entity_type @ModAssignWith.1, @ModAssignWith.1(%i16) [concrete]
// CHECK:STDOUT:   %assoc0.b7f: %ModAssignWith.assoc_type.91d = assoc_entity element0, imports.%Core.import_ref.cf6 [concrete]
// CHECK:STDOUT:   %Int16.Op.type.83af6e.17: type = fn_type @Int16.Op.17 [concrete]
// CHECK:STDOUT:   %Int16.Op.4dca31.17: %Int16.Op.type.83af6e.17 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.17b: <witness> = custom_witness (%Int16.Op.4dca31.17), @ModAssignWith.1, @ModAssignWith.1(%i16) [concrete]
// CHECK:STDOUT:   %ModAssignWith.facet.4a2: %ModAssignWith.type.1ea = facet_value %Int16, (%custom_witness.17b) [concrete]
// CHECK:STDOUT:   %pattern_type.6ef: type = pattern_type %ModAssignWith.type.1ea [concrete]
// CHECK:STDOUT:   %T.patt.dbe: %pattern_type.6ef = symbolic_binding_pattern T, 1 [symbolic]
// CHECK:STDOUT:   %AddAssignWith.type.23a: type = facet_type <@AddAssignWith.1, @AddAssignWith.1(%Int32)> [concrete]
// CHECK:STDOUT:   %AddAssignWith.assoc_type.4c7: type = assoc_entity_type @AddAssignWith.1, @AddAssignWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %assoc0.114: %AddAssignWith.assoc_type.4c7 = assoc_entity element0, imports.%Core.import_ref.3b2 [concrete]
// CHECK:STDOUT:   %Int32.Op.type.83af6e.12: type = fn_type @Int32.Op.12 [concrete]
// CHECK:STDOUT:   %Int32.Op.4dca31.12: %Int32.Op.type.83af6e.12 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.ebf67a.1: <witness> = custom_witness (%Int32.Op.4dca31.12), @AddAssignWith.1, @AddAssignWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %AddAssignWith.facet.348: %AddAssignWith.type.23a = facet_value %Int32, (%custom_witness.ebf67a.1) [concrete]
// CHECK:STDOUT:   %pattern_type.5ad: type = pattern_type %AddAssignWith.type.23a [concrete]
// CHECK:STDOUT:   %T.patt.96c: %pattern_type.5ad = symbolic_binding_pattern T, 1 [symbolic]
// CHECK:STDOUT:   %x.patt.d79: %pattern_type.8b4 = ref_binding_pattern x [concrete]
// CHECK:STDOUT:   %x.param_patt.0c7: %pattern_type.8b4 = var_param_pattern %x.patt.d79 [concrete]
// CHECK:STDOUT:   %SubAssignWith.type.4fc: type = facet_type <@SubAssignWith.1, @SubAssignWith.1(%Int32)> [concrete]
// CHECK:STDOUT:   %SubAssignWith.assoc_type.ac6: type = assoc_entity_type @SubAssignWith.1, @SubAssignWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %assoc0.c2a: %SubAssignWith.assoc_type.ac6 = assoc_entity element0, imports.%Core.import_ref.8f2 [concrete]
// CHECK:STDOUT:   %Int32.Op.type.83af6e.14: type = fn_type @Int32.Op.14 [concrete]
// CHECK:STDOUT:   %Int32.Op.4dca31.14: %Int32.Op.type.83af6e.14 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.c2b241.1: <witness> = custom_witness (%Int32.Op.4dca31.14), @SubAssignWith.1, @SubAssignWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %SubAssignWith.facet.e0e: %SubAssignWith.type.4fc = facet_value %Int32, (%custom_witness.c2b241.1) [concrete]
// CHECK:STDOUT:   %pattern_type.190: type = pattern_type %SubAssignWith.type.4fc [concrete]
// CHECK:STDOUT:   %T.patt.a39: %pattern_type.190 = symbolic_binding_pattern T, 1 [symbolic]
// CHECK:STDOUT:   %MulAssignWith.type.5fb: type = facet_type <@MulAssignWith.1, @MulAssignWith.1(%Int32)> [concrete]
// CHECK:STDOUT:   %MulAssignWith.assoc_type.c04: type = assoc_entity_type @MulAssignWith.1, @MulAssignWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %assoc0.2db: %MulAssignWith.assoc_type.c04 = assoc_entity element0, imports.%Core.import_ref.ba3 [concrete]
// CHECK:STDOUT:   %Int32.Op.type.83af6e.15: type = fn_type @Int32.Op.15 [concrete]
// CHECK:STDOUT:   %Int32.Op.4dca31.15: %Int32.Op.type.83af6e.15 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.ec9760.1: <witness> = custom_witness (%Int32.Op.4dca31.15), @MulAssignWith.1, @MulAssignWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %MulAssignWith.facet.13c: %MulAssignWith.type.5fb = facet_value %Int32, (%custom_witness.ec9760.1) [concrete]
// CHECK:STDOUT:   %pattern_type.f3b: type = pattern_type %MulAssignWith.type.5fb [concrete]
// CHECK:STDOUT:   %T.patt.f66: %pattern_type.f3b = symbolic_binding_pattern T, 1 [symbolic]
// CHECK:STDOUT:   %DivAssignWith.type.3ea: type = facet_type <@DivAssignWith.1, @DivAssignWith.1(%Int32)> [concrete]
// CHECK:STDOUT:   %DivAssignWith.assoc_type.ccd: type = assoc_entity_type @DivAssignWith.1, @DivAssignWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %assoc0.d23: %DivAssignWith.assoc_type.ccd = assoc_entity element0, imports.%Core.import_ref.a5d [concrete]
// CHECK:STDOUT:   %Int32.Op.type.83af6e.16: type = fn_type @Int32.Op.16 [concrete]
// CHECK:STDOUT:   %Int32.Op.4dca31.16: %Int32.Op.type.83af6e.16 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.9697f7.1: <witness> = custom_witness (%Int32.Op.4dca31.16), @DivAssignWith.1, @DivAssignWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %DivAssignWith.facet.e59: %DivAssignWith.type.3ea = facet_value %Int32, (%custom_witness.9697f7.1) [concrete]
// CHECK:STDOUT:   %pattern_type.8f3: type = pattern_type %DivAssignWith.type.3ea [concrete]
// CHECK:STDOUT:   %T.patt.316: %pattern_type.8f3 = symbolic_binding_pattern T, 1 [symbolic]
// CHECK:STDOUT:   %ModAssignWith.type.567: type = facet_type <@ModAssignWith.1, @ModAssignWith.1(%Int32)> [concrete]
// CHECK:STDOUT:   %ModAssignWith.assoc_type.fc5: type = assoc_entity_type @ModAssignWith.1, @ModAssignWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %assoc0.1c63: %ModAssignWith.assoc_type.fc5 = assoc_entity element0, imports.%Core.import_ref.cf6 [concrete]
// CHECK:STDOUT:   %Int32.Op.type.83af6e.17: type = fn_type @Int32.Op.17 [concrete]
// CHECK:STDOUT:   %Int32.Op.4dca31.17: %Int32.Op.type.83af6e.17 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.df8f2f.1: <witness> = custom_witness (%Int32.Op.4dca31.17), @ModAssignWith.1, @ModAssignWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %ModAssignWith.facet.dac: %ModAssignWith.type.567 = facet_value %Int32, (%custom_witness.df8f2f.1) [concrete]
// CHECK:STDOUT:   %pattern_type.9ff: type = pattern_type %ModAssignWith.type.567 [concrete]
// CHECK:STDOUT:   %T.patt.6f1: %pattern_type.9ff = symbolic_binding_pattern T, 1 [symbolic]
// CHECK:STDOUT:   %Int64.Op.type.83af6e.2: type = fn_type @Int64.Op.2 [concrete]
// CHECK:STDOUT:   %Int64.Op.4dca31.2: %Int64.Op.type.83af6e.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.ebf67a.2: <witness> = custom_witness (%Int64.Op.4dca31.2), @AddAssignWith.1, @AddAssignWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %AddAssignWith.facet.144: %AddAssignWith.type.23a = facet_value %Int64, (%custom_witness.ebf67a.2) [concrete]
// CHECK:STDOUT:   %x.patt.ace: %pattern_type.4aa = ref_binding_pattern x [concrete]
// CHECK:STDOUT:   %x.param_patt.b33: %pattern_type.4aa = var_param_pattern %x.patt.ace [concrete]
// CHECK:STDOUT:   %Int64.Op.type.83af6e.4: type = fn_type @Int64.Op.4 [concrete]
// CHECK:STDOUT:   %Int64.Op.4dca31.4: %Int64.Op.type.83af6e.4 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.c2b241.2: <witness> = custom_witness (%Int64.Op.4dca31.4), @SubAssignWith.1, @SubAssignWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %SubAssignWith.facet.a7e: %SubAssignWith.type.4fc = facet_value %Int64, (%custom_witness.c2b241.2) [concrete]
// CHECK:STDOUT:   %Int64.Op.type.83af6e.5: type = fn_type @Int64.Op.5 [concrete]
// CHECK:STDOUT:   %Int64.Op.4dca31.5: %Int64.Op.type.83af6e.5 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.ec9760.2: <witness> = custom_witness (%Int64.Op.4dca31.5), @MulAssignWith.1, @MulAssignWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %MulAssignWith.facet.b35: %MulAssignWith.type.5fb = facet_value %Int64, (%custom_witness.ec9760.2) [concrete]
// CHECK:STDOUT:   %Int64.Op.type.83af6e.6: type = fn_type @Int64.Op.6 [concrete]
// CHECK:STDOUT:   %Int64.Op.4dca31.6: %Int64.Op.type.83af6e.6 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.9697f7.2: <witness> = custom_witness (%Int64.Op.4dca31.6), @DivAssignWith.1, @DivAssignWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %DivAssignWith.facet.b1e: %DivAssignWith.type.3ea = facet_value %Int64, (%custom_witness.9697f7.2) [concrete]
// CHECK:STDOUT:   %Int64.Op.type.83af6e.7: type = fn_type @Int64.Op.7 [concrete]
// CHECK:STDOUT:   %Int64.Op.4dca31.7: %Int64.Op.type.83af6e.7 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.df8f2f.2: <witness> = custom_witness (%Int64.Op.4dca31.7), @ModAssignWith.1, @ModAssignWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %ModAssignWith.facet.342: %ModAssignWith.type.567 = facet_value %Int64, (%custom_witness.df8f2f.2) [concrete]
// CHECK:STDOUT:   %Inc.type: type = facet_type <@Inc> [concrete]
// CHECK:STDOUT:   %pattern_type.8c1: type = pattern_type %Inc.type [concrete]
// CHECK:STDOUT:   %T.patt.eab: %pattern_type.8c1 = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T.8f3: %Inc.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.232: type = facet_access_type %T.8f3 [symbolic]
// CHECK:STDOUT:   %pattern_type.427bfc.2: type = pattern_type %T.as_type.232 [symbolic]
// CHECK:STDOUT:   %x.patt.a75: %pattern_type.427bfc.2 = ref_binding_pattern x [symbolic]
// CHECK:STDOUT:   %x.param_patt.5da: %pattern_type.427bfc.2 = var_param_pattern %x.patt.a75 [symbolic]
// CHECK:STDOUT:   %Inc.lookup_impl_witness: <witness> = lookup_impl_witness %T.8f3, @Inc [symbolic]
// CHECK:STDOUT:   %Inc.WithSelf.Op.type.e10ef6.2: type = fn_type @Inc.WithSelf.Op, @Inc.WithSelf(%T.8f3) [symbolic]
// CHECK:STDOUT:   %.a08: type = fn_type_with_self_type %Inc.WithSelf.Op.type.e10ef6.2, %T.8f3 [symbolic]
// CHECK:STDOUT:   %impl.elem0.b92: %.a08 = impl_witness_access %Inc.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.855: <specific function> = specific_impl_function %impl.elem0.b92, @Inc.WithSelf.Op(%T.8f3) [symbolic]
// CHECK:STDOUT:   %Dec.type: type = facet_type <@Dec> [concrete]
// CHECK:STDOUT:   %pattern_type.72a: type = pattern_type %Dec.type [concrete]
// CHECK:STDOUT:   %T.patt.de9: %pattern_type.72a = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T.e8e: %Dec.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.c32: type = facet_access_type %T.e8e [symbolic]
// CHECK:STDOUT:   %pattern_type.2754f0.2: type = pattern_type %T.as_type.c32 [symbolic]
// CHECK:STDOUT:   %x.patt.08b: %pattern_type.2754f0.2 = ref_binding_pattern x [symbolic]
// CHECK:STDOUT:   %x.param_patt.678: %pattern_type.2754f0.2 = var_param_pattern %x.patt.08b [symbolic]
// CHECK:STDOUT:   %Dec.lookup_impl_witness: <witness> = lookup_impl_witness %T.e8e, @Dec [symbolic]
// CHECK:STDOUT:   %Dec.WithSelf.Op.type.bfa830.2: type = fn_type @Dec.WithSelf.Op, @Dec.WithSelf(%T.e8e) [symbolic]
// CHECK:STDOUT:   %.aae: type = fn_type_with_self_type %Dec.WithSelf.Op.type.bfa830.2, %T.e8e [symbolic]
// CHECK:STDOUT:   %impl.elem0.950: %.aae = impl_witness_access %Dec.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.539: <specific function> = specific_impl_function %impl.elem0.950, @Dec.WithSelf.Op(%T.e8e) [symbolic]
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
// CHECK:STDOUT:   %y.patt.053: %pattern_type.f4e485.2 = at_binding_pattern y, %y.param_patt.023 [symbolic]
// CHECK:STDOUT:   %T.as_type.684: type = facet_access_type %T.e01 [symbolic]
// CHECK:STDOUT:   %pattern_type.2dd: type = pattern_type %T.as_type.684 [symbolic]
// CHECK:STDOUT:   %x.param_patt.cbc: %pattern_type.2dd = value_param_pattern [symbolic]
// CHECK:STDOUT:   %x.patt.7fb: %pattern_type.2dd = at_binding_pattern x, %x.param_patt.cbc [symbolic]
// CHECK:STDOUT:   %Negate.lookup_impl_witness.218: <witness> = lookup_impl_witness %T.e01, @Negate [symbolic]
// CHECK:STDOUT:   %Negate.facet.5e5: %Negate.type = facet_value %T.as_type.684, (%Negate.lookup_impl_witness.218) [symbolic]
// CHECK:STDOUT:   %Negate.WithSelf.Op.type.d8a: type = fn_type @Negate.WithSelf.Op, @Negate.WithSelf(%Negate.facet.5e5) [symbolic]
// CHECK:STDOUT:   %.16a: type = fn_type_with_self_type %Negate.WithSelf.Op.type.d8a, %Negate.facet.5e5 [symbolic]
// CHECK:STDOUT:   %impl.elem1.ee9: %.16a = impl_witness_access %Negate.lookup_impl_witness.218, element1 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.923: <specific function> = specific_impl_function %impl.elem1.ee9, @Negate.WithSelf.Op(%Negate.facet.5e5) [symbolic]
// CHECK:STDOUT:   %Int32.Op.type.83af6e.18: type = fn_type @Int32.Op.18 [concrete]
// CHECK:STDOUT:   %Int32.Op.4dca31.18: %Int32.Op.type.83af6e.18 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.f3f: <witness> = custom_witness (%Int32.Op.4dca31.18), @Inc [concrete]
// CHECK:STDOUT:   %Inc.facet.dd2: %Inc.type = facet_value %Int32, (%custom_witness.f3f) [concrete]
// CHECK:STDOUT:   %Int32.Op.type.83af6e.19: type = fn_type @Int32.Op.19 [concrete]
// CHECK:STDOUT:   %Int32.Op.4dca31.19: %Int32.Op.type.83af6e.19 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b24: <witness> = custom_witness (%Int32.Op.4dca31.19), @Dec [concrete]
// CHECK:STDOUT:   %Dec.facet.e1b: %Dec.type = facet_value %Int32, (%custom_witness.b24) [concrete]
// CHECK:STDOUT:   %facet_type.d8c: type = facet_type <@Destroy & @Negate where %impl.elem0.e29 = %Int64> [concrete]
// CHECK:STDOUT:   %Int16.Op.type.83af6e.18: type = fn_type @Int16.Op.18 [concrete]
// CHECK:STDOUT:   %Int16.Op.4dca31.18: %Int16.Op.type.83af6e.18 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.100: <witness> = custom_witness (%Int64, %Int16.Op.4dca31.18), @Negate [concrete]
// CHECK:STDOUT:   %facet_value.9cc: %facet_type.d8c = facet_value %Int16, (%custom_witness.e7a31c.1, %custom_witness.100) [concrete]
// CHECK:STDOUT:   %pattern_type.54c: type = pattern_type %facet_type.d8c [concrete]
// CHECK:STDOUT:   %T.patt.16f: %pattern_type.54c = symbolic_binding_pattern T, 1 [symbolic]
// CHECK:STDOUT:   %facet_type.021: type = facet_type <@Destroy & @Negate where %impl.elem0.e29 = %Int32> [concrete]
// CHECK:STDOUT:   %Int32.Op.type.83af6e.20: type = fn_type @Int32.Op.20 [concrete]
// CHECK:STDOUT:   %Int32.Op.4dca31.20: %Int32.Op.type.83af6e.20 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.a85: <witness> = custom_witness (%Int32, %Int32.Op.4dca31.20), @Negate [concrete]
// CHECK:STDOUT:   %facet_value.dcd: %facet_type.021 = facet_value %Int32, (%custom_witness.e7a31c.2, %custom_witness.a85) [concrete]
// CHECK:STDOUT:   %pattern_type.4cd: type = pattern_type %facet_type.021 [concrete]
// CHECK:STDOUT:   %T.patt.d74: %pattern_type.4cd = symbolic_binding_pattern T, 1 [symbolic]
// CHECK:STDOUT:   %complete_type.7a6: <witness> = complete_type_witness %AddWith.type.b62 [concrete]
// CHECK:STDOUT:   %AddWith.facet.ff0: %AddWith.type.b62 = facet_value %i16, (%custom_witness.248) [concrete]
// CHECK:STDOUT:   %AddWith.WithSelf.Op.type.e1f: type = fn_type @AddWith.WithSelf.Op, @AddWith.WithSelf(%Int16, %AddWith.facet.ff0) [concrete]
// CHECK:STDOUT:   %.2c3: type = fn_type_with_self_type %AddWith.WithSelf.Op.type.e1f, %AddWith.facet.ff0 [concrete]
// CHECK:STDOUT:   %complete_type.b9a: <witness> = complete_type_witness %SubWith.type.b12 [concrete]
// CHECK:STDOUT:   %SubWith.facet.49f: %SubWith.type.b12 = facet_value %i16, (%custom_witness.f14) [concrete]
// CHECK:STDOUT:   %SubWith.WithSelf.Op.type.843: type = fn_type @SubWith.WithSelf.Op, @SubWith.WithSelf(%Int16, %SubWith.facet.49f) [concrete]
// CHECK:STDOUT:   %.00a: type = fn_type_with_self_type %SubWith.WithSelf.Op.type.843, %SubWith.facet.49f [concrete]
// CHECK:STDOUT:   %complete_type.433: <witness> = complete_type_witness %MulWith.type.c18 [concrete]
// CHECK:STDOUT:   %MulWith.facet.d6e: %MulWith.type.c18 = facet_value %i16, (%custom_witness.6bd) [concrete]
// CHECK:STDOUT:   %MulWith.WithSelf.Op.type.f9f: type = fn_type @MulWith.WithSelf.Op, @MulWith.WithSelf(%Int16, %MulWith.facet.d6e) [concrete]
// CHECK:STDOUT:   %.0ed: type = fn_type_with_self_type %MulWith.WithSelf.Op.type.f9f, %MulWith.facet.d6e [concrete]
// CHECK:STDOUT:   %complete_type.64f: <witness> = complete_type_witness %DivWith.type.6f9 [concrete]
// CHECK:STDOUT:   %DivWith.facet.437: %DivWith.type.6f9 = facet_value %i16, (%custom_witness.22c) [concrete]
// CHECK:STDOUT:   %DivWith.WithSelf.Op.type.51a: type = fn_type @DivWith.WithSelf.Op, @DivWith.WithSelf(%Int16, %DivWith.facet.437) [concrete]
// CHECK:STDOUT:   %.316: type = fn_type_with_self_type %DivWith.WithSelf.Op.type.51a, %DivWith.facet.437 [concrete]
// CHECK:STDOUT:   %complete_type.dea: <witness> = complete_type_witness %ModWith.type.763 [concrete]
// CHECK:STDOUT:   %ModWith.facet.251: %ModWith.type.763 = facet_value %i16, (%custom_witness.9e6) [concrete]
// CHECK:STDOUT:   %ModWith.WithSelf.Op.type.62d: type = fn_type @ModWith.WithSelf.Op, @ModWith.WithSelf(%Int16, %ModWith.facet.251) [concrete]
// CHECK:STDOUT:   %.9b3: type = fn_type_with_self_type %ModWith.WithSelf.Op.type.62d, %ModWith.facet.251 [concrete]
// CHECK:STDOUT:   %complete_type.a53: <witness> = complete_type_witness %AddWith.type.a36 [concrete]
// CHECK:STDOUT:   %AddWith.facet.b59: %AddWith.type.a36 = facet_value %Int16, (%custom_witness.b17) [concrete]
// CHECK:STDOUT:   %AddWith.WithSelf.Op.type.de7: type = fn_type @AddWith.WithSelf.Op, @AddWith.WithSelf(%i16, %AddWith.facet.b59) [concrete]
// CHECK:STDOUT:   %.c9e: type = fn_type_with_self_type %AddWith.WithSelf.Op.type.de7, %AddWith.facet.b59 [concrete]
// CHECK:STDOUT:   %complete_type.fc0: <witness> = complete_type_witness %SubWith.type.08b [concrete]
// CHECK:STDOUT:   %SubWith.facet.ef4: %SubWith.type.08b = facet_value %Int16, (%custom_witness.b07) [concrete]
// CHECK:STDOUT:   %SubWith.WithSelf.Op.type.723: type = fn_type @SubWith.WithSelf.Op, @SubWith.WithSelf(%i16, %SubWith.facet.ef4) [concrete]
// CHECK:STDOUT:   %.6f3: type = fn_type_with_self_type %SubWith.WithSelf.Op.type.723, %SubWith.facet.ef4 [concrete]
// CHECK:STDOUT:   %complete_type.850: <witness> = complete_type_witness %MulWith.type.55d [concrete]
// CHECK:STDOUT:   %MulWith.facet.19a: %MulWith.type.55d = facet_value %Int16, (%custom_witness.8bd) [concrete]
// CHECK:STDOUT:   %MulWith.WithSelf.Op.type.7b1: type = fn_type @MulWith.WithSelf.Op, @MulWith.WithSelf(%i16, %MulWith.facet.19a) [concrete]
// CHECK:STDOUT:   %.d82: type = fn_type_with_self_type %MulWith.WithSelf.Op.type.7b1, %MulWith.facet.19a [concrete]
// CHECK:STDOUT:   %complete_type.8a2: <witness> = complete_type_witness %DivWith.type.ab0 [concrete]
// CHECK:STDOUT:   %DivWith.facet.115: %DivWith.type.ab0 = facet_value %Int16, (%custom_witness.963) [concrete]
// CHECK:STDOUT:   %DivWith.WithSelf.Op.type.a49: type = fn_type @DivWith.WithSelf.Op, @DivWith.WithSelf(%i16, %DivWith.facet.115) [concrete]
// CHECK:STDOUT:   %.251: type = fn_type_with_self_type %DivWith.WithSelf.Op.type.a49, %DivWith.facet.115 [concrete]
// CHECK:STDOUT:   %complete_type.385: <witness> = complete_type_witness %ModWith.type.0b2 [concrete]
// CHECK:STDOUT:   %ModWith.facet.925: %ModWith.type.0b2 = facet_value %Int16, (%custom_witness.8f4) [concrete]
// CHECK:STDOUT:   %ModWith.WithSelf.Op.type.519: type = fn_type @ModWith.WithSelf.Op, @ModWith.WithSelf(%i16, %ModWith.facet.925) [concrete]
// CHECK:STDOUT:   %.66e2: type = fn_type_with_self_type %ModWith.WithSelf.Op.type.519, %ModWith.facet.925 [concrete]
// CHECK:STDOUT:   %complete_type.2cc: <witness> = complete_type_witness %AddWith.type.c71 [concrete]
// CHECK:STDOUT:   %AddWith.facet.007: %AddWith.type.c71 = facet_value %Int16, (%custom_witness.e02e0c.1) [concrete]
// CHECK:STDOUT:   %AddWith.WithSelf.Op.type.dc4: type = fn_type @AddWith.WithSelf.Op, @AddWith.WithSelf(%Int32, %AddWith.facet.007) [concrete]
// CHECK:STDOUT:   %.dbb: type = fn_type_with_self_type %AddWith.WithSelf.Op.type.dc4, %AddWith.facet.007 [concrete]
// CHECK:STDOUT:   %complete_type.cf9: <witness> = complete_type_witness %SubWith.type.c81 [concrete]
// CHECK:STDOUT:   %SubWith.facet.703: %SubWith.type.c81 = facet_value %Int16, (%custom_witness.4dea15.1) [concrete]
// CHECK:STDOUT:   %SubWith.WithSelf.Op.type.6eb: type = fn_type @SubWith.WithSelf.Op, @SubWith.WithSelf(%Int32, %SubWith.facet.703) [concrete]
// CHECK:STDOUT:   %.114: type = fn_type_with_self_type %SubWith.WithSelf.Op.type.6eb, %SubWith.facet.703 [concrete]
// CHECK:STDOUT:   %complete_type.e76: <witness> = complete_type_witness %MulWith.type.eeb [concrete]
// CHECK:STDOUT:   %MulWith.facet.fcd: %MulWith.type.eeb = facet_value %Int16, (%custom_witness.337c88.1) [concrete]
// CHECK:STDOUT:   %MulWith.WithSelf.Op.type.99f: type = fn_type @MulWith.WithSelf.Op, @MulWith.WithSelf(%Int32, %MulWith.facet.fcd) [concrete]
// CHECK:STDOUT:   %.c1c: type = fn_type_with_self_type %MulWith.WithSelf.Op.type.99f, %MulWith.facet.fcd [concrete]
// CHECK:STDOUT:   %complete_type.595: <witness> = complete_type_witness %DivWith.type.fb1 [concrete]
// CHECK:STDOUT:   %DivWith.facet.3bd: %DivWith.type.fb1 = facet_value %Int16, (%custom_witness.3eb227.1) [concrete]
// CHECK:STDOUT:   %DivWith.WithSelf.Op.type.500: type = fn_type @DivWith.WithSelf.Op, @DivWith.WithSelf(%Int32, %DivWith.facet.3bd) [concrete]
// CHECK:STDOUT:   %.ebc: type = fn_type_with_self_type %DivWith.WithSelf.Op.type.500, %DivWith.facet.3bd [concrete]
// CHECK:STDOUT:   %complete_type.80f: <witness> = complete_type_witness %ModWith.type.253 [concrete]
// CHECK:STDOUT:   %ModWith.facet.a29: %ModWith.type.253 = facet_value %Int16, (%custom_witness.501c6d.1) [concrete]
// CHECK:STDOUT:   %ModWith.WithSelf.Op.type.d03: type = fn_type @ModWith.WithSelf.Op, @ModWith.WithSelf(%Int32, %ModWith.facet.a29) [concrete]
// CHECK:STDOUT:   %.839: type = fn_type_with_self_type %ModWith.WithSelf.Op.type.d03, %ModWith.facet.a29 [concrete]
// CHECK:STDOUT:   %AddWith.facet.35b: %AddWith.type.b62 = facet_value %Int32, (%custom_witness.15f) [concrete]
// CHECK:STDOUT:   %AddWith.WithSelf.Op.type.29e: type = fn_type @AddWith.WithSelf.Op, @AddWith.WithSelf(%Int16, %AddWith.facet.35b) [concrete]
// CHECK:STDOUT:   %.45e: type = fn_type_with_self_type %AddWith.WithSelf.Op.type.29e, %AddWith.facet.35b [concrete]
// CHECK:STDOUT:   %SubWith.facet.8da: %SubWith.type.b12 = facet_value %Int32, (%custom_witness.bb4) [concrete]
// CHECK:STDOUT:   %SubWith.WithSelf.Op.type.85b: type = fn_type @SubWith.WithSelf.Op, @SubWith.WithSelf(%Int16, %SubWith.facet.8da) [concrete]
// CHECK:STDOUT:   %.db9: type = fn_type_with_self_type %SubWith.WithSelf.Op.type.85b, %SubWith.facet.8da [concrete]
// CHECK:STDOUT:   %MulWith.facet.920: %MulWith.type.c18 = facet_value %Int32, (%custom_witness.0ab) [concrete]
// CHECK:STDOUT:   %MulWith.WithSelf.Op.type.545: type = fn_type @MulWith.WithSelf.Op, @MulWith.WithSelf(%Int16, %MulWith.facet.920) [concrete]
// CHECK:STDOUT:   %.c5e: type = fn_type_with_self_type %MulWith.WithSelf.Op.type.545, %MulWith.facet.920 [concrete]
// CHECK:STDOUT:   %DivWith.facet.c79: %DivWith.type.6f9 = facet_value %Int32, (%custom_witness.3ec) [concrete]
// CHECK:STDOUT:   %DivWith.WithSelf.Op.type.a6b: type = fn_type @DivWith.WithSelf.Op, @DivWith.WithSelf(%Int16, %DivWith.facet.c79) [concrete]
// CHECK:STDOUT:   %.558: type = fn_type_with_self_type %DivWith.WithSelf.Op.type.a6b, %DivWith.facet.c79 [concrete]
// CHECK:STDOUT:   %ModWith.facet.bef: %ModWith.type.763 = facet_value %Int32, (%custom_witness.de0) [concrete]
// CHECK:STDOUT:   %ModWith.WithSelf.Op.type.d54: type = fn_type @ModWith.WithSelf.Op, @ModWith.WithSelf(%Int16, %ModWith.facet.bef) [concrete]
// CHECK:STDOUT:   %.8c0: type = fn_type_with_self_type %ModWith.WithSelf.Op.type.d54, %ModWith.facet.bef [concrete]
// CHECK:STDOUT:   %AddWith.facet.ee1: %AddWith.type.c71 = facet_value %Int32, (%custom_witness.e02e0c.2) [concrete]
// CHECK:STDOUT:   %AddWith.WithSelf.Op.type.5a5: type = fn_type @AddWith.WithSelf.Op, @AddWith.WithSelf(%Int32, %AddWith.facet.ee1) [concrete]
// CHECK:STDOUT:   %.dcb: type = fn_type_with_self_type %AddWith.WithSelf.Op.type.5a5, %AddWith.facet.ee1 [concrete]
// CHECK:STDOUT:   %SubWith.facet.3dd: %SubWith.type.c81 = facet_value %Int32, (%custom_witness.4dea15.2) [concrete]
// CHECK:STDOUT:   %SubWith.WithSelf.Op.type.ccc: type = fn_type @SubWith.WithSelf.Op, @SubWith.WithSelf(%Int32, %SubWith.facet.3dd) [concrete]
// CHECK:STDOUT:   %.237: type = fn_type_with_self_type %SubWith.WithSelf.Op.type.ccc, %SubWith.facet.3dd [concrete]
// CHECK:STDOUT:   %MulWith.facet.4d3: %MulWith.type.eeb = facet_value %Int32, (%custom_witness.337c88.2) [concrete]
// CHECK:STDOUT:   %MulWith.WithSelf.Op.type.843: type = fn_type @MulWith.WithSelf.Op, @MulWith.WithSelf(%Int32, %MulWith.facet.4d3) [concrete]
// CHECK:STDOUT:   %.8c9: type = fn_type_with_self_type %MulWith.WithSelf.Op.type.843, %MulWith.facet.4d3 [concrete]
// CHECK:STDOUT:   %DivWith.facet.4fe: %DivWith.type.fb1 = facet_value %Int32, (%custom_witness.3eb227.2) [concrete]
// CHECK:STDOUT:   %DivWith.WithSelf.Op.type.24e: type = fn_type @DivWith.WithSelf.Op, @DivWith.WithSelf(%Int32, %DivWith.facet.4fe) [concrete]
// CHECK:STDOUT:   %.450: type = fn_type_with_self_type %DivWith.WithSelf.Op.type.24e, %DivWith.facet.4fe [concrete]
// CHECK:STDOUT:   %ModWith.facet.60c: %ModWith.type.253 = facet_value %Int32, (%custom_witness.501c6d.2) [concrete]
// CHECK:STDOUT:   %ModWith.WithSelf.Op.type.bed: type = fn_type @ModWith.WithSelf.Op, @ModWith.WithSelf(%Int32, %ModWith.facet.60c) [concrete]
// CHECK:STDOUT:   %.382: type = fn_type_with_self_type %ModWith.WithSelf.Op.type.bed, %ModWith.facet.60c [concrete]
// CHECK:STDOUT:   %complete_type.d6f: <witness> = complete_type_witness %AddWith.type.b5b [concrete]
// CHECK:STDOUT:   %AddWith.facet.45c: %AddWith.type.b5b = facet_value %Int32, (%custom_witness.68d) [concrete]
// CHECK:STDOUT:   %AddWith.WithSelf.Op.type.6ad: type = fn_type @AddWith.WithSelf.Op, @AddWith.WithSelf(%Int64, %AddWith.facet.45c) [concrete]
// CHECK:STDOUT:   %.b22: type = fn_type_with_self_type %AddWith.WithSelf.Op.type.6ad, %AddWith.facet.45c [concrete]
// CHECK:STDOUT:   %complete_type.b50: <witness> = complete_type_witness %SubWith.type.a46 [concrete]
// CHECK:STDOUT:   %SubWith.facet.3c0: %SubWith.type.a46 = facet_value %Int32, (%custom_witness.1fa) [concrete]
// CHECK:STDOUT:   %SubWith.WithSelf.Op.type.bdd: type = fn_type @SubWith.WithSelf.Op, @SubWith.WithSelf(%Int64, %SubWith.facet.3c0) [concrete]
// CHECK:STDOUT:   %.c3d: type = fn_type_with_self_type %SubWith.WithSelf.Op.type.bdd, %SubWith.facet.3c0 [concrete]
// CHECK:STDOUT:   %complete_type.77d: <witness> = complete_type_witness %MulWith.type.76f [concrete]
// CHECK:STDOUT:   %MulWith.facet.dbd: %MulWith.type.76f = facet_value %Int32, (%custom_witness.99d) [concrete]
// CHECK:STDOUT:   %MulWith.WithSelf.Op.type.cc8: type = fn_type @MulWith.WithSelf.Op, @MulWith.WithSelf(%Int64, %MulWith.facet.dbd) [concrete]
// CHECK:STDOUT:   %.921: type = fn_type_with_self_type %MulWith.WithSelf.Op.type.cc8, %MulWith.facet.dbd [concrete]
// CHECK:STDOUT:   %complete_type.f76: <witness> = complete_type_witness %DivWith.type.095 [concrete]
// CHECK:STDOUT:   %DivWith.facet.91e: %DivWith.type.095 = facet_value %Int32, (%custom_witness.a8f) [concrete]
// CHECK:STDOUT:   %DivWith.WithSelf.Op.type.dd6: type = fn_type @DivWith.WithSelf.Op, @DivWith.WithSelf(%Int64, %DivWith.facet.91e) [concrete]
// CHECK:STDOUT:   %.26f: type = fn_type_with_self_type %DivWith.WithSelf.Op.type.dd6, %DivWith.facet.91e [concrete]
// CHECK:STDOUT:   %complete_type.9d2: <witness> = complete_type_witness %ModWith.type.204 [concrete]
// CHECK:STDOUT:   %ModWith.facet.c38: %ModWith.type.204 = facet_value %Int32, (%custom_witness.972) [concrete]
// CHECK:STDOUT:   %ModWith.WithSelf.Op.type.663: type = fn_type @ModWith.WithSelf.Op, @ModWith.WithSelf(%Int64, %ModWith.facet.c38) [concrete]
// CHECK:STDOUT:   %.ed0: type = fn_type_with_self_type %ModWith.WithSelf.Op.type.663, %ModWith.facet.c38 [concrete]
// CHECK:STDOUT:   %AddWith.facet.3de: %AddWith.type.c71 = facet_value %Int64, (%custom_witness.98f) [concrete]
// CHECK:STDOUT:   %AddWith.WithSelf.Op.type.1c0: type = fn_type @AddWith.WithSelf.Op, @AddWith.WithSelf(%Int32, %AddWith.facet.3de) [concrete]
// CHECK:STDOUT:   %.df7: type = fn_type_with_self_type %AddWith.WithSelf.Op.type.1c0, %AddWith.facet.3de [concrete]
// CHECK:STDOUT:   %SubWith.facet.320: %SubWith.type.c81 = facet_value %Int64, (%custom_witness.5c3) [concrete]
// CHECK:STDOUT:   %SubWith.WithSelf.Op.type.8fd: type = fn_type @SubWith.WithSelf.Op, @SubWith.WithSelf(%Int32, %SubWith.facet.320) [concrete]
// CHECK:STDOUT:   %.ad9: type = fn_type_with_self_type %SubWith.WithSelf.Op.type.8fd, %SubWith.facet.320 [concrete]
// CHECK:STDOUT:   %MulWith.facet.d72: %MulWith.type.eeb = facet_value %Int64, (%custom_witness.ac0) [concrete]
// CHECK:STDOUT:   %MulWith.WithSelf.Op.type.10e: type = fn_type @MulWith.WithSelf.Op, @MulWith.WithSelf(%Int32, %MulWith.facet.d72) [concrete]
// CHECK:STDOUT:   %.be6: type = fn_type_with_self_type %MulWith.WithSelf.Op.type.10e, %MulWith.facet.d72 [concrete]
// CHECK:STDOUT:   %DivWith.facet.4d9: %DivWith.type.fb1 = facet_value %Int64, (%custom_witness.c98) [concrete]
// CHECK:STDOUT:   %DivWith.WithSelf.Op.type.354: type = fn_type @DivWith.WithSelf.Op, @DivWith.WithSelf(%Int32, %DivWith.facet.4d9) [concrete]
// CHECK:STDOUT:   %.436: type = fn_type_with_self_type %DivWith.WithSelf.Op.type.354, %DivWith.facet.4d9 [concrete]
// CHECK:STDOUT:   %ModWith.facet.ca3: %ModWith.type.253 = facet_value %Int64, (%custom_witness.e66) [concrete]
// CHECK:STDOUT:   %ModWith.WithSelf.Op.type.dcc: type = fn_type @ModWith.WithSelf.Op, @ModWith.WithSelf(%Int32, %ModWith.facet.ca3) [concrete]
// CHECK:STDOUT:   %.66e6: type = fn_type_with_self_type %ModWith.WithSelf.Op.type.dcc, %ModWith.facet.ca3 [concrete]
// CHECK:STDOUT:   %complete_type.a21: <witness> = complete_type_witness %AddAssignWith.type.2a0 [concrete]
// CHECK:STDOUT:   %AddAssignWith.WithSelf.Op.type.eb5: type = fn_type @AddAssignWith.WithSelf.Op, @AddAssignWith.WithSelf(%i16, %AddAssignWith.facet.e34) [concrete]
// CHECK:STDOUT:   %.063: type = fn_type_with_self_type %AddAssignWith.WithSelf.Op.type.eb5, %AddAssignWith.facet.e34 [concrete]
// CHECK:STDOUT:   %complete_type.002: <witness> = complete_type_witness %SubAssignWith.type.e8b [concrete]
// CHECK:STDOUT:   %SubAssignWith.WithSelf.Op.type.163: type = fn_type @SubAssignWith.WithSelf.Op, @SubAssignWith.WithSelf(%i16, %SubAssignWith.facet.a52) [concrete]
// CHECK:STDOUT:   %.dc4: type = fn_type_with_self_type %SubAssignWith.WithSelf.Op.type.163, %SubAssignWith.facet.a52 [concrete]
// CHECK:STDOUT:   %complete_type.60a: <witness> = complete_type_witness %MulAssignWith.type.7d4 [concrete]
// CHECK:STDOUT:   %MulAssignWith.WithSelf.Op.type.170: type = fn_type @MulAssignWith.WithSelf.Op, @MulAssignWith.WithSelf(%i16, %MulAssignWith.facet.201) [concrete]
// CHECK:STDOUT:   %.69d: type = fn_type_with_self_type %MulAssignWith.WithSelf.Op.type.170, %MulAssignWith.facet.201 [concrete]
// CHECK:STDOUT:   %complete_type.531: <witness> = complete_type_witness %DivAssignWith.type.081 [concrete]
// CHECK:STDOUT:   %DivAssignWith.WithSelf.Op.type.d5b: type = fn_type @DivAssignWith.WithSelf.Op, @DivAssignWith.WithSelf(%i16, %DivAssignWith.facet.a72) [concrete]
// CHECK:STDOUT:   %.8d8: type = fn_type_with_self_type %DivAssignWith.WithSelf.Op.type.d5b, %DivAssignWith.facet.a72 [concrete]
// CHECK:STDOUT:   %complete_type.450: <witness> = complete_type_witness %ModAssignWith.type.1ea [concrete]
// CHECK:STDOUT:   %ModAssignWith.WithSelf.Op.type.374: type = fn_type @ModAssignWith.WithSelf.Op, @ModAssignWith.WithSelf(%i16, %ModAssignWith.facet.4a2) [concrete]
// CHECK:STDOUT:   %.eca: type = fn_type_with_self_type %ModAssignWith.WithSelf.Op.type.374, %ModAssignWith.facet.4a2 [concrete]
// CHECK:STDOUT:   %complete_type.e1b: <witness> = complete_type_witness %AddAssignWith.type.23a [concrete]
// CHECK:STDOUT:   %AddAssignWith.WithSelf.Op.type.96d: type = fn_type @AddAssignWith.WithSelf.Op, @AddAssignWith.WithSelf(%Int32, %AddAssignWith.facet.348) [concrete]
// CHECK:STDOUT:   %.65c: type = fn_type_with_self_type %AddAssignWith.WithSelf.Op.type.96d, %AddAssignWith.facet.348 [concrete]
// CHECK:STDOUT:   %complete_type.1eb: <witness> = complete_type_witness %SubAssignWith.type.4fc [concrete]
// CHECK:STDOUT:   %SubAssignWith.WithSelf.Op.type.6ac: type = fn_type @SubAssignWith.WithSelf.Op, @SubAssignWith.WithSelf(%Int32, %SubAssignWith.facet.e0e) [concrete]
// CHECK:STDOUT:   %.740: type = fn_type_with_self_type %SubAssignWith.WithSelf.Op.type.6ac, %SubAssignWith.facet.e0e [concrete]
// CHECK:STDOUT:   %complete_type.ef7: <witness> = complete_type_witness %MulAssignWith.type.5fb [concrete]
// CHECK:STDOUT:   %MulAssignWith.WithSelf.Op.type.73c: type = fn_type @MulAssignWith.WithSelf.Op, @MulAssignWith.WithSelf(%Int32, %MulAssignWith.facet.13c) [concrete]
// CHECK:STDOUT:   %.a28: type = fn_type_with_self_type %MulAssignWith.WithSelf.Op.type.73c, %MulAssignWith.facet.13c [concrete]
// CHECK:STDOUT:   %complete_type.502: <witness> = complete_type_witness %DivAssignWith.type.3ea [concrete]
// CHECK:STDOUT:   %DivAssignWith.WithSelf.Op.type.b74: type = fn_type @DivAssignWith.WithSelf.Op, @DivAssignWith.WithSelf(%Int32, %DivAssignWith.facet.e59) [concrete]
// CHECK:STDOUT:   %.204: type = fn_type_with_self_type %DivAssignWith.WithSelf.Op.type.b74, %DivAssignWith.facet.e59 [concrete]
// CHECK:STDOUT:   %complete_type.446: <witness> = complete_type_witness %ModAssignWith.type.567 [concrete]
// CHECK:STDOUT:   %ModAssignWith.WithSelf.Op.type.16f: type = fn_type @ModAssignWith.WithSelf.Op, @ModAssignWith.WithSelf(%Int32, %ModAssignWith.facet.dac) [concrete]
// CHECK:STDOUT:   %.c84: type = fn_type_with_self_type %ModAssignWith.WithSelf.Op.type.16f, %ModAssignWith.facet.dac [concrete]
// CHECK:STDOUT:   %AddAssignWith.WithSelf.Op.type.cf5: type = fn_type @AddAssignWith.WithSelf.Op, @AddAssignWith.WithSelf(%Int32, %AddAssignWith.facet.144) [concrete]
// CHECK:STDOUT:   %.6e0: type = fn_type_with_self_type %AddAssignWith.WithSelf.Op.type.cf5, %AddAssignWith.facet.144 [concrete]
// CHECK:STDOUT:   %SubAssignWith.WithSelf.Op.type.a31: type = fn_type @SubAssignWith.WithSelf.Op, @SubAssignWith.WithSelf(%Int32, %SubAssignWith.facet.a7e) [concrete]
// CHECK:STDOUT:   %.70d: type = fn_type_with_self_type %SubAssignWith.WithSelf.Op.type.a31, %SubAssignWith.facet.a7e [concrete]
// CHECK:STDOUT:   %MulAssignWith.WithSelf.Op.type.606: type = fn_type @MulAssignWith.WithSelf.Op, @MulAssignWith.WithSelf(%Int32, %MulAssignWith.facet.b35) [concrete]
// CHECK:STDOUT:   %.375: type = fn_type_with_self_type %MulAssignWith.WithSelf.Op.type.606, %MulAssignWith.facet.b35 [concrete]
// CHECK:STDOUT:   %DivAssignWith.WithSelf.Op.type.d14: type = fn_type @DivAssignWith.WithSelf.Op, @DivAssignWith.WithSelf(%Int32, %DivAssignWith.facet.b1e) [concrete]
// CHECK:STDOUT:   %.f69: type = fn_type_with_self_type %DivAssignWith.WithSelf.Op.type.d14, %DivAssignWith.facet.b1e [concrete]
// CHECK:STDOUT:   %ModAssignWith.WithSelf.Op.type.690: type = fn_type @ModAssignWith.WithSelf.Op, @ModAssignWith.WithSelf(%Int32, %ModAssignWith.facet.342) [concrete]
// CHECK:STDOUT:   %.af4: type = fn_type_with_self_type %ModAssignWith.WithSelf.Op.type.690, %ModAssignWith.facet.342 [concrete]
// CHECK:STDOUT:   %Inc.WithSelf.Op.type.d1d: type = fn_type @Inc.WithSelf.Op, @Inc.WithSelf(%Inc.facet.dd2) [concrete]
// CHECK:STDOUT:   %.176: type = fn_type_with_self_type %Inc.WithSelf.Op.type.d1d, %Inc.facet.dd2 [concrete]
// CHECK:STDOUT:   %Dec.WithSelf.Op.type.e55: type = fn_type @Dec.WithSelf.Op, @Dec.WithSelf(%Dec.facet.e1b) [concrete]
// CHECK:STDOUT:   %.d09: type = fn_type_with_self_type %Dec.WithSelf.Op.type.e55, %Dec.facet.e1b [concrete]
// CHECK:STDOUT:   %Negate.facet.8f4: %Negate.type = facet_value %Int16, (%custom_witness.100) [concrete]
// CHECK:STDOUT:   %Negate.WithSelf.Op.type.181: type = fn_type @Negate.WithSelf.Op, @Negate.WithSelf(%Negate.facet.8f4) [concrete]
// CHECK:STDOUT:   %.66a: type = fn_type_with_self_type %Negate.WithSelf.Op.type.181, %Negate.facet.8f4 [concrete]
// CHECK:STDOUT:   %Negate.facet.aa9: %Negate.type = facet_value %Int32, (%custom_witness.a85) [concrete]
// CHECK:STDOUT:   %Negate.WithSelf.Op.type.050: type = fn_type @Negate.WithSelf.Op, @Negate.WithSelf(%Negate.facet.aa9) [concrete]
// CHECK:STDOUT:   %.668: type = fn_type_with_self_type %Negate.WithSelf.Op.type.050, %Negate.facet.aa9 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.5a18: @AddWith.WithSelf.%AddWith.assoc_type (%AddWith.assoc_type.588) = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [symbolic = @AddWith.WithSelf.%assoc1 (constants.%assoc1.57d)]
// CHECK:STDOUT:   %Core.import_ref.3cf: @AddWith.WithSelf.%AddWith.WithSelf.Op.type (%AddWith.WithSelf.Op.type.28e) = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [symbolic = @AddWith.WithSelf.%AddWith.WithSelf.Op (constants.%AddWith.WithSelf.Op.eab)]
// CHECK:STDOUT:   %Core.import_ref.91f8: type = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [concrete = %Result.9a4]
// CHECK:STDOUT:   %Result.9a4: type = assoc_const_decl @Result.1 [concrete] {}
// CHECK:STDOUT:   %Core.import_ref.9768 = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.0f3: @SubWith.WithSelf.%SubWith.assoc_type (%SubWith.assoc_type.b0b) = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [symbolic = @SubWith.WithSelf.%assoc1 (constants.%assoc1.178)]
// CHECK:STDOUT:   %Core.import_ref.3a4: @SubWith.WithSelf.%SubWith.WithSelf.Op.type (%SubWith.WithSelf.Op.type.68d) = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [symbolic = @SubWith.WithSelf.%SubWith.WithSelf.Op (constants.%SubWith.WithSelf.Op.97a)]
// CHECK:STDOUT:   %Core.import_ref.190e: type = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [concrete = %Result.7bd]
// CHECK:STDOUT:   %Result.7bd: type = assoc_const_decl @Result.2 [concrete] {}
// CHECK:STDOUT:   %Core.import_ref.b59 = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.8c1: @MulWith.WithSelf.%MulWith.assoc_type (%MulWith.assoc_type.7b2) = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [symbolic = @MulWith.WithSelf.%assoc1 (constants.%assoc1.8f7)]
// CHECK:STDOUT:   %Core.import_ref.5a0: @MulWith.WithSelf.%MulWith.WithSelf.Op.type (%MulWith.WithSelf.Op.type.176) = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [symbolic = @MulWith.WithSelf.%MulWith.WithSelf.Op (constants.%MulWith.WithSelf.Op.a75)]
// CHECK:STDOUT:   %Core.import_ref.59c: type = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [concrete = %Result.a64]
// CHECK:STDOUT:   %Result.a64: type = assoc_const_decl @Result.3 [concrete] {}
// CHECK:STDOUT:   %Core.import_ref.955 = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.ca2: @DivWith.WithSelf.%DivWith.assoc_type (%DivWith.assoc_type.d8b) = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [symbolic = @DivWith.WithSelf.%assoc1 (constants.%assoc1.7b1)]
// CHECK:STDOUT:   %Core.import_ref.7845: @DivWith.WithSelf.%DivWith.WithSelf.Op.type (%DivWith.WithSelf.Op.type.0f0) = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [symbolic = @DivWith.WithSelf.%DivWith.WithSelf.Op (constants.%DivWith.WithSelf.Op.51e)]
// CHECK:STDOUT:   %Core.import_ref.dee: type = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [concrete = %Result.cb0]
// CHECK:STDOUT:   %Result.cb0: type = assoc_const_decl @Result.4 [concrete] {}
// CHECK:STDOUT:   %Core.import_ref.8b6 = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.ffe: @ModWith.WithSelf.%ModWith.assoc_type (%ModWith.assoc_type.316) = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [symbolic = @ModWith.WithSelf.%assoc1 (constants.%assoc1.132)]
// CHECK:STDOUT:   %Core.import_ref.b56: @ModWith.WithSelf.%ModWith.WithSelf.Op.type (%ModWith.WithSelf.Op.type.d14) = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [symbolic = @ModWith.WithSelf.%ModWith.WithSelf.Op (constants.%ModWith.WithSelf.Op.9e4)]
// CHECK:STDOUT:   %Core.import_ref.61a: type = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [concrete = %Result.803]
// CHECK:STDOUT:   %Result.803: type = assoc_const_decl @Result.5 [concrete] {}
// CHECK:STDOUT:   %Core.import_ref.33c = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.bdd: @AddAssignWith.WithSelf.%AddAssignWith.assoc_type (%AddAssignWith.assoc_type.8ddaf4.1) = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [symbolic = @AddAssignWith.WithSelf.%assoc0 (constants.%assoc0.86d)]
// CHECK:STDOUT:   %Core.import_ref.3b2: @AddAssignWith.WithSelf.%AddAssignWith.WithSelf.Op.type (%AddAssignWith.WithSelf.Op.type.cd574e.1) = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [symbolic = @AddAssignWith.WithSelf.%AddAssignWith.WithSelf.Op (constants.%AddAssignWith.WithSelf.Op.106889.1)]
// CHECK:STDOUT:   %Core.import_ref.87e = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.509: @SubAssignWith.WithSelf.%SubAssignWith.assoc_type (%SubAssignWith.assoc_type.6e3877.1) = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [symbolic = @SubAssignWith.WithSelf.%assoc0 (constants.%assoc0.b8e)]
// CHECK:STDOUT:   %Core.import_ref.8f2: @SubAssignWith.WithSelf.%SubAssignWith.WithSelf.Op.type (%SubAssignWith.WithSelf.Op.type.42f7cc.1) = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [symbolic = @SubAssignWith.WithSelf.%SubAssignWith.WithSelf.Op (constants.%SubAssignWith.WithSelf.Op.25af21.1)]
// CHECK:STDOUT:   %Core.import_ref.e7f = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.b6e: @MulAssignWith.WithSelf.%MulAssignWith.assoc_type (%MulAssignWith.assoc_type.9aa18f.1) = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [symbolic = @MulAssignWith.WithSelf.%assoc0 (constants.%assoc0.2ec)]
// CHECK:STDOUT:   %Core.import_ref.ba3: @MulAssignWith.WithSelf.%MulAssignWith.WithSelf.Op.type (%MulAssignWith.WithSelf.Op.type.5df5bd.1) = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [symbolic = @MulAssignWith.WithSelf.%MulAssignWith.WithSelf.Op (constants.%MulAssignWith.WithSelf.Op.ee0eda.1)]
// CHECK:STDOUT:   %Core.import_ref.408 = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.988: @DivAssignWith.WithSelf.%DivAssignWith.assoc_type (%DivAssignWith.assoc_type.bb68ec.1) = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [symbolic = @DivAssignWith.WithSelf.%assoc0 (constants.%assoc0.de1)]
// CHECK:STDOUT:   %Core.import_ref.a5d: @DivAssignWith.WithSelf.%DivAssignWith.WithSelf.Op.type (%DivAssignWith.WithSelf.Op.type.ff5775.1) = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [symbolic = @DivAssignWith.WithSelf.%DivAssignWith.WithSelf.Op (constants.%DivAssignWith.WithSelf.Op.91e0ff.1)]
// CHECK:STDOUT:   %Core.import_ref.df2 = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.3ff: @ModAssignWith.WithSelf.%ModAssignWith.assoc_type (%ModAssignWith.assoc_type.bc79ec.1) = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [symbolic = @ModAssignWith.WithSelf.%assoc0 (constants.%assoc0.0f3)]
// CHECK:STDOUT:   %Core.import_ref.cf6: @ModAssignWith.WithSelf.%ModAssignWith.WithSelf.Op.type (%ModAssignWith.WithSelf.Op.type.ec4109.1) = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [symbolic = @ModAssignWith.WithSelf.%ModAssignWith.WithSelf.Op (constants.%ModAssignWith.WithSelf.Op.ac45fc.1)]
// CHECK:STDOUT:   %Core.import_ref.d29 = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @AddWith.loc124(%Result.loc121_9.2: %Destroy.type, %U.loc122_4.2: %Destroy.type, %T.loc123_4.2: @AddWith.loc124.%facet_type.loc123_37 (%facet_type.c2e)) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %require_complete.loc126: <witness> = require_complete_type %AddWith.type.loc123_35.1 [symbolic = %require_complete.loc126 (constants.%require_complete.15e)]
// CHECK:STDOUT:   %assoc1: @AddWith.loc124.%AddWith.assoc_type (%AddWith.assoc_type.bc8) = assoc_entity element1, imports.%Core.import_ref.3cf [symbolic = %assoc1 (constants.%assoc1.13e)]
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.loc126: <witness> = lookup_impl_witness %T.loc123_4.1, @AddWith.1, @AddWith.1(%U.as_type.loc123_35.1) [symbolic = %AddWith.lookup_impl_witness.loc126 (constants.%AddWith.lookup_impl_witness.2b1)]
// CHECK:STDOUT:   %AddWith.facet: @AddWith.loc124.%AddWith.type.loc123_35.1 (%AddWith.type.9da) = facet_value %T.as_type.loc124_29.1, (%AddWith.lookup_impl_witness.loc126) [symbolic = %AddWith.facet (constants.%AddWith.facet.fea)]
// CHECK:STDOUT:   %AddWith.WithSelf.Op.type: type = fn_type @AddWith.WithSelf.Op, @AddWith.WithSelf(%U.as_type.loc123_35.1, %AddWith.facet) [symbolic = %AddWith.WithSelf.Op.type (constants.%AddWith.WithSelf.Op.type.977)]
// CHECK:STDOUT:   %.loc126_5.6: type = fn_type_with_self_type %AddWith.WithSelf.Op.type, %AddWith.facet [symbolic = %.loc126_5.6 (constants.%.a30)]
// CHECK:STDOUT:   %impl.elem1.loc126_5.2: @AddWith.loc124.%.loc126_5.6 (%.a30) = impl_witness_access %AddWith.lookup_impl_witness.loc126, element1 [symbolic = %impl.elem1.loc126_5.2 (constants.%impl.elem1.b05)]
// CHECK:STDOUT:   %specific_impl_fn.loc126_5.3: <specific function> = specific_impl_function %impl.elem1.loc126_5.2, @AddWith.WithSelf.Op(%U.as_type.loc123_35.1, %AddWith.facet) [symbolic = %specific_impl_fn.loc126_5.3 (constants.%specific_impl_fn.78d)]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Result.loc121_9.1) [symbolic = %Destroy.WithSelf.Op.type (constants.%Destroy.WithSelf.Op.type.d3ece9.2)]
// CHECK:STDOUT:   %.loc126_5.7: type = fn_type_with_self_type %Destroy.WithSelf.Op.type, %Result.loc121_9.1 [symbolic = %.loc126_5.7 (constants.%.53a)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %Result.loc121_9.1, @Destroy [symbolic = %Destroy.lookup_impl_witness (constants.%Destroy.lookup_impl_witness)]
// CHECK:STDOUT:   %impl.elem0.loc126_5.2: @AddWith.loc124.%.loc126_5.7 (%.53a) = impl_witness_access %Destroy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc126_5.2 (constants.%impl.elem0.0f9)]
// CHECK:STDOUT:   %specific_impl_fn.loc126_5.4: <specific function> = specific_impl_function %impl.elem0.loc126_5.2, @Destroy.WithSelf.Op(%Result.loc121_9.1) [symbolic = %specific_impl_fn.loc126_5.4 (constants.%specific_impl_fn.afc)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%result.param: @AddWith.loc124.%Result.as_type.loc123_53.1 (%Result.as_type), %x.param: @AddWith.loc124.%T.as_type.loc124_29.1 (%T.as_type.6fc), %y.param: @AddWith.loc124.%U.as_type.loc123_35.1 (%U.as_type.b61)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: @AddWith.loc124.%T.as_type.loc124_29.1 (%T.as_type.6fc) = name_ref x, %x
// CHECK:STDOUT:     %y.ref: @AddWith.loc124.%U.as_type.loc123_35.1 (%U.as_type.b61) = name_ref y, %y
// CHECK:STDOUT:     %AddWith.type.loc126: type = facet_type <@AddWith.1, @AddWith.1(constants.%U.as_type.b61)> [symbolic = %AddWith.type.loc123_35.1 (constants.%AddWith.type.9da)]
// CHECK:STDOUT:     %.loc126_5.1: @AddWith.loc124.%AddWith.assoc_type (%AddWith.assoc_type.bc8) = specific_constant imports.%Core.import_ref.5a18, @AddWith.WithSelf(constants.%U.as_type.b61, constants.%Self.96a) [symbolic = %assoc1 (constants.%assoc1.13e)]
// CHECK:STDOUT:     %Op.ref: @AddWith.loc124.%AddWith.assoc_type (%AddWith.assoc_type.bc8) = name_ref Op, %.loc126_5.1 [symbolic = %assoc1 (constants.%assoc1.13e)]
// CHECK:STDOUT:     %impl.elem1.loc126_5.1: @AddWith.loc124.%.loc126_5.6 (%.a30) = impl_witness_access constants.%AddWith.lookup_impl_witness.2b1, element1 [symbolic = %impl.elem1.loc126_5.2 (constants.%impl.elem1.b05)]
// CHECK:STDOUT:     %bound_method.loc126_5.1: <bound method> = bound_method %x.ref, %impl.elem1.loc126_5.1
// CHECK:STDOUT:     %specific_impl_fn.loc126_5.1: <specific function> = specific_impl_function %impl.elem1.loc126_5.1, @AddWith.WithSelf.Op(constants.%U.as_type.b61, constants.%AddWith.facet.fea) [symbolic = %specific_impl_fn.loc126_5.3 (constants.%specific_impl_fn.78d)]
// CHECK:STDOUT:     %bound_method.loc126_5.2: <bound method> = bound_method %x.ref, %specific_impl_fn.loc126_5.1
// CHECK:STDOUT:     %.loc126_5.2: ref @AddWith.loc124.%Result.as_type.loc123_53.1 (%Result.as_type) = temporary_storage
// CHECK:STDOUT:     %AddWith.WithSelf.Op.call: init @AddWith.loc124.%Result.as_type.loc123_53.1 (%Result.as_type) to %.loc126_5.2 = call %bound_method.loc126_5.2(%x.ref, %y.ref)
// CHECK:STDOUT:     %.loc126_5.3: ref @AddWith.loc124.%Result.as_type.loc123_53.1 (%Result.as_type) = temporary %.loc126_5.2, %AddWith.WithSelf.Op.call
// CHECK:STDOUT:     %impl.elem0.loc126_5.1: @AddWith.loc124.%.loc126_5.7 (%.53a) = impl_witness_access constants.%Destroy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc126_5.2 (constants.%impl.elem0.0f9)]
// CHECK:STDOUT:     %bound_method.loc126_5.3: <bound method> = bound_method %.loc126_5.3, %impl.elem0.loc126_5.1
// CHECK:STDOUT:     %.loc126_5.4: %Destroy.type = converted constants.%Result.as_type, constants.%Result [symbolic = %Result.loc121_9.1 (constants.%Result)]
// CHECK:STDOUT:     %.loc126_5.5: %Destroy.type = converted constants.%Result.as_type, constants.%Result [symbolic = %Result.loc121_9.1 (constants.%Result)]
// CHECK:STDOUT:     %specific_impl_fn.loc126_5.2: <specific function> = specific_impl_function %impl.elem0.loc126_5.1, @Destroy.WithSelf.Op(constants.%Result) [symbolic = %specific_impl_fn.loc126_5.4 (constants.%specific_impl_fn.afc)]
// CHECK:STDOUT:     %bound_method.loc126_5.4: <bound method> = bound_method %.loc126_5.3, %specific_impl_fn.loc126_5.2
// CHECK:STDOUT:     %Destroy.WithSelf.Op.call: init %empty_tuple.type = call %bound_method.loc126_5.4(%.loc126_5.3)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @SubWith.loc135(%Result.loc132_9.2: %Destroy.type, %U.loc133_4.2: %Destroy.type, %T.loc134_4.2: @SubWith.loc135.%facet_type.loc134_37 (%facet_type.eeb)) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %require_complete.loc137: <witness> = require_complete_type %SubWith.type.loc134_35.1 [symbolic = %require_complete.loc137 (constants.%require_complete.6ff)]
// CHECK:STDOUT:   %assoc1: @SubWith.loc135.%SubWith.assoc_type (%SubWith.assoc_type.3a1) = assoc_entity element1, imports.%Core.import_ref.3a4 [symbolic = %assoc1 (constants.%assoc1.8fd)]
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.loc137: <witness> = lookup_impl_witness %T.loc134_4.1, @SubWith.1, @SubWith.1(%U.as_type.loc134_35.1) [symbolic = %SubWith.lookup_impl_witness.loc137 (constants.%SubWith.lookup_impl_witness.ce5)]
// CHECK:STDOUT:   %SubWith.facet: @SubWith.loc135.%SubWith.type.loc134_35.1 (%SubWith.type.b5d) = facet_value %T.as_type.loc135_29.1, (%SubWith.lookup_impl_witness.loc137) [symbolic = %SubWith.facet (constants.%SubWith.facet.f6a)]
// CHECK:STDOUT:   %SubWith.WithSelf.Op.type: type = fn_type @SubWith.WithSelf.Op, @SubWith.WithSelf(%U.as_type.loc134_35.1, %SubWith.facet) [symbolic = %SubWith.WithSelf.Op.type (constants.%SubWith.WithSelf.Op.type.4fe)]
// CHECK:STDOUT:   %.loc137_5.6: type = fn_type_with_self_type %SubWith.WithSelf.Op.type, %SubWith.facet [symbolic = %.loc137_5.6 (constants.%.a4a)]
// CHECK:STDOUT:   %impl.elem1.loc137_5.2: @SubWith.loc135.%.loc137_5.6 (%.a4a) = impl_witness_access %SubWith.lookup_impl_witness.loc137, element1 [symbolic = %impl.elem1.loc137_5.2 (constants.%impl.elem1.e4f)]
// CHECK:STDOUT:   %specific_impl_fn.loc137_5.3: <specific function> = specific_impl_function %impl.elem1.loc137_5.2, @SubWith.WithSelf.Op(%U.as_type.loc134_35.1, %SubWith.facet) [symbolic = %specific_impl_fn.loc137_5.3 (constants.%specific_impl_fn.46d)]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Result.loc132_9.1) [symbolic = %Destroy.WithSelf.Op.type (constants.%Destroy.WithSelf.Op.type.d3ece9.2)]
// CHECK:STDOUT:   %.loc137_5.7: type = fn_type_with_self_type %Destroy.WithSelf.Op.type, %Result.loc132_9.1 [symbolic = %.loc137_5.7 (constants.%.53a)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %Result.loc132_9.1, @Destroy [symbolic = %Destroy.lookup_impl_witness (constants.%Destroy.lookup_impl_witness)]
// CHECK:STDOUT:   %impl.elem0.loc137_5.2: @SubWith.loc135.%.loc137_5.7 (%.53a) = impl_witness_access %Destroy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc137_5.2 (constants.%impl.elem0.0f9)]
// CHECK:STDOUT:   %specific_impl_fn.loc137_5.4: <specific function> = specific_impl_function %impl.elem0.loc137_5.2, @Destroy.WithSelf.Op(%Result.loc132_9.1) [symbolic = %specific_impl_fn.loc137_5.4 (constants.%specific_impl_fn.afc)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%result.param: @SubWith.loc135.%Result.as_type.loc134_53.1 (%Result.as_type), %x.param: @SubWith.loc135.%T.as_type.loc135_29.1 (%T.as_type.58f), %y.param: @SubWith.loc135.%U.as_type.loc134_35.1 (%U.as_type.b61)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: @SubWith.loc135.%T.as_type.loc135_29.1 (%T.as_type.58f) = name_ref x, %x
// CHECK:STDOUT:     %y.ref: @SubWith.loc135.%U.as_type.loc134_35.1 (%U.as_type.b61) = name_ref y, %y
// CHECK:STDOUT:     %SubWith.type.loc137: type = facet_type <@SubWith.1, @SubWith.1(constants.%U.as_type.b61)> [symbolic = %SubWith.type.loc134_35.1 (constants.%SubWith.type.b5d)]
// CHECK:STDOUT:     %.loc137_5.1: @SubWith.loc135.%SubWith.assoc_type (%SubWith.assoc_type.3a1) = specific_constant imports.%Core.import_ref.0f3, @SubWith.WithSelf(constants.%U.as_type.b61, constants.%Self.d6b) [symbolic = %assoc1 (constants.%assoc1.8fd)]
// CHECK:STDOUT:     %Op.ref: @SubWith.loc135.%SubWith.assoc_type (%SubWith.assoc_type.3a1) = name_ref Op, %.loc137_5.1 [symbolic = %assoc1 (constants.%assoc1.8fd)]
// CHECK:STDOUT:     %impl.elem1.loc137_5.1: @SubWith.loc135.%.loc137_5.6 (%.a4a) = impl_witness_access constants.%SubWith.lookup_impl_witness.ce5, element1 [symbolic = %impl.elem1.loc137_5.2 (constants.%impl.elem1.e4f)]
// CHECK:STDOUT:     %bound_method.loc137_5.1: <bound method> = bound_method %x.ref, %impl.elem1.loc137_5.1
// CHECK:STDOUT:     %specific_impl_fn.loc137_5.1: <specific function> = specific_impl_function %impl.elem1.loc137_5.1, @SubWith.WithSelf.Op(constants.%U.as_type.b61, constants.%SubWith.facet.f6a) [symbolic = %specific_impl_fn.loc137_5.3 (constants.%specific_impl_fn.46d)]
// CHECK:STDOUT:     %bound_method.loc137_5.2: <bound method> = bound_method %x.ref, %specific_impl_fn.loc137_5.1
// CHECK:STDOUT:     %.loc137_5.2: ref @SubWith.loc135.%Result.as_type.loc134_53.1 (%Result.as_type) = temporary_storage
// CHECK:STDOUT:     %SubWith.WithSelf.Op.call: init @SubWith.loc135.%Result.as_type.loc134_53.1 (%Result.as_type) to %.loc137_5.2 = call %bound_method.loc137_5.2(%x.ref, %y.ref)
// CHECK:STDOUT:     %.loc137_5.3: ref @SubWith.loc135.%Result.as_type.loc134_53.1 (%Result.as_type) = temporary %.loc137_5.2, %SubWith.WithSelf.Op.call
// CHECK:STDOUT:     %impl.elem0.loc137_5.1: @SubWith.loc135.%.loc137_5.7 (%.53a) = impl_witness_access constants.%Destroy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc137_5.2 (constants.%impl.elem0.0f9)]
// CHECK:STDOUT:     %bound_method.loc137_5.3: <bound method> = bound_method %.loc137_5.3, %impl.elem0.loc137_5.1
// CHECK:STDOUT:     %.loc137_5.4: %Destroy.type = converted constants.%Result.as_type, constants.%Result [symbolic = %Result.loc132_9.1 (constants.%Result)]
// CHECK:STDOUT:     %.loc137_5.5: %Destroy.type = converted constants.%Result.as_type, constants.%Result [symbolic = %Result.loc132_9.1 (constants.%Result)]
// CHECK:STDOUT:     %specific_impl_fn.loc137_5.2: <specific function> = specific_impl_function %impl.elem0.loc137_5.1, @Destroy.WithSelf.Op(constants.%Result) [symbolic = %specific_impl_fn.loc137_5.4 (constants.%specific_impl_fn.afc)]
// CHECK:STDOUT:     %bound_method.loc137_5.4: <bound method> = bound_method %.loc137_5.3, %specific_impl_fn.loc137_5.2
// CHECK:STDOUT:     %Destroy.WithSelf.Op.call: init %empty_tuple.type = call %bound_method.loc137_5.4(%.loc137_5.3)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @MulWith.loc146(%Result.loc143_9.2: %Destroy.type, %U.loc144_4.2: %Destroy.type, %T.loc145_4.2: @MulWith.loc146.%facet_type.loc145_37 (%facet_type.0e5)) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %require_complete.loc148: <witness> = require_complete_type %MulWith.type.loc145_35.1 [symbolic = %require_complete.loc148 (constants.%require_complete.861)]
// CHECK:STDOUT:   %assoc1: @MulWith.loc146.%MulWith.assoc_type (%MulWith.assoc_type.a12) = assoc_entity element1, imports.%Core.import_ref.5a0 [symbolic = %assoc1 (constants.%assoc1.a0f)]
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.loc148: <witness> = lookup_impl_witness %T.loc145_4.1, @MulWith.1, @MulWith.1(%U.as_type.loc145_35.1) [symbolic = %MulWith.lookup_impl_witness.loc148 (constants.%MulWith.lookup_impl_witness.444)]
// CHECK:STDOUT:   %MulWith.facet: @MulWith.loc146.%MulWith.type.loc145_35.1 (%MulWith.type.f02) = facet_value %T.as_type.loc146_29.1, (%MulWith.lookup_impl_witness.loc148) [symbolic = %MulWith.facet (constants.%MulWith.facet.b99)]
// CHECK:STDOUT:   %MulWith.WithSelf.Op.type: type = fn_type @MulWith.WithSelf.Op, @MulWith.WithSelf(%U.as_type.loc145_35.1, %MulWith.facet) [symbolic = %MulWith.WithSelf.Op.type (constants.%MulWith.WithSelf.Op.type.c95)]
// CHECK:STDOUT:   %.loc148_5.6: type = fn_type_with_self_type %MulWith.WithSelf.Op.type, %MulWith.facet [symbolic = %.loc148_5.6 (constants.%.a49)]
// CHECK:STDOUT:   %impl.elem1.loc148_5.2: @MulWith.loc146.%.loc148_5.6 (%.a49) = impl_witness_access %MulWith.lookup_impl_witness.loc148, element1 [symbolic = %impl.elem1.loc148_5.2 (constants.%impl.elem1.a7b)]
// CHECK:STDOUT:   %specific_impl_fn.loc148_5.3: <specific function> = specific_impl_function %impl.elem1.loc148_5.2, @MulWith.WithSelf.Op(%U.as_type.loc145_35.1, %MulWith.facet) [symbolic = %specific_impl_fn.loc148_5.3 (constants.%specific_impl_fn.eab)]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Result.loc143_9.1) [symbolic = %Destroy.WithSelf.Op.type (constants.%Destroy.WithSelf.Op.type.d3ece9.2)]
// CHECK:STDOUT:   %.loc148_5.7: type = fn_type_with_self_type %Destroy.WithSelf.Op.type, %Result.loc143_9.1 [symbolic = %.loc148_5.7 (constants.%.53a)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %Result.loc143_9.1, @Destroy [symbolic = %Destroy.lookup_impl_witness (constants.%Destroy.lookup_impl_witness)]
// CHECK:STDOUT:   %impl.elem0.loc148_5.2: @MulWith.loc146.%.loc148_5.7 (%.53a) = impl_witness_access %Destroy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc148_5.2 (constants.%impl.elem0.0f9)]
// CHECK:STDOUT:   %specific_impl_fn.loc148_5.4: <specific function> = specific_impl_function %impl.elem0.loc148_5.2, @Destroy.WithSelf.Op(%Result.loc143_9.1) [symbolic = %specific_impl_fn.loc148_5.4 (constants.%specific_impl_fn.afc)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%result.param: @MulWith.loc146.%Result.as_type.loc145_53.1 (%Result.as_type), %x.param: @MulWith.loc146.%T.as_type.loc146_29.1 (%T.as_type.83f), %y.param: @MulWith.loc146.%U.as_type.loc145_35.1 (%U.as_type.b61)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: @MulWith.loc146.%T.as_type.loc146_29.1 (%T.as_type.83f) = name_ref x, %x
// CHECK:STDOUT:     %y.ref: @MulWith.loc146.%U.as_type.loc145_35.1 (%U.as_type.b61) = name_ref y, %y
// CHECK:STDOUT:     %MulWith.type.loc148: type = facet_type <@MulWith.1, @MulWith.1(constants.%U.as_type.b61)> [symbolic = %MulWith.type.loc145_35.1 (constants.%MulWith.type.f02)]
// CHECK:STDOUT:     %.loc148_5.1: @MulWith.loc146.%MulWith.assoc_type (%MulWith.assoc_type.a12) = specific_constant imports.%Core.import_ref.8c1, @MulWith.WithSelf(constants.%U.as_type.b61, constants.%Self.941) [symbolic = %assoc1 (constants.%assoc1.a0f)]
// CHECK:STDOUT:     %Op.ref: @MulWith.loc146.%MulWith.assoc_type (%MulWith.assoc_type.a12) = name_ref Op, %.loc148_5.1 [symbolic = %assoc1 (constants.%assoc1.a0f)]
// CHECK:STDOUT:     %impl.elem1.loc148_5.1: @MulWith.loc146.%.loc148_5.6 (%.a49) = impl_witness_access constants.%MulWith.lookup_impl_witness.444, element1 [symbolic = %impl.elem1.loc148_5.2 (constants.%impl.elem1.a7b)]
// CHECK:STDOUT:     %bound_method.loc148_5.1: <bound method> = bound_method %x.ref, %impl.elem1.loc148_5.1
// CHECK:STDOUT:     %specific_impl_fn.loc148_5.1: <specific function> = specific_impl_function %impl.elem1.loc148_5.1, @MulWith.WithSelf.Op(constants.%U.as_type.b61, constants.%MulWith.facet.b99) [symbolic = %specific_impl_fn.loc148_5.3 (constants.%specific_impl_fn.eab)]
// CHECK:STDOUT:     %bound_method.loc148_5.2: <bound method> = bound_method %x.ref, %specific_impl_fn.loc148_5.1
// CHECK:STDOUT:     %.loc148_5.2: ref @MulWith.loc146.%Result.as_type.loc145_53.1 (%Result.as_type) = temporary_storage
// CHECK:STDOUT:     %MulWith.WithSelf.Op.call: init @MulWith.loc146.%Result.as_type.loc145_53.1 (%Result.as_type) to %.loc148_5.2 = call %bound_method.loc148_5.2(%x.ref, %y.ref)
// CHECK:STDOUT:     %.loc148_5.3: ref @MulWith.loc146.%Result.as_type.loc145_53.1 (%Result.as_type) = temporary %.loc148_5.2, %MulWith.WithSelf.Op.call
// CHECK:STDOUT:     %impl.elem0.loc148_5.1: @MulWith.loc146.%.loc148_5.7 (%.53a) = impl_witness_access constants.%Destroy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc148_5.2 (constants.%impl.elem0.0f9)]
// CHECK:STDOUT:     %bound_method.loc148_5.3: <bound method> = bound_method %.loc148_5.3, %impl.elem0.loc148_5.1
// CHECK:STDOUT:     %.loc148_5.4: %Destroy.type = converted constants.%Result.as_type, constants.%Result [symbolic = %Result.loc143_9.1 (constants.%Result)]
// CHECK:STDOUT:     %.loc148_5.5: %Destroy.type = converted constants.%Result.as_type, constants.%Result [symbolic = %Result.loc143_9.1 (constants.%Result)]
// CHECK:STDOUT:     %specific_impl_fn.loc148_5.2: <specific function> = specific_impl_function %impl.elem0.loc148_5.1, @Destroy.WithSelf.Op(constants.%Result) [symbolic = %specific_impl_fn.loc148_5.4 (constants.%specific_impl_fn.afc)]
// CHECK:STDOUT:     %bound_method.loc148_5.4: <bound method> = bound_method %.loc148_5.3, %specific_impl_fn.loc148_5.2
// CHECK:STDOUT:     %Destroy.WithSelf.Op.call: init %empty_tuple.type = call %bound_method.loc148_5.4(%.loc148_5.3)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @DivWith.loc157(%Result.loc154_9.2: %Destroy.type, %U.loc155_4.2: %Destroy.type, %T.loc156_4.2: @DivWith.loc157.%facet_type.loc156_37 (%facet_type.ed9)) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %require_complete.loc159: <witness> = require_complete_type %DivWith.type.loc156_35.1 [symbolic = %require_complete.loc159 (constants.%require_complete.026)]
// CHECK:STDOUT:   %assoc1: @DivWith.loc157.%DivWith.assoc_type (%DivWith.assoc_type.cf4) = assoc_entity element1, imports.%Core.import_ref.7845 [symbolic = %assoc1 (constants.%assoc1.4eb)]
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.loc159: <witness> = lookup_impl_witness %T.loc156_4.1, @DivWith.1, @DivWith.1(%U.as_type.loc156_35.1) [symbolic = %DivWith.lookup_impl_witness.loc159 (constants.%DivWith.lookup_impl_witness.831)]
// CHECK:STDOUT:   %DivWith.facet: @DivWith.loc157.%DivWith.type.loc156_35.1 (%DivWith.type.523) = facet_value %T.as_type.loc157_29.1, (%DivWith.lookup_impl_witness.loc159) [symbolic = %DivWith.facet (constants.%DivWith.facet.5f7)]
// CHECK:STDOUT:   %DivWith.WithSelf.Op.type: type = fn_type @DivWith.WithSelf.Op, @DivWith.WithSelf(%U.as_type.loc156_35.1, %DivWith.facet) [symbolic = %DivWith.WithSelf.Op.type (constants.%DivWith.WithSelf.Op.type.1b2)]
// CHECK:STDOUT:   %.loc159_5.6: type = fn_type_with_self_type %DivWith.WithSelf.Op.type, %DivWith.facet [symbolic = %.loc159_5.6 (constants.%.6b1)]
// CHECK:STDOUT:   %impl.elem1.loc159_5.2: @DivWith.loc157.%.loc159_5.6 (%.6b1) = impl_witness_access %DivWith.lookup_impl_witness.loc159, element1 [symbolic = %impl.elem1.loc159_5.2 (constants.%impl.elem1.bfe)]
// CHECK:STDOUT:   %specific_impl_fn.loc159_5.3: <specific function> = specific_impl_function %impl.elem1.loc159_5.2, @DivWith.WithSelf.Op(%U.as_type.loc156_35.1, %DivWith.facet) [symbolic = %specific_impl_fn.loc159_5.3 (constants.%specific_impl_fn.e98)]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Result.loc154_9.1) [symbolic = %Destroy.WithSelf.Op.type (constants.%Destroy.WithSelf.Op.type.d3ece9.2)]
// CHECK:STDOUT:   %.loc159_5.7: type = fn_type_with_self_type %Destroy.WithSelf.Op.type, %Result.loc154_9.1 [symbolic = %.loc159_5.7 (constants.%.53a)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %Result.loc154_9.1, @Destroy [symbolic = %Destroy.lookup_impl_witness (constants.%Destroy.lookup_impl_witness)]
// CHECK:STDOUT:   %impl.elem0.loc159_5.2: @DivWith.loc157.%.loc159_5.7 (%.53a) = impl_witness_access %Destroy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc159_5.2 (constants.%impl.elem0.0f9)]
// CHECK:STDOUT:   %specific_impl_fn.loc159_5.4: <specific function> = specific_impl_function %impl.elem0.loc159_5.2, @Destroy.WithSelf.Op(%Result.loc154_9.1) [symbolic = %specific_impl_fn.loc159_5.4 (constants.%specific_impl_fn.afc)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%result.param: @DivWith.loc157.%Result.as_type.loc156_53.1 (%Result.as_type), %x.param: @DivWith.loc157.%T.as_type.loc157_29.1 (%T.as_type.a74), %y.param: @DivWith.loc157.%U.as_type.loc156_35.1 (%U.as_type.b61)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: @DivWith.loc157.%T.as_type.loc157_29.1 (%T.as_type.a74) = name_ref x, %x
// CHECK:STDOUT:     %y.ref: @DivWith.loc157.%U.as_type.loc156_35.1 (%U.as_type.b61) = name_ref y, %y
// CHECK:STDOUT:     %DivWith.type.loc159: type = facet_type <@DivWith.1, @DivWith.1(constants.%U.as_type.b61)> [symbolic = %DivWith.type.loc156_35.1 (constants.%DivWith.type.523)]
// CHECK:STDOUT:     %.loc159_5.1: @DivWith.loc157.%DivWith.assoc_type (%DivWith.assoc_type.cf4) = specific_constant imports.%Core.import_ref.ca2, @DivWith.WithSelf(constants.%U.as_type.b61, constants.%Self.16a) [symbolic = %assoc1 (constants.%assoc1.4eb)]
// CHECK:STDOUT:     %Op.ref: @DivWith.loc157.%DivWith.assoc_type (%DivWith.assoc_type.cf4) = name_ref Op, %.loc159_5.1 [symbolic = %assoc1 (constants.%assoc1.4eb)]
// CHECK:STDOUT:     %impl.elem1.loc159_5.1: @DivWith.loc157.%.loc159_5.6 (%.6b1) = impl_witness_access constants.%DivWith.lookup_impl_witness.831, element1 [symbolic = %impl.elem1.loc159_5.2 (constants.%impl.elem1.bfe)]
// CHECK:STDOUT:     %bound_method.loc159_5.1: <bound method> = bound_method %x.ref, %impl.elem1.loc159_5.1
// CHECK:STDOUT:     %specific_impl_fn.loc159_5.1: <specific function> = specific_impl_function %impl.elem1.loc159_5.1, @DivWith.WithSelf.Op(constants.%U.as_type.b61, constants.%DivWith.facet.5f7) [symbolic = %specific_impl_fn.loc159_5.3 (constants.%specific_impl_fn.e98)]
// CHECK:STDOUT:     %bound_method.loc159_5.2: <bound method> = bound_method %x.ref, %specific_impl_fn.loc159_5.1
// CHECK:STDOUT:     %.loc159_5.2: ref @DivWith.loc157.%Result.as_type.loc156_53.1 (%Result.as_type) = temporary_storage
// CHECK:STDOUT:     %DivWith.WithSelf.Op.call: init @DivWith.loc157.%Result.as_type.loc156_53.1 (%Result.as_type) to %.loc159_5.2 = call %bound_method.loc159_5.2(%x.ref, %y.ref)
// CHECK:STDOUT:     %.loc159_5.3: ref @DivWith.loc157.%Result.as_type.loc156_53.1 (%Result.as_type) = temporary %.loc159_5.2, %DivWith.WithSelf.Op.call
// CHECK:STDOUT:     %impl.elem0.loc159_5.1: @DivWith.loc157.%.loc159_5.7 (%.53a) = impl_witness_access constants.%Destroy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc159_5.2 (constants.%impl.elem0.0f9)]
// CHECK:STDOUT:     %bound_method.loc159_5.3: <bound method> = bound_method %.loc159_5.3, %impl.elem0.loc159_5.1
// CHECK:STDOUT:     %.loc159_5.4: %Destroy.type = converted constants.%Result.as_type, constants.%Result [symbolic = %Result.loc154_9.1 (constants.%Result)]
// CHECK:STDOUT:     %.loc159_5.5: %Destroy.type = converted constants.%Result.as_type, constants.%Result [symbolic = %Result.loc154_9.1 (constants.%Result)]
// CHECK:STDOUT:     %specific_impl_fn.loc159_5.2: <specific function> = specific_impl_function %impl.elem0.loc159_5.1, @Destroy.WithSelf.Op(constants.%Result) [symbolic = %specific_impl_fn.loc159_5.4 (constants.%specific_impl_fn.afc)]
// CHECK:STDOUT:     %bound_method.loc159_5.4: <bound method> = bound_method %.loc159_5.3, %specific_impl_fn.loc159_5.2
// CHECK:STDOUT:     %Destroy.WithSelf.Op.call: init %empty_tuple.type = call %bound_method.loc159_5.4(%.loc159_5.3)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @ModWith.loc168(%Result.loc165_9.2: %Destroy.type, %U.loc166_4.2: %Destroy.type, %T.loc167_4.2: @ModWith.loc168.%facet_type.loc167_37 (%facet_type.639)) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %require_complete.loc170: <witness> = require_complete_type %ModWith.type.loc167_35.1 [symbolic = %require_complete.loc170 (constants.%require_complete.a4a)]
// CHECK:STDOUT:   %assoc1: @ModWith.loc168.%ModWith.assoc_type (%ModWith.assoc_type.626) = assoc_entity element1, imports.%Core.import_ref.b56 [symbolic = %assoc1 (constants.%assoc1.6d1)]
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.loc170: <witness> = lookup_impl_witness %T.loc167_4.1, @ModWith.1, @ModWith.1(%U.as_type.loc167_35.1) [symbolic = %ModWith.lookup_impl_witness.loc170 (constants.%ModWith.lookup_impl_witness.e17)]
// CHECK:STDOUT:   %ModWith.facet: @ModWith.loc168.%ModWith.type.loc167_35.1 (%ModWith.type.937) = facet_value %T.as_type.loc168_29.1, (%ModWith.lookup_impl_witness.loc170) [symbolic = %ModWith.facet (constants.%ModWith.facet.874)]
// CHECK:STDOUT:   %ModWith.WithSelf.Op.type: type = fn_type @ModWith.WithSelf.Op, @ModWith.WithSelf(%U.as_type.loc167_35.1, %ModWith.facet) [symbolic = %ModWith.WithSelf.Op.type (constants.%ModWith.WithSelf.Op.type.314)]
// CHECK:STDOUT:   %.loc170_5.6: type = fn_type_with_self_type %ModWith.WithSelf.Op.type, %ModWith.facet [symbolic = %.loc170_5.6 (constants.%.195)]
// CHECK:STDOUT:   %impl.elem1.loc170_5.2: @ModWith.loc168.%.loc170_5.6 (%.195) = impl_witness_access %ModWith.lookup_impl_witness.loc170, element1 [symbolic = %impl.elem1.loc170_5.2 (constants.%impl.elem1.715)]
// CHECK:STDOUT:   %specific_impl_fn.loc170_5.3: <specific function> = specific_impl_function %impl.elem1.loc170_5.2, @ModWith.WithSelf.Op(%U.as_type.loc167_35.1, %ModWith.facet) [symbolic = %specific_impl_fn.loc170_5.3 (constants.%specific_impl_fn.0b2)]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Result.loc165_9.1) [symbolic = %Destroy.WithSelf.Op.type (constants.%Destroy.WithSelf.Op.type.d3ece9.2)]
// CHECK:STDOUT:   %.loc170_5.7: type = fn_type_with_self_type %Destroy.WithSelf.Op.type, %Result.loc165_9.1 [symbolic = %.loc170_5.7 (constants.%.53a)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %Result.loc165_9.1, @Destroy [symbolic = %Destroy.lookup_impl_witness (constants.%Destroy.lookup_impl_witness)]
// CHECK:STDOUT:   %impl.elem0.loc170_5.2: @ModWith.loc168.%.loc170_5.7 (%.53a) = impl_witness_access %Destroy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc170_5.2 (constants.%impl.elem0.0f9)]
// CHECK:STDOUT:   %specific_impl_fn.loc170_5.4: <specific function> = specific_impl_function %impl.elem0.loc170_5.2, @Destroy.WithSelf.Op(%Result.loc165_9.1) [symbolic = %specific_impl_fn.loc170_5.4 (constants.%specific_impl_fn.afc)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%result.param: @ModWith.loc168.%Result.as_type.loc167_53.1 (%Result.as_type), %x.param: @ModWith.loc168.%T.as_type.loc168_29.1 (%T.as_type.1c2), %y.param: @ModWith.loc168.%U.as_type.loc167_35.1 (%U.as_type.b61)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: @ModWith.loc168.%T.as_type.loc168_29.1 (%T.as_type.1c2) = name_ref x, %x
// CHECK:STDOUT:     %y.ref: @ModWith.loc168.%U.as_type.loc167_35.1 (%U.as_type.b61) = name_ref y, %y
// CHECK:STDOUT:     %ModWith.type.loc170: type = facet_type <@ModWith.1, @ModWith.1(constants.%U.as_type.b61)> [symbolic = %ModWith.type.loc167_35.1 (constants.%ModWith.type.937)]
// CHECK:STDOUT:     %.loc170_5.1: @ModWith.loc168.%ModWith.assoc_type (%ModWith.assoc_type.626) = specific_constant imports.%Core.import_ref.ffe, @ModWith.WithSelf(constants.%U.as_type.b61, constants.%Self.0ef) [symbolic = %assoc1 (constants.%assoc1.6d1)]
// CHECK:STDOUT:     %Op.ref: @ModWith.loc168.%ModWith.assoc_type (%ModWith.assoc_type.626) = name_ref Op, %.loc170_5.1 [symbolic = %assoc1 (constants.%assoc1.6d1)]
// CHECK:STDOUT:     %impl.elem1.loc170_5.1: @ModWith.loc168.%.loc170_5.6 (%.195) = impl_witness_access constants.%ModWith.lookup_impl_witness.e17, element1 [symbolic = %impl.elem1.loc170_5.2 (constants.%impl.elem1.715)]
// CHECK:STDOUT:     %bound_method.loc170_5.1: <bound method> = bound_method %x.ref, %impl.elem1.loc170_5.1
// CHECK:STDOUT:     %specific_impl_fn.loc170_5.1: <specific function> = specific_impl_function %impl.elem1.loc170_5.1, @ModWith.WithSelf.Op(constants.%U.as_type.b61, constants.%ModWith.facet.874) [symbolic = %specific_impl_fn.loc170_5.3 (constants.%specific_impl_fn.0b2)]
// CHECK:STDOUT:     %bound_method.loc170_5.2: <bound method> = bound_method %x.ref, %specific_impl_fn.loc170_5.1
// CHECK:STDOUT:     %.loc170_5.2: ref @ModWith.loc168.%Result.as_type.loc167_53.1 (%Result.as_type) = temporary_storage
// CHECK:STDOUT:     %ModWith.WithSelf.Op.call: init @ModWith.loc168.%Result.as_type.loc167_53.1 (%Result.as_type) to %.loc170_5.2 = call %bound_method.loc170_5.2(%x.ref, %y.ref)
// CHECK:STDOUT:     %.loc170_5.3: ref @ModWith.loc168.%Result.as_type.loc167_53.1 (%Result.as_type) = temporary %.loc170_5.2, %ModWith.WithSelf.Op.call
// CHECK:STDOUT:     %impl.elem0.loc170_5.1: @ModWith.loc168.%.loc170_5.7 (%.53a) = impl_witness_access constants.%Destroy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc170_5.2 (constants.%impl.elem0.0f9)]
// CHECK:STDOUT:     %bound_method.loc170_5.3: <bound method> = bound_method %.loc170_5.3, %impl.elem0.loc170_5.1
// CHECK:STDOUT:     %.loc170_5.4: %Destroy.type = converted constants.%Result.as_type, constants.%Result [symbolic = %Result.loc165_9.1 (constants.%Result)]
// CHECK:STDOUT:     %.loc170_5.5: %Destroy.type = converted constants.%Result.as_type, constants.%Result [symbolic = %Result.loc165_9.1 (constants.%Result)]
// CHECK:STDOUT:     %specific_impl_fn.loc170_5.2: <specific function> = specific_impl_function %impl.elem0.loc170_5.1, @Destroy.WithSelf.Op(constants.%Result) [symbolic = %specific_impl_fn.loc170_5.4 (constants.%specific_impl_fn.afc)]
// CHECK:STDOUT:     %bound_method.loc170_5.4: <bound method> = bound_method %.loc170_5.3, %specific_impl_fn.loc170_5.2
// CHECK:STDOUT:     %Destroy.WithSelf.Op.call: init %empty_tuple.type = call %bound_method.loc170_5.4(%.loc170_5.3)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @AddAssignWith.loc225(%U.loc225_19.2: type, %T.loc225_28.2: @AddAssignWith.loc225.%AddAssignWith.type.loc225_50.1 (%AddAssignWith.type.2c8890.2)) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %require_complete.loc227: <witness> = require_complete_type %AddAssignWith.type.loc225_50.1 [symbolic = %require_complete.loc227 (constants.%require_complete.2a8)]
// CHECK:STDOUT:   %AddAssignWith.assoc_type: type = assoc_entity_type @AddAssignWith.1, @AddAssignWith.1(%U.loc225_19.1) [symbolic = %AddAssignWith.assoc_type (constants.%AddAssignWith.assoc_type.8ddaf4.2)]
// CHECK:STDOUT:   %assoc0: @AddAssignWith.loc225.%AddAssignWith.assoc_type (%AddAssignWith.assoc_type.8ddaf4.2) = assoc_entity element0, imports.%Core.import_ref.3b2 [symbolic = %assoc0 (constants.%assoc0.8c5739.2)]
// CHECK:STDOUT:   %AddAssignWith.WithSelf.Op.type: type = fn_type @AddAssignWith.WithSelf.Op, @AddAssignWith.WithSelf(%U.loc225_19.1, %T.loc225_28.1) [symbolic = %AddAssignWith.WithSelf.Op.type (constants.%AddAssignWith.WithSelf.Op.type.cd574e.3)]
// CHECK:STDOUT:   %.loc227_5.2: type = fn_type_with_self_type %AddAssignWith.WithSelf.Op.type, %T.loc225_28.1 [symbolic = %.loc227_5.2 (constants.%.9b1)]
// CHECK:STDOUT:   %AddAssignWith.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc225_28.1, @AddAssignWith.1, @AddAssignWith.1(%U.loc225_19.1) [symbolic = %AddAssignWith.lookup_impl_witness (constants.%AddAssignWith.lookup_impl_witness)]
// CHECK:STDOUT:   %impl.elem0.loc227_5.2: @AddAssignWith.loc225.%.loc227_5.2 (%.9b1) = impl_witness_access %AddAssignWith.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc227_5.2 (constants.%impl.elem0.985)]
// CHECK:STDOUT:   %specific_impl_fn.loc227_5.2: <specific function> = specific_impl_function %impl.elem0.loc227_5.2, @AddAssignWith.WithSelf.Op(%U.loc225_19.1, %T.loc225_28.1) [symbolic = %specific_impl_fn.loc227_5.2 (constants.%specific_impl_fn.d36)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: ref @AddAssignWith.loc225.%T.as_type.loc225_60.1 (%T.as_type.50c), %y.param: @AddAssignWith.loc225.%U.loc225_19.1 (%U.67d)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: ref @AddAssignWith.loc225.%T.as_type.loc225_60.1 (%T.as_type.50c) = name_ref x, %x
// CHECK:STDOUT:     %y.ref: @AddAssignWith.loc225.%U.loc225_19.1 (%U.67d) = name_ref y, %y
// CHECK:STDOUT:     %AddAssignWith.type.loc227: type = facet_type <@AddAssignWith.1, @AddAssignWith.1(constants.%U.67d)> [symbolic = %AddAssignWith.type.loc225_50.1 (constants.%AddAssignWith.type.2c8890.2)]
// CHECK:STDOUT:     %.loc227_5.1: @AddAssignWith.loc225.%AddAssignWith.assoc_type (%AddAssignWith.assoc_type.8ddaf4.2) = specific_constant imports.%Core.import_ref.bdd, @AddAssignWith.WithSelf(constants.%U.67d, constants.%Self.792732.1) [symbolic = %assoc0 (constants.%assoc0.8c5739.2)]
// CHECK:STDOUT:     %Op.ref: @AddAssignWith.loc225.%AddAssignWith.assoc_type (%AddAssignWith.assoc_type.8ddaf4.2) = name_ref Op, %.loc227_5.1 [symbolic = %assoc0 (constants.%assoc0.8c5739.2)]
// CHECK:STDOUT:     %impl.elem0.loc227_5.1: @AddAssignWith.loc225.%.loc227_5.2 (%.9b1) = impl_witness_access constants.%AddAssignWith.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc227_5.2 (constants.%impl.elem0.985)]
// CHECK:STDOUT:     %bound_method.loc227_5.1: <bound method> = bound_method %x.ref, %impl.elem0.loc227_5.1
// CHECK:STDOUT:     %specific_impl_fn.loc227_5.1: <specific function> = specific_impl_function %impl.elem0.loc227_5.1, @AddAssignWith.WithSelf.Op(constants.%U.67d, constants.%T.792) [symbolic = %specific_impl_fn.loc227_5.2 (constants.%specific_impl_fn.d36)]
// CHECK:STDOUT:     %bound_method.loc227_5.2: <bound method> = bound_method %x.ref, %specific_impl_fn.loc227_5.1
// CHECK:STDOUT:     %AddAssignWith.WithSelf.Op.call: init %empty_tuple.type = call %bound_method.loc227_5.2(%x.ref, %y.ref)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @SubAssignWith.loc231(%U.loc231_19.2: type, %T.loc231_28.2: @SubAssignWith.loc231.%SubAssignWith.type.loc231_50.1 (%SubAssignWith.type.7e53cf.2)) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %require_complete.loc233: <witness> = require_complete_type %SubAssignWith.type.loc231_50.1 [symbolic = %require_complete.loc233 (constants.%require_complete.862)]
// CHECK:STDOUT:   %SubAssignWith.assoc_type: type = assoc_entity_type @SubAssignWith.1, @SubAssignWith.1(%U.loc231_19.1) [symbolic = %SubAssignWith.assoc_type (constants.%SubAssignWith.assoc_type.6e3877.2)]
// CHECK:STDOUT:   %assoc0: @SubAssignWith.loc231.%SubAssignWith.assoc_type (%SubAssignWith.assoc_type.6e3877.2) = assoc_entity element0, imports.%Core.import_ref.8f2 [symbolic = %assoc0 (constants.%assoc0.701e5f.2)]
// CHECK:STDOUT:   %SubAssignWith.WithSelf.Op.type: type = fn_type @SubAssignWith.WithSelf.Op, @SubAssignWith.WithSelf(%U.loc231_19.1, %T.loc231_28.1) [symbolic = %SubAssignWith.WithSelf.Op.type (constants.%SubAssignWith.WithSelf.Op.type.42f7cc.3)]
// CHECK:STDOUT:   %.loc233_5.2: type = fn_type_with_self_type %SubAssignWith.WithSelf.Op.type, %T.loc231_28.1 [symbolic = %.loc233_5.2 (constants.%.e56)]
// CHECK:STDOUT:   %SubAssignWith.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc231_28.1, @SubAssignWith.1, @SubAssignWith.1(%U.loc231_19.1) [symbolic = %SubAssignWith.lookup_impl_witness (constants.%SubAssignWith.lookup_impl_witness)]
// CHECK:STDOUT:   %impl.elem0.loc233_5.2: @SubAssignWith.loc231.%.loc233_5.2 (%.e56) = impl_witness_access %SubAssignWith.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc233_5.2 (constants.%impl.elem0.dd3)]
// CHECK:STDOUT:   %specific_impl_fn.loc233_5.2: <specific function> = specific_impl_function %impl.elem0.loc233_5.2, @SubAssignWith.WithSelf.Op(%U.loc231_19.1, %T.loc231_28.1) [symbolic = %specific_impl_fn.loc233_5.2 (constants.%specific_impl_fn.5da)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: ref @SubAssignWith.loc231.%T.as_type.loc231_60.1 (%T.as_type.3ff), %y.param: @SubAssignWith.loc231.%U.loc231_19.1 (%U.67d)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: ref @SubAssignWith.loc231.%T.as_type.loc231_60.1 (%T.as_type.3ff) = name_ref x, %x
// CHECK:STDOUT:     %y.ref: @SubAssignWith.loc231.%U.loc231_19.1 (%U.67d) = name_ref y, %y
// CHECK:STDOUT:     %SubAssignWith.type.loc233: type = facet_type <@SubAssignWith.1, @SubAssignWith.1(constants.%U.67d)> [symbolic = %SubAssignWith.type.loc231_50.1 (constants.%SubAssignWith.type.7e53cf.2)]
// CHECK:STDOUT:     %.loc233_5.1: @SubAssignWith.loc231.%SubAssignWith.assoc_type (%SubAssignWith.assoc_type.6e3877.2) = specific_constant imports.%Core.import_ref.509, @SubAssignWith.WithSelf(constants.%U.67d, constants.%Self.f6c9aa.1) [symbolic = %assoc0 (constants.%assoc0.701e5f.2)]
// CHECK:STDOUT:     %Op.ref: @SubAssignWith.loc231.%SubAssignWith.assoc_type (%SubAssignWith.assoc_type.6e3877.2) = name_ref Op, %.loc233_5.1 [symbolic = %assoc0 (constants.%assoc0.701e5f.2)]
// CHECK:STDOUT:     %impl.elem0.loc233_5.1: @SubAssignWith.loc231.%.loc233_5.2 (%.e56) = impl_witness_access constants.%SubAssignWith.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc233_5.2 (constants.%impl.elem0.dd3)]
// CHECK:STDOUT:     %bound_method.loc233_5.1: <bound method> = bound_method %x.ref, %impl.elem0.loc233_5.1
// CHECK:STDOUT:     %specific_impl_fn.loc233_5.1: <specific function> = specific_impl_function %impl.elem0.loc233_5.1, @SubAssignWith.WithSelf.Op(constants.%U.67d, constants.%T.f6c) [symbolic = %specific_impl_fn.loc233_5.2 (constants.%specific_impl_fn.5da)]
// CHECK:STDOUT:     %bound_method.loc233_5.2: <bound method> = bound_method %x.ref, %specific_impl_fn.loc233_5.1
// CHECK:STDOUT:     %SubAssignWith.WithSelf.Op.call: init %empty_tuple.type = call %bound_method.loc233_5.2(%x.ref, %y.ref)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @MulAssignWith.loc237(%U.loc237_19.2: type, %T.loc237_28.2: @MulAssignWith.loc237.%MulAssignWith.type.loc237_50.1 (%MulAssignWith.type.3a0e4d.2)) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %require_complete.loc239: <witness> = require_complete_type %MulAssignWith.type.loc237_50.1 [symbolic = %require_complete.loc239 (constants.%require_complete.6b1)]
// CHECK:STDOUT:   %MulAssignWith.assoc_type: type = assoc_entity_type @MulAssignWith.1, @MulAssignWith.1(%U.loc237_19.1) [symbolic = %MulAssignWith.assoc_type (constants.%MulAssignWith.assoc_type.9aa18f.2)]
// CHECK:STDOUT:   %assoc0: @MulAssignWith.loc237.%MulAssignWith.assoc_type (%MulAssignWith.assoc_type.9aa18f.2) = assoc_entity element0, imports.%Core.import_ref.ba3 [symbolic = %assoc0 (constants.%assoc0.22264a.2)]
// CHECK:STDOUT:   %MulAssignWith.WithSelf.Op.type: type = fn_type @MulAssignWith.WithSelf.Op, @MulAssignWith.WithSelf(%U.loc237_19.1, %T.loc237_28.1) [symbolic = %MulAssignWith.WithSelf.Op.type (constants.%MulAssignWith.WithSelf.Op.type.5df5bd.3)]
// CHECK:STDOUT:   %.loc239_5.2: type = fn_type_with_self_type %MulAssignWith.WithSelf.Op.type, %T.loc237_28.1 [symbolic = %.loc239_5.2 (constants.%.2db)]
// CHECK:STDOUT:   %MulAssignWith.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc237_28.1, @MulAssignWith.1, @MulAssignWith.1(%U.loc237_19.1) [symbolic = %MulAssignWith.lookup_impl_witness (constants.%MulAssignWith.lookup_impl_witness)]
// CHECK:STDOUT:   %impl.elem0.loc239_5.2: @MulAssignWith.loc237.%.loc239_5.2 (%.2db) = impl_witness_access %MulAssignWith.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc239_5.2 (constants.%impl.elem0.c0f)]
// CHECK:STDOUT:   %specific_impl_fn.loc239_5.2: <specific function> = specific_impl_function %impl.elem0.loc239_5.2, @MulAssignWith.WithSelf.Op(%U.loc237_19.1, %T.loc237_28.1) [symbolic = %specific_impl_fn.loc239_5.2 (constants.%specific_impl_fn.54d)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: ref @MulAssignWith.loc237.%T.as_type.loc237_60.1 (%T.as_type.1de), %y.param: @MulAssignWith.loc237.%U.loc237_19.1 (%U.67d)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: ref @MulAssignWith.loc237.%T.as_type.loc237_60.1 (%T.as_type.1de) = name_ref x, %x
// CHECK:STDOUT:     %y.ref: @MulAssignWith.loc237.%U.loc237_19.1 (%U.67d) = name_ref y, %y
// CHECK:STDOUT:     %MulAssignWith.type.loc239: type = facet_type <@MulAssignWith.1, @MulAssignWith.1(constants.%U.67d)> [symbolic = %MulAssignWith.type.loc237_50.1 (constants.%MulAssignWith.type.3a0e4d.2)]
// CHECK:STDOUT:     %.loc239_5.1: @MulAssignWith.loc237.%MulAssignWith.assoc_type (%MulAssignWith.assoc_type.9aa18f.2) = specific_constant imports.%Core.import_ref.b6e, @MulAssignWith.WithSelf(constants.%U.67d, constants.%Self.64dcb0.1) [symbolic = %assoc0 (constants.%assoc0.22264a.2)]
// CHECK:STDOUT:     %Op.ref: @MulAssignWith.loc237.%MulAssignWith.assoc_type (%MulAssignWith.assoc_type.9aa18f.2) = name_ref Op, %.loc239_5.1 [symbolic = %assoc0 (constants.%assoc0.22264a.2)]
// CHECK:STDOUT:     %impl.elem0.loc239_5.1: @MulAssignWith.loc237.%.loc239_5.2 (%.2db) = impl_witness_access constants.%MulAssignWith.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc239_5.2 (constants.%impl.elem0.c0f)]
// CHECK:STDOUT:     %bound_method.loc239_5.1: <bound method> = bound_method %x.ref, %impl.elem0.loc239_5.1
// CHECK:STDOUT:     %specific_impl_fn.loc239_5.1: <specific function> = specific_impl_function %impl.elem0.loc239_5.1, @MulAssignWith.WithSelf.Op(constants.%U.67d, constants.%T.64d) [symbolic = %specific_impl_fn.loc239_5.2 (constants.%specific_impl_fn.54d)]
// CHECK:STDOUT:     %bound_method.loc239_5.2: <bound method> = bound_method %x.ref, %specific_impl_fn.loc239_5.1
// CHECK:STDOUT:     %MulAssignWith.WithSelf.Op.call: init %empty_tuple.type = call %bound_method.loc239_5.2(%x.ref, %y.ref)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @DivAssignWith.loc243(%U.loc243_19.2: type, %T.loc243_28.2: @DivAssignWith.loc243.%DivAssignWith.type.loc243_50.1 (%DivAssignWith.type.7e58ca.2)) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %require_complete.loc245: <witness> = require_complete_type %DivAssignWith.type.loc243_50.1 [symbolic = %require_complete.loc245 (constants.%require_complete.c5c)]
// CHECK:STDOUT:   %DivAssignWith.assoc_type: type = assoc_entity_type @DivAssignWith.1, @DivAssignWith.1(%U.loc243_19.1) [symbolic = %DivAssignWith.assoc_type (constants.%DivAssignWith.assoc_type.bb68ec.2)]
// CHECK:STDOUT:   %assoc0: @DivAssignWith.loc243.%DivAssignWith.assoc_type (%DivAssignWith.assoc_type.bb68ec.2) = assoc_entity element0, imports.%Core.import_ref.a5d [symbolic = %assoc0 (constants.%assoc0.32f100.2)]
// CHECK:STDOUT:   %DivAssignWith.WithSelf.Op.type: type = fn_type @DivAssignWith.WithSelf.Op, @DivAssignWith.WithSelf(%U.loc243_19.1, %T.loc243_28.1) [symbolic = %DivAssignWith.WithSelf.Op.type (constants.%DivAssignWith.WithSelf.Op.type.ff5775.3)]
// CHECK:STDOUT:   %.loc245_5.2: type = fn_type_with_self_type %DivAssignWith.WithSelf.Op.type, %T.loc243_28.1 [symbolic = %.loc245_5.2 (constants.%.754)]
// CHECK:STDOUT:   %DivAssignWith.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc243_28.1, @DivAssignWith.1, @DivAssignWith.1(%U.loc243_19.1) [symbolic = %DivAssignWith.lookup_impl_witness (constants.%DivAssignWith.lookup_impl_witness)]
// CHECK:STDOUT:   %impl.elem0.loc245_5.2: @DivAssignWith.loc243.%.loc245_5.2 (%.754) = impl_witness_access %DivAssignWith.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc245_5.2 (constants.%impl.elem0.0c4)]
// CHECK:STDOUT:   %specific_impl_fn.loc245_5.2: <specific function> = specific_impl_function %impl.elem0.loc245_5.2, @DivAssignWith.WithSelf.Op(%U.loc243_19.1, %T.loc243_28.1) [symbolic = %specific_impl_fn.loc245_5.2 (constants.%specific_impl_fn.011)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: ref @DivAssignWith.loc243.%T.as_type.loc243_60.1 (%T.as_type.e57), %y.param: @DivAssignWith.loc243.%U.loc243_19.1 (%U.67d)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: ref @DivAssignWith.loc243.%T.as_type.loc243_60.1 (%T.as_type.e57) = name_ref x, %x
// CHECK:STDOUT:     %y.ref: @DivAssignWith.loc243.%U.loc243_19.1 (%U.67d) = name_ref y, %y
// CHECK:STDOUT:     %DivAssignWith.type.loc245: type = facet_type <@DivAssignWith.1, @DivAssignWith.1(constants.%U.67d)> [symbolic = %DivAssignWith.type.loc243_50.1 (constants.%DivAssignWith.type.7e58ca.2)]
// CHECK:STDOUT:     %.loc245_5.1: @DivAssignWith.loc243.%DivAssignWith.assoc_type (%DivAssignWith.assoc_type.bb68ec.2) = specific_constant imports.%Core.import_ref.988, @DivAssignWith.WithSelf(constants.%U.67d, constants.%Self.c33c81.1) [symbolic = %assoc0 (constants.%assoc0.32f100.2)]
// CHECK:STDOUT:     %Op.ref: @DivAssignWith.loc243.%DivAssignWith.assoc_type (%DivAssignWith.assoc_type.bb68ec.2) = name_ref Op, %.loc245_5.1 [symbolic = %assoc0 (constants.%assoc0.32f100.2)]
// CHECK:STDOUT:     %impl.elem0.loc245_5.1: @DivAssignWith.loc243.%.loc245_5.2 (%.754) = impl_witness_access constants.%DivAssignWith.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc245_5.2 (constants.%impl.elem0.0c4)]
// CHECK:STDOUT:     %bound_method.loc245_5.1: <bound method> = bound_method %x.ref, %impl.elem0.loc245_5.1
// CHECK:STDOUT:     %specific_impl_fn.loc245_5.1: <specific function> = specific_impl_function %impl.elem0.loc245_5.1, @DivAssignWith.WithSelf.Op(constants.%U.67d, constants.%T.c33) [symbolic = %specific_impl_fn.loc245_5.2 (constants.%specific_impl_fn.011)]
// CHECK:STDOUT:     %bound_method.loc245_5.2: <bound method> = bound_method %x.ref, %specific_impl_fn.loc245_5.1
// CHECK:STDOUT:     %DivAssignWith.WithSelf.Op.call: init %empty_tuple.type = call %bound_method.loc245_5.2(%x.ref, %y.ref)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @ModAssignWith.loc249(%U.loc249_19.2: type, %T.loc249_28.2: @ModAssignWith.loc249.%ModAssignWith.type.loc249_50.1 (%ModAssignWith.type.778d68.2)) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %require_complete.loc251: <witness> = require_complete_type %ModAssignWith.type.loc249_50.1 [symbolic = %require_complete.loc251 (constants.%require_complete.680)]
// CHECK:STDOUT:   %ModAssignWith.assoc_type: type = assoc_entity_type @ModAssignWith.1, @ModAssignWith.1(%U.loc249_19.1) [symbolic = %ModAssignWith.assoc_type (constants.%ModAssignWith.assoc_type.bc79ec.2)]
// CHECK:STDOUT:   %assoc0: @ModAssignWith.loc249.%ModAssignWith.assoc_type (%ModAssignWith.assoc_type.bc79ec.2) = assoc_entity element0, imports.%Core.import_ref.cf6 [symbolic = %assoc0 (constants.%assoc0.8c0719.2)]
// CHECK:STDOUT:   %ModAssignWith.WithSelf.Op.type: type = fn_type @ModAssignWith.WithSelf.Op, @ModAssignWith.WithSelf(%U.loc249_19.1, %T.loc249_28.1) [symbolic = %ModAssignWith.WithSelf.Op.type (constants.%ModAssignWith.WithSelf.Op.type.ec4109.3)]
// CHECK:STDOUT:   %.loc251_5.2: type = fn_type_with_self_type %ModAssignWith.WithSelf.Op.type, %T.loc249_28.1 [symbolic = %.loc251_5.2 (constants.%.965)]
// CHECK:STDOUT:   %ModAssignWith.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc249_28.1, @ModAssignWith.1, @ModAssignWith.1(%U.loc249_19.1) [symbolic = %ModAssignWith.lookup_impl_witness (constants.%ModAssignWith.lookup_impl_witness)]
// CHECK:STDOUT:   %impl.elem0.loc251_5.2: @ModAssignWith.loc249.%.loc251_5.2 (%.965) = impl_witness_access %ModAssignWith.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc251_5.2 (constants.%impl.elem0.ec7)]
// CHECK:STDOUT:   %specific_impl_fn.loc251_5.2: <specific function> = specific_impl_function %impl.elem0.loc251_5.2, @ModAssignWith.WithSelf.Op(%U.loc249_19.1, %T.loc249_28.1) [symbolic = %specific_impl_fn.loc251_5.2 (constants.%specific_impl_fn.cf7)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: ref @ModAssignWith.loc249.%T.as_type.loc249_60.1 (%T.as_type.ce6), %y.param: @ModAssignWith.loc249.%U.loc249_19.1 (%U.67d)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: ref @ModAssignWith.loc249.%T.as_type.loc249_60.1 (%T.as_type.ce6) = name_ref x, %x
// CHECK:STDOUT:     %y.ref: @ModAssignWith.loc249.%U.loc249_19.1 (%U.67d) = name_ref y, %y
// CHECK:STDOUT:     %ModAssignWith.type.loc251: type = facet_type <@ModAssignWith.1, @ModAssignWith.1(constants.%U.67d)> [symbolic = %ModAssignWith.type.loc249_50.1 (constants.%ModAssignWith.type.778d68.2)]
// CHECK:STDOUT:     %.loc251_5.1: @ModAssignWith.loc249.%ModAssignWith.assoc_type (%ModAssignWith.assoc_type.bc79ec.2) = specific_constant imports.%Core.import_ref.3ff, @ModAssignWith.WithSelf(constants.%U.67d, constants.%Self.bca00b.1) [symbolic = %assoc0 (constants.%assoc0.8c0719.2)]
// CHECK:STDOUT:     %Op.ref: @ModAssignWith.loc249.%ModAssignWith.assoc_type (%ModAssignWith.assoc_type.bc79ec.2) = name_ref Op, %.loc251_5.1 [symbolic = %assoc0 (constants.%assoc0.8c0719.2)]
// CHECK:STDOUT:     %impl.elem0.loc251_5.1: @ModAssignWith.loc249.%.loc251_5.2 (%.965) = impl_witness_access constants.%ModAssignWith.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc251_5.2 (constants.%impl.elem0.ec7)]
// CHECK:STDOUT:     %bound_method.loc251_5.1: <bound method> = bound_method %x.ref, %impl.elem0.loc251_5.1
// CHECK:STDOUT:     %specific_impl_fn.loc251_5.1: <specific function> = specific_impl_function %impl.elem0.loc251_5.1, @ModAssignWith.WithSelf.Op(constants.%U.67d, constants.%T.bca) [symbolic = %specific_impl_fn.loc251_5.2 (constants.%specific_impl_fn.cf7)]
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
// CHECK:STDOUT:   %Inc.WithSelf.Op.type: type = fn_type @Inc.WithSelf.Op, @Inc.WithSelf(%T.loc278_13.1) [symbolic = %Inc.WithSelf.Op.type (constants.%Inc.WithSelf.Op.type.e10ef6.2)]
// CHECK:STDOUT:   %.loc280_3.3: type = fn_type_with_self_type %Inc.WithSelf.Op.type, %T.loc278_13.1 [symbolic = %.loc280_3.3 (constants.%.a08)]
// CHECK:STDOUT:   %Inc.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc278_13.1, @Inc [symbolic = %Inc.lookup_impl_witness (constants.%Inc.lookup_impl_witness)]
// CHECK:STDOUT:   %impl.elem0.loc280_3.2: @TestInc.%.loc280_3.3 (%.a08) = impl_witness_access %Inc.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc280_3.2 (constants.%impl.elem0.b92)]
// CHECK:STDOUT:   %specific_impl_fn.loc280_3.2: <specific function> = specific_impl_function %impl.elem0.loc280_3.2, @Inc.WithSelf.Op(%T.loc278_13.1) [symbolic = %specific_impl_fn.loc280_3.2 (constants.%specific_impl_fn.855)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: ref @TestInc.%T.as_type.loc278_32.1 (%T.as_type.232)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: ref @TestInc.%T.as_type.loc278_32.1 (%T.as_type.232) = name_ref x, %x
// CHECK:STDOUT:     %impl.elem0.loc280_3.1: @TestInc.%.loc280_3.3 (%.a08) = impl_witness_access constants.%Inc.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc280_3.2 (constants.%impl.elem0.b92)]
// CHECK:STDOUT:     %bound_method.loc280_3.1: <bound method> = bound_method %x.ref, %impl.elem0.loc280_3.1
// CHECK:STDOUT:     %.loc280_3.1: %Inc.type = converted constants.%T.as_type.232, constants.%T.8f3 [symbolic = %T.loc278_13.1 (constants.%T.8f3)]
// CHECK:STDOUT:     %.loc280_3.2: %Inc.type = converted constants.%T.as_type.232, constants.%T.8f3 [symbolic = %T.loc278_13.1 (constants.%T.8f3)]
// CHECK:STDOUT:     %specific_impl_fn.loc280_3.1: <specific function> = specific_impl_function %impl.elem0.loc280_3.1, @Inc.WithSelf.Op(constants.%T.8f3) [symbolic = %specific_impl_fn.loc280_3.2 (constants.%specific_impl_fn.855)]
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
// CHECK:STDOUT:   %Dec.WithSelf.Op.type: type = fn_type @Dec.WithSelf.Op, @Dec.WithSelf(%T.loc284_13.1) [symbolic = %Dec.WithSelf.Op.type (constants.%Dec.WithSelf.Op.type.bfa830.2)]
// CHECK:STDOUT:   %.loc286_3.3: type = fn_type_with_self_type %Dec.WithSelf.Op.type, %T.loc284_13.1 [symbolic = %.loc286_3.3 (constants.%.aae)]
// CHECK:STDOUT:   %Dec.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc284_13.1, @Dec [symbolic = %Dec.lookup_impl_witness (constants.%Dec.lookup_impl_witness)]
// CHECK:STDOUT:   %impl.elem0.loc286_3.2: @TestDec.%.loc286_3.3 (%.aae) = impl_witness_access %Dec.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc286_3.2 (constants.%impl.elem0.950)]
// CHECK:STDOUT:   %specific_impl_fn.loc286_3.2: <specific function> = specific_impl_function %impl.elem0.loc286_3.2, @Dec.WithSelf.Op(%T.loc284_13.1) [symbolic = %specific_impl_fn.loc286_3.2 (constants.%specific_impl_fn.539)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: ref @TestDec.%T.as_type.loc284_32.1 (%T.as_type.c32)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: ref @TestDec.%T.as_type.loc284_32.1 (%T.as_type.c32) = name_ref x, %x
// CHECK:STDOUT:     %impl.elem0.loc286_3.1: @TestDec.%.loc286_3.3 (%.aae) = impl_witness_access constants.%Dec.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc286_3.2 (constants.%impl.elem0.950)]
// CHECK:STDOUT:     %bound_method.loc286_3.1: <bound method> = bound_method %x.ref, %impl.elem0.loc286_3.1
// CHECK:STDOUT:     %.loc286_3.1: %Dec.type = converted constants.%T.as_type.c32, constants.%T.e8e [symbolic = %T.loc284_13.1 (constants.%T.e8e)]
// CHECK:STDOUT:     %.loc286_3.2: %Dec.type = converted constants.%T.as_type.c32, constants.%T.e8e [symbolic = %T.loc284_13.1 (constants.%T.e8e)]
// CHECK:STDOUT:     %specific_impl_fn.loc286_3.1: <specific function> = specific_impl_function %impl.elem0.loc286_3.1, @Dec.WithSelf.Op(constants.%T.e8e) [symbolic = %specific_impl_fn.loc286_3.2 (constants.%specific_impl_fn.539)]
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
// CHECK:STDOUT:   %Negate.WithSelf.Op.type: type = fn_type @Negate.WithSelf.Op, @Negate.WithSelf(%Negate.facet.loc296_3.3) [symbolic = %Negate.WithSelf.Op.type (constants.%Negate.WithSelf.Op.type.d8a)]
// CHECK:STDOUT:   %.loc296_3.7: type = fn_type_with_self_type %Negate.WithSelf.Op.type, %Negate.facet.loc296_3.3 [symbolic = %.loc296_3.7 (constants.%.16a)]
// CHECK:STDOUT:   %impl.elem1.loc296_3.2: @TestNegate.%.loc296_3.7 (%.16a) = impl_witness_access %Negate.lookup_impl_witness, element1 [symbolic = %impl.elem1.loc296_3.2 (constants.%impl.elem1.ee9)]
// CHECK:STDOUT:   %specific_impl_fn.loc296_3.3: <specific function> = specific_impl_function %impl.elem1.loc296_3.2, @Negate.WithSelf.Op(%Negate.facet.loc296_3.3) [symbolic = %specific_impl_fn.loc296_3.3 (constants.%specific_impl_fn.923)]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Result.loc292_9.1) [symbolic = %Destroy.WithSelf.Op.type (constants.%Destroy.WithSelf.Op.type.d3ece9.2)]
// CHECK:STDOUT:   %.loc296_3.8: type = fn_type_with_self_type %Destroy.WithSelf.Op.type, %Result.loc292_9.1 [symbolic = %.loc296_3.8 (constants.%.53a)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %Result.loc292_9.1, @Destroy [symbolic = %Destroy.lookup_impl_witness (constants.%Destroy.lookup_impl_witness)]
// CHECK:STDOUT:   %impl.elem0.loc296_3.2: @TestNegate.%.loc296_3.8 (%.53a) = impl_witness_access %Destroy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc296_3.2 (constants.%impl.elem0.0f9)]
// CHECK:STDOUT:   %specific_impl_fn.loc296_3.4: <specific function> = specific_impl_function %impl.elem0.loc296_3.2, @Destroy.WithSelf.Op(%Result.loc292_9.1) [symbolic = %specific_impl_fn.loc296_3.4 (constants.%specific_impl_fn.afc)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%y.param: @TestNegate.%Result.as_type.loc293_49.1 (%Result.as_type), %x.param: @TestNegate.%T.as_type.loc294_24.1 (%T.as_type.684)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: @TestNegate.%T.as_type.loc294_24.1 (%T.as_type.684) = name_ref x, %x
// CHECK:STDOUT:     %impl.elem1.loc296_3.1: @TestNegate.%.loc296_3.7 (%.16a) = impl_witness_access constants.%Negate.lookup_impl_witness.218, element1 [symbolic = %impl.elem1.loc296_3.2 (constants.%impl.elem1.ee9)]
// CHECK:STDOUT:     %bound_method.loc296_3.1: <bound method> = bound_method %x.ref, %impl.elem1.loc296_3.1
// CHECK:STDOUT:     %Negate.facet.loc296_3.1: %Negate.type = facet_value constants.%T.as_type.684, (constants.%Negate.lookup_impl_witness.218) [symbolic = %Negate.facet.loc296_3.3 (constants.%Negate.facet.5e5)]
// CHECK:STDOUT:     %.loc296_3.1: %Negate.type = converted constants.%T.as_type.684, %Negate.facet.loc296_3.1 [symbolic = %Negate.facet.loc296_3.3 (constants.%Negate.facet.5e5)]
// CHECK:STDOUT:     %Negate.facet.loc296_3.2: %Negate.type = facet_value constants.%T.as_type.684, (constants.%Negate.lookup_impl_witness.218) [symbolic = %Negate.facet.loc296_3.3 (constants.%Negate.facet.5e5)]
// CHECK:STDOUT:     %.loc296_3.2: %Negate.type = converted constants.%T.as_type.684, %Negate.facet.loc296_3.2 [symbolic = %Negate.facet.loc296_3.3 (constants.%Negate.facet.5e5)]
// CHECK:STDOUT:     %specific_impl_fn.loc296_3.1: <specific function> = specific_impl_function %impl.elem1.loc296_3.1, @Negate.WithSelf.Op(constants.%Negate.facet.5e5) [symbolic = %specific_impl_fn.loc296_3.3 (constants.%specific_impl_fn.923)]
// CHECK:STDOUT:     %bound_method.loc296_3.2: <bound method> = bound_method %x.ref, %specific_impl_fn.loc296_3.1
// CHECK:STDOUT:     %.loc296_3.3: ref @TestNegate.%Result.as_type.loc293_49.1 (%Result.as_type) = temporary_storage
// CHECK:STDOUT:     %Negate.WithSelf.Op.call: init @TestNegate.%Result.as_type.loc293_49.1 (%Result.as_type) to %.loc296_3.3 = call %bound_method.loc296_3.2(%x.ref)
// CHECK:STDOUT:     %.loc296_3.4: ref @TestNegate.%Result.as_type.loc293_49.1 (%Result.as_type) = temporary %.loc296_3.3, %Negate.WithSelf.Op.call
// CHECK:STDOUT:     %impl.elem0.loc296_3.1: @TestNegate.%.loc296_3.8 (%.53a) = impl_witness_access constants.%Destroy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc296_3.2 (constants.%impl.elem0.0f9)]
// CHECK:STDOUT:     %bound_method.loc296_3.3: <bound method> = bound_method %.loc296_3.4, %impl.elem0.loc296_3.1
// CHECK:STDOUT:     %.loc296_3.5: %Destroy.type = converted constants.%Result.as_type, constants.%Result [symbolic = %Result.loc292_9.1 (constants.%Result)]
// CHECK:STDOUT:     %.loc296_3.6: %Destroy.type = converted constants.%Result.as_type, constants.%Result [symbolic = %Result.loc292_9.1 (constants.%Result)]
// CHECK:STDOUT:     %specific_impl_fn.loc296_3.2: <specific function> = specific_impl_function %impl.elem0.loc296_3.1, @Destroy.WithSelf.Op(constants.%Result) [symbolic = %specific_impl_fn.loc296_3.4 (constants.%specific_impl_fn.afc)]
// CHECK:STDOUT:     %bound_method.loc296_3.4: <bound method> = bound_method %.loc296_3.4, %specific_impl_fn.loc296_3.2
// CHECK:STDOUT:     %Destroy.WithSelf.Op.call: init %empty_tuple.type = call %bound_method.loc296_3.4(%.loc296_3.4)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AddWith.loc124(constants.%Result, constants.%U.f40, constants.%T.ceb) {
// CHECK:STDOUT:   %Result.patt.loc121_9.2 => constants.%Result.patt
// CHECK:STDOUT:   %Result.loc121_9.1 => constants.%Result
// CHECK:STDOUT:   %U.patt.loc122_4.2 => constants.%U.patt.7a2
// CHECK:STDOUT:   %U.loc122_4.1 => constants.%U.f40
// CHECK:STDOUT:   %U.as_type.loc123_35.1 => constants.%U.as_type.b61
// CHECK:STDOUT:   %AddWith.type.loc123_35.1 => constants.%AddWith.type.9da
// CHECK:STDOUT:   %facet_type.loc123_19 => constants.%facet_type.b1e
// CHECK:STDOUT:   %.Self.frozen.loc123_37.1 => constants.%.Self.frozen.1ea
// CHECK:STDOUT:   %require_complete.loc123 => constants.%require_complete.c7d
// CHECK:STDOUT:   %.Self.frozen.as_type => constants.%.Self.frozen.as_type.015
// CHECK:STDOUT:   %AddWith.assoc_type => constants.%AddWith.assoc_type.bc8
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.21f
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.loc123_43.1 => constants.%AddWith.lookup_impl_witness.2f6
// CHECK:STDOUT:   %impl.elem0.loc123_43.1 => constants.%impl.elem0.1c6
// CHECK:STDOUT:   %Result.as_type.loc123_53.1 => constants.%Result.as_type
// CHECK:STDOUT:   %.Self => constants.%.Self.014
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.loc123_43.2 => constants.%AddWith.lookup_impl_witness.0d5
// CHECK:STDOUT:   %impl.elem0.loc123_43.2 => constants.%impl.elem0.ea5
// CHECK:STDOUT:   %facet_type.loc123_37 => constants.%facet_type.c2e
// CHECK:STDOUT:   %pattern_type.loc123 => constants.%pattern_type.0a9
// CHECK:STDOUT:   %T.patt.loc123_4.2 => constants.%T.patt.69e
// CHECK:STDOUT:   %T.loc123_4.1 => constants.%T.ceb
// CHECK:STDOUT:   %pattern_type.loc124_16 => constants.%pattern_type.f4e485.2
// CHECK:STDOUT:   %result.param_patt.loc124_16.2 => constants.%result.param_patt.6ba
// CHECK:STDOUT:   %result.patt.loc124_16.2 => constants.%result.patt.748373.1
// CHECK:STDOUT:   %T.as_type.loc124_29.1 => constants.%T.as_type.6fc
// CHECK:STDOUT:   %pattern_type.loc124_27 => constants.%pattern_type.786
// CHECK:STDOUT:   %x.param_patt.loc124_27.2 => constants.%x.param_patt.0c5
// CHECK:STDOUT:   %x.patt.loc124_27.2 => constants.%x.patt.4db
// CHECK:STDOUT:   %pattern_type.loc124_33 => constants.%pattern_type.753
// CHECK:STDOUT:   %y.param_patt.loc124_33.2 => constants.%y.param_patt.740
// CHECK:STDOUT:   %y.patt.loc124_33.2 => constants.%y.patt.4ae
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @SubWith.loc135(constants.%Result, constants.%U.f40, constants.%T.bf6) {
// CHECK:STDOUT:   %Result.patt.loc132_9.2 => constants.%Result.patt
// CHECK:STDOUT:   %Result.loc132_9.1 => constants.%Result
// CHECK:STDOUT:   %U.patt.loc133_4.2 => constants.%U.patt.7a2
// CHECK:STDOUT:   %U.loc133_4.1 => constants.%U.f40
// CHECK:STDOUT:   %U.as_type.loc134_35.1 => constants.%U.as_type.b61
// CHECK:STDOUT:   %SubWith.type.loc134_35.1 => constants.%SubWith.type.b5d
// CHECK:STDOUT:   %facet_type.loc134_19 => constants.%facet_type.d46
// CHECK:STDOUT:   %.Self.frozen.loc134_37.1 => constants.%.Self.frozen.efa
// CHECK:STDOUT:   %require_complete.loc134 => constants.%require_complete.cda
// CHECK:STDOUT:   %.Self.frozen.as_type => constants.%.Self.frozen.as_type.79f
// CHECK:STDOUT:   %SubWith.assoc_type => constants.%SubWith.assoc_type.3a1
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.729
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.loc134_43.1 => constants.%SubWith.lookup_impl_witness.acf
// CHECK:STDOUT:   %impl.elem0.loc134_43.1 => constants.%impl.elem0.050
// CHECK:STDOUT:   %Result.as_type.loc134_53.1 => constants.%Result.as_type
// CHECK:STDOUT:   %.Self => constants.%.Self.d13
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.loc134_43.2 => constants.%SubWith.lookup_impl_witness.5d0
// CHECK:STDOUT:   %impl.elem0.loc134_43.2 => constants.%impl.elem0.e4e
// CHECK:STDOUT:   %facet_type.loc134_37 => constants.%facet_type.eeb
// CHECK:STDOUT:   %pattern_type.loc134 => constants.%pattern_type.502
// CHECK:STDOUT:   %T.patt.loc134_4.2 => constants.%T.patt.cd4
// CHECK:STDOUT:   %T.loc134_4.1 => constants.%T.bf6
// CHECK:STDOUT:   %pattern_type.loc135_16 => constants.%pattern_type.f4e485.2
// CHECK:STDOUT:   %result.param_patt.loc135_16.2 => constants.%result.param_patt.6ba
// CHECK:STDOUT:   %result.patt.loc135_16.2 => constants.%result.patt.748373.1
// CHECK:STDOUT:   %T.as_type.loc135_29.1 => constants.%T.as_type.58f
// CHECK:STDOUT:   %pattern_type.loc135_27 => constants.%pattern_type.a10
// CHECK:STDOUT:   %x.param_patt.loc135_27.2 => constants.%x.param_patt.02e
// CHECK:STDOUT:   %x.patt.loc135_27.2 => constants.%x.patt.eb5
// CHECK:STDOUT:   %pattern_type.loc135_33 => constants.%pattern_type.753
// CHECK:STDOUT:   %y.param_patt.loc135_33.2 => constants.%y.param_patt.740
// CHECK:STDOUT:   %y.patt.loc135_33.2 => constants.%y.patt.4ae
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @MulWith.loc146(constants.%Result, constants.%U.f40, constants.%T.9a2) {
// CHECK:STDOUT:   %Result.patt.loc143_9.2 => constants.%Result.patt
// CHECK:STDOUT:   %Result.loc143_9.1 => constants.%Result
// CHECK:STDOUT:   %U.patt.loc144_4.2 => constants.%U.patt.7a2
// CHECK:STDOUT:   %U.loc144_4.1 => constants.%U.f40
// CHECK:STDOUT:   %U.as_type.loc145_35.1 => constants.%U.as_type.b61
// CHECK:STDOUT:   %MulWith.type.loc145_35.1 => constants.%MulWith.type.f02
// CHECK:STDOUT:   %facet_type.loc145_19 => constants.%facet_type.f87
// CHECK:STDOUT:   %.Self.frozen.loc145_37.1 => constants.%.Self.frozen.463
// CHECK:STDOUT:   %require_complete.loc145 => constants.%require_complete.0a3
// CHECK:STDOUT:   %.Self.frozen.as_type => constants.%.Self.frozen.as_type.e91
// CHECK:STDOUT:   %MulWith.assoc_type => constants.%MulWith.assoc_type.a12
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.116
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.loc145_43.1 => constants.%MulWith.lookup_impl_witness.637
// CHECK:STDOUT:   %impl.elem0.loc145_43.1 => constants.%impl.elem0.132
// CHECK:STDOUT:   %Result.as_type.loc145_53.1 => constants.%Result.as_type
// CHECK:STDOUT:   %.Self => constants.%.Self.588
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.loc145_43.2 => constants.%MulWith.lookup_impl_witness.d4f
// CHECK:STDOUT:   %impl.elem0.loc145_43.2 => constants.%impl.elem0.62f
// CHECK:STDOUT:   %facet_type.loc145_37 => constants.%facet_type.0e5
// CHECK:STDOUT:   %pattern_type.loc145 => constants.%pattern_type.e8a
// CHECK:STDOUT:   %T.patt.loc145_4.2 => constants.%T.patt.dec
// CHECK:STDOUT:   %T.loc145_4.1 => constants.%T.9a2
// CHECK:STDOUT:   %pattern_type.loc146_16 => constants.%pattern_type.f4e485.2
// CHECK:STDOUT:   %result.param_patt.loc146_16.2 => constants.%result.param_patt.6ba
// CHECK:STDOUT:   %result.patt.loc146_16.2 => constants.%result.patt.748373.1
// CHECK:STDOUT:   %T.as_type.loc146_29.1 => constants.%T.as_type.83f
// CHECK:STDOUT:   %pattern_type.loc146_27 => constants.%pattern_type.fff
// CHECK:STDOUT:   %x.param_patt.loc146_27.2 => constants.%x.param_patt.f8f
// CHECK:STDOUT:   %x.patt.loc146_27.2 => constants.%x.patt.909
// CHECK:STDOUT:   %pattern_type.loc146_33 => constants.%pattern_type.753
// CHECK:STDOUT:   %y.param_patt.loc146_33.2 => constants.%y.param_patt.740
// CHECK:STDOUT:   %y.patt.loc146_33.2 => constants.%y.patt.4ae
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @DivWith.loc157(constants.%Result, constants.%U.f40, constants.%T.4bb) {
// CHECK:STDOUT:   %Result.patt.loc154_9.2 => constants.%Result.patt
// CHECK:STDOUT:   %Result.loc154_9.1 => constants.%Result
// CHECK:STDOUT:   %U.patt.loc155_4.2 => constants.%U.patt.7a2
// CHECK:STDOUT:   %U.loc155_4.1 => constants.%U.f40
// CHECK:STDOUT:   %U.as_type.loc156_35.1 => constants.%U.as_type.b61
// CHECK:STDOUT:   %DivWith.type.loc156_35.1 => constants.%DivWith.type.523
// CHECK:STDOUT:   %facet_type.loc156_19 => constants.%facet_type.86cd
// CHECK:STDOUT:   %.Self.frozen.loc156_37.1 => constants.%.Self.frozen.1f9
// CHECK:STDOUT:   %require_complete.loc156 => constants.%require_complete.4d7
// CHECK:STDOUT:   %.Self.frozen.as_type => constants.%.Self.frozen.as_type.8ca
// CHECK:STDOUT:   %DivWith.assoc_type => constants.%DivWith.assoc_type.cf4
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.c57
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.loc156_43.1 => constants.%DivWith.lookup_impl_witness.154
// CHECK:STDOUT:   %impl.elem0.loc156_43.1 => constants.%impl.elem0.252
// CHECK:STDOUT:   %Result.as_type.loc156_53.1 => constants.%Result.as_type
// CHECK:STDOUT:   %.Self => constants.%.Self.862
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.loc156_43.2 => constants.%DivWith.lookup_impl_witness.d02
// CHECK:STDOUT:   %impl.elem0.loc156_43.2 => constants.%impl.elem0.606
// CHECK:STDOUT:   %facet_type.loc156_37 => constants.%facet_type.ed9
// CHECK:STDOUT:   %pattern_type.loc156 => constants.%pattern_type.07c
// CHECK:STDOUT:   %T.patt.loc156_4.2 => constants.%T.patt.869
// CHECK:STDOUT:   %T.loc156_4.1 => constants.%T.4bb
// CHECK:STDOUT:   %pattern_type.loc157_16 => constants.%pattern_type.f4e485.2
// CHECK:STDOUT:   %result.param_patt.loc157_16.2 => constants.%result.param_patt.6ba
// CHECK:STDOUT:   %result.patt.loc157_16.2 => constants.%result.patt.748373.1
// CHECK:STDOUT:   %T.as_type.loc157_29.1 => constants.%T.as_type.a74
// CHECK:STDOUT:   %pattern_type.loc157_27 => constants.%pattern_type.b83
// CHECK:STDOUT:   %x.param_patt.loc157_27.2 => constants.%x.param_patt.21f
// CHECK:STDOUT:   %x.patt.loc157_27.2 => constants.%x.patt.46c
// CHECK:STDOUT:   %pattern_type.loc157_33 => constants.%pattern_type.753
// CHECK:STDOUT:   %y.param_patt.loc157_33.2 => constants.%y.param_patt.740
// CHECK:STDOUT:   %y.patt.loc157_33.2 => constants.%y.patt.4ae
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ModWith.loc168(constants.%Result, constants.%U.f40, constants.%T.999) {
// CHECK:STDOUT:   %Result.patt.loc165_9.2 => constants.%Result.patt
// CHECK:STDOUT:   %Result.loc165_9.1 => constants.%Result
// CHECK:STDOUT:   %U.patt.loc166_4.2 => constants.%U.patt.7a2
// CHECK:STDOUT:   %U.loc166_4.1 => constants.%U.f40
// CHECK:STDOUT:   %U.as_type.loc167_35.1 => constants.%U.as_type.b61
// CHECK:STDOUT:   %ModWith.type.loc167_35.1 => constants.%ModWith.type.937
// CHECK:STDOUT:   %facet_type.loc167_19 => constants.%facet_type.b18
// CHECK:STDOUT:   %.Self.frozen.loc167_37.1 => constants.%.Self.frozen.f70
// CHECK:STDOUT:   %require_complete.loc167 => constants.%require_complete.708
// CHECK:STDOUT:   %.Self.frozen.as_type => constants.%.Self.frozen.as_type.447
// CHECK:STDOUT:   %ModWith.assoc_type => constants.%ModWith.assoc_type.626
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.56e
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.loc167_43.1 => constants.%ModWith.lookup_impl_witness.80e
// CHECK:STDOUT:   %impl.elem0.loc167_43.1 => constants.%impl.elem0.523
// CHECK:STDOUT:   %Result.as_type.loc167_53.1 => constants.%Result.as_type
// CHECK:STDOUT:   %.Self => constants.%.Self.a68
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.loc167_43.2 => constants.%ModWith.lookup_impl_witness.569
// CHECK:STDOUT:   %impl.elem0.loc167_43.2 => constants.%impl.elem0.c51
// CHECK:STDOUT:   %facet_type.loc167_37 => constants.%facet_type.639
// CHECK:STDOUT:   %pattern_type.loc167 => constants.%pattern_type.d85
// CHECK:STDOUT:   %T.patt.loc167_4.2 => constants.%T.patt.49b
// CHECK:STDOUT:   %T.loc167_4.1 => constants.%T.999
// CHECK:STDOUT:   %pattern_type.loc168_16 => constants.%pattern_type.f4e485.2
// CHECK:STDOUT:   %result.param_patt.loc168_16.2 => constants.%result.param_patt.6ba
// CHECK:STDOUT:   %result.patt.loc168_16.2 => constants.%result.patt.748373.1
// CHECK:STDOUT:   %T.as_type.loc168_29.1 => constants.%T.as_type.1c2
// CHECK:STDOUT:   %pattern_type.loc168_27 => constants.%pattern_type.01b
// CHECK:STDOUT:   %x.param_patt.loc168_27.2 => constants.%x.param_patt.f25
// CHECK:STDOUT:   %x.patt.loc168_27.2 => constants.%x.patt.804
// CHECK:STDOUT:   %pattern_type.loc168_33 => constants.%pattern_type.753
// CHECK:STDOUT:   %y.param_patt.loc168_33.2 => constants.%y.param_patt.740
// CHECK:STDOUT:   %y.patt.loc168_33.2 => constants.%y.patt.4ae
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AddWith.loc124(constants.%Destroy.facet.2cc, constants.%Destroy.facet.2cc, constants.%facet_value.696) {
// CHECK:STDOUT:   %Result.patt.loc121_9.2 => constants.%Result.patt
// CHECK:STDOUT:   %Result.loc121_9.1 => constants.%Destroy.facet.2cc
// CHECK:STDOUT:   %U.patt.loc122_4.2 => constants.%U.patt.7a2
// CHECK:STDOUT:   %U.loc122_4.1 => constants.%Destroy.facet.2cc
// CHECK:STDOUT:   %U.as_type.loc123_35.1 => constants.%Int16
// CHECK:STDOUT:   %AddWith.type.loc123_35.1 => constants.%AddWith.type.b62
// CHECK:STDOUT:   %facet_type.loc123_19 => constants.%facet_type.297
// CHECK:STDOUT:   %.Self.frozen.loc123_37.1 => constants.%.Self.frozen.7e3
// CHECK:STDOUT:   %require_complete.loc123 => constants.%complete_type.a82
// CHECK:STDOUT:   %.Self.frozen.as_type => constants.%.Self.frozen.as_type.85e
// CHECK:STDOUT:   %AddWith.assoc_type => constants.%AddWith.assoc_type.c72
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.261
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.loc123_43.1 => constants.%AddWith.lookup_impl_witness.5b7
// CHECK:STDOUT:   %impl.elem0.loc123_43.1 => constants.%impl.elem0.d80
// CHECK:STDOUT:   %Result.as_type.loc123_53.1 => constants.%Int16
// CHECK:STDOUT:   %.Self => constants.%.Self.45a
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.loc123_43.2 => constants.%AddWith.lookup_impl_witness.131
// CHECK:STDOUT:   %impl.elem0.loc123_43.2 => constants.%impl.elem0.28c
// CHECK:STDOUT:   %facet_type.loc123_37 => constants.%facet_type.225
// CHECK:STDOUT:   %pattern_type.loc123 => constants.%pattern_type.34a
// CHECK:STDOUT:   %T.patt.loc123_4.2 => constants.%T.patt.3ec
// CHECK:STDOUT:   %T.loc123_4.1 => constants.%facet_value.696
// CHECK:STDOUT:   %pattern_type.loc124_16 => constants.%pattern_type.785
// CHECK:STDOUT:   %result.param_patt.loc124_16.2 => constants.%result.param_patt.be6
// CHECK:STDOUT:   %result.patt.loc124_16.2 => constants.%result.patt.488
// CHECK:STDOUT:   %T.as_type.loc124_29.1 => constants.%i16
// CHECK:STDOUT:   %pattern_type.loc124_27 => constants.%pattern_type.203
// CHECK:STDOUT:   %x.param_patt.loc124_27.2 => constants.%x.param_patt.676
// CHECK:STDOUT:   %x.patt.loc124_27.2 => constants.%x.patt.26e
// CHECK:STDOUT:   %pattern_type.loc124_33 => constants.%pattern_type.785
// CHECK:STDOUT:   %y.param_patt.loc124_33.2 => constants.%y.param_patt.418
// CHECK:STDOUT:   %y.patt.loc124_33.2 => constants.%y.patt.4e6
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc124_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc124_27 => constants.%complete_type.2da
// CHECK:STDOUT:   %require_complete.loc124_33 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc126 => constants.%complete_type.7a6
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.006
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.loc126 => constants.%custom_witness.248
// CHECK:STDOUT:   %AddWith.facet => constants.%AddWith.facet.ff0
// CHECK:STDOUT:   %AddWith.WithSelf.Op.type => constants.%AddWith.WithSelf.Op.type.e1f
// CHECK:STDOUT:   %.loc126_5.6 => constants.%.2c3
// CHECK:STDOUT:   %impl.elem1.loc126_5.2 => constants.%Op.72717c.1
// CHECK:STDOUT:   %specific_impl_fn.loc126_5.3 => constants.%Op.72717c.1
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.9d5
// CHECK:STDOUT:   %.loc126_5.7 => constants.%.6da
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.e7a31c.1
// CHECK:STDOUT:   %impl.elem0.loc126_5.2 => constants.%Int16.Op.4dca31.1
// CHECK:STDOUT:   %specific_impl_fn.loc126_5.4 => constants.%Int16.Op.4dca31.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @SubWith.loc135(constants.%Destroy.facet.2cc, constants.%Destroy.facet.2cc, constants.%facet_value.031) {
// CHECK:STDOUT:   %Result.patt.loc132_9.2 => constants.%Result.patt
// CHECK:STDOUT:   %Result.loc132_9.1 => constants.%Destroy.facet.2cc
// CHECK:STDOUT:   %U.patt.loc133_4.2 => constants.%U.patt.7a2
// CHECK:STDOUT:   %U.loc133_4.1 => constants.%Destroy.facet.2cc
// CHECK:STDOUT:   %U.as_type.loc134_35.1 => constants.%Int16
// CHECK:STDOUT:   %SubWith.type.loc134_35.1 => constants.%SubWith.type.b12
// CHECK:STDOUT:   %facet_type.loc134_19 => constants.%facet_type.b15
// CHECK:STDOUT:   %.Self.frozen.loc134_37.1 => constants.%.Self.frozen.581
// CHECK:STDOUT:   %require_complete.loc134 => constants.%complete_type.647
// CHECK:STDOUT:   %.Self.frozen.as_type => constants.%.Self.frozen.as_type.092
// CHECK:STDOUT:   %SubWith.assoc_type => constants.%SubWith.assoc_type.eb8
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.a90
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.loc134_43.1 => constants.%SubWith.lookup_impl_witness.b30
// CHECK:STDOUT:   %impl.elem0.loc134_43.1 => constants.%impl.elem0.748
// CHECK:STDOUT:   %Result.as_type.loc134_53.1 => constants.%Int16
// CHECK:STDOUT:   %.Self => constants.%.Self.415
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.loc134_43.2 => constants.%SubWith.lookup_impl_witness.266
// CHECK:STDOUT:   %impl.elem0.loc134_43.2 => constants.%impl.elem0.4a0
// CHECK:STDOUT:   %facet_type.loc134_37 => constants.%facet_type.b71
// CHECK:STDOUT:   %pattern_type.loc134 => constants.%pattern_type.f54
// CHECK:STDOUT:   %T.patt.loc134_4.2 => constants.%T.patt.842
// CHECK:STDOUT:   %T.loc134_4.1 => constants.%facet_value.031
// CHECK:STDOUT:   %pattern_type.loc135_16 => constants.%pattern_type.785
// CHECK:STDOUT:   %result.param_patt.loc135_16.2 => constants.%result.param_patt.be6
// CHECK:STDOUT:   %result.patt.loc135_16.2 => constants.%result.patt.488
// CHECK:STDOUT:   %T.as_type.loc135_29.1 => constants.%i16
// CHECK:STDOUT:   %pattern_type.loc135_27 => constants.%pattern_type.203
// CHECK:STDOUT:   %x.param_patt.loc135_27.2 => constants.%x.param_patt.676
// CHECK:STDOUT:   %x.patt.loc135_27.2 => constants.%x.patt.26e
// CHECK:STDOUT:   %pattern_type.loc135_33 => constants.%pattern_type.785
// CHECK:STDOUT:   %y.param_patt.loc135_33.2 => constants.%y.param_patt.418
// CHECK:STDOUT:   %y.patt.loc135_33.2 => constants.%y.patt.4e6
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc135_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc135_27 => constants.%complete_type.2da
// CHECK:STDOUT:   %require_complete.loc135_33 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc137 => constants.%complete_type.b9a
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.2cf
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.loc137 => constants.%custom_witness.f14
// CHECK:STDOUT:   %SubWith.facet => constants.%SubWith.facet.49f
// CHECK:STDOUT:   %SubWith.WithSelf.Op.type => constants.%SubWith.WithSelf.Op.type.843
// CHECK:STDOUT:   %.loc137_5.6 => constants.%.00a
// CHECK:STDOUT:   %impl.elem1.loc137_5.2 => constants.%Op.72717c.2
// CHECK:STDOUT:   %specific_impl_fn.loc137_5.3 => constants.%Op.72717c.2
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.9d5
// CHECK:STDOUT:   %.loc137_5.7 => constants.%.6da
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.e7a31c.1
// CHECK:STDOUT:   %impl.elem0.loc137_5.2 => constants.%Int16.Op.4dca31.1
// CHECK:STDOUT:   %specific_impl_fn.loc137_5.4 => constants.%Int16.Op.4dca31.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @MulWith.loc146(constants.%Destroy.facet.2cc, constants.%Destroy.facet.2cc, constants.%facet_value.920) {
// CHECK:STDOUT:   %Result.patt.loc143_9.2 => constants.%Result.patt
// CHECK:STDOUT:   %Result.loc143_9.1 => constants.%Destroy.facet.2cc
// CHECK:STDOUT:   %U.patt.loc144_4.2 => constants.%U.patt.7a2
// CHECK:STDOUT:   %U.loc144_4.1 => constants.%Destroy.facet.2cc
// CHECK:STDOUT:   %U.as_type.loc145_35.1 => constants.%Int16
// CHECK:STDOUT:   %MulWith.type.loc145_35.1 => constants.%MulWith.type.c18
// CHECK:STDOUT:   %facet_type.loc145_19 => constants.%facet_type.a64
// CHECK:STDOUT:   %.Self.frozen.loc145_37.1 => constants.%.Self.frozen.05e
// CHECK:STDOUT:   %require_complete.loc145 => constants.%complete_type.4d3
// CHECK:STDOUT:   %.Self.frozen.as_type => constants.%.Self.frozen.as_type.2d1
// CHECK:STDOUT:   %MulWith.assoc_type => constants.%MulWith.assoc_type.f15
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.f8b
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.loc145_43.1 => constants.%MulWith.lookup_impl_witness.701
// CHECK:STDOUT:   %impl.elem0.loc145_43.1 => constants.%impl.elem0.381
// CHECK:STDOUT:   %Result.as_type.loc145_53.1 => constants.%Int16
// CHECK:STDOUT:   %.Self => constants.%.Self.00f
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.loc145_43.2 => constants.%MulWith.lookup_impl_witness.782
// CHECK:STDOUT:   %impl.elem0.loc145_43.2 => constants.%impl.elem0.8de
// CHECK:STDOUT:   %facet_type.loc145_37 => constants.%facet_type.662
// CHECK:STDOUT:   %pattern_type.loc145 => constants.%pattern_type.90f
// CHECK:STDOUT:   %T.patt.loc145_4.2 => constants.%T.patt.8ad
// CHECK:STDOUT:   %T.loc145_4.1 => constants.%facet_value.920
// CHECK:STDOUT:   %pattern_type.loc146_16 => constants.%pattern_type.785
// CHECK:STDOUT:   %result.param_patt.loc146_16.2 => constants.%result.param_patt.be6
// CHECK:STDOUT:   %result.patt.loc146_16.2 => constants.%result.patt.488
// CHECK:STDOUT:   %T.as_type.loc146_29.1 => constants.%i16
// CHECK:STDOUT:   %pattern_type.loc146_27 => constants.%pattern_type.203
// CHECK:STDOUT:   %x.param_patt.loc146_27.2 => constants.%x.param_patt.676
// CHECK:STDOUT:   %x.patt.loc146_27.2 => constants.%x.patt.26e
// CHECK:STDOUT:   %pattern_type.loc146_33 => constants.%pattern_type.785
// CHECK:STDOUT:   %y.param_patt.loc146_33.2 => constants.%y.param_patt.418
// CHECK:STDOUT:   %y.patt.loc146_33.2 => constants.%y.patt.4e6
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc146_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc146_27 => constants.%complete_type.2da
// CHECK:STDOUT:   %require_complete.loc146_33 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc148 => constants.%complete_type.433
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.eb5
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.loc148 => constants.%custom_witness.6bd
// CHECK:STDOUT:   %MulWith.facet => constants.%MulWith.facet.d6e
// CHECK:STDOUT:   %MulWith.WithSelf.Op.type => constants.%MulWith.WithSelf.Op.type.f9f
// CHECK:STDOUT:   %.loc148_5.6 => constants.%.0ed
// CHECK:STDOUT:   %impl.elem1.loc148_5.2 => constants.%Op.72717c.3
// CHECK:STDOUT:   %specific_impl_fn.loc148_5.3 => constants.%Op.72717c.3
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.9d5
// CHECK:STDOUT:   %.loc148_5.7 => constants.%.6da
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.e7a31c.1
// CHECK:STDOUT:   %impl.elem0.loc148_5.2 => constants.%Int16.Op.4dca31.1
// CHECK:STDOUT:   %specific_impl_fn.loc148_5.4 => constants.%Int16.Op.4dca31.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @DivWith.loc157(constants.%Destroy.facet.2cc, constants.%Destroy.facet.2cc, constants.%facet_value.ce6) {
// CHECK:STDOUT:   %Result.patt.loc154_9.2 => constants.%Result.patt
// CHECK:STDOUT:   %Result.loc154_9.1 => constants.%Destroy.facet.2cc
// CHECK:STDOUT:   %U.patt.loc155_4.2 => constants.%U.patt.7a2
// CHECK:STDOUT:   %U.loc155_4.1 => constants.%Destroy.facet.2cc
// CHECK:STDOUT:   %U.as_type.loc156_35.1 => constants.%Int16
// CHECK:STDOUT:   %DivWith.type.loc156_35.1 => constants.%DivWith.type.6f9
// CHECK:STDOUT:   %facet_type.loc156_19 => constants.%facet_type.0b4
// CHECK:STDOUT:   %.Self.frozen.loc156_37.1 => constants.%.Self.frozen.ecc
// CHECK:STDOUT:   %require_complete.loc156 => constants.%complete_type.acc
// CHECK:STDOUT:   %.Self.frozen.as_type => constants.%.Self.frozen.as_type.851
// CHECK:STDOUT:   %DivWith.assoc_type => constants.%DivWith.assoc_type.333
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.962
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.loc156_43.1 => constants.%DivWith.lookup_impl_witness.6fa
// CHECK:STDOUT:   %impl.elem0.loc156_43.1 => constants.%impl.elem0.e0f
// CHECK:STDOUT:   %Result.as_type.loc156_53.1 => constants.%Int16
// CHECK:STDOUT:   %.Self => constants.%.Self.9b5
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.loc156_43.2 => constants.%DivWith.lookup_impl_witness.648
// CHECK:STDOUT:   %impl.elem0.loc156_43.2 => constants.%impl.elem0.36a
// CHECK:STDOUT:   %facet_type.loc156_37 => constants.%facet_type.bd6
// CHECK:STDOUT:   %pattern_type.loc156 => constants.%pattern_type.692
// CHECK:STDOUT:   %T.patt.loc156_4.2 => constants.%T.patt.00c
// CHECK:STDOUT:   %T.loc156_4.1 => constants.%facet_value.ce6
// CHECK:STDOUT:   %pattern_type.loc157_16 => constants.%pattern_type.785
// CHECK:STDOUT:   %result.param_patt.loc157_16.2 => constants.%result.param_patt.be6
// CHECK:STDOUT:   %result.patt.loc157_16.2 => constants.%result.patt.488
// CHECK:STDOUT:   %T.as_type.loc157_29.1 => constants.%i16
// CHECK:STDOUT:   %pattern_type.loc157_27 => constants.%pattern_type.203
// CHECK:STDOUT:   %x.param_patt.loc157_27.2 => constants.%x.param_patt.676
// CHECK:STDOUT:   %x.patt.loc157_27.2 => constants.%x.patt.26e
// CHECK:STDOUT:   %pattern_type.loc157_33 => constants.%pattern_type.785
// CHECK:STDOUT:   %y.param_patt.loc157_33.2 => constants.%y.param_patt.418
// CHECK:STDOUT:   %y.patt.loc157_33.2 => constants.%y.patt.4e6
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc157_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc157_27 => constants.%complete_type.2da
// CHECK:STDOUT:   %require_complete.loc157_33 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc159 => constants.%complete_type.64f
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.8b5
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.loc159 => constants.%custom_witness.22c
// CHECK:STDOUT:   %DivWith.facet => constants.%DivWith.facet.437
// CHECK:STDOUT:   %DivWith.WithSelf.Op.type => constants.%DivWith.WithSelf.Op.type.51a
// CHECK:STDOUT:   %.loc159_5.6 => constants.%.316
// CHECK:STDOUT:   %impl.elem1.loc159_5.2 => constants.%Op.72717c.4
// CHECK:STDOUT:   %specific_impl_fn.loc159_5.3 => constants.%Op.72717c.4
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.9d5
// CHECK:STDOUT:   %.loc159_5.7 => constants.%.6da
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.e7a31c.1
// CHECK:STDOUT:   %impl.elem0.loc159_5.2 => constants.%Int16.Op.4dca31.1
// CHECK:STDOUT:   %specific_impl_fn.loc159_5.4 => constants.%Int16.Op.4dca31.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ModWith.loc168(constants.%Destroy.facet.2cc, constants.%Destroy.facet.2cc, constants.%facet_value.cde) {
// CHECK:STDOUT:   %Result.patt.loc165_9.2 => constants.%Result.patt
// CHECK:STDOUT:   %Result.loc165_9.1 => constants.%Destroy.facet.2cc
// CHECK:STDOUT:   %U.patt.loc166_4.2 => constants.%U.patt.7a2
// CHECK:STDOUT:   %U.loc166_4.1 => constants.%Destroy.facet.2cc
// CHECK:STDOUT:   %U.as_type.loc167_35.1 => constants.%Int16
// CHECK:STDOUT:   %ModWith.type.loc167_35.1 => constants.%ModWith.type.763
// CHECK:STDOUT:   %facet_type.loc167_19 => constants.%facet_type.ee8
// CHECK:STDOUT:   %.Self.frozen.loc167_37.1 => constants.%.Self.frozen.9a7
// CHECK:STDOUT:   %require_complete.loc167 => constants.%complete_type.8fc
// CHECK:STDOUT:   %.Self.frozen.as_type => constants.%.Self.frozen.as_type.954
// CHECK:STDOUT:   %ModWith.assoc_type => constants.%ModWith.assoc_type.e6a
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.2cc
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.loc167_43.1 => constants.%ModWith.lookup_impl_witness.689
// CHECK:STDOUT:   %impl.elem0.loc167_43.1 => constants.%impl.elem0.42c
// CHECK:STDOUT:   %Result.as_type.loc167_53.1 => constants.%Int16
// CHECK:STDOUT:   %.Self => constants.%.Self.67f
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.loc167_43.2 => constants.%ModWith.lookup_impl_witness.af3
// CHECK:STDOUT:   %impl.elem0.loc167_43.2 => constants.%impl.elem0.436
// CHECK:STDOUT:   %facet_type.loc167_37 => constants.%facet_type.a2b
// CHECK:STDOUT:   %pattern_type.loc167 => constants.%pattern_type.61d
// CHECK:STDOUT:   %T.patt.loc167_4.2 => constants.%T.patt.a00
// CHECK:STDOUT:   %T.loc167_4.1 => constants.%facet_value.cde
// CHECK:STDOUT:   %pattern_type.loc168_16 => constants.%pattern_type.785
// CHECK:STDOUT:   %result.param_patt.loc168_16.2 => constants.%result.param_patt.be6
// CHECK:STDOUT:   %result.patt.loc168_16.2 => constants.%result.patt.488
// CHECK:STDOUT:   %T.as_type.loc168_29.1 => constants.%i16
// CHECK:STDOUT:   %pattern_type.loc168_27 => constants.%pattern_type.203
// CHECK:STDOUT:   %x.param_patt.loc168_27.2 => constants.%x.param_patt.676
// CHECK:STDOUT:   %x.patt.loc168_27.2 => constants.%x.patt.26e
// CHECK:STDOUT:   %pattern_type.loc168_33 => constants.%pattern_type.785
// CHECK:STDOUT:   %y.param_patt.loc168_33.2 => constants.%y.param_patt.418
// CHECK:STDOUT:   %y.patt.loc168_33.2 => constants.%y.patt.4e6
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc168_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc168_27 => constants.%complete_type.2da
// CHECK:STDOUT:   %require_complete.loc168_33 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc170 => constants.%complete_type.dea
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.3dd
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.loc170 => constants.%custom_witness.9e6
// CHECK:STDOUT:   %ModWith.facet => constants.%ModWith.facet.251
// CHECK:STDOUT:   %ModWith.WithSelf.Op.type => constants.%ModWith.WithSelf.Op.type.62d
// CHECK:STDOUT:   %.loc170_5.6 => constants.%.9b3
// CHECK:STDOUT:   %impl.elem1.loc170_5.2 => constants.%Op.72717c.5
// CHECK:STDOUT:   %specific_impl_fn.loc170_5.3 => constants.%Op.72717c.5
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.9d5
// CHECK:STDOUT:   %.loc170_5.7 => constants.%.6da
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.e7a31c.1
// CHECK:STDOUT:   %impl.elem0.loc170_5.2 => constants.%Int16.Op.4dca31.1
// CHECK:STDOUT:   %specific_impl_fn.loc170_5.4 => constants.%Int16.Op.4dca31.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AddWith.loc124(constants.%Destroy.facet.2cc, constants.%Destroy.facet.e5c, constants.%facet_value.0c7) {
// CHECK:STDOUT:   %Result.patt.loc121_9.2 => constants.%Result.patt
// CHECK:STDOUT:   %Result.loc121_9.1 => constants.%Destroy.facet.2cc
// CHECK:STDOUT:   %U.patt.loc122_4.2 => constants.%U.patt.7a2
// CHECK:STDOUT:   %U.loc122_4.1 => constants.%Destroy.facet.e5c
// CHECK:STDOUT:   %U.as_type.loc123_35.1 => constants.%i16
// CHECK:STDOUT:   %AddWith.type.loc123_35.1 => constants.%AddWith.type.a36
// CHECK:STDOUT:   %facet_type.loc123_19 => constants.%facet_type.1cc
// CHECK:STDOUT:   %.Self.frozen.loc123_37.1 => constants.%.Self.frozen.e5b
// CHECK:STDOUT:   %require_complete.loc123 => constants.%complete_type.5a6
// CHECK:STDOUT:   %.Self.frozen.as_type => constants.%.Self.frozen.as_type.b7c
// CHECK:STDOUT:   %AddWith.assoc_type => constants.%AddWith.assoc_type.6bf
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.e6b
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.loc123_43.1 => constants.%AddWith.lookup_impl_witness.047
// CHECK:STDOUT:   %impl.elem0.loc123_43.1 => constants.%impl.elem0.f6a
// CHECK:STDOUT:   %Result.as_type.loc123_53.1 => constants.%Int16
// CHECK:STDOUT:   %.Self => constants.%.Self.a82
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.loc123_43.2 => constants.%AddWith.lookup_impl_witness.e27
// CHECK:STDOUT:   %impl.elem0.loc123_43.2 => constants.%impl.elem0.115
// CHECK:STDOUT:   %facet_type.loc123_37 => constants.%facet_type.b7b
// CHECK:STDOUT:   %pattern_type.loc123 => constants.%pattern_type.c51
// CHECK:STDOUT:   %T.patt.loc123_4.2 => constants.%T.patt.5cb
// CHECK:STDOUT:   %T.loc123_4.1 => constants.%facet_value.0c7
// CHECK:STDOUT:   %pattern_type.loc124_16 => constants.%pattern_type.785
// CHECK:STDOUT:   %result.param_patt.loc124_16.2 => constants.%result.param_patt.be6
// CHECK:STDOUT:   %result.patt.loc124_16.2 => constants.%result.patt.488
// CHECK:STDOUT:   %T.as_type.loc124_29.1 => constants.%Int16
// CHECK:STDOUT:   %pattern_type.loc124_27 => constants.%pattern_type.785
// CHECK:STDOUT:   %x.param_patt.loc124_27.2 => constants.%x.param_patt.3b0
// CHECK:STDOUT:   %x.patt.loc124_27.2 => constants.%x.patt.b76
// CHECK:STDOUT:   %pattern_type.loc124_33 => constants.%pattern_type.203
// CHECK:STDOUT:   %y.param_patt.loc124_33.2 => constants.%y.param_patt.4b4
// CHECK:STDOUT:   %y.patt.loc124_33.2 => constants.%y.patt.dd3
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc124_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc124_27 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc124_33 => constants.%complete_type.2da
// CHECK:STDOUT:   %require_complete.loc126 => constants.%complete_type.a53
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.80e
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.loc126 => constants.%custom_witness.b17
// CHECK:STDOUT:   %AddWith.facet => constants.%AddWith.facet.b59
// CHECK:STDOUT:   %AddWith.WithSelf.Op.type => constants.%AddWith.WithSelf.Op.type.de7
// CHECK:STDOUT:   %.loc126_5.6 => constants.%.c9e
// CHECK:STDOUT:   %impl.elem1.loc126_5.2 => constants.%Int16.Op.4dca31.3
// CHECK:STDOUT:   %specific_impl_fn.loc126_5.3 => constants.%Int16.Op.4dca31.3
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.9d5
// CHECK:STDOUT:   %.loc126_5.7 => constants.%.6da
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.e7a31c.1
// CHECK:STDOUT:   %impl.elem0.loc126_5.2 => constants.%Int16.Op.4dca31.1
// CHECK:STDOUT:   %specific_impl_fn.loc126_5.4 => constants.%Int16.Op.4dca31.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @SubWith.loc135(constants.%Destroy.facet.2cc, constants.%Destroy.facet.e5c, constants.%facet_value.33e) {
// CHECK:STDOUT:   %Result.patt.loc132_9.2 => constants.%Result.patt
// CHECK:STDOUT:   %Result.loc132_9.1 => constants.%Destroy.facet.2cc
// CHECK:STDOUT:   %U.patt.loc133_4.2 => constants.%U.patt.7a2
// CHECK:STDOUT:   %U.loc133_4.1 => constants.%Destroy.facet.e5c
// CHECK:STDOUT:   %U.as_type.loc134_35.1 => constants.%i16
// CHECK:STDOUT:   %SubWith.type.loc134_35.1 => constants.%SubWith.type.08b
// CHECK:STDOUT:   %facet_type.loc134_19 => constants.%facet_type.cb2
// CHECK:STDOUT:   %.Self.frozen.loc134_37.1 => constants.%.Self.frozen.bbc
// CHECK:STDOUT:   %require_complete.loc134 => constants.%complete_type.090
// CHECK:STDOUT:   %.Self.frozen.as_type => constants.%.Self.frozen.as_type.ae6
// CHECK:STDOUT:   %SubWith.assoc_type => constants.%SubWith.assoc_type.077
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.465
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.loc134_43.1 => constants.%SubWith.lookup_impl_witness.596
// CHECK:STDOUT:   %impl.elem0.loc134_43.1 => constants.%impl.elem0.733
// CHECK:STDOUT:   %Result.as_type.loc134_53.1 => constants.%Int16
// CHECK:STDOUT:   %.Self => constants.%.Self.a84
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.loc134_43.2 => constants.%SubWith.lookup_impl_witness.581
// CHECK:STDOUT:   %impl.elem0.loc134_43.2 => constants.%impl.elem0.a77
// CHECK:STDOUT:   %facet_type.loc134_37 => constants.%facet_type.a67
// CHECK:STDOUT:   %pattern_type.loc134 => constants.%pattern_type.486
// CHECK:STDOUT:   %T.patt.loc134_4.2 => constants.%T.patt.eb2
// CHECK:STDOUT:   %T.loc134_4.1 => constants.%facet_value.33e
// CHECK:STDOUT:   %pattern_type.loc135_16 => constants.%pattern_type.785
// CHECK:STDOUT:   %result.param_patt.loc135_16.2 => constants.%result.param_patt.be6
// CHECK:STDOUT:   %result.patt.loc135_16.2 => constants.%result.patt.488
// CHECK:STDOUT:   %T.as_type.loc135_29.1 => constants.%Int16
// CHECK:STDOUT:   %pattern_type.loc135_27 => constants.%pattern_type.785
// CHECK:STDOUT:   %x.param_patt.loc135_27.2 => constants.%x.param_patt.3b0
// CHECK:STDOUT:   %x.patt.loc135_27.2 => constants.%x.patt.b76
// CHECK:STDOUT:   %pattern_type.loc135_33 => constants.%pattern_type.203
// CHECK:STDOUT:   %y.param_patt.loc135_33.2 => constants.%y.param_patt.4b4
// CHECK:STDOUT:   %y.patt.loc135_33.2 => constants.%y.patt.dd3
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc135_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc135_27 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc135_33 => constants.%complete_type.2da
// CHECK:STDOUT:   %require_complete.loc137 => constants.%complete_type.fc0
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.9f7
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.loc137 => constants.%custom_witness.b07
// CHECK:STDOUT:   %SubWith.facet => constants.%SubWith.facet.ef4
// CHECK:STDOUT:   %SubWith.WithSelf.Op.type => constants.%SubWith.WithSelf.Op.type.723
// CHECK:STDOUT:   %.loc137_5.6 => constants.%.6f3
// CHECK:STDOUT:   %impl.elem1.loc137_5.2 => constants.%Int16.Op.4dca31.4
// CHECK:STDOUT:   %specific_impl_fn.loc137_5.3 => constants.%Int16.Op.4dca31.4
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.9d5
// CHECK:STDOUT:   %.loc137_5.7 => constants.%.6da
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.e7a31c.1
// CHECK:STDOUT:   %impl.elem0.loc137_5.2 => constants.%Int16.Op.4dca31.1
// CHECK:STDOUT:   %specific_impl_fn.loc137_5.4 => constants.%Int16.Op.4dca31.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @MulWith.loc146(constants.%Destroy.facet.2cc, constants.%Destroy.facet.e5c, constants.%facet_value.ba4) {
// CHECK:STDOUT:   %Result.patt.loc143_9.2 => constants.%Result.patt
// CHECK:STDOUT:   %Result.loc143_9.1 => constants.%Destroy.facet.2cc
// CHECK:STDOUT:   %U.patt.loc144_4.2 => constants.%U.patt.7a2
// CHECK:STDOUT:   %U.loc144_4.1 => constants.%Destroy.facet.e5c
// CHECK:STDOUT:   %U.as_type.loc145_35.1 => constants.%i16
// CHECK:STDOUT:   %MulWith.type.loc145_35.1 => constants.%MulWith.type.55d
// CHECK:STDOUT:   %facet_type.loc145_19 => constants.%facet_type.67c
// CHECK:STDOUT:   %.Self.frozen.loc145_37.1 => constants.%.Self.frozen.c15
// CHECK:STDOUT:   %require_complete.loc145 => constants.%complete_type.a8b
// CHECK:STDOUT:   %.Self.frozen.as_type => constants.%.Self.frozen.as_type.225
// CHECK:STDOUT:   %MulWith.assoc_type => constants.%MulWith.assoc_type.212
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.455
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.loc145_43.1 => constants.%MulWith.lookup_impl_witness.481
// CHECK:STDOUT:   %impl.elem0.loc145_43.1 => constants.%impl.elem0.574
// CHECK:STDOUT:   %Result.as_type.loc145_53.1 => constants.%Int16
// CHECK:STDOUT:   %.Self => constants.%.Self.5dd
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.loc145_43.2 => constants.%MulWith.lookup_impl_witness.092
// CHECK:STDOUT:   %impl.elem0.loc145_43.2 => constants.%impl.elem0.163
// CHECK:STDOUT:   %facet_type.loc145_37 => constants.%facet_type.d40
// CHECK:STDOUT:   %pattern_type.loc145 => constants.%pattern_type.f6f
// CHECK:STDOUT:   %T.patt.loc145_4.2 => constants.%T.patt.dc7
// CHECK:STDOUT:   %T.loc145_4.1 => constants.%facet_value.ba4
// CHECK:STDOUT:   %pattern_type.loc146_16 => constants.%pattern_type.785
// CHECK:STDOUT:   %result.param_patt.loc146_16.2 => constants.%result.param_patt.be6
// CHECK:STDOUT:   %result.patt.loc146_16.2 => constants.%result.patt.488
// CHECK:STDOUT:   %T.as_type.loc146_29.1 => constants.%Int16
// CHECK:STDOUT:   %pattern_type.loc146_27 => constants.%pattern_type.785
// CHECK:STDOUT:   %x.param_patt.loc146_27.2 => constants.%x.param_patt.3b0
// CHECK:STDOUT:   %x.patt.loc146_27.2 => constants.%x.patt.b76
// CHECK:STDOUT:   %pattern_type.loc146_33 => constants.%pattern_type.203
// CHECK:STDOUT:   %y.param_patt.loc146_33.2 => constants.%y.param_patt.4b4
// CHECK:STDOUT:   %y.patt.loc146_33.2 => constants.%y.patt.dd3
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc146_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc146_27 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc146_33 => constants.%complete_type.2da
// CHECK:STDOUT:   %require_complete.loc148 => constants.%complete_type.850
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.2d0
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.loc148 => constants.%custom_witness.8bd
// CHECK:STDOUT:   %MulWith.facet => constants.%MulWith.facet.19a
// CHECK:STDOUT:   %MulWith.WithSelf.Op.type => constants.%MulWith.WithSelf.Op.type.7b1
// CHECK:STDOUT:   %.loc148_5.6 => constants.%.d82
// CHECK:STDOUT:   %impl.elem1.loc148_5.2 => constants.%Int16.Op.4dca31.5
// CHECK:STDOUT:   %specific_impl_fn.loc148_5.3 => constants.%Int16.Op.4dca31.5
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.9d5
// CHECK:STDOUT:   %.loc148_5.7 => constants.%.6da
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.e7a31c.1
// CHECK:STDOUT:   %impl.elem0.loc148_5.2 => constants.%Int16.Op.4dca31.1
// CHECK:STDOUT:   %specific_impl_fn.loc148_5.4 => constants.%Int16.Op.4dca31.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @DivWith.loc157(constants.%Destroy.facet.2cc, constants.%Destroy.facet.e5c, constants.%facet_value.521) {
// CHECK:STDOUT:   %Result.patt.loc154_9.2 => constants.%Result.patt
// CHECK:STDOUT:   %Result.loc154_9.1 => constants.%Destroy.facet.2cc
// CHECK:STDOUT:   %U.patt.loc155_4.2 => constants.%U.patt.7a2
// CHECK:STDOUT:   %U.loc155_4.1 => constants.%Destroy.facet.e5c
// CHECK:STDOUT:   %U.as_type.loc156_35.1 => constants.%i16
// CHECK:STDOUT:   %DivWith.type.loc156_35.1 => constants.%DivWith.type.ab0
// CHECK:STDOUT:   %facet_type.loc156_19 => constants.%facet_type.86c8
// CHECK:STDOUT:   %.Self.frozen.loc156_37.1 => constants.%.Self.frozen.128
// CHECK:STDOUT:   %require_complete.loc156 => constants.%complete_type.1cb
// CHECK:STDOUT:   %.Self.frozen.as_type => constants.%.Self.frozen.as_type.ac3
// CHECK:STDOUT:   %DivWith.assoc_type => constants.%DivWith.assoc_type.1f2
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.be2
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.loc156_43.1 => constants.%DivWith.lookup_impl_witness.a5b
// CHECK:STDOUT:   %impl.elem0.loc156_43.1 => constants.%impl.elem0.4e9
// CHECK:STDOUT:   %Result.as_type.loc156_53.1 => constants.%Int16
// CHECK:STDOUT:   %.Self => constants.%.Self.d04
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.loc156_43.2 => constants.%DivWith.lookup_impl_witness.6ec
// CHECK:STDOUT:   %impl.elem0.loc156_43.2 => constants.%impl.elem0.58b
// CHECK:STDOUT:   %facet_type.loc156_37 => constants.%facet_type.541
// CHECK:STDOUT:   %pattern_type.loc156 => constants.%pattern_type.80a
// CHECK:STDOUT:   %T.patt.loc156_4.2 => constants.%T.patt.c06
// CHECK:STDOUT:   %T.loc156_4.1 => constants.%facet_value.521
// CHECK:STDOUT:   %pattern_type.loc157_16 => constants.%pattern_type.785
// CHECK:STDOUT:   %result.param_patt.loc157_16.2 => constants.%result.param_patt.be6
// CHECK:STDOUT:   %result.patt.loc157_16.2 => constants.%result.patt.488
// CHECK:STDOUT:   %T.as_type.loc157_29.1 => constants.%Int16
// CHECK:STDOUT:   %pattern_type.loc157_27 => constants.%pattern_type.785
// CHECK:STDOUT:   %x.param_patt.loc157_27.2 => constants.%x.param_patt.3b0
// CHECK:STDOUT:   %x.patt.loc157_27.2 => constants.%x.patt.b76
// CHECK:STDOUT:   %pattern_type.loc157_33 => constants.%pattern_type.203
// CHECK:STDOUT:   %y.param_patt.loc157_33.2 => constants.%y.param_patt.4b4
// CHECK:STDOUT:   %y.patt.loc157_33.2 => constants.%y.patt.dd3
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc157_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc157_27 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc157_33 => constants.%complete_type.2da
// CHECK:STDOUT:   %require_complete.loc159 => constants.%complete_type.8a2
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.885
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.loc159 => constants.%custom_witness.963
// CHECK:STDOUT:   %DivWith.facet => constants.%DivWith.facet.115
// CHECK:STDOUT:   %DivWith.WithSelf.Op.type => constants.%DivWith.WithSelf.Op.type.a49
// CHECK:STDOUT:   %.loc159_5.6 => constants.%.251
// CHECK:STDOUT:   %impl.elem1.loc159_5.2 => constants.%Int16.Op.4dca31.6
// CHECK:STDOUT:   %specific_impl_fn.loc159_5.3 => constants.%Int16.Op.4dca31.6
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.9d5
// CHECK:STDOUT:   %.loc159_5.7 => constants.%.6da
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.e7a31c.1
// CHECK:STDOUT:   %impl.elem0.loc159_5.2 => constants.%Int16.Op.4dca31.1
// CHECK:STDOUT:   %specific_impl_fn.loc159_5.4 => constants.%Int16.Op.4dca31.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ModWith.loc168(constants.%Destroy.facet.2cc, constants.%Destroy.facet.e5c, constants.%facet_value.ce1) {
// CHECK:STDOUT:   %Result.patt.loc165_9.2 => constants.%Result.patt
// CHECK:STDOUT:   %Result.loc165_9.1 => constants.%Destroy.facet.2cc
// CHECK:STDOUT:   %U.patt.loc166_4.2 => constants.%U.patt.7a2
// CHECK:STDOUT:   %U.loc166_4.1 => constants.%Destroy.facet.e5c
// CHECK:STDOUT:   %U.as_type.loc167_35.1 => constants.%i16
// CHECK:STDOUT:   %ModWith.type.loc167_35.1 => constants.%ModWith.type.0b2
// CHECK:STDOUT:   %facet_type.loc167_19 => constants.%facet_type.afdb
// CHECK:STDOUT:   %.Self.frozen.loc167_37.1 => constants.%.Self.frozen.46c
// CHECK:STDOUT:   %require_complete.loc167 => constants.%complete_type.1f9
// CHECK:STDOUT:   %.Self.frozen.as_type => constants.%.Self.frozen.as_type.12f
// CHECK:STDOUT:   %ModWith.assoc_type => constants.%ModWith.assoc_type.e06
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.b5a
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.loc167_43.1 => constants.%ModWith.lookup_impl_witness.5ab
// CHECK:STDOUT:   %impl.elem0.loc167_43.1 => constants.%impl.elem0.172
// CHECK:STDOUT:   %Result.as_type.loc167_53.1 => constants.%Int16
// CHECK:STDOUT:   %.Self => constants.%.Self.69d
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.loc167_43.2 => constants.%ModWith.lookup_impl_witness.b3b
// CHECK:STDOUT:   %impl.elem0.loc167_43.2 => constants.%impl.elem0.0f8
// CHECK:STDOUT:   %facet_type.loc167_37 => constants.%facet_type.ee6
// CHECK:STDOUT:   %pattern_type.loc167 => constants.%pattern_type.4d8
// CHECK:STDOUT:   %T.patt.loc167_4.2 => constants.%T.patt.a7e
// CHECK:STDOUT:   %T.loc167_4.1 => constants.%facet_value.ce1
// CHECK:STDOUT:   %pattern_type.loc168_16 => constants.%pattern_type.785
// CHECK:STDOUT:   %result.param_patt.loc168_16.2 => constants.%result.param_patt.be6
// CHECK:STDOUT:   %result.patt.loc168_16.2 => constants.%result.patt.488
// CHECK:STDOUT:   %T.as_type.loc168_29.1 => constants.%Int16
// CHECK:STDOUT:   %pattern_type.loc168_27 => constants.%pattern_type.785
// CHECK:STDOUT:   %x.param_patt.loc168_27.2 => constants.%x.param_patt.3b0
// CHECK:STDOUT:   %x.patt.loc168_27.2 => constants.%x.patt.b76
// CHECK:STDOUT:   %pattern_type.loc168_33 => constants.%pattern_type.203
// CHECK:STDOUT:   %y.param_patt.loc168_33.2 => constants.%y.param_patt.4b4
// CHECK:STDOUT:   %y.patt.loc168_33.2 => constants.%y.patt.dd3
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc168_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc168_27 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc168_33 => constants.%complete_type.2da
// CHECK:STDOUT:   %require_complete.loc170 => constants.%complete_type.385
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.972
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.loc170 => constants.%custom_witness.8f4
// CHECK:STDOUT:   %ModWith.facet => constants.%ModWith.facet.925
// CHECK:STDOUT:   %ModWith.WithSelf.Op.type => constants.%ModWith.WithSelf.Op.type.519
// CHECK:STDOUT:   %.loc170_5.6 => constants.%.66e2
// CHECK:STDOUT:   %impl.elem1.loc170_5.2 => constants.%Int16.Op.4dca31.7
// CHECK:STDOUT:   %specific_impl_fn.loc170_5.3 => constants.%Int16.Op.4dca31.7
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.9d5
// CHECK:STDOUT:   %.loc170_5.7 => constants.%.6da
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.e7a31c.1
// CHECK:STDOUT:   %impl.elem0.loc170_5.2 => constants.%Int16.Op.4dca31.1
// CHECK:STDOUT:   %specific_impl_fn.loc170_5.4 => constants.%Int16.Op.4dca31.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AddWith.loc124(constants.%Destroy.facet.099, constants.%Destroy.facet.099, constants.%facet_value.17e) {
// CHECK:STDOUT:   %Result.patt.loc121_9.2 => constants.%Result.patt
// CHECK:STDOUT:   %Result.loc121_9.1 => constants.%Destroy.facet.099
// CHECK:STDOUT:   %U.patt.loc122_4.2 => constants.%U.patt.7a2
// CHECK:STDOUT:   %U.loc122_4.1 => constants.%Destroy.facet.099
// CHECK:STDOUT:   %U.as_type.loc123_35.1 => constants.%Int32
// CHECK:STDOUT:   %AddWith.type.loc123_35.1 => constants.%AddWith.type.c71
// CHECK:STDOUT:   %facet_type.loc123_19 => constants.%facet_type.698
// CHECK:STDOUT:   %.Self.frozen.loc123_37.1 => constants.%.Self.frozen.f1e
// CHECK:STDOUT:   %require_complete.loc123 => constants.%complete_type.224
// CHECK:STDOUT:   %.Self.frozen.as_type => constants.%.Self.frozen.as_type.440
// CHECK:STDOUT:   %AddWith.assoc_type => constants.%AddWith.assoc_type.72d
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.c2b
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.loc123_43.1 => constants.%AddWith.lookup_impl_witness.7f6
// CHECK:STDOUT:   %impl.elem0.loc123_43.1 => constants.%impl.elem0.7a4
// CHECK:STDOUT:   %Result.as_type.loc123_53.1 => constants.%Int32
// CHECK:STDOUT:   %.Self => constants.%.Self.591
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.loc123_43.2 => constants.%AddWith.lookup_impl_witness.47a
// CHECK:STDOUT:   %impl.elem0.loc123_43.2 => constants.%impl.elem0.6e6
// CHECK:STDOUT:   %facet_type.loc123_37 => constants.%facet_type.ecf
// CHECK:STDOUT:   %pattern_type.loc123 => constants.%pattern_type.c976
// CHECK:STDOUT:   %T.patt.loc123_4.2 => constants.%T.patt.0a8
// CHECK:STDOUT:   %T.loc123_4.1 => constants.%facet_value.17e
// CHECK:STDOUT:   %pattern_type.loc124_16 => constants.%pattern_type.8b4
// CHECK:STDOUT:   %result.param_patt.loc124_16.2 => constants.%result.param_patt.e81
// CHECK:STDOUT:   %result.patt.loc124_16.2 => constants.%result.patt.830
// CHECK:STDOUT:   %T.as_type.loc124_29.1 => constants.%Int16
// CHECK:STDOUT:   %pattern_type.loc124_27 => constants.%pattern_type.785
// CHECK:STDOUT:   %x.param_patt.loc124_27.2 => constants.%x.param_patt.3b0
// CHECK:STDOUT:   %x.patt.loc124_27.2 => constants.%x.patt.b76
// CHECK:STDOUT:   %pattern_type.loc124_33 => constants.%pattern_type.8b4
// CHECK:STDOUT:   %y.param_patt.loc124_33.2 => constants.%y.param_patt.e0c
// CHECK:STDOUT:   %y.patt.loc124_33.2 => constants.%y.patt.0a3
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc124_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc124_27 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc124_33 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc126 => constants.%complete_type.2cc
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.0fc
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.loc126 => constants.%custom_witness.e02e0c.1
// CHECK:STDOUT:   %AddWith.facet => constants.%AddWith.facet.007
// CHECK:STDOUT:   %AddWith.WithSelf.Op.type => constants.%AddWith.WithSelf.Op.type.dc4
// CHECK:STDOUT:   %.loc126_5.6 => constants.%.dbb
// CHECK:STDOUT:   %impl.elem1.loc126_5.2 => constants.%Int16.Op.4dca31.8
// CHECK:STDOUT:   %specific_impl_fn.loc126_5.3 => constants.%Int16.Op.4dca31.8
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.b76
// CHECK:STDOUT:   %.loc126_5.7 => constants.%.043
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.e7a31c.2
// CHECK:STDOUT:   %impl.elem0.loc126_5.2 => constants.%Int32.Op.4dca31.1
// CHECK:STDOUT:   %specific_impl_fn.loc126_5.4 => constants.%Int32.Op.4dca31.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @SubWith.loc135(constants.%Destroy.facet.099, constants.%Destroy.facet.099, constants.%facet_value.9eb) {
// CHECK:STDOUT:   %Result.patt.loc132_9.2 => constants.%Result.patt
// CHECK:STDOUT:   %Result.loc132_9.1 => constants.%Destroy.facet.099
// CHECK:STDOUT:   %U.patt.loc133_4.2 => constants.%U.patt.7a2
// CHECK:STDOUT:   %U.loc133_4.1 => constants.%Destroy.facet.099
// CHECK:STDOUT:   %U.as_type.loc134_35.1 => constants.%Int32
// CHECK:STDOUT:   %SubWith.type.loc134_35.1 => constants.%SubWith.type.c81
// CHECK:STDOUT:   %facet_type.loc134_19 => constants.%facet_type.44e
// CHECK:STDOUT:   %.Self.frozen.loc134_37.1 => constants.%.Self.frozen.db8
// CHECK:STDOUT:   %require_complete.loc134 => constants.%complete_type.63f
// CHECK:STDOUT:   %.Self.frozen.as_type => constants.%.Self.frozen.as_type.068
// CHECK:STDOUT:   %SubWith.assoc_type => constants.%SubWith.assoc_type.68f
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.f11
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.loc134_43.1 => constants.%SubWith.lookup_impl_witness.32d
// CHECK:STDOUT:   %impl.elem0.loc134_43.1 => constants.%impl.elem0.821
// CHECK:STDOUT:   %Result.as_type.loc134_53.1 => constants.%Int32
// CHECK:STDOUT:   %.Self => constants.%.Self.8ac
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.loc134_43.2 => constants.%SubWith.lookup_impl_witness.f41
// CHECK:STDOUT:   %impl.elem0.loc134_43.2 => constants.%impl.elem0.e31
// CHECK:STDOUT:   %facet_type.loc134_37 => constants.%facet_type.630
// CHECK:STDOUT:   %pattern_type.loc134 => constants.%pattern_type.c58
// CHECK:STDOUT:   %T.patt.loc134_4.2 => constants.%T.patt.315
// CHECK:STDOUT:   %T.loc134_4.1 => constants.%facet_value.9eb
// CHECK:STDOUT:   %pattern_type.loc135_16 => constants.%pattern_type.8b4
// CHECK:STDOUT:   %result.param_patt.loc135_16.2 => constants.%result.param_patt.e81
// CHECK:STDOUT:   %result.patt.loc135_16.2 => constants.%result.patt.830
// CHECK:STDOUT:   %T.as_type.loc135_29.1 => constants.%Int16
// CHECK:STDOUT:   %pattern_type.loc135_27 => constants.%pattern_type.785
// CHECK:STDOUT:   %x.param_patt.loc135_27.2 => constants.%x.param_patt.3b0
// CHECK:STDOUT:   %x.patt.loc135_27.2 => constants.%x.patt.b76
// CHECK:STDOUT:   %pattern_type.loc135_33 => constants.%pattern_type.8b4
// CHECK:STDOUT:   %y.param_patt.loc135_33.2 => constants.%y.param_patt.e0c
// CHECK:STDOUT:   %y.patt.loc135_33.2 => constants.%y.patt.0a3
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc135_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc135_27 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc135_33 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc137 => constants.%complete_type.cf9
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.799
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.loc137 => constants.%custom_witness.4dea15.1
// CHECK:STDOUT:   %SubWith.facet => constants.%SubWith.facet.703
// CHECK:STDOUT:   %SubWith.WithSelf.Op.type => constants.%SubWith.WithSelf.Op.type.6eb
// CHECK:STDOUT:   %.loc137_5.6 => constants.%.114
// CHECK:STDOUT:   %impl.elem1.loc137_5.2 => constants.%Int16.Op.4dca31.9
// CHECK:STDOUT:   %specific_impl_fn.loc137_5.3 => constants.%Int16.Op.4dca31.9
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.b76
// CHECK:STDOUT:   %.loc137_5.7 => constants.%.043
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.e7a31c.2
// CHECK:STDOUT:   %impl.elem0.loc137_5.2 => constants.%Int32.Op.4dca31.1
// CHECK:STDOUT:   %specific_impl_fn.loc137_5.4 => constants.%Int32.Op.4dca31.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @MulWith.loc146(constants.%Destroy.facet.099, constants.%Destroy.facet.099, constants.%facet_value.9ae) {
// CHECK:STDOUT:   %Result.patt.loc143_9.2 => constants.%Result.patt
// CHECK:STDOUT:   %Result.loc143_9.1 => constants.%Destroy.facet.099
// CHECK:STDOUT:   %U.patt.loc144_4.2 => constants.%U.patt.7a2
// CHECK:STDOUT:   %U.loc144_4.1 => constants.%Destroy.facet.099
// CHECK:STDOUT:   %U.as_type.loc145_35.1 => constants.%Int32
// CHECK:STDOUT:   %MulWith.type.loc145_35.1 => constants.%MulWith.type.eeb
// CHECK:STDOUT:   %facet_type.loc145_19 => constants.%facet_type.ff6
// CHECK:STDOUT:   %.Self.frozen.loc145_37.1 => constants.%.Self.frozen.be6
// CHECK:STDOUT:   %require_complete.loc145 => constants.%complete_type.3fe
// CHECK:STDOUT:   %.Self.frozen.as_type => constants.%.Self.frozen.as_type.aaf
// CHECK:STDOUT:   %MulWith.assoc_type => constants.%MulWith.assoc_type.083
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.fc0
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.loc145_43.1 => constants.%MulWith.lookup_impl_witness.4cf
// CHECK:STDOUT:   %impl.elem0.loc145_43.1 => constants.%impl.elem0.167
// CHECK:STDOUT:   %Result.as_type.loc145_53.1 => constants.%Int32
// CHECK:STDOUT:   %.Self => constants.%.Self.d4f
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.loc145_43.2 => constants.%MulWith.lookup_impl_witness.07b
// CHECK:STDOUT:   %impl.elem0.loc145_43.2 => constants.%impl.elem0.0ea
// CHECK:STDOUT:   %facet_type.loc145_37 => constants.%facet_type.26b
// CHECK:STDOUT:   %pattern_type.loc145 => constants.%pattern_type.354
// CHECK:STDOUT:   %T.patt.loc145_4.2 => constants.%T.patt.4b7
// CHECK:STDOUT:   %T.loc145_4.1 => constants.%facet_value.9ae
// CHECK:STDOUT:   %pattern_type.loc146_16 => constants.%pattern_type.8b4
// CHECK:STDOUT:   %result.param_patt.loc146_16.2 => constants.%result.param_patt.e81
// CHECK:STDOUT:   %result.patt.loc146_16.2 => constants.%result.patt.830
// CHECK:STDOUT:   %T.as_type.loc146_29.1 => constants.%Int16
// CHECK:STDOUT:   %pattern_type.loc146_27 => constants.%pattern_type.785
// CHECK:STDOUT:   %x.param_patt.loc146_27.2 => constants.%x.param_patt.3b0
// CHECK:STDOUT:   %x.patt.loc146_27.2 => constants.%x.patt.b76
// CHECK:STDOUT:   %pattern_type.loc146_33 => constants.%pattern_type.8b4
// CHECK:STDOUT:   %y.param_patt.loc146_33.2 => constants.%y.param_patt.e0c
// CHECK:STDOUT:   %y.patt.loc146_33.2 => constants.%y.patt.0a3
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc146_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc146_27 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc146_33 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc148 => constants.%complete_type.e76
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.019
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.loc148 => constants.%custom_witness.337c88.1
// CHECK:STDOUT:   %MulWith.facet => constants.%MulWith.facet.fcd
// CHECK:STDOUT:   %MulWith.WithSelf.Op.type => constants.%MulWith.WithSelf.Op.type.99f
// CHECK:STDOUT:   %.loc148_5.6 => constants.%.c1c
// CHECK:STDOUT:   %impl.elem1.loc148_5.2 => constants.%Int16.Op.4dca31.10
// CHECK:STDOUT:   %specific_impl_fn.loc148_5.3 => constants.%Int16.Op.4dca31.10
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.b76
// CHECK:STDOUT:   %.loc148_5.7 => constants.%.043
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.e7a31c.2
// CHECK:STDOUT:   %impl.elem0.loc148_5.2 => constants.%Int32.Op.4dca31.1
// CHECK:STDOUT:   %specific_impl_fn.loc148_5.4 => constants.%Int32.Op.4dca31.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @DivWith.loc157(constants.%Destroy.facet.099, constants.%Destroy.facet.099, constants.%facet_value.38c) {
// CHECK:STDOUT:   %Result.patt.loc154_9.2 => constants.%Result.patt
// CHECK:STDOUT:   %Result.loc154_9.1 => constants.%Destroy.facet.099
// CHECK:STDOUT:   %U.patt.loc155_4.2 => constants.%U.patt.7a2
// CHECK:STDOUT:   %U.loc155_4.1 => constants.%Destroy.facet.099
// CHECK:STDOUT:   %U.as_type.loc156_35.1 => constants.%Int32
// CHECK:STDOUT:   %DivWith.type.loc156_35.1 => constants.%DivWith.type.fb1
// CHECK:STDOUT:   %facet_type.loc156_19 => constants.%facet_type.d41
// CHECK:STDOUT:   %.Self.frozen.loc156_37.1 => constants.%.Self.frozen.be1
// CHECK:STDOUT:   %require_complete.loc156 => constants.%complete_type.11b
// CHECK:STDOUT:   %.Self.frozen.as_type => constants.%.Self.frozen.as_type.88a
// CHECK:STDOUT:   %DivWith.assoc_type => constants.%DivWith.assoc_type.725
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.2da
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.loc156_43.1 => constants.%DivWith.lookup_impl_witness.385
// CHECK:STDOUT:   %impl.elem0.loc156_43.1 => constants.%impl.elem0.b42
// CHECK:STDOUT:   %Result.as_type.loc156_53.1 => constants.%Int32
// CHECK:STDOUT:   %.Self => constants.%.Self.13f
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.loc156_43.2 => constants.%DivWith.lookup_impl_witness.abd
// CHECK:STDOUT:   %impl.elem0.loc156_43.2 => constants.%impl.elem0.1a6
// CHECK:STDOUT:   %facet_type.loc156_37 => constants.%facet_type.1a8
// CHECK:STDOUT:   %pattern_type.loc156 => constants.%pattern_type.761
// CHECK:STDOUT:   %T.patt.loc156_4.2 => constants.%T.patt.f94
// CHECK:STDOUT:   %T.loc156_4.1 => constants.%facet_value.38c
// CHECK:STDOUT:   %pattern_type.loc157_16 => constants.%pattern_type.8b4
// CHECK:STDOUT:   %result.param_patt.loc157_16.2 => constants.%result.param_patt.e81
// CHECK:STDOUT:   %result.patt.loc157_16.2 => constants.%result.patt.830
// CHECK:STDOUT:   %T.as_type.loc157_29.1 => constants.%Int16
// CHECK:STDOUT:   %pattern_type.loc157_27 => constants.%pattern_type.785
// CHECK:STDOUT:   %x.param_patt.loc157_27.2 => constants.%x.param_patt.3b0
// CHECK:STDOUT:   %x.patt.loc157_27.2 => constants.%x.patt.b76
// CHECK:STDOUT:   %pattern_type.loc157_33 => constants.%pattern_type.8b4
// CHECK:STDOUT:   %y.param_patt.loc157_33.2 => constants.%y.param_patt.e0c
// CHECK:STDOUT:   %y.patt.loc157_33.2 => constants.%y.patt.0a3
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc157_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc157_27 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc157_33 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc159 => constants.%complete_type.595
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.580
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.loc159 => constants.%custom_witness.3eb227.1
// CHECK:STDOUT:   %DivWith.facet => constants.%DivWith.facet.3bd
// CHECK:STDOUT:   %DivWith.WithSelf.Op.type => constants.%DivWith.WithSelf.Op.type.500
// CHECK:STDOUT:   %.loc159_5.6 => constants.%.ebc
// CHECK:STDOUT:   %impl.elem1.loc159_5.2 => constants.%Int16.Op.4dca31.11
// CHECK:STDOUT:   %specific_impl_fn.loc159_5.3 => constants.%Int16.Op.4dca31.11
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.b76
// CHECK:STDOUT:   %.loc159_5.7 => constants.%.043
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.e7a31c.2
// CHECK:STDOUT:   %impl.elem0.loc159_5.2 => constants.%Int32.Op.4dca31.1
// CHECK:STDOUT:   %specific_impl_fn.loc159_5.4 => constants.%Int32.Op.4dca31.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ModWith.loc168(constants.%Destroy.facet.099, constants.%Destroy.facet.099, constants.%facet_value.68d) {
// CHECK:STDOUT:   %Result.patt.loc165_9.2 => constants.%Result.patt
// CHECK:STDOUT:   %Result.loc165_9.1 => constants.%Destroy.facet.099
// CHECK:STDOUT:   %U.patt.loc166_4.2 => constants.%U.patt.7a2
// CHECK:STDOUT:   %U.loc166_4.1 => constants.%Destroy.facet.099
// CHECK:STDOUT:   %U.as_type.loc167_35.1 => constants.%Int32
// CHECK:STDOUT:   %ModWith.type.loc167_35.1 => constants.%ModWith.type.253
// CHECK:STDOUT:   %facet_type.loc167_19 => constants.%facet_type.ea8
// CHECK:STDOUT:   %.Self.frozen.loc167_37.1 => constants.%.Self.frozen.4b6
// CHECK:STDOUT:   %require_complete.loc167 => constants.%complete_type.2aa
// CHECK:STDOUT:   %.Self.frozen.as_type => constants.%.Self.frozen.as_type.080
// CHECK:STDOUT:   %ModWith.assoc_type => constants.%ModWith.assoc_type.47d
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.45c
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.loc167_43.1 => constants.%ModWith.lookup_impl_witness.fe2
// CHECK:STDOUT:   %impl.elem0.loc167_43.1 => constants.%impl.elem0.e60
// CHECK:STDOUT:   %Result.as_type.loc167_53.1 => constants.%Int32
// CHECK:STDOUT:   %.Self => constants.%.Self.008
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.loc167_43.2 => constants.%ModWith.lookup_impl_witness.557
// CHECK:STDOUT:   %impl.elem0.loc167_43.2 => constants.%impl.elem0.cdb
// CHECK:STDOUT:   %facet_type.loc167_37 => constants.%facet_type.c8c
// CHECK:STDOUT:   %pattern_type.loc167 => constants.%pattern_type.c9a
// CHECK:STDOUT:   %T.patt.loc167_4.2 => constants.%T.patt.b4d
// CHECK:STDOUT:   %T.loc167_4.1 => constants.%facet_value.68d
// CHECK:STDOUT:   %pattern_type.loc168_16 => constants.%pattern_type.8b4
// CHECK:STDOUT:   %result.param_patt.loc168_16.2 => constants.%result.param_patt.e81
// CHECK:STDOUT:   %result.patt.loc168_16.2 => constants.%result.patt.830
// CHECK:STDOUT:   %T.as_type.loc168_29.1 => constants.%Int16
// CHECK:STDOUT:   %pattern_type.loc168_27 => constants.%pattern_type.785
// CHECK:STDOUT:   %x.param_patt.loc168_27.2 => constants.%x.param_patt.3b0
// CHECK:STDOUT:   %x.patt.loc168_27.2 => constants.%x.patt.b76
// CHECK:STDOUT:   %pattern_type.loc168_33 => constants.%pattern_type.8b4
// CHECK:STDOUT:   %y.param_patt.loc168_33.2 => constants.%y.param_patt.e0c
// CHECK:STDOUT:   %y.patt.loc168_33.2 => constants.%y.patt.0a3
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc168_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc168_27 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc168_33 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc170 => constants.%complete_type.80f
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.878
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.loc170 => constants.%custom_witness.501c6d.1
// CHECK:STDOUT:   %ModWith.facet => constants.%ModWith.facet.a29
// CHECK:STDOUT:   %ModWith.WithSelf.Op.type => constants.%ModWith.WithSelf.Op.type.d03
// CHECK:STDOUT:   %.loc170_5.6 => constants.%.839
// CHECK:STDOUT:   %impl.elem1.loc170_5.2 => constants.%Int16.Op.4dca31.12
// CHECK:STDOUT:   %specific_impl_fn.loc170_5.3 => constants.%Int16.Op.4dca31.12
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.b76
// CHECK:STDOUT:   %.loc170_5.7 => constants.%.043
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.e7a31c.2
// CHECK:STDOUT:   %impl.elem0.loc170_5.2 => constants.%Int32.Op.4dca31.1
// CHECK:STDOUT:   %specific_impl_fn.loc170_5.4 => constants.%Int32.Op.4dca31.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AddWith.loc124(constants.%Destroy.facet.099, constants.%Destroy.facet.2cc, constants.%facet_value.6ea) {
// CHECK:STDOUT:   %Result.patt.loc121_9.2 => constants.%Result.patt
// CHECK:STDOUT:   %Result.loc121_9.1 => constants.%Destroy.facet.099
// CHECK:STDOUT:   %U.patt.loc122_4.2 => constants.%U.patt.7a2
// CHECK:STDOUT:   %U.loc122_4.1 => constants.%Destroy.facet.2cc
// CHECK:STDOUT:   %U.as_type.loc123_35.1 => constants.%Int16
// CHECK:STDOUT:   %AddWith.type.loc123_35.1 => constants.%AddWith.type.b62
// CHECK:STDOUT:   %facet_type.loc123_19 => constants.%facet_type.297
// CHECK:STDOUT:   %.Self.frozen.loc123_37.1 => constants.%.Self.frozen.7e3
// CHECK:STDOUT:   %require_complete.loc123 => constants.%complete_type.a82
// CHECK:STDOUT:   %.Self.frozen.as_type => constants.%.Self.frozen.as_type.85e
// CHECK:STDOUT:   %AddWith.assoc_type => constants.%AddWith.assoc_type.c72
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.261
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.loc123_43.1 => constants.%AddWith.lookup_impl_witness.5b7
// CHECK:STDOUT:   %impl.elem0.loc123_43.1 => constants.%impl.elem0.d80
// CHECK:STDOUT:   %Result.as_type.loc123_53.1 => constants.%Int32
// CHECK:STDOUT:   %.Self => constants.%.Self.45a
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.loc123_43.2 => constants.%AddWith.lookup_impl_witness.131
// CHECK:STDOUT:   %impl.elem0.loc123_43.2 => constants.%impl.elem0.28c
// CHECK:STDOUT:   %facet_type.loc123_37 => constants.%facet_type.37e
// CHECK:STDOUT:   %pattern_type.loc123 => constants.%pattern_type.363
// CHECK:STDOUT:   %T.patt.loc123_4.2 => constants.%T.patt.ccd
// CHECK:STDOUT:   %T.loc123_4.1 => constants.%facet_value.6ea
// CHECK:STDOUT:   %pattern_type.loc124_16 => constants.%pattern_type.8b4
// CHECK:STDOUT:   %result.param_patt.loc124_16.2 => constants.%result.param_patt.e81
// CHECK:STDOUT:   %result.patt.loc124_16.2 => constants.%result.patt.830
// CHECK:STDOUT:   %T.as_type.loc124_29.1 => constants.%Int32
// CHECK:STDOUT:   %pattern_type.loc124_27 => constants.%pattern_type.8b4
// CHECK:STDOUT:   %x.param_patt.loc124_27.2 => constants.%x.param_patt.27c
// CHECK:STDOUT:   %x.patt.loc124_27.2 => constants.%x.patt.4c9
// CHECK:STDOUT:   %pattern_type.loc124_33 => constants.%pattern_type.785
// CHECK:STDOUT:   %y.param_patt.loc124_33.2 => constants.%y.param_patt.418
// CHECK:STDOUT:   %y.patt.loc124_33.2 => constants.%y.patt.4e6
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc124_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc124_27 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc124_33 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc126 => constants.%complete_type.7a6
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.006
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.loc126 => constants.%custom_witness.15f
// CHECK:STDOUT:   %AddWith.facet => constants.%AddWith.facet.35b
// CHECK:STDOUT:   %AddWith.WithSelf.Op.type => constants.%AddWith.WithSelf.Op.type.29e
// CHECK:STDOUT:   %.loc126_5.6 => constants.%.45e
// CHECK:STDOUT:   %impl.elem1.loc126_5.2 => constants.%Int32.Op.4dca31.2
// CHECK:STDOUT:   %specific_impl_fn.loc126_5.3 => constants.%Int32.Op.4dca31.2
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.b76
// CHECK:STDOUT:   %.loc126_5.7 => constants.%.043
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.e7a31c.2
// CHECK:STDOUT:   %impl.elem0.loc126_5.2 => constants.%Int32.Op.4dca31.1
// CHECK:STDOUT:   %specific_impl_fn.loc126_5.4 => constants.%Int32.Op.4dca31.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @SubWith.loc135(constants.%Destroy.facet.099, constants.%Destroy.facet.2cc, constants.%facet_value.bcc) {
// CHECK:STDOUT:   %Result.patt.loc132_9.2 => constants.%Result.patt
// CHECK:STDOUT:   %Result.loc132_9.1 => constants.%Destroy.facet.099
// CHECK:STDOUT:   %U.patt.loc133_4.2 => constants.%U.patt.7a2
// CHECK:STDOUT:   %U.loc133_4.1 => constants.%Destroy.facet.2cc
// CHECK:STDOUT:   %U.as_type.loc134_35.1 => constants.%Int16
// CHECK:STDOUT:   %SubWith.type.loc134_35.1 => constants.%SubWith.type.b12
// CHECK:STDOUT:   %facet_type.loc134_19 => constants.%facet_type.b15
// CHECK:STDOUT:   %.Self.frozen.loc134_37.1 => constants.%.Self.frozen.581
// CHECK:STDOUT:   %require_complete.loc134 => constants.%complete_type.647
// CHECK:STDOUT:   %.Self.frozen.as_type => constants.%.Self.frozen.as_type.092
// CHECK:STDOUT:   %SubWith.assoc_type => constants.%SubWith.assoc_type.eb8
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.a90
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.loc134_43.1 => constants.%SubWith.lookup_impl_witness.b30
// CHECK:STDOUT:   %impl.elem0.loc134_43.1 => constants.%impl.elem0.748
// CHECK:STDOUT:   %Result.as_type.loc134_53.1 => constants.%Int32
// CHECK:STDOUT:   %.Self => constants.%.Self.415
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.loc134_43.2 => constants.%SubWith.lookup_impl_witness.266
// CHECK:STDOUT:   %impl.elem0.loc134_43.2 => constants.%impl.elem0.4a0
// CHECK:STDOUT:   %facet_type.loc134_37 => constants.%facet_type.983
// CHECK:STDOUT:   %pattern_type.loc134 => constants.%pattern_type.af3
// CHECK:STDOUT:   %T.patt.loc134_4.2 => constants.%T.patt.14c
// CHECK:STDOUT:   %T.loc134_4.1 => constants.%facet_value.bcc
// CHECK:STDOUT:   %pattern_type.loc135_16 => constants.%pattern_type.8b4
// CHECK:STDOUT:   %result.param_patt.loc135_16.2 => constants.%result.param_patt.e81
// CHECK:STDOUT:   %result.patt.loc135_16.2 => constants.%result.patt.830
// CHECK:STDOUT:   %T.as_type.loc135_29.1 => constants.%Int32
// CHECK:STDOUT:   %pattern_type.loc135_27 => constants.%pattern_type.8b4
// CHECK:STDOUT:   %x.param_patt.loc135_27.2 => constants.%x.param_patt.27c
// CHECK:STDOUT:   %x.patt.loc135_27.2 => constants.%x.patt.4c9
// CHECK:STDOUT:   %pattern_type.loc135_33 => constants.%pattern_type.785
// CHECK:STDOUT:   %y.param_patt.loc135_33.2 => constants.%y.param_patt.418
// CHECK:STDOUT:   %y.patt.loc135_33.2 => constants.%y.patt.4e6
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc135_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc135_27 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc135_33 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc137 => constants.%complete_type.b9a
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.2cf
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.loc137 => constants.%custom_witness.bb4
// CHECK:STDOUT:   %SubWith.facet => constants.%SubWith.facet.8da
// CHECK:STDOUT:   %SubWith.WithSelf.Op.type => constants.%SubWith.WithSelf.Op.type.85b
// CHECK:STDOUT:   %.loc137_5.6 => constants.%.db9
// CHECK:STDOUT:   %impl.elem1.loc137_5.2 => constants.%Int32.Op.4dca31.3
// CHECK:STDOUT:   %specific_impl_fn.loc137_5.3 => constants.%Int32.Op.4dca31.3
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.b76
// CHECK:STDOUT:   %.loc137_5.7 => constants.%.043
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.e7a31c.2
// CHECK:STDOUT:   %impl.elem0.loc137_5.2 => constants.%Int32.Op.4dca31.1
// CHECK:STDOUT:   %specific_impl_fn.loc137_5.4 => constants.%Int32.Op.4dca31.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @MulWith.loc146(constants.%Destroy.facet.099, constants.%Destroy.facet.2cc, constants.%facet_value.795) {
// CHECK:STDOUT:   %Result.patt.loc143_9.2 => constants.%Result.patt
// CHECK:STDOUT:   %Result.loc143_9.1 => constants.%Destroy.facet.099
// CHECK:STDOUT:   %U.patt.loc144_4.2 => constants.%U.patt.7a2
// CHECK:STDOUT:   %U.loc144_4.1 => constants.%Destroy.facet.2cc
// CHECK:STDOUT:   %U.as_type.loc145_35.1 => constants.%Int16
// CHECK:STDOUT:   %MulWith.type.loc145_35.1 => constants.%MulWith.type.c18
// CHECK:STDOUT:   %facet_type.loc145_19 => constants.%facet_type.a64
// CHECK:STDOUT:   %.Self.frozen.loc145_37.1 => constants.%.Self.frozen.05e
// CHECK:STDOUT:   %require_complete.loc145 => constants.%complete_type.4d3
// CHECK:STDOUT:   %.Self.frozen.as_type => constants.%.Self.frozen.as_type.2d1
// CHECK:STDOUT:   %MulWith.assoc_type => constants.%MulWith.assoc_type.f15
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.f8b
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.loc145_43.1 => constants.%MulWith.lookup_impl_witness.701
// CHECK:STDOUT:   %impl.elem0.loc145_43.1 => constants.%impl.elem0.381
// CHECK:STDOUT:   %Result.as_type.loc145_53.1 => constants.%Int32
// CHECK:STDOUT:   %.Self => constants.%.Self.00f
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.loc145_43.2 => constants.%MulWith.lookup_impl_witness.782
// CHECK:STDOUT:   %impl.elem0.loc145_43.2 => constants.%impl.elem0.8de
// CHECK:STDOUT:   %facet_type.loc145_37 => constants.%facet_type.f48
// CHECK:STDOUT:   %pattern_type.loc145 => constants.%pattern_type.02f
// CHECK:STDOUT:   %T.patt.loc145_4.2 => constants.%T.patt.a9e
// CHECK:STDOUT:   %T.loc145_4.1 => constants.%facet_value.795
// CHECK:STDOUT:   %pattern_type.loc146_16 => constants.%pattern_type.8b4
// CHECK:STDOUT:   %result.param_patt.loc146_16.2 => constants.%result.param_patt.e81
// CHECK:STDOUT:   %result.patt.loc146_16.2 => constants.%result.patt.830
// CHECK:STDOUT:   %T.as_type.loc146_29.1 => constants.%Int32
// CHECK:STDOUT:   %pattern_type.loc146_27 => constants.%pattern_type.8b4
// CHECK:STDOUT:   %x.param_patt.loc146_27.2 => constants.%x.param_patt.27c
// CHECK:STDOUT:   %x.patt.loc146_27.2 => constants.%x.patt.4c9
// CHECK:STDOUT:   %pattern_type.loc146_33 => constants.%pattern_type.785
// CHECK:STDOUT:   %y.param_patt.loc146_33.2 => constants.%y.param_patt.418
// CHECK:STDOUT:   %y.patt.loc146_33.2 => constants.%y.patt.4e6
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc146_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc146_27 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc146_33 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc148 => constants.%complete_type.433
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.eb5
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.loc148 => constants.%custom_witness.0ab
// CHECK:STDOUT:   %MulWith.facet => constants.%MulWith.facet.920
// CHECK:STDOUT:   %MulWith.WithSelf.Op.type => constants.%MulWith.WithSelf.Op.type.545
// CHECK:STDOUT:   %.loc148_5.6 => constants.%.c5e
// CHECK:STDOUT:   %impl.elem1.loc148_5.2 => constants.%Int32.Op.4dca31.4
// CHECK:STDOUT:   %specific_impl_fn.loc148_5.3 => constants.%Int32.Op.4dca31.4
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.b76
// CHECK:STDOUT:   %.loc148_5.7 => constants.%.043
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.e7a31c.2
// CHECK:STDOUT:   %impl.elem0.loc148_5.2 => constants.%Int32.Op.4dca31.1
// CHECK:STDOUT:   %specific_impl_fn.loc148_5.4 => constants.%Int32.Op.4dca31.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @DivWith.loc157(constants.%Destroy.facet.099, constants.%Destroy.facet.2cc, constants.%facet_value.c8d) {
// CHECK:STDOUT:   %Result.patt.loc154_9.2 => constants.%Result.patt
// CHECK:STDOUT:   %Result.loc154_9.1 => constants.%Destroy.facet.099
// CHECK:STDOUT:   %U.patt.loc155_4.2 => constants.%U.patt.7a2
// CHECK:STDOUT:   %U.loc155_4.1 => constants.%Destroy.facet.2cc
// CHECK:STDOUT:   %U.as_type.loc156_35.1 => constants.%Int16
// CHECK:STDOUT:   %DivWith.type.loc156_35.1 => constants.%DivWith.type.6f9
// CHECK:STDOUT:   %facet_type.loc156_19 => constants.%facet_type.0b4
// CHECK:STDOUT:   %.Self.frozen.loc156_37.1 => constants.%.Self.frozen.ecc
// CHECK:STDOUT:   %require_complete.loc156 => constants.%complete_type.acc
// CHECK:STDOUT:   %.Self.frozen.as_type => constants.%.Self.frozen.as_type.851
// CHECK:STDOUT:   %DivWith.assoc_type => constants.%DivWith.assoc_type.333
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.962
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.loc156_43.1 => constants.%DivWith.lookup_impl_witness.6fa
// CHECK:STDOUT:   %impl.elem0.loc156_43.1 => constants.%impl.elem0.e0f
// CHECK:STDOUT:   %Result.as_type.loc156_53.1 => constants.%Int32
// CHECK:STDOUT:   %.Self => constants.%.Self.9b5
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.loc156_43.2 => constants.%DivWith.lookup_impl_witness.648
// CHECK:STDOUT:   %impl.elem0.loc156_43.2 => constants.%impl.elem0.36a
// CHECK:STDOUT:   %facet_type.loc156_37 => constants.%facet_type.50d
// CHECK:STDOUT:   %pattern_type.loc156 => constants.%pattern_type.9af
// CHECK:STDOUT:   %T.patt.loc156_4.2 => constants.%T.patt.dfa
// CHECK:STDOUT:   %T.loc156_4.1 => constants.%facet_value.c8d
// CHECK:STDOUT:   %pattern_type.loc157_16 => constants.%pattern_type.8b4
// CHECK:STDOUT:   %result.param_patt.loc157_16.2 => constants.%result.param_patt.e81
// CHECK:STDOUT:   %result.patt.loc157_16.2 => constants.%result.patt.830
// CHECK:STDOUT:   %T.as_type.loc157_29.1 => constants.%Int32
// CHECK:STDOUT:   %pattern_type.loc157_27 => constants.%pattern_type.8b4
// CHECK:STDOUT:   %x.param_patt.loc157_27.2 => constants.%x.param_patt.27c
// CHECK:STDOUT:   %x.patt.loc157_27.2 => constants.%x.patt.4c9
// CHECK:STDOUT:   %pattern_type.loc157_33 => constants.%pattern_type.785
// CHECK:STDOUT:   %y.param_patt.loc157_33.2 => constants.%y.param_patt.418
// CHECK:STDOUT:   %y.patt.loc157_33.2 => constants.%y.patt.4e6
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc157_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc157_27 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc157_33 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc159 => constants.%complete_type.64f
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.8b5
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.loc159 => constants.%custom_witness.3ec
// CHECK:STDOUT:   %DivWith.facet => constants.%DivWith.facet.c79
// CHECK:STDOUT:   %DivWith.WithSelf.Op.type => constants.%DivWith.WithSelf.Op.type.a6b
// CHECK:STDOUT:   %.loc159_5.6 => constants.%.558
// CHECK:STDOUT:   %impl.elem1.loc159_5.2 => constants.%Int32.Op.4dca31.5
// CHECK:STDOUT:   %specific_impl_fn.loc159_5.3 => constants.%Int32.Op.4dca31.5
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.b76
// CHECK:STDOUT:   %.loc159_5.7 => constants.%.043
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.e7a31c.2
// CHECK:STDOUT:   %impl.elem0.loc159_5.2 => constants.%Int32.Op.4dca31.1
// CHECK:STDOUT:   %specific_impl_fn.loc159_5.4 => constants.%Int32.Op.4dca31.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ModWith.loc168(constants.%Destroy.facet.099, constants.%Destroy.facet.2cc, constants.%facet_value.da8) {
// CHECK:STDOUT:   %Result.patt.loc165_9.2 => constants.%Result.patt
// CHECK:STDOUT:   %Result.loc165_9.1 => constants.%Destroy.facet.099
// CHECK:STDOUT:   %U.patt.loc166_4.2 => constants.%U.patt.7a2
// CHECK:STDOUT:   %U.loc166_4.1 => constants.%Destroy.facet.2cc
// CHECK:STDOUT:   %U.as_type.loc167_35.1 => constants.%Int16
// CHECK:STDOUT:   %ModWith.type.loc167_35.1 => constants.%ModWith.type.763
// CHECK:STDOUT:   %facet_type.loc167_19 => constants.%facet_type.ee8
// CHECK:STDOUT:   %.Self.frozen.loc167_37.1 => constants.%.Self.frozen.9a7
// CHECK:STDOUT:   %require_complete.loc167 => constants.%complete_type.8fc
// CHECK:STDOUT:   %.Self.frozen.as_type => constants.%.Self.frozen.as_type.954
// CHECK:STDOUT:   %ModWith.assoc_type => constants.%ModWith.assoc_type.e6a
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.2cc
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.loc167_43.1 => constants.%ModWith.lookup_impl_witness.689
// CHECK:STDOUT:   %impl.elem0.loc167_43.1 => constants.%impl.elem0.42c
// CHECK:STDOUT:   %Result.as_type.loc167_53.1 => constants.%Int32
// CHECK:STDOUT:   %.Self => constants.%.Self.67f
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.loc167_43.2 => constants.%ModWith.lookup_impl_witness.af3
// CHECK:STDOUT:   %impl.elem0.loc167_43.2 => constants.%impl.elem0.436
// CHECK:STDOUT:   %facet_type.loc167_37 => constants.%facet_type.b37
// CHECK:STDOUT:   %pattern_type.loc167 => constants.%pattern_type.448
// CHECK:STDOUT:   %T.patt.loc167_4.2 => constants.%T.patt.528
// CHECK:STDOUT:   %T.loc167_4.1 => constants.%facet_value.da8
// CHECK:STDOUT:   %pattern_type.loc168_16 => constants.%pattern_type.8b4
// CHECK:STDOUT:   %result.param_patt.loc168_16.2 => constants.%result.param_patt.e81
// CHECK:STDOUT:   %result.patt.loc168_16.2 => constants.%result.patt.830
// CHECK:STDOUT:   %T.as_type.loc168_29.1 => constants.%Int32
// CHECK:STDOUT:   %pattern_type.loc168_27 => constants.%pattern_type.8b4
// CHECK:STDOUT:   %x.param_patt.loc168_27.2 => constants.%x.param_patt.27c
// CHECK:STDOUT:   %x.patt.loc168_27.2 => constants.%x.patt.4c9
// CHECK:STDOUT:   %pattern_type.loc168_33 => constants.%pattern_type.785
// CHECK:STDOUT:   %y.param_patt.loc168_33.2 => constants.%y.param_patt.418
// CHECK:STDOUT:   %y.patt.loc168_33.2 => constants.%y.patt.4e6
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc168_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc168_27 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc168_33 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc170 => constants.%complete_type.dea
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.3dd
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.loc170 => constants.%custom_witness.de0
// CHECK:STDOUT:   %ModWith.facet => constants.%ModWith.facet.bef
// CHECK:STDOUT:   %ModWith.WithSelf.Op.type => constants.%ModWith.WithSelf.Op.type.d54
// CHECK:STDOUT:   %.loc170_5.6 => constants.%.8c0
// CHECK:STDOUT:   %impl.elem1.loc170_5.2 => constants.%Int32.Op.4dca31.6
// CHECK:STDOUT:   %specific_impl_fn.loc170_5.3 => constants.%Int32.Op.4dca31.6
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.b76
// CHECK:STDOUT:   %.loc170_5.7 => constants.%.043
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.e7a31c.2
// CHECK:STDOUT:   %impl.elem0.loc170_5.2 => constants.%Int32.Op.4dca31.1
// CHECK:STDOUT:   %specific_impl_fn.loc170_5.4 => constants.%Int32.Op.4dca31.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AddWith.loc124(constants.%Destroy.facet.099, constants.%Destroy.facet.099, constants.%facet_value.f1a) {
// CHECK:STDOUT:   %Result.patt.loc121_9.2 => constants.%Result.patt
// CHECK:STDOUT:   %Result.loc121_9.1 => constants.%Destroy.facet.099
// CHECK:STDOUT:   %U.patt.loc122_4.2 => constants.%U.patt.7a2
// CHECK:STDOUT:   %U.loc122_4.1 => constants.%Destroy.facet.099
// CHECK:STDOUT:   %U.as_type.loc123_35.1 => constants.%Int32
// CHECK:STDOUT:   %AddWith.type.loc123_35.1 => constants.%AddWith.type.c71
// CHECK:STDOUT:   %facet_type.loc123_19 => constants.%facet_type.698
// CHECK:STDOUT:   %.Self.frozen.loc123_37.1 => constants.%.Self.frozen.f1e
// CHECK:STDOUT:   %require_complete.loc123 => constants.%complete_type.224
// CHECK:STDOUT:   %.Self.frozen.as_type => constants.%.Self.frozen.as_type.440
// CHECK:STDOUT:   %AddWith.assoc_type => constants.%AddWith.assoc_type.72d
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.c2b
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.loc123_43.1 => constants.%AddWith.lookup_impl_witness.7f6
// CHECK:STDOUT:   %impl.elem0.loc123_43.1 => constants.%impl.elem0.7a4
// CHECK:STDOUT:   %Result.as_type.loc123_53.1 => constants.%Int32
// CHECK:STDOUT:   %.Self => constants.%.Self.591
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.loc123_43.2 => constants.%AddWith.lookup_impl_witness.47a
// CHECK:STDOUT:   %impl.elem0.loc123_43.2 => constants.%impl.elem0.6e6
// CHECK:STDOUT:   %facet_type.loc123_37 => constants.%facet_type.ecf
// CHECK:STDOUT:   %pattern_type.loc123 => constants.%pattern_type.c976
// CHECK:STDOUT:   %T.patt.loc123_4.2 => constants.%T.patt.0a8
// CHECK:STDOUT:   %T.loc123_4.1 => constants.%facet_value.f1a
// CHECK:STDOUT:   %pattern_type.loc124_16 => constants.%pattern_type.8b4
// CHECK:STDOUT:   %result.param_patt.loc124_16.2 => constants.%result.param_patt.e81
// CHECK:STDOUT:   %result.patt.loc124_16.2 => constants.%result.patt.830
// CHECK:STDOUT:   %T.as_type.loc124_29.1 => constants.%Int32
// CHECK:STDOUT:   %pattern_type.loc124_27 => constants.%pattern_type.8b4
// CHECK:STDOUT:   %x.param_patt.loc124_27.2 => constants.%x.param_patt.27c
// CHECK:STDOUT:   %x.patt.loc124_27.2 => constants.%x.patt.4c9
// CHECK:STDOUT:   %pattern_type.loc124_33 => constants.%pattern_type.8b4
// CHECK:STDOUT:   %y.param_patt.loc124_33.2 => constants.%y.param_patt.e0c
// CHECK:STDOUT:   %y.patt.loc124_33.2 => constants.%y.patt.0a3
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc124_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc124_27 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc124_33 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc126 => constants.%complete_type.2cc
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.0fc
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.loc126 => constants.%custom_witness.e02e0c.2
// CHECK:STDOUT:   %AddWith.facet => constants.%AddWith.facet.ee1
// CHECK:STDOUT:   %AddWith.WithSelf.Op.type => constants.%AddWith.WithSelf.Op.type.5a5
// CHECK:STDOUT:   %.loc126_5.6 => constants.%.dcb
// CHECK:STDOUT:   %impl.elem1.loc126_5.2 => constants.%Int32.Op.4dca31.7
// CHECK:STDOUT:   %specific_impl_fn.loc126_5.3 => constants.%Int32.Op.4dca31.7
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.b76
// CHECK:STDOUT:   %.loc126_5.7 => constants.%.043
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.e7a31c.2
// CHECK:STDOUT:   %impl.elem0.loc126_5.2 => constants.%Int32.Op.4dca31.1
// CHECK:STDOUT:   %specific_impl_fn.loc126_5.4 => constants.%Int32.Op.4dca31.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @SubWith.loc135(constants.%Destroy.facet.099, constants.%Destroy.facet.099, constants.%facet_value.6fc) {
// CHECK:STDOUT:   %Result.patt.loc132_9.2 => constants.%Result.patt
// CHECK:STDOUT:   %Result.loc132_9.1 => constants.%Destroy.facet.099
// CHECK:STDOUT:   %U.patt.loc133_4.2 => constants.%U.patt.7a2
// CHECK:STDOUT:   %U.loc133_4.1 => constants.%Destroy.facet.099
// CHECK:STDOUT:   %U.as_type.loc134_35.1 => constants.%Int32
// CHECK:STDOUT:   %SubWith.type.loc134_35.1 => constants.%SubWith.type.c81
// CHECK:STDOUT:   %facet_type.loc134_19 => constants.%facet_type.44e
// CHECK:STDOUT:   %.Self.frozen.loc134_37.1 => constants.%.Self.frozen.db8
// CHECK:STDOUT:   %require_complete.loc134 => constants.%complete_type.63f
// CHECK:STDOUT:   %.Self.frozen.as_type => constants.%.Self.frozen.as_type.068
// CHECK:STDOUT:   %SubWith.assoc_type => constants.%SubWith.assoc_type.68f
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.f11
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.loc134_43.1 => constants.%SubWith.lookup_impl_witness.32d
// CHECK:STDOUT:   %impl.elem0.loc134_43.1 => constants.%impl.elem0.821
// CHECK:STDOUT:   %Result.as_type.loc134_53.1 => constants.%Int32
// CHECK:STDOUT:   %.Self => constants.%.Self.8ac
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.loc134_43.2 => constants.%SubWith.lookup_impl_witness.f41
// CHECK:STDOUT:   %impl.elem0.loc134_43.2 => constants.%impl.elem0.e31
// CHECK:STDOUT:   %facet_type.loc134_37 => constants.%facet_type.630
// CHECK:STDOUT:   %pattern_type.loc134 => constants.%pattern_type.c58
// CHECK:STDOUT:   %T.patt.loc134_4.2 => constants.%T.patt.315
// CHECK:STDOUT:   %T.loc134_4.1 => constants.%facet_value.6fc
// CHECK:STDOUT:   %pattern_type.loc135_16 => constants.%pattern_type.8b4
// CHECK:STDOUT:   %result.param_patt.loc135_16.2 => constants.%result.param_patt.e81
// CHECK:STDOUT:   %result.patt.loc135_16.2 => constants.%result.patt.830
// CHECK:STDOUT:   %T.as_type.loc135_29.1 => constants.%Int32
// CHECK:STDOUT:   %pattern_type.loc135_27 => constants.%pattern_type.8b4
// CHECK:STDOUT:   %x.param_patt.loc135_27.2 => constants.%x.param_patt.27c
// CHECK:STDOUT:   %x.patt.loc135_27.2 => constants.%x.patt.4c9
// CHECK:STDOUT:   %pattern_type.loc135_33 => constants.%pattern_type.8b4
// CHECK:STDOUT:   %y.param_patt.loc135_33.2 => constants.%y.param_patt.e0c
// CHECK:STDOUT:   %y.patt.loc135_33.2 => constants.%y.patt.0a3
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc135_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc135_27 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc135_33 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc137 => constants.%complete_type.cf9
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.799
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.loc137 => constants.%custom_witness.4dea15.2
// CHECK:STDOUT:   %SubWith.facet => constants.%SubWith.facet.3dd
// CHECK:STDOUT:   %SubWith.WithSelf.Op.type => constants.%SubWith.WithSelf.Op.type.ccc
// CHECK:STDOUT:   %.loc137_5.6 => constants.%.237
// CHECK:STDOUT:   %impl.elem1.loc137_5.2 => constants.%Int32.Op.4dca31.8
// CHECK:STDOUT:   %specific_impl_fn.loc137_5.3 => constants.%Int32.Op.4dca31.8
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.b76
// CHECK:STDOUT:   %.loc137_5.7 => constants.%.043
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.e7a31c.2
// CHECK:STDOUT:   %impl.elem0.loc137_5.2 => constants.%Int32.Op.4dca31.1
// CHECK:STDOUT:   %specific_impl_fn.loc137_5.4 => constants.%Int32.Op.4dca31.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @MulWith.loc146(constants.%Destroy.facet.099, constants.%Destroy.facet.099, constants.%facet_value.c80) {
// CHECK:STDOUT:   %Result.patt.loc143_9.2 => constants.%Result.patt
// CHECK:STDOUT:   %Result.loc143_9.1 => constants.%Destroy.facet.099
// CHECK:STDOUT:   %U.patt.loc144_4.2 => constants.%U.patt.7a2
// CHECK:STDOUT:   %U.loc144_4.1 => constants.%Destroy.facet.099
// CHECK:STDOUT:   %U.as_type.loc145_35.1 => constants.%Int32
// CHECK:STDOUT:   %MulWith.type.loc145_35.1 => constants.%MulWith.type.eeb
// CHECK:STDOUT:   %facet_type.loc145_19 => constants.%facet_type.ff6
// CHECK:STDOUT:   %.Self.frozen.loc145_37.1 => constants.%.Self.frozen.be6
// CHECK:STDOUT:   %require_complete.loc145 => constants.%complete_type.3fe
// CHECK:STDOUT:   %.Self.frozen.as_type => constants.%.Self.frozen.as_type.aaf
// CHECK:STDOUT:   %MulWith.assoc_type => constants.%MulWith.assoc_type.083
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.fc0
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.loc145_43.1 => constants.%MulWith.lookup_impl_witness.4cf
// CHECK:STDOUT:   %impl.elem0.loc145_43.1 => constants.%impl.elem0.167
// CHECK:STDOUT:   %Result.as_type.loc145_53.1 => constants.%Int32
// CHECK:STDOUT:   %.Self => constants.%.Self.d4f
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.loc145_43.2 => constants.%MulWith.lookup_impl_witness.07b
// CHECK:STDOUT:   %impl.elem0.loc145_43.2 => constants.%impl.elem0.0ea
// CHECK:STDOUT:   %facet_type.loc145_37 => constants.%facet_type.26b
// CHECK:STDOUT:   %pattern_type.loc145 => constants.%pattern_type.354
// CHECK:STDOUT:   %T.patt.loc145_4.2 => constants.%T.patt.4b7
// CHECK:STDOUT:   %T.loc145_4.1 => constants.%facet_value.c80
// CHECK:STDOUT:   %pattern_type.loc146_16 => constants.%pattern_type.8b4
// CHECK:STDOUT:   %result.param_patt.loc146_16.2 => constants.%result.param_patt.e81
// CHECK:STDOUT:   %result.patt.loc146_16.2 => constants.%result.patt.830
// CHECK:STDOUT:   %T.as_type.loc146_29.1 => constants.%Int32
// CHECK:STDOUT:   %pattern_type.loc146_27 => constants.%pattern_type.8b4
// CHECK:STDOUT:   %x.param_patt.loc146_27.2 => constants.%x.param_patt.27c
// CHECK:STDOUT:   %x.patt.loc146_27.2 => constants.%x.patt.4c9
// CHECK:STDOUT:   %pattern_type.loc146_33 => constants.%pattern_type.8b4
// CHECK:STDOUT:   %y.param_patt.loc146_33.2 => constants.%y.param_patt.e0c
// CHECK:STDOUT:   %y.patt.loc146_33.2 => constants.%y.patt.0a3
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc146_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc146_27 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc146_33 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc148 => constants.%complete_type.e76
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.019
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.loc148 => constants.%custom_witness.337c88.2
// CHECK:STDOUT:   %MulWith.facet => constants.%MulWith.facet.4d3
// CHECK:STDOUT:   %MulWith.WithSelf.Op.type => constants.%MulWith.WithSelf.Op.type.843
// CHECK:STDOUT:   %.loc148_5.6 => constants.%.8c9
// CHECK:STDOUT:   %impl.elem1.loc148_5.2 => constants.%Int32.Op.4dca31.9
// CHECK:STDOUT:   %specific_impl_fn.loc148_5.3 => constants.%Int32.Op.4dca31.9
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.b76
// CHECK:STDOUT:   %.loc148_5.7 => constants.%.043
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.e7a31c.2
// CHECK:STDOUT:   %impl.elem0.loc148_5.2 => constants.%Int32.Op.4dca31.1
// CHECK:STDOUT:   %specific_impl_fn.loc148_5.4 => constants.%Int32.Op.4dca31.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @DivWith.loc157(constants.%Destroy.facet.099, constants.%Destroy.facet.099, constants.%facet_value.b2c) {
// CHECK:STDOUT:   %Result.patt.loc154_9.2 => constants.%Result.patt
// CHECK:STDOUT:   %Result.loc154_9.1 => constants.%Destroy.facet.099
// CHECK:STDOUT:   %U.patt.loc155_4.2 => constants.%U.patt.7a2
// CHECK:STDOUT:   %U.loc155_4.1 => constants.%Destroy.facet.099
// CHECK:STDOUT:   %U.as_type.loc156_35.1 => constants.%Int32
// CHECK:STDOUT:   %DivWith.type.loc156_35.1 => constants.%DivWith.type.fb1
// CHECK:STDOUT:   %facet_type.loc156_19 => constants.%facet_type.d41
// CHECK:STDOUT:   %.Self.frozen.loc156_37.1 => constants.%.Self.frozen.be1
// CHECK:STDOUT:   %require_complete.loc156 => constants.%complete_type.11b
// CHECK:STDOUT:   %.Self.frozen.as_type => constants.%.Self.frozen.as_type.88a
// CHECK:STDOUT:   %DivWith.assoc_type => constants.%DivWith.assoc_type.725
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.2da
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.loc156_43.1 => constants.%DivWith.lookup_impl_witness.385
// CHECK:STDOUT:   %impl.elem0.loc156_43.1 => constants.%impl.elem0.b42
// CHECK:STDOUT:   %Result.as_type.loc156_53.1 => constants.%Int32
// CHECK:STDOUT:   %.Self => constants.%.Self.13f
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.loc156_43.2 => constants.%DivWith.lookup_impl_witness.abd
// CHECK:STDOUT:   %impl.elem0.loc156_43.2 => constants.%impl.elem0.1a6
// CHECK:STDOUT:   %facet_type.loc156_37 => constants.%facet_type.1a8
// CHECK:STDOUT:   %pattern_type.loc156 => constants.%pattern_type.761
// CHECK:STDOUT:   %T.patt.loc156_4.2 => constants.%T.patt.f94
// CHECK:STDOUT:   %T.loc156_4.1 => constants.%facet_value.b2c
// CHECK:STDOUT:   %pattern_type.loc157_16 => constants.%pattern_type.8b4
// CHECK:STDOUT:   %result.param_patt.loc157_16.2 => constants.%result.param_patt.e81
// CHECK:STDOUT:   %result.patt.loc157_16.2 => constants.%result.patt.830
// CHECK:STDOUT:   %T.as_type.loc157_29.1 => constants.%Int32
// CHECK:STDOUT:   %pattern_type.loc157_27 => constants.%pattern_type.8b4
// CHECK:STDOUT:   %x.param_patt.loc157_27.2 => constants.%x.param_patt.27c
// CHECK:STDOUT:   %x.patt.loc157_27.2 => constants.%x.patt.4c9
// CHECK:STDOUT:   %pattern_type.loc157_33 => constants.%pattern_type.8b4
// CHECK:STDOUT:   %y.param_patt.loc157_33.2 => constants.%y.param_patt.e0c
// CHECK:STDOUT:   %y.patt.loc157_33.2 => constants.%y.patt.0a3
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc157_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc157_27 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc157_33 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc159 => constants.%complete_type.595
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.580
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.loc159 => constants.%custom_witness.3eb227.2
// CHECK:STDOUT:   %DivWith.facet => constants.%DivWith.facet.4fe
// CHECK:STDOUT:   %DivWith.WithSelf.Op.type => constants.%DivWith.WithSelf.Op.type.24e
// CHECK:STDOUT:   %.loc159_5.6 => constants.%.450
// CHECK:STDOUT:   %impl.elem1.loc159_5.2 => constants.%Int32.Op.4dca31.10
// CHECK:STDOUT:   %specific_impl_fn.loc159_5.3 => constants.%Int32.Op.4dca31.10
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.b76
// CHECK:STDOUT:   %.loc159_5.7 => constants.%.043
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.e7a31c.2
// CHECK:STDOUT:   %impl.elem0.loc159_5.2 => constants.%Int32.Op.4dca31.1
// CHECK:STDOUT:   %specific_impl_fn.loc159_5.4 => constants.%Int32.Op.4dca31.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ModWith.loc168(constants.%Destroy.facet.099, constants.%Destroy.facet.099, constants.%facet_value.7ec) {
// CHECK:STDOUT:   %Result.patt.loc165_9.2 => constants.%Result.patt
// CHECK:STDOUT:   %Result.loc165_9.1 => constants.%Destroy.facet.099
// CHECK:STDOUT:   %U.patt.loc166_4.2 => constants.%U.patt.7a2
// CHECK:STDOUT:   %U.loc166_4.1 => constants.%Destroy.facet.099
// CHECK:STDOUT:   %U.as_type.loc167_35.1 => constants.%Int32
// CHECK:STDOUT:   %ModWith.type.loc167_35.1 => constants.%ModWith.type.253
// CHECK:STDOUT:   %facet_type.loc167_19 => constants.%facet_type.ea8
// CHECK:STDOUT:   %.Self.frozen.loc167_37.1 => constants.%.Self.frozen.4b6
// CHECK:STDOUT:   %require_complete.loc167 => constants.%complete_type.2aa
// CHECK:STDOUT:   %.Self.frozen.as_type => constants.%.Self.frozen.as_type.080
// CHECK:STDOUT:   %ModWith.assoc_type => constants.%ModWith.assoc_type.47d
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.45c
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.loc167_43.1 => constants.%ModWith.lookup_impl_witness.fe2
// CHECK:STDOUT:   %impl.elem0.loc167_43.1 => constants.%impl.elem0.e60
// CHECK:STDOUT:   %Result.as_type.loc167_53.1 => constants.%Int32
// CHECK:STDOUT:   %.Self => constants.%.Self.008
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.loc167_43.2 => constants.%ModWith.lookup_impl_witness.557
// CHECK:STDOUT:   %impl.elem0.loc167_43.2 => constants.%impl.elem0.cdb
// CHECK:STDOUT:   %facet_type.loc167_37 => constants.%facet_type.c8c
// CHECK:STDOUT:   %pattern_type.loc167 => constants.%pattern_type.c9a
// CHECK:STDOUT:   %T.patt.loc167_4.2 => constants.%T.patt.b4d
// CHECK:STDOUT:   %T.loc167_4.1 => constants.%facet_value.7ec
// CHECK:STDOUT:   %pattern_type.loc168_16 => constants.%pattern_type.8b4
// CHECK:STDOUT:   %result.param_patt.loc168_16.2 => constants.%result.param_patt.e81
// CHECK:STDOUT:   %result.patt.loc168_16.2 => constants.%result.patt.830
// CHECK:STDOUT:   %T.as_type.loc168_29.1 => constants.%Int32
// CHECK:STDOUT:   %pattern_type.loc168_27 => constants.%pattern_type.8b4
// CHECK:STDOUT:   %x.param_patt.loc168_27.2 => constants.%x.param_patt.27c
// CHECK:STDOUT:   %x.patt.loc168_27.2 => constants.%x.patt.4c9
// CHECK:STDOUT:   %pattern_type.loc168_33 => constants.%pattern_type.8b4
// CHECK:STDOUT:   %y.param_patt.loc168_33.2 => constants.%y.param_patt.e0c
// CHECK:STDOUT:   %y.patt.loc168_33.2 => constants.%y.patt.0a3
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc168_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc168_27 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc168_33 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc170 => constants.%complete_type.80f
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.878
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.loc170 => constants.%custom_witness.501c6d.2
// CHECK:STDOUT:   %ModWith.facet => constants.%ModWith.facet.60c
// CHECK:STDOUT:   %ModWith.WithSelf.Op.type => constants.%ModWith.WithSelf.Op.type.bed
// CHECK:STDOUT:   %.loc170_5.6 => constants.%.382
// CHECK:STDOUT:   %impl.elem1.loc170_5.2 => constants.%Int32.Op.4dca31.11
// CHECK:STDOUT:   %specific_impl_fn.loc170_5.3 => constants.%Int32.Op.4dca31.11
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.b76
// CHECK:STDOUT:   %.loc170_5.7 => constants.%.043
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.e7a31c.2
// CHECK:STDOUT:   %impl.elem0.loc170_5.2 => constants.%Int32.Op.4dca31.1
// CHECK:STDOUT:   %specific_impl_fn.loc170_5.4 => constants.%Int32.Op.4dca31.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AddWith.loc124(constants.%Destroy.facet.191, constants.%Destroy.facet.191, constants.%facet_value.178) {
// CHECK:STDOUT:   %Result.patt.loc121_9.2 => constants.%Result.patt
// CHECK:STDOUT:   %Result.loc121_9.1 => constants.%Destroy.facet.191
// CHECK:STDOUT:   %U.patt.loc122_4.2 => constants.%U.patt.7a2
// CHECK:STDOUT:   %U.loc122_4.1 => constants.%Destroy.facet.191
// CHECK:STDOUT:   %U.as_type.loc123_35.1 => constants.%Int64
// CHECK:STDOUT:   %AddWith.type.loc123_35.1 => constants.%AddWith.type.b5b
// CHECK:STDOUT:   %facet_type.loc123_19 => constants.%facet_type.afd1
// CHECK:STDOUT:   %.Self.frozen.loc123_37.1 => constants.%.Self.frozen.2e1
// CHECK:STDOUT:   %require_complete.loc123 => constants.%complete_type.412
// CHECK:STDOUT:   %.Self.frozen.as_type => constants.%.Self.frozen.as_type.d9c
// CHECK:STDOUT:   %AddWith.assoc_type => constants.%AddWith.assoc_type.0c5
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.947
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.loc123_43.1 => constants.%AddWith.lookup_impl_witness.bc7
// CHECK:STDOUT:   %impl.elem0.loc123_43.1 => constants.%impl.elem0.a41
// CHECK:STDOUT:   %Result.as_type.loc123_53.1 => constants.%Int64
// CHECK:STDOUT:   %.Self => constants.%.Self.d7b
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.loc123_43.2 => constants.%AddWith.lookup_impl_witness.085
// CHECK:STDOUT:   %impl.elem0.loc123_43.2 => constants.%impl.elem0.7e8
// CHECK:STDOUT:   %facet_type.loc123_37 => constants.%facet_type.522
// CHECK:STDOUT:   %pattern_type.loc123 => constants.%pattern_type.613
// CHECK:STDOUT:   %T.patt.loc123_4.2 => constants.%T.patt.f5c
// CHECK:STDOUT:   %T.loc123_4.1 => constants.%facet_value.178
// CHECK:STDOUT:   %pattern_type.loc124_16 => constants.%pattern_type.4aa
// CHECK:STDOUT:   %result.param_patt.loc124_16.2 => constants.%result.param_patt.1e1
// CHECK:STDOUT:   %result.patt.loc124_16.2 => constants.%result.patt.41f
// CHECK:STDOUT:   %T.as_type.loc124_29.1 => constants.%Int32
// CHECK:STDOUT:   %pattern_type.loc124_27 => constants.%pattern_type.8b4
// CHECK:STDOUT:   %x.param_patt.loc124_27.2 => constants.%x.param_patt.27c
// CHECK:STDOUT:   %x.patt.loc124_27.2 => constants.%x.patt.4c9
// CHECK:STDOUT:   %pattern_type.loc124_33 => constants.%pattern_type.4aa
// CHECK:STDOUT:   %y.param_patt.loc124_33.2 => constants.%y.param_patt.a08
// CHECK:STDOUT:   %y.patt.loc124_33.2 => constants.%y.patt.63b
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc124_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc124_27 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc124_33 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc126 => constants.%complete_type.d6f
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.18c
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.loc126 => constants.%custom_witness.68d
// CHECK:STDOUT:   %AddWith.facet => constants.%AddWith.facet.45c
// CHECK:STDOUT:   %AddWith.WithSelf.Op.type => constants.%AddWith.WithSelf.Op.type.6ad
// CHECK:STDOUT:   %.loc126_5.6 => constants.%.b22
// CHECK:STDOUT:   %impl.elem1.loc126_5.2 => constants.%Op.72717c.6
// CHECK:STDOUT:   %specific_impl_fn.loc126_5.3 => constants.%Op.72717c.6
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.c2c
// CHECK:STDOUT:   %.loc126_5.7 => constants.%.c2e
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.e7a31c.3
// CHECK:STDOUT:   %impl.elem0.loc126_5.2 => constants.%Int64.Op.4dca31.1
// CHECK:STDOUT:   %specific_impl_fn.loc126_5.4 => constants.%Int64.Op.4dca31.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @SubWith.loc135(constants.%Destroy.facet.191, constants.%Destroy.facet.191, constants.%facet_value.c78) {
// CHECK:STDOUT:   %Result.patt.loc132_9.2 => constants.%Result.patt
// CHECK:STDOUT:   %Result.loc132_9.1 => constants.%Destroy.facet.191
// CHECK:STDOUT:   %U.patt.loc133_4.2 => constants.%U.patt.7a2
// CHECK:STDOUT:   %U.loc133_4.1 => constants.%Destroy.facet.191
// CHECK:STDOUT:   %U.as_type.loc134_35.1 => constants.%Int64
// CHECK:STDOUT:   %SubWith.type.loc134_35.1 => constants.%SubWith.type.a46
// CHECK:STDOUT:   %facet_type.loc134_19 => constants.%facet_type.502
// CHECK:STDOUT:   %.Self.frozen.loc134_37.1 => constants.%.Self.frozen.da7
// CHECK:STDOUT:   %require_complete.loc134 => constants.%complete_type.181
// CHECK:STDOUT:   %.Self.frozen.as_type => constants.%.Self.frozen.as_type.436
// CHECK:STDOUT:   %SubWith.assoc_type => constants.%SubWith.assoc_type.e16
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.88a
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.loc134_43.1 => constants.%SubWith.lookup_impl_witness.362
// CHECK:STDOUT:   %impl.elem0.loc134_43.1 => constants.%impl.elem0.f3a
// CHECK:STDOUT:   %Result.as_type.loc134_53.1 => constants.%Int64
// CHECK:STDOUT:   %.Self => constants.%.Self.074
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.loc134_43.2 => constants.%SubWith.lookup_impl_witness.0c4
// CHECK:STDOUT:   %impl.elem0.loc134_43.2 => constants.%impl.elem0.68d
// CHECK:STDOUT:   %facet_type.loc134_37 => constants.%facet_type.e5e
// CHECK:STDOUT:   %pattern_type.loc134 => constants.%pattern_type.41a
// CHECK:STDOUT:   %T.patt.loc134_4.2 => constants.%T.patt.282
// CHECK:STDOUT:   %T.loc134_4.1 => constants.%facet_value.c78
// CHECK:STDOUT:   %pattern_type.loc135_16 => constants.%pattern_type.4aa
// CHECK:STDOUT:   %result.param_patt.loc135_16.2 => constants.%result.param_patt.1e1
// CHECK:STDOUT:   %result.patt.loc135_16.2 => constants.%result.patt.41f
// CHECK:STDOUT:   %T.as_type.loc135_29.1 => constants.%Int32
// CHECK:STDOUT:   %pattern_type.loc135_27 => constants.%pattern_type.8b4
// CHECK:STDOUT:   %x.param_patt.loc135_27.2 => constants.%x.param_patt.27c
// CHECK:STDOUT:   %x.patt.loc135_27.2 => constants.%x.patt.4c9
// CHECK:STDOUT:   %pattern_type.loc135_33 => constants.%pattern_type.4aa
// CHECK:STDOUT:   %y.param_patt.loc135_33.2 => constants.%y.param_patt.a08
// CHECK:STDOUT:   %y.patt.loc135_33.2 => constants.%y.patt.63b
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc135_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc135_27 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc135_33 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc137 => constants.%complete_type.b50
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.3b1
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.loc137 => constants.%custom_witness.1fa
// CHECK:STDOUT:   %SubWith.facet => constants.%SubWith.facet.3c0
// CHECK:STDOUT:   %SubWith.WithSelf.Op.type => constants.%SubWith.WithSelf.Op.type.bdd
// CHECK:STDOUT:   %.loc137_5.6 => constants.%.c3d
// CHECK:STDOUT:   %impl.elem1.loc137_5.2 => constants.%Op.72717c.7
// CHECK:STDOUT:   %specific_impl_fn.loc137_5.3 => constants.%Op.72717c.7
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.c2c
// CHECK:STDOUT:   %.loc137_5.7 => constants.%.c2e
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.e7a31c.3
// CHECK:STDOUT:   %impl.elem0.loc137_5.2 => constants.%Int64.Op.4dca31.1
// CHECK:STDOUT:   %specific_impl_fn.loc137_5.4 => constants.%Int64.Op.4dca31.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @MulWith.loc146(constants.%Destroy.facet.191, constants.%Destroy.facet.191, constants.%facet_value.34a) {
// CHECK:STDOUT:   %Result.patt.loc143_9.2 => constants.%Result.patt
// CHECK:STDOUT:   %Result.loc143_9.1 => constants.%Destroy.facet.191
// CHECK:STDOUT:   %U.patt.loc144_4.2 => constants.%U.patt.7a2
// CHECK:STDOUT:   %U.loc144_4.1 => constants.%Destroy.facet.191
// CHECK:STDOUT:   %U.as_type.loc145_35.1 => constants.%Int64
// CHECK:STDOUT:   %MulWith.type.loc145_35.1 => constants.%MulWith.type.76f
// CHECK:STDOUT:   %facet_type.loc145_19 => constants.%facet_type.770
// CHECK:STDOUT:   %.Self.frozen.loc145_37.1 => constants.%.Self.frozen.f85
// CHECK:STDOUT:   %require_complete.loc145 => constants.%complete_type.16e
// CHECK:STDOUT:   %.Self.frozen.as_type => constants.%.Self.frozen.as_type.f77
// CHECK:STDOUT:   %MulWith.assoc_type => constants.%MulWith.assoc_type.a37
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.210
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.loc145_43.1 => constants.%MulWith.lookup_impl_witness.472
// CHECK:STDOUT:   %impl.elem0.loc145_43.1 => constants.%impl.elem0.c8c
// CHECK:STDOUT:   %Result.as_type.loc145_53.1 => constants.%Int64
// CHECK:STDOUT:   %.Self => constants.%.Self.4e1
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.loc145_43.2 => constants.%MulWith.lookup_impl_witness.55a
// CHECK:STDOUT:   %impl.elem0.loc145_43.2 => constants.%impl.elem0.641
// CHECK:STDOUT:   %facet_type.loc145_37 => constants.%facet_type.9a3
// CHECK:STDOUT:   %pattern_type.loc145 => constants.%pattern_type.346
// CHECK:STDOUT:   %T.patt.loc145_4.2 => constants.%T.patt.6f8
// CHECK:STDOUT:   %T.loc145_4.1 => constants.%facet_value.34a
// CHECK:STDOUT:   %pattern_type.loc146_16 => constants.%pattern_type.4aa
// CHECK:STDOUT:   %result.param_patt.loc146_16.2 => constants.%result.param_patt.1e1
// CHECK:STDOUT:   %result.patt.loc146_16.2 => constants.%result.patt.41f
// CHECK:STDOUT:   %T.as_type.loc146_29.1 => constants.%Int32
// CHECK:STDOUT:   %pattern_type.loc146_27 => constants.%pattern_type.8b4
// CHECK:STDOUT:   %x.param_patt.loc146_27.2 => constants.%x.param_patt.27c
// CHECK:STDOUT:   %x.patt.loc146_27.2 => constants.%x.patt.4c9
// CHECK:STDOUT:   %pattern_type.loc146_33 => constants.%pattern_type.4aa
// CHECK:STDOUT:   %y.param_patt.loc146_33.2 => constants.%y.param_patt.a08
// CHECK:STDOUT:   %y.patt.loc146_33.2 => constants.%y.patt.63b
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc146_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc146_27 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc146_33 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc148 => constants.%complete_type.77d
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.cd1
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.loc148 => constants.%custom_witness.99d
// CHECK:STDOUT:   %MulWith.facet => constants.%MulWith.facet.dbd
// CHECK:STDOUT:   %MulWith.WithSelf.Op.type => constants.%MulWith.WithSelf.Op.type.cc8
// CHECK:STDOUT:   %.loc148_5.6 => constants.%.921
// CHECK:STDOUT:   %impl.elem1.loc148_5.2 => constants.%Op.72717c.8
// CHECK:STDOUT:   %specific_impl_fn.loc148_5.3 => constants.%Op.72717c.8
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.c2c
// CHECK:STDOUT:   %.loc148_5.7 => constants.%.c2e
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.e7a31c.3
// CHECK:STDOUT:   %impl.elem0.loc148_5.2 => constants.%Int64.Op.4dca31.1
// CHECK:STDOUT:   %specific_impl_fn.loc148_5.4 => constants.%Int64.Op.4dca31.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @DivWith.loc157(constants.%Destroy.facet.191, constants.%Destroy.facet.191, constants.%facet_value.dd9) {
// CHECK:STDOUT:   %Result.patt.loc154_9.2 => constants.%Result.patt
// CHECK:STDOUT:   %Result.loc154_9.1 => constants.%Destroy.facet.191
// CHECK:STDOUT:   %U.patt.loc155_4.2 => constants.%U.patt.7a2
// CHECK:STDOUT:   %U.loc155_4.1 => constants.%Destroy.facet.191
// CHECK:STDOUT:   %U.as_type.loc156_35.1 => constants.%Int64
// CHECK:STDOUT:   %DivWith.type.loc156_35.1 => constants.%DivWith.type.095
// CHECK:STDOUT:   %facet_type.loc156_19 => constants.%facet_type.c98
// CHECK:STDOUT:   %.Self.frozen.loc156_37.1 => constants.%.Self.frozen.8cb
// CHECK:STDOUT:   %require_complete.loc156 => constants.%complete_type.621
// CHECK:STDOUT:   %.Self.frozen.as_type => constants.%.Self.frozen.as_type.bdb
// CHECK:STDOUT:   %DivWith.assoc_type => constants.%DivWith.assoc_type.8a4
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.982
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.loc156_43.1 => constants.%DivWith.lookup_impl_witness.a82
// CHECK:STDOUT:   %impl.elem0.loc156_43.1 => constants.%impl.elem0.709
// CHECK:STDOUT:   %Result.as_type.loc156_53.1 => constants.%Int64
// CHECK:STDOUT:   %.Self => constants.%.Self.f70
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.loc156_43.2 => constants.%DivWith.lookup_impl_witness.4c3
// CHECK:STDOUT:   %impl.elem0.loc156_43.2 => constants.%impl.elem0.622
// CHECK:STDOUT:   %facet_type.loc156_37 => constants.%facet_type.641
// CHECK:STDOUT:   %pattern_type.loc156 => constants.%pattern_type.719
// CHECK:STDOUT:   %T.patt.loc156_4.2 => constants.%T.patt.e0b
// CHECK:STDOUT:   %T.loc156_4.1 => constants.%facet_value.dd9
// CHECK:STDOUT:   %pattern_type.loc157_16 => constants.%pattern_type.4aa
// CHECK:STDOUT:   %result.param_patt.loc157_16.2 => constants.%result.param_patt.1e1
// CHECK:STDOUT:   %result.patt.loc157_16.2 => constants.%result.patt.41f
// CHECK:STDOUT:   %T.as_type.loc157_29.1 => constants.%Int32
// CHECK:STDOUT:   %pattern_type.loc157_27 => constants.%pattern_type.8b4
// CHECK:STDOUT:   %x.param_patt.loc157_27.2 => constants.%x.param_patt.27c
// CHECK:STDOUT:   %x.patt.loc157_27.2 => constants.%x.patt.4c9
// CHECK:STDOUT:   %pattern_type.loc157_33 => constants.%pattern_type.4aa
// CHECK:STDOUT:   %y.param_patt.loc157_33.2 => constants.%y.param_patt.a08
// CHECK:STDOUT:   %y.patt.loc157_33.2 => constants.%y.patt.63b
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc157_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc157_27 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc157_33 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc159 => constants.%complete_type.f76
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.c76
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.loc159 => constants.%custom_witness.a8f
// CHECK:STDOUT:   %DivWith.facet => constants.%DivWith.facet.91e
// CHECK:STDOUT:   %DivWith.WithSelf.Op.type => constants.%DivWith.WithSelf.Op.type.dd6
// CHECK:STDOUT:   %.loc159_5.6 => constants.%.26f
// CHECK:STDOUT:   %impl.elem1.loc159_5.2 => constants.%Op.72717c.9
// CHECK:STDOUT:   %specific_impl_fn.loc159_5.3 => constants.%Op.72717c.9
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.c2c
// CHECK:STDOUT:   %.loc159_5.7 => constants.%.c2e
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.e7a31c.3
// CHECK:STDOUT:   %impl.elem0.loc159_5.2 => constants.%Int64.Op.4dca31.1
// CHECK:STDOUT:   %specific_impl_fn.loc159_5.4 => constants.%Int64.Op.4dca31.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ModWith.loc168(constants.%Destroy.facet.191, constants.%Destroy.facet.191, constants.%facet_value.ea3) {
// CHECK:STDOUT:   %Result.patt.loc165_9.2 => constants.%Result.patt
// CHECK:STDOUT:   %Result.loc165_9.1 => constants.%Destroy.facet.191
// CHECK:STDOUT:   %U.patt.loc166_4.2 => constants.%U.patt.7a2
// CHECK:STDOUT:   %U.loc166_4.1 => constants.%Destroy.facet.191
// CHECK:STDOUT:   %U.as_type.loc167_35.1 => constants.%Int64
// CHECK:STDOUT:   %ModWith.type.loc167_35.1 => constants.%ModWith.type.204
// CHECK:STDOUT:   %facet_type.loc167_19 => constants.%facet_type.f26
// CHECK:STDOUT:   %.Self.frozen.loc167_37.1 => constants.%.Self.frozen.ac6
// CHECK:STDOUT:   %require_complete.loc167 => constants.%complete_type.359
// CHECK:STDOUT:   %.Self.frozen.as_type => constants.%.Self.frozen.as_type.17a
// CHECK:STDOUT:   %ModWith.assoc_type => constants.%ModWith.assoc_type.1f7
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.0a6
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.loc167_43.1 => constants.%ModWith.lookup_impl_witness.072
// CHECK:STDOUT:   %impl.elem0.loc167_43.1 => constants.%impl.elem0.220f
// CHECK:STDOUT:   %Result.as_type.loc167_53.1 => constants.%Int64
// CHECK:STDOUT:   %.Self => constants.%.Self.494
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.loc167_43.2 => constants.%ModWith.lookup_impl_witness.103
// CHECK:STDOUT:   %impl.elem0.loc167_43.2 => constants.%impl.elem0.ffb
// CHECK:STDOUT:   %facet_type.loc167_37 => constants.%facet_type.096
// CHECK:STDOUT:   %pattern_type.loc167 => constants.%pattern_type.df3
// CHECK:STDOUT:   %T.patt.loc167_4.2 => constants.%T.patt.c87
// CHECK:STDOUT:   %T.loc167_4.1 => constants.%facet_value.ea3
// CHECK:STDOUT:   %pattern_type.loc168_16 => constants.%pattern_type.4aa
// CHECK:STDOUT:   %result.param_patt.loc168_16.2 => constants.%result.param_patt.1e1
// CHECK:STDOUT:   %result.patt.loc168_16.2 => constants.%result.patt.41f
// CHECK:STDOUT:   %T.as_type.loc168_29.1 => constants.%Int32
// CHECK:STDOUT:   %pattern_type.loc168_27 => constants.%pattern_type.8b4
// CHECK:STDOUT:   %x.param_patt.loc168_27.2 => constants.%x.param_patt.27c
// CHECK:STDOUT:   %x.patt.loc168_27.2 => constants.%x.patt.4c9
// CHECK:STDOUT:   %pattern_type.loc168_33 => constants.%pattern_type.4aa
// CHECK:STDOUT:   %y.param_patt.loc168_33.2 => constants.%y.param_patt.a08
// CHECK:STDOUT:   %y.patt.loc168_33.2 => constants.%y.patt.63b
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc168_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc168_27 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc168_33 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc170 => constants.%complete_type.9d2
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.ee5
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.loc170 => constants.%custom_witness.972
// CHECK:STDOUT:   %ModWith.facet => constants.%ModWith.facet.c38
// CHECK:STDOUT:   %ModWith.WithSelf.Op.type => constants.%ModWith.WithSelf.Op.type.663
// CHECK:STDOUT:   %.loc170_5.6 => constants.%.ed0
// CHECK:STDOUT:   %impl.elem1.loc170_5.2 => constants.%Op.72717c.10
// CHECK:STDOUT:   %specific_impl_fn.loc170_5.3 => constants.%Op.72717c.10
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.c2c
// CHECK:STDOUT:   %.loc170_5.7 => constants.%.c2e
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.e7a31c.3
// CHECK:STDOUT:   %impl.elem0.loc170_5.2 => constants.%Int64.Op.4dca31.1
// CHECK:STDOUT:   %specific_impl_fn.loc170_5.4 => constants.%Int64.Op.4dca31.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AddWith.loc124(constants.%Destroy.facet.191, constants.%Destroy.facet.099, constants.%facet_value.393) {
// CHECK:STDOUT:   %Result.patt.loc121_9.2 => constants.%Result.patt
// CHECK:STDOUT:   %Result.loc121_9.1 => constants.%Destroy.facet.191
// CHECK:STDOUT:   %U.patt.loc122_4.2 => constants.%U.patt.7a2
// CHECK:STDOUT:   %U.loc122_4.1 => constants.%Destroy.facet.099
// CHECK:STDOUT:   %U.as_type.loc123_35.1 => constants.%Int32
// CHECK:STDOUT:   %AddWith.type.loc123_35.1 => constants.%AddWith.type.c71
// CHECK:STDOUT:   %facet_type.loc123_19 => constants.%facet_type.698
// CHECK:STDOUT:   %.Self.frozen.loc123_37.1 => constants.%.Self.frozen.f1e
// CHECK:STDOUT:   %require_complete.loc123 => constants.%complete_type.224
// CHECK:STDOUT:   %.Self.frozen.as_type => constants.%.Self.frozen.as_type.440
// CHECK:STDOUT:   %AddWith.assoc_type => constants.%AddWith.assoc_type.72d
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.c2b
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.loc123_43.1 => constants.%AddWith.lookup_impl_witness.7f6
// CHECK:STDOUT:   %impl.elem0.loc123_43.1 => constants.%impl.elem0.7a4
// CHECK:STDOUT:   %Result.as_type.loc123_53.1 => constants.%Int64
// CHECK:STDOUT:   %.Self => constants.%.Self.591
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.loc123_43.2 => constants.%AddWith.lookup_impl_witness.47a
// CHECK:STDOUT:   %impl.elem0.loc123_43.2 => constants.%impl.elem0.6e6
// CHECK:STDOUT:   %facet_type.loc123_37 => constants.%facet_type.840
// CHECK:STDOUT:   %pattern_type.loc123 => constants.%pattern_type.46d
// CHECK:STDOUT:   %T.patt.loc123_4.2 => constants.%T.patt.211
// CHECK:STDOUT:   %T.loc123_4.1 => constants.%facet_value.393
// CHECK:STDOUT:   %pattern_type.loc124_16 => constants.%pattern_type.4aa
// CHECK:STDOUT:   %result.param_patt.loc124_16.2 => constants.%result.param_patt.1e1
// CHECK:STDOUT:   %result.patt.loc124_16.2 => constants.%result.patt.41f
// CHECK:STDOUT:   %T.as_type.loc124_29.1 => constants.%Int64
// CHECK:STDOUT:   %pattern_type.loc124_27 => constants.%pattern_type.4aa
// CHECK:STDOUT:   %x.param_patt.loc124_27.2 => constants.%x.param_patt.e68
// CHECK:STDOUT:   %x.patt.loc124_27.2 => constants.%x.patt.52a
// CHECK:STDOUT:   %pattern_type.loc124_33 => constants.%pattern_type.8b4
// CHECK:STDOUT:   %y.param_patt.loc124_33.2 => constants.%y.param_patt.e0c
// CHECK:STDOUT:   %y.patt.loc124_33.2 => constants.%y.patt.0a3
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc124_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc124_27 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc124_33 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc126 => constants.%complete_type.2cc
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.0fc
// CHECK:STDOUT:   %AddWith.lookup_impl_witness.loc126 => constants.%custom_witness.98f
// CHECK:STDOUT:   %AddWith.facet => constants.%AddWith.facet.3de
// CHECK:STDOUT:   %AddWith.WithSelf.Op.type => constants.%AddWith.WithSelf.Op.type.1c0
// CHECK:STDOUT:   %.loc126_5.6 => constants.%.df7
// CHECK:STDOUT:   %impl.elem1.loc126_5.2 => constants.%Op.72717c.11
// CHECK:STDOUT:   %specific_impl_fn.loc126_5.3 => constants.%Op.72717c.11
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.c2c
// CHECK:STDOUT:   %.loc126_5.7 => constants.%.c2e
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.e7a31c.3
// CHECK:STDOUT:   %impl.elem0.loc126_5.2 => constants.%Int64.Op.4dca31.1
// CHECK:STDOUT:   %specific_impl_fn.loc126_5.4 => constants.%Int64.Op.4dca31.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @SubWith.loc135(constants.%Destroy.facet.191, constants.%Destroy.facet.099, constants.%facet_value.e02) {
// CHECK:STDOUT:   %Result.patt.loc132_9.2 => constants.%Result.patt
// CHECK:STDOUT:   %Result.loc132_9.1 => constants.%Destroy.facet.191
// CHECK:STDOUT:   %U.patt.loc133_4.2 => constants.%U.patt.7a2
// CHECK:STDOUT:   %U.loc133_4.1 => constants.%Destroy.facet.099
// CHECK:STDOUT:   %U.as_type.loc134_35.1 => constants.%Int32
// CHECK:STDOUT:   %SubWith.type.loc134_35.1 => constants.%SubWith.type.c81
// CHECK:STDOUT:   %facet_type.loc134_19 => constants.%facet_type.44e
// CHECK:STDOUT:   %.Self.frozen.loc134_37.1 => constants.%.Self.frozen.db8
// CHECK:STDOUT:   %require_complete.loc134 => constants.%complete_type.63f
// CHECK:STDOUT:   %.Self.frozen.as_type => constants.%.Self.frozen.as_type.068
// CHECK:STDOUT:   %SubWith.assoc_type => constants.%SubWith.assoc_type.68f
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.f11
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.loc134_43.1 => constants.%SubWith.lookup_impl_witness.32d
// CHECK:STDOUT:   %impl.elem0.loc134_43.1 => constants.%impl.elem0.821
// CHECK:STDOUT:   %Result.as_type.loc134_53.1 => constants.%Int64
// CHECK:STDOUT:   %.Self => constants.%.Self.8ac
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.loc134_43.2 => constants.%SubWith.lookup_impl_witness.f41
// CHECK:STDOUT:   %impl.elem0.loc134_43.2 => constants.%impl.elem0.e31
// CHECK:STDOUT:   %facet_type.loc134_37 => constants.%facet_type.99d
// CHECK:STDOUT:   %pattern_type.loc134 => constants.%pattern_type.b19
// CHECK:STDOUT:   %T.patt.loc134_4.2 => constants.%T.patt.e50
// CHECK:STDOUT:   %T.loc134_4.1 => constants.%facet_value.e02
// CHECK:STDOUT:   %pattern_type.loc135_16 => constants.%pattern_type.4aa
// CHECK:STDOUT:   %result.param_patt.loc135_16.2 => constants.%result.param_patt.1e1
// CHECK:STDOUT:   %result.patt.loc135_16.2 => constants.%result.patt.41f
// CHECK:STDOUT:   %T.as_type.loc135_29.1 => constants.%Int64
// CHECK:STDOUT:   %pattern_type.loc135_27 => constants.%pattern_type.4aa
// CHECK:STDOUT:   %x.param_patt.loc135_27.2 => constants.%x.param_patt.e68
// CHECK:STDOUT:   %x.patt.loc135_27.2 => constants.%x.patt.52a
// CHECK:STDOUT:   %pattern_type.loc135_33 => constants.%pattern_type.8b4
// CHECK:STDOUT:   %y.param_patt.loc135_33.2 => constants.%y.param_patt.e0c
// CHECK:STDOUT:   %y.patt.loc135_33.2 => constants.%y.patt.0a3
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc135_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc135_27 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc135_33 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc137 => constants.%complete_type.cf9
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.799
// CHECK:STDOUT:   %SubWith.lookup_impl_witness.loc137 => constants.%custom_witness.5c3
// CHECK:STDOUT:   %SubWith.facet => constants.%SubWith.facet.320
// CHECK:STDOUT:   %SubWith.WithSelf.Op.type => constants.%SubWith.WithSelf.Op.type.8fd
// CHECK:STDOUT:   %.loc137_5.6 => constants.%.ad9
// CHECK:STDOUT:   %impl.elem1.loc137_5.2 => constants.%Op.72717c.12
// CHECK:STDOUT:   %specific_impl_fn.loc137_5.3 => constants.%Op.72717c.12
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.c2c
// CHECK:STDOUT:   %.loc137_5.7 => constants.%.c2e
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.e7a31c.3
// CHECK:STDOUT:   %impl.elem0.loc137_5.2 => constants.%Int64.Op.4dca31.1
// CHECK:STDOUT:   %specific_impl_fn.loc137_5.4 => constants.%Int64.Op.4dca31.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @MulWith.loc146(constants.%Destroy.facet.191, constants.%Destroy.facet.099, constants.%facet_value.dda) {
// CHECK:STDOUT:   %Result.patt.loc143_9.2 => constants.%Result.patt
// CHECK:STDOUT:   %Result.loc143_9.1 => constants.%Destroy.facet.191
// CHECK:STDOUT:   %U.patt.loc144_4.2 => constants.%U.patt.7a2
// CHECK:STDOUT:   %U.loc144_4.1 => constants.%Destroy.facet.099
// CHECK:STDOUT:   %U.as_type.loc145_35.1 => constants.%Int32
// CHECK:STDOUT:   %MulWith.type.loc145_35.1 => constants.%MulWith.type.eeb
// CHECK:STDOUT:   %facet_type.loc145_19 => constants.%facet_type.ff6
// CHECK:STDOUT:   %.Self.frozen.loc145_37.1 => constants.%.Self.frozen.be6
// CHECK:STDOUT:   %require_complete.loc145 => constants.%complete_type.3fe
// CHECK:STDOUT:   %.Self.frozen.as_type => constants.%.Self.frozen.as_type.aaf
// CHECK:STDOUT:   %MulWith.assoc_type => constants.%MulWith.assoc_type.083
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.fc0
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.loc145_43.1 => constants.%MulWith.lookup_impl_witness.4cf
// CHECK:STDOUT:   %impl.elem0.loc145_43.1 => constants.%impl.elem0.167
// CHECK:STDOUT:   %Result.as_type.loc145_53.1 => constants.%Int64
// CHECK:STDOUT:   %.Self => constants.%.Self.d4f
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.loc145_43.2 => constants.%MulWith.lookup_impl_witness.07b
// CHECK:STDOUT:   %impl.elem0.loc145_43.2 => constants.%impl.elem0.0ea
// CHECK:STDOUT:   %facet_type.loc145_37 => constants.%facet_type.5e8
// CHECK:STDOUT:   %pattern_type.loc145 => constants.%pattern_type.d24
// CHECK:STDOUT:   %T.patt.loc145_4.2 => constants.%T.patt.54a
// CHECK:STDOUT:   %T.loc145_4.1 => constants.%facet_value.dda
// CHECK:STDOUT:   %pattern_type.loc146_16 => constants.%pattern_type.4aa
// CHECK:STDOUT:   %result.param_patt.loc146_16.2 => constants.%result.param_patt.1e1
// CHECK:STDOUT:   %result.patt.loc146_16.2 => constants.%result.patt.41f
// CHECK:STDOUT:   %T.as_type.loc146_29.1 => constants.%Int64
// CHECK:STDOUT:   %pattern_type.loc146_27 => constants.%pattern_type.4aa
// CHECK:STDOUT:   %x.param_patt.loc146_27.2 => constants.%x.param_patt.e68
// CHECK:STDOUT:   %x.patt.loc146_27.2 => constants.%x.patt.52a
// CHECK:STDOUT:   %pattern_type.loc146_33 => constants.%pattern_type.8b4
// CHECK:STDOUT:   %y.param_patt.loc146_33.2 => constants.%y.param_patt.e0c
// CHECK:STDOUT:   %y.patt.loc146_33.2 => constants.%y.patt.0a3
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc146_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc146_27 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc146_33 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc148 => constants.%complete_type.e76
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.019
// CHECK:STDOUT:   %MulWith.lookup_impl_witness.loc148 => constants.%custom_witness.ac0
// CHECK:STDOUT:   %MulWith.facet => constants.%MulWith.facet.d72
// CHECK:STDOUT:   %MulWith.WithSelf.Op.type => constants.%MulWith.WithSelf.Op.type.10e
// CHECK:STDOUT:   %.loc148_5.6 => constants.%.be6
// CHECK:STDOUT:   %impl.elem1.loc148_5.2 => constants.%Op.72717c.13
// CHECK:STDOUT:   %specific_impl_fn.loc148_5.3 => constants.%Op.72717c.13
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.c2c
// CHECK:STDOUT:   %.loc148_5.7 => constants.%.c2e
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.e7a31c.3
// CHECK:STDOUT:   %impl.elem0.loc148_5.2 => constants.%Int64.Op.4dca31.1
// CHECK:STDOUT:   %specific_impl_fn.loc148_5.4 => constants.%Int64.Op.4dca31.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @DivWith.loc157(constants.%Destroy.facet.191, constants.%Destroy.facet.099, constants.%facet_value.886) {
// CHECK:STDOUT:   %Result.patt.loc154_9.2 => constants.%Result.patt
// CHECK:STDOUT:   %Result.loc154_9.1 => constants.%Destroy.facet.191
// CHECK:STDOUT:   %U.patt.loc155_4.2 => constants.%U.patt.7a2
// CHECK:STDOUT:   %U.loc155_4.1 => constants.%Destroy.facet.099
// CHECK:STDOUT:   %U.as_type.loc156_35.1 => constants.%Int32
// CHECK:STDOUT:   %DivWith.type.loc156_35.1 => constants.%DivWith.type.fb1
// CHECK:STDOUT:   %facet_type.loc156_19 => constants.%facet_type.d41
// CHECK:STDOUT:   %.Self.frozen.loc156_37.1 => constants.%.Self.frozen.be1
// CHECK:STDOUT:   %require_complete.loc156 => constants.%complete_type.11b
// CHECK:STDOUT:   %.Self.frozen.as_type => constants.%.Self.frozen.as_type.88a
// CHECK:STDOUT:   %DivWith.assoc_type => constants.%DivWith.assoc_type.725
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.2da
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.loc156_43.1 => constants.%DivWith.lookup_impl_witness.385
// CHECK:STDOUT:   %impl.elem0.loc156_43.1 => constants.%impl.elem0.b42
// CHECK:STDOUT:   %Result.as_type.loc156_53.1 => constants.%Int64
// CHECK:STDOUT:   %.Self => constants.%.Self.13f
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.loc156_43.2 => constants.%DivWith.lookup_impl_witness.abd
// CHECK:STDOUT:   %impl.elem0.loc156_43.2 => constants.%impl.elem0.1a6
// CHECK:STDOUT:   %facet_type.loc156_37 => constants.%facet_type.97b
// CHECK:STDOUT:   %pattern_type.loc156 => constants.%pattern_type.6b7
// CHECK:STDOUT:   %T.patt.loc156_4.2 => constants.%T.patt.0b0
// CHECK:STDOUT:   %T.loc156_4.1 => constants.%facet_value.886
// CHECK:STDOUT:   %pattern_type.loc157_16 => constants.%pattern_type.4aa
// CHECK:STDOUT:   %result.param_patt.loc157_16.2 => constants.%result.param_patt.1e1
// CHECK:STDOUT:   %result.patt.loc157_16.2 => constants.%result.patt.41f
// CHECK:STDOUT:   %T.as_type.loc157_29.1 => constants.%Int64
// CHECK:STDOUT:   %pattern_type.loc157_27 => constants.%pattern_type.4aa
// CHECK:STDOUT:   %x.param_patt.loc157_27.2 => constants.%x.param_patt.e68
// CHECK:STDOUT:   %x.patt.loc157_27.2 => constants.%x.patt.52a
// CHECK:STDOUT:   %pattern_type.loc157_33 => constants.%pattern_type.8b4
// CHECK:STDOUT:   %y.param_patt.loc157_33.2 => constants.%y.param_patt.e0c
// CHECK:STDOUT:   %y.patt.loc157_33.2 => constants.%y.patt.0a3
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc157_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc157_27 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc157_33 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc159 => constants.%complete_type.595
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.580
// CHECK:STDOUT:   %DivWith.lookup_impl_witness.loc159 => constants.%custom_witness.c98
// CHECK:STDOUT:   %DivWith.facet => constants.%DivWith.facet.4d9
// CHECK:STDOUT:   %DivWith.WithSelf.Op.type => constants.%DivWith.WithSelf.Op.type.354
// CHECK:STDOUT:   %.loc159_5.6 => constants.%.436
// CHECK:STDOUT:   %impl.elem1.loc159_5.2 => constants.%Op.72717c.14
// CHECK:STDOUT:   %specific_impl_fn.loc159_5.3 => constants.%Op.72717c.14
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.c2c
// CHECK:STDOUT:   %.loc159_5.7 => constants.%.c2e
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.e7a31c.3
// CHECK:STDOUT:   %impl.elem0.loc159_5.2 => constants.%Int64.Op.4dca31.1
// CHECK:STDOUT:   %specific_impl_fn.loc159_5.4 => constants.%Int64.Op.4dca31.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ModWith.loc168(constants.%Destroy.facet.191, constants.%Destroy.facet.099, constants.%facet_value.20a) {
// CHECK:STDOUT:   %Result.patt.loc165_9.2 => constants.%Result.patt
// CHECK:STDOUT:   %Result.loc165_9.1 => constants.%Destroy.facet.191
// CHECK:STDOUT:   %U.patt.loc166_4.2 => constants.%U.patt.7a2
// CHECK:STDOUT:   %U.loc166_4.1 => constants.%Destroy.facet.099
// CHECK:STDOUT:   %U.as_type.loc167_35.1 => constants.%Int32
// CHECK:STDOUT:   %ModWith.type.loc167_35.1 => constants.%ModWith.type.253
// CHECK:STDOUT:   %facet_type.loc167_19 => constants.%facet_type.ea8
// CHECK:STDOUT:   %.Self.frozen.loc167_37.1 => constants.%.Self.frozen.4b6
// CHECK:STDOUT:   %require_complete.loc167 => constants.%complete_type.2aa
// CHECK:STDOUT:   %.Self.frozen.as_type => constants.%.Self.frozen.as_type.080
// CHECK:STDOUT:   %ModWith.assoc_type => constants.%ModWith.assoc_type.47d
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.45c
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.loc167_43.1 => constants.%ModWith.lookup_impl_witness.fe2
// CHECK:STDOUT:   %impl.elem0.loc167_43.1 => constants.%impl.elem0.e60
// CHECK:STDOUT:   %Result.as_type.loc167_53.1 => constants.%Int64
// CHECK:STDOUT:   %.Self => constants.%.Self.008
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.loc167_43.2 => constants.%ModWith.lookup_impl_witness.557
// CHECK:STDOUT:   %impl.elem0.loc167_43.2 => constants.%impl.elem0.cdb
// CHECK:STDOUT:   %facet_type.loc167_37 => constants.%facet_type.546
// CHECK:STDOUT:   %pattern_type.loc167 => constants.%pattern_type.5db
// CHECK:STDOUT:   %T.patt.loc167_4.2 => constants.%T.patt.ed0
// CHECK:STDOUT:   %T.loc167_4.1 => constants.%facet_value.20a
// CHECK:STDOUT:   %pattern_type.loc168_16 => constants.%pattern_type.4aa
// CHECK:STDOUT:   %result.param_patt.loc168_16.2 => constants.%result.param_patt.1e1
// CHECK:STDOUT:   %result.patt.loc168_16.2 => constants.%result.patt.41f
// CHECK:STDOUT:   %T.as_type.loc168_29.1 => constants.%Int64
// CHECK:STDOUT:   %pattern_type.loc168_27 => constants.%pattern_type.4aa
// CHECK:STDOUT:   %x.param_patt.loc168_27.2 => constants.%x.param_patt.e68
// CHECK:STDOUT:   %x.patt.loc168_27.2 => constants.%x.patt.52a
// CHECK:STDOUT:   %pattern_type.loc168_33 => constants.%pattern_type.8b4
// CHECK:STDOUT:   %y.param_patt.loc168_33.2 => constants.%y.param_patt.e0c
// CHECK:STDOUT:   %y.patt.loc168_33.2 => constants.%y.patt.0a3
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc168_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc168_27 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc168_33 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc170 => constants.%complete_type.80f
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.878
// CHECK:STDOUT:   %ModWith.lookup_impl_witness.loc170 => constants.%custom_witness.e66
// CHECK:STDOUT:   %ModWith.facet => constants.%ModWith.facet.ca3
// CHECK:STDOUT:   %ModWith.WithSelf.Op.type => constants.%ModWith.WithSelf.Op.type.dcc
// CHECK:STDOUT:   %.loc170_5.6 => constants.%.66e6
// CHECK:STDOUT:   %impl.elem1.loc170_5.2 => constants.%Op.72717c.15
// CHECK:STDOUT:   %specific_impl_fn.loc170_5.3 => constants.%Op.72717c.15
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.c2c
// CHECK:STDOUT:   %.loc170_5.7 => constants.%.c2e
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.e7a31c.3
// CHECK:STDOUT:   %impl.elem0.loc170_5.2 => constants.%Int64.Op.4dca31.1
// CHECK:STDOUT:   %specific_impl_fn.loc170_5.4 => constants.%Int64.Op.4dca31.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AddAssignWith.loc225(constants.%U.67d, constants.%T.792) {
// CHECK:STDOUT:   %U.patt.loc225_19.2 => constants.%U.patt.d47
// CHECK:STDOUT:   %U.loc225_19.1 => constants.%U.67d
// CHECK:STDOUT:   %AddAssignWith.type.loc225_50.1 => constants.%AddAssignWith.type.2c8890.2
// CHECK:STDOUT:   %pattern_type.loc225_28 => constants.%pattern_type.82d
// CHECK:STDOUT:   %T.patt.loc225_28.2 => constants.%T.patt.a22
// CHECK:STDOUT:   %T.loc225_28.1 => constants.%T.792
// CHECK:STDOUT:   %T.as_type.loc225_60.1 => constants.%T.as_type.50c
// CHECK:STDOUT:   %pattern_type.loc225_58 => constants.%pattern_type.643306.2
// CHECK:STDOUT:   %x.patt.loc225_58.2 => constants.%x.patt.227
// CHECK:STDOUT:   %x.param_patt.loc225_53.2 => constants.%x.param_patt.343
// CHECK:STDOUT:   %pattern_type.loc225_64 => constants.%pattern_type.51d1c4.4
// CHECK:STDOUT:   %y.param_patt.loc225_64.2 => constants.%y.param_patt.be3
// CHECK:STDOUT:   %y.patt.loc225_64.2 => constants.%y.patt.b85
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @SubAssignWith.loc231(constants.%U.67d, constants.%T.f6c) {
// CHECK:STDOUT:   %U.patt.loc231_19.2 => constants.%U.patt.d47
// CHECK:STDOUT:   %U.loc231_19.1 => constants.%U.67d
// CHECK:STDOUT:   %SubAssignWith.type.loc231_50.1 => constants.%SubAssignWith.type.7e53cf.2
// CHECK:STDOUT:   %pattern_type.loc231_28 => constants.%pattern_type.260
// CHECK:STDOUT:   %T.patt.loc231_28.2 => constants.%T.patt.f45
// CHECK:STDOUT:   %T.loc231_28.1 => constants.%T.f6c
// CHECK:STDOUT:   %T.as_type.loc231_60.1 => constants.%T.as_type.3ff
// CHECK:STDOUT:   %pattern_type.loc231_58 => constants.%pattern_type.f08ed5.2
// CHECK:STDOUT:   %x.patt.loc231_58.2 => constants.%x.patt.95a
// CHECK:STDOUT:   %x.param_patt.loc231_53.2 => constants.%x.param_patt.a45
// CHECK:STDOUT:   %pattern_type.loc231_64 => constants.%pattern_type.51d1c4.4
// CHECK:STDOUT:   %y.param_patt.loc231_64.2 => constants.%y.param_patt.be3
// CHECK:STDOUT:   %y.patt.loc231_64.2 => constants.%y.patt.b85
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @MulAssignWith.loc237(constants.%U.67d, constants.%T.64d) {
// CHECK:STDOUT:   %U.patt.loc237_19.2 => constants.%U.patt.d47
// CHECK:STDOUT:   %U.loc237_19.1 => constants.%U.67d
// CHECK:STDOUT:   %MulAssignWith.type.loc237_50.1 => constants.%MulAssignWith.type.3a0e4d.2
// CHECK:STDOUT:   %pattern_type.loc237_28 => constants.%pattern_type.8c2
// CHECK:STDOUT:   %T.patt.loc237_28.2 => constants.%T.patt.db5
// CHECK:STDOUT:   %T.loc237_28.1 => constants.%T.64d
// CHECK:STDOUT:   %T.as_type.loc237_60.1 => constants.%T.as_type.1de
// CHECK:STDOUT:   %pattern_type.loc237_58 => constants.%pattern_type.8bf366.2
// CHECK:STDOUT:   %x.patt.loc237_58.2 => constants.%x.patt.4e9
// CHECK:STDOUT:   %x.param_patt.loc237_53.2 => constants.%x.param_patt.6c8
// CHECK:STDOUT:   %pattern_type.loc237_64 => constants.%pattern_type.51d1c4.4
// CHECK:STDOUT:   %y.param_patt.loc237_64.2 => constants.%y.param_patt.be3
// CHECK:STDOUT:   %y.patt.loc237_64.2 => constants.%y.patt.b85
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @DivAssignWith.loc243(constants.%U.67d, constants.%T.c33) {
// CHECK:STDOUT:   %U.patt.loc243_19.2 => constants.%U.patt.d47
// CHECK:STDOUT:   %U.loc243_19.1 => constants.%U.67d
// CHECK:STDOUT:   %DivAssignWith.type.loc243_50.1 => constants.%DivAssignWith.type.7e58ca.2
// CHECK:STDOUT:   %pattern_type.loc243_28 => constants.%pattern_type.f74
// CHECK:STDOUT:   %T.patt.loc243_28.2 => constants.%T.patt.9a5
// CHECK:STDOUT:   %T.loc243_28.1 => constants.%T.c33
// CHECK:STDOUT:   %T.as_type.loc243_60.1 => constants.%T.as_type.e57
// CHECK:STDOUT:   %pattern_type.loc243_58 => constants.%pattern_type.da303b.2
// CHECK:STDOUT:   %x.patt.loc243_58.2 => constants.%x.patt.cf6
// CHECK:STDOUT:   %x.param_patt.loc243_53.2 => constants.%x.param_patt.660
// CHECK:STDOUT:   %pattern_type.loc243_64 => constants.%pattern_type.51d1c4.4
// CHECK:STDOUT:   %y.param_patt.loc243_64.2 => constants.%y.param_patt.be3
// CHECK:STDOUT:   %y.patt.loc243_64.2 => constants.%y.patt.b85
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ModAssignWith.loc249(constants.%U.67d, constants.%T.bca) {
// CHECK:STDOUT:   %U.patt.loc249_19.2 => constants.%U.patt.d47
// CHECK:STDOUT:   %U.loc249_19.1 => constants.%U.67d
// CHECK:STDOUT:   %ModAssignWith.type.loc249_50.1 => constants.%ModAssignWith.type.778d68.2
// CHECK:STDOUT:   %pattern_type.loc249_28 => constants.%pattern_type.fb8
// CHECK:STDOUT:   %T.patt.loc249_28.2 => constants.%T.patt.c29
// CHECK:STDOUT:   %T.loc249_28.1 => constants.%T.bca
// CHECK:STDOUT:   %T.as_type.loc249_60.1 => constants.%T.as_type.ce6
// CHECK:STDOUT:   %pattern_type.loc249_58 => constants.%pattern_type.1d5f30.2
// CHECK:STDOUT:   %x.patt.loc249_58.2 => constants.%x.patt.fa4
// CHECK:STDOUT:   %x.param_patt.loc249_53.2 => constants.%x.param_patt.749
// CHECK:STDOUT:   %pattern_type.loc249_64 => constants.%pattern_type.51d1c4.4
// CHECK:STDOUT:   %y.param_patt.loc249_64.2 => constants.%y.param_patt.be3
// CHECK:STDOUT:   %y.patt.loc249_64.2 => constants.%y.patt.b85
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AddAssignWith.loc225(constants.%i16, constants.%AddAssignWith.facet.e34) {
// CHECK:STDOUT:   %U.patt.loc225_19.2 => constants.%U.patt.d47
// CHECK:STDOUT:   %U.loc225_19.1 => constants.%i16
// CHECK:STDOUT:   %AddAssignWith.type.loc225_50.1 => constants.%AddAssignWith.type.2a0
// CHECK:STDOUT:   %pattern_type.loc225_28 => constants.%pattern_type.b256
// CHECK:STDOUT:   %T.patt.loc225_28.2 => constants.%T.patt.d2a
// CHECK:STDOUT:   %T.loc225_28.1 => constants.%AddAssignWith.facet.e34
// CHECK:STDOUT:   %T.as_type.loc225_60.1 => constants.%Int16
// CHECK:STDOUT:   %pattern_type.loc225_58 => constants.%pattern_type.785
// CHECK:STDOUT:   %x.patt.loc225_58.2 => constants.%x.patt.49e
// CHECK:STDOUT:   %x.param_patt.loc225_53.2 => constants.%x.param_patt.69a
// CHECK:STDOUT:   %pattern_type.loc225_64 => constants.%pattern_type.203
// CHECK:STDOUT:   %y.param_patt.loc225_64.2 => constants.%y.param_patt.4b4
// CHECK:STDOUT:   %y.patt.loc225_64.2 => constants.%y.patt.dd3
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc225_53 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc225_64 => constants.%complete_type.2da
// CHECK:STDOUT:   %require_complete.loc227 => constants.%complete_type.a21
// CHECK:STDOUT:   %AddAssignWith.assoc_type => constants.%AddAssignWith.assoc_type.1bb
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.24b
// CHECK:STDOUT:   %AddAssignWith.WithSelf.Op.type => constants.%AddAssignWith.WithSelf.Op.type.eb5
// CHECK:STDOUT:   %.loc227_5.2 => constants.%.063
// CHECK:STDOUT:   %AddAssignWith.lookup_impl_witness => constants.%custom_witness.6a0
// CHECK:STDOUT:   %impl.elem0.loc227_5.2 => constants.%Int16.Op.4dca31.13
// CHECK:STDOUT:   %specific_impl_fn.loc227_5.2 => constants.%Int16.Op.4dca31.13
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @SubAssignWith.loc231(constants.%i16, constants.%SubAssignWith.facet.a52) {
// CHECK:STDOUT:   %U.patt.loc231_19.2 => constants.%U.patt.d47
// CHECK:STDOUT:   %U.loc231_19.1 => constants.%i16
// CHECK:STDOUT:   %SubAssignWith.type.loc231_50.1 => constants.%SubAssignWith.type.e8b
// CHECK:STDOUT:   %pattern_type.loc231_28 => constants.%pattern_type.c22
// CHECK:STDOUT:   %T.patt.loc231_28.2 => constants.%T.patt.0f1
// CHECK:STDOUT:   %T.loc231_28.1 => constants.%SubAssignWith.facet.a52
// CHECK:STDOUT:   %T.as_type.loc231_60.1 => constants.%Int16
// CHECK:STDOUT:   %pattern_type.loc231_58 => constants.%pattern_type.785
// CHECK:STDOUT:   %x.patt.loc231_58.2 => constants.%x.patt.49e
// CHECK:STDOUT:   %x.param_patt.loc231_53.2 => constants.%x.param_patt.69a
// CHECK:STDOUT:   %pattern_type.loc231_64 => constants.%pattern_type.203
// CHECK:STDOUT:   %y.param_patt.loc231_64.2 => constants.%y.param_patt.4b4
// CHECK:STDOUT:   %y.patt.loc231_64.2 => constants.%y.patt.dd3
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc231_53 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc231_64 => constants.%complete_type.2da
// CHECK:STDOUT:   %require_complete.loc233 => constants.%complete_type.002
// CHECK:STDOUT:   %SubAssignWith.assoc_type => constants.%SubAssignWith.assoc_type.e13
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.47c
// CHECK:STDOUT:   %SubAssignWith.WithSelf.Op.type => constants.%SubAssignWith.WithSelf.Op.type.163
// CHECK:STDOUT:   %.loc233_5.2 => constants.%.dc4
// CHECK:STDOUT:   %SubAssignWith.lookup_impl_witness => constants.%custom_witness.722
// CHECK:STDOUT:   %impl.elem0.loc233_5.2 => constants.%Int16.Op.4dca31.14
// CHECK:STDOUT:   %specific_impl_fn.loc233_5.2 => constants.%Int16.Op.4dca31.14
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @MulAssignWith.loc237(constants.%i16, constants.%MulAssignWith.facet.201) {
// CHECK:STDOUT:   %U.patt.loc237_19.2 => constants.%U.patt.d47
// CHECK:STDOUT:   %U.loc237_19.1 => constants.%i16
// CHECK:STDOUT:   %MulAssignWith.type.loc237_50.1 => constants.%MulAssignWith.type.7d4
// CHECK:STDOUT:   %pattern_type.loc237_28 => constants.%pattern_type.a24
// CHECK:STDOUT:   %T.patt.loc237_28.2 => constants.%T.patt.f71
// CHECK:STDOUT:   %T.loc237_28.1 => constants.%MulAssignWith.facet.201
// CHECK:STDOUT:   %T.as_type.loc237_60.1 => constants.%Int16
// CHECK:STDOUT:   %pattern_type.loc237_58 => constants.%pattern_type.785
// CHECK:STDOUT:   %x.patt.loc237_58.2 => constants.%x.patt.49e
// CHECK:STDOUT:   %x.param_patt.loc237_53.2 => constants.%x.param_patt.69a
// CHECK:STDOUT:   %pattern_type.loc237_64 => constants.%pattern_type.203
// CHECK:STDOUT:   %y.param_patt.loc237_64.2 => constants.%y.param_patt.4b4
// CHECK:STDOUT:   %y.patt.loc237_64.2 => constants.%y.patt.dd3
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc237_53 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc237_64 => constants.%complete_type.2da
// CHECK:STDOUT:   %require_complete.loc239 => constants.%complete_type.60a
// CHECK:STDOUT:   %MulAssignWith.assoc_type => constants.%MulAssignWith.assoc_type.aa8
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.137
// CHECK:STDOUT:   %MulAssignWith.WithSelf.Op.type => constants.%MulAssignWith.WithSelf.Op.type.170
// CHECK:STDOUT:   %.loc239_5.2 => constants.%.69d
// CHECK:STDOUT:   %MulAssignWith.lookup_impl_witness => constants.%custom_witness.c67
// CHECK:STDOUT:   %impl.elem0.loc239_5.2 => constants.%Int16.Op.4dca31.15
// CHECK:STDOUT:   %specific_impl_fn.loc239_5.2 => constants.%Int16.Op.4dca31.15
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @DivAssignWith.loc243(constants.%i16, constants.%DivAssignWith.facet.a72) {
// CHECK:STDOUT:   %U.patt.loc243_19.2 => constants.%U.patt.d47
// CHECK:STDOUT:   %U.loc243_19.1 => constants.%i16
// CHECK:STDOUT:   %DivAssignWith.type.loc243_50.1 => constants.%DivAssignWith.type.081
// CHECK:STDOUT:   %pattern_type.loc243_28 => constants.%pattern_type.c970
// CHECK:STDOUT:   %T.patt.loc243_28.2 => constants.%T.patt.35e
// CHECK:STDOUT:   %T.loc243_28.1 => constants.%DivAssignWith.facet.a72
// CHECK:STDOUT:   %T.as_type.loc243_60.1 => constants.%Int16
// CHECK:STDOUT:   %pattern_type.loc243_58 => constants.%pattern_type.785
// CHECK:STDOUT:   %x.patt.loc243_58.2 => constants.%x.patt.49e
// CHECK:STDOUT:   %x.param_patt.loc243_53.2 => constants.%x.param_patt.69a
// CHECK:STDOUT:   %pattern_type.loc243_64 => constants.%pattern_type.203
// CHECK:STDOUT:   %y.param_patt.loc243_64.2 => constants.%y.param_patt.4b4
// CHECK:STDOUT:   %y.patt.loc243_64.2 => constants.%y.patt.dd3
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc243_53 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc243_64 => constants.%complete_type.2da
// CHECK:STDOUT:   %require_complete.loc245 => constants.%complete_type.531
// CHECK:STDOUT:   %DivAssignWith.assoc_type => constants.%DivAssignWith.assoc_type.791
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.a3b
// CHECK:STDOUT:   %DivAssignWith.WithSelf.Op.type => constants.%DivAssignWith.WithSelf.Op.type.d5b
// CHECK:STDOUT:   %.loc245_5.2 => constants.%.8d8
// CHECK:STDOUT:   %DivAssignWith.lookup_impl_witness => constants.%custom_witness.91d
// CHECK:STDOUT:   %impl.elem0.loc245_5.2 => constants.%Int16.Op.4dca31.16
// CHECK:STDOUT:   %specific_impl_fn.loc245_5.2 => constants.%Int16.Op.4dca31.16
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ModAssignWith.loc249(constants.%i16, constants.%ModAssignWith.facet.4a2) {
// CHECK:STDOUT:   %U.patt.loc249_19.2 => constants.%U.patt.d47
// CHECK:STDOUT:   %U.loc249_19.1 => constants.%i16
// CHECK:STDOUT:   %ModAssignWith.type.loc249_50.1 => constants.%ModAssignWith.type.1ea
// CHECK:STDOUT:   %pattern_type.loc249_28 => constants.%pattern_type.6ef
// CHECK:STDOUT:   %T.patt.loc249_28.2 => constants.%T.patt.dbe
// CHECK:STDOUT:   %T.loc249_28.1 => constants.%ModAssignWith.facet.4a2
// CHECK:STDOUT:   %T.as_type.loc249_60.1 => constants.%Int16
// CHECK:STDOUT:   %pattern_type.loc249_58 => constants.%pattern_type.785
// CHECK:STDOUT:   %x.patt.loc249_58.2 => constants.%x.patt.49e
// CHECK:STDOUT:   %x.param_patt.loc249_53.2 => constants.%x.param_patt.69a
// CHECK:STDOUT:   %pattern_type.loc249_64 => constants.%pattern_type.203
// CHECK:STDOUT:   %y.param_patt.loc249_64.2 => constants.%y.param_patt.4b4
// CHECK:STDOUT:   %y.patt.loc249_64.2 => constants.%y.patt.dd3
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc249_53 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc249_64 => constants.%complete_type.2da
// CHECK:STDOUT:   %require_complete.loc251 => constants.%complete_type.450
// CHECK:STDOUT:   %ModAssignWith.assoc_type => constants.%ModAssignWith.assoc_type.91d
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.b7f
// CHECK:STDOUT:   %ModAssignWith.WithSelf.Op.type => constants.%ModAssignWith.WithSelf.Op.type.374
// CHECK:STDOUT:   %.loc251_5.2 => constants.%.eca
// CHECK:STDOUT:   %ModAssignWith.lookup_impl_witness => constants.%custom_witness.17b
// CHECK:STDOUT:   %impl.elem0.loc251_5.2 => constants.%Int16.Op.4dca31.17
// CHECK:STDOUT:   %specific_impl_fn.loc251_5.2 => constants.%Int16.Op.4dca31.17
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AddAssignWith.loc225(constants.%Int32, constants.%AddAssignWith.facet.348) {
// CHECK:STDOUT:   %U.patt.loc225_19.2 => constants.%U.patt.d47
// CHECK:STDOUT:   %U.loc225_19.1 => constants.%Int32
// CHECK:STDOUT:   %AddAssignWith.type.loc225_50.1 => constants.%AddAssignWith.type.23a
// CHECK:STDOUT:   %pattern_type.loc225_28 => constants.%pattern_type.5ad
// CHECK:STDOUT:   %T.patt.loc225_28.2 => constants.%T.patt.96c
// CHECK:STDOUT:   %T.loc225_28.1 => constants.%AddAssignWith.facet.348
// CHECK:STDOUT:   %T.as_type.loc225_60.1 => constants.%Int32
// CHECK:STDOUT:   %pattern_type.loc225_58 => constants.%pattern_type.8b4
// CHECK:STDOUT:   %x.patt.loc225_58.2 => constants.%x.patt.d79
// CHECK:STDOUT:   %x.param_patt.loc225_53.2 => constants.%x.param_patt.0c7
// CHECK:STDOUT:   %pattern_type.loc225_64 => constants.%pattern_type.8b4
// CHECK:STDOUT:   %y.param_patt.loc225_64.2 => constants.%y.param_patt.e0c
// CHECK:STDOUT:   %y.patt.loc225_64.2 => constants.%y.patt.0a3
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc225_53 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc225_64 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc227 => constants.%complete_type.e1b
// CHECK:STDOUT:   %AddAssignWith.assoc_type => constants.%AddAssignWith.assoc_type.4c7
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.114
// CHECK:STDOUT:   %AddAssignWith.WithSelf.Op.type => constants.%AddAssignWith.WithSelf.Op.type.96d
// CHECK:STDOUT:   %.loc227_5.2 => constants.%.65c
// CHECK:STDOUT:   %AddAssignWith.lookup_impl_witness => constants.%custom_witness.ebf67a.1
// CHECK:STDOUT:   %impl.elem0.loc227_5.2 => constants.%Int32.Op.4dca31.12
// CHECK:STDOUT:   %specific_impl_fn.loc227_5.2 => constants.%Int32.Op.4dca31.12
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @SubAssignWith.loc231(constants.%Int32, constants.%SubAssignWith.facet.e0e) {
// CHECK:STDOUT:   %U.patt.loc231_19.2 => constants.%U.patt.d47
// CHECK:STDOUT:   %U.loc231_19.1 => constants.%Int32
// CHECK:STDOUT:   %SubAssignWith.type.loc231_50.1 => constants.%SubAssignWith.type.4fc
// CHECK:STDOUT:   %pattern_type.loc231_28 => constants.%pattern_type.190
// CHECK:STDOUT:   %T.patt.loc231_28.2 => constants.%T.patt.a39
// CHECK:STDOUT:   %T.loc231_28.1 => constants.%SubAssignWith.facet.e0e
// CHECK:STDOUT:   %T.as_type.loc231_60.1 => constants.%Int32
// CHECK:STDOUT:   %pattern_type.loc231_58 => constants.%pattern_type.8b4
// CHECK:STDOUT:   %x.patt.loc231_58.2 => constants.%x.patt.d79
// CHECK:STDOUT:   %x.param_patt.loc231_53.2 => constants.%x.param_patt.0c7
// CHECK:STDOUT:   %pattern_type.loc231_64 => constants.%pattern_type.8b4
// CHECK:STDOUT:   %y.param_patt.loc231_64.2 => constants.%y.param_patt.e0c
// CHECK:STDOUT:   %y.patt.loc231_64.2 => constants.%y.patt.0a3
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc231_53 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc231_64 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc233 => constants.%complete_type.1eb
// CHECK:STDOUT:   %SubAssignWith.assoc_type => constants.%SubAssignWith.assoc_type.ac6
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.c2a
// CHECK:STDOUT:   %SubAssignWith.WithSelf.Op.type => constants.%SubAssignWith.WithSelf.Op.type.6ac
// CHECK:STDOUT:   %.loc233_5.2 => constants.%.740
// CHECK:STDOUT:   %SubAssignWith.lookup_impl_witness => constants.%custom_witness.c2b241.1
// CHECK:STDOUT:   %impl.elem0.loc233_5.2 => constants.%Int32.Op.4dca31.14
// CHECK:STDOUT:   %specific_impl_fn.loc233_5.2 => constants.%Int32.Op.4dca31.14
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @MulAssignWith.loc237(constants.%Int32, constants.%MulAssignWith.facet.13c) {
// CHECK:STDOUT:   %U.patt.loc237_19.2 => constants.%U.patt.d47
// CHECK:STDOUT:   %U.loc237_19.1 => constants.%Int32
// CHECK:STDOUT:   %MulAssignWith.type.loc237_50.1 => constants.%MulAssignWith.type.5fb
// CHECK:STDOUT:   %pattern_type.loc237_28 => constants.%pattern_type.f3b
// CHECK:STDOUT:   %T.patt.loc237_28.2 => constants.%T.patt.f66
// CHECK:STDOUT:   %T.loc237_28.1 => constants.%MulAssignWith.facet.13c
// CHECK:STDOUT:   %T.as_type.loc237_60.1 => constants.%Int32
// CHECK:STDOUT:   %pattern_type.loc237_58 => constants.%pattern_type.8b4
// CHECK:STDOUT:   %x.patt.loc237_58.2 => constants.%x.patt.d79
// CHECK:STDOUT:   %x.param_patt.loc237_53.2 => constants.%x.param_patt.0c7
// CHECK:STDOUT:   %pattern_type.loc237_64 => constants.%pattern_type.8b4
// CHECK:STDOUT:   %y.param_patt.loc237_64.2 => constants.%y.param_patt.e0c
// CHECK:STDOUT:   %y.patt.loc237_64.2 => constants.%y.patt.0a3
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc237_53 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc237_64 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc239 => constants.%complete_type.ef7
// CHECK:STDOUT:   %MulAssignWith.assoc_type => constants.%MulAssignWith.assoc_type.c04
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.2db
// CHECK:STDOUT:   %MulAssignWith.WithSelf.Op.type => constants.%MulAssignWith.WithSelf.Op.type.73c
// CHECK:STDOUT:   %.loc239_5.2 => constants.%.a28
// CHECK:STDOUT:   %MulAssignWith.lookup_impl_witness => constants.%custom_witness.ec9760.1
// CHECK:STDOUT:   %impl.elem0.loc239_5.2 => constants.%Int32.Op.4dca31.15
// CHECK:STDOUT:   %specific_impl_fn.loc239_5.2 => constants.%Int32.Op.4dca31.15
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @DivAssignWith.loc243(constants.%Int32, constants.%DivAssignWith.facet.e59) {
// CHECK:STDOUT:   %U.patt.loc243_19.2 => constants.%U.patt.d47
// CHECK:STDOUT:   %U.loc243_19.1 => constants.%Int32
// CHECK:STDOUT:   %DivAssignWith.type.loc243_50.1 => constants.%DivAssignWith.type.3ea
// CHECK:STDOUT:   %pattern_type.loc243_28 => constants.%pattern_type.8f3
// CHECK:STDOUT:   %T.patt.loc243_28.2 => constants.%T.patt.316
// CHECK:STDOUT:   %T.loc243_28.1 => constants.%DivAssignWith.facet.e59
// CHECK:STDOUT:   %T.as_type.loc243_60.1 => constants.%Int32
// CHECK:STDOUT:   %pattern_type.loc243_58 => constants.%pattern_type.8b4
// CHECK:STDOUT:   %x.patt.loc243_58.2 => constants.%x.patt.d79
// CHECK:STDOUT:   %x.param_patt.loc243_53.2 => constants.%x.param_patt.0c7
// CHECK:STDOUT:   %pattern_type.loc243_64 => constants.%pattern_type.8b4
// CHECK:STDOUT:   %y.param_patt.loc243_64.2 => constants.%y.param_patt.e0c
// CHECK:STDOUT:   %y.patt.loc243_64.2 => constants.%y.patt.0a3
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc243_53 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc243_64 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc245 => constants.%complete_type.502
// CHECK:STDOUT:   %DivAssignWith.assoc_type => constants.%DivAssignWith.assoc_type.ccd
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.d23
// CHECK:STDOUT:   %DivAssignWith.WithSelf.Op.type => constants.%DivAssignWith.WithSelf.Op.type.b74
// CHECK:STDOUT:   %.loc245_5.2 => constants.%.204
// CHECK:STDOUT:   %DivAssignWith.lookup_impl_witness => constants.%custom_witness.9697f7.1
// CHECK:STDOUT:   %impl.elem0.loc245_5.2 => constants.%Int32.Op.4dca31.16
// CHECK:STDOUT:   %specific_impl_fn.loc245_5.2 => constants.%Int32.Op.4dca31.16
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ModAssignWith.loc249(constants.%Int32, constants.%ModAssignWith.facet.dac) {
// CHECK:STDOUT:   %U.patt.loc249_19.2 => constants.%U.patt.d47
// CHECK:STDOUT:   %U.loc249_19.1 => constants.%Int32
// CHECK:STDOUT:   %ModAssignWith.type.loc249_50.1 => constants.%ModAssignWith.type.567
// CHECK:STDOUT:   %pattern_type.loc249_28 => constants.%pattern_type.9ff
// CHECK:STDOUT:   %T.patt.loc249_28.2 => constants.%T.patt.6f1
// CHECK:STDOUT:   %T.loc249_28.1 => constants.%ModAssignWith.facet.dac
// CHECK:STDOUT:   %T.as_type.loc249_60.1 => constants.%Int32
// CHECK:STDOUT:   %pattern_type.loc249_58 => constants.%pattern_type.8b4
// CHECK:STDOUT:   %x.patt.loc249_58.2 => constants.%x.patt.d79
// CHECK:STDOUT:   %x.param_patt.loc249_53.2 => constants.%x.param_patt.0c7
// CHECK:STDOUT:   %pattern_type.loc249_64 => constants.%pattern_type.8b4
// CHECK:STDOUT:   %y.param_patt.loc249_64.2 => constants.%y.param_patt.e0c
// CHECK:STDOUT:   %y.patt.loc249_64.2 => constants.%y.patt.0a3
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc249_53 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc249_64 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc251 => constants.%complete_type.446
// CHECK:STDOUT:   %ModAssignWith.assoc_type => constants.%ModAssignWith.assoc_type.fc5
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.1c63
// CHECK:STDOUT:   %ModAssignWith.WithSelf.Op.type => constants.%ModAssignWith.WithSelf.Op.type.16f
// CHECK:STDOUT:   %.loc251_5.2 => constants.%.c84
// CHECK:STDOUT:   %ModAssignWith.lookup_impl_witness => constants.%custom_witness.df8f2f.1
// CHECK:STDOUT:   %impl.elem0.loc251_5.2 => constants.%Int32.Op.4dca31.17
// CHECK:STDOUT:   %specific_impl_fn.loc251_5.2 => constants.%Int32.Op.4dca31.17
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AddAssignWith.loc225(constants.%Int32, constants.%AddAssignWith.facet.144) {
// CHECK:STDOUT:   %U.patt.loc225_19.2 => constants.%U.patt.d47
// CHECK:STDOUT:   %U.loc225_19.1 => constants.%Int32
// CHECK:STDOUT:   %AddAssignWith.type.loc225_50.1 => constants.%AddAssignWith.type.23a
// CHECK:STDOUT:   %pattern_type.loc225_28 => constants.%pattern_type.5ad
// CHECK:STDOUT:   %T.patt.loc225_28.2 => constants.%T.patt.96c
// CHECK:STDOUT:   %T.loc225_28.1 => constants.%AddAssignWith.facet.144
// CHECK:STDOUT:   %T.as_type.loc225_60.1 => constants.%Int64
// CHECK:STDOUT:   %pattern_type.loc225_58 => constants.%pattern_type.4aa
// CHECK:STDOUT:   %x.patt.loc225_58.2 => constants.%x.patt.ace
// CHECK:STDOUT:   %x.param_patt.loc225_53.2 => constants.%x.param_patt.b33
// CHECK:STDOUT:   %pattern_type.loc225_64 => constants.%pattern_type.8b4
// CHECK:STDOUT:   %y.param_patt.loc225_64.2 => constants.%y.param_patt.e0c
// CHECK:STDOUT:   %y.patt.loc225_64.2 => constants.%y.patt.0a3
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc225_53 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc225_64 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc227 => constants.%complete_type.e1b
// CHECK:STDOUT:   %AddAssignWith.assoc_type => constants.%AddAssignWith.assoc_type.4c7
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.114
// CHECK:STDOUT:   %AddAssignWith.WithSelf.Op.type => constants.%AddAssignWith.WithSelf.Op.type.cf5
// CHECK:STDOUT:   %.loc227_5.2 => constants.%.6e0
// CHECK:STDOUT:   %AddAssignWith.lookup_impl_witness => constants.%custom_witness.ebf67a.2
// CHECK:STDOUT:   %impl.elem0.loc227_5.2 => constants.%Int64.Op.4dca31.2
// CHECK:STDOUT:   %specific_impl_fn.loc227_5.2 => constants.%Int64.Op.4dca31.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @SubAssignWith.loc231(constants.%Int32, constants.%SubAssignWith.facet.a7e) {
// CHECK:STDOUT:   %U.patt.loc231_19.2 => constants.%U.patt.d47
// CHECK:STDOUT:   %U.loc231_19.1 => constants.%Int32
// CHECK:STDOUT:   %SubAssignWith.type.loc231_50.1 => constants.%SubAssignWith.type.4fc
// CHECK:STDOUT:   %pattern_type.loc231_28 => constants.%pattern_type.190
// CHECK:STDOUT:   %T.patt.loc231_28.2 => constants.%T.patt.a39
// CHECK:STDOUT:   %T.loc231_28.1 => constants.%SubAssignWith.facet.a7e
// CHECK:STDOUT:   %T.as_type.loc231_60.1 => constants.%Int64
// CHECK:STDOUT:   %pattern_type.loc231_58 => constants.%pattern_type.4aa
// CHECK:STDOUT:   %x.patt.loc231_58.2 => constants.%x.patt.ace
// CHECK:STDOUT:   %x.param_patt.loc231_53.2 => constants.%x.param_patt.b33
// CHECK:STDOUT:   %pattern_type.loc231_64 => constants.%pattern_type.8b4
// CHECK:STDOUT:   %y.param_patt.loc231_64.2 => constants.%y.param_patt.e0c
// CHECK:STDOUT:   %y.patt.loc231_64.2 => constants.%y.patt.0a3
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc231_53 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc231_64 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc233 => constants.%complete_type.1eb
// CHECK:STDOUT:   %SubAssignWith.assoc_type => constants.%SubAssignWith.assoc_type.ac6
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.c2a
// CHECK:STDOUT:   %SubAssignWith.WithSelf.Op.type => constants.%SubAssignWith.WithSelf.Op.type.a31
// CHECK:STDOUT:   %.loc233_5.2 => constants.%.70d
// CHECK:STDOUT:   %SubAssignWith.lookup_impl_witness => constants.%custom_witness.c2b241.2
// CHECK:STDOUT:   %impl.elem0.loc233_5.2 => constants.%Int64.Op.4dca31.4
// CHECK:STDOUT:   %specific_impl_fn.loc233_5.2 => constants.%Int64.Op.4dca31.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @MulAssignWith.loc237(constants.%Int32, constants.%MulAssignWith.facet.b35) {
// CHECK:STDOUT:   %U.patt.loc237_19.2 => constants.%U.patt.d47
// CHECK:STDOUT:   %U.loc237_19.1 => constants.%Int32
// CHECK:STDOUT:   %MulAssignWith.type.loc237_50.1 => constants.%MulAssignWith.type.5fb
// CHECK:STDOUT:   %pattern_type.loc237_28 => constants.%pattern_type.f3b
// CHECK:STDOUT:   %T.patt.loc237_28.2 => constants.%T.patt.f66
// CHECK:STDOUT:   %T.loc237_28.1 => constants.%MulAssignWith.facet.b35
// CHECK:STDOUT:   %T.as_type.loc237_60.1 => constants.%Int64
// CHECK:STDOUT:   %pattern_type.loc237_58 => constants.%pattern_type.4aa
// CHECK:STDOUT:   %x.patt.loc237_58.2 => constants.%x.patt.ace
// CHECK:STDOUT:   %x.param_patt.loc237_53.2 => constants.%x.param_patt.b33
// CHECK:STDOUT:   %pattern_type.loc237_64 => constants.%pattern_type.8b4
// CHECK:STDOUT:   %y.param_patt.loc237_64.2 => constants.%y.param_patt.e0c
// CHECK:STDOUT:   %y.patt.loc237_64.2 => constants.%y.patt.0a3
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc237_53 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc237_64 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc239 => constants.%complete_type.ef7
// CHECK:STDOUT:   %MulAssignWith.assoc_type => constants.%MulAssignWith.assoc_type.c04
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.2db
// CHECK:STDOUT:   %MulAssignWith.WithSelf.Op.type => constants.%MulAssignWith.WithSelf.Op.type.606
// CHECK:STDOUT:   %.loc239_5.2 => constants.%.375
// CHECK:STDOUT:   %MulAssignWith.lookup_impl_witness => constants.%custom_witness.ec9760.2
// CHECK:STDOUT:   %impl.elem0.loc239_5.2 => constants.%Int64.Op.4dca31.5
// CHECK:STDOUT:   %specific_impl_fn.loc239_5.2 => constants.%Int64.Op.4dca31.5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @DivAssignWith.loc243(constants.%Int32, constants.%DivAssignWith.facet.b1e) {
// CHECK:STDOUT:   %U.patt.loc243_19.2 => constants.%U.patt.d47
// CHECK:STDOUT:   %U.loc243_19.1 => constants.%Int32
// CHECK:STDOUT:   %DivAssignWith.type.loc243_50.1 => constants.%DivAssignWith.type.3ea
// CHECK:STDOUT:   %pattern_type.loc243_28 => constants.%pattern_type.8f3
// CHECK:STDOUT:   %T.patt.loc243_28.2 => constants.%T.patt.316
// CHECK:STDOUT:   %T.loc243_28.1 => constants.%DivAssignWith.facet.b1e
// CHECK:STDOUT:   %T.as_type.loc243_60.1 => constants.%Int64
// CHECK:STDOUT:   %pattern_type.loc243_58 => constants.%pattern_type.4aa
// CHECK:STDOUT:   %x.patt.loc243_58.2 => constants.%x.patt.ace
// CHECK:STDOUT:   %x.param_patt.loc243_53.2 => constants.%x.param_patt.b33
// CHECK:STDOUT:   %pattern_type.loc243_64 => constants.%pattern_type.8b4
// CHECK:STDOUT:   %y.param_patt.loc243_64.2 => constants.%y.param_patt.e0c
// CHECK:STDOUT:   %y.patt.loc243_64.2 => constants.%y.patt.0a3
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc243_53 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc243_64 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc245 => constants.%complete_type.502
// CHECK:STDOUT:   %DivAssignWith.assoc_type => constants.%DivAssignWith.assoc_type.ccd
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.d23
// CHECK:STDOUT:   %DivAssignWith.WithSelf.Op.type => constants.%DivAssignWith.WithSelf.Op.type.d14
// CHECK:STDOUT:   %.loc245_5.2 => constants.%.f69
// CHECK:STDOUT:   %DivAssignWith.lookup_impl_witness => constants.%custom_witness.9697f7.2
// CHECK:STDOUT:   %impl.elem0.loc245_5.2 => constants.%Int64.Op.4dca31.6
// CHECK:STDOUT:   %specific_impl_fn.loc245_5.2 => constants.%Int64.Op.4dca31.6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ModAssignWith.loc249(constants.%Int32, constants.%ModAssignWith.facet.342) {
// CHECK:STDOUT:   %U.patt.loc249_19.2 => constants.%U.patt.d47
// CHECK:STDOUT:   %U.loc249_19.1 => constants.%Int32
// CHECK:STDOUT:   %ModAssignWith.type.loc249_50.1 => constants.%ModAssignWith.type.567
// CHECK:STDOUT:   %pattern_type.loc249_28 => constants.%pattern_type.9ff
// CHECK:STDOUT:   %T.patt.loc249_28.2 => constants.%T.patt.6f1
// CHECK:STDOUT:   %T.loc249_28.1 => constants.%ModAssignWith.facet.342
// CHECK:STDOUT:   %T.as_type.loc249_60.1 => constants.%Int64
// CHECK:STDOUT:   %pattern_type.loc249_58 => constants.%pattern_type.4aa
// CHECK:STDOUT:   %x.patt.loc249_58.2 => constants.%x.patt.ace
// CHECK:STDOUT:   %x.param_patt.loc249_53.2 => constants.%x.param_patt.b33
// CHECK:STDOUT:   %pattern_type.loc249_64 => constants.%pattern_type.8b4
// CHECK:STDOUT:   %y.param_patt.loc249_64.2 => constants.%y.param_patt.e0c
// CHECK:STDOUT:   %y.patt.loc249_64.2 => constants.%y.patt.0a3
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc249_53 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc249_64 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc251 => constants.%complete_type.446
// CHECK:STDOUT:   %ModAssignWith.assoc_type => constants.%ModAssignWith.assoc_type.fc5
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.1c63
// CHECK:STDOUT:   %ModAssignWith.WithSelf.Op.type => constants.%ModAssignWith.WithSelf.Op.type.690
// CHECK:STDOUT:   %.loc251_5.2 => constants.%.af4
// CHECK:STDOUT:   %ModAssignWith.lookup_impl_witness => constants.%custom_witness.df8f2f.2
// CHECK:STDOUT:   %impl.elem0.loc251_5.2 => constants.%Int64.Op.4dca31.7
// CHECK:STDOUT:   %specific_impl_fn.loc251_5.2 => constants.%Int64.Op.4dca31.7
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
// CHECK:STDOUT:   %y.patt.loc294_11.2 => constants.%y.patt.053
// CHECK:STDOUT:   %T.as_type.loc294_24.1 => constants.%T.as_type.684
// CHECK:STDOUT:   %pattern_type.loc294_22 => constants.%pattern_type.2dd
// CHECK:STDOUT:   %x.param_patt.loc294_22.2 => constants.%x.param_patt.cbc
// CHECK:STDOUT:   %x.patt.loc294_22.2 => constants.%x.patt.7fb
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
// CHECK:STDOUT:   %Inc.WithSelf.Op.type => constants.%Inc.WithSelf.Op.type.d1d
// CHECK:STDOUT:   %.loc280_3.3 => constants.%.176
// CHECK:STDOUT:   %Inc.lookup_impl_witness => constants.%custom_witness.f3f
// CHECK:STDOUT:   %impl.elem0.loc280_3.2 => constants.%Int32.Op.4dca31.18
// CHECK:STDOUT:   %specific_impl_fn.loc280_3.2 => constants.%Int32.Op.4dca31.18
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
// CHECK:STDOUT:   %Dec.WithSelf.Op.type => constants.%Dec.WithSelf.Op.type.e55
// CHECK:STDOUT:   %.loc286_3.3 => constants.%.d09
// CHECK:STDOUT:   %Dec.lookup_impl_witness => constants.%custom_witness.b24
// CHECK:STDOUT:   %impl.elem0.loc286_3.2 => constants.%Int32.Op.4dca31.19
// CHECK:STDOUT:   %specific_impl_fn.loc286_3.2 => constants.%Int32.Op.4dca31.19
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestNegate(constants.%Destroy.facet.191, constants.%facet_value.9cc) {
// CHECK:STDOUT:   %Result.patt.loc292_9.2 => constants.%Result.patt
// CHECK:STDOUT:   %Result.loc292_9.1 => constants.%Destroy.facet.191
// CHECK:STDOUT:   %Result.as_type.loc293_49.1 => constants.%Int64
// CHECK:STDOUT:   %facet_type => constants.%facet_type.d8c
// CHECK:STDOUT:   %pattern_type.loc293 => constants.%pattern_type.54c
// CHECK:STDOUT:   %T.patt.loc293_4.2 => constants.%T.patt.16f
// CHECK:STDOUT:   %T.loc293_4.1 => constants.%facet_value.9cc
// CHECK:STDOUT:   %pattern_type.loc294_11 => constants.%pattern_type.4aa
// CHECK:STDOUT:   %y.param_patt.loc294_11.2 => constants.%y.param_patt.a08
// CHECK:STDOUT:   %y.patt.loc294_11.2 => constants.%y.patt.63b
// CHECK:STDOUT:   %T.as_type.loc294_24.1 => constants.%Int16
// CHECK:STDOUT:   %pattern_type.loc294_22 => constants.%pattern_type.785
// CHECK:STDOUT:   %x.param_patt.loc294_22.2 => constants.%x.param_patt.3b0
// CHECK:STDOUT:   %x.patt.loc294_22.2 => constants.%x.patt.b76
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc294_11 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc294_22 => constants.%complete_type.357
// CHECK:STDOUT:   %Negate.lookup_impl_witness => constants.%custom_witness.100
// CHECK:STDOUT:   %Negate.facet.loc296_3.3 => constants.%Negate.facet.8f4
// CHECK:STDOUT:   %Negate.WithSelf.Op.type => constants.%Negate.WithSelf.Op.type.181
// CHECK:STDOUT:   %.loc296_3.7 => constants.%.66a
// CHECK:STDOUT:   %impl.elem1.loc296_3.2 => constants.%Int16.Op.4dca31.18
// CHECK:STDOUT:   %specific_impl_fn.loc296_3.3 => constants.%Int16.Op.4dca31.18
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.c2c
// CHECK:STDOUT:   %.loc296_3.8 => constants.%.c2e
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.e7a31c.3
// CHECK:STDOUT:   %impl.elem0.loc296_3.2 => constants.%Int64.Op.4dca31.1
// CHECK:STDOUT:   %specific_impl_fn.loc296_3.4 => constants.%Int64.Op.4dca31.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestNegate(constants.%Destroy.facet.099, constants.%facet_value.dcd) {
// CHECK:STDOUT:   %Result.patt.loc292_9.2 => constants.%Result.patt
// CHECK:STDOUT:   %Result.loc292_9.1 => constants.%Destroy.facet.099
// CHECK:STDOUT:   %Result.as_type.loc293_49.1 => constants.%Int32
// CHECK:STDOUT:   %facet_type => constants.%facet_type.021
// CHECK:STDOUT:   %pattern_type.loc293 => constants.%pattern_type.4cd
// CHECK:STDOUT:   %T.patt.loc293_4.2 => constants.%T.patt.d74
// CHECK:STDOUT:   %T.loc293_4.1 => constants.%facet_value.dcd
// CHECK:STDOUT:   %pattern_type.loc294_11 => constants.%pattern_type.8b4
// CHECK:STDOUT:   %y.param_patt.loc294_11.2 => constants.%y.param_patt.e0c
// CHECK:STDOUT:   %y.patt.loc294_11.2 => constants.%y.patt.0a3
// CHECK:STDOUT:   %T.as_type.loc294_24.1 => constants.%Int32
// CHECK:STDOUT:   %pattern_type.loc294_22 => constants.%pattern_type.8b4
// CHECK:STDOUT:   %x.param_patt.loc294_22.2 => constants.%x.param_patt.27c
// CHECK:STDOUT:   %x.patt.loc294_22.2 => constants.%x.patt.4c9
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc294_11 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc294_22 => constants.%complete_type.357
// CHECK:STDOUT:   %Negate.lookup_impl_witness => constants.%custom_witness.a85
// CHECK:STDOUT:   %Negate.facet.loc296_3.3 => constants.%Negate.facet.aa9
// CHECK:STDOUT:   %Negate.WithSelf.Op.type => constants.%Negate.WithSelf.Op.type.050
// CHECK:STDOUT:   %.loc296_3.7 => constants.%.668
// CHECK:STDOUT:   %impl.elem1.loc296_3.2 => constants.%Int32.Op.4dca31.20
// CHECK:STDOUT:   %specific_impl_fn.loc296_3.3 => constants.%Int32.Op.4dca31.20
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.b76
// CHECK:STDOUT:   %.loc296_3.8 => constants.%.043
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.e7a31c.2
// CHECK:STDOUT:   %impl.elem0.loc296_3.2 => constants.%Int32.Op.4dca31.1
// CHECK:STDOUT:   %specific_impl_fn.loc296_3.4 => constants.%Int32.Op.4dca31.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
