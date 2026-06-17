// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
// EXTRA-ARGS: --target=x86_64-linux-gnu --clang-arg=-std=c++20
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/operators/bitwise_operators.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/operators/bitwise_operators.carbon

// TODO: test implicit conversions when supported

library "[[@TEST_NAME]]";
import Cpp inline '''c++
class Int16;
class Int32;
class Int64;

class Int16 {
public:
  auto operator&(Int32) const -> Int32;
  auto operator|(Int32) const -> Int32;
  auto operator^(Int32) const -> Int32;
  auto operator<<(Int32) const -> Int32;
  auto operator>>(Int32) const -> Int32;

  // Tests `class op builtin` works
  auto operator&(short) const -> Int16;
  auto operator|(short) const -> Int16;
  auto operator^(short) const -> Int16;
  auto operator<<(short) const -> Int16;
  auto operator>>(short) const -> Int16;

  auto operator&=(short) const -> Int16&;
  auto operator|=(short) const -> Int16&;
  auto operator^=(short) const -> Int16&;
  auto operator<<=(short) const -> Int16&;
  auto operator>>=(short) const -> Int16&;

  auto operator~() const -> Int64;
};

auto operator&(short, Int16) -> Int16;
auto operator|(short, Int16) -> Int16;
auto operator^(short, Int16) -> Int16;
auto operator<<(short, Int16) -> Int16;
auto operator>>(short, Int16) -> Int16;

class Int32 {
public:
  auto operator&(Int16) const -> Int32;
  auto operator|(Int16) const -> Int32;
  auto operator^(Int16) const -> Int32;
  auto operator<<(Int16) const -> Int32;
  auto operator>>(Int16) const -> Int32;

  auto operator&(Int32) const -> Int32;
  auto operator|(Int32) const -> Int32;
  auto operator^(Int32) const -> Int32;
  auto operator<<(Int32) const -> Int32;
  auto operator>>(Int32) const -> Int32;

  auto operator~() const -> Int32;

  auto operator&=(Int32) -> void;
  auto operator|=(Int32) -> void;
  auto operator^=(Int32) -> void;
  auto operator<<=(Int32) -> void;
  auto operator>>=(Int32) -> void;
};

class Int64;
auto operator&(Int64 x, Int32 y) -> Int64;
auto operator|(Int64 x, Int32 y) -> Int64;
auto operator^(Int64 x, Int32 y) -> Int64;
auto operator<<(Int64 x, Int32 y) -> Int64;
auto operator>>(Int64 x, Int32 y) -> Int64;

class Int64 {
public:
  auto operator&=(Int32) -> void;
  auto operator|=(Int32) -> void;
  auto operator^=(Int32) -> void;
  auto operator<<=(Int32) -> void;
  auto operator>>=(Int32) -> void;

  // Tests hidden friends work
  friend auto operator&(Int32 x, Int64 y) -> Int64 {
    return y;
  }
  friend auto operator|(Int32 x, Int64 y) -> Int64 {
    return y;
  }
  friend auto operator^(Int32 x, Int64 y) -> Int64 {
    return y;
  }
  friend auto operator<<(Int32 x, Int64 y) -> Int64 {
    return y;
  }
  friend auto operator>>(Int32 x, Int64 y) -> Int64 {
    return y;
  }
};
''';

fn BitAndWith[
  U:! type,
  T:! Core.BitAndWith(U) where .Result impls Core.Destroy
](x: T, y: U) {
  //@dump-sem-ir-begin
  x & y;
  //@dump-sem-ir-end
}

fn BitOrWith[
  U:! type,
  T:! Core.BitOrWith(U) where .Result impls Core.Destroy
](x: T, y: U) {
  //@dump-sem-ir-begin
  x | y;
  //@dump-sem-ir-end
}

fn BitXorWith[
  U:! type,
  T:! Core.BitXorWith(U) where .Result impls Core.Destroy
](x: T, y: U) {
  //@dump-sem-ir-begin
  x ^ y;
  //@dump-sem-ir-end
}

fn LeftShiftWith[
  U:! type,
  T:! Core.LeftShiftWith(U) where .Result impls Core.Destroy
](x: T, y: U) {
  //@dump-sem-ir-begin
  x << y;
  //@dump-sem-ir-end
}

fn RightShiftWith[
  U:! type,
  T:! Core.RightShiftWith(U) where .Result impls Core.Destroy
](x: T, y: U) {
  //@dump-sem-ir-begin
  x >> y;
  //@dump-sem-ir-end
}

fn TestOpWith(a: Cpp.Int16, b: Cpp.Int32, c: Cpp.Int32, d: Cpp.Int64, x: i16) {
  // i16 `op` Int16
  BitAndWith(x, a);
  BitOrWith(x, a);
  BitXorWith(x, a);
  LeftShiftWith(x, a);
  RightShiftWith(x, a);

  // Int16 `op` i16
  BitAndWith(a, x);
  BitOrWith(a, x);
  BitXorWith(a, x);
  LeftShiftWith(a, x);
  RightShiftWith(a, x);

  // Int16 `op` Int32
  BitAndWith(a, b);
  BitOrWith(a, b);
  BitXorWith(a, b);
  LeftShiftWith(a, b);
  RightShiftWith(a, b);

  // Int32 `op` Int16
  BitAndWith(b, a);
  BitOrWith(b, a);
  BitXorWith(b, a);
  LeftShiftWith(b, a);
  RightShiftWith(b, a);

  // Int32 `op` Int32
  BitAndWith(b, c);
  BitOrWith(b, c);
  BitXorWith(b, c);
  LeftShiftWith(b, c);
  RightShiftWith(b, c);

  // Int32 `op` Int64
  BitAndWith(c, d);
  BitOrWith(c, d);
  BitXorWith(c, d);
  LeftShiftWith(c, d);
  RightShiftWith(c, d);

  // Int64 `op` Int32
  BitAndWith(d, c);
  BitOrWith(d, c);
  BitXorWith(d, c);
  LeftShiftWith(d, c);
  RightShiftWith(d, c);
}

fn BitAndAssignWith[U:! type, T:! Core.BitAndAssignWith(U)](var x: T, y: U) {
  //@dump-sem-ir-begin
  x &= y;
  //@dump-sem-ir-end
}

fn BitOrAssignWith[U:! type, T:! Core.BitOrAssignWith(U)](var x: T, y: U) {
  //@dump-sem-ir-begin
  x |= y;
  //@dump-sem-ir-end
}

fn BitXorAssignWith[U:! type, T:! Core.BitXorAssignWith(U)](var x: T, y: U) {
  //@dump-sem-ir-begin
  x ^= y;
  //@dump-sem-ir-end
}

fn LeftShiftAssignWith[U:! type, T:! Core.LeftShiftAssignWith(U)](var x: T, y: U) {
  //@dump-sem-ir-begin
  x <<= y;
  //@dump-sem-ir-end
}

fn RightShiftAssignWith[U:! type, T:! Core.RightShiftAssignWith(U)](var x: T, y: U) {
  //@dump-sem-ir-begin
  x >>= y;
  //@dump-sem-ir-end
}

fn TestOpAssignWith(int16: Cpp.Int16, int32: Cpp.Int32, int64: Cpp.Int64, x: i16) {
  // Int16 `op` i16
  BitAndAssignWith(int16, x);
  BitOrAssignWith(int16, x);
  BitXorAssignWith(int16, x);
  LeftShiftAssignWith(int16, x);
  RightShiftAssignWith(int16, x);

  // Int32 `op` Int32
  BitAndAssignWith(int32, int32);
  BitOrAssignWith(int32, int32);
  BitXorAssignWith(int32, int32);
  LeftShiftAssignWith(int32, int32);
  RightShiftAssignWith(int32, int32);

  // Int64 `op` Int32
  BitAndAssignWith(int64, int32);
  BitOrAssignWith(int64, int32);
  BitXorAssignWith(int64, int32);
  LeftShiftAssignWith(int64, int32);
  RightShiftAssignWith(int64, int32);
}

fn TestBitComplement[
  T:! Core.BitComplement where .Result impls Core.Destroy
](x: T) {
  //@dump-sem-ir-begin
  ^x;
  //@dump-sem-ir-end
}

fn TestUnaryOperators(a: Cpp.Int16, b: Cpp.Int32) {
  TestBitComplement(a);
  TestBitComplement(b);
}

// CHECK:STDOUT: --- bitwise_operators.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %U.patt.d47: %pattern_type.98f = symbolic_binding_pattern U, 0 [symbolic]
// CHECK:STDOUT:   %U.67d: type = symbolic_binding U, 0 [symbolic]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Other: type = symbolic_binding Other, 0 [symbolic]
// CHECK:STDOUT:   %BitAndWith.type.4f130e.1: type = facet_type <@BitAndWith.1, @BitAndWith.1(%Other)> [symbolic]
// CHECK:STDOUT:   %Self.274af9.1: %BitAndWith.type.4f130e.1 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %BitAndWith.assoc_type.a6e8f9.1: type = assoc_entity_type @BitAndWith.1, @BitAndWith.1(%Other) [symbolic]
// CHECK:STDOUT:   %BitAndWith.WithSelf.Op.type.4e1bca.1: type = fn_type @BitAndWith.WithSelf.Op, @BitAndWith.WithSelf(%Other, %Self.274af9.1) [symbolic]
// CHECK:STDOUT:   %BitAndWith.WithSelf.Op.549040.1: %BitAndWith.WithSelf.Op.type.4e1bca.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %BitAndWith.type.4f130e.2: type = facet_type <@BitAndWith.1, @BitAndWith.1(%U.67d)> [symbolic]
// CHECK:STDOUT:   %.Self.ace: %BitAndWith.type.4f130e.2 = symbolic_binding .Self [symbolic]
// CHECK:STDOUT:   %BitAndWith.assoc_type.a6e8f9.2: type = assoc_entity_type @BitAndWith.1, @BitAndWith.1(%U.67d) [symbolic]
// CHECK:STDOUT:   %assoc0.24f955.2: %BitAndWith.assoc_type.a6e8f9.2 = assoc_entity element0, imports.%Core.import_ref.efd [symbolic]
// CHECK:STDOUT:   %assoc1.acd56d.2: %BitAndWith.assoc_type.a6e8f9.2 = assoc_entity element1, imports.%Core.import_ref.2c4 [symbolic]
// CHECK:STDOUT:   %require_complete.a4e: <witness> = require_complete_type %BitAndWith.type.4f130e.2 [symbolic]
// CHECK:STDOUT:   %.Self.as_type.107: type = facet_access_type %.Self.ace [symbolic]
// CHECK:STDOUT:   %BitAndWith.lookup_impl_witness.4ca: <witness> = lookup_impl_witness %.Self.ace, @BitAndWith.1, @BitAndWith.1(%U.67d) [symbolic]
// CHECK:STDOUT:   %impl.elem0.c32: type = impl_witness_access %BitAndWith.lookup_impl_witness.4ca, element0 [symbolic]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %BitAndWith_where.type.4f1: type = facet_type <@BitAndWith.1, @BitAndWith.1(%U.67d) where %impl.elem0.c32 impls @Destroy> [symbolic]
// CHECK:STDOUT:   %pattern_type.626: type = pattern_type %BitAndWith_where.type.4f1 [symbolic]
// CHECK:STDOUT:   %T.patt.96b: %pattern_type.626 = symbolic_binding_pattern T, 1 [symbolic]
// CHECK:STDOUT:   %T.274: %BitAndWith_where.type.4f1 = symbolic_binding T, 1 [symbolic]
// CHECK:STDOUT:   %T.as_type.417: type = facet_access_type %T.274 [symbolic]
// CHECK:STDOUT:   %pattern_type.acade5.2: type = pattern_type %T.as_type.417 [symbolic]
// CHECK:STDOUT:   %x.param_patt.1ce: %pattern_type.acade5.2 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %x.patt.b16: %pattern_type.acade5.2 = at_binding_pattern x, %x.param_patt.1ce [symbolic]
// CHECK:STDOUT:   %pattern_type.51d1c4.2: type = pattern_type %U.67d [symbolic]
// CHECK:STDOUT:   %y.param_patt.be3: %pattern_type.51d1c4.2 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %y.patt.b85: %pattern_type.51d1c4.2 = at_binding_pattern y, %y.param_patt.be3 [symbolic]
// CHECK:STDOUT:   %assoc1.85c: %BitAndWith.assoc_type.a6e8f9.1 = assoc_entity element1, imports.%Core.import_ref.a28 [symbolic]
// CHECK:STDOUT:   %BitAndWith.lookup_impl_witness.25c49f.2: <witness> = lookup_impl_witness %T.274, @BitAndWith.1, @BitAndWith.1(%U.67d) [symbolic]
// CHECK:STDOUT:   %BitAndWith.facet.e71: %BitAndWith.type.4f130e.2 = facet_value %T.as_type.417, (%BitAndWith.lookup_impl_witness.25c49f.2) [symbolic]
// CHECK:STDOUT:   %impl.elem0.7c98f0.2: type = impl_witness_access %BitAndWith.lookup_impl_witness.25c49f.2, element0 [symbolic]
// CHECK:STDOUT:   %BitAndWith.WithSelf.Op.type.4f2: type = fn_type @BitAndWith.WithSelf.Op, @BitAndWith.WithSelf(%U.67d, %BitAndWith.facet.e71) [symbolic]
// CHECK:STDOUT:   %.0cdb: type = fn_type_with_self_type %BitAndWith.WithSelf.Op.type.4f2, %BitAndWith.facet.e71 [symbolic]
// CHECK:STDOUT:   %impl.elem1.69c: %.0cdb = impl_witness_access %BitAndWith.lookup_impl_witness.25c49f.2, element1 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.cfc: <specific function> = specific_impl_function %impl.elem1.69c, @BitAndWith.WithSelf.Op(%U.67d, %BitAndWith.facet.e71) [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.61e: <witness> = lookup_impl_witness %impl.elem0.7c98f0.2, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.c83: %Destroy.type = facet_value %impl.elem0.7c98f0.2, (%Destroy.lookup_impl_witness.61e) [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.230: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.c83) [symbolic]
// CHECK:STDOUT:   %.8bd: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.230, %Destroy.facet.c83 [symbolic]
// CHECK:STDOUT:   %impl.elem0.5a7: %.8bd = impl_witness_access %Destroy.lookup_impl_witness.61e, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.5a8: <specific function> = specific_impl_function %impl.elem0.5a7, @Destroy.WithSelf.Op(%Destroy.facet.c83) [symbolic]
// CHECK:STDOUT:   %BitOrWith.type.d96408.1: type = facet_type <@BitOrWith.1, @BitOrWith.1(%Other)> [symbolic]
// CHECK:STDOUT:   %Self.6cdc7c.1: %BitOrWith.type.d96408.1 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %BitOrWith.assoc_type.10d7b7.1: type = assoc_entity_type @BitOrWith.1, @BitOrWith.1(%Other) [symbolic]
// CHECK:STDOUT:   %BitOrWith.WithSelf.Op.type.0e2995.1: type = fn_type @BitOrWith.WithSelf.Op, @BitOrWith.WithSelf(%Other, %Self.6cdc7c.1) [symbolic]
// CHECK:STDOUT:   %BitOrWith.WithSelf.Op.d71e20.1: %BitOrWith.WithSelf.Op.type.0e2995.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %BitOrWith.type.d96408.2: type = facet_type <@BitOrWith.1, @BitOrWith.1(%U.67d)> [symbolic]
// CHECK:STDOUT:   %.Self.cf4: %BitOrWith.type.d96408.2 = symbolic_binding .Self [symbolic]
// CHECK:STDOUT:   %BitOrWith.assoc_type.10d7b7.2: type = assoc_entity_type @BitOrWith.1, @BitOrWith.1(%U.67d) [symbolic]
// CHECK:STDOUT:   %assoc0.7f25cd.2: %BitOrWith.assoc_type.10d7b7.2 = assoc_entity element0, imports.%Core.import_ref.4f6 [symbolic]
// CHECK:STDOUT:   %assoc1.53369c.2: %BitOrWith.assoc_type.10d7b7.2 = assoc_entity element1, imports.%Core.import_ref.c14 [symbolic]
// CHECK:STDOUT:   %require_complete.818: <witness> = require_complete_type %BitOrWith.type.d96408.2 [symbolic]
// CHECK:STDOUT:   %.Self.as_type.7a4: type = facet_access_type %.Self.cf4 [symbolic]
// CHECK:STDOUT:   %BitOrWith.lookup_impl_witness.c5e: <witness> = lookup_impl_witness %.Self.cf4, @BitOrWith.1, @BitOrWith.1(%U.67d) [symbolic]
// CHECK:STDOUT:   %impl.elem0.324: type = impl_witness_access %BitOrWith.lookup_impl_witness.c5e, element0 [symbolic]
// CHECK:STDOUT:   %BitOrWith_where.type.d96: type = facet_type <@BitOrWith.1, @BitOrWith.1(%U.67d) where %impl.elem0.324 impls @Destroy> [symbolic]
// CHECK:STDOUT:   %pattern_type.f1d: type = pattern_type %BitOrWith_where.type.d96 [symbolic]
// CHECK:STDOUT:   %T.patt.ecb: %pattern_type.f1d = symbolic_binding_pattern T, 1 [symbolic]
// CHECK:STDOUT:   %T.6cd: %BitOrWith_where.type.d96 = symbolic_binding T, 1 [symbolic]
// CHECK:STDOUT:   %T.as_type.725: type = facet_access_type %T.6cd [symbolic]
// CHECK:STDOUT:   %pattern_type.29dfc2.2: type = pattern_type %T.as_type.725 [symbolic]
// CHECK:STDOUT:   %x.param_patt.42d: %pattern_type.29dfc2.2 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %x.patt.b29: %pattern_type.29dfc2.2 = at_binding_pattern x, %x.param_patt.42d [symbolic]
// CHECK:STDOUT:   %assoc1.25a: %BitOrWith.assoc_type.10d7b7.1 = assoc_entity element1, imports.%Core.import_ref.532 [symbolic]
// CHECK:STDOUT:   %BitOrWith.lookup_impl_witness.1faae5.2: <witness> = lookup_impl_witness %T.6cd, @BitOrWith.1, @BitOrWith.1(%U.67d) [symbolic]
// CHECK:STDOUT:   %BitOrWith.facet.915: %BitOrWith.type.d96408.2 = facet_value %T.as_type.725, (%BitOrWith.lookup_impl_witness.1faae5.2) [symbolic]
// CHECK:STDOUT:   %impl.elem0.7ec769.2: type = impl_witness_access %BitOrWith.lookup_impl_witness.1faae5.2, element0 [symbolic]
// CHECK:STDOUT:   %BitOrWith.WithSelf.Op.type.be5: type = fn_type @BitOrWith.WithSelf.Op, @BitOrWith.WithSelf(%U.67d, %BitOrWith.facet.915) [symbolic]
// CHECK:STDOUT:   %.4cd: type = fn_type_with_self_type %BitOrWith.WithSelf.Op.type.be5, %BitOrWith.facet.915 [symbolic]
// CHECK:STDOUT:   %impl.elem1.4a2: %.4cd = impl_witness_access %BitOrWith.lookup_impl_witness.1faae5.2, element1 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.37b: <specific function> = specific_impl_function %impl.elem1.4a2, @BitOrWith.WithSelf.Op(%U.67d, %BitOrWith.facet.915) [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.7f4: <witness> = lookup_impl_witness %impl.elem0.7ec769.2, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.786: %Destroy.type = facet_value %impl.elem0.7ec769.2, (%Destroy.lookup_impl_witness.7f4) [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.e11: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.786) [symbolic]
// CHECK:STDOUT:   %.82b: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.e11, %Destroy.facet.786 [symbolic]
// CHECK:STDOUT:   %impl.elem0.5b8: %.82b = impl_witness_access %Destroy.lookup_impl_witness.7f4, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.056: <specific function> = specific_impl_function %impl.elem0.5b8, @Destroy.WithSelf.Op(%Destroy.facet.786) [symbolic]
// CHECK:STDOUT:   %BitXorWith.type.7de562.1: type = facet_type <@BitXorWith.1, @BitXorWith.1(%Other)> [symbolic]
// CHECK:STDOUT:   %Self.5c244d.1: %BitXorWith.type.7de562.1 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %BitXorWith.assoc_type.4ad612.1: type = assoc_entity_type @BitXorWith.1, @BitXorWith.1(%Other) [symbolic]
// CHECK:STDOUT:   %BitXorWith.WithSelf.Op.type.2f1ddb.1: type = fn_type @BitXorWith.WithSelf.Op, @BitXorWith.WithSelf(%Other, %Self.5c244d.1) [symbolic]
// CHECK:STDOUT:   %BitXorWith.WithSelf.Op.b3b3a4.1: %BitXorWith.WithSelf.Op.type.2f1ddb.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %BitXorWith.type.7de562.2: type = facet_type <@BitXorWith.1, @BitXorWith.1(%U.67d)> [symbolic]
// CHECK:STDOUT:   %.Self.9cb: %BitXorWith.type.7de562.2 = symbolic_binding .Self [symbolic]
// CHECK:STDOUT:   %BitXorWith.assoc_type.4ad612.2: type = assoc_entity_type @BitXorWith.1, @BitXorWith.1(%U.67d) [symbolic]
// CHECK:STDOUT:   %assoc0.8aa692.2: %BitXorWith.assoc_type.4ad612.2 = assoc_entity element0, imports.%Core.import_ref.37d [symbolic]
// CHECK:STDOUT:   %assoc1.58565f.2: %BitXorWith.assoc_type.4ad612.2 = assoc_entity element1, imports.%Core.import_ref.808 [symbolic]
// CHECK:STDOUT:   %require_complete.189: <witness> = require_complete_type %BitXorWith.type.7de562.2 [symbolic]
// CHECK:STDOUT:   %.Self.as_type.ea1: type = facet_access_type %.Self.9cb [symbolic]
// CHECK:STDOUT:   %BitXorWith.lookup_impl_witness.e34: <witness> = lookup_impl_witness %.Self.9cb, @BitXorWith.1, @BitXorWith.1(%U.67d) [symbolic]
// CHECK:STDOUT:   %impl.elem0.4e7: type = impl_witness_access %BitXorWith.lookup_impl_witness.e34, element0 [symbolic]
// CHECK:STDOUT:   %BitXorWith_where.type.7de: type = facet_type <@BitXorWith.1, @BitXorWith.1(%U.67d) where %impl.elem0.4e7 impls @Destroy> [symbolic]
// CHECK:STDOUT:   %pattern_type.676: type = pattern_type %BitXorWith_where.type.7de [symbolic]
// CHECK:STDOUT:   %T.patt.9d1: %pattern_type.676 = symbolic_binding_pattern T, 1 [symbolic]
// CHECK:STDOUT:   %T.5c2: %BitXorWith_where.type.7de = symbolic_binding T, 1 [symbolic]
// CHECK:STDOUT:   %T.as_type.f34: type = facet_access_type %T.5c2 [symbolic]
// CHECK:STDOUT:   %pattern_type.3ed574.2: type = pattern_type %T.as_type.f34 [symbolic]
// CHECK:STDOUT:   %x.param_patt.f3e: %pattern_type.3ed574.2 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %x.patt.260: %pattern_type.3ed574.2 = at_binding_pattern x, %x.param_patt.f3e [symbolic]
// CHECK:STDOUT:   %assoc1.fb3: %BitXorWith.assoc_type.4ad612.1 = assoc_entity element1, imports.%Core.import_ref.63c [symbolic]
// CHECK:STDOUT:   %BitXorWith.lookup_impl_witness.28dfc5.2: <witness> = lookup_impl_witness %T.5c2, @BitXorWith.1, @BitXorWith.1(%U.67d) [symbolic]
// CHECK:STDOUT:   %BitXorWith.facet.9e4: %BitXorWith.type.7de562.2 = facet_value %T.as_type.f34, (%BitXorWith.lookup_impl_witness.28dfc5.2) [symbolic]
// CHECK:STDOUT:   %impl.elem0.9a9bb7.2: type = impl_witness_access %BitXorWith.lookup_impl_witness.28dfc5.2, element0 [symbolic]
// CHECK:STDOUT:   %BitXorWith.WithSelf.Op.type.55f: type = fn_type @BitXorWith.WithSelf.Op, @BitXorWith.WithSelf(%U.67d, %BitXorWith.facet.9e4) [symbolic]
// CHECK:STDOUT:   %.037: type = fn_type_with_self_type %BitXorWith.WithSelf.Op.type.55f, %BitXorWith.facet.9e4 [symbolic]
// CHECK:STDOUT:   %impl.elem1.01b: %.037 = impl_witness_access %BitXorWith.lookup_impl_witness.28dfc5.2, element1 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.7831: <specific function> = specific_impl_function %impl.elem1.01b, @BitXorWith.WithSelf.Op(%U.67d, %BitXorWith.facet.9e4) [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.b85: <witness> = lookup_impl_witness %impl.elem0.9a9bb7.2, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.a2d: %Destroy.type = facet_value %impl.elem0.9a9bb7.2, (%Destroy.lookup_impl_witness.b85) [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.8b0: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.a2d) [symbolic]
// CHECK:STDOUT:   %.1b3: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.8b0, %Destroy.facet.a2d [symbolic]
// CHECK:STDOUT:   %impl.elem0.7b5: %.1b3 = impl_witness_access %Destroy.lookup_impl_witness.b85, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.875: <specific function> = specific_impl_function %impl.elem0.7b5, @Destroy.WithSelf.Op(%Destroy.facet.a2d) [symbolic]
// CHECK:STDOUT:   %LeftShiftWith.type.cd59d0.1: type = facet_type <@LeftShiftWith.1, @LeftShiftWith.1(%Other)> [symbolic]
// CHECK:STDOUT:   %Self.1e32ce.1: %LeftShiftWith.type.cd59d0.1 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %LeftShiftWith.assoc_type.739d6a.1: type = assoc_entity_type @LeftShiftWith.1, @LeftShiftWith.1(%Other) [symbolic]
// CHECK:STDOUT:   %LeftShiftWith.WithSelf.Op.type.30f247.1: type = fn_type @LeftShiftWith.WithSelf.Op, @LeftShiftWith.WithSelf(%Other, %Self.1e32ce.1) [symbolic]
// CHECK:STDOUT:   %LeftShiftWith.WithSelf.Op.4ac30d.1: %LeftShiftWith.WithSelf.Op.type.30f247.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %LeftShiftWith.type.cd59d0.2: type = facet_type <@LeftShiftWith.1, @LeftShiftWith.1(%U.67d)> [symbolic]
// CHECK:STDOUT:   %.Self.e0e: %LeftShiftWith.type.cd59d0.2 = symbolic_binding .Self [symbolic]
// CHECK:STDOUT:   %LeftShiftWith.assoc_type.739d6a.2: type = assoc_entity_type @LeftShiftWith.1, @LeftShiftWith.1(%U.67d) [symbolic]
// CHECK:STDOUT:   %assoc0.0c7500.2: %LeftShiftWith.assoc_type.739d6a.2 = assoc_entity element0, imports.%Core.import_ref.1a5 [symbolic]
// CHECK:STDOUT:   %assoc1.baaa39.2: %LeftShiftWith.assoc_type.739d6a.2 = assoc_entity element1, imports.%Core.import_ref.36a [symbolic]
// CHECK:STDOUT:   %require_complete.d6f: <witness> = require_complete_type %LeftShiftWith.type.cd59d0.2 [symbolic]
// CHECK:STDOUT:   %.Self.as_type.332: type = facet_access_type %.Self.e0e [symbolic]
// CHECK:STDOUT:   %LeftShiftWith.lookup_impl_witness.b27: <witness> = lookup_impl_witness %.Self.e0e, @LeftShiftWith.1, @LeftShiftWith.1(%U.67d) [symbolic]
// CHECK:STDOUT:   %impl.elem0.554: type = impl_witness_access %LeftShiftWith.lookup_impl_witness.b27, element0 [symbolic]
// CHECK:STDOUT:   %LeftShiftWith_where.type.cd5: type = facet_type <@LeftShiftWith.1, @LeftShiftWith.1(%U.67d) where %impl.elem0.554 impls @Destroy> [symbolic]
// CHECK:STDOUT:   %pattern_type.8ee: type = pattern_type %LeftShiftWith_where.type.cd5 [symbolic]
// CHECK:STDOUT:   %T.patt.09f: %pattern_type.8ee = symbolic_binding_pattern T, 1 [symbolic]
// CHECK:STDOUT:   %T.1e3: %LeftShiftWith_where.type.cd5 = symbolic_binding T, 1 [symbolic]
// CHECK:STDOUT:   %T.as_type.9e2: type = facet_access_type %T.1e3 [symbolic]
// CHECK:STDOUT:   %pattern_type.10f0b4.2: type = pattern_type %T.as_type.9e2 [symbolic]
// CHECK:STDOUT:   %x.param_patt.6c9: %pattern_type.10f0b4.2 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %x.patt.63e: %pattern_type.10f0b4.2 = at_binding_pattern x, %x.param_patt.6c9 [symbolic]
// CHECK:STDOUT:   %assoc1.113: %LeftShiftWith.assoc_type.739d6a.1 = assoc_entity element1, imports.%Core.import_ref.5d8 [symbolic]
// CHECK:STDOUT:   %LeftShiftWith.lookup_impl_witness.209995.2: <witness> = lookup_impl_witness %T.1e3, @LeftShiftWith.1, @LeftShiftWith.1(%U.67d) [symbolic]
// CHECK:STDOUT:   %LeftShiftWith.facet.568: %LeftShiftWith.type.cd59d0.2 = facet_value %T.as_type.9e2, (%LeftShiftWith.lookup_impl_witness.209995.2) [symbolic]
// CHECK:STDOUT:   %impl.elem0.fd5c74.2: type = impl_witness_access %LeftShiftWith.lookup_impl_witness.209995.2, element0 [symbolic]
// CHECK:STDOUT:   %LeftShiftWith.WithSelf.Op.type.4a4: type = fn_type @LeftShiftWith.WithSelf.Op, @LeftShiftWith.WithSelf(%U.67d, %LeftShiftWith.facet.568) [symbolic]
// CHECK:STDOUT:   %.420: type = fn_type_with_self_type %LeftShiftWith.WithSelf.Op.type.4a4, %LeftShiftWith.facet.568 [symbolic]
// CHECK:STDOUT:   %impl.elem1.f33: %.420 = impl_witness_access %LeftShiftWith.lookup_impl_witness.209995.2, element1 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.502: <specific function> = specific_impl_function %impl.elem1.f33, @LeftShiftWith.WithSelf.Op(%U.67d, %LeftShiftWith.facet.568) [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.acb: <witness> = lookup_impl_witness %impl.elem0.fd5c74.2, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.c90: %Destroy.type = facet_value %impl.elem0.fd5c74.2, (%Destroy.lookup_impl_witness.acb) [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.fc0: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.c90) [symbolic]
// CHECK:STDOUT:   %.06e: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.fc0, %Destroy.facet.c90 [symbolic]
// CHECK:STDOUT:   %impl.elem0.94b: %.06e = impl_witness_access %Destroy.lookup_impl_witness.acb, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.418: <specific function> = specific_impl_function %impl.elem0.94b, @Destroy.WithSelf.Op(%Destroy.facet.c90) [symbolic]
// CHECK:STDOUT:   %RightShiftWith.type.60db85.1: type = facet_type <@RightShiftWith.1, @RightShiftWith.1(%Other)> [symbolic]
// CHECK:STDOUT:   %Self.665757.1: %RightShiftWith.type.60db85.1 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %RightShiftWith.assoc_type.4d2607.1: type = assoc_entity_type @RightShiftWith.1, @RightShiftWith.1(%Other) [symbolic]
// CHECK:STDOUT:   %RightShiftWith.WithSelf.Op.type.47e321.1: type = fn_type @RightShiftWith.WithSelf.Op, @RightShiftWith.WithSelf(%Other, %Self.665757.1) [symbolic]
// CHECK:STDOUT:   %RightShiftWith.WithSelf.Op.bec3f0.1: %RightShiftWith.WithSelf.Op.type.47e321.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %RightShiftWith.type.60db85.2: type = facet_type <@RightShiftWith.1, @RightShiftWith.1(%U.67d)> [symbolic]
// CHECK:STDOUT:   %.Self.7eb: %RightShiftWith.type.60db85.2 = symbolic_binding .Self [symbolic]
// CHECK:STDOUT:   %RightShiftWith.assoc_type.4d2607.2: type = assoc_entity_type @RightShiftWith.1, @RightShiftWith.1(%U.67d) [symbolic]
// CHECK:STDOUT:   %assoc0.01e147.2: %RightShiftWith.assoc_type.4d2607.2 = assoc_entity element0, imports.%Core.import_ref.416 [symbolic]
// CHECK:STDOUT:   %assoc1.53f1a1.2: %RightShiftWith.assoc_type.4d2607.2 = assoc_entity element1, imports.%Core.import_ref.e6b [symbolic]
// CHECK:STDOUT:   %require_complete.4ce: <witness> = require_complete_type %RightShiftWith.type.60db85.2 [symbolic]
// CHECK:STDOUT:   %.Self.as_type.66a: type = facet_access_type %.Self.7eb [symbolic]
// CHECK:STDOUT:   %RightShiftWith.lookup_impl_witness.d40: <witness> = lookup_impl_witness %.Self.7eb, @RightShiftWith.1, @RightShiftWith.1(%U.67d) [symbolic]
// CHECK:STDOUT:   %impl.elem0.8d0: type = impl_witness_access %RightShiftWith.lookup_impl_witness.d40, element0 [symbolic]
// CHECK:STDOUT:   %RightShiftWith_where.type.60d: type = facet_type <@RightShiftWith.1, @RightShiftWith.1(%U.67d) where %impl.elem0.8d0 impls @Destroy> [symbolic]
// CHECK:STDOUT:   %pattern_type.26c: type = pattern_type %RightShiftWith_where.type.60d [symbolic]
// CHECK:STDOUT:   %T.patt.3d2: %pattern_type.26c = symbolic_binding_pattern T, 1 [symbolic]
// CHECK:STDOUT:   %T.665: %RightShiftWith_where.type.60d = symbolic_binding T, 1 [symbolic]
// CHECK:STDOUT:   %T.as_type.e4f: type = facet_access_type %T.665 [symbolic]
// CHECK:STDOUT:   %pattern_type.8b4a4f.2: type = pattern_type %T.as_type.e4f [symbolic]
// CHECK:STDOUT:   %x.param_patt.c4e: %pattern_type.8b4a4f.2 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %x.patt.f73: %pattern_type.8b4a4f.2 = at_binding_pattern x, %x.param_patt.c4e [symbolic]
// CHECK:STDOUT:   %assoc1.c0c: %RightShiftWith.assoc_type.4d2607.1 = assoc_entity element1, imports.%Core.import_ref.e42 [symbolic]
// CHECK:STDOUT:   %RightShiftWith.lookup_impl_witness.e14634.2: <witness> = lookup_impl_witness %T.665, @RightShiftWith.1, @RightShiftWith.1(%U.67d) [symbolic]
// CHECK:STDOUT:   %RightShiftWith.facet.78e: %RightShiftWith.type.60db85.2 = facet_value %T.as_type.e4f, (%RightShiftWith.lookup_impl_witness.e14634.2) [symbolic]
// CHECK:STDOUT:   %impl.elem0.bc591e.2: type = impl_witness_access %RightShiftWith.lookup_impl_witness.e14634.2, element0 [symbolic]
// CHECK:STDOUT:   %RightShiftWith.WithSelf.Op.type.38b: type = fn_type @RightShiftWith.WithSelf.Op, @RightShiftWith.WithSelf(%U.67d, %RightShiftWith.facet.78e) [symbolic]
// CHECK:STDOUT:   %.209: type = fn_type_with_self_type %RightShiftWith.WithSelf.Op.type.38b, %RightShiftWith.facet.78e [symbolic]
// CHECK:STDOUT:   %impl.elem1.cc9: %.209 = impl_witness_access %RightShiftWith.lookup_impl_witness.e14634.2, element1 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.c69: <specific function> = specific_impl_function %impl.elem1.cc9, @RightShiftWith.WithSelf.Op(%U.67d, %RightShiftWith.facet.78e) [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.708: <witness> = lookup_impl_witness %impl.elem0.bc591e.2, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.0ac: %Destroy.type = facet_value %impl.elem0.bc591e.2, (%Destroy.lookup_impl_witness.708) [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.bb8: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.0ac) [symbolic]
// CHECK:STDOUT:   %.1f0: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.bb8, %Destroy.facet.0ac [symbolic]
// CHECK:STDOUT:   %impl.elem0.828: %.1f0 = impl_witness_access %Destroy.lookup_impl_witness.708, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.7834: <specific function> = specific_impl_function %impl.elem0.828, @Destroy.WithSelf.Op(%Destroy.facet.0ac) [symbolic]
// CHECK:STDOUT:   %Int16: type = class_type @Int16 [concrete]
// CHECK:STDOUT:   %pattern_type.785: type = pattern_type %Int16 [concrete]
// CHECK:STDOUT:   %Int32: type = class_type @Int32 [concrete]
// CHECK:STDOUT:   %pattern_type.8b45: type = pattern_type %Int32 [concrete]
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
// CHECK:STDOUT:   %BitAndWith.type.ff7: type = facet_type <@BitAndWith.1, @BitAndWith.1(%Int16)> [concrete]
// CHECK:STDOUT:   %.Self.6b1: %BitAndWith.type.ff7 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %BitAndWith.lookup_impl_witness.194: <witness> = lookup_impl_witness %.Self.6b1, @BitAndWith.1, @BitAndWith.1(%Int16) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.38f: type = impl_witness_access %BitAndWith.lookup_impl_witness.194, element0 [symbolic_self]
// CHECK:STDOUT:   %BitAndWith_where.type.ff7: type = facet_type <@BitAndWith.1, @BitAndWith.1(%Int16) where %impl.elem0.38f impls @Destroy> [concrete]
// CHECK:STDOUT:   %BitAndWith.assoc_type.33d: type = assoc_entity_type @BitAndWith.1, @BitAndWith.1(%Int16) [concrete]
// CHECK:STDOUT:   %assoc0.261: %BitAndWith.assoc_type.33d = assoc_entity element0, imports.%Core.import_ref.efd [concrete]
// CHECK:STDOUT:   %assoc1.0ca: %BitAndWith.assoc_type.33d = assoc_entity element1, imports.%Core.import_ref.2c4 [concrete]
// CHECK:STDOUT:   %Op.type.442b68.1: type = fn_type @Op.1 [concrete]
// CHECK:STDOUT:   %Op.72717c.1: %Op.type.442b68.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.703: <witness> = custom_witness (%Int16, %Op.72717c.1), @BitAndWith.1, @BitAndWith.1(%Int16) [concrete]
// CHECK:STDOUT:   %BitAndWith.facet.b2d: %BitAndWith.type.ff7 = facet_value %i16, (%custom_witness.703) [concrete]
// CHECK:STDOUT:   %Int16.Op.type.83af6e.2: type = fn_type @Int16.Op.2 [concrete]
// CHECK:STDOUT:   %Int16.Op.4dca31.2: %Int16.Op.type.83af6e.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.e7a31c.1: <witness> = custom_witness (%Int16.Op.4dca31.2), @Destroy [concrete]
// CHECK:STDOUT:   %facet_value.4da: %BitAndWith_where.type.ff7 = facet_value %i16, (%custom_witness.e7a31c.1, %custom_witness.703) [concrete]
// CHECK:STDOUT:   %complete_type.7b3: <witness> = complete_type_witness %BitAndWith.type.ff7 [concrete]
// CHECK:STDOUT:   %.Self.as_type.d50: type = facet_access_type %.Self.6b1 [symbolic_self]
// CHECK:STDOUT:   %pattern_type.9a6: type = pattern_type %BitAndWith_where.type.ff7 [concrete]
// CHECK:STDOUT:   %T.patt.30e: %pattern_type.9a6 = symbolic_binding_pattern T, 1 [symbolic]
// CHECK:STDOUT:   %y.param_patt.418: %pattern_type.785 = value_param_pattern [concrete]
// CHECK:STDOUT:   %y.patt.4e6: %pattern_type.785 = at_binding_pattern y, %y.param_patt.418 [concrete]
// CHECK:STDOUT:   %BitOrWith.type.bbb: type = facet_type <@BitOrWith.1, @BitOrWith.1(%Int16)> [concrete]
// CHECK:STDOUT:   %.Self.6b8: %BitOrWith.type.bbb = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %BitOrWith.lookup_impl_witness.08b: <witness> = lookup_impl_witness %.Self.6b8, @BitOrWith.1, @BitOrWith.1(%Int16) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.8bf: type = impl_witness_access %BitOrWith.lookup_impl_witness.08b, element0 [symbolic_self]
// CHECK:STDOUT:   %BitOrWith_where.type.bbb: type = facet_type <@BitOrWith.1, @BitOrWith.1(%Int16) where %impl.elem0.8bf impls @Destroy> [concrete]
// CHECK:STDOUT:   %BitOrWith.assoc_type.fd3: type = assoc_entity_type @BitOrWith.1, @BitOrWith.1(%Int16) [concrete]
// CHECK:STDOUT:   %assoc0.e30: %BitOrWith.assoc_type.fd3 = assoc_entity element0, imports.%Core.import_ref.4f6 [concrete]
// CHECK:STDOUT:   %assoc1.6f1: %BitOrWith.assoc_type.fd3 = assoc_entity element1, imports.%Core.import_ref.c14 [concrete]
// CHECK:STDOUT:   %Op.type.442b68.2: type = fn_type @Op.2 [concrete]
// CHECK:STDOUT:   %Op.72717c.2: %Op.type.442b68.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.55d8: <witness> = custom_witness (%Int16, %Op.72717c.2), @BitOrWith.1, @BitOrWith.1(%Int16) [concrete]
// CHECK:STDOUT:   %BitOrWith.facet.ed1: %BitOrWith.type.bbb = facet_value %i16, (%custom_witness.55d8) [concrete]
// CHECK:STDOUT:   %facet_value.a01: %BitOrWith_where.type.bbb = facet_value %i16, (%custom_witness.e7a31c.1, %custom_witness.55d8) [concrete]
// CHECK:STDOUT:   %complete_type.4d3: <witness> = complete_type_witness %BitOrWith.type.bbb [concrete]
// CHECK:STDOUT:   %.Self.as_type.cf1: type = facet_access_type %.Self.6b8 [symbolic_self]
// CHECK:STDOUT:   %pattern_type.69e: type = pattern_type %BitOrWith_where.type.bbb [concrete]
// CHECK:STDOUT:   %T.patt.476: %pattern_type.69e = symbolic_binding_pattern T, 1 [symbolic]
// CHECK:STDOUT:   %BitXorWith.type.bd3: type = facet_type <@BitXorWith.1, @BitXorWith.1(%Int16)> [concrete]
// CHECK:STDOUT:   %.Self.2ca: %BitXorWith.type.bd3 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %BitXorWith.lookup_impl_witness.c72: <witness> = lookup_impl_witness %.Self.2ca, @BitXorWith.1, @BitXorWith.1(%Int16) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.b16: type = impl_witness_access %BitXorWith.lookup_impl_witness.c72, element0 [symbolic_self]
// CHECK:STDOUT:   %BitXorWith_where.type.bd3: type = facet_type <@BitXorWith.1, @BitXorWith.1(%Int16) where %impl.elem0.b16 impls @Destroy> [concrete]
// CHECK:STDOUT:   %BitXorWith.assoc_type.679: type = assoc_entity_type @BitXorWith.1, @BitXorWith.1(%Int16) [concrete]
// CHECK:STDOUT:   %assoc0.22f: %BitXorWith.assoc_type.679 = assoc_entity element0, imports.%Core.import_ref.37d [concrete]
// CHECK:STDOUT:   %assoc1.d55: %BitXorWith.assoc_type.679 = assoc_entity element1, imports.%Core.import_ref.808 [concrete]
// CHECK:STDOUT:   %Op.type.442b68.3: type = fn_type @Op.3 [concrete]
// CHECK:STDOUT:   %Op.72717c.3: %Op.type.442b68.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.c18: <witness> = custom_witness (%Int16, %Op.72717c.3), @BitXorWith.1, @BitXorWith.1(%Int16) [concrete]
// CHECK:STDOUT:   %BitXorWith.facet.d55: %BitXorWith.type.bd3 = facet_value %i16, (%custom_witness.c18) [concrete]
// CHECK:STDOUT:   %facet_value.547: %BitXorWith_where.type.bd3 = facet_value %i16, (%custom_witness.e7a31c.1, %custom_witness.c18) [concrete]
// CHECK:STDOUT:   %complete_type.430: <witness> = complete_type_witness %BitXorWith.type.bd3 [concrete]
// CHECK:STDOUT:   %.Self.as_type.75e4: type = facet_access_type %.Self.2ca [symbolic_self]
// CHECK:STDOUT:   %pattern_type.80d: type = pattern_type %BitXorWith_where.type.bd3 [concrete]
// CHECK:STDOUT:   %T.patt.f57: %pattern_type.80d = symbolic_binding_pattern T, 1 [symbolic]
// CHECK:STDOUT:   %LeftShiftWith.type.fff: type = facet_type <@LeftShiftWith.1, @LeftShiftWith.1(%Int16)> [concrete]
// CHECK:STDOUT:   %.Self.dce: %LeftShiftWith.type.fff = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %LeftShiftWith.lookup_impl_witness.635: <witness> = lookup_impl_witness %.Self.dce, @LeftShiftWith.1, @LeftShiftWith.1(%Int16) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.df3: type = impl_witness_access %LeftShiftWith.lookup_impl_witness.635, element0 [symbolic_self]
// CHECK:STDOUT:   %LeftShiftWith_where.type.fff: type = facet_type <@LeftShiftWith.1, @LeftShiftWith.1(%Int16) where %impl.elem0.df3 impls @Destroy> [concrete]
// CHECK:STDOUT:   %LeftShiftWith.assoc_type.ee4: type = assoc_entity_type @LeftShiftWith.1, @LeftShiftWith.1(%Int16) [concrete]
// CHECK:STDOUT:   %assoc0.749: %LeftShiftWith.assoc_type.ee4 = assoc_entity element0, imports.%Core.import_ref.1a5 [concrete]
// CHECK:STDOUT:   %assoc1.7f6: %LeftShiftWith.assoc_type.ee4 = assoc_entity element1, imports.%Core.import_ref.36a [concrete]
// CHECK:STDOUT:   %Op.type.442b68.4: type = fn_type @Op.4 [concrete]
// CHECK:STDOUT:   %Op.72717c.4: %Op.type.442b68.4 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.598: <witness> = custom_witness (%Int16, %Op.72717c.4), @LeftShiftWith.1, @LeftShiftWith.1(%Int16) [concrete]
// CHECK:STDOUT:   %LeftShiftWith.facet.ce6: %LeftShiftWith.type.fff = facet_value %i16, (%custom_witness.598) [concrete]
// CHECK:STDOUT:   %facet_value.931: %LeftShiftWith_where.type.fff = facet_value %i16, (%custom_witness.e7a31c.1, %custom_witness.598) [concrete]
// CHECK:STDOUT:   %complete_type.68c: <witness> = complete_type_witness %LeftShiftWith.type.fff [concrete]
// CHECK:STDOUT:   %.Self.as_type.0bd: type = facet_access_type %.Self.dce [symbolic_self]
// CHECK:STDOUT:   %pattern_type.64f: type = pattern_type %LeftShiftWith_where.type.fff [concrete]
// CHECK:STDOUT:   %T.patt.209: %pattern_type.64f = symbolic_binding_pattern T, 1 [symbolic]
// CHECK:STDOUT:   %RightShiftWith.type.386: type = facet_type <@RightShiftWith.1, @RightShiftWith.1(%Int16)> [concrete]
// CHECK:STDOUT:   %.Self.1fe: %RightShiftWith.type.386 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %RightShiftWith.lookup_impl_witness.1b3: <witness> = lookup_impl_witness %.Self.1fe, @RightShiftWith.1, @RightShiftWith.1(%Int16) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.baa: type = impl_witness_access %RightShiftWith.lookup_impl_witness.1b3, element0 [symbolic_self]
// CHECK:STDOUT:   %RightShiftWith_where.type.386: type = facet_type <@RightShiftWith.1, @RightShiftWith.1(%Int16) where %impl.elem0.baa impls @Destroy> [concrete]
// CHECK:STDOUT:   %RightShiftWith.assoc_type.656: type = assoc_entity_type @RightShiftWith.1, @RightShiftWith.1(%Int16) [concrete]
// CHECK:STDOUT:   %assoc0.148: %RightShiftWith.assoc_type.656 = assoc_entity element0, imports.%Core.import_ref.416 [concrete]
// CHECK:STDOUT:   %assoc1.5a2: %RightShiftWith.assoc_type.656 = assoc_entity element1, imports.%Core.import_ref.e6b [concrete]
// CHECK:STDOUT:   %Op.type.442b68.5: type = fn_type @Op.5 [concrete]
// CHECK:STDOUT:   %Op.72717c.5: %Op.type.442b68.5 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.24d: <witness> = custom_witness (%Int16, %Op.72717c.5), @RightShiftWith.1, @RightShiftWith.1(%Int16) [concrete]
// CHECK:STDOUT:   %RightShiftWith.facet.4d1: %RightShiftWith.type.386 = facet_value %i16, (%custom_witness.24d) [concrete]
// CHECK:STDOUT:   %facet_value.1bd: %RightShiftWith_where.type.386 = facet_value %i16, (%custom_witness.e7a31c.1, %custom_witness.24d) [concrete]
// CHECK:STDOUT:   %complete_type.938: <witness> = complete_type_witness %RightShiftWith.type.386 [concrete]
// CHECK:STDOUT:   %.Self.as_type.7d1: type = facet_access_type %.Self.1fe [symbolic_self]
// CHECK:STDOUT:   %pattern_type.761: type = pattern_type %RightShiftWith_where.type.386 [concrete]
// CHECK:STDOUT:   %T.patt.806: %pattern_type.761 = symbolic_binding_pattern T, 1 [symbolic]
// CHECK:STDOUT:   %BitAndWith.type.569: type = facet_type <@BitAndWith.1, @BitAndWith.1(%i16)> [concrete]
// CHECK:STDOUT:   %.Self.1ec: %BitAndWith.type.569 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %BitAndWith.lookup_impl_witness.dcb: <witness> = lookup_impl_witness %.Self.1ec, @BitAndWith.1, @BitAndWith.1(%i16) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.c1c: type = impl_witness_access %BitAndWith.lookup_impl_witness.dcb, element0 [symbolic_self]
// CHECK:STDOUT:   %BitAndWith_where.type.569: type = facet_type <@BitAndWith.1, @BitAndWith.1(%i16) where %impl.elem0.c1c impls @Destroy> [concrete]
// CHECK:STDOUT:   %BitAndWith.assoc_type.419: type = assoc_entity_type @BitAndWith.1, @BitAndWith.1(%i16) [concrete]
// CHECK:STDOUT:   %assoc0.bb5: %BitAndWith.assoc_type.419 = assoc_entity element0, imports.%Core.import_ref.efd [concrete]
// CHECK:STDOUT:   %assoc1.38a: %BitAndWith.assoc_type.419 = assoc_entity element1, imports.%Core.import_ref.2c4 [concrete]
// CHECK:STDOUT:   %Int16.Op.type.83af6e.3: type = fn_type @Int16.Op.3 [concrete]
// CHECK:STDOUT:   %Int16.Op.4dca31.3: %Int16.Op.type.83af6e.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.e77: <witness> = custom_witness (%Int16, %Int16.Op.4dca31.3), @BitAndWith.1, @BitAndWith.1(%i16) [concrete]
// CHECK:STDOUT:   %BitAndWith.facet.064: %BitAndWith.type.569 = facet_value %Int16, (%custom_witness.e77) [concrete]
// CHECK:STDOUT:   %facet_value.e6c: %BitAndWith_where.type.569 = facet_value %Int16, (%custom_witness.e77, %custom_witness.e7a31c.1) [concrete]
// CHECK:STDOUT:   %complete_type.c28: <witness> = complete_type_witness %BitAndWith.type.569 [concrete]
// CHECK:STDOUT:   %.Self.as_type.986: type = facet_access_type %.Self.1ec [symbolic_self]
// CHECK:STDOUT:   %pattern_type.06f: type = pattern_type %BitAndWith_where.type.569 [concrete]
// CHECK:STDOUT:   %T.patt.2aa: %pattern_type.06f = symbolic_binding_pattern T, 1 [symbolic]
// CHECK:STDOUT:   %x.param_patt.3b0: %pattern_type.785 = value_param_pattern [concrete]
// CHECK:STDOUT:   %x.patt.b76: %pattern_type.785 = at_binding_pattern x, %x.param_patt.3b0 [concrete]
// CHECK:STDOUT:   %y.param_patt.4b4: %pattern_type.203 = value_param_pattern [concrete]
// CHECK:STDOUT:   %y.patt.dd3: %pattern_type.203 = at_binding_pattern y, %y.param_patt.4b4 [concrete]
// CHECK:STDOUT:   %BitOrWith.type.e34: type = facet_type <@BitOrWith.1, @BitOrWith.1(%i16)> [concrete]
// CHECK:STDOUT:   %.Self.5cb: %BitOrWith.type.e34 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %BitOrWith.lookup_impl_witness.946: <witness> = lookup_impl_witness %.Self.5cb, @BitOrWith.1, @BitOrWith.1(%i16) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.5b0: type = impl_witness_access %BitOrWith.lookup_impl_witness.946, element0 [symbolic_self]
// CHECK:STDOUT:   %BitOrWith_where.type.e34: type = facet_type <@BitOrWith.1, @BitOrWith.1(%i16) where %impl.elem0.5b0 impls @Destroy> [concrete]
// CHECK:STDOUT:   %BitOrWith.assoc_type.2db: type = assoc_entity_type @BitOrWith.1, @BitOrWith.1(%i16) [concrete]
// CHECK:STDOUT:   %assoc0.b52: %BitOrWith.assoc_type.2db = assoc_entity element0, imports.%Core.import_ref.4f6 [concrete]
// CHECK:STDOUT:   %assoc1.90e: %BitOrWith.assoc_type.2db = assoc_entity element1, imports.%Core.import_ref.c14 [concrete]
// CHECK:STDOUT:   %Int16.Op.type.83af6e.4: type = fn_type @Int16.Op.4 [concrete]
// CHECK:STDOUT:   %Int16.Op.4dca31.4: %Int16.Op.type.83af6e.4 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.9c0: <witness> = custom_witness (%Int16, %Int16.Op.4dca31.4), @BitOrWith.1, @BitOrWith.1(%i16) [concrete]
// CHECK:STDOUT:   %BitOrWith.facet.bab: %BitOrWith.type.e34 = facet_value %Int16, (%custom_witness.9c0) [concrete]
// CHECK:STDOUT:   %facet_value.d1b: %BitOrWith_where.type.e34 = facet_value %Int16, (%custom_witness.e7a31c.1, %custom_witness.9c0) [concrete]
// CHECK:STDOUT:   %complete_type.440: <witness> = complete_type_witness %BitOrWith.type.e34 [concrete]
// CHECK:STDOUT:   %.Self.as_type.91b: type = facet_access_type %.Self.5cb [symbolic_self]
// CHECK:STDOUT:   %pattern_type.6f1: type = pattern_type %BitOrWith_where.type.e34 [concrete]
// CHECK:STDOUT:   %T.patt.273: %pattern_type.6f1 = symbolic_binding_pattern T, 1 [symbolic]
// CHECK:STDOUT:   %BitXorWith.type.716: type = facet_type <@BitXorWith.1, @BitXorWith.1(%i16)> [concrete]
// CHECK:STDOUT:   %.Self.ca9: %BitXorWith.type.716 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %BitXorWith.lookup_impl_witness.220: <witness> = lookup_impl_witness %.Self.ca9, @BitXorWith.1, @BitXorWith.1(%i16) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.260: type = impl_witness_access %BitXorWith.lookup_impl_witness.220, element0 [symbolic_self]
// CHECK:STDOUT:   %BitXorWith_where.type.716: type = facet_type <@BitXorWith.1, @BitXorWith.1(%i16) where %impl.elem0.260 impls @Destroy> [concrete]
// CHECK:STDOUT:   %BitXorWith.assoc_type.a40: type = assoc_entity_type @BitXorWith.1, @BitXorWith.1(%i16) [concrete]
// CHECK:STDOUT:   %assoc0.b33: %BitXorWith.assoc_type.a40 = assoc_entity element0, imports.%Core.import_ref.37d [concrete]
// CHECK:STDOUT:   %assoc1.9b1: %BitXorWith.assoc_type.a40 = assoc_entity element1, imports.%Core.import_ref.808 [concrete]
// CHECK:STDOUT:   %Int16.Op.type.83af6e.5: type = fn_type @Int16.Op.5 [concrete]
// CHECK:STDOUT:   %Int16.Op.4dca31.5: %Int16.Op.type.83af6e.5 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.3ee: <witness> = custom_witness (%Int16, %Int16.Op.4dca31.5), @BitXorWith.1, @BitXorWith.1(%i16) [concrete]
// CHECK:STDOUT:   %BitXorWith.facet.7da: %BitXorWith.type.716 = facet_value %Int16, (%custom_witness.3ee) [concrete]
// CHECK:STDOUT:   %facet_value.4f5: %BitXorWith_where.type.716 = facet_value %Int16, (%custom_witness.e7a31c.1, %custom_witness.3ee) [concrete]
// CHECK:STDOUT:   %complete_type.5e0: <witness> = complete_type_witness %BitXorWith.type.716 [concrete]
// CHECK:STDOUT:   %.Self.as_type.684: type = facet_access_type %.Self.ca9 [symbolic_self]
// CHECK:STDOUT:   %pattern_type.4c9: type = pattern_type %BitXorWith_where.type.716 [concrete]
// CHECK:STDOUT:   %T.patt.187: %pattern_type.4c9 = symbolic_binding_pattern T, 1 [symbolic]
// CHECK:STDOUT:   %LeftShiftWith.type.0ee: type = facet_type <@LeftShiftWith.1, @LeftShiftWith.1(%i16)> [concrete]
// CHECK:STDOUT:   %.Self.859: %LeftShiftWith.type.0ee = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %LeftShiftWith.lookup_impl_witness.a61: <witness> = lookup_impl_witness %.Self.859, @LeftShiftWith.1, @LeftShiftWith.1(%i16) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.4d6: type = impl_witness_access %LeftShiftWith.lookup_impl_witness.a61, element0 [symbolic_self]
// CHECK:STDOUT:   %LeftShiftWith_where.type.0ee: type = facet_type <@LeftShiftWith.1, @LeftShiftWith.1(%i16) where %impl.elem0.4d6 impls @Destroy> [concrete]
// CHECK:STDOUT:   %LeftShiftWith.assoc_type.fae: type = assoc_entity_type @LeftShiftWith.1, @LeftShiftWith.1(%i16) [concrete]
// CHECK:STDOUT:   %assoc0.78f: %LeftShiftWith.assoc_type.fae = assoc_entity element0, imports.%Core.import_ref.1a5 [concrete]
// CHECK:STDOUT:   %assoc1.a06: %LeftShiftWith.assoc_type.fae = assoc_entity element1, imports.%Core.import_ref.36a [concrete]
// CHECK:STDOUT:   %Int16.Op.type.83af6e.6: type = fn_type @Int16.Op.6 [concrete]
// CHECK:STDOUT:   %Int16.Op.4dca31.6: %Int16.Op.type.83af6e.6 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.e12: <witness> = custom_witness (%Int16, %Int16.Op.4dca31.6), @LeftShiftWith.1, @LeftShiftWith.1(%i16) [concrete]
// CHECK:STDOUT:   %LeftShiftWith.facet.071: %LeftShiftWith.type.0ee = facet_value %Int16, (%custom_witness.e12) [concrete]
// CHECK:STDOUT:   %facet_value.f61: %LeftShiftWith_where.type.0ee = facet_value %Int16, (%custom_witness.e7a31c.1, %custom_witness.e12) [concrete]
// CHECK:STDOUT:   %complete_type.bdc: <witness> = complete_type_witness %LeftShiftWith.type.0ee [concrete]
// CHECK:STDOUT:   %.Self.as_type.81e: type = facet_access_type %.Self.859 [symbolic_self]
// CHECK:STDOUT:   %pattern_type.117: type = pattern_type %LeftShiftWith_where.type.0ee [concrete]
// CHECK:STDOUT:   %T.patt.985: %pattern_type.117 = symbolic_binding_pattern T, 1 [symbolic]
// CHECK:STDOUT:   %RightShiftWith.type.f4e: type = facet_type <@RightShiftWith.1, @RightShiftWith.1(%i16)> [concrete]
// CHECK:STDOUT:   %.Self.487: %RightShiftWith.type.f4e = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %RightShiftWith.lookup_impl_witness.f7c: <witness> = lookup_impl_witness %.Self.487, @RightShiftWith.1, @RightShiftWith.1(%i16) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.54e: type = impl_witness_access %RightShiftWith.lookup_impl_witness.f7c, element0 [symbolic_self]
// CHECK:STDOUT:   %RightShiftWith_where.type.f4e: type = facet_type <@RightShiftWith.1, @RightShiftWith.1(%i16) where %impl.elem0.54e impls @Destroy> [concrete]
// CHECK:STDOUT:   %RightShiftWith.assoc_type.b9f: type = assoc_entity_type @RightShiftWith.1, @RightShiftWith.1(%i16) [concrete]
// CHECK:STDOUT:   %assoc0.d8e: %RightShiftWith.assoc_type.b9f = assoc_entity element0, imports.%Core.import_ref.416 [concrete]
// CHECK:STDOUT:   %assoc1.1d4: %RightShiftWith.assoc_type.b9f = assoc_entity element1, imports.%Core.import_ref.e6b [concrete]
// CHECK:STDOUT:   %Int16.Op.type.83af6e.7: type = fn_type @Int16.Op.7 [concrete]
// CHECK:STDOUT:   %Int16.Op.4dca31.7: %Int16.Op.type.83af6e.7 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.ba0: <witness> = custom_witness (%Int16, %Int16.Op.4dca31.7), @RightShiftWith.1, @RightShiftWith.1(%i16) [concrete]
// CHECK:STDOUT:   %RightShiftWith.facet.fa8: %RightShiftWith.type.f4e = facet_value %Int16, (%custom_witness.ba0) [concrete]
// CHECK:STDOUT:   %facet_value.475: %RightShiftWith_where.type.f4e = facet_value %Int16, (%custom_witness.e7a31c.1, %custom_witness.ba0) [concrete]
// CHECK:STDOUT:   %complete_type.5ea: <witness> = complete_type_witness %RightShiftWith.type.f4e [concrete]
// CHECK:STDOUT:   %.Self.as_type.73f: type = facet_access_type %.Self.487 [symbolic_self]
// CHECK:STDOUT:   %pattern_type.9cd: type = pattern_type %RightShiftWith_where.type.f4e [concrete]
// CHECK:STDOUT:   %T.patt.9f0: %pattern_type.9cd = symbolic_binding_pattern T, 1 [symbolic]
// CHECK:STDOUT:   %BitAndWith.type.f43: type = facet_type <@BitAndWith.1, @BitAndWith.1(%Int32)> [concrete]
// CHECK:STDOUT:   %.Self.b06: %BitAndWith.type.f43 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %BitAndWith.lookup_impl_witness.773: <witness> = lookup_impl_witness %.Self.b06, @BitAndWith.1, @BitAndWith.1(%Int32) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.b5e: type = impl_witness_access %BitAndWith.lookup_impl_witness.773, element0 [symbolic_self]
// CHECK:STDOUT:   %BitAndWith_where.type.f43: type = facet_type <@BitAndWith.1, @BitAndWith.1(%Int32) where %impl.elem0.b5e impls @Destroy> [concrete]
// CHECK:STDOUT:   %BitAndWith.assoc_type.6e7: type = assoc_entity_type @BitAndWith.1, @BitAndWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %assoc0.543: %BitAndWith.assoc_type.6e7 = assoc_entity element0, imports.%Core.import_ref.efd [concrete]
// CHECK:STDOUT:   %assoc1.5b1: %BitAndWith.assoc_type.6e7 = assoc_entity element1, imports.%Core.import_ref.2c4 [concrete]
// CHECK:STDOUT:   %Int16.Op.type.83af6e.8: type = fn_type @Int16.Op.8 [concrete]
// CHECK:STDOUT:   %Int16.Op.4dca31.8: %Int16.Op.type.83af6e.8 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.9a1dfa.1: <witness> = custom_witness (%Int32, %Int16.Op.4dca31.8), @BitAndWith.1, @BitAndWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %BitAndWith.facet.fc0: %BitAndWith.type.f43 = facet_value %Int16, (%custom_witness.9a1dfa.1) [concrete]
// CHECK:STDOUT:   %Int32.Op.type.83af6e.1: type = fn_type @Int32.Op.1 [concrete]
// CHECK:STDOUT:   %Int32.Op.4dca31.1: %Int32.Op.type.83af6e.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.e7a31c.2: <witness> = custom_witness (%Int32.Op.4dca31.1), @Destroy [concrete]
// CHECK:STDOUT:   %facet_value.e6f: %BitAndWith_where.type.f43 = facet_value %Int16, (%custom_witness.9a1dfa.1, %custom_witness.e7a31c.2) [concrete]
// CHECK:STDOUT:   %complete_type.008: <witness> = complete_type_witness %BitAndWith.type.f43 [concrete]
// CHECK:STDOUT:   %.Self.as_type.b4c: type = facet_access_type %.Self.b06 [symbolic_self]
// CHECK:STDOUT:   %pattern_type.f97: type = pattern_type %BitAndWith_where.type.f43 [concrete]
// CHECK:STDOUT:   %T.patt.126: %pattern_type.f97 = symbolic_binding_pattern T, 1 [symbolic]
// CHECK:STDOUT:   %y.param_patt.e0c: %pattern_type.8b45 = value_param_pattern [concrete]
// CHECK:STDOUT:   %y.patt.0a3: %pattern_type.8b45 = at_binding_pattern y, %y.param_patt.e0c [concrete]
// CHECK:STDOUT:   %BitOrWith.type.878: type = facet_type <@BitOrWith.1, @BitOrWith.1(%Int32)> [concrete]
// CHECK:STDOUT:   %.Self.f1c: %BitOrWith.type.878 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %BitOrWith.lookup_impl_witness.93b: <witness> = lookup_impl_witness %.Self.f1c, @BitOrWith.1, @BitOrWith.1(%Int32) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.181: type = impl_witness_access %BitOrWith.lookup_impl_witness.93b, element0 [symbolic_self]
// CHECK:STDOUT:   %BitOrWith_where.type.878: type = facet_type <@BitOrWith.1, @BitOrWith.1(%Int32) where %impl.elem0.181 impls @Destroy> [concrete]
// CHECK:STDOUT:   %BitOrWith.assoc_type.0490: type = assoc_entity_type @BitOrWith.1, @BitOrWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %assoc0.d36: %BitOrWith.assoc_type.0490 = assoc_entity element0, imports.%Core.import_ref.4f6 [concrete]
// CHECK:STDOUT:   %assoc1.953: %BitOrWith.assoc_type.0490 = assoc_entity element1, imports.%Core.import_ref.c14 [concrete]
// CHECK:STDOUT:   %Int16.Op.type.83af6e.9: type = fn_type @Int16.Op.9 [concrete]
// CHECK:STDOUT:   %Int16.Op.4dca31.9: %Int16.Op.type.83af6e.9 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.2e067a.1: <witness> = custom_witness (%Int32, %Int16.Op.4dca31.9), @BitOrWith.1, @BitOrWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %BitOrWith.facet.da7: %BitOrWith.type.878 = facet_value %Int16, (%custom_witness.2e067a.1) [concrete]
// CHECK:STDOUT:   %facet_value.b44: %BitOrWith_where.type.878 = facet_value %Int16, (%custom_witness.2e067a.1, %custom_witness.e7a31c.2) [concrete]
// CHECK:STDOUT:   %complete_type.4fe: <witness> = complete_type_witness %BitOrWith.type.878 [concrete]
// CHECK:STDOUT:   %.Self.as_type.4f4: type = facet_access_type %.Self.f1c [symbolic_self]
// CHECK:STDOUT:   %pattern_type.a2c: type = pattern_type %BitOrWith_where.type.878 [concrete]
// CHECK:STDOUT:   %T.patt.3c3: %pattern_type.a2c = symbolic_binding_pattern T, 1 [symbolic]
// CHECK:STDOUT:   %BitXorWith.type.729: type = facet_type <@BitXorWith.1, @BitXorWith.1(%Int32)> [concrete]
// CHECK:STDOUT:   %.Self.be9: %BitXorWith.type.729 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %BitXorWith.lookup_impl_witness.de4: <witness> = lookup_impl_witness %.Self.be9, @BitXorWith.1, @BitXorWith.1(%Int32) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.1ab: type = impl_witness_access %BitXorWith.lookup_impl_witness.de4, element0 [symbolic_self]
// CHECK:STDOUT:   %BitXorWith_where.type.729: type = facet_type <@BitXorWith.1, @BitXorWith.1(%Int32) where %impl.elem0.1ab impls @Destroy> [concrete]
// CHECK:STDOUT:   %BitXorWith.assoc_type.36a: type = assoc_entity_type @BitXorWith.1, @BitXorWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %assoc0.0ab: %BitXorWith.assoc_type.36a = assoc_entity element0, imports.%Core.import_ref.37d [concrete]
// CHECK:STDOUT:   %assoc1.951: %BitXorWith.assoc_type.36a = assoc_entity element1, imports.%Core.import_ref.808 [concrete]
// CHECK:STDOUT:   %Int16.Op.type.83af6e.10: type = fn_type @Int16.Op.10 [concrete]
// CHECK:STDOUT:   %Int16.Op.4dca31.10: %Int16.Op.type.83af6e.10 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.5d016d.1: <witness> = custom_witness (%Int32, %Int16.Op.4dca31.10), @BitXorWith.1, @BitXorWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %BitXorWith.facet.121: %BitXorWith.type.729 = facet_value %Int16, (%custom_witness.5d016d.1) [concrete]
// CHECK:STDOUT:   %facet_value.857: %BitXorWith_where.type.729 = facet_value %Int16, (%custom_witness.5d016d.1, %custom_witness.e7a31c.2) [concrete]
// CHECK:STDOUT:   %complete_type.8eb: <witness> = complete_type_witness %BitXorWith.type.729 [concrete]
// CHECK:STDOUT:   %.Self.as_type.192: type = facet_access_type %.Self.be9 [symbolic_self]
// CHECK:STDOUT:   %pattern_type.ab4: type = pattern_type %BitXorWith_where.type.729 [concrete]
// CHECK:STDOUT:   %T.patt.1c4: %pattern_type.ab4 = symbolic_binding_pattern T, 1 [symbolic]
// CHECK:STDOUT:   %LeftShiftWith.type.15a: type = facet_type <@LeftShiftWith.1, @LeftShiftWith.1(%Int32)> [concrete]
// CHECK:STDOUT:   %.Self.b90: %LeftShiftWith.type.15a = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %LeftShiftWith.lookup_impl_witness.140: <witness> = lookup_impl_witness %.Self.b90, @LeftShiftWith.1, @LeftShiftWith.1(%Int32) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.e18: type = impl_witness_access %LeftShiftWith.lookup_impl_witness.140, element0 [symbolic_self]
// CHECK:STDOUT:   %LeftShiftWith_where.type.15a: type = facet_type <@LeftShiftWith.1, @LeftShiftWith.1(%Int32) where %impl.elem0.e18 impls @Destroy> [concrete]
// CHECK:STDOUT:   %LeftShiftWith.assoc_type.e1d: type = assoc_entity_type @LeftShiftWith.1, @LeftShiftWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %assoc0.6e4: %LeftShiftWith.assoc_type.e1d = assoc_entity element0, imports.%Core.import_ref.1a5 [concrete]
// CHECK:STDOUT:   %assoc1.f2a: %LeftShiftWith.assoc_type.e1d = assoc_entity element1, imports.%Core.import_ref.36a [concrete]
// CHECK:STDOUT:   %Int16.Op.type.83af6e.11: type = fn_type @Int16.Op.11 [concrete]
// CHECK:STDOUT:   %Int16.Op.4dca31.11: %Int16.Op.type.83af6e.11 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.65d217.1: <witness> = custom_witness (%Int32, %Int16.Op.4dca31.11), @LeftShiftWith.1, @LeftShiftWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %LeftShiftWith.facet.4c3e: %LeftShiftWith.type.15a = facet_value %Int16, (%custom_witness.65d217.1) [concrete]
// CHECK:STDOUT:   %facet_value.d98: %LeftShiftWith_where.type.15a = facet_value %Int16, (%custom_witness.65d217.1, %custom_witness.e7a31c.2) [concrete]
// CHECK:STDOUT:   %complete_type.5b7: <witness> = complete_type_witness %LeftShiftWith.type.15a [concrete]
// CHECK:STDOUT:   %.Self.as_type.b53: type = facet_access_type %.Self.b90 [symbolic_self]
// CHECK:STDOUT:   %pattern_type.538: type = pattern_type %LeftShiftWith_where.type.15a [concrete]
// CHECK:STDOUT:   %T.patt.4a0: %pattern_type.538 = symbolic_binding_pattern T, 1 [symbolic]
// CHECK:STDOUT:   %RightShiftWith.type.8f0: type = facet_type <@RightShiftWith.1, @RightShiftWith.1(%Int32)> [concrete]
// CHECK:STDOUT:   %.Self.c91: %RightShiftWith.type.8f0 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %RightShiftWith.lookup_impl_witness.bb0: <witness> = lookup_impl_witness %.Self.c91, @RightShiftWith.1, @RightShiftWith.1(%Int32) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.cfe: type = impl_witness_access %RightShiftWith.lookup_impl_witness.bb0, element0 [symbolic_self]
// CHECK:STDOUT:   %RightShiftWith_where.type.8f0: type = facet_type <@RightShiftWith.1, @RightShiftWith.1(%Int32) where %impl.elem0.cfe impls @Destroy> [concrete]
// CHECK:STDOUT:   %RightShiftWith.assoc_type.813: type = assoc_entity_type @RightShiftWith.1, @RightShiftWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %assoc0.99a: %RightShiftWith.assoc_type.813 = assoc_entity element0, imports.%Core.import_ref.416 [concrete]
// CHECK:STDOUT:   %assoc1.2c8: %RightShiftWith.assoc_type.813 = assoc_entity element1, imports.%Core.import_ref.e6b [concrete]
// CHECK:STDOUT:   %Int16.Op.type.83af6e.12: type = fn_type @Int16.Op.12 [concrete]
// CHECK:STDOUT:   %Int16.Op.4dca31.12: %Int16.Op.type.83af6e.12 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.884138.1: <witness> = custom_witness (%Int32, %Int16.Op.4dca31.12), @RightShiftWith.1, @RightShiftWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %RightShiftWith.facet.0ca: %RightShiftWith.type.8f0 = facet_value %Int16, (%custom_witness.884138.1) [concrete]
// CHECK:STDOUT:   %facet_value.431: %RightShiftWith_where.type.8f0 = facet_value %Int16, (%custom_witness.884138.1, %custom_witness.e7a31c.2) [concrete]
// CHECK:STDOUT:   %complete_type.93f: <witness> = complete_type_witness %RightShiftWith.type.8f0 [concrete]
// CHECK:STDOUT:   %.Self.as_type.d6f: type = facet_access_type %.Self.c91 [symbolic_self]
// CHECK:STDOUT:   %pattern_type.5a0: type = pattern_type %RightShiftWith_where.type.8f0 [concrete]
// CHECK:STDOUT:   %T.patt.a69: %pattern_type.5a0 = symbolic_binding_pattern T, 1 [symbolic]
// CHECK:STDOUT:   %Int32.Op.type.83af6e.2: type = fn_type @Int32.Op.2 [concrete]
// CHECK:STDOUT:   %Int32.Op.4dca31.2: %Int32.Op.type.83af6e.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.25b: <witness> = custom_witness (%Int32, %Int32.Op.4dca31.2), @BitAndWith.1, @BitAndWith.1(%Int16) [concrete]
// CHECK:STDOUT:   %BitAndWith.facet.58a: %BitAndWith.type.ff7 = facet_value %Int32, (%custom_witness.25b) [concrete]
// CHECK:STDOUT:   %facet_value.454: %BitAndWith_where.type.ff7 = facet_value %Int32, (%custom_witness.25b, %custom_witness.e7a31c.2) [concrete]
// CHECK:STDOUT:   %x.param_patt.27c: %pattern_type.8b45 = value_param_pattern [concrete]
// CHECK:STDOUT:   %x.patt.4c9: %pattern_type.8b45 = at_binding_pattern x, %x.param_patt.27c [concrete]
// CHECK:STDOUT:   %Int32.Op.type.83af6e.3: type = fn_type @Int32.Op.3 [concrete]
// CHECK:STDOUT:   %Int32.Op.4dca31.3: %Int32.Op.type.83af6e.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.205: <witness> = custom_witness (%Int32, %Int32.Op.4dca31.3), @BitOrWith.1, @BitOrWith.1(%Int16) [concrete]
// CHECK:STDOUT:   %BitOrWith.facet.c7b: %BitOrWith.type.bbb = facet_value %Int32, (%custom_witness.205) [concrete]
// CHECK:STDOUT:   %facet_value.d4b: %BitOrWith_where.type.bbb = facet_value %Int32, (%custom_witness.e7a31c.2, %custom_witness.205) [concrete]
// CHECK:STDOUT:   %Int32.Op.type.83af6e.4: type = fn_type @Int32.Op.4 [concrete]
// CHECK:STDOUT:   %Int32.Op.4dca31.4: %Int32.Op.type.83af6e.4 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.cc2: <witness> = custom_witness (%Int32, %Int32.Op.4dca31.4), @BitXorWith.1, @BitXorWith.1(%Int16) [concrete]
// CHECK:STDOUT:   %BitXorWith.facet.ac3: %BitXorWith.type.bd3 = facet_value %Int32, (%custom_witness.cc2) [concrete]
// CHECK:STDOUT:   %facet_value.fd3: %BitXorWith_where.type.bd3 = facet_value %Int32, (%custom_witness.e7a31c.2, %custom_witness.cc2) [concrete]
// CHECK:STDOUT:   %Int32.Op.type.83af6e.5: type = fn_type @Int32.Op.5 [concrete]
// CHECK:STDOUT:   %Int32.Op.4dca31.5: %Int32.Op.type.83af6e.5 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.672: <witness> = custom_witness (%Int32, %Int32.Op.4dca31.5), @LeftShiftWith.1, @LeftShiftWith.1(%Int16) [concrete]
// CHECK:STDOUT:   %LeftShiftWith.facet.453: %LeftShiftWith.type.fff = facet_value %Int32, (%custom_witness.672) [concrete]
// CHECK:STDOUT:   %facet_value.281: %LeftShiftWith_where.type.fff = facet_value %Int32, (%custom_witness.e7a31c.2, %custom_witness.672) [concrete]
// CHECK:STDOUT:   %Int32.Op.type.83af6e.6: type = fn_type @Int32.Op.6 [concrete]
// CHECK:STDOUT:   %Int32.Op.4dca31.6: %Int32.Op.type.83af6e.6 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.9fa: <witness> = custom_witness (%Int32, %Int32.Op.4dca31.6), @RightShiftWith.1, @RightShiftWith.1(%Int16) [concrete]
// CHECK:STDOUT:   %RightShiftWith.facet.714: %RightShiftWith.type.386 = facet_value %Int32, (%custom_witness.9fa) [concrete]
// CHECK:STDOUT:   %facet_value.65f: %RightShiftWith_where.type.386 = facet_value %Int32, (%custom_witness.e7a31c.2, %custom_witness.9fa) [concrete]
// CHECK:STDOUT:   %Int32.Op.type.83af6e.7: type = fn_type @Int32.Op.7 [concrete]
// CHECK:STDOUT:   %Int32.Op.4dca31.7: %Int32.Op.type.83af6e.7 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.9a1dfa.2: <witness> = custom_witness (%Int32, %Int32.Op.4dca31.7), @BitAndWith.1, @BitAndWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %BitAndWith.facet.2d7: %BitAndWith.type.f43 = facet_value %Int32, (%custom_witness.9a1dfa.2) [concrete]
// CHECK:STDOUT:   %facet_value.35c: %BitAndWith_where.type.f43 = facet_value %Int32, (%custom_witness.9a1dfa.2, %custom_witness.e7a31c.2) [concrete]
// CHECK:STDOUT:   %Int32.Op.type.83af6e.8: type = fn_type @Int32.Op.8 [concrete]
// CHECK:STDOUT:   %Int32.Op.4dca31.8: %Int32.Op.type.83af6e.8 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.2e067a.2: <witness> = custom_witness (%Int32, %Int32.Op.4dca31.8), @BitOrWith.1, @BitOrWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %BitOrWith.facet.f97: %BitOrWith.type.878 = facet_value %Int32, (%custom_witness.2e067a.2) [concrete]
// CHECK:STDOUT:   %facet_value.51e: %BitOrWith_where.type.878 = facet_value %Int32, (%custom_witness.e7a31c.2, %custom_witness.2e067a.2) [concrete]
// CHECK:STDOUT:   %Int32.Op.type.83af6e.9: type = fn_type @Int32.Op.9 [concrete]
// CHECK:STDOUT:   %Int32.Op.4dca31.9: %Int32.Op.type.83af6e.9 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.5d016d.2: <witness> = custom_witness (%Int32, %Int32.Op.4dca31.9), @BitXorWith.1, @BitXorWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %BitXorWith.facet.4e5: %BitXorWith.type.729 = facet_value %Int32, (%custom_witness.5d016d.2) [concrete]
// CHECK:STDOUT:   %facet_value.bdf: %BitXorWith_where.type.729 = facet_value %Int32, (%custom_witness.e7a31c.2, %custom_witness.5d016d.2) [concrete]
// CHECK:STDOUT:   %Int32.Op.type.83af6e.10: type = fn_type @Int32.Op.10 [concrete]
// CHECK:STDOUT:   %Int32.Op.4dca31.10: %Int32.Op.type.83af6e.10 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.65d217.2: <witness> = custom_witness (%Int32, %Int32.Op.4dca31.10), @LeftShiftWith.1, @LeftShiftWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %LeftShiftWith.facet.0bb: %LeftShiftWith.type.15a = facet_value %Int32, (%custom_witness.65d217.2) [concrete]
// CHECK:STDOUT:   %facet_value.dcd: %LeftShiftWith_where.type.15a = facet_value %Int32, (%custom_witness.e7a31c.2, %custom_witness.65d217.2) [concrete]
// CHECK:STDOUT:   %Int32.Op.type.83af6e.11: type = fn_type @Int32.Op.11 [concrete]
// CHECK:STDOUT:   %Int32.Op.4dca31.11: %Int32.Op.type.83af6e.11 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.884138.2: <witness> = custom_witness (%Int32, %Int32.Op.4dca31.11), @RightShiftWith.1, @RightShiftWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %RightShiftWith.facet.0bb: %RightShiftWith.type.8f0 = facet_value %Int32, (%custom_witness.884138.2) [concrete]
// CHECK:STDOUT:   %facet_value.624: %RightShiftWith_where.type.8f0 = facet_value %Int32, (%custom_witness.e7a31c.2, %custom_witness.884138.2) [concrete]
// CHECK:STDOUT:   %BitAndWith.type.0cc: type = facet_type <@BitAndWith.1, @BitAndWith.1(%Int64)> [concrete]
// CHECK:STDOUT:   %.Self.bfc: %BitAndWith.type.0cc = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %BitAndWith.lookup_impl_witness.bf6: <witness> = lookup_impl_witness %.Self.bfc, @BitAndWith.1, @BitAndWith.1(%Int64) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.cbe: type = impl_witness_access %BitAndWith.lookup_impl_witness.bf6, element0 [symbolic_self]
// CHECK:STDOUT:   %BitAndWith_where.type.0cc: type = facet_type <@BitAndWith.1, @BitAndWith.1(%Int64) where %impl.elem0.cbe impls @Destroy> [concrete]
// CHECK:STDOUT:   %y.param_patt.a08: %pattern_type.4aa = value_param_pattern [concrete]
// CHECK:STDOUT:   %y.patt.63b: %pattern_type.4aa = at_binding_pattern y, %y.param_patt.a08 [concrete]
// CHECK:STDOUT:   %BitAndWith.assoc_type.34a: type = assoc_entity_type @BitAndWith.1, @BitAndWith.1(%Int64) [concrete]
// CHECK:STDOUT:   %assoc0.be9: %BitAndWith.assoc_type.34a = assoc_entity element0, imports.%Core.import_ref.efd [concrete]
// CHECK:STDOUT:   %assoc1.e93: %BitAndWith.assoc_type.34a = assoc_entity element1, imports.%Core.import_ref.2c4 [concrete]
// CHECK:STDOUT:   %Op.type.442b68.6: type = fn_type @Op.6 [concrete]
// CHECK:STDOUT:   %Op.72717c.6: %Op.type.442b68.6 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.946: <witness> = custom_witness (%Int64, %Op.72717c.6), @BitAndWith.1, @BitAndWith.1(%Int64) [concrete]
// CHECK:STDOUT:   %BitAndWith.facet.401: %BitAndWith.type.0cc = facet_value %Int32, (%custom_witness.946) [concrete]
// CHECK:STDOUT:   %Int64.Op.type.83af6e.1: type = fn_type @Int64.Op.1 [concrete]
// CHECK:STDOUT:   %Int64.Op.4dca31.1: %Int64.Op.type.83af6e.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.e7a31c.3: <witness> = custom_witness (%Int64.Op.4dca31.1), @Destroy [concrete]
// CHECK:STDOUT:   %facet_value.88d: %BitAndWith_where.type.0cc = facet_value %Int32, (%custom_witness.946, %custom_witness.e7a31c.3) [concrete]
// CHECK:STDOUT:   %complete_type.3d2: <witness> = complete_type_witness %BitAndWith.type.0cc [concrete]
// CHECK:STDOUT:   %.Self.as_type.75e6: type = facet_access_type %.Self.bfc [symbolic_self]
// CHECK:STDOUT:   %pattern_type.0d8: type = pattern_type %BitAndWith_where.type.0cc [concrete]
// CHECK:STDOUT:   %T.patt.e0d: %pattern_type.0d8 = symbolic_binding_pattern T, 1 [symbolic]
// CHECK:STDOUT:   %BitOrWith.type.1a2: type = facet_type <@BitOrWith.1, @BitOrWith.1(%Int64)> [concrete]
// CHECK:STDOUT:   %.Self.4a6: %BitOrWith.type.1a2 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %BitOrWith.lookup_impl_witness.6d7: <witness> = lookup_impl_witness %.Self.4a6, @BitOrWith.1, @BitOrWith.1(%Int64) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.7df: type = impl_witness_access %BitOrWith.lookup_impl_witness.6d7, element0 [symbolic_self]
// CHECK:STDOUT:   %BitOrWith_where.type.1a2: type = facet_type <@BitOrWith.1, @BitOrWith.1(%Int64) where %impl.elem0.7df impls @Destroy> [concrete]
// CHECK:STDOUT:   %BitOrWith.assoc_type.ccc: type = assoc_entity_type @BitOrWith.1, @BitOrWith.1(%Int64) [concrete]
// CHECK:STDOUT:   %assoc0.f1a: %BitOrWith.assoc_type.ccc = assoc_entity element0, imports.%Core.import_ref.4f6 [concrete]
// CHECK:STDOUT:   %assoc1.fe7: %BitOrWith.assoc_type.ccc = assoc_entity element1, imports.%Core.import_ref.c14 [concrete]
// CHECK:STDOUT:   %Op.type.442b68.7: type = fn_type @Op.7 [concrete]
// CHECK:STDOUT:   %Op.72717c.7: %Op.type.442b68.7 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.d82: <witness> = custom_witness (%Int64, %Op.72717c.7), @BitOrWith.1, @BitOrWith.1(%Int64) [concrete]
// CHECK:STDOUT:   %BitOrWith.facet.7a3: %BitOrWith.type.1a2 = facet_value %Int32, (%custom_witness.d82) [concrete]
// CHECK:STDOUT:   %facet_value.d8d: %BitOrWith_where.type.1a2 = facet_value %Int32, (%custom_witness.d82, %custom_witness.e7a31c.3) [concrete]
// CHECK:STDOUT:   %complete_type.33f: <witness> = complete_type_witness %BitOrWith.type.1a2 [concrete]
// CHECK:STDOUT:   %.Self.as_type.d1f: type = facet_access_type %.Self.4a6 [symbolic_self]
// CHECK:STDOUT:   %pattern_type.f3b: type = pattern_type %BitOrWith_where.type.1a2 [concrete]
// CHECK:STDOUT:   %T.patt.880: %pattern_type.f3b = symbolic_binding_pattern T, 1 [symbolic]
// CHECK:STDOUT:   %BitXorWith.type.a90: type = facet_type <@BitXorWith.1, @BitXorWith.1(%Int64)> [concrete]
// CHECK:STDOUT:   %.Self.32b: %BitXorWith.type.a90 = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %BitXorWith.lookup_impl_witness.c28: <witness> = lookup_impl_witness %.Self.32b, @BitXorWith.1, @BitXorWith.1(%Int64) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.361: type = impl_witness_access %BitXorWith.lookup_impl_witness.c28, element0 [symbolic_self]
// CHECK:STDOUT:   %BitXorWith_where.type.a90: type = facet_type <@BitXorWith.1, @BitXorWith.1(%Int64) where %impl.elem0.361 impls @Destroy> [concrete]
// CHECK:STDOUT:   %BitXorWith.assoc_type.8b9: type = assoc_entity_type @BitXorWith.1, @BitXorWith.1(%Int64) [concrete]
// CHECK:STDOUT:   %assoc0.736: %BitXorWith.assoc_type.8b9 = assoc_entity element0, imports.%Core.import_ref.37d [concrete]
// CHECK:STDOUT:   %assoc1.957: %BitXorWith.assoc_type.8b9 = assoc_entity element1, imports.%Core.import_ref.808 [concrete]
// CHECK:STDOUT:   %Op.type.442b68.8: type = fn_type @Op.8 [concrete]
// CHECK:STDOUT:   %Op.72717c.8: %Op.type.442b68.8 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.888: <witness> = custom_witness (%Int64, %Op.72717c.8), @BitXorWith.1, @BitXorWith.1(%Int64) [concrete]
// CHECK:STDOUT:   %BitXorWith.facet.073: %BitXorWith.type.a90 = facet_value %Int32, (%custom_witness.888) [concrete]
// CHECK:STDOUT:   %facet_value.459: %BitXorWith_where.type.a90 = facet_value %Int32, (%custom_witness.888, %custom_witness.e7a31c.3) [concrete]
// CHECK:STDOUT:   %complete_type.b22: <witness> = complete_type_witness %BitXorWith.type.a90 [concrete]
// CHECK:STDOUT:   %.Self.as_type.94f: type = facet_access_type %.Self.32b [symbolic_self]
// CHECK:STDOUT:   %pattern_type.481: type = pattern_type %BitXorWith_where.type.a90 [concrete]
// CHECK:STDOUT:   %T.patt.304: %pattern_type.481 = symbolic_binding_pattern T, 1 [symbolic]
// CHECK:STDOUT:   %LeftShiftWith.type.d4a: type = facet_type <@LeftShiftWith.1, @LeftShiftWith.1(%Int64)> [concrete]
// CHECK:STDOUT:   %.Self.21d: %LeftShiftWith.type.d4a = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %LeftShiftWith.lookup_impl_witness.fd4: <witness> = lookup_impl_witness %.Self.21d, @LeftShiftWith.1, @LeftShiftWith.1(%Int64) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.f0c: type = impl_witness_access %LeftShiftWith.lookup_impl_witness.fd4, element0 [symbolic_self]
// CHECK:STDOUT:   %LeftShiftWith_where.type.d4a: type = facet_type <@LeftShiftWith.1, @LeftShiftWith.1(%Int64) where %impl.elem0.f0c impls @Destroy> [concrete]
// CHECK:STDOUT:   %LeftShiftWith.assoc_type.b9e: type = assoc_entity_type @LeftShiftWith.1, @LeftShiftWith.1(%Int64) [concrete]
// CHECK:STDOUT:   %assoc0.5b7: %LeftShiftWith.assoc_type.b9e = assoc_entity element0, imports.%Core.import_ref.1a5 [concrete]
// CHECK:STDOUT:   %assoc1.7dc: %LeftShiftWith.assoc_type.b9e = assoc_entity element1, imports.%Core.import_ref.36a [concrete]
// CHECK:STDOUT:   %Op.type.442b68.9: type = fn_type @Op.9 [concrete]
// CHECK:STDOUT:   %Op.72717c.9: %Op.type.442b68.9 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.cdd: <witness> = custom_witness (%Int64, %Op.72717c.9), @LeftShiftWith.1, @LeftShiftWith.1(%Int64) [concrete]
// CHECK:STDOUT:   %LeftShiftWith.facet.3c0: %LeftShiftWith.type.d4a = facet_value %Int32, (%custom_witness.cdd) [concrete]
// CHECK:STDOUT:   %facet_value.ef8: %LeftShiftWith_where.type.d4a = facet_value %Int32, (%custom_witness.cdd, %custom_witness.e7a31c.3) [concrete]
// CHECK:STDOUT:   %complete_type.5f3: <witness> = complete_type_witness %LeftShiftWith.type.d4a [concrete]
// CHECK:STDOUT:   %.Self.as_type.3c7: type = facet_access_type %.Self.21d [symbolic_self]
// CHECK:STDOUT:   %pattern_type.e6d: type = pattern_type %LeftShiftWith_where.type.d4a [concrete]
// CHECK:STDOUT:   %T.patt.795: %pattern_type.e6d = symbolic_binding_pattern T, 1 [symbolic]
// CHECK:STDOUT:   %RightShiftWith.type.39e: type = facet_type <@RightShiftWith.1, @RightShiftWith.1(%Int64)> [concrete]
// CHECK:STDOUT:   %.Self.ac8: %RightShiftWith.type.39e = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %RightShiftWith.lookup_impl_witness.531: <witness> = lookup_impl_witness %.Self.ac8, @RightShiftWith.1, @RightShiftWith.1(%Int64) [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.4cb: type = impl_witness_access %RightShiftWith.lookup_impl_witness.531, element0 [symbolic_self]
// CHECK:STDOUT:   %RightShiftWith_where.type.39e: type = facet_type <@RightShiftWith.1, @RightShiftWith.1(%Int64) where %impl.elem0.4cb impls @Destroy> [concrete]
// CHECK:STDOUT:   %RightShiftWith.assoc_type.56a: type = assoc_entity_type @RightShiftWith.1, @RightShiftWith.1(%Int64) [concrete]
// CHECK:STDOUT:   %assoc0.c11: %RightShiftWith.assoc_type.56a = assoc_entity element0, imports.%Core.import_ref.416 [concrete]
// CHECK:STDOUT:   %assoc1.c9e: %RightShiftWith.assoc_type.56a = assoc_entity element1, imports.%Core.import_ref.e6b [concrete]
// CHECK:STDOUT:   %Op.type.442b68.10: type = fn_type @Op.10 [concrete]
// CHECK:STDOUT:   %Op.72717c.10: %Op.type.442b68.10 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.3bf: <witness> = custom_witness (%Int64, %Op.72717c.10), @RightShiftWith.1, @RightShiftWith.1(%Int64) [concrete]
// CHECK:STDOUT:   %RightShiftWith.facet.e89: %RightShiftWith.type.39e = facet_value %Int32, (%custom_witness.3bf) [concrete]
// CHECK:STDOUT:   %facet_value.b0f: %RightShiftWith_where.type.39e = facet_value %Int32, (%custom_witness.3bf, %custom_witness.e7a31c.3) [concrete]
// CHECK:STDOUT:   %complete_type.7de: <witness> = complete_type_witness %RightShiftWith.type.39e [concrete]
// CHECK:STDOUT:   %.Self.as_type.915: type = facet_access_type %.Self.ac8 [symbolic_self]
// CHECK:STDOUT:   %pattern_type.a3d: type = pattern_type %RightShiftWith_where.type.39e [concrete]
// CHECK:STDOUT:   %T.patt.4f5: %pattern_type.a3d = symbolic_binding_pattern T, 1 [symbolic]
// CHECK:STDOUT:   %x.param_patt.e68: %pattern_type.4aa = value_param_pattern [concrete]
// CHECK:STDOUT:   %x.patt.52a: %pattern_type.4aa = at_binding_pattern x, %x.param_patt.e68 [concrete]
// CHECK:STDOUT:   %Op.type.442b68.11: type = fn_type @Op.11 [concrete]
// CHECK:STDOUT:   %Op.72717c.11: %Op.type.442b68.11 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.20d: <witness> = custom_witness (%Int64, %Op.72717c.11), @BitAndWith.1, @BitAndWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %BitAndWith.facet.7b2: %BitAndWith.type.f43 = facet_value %Int64, (%custom_witness.20d) [concrete]
// CHECK:STDOUT:   %facet_value.797: %BitAndWith_where.type.f43 = facet_value %Int64, (%custom_witness.20d, %custom_witness.e7a31c.3) [concrete]
// CHECK:STDOUT:   %Op.type.442b68.12: type = fn_type @Op.12 [concrete]
// CHECK:STDOUT:   %Op.72717c.12: %Op.type.442b68.12 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.bbd: <witness> = custom_witness (%Int64, %Op.72717c.12), @BitOrWith.1, @BitOrWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %BitOrWith.facet.b65: %BitOrWith.type.878 = facet_value %Int64, (%custom_witness.bbd) [concrete]
// CHECK:STDOUT:   %facet_value.92d: %BitOrWith_where.type.878 = facet_value %Int64, (%custom_witness.e7a31c.3, %custom_witness.bbd) [concrete]
// CHECK:STDOUT:   %Op.type.442b68.13: type = fn_type @Op.13 [concrete]
// CHECK:STDOUT:   %Op.72717c.13: %Op.type.442b68.13 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.8aa: <witness> = custom_witness (%Int64, %Op.72717c.13), @BitXorWith.1, @BitXorWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %BitXorWith.facet.4a6: %BitXorWith.type.729 = facet_value %Int64, (%custom_witness.8aa) [concrete]
// CHECK:STDOUT:   %facet_value.0c5: %BitXorWith_where.type.729 = facet_value %Int64, (%custom_witness.e7a31c.3, %custom_witness.8aa) [concrete]
// CHECK:STDOUT:   %Op.type.442b68.14: type = fn_type @Op.14 [concrete]
// CHECK:STDOUT:   %Op.72717c.14: %Op.type.442b68.14 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.bee: <witness> = custom_witness (%Int64, %Op.72717c.14), @LeftShiftWith.1, @LeftShiftWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %LeftShiftWith.facet.22c: %LeftShiftWith.type.15a = facet_value %Int64, (%custom_witness.bee) [concrete]
// CHECK:STDOUT:   %facet_value.608: %LeftShiftWith_where.type.15a = facet_value %Int64, (%custom_witness.e7a31c.3, %custom_witness.bee) [concrete]
// CHECK:STDOUT:   %Op.type.442b68.15: type = fn_type @Op.15 [concrete]
// CHECK:STDOUT:   %Op.72717c.15: %Op.type.442b68.15 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.9e2: <witness> = custom_witness (%Int64, %Op.72717c.15), @RightShiftWith.1, @RightShiftWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %RightShiftWith.facet.ffd: %RightShiftWith.type.8f0 = facet_value %Int64, (%custom_witness.9e2) [concrete]
// CHECK:STDOUT:   %facet_value.2e1: %RightShiftWith_where.type.8f0 = facet_value %Int64, (%custom_witness.e7a31c.3, %custom_witness.9e2) [concrete]
// CHECK:STDOUT:   %BitAndAssignWith.type.24c915.1: type = facet_type <@BitAndAssignWith.1, @BitAndAssignWith.1(%Other)> [symbolic]
// CHECK:STDOUT:   %Self.80b651.1: %BitAndAssignWith.type.24c915.1 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %BitAndAssignWith.assoc_type.5d10e3.1: type = assoc_entity_type @BitAndAssignWith.1, @BitAndAssignWith.1(%Other) [symbolic]
// CHECK:STDOUT:   %BitAndAssignWith.WithSelf.Op.type.e3224f.1: type = fn_type @BitAndAssignWith.WithSelf.Op, @BitAndAssignWith.WithSelf(%Other, %Self.80b651.1) [symbolic]
// CHECK:STDOUT:   %BitAndAssignWith.WithSelf.Op.b3572a.1: %BitAndAssignWith.WithSelf.Op.type.e3224f.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %BitAndAssignWith.type.24c915.2: type = facet_type <@BitAndAssignWith.1, @BitAndAssignWith.1(%U.67d)> [symbolic]
// CHECK:STDOUT:   %pattern_type.89d: type = pattern_type %BitAndAssignWith.type.24c915.2 [symbolic]
// CHECK:STDOUT:   %T.patt.7a6: %pattern_type.89d = symbolic_binding_pattern T, 1 [symbolic]
// CHECK:STDOUT:   %T.80b: %BitAndAssignWith.type.24c915.2 = symbolic_binding T, 1 [symbolic]
// CHECK:STDOUT:   %BitAndAssignWith.assoc_type.5d10e3.2: type = assoc_entity_type @BitAndAssignWith.1, @BitAndAssignWith.1(%U.67d) [symbolic]
// CHECK:STDOUT:   %assoc0.43e266.2: %BitAndAssignWith.assoc_type.5d10e3.2 = assoc_entity element0, imports.%Core.import_ref.618 [symbolic]
// CHECK:STDOUT:   %T.as_type.dca: type = facet_access_type %T.80b [symbolic]
// CHECK:STDOUT:   %pattern_type.d3a81a.2: type = pattern_type %T.as_type.dca [symbolic]
// CHECK:STDOUT:   %x.patt.66b: %pattern_type.d3a81a.2 = ref_binding_pattern x [symbolic]
// CHECK:STDOUT:   %x.param_patt.9e7: %pattern_type.d3a81a.2 = var_param_pattern %x.patt.66b [symbolic]
// CHECK:STDOUT:   %require_complete.9c4: <witness> = require_complete_type %BitAndAssignWith.type.24c915.2 [symbolic]
// CHECK:STDOUT:   %assoc0.4dc: %BitAndAssignWith.assoc_type.5d10e3.1 = assoc_entity element0, imports.%Core.import_ref.4c2 [symbolic]
// CHECK:STDOUT:   %BitAndAssignWith.lookup_impl_witness: <witness> = lookup_impl_witness %T.80b, @BitAndAssignWith.1, @BitAndAssignWith.1(%U.67d) [symbolic]
// CHECK:STDOUT:   %BitAndAssignWith.WithSelf.Op.type.e3224f.3: type = fn_type @BitAndAssignWith.WithSelf.Op, @BitAndAssignWith.WithSelf(%U.67d, %T.80b) [symbolic]
// CHECK:STDOUT:   %.bea: type = fn_type_with_self_type %BitAndAssignWith.WithSelf.Op.type.e3224f.3, %T.80b [symbolic]
// CHECK:STDOUT:   %impl.elem0.a31: %.bea = impl_witness_access %BitAndAssignWith.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.f91: <specific function> = specific_impl_function %impl.elem0.a31, @BitAndAssignWith.WithSelf.Op(%U.67d, %T.80b) [symbolic]
// CHECK:STDOUT:   %BitOrAssignWith.type.3ef7d1.1: type = facet_type <@BitOrAssignWith.1, @BitOrAssignWith.1(%Other)> [symbolic]
// CHECK:STDOUT:   %Self.cc8574.1: %BitOrAssignWith.type.3ef7d1.1 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %BitOrAssignWith.assoc_type.e05d8b.1: type = assoc_entity_type @BitOrAssignWith.1, @BitOrAssignWith.1(%Other) [symbolic]
// CHECK:STDOUT:   %BitOrAssignWith.WithSelf.Op.type.130458.1: type = fn_type @BitOrAssignWith.WithSelf.Op, @BitOrAssignWith.WithSelf(%Other, %Self.cc8574.1) [symbolic]
// CHECK:STDOUT:   %BitOrAssignWith.WithSelf.Op.53e52f.1: %BitOrAssignWith.WithSelf.Op.type.130458.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %BitOrAssignWith.type.3ef7d1.2: type = facet_type <@BitOrAssignWith.1, @BitOrAssignWith.1(%U.67d)> [symbolic]
// CHECK:STDOUT:   %pattern_type.12d: type = pattern_type %BitOrAssignWith.type.3ef7d1.2 [symbolic]
// CHECK:STDOUT:   %T.patt.ae4: %pattern_type.12d = symbolic_binding_pattern T, 1 [symbolic]
// CHECK:STDOUT:   %T.cc8: %BitOrAssignWith.type.3ef7d1.2 = symbolic_binding T, 1 [symbolic]
// CHECK:STDOUT:   %BitOrAssignWith.assoc_type.e05d8b.2: type = assoc_entity_type @BitOrAssignWith.1, @BitOrAssignWith.1(%U.67d) [symbolic]
// CHECK:STDOUT:   %assoc0.b0b078.2: %BitOrAssignWith.assoc_type.e05d8b.2 = assoc_entity element0, imports.%Core.import_ref.b03 [symbolic]
// CHECK:STDOUT:   %T.as_type.72f: type = facet_access_type %T.cc8 [symbolic]
// CHECK:STDOUT:   %pattern_type.cefac8.2: type = pattern_type %T.as_type.72f [symbolic]
// CHECK:STDOUT:   %x.patt.5d2: %pattern_type.cefac8.2 = ref_binding_pattern x [symbolic]
// CHECK:STDOUT:   %x.param_patt.d0d: %pattern_type.cefac8.2 = var_param_pattern %x.patt.5d2 [symbolic]
// CHECK:STDOUT:   %require_complete.abb: <witness> = require_complete_type %BitOrAssignWith.type.3ef7d1.2 [symbolic]
// CHECK:STDOUT:   %assoc0.be6: %BitOrAssignWith.assoc_type.e05d8b.1 = assoc_entity element0, imports.%Core.import_ref.7fc [symbolic]
// CHECK:STDOUT:   %BitOrAssignWith.lookup_impl_witness: <witness> = lookup_impl_witness %T.cc8, @BitOrAssignWith.1, @BitOrAssignWith.1(%U.67d) [symbolic]
// CHECK:STDOUT:   %BitOrAssignWith.WithSelf.Op.type.130458.3: type = fn_type @BitOrAssignWith.WithSelf.Op, @BitOrAssignWith.WithSelf(%U.67d, %T.cc8) [symbolic]
// CHECK:STDOUT:   %.291: type = fn_type_with_self_type %BitOrAssignWith.WithSelf.Op.type.130458.3, %T.cc8 [symbolic]
// CHECK:STDOUT:   %impl.elem0.fd9: %.291 = impl_witness_access %BitOrAssignWith.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.be4: <specific function> = specific_impl_function %impl.elem0.fd9, @BitOrAssignWith.WithSelf.Op(%U.67d, %T.cc8) [symbolic]
// CHECK:STDOUT:   %BitXorAssignWith.type.70c283.1: type = facet_type <@BitXorAssignWith.1, @BitXorAssignWith.1(%Other)> [symbolic]
// CHECK:STDOUT:   %Self.c383a6.1: %BitXorAssignWith.type.70c283.1 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %BitXorAssignWith.assoc_type.2be5ac.1: type = assoc_entity_type @BitXorAssignWith.1, @BitXorAssignWith.1(%Other) [symbolic]
// CHECK:STDOUT:   %BitXorAssignWith.WithSelf.Op.type.8f9d71.1: type = fn_type @BitXorAssignWith.WithSelf.Op, @BitXorAssignWith.WithSelf(%Other, %Self.c383a6.1) [symbolic]
// CHECK:STDOUT:   %BitXorAssignWith.WithSelf.Op.848804.1: %BitXorAssignWith.WithSelf.Op.type.8f9d71.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %BitXorAssignWith.type.70c283.2: type = facet_type <@BitXorAssignWith.1, @BitXorAssignWith.1(%U.67d)> [symbolic]
// CHECK:STDOUT:   %pattern_type.472: type = pattern_type %BitXorAssignWith.type.70c283.2 [symbolic]
// CHECK:STDOUT:   %T.patt.55d: %pattern_type.472 = symbolic_binding_pattern T, 1 [symbolic]
// CHECK:STDOUT:   %T.c38: %BitXorAssignWith.type.70c283.2 = symbolic_binding T, 1 [symbolic]
// CHECK:STDOUT:   %BitXorAssignWith.assoc_type.2be5ac.2: type = assoc_entity_type @BitXorAssignWith.1, @BitXorAssignWith.1(%U.67d) [symbolic]
// CHECK:STDOUT:   %assoc0.5768f4.2: %BitXorAssignWith.assoc_type.2be5ac.2 = assoc_entity element0, imports.%Core.import_ref.194 [symbolic]
// CHECK:STDOUT:   %T.as_type.a84: type = facet_access_type %T.c38 [symbolic]
// CHECK:STDOUT:   %pattern_type.1bffb5.2: type = pattern_type %T.as_type.a84 [symbolic]
// CHECK:STDOUT:   %x.patt.f3d: %pattern_type.1bffb5.2 = ref_binding_pattern x [symbolic]
// CHECK:STDOUT:   %x.param_patt.1d5: %pattern_type.1bffb5.2 = var_param_pattern %x.patt.f3d [symbolic]
// CHECK:STDOUT:   %require_complete.023: <witness> = require_complete_type %BitXorAssignWith.type.70c283.2 [symbolic]
// CHECK:STDOUT:   %assoc0.ef6: %BitXorAssignWith.assoc_type.2be5ac.1 = assoc_entity element0, imports.%Core.import_ref.d20 [symbolic]
// CHECK:STDOUT:   %BitXorAssignWith.lookup_impl_witness: <witness> = lookup_impl_witness %T.c38, @BitXorAssignWith.1, @BitXorAssignWith.1(%U.67d) [symbolic]
// CHECK:STDOUT:   %BitXorAssignWith.WithSelf.Op.type.8f9d71.3: type = fn_type @BitXorAssignWith.WithSelf.Op, @BitXorAssignWith.WithSelf(%U.67d, %T.c38) [symbolic]
// CHECK:STDOUT:   %.672: type = fn_type_with_self_type %BitXorAssignWith.WithSelf.Op.type.8f9d71.3, %T.c38 [symbolic]
// CHECK:STDOUT:   %impl.elem0.692: %.672 = impl_witness_access %BitXorAssignWith.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.5c0: <specific function> = specific_impl_function %impl.elem0.692, @BitXorAssignWith.WithSelf.Op(%U.67d, %T.c38) [symbolic]
// CHECK:STDOUT:   %LeftShiftAssignWith.type.858756.1: type = facet_type <@LeftShiftAssignWith.1, @LeftShiftAssignWith.1(%Other)> [symbolic]
// CHECK:STDOUT:   %Self.de3088.1: %LeftShiftAssignWith.type.858756.1 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %LeftShiftAssignWith.assoc_type.239476.1: type = assoc_entity_type @LeftShiftAssignWith.1, @LeftShiftAssignWith.1(%Other) [symbolic]
// CHECK:STDOUT:   %LeftShiftAssignWith.WithSelf.Op.type.ce10f9.1: type = fn_type @LeftShiftAssignWith.WithSelf.Op, @LeftShiftAssignWith.WithSelf(%Other, %Self.de3088.1) [symbolic]
// CHECK:STDOUT:   %LeftShiftAssignWith.WithSelf.Op.308429.1: %LeftShiftAssignWith.WithSelf.Op.type.ce10f9.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %LeftShiftAssignWith.type.858756.2: type = facet_type <@LeftShiftAssignWith.1, @LeftShiftAssignWith.1(%U.67d)> [symbolic]
// CHECK:STDOUT:   %pattern_type.08d: type = pattern_type %LeftShiftAssignWith.type.858756.2 [symbolic]
// CHECK:STDOUT:   %T.patt.bd5: %pattern_type.08d = symbolic_binding_pattern T, 1 [symbolic]
// CHECK:STDOUT:   %T.de3: %LeftShiftAssignWith.type.858756.2 = symbolic_binding T, 1 [symbolic]
// CHECK:STDOUT:   %LeftShiftAssignWith.assoc_type.239476.2: type = assoc_entity_type @LeftShiftAssignWith.1, @LeftShiftAssignWith.1(%U.67d) [symbolic]
// CHECK:STDOUT:   %assoc0.d11495.2: %LeftShiftAssignWith.assoc_type.239476.2 = assoc_entity element0, imports.%Core.import_ref.2b9 [symbolic]
// CHECK:STDOUT:   %T.as_type.09c: type = facet_access_type %T.de3 [symbolic]
// CHECK:STDOUT:   %pattern_type.6d2709.2: type = pattern_type %T.as_type.09c [symbolic]
// CHECK:STDOUT:   %x.patt.23b: %pattern_type.6d2709.2 = ref_binding_pattern x [symbolic]
// CHECK:STDOUT:   %x.param_patt.02d: %pattern_type.6d2709.2 = var_param_pattern %x.patt.23b [symbolic]
// CHECK:STDOUT:   %require_complete.5ca: <witness> = require_complete_type %LeftShiftAssignWith.type.858756.2 [symbolic]
// CHECK:STDOUT:   %assoc0.ad5: %LeftShiftAssignWith.assoc_type.239476.1 = assoc_entity element0, imports.%Core.import_ref.48dce [symbolic]
// CHECK:STDOUT:   %LeftShiftAssignWith.lookup_impl_witness: <witness> = lookup_impl_witness %T.de3, @LeftShiftAssignWith.1, @LeftShiftAssignWith.1(%U.67d) [symbolic]
// CHECK:STDOUT:   %LeftShiftAssignWith.WithSelf.Op.type.ce10f9.3: type = fn_type @LeftShiftAssignWith.WithSelf.Op, @LeftShiftAssignWith.WithSelf(%U.67d, %T.de3) [symbolic]
// CHECK:STDOUT:   %.591: type = fn_type_with_self_type %LeftShiftAssignWith.WithSelf.Op.type.ce10f9.3, %T.de3 [symbolic]
// CHECK:STDOUT:   %impl.elem0.a9d: %.591 = impl_witness_access %LeftShiftAssignWith.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.459: <specific function> = specific_impl_function %impl.elem0.a9d, @LeftShiftAssignWith.WithSelf.Op(%U.67d, %T.de3) [symbolic]
// CHECK:STDOUT:   %RightShiftAssignWith.type.abebad.1: type = facet_type <@RightShiftAssignWith.1, @RightShiftAssignWith.1(%Other)> [symbolic]
// CHECK:STDOUT:   %Self.d7c2e9.1: %RightShiftAssignWith.type.abebad.1 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %RightShiftAssignWith.assoc_type.95eeb2.1: type = assoc_entity_type @RightShiftAssignWith.1, @RightShiftAssignWith.1(%Other) [symbolic]
// CHECK:STDOUT:   %RightShiftAssignWith.WithSelf.Op.type.f62b2f.1: type = fn_type @RightShiftAssignWith.WithSelf.Op, @RightShiftAssignWith.WithSelf(%Other, %Self.d7c2e9.1) [symbolic]
// CHECK:STDOUT:   %RightShiftAssignWith.WithSelf.Op.500e9a.1: %RightShiftAssignWith.WithSelf.Op.type.f62b2f.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %RightShiftAssignWith.type.abebad.2: type = facet_type <@RightShiftAssignWith.1, @RightShiftAssignWith.1(%U.67d)> [symbolic]
// CHECK:STDOUT:   %pattern_type.9e9: type = pattern_type %RightShiftAssignWith.type.abebad.2 [symbolic]
// CHECK:STDOUT:   %T.patt.0c5: %pattern_type.9e9 = symbolic_binding_pattern T, 1 [symbolic]
// CHECK:STDOUT:   %T.d7c: %RightShiftAssignWith.type.abebad.2 = symbolic_binding T, 1 [symbolic]
// CHECK:STDOUT:   %RightShiftAssignWith.assoc_type.95eeb2.2: type = assoc_entity_type @RightShiftAssignWith.1, @RightShiftAssignWith.1(%U.67d) [symbolic]
// CHECK:STDOUT:   %assoc0.31355d.2: %RightShiftAssignWith.assoc_type.95eeb2.2 = assoc_entity element0, imports.%Core.import_ref.0e8 [symbolic]
// CHECK:STDOUT:   %T.as_type.8e8: type = facet_access_type %T.d7c [symbolic]
// CHECK:STDOUT:   %pattern_type.e901da.2: type = pattern_type %T.as_type.8e8 [symbolic]
// CHECK:STDOUT:   %x.patt.cf9: %pattern_type.e901da.2 = ref_binding_pattern x [symbolic]
// CHECK:STDOUT:   %x.param_patt.e72: %pattern_type.e901da.2 = var_param_pattern %x.patt.cf9 [symbolic]
// CHECK:STDOUT:   %require_complete.e9c: <witness> = require_complete_type %RightShiftAssignWith.type.abebad.2 [symbolic]
// CHECK:STDOUT:   %assoc0.c59: %RightShiftAssignWith.assoc_type.95eeb2.1 = assoc_entity element0, imports.%Core.import_ref.88f [symbolic]
// CHECK:STDOUT:   %RightShiftAssignWith.lookup_impl_witness: <witness> = lookup_impl_witness %T.d7c, @RightShiftAssignWith.1, @RightShiftAssignWith.1(%U.67d) [symbolic]
// CHECK:STDOUT:   %RightShiftAssignWith.WithSelf.Op.type.f62b2f.3: type = fn_type @RightShiftAssignWith.WithSelf.Op, @RightShiftAssignWith.WithSelf(%U.67d, %T.d7c) [symbolic]
// CHECK:STDOUT:   %.609: type = fn_type_with_self_type %RightShiftAssignWith.WithSelf.Op.type.f62b2f.3, %T.d7c [symbolic]
// CHECK:STDOUT:   %impl.elem0.82d: %.609 = impl_witness_access %RightShiftAssignWith.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.42f: <specific function> = specific_impl_function %impl.elem0.82d, @RightShiftAssignWith.WithSelf.Op(%U.67d, %T.d7c) [symbolic]
// CHECK:STDOUT:   %BitAndAssignWith.type.0b5: type = facet_type <@BitAndAssignWith.1, @BitAndAssignWith.1(%i16)> [concrete]
// CHECK:STDOUT:   %BitAndAssignWith.assoc_type.c34: type = assoc_entity_type @BitAndAssignWith.1, @BitAndAssignWith.1(%i16) [concrete]
// CHECK:STDOUT:   %assoc0.f42: %BitAndAssignWith.assoc_type.c34 = assoc_entity element0, imports.%Core.import_ref.618 [concrete]
// CHECK:STDOUT:   %Int16.Op.type.83af6e.13: type = fn_type @Int16.Op.13 [concrete]
// CHECK:STDOUT:   %Int16.Op.4dca31.13: %Int16.Op.type.83af6e.13 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.f28: <witness> = custom_witness (%Int16.Op.4dca31.13), @BitAndAssignWith.1, @BitAndAssignWith.1(%i16) [concrete]
// CHECK:STDOUT:   %BitAndAssignWith.facet.355: %BitAndAssignWith.type.0b5 = facet_value %Int16, (%custom_witness.f28) [concrete]
// CHECK:STDOUT:   %pattern_type.628: type = pattern_type %BitAndAssignWith.type.0b5 [concrete]
// CHECK:STDOUT:   %T.patt.9fc: %pattern_type.628 = symbolic_binding_pattern T, 1 [symbolic]
// CHECK:STDOUT:   %x.patt.49e: %pattern_type.785 = ref_binding_pattern x [concrete]
// CHECK:STDOUT:   %x.param_patt.69a: %pattern_type.785 = var_param_pattern %x.patt.49e [concrete]
// CHECK:STDOUT:   %BitOrAssignWith.type.8ff: type = facet_type <@BitOrAssignWith.1, @BitOrAssignWith.1(%i16)> [concrete]
// CHECK:STDOUT:   %BitOrAssignWith.assoc_type.b1c: type = assoc_entity_type @BitOrAssignWith.1, @BitOrAssignWith.1(%i16) [concrete]
// CHECK:STDOUT:   %assoc0.392: %BitOrAssignWith.assoc_type.b1c = assoc_entity element0, imports.%Core.import_ref.b03 [concrete]
// CHECK:STDOUT:   %Int16.Op.type.83af6e.14: type = fn_type @Int16.Op.14 [concrete]
// CHECK:STDOUT:   %Int16.Op.4dca31.14: %Int16.Op.type.83af6e.14 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.c53: <witness> = custom_witness (%Int16.Op.4dca31.14), @BitOrAssignWith.1, @BitOrAssignWith.1(%i16) [concrete]
// CHECK:STDOUT:   %BitOrAssignWith.facet.ea2: %BitOrAssignWith.type.8ff = facet_value %Int16, (%custom_witness.c53) [concrete]
// CHECK:STDOUT:   %pattern_type.1db: type = pattern_type %BitOrAssignWith.type.8ff [concrete]
// CHECK:STDOUT:   %T.patt.670: %pattern_type.1db = symbolic_binding_pattern T, 1 [symbolic]
// CHECK:STDOUT:   %BitXorAssignWith.type.1ee: type = facet_type <@BitXorAssignWith.1, @BitXorAssignWith.1(%i16)> [concrete]
// CHECK:STDOUT:   %BitXorAssignWith.assoc_type.313: type = assoc_entity_type @BitXorAssignWith.1, @BitXorAssignWith.1(%i16) [concrete]
// CHECK:STDOUT:   %assoc0.b72: %BitXorAssignWith.assoc_type.313 = assoc_entity element0, imports.%Core.import_ref.194 [concrete]
// CHECK:STDOUT:   %Int16.Op.type.83af6e.15: type = fn_type @Int16.Op.15 [concrete]
// CHECK:STDOUT:   %Int16.Op.4dca31.15: %Int16.Op.type.83af6e.15 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.696: <witness> = custom_witness (%Int16.Op.4dca31.15), @BitXorAssignWith.1, @BitXorAssignWith.1(%i16) [concrete]
// CHECK:STDOUT:   %BitXorAssignWith.facet.84f: %BitXorAssignWith.type.1ee = facet_value %Int16, (%custom_witness.696) [concrete]
// CHECK:STDOUT:   %pattern_type.fba: type = pattern_type %BitXorAssignWith.type.1ee [concrete]
// CHECK:STDOUT:   %T.patt.72a: %pattern_type.fba = symbolic_binding_pattern T, 1 [symbolic]
// CHECK:STDOUT:   %LeftShiftAssignWith.type.62a: type = facet_type <@LeftShiftAssignWith.1, @LeftShiftAssignWith.1(%i16)> [concrete]
// CHECK:STDOUT:   %LeftShiftAssignWith.assoc_type.af0: type = assoc_entity_type @LeftShiftAssignWith.1, @LeftShiftAssignWith.1(%i16) [concrete]
// CHECK:STDOUT:   %assoc0.c84: %LeftShiftAssignWith.assoc_type.af0 = assoc_entity element0, imports.%Core.import_ref.2b9 [concrete]
// CHECK:STDOUT:   %Int16.Op.type.83af6e.16: type = fn_type @Int16.Op.16 [concrete]
// CHECK:STDOUT:   %Int16.Op.4dca31.16: %Int16.Op.type.83af6e.16 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.95b: <witness> = custom_witness (%Int16.Op.4dca31.16), @LeftShiftAssignWith.1, @LeftShiftAssignWith.1(%i16) [concrete]
// CHECK:STDOUT:   %LeftShiftAssignWith.facet.71f: %LeftShiftAssignWith.type.62a = facet_value %Int16, (%custom_witness.95b) [concrete]
// CHECK:STDOUT:   %pattern_type.65f: type = pattern_type %LeftShiftAssignWith.type.62a [concrete]
// CHECK:STDOUT:   %T.patt.fe8: %pattern_type.65f = symbolic_binding_pattern T, 1 [symbolic]
// CHECK:STDOUT:   %RightShiftAssignWith.type.b1f: type = facet_type <@RightShiftAssignWith.1, @RightShiftAssignWith.1(%i16)> [concrete]
// CHECK:STDOUT:   %RightShiftAssignWith.assoc_type.d35: type = assoc_entity_type @RightShiftAssignWith.1, @RightShiftAssignWith.1(%i16) [concrete]
// CHECK:STDOUT:   %assoc0.655: %RightShiftAssignWith.assoc_type.d35 = assoc_entity element0, imports.%Core.import_ref.0e8 [concrete]
// CHECK:STDOUT:   %Int16.Op.type.83af6e.17: type = fn_type @Int16.Op.17 [concrete]
// CHECK:STDOUT:   %Int16.Op.4dca31.17: %Int16.Op.type.83af6e.17 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.114: <witness> = custom_witness (%Int16.Op.4dca31.17), @RightShiftAssignWith.1, @RightShiftAssignWith.1(%i16) [concrete]
// CHECK:STDOUT:   %RightShiftAssignWith.facet.f54: %RightShiftAssignWith.type.b1f = facet_value %Int16, (%custom_witness.114) [concrete]
// CHECK:STDOUT:   %pattern_type.193: type = pattern_type %RightShiftAssignWith.type.b1f [concrete]
// CHECK:STDOUT:   %T.patt.613: %pattern_type.193 = symbolic_binding_pattern T, 1 [symbolic]
// CHECK:STDOUT:   %BitAndAssignWith.type.53c: type = facet_type <@BitAndAssignWith.1, @BitAndAssignWith.1(%Int32)> [concrete]
// CHECK:STDOUT:   %BitAndAssignWith.assoc_type.3ad: type = assoc_entity_type @BitAndAssignWith.1, @BitAndAssignWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %assoc0.04c: %BitAndAssignWith.assoc_type.3ad = assoc_entity element0, imports.%Core.import_ref.618 [concrete]
// CHECK:STDOUT:   %Int32.Op.type.83af6e.12: type = fn_type @Int32.Op.12 [concrete]
// CHECK:STDOUT:   %Int32.Op.4dca31.12: %Int32.Op.type.83af6e.12 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.4ae282.1: <witness> = custom_witness (%Int32.Op.4dca31.12), @BitAndAssignWith.1, @BitAndAssignWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %BitAndAssignWith.facet.5d5: %BitAndAssignWith.type.53c = facet_value %Int32, (%custom_witness.4ae282.1) [concrete]
// CHECK:STDOUT:   %pattern_type.cf6: type = pattern_type %BitAndAssignWith.type.53c [concrete]
// CHECK:STDOUT:   %T.patt.c2f: %pattern_type.cf6 = symbolic_binding_pattern T, 1 [symbolic]
// CHECK:STDOUT:   %x.patt.d79: %pattern_type.8b45 = ref_binding_pattern x [concrete]
// CHECK:STDOUT:   %x.param_patt.0c7: %pattern_type.8b45 = var_param_pattern %x.patt.d79 [concrete]
// CHECK:STDOUT:   %BitOrAssignWith.type.46f: type = facet_type <@BitOrAssignWith.1, @BitOrAssignWith.1(%Int32)> [concrete]
// CHECK:STDOUT:   %BitOrAssignWith.assoc_type.bc6: type = assoc_entity_type @BitOrAssignWith.1, @BitOrAssignWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %assoc0.63a: %BitOrAssignWith.assoc_type.bc6 = assoc_entity element0, imports.%Core.import_ref.b03 [concrete]
// CHECK:STDOUT:   %Int32.Op.type.83af6e.14: type = fn_type @Int32.Op.14 [concrete]
// CHECK:STDOUT:   %Int32.Op.4dca31.14: %Int32.Op.type.83af6e.14 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.9a90d3.1: <witness> = custom_witness (%Int32.Op.4dca31.14), @BitOrAssignWith.1, @BitOrAssignWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %BitOrAssignWith.facet.ad2: %BitOrAssignWith.type.46f = facet_value %Int32, (%custom_witness.9a90d3.1) [concrete]
// CHECK:STDOUT:   %pattern_type.b51: type = pattern_type %BitOrAssignWith.type.46f [concrete]
// CHECK:STDOUT:   %T.patt.9dc: %pattern_type.b51 = symbolic_binding_pattern T, 1 [symbolic]
// CHECK:STDOUT:   %BitXorAssignWith.type.0e4: type = facet_type <@BitXorAssignWith.1, @BitXorAssignWith.1(%Int32)> [concrete]
// CHECK:STDOUT:   %BitXorAssignWith.assoc_type.e5b: type = assoc_entity_type @BitXorAssignWith.1, @BitXorAssignWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %assoc0.1fe: %BitXorAssignWith.assoc_type.e5b = assoc_entity element0, imports.%Core.import_ref.194 [concrete]
// CHECK:STDOUT:   %Int32.Op.type.83af6e.15: type = fn_type @Int32.Op.15 [concrete]
// CHECK:STDOUT:   %Int32.Op.4dca31.15: %Int32.Op.type.83af6e.15 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.9bdd97.1: <witness> = custom_witness (%Int32.Op.4dca31.15), @BitXorAssignWith.1, @BitXorAssignWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %BitXorAssignWith.facet.627: %BitXorAssignWith.type.0e4 = facet_value %Int32, (%custom_witness.9bdd97.1) [concrete]
// CHECK:STDOUT:   %pattern_type.e71: type = pattern_type %BitXorAssignWith.type.0e4 [concrete]
// CHECK:STDOUT:   %T.patt.6f4: %pattern_type.e71 = symbolic_binding_pattern T, 1 [symbolic]
// CHECK:STDOUT:   %LeftShiftAssignWith.type.635: type = facet_type <@LeftShiftAssignWith.1, @LeftShiftAssignWith.1(%Int32)> [concrete]
// CHECK:STDOUT:   %LeftShiftAssignWith.assoc_type.be9: type = assoc_entity_type @LeftShiftAssignWith.1, @LeftShiftAssignWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %assoc0.765: %LeftShiftAssignWith.assoc_type.be9 = assoc_entity element0, imports.%Core.import_ref.2b9 [concrete]
// CHECK:STDOUT:   %Int32.Op.type.83af6e.16: type = fn_type @Int32.Op.16 [concrete]
// CHECK:STDOUT:   %Int32.Op.4dca31.16: %Int32.Op.type.83af6e.16 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.3e2e0c.1: <witness> = custom_witness (%Int32.Op.4dca31.16), @LeftShiftAssignWith.1, @LeftShiftAssignWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %LeftShiftAssignWith.facet.fb8: %LeftShiftAssignWith.type.635 = facet_value %Int32, (%custom_witness.3e2e0c.1) [concrete]
// CHECK:STDOUT:   %pattern_type.1d0: type = pattern_type %LeftShiftAssignWith.type.635 [concrete]
// CHECK:STDOUT:   %T.patt.a16: %pattern_type.1d0 = symbolic_binding_pattern T, 1 [symbolic]
// CHECK:STDOUT:   %RightShiftAssignWith.type.239: type = facet_type <@RightShiftAssignWith.1, @RightShiftAssignWith.1(%Int32)> [concrete]
// CHECK:STDOUT:   %RightShiftAssignWith.assoc_type.21a: type = assoc_entity_type @RightShiftAssignWith.1, @RightShiftAssignWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %assoc0.e7b: %RightShiftAssignWith.assoc_type.21a = assoc_entity element0, imports.%Core.import_ref.0e8 [concrete]
// CHECK:STDOUT:   %Int32.Op.type.83af6e.17: type = fn_type @Int32.Op.17 [concrete]
// CHECK:STDOUT:   %Int32.Op.4dca31.17: %Int32.Op.type.83af6e.17 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.cbf0fb.1: <witness> = custom_witness (%Int32.Op.4dca31.17), @RightShiftAssignWith.1, @RightShiftAssignWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %RightShiftAssignWith.facet.288: %RightShiftAssignWith.type.239 = facet_value %Int32, (%custom_witness.cbf0fb.1) [concrete]
// CHECK:STDOUT:   %pattern_type.0b1: type = pattern_type %RightShiftAssignWith.type.239 [concrete]
// CHECK:STDOUT:   %T.patt.04a: %pattern_type.0b1 = symbolic_binding_pattern T, 1 [symbolic]
// CHECK:STDOUT:   %Int64.Op.type.83af6e.2: type = fn_type @Int64.Op.2 [concrete]
// CHECK:STDOUT:   %Int64.Op.4dca31.2: %Int64.Op.type.83af6e.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.4ae282.2: <witness> = custom_witness (%Int64.Op.4dca31.2), @BitAndAssignWith.1, @BitAndAssignWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %BitAndAssignWith.facet.cd3: %BitAndAssignWith.type.53c = facet_value %Int64, (%custom_witness.4ae282.2) [concrete]
// CHECK:STDOUT:   %x.patt.ace: %pattern_type.4aa = ref_binding_pattern x [concrete]
// CHECK:STDOUT:   %x.param_patt.b33: %pattern_type.4aa = var_param_pattern %x.patt.ace [concrete]
// CHECK:STDOUT:   %Int64.Op.type.83af6e.4: type = fn_type @Int64.Op.4 [concrete]
// CHECK:STDOUT:   %Int64.Op.4dca31.4: %Int64.Op.type.83af6e.4 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.9a90d3.2: <witness> = custom_witness (%Int64.Op.4dca31.4), @BitOrAssignWith.1, @BitOrAssignWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %BitOrAssignWith.facet.c6e: %BitOrAssignWith.type.46f = facet_value %Int64, (%custom_witness.9a90d3.2) [concrete]
// CHECK:STDOUT:   %Int64.Op.type.83af6e.5: type = fn_type @Int64.Op.5 [concrete]
// CHECK:STDOUT:   %Int64.Op.4dca31.5: %Int64.Op.type.83af6e.5 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.9bdd97.2: <witness> = custom_witness (%Int64.Op.4dca31.5), @BitXorAssignWith.1, @BitXorAssignWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %BitXorAssignWith.facet.896: %BitXorAssignWith.type.0e4 = facet_value %Int64, (%custom_witness.9bdd97.2) [concrete]
// CHECK:STDOUT:   %Int64.Op.type.83af6e.6: type = fn_type @Int64.Op.6 [concrete]
// CHECK:STDOUT:   %Int64.Op.4dca31.6: %Int64.Op.type.83af6e.6 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.3e2e0c.2: <witness> = custom_witness (%Int64.Op.4dca31.6), @LeftShiftAssignWith.1, @LeftShiftAssignWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %LeftShiftAssignWith.facet.8cb: %LeftShiftAssignWith.type.635 = facet_value %Int64, (%custom_witness.3e2e0c.2) [concrete]
// CHECK:STDOUT:   %Int64.Op.type.83af6e.7: type = fn_type @Int64.Op.7 [concrete]
// CHECK:STDOUT:   %Int64.Op.4dca31.7: %Int64.Op.type.83af6e.7 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.cbf0fb.2: <witness> = custom_witness (%Int64.Op.4dca31.7), @RightShiftAssignWith.1, @RightShiftAssignWith.1(%Int32) [concrete]
// CHECK:STDOUT:   %RightShiftAssignWith.facet.040: %RightShiftAssignWith.type.239 = facet_value %Int64, (%custom_witness.cbf0fb.2) [concrete]
// CHECK:STDOUT:   %Destroy.facet.191: %Destroy.type = facet_value %Int64, (%custom_witness.e7a31c.3) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.c2c: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.191) [concrete]
// CHECK:STDOUT:   %.c2e: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.c2c, %Destroy.facet.191 [concrete]
// CHECK:STDOUT:   %Destroy.facet.099: %Destroy.type = facet_value %Int32, (%custom_witness.e7a31c.2) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.b76: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.099) [concrete]
// CHECK:STDOUT:   %.043: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.b76, %Destroy.facet.099 [concrete]
// CHECK:STDOUT:   %Destroy.facet.2cc: %Destroy.type = facet_value %Int16, (%custom_witness.e7a31c.1) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.9d5: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.2cc) [concrete]
// CHECK:STDOUT:   %.6da: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.9d5, %Destroy.facet.2cc [concrete]
// CHECK:STDOUT:   %BitComplement.type: type = facet_type <@BitComplement> [concrete]
// CHECK:STDOUT:   %.Self.276: %BitComplement.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %BitComplement.lookup_impl_witness.565: <witness> = lookup_impl_witness %.Self.276, @BitComplement [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.cd8: type = impl_witness_access %BitComplement.lookup_impl_witness.565, element0 [symbolic_self]
// CHECK:STDOUT:   %BitComplement_where.type.283: type = facet_type <@BitComplement where %impl.elem0.cd8 impls @Destroy> [concrete]
// CHECK:STDOUT:   %pattern_type.9fd: type = pattern_type %BitComplement_where.type.283 [concrete]
// CHECK:STDOUT:   %T.patt.d8c: %pattern_type.9fd = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T.438: %BitComplement_where.type.283 = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.e33: type = facet_access_type %T.438 [symbolic]
// CHECK:STDOUT:   %pattern_type.834f91.2: type = pattern_type %T.as_type.e33 [symbolic]
// CHECK:STDOUT:   %x.param_patt.b2f: %pattern_type.834f91.2 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %x.patt.647: %pattern_type.834f91.2 = at_binding_pattern x, %x.param_patt.b2f [symbolic]
// CHECK:STDOUT:   %BitComplement.lookup_impl_witness.e4757a.2: <witness> = lookup_impl_witness %T.438, @BitComplement [symbolic]
// CHECK:STDOUT:   %BitComplement.facet.5b0: %BitComplement.type = facet_value %T.as_type.e33, (%BitComplement.lookup_impl_witness.e4757a.2) [symbolic]
// CHECK:STDOUT:   %impl.elem0.ed21cd.2: type = impl_witness_access %BitComplement.lookup_impl_witness.e4757a.2, element0 [symbolic]
// CHECK:STDOUT:   %BitComplement.WithSelf.Op.type.c5e: type = fn_type @BitComplement.WithSelf.Op, @BitComplement.WithSelf(%BitComplement.facet.5b0) [symbolic]
// CHECK:STDOUT:   %.bf9: type = fn_type_with_self_type %BitComplement.WithSelf.Op.type.c5e, %BitComplement.facet.5b0 [symbolic]
// CHECK:STDOUT:   %impl.elem1.2ca: %.bf9 = impl_witness_access %BitComplement.lookup_impl_witness.e4757a.2, element1 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.856: <specific function> = specific_impl_function %impl.elem1.2ca, @BitComplement.WithSelf.Op(%BitComplement.facet.5b0) [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.9f9: <witness> = lookup_impl_witness %impl.elem0.ed21cd.2, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.10a: %Destroy.type = facet_value %impl.elem0.ed21cd.2, (%Destroy.lookup_impl_witness.9f9) [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.f14: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.10a) [symbolic]
// CHECK:STDOUT:   %.73b: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.f14, %Destroy.facet.10a [symbolic]
// CHECK:STDOUT:   %impl.elem0.39c: %.73b = impl_witness_access %Destroy.lookup_impl_witness.9f9, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.271: <specific function> = specific_impl_function %impl.elem0.39c, @Destroy.WithSelf.Op(%Destroy.facet.10a) [symbolic]
// CHECK:STDOUT:   %Int16.Op.type.83af6e.18: type = fn_type @Int16.Op.18 [concrete]
// CHECK:STDOUT:   %Int16.Op.4dca31.18: %Int16.Op.type.83af6e.18 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.2b7: <witness> = custom_witness (%Int64, %Int16.Op.4dca31.18), @BitComplement [concrete]
// CHECK:STDOUT:   %BitComplement.facet.610: %BitComplement.type = facet_value %Int16, (%custom_witness.2b7) [concrete]
// CHECK:STDOUT:   %facet_value.8bd: %BitComplement_where.type.283 = facet_value %Int16, (%custom_witness.2b7, %custom_witness.e7a31c.3) [concrete]
// CHECK:STDOUT:   %Int32.Op.type.83af6e.18: type = fn_type @Int32.Op.18 [concrete]
// CHECK:STDOUT:   %Int32.Op.4dca31.18: %Int32.Op.type.83af6e.18 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.f78: <witness> = custom_witness (%Int32, %Int32.Op.4dca31.18), @BitComplement [concrete]
// CHECK:STDOUT:   %BitComplement.facet.041: %BitComplement.type = facet_value %Int32, (%custom_witness.f78) [concrete]
// CHECK:STDOUT:   %facet_value.103: %BitComplement_where.type.283 = facet_value %Int32, (%custom_witness.e7a31c.2, %custom_witness.f78) [concrete]
// CHECK:STDOUT:   %BitAndWith.WithSelf.Op.type.c4e: type = fn_type @BitAndWith.WithSelf.Op, @BitAndWith.WithSelf(%Int16, %BitAndWith.facet.b2d) [concrete]
// CHECK:STDOUT:   %.566: type = fn_type_with_self_type %BitAndWith.WithSelf.Op.type.c4e, %BitAndWith.facet.b2d [concrete]
// CHECK:STDOUT:   %BitOrWith.WithSelf.Op.type.2a3: type = fn_type @BitOrWith.WithSelf.Op, @BitOrWith.WithSelf(%Int16, %BitOrWith.facet.ed1) [concrete]
// CHECK:STDOUT:   %.980: type = fn_type_with_self_type %BitOrWith.WithSelf.Op.type.2a3, %BitOrWith.facet.ed1 [concrete]
// CHECK:STDOUT:   %BitXorWith.WithSelf.Op.type.69e: type = fn_type @BitXorWith.WithSelf.Op, @BitXorWith.WithSelf(%Int16, %BitXorWith.facet.d55) [concrete]
// CHECK:STDOUT:   %.88e: type = fn_type_with_self_type %BitXorWith.WithSelf.Op.type.69e, %BitXorWith.facet.d55 [concrete]
// CHECK:STDOUT:   %LeftShiftWith.WithSelf.Op.type.2cb: type = fn_type @LeftShiftWith.WithSelf.Op, @LeftShiftWith.WithSelf(%Int16, %LeftShiftWith.facet.ce6) [concrete]
// CHECK:STDOUT:   %.8ba: type = fn_type_with_self_type %LeftShiftWith.WithSelf.Op.type.2cb, %LeftShiftWith.facet.ce6 [concrete]
// CHECK:STDOUT:   %RightShiftWith.WithSelf.Op.type.808: type = fn_type @RightShiftWith.WithSelf.Op, @RightShiftWith.WithSelf(%Int16, %RightShiftWith.facet.4d1) [concrete]
// CHECK:STDOUT:   %.c0a: type = fn_type_with_self_type %RightShiftWith.WithSelf.Op.type.808, %RightShiftWith.facet.4d1 [concrete]
// CHECK:STDOUT:   %BitAndWith.WithSelf.Op.type.37a: type = fn_type @BitAndWith.WithSelf.Op, @BitAndWith.WithSelf(%i16, %BitAndWith.facet.064) [concrete]
// CHECK:STDOUT:   %.732: type = fn_type_with_self_type %BitAndWith.WithSelf.Op.type.37a, %BitAndWith.facet.064 [concrete]
// CHECK:STDOUT:   %BitOrWith.WithSelf.Op.type.45d: type = fn_type @BitOrWith.WithSelf.Op, @BitOrWith.WithSelf(%i16, %BitOrWith.facet.bab) [concrete]
// CHECK:STDOUT:   %.855: type = fn_type_with_self_type %BitOrWith.WithSelf.Op.type.45d, %BitOrWith.facet.bab [concrete]
// CHECK:STDOUT:   %BitXorWith.WithSelf.Op.type.b89: type = fn_type @BitXorWith.WithSelf.Op, @BitXorWith.WithSelf(%i16, %BitXorWith.facet.7da) [concrete]
// CHECK:STDOUT:   %.15f: type = fn_type_with_self_type %BitXorWith.WithSelf.Op.type.b89, %BitXorWith.facet.7da [concrete]
// CHECK:STDOUT:   %LeftShiftWith.WithSelf.Op.type.c7d: type = fn_type @LeftShiftWith.WithSelf.Op, @LeftShiftWith.WithSelf(%i16, %LeftShiftWith.facet.071) [concrete]
// CHECK:STDOUT:   %.e97: type = fn_type_with_self_type %LeftShiftWith.WithSelf.Op.type.c7d, %LeftShiftWith.facet.071 [concrete]
// CHECK:STDOUT:   %RightShiftWith.WithSelf.Op.type.2ed: type = fn_type @RightShiftWith.WithSelf.Op, @RightShiftWith.WithSelf(%i16, %RightShiftWith.facet.fa8) [concrete]
// CHECK:STDOUT:   %.0c2: type = fn_type_with_self_type %RightShiftWith.WithSelf.Op.type.2ed, %RightShiftWith.facet.fa8 [concrete]
// CHECK:STDOUT:   %BitAndWith.WithSelf.Op.type.7c5: type = fn_type @BitAndWith.WithSelf.Op, @BitAndWith.WithSelf(%Int32, %BitAndWith.facet.fc0) [concrete]
// CHECK:STDOUT:   %.463: type = fn_type_with_self_type %BitAndWith.WithSelf.Op.type.7c5, %BitAndWith.facet.fc0 [concrete]
// CHECK:STDOUT:   %BitOrWith.WithSelf.Op.type.acf: type = fn_type @BitOrWith.WithSelf.Op, @BitOrWith.WithSelf(%Int32, %BitOrWith.facet.da7) [concrete]
// CHECK:STDOUT:   %.23d2b: type = fn_type_with_self_type %BitOrWith.WithSelf.Op.type.acf, %BitOrWith.facet.da7 [concrete]
// CHECK:STDOUT:   %BitXorWith.WithSelf.Op.type.713: type = fn_type @BitXorWith.WithSelf.Op, @BitXorWith.WithSelf(%Int32, %BitXorWith.facet.121) [concrete]
// CHECK:STDOUT:   %.ac9: type = fn_type_with_self_type %BitXorWith.WithSelf.Op.type.713, %BitXorWith.facet.121 [concrete]
// CHECK:STDOUT:   %LeftShiftWith.WithSelf.Op.type.b2c: type = fn_type @LeftShiftWith.WithSelf.Op, @LeftShiftWith.WithSelf(%Int32, %LeftShiftWith.facet.4c3e) [concrete]
// CHECK:STDOUT:   %.489: type = fn_type_with_self_type %LeftShiftWith.WithSelf.Op.type.b2c, %LeftShiftWith.facet.4c3e [concrete]
// CHECK:STDOUT:   %RightShiftWith.WithSelf.Op.type.66e: type = fn_type @RightShiftWith.WithSelf.Op, @RightShiftWith.WithSelf(%Int32, %RightShiftWith.facet.0ca) [concrete]
// CHECK:STDOUT:   %.e00: type = fn_type_with_self_type %RightShiftWith.WithSelf.Op.type.66e, %RightShiftWith.facet.0ca [concrete]
// CHECK:STDOUT:   %BitAndWith.WithSelf.Op.type.ffe: type = fn_type @BitAndWith.WithSelf.Op, @BitAndWith.WithSelf(%Int16, %BitAndWith.facet.58a) [concrete]
// CHECK:STDOUT:   %.1f1: type = fn_type_with_self_type %BitAndWith.WithSelf.Op.type.ffe, %BitAndWith.facet.58a [concrete]
// CHECK:STDOUT:   %BitOrWith.WithSelf.Op.type.b5d: type = fn_type @BitOrWith.WithSelf.Op, @BitOrWith.WithSelf(%Int16, %BitOrWith.facet.c7b) [concrete]
// CHECK:STDOUT:   %.fb6: type = fn_type_with_self_type %BitOrWith.WithSelf.Op.type.b5d, %BitOrWith.facet.c7b [concrete]
// CHECK:STDOUT:   %BitXorWith.WithSelf.Op.type.100: type = fn_type @BitXorWith.WithSelf.Op, @BitXorWith.WithSelf(%Int16, %BitXorWith.facet.ac3) [concrete]
// CHECK:STDOUT:   %.8ad: type = fn_type_with_self_type %BitXorWith.WithSelf.Op.type.100, %BitXorWith.facet.ac3 [concrete]
// CHECK:STDOUT:   %LeftShiftWith.WithSelf.Op.type.1bb: type = fn_type @LeftShiftWith.WithSelf.Op, @LeftShiftWith.WithSelf(%Int16, %LeftShiftWith.facet.453) [concrete]
// CHECK:STDOUT:   %.359: type = fn_type_with_self_type %LeftShiftWith.WithSelf.Op.type.1bb, %LeftShiftWith.facet.453 [concrete]
// CHECK:STDOUT:   %RightShiftWith.WithSelf.Op.type.ea7: type = fn_type @RightShiftWith.WithSelf.Op, @RightShiftWith.WithSelf(%Int16, %RightShiftWith.facet.714) [concrete]
// CHECK:STDOUT:   %.5d9: type = fn_type_with_self_type %RightShiftWith.WithSelf.Op.type.ea7, %RightShiftWith.facet.714 [concrete]
// CHECK:STDOUT:   %BitAndWith.WithSelf.Op.type.7e2: type = fn_type @BitAndWith.WithSelf.Op, @BitAndWith.WithSelf(%Int32, %BitAndWith.facet.2d7) [concrete]
// CHECK:STDOUT:   %.09e: type = fn_type_with_self_type %BitAndWith.WithSelf.Op.type.7e2, %BitAndWith.facet.2d7 [concrete]
// CHECK:STDOUT:   %BitOrWith.WithSelf.Op.type.172: type = fn_type @BitOrWith.WithSelf.Op, @BitOrWith.WithSelf(%Int32, %BitOrWith.facet.f97) [concrete]
// CHECK:STDOUT:   %.62c: type = fn_type_with_self_type %BitOrWith.WithSelf.Op.type.172, %BitOrWith.facet.f97 [concrete]
// CHECK:STDOUT:   %BitXorWith.WithSelf.Op.type.0df: type = fn_type @BitXorWith.WithSelf.Op, @BitXorWith.WithSelf(%Int32, %BitXorWith.facet.4e5) [concrete]
// CHECK:STDOUT:   %.6aa: type = fn_type_with_self_type %BitXorWith.WithSelf.Op.type.0df, %BitXorWith.facet.4e5 [concrete]
// CHECK:STDOUT:   %LeftShiftWith.WithSelf.Op.type.2be: type = fn_type @LeftShiftWith.WithSelf.Op, @LeftShiftWith.WithSelf(%Int32, %LeftShiftWith.facet.0bb) [concrete]
// CHECK:STDOUT:   %.de5: type = fn_type_with_self_type %LeftShiftWith.WithSelf.Op.type.2be, %LeftShiftWith.facet.0bb [concrete]
// CHECK:STDOUT:   %RightShiftWith.WithSelf.Op.type.43d: type = fn_type @RightShiftWith.WithSelf.Op, @RightShiftWith.WithSelf(%Int32, %RightShiftWith.facet.0bb) [concrete]
// CHECK:STDOUT:   %.ae2: type = fn_type_with_self_type %RightShiftWith.WithSelf.Op.type.43d, %RightShiftWith.facet.0bb [concrete]
// CHECK:STDOUT:   %BitAndWith.WithSelf.Op.type.e00: type = fn_type @BitAndWith.WithSelf.Op, @BitAndWith.WithSelf(%Int64, %BitAndWith.facet.401) [concrete]
// CHECK:STDOUT:   %.0ba: type = fn_type_with_self_type %BitAndWith.WithSelf.Op.type.e00, %BitAndWith.facet.401 [concrete]
// CHECK:STDOUT:   %BitOrWith.WithSelf.Op.type.d10: type = fn_type @BitOrWith.WithSelf.Op, @BitOrWith.WithSelf(%Int64, %BitOrWith.facet.7a3) [concrete]
// CHECK:STDOUT:   %.a76: type = fn_type_with_self_type %BitOrWith.WithSelf.Op.type.d10, %BitOrWith.facet.7a3 [concrete]
// CHECK:STDOUT:   %BitXorWith.WithSelf.Op.type.946: type = fn_type @BitXorWith.WithSelf.Op, @BitXorWith.WithSelf(%Int64, %BitXorWith.facet.073) [concrete]
// CHECK:STDOUT:   %.82d: type = fn_type_with_self_type %BitXorWith.WithSelf.Op.type.946, %BitXorWith.facet.073 [concrete]
// CHECK:STDOUT:   %LeftShiftWith.WithSelf.Op.type.56e: type = fn_type @LeftShiftWith.WithSelf.Op, @LeftShiftWith.WithSelf(%Int64, %LeftShiftWith.facet.3c0) [concrete]
// CHECK:STDOUT:   %.f1e: type = fn_type_with_self_type %LeftShiftWith.WithSelf.Op.type.56e, %LeftShiftWith.facet.3c0 [concrete]
// CHECK:STDOUT:   %RightShiftWith.WithSelf.Op.type.719: type = fn_type @RightShiftWith.WithSelf.Op, @RightShiftWith.WithSelf(%Int64, %RightShiftWith.facet.e89) [concrete]
// CHECK:STDOUT:   %.7e1d: type = fn_type_with_self_type %RightShiftWith.WithSelf.Op.type.719, %RightShiftWith.facet.e89 [concrete]
// CHECK:STDOUT:   %BitAndWith.WithSelf.Op.type.953: type = fn_type @BitAndWith.WithSelf.Op, @BitAndWith.WithSelf(%Int32, %BitAndWith.facet.7b2) [concrete]
// CHECK:STDOUT:   %.b2a: type = fn_type_with_self_type %BitAndWith.WithSelf.Op.type.953, %BitAndWith.facet.7b2 [concrete]
// CHECK:STDOUT:   %BitOrWith.WithSelf.Op.type.406: type = fn_type @BitOrWith.WithSelf.Op, @BitOrWith.WithSelf(%Int32, %BitOrWith.facet.b65) [concrete]
// CHECK:STDOUT:   %.549: type = fn_type_with_self_type %BitOrWith.WithSelf.Op.type.406, %BitOrWith.facet.b65 [concrete]
// CHECK:STDOUT:   %BitXorWith.WithSelf.Op.type.668: type = fn_type @BitXorWith.WithSelf.Op, @BitXorWith.WithSelf(%Int32, %BitXorWith.facet.4a6) [concrete]
// CHECK:STDOUT:   %.4ea: type = fn_type_with_self_type %BitXorWith.WithSelf.Op.type.668, %BitXorWith.facet.4a6 [concrete]
// CHECK:STDOUT:   %LeftShiftWith.WithSelf.Op.type.0a2: type = fn_type @LeftShiftWith.WithSelf.Op, @LeftShiftWith.WithSelf(%Int32, %LeftShiftWith.facet.22c) [concrete]
// CHECK:STDOUT:   %.c1d: type = fn_type_with_self_type %LeftShiftWith.WithSelf.Op.type.0a2, %LeftShiftWith.facet.22c [concrete]
// CHECK:STDOUT:   %RightShiftWith.WithSelf.Op.type.bc1: type = fn_type @RightShiftWith.WithSelf.Op, @RightShiftWith.WithSelf(%Int32, %RightShiftWith.facet.ffd) [concrete]
// CHECK:STDOUT:   %.54e: type = fn_type_with_self_type %RightShiftWith.WithSelf.Op.type.bc1, %RightShiftWith.facet.ffd [concrete]
// CHECK:STDOUT:   %complete_type.9ba: <witness> = complete_type_witness %BitAndAssignWith.type.0b5 [concrete]
// CHECK:STDOUT:   %BitAndAssignWith.WithSelf.Op.type.daa: type = fn_type @BitAndAssignWith.WithSelf.Op, @BitAndAssignWith.WithSelf(%i16, %BitAndAssignWith.facet.355) [concrete]
// CHECK:STDOUT:   %.483: type = fn_type_with_self_type %BitAndAssignWith.WithSelf.Op.type.daa, %BitAndAssignWith.facet.355 [concrete]
// CHECK:STDOUT:   %complete_type.650: <witness> = complete_type_witness %BitOrAssignWith.type.8ff [concrete]
// CHECK:STDOUT:   %BitOrAssignWith.WithSelf.Op.type.581: type = fn_type @BitOrAssignWith.WithSelf.Op, @BitOrAssignWith.WithSelf(%i16, %BitOrAssignWith.facet.ea2) [concrete]
// CHECK:STDOUT:   %.233: type = fn_type_with_self_type %BitOrAssignWith.WithSelf.Op.type.581, %BitOrAssignWith.facet.ea2 [concrete]
// CHECK:STDOUT:   %complete_type.054: <witness> = complete_type_witness %BitXorAssignWith.type.1ee [concrete]
// CHECK:STDOUT:   %BitXorAssignWith.WithSelf.Op.type.0db: type = fn_type @BitXorAssignWith.WithSelf.Op, @BitXorAssignWith.WithSelf(%i16, %BitXorAssignWith.facet.84f) [concrete]
// CHECK:STDOUT:   %.298: type = fn_type_with_self_type %BitXorAssignWith.WithSelf.Op.type.0db, %BitXorAssignWith.facet.84f [concrete]
// CHECK:STDOUT:   %complete_type.454: <witness> = complete_type_witness %LeftShiftAssignWith.type.62a [concrete]
// CHECK:STDOUT:   %LeftShiftAssignWith.WithSelf.Op.type.412: type = fn_type @LeftShiftAssignWith.WithSelf.Op, @LeftShiftAssignWith.WithSelf(%i16, %LeftShiftAssignWith.facet.71f) [concrete]
// CHECK:STDOUT:   %.308: type = fn_type_with_self_type %LeftShiftAssignWith.WithSelf.Op.type.412, %LeftShiftAssignWith.facet.71f [concrete]
// CHECK:STDOUT:   %complete_type.f10: <witness> = complete_type_witness %RightShiftAssignWith.type.b1f [concrete]
// CHECK:STDOUT:   %RightShiftAssignWith.WithSelf.Op.type.5a9: type = fn_type @RightShiftAssignWith.WithSelf.Op, @RightShiftAssignWith.WithSelf(%i16, %RightShiftAssignWith.facet.f54) [concrete]
// CHECK:STDOUT:   %.6bf: type = fn_type_with_self_type %RightShiftAssignWith.WithSelf.Op.type.5a9, %RightShiftAssignWith.facet.f54 [concrete]
// CHECK:STDOUT:   %complete_type.824: <witness> = complete_type_witness %BitAndAssignWith.type.53c [concrete]
// CHECK:STDOUT:   %BitAndAssignWith.WithSelf.Op.type.3ea: type = fn_type @BitAndAssignWith.WithSelf.Op, @BitAndAssignWith.WithSelf(%Int32, %BitAndAssignWith.facet.5d5) [concrete]
// CHECK:STDOUT:   %.0eb: type = fn_type_with_self_type %BitAndAssignWith.WithSelf.Op.type.3ea, %BitAndAssignWith.facet.5d5 [concrete]
// CHECK:STDOUT:   %complete_type.c85: <witness> = complete_type_witness %BitOrAssignWith.type.46f [concrete]
// CHECK:STDOUT:   %BitOrAssignWith.WithSelf.Op.type.877: type = fn_type @BitOrAssignWith.WithSelf.Op, @BitOrAssignWith.WithSelf(%Int32, %BitOrAssignWith.facet.ad2) [concrete]
// CHECK:STDOUT:   %.999: type = fn_type_with_self_type %BitOrAssignWith.WithSelf.Op.type.877, %BitOrAssignWith.facet.ad2 [concrete]
// CHECK:STDOUT:   %complete_type.267: <witness> = complete_type_witness %BitXorAssignWith.type.0e4 [concrete]
// CHECK:STDOUT:   %BitXorAssignWith.WithSelf.Op.type.243: type = fn_type @BitXorAssignWith.WithSelf.Op, @BitXorAssignWith.WithSelf(%Int32, %BitXorAssignWith.facet.627) [concrete]
// CHECK:STDOUT:   %.384: type = fn_type_with_self_type %BitXorAssignWith.WithSelf.Op.type.243, %BitXorAssignWith.facet.627 [concrete]
// CHECK:STDOUT:   %complete_type.bf0: <witness> = complete_type_witness %LeftShiftAssignWith.type.635 [concrete]
// CHECK:STDOUT:   %LeftShiftAssignWith.WithSelf.Op.type.195: type = fn_type @LeftShiftAssignWith.WithSelf.Op, @LeftShiftAssignWith.WithSelf(%Int32, %LeftShiftAssignWith.facet.fb8) [concrete]
// CHECK:STDOUT:   %.8fd: type = fn_type_with_self_type %LeftShiftAssignWith.WithSelf.Op.type.195, %LeftShiftAssignWith.facet.fb8 [concrete]
// CHECK:STDOUT:   %complete_type.f23: <witness> = complete_type_witness %RightShiftAssignWith.type.239 [concrete]
// CHECK:STDOUT:   %RightShiftAssignWith.WithSelf.Op.type.738: type = fn_type @RightShiftAssignWith.WithSelf.Op, @RightShiftAssignWith.WithSelf(%Int32, %RightShiftAssignWith.facet.288) [concrete]
// CHECK:STDOUT:   %.985: type = fn_type_with_self_type %RightShiftAssignWith.WithSelf.Op.type.738, %RightShiftAssignWith.facet.288 [concrete]
// CHECK:STDOUT:   %BitAndAssignWith.WithSelf.Op.type.896: type = fn_type @BitAndAssignWith.WithSelf.Op, @BitAndAssignWith.WithSelf(%Int32, %BitAndAssignWith.facet.cd3) [concrete]
// CHECK:STDOUT:   %.c50: type = fn_type_with_self_type %BitAndAssignWith.WithSelf.Op.type.896, %BitAndAssignWith.facet.cd3 [concrete]
// CHECK:STDOUT:   %BitOrAssignWith.WithSelf.Op.type.18d: type = fn_type @BitOrAssignWith.WithSelf.Op, @BitOrAssignWith.WithSelf(%Int32, %BitOrAssignWith.facet.c6e) [concrete]
// CHECK:STDOUT:   %.5f2: type = fn_type_with_self_type %BitOrAssignWith.WithSelf.Op.type.18d, %BitOrAssignWith.facet.c6e [concrete]
// CHECK:STDOUT:   %BitXorAssignWith.WithSelf.Op.type.198: type = fn_type @BitXorAssignWith.WithSelf.Op, @BitXorAssignWith.WithSelf(%Int32, %BitXorAssignWith.facet.896) [concrete]
// CHECK:STDOUT:   %.52d: type = fn_type_with_self_type %BitXorAssignWith.WithSelf.Op.type.198, %BitXorAssignWith.facet.896 [concrete]
// CHECK:STDOUT:   %LeftShiftAssignWith.WithSelf.Op.type.f6f: type = fn_type @LeftShiftAssignWith.WithSelf.Op, @LeftShiftAssignWith.WithSelf(%Int32, %LeftShiftAssignWith.facet.8cb) [concrete]
// CHECK:STDOUT:   %.a80: type = fn_type_with_self_type %LeftShiftAssignWith.WithSelf.Op.type.f6f, %LeftShiftAssignWith.facet.8cb [concrete]
// CHECK:STDOUT:   %RightShiftAssignWith.WithSelf.Op.type.2ee: type = fn_type @RightShiftAssignWith.WithSelf.Op, @RightShiftAssignWith.WithSelf(%Int32, %RightShiftAssignWith.facet.040) [concrete]
// CHECK:STDOUT:   %.7e16: type = fn_type_with_self_type %RightShiftAssignWith.WithSelf.Op.type.2ee, %RightShiftAssignWith.facet.040 [concrete]
// CHECK:STDOUT:   %BitComplement.WithSelf.Op.type.80e: type = fn_type @BitComplement.WithSelf.Op, @BitComplement.WithSelf(%BitComplement.facet.610) [concrete]
// CHECK:STDOUT:   %.c87: type = fn_type_with_self_type %BitComplement.WithSelf.Op.type.80e, %BitComplement.facet.610 [concrete]
// CHECK:STDOUT:   %BitComplement.WithSelf.Op.type.8e2: type = fn_type @BitComplement.WithSelf.Op, @BitComplement.WithSelf(%BitComplement.facet.041) [concrete]
// CHECK:STDOUT:   %.681: type = fn_type_with_self_type %BitComplement.WithSelf.Op.type.8e2, %BitComplement.facet.041 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.b66: @BitAndWith.WithSelf.%BitAndWith.assoc_type (%BitAndWith.assoc_type.a6e8f9.1) = import_ref Core//prelude/operators/bitwise, loc{{\d+_\d+}}, loaded [symbolic = @BitAndWith.WithSelf.%assoc1 (constants.%assoc1.85c)]
// CHECK:STDOUT:   %Core.import_ref.2c4: @BitAndWith.WithSelf.%BitAndWith.WithSelf.Op.type (%BitAndWith.WithSelf.Op.type.4e1bca.1) = import_ref Core//prelude/operators/bitwise, loc{{\d+_\d+}}, loaded [symbolic = @BitAndWith.WithSelf.%BitAndWith.WithSelf.Op (constants.%BitAndWith.WithSelf.Op.549040.1)]
// CHECK:STDOUT:   %Core.import_ref.efd: type = import_ref Core//prelude/operators/bitwise, loc{{\d+_\d+}}, loaded [concrete = %Result.25a]
// CHECK:STDOUT:   %Result.25a: type = assoc_const_decl @Result.1 [concrete] {}
// CHECK:STDOUT:   %Core.import_ref.a28 = import_ref Core//prelude/operators/bitwise, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.f87: @BitOrWith.WithSelf.%BitOrWith.assoc_type (%BitOrWith.assoc_type.10d7b7.1) = import_ref Core//prelude/operators/bitwise, loc{{\d+_\d+}}, loaded [symbolic = @BitOrWith.WithSelf.%assoc1 (constants.%assoc1.25a)]
// CHECK:STDOUT:   %Core.import_ref.c14: @BitOrWith.WithSelf.%BitOrWith.WithSelf.Op.type (%BitOrWith.WithSelf.Op.type.0e2995.1) = import_ref Core//prelude/operators/bitwise, loc{{\d+_\d+}}, loaded [symbolic = @BitOrWith.WithSelf.%BitOrWith.WithSelf.Op (constants.%BitOrWith.WithSelf.Op.d71e20.1)]
// CHECK:STDOUT:   %Core.import_ref.4f6: type = import_ref Core//prelude/operators/bitwise, loc{{\d+_\d+}}, loaded [concrete = %Result.801]
// CHECK:STDOUT:   %Result.801: type = assoc_const_decl @Result.2 [concrete] {}
// CHECK:STDOUT:   %Core.import_ref.532 = import_ref Core//prelude/operators/bitwise, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.68e: @BitXorWith.WithSelf.%BitXorWith.assoc_type (%BitXorWith.assoc_type.4ad612.1) = import_ref Core//prelude/operators/bitwise, loc{{\d+_\d+}}, loaded [symbolic = @BitXorWith.WithSelf.%assoc1 (constants.%assoc1.fb3)]
// CHECK:STDOUT:   %Core.import_ref.808: @BitXorWith.WithSelf.%BitXorWith.WithSelf.Op.type (%BitXorWith.WithSelf.Op.type.2f1ddb.1) = import_ref Core//prelude/operators/bitwise, loc{{\d+_\d+}}, loaded [symbolic = @BitXorWith.WithSelf.%BitXorWith.WithSelf.Op (constants.%BitXorWith.WithSelf.Op.b3b3a4.1)]
// CHECK:STDOUT:   %Core.import_ref.37d: type = import_ref Core//prelude/operators/bitwise, loc{{\d+_\d+}}, loaded [concrete = %Result.8c1]
// CHECK:STDOUT:   %Result.8c1: type = assoc_const_decl @Result.3 [concrete] {}
// CHECK:STDOUT:   %Core.import_ref.63c = import_ref Core//prelude/operators/bitwise, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.3b3: @LeftShiftWith.WithSelf.%LeftShiftWith.assoc_type (%LeftShiftWith.assoc_type.739d6a.1) = import_ref Core//prelude/operators/bitwise, loc{{\d+_\d+}}, loaded [symbolic = @LeftShiftWith.WithSelf.%assoc1 (constants.%assoc1.113)]
// CHECK:STDOUT:   %Core.import_ref.36a: @LeftShiftWith.WithSelf.%LeftShiftWith.WithSelf.Op.type (%LeftShiftWith.WithSelf.Op.type.30f247.1) = import_ref Core//prelude/operators/bitwise, loc{{\d+_\d+}}, loaded [symbolic = @LeftShiftWith.WithSelf.%LeftShiftWith.WithSelf.Op (constants.%LeftShiftWith.WithSelf.Op.4ac30d.1)]
// CHECK:STDOUT:   %Core.import_ref.1a5: type = import_ref Core//prelude/operators/bitwise, loc{{\d+_\d+}}, loaded [concrete = %Result.f24]
// CHECK:STDOUT:   %Result.f24: type = assoc_const_decl @Result.4 [concrete] {}
// CHECK:STDOUT:   %Core.import_ref.5d8 = import_ref Core//prelude/operators/bitwise, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.573: @RightShiftWith.WithSelf.%RightShiftWith.assoc_type (%RightShiftWith.assoc_type.4d2607.1) = import_ref Core//prelude/operators/bitwise, loc{{\d+_\d+}}, loaded [symbolic = @RightShiftWith.WithSelf.%assoc1 (constants.%assoc1.c0c)]
// CHECK:STDOUT:   %Core.import_ref.e6b: @RightShiftWith.WithSelf.%RightShiftWith.WithSelf.Op.type (%RightShiftWith.WithSelf.Op.type.47e321.1) = import_ref Core//prelude/operators/bitwise, loc{{\d+_\d+}}, loaded [symbolic = @RightShiftWith.WithSelf.%RightShiftWith.WithSelf.Op (constants.%RightShiftWith.WithSelf.Op.bec3f0.1)]
// CHECK:STDOUT:   %Core.import_ref.416: type = import_ref Core//prelude/operators/bitwise, loc{{\d+_\d+}}, loaded [concrete = %Result.c31]
// CHECK:STDOUT:   %Result.c31: type = assoc_const_decl @Result.5 [concrete] {}
// CHECK:STDOUT:   %Core.import_ref.e42 = import_ref Core//prelude/operators/bitwise, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.feb0: @BitAndAssignWith.WithSelf.%BitAndAssignWith.assoc_type (%BitAndAssignWith.assoc_type.5d10e3.1) = import_ref Core//prelude/operators/bitwise, loc{{\d+_\d+}}, loaded [symbolic = @BitAndAssignWith.WithSelf.%assoc0 (constants.%assoc0.4dc)]
// CHECK:STDOUT:   %Core.import_ref.618: @BitAndAssignWith.WithSelf.%BitAndAssignWith.WithSelf.Op.type (%BitAndAssignWith.WithSelf.Op.type.e3224f.1) = import_ref Core//prelude/operators/bitwise, loc{{\d+_\d+}}, loaded [symbolic = @BitAndAssignWith.WithSelf.%BitAndAssignWith.WithSelf.Op (constants.%BitAndAssignWith.WithSelf.Op.b3572a.1)]
// CHECK:STDOUT:   %Core.import_ref.4c2 = import_ref Core//prelude/operators/bitwise, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.8034: @BitOrAssignWith.WithSelf.%BitOrAssignWith.assoc_type (%BitOrAssignWith.assoc_type.e05d8b.1) = import_ref Core//prelude/operators/bitwise, loc{{\d+_\d+}}, loaded [symbolic = @BitOrAssignWith.WithSelf.%assoc0 (constants.%assoc0.be6)]
// CHECK:STDOUT:   %Core.import_ref.b03: @BitOrAssignWith.WithSelf.%BitOrAssignWith.WithSelf.Op.type (%BitOrAssignWith.WithSelf.Op.type.130458.1) = import_ref Core//prelude/operators/bitwise, loc{{\d+_\d+}}, loaded [symbolic = @BitOrAssignWith.WithSelf.%BitOrAssignWith.WithSelf.Op (constants.%BitOrAssignWith.WithSelf.Op.53e52f.1)]
// CHECK:STDOUT:   %Core.import_ref.7fc = import_ref Core//prelude/operators/bitwise, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.6f9: @BitXorAssignWith.WithSelf.%BitXorAssignWith.assoc_type (%BitXorAssignWith.assoc_type.2be5ac.1) = import_ref Core//prelude/operators/bitwise, loc{{\d+_\d+}}, loaded [symbolic = @BitXorAssignWith.WithSelf.%assoc0 (constants.%assoc0.ef6)]
// CHECK:STDOUT:   %Core.import_ref.194: @BitXorAssignWith.WithSelf.%BitXorAssignWith.WithSelf.Op.type (%BitXorAssignWith.WithSelf.Op.type.8f9d71.1) = import_ref Core//prelude/operators/bitwise, loc{{\d+_\d+}}, loaded [symbolic = @BitXorAssignWith.WithSelf.%BitXorAssignWith.WithSelf.Op (constants.%BitXorAssignWith.WithSelf.Op.848804.1)]
// CHECK:STDOUT:   %Core.import_ref.d20 = import_ref Core//prelude/operators/bitwise, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.6ac: @LeftShiftAssignWith.WithSelf.%LeftShiftAssignWith.assoc_type (%LeftShiftAssignWith.assoc_type.239476.1) = import_ref Core//prelude/operators/bitwise, loc{{\d+_\d+}}, loaded [symbolic = @LeftShiftAssignWith.WithSelf.%assoc0 (constants.%assoc0.ad5)]
// CHECK:STDOUT:   %Core.import_ref.2b9: @LeftShiftAssignWith.WithSelf.%LeftShiftAssignWith.WithSelf.Op.type (%LeftShiftAssignWith.WithSelf.Op.type.ce10f9.1) = import_ref Core//prelude/operators/bitwise, loc{{\d+_\d+}}, loaded [symbolic = @LeftShiftAssignWith.WithSelf.%LeftShiftAssignWith.WithSelf.Op (constants.%LeftShiftAssignWith.WithSelf.Op.308429.1)]
// CHECK:STDOUT:   %Core.import_ref.48dce = import_ref Core//prelude/operators/bitwise, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.eebc: @RightShiftAssignWith.WithSelf.%RightShiftAssignWith.assoc_type (%RightShiftAssignWith.assoc_type.95eeb2.1) = import_ref Core//prelude/operators/bitwise, loc{{\d+_\d+}}, loaded [symbolic = @RightShiftAssignWith.WithSelf.%assoc0 (constants.%assoc0.c59)]
// CHECK:STDOUT:   %Core.import_ref.0e8: @RightShiftAssignWith.WithSelf.%RightShiftAssignWith.WithSelf.Op.type (%RightShiftAssignWith.WithSelf.Op.type.f62b2f.1) = import_ref Core//prelude/operators/bitwise, loc{{\d+_\d+}}, loaded [symbolic = @RightShiftAssignWith.WithSelf.%RightShiftAssignWith.WithSelf.Op (constants.%RightShiftAssignWith.WithSelf.Op.500e9a.1)]
// CHECK:STDOUT:   %Core.import_ref.88f = import_ref Core//prelude/operators/bitwise, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @BitAndWith.loc112(%U.loc110_4.2: type, %T.loc111_4.2: @BitAndWith.loc112.%BitAndWith_where.type (%BitAndWith_where.type.4f1)) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %assoc1: @BitAndWith.loc112.%BitAndWith.assoc_type (%BitAndWith.assoc_type.a6e8f9.2) = assoc_entity element1, imports.%Core.import_ref.2c4 [symbolic = %assoc1 (constants.%assoc1.acd56d.2)]
// CHECK:STDOUT:   %BitAndWith.lookup_impl_witness.loc114: <witness> = lookup_impl_witness %T.loc111_4.1, @BitAndWith.1, @BitAndWith.1(%U.loc110_4.1) [symbolic = %BitAndWith.lookup_impl_witness.loc114 (constants.%BitAndWith.lookup_impl_witness.25c49f.2)]
// CHECK:STDOUT:   %BitAndWith.facet: @BitAndWith.loc112.%BitAndWith.type.loc111_24.1 (%BitAndWith.type.4f130e.2) = facet_value %T.as_type.loc112_6.1, (%BitAndWith.lookup_impl_witness.loc114) [symbolic = %BitAndWith.facet (constants.%BitAndWith.facet.e71)]
// CHECK:STDOUT:   %BitAndWith.WithSelf.Op.type: type = fn_type @BitAndWith.WithSelf.Op, @BitAndWith.WithSelf(%U.loc110_4.1, %BitAndWith.facet) [symbolic = %BitAndWith.WithSelf.Op.type (constants.%BitAndWith.WithSelf.Op.type.4f2)]
// CHECK:STDOUT:   %.loc114_5.6: type = fn_type_with_self_type %BitAndWith.WithSelf.Op.type, %BitAndWith.facet [symbolic = %.loc114_5.6 (constants.%.0cdb)]
// CHECK:STDOUT:   %impl.elem1.loc114_5.2: @BitAndWith.loc112.%.loc114_5.6 (%.0cdb) = impl_witness_access %BitAndWith.lookup_impl_witness.loc114, element1 [symbolic = %impl.elem1.loc114_5.2 (constants.%impl.elem1.69c)]
// CHECK:STDOUT:   %specific_impl_fn.loc114_5.3: <specific function> = specific_impl_function %impl.elem1.loc114_5.2, @BitAndWith.WithSelf.Op(%U.loc110_4.1, %BitAndWith.facet) [symbolic = %specific_impl_fn.loc114_5.3 (constants.%specific_impl_fn.cfc)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %impl.elem0.1, @Destroy [symbolic = %Destroy.lookup_impl_witness (constants.%Destroy.lookup_impl_witness.61e)]
// CHECK:STDOUT:   %Destroy.facet.loc114_5.3: %Destroy.type = facet_value %impl.elem0.1, (%Destroy.lookup_impl_witness) [symbolic = %Destroy.facet.loc114_5.3 (constants.%Destroy.facet.c83)]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.loc114_5.3) [symbolic = %Destroy.WithSelf.Op.type (constants.%Destroy.WithSelf.Op.type.230)]
// CHECK:STDOUT:   %.loc114_5.7: type = fn_type_with_self_type %Destroy.WithSelf.Op.type, %Destroy.facet.loc114_5.3 [symbolic = %.loc114_5.7 (constants.%.8bd)]
// CHECK:STDOUT:   %impl.elem0.loc114_5.2: @BitAndWith.loc112.%.loc114_5.7 (%.8bd) = impl_witness_access %Destroy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc114_5.2 (constants.%impl.elem0.5a7)]
// CHECK:STDOUT:   %specific_impl_fn.loc114_5.4: <specific function> = specific_impl_function %impl.elem0.loc114_5.2, @Destroy.WithSelf.Op(%Destroy.facet.loc114_5.3) [symbolic = %specific_impl_fn.loc114_5.4 (constants.%specific_impl_fn.5a8)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @BitAndWith.loc112.%T.as_type.loc112_6.1 (%T.as_type.417), %y.param: @BitAndWith.loc112.%U.loc110_4.1 (%U.67d)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: @BitAndWith.loc112.%T.as_type.loc112_6.1 (%T.as_type.417) = name_ref x, %x
// CHECK:STDOUT:     %y.ref: @BitAndWith.loc112.%U.loc110_4.1 (%U.67d) = name_ref y, %y
// CHECK:STDOUT:     %BitAndWith.type.loc114: type = facet_type <@BitAndWith.1, @BitAndWith.1(constants.%U.67d)> [symbolic = %BitAndWith.type.loc111_24.1 (constants.%BitAndWith.type.4f130e.2)]
// CHECK:STDOUT:     %.loc114_5.1: @BitAndWith.loc112.%BitAndWith.assoc_type (%BitAndWith.assoc_type.a6e8f9.2) = specific_constant imports.%Core.import_ref.b66, @BitAndWith.WithSelf(constants.%U.67d, constants.%Self.274af9.1) [symbolic = %assoc1 (constants.%assoc1.acd56d.2)]
// CHECK:STDOUT:     %Op.ref: @BitAndWith.loc112.%BitAndWith.assoc_type (%BitAndWith.assoc_type.a6e8f9.2) = name_ref Op, %.loc114_5.1 [symbolic = %assoc1 (constants.%assoc1.acd56d.2)]
// CHECK:STDOUT:     %impl.elem1.loc114_5.1: @BitAndWith.loc112.%.loc114_5.6 (%.0cdb) = impl_witness_access constants.%BitAndWith.lookup_impl_witness.25c49f.2, element1 [symbolic = %impl.elem1.loc114_5.2 (constants.%impl.elem1.69c)]
// CHECK:STDOUT:     %bound_method.loc114_5.1: <bound method> = bound_method %x.ref, %impl.elem1.loc114_5.1
// CHECK:STDOUT:     %specific_impl_fn.loc114_5.1: <specific function> = specific_impl_function %impl.elem1.loc114_5.1, @BitAndWith.WithSelf.Op(constants.%U.67d, constants.%BitAndWith.facet.e71) [symbolic = %specific_impl_fn.loc114_5.3 (constants.%specific_impl_fn.cfc)]
// CHECK:STDOUT:     %bound_method.loc114_5.2: <bound method> = bound_method %x.ref, %specific_impl_fn.loc114_5.1
// CHECK:STDOUT:     %.loc114_5.2: ref @BitAndWith.loc112.%impl.elem0.1 (%impl.elem0.7c98f0.2) = temporary_storage
// CHECK:STDOUT:     %BitAndWith.WithSelf.Op.call: init @BitAndWith.loc112.%impl.elem0.1 (%impl.elem0.7c98f0.2) to %.loc114_5.2 = call %bound_method.loc114_5.2(%x.ref, %y.ref)
// CHECK:STDOUT:     %.loc114_5.3: ref @BitAndWith.loc112.%impl.elem0.1 (%impl.elem0.7c98f0.2) = temporary %.loc114_5.2, %BitAndWith.WithSelf.Op.call
// CHECK:STDOUT:     %impl.elem0.loc114_5.1: @BitAndWith.loc112.%.loc114_5.7 (%.8bd) = impl_witness_access constants.%Destroy.lookup_impl_witness.61e, element0 [symbolic = %impl.elem0.loc114_5.2 (constants.%impl.elem0.5a7)]
// CHECK:STDOUT:     %bound_method.loc114_5.3: <bound method> = bound_method %.loc114_5.3, %impl.elem0.loc114_5.1
// CHECK:STDOUT:     %Destroy.facet.loc114_5.1: %Destroy.type = facet_value constants.%impl.elem0.7c98f0.2, (constants.%Destroy.lookup_impl_witness.61e) [symbolic = %Destroy.facet.loc114_5.3 (constants.%Destroy.facet.c83)]
// CHECK:STDOUT:     %.loc114_5.4: %Destroy.type = converted constants.%impl.elem0.7c98f0.2, %Destroy.facet.loc114_5.1 [symbolic = %Destroy.facet.loc114_5.3 (constants.%Destroy.facet.c83)]
// CHECK:STDOUT:     %Destroy.facet.loc114_5.2: %Destroy.type = facet_value constants.%impl.elem0.7c98f0.2, (constants.%Destroy.lookup_impl_witness.61e) [symbolic = %Destroy.facet.loc114_5.3 (constants.%Destroy.facet.c83)]
// CHECK:STDOUT:     %.loc114_5.5: %Destroy.type = converted constants.%impl.elem0.7c98f0.2, %Destroy.facet.loc114_5.2 [symbolic = %Destroy.facet.loc114_5.3 (constants.%Destroy.facet.c83)]
// CHECK:STDOUT:     %specific_impl_fn.loc114_5.2: <specific function> = specific_impl_function %impl.elem0.loc114_5.1, @Destroy.WithSelf.Op(constants.%Destroy.facet.c83) [symbolic = %specific_impl_fn.loc114_5.4 (constants.%specific_impl_fn.5a8)]
// CHECK:STDOUT:     %bound_method.loc114_5.4: <bound method> = bound_method %.loc114_5.3, %specific_impl_fn.loc114_5.2
// CHECK:STDOUT:     %Destroy.WithSelf.Op.call: init %empty_tuple.type = call %bound_method.loc114_5.4(%.loc114_5.3)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @BitOrWith.loc121(%U.loc119_4.2: type, %T.loc120_4.2: @BitOrWith.loc121.%BitOrWith_where.type (%BitOrWith_where.type.d96)) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %assoc1: @BitOrWith.loc121.%BitOrWith.assoc_type (%BitOrWith.assoc_type.10d7b7.2) = assoc_entity element1, imports.%Core.import_ref.c14 [symbolic = %assoc1 (constants.%assoc1.53369c.2)]
// CHECK:STDOUT:   %BitOrWith.lookup_impl_witness.loc123: <witness> = lookup_impl_witness %T.loc120_4.1, @BitOrWith.1, @BitOrWith.1(%U.loc119_4.1) [symbolic = %BitOrWith.lookup_impl_witness.loc123 (constants.%BitOrWith.lookup_impl_witness.1faae5.2)]
// CHECK:STDOUT:   %BitOrWith.facet: @BitOrWith.loc121.%BitOrWith.type.loc120_23.1 (%BitOrWith.type.d96408.2) = facet_value %T.as_type.loc121_6.1, (%BitOrWith.lookup_impl_witness.loc123) [symbolic = %BitOrWith.facet (constants.%BitOrWith.facet.915)]
// CHECK:STDOUT:   %BitOrWith.WithSelf.Op.type: type = fn_type @BitOrWith.WithSelf.Op, @BitOrWith.WithSelf(%U.loc119_4.1, %BitOrWith.facet) [symbolic = %BitOrWith.WithSelf.Op.type (constants.%BitOrWith.WithSelf.Op.type.be5)]
// CHECK:STDOUT:   %.loc123_5.6: type = fn_type_with_self_type %BitOrWith.WithSelf.Op.type, %BitOrWith.facet [symbolic = %.loc123_5.6 (constants.%.4cd)]
// CHECK:STDOUT:   %impl.elem1.loc123_5.2: @BitOrWith.loc121.%.loc123_5.6 (%.4cd) = impl_witness_access %BitOrWith.lookup_impl_witness.loc123, element1 [symbolic = %impl.elem1.loc123_5.2 (constants.%impl.elem1.4a2)]
// CHECK:STDOUT:   %specific_impl_fn.loc123_5.3: <specific function> = specific_impl_function %impl.elem1.loc123_5.2, @BitOrWith.WithSelf.Op(%U.loc119_4.1, %BitOrWith.facet) [symbolic = %specific_impl_fn.loc123_5.3 (constants.%specific_impl_fn.37b)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %impl.elem0.1, @Destroy [symbolic = %Destroy.lookup_impl_witness (constants.%Destroy.lookup_impl_witness.7f4)]
// CHECK:STDOUT:   %Destroy.facet.loc123_5.3: %Destroy.type = facet_value %impl.elem0.1, (%Destroy.lookup_impl_witness) [symbolic = %Destroy.facet.loc123_5.3 (constants.%Destroy.facet.786)]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.loc123_5.3) [symbolic = %Destroy.WithSelf.Op.type (constants.%Destroy.WithSelf.Op.type.e11)]
// CHECK:STDOUT:   %.loc123_5.7: type = fn_type_with_self_type %Destroy.WithSelf.Op.type, %Destroy.facet.loc123_5.3 [symbolic = %.loc123_5.7 (constants.%.82b)]
// CHECK:STDOUT:   %impl.elem0.loc123_5.2: @BitOrWith.loc121.%.loc123_5.7 (%.82b) = impl_witness_access %Destroy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc123_5.2 (constants.%impl.elem0.5b8)]
// CHECK:STDOUT:   %specific_impl_fn.loc123_5.4: <specific function> = specific_impl_function %impl.elem0.loc123_5.2, @Destroy.WithSelf.Op(%Destroy.facet.loc123_5.3) [symbolic = %specific_impl_fn.loc123_5.4 (constants.%specific_impl_fn.056)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @BitOrWith.loc121.%T.as_type.loc121_6.1 (%T.as_type.725), %y.param: @BitOrWith.loc121.%U.loc119_4.1 (%U.67d)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: @BitOrWith.loc121.%T.as_type.loc121_6.1 (%T.as_type.725) = name_ref x, %x
// CHECK:STDOUT:     %y.ref: @BitOrWith.loc121.%U.loc119_4.1 (%U.67d) = name_ref y, %y
// CHECK:STDOUT:     %BitOrWith.type.loc123: type = facet_type <@BitOrWith.1, @BitOrWith.1(constants.%U.67d)> [symbolic = %BitOrWith.type.loc120_23.1 (constants.%BitOrWith.type.d96408.2)]
// CHECK:STDOUT:     %.loc123_5.1: @BitOrWith.loc121.%BitOrWith.assoc_type (%BitOrWith.assoc_type.10d7b7.2) = specific_constant imports.%Core.import_ref.f87, @BitOrWith.WithSelf(constants.%U.67d, constants.%Self.6cdc7c.1) [symbolic = %assoc1 (constants.%assoc1.53369c.2)]
// CHECK:STDOUT:     %Op.ref: @BitOrWith.loc121.%BitOrWith.assoc_type (%BitOrWith.assoc_type.10d7b7.2) = name_ref Op, %.loc123_5.1 [symbolic = %assoc1 (constants.%assoc1.53369c.2)]
// CHECK:STDOUT:     %impl.elem1.loc123_5.1: @BitOrWith.loc121.%.loc123_5.6 (%.4cd) = impl_witness_access constants.%BitOrWith.lookup_impl_witness.1faae5.2, element1 [symbolic = %impl.elem1.loc123_5.2 (constants.%impl.elem1.4a2)]
// CHECK:STDOUT:     %bound_method.loc123_5.1: <bound method> = bound_method %x.ref, %impl.elem1.loc123_5.1
// CHECK:STDOUT:     %specific_impl_fn.loc123_5.1: <specific function> = specific_impl_function %impl.elem1.loc123_5.1, @BitOrWith.WithSelf.Op(constants.%U.67d, constants.%BitOrWith.facet.915) [symbolic = %specific_impl_fn.loc123_5.3 (constants.%specific_impl_fn.37b)]
// CHECK:STDOUT:     %bound_method.loc123_5.2: <bound method> = bound_method %x.ref, %specific_impl_fn.loc123_5.1
// CHECK:STDOUT:     %.loc123_5.2: ref @BitOrWith.loc121.%impl.elem0.1 (%impl.elem0.7ec769.2) = temporary_storage
// CHECK:STDOUT:     %BitOrWith.WithSelf.Op.call: init @BitOrWith.loc121.%impl.elem0.1 (%impl.elem0.7ec769.2) to %.loc123_5.2 = call %bound_method.loc123_5.2(%x.ref, %y.ref)
// CHECK:STDOUT:     %.loc123_5.3: ref @BitOrWith.loc121.%impl.elem0.1 (%impl.elem0.7ec769.2) = temporary %.loc123_5.2, %BitOrWith.WithSelf.Op.call
// CHECK:STDOUT:     %impl.elem0.loc123_5.1: @BitOrWith.loc121.%.loc123_5.7 (%.82b) = impl_witness_access constants.%Destroy.lookup_impl_witness.7f4, element0 [symbolic = %impl.elem0.loc123_5.2 (constants.%impl.elem0.5b8)]
// CHECK:STDOUT:     %bound_method.loc123_5.3: <bound method> = bound_method %.loc123_5.3, %impl.elem0.loc123_5.1
// CHECK:STDOUT:     %Destroy.facet.loc123_5.1: %Destroy.type = facet_value constants.%impl.elem0.7ec769.2, (constants.%Destroy.lookup_impl_witness.7f4) [symbolic = %Destroy.facet.loc123_5.3 (constants.%Destroy.facet.786)]
// CHECK:STDOUT:     %.loc123_5.4: %Destroy.type = converted constants.%impl.elem0.7ec769.2, %Destroy.facet.loc123_5.1 [symbolic = %Destroy.facet.loc123_5.3 (constants.%Destroy.facet.786)]
// CHECK:STDOUT:     %Destroy.facet.loc123_5.2: %Destroy.type = facet_value constants.%impl.elem0.7ec769.2, (constants.%Destroy.lookup_impl_witness.7f4) [symbolic = %Destroy.facet.loc123_5.3 (constants.%Destroy.facet.786)]
// CHECK:STDOUT:     %.loc123_5.5: %Destroy.type = converted constants.%impl.elem0.7ec769.2, %Destroy.facet.loc123_5.2 [symbolic = %Destroy.facet.loc123_5.3 (constants.%Destroy.facet.786)]
// CHECK:STDOUT:     %specific_impl_fn.loc123_5.2: <specific function> = specific_impl_function %impl.elem0.loc123_5.1, @Destroy.WithSelf.Op(constants.%Destroy.facet.786) [symbolic = %specific_impl_fn.loc123_5.4 (constants.%specific_impl_fn.056)]
// CHECK:STDOUT:     %bound_method.loc123_5.4: <bound method> = bound_method %.loc123_5.3, %specific_impl_fn.loc123_5.2
// CHECK:STDOUT:     %Destroy.WithSelf.Op.call: init %empty_tuple.type = call %bound_method.loc123_5.4(%.loc123_5.3)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @BitXorWith.loc130(%U.loc128_4.2: type, %T.loc129_4.2: @BitXorWith.loc130.%BitXorWith_where.type (%BitXorWith_where.type.7de)) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %assoc1: @BitXorWith.loc130.%BitXorWith.assoc_type (%BitXorWith.assoc_type.4ad612.2) = assoc_entity element1, imports.%Core.import_ref.808 [symbolic = %assoc1 (constants.%assoc1.58565f.2)]
// CHECK:STDOUT:   %BitXorWith.lookup_impl_witness.loc132: <witness> = lookup_impl_witness %T.loc129_4.1, @BitXorWith.1, @BitXorWith.1(%U.loc128_4.1) [symbolic = %BitXorWith.lookup_impl_witness.loc132 (constants.%BitXorWith.lookup_impl_witness.28dfc5.2)]
// CHECK:STDOUT:   %BitXorWith.facet: @BitXorWith.loc130.%BitXorWith.type.loc129_24.1 (%BitXorWith.type.7de562.2) = facet_value %T.as_type.loc130_6.1, (%BitXorWith.lookup_impl_witness.loc132) [symbolic = %BitXorWith.facet (constants.%BitXorWith.facet.9e4)]
// CHECK:STDOUT:   %BitXorWith.WithSelf.Op.type: type = fn_type @BitXorWith.WithSelf.Op, @BitXorWith.WithSelf(%U.loc128_4.1, %BitXorWith.facet) [symbolic = %BitXorWith.WithSelf.Op.type (constants.%BitXorWith.WithSelf.Op.type.55f)]
// CHECK:STDOUT:   %.loc132_5.6: type = fn_type_with_self_type %BitXorWith.WithSelf.Op.type, %BitXorWith.facet [symbolic = %.loc132_5.6 (constants.%.037)]
// CHECK:STDOUT:   %impl.elem1.loc132_5.2: @BitXorWith.loc130.%.loc132_5.6 (%.037) = impl_witness_access %BitXorWith.lookup_impl_witness.loc132, element1 [symbolic = %impl.elem1.loc132_5.2 (constants.%impl.elem1.01b)]
// CHECK:STDOUT:   %specific_impl_fn.loc132_5.3: <specific function> = specific_impl_function %impl.elem1.loc132_5.2, @BitXorWith.WithSelf.Op(%U.loc128_4.1, %BitXorWith.facet) [symbolic = %specific_impl_fn.loc132_5.3 (constants.%specific_impl_fn.7831)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %impl.elem0.1, @Destroy [symbolic = %Destroy.lookup_impl_witness (constants.%Destroy.lookup_impl_witness.b85)]
// CHECK:STDOUT:   %Destroy.facet.loc132_5.3: %Destroy.type = facet_value %impl.elem0.1, (%Destroy.lookup_impl_witness) [symbolic = %Destroy.facet.loc132_5.3 (constants.%Destroy.facet.a2d)]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.loc132_5.3) [symbolic = %Destroy.WithSelf.Op.type (constants.%Destroy.WithSelf.Op.type.8b0)]
// CHECK:STDOUT:   %.loc132_5.7: type = fn_type_with_self_type %Destroy.WithSelf.Op.type, %Destroy.facet.loc132_5.3 [symbolic = %.loc132_5.7 (constants.%.1b3)]
// CHECK:STDOUT:   %impl.elem0.loc132_5.2: @BitXorWith.loc130.%.loc132_5.7 (%.1b3) = impl_witness_access %Destroy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc132_5.2 (constants.%impl.elem0.7b5)]
// CHECK:STDOUT:   %specific_impl_fn.loc132_5.4: <specific function> = specific_impl_function %impl.elem0.loc132_5.2, @Destroy.WithSelf.Op(%Destroy.facet.loc132_5.3) [symbolic = %specific_impl_fn.loc132_5.4 (constants.%specific_impl_fn.875)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @BitXorWith.loc130.%T.as_type.loc130_6.1 (%T.as_type.f34), %y.param: @BitXorWith.loc130.%U.loc128_4.1 (%U.67d)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: @BitXorWith.loc130.%T.as_type.loc130_6.1 (%T.as_type.f34) = name_ref x, %x
// CHECK:STDOUT:     %y.ref: @BitXorWith.loc130.%U.loc128_4.1 (%U.67d) = name_ref y, %y
// CHECK:STDOUT:     %BitXorWith.type.loc132: type = facet_type <@BitXorWith.1, @BitXorWith.1(constants.%U.67d)> [symbolic = %BitXorWith.type.loc129_24.1 (constants.%BitXorWith.type.7de562.2)]
// CHECK:STDOUT:     %.loc132_5.1: @BitXorWith.loc130.%BitXorWith.assoc_type (%BitXorWith.assoc_type.4ad612.2) = specific_constant imports.%Core.import_ref.68e, @BitXorWith.WithSelf(constants.%U.67d, constants.%Self.5c244d.1) [symbolic = %assoc1 (constants.%assoc1.58565f.2)]
// CHECK:STDOUT:     %Op.ref: @BitXorWith.loc130.%BitXorWith.assoc_type (%BitXorWith.assoc_type.4ad612.2) = name_ref Op, %.loc132_5.1 [symbolic = %assoc1 (constants.%assoc1.58565f.2)]
// CHECK:STDOUT:     %impl.elem1.loc132_5.1: @BitXorWith.loc130.%.loc132_5.6 (%.037) = impl_witness_access constants.%BitXorWith.lookup_impl_witness.28dfc5.2, element1 [symbolic = %impl.elem1.loc132_5.2 (constants.%impl.elem1.01b)]
// CHECK:STDOUT:     %bound_method.loc132_5.1: <bound method> = bound_method %x.ref, %impl.elem1.loc132_5.1
// CHECK:STDOUT:     %specific_impl_fn.loc132_5.1: <specific function> = specific_impl_function %impl.elem1.loc132_5.1, @BitXorWith.WithSelf.Op(constants.%U.67d, constants.%BitXorWith.facet.9e4) [symbolic = %specific_impl_fn.loc132_5.3 (constants.%specific_impl_fn.7831)]
// CHECK:STDOUT:     %bound_method.loc132_5.2: <bound method> = bound_method %x.ref, %specific_impl_fn.loc132_5.1
// CHECK:STDOUT:     %.loc132_5.2: ref @BitXorWith.loc130.%impl.elem0.1 (%impl.elem0.9a9bb7.2) = temporary_storage
// CHECK:STDOUT:     %BitXorWith.WithSelf.Op.call: init @BitXorWith.loc130.%impl.elem0.1 (%impl.elem0.9a9bb7.2) to %.loc132_5.2 = call %bound_method.loc132_5.2(%x.ref, %y.ref)
// CHECK:STDOUT:     %.loc132_5.3: ref @BitXorWith.loc130.%impl.elem0.1 (%impl.elem0.9a9bb7.2) = temporary %.loc132_5.2, %BitXorWith.WithSelf.Op.call
// CHECK:STDOUT:     %impl.elem0.loc132_5.1: @BitXorWith.loc130.%.loc132_5.7 (%.1b3) = impl_witness_access constants.%Destroy.lookup_impl_witness.b85, element0 [symbolic = %impl.elem0.loc132_5.2 (constants.%impl.elem0.7b5)]
// CHECK:STDOUT:     %bound_method.loc132_5.3: <bound method> = bound_method %.loc132_5.3, %impl.elem0.loc132_5.1
// CHECK:STDOUT:     %Destroy.facet.loc132_5.1: %Destroy.type = facet_value constants.%impl.elem0.9a9bb7.2, (constants.%Destroy.lookup_impl_witness.b85) [symbolic = %Destroy.facet.loc132_5.3 (constants.%Destroy.facet.a2d)]
// CHECK:STDOUT:     %.loc132_5.4: %Destroy.type = converted constants.%impl.elem0.9a9bb7.2, %Destroy.facet.loc132_5.1 [symbolic = %Destroy.facet.loc132_5.3 (constants.%Destroy.facet.a2d)]
// CHECK:STDOUT:     %Destroy.facet.loc132_5.2: %Destroy.type = facet_value constants.%impl.elem0.9a9bb7.2, (constants.%Destroy.lookup_impl_witness.b85) [symbolic = %Destroy.facet.loc132_5.3 (constants.%Destroy.facet.a2d)]
// CHECK:STDOUT:     %.loc132_5.5: %Destroy.type = converted constants.%impl.elem0.9a9bb7.2, %Destroy.facet.loc132_5.2 [symbolic = %Destroy.facet.loc132_5.3 (constants.%Destroy.facet.a2d)]
// CHECK:STDOUT:     %specific_impl_fn.loc132_5.2: <specific function> = specific_impl_function %impl.elem0.loc132_5.1, @Destroy.WithSelf.Op(constants.%Destroy.facet.a2d) [symbolic = %specific_impl_fn.loc132_5.4 (constants.%specific_impl_fn.875)]
// CHECK:STDOUT:     %bound_method.loc132_5.4: <bound method> = bound_method %.loc132_5.3, %specific_impl_fn.loc132_5.2
// CHECK:STDOUT:     %Destroy.WithSelf.Op.call: init %empty_tuple.type = call %bound_method.loc132_5.4(%.loc132_5.3)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @LeftShiftWith.loc139(%U.loc137_4.2: type, %T.loc138_4.2: @LeftShiftWith.loc139.%LeftShiftWith_where.type (%LeftShiftWith_where.type.cd5)) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %assoc1: @LeftShiftWith.loc139.%LeftShiftWith.assoc_type (%LeftShiftWith.assoc_type.739d6a.2) = assoc_entity element1, imports.%Core.import_ref.36a [symbolic = %assoc1 (constants.%assoc1.baaa39.2)]
// CHECK:STDOUT:   %LeftShiftWith.lookup_impl_witness.loc141: <witness> = lookup_impl_witness %T.loc138_4.1, @LeftShiftWith.1, @LeftShiftWith.1(%U.loc137_4.1) [symbolic = %LeftShiftWith.lookup_impl_witness.loc141 (constants.%LeftShiftWith.lookup_impl_witness.209995.2)]
// CHECK:STDOUT:   %LeftShiftWith.facet: @LeftShiftWith.loc139.%LeftShiftWith.type.loc138_27.1 (%LeftShiftWith.type.cd59d0.2) = facet_value %T.as_type.loc139_6.1, (%LeftShiftWith.lookup_impl_witness.loc141) [symbolic = %LeftShiftWith.facet (constants.%LeftShiftWith.facet.568)]
// CHECK:STDOUT:   %LeftShiftWith.WithSelf.Op.type: type = fn_type @LeftShiftWith.WithSelf.Op, @LeftShiftWith.WithSelf(%U.loc137_4.1, %LeftShiftWith.facet) [symbolic = %LeftShiftWith.WithSelf.Op.type (constants.%LeftShiftWith.WithSelf.Op.type.4a4)]
// CHECK:STDOUT:   %.loc141_5.6: type = fn_type_with_self_type %LeftShiftWith.WithSelf.Op.type, %LeftShiftWith.facet [symbolic = %.loc141_5.6 (constants.%.420)]
// CHECK:STDOUT:   %impl.elem1.loc141_5.2: @LeftShiftWith.loc139.%.loc141_5.6 (%.420) = impl_witness_access %LeftShiftWith.lookup_impl_witness.loc141, element1 [symbolic = %impl.elem1.loc141_5.2 (constants.%impl.elem1.f33)]
// CHECK:STDOUT:   %specific_impl_fn.loc141_5.3: <specific function> = specific_impl_function %impl.elem1.loc141_5.2, @LeftShiftWith.WithSelf.Op(%U.loc137_4.1, %LeftShiftWith.facet) [symbolic = %specific_impl_fn.loc141_5.3 (constants.%specific_impl_fn.502)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %impl.elem0.1, @Destroy [symbolic = %Destroy.lookup_impl_witness (constants.%Destroy.lookup_impl_witness.acb)]
// CHECK:STDOUT:   %Destroy.facet.loc141_5.3: %Destroy.type = facet_value %impl.elem0.1, (%Destroy.lookup_impl_witness) [symbolic = %Destroy.facet.loc141_5.3 (constants.%Destroy.facet.c90)]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.loc141_5.3) [symbolic = %Destroy.WithSelf.Op.type (constants.%Destroy.WithSelf.Op.type.fc0)]
// CHECK:STDOUT:   %.loc141_5.7: type = fn_type_with_self_type %Destroy.WithSelf.Op.type, %Destroy.facet.loc141_5.3 [symbolic = %.loc141_5.7 (constants.%.06e)]
// CHECK:STDOUT:   %impl.elem0.loc141_5.2: @LeftShiftWith.loc139.%.loc141_5.7 (%.06e) = impl_witness_access %Destroy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc141_5.2 (constants.%impl.elem0.94b)]
// CHECK:STDOUT:   %specific_impl_fn.loc141_5.4: <specific function> = specific_impl_function %impl.elem0.loc141_5.2, @Destroy.WithSelf.Op(%Destroy.facet.loc141_5.3) [symbolic = %specific_impl_fn.loc141_5.4 (constants.%specific_impl_fn.418)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @LeftShiftWith.loc139.%T.as_type.loc139_6.1 (%T.as_type.9e2), %y.param: @LeftShiftWith.loc139.%U.loc137_4.1 (%U.67d)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: @LeftShiftWith.loc139.%T.as_type.loc139_6.1 (%T.as_type.9e2) = name_ref x, %x
// CHECK:STDOUT:     %y.ref: @LeftShiftWith.loc139.%U.loc137_4.1 (%U.67d) = name_ref y, %y
// CHECK:STDOUT:     %LeftShiftWith.type.loc141: type = facet_type <@LeftShiftWith.1, @LeftShiftWith.1(constants.%U.67d)> [symbolic = %LeftShiftWith.type.loc138_27.1 (constants.%LeftShiftWith.type.cd59d0.2)]
// CHECK:STDOUT:     %.loc141_5.1: @LeftShiftWith.loc139.%LeftShiftWith.assoc_type (%LeftShiftWith.assoc_type.739d6a.2) = specific_constant imports.%Core.import_ref.3b3, @LeftShiftWith.WithSelf(constants.%U.67d, constants.%Self.1e32ce.1) [symbolic = %assoc1 (constants.%assoc1.baaa39.2)]
// CHECK:STDOUT:     %Op.ref: @LeftShiftWith.loc139.%LeftShiftWith.assoc_type (%LeftShiftWith.assoc_type.739d6a.2) = name_ref Op, %.loc141_5.1 [symbolic = %assoc1 (constants.%assoc1.baaa39.2)]
// CHECK:STDOUT:     %impl.elem1.loc141_5.1: @LeftShiftWith.loc139.%.loc141_5.6 (%.420) = impl_witness_access constants.%LeftShiftWith.lookup_impl_witness.209995.2, element1 [symbolic = %impl.elem1.loc141_5.2 (constants.%impl.elem1.f33)]
// CHECK:STDOUT:     %bound_method.loc141_5.1: <bound method> = bound_method %x.ref, %impl.elem1.loc141_5.1
// CHECK:STDOUT:     %specific_impl_fn.loc141_5.1: <specific function> = specific_impl_function %impl.elem1.loc141_5.1, @LeftShiftWith.WithSelf.Op(constants.%U.67d, constants.%LeftShiftWith.facet.568) [symbolic = %specific_impl_fn.loc141_5.3 (constants.%specific_impl_fn.502)]
// CHECK:STDOUT:     %bound_method.loc141_5.2: <bound method> = bound_method %x.ref, %specific_impl_fn.loc141_5.1
// CHECK:STDOUT:     %.loc141_5.2: ref @LeftShiftWith.loc139.%impl.elem0.1 (%impl.elem0.fd5c74.2) = temporary_storage
// CHECK:STDOUT:     %LeftShiftWith.WithSelf.Op.call: init @LeftShiftWith.loc139.%impl.elem0.1 (%impl.elem0.fd5c74.2) to %.loc141_5.2 = call %bound_method.loc141_5.2(%x.ref, %y.ref)
// CHECK:STDOUT:     %.loc141_5.3: ref @LeftShiftWith.loc139.%impl.elem0.1 (%impl.elem0.fd5c74.2) = temporary %.loc141_5.2, %LeftShiftWith.WithSelf.Op.call
// CHECK:STDOUT:     %impl.elem0.loc141_5.1: @LeftShiftWith.loc139.%.loc141_5.7 (%.06e) = impl_witness_access constants.%Destroy.lookup_impl_witness.acb, element0 [symbolic = %impl.elem0.loc141_5.2 (constants.%impl.elem0.94b)]
// CHECK:STDOUT:     %bound_method.loc141_5.3: <bound method> = bound_method %.loc141_5.3, %impl.elem0.loc141_5.1
// CHECK:STDOUT:     %Destroy.facet.loc141_5.1: %Destroy.type = facet_value constants.%impl.elem0.fd5c74.2, (constants.%Destroy.lookup_impl_witness.acb) [symbolic = %Destroy.facet.loc141_5.3 (constants.%Destroy.facet.c90)]
// CHECK:STDOUT:     %.loc141_5.4: %Destroy.type = converted constants.%impl.elem0.fd5c74.2, %Destroy.facet.loc141_5.1 [symbolic = %Destroy.facet.loc141_5.3 (constants.%Destroy.facet.c90)]
// CHECK:STDOUT:     %Destroy.facet.loc141_5.2: %Destroy.type = facet_value constants.%impl.elem0.fd5c74.2, (constants.%Destroy.lookup_impl_witness.acb) [symbolic = %Destroy.facet.loc141_5.3 (constants.%Destroy.facet.c90)]
// CHECK:STDOUT:     %.loc141_5.5: %Destroy.type = converted constants.%impl.elem0.fd5c74.2, %Destroy.facet.loc141_5.2 [symbolic = %Destroy.facet.loc141_5.3 (constants.%Destroy.facet.c90)]
// CHECK:STDOUT:     %specific_impl_fn.loc141_5.2: <specific function> = specific_impl_function %impl.elem0.loc141_5.1, @Destroy.WithSelf.Op(constants.%Destroy.facet.c90) [symbolic = %specific_impl_fn.loc141_5.4 (constants.%specific_impl_fn.418)]
// CHECK:STDOUT:     %bound_method.loc141_5.4: <bound method> = bound_method %.loc141_5.3, %specific_impl_fn.loc141_5.2
// CHECK:STDOUT:     %Destroy.WithSelf.Op.call: init %empty_tuple.type = call %bound_method.loc141_5.4(%.loc141_5.3)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @RightShiftWith.loc148(%U.loc146_4.2: type, %T.loc147_4.2: @RightShiftWith.loc148.%RightShiftWith_where.type (%RightShiftWith_where.type.60d)) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %assoc1: @RightShiftWith.loc148.%RightShiftWith.assoc_type (%RightShiftWith.assoc_type.4d2607.2) = assoc_entity element1, imports.%Core.import_ref.e6b [symbolic = %assoc1 (constants.%assoc1.53f1a1.2)]
// CHECK:STDOUT:   %RightShiftWith.lookup_impl_witness.loc150: <witness> = lookup_impl_witness %T.loc147_4.1, @RightShiftWith.1, @RightShiftWith.1(%U.loc146_4.1) [symbolic = %RightShiftWith.lookup_impl_witness.loc150 (constants.%RightShiftWith.lookup_impl_witness.e14634.2)]
// CHECK:STDOUT:   %RightShiftWith.facet: @RightShiftWith.loc148.%RightShiftWith.type.loc147_28.1 (%RightShiftWith.type.60db85.2) = facet_value %T.as_type.loc148_6.1, (%RightShiftWith.lookup_impl_witness.loc150) [symbolic = %RightShiftWith.facet (constants.%RightShiftWith.facet.78e)]
// CHECK:STDOUT:   %RightShiftWith.WithSelf.Op.type: type = fn_type @RightShiftWith.WithSelf.Op, @RightShiftWith.WithSelf(%U.loc146_4.1, %RightShiftWith.facet) [symbolic = %RightShiftWith.WithSelf.Op.type (constants.%RightShiftWith.WithSelf.Op.type.38b)]
// CHECK:STDOUT:   %.loc150_5.6: type = fn_type_with_self_type %RightShiftWith.WithSelf.Op.type, %RightShiftWith.facet [symbolic = %.loc150_5.6 (constants.%.209)]
// CHECK:STDOUT:   %impl.elem1.loc150_5.2: @RightShiftWith.loc148.%.loc150_5.6 (%.209) = impl_witness_access %RightShiftWith.lookup_impl_witness.loc150, element1 [symbolic = %impl.elem1.loc150_5.2 (constants.%impl.elem1.cc9)]
// CHECK:STDOUT:   %specific_impl_fn.loc150_5.3: <specific function> = specific_impl_function %impl.elem1.loc150_5.2, @RightShiftWith.WithSelf.Op(%U.loc146_4.1, %RightShiftWith.facet) [symbolic = %specific_impl_fn.loc150_5.3 (constants.%specific_impl_fn.c69)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %impl.elem0.1, @Destroy [symbolic = %Destroy.lookup_impl_witness (constants.%Destroy.lookup_impl_witness.708)]
// CHECK:STDOUT:   %Destroy.facet.loc150_5.3: %Destroy.type = facet_value %impl.elem0.1, (%Destroy.lookup_impl_witness) [symbolic = %Destroy.facet.loc150_5.3 (constants.%Destroy.facet.0ac)]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.loc150_5.3) [symbolic = %Destroy.WithSelf.Op.type (constants.%Destroy.WithSelf.Op.type.bb8)]
// CHECK:STDOUT:   %.loc150_5.7: type = fn_type_with_self_type %Destroy.WithSelf.Op.type, %Destroy.facet.loc150_5.3 [symbolic = %.loc150_5.7 (constants.%.1f0)]
// CHECK:STDOUT:   %impl.elem0.loc150_5.2: @RightShiftWith.loc148.%.loc150_5.7 (%.1f0) = impl_witness_access %Destroy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc150_5.2 (constants.%impl.elem0.828)]
// CHECK:STDOUT:   %specific_impl_fn.loc150_5.4: <specific function> = specific_impl_function %impl.elem0.loc150_5.2, @Destroy.WithSelf.Op(%Destroy.facet.loc150_5.3) [symbolic = %specific_impl_fn.loc150_5.4 (constants.%specific_impl_fn.7834)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @RightShiftWith.loc148.%T.as_type.loc148_6.1 (%T.as_type.e4f), %y.param: @RightShiftWith.loc148.%U.loc146_4.1 (%U.67d)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: @RightShiftWith.loc148.%T.as_type.loc148_6.1 (%T.as_type.e4f) = name_ref x, %x
// CHECK:STDOUT:     %y.ref: @RightShiftWith.loc148.%U.loc146_4.1 (%U.67d) = name_ref y, %y
// CHECK:STDOUT:     %RightShiftWith.type.loc150: type = facet_type <@RightShiftWith.1, @RightShiftWith.1(constants.%U.67d)> [symbolic = %RightShiftWith.type.loc147_28.1 (constants.%RightShiftWith.type.60db85.2)]
// CHECK:STDOUT:     %.loc150_5.1: @RightShiftWith.loc148.%RightShiftWith.assoc_type (%RightShiftWith.assoc_type.4d2607.2) = specific_constant imports.%Core.import_ref.573, @RightShiftWith.WithSelf(constants.%U.67d, constants.%Self.665757.1) [symbolic = %assoc1 (constants.%assoc1.53f1a1.2)]
// CHECK:STDOUT:     %Op.ref: @RightShiftWith.loc148.%RightShiftWith.assoc_type (%RightShiftWith.assoc_type.4d2607.2) = name_ref Op, %.loc150_5.1 [symbolic = %assoc1 (constants.%assoc1.53f1a1.2)]
// CHECK:STDOUT:     %impl.elem1.loc150_5.1: @RightShiftWith.loc148.%.loc150_5.6 (%.209) = impl_witness_access constants.%RightShiftWith.lookup_impl_witness.e14634.2, element1 [symbolic = %impl.elem1.loc150_5.2 (constants.%impl.elem1.cc9)]
// CHECK:STDOUT:     %bound_method.loc150_5.1: <bound method> = bound_method %x.ref, %impl.elem1.loc150_5.1
// CHECK:STDOUT:     %specific_impl_fn.loc150_5.1: <specific function> = specific_impl_function %impl.elem1.loc150_5.1, @RightShiftWith.WithSelf.Op(constants.%U.67d, constants.%RightShiftWith.facet.78e) [symbolic = %specific_impl_fn.loc150_5.3 (constants.%specific_impl_fn.c69)]
// CHECK:STDOUT:     %bound_method.loc150_5.2: <bound method> = bound_method %x.ref, %specific_impl_fn.loc150_5.1
// CHECK:STDOUT:     %.loc150_5.2: ref @RightShiftWith.loc148.%impl.elem0.1 (%impl.elem0.bc591e.2) = temporary_storage
// CHECK:STDOUT:     %RightShiftWith.WithSelf.Op.call: init @RightShiftWith.loc148.%impl.elem0.1 (%impl.elem0.bc591e.2) to %.loc150_5.2 = call %bound_method.loc150_5.2(%x.ref, %y.ref)
// CHECK:STDOUT:     %.loc150_5.3: ref @RightShiftWith.loc148.%impl.elem0.1 (%impl.elem0.bc591e.2) = temporary %.loc150_5.2, %RightShiftWith.WithSelf.Op.call
// CHECK:STDOUT:     %impl.elem0.loc150_5.1: @RightShiftWith.loc148.%.loc150_5.7 (%.1f0) = impl_witness_access constants.%Destroy.lookup_impl_witness.708, element0 [symbolic = %impl.elem0.loc150_5.2 (constants.%impl.elem0.828)]
// CHECK:STDOUT:     %bound_method.loc150_5.3: <bound method> = bound_method %.loc150_5.3, %impl.elem0.loc150_5.1
// CHECK:STDOUT:     %Destroy.facet.loc150_5.1: %Destroy.type = facet_value constants.%impl.elem0.bc591e.2, (constants.%Destroy.lookup_impl_witness.708) [symbolic = %Destroy.facet.loc150_5.3 (constants.%Destroy.facet.0ac)]
// CHECK:STDOUT:     %.loc150_5.4: %Destroy.type = converted constants.%impl.elem0.bc591e.2, %Destroy.facet.loc150_5.1 [symbolic = %Destroy.facet.loc150_5.3 (constants.%Destroy.facet.0ac)]
// CHECK:STDOUT:     %Destroy.facet.loc150_5.2: %Destroy.type = facet_value constants.%impl.elem0.bc591e.2, (constants.%Destroy.lookup_impl_witness.708) [symbolic = %Destroy.facet.loc150_5.3 (constants.%Destroy.facet.0ac)]
// CHECK:STDOUT:     %.loc150_5.5: %Destroy.type = converted constants.%impl.elem0.bc591e.2, %Destroy.facet.loc150_5.2 [symbolic = %Destroy.facet.loc150_5.3 (constants.%Destroy.facet.0ac)]
// CHECK:STDOUT:     %specific_impl_fn.loc150_5.2: <specific function> = specific_impl_function %impl.elem0.loc150_5.1, @Destroy.WithSelf.Op(constants.%Destroy.facet.0ac) [symbolic = %specific_impl_fn.loc150_5.4 (constants.%specific_impl_fn.7834)]
// CHECK:STDOUT:     %bound_method.loc150_5.4: <bound method> = bound_method %.loc150_5.3, %specific_impl_fn.loc150_5.2
// CHECK:STDOUT:     %Destroy.WithSelf.Op.call: init %empty_tuple.type = call %bound_method.loc150_5.4(%.loc150_5.3)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @BitAndAssignWith.loc205(%U.loc205_22.2: type, %T.loc205_32.2: @BitAndAssignWith.loc205.%BitAndAssignWith.type.loc205_58.1 (%BitAndAssignWith.type.24c915.2)) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %require_complete.loc207: <witness> = require_complete_type %BitAndAssignWith.type.loc205_58.1 [symbolic = %require_complete.loc207 (constants.%require_complete.9c4)]
// CHECK:STDOUT:   %BitAndAssignWith.assoc_type: type = assoc_entity_type @BitAndAssignWith.1, @BitAndAssignWith.1(%U.loc205_22.1) [symbolic = %BitAndAssignWith.assoc_type (constants.%BitAndAssignWith.assoc_type.5d10e3.2)]
// CHECK:STDOUT:   %assoc0: @BitAndAssignWith.loc205.%BitAndAssignWith.assoc_type (%BitAndAssignWith.assoc_type.5d10e3.2) = assoc_entity element0, imports.%Core.import_ref.618 [symbolic = %assoc0 (constants.%assoc0.43e266.2)]
// CHECK:STDOUT:   %BitAndAssignWith.WithSelf.Op.type: type = fn_type @BitAndAssignWith.WithSelf.Op, @BitAndAssignWith.WithSelf(%U.loc205_22.1, %T.loc205_32.1) [symbolic = %BitAndAssignWith.WithSelf.Op.type (constants.%BitAndAssignWith.WithSelf.Op.type.e3224f.3)]
// CHECK:STDOUT:   %.loc207_5.2: type = fn_type_with_self_type %BitAndAssignWith.WithSelf.Op.type, %T.loc205_32.1 [symbolic = %.loc207_5.2 (constants.%.bea)]
// CHECK:STDOUT:   %BitAndAssignWith.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc205_32.1, @BitAndAssignWith.1, @BitAndAssignWith.1(%U.loc205_22.1) [symbolic = %BitAndAssignWith.lookup_impl_witness (constants.%BitAndAssignWith.lookup_impl_witness)]
// CHECK:STDOUT:   %impl.elem0.loc207_5.2: @BitAndAssignWith.loc205.%.loc207_5.2 (%.bea) = impl_witness_access %BitAndAssignWith.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc207_5.2 (constants.%impl.elem0.a31)]
// CHECK:STDOUT:   %specific_impl_fn.loc207_5.2: <specific function> = specific_impl_function %impl.elem0.loc207_5.2, @BitAndAssignWith.WithSelf.Op(%U.loc205_22.1, %T.loc205_32.1) [symbolic = %specific_impl_fn.loc207_5.2 (constants.%specific_impl_fn.f91)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: ref @BitAndAssignWith.loc205.%T.as_type.loc205_68.1 (%T.as_type.dca), %y.param: @BitAndAssignWith.loc205.%U.loc205_22.1 (%U.67d)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: ref @BitAndAssignWith.loc205.%T.as_type.loc205_68.1 (%T.as_type.dca) = name_ref x, %x
// CHECK:STDOUT:     %y.ref: @BitAndAssignWith.loc205.%U.loc205_22.1 (%U.67d) = name_ref y, %y
// CHECK:STDOUT:     %BitAndAssignWith.type.loc207: type = facet_type <@BitAndAssignWith.1, @BitAndAssignWith.1(constants.%U.67d)> [symbolic = %BitAndAssignWith.type.loc205_58.1 (constants.%BitAndAssignWith.type.24c915.2)]
// CHECK:STDOUT:     %.loc207_5.1: @BitAndAssignWith.loc205.%BitAndAssignWith.assoc_type (%BitAndAssignWith.assoc_type.5d10e3.2) = specific_constant imports.%Core.import_ref.feb0, @BitAndAssignWith.WithSelf(constants.%U.67d, constants.%Self.80b651.1) [symbolic = %assoc0 (constants.%assoc0.43e266.2)]
// CHECK:STDOUT:     %Op.ref: @BitAndAssignWith.loc205.%BitAndAssignWith.assoc_type (%BitAndAssignWith.assoc_type.5d10e3.2) = name_ref Op, %.loc207_5.1 [symbolic = %assoc0 (constants.%assoc0.43e266.2)]
// CHECK:STDOUT:     %impl.elem0.loc207_5.1: @BitAndAssignWith.loc205.%.loc207_5.2 (%.bea) = impl_witness_access constants.%BitAndAssignWith.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc207_5.2 (constants.%impl.elem0.a31)]
// CHECK:STDOUT:     %bound_method.loc207_5.1: <bound method> = bound_method %x.ref, %impl.elem0.loc207_5.1
// CHECK:STDOUT:     %specific_impl_fn.loc207_5.1: <specific function> = specific_impl_function %impl.elem0.loc207_5.1, @BitAndAssignWith.WithSelf.Op(constants.%U.67d, constants.%T.80b) [symbolic = %specific_impl_fn.loc207_5.2 (constants.%specific_impl_fn.f91)]
// CHECK:STDOUT:     %bound_method.loc207_5.2: <bound method> = bound_method %x.ref, %specific_impl_fn.loc207_5.1
// CHECK:STDOUT:     %BitAndAssignWith.WithSelf.Op.call: init %empty_tuple.type = call %bound_method.loc207_5.2(%x.ref, %y.ref)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @BitOrAssignWith.loc211(%U.loc211_21.2: type, %T.loc211_31.2: @BitOrAssignWith.loc211.%BitOrAssignWith.type.loc211_56.1 (%BitOrAssignWith.type.3ef7d1.2)) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %require_complete.loc213: <witness> = require_complete_type %BitOrAssignWith.type.loc211_56.1 [symbolic = %require_complete.loc213 (constants.%require_complete.abb)]
// CHECK:STDOUT:   %BitOrAssignWith.assoc_type: type = assoc_entity_type @BitOrAssignWith.1, @BitOrAssignWith.1(%U.loc211_21.1) [symbolic = %BitOrAssignWith.assoc_type (constants.%BitOrAssignWith.assoc_type.e05d8b.2)]
// CHECK:STDOUT:   %assoc0: @BitOrAssignWith.loc211.%BitOrAssignWith.assoc_type (%BitOrAssignWith.assoc_type.e05d8b.2) = assoc_entity element0, imports.%Core.import_ref.b03 [symbolic = %assoc0 (constants.%assoc0.b0b078.2)]
// CHECK:STDOUT:   %BitOrAssignWith.WithSelf.Op.type: type = fn_type @BitOrAssignWith.WithSelf.Op, @BitOrAssignWith.WithSelf(%U.loc211_21.1, %T.loc211_31.1) [symbolic = %BitOrAssignWith.WithSelf.Op.type (constants.%BitOrAssignWith.WithSelf.Op.type.130458.3)]
// CHECK:STDOUT:   %.loc213_5.2: type = fn_type_with_self_type %BitOrAssignWith.WithSelf.Op.type, %T.loc211_31.1 [symbolic = %.loc213_5.2 (constants.%.291)]
// CHECK:STDOUT:   %BitOrAssignWith.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc211_31.1, @BitOrAssignWith.1, @BitOrAssignWith.1(%U.loc211_21.1) [symbolic = %BitOrAssignWith.lookup_impl_witness (constants.%BitOrAssignWith.lookup_impl_witness)]
// CHECK:STDOUT:   %impl.elem0.loc213_5.2: @BitOrAssignWith.loc211.%.loc213_5.2 (%.291) = impl_witness_access %BitOrAssignWith.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc213_5.2 (constants.%impl.elem0.fd9)]
// CHECK:STDOUT:   %specific_impl_fn.loc213_5.2: <specific function> = specific_impl_function %impl.elem0.loc213_5.2, @BitOrAssignWith.WithSelf.Op(%U.loc211_21.1, %T.loc211_31.1) [symbolic = %specific_impl_fn.loc213_5.2 (constants.%specific_impl_fn.be4)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: ref @BitOrAssignWith.loc211.%T.as_type.loc211_66.1 (%T.as_type.72f), %y.param: @BitOrAssignWith.loc211.%U.loc211_21.1 (%U.67d)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: ref @BitOrAssignWith.loc211.%T.as_type.loc211_66.1 (%T.as_type.72f) = name_ref x, %x
// CHECK:STDOUT:     %y.ref: @BitOrAssignWith.loc211.%U.loc211_21.1 (%U.67d) = name_ref y, %y
// CHECK:STDOUT:     %BitOrAssignWith.type.loc213: type = facet_type <@BitOrAssignWith.1, @BitOrAssignWith.1(constants.%U.67d)> [symbolic = %BitOrAssignWith.type.loc211_56.1 (constants.%BitOrAssignWith.type.3ef7d1.2)]
// CHECK:STDOUT:     %.loc213_5.1: @BitOrAssignWith.loc211.%BitOrAssignWith.assoc_type (%BitOrAssignWith.assoc_type.e05d8b.2) = specific_constant imports.%Core.import_ref.8034, @BitOrAssignWith.WithSelf(constants.%U.67d, constants.%Self.cc8574.1) [symbolic = %assoc0 (constants.%assoc0.b0b078.2)]
// CHECK:STDOUT:     %Op.ref: @BitOrAssignWith.loc211.%BitOrAssignWith.assoc_type (%BitOrAssignWith.assoc_type.e05d8b.2) = name_ref Op, %.loc213_5.1 [symbolic = %assoc0 (constants.%assoc0.b0b078.2)]
// CHECK:STDOUT:     %impl.elem0.loc213_5.1: @BitOrAssignWith.loc211.%.loc213_5.2 (%.291) = impl_witness_access constants.%BitOrAssignWith.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc213_5.2 (constants.%impl.elem0.fd9)]
// CHECK:STDOUT:     %bound_method.loc213_5.1: <bound method> = bound_method %x.ref, %impl.elem0.loc213_5.1
// CHECK:STDOUT:     %specific_impl_fn.loc213_5.1: <specific function> = specific_impl_function %impl.elem0.loc213_5.1, @BitOrAssignWith.WithSelf.Op(constants.%U.67d, constants.%T.cc8) [symbolic = %specific_impl_fn.loc213_5.2 (constants.%specific_impl_fn.be4)]
// CHECK:STDOUT:     %bound_method.loc213_5.2: <bound method> = bound_method %x.ref, %specific_impl_fn.loc213_5.1
// CHECK:STDOUT:     %BitOrAssignWith.WithSelf.Op.call: init %empty_tuple.type = call %bound_method.loc213_5.2(%x.ref, %y.ref)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @BitXorAssignWith.loc217(%U.loc217_22.2: type, %T.loc217_32.2: @BitXorAssignWith.loc217.%BitXorAssignWith.type.loc217_58.1 (%BitXorAssignWith.type.70c283.2)) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %require_complete.loc219: <witness> = require_complete_type %BitXorAssignWith.type.loc217_58.1 [symbolic = %require_complete.loc219 (constants.%require_complete.023)]
// CHECK:STDOUT:   %BitXorAssignWith.assoc_type: type = assoc_entity_type @BitXorAssignWith.1, @BitXorAssignWith.1(%U.loc217_22.1) [symbolic = %BitXorAssignWith.assoc_type (constants.%BitXorAssignWith.assoc_type.2be5ac.2)]
// CHECK:STDOUT:   %assoc0: @BitXorAssignWith.loc217.%BitXorAssignWith.assoc_type (%BitXorAssignWith.assoc_type.2be5ac.2) = assoc_entity element0, imports.%Core.import_ref.194 [symbolic = %assoc0 (constants.%assoc0.5768f4.2)]
// CHECK:STDOUT:   %BitXorAssignWith.WithSelf.Op.type: type = fn_type @BitXorAssignWith.WithSelf.Op, @BitXorAssignWith.WithSelf(%U.loc217_22.1, %T.loc217_32.1) [symbolic = %BitXorAssignWith.WithSelf.Op.type (constants.%BitXorAssignWith.WithSelf.Op.type.8f9d71.3)]
// CHECK:STDOUT:   %.loc219_5.2: type = fn_type_with_self_type %BitXorAssignWith.WithSelf.Op.type, %T.loc217_32.1 [symbolic = %.loc219_5.2 (constants.%.672)]
// CHECK:STDOUT:   %BitXorAssignWith.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc217_32.1, @BitXorAssignWith.1, @BitXorAssignWith.1(%U.loc217_22.1) [symbolic = %BitXorAssignWith.lookup_impl_witness (constants.%BitXorAssignWith.lookup_impl_witness)]
// CHECK:STDOUT:   %impl.elem0.loc219_5.2: @BitXorAssignWith.loc217.%.loc219_5.2 (%.672) = impl_witness_access %BitXorAssignWith.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc219_5.2 (constants.%impl.elem0.692)]
// CHECK:STDOUT:   %specific_impl_fn.loc219_5.2: <specific function> = specific_impl_function %impl.elem0.loc219_5.2, @BitXorAssignWith.WithSelf.Op(%U.loc217_22.1, %T.loc217_32.1) [symbolic = %specific_impl_fn.loc219_5.2 (constants.%specific_impl_fn.5c0)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: ref @BitXorAssignWith.loc217.%T.as_type.loc217_68.1 (%T.as_type.a84), %y.param: @BitXorAssignWith.loc217.%U.loc217_22.1 (%U.67d)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: ref @BitXorAssignWith.loc217.%T.as_type.loc217_68.1 (%T.as_type.a84) = name_ref x, %x
// CHECK:STDOUT:     %y.ref: @BitXorAssignWith.loc217.%U.loc217_22.1 (%U.67d) = name_ref y, %y
// CHECK:STDOUT:     %BitXorAssignWith.type.loc219: type = facet_type <@BitXorAssignWith.1, @BitXorAssignWith.1(constants.%U.67d)> [symbolic = %BitXorAssignWith.type.loc217_58.1 (constants.%BitXorAssignWith.type.70c283.2)]
// CHECK:STDOUT:     %.loc219_5.1: @BitXorAssignWith.loc217.%BitXorAssignWith.assoc_type (%BitXorAssignWith.assoc_type.2be5ac.2) = specific_constant imports.%Core.import_ref.6f9, @BitXorAssignWith.WithSelf(constants.%U.67d, constants.%Self.c383a6.1) [symbolic = %assoc0 (constants.%assoc0.5768f4.2)]
// CHECK:STDOUT:     %Op.ref: @BitXorAssignWith.loc217.%BitXorAssignWith.assoc_type (%BitXorAssignWith.assoc_type.2be5ac.2) = name_ref Op, %.loc219_5.1 [symbolic = %assoc0 (constants.%assoc0.5768f4.2)]
// CHECK:STDOUT:     %impl.elem0.loc219_5.1: @BitXorAssignWith.loc217.%.loc219_5.2 (%.672) = impl_witness_access constants.%BitXorAssignWith.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc219_5.2 (constants.%impl.elem0.692)]
// CHECK:STDOUT:     %bound_method.loc219_5.1: <bound method> = bound_method %x.ref, %impl.elem0.loc219_5.1
// CHECK:STDOUT:     %specific_impl_fn.loc219_5.1: <specific function> = specific_impl_function %impl.elem0.loc219_5.1, @BitXorAssignWith.WithSelf.Op(constants.%U.67d, constants.%T.c38) [symbolic = %specific_impl_fn.loc219_5.2 (constants.%specific_impl_fn.5c0)]
// CHECK:STDOUT:     %bound_method.loc219_5.2: <bound method> = bound_method %x.ref, %specific_impl_fn.loc219_5.1
// CHECK:STDOUT:     %BitXorAssignWith.WithSelf.Op.call: init %empty_tuple.type = call %bound_method.loc219_5.2(%x.ref, %y.ref)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @LeftShiftAssignWith.loc223(%U.loc223_25.2: type, %T.loc223_35.2: @LeftShiftAssignWith.loc223.%LeftShiftAssignWith.type.loc223_64.1 (%LeftShiftAssignWith.type.858756.2)) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %require_complete.loc225: <witness> = require_complete_type %LeftShiftAssignWith.type.loc223_64.1 [symbolic = %require_complete.loc225 (constants.%require_complete.5ca)]
// CHECK:STDOUT:   %LeftShiftAssignWith.assoc_type: type = assoc_entity_type @LeftShiftAssignWith.1, @LeftShiftAssignWith.1(%U.loc223_25.1) [symbolic = %LeftShiftAssignWith.assoc_type (constants.%LeftShiftAssignWith.assoc_type.239476.2)]
// CHECK:STDOUT:   %assoc0: @LeftShiftAssignWith.loc223.%LeftShiftAssignWith.assoc_type (%LeftShiftAssignWith.assoc_type.239476.2) = assoc_entity element0, imports.%Core.import_ref.2b9 [symbolic = %assoc0 (constants.%assoc0.d11495.2)]
// CHECK:STDOUT:   %LeftShiftAssignWith.WithSelf.Op.type: type = fn_type @LeftShiftAssignWith.WithSelf.Op, @LeftShiftAssignWith.WithSelf(%U.loc223_25.1, %T.loc223_35.1) [symbolic = %LeftShiftAssignWith.WithSelf.Op.type (constants.%LeftShiftAssignWith.WithSelf.Op.type.ce10f9.3)]
// CHECK:STDOUT:   %.loc225_5.2: type = fn_type_with_self_type %LeftShiftAssignWith.WithSelf.Op.type, %T.loc223_35.1 [symbolic = %.loc225_5.2 (constants.%.591)]
// CHECK:STDOUT:   %LeftShiftAssignWith.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc223_35.1, @LeftShiftAssignWith.1, @LeftShiftAssignWith.1(%U.loc223_25.1) [symbolic = %LeftShiftAssignWith.lookup_impl_witness (constants.%LeftShiftAssignWith.lookup_impl_witness)]
// CHECK:STDOUT:   %impl.elem0.loc225_5.2: @LeftShiftAssignWith.loc223.%.loc225_5.2 (%.591) = impl_witness_access %LeftShiftAssignWith.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc225_5.2 (constants.%impl.elem0.a9d)]
// CHECK:STDOUT:   %specific_impl_fn.loc225_5.2: <specific function> = specific_impl_function %impl.elem0.loc225_5.2, @LeftShiftAssignWith.WithSelf.Op(%U.loc223_25.1, %T.loc223_35.1) [symbolic = %specific_impl_fn.loc225_5.2 (constants.%specific_impl_fn.459)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: ref @LeftShiftAssignWith.loc223.%T.as_type.loc223_74.1 (%T.as_type.09c), %y.param: @LeftShiftAssignWith.loc223.%U.loc223_25.1 (%U.67d)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: ref @LeftShiftAssignWith.loc223.%T.as_type.loc223_74.1 (%T.as_type.09c) = name_ref x, %x
// CHECK:STDOUT:     %y.ref: @LeftShiftAssignWith.loc223.%U.loc223_25.1 (%U.67d) = name_ref y, %y
// CHECK:STDOUT:     %LeftShiftAssignWith.type.loc225: type = facet_type <@LeftShiftAssignWith.1, @LeftShiftAssignWith.1(constants.%U.67d)> [symbolic = %LeftShiftAssignWith.type.loc223_64.1 (constants.%LeftShiftAssignWith.type.858756.2)]
// CHECK:STDOUT:     %.loc225_5.1: @LeftShiftAssignWith.loc223.%LeftShiftAssignWith.assoc_type (%LeftShiftAssignWith.assoc_type.239476.2) = specific_constant imports.%Core.import_ref.6ac, @LeftShiftAssignWith.WithSelf(constants.%U.67d, constants.%Self.de3088.1) [symbolic = %assoc0 (constants.%assoc0.d11495.2)]
// CHECK:STDOUT:     %Op.ref: @LeftShiftAssignWith.loc223.%LeftShiftAssignWith.assoc_type (%LeftShiftAssignWith.assoc_type.239476.2) = name_ref Op, %.loc225_5.1 [symbolic = %assoc0 (constants.%assoc0.d11495.2)]
// CHECK:STDOUT:     %impl.elem0.loc225_5.1: @LeftShiftAssignWith.loc223.%.loc225_5.2 (%.591) = impl_witness_access constants.%LeftShiftAssignWith.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc225_5.2 (constants.%impl.elem0.a9d)]
// CHECK:STDOUT:     %bound_method.loc225_5.1: <bound method> = bound_method %x.ref, %impl.elem0.loc225_5.1
// CHECK:STDOUT:     %specific_impl_fn.loc225_5.1: <specific function> = specific_impl_function %impl.elem0.loc225_5.1, @LeftShiftAssignWith.WithSelf.Op(constants.%U.67d, constants.%T.de3) [symbolic = %specific_impl_fn.loc225_5.2 (constants.%specific_impl_fn.459)]
// CHECK:STDOUT:     %bound_method.loc225_5.2: <bound method> = bound_method %x.ref, %specific_impl_fn.loc225_5.1
// CHECK:STDOUT:     %LeftShiftAssignWith.WithSelf.Op.call: init %empty_tuple.type = call %bound_method.loc225_5.2(%x.ref, %y.ref)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @RightShiftAssignWith.loc229(%U.loc229_26.2: type, %T.loc229_36.2: @RightShiftAssignWith.loc229.%RightShiftAssignWith.type.loc229_66.1 (%RightShiftAssignWith.type.abebad.2)) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %require_complete.loc231: <witness> = require_complete_type %RightShiftAssignWith.type.loc229_66.1 [symbolic = %require_complete.loc231 (constants.%require_complete.e9c)]
// CHECK:STDOUT:   %RightShiftAssignWith.assoc_type: type = assoc_entity_type @RightShiftAssignWith.1, @RightShiftAssignWith.1(%U.loc229_26.1) [symbolic = %RightShiftAssignWith.assoc_type (constants.%RightShiftAssignWith.assoc_type.95eeb2.2)]
// CHECK:STDOUT:   %assoc0: @RightShiftAssignWith.loc229.%RightShiftAssignWith.assoc_type (%RightShiftAssignWith.assoc_type.95eeb2.2) = assoc_entity element0, imports.%Core.import_ref.0e8 [symbolic = %assoc0 (constants.%assoc0.31355d.2)]
// CHECK:STDOUT:   %RightShiftAssignWith.WithSelf.Op.type: type = fn_type @RightShiftAssignWith.WithSelf.Op, @RightShiftAssignWith.WithSelf(%U.loc229_26.1, %T.loc229_36.1) [symbolic = %RightShiftAssignWith.WithSelf.Op.type (constants.%RightShiftAssignWith.WithSelf.Op.type.f62b2f.3)]
// CHECK:STDOUT:   %.loc231_5.2: type = fn_type_with_self_type %RightShiftAssignWith.WithSelf.Op.type, %T.loc229_36.1 [symbolic = %.loc231_5.2 (constants.%.609)]
// CHECK:STDOUT:   %RightShiftAssignWith.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc229_36.1, @RightShiftAssignWith.1, @RightShiftAssignWith.1(%U.loc229_26.1) [symbolic = %RightShiftAssignWith.lookup_impl_witness (constants.%RightShiftAssignWith.lookup_impl_witness)]
// CHECK:STDOUT:   %impl.elem0.loc231_5.2: @RightShiftAssignWith.loc229.%.loc231_5.2 (%.609) = impl_witness_access %RightShiftAssignWith.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc231_5.2 (constants.%impl.elem0.82d)]
// CHECK:STDOUT:   %specific_impl_fn.loc231_5.2: <specific function> = specific_impl_function %impl.elem0.loc231_5.2, @RightShiftAssignWith.WithSelf.Op(%U.loc229_26.1, %T.loc229_36.1) [symbolic = %specific_impl_fn.loc231_5.2 (constants.%specific_impl_fn.42f)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: ref @RightShiftAssignWith.loc229.%T.as_type.loc229_76.1 (%T.as_type.8e8), %y.param: @RightShiftAssignWith.loc229.%U.loc229_26.1 (%U.67d)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: ref @RightShiftAssignWith.loc229.%T.as_type.loc229_76.1 (%T.as_type.8e8) = name_ref x, %x
// CHECK:STDOUT:     %y.ref: @RightShiftAssignWith.loc229.%U.loc229_26.1 (%U.67d) = name_ref y, %y
// CHECK:STDOUT:     %RightShiftAssignWith.type.loc231: type = facet_type <@RightShiftAssignWith.1, @RightShiftAssignWith.1(constants.%U.67d)> [symbolic = %RightShiftAssignWith.type.loc229_66.1 (constants.%RightShiftAssignWith.type.abebad.2)]
// CHECK:STDOUT:     %.loc231_5.1: @RightShiftAssignWith.loc229.%RightShiftAssignWith.assoc_type (%RightShiftAssignWith.assoc_type.95eeb2.2) = specific_constant imports.%Core.import_ref.eebc, @RightShiftAssignWith.WithSelf(constants.%U.67d, constants.%Self.d7c2e9.1) [symbolic = %assoc0 (constants.%assoc0.31355d.2)]
// CHECK:STDOUT:     %Op.ref: @RightShiftAssignWith.loc229.%RightShiftAssignWith.assoc_type (%RightShiftAssignWith.assoc_type.95eeb2.2) = name_ref Op, %.loc231_5.1 [symbolic = %assoc0 (constants.%assoc0.31355d.2)]
// CHECK:STDOUT:     %impl.elem0.loc231_5.1: @RightShiftAssignWith.loc229.%.loc231_5.2 (%.609) = impl_witness_access constants.%RightShiftAssignWith.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc231_5.2 (constants.%impl.elem0.82d)]
// CHECK:STDOUT:     %bound_method.loc231_5.1: <bound method> = bound_method %x.ref, %impl.elem0.loc231_5.1
// CHECK:STDOUT:     %specific_impl_fn.loc231_5.1: <specific function> = specific_impl_function %impl.elem0.loc231_5.1, @RightShiftAssignWith.WithSelf.Op(constants.%U.67d, constants.%T.d7c) [symbolic = %specific_impl_fn.loc231_5.2 (constants.%specific_impl_fn.42f)]
// CHECK:STDOUT:     %bound_method.loc231_5.2: <bound method> = bound_method %x.ref, %specific_impl_fn.loc231_5.1
// CHECK:STDOUT:     %RightShiftAssignWith.WithSelf.Op.call: init %empty_tuple.type = call %bound_method.loc231_5.2(%x.ref, %y.ref)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @TestBitComplement(%T.loc259_4.2: %BitComplement_where.type.283) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %BitComplement.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc259_4.1, @BitComplement [symbolic = %BitComplement.lookup_impl_witness (constants.%BitComplement.lookup_impl_witness.e4757a.2)]
// CHECK:STDOUT:   %BitComplement.facet.loc262_3.3: %BitComplement.type = facet_value %T.as_type.loc260_6.1, (%BitComplement.lookup_impl_witness) [symbolic = %BitComplement.facet.loc262_3.3 (constants.%BitComplement.facet.5b0)]
// CHECK:STDOUT:   %BitComplement.WithSelf.Op.type: type = fn_type @BitComplement.WithSelf.Op, @BitComplement.WithSelf(%BitComplement.facet.loc262_3.3) [symbolic = %BitComplement.WithSelf.Op.type (constants.%BitComplement.WithSelf.Op.type.c5e)]
// CHECK:STDOUT:   %.loc262_3.7: type = fn_type_with_self_type %BitComplement.WithSelf.Op.type, %BitComplement.facet.loc262_3.3 [symbolic = %.loc262_3.7 (constants.%.bf9)]
// CHECK:STDOUT:   %impl.elem1.loc262_3.2: @TestBitComplement.%.loc262_3.7 (%.bf9) = impl_witness_access %BitComplement.lookup_impl_witness, element1 [symbolic = %impl.elem1.loc262_3.2 (constants.%impl.elem1.2ca)]
// CHECK:STDOUT:   %specific_impl_fn.loc262_3.3: <specific function> = specific_impl_function %impl.elem1.loc262_3.2, @BitComplement.WithSelf.Op(%BitComplement.facet.loc262_3.3) [symbolic = %specific_impl_fn.loc262_3.3 (constants.%specific_impl_fn.856)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %impl.elem0.1, @Destroy [symbolic = %Destroy.lookup_impl_witness (constants.%Destroy.lookup_impl_witness.9f9)]
// CHECK:STDOUT:   %Destroy.facet.loc262_3.3: %Destroy.type = facet_value %impl.elem0.1, (%Destroy.lookup_impl_witness) [symbolic = %Destroy.facet.loc262_3.3 (constants.%Destroy.facet.10a)]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.loc262_3.3) [symbolic = %Destroy.WithSelf.Op.type (constants.%Destroy.WithSelf.Op.type.f14)]
// CHECK:STDOUT:   %.loc262_3.8: type = fn_type_with_self_type %Destroy.WithSelf.Op.type, %Destroy.facet.loc262_3.3 [symbolic = %.loc262_3.8 (constants.%.73b)]
// CHECK:STDOUT:   %impl.elem0.loc262_3.2: @TestBitComplement.%.loc262_3.8 (%.73b) = impl_witness_access %Destroy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc262_3.2 (constants.%impl.elem0.39c)]
// CHECK:STDOUT:   %specific_impl_fn.loc262_3.4: <specific function> = specific_impl_function %impl.elem0.loc262_3.2, @Destroy.WithSelf.Op(%Destroy.facet.loc262_3.3) [symbolic = %specific_impl_fn.loc262_3.4 (constants.%specific_impl_fn.271)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @TestBitComplement.%T.as_type.loc260_6.1 (%T.as_type.e33)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: @TestBitComplement.%T.as_type.loc260_6.1 (%T.as_type.e33) = name_ref x, %x
// CHECK:STDOUT:     %impl.elem1.loc262_3.1: @TestBitComplement.%.loc262_3.7 (%.bf9) = impl_witness_access constants.%BitComplement.lookup_impl_witness.e4757a.2, element1 [symbolic = %impl.elem1.loc262_3.2 (constants.%impl.elem1.2ca)]
// CHECK:STDOUT:     %bound_method.loc262_3.1: <bound method> = bound_method %x.ref, %impl.elem1.loc262_3.1
// CHECK:STDOUT:     %BitComplement.facet.loc262_3.1: %BitComplement.type = facet_value constants.%T.as_type.e33, (constants.%BitComplement.lookup_impl_witness.e4757a.2) [symbolic = %BitComplement.facet.loc262_3.3 (constants.%BitComplement.facet.5b0)]
// CHECK:STDOUT:     %.loc262_3.1: %BitComplement.type = converted constants.%T.as_type.e33, %BitComplement.facet.loc262_3.1 [symbolic = %BitComplement.facet.loc262_3.3 (constants.%BitComplement.facet.5b0)]
// CHECK:STDOUT:     %BitComplement.facet.loc262_3.2: %BitComplement.type = facet_value constants.%T.as_type.e33, (constants.%BitComplement.lookup_impl_witness.e4757a.2) [symbolic = %BitComplement.facet.loc262_3.3 (constants.%BitComplement.facet.5b0)]
// CHECK:STDOUT:     %.loc262_3.2: %BitComplement.type = converted constants.%T.as_type.e33, %BitComplement.facet.loc262_3.2 [symbolic = %BitComplement.facet.loc262_3.3 (constants.%BitComplement.facet.5b0)]
// CHECK:STDOUT:     %specific_impl_fn.loc262_3.1: <specific function> = specific_impl_function %impl.elem1.loc262_3.1, @BitComplement.WithSelf.Op(constants.%BitComplement.facet.5b0) [symbolic = %specific_impl_fn.loc262_3.3 (constants.%specific_impl_fn.856)]
// CHECK:STDOUT:     %bound_method.loc262_3.2: <bound method> = bound_method %x.ref, %specific_impl_fn.loc262_3.1
// CHECK:STDOUT:     %.loc262_3.3: ref @TestBitComplement.%impl.elem0.1 (%impl.elem0.ed21cd.2) = temporary_storage
// CHECK:STDOUT:     %BitComplement.WithSelf.Op.call: init @TestBitComplement.%impl.elem0.1 (%impl.elem0.ed21cd.2) to %.loc262_3.3 = call %bound_method.loc262_3.2(%x.ref)
// CHECK:STDOUT:     %.loc262_3.4: ref @TestBitComplement.%impl.elem0.1 (%impl.elem0.ed21cd.2) = temporary %.loc262_3.3, %BitComplement.WithSelf.Op.call
// CHECK:STDOUT:     %impl.elem0.loc262_3.1: @TestBitComplement.%.loc262_3.8 (%.73b) = impl_witness_access constants.%Destroy.lookup_impl_witness.9f9, element0 [symbolic = %impl.elem0.loc262_3.2 (constants.%impl.elem0.39c)]
// CHECK:STDOUT:     %bound_method.loc262_3.3: <bound method> = bound_method %.loc262_3.4, %impl.elem0.loc262_3.1
// CHECK:STDOUT:     %Destroy.facet.loc262_3.1: %Destroy.type = facet_value constants.%impl.elem0.ed21cd.2, (constants.%Destroy.lookup_impl_witness.9f9) [symbolic = %Destroy.facet.loc262_3.3 (constants.%Destroy.facet.10a)]
// CHECK:STDOUT:     %.loc262_3.5: %Destroy.type = converted constants.%impl.elem0.ed21cd.2, %Destroy.facet.loc262_3.1 [symbolic = %Destroy.facet.loc262_3.3 (constants.%Destroy.facet.10a)]
// CHECK:STDOUT:     %Destroy.facet.loc262_3.2: %Destroy.type = facet_value constants.%impl.elem0.ed21cd.2, (constants.%Destroy.lookup_impl_witness.9f9) [symbolic = %Destroy.facet.loc262_3.3 (constants.%Destroy.facet.10a)]
// CHECK:STDOUT:     %.loc262_3.6: %Destroy.type = converted constants.%impl.elem0.ed21cd.2, %Destroy.facet.loc262_3.2 [symbolic = %Destroy.facet.loc262_3.3 (constants.%Destroy.facet.10a)]
// CHECK:STDOUT:     %specific_impl_fn.loc262_3.2: <specific function> = specific_impl_function %impl.elem0.loc262_3.1, @Destroy.WithSelf.Op(constants.%Destroy.facet.10a) [symbolic = %specific_impl_fn.loc262_3.4 (constants.%specific_impl_fn.271)]
// CHECK:STDOUT:     %bound_method.loc262_3.4: <bound method> = bound_method %.loc262_3.4, %specific_impl_fn.loc262_3.2
// CHECK:STDOUT:     %Destroy.WithSelf.Op.call: init %empty_tuple.type = call %bound_method.loc262_3.4(%.loc262_3.4)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @BitAndWith.loc112(constants.%U.67d, constants.%T.274) {
// CHECK:STDOUT:   %U.patt.loc110_4.2 => constants.%U.patt.d47
// CHECK:STDOUT:   %U.loc110_4.1 => constants.%U.67d
// CHECK:STDOUT:   %BitAndWith.type.loc111_24.1 => constants.%BitAndWith.type.4f130e.2
// CHECK:STDOUT:   %.Self.loc111_26.1 => constants.%.Self.ace
// CHECK:STDOUT:   %require_complete.loc111 => constants.%require_complete.a4e
// CHECK:STDOUT:   %.Self.as_type.loc111_32.1 => constants.%.Self.as_type.107
// CHECK:STDOUT:   %BitAndWith.assoc_type => constants.%BitAndWith.assoc_type.a6e8f9.2
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.24f955.2
// CHECK:STDOUT:   %BitAndWith.lookup_impl_witness.loc111 => constants.%BitAndWith.lookup_impl_witness.4ca
// CHECK:STDOUT:   %impl.elem0.loc111_32.1 => constants.%impl.elem0.c32
// CHECK:STDOUT:   %BitAndWith_where.type => constants.%BitAndWith_where.type.4f1
// CHECK:STDOUT:   %pattern_type.loc111 => constants.%pattern_type.626
// CHECK:STDOUT:   %T.patt.loc111_4.2 => constants.%T.patt.96b
// CHECK:STDOUT:   %T.loc111_4.1 => constants.%T.274
// CHECK:STDOUT:   %T.as_type.loc112_6.1 => constants.%T.as_type.417
// CHECK:STDOUT:   %pattern_type.loc112_4 => constants.%pattern_type.acade5.2
// CHECK:STDOUT:   %x.param_patt.loc112_4.2 => constants.%x.param_patt.1ce
// CHECK:STDOUT:   %x.patt.loc112_4.2 => constants.%x.patt.b16
// CHECK:STDOUT:   %pattern_type.loc112_10 => constants.%pattern_type.51d1c4.2
// CHECK:STDOUT:   %y.param_patt.loc112_10.2 => constants.%y.param_patt.be3
// CHECK:STDOUT:   %y.patt.loc112_10.2 => constants.%y.patt.b85
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @BitOrWith.loc121(constants.%U.67d, constants.%T.6cd) {
// CHECK:STDOUT:   %U.patt.loc119_4.2 => constants.%U.patt.d47
// CHECK:STDOUT:   %U.loc119_4.1 => constants.%U.67d
// CHECK:STDOUT:   %BitOrWith.type.loc120_23.1 => constants.%BitOrWith.type.d96408.2
// CHECK:STDOUT:   %.Self.loc120_25.1 => constants.%.Self.cf4
// CHECK:STDOUT:   %require_complete.loc120 => constants.%require_complete.818
// CHECK:STDOUT:   %.Self.as_type.loc120_31.1 => constants.%.Self.as_type.7a4
// CHECK:STDOUT:   %BitOrWith.assoc_type => constants.%BitOrWith.assoc_type.10d7b7.2
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.7f25cd.2
// CHECK:STDOUT:   %BitOrWith.lookup_impl_witness.loc120 => constants.%BitOrWith.lookup_impl_witness.c5e
// CHECK:STDOUT:   %impl.elem0.loc120_31.1 => constants.%impl.elem0.324
// CHECK:STDOUT:   %BitOrWith_where.type => constants.%BitOrWith_where.type.d96
// CHECK:STDOUT:   %pattern_type.loc120 => constants.%pattern_type.f1d
// CHECK:STDOUT:   %T.patt.loc120_4.2 => constants.%T.patt.ecb
// CHECK:STDOUT:   %T.loc120_4.1 => constants.%T.6cd
// CHECK:STDOUT:   %T.as_type.loc121_6.1 => constants.%T.as_type.725
// CHECK:STDOUT:   %pattern_type.loc121_4 => constants.%pattern_type.29dfc2.2
// CHECK:STDOUT:   %x.param_patt.loc121_4.2 => constants.%x.param_patt.42d
// CHECK:STDOUT:   %x.patt.loc121_4.2 => constants.%x.patt.b29
// CHECK:STDOUT:   %pattern_type.loc121_10 => constants.%pattern_type.51d1c4.2
// CHECK:STDOUT:   %y.param_patt.loc121_10.2 => constants.%y.param_patt.be3
// CHECK:STDOUT:   %y.patt.loc121_10.2 => constants.%y.patt.b85
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @BitXorWith.loc130(constants.%U.67d, constants.%T.5c2) {
// CHECK:STDOUT:   %U.patt.loc128_4.2 => constants.%U.patt.d47
// CHECK:STDOUT:   %U.loc128_4.1 => constants.%U.67d
// CHECK:STDOUT:   %BitXorWith.type.loc129_24.1 => constants.%BitXorWith.type.7de562.2
// CHECK:STDOUT:   %.Self.loc129_26.1 => constants.%.Self.9cb
// CHECK:STDOUT:   %require_complete.loc129 => constants.%require_complete.189
// CHECK:STDOUT:   %.Self.as_type.loc129_32.1 => constants.%.Self.as_type.ea1
// CHECK:STDOUT:   %BitXorWith.assoc_type => constants.%BitXorWith.assoc_type.4ad612.2
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.8aa692.2
// CHECK:STDOUT:   %BitXorWith.lookup_impl_witness.loc129 => constants.%BitXorWith.lookup_impl_witness.e34
// CHECK:STDOUT:   %impl.elem0.loc129_32.1 => constants.%impl.elem0.4e7
// CHECK:STDOUT:   %BitXorWith_where.type => constants.%BitXorWith_where.type.7de
// CHECK:STDOUT:   %pattern_type.loc129 => constants.%pattern_type.676
// CHECK:STDOUT:   %T.patt.loc129_4.2 => constants.%T.patt.9d1
// CHECK:STDOUT:   %T.loc129_4.1 => constants.%T.5c2
// CHECK:STDOUT:   %T.as_type.loc130_6.1 => constants.%T.as_type.f34
// CHECK:STDOUT:   %pattern_type.loc130_4 => constants.%pattern_type.3ed574.2
// CHECK:STDOUT:   %x.param_patt.loc130_4.2 => constants.%x.param_patt.f3e
// CHECK:STDOUT:   %x.patt.loc130_4.2 => constants.%x.patt.260
// CHECK:STDOUT:   %pattern_type.loc130_10 => constants.%pattern_type.51d1c4.2
// CHECK:STDOUT:   %y.param_patt.loc130_10.2 => constants.%y.param_patt.be3
// CHECK:STDOUT:   %y.patt.loc130_10.2 => constants.%y.patt.b85
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @LeftShiftWith.loc139(constants.%U.67d, constants.%T.1e3) {
// CHECK:STDOUT:   %U.patt.loc137_4.2 => constants.%U.patt.d47
// CHECK:STDOUT:   %U.loc137_4.1 => constants.%U.67d
// CHECK:STDOUT:   %LeftShiftWith.type.loc138_27.1 => constants.%LeftShiftWith.type.cd59d0.2
// CHECK:STDOUT:   %.Self.loc138_29.1 => constants.%.Self.e0e
// CHECK:STDOUT:   %require_complete.loc138 => constants.%require_complete.d6f
// CHECK:STDOUT:   %.Self.as_type.loc138_35.1 => constants.%.Self.as_type.332
// CHECK:STDOUT:   %LeftShiftWith.assoc_type => constants.%LeftShiftWith.assoc_type.739d6a.2
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.0c7500.2
// CHECK:STDOUT:   %LeftShiftWith.lookup_impl_witness.loc138 => constants.%LeftShiftWith.lookup_impl_witness.b27
// CHECK:STDOUT:   %impl.elem0.loc138_35.1 => constants.%impl.elem0.554
// CHECK:STDOUT:   %LeftShiftWith_where.type => constants.%LeftShiftWith_where.type.cd5
// CHECK:STDOUT:   %pattern_type.loc138 => constants.%pattern_type.8ee
// CHECK:STDOUT:   %T.patt.loc138_4.2 => constants.%T.patt.09f
// CHECK:STDOUT:   %T.loc138_4.1 => constants.%T.1e3
// CHECK:STDOUT:   %T.as_type.loc139_6.1 => constants.%T.as_type.9e2
// CHECK:STDOUT:   %pattern_type.loc139_4 => constants.%pattern_type.10f0b4.2
// CHECK:STDOUT:   %x.param_patt.loc139_4.2 => constants.%x.param_patt.6c9
// CHECK:STDOUT:   %x.patt.loc139_4.2 => constants.%x.patt.63e
// CHECK:STDOUT:   %pattern_type.loc139_10 => constants.%pattern_type.51d1c4.2
// CHECK:STDOUT:   %y.param_patt.loc139_10.2 => constants.%y.param_patt.be3
// CHECK:STDOUT:   %y.patt.loc139_10.2 => constants.%y.patt.b85
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @RightShiftWith.loc148(constants.%U.67d, constants.%T.665) {
// CHECK:STDOUT:   %U.patt.loc146_4.2 => constants.%U.patt.d47
// CHECK:STDOUT:   %U.loc146_4.1 => constants.%U.67d
// CHECK:STDOUT:   %RightShiftWith.type.loc147_28.1 => constants.%RightShiftWith.type.60db85.2
// CHECK:STDOUT:   %.Self.loc147_30.1 => constants.%.Self.7eb
// CHECK:STDOUT:   %require_complete.loc147 => constants.%require_complete.4ce
// CHECK:STDOUT:   %.Self.as_type.loc147_36.1 => constants.%.Self.as_type.66a
// CHECK:STDOUT:   %RightShiftWith.assoc_type => constants.%RightShiftWith.assoc_type.4d2607.2
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.01e147.2
// CHECK:STDOUT:   %RightShiftWith.lookup_impl_witness.loc147 => constants.%RightShiftWith.lookup_impl_witness.d40
// CHECK:STDOUT:   %impl.elem0.loc147_36.1 => constants.%impl.elem0.8d0
// CHECK:STDOUT:   %RightShiftWith_where.type => constants.%RightShiftWith_where.type.60d
// CHECK:STDOUT:   %pattern_type.loc147 => constants.%pattern_type.26c
// CHECK:STDOUT:   %T.patt.loc147_4.2 => constants.%T.patt.3d2
// CHECK:STDOUT:   %T.loc147_4.1 => constants.%T.665
// CHECK:STDOUT:   %T.as_type.loc148_6.1 => constants.%T.as_type.e4f
// CHECK:STDOUT:   %pattern_type.loc148_4 => constants.%pattern_type.8b4a4f.2
// CHECK:STDOUT:   %x.param_patt.loc148_4.2 => constants.%x.param_patt.c4e
// CHECK:STDOUT:   %x.patt.loc148_4.2 => constants.%x.patt.f73
// CHECK:STDOUT:   %pattern_type.loc148_10 => constants.%pattern_type.51d1c4.2
// CHECK:STDOUT:   %y.param_patt.loc148_10.2 => constants.%y.param_patt.be3
// CHECK:STDOUT:   %y.patt.loc148_10.2 => constants.%y.patt.b85
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @BitAndWith.loc112(constants.%Int16, constants.%facet_value.4da) {
// CHECK:STDOUT:   %U.patt.loc110_4.2 => constants.%U.patt.d47
// CHECK:STDOUT:   %U.loc110_4.1 => constants.%Int16
// CHECK:STDOUT:   %BitAndWith.type.loc111_24.1 => constants.%BitAndWith.type.ff7
// CHECK:STDOUT:   %.Self.loc111_26.1 => constants.%.Self.6b1
// CHECK:STDOUT:   %require_complete.loc111 => constants.%complete_type.7b3
// CHECK:STDOUT:   %.Self.as_type.loc111_32.1 => constants.%.Self.as_type.d50
// CHECK:STDOUT:   %BitAndWith.assoc_type => constants.%BitAndWith.assoc_type.33d
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.261
// CHECK:STDOUT:   %BitAndWith.lookup_impl_witness.loc111 => constants.%BitAndWith.lookup_impl_witness.194
// CHECK:STDOUT:   %impl.elem0.loc111_32.1 => constants.%impl.elem0.38f
// CHECK:STDOUT:   %BitAndWith_where.type => constants.%BitAndWith_where.type.ff7
// CHECK:STDOUT:   %pattern_type.loc111 => constants.%pattern_type.9a6
// CHECK:STDOUT:   %T.patt.loc111_4.2 => constants.%T.patt.30e
// CHECK:STDOUT:   %T.loc111_4.1 => constants.%facet_value.4da
// CHECK:STDOUT:   %T.as_type.loc112_6.1 => constants.%i16
// CHECK:STDOUT:   %pattern_type.loc112_4 => constants.%pattern_type.203
// CHECK:STDOUT:   %x.param_patt.loc112_4.2 => constants.%x.param_patt.676
// CHECK:STDOUT:   %x.patt.loc112_4.2 => constants.%x.patt.26e
// CHECK:STDOUT:   %pattern_type.loc112_10 => constants.%pattern_type.785
// CHECK:STDOUT:   %y.param_patt.loc112_10.2 => constants.%y.param_patt.418
// CHECK:STDOUT:   %y.patt.loc112_10.2 => constants.%y.patt.4e6
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc112_4 => constants.%complete_type.2da
// CHECK:STDOUT:   %require_complete.loc112_10 => constants.%complete_type.357
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.0ca
// CHECK:STDOUT:   %BitAndWith.lookup_impl_witness.loc114 => constants.%custom_witness.703
// CHECK:STDOUT:   %BitAndWith.facet => constants.%BitAndWith.facet.b2d
// CHECK:STDOUT:   %BitAndWith.WithSelf.Op.type => constants.%BitAndWith.WithSelf.Op.type.c4e
// CHECK:STDOUT:   %.loc114_5.6 => constants.%.566
// CHECK:STDOUT:   %impl.elem1.loc114_5.2 => constants.%Op.72717c.1
// CHECK:STDOUT:   %specific_impl_fn.loc114_5.3 => constants.%Op.72717c.1
// CHECK:STDOUT:   %impl.elem0.1 => constants.%Int16
// CHECK:STDOUT:   %require_complete.1 => constants.%complete_type.357
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.e7a31c.1
// CHECK:STDOUT:   %Destroy.facet.loc114_5.3 => constants.%Destroy.facet.2cc
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.9d5
// CHECK:STDOUT:   %.loc114_5.7 => constants.%.6da
// CHECK:STDOUT:   %impl.elem0.loc114_5.2 => constants.%Int16.Op.4dca31.2
// CHECK:STDOUT:   %specific_impl_fn.loc114_5.4 => constants.%Int16.Op.4dca31.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @BitOrWith.loc121(constants.%Int16, constants.%facet_value.a01) {
// CHECK:STDOUT:   %U.patt.loc119_4.2 => constants.%U.patt.d47
// CHECK:STDOUT:   %U.loc119_4.1 => constants.%Int16
// CHECK:STDOUT:   %BitOrWith.type.loc120_23.1 => constants.%BitOrWith.type.bbb
// CHECK:STDOUT:   %.Self.loc120_25.1 => constants.%.Self.6b8
// CHECK:STDOUT:   %require_complete.loc120 => constants.%complete_type.4d3
// CHECK:STDOUT:   %.Self.as_type.loc120_31.1 => constants.%.Self.as_type.cf1
// CHECK:STDOUT:   %BitOrWith.assoc_type => constants.%BitOrWith.assoc_type.fd3
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.e30
// CHECK:STDOUT:   %BitOrWith.lookup_impl_witness.loc120 => constants.%BitOrWith.lookup_impl_witness.08b
// CHECK:STDOUT:   %impl.elem0.loc120_31.1 => constants.%impl.elem0.8bf
// CHECK:STDOUT:   %BitOrWith_where.type => constants.%BitOrWith_where.type.bbb
// CHECK:STDOUT:   %pattern_type.loc120 => constants.%pattern_type.69e
// CHECK:STDOUT:   %T.patt.loc120_4.2 => constants.%T.patt.476
// CHECK:STDOUT:   %T.loc120_4.1 => constants.%facet_value.a01
// CHECK:STDOUT:   %T.as_type.loc121_6.1 => constants.%i16
// CHECK:STDOUT:   %pattern_type.loc121_4 => constants.%pattern_type.203
// CHECK:STDOUT:   %x.param_patt.loc121_4.2 => constants.%x.param_patt.676
// CHECK:STDOUT:   %x.patt.loc121_4.2 => constants.%x.patt.26e
// CHECK:STDOUT:   %pattern_type.loc121_10 => constants.%pattern_type.785
// CHECK:STDOUT:   %y.param_patt.loc121_10.2 => constants.%y.param_patt.418
// CHECK:STDOUT:   %y.patt.loc121_10.2 => constants.%y.patt.4e6
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc121_4 => constants.%complete_type.2da
// CHECK:STDOUT:   %require_complete.loc121_10 => constants.%complete_type.357
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.6f1
// CHECK:STDOUT:   %BitOrWith.lookup_impl_witness.loc123 => constants.%custom_witness.55d8
// CHECK:STDOUT:   %BitOrWith.facet => constants.%BitOrWith.facet.ed1
// CHECK:STDOUT:   %BitOrWith.WithSelf.Op.type => constants.%BitOrWith.WithSelf.Op.type.2a3
// CHECK:STDOUT:   %.loc123_5.6 => constants.%.980
// CHECK:STDOUT:   %impl.elem1.loc123_5.2 => constants.%Op.72717c.2
// CHECK:STDOUT:   %specific_impl_fn.loc123_5.3 => constants.%Op.72717c.2
// CHECK:STDOUT:   %impl.elem0.1 => constants.%Int16
// CHECK:STDOUT:   %require_complete.1 => constants.%complete_type.357
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.e7a31c.1
// CHECK:STDOUT:   %Destroy.facet.loc123_5.3 => constants.%Destroy.facet.2cc
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.9d5
// CHECK:STDOUT:   %.loc123_5.7 => constants.%.6da
// CHECK:STDOUT:   %impl.elem0.loc123_5.2 => constants.%Int16.Op.4dca31.2
// CHECK:STDOUT:   %specific_impl_fn.loc123_5.4 => constants.%Int16.Op.4dca31.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @BitXorWith.loc130(constants.%Int16, constants.%facet_value.547) {
// CHECK:STDOUT:   %U.patt.loc128_4.2 => constants.%U.patt.d47
// CHECK:STDOUT:   %U.loc128_4.1 => constants.%Int16
// CHECK:STDOUT:   %BitXorWith.type.loc129_24.1 => constants.%BitXorWith.type.bd3
// CHECK:STDOUT:   %.Self.loc129_26.1 => constants.%.Self.2ca
// CHECK:STDOUT:   %require_complete.loc129 => constants.%complete_type.430
// CHECK:STDOUT:   %.Self.as_type.loc129_32.1 => constants.%.Self.as_type.75e4
// CHECK:STDOUT:   %BitXorWith.assoc_type => constants.%BitXorWith.assoc_type.679
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.22f
// CHECK:STDOUT:   %BitXorWith.lookup_impl_witness.loc129 => constants.%BitXorWith.lookup_impl_witness.c72
// CHECK:STDOUT:   %impl.elem0.loc129_32.1 => constants.%impl.elem0.b16
// CHECK:STDOUT:   %BitXorWith_where.type => constants.%BitXorWith_where.type.bd3
// CHECK:STDOUT:   %pattern_type.loc129 => constants.%pattern_type.80d
// CHECK:STDOUT:   %T.patt.loc129_4.2 => constants.%T.patt.f57
// CHECK:STDOUT:   %T.loc129_4.1 => constants.%facet_value.547
// CHECK:STDOUT:   %T.as_type.loc130_6.1 => constants.%i16
// CHECK:STDOUT:   %pattern_type.loc130_4 => constants.%pattern_type.203
// CHECK:STDOUT:   %x.param_patt.loc130_4.2 => constants.%x.param_patt.676
// CHECK:STDOUT:   %x.patt.loc130_4.2 => constants.%x.patt.26e
// CHECK:STDOUT:   %pattern_type.loc130_10 => constants.%pattern_type.785
// CHECK:STDOUT:   %y.param_patt.loc130_10.2 => constants.%y.param_patt.418
// CHECK:STDOUT:   %y.patt.loc130_10.2 => constants.%y.patt.4e6
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc130_4 => constants.%complete_type.2da
// CHECK:STDOUT:   %require_complete.loc130_10 => constants.%complete_type.357
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.d55
// CHECK:STDOUT:   %BitXorWith.lookup_impl_witness.loc132 => constants.%custom_witness.c18
// CHECK:STDOUT:   %BitXorWith.facet => constants.%BitXorWith.facet.d55
// CHECK:STDOUT:   %BitXorWith.WithSelf.Op.type => constants.%BitXorWith.WithSelf.Op.type.69e
// CHECK:STDOUT:   %.loc132_5.6 => constants.%.88e
// CHECK:STDOUT:   %impl.elem1.loc132_5.2 => constants.%Op.72717c.3
// CHECK:STDOUT:   %specific_impl_fn.loc132_5.3 => constants.%Op.72717c.3
// CHECK:STDOUT:   %impl.elem0.1 => constants.%Int16
// CHECK:STDOUT:   %require_complete.1 => constants.%complete_type.357
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.e7a31c.1
// CHECK:STDOUT:   %Destroy.facet.loc132_5.3 => constants.%Destroy.facet.2cc
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.9d5
// CHECK:STDOUT:   %.loc132_5.7 => constants.%.6da
// CHECK:STDOUT:   %impl.elem0.loc132_5.2 => constants.%Int16.Op.4dca31.2
// CHECK:STDOUT:   %specific_impl_fn.loc132_5.4 => constants.%Int16.Op.4dca31.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @LeftShiftWith.loc139(constants.%Int16, constants.%facet_value.931) {
// CHECK:STDOUT:   %U.patt.loc137_4.2 => constants.%U.patt.d47
// CHECK:STDOUT:   %U.loc137_4.1 => constants.%Int16
// CHECK:STDOUT:   %LeftShiftWith.type.loc138_27.1 => constants.%LeftShiftWith.type.fff
// CHECK:STDOUT:   %.Self.loc138_29.1 => constants.%.Self.dce
// CHECK:STDOUT:   %require_complete.loc138 => constants.%complete_type.68c
// CHECK:STDOUT:   %.Self.as_type.loc138_35.1 => constants.%.Self.as_type.0bd
// CHECK:STDOUT:   %LeftShiftWith.assoc_type => constants.%LeftShiftWith.assoc_type.ee4
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.749
// CHECK:STDOUT:   %LeftShiftWith.lookup_impl_witness.loc138 => constants.%LeftShiftWith.lookup_impl_witness.635
// CHECK:STDOUT:   %impl.elem0.loc138_35.1 => constants.%impl.elem0.df3
// CHECK:STDOUT:   %LeftShiftWith_where.type => constants.%LeftShiftWith_where.type.fff
// CHECK:STDOUT:   %pattern_type.loc138 => constants.%pattern_type.64f
// CHECK:STDOUT:   %T.patt.loc138_4.2 => constants.%T.patt.209
// CHECK:STDOUT:   %T.loc138_4.1 => constants.%facet_value.931
// CHECK:STDOUT:   %T.as_type.loc139_6.1 => constants.%i16
// CHECK:STDOUT:   %pattern_type.loc139_4 => constants.%pattern_type.203
// CHECK:STDOUT:   %x.param_patt.loc139_4.2 => constants.%x.param_patt.676
// CHECK:STDOUT:   %x.patt.loc139_4.2 => constants.%x.patt.26e
// CHECK:STDOUT:   %pattern_type.loc139_10 => constants.%pattern_type.785
// CHECK:STDOUT:   %y.param_patt.loc139_10.2 => constants.%y.param_patt.418
// CHECK:STDOUT:   %y.patt.loc139_10.2 => constants.%y.patt.4e6
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc139_4 => constants.%complete_type.2da
// CHECK:STDOUT:   %require_complete.loc139_10 => constants.%complete_type.357
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.7f6
// CHECK:STDOUT:   %LeftShiftWith.lookup_impl_witness.loc141 => constants.%custom_witness.598
// CHECK:STDOUT:   %LeftShiftWith.facet => constants.%LeftShiftWith.facet.ce6
// CHECK:STDOUT:   %LeftShiftWith.WithSelf.Op.type => constants.%LeftShiftWith.WithSelf.Op.type.2cb
// CHECK:STDOUT:   %.loc141_5.6 => constants.%.8ba
// CHECK:STDOUT:   %impl.elem1.loc141_5.2 => constants.%Op.72717c.4
// CHECK:STDOUT:   %specific_impl_fn.loc141_5.3 => constants.%Op.72717c.4
// CHECK:STDOUT:   %impl.elem0.1 => constants.%Int16
// CHECK:STDOUT:   %require_complete.1 => constants.%complete_type.357
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.e7a31c.1
// CHECK:STDOUT:   %Destroy.facet.loc141_5.3 => constants.%Destroy.facet.2cc
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.9d5
// CHECK:STDOUT:   %.loc141_5.7 => constants.%.6da
// CHECK:STDOUT:   %impl.elem0.loc141_5.2 => constants.%Int16.Op.4dca31.2
// CHECK:STDOUT:   %specific_impl_fn.loc141_5.4 => constants.%Int16.Op.4dca31.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @RightShiftWith.loc148(constants.%Int16, constants.%facet_value.1bd) {
// CHECK:STDOUT:   %U.patt.loc146_4.2 => constants.%U.patt.d47
// CHECK:STDOUT:   %U.loc146_4.1 => constants.%Int16
// CHECK:STDOUT:   %RightShiftWith.type.loc147_28.1 => constants.%RightShiftWith.type.386
// CHECK:STDOUT:   %.Self.loc147_30.1 => constants.%.Self.1fe
// CHECK:STDOUT:   %require_complete.loc147 => constants.%complete_type.938
// CHECK:STDOUT:   %.Self.as_type.loc147_36.1 => constants.%.Self.as_type.7d1
// CHECK:STDOUT:   %RightShiftWith.assoc_type => constants.%RightShiftWith.assoc_type.656
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.148
// CHECK:STDOUT:   %RightShiftWith.lookup_impl_witness.loc147 => constants.%RightShiftWith.lookup_impl_witness.1b3
// CHECK:STDOUT:   %impl.elem0.loc147_36.1 => constants.%impl.elem0.baa
// CHECK:STDOUT:   %RightShiftWith_where.type => constants.%RightShiftWith_where.type.386
// CHECK:STDOUT:   %pattern_type.loc147 => constants.%pattern_type.761
// CHECK:STDOUT:   %T.patt.loc147_4.2 => constants.%T.patt.806
// CHECK:STDOUT:   %T.loc147_4.1 => constants.%facet_value.1bd
// CHECK:STDOUT:   %T.as_type.loc148_6.1 => constants.%i16
// CHECK:STDOUT:   %pattern_type.loc148_4 => constants.%pattern_type.203
// CHECK:STDOUT:   %x.param_patt.loc148_4.2 => constants.%x.param_patt.676
// CHECK:STDOUT:   %x.patt.loc148_4.2 => constants.%x.patt.26e
// CHECK:STDOUT:   %pattern_type.loc148_10 => constants.%pattern_type.785
// CHECK:STDOUT:   %y.param_patt.loc148_10.2 => constants.%y.param_patt.418
// CHECK:STDOUT:   %y.patt.loc148_10.2 => constants.%y.patt.4e6
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc148_4 => constants.%complete_type.2da
// CHECK:STDOUT:   %require_complete.loc148_10 => constants.%complete_type.357
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.5a2
// CHECK:STDOUT:   %RightShiftWith.lookup_impl_witness.loc150 => constants.%custom_witness.24d
// CHECK:STDOUT:   %RightShiftWith.facet => constants.%RightShiftWith.facet.4d1
// CHECK:STDOUT:   %RightShiftWith.WithSelf.Op.type => constants.%RightShiftWith.WithSelf.Op.type.808
// CHECK:STDOUT:   %.loc150_5.6 => constants.%.c0a
// CHECK:STDOUT:   %impl.elem1.loc150_5.2 => constants.%Op.72717c.5
// CHECK:STDOUT:   %specific_impl_fn.loc150_5.3 => constants.%Op.72717c.5
// CHECK:STDOUT:   %impl.elem0.1 => constants.%Int16
// CHECK:STDOUT:   %require_complete.1 => constants.%complete_type.357
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.e7a31c.1
// CHECK:STDOUT:   %Destroy.facet.loc150_5.3 => constants.%Destroy.facet.2cc
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.9d5
// CHECK:STDOUT:   %.loc150_5.7 => constants.%.6da
// CHECK:STDOUT:   %impl.elem0.loc150_5.2 => constants.%Int16.Op.4dca31.2
// CHECK:STDOUT:   %specific_impl_fn.loc150_5.4 => constants.%Int16.Op.4dca31.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @BitAndWith.loc112(constants.%i16, constants.%facet_value.e6c) {
// CHECK:STDOUT:   %U.patt.loc110_4.2 => constants.%U.patt.d47
// CHECK:STDOUT:   %U.loc110_4.1 => constants.%i16
// CHECK:STDOUT:   %BitAndWith.type.loc111_24.1 => constants.%BitAndWith.type.569
// CHECK:STDOUT:   %.Self.loc111_26.1 => constants.%.Self.1ec
// CHECK:STDOUT:   %require_complete.loc111 => constants.%complete_type.c28
// CHECK:STDOUT:   %.Self.as_type.loc111_32.1 => constants.%.Self.as_type.986
// CHECK:STDOUT:   %BitAndWith.assoc_type => constants.%BitAndWith.assoc_type.419
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.bb5
// CHECK:STDOUT:   %BitAndWith.lookup_impl_witness.loc111 => constants.%BitAndWith.lookup_impl_witness.dcb
// CHECK:STDOUT:   %impl.elem0.loc111_32.1 => constants.%impl.elem0.c1c
// CHECK:STDOUT:   %BitAndWith_where.type => constants.%BitAndWith_where.type.569
// CHECK:STDOUT:   %pattern_type.loc111 => constants.%pattern_type.06f
// CHECK:STDOUT:   %T.patt.loc111_4.2 => constants.%T.patt.2aa
// CHECK:STDOUT:   %T.loc111_4.1 => constants.%facet_value.e6c
// CHECK:STDOUT:   %T.as_type.loc112_6.1 => constants.%Int16
// CHECK:STDOUT:   %pattern_type.loc112_4 => constants.%pattern_type.785
// CHECK:STDOUT:   %x.param_patt.loc112_4.2 => constants.%x.param_patt.3b0
// CHECK:STDOUT:   %x.patt.loc112_4.2 => constants.%x.patt.b76
// CHECK:STDOUT:   %pattern_type.loc112_10 => constants.%pattern_type.203
// CHECK:STDOUT:   %y.param_patt.loc112_10.2 => constants.%y.param_patt.4b4
// CHECK:STDOUT:   %y.patt.loc112_10.2 => constants.%y.patt.dd3
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc112_4 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc112_10 => constants.%complete_type.2da
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.38a
// CHECK:STDOUT:   %BitAndWith.lookup_impl_witness.loc114 => constants.%custom_witness.e77
// CHECK:STDOUT:   %BitAndWith.facet => constants.%BitAndWith.facet.064
// CHECK:STDOUT:   %BitAndWith.WithSelf.Op.type => constants.%BitAndWith.WithSelf.Op.type.37a
// CHECK:STDOUT:   %.loc114_5.6 => constants.%.732
// CHECK:STDOUT:   %impl.elem1.loc114_5.2 => constants.%Int16.Op.4dca31.3
// CHECK:STDOUT:   %specific_impl_fn.loc114_5.3 => constants.%Int16.Op.4dca31.3
// CHECK:STDOUT:   %impl.elem0.1 => constants.%Int16
// CHECK:STDOUT:   %require_complete.1 => constants.%complete_type.357
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.e7a31c.1
// CHECK:STDOUT:   %Destroy.facet.loc114_5.3 => constants.%Destroy.facet.2cc
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.9d5
// CHECK:STDOUT:   %.loc114_5.7 => constants.%.6da
// CHECK:STDOUT:   %impl.elem0.loc114_5.2 => constants.%Int16.Op.4dca31.2
// CHECK:STDOUT:   %specific_impl_fn.loc114_5.4 => constants.%Int16.Op.4dca31.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @BitOrWith.loc121(constants.%i16, constants.%facet_value.d1b) {
// CHECK:STDOUT:   %U.patt.loc119_4.2 => constants.%U.patt.d47
// CHECK:STDOUT:   %U.loc119_4.1 => constants.%i16
// CHECK:STDOUT:   %BitOrWith.type.loc120_23.1 => constants.%BitOrWith.type.e34
// CHECK:STDOUT:   %.Self.loc120_25.1 => constants.%.Self.5cb
// CHECK:STDOUT:   %require_complete.loc120 => constants.%complete_type.440
// CHECK:STDOUT:   %.Self.as_type.loc120_31.1 => constants.%.Self.as_type.91b
// CHECK:STDOUT:   %BitOrWith.assoc_type => constants.%BitOrWith.assoc_type.2db
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.b52
// CHECK:STDOUT:   %BitOrWith.lookup_impl_witness.loc120 => constants.%BitOrWith.lookup_impl_witness.946
// CHECK:STDOUT:   %impl.elem0.loc120_31.1 => constants.%impl.elem0.5b0
// CHECK:STDOUT:   %BitOrWith_where.type => constants.%BitOrWith_where.type.e34
// CHECK:STDOUT:   %pattern_type.loc120 => constants.%pattern_type.6f1
// CHECK:STDOUT:   %T.patt.loc120_4.2 => constants.%T.patt.273
// CHECK:STDOUT:   %T.loc120_4.1 => constants.%facet_value.d1b
// CHECK:STDOUT:   %T.as_type.loc121_6.1 => constants.%Int16
// CHECK:STDOUT:   %pattern_type.loc121_4 => constants.%pattern_type.785
// CHECK:STDOUT:   %x.param_patt.loc121_4.2 => constants.%x.param_patt.3b0
// CHECK:STDOUT:   %x.patt.loc121_4.2 => constants.%x.patt.b76
// CHECK:STDOUT:   %pattern_type.loc121_10 => constants.%pattern_type.203
// CHECK:STDOUT:   %y.param_patt.loc121_10.2 => constants.%y.param_patt.4b4
// CHECK:STDOUT:   %y.patt.loc121_10.2 => constants.%y.patt.dd3
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc121_4 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc121_10 => constants.%complete_type.2da
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.90e
// CHECK:STDOUT:   %BitOrWith.lookup_impl_witness.loc123 => constants.%custom_witness.9c0
// CHECK:STDOUT:   %BitOrWith.facet => constants.%BitOrWith.facet.bab
// CHECK:STDOUT:   %BitOrWith.WithSelf.Op.type => constants.%BitOrWith.WithSelf.Op.type.45d
// CHECK:STDOUT:   %.loc123_5.6 => constants.%.855
// CHECK:STDOUT:   %impl.elem1.loc123_5.2 => constants.%Int16.Op.4dca31.4
// CHECK:STDOUT:   %specific_impl_fn.loc123_5.3 => constants.%Int16.Op.4dca31.4
// CHECK:STDOUT:   %impl.elem0.1 => constants.%Int16
// CHECK:STDOUT:   %require_complete.1 => constants.%complete_type.357
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.e7a31c.1
// CHECK:STDOUT:   %Destroy.facet.loc123_5.3 => constants.%Destroy.facet.2cc
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.9d5
// CHECK:STDOUT:   %.loc123_5.7 => constants.%.6da
// CHECK:STDOUT:   %impl.elem0.loc123_5.2 => constants.%Int16.Op.4dca31.2
// CHECK:STDOUT:   %specific_impl_fn.loc123_5.4 => constants.%Int16.Op.4dca31.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @BitXorWith.loc130(constants.%i16, constants.%facet_value.4f5) {
// CHECK:STDOUT:   %U.patt.loc128_4.2 => constants.%U.patt.d47
// CHECK:STDOUT:   %U.loc128_4.1 => constants.%i16
// CHECK:STDOUT:   %BitXorWith.type.loc129_24.1 => constants.%BitXorWith.type.716
// CHECK:STDOUT:   %.Self.loc129_26.1 => constants.%.Self.ca9
// CHECK:STDOUT:   %require_complete.loc129 => constants.%complete_type.5e0
// CHECK:STDOUT:   %.Self.as_type.loc129_32.1 => constants.%.Self.as_type.684
// CHECK:STDOUT:   %BitXorWith.assoc_type => constants.%BitXorWith.assoc_type.a40
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.b33
// CHECK:STDOUT:   %BitXorWith.lookup_impl_witness.loc129 => constants.%BitXorWith.lookup_impl_witness.220
// CHECK:STDOUT:   %impl.elem0.loc129_32.1 => constants.%impl.elem0.260
// CHECK:STDOUT:   %BitXorWith_where.type => constants.%BitXorWith_where.type.716
// CHECK:STDOUT:   %pattern_type.loc129 => constants.%pattern_type.4c9
// CHECK:STDOUT:   %T.patt.loc129_4.2 => constants.%T.patt.187
// CHECK:STDOUT:   %T.loc129_4.1 => constants.%facet_value.4f5
// CHECK:STDOUT:   %T.as_type.loc130_6.1 => constants.%Int16
// CHECK:STDOUT:   %pattern_type.loc130_4 => constants.%pattern_type.785
// CHECK:STDOUT:   %x.param_patt.loc130_4.2 => constants.%x.param_patt.3b0
// CHECK:STDOUT:   %x.patt.loc130_4.2 => constants.%x.patt.b76
// CHECK:STDOUT:   %pattern_type.loc130_10 => constants.%pattern_type.203
// CHECK:STDOUT:   %y.param_patt.loc130_10.2 => constants.%y.param_patt.4b4
// CHECK:STDOUT:   %y.patt.loc130_10.2 => constants.%y.patt.dd3
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc130_4 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc130_10 => constants.%complete_type.2da
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.9b1
// CHECK:STDOUT:   %BitXorWith.lookup_impl_witness.loc132 => constants.%custom_witness.3ee
// CHECK:STDOUT:   %BitXorWith.facet => constants.%BitXorWith.facet.7da
// CHECK:STDOUT:   %BitXorWith.WithSelf.Op.type => constants.%BitXorWith.WithSelf.Op.type.b89
// CHECK:STDOUT:   %.loc132_5.6 => constants.%.15f
// CHECK:STDOUT:   %impl.elem1.loc132_5.2 => constants.%Int16.Op.4dca31.5
// CHECK:STDOUT:   %specific_impl_fn.loc132_5.3 => constants.%Int16.Op.4dca31.5
// CHECK:STDOUT:   %impl.elem0.1 => constants.%Int16
// CHECK:STDOUT:   %require_complete.1 => constants.%complete_type.357
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.e7a31c.1
// CHECK:STDOUT:   %Destroy.facet.loc132_5.3 => constants.%Destroy.facet.2cc
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.9d5
// CHECK:STDOUT:   %.loc132_5.7 => constants.%.6da
// CHECK:STDOUT:   %impl.elem0.loc132_5.2 => constants.%Int16.Op.4dca31.2
// CHECK:STDOUT:   %specific_impl_fn.loc132_5.4 => constants.%Int16.Op.4dca31.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @LeftShiftWith.loc139(constants.%i16, constants.%facet_value.f61) {
// CHECK:STDOUT:   %U.patt.loc137_4.2 => constants.%U.patt.d47
// CHECK:STDOUT:   %U.loc137_4.1 => constants.%i16
// CHECK:STDOUT:   %LeftShiftWith.type.loc138_27.1 => constants.%LeftShiftWith.type.0ee
// CHECK:STDOUT:   %.Self.loc138_29.1 => constants.%.Self.859
// CHECK:STDOUT:   %require_complete.loc138 => constants.%complete_type.bdc
// CHECK:STDOUT:   %.Self.as_type.loc138_35.1 => constants.%.Self.as_type.81e
// CHECK:STDOUT:   %LeftShiftWith.assoc_type => constants.%LeftShiftWith.assoc_type.fae
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.78f
// CHECK:STDOUT:   %LeftShiftWith.lookup_impl_witness.loc138 => constants.%LeftShiftWith.lookup_impl_witness.a61
// CHECK:STDOUT:   %impl.elem0.loc138_35.1 => constants.%impl.elem0.4d6
// CHECK:STDOUT:   %LeftShiftWith_where.type => constants.%LeftShiftWith_where.type.0ee
// CHECK:STDOUT:   %pattern_type.loc138 => constants.%pattern_type.117
// CHECK:STDOUT:   %T.patt.loc138_4.2 => constants.%T.patt.985
// CHECK:STDOUT:   %T.loc138_4.1 => constants.%facet_value.f61
// CHECK:STDOUT:   %T.as_type.loc139_6.1 => constants.%Int16
// CHECK:STDOUT:   %pattern_type.loc139_4 => constants.%pattern_type.785
// CHECK:STDOUT:   %x.param_patt.loc139_4.2 => constants.%x.param_patt.3b0
// CHECK:STDOUT:   %x.patt.loc139_4.2 => constants.%x.patt.b76
// CHECK:STDOUT:   %pattern_type.loc139_10 => constants.%pattern_type.203
// CHECK:STDOUT:   %y.param_patt.loc139_10.2 => constants.%y.param_patt.4b4
// CHECK:STDOUT:   %y.patt.loc139_10.2 => constants.%y.patt.dd3
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc139_4 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc139_10 => constants.%complete_type.2da
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.a06
// CHECK:STDOUT:   %LeftShiftWith.lookup_impl_witness.loc141 => constants.%custom_witness.e12
// CHECK:STDOUT:   %LeftShiftWith.facet => constants.%LeftShiftWith.facet.071
// CHECK:STDOUT:   %LeftShiftWith.WithSelf.Op.type => constants.%LeftShiftWith.WithSelf.Op.type.c7d
// CHECK:STDOUT:   %.loc141_5.6 => constants.%.e97
// CHECK:STDOUT:   %impl.elem1.loc141_5.2 => constants.%Int16.Op.4dca31.6
// CHECK:STDOUT:   %specific_impl_fn.loc141_5.3 => constants.%Int16.Op.4dca31.6
// CHECK:STDOUT:   %impl.elem0.1 => constants.%Int16
// CHECK:STDOUT:   %require_complete.1 => constants.%complete_type.357
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.e7a31c.1
// CHECK:STDOUT:   %Destroy.facet.loc141_5.3 => constants.%Destroy.facet.2cc
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.9d5
// CHECK:STDOUT:   %.loc141_5.7 => constants.%.6da
// CHECK:STDOUT:   %impl.elem0.loc141_5.2 => constants.%Int16.Op.4dca31.2
// CHECK:STDOUT:   %specific_impl_fn.loc141_5.4 => constants.%Int16.Op.4dca31.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @RightShiftWith.loc148(constants.%i16, constants.%facet_value.475) {
// CHECK:STDOUT:   %U.patt.loc146_4.2 => constants.%U.patt.d47
// CHECK:STDOUT:   %U.loc146_4.1 => constants.%i16
// CHECK:STDOUT:   %RightShiftWith.type.loc147_28.1 => constants.%RightShiftWith.type.f4e
// CHECK:STDOUT:   %.Self.loc147_30.1 => constants.%.Self.487
// CHECK:STDOUT:   %require_complete.loc147 => constants.%complete_type.5ea
// CHECK:STDOUT:   %.Self.as_type.loc147_36.1 => constants.%.Self.as_type.73f
// CHECK:STDOUT:   %RightShiftWith.assoc_type => constants.%RightShiftWith.assoc_type.b9f
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.d8e
// CHECK:STDOUT:   %RightShiftWith.lookup_impl_witness.loc147 => constants.%RightShiftWith.lookup_impl_witness.f7c
// CHECK:STDOUT:   %impl.elem0.loc147_36.1 => constants.%impl.elem0.54e
// CHECK:STDOUT:   %RightShiftWith_where.type => constants.%RightShiftWith_where.type.f4e
// CHECK:STDOUT:   %pattern_type.loc147 => constants.%pattern_type.9cd
// CHECK:STDOUT:   %T.patt.loc147_4.2 => constants.%T.patt.9f0
// CHECK:STDOUT:   %T.loc147_4.1 => constants.%facet_value.475
// CHECK:STDOUT:   %T.as_type.loc148_6.1 => constants.%Int16
// CHECK:STDOUT:   %pattern_type.loc148_4 => constants.%pattern_type.785
// CHECK:STDOUT:   %x.param_patt.loc148_4.2 => constants.%x.param_patt.3b0
// CHECK:STDOUT:   %x.patt.loc148_4.2 => constants.%x.patt.b76
// CHECK:STDOUT:   %pattern_type.loc148_10 => constants.%pattern_type.203
// CHECK:STDOUT:   %y.param_patt.loc148_10.2 => constants.%y.param_patt.4b4
// CHECK:STDOUT:   %y.patt.loc148_10.2 => constants.%y.patt.dd3
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc148_4 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc148_10 => constants.%complete_type.2da
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.1d4
// CHECK:STDOUT:   %RightShiftWith.lookup_impl_witness.loc150 => constants.%custom_witness.ba0
// CHECK:STDOUT:   %RightShiftWith.facet => constants.%RightShiftWith.facet.fa8
// CHECK:STDOUT:   %RightShiftWith.WithSelf.Op.type => constants.%RightShiftWith.WithSelf.Op.type.2ed
// CHECK:STDOUT:   %.loc150_5.6 => constants.%.0c2
// CHECK:STDOUT:   %impl.elem1.loc150_5.2 => constants.%Int16.Op.4dca31.7
// CHECK:STDOUT:   %specific_impl_fn.loc150_5.3 => constants.%Int16.Op.4dca31.7
// CHECK:STDOUT:   %impl.elem0.1 => constants.%Int16
// CHECK:STDOUT:   %require_complete.1 => constants.%complete_type.357
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.e7a31c.1
// CHECK:STDOUT:   %Destroy.facet.loc150_5.3 => constants.%Destroy.facet.2cc
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.9d5
// CHECK:STDOUT:   %.loc150_5.7 => constants.%.6da
// CHECK:STDOUT:   %impl.elem0.loc150_5.2 => constants.%Int16.Op.4dca31.2
// CHECK:STDOUT:   %specific_impl_fn.loc150_5.4 => constants.%Int16.Op.4dca31.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @BitAndWith.loc112(constants.%Int32, constants.%facet_value.e6f) {
// CHECK:STDOUT:   %U.patt.loc110_4.2 => constants.%U.patt.d47
// CHECK:STDOUT:   %U.loc110_4.1 => constants.%Int32
// CHECK:STDOUT:   %BitAndWith.type.loc111_24.1 => constants.%BitAndWith.type.f43
// CHECK:STDOUT:   %.Self.loc111_26.1 => constants.%.Self.b06
// CHECK:STDOUT:   %require_complete.loc111 => constants.%complete_type.008
// CHECK:STDOUT:   %.Self.as_type.loc111_32.1 => constants.%.Self.as_type.b4c
// CHECK:STDOUT:   %BitAndWith.assoc_type => constants.%BitAndWith.assoc_type.6e7
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.543
// CHECK:STDOUT:   %BitAndWith.lookup_impl_witness.loc111 => constants.%BitAndWith.lookup_impl_witness.773
// CHECK:STDOUT:   %impl.elem0.loc111_32.1 => constants.%impl.elem0.b5e
// CHECK:STDOUT:   %BitAndWith_where.type => constants.%BitAndWith_where.type.f43
// CHECK:STDOUT:   %pattern_type.loc111 => constants.%pattern_type.f97
// CHECK:STDOUT:   %T.patt.loc111_4.2 => constants.%T.patt.126
// CHECK:STDOUT:   %T.loc111_4.1 => constants.%facet_value.e6f
// CHECK:STDOUT:   %T.as_type.loc112_6.1 => constants.%Int16
// CHECK:STDOUT:   %pattern_type.loc112_4 => constants.%pattern_type.785
// CHECK:STDOUT:   %x.param_patt.loc112_4.2 => constants.%x.param_patt.3b0
// CHECK:STDOUT:   %x.patt.loc112_4.2 => constants.%x.patt.b76
// CHECK:STDOUT:   %pattern_type.loc112_10 => constants.%pattern_type.8b45
// CHECK:STDOUT:   %y.param_patt.loc112_10.2 => constants.%y.param_patt.e0c
// CHECK:STDOUT:   %y.patt.loc112_10.2 => constants.%y.patt.0a3
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc112_4 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc112_10 => constants.%complete_type.357
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.5b1
// CHECK:STDOUT:   %BitAndWith.lookup_impl_witness.loc114 => constants.%custom_witness.9a1dfa.1
// CHECK:STDOUT:   %BitAndWith.facet => constants.%BitAndWith.facet.fc0
// CHECK:STDOUT:   %BitAndWith.WithSelf.Op.type => constants.%BitAndWith.WithSelf.Op.type.7c5
// CHECK:STDOUT:   %.loc114_5.6 => constants.%.463
// CHECK:STDOUT:   %impl.elem1.loc114_5.2 => constants.%Int16.Op.4dca31.8
// CHECK:STDOUT:   %specific_impl_fn.loc114_5.3 => constants.%Int16.Op.4dca31.8
// CHECK:STDOUT:   %impl.elem0.1 => constants.%Int32
// CHECK:STDOUT:   %require_complete.1 => constants.%complete_type.357
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.e7a31c.2
// CHECK:STDOUT:   %Destroy.facet.loc114_5.3 => constants.%Destroy.facet.099
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.b76
// CHECK:STDOUT:   %.loc114_5.7 => constants.%.043
// CHECK:STDOUT:   %impl.elem0.loc114_5.2 => constants.%Int32.Op.4dca31.1
// CHECK:STDOUT:   %specific_impl_fn.loc114_5.4 => constants.%Int32.Op.4dca31.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @BitOrWith.loc121(constants.%Int32, constants.%facet_value.b44) {
// CHECK:STDOUT:   %U.patt.loc119_4.2 => constants.%U.patt.d47
// CHECK:STDOUT:   %U.loc119_4.1 => constants.%Int32
// CHECK:STDOUT:   %BitOrWith.type.loc120_23.1 => constants.%BitOrWith.type.878
// CHECK:STDOUT:   %.Self.loc120_25.1 => constants.%.Self.f1c
// CHECK:STDOUT:   %require_complete.loc120 => constants.%complete_type.4fe
// CHECK:STDOUT:   %.Self.as_type.loc120_31.1 => constants.%.Self.as_type.4f4
// CHECK:STDOUT:   %BitOrWith.assoc_type => constants.%BitOrWith.assoc_type.0490
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.d36
// CHECK:STDOUT:   %BitOrWith.lookup_impl_witness.loc120 => constants.%BitOrWith.lookup_impl_witness.93b
// CHECK:STDOUT:   %impl.elem0.loc120_31.1 => constants.%impl.elem0.181
// CHECK:STDOUT:   %BitOrWith_where.type => constants.%BitOrWith_where.type.878
// CHECK:STDOUT:   %pattern_type.loc120 => constants.%pattern_type.a2c
// CHECK:STDOUT:   %T.patt.loc120_4.2 => constants.%T.patt.3c3
// CHECK:STDOUT:   %T.loc120_4.1 => constants.%facet_value.b44
// CHECK:STDOUT:   %T.as_type.loc121_6.1 => constants.%Int16
// CHECK:STDOUT:   %pattern_type.loc121_4 => constants.%pattern_type.785
// CHECK:STDOUT:   %x.param_patt.loc121_4.2 => constants.%x.param_patt.3b0
// CHECK:STDOUT:   %x.patt.loc121_4.2 => constants.%x.patt.b76
// CHECK:STDOUT:   %pattern_type.loc121_10 => constants.%pattern_type.8b45
// CHECK:STDOUT:   %y.param_patt.loc121_10.2 => constants.%y.param_patt.e0c
// CHECK:STDOUT:   %y.patt.loc121_10.2 => constants.%y.patt.0a3
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc121_4 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc121_10 => constants.%complete_type.357
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.953
// CHECK:STDOUT:   %BitOrWith.lookup_impl_witness.loc123 => constants.%custom_witness.2e067a.1
// CHECK:STDOUT:   %BitOrWith.facet => constants.%BitOrWith.facet.da7
// CHECK:STDOUT:   %BitOrWith.WithSelf.Op.type => constants.%BitOrWith.WithSelf.Op.type.acf
// CHECK:STDOUT:   %.loc123_5.6 => constants.%.23d2b
// CHECK:STDOUT:   %impl.elem1.loc123_5.2 => constants.%Int16.Op.4dca31.9
// CHECK:STDOUT:   %specific_impl_fn.loc123_5.3 => constants.%Int16.Op.4dca31.9
// CHECK:STDOUT:   %impl.elem0.1 => constants.%Int32
// CHECK:STDOUT:   %require_complete.1 => constants.%complete_type.357
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.e7a31c.2
// CHECK:STDOUT:   %Destroy.facet.loc123_5.3 => constants.%Destroy.facet.099
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.b76
// CHECK:STDOUT:   %.loc123_5.7 => constants.%.043
// CHECK:STDOUT:   %impl.elem0.loc123_5.2 => constants.%Int32.Op.4dca31.1
// CHECK:STDOUT:   %specific_impl_fn.loc123_5.4 => constants.%Int32.Op.4dca31.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @BitXorWith.loc130(constants.%Int32, constants.%facet_value.857) {
// CHECK:STDOUT:   %U.patt.loc128_4.2 => constants.%U.patt.d47
// CHECK:STDOUT:   %U.loc128_4.1 => constants.%Int32
// CHECK:STDOUT:   %BitXorWith.type.loc129_24.1 => constants.%BitXorWith.type.729
// CHECK:STDOUT:   %.Self.loc129_26.1 => constants.%.Self.be9
// CHECK:STDOUT:   %require_complete.loc129 => constants.%complete_type.8eb
// CHECK:STDOUT:   %.Self.as_type.loc129_32.1 => constants.%.Self.as_type.192
// CHECK:STDOUT:   %BitXorWith.assoc_type => constants.%BitXorWith.assoc_type.36a
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.0ab
// CHECK:STDOUT:   %BitXorWith.lookup_impl_witness.loc129 => constants.%BitXorWith.lookup_impl_witness.de4
// CHECK:STDOUT:   %impl.elem0.loc129_32.1 => constants.%impl.elem0.1ab
// CHECK:STDOUT:   %BitXorWith_where.type => constants.%BitXorWith_where.type.729
// CHECK:STDOUT:   %pattern_type.loc129 => constants.%pattern_type.ab4
// CHECK:STDOUT:   %T.patt.loc129_4.2 => constants.%T.patt.1c4
// CHECK:STDOUT:   %T.loc129_4.1 => constants.%facet_value.857
// CHECK:STDOUT:   %T.as_type.loc130_6.1 => constants.%Int16
// CHECK:STDOUT:   %pattern_type.loc130_4 => constants.%pattern_type.785
// CHECK:STDOUT:   %x.param_patt.loc130_4.2 => constants.%x.param_patt.3b0
// CHECK:STDOUT:   %x.patt.loc130_4.2 => constants.%x.patt.b76
// CHECK:STDOUT:   %pattern_type.loc130_10 => constants.%pattern_type.8b45
// CHECK:STDOUT:   %y.param_patt.loc130_10.2 => constants.%y.param_patt.e0c
// CHECK:STDOUT:   %y.patt.loc130_10.2 => constants.%y.patt.0a3
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc130_4 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc130_10 => constants.%complete_type.357
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.951
// CHECK:STDOUT:   %BitXorWith.lookup_impl_witness.loc132 => constants.%custom_witness.5d016d.1
// CHECK:STDOUT:   %BitXorWith.facet => constants.%BitXorWith.facet.121
// CHECK:STDOUT:   %BitXorWith.WithSelf.Op.type => constants.%BitXorWith.WithSelf.Op.type.713
// CHECK:STDOUT:   %.loc132_5.6 => constants.%.ac9
// CHECK:STDOUT:   %impl.elem1.loc132_5.2 => constants.%Int16.Op.4dca31.10
// CHECK:STDOUT:   %specific_impl_fn.loc132_5.3 => constants.%Int16.Op.4dca31.10
// CHECK:STDOUT:   %impl.elem0.1 => constants.%Int32
// CHECK:STDOUT:   %require_complete.1 => constants.%complete_type.357
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.e7a31c.2
// CHECK:STDOUT:   %Destroy.facet.loc132_5.3 => constants.%Destroy.facet.099
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.b76
// CHECK:STDOUT:   %.loc132_5.7 => constants.%.043
// CHECK:STDOUT:   %impl.elem0.loc132_5.2 => constants.%Int32.Op.4dca31.1
// CHECK:STDOUT:   %specific_impl_fn.loc132_5.4 => constants.%Int32.Op.4dca31.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @LeftShiftWith.loc139(constants.%Int32, constants.%facet_value.d98) {
// CHECK:STDOUT:   %U.patt.loc137_4.2 => constants.%U.patt.d47
// CHECK:STDOUT:   %U.loc137_4.1 => constants.%Int32
// CHECK:STDOUT:   %LeftShiftWith.type.loc138_27.1 => constants.%LeftShiftWith.type.15a
// CHECK:STDOUT:   %.Self.loc138_29.1 => constants.%.Self.b90
// CHECK:STDOUT:   %require_complete.loc138 => constants.%complete_type.5b7
// CHECK:STDOUT:   %.Self.as_type.loc138_35.1 => constants.%.Self.as_type.b53
// CHECK:STDOUT:   %LeftShiftWith.assoc_type => constants.%LeftShiftWith.assoc_type.e1d
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.6e4
// CHECK:STDOUT:   %LeftShiftWith.lookup_impl_witness.loc138 => constants.%LeftShiftWith.lookup_impl_witness.140
// CHECK:STDOUT:   %impl.elem0.loc138_35.1 => constants.%impl.elem0.e18
// CHECK:STDOUT:   %LeftShiftWith_where.type => constants.%LeftShiftWith_where.type.15a
// CHECK:STDOUT:   %pattern_type.loc138 => constants.%pattern_type.538
// CHECK:STDOUT:   %T.patt.loc138_4.2 => constants.%T.patt.4a0
// CHECK:STDOUT:   %T.loc138_4.1 => constants.%facet_value.d98
// CHECK:STDOUT:   %T.as_type.loc139_6.1 => constants.%Int16
// CHECK:STDOUT:   %pattern_type.loc139_4 => constants.%pattern_type.785
// CHECK:STDOUT:   %x.param_patt.loc139_4.2 => constants.%x.param_patt.3b0
// CHECK:STDOUT:   %x.patt.loc139_4.2 => constants.%x.patt.b76
// CHECK:STDOUT:   %pattern_type.loc139_10 => constants.%pattern_type.8b45
// CHECK:STDOUT:   %y.param_patt.loc139_10.2 => constants.%y.param_patt.e0c
// CHECK:STDOUT:   %y.patt.loc139_10.2 => constants.%y.patt.0a3
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc139_4 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc139_10 => constants.%complete_type.357
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.f2a
// CHECK:STDOUT:   %LeftShiftWith.lookup_impl_witness.loc141 => constants.%custom_witness.65d217.1
// CHECK:STDOUT:   %LeftShiftWith.facet => constants.%LeftShiftWith.facet.4c3e
// CHECK:STDOUT:   %LeftShiftWith.WithSelf.Op.type => constants.%LeftShiftWith.WithSelf.Op.type.b2c
// CHECK:STDOUT:   %.loc141_5.6 => constants.%.489
// CHECK:STDOUT:   %impl.elem1.loc141_5.2 => constants.%Int16.Op.4dca31.11
// CHECK:STDOUT:   %specific_impl_fn.loc141_5.3 => constants.%Int16.Op.4dca31.11
// CHECK:STDOUT:   %impl.elem0.1 => constants.%Int32
// CHECK:STDOUT:   %require_complete.1 => constants.%complete_type.357
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.e7a31c.2
// CHECK:STDOUT:   %Destroy.facet.loc141_5.3 => constants.%Destroy.facet.099
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.b76
// CHECK:STDOUT:   %.loc141_5.7 => constants.%.043
// CHECK:STDOUT:   %impl.elem0.loc141_5.2 => constants.%Int32.Op.4dca31.1
// CHECK:STDOUT:   %specific_impl_fn.loc141_5.4 => constants.%Int32.Op.4dca31.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @RightShiftWith.loc148(constants.%Int32, constants.%facet_value.431) {
// CHECK:STDOUT:   %U.patt.loc146_4.2 => constants.%U.patt.d47
// CHECK:STDOUT:   %U.loc146_4.1 => constants.%Int32
// CHECK:STDOUT:   %RightShiftWith.type.loc147_28.1 => constants.%RightShiftWith.type.8f0
// CHECK:STDOUT:   %.Self.loc147_30.1 => constants.%.Self.c91
// CHECK:STDOUT:   %require_complete.loc147 => constants.%complete_type.93f
// CHECK:STDOUT:   %.Self.as_type.loc147_36.1 => constants.%.Self.as_type.d6f
// CHECK:STDOUT:   %RightShiftWith.assoc_type => constants.%RightShiftWith.assoc_type.813
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.99a
// CHECK:STDOUT:   %RightShiftWith.lookup_impl_witness.loc147 => constants.%RightShiftWith.lookup_impl_witness.bb0
// CHECK:STDOUT:   %impl.elem0.loc147_36.1 => constants.%impl.elem0.cfe
// CHECK:STDOUT:   %RightShiftWith_where.type => constants.%RightShiftWith_where.type.8f0
// CHECK:STDOUT:   %pattern_type.loc147 => constants.%pattern_type.5a0
// CHECK:STDOUT:   %T.patt.loc147_4.2 => constants.%T.patt.a69
// CHECK:STDOUT:   %T.loc147_4.1 => constants.%facet_value.431
// CHECK:STDOUT:   %T.as_type.loc148_6.1 => constants.%Int16
// CHECK:STDOUT:   %pattern_type.loc148_4 => constants.%pattern_type.785
// CHECK:STDOUT:   %x.param_patt.loc148_4.2 => constants.%x.param_patt.3b0
// CHECK:STDOUT:   %x.patt.loc148_4.2 => constants.%x.patt.b76
// CHECK:STDOUT:   %pattern_type.loc148_10 => constants.%pattern_type.8b45
// CHECK:STDOUT:   %y.param_patt.loc148_10.2 => constants.%y.param_patt.e0c
// CHECK:STDOUT:   %y.patt.loc148_10.2 => constants.%y.patt.0a3
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc148_4 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc148_10 => constants.%complete_type.357
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.2c8
// CHECK:STDOUT:   %RightShiftWith.lookup_impl_witness.loc150 => constants.%custom_witness.884138.1
// CHECK:STDOUT:   %RightShiftWith.facet => constants.%RightShiftWith.facet.0ca
// CHECK:STDOUT:   %RightShiftWith.WithSelf.Op.type => constants.%RightShiftWith.WithSelf.Op.type.66e
// CHECK:STDOUT:   %.loc150_5.6 => constants.%.e00
// CHECK:STDOUT:   %impl.elem1.loc150_5.2 => constants.%Int16.Op.4dca31.12
// CHECK:STDOUT:   %specific_impl_fn.loc150_5.3 => constants.%Int16.Op.4dca31.12
// CHECK:STDOUT:   %impl.elem0.1 => constants.%Int32
// CHECK:STDOUT:   %require_complete.1 => constants.%complete_type.357
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.e7a31c.2
// CHECK:STDOUT:   %Destroy.facet.loc150_5.3 => constants.%Destroy.facet.099
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.b76
// CHECK:STDOUT:   %.loc150_5.7 => constants.%.043
// CHECK:STDOUT:   %impl.elem0.loc150_5.2 => constants.%Int32.Op.4dca31.1
// CHECK:STDOUT:   %specific_impl_fn.loc150_5.4 => constants.%Int32.Op.4dca31.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @BitAndWith.loc112(constants.%Int16, constants.%facet_value.454) {
// CHECK:STDOUT:   %U.patt.loc110_4.2 => constants.%U.patt.d47
// CHECK:STDOUT:   %U.loc110_4.1 => constants.%Int16
// CHECK:STDOUT:   %BitAndWith.type.loc111_24.1 => constants.%BitAndWith.type.ff7
// CHECK:STDOUT:   %.Self.loc111_26.1 => constants.%.Self.6b1
// CHECK:STDOUT:   %require_complete.loc111 => constants.%complete_type.7b3
// CHECK:STDOUT:   %.Self.as_type.loc111_32.1 => constants.%.Self.as_type.d50
// CHECK:STDOUT:   %BitAndWith.assoc_type => constants.%BitAndWith.assoc_type.33d
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.261
// CHECK:STDOUT:   %BitAndWith.lookup_impl_witness.loc111 => constants.%BitAndWith.lookup_impl_witness.194
// CHECK:STDOUT:   %impl.elem0.loc111_32.1 => constants.%impl.elem0.38f
// CHECK:STDOUT:   %BitAndWith_where.type => constants.%BitAndWith_where.type.ff7
// CHECK:STDOUT:   %pattern_type.loc111 => constants.%pattern_type.9a6
// CHECK:STDOUT:   %T.patt.loc111_4.2 => constants.%T.patt.30e
// CHECK:STDOUT:   %T.loc111_4.1 => constants.%facet_value.454
// CHECK:STDOUT:   %T.as_type.loc112_6.1 => constants.%Int32
// CHECK:STDOUT:   %pattern_type.loc112_4 => constants.%pattern_type.8b45
// CHECK:STDOUT:   %x.param_patt.loc112_4.2 => constants.%x.param_patt.27c
// CHECK:STDOUT:   %x.patt.loc112_4.2 => constants.%x.patt.4c9
// CHECK:STDOUT:   %pattern_type.loc112_10 => constants.%pattern_type.785
// CHECK:STDOUT:   %y.param_patt.loc112_10.2 => constants.%y.param_patt.418
// CHECK:STDOUT:   %y.patt.loc112_10.2 => constants.%y.patt.4e6
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc112_4 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc112_10 => constants.%complete_type.357
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.0ca
// CHECK:STDOUT:   %BitAndWith.lookup_impl_witness.loc114 => constants.%custom_witness.25b
// CHECK:STDOUT:   %BitAndWith.facet => constants.%BitAndWith.facet.58a
// CHECK:STDOUT:   %BitAndWith.WithSelf.Op.type => constants.%BitAndWith.WithSelf.Op.type.ffe
// CHECK:STDOUT:   %.loc114_5.6 => constants.%.1f1
// CHECK:STDOUT:   %impl.elem1.loc114_5.2 => constants.%Int32.Op.4dca31.2
// CHECK:STDOUT:   %specific_impl_fn.loc114_5.3 => constants.%Int32.Op.4dca31.2
// CHECK:STDOUT:   %impl.elem0.1 => constants.%Int32
// CHECK:STDOUT:   %require_complete.1 => constants.%complete_type.357
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.e7a31c.2
// CHECK:STDOUT:   %Destroy.facet.loc114_5.3 => constants.%Destroy.facet.099
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.b76
// CHECK:STDOUT:   %.loc114_5.7 => constants.%.043
// CHECK:STDOUT:   %impl.elem0.loc114_5.2 => constants.%Int32.Op.4dca31.1
// CHECK:STDOUT:   %specific_impl_fn.loc114_5.4 => constants.%Int32.Op.4dca31.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @BitOrWith.loc121(constants.%Int16, constants.%facet_value.d4b) {
// CHECK:STDOUT:   %U.patt.loc119_4.2 => constants.%U.patt.d47
// CHECK:STDOUT:   %U.loc119_4.1 => constants.%Int16
// CHECK:STDOUT:   %BitOrWith.type.loc120_23.1 => constants.%BitOrWith.type.bbb
// CHECK:STDOUT:   %.Self.loc120_25.1 => constants.%.Self.6b8
// CHECK:STDOUT:   %require_complete.loc120 => constants.%complete_type.4d3
// CHECK:STDOUT:   %.Self.as_type.loc120_31.1 => constants.%.Self.as_type.cf1
// CHECK:STDOUT:   %BitOrWith.assoc_type => constants.%BitOrWith.assoc_type.fd3
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.e30
// CHECK:STDOUT:   %BitOrWith.lookup_impl_witness.loc120 => constants.%BitOrWith.lookup_impl_witness.08b
// CHECK:STDOUT:   %impl.elem0.loc120_31.1 => constants.%impl.elem0.8bf
// CHECK:STDOUT:   %BitOrWith_where.type => constants.%BitOrWith_where.type.bbb
// CHECK:STDOUT:   %pattern_type.loc120 => constants.%pattern_type.69e
// CHECK:STDOUT:   %T.patt.loc120_4.2 => constants.%T.patt.476
// CHECK:STDOUT:   %T.loc120_4.1 => constants.%facet_value.d4b
// CHECK:STDOUT:   %T.as_type.loc121_6.1 => constants.%Int32
// CHECK:STDOUT:   %pattern_type.loc121_4 => constants.%pattern_type.8b45
// CHECK:STDOUT:   %x.param_patt.loc121_4.2 => constants.%x.param_patt.27c
// CHECK:STDOUT:   %x.patt.loc121_4.2 => constants.%x.patt.4c9
// CHECK:STDOUT:   %pattern_type.loc121_10 => constants.%pattern_type.785
// CHECK:STDOUT:   %y.param_patt.loc121_10.2 => constants.%y.param_patt.418
// CHECK:STDOUT:   %y.patt.loc121_10.2 => constants.%y.patt.4e6
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc121_4 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc121_10 => constants.%complete_type.357
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.6f1
// CHECK:STDOUT:   %BitOrWith.lookup_impl_witness.loc123 => constants.%custom_witness.205
// CHECK:STDOUT:   %BitOrWith.facet => constants.%BitOrWith.facet.c7b
// CHECK:STDOUT:   %BitOrWith.WithSelf.Op.type => constants.%BitOrWith.WithSelf.Op.type.b5d
// CHECK:STDOUT:   %.loc123_5.6 => constants.%.fb6
// CHECK:STDOUT:   %impl.elem1.loc123_5.2 => constants.%Int32.Op.4dca31.3
// CHECK:STDOUT:   %specific_impl_fn.loc123_5.3 => constants.%Int32.Op.4dca31.3
// CHECK:STDOUT:   %impl.elem0.1 => constants.%Int32
// CHECK:STDOUT:   %require_complete.1 => constants.%complete_type.357
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.e7a31c.2
// CHECK:STDOUT:   %Destroy.facet.loc123_5.3 => constants.%Destroy.facet.099
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.b76
// CHECK:STDOUT:   %.loc123_5.7 => constants.%.043
// CHECK:STDOUT:   %impl.elem0.loc123_5.2 => constants.%Int32.Op.4dca31.1
// CHECK:STDOUT:   %specific_impl_fn.loc123_5.4 => constants.%Int32.Op.4dca31.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @BitXorWith.loc130(constants.%Int16, constants.%facet_value.fd3) {
// CHECK:STDOUT:   %U.patt.loc128_4.2 => constants.%U.patt.d47
// CHECK:STDOUT:   %U.loc128_4.1 => constants.%Int16
// CHECK:STDOUT:   %BitXorWith.type.loc129_24.1 => constants.%BitXorWith.type.bd3
// CHECK:STDOUT:   %.Self.loc129_26.1 => constants.%.Self.2ca
// CHECK:STDOUT:   %require_complete.loc129 => constants.%complete_type.430
// CHECK:STDOUT:   %.Self.as_type.loc129_32.1 => constants.%.Self.as_type.75e4
// CHECK:STDOUT:   %BitXorWith.assoc_type => constants.%BitXorWith.assoc_type.679
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.22f
// CHECK:STDOUT:   %BitXorWith.lookup_impl_witness.loc129 => constants.%BitXorWith.lookup_impl_witness.c72
// CHECK:STDOUT:   %impl.elem0.loc129_32.1 => constants.%impl.elem0.b16
// CHECK:STDOUT:   %BitXorWith_where.type => constants.%BitXorWith_where.type.bd3
// CHECK:STDOUT:   %pattern_type.loc129 => constants.%pattern_type.80d
// CHECK:STDOUT:   %T.patt.loc129_4.2 => constants.%T.patt.f57
// CHECK:STDOUT:   %T.loc129_4.1 => constants.%facet_value.fd3
// CHECK:STDOUT:   %T.as_type.loc130_6.1 => constants.%Int32
// CHECK:STDOUT:   %pattern_type.loc130_4 => constants.%pattern_type.8b45
// CHECK:STDOUT:   %x.param_patt.loc130_4.2 => constants.%x.param_patt.27c
// CHECK:STDOUT:   %x.patt.loc130_4.2 => constants.%x.patt.4c9
// CHECK:STDOUT:   %pattern_type.loc130_10 => constants.%pattern_type.785
// CHECK:STDOUT:   %y.param_patt.loc130_10.2 => constants.%y.param_patt.418
// CHECK:STDOUT:   %y.patt.loc130_10.2 => constants.%y.patt.4e6
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc130_4 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc130_10 => constants.%complete_type.357
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.d55
// CHECK:STDOUT:   %BitXorWith.lookup_impl_witness.loc132 => constants.%custom_witness.cc2
// CHECK:STDOUT:   %BitXorWith.facet => constants.%BitXorWith.facet.ac3
// CHECK:STDOUT:   %BitXorWith.WithSelf.Op.type => constants.%BitXorWith.WithSelf.Op.type.100
// CHECK:STDOUT:   %.loc132_5.6 => constants.%.8ad
// CHECK:STDOUT:   %impl.elem1.loc132_5.2 => constants.%Int32.Op.4dca31.4
// CHECK:STDOUT:   %specific_impl_fn.loc132_5.3 => constants.%Int32.Op.4dca31.4
// CHECK:STDOUT:   %impl.elem0.1 => constants.%Int32
// CHECK:STDOUT:   %require_complete.1 => constants.%complete_type.357
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.e7a31c.2
// CHECK:STDOUT:   %Destroy.facet.loc132_5.3 => constants.%Destroy.facet.099
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.b76
// CHECK:STDOUT:   %.loc132_5.7 => constants.%.043
// CHECK:STDOUT:   %impl.elem0.loc132_5.2 => constants.%Int32.Op.4dca31.1
// CHECK:STDOUT:   %specific_impl_fn.loc132_5.4 => constants.%Int32.Op.4dca31.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @LeftShiftWith.loc139(constants.%Int16, constants.%facet_value.281) {
// CHECK:STDOUT:   %U.patt.loc137_4.2 => constants.%U.patt.d47
// CHECK:STDOUT:   %U.loc137_4.1 => constants.%Int16
// CHECK:STDOUT:   %LeftShiftWith.type.loc138_27.1 => constants.%LeftShiftWith.type.fff
// CHECK:STDOUT:   %.Self.loc138_29.1 => constants.%.Self.dce
// CHECK:STDOUT:   %require_complete.loc138 => constants.%complete_type.68c
// CHECK:STDOUT:   %.Self.as_type.loc138_35.1 => constants.%.Self.as_type.0bd
// CHECK:STDOUT:   %LeftShiftWith.assoc_type => constants.%LeftShiftWith.assoc_type.ee4
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.749
// CHECK:STDOUT:   %LeftShiftWith.lookup_impl_witness.loc138 => constants.%LeftShiftWith.lookup_impl_witness.635
// CHECK:STDOUT:   %impl.elem0.loc138_35.1 => constants.%impl.elem0.df3
// CHECK:STDOUT:   %LeftShiftWith_where.type => constants.%LeftShiftWith_where.type.fff
// CHECK:STDOUT:   %pattern_type.loc138 => constants.%pattern_type.64f
// CHECK:STDOUT:   %T.patt.loc138_4.2 => constants.%T.patt.209
// CHECK:STDOUT:   %T.loc138_4.1 => constants.%facet_value.281
// CHECK:STDOUT:   %T.as_type.loc139_6.1 => constants.%Int32
// CHECK:STDOUT:   %pattern_type.loc139_4 => constants.%pattern_type.8b45
// CHECK:STDOUT:   %x.param_patt.loc139_4.2 => constants.%x.param_patt.27c
// CHECK:STDOUT:   %x.patt.loc139_4.2 => constants.%x.patt.4c9
// CHECK:STDOUT:   %pattern_type.loc139_10 => constants.%pattern_type.785
// CHECK:STDOUT:   %y.param_patt.loc139_10.2 => constants.%y.param_patt.418
// CHECK:STDOUT:   %y.patt.loc139_10.2 => constants.%y.patt.4e6
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc139_4 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc139_10 => constants.%complete_type.357
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.7f6
// CHECK:STDOUT:   %LeftShiftWith.lookup_impl_witness.loc141 => constants.%custom_witness.672
// CHECK:STDOUT:   %LeftShiftWith.facet => constants.%LeftShiftWith.facet.453
// CHECK:STDOUT:   %LeftShiftWith.WithSelf.Op.type => constants.%LeftShiftWith.WithSelf.Op.type.1bb
// CHECK:STDOUT:   %.loc141_5.6 => constants.%.359
// CHECK:STDOUT:   %impl.elem1.loc141_5.2 => constants.%Int32.Op.4dca31.5
// CHECK:STDOUT:   %specific_impl_fn.loc141_5.3 => constants.%Int32.Op.4dca31.5
// CHECK:STDOUT:   %impl.elem0.1 => constants.%Int32
// CHECK:STDOUT:   %require_complete.1 => constants.%complete_type.357
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.e7a31c.2
// CHECK:STDOUT:   %Destroy.facet.loc141_5.3 => constants.%Destroy.facet.099
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.b76
// CHECK:STDOUT:   %.loc141_5.7 => constants.%.043
// CHECK:STDOUT:   %impl.elem0.loc141_5.2 => constants.%Int32.Op.4dca31.1
// CHECK:STDOUT:   %specific_impl_fn.loc141_5.4 => constants.%Int32.Op.4dca31.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @RightShiftWith.loc148(constants.%Int16, constants.%facet_value.65f) {
// CHECK:STDOUT:   %U.patt.loc146_4.2 => constants.%U.patt.d47
// CHECK:STDOUT:   %U.loc146_4.1 => constants.%Int16
// CHECK:STDOUT:   %RightShiftWith.type.loc147_28.1 => constants.%RightShiftWith.type.386
// CHECK:STDOUT:   %.Self.loc147_30.1 => constants.%.Self.1fe
// CHECK:STDOUT:   %require_complete.loc147 => constants.%complete_type.938
// CHECK:STDOUT:   %.Self.as_type.loc147_36.1 => constants.%.Self.as_type.7d1
// CHECK:STDOUT:   %RightShiftWith.assoc_type => constants.%RightShiftWith.assoc_type.656
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.148
// CHECK:STDOUT:   %RightShiftWith.lookup_impl_witness.loc147 => constants.%RightShiftWith.lookup_impl_witness.1b3
// CHECK:STDOUT:   %impl.elem0.loc147_36.1 => constants.%impl.elem0.baa
// CHECK:STDOUT:   %RightShiftWith_where.type => constants.%RightShiftWith_where.type.386
// CHECK:STDOUT:   %pattern_type.loc147 => constants.%pattern_type.761
// CHECK:STDOUT:   %T.patt.loc147_4.2 => constants.%T.patt.806
// CHECK:STDOUT:   %T.loc147_4.1 => constants.%facet_value.65f
// CHECK:STDOUT:   %T.as_type.loc148_6.1 => constants.%Int32
// CHECK:STDOUT:   %pattern_type.loc148_4 => constants.%pattern_type.8b45
// CHECK:STDOUT:   %x.param_patt.loc148_4.2 => constants.%x.param_patt.27c
// CHECK:STDOUT:   %x.patt.loc148_4.2 => constants.%x.patt.4c9
// CHECK:STDOUT:   %pattern_type.loc148_10 => constants.%pattern_type.785
// CHECK:STDOUT:   %y.param_patt.loc148_10.2 => constants.%y.param_patt.418
// CHECK:STDOUT:   %y.patt.loc148_10.2 => constants.%y.patt.4e6
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc148_4 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc148_10 => constants.%complete_type.357
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.5a2
// CHECK:STDOUT:   %RightShiftWith.lookup_impl_witness.loc150 => constants.%custom_witness.9fa
// CHECK:STDOUT:   %RightShiftWith.facet => constants.%RightShiftWith.facet.714
// CHECK:STDOUT:   %RightShiftWith.WithSelf.Op.type => constants.%RightShiftWith.WithSelf.Op.type.ea7
// CHECK:STDOUT:   %.loc150_5.6 => constants.%.5d9
// CHECK:STDOUT:   %impl.elem1.loc150_5.2 => constants.%Int32.Op.4dca31.6
// CHECK:STDOUT:   %specific_impl_fn.loc150_5.3 => constants.%Int32.Op.4dca31.6
// CHECK:STDOUT:   %impl.elem0.1 => constants.%Int32
// CHECK:STDOUT:   %require_complete.1 => constants.%complete_type.357
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.e7a31c.2
// CHECK:STDOUT:   %Destroy.facet.loc150_5.3 => constants.%Destroy.facet.099
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.b76
// CHECK:STDOUT:   %.loc150_5.7 => constants.%.043
// CHECK:STDOUT:   %impl.elem0.loc150_5.2 => constants.%Int32.Op.4dca31.1
// CHECK:STDOUT:   %specific_impl_fn.loc150_5.4 => constants.%Int32.Op.4dca31.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @BitAndWith.loc112(constants.%Int32, constants.%facet_value.35c) {
// CHECK:STDOUT:   %U.patt.loc110_4.2 => constants.%U.patt.d47
// CHECK:STDOUT:   %U.loc110_4.1 => constants.%Int32
// CHECK:STDOUT:   %BitAndWith.type.loc111_24.1 => constants.%BitAndWith.type.f43
// CHECK:STDOUT:   %.Self.loc111_26.1 => constants.%.Self.b06
// CHECK:STDOUT:   %require_complete.loc111 => constants.%complete_type.008
// CHECK:STDOUT:   %.Self.as_type.loc111_32.1 => constants.%.Self.as_type.b4c
// CHECK:STDOUT:   %BitAndWith.assoc_type => constants.%BitAndWith.assoc_type.6e7
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.543
// CHECK:STDOUT:   %BitAndWith.lookup_impl_witness.loc111 => constants.%BitAndWith.lookup_impl_witness.773
// CHECK:STDOUT:   %impl.elem0.loc111_32.1 => constants.%impl.elem0.b5e
// CHECK:STDOUT:   %BitAndWith_where.type => constants.%BitAndWith_where.type.f43
// CHECK:STDOUT:   %pattern_type.loc111 => constants.%pattern_type.f97
// CHECK:STDOUT:   %T.patt.loc111_4.2 => constants.%T.patt.126
// CHECK:STDOUT:   %T.loc111_4.1 => constants.%facet_value.35c
// CHECK:STDOUT:   %T.as_type.loc112_6.1 => constants.%Int32
// CHECK:STDOUT:   %pattern_type.loc112_4 => constants.%pattern_type.8b45
// CHECK:STDOUT:   %x.param_patt.loc112_4.2 => constants.%x.param_patt.27c
// CHECK:STDOUT:   %x.patt.loc112_4.2 => constants.%x.patt.4c9
// CHECK:STDOUT:   %pattern_type.loc112_10 => constants.%pattern_type.8b45
// CHECK:STDOUT:   %y.param_patt.loc112_10.2 => constants.%y.param_patt.e0c
// CHECK:STDOUT:   %y.patt.loc112_10.2 => constants.%y.patt.0a3
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc112_4 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc112_10 => constants.%complete_type.357
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.5b1
// CHECK:STDOUT:   %BitAndWith.lookup_impl_witness.loc114 => constants.%custom_witness.9a1dfa.2
// CHECK:STDOUT:   %BitAndWith.facet => constants.%BitAndWith.facet.2d7
// CHECK:STDOUT:   %BitAndWith.WithSelf.Op.type => constants.%BitAndWith.WithSelf.Op.type.7e2
// CHECK:STDOUT:   %.loc114_5.6 => constants.%.09e
// CHECK:STDOUT:   %impl.elem1.loc114_5.2 => constants.%Int32.Op.4dca31.7
// CHECK:STDOUT:   %specific_impl_fn.loc114_5.3 => constants.%Int32.Op.4dca31.7
// CHECK:STDOUT:   %impl.elem0.1 => constants.%Int32
// CHECK:STDOUT:   %require_complete.1 => constants.%complete_type.357
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.e7a31c.2
// CHECK:STDOUT:   %Destroy.facet.loc114_5.3 => constants.%Destroy.facet.099
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.b76
// CHECK:STDOUT:   %.loc114_5.7 => constants.%.043
// CHECK:STDOUT:   %impl.elem0.loc114_5.2 => constants.%Int32.Op.4dca31.1
// CHECK:STDOUT:   %specific_impl_fn.loc114_5.4 => constants.%Int32.Op.4dca31.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @BitOrWith.loc121(constants.%Int32, constants.%facet_value.51e) {
// CHECK:STDOUT:   %U.patt.loc119_4.2 => constants.%U.patt.d47
// CHECK:STDOUT:   %U.loc119_4.1 => constants.%Int32
// CHECK:STDOUT:   %BitOrWith.type.loc120_23.1 => constants.%BitOrWith.type.878
// CHECK:STDOUT:   %.Self.loc120_25.1 => constants.%.Self.f1c
// CHECK:STDOUT:   %require_complete.loc120 => constants.%complete_type.4fe
// CHECK:STDOUT:   %.Self.as_type.loc120_31.1 => constants.%.Self.as_type.4f4
// CHECK:STDOUT:   %BitOrWith.assoc_type => constants.%BitOrWith.assoc_type.0490
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.d36
// CHECK:STDOUT:   %BitOrWith.lookup_impl_witness.loc120 => constants.%BitOrWith.lookup_impl_witness.93b
// CHECK:STDOUT:   %impl.elem0.loc120_31.1 => constants.%impl.elem0.181
// CHECK:STDOUT:   %BitOrWith_where.type => constants.%BitOrWith_where.type.878
// CHECK:STDOUT:   %pattern_type.loc120 => constants.%pattern_type.a2c
// CHECK:STDOUT:   %T.patt.loc120_4.2 => constants.%T.patt.3c3
// CHECK:STDOUT:   %T.loc120_4.1 => constants.%facet_value.51e
// CHECK:STDOUT:   %T.as_type.loc121_6.1 => constants.%Int32
// CHECK:STDOUT:   %pattern_type.loc121_4 => constants.%pattern_type.8b45
// CHECK:STDOUT:   %x.param_patt.loc121_4.2 => constants.%x.param_patt.27c
// CHECK:STDOUT:   %x.patt.loc121_4.2 => constants.%x.patt.4c9
// CHECK:STDOUT:   %pattern_type.loc121_10 => constants.%pattern_type.8b45
// CHECK:STDOUT:   %y.param_patt.loc121_10.2 => constants.%y.param_patt.e0c
// CHECK:STDOUT:   %y.patt.loc121_10.2 => constants.%y.patt.0a3
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc121_4 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc121_10 => constants.%complete_type.357
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.953
// CHECK:STDOUT:   %BitOrWith.lookup_impl_witness.loc123 => constants.%custom_witness.2e067a.2
// CHECK:STDOUT:   %BitOrWith.facet => constants.%BitOrWith.facet.f97
// CHECK:STDOUT:   %BitOrWith.WithSelf.Op.type => constants.%BitOrWith.WithSelf.Op.type.172
// CHECK:STDOUT:   %.loc123_5.6 => constants.%.62c
// CHECK:STDOUT:   %impl.elem1.loc123_5.2 => constants.%Int32.Op.4dca31.8
// CHECK:STDOUT:   %specific_impl_fn.loc123_5.3 => constants.%Int32.Op.4dca31.8
// CHECK:STDOUT:   %impl.elem0.1 => constants.%Int32
// CHECK:STDOUT:   %require_complete.1 => constants.%complete_type.357
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.e7a31c.2
// CHECK:STDOUT:   %Destroy.facet.loc123_5.3 => constants.%Destroy.facet.099
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.b76
// CHECK:STDOUT:   %.loc123_5.7 => constants.%.043
// CHECK:STDOUT:   %impl.elem0.loc123_5.2 => constants.%Int32.Op.4dca31.1
// CHECK:STDOUT:   %specific_impl_fn.loc123_5.4 => constants.%Int32.Op.4dca31.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @BitXorWith.loc130(constants.%Int32, constants.%facet_value.bdf) {
// CHECK:STDOUT:   %U.patt.loc128_4.2 => constants.%U.patt.d47
// CHECK:STDOUT:   %U.loc128_4.1 => constants.%Int32
// CHECK:STDOUT:   %BitXorWith.type.loc129_24.1 => constants.%BitXorWith.type.729
// CHECK:STDOUT:   %.Self.loc129_26.1 => constants.%.Self.be9
// CHECK:STDOUT:   %require_complete.loc129 => constants.%complete_type.8eb
// CHECK:STDOUT:   %.Self.as_type.loc129_32.1 => constants.%.Self.as_type.192
// CHECK:STDOUT:   %BitXorWith.assoc_type => constants.%BitXorWith.assoc_type.36a
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.0ab
// CHECK:STDOUT:   %BitXorWith.lookup_impl_witness.loc129 => constants.%BitXorWith.lookup_impl_witness.de4
// CHECK:STDOUT:   %impl.elem0.loc129_32.1 => constants.%impl.elem0.1ab
// CHECK:STDOUT:   %BitXorWith_where.type => constants.%BitXorWith_where.type.729
// CHECK:STDOUT:   %pattern_type.loc129 => constants.%pattern_type.ab4
// CHECK:STDOUT:   %T.patt.loc129_4.2 => constants.%T.patt.1c4
// CHECK:STDOUT:   %T.loc129_4.1 => constants.%facet_value.bdf
// CHECK:STDOUT:   %T.as_type.loc130_6.1 => constants.%Int32
// CHECK:STDOUT:   %pattern_type.loc130_4 => constants.%pattern_type.8b45
// CHECK:STDOUT:   %x.param_patt.loc130_4.2 => constants.%x.param_patt.27c
// CHECK:STDOUT:   %x.patt.loc130_4.2 => constants.%x.patt.4c9
// CHECK:STDOUT:   %pattern_type.loc130_10 => constants.%pattern_type.8b45
// CHECK:STDOUT:   %y.param_patt.loc130_10.2 => constants.%y.param_patt.e0c
// CHECK:STDOUT:   %y.patt.loc130_10.2 => constants.%y.patt.0a3
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc130_4 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc130_10 => constants.%complete_type.357
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.951
// CHECK:STDOUT:   %BitXorWith.lookup_impl_witness.loc132 => constants.%custom_witness.5d016d.2
// CHECK:STDOUT:   %BitXorWith.facet => constants.%BitXorWith.facet.4e5
// CHECK:STDOUT:   %BitXorWith.WithSelf.Op.type => constants.%BitXorWith.WithSelf.Op.type.0df
// CHECK:STDOUT:   %.loc132_5.6 => constants.%.6aa
// CHECK:STDOUT:   %impl.elem1.loc132_5.2 => constants.%Int32.Op.4dca31.9
// CHECK:STDOUT:   %specific_impl_fn.loc132_5.3 => constants.%Int32.Op.4dca31.9
// CHECK:STDOUT:   %impl.elem0.1 => constants.%Int32
// CHECK:STDOUT:   %require_complete.1 => constants.%complete_type.357
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.e7a31c.2
// CHECK:STDOUT:   %Destroy.facet.loc132_5.3 => constants.%Destroy.facet.099
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.b76
// CHECK:STDOUT:   %.loc132_5.7 => constants.%.043
// CHECK:STDOUT:   %impl.elem0.loc132_5.2 => constants.%Int32.Op.4dca31.1
// CHECK:STDOUT:   %specific_impl_fn.loc132_5.4 => constants.%Int32.Op.4dca31.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @LeftShiftWith.loc139(constants.%Int32, constants.%facet_value.dcd) {
// CHECK:STDOUT:   %U.patt.loc137_4.2 => constants.%U.patt.d47
// CHECK:STDOUT:   %U.loc137_4.1 => constants.%Int32
// CHECK:STDOUT:   %LeftShiftWith.type.loc138_27.1 => constants.%LeftShiftWith.type.15a
// CHECK:STDOUT:   %.Self.loc138_29.1 => constants.%.Self.b90
// CHECK:STDOUT:   %require_complete.loc138 => constants.%complete_type.5b7
// CHECK:STDOUT:   %.Self.as_type.loc138_35.1 => constants.%.Self.as_type.b53
// CHECK:STDOUT:   %LeftShiftWith.assoc_type => constants.%LeftShiftWith.assoc_type.e1d
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.6e4
// CHECK:STDOUT:   %LeftShiftWith.lookup_impl_witness.loc138 => constants.%LeftShiftWith.lookup_impl_witness.140
// CHECK:STDOUT:   %impl.elem0.loc138_35.1 => constants.%impl.elem0.e18
// CHECK:STDOUT:   %LeftShiftWith_where.type => constants.%LeftShiftWith_where.type.15a
// CHECK:STDOUT:   %pattern_type.loc138 => constants.%pattern_type.538
// CHECK:STDOUT:   %T.patt.loc138_4.2 => constants.%T.patt.4a0
// CHECK:STDOUT:   %T.loc138_4.1 => constants.%facet_value.dcd
// CHECK:STDOUT:   %T.as_type.loc139_6.1 => constants.%Int32
// CHECK:STDOUT:   %pattern_type.loc139_4 => constants.%pattern_type.8b45
// CHECK:STDOUT:   %x.param_patt.loc139_4.2 => constants.%x.param_patt.27c
// CHECK:STDOUT:   %x.patt.loc139_4.2 => constants.%x.patt.4c9
// CHECK:STDOUT:   %pattern_type.loc139_10 => constants.%pattern_type.8b45
// CHECK:STDOUT:   %y.param_patt.loc139_10.2 => constants.%y.param_patt.e0c
// CHECK:STDOUT:   %y.patt.loc139_10.2 => constants.%y.patt.0a3
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc139_4 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc139_10 => constants.%complete_type.357
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.f2a
// CHECK:STDOUT:   %LeftShiftWith.lookup_impl_witness.loc141 => constants.%custom_witness.65d217.2
// CHECK:STDOUT:   %LeftShiftWith.facet => constants.%LeftShiftWith.facet.0bb
// CHECK:STDOUT:   %LeftShiftWith.WithSelf.Op.type => constants.%LeftShiftWith.WithSelf.Op.type.2be
// CHECK:STDOUT:   %.loc141_5.6 => constants.%.de5
// CHECK:STDOUT:   %impl.elem1.loc141_5.2 => constants.%Int32.Op.4dca31.10
// CHECK:STDOUT:   %specific_impl_fn.loc141_5.3 => constants.%Int32.Op.4dca31.10
// CHECK:STDOUT:   %impl.elem0.1 => constants.%Int32
// CHECK:STDOUT:   %require_complete.1 => constants.%complete_type.357
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.e7a31c.2
// CHECK:STDOUT:   %Destroy.facet.loc141_5.3 => constants.%Destroy.facet.099
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.b76
// CHECK:STDOUT:   %.loc141_5.7 => constants.%.043
// CHECK:STDOUT:   %impl.elem0.loc141_5.2 => constants.%Int32.Op.4dca31.1
// CHECK:STDOUT:   %specific_impl_fn.loc141_5.4 => constants.%Int32.Op.4dca31.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @RightShiftWith.loc148(constants.%Int32, constants.%facet_value.624) {
// CHECK:STDOUT:   %U.patt.loc146_4.2 => constants.%U.patt.d47
// CHECK:STDOUT:   %U.loc146_4.1 => constants.%Int32
// CHECK:STDOUT:   %RightShiftWith.type.loc147_28.1 => constants.%RightShiftWith.type.8f0
// CHECK:STDOUT:   %.Self.loc147_30.1 => constants.%.Self.c91
// CHECK:STDOUT:   %require_complete.loc147 => constants.%complete_type.93f
// CHECK:STDOUT:   %.Self.as_type.loc147_36.1 => constants.%.Self.as_type.d6f
// CHECK:STDOUT:   %RightShiftWith.assoc_type => constants.%RightShiftWith.assoc_type.813
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.99a
// CHECK:STDOUT:   %RightShiftWith.lookup_impl_witness.loc147 => constants.%RightShiftWith.lookup_impl_witness.bb0
// CHECK:STDOUT:   %impl.elem0.loc147_36.1 => constants.%impl.elem0.cfe
// CHECK:STDOUT:   %RightShiftWith_where.type => constants.%RightShiftWith_where.type.8f0
// CHECK:STDOUT:   %pattern_type.loc147 => constants.%pattern_type.5a0
// CHECK:STDOUT:   %T.patt.loc147_4.2 => constants.%T.patt.a69
// CHECK:STDOUT:   %T.loc147_4.1 => constants.%facet_value.624
// CHECK:STDOUT:   %T.as_type.loc148_6.1 => constants.%Int32
// CHECK:STDOUT:   %pattern_type.loc148_4 => constants.%pattern_type.8b45
// CHECK:STDOUT:   %x.param_patt.loc148_4.2 => constants.%x.param_patt.27c
// CHECK:STDOUT:   %x.patt.loc148_4.2 => constants.%x.patt.4c9
// CHECK:STDOUT:   %pattern_type.loc148_10 => constants.%pattern_type.8b45
// CHECK:STDOUT:   %y.param_patt.loc148_10.2 => constants.%y.param_patt.e0c
// CHECK:STDOUT:   %y.patt.loc148_10.2 => constants.%y.patt.0a3
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc148_4 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc148_10 => constants.%complete_type.357
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.2c8
// CHECK:STDOUT:   %RightShiftWith.lookup_impl_witness.loc150 => constants.%custom_witness.884138.2
// CHECK:STDOUT:   %RightShiftWith.facet => constants.%RightShiftWith.facet.0bb
// CHECK:STDOUT:   %RightShiftWith.WithSelf.Op.type => constants.%RightShiftWith.WithSelf.Op.type.43d
// CHECK:STDOUT:   %.loc150_5.6 => constants.%.ae2
// CHECK:STDOUT:   %impl.elem1.loc150_5.2 => constants.%Int32.Op.4dca31.11
// CHECK:STDOUT:   %specific_impl_fn.loc150_5.3 => constants.%Int32.Op.4dca31.11
// CHECK:STDOUT:   %impl.elem0.1 => constants.%Int32
// CHECK:STDOUT:   %require_complete.1 => constants.%complete_type.357
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.e7a31c.2
// CHECK:STDOUT:   %Destroy.facet.loc150_5.3 => constants.%Destroy.facet.099
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.b76
// CHECK:STDOUT:   %.loc150_5.7 => constants.%.043
// CHECK:STDOUT:   %impl.elem0.loc150_5.2 => constants.%Int32.Op.4dca31.1
// CHECK:STDOUT:   %specific_impl_fn.loc150_5.4 => constants.%Int32.Op.4dca31.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @BitAndWith.loc112(constants.%Int64, constants.%facet_value.88d) {
// CHECK:STDOUT:   %U.patt.loc110_4.2 => constants.%U.patt.d47
// CHECK:STDOUT:   %U.loc110_4.1 => constants.%Int64
// CHECK:STDOUT:   %BitAndWith.type.loc111_24.1 => constants.%BitAndWith.type.0cc
// CHECK:STDOUT:   %.Self.loc111_26.1 => constants.%.Self.bfc
// CHECK:STDOUT:   %require_complete.loc111 => constants.%complete_type.3d2
// CHECK:STDOUT:   %.Self.as_type.loc111_32.1 => constants.%.Self.as_type.75e6
// CHECK:STDOUT:   %BitAndWith.assoc_type => constants.%BitAndWith.assoc_type.34a
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.be9
// CHECK:STDOUT:   %BitAndWith.lookup_impl_witness.loc111 => constants.%BitAndWith.lookup_impl_witness.bf6
// CHECK:STDOUT:   %impl.elem0.loc111_32.1 => constants.%impl.elem0.cbe
// CHECK:STDOUT:   %BitAndWith_where.type => constants.%BitAndWith_where.type.0cc
// CHECK:STDOUT:   %pattern_type.loc111 => constants.%pattern_type.0d8
// CHECK:STDOUT:   %T.patt.loc111_4.2 => constants.%T.patt.e0d
// CHECK:STDOUT:   %T.loc111_4.1 => constants.%facet_value.88d
// CHECK:STDOUT:   %T.as_type.loc112_6.1 => constants.%Int32
// CHECK:STDOUT:   %pattern_type.loc112_4 => constants.%pattern_type.8b45
// CHECK:STDOUT:   %x.param_patt.loc112_4.2 => constants.%x.param_patt.27c
// CHECK:STDOUT:   %x.patt.loc112_4.2 => constants.%x.patt.4c9
// CHECK:STDOUT:   %pattern_type.loc112_10 => constants.%pattern_type.4aa
// CHECK:STDOUT:   %y.param_patt.loc112_10.2 => constants.%y.param_patt.a08
// CHECK:STDOUT:   %y.patt.loc112_10.2 => constants.%y.patt.63b
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc112_4 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc112_10 => constants.%complete_type.357
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.e93
// CHECK:STDOUT:   %BitAndWith.lookup_impl_witness.loc114 => constants.%custom_witness.946
// CHECK:STDOUT:   %BitAndWith.facet => constants.%BitAndWith.facet.401
// CHECK:STDOUT:   %BitAndWith.WithSelf.Op.type => constants.%BitAndWith.WithSelf.Op.type.e00
// CHECK:STDOUT:   %.loc114_5.6 => constants.%.0ba
// CHECK:STDOUT:   %impl.elem1.loc114_5.2 => constants.%Op.72717c.6
// CHECK:STDOUT:   %specific_impl_fn.loc114_5.3 => constants.%Op.72717c.6
// CHECK:STDOUT:   %impl.elem0.1 => constants.%Int64
// CHECK:STDOUT:   %require_complete.1 => constants.%complete_type.357
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.e7a31c.3
// CHECK:STDOUT:   %Destroy.facet.loc114_5.3 => constants.%Destroy.facet.191
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.c2c
// CHECK:STDOUT:   %.loc114_5.7 => constants.%.c2e
// CHECK:STDOUT:   %impl.elem0.loc114_5.2 => constants.%Int64.Op.4dca31.1
// CHECK:STDOUT:   %specific_impl_fn.loc114_5.4 => constants.%Int64.Op.4dca31.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @BitOrWith.loc121(constants.%Int64, constants.%facet_value.d8d) {
// CHECK:STDOUT:   %U.patt.loc119_4.2 => constants.%U.patt.d47
// CHECK:STDOUT:   %U.loc119_4.1 => constants.%Int64
// CHECK:STDOUT:   %BitOrWith.type.loc120_23.1 => constants.%BitOrWith.type.1a2
// CHECK:STDOUT:   %.Self.loc120_25.1 => constants.%.Self.4a6
// CHECK:STDOUT:   %require_complete.loc120 => constants.%complete_type.33f
// CHECK:STDOUT:   %.Self.as_type.loc120_31.1 => constants.%.Self.as_type.d1f
// CHECK:STDOUT:   %BitOrWith.assoc_type => constants.%BitOrWith.assoc_type.ccc
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.f1a
// CHECK:STDOUT:   %BitOrWith.lookup_impl_witness.loc120 => constants.%BitOrWith.lookup_impl_witness.6d7
// CHECK:STDOUT:   %impl.elem0.loc120_31.1 => constants.%impl.elem0.7df
// CHECK:STDOUT:   %BitOrWith_where.type => constants.%BitOrWith_where.type.1a2
// CHECK:STDOUT:   %pattern_type.loc120 => constants.%pattern_type.f3b
// CHECK:STDOUT:   %T.patt.loc120_4.2 => constants.%T.patt.880
// CHECK:STDOUT:   %T.loc120_4.1 => constants.%facet_value.d8d
// CHECK:STDOUT:   %T.as_type.loc121_6.1 => constants.%Int32
// CHECK:STDOUT:   %pattern_type.loc121_4 => constants.%pattern_type.8b45
// CHECK:STDOUT:   %x.param_patt.loc121_4.2 => constants.%x.param_patt.27c
// CHECK:STDOUT:   %x.patt.loc121_4.2 => constants.%x.patt.4c9
// CHECK:STDOUT:   %pattern_type.loc121_10 => constants.%pattern_type.4aa
// CHECK:STDOUT:   %y.param_patt.loc121_10.2 => constants.%y.param_patt.a08
// CHECK:STDOUT:   %y.patt.loc121_10.2 => constants.%y.patt.63b
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc121_4 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc121_10 => constants.%complete_type.357
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.fe7
// CHECK:STDOUT:   %BitOrWith.lookup_impl_witness.loc123 => constants.%custom_witness.d82
// CHECK:STDOUT:   %BitOrWith.facet => constants.%BitOrWith.facet.7a3
// CHECK:STDOUT:   %BitOrWith.WithSelf.Op.type => constants.%BitOrWith.WithSelf.Op.type.d10
// CHECK:STDOUT:   %.loc123_5.6 => constants.%.a76
// CHECK:STDOUT:   %impl.elem1.loc123_5.2 => constants.%Op.72717c.7
// CHECK:STDOUT:   %specific_impl_fn.loc123_5.3 => constants.%Op.72717c.7
// CHECK:STDOUT:   %impl.elem0.1 => constants.%Int64
// CHECK:STDOUT:   %require_complete.1 => constants.%complete_type.357
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.e7a31c.3
// CHECK:STDOUT:   %Destroy.facet.loc123_5.3 => constants.%Destroy.facet.191
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.c2c
// CHECK:STDOUT:   %.loc123_5.7 => constants.%.c2e
// CHECK:STDOUT:   %impl.elem0.loc123_5.2 => constants.%Int64.Op.4dca31.1
// CHECK:STDOUT:   %specific_impl_fn.loc123_5.4 => constants.%Int64.Op.4dca31.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @BitXorWith.loc130(constants.%Int64, constants.%facet_value.459) {
// CHECK:STDOUT:   %U.patt.loc128_4.2 => constants.%U.patt.d47
// CHECK:STDOUT:   %U.loc128_4.1 => constants.%Int64
// CHECK:STDOUT:   %BitXorWith.type.loc129_24.1 => constants.%BitXorWith.type.a90
// CHECK:STDOUT:   %.Self.loc129_26.1 => constants.%.Self.32b
// CHECK:STDOUT:   %require_complete.loc129 => constants.%complete_type.b22
// CHECK:STDOUT:   %.Self.as_type.loc129_32.1 => constants.%.Self.as_type.94f
// CHECK:STDOUT:   %BitXorWith.assoc_type => constants.%BitXorWith.assoc_type.8b9
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.736
// CHECK:STDOUT:   %BitXorWith.lookup_impl_witness.loc129 => constants.%BitXorWith.lookup_impl_witness.c28
// CHECK:STDOUT:   %impl.elem0.loc129_32.1 => constants.%impl.elem0.361
// CHECK:STDOUT:   %BitXorWith_where.type => constants.%BitXorWith_where.type.a90
// CHECK:STDOUT:   %pattern_type.loc129 => constants.%pattern_type.481
// CHECK:STDOUT:   %T.patt.loc129_4.2 => constants.%T.patt.304
// CHECK:STDOUT:   %T.loc129_4.1 => constants.%facet_value.459
// CHECK:STDOUT:   %T.as_type.loc130_6.1 => constants.%Int32
// CHECK:STDOUT:   %pattern_type.loc130_4 => constants.%pattern_type.8b45
// CHECK:STDOUT:   %x.param_patt.loc130_4.2 => constants.%x.param_patt.27c
// CHECK:STDOUT:   %x.patt.loc130_4.2 => constants.%x.patt.4c9
// CHECK:STDOUT:   %pattern_type.loc130_10 => constants.%pattern_type.4aa
// CHECK:STDOUT:   %y.param_patt.loc130_10.2 => constants.%y.param_patt.a08
// CHECK:STDOUT:   %y.patt.loc130_10.2 => constants.%y.patt.63b
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc130_4 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc130_10 => constants.%complete_type.357
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.957
// CHECK:STDOUT:   %BitXorWith.lookup_impl_witness.loc132 => constants.%custom_witness.888
// CHECK:STDOUT:   %BitXorWith.facet => constants.%BitXorWith.facet.073
// CHECK:STDOUT:   %BitXorWith.WithSelf.Op.type => constants.%BitXorWith.WithSelf.Op.type.946
// CHECK:STDOUT:   %.loc132_5.6 => constants.%.82d
// CHECK:STDOUT:   %impl.elem1.loc132_5.2 => constants.%Op.72717c.8
// CHECK:STDOUT:   %specific_impl_fn.loc132_5.3 => constants.%Op.72717c.8
// CHECK:STDOUT:   %impl.elem0.1 => constants.%Int64
// CHECK:STDOUT:   %require_complete.1 => constants.%complete_type.357
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.e7a31c.3
// CHECK:STDOUT:   %Destroy.facet.loc132_5.3 => constants.%Destroy.facet.191
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.c2c
// CHECK:STDOUT:   %.loc132_5.7 => constants.%.c2e
// CHECK:STDOUT:   %impl.elem0.loc132_5.2 => constants.%Int64.Op.4dca31.1
// CHECK:STDOUT:   %specific_impl_fn.loc132_5.4 => constants.%Int64.Op.4dca31.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @LeftShiftWith.loc139(constants.%Int64, constants.%facet_value.ef8) {
// CHECK:STDOUT:   %U.patt.loc137_4.2 => constants.%U.patt.d47
// CHECK:STDOUT:   %U.loc137_4.1 => constants.%Int64
// CHECK:STDOUT:   %LeftShiftWith.type.loc138_27.1 => constants.%LeftShiftWith.type.d4a
// CHECK:STDOUT:   %.Self.loc138_29.1 => constants.%.Self.21d
// CHECK:STDOUT:   %require_complete.loc138 => constants.%complete_type.5f3
// CHECK:STDOUT:   %.Self.as_type.loc138_35.1 => constants.%.Self.as_type.3c7
// CHECK:STDOUT:   %LeftShiftWith.assoc_type => constants.%LeftShiftWith.assoc_type.b9e
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.5b7
// CHECK:STDOUT:   %LeftShiftWith.lookup_impl_witness.loc138 => constants.%LeftShiftWith.lookup_impl_witness.fd4
// CHECK:STDOUT:   %impl.elem0.loc138_35.1 => constants.%impl.elem0.f0c
// CHECK:STDOUT:   %LeftShiftWith_where.type => constants.%LeftShiftWith_where.type.d4a
// CHECK:STDOUT:   %pattern_type.loc138 => constants.%pattern_type.e6d
// CHECK:STDOUT:   %T.patt.loc138_4.2 => constants.%T.patt.795
// CHECK:STDOUT:   %T.loc138_4.1 => constants.%facet_value.ef8
// CHECK:STDOUT:   %T.as_type.loc139_6.1 => constants.%Int32
// CHECK:STDOUT:   %pattern_type.loc139_4 => constants.%pattern_type.8b45
// CHECK:STDOUT:   %x.param_patt.loc139_4.2 => constants.%x.param_patt.27c
// CHECK:STDOUT:   %x.patt.loc139_4.2 => constants.%x.patt.4c9
// CHECK:STDOUT:   %pattern_type.loc139_10 => constants.%pattern_type.4aa
// CHECK:STDOUT:   %y.param_patt.loc139_10.2 => constants.%y.param_patt.a08
// CHECK:STDOUT:   %y.patt.loc139_10.2 => constants.%y.patt.63b
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc139_4 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc139_10 => constants.%complete_type.357
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.7dc
// CHECK:STDOUT:   %LeftShiftWith.lookup_impl_witness.loc141 => constants.%custom_witness.cdd
// CHECK:STDOUT:   %LeftShiftWith.facet => constants.%LeftShiftWith.facet.3c0
// CHECK:STDOUT:   %LeftShiftWith.WithSelf.Op.type => constants.%LeftShiftWith.WithSelf.Op.type.56e
// CHECK:STDOUT:   %.loc141_5.6 => constants.%.f1e
// CHECK:STDOUT:   %impl.elem1.loc141_5.2 => constants.%Op.72717c.9
// CHECK:STDOUT:   %specific_impl_fn.loc141_5.3 => constants.%Op.72717c.9
// CHECK:STDOUT:   %impl.elem0.1 => constants.%Int64
// CHECK:STDOUT:   %require_complete.1 => constants.%complete_type.357
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.e7a31c.3
// CHECK:STDOUT:   %Destroy.facet.loc141_5.3 => constants.%Destroy.facet.191
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.c2c
// CHECK:STDOUT:   %.loc141_5.7 => constants.%.c2e
// CHECK:STDOUT:   %impl.elem0.loc141_5.2 => constants.%Int64.Op.4dca31.1
// CHECK:STDOUT:   %specific_impl_fn.loc141_5.4 => constants.%Int64.Op.4dca31.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @RightShiftWith.loc148(constants.%Int64, constants.%facet_value.b0f) {
// CHECK:STDOUT:   %U.patt.loc146_4.2 => constants.%U.patt.d47
// CHECK:STDOUT:   %U.loc146_4.1 => constants.%Int64
// CHECK:STDOUT:   %RightShiftWith.type.loc147_28.1 => constants.%RightShiftWith.type.39e
// CHECK:STDOUT:   %.Self.loc147_30.1 => constants.%.Self.ac8
// CHECK:STDOUT:   %require_complete.loc147 => constants.%complete_type.7de
// CHECK:STDOUT:   %.Self.as_type.loc147_36.1 => constants.%.Self.as_type.915
// CHECK:STDOUT:   %RightShiftWith.assoc_type => constants.%RightShiftWith.assoc_type.56a
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.c11
// CHECK:STDOUT:   %RightShiftWith.lookup_impl_witness.loc147 => constants.%RightShiftWith.lookup_impl_witness.531
// CHECK:STDOUT:   %impl.elem0.loc147_36.1 => constants.%impl.elem0.4cb
// CHECK:STDOUT:   %RightShiftWith_where.type => constants.%RightShiftWith_where.type.39e
// CHECK:STDOUT:   %pattern_type.loc147 => constants.%pattern_type.a3d
// CHECK:STDOUT:   %T.patt.loc147_4.2 => constants.%T.patt.4f5
// CHECK:STDOUT:   %T.loc147_4.1 => constants.%facet_value.b0f
// CHECK:STDOUT:   %T.as_type.loc148_6.1 => constants.%Int32
// CHECK:STDOUT:   %pattern_type.loc148_4 => constants.%pattern_type.8b45
// CHECK:STDOUT:   %x.param_patt.loc148_4.2 => constants.%x.param_patt.27c
// CHECK:STDOUT:   %x.patt.loc148_4.2 => constants.%x.patt.4c9
// CHECK:STDOUT:   %pattern_type.loc148_10 => constants.%pattern_type.4aa
// CHECK:STDOUT:   %y.param_patt.loc148_10.2 => constants.%y.param_patt.a08
// CHECK:STDOUT:   %y.patt.loc148_10.2 => constants.%y.patt.63b
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc148_4 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc148_10 => constants.%complete_type.357
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.c9e
// CHECK:STDOUT:   %RightShiftWith.lookup_impl_witness.loc150 => constants.%custom_witness.3bf
// CHECK:STDOUT:   %RightShiftWith.facet => constants.%RightShiftWith.facet.e89
// CHECK:STDOUT:   %RightShiftWith.WithSelf.Op.type => constants.%RightShiftWith.WithSelf.Op.type.719
// CHECK:STDOUT:   %.loc150_5.6 => constants.%.7e1d
// CHECK:STDOUT:   %impl.elem1.loc150_5.2 => constants.%Op.72717c.10
// CHECK:STDOUT:   %specific_impl_fn.loc150_5.3 => constants.%Op.72717c.10
// CHECK:STDOUT:   %impl.elem0.1 => constants.%Int64
// CHECK:STDOUT:   %require_complete.1 => constants.%complete_type.357
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.e7a31c.3
// CHECK:STDOUT:   %Destroy.facet.loc150_5.3 => constants.%Destroy.facet.191
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.c2c
// CHECK:STDOUT:   %.loc150_5.7 => constants.%.c2e
// CHECK:STDOUT:   %impl.elem0.loc150_5.2 => constants.%Int64.Op.4dca31.1
// CHECK:STDOUT:   %specific_impl_fn.loc150_5.4 => constants.%Int64.Op.4dca31.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @BitAndWith.loc112(constants.%Int32, constants.%facet_value.797) {
// CHECK:STDOUT:   %U.patt.loc110_4.2 => constants.%U.patt.d47
// CHECK:STDOUT:   %U.loc110_4.1 => constants.%Int32
// CHECK:STDOUT:   %BitAndWith.type.loc111_24.1 => constants.%BitAndWith.type.f43
// CHECK:STDOUT:   %.Self.loc111_26.1 => constants.%.Self.b06
// CHECK:STDOUT:   %require_complete.loc111 => constants.%complete_type.008
// CHECK:STDOUT:   %.Self.as_type.loc111_32.1 => constants.%.Self.as_type.b4c
// CHECK:STDOUT:   %BitAndWith.assoc_type => constants.%BitAndWith.assoc_type.6e7
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.543
// CHECK:STDOUT:   %BitAndWith.lookup_impl_witness.loc111 => constants.%BitAndWith.lookup_impl_witness.773
// CHECK:STDOUT:   %impl.elem0.loc111_32.1 => constants.%impl.elem0.b5e
// CHECK:STDOUT:   %BitAndWith_where.type => constants.%BitAndWith_where.type.f43
// CHECK:STDOUT:   %pattern_type.loc111 => constants.%pattern_type.f97
// CHECK:STDOUT:   %T.patt.loc111_4.2 => constants.%T.patt.126
// CHECK:STDOUT:   %T.loc111_4.1 => constants.%facet_value.797
// CHECK:STDOUT:   %T.as_type.loc112_6.1 => constants.%Int64
// CHECK:STDOUT:   %pattern_type.loc112_4 => constants.%pattern_type.4aa
// CHECK:STDOUT:   %x.param_patt.loc112_4.2 => constants.%x.param_patt.e68
// CHECK:STDOUT:   %x.patt.loc112_4.2 => constants.%x.patt.52a
// CHECK:STDOUT:   %pattern_type.loc112_10 => constants.%pattern_type.8b45
// CHECK:STDOUT:   %y.param_patt.loc112_10.2 => constants.%y.param_patt.e0c
// CHECK:STDOUT:   %y.patt.loc112_10.2 => constants.%y.patt.0a3
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc112_4 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc112_10 => constants.%complete_type.357
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.5b1
// CHECK:STDOUT:   %BitAndWith.lookup_impl_witness.loc114 => constants.%custom_witness.20d
// CHECK:STDOUT:   %BitAndWith.facet => constants.%BitAndWith.facet.7b2
// CHECK:STDOUT:   %BitAndWith.WithSelf.Op.type => constants.%BitAndWith.WithSelf.Op.type.953
// CHECK:STDOUT:   %.loc114_5.6 => constants.%.b2a
// CHECK:STDOUT:   %impl.elem1.loc114_5.2 => constants.%Op.72717c.11
// CHECK:STDOUT:   %specific_impl_fn.loc114_5.3 => constants.%Op.72717c.11
// CHECK:STDOUT:   %impl.elem0.1 => constants.%Int64
// CHECK:STDOUT:   %require_complete.1 => constants.%complete_type.357
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.e7a31c.3
// CHECK:STDOUT:   %Destroy.facet.loc114_5.3 => constants.%Destroy.facet.191
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.c2c
// CHECK:STDOUT:   %.loc114_5.7 => constants.%.c2e
// CHECK:STDOUT:   %impl.elem0.loc114_5.2 => constants.%Int64.Op.4dca31.1
// CHECK:STDOUT:   %specific_impl_fn.loc114_5.4 => constants.%Int64.Op.4dca31.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @BitOrWith.loc121(constants.%Int32, constants.%facet_value.92d) {
// CHECK:STDOUT:   %U.patt.loc119_4.2 => constants.%U.patt.d47
// CHECK:STDOUT:   %U.loc119_4.1 => constants.%Int32
// CHECK:STDOUT:   %BitOrWith.type.loc120_23.1 => constants.%BitOrWith.type.878
// CHECK:STDOUT:   %.Self.loc120_25.1 => constants.%.Self.f1c
// CHECK:STDOUT:   %require_complete.loc120 => constants.%complete_type.4fe
// CHECK:STDOUT:   %.Self.as_type.loc120_31.1 => constants.%.Self.as_type.4f4
// CHECK:STDOUT:   %BitOrWith.assoc_type => constants.%BitOrWith.assoc_type.0490
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.d36
// CHECK:STDOUT:   %BitOrWith.lookup_impl_witness.loc120 => constants.%BitOrWith.lookup_impl_witness.93b
// CHECK:STDOUT:   %impl.elem0.loc120_31.1 => constants.%impl.elem0.181
// CHECK:STDOUT:   %BitOrWith_where.type => constants.%BitOrWith_where.type.878
// CHECK:STDOUT:   %pattern_type.loc120 => constants.%pattern_type.a2c
// CHECK:STDOUT:   %T.patt.loc120_4.2 => constants.%T.patt.3c3
// CHECK:STDOUT:   %T.loc120_4.1 => constants.%facet_value.92d
// CHECK:STDOUT:   %T.as_type.loc121_6.1 => constants.%Int64
// CHECK:STDOUT:   %pattern_type.loc121_4 => constants.%pattern_type.4aa
// CHECK:STDOUT:   %x.param_patt.loc121_4.2 => constants.%x.param_patt.e68
// CHECK:STDOUT:   %x.patt.loc121_4.2 => constants.%x.patt.52a
// CHECK:STDOUT:   %pattern_type.loc121_10 => constants.%pattern_type.8b45
// CHECK:STDOUT:   %y.param_patt.loc121_10.2 => constants.%y.param_patt.e0c
// CHECK:STDOUT:   %y.patt.loc121_10.2 => constants.%y.patt.0a3
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc121_4 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc121_10 => constants.%complete_type.357
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.953
// CHECK:STDOUT:   %BitOrWith.lookup_impl_witness.loc123 => constants.%custom_witness.bbd
// CHECK:STDOUT:   %BitOrWith.facet => constants.%BitOrWith.facet.b65
// CHECK:STDOUT:   %BitOrWith.WithSelf.Op.type => constants.%BitOrWith.WithSelf.Op.type.406
// CHECK:STDOUT:   %.loc123_5.6 => constants.%.549
// CHECK:STDOUT:   %impl.elem1.loc123_5.2 => constants.%Op.72717c.12
// CHECK:STDOUT:   %specific_impl_fn.loc123_5.3 => constants.%Op.72717c.12
// CHECK:STDOUT:   %impl.elem0.1 => constants.%Int64
// CHECK:STDOUT:   %require_complete.1 => constants.%complete_type.357
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.e7a31c.3
// CHECK:STDOUT:   %Destroy.facet.loc123_5.3 => constants.%Destroy.facet.191
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.c2c
// CHECK:STDOUT:   %.loc123_5.7 => constants.%.c2e
// CHECK:STDOUT:   %impl.elem0.loc123_5.2 => constants.%Int64.Op.4dca31.1
// CHECK:STDOUT:   %specific_impl_fn.loc123_5.4 => constants.%Int64.Op.4dca31.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @BitXorWith.loc130(constants.%Int32, constants.%facet_value.0c5) {
// CHECK:STDOUT:   %U.patt.loc128_4.2 => constants.%U.patt.d47
// CHECK:STDOUT:   %U.loc128_4.1 => constants.%Int32
// CHECK:STDOUT:   %BitXorWith.type.loc129_24.1 => constants.%BitXorWith.type.729
// CHECK:STDOUT:   %.Self.loc129_26.1 => constants.%.Self.be9
// CHECK:STDOUT:   %require_complete.loc129 => constants.%complete_type.8eb
// CHECK:STDOUT:   %.Self.as_type.loc129_32.1 => constants.%.Self.as_type.192
// CHECK:STDOUT:   %BitXorWith.assoc_type => constants.%BitXorWith.assoc_type.36a
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.0ab
// CHECK:STDOUT:   %BitXorWith.lookup_impl_witness.loc129 => constants.%BitXorWith.lookup_impl_witness.de4
// CHECK:STDOUT:   %impl.elem0.loc129_32.1 => constants.%impl.elem0.1ab
// CHECK:STDOUT:   %BitXorWith_where.type => constants.%BitXorWith_where.type.729
// CHECK:STDOUT:   %pattern_type.loc129 => constants.%pattern_type.ab4
// CHECK:STDOUT:   %T.patt.loc129_4.2 => constants.%T.patt.1c4
// CHECK:STDOUT:   %T.loc129_4.1 => constants.%facet_value.0c5
// CHECK:STDOUT:   %T.as_type.loc130_6.1 => constants.%Int64
// CHECK:STDOUT:   %pattern_type.loc130_4 => constants.%pattern_type.4aa
// CHECK:STDOUT:   %x.param_patt.loc130_4.2 => constants.%x.param_patt.e68
// CHECK:STDOUT:   %x.patt.loc130_4.2 => constants.%x.patt.52a
// CHECK:STDOUT:   %pattern_type.loc130_10 => constants.%pattern_type.8b45
// CHECK:STDOUT:   %y.param_patt.loc130_10.2 => constants.%y.param_patt.e0c
// CHECK:STDOUT:   %y.patt.loc130_10.2 => constants.%y.patt.0a3
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc130_4 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc130_10 => constants.%complete_type.357
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.951
// CHECK:STDOUT:   %BitXorWith.lookup_impl_witness.loc132 => constants.%custom_witness.8aa
// CHECK:STDOUT:   %BitXorWith.facet => constants.%BitXorWith.facet.4a6
// CHECK:STDOUT:   %BitXorWith.WithSelf.Op.type => constants.%BitXorWith.WithSelf.Op.type.668
// CHECK:STDOUT:   %.loc132_5.6 => constants.%.4ea
// CHECK:STDOUT:   %impl.elem1.loc132_5.2 => constants.%Op.72717c.13
// CHECK:STDOUT:   %specific_impl_fn.loc132_5.3 => constants.%Op.72717c.13
// CHECK:STDOUT:   %impl.elem0.1 => constants.%Int64
// CHECK:STDOUT:   %require_complete.1 => constants.%complete_type.357
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.e7a31c.3
// CHECK:STDOUT:   %Destroy.facet.loc132_5.3 => constants.%Destroy.facet.191
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.c2c
// CHECK:STDOUT:   %.loc132_5.7 => constants.%.c2e
// CHECK:STDOUT:   %impl.elem0.loc132_5.2 => constants.%Int64.Op.4dca31.1
// CHECK:STDOUT:   %specific_impl_fn.loc132_5.4 => constants.%Int64.Op.4dca31.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @LeftShiftWith.loc139(constants.%Int32, constants.%facet_value.608) {
// CHECK:STDOUT:   %U.patt.loc137_4.2 => constants.%U.patt.d47
// CHECK:STDOUT:   %U.loc137_4.1 => constants.%Int32
// CHECK:STDOUT:   %LeftShiftWith.type.loc138_27.1 => constants.%LeftShiftWith.type.15a
// CHECK:STDOUT:   %.Self.loc138_29.1 => constants.%.Self.b90
// CHECK:STDOUT:   %require_complete.loc138 => constants.%complete_type.5b7
// CHECK:STDOUT:   %.Self.as_type.loc138_35.1 => constants.%.Self.as_type.b53
// CHECK:STDOUT:   %LeftShiftWith.assoc_type => constants.%LeftShiftWith.assoc_type.e1d
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.6e4
// CHECK:STDOUT:   %LeftShiftWith.lookup_impl_witness.loc138 => constants.%LeftShiftWith.lookup_impl_witness.140
// CHECK:STDOUT:   %impl.elem0.loc138_35.1 => constants.%impl.elem0.e18
// CHECK:STDOUT:   %LeftShiftWith_where.type => constants.%LeftShiftWith_where.type.15a
// CHECK:STDOUT:   %pattern_type.loc138 => constants.%pattern_type.538
// CHECK:STDOUT:   %T.patt.loc138_4.2 => constants.%T.patt.4a0
// CHECK:STDOUT:   %T.loc138_4.1 => constants.%facet_value.608
// CHECK:STDOUT:   %T.as_type.loc139_6.1 => constants.%Int64
// CHECK:STDOUT:   %pattern_type.loc139_4 => constants.%pattern_type.4aa
// CHECK:STDOUT:   %x.param_patt.loc139_4.2 => constants.%x.param_patt.e68
// CHECK:STDOUT:   %x.patt.loc139_4.2 => constants.%x.patt.52a
// CHECK:STDOUT:   %pattern_type.loc139_10 => constants.%pattern_type.8b45
// CHECK:STDOUT:   %y.param_patt.loc139_10.2 => constants.%y.param_patt.e0c
// CHECK:STDOUT:   %y.patt.loc139_10.2 => constants.%y.patt.0a3
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc139_4 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc139_10 => constants.%complete_type.357
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.f2a
// CHECK:STDOUT:   %LeftShiftWith.lookup_impl_witness.loc141 => constants.%custom_witness.bee
// CHECK:STDOUT:   %LeftShiftWith.facet => constants.%LeftShiftWith.facet.22c
// CHECK:STDOUT:   %LeftShiftWith.WithSelf.Op.type => constants.%LeftShiftWith.WithSelf.Op.type.0a2
// CHECK:STDOUT:   %.loc141_5.6 => constants.%.c1d
// CHECK:STDOUT:   %impl.elem1.loc141_5.2 => constants.%Op.72717c.14
// CHECK:STDOUT:   %specific_impl_fn.loc141_5.3 => constants.%Op.72717c.14
// CHECK:STDOUT:   %impl.elem0.1 => constants.%Int64
// CHECK:STDOUT:   %require_complete.1 => constants.%complete_type.357
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.e7a31c.3
// CHECK:STDOUT:   %Destroy.facet.loc141_5.3 => constants.%Destroy.facet.191
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.c2c
// CHECK:STDOUT:   %.loc141_5.7 => constants.%.c2e
// CHECK:STDOUT:   %impl.elem0.loc141_5.2 => constants.%Int64.Op.4dca31.1
// CHECK:STDOUT:   %specific_impl_fn.loc141_5.4 => constants.%Int64.Op.4dca31.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @RightShiftWith.loc148(constants.%Int32, constants.%facet_value.2e1) {
// CHECK:STDOUT:   %U.patt.loc146_4.2 => constants.%U.patt.d47
// CHECK:STDOUT:   %U.loc146_4.1 => constants.%Int32
// CHECK:STDOUT:   %RightShiftWith.type.loc147_28.1 => constants.%RightShiftWith.type.8f0
// CHECK:STDOUT:   %.Self.loc147_30.1 => constants.%.Self.c91
// CHECK:STDOUT:   %require_complete.loc147 => constants.%complete_type.93f
// CHECK:STDOUT:   %.Self.as_type.loc147_36.1 => constants.%.Self.as_type.d6f
// CHECK:STDOUT:   %RightShiftWith.assoc_type => constants.%RightShiftWith.assoc_type.813
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.99a
// CHECK:STDOUT:   %RightShiftWith.lookup_impl_witness.loc147 => constants.%RightShiftWith.lookup_impl_witness.bb0
// CHECK:STDOUT:   %impl.elem0.loc147_36.1 => constants.%impl.elem0.cfe
// CHECK:STDOUT:   %RightShiftWith_where.type => constants.%RightShiftWith_where.type.8f0
// CHECK:STDOUT:   %pattern_type.loc147 => constants.%pattern_type.5a0
// CHECK:STDOUT:   %T.patt.loc147_4.2 => constants.%T.patt.a69
// CHECK:STDOUT:   %T.loc147_4.1 => constants.%facet_value.2e1
// CHECK:STDOUT:   %T.as_type.loc148_6.1 => constants.%Int64
// CHECK:STDOUT:   %pattern_type.loc148_4 => constants.%pattern_type.4aa
// CHECK:STDOUT:   %x.param_patt.loc148_4.2 => constants.%x.param_patt.e68
// CHECK:STDOUT:   %x.patt.loc148_4.2 => constants.%x.patt.52a
// CHECK:STDOUT:   %pattern_type.loc148_10 => constants.%pattern_type.8b45
// CHECK:STDOUT:   %y.param_patt.loc148_10.2 => constants.%y.param_patt.e0c
// CHECK:STDOUT:   %y.patt.loc148_10.2 => constants.%y.patt.0a3
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc148_4 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc148_10 => constants.%complete_type.357
// CHECK:STDOUT:   %assoc1 => constants.%assoc1.2c8
// CHECK:STDOUT:   %RightShiftWith.lookup_impl_witness.loc150 => constants.%custom_witness.9e2
// CHECK:STDOUT:   %RightShiftWith.facet => constants.%RightShiftWith.facet.ffd
// CHECK:STDOUT:   %RightShiftWith.WithSelf.Op.type => constants.%RightShiftWith.WithSelf.Op.type.bc1
// CHECK:STDOUT:   %.loc150_5.6 => constants.%.54e
// CHECK:STDOUT:   %impl.elem1.loc150_5.2 => constants.%Op.72717c.15
// CHECK:STDOUT:   %specific_impl_fn.loc150_5.3 => constants.%Op.72717c.15
// CHECK:STDOUT:   %impl.elem0.1 => constants.%Int64
// CHECK:STDOUT:   %require_complete.1 => constants.%complete_type.357
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.e7a31c.3
// CHECK:STDOUT:   %Destroy.facet.loc150_5.3 => constants.%Destroy.facet.191
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.c2c
// CHECK:STDOUT:   %.loc150_5.7 => constants.%.c2e
// CHECK:STDOUT:   %impl.elem0.loc150_5.2 => constants.%Int64.Op.4dca31.1
// CHECK:STDOUT:   %specific_impl_fn.loc150_5.4 => constants.%Int64.Op.4dca31.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @BitAndAssignWith.loc205(constants.%U.67d, constants.%T.80b) {
// CHECK:STDOUT:   %U.patt.loc205_22.2 => constants.%U.patt.d47
// CHECK:STDOUT:   %U.loc205_22.1 => constants.%U.67d
// CHECK:STDOUT:   %BitAndAssignWith.type.loc205_58.1 => constants.%BitAndAssignWith.type.24c915.2
// CHECK:STDOUT:   %pattern_type.loc205_32 => constants.%pattern_type.89d
// CHECK:STDOUT:   %T.patt.loc205_32.2 => constants.%T.patt.7a6
// CHECK:STDOUT:   %T.loc205_32.1 => constants.%T.80b
// CHECK:STDOUT:   %T.as_type.loc205_68.1 => constants.%T.as_type.dca
// CHECK:STDOUT:   %pattern_type.loc205_66 => constants.%pattern_type.d3a81a.2
// CHECK:STDOUT:   %x.patt.loc205_66.2 => constants.%x.patt.66b
// CHECK:STDOUT:   %x.param_patt.loc205_61.2 => constants.%x.param_patt.9e7
// CHECK:STDOUT:   %pattern_type.loc205_72 => constants.%pattern_type.51d1c4.2
// CHECK:STDOUT:   %y.param_patt.loc205_72.2 => constants.%y.param_patt.be3
// CHECK:STDOUT:   %y.patt.loc205_72.2 => constants.%y.patt.b85
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @BitOrAssignWith.loc211(constants.%U.67d, constants.%T.cc8) {
// CHECK:STDOUT:   %U.patt.loc211_21.2 => constants.%U.patt.d47
// CHECK:STDOUT:   %U.loc211_21.1 => constants.%U.67d
// CHECK:STDOUT:   %BitOrAssignWith.type.loc211_56.1 => constants.%BitOrAssignWith.type.3ef7d1.2
// CHECK:STDOUT:   %pattern_type.loc211_31 => constants.%pattern_type.12d
// CHECK:STDOUT:   %T.patt.loc211_31.2 => constants.%T.patt.ae4
// CHECK:STDOUT:   %T.loc211_31.1 => constants.%T.cc8
// CHECK:STDOUT:   %T.as_type.loc211_66.1 => constants.%T.as_type.72f
// CHECK:STDOUT:   %pattern_type.loc211_64 => constants.%pattern_type.cefac8.2
// CHECK:STDOUT:   %x.patt.loc211_64.2 => constants.%x.patt.5d2
// CHECK:STDOUT:   %x.param_patt.loc211_59.2 => constants.%x.param_patt.d0d
// CHECK:STDOUT:   %pattern_type.loc211_70 => constants.%pattern_type.51d1c4.2
// CHECK:STDOUT:   %y.param_patt.loc211_70.2 => constants.%y.param_patt.be3
// CHECK:STDOUT:   %y.patt.loc211_70.2 => constants.%y.patt.b85
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @BitXorAssignWith.loc217(constants.%U.67d, constants.%T.c38) {
// CHECK:STDOUT:   %U.patt.loc217_22.2 => constants.%U.patt.d47
// CHECK:STDOUT:   %U.loc217_22.1 => constants.%U.67d
// CHECK:STDOUT:   %BitXorAssignWith.type.loc217_58.1 => constants.%BitXorAssignWith.type.70c283.2
// CHECK:STDOUT:   %pattern_type.loc217_32 => constants.%pattern_type.472
// CHECK:STDOUT:   %T.patt.loc217_32.2 => constants.%T.patt.55d
// CHECK:STDOUT:   %T.loc217_32.1 => constants.%T.c38
// CHECK:STDOUT:   %T.as_type.loc217_68.1 => constants.%T.as_type.a84
// CHECK:STDOUT:   %pattern_type.loc217_66 => constants.%pattern_type.1bffb5.2
// CHECK:STDOUT:   %x.patt.loc217_66.2 => constants.%x.patt.f3d
// CHECK:STDOUT:   %x.param_patt.loc217_61.2 => constants.%x.param_patt.1d5
// CHECK:STDOUT:   %pattern_type.loc217_72 => constants.%pattern_type.51d1c4.2
// CHECK:STDOUT:   %y.param_patt.loc217_72.2 => constants.%y.param_patt.be3
// CHECK:STDOUT:   %y.patt.loc217_72.2 => constants.%y.patt.b85
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @LeftShiftAssignWith.loc223(constants.%U.67d, constants.%T.de3) {
// CHECK:STDOUT:   %U.patt.loc223_25.2 => constants.%U.patt.d47
// CHECK:STDOUT:   %U.loc223_25.1 => constants.%U.67d
// CHECK:STDOUT:   %LeftShiftAssignWith.type.loc223_64.1 => constants.%LeftShiftAssignWith.type.858756.2
// CHECK:STDOUT:   %pattern_type.loc223_35 => constants.%pattern_type.08d
// CHECK:STDOUT:   %T.patt.loc223_35.2 => constants.%T.patt.bd5
// CHECK:STDOUT:   %T.loc223_35.1 => constants.%T.de3
// CHECK:STDOUT:   %T.as_type.loc223_74.1 => constants.%T.as_type.09c
// CHECK:STDOUT:   %pattern_type.loc223_72 => constants.%pattern_type.6d2709.2
// CHECK:STDOUT:   %x.patt.loc223_72.2 => constants.%x.patt.23b
// CHECK:STDOUT:   %x.param_patt.loc223_67.2 => constants.%x.param_patt.02d
// CHECK:STDOUT:   %pattern_type.loc223_78 => constants.%pattern_type.51d1c4.2
// CHECK:STDOUT:   %y.param_patt.loc223_78.2 => constants.%y.param_patt.be3
// CHECK:STDOUT:   %y.patt.loc223_78.2 => constants.%y.patt.b85
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @RightShiftAssignWith.loc229(constants.%U.67d, constants.%T.d7c) {
// CHECK:STDOUT:   %U.patt.loc229_26.2 => constants.%U.patt.d47
// CHECK:STDOUT:   %U.loc229_26.1 => constants.%U.67d
// CHECK:STDOUT:   %RightShiftAssignWith.type.loc229_66.1 => constants.%RightShiftAssignWith.type.abebad.2
// CHECK:STDOUT:   %pattern_type.loc229_36 => constants.%pattern_type.9e9
// CHECK:STDOUT:   %T.patt.loc229_36.2 => constants.%T.patt.0c5
// CHECK:STDOUT:   %T.loc229_36.1 => constants.%T.d7c
// CHECK:STDOUT:   %T.as_type.loc229_76.1 => constants.%T.as_type.8e8
// CHECK:STDOUT:   %pattern_type.loc229_74 => constants.%pattern_type.e901da.2
// CHECK:STDOUT:   %x.patt.loc229_74.2 => constants.%x.patt.cf9
// CHECK:STDOUT:   %x.param_patt.loc229_69.2 => constants.%x.param_patt.e72
// CHECK:STDOUT:   %pattern_type.loc229_80 => constants.%pattern_type.51d1c4.2
// CHECK:STDOUT:   %y.param_patt.loc229_80.2 => constants.%y.param_patt.be3
// CHECK:STDOUT:   %y.patt.loc229_80.2 => constants.%y.patt.b85
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @BitAndAssignWith.loc205(constants.%i16, constants.%BitAndAssignWith.facet.355) {
// CHECK:STDOUT:   %U.patt.loc205_22.2 => constants.%U.patt.d47
// CHECK:STDOUT:   %U.loc205_22.1 => constants.%i16
// CHECK:STDOUT:   %BitAndAssignWith.type.loc205_58.1 => constants.%BitAndAssignWith.type.0b5
// CHECK:STDOUT:   %pattern_type.loc205_32 => constants.%pattern_type.628
// CHECK:STDOUT:   %T.patt.loc205_32.2 => constants.%T.patt.9fc
// CHECK:STDOUT:   %T.loc205_32.1 => constants.%BitAndAssignWith.facet.355
// CHECK:STDOUT:   %T.as_type.loc205_68.1 => constants.%Int16
// CHECK:STDOUT:   %pattern_type.loc205_66 => constants.%pattern_type.785
// CHECK:STDOUT:   %x.patt.loc205_66.2 => constants.%x.patt.49e
// CHECK:STDOUT:   %x.param_patt.loc205_61.2 => constants.%x.param_patt.69a
// CHECK:STDOUT:   %pattern_type.loc205_72 => constants.%pattern_type.203
// CHECK:STDOUT:   %y.param_patt.loc205_72.2 => constants.%y.param_patt.4b4
// CHECK:STDOUT:   %y.patt.loc205_72.2 => constants.%y.patt.dd3
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc205_61 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc205_72 => constants.%complete_type.2da
// CHECK:STDOUT:   %require_complete.loc207 => constants.%complete_type.9ba
// CHECK:STDOUT:   %BitAndAssignWith.assoc_type => constants.%BitAndAssignWith.assoc_type.c34
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.f42
// CHECK:STDOUT:   %BitAndAssignWith.WithSelf.Op.type => constants.%BitAndAssignWith.WithSelf.Op.type.daa
// CHECK:STDOUT:   %.loc207_5.2 => constants.%.483
// CHECK:STDOUT:   %BitAndAssignWith.lookup_impl_witness => constants.%custom_witness.f28
// CHECK:STDOUT:   %impl.elem0.loc207_5.2 => constants.%Int16.Op.4dca31.13
// CHECK:STDOUT:   %specific_impl_fn.loc207_5.2 => constants.%Int16.Op.4dca31.13
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @BitOrAssignWith.loc211(constants.%i16, constants.%BitOrAssignWith.facet.ea2) {
// CHECK:STDOUT:   %U.patt.loc211_21.2 => constants.%U.patt.d47
// CHECK:STDOUT:   %U.loc211_21.1 => constants.%i16
// CHECK:STDOUT:   %BitOrAssignWith.type.loc211_56.1 => constants.%BitOrAssignWith.type.8ff
// CHECK:STDOUT:   %pattern_type.loc211_31 => constants.%pattern_type.1db
// CHECK:STDOUT:   %T.patt.loc211_31.2 => constants.%T.patt.670
// CHECK:STDOUT:   %T.loc211_31.1 => constants.%BitOrAssignWith.facet.ea2
// CHECK:STDOUT:   %T.as_type.loc211_66.1 => constants.%Int16
// CHECK:STDOUT:   %pattern_type.loc211_64 => constants.%pattern_type.785
// CHECK:STDOUT:   %x.patt.loc211_64.2 => constants.%x.patt.49e
// CHECK:STDOUT:   %x.param_patt.loc211_59.2 => constants.%x.param_patt.69a
// CHECK:STDOUT:   %pattern_type.loc211_70 => constants.%pattern_type.203
// CHECK:STDOUT:   %y.param_patt.loc211_70.2 => constants.%y.param_patt.4b4
// CHECK:STDOUT:   %y.patt.loc211_70.2 => constants.%y.patt.dd3
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc211_59 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc211_70 => constants.%complete_type.2da
// CHECK:STDOUT:   %require_complete.loc213 => constants.%complete_type.650
// CHECK:STDOUT:   %BitOrAssignWith.assoc_type => constants.%BitOrAssignWith.assoc_type.b1c
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.392
// CHECK:STDOUT:   %BitOrAssignWith.WithSelf.Op.type => constants.%BitOrAssignWith.WithSelf.Op.type.581
// CHECK:STDOUT:   %.loc213_5.2 => constants.%.233
// CHECK:STDOUT:   %BitOrAssignWith.lookup_impl_witness => constants.%custom_witness.c53
// CHECK:STDOUT:   %impl.elem0.loc213_5.2 => constants.%Int16.Op.4dca31.14
// CHECK:STDOUT:   %specific_impl_fn.loc213_5.2 => constants.%Int16.Op.4dca31.14
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @BitXorAssignWith.loc217(constants.%i16, constants.%BitXorAssignWith.facet.84f) {
// CHECK:STDOUT:   %U.patt.loc217_22.2 => constants.%U.patt.d47
// CHECK:STDOUT:   %U.loc217_22.1 => constants.%i16
// CHECK:STDOUT:   %BitXorAssignWith.type.loc217_58.1 => constants.%BitXorAssignWith.type.1ee
// CHECK:STDOUT:   %pattern_type.loc217_32 => constants.%pattern_type.fba
// CHECK:STDOUT:   %T.patt.loc217_32.2 => constants.%T.patt.72a
// CHECK:STDOUT:   %T.loc217_32.1 => constants.%BitXorAssignWith.facet.84f
// CHECK:STDOUT:   %T.as_type.loc217_68.1 => constants.%Int16
// CHECK:STDOUT:   %pattern_type.loc217_66 => constants.%pattern_type.785
// CHECK:STDOUT:   %x.patt.loc217_66.2 => constants.%x.patt.49e
// CHECK:STDOUT:   %x.param_patt.loc217_61.2 => constants.%x.param_patt.69a
// CHECK:STDOUT:   %pattern_type.loc217_72 => constants.%pattern_type.203
// CHECK:STDOUT:   %y.param_patt.loc217_72.2 => constants.%y.param_patt.4b4
// CHECK:STDOUT:   %y.patt.loc217_72.2 => constants.%y.patt.dd3
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc217_61 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc217_72 => constants.%complete_type.2da
// CHECK:STDOUT:   %require_complete.loc219 => constants.%complete_type.054
// CHECK:STDOUT:   %BitXorAssignWith.assoc_type => constants.%BitXorAssignWith.assoc_type.313
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.b72
// CHECK:STDOUT:   %BitXorAssignWith.WithSelf.Op.type => constants.%BitXorAssignWith.WithSelf.Op.type.0db
// CHECK:STDOUT:   %.loc219_5.2 => constants.%.298
// CHECK:STDOUT:   %BitXorAssignWith.lookup_impl_witness => constants.%custom_witness.696
// CHECK:STDOUT:   %impl.elem0.loc219_5.2 => constants.%Int16.Op.4dca31.15
// CHECK:STDOUT:   %specific_impl_fn.loc219_5.2 => constants.%Int16.Op.4dca31.15
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @LeftShiftAssignWith.loc223(constants.%i16, constants.%LeftShiftAssignWith.facet.71f) {
// CHECK:STDOUT:   %U.patt.loc223_25.2 => constants.%U.patt.d47
// CHECK:STDOUT:   %U.loc223_25.1 => constants.%i16
// CHECK:STDOUT:   %LeftShiftAssignWith.type.loc223_64.1 => constants.%LeftShiftAssignWith.type.62a
// CHECK:STDOUT:   %pattern_type.loc223_35 => constants.%pattern_type.65f
// CHECK:STDOUT:   %T.patt.loc223_35.2 => constants.%T.patt.fe8
// CHECK:STDOUT:   %T.loc223_35.1 => constants.%LeftShiftAssignWith.facet.71f
// CHECK:STDOUT:   %T.as_type.loc223_74.1 => constants.%Int16
// CHECK:STDOUT:   %pattern_type.loc223_72 => constants.%pattern_type.785
// CHECK:STDOUT:   %x.patt.loc223_72.2 => constants.%x.patt.49e
// CHECK:STDOUT:   %x.param_patt.loc223_67.2 => constants.%x.param_patt.69a
// CHECK:STDOUT:   %pattern_type.loc223_78 => constants.%pattern_type.203
// CHECK:STDOUT:   %y.param_patt.loc223_78.2 => constants.%y.param_patt.4b4
// CHECK:STDOUT:   %y.patt.loc223_78.2 => constants.%y.patt.dd3
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc223_67 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc223_78 => constants.%complete_type.2da
// CHECK:STDOUT:   %require_complete.loc225 => constants.%complete_type.454
// CHECK:STDOUT:   %LeftShiftAssignWith.assoc_type => constants.%LeftShiftAssignWith.assoc_type.af0
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.c84
// CHECK:STDOUT:   %LeftShiftAssignWith.WithSelf.Op.type => constants.%LeftShiftAssignWith.WithSelf.Op.type.412
// CHECK:STDOUT:   %.loc225_5.2 => constants.%.308
// CHECK:STDOUT:   %LeftShiftAssignWith.lookup_impl_witness => constants.%custom_witness.95b
// CHECK:STDOUT:   %impl.elem0.loc225_5.2 => constants.%Int16.Op.4dca31.16
// CHECK:STDOUT:   %specific_impl_fn.loc225_5.2 => constants.%Int16.Op.4dca31.16
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @RightShiftAssignWith.loc229(constants.%i16, constants.%RightShiftAssignWith.facet.f54) {
// CHECK:STDOUT:   %U.patt.loc229_26.2 => constants.%U.patt.d47
// CHECK:STDOUT:   %U.loc229_26.1 => constants.%i16
// CHECK:STDOUT:   %RightShiftAssignWith.type.loc229_66.1 => constants.%RightShiftAssignWith.type.b1f
// CHECK:STDOUT:   %pattern_type.loc229_36 => constants.%pattern_type.193
// CHECK:STDOUT:   %T.patt.loc229_36.2 => constants.%T.patt.613
// CHECK:STDOUT:   %T.loc229_36.1 => constants.%RightShiftAssignWith.facet.f54
// CHECK:STDOUT:   %T.as_type.loc229_76.1 => constants.%Int16
// CHECK:STDOUT:   %pattern_type.loc229_74 => constants.%pattern_type.785
// CHECK:STDOUT:   %x.patt.loc229_74.2 => constants.%x.patt.49e
// CHECK:STDOUT:   %x.param_patt.loc229_69.2 => constants.%x.param_patt.69a
// CHECK:STDOUT:   %pattern_type.loc229_80 => constants.%pattern_type.203
// CHECK:STDOUT:   %y.param_patt.loc229_80.2 => constants.%y.param_patt.4b4
// CHECK:STDOUT:   %y.patt.loc229_80.2 => constants.%y.patt.dd3
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc229_69 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc229_80 => constants.%complete_type.2da
// CHECK:STDOUT:   %require_complete.loc231 => constants.%complete_type.f10
// CHECK:STDOUT:   %RightShiftAssignWith.assoc_type => constants.%RightShiftAssignWith.assoc_type.d35
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.655
// CHECK:STDOUT:   %RightShiftAssignWith.WithSelf.Op.type => constants.%RightShiftAssignWith.WithSelf.Op.type.5a9
// CHECK:STDOUT:   %.loc231_5.2 => constants.%.6bf
// CHECK:STDOUT:   %RightShiftAssignWith.lookup_impl_witness => constants.%custom_witness.114
// CHECK:STDOUT:   %impl.elem0.loc231_5.2 => constants.%Int16.Op.4dca31.17
// CHECK:STDOUT:   %specific_impl_fn.loc231_5.2 => constants.%Int16.Op.4dca31.17
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @BitAndAssignWith.loc205(constants.%Int32, constants.%BitAndAssignWith.facet.5d5) {
// CHECK:STDOUT:   %U.patt.loc205_22.2 => constants.%U.patt.d47
// CHECK:STDOUT:   %U.loc205_22.1 => constants.%Int32
// CHECK:STDOUT:   %BitAndAssignWith.type.loc205_58.1 => constants.%BitAndAssignWith.type.53c
// CHECK:STDOUT:   %pattern_type.loc205_32 => constants.%pattern_type.cf6
// CHECK:STDOUT:   %T.patt.loc205_32.2 => constants.%T.patt.c2f
// CHECK:STDOUT:   %T.loc205_32.1 => constants.%BitAndAssignWith.facet.5d5
// CHECK:STDOUT:   %T.as_type.loc205_68.1 => constants.%Int32
// CHECK:STDOUT:   %pattern_type.loc205_66 => constants.%pattern_type.8b45
// CHECK:STDOUT:   %x.patt.loc205_66.2 => constants.%x.patt.d79
// CHECK:STDOUT:   %x.param_patt.loc205_61.2 => constants.%x.param_patt.0c7
// CHECK:STDOUT:   %pattern_type.loc205_72 => constants.%pattern_type.8b45
// CHECK:STDOUT:   %y.param_patt.loc205_72.2 => constants.%y.param_patt.e0c
// CHECK:STDOUT:   %y.patt.loc205_72.2 => constants.%y.patt.0a3
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc205_61 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc205_72 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc207 => constants.%complete_type.824
// CHECK:STDOUT:   %BitAndAssignWith.assoc_type => constants.%BitAndAssignWith.assoc_type.3ad
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.04c
// CHECK:STDOUT:   %BitAndAssignWith.WithSelf.Op.type => constants.%BitAndAssignWith.WithSelf.Op.type.3ea
// CHECK:STDOUT:   %.loc207_5.2 => constants.%.0eb
// CHECK:STDOUT:   %BitAndAssignWith.lookup_impl_witness => constants.%custom_witness.4ae282.1
// CHECK:STDOUT:   %impl.elem0.loc207_5.2 => constants.%Int32.Op.4dca31.12
// CHECK:STDOUT:   %specific_impl_fn.loc207_5.2 => constants.%Int32.Op.4dca31.12
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @BitOrAssignWith.loc211(constants.%Int32, constants.%BitOrAssignWith.facet.ad2) {
// CHECK:STDOUT:   %U.patt.loc211_21.2 => constants.%U.patt.d47
// CHECK:STDOUT:   %U.loc211_21.1 => constants.%Int32
// CHECK:STDOUT:   %BitOrAssignWith.type.loc211_56.1 => constants.%BitOrAssignWith.type.46f
// CHECK:STDOUT:   %pattern_type.loc211_31 => constants.%pattern_type.b51
// CHECK:STDOUT:   %T.patt.loc211_31.2 => constants.%T.patt.9dc
// CHECK:STDOUT:   %T.loc211_31.1 => constants.%BitOrAssignWith.facet.ad2
// CHECK:STDOUT:   %T.as_type.loc211_66.1 => constants.%Int32
// CHECK:STDOUT:   %pattern_type.loc211_64 => constants.%pattern_type.8b45
// CHECK:STDOUT:   %x.patt.loc211_64.2 => constants.%x.patt.d79
// CHECK:STDOUT:   %x.param_patt.loc211_59.2 => constants.%x.param_patt.0c7
// CHECK:STDOUT:   %pattern_type.loc211_70 => constants.%pattern_type.8b45
// CHECK:STDOUT:   %y.param_patt.loc211_70.2 => constants.%y.param_patt.e0c
// CHECK:STDOUT:   %y.patt.loc211_70.2 => constants.%y.patt.0a3
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc211_59 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc211_70 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc213 => constants.%complete_type.c85
// CHECK:STDOUT:   %BitOrAssignWith.assoc_type => constants.%BitOrAssignWith.assoc_type.bc6
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.63a
// CHECK:STDOUT:   %BitOrAssignWith.WithSelf.Op.type => constants.%BitOrAssignWith.WithSelf.Op.type.877
// CHECK:STDOUT:   %.loc213_5.2 => constants.%.999
// CHECK:STDOUT:   %BitOrAssignWith.lookup_impl_witness => constants.%custom_witness.9a90d3.1
// CHECK:STDOUT:   %impl.elem0.loc213_5.2 => constants.%Int32.Op.4dca31.14
// CHECK:STDOUT:   %specific_impl_fn.loc213_5.2 => constants.%Int32.Op.4dca31.14
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @BitXorAssignWith.loc217(constants.%Int32, constants.%BitXorAssignWith.facet.627) {
// CHECK:STDOUT:   %U.patt.loc217_22.2 => constants.%U.patt.d47
// CHECK:STDOUT:   %U.loc217_22.1 => constants.%Int32
// CHECK:STDOUT:   %BitXorAssignWith.type.loc217_58.1 => constants.%BitXorAssignWith.type.0e4
// CHECK:STDOUT:   %pattern_type.loc217_32 => constants.%pattern_type.e71
// CHECK:STDOUT:   %T.patt.loc217_32.2 => constants.%T.patt.6f4
// CHECK:STDOUT:   %T.loc217_32.1 => constants.%BitXorAssignWith.facet.627
// CHECK:STDOUT:   %T.as_type.loc217_68.1 => constants.%Int32
// CHECK:STDOUT:   %pattern_type.loc217_66 => constants.%pattern_type.8b45
// CHECK:STDOUT:   %x.patt.loc217_66.2 => constants.%x.patt.d79
// CHECK:STDOUT:   %x.param_patt.loc217_61.2 => constants.%x.param_patt.0c7
// CHECK:STDOUT:   %pattern_type.loc217_72 => constants.%pattern_type.8b45
// CHECK:STDOUT:   %y.param_patt.loc217_72.2 => constants.%y.param_patt.e0c
// CHECK:STDOUT:   %y.patt.loc217_72.2 => constants.%y.patt.0a3
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc217_61 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc217_72 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc219 => constants.%complete_type.267
// CHECK:STDOUT:   %BitXorAssignWith.assoc_type => constants.%BitXorAssignWith.assoc_type.e5b
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.1fe
// CHECK:STDOUT:   %BitXorAssignWith.WithSelf.Op.type => constants.%BitXorAssignWith.WithSelf.Op.type.243
// CHECK:STDOUT:   %.loc219_5.2 => constants.%.384
// CHECK:STDOUT:   %BitXorAssignWith.lookup_impl_witness => constants.%custom_witness.9bdd97.1
// CHECK:STDOUT:   %impl.elem0.loc219_5.2 => constants.%Int32.Op.4dca31.15
// CHECK:STDOUT:   %specific_impl_fn.loc219_5.2 => constants.%Int32.Op.4dca31.15
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @LeftShiftAssignWith.loc223(constants.%Int32, constants.%LeftShiftAssignWith.facet.fb8) {
// CHECK:STDOUT:   %U.patt.loc223_25.2 => constants.%U.patt.d47
// CHECK:STDOUT:   %U.loc223_25.1 => constants.%Int32
// CHECK:STDOUT:   %LeftShiftAssignWith.type.loc223_64.1 => constants.%LeftShiftAssignWith.type.635
// CHECK:STDOUT:   %pattern_type.loc223_35 => constants.%pattern_type.1d0
// CHECK:STDOUT:   %T.patt.loc223_35.2 => constants.%T.patt.a16
// CHECK:STDOUT:   %T.loc223_35.1 => constants.%LeftShiftAssignWith.facet.fb8
// CHECK:STDOUT:   %T.as_type.loc223_74.1 => constants.%Int32
// CHECK:STDOUT:   %pattern_type.loc223_72 => constants.%pattern_type.8b45
// CHECK:STDOUT:   %x.patt.loc223_72.2 => constants.%x.patt.d79
// CHECK:STDOUT:   %x.param_patt.loc223_67.2 => constants.%x.param_patt.0c7
// CHECK:STDOUT:   %pattern_type.loc223_78 => constants.%pattern_type.8b45
// CHECK:STDOUT:   %y.param_patt.loc223_78.2 => constants.%y.param_patt.e0c
// CHECK:STDOUT:   %y.patt.loc223_78.2 => constants.%y.patt.0a3
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc223_67 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc223_78 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc225 => constants.%complete_type.bf0
// CHECK:STDOUT:   %LeftShiftAssignWith.assoc_type => constants.%LeftShiftAssignWith.assoc_type.be9
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.765
// CHECK:STDOUT:   %LeftShiftAssignWith.WithSelf.Op.type => constants.%LeftShiftAssignWith.WithSelf.Op.type.195
// CHECK:STDOUT:   %.loc225_5.2 => constants.%.8fd
// CHECK:STDOUT:   %LeftShiftAssignWith.lookup_impl_witness => constants.%custom_witness.3e2e0c.1
// CHECK:STDOUT:   %impl.elem0.loc225_5.2 => constants.%Int32.Op.4dca31.16
// CHECK:STDOUT:   %specific_impl_fn.loc225_5.2 => constants.%Int32.Op.4dca31.16
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @RightShiftAssignWith.loc229(constants.%Int32, constants.%RightShiftAssignWith.facet.288) {
// CHECK:STDOUT:   %U.patt.loc229_26.2 => constants.%U.patt.d47
// CHECK:STDOUT:   %U.loc229_26.1 => constants.%Int32
// CHECK:STDOUT:   %RightShiftAssignWith.type.loc229_66.1 => constants.%RightShiftAssignWith.type.239
// CHECK:STDOUT:   %pattern_type.loc229_36 => constants.%pattern_type.0b1
// CHECK:STDOUT:   %T.patt.loc229_36.2 => constants.%T.patt.04a
// CHECK:STDOUT:   %T.loc229_36.1 => constants.%RightShiftAssignWith.facet.288
// CHECK:STDOUT:   %T.as_type.loc229_76.1 => constants.%Int32
// CHECK:STDOUT:   %pattern_type.loc229_74 => constants.%pattern_type.8b45
// CHECK:STDOUT:   %x.patt.loc229_74.2 => constants.%x.patt.d79
// CHECK:STDOUT:   %x.param_patt.loc229_69.2 => constants.%x.param_patt.0c7
// CHECK:STDOUT:   %pattern_type.loc229_80 => constants.%pattern_type.8b45
// CHECK:STDOUT:   %y.param_patt.loc229_80.2 => constants.%y.param_patt.e0c
// CHECK:STDOUT:   %y.patt.loc229_80.2 => constants.%y.patt.0a3
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc229_69 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc229_80 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc231 => constants.%complete_type.f23
// CHECK:STDOUT:   %RightShiftAssignWith.assoc_type => constants.%RightShiftAssignWith.assoc_type.21a
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.e7b
// CHECK:STDOUT:   %RightShiftAssignWith.WithSelf.Op.type => constants.%RightShiftAssignWith.WithSelf.Op.type.738
// CHECK:STDOUT:   %.loc231_5.2 => constants.%.985
// CHECK:STDOUT:   %RightShiftAssignWith.lookup_impl_witness => constants.%custom_witness.cbf0fb.1
// CHECK:STDOUT:   %impl.elem0.loc231_5.2 => constants.%Int32.Op.4dca31.17
// CHECK:STDOUT:   %specific_impl_fn.loc231_5.2 => constants.%Int32.Op.4dca31.17
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @BitAndAssignWith.loc205(constants.%Int32, constants.%BitAndAssignWith.facet.cd3) {
// CHECK:STDOUT:   %U.patt.loc205_22.2 => constants.%U.patt.d47
// CHECK:STDOUT:   %U.loc205_22.1 => constants.%Int32
// CHECK:STDOUT:   %BitAndAssignWith.type.loc205_58.1 => constants.%BitAndAssignWith.type.53c
// CHECK:STDOUT:   %pattern_type.loc205_32 => constants.%pattern_type.cf6
// CHECK:STDOUT:   %T.patt.loc205_32.2 => constants.%T.patt.c2f
// CHECK:STDOUT:   %T.loc205_32.1 => constants.%BitAndAssignWith.facet.cd3
// CHECK:STDOUT:   %T.as_type.loc205_68.1 => constants.%Int64
// CHECK:STDOUT:   %pattern_type.loc205_66 => constants.%pattern_type.4aa
// CHECK:STDOUT:   %x.patt.loc205_66.2 => constants.%x.patt.ace
// CHECK:STDOUT:   %x.param_patt.loc205_61.2 => constants.%x.param_patt.b33
// CHECK:STDOUT:   %pattern_type.loc205_72 => constants.%pattern_type.8b45
// CHECK:STDOUT:   %y.param_patt.loc205_72.2 => constants.%y.param_patt.e0c
// CHECK:STDOUT:   %y.patt.loc205_72.2 => constants.%y.patt.0a3
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc205_61 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc205_72 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc207 => constants.%complete_type.824
// CHECK:STDOUT:   %BitAndAssignWith.assoc_type => constants.%BitAndAssignWith.assoc_type.3ad
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.04c
// CHECK:STDOUT:   %BitAndAssignWith.WithSelf.Op.type => constants.%BitAndAssignWith.WithSelf.Op.type.896
// CHECK:STDOUT:   %.loc207_5.2 => constants.%.c50
// CHECK:STDOUT:   %BitAndAssignWith.lookup_impl_witness => constants.%custom_witness.4ae282.2
// CHECK:STDOUT:   %impl.elem0.loc207_5.2 => constants.%Int64.Op.4dca31.2
// CHECK:STDOUT:   %specific_impl_fn.loc207_5.2 => constants.%Int64.Op.4dca31.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @BitOrAssignWith.loc211(constants.%Int32, constants.%BitOrAssignWith.facet.c6e) {
// CHECK:STDOUT:   %U.patt.loc211_21.2 => constants.%U.patt.d47
// CHECK:STDOUT:   %U.loc211_21.1 => constants.%Int32
// CHECK:STDOUT:   %BitOrAssignWith.type.loc211_56.1 => constants.%BitOrAssignWith.type.46f
// CHECK:STDOUT:   %pattern_type.loc211_31 => constants.%pattern_type.b51
// CHECK:STDOUT:   %T.patt.loc211_31.2 => constants.%T.patt.9dc
// CHECK:STDOUT:   %T.loc211_31.1 => constants.%BitOrAssignWith.facet.c6e
// CHECK:STDOUT:   %T.as_type.loc211_66.1 => constants.%Int64
// CHECK:STDOUT:   %pattern_type.loc211_64 => constants.%pattern_type.4aa
// CHECK:STDOUT:   %x.patt.loc211_64.2 => constants.%x.patt.ace
// CHECK:STDOUT:   %x.param_patt.loc211_59.2 => constants.%x.param_patt.b33
// CHECK:STDOUT:   %pattern_type.loc211_70 => constants.%pattern_type.8b45
// CHECK:STDOUT:   %y.param_patt.loc211_70.2 => constants.%y.param_patt.e0c
// CHECK:STDOUT:   %y.patt.loc211_70.2 => constants.%y.patt.0a3
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc211_59 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc211_70 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc213 => constants.%complete_type.c85
// CHECK:STDOUT:   %BitOrAssignWith.assoc_type => constants.%BitOrAssignWith.assoc_type.bc6
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.63a
// CHECK:STDOUT:   %BitOrAssignWith.WithSelf.Op.type => constants.%BitOrAssignWith.WithSelf.Op.type.18d
// CHECK:STDOUT:   %.loc213_5.2 => constants.%.5f2
// CHECK:STDOUT:   %BitOrAssignWith.lookup_impl_witness => constants.%custom_witness.9a90d3.2
// CHECK:STDOUT:   %impl.elem0.loc213_5.2 => constants.%Int64.Op.4dca31.4
// CHECK:STDOUT:   %specific_impl_fn.loc213_5.2 => constants.%Int64.Op.4dca31.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @BitXorAssignWith.loc217(constants.%Int32, constants.%BitXorAssignWith.facet.896) {
// CHECK:STDOUT:   %U.patt.loc217_22.2 => constants.%U.patt.d47
// CHECK:STDOUT:   %U.loc217_22.1 => constants.%Int32
// CHECK:STDOUT:   %BitXorAssignWith.type.loc217_58.1 => constants.%BitXorAssignWith.type.0e4
// CHECK:STDOUT:   %pattern_type.loc217_32 => constants.%pattern_type.e71
// CHECK:STDOUT:   %T.patt.loc217_32.2 => constants.%T.patt.6f4
// CHECK:STDOUT:   %T.loc217_32.1 => constants.%BitXorAssignWith.facet.896
// CHECK:STDOUT:   %T.as_type.loc217_68.1 => constants.%Int64
// CHECK:STDOUT:   %pattern_type.loc217_66 => constants.%pattern_type.4aa
// CHECK:STDOUT:   %x.patt.loc217_66.2 => constants.%x.patt.ace
// CHECK:STDOUT:   %x.param_patt.loc217_61.2 => constants.%x.param_patt.b33
// CHECK:STDOUT:   %pattern_type.loc217_72 => constants.%pattern_type.8b45
// CHECK:STDOUT:   %y.param_patt.loc217_72.2 => constants.%y.param_patt.e0c
// CHECK:STDOUT:   %y.patt.loc217_72.2 => constants.%y.patt.0a3
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc217_61 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc217_72 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc219 => constants.%complete_type.267
// CHECK:STDOUT:   %BitXorAssignWith.assoc_type => constants.%BitXorAssignWith.assoc_type.e5b
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.1fe
// CHECK:STDOUT:   %BitXorAssignWith.WithSelf.Op.type => constants.%BitXorAssignWith.WithSelf.Op.type.198
// CHECK:STDOUT:   %.loc219_5.2 => constants.%.52d
// CHECK:STDOUT:   %BitXorAssignWith.lookup_impl_witness => constants.%custom_witness.9bdd97.2
// CHECK:STDOUT:   %impl.elem0.loc219_5.2 => constants.%Int64.Op.4dca31.5
// CHECK:STDOUT:   %specific_impl_fn.loc219_5.2 => constants.%Int64.Op.4dca31.5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @LeftShiftAssignWith.loc223(constants.%Int32, constants.%LeftShiftAssignWith.facet.8cb) {
// CHECK:STDOUT:   %U.patt.loc223_25.2 => constants.%U.patt.d47
// CHECK:STDOUT:   %U.loc223_25.1 => constants.%Int32
// CHECK:STDOUT:   %LeftShiftAssignWith.type.loc223_64.1 => constants.%LeftShiftAssignWith.type.635
// CHECK:STDOUT:   %pattern_type.loc223_35 => constants.%pattern_type.1d0
// CHECK:STDOUT:   %T.patt.loc223_35.2 => constants.%T.patt.a16
// CHECK:STDOUT:   %T.loc223_35.1 => constants.%LeftShiftAssignWith.facet.8cb
// CHECK:STDOUT:   %T.as_type.loc223_74.1 => constants.%Int64
// CHECK:STDOUT:   %pattern_type.loc223_72 => constants.%pattern_type.4aa
// CHECK:STDOUT:   %x.patt.loc223_72.2 => constants.%x.patt.ace
// CHECK:STDOUT:   %x.param_patt.loc223_67.2 => constants.%x.param_patt.b33
// CHECK:STDOUT:   %pattern_type.loc223_78 => constants.%pattern_type.8b45
// CHECK:STDOUT:   %y.param_patt.loc223_78.2 => constants.%y.param_patt.e0c
// CHECK:STDOUT:   %y.patt.loc223_78.2 => constants.%y.patt.0a3
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc223_67 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc223_78 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc225 => constants.%complete_type.bf0
// CHECK:STDOUT:   %LeftShiftAssignWith.assoc_type => constants.%LeftShiftAssignWith.assoc_type.be9
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.765
// CHECK:STDOUT:   %LeftShiftAssignWith.WithSelf.Op.type => constants.%LeftShiftAssignWith.WithSelf.Op.type.f6f
// CHECK:STDOUT:   %.loc225_5.2 => constants.%.a80
// CHECK:STDOUT:   %LeftShiftAssignWith.lookup_impl_witness => constants.%custom_witness.3e2e0c.2
// CHECK:STDOUT:   %impl.elem0.loc225_5.2 => constants.%Int64.Op.4dca31.6
// CHECK:STDOUT:   %specific_impl_fn.loc225_5.2 => constants.%Int64.Op.4dca31.6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @RightShiftAssignWith.loc229(constants.%Int32, constants.%RightShiftAssignWith.facet.040) {
// CHECK:STDOUT:   %U.patt.loc229_26.2 => constants.%U.patt.d47
// CHECK:STDOUT:   %U.loc229_26.1 => constants.%Int32
// CHECK:STDOUT:   %RightShiftAssignWith.type.loc229_66.1 => constants.%RightShiftAssignWith.type.239
// CHECK:STDOUT:   %pattern_type.loc229_36 => constants.%pattern_type.0b1
// CHECK:STDOUT:   %T.patt.loc229_36.2 => constants.%T.patt.04a
// CHECK:STDOUT:   %T.loc229_36.1 => constants.%RightShiftAssignWith.facet.040
// CHECK:STDOUT:   %T.as_type.loc229_76.1 => constants.%Int64
// CHECK:STDOUT:   %pattern_type.loc229_74 => constants.%pattern_type.4aa
// CHECK:STDOUT:   %x.patt.loc229_74.2 => constants.%x.patt.ace
// CHECK:STDOUT:   %x.param_patt.loc229_69.2 => constants.%x.param_patt.b33
// CHECK:STDOUT:   %pattern_type.loc229_80 => constants.%pattern_type.8b45
// CHECK:STDOUT:   %y.param_patt.loc229_80.2 => constants.%y.param_patt.e0c
// CHECK:STDOUT:   %y.patt.loc229_80.2 => constants.%y.patt.0a3
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc229_69 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc229_80 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc231 => constants.%complete_type.f23
// CHECK:STDOUT:   %RightShiftAssignWith.assoc_type => constants.%RightShiftAssignWith.assoc_type.21a
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.e7b
// CHECK:STDOUT:   %RightShiftAssignWith.WithSelf.Op.type => constants.%RightShiftAssignWith.WithSelf.Op.type.2ee
// CHECK:STDOUT:   %.loc231_5.2 => constants.%.7e16
// CHECK:STDOUT:   %RightShiftAssignWith.lookup_impl_witness => constants.%custom_witness.cbf0fb.2
// CHECK:STDOUT:   %impl.elem0.loc231_5.2 => constants.%Int64.Op.4dca31.7
// CHECK:STDOUT:   %specific_impl_fn.loc231_5.2 => constants.%Int64.Op.4dca31.7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestBitComplement(constants.%T.438) {
// CHECK:STDOUT:   %T.patt.loc259_4.2 => constants.%T.patt.d8c
// CHECK:STDOUT:   %T.loc259_4.1 => constants.%T.438
// CHECK:STDOUT:   %T.as_type.loc260_6.1 => constants.%T.as_type.e33
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.834f91.2
// CHECK:STDOUT:   %x.param_patt.loc260_4.2 => constants.%x.param_patt.b2f
// CHECK:STDOUT:   %x.patt.loc260_4.2 => constants.%x.patt.647
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestBitComplement(constants.%facet_value.8bd) {
// CHECK:STDOUT:   %T.patt.loc259_4.2 => constants.%T.patt.d8c
// CHECK:STDOUT:   %T.loc259_4.1 => constants.%facet_value.8bd
// CHECK:STDOUT:   %T.as_type.loc260_6.1 => constants.%Int16
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.785
// CHECK:STDOUT:   %x.param_patt.loc260_4.2 => constants.%x.param_patt.3b0
// CHECK:STDOUT:   %x.patt.loc260_4.2 => constants.%x.patt.b76
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc260 => constants.%complete_type.357
// CHECK:STDOUT:   %BitComplement.lookup_impl_witness => constants.%custom_witness.2b7
// CHECK:STDOUT:   %BitComplement.facet.loc262_3.3 => constants.%BitComplement.facet.610
// CHECK:STDOUT:   %BitComplement.WithSelf.Op.type => constants.%BitComplement.WithSelf.Op.type.80e
// CHECK:STDOUT:   %.loc262_3.7 => constants.%.c87
// CHECK:STDOUT:   %impl.elem1.loc262_3.2 => constants.%Int16.Op.4dca31.18
// CHECK:STDOUT:   %specific_impl_fn.loc262_3.3 => constants.%Int16.Op.4dca31.18
// CHECK:STDOUT:   %impl.elem0.1 => constants.%Int64
// CHECK:STDOUT:   %require_complete.1 => constants.%complete_type.357
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.e7a31c.3
// CHECK:STDOUT:   %Destroy.facet.loc262_3.3 => constants.%Destroy.facet.191
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.c2c
// CHECK:STDOUT:   %.loc262_3.8 => constants.%.c2e
// CHECK:STDOUT:   %impl.elem0.loc262_3.2 => constants.%Int64.Op.4dca31.1
// CHECK:STDOUT:   %specific_impl_fn.loc262_3.4 => constants.%Int64.Op.4dca31.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TestBitComplement(constants.%facet_value.103) {
// CHECK:STDOUT:   %T.patt.loc259_4.2 => constants.%T.patt.d8c
// CHECK:STDOUT:   %T.loc259_4.1 => constants.%facet_value.103
// CHECK:STDOUT:   %T.as_type.loc260_6.1 => constants.%Int32
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.8b45
// CHECK:STDOUT:   %x.param_patt.loc260_4.2 => constants.%x.param_patt.27c
// CHECK:STDOUT:   %x.patt.loc260_4.2 => constants.%x.patt.4c9
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc260 => constants.%complete_type.357
// CHECK:STDOUT:   %BitComplement.lookup_impl_witness => constants.%custom_witness.f78
// CHECK:STDOUT:   %BitComplement.facet.loc262_3.3 => constants.%BitComplement.facet.041
// CHECK:STDOUT:   %BitComplement.WithSelf.Op.type => constants.%BitComplement.WithSelf.Op.type.8e2
// CHECK:STDOUT:   %.loc262_3.7 => constants.%.681
// CHECK:STDOUT:   %impl.elem1.loc262_3.2 => constants.%Int32.Op.4dca31.18
// CHECK:STDOUT:   %specific_impl_fn.loc262_3.3 => constants.%Int32.Op.4dca31.18
// CHECK:STDOUT:   %impl.elem0.1 => constants.%Int32
// CHECK:STDOUT:   %require_complete.1 => constants.%complete_type.357
// CHECK:STDOUT:   %Destroy.lookup_impl_witness => constants.%custom_witness.e7a31c.2
// CHECK:STDOUT:   %Destroy.facet.loc262_3.3 => constants.%Destroy.facet.099
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type => constants.%Destroy.WithSelf.Op.type.b76
// CHECK:STDOUT:   %.loc262_3.8 => constants.%.043
// CHECK:STDOUT:   %impl.elem0.loc262_3.2 => constants.%Int32.Op.4dca31.1
// CHECK:STDOUT:   %specific_impl_fn.loc262_3.4 => constants.%Int32.Op.4dca31.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
