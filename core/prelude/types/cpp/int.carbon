// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

package Core library "prelude/types/cpp/int";

import library "prelude/copy";
import library "prelude/operators";
import library "prelude/types/int";
import library "prelude/types/int_literal";
import library "prelude/types/uint";

namespace CppCompat;

class CppCompat.Long32 {
  adapt i32;
}

class CppCompat.ULong32 {
  adapt u32;
}

class CppCompat.LongLong64 {
  adapt i64;
}

class CppCompat.ULongLong64 {
  adapt u64;
}

// Copy

impl CppCompat.Long32 as Copy {
  fn Op[self: Self]() -> Self = "primitive_copy";
}

impl CppCompat.ULong32 as Copy {
  fn Op[self: Self]() -> Self = "primitive_copy";
}

impl CppCompat.LongLong64 as Copy {
  fn Op[self: Self]() -> Self = "primitive_copy";
}

impl CppCompat.ULongLong64 as Copy {
  fn Op[self: Self]() -> Self = "primitive_copy";
}

// Conversions.

impl IntLiteral() as ImplicitAs(CppCompat.Long32) {
  fn Convert[self: Self]() -> CppCompat.Long32 = "int.convert_checked";
}

// TODO: Remove this once `ImplicitAs` extends `As`.
impl IntLiteral() as As(CppCompat.Long32) {
  fn Convert[self: Self]() -> CppCompat.Long32 = "int.convert_checked";
}

final impl CppCompat.Long32 as ImplicitAs(IntLiteral()) {
  fn Convert[self: Self]() -> IntLiteral() = "int.convert_checked";
}

// TODO: ImplicitAs from Int(N) to Long32 if N < 32.
impl i32 as ImplicitAs(CppCompat.Long32) {
  fn Convert[self: Self]() -> CppCompat.Long32 = "int.convert";
}

// TODO: Generalize ImplicitAs from Long32 to Int(N) for all N > 32.
// N == 32 is explicitly skipped, as Long32 is considered to
// be between i32 and i33 (details:
// https://github.com/carbon-language/carbon-lang/issues/6275#issuecomment-3488010485).
// This follows the rule that implicit conversions exist for all iN -> iM if
// M > N. Otherwise, operations like i32 + Cpp.long result in i32, instead of
// Cpp.long, which differs from the result for Cpp.long + i32 -> Cpp.long, as
// the best match for the left operand is being selected.
final impl CppCompat.Long32 as ImplicitAs(i64) {
  fn Convert[self: Self]() -> i64 = "int.convert";
}

impl IntLiteral() as ImplicitAs(CppCompat.ULong32) {
  fn Convert[self: Self]() -> CppCompat.ULong32 = "int.convert_checked";
}

// TODO: Remove this once `ImplicitAs` extends `As`.
impl IntLiteral() as As(CppCompat.ULong32) {
  fn Convert[self: Self]() -> CppCompat.ULong32 = "int.convert_checked";
}

final impl CppCompat.ULong32 as ImplicitAs(IntLiteral()) {
  fn Convert[self: Self]() -> IntLiteral() = "int.convert_checked";
}

// TODO: ImplicitAs from UInt(N) to ULong32 if N < 32.
impl u32 as ImplicitAs(CppCompat.ULong32) {
  fn Convert[self: Self]() -> CppCompat.ULong32 = "int.convert";
}

// TODO: ImplicitAs from ULong32 to UInt(N) if N > 32.
final impl CppCompat.ULong32 as ImplicitAs(u32) {
  fn Convert[self: Self]() -> u32 = "int.convert";
}

impl IntLiteral() as ImplicitAs(CppCompat.LongLong64) {
  fn Convert[self: Self]() -> CppCompat.LongLong64 = "int.convert_checked";
}

// TODO: Remove this once `ImplicitAs` extends `As`.
impl IntLiteral() as As(CppCompat.LongLong64) {
  fn Convert[self: Self]() -> CppCompat.LongLong64 = "int.convert_checked";
}

final impl CppCompat.LongLong64 as ImplicitAs(IntLiteral()) {
  fn Convert[self: Self]() -> IntLiteral() = "int.convert_checked";
}

// TODO: ImplicitAs from Int(N) to LongLong64 if N < 64.
impl i64 as ImplicitAs(CppCompat.LongLong64) {
  fn Convert[self: Self]() -> CppCompat.LongLong64 = "int.convert";
}

// TODO: ImplicitAs from LongLong64 to Int(N) if N > 64.
// N == 64 is explicitly skipped as LongLong64 is considered to
// be between i64 and i65 (details:
// https://github.com/carbon-language/carbon-lang/issues/6275#issuecomment-3488010485).
final impl CppCompat.LongLong64 as ImplicitAs(i128) {
  fn Convert[self: Self]() -> i128 = "int.convert";
}

impl IntLiteral() as ImplicitAs(CppCompat.ULongLong64) {
  fn Convert[self: Self]() -> CppCompat.ULongLong64 = "int.convert_checked";
}

// TODO: Remove this once `ImplicitAs` extends `As`.
impl IntLiteral() as As(CppCompat.ULongLong64) {
  fn Convert[self: Self]() -> CppCompat.ULongLong64 = "int.convert_checked";
}

final impl CppCompat.ULongLong64 as ImplicitAs(IntLiteral()) {
  fn Convert[self: Self]() -> IntLiteral() = "int.convert_checked";
}

// TODO: ImplicitAs from UInt(N) to ULongLong64 if N < 64.
impl u64 as ImplicitAs(CppCompat.ULongLong64) {
  fn Convert[self: Self]() -> CppCompat.ULongLong64 = "int.convert";
}

// TODO: ImplicitAs from ULongLong64 to UInt(N) if N > 64.
final impl CppCompat.ULongLong64 as ImplicitAs(u64) {
  fn Convert[self: Self]() -> u64 = "int.convert";
}

// TODO: As from Long32 to Int(N) if N < 32.
// TODO: As from ULong32 to UInt(N) if N < 32.
// TODO: As from LongLong64 to Int(N) if N < 64.
// TODO: As from ULongLong64 to UInt(N) if N < 64.
// TODO: As from Int(N) to Long32 if N > 32.
// TODO: As from UInt(N) to ULong32 if N > 32.
// TODO: As from Int(N) to LongLong64 if N > 64.
// TODO: As from UInt(N) to ULongLong64 if N > 64.

// Comparisons.

// - Homogeneous.

// - CppCompat.T vs CppCompat.T

final impl CppCompat.Long32 as EqWith(CppCompat.Long32) {
  fn Equal[self: Self](other: Self) -> bool = "int.eq";
  fn NotEqual[self: Self](other: Self) -> bool = "int.neq";
}

final impl CppCompat.Long32 as OrderedWith(CppCompat.Long32) {
  fn Less[self: Self](other: Self) -> bool = "int.less";
  fn LessOrEquivalent[self: Self](other: Self) -> bool = "int.less_eq";
  fn Greater[self: Self](other: Self) -> bool = "int.greater";
  fn GreaterOrEquivalent[self: Self](other: Self) -> bool = "int.greater_eq";
}

final impl CppCompat.LongLong64 as EqWith(CppCompat.LongLong64) {
  fn Equal[self: Self](other: Self) -> bool = "int.eq";
  fn NotEqual[self: Self](other: Self) -> bool = "int.neq";
}

final impl CppCompat.LongLong64 as OrderedWith(CppCompat.LongLong64) {
  fn Less[self: Self](other: Self) -> bool = "int.less";
  fn LessOrEquivalent[self: Self](other: Self) -> bool = "int.less_eq";
  fn Greater[self: Self](other: Self) -> bool = "int.greater";
  fn GreaterOrEquivalent[self: Self](other: Self) -> bool = "int.greater_eq";
}

// - Heterogeneous.

// - CppCompat.T on left-hand side.

impl forall [T:! ImplicitAs(CppCompat.Long32)] CppCompat.Long32 as EqWith(T) {
  fn Equal[self: Self](other: Self) -> bool = "int.eq";
  fn NotEqual[self: Self](other: Self) -> bool = "int.neq";
}

impl forall [T:! ImplicitAs(CppCompat.Long32)] CppCompat.Long32 as OrderedWith(T) {
  fn Less[self: Self](other: Self) -> bool = "int.less";
  fn LessOrEquivalent[self: Self](other: Self) -> bool = "int.less_eq";
  fn Greater[self: Self](other: Self) -> bool = "int.greater";
  fn GreaterOrEquivalent[self: Self](other: Self) -> bool = "int.greater_eq";
}

impl forall [T:! ImplicitAs(CppCompat.LongLong64)]
    CppCompat.LongLong64 as EqWith(T) {
  fn Equal[self: Self](other: Self) -> bool = "int.eq";
  fn NotEqual[self: Self](other: Self) -> bool = "int.neq";
}

impl forall [T:! ImplicitAs(CppCompat.LongLong64)]
    CppCompat.LongLong64 as OrderedWith(T) {
  fn Less[self: Self](other: Self) -> bool = "int.less";
  fn LessOrEquivalent[self: Self](other: Self) -> bool = "int.less_eq";
  fn Greater[self: Self](other: Self) -> bool = "int.greater";
  fn GreaterOrEquivalent[self: Self](other: Self) -> bool = "int.greater_eq";
}

// - CppCompat.T on right-hand side.

impl forall [T:! ImplicitAs(CppCompat.Long32)] T as EqWith(CppCompat.Long32) {
  fn Equal[self: CppCompat.Long32](other: CppCompat.Long32) -> bool = "int.eq";
  fn NotEqual[self: CppCompat.Long32](other: CppCompat.Long32) -> bool = "int.neq";
}

impl forall [T:! ImplicitAs(CppCompat.Long32)] T as OrderedWith(CppCompat.Long32) {
  fn Less[self: CppCompat.Long32](other: CppCompat.Long32) -> bool = "int.less";
  fn LessOrEquivalent[self: CppCompat.Long32](other: CppCompat.Long32) -> bool = "int.less_eq";
  fn Greater[self: CppCompat.Long32](other: CppCompat.Long32) -> bool = "int.greater";
  fn GreaterOrEquivalent[self: CppCompat.Long32](other: CppCompat.Long32) -> bool = "int.greater_eq";
}

impl forall [T:! ImplicitAs(CppCompat.LongLong64)]
    T as EqWith(CppCompat.LongLong64) {
  fn Equal[self: CppCompat.LongLong64](
    other: CppCompat.LongLong64) -> bool = "int.eq";
  fn NotEqual[self: CppCompat.LongLong64](
    other: CppCompat.LongLong64) -> bool = "int.neq";
}

impl forall [T:! ImplicitAs(CppCompat.LongLong64)]
    T as OrderedWith(CppCompat.LongLong64) {
  fn Less[self: CppCompat.LongLong64](
    other: CppCompat.LongLong64) -> bool = "int.less";
  fn LessOrEquivalent[self: CppCompat.LongLong64](
    other: CppCompat.LongLong64) -> bool = "int.less_eq";
  fn Greater[self: CppCompat.LongLong64](
    other: CppCompat.LongLong64) -> bool = "int.greater";
  fn GreaterOrEquivalent[self: CppCompat.LongLong64](
    other: CppCompat.LongLong64) -> bool = "int.greater_eq";
}

// TODO: Comparisons for ULong32, ULongLong64.

// Arithmetic.

// - Homogeneous.

// - CppCompat.Long32

final impl CppCompat.Long32 as Negate where .Result = Self {
  fn Op[self: Self]() -> Self = "int.snegate";
}

final impl CppCompat.Long32 as AddWith(Self) where .Result = Self {
  fn Op[self: Self](other: Self) -> Self = "int.sadd";
}

final impl CppCompat.Long32 as SubWith(Self) where .Result = Self {
  fn Op[self: Self](other: Self) -> Self = "int.ssub";
}

final impl CppCompat.Long32 as MulWith(Self) where .Result = Self {
  fn Op[self: Self](other: Self) -> Self = "int.smul";
}

final impl CppCompat.Long32 as DivWith(Self) where .Result = Self {
  fn Op[self: Self](other: Self) -> Self = "int.sdiv";
}

final impl CppCompat.Long32 as ModWith(Self) where .Result = Self {
  fn Op[self: Self](other: Self) -> Self = "int.smod";
}

// - CppCompat.LongLong64

final impl CppCompat.LongLong64 as Negate where .Result = Self {
  fn Op[self: Self]() -> Self = "int.snegate";
}

final impl CppCompat.LongLong64 as AddWith(Self) where .Result = Self {
  fn Op[self: Self](other: Self) -> Self = "int.sadd";
}

final impl CppCompat.LongLong64 as SubWith(Self) where .Result = Self {
  fn Op[self: Self](other: Self) -> Self = "int.ssub";
}

final impl CppCompat.LongLong64 as MulWith(Self) where .Result = Self {
  fn Op[self: Self](other: Self) -> Self = "int.smul";
}

final impl CppCompat.LongLong64 as DivWith(Self) where .Result = Self {
  fn Op[self: Self](other: Self) -> Self = "int.sdiv";
}

final impl CppCompat.LongLong64 as ModWith(Self) where .Result = Self {
  fn Op[self: Self](other: Self) -> Self = "int.smod";
}

// - Heterogeneous.

// - CppCompat.Long32 on left-hand side.

impl forall [T:! ImplicitAs(CppCompat.Long32)]
    CppCompat.Long32 as AddWith(T) where .Result = CppCompat.Long32 {
  fn Op[self: CppCompat.Long32](other: CppCompat.Long32) -> CppCompat.Long32 = "int.sadd";
}

impl forall [T:! ImplicitAs(CppCompat.Long32)]
    CppCompat.Long32 as SubWith(T) where .Result = CppCompat.Long32 {
  fn Op[self: CppCompat.Long32](other: CppCompat.Long32) -> CppCompat.Long32 = "int.ssub";
}

impl forall [T:! ImplicitAs(CppCompat.Long32)]
    CppCompat.Long32 as MulWith(T) where .Result = CppCompat.Long32 {
  fn Op[self: CppCompat.Long32](other: CppCompat.Long32) -> CppCompat.Long32 = "int.smul";
}

impl forall [T:! ImplicitAs(CppCompat.Long32)]
    CppCompat.Long32 as DivWith(T) where .Result = CppCompat.Long32 {
  fn Op[self: CppCompat.Long32](other: CppCompat.Long32) -> CppCompat.Long32 = "int.sdiv";
}

impl forall [T:! ImplicitAs(CppCompat.Long32)]
    CppCompat.Long32 as ModWith(T) where .Result = CppCompat.Long32 {
  fn Op[self: CppCompat.Long32](other: CppCompat.Long32) -> CppCompat.Long32 = "int.smod";
}

// - CppCompat.LongLong64 on left-hand side.

impl forall [T:! ImplicitAs(CppCompat.LongLong64)]
    CppCompat.LongLong64 as AddWith(T) where .Result = CppCompat.LongLong64 {
  fn Op[self: CppCompat.LongLong64](
    other: CppCompat.LongLong64) -> CppCompat.LongLong64 = "int.sadd";
}

impl forall [T:! ImplicitAs(CppCompat.LongLong64)]
    CppCompat.LongLong64 as SubWith(T) where .Result = CppCompat.LongLong64 {
  fn Op[self: CppCompat.LongLong64](
    other: CppCompat.LongLong64) -> CppCompat.LongLong64 = "int.ssub";
}

impl forall [T:! ImplicitAs(CppCompat.LongLong64)]
    CppCompat.LongLong64 as MulWith(T) where .Result = CppCompat.LongLong64 {
  fn Op[self: CppCompat.LongLong64](
    other: CppCompat.LongLong64) -> CppCompat.LongLong64 = "int.smul";
}

impl forall [T:! ImplicitAs(CppCompat.LongLong64)]
    CppCompat.LongLong64 as DivWith(T) where .Result = CppCompat.LongLong64 {
  fn Op[self: CppCompat.LongLong64](
    other: CppCompat.LongLong64) -> CppCompat.LongLong64 = "int.sdiv";
}

impl forall [T:! ImplicitAs(CppCompat.LongLong64)]
    CppCompat.LongLong64 as ModWith(T) where .Result = CppCompat.LongLong64 {
  fn Op[self: CppCompat.LongLong64](
    other: CppCompat.LongLong64) -> CppCompat.LongLong64 = "int.smod";
}

// - CppCompat.Long32 on right-hand side.

impl forall [T:! ImplicitAs(CppCompat.Long32)]
    T as AddWith(CppCompat.Long32) where .Result = CppCompat.Long32 {
  fn Op[self: CppCompat.Long32](other: CppCompat.Long32) -> CppCompat.Long32 = "int.sadd";
}

impl forall [T:! ImplicitAs(CppCompat.Long32)]
    T as SubWith(CppCompat.Long32) where .Result = CppCompat.Long32 {
  fn Op[self: CppCompat.Long32](other: CppCompat.Long32) -> CppCompat.Long32 = "int.ssub";
}

impl forall [T:! ImplicitAs(CppCompat.Long32)]
    T as MulWith(CppCompat.Long32) where .Result = CppCompat.Long32 {
  fn Op[self: CppCompat.Long32](other: CppCompat.Long32) -> CppCompat.Long32 = "int.smul";
}

impl forall [T:! ImplicitAs(CppCompat.Long32)]
    T as DivWith(CppCompat.Long32) where .Result = CppCompat.Long32 {
  fn Op[self: CppCompat.Long32](other: CppCompat.Long32) -> CppCompat.Long32 = "int.sdiv";
}

impl forall [T:! ImplicitAs(CppCompat.Long32)]
    T as ModWith(CppCompat.Long32) where .Result = CppCompat.Long32 {
  fn Op[self: CppCompat.Long32](other: CppCompat.Long32) -> CppCompat.Long32 = "int.smod";
}

// - CppCompat.LongLong64 on right-hand side.

impl forall [T:! ImplicitAs(CppCompat.LongLong64)]
    T as AddWith(CppCompat.LongLong64) where .Result = CppCompat.LongLong64 {
  fn Op[self: CppCompat.LongLong64](
    other: CppCompat.LongLong64) -> CppCompat.LongLong64 = "int.sadd";
}

impl forall [T:! ImplicitAs(CppCompat.LongLong64)]
    T as SubWith(CppCompat.LongLong64) where .Result = CppCompat.LongLong64 {
  fn Op[self: CppCompat.LongLong64](
    other: CppCompat.LongLong64) -> CppCompat.LongLong64 = "int.ssub";
}

impl forall [T:! ImplicitAs(CppCompat.LongLong64)]
    T as MulWith(CppCompat.LongLong64) where .Result = CppCompat.LongLong64 {
  fn Op[self: CppCompat.LongLong64](
    other: CppCompat.LongLong64) -> CppCompat.LongLong64 = "int.smul";
}

impl forall [T:! ImplicitAs(CppCompat.LongLong64)]
    T as DivWith(CppCompat.LongLong64) where .Result = CppCompat.LongLong64 {
  fn Op[self: CppCompat.LongLong64](
    other: CppCompat.LongLong64) -> CppCompat.LongLong64 = "int.sdiv";
}

impl forall [T:! ImplicitAs(CppCompat.LongLong64)]
    T as ModWith(CppCompat.LongLong64) where .Result = CppCompat.LongLong64 {
  fn Op[self: CppCompat.LongLong64](
    other: CppCompat.LongLong64) -> CppCompat.LongLong64 = "int.smod";
}

// TODO: ULong32, ULongLong64: arithmetic operators.

// Bitwise operators.

// - Homogeneous.

// - CppCompat.Long32

final impl CppCompat.Long32 as BitComplement where .Result = Self {
  fn Op[self: Self]() -> Self = "int.complement";
}

final impl CppCompat.Long32 as BitAndWith(Self) where .Result = Self {
  fn Op[self: Self](other: Self) -> Self = "int.and";
}

final impl CppCompat.Long32 as BitOrWith(Self) where .Result = Self {
  fn Op[self: Self](other: Self) -> Self = "int.or";
}

final impl CppCompat.Long32 as BitXorWith(Self) where .Result = Self {
  fn Op[self: Self](other: Self) -> Self = "int.xor";
}

final impl CppCompat.Long32 as LeftShiftWith(Self) where .Result = Self {
  fn Op[self: Self](other: Self) -> Self = "int.left_shift";
}

final impl CppCompat.Long32 as RightShiftWith(Self) where .Result = Self {
  fn Op[self: Self](other: Self) -> Self = "int.right_shift";
}

// - CppCompat.LongLong64

final impl CppCompat.LongLong64 as BitComplement where .Result = Self {
  fn Op[self: Self]() -> Self = "int.complement";
}

final impl CppCompat.LongLong64 as BitAndWith(Self) where .Result = Self {
  fn Op[self: Self](other: Self) -> Self = "int.and";
}

final impl CppCompat.LongLong64 as BitOrWith(Self) where .Result = Self {
  fn Op[self: Self](other: Self) -> Self = "int.or";
}

final impl CppCompat.LongLong64 as BitXorWith(Self) where .Result = Self {
  fn Op[self: Self](other: Self) -> Self = "int.xor";
}

final impl CppCompat.LongLong64 as LeftShiftWith(Self) where .Result = Self {
  fn Op[self: Self](other: Self) -> Self = "int.left_shift";
}

final impl CppCompat.LongLong64 as RightShiftWith(Self) where .Result = Self {
  fn Op[self: Self](other: Self) -> Self = "int.right_shift";
}

// - Heterogeneous.

// - CppCompat.Long32 on left-hand side.

impl forall [T:! ImplicitAs(CppCompat.Long32)]
    CppCompat.Long32 as BitAndWith(T) where .Result = CppCompat.Long32 {
  fn Op[self: CppCompat.Long32](
    other:CppCompat.Long32) -> CppCompat.Long32 = "int.and";
}

impl forall [T:! ImplicitAs(CppCompat.Long32)]
    CppCompat.Long32 as BitOrWith(T) where .Result = CppCompat.Long32 {
  fn Op[self: CppCompat.Long32](
    other: CppCompat.Long32) -> CppCompat.Long32 = "int.or";
}

impl forall [T:! ImplicitAs(CppCompat.Long32)]
    CppCompat.Long32 as BitXorWith(T) where .Result = CppCompat.Long32 {
  fn Op[self: CppCompat.Long32](
    other: CppCompat.Long32) -> CppCompat.Long32 = "int.xor";
}

impl forall [T:! ImplicitAs(CppCompat.Long32)]
    CppCompat.Long32 as LeftShiftWith(T) where .Result = CppCompat.Long32 {
  fn Op[self: CppCompat.Long32](
    other: CppCompat.Long32) -> CppCompat.Long32 = "int.left_shift";
}

impl forall [T:! ImplicitAs(CppCompat.Long32)]
    CppCompat.Long32 as RightShiftWith(T) where .Result = CppCompat.Long32 {
  fn Op[self: CppCompat.Long32](
    other: CppCompat.Long32) -> CppCompat.Long32 = "int.right_shift";
}

// - CppCompat.LongLong64 on left-hand side.

impl forall [T:! ImplicitAs(CppCompat.LongLong64)]
    CppCompat.LongLong64 as BitAndWith(T) where .Result = CppCompat.LongLong64 {
  fn Op[self: CppCompat.LongLong64](
    other:CppCompat.LongLong64) -> CppCompat.LongLong64 = "int.and";
}

impl forall [T:! ImplicitAs(CppCompat.LongLong64)]
    CppCompat.LongLong64 as BitOrWith(T) where .Result = CppCompat.LongLong64 {
  fn Op[self: CppCompat.LongLong64](
    other: CppCompat.LongLong64) -> CppCompat.LongLong64 = "int.or";
}

impl forall [T:! ImplicitAs(CppCompat.LongLong64)]
    CppCompat.LongLong64 as BitXorWith(T) where .Result = CppCompat.LongLong64 {
  fn Op[self: CppCompat.LongLong64](
    other: CppCompat.LongLong64) -> CppCompat.LongLong64 = "int.xor";
}

impl forall [T:! ImplicitAs(CppCompat.LongLong64)]
    CppCompat.LongLong64 as LeftShiftWith(T) where .Result = CppCompat.LongLong64 {
  fn Op[self: CppCompat.LongLong64](
    other: CppCompat.LongLong64) -> CppCompat.LongLong64 = "int.left_shift";
}

impl forall [T:! ImplicitAs(CppCompat.LongLong64)]
    CppCompat.LongLong64 as RightShiftWith(T) where .Result = CppCompat.LongLong64 {
  fn Op[self: CppCompat.LongLong64](
    other: CppCompat.LongLong64) -> CppCompat.LongLong64 = "int.right_shift";
}

// - CppCompat.Long32 on right-hand side.

impl forall [T:! ImplicitAs(CppCompat.Long32)]
    T as BitAndWith(CppCompat.Long32) where .Result = CppCompat.Long32 {
  fn Op[self: CppCompat.Long32](
    other: CppCompat.Long32) -> CppCompat.Long32 = "int.and";
}

impl forall [T:! ImplicitAs(CppCompat.Long32)]
    T as BitOrWith(CppCompat.Long32) where .Result = CppCompat.Long32 {
  fn Op[self: CppCompat.Long32](
    other: CppCompat.Long32) -> CppCompat.Long32 = "int.or";
}

impl forall [T:! ImplicitAs(CppCompat.Long32)]
    T as BitXorWith(CppCompat.Long32) where .Result = CppCompat.Long32 {
  fn Op[self: CppCompat.Long32](
    other: CppCompat.Long32) -> CppCompat.Long32 = "int.xor";
}

impl forall [T:! ImplicitAs(CppCompat.Long32)]
    T as LeftShiftWith(CppCompat.Long32) where .Result = CppCompat.Long32 {
  fn Op[self: CppCompat.Long32](
    other: CppCompat.Long32) -> CppCompat.Long32 = "int.left_shift";
}

impl forall [T:! ImplicitAs(CppCompat.Long32)]
    T as RightShiftWith(CppCompat.Long32) where .Result = CppCompat.Long32 {
  fn Op[self: CppCompat.Long32](
    other: CppCompat.Long32) -> CppCompat.Long32 = "int.right_shift";
}

// - CppCompat.LongLong64 on right-hand side.

impl forall [T:! ImplicitAs(CppCompat.LongLong64)]
    T as BitAndWith(CppCompat.LongLong64) where .Result = CppCompat.LongLong64 {
  fn Op[self: CppCompat.LongLong64](
    other: CppCompat.LongLong64) -> CppCompat.LongLong64 = "int.and";
}

impl forall [T:! ImplicitAs(CppCompat.LongLong64)]
    T as BitOrWith(CppCompat.LongLong64) where .Result = CppCompat.LongLong64 {
  fn Op[self: CppCompat.LongLong64](
    other: CppCompat.LongLong64) -> CppCompat.LongLong64 = "int.or";
}

impl forall [T:! ImplicitAs(CppCompat.LongLong64)]
    T as BitXorWith(CppCompat.LongLong64) where .Result = CppCompat.LongLong64 {
  fn Op[self: CppCompat.LongLong64](
    other: CppCompat.LongLong64) -> CppCompat.LongLong64 = "int.xor";
}

impl forall [T:! ImplicitAs(CppCompat.LongLong64)]
    T as LeftShiftWith(CppCompat.LongLong64) where .Result = CppCompat.LongLong64 {
  fn Op[self: CppCompat.LongLong64](
    other: CppCompat.LongLong64) -> CppCompat.LongLong64 = "int.left_shift";
}

impl forall [T:! ImplicitAs(CppCompat.LongLong64)]
    T as RightShiftWith(CppCompat.LongLong64) where .Result = CppCompat.LongLong64 {
  fn Op[self: CppCompat.LongLong64](
    other: CppCompat.LongLong64) -> CppCompat.LongLong64 = "int.right_shift";
}

// TODO: ULong32, ULongLong64: bitwise operators.

// Compound assignments.

// Compound arithmetic assignments.

// - CppCompat.Long32

impl forall [U:! ImplicitAs(CppCompat.Long32)]
	  CppCompat.Long32 as AddAssignWith(U) {
  fn Op[ref self: Self](other: Self) = "int.sadd_assign";
}

impl forall [U:! ImplicitAs(CppCompat.Long32)]
	  CppCompat.Long32 as SubAssignWith(U) {
  fn Op[ref self: Self](other: Self) = "int.ssub_assign";
}

impl forall [U:! ImplicitAs(CppCompat.Long32)]
	  CppCompat.Long32 as MulAssignWith(U) {
  fn Op[ref self: Self](other: Self) = "int.smul_assign";
}

impl forall [U:! ImplicitAs(CppCompat.Long32)]
	  CppCompat.Long32 as DivAssignWith(U) {
  fn Op[ref self: Self](other: Self) = "int.sdiv_assign";
}

impl forall [U:! ImplicitAs(CppCompat.Long32)]
	  CppCompat.Long32 as ModAssignWith(U) {
  fn Op[ref self: Self](other: Self) = "int.smod_assign";
}

// - CppCompat.LongLong64

impl forall [U:! ImplicitAs(CppCompat.LongLong64)]
	  CppCompat.LongLong64 as AddAssignWith(U) {
  fn Op[ref self: Self](other: Self) = "int.sadd_assign";
}

impl forall [U:! ImplicitAs(CppCompat.LongLong64)]
	  CppCompat.LongLong64 as SubAssignWith(U) {
  fn Op[ref self: Self](other: Self) = "int.ssub_assign";
}

impl forall [U:! ImplicitAs(CppCompat.LongLong64)]
	  CppCompat.LongLong64 as MulAssignWith(U) {
  fn Op[ref self: Self](other: Self) = "int.smul_assign";
}

impl forall [U:! ImplicitAs(CppCompat.LongLong64)]
	  CppCompat.LongLong64 as DivAssignWith(U) {
  fn Op[ref self: Self](other: Self) = "int.sdiv_assign";
}

impl forall [U:! ImplicitAs(CppCompat.LongLong64)]
	  CppCompat.LongLong64 as ModAssignWith(U) {
  fn Op[ref self: Self](other: Self) = "int.smod_assign";
}

// Compound bitwise assignments.

// - CppCompat.Long32

impl forall [U:! ImplicitAs(CppCompat.Long32)]
	  CppCompat.Long32 as BitAndAssignWith(U) {
  fn Op[ref self: Self](other: Self) = "int.and_assign";
}

impl forall [U:! ImplicitAs(CppCompat.Long32)]
	  CppCompat.Long32 as BitOrAssignWith(U) {
  fn Op[ref self: Self](other: Self) = "int.or_assign";
}

impl forall [U:! ImplicitAs(CppCompat.Long32)]
	  CppCompat.Long32 as BitXorAssignWith(U) {
  fn Op[ref self: Self](other: Self) = "int.xor_assign";
}

impl forall [U:! ImplicitAs(CppCompat.Long32)]
	  CppCompat.Long32 as LeftShiftAssignWith(U) {
  fn Op[ref self: Self](other: Self) = "int.left_shift_assign";
}

impl forall [U:! ImplicitAs(CppCompat.Long32)]
	  CppCompat.Long32 as RightShiftAssignWith(U) {
  fn Op[ref self: Self](other: Self) = "int.right_shift_assign";
}

// - CppCompat.LongLong64

impl forall [U:! ImplicitAs(CppCompat.LongLong64)]
	  CppCompat.LongLong64 as BitAndAssignWith(U) {
  fn Op[ref self: Self](other: Self) = "int.and_assign";
}

impl forall [U:! ImplicitAs(CppCompat.LongLong64)]
	  CppCompat.LongLong64 as BitOrAssignWith(U) {
  fn Op[ref self: Self](other: Self) = "int.or_assign";
}

impl forall [U:! ImplicitAs(CppCompat.LongLong64)]
	  CppCompat.LongLong64 as BitXorAssignWith(U) {
  fn Op[ref self: Self](other: Self) = "int.xor_assign";
}

impl forall [U:! ImplicitAs(CppCompat.LongLong64)]
	  CppCompat.LongLong64 as LeftShiftAssignWith(U) {
  fn Op[ref self: Self](other: Self) = "int.left_shift_assign";
}

impl forall [U:! ImplicitAs(CppCompat.LongLong64)]
	  CppCompat.LongLong64 as RightShiftAssignWith(U) {
  fn Op[ref self: Self](other: Self) = "int.right_shift_assign";
}

// TODO: ULong32, ULongLong64: compound assignments.

// Increment and decrement.

// - CppCompat.Long32

impl CppCompat.Long32 as Dec {
  fn Op[ref self: Self]() {
    self -= (1 as CppCompat.Long32);
  }
}

impl CppCompat.Long32 as Inc {
  fn Op[ref self: Self]() {
    self += (1 as CppCompat.Long32);
  }
}

// - CppCompat.LongLong64

impl CppCompat.LongLong64 as Dec {
  fn Op[ref self: Self]() {
    self -= (1 as CppCompat.LongLong64);
  }
}

impl CppCompat.LongLong64 as Inc {
  fn Op[ref self: Self]() {
    self += (1 as CppCompat.LongLong64);
  }
}

// TODO: Increment/decrement operations for ULong32, ULongLong64.
