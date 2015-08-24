//===-- ubsan_checks.inc ----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// List of checks handled by UBSan runtime.
//
//===----------------------------------------------------------------------===//
#ifndef UBSAN_CHECK
# error "Define UBSAN_CHECK prior to including this file!"
#endif

// UBSAN_CHECK(Name, SummaryKind, FlagName)
// SummaryKind and FlagName should be string literals.

UBSAN_CHECK(GenericUB, "undefined-behavior", "-fsanitize=undefined")
UBSAN_CHECK(NullPointerUse, "null-pointer-use", "-fsanitize=null")
UBSAN_CHECK(MisalignedPointerUse, "misaligned-pointer-use",
            "-fsanitize=alignment")
UBSAN_CHECK(InsufficientObjectSize, "insufficient-object-size",
            "-fsanitize=object-size")
UBSAN_CHECK(SignedIntegerOverflow, "signed-integer-overflow",
            "-fsanitize=signed-integer-overflow")
UBSAN_CHECK(UnsignedIntegerOverflow, "unsigned-integer-overflow",
            "-fsanitize=unsigned-integer-overflow")
UBSAN_CHECK(IntegerDivideByZero, "integer-divide-by-zero",
            "-fsanitize=integer-divide-by-zero")
UBSAN_CHECK(FloatDivideByZero, "float-divide-by-zero",
            "-fsanitize=float-divide-by-zero")
UBSAN_CHECK(InvalidShiftBase, "invalid-shift-base", "-fsanitize=shift-base")
UBSAN_CHECK(InvalidShiftExponent, "invalid-shift-exponent",
            "-fsanitize=shift-exponent")
UBSAN_CHECK(OutOfBoundsIndex, "out-of-bounds-index", "-fsanitize=bounds")
UBSAN_CHECK(UnreachableCall, "unreachable-call", "-fsanitize=unreachable")
UBSAN_CHECK(MissingReturn, "missing-return", "-fsanitize=return")
UBSAN_CHECK(NonPositiveVLAIndex, "non-positive-vla-index",
            "-fsanitize=vla-bound")
UBSAN_CHECK(FloatCastOverflow, "float-cast-overflow",
            "-fsanitize=float-cast-overflow")
UBSAN_CHECK(InvalidBoolLoad, "invalid-bool-load", "-fsanitize=bool")
UBSAN_CHECK(InvalidEnumLoad, "invalid-enum-load", "-fsanitize=enum")
UBSAN_CHECK(FunctionTypeMismatch, "function-type-mismatch",
            "-fsanitize=function")
UBSAN_CHECK(InvalidNullReturn, "invalid-null-return",
            "-fsanitize=returns-nonnull-attribute")
UBSAN_CHECK(InvalidNullArgument, "invalid-null-argument",
            "-fsanitize=nonnull-attribute")
UBSAN_CHECK(DynamicTypeMismatch, "dynamic-type-mismatch", "-fsanitize=vptr")
UBSAN_CHECK(CFIBadType, "cfi-bad-type", "-fsanitize=cfi")
