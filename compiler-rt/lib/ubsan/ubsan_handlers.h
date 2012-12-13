//===-- ubsan_handlers.h ----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Entry points to the runtime library for Clang's undefined behavior sanitizer.
//
//===----------------------------------------------------------------------===//
#ifndef UBSAN_HANDLERS_H
#define UBSAN_HANDLERS_H

#include "ubsan_value.h"

namespace __ubsan {

struct TypeMismatchData {
  SourceLocation Loc;
  const TypeDescriptor &Type;
  uptr Alignment;
  unsigned char TypeCheckKind;
};

#define RECOVERABLE(checkname, ...) \
  extern "C" void __ubsan_handle_ ## checkname( __VA_ARGS__ ); \
  extern "C" void __ubsan_handle_ ## checkname ## _abort( __VA_ARGS__ );

/// \brief Handle a runtime type check failure, caused by either a misaligned
/// pointer, a null pointer, or a pointer to insufficient storage for the
/// type.
RECOVERABLE(type_mismatch, TypeMismatchData *Data, ValueHandle Pointer)

struct OverflowData {
  SourceLocation Loc;
  const TypeDescriptor &Type;
};

/// \brief Handle an integer addition overflow.
RECOVERABLE(add_overflow, OverflowData *Data, ValueHandle LHS, ValueHandle RHS)

/// \brief Handle an integer subtraction overflow.
RECOVERABLE(sub_overflow, OverflowData *Data, ValueHandle LHS, ValueHandle RHS)

/// \brief Handle an integer multiplication overflow.
RECOVERABLE(mul_overflow, OverflowData *Data, ValueHandle LHS, ValueHandle RHS)

/// \brief Handle a signed integer overflow for a unary negate operator.
RECOVERABLE(negate_overflow, OverflowData *Data, ValueHandle OldVal)

/// \brief Handle an INT_MIN/-1 overflow or division by zero.
RECOVERABLE(divrem_overflow, OverflowData *Data,
            ValueHandle LHS, ValueHandle RHS)

struct ShiftOutOfBoundsData {
  SourceLocation Loc;
  const TypeDescriptor &LHSType;
  const TypeDescriptor &RHSType;
};

/// \brief Handle a shift where the RHS is out of bounds or a left shift where
/// the LHS is negative or overflows.
RECOVERABLE(shift_out_of_bounds, ShiftOutOfBoundsData *Data,
            ValueHandle LHS, ValueHandle RHS)

struct UnreachableData {
  SourceLocation Loc;
};

/// \brief Handle a __builtin_unreachable which is reached.
extern "C" void __ubsan_handle_builtin_unreachable(UnreachableData *Data);
/// \brief Handle reaching the end of a value-returning function.
extern "C" void __ubsan_handle_missing_return(UnreachableData *Data);

struct VLABoundData {
  SourceLocation Loc;
  const TypeDescriptor &Type;
};

/// \brief Handle a VLA with a non-positive bound.
RECOVERABLE(vla_bound_not_positive, VLABoundData *Data, ValueHandle Bound)

struct FloatCastOverflowData {
  // FIXME: SourceLocation Loc;
  const TypeDescriptor &FromType;
  const TypeDescriptor &ToType;
};

/// \brief Handle overflow in a conversion to or from a floating-point type.
RECOVERABLE(float_cast_overflow, FloatCastOverflowData *Data, ValueHandle From)

struct InvalidValueData {
  // FIXME: SourceLocation Loc;
  const TypeDescriptor &Type;
};

/// \brief Handle a load of an invalid value for the type.
RECOVERABLE(load_invalid_value, InvalidValueData *Data, ValueHandle Val)

}

#endif // UBSAN_HANDLERS_H
