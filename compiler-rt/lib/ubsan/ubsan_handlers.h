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

/// \brief Handle a runtime type check failure, caused by either a misaligned
/// pointer, a null pointer, or a pointer to insufficient storage for the
/// type.
extern "C" void __ubsan_handle_type_mismatch(TypeMismatchData *Data,
                                             ValueHandle Pointer);

struct OverflowData {
  SourceLocation Loc;
  const TypeDescriptor &Type;
};

/// \brief Handle a signed integer addition overflow.
extern "C" void __ubsan_handle_add_overflow(OverflowData *Data,
                                            ValueHandle LHS,
                                            ValueHandle RHS);
/// \brief Handle a signed integer subtraction overflow.
extern "C" void __ubsan_handle_sub_overflow(OverflowData *Data,
                                            ValueHandle LHS,
                                            ValueHandle RHS);
/// \brief Handle a signed integer multiplication overflow.
extern "C" void __ubsan_handle_mul_overflow(OverflowData *Data,
                                            ValueHandle LHS,
                                            ValueHandle RHS);
/// \brief Handle a signed integer overflow for a unary negate operator.
extern "C" void __ubsan_handle_negate_overflow(OverflowData *Data,
                                               ValueHandle OldVal);
/// \brief Handle an INT_MIN/-1 overflow or division by zero.
extern "C" void __ubsan_handle_divrem_overflow(OverflowData *Data,
                                               ValueHandle LHS,
                                               ValueHandle RHS);

struct ShiftOutOfBoundsData {
  SourceLocation Loc;
  const TypeDescriptor &LHSType;
  const TypeDescriptor &RHSType;
};

/// \brief Handle a shift where the RHS is out of bounds or a left shift where
/// the LHS is negative or overflows.
extern "C" void __ubsan_handle_shift_out_of_bounds(ShiftOutOfBoundsData *Data,
                                                   ValueHandle LHS,
                                                   ValueHandle RHS);

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
extern "C" void __ubsan_handle_vla_bound_not_positive(VLABoundData *Data,
                                                      ValueHandle Bound);

}

#endif // UBSAN_HANDLERS_H
