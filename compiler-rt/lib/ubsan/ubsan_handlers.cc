//===-- ubsan_report.cc ---------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Error logging entry points for the UBSan runtime.
//
//===----------------------------------------------------------------------===//

#include "ubsan_handlers.h"
#include "ubsan_diag.h"

#include "sanitizer_common/sanitizer_common.h"

using namespace __sanitizer;
using namespace __ubsan;

NORETURN void __sanitizer::Die() {
  __builtin_trap();
}

NORETURN void __sanitizer::CheckFailed(const char *File, int Line,
                                       const char *Cond, u64 V1, u64 V2) {
  Diag(SourceLocation(File, Line, 0),
       "CHECK failed: %0 (with values %1 and %2)")
    << Cond << V1 << V2;
  Die();
}

void __ubsan::__ubsan_handle_type_mismatch(TypeMismatchData *Data,
                                           ValueHandle Pointer) {
  const char *TypeCheckKinds[] = {
    "load of", "store to", "reference binding to", "member access within",
    "member call on"
  };
  if (!Pointer)
    Diag(Data->Loc, "%0 null pointer of type %1")
      << TypeCheckKinds[Data->TypeCheckKind] << Data->Type;
  else if (Data->Alignment && (Pointer & (Data->Alignment - 1)))
    Diag(Data->Loc, "%0 misaligned address %1 for type %3, "
                    "which requires %2 byte alignment")
      << TypeCheckKinds[Data->TypeCheckKind] << (void*)Pointer
      << Data->Alignment << Data->Type;
  else
    Diag(Data->Loc, "%0 address %1 with insufficient space "
                    "for an object of type %2")
      << TypeCheckKinds[Data->TypeCheckKind] << (void*)Pointer << Data->Type;
  Die();
}

/// \brief Common diagnostic emission for various forms of signed overflow.
template<typename T> static void HandleSignedOverflow(OverflowData *Data,
                                                      ValueHandle LHS,
                                                      const char *Operator,
                                                      T RHS) {
  Diag(Data->Loc, "signed integer overflow: "
                  "%0 %1 %2 cannot be represented in type %3")
    << Value(Data->Type, LHS) << Operator << RHS << Data->Type;
  Die();
}

void __ubsan::__ubsan_handle_add_overflow(OverflowData *Data,
                                          ValueHandle LHS, ValueHandle RHS) {
  HandleSignedOverflow(Data, LHS, "+", Value(Data->Type, RHS));
}

void __ubsan::__ubsan_handle_sub_overflow(OverflowData *Data,
                                          ValueHandle LHS, ValueHandle RHS) {
  HandleSignedOverflow(Data, LHS, "-", Value(Data->Type, RHS));
}

void __ubsan::__ubsan_handle_mul_overflow(OverflowData *Data,
                                          ValueHandle LHS, ValueHandle RHS) {
  HandleSignedOverflow(Data, LHS, "*", Value(Data->Type, RHS));
}

void __ubsan::__ubsan_handle_negate_overflow(OverflowData *Data,
                                             ValueHandle OldVal) {
  Diag(Data->Loc, "negation of %0 cannot be represented in type %1; "
                  "cast to an unsigned type to negate this value to itself")
    << Value(Data->Type, OldVal) << Data->Type;
  Die();
}

void __ubsan::__ubsan_handle_divrem_overflow(OverflowData *Data,
                                             ValueHandle LHS, ValueHandle RHS) {
  Value LHSVal(Data->Type, LHS);
  Value RHSVal(Data->Type, RHS);
  if (RHSVal.isMinusOne())
    Diag(Data->Loc, "division of %0 by -1 cannot be represented in type %1")
      << LHSVal << Data->Type;
  else
    Diag(Data->Loc, "division by zero");
  Die();
}

void __ubsan::__ubsan_handle_shift_out_of_bounds(ShiftOutOfBoundsData *Data,
                                                 ValueHandle LHS,
                                                 ValueHandle RHS) {
  Value LHSVal(Data->LHSType, LHS);
  Value RHSVal(Data->RHSType, RHS);
  if (RHSVal.isNegative())
    Diag(Data->Loc, "shift exponent %0 is negative") << RHSVal;
  else if (RHSVal.getPositiveIntValue() >= Data->LHSType.getIntegerBitWidth())
    Diag(Data->Loc, "shift exponent %0 is too large for %1-bit type %2")
      << RHSVal << Data->LHSType.getIntegerBitWidth() << Data->LHSType;
  else if (LHSVal.isNegative())
    Diag(Data->Loc, "left shift of negative value %0") << LHSVal;
  else
    Diag(Data->Loc, "left shift of %0 by %1 places cannot be represented "
                    "in type %2") << LHSVal << RHSVal << Data->LHSType;
  Die();
}

void __ubsan::__ubsan_handle_builtin_unreachable(UnreachableData *Data) {
  Diag(Data->Loc, "execution reached a __builtin_unreachable() call");
  Die();
}

void __ubsan::__ubsan_handle_missing_return(UnreachableData *Data) {
  Diag(Data->Loc, "execution reached the end of a value-returning function "
                  "without returning a value");
  Die();
}

void __ubsan::__ubsan_handle_vla_bound_not_positive(VLABoundData *Data,
                                                    ValueHandle Bound) {
  Diag(Data->Loc, "variable length array bound evaluates to "
                  "non-positive value %0")
    << Value(Data->Type, Bound);
  Die();
}

void __ubsan::__ubsan_handle_float_cast_overflow(FloatCastOverflowData *Data,
                                                 ValueHandle From) {
  Diag(SourceLocation(), "value %0 is outside the range of representable "
                         "values of type %2")
    << Value(Data->FromType, From) << Data->FromType << Data->ToType;
  Die();
}
