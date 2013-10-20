//===-- ubsan_handlers.cc -------------------------------------------------===//
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

namespace __ubsan {
  const char *TypeCheckKinds[] = {
    "load of", "store to", "reference binding to", "member access within",
    "member call on", "constructor call on", "downcast of", "downcast of"
  };
}

static void handleTypeMismatchImpl(TypeMismatchData *Data, ValueHandle Pointer,
                                   Location FallbackLoc) {
  Location Loc = Data->Loc.acquire();

  // Use the SourceLocation from Data to track deduplication, even if 'invalid'
  if (Loc.getSourceLocation().isDisabled())
    return;
  if (Data->Loc.isInvalid())
    Loc = FallbackLoc;

  if (!Pointer)
    Diag(Loc, DL_Error, "%0 null pointer of type %1")
      << TypeCheckKinds[Data->TypeCheckKind] << Data->Type;
  else if (Data->Alignment && (Pointer & (Data->Alignment - 1)))
    Diag(Loc, DL_Error, "%0 misaligned address %1 for type %3, "
                        "which requires %2 byte alignment")
      << TypeCheckKinds[Data->TypeCheckKind] << (void*)Pointer
      << Data->Alignment << Data->Type;
  else
    Diag(Loc, DL_Error, "%0 address %1 with insufficient space "
                        "for an object of type %2")
      << TypeCheckKinds[Data->TypeCheckKind] << (void*)Pointer << Data->Type;
  if (Pointer)
    Diag(Pointer, DL_Note, "pointer points here");
}
void __ubsan::__ubsan_handle_type_mismatch(TypeMismatchData *Data,
                                           ValueHandle Pointer) {
  handleTypeMismatchImpl(Data, Pointer, getCallerLocation());
}
void __ubsan::__ubsan_handle_type_mismatch_abort(TypeMismatchData *Data,
                                                 ValueHandle Pointer) {
  handleTypeMismatchImpl(Data, Pointer, getCallerLocation());
  Die();
}

/// \brief Common diagnostic emission for various forms of integer overflow.
template<typename T> static void HandleIntegerOverflow(OverflowData *Data,
                                                       ValueHandle LHS,
                                                       const char *Operator,
                                                       T RHS) {
  SourceLocation Loc = Data->Loc.acquire();
  if (Loc.isDisabled())
    return;

  Diag(Loc, DL_Error, "%0 integer overflow: "
                      "%1 %2 %3 cannot be represented in type %4")
    << (Data->Type.isSignedIntegerTy() ? "signed" : "unsigned")
    << Value(Data->Type, LHS) << Operator << RHS << Data->Type;
}

void __ubsan::__ubsan_handle_add_overflow(OverflowData *Data,
                                          ValueHandle LHS, ValueHandle RHS) {
  HandleIntegerOverflow(Data, LHS, "+", Value(Data->Type, RHS));
}
void __ubsan::__ubsan_handle_add_overflow_abort(OverflowData *Data,
                                                 ValueHandle LHS,
                                                 ValueHandle RHS) {
  __ubsan_handle_add_overflow(Data, LHS, RHS);
  Die();
}

void __ubsan::__ubsan_handle_sub_overflow(OverflowData *Data,
                                          ValueHandle LHS, ValueHandle RHS) {
  HandleIntegerOverflow(Data, LHS, "-", Value(Data->Type, RHS));
}
void __ubsan::__ubsan_handle_sub_overflow_abort(OverflowData *Data,
                                                 ValueHandle LHS,
                                                 ValueHandle RHS) {
  __ubsan_handle_sub_overflow(Data, LHS, RHS);
  Die();
}

void __ubsan::__ubsan_handle_mul_overflow(OverflowData *Data,
                                          ValueHandle LHS, ValueHandle RHS) {
  HandleIntegerOverflow(Data, LHS, "*", Value(Data->Type, RHS));
}
void __ubsan::__ubsan_handle_mul_overflow_abort(OverflowData *Data,
                                                 ValueHandle LHS,
                                                 ValueHandle RHS) {
  __ubsan_handle_mul_overflow(Data, LHS, RHS);
  Die();
}

void __ubsan::__ubsan_handle_negate_overflow(OverflowData *Data,
                                             ValueHandle OldVal) {
  SourceLocation Loc = Data->Loc.acquire();
  if (Loc.isDisabled())
    return;

  if (Data->Type.isSignedIntegerTy())
    Diag(Loc, DL_Error,
         "negation of %0 cannot be represented in type %1; "
         "cast to an unsigned type to negate this value to itself")
      << Value(Data->Type, OldVal) << Data->Type;
  else
    Diag(Loc, DL_Error,
         "negation of %0 cannot be represented in type %1")
      << Value(Data->Type, OldVal) << Data->Type;
}
void __ubsan::__ubsan_handle_negate_overflow_abort(OverflowData *Data,
                                                    ValueHandle OldVal) {
  __ubsan_handle_negate_overflow(Data, OldVal);
  Die();
}

void __ubsan::__ubsan_handle_divrem_overflow(OverflowData *Data,
                                             ValueHandle LHS, ValueHandle RHS) {
  SourceLocation Loc = Data->Loc.acquire();
  if (Loc.isDisabled())
    return;

  Value LHSVal(Data->Type, LHS);
  Value RHSVal(Data->Type, RHS);
  if (RHSVal.isMinusOne())
    Diag(Loc, DL_Error,
         "division of %0 by -1 cannot be represented in type %1")
      << LHSVal << Data->Type;
  else
    Diag(Loc, DL_Error, "division by zero");
}
void __ubsan::__ubsan_handle_divrem_overflow_abort(OverflowData *Data,
                                                    ValueHandle LHS,
                                                    ValueHandle RHS) {
  __ubsan_handle_divrem_overflow(Data, LHS, RHS);
  Die();
}

void __ubsan::__ubsan_handle_shift_out_of_bounds(ShiftOutOfBoundsData *Data,
                                                 ValueHandle LHS,
                                                 ValueHandle RHS) {
  SourceLocation Loc = Data->Loc.acquire();
  if (Loc.isDisabled())
    return;

  Value LHSVal(Data->LHSType, LHS);
  Value RHSVal(Data->RHSType, RHS);
  if (RHSVal.isNegative())
    Diag(Loc, DL_Error, "shift exponent %0 is negative") << RHSVal;
  else if (RHSVal.getPositiveIntValue() >= Data->LHSType.getIntegerBitWidth())
    Diag(Loc, DL_Error,
         "shift exponent %0 is too large for %1-bit type %2")
      << RHSVal << Data->LHSType.getIntegerBitWidth() << Data->LHSType;
  else if (LHSVal.isNegative())
    Diag(Loc, DL_Error, "left shift of negative value %0") << LHSVal;
  else
    Diag(Loc, DL_Error,
         "left shift of %0 by %1 places cannot be represented in type %2")
      << LHSVal << RHSVal << Data->LHSType;
}
void __ubsan::__ubsan_handle_shift_out_of_bounds_abort(
                                                     ShiftOutOfBoundsData *Data,
                                                     ValueHandle LHS,
                                                     ValueHandle RHS) {
  __ubsan_handle_shift_out_of_bounds(Data, LHS, RHS);
  Die();
}

void __ubsan::__ubsan_handle_out_of_bounds(OutOfBoundsData *Data,
                                           ValueHandle Index) {
  SourceLocation Loc = Data->Loc.acquire();
  if (Loc.isDisabled())
    return;

  Value IndexVal(Data->IndexType, Index);
  Diag(Loc, DL_Error, "index %0 out of bounds for type %1")
    << IndexVal << Data->ArrayType;
}
void __ubsan::__ubsan_handle_out_of_bounds_abort(OutOfBoundsData *Data,
                                                 ValueHandle Index) {
  __ubsan_handle_out_of_bounds(Data, Index);
  Die();
}

void __ubsan::__ubsan_handle_builtin_unreachable(UnreachableData *Data) {
  Diag(Data->Loc, DL_Error, "execution reached a __builtin_unreachable() call");
  Die();
}

void __ubsan::__ubsan_handle_missing_return(UnreachableData *Data) {
  Diag(Data->Loc, DL_Error,
       "execution reached the end of a value-returning function "
       "without returning a value");
  Die();
}

void __ubsan::__ubsan_handle_vla_bound_not_positive(VLABoundData *Data,
                                                    ValueHandle Bound) {
  SourceLocation Loc = Data->Loc.acquire();
  if (Loc.isDisabled())
    return;

  Diag(Loc, DL_Error, "variable length array bound evaluates to "
                      "non-positive value %0")
    << Value(Data->Type, Bound);
}
void __ubsan::__ubsan_handle_vla_bound_not_positive_abort(VLABoundData *Data,
                                                           ValueHandle Bound) {
  __ubsan_handle_vla_bound_not_positive(Data, Bound);
  Die();
}


void __ubsan::__ubsan_handle_float_cast_overflow(FloatCastOverflowData *Data,
                                                 ValueHandle From) {
  // TODO: Add deduplication once a SourceLocation is generated for this check.
  Diag(getCallerLocation(), DL_Error,
       "value %0 is outside the range of representable values of type %2")
    << Value(Data->FromType, From) << Data->FromType << Data->ToType;
}
void __ubsan::__ubsan_handle_float_cast_overflow_abort(
                                                    FloatCastOverflowData *Data,
                                                    ValueHandle From) {
  Diag(getCallerLocation(), DL_Error,
       "value %0 is outside the range of representable values of type %2")
    << Value(Data->FromType, From) << Data->FromType << Data->ToType;
  Die();
}

void __ubsan::__ubsan_handle_load_invalid_value(InvalidValueData *Data,
                                                ValueHandle Val) {
  SourceLocation Loc = Data->Loc.acquire();
  if (Loc.isDisabled())
    return;

  Diag(Loc, DL_Error,
       "load of value %0, which is not a valid value for type %1")
    << Value(Data->Type, Val) << Data->Type;
}
void __ubsan::__ubsan_handle_load_invalid_value_abort(InvalidValueData *Data,
                                                      ValueHandle Val) {
  __ubsan_handle_load_invalid_value(Data, Val);
  Die();
}

void __ubsan::__ubsan_handle_function_type_mismatch(
    FunctionTypeMismatchData *Data,
    ValueHandle Function) {
  const char *FName = "(unknown)";

  Location Loc = getFunctionLocation(Function, &FName);

  Diag(Data->Loc, DL_Error,
       "call to function %0 through pointer to incorrect function type %1")
    << FName << Data->Type;
  Diag(Loc, DL_Note, "%0 defined here") << FName;
}

void __ubsan::__ubsan_handle_function_type_mismatch_abort(
    FunctionTypeMismatchData *Data,
    ValueHandle Function) {
  __ubsan_handle_function_type_mismatch(Data, Function);
  Die();
}
