//===-- ubsan_value.cc ----------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Representation of a runtime value, as marshaled from the generated code to
// the ubsan runtime.
//
//===----------------------------------------------------------------------===//

#include "ubsan_value.h"

using namespace __ubsan;

SIntMax Value::getSIntValue() const {
  CHECK(getType().isSignedIntegerTy());
  if (isInlineInt()) {
    // Val was zero-extended to ValueHandle. Sign-extend from original width
    // to SIntMax.
    const unsigned ExtraBits =
      sizeof(SIntMax) * 8 - getType().getIntegerBitWidth();
    return SIntMax(Val) << ExtraBits >> ExtraBits;
  }
  if (getType().getIntegerBitWidth() == 64)
    return *reinterpret_cast<s64*>(Val);
#ifdef HAVE_INT128_T
  if (getType().getIntegerBitWidth() == 128)
    return *reinterpret_cast<s128*>(Val);
#endif
  UNREACHABLE("unexpected bit width");
}

UIntMax Value::getUIntValue() const {
  CHECK(getType().isUnsignedIntegerTy());
  if (isInlineInt())
    return Val;
  if (getType().getIntegerBitWidth() == 64)
    return *reinterpret_cast<u64*>(Val);
#ifdef HAVE_INT128_T
  if (getType().getIntegerBitWidth() == 128)
    return *reinterpret_cast<u128*>(Val);
#endif
  UNREACHABLE("unexpected bit width");
}

UIntMax Value::getPositiveIntValue() const {
  if (getType().isUnsignedIntegerTy())
    return getUIntValue();
  SIntMax Val = getSIntValue();
  CHECK(Val >= 0);
  return Val;
}

/// Get the floating-point value of this object, extended to a long double.
/// These are always passed by address (our calling convention doesn't allow
/// them to be passed in floating-point registers, so this has little cost).
FloatMax Value::getFloatValue() const {
  CHECK(getType().isFloatTy());
  switch (getType().getFloatBitWidth()) {
#if 0
  // FIXME: OpenCL / NEON 'half' type. LLVM can't lower the conversion
  //        from this to 'long double'.
  case 16: return *reinterpret_cast<__fp16*>(Val);
#endif
  case 32: return *reinterpret_cast<float*>(Val);
  case 64: return *reinterpret_cast<double*>(Val);
  case 80: return *reinterpret_cast<long double*>(Val);
  case 128: return *reinterpret_cast<long double*>(Val);
  }
  UNREACHABLE("unexpected floating point bit width");
}
