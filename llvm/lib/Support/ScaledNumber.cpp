//==- lib/Support/ScaledNumber.cpp - Support for scaled numbers -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Implementation of some scaled number algorithms.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/ScaledNumber.h"

using namespace llvm;
using namespace llvm::ScaledNumbers;

std::pair<uint64_t, int16_t> ScaledNumbers::multiply64(uint64_t LHS,
                                                       uint64_t RHS) {
  // Separate into two 32-bit digits (U.L).
  auto getU = [](uint64_t N) { return N >> 32; };
  auto getL = [](uint64_t N) { return N & UINT32_MAX; };
  uint64_t UL = getU(LHS), LL = getL(LHS), UR = getU(RHS), LR = getL(RHS);

  // Compute cross products.
  uint64_t P1 = UL * UR, P2 = UL * LR, P3 = LL * UR, P4 = LL * LR;

  // Sum into two 64-bit digits.
  uint64_t Upper = P1, Lower = P4;
  auto addWithCarry = [&](uint64_t N) {
    uint64_t NewLower = Lower + (getL(N) << 32);
    Upper += getU(N) + (NewLower < Lower);
    Lower = NewLower;
  };
  addWithCarry(P2);
  addWithCarry(P3);

  // Check whether the upper digit is empty.
  if (!Upper)
    return std::make_pair(Lower, 0);

  // Shift as little as possible to maximize precision.
  unsigned LeadingZeros = countLeadingZeros(Upper);
  int Shift = 64 - LeadingZeros;
  if (LeadingZeros)
    Upper = Upper << LeadingZeros | Lower >> Shift;
  return getRounded(Upper, Shift,
                    Shift && (Lower & UINT64_C(1) << (Shift - 1)));
}

static uint64_t getHalf(uint64_t N) { return (N >> 1) + (N & 1); }

std::pair<uint32_t, int16_t> ScaledNumbers::divide32(uint32_t Dividend,
                                                     uint32_t Divisor) {
  assert(Dividend && "expected non-zero dividend");
  assert(Divisor && "expected non-zero divisor");

  // Use 64-bit math and canonicalize the dividend to gain precision.
  uint64_t Dividend64 = Dividend;
  int Shift = 0;
  if (int Zeros = countLeadingZeros(Dividend64)) {
    Shift -= Zeros;
    Dividend64 <<= Zeros;
  }
  uint64_t Quotient = Dividend64 / Divisor;
  uint64_t Remainder = Dividend64 % Divisor;

  // If Quotient needs to be shifted, leave the rounding to getAdjusted().
  if (Quotient > UINT32_MAX)
    return getAdjusted<uint32_t>(Quotient, Shift);

  // Round based on the value of the next bit.
  return getRounded<uint32_t>(Quotient, Shift, Remainder >= getHalf(Divisor));
}

std::pair<uint64_t, int16_t> ScaledNumbers::divide64(uint64_t Dividend,
                                                     uint64_t Divisor) {
  assert(Dividend && "expected non-zero dividend");
  assert(Divisor && "expected non-zero divisor");

  // Minimize size of divisor.
  int Shift = 0;
  if (int Zeros = countTrailingZeros(Divisor)) {
    Shift -= Zeros;
    Divisor >>= Zeros;
  }

  // Check for powers of two.
  if (Divisor == 1)
    return std::make_pair(Dividend, Shift);

  // Maximize size of dividend.
  if (int Zeros = countLeadingZeros(Dividend)) {
    Shift -= Zeros;
    Dividend <<= Zeros;
  }

  // Start with the result of a divide.
  uint64_t Quotient = Dividend / Divisor;
  Dividend %= Divisor;

  // Continue building the quotient with long division.
  while (!(Quotient >> 63) && Dividend) {
    // Shift Dividend and check for overflow.
    bool IsOverflow = Dividend >> 63;
    Dividend <<= 1;
    --Shift;

    // Get the next bit of Quotient.
    Quotient <<= 1;
    if (IsOverflow || Divisor <= Dividend) {
      Quotient |= 1;
      Dividend -= Divisor;
    }
  }

  return getRounded(Quotient, Shift, Dividend >= getHalf(Divisor));
}

int ScaledNumbers::compareImpl(uint64_t L, uint64_t R, int ScaleDiff) {
  assert(ScaleDiff >= 0 && "wrong argument order");
  assert(ScaleDiff < 64 && "numbers too far apart");

  uint64_t L_adjusted = L >> ScaleDiff;
  if (L_adjusted < R)
    return -1;
  if (L_adjusted > R)
    return 1;

  return L > L_adjusted << ScaleDiff ? 1 : 0;
}
