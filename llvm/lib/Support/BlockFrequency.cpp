//====--------------- lib/Support/BlockFrequency.cpp -----------*- C++ -*-====//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements Block Frequency class.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/BranchProbability.h"
#include "llvm/Support/BlockFrequency.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>

using namespace llvm;

/// Multiply FREQ by N and store result in W array.
static void mult96bit(uint64_t freq, uint32_t N, uint32_t W[3]) {
  uint64_t u0 = freq & UINT32_MAX;
  uint64_t u1 = freq >> 32;

  // Represent 96-bit value as W[2]:W[1]:W[0];
  uint64_t t = u0 * N;
  uint64_t k = t >> 32;
  W[0] = t;
  t = u1 * N + k;
  W[1] = t;
  W[2] = t >> 32;
}

/// Divide 96-bit value stored in W[2]:W[1]:W[0] by D. Since our word size is a
/// 32 bit unsigned integer, we can use a short division algorithm.
static uint64_t divrem96bit(uint32_t W[3], uint32_t D, uint32_t *Rout) {
  // We assume that W[2] is non-zero since if W[2] is not then the user should
  // just use hardware division.
  assert(W[2] && "This routine assumes that W[2] is non-zero since if W[2] is "
         "zero, the caller should just use 64/32 hardware.");
  uint32_t Q[3] = { 0, 0, 0 };

  // The generalized short division algorithm sets i to m + n - 1, where n is
  // the number of words in the divisior and m is the number of words by which
  // the divident exceeds the divisor (i.e. m + n == the length of the dividend
  // in words). Due to our assumption that W[2] is non-zero, we know that the
  // dividend is of length 3 implying since n is 1 that m = 2. Thus we set i to
  // m + n - 1 = 2 + 1 - 1 = 2.
  uint32_t R = 0;
  for (int i = 2; i >= 0; --i) {
    uint64_t PartialD = uint64_t(R) << 32 | W[i];
    if (PartialD == 0) {
      Q[i] = 0;
      R = 0;
    } else if (PartialD < D) {
      Q[i] = 0;
      R = uint32_t(PartialD);
    } else if (PartialD == D) {
      Q[i] = 1;
      R = 0;
    } else {
      Q[i] = uint32_t(PartialD / D);
      R = uint32_t(PartialD - (Q[i] * D));
    }
  }

  // If Q[2] is non-zero, then we overflowed.
  uint64_t Result;
  if (Q[2]) {
    Result = UINT64_MAX;
    R = D;
  } else {
    // Form the final uint64_t result, avoiding endianness issues.
    Result = uint64_t(Q[0]) | (uint64_t(Q[1]) << 32);
  }

  if (Rout)
    *Rout = R;

  return Result;
}

uint32_t BlockFrequency::scale(uint32_t N, uint32_t D) {
  assert(D != 0 && "Division by zero");

  // Calculate Frequency * N.
  uint64_t MulLo = (Frequency & UINT32_MAX) * N;
  uint64_t MulHi = (Frequency >> 32) * N;
  uint64_t MulRes = (MulHi << 32) + MulLo;

  // If the product fits in 64 bits, just use built-in division.
  if (MulHi <= UINT32_MAX && MulRes >= MulLo) {
    Frequency = MulRes / D;
    return MulRes % D;
  }

  // Product overflowed, use 96-bit operations.
  // 96-bit value represented as W[2]:W[1]:W[0].
  uint32_t W[3];
  uint32_t R;
  mult96bit(Frequency, N, W);
  Frequency = divrem96bit(W, D, &R);
  return R;
}

BlockFrequency &BlockFrequency::operator*=(const BranchProbability &Prob) {
  scale(Prob.getNumerator(), Prob.getDenominator());
  return *this;
}

const BlockFrequency
BlockFrequency::operator*(const BranchProbability &Prob) const {
  BlockFrequency Freq(Frequency);
  Freq *= Prob;
  return Freq;
}

BlockFrequency &BlockFrequency::operator/=(const BranchProbability &Prob) {
  scale(Prob.getDenominator(), Prob.getNumerator());
  return *this;
}

BlockFrequency BlockFrequency::operator/(const BranchProbability &Prob) const {
  BlockFrequency Freq(Frequency);
  Freq /= Prob;
  return Freq;
}

BlockFrequency &BlockFrequency::operator+=(const BlockFrequency &Freq) {
  uint64_t Before = Freq.Frequency;
  Frequency += Freq.Frequency;

  // If overflow, set frequency to the maximum value.
  if (Frequency < Before)
    Frequency = UINT64_MAX;

  return *this;
}

const BlockFrequency
BlockFrequency::operator+(const BlockFrequency &Prob) const {
  BlockFrequency Freq(Frequency);
  Freq += Prob;
  return Freq;
}

uint32_t BlockFrequency::scale(const BranchProbability &Prob) {
  return scale(Prob.getNumerator(), Prob.getDenominator());
}

void BlockFrequency::print(raw_ostream &OS) const {
  // Convert fixed-point number to decimal.
  OS << Frequency / getEntryFrequency() << ".";
  uint64_t Rem = Frequency % getEntryFrequency();
  uint64_t Eps = 1;
  do {
    Rem *= 10;
    Eps *= 10;
    OS << Rem / getEntryFrequency();
    Rem = Rem % getEntryFrequency();
  } while (Rem >= Eps/2);
}

namespace llvm {

raw_ostream &operator<<(raw_ostream &OS, const BlockFrequency &Freq) {
  Freq.print(OS);
  return OS;
}

}
