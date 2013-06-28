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
static void mult96bit(uint64_t freq, uint32_t N, uint64_t W[2]) {
  uint64_t u0 = freq & UINT32_MAX;
  uint64_t u1 = freq >> 32;

  // Represent 96-bit value as w[2]:w[1]:w[0];
  uint32_t w[3] = { 0, 0, 0 };

  uint64_t t = u0 * N;
  uint64_t k = t >> 32;
  w[0] = t;
  t = u1 * N + k;
  w[1] = t;
  w[2] = t >> 32;

  // W[1] - higher bits.
  // W[0] - lower bits.
  W[0] = w[0] + ((uint64_t) w[1] << 32);
  W[1] = w[2];
}


/// Divide 96-bit value stored in W array by D.
/// Return 64-bit quotient, saturated to UINT64_MAX on overflow.
static uint64_t div96bit(uint64_t W[2], uint32_t D) {
  uint64_t y = W[0];
  uint64_t x = W[1];
  unsigned i;

  // This is really a 64-bit division.
  if (!x)
    return y / D;

  // This long division algorithm automatically saturates on overflow.
  for (i = 0; i < 64 && x; ++i) {
    uint32_t t = -((x >> 31) & 1); // Splat bit 31 to bits 0-31.
    x = (x << 1) | (y >> 63);
    y = y << 1;
    if ((x | t) >= D) {
      x -= D;
      ++y;
    }
  }

  return y << (64 - i);
}


void BlockFrequency::scale(uint32_t N, uint32_t D) {
  assert(D != 0 && "Division by zero");

  // Calculate Frequency * N.
  uint64_t MulLo = (Frequency & UINT32_MAX) * N;
  uint64_t MulHi = (Frequency >> 32) * N;
  uint64_t MulRes = (MulHi << 32) + MulLo;

  // If the product fits in 64 bits, just use built-in division.
  if (MulHi <= UINT32_MAX && MulRes <= MulLo) {
    Frequency = MulRes / D;
    return;
  }

  // Product overflowed, use 96-bit operations.
  // 96-bit value represented as W[1]:W[0].
  uint64_t W[2];
  mult96bit(Frequency, N, W);
  Frequency = div96bit(W, D);
  return;
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
