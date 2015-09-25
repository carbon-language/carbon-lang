//===-------------- lib/Support/BranchProbability.cpp -----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements Branch Probability class.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/BranchProbability.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>

using namespace llvm;

raw_ostream &BranchProbability::print(raw_ostream &OS) const {
  auto GetHexDigit = [](int Val) -> char {
    assert(Val < 16);
    if (Val < 10)
      return '0' + Val;
    return 'a' + Val - 10;
  };
  OS << "0x";
  for (int Digits = 0; Digits < 8; ++Digits)
    OS << GetHexDigit(N >> (28 - Digits * 4) & 0xf);
  OS << " / 0x";
  for (int Digits = 0; Digits < 8; ++Digits)
    OS << GetHexDigit(D >> (28 - Digits * 4) & 0xf);
  OS << " = " << format("%.2f%%", ((double)N / D) * 100.0);
  return OS;
}

void BranchProbability::dump() const { print(dbgs()) << '\n'; }

BranchProbability::BranchProbability(uint32_t Numerator, uint32_t Denominator) {
  assert(Denominator > 0 && "Denominator cannot be 0!");
  assert(Numerator <= Denominator && "Probability cannot be bigger than 1!");
  if (Denominator == D)
    N = Numerator;
  else {
    uint64_t Prob64 =
        (Numerator * static_cast<uint64_t>(D) + Denominator / 2) / Denominator;
    N = static_cast<uint32_t>(Prob64);
  }
}

BranchProbability &BranchProbability::operator+=(BranchProbability RHS) {
  assert(N <= D - RHS.N &&
         "The sum of branch probabilities should not exceed one!");
  N += RHS.N;
  return *this;
}

BranchProbability &BranchProbability::operator-=(BranchProbability RHS) {
  assert(N >= RHS.N &&
         "Can only subtract a smaller probability from a larger one!");
  N -= RHS.N;
  return *this;
}

// If ConstD is not zero, then replace D by ConstD so that division and modulo
// operations by D can be optimized, in case this function is not inlined by the
// compiler.
template <uint32_t ConstD>
inline uint64_t scale(uint64_t Num, uint32_t N, uint32_t D) {
  if (ConstD > 0)
    D = ConstD;

  assert(D && "divide by 0");

  // Fast path for multiplying by 1.0.
  if (!Num || D == N)
    return Num;

  // Split Num into upper and lower parts to multiply, then recombine.
  uint64_t ProductHigh = (Num >> 32) * N;
  uint64_t ProductLow = (Num & UINT32_MAX) * N;

  // Split into 32-bit digits.
  uint32_t Upper32 = ProductHigh >> 32;
  uint32_t Lower32 = ProductLow & UINT32_MAX;
  uint32_t Mid32Partial = ProductHigh & UINT32_MAX;
  uint32_t Mid32 = Mid32Partial + (ProductLow >> 32);

  // Carry.
  Upper32 += Mid32 < Mid32Partial;

  // Check for overflow.
  if (Upper32 >= D)
    return UINT64_MAX;

  uint64_t Rem = (uint64_t(Upper32) << 32) | Mid32;
  uint64_t UpperQ = Rem / D;

  // Check for overflow.
  if (UpperQ > UINT32_MAX)
    return UINT64_MAX;

  Rem = ((Rem % D) << 32) | Lower32;
  uint64_t LowerQ = Rem / D;
  uint64_t Q = (UpperQ << 32) + LowerQ;

  // Check for overflow.
  return Q < LowerQ ? UINT64_MAX : Q;
}

uint64_t BranchProbability::scale(uint64_t Num) const {
  return ::scale<D>(Num, N, D);
}

uint64_t BranchProbability::scaleByInverse(uint64_t Num) const {
  return ::scale<0>(Num, D, N);
}
