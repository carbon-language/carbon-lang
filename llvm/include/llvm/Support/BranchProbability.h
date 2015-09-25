//===- BranchProbability.h - Branch Probability Wrapper ---------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Definition of BranchProbability shared by IR and Machine Instructions.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_BRANCHPROBABILITY_H
#define LLVM_SUPPORT_BRANCHPROBABILITY_H

#include "llvm/Support/DataTypes.h"
#include <cassert>

namespace llvm {

class raw_ostream;

// This class represents Branch Probability as a non-negative fraction that is
// no greater than 1. It uses a fixed-point-like implementation, in which the
// denominator is always a constant value (here we use 1<<31 for maximum
// precision).
class BranchProbability {
  // Numerator
  uint32_t N;

  // Denominator, which is a constant value.
  static const uint32_t D = 1u << 31;

  // Construct a BranchProbability with only numerator assuming the denominator
  // is 1<<31. For internal use only.
  explicit BranchProbability(uint32_t n) : N(n) {}

public:
  BranchProbability() : N(0) {}
  BranchProbability(uint32_t Numerator, uint32_t Denominator);

  bool isZero() const { return N == 0; }

  static BranchProbability getZero() { return BranchProbability(0); }
  static BranchProbability getOne() { return BranchProbability(D); }
  // Create a BranchProbability object with the given numerator and 1<<31
  // as denominator.
  static BranchProbability getRaw(uint32_t N) { return BranchProbability(N); }

  // Normalize given probabilties so that the sum of them becomes approximate
  // one.
  template <class ProbabilityList>
  static void normalizeProbabilities(ProbabilityList &Probs);

  uint32_t getNumerator() const { return N; }
  static uint32_t getDenominator() { return D; }

  // Return (1 - Probability).
  BranchProbability getCompl() const { return BranchProbability(D - N); }

  raw_ostream &print(raw_ostream &OS) const;

  void dump() const;

  /// \brief Scale a large integer.
  ///
  /// Scales \c Num.  Guarantees full precision.  Returns the floor of the
  /// result.
  ///
  /// \return \c Num times \c this.
  uint64_t scale(uint64_t Num) const;

  /// \brief Scale a large integer by the inverse.
  ///
  /// Scales \c Num by the inverse of \c this.  Guarantees full precision.
  /// Returns the floor of the result.
  ///
  /// \return \c Num divided by \c this.
  uint64_t scaleByInverse(uint64_t Num) const;

  BranchProbability &operator+=(BranchProbability RHS);
  BranchProbability &operator-=(BranchProbability RHS);
  BranchProbability &operator*=(BranchProbability RHS) {
    N = (static_cast<uint64_t>(N) * RHS.N + D / 2) / D;
    return *this;
  }

  BranchProbability operator+(BranchProbability RHS) const {
    BranchProbability Prob(*this);
    return Prob += RHS;
  }

  BranchProbability operator-(BranchProbability RHS) const {
    BranchProbability Prob(*this);
    return Prob -= RHS;
  }

  BranchProbability operator*(BranchProbability RHS) const {
    BranchProbability Prob(*this);
    return Prob *= RHS;
  }

  bool operator==(BranchProbability RHS) const { return N == RHS.N; }
  bool operator!=(BranchProbability RHS) const { return !(*this == RHS); }
  bool operator<(BranchProbability RHS) const { return N < RHS.N; }
  bool operator>(BranchProbability RHS) const { return RHS < *this; }
  bool operator<=(BranchProbability RHS) const { return !(RHS < *this); }
  bool operator>=(BranchProbability RHS) const { return !(*this < RHS); }
};

inline raw_ostream &operator<<(raw_ostream &OS, BranchProbability Prob) {
  return Prob.print(OS);
}

template <class ProbabilityList>
void BranchProbability::normalizeProbabilities(ProbabilityList &Probs) {
  uint64_t Sum = 0;
  for (auto Prob : Probs)
    Sum += Prob.N;
  assert(Sum > 0);
  for (auto &Prob : Probs)
    Prob.N = (Prob.N * uint64_t(D) + Sum / 2) / Sum;
}

}

#endif
