//===-------- BlockFrequency.h - Block Frequency Wrapper --------*- C++ -*-===//
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

#ifndef LLVM_SUPPORT_BLOCKFREQUENCY_H
#define LLVM_SUPPORT_BLOCKFREQUENCY_H

namespace llvm {

class raw_ostream;
class BranchProbability;

// This class represents Block Frequency as a 64-bit value.
class BlockFrequency {

  uint64_t Frequency;

  static void mult96bit(uint64_t freq, uint32_t N, uint64_t W[2]);
  static uint64_t div96bit(uint64_t W[2], uint32_t D);

public:
  BlockFrequency(uint64_t Freq = 0) : Frequency(Freq) { }

  uint64_t getFrequency() const { return Frequency; }

  BlockFrequency &operator*=(const BranchProbability &Prob);
  const BlockFrequency operator*(const BranchProbability &Prob) const;

  BlockFrequency &operator+=(const BlockFrequency &Freq);
  const BlockFrequency operator+(const BlockFrequency &Freq) const;

  bool operator<(const BlockFrequency &RHS) const {
    return Frequency < RHS.Frequency;
  }

  bool operator<=(const BlockFrequency &RHS) const {
    return Frequency <= RHS.Frequency;
  }

  bool operator>(const BlockFrequency &RHS) const {
    return Frequency > RHS.Frequency;
  }

  bool operator>=(const BlockFrequency &RHS) const {
    return Frequency >= RHS.Frequency;
  }

  void print(raw_ostream &OS) const;
};

raw_ostream &operator<<(raw_ostream &OS, const BlockFrequency &Freq);

}

#endif
