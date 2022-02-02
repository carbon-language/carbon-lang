//====--------------- lib/Support/BlockFrequency.cpp -----------*- C++ -*-====//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements Block Frequency class.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/BlockFrequency.h"
#include <cassert>

using namespace llvm;

BlockFrequency &BlockFrequency::operator*=(BranchProbability Prob) {
  Frequency = Prob.scale(Frequency);
  return *this;
}

BlockFrequency BlockFrequency::operator*(BranchProbability Prob) const {
  BlockFrequency Freq(Frequency);
  Freq *= Prob;
  return Freq;
}

BlockFrequency &BlockFrequency::operator/=(BranchProbability Prob) {
  Frequency = Prob.scaleByInverse(Frequency);
  return *this;
}

BlockFrequency BlockFrequency::operator/(BranchProbability Prob) const {
  BlockFrequency Freq(Frequency);
  Freq /= Prob;
  return Freq;
}

BlockFrequency &BlockFrequency::operator+=(BlockFrequency Freq) {
  uint64_t Before = Freq.Frequency;
  Frequency += Freq.Frequency;

  // If overflow, set frequency to the maximum value.
  if (Frequency < Before)
    Frequency = UINT64_MAX;

  return *this;
}

BlockFrequency BlockFrequency::operator+(BlockFrequency Freq) const {
  BlockFrequency NewFreq(Frequency);
  NewFreq += Freq;
  return NewFreq;
}

BlockFrequency &BlockFrequency::operator-=(BlockFrequency Freq) {
  // If underflow, set frequency to 0.
  if (Frequency <= Freq.Frequency)
    Frequency = 0;
  else
    Frequency -= Freq.Frequency;
  return *this;
}

BlockFrequency BlockFrequency::operator-(BlockFrequency Freq) const {
  BlockFrequency NewFreq(Frequency);
  NewFreq -= Freq;
  return NewFreq;
}

BlockFrequency &BlockFrequency::operator>>=(const unsigned count) {
  // Frequency can never be 0 by design.
  assert(Frequency != 0);

  // Shift right by count.
  Frequency >>= count;

  // Saturate to 1 if we are 0.
  Frequency |= Frequency == 0;
  return *this;
}
