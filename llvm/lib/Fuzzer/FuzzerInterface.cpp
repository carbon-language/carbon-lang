//===- FuzzerInterface.cpp - Mutate a test input --------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
// Parts of public interface for libFuzzer.
//===----------------------------------------------------------------------===//


#include "FuzzerInterface.h"
#include "FuzzerInternal.h"
#include <random>

namespace fuzzer {

UserSuppliedFuzzer::UserSuppliedFuzzer(Random *Rand)
    : Rand(Rand), MD(new MutationDispatcher(*Rand)) {}

UserSuppliedFuzzer::~UserSuppliedFuzzer() {
  if (OwnRand)
    delete Rand;
  delete MD;
}

size_t UserSuppliedFuzzer::Mutate(uint8_t *Data, size_t Size, size_t MaxSize) {
  return GetMD().Mutate(Data, Size, MaxSize);
}

size_t UserSuppliedFuzzer::CrossOver(const uint8_t *Data1, size_t Size1,
                                     const uint8_t *Data2, size_t Size2,
                                     uint8_t *Out, size_t MaxOutSize) {
  return GetMD().CrossOver(Data1, Size1, Data2, Size2, Out, MaxOutSize);
}


size_t Mutate(uint8_t *Data, size_t Size, size_t MaxSize, unsigned int Seed) {
  Random R(Seed);
  MutationDispatcher MD(R);
  return MD.Mutate(Data, Size, MaxSize);
}

}  // namespace fuzzer.
