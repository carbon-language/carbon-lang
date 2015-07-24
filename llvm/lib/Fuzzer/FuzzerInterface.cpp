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

namespace fuzzer {

void FuzzerRandomLibc::ResetSeed(int seed) { srand(seed); }

size_t FuzzerRandomLibc::Rand() { return rand(); }

UserSuppliedFuzzer::UserSuppliedFuzzer()
    : OwnRand(true), Rand(new FuzzerRandomLibc(0)) {}

UserSuppliedFuzzer::UserSuppliedFuzzer(FuzzerRandomBase *Rand) : Rand(Rand) {}

UserSuppliedFuzzer::~UserSuppliedFuzzer() {
  if (OwnRand)
    delete Rand;
}

size_t UserSuppliedFuzzer::BasicMutate(uint8_t *Data, size_t Size,
                                       size_t MaxSize) {
  return ::fuzzer::Mutate(Data, Size, MaxSize, *Rand);
}
size_t UserSuppliedFuzzer::BasicCrossOver(const uint8_t *Data1, size_t Size1,
                                          const uint8_t *Data2, size_t Size2,
                                          uint8_t *Out, size_t MaxOutSize) {
  return ::fuzzer::CrossOver(Data1, Size1, Data2, Size2, Out, MaxOutSize,
                             *Rand);
}

}  // namespace fuzzer.
