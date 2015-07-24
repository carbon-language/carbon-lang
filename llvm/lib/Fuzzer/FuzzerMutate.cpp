//===- FuzzerMutate.cpp - Mutate a test input -----------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
// Mutate a test input.
//===----------------------------------------------------------------------===//

#include <cstring>

#include "FuzzerInternal.h"

namespace fuzzer {

static char FlipRandomBit(char X, FuzzerRandomBase &Rand) {
  int Bit = Rand(8);
  char Mask = 1 << Bit;
  char R;
  if (X & (1 << Bit))
    R = X & ~Mask;
  else
    R = X | Mask;
  assert(R != X);
  return R;
}

static char RandCh(FuzzerRandomBase &Rand) {
  if (Rand.RandBool()) return Rand(256);
  const char *Special = "!*'();:@&=+$,/?%#[]123ABCxyz-`~.";
  return Special[Rand(sizeof(Special) - 1)];
}

// Mutates Data in place, returns new size.
size_t Mutate(uint8_t *Data, size_t Size, size_t MaxSize,
              FuzzerRandomBase &Rand) {
  assert(MaxSize > 0);
  assert(Size <= MaxSize);
  if (Size == 0) {
    for (size_t i = 0; i < MaxSize; i++)
      Data[i] = RandCh(Rand);
    return MaxSize;
  }
  assert(Size > 0);
  size_t Idx = Rand(Size);
  switch (Rand(3)) {
  case 0:
    if (Size > 1) {
      // Erase Data[Idx].
      memmove(Data + Idx, Data + Idx + 1, Size - Idx - 1);
      Size = Size - 1;
    }
    [[clang::fallthrough]];
  case 1:
    if (Size < MaxSize) {
      // Insert new value at Data[Idx].
      memmove(Data + Idx + 1, Data + Idx, Size - Idx);
      Data[Idx] = RandCh(Rand);
    }
    Data[Idx] = RandCh(Rand);
    break;
  case 2:
    Data[Idx] = FlipRandomBit(Data[Idx], Rand);
    break;
  }
  assert(Size > 0);
  return Size;
}

}  // namespace fuzzer
