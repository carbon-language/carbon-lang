//===- FuzzerCrossOver.cpp - Cross over two test inputs -------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
// Cross over test inputs.
//===----------------------------------------------------------------------===//

#include "FuzzerInternal.h"
#include <algorithm>

namespace fuzzer {

// Cross A and B, store the result (ap to MaxLen bytes) in U.
void CrossOver(const Unit &A, const Unit &B, Unit *U, size_t MaxLen) {
  size_t Size = rand() % MaxLen + 1;
  U->clear();
  const Unit *V = &A;
  size_t PosA = 0;
  size_t PosB = 0;
  size_t *Pos = &PosA;
  while (U->size() < Size && (PosA < A.size() || PosB < B.size())) {
    // Merge a part of V into U.
    size_t SizeLeftU = Size - U->size();
    if (*Pos < V->size()) {
      size_t SizeLeftV = V->size() - *Pos;
      size_t MaxExtraSize = std::min(SizeLeftU, SizeLeftV);
      size_t ExtraSize = rand() % MaxExtraSize + 1;
      U->insert(U->end(), V->begin() + *Pos, V->begin() + *Pos + ExtraSize);
      (*Pos) += ExtraSize;
    }

    // Use the other Unit on the next iteration.
    if (Pos == &PosA) {
      Pos = &PosB;
      V = &B;
    } else {
      Pos = &PosA;
      V = &A;
    }
  }
}

}  // namespace fuzzer
