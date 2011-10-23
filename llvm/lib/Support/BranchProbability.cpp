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
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

void BranchProbability::print(raw_ostream &OS) const {
  OS << N << " / " << D << " = " << ((double)N / D);
}

void BranchProbability::dump() const {
  print(dbgs());
  dbgs() << "\n";
}

namespace llvm {

raw_ostream &operator<<(raw_ostream &OS, const BranchProbability &Prob) {
  Prob.print(OS);
  return OS;
}

}
