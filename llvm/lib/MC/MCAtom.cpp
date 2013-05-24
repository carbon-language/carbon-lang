//===- lib/MC/MCAtom.cpp - MCAtom implementation --------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/MCAtom.h"
#include "llvm/MC/MCModule.h"
#include "llvm/Support/ErrorHandling.h"
#include <iterator>

using namespace llvm;

void MCAtom::remap(uint64_t NewBegin, uint64_t NewEnd) {
  Parent->remap(this, NewBegin, NewEnd);
}

void MCAtom::remapForTruncate(uint64_t TruncPt) {
  assert((TruncPt >= Begin && TruncPt < End) &&
         "Truncation point not contained in atom!");
  remap(Begin, TruncPt);
}

void MCAtom::remapForSplit(uint64_t SplitPt,
                           uint64_t &LBegin, uint64_t &LEnd,
                           uint64_t &RBegin, uint64_t &REnd) {
  assert((SplitPt > Begin && SplitPt <= End) &&
         "Splitting at point not contained in atom!");

  // Compute the new begin/end points.
  LBegin = Begin;
  LEnd = SplitPt - 1;
  RBegin = SplitPt;
  REnd = End;

  // Remap this atom to become the lower of the two new ones.
  remap(LBegin, LEnd);
}

// MCDataAtom

void MCDataAtom::addData(const MCData &D) {
  Data.push_back(D);
  if (Data.size() > Begin - End)
    remap(Begin, End + 1);
}

void MCDataAtom::truncate(uint64_t TruncPt) {
  remapForTruncate(TruncPt);

  Data.resize(TruncPt - Begin + 1);
}

MCDataAtom *MCDataAtom::split(uint64_t SplitPt) {
  uint64_t LBegin, LEnd, RBegin, REnd;
  remapForSplit(SplitPt, LBegin, LEnd, RBegin, REnd);

  MCDataAtom *RightAtom = Parent->createDataAtom(RBegin, REnd);
  RightAtom->setName(getName());

  std::vector<MCData>::iterator I = Data.begin() + (RBegin - LBegin);
  assert(I != Data.end() && "Split point not found in range!");

  std::copy(I, Data.end(), std::back_inserter(RightAtom->Data));
  Data.erase(I, Data.end());
  return RightAtom;
}

// MCTextAtom

void MCTextAtom::addInst(const MCInst &I, uint64_t Size) {
  if (NextInstAddress > End)
    remap(Begin, NextInstAddress);
  Insts.push_back(MCDecodedInst(I, NextInstAddress, Size));
  NextInstAddress += Size;
}

void MCTextAtom::truncate(uint64_t TruncPt) {
  remapForTruncate(TruncPt);

  InstListTy::iterator I = Insts.begin();
  while (I != Insts.end() && I->Address <= TruncPt) ++I;

  assert(I != Insts.end() && "Truncation point not found in disassembly!");
  assert(I->Address == TruncPt + 1 &&
         "Truncation point does not fall on instruction boundary");

  Insts.erase(I, Insts.end());
}

MCTextAtom *MCTextAtom::split(uint64_t SplitPt) {
  uint64_t LBegin, LEnd, RBegin, REnd;
  remapForSplit(SplitPt, LBegin, LEnd, RBegin, REnd);

  MCTextAtom *RightAtom = Parent->createTextAtom(RBegin, REnd);
  RightAtom->setName(getName());

  InstListTy::iterator I = Insts.begin();
  while (I != Insts.end() && I->Address < SplitPt) ++I;
  assert(I != Insts.end() && "Split point not found in disassembly!");
  assert(I->Address == SplitPt &&
         "Split point does not fall on instruction boundary!");

  std::copy(I, Insts.end(), std::back_inserter(RightAtom->Insts));
  Insts.erase(I, Insts.end());
  return RightAtom;
}
