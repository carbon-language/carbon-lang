//===- BasicBlockOffsetRanges.cpp - list of address ranges relative to BBs ===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#include "BasicBlockOffsetRanges.h"
#include "BinaryBasicBlock.h"
#include "BinaryFunction.h"
#include <algorithm>

namespace llvm {
namespace bolt {

void BasicBlockOffsetRanges::addAddressRange(BinaryFunction &Function,
                                             uint64_t BeginAddress,
                                             uint64_t EndAddress,
                                             const BinaryData *Data) {
  auto FirstBB = Function.getBasicBlockContainingOffset(
      BeginAddress - Function.getAddress());
  assert(FirstBB && "No basic blocks in the function intersect given range.");

  for (auto I = Function.getIndex(FirstBB), S = Function.size(); I != S; ++I) {
    auto BB = Function.getBasicBlockAtIndex(I);
    uint64_t BBAddress = Function.getAddress() + BB->getOffset();
    if (BBAddress >= EndAddress)
      break;

    uint64_t InternalAddressRangeBegin = std::max(BBAddress, BeginAddress);
    assert(BB->getFunction() == &Function &&
           "Mismatching functions.\n");
    uint64_t InternalAddressRangeEnd =
      std::min(BBAddress + Function.getBasicBlockOriginalSize(BB),
               EndAddress);

    AddressRanges.push_back(
        BBAddressRange{
            BB,
            static_cast<uint16_t>(InternalAddressRangeBegin - BBAddress),
            static_cast<uint16_t>(InternalAddressRangeEnd - BBAddress),
            Data});
  }
}

std::vector<BasicBlockOffsetRanges::AbsoluteRange>
BasicBlockOffsetRanges::getAbsoluteAddressRanges() const {
  std::vector<AbsoluteRange> AbsoluteRanges;
  for (const auto &BBAddressRange : AddressRanges) {
    auto BBOutputAddressRange =
        BBAddressRange.BasicBlock->getOutputAddressRange();
    uint64_t NewRangeBegin = BBOutputAddressRange.first +
        BBAddressRange.RangeBeginOffset;
    // If the end offset pointed to the end of the basic block, then we set
    // the new end range to cover the whole basic block as the BB's size
    // might have increased.
    auto BBFunction = BBAddressRange.BasicBlock->getFunction();
    uint64_t NewRangeEnd =
        (BBAddressRange.RangeEndOffset ==
         BBFunction->getBasicBlockOriginalSize(BBAddressRange.BasicBlock))
        ? BBOutputAddressRange.second
        : (BBOutputAddressRange.first + BBAddressRange.RangeEndOffset);
    AbsoluteRanges.emplace_back(AbsoluteRange{NewRangeBegin, NewRangeEnd,
                                              BBAddressRange.Data});
  }
  if (AbsoluteRanges.empty()) {
    return AbsoluteRanges;
  }
  // Merge adjacent ranges that have the same data.
  std::sort(AbsoluteRanges.begin(), AbsoluteRanges.end(),
            [](const AbsoluteRange &A, const AbsoluteRange &B) {
                return A.Begin < B.Begin;
            });
  decltype(AbsoluteRanges) MergedRanges;

  MergedRanges.emplace_back(AbsoluteRanges[0]);
  for (unsigned I = 1, S = AbsoluteRanges.size(); I != S; ++I) {
    // If this range complements the last one and they point to the same
    // (possibly null) data, merge them instead of creating another one.
    if (AbsoluteRanges[I].Begin == MergedRanges.back().End &&
        AbsoluteRanges[I].Data == MergedRanges.back().Data) {
      MergedRanges.back().End = AbsoluteRanges[I].End;
    } else {
      MergedRanges.emplace_back(AbsoluteRanges[I]);
    }
  }

  return MergedRanges;
}

} // namespace bolt
} // namespace llvm
