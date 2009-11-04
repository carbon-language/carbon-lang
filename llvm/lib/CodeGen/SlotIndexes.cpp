//===-- SlotIndexes.cpp - Slot Indexes Pass  ------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "slotindexes"

#include "llvm/CodeGen/SlotIndexes.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

std::auto_ptr<IndexListEntry> IndexListEntry::emptyKeyEntry,
                              IndexListEntry::tombstoneKeyEntry;

char SlotIndexes::ID = 0;
static RegisterPass<SlotIndexes> X("slotindexes", "Slot index numbering");

void SlotIndexes::getAnalysisUsage(AnalysisUsage &au) const {
  au.setPreservesAll();
  MachineFunctionPass::getAnalysisUsage(au);
}

void SlotIndexes::releaseMemory() {
  mi2iMap.clear();
  mbb2IdxMap.clear();
  idx2MBBMap.clear();
  terminatorGaps.clear();
  clearList();
}

bool SlotIndexes::runOnMachineFunction(MachineFunction &fn) {

  // Compute numbering as follows:
  // Grab an iterator to the start of the index list.
  // Iterate over all MBBs, and within each MBB all MIs, keeping the MI
  // iterator in lock-step (though skipping it over indexes which have
  // null pointers in the instruction field).
  // At each iteration assert that the instruction pointed to in the index
  // is the same one pointed to by the MI iterator. This 

  // FIXME: This can be simplified. The mi2iMap_, Idx2MBBMap, etc. should
  // only need to be set up once after the first numbering is computed.

  mf = &fn;
  initList();

  const unsigned gap = 1;

  // Check that the list contains only the sentinal.
  assert(indexListHead->getNext() == 0 &&
         "Index list non-empty at initial numbering?");
  assert(idx2MBBMap.empty() &&
         "Index -> MBB mapping non-empty at initial numbering?");
  assert(mbb2IdxMap.empty() &&
         "MBB -> Index mapping non-empty at initial numbering?");
  assert(mi2iMap.empty() &&
         "MachineInstr -> Index mapping non-empty at initial numbering?");

  functionSize = 0;
  /*  
  for (unsigned s = 0; s < SlotIndex::NUM; ++s) {  
    indexList.push_back(createEntry(0, s));
  }

  unsigned index = gap * SlotIndex::NUM;
  */

  unsigned index = 0;

  // Iterate over the the function.
  for (MachineFunction::iterator mbbItr = mf->begin(), mbbEnd = mf->end();
       mbbItr != mbbEnd; ++mbbItr) {
    MachineBasicBlock *mbb = &*mbbItr;

    // Insert an index for the MBB start.
    push_back(createEntry(0, index));
    SlotIndex blockStartIndex(back(), SlotIndex::LOAD);

    index += gap * SlotIndex::NUM;

    for (MachineBasicBlock::iterator miItr = mbb->begin(), miEnd = mbb->end();
         miItr != miEnd; ++miItr) {
      MachineInstr *mi = &*miItr;

      if (miItr == mbb->getFirstTerminator()) {
        push_back(createEntry(0, index));
        terminatorGaps.insert(
          std::make_pair(mbb, SlotIndex(back(), SlotIndex::PHI_BIT)));
        index += gap * SlotIndex::NUM;
      }

      // Insert a store index for the instr.
      push_back(createEntry(mi, index));

      // Save this base index in the maps.
      mi2iMap.insert(
        std::make_pair(mi, SlotIndex(back(), SlotIndex::LOAD)));
 
      ++functionSize;

      unsigned Slots = mi->getDesc().getNumDefs();
      if (Slots == 0)
        Slots = 1;

      index += (Slots + 1) * gap * SlotIndex::NUM;
    }

    if (mbb->getFirstTerminator() == mbb->end()) {
      push_back(createEntry(0, index));
      terminatorGaps.insert(
        std::make_pair(mbb, SlotIndex(back(), SlotIndex::PHI_BIT)));
      index += gap * SlotIndex::NUM;
    }

    SlotIndex blockEndIndex(back(), SlotIndex::STORE);
    mbb2IdxMap.insert(
      std::make_pair(mbb, std::make_pair(blockStartIndex, blockEndIndex)));

    idx2MBBMap.push_back(IdxMBBPair(blockStartIndex, mbb));
  }

  // One blank instruction at the end.
  push_back(createEntry(0, index));

  // Sort the Idx2MBBMap
  std::sort(idx2MBBMap.begin(), idx2MBBMap.end(), Idx2MBBCompare());

  DEBUG(dump());

  // And we're done!
  return false;
}

void SlotIndexes::renumber() {
  assert(false && "SlotIndexes::runmuber is not fully implemented yet.");

  // Compute numbering as follows:
  // Grab an iterator to the start of the index list.
  // Iterate over all MBBs, and within each MBB all MIs, keeping the MI
  // iterator in lock-step (though skipping it over indexes which have
  // null pointers in the instruction field).
  // At each iteration assert that the instruction pointed to in the index
  // is the same one pointed to by the MI iterator. This 

  // FIXME: This can be simplified. The mi2iMap_, Idx2MBBMap, etc. should
  // only need to be set up once - when the first numbering is computed.

  assert(false && "Renumbering not supported yet.");
}

void SlotIndexes::dump() const {
  for (const IndexListEntry *itr = front(); itr != getTail();
       itr = itr->getNext()) {
    errs() << itr->getIndex() << " ";

    if (itr->getInstr() != 0) {
      errs() << *itr->getInstr();
    } else {
      errs() << "\n";
    }
  }

  for (MBB2IdxMap::iterator itr = mbb2IdxMap.begin();
       itr != mbb2IdxMap.end(); ++itr) {
    errs() << "MBB " << itr->first->getNumber() << " (" << itr->first << ") - ["
           << itr->second.first << ", " << itr->second.second << "]\n";
  }
}

// Print a SlotIndex to a raw_ostream.
void SlotIndex::print(raw_ostream &os) const {
  os << getIndex();
  if (isPHI())
    os << "*";
}

// Dump a SlotIndex to stderr.
void SlotIndex::dump() const {
  print(errs());
  errs() << "\n";
}

