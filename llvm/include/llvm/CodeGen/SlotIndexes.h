//===- llvm/CodeGen/SlotIndexes.h - Slot indexes representation -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements SlotIndex and related classes. The purpuse of SlotIndex
// is to describe a position at which a register can become live, or cease to
// be live.
//
// SlotIndex is mostly a proxy for entries of the SlotIndexList, a class which
// is held is LiveIntervals and provides the real numbering. This allows
// LiveIntervals to perform largely transparent renumbering. The SlotIndex
// class does hold a PHI bit, which determines whether the index relates to a
// PHI use or def point, or an actual instruction. See the SlotIndex class
// description for futher information.
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_SLOTINDEXES_H
#define LLVM_CODEGEN_SLOTINDEXES_H

#include "llvm/ADT/PointerIntPair.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/ErrorHandling.h"

namespace llvm {

  /// This class represents an entry in the slot index list held in the
  /// SlotIndexes pass. It should not be used directly. See the
  /// SlotIndex & SlotIndexes classes for the public interface to this
  /// information.
  class IndexListEntry {
  private:

    static const unsigned EMPTY_KEY_INDEX = ~0U & ~3U,
                          TOMBSTONE_KEY_INDEX = ~0U & ~7U;

    IndexListEntry *next, *prev;
    MachineInstr *mi;
    unsigned index;

  protected:

    typedef enum { EMPTY_KEY, TOMBSTONE_KEY } ReservedEntryType;

    // This constructor is only to be used by getEmptyKeyEntry
    // & getTombstoneKeyEntry. It sets index to the given
    // value and mi to zero.
    IndexListEntry(ReservedEntryType r) : mi(0) {
      switch(r) {
        case EMPTY_KEY: index = EMPTY_KEY_INDEX; break;
        case TOMBSTONE_KEY: index = TOMBSTONE_KEY_INDEX; break;
        default: assert(false && "Invalid value for constructor."); 
      }
      next = this;
      prev = this;
    }

  public:

    IndexListEntry(MachineInstr *mi, unsigned index) : mi(mi), index(index) {
      if (index == EMPTY_KEY_INDEX || index == TOMBSTONE_KEY_INDEX) {
        llvm_report_error("Attempt to create invalid index. "
                          "Available indexes may have been exhausted?.");
      }
    }

    MachineInstr* getInstr() const { return mi; }
    void setInstr(MachineInstr *mi) {
      assert(index != EMPTY_KEY_INDEX && index != TOMBSTONE_KEY_INDEX &&
             "Attempt to modify reserved index.");
      this->mi = mi;
    }

    unsigned getIndex() const { return index; }
    void setIndex(unsigned index) {
      assert(index != EMPTY_KEY_INDEX && index != TOMBSTONE_KEY_INDEX &&
             "Attempt to set index to invalid value.");
      assert(this->index != EMPTY_KEY_INDEX &&
             this->index != TOMBSTONE_KEY_INDEX &&
             "Attempt to reset reserved index value.");
      this->index = index;
    }
    
    IndexListEntry* getNext() { return next; }
    const IndexListEntry* getNext() const { return next; }
    void setNext(IndexListEntry *next) {
      assert(index != EMPTY_KEY_INDEX && index != TOMBSTONE_KEY_INDEX &&
             "Attempt to modify reserved index.");
      this->next = next;
    }

    IndexListEntry* getPrev() { return prev; }
    const IndexListEntry* getPrev() const { return prev; }
    void setPrev(IndexListEntry *prev) {
      assert(index != EMPTY_KEY_INDEX && index != TOMBSTONE_KEY_INDEX &&
             "Attempt to modify reserved index.");
      this->prev = prev;
    }

    // This function returns the index list entry that is to be used for empty
    // SlotIndex keys.
    static IndexListEntry* getEmptyKeyEntry();

    // This function returns the index list entry that is to be used for
    // tombstone SlotIndex keys.
    static IndexListEntry* getTombstoneKeyEntry();
  };

  // Specialize PointerLikeTypeTraits for IndexListEntry.
  template <>
  class PointerLikeTypeTraits<IndexListEntry*> { 
  public:
    static inline void* getAsVoidPointer(IndexListEntry *p) {
      return p;
    }
    static inline IndexListEntry* getFromVoidPointer(void *p) {
      return static_cast<IndexListEntry*>(p);
    }
    enum { NumLowBitsAvailable = 3 };
  };

  /// SlotIndex - An opaque wrapper around machine indexes.
  class SlotIndex {
    friend class SlotIndexes;
    friend struct DenseMapInfo<SlotIndex>;

  private:
    static const unsigned PHI_BIT = 1 << 2;

    PointerIntPair<IndexListEntry*, 3, unsigned> lie;

    SlotIndex(IndexListEntry *entry, unsigned phiAndSlot)
      : lie(entry, phiAndSlot) {
      assert(entry != 0 && "Attempt to construct index with 0 pointer.");
    }

    IndexListEntry& entry() const {
      return *lie.getPointer();
    }

    int getIndex() const {
      return entry().getIndex() | getSlot();
    }

    static inline unsigned getHashValue(const SlotIndex &v) {
      IndexListEntry *ptrVal = &v.entry();
      return (unsigned((intptr_t)ptrVal) >> 4) ^
             (unsigned((intptr_t)ptrVal) >> 9);
    }

  public:

    // FIXME: Ugh. This is public because LiveIntervalAnalysis is still using it
    // for some spill weight stuff. Fix that, then make this private.
    enum Slot { LOAD, USE, DEF, STORE, NUM };

    static inline SlotIndex getEmptyKey() {
      return SlotIndex(IndexListEntry::getEmptyKeyEntry(), 0);
    }

    static inline SlotIndex getTombstoneKey() {
      return SlotIndex(IndexListEntry::getTombstoneKeyEntry(), 0);
    }
    
    /// Construct an invalid index.
    SlotIndex() : lie(IndexListEntry::getEmptyKeyEntry(), 0) {}

    // Construct a new slot index from the given one, set the phi flag on the
    // new index to the value of the phi parameter.
    SlotIndex(const SlotIndex &li, bool phi)
      : lie(&li.entry(), phi ? PHI_BIT & li.getSlot() : (unsigned)li.getSlot()){
      assert(lie.getPointer() != 0 &&
             "Attempt to construct index with 0 pointer.");
    }

    // Construct a new slot index from the given one, set the phi flag on the
    // new index to the value of the phi parameter, and the slot to the new slot.
    SlotIndex(const SlotIndex &li, bool phi, Slot s)
      : lie(&li.entry(), phi ? PHI_BIT & s : (unsigned)s) {
      assert(lie.getPointer() != 0 &&
             "Attempt to construct index with 0 pointer.");
    }

    /// Returns true if this is a valid index. Invalid indicies do
    /// not point into an index table, and cannot be compared.
    bool isValid() const {
      return (lie.getPointer() != 0) && (lie.getPointer()->getIndex() != 0);
    }

    /// Print this index to the given raw_ostream.
    void print(raw_ostream &os) const;

    /// Dump this index to stderr.
    void dump() const;

    /// Compare two SlotIndex objects for equality.
    bool operator==(SlotIndex other) const {
      return getIndex() == other.getIndex();
    }
    /// Compare two SlotIndex objects for inequality.
    bool operator!=(SlotIndex other) const {
      return getIndex() != other.getIndex(); 
    }
   
    /// Compare two SlotIndex objects. Return true if the first index
    /// is strictly lower than the second.
    bool operator<(SlotIndex other) const {
      return getIndex() < other.getIndex();
    }
    /// Compare two SlotIndex objects. Return true if the first index
    /// is lower than, or equal to, the second.
    bool operator<=(SlotIndex other) const {
      return getIndex() <= other.getIndex();
    }

    /// Compare two SlotIndex objects. Return true if the first index
    /// is greater than the second.
    bool operator>(SlotIndex other) const {
      return getIndex() > other.getIndex();
    }

    /// Compare two SlotIndex objects. Return true if the first index
    /// is greater than, or equal to, the second.
    bool operator>=(SlotIndex other) const {
      return getIndex() >= other.getIndex();
    }

    /// Return the distance from this index to the given one.
    int distance(SlotIndex other) const {
      return other.getIndex() - getIndex();
    }

    /// Returns the slot for this SlotIndex.
    Slot getSlot() const {
      return static_cast<Slot>(lie.getInt()  & ~PHI_BIT);
    }

    /// Returns the state of the PHI bit.
    bool isPHI() const {
      return lie.getInt() & PHI_BIT;
    }

    /// Returns the base index for associated with this index. The base index
    /// is the one associated with the LOAD slot for the instruction pointed to
    /// by this index.
    SlotIndex getBaseIndex() const {
      return getLoadIndex();
    }

    /// Returns the boundary index for associated with this index. The boundary
    /// index is the one associated with the LOAD slot for the instruction
    /// pointed to by this index.
    SlotIndex getBoundaryIndex() const {
      return getStoreIndex();
    }

    /// Returns the index of the LOAD slot for the instruction pointed to by
    /// this index.
    SlotIndex getLoadIndex() const {
      return SlotIndex(&entry(), SlotIndex::LOAD);
    }    

    /// Returns the index of the USE slot for the instruction pointed to by
    /// this index.
    SlotIndex getUseIndex() const {
      return SlotIndex(&entry(), SlotIndex::USE);
    }

    /// Returns the index of the DEF slot for the instruction pointed to by
    /// this index.
    SlotIndex getDefIndex() const {
      return SlotIndex(&entry(), SlotIndex::DEF);
    }

    /// Returns the index of the STORE slot for the instruction pointed to by
    /// this index.
    SlotIndex getStoreIndex() const {
      return SlotIndex(&entry(), SlotIndex::STORE);
    }    

    /// Returns the next slot in the index list. This could be either the
    /// next slot for the instruction pointed to by this index or, if this
    /// index is a STORE, the first slot for the next instruction.
    /// WARNING: This method is considerably more expensive than the methods
    /// that return specific slots (getUseIndex(), etc). If you can - please
    /// use one of those methods.
    SlotIndex getNextSlot() const {
      Slot s = getSlot();
      if (s == SlotIndex::STORE) {
        return SlotIndex(entry().getNext(), SlotIndex::LOAD);
      }
      return SlotIndex(&entry(), s + 1);
    }

    /// Returns the next index. This is the index corresponding to the this
    /// index's slot, but for the next instruction.
    SlotIndex getNextIndex() const {
      return SlotIndex(entry().getNext(), getSlot());
    }

    /// Returns the previous slot in the index list. This could be either the
    /// previous slot for the instruction pointed to by this index or, if this
    /// index is a LOAD, the last slot for the previous instruction.
    /// WARNING: This method is considerably more expensive than the methods
    /// that return specific slots (getUseIndex(), etc). If you can - please
    /// use one of those methods.
    SlotIndex getPrevSlot() const {
      Slot s = getSlot();
      if (s == SlotIndex::LOAD) {
        return SlotIndex(entry().getPrev(), SlotIndex::STORE);
      }
      return SlotIndex(&entry(), s - 1);
    }

    /// Returns the previous index. This is the index corresponding to this
    /// index's slot, but for the previous instruction.
    SlotIndex getPrevIndex() const {
      return SlotIndex(entry().getPrev(), getSlot());
    }

  };

  /// DenseMapInfo specialization for SlotIndex.
  template <>
  struct DenseMapInfo<SlotIndex> {
    static inline SlotIndex getEmptyKey() {
      return SlotIndex::getEmptyKey();
    }
    static inline SlotIndex getTombstoneKey() {
      return SlotIndex::getTombstoneKey();
    }
    static inline unsigned getHashValue(const SlotIndex &v) {
      return SlotIndex::getHashValue(v);
    }
    static inline bool isEqual(const SlotIndex &LHS, const SlotIndex &RHS) {
      return (LHS == RHS);
    }
    static inline bool isPod() { return false; }
  };

  inline raw_ostream& operator<<(raw_ostream &os, SlotIndex li) {
    li.print(os);
    return os;
  }

  typedef std::pair<SlotIndex, MachineBasicBlock*> IdxMBBPair;

  inline bool operator<(SlotIndex V, const IdxMBBPair &IM) {
    return V < IM.first;
  }

  inline bool operator<(const IdxMBBPair &IM, SlotIndex V) {
    return IM.first < V;
  }

  struct Idx2MBBCompare {
    bool operator()(const IdxMBBPair &LHS, const IdxMBBPair &RHS) const {
      return LHS.first < RHS.first;
    }
  };

  /// SlotIndexes pass.
  ///
  /// This pass assigns indexes to each instruction.
  class SlotIndexes : public MachineFunctionPass {
  private:

    MachineFunction *mf;
    IndexListEntry *indexListHead;
    unsigned functionSize;

    typedef DenseMap<const MachineInstr*, SlotIndex> Mi2IndexMap;
    Mi2IndexMap mi2iMap;

    /// MBB2IdxMap - The indexes of the first and last instructions in the
    /// specified basic block.
    typedef DenseMap<const MachineBasicBlock*,
                     std::pair<SlotIndex, SlotIndex> > MBB2IdxMap;
    MBB2IdxMap mbb2IdxMap;

    /// Idx2MBBMap - Sorted list of pairs of index of first instruction
    /// and MBB id.
    std::vector<IdxMBBPair> idx2MBBMap;

    typedef DenseMap<const MachineBasicBlock*, SlotIndex> TerminatorGapsMap;
    TerminatorGapsMap terminatorGaps;

    // IndexListEntry allocator.
    BumpPtrAllocator ileAllocator;

    IndexListEntry* createEntry(MachineInstr *mi, unsigned index) {
      IndexListEntry *entry =
        static_cast<IndexListEntry*>(
          ileAllocator.Allocate(sizeof(IndexListEntry),
          alignof<IndexListEntry>()));

      new (entry) IndexListEntry(mi, index);

      return entry;
    }

    void initList() {
      assert(indexListHead == 0 && "Zero entry non-null at initialisation.");
      indexListHead = createEntry(0, ~0U);
      indexListHead->setNext(0);
      indexListHead->setPrev(indexListHead);
    }

    void clearList() {
      indexListHead = 0;
      ileAllocator.Reset();
    }

    IndexListEntry* getTail() {
      assert(indexListHead != 0 && "Call to getTail on uninitialized list.");
      return indexListHead->getPrev();
    }

    const IndexListEntry* getTail() const {
      assert(indexListHead != 0 && "Call to getTail on uninitialized list.");
      return indexListHead->getPrev();
    }

    // Returns true if the index list is empty.
    bool empty() const { return (indexListHead == getTail()); }

    IndexListEntry* front() {
      assert(!empty() && "front() called on empty index list.");
      return indexListHead;
    }

    const IndexListEntry* front() const {
      assert(!empty() && "front() called on empty index list.");
      return indexListHead;
    }

    IndexListEntry* back() {
      assert(!empty() && "back() called on empty index list.");
      return getTail()->getPrev();
    }

    const IndexListEntry* back() const {
      assert(!empty() && "back() called on empty index list.");
      return getTail()->getPrev();
    }

    /// Insert a new entry before itr.
    void insert(IndexListEntry *itr, IndexListEntry *val) {
      assert(itr != 0 && "itr should not be null.");
      IndexListEntry *prev = itr->getPrev();
      val->setNext(itr);
      val->setPrev(prev);
      
      if (itr != indexListHead) {
        prev->setNext(val);
      }
      else {
        indexListHead = val;
      }
      itr->setPrev(val);
    }

    /// Push a new entry on to the end of the list.
    void push_back(IndexListEntry *val) {
      insert(getTail(), val);
    }

  public:
    static char ID;

    SlotIndexes() : MachineFunctionPass(&ID), indexListHead(0) {}

    virtual void getAnalysisUsage(AnalysisUsage &au) const;
    virtual void releaseMemory(); 

    virtual bool runOnMachineFunction(MachineFunction &fn);

    /// Dump the indexes.
    void dump() const;

    /// Renumber the index list, providing space for new instructions.
    void renumber();

    /// Returns the zero index for this analysis.
    SlotIndex getZeroIndex() {
      assert(front()->getIndex() == 0 && "First index is not 0?");
      return SlotIndex(front(), 0);
    }

    /// Returns the invalid index marker for this analysis.
    SlotIndex getInvalidIndex() {
      return getZeroIndex();
    }

    /// Returns the distance between the highest and lowest indexes allocated
    /// so far.
    unsigned getIndexesLength() const {
      assert(front()->getIndex() == 0 &&
             "Initial index isn't zero?");

      return back()->getIndex();
    }

    /// Returns the number of instructions in the function.
    unsigned getFunctionSize() const {
      return functionSize;
    }

    /// Returns true if the given machine instr is mapped to an index,
    /// otherwise returns false.
    bool hasIndex(const MachineInstr *instr) const {
      return (mi2iMap.find(instr) != mi2iMap.end());
    }

    /// Returns the base index for the given instruction.
    SlotIndex getInstructionIndex(const MachineInstr *instr) const {
      Mi2IndexMap::const_iterator itr = mi2iMap.find(instr);
      assert(itr != mi2iMap.end() && "Instruction not found in maps.");
      return itr->second;
    }

    /// Returns the instruction for the given index, or null if the given
    /// index has no instruction associated with it.
    MachineInstr* getInstructionFromIndex(SlotIndex index) const {
      return index.entry().getInstr();
    }

    /// Returns the next non-null index.
    SlotIndex getNextNonNullIndex(SlotIndex index) {
      SlotIndex nextNonNull = index.getNextIndex();

      while (&nextNonNull.entry() != getTail() &&
             getInstructionFromIndex(nextNonNull) == 0) {
        nextNonNull = nextNonNull.getNextIndex();
      }

      return nextNonNull;
    }

    /// Returns the first index in the given basic block.
    SlotIndex getMBBStartIdx(const MachineBasicBlock *mbb) const {
      MBB2IdxMap::const_iterator itr = mbb2IdxMap.find(mbb);
      assert(itr != mbb2IdxMap.end() && "MBB not found in maps.");
      return itr->second.first;
    }

    /// Returns the last index in the given basic block.
    SlotIndex getMBBEndIdx(const MachineBasicBlock *mbb) const {
      MBB2IdxMap::const_iterator itr = mbb2IdxMap.find(mbb);
      assert(itr != mbb2IdxMap.end() && "MBB not found in maps.");
      return itr->second.second;
    }

    /// Returns the terminator gap for the given index.
    SlotIndex getTerminatorGap(const MachineBasicBlock *mbb) {
      TerminatorGapsMap::iterator itr = terminatorGaps.find(mbb);
      assert(itr != terminatorGaps.end() &&
             "All MBBs should have terminator gaps in their indexes.");
      return itr->second;
    }

    /// Returns the basic block which the given index falls in.
    MachineBasicBlock* getMBBFromIndex(SlotIndex index) const {
      std::vector<IdxMBBPair>::const_iterator I =
        std::lower_bound(idx2MBBMap.begin(), idx2MBBMap.end(), index);
      // Take the pair containing the index
      std::vector<IdxMBBPair>::const_iterator J =
        ((I != idx2MBBMap.end() && I->first > index) ||
         (I == idx2MBBMap.end() && idx2MBBMap.size()>0)) ? (I-1): I;

      assert(J != idx2MBBMap.end() && J->first <= index &&
             index <= getMBBEndIdx(J->second) &&
             "index does not correspond to an MBB");
      return J->second;
    }

    bool findLiveInMBBs(SlotIndex start, SlotIndex end,
                        SmallVectorImpl<MachineBasicBlock*> &mbbs) const {
      std::vector<IdxMBBPair>::const_iterator itr =
        std::lower_bound(idx2MBBMap.begin(), idx2MBBMap.end(), start);
      bool resVal = false;

      while (itr != idx2MBBMap.end()) {
        if (itr->first >= end)
          break;
        mbbs.push_back(itr->second);
        resVal = true;
        ++itr;
      }
      return resVal;
    }

    /// Return a list of MBBs that can be reach via any branches or
    /// fall-throughs.
    bool findReachableMBBs(SlotIndex start, SlotIndex end,
                           SmallVectorImpl<MachineBasicBlock*> &mbbs) const {
      std::vector<IdxMBBPair>::const_iterator itr =
        std::lower_bound(idx2MBBMap.begin(), idx2MBBMap.end(), start);

      bool resVal = false;
      while (itr != idx2MBBMap.end()) {
        if (itr->first > end)
          break;
        MachineBasicBlock *mbb = itr->second;
        if (getMBBEndIdx(mbb) > end)
          break;
        for (MachineBasicBlock::succ_iterator si = mbb->succ_begin(),
             se = mbb->succ_end(); si != se; ++si)
          mbbs.push_back(*si);
        resVal = true;
        ++itr;
      }
      return resVal;
    }

    /// Returns the MBB covering the given range, or null if the range covers
    /// more than one basic block.
    MachineBasicBlock* getMBBCoveringRange(SlotIndex start, SlotIndex end) const {

      assert(start < end && "Backwards ranges not allowed.");

      std::vector<IdxMBBPair>::const_iterator itr =
        std::lower_bound(idx2MBBMap.begin(), idx2MBBMap.end(), start);

      if (itr == idx2MBBMap.end()) {
        itr = prior(itr);
        return itr->second;
      }

      // Check that we don't cross the boundary into this block.
      if (itr->first < end)
        return 0;

      itr = prior(itr);

      if (itr->first <= start)
        return itr->second;

      return 0;
    }

    /// Returns true if there is a gap in the numbering before the given index.
    bool hasGapBeforeInstr(SlotIndex index) {
      index = index.getBaseIndex();
      SlotIndex prevIndex = index.getPrevIndex();
      
      if (prevIndex == getZeroIndex())
        return false;

      if (getInstructionFromIndex(prevIndex) == 0)
        return true;

      if (prevIndex.distance(index) >= 2 * SlotIndex::NUM)
        return true;

      return false;
    }

    /// Returns true if there is a gap in the numbering after the given index.
    bool hasGapAfterInstr(SlotIndex index) const {
      // Not implemented yet.
      assert(false &&
             "SlotIndexes::hasGapAfterInstr(SlotIndex) not implemented yet.");
      return false;
    }

    /// findGapBeforeInstr - Find an empty instruction slot before the
    /// specified index. If "Furthest" is true, find one that's furthest
    /// away from the index (but before any index that's occupied).
    // FIXME: This whole method should go away in future. It should
    // always be possible to insert code between existing indices.
    SlotIndex findGapBeforeInstr(SlotIndex index, bool furthest = false) {
      if (index == getZeroIndex())
        return getInvalidIndex();

      index = index.getBaseIndex();
      SlotIndex prevIndex = index.getPrevIndex();

      if (prevIndex == getZeroIndex())
        return getInvalidIndex();

      // Try to reuse existing index objects with null-instrs.
      if (getInstructionFromIndex(prevIndex) == 0) {
        if (furthest) {
          while (getInstructionFromIndex(prevIndex) == 0 &&
                 prevIndex != getZeroIndex()) {
            prevIndex = prevIndex.getPrevIndex();
          }

          prevIndex = prevIndex.getNextIndex();
        }
 
        assert(getInstructionFromIndex(prevIndex) == 0 && "Index list is broken.");

        return prevIndex;
      }

      int dist = prevIndex.distance(index);

      // Double check that the spacing between this instruction and
      // the last is sane.
      assert(dist >= SlotIndex::NUM &&
             "Distance between indexes too small.");

      // If there's no gap return an invalid index.
      if (dist < 2*SlotIndex::NUM) {
        return getInvalidIndex();
      }

      // Otherwise insert new index entries into the list using the
      // gap in the numbering.
      IndexListEntry *newEntry =
        createEntry(0, prevIndex.entry().getIndex() + SlotIndex::NUM);

      insert(&index.entry(), newEntry);

      // And return a pointer to the entry at the start of the gap.
      return index.getPrevIndex();
    }

    /// Insert the given machine instruction into the mapping at the given
    /// index.
    void insertMachineInstrInMaps(MachineInstr *mi, SlotIndex index) {
      index = index.getBaseIndex();
      IndexListEntry *miEntry = &index.entry();
      assert(miEntry->getInstr() == 0 && "Index already in use.");
      miEntry->setInstr(mi);

      assert(mi2iMap.find(mi) == mi2iMap.end() &&
             "MachineInstr already has an index.");

      mi2iMap.insert(std::make_pair(mi, index));
    }

    /// Remove the given machine instruction from the mapping.
    void removeMachineInstrFromMaps(MachineInstr *mi) {
      // remove index -> MachineInstr and
      // MachineInstr -> index mappings
      Mi2IndexMap::iterator mi2iItr = mi2iMap.find(mi);
      if (mi2iItr != mi2iMap.end()) {
        IndexListEntry *miEntry(&mi2iItr->second.entry());        
        assert(miEntry->getInstr() == mi && "Instruction indexes broken.");
        // FIXME: Eventually we want to actually delete these indexes.
        miEntry->setInstr(0);
        mi2iMap.erase(mi2iItr);
      }
    }

    /// ReplaceMachineInstrInMaps - Replacing a machine instr with a new one in
    /// maps used by register allocator.
    void replaceMachineInstrInMaps(MachineInstr *mi, MachineInstr *newMI) {
      Mi2IndexMap::iterator mi2iItr = mi2iMap.find(mi);
      if (mi2iItr == mi2iMap.end())
        return;
      SlotIndex replaceBaseIndex = mi2iItr->second;
      IndexListEntry *miEntry(&replaceBaseIndex.entry());
      assert(miEntry->getInstr() == mi &&
             "Mismatched instruction in index tables.");
      miEntry->setInstr(newMI);
      mi2iMap.erase(mi2iItr);
      mi2iMap.insert(std::make_pair(newMI, replaceBaseIndex));
    }

  };


}

#endif // LLVM_CODEGEN_LIVEINDEX_H 
