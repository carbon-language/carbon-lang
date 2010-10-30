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
// LiveIntervals to perform largely transparent renumbering.
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_SLOTINDEXES_H
#define LLVM_CODEGEN_SLOTINDEXES_H

#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/ADT/PointerIntPair.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/Allocator.h"

namespace llvm {

  /// This class represents an entry in the slot index list held in the
  /// SlotIndexes pass. It should not be used directly. See the
  /// SlotIndex & SlotIndexes classes for the public interface to this
  /// information.
  class IndexListEntry {
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
      assert(index != EMPTY_KEY_INDEX && index != TOMBSTONE_KEY_INDEX &&
             "Attempt to create invalid index. "
             "Available indexes may have been exhausted?.");
    }

    bool isValid() const {
      return (index != EMPTY_KEY_INDEX && index != TOMBSTONE_KEY_INDEX);
    }

    MachineInstr* getInstr() const { return mi; }
    void setInstr(MachineInstr *mi) {
      assert(isValid() && "Attempt to modify reserved index.");
      this->mi = mi;
    }

    unsigned getIndex() const { return index; }
    void setIndex(unsigned index) {
      assert(index != EMPTY_KEY_INDEX && index != TOMBSTONE_KEY_INDEX &&
             "Attempt to set index to invalid value.");
      assert(isValid() && "Attempt to reset reserved index value.");
      this->index = index;
    }
    
    IndexListEntry* getNext() { return next; }
    const IndexListEntry* getNext() const { return next; }
    void setNext(IndexListEntry *next) {
      assert(isValid() && "Attempt to modify reserved index.");
      this->next = next;
    }

    IndexListEntry* getPrev() { return prev; }
    const IndexListEntry* getPrev() const { return prev; }
    void setPrev(IndexListEntry *prev) {
      assert(isValid() && "Attempt to modify reserved index.");
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

    enum Slot { LOAD, USE, DEF, STORE, NUM };

    PointerIntPair<IndexListEntry*, 2, unsigned> lie;

    SlotIndex(IndexListEntry *entry, unsigned slot)
      : lie(entry, slot) {
      assert(entry != 0 && "Attempt to construct index with 0 pointer.");
    }

    IndexListEntry& entry() const {
      return *lie.getPointer();
    }

    int getIndex() const {
      return entry().getIndex() | getSlot();
    }

    /// Returns the slot for this SlotIndex.
    Slot getSlot() const {
      return static_cast<Slot>(lie.getInt());
    }

    static inline unsigned getHashValue(const SlotIndex &v) {
      IndexListEntry *ptrVal = &v.entry();
      return (unsigned((intptr_t)ptrVal) >> 4) ^
             (unsigned((intptr_t)ptrVal) >> 9);
    }

  public:
    static inline SlotIndex getEmptyKey() {
      return SlotIndex(IndexListEntry::getEmptyKeyEntry(), 0);
    }

    static inline SlotIndex getTombstoneKey() {
      return SlotIndex(IndexListEntry::getTombstoneKeyEntry(), 0);
    }

    /// Construct an invalid index.
    SlotIndex() : lie(IndexListEntry::getEmptyKeyEntry(), 0) {}

    // Construct a new slot index from the given one, and set the slot.
    SlotIndex(const SlotIndex &li, Slot s)
      : lie(&li.entry(), unsigned(s)) {
      assert(lie.getPointer() != 0 &&
             "Attempt to construct index with 0 pointer.");
    }

    /// Returns true if this is a valid index. Invalid indicies do
    /// not point into an index table, and cannot be compared.
    bool isValid() const {
      IndexListEntry *entry = lie.getPointer();
      return ((entry!= 0) && (entry->isValid()));
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

    /// isLoad - Return true if this is a LOAD slot.
    bool isLoad() const {
      return getSlot() == LOAD;
    }

    /// isDef - Return true if this is a DEF slot.
    bool isDef() const {
      return getSlot() == DEF;
    }

    /// isUse - Return true if this is a USE slot.
    bool isUse() const {
      return getSlot() == USE;
    }

    /// isStore - Return true if this is a STORE slot.
    bool isStore() const {
      return getSlot() == STORE;
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
  };
  
  template <> struct isPodLike<SlotIndex> { static const bool value = true; };


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

    // IndexListEntry allocator.
    BumpPtrAllocator ileAllocator;

    IndexListEntry* createEntry(MachineInstr *mi, unsigned index) {
      IndexListEntry *entry =
        static_cast<IndexListEntry*>(
          ileAllocator.Allocate(sizeof(IndexListEntry),
          alignOf<IndexListEntry>()));

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

    SlotIndexes() : MachineFunctionPass(ID), indexListHead(0) {
      initializeSlotIndexesPass(*PassRegistry::getPassRegistry());
    }

    virtual void getAnalysisUsage(AnalysisUsage &au) const;
    virtual void releaseMemory(); 

    virtual bool runOnMachineFunction(MachineFunction &fn);

    /// Dump the indexes.
    void dump() const;

    /// Renumber the index list, providing space for new instructions.
    void renumberIndexes();

    /// Returns the zero index for this analysis.
    SlotIndex getZeroIndex() {
      assert(front()->getIndex() == 0 && "First index is not 0?");
      return SlotIndex(front(), 0);
    }

    /// Returns the base index of the last slot in this analysis.
    SlotIndex getLastIndex() {
      return SlotIndex(back(), 0);
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

    /// Returns the basic block which the given index falls in.
    MachineBasicBlock* getMBBFromIndex(SlotIndex index) const {
      std::vector<IdxMBBPair>::const_iterator I =
        std::lower_bound(idx2MBBMap.begin(), idx2MBBMap.end(), index);
      // Take the pair containing the index
      std::vector<IdxMBBPair>::const_iterator J =
        ((I != idx2MBBMap.end() && I->first > index) ||
         (I == idx2MBBMap.end() && idx2MBBMap.size()>0)) ? (I-1): I;

      assert(J != idx2MBBMap.end() && J->first <= index &&
             index < getMBBEndIdx(J->second) &&
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

    /// Insert the given machine instruction into the mapping. Returns the
    /// assigned index.
    SlotIndex insertMachineInstrInMaps(MachineInstr *mi,
                                        bool *deferredRenumber = 0) {
      assert(mi2iMap.find(mi) == mi2iMap.end() && "Instr already indexed.");

      MachineBasicBlock *mbb = mi->getParent();

      assert(mbb != 0 && "Instr must be added to function.");

      MBB2IdxMap::iterator mbbRangeItr = mbb2IdxMap.find(mbb);

      assert(mbbRangeItr != mbb2IdxMap.end() &&
             "Instruction's parent MBB has not been added to SlotIndexes.");

      MachineBasicBlock::iterator miItr(mi);
      bool needRenumber = false;
      IndexListEntry *newEntry;
      // Get previous index, considering that not all instructions are indexed.
      IndexListEntry *prevEntry;
      for (;;) {
        // If mi is at the mbb beginning, get the prev index from the mbb.
        if (miItr == mbb->begin()) {
          prevEntry = &mbbRangeItr->second.first.entry();
          break;
        }
        // Otherwise rewind until we find a mapped instruction.
        Mi2IndexMap::const_iterator itr = mi2iMap.find(--miItr);
        if (itr != mi2iMap.end()) {
          prevEntry = &itr->second.entry();
          break;
        }
      }

      // Get next entry from previous entry.
      IndexListEntry *nextEntry = prevEntry->getNext();

      // Get a number for the new instr, or 0 if there's no room currently.
      // In the latter case we'll force a renumber later.
      unsigned dist = nextEntry->getIndex() - prevEntry->getIndex();
      unsigned newNumber = dist > SlotIndex::NUM ?
        prevEntry->getIndex() + ((dist >> 1) & ~3U) : 0;

      if (newNumber == 0) {
        needRenumber = true;
      }

      // Insert a new list entry for mi.
      newEntry = createEntry(mi, newNumber);
      insert(nextEntry, newEntry);
  
      SlotIndex newIndex(newEntry, SlotIndex::LOAD);
      mi2iMap.insert(std::make_pair(mi, newIndex));

      if (miItr == mbb->end()) {
        // If this is the last instr in the MBB then we need to fix up the bb
        // range:
        mbbRangeItr->second.second = SlotIndex(newEntry, SlotIndex::STORE);
      }

      // Renumber if we need to.
      if (needRenumber) {
        if (deferredRenumber == 0)
          renumberIndexes();
        else
          *deferredRenumber = true;
      }

      return newIndex;
    }

    /// Add all instructions in the vector to the index list. This method will
    /// defer renumbering until all instrs have been added, and should be 
    /// preferred when adding multiple instrs.
    void insertMachineInstrsInMaps(SmallVectorImpl<MachineInstr*> &mis) {
      bool renumber = false;

      for (SmallVectorImpl<MachineInstr*>::iterator
           miItr = mis.begin(), miEnd = mis.end();
           miItr != miEnd; ++miItr) {
        insertMachineInstrInMaps(*miItr, &renumber);
      }

      if (renumber)
        renumberIndexes();
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

    /// Add the given MachineBasicBlock into the maps.
    void insertMBBInMaps(MachineBasicBlock *mbb) {
      MachineFunction::iterator nextMBB =
        llvm::next(MachineFunction::iterator(mbb));
      IndexListEntry *startEntry = createEntry(0, 0);
      IndexListEntry *nextEntry = 0;

      if (nextMBB == mbb->getParent()->end()) {
        nextEntry = getTail();
      } else {
        nextEntry = &getMBBStartIdx(nextMBB).entry();
      }

      insert(nextEntry, startEntry);

      SlotIndex startIdx(startEntry, SlotIndex::LOAD);
      SlotIndex endIdx(nextEntry, SlotIndex::LOAD);

      mbb2IdxMap.insert(
        std::make_pair(mbb, std::make_pair(startIdx, endIdx)));

      idx2MBBMap.push_back(IdxMBBPair(startIdx, mbb));

      if (MachineFunction::iterator(mbb) != mbb->getParent()->begin()) {
        // Have to update the end index of the previous block.
        MachineBasicBlock *priorMBB =
          llvm::prior(MachineFunction::iterator(mbb));
        mbb2IdxMap[priorMBB].second = startIdx;
      }

      renumberIndexes();
      std::sort(idx2MBBMap.begin(), idx2MBBMap.end(), Idx2MBBCompare());

    }

  };


}

#endif // LLVM_CODEGEN_LIVEINDEX_H 
