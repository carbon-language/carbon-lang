//===- llvm/Analysis/AliasSetTracker.h - Build Alias Sets -------*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file defines two classes: AliasSetTracker and AliasSet.  These interface
// are used to classify a collection of pointer references into a maximal number
// of disjoint sets.  Each AliasSet object constructed by the AliasSetTracker
// object refers to memory disjoint from the other sets.
// 
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_ALIASSETTRACKER_H
#define LLVM_ANALYSIS_ALIASSETTRACKER_H

#include "llvm/Support/CallSite.h"
#include "Support/iterator"
#include "Support/hash_map"
#include "Support/ilist"

namespace llvm {

class AliasAnalysis;
class LoadInst;
class StoreInst;
class AliasSetTracker;
class AliasSet;

class AliasSet {
  friend class AliasSetTracker;

  struct PointerRec;
  typedef std::pair<Value* const, PointerRec> HashNodePair;

  class PointerRec {
    HashNodePair *NextInList;
    AliasSet *AS;
    unsigned Size;
  public:
    PointerRec() : NextInList(0), AS(0), Size(0) {}

    HashNodePair *getNext() const { return NextInList; }
    bool hasAliasSet() const { return AS != 0; }

    void updateSize(unsigned NewSize) {
      if (NewSize > Size) Size = NewSize;
    }

    unsigned getSize() const { return Size; }

    AliasSet *getAliasSet(AliasSetTracker &AST) { 
      assert(AS && "No AliasSet yet!");
      if (AS->Forward) {
        AliasSet *OldAS = AS;
        AS = OldAS->getForwardedTarget(AST);
        if (--OldAS->RefCount == 0)
          OldAS->removeFromTracker(AST);
        AS->RefCount++;
      }
      return AS;
    }

    void setAliasSet(AliasSet *as) {
      assert(AS == 0 && "Already have an alias set!");
      AS = as;
    }
    void setTail(HashNodePair *T) {
      assert(NextInList == 0 && "Already have tail!");
      NextInList = T;
    }
  };

  HashNodePair *PtrListHead, *PtrListTail; // Singly linked list of nodes
  AliasSet *Forward;             // Forwarding pointer
  AliasSet *Next, *Prev;         // Doubly linked list of AliasSets

  std::vector<CallSite> CallSites; // All calls & invokes in this node

  // RefCount - Number of nodes pointing to this AliasSet plus the number of
  // AliasSets forwarding to it.
  unsigned RefCount : 28;

  /// AccessType - Keep track of whether this alias set merely refers to the
  /// locations of memory, whether it modifies the memory, or whether it does
  /// both.  The lattice goes from "NoModRef" to either Refs or Mods, then to
  /// ModRef as necessary.
  ///
  enum AccessType {
    NoModRef = 0, Refs = 1,         // Ref = bit 1
    Mods     = 2, ModRef = 3        // Mod = bit 2
  };
  unsigned AccessTy : 2;

  /// AliasType - Keep track the relationships between the pointers in the set.
  /// Lattice goes from MustAlias to MayAlias.
  ///
  enum AliasType {
    MustAlias = 0, MayAlias = 1
  };
  unsigned AliasTy : 1;

  // Volatile - True if this alias set contains volatile loads or stores.
  bool Volatile : 1;

  friend class ilist_traits<AliasSet>;
  AliasSet *getPrev() const { return Prev; }
  AliasSet *getNext() const { return Next; }
  void setPrev(AliasSet *P) { Prev = P; }
  void setNext(AliasSet *N) { Next = N; }

public:
  /// Accessors...
  bool isRef() const { return AccessTy & Refs; }
  bool isMod() const { return AccessTy & Mods; }
  bool isMustAlias() const { return AliasTy == MustAlias; }
  bool isMayAlias()  const { return AliasTy == MayAlias; }

  // isVolatile - Return true if this alias set contains volatile loads or
  // stores.
  bool isVolatile() const { return Volatile; }


  /// isForwardingAliasSet - Return true if this alias set should be ignored as
  /// part of the AliasSetTracker object.
  bool isForwardingAliasSet() const { return Forward; }

  /// mergeSetIn - Merge the specified alias set into this alias set...
  ///
  void mergeSetIn(AliasSet &AS);

  // Alias Set iteration - Allow access to all of the pointer which are part of
  // this alias set...
  class iterator;
  iterator begin() const { return iterator(PtrListHead); }
  iterator end()   const { return iterator(); }

  void print(std::ostream &OS) const;
  void dump() const;

  /// Define an iterator for alias sets... this is just a forward iterator.
  class iterator : public forward_iterator<HashNodePair, ptrdiff_t> {
    HashNodePair *CurNode;
  public:
    iterator(HashNodePair *CN = 0) : CurNode(CN) {}
    
    bool operator==(const iterator& x) const {
      return CurNode == x.CurNode;
    }
    bool operator!=(const iterator& x) const { return !operator==(x); }

    const iterator &operator=(const iterator &I) {
      CurNode = I.CurNode;
      return *this;
    }
  
    value_type &operator*() const {
      assert(CurNode && "Dereferencing AliasSet.end()!");
      return *CurNode;
    }
    value_type *operator->() const { return &operator*(); }
  
    iterator& operator++() {                // Preincrement
      assert(CurNode && "Advancing past AliasSet.end()!");
      CurNode = CurNode->second.getNext();
      return *this;
    }
    iterator operator++(int) { // Postincrement
      iterator tmp = *this; ++*this; return tmp; 
    }
  };

private:
  // Can only be created by AliasSetTracker
  AliasSet() : PtrListHead(0), PtrListTail(0), Forward(0), RefCount(0),
               AccessTy(NoModRef), AliasTy(MustAlias), Volatile(false) {
  }
  HashNodePair *getSomePointer() const {
    return PtrListHead ? PtrListHead : 0;
  }

  /// getForwardedTarget - Return the real alias set this represents.  If this
  /// has been merged with another set and is forwarding, return the ultimate
  /// destination set.  This also implements the union-find collapsing as well.
  AliasSet *getForwardedTarget(AliasSetTracker &AST) {
    if (!Forward) return this;

    AliasSet *Dest = Forward->getForwardedTarget(AST);
    if (Dest != Forward) {
      Dest->RefCount++;
      if (--Forward->RefCount == 0)
        Forward->removeFromTracker(AST);
      Forward = Dest;
    }
    return Dest;
  }

  void removeFromTracker(AliasSetTracker &AST);

  void addPointer(AliasSetTracker &AST, HashNodePair &Entry, unsigned Size);
  void addCallSite(CallSite CS);
  void setVolatile() { Volatile = true; }

  /// aliasesPointer - Return true if the specified pointer "may" (or must)
  /// alias one of the members in the set.
  ///
  bool aliasesPointer(const Value *Ptr, unsigned Size, AliasAnalysis &AA) const;
  bool aliasesCallSite(CallSite CS, AliasAnalysis &AA) const;
};

inline std::ostream& operator<<(std::ostream &OS, const AliasSet &AS) {
  AS.print(OS);
  return OS;
}


class AliasSetTracker {
  AliasAnalysis &AA;
  ilist<AliasSet> AliasSets;

  // Map from pointers to their node
  hash_map<Value*, AliasSet::PointerRec> PointerMap;
public:
  /// AliasSetTracker ctor - Create an empty collection of AliasSets, and use
  /// the specified alias analysis object to disambiguate load and store
  /// addresses.
  AliasSetTracker(AliasAnalysis &aa) : AA(aa) {}

  /// add methods - These methods are used to add different types of
  /// instructions to the alias sets.  Adding a new instruction can result in
  /// one of three actions happening:
  ///
  ///   1. If the instruction doesn't alias any other sets, create a new set.
  ///   2. If the instruction aliases exactly one set, add it to the set
  ///   3. If the instruction aliases multiple sets, merge the sets, and add
  ///      the instruction to the result.
  ///
  void add(LoadInst *LI);
  void add(StoreInst *SI);
  void add(CallSite CS);          // Call/Invoke instructions
  void add(CallInst *CI)   { add(CallSite(CI)); }
  void add(InvokeInst *II) { add(CallSite(II)); }
  void add(Instruction *I);       // Dispatch to one of the other add methods...
  void add(BasicBlock &BB);       // Add all instructions in basic block
  void add(const AliasSetTracker &AST); // Add alias relations from another AST

  /// getAliasSets - Return the alias sets that are active.
  const ilist<AliasSet> &getAliasSets() const { return AliasSets; }

  /// getAliasSetForPointer - Return the alias set that the specified pointer
  /// lives in...
  AliasSet &getAliasSetForPointer(Value *P, unsigned Size);

  /// getAliasAnalysis - Return the underlying alias analysis object used by
  /// this tracker.
  AliasAnalysis &getAliasAnalysis() const { return AA; }

  typedef ilist<AliasSet>::iterator iterator;
  typedef ilist<AliasSet>::const_iterator const_iterator;

  const_iterator begin() const { return AliasSets.begin(); }
  const_iterator end()   const { return AliasSets.end(); }

  iterator begin() { return AliasSets.begin(); }
  iterator end()   { return AliasSets.end(); }

  void print(std::ostream &OS) const;
  void dump() const;

private:
  friend class AliasSet;
  void removeAliasSet(AliasSet *AS);

  AliasSet::HashNodePair &getEntryFor(Value *V) {
    // Standard operator[], except that it returns the whole pair, not just
    // ->second.
    return *PointerMap.insert(AliasSet::HashNodePair(V,
                                            AliasSet::PointerRec())).first;
  }

  AliasSet &addPointer(Value *P, unsigned Size, AliasSet::AccessType E) {
    AliasSet &AS = getAliasSetForPointer(P, Size);
    AS.AccessTy |= E;
    return AS;
  }
  AliasSet *findAliasSetForPointer(const Value *Ptr, unsigned Size);

  AliasSet *findAliasSetForCallSite(CallSite CS);
};

inline std::ostream& operator<<(std::ostream &OS, const AliasSetTracker &AST) {
  AST.print(OS);
  return OS;
}

} // End llvm namespace

#endif
