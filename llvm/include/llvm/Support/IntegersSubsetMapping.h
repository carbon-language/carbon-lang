//===- IntegersSubsetMapping.h - Mapping subset ==> Successor ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
/// @file
/// IntegersSubsetMapping is mapping from A to B, where
/// Items in A is subsets of integers,
/// Items in B some pointers (Successors).
/// If user which to add another subset for successor that is already
/// exists in mapping, IntegersSubsetMapping merges existing subset with
/// added one.
//
//===----------------------------------------------------------------------===//

#ifndef CRSBUILDER_H_
#define CRSBUILDER_H_

#include "llvm/Support/IntegersSubset.h"
#include <list>
#include <map>
#include <vector>

namespace llvm {

template <class SuccessorClass,
          class IntegersSubsetTy = IntegersSubset,
          class IntTy = IntItem>
class IntegersSubsetMapping {
public:
  
  typedef IntRange<IntTy> RangeTy;
  
  struct RangeEx : public RangeTy {
    RangeEx() : Weight(1) {}
    RangeEx(const RangeTy &R) : RangeTy(R), Weight(1) {}
    RangeEx(const IntTy &C) : RangeTy(C), Weight(1) {}
    RangeEx(const IntTy &L, const IntTy &H) : RangeTy(L, H), Weight(1) {}
    RangeEx(const IntTy &L, const IntTy &H, unsigned W) :
      RangeTy(L, H), Weight(W) {}
    unsigned Weight;
  };

  typedef std::pair<RangeEx, SuccessorClass*> Cluster;

protected:

  typedef std::list<Cluster> CaseItems;
  typedef typename CaseItems::iterator CaseItemIt;
  typedef typename CaseItems::const_iterator CaseItemConstIt;
  
  typedef std::list<RangeTy> RangesCollection;
  typedef typename RangesCollection::iterator RangesCollectionIt;
  
  // TODO: Change unclean CRS prefixes to SubsetMap for example.
  typedef std::map<SuccessorClass*, RangesCollection > CRSMap;
  typedef typename CRSMap::iterator CRSMapIt;

  struct ClustersCmp {
    bool operator()(const Cluster &C1, const Cluster &C2) {
      return C1.first < C2.first;
    }
  };
  
  CaseItems Items;
  bool Sorted;
  
  bool isIntersected(CaseItemIt& LItem, CaseItemIt& RItem) {
    return LItem->first.getHigh() >= RItem->first.getLow();
  }

  bool isJoinable(CaseItemIt& LItem, CaseItemIt& RItem) {
    if (LItem->second != RItem->second) {
      assert(!isIntersected(LItem, RItem) &&
             "Intersected items with different successors!");
      return false;
    }
    APInt RLow = RItem->first.getLow();
    if (RLow != APInt::getNullValue(RLow.getBitWidth()))
      --RLow;
    return LItem->first.getHigh() >= RLow;
  }
  
  void sort() {
    if (!Sorted) {
      std::vector<Cluster> clustersVector;
      clustersVector.reserve(Items.size());
      clustersVector.insert(clustersVector.begin(), Items.begin(), Items.end());
      std::sort(clustersVector.begin(), clustersVector.end(), ClustersCmp());
      Items.clear();
      Items.insert(Items.begin(), clustersVector.begin(), clustersVector.end());
      Sorted = true;
    }
  }

public:
  
  // Don't public CaseItems itself. Don't allow edit the Items directly. 
  // Just present the user way to iterate over the internal collection
  // sharing iterator, begin() and end(). Editing should be controlled by
  // factory.
  typedef CaseItemIt RangeIterator;
  
  typedef std::pair<SuccessorClass*, IntegersSubsetTy> Case;
  typedef std::list<Case> Cases;
  
  IntegersSubsetMapping() {
    Sorted = false;
  }
  
  bool verify(RangeIterator& errItem) {
    if (Items.empty())
      return true;
    sort();
    for (CaseItemIt j = Items.begin(), i = j++, e = Items.end();
         j != e; i = j++) {
      if (isIntersected(i, j) && i->second != j->second) {
        errItem = j;
        return false;
      }
    }
    return true;
  }
  
  void optimize() {
    if (Items.size() < 2)
      return;
    sort();
    CaseItems OldItems = Items;
    Items.clear();
    const IntTy *Low = &OldItems.begin()->first.getLow();
    const IntTy *High = &OldItems.begin()->first.getHigh();
    unsigned Weight = 1;
    SuccessorClass *Successor = OldItems.begin()->second;
    for (CaseItemIt j = OldItems.begin(), i = j++, e = OldItems.end();
         j != e; i = j++) {
      if (isJoinable(i, j)) {
        const IntTy *CurHigh = &j->first.getHigh();
        ++Weight;
        if (*CurHigh > *High)
          High = CurHigh;
      } else {
        RangeEx R(*Low, *High, Weight);
        add(R, Successor);
        Low = &j->first.getLow();
        High = &j->first.getHigh(); 
        Weight = 1;
        Successor = j->second;
      }
    }
    RangeEx R(*Low, *High, Weight);
    add(R, Successor);
    // We recollected the Items, but we kept it sorted.
    Sorted = true;
  }
  
  /// Adds a constant value.
  void add(const IntTy &C, SuccessorClass *S = 0) {
    RangeTy R(C);
    add(R, S);
  }
  
  /// Adds a range.
  void add(const IntTy &Low, const IntTy &High, SuccessorClass *S = 0) {
    RangeTy R(Low, High);
    add(R, S);
  }
  void add(const RangeTy &R, SuccessorClass *S = 0) {
    RangeEx REx = R;
    add(REx, S);
  }   
  void add(const RangeEx &R, SuccessorClass *S = 0) {
    Items.push_back(std::make_pair(R, S));
    Sorted = false;
  }  
  
  /// Adds all ranges and values from given ranges set to the current
  /// mapping.
  void add(const IntegersSubsetTy &CRS, SuccessorClass *S = 0) {
    for (unsigned i = 0, e = CRS.getNumItems(); i < e; ++i) {
      RangeTy R = CRS.getItem(i);
      add(R, S);
    }
  }
  
  /// Removes items from set.
  void removeItem(RangeIterator i) { Items.erase(i); }
  
  /// Builds the finalized case objects.
  void getCases(Cases& TheCases) {
    CRSMap TheCRSMap;
    for (RangeIterator i = this->begin(); i != this->end(); ++i)
      TheCRSMap[i->second].push_back(i->first);
    for (CRSMapIt i = TheCRSMap.begin(), e = TheCRSMap.end(); i != e; ++i)
      TheCases.push_back(std::make_pair(i->first, IntegersSubsetTy(i->second)));
  }
  
  /// Builds the finalized case objects ignoring successor values, as though
  /// all ranges belongs to the same successor.
  IntegersSubsetTy getCase() {
    RangesCollection Ranges;
    for (RangeIterator i = this->begin(); i != this->end(); ++i)
      Ranges.push_back(i->first);
    return IntegersSubsetTy(Ranges);
  }  
  
  /// Returns true if there is no ranges and values inside.
  bool empty() const { return Items.empty(); }
  
  RangeIterator begin() { return Items.begin(); }
  RangeIterator end() { return Items.end(); }
};

class BasicBlock;
typedef IntegersSubsetMapping<BasicBlock> IntegersSubsetToBB;

}

#endif /* CRSBUILDER_H_ */
