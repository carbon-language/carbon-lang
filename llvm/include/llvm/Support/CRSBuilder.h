//===- CRSBuilder.h - ConstantRangesSet Builder -----------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
/// @file
/// CRSBuilder allows to build and parse ConstantRangesSet objects.
/// There is such features like add/remove range, or combine
/// Two ConstantRangesSet object with neighboring ranges merging.
/// Set IsReadonly=true if you want to operate with "const ConstantInt" and
/// "const ConstantRangesSet" objects.
//
//===----------------------------------------------------------------------===//

#ifndef CRSBUILDER_H_
#define CRSBUILDER_H_

#include "llvm/Support/ConstantRangesSet.h"
#include <list>
#include <map>
#include <vector>

namespace llvm {

template <class SuccessorClass, bool IsReadonly>
class CRSBuilderBase {
public:
  
  typedef ConstantRangesSet::RangeT<IsReadonly> RangeTy;
  
  struct RangeEx : public RangeTy {
    typedef ConstantRangesSet::RangeT<IsReadonly> RangeTy;
    typedef typename RangeTy::ConstantIntTy ConstantIntTy;
    RangeEx() : Weight(1) {}
    RangeEx(const RangeTy &R) : RangeTy(R.Low, R.High), Weight(1) {}
    RangeEx(ConstantIntTy *C) : RangeTy(C), Weight(1) {}
    RangeEx(ConstantIntTy *L, ConstantIntTy *H) : RangeTy(L, H), Weight(1) {}
    RangeEx(ConstantIntTy *L, ConstantIntTy *H, unsigned W) :
      RangeTy(L, H), Weight(W) {}
    unsigned Weight;
  };

  typedef std::pair<RangeEx, SuccessorClass*> Cluster;

protected:

  typedef std::vector<Cluster> CaseItems;
  typedef typename CaseItems::iterator CaseItemIt;
  typedef typename CaseItems::const_iterator CaseItemConstIt;

  struct ClustersCmp {
    bool operator()(const Cluster &C1, const Cluster &C2) {
      return C1.first < C2.first;
    }
  };
  
  CaseItems Items;
  bool Sorted;
  
  bool isIntersected(CaseItemIt& LItem, CaseItemIt& RItem) {
    return LItem->first.High->getValue().uge(RItem->first.Low->getValue());
  }

  bool isJoinable(CaseItemIt& LItem, CaseItemIt& RItem) {
    if (LItem->second != RItem->second) {
      assert(!isIntersected(LItem, RItem) &&
             "Intersected items with different successors!");
      return false;
    }
    APInt RLow = RItem->first.Low->getValue();
    if (RLow != APInt::getNullValue(RLow.getBitWidth()))
      --RLow;
    return LItem->first.High->getValue().uge(RLow);
  }
  
  void sort() {
    if (!Sorted) {
      std::sort(Items.begin(), Items.end(), ClustersCmp());
      Sorted = true;
    }
  }
  
public:
  
  typedef typename CRSConstantTypes<IsReadonly>::ConstantIntTy ConstantIntTy;
  typedef typename CRSConstantTypes<IsReadonly>::ConstantRangesSetTy ConstantRangesSetTy;
  
  // Don't public CaseItems itself. Don't allow edit the Items directly. 
  // Just present the user way to iterate over the internal collection
  // sharing iterator, begin() and end(). Editing should be controlled by
  // factory.
  typedef CaseItemIt RangeIterator;
  
  CRSBuilderBase() {
    Items.reserve(32);
    Sorted = false;
  }
  
  bool verify() {
    if (Items.empty())
      return true;
    sort();
    for (CaseItemIt i = Items.begin(), j = i+1, e = Items.end();
         j != e; i = j++) {
      if (isIntersected(j, i) && j->second != i->second)
        return false;
    }
    return true;
  }
  
  void optimize() {
    if (Items.size() < 2)
      return;
    sort();
    CaseItems OldItems = Items;
    Items.clear();
    ConstantIntTy *Low = OldItems.begin()->first.Low;
    ConstantIntTy *High = OldItems.begin()->first.High;
    unsigned Weight = 1;
    SuccessorClass *Successor = OldItems.begin()->second;
    for (CaseItemIt i = OldItems.begin(), j = i+1, e = OldItems.end();
        j != e; i = j++) {
      if (isJoinable(i, j)) {
        ConstantIntTy *CurHigh = j->first.High;
        ++Weight;
        if (CurHigh->getValue().ugt(High->getValue()))
          High = CurHigh;
      } else {
        RangeEx R(Low, High, Weight);
        add(R, Successor);
        Low = j->first.Low;
        High = j->first.High; 
        Weight = 1;
        Successor = j->second;
      }
    }
    RangeEx R(Low, High, Weight);
    add(R, Successor);
    // We recollected the Items, but we kept it sorted.
    Sorted = true;
  }
  
  /// Adds a constant value.
  void add(ConstantIntTy *C, SuccessorClass *S = 0) {
    RangeTy R(C);
    add(R, S);
  }
  
  /// Adds a range.
  void add(ConstantIntTy *Low, ConstantIntTy *High, SuccessorClass *S = 0) {
    RangeTy R(Low, High);
    add(R, S);
  }
  void add(RangeTy &R, SuccessorClass *S = 0) {
    RangeEx REx = R;
    add(REx, S);
  }   
  void add(RangeEx &R, SuccessorClass *S = 0) {
    Items.push_back(std::make_pair(R, S));
    Sorted = false;
  }  
  
  /// Adds all ranges and values from given ranges set to the current
  /// CRSBuilder object.
  void add(ConstantRangesSetTy &CRS, SuccessorClass *S = 0) {
    for (unsigned i = 0, e = CRS.getNumItems(); i < e; ++i) {
      RangeTy R = CRS.getItem(i);
      add(R, S);
    }
  }
  
  /// Removes items from set.
  void removeItem(RangeIterator i) { Items.erase(i); }
  
  /// Returns true if there is no ranges and values inside.
  bool empty() const { return Items.empty(); }
  
  RangeIterator begin() { return Items.begin(); }
  RangeIterator end() { return Items.end(); }
};

template <class SuccessorClass>
class CRSBuilderT : public CRSBuilderBase<SuccessorClass, false> {
  
  typedef typename CRSBuilderBase<SuccessorClass, false>::RangeTy RangeTy;
  typedef typename CRSBuilderBase<SuccessorClass, false>::RangeIterator
      RangeIterator;
  
  typedef std::list<RangeTy> RangesCollection;
  typedef typename RangesCollection::iterator RangesCollectionIt;
  
  typedef std::map<SuccessorClass*, RangesCollection > CRSMap;
  typedef typename CRSMap::iterator CRSMapIt;
  
  ConstantRangesSet getCase(RangesCollection& Src) {
    std::vector<Constant*> Elts;
    Elts.reserve(Src.size());
    for (RangesCollectionIt i = Src.begin(), e = Src.end(); i != e; ++i) {
      const RangeTy &R = *i;
      std::vector<Constant*> r;
      if (R.Low != R.High) {
        r.reserve(2);
        r.push_back(R.Low);
        r.push_back(R.High);
      } else {
        r.reserve(1);
        r.push_back(R.Low);
      }
      Constant *CV = ConstantVector::get(r);
      Elts.push_back(CV);    
    }
    ArrayType *ArrTy =
        ArrayType::get(Elts.front()->getType(), (uint64_t)Elts.size());
    Constant *Array = ConstantArray::get(ArrTy, Elts);
    return ConstantRangesSet(Array);     
  }

public:
  
  typedef std::pair<SuccessorClass*, ConstantRangesSet> Case;
  typedef std::list<Case> Cases;
  
  /// Builds the finalized case objects.
  void getCases(Cases& TheCases) {
    CRSMap TheCRSMap;
    for (RangeIterator i = this->begin(); i != this->end(); ++i)
      TheCRSMap[i->second].push_back(i->first);
    for (CRSMapIt i = TheCRSMap.begin(), e = TheCRSMap.end(); i != e; ++i)
      TheCases.push_back(std::make_pair(i->first, getCase(i->second)));
  }
  
  /// Builds the finalized case objects ignoring successor values, as though
  /// all ranges belongs to the same successor.
  ConstantRangesSet getCase() {
    RangesCollection Ranges;
    for (RangeIterator i = this->begin(); i != this->end(); ++i)
      Ranges.push_back(i->first);
    return getCase(Ranges);
  }
};

class BasicBlock;
typedef CRSBuilderT<BasicBlock> CRSBuilder;
typedef CRSBuilderBase<BasicBlock, true> CRSBuilderConst;  

}

#endif /* CRSBUILDER_H_ */
