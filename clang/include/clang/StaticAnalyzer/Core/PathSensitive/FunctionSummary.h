//== FunctionSummary.h - Stores summaries of functions. ------------*- C++ -*-//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines a summary of a function gathered/used by static analysis.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_GR_FUNCTIONSUMMARY_H
#define LLVM_CLANG_GR_FUNCTIONSUMMARY_H

#include "clang/AST/Decl.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include <deque>

namespace clang {
namespace ento {
typedef std::deque<Decl*> SetOfDecls;
typedef llvm::DenseSet<const Decl*> SetOfConstDecls;

class FunctionSummariesTy {
  struct FunctionSummary {
    /// Marks the IDs of the basic blocks visited during the analyzes.
    llvm::SmallBitVector VisitedBasicBlocks;

    /// Total number of blocks in the function.
    unsigned TotalBasicBlocks : 31;

    /// True if this function has reached a max block count while inlined from
    /// at least one call site.
    unsigned MayReachMaxBlockCount : 1;

    /// The number of times the function has been inlined.
    unsigned TimesInlined : 32;

    FunctionSummary() :
      TotalBasicBlocks(0),
      MayReachMaxBlockCount(0),
      TimesInlined(0) {}
  };

  typedef llvm::DenseMap<const Decl *, FunctionSummary> MapTy;
  MapTy Map;

public:
  MapTy::iterator findOrInsertSummary(const Decl *D) {
    MapTy::iterator I = Map.find(D);
    if (I != Map.end())
      return I;

    typedef std::pair<const Decl *, FunctionSummary> KVPair;
    I = Map.insert(KVPair(D, FunctionSummary())).first;
    assert(I != Map.end());
    return I;
  }

  void markReachedMaxBlockCount(const Decl* D) {
    MapTy::iterator I = findOrInsertSummary(D);
    I->second.MayReachMaxBlockCount = 1;
  }

  bool hasReachedMaxBlockCount(const Decl* D) {
  MapTy::const_iterator I = Map.find(D);
    if (I != Map.end())
      return I->second.MayReachMaxBlockCount;
    return false;
  }

  void markVisitedBasicBlock(unsigned ID, const Decl* D, unsigned TotalIDs) {
    MapTy::iterator I = findOrInsertSummary(D);
    llvm::SmallBitVector &Blocks = I->second.VisitedBasicBlocks;
    assert(ID < TotalIDs);
    if (TotalIDs > Blocks.size()) {
      Blocks.resize(TotalIDs);
      I->second.TotalBasicBlocks = TotalIDs;
    }
    Blocks.set(ID);
  }

  unsigned getNumVisitedBasicBlocks(const Decl* D) {
    MapTy::const_iterator I = Map.find(D);
    if (I != Map.end())
      return I->second.VisitedBasicBlocks.count();
    return 0;
  }

  unsigned getNumTimesInlined(const Decl* D) {
    MapTy::const_iterator I = Map.find(D);
    if (I != Map.end())
      return I->second.TimesInlined;
    return 0;
  }

  void bumpNumTimesInlined(const Decl* D) {
    MapTy::iterator I = findOrInsertSummary(D);
    I->second.TimesInlined++;
  }

  /// Get the percentage of the reachable blocks.
  unsigned getPercentBlocksReachable(const Decl *D) {
    MapTy::const_iterator I = Map.find(D);
      if (I != Map.end())
        return ((I->second.VisitedBasicBlocks.count() * 100) /
                 I->second.TotalBasicBlocks);
    return 0;
  }

  unsigned getTotalNumBasicBlocks();
  unsigned getTotalNumVisitedBasicBlocks();

};

}} // end clang ento namespaces

#endif
