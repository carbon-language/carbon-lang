//=== BasicValueFactory.h - Basic values for Path Sens analysis --*- C++ -*---//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines BasicValueFactory, a class that manages the lifetime
//  of APSInt objects and symbolic constraints used by GRExprEngine 
//  and related classes.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_ANALYSIS_BASICVALUEFACTORY_H
#define LLVM_CLANG_ANALYSIS_BASICVALUEFACTORY_H

#include "clang/Analysis/PathSensitive/SymbolManager.h"
#include "clang/Analysis/PathSensitive/SVals.h"
#include "clang/AST/ASTContext.h"
#include "llvm/ADT/FoldingSet.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/ImmutableList.h"

namespace clang {
  
class CompoundValData : public llvm::FoldingSetNode {
  QualType T;
  llvm::ImmutableList<SVal> L;

public:
  CompoundValData(QualType t, llvm::ImmutableList<SVal> l) 
    : T(t), L(l) {}

  typedef llvm::ImmutableList<SVal>::iterator iterator;
  iterator begin() const { return L.begin(); }
  iterator end() const { return L.end(); }  
  
  static void Profile(llvm::FoldingSetNodeID& ID, QualType T,
                      llvm::ImmutableList<SVal> L);

  void Profile(llvm::FoldingSetNodeID& ID) { Profile(ID, T, L); }
};

class BasicValueFactory {
  typedef llvm::FoldingSet<llvm::FoldingSetNodeWrapper<llvm::APSInt> >
          APSIntSetTy;

  typedef llvm::FoldingSet<SymIntConstraint>
          SymIntCSetTy;
  

  ASTContext& Ctx;
  llvm::BumpPtrAllocator& BPAlloc;

  APSIntSetTy   APSIntSet;
  SymIntCSetTy  SymIntCSet;
  void*         PersistentSVals;
  void*         PersistentSValPairs;

  llvm::ImmutableList<SVal>::Factory SValListFactory;
  llvm::FoldingSet<CompoundValData>  CompoundValDataSet;

public:
  BasicValueFactory(ASTContext& ctx, llvm::BumpPtrAllocator& Alloc) 
  : Ctx(ctx), BPAlloc(Alloc), PersistentSVals(0), PersistentSValPairs(0),
    SValListFactory(Alloc) {}

  ~BasicValueFactory();

  ASTContext& getContext() const { return Ctx; }  

  const llvm::APSInt& getValue(const llvm::APSInt& X);
  const llvm::APSInt& getValue(uint64_t X, unsigned BitWidth, bool isUnsigned);
  const llvm::APSInt& getValue(uint64_t X, QualType T);

  inline const llvm::APSInt& getZeroWithPtrWidth() {
    return getValue(0, Ctx.getTypeSize(Ctx.VoidPtrTy), true);
  }

  inline const llvm::APSInt& getTruthValue(bool b) {
    return getValue(b ? 1 : 0, Ctx.getTypeSize(Ctx.IntTy), false);
  }

  const SymIntConstraint& getConstraint(SymbolID sym, BinaryOperator::Opcode Op,
                                        const llvm::APSInt& V);

  const CompoundValData* getCompoundValData(QualType T, 
                                            llvm::ImmutableList<SVal> Vals);
  
  llvm::ImmutableList<SVal> getEmptySValList() {
    return SValListFactory.GetEmptyList();
  }
  
  llvm::ImmutableList<SVal> consVals(SVal X, llvm::ImmutableList<SVal> L) {
    return SValListFactory.Add(X, L);
  }

  const llvm::APSInt* EvaluateAPSInt(BinaryOperator::Opcode Op,
                                     const llvm::APSInt& V1,
                                     const llvm::APSInt& V2);
  
  const std::pair<SVal, uintptr_t>&
  getPersistentSValWithData(const SVal& V, uintptr_t Data);
  
  const std::pair<SVal, SVal>&
  getPersistentSValPair(const SVal& V1, const SVal& V2);  
  
  const SVal* getPersistentSVal(SVal X);
};

} // end clang namespace

#endif
