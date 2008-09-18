//=== BasicValueFactory.cpp - Basic values for Path Sens analysis --*- C++ -*-//
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

#include "clang/Analysis/PathSensitive/BasicValueFactory.h"
#include "clang/Analysis/PathSensitive/RValues.h"

using namespace clang;

typedef std::pair<RVal, uintptr_t> RValData;
typedef std::pair<RVal, RVal> RValPair;


namespace llvm {
template<> struct FoldingSetTrait<RValData> {
  static inline void Profile(const RValData& X, llvm::FoldingSetNodeID& ID) {
    X.first.Profile(ID);
    ID.AddPointer( (void*) X.second);
  }
};
  
template<> struct FoldingSetTrait<RValPair> {
  static inline void Profile(const RValPair& X, llvm::FoldingSetNodeID& ID) {
    X.first.Profile(ID);
    X.second.Profile(ID);
  }
};
}

typedef llvm::FoldingSet<llvm::FoldingSetNodeWrapper<RValData> >
  PersistentRValsTy;

typedef llvm::FoldingSet<llvm::FoldingSetNodeWrapper<RValPair> >
  PersistentRValPairsTy;

BasicValueFactory::~BasicValueFactory() {
  // Note that the dstor for the contents of APSIntSet will never be called,
  // so we iterate over the set and invoke the dstor for each APSInt.  This
  // frees an aux. memory allocated to represent very large constants.
  for (APSIntSetTy::iterator I=APSIntSet.begin(), E=APSIntSet.end(); I!=E; ++I)
    I->getValue().~APSInt();
  
  delete (PersistentRValsTy*) PersistentRVals;  
  delete (PersistentRValPairsTy*) PersistentRValPairs;
}

const llvm::APSInt& BasicValueFactory::getValue(const llvm::APSInt& X) {
  llvm::FoldingSetNodeID ID;
  void* InsertPos;
  typedef llvm::FoldingSetNodeWrapper<llvm::APSInt> FoldNodeTy;
  
  X.Profile(ID);
  FoldNodeTy* P = APSIntSet.FindNodeOrInsertPos(ID, InsertPos);
  
  if (!P) {  
    P = (FoldNodeTy*) BPAlloc.Allocate<FoldNodeTy>();
    new (P) FoldNodeTy(X);
    APSIntSet.InsertNode(P, InsertPos);
  }
  
  return *P;
}

const llvm::APSInt& BasicValueFactory::getValue(uint64_t X, unsigned BitWidth,
                                           bool isUnsigned) {
  llvm::APSInt V(BitWidth, isUnsigned);
  V = X;  
  return getValue(V);
}

const llvm::APSInt& BasicValueFactory::getValue(uint64_t X, QualType T) {
  
  unsigned bits = Ctx.getTypeSize(T);
  llvm::APSInt V(bits, T->isUnsignedIntegerType());
  V = X;
  return getValue(V);
}

const SymIntConstraint&
BasicValueFactory::getConstraint(SymbolID sym, BinaryOperator::Opcode Op,
                            const llvm::APSInt& V) {
  
  llvm::FoldingSetNodeID ID;
  SymIntConstraint::Profile(ID, sym, Op, V);
  void* InsertPos;
  
  SymIntConstraint* C = SymIntCSet.FindNodeOrInsertPos(ID, InsertPos);
  
  if (!C) {
    C = (SymIntConstraint*) BPAlloc.Allocate<SymIntConstraint>();
    new (C) SymIntConstraint(sym, Op, V);
    SymIntCSet.InsertNode(C, InsertPos);
  }
  
  return *C;
}

const llvm::APSInt*
BasicValueFactory::EvaluateAPSInt(BinaryOperator::Opcode Op,
                             const llvm::APSInt& V1, const llvm::APSInt& V2) {
  
  switch (Op) {
    default:
      assert (false && "Invalid Opcode.");
      
    case BinaryOperator::Mul:
      return &getValue( V1 * V2 );
      
    case BinaryOperator::Div:
      return &getValue( V1 / V2 );
      
    case BinaryOperator::Rem:
      return &getValue( V1 % V2 );
      
    case BinaryOperator::Add:
      return &getValue( V1 + V2 );
      
    case BinaryOperator::Sub:
      return &getValue( V1 - V2 );
      
    case BinaryOperator::Shl: {

      // FIXME: This logic should probably go higher up, where we can
      // test these conditions symbolically.
      
      // FIXME: Expand these checks to include all undefined behavior.
      
      if (V2.isSigned() && V2.isNegative())
        return NULL;
      
      uint64_t Amt = V2.getZExtValue();
      
      if (Amt > V1.getBitWidth())
        return NULL;
      
      return &getValue( V1.operator<<( (unsigned) Amt ));
    }
      
    case BinaryOperator::Shr: {
      
      // FIXME: This logic should probably go higher up, where we can
      // test these conditions symbolically.
      
      // FIXME: Expand these checks to include all undefined behavior.
      
      if (V2.isSigned() && V2.isNegative())
        return NULL;
      
      uint64_t Amt = V2.getZExtValue();
      
      if (Amt > V1.getBitWidth())
        return NULL;
      
      return &getValue( V1.operator>>( (unsigned) Amt ));
    }
      
    case BinaryOperator::LT:
      return &getTruthValue( V1 < V2 );
      
    case BinaryOperator::GT:
      return &getTruthValue( V1 > V2 );
      
    case BinaryOperator::LE:
      return &getTruthValue( V1 <= V2 );
      
    case BinaryOperator::GE:
      return &getTruthValue( V1 >= V2 );
      
    case BinaryOperator::EQ:
      return &getTruthValue( V1 == V2 );
      
    case BinaryOperator::NE:
      return &getTruthValue( V1 != V2 );
      
      // Note: LAnd, LOr, Comma are handled specially by higher-level logic.
      
    case BinaryOperator::And:
      return &getValue( V1 & V2 );
      
    case BinaryOperator::Or:
      return &getValue( V1 | V2 );
      
    case BinaryOperator::Xor:
      return &getValue( V1 ^ V2 );
  }
}


const std::pair<RVal, uintptr_t>&
BasicValueFactory::getPersistentRValWithData(const RVal& V, uintptr_t Data) {
  
  // Lazily create the folding set.
  if (!PersistentRVals) PersistentRVals = new PersistentRValsTy();
    
  llvm::FoldingSetNodeID ID;
  void* InsertPos;
  V.Profile(ID);
  ID.AddPointer((void*) Data);
  
  PersistentRValsTy& Map = *((PersistentRValsTy*) PersistentRVals);
  
  typedef llvm::FoldingSetNodeWrapper<RValData> FoldNodeTy;
  FoldNodeTy* P = Map.FindNodeOrInsertPos(ID, InsertPos);
  
  if (!P) {  
    P = (FoldNodeTy*) BPAlloc.Allocate<FoldNodeTy>();
    new (P) FoldNodeTy(std::make_pair(V, Data));
    Map.InsertNode(P, InsertPos);
  }

  return P->getValue();
}

const std::pair<RVal, RVal>&
BasicValueFactory::getPersistentRValPair(const RVal& V1, const RVal& V2) {
  
  // Lazily create the folding set.
  if (!PersistentRValPairs) PersistentRValPairs = new PersistentRValPairsTy();
  
  llvm::FoldingSetNodeID ID;
  void* InsertPos;
  V1.Profile(ID);
  V2.Profile(ID);
  
  PersistentRValPairsTy& Map = *((PersistentRValPairsTy*) PersistentRValPairs);
  
  typedef llvm::FoldingSetNodeWrapper<RValPair> FoldNodeTy;
  FoldNodeTy* P = Map.FindNodeOrInsertPos(ID, InsertPos);
  
  if (!P) {  
    P = (FoldNodeTy*) BPAlloc.Allocate<FoldNodeTy>();
    new (P) FoldNodeTy(std::make_pair(V1, V2));
    Map.InsertNode(P, InsertPos);
  }
  
  return P->getValue();
}

const RVal* BasicValueFactory::getPersistentRVal(RVal X) {
  return &getPersistentRValWithData(X, 0).first;
}  


