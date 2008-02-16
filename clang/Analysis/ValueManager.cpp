// ValueManager.h - Low-level value management for Value Tracking -*- C++ -*--==
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This files defines ValueManager, a class that manages the lifetime of APSInt
//  objects and symbolic constraints used by GRExprEngine and related classes.
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/PathSensitive/ValueManager.h"

using namespace clang;

ValueManager::~ValueManager() {
  // Note that the dstor for the contents of APSIntSet will never be called,
  // so we iterate over the set and invoke the dstor for each APSInt.  This
  // frees an aux. memory allocated to represent very large constants.
  for (APSIntSetTy::iterator I=APSIntSet.begin(), E=APSIntSet.end(); I!=E; ++I)
    I->getValue().~APSInt();
}

const llvm::APSInt& ValueManager::getValue(const llvm::APSInt& X) {
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

const llvm::APSInt& ValueManager::getValue(uint64_t X, unsigned BitWidth,
                                           bool isUnsigned) {
  llvm::APSInt V(BitWidth, isUnsigned);
  V = X;  
  return getValue(V);
}

const llvm::APSInt& ValueManager::getValue(uint64_t X, QualType T,
                                           SourceLocation Loc) {
  
  unsigned bits = Ctx.getTypeSize(T, Loc);
  llvm::APSInt V(bits, T->isUnsignedIntegerType());
  V = X;
  return getValue(V);
}

const SymIntConstraint&
ValueManager::getConstraint(SymbolID sym, BinaryOperator::Opcode Op,
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

const llvm::APSInt&
ValueManager::EvaluateAPSInt(BinaryOperator::Opcode Op,
                             const llvm::APSInt& V1, const llvm::APSInt& V2) {
  
  switch (Op) {
    default:
      assert (false && "Invalid Opcode.");
      
    case BinaryOperator::Mul:
      return getValue( V1 * V2 );
      
    case BinaryOperator::Div:
      return getValue( V1 / V2 );
      
    case BinaryOperator::Rem:
      return getValue( V1 % V2 );
      
    case BinaryOperator::Add:
      return getValue( V1 + V2 );
      
    case BinaryOperator::Sub:
      return getValue( V1 - V2 );
      
    case BinaryOperator::Shl:
      return getValue( V1.operator<<( (unsigned) V2.getZExtValue() ));
      
    case BinaryOperator::Shr:
      return getValue( V1.operator>>( (unsigned) V2.getZExtValue() ));
      
    case BinaryOperator::LT:
      return getTruthValue( V1 < V2 );
      
    case BinaryOperator::GT:
      return getTruthValue( V1 > V2 );
      
    case BinaryOperator::LE:
      return getTruthValue( V1 <= V2 );
      
    case BinaryOperator::GE:
      return getTruthValue( V1 >= V2 );
      
    case BinaryOperator::EQ:
      return getTruthValue( V1 == V2 );
      
    case BinaryOperator::NE:
      return getTruthValue( V1 != V2 );
      
      // Note: LAnd, LOr, Comma are handled specially by higher-level logic.
      
    case BinaryOperator::And:
      return getValue( V1 & V2 );
      
    case BinaryOperator::Or:
      return getValue( V1 | V2 );
  }
}
