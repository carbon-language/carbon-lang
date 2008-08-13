//== SymbolManager.h - Management of Symbolic Values ------------*- C++ -*--==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines SymbolManager, a class that manages symbolic values
//  created for use by GRExprEngine and related classes.
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/PathSensitive/SymbolManager.h"

using namespace clang;

SymbolID SymbolManager::getSymbol(VarDecl* D) {

  assert (isa<ParmVarDecl>(D) || isa<ImplicitParamDecl>(D) || 
          D->hasGlobalStorage());
  
  llvm::FoldingSetNodeID profile;
  
  ParmVarDecl* PD = dyn_cast<ParmVarDecl>(D);
  
  if (PD)
    SymbolDataParmVar::Profile(profile, PD);
  else
    SymbolDataGlobalVar::Profile(profile, D);
  
  void* InsertPos;
  
  SymbolData* SD = DataSet.FindNodeOrInsertPos(profile, InsertPos);

  if (SD)
    return SD->getSymbol();
  
  if (PD) {
    SD = (SymbolData*) BPAlloc.Allocate<SymbolDataParmVar>();
    new (SD) SymbolDataParmVar(SymbolCounter, PD);
  }
  else {
    SD = (SymbolData*) BPAlloc.Allocate<SymbolDataGlobalVar>();
    new (SD) SymbolDataGlobalVar(SymbolCounter, D);
  }
  
  DataSet.InsertNode(SD, InsertPos);
  
  DataMap[SymbolCounter] = SD;
  return SymbolCounter++;
}  
 
SymbolID SymbolManager::getContentsOfSymbol(SymbolID sym) {
  
  llvm::FoldingSetNodeID profile;
  SymbolDataContentsOf::Profile(profile, sym);
  void* InsertPos;
  
  SymbolData* SD = DataSet.FindNodeOrInsertPos(profile, InsertPos);
  
  if (SD)
    return SD->getSymbol();

  SD = (SymbolData*) BPAlloc.Allocate<SymbolDataContentsOf>();
  new (SD) SymbolDataContentsOf(SymbolCounter, sym);


  DataSet.InsertNode(SD, InsertPos);  
  DataMap[SymbolCounter] = SD;
  
  return SymbolCounter++;
}
  
SymbolID SymbolManager::getConjuredSymbol(Expr* E, unsigned Count) {
  
  llvm::FoldingSetNodeID profile;
  SymbolConjured::Profile(profile, E, Count);
  void* InsertPos;
  
  SymbolData* SD = DataSet.FindNodeOrInsertPos(profile, InsertPos);
  
  if (SD)
    return SD->getSymbol();
  
  SD = (SymbolData*) BPAlloc.Allocate<SymbolConjured>();
  new (SD) SymbolConjured(SymbolCounter, E, Count);
  
  DataSet.InsertNode(SD, InsertPos);  
  DataMap[SymbolCounter] = SD;
  
  return SymbolCounter++;
}

const SymbolData& SymbolManager::getSymbolData(SymbolID Sym) const {  
  DataMapTy::const_iterator I = DataMap.find(Sym);
  assert (I != DataMap.end());  
  return *I->second;
}


QualType SymbolData::getType(const SymbolManager& SymMgr) const {
  switch (getKind()) {
    default:
      assert (false && "getType() not implemented for this symbol.");
      
    case ParmKind:
      return cast<SymbolDataParmVar>(this)->getDecl()->getType();

    case GlobalKind:
      return cast<SymbolDataGlobalVar>(this)->getDecl()->getType();
      
    case ContentsOfKind: {
      SymbolID x = cast<SymbolDataContentsOf>(this)->getContainerSymbol();
      QualType T = SymMgr.getSymbolData(x).getType(SymMgr);
      return T->getAsPointerType()->getPointeeType();
    }
      
    case ConjuredKind:
      return cast<SymbolConjured>(this)->getExpr()->getType();
  }
}

SymbolManager::~SymbolManager() {}
