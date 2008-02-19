//== SymbolManager.h - Management of Symbolic Values ------------*- C++ -*--==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This files defines SymbolManager, a class that manages symbolic values
//  created for use by GRExprEngine and related classes.
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/PathSensitive/SymbolManager.h"

using namespace clang;

SymbolID SymbolManager::getSymbol(ParmVarDecl* D) {
  SymbolID& X = DataToSymbol[getKey(D)];
  
  if (!X.isInitialized()) {
    X = SymbolToData.size();
    SymbolToData.push_back(SymbolDataParmVar(D));
  }
  
  return X;
}

SymbolID SymbolManager::getContentsOfSymbol(SymbolID sym) {
  SymbolID& X = DataToSymbol[getKey(sym)];
  
  if (!X.isInitialized()) {
    X = SymbolToData.size();
    SymbolToData.push_back(SymbolDataContentsOf(sym));
  }
  
  return X;  
}

QualType SymbolData::getType(const SymbolManager& SymMgr) const {
  switch (getKind()) {
    default:
      assert (false && "getType() not implemented for this symbol.");
      
    case ParmKind:
      return cast<SymbolDataParmVar>(this)->getDecl()->getType();
      
    case ContentsOfKind: {
      SymbolID x = cast<SymbolDataContentsOf>(this)->getSymbol();
      QualType T = SymMgr.getSymbolData(x).getType(SymMgr);
      return T->getAsPointerType()->getPointeeType();
    }
  }
}

SymbolManager::SymbolManager() {}
SymbolManager::~SymbolManager() {}
