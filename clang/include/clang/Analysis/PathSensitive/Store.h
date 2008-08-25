//== Store.h - Interface for maps from Locations to Values ------*- C++ -*--==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defined the types Store and StoreManager.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_ANALYSIS_STORE_H
#define LLVM_CLANG_ANALYSIS_STORE_H

#include "clang/Analysis/PathSensitive/RValues.h"
#include "clang/Analysis/PathSensitive/Regions.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/DenseSet.h"
#include <vector>
#include <iosfwd>

namespace clang {
  
typedef const void* Store;
class GRStateManager;
class LiveVariables;
class Stmt;
  
class StoreManager {
public:
  typedef llvm::SmallSet<SymbolID, 20>      LiveSymbolsTy;
  typedef llvm::DenseSet<SymbolID>          DeadSymbolsTy;
  typedef std::vector<ValueDecl*>           DeclRootsTy;
  
  virtual ~StoreManager() {}
  virtual RVal GetRVal(Store St, LVal LV, QualType T = QualType()) = 0;
  virtual Store SetRVal(Store St, LVal LV, RVal V) = 0;
  virtual Store Remove(Store St, LVal LV) = 0;
  virtual Store getInitialStore(GRStateManager& StateMgr) = 0;
  
  virtual Store RemoveDeadBindings(Store store, Stmt* Loc,
                                   const LiveVariables& Live,
                                   DeclRootsTy& DRoots, LiveSymbolsTy& LSymbols,                                  
                                   DeadSymbolsTy& DSymbols) = 0;

  virtual Store AddDecl(Store store, GRStateManager& StMgr,
                        const VarDecl* VD, Expr* Ex, 
                        RVal InitVal = UndefinedVal(), unsigned Count = 0) = 0;

  virtual void print(Store store, std::ostream& Out,
                     const char* nl, const char *sep) = 0;
    
  /// getExtent - Returns the size of the region in bits.
  virtual RegionExtent getExtent(GRStateManager& SM, Region R) = 0;
};
  
} // end clang namespace

#endif
