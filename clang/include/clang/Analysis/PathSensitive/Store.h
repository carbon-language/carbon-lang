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
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/DenseSet.h"
#include <vector>
#include <iosfwd>

namespace clang {
  
typedef const void* Store;
  
namespace store {
  typedef const void* Binding;
  typedef const void* Region;
  
  class RegionExtent {
  public:
    enum Kind { Unknown = 0, Int = 0, Sym = 1 };
    
  protected:
    const uintptr_t Raw;
    RegionExtent(uintptr_t raw, Kind k) : Raw(raw | k) {}
    uintptr_t getData() const { return Raw & ~0x1; }
    
  public:
    // Folding-set profiling.
    void Profile(llvm::FoldingSetNodeID& ID) const {
      ID.AddPointer((void*) Raw);
    }  
    // Comparing extents.
    bool operator==(const RegionExtent& R) const {
      return Raw == R.Raw;
    }
    bool operator!=(const RegionExtent& R) const {
      return Raw != R.Raw;
    }  
    // Implement isa<T> support.
    Kind getKind() const { return Kind(Raw & 0x1); }
    uintptr_t getRaw() const { return Raw; }
    
    static inline bool classof(const RegionExtent*) {
      return true;
    }
  };
  
  class UnknownExtent : public RegionExtent {
  public:
    UnknownExtent() : RegionExtent(0,Unknown) {}
    
    // Implement isa<T> support.
    static inline bool classof(const RegionExtent* E) {
      return E->getRaw() == 0;
    }  
  };
  
  class IntExtent : public RegionExtent {
  public:
    IntExtent(const llvm::APSInt& X) : RegionExtent((uintptr_t) &X, Int) {}
    
    const llvm::APSInt& getInt() const {
      return *((llvm::APSInt*) getData());
    }
    
    // Implement isa<T> support.
    static inline bool classof(const RegionExtent* E) {
      return E->getKind() == Int && E->getRaw() != 0;
    }
  };
  
  class SymExtent : public RegionExtent {
  public:
    SymExtent(SymbolID S) : RegionExtent(S.getNumber() << 1, Sym) {}
    
    SymbolID getSymbol() const { return SymbolID(getData() >> 1); }
    
    // Implement isa<T> support.
    static inline bool classof(const RegionExtent* E) {
      return E->getKind() == Sym;
    }  
  };
} // end store namespace
  
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
  virtual store::RegionExtent getExtent(store::Region R) =0;
};
  
} // end clang namespace

#endif
