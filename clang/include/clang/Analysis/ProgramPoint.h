//==- ProgramPoint.h - Program Points for Path-Sensitive Analysis --*- C++ -*-//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the interface ProgramPoint, which identifies a
//  distinct location in a function.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_ANALYSIS_PROGRAM_POINT
#define LLVM_CLANG_ANALYSIS_PROGRAM_POINT

#include "clang/AST/CFG.h"
#include "llvm/Support/DataTypes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/FoldingSet.h"
#include <cassert>

namespace clang {
    
class ProgramPoint {
public:
  enum Kind { BlockEdgeKind=0, BlockEntranceKind, BlockExitKind, 
              // Keep the following four together and in this order.
              PostStmtKind, PostLoadKind, PostStoreKind,
              PostPurgeDeadSymbolsKind };

private:
  std::pair<uintptr_t,uintptr_t> Data;
  
protected:
  ProgramPoint(const void* P, Kind k)
    : Data(reinterpret_cast<uintptr_t>(P), (uintptr_t) k) {}
    
  ProgramPoint(const void* P1, const void* P2)
    : Data(reinterpret_cast<uintptr_t>(P1) | 0x1,
           reinterpret_cast<uintptr_t>(P2)) {}
  
protected:
  void* getData1NoMask() const {
    assert (getKind() != BlockEdgeKind);
    return reinterpret_cast<void*>(Data.first);
  }
  
  void* getData1() const {
    assert (getKind() == BlockEdgeKind);
    return reinterpret_cast<void*>(Data.first & ~0x1);
  }

  void* getData2() const { 
    assert (getKind() == BlockEdgeKind);
    return reinterpret_cast<void*>(Data.second);
  }
  
public:    

  uintptr_t getKind() const {
    return Data.first & 0x1 ? (uintptr_t) BlockEdgeKind : Data.second;
  }

  // For use with DenseMap.
  unsigned getHashValue() const {
    std::pair<void*,void*> P(reinterpret_cast<void*>(Data.first),
                             reinterpret_cast<void*>(Data.second));
    return llvm::DenseMapInfo<std::pair<void*,void*> >::getHashValue(P);
  }
  
  static bool classof(const ProgramPoint*) { return true; }

  bool operator==(const ProgramPoint & RHS) const {
    return Data == RHS.Data;
  }

  bool operator!=(const ProgramPoint& RHS) const {
    return Data != RHS.Data;
  }
    
  bool operator<(const ProgramPoint& RHS) const {
    return Data < RHS.Data;
  }
  
  void Profile(llvm::FoldingSetNodeID& ID) const {
    ID.AddPointer(reinterpret_cast<void*>(Data.first));
    ID.AddPointer(reinterpret_cast<void*>(Data.second));
  }    
};
               
class BlockEntrance : public ProgramPoint {
public:
  BlockEntrance(const CFGBlock* B) : ProgramPoint(B, BlockEntranceKind) {}
    
  CFGBlock* getBlock() const {
    return reinterpret_cast<CFGBlock*>(getData1NoMask());
  }
  
  Stmt* getFirstStmt() const {
    CFGBlock* B = getBlock();
    return B->empty() ? NULL : B->front();
  }

  static bool classof(const ProgramPoint* Location) {
    return Location->getKind() == BlockEntranceKind;
  }
};

class BlockExit : public ProgramPoint {
public:
  BlockExit(const CFGBlock* B) : ProgramPoint(B, BlockExitKind) {}
  
  CFGBlock* getBlock() const {
    return reinterpret_cast<CFGBlock*>(getData1NoMask());
  }

  Stmt* getLastStmt() const {
    CFGBlock* B = getBlock();
    return B->empty() ? NULL : B->back();
  }
  
  Stmt* getTerminator() const {
    return getBlock()->getTerminator();
  }
    
  static bool classof(const ProgramPoint* Location) {
    return Location->getKind() == BlockExitKind;
  }
};


class PostStmt : public ProgramPoint {
protected:
  PostStmt(const Stmt* S, Kind k) : ProgramPoint(S, k) {}    
public:
  PostStmt(const Stmt* S) : ProgramPoint(S, PostStmtKind) {}
      
  Stmt* getStmt() const { return (Stmt*) getData1NoMask(); }

  static bool classof(const ProgramPoint* Location) {
    unsigned k = Location->getKind();
    return k >= PostStmtKind && k <= PostPurgeDeadSymbolsKind;
  }
};
  
class PostLoad : public PostStmt {
public:
  PostLoad(const Stmt* S) : PostStmt(S, PostLoadKind) {}
  
  static bool classof(const ProgramPoint* Location) {
    return Location->getKind() == PostLoadKind;
  }
};
  
class PostStore : public PostStmt {
public:
  PostStore(const Stmt* S) : PostStmt(S, PostLoadKind) {}
  
  static bool classof(const ProgramPoint* Location) {
    return Location->getKind() == PostStoreKind;
  }
};
  
class PostPurgeDeadSymbols : public PostStmt {
public:
  PostPurgeDeadSymbols(const Stmt* S) : PostStmt(S, PostPurgeDeadSymbolsKind) {}
  
  static bool classof(const ProgramPoint* Location) {
    return Location->getKind() == PostPurgeDeadSymbolsKind;
  }
};
  
class BlockEdge : public ProgramPoint {
public:
  BlockEdge(const CFGBlock* B1, const CFGBlock* B2)
    : ProgramPoint(B1, B2) {}
    
  CFGBlock* getSrc() const {
    return static_cast<CFGBlock*>(getData1());
  }
    
  CFGBlock* getDst() const {
    return static_cast<CFGBlock*>(getData2());
  }
  
  static bool classof(const ProgramPoint* Location) {
    return Location->getKind() == BlockEdgeKind;
  }
};

  
} // end namespace clang


namespace llvm { // Traits specialization for DenseMap 
  
template <> struct DenseMapInfo<clang::ProgramPoint> {

static inline clang::ProgramPoint getEmptyKey() {
  uintptr_t x =
   reinterpret_cast<uintptr_t>(DenseMapInfo<void*>::getEmptyKey()) & ~0x7;    
  return clang::BlockEntrance(reinterpret_cast<clang::CFGBlock*>(x));
}

static inline clang::ProgramPoint getTombstoneKey() {
  uintptr_t x =
   reinterpret_cast<uintptr_t>(DenseMapInfo<void*>::getTombstoneKey()) & ~0x7;    
  return clang::BlockEntrance(reinterpret_cast<clang::CFGBlock*>(x));
}

static unsigned getHashValue(const clang::ProgramPoint& Loc) {
  return Loc.getHashValue();
}

static bool isEqual(const clang::ProgramPoint& L,
                    const clang::ProgramPoint& R) {
  return L == R;
}

static bool isPod() {
  return true;
}
};
} // end namespace llvm

#endif
