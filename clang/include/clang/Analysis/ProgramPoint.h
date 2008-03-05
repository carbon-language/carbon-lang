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
  enum Kind { LayeredNodeKind  =  0x0,
              BlockEntranceKind = 0x1,
              PostStmtKind      = 0x2,
              BlockExitKind     = 0x3,
              BlockEdgeSrcKind  = 0x5, // Skip 0x4.
              BlockEdgeDstKind  = 0x6,
              BlockEdgeAuxKind  = 0x7 }; 
protected:
  uintptr_t Data;

  ProgramPoint(const void* Ptr, Kind k) {
    assert ((reinterpret_cast<uintptr_t>(const_cast<void*>(Ptr)) & 0x7) == 0
            && "Address must have at least an 8-byte alignment.");
    
    Data = reinterpret_cast<uintptr_t>(const_cast<void*>(Ptr)) | k;
  }
  
  ProgramPoint() : Data(0) {}
  
public:    
  
  unsigned getKind() const { 
    unsigned x = Data & 0x7;
    return x & 0x3 ? x : 0;  // Use only lower 2 bits for 0x0.
  }  
  
  void* getRawPtr() const {
    return (void*) (getKind() ? Data & ~0x7 : Data & ~0x3);
  }

  void* getRawData() const { return reinterpret_cast<void*>(Data); }
  
  static bool classof(const ProgramPoint*) { return true; }
  bool operator==(const ProgramPoint & RHS) const { return Data == RHS.Data; }
  bool operator!=(const ProgramPoint& RHS) const { return Data != RHS.Data; }
  
  void Profile(llvm::FoldingSetNodeID& ID) const {
    ID.AddInteger(getKind());
    ID.AddPointer(getRawPtr());
  }    
};
  
class ExplodedNodeImpl;
template <typename StateTy> class ExplodedNode;
  
class LayeredNode : public ProgramPoint {
public:
  LayeredNode(ExplodedNodeImpl* N) : ProgramPoint(N, LayeredNodeKind) {
    assert (reinterpret_cast<uintptr_t>(N) & 0x3 == 0 &&
            "Address of ExplodedNode must have 4-byte alignment.");
  }
  
  ExplodedNodeImpl* getNodeImpl() const {
    return (ExplodedNodeImpl*) getRawPtr();
  }

  template <typename StateTy>
  ExplodedNode<StateTy>* getNode() const {
    return (ExplodedNode<StateTy>*) getRawPtr();
  }

};
               
class BlockEntrance : public ProgramPoint {
public:
  BlockEntrance(const CFGBlock* B) : ProgramPoint(B, BlockEntranceKind) {}
    
  CFGBlock* getBlock() const {
    return reinterpret_cast<CFGBlock*>(getRawPtr());
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
    return reinterpret_cast<CFGBlock*>(getRawPtr());
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
public:
  PostStmt(const Stmt* S) : ProgramPoint(S, PostStmtKind) {}
  
  Stmt* getStmt() const { return (Stmt*) getRawPtr(); }

  static bool classof(const ProgramPoint* Location) {
    return Location->getKind() == PostStmtKind;
  }
};
  
class BlockEdge : public ProgramPoint {
  typedef std::pair<CFGBlock*,CFGBlock*> BPair;
public:
  BlockEdge(CFG& cfg, const CFGBlock* B1, const CFGBlock* B2);
  
  /// This ctor forces the BlockEdge to be constructed using an explicitly
  ///  allocated pair object that is stored in the CFG.  This is usually
  ///  used to construct edges representing jumps using computed gotos.
  BlockEdge(CFG& cfg, const CFGBlock* B1, const CFGBlock* B2, bool) {
    Data = reinterpret_cast<uintptr_t>(cfg.getBlockEdgeImpl(B1, B2))
           | BlockEdgeAuxKind;
  }


  CFGBlock* getSrc() const;
  CFGBlock* getDst() const;
  
  static bool classof(const ProgramPoint* Location) {
    unsigned k = Location->getKind();
    return k >= BlockEdgeSrcKind && k <= BlockEdgeAuxKind;
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
    return DenseMapInfo<void*>::getHashValue(Loc.getRawData());
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
