//==- ProgramEdge.h - Program Points for Path-Sensitive Analysis --*- C++ -*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Ted Kremenek and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the interface ProgramEdge, which identifies a distinct
//  location in a function based on edges within its CFG.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_ANALYSIS_PATHSENS_PROGRAM_POINT
#define LLVM_CLANG_ANALYSIS_PATHSENS_PROGRAM_POINT

#include "llvm/Support/DataTypes.h"
#include "llvm/ADT/DenseMap.h"
#include <cassert>

namespace clang {
  
  class CFGBlock;
  class Stmt;
  
class ProgramEdge {
  uintptr_t Src, Dst;
public:
  enum EdgeKind { StmtBlk=0, BlkStmt=1, StmtStmt=2, BlkBlk=3 };
  static bool classof(const ProgramEdge*) { return true; }
  
  unsigned getKind() const { return (unsigned) Src & 0x3; }
  void* RawSrc() const { return reinterpret_cast<void*>(Src & ~0x3); }
  void* RawDst() const { return reinterpret_cast<void*>(Dst); }

  bool operator==(const ProgramEdge & RHS) const {
    // comparing pointer values canoncalizes "NULL" edges where both pointers 
    // are NULL without having to worry about edgekind.  We can otherwise
    // ignore edgekind because no CFGBlock* or Stmt* will have the same value.
    return RawSrc() == RHS.RawSrc() && RawDst() == RHS.RawDst();    
  }
  
  bool operator!=(const ProgramEdge& RHS) const {
    return RawSrc() != RHS.RawSrc() || RawDst() != RHS.RawDst();
  }
  
  unsigned getHashValue() const {
    uintptr_t v1 = reinterpret_cast<uintptr_t>(RawSrc());
    uintptr_t v2 = reinterpret_cast<uintptr_t>(RawDst());
    return static_cast<unsigned>( (v1 >> 4) ^ (v1 >> 9) ^ 
                                  (v2 >> 5) ^ (v2 >> 10) );
  }
  
protected:
  
  ProgramEdge(const void* src, const void* dst, EdgeKind k) {
    assert (k >= StmtBlk && k <= BlkBlk);
    Src = reinterpret_cast<uintptr_t>(const_cast<void*>(src)) | k;
    Dst = reinterpret_cast<uintptr_t>(const_cast<void*>(dst));
  }
};

class StmtBlkEdge : public ProgramEdge {
public:
  StmtBlkEdge(const Stmt* S,const CFGBlock* B)
  : ProgramEdge(S,B,StmtBlk) {}
  
  const Stmt*     Src() const { return reinterpret_cast<Stmt*>(RawSrc()); }
  const CFGBlock* Dst() const { return reinterpret_cast<CFGBlock*>(RawDst()); }  
  
  static bool classof(const ProgramEdge* E) { return E->getKind() == StmtBlk; }
};

class BlkStmtEdge : public ProgramEdge {
public:
  BlkStmtEdge(const CFGBlock* B, const Stmt* S)
  : ProgramEdge(B,S,BlkStmt) {}
  
  const CFGBlock* Src() const { return reinterpret_cast<CFGBlock*>(RawSrc()); }  
  const Stmt*     Dst() const { return reinterpret_cast<Stmt*>(RawDst()); }

  static bool classof(const ProgramEdge* E) { return E->getKind() == StmtBlk; }
};
  
class StmtStmtEdge : public ProgramEdge {
public:
  StmtStmtEdge(const Stmt* S1, const Stmt* S2)
  : ProgramEdge(S1,S2,StmtStmt) {}
  
  const Stmt*     Src() const { return reinterpret_cast<Stmt*>(RawSrc()); }  
  const Stmt*     Dst() const { return reinterpret_cast<Stmt*>(RawDst()); }
  
  static bool classof(const ProgramEdge* E) { return E->getKind() == StmtStmt; }
};


class BlkBlkEdge : public ProgramEdge {
public:
  BlkBlkEdge(const CFGBlock* B1, const CFGBlock* B2)
  : ProgramEdge(B1,B2,BlkBlk) {}
  
  const CFGBlock* Src() const { return reinterpret_cast<CFGBlock*>(RawSrc()); }  
  const CFGBlock* Dst() const { return reinterpret_cast<CFGBlock*>(RawDst()); }
  
  static bool classof(const ProgramEdge* E) { return E->getKind() == BlkBlk; }
};

} // end namespace clang


namespace llvm { // Traits specialization for DenseMap 
  
template <> struct DenseMapInfo<clang::ProgramEdge> {

  static inline clang::ProgramEdge getEmptyKey() {
    return clang::BlkBlkEdge(NULL,NULL);
  }
  
  static inline clang::ProgramEdge getTombstoneKey() {
    return clang::BlkBlkEdge(reinterpret_cast<clang::CFGBlock*>(-1),
                             reinterpret_cast<clang::CFGBlock*>(-1));
  }
  
  static unsigned getHashValue(const clang::ProgramEdge& E) {
    return E.getHashValue();
  }
  
  static bool isEqual(const clang::ProgramEdge& LHS,
                       const clang::ProgramEdge& RHS) {
    return LHS == RHS;
  }
  
  static bool isPod() { return true; }
};
} // end namespace llvm

#endif
