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

#include "clang/Analysis/AnalysisContext.h"
#include "clang/Analysis/CFG.h"
#include "llvm/Support/DataTypes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/FoldingSet.h"
#include "llvm/Support/Casting.h"
#include <cassert>
#include <utility>

namespace clang {

class LocationContext;
class AnalysisContext;
class FunctionDecl;

class ProgramPoint {
public:
  enum Kind { BlockEdgeKind,
              BlockEntranceKind,
              BlockExitKind,
              PreStmtKind,
              PostStmtKind,
              PreLoadKind,
              PostLoadKind,
              PreStoreKind,
              PostStoreKind,
              PostPurgeDeadSymbolsKind,
              PostStmtCustomKind,
              PostLValueKind,
              PostInitializerKind,
              CallEnterKind,
              CallExitKind,
              MinPostStmtKind = PostStmtKind,
              MaxPostStmtKind = CallExitKind };

private:
  std::pair<const void *, const void *> Data;
  Kind K;

  // The LocationContext could be NULL to allow ProgramPoint to be used in
  // context insensitive analysis.
  const LocationContext *L;
  const void *Tag;

protected:
  ProgramPoint(const void* P, Kind k, const LocationContext *l,
               const void *tag = 0)
    : Data(P, static_cast<const void*>(NULL)), K(k), L(l), Tag(tag) {}

  ProgramPoint(const void* P1, const void* P2, Kind k, const LocationContext *l,
               const void *tag = 0)
    : Data(P1, P2), K(k), L(l), Tag(tag) {}

protected:
  const void* getData1() const { return Data.first; }
  const void* getData2() const { return Data.second; }
  const void *getTag() const { return Tag; }

public:
  Kind getKind() const { return K; }

  const LocationContext *getLocationContext() const { return L; }

  // For use with DenseMap.  This hash is probably slow.
  unsigned getHashValue() const {
    llvm::FoldingSetNodeID ID;
    Profile(ID);
    return ID.ComputeHash();
  }

  static bool classof(const ProgramPoint*) { return true; }

  bool operator==(const ProgramPoint & RHS) const {
    return K == RHS.K && Data == RHS.Data && L == RHS.L && Tag == RHS.Tag;
  }

  bool operator!=(const ProgramPoint& RHS) const {
    return K != RHS.K || Data != RHS.Data || L != RHS.L || Tag != RHS.Tag;
  }

  void Profile(llvm::FoldingSetNodeID& ID) const {
    ID.AddInteger((unsigned) K);
    ID.AddPointer(Data.first);
    ID.AddPointer(Data.second);
    ID.AddPointer(L);
    ID.AddPointer(Tag);
  }
};

class BlockEntrance : public ProgramPoint {
public:
  BlockEntrance(const CFGBlock* B, const LocationContext *L,
                const void *tag = 0)
    : ProgramPoint(B, BlockEntranceKind, L, tag) {}

  const CFGBlock* getBlock() const {
    return reinterpret_cast<const CFGBlock*>(getData1());
  }

  const CFGElement getFirstElement() const {
    const CFGBlock* B = getBlock();
    return B->empty() ? CFGElement() : B->front();
  }
  
  /// Create a new BlockEntrance object that is the same as the original
  /// except for using the specified tag value.
  BlockEntrance withTag(const void *tag) {
    return BlockEntrance(getBlock(), getLocationContext(), tag);
  }
  
  static bool classof(const ProgramPoint* Location) {
    return Location->getKind() == BlockEntranceKind;
  }
};

class BlockExit : public ProgramPoint {
public:
  BlockExit(const CFGBlock* B, const LocationContext *L)
    : ProgramPoint(B, BlockExitKind, L) {}

  const CFGBlock* getBlock() const {
    return reinterpret_cast<const CFGBlock*>(getData1());
  }

  const Stmt* getTerminator() const {
    return getBlock()->getTerminator();
  }

  static bool classof(const ProgramPoint* Location) {
    return Location->getKind() == BlockExitKind;
  }
};

class StmtPoint : public ProgramPoint {
public:
  StmtPoint(const Stmt *S, const void *p2, Kind k, const LocationContext *L,
            const void *tag)
    : ProgramPoint(S, p2, k, L, tag) {}

  const Stmt *getStmt() const { return (const Stmt*) getData1(); }

  template <typename T>
  const T* getStmtAs() const { return llvm::dyn_cast<T>(getStmt()); }

  static bool classof(const ProgramPoint* Location) {
    unsigned k = Location->getKind();
    return k >= PreStmtKind && k <= MaxPostStmtKind;
  }
};


class PreStmt : public StmtPoint {
public:
  PreStmt(const Stmt *S, const LocationContext *L, const void *tag,
          const Stmt *SubStmt = 0)
    : StmtPoint(S, SubStmt, PreStmtKind, L, tag) {}

  const Stmt *getSubStmt() const { return (const Stmt*) getData2(); }

  static bool classof(const ProgramPoint* Location) {
    return Location->getKind() == PreStmtKind;
  }
};

class PostStmt : public StmtPoint {
protected:
  PostStmt(const Stmt* S, const void* data, Kind k, const LocationContext *L,
           const void *tag =0)
    : StmtPoint(S, data, k, L, tag) {}

public:
  explicit PostStmt(const Stmt* S, Kind k, 
                    const LocationContext *L, const void *tag = 0)
    : StmtPoint(S, NULL, k, L, tag) {}

  explicit PostStmt(const Stmt* S, const LocationContext *L,const void *tag = 0)
    : StmtPoint(S, NULL, PostStmtKind, L, tag) {}

  static bool classof(const ProgramPoint* Location) {
    unsigned k = Location->getKind();
    return k >= MinPostStmtKind && k <= MaxPostStmtKind;
  }
};

class PostStmtCustom : public PostStmt {
public:
  PostStmtCustom(const Stmt* S,
                 const std::pair<const void*, const void*>* TaggedData,\
                 const LocationContext *L)
    : PostStmt(S, TaggedData, PostStmtCustomKind, L) {}

  const std::pair<const void*, const void*>& getTaggedPair() const {
    return
      *reinterpret_cast<const std::pair<const void*, const void*>*>(getData2());
  }

  const void* getTag() const { return getTaggedPair().first; }

  const void* getTaggedData() const { return getTaggedPair().second; }

  static bool classof(const ProgramPoint* Location) {
    return Location->getKind() == PostStmtCustomKind;
  }
};

  
class LocationCheck : public StmtPoint {
protected:
  LocationCheck(const Stmt *S, const LocationContext *L,
                ProgramPoint::Kind K, const void *tag)
    : StmtPoint(S, NULL, K, L, tag) {}
    
  static bool classof(const ProgramPoint *location) {
    unsigned k = location->getKind();
    return k == PreLoadKind || k == PreStoreKind;
  }
};
  
class PreLoad : public LocationCheck {
public:
  PreLoad(const Stmt *S, const LocationContext *L, const void *tag = 0)
    : LocationCheck(S, L, PreLoadKind, tag) {}
  
  static bool classof(const ProgramPoint *location) {
    return location->getKind() == PreLoadKind;
  }
};

class PreStore : public LocationCheck {
public:
  PreStore(const Stmt *S, const LocationContext *L, const void *tag = 0)
  : LocationCheck(S, L, PreStoreKind, tag) {}
  
  static bool classof(const ProgramPoint *location) {
    return location->getKind() == PreStoreKind;
  }
};

class PostLoad : public PostStmt {
public:
  PostLoad(const Stmt* S, const LocationContext *L, const void *tag = 0)
    : PostStmt(S, PostLoadKind, L, tag) {}

  static bool classof(const ProgramPoint* Location) {
    return Location->getKind() == PostLoadKind;
  }
};

class PostStore : public PostStmt {
public:
  PostStore(const Stmt* S, const LocationContext *L, const void *tag = 0)
    : PostStmt(S, PostStoreKind, L, tag) {}

  static bool classof(const ProgramPoint* Location) {
    return Location->getKind() == PostStoreKind;
  }
};

class PostLValue : public PostStmt {
public:
  PostLValue(const Stmt* S, const LocationContext *L, const void *tag = 0)
    : PostStmt(S, PostLValueKind, L, tag) {}

  static bool classof(const ProgramPoint* Location) {
    return Location->getKind() == PostLValueKind;
  }
};

class PostPurgeDeadSymbols : public PostStmt {
public:
  PostPurgeDeadSymbols(const Stmt* S, const LocationContext *L,
                       const void *tag = 0)
    : PostStmt(S, PostPurgeDeadSymbolsKind, L, tag) {}

  static bool classof(const ProgramPoint* Location) {
    return Location->getKind() == PostPurgeDeadSymbolsKind;
  }
};

class BlockEdge : public ProgramPoint {
public:
  BlockEdge(const CFGBlock* B1, const CFGBlock* B2, const LocationContext *L)
    : ProgramPoint(B1, B2, BlockEdgeKind, L) {}

  const CFGBlock* getSrc() const {
    return static_cast<const CFGBlock*>(getData1());
  }

  const CFGBlock* getDst() const {
    return static_cast<const CFGBlock*>(getData2());
  }

  static bool classof(const ProgramPoint* Location) {
    return Location->getKind() == BlockEdgeKind;
  }
};

class PostInitializer : public ProgramPoint {
public:
  PostInitializer(const CXXCtorInitializer *I, 
                  const LocationContext *L)
    : ProgramPoint(I, PostInitializerKind, L) {}

  static bool classof(const ProgramPoint *Location) {
    return Location->getKind() == PostInitializerKind;
  }
};

class CallEnter : public StmtPoint {
public:
  CallEnter(const Stmt *stmt, const StackFrameContext *calleeCtx, 
            const LocationContext *callerCtx)
    : StmtPoint(stmt, calleeCtx, CallEnterKind, callerCtx, 0) {}

  const Stmt *getCallExpr() const {
    return static_cast<const Stmt *>(getData1());
  }

  const StackFrameContext *getCalleeContext() const {
    return static_cast<const StackFrameContext *>(getData2());
  }

  static bool classof(const ProgramPoint *Location) {
    return Location->getKind() == CallEnterKind;
  }
};

class CallExit : public StmtPoint {
public:
  // CallExit uses the callee's location context.
  CallExit(const Stmt *S, const LocationContext *L)
    : StmtPoint(S, 0, CallExitKind, L, 0) {}

  static bool classof(const ProgramPoint *Location) {
    return Location->getKind() == CallExitKind;
  }
};


} // end namespace clang


namespace llvm { // Traits specialization for DenseMap

template <> struct DenseMapInfo<clang::ProgramPoint> {

static inline clang::ProgramPoint getEmptyKey() {
  uintptr_t x =
   reinterpret_cast<uintptr_t>(DenseMapInfo<void*>::getEmptyKey()) & ~0x7;
  return clang::BlockEntrance(reinterpret_cast<clang::CFGBlock*>(x), 0);
}

static inline clang::ProgramPoint getTombstoneKey() {
  uintptr_t x =
   reinterpret_cast<uintptr_t>(DenseMapInfo<void*>::getTombstoneKey()) & ~0x7;
  return clang::BlockEntrance(reinterpret_cast<clang::CFGBlock*>(x), 0);
}

static unsigned getHashValue(const clang::ProgramPoint& Loc) {
  return Loc.getHashValue();
}

static bool isEqual(const clang::ProgramPoint& L,
                    const clang::ProgramPoint& R) {
  return L == R;
}

};
  
template <>
struct isPodLike<clang::ProgramPoint> { static const bool value = true; };

} // end namespace llvm

#endif
