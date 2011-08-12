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
//  created for use by ExprEngine and related classes.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_GR_SYMMGR_H
#define LLVM_CLANG_GR_SYMMGR_H

#include "clang/AST/Decl.h"
#include "clang/AST/Expr.h"
#include "clang/Analysis/AnalysisContext.h"
#include "clang/Basic/LLVM.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/StoreRef.h"
#include "llvm/Support/DataTypes.h"
#include "llvm/ADT/FoldingSet.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/DenseMap.h"

namespace llvm {
class BumpPtrAllocator;
}

namespace clang {
  class ASTContext;
  class StackFrameContext;

namespace ento {
  class BasicValueFactory;
  class MemRegion;
  class SubRegion;
  class TypedRegion;
  class VarRegion;

class SymExpr : public llvm::FoldingSetNode {
public:
  enum Kind { RegionValueKind, ConjuredKind, DerivedKind, ExtentKind,
              MetadataKind,
              BEGIN_SYMBOLS = RegionValueKind,
              END_SYMBOLS = MetadataKind,
              SymIntKind, SymSymKind };
private:
  Kind K;

protected:
  SymExpr(Kind k) : K(k) {}

public:
  virtual ~SymExpr() {}

  Kind getKind() const { return K; }

  void dump() const;

  virtual void dumpToStream(raw_ostream &os) const = 0;

  virtual QualType getType(ASTContext&) const = 0;
  virtual void Profile(llvm::FoldingSetNodeID& profile) = 0;

  // Implement isa<T> support.
  static inline bool classof(const SymExpr*) { return true; }
};

typedef unsigned SymbolID;

class SymbolData : public SymExpr {
private:
  const SymbolID Sym;

protected:
  SymbolData(Kind k, SymbolID sym) : SymExpr(k), Sym(sym) {}

public:
  virtual ~SymbolData() {}

  SymbolID getSymbolID() const { return Sym; }

  // Implement isa<T> support.
  static inline bool classof(const SymExpr* SE) {
    Kind k = SE->getKind();
    return k >= BEGIN_SYMBOLS && k <= END_SYMBOLS;
  }
};

typedef const SymbolData* SymbolRef;
typedef llvm::SmallVector<SymbolRef, 2> SymbolRefSmallVectorTy;

/// A symbol representing the value of a MemRegion.
class SymbolRegionValue : public SymbolData {
  const TypedRegion *R;

public:
  SymbolRegionValue(SymbolID sym, const TypedRegion *r)
    : SymbolData(RegionValueKind, sym), R(r) {}

  const TypedRegion* getRegion() const { return R; }

  static void Profile(llvm::FoldingSetNodeID& profile, const TypedRegion* R) {
    profile.AddInteger((unsigned) RegionValueKind);
    profile.AddPointer(R);
  }

  virtual void Profile(llvm::FoldingSetNodeID& profile) {
    Profile(profile, R);
  }

  void dumpToStream(raw_ostream &os) const;

  QualType getType(ASTContext&) const;

  // Implement isa<T> support.
  static inline bool classof(const SymExpr* SE) {
    return SE->getKind() == RegionValueKind;
  }
};

/// A symbol representing the result of an expression.
class SymbolConjured : public SymbolData {
  const Stmt* S;
  QualType T;
  unsigned Count;
  const void* SymbolTag;

public:
  SymbolConjured(SymbolID sym, const Stmt* s, QualType t, unsigned count,
                 const void* symbolTag)
    : SymbolData(ConjuredKind, sym), S(s), T(t), Count(count),
      SymbolTag(symbolTag) {}

  const Stmt* getStmt() const { return S; }
  unsigned getCount() const { return Count; }
  const void* getTag() const { return SymbolTag; }

  QualType getType(ASTContext&) const;

  void dumpToStream(raw_ostream &os) const;

  static void Profile(llvm::FoldingSetNodeID& profile, const Stmt* S,
                      QualType T, unsigned Count, const void* SymbolTag) {
    profile.AddInteger((unsigned) ConjuredKind);
    profile.AddPointer(S);
    profile.Add(T);
    profile.AddInteger(Count);
    profile.AddPointer(SymbolTag);
  }

  virtual void Profile(llvm::FoldingSetNodeID& profile) {
    Profile(profile, S, T, Count, SymbolTag);
  }

  // Implement isa<T> support.
  static inline bool classof(const SymExpr* SE) {
    return SE->getKind() == ConjuredKind;
  }
};

/// A symbol representing the value of a MemRegion whose parent region has
/// symbolic value.
class SymbolDerived : public SymbolData {
  SymbolRef parentSymbol;
  const TypedRegion *R;

public:
  SymbolDerived(SymbolID sym, SymbolRef parent, const TypedRegion *r)
    : SymbolData(DerivedKind, sym), parentSymbol(parent), R(r) {}

  SymbolRef getParentSymbol() const { return parentSymbol; }
  const TypedRegion *getRegion() const { return R; }

  QualType getType(ASTContext&) const;

  void dumpToStream(raw_ostream &os) const;

  static void Profile(llvm::FoldingSetNodeID& profile, SymbolRef parent,
                      const TypedRegion *r) {
    profile.AddInteger((unsigned) DerivedKind);
    profile.AddPointer(r);
    profile.AddPointer(parent);
  }

  virtual void Profile(llvm::FoldingSetNodeID& profile) {
    Profile(profile, parentSymbol, R);
  }

  // Implement isa<T> support.
  static inline bool classof(const SymExpr* SE) {
    return SE->getKind() == DerivedKind;
  }
};

/// SymbolExtent - Represents the extent (size in bytes) of a bounded region.
///  Clients should not ask the SymbolManager for a region's extent. Always use
///  SubRegion::getExtent instead -- the value returned may not be a symbol.
class SymbolExtent : public SymbolData {
  const SubRegion *R;
  
public:
  SymbolExtent(SymbolID sym, const SubRegion *r)
  : SymbolData(ExtentKind, sym), R(r) {}

  const SubRegion *getRegion() const { return R; }

  QualType getType(ASTContext&) const;

  void dumpToStream(raw_ostream &os) const;

  static void Profile(llvm::FoldingSetNodeID& profile, const SubRegion *R) {
    profile.AddInteger((unsigned) ExtentKind);
    profile.AddPointer(R);
  }

  virtual void Profile(llvm::FoldingSetNodeID& profile) {
    Profile(profile, R);
  }

  // Implement isa<T> support.
  static inline bool classof(const SymExpr* SE) {
    return SE->getKind() == ExtentKind;
  }
};

/// SymbolMetadata - Represents path-dependent metadata about a specific region.
///  Metadata symbols remain live as long as they are marked as in use before
///  dead-symbol sweeping AND their associated regions are still alive.
///  Intended for use by checkers.
class SymbolMetadata : public SymbolData {
  const MemRegion* R;
  const Stmt* S;
  QualType T;
  unsigned Count;
  const void* Tag;
public:
  SymbolMetadata(SymbolID sym, const MemRegion* r, const Stmt* s, QualType t,
                 unsigned count, const void* tag)
  : SymbolData(MetadataKind, sym), R(r), S(s), T(t), Count(count), Tag(tag) {}

  const MemRegion *getRegion() const { return R; }
  const Stmt* getStmt() const { return S; }
  unsigned getCount() const { return Count; }
  const void* getTag() const { return Tag; }

  QualType getType(ASTContext&) const;

  void dumpToStream(raw_ostream &os) const;

  static void Profile(llvm::FoldingSetNodeID& profile, const MemRegion *R,
                      const Stmt *S, QualType T, unsigned Count,
                      const void *Tag) {
    profile.AddInteger((unsigned) MetadataKind);
    profile.AddPointer(R);
    profile.AddPointer(S);
    profile.Add(T);
    profile.AddInteger(Count);
    profile.AddPointer(Tag);
  }

  virtual void Profile(llvm::FoldingSetNodeID& profile) {
    Profile(profile, R, S, T, Count, Tag);
  }

  // Implement isa<T> support.
  static inline bool classof(const SymExpr* SE) {
    return SE->getKind() == MetadataKind;
  }
};

/// SymIntExpr - Represents symbolic expression like 'x' + 3.
class SymIntExpr : public SymExpr {
  const SymExpr *LHS;
  BinaryOperator::Opcode Op;
  const llvm::APSInt& RHS;
  QualType T;

public:
  SymIntExpr(const SymExpr *lhs, BinaryOperator::Opcode op,
             const llvm::APSInt& rhs, QualType t)
    : SymExpr(SymIntKind), LHS(lhs), Op(op), RHS(rhs), T(t) {}

  // FIXME: We probably need to make this out-of-line to avoid redundant
  // generation of virtual functions.
  QualType getType(ASTContext& C) const { return T; }

  BinaryOperator::Opcode getOpcode() const { return Op; }

  void dumpToStream(raw_ostream &os) const;

  const SymExpr *getLHS() const { return LHS; }
  const llvm::APSInt &getRHS() const { return RHS; }

  static void Profile(llvm::FoldingSetNodeID& ID, const SymExpr *lhs,
                      BinaryOperator::Opcode op, const llvm::APSInt& rhs,
                      QualType t) {
    ID.AddInteger((unsigned) SymIntKind);
    ID.AddPointer(lhs);
    ID.AddInteger(op);
    ID.AddPointer(&rhs);
    ID.Add(t);
  }

  void Profile(llvm::FoldingSetNodeID& ID) {
    Profile(ID, LHS, Op, RHS, T);
  }

  // Implement isa<T> support.
  static inline bool classof(const SymExpr* SE) {
    return SE->getKind() == SymIntKind;
  }
};

/// SymSymExpr - Represents symbolic expression like 'x' + 'y'.
class SymSymExpr : public SymExpr {
  const SymExpr *LHS;
  BinaryOperator::Opcode Op;
  const SymExpr *RHS;
  QualType T;

public:
  SymSymExpr(const SymExpr *lhs, BinaryOperator::Opcode op, const SymExpr *rhs,
             QualType t)
    : SymExpr(SymSymKind), LHS(lhs), Op(op), RHS(rhs), T(t) {}

  BinaryOperator::Opcode getOpcode() const { return Op; }
  const SymExpr *getLHS() const { return LHS; }
  const SymExpr *getRHS() const { return RHS; }

  // FIXME: We probably need to make this out-of-line to avoid redundant
  // generation of virtual functions.
  QualType getType(ASTContext& C) const { return T; }

  void dumpToStream(raw_ostream &os) const;

  static void Profile(llvm::FoldingSetNodeID& ID, const SymExpr *lhs,
                    BinaryOperator::Opcode op, const SymExpr *rhs, QualType t) {
    ID.AddInteger((unsigned) SymSymKind);
    ID.AddPointer(lhs);
    ID.AddInteger(op);
    ID.AddPointer(rhs);
    ID.Add(t);
  }

  void Profile(llvm::FoldingSetNodeID& ID) {
    Profile(ID, LHS, Op, RHS, T);
  }

  // Implement isa<T> support.
  static inline bool classof(const SymExpr* SE) {
    return SE->getKind() == SymSymKind;
  }
};

class SymbolManager {
  typedef llvm::FoldingSet<SymExpr> DataSetTy;
  typedef llvm::DenseMap<SymbolRef, SymbolRefSmallVectorTy*> SymbolDependTy;

  DataSetTy DataSet;
  /// Stores the extra dependencies between symbols: the data should be kept
  /// alive as long as the key is live.
  SymbolDependTy SymbolDependencies;
  unsigned SymbolCounter;
  llvm::BumpPtrAllocator& BPAlloc;
  BasicValueFactory &BV;
  ASTContext& Ctx;

public:
  SymbolManager(ASTContext& ctx, BasicValueFactory &bv,
                llvm::BumpPtrAllocator& bpalloc)
    : SymbolDependencies(16), SymbolCounter(0),
      BPAlloc(bpalloc), BV(bv), Ctx(ctx) {}

  ~SymbolManager();

  static bool canSymbolicate(QualType T);

  /// \brief Make a unique symbol for MemRegion R according to its kind.
  const SymbolRegionValue* getRegionValueSymbol(const TypedRegion* R);

  const SymbolConjured* getConjuredSymbol(const Stmt* E, QualType T,
                                          unsigned VisitCount,
                                          const void* SymbolTag = 0);

  const SymbolConjured* getConjuredSymbol(const Expr* E, unsigned VisitCount,
                                          const void* SymbolTag = 0) {
    return getConjuredSymbol(E, E->getType(), VisitCount, SymbolTag);
  }

  const SymbolDerived *getDerivedSymbol(SymbolRef parentSymbol,
                                        const TypedRegion *R);

  const SymbolExtent *getExtentSymbol(const SubRegion *R);

  /// \brief Creates a metadata symbol associated with a specific region.
  ///
  /// VisitCount can be used to differentiate regions corresponding to
  /// different loop iterations, thus, making the symbol path-dependent.
  const SymbolMetadata* getMetadataSymbol(const MemRegion* R, const Stmt* S,
                                          QualType T, unsigned VisitCount,
                                          const void* SymbolTag = 0);

  const SymIntExpr *getSymIntExpr(const SymExpr *lhs, BinaryOperator::Opcode op,
                                  const llvm::APSInt& rhs, QualType t);

  const SymIntExpr *getSymIntExpr(const SymExpr &lhs, BinaryOperator::Opcode op,
                                  const llvm::APSInt& rhs, QualType t) {
    return getSymIntExpr(&lhs, op, rhs, t);
  }

  const SymSymExpr *getSymSymExpr(const SymExpr *lhs, BinaryOperator::Opcode op,
                                  const SymExpr *rhs, QualType t);

  QualType getType(const SymExpr *SE) const {
    return SE->getType(Ctx);
  }

  /// \brief Add artificial symbol dependency.
  ///
  /// The dependent symbol should stay alive as long as the primary is alive.
  void addSymbolDependency(const SymbolRef Primary, const SymbolRef Dependent);

  const SymbolRefSmallVectorTy *getDependentSymbols(const SymbolRef Primary);

  ASTContext &getContext() { return Ctx; }
  BasicValueFactory &getBasicVals() { return BV; }
};

class SymbolReaper {
  enum SymbolStatus {
    NotProcessed,
    HaveMarkedDependents
  };

  typedef llvm::DenseSet<SymbolRef> SymbolSetTy;
  typedef llvm::DenseMap<SymbolRef, SymbolStatus> SymbolMapTy;
  typedef llvm::DenseSet<const MemRegion *> RegionSetTy;

  SymbolMapTy TheLiving;
  SymbolSetTy MetadataInUse;
  SymbolSetTy TheDead;

  RegionSetTy RegionRoots;
  
  const LocationContext *LCtx;
  const Stmt *Loc;
  SymbolManager& SymMgr;
  StoreRef reapedStore;
  llvm::DenseMap<const MemRegion *, unsigned> includedRegionCache;

public:
  SymbolReaper(const LocationContext *ctx, const Stmt *s, SymbolManager& symmgr,
               StoreManager &storeMgr)
   : LCtx(ctx), Loc(s), SymMgr(symmgr), reapedStore(0, storeMgr) {}

  ~SymbolReaper() {}

  const LocationContext *getLocationContext() const { return LCtx; }
  const Stmt *getCurrentStatement() const { return Loc; }

  bool isLive(SymbolRef sym);
  bool isLiveRegion(const MemRegion *region);
  bool isLive(const Stmt *ExprVal) const;
  bool isLive(const VarRegion *VR, bool includeStoreBindings = false) const;

  /// \brief Unconditionally marks a symbol as live.
  ///
  /// This should never be
  /// used by checkers, only by the state infrastructure such as the store and
  /// environment. Checkers should instead use metadata symbols and markInUse.
  void markLive(SymbolRef sym);

  /// \brief Marks a symbol as important to a checker.
  ///
  /// For metadata symbols,
  /// this will keep the symbol alive as long as its associated region is also
  /// live. For other symbols, this has no effect; checkers are not permitted
  /// to influence the life of other symbols. This should be used before any
  /// symbol marking has occurred, i.e. in the MarkLiveSymbols callback.
  void markInUse(SymbolRef sym);

  /// \brief If a symbol is known to be live, marks the symbol as live.
  ///
  ///  Otherwise, if the symbol cannot be proven live, it is marked as dead.
  ///  Returns true if the symbol is dead, false if live.
  bool maybeDead(SymbolRef sym);

  typedef SymbolSetTy::const_iterator dead_iterator;
  dead_iterator dead_begin() const { return TheDead.begin(); }
  dead_iterator dead_end() const { return TheDead.end(); }

  bool hasDeadSymbols() const {
    return !TheDead.empty();
  }
  
  typedef RegionSetTy::const_iterator region_iterator;
  region_iterator region_begin() const { return RegionRoots.begin(); }
  region_iterator region_end() const { return RegionRoots.end(); }

  /// \brief Returns whether or not a symbol has been confirmed dead.
  ///
  /// This should only be called once all marking of dead symbols has completed.
  /// (For checkers, this means only in the evalDeadSymbols callback.)
  bool isDead(SymbolRef sym) const {
    return TheDead.count(sym);
  }
  
  void markLive(const MemRegion *region);
  
  /// \brief Set to the value of the symbolic store after
  /// StoreManager::removeDeadBindings has been called.
  void setReapedStore(StoreRef st) { reapedStore = st; }

private:
  /// Mark the symbols dependent on the input symbol as live.
  void markDependentsLive(SymbolRef sym);
};

class SymbolVisitor {
public:
  /// \brief A visitor method invoked by GRStateManager::scanReachableSymbols.
  ///
  /// The method returns \c true if symbols should continue be scanned and \c
  /// false otherwise.
  virtual bool VisitSymbol(SymbolRef sym) = 0;
  virtual bool VisitMemRegion(const MemRegion *region) { return true; };
  virtual ~SymbolVisitor();
};

} // end GR namespace

} // end clang namespace

namespace llvm {
static inline raw_ostream& operator<<(raw_ostream& os,
                                            const clang::ento::SymExpr *SE) {
  SE->dumpToStream(os);
  return os;
}
} // end llvm namespace
#endif
