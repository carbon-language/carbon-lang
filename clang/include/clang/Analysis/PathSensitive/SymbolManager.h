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

#ifndef LLVM_CLANG_ANALYSIS_SYMMGR_H
#define LLVM_CLANG_ANALYSIS_SYMMGR_H

#include "clang/AST/Decl.h"
#include "clang/AST/Expr.h"
#include "clang/Analysis/Analyses/LiveVariables.h"
#include "llvm/Support/DataTypes.h"
#include "llvm/Support/Allocator.h"
#include "llvm/ADT/FoldingSet.h"
#include "llvm/ADT/DenseSet.h"

namespace llvm {
  class raw_ostream;
}

namespace clang {
  class MemRegion;
  class TypedRegion;
  class ASTContext;
  class BasicValueFactory;
}

namespace clang {

class SymExpr : public llvm::FoldingSetNode {
public:
  enum Kind { BEGIN_SYMBOLS,
              RegionValueKind, ConjuredKind, DerivedKind,
              END_SYMBOLS,
              SymIntKind, SymSymKind };
private:
  Kind K;

protected:
  SymExpr(Kind k) : K(k) {}

public:
  virtual ~SymExpr() {}

  Kind getKind() const { return K; }

  void dump() const;

  virtual void dumpToStream(llvm::raw_ostream &os) const = 0;

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
    return k > BEGIN_SYMBOLS && k < END_SYMBOLS;
  }
};

typedef const SymbolData* SymbolRef;

class SymbolRegionValue : public SymbolData {
  const MemRegion *R;
  // We may cast the region to another type, so the expected type of the symbol
  // may be different from the region's original type.
  QualType T;

public:
  SymbolRegionValue(SymbolID sym, const MemRegion *r, QualType t = QualType())
    : SymbolData(RegionValueKind, sym), R(r), T(t) {}

  const MemRegion* getRegion() const { return R; }

  static void Profile(llvm::FoldingSetNodeID& profile, const MemRegion* R,
		      QualType T) {
    profile.AddInteger((unsigned) RegionValueKind);
    profile.AddPointer(R);
    T.Profile(profile);
  }

  virtual void Profile(llvm::FoldingSetNodeID& profile) {
    Profile(profile, R, T);
  }

  void dumpToStream(llvm::raw_ostream &os) const;

  QualType getType(ASTContext&) const;

  // Implement isa<T> support.
  static inline bool classof(const SymExpr* SE) {
    return SE->getKind() == RegionValueKind;
  }
};

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

  void dumpToStream(llvm::raw_ostream &os) const;

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

class SymbolDerived : public SymbolData {
  SymbolRef parentSymbol;
  const TypedRegion *R;

public:
  SymbolDerived(SymbolID sym, SymbolRef parent, const TypedRegion *r)
    : SymbolData(DerivedKind, sym), parentSymbol(parent), R(r) {}

  SymbolRef getParentSymbol() const { return parentSymbol; }
  const TypedRegion *getRegion() const { return R; }

  QualType getType(ASTContext&) const;

  void dumpToStream(llvm::raw_ostream &os) const;

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

// SymIntExpr - Represents symbolic expression like 'x' + 3.
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

  void dumpToStream(llvm::raw_ostream &os) const;

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

// SymSymExpr - Represents symbolic expression like 'x' + 'y'.
class SymSymExpr : public SymExpr {
  const SymExpr *LHS;
  BinaryOperator::Opcode Op;
  const SymExpr *RHS;
  QualType T;

public:
  SymSymExpr(const SymExpr *lhs, BinaryOperator::Opcode op, const SymExpr *rhs,
             QualType t)
    : SymExpr(SymSymKind), LHS(lhs), Op(op), RHS(rhs), T(t) {}

  const SymExpr *getLHS() const { return LHS; }
  const SymExpr *getRHS() const { return RHS; }

  // FIXME: We probably need to make this out-of-line to avoid redundant
  // generation of virtual functions.
  QualType getType(ASTContext& C) const { return T; }

  void dumpToStream(llvm::raw_ostream &os) const;

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
  DataSetTy DataSet;
  unsigned SymbolCounter;
  llvm::BumpPtrAllocator& BPAlloc;
  BasicValueFactory &BV;
  ASTContext& Ctx;

public:
  SymbolManager(ASTContext& ctx, BasicValueFactory &bv,
                llvm::BumpPtrAllocator& bpalloc)
    : SymbolCounter(0), BPAlloc(bpalloc), BV(bv), Ctx(ctx) {}

  ~SymbolManager();

  static bool canSymbolicate(QualType T);

  /// Make a unique symbol for MemRegion R according to its kind.
  const SymbolRegionValue* getRegionValueSymbol(const MemRegion* R,
						QualType T = QualType());
  const SymbolConjured* getConjuredSymbol(const Stmt* E, QualType T,
                                          unsigned VisitCount,
                                          const void* SymbolTag = 0);

  const SymbolConjured* getConjuredSymbol(const Expr* E, unsigned VisitCount,
                                          const void* SymbolTag = 0) {
    return getConjuredSymbol(E, E->getType(), VisitCount, SymbolTag);
  }

  const SymbolDerived *getDerivedSymbol(SymbolRef parentSymbol,
                                        const TypedRegion *R);

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

  ASTContext &getContext() { return Ctx; }
  BasicValueFactory &getBasicVals() { return BV; }
};

class SymbolReaper {
  typedef llvm::DenseSet<SymbolRef> SetTy;

  SetTy TheLiving;
  SetTy TheDead;
  LiveVariables& Liveness;
  SymbolManager& SymMgr;

public:
  SymbolReaper(LiveVariables& liveness, SymbolManager& symmgr)
    : Liveness(liveness), SymMgr(symmgr) {}

  ~SymbolReaper() {}

  bool isLive(SymbolRef sym);

  bool isLive(const Stmt* Loc, const Stmt* ExprVal) const {
    return Liveness.isLive(Loc, ExprVal);
  }

  bool isLive(const Stmt* Loc, const VarDecl* VD) const {
    return Liveness.isLive(Loc, VD);
  }

  void markLive(SymbolRef sym);
  bool maybeDead(SymbolRef sym);

  typedef SetTy::const_iterator dead_iterator;
  dead_iterator dead_begin() const { return TheDead.begin(); }
  dead_iterator dead_end() const { return TheDead.end(); }

  bool hasDeadSymbols() const {
    return !TheDead.empty();
  }
};

class SymbolVisitor {
public:
  // VisitSymbol - A visitor method invoked by
  //  GRStateManager::scanReachableSymbols.  The method returns \c true if
  //  symbols should continue be scanned and \c false otherwise.
  virtual bool VisitSymbol(SymbolRef sym) = 0;
  virtual ~SymbolVisitor();
};

} // end clang namespace

namespace llvm {
static inline llvm::raw_ostream& operator<<(llvm::raw_ostream& os,
                                            const clang::SymExpr *SE) {
  SE->dumpToStream(os);
  return os;
}
} // end llvm namespace
#endif
