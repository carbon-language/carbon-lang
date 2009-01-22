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
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/ImmutableSet.h"

namespace llvm {
  class raw_ostream;
}

namespace clang {
  
class MemRegion;
class SymbolManager;
class ASTContext;

class SymbolRef {
  unsigned Data;
public:
  SymbolRef() : Data(~0U - 2) {}
  SymbolRef(unsigned x) : Data(x) {}
    
  bool isInitialized() const { return Data != (unsigned) (~0U - 2); }
  operator unsigned() const { return getNumber(); }
  unsigned getNumber() const { assert (isInitialized()); return Data; }
  
  bool operator==(const SymbolRef& X) const { return Data == X.Data; }
  bool operator!=(const SymbolRef& X) const { return Data != X.Data; }
    
  void print(llvm::raw_ostream& os) const;
  
  void Profile(llvm::FoldingSetNodeID& ID) const { 
    assert (isInitialized());
    ID.AddInteger(Data);
  }
};
  
} // end clang namespace

namespace llvm {
  template <> struct DenseMapInfo<clang::SymbolRef> {
    static inline clang::SymbolRef getEmptyKey() {
      return clang::SymbolRef(~0U);
    }
    static inline clang::SymbolRef getTombstoneKey() {
      return clang::SymbolRef(~0U - 1);
    }
    static unsigned getHashValue(clang::SymbolRef X) {
      return X.getNumber();
    }
    static bool isEqual(clang::SymbolRef X, clang::SymbolRef Y) {
      return X.getNumber() == Y.getNumber();
    }
    static bool isPod() { return true; }
  };
}

// SymbolData: Used to record meta data about symbols.

namespace clang {
  
class SymbolData : public llvm::FoldingSetNode {
public:
  enum Kind { RegionRValue, ConjuredKind };
  
private:
  Kind K;
  SymbolRef Sym;
  
protected:
  SymbolData(Kind k, SymbolRef sym) : K(k), Sym(sym) {}  

public:
  virtual ~SymbolData() {}
  
  Kind getKind() const { return K; }  
  
  SymbolRef getSymbol() const { return Sym; }
    
  virtual QualType getType(ASTContext&) const = 0;
  
  virtual void Profile(llvm::FoldingSetNodeID& profile) = 0;
  
  // Implement isa<T> support.
  static inline bool classof(const SymbolData*) { return true; }
};
  
class SymbolRegionRValue : public SymbolData {
  const MemRegion *R;
public:
  SymbolRegionRValue(SymbolRef MySym, const MemRegion *r)
    : SymbolData(RegionRValue, MySym), R(r) {}
  
  const MemRegion* getRegion() const { return R; }

  static void Profile(llvm::FoldingSetNodeID& profile, const MemRegion* R) {
    profile.AddInteger((unsigned) RegionRValue);
    profile.AddPointer(R);
  }
  
  virtual void Profile(llvm::FoldingSetNodeID& profile) {
    Profile(profile, R);
  }
  
  QualType getType(ASTContext&) const;

  // Implement isa<T> support.
  static inline bool classof(const SymbolData* D) {
    return D->getKind() == RegionRValue;
  }
};

class SymbolConjured : public SymbolData {
  Stmt* S;
  QualType T;
  unsigned Count;

public:
  SymbolConjured(SymbolRef Sym, Stmt* s, QualType t, unsigned count)
    : SymbolData(ConjuredKind, Sym), S(s), T(t), Count(count) {}
  
  Stmt* getStmt() const { return S; }
  unsigned getCount() const { return Count; }    
  QualType getType(ASTContext&) const;
  
  static void Profile(llvm::FoldingSetNodeID& profile,
                      Stmt* S, QualType T, unsigned Count) {
    
    profile.AddInteger((unsigned) ConjuredKind);
    profile.AddPointer(S);
    profile.Add(T);
    profile.AddInteger(Count);
  }
  
  virtual void Profile(llvm::FoldingSetNodeID& profile) {
    Profile(profile, S, T, Count);
  }
  
  // Implement isa<T> support.
  static inline bool classof(const SymbolData* D) {
    return D->getKind() == ConjuredKind;
  }  
};

// Constraints on symbols.  Usually wrapped by SValues.

class SymIntConstraint : public llvm::FoldingSetNode {
  SymbolRef Symbol;
  BinaryOperator::Opcode Op;
  const llvm::APSInt& Val;
public:  
  SymIntConstraint(SymbolRef sym, BinaryOperator::Opcode op, 
                   const llvm::APSInt& V)
  : Symbol(sym),
  Op(op), Val(V) {}
  
  BinaryOperator::Opcode getOpcode() const { return Op; }
  const SymbolRef& getSymbol() const { return Symbol; }
  const llvm::APSInt& getInt() const { return Val; }
  
  static inline void Profile(llvm::FoldingSetNodeID& ID,
                             SymbolRef Symbol,
                             BinaryOperator::Opcode Op,
                             const llvm::APSInt& Val) {
    Symbol.Profile(ID);
    ID.AddInteger(Op);
    ID.AddPointer(&Val);
  }
  
  void Profile(llvm::FoldingSetNodeID& ID) {
    Profile(ID, Symbol, Op, Val);
  }
};


class SymbolManager {
  typedef llvm::FoldingSet<SymbolData> DataSetTy;
  typedef llvm::DenseMap<SymbolRef, SymbolData*> DataMapTy;
  
  DataSetTy DataSet;
  DataMapTy DataMap;
  
  unsigned SymbolCounter;
  llvm::BumpPtrAllocator& BPAlloc;
  ASTContext& Ctx;
  
public:
  SymbolManager(ASTContext& ctx, llvm::BumpPtrAllocator& bpalloc)
    : SymbolCounter(0), BPAlloc(bpalloc), Ctx(ctx) {}
  
  ~SymbolManager();

  /// Make a unique symbol for MemRegion R according to its kind.
  SymbolRef getRegionRValueSymbol(const MemRegion* R);
  SymbolRef getConjuredSymbol(Stmt* E, QualType T, unsigned VisitCount);
  SymbolRef getConjuredSymbol(Expr* E, unsigned VisitCount) {
    return getConjuredSymbol(E, E->getType(), VisitCount);
  }
  
  const SymbolData& getSymbolData(SymbolRef ID) const;
  
  QualType getType(SymbolRef ID) const {
    return getSymbolData(ID).getType(Ctx);
  }
  
  ASTContext& getContext() { return Ctx; }
};
  
class SymbolReaper {
  typedef llvm::ImmutableSet<SymbolRef> SetTy;
  typedef SetTy::Factory FactoryTy;
  
  FactoryTy F;
  SetTy TheLiving;
  SetTy TheDead;
  LiveVariables& Liveness;
  SymbolManager& SymMgr;
  
public:
  SymbolReaper(LiveVariables& liveness, SymbolManager& symmgr)
  : TheLiving(F.GetEmptySet()), TheDead(F.GetEmptySet()),
    Liveness(liveness), SymMgr(symmgr) {}

  bool isLive(SymbolRef sym);

  bool isLive(const Stmt* Loc, const Stmt* ExprVal) const {
    return Liveness.isLive(Loc, ExprVal);
  }

  bool isLive(const Stmt* Loc, const VarDecl* VD) const {
    return Liveness.isLive(Loc, VD);
  }
  
  void markLive(SymbolRef sym);
  bool maybeDead(SymbolRef sym);
  
  typedef SetTy::iterator dead_iterator;
  dead_iterator dead_begin() const { return TheDead.begin(); }
  dead_iterator dead_end() const { return TheDead.end(); }
  
  bool hasDeadSymbols() const {
    return !TheDead.isEmpty();
  }
};
  
} // end clang namespace

#endif
