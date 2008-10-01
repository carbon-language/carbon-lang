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
#include "llvm/ADT/FoldingSet.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/DataTypes.h"
#include "llvm/Support/Allocator.h"

namespace clang {
  

class SymbolManager;

class SymbolID {
  unsigned Data;
public:
  SymbolID() : Data(~0U - 2) {}
  SymbolID(unsigned x) : Data(x) {}
    
  bool isInitialized() const { return Data != (unsigned) (~0U - 2); }
  operator unsigned() const { return getNumber(); }
  unsigned getNumber() const { assert (isInitialized()); return Data; }
    
  void Profile(llvm::FoldingSetNodeID& ID) const { 
    assert (isInitialized());
    ID.AddInteger(Data);
  }
};
  
} // end clang namespace

namespace llvm {
  template <> struct DenseMapInfo<clang::SymbolID> {
    static inline clang::SymbolID getEmptyKey() {
      return clang::SymbolID(~0U);
    }
    static inline clang::SymbolID getTombstoneKey() {
      return clang::SymbolID(~0U - 1);
    }
    static unsigned getHashValue(clang::SymbolID X) {
      return X.getNumber();
    }
    static bool isEqual(clang::SymbolID X, clang::SymbolID Y) {
      return X.getNumber() == Y.getNumber();
    }
    static bool isPod() { return true; }
  };
}

// SymbolData: Used to record meta data about symbols.

namespace clang {
  
class SymbolData : public llvm::FoldingSetNode {
public:
  enum Kind { UndefKind, ParmKind, GlobalKind, ContentsOfKind, ConjuredKind };
  
private:
  Kind K;
  SymbolID Sym;
  
protected:
  SymbolData(Kind k, SymbolID sym) : K(k), Sym(sym) {}  

public:
  virtual ~SymbolData() {}
  
  Kind getKind() const { return K; }  
  
  SymbolID getSymbol() const { return Sym; }
    
  QualType getType(const SymbolManager& SymMgr) const;
  
  virtual void Profile(llvm::FoldingSetNodeID& profile) = 0;
  
  // Implement isa<T> support.
  static inline bool classof(const SymbolData*) { return true; }
};

class SymbolDataParmVar : public SymbolData {
  ParmVarDecl *VD;

public:  
  SymbolDataParmVar(SymbolID MySym, ParmVarDecl* vd)
    : SymbolData(ParmKind, MySym), VD(vd) {}
  
  ParmVarDecl* getDecl() const { return VD; }  
  
  static void Profile(llvm::FoldingSetNodeID& profile, ParmVarDecl* VD) {
    profile.AddInteger((unsigned) ParmKind);
    profile.AddPointer(VD);
  }
  
  virtual void Profile(llvm::FoldingSetNodeID& profile) {
    Profile(profile, VD);
  }
  
  // Implement isa<T> support.
  static inline bool classof(const SymbolData* D) {
    return D->getKind() == ParmKind;
  }
};
  
class SymbolDataGlobalVar : public SymbolData {
  VarDecl *VD;

public:
  SymbolDataGlobalVar(SymbolID MySym, VarDecl* vd) :
    SymbolData(GlobalKind, MySym), VD(vd) {}
  
  VarDecl* getDecl() const { return VD; }
  
  static void Profile(llvm::FoldingSetNodeID& profile, VarDecl* VD) {
    profile.AddInteger((unsigned) GlobalKind);
    profile.AddPointer(VD);
  }
  
  virtual void Profile(llvm::FoldingSetNodeID& profile) {
    Profile(profile, VD);
  }
  
  // Implement isa<T> support.
  static inline bool classof(const SymbolData* D) {
    return D->getKind() == GlobalKind;
  }
};

class SymbolDataContentsOf : public SymbolData {
  SymbolID Sym;
      
public:
  SymbolDataContentsOf(SymbolID MySym, SymbolID sym) : 
    SymbolData(ContentsOfKind, MySym), Sym(sym) {}
  
  SymbolID getContainerSymbol() const { return Sym; }
  
  static void Profile(llvm::FoldingSetNodeID& profile, SymbolID Sym) {
    profile.AddInteger((unsigned) ContentsOfKind);
    profile.AddInteger(Sym);
  }
  
  virtual void Profile(llvm::FoldingSetNodeID& profile) {
    Profile(profile, Sym);
  }
  
  // Implement isa<T> support.
  static inline bool classof(const SymbolData* D) {
    return D->getKind() == ContentsOfKind;
  }  
};
  
class SymbolConjured : public SymbolData {
  Expr* E;
  QualType T;
  unsigned Count;

public:
  SymbolConjured(SymbolID Sym, Expr* exp, QualType t, unsigned count)
    : SymbolData(ConjuredKind, Sym), E(exp), T(t), Count(count) {}
  
  Expr* getExpr() const { return E; }
  unsigned getCount() const { return Count; }  
  
  QualType getType() const { return T; }
  
  static void Profile(llvm::FoldingSetNodeID& profile,
                      Expr* E, QualType T, unsigned Count) {
    
    profile.AddInteger((unsigned) ConjuredKind);
    profile.AddPointer(E);
    profile.Add(T);
    profile.AddInteger(Count);
  }
  
  virtual void Profile(llvm::FoldingSetNodeID& profile) {
    Profile(profile, E, T, Count);
  }
  
  // Implement isa<T> support.
  static inline bool classof(const SymbolData* D) {
    return D->getKind() == ConjuredKind;
  }  
};

// Constraints on symbols.  Usually wrapped by RValues.

class SymIntConstraint : public llvm::FoldingSetNode {
  SymbolID Symbol;
  BinaryOperator::Opcode Op;
  const llvm::APSInt& Val;
public:  
  SymIntConstraint(SymbolID sym, BinaryOperator::Opcode op, 
                   const llvm::APSInt& V)
  : Symbol(sym),
  Op(op), Val(V) {}
  
  BinaryOperator::Opcode getOpcode() const { return Op; }
  const SymbolID& getSymbol() const { return Symbol; }
  const llvm::APSInt& getInt() const { return Val; }
  
  static inline void Profile(llvm::FoldingSetNodeID& ID,
                             SymbolID Symbol,
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
  typedef llvm::DenseMap<SymbolID, SymbolData*> DataMapTy;
  
  DataSetTy DataSet;
  DataMapTy DataMap;
  
  unsigned SymbolCounter;
  llvm::BumpPtrAllocator& BPAlloc;
  
public:
  SymbolManager(llvm::BumpPtrAllocator& bpalloc)
    : SymbolCounter(0), BPAlloc(bpalloc) {}
  
  ~SymbolManager();
  
  SymbolID getSymbol(VarDecl* D);
  SymbolID getContentsOfSymbol(SymbolID sym);
  SymbolID getConjuredSymbol(Expr* E, QualType T, unsigned VisitCount);
  SymbolID getConjuredSymbol(Expr* E, unsigned VisitCount) {
    return getConjuredSymbol(E, E->getType(), VisitCount);
  }
  
  const SymbolData& getSymbolData(SymbolID ID) const;
  
  QualType getType(SymbolID ID) const {
    return getSymbolData(ID).getType(*this);
  }
};

} // end clang namespace

#endif
