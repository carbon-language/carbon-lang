//== SymbolManager.h - Management of Symbolic Values ------------*- C++ -*--==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This files defines SymbolManager, a class that manages symbolic values
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
#include <vector>

namespace clang {
  
class SymbolManager;
  
class SymbolID {
  unsigned Data;
public:
  SymbolID() : Data(~0) {}
  SymbolID(unsigned x) : Data(x) {}
  
  bool isInitialized() const { return Data != (unsigned) ~0; }
  operator unsigned() const { assert (isInitialized()); return Data; }
  
  void Profile(llvm::FoldingSetNodeID& ID) const { 
    assert (isInitialized());
    ID.AddInteger(Data);
  }
  
  static inline void Profile(llvm::FoldingSetNodeID& ID, SymbolID X) {
    X.Profile(ID);
  }
};

// SymbolData: Used to record meta data about symbols.

class SymbolData {
public:
  enum Kind { UninitKind, ParmKind, GlobalKind, ContentsOfKind };
  
private:
  uintptr_t Data;
  Kind K;
  
protected:
  SymbolData(uintptr_t D, Kind k) : Data(D), K(k) {}
  SymbolData(void* D, Kind k) : Data(reinterpret_cast<uintptr_t>(D)), K(k) {}
  
  void* getPtr() const { 
    assert (K != UninitKind);
    return reinterpret_cast<void*>(Data);
  }
  
  uintptr_t getInt() const {
    assert (K != UninitKind);
    return Data;
  }
  
public:
  SymbolData() : Data(0), K(UninitKind) {}
  
  Kind  getKind() const { return K; }  
  
  inline bool operator==(const SymbolData& R) const { 
    return K == R.K && Data == R.Data;
  }
  
  QualType getType(const SymbolManager& SymMgr) const;
  
  // Implement isa<T> support.
  static inline bool classof(const SymbolData*) { return true; }
};

class SymbolDataParmVar : public SymbolData {
public:
  SymbolDataParmVar(ParmVarDecl* VD) : SymbolData(VD, ParmKind) {}
  
  ParmVarDecl* getDecl() const { return (ParmVarDecl*) getPtr(); }
  
  // Implement isa<T> support.
  static inline bool classof(const SymbolData* D) {
    return D->getKind() == ParmKind;
  }
};
  
class SymbolDataGlobalVar : public SymbolData {
public:
  SymbolDataGlobalVar(VarDecl* VD) : SymbolData(VD, GlobalKind) {}
  
  VarDecl* getDecl() const { return (VarDecl*) getPtr(); }
  
  // Implement isa<T> support.
  static inline bool classof(const SymbolData* D) {
    return D->getKind() == GlobalKind;
  }
};

class SymbolDataContentsOf : public SymbolData {
public:
  SymbolDataContentsOf(SymbolID ID) : SymbolData(ID, ContentsOfKind) {}
  
  SymbolID getSymbol() const { return (SymbolID) getInt(); }
  
  // Implement isa<T> support.
  static inline bool classof(const SymbolData* D) {
    return D->getKind() == ContentsOfKind;
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
                             const SymbolID& Symbol,
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
  std::vector<SymbolData> SymbolToData;
  
  typedef llvm::DenseMap<void*,SymbolID> MapTy;
  MapTy DataToSymbol;
  
  void* getKey(void* P) const {
    return reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(P) | 0x1);
  }
  
  void* getKey(SymbolID sym) const {
    return reinterpret_cast<void*>((uintptr_t) (sym << 1));
  }
  
public:
  SymbolManager();
  ~SymbolManager();
  
  SymbolID getSymbol(VarDecl* D);
  SymbolID getContentsOfSymbol(SymbolID sym);
  
  inline const SymbolData& getSymbolData(SymbolID ID) const {
    assert (ID < SymbolToData.size());
    return SymbolToData[ID];
  }
  
  inline QualType getType(SymbolID ID) const {
    return getSymbolData(ID).getType(*this);
  }
};

} // end clang namespace

#endif
