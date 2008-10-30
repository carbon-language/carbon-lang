//== SVals.h - Abstract Values for Static Analysis ---------*- C++ -*--==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines SVal, Loc, and NonLoc, classes that represent
//  abstract r-values for use with path-sensitive value tracking.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_ANALYSIS_RVALUE_H
#define LLVM_CLANG_ANALYSIS_RVALUE_H

#include "clang/Analysis/PathSensitive/SymbolManager.h"
#include "llvm/Support/Casting.h"
#include "llvm/ADT/ImmutableList.h"
  
//==------------------------------------------------------------------------==//
//  Base SVal types.
//==------------------------------------------------------------------------==//

namespace clang {

class CompoundValData;
class BasicValueFactory;
class MemRegion;
class GRStateManager;
  
class SVal {
public:
  enum BaseKind { UndefinedKind, UnknownKind, LocKind, NonLocKind };
  enum { BaseBits = 2, BaseMask = 0x3 };
  
protected:
  void* Data;
  unsigned Kind;
  
protected:
  SVal(const void* d, bool isLoc, unsigned ValKind)
  : Data(const_cast<void*>(d)),
    Kind((isLoc ? LocKind : NonLocKind) | (ValKind << BaseBits)) {}
  
  explicit SVal(BaseKind k, void* D = NULL)
    : Data(D), Kind(k) {}
  
public:
  SVal() : Data(0), Kind(0) {}
  ~SVal() {};
  
  /// BufferTy - A temporary buffer to hold a set of SVals.
  typedef llvm::SmallVector<SVal,5> BufferTy;
  
  inline unsigned getRawKind() const { return Kind; }
  inline BaseKind getBaseKind() const { return (BaseKind) (Kind & BaseMask); }
  inline unsigned getSubKind() const { return (Kind & ~BaseMask) >> BaseBits; }
  
  inline void Profile(llvm::FoldingSetNodeID& ID) const {
    ID.AddInteger((unsigned) getRawKind());
    ID.AddPointer(reinterpret_cast<void*>(Data));
  }

  inline bool operator==(const SVal& R) const {
    return getRawKind() == R.getRawKind() && Data == R.Data;
  }
  
  
  inline bool operator!=(const SVal& R) const {
    return !(*this == R);
  }
  
  static SVal GetSymbolValue(SymbolManager& SymMgr, VarDecl *D);
  
  inline bool isUnknown() const {
    return getRawKind() == UnknownKind;
  }

  inline bool isUndef() const {
    return getRawKind() == UndefinedKind;
  }

  inline bool isUnknownOrUndef() const {
    return getRawKind() <= UnknownKind;
  }
  
  inline bool isValid() const {
    return getRawKind() > UnknownKind;
  }
  
  bool isZeroConstant() const;
  
  void print(std::ostream& OS) const;
  void print(llvm::raw_ostream& OS) const;
  void printStdErr() const;
  
  class symbol_iterator {
    const enum { One, Many } HowMany;
    union { uintptr_t sym; const SymbolID* sptr; };
  public:
    
    bool operator==(const symbol_iterator& X) {
      return X.sym == sym;
    }
    
    bool operator!=(const symbol_iterator& X) {
      return X.sym != sym;
    }
    
    symbol_iterator& operator++() {
      if (HowMany == Many)
        ++sptr;
      else
        sym = ~0x0;
      
      return *this;
    }
    
    SymbolID operator*() const {
      if (HowMany)
        return *sptr;
      
      return SymbolID(sym);
    }
    
    symbol_iterator(SymbolID x) : HowMany(One), sym(x.getNumber()) {}
    symbol_iterator() : HowMany(One), sym(~0x0) {}
    symbol_iterator(const SymbolID* x) : HowMany(Many), sptr(x) {}
  };
  
  symbol_iterator symbol_begin() const;
  symbol_iterator symbol_end() const;
  
  // Implement isa<T> support.
  static inline bool classof(const SVal*) { return true; }
};

class UnknownVal : public SVal {
public:
  UnknownVal() : SVal(UnknownKind) {}
  
  static inline bool classof(const SVal* V) {
    return V->getBaseKind() == UnknownKind;
  }  
};

class UndefinedVal : public SVal {
public:
  UndefinedVal() : SVal(UndefinedKind) {}
  UndefinedVal(void* D) : SVal(UndefinedKind, D) {}
  
  static inline bool classof(const SVal* V) {
    return V->getBaseKind() == UndefinedKind;
  }
  
  void* getData() const { return Data; }  
};

class NonLoc : public SVal {
protected:
  NonLoc(unsigned SubKind, const void* d) : SVal(d, false, SubKind) {}
  
public:
  void print(std::ostream& Out) const;
  void print(llvm::raw_ostream& Out) const;
  
  // Utility methods to create NonLocs.
  static NonLoc MakeVal(BasicValueFactory& BasicVals, uint64_t X, QualType T);
  
  static NonLoc MakeVal(BasicValueFactory& BasicVals, IntegerLiteral* I);
    
  static NonLoc MakeIntTruthVal(BasicValueFactory& BasicVals, bool b);

  static NonLoc MakeCompoundVal(QualType T, llvm::ImmutableList<SVal> Vals,
                                BasicValueFactory& BasicVals);

  // Implement isa<T> support.
  static inline bool classof(const SVal* V) {
    return V->getBaseKind() == NonLocKind;
  }
};

class Loc : public SVal {
protected:
  Loc(unsigned SubKind, const void* D)
    : SVal(const_cast<void*>(D), true, SubKind) {}
  
  // Equality operators.
  NonLoc EQ(BasicValueFactory& BasicVals, const Loc& R) const;
  NonLoc NE(BasicValueFactory& BasicVals, const Loc& R) const;
  
public:
  void print(std::ostream& Out) const;
  void print(llvm::raw_ostream& Out) const;
    
  static Loc MakeVal(AddrLabelExpr* E);
  
  static Loc MakeVal(StringLiteral* S);
  
  // Implement isa<T> support.
  static inline bool classof(const SVal* V) {
    return V->getBaseKind() == LocKind;
  }
  
  static inline bool IsLocType(QualType T) {
    return T->isPointerType() || T->isObjCQualifiedIdType() 
      || T->isBlockPointerType();
  }
};
  
//==------------------------------------------------------------------------==//
//  Subclasses of NonLoc.
//==------------------------------------------------------------------------==//

namespace nonloc {
  
enum Kind { ConcreteIntKind, SymbolValKind, SymIntConstraintValKind,
            LocAsIntegerKind, CompoundValKind };

class SymbolVal : public NonLoc {
public:
  SymbolVal(unsigned SymID)
    : NonLoc(SymbolValKind, reinterpret_cast<void*>((uintptr_t) SymID)) {}
  
  SymbolID getSymbol() const {
    return (SymbolID) reinterpret_cast<uintptr_t>(Data);
  }
  
  static inline bool classof(const SVal* V) {
    return V->getBaseKind() == NonLocKind && 
           V->getSubKind() == SymbolValKind;
  }
  
  static inline bool classof(const NonLoc* V) {
    return V->getSubKind() == SymbolValKind;
  }
};

class SymIntConstraintVal : public NonLoc {    
public:
  SymIntConstraintVal(const SymIntConstraint& C)
    : NonLoc(SymIntConstraintValKind, reinterpret_cast<const void*>(&C)) {}

  const SymIntConstraint& getConstraint() const {
    return *reinterpret_cast<SymIntConstraint*>(Data);
  }
  
  static inline bool classof(const SVal* V) {
    return V->getBaseKind() == NonLocKind &&
           V->getSubKind() == SymIntConstraintValKind;
  }
  
  static inline bool classof(const NonLoc* V) {
    return V->getSubKind() == SymIntConstraintValKind;
  }
};

class ConcreteInt : public NonLoc {
public:
  ConcreteInt(const llvm::APSInt& V) : NonLoc(ConcreteIntKind, &V) {}
  
  const llvm::APSInt& getValue() const {
    return *static_cast<llvm::APSInt*>(Data);
  }
  
  // Transfer functions for binary/unary operations on ConcreteInts.
  SVal EvalBinOp(BasicValueFactory& BasicVals, BinaryOperator::Opcode Op,
                 const ConcreteInt& R) const;
  
  ConcreteInt EvalComplement(BasicValueFactory& BasicVals) const;
  
  ConcreteInt EvalMinus(BasicValueFactory& BasicVals, UnaryOperator* U) const;
  
  // Implement isa<T> support.
  static inline bool classof(const SVal* V) {
    return V->getBaseKind() == NonLocKind &&
           V->getSubKind() == ConcreteIntKind;
  }
  
  static inline bool classof(const NonLoc* V) {
    return V->getSubKind() == ConcreteIntKind;
  }
};
  
class LocAsInteger : public NonLoc {
  LocAsInteger(const std::pair<SVal, uintptr_t>& data) :
    NonLoc(LocAsIntegerKind, &data) {
      assert (isa<Loc>(data.first));
    }
  
public:
    
  Loc getLoc() const {
    return cast<Loc>(((std::pair<SVal, uintptr_t>*) Data)->first);
  }
  
  const Loc& getPersistentLoc() const {
    const SVal& V = ((std::pair<SVal, uintptr_t>*) Data)->first;
    return cast<Loc>(V);
  }    
  
  unsigned getNumBits() const {
    return ((std::pair<SVal, unsigned>*) Data)->second;
  }
  
  // Implement isa<T> support.
  static inline bool classof(const SVal* V) {
    return V->getBaseKind() == NonLocKind &&
           V->getSubKind() == LocAsIntegerKind;
  }
  
  static inline bool classof(const NonLoc* V) {
    return V->getSubKind() == LocAsIntegerKind;
  }
  
  static LocAsInteger Make(BasicValueFactory& Vals, Loc V, unsigned Bits);
};

class CompoundVal : public NonLoc {
  friend class NonLoc;

  CompoundVal(const CompoundValData* D) : NonLoc(CompoundValKind, D) {}

public:
  const CompoundValData* getValue() const {
    return static_cast<CompoundValData*>(Data);
  }
  
  typedef llvm::ImmutableList<SVal>::iterator iterator;
  iterator begin() const;
  iterator end() const;  

  static bool classof(const SVal* V) {
    return V->getBaseKind() == NonLocKind && V->getSubKind() == CompoundValKind;
  }

  static bool classof(const NonLoc* V) {
    return V->getSubKind() == CompoundValKind;
  }
};
  
} // end namespace clang::nonloc

//==------------------------------------------------------------------------==//
//  Subclasses of Loc.
//==------------------------------------------------------------------------==//

namespace loc {
  
enum Kind { SymbolValKind, GotoLabelKind, MemRegionKind, FuncValKind,
            ConcreteIntKind };

class SymbolVal : public Loc {
public:
  SymbolVal(unsigned SymID)
  : Loc(SymbolValKind, reinterpret_cast<void*>((uintptr_t) SymID)) {}
  
  SymbolID getSymbol() const {
    return (SymbolID) reinterpret_cast<uintptr_t>(Data);
  }
  
  static inline bool classof(const SVal* V) {
    return V->getBaseKind() == LocKind &&
           V->getSubKind() == SymbolValKind;
  }
  
  static inline bool classof(const Loc* V) {
    return V->getSubKind() == SymbolValKind;
  }
};

class GotoLabel : public Loc {
public:
  GotoLabel(LabelStmt* Label) : Loc(GotoLabelKind, Label) {}
  
  LabelStmt* getLabel() const {
    return static_cast<LabelStmt*>(Data);
  }
  
  static inline bool classof(const SVal* V) {
    return V->getBaseKind() == LocKind &&
           V->getSubKind() == GotoLabelKind;
  }
  
  static inline bool classof(const Loc* V) {
    return V->getSubKind() == GotoLabelKind;
  } 
};
  

class MemRegionVal : public Loc {
public:
  MemRegionVal(const MemRegion* r) : Loc(MemRegionKind, r) {}

  const MemRegion* getRegion() const {
    return static_cast<MemRegion*>(Data);
  }
  
  template <typename REGION>
  const REGION* getRegionAs() const {
    return llvm::dyn_cast<REGION>(getRegion());
  }  
  
  inline bool operator==(const MemRegionVal& R) const {
    return getRegion() == R.getRegion();
  }
  
  inline bool operator!=(const MemRegionVal& R) const {
    return getRegion() != R.getRegion();
  }
  
  // Implement isa<T> support.
  static inline bool classof(const SVal* V) {
    return V->getBaseKind() == LocKind &&
           V->getSubKind() == MemRegionKind;
  }
  
  static inline bool classof(const Loc* V) {
    return V->getSubKind() == MemRegionKind;
  }    
};

class FuncVal : public Loc {
public:
  FuncVal(const FunctionDecl* fd) : Loc(FuncValKind, fd) {}
  
  FunctionDecl* getDecl() const {
    return static_cast<FunctionDecl*>(Data);
  }
  
  inline bool operator==(const FuncVal& R) const {
    return getDecl() == R.getDecl();
  }
  
  inline bool operator!=(const FuncVal& R) const {
    return getDecl() != R.getDecl();
  }
  
  // Implement isa<T> support.
  static inline bool classof(const SVal* V) {
    return V->getBaseKind() == LocKind &&
           V->getSubKind() == FuncValKind;
  }
  
  static inline bool classof(const Loc* V) {
    return V->getSubKind() == FuncValKind;
  }
};

class ConcreteInt : public Loc {
public:
  ConcreteInt(const llvm::APSInt& V) : Loc(ConcreteIntKind, &V) {}
  
  const llvm::APSInt& getValue() const {
    return *static_cast<llvm::APSInt*>(Data);
  }

  // Transfer functions for binary/unary operations on ConcreteInts.
  SVal EvalBinOp(BasicValueFactory& BasicVals, BinaryOperator::Opcode Op,
                 const ConcreteInt& R) const;
      
  // Implement isa<T> support.
  static inline bool classof(const SVal* V) {
    return V->getBaseKind() == LocKind &&
           V->getSubKind() == ConcreteIntKind;
  }
  
  static inline bool classof(const Loc* V) {
    return V->getSubKind() == ConcreteIntKind;
  }
};
  
} // end clang::loc namespace
} // end clang namespace  

#endif
