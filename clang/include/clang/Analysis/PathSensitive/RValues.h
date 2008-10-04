//== RValues.h - Abstract RValues for Path-Sens. Value Tracking -*- C++ -*--==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines RVal, LVal, and NonLVal, classes that represent
//  abstract r-values for use with path-sensitive value tracking.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_ANALYSIS_RVALUE_H
#define LLVM_CLANG_ANALYSIS_RVALUE_H

#include "clang/Analysis/PathSensitive/BasicValueFactory.h"
#include "llvm/Support/Casting.h"
  
//==------------------------------------------------------------------------==//
//  Base RVal types.
//==------------------------------------------------------------------------==// 

namespace clang {
  
class MemRegion;
class GRStateManager;
  
class RVal {
public:
  enum BaseKind { UndefinedKind, UnknownKind, LValKind, NonLValKind };
  enum { BaseBits = 2, BaseMask = 0x3 };
  
protected:
  void* Data;
  unsigned Kind;
  
protected:
  RVal(const void* d, bool isLVal, unsigned ValKind)
  : Data(const_cast<void*>(d)),
    Kind((isLVal ? LValKind : NonLValKind) | (ValKind << BaseBits)) {}
  
  explicit RVal(BaseKind k, void* D = NULL)
    : Data(D), Kind(k) {}
  
public:
  ~RVal() {};
  
  /// BufferTy - A temporary buffer to hold a set of RVals.
  typedef llvm::SmallVector<RVal,5> BufferTy;
  
  inline unsigned getRawKind() const { return Kind; }
  inline BaseKind getBaseKind() const { return (BaseKind) (Kind & BaseMask); }
  inline unsigned getSubKind() const { return (Kind & ~BaseMask) >> BaseBits; }
  
  inline void Profile(llvm::FoldingSetNodeID& ID) const {
    ID.AddInteger((unsigned) getRawKind());
    ID.AddPointer(reinterpret_cast<void*>(Data));
  }
  
  inline bool operator==(const RVal& R) const {
    return getRawKind() == R.getRawKind() && Data == R.Data;
  }
  
  
  inline bool operator!=(const RVal& R) const {
    return !(*this == R);
  }
  
  static RVal GetSymbolValue(SymbolManager& SymMgr, VarDecl *D);
  
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
  void printStdErr() const;
  
  typedef const SymbolID* symbol_iterator;
  symbol_iterator symbol_begin() const;
  symbol_iterator symbol_end() const;  
  
  static RVal MakeVal(GRStateManager& SMgr, DeclRefExpr* E);
  
  // Implement isa<T> support.
  static inline bool classof(const RVal*) { return true; }
};

class UnknownVal : public RVal {
public:
  UnknownVal() : RVal(UnknownKind) {}
  
  static inline bool classof(const RVal* V) {
    return V->getBaseKind() == UnknownKind;
  }  
};

class UndefinedVal : public RVal {
public:
  UndefinedVal() : RVal(UndefinedKind) {}
  UndefinedVal(void* D) : RVal(UndefinedKind, D) {}
  
  static inline bool classof(const RVal* V) {
    return V->getBaseKind() == UndefinedKind;
  }
  
  void* getData() const { return Data; }  
};

class NonLVal : public RVal {
protected:
  NonLVal(unsigned SubKind, const void* d) : RVal(d, false, SubKind) {}
  
public:
  void print(std::ostream& Out) const;
  
  // Utility methods to create NonLVals.
  static NonLVal MakeVal(BasicValueFactory& BasicVals, uint64_t X, QualType T);
  
  static NonLVal MakeVal(BasicValueFactory& BasicVals, IntegerLiteral* I);
    
  static NonLVal MakeIntTruthVal(BasicValueFactory& BasicVals, bool b);
    
  // Implement isa<T> support.
  static inline bool classof(const RVal* V) {
    return V->getBaseKind() == NonLValKind;
  }
};

class LVal : public RVal {
protected:
  LVal(unsigned SubKind, const void* D)
    : RVal(const_cast<void*>(D), true, SubKind) {}
  
  // Equality operators.
  NonLVal EQ(BasicValueFactory& BasicVals, const LVal& R) const;
  NonLVal NE(BasicValueFactory& BasicVals, const LVal& R) const;
  
public:
  void print(std::ostream& Out) const;
    
  static LVal MakeVal(AddrLabelExpr* E);
  
  static LVal MakeVal(StringLiteral* S);
  
  // Implement isa<T> support.
  static inline bool classof(const RVal* V) {
    return V->getBaseKind() == LValKind;
  }
  
  static inline bool IsLValType(QualType T) {
    return T->isPointerType() || T->isObjCQualifiedIdType() 
      || T->isBlockPointerType();
  }
};
  
//==------------------------------------------------------------------------==//
//  Subclasses of NonLVal.
//==------------------------------------------------------------------------==// 

namespace nonlval {
  
enum Kind { ConcreteIntKind, SymbolValKind, SymIntConstraintValKind,
            LValAsIntegerKind };

class SymbolVal : public NonLVal {
public:
  SymbolVal(unsigned SymID)
    : NonLVal(SymbolValKind, reinterpret_cast<void*>((uintptr_t) SymID)) {}
  
  SymbolID getSymbol() const {
    return (SymbolID) reinterpret_cast<uintptr_t>(Data);
  }
  
  static inline bool classof(const RVal* V) {
    return V->getBaseKind() == NonLValKind && 
           V->getSubKind() == SymbolValKind;
  }
  
  static inline bool classof(const NonLVal* V) {
    return V->getSubKind() == SymbolValKind;
  }
};

class SymIntConstraintVal : public NonLVal {    
public:
  SymIntConstraintVal(const SymIntConstraint& C)
    : NonLVal(SymIntConstraintValKind, reinterpret_cast<const void*>(&C)) {}

  const SymIntConstraint& getConstraint() const {
    return *reinterpret_cast<SymIntConstraint*>(Data);
  }
  
  static inline bool classof(const RVal* V) {
    return V->getBaseKind() == NonLValKind &&
           V->getSubKind() == SymIntConstraintValKind;
  }
  
  static inline bool classof(const NonLVal* V) {
    return V->getSubKind() == SymIntConstraintValKind;
  }
};

class ConcreteInt : public NonLVal {
public:
  ConcreteInt(const llvm::APSInt& V) : NonLVal(ConcreteIntKind, &V) {}
  
  const llvm::APSInt& getValue() const {
    return *static_cast<llvm::APSInt*>(Data);
  }
  
  // Transfer functions for binary/unary operations on ConcreteInts.
  RVal EvalBinOp(BasicValueFactory& BasicVals, BinaryOperator::Opcode Op,
                 const ConcreteInt& R) const;
  
  ConcreteInt EvalComplement(BasicValueFactory& BasicVals) const;
  
  ConcreteInt EvalMinus(BasicValueFactory& BasicVals, UnaryOperator* U) const;
  
  // Implement isa<T> support.
  static inline bool classof(const RVal* V) {
    return V->getBaseKind() == NonLValKind &&
           V->getSubKind() == ConcreteIntKind;
  }
  
  static inline bool classof(const NonLVal* V) {
    return V->getSubKind() == ConcreteIntKind;
  }
};
  
class LValAsInteger : public NonLVal {
  LValAsInteger(const std::pair<RVal, uintptr_t>& data) :
    NonLVal(LValAsIntegerKind, &data) {
      assert (isa<LVal>(data.first));
    }
  
public:
    
  LVal getLVal() const {
    return cast<LVal>(((std::pair<RVal, uintptr_t>*) Data)->first);
  }
  
  const LVal& getPersistentLVal() const {
    const RVal& V = ((std::pair<RVal, uintptr_t>*) Data)->first;
    return cast<LVal>(V);
  }    
  
  unsigned getNumBits() const {
    return ((std::pair<RVal, unsigned>*) Data)->second;
  }
  
  // Implement isa<T> support.
  static inline bool classof(const RVal* V) {
    return V->getBaseKind() == NonLValKind &&
           V->getSubKind() == LValAsIntegerKind;
  }
  
  static inline bool classof(const NonLVal* V) {
    return V->getSubKind() == LValAsIntegerKind;
  }
  
  static inline LValAsInteger Make(BasicValueFactory& Vals, LVal V,
                                   unsigned Bits) {    
    return LValAsInteger(Vals.getPersistentRValWithData(V, Bits));
  }
};
  
} // end namespace clang::nonlval

//==------------------------------------------------------------------------==//
//  Subclasses of LVal.
//==------------------------------------------------------------------------==// 

namespace lval {
  
enum Kind { SymbolValKind, GotoLabelKind, MemRegionKind, FuncValKind,
            ConcreteIntKind, StringLiteralValKind, FieldOffsetKind,
            ArrayOffsetKind };

class SymbolVal : public LVal {
public:
  SymbolVal(unsigned SymID)
  : LVal(SymbolValKind, reinterpret_cast<void*>((uintptr_t) SymID)) {}
  
  SymbolID getSymbol() const {
    return (SymbolID) reinterpret_cast<uintptr_t>(Data);
  }
  
  static inline bool classof(const RVal* V) {
    return V->getBaseKind() == LValKind &&
           V->getSubKind() == SymbolValKind;
  }
  
  static inline bool classof(const LVal* V) {
    return V->getSubKind() == SymbolValKind;
  }
};

class GotoLabel : public LVal {
public:
  GotoLabel(LabelStmt* Label) : LVal(GotoLabelKind, Label) {}
  
  LabelStmt* getLabel() const {
    return static_cast<LabelStmt*>(Data);
  }
  
  static inline bool classof(const RVal* V) {
    return V->getBaseKind() == LValKind &&
           V->getSubKind() == GotoLabelKind;
  }
  
  static inline bool classof(const LVal* V) {
    return V->getSubKind() == GotoLabelKind;
  } 
};
  

class MemRegionVal : public LVal {
public:
  MemRegionVal(const MemRegion* r) : LVal(MemRegionKind, r) {}

  MemRegion* getRegion() const {
    return static_cast<MemRegion*>(Data);
  }
  
  inline bool operator==(const MemRegionVal& R) const {
    return getRegion() == R.getRegion();
  }
  
  inline bool operator!=(const MemRegionVal& R) const {
    return getRegion() != R.getRegion();
  }
  
  // Implement isa<T> support.
  static inline bool classof(const RVal* V) {
    return V->getBaseKind() == LValKind &&
           V->getSubKind() == MemRegionKind;
  }
  
  static inline bool classof(const LVal* V) {
    return V->getSubKind() == MemRegionKind;
  }    
};

class FuncVal : public LVal {
public:
  FuncVal(const FunctionDecl* fd) : LVal(FuncValKind, fd) {}
  
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
  static inline bool classof(const RVal* V) {
    return V->getBaseKind() == LValKind &&
           V->getSubKind() == FuncValKind;
  }
  
  static inline bool classof(const LVal* V) {
    return V->getSubKind() == FuncValKind;
  }
};

class ConcreteInt : public LVal {
public:
  ConcreteInt(const llvm::APSInt& V) : LVal(ConcreteIntKind, &V) {}
  
  const llvm::APSInt& getValue() const {
    return *static_cast<llvm::APSInt*>(Data);
  }

  // Transfer functions for binary/unary operations on ConcreteInts.
  RVal EvalBinOp(BasicValueFactory& BasicVals, BinaryOperator::Opcode Op,
                 const ConcreteInt& R) const;
      
  // Implement isa<T> support.
  static inline bool classof(const RVal* V) {
    return V->getBaseKind() == LValKind &&
           V->getSubKind() == ConcreteIntKind;
  }
  
  static inline bool classof(const LVal* V) {
    return V->getSubKind() == ConcreteIntKind;
  }
};
  
class StringLiteralVal : public LVal {
public:
  StringLiteralVal(StringLiteral* L) : LVal(StringLiteralValKind, L) {}
  
  StringLiteral* getLiteral() const { return (StringLiteral*) Data; }
  
  // Implement isa<T> support.
  static inline bool classof(const RVal* V) {
    return V->getBaseKind() == LValKind &&
           V->getSubKind() == StringLiteralValKind;
  }
  
  static inline bool classof(const LVal* V) {
    return V->getSubKind() == StringLiteralValKind;
  }
};  
  
class FieldOffset : public LVal {
  FieldOffset(const std::pair<RVal, uintptr_t>& data)
    : LVal(FieldOffsetKind, &data) {}
  
public:
  
  LVal getBase() const {
    return reinterpret_cast<const std::pair<LVal,uintptr_t>*> (Data)->first;
  }  
  
  const LVal& getPersistentBase() const {
    return reinterpret_cast<const std::pair<LVal,uintptr_t>*> (Data)->first;
  }    
  
    
  FieldDecl* getFieldDecl() const {    
    return (FieldDecl*)
      reinterpret_cast<const std::pair<LVal,uintptr_t>*> (Data)->second;
  }

  // Implement isa<T> support.
  static inline bool classof(const RVal* V) {
    return V->getBaseKind() == LValKind &&
           V->getSubKind() == FieldOffsetKind;
  }
  
  static inline bool classof(const LVal* V) {
    return V->getSubKind() == FieldOffsetKind;
  }

  static inline RVal Make(BasicValueFactory& Vals, RVal Base, FieldDecl* D) {
    
    if (Base.isUnknownOrUndef())
      return Base;
    
    return FieldOffset(Vals.getPersistentRValWithData(cast<LVal>(Base),
                                                      (uintptr_t) D));
  }
};
  
class ArrayOffset : public LVal {
  ArrayOffset(const std::pair<RVal,RVal>& data) : LVal(ArrayOffsetKind,&data) {}  
public:
  
  LVal getBase() const {
    return reinterpret_cast<const std::pair<LVal,RVal>*> (Data)->first;
  }  
  
  const LVal& getPersistentBase() const {
    return reinterpret_cast<const std::pair<LVal,RVal>*> (Data)->first;
  }   
  
  RVal getOffset() const {
    return reinterpret_cast<const std::pair<LVal,RVal>*> (Data)->second;
  }  
  
  const RVal& getPersistentOffset() const {
    return reinterpret_cast<const std::pair<LVal,RVal>*> (Data)->second;
  }   
  

  // Implement isa<T> support.
  static inline bool classof(const RVal* V) {
    return V->getBaseKind() == LValKind &&
           V->getSubKind() == ArrayOffsetKind;
  }
  
  static inline bool classof(const LVal* V) {
    return V->getSubKind() == ArrayOffsetKind;
  }
  
  static inline RVal Make(BasicValueFactory& Vals, RVal Base, RVal Offset) {
    
    if (Base.isUnknownOrUndef())
      return Base;
    
    if (Offset.isUndef())
      return Offset;
    
    return ArrayOffset(Vals.getPersistentRValPair(cast<LVal>(Base), Offset));
  }
};
  
} // end clang::lval namespace
} // end clang namespace  

#endif
