//== RValues.h - Abstract RValues for Path-Sens. Value Tracking -*- C++ -*--==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This files defines RValue, LValue, and NonLValue, classes that represent
//  abstract r-values for use with path-sensitive value tracking.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_ANALYSIS_RVALUE_H
#define LLVM_CLANG_ANALYSIS_RVALUE_H

#include "clang/Analysis/PathSensitive/ValueManager.h"
#include "llvm/Support/Casting.h"
  
//==------------------------------------------------------------------------==//
//  Base RValue types.
//==------------------------------------------------------------------------==// 

namespace clang {
  
class RValue {
public:
  enum BaseKind { LValueKind=0x0,
                  NonLValueKind=0x1,
                  UninitializedKind=0x2,
                  UnknownKind=0x3 };
  
  enum { BaseBits = 2, 
         BaseMask = 0x3 };
  
protected:
  void* Data;
  unsigned Kind;
  
protected:
  RValue(const void* d, bool isLValue, unsigned ValKind)
  : Data(const_cast<void*>(d)),
    Kind((isLValue ? LValueKind : NonLValueKind) | (ValKind << BaseBits)) {}
  
  explicit RValue(BaseKind k)
    : Data(0), Kind(k) {}
  
public:
  ~RValue() {};
  
  /// BufferTy - A temporary buffer to hold a set of RValues.
  typedef llvm::SmallVector<RValue,5> BufferTy;
  
  unsigned getRawKind() const { return Kind; }
  BaseKind getBaseKind() const { return (BaseKind) (Kind & BaseMask); }
  unsigned getSubKind() const { return (Kind & ~BaseMask) >> BaseBits; }
  
  void Profile(llvm::FoldingSetNodeID& ID) const {
    ID.AddInteger((unsigned) getRawKind());
    ID.AddPointer(reinterpret_cast<void*>(Data));
  }
  
  bool operator==(const RValue& RHS) const {
    return getRawKind() == RHS.getRawKind() && Data == RHS.Data;
  }
  
  static RValue GetSymbolValue(SymbolManager& SymMgr, ParmVarDecl *D);
  
  
  inline bool isKnown() const { return getRawKind() != UnknownKind; }
  inline bool isUnknown() const { return getRawKind() == UnknownKind; }
  
  void print(std::ostream& OS) const;
  void print() const;
  
  typedef const SymbolID* symbol_iterator;
  symbol_iterator symbol_begin() const;
  symbol_iterator symbol_end() const;  
  
  // Implement isa<T> support.
  static inline bool classof(const RValue*) { return true; }
};

class UnknownVal : public RValue {
public:
  UnknownVal() : RValue(UnknownKind) {}
  
  static inline bool classof(const RValue* V) {
    return V->getBaseKind() == UnknownKind;
  }  
};

class UninitializedVal : public RValue {
public:
  UninitializedVal() : RValue(UninitializedKind) {}
  
  static inline bool classof(const RValue* V) {
    return V->getBaseKind() == UninitializedKind;
  }  
};

class NonLValue : public RValue {
protected:
  NonLValue(unsigned SubKind, const void* d) : RValue(d, false, SubKind) {}
  
public:
  void print(std::ostream& Out) const;
  
  // Utility methods to create NonLValues.
  static NonLValue GetValue(ValueManager& ValMgr, uint64_t X, QualType T,
                            SourceLocation Loc = SourceLocation());
  
  static NonLValue GetValue(ValueManager& ValMgr, IntegerLiteral* I);
  
  static NonLValue GetIntTruthValue(ValueManager& ValMgr, bool b);
    
  // Implement isa<T> support.
  static inline bool classof(const RValue* V) {
    return V->getBaseKind() >= NonLValueKind;
  }
};

class LValue : public RValue {
protected:
  LValue(unsigned SubKind, const void* D) : RValue(const_cast<void*>(D), 
                                                   true, SubKind) {}

  // Equality operators.
  NonLValue EQ(ValueManager& ValMgr, const LValue& RHS) const;
  NonLValue NE(ValueManager& ValMgr, const LValue& RHS) const;
  
public:
  void print(std::ostream& Out) const;
    
  static LValue GetValue(AddrLabelExpr* E);
  
  // Implement isa<T> support.
  static inline bool classof(const RValue* V) {
    return V->getBaseKind() != NonLValueKind;
  }
};
  
//==------------------------------------------------------------------------==//
//  Subclasses of NonLValue.
//==------------------------------------------------------------------------==// 

namespace nonlval {
  
  enum Kind { ConcreteIntKind,
              SymbolValKind,
              SymIntConstraintValKind,
              NumKind };

  class SymbolVal : public NonLValue {
  public:
    SymbolVal(unsigned SymID)
    : NonLValue(SymbolValKind,
                reinterpret_cast<void*>((uintptr_t) SymID)) {}
    
    SymbolID getSymbol() const {
      return (SymbolID) reinterpret_cast<uintptr_t>(Data);
    }
    
    static inline bool classof(const RValue* V) {
      return isa<NonLValue>(V) && V->getSubKind() == SymbolValKind;
    }
  };
  
  class SymIntConstraintVal : public NonLValue {    
  public:
    SymIntConstraintVal(const SymIntConstraint& C)
    : NonLValue(SymIntConstraintValKind, reinterpret_cast<const void*>(&C)) {}

    const SymIntConstraint& getConstraint() const {
      return *reinterpret_cast<SymIntConstraint*>(Data);
    }
    
    static inline bool classof(const RValue* V) {
      return isa<NonLValue>(V) && V->getSubKind() == SymIntConstraintValKind;
    }    
  };

  class ConcreteInt : public NonLValue {
  public:
    ConcreteInt(const llvm::APSInt& V) : NonLValue(ConcreteIntKind, &V) {}
    
    const llvm::APSInt& getValue() const {
      return *static_cast<llvm::APSInt*>(Data);
    }
    
    // Transfer functions for binary/unary operations on ConcreteInts.
    ConcreteInt EvalBinaryOp(ValueManager& ValMgr,
                             BinaryOperator::Opcode Op,
                             const ConcreteInt& RHS) const;
    
    ConcreteInt EvalComplement(ValueManager& ValMgr) const;
    ConcreteInt EvalMinus(ValueManager& ValMgr, UnaryOperator* U) const;
    
    // Implement isa<T> support.
    static inline bool classof(const RValue* V) {
      return isa<NonLValue>(V) && V->getSubKind() == ConcreteIntKind;
    }
    
    static inline bool classof(const NonLValue* V) {
      return V->getSubKind() == ConcreteIntKind;
    }
  };
  
} // end namespace clang::nonlval

//==------------------------------------------------------------------------==//
//  Subclasses of LValue.
//==------------------------------------------------------------------------==// 

namespace lval {
  
  enum Kind { SymbolValKind,
              GotoLabelKind,
              DeclValKind,
              FuncValKind,
              ConcreteIntKind,
              NumKind };
  
  class SymbolVal : public LValue {
  public:
    SymbolVal(unsigned SymID)
    : LValue(SymbolValKind, reinterpret_cast<void*>((uintptr_t) SymID)) {}
    
    SymbolID getSymbol() const {
      return (SymbolID) reinterpret_cast<uintptr_t>(Data);
    }
    
    static inline bool classof(const RValue* V) {
      return isa<LValue>(V) && V->getSubKind() == SymbolValKind;
    }
    
    static inline bool classof(const LValue* V) {
      return V->getSubKind() == SymbolValKind;
    }
  };
  
  class GotoLabel : public LValue {
  public:
    GotoLabel(LabelStmt* Label) : LValue(GotoLabelKind, Label) {}
    
    LabelStmt* getLabel() const {
      return static_cast<LabelStmt*>(Data);
    }
    
    static inline bool classof(const RValue* V) {
      return isa<LValue>(V) && V->getSubKind() == GotoLabelKind;
    }
    
    static inline bool classof(const LValue* V) {
      return V->getSubKind() == GotoLabelKind;
    }
  };
    
  
  class DeclVal : public LValue {
  public:
    DeclVal(const VarDecl* vd) : LValue(DeclValKind, vd) {}
    
    VarDecl* getDecl() const {
      return static_cast<VarDecl*>(Data);
    }
    
    inline bool operator==(const DeclVal& R) const {
      return getDecl() == R.getDecl();
    }
    
    inline bool operator!=(const DeclVal& R) const {
      return getDecl() != R.getDecl();
    }
    
    // Implement isa<T> support.
    static inline bool classof(const RValue* V) {
      return isa<LValue>(V) && V->getSubKind() == DeclValKind;
    }
    
    static inline bool classof(const LValue* V) {
      return V->getSubKind() == DeclValKind;
    }
  };
  
  class FuncVal : public LValue {
  public:
    FuncVal(const FunctionDecl* fd) : LValue(FuncValKind, fd) {}
    
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
    static inline bool classof(const RValue* V) {
      return isa<LValue>(V) && V->getSubKind() == FuncValKind;
    }
    
    static inline bool classof(const LValue* V) {
      return V->getSubKind() == FuncValKind;
    }
  };

  class ConcreteInt : public LValue {
  public:
    ConcreteInt(const llvm::APSInt& V) : LValue(ConcreteIntKind, &V) {}
    
    const llvm::APSInt& getValue() const {
      return *static_cast<llvm::APSInt*>(Data);
    }
    

    // Transfer functions for binary/unary operations on ConcreteInts.
    ConcreteInt EvalBinaryOp(ValueManager& ValMgr,
                             BinaryOperator::Opcode Op,
                             const ConcreteInt& RHS) const;
        
    // Implement isa<T> support.
    static inline bool classof(const RValue* V) {
      return isa<LValue>(V) && V->getSubKind() == ConcreteIntKind;
    }
    
    static inline bool classof(const LValue* V) {
      return V->getSubKind() == ConcreteIntKind;
    }
    
  };  
} // end clang::lval namespace

} // end clang namespace  

#endif
