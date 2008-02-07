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

// FIXME: reduce the number of includes.

#include "clang/Analysis/PathSensitive/GREngine.h"
#include "clang/AST/Expr.h"
#include "clang/AST/Decl.h"
#include "clang/AST/ASTContext.h"
#include "clang/Analysis/Analyses/LiveVariables.h"

#include "llvm/Support/Casting.h"
#include "llvm/Support/DataTypes.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/FoldingSet.h"
#include "llvm/ADT/ImmutableMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Streams.h"

#include <functional>

//==------------------------------------------------------------------------==//
//  Values and ValueManager.
//==------------------------------------------------------------------------==// 

namespace clang {

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
  
class SymbolData {
  uintptr_t Data;
public:
  enum Kind { ParmKind = 0x0, Mask = 0x3 };
  
  SymbolData(ParmVarDecl* D)
  : Data(reinterpret_cast<uintptr_t>(D) | ParmKind) {}
  
  inline Kind getKind() const { return (Kind) (Data & Mask); }
  inline void* getPtr() const { return reinterpret_cast<void*>(Data & ~Mask); }  
  inline bool operator==(const SymbolData& R) const { return Data == R.Data; }
  
  QualType getType() const;  
};
  

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
  SymbolID getSymbol() const { return Symbol; }
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
  
public:
  SymbolManager();
  ~SymbolManager();
  
  SymbolID getSymbol(ParmVarDecl* D);
  
  inline SymbolData getSymbolData(SymbolID ID) const {
    assert (ID < SymbolToData.size());
    return SymbolToData[ID];
  }
  
  inline QualType getType(SymbolID ID) const {
    return getSymbolData(ID).getType();
  }
};
  

class ValueManager {
  typedef llvm::FoldingSet<llvm::FoldingSetNodeWrapper<llvm::APSInt> >
          APSIntSetTy;
  
  typedef llvm::FoldingSet<SymIntConstraint>
          SymIntCSetTy;
  
  
  ASTContext& Ctx;
  llvm::BumpPtrAllocator& BPAlloc;
  
  APSIntSetTy   APSIntSet;
  SymIntCSetTy  SymIntCSet;
  
public:
  ValueManager(ASTContext& ctx, llvm::BumpPtrAllocator& Alloc) 
    : Ctx(ctx), BPAlloc(Alloc) {}
  
  ~ValueManager();
  
  ASTContext& getContext() const { return Ctx; }  

  const llvm::APSInt& getValue(const llvm::APSInt& X);
  const llvm::APSInt& getValue(uint64_t X, unsigned BitWidth, bool isUnsigned);
  const llvm::APSInt& getValue(uint64_t X, QualType T,
                               SourceLocation Loc = SourceLocation());
  
  inline const llvm::APSInt& getZeroWithPtrWidth() {
    return getValue( 0,
                     Ctx.getTypeSize(Ctx.VoidPtrTy, SourceLocation()),
                     true );
  }
  
  inline const llvm::APSInt& getTruthValue(bool b) {
    return getValue( b ? 1 : 0,
                     Ctx.getTypeSize(Ctx.IntTy, SourceLocation()),
                     false );
    
  }
  
  const SymIntConstraint& getConstraint(SymbolID sym, BinaryOperator::Opcode Op,
                                        const llvm::APSInt& V);
};
  
} // end clang namespace

//==------------------------------------------------------------------------==//
// Casting machinery to get cast<> and dyn_cast<> working with SymbolData.
//==------------------------------------------------------------------------==//

namespace llvm {
  
  template<> inline bool
  isa<clang::ParmVarDecl,clang::SymbolData>(const clang::SymbolData& V) {
    return V.getKind() == clang::SymbolData::ParmKind;
  }
  
  template<> struct cast_retty_impl<clang::ParmVarDecl,clang::SymbolData> {
    typedef const clang::ParmVarDecl* ret_type;
  };
  
  template<> struct simplify_type<clang::SymbolData> {
    typedef void* SimpleType;
    static inline SimpleType getSimplifiedValue(const clang::SymbolData &V) {
      return V.getPtr();
    }
  };
  
} // end llvm namespace

//==------------------------------------------------------------------------==//
//  Base RValue types.
//==------------------------------------------------------------------------==// 

namespace clang {
  
class RValue {
public:
  enum BaseKind { LValueKind=0x0,
                  NonLValueKind=0x1,
                  UninitializedKind=0x2,
                  InvalidKind=0x3 };
  
  enum { BaseBits = 2, 
         BaseMask = 0x3 };
  
private:
  void* Data;
  unsigned Kind;
  
protected:
  RValue(const void* d, bool isLValue, unsigned ValKind)
  : Data(const_cast<void*>(d)),
    Kind((isLValue ? LValueKind : NonLValueKind) | (ValKind << BaseBits)) {}
  
  explicit RValue(BaseKind k)
    : Data(0), Kind(k) {}
  
  void* getRawPtr() const {
    return reinterpret_cast<void*>(Data);
  }
  
public:
  ~RValue() {};
  
  /// BufferTy - A temporary buffer to hold a set of RValues.
  typedef llvm::SmallVector<RValue,5> BufferTy;

  
  RValue EvalCast(ValueManager& ValMgr, Expr* CastExpr) const;
  
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
  
  
  inline bool isValid() const { return getRawKind() != InvalidKind; }
  inline bool isInvalid() const { return getRawKind() == InvalidKind; }
  
  void print(std::ostream& OS) const;
  void print() const;
  
  // Implement isa<T> support.
  static inline bool classof(const RValue*) { return true; }
};

class InvalidValue : public RValue {
public:
  InvalidValue() : RValue(InvalidKind) {}
  
  static inline bool classof(const RValue* V) {
    return V->getBaseKind() == InvalidKind;
  }  
};

class UninitializedValue : public RValue {
public:
  UninitializedValue() : RValue(UninitializedKind) {}
  
  static inline bool classof(const RValue* V) {
    return V->getBaseKind() == UninitializedKind;
  }  
};

class NonLValue : public RValue {
protected:
  NonLValue(unsigned SubKind, const void* d) : RValue(d, false, SubKind) {}
  
public:
  void print(std::ostream& Out) const;
  
  NonLValue EvalBinaryOp(ValueManager& ValMgr,
                               BinaryOperator::Opcode Op,
                               const NonLValue& RHS) const;

  RValue EvalCast(ValueManager& ValMgr, Expr* CastExpr) const;
  NonLValue EvalMinus(ValueManager& ValMgr, UnaryOperator* U) const;
  NonLValue EvalComplement(ValueManager& ValMgr) const;
  
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
//  void print(std::ostream& Out) const;

  RValue EvalBinaryOp(ValueManager& ValMgr, BinaryOperator::Opcode Op,
                            const LValue& RHS) const;
  
  RValue EvalCast(ValueManager& ValMgr, Expr* CastExpr) const;
  
  // Implement isa<T> support.
  static inline bool classof(const RValue* V) {
    return V->getBaseKind() == LValueKind;
  }
};
  
//==------------------------------------------------------------------------==//
//  Subclasses of NonLValue.
//==------------------------------------------------------------------------==// 

namespace nonlval {
  
  enum Kind { SymbolValKind,
              SymIntConstraintValKind,
              ConcreteIntKind,
              NumKind };

  class SymbolVal : public NonLValue {
  public:
    SymbolVal(unsigned SymID)
    : NonLValue(SymbolValKind,
                reinterpret_cast<void*>((uintptr_t) SymID)) {}
    
    SymbolID getSymbol() const {
      return (SymbolID) reinterpret_cast<uintptr_t>(getRawPtr());
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
      return *reinterpret_cast<SymIntConstraint*>(getRawPtr());
    }
    
    static inline bool classof(const RValue* V) {
      return isa<NonLValue>(V) && V->getSubKind() == SymIntConstraintValKind;
    }    
  };

  class ConcreteInt : public NonLValue {
  public:
    ConcreteInt(const llvm::APSInt& V) : NonLValue(ConcreteIntKind, &V) {}
    
    const llvm::APSInt& getValue() const {
      return *static_cast<llvm::APSInt*>(getRawPtr());
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
              DeclValKind,
              ConcreteIntKind,
              NumKind };
  
  class SymbolVal : public LValue {
  public:
    SymbolVal(unsigned SymID)
    : LValue(SymbolValKind, reinterpret_cast<void*>((uintptr_t) SymID)) {}
    
    SymbolID getSymbol() const {
      return (SymbolID) reinterpret_cast<uintptr_t>(getRawPtr());
    }
    
    static inline bool classof(const RValue* V) {
      return V->getSubKind() == SymbolValKind;
    }  
  };
  
  class DeclVal : public LValue {
  public:
    DeclVal(const ValueDecl* vd) : LValue(DeclValKind,vd) {}
    
    ValueDecl* getDecl() const {
      return static_cast<ValueDecl*>(getRawPtr());
    }
    
    inline bool operator==(const DeclVal& R) const {
      return getDecl() == R.getDecl();
    }
    
    inline bool operator!=(const DeclVal& R) const {
      return getDecl() != R.getDecl();
    }
    
    // Implement isa<T> support.
    static inline bool classof(const RValue* V) {
      return V->getSubKind() == DeclValKind;
    }
  };

  class ConcreteInt : public LValue {
  public:
    ConcreteInt(const llvm::APSInt& V) : LValue(ConcreteIntKind, &V) {}
    
    const llvm::APSInt& getValue() const {
      return *static_cast<llvm::APSInt*>(getRawPtr());
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
