//===-- GRConstants.cpp - Simple, Path-Sens. Constant Prop. ------*- C++ -*-==//
//   
//                     The LLValM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//               Constant Propagation via Graph Reachability
//
//  This files defines a simple analysis that performs path-sensitive
//  constant propagation within a function.  An example use of this analysis
//  is to perform simple checks for NULL dereferences.
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/PathSensitive/GREngine.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ASTContext.h"
#include "clang/Analysis/Analyses/LiveVariables.h"

#include "llvm/Support/Casting.h"
#include "llvm/Support/DataTypes.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/FoldingSet.h"
#include "llvm/ADT/ImmutableMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Streams.h"

#include <functional>

#ifndef NDEBUG
#include "llvm/Support/GraphWriter.h"
#include <sstream>
#endif

using namespace clang;
using llvm::dyn_cast;
using llvm::cast;
using llvm::APSInt;

//===----------------------------------------------------------------------===//
/// ValueKey - A variant smart pointer that wraps either a ValueDecl* or a
///  Stmt*.  Use cast<> or dyn_cast<> to get actual pointer type
//===----------------------------------------------------------------------===//
namespace {

typedef unsigned SymbolID;
  
class VISIBILITY_HIDDEN ValueKey {
  uintptr_t Raw;  
  void operator=(const ValueKey& RHS); // Do not implement.
  
public:
  enum  Kind { IsSubExpr=0x0, IsBlkExpr=0x1, IsDecl=0x2, // L-Value Bindings.
               IsSymbol=0x3, // Symbol Bindings.
               Flags=0x3 };
  
  inline Kind getKind() const {
    return (Kind) (Raw & Flags);
  }
  
  inline void* getPtr() const { 
    assert (getKind() != IsSymbol);
    return reinterpret_cast<void*>(Raw & ~Flags);
  }
  
  inline SymbolID getSymbolID() const {
    assert (getKind() == IsSymbol);
    return (SymbolID) (Raw >> 2);
  }
  
  ValueKey(const ValueDecl* VD)
    : Raw(reinterpret_cast<uintptr_t>(VD) | IsDecl) {
      assert(VD && "ValueDecl cannot be NULL.");
    }

  ValueKey(Stmt* S, bool isBlkExpr = false) 
    : Raw(reinterpret_cast<uintptr_t>(S) | (isBlkExpr ? IsBlkExpr : IsSubExpr)){
      assert(S && "Tracked statement cannot be NULL.");
    }
  
  ValueKey(SymbolID V)
    : Raw((V << 2) | IsSymbol) {}  
  
  bool isSymbol()  const { return getKind() == IsSymbol; }
  bool isSubExpr() const { return getKind() == IsSubExpr; }
  bool isDecl()    const { return getKind() == IsDecl; }
  
  inline void Profile(llvm::FoldingSetNodeID& ID) const {
    ID.AddInteger(isSymbol() ? 1 : 0);

    if (isSymbol())
      ID.AddInteger(getSymbolID());
    else    
      ID.AddPointer(getPtr());
  }
  
  inline bool operator==(const ValueKey& X) const {
    return isSymbol() ? getSymbolID() == X.getSymbolID()
                      : getPtr() == X.getPtr();
  }
  
  inline bool operator!=(const ValueKey& X) const {
    return !operator==(X);
  }
  
  inline bool operator<(const ValueKey& X) const { 
    if (isSymbol())
      return X.isSymbol() ? getSymbolID() < X.getSymbolID() : false;
    
    return getPtr() < X.getPtr();
  }
};
} // end anonymous namespace

// Machinery to get cast<> and dyn_cast<> working with ValueKey.
namespace llvm {
  template<> inline bool isa<ValueDecl,ValueKey>(const ValueKey& V) {
    return V.getKind() == ValueKey::IsDecl;
  }
  template<> inline bool isa<Stmt,ValueKey>(const ValueKey& V) {
    return ((unsigned) V.getKind()) < ValueKey::IsDecl;
  }
  template<> struct VISIBILITY_HIDDEN cast_retty_impl<ValueDecl,ValueKey> {
    typedef const ValueDecl* ret_type;
  };
  template<> struct VISIBILITY_HIDDEN cast_retty_impl<Stmt,ValueKey> {
    typedef const Stmt* ret_type;
  };
  template<> struct VISIBILITY_HIDDEN simplify_type<ValueKey> {
    typedef void* SimpleType;
    static inline SimpleType getSimplifiedValue(const ValueKey &V) {
      return V.getPtr();
    }
  };
} // end llvm namespace

//===----------------------------------------------------------------------===//
// ValueManager.
//===----------------------------------------------------------------------===//

namespace {
  
typedef llvm::ImmutableSet<APSInt > APSIntSetTy;

  
class VISIBILITY_HIDDEN ValueManager {
  ASTContext* Ctx;
  
  typedef  llvm::FoldingSet<llvm::FoldingSetNodeWrapper<APSInt> > APSIntSetTy;
  APSIntSetTy APSIntSet;
  
  llvm::BumpPtrAllocator BPAlloc;
  
public:
  ValueManager() {}
  ~ValueManager();
  
  void setContext(ASTContext* ctx) { Ctx = ctx; }
  ASTContext* getContext() const { return Ctx; }
  
  APSInt& getValue(const APSInt& X);      

};
} // end anonymous namespace

ValueManager::~ValueManager() {
  // Note that the dstor for the contents of APSIntSet will never be called,
  // so we iterate over the set and invoke the dstor for each APSInt.  This
  // frees an aux. memory allocated to represent very large constants.
  for (APSIntSetTy::iterator I=APSIntSet.begin(), E=APSIntSet.end(); I!=E; ++I)
    I->getValue().~APSInt();
}
  
  

APSInt& ValueManager::getValue(const APSInt& X) {
  llvm::FoldingSetNodeID ID;
  void* InsertPos;
  typedef llvm::FoldingSetNodeWrapper<APSInt> FoldNodeTy;
  
  X.Profile(ID);
  FoldNodeTy* P = APSIntSet.FindNodeOrInsertPos(ID, InsertPos);
  
  if (!P) {  
    P = (FoldNodeTy*) BPAlloc.Allocate<FoldNodeTy>();
    new (P) FoldNodeTy(X);
    APSIntSet.InsertNode(P, InsertPos);
  }
  
  return *P;
}

//===----------------------------------------------------------------------===//
// Expression Values.
//===----------------------------------------------------------------------===//

namespace {
  
class VISIBILITY_HIDDEN RValue {
public:
  enum BaseKind { InvalidKind=0x0, LValueKind=0x1, NonLValueKind=0x2,
                  BaseFlags = 0x3 };
    
private:
  void* Data;
  unsigned Kind;
    
protected:
  RValue(const void* d, bool isLValue, unsigned ValKind)
    : Data(const_cast<void*>(d)),
      Kind((isLValue ? LValueKind : NonLValueKind) | (ValKind << 2)) {}
  
  explicit RValue()
    : Data(0), Kind(InvalidKind) {}

  void* getRawPtr() const {
    return reinterpret_cast<void*>(Data);
  }
  
public:
  ~RValue() {};

  RValue Cast(ValueManager& ValMgr, Expr* CastExpr) const;
  
  unsigned getRawKind() const { return Kind; }
  BaseKind getBaseKind() const { return (BaseKind) (Kind & 0x3); }
  unsigned getSubKind() const { return (Kind & ~0x3) >> 2; }   
  
  void Profile(llvm::FoldingSetNodeID& ID) const {
    ID.AddInteger((unsigned) getRawKind());
    ID.AddPointer(reinterpret_cast<void*>(Data));
  }
  
  bool operator==(const RValue& RHS) const {
    return getRawKind() == RHS.getRawKind() && Data == RHS.Data;
  }

  inline bool isValid() const { return getRawKind() != InvalidKind; }
  inline bool isInvalid() const { return getRawKind() == InvalidKind; }
  
  void print(std::ostream& OS) const;
  void print() const { print(*llvm::cerr.stream()); }

  // Implement isa<T> support.
  static inline bool classof(const RValue*) { return true; }
};

class VISIBILITY_HIDDEN InvalidValue : public RValue {
public:
  InvalidValue() {}
  
  static inline bool classof(const RValue* V) {
    return V->getBaseKind() == InvalidKind;
  }  
};

class VISIBILITY_HIDDEN LValue : public RValue {
protected:
  LValue(unsigned SubKind, void* D) : RValue(D, true, SubKind) {}
  
public:  
  // Implement isa<T> support.
  static inline bool classof(const RValue* V) {
    return V->getBaseKind() == LValueKind;
  }
};

class VISIBILITY_HIDDEN NonLValue : public RValue {
protected:
  NonLValue(unsigned SubKind, const void* d) : RValue(d, false, SubKind) {}
  
public:
  void print(std::ostream& Out) const;
  
  NonLValue Add(ValueManager& ValMgr, const NonLValue& RHS) const;
  NonLValue Sub(ValueManager& ValMgr, const NonLValue& RHS) const;
  NonLValue Mul(ValueManager& ValMgr, const NonLValue& RHS) const;
  NonLValue Div(ValueManager& ValMgr, const NonLValue& RHS) const;
  NonLValue Rem(ValueManager& ValMgr, const NonLValue& RHS) const;
  NonLValue UnaryMinus(ValueManager& ValMgr, UnaryOperator* U) const;
  
  static NonLValue GetValue(ValueManager& ValMgr, const APSInt& V);
  static NonLValue GetValue(ValueManager& ValMgr, IntegerLiteral* I);
  
  // Implement isa<T> support.
  static inline bool classof(const RValue* V) {
    return V->getBaseKind() == NonLValueKind;
  }
};
    
} // end anonymous namespace

//===----------------------------------------------------------------------===//
// LValues.
//===----------------------------------------------------------------------===//

namespace {
  
enum { LValueDeclKind, NumLValueKind };

class VISIBILITY_HIDDEN LValueDecl : public LValue {
public:
  LValueDecl(const ValueDecl* vd) 
  : LValue(LValueDeclKind,const_cast<ValueDecl*>(vd)) {}
  
  ValueDecl* getDecl() const {
    return static_cast<ValueDecl*>(getRawPtr());
  }
  
  // Implement isa<T> support.
  static inline bool classof(const RValue* V) {
    return V->getSubKind() == LValueDeclKind;
  }
};
  
} // end anonymous namespace

//===----------------------------------------------------------------------===//
// Non-LValues.
//===----------------------------------------------------------------------===//

namespace {
  
enum { SymbolicNonLValueKind, ConcreteIntKind, ConstrainedIntegerKind,
       NumNonLValueKind };

class VISIBILITY_HIDDEN SymbolicNonLValue : public NonLValue {
public:
  SymbolicNonLValue(unsigned SymID)
    : NonLValue(SymbolicNonLValueKind,
                reinterpret_cast<void*>((uintptr_t) SymID)) {}
  
  SymbolID getSymbolID() const {
    return (SymbolID) reinterpret_cast<uintptr_t>(getRawPtr());
  }
  
  static inline bool classof(const RValue* V) {
    return V->getSubKind() == SymbolicNonLValueKind;
  }  
};
  
class VISIBILITY_HIDDEN ConcreteInt : public NonLValue {
public:
  ConcreteInt(const APSInt& V) : NonLValue(ConcreteIntKind, &V) {}
  
  const APSInt& getValue() const {
    return *static_cast<APSInt*>(getRawPtr());
  }

  ConcreteInt Add(ValueManager& ValMgr, const ConcreteInt& V) const {
    return ValMgr.getValue(getValue() + V.getValue());
  }

  ConcreteInt Sub(ValueManager& ValMgr, const ConcreteInt& V) const {
    return ValMgr.getValue(getValue() - V.getValue());
  }
  
  ConcreteInt Mul(ValueManager& ValMgr, const ConcreteInt& V) const {
    return ValMgr.getValue(getValue() * V.getValue());
  }
  
  ConcreteInt Div(ValueManager& ValMgr, const ConcreteInt& V) const {
    return ValMgr.getValue(getValue() / V.getValue());
  }
  
  ConcreteInt Rem(ValueManager& ValMgr, const ConcreteInt& V) const {
    return ValMgr.getValue(getValue() % V.getValue());
  }
  
  ConcreteInt Cast(ValueManager& ValMgr, Expr* CastExpr) const {
    assert (CastExpr->getType()->isIntegerType());
    
    APSInt X(getValue());  
    X.extOrTrunc(ValMgr.getContext()->getTypeSize(CastExpr->getType(),
                                                  CastExpr->getLocStart()));
    return ValMgr.getValue(X);
  }
  
  ConcreteInt UnaryMinus(ValueManager& ValMgr, UnaryOperator* U) const {
    assert (U->getType() == U->getSubExpr()->getType());  
    assert (U->getType()->isIntegerType());  
    return ValMgr.getValue(-getValue()); 
  }

  // Implement isa<T> support.
  static inline bool classof(const RValue* V) {
    return V->getSubKind() == ConcreteIntKind;
  }
};
  
} // end anonymous namespace

//===----------------------------------------------------------------------===//
// Transfer function dispatch.
//===----------------------------------------------------------------------===//

RValue RValue::Cast(ValueManager& ValMgr, Expr* CastExpr) const {
  switch (getSubKind()) {
    case ConcreteIntKind:
      return cast<ConcreteInt>(this)->Cast(ValMgr, CastExpr);
    default:
      return InvalidValue();
  }
}

NonLValue NonLValue::UnaryMinus(ValueManager& ValMgr, UnaryOperator* U) const {
  switch (getSubKind()) {
    case ConcreteIntKind:
      return cast<ConcreteInt>(this)->UnaryMinus(ValMgr, U);
    default:
      return cast<NonLValue>(InvalidValue());
  }
}

#define RVALUE_DISPATCH_CASE(k1,k2,Op)\
case (k1##Kind*NumNonLValueKind+k2##Kind):\
  return cast<k1>(*this).Op(ValMgr,cast<k2>(RHS));

#define RVALUE_DISPATCH(Op)\
switch (getSubKind()*NumNonLValueKind+RHS.getSubKind()){\
  RVALUE_DISPATCH_CASE(ConcreteInt,ConcreteInt,Op)\
  default:\
    assert (!isValid() || !RHS.isValid() && "Missing case.");\
    break;\
}\
return cast<NonLValue>(InvalidValue());

NonLValue NonLValue::Add(ValueManager& ValMgr, const NonLValue& RHS) const {
  RVALUE_DISPATCH(Add)
}

NonLValue NonLValue::Sub(ValueManager& ValMgr, const NonLValue& RHS) const {
  RVALUE_DISPATCH(Sub)
}

NonLValue NonLValue::Mul(ValueManager& ValMgr, const NonLValue& RHS) const {
  RVALUE_DISPATCH(Mul)
}

NonLValue NonLValue::Div(ValueManager& ValMgr, const NonLValue& RHS) const {
  RVALUE_DISPATCH(Div)
}

NonLValue NonLValue::Rem(ValueManager& ValMgr, const NonLValue& RHS) const {
  RVALUE_DISPATCH(Rem)
}


#undef RVALUE_DISPATCH_CASE
#undef RVALUE_DISPATCH

//===----------------------------------------------------------------------===//
// Utility methods for constructing RValues.
//===----------------------------------------------------------------------===//

NonLValue NonLValue::GetValue(ValueManager& ValMgr, const APSInt& V) {
  return ConcreteInt(ValMgr.getValue(V));
}

NonLValue NonLValue::GetValue(ValueManager& ValMgr, IntegerLiteral* I) {
  return ConcreteInt(ValMgr.getValue(APSInt(I->getValue(),
                                       I->getType()->isUnsignedIntegerType())));
}

//===----------------------------------------------------------------------===//
// Pretty-Printing.
//===----------------------------------------------------------------------===//

void RValue::print(std::ostream& Out) const {
  switch (getBaseKind()) {
    case InvalidKind:
      Out << "Invalid";
      break;
      
    case NonLValueKind:
      cast<NonLValue>(this)->print(Out);
      break;
      
    case LValueKind:
      assert (false && "FIXME: LValue printing not implemented.");  
      break;
      
    default:
      assert (false && "Invalid RValue.");
  }
}

void NonLValue::print(std::ostream& Out) const {
  switch (getSubKind()) {  
    case ConcreteIntKind:
      Out << cast<ConcreteInt>(this)->getValue().toString();
      break;
      
    default:
      assert (false && "Pretty-printed not implemented for this NonLValue.");
      break;
  }
}

//===----------------------------------------------------------------------===//
// ValueMapTy - A ImmutableMap type Stmt*/Decl*/Symbols to RValues.
//===----------------------------------------------------------------------===//

typedef llvm::ImmutableMap<ValueKey,RValue> ValueMapTy;

namespace clang {
  template<>
  struct VISIBILITY_HIDDEN GRTrait<ValueMapTy> {
    static inline void* toPtr(ValueMapTy M) {
      return reinterpret_cast<void*>(M.getRoot());
    }  
    static inline ValueMapTy toState(void* P) {
      return ValueMapTy(static_cast<ValueMapTy::TreeTy*>(P));
    }
  };
}

//===----------------------------------------------------------------------===//
// The Checker.
//===----------------------------------------------------------------------===//

namespace {
  
class VISIBILITY_HIDDEN GRConstants {
    
public:
  typedef ValueMapTy StateTy;
  typedef GRNodeBuilder<GRConstants> NodeBuilder;
  typedef ExplodedNode<StateTy> NodeTy;
  
  class NodeSet {
    typedef llvm::SmallVector<NodeTy*,3> ImplTy;
    ImplTy Impl;
  public:
    
    NodeSet() {}
    NodeSet(NodeTy* N) { assert (N && !N->isInfeasible()); Impl.push_back(N); }
    
    void Add(NodeTy* N) { if (N && !N->isInfeasible()) Impl.push_back(N); }
    
    typedef ImplTy::iterator       iterator;
    typedef ImplTy::const_iterator const_iterator;
        
    unsigned size() const { return Impl.size(); }
    bool empty() const { return Impl.empty(); }
    
    iterator begin() { return Impl.begin(); }
    iterator end()   { return Impl.end(); }

    const_iterator begin() const { return Impl.begin(); }
    const_iterator end() const { return Impl.end(); }
  };
                                                              
protected:
  /// Liveness - live-variables information the ValueDecl* and block-level
  ///  Expr* in the CFG.  Used to prune out dead state.
  LiveVariables* Liveness;

  /// Builder - The current GRNodeBuilder which is used when building the nodes
  ///  for a given statement.
  NodeBuilder* Builder;

  /// StateMgr - Object that manages the data for all created states.
  ValueMapTy::Factory StateMgr;
  
  /// ValueMgr - Object that manages the data for all created RValues.
  ValueManager ValMgr;
  
  /// cfg - the current CFG.
  CFG* cfg;
  
  /// StmtEntryNode - The immediate predecessor node.
  NodeTy* StmtEntryNode;
  
  /// CurrentStmt - The current block-level statement.
  Stmt* CurrentStmt;
  
  bool StateCleaned;
  
  ASTContext* getContext() const { return ValMgr.getContext(); }
  
public:
  GRConstants() : Liveness(NULL), Builder(NULL), cfg(NULL),
    StmtEntryNode(NULL), CurrentStmt(NULL) {}
    
  ~GRConstants() { delete Liveness; }
  
  /// getCFG - Returns the CFG associated with this analysis.
  CFG& getCFG() { assert (cfg); return *cfg; }
      
  /// Initialize - Initialize the checker's state based on the specified
  ///  CFG.  This results in liveness information being computed for
  ///  each block-level statement in the CFG.
  void Initialize(CFG& c, ASTContext& ctx) {
    cfg = &c;
    ValMgr.setContext(&ctx);
    Liveness = new LiveVariables(c);
    Liveness->runOnCFG(c);
    Liveness->runOnAllBlocks(c, NULL, true);
  }
  
  /// getInitialState - Return the initial state used for the root vertex
  ///  in the ExplodedGraph.
  StateTy getInitialState() {
    return StateMgr.GetEmptyMap();
  }

  /// ProcessStmt - Called by GREngine. Used to generate new successor
  ///  nodes by processing the 'effects' of a block-level statement.
  void ProcessStmt(Stmt* S, NodeBuilder& builder);    

  /// RemoveDeadBindings - Return a new state that is the same as 'M' except
  ///  that all subexpression mappings are removed and that any
  ///  block-level expressions that are not live at 'S' also have their
  ///  mappings removed.
  StateTy RemoveDeadBindings(Stmt* S, StateTy M);

  StateTy SetValue(StateTy St, Stmt* S, const RValue& V);  

  StateTy SetValue(StateTy St, const Stmt* S, const RValue& V) {
    return SetValue(St, const_cast<Stmt*>(S), V);
  }
  
  StateTy SetValue(StateTy St, const LValue& LV, const RValue& V);
  
  RValue GetValue(const StateTy& St, Stmt* S);  
  inline RValue GetValue(const StateTy& St, const Stmt* S) {
    return GetValue(St, const_cast<Stmt*>(S));
  }
  
  RValue GetValue(const StateTy& St, const LValue& LV);
  LValue GetLValue(const StateTy& St, Stmt* S);
  
  void Nodify(NodeSet& Dst, Stmt* S, NodeTy* Pred, StateTy St);
  
  /// Visit - Transfer function logic for all statements.  Dispatches to
  ///  other functions that handle specific kinds of statements.
  void Visit(Stmt* S, NodeTy* Pred, NodeSet& Dst);

  /// VisitCast - Transfer function logic for all casts (implicit and explicit).
  void VisitCast(Expr* CastE, Expr* E, NodeTy* Pred, NodeSet& Dst);
  
  /// VisitUnaryOperator - Transfer function logic for unary operators.
  void VisitUnaryOperator(UnaryOperator* B, NodeTy* Pred, NodeSet& Dst);
  
  /// VisitBinaryOperator - Transfer function logic for binary operators.
  void VisitBinaryOperator(BinaryOperator* B, NodeTy* Pred, NodeSet& Dst);
  
  /// VisitDeclStmt - Transfer function logic for DeclStmts.
  void VisitDeclStmt(DeclStmt* DS, NodeTy* Pred, NodeSet& Dst);
};
} // end anonymous namespace


void GRConstants::ProcessStmt(Stmt* S, NodeBuilder& builder) {
  Builder = &builder;

  StmtEntryNode = builder.getLastNode();
  CurrentStmt = S;
  NodeSet Dst;
  StateCleaned = false;

  Visit(S, StmtEntryNode, Dst);

  // If no nodes were generated, generate a new node that has all the
  // dead mappings removed.
  if (Dst.size() == 1 && *Dst.begin() == StmtEntryNode) {
    StateTy St = RemoveDeadBindings(S, StmtEntryNode->getState());
    builder.generateNode(S, St, StmtEntryNode);
  }
  
  CurrentStmt = NULL;
  StmtEntryNode = NULL;
  Builder = NULL;
}


RValue GRConstants::GetValue(const StateTy& St, const LValue& LV) {
  switch (LV.getSubKind()) {
    case LValueDeclKind: {
      StateTy::TreeTy* T = St.SlimFind(cast<LValueDecl>(LV).getDecl()); 
      return T ? T->getValue().second : InvalidValue();
    }
    default:
      assert (false && "Invalid LValue.");
      break;
  }
  
  return InvalidValue();
}
  
RValue GRConstants::GetValue(const StateTy& St, Stmt* S) {
  for (;;) {
    switch (S->getStmtClass()) {
        
      // ParenExprs are no-ops.
        
      case Stmt::ParenExprClass:
        S = cast<ParenExpr>(S)->getSubExpr();
        continue;
        
      // DeclRefExprs can either evaluate to an LValue or a Non-LValue
      // (assuming an implicit "load") depending on the context.  In this
      // context we assume that we are retrieving the value contained
      // within the referenced variables.
        
      case Stmt::DeclRefExprClass:
        return GetValue(St, LValueDecl(cast<DeclRefExpr>(S)->getDecl()));

      // Integer literals evaluate to an RValue.  Simply retrieve the
      // RValue for the literal.
        
      case Stmt::IntegerLiteralClass:
        return NonLValue::GetValue(ValMgr, cast<IntegerLiteral>(S));
              
      // Casts where the source and target type are the same
      // are no-ops.  We blast through these to get the descendant
      // subexpression that has a value.
        
      case Stmt::ImplicitCastExprClass: {
        ImplicitCastExpr* C = cast<ImplicitCastExpr>(S);
        if (C->getType() == C->getSubExpr()->getType()) {
          S = C->getSubExpr();
          continue;
        }
        break;
      }

      case Stmt::CastExprClass: {
        CastExpr* C = cast<CastExpr>(S);
        if (C->getType() == C->getSubExpr()->getType()) {
          S = C->getSubExpr();
          continue;
        }
        break;
      }

      // Handle all other Stmt* using a lookup.

      default:
        break;
    };
    
    break;
  }
  
  StateTy::TreeTy* T = St.SlimFind(S);
    
  return T ? T->getValue().second : InvalidValue();
}

LValue GRConstants::GetLValue(const StateTy& St, Stmt* S) {
  while (ParenExpr* P = dyn_cast<ParenExpr>(S))
    S = P->getSubExpr();
  
  if (DeclRefExpr* DR = dyn_cast<DeclRefExpr>(S))
    return LValueDecl(DR->getDecl());
    
  return cast<LValue>(GetValue(St, S));
}


GRConstants::StateTy GRConstants::SetValue(StateTy St, Stmt* S,
                                           const RValue& V) {
  assert (S);
  
  if (!StateCleaned) {
    St = RemoveDeadBindings(CurrentStmt, St);
    StateCleaned = true;
  }
  
  bool isBlkExpr = false;
  
  if (S == CurrentStmt) {
    isBlkExpr = getCFG().isBlkExpr(S);

    if (!isBlkExpr)
      return St;
  }
  
  return V.isValid() ? StateMgr.Add(St, ValueKey(S,isBlkExpr), V)
                     : St;
}

GRConstants::StateTy GRConstants::SetValue(StateTy St, const LValue& LV,
                                           const RValue& V) {
  if (!LV.isValid())
    return St;
  
  if (!StateCleaned) {
    St = RemoveDeadBindings(CurrentStmt, St);
    StateCleaned = true;
  }

  switch (LV.getSubKind()) {
    case LValueDeclKind:        
      return V.isValid() ? StateMgr.Add(St, cast<LValueDecl>(LV).getDecl(), V)
                         : StateMgr.Remove(St, cast<LValueDecl>(LV).getDecl());
      
    default:
      assert ("SetValue for given LValue type not yet implemented.");
      return St;
  }
}

GRConstants::StateTy GRConstants::RemoveDeadBindings(Stmt* Loc, StateTy M) {
  // Note: in the code below, we can assign a new map to M since the
  //  iterators are iterating over the tree of the *original* map.
  StateTy::iterator I = M.begin(), E = M.end();

  // Remove old bindings for subexpressions and "dead" block-level expressions.
  for (; I!=E && !I.getKey().isDecl(); ++I) {
    if (I.getKey().isSubExpr() || !Liveness->isLive(Loc,cast<Stmt>(I.getKey())))
      M = StateMgr.Remove(M, I.getKey());
  }

  // Remove bindings for "dead" decls.
  for (; I!=E && I.getKey().isDecl(); ++I)
    if (VarDecl* V = dyn_cast<VarDecl>(cast<ValueDecl>(I.getKey())))
      if (!Liveness->isLive(Loc, V))
        M = StateMgr.Remove(M, I.getKey());

  return M;
}

void GRConstants::Nodify(NodeSet& Dst, Stmt* S, GRConstants::NodeTy* Pred, 
                         GRConstants::StateTy St) {
 
  // If the state hasn't changed, don't generate a new node.
  if (St == Pred->getState())
    return;
  
  Dst.Add(Builder->generateNode(S, St, Pred));
}

void GRConstants::VisitCast(Expr* CastE, Expr* E, GRConstants::NodeTy* Pred,
                            GRConstants::NodeSet& Dst) {
  
  QualType T = CastE->getType();

  // Check for redundant casts.
  if (E->getType() == T) {
    Dst.Add(Pred);
    return;
  }
  
  NodeSet S1;
  Visit(E, Pred, S1);
  
  for (NodeSet::iterator I1=S1.begin(), E1=S1.end(); I1 != E1; ++I1) {
    NodeTy* N = *I1;
    StateTy St = N->getState();
    const RValue& V = GetValue(St, E);
    Nodify(Dst, CastE, N, SetValue(St, CastE, V.Cast(ValMgr, CastE)));
  }
}

void GRConstants::VisitDeclStmt(DeclStmt* DS, GRConstants::NodeTy* Pred,
                                GRConstants::NodeSet& Dst) {
  
  StateTy St = Pred->getState();
  
  for (const ScopedDecl* D = DS->getDecl(); D; D = D->getNextDeclarator())
    if (const VarDecl* VD = dyn_cast<VarDecl>(D))
      St = SetValue(St, LValueDecl(VD), GetValue(St, VD->getInit()));

  Nodify(Dst, DS, Pred, St);
  
  if (Dst.empty())
    Dst.Add(Pred);  
}

void GRConstants::VisitUnaryOperator(UnaryOperator* U,
                                     GRConstants::NodeTy* Pred,
                                     GRConstants::NodeSet& Dst) {
  NodeSet S1;
  Visit(U->getSubExpr(), Pred, S1);
    
  for (NodeSet::iterator I1=S1.begin(), E1=S1.end(); I1 != E1; ++I1) {
    NodeTy* N1 = *I1;
    StateTy St = N1->getState();
    
    switch (U->getOpcode()) {
      case UnaryOperator::PostInc: {
        const LValue& L1 = GetLValue(St, U->getSubExpr());
        NonLValue R1 = cast<NonLValue>(GetValue(St, L1));

        QualType T = U->getType();
        unsigned bits = getContext()->getTypeSize(T, U->getLocStart());
        APSInt One(llvm::APInt(bits, 1), T->isUnsignedIntegerType());
        NonLValue R2 = NonLValue::GetValue(ValMgr, One);
        
        NonLValue Result = R1.Add(ValMgr, R2);
        Nodify(Dst, U, N1, SetValue(SetValue(St, U, R1), L1, Result));
        break;
      }
        
      case UnaryOperator::PostDec: {
        const LValue& L1 = GetLValue(St, U->getSubExpr());
        NonLValue R1 = cast<NonLValue>(GetValue(St, L1));
        
        QualType T = U->getType();
        unsigned bits = getContext()->getTypeSize(T, U->getLocStart());
        APSInt One(llvm::APInt(bits, 1), T->isUnsignedIntegerType());
        NonLValue R2 = NonLValue::GetValue(ValMgr, One);
        
        NonLValue Result = R1.Sub(ValMgr, R2);
        Nodify(Dst, U, N1, SetValue(SetValue(St, U, R1), L1, Result));
        break;
      }
        
      case UnaryOperator::PreInc: {
        const LValue& L1 = GetLValue(St, U->getSubExpr());
        NonLValue R1 = cast<NonLValue>(GetValue(St, L1));
        
        QualType T = U->getType();
        unsigned bits = getContext()->getTypeSize(T, U->getLocStart());
        APSInt One(llvm::APInt(bits, 1), T->isUnsignedIntegerType());
        NonLValue R2 = NonLValue::GetValue(ValMgr, One);        
        
        NonLValue Result = R1.Add(ValMgr, R2);
        Nodify(Dst, U, N1, SetValue(SetValue(St, U, Result), L1, Result));
        break;
      }
        
      case UnaryOperator::PreDec: {
        const LValue& L1 = GetLValue(St, U->getSubExpr());
        NonLValue R1 = cast<NonLValue>(GetValue(St, L1));
        
        QualType T = U->getType();
        unsigned bits = getContext()->getTypeSize(T, U->getLocStart());
        APSInt One(llvm::APInt(bits, 1), T->isUnsignedIntegerType());
        NonLValue R2 = NonLValue::GetValue(ValMgr, One);       
        
        NonLValue Result = R1.Sub(ValMgr, R2);
        Nodify(Dst, U, N1, SetValue(SetValue(St, U, Result), L1, Result));
        break;
      }
        
      case UnaryOperator::Minus: {
        const NonLValue& R1 = cast<NonLValue>(GetValue(St, U->getSubExpr()));
        Nodify(Dst, U, N1, SetValue(St, U, R1.UnaryMinus(ValMgr, U)));
        break;
      }
        
      default: ;
        assert (false && "Not implemented.");
    }    
  }
}

void GRConstants::VisitBinaryOperator(BinaryOperator* B,
                                      GRConstants::NodeTy* Pred,
                                      GRConstants::NodeSet& Dst) {
  NodeSet S1;
  Visit(B->getLHS(), Pred, S1);

  for (NodeSet::iterator I1=S1.begin(), E1=S1.end(); I1 != E1; ++I1) {
    NodeTy* N1 = *I1;
    
    // When getting the value for the LHS, check if we are in an assignment.
    // In such cases, we want to (initially) treat the LHS as an LValue,
    // so we use GetLValue instead of GetValue so that DeclRefExpr's are
    // evaluated to LValueDecl's instead of to an NonLValue.
    const RValue& V1 = 
      B->isAssignmentOp() ? GetLValue(N1->getState(), B->getLHS())
                          : GetValue(N1->getState(), B->getLHS());
    
    NodeSet S2;
    Visit(B->getRHS(), N1, S2);
  
    for (NodeSet::iterator I2=S2.begin(), E2=S2.end(); I2 != E2; ++I2) {
      NodeTy* N2 = *I2;
      StateTy St = N2->getState();
      const RValue& V2 = GetValue(St, B->getRHS());

      switch (B->getOpcode()) {
        case BinaryOperator::Add: {
          const NonLValue& R1 = cast<NonLValue>(V1);
          const NonLValue& R2 = cast<NonLValue>(V2);
          
          Nodify(Dst, B, N2, SetValue(St, B, R1.Add(ValMgr, R2)));
          break;
        }

        case BinaryOperator::Sub: {
          const NonLValue& R1 = cast<NonLValue>(V1);
          const NonLValue& R2 = cast<NonLValue>(V2);
	        Nodify(Dst, B, N2, SetValue(St, B, R1.Sub(ValMgr, R2)));
          break;
        }
          
        case BinaryOperator::Mul: {
          const NonLValue& R1 = cast<NonLValue>(V1);
          const NonLValue& R2 = cast<NonLValue>(V2);
	        Nodify(Dst, B, N2, SetValue(St, B, R1.Mul(ValMgr, R2)));
          break;
        }
          
        case BinaryOperator::Div: {
          const NonLValue& R1 = cast<NonLValue>(V1);
          const NonLValue& R2 = cast<NonLValue>(V2);
	        Nodify(Dst, B, N2, SetValue(St, B, R1.Div(ValMgr, R2)));
          break;
        }
          
        case BinaryOperator::Rem: {
          const NonLValue& R1 = cast<NonLValue>(V1);
          const NonLValue& R2 = cast<NonLValue>(V2);
	        Nodify(Dst, B, N2, SetValue(St, B, R1.Rem(ValMgr, R2)));
          break;
        }
          
        case BinaryOperator::Assign: {
          const LValue& L1 = cast<LValue>(V1);
          const NonLValue& R2 = cast<NonLValue>(V2);
          Nodify(Dst, B, N2, SetValue(SetValue(St, B, R2), L1, R2));
          break;
        }
          
        case BinaryOperator::AddAssign: {
          const LValue& L1 = cast<LValue>(V1);
          NonLValue R1 = cast<NonLValue>(GetValue(N1->getState(), L1));
          NonLValue Result = R1.Add(ValMgr, cast<NonLValue>(V2));
          Nodify(Dst, B, N2, SetValue(SetValue(St, B, Result), L1, Result));
          break;
        }
          
        case BinaryOperator::SubAssign: {
          const LValue& L1 = cast<LValue>(V1);
          NonLValue R1 = cast<NonLValue>(GetValue(N1->getState(), L1));
          NonLValue Result = R1.Sub(ValMgr, cast<NonLValue>(V2));
          Nodify(Dst, B, N2, SetValue(SetValue(St, B, Result), L1, Result));
          break;
        }
          
        case BinaryOperator::MulAssign: {
          const LValue& L1 = cast<LValue>(V1);
          NonLValue R1 = cast<NonLValue>(GetValue(N1->getState(), L1));
          NonLValue Result = R1.Mul(ValMgr, cast<NonLValue>(V2));
          Nodify(Dst, B, N2, SetValue(SetValue(St, B, Result), L1, Result));
          break;
        }
          
        case BinaryOperator::DivAssign: {
          const LValue& L1 = cast<LValue>(V1);
          NonLValue R1 = cast<NonLValue>(GetValue(N1->getState(), L1));
          NonLValue Result = R1.Div(ValMgr, cast<NonLValue>(V2));
          Nodify(Dst, B, N2, SetValue(SetValue(St, B, Result), L1, Result));
          break;
        }

        default: 
          Dst.Add(N2);
          break;
      }
    }
  }
}


void GRConstants::Visit(Stmt* S, GRConstants::NodeTy* Pred,
                        GRConstants::NodeSet& Dst) {

  // FIXME: add metadata to the CFG so that we can disable
  //  this check when we KNOW that there is no block-level subexpression.
  //  The motivation is that this check requires a hashtable lookup.

  if (S != CurrentStmt && getCFG().isBlkExpr(S)) {
    Dst.Add(Pred);
    return;
  }

  switch (S->getStmtClass()) {
    case Stmt::BinaryOperatorClass:
    case Stmt::CompoundAssignOperatorClass:
      VisitBinaryOperator(cast<BinaryOperator>(S), Pred, Dst);
      break;
      
    case Stmt::UnaryOperatorClass:
      VisitUnaryOperator(cast<UnaryOperator>(S), Pred, Dst);
      break;
      
    case Stmt::ParenExprClass:
      Visit(cast<ParenExpr>(S)->getSubExpr(), Pred, Dst);
      break;
      
    case Stmt::ImplicitCastExprClass: {
      ImplicitCastExpr* C = cast<ImplicitCastExpr>(S);
      VisitCast(C, C->getSubExpr(), Pred, Dst);
      break;
    }
      
    case Stmt::CastExprClass: {
      CastExpr* C = cast<CastExpr>(S);
      VisitCast(C, C->getSubExpr(), Pred, Dst);
      break;
    }
      
    case Stmt::DeclStmtClass:
      VisitDeclStmt(cast<DeclStmt>(S), Pred, Dst);
      break;
      
    default:
      Dst.Add(Pred); // No-op. Simply propagate the current state unchanged.
      break;
  }
}

//===----------------------------------------------------------------------===//
// Driver.
//===----------------------------------------------------------------------===//

#ifndef NDEBUG
namespace llvm {
template<>
struct VISIBILITY_HIDDEN DOTGraphTraits<GRConstants::NodeTy*> :
  public DefaultDOTGraphTraits {

  static void PrintKindLabel(std::ostream& Out, ValueKey::Kind kind) {
    switch (kind) {
      case ValueKey::IsSubExpr:  Out << "Sub-Expressions:\\l"; break;
      case ValueKey::IsDecl:    Out << "Variables:\\l"; break;
      case ValueKey::IsBlkExpr: Out << "Block-level Expressions:\\l"; break;
      default: assert (false && "Unknown ValueKey type.");
    }
  }
    
  static void PrintKind(std::ostream& Out, GRConstants::StateTy M,
                        ValueKey::Kind kind, bool isFirstGroup = false) {
    bool isFirst = true;
    
    for (GRConstants::StateTy::iterator I=M.begin(), E=M.end();I!=E;++I) {        
      if (I.getKey().getKind() != kind)
        continue;
    
      if (isFirst) {
        if (!isFirstGroup) Out << "\\l\\l";
        PrintKindLabel(Out, kind);
        isFirst = false;
      }
      else
        Out << "\\l";
      
      Out << ' ';
    
      if (ValueDecl* V = dyn_cast<ValueDecl>(I.getKey()))
        Out << V->getName();          
      else {
        Stmt* E = cast<Stmt>(I.getKey());
        Out << " (" << (void*) E << ") ";
        E->printPretty(Out);
      }
    
      Out << " : ";
      I.getData().print(Out);
    }
  }
    
  static std::string getNodeLabel(const GRConstants::NodeTy* N, void*) {
    std::ostringstream Out;

    // Program Location.
    ProgramPoint Loc = N->getLocation();
    
    switch (Loc.getKind()) {
      case ProgramPoint::BlockEntranceKind:
        Out << "Block Entrance: B" 
            << cast<BlockEntrance>(Loc).getBlock()->getBlockID();
        break;
      
      case ProgramPoint::BlockExitKind:
        assert (false);
        break;
        
      case ProgramPoint::PostStmtKind: {
        const PostStmt& L = cast<PostStmt>(Loc);
        Out << L.getStmt()->getStmtClassName() << ':' 
            << (void*) L.getStmt() << ' ';
        
        L.getStmt()->printPretty(Out);
        break;
      }
    
      default: {
        const BlockEdge& E = cast<BlockEdge>(Loc);
        Out << "Edge: (B" << E.getSrc()->getBlockID() << ", B"
            << E.getDst()->getBlockID()  << ')';
      }
    }
    
    Out << "\\|";
    
    PrintKind(Out, N->getState(), ValueKey::IsDecl, true);
    PrintKind(Out, N->getState(), ValueKey::IsBlkExpr);
    PrintKind(Out, N->getState(), ValueKey::IsSubExpr);
      
    Out << "\\l";
    return Out.str();
  }
};
} // end llvm namespace    
#endif

namespace clang {
void RunGRConstants(CFG& cfg, ASTContext& Ctx) {
  GREngine<GRConstants> Engine(cfg, Ctx);
  Engine.ExecuteWorkList();  
#ifndef NDEBUG
  llvm::ViewGraph(*Engine.getGraph().roots_begin(),"GRConstants");
#endif  
}
} // end clang namespace
