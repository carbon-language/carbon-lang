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

#ifndef NDEBUG
#include "llvm/Support/GraphWriter.h"
#include <sstream>
#endif

using namespace clang;
using llvm::dyn_cast;
using llvm::cast;

//===----------------------------------------------------------------------===//
/// ValueKey - A variant smart pointer that wraps either a ValueDecl* or a
///  Stmt*.  Use cast<> or dyn_cast<> to get actual pointer type
//===----------------------------------------------------------------------===//
namespace {
  
class VISIBILITY_HIDDEN ValueKey {
  uintptr_t Raw;
public:
  enum  Kind { IsSubExp=0x0, IsDecl=0x1, IsBlkExpr=0x2, Flags=0x3 };
  inline void* getPtr() const { return reinterpret_cast<void*>(Raw & ~Flags); }
  inline Kind getKind() const { return (Kind) (Raw & Flags); }
  
  ValueKey(const ValueDecl* VD)
    : Raw(reinterpret_cast<uintptr_t>(VD) | IsDecl) {}

  ValueKey(Stmt* S, bool isBlkExpr) 
    : Raw(reinterpret_cast<uintptr_t>(S) | (isBlkExpr ? IsBlkExpr : IsSubExp)){}
  
  bool isSubExpr() const { return getKind() == IsSubExp; }
  bool isDecl() const { return getKind() == IsDecl; }
  
  inline void Profile(llvm::FoldingSetNodeID& ID) const {
    ID.AddPointer(getPtr());
    ID.AddInteger((unsigned) getKind());
  }
  
  inline bool operator==(const ValueKey& X) const {
    return Raw == X.Raw;
  }
  
  inline bool operator!=(const ValueKey& X) const {
    return !operator==(X);
  }
  
  inline bool operator<(const ValueKey& X) const { 
    Kind k = getKind(), Xk = X.getKind();

    return k == Xk ? getPtr() < X.getPtr() 
                   : ((unsigned) k) < ((unsigned) Xk);
  }
};
} // end anonymous namespace

// Machinery to get cast<> and dyn_cast<> working with ValueKey.
namespace llvm {
  template<> inline bool isa<ValueDecl,ValueKey>(const ValueKey& V) {
    return V.getKind() == ValueKey::IsDecl;
  }
  template<> inline bool isa<Stmt,ValueKey>(const ValueKey& V) {
    return ((unsigned) V.getKind()) != ValueKey::IsDecl;
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
  
typedef llvm::ImmutableSet<llvm::APSInt > APSIntSetTy;
    
class VISIBILITY_HIDDEN ValueManager {
  
  
  APSIntSetTy::Factory APSIntSetFactory;
  
public:
  ValueManager() {}
  ~ValueManager() {}
  
  APSIntSetTy GetEmptyAPSIntSet() {
    return APSIntSetFactory.GetEmptySet();
  }
  
  APSIntSetTy AddToSet(const APSIntSetTy& Set, const llvm::APSInt& Val) {
    return APSIntSetFactory.Add(Set, Val);
  }
};
} // end anonymous namespace

//===----------------------------------------------------------------------===//
// Expression Values.
//===----------------------------------------------------------------------===//
  
namespace {
  
class VISIBILITY_HIDDEN ExprValue {
public:
  enum Kind { // L-Values.
              LValueDeclKind = 0x0,
              // Special "Invalid" value.
              InvalidValueKind = 0x1, 
              // R-Values.
              RValueMayEqualSetKind = 0x2,
              // Note that the Lvalue and RValue "kinds" overlap; 
              // the "InvalidValue" class can be used either as
              // an LValue or RValue.  
              MinLValueKind = 0x0, MaxLValueKind = 0x1,
              MinRValueKind = 0x1, MaxRValueKind = 0x2 };
  
private:
  enum Kind kind;
  void* Data;
    
protected:
  ExprValue(Kind k, void* d) : kind(k), Data(d) {}
  
  void* getRawPtr() const { return Data; }
  
public:
  ~ExprValue() {};

  void Profile(llvm::FoldingSetNodeID& ID) const {
    ID.AddInteger((unsigned) getKind());
    ID.AddPointer(Data);
  }
  
  bool operator==(const ExprValue& RHS) const {
    return kind == RHS.kind && Data == RHS.Data;
  }

  Kind getKind() const { return kind; }  
  bool isValid() const { return getKind() != InvalidValueKind; }
  
  void print(std::ostream& OS) const;
  void print() const { print(*llvm::cerr.stream()); }

  // Implement isa<T> support.
  static inline bool classof(const ExprValue*) { return true; }
};

class VISIBILITY_HIDDEN InvalidValue : public ExprValue {
public:
  InvalidValue() : ExprValue(InvalidValueKind, NULL) {}
  
  static inline bool classof(const ExprValue* V) {
    return V->getKind() == InvalidValueKind;
  }  
};

} // end anonymous namespace

//===----------------------------------------------------------------------===//
// "R-Values": Interface.
//===----------------------------------------------------------------------===//

namespace {
class VISIBILITY_HIDDEN RValue : public ExprValue {
protected:
  RValue(Kind k, void* d) : ExprValue(k,d) {}
  
public:
  RValue Add(ValueManager& ValMgr, const RValue& RHS) const;
  RValue Sub(ValueManager& ValMgr, const RValue& RHS) const;
  
  static RValue GetRValue(ValueManager& ValMgr, IntegerLiteral* S);
  
  // Implement isa<T> support.
  static inline bool classof(const ExprValue* V) {
    return V->getKind() >= MinRValueKind;
  }
};
  
class VISIBILITY_HIDDEN RValueMayEqualSet : public RValue {
public:
  RValueMayEqualSet(const APSIntSetTy& S)
    : RValue(RValueMayEqualSetKind, S.getRoot()) {}
  
  APSIntSetTy GetValues() const {
    return APSIntSetTy(reinterpret_cast<APSIntSetTy::TreeTy*>(getRawPtr()));
  }
  
  RValueMayEqualSet Add(ValueManager& ValMgr, const RValueMayEqualSet& V) const;
  RValueMayEqualSet Sub(ValueManager& ValMgr, const RValueMayEqualSet& V) const;  
  
  // Implement isa<T> support.
  static inline bool classof(const ExprValue* V) {
    return V->getKind() == RValueMayEqualSetKind;
  }
};
} // end anonymous namespace

//===----------------------------------------------------------------------===//
// "R-Values": Implementation.
//===----------------------------------------------------------------------===//

#define RVALUE_DISPATCH_CASE(k1,k2,Op)\
case ((k1##Kind+(MaxRValueKind-MinRValueKind))+(k2##Kind - MinRValueKind)):\
  return cast<k1>(*this).Op(ValMgr,cast<k2>(RHS));

#define RVALUE_DISPATCH(Op)\
switch (getKind()+(MaxRValueKind-MinRValueKind)+(RHS.getKind()-MinRValueKind)){\
  RVALUE_DISPATCH_CASE(RValueMayEqualSet,RValueMayEqualSet,Op)\
  default:\
    assert (!isValid() || !RHS.isValid() && "Missing case.");\
    break;\
}\
return cast<RValue>(InvalidValue());

RValue RValue::Add(ValueManager& ValMgr, const RValue& RHS) const {
  RVALUE_DISPATCH(Add)
}

RValue RValue::Sub(ValueManager& ValMgr, const RValue& RHS) const {
  RVALUE_DISPATCH(Sub)
}

#undef RVALUE_DISPATCH_CASE
#undef RVALUE_DISPATCH

RValueMayEqualSet
RValueMayEqualSet::Add(ValueManager& ValMgr,
                       const RValueMayEqualSet& RHS) const {
  
  APSIntSetTy S1 = GetValues();
  APSIntSetTy S2 = RHS.GetValues();
  
  APSIntSetTy M = ValMgr.GetEmptyAPSIntSet();
    
  for (APSIntSetTy::iterator I1=S1.begin(), E1=S2.end(); I1!=E1; ++I1)
    for (APSIntSetTy::iterator I2=S2.begin(), E2=S2.end(); I2!=E2; ++I2)
      M = ValMgr.AddToSet(M, *I1 + *I2);
  
  return M;
}

RValueMayEqualSet
RValueMayEqualSet::Sub(ValueManager& ValMgr,
                       const RValueMayEqualSet& RHS) const {
  
  APSIntSetTy S1 = GetValues();
  APSIntSetTy S2 = RHS.GetValues();
  
  APSIntSetTy M = ValMgr.GetEmptyAPSIntSet();
  
  for (APSIntSetTy::iterator I1=S1.begin(), E1=S2.end(); I1!=E1; ++I1)
    for (APSIntSetTy::iterator I2=S2.begin(), E2=S2.end(); I2!=E2; ++I2)
      M = ValMgr.AddToSet(M, *I1 - *I2);
  
  return M;
}

RValue RValue::GetRValue(ValueManager& ValMgr, IntegerLiteral* S) {
  return RValueMayEqualSet(ValMgr.AddToSet(ValMgr.GetEmptyAPSIntSet(),
                                           S->getValue()));
}    

//===----------------------------------------------------------------------===//
// "L-Values".
//===----------------------------------------------------------------------===//

namespace {
  
class VISIBILITY_HIDDEN LValue : public ExprValue {
protected:
  LValue(Kind k, void* D) : ExprValue(k, D) {}
  
public:  
  // Implement isa<T> support.
  static inline bool classof(const ExprValue* V) {
    return V->getKind() <= MaxLValueKind;
  }
};

class VISIBILITY_HIDDEN LValueDecl : public LValue {
public:
  LValueDecl(ValueDecl* vd) : LValue(LValueDeclKind,vd) {}
  
  ValueDecl* getDecl() const {
    return static_cast<ValueDecl*>(getRawPtr());
  }
  
  // Implement isa<T> support.
  static inline bool classof(const ExprValue* V) {
    return V->getKind() == LValueDeclKind;
  }
};  
} // end anonymous namespace

//===----------------------------------------------------------------------===//
// Pretty-Printing.
//===----------------------------------------------------------------------===//

void ExprValue::print(std::ostream& Out) const {
  switch (getKind()) {
    case InvalidValueKind:
        Out << "Invalid"; break;
      
    case RValueMayEqualSetKind: {
      APSIntSetTy S = cast<RValueMayEqualSet>(this)->GetValues();
      bool first = true;

      for (APSIntSetTy::iterator I=S.begin(), E=S.end(); I!=E; ++I) {
        if (first) first = false;
        else Out << " | ";
        
        Out << (*I).toString();
      }
      
      break;
    }
      
    default:
      assert (false && "Pretty-printed not implemented for this ExprValue.");
      break;
  }
}

//===----------------------------------------------------------------------===//
// ValueMapTy - A ImmutableMap type Stmt*/Decl* to ExprValues.
//===----------------------------------------------------------------------===//

typedef llvm::ImmutableMap<ValueKey,ExprValue> ValueMapTy;

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
// The Checker!
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
  
  /// ValueMgr - Object that manages the data for all created ExprValues.
  ValueManager ValMgr;
  
  /// cfg - the current CFG.
  CFG* cfg;
  
  /// StmtEntryNode - The immediate predecessor node.
  NodeTy* StmtEntryNode;
  
  /// CurrentStmt - The current block-level statement.
  Stmt* CurrentStmt;
  
  bool StateCleaned;
  
public:
  GRConstants() : Liveness(NULL), Builder(NULL), cfg(NULL),
    StmtEntryNode(NULL), CurrentStmt(NULL) {}
    
  ~GRConstants() { delete Liveness; }
  
  /// getCFG - Returns the CFG associated with this analysis.
  CFG& getCFG() { assert (cfg); return *cfg; }
      
  /// Initialize - Initialize the checker's state based on the specified
  ///  CFG.  This results in liveness information being computed for
  ///  each block-level statement in the CFG.
  void Initialize(CFG& c) {
    cfg = &c;    
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

  StateTy SetValue(StateTy St, Stmt* S, const ExprValue& V,
                   bool isBlkExpr = false);

  StateTy SetValue(StateTy St, const LValue& LV, const ExprValue& V);
  
  ExprValue GetValue(const StateTy& St, Stmt* S);
  ExprValue GetValue(const StateTy& St, const LValue& LV);
  LValue GetLValue(const StateTy& St, Stmt* S);
  
  void Nodify(NodeSet& Dst, Stmt* S, NodeTy* Pred, StateTy St);
  
  /// Visit - Transfer function logic for all statements.  Dispatches to
  ///  other functions that handle specific kinds of statements.
  void Visit(Stmt* S, NodeTy* Pred, NodeSet& Dst);
  
  /// VisitBinaryOperator - Transfer function logic for binary operators.
  void VisitBinaryOperator(BinaryOperator* B, NodeTy* Pred, NodeSet& Dst);  
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


ExprValue GRConstants::GetValue(const StateTy& St, const LValue& LV) {
  switch (LV.getKind()) {
    case ExprValue::LValueDeclKind: {
      StateTy::TreeTy* T = St.SlimFind(cast<LValueDecl>(LV).getDecl()); 
      return T ? T->getValue().second : InvalidValue();
    }
    default:
      assert (false && "Invalid LValue.");
      break;
  }
  
  return InvalidValue();
}
  
ExprValue GRConstants::GetValue(const StateTy& St, Stmt* S) {
  if (Expr* E = dyn_cast<Expr>(S))
    S = E->IgnoreParens();
  
  switch (S->getStmtClass()) {
    case Stmt::DeclRefExprClass:
      return GetValue(St, LValueDecl(cast<DeclRefExpr>(S)->getDecl()));

    case Stmt::IntegerLiteralClass:
      return RValue::GetRValue(ValMgr, cast<IntegerLiteral>(S));

    default:
      break;
  };
  
  StateTy::TreeTy* T = St.SlimFind(ValueKey(S, getCFG().isBlkExpr(S)));
  return T ? T->getValue().second : InvalidValue();
}

LValue GRConstants::GetLValue(const StateTy& St, Stmt* S) {
  if (Expr* E = dyn_cast<Expr>(S))
    S = E->IgnoreParens();
  
  if (DeclRefExpr* DR = dyn_cast<DeclRefExpr>(S))
    return LValueDecl(DR->getDecl());
    
  return cast<LValue>(GetValue(St, S));
}

GRConstants::StateTy GRConstants::SetValue(StateTy St, Stmt* S,
                                           const ExprValue& V, bool isBlkExpr) {
  
  if (!StateCleaned) {
    St = RemoveDeadBindings(CurrentStmt, St);
    StateCleaned = true;
  }
  
  return V.isValid() ? StateMgr.Add(St, ValueKey(S,isBlkExpr), V)
                     : St;
}

GRConstants::StateTy GRConstants::SetValue(StateTy St, const LValue& LV,
                                           const ExprValue& V) {
  if (!LV.isValid())
    return St;
  
  if (!StateCleaned) {
    St = RemoveDeadBindings(CurrentStmt, St);
    StateCleaned = true;
  }

  switch (LV.getKind()) {
    case ExprValue::LValueDeclKind:        
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

  // Remove old bindings for subexpressions.
  for (; I!=E && I.getKey().getKind() == ValueKey::IsSubExp; ++I)
    M = StateMgr.Remove(M, I.getKey());

  // Remove bindings for "dead" decls.
  for (; I!=E && I.getKey().getKind() == ValueKey::IsDecl; ++I)
    if (VarDecl* V = dyn_cast<VarDecl>(cast<ValueDecl>(I.getKey())))
      if (!Liveness->isLive(Loc, V))
        M = StateMgr.Remove(M, I.getKey());

  // Remove bindings for "dead" block-level expressions.
  for (; I!=E; ++I)
    if (!Liveness->isLive(Loc, cast<Stmt>(I.getKey())))
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
    // evaluated to LValueDecl's instead of to an RValue.
    const ExprValue& V1 = 
      B->isAssignmentOp() ? GetLValue(N1->getState(), B->getLHS())
                          : GetValue(N1->getState(), B->getLHS());
    
    NodeSet S2;
    Visit(B->getRHS(), N1, S2);
  
    for (NodeSet::iterator I2=S2.begin(), E2=S2.end(); I2 != E2; ++I2) {
      NodeTy* N2 = *I2;
      StateTy St = N2->getState();
      const ExprValue& V2 = GetValue(St, B->getRHS());

      switch (B->getOpcode()) {
        case BinaryOperator::Add: {
          const RValue& R1 = cast<RValue>(V1);
          const RValue& R2 = cast<RValue>(V2);
          
          Nodify(Dst, B, N2, SetValue(St, B, R1.Add(ValMgr, R2)));
          break;
        }

        case BinaryOperator::Sub: {
          const RValue& R1 = cast<RValue>(V1);
          const RValue& R2 = cast<RValue>(V2);
	        Nodify(Dst, B, N2, SetValue(St, B, R1.Sub(ValMgr, R2)));
          break;
        }
          
        case BinaryOperator::Assign: {
          const LValue& L1 = cast<LValue>(V1);
          const RValue& R2 = cast<RValue>(V2);
          Nodify(Dst, B, N2, SetValue(SetValue(St, B, R2), L1, R2));
          break;
        }
          
        case BinaryOperator::AddAssign: {
          const LValue& L1 = cast<LValue>(V1);
          RValue R1 = cast<RValue>(GetValue(N1->getState(), L1));
          RValue Result = R1.Add(ValMgr, cast<RValue>(V2));
          Nodify(Dst, B, N2, SetValue(SetValue(St, B, Result), L1, Result));
          break;
        }
          
        case BinaryOperator::SubAssign: {
          const LValue& L1 = cast<LValue>(V1);
          RValue R1 = cast<RValue>(GetValue(N1->getState(), L1));
          RValue Result = R1.Sub(ValMgr, cast<RValue>(V2));
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
      
    case Stmt::ParenExprClass:
      Visit(cast<ParenExpr>(S)->getSubExpr(), Pred, Dst);
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
    
  static void PrintKind(std::ostringstream& Out, ValueKey::Kind kind) {
    switch (kind) {
      case ValueKey::IsSubExp:  Out << "Sub-Expressions:\\l"; break;
      case ValueKey::IsDecl:    Out << "Variables:\\l"; break;
      case ValueKey::IsBlkExpr: Out << "Block-level Expressions:\\l"; break;
      default: assert (false && "Unknown ValueKey type.");
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
        Out << "(" << (void*) L.getStmt() << ") ";
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
    
    GRConstants::StateTy M = N->getState();
    bool isFirst = true;
    ValueKey::Kind kind;

    for (GRConstants::StateTy::iterator I=M.begin(), E=M.end(); I!=E; ++I) {
      if (!isFirst) {
        ValueKey::Kind newKind = I.getKey().getKind();
        
        if (newKind != kind) {
          Out << "\\l\\l";
          PrintKind(Out, newKind);
        }
        else
          Out << "\\l";
        
        kind = newKind;
      }
      else {
        kind = I.getKey().getKind();
        PrintKind(Out, kind); 
        isFirst = false;
      }
      
      Out << ' ';
      
      if (ValueDecl* V = dyn_cast<ValueDecl>(I.getKey())) {
        Out << V->getName();          
      }
      else {
        Stmt* E = cast<Stmt>(I.getKey());
        Out << " (" << (void*) E << ") ";
        E->printPretty(Out);
      }
      
      Out << " : ";
      I.getData().print(Out);
    }
    
    Out << "\\l";
    return Out.str();
  }
};
} // end llvm namespace    
#endif

namespace clang {
void RunGRConstants(CFG& cfg) {
  GREngine<GRConstants> Engine(cfg);
  Engine.ExecuteWorkList();  
#ifndef NDEBUG
  llvm::ViewGraph(*Engine.getGraph().roots_begin(),"GRConstants");
#endif  
}
} // end clang namespace
