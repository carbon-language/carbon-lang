//===-- GRConstants.cpp - Simple, Path-Sens. Constant Prop. ------*- C++ -*-==//
//   
//                     The LLVM Compiler Infrastructure
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
#include "clang/Analysis/Visitors/CFGStmtVisitor.h"

#include "llvm/Support/Casting.h"
#include "llvm/Support/DataTypes.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/FoldingSet.h"
#include "llvm/ADT/ImmutableMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Compiler.h"

using namespace clang;
using llvm::APInt;
using llvm::APFloat;
using llvm::dyn_cast;
using llvm::cast;

//===----------------------------------------------------------------------===//
/// DSPtr - A variant smart pointer that wraps either a Decl* or a
///  Stmt*.  Use cast<> or dyn_cast<> to get actual pointer type
//===----------------------------------------------------------------------===//
namespace {
class VISIBILITY_HIDDEN DSPtr {
  const uintptr_t Raw;
public:
  enum  VariantKind { IsDecl=0x1, IsBlkLvl=0x2, IsSubExp=0x3, Flags=0x3 };
  inline void* getPtr() const { return reinterpret_cast<void*>(Raw & ~Flags); }
  inline VariantKind getKind() const { return (VariantKind) (Raw & Flags); }
  
  DSPtr(Decl* D) : Raw(reinterpret_cast<uintptr_t>(D) | IsDecl) {}
  DSPtr(Stmt* S, bool isBlkLvl) 
    : Raw(reinterpret_cast<uintptr_t>(S) | (isBlkLvl ? IsBlkLvl : IsSubExp)) {}
  
  bool isSubExpr() const { return getKind() == IsSubExp; }
  
  inline void Profile(llvm::FoldingSetNodeID& ID) const {
    ID.AddPointer(getPtr());
    ID.AddInteger((unsigned) getKind());
  }      
  inline bool operator==(const DSPtr& X) const { return Raw == X.Raw; }  
  inline bool operator!=(const DSPtr& X) const { return Raw != X.Raw; }
  inline bool operator<(const DSPtr& X) const { return Raw < X.Raw; }  
};
} // end anonymous namespace

// Machinery to get cast<> and dyn_cast<> working with DSPtr.
namespace llvm {
  template<> inline bool isa<Decl,DSPtr>(const DSPtr& V) {
    return V.getKind() == DSPtr::IsDecl;
  }
  template<> inline bool isa<Stmt,DSPtr>(const DSPtr& V) {
    return ((unsigned) V.getKind()) > DSPtr::IsDecl;
  }
  template<> struct VISIBILITY_HIDDEN cast_retty_impl<Decl,DSPtr> {
    typedef const Decl* ret_type;
  };
  template<> struct VISIBILITY_HIDDEN cast_retty_impl<Stmt,DSPtr> {
    typedef const Stmt* ret_type;
  };
  template<> struct VISIBILITY_HIDDEN simplify_type<DSPtr> {
    typedef void* SimpleType;
    static inline SimpleType getSimplifiedValue(const DSPtr &V) {
      return V.getPtr();
    }
  };
} // end llvm namespace

//===----------------------------------------------------------------------===//
// DeclStmtMapTy - A ImmutableMap type from Decl*/Stmt* to integers.
//
//  FIXME: We may eventually use APSInt, or a mixture of APSInt and
//         integer primitives to do this right; this will handle both
//         different bit-widths and allow us to detect integer overflows, etc.
//
//===----------------------------------------------------------------------===//

typedef llvm::ImmutableMap<DSPtr,uint64_t> DeclStmtMapTy;

namespace clang {
template<>
struct VISIBILITY_HIDDEN GRTrait<DeclStmtMapTy> {
  static inline void* toPtr(DeclStmtMapTy M) {
    return reinterpret_cast<void*>(M.getRoot());
  }  
  static inline DeclStmtMapTy toState(void* P) {
    return DeclStmtMapTy(static_cast<DeclStmtMapTy::TreeTy*>(P));
  }
};
}

//===----------------------------------------------------------------------===//
// The Checker!
//===----------------------------------------------------------------------===//

namespace {
class VISIBILITY_HIDDEN ExprVariantTy {
  const uint64_t val;
  const bool isConstant;
public:
  ExprVariantTy() : val(0), isConstant(false) {}
  ExprVariantTy(uint64_t v) : val(v), isConstant(true) {}
  
  operator bool() const { return isConstant; }
  uint64_t getVal() const { assert (isConstant); return val; }
  
  ExprVariantTy operator+(const ExprVariantTy& X) const {
    if (!isConstant || !X.isConstant) return ExprVariantTy();
    else return ExprVariantTy(val+X.val);
  }
  
  ExprVariantTy operator-(const ExprVariantTy& X) const {
    if (!isConstant || !X.isConstant) return ExprVariantTy();
    else return ExprVariantTy(val+X.val);
  }    
};
} // end anonymous namespace

//===----------------------------------------------------------------------===//
// The Checker!
//===----------------------------------------------------------------------===//

namespace {
class VISIBILITY_HIDDEN GRConstants : public CFGStmtVisitor<GRConstants> {
    
public:
  typedef DeclStmtMapTy StateTy;
  typedef GRNodeBuilder<GRConstants> NodeBuilder;
  typedef ExplodedNode<StateTy> NodeTy;
                                                              
protected:
  // Liveness - live-variables information the Decl* and Expr* (block-level)
  //  in the CFG.  Used to prune out dead state.
  LiveVariables* Liveness;

  // Builder - The current GRNodeBuilder which is used when building the nodes
  //  for a given statement.
  NodeBuilder* Builder;

  DeclStmtMapTy::Factory StateMgr;
  
  // cfg - the current CFG.
  CFG* cfg;

  typedef llvm::SmallVector<NodeTy*,8> NodeSetTy;
  NodeSetTy NodeSetA;
  NodeSetTy NodeSetB;
  NodeSetTy* Nodes;
  NodeSetTy* OldNodes;
  StateTy CurrentState;
  
public:
  GRConstants() : Liveness(NULL), Builder(NULL), cfg(NULL), 
    Nodes(&NodeSetA), OldNodes(&NodeSetB), 
    CurrentState(StateMgr.GetEmptyMap()) {} 
    
  ~GRConstants() { delete Liveness; }
  
  CFG& getCFG() { assert (cfg); return *cfg; }
      
  void Initialize(CFG& c) {
    cfg = &c;    
    Liveness = new LiveVariables(c);
    Liveness->runOnCFG(c);
  }
  
  StateTy getInitialState() {
    return StateMgr.GetEmptyMap();
  }
    
  void ProcessStmt(Stmt* S, NodeBuilder& builder);    
  void SwitchNodeSets();
  void DoStmt(Stmt* S);
  StateTy RemoveGrandchildrenMappings(Stmt* S, StateTy M);

  void AddBinding(Expr* E, ExprVariantTy V, bool isBlkLvl = false);
  ExprVariantTy GetBinding(Expr* E);
  
  void BlockStmt_VisitStmt(Stmt* S) { DoStmt(S); }
  
  void VisitAssign(BinaryOperator* O);
  void VisitIntegerLiteral(IntegerLiteral* L);
  void VisitBinAdd(BinaryOperator* O);
  void VisitBinSub(BinaryOperator* O);
};
} // end anonymous namespace

void GRConstants::ProcessStmt(Stmt* S, NodeBuilder& builder) {
  Builder = &builder;
  Nodes->clear();
  OldNodes->clear();
  NodeTy* N = Builder->getLastNode();
  assert (N);
  OldNodes->push_back(N);
  BlockStmt_Visit(S);
  Builder = NULL;  
}

ExprVariantTy GRConstants::GetBinding(Expr* E) {
  DSPtr P(E, getCFG().isBlkExpr(E));
  StateTy::iterator I = CurrentState.find(P);

  if (I == CurrentState.end())
    return ExprVariantTy();
  
  return (*I).second;
}

void GRConstants::AddBinding(Expr* E, ExprVariantTy V, bool isBlkLvl) {
  CurrentState = StateMgr.Add(CurrentState, DSPtr(E,isBlkLvl), V.getVal());
}

void GRConstants::SwitchNodeSets() {
  NodeSetTy* Tmp = OldNodes;
  OldNodes = Nodes;
  Nodes = Tmp;
  Nodes->clear(); 
}

GRConstants::StateTy
GRConstants::RemoveGrandchildrenMappings(Stmt* S, GRConstants::StateTy State) {
  
  typedef Stmt::child_iterator iterator;
  
  for (iterator I=S->child_begin(), E=S->child_end(); I!=E; ++I)
    if (Stmt* C = *I)
      for (iterator CI=C->child_begin(), CE=C->child_end(); CI!=CE; ++CI) {
        // Observe that this will only remove mappings to non-block level
        // expressions.  This is valid even if *CI is a block-level expression,
        // since it simply won't be in the map in the first place.
        // Note: This should also work if 'C' is a block-level expression,
        // although ideally we would want to skip processing C's children.
        State = StateMgr.Remove(State, DSPtr(*CI,false));
      }
  
  return State;
}

void GRConstants::DoStmt(Stmt* S) {  
  for (Stmt::child_iterator I=S->child_begin(), E=S->child_end(); I!=E; ++I)
    if (*I) DoStmt(*I);
  
  for (NodeSetTy::iterator I=OldNodes->begin(), E=OldNodes->end(); I!=E; ++I) {
    NodeTy* Pred = *I;
    CurrentState = Pred->getState();

    StateTy OldState = CurrentState;
    CurrentState = RemoveGrandchildrenMappings(S, CurrentState);
    
    Visit(S);
    
    if (CurrentState != OldState) {
      NodeTy* N = Builder->generateNode(S, CurrentState, Pred);
      if (N) Nodes->push_back(N);
    }
    else Nodes->push_back(Pred);    
  }
  
  SwitchNodeSets();
}

void GRConstants::VisitIntegerLiteral(IntegerLiteral* L) {
  AddBinding(L, L->getValue().getZExtValue());
}

void GRConstants::VisitBinAdd(BinaryOperator* B) {
  AddBinding(B, GetBinding(B->getLHS()) + GetBinding(B->getRHS()));
}

void GRConstants::VisitBinSub(BinaryOperator* B) {
  AddBinding(B, GetBinding(B->getLHS()) - GetBinding(B->getRHS()));
}
    