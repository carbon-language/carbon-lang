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

#ifndef NDEBUG
#include "llvm/Support/GraphWriter.h"
#include <sstream>
#endif

using namespace clang;
using llvm::APInt;
using llvm::APFloat;
using llvm::dyn_cast;
using llvm::cast;

//===----------------------------------------------------------------------===//
/// DSPtr - A variant smart pointer that wraps either a ValueDecl* or a
///  Stmt*.  Use cast<> or dyn_cast<> to get actual pointer type
//===----------------------------------------------------------------------===//
namespace {
class VISIBILITY_HIDDEN DSPtr {
  uintptr_t Raw;
public:
  enum  VariantKind { IsSubExp=0x0, IsValueDecl=0x1, IsBlkLvl=0x2, Flags=0x3 };
  inline void* getPtr() const { return reinterpret_cast<void*>(Raw & ~Flags); }
  inline VariantKind getKind() const { return (VariantKind) (Raw & Flags); }
  
  DSPtr(ValueDecl* D) : Raw(reinterpret_cast<uintptr_t>(D) | IsValueDecl) {}
  DSPtr(Stmt* S, bool isBlkLvl) 
    : Raw(reinterpret_cast<uintptr_t>(S) | (isBlkLvl ? IsBlkLvl : IsSubExp)) {}
  
  bool isSubExpr() const { return getKind() == IsSubExp; }
  
  inline void Profile(llvm::FoldingSetNodeID& ID) const {
    ID.AddPointer(getPtr());
    ID.AddInteger((unsigned) getKind());
  }      
  inline bool operator==(const DSPtr& X) const { return Raw == X.Raw; }  
  inline bool operator!=(const DSPtr& X) const { return Raw != X.Raw; }
  inline bool operator<(const DSPtr& X) const { 
    VariantKind k = getKind(), Xk = X.getKind();
    return k == Xk ? getPtr() < X.getPtr() : ((unsigned) k) < ((unsigned) Xk);
  }
};
} // end anonymous namespace

// Machinery to get cast<> and dyn_cast<> working with DSPtr.
namespace llvm {
  template<> inline bool isa<ValueDecl,DSPtr>(const DSPtr& V) {
    return V.getKind() == DSPtr::IsValueDecl;
  }
  template<> inline bool isa<Stmt,DSPtr>(const DSPtr& V) {
    return ((unsigned) V.getKind()) != DSPtr::IsValueDecl;
  }
  template<> struct VISIBILITY_HIDDEN cast_retty_impl<ValueDecl,DSPtr> {
    typedef const ValueDecl* ret_type;
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
    else return ExprVariantTy(val-X.val);
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
  // Liveness - live-variables information the ValueDecl* and Expr* (block-level)
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
  NodeTy* InitialPred;
  
public:
  GRConstants() : Liveness(NULL), Builder(NULL), cfg(NULL), 
    Nodes(&NodeSetA), OldNodes(&NodeSetB), 
    CurrentState(StateMgr.GetEmptyMap()), InitialPred(NULL) {} 
    
  ~GRConstants() { delete Liveness; }
  
  CFG& getCFG() { assert (cfg); return *cfg; }
      
  void Initialize(CFG& c) {
    cfg = &c;    
    Liveness = new LiveVariables(c);
    Liveness->runOnCFG(c);
    Liveness->runOnAllBlocks(c, NULL, true);
  }
  
  StateTy getInitialState() {
    return StateMgr.GetEmptyMap();
  }
    
  void ProcessStmt(Stmt* S, NodeBuilder& builder);    
  void SwitchNodeSets();
  void DoStmt(Stmt* S);
  
  StateTy RemoveDescendantMappings(Stmt* S, StateTy M, unsigned Levels=2);
  StateTy RemoveSubExprMappings(StateTy M);

  void AddBinding(Expr* E, ExprVariantTy V, bool isBlkLvl = false);
  void AddBinding(ValueDecl* D, ExprVariantTy V);
  
  ExprVariantTy GetBinding(Expr* E);
  
  void BlockStmt_VisitStmt(Stmt* S) { DoStmt(S); }
  
  void VisitAssign(BinaryOperator* O);
  void VisitBinAdd(BinaryOperator* O);
  void VisitBinSub(BinaryOperator* O);
  void VisitBinAssign(BinaryOperator* D);
};
} // end anonymous namespace

void GRConstants::ProcessStmt(Stmt* S, NodeBuilder& builder) {
  Builder = &builder;
  Nodes->clear();
  OldNodes->clear();
  InitialPred = Builder->getLastNode();
  assert (InitialPred);
  OldNodes->push_back(InitialPred);  
  CurrentState = RemoveSubExprMappings(InitialPred->getState());  
  BlockStmt_Visit(S);
  Builder = NULL;  
}

ExprVariantTy GRConstants::GetBinding(Expr* E) {
  DSPtr P(NULL);
  E = E->IgnoreParens();
  
  switch (E->getStmtClass()) {
    case Stmt::DeclRefExprClass:
      P = DSPtr(cast<DeclRefExpr>(E)->getDecl());
      break;

    case Stmt::IntegerLiteralClass:
      return cast<IntegerLiteral>(E)->getValue().getZExtValue();
      
    default:    
      P = DSPtr(E, getCFG().isBlkExpr(E));
      break;
  }

  StateTy::iterator I = CurrentState.find(P);

  if (I == CurrentState.end())
    return ExprVariantTy();
  
  return (*I).second;
}

void GRConstants::AddBinding(Expr* E, ExprVariantTy V, bool isBlkLvl) {
  if (V) 
    CurrentState = StateMgr.Add(CurrentState, DSPtr(E,isBlkLvl), V.getVal());
}

void GRConstants::AddBinding(ValueDecl* D, ExprVariantTy V) {
  if (V)
    CurrentState = StateMgr.Add(CurrentState, DSPtr(D), V.getVal());
  else
    CurrentState = StateMgr.Remove(CurrentState, DSPtr(D));
}

void GRConstants::SwitchNodeSets() {
  NodeSetTy* Tmp = OldNodes;
  OldNodes = Nodes;
  Nodes = Tmp;
  Nodes->clear(); 
}

GRConstants::StateTy
GRConstants::RemoveSubExprMappings(StateTy M) {
  for (StateTy::iterator I = M.begin(), E = M.end();
       I!=E && I.getKey().getKind() == DSPtr::IsSubExp; ++I) {
    // Note: we can assign a new map to M since the iterators are
    //  iterating over the tree of the original map (aren't immutable maps
    //  nice?).
    M = StateMgr.Remove(M, I.getKey());
  }

  return M;
}


GRConstants::StateTy
GRConstants::RemoveDescendantMappings(Stmt* S, GRConstants::StateTy State,
                                      unsigned Levels) {
  typedef Stmt::child_iterator iterator;
  
  for (iterator I=S->child_begin(), E=S->child_end(); I!=E; ++I)
    if (Stmt* C = *I) {
      if (Levels == 1) {
        // Observe that this will only remove mappings to non-block level
        // expressions.  This is valid even if *CI is a block-level expression,
        // since it simply won't be in the map in the first place.
        // Note: This should also work if 'C' is a block-level expression,
        // although ideally we would want to skip processing C's children.
        State = StateMgr.Remove(State, DSPtr(C,false));
      }
      else {
        if (ParenExpr* P = dyn_cast<ParenExpr>(C))
          State = RemoveDescendantMappings(P, State, Levels);
        else
          State = RemoveDescendantMappings(C, State, Levels-1);
      }
    }
  
  return State;
}

void GRConstants::DoStmt(Stmt* S) {  
  for (Stmt::child_iterator I=S->child_begin(), E=S->child_end(); I!=E; ++I)
    if (*I) DoStmt(*I);
  
  for (NodeSetTy::iterator I=OldNodes->begin(), E=OldNodes->end(); I!=E; ++I) {
    NodeTy* Pred = *I;
    
    if (Pred != InitialPred)
      CurrentState = Pred->getState();

    StateTy OldState = CurrentState;
    
    if (Pred != InitialPred)
      CurrentState = RemoveDescendantMappings(S, CurrentState);
    
    Visit(S);
    
    if (CurrentState != OldState) {
      NodeTy* N = Builder->generateNode(S, CurrentState, Pred);
      if (N) Nodes->push_back(N);
    }
    else Nodes->push_back(Pred);    
  }
  
  SwitchNodeSets();
}

void GRConstants::VisitBinAdd(BinaryOperator* B) {
  AddBinding(B, GetBinding(B->getLHS()) + GetBinding(B->getRHS()));
}

void GRConstants::VisitBinSub(BinaryOperator* B) {
  AddBinding(B, GetBinding(B->getLHS()) - GetBinding(B->getRHS()));
}


void GRConstants::VisitBinAssign(BinaryOperator* B) {
  if (DeclRefExpr* D = dyn_cast<DeclRefExpr>(B->getLHS()->IgnoreParens())) {
    ExprVariantTy V = GetBinding(B->getRHS());
    AddBinding(D->getDecl(), V);
    AddBinding(B, V);
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
    
  static std::string getNodeLabel(const GRConstants::NodeTy* N, void*) {
    std::ostringstream Out;
        
    Out << "Vertex: " << (void*) N << '\n';
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
        Out << "Stmt: " << (void*) L.getStmt() << '\n';
        L.getStmt()->printPretty(Out);
        break;
      }
    
      default: {
        const BlockEdge& E = cast<BlockEdge>(Loc);
        Out << "Edge: (B" << E.getSrc()->getBlockID() << ", B"
            << E.getDst()->getBlockID()  << ')';
      }
    }
    
    Out << "\n{";
    
    GRConstants::StateTy M = N->getState();
    bool isFirst = true;

    for (GRConstants::StateTy::iterator I=M.begin(), E=M.end(); I!=E; ++I) {
      if (!isFirst)
        Out << '\n';
      else
        isFirst = false;
      
      if (ValueDecl* V = dyn_cast<ValueDecl>(I.getKey())) {
        Out << "Decl: " << (void*) V << ", " << V->getName();          
      }
      else {
        Stmt* E = cast<Stmt>(I.getKey());
        Out << "Stmt: " << (void*) E;
      }
      
      Out << " => " << I.getData();
    }
    
    Out << " }";
    
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
}
