//===- Consumed.cpp --------------------------------------------*- C++ --*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// A intra-procedural analysis for checking consumed properties.  This is based,
// in part, on research on linear types.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/ASTContext.h"
#include "clang/AST/Attr.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/StmtVisitor.h"
#include "clang/AST/StmtCXX.h"
#include "clang/AST/Type.h"
#include "clang/Analysis/Analyses/PostOrderCFGView.h"
#include "clang/Analysis/AnalysisContext.h"
#include "clang/Analysis/CFG.h"
#include "clang/Analysis/Analyses/Consumed.h"
#include "clang/Basic/OperatorKinds.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Sema/ConsumedWarningsHandler.h"
#include "clang/Sema/SemaDiagnostic.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

// TODO: Add support for methods with CallableWhenUnconsumed.
// TODO: Mark variables as Unknown going into while- or for-loops only if they
//       are referenced inside that block. (Deferred)
// TODO: Add a method(s) to identify which method calls perform what state
//       transitions. (Deferred)
// TODO: Take notes on state transitions to provide better warning messages.
//       (Deferred)
// TODO: Test nested conditionals: A) Checking the same value multiple times,
//       and 2) Checking different values. (Deferred)
// TODO: Test IsFalseVisitor with values in the unknown state. (Deferred)
// TODO: Look into combining IsFalseVisitor and TestedVarsVisitor. (Deferred)

using namespace clang;
using namespace consumed;

// Key method definition
ConsumedWarningsHandlerBase::~ConsumedWarningsHandlerBase() {}

static StringRef stateToString(ConsumedState State) {
  switch (State) {
  case consumed::CS_None:
    return "none";
  
  case consumed::CS_Unknown:
    return "unknown";
  
  case consumed::CS_Unconsumed:
    return "unconsumed";
  
  case consumed::CS_Consumed:
    return "consumed";
  }
}

namespace {
class ConsumedStmtVisitor : public ConstStmtVisitor<ConsumedStmtVisitor> {
  
  union PropagationUnion {
    ConsumedState State;
    const VarDecl *Var;
  };
  
  class PropagationInfo {
    PropagationUnion StateOrVar;
  
  public:
    bool IsVar;
    
    PropagationInfo() : IsVar(false) {
      StateOrVar.State = consumed::CS_None;
    }
    
    PropagationInfo(ConsumedState State) : IsVar(false) {
      StateOrVar.State = State;
    }
    
    PropagationInfo(const VarDecl *Var) : IsVar(true) {
      StateOrVar.Var = Var;
    }
    
    ConsumedState getState() { return StateOrVar.State; };
    
    const VarDecl * getVar() { return IsVar ? StateOrVar.Var : NULL; };
  };
  
  typedef llvm::DenseMap<const Stmt *, PropagationInfo> MapType;
  typedef std::pair<const Stmt *, PropagationInfo> PairType;
  typedef MapType::iterator InfoEntry;
  
  ConsumedAnalyzer &Analyzer;
  ConsumedStateMap *StateMap;
  MapType PropagationMap;
  
  void forwardInfo(const Stmt *From, const Stmt *To);
  bool isLikeMoveAssignment(const CXXMethodDecl *MethodDecl);
  
public:
  
  void Visit(const Stmt *StmtNode);
  
  void VisitBinaryOperator(const BinaryOperator *BinOp);
  void VisitCallExpr(const CallExpr *Call);
  void VisitCastExpr(const CastExpr *Cast);
  void VisitCXXConstructExpr(const CXXConstructExpr *Call);
  void VisitCXXMemberCallExpr(const CXXMemberCallExpr *Call);
  void VisitCXXOperatorCallExpr(const CXXOperatorCallExpr *Call);
  void VisitDeclRefExpr(const DeclRefExpr *DeclRef);
  void VisitDeclStmt(const DeclStmt *DelcS);
  void VisitMaterializeTemporaryExpr(const MaterializeTemporaryExpr *Temp);
  void VisitMemberExpr(const MemberExpr *MExpr);
  void VisitUnaryOperator(const UnaryOperator *UOp);
  void VisitVarDecl(const VarDecl *Var);
  
  ConsumedStmtVisitor(ConsumedAnalyzer &Analyzer, ConsumedStateMap *StateMap) :
    Analyzer(Analyzer), StateMap(StateMap) {}
  
  void reset() {
    PropagationMap.clear();
  }
};

void ConsumedStmtVisitor::forwardInfo(const Stmt *From, const Stmt *To) {
  InfoEntry Entry = PropagationMap.find(From);
  
  if (Entry != PropagationMap.end()) {
    PropagationMap.insert(PairType(To, PropagationInfo(Entry->second)));
  }
}

bool ConsumedStmtVisitor::isLikeMoveAssignment(
  const CXXMethodDecl *MethodDecl) {
  
  return MethodDecl->isMoveAssignmentOperator() ||
         (MethodDecl->getOverloadedOperator() == OO_Equal &&
          MethodDecl->getNumParams() == 1 &&
          MethodDecl->getParamDecl(0)->getType()->isRValueReferenceType());
}

void ConsumedStmtVisitor::VisitBinaryOperator(const BinaryOperator *BinOp) {
  switch (BinOp->getOpcode()) {
  case BO_PtrMemD:
  case BO_PtrMemI:
    forwardInfo(BinOp->getLHS(), BinOp);
    break;
    
  default:
    break;
  }
}

void ConsumedStmtVisitor::Visit(const Stmt *StmtNode) {
  ConstStmtVisitor<ConsumedStmtVisitor>::Visit(StmtNode);
  
  for (Stmt::const_child_iterator CI = StmtNode->child_begin(),
       CE = StmtNode->child_end(); CI != CE; ++CI) {
    
    PropagationMap.erase(*CI);
  }
}

void ConsumedStmtVisitor::VisitCallExpr(const CallExpr *Call) {
  if (const FunctionDecl *FunDecl =
    dyn_cast_or_null<FunctionDecl>(Call->getDirectCallee())) {
    
    // Special case for the std::move function.
    // TODO: Make this more specific. (Deferred)
    if (FunDecl->getNameAsString() == "move") {
      InfoEntry Entry = PropagationMap.find(Call->getArg(0));
      
      if (Entry != PropagationMap.end()) {
        PropagationMap.insert(PairType(Call, Entry->second));
      }
      
      return;
    }
    
    unsigned Offset = Call->getNumArgs() - FunDecl->getNumParams();
    
    for (unsigned Index = Offset; Index < Call->getNumArgs(); ++Index) {
      QualType ParamType = FunDecl->getParamDecl(Index - Offset)->getType();
      
      InfoEntry Entry = PropagationMap.find(Call->getArg(Index));
      
      if (Entry == PropagationMap.end() || !Entry->second.IsVar) {
        continue;
      }
      
      PropagationInfo PState = Entry->second;
      
      if (ParamType->isRValueReferenceType() ||
          (ParamType->isLValueReferenceType() &&
           !cast<LValueReferenceType>(*ParamType).isSpelledAsLValue())) {
        
        StateMap->setState(PState.getVar(), consumed::CS_Consumed);
        
      } else if (!(ParamType.isConstQualified() ||
                   ((ParamType->isReferenceType() ||
                     ParamType->isPointerType()) &&
                    ParamType->getPointeeType().isConstQualified()))) {
        
        StateMap->setState(PState.getVar(), consumed::CS_Unknown);
      }
    }
  }
}

void ConsumedStmtVisitor::VisitCastExpr(const CastExpr *Cast) {
  InfoEntry Entry = PropagationMap.find(Cast->getSubExpr());
  
  if (Entry != PropagationMap.end())
    PropagationMap.insert(PairType(Cast, Entry->second));
}

void ConsumedStmtVisitor::VisitCXXConstructExpr(const CXXConstructExpr *Call) {
  CXXConstructorDecl *Constructor = Call->getConstructor();
  
  ASTContext &CurrContext = Analyzer.getSema().getASTContext();
  QualType ThisType = Constructor->getThisType(CurrContext)->getPointeeType();
  
  if (Analyzer.isConsumableType(ThisType)) {
    if (Constructor->hasAttr<ConsumesAttr>() ||
        Constructor->isDefaultConstructor()) {
      
      PropagationMap.insert(PairType(Call,
        PropagationInfo(consumed::CS_Consumed)));
      
    } else if (Constructor->isMoveConstructor()) {
      
      PropagationInfo PState =
        PropagationMap.find(Call->getArg(0))->second;
      
      if (PState.IsVar) {
        const VarDecl* Var = PState.getVar();
        
        PropagationMap.insert(PairType(Call,
          PropagationInfo(StateMap->getState(Var))));
        
        StateMap->setState(Var, consumed::CS_Consumed);
        
      } else {
        PropagationMap.insert(PairType(Call, PState));
      }
        
    } else if (Constructor->isCopyConstructor()) {
      MapType::iterator Entry = PropagationMap.find(Call->getArg(0));
    
      if (Entry != PropagationMap.end())
        PropagationMap.insert(PairType(Call, Entry->second));
      
    } else {
      PropagationMap.insert(PairType(Call,
        PropagationInfo(consumed::CS_Unconsumed)));
    }
  }
}

void ConsumedStmtVisitor::VisitCXXMemberCallExpr(
  const CXXMemberCallExpr *Call) {
  
  VisitCallExpr(Call);
  
  InfoEntry Entry = PropagationMap.find(Call->getCallee()->IgnoreParens());
  
  if (Entry != PropagationMap.end()) {
    PropagationInfo PState = Entry->second;
    if (!PState.IsVar) return;
    
    const CXXMethodDecl *Method = Call->getMethodDecl();
    
    if (Method->hasAttr<ConsumesAttr>())
      StateMap->setState(PState.getVar(), consumed::CS_Consumed);
    else if (!Method->isConst())
      StateMap->setState(PState.getVar(), consumed::CS_Unknown);
  }
}

void ConsumedStmtVisitor::VisitCXXOperatorCallExpr(
  const CXXOperatorCallExpr *Call) {
  
  const FunctionDecl *FunDecl =
    dyn_cast_or_null<FunctionDecl>(Call->getDirectCallee());
  
  if (!FunDecl) return;
    
  if (isa<CXXMethodDecl>(FunDecl) &&
      isLikeMoveAssignment(cast<CXXMethodDecl>(FunDecl))) {
    
    InfoEntry LEntry = PropagationMap.find(Call->getArg(0));
    InfoEntry REntry = PropagationMap.find(Call->getArg(1));
    
    PropagationInfo LPState, RPState;
    
    if (LEntry != PropagationMap.end() &&
        REntry != PropagationMap.end()) {
      
      LPState = LEntry->second;
      RPState = REntry->second;
      
      if (LPState.IsVar && RPState.IsVar) {
        StateMap->setState(LPState.getVar(),
          StateMap->getState(RPState.getVar()));
        
        StateMap->setState(RPState.getVar(), consumed::CS_Consumed);
        
        PropagationMap.insert(PairType(Call, LPState));
        
      } else if (LPState.IsVar && !RPState.IsVar) {
        StateMap->setState(LPState.getVar(), RPState.getState());
        
        PropagationMap.insert(PairType(Call, LPState));
        
      } else if (!LPState.IsVar && RPState.IsVar) {
        PropagationMap.insert(PairType(Call,
          PropagationInfo(StateMap->getState(RPState.getVar()))));
        
        StateMap->setState(RPState.getVar(), consumed::CS_Consumed);
        
      } else {
        PropagationMap.insert(PairType(Call, RPState));
      }
      
    } else if (LEntry != PropagationMap.end() &&
               REntry == PropagationMap.end()) {
      
      LPState = LEntry->second;
      
      if (LPState.IsVar) {
        StateMap->setState(LPState.getVar(), consumed::CS_Unknown);
        
        PropagationMap.insert(PairType(Call, LPState));
        
      } else {
        PropagationMap.insert(PairType(Call,
          PropagationInfo(consumed::CS_Unknown)));
      }
      
    } else if (LEntry == PropagationMap.end() &&
               REntry != PropagationMap.end()) {
      
      RPState = REntry->second;
      
      if (RPState.IsVar) {
        const VarDecl *Var = RPState.getVar();
        
        PropagationMap.insert(PairType(Call,
          PropagationInfo(StateMap->getState(Var))));
        
        StateMap->setState(Var, consumed::CS_Consumed);
        
      } else {
        PropagationMap.insert(PairType(Call, RPState));
      }
    }
    
  } else {
    
    VisitCallExpr(Call);
    
    InfoEntry Entry = PropagationMap.find(Call->getArg(0));
    
    if (Entry != PropagationMap.end()) {
      
      PropagationInfo PState = Entry->second;
      
      // TODO: When we support CallableWhenConsumed this will have to check for
      //       the different attributes and change the behavior bellow.
      //       (Deferred)
      if (FunDecl->hasAttr<CallableWhenUnconsumedAttr>()) {
        if (PState.IsVar) {
          const VarDecl *Var = PState.getVar();
          
          switch (StateMap->getState(Var)) {
          case CS_Consumed:
            Analyzer.WarningsHandler.warnUseWhileConsumed(
              FunDecl->getNameAsString(), Var->getNameAsString(),
              Call->getExprLoc());
            break;
          
          case CS_Unknown:
            Analyzer.WarningsHandler.warnUseInUnknownState(
              FunDecl->getNameAsString(), Var->getNameAsString(),
              Call->getExprLoc());
            break;
            
          default:
            break;
          }
          
        } else {
          switch (PState.getState()) {
          case CS_Consumed:
            Analyzer.WarningsHandler.warnUseOfTempWhileConsumed(
              FunDecl->getNameAsString(), Call->getExprLoc());
            break;
          
          case CS_Unknown:
            Analyzer.WarningsHandler.warnUseOfTempInUnknownState(
              FunDecl->getNameAsString(), Call->getExprLoc());
            break;
            
          default:
            break;
          }
        }
      }
      
      // Handle non-constant member operators.
      if (const CXXMethodDecl *MethodDecl =
        dyn_cast_or_null<CXXMethodDecl>(FunDecl)) {
        
        if (!MethodDecl->isConst() && PState.IsVar)
          StateMap->setState(PState.getVar(), consumed::CS_Unknown);
      }
    }
  }
}

void ConsumedStmtVisitor::VisitDeclRefExpr(const DeclRefExpr *DeclRef) {
  if (const VarDecl *Var = dyn_cast_or_null<VarDecl>(DeclRef->getDecl()))
    if (StateMap->getState(Var) != consumed::CS_None)
      PropagationMap.insert(PairType(DeclRef, PropagationInfo(Var)));
}

void ConsumedStmtVisitor::VisitDeclStmt(const DeclStmt *DeclS) {
  for (DeclStmt::const_decl_iterator DI = DeclS->decl_begin(),
       DE = DeclS->decl_end(); DI != DE; ++DI) {
    
    if (isa<VarDecl>(*DI)) VisitVarDecl(cast<VarDecl>(*DI));
  }
  
  if (DeclS->isSingleDecl())
    if (const VarDecl *Var = dyn_cast_or_null<VarDecl>(DeclS->getSingleDecl()))
      PropagationMap.insert(PairType(DeclS, PropagationInfo(Var)));
}

void ConsumedStmtVisitor::VisitMaterializeTemporaryExpr(
  const MaterializeTemporaryExpr *Temp) {
  
  InfoEntry Entry = PropagationMap.find(Temp->GetTemporaryExpr());
  
  if (Entry != PropagationMap.end())
    PropagationMap.insert(PairType(Temp, Entry->second));
}

void ConsumedStmtVisitor::VisitMemberExpr(const MemberExpr *MExpr) {
  forwardInfo(MExpr->getBase(), MExpr);
}

void ConsumedStmtVisitor::VisitUnaryOperator(const UnaryOperator *UOp) {
  if (UOp->getOpcode() == UO_AddrOf) {
    InfoEntry Entry = PropagationMap.find(UOp->getSubExpr());
    
    if (Entry != PropagationMap.end())
      PropagationMap.insert(PairType(UOp, Entry->second));
  }
}

void ConsumedStmtVisitor::VisitVarDecl(const VarDecl *Var) {
  if (Analyzer.isConsumableType(Var->getType())) {
    PropagationInfo PState =
      PropagationMap.find(Var->getInit())->second;
    
    StateMap->setState(Var, PState.IsVar ?
      StateMap->getState(PState.getVar()) : PState.getState());
  }
}
} // end anonymous::ConsumedStmtVisitor

namespace {

// TODO: Handle variable definitions, e.g. bool valid = x.isValid();
//       if (valid) ...; (Deferred)
class TestedVarsVisitor : public RecursiveASTVisitor<TestedVarsVisitor> {
  
  bool Invert;
  SourceLocation CurrTestLoc;
  
  ConsumedStateMap *StateMap;
  
public:
  bool IsUsefulConditional;
  VarTestResult Test;
  
  TestedVarsVisitor(ConsumedStateMap *StateMap) : Invert(false),
    StateMap(StateMap), IsUsefulConditional(false) {}
  
  bool VisitCallExpr(CallExpr *Call);
  bool VisitDeclRefExpr(DeclRefExpr *DeclRef);
  bool VisitUnaryOperator(UnaryOperator *UnaryOp);
};

bool TestedVarsVisitor::VisitCallExpr(CallExpr *Call) {
  if (const CXXMethodDecl *Method =
    dyn_cast_or_null<CXXMethodDecl>(Call->getDirectCallee())) {
    
    if (isTestingFunction(Method)) {
      CurrTestLoc = Call->getExprLoc();
      IsUsefulConditional = true;
      return true;
    }
    
    IsUsefulConditional = false;
  }
  
  return false;
}

bool TestedVarsVisitor::VisitDeclRefExpr(DeclRefExpr *DeclRef) {
  if (const VarDecl *Var = dyn_cast_or_null<VarDecl>(DeclRef->getDecl())) {
    if (StateMap->getState(Var) != consumed::CS_None) {
      Test = VarTestResult(Var, CurrTestLoc, !Invert);
    }
    
  } else {
    IsUsefulConditional = false;
  }
  
  return IsUsefulConditional;
}

bool TestedVarsVisitor::VisitUnaryOperator(UnaryOperator *UnaryOp) {
  if (UnaryOp->getOpcode() == UO_LNot) {
    Invert = true;
    TraverseStmt(UnaryOp->getSubExpr());
    
  } else {
    IsUsefulConditional = false;
  }
  
  return false;
}
} // end anonymouse::TestedVarsVisitor

namespace clang {
namespace consumed {

void ConsumedBlockInfo::addInfo(const CFGBlock *Block,
                                ConsumedStateMap *StateMap,
                                bool &AlreadyOwned) {
  
  if (VisitedBlocks.alreadySet(Block)) return;
  
  ConsumedStateMap *Entry = StateMapsArray[Block->getBlockID()];
    
  if (Entry) {
    Entry->intersect(StateMap);
    
  } else if (AlreadyOwned) {
    StateMapsArray[Block->getBlockID()] = new ConsumedStateMap(*StateMap);
    
  } else {
    StateMapsArray[Block->getBlockID()] = StateMap;
    AlreadyOwned = true;
  }
}

void ConsumedBlockInfo::addInfo(const CFGBlock *Block,
                                ConsumedStateMap *StateMap) {
  
  if (VisitedBlocks.alreadySet(Block)) {
    delete StateMap;
    return;
  }
  
  ConsumedStateMap *Entry = StateMapsArray[Block->getBlockID()];
    
  if (Entry) {
    Entry->intersect(StateMap);
    delete StateMap;
    
  } else {
    StateMapsArray[Block->getBlockID()] = StateMap;
  }
}

ConsumedStateMap* ConsumedBlockInfo::getInfo(const CFGBlock *Block) {
  return StateMapsArray[Block->getBlockID()];
}

void ConsumedBlockInfo::markVisited(const CFGBlock *Block) {
  VisitedBlocks.insert(Block);
}

ConsumedState ConsumedStateMap::getState(const VarDecl *Var) {
  MapType::const_iterator Entry = Map.find(Var);
  
  if (Entry != Map.end()) {
    return Entry->second;
    
  } else {
    return CS_None;
  }
}

void ConsumedStateMap::intersect(const ConsumedStateMap *Other) {
  ConsumedState LocalState;
  
  for (MapType::const_iterator DMI = Other->Map.begin(),
       DME = Other->Map.end(); DMI != DME; ++DMI) {
    
    LocalState = this->getState(DMI->first);
    
    if (LocalState != CS_None && LocalState != DMI->second)
      setState(DMI->first, CS_Unknown);
  }
}

void ConsumedStateMap::makeUnknown() {
  PairType Pair;
  
  for (MapType::const_iterator DMI = Map.begin(), DME = Map.end(); DMI != DME;
       ++DMI) {
    
    Pair = *DMI;
    
    Map.erase(Pair.first);
    Map.insert(PairType(Pair.first, CS_Unknown));
  }
}

void ConsumedStateMap::setState(const VarDecl *Var, ConsumedState State) {
  Map[Var] = State;
}

const Sema & ConsumedAnalyzer::getSema() {
  return S;
}


bool ConsumedAnalyzer::isConsumableType(QualType Type) {
  const CXXRecordDecl *RD =
    dyn_cast_or_null<CXXRecordDecl>(Type->getAsCXXRecordDecl());
  
  if (!RD) return false;
  
  std::pair<CacheMapType::iterator, bool> Entry =
    ConsumableTypeCache.insert(std::make_pair(RD, false));
  
  if (Entry.second)
    Entry.first->second = hasConsumableAttributes(RD);
  
  return Entry.first->second;
}

// TODO: Walk the base classes to see if any of them are unique types.
//       (Deferred)
bool ConsumedAnalyzer::hasConsumableAttributes(const CXXRecordDecl *RD) {
  for (CXXRecordDecl::method_iterator MI = RD->method_begin(),
       ME = RD->method_end(); MI != ME; ++MI) {
    
    for (Decl::attr_iterator AI = (*MI)->attr_begin(), AE = (*MI)->attr_end();
         AI != AE; ++AI) {
      
      switch ((*AI)->getKind()) {
      case attr::CallableWhenUnconsumed:
      case attr::TestsUnconsumed:
        return true;
      
      default:
        break;
      }
    }
  }
  
  return false;
}

// TODO: Handle other forms of branching with precision, including while- and
//       for-loops. (Deferred)
void ConsumedAnalyzer::splitState(const CFGBlock *CurrBlock,
                                  const IfStmt *Terminator) {
  
  TestedVarsVisitor Visitor(CurrStates);
  Visitor.TraverseStmt(const_cast<Expr*>(Terminator->getCond()));
  
  bool HasElse = Terminator->getElse() != NULL;
  
  ConsumedStateMap *ElseOrMergeStates = new ConsumedStateMap(*CurrStates);
  
  if (Visitor.IsUsefulConditional) {
    ConsumedState VarState = CurrStates->getState(Visitor.Test.Var);
    
    if (VarState != CS_Unknown) {
      // FIXME: Make this not warn if the test is from a macro expansion.
      //        (Deferred)
      WarningsHandler.warnUnnecessaryTest(Visitor.Test.Var->getNameAsString(),
        stateToString(VarState), Visitor.Test.Loc);
    }
    
    if (Visitor.Test.UnconsumedInTrueBranch) {
      CurrStates->setState(Visitor.Test.Var, CS_Unconsumed);
      if (HasElse) ElseOrMergeStates->setState(Visitor.Test.Var, CS_Consumed);
      
    } else {
      CurrStates->setState(Visitor.Test.Var, CS_Consumed);
      if (HasElse) ElseOrMergeStates->setState(Visitor.Test.Var, CS_Unconsumed);
    }
  }
    
  CFGBlock::const_succ_iterator SI = CurrBlock->succ_begin();
  
  if (*SI)   BlockInfo.addInfo(*SI,        CurrStates);
  if (*++SI) BlockInfo.addInfo(*SI, ElseOrMergeStates);
}

void ConsumedAnalyzer::run(AnalysisDeclContext &AC) {
  const FunctionDecl *D = dyn_cast_or_null<FunctionDecl>(AC.getDecl());
  
  if (!D) return;
  
  BlockInfo = ConsumedBlockInfo(AC.getCFG());
  
  PostOrderCFGView *SortedGraph = AC.getAnalysis<PostOrderCFGView>();
  
  CurrStates = new ConsumedStateMap();
  
  // Visit all of the function's basic blocks.
  for (PostOrderCFGView::iterator I = SortedGraph->begin(),
       E = SortedGraph->end(); I != E; ++I) {
    
    const CFGBlock *CurrBlock = *I;
    BlockInfo.markVisited(CurrBlock);
    
    if (CurrStates == NULL)
      CurrStates = BlockInfo.getInfo(CurrBlock);
    
    ConsumedStmtVisitor Visitor(*this, CurrStates);
    
    // Visit all of the basic block's statements.
    for (CFGBlock::const_iterator BI = CurrBlock->begin(),
         BE = CurrBlock->end(); BI != BE; ++BI) {
      
      if (BI->getKind() == CFGElement::Statement)
        Visitor.Visit(BI->castAs<CFGStmt>().getStmt());
    }
    
    // TODO: Remove any variables that have reached the end of their
    //       lifetimes from the state map. (Deferred)
    
    if (const IfStmt *Terminator =
      dyn_cast_or_null<IfStmt>(CurrBlock->getTerminator().getStmt())) {
      
      splitState(CurrBlock, Terminator);
      CurrStates = NULL;
    
    } else if (CurrBlock->succ_size() > 1) {
      CurrStates->makeUnknown();
      
      bool OwnershipTaken = false;
      
      for (CFGBlock::const_succ_iterator SI = CurrBlock->succ_begin(),
           SE = CurrBlock->succ_end(); SI != SE; ++SI) {
        
        if (*SI) BlockInfo.addInfo(*SI, CurrStates, OwnershipTaken);
      }
      
      if (!OwnershipTaken)
        delete CurrStates;
      
      CurrStates = NULL;
      
    } else if (CurrBlock->succ_size() == 1 &&
               (*CurrBlock->succ_begin())->pred_size() > 1) {
      
      BlockInfo.addInfo(*CurrBlock->succ_begin(), CurrStates);
      CurrStates = NULL;
    }
    
    Visitor.reset();
  } // End of block iterator.
  
  // Delete the last existing state map.
  delete CurrStates;
  
  WarningsHandler.emitDiagnostics();
}

unsigned checkEnabled(DiagnosticsEngine &D) {
  return (unsigned)
    (D.getDiagnosticLevel(diag::warn_use_while_consumed, SourceLocation()) !=
     DiagnosticsEngine::Ignored);
}

bool isTestingFunction(const CXXMethodDecl *Method) {
  return Method->hasAttr<TestsUnconsumedAttr>();
}

}} // end namespace clang::consumed
