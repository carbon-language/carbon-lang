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
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/raw_ostream.h"

// TODO: Use information from tests in while-loop conditional.
// TODO: Add notes about the actual and expected state for 
// TODO: Correctly identify unreachable blocks when chaining boolean operators.
// TODO: Adjust the parser and AttributesList class to support lists of
//       identifiers.
// TODO: Warn about unreachable code.
// TODO: Switch to using a bitmap to track unreachable blocks.
// TODO: Handle variable definitions, e.g. bool valid = x.isValid();
//       if (valid) ...; (Deferred)
// TODO: Take notes on state transitions to provide better warning messages.
//       (Deferred)
// TODO: Test nested conditionals: A) Checking the same value multiple times,
//       and 2) Checking different values. (Deferred)

using namespace clang;
using namespace consumed;

// Key method definition
ConsumedWarningsHandlerBase::~ConsumedWarningsHandlerBase() {}

static SourceLocation getLastStmtLoc(const CFGBlock *Block) {
  // Find the source location of the last statement in the block, if the block
  // is not empty.
  if (const Stmt *StmtNode = Block->getTerminator()) {
    return StmtNode->getLocStart();
  } else {
    for (CFGBlock::const_reverse_iterator BI = Block->rbegin(),
         BE = Block->rend(); BI != BE; ++BI) {
      // FIXME: Handle other CFGElement kinds.
      if (Optional<CFGStmt> CS = BI->getAs<CFGStmt>())
        return CS->getStmt()->getLocStart();
    }
  }
  
  // The block is empty, and has a single predecessor. Use its exit location.
  assert(Block->pred_size() == 1 && *Block->pred_begin() &&
         Block->succ_size() != 0);
    
  return getLastStmtLoc(*Block->pred_begin());
}

static ConsumedState invertConsumedUnconsumed(ConsumedState State) {
  switch (State) {
  case CS_Unconsumed:
    return CS_Consumed;
  case CS_Consumed:
    return CS_Unconsumed;
  case CS_None:
    return CS_None;
  case CS_Unknown:
    return CS_Unknown;
  }
  llvm_unreachable("invalid enum");
}

static bool isCallableInState(const CallableWhenAttr *CWAttr,
                              ConsumedState State) {
  
  CallableWhenAttr::callableState_iterator I = CWAttr->callableState_begin(),
                                           E = CWAttr->callableState_end();
  
  for (; I != E; ++I) {
    
    ConsumedState MappedAttrState = CS_None;
    
    switch (*I) {
    case CallableWhenAttr::Unknown:
      MappedAttrState = CS_Unknown;
      break;
      
    case CallableWhenAttr::Unconsumed:
      MappedAttrState = CS_Unconsumed;
      break;
      
    case CallableWhenAttr::Consumed:
      MappedAttrState = CS_Consumed;
      break;
    }
    
    if (MappedAttrState == State)
      return true;
  }
  
  return false;
}

static bool isConsumableType(const QualType &QT) {
  if (const CXXRecordDecl *RD = QT->getAsCXXRecordDecl())
    return RD->hasAttr<ConsumableAttr>();
  else
    return false;
}

static bool isKnownState(ConsumedState State) {
  switch (State) {
  case CS_Unconsumed:
  case CS_Consumed:
    return true;
  case CS_None:
  case CS_Unknown:
    return false;
  }
  llvm_unreachable("invalid enum");
}

static bool isTestingFunction(const FunctionDecl *FunDecl) {
  return FunDecl->hasAttr<TestsTypestateAttr>();
}

static ConsumedState mapConsumableAttrState(const QualType QT) {
  assert(isConsumableType(QT));

  const ConsumableAttr *CAttr =
      QT->getAsCXXRecordDecl()->getAttr<ConsumableAttr>();

  switch (CAttr->getDefaultState()) {
  case ConsumableAttr::Unknown:
    return CS_Unknown;
  case ConsumableAttr::Unconsumed:
    return CS_Unconsumed;
  case ConsumableAttr::Consumed:
    return CS_Consumed;
  }
  llvm_unreachable("invalid enum");
}

static ConsumedState mapSetTypestateAttrState(const SetTypestateAttr *STAttr) {
  switch (STAttr->getNewState()) {
  case SetTypestateAttr::Unknown:
    return CS_Unknown;
  case SetTypestateAttr::Unconsumed:
    return CS_Unconsumed;
  case SetTypestateAttr::Consumed:
    return CS_Consumed;
  }
  llvm_unreachable("invalid_enum");
}

static ConsumedState
mapReturnTypestateAttrState(const ReturnTypestateAttr *RTSAttr) {
  switch (RTSAttr->getState()) {
  case ReturnTypestateAttr::Unknown:
    return CS_Unknown;
  case ReturnTypestateAttr::Unconsumed:
    return CS_Unconsumed;
  case ReturnTypestateAttr::Consumed:
    return CS_Consumed;
  }
  llvm_unreachable("invalid enum");
}

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
  llvm_unreachable("invalid enum");
}

static ConsumedState testsFor(const FunctionDecl *FunDecl) {
  assert(isTestingFunction(FunDecl));
  switch (FunDecl->getAttr<TestsTypestateAttr>()->getTestState()) {
  case TestsTypestateAttr::Unconsumed:
    return CS_Unconsumed;
  case TestsTypestateAttr::Consumed:
    return CS_Consumed;
  }
  llvm_unreachable("invalid enum");
}

namespace {
struct VarTestResult {
  const VarDecl *Var;
  ConsumedState TestsFor;
};
} // end anonymous::VarTestResult

namespace clang {
namespace consumed {

enum EffectiveOp {
  EO_And,
  EO_Or
};

class PropagationInfo {
  enum {
    IT_None,
    IT_State,
    IT_Test,
    IT_BinTest,
    IT_Var
  } InfoType;

  struct BinTestTy {
    const BinaryOperator *Source;
    EffectiveOp EOp;
    VarTestResult LTest;
    VarTestResult RTest;
  };
  
  union {
    ConsumedState State;
    VarTestResult Test;
    const VarDecl *Var;
    BinTestTy BinTest;
  };
  
  QualType TempType;
  
public:
  PropagationInfo() : InfoType(IT_None) {}
  
  PropagationInfo(const VarTestResult &Test) : InfoType(IT_Test), Test(Test) {}
  PropagationInfo(const VarDecl *Var, ConsumedState TestsFor)
    : InfoType(IT_Test) {
    
    Test.Var      = Var;
    Test.TestsFor = TestsFor;
  }
  
  PropagationInfo(const BinaryOperator *Source, EffectiveOp EOp,
                  const VarTestResult &LTest, const VarTestResult &RTest)
    : InfoType(IT_BinTest) {
    
    BinTest.Source  = Source;
    BinTest.EOp     = EOp;
    BinTest.LTest   = LTest;
    BinTest.RTest   = RTest;
  }
  
  PropagationInfo(const BinaryOperator *Source, EffectiveOp EOp,
                  const VarDecl *LVar, ConsumedState LTestsFor,
                  const VarDecl *RVar, ConsumedState RTestsFor)
    : InfoType(IT_BinTest) {
    
    BinTest.Source         = Source;
    BinTest.EOp            = EOp;
    BinTest.LTest.Var      = LVar;
    BinTest.LTest.TestsFor = LTestsFor;
    BinTest.RTest.Var      = RVar;
    BinTest.RTest.TestsFor = RTestsFor;
  }
  
  PropagationInfo(ConsumedState State, QualType TempType)
    : InfoType(IT_State), State(State), TempType(TempType) {}
  
  PropagationInfo(const VarDecl *Var) : InfoType(IT_Var), Var(Var) {}
  
  const ConsumedState & getState() const {
    assert(InfoType == IT_State);
    return State;
  }
  
  const QualType & getTempType() const {
    assert(InfoType == IT_State);
    return TempType;
  }
  
  const VarTestResult & getTest() const {
    assert(InfoType == IT_Test);
    return Test;
  }
  
  const VarTestResult & getLTest() const {
    assert(InfoType == IT_BinTest);
    return BinTest.LTest;
  }
  
  const VarTestResult & getRTest() const {
    assert(InfoType == IT_BinTest);
    return BinTest.RTest;
  }
  
  const VarDecl * getVar() const {
    assert(InfoType == IT_Var);
    return Var;
  }
  
  EffectiveOp testEffectiveOp() const {
    assert(InfoType == IT_BinTest);
    return BinTest.EOp;
  }
  
  const BinaryOperator * testSourceNode() const {
    assert(InfoType == IT_BinTest);
    return BinTest.Source;
  }
  
  bool isValid()   const { return InfoType != IT_None;     }
  bool isState()   const { return InfoType == IT_State;    }
  bool isTest()    const { return InfoType == IT_Test;     }
  bool isBinTest() const { return InfoType == IT_BinTest;  }
  bool isVar()     const { return InfoType == IT_Var;      }
  
  PropagationInfo invertTest() const {
    assert(InfoType == IT_Test || InfoType == IT_BinTest);
    
    if (InfoType == IT_Test) {
      return PropagationInfo(Test.Var, invertConsumedUnconsumed(Test.TestsFor));
    
    } else if (InfoType == IT_BinTest) {
      return PropagationInfo(BinTest.Source,
        BinTest.EOp == EO_And ? EO_Or : EO_And,
        BinTest.LTest.Var, invertConsumedUnconsumed(BinTest.LTest.TestsFor),
        BinTest.RTest.Var, invertConsumedUnconsumed(BinTest.RTest.TestsFor));
    } else {
      return PropagationInfo();
    }
  }
};

class ConsumedStmtVisitor : public ConstStmtVisitor<ConsumedStmtVisitor> {
  
  typedef llvm::DenseMap<const Stmt *, PropagationInfo> MapType;
  typedef std::pair<const Stmt *, PropagationInfo> PairType;
  typedef MapType::iterator InfoEntry;
  typedef MapType::const_iterator ConstInfoEntry;
  
  AnalysisDeclContext &AC;
  ConsumedAnalyzer &Analyzer;
  ConsumedStateMap *StateMap;
  MapType PropagationMap;
  void forwardInfo(const Stmt *From, const Stmt *To);
  bool isLikeMoveAssignment(const CXXMethodDecl *MethodDecl);
  void propagateReturnType(const Stmt *Call, const FunctionDecl *Fun,
                           QualType ReturnType);
  
public:
  void checkCallability(const PropagationInfo &PInfo,
                        const FunctionDecl *FunDecl,
                        SourceLocation BlameLoc);

  void Visit(const Stmt *StmtNode);
  
  void VisitBinaryOperator(const BinaryOperator *BinOp);
  void VisitCallExpr(const CallExpr *Call);
  void VisitCastExpr(const CastExpr *Cast);
  void VisitCXXBindTemporaryExpr(const CXXBindTemporaryExpr *Temp);
  void VisitCXXConstructExpr(const CXXConstructExpr *Call);
  void VisitCXXMemberCallExpr(const CXXMemberCallExpr *Call);
  void VisitCXXOperatorCallExpr(const CXXOperatorCallExpr *Call);
  void VisitDeclRefExpr(const DeclRefExpr *DeclRef);
  void VisitDeclStmt(const DeclStmt *DelcS);
  void VisitMaterializeTemporaryExpr(const MaterializeTemporaryExpr *Temp);
  void VisitMemberExpr(const MemberExpr *MExpr);
  void VisitParmVarDecl(const ParmVarDecl *Param);
  void VisitReturnStmt(const ReturnStmt *Ret);
  void VisitUnaryOperator(const UnaryOperator *UOp);
  void VisitVarDecl(const VarDecl *Var);

  ConsumedStmtVisitor(AnalysisDeclContext &AC, ConsumedAnalyzer &Analyzer,
                      ConsumedStateMap *StateMap)
      : AC(AC), Analyzer(Analyzer), StateMap(StateMap) {}
  
  PropagationInfo getInfo(const Stmt *StmtNode) const {
    ConstInfoEntry Entry = PropagationMap.find(StmtNode);
    
    if (Entry != PropagationMap.end())
      return Entry->second;
    else
      return PropagationInfo();
  }
  
  void reset(ConsumedStateMap *NewStateMap) {
    StateMap = NewStateMap;
  }
};

void ConsumedStmtVisitor::checkCallability(const PropagationInfo &PInfo,
                                           const FunctionDecl *FunDecl,
                                           SourceLocation BlameLoc) {
  
  if (!FunDecl->hasAttr<CallableWhenAttr>())
    return;
  
  const CallableWhenAttr *CWAttr = FunDecl->getAttr<CallableWhenAttr>();
  
  if (PInfo.isVar()) {
    const VarDecl *Var = PInfo.getVar();
    ConsumedState VarState = StateMap->getState(Var);
    
    assert(VarState != CS_None && "Invalid state");
    
    if (isCallableInState(CWAttr, VarState))
      return;
    
    Analyzer.WarningsHandler.warnUseInInvalidState(
      FunDecl->getNameAsString(), Var->getNameAsString(),
      stateToString(VarState), BlameLoc);
    
  } else if (PInfo.isState()) {
    
    assert(PInfo.getState() != CS_None && "Invalid state");
    
    if (isCallableInState(CWAttr, PInfo.getState()))
      return;
    
    Analyzer.WarningsHandler.warnUseOfTempInInvalidState(
      FunDecl->getNameAsString(), stateToString(PInfo.getState()), BlameLoc);
  }
}

void ConsumedStmtVisitor::forwardInfo(const Stmt *From, const Stmt *To) {
  InfoEntry Entry = PropagationMap.find(From);
  
  if (Entry != PropagationMap.end())
    PropagationMap.insert(PairType(To, Entry->second));
}

bool ConsumedStmtVisitor::isLikeMoveAssignment(
  const CXXMethodDecl *MethodDecl) {
  
  return MethodDecl->isMoveAssignmentOperator() ||
         (MethodDecl->getOverloadedOperator() == OO_Equal &&
          MethodDecl->getNumParams() == 1 &&
          MethodDecl->getParamDecl(0)->getType()->isRValueReferenceType());
}

void ConsumedStmtVisitor::propagateReturnType(const Stmt *Call,
                                              const FunctionDecl *Fun,
                                              QualType ReturnType) {
  if (isConsumableType(ReturnType)) {
    
    ConsumedState ReturnState;
    
    if (Fun->hasAttr<ReturnTypestateAttr>())
      ReturnState = mapReturnTypestateAttrState(
        Fun->getAttr<ReturnTypestateAttr>());
    else
      ReturnState = mapConsumableAttrState(ReturnType);
    
    PropagationMap.insert(PairType(Call,
      PropagationInfo(ReturnState, ReturnType)));
  }
}

void ConsumedStmtVisitor::Visit(const Stmt *StmtNode) {
  
  ConstStmtVisitor<ConsumedStmtVisitor>::Visit(StmtNode);
  
  for (Stmt::const_child_iterator CI = StmtNode->child_begin(),
       CE = StmtNode->child_end(); CI != CE; ++CI) {
    
    PropagationMap.erase(*CI);
  }
}

void ConsumedStmtVisitor::VisitBinaryOperator(const BinaryOperator *BinOp) {
  switch (BinOp->getOpcode()) {
  case BO_LAnd:
  case BO_LOr : {
    InfoEntry LEntry = PropagationMap.find(BinOp->getLHS()),
              REntry = PropagationMap.find(BinOp->getRHS());
    
    VarTestResult LTest, RTest;
    
    if (LEntry != PropagationMap.end() && LEntry->second.isTest()) {
      LTest = LEntry->second.getTest();
      
    } else {
      LTest.Var      = NULL;
      LTest.TestsFor = CS_None;
    }
    
    if (REntry != PropagationMap.end() && REntry->second.isTest()) {
      RTest = REntry->second.getTest();
      
    } else {
      RTest.Var      = NULL;
      RTest.TestsFor = CS_None;
    }
    
    if (!(LTest.Var == NULL && RTest.Var == NULL))
      PropagationMap.insert(PairType(BinOp, PropagationInfo(BinOp,
        static_cast<EffectiveOp>(BinOp->getOpcode() == BO_LOr), LTest, RTest)));
    
    break;
  }
    
  case BO_PtrMemD:
  case BO_PtrMemI:
    forwardInfo(BinOp->getLHS(), BinOp);
    break;
    
  default:
    break;
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
      
      if (Entry == PropagationMap.end() || !Entry->second.isVar()) {
        continue;
      }
      
      PropagationInfo PInfo = Entry->second;
      
      if (ParamType->isRValueReferenceType() ||
          (ParamType->isLValueReferenceType() &&
           !cast<LValueReferenceType>(*ParamType).isSpelledAsLValue())) {
        
        StateMap->setState(PInfo.getVar(), consumed::CS_Consumed);
        
      } else if (!(ParamType.isConstQualified() ||
                   ((ParamType->isReferenceType() ||
                     ParamType->isPointerType()) &&
                    ParamType->getPointeeType().isConstQualified()))) {
        
        StateMap->setState(PInfo.getVar(), consumed::CS_Unknown);
      }
    }
    
    QualType RetType = FunDecl->getCallResultType();
    if (RetType->isReferenceType())
      RetType = RetType->getPointeeType();
    
    propagateReturnType(Call, FunDecl, RetType);
  }
}

void ConsumedStmtVisitor::VisitCastExpr(const CastExpr *Cast) {
  forwardInfo(Cast->getSubExpr(), Cast);
}

void ConsumedStmtVisitor::VisitCXXBindTemporaryExpr(
  const CXXBindTemporaryExpr *Temp) {
  
  forwardInfo(Temp->getSubExpr(), Temp);
}

void ConsumedStmtVisitor::VisitCXXConstructExpr(const CXXConstructExpr *Call) {
  CXXConstructorDecl *Constructor = Call->getConstructor();

  ASTContext &CurrContext = AC.getASTContext();
  QualType ThisType = Constructor->getThisType(CurrContext)->getPointeeType();
  
  if (isConsumableType(ThisType)) {
    if (Constructor->isDefaultConstructor()) {
      
      PropagationMap.insert(PairType(Call,
        PropagationInfo(consumed::CS_Consumed, ThisType)));
      
    } else if (Constructor->isMoveConstructor()) {
      
      PropagationInfo PInfo =
        PropagationMap.find(Call->getArg(0))->second;
      
      if (PInfo.isVar()) {
        const VarDecl* Var = PInfo.getVar();
        
        PropagationMap.insert(PairType(Call,
          PropagationInfo(StateMap->getState(Var), ThisType)));
        
        StateMap->setState(Var, consumed::CS_Consumed);
        
      } else {
        PropagationMap.insert(PairType(Call, PInfo));
      }
        
    } else if (Constructor->isCopyConstructor()) {
      MapType::iterator Entry = PropagationMap.find(Call->getArg(0));
    
      if (Entry != PropagationMap.end())
        PropagationMap.insert(PairType(Call, Entry->second));
      
    } else {
      propagateReturnType(Call, Constructor, ThisType);
    }
  }
}


void ConsumedStmtVisitor::VisitCXXMemberCallExpr(
  const CXXMemberCallExpr *Call) {
  
  VisitCallExpr(Call);
  
  InfoEntry Entry = PropagationMap.find(Call->getCallee()->IgnoreParens());
  
  if (Entry != PropagationMap.end()) {
    PropagationInfo PInfo = Entry->second;
    const CXXMethodDecl *MethodDecl = Call->getMethodDecl();
    
    checkCallability(PInfo, MethodDecl, Call->getExprLoc());
    
    if (PInfo.isVar()) {
      if (isTestingFunction(MethodDecl))
        PropagationMap.insert(PairType(Call,
          PropagationInfo(PInfo.getVar(), testsFor(MethodDecl))));
      else if (MethodDecl->hasAttr<SetTypestateAttr>())
        StateMap->setState(PInfo.getVar(),
          mapSetTypestateAttrState(MethodDecl->getAttr<SetTypestateAttr>()));
    }
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
    
    PropagationInfo LPInfo, RPInfo;
    
    if (LEntry != PropagationMap.end() &&
        REntry != PropagationMap.end()) {
      
      LPInfo = LEntry->second;
      RPInfo = REntry->second;
      
      if (LPInfo.isVar() && RPInfo.isVar()) {
        StateMap->setState(LPInfo.getVar(),
          StateMap->getState(RPInfo.getVar()));
        
        StateMap->setState(RPInfo.getVar(), consumed::CS_Consumed);
        
        PropagationMap.insert(PairType(Call, LPInfo));
        
      } else if (LPInfo.isVar() && !RPInfo.isVar()) {
        StateMap->setState(LPInfo.getVar(), RPInfo.getState());
        
        PropagationMap.insert(PairType(Call, LPInfo));
        
      } else if (!LPInfo.isVar() && RPInfo.isVar()) {
        PropagationMap.insert(PairType(Call,
          PropagationInfo(StateMap->getState(RPInfo.getVar()),
                          LPInfo.getTempType())));
        
        StateMap->setState(RPInfo.getVar(), consumed::CS_Consumed);
        
      } else {
        PropagationMap.insert(PairType(Call, RPInfo));
      }
      
    } else if (LEntry != PropagationMap.end() &&
               REntry == PropagationMap.end()) {
      
      LPInfo = LEntry->second;
      
      if (LPInfo.isVar()) {
        StateMap->setState(LPInfo.getVar(), consumed::CS_Unknown);
        
        PropagationMap.insert(PairType(Call, LPInfo));
        
      } else if (LPInfo.isState()) {
        PropagationMap.insert(PairType(Call,
          PropagationInfo(consumed::CS_Unknown, LPInfo.getTempType())));
      }
      
    } else if (LEntry == PropagationMap.end() &&
               REntry != PropagationMap.end()) {
      
      if (REntry->second.isVar())
        StateMap->setState(REntry->second.getVar(), consumed::CS_Consumed);
    }
    
  } else {
    
    VisitCallExpr(Call);
    
    InfoEntry Entry = PropagationMap.find(Call->getArg(0));
    
    if (Entry != PropagationMap.end()) {
      PropagationInfo PInfo = Entry->second;
      
      checkCallability(PInfo, FunDecl, Call->getExprLoc());
      
      if (PInfo.isVar()) {
        if (isTestingFunction(FunDecl))
          PropagationMap.insert(PairType(Call,
            PropagationInfo(PInfo.getVar(), testsFor(FunDecl))));
        else if (FunDecl->hasAttr<SetTypestateAttr>())
          StateMap->setState(PInfo.getVar(),
            mapSetTypestateAttrState(FunDecl->getAttr<SetTypestateAttr>()));
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


void ConsumedStmtVisitor::VisitParmVarDecl(const ParmVarDecl *Param) {
  QualType ParamType = Param->getType();
  ConsumedState ParamState = consumed::CS_None;

  if (!(ParamType->isPointerType() || ParamType->isReferenceType()) &&
      isConsumableType(ParamType))
    ParamState = mapConsumableAttrState(ParamType);
  else if (ParamType->isReferenceType() &&
           isConsumableType(ParamType->getPointeeType()))
    ParamState = consumed::CS_Unknown;

  if (ParamState)
    StateMap->setState(Param, ParamState);
}

void ConsumedStmtVisitor::VisitReturnStmt(const ReturnStmt *Ret) {
  if (ConsumedState ExpectedState = Analyzer.getExpectedReturnState()) {
    InfoEntry Entry = PropagationMap.find(Ret->getRetValue());
    
    if (Entry != PropagationMap.end()) {
      assert(Entry->second.isState() || Entry->second.isVar());
       
      ConsumedState RetState = Entry->second.isState() ?
        Entry->second.getState() : StateMap->getState(Entry->second.getVar());
        
      if (RetState != ExpectedState)
        Analyzer.WarningsHandler.warnReturnTypestateMismatch(
          Ret->getReturnLoc(), stateToString(ExpectedState),
          stateToString(RetState));
    }
  }
}

void ConsumedStmtVisitor::VisitUnaryOperator(const UnaryOperator *UOp) {
  InfoEntry Entry = PropagationMap.find(UOp->getSubExpr()->IgnoreParens());
  if (Entry == PropagationMap.end()) return;
  
  switch (UOp->getOpcode()) {
  case UO_AddrOf:
    PropagationMap.insert(PairType(UOp, Entry->second));
    break;
  
  case UO_LNot:
    if (Entry->second.isTest() || Entry->second.isBinTest())
      PropagationMap.insert(PairType(UOp, Entry->second.invertTest()));
    break;
  
  default:
    break;
  }
}

// TODO: See if I need to check for reference types here.
void ConsumedStmtVisitor::VisitVarDecl(const VarDecl *Var) {
  if (isConsumableType(Var->getType())) {
    if (Var->hasInit()) {
      PropagationInfo PInfo =
        PropagationMap.find(Var->getInit())->second;
      
      StateMap->setState(Var, PInfo.isVar() ?
        StateMap->getState(PInfo.getVar()) : PInfo.getState());
      
    } else {
      StateMap->setState(Var, consumed::CS_Unknown);
    }
  }
}
}} // end clang::consumed::ConsumedStmtVisitor

namespace clang {
namespace consumed {

void splitVarStateForIf(const IfStmt * IfNode, const VarTestResult &Test,
                        ConsumedStateMap *ThenStates,
                        ConsumedStateMap *ElseStates) {

  ConsumedState VarState = ThenStates->getState(Test.Var);
  
  if (VarState == CS_Unknown) {
    ThenStates->setState(Test.Var, Test.TestsFor);
    ElseStates->setState(Test.Var, invertConsumedUnconsumed(Test.TestsFor));
  
  } else if (VarState == invertConsumedUnconsumed(Test.TestsFor)) {
    ThenStates->markUnreachable();
    
  } else if (VarState == Test.TestsFor) {
    ElseStates->markUnreachable();
  }
}

void splitVarStateForIfBinOp(const PropagationInfo &PInfo,
  ConsumedStateMap *ThenStates, ConsumedStateMap *ElseStates) {
  
  const VarTestResult &LTest = PInfo.getLTest(),
                      &RTest = PInfo.getRTest();
  
  ConsumedState LState = LTest.Var ? ThenStates->getState(LTest.Var) : CS_None,
                RState = RTest.Var ? ThenStates->getState(RTest.Var) : CS_None;
  
  if (LTest.Var) {
    if (PInfo.testEffectiveOp() == EO_And) {
      if (LState == CS_Unknown) {
        ThenStates->setState(LTest.Var, LTest.TestsFor);
        
      } else if (LState == invertConsumedUnconsumed(LTest.TestsFor)) {
        ThenStates->markUnreachable();
        
      } else if (LState == LTest.TestsFor && isKnownState(RState)) {
        if (RState == RTest.TestsFor)
          ElseStates->markUnreachable();
        else
          ThenStates->markUnreachable();
      }
      
    } else {
      if (LState == CS_Unknown) {
        ElseStates->setState(LTest.Var,
                             invertConsumedUnconsumed(LTest.TestsFor));
      
      } else if (LState == LTest.TestsFor) {
        ElseStates->markUnreachable();
        
      } else if (LState == invertConsumedUnconsumed(LTest.TestsFor) &&
                 isKnownState(RState)) {
        
        if (RState == RTest.TestsFor)
          ElseStates->markUnreachable();
        else
          ThenStates->markUnreachable();
      }
    }
  }
  
  if (RTest.Var) {
    if (PInfo.testEffectiveOp() == EO_And) {
      if (RState == CS_Unknown)
        ThenStates->setState(RTest.Var, RTest.TestsFor);
      else if (RState == invertConsumedUnconsumed(RTest.TestsFor))
        ThenStates->markUnreachable();
      
    } else {
      if (RState == CS_Unknown)
        ElseStates->setState(RTest.Var,
                             invertConsumedUnconsumed(RTest.TestsFor));
      else if (RState == RTest.TestsFor)
        ElseStates->markUnreachable();
    }
  }
}

bool ConsumedBlockInfo::allBackEdgesVisited(const CFGBlock *CurrBlock,
                                            const CFGBlock *TargetBlock) {
  
  assert(CurrBlock && "Block pointer must not be NULL");
  assert(TargetBlock && "TargetBlock pointer must not be NULL");
  
  unsigned int CurrBlockOrder = VisitOrder[CurrBlock->getBlockID()];
  for (CFGBlock::const_pred_iterator PI = TargetBlock->pred_begin(),
       PE = TargetBlock->pred_end(); PI != PE; ++PI) {
    if (*PI && CurrBlockOrder < VisitOrder[(*PI)->getBlockID()] )
      return false;
  }
  return true;
}

void ConsumedBlockInfo::addInfo(const CFGBlock *Block,
                                ConsumedStateMap *StateMap,
                                bool &AlreadyOwned) {
  
  assert(Block && "Block pointer must not be NULL");
  
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
  
  assert(Block != NULL && "Block pointer must not be NULL");
  
  ConsumedStateMap *Entry = StateMapsArray[Block->getBlockID()];
    
  if (Entry) {
    Entry->intersect(StateMap);
    delete StateMap;
    
  } else {
    StateMapsArray[Block->getBlockID()] = StateMap;
  }
}

ConsumedStateMap* ConsumedBlockInfo::borrowInfo(const CFGBlock *Block) {
  assert(Block && "Block pointer must not be NULL");
  assert(StateMapsArray[Block->getBlockID()] && "Block has no block info");
  
  return StateMapsArray[Block->getBlockID()];
}

void ConsumedBlockInfo::discardInfo(const CFGBlock *Block) {
  unsigned int BlockID = Block->getBlockID();
  delete StateMapsArray[BlockID];
  StateMapsArray[BlockID] = NULL;
}

ConsumedStateMap* ConsumedBlockInfo::getInfo(const CFGBlock *Block) {
  assert(Block && "Block pointer must not be NULL");
  
  ConsumedStateMap *StateMap = StateMapsArray[Block->getBlockID()];
  if (isBackEdgeTarget(Block)) {
    return new ConsumedStateMap(*StateMap);
  } else {
    StateMapsArray[Block->getBlockID()] = NULL;
    return StateMap;
  }
}

bool ConsumedBlockInfo::isBackEdge(const CFGBlock *From, const CFGBlock *To) {
  assert(From && "From block must not be NULL");
  assert(To   && "From block must not be NULL");
  
  return VisitOrder[From->getBlockID()] > VisitOrder[To->getBlockID()];
}

bool ConsumedBlockInfo::isBackEdgeTarget(const CFGBlock *Block) {
  assert(Block != NULL && "Block pointer must not be NULL");
  
  // Anything with less than two predecessors can't be the target of a back
  // edge.
  if (Block->pred_size() < 2)
    return false;
  
  unsigned int BlockVisitOrder = VisitOrder[Block->getBlockID()];
  for (CFGBlock::const_pred_iterator PI = Block->pred_begin(),
       PE = Block->pred_end(); PI != PE; ++PI) {
    if (*PI && BlockVisitOrder < VisitOrder[(*PI)->getBlockID()])
      return true;
  }
  return false;
}

ConsumedState ConsumedStateMap::getState(const VarDecl *Var) const {
  MapType::const_iterator Entry = Map.find(Var);
  
  if (Entry != Map.end()) {
    return Entry->second;
    
  } else {
    return CS_None;
  }
}

void ConsumedStateMap::intersect(const ConsumedStateMap *Other) {
  ConsumedState LocalState;
  
  if (this->From && this->From == Other->From && !Other->Reachable) {
    this->markUnreachable();
    return;
  }
  
  for (MapType::const_iterator DMI = Other->Map.begin(), DME = Other->Map.end();
       DMI != DME; ++DMI) {
    
    LocalState = this->getState(DMI->first);
    
    if (LocalState == CS_None)
      continue;
    
    if (LocalState != DMI->second)
       Map[DMI->first] = CS_Unknown;
  }
}

void ConsumedStateMap::intersectAtLoopHead(const CFGBlock *LoopHead,
  const CFGBlock *LoopBack, const ConsumedStateMap *LoopBackStates,
  ConsumedWarningsHandlerBase &WarningsHandler) {
  
  ConsumedState LocalState;
  SourceLocation BlameLoc = getLastStmtLoc(LoopBack);
  
  for (MapType::const_iterator DMI = LoopBackStates->Map.begin(),
       DME = LoopBackStates->Map.end(); DMI != DME; ++DMI) {
    
    LocalState = this->getState(DMI->first);
    
    if (LocalState == CS_None)
      continue;
    
    if (LocalState != DMI->second) {
      Map[DMI->first] = CS_Unknown;
      WarningsHandler.warnLoopStateMismatch(
        BlameLoc, DMI->first->getNameAsString());
    }
  }
}

void ConsumedStateMap::markUnreachable() {
  this->Reachable = false;
  Map.clear();
}

void ConsumedStateMap::setState(const VarDecl *Var, ConsumedState State) {
  Map[Var] = State;
}

void ConsumedStateMap::remove(const VarDecl *Var) {
  Map.erase(Var);
}

bool ConsumedStateMap::operator!=(const ConsumedStateMap *Other) const {
  for (MapType::const_iterator DMI = Other->Map.begin(), DME = Other->Map.end();
       DMI != DME; ++DMI) {
    
    if (this->getState(DMI->first) != DMI->second)
      return true;
  }
  
  return false;
}

void ConsumedAnalyzer::determineExpectedReturnState(AnalysisDeclContext &AC,
                                                    const FunctionDecl *D) {
  QualType ReturnType;
  if (const CXXConstructorDecl *Constructor = dyn_cast<CXXConstructorDecl>(D)) {
    ASTContext &CurrContext = AC.getASTContext();
    ReturnType = Constructor->getThisType(CurrContext)->getPointeeType();
  } else
    ReturnType = D->getCallResultType();

  if (D->hasAttr<ReturnTypestateAttr>()) {
    const ReturnTypestateAttr *RTSAttr = D->getAttr<ReturnTypestateAttr>();

    const CXXRecordDecl *RD = ReturnType->getAsCXXRecordDecl();
    if (!RD || !RD->hasAttr<ConsumableAttr>()) {
      // FIXME: This should be removed when template instantiation propagates
      //        attributes at template specialization definition, not
      //        declaration. When it is removed the test needs to be enabled
      //        in SemaDeclAttr.cpp.
      WarningsHandler.warnReturnTypestateForUnconsumableType(
          RTSAttr->getLocation(), ReturnType.getAsString());
      ExpectedReturnState = CS_None;
    } else
      ExpectedReturnState = mapReturnTypestateAttrState(RTSAttr);
  } else if (isConsumableType(ReturnType))
    ExpectedReturnState = mapConsumableAttrState(ReturnType);
  else
    ExpectedReturnState = CS_None;
}

bool ConsumedAnalyzer::splitState(const CFGBlock *CurrBlock,
                                  const ConsumedStmtVisitor &Visitor) {
  
  ConsumedStateMap *FalseStates = new ConsumedStateMap(*CurrStates);
  PropagationInfo PInfo;
  
  if (const IfStmt *IfNode =
    dyn_cast_or_null<IfStmt>(CurrBlock->getTerminator().getStmt())) {
    
    const Stmt *Cond = IfNode->getCond();
    
    PInfo = Visitor.getInfo(Cond);
    if (!PInfo.isValid() && isa<BinaryOperator>(Cond))
      PInfo = Visitor.getInfo(cast<BinaryOperator>(Cond)->getRHS());
    
    if (PInfo.isTest()) {
      CurrStates->setSource(Cond);
      FalseStates->setSource(Cond);
      splitVarStateForIf(IfNode, PInfo.getTest(), CurrStates, FalseStates);
      
    } else if (PInfo.isBinTest()) {
      CurrStates->setSource(PInfo.testSourceNode());
      FalseStates->setSource(PInfo.testSourceNode());
      splitVarStateForIfBinOp(PInfo, CurrStates, FalseStates);
      
    } else {
      delete FalseStates;
      return false;
    }
    
  } else if (const BinaryOperator *BinOp =
    dyn_cast_or_null<BinaryOperator>(CurrBlock->getTerminator().getStmt())) {
    
    PInfo = Visitor.getInfo(BinOp->getLHS());
    if (!PInfo.isTest()) {
      if ((BinOp = dyn_cast_or_null<BinaryOperator>(BinOp->getLHS()))) {
        PInfo = Visitor.getInfo(BinOp->getRHS());
        
        if (!PInfo.isTest()) {
          delete FalseStates;
          return false;
        }
        
      } else {
        delete FalseStates;
        return false;
      }
    }
    
    CurrStates->setSource(BinOp);
    FalseStates->setSource(BinOp);
    
    const VarTestResult &Test = PInfo.getTest();
    ConsumedState VarState = CurrStates->getState(Test.Var);
    
    if (BinOp->getOpcode() == BO_LAnd) {
      if (VarState == CS_Unknown)
        CurrStates->setState(Test.Var, Test.TestsFor);
      else if (VarState == invertConsumedUnconsumed(Test.TestsFor))
        CurrStates->markUnreachable();
      
    } else if (BinOp->getOpcode() == BO_LOr) {
      if (VarState == CS_Unknown)
        FalseStates->setState(Test.Var,
                              invertConsumedUnconsumed(Test.TestsFor));
      else if (VarState == Test.TestsFor)
        FalseStates->markUnreachable();
    }
    
  } else {
    delete FalseStates;
    return false;
  }
  
  CFGBlock::const_succ_iterator SI = CurrBlock->succ_begin();
  
  if (*SI)
    BlockInfo.addInfo(*SI, CurrStates);
  else
    delete CurrStates;
    
  if (*++SI)
    BlockInfo.addInfo(*SI, FalseStates);
  else
    delete FalseStates;
  
  CurrStates = NULL;
  return true;
}

void ConsumedAnalyzer::run(AnalysisDeclContext &AC) {
  const FunctionDecl *D = dyn_cast_or_null<FunctionDecl>(AC.getDecl());
  if (!D)
    return;
  
  CFG *CFGraph = AC.getCFG();
  if (!CFGraph)
    return;
  
  determineExpectedReturnState(AC, D);

  PostOrderCFGView *SortedGraph = AC.getAnalysis<PostOrderCFGView>();
  // AC.getCFG()->viewCFG(LangOptions());
  
  BlockInfo = ConsumedBlockInfo(CFGraph->getNumBlockIDs(), SortedGraph);
  
  CurrStates = new ConsumedStateMap();
  ConsumedStmtVisitor Visitor(AC, *this, CurrStates);
  
  // Add all trackable parameters to the state map.
  for (FunctionDecl::param_const_iterator PI = D->param_begin(),
       PE = D->param_end(); PI != PE; ++PI) {
    Visitor.VisitParmVarDecl(*PI);
  }
  
  // Visit all of the function's basic blocks.
  for (PostOrderCFGView::iterator I = SortedGraph->begin(),
       E = SortedGraph->end(); I != E; ++I) {
    
    const CFGBlock *CurrBlock = *I;
    
    if (CurrStates == NULL)
      CurrStates = BlockInfo.getInfo(CurrBlock);
    
    if (!CurrStates) {
      continue;
      
    } else if (!CurrStates->isReachable()) {
      delete CurrStates;
      CurrStates = NULL;
      continue;
    }
    
    Visitor.reset(CurrStates);
    
    // Visit all of the basic block's statements.
    for (CFGBlock::const_iterator BI = CurrBlock->begin(),
         BE = CurrBlock->end(); BI != BE; ++BI) {
      
      switch (BI->getKind()) {
      case CFGElement::Statement:
        Visitor.Visit(BI->castAs<CFGStmt>().getStmt());
        break;
        
      case CFGElement::TemporaryDtor: {
        const CFGTemporaryDtor DTor = BI->castAs<CFGTemporaryDtor>();
        const CXXBindTemporaryExpr *BTE = DTor.getBindTemporaryExpr();
        PropagationInfo PInfo = Visitor.getInfo(BTE);
        
        if (PInfo.isValid())
          Visitor.checkCallability(PInfo,
                                   DTor.getDestructorDecl(AC.getASTContext()),
                                   BTE->getExprLoc());
        break;
      }
      
      case CFGElement::AutomaticObjectDtor: {
        const CFGAutomaticObjDtor DTor = BI->castAs<CFGAutomaticObjDtor>();
        
        const VarDecl *Var = DTor.getVarDecl();
        ConsumedState VarState = CurrStates->getState(Var);
        
        if (VarState != CS_None) {
          PropagationInfo PInfo(Var);
          
          Visitor.checkCallability(PInfo,
                                   DTor.getDestructorDecl(AC.getASTContext()),
                                   getLastStmtLoc(CurrBlock));
          
          CurrStates->remove(Var);
        }
        break;
      }
      
      default:
        break;
      }
    }
    
    // TODO: Handle other forms of branching with precision, including while-
    //       and for-loops. (Deferred)
    if (!splitState(CurrBlock, Visitor)) {
      CurrStates->setSource(NULL);
      
      if (CurrBlock->succ_size() > 1 ||
          (CurrBlock->succ_size() == 1 &&
           (*CurrBlock->succ_begin())->pred_size() > 1)) {
        
        bool OwnershipTaken = false;
        
        for (CFGBlock::const_succ_iterator SI = CurrBlock->succ_begin(),
             SE = CurrBlock->succ_end(); SI != SE; ++SI) {
          
          if (*SI == NULL) continue;
          
          if (BlockInfo.isBackEdge(CurrBlock, *SI)) {
            BlockInfo.borrowInfo(*SI)->intersectAtLoopHead(*SI, CurrBlock,
                                                           CurrStates,
                                                           WarningsHandler);
            
            if (BlockInfo.allBackEdgesVisited(*SI, CurrBlock))
              BlockInfo.discardInfo(*SI);
          } else {
            BlockInfo.addInfo(*SI, CurrStates, OwnershipTaken);
          }
        }
        
        if (!OwnershipTaken)
          delete CurrStates;
        
        CurrStates = NULL;
      }
    }
  } // End of block iterator.
  
  // Delete the last existing state map.
  delete CurrStates;
  
  WarningsHandler.emitDiagnostics();
}
}} // end namespace clang::consumed
