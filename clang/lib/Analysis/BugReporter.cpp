// BugReporter.cpp - Generate PathDiagnostics for Bugs ------------*- C++ -*--//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines BugReporter, a utility class for generating
//  PathDiagnostics for analyses based on GRSimpleVals.
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/PathSensitive/BugReporter.h"
#include "clang/Analysis/PathSensitive/GRExprEngine.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/CFG.h"
#include "clang/AST/Expr.h"
#include "clang/Analysis/ProgramPoint.h"
#include "clang/Analysis/PathDiagnostic.h"
#include "llvm/ADT/DenseMap.h"
#include <sstream>

using namespace clang;

BugReporter::~BugReporter() {}
GRBugReporter::~GRBugReporter() {}
BugReporterData::~BugReporterData() {}
BugType::~BugType() {}
BugReport::~BugReport() {}
RangedBugReport::~RangedBugReport() {}

ExplodedGraph<ValueState>& GRBugReporter::getGraph() {
  return Eng.getGraph();
}

ValueStateManager& GRBugReporter::getStateManager() { 
  return Eng.getStateManager();
}

static inline Stmt* GetStmt(const ProgramPoint& P) {
  if (const PostStmt* PS = dyn_cast<PostStmt>(&P)) {
    return PS->getStmt();
  }
  else if (const BlockEdge* BE = dyn_cast<BlockEdge>(&P)) {
    return BE->getSrc()->getTerminator();
  }
  else if (const BlockEntrance* BE = dyn_cast<BlockEntrance>(&P)) {
    return BE->getFirstStmt();
  }
  
  assert (false && "Unsupported ProgramPoint.");
  return NULL;
}

static inline Stmt* GetStmt(const CFGBlock* B) {
  if (B->empty())
    return const_cast<Stmt*>(B->getTerminator());
  else
    return (*B)[0];
}

static inline ExplodedNode<ValueState>*
GetNextNode(ExplodedNode<ValueState>* N) {
  return N->pred_empty() ? NULL : *(N->pred_begin());
}

static Stmt* GetLastStmt(ExplodedNode<ValueState>* N) {
  assert (isa<BlockEntrance>(N->getLocation()));
  
  for (N = GetNextNode(N); N; N = GetNextNode(N)) {
    
    ProgramPoint P = N->getLocation();
    
    if (PostStmt* PS = dyn_cast<PostStmt>(&P))
      return PS->getStmt();
  }
  
  return NULL;
}

static void ExecutionContinues(std::ostringstream& os, SourceManager& SMgr,
                               Stmt* S) {
  
  if (!S)
    return;
  
  // Slow, but probably doesn't matter.
  if (os.str().empty())
    os << ' ';
  
  os << "Execution continues on line "
     << SMgr.getLogicalLineNumber(S->getLocStart()) << '.';
}


static inline void ExecutionContinues(std::ostringstream& os,
                                      SourceManager& SMgr,
                                      ExplodedNode<ValueState>* N) {

  ExecutionContinues(os, SMgr, GetStmt(N->getLocation()));
}

static inline void ExecutionContinues(std::ostringstream& os,
                                      SourceManager& SMgr,
                                      const CFGBlock* B) {  
  
  ExecutionContinues(os, SMgr, GetStmt(B));
}


Stmt* BugReport::getStmt(BugReporter& BR) const {
  
  ProgramPoint ProgP = EndNode->getLocation();  
  Stmt *S = NULL;
  
  if (BlockEntrance* BE = dyn_cast<BlockEntrance>(&ProgP))
    if (BE->getBlock() == &BR.getCFG()->getExit())
      S = GetLastStmt(EndNode);
  if (!S)
    S = GetStmt(ProgP);  

  return S;  
}

PathDiagnosticPiece*
BugReport::getEndPath(BugReporter& BR,
                      ExplodedNode<ValueState>* EndPathNode) {
  
  Stmt* S = getStmt(BR);
  
  if (!S)
    return NULL;
  
  FullSourceLoc L(S->getLocStart(), BR.getContext().getSourceManager());
  PathDiagnosticPiece* P = new PathDiagnosticPiece(L, getDescription());
  
  const SourceRange *Beg, *End;
  getRanges(BR, Beg, End);  

  for (; Beg != End; ++Beg)
    P->addRange(*Beg);
  
  return P;
}

void BugReport::getRanges(BugReporter& BR, const SourceRange*& beg,
                          const SourceRange*& end) {  
  
  if (Expr* E = dyn_cast_or_null<Expr>(getStmt(BR))) {
    R = E->getSourceRange();
    beg = &R;
    end = beg+1;
  }
  else
    beg = end = 0;
}

FullSourceLoc BugReport::getLocation(SourceManager& Mgr) {
  
  if (!EndNode)
    return FullSourceLoc();
  
  Stmt* S = GetStmt(EndNode->getLocation());
  
  if (!S)
    return FullSourceLoc();
  
  return FullSourceLoc(S->getLocStart(), Mgr);
}

PathDiagnosticPiece* BugReport::VisitNode(ExplodedNode<ValueState>* N,
                                          ExplodedNode<ValueState>* PrevN,
                                          ExplodedGraph<ValueState>& G,
                                          BugReporter& BR) {
  return NULL;
}

static std::pair<ExplodedGraph<ValueState>*, ExplodedNode<ValueState>*>
MakeReportGraph(ExplodedGraph<ValueState>* G, ExplodedNode<ValueState>* N) {
  
  llvm::OwningPtr<ExplodedGraph<ValueState> > GTrim(G->Trim(&N, &N+1));    
    
  // Find the error node in the trimmed graph.  
  
  ExplodedNode<ValueState>* NOld = N;
  N = 0;
  
  for (ExplodedGraph<ValueState>::node_iterator
       I = GTrim->nodes_begin(), E = GTrim->nodes_end(); I != E; ++I) {
    
    if (I->getState() == NOld->getState() &&
        I->getLocation() == NOld->getLocation()) {
      N = &*I;
      break;
    }    
  }
  
  assert(N);
    
  // Create a new graph with a single path.

  G = new ExplodedGraph<ValueState>(GTrim->getCFG(), GTrim->getCodeDecl(),
                                     GTrim->getContext());
                                     
  // Sometimes TrimGraph can contain a cycle.  Perform a reverse DFS
  // to the root node, and then construct a new graph that contains only
  // a single path.
  llvm::DenseMap<void*,unsigned> Visited;
  llvm::SmallVector<ExplodedNode<ValueState>*, 10> WS;
  WS.push_back(N);
  unsigned cnt = 0;
  ExplodedNode<ValueState>* Root = 0;
  
  while (!WS.empty()) {
    ExplodedNode<ValueState>* Node = WS.back();
    WS.pop_back();
    
    if (Visited.find(Node) != Visited.end())
      continue;
    
    Visited[Node] = cnt++;
    
    if (Node->pred_empty()) {
      Root = Node;
      break;
    }
    
    for (ExplodedNode<ValueState>::pred_iterator I=Node->pred_begin(),
         E=Node->pred_end(); I!=E; ++I)
      WS.push_back(*I);
  }

  assert (Root);
  
  // Now walk from the root down the DFS path, always taking the successor
  // with the lowest number.
  ExplodedNode<ValueState> *Last = 0, *First = 0;  
    
  for ( N = Root ;;) {
    
    // Lookup the number associated with the current node.
    llvm::DenseMap<void*,unsigned>::iterator I=Visited.find(N);
    assert (I != Visited.end());
    
    // Create the equivalent node in the new graph with the same state
    // and location.
    ExplodedNode<ValueState>* NewN =
      G->getNode(N->getLocation(), N->getState());    

    // Link up the new node with the previous node.
    if (Last)
      NewN->addPredecessor(Last);
    
    Last = NewN;

    // Are we at the final node?
    if (I->second == 0) {
      First = NewN;
      break;
    }

    // Find the next successor node.  We choose the node that is marked
    // with the lowest DFS number.
    ExplodedNode<ValueState>::succ_iterator SI = N->succ_begin();
    ExplodedNode<ValueState>::succ_iterator SE = N->succ_end();
    N = 0;
    
    for (unsigned MinVal = 0; SI != SE; ++SI) {

      I = Visited.find(*SI);
      
      if (I == Visited.end())
        continue;
      
      if (!N || I->second < MinVal) {
        N = *SI;
        MinVal = I->second;
      }
    }

    assert (N);
  }

  assert (First);
  return std::make_pair(G, First);
}

static VarDecl* GetMostRecentVarDeclBinding(ExplodedNode<ValueState>* N,
                                            ValueStateManager& VMgr,
                                            RVal X) {
  
  for ( ; N ; N = N->pred_empty() ? 0 : *N->pred_begin()) {
    
    ProgramPoint P = N->getLocation();

    if (!isa<PostStmt>(P))
      continue;
    
    DeclRefExpr* DR = dyn_cast<DeclRefExpr>(cast<PostStmt>(P).getStmt());

    if (!DR)
      continue;
    
    RVal Y = VMgr.GetRVal(N->getState(), DR);
    
    if (X != Y)
      continue;
    
    VarDecl* VD = dyn_cast<VarDecl>(DR->getDecl());
    
    if (!VD)
      continue;
    
    return VD;
  }
  
  return 0;
}


static void HandleNotableSymbol(ExplodedNode<ValueState>* N, Stmt* S,
                                SymbolID Sym, BugReporter& BR,
                                PathDiagnostic& PD) {
  
  ExplodedNode<ValueState>* Pred = N->pred_empty() ? 0 : *N->pred_begin();
  const ValueState* PrevSt = Pred ? Pred->getState() : 0;
  
  if (!PrevSt)
    return;
  
  // Look at the variable bindings of the current state that map to the
  // specified symbol.  Are any of them not in the previous state.
  
  const ValueState* St = N->getState();
  ValueStateManager& VMgr = cast<GRBugReporter>(BR).getStateManager();
  
  // FIXME: Later generalize for a broader memory model.

  // FIXME: This is quadratic, since its nested in another loop.  Probably
  //   doesn't matter, but keep an eye out for performance issues.  It's
  //   also a bunch of copy-paste.  Bad.  Cleanup later.
  
  for (ValueState::vb_iterator I=St->vb_begin(), E=St->vb_end(); I!=E; ++I){
    
    RVal V = I.getData();
    SymbolID ScanSym;
    
    if (lval::SymbolVal* SV = dyn_cast<lval::SymbolVal>(&V))
      ScanSym = SV->getSymbol();
    else if (nonlval::SymbolVal* SV = dyn_cast<nonlval::SymbolVal>(&V))
      ScanSym = SV->getSymbol();
    else
      continue;
    
    if (ScanSym != Sym)
      continue;
    
    // Check if the previous state has this binding.
    
    RVal X = VMgr.GetRVal(PrevSt, lval::DeclVal(I.getKey()));
    
    if (X == V) // Same binding?
      continue;

    // Different binding.  Only handle assignments for now.  We don't pull
    // this check out of the loop because we will eventually handle other 
    // cases.
    
    VarDecl *VD = 0;
    
    if (BinaryOperator* B = dyn_cast<BinaryOperator>(S)) {
      if (!B->isAssignmentOp())
        continue;
      
      // What variable did we assign to?
      DeclRefExpr* DR = dyn_cast<DeclRefExpr>(B->getLHS()->IgnoreParenCasts());
      
      if (!DR)
        continue;
      
      VD = dyn_cast<VarDecl>(DR->getDecl());
    }
    else if (DeclStmt* DS = dyn_cast<DeclStmt>(S))
      VD = dyn_cast<VarDecl>(DS->getDecl());
      
    if (!VD)
      continue;
      
    // What is the most recently referenced variable with this binding?
    VarDecl* MostRecent = GetMostRecentVarDeclBinding(Pred, VMgr, V);
        
    if (!MostRecent)
      continue;

    // Create the diagnostic.
    
    FullSourceLoc L(S->getLocStart(), BR.getSourceManager());
      
    if (VD->getType()->isPointerLikeType()) {
      std::string msg = "'" + std::string(VD->getName()) +
                        "' now aliases '" + MostRecent->getName() + "'";
      
      PD.push_front(new PathDiagnosticPiece(L, msg));
    }
  }
}

void GRBugReporter::GeneratePathDiagnostic(PathDiagnostic& PD,
                                           BugReport& R) {

  ExplodedNode<ValueState>* N = R.getEndNode();

  if (!N) return;
  
  // Construct a new graph that contains only a single path from the error
  // node to a root.
  
  const std::pair<ExplodedGraph<ValueState>*,ExplodedNode<ValueState>*>
    GPair = MakeReportGraph(&getGraph(), N);
  
  llvm::OwningPtr<ExplodedGraph<ValueState> > ReportGraph(GPair.first);
  assert(GPair.second->getLocation() == N->getLocation());
  N = GPair.second;

  // Start building the path diagnostic...
  
  if (PathDiagnosticPiece* Piece = R.getEndPath(*this, N))
    PD.push_back(Piece);
  else
    return;
  
  ExplodedNode<ValueState>* NextNode = N->pred_empty() 
                                       ? NULL : *(N->pred_begin());
  
  ASTContext& Ctx = getContext();
  SourceManager& SMgr = Ctx.getSourceManager();
  
  while (NextNode) {
    
    ExplodedNode<ValueState>* LastNode = N;
    N = NextNode;    
    NextNode = GetNextNode(N);
    
    ProgramPoint P = N->getLocation();
    
    if (const BlockEdge* BE = dyn_cast<BlockEdge>(&P)) {
      
      CFGBlock* Src = BE->getSrc();
      CFGBlock* Dst = BE->getDst();
      
      Stmt* T = Src->getTerminator();
      
      if (!T)
        continue;
      
      FullSourceLoc L(T->getLocStart(), SMgr);
      
      switch (T->getStmtClass()) {
        default:
          break;
          
        case Stmt::GotoStmtClass:
        case Stmt::IndirectGotoStmtClass: {
          
          Stmt* S = GetStmt(LastNode->getLocation());
          
          if (!S)
            continue;
          
          std::ostringstream os;
          
          os << "Control jumps to line "
             << SMgr.getLogicalLineNumber(S->getLocStart()) << ".\n";
          
          PD.push_front(new PathDiagnosticPiece(L, os.str()));
          break;
        }
          
        case Stmt::SwitchStmtClass: {
          
          // Figure out what case arm we took.

          std::ostringstream os;

          if (Stmt* S = Dst->getLabel())
            switch (S->getStmtClass()) {
                
            default:
              assert(false && "Not a valid switch label.");
              continue;
                
            case Stmt::DefaultStmtClass: {              
              
              os << "Control jumps to the 'default' case at line "
                 << SMgr.getLogicalLineNumber(S->getLocStart()) << ".\n";
              
              break;
            }
              
            case Stmt::CaseStmtClass: {
              
              os << "Control jumps to 'case ";
              
              CaseStmt* Case = cast<CaseStmt>(S);              
              Expr* LHS = Case->getLHS()->IgnoreParenCasts();
              
              // Determine if it is an enum.
              
              bool GetRawInt = true;
              
              if (DeclRefExpr* DR = dyn_cast<DeclRefExpr>(LHS)) {

                // FIXME: Maybe this should be an assertion.  Are there cases
                // were it is not an EnumConstantDecl?
                
                EnumConstantDecl* D = dyn_cast<EnumConstantDecl>(DR->getDecl());                
                
                if (D) {
                  GetRawInt = false;
                  os << D->getName();
                }
              }

              if (GetRawInt) {
              
                // Not an enum.
                Expr* CondE = cast<SwitchStmt>(T)->getCond();
                unsigned bits = Ctx.getTypeSize(CondE->getType());
                llvm::APSInt V(bits, false);
                
                if (!LHS->isIntegerConstantExpr(V, Ctx, 0, true)) {
                  assert (false && "Case condition must be constant.");
                  continue;
                }
                
                os << V.toString();
              }              
              
              os << ":'  at line " 
                 << SMgr.getLogicalLineNumber(S->getLocStart()) << ".\n";
              
              break;
              
            }
          }
          else {
            os << "'Default' branch taken.";
            ExecutionContinues(os, SMgr, LastNode);
          }
          
          PD.push_front(new PathDiagnosticPiece(L, os.str()));
          break;
        }
          
        case Stmt::BreakStmtClass:
        case Stmt::ContinueStmtClass: {
          std::ostringstream os;
          ExecutionContinues(os, SMgr, LastNode);
          PD.push_front(new PathDiagnosticPiece(L, os.str()));
          break;
        }

        case Stmt::ConditionalOperatorClass: {
          
          std::ostringstream os;
          os << "'?' condition evaluates to ";

          if (*(Src->succ_begin()+1) == Dst)
            os << "false.";
          else
            os << "true.";
          
          PD.push_front(new PathDiagnosticPiece(L, os.str()));
          
          break;
        }
          
        case Stmt::DoStmtClass:  {
          
          if (*(Src->succ_begin()) == Dst) {
            
            std::ostringstream os;          
            
            os << "Loop condition is true.";
            ExecutionContinues(os, SMgr, Dst);
            
            PD.push_front(new PathDiagnosticPiece(L, os.str()));
          }
          else
            PD.push_front(new PathDiagnosticPiece(L,
                              "Loop condition is false.  Exiting loop."));
          
          break;
        }
          
        case Stmt::WhileStmtClass:
        case Stmt::ForStmtClass: {
          
          if (*(Src->succ_begin()+1) == Dst) {
            
            std::ostringstream os;          

            os << "Loop condition is false.";
            ExecutionContinues(os, SMgr, Dst);
          
            PD.push_front(new PathDiagnosticPiece(L, os.str()));
          }
          else
            PD.push_front(new PathDiagnosticPiece(L,
                            "Loop condition is true.  Entering loop body."));
          
          break;
        }
          
        case Stmt::IfStmtClass: {
          
          if (*(Src->succ_begin()+1) == Dst)
            PD.push_front(new PathDiagnosticPiece(L, "Taking false branch."));
          else 
            PD.push_front(new PathDiagnosticPiece(L, "Taking true branch."));
          
          break;
        }
      }
    }

    if (PathDiagnosticPiece* p = R.VisitNode(N, NextNode, *ReportGraph, *this))
      PD.push_front(p);
    
    if (const PostStmt* PS = dyn_cast<PostStmt>(&P)) {
      
      const ValueState* St = N->getState();
      
      // Scan the lval bindings, and see if a "notable" symbol has a new
      // lval binding.
      
      // FIXME: In the future, when we generalize the memory model, we'll
      //  need a way to iterate over binded locations.
      
      llvm::SmallSet<SymbolID, 10> AlreadyProcessed;
      
      for (ValueState::vb_iterator I=St->vb_begin(), E=St->vb_end(); I!=E; ++I){
        
        RVal V = I.getData();
        SymbolID ScanSym;
        
        if (lval::SymbolVal* SV = dyn_cast<lval::SymbolVal>(&V))
          ScanSym = SV->getSymbol();
        else if (nonlval::SymbolVal* SV = dyn_cast<nonlval::SymbolVal>(&V))
          ScanSym = SV->getSymbol();
        else
          continue;
        
        assert (ScanSym.isInitialized());
        
        if (!isNotable(ScanSym))
          continue;
        
        if (AlreadyProcessed.count(ScanSym))
          continue;
        
        AlreadyProcessed.insert(ScanSym);
        
        HandleNotableSymbol(N, PS->getStmt(), ScanSym, *this, PD);        
      }
    }
  }
}


bool BugTypeCacheLocation::isCached(BugReport& R) {
  
  ExplodedNode<ValueState>* N = R.getEndNode();
  
  if (!N)
    return false;

  // Cache the location of the error.  Don't emit the same
  // warning for the same error type that occurs at the same program
  // location but along a different path.
  
  return isCached(N->getLocation());
}

bool BugTypeCacheLocation::isCached(ProgramPoint P) {
  
  void* p = P.getRawData();
  
  if (CachedErrors.count(p))
    return true;
  
  CachedErrors.insert(p);  
  return false;
}

void BugReporter::EmitWarning(BugReport& R) {

  if (R.getBugType().isCached(R))
    return;

  llvm::OwningPtr<PathDiagnostic> D(new PathDiagnostic(R.getName()));
  GeneratePathDiagnostic(*D.get(), R);
  
  // Get the meta data.
  
  std::pair<const char**, const char**> Meta = R.getExtraDescriptiveText();
  
  for (const char** s = Meta.first; s != Meta.second; ++s)
    D->addMeta(*s);

  // Emit a full diagnostic for the path if we have a PathDiagnosticClient.
  
  PathDiagnosticClient* PD = getPathDiagnosticClient();
  
  if (PD && !D->empty()) { 
    PD->HandlePathDiagnostic(D.take());
    return;    
  }
  
  // We don't have a PathDiagnosticClient, but we can still emit a single
  // line diagnostic.  Determine the location.
  
  FullSourceLoc L = D->empty() ? R.getLocation(getSourceManager())
                               : D->back()->getLocation();
  
  
  // Determine the range.
  
  const SourceRange *Beg, *End;
  
  if (!D->empty()) {
    Beg = D->back()->ranges_begin();
    End = D->back()->ranges_end();
  }
  else  
    R.getRanges(*this, Beg, End);

  if (PD) {
    PathDiagnosticPiece* piece = new PathDiagnosticPiece(L, R.getDescription());

    for ( ; Beg != End; ++Beg)
      piece->addRange(*Beg);

    D->push_back(piece);    
    PD->HandlePathDiagnostic(D.take());
  }
  else {
    std::ostringstream os;  
    
    if (D->empty())
      os << R.getDescription();
    else
      os << D->back()->getString();
    
    
    Diagnostic& Diag = getDiagnostic();
    unsigned ErrorDiag = Diag.getCustomDiagID(Diagnostic::Warning,
                                              os.str().c_str());

    Diag.Report(L, ErrorDiag, NULL, 0, Beg, End - Beg);
  }
}
