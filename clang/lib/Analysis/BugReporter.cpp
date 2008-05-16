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
#include <sstream>

using namespace clang;

BugReporter::~BugReporter() {}
BugType::~BugType() {}
BugReport::~BugReport() {}
RangedBugReport::~RangedBugReport() {}

ExplodedGraph<ValueState>& BugReporter::getGraph() { return Eng.getGraph(); }

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
    if (BE->getBlock() == &BR.getCFG().getExit())
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
                                     
                                     
  ExplodedNode<ValueState> *Last = 0, *First = 0;

  while (N) {
    ExplodedNode<ValueState>* NewN =
      G->getNode(N->getLocation(), N->getState());
    
    if (!First) First = NewN;
    if (Last) Last->addPredecessor(NewN);
    
    Last = NewN;
    N = N->pred_empty() ? 0 : *(N->pred_begin());
  }
  
  return std::make_pair(G, First);
}

void BugReporter::GeneratePathDiagnostic(PathDiagnostic& PD,
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
  
  if (PD && !D->empty()) { 
    PD->HandlePathDiagnostic(D.take());
    return;    
  }
  
  // We don't have a PathDiagnosticClient, but we can still emit a single
  // line diagnostic.  Determine the location.
  
  FullSourceLoc L = D->empty() ? R.getLocation(Ctx.getSourceManager())
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
    os << "[CHECKER] ";
    
    if (D->empty())
      os << R.getDescription();
    else
      os << D->back()->getString();
    
    
    unsigned ErrorDiag = Diag.getCustomDiagID(Diagnostic::Warning,
                                              os.str().c_str());

    Diag.Report(L, ErrorDiag, NULL, 0, Beg, End - Beg);
  }
}
