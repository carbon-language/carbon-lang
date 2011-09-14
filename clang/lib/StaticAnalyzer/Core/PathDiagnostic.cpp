//===--- PathDiagnostic.cpp - Path-Specific Diagnostic Handling -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the PathDiagnostic-related interfaces.
//
//===----------------------------------------------------------------------===//

#include "clang/StaticAnalyzer/Core/BugReporter/PathDiagnostic.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/ExplodedGraph.h"
#include "clang/AST/Expr.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/ParentMap.h"
#include "clang/AST/StmtCXX.h"
#include "llvm/ADT/SmallString.h"

using namespace clang;
using namespace ento;

bool PathDiagnosticMacroPiece::containsEvent() const {
  for (const_iterator I = begin(), E = end(); I!=E; ++I) {
    if (isa<PathDiagnosticEventPiece>(*I))
      return true;

    if (PathDiagnosticMacroPiece *MP = dyn_cast<PathDiagnosticMacroPiece>(*I))
      if (MP->containsEvent())
        return true;
  }

  return false;
}

static StringRef StripTrailingDots(StringRef s) {
  for (StringRef::size_type i = s.size(); i != 0; --i)
    if (s[i - 1] != '.')
      return s.substr(0, i);
  return "";
}

PathDiagnosticPiece::PathDiagnosticPiece(StringRef s,
                                         Kind k, DisplayHint hint)
  : str(StripTrailingDots(s)), kind(k), Hint(hint) {}

PathDiagnosticPiece::PathDiagnosticPiece(Kind k, DisplayHint hint)
  : kind(k), Hint(hint) {}

PathDiagnosticPiece::~PathDiagnosticPiece() {}
PathDiagnosticEventPiece::~PathDiagnosticEventPiece() {}
PathDiagnosticControlFlowPiece::~PathDiagnosticControlFlowPiece() {}

PathDiagnosticMacroPiece::~PathDiagnosticMacroPiece() {
  for (iterator I = begin(), E = end(); I != E; ++I) delete *I;
}

PathDiagnostic::PathDiagnostic() : Size(0) {}

PathDiagnostic::~PathDiagnostic() {
  for (iterator I = begin(), E = end(); I != E; ++I) delete &*I;
}

void PathDiagnostic::resetPath(bool deletePieces) {
  Size = 0;

  if (deletePieces)
    for (iterator I=begin(), E=end(); I!=E; ++I)
      delete &*I;

  path.clear();
}


PathDiagnostic::PathDiagnostic(StringRef bugtype, StringRef desc,
                               StringRef category)
  : Size(0),
    BugType(StripTrailingDots(bugtype)),
    Desc(StripTrailingDots(desc)),
    Category(StripTrailingDots(category)) {}

void PathDiagnosticClient::HandleDiagnostic(Diagnostic::Level DiagLevel,
                                            const DiagnosticInfo &Info) {
  // Default implementation (Warnings/errors count).
  DiagnosticClient::HandleDiagnostic(DiagLevel, Info);

  // Create a PathDiagnostic with a single piece.

  PathDiagnostic* D = new PathDiagnostic();

  const char *LevelStr;
  switch (DiagLevel) {
  default:
  case Diagnostic::Ignored: assert(0 && "Invalid diagnostic type");
  case Diagnostic::Note:    LevelStr = "note: "; break;
  case Diagnostic::Warning: LevelStr = "warning: "; break;
  case Diagnostic::Error:   LevelStr = "error: "; break;
  case Diagnostic::Fatal:   LevelStr = "fatal error: "; break;
  }

  llvm::SmallString<100> StrC;
  StrC += LevelStr;
  Info.FormatDiagnostic(StrC);

  PathDiagnosticPiece *P =
    new PathDiagnosticEventPiece(FullSourceLoc(Info.getLocation(),
                                               Info.getSourceManager()),
                                 StrC.str());

  for (unsigned i = 0, e = Info.getNumRanges(); i != e; ++i)
    P->addRange(Info.getRange(i).getAsRange());
  for (unsigned i = 0, e = Info.getNumFixItHints(); i != e; ++i)
    P->addFixItHint(Info.getFixItHint(i));
  D->push_front(P);

  HandlePathDiagnostic(D);
}

void PathDiagnosticClient::HandlePathDiagnostic(const PathDiagnostic *D) {
  // For now this simply forwards to HandlePathDiagnosticImpl.  In the future
  // we can use this indirection to control for multi-threaded access to
  // the PathDiagnosticClient from multiple bug reporters.
  HandlePathDiagnosticImpl(D);
}

//===----------------------------------------------------------------------===//
// PathDiagnosticLocation methods.
//===----------------------------------------------------------------------===//

PathDiagnosticLocation PathDiagnosticLocation::create(const ExplodedNode* N,
                                                      const SourceManager &SM) {
  assert(N && "Cannot create a location with a null node.");

  const ExplodedNode *NI = N;

  while (NI) {
    ProgramPoint P = NI->getLocation();
    const LocationContext *LC = P.getLocationContext();
    if (const StmtPoint *PS = dyn_cast<StmtPoint>(&P)) {
      return PathDiagnosticLocation(PS->getStmt(), SM, LC);
    }
    else if (const BlockEdge *BE = dyn_cast<BlockEdge>(&P)) {
      const Stmt *Term = BE->getSrc()->getTerminator();
      assert(Term);
      return PathDiagnosticLocation(Term, SM, LC);
    }
    NI = NI->succ_empty() ? 0 : *(NI->succ_begin());
  }

  const Decl &D = N->getCodeDecl();
  return PathDiagnosticLocation(D.getBodyRBrace(), SM);
}

static SourceLocation getValidSourceLocation(const Stmt* S,
                                             const LocationContext *LC) {
  SourceLocation L = S->getLocStart();

  // S might be a temporary statement that does not have a location in the
  // source code, so find an enclosing statement and use it's location.
  if (!L.isValid() && LC) {
    assert(LC);
    ParentMap & PM = LC->getParentMap();

    const Stmt *PS = S;
    while (!L.isValid()) {
      PS = PM.getParent(PS);
      L = PS->getLocStart();
    }
  }

  // TODO: either change the name or uncomment the assert.
  //assert(L.isValid());
  return L;
}

FullSourceLoc PathDiagnosticLocation::asLocation() const {
  assert(isValid());
  // Note that we want a 'switch' here so that the compiler can warn us in
  // case we add more cases.
  switch (K) {
    case SingleLocK:
    case RangeK:
      break;
    case StmtK:
      return FullSourceLoc(getValidSourceLocation(S, LC),
                           const_cast<SourceManager&>(*SM));
    case DeclK:
      return FullSourceLoc(D->getLocation(), const_cast<SourceManager&>(*SM));
  }

  return FullSourceLoc(R.getBegin(), const_cast<SourceManager&>(*SM));
}

PathDiagnosticRange PathDiagnosticLocation::asRange() const {
  assert(isValid());
  // Note that we want a 'switch' here so that the compiler can warn us in
  // case we add more cases.
  switch (K) {
    case SingleLocK:
      return PathDiagnosticRange(R, true);
    case RangeK:
      break;
    case StmtK: {
      const Stmt *S = asStmt();
      switch (S->getStmtClass()) {
        default:
          break;
        case Stmt::DeclStmtClass: {
          const DeclStmt *DS = cast<DeclStmt>(S);
          if (DS->isSingleDecl()) {
            // Should always be the case, but we'll be defensive.
            return SourceRange(DS->getLocStart(),
                               DS->getSingleDecl()->getLocation());
          }
          break;
        }
          // FIXME: Provide better range information for different
          //  terminators.
        case Stmt::IfStmtClass:
        case Stmt::WhileStmtClass:
        case Stmt::DoStmtClass:
        case Stmt::ForStmtClass:
        case Stmt::ChooseExprClass:
        case Stmt::IndirectGotoStmtClass:
        case Stmt::SwitchStmtClass:
        case Stmt::BinaryConditionalOperatorClass:
        case Stmt::ConditionalOperatorClass:
        case Stmt::ObjCForCollectionStmtClass: {
          SourceLocation L = getValidSourceLocation(S, LC);
          return SourceRange(L, L);
        }
      }

      return S->getSourceRange();
    }
    case DeclK:
      if (const ObjCMethodDecl *MD = dyn_cast<ObjCMethodDecl>(D))
        return MD->getSourceRange();
      if (const FunctionDecl *FD = dyn_cast<FunctionDecl>(D)) {
        if (Stmt *Body = FD->getBody())
          return Body->getSourceRange();
      }
      else {
        SourceLocation L = D->getLocation();
        return PathDiagnosticRange(SourceRange(L, L), true);
      }
  }

  return R;
}

void PathDiagnosticLocation::flatten() {
  if (K == StmtK) {
    R = asRange();
    K = RangeK;
    S = 0;
    D = 0;
  }
  else if (K == DeclK) {
    SourceLocation L = D->getLocation();
    R = SourceRange(L, L);
    K = SingleLocK;
    S = 0;
    D = 0;
  }
}

//===----------------------------------------------------------------------===//
// FoldingSet profiling methods.
//===----------------------------------------------------------------------===//

void PathDiagnosticLocation::Profile(llvm::FoldingSetNodeID &ID) const {
  ID.AddInteger((unsigned) K);
  switch (K) {
    case RangeK:
      ID.AddInteger(R.getBegin().getRawEncoding());
      ID.AddInteger(R.getEnd().getRawEncoding());
      break;      
    case SingleLocK:
      ID.AddInteger(R.getBegin().getRawEncoding());
      break;
    case StmtK:
      ID.Add(S);
      break;
    case DeclK:
      ID.Add(D);
      break;
  }
  return;
}

void PathDiagnosticPiece::Profile(llvm::FoldingSetNodeID &ID) const {
  ID.AddInteger((unsigned) getKind());
  ID.AddString(str);
  // FIXME: Add profiling support for code hints.
  ID.AddInteger((unsigned) getDisplayHint());
  for (range_iterator I = ranges_begin(), E = ranges_end(); I != E; ++I) {
    ID.AddInteger(I->getBegin().getRawEncoding());
    ID.AddInteger(I->getEnd().getRawEncoding());
  }  
}

void PathDiagnosticSpotPiece::Profile(llvm::FoldingSetNodeID &ID) const {
  PathDiagnosticPiece::Profile(ID);
  ID.Add(Pos);
}

void PathDiagnosticControlFlowPiece::Profile(llvm::FoldingSetNodeID &ID) const {
  PathDiagnosticPiece::Profile(ID);
  for (const_iterator I = begin(), E = end(); I != E; ++I)
    ID.Add(*I);
}

void PathDiagnosticMacroPiece::Profile(llvm::FoldingSetNodeID &ID) const {
  PathDiagnosticSpotPiece::Profile(ID);
  for (const_iterator I = begin(), E = end(); I != E; ++I)
    ID.Add(**I);
}

void PathDiagnostic::Profile(llvm::FoldingSetNodeID &ID) const {
  ID.AddInteger(Size);
  ID.AddString(BugType);
  ID.AddString(Desc);
  ID.AddString(Category);
  for (const_iterator I = begin(), E = end(); I != E; ++I)
    ID.Add(*I);
  
  for (meta_iterator I = meta_begin(), E = meta_end(); I != E; ++I)
    ID.AddString(*I);
}
