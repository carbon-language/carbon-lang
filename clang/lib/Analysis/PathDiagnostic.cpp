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

#include "clang/Analysis/PathDiagnostic.h"
#include "clang/AST/Expr.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/StmtCXX.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/Casting.h"
#include <sstream>
using namespace clang;
using llvm::dyn_cast;
using llvm::isa;

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

static size_t GetNumCharsToLastNonPeriod(const char *s) {
  const char *start = s;
  const char *lastNonPeriod = 0;  

  for ( ; *s != '\0' ; ++s)
    if (*s != '.') lastNonPeriod = s;
  
  if (!lastNonPeriod)
    return 0;
  
  return (lastNonPeriod - start) + 1;
}

static inline size_t GetNumCharsToLastNonPeriod(const std::string &s) {
  return s.empty () ? 0 : GetNumCharsToLastNonPeriod(&s[0]);
}

PathDiagnosticPiece::PathDiagnosticPiece(const std::string& s,
                                         Kind k, DisplayHint hint)
  : str(s, 0, GetNumCharsToLastNonPeriod(s)), kind(k), Hint(hint) {}

PathDiagnosticPiece::PathDiagnosticPiece(const char* s, Kind k,
                                         DisplayHint hint)
  : str(s, GetNumCharsToLastNonPeriod(s)), kind(k), Hint(hint) {}

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


PathDiagnostic::PathDiagnostic(const char* bugtype, const char* desc,
                               const char* category)
  : Size(0),
    BugType(bugtype, GetNumCharsToLastNonPeriod(bugtype)),
    Desc(desc, GetNumCharsToLastNonPeriod(desc)),
    Category(category, GetNumCharsToLastNonPeriod(category)) {}

PathDiagnostic::PathDiagnostic(const std::string& bugtype,
                               const std::string& desc, 
                               const std::string& category)
  : Size(0),
    BugType(bugtype, 0, GetNumCharsToLastNonPeriod(bugtype)),
    Desc(desc, 0, GetNumCharsToLastNonPeriod(desc)),
    Category(category, 0, GetNumCharsToLastNonPeriod(category)) {}

void PathDiagnosticClient::HandleDiagnostic(Diagnostic::Level DiagLevel,
                                            const DiagnosticInfo &Info) {
  
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
    new PathDiagnosticEventPiece(Info.getLocation(),
                            std::string(StrC.begin(), StrC.end()));
  
  for (unsigned i = 0, e = Info.getNumRanges(); i != e; ++i)
    P->addRange(Info.getRange(i));
  for (unsigned i = 0, e = Info.getNumCodeModificationHints(); i != e; ++i)
    P->addCodeModificationHint(Info.getCodeModificationHint(i));
  D->push_front(P);

  HandlePathDiagnostic(D);  
}

//===----------------------------------------------------------------------===//
// PathDiagnosticLocation methods.
//===----------------------------------------------------------------------===//

FullSourceLoc PathDiagnosticLocation::asLocation() const {
  assert(isValid());
  // Note that we want a 'switch' here so that the compiler can warn us in
  // case we add more cases.
  switch (K) {
    case SingleLocK:
    case RangeK:
      break;
    case StmtK:
      return FullSourceLoc(S->getLocStart(), const_cast<SourceManager&>(*SM));
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
        case Stmt::ConditionalOperatorClass:
        case Stmt::ObjCForCollectionStmtClass: {
          SourceLocation L = S->getLocStart();
          return SourceRange(L, L);
        }
      }
      
      return S->getSourceRange();
    }
    case DeclK:
      if (const ObjCMethodDecl *MD = dyn_cast<ObjCMethodDecl>(D))
        return MD->getSourceRange();
      if (const FunctionDecl *FD = dyn_cast<FunctionDecl>(D)) {
        // FIXME: We would like to always get the function body, even
        // when it needs to be de-serialized, but getting the
        // ASTContext here requires significant changes.
        if (Stmt *Body = FD->getBodyIfAvailable()) {
          if (CompoundStmt *CS = dyn_cast<CompoundStmt>(Body))
            return CS->getSourceRange();
          else
            return cast<CXXTryStmt>(Body)->getSourceRange();
        }
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


