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
    case SingleLoc:
    case Range:
      break;
    case Statement:
      return FullSourceLoc(S->getLocStart(), const_cast<SourceManager&>(*SM));
  }
  
  return FullSourceLoc(R.getBegin(), const_cast<SourceManager&>(*SM));
}

SourceRange PathDiagnosticLocation::asRange() const {
  assert(isValid());
  // Note that we want a 'switch' here so that the compiler can warn us in
  // case we add more cases.
  switch (K) {
    case SingleLoc:
    case Range:
      break;
    case Statement:
      return S->getSourceRange();
  }
  
  return R;
}

