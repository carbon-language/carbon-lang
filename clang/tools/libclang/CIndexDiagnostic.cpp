/*===-- CIndexDiagnostics.cpp - Diagnostics C Interface ---------*- C++ -*-===*\
|*                                                                            *|
|*                     The LLVM Compiler Infrastructure                       *|
|*                                                                            *|
|* This file is distributed under the University of Illinois Open Source      *|
|* License. See LICENSE.TXT for details.                                      *|
|*                                                                            *|
|*===----------------------------------------------------------------------===*|
|*                                                                            *|
|* Implements the diagnostic functions of the Clang C interface.              *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/
#include "CIndexDiagnostic.h"
#include "CIndexer.h"
#include "CXSourceLocation.h"

#include "clang/Frontend/ASTUnit.h"
#include "clang/Frontend/FrontendDiagnostic.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

using namespace clang;
using namespace clang::cxloc;
using namespace clang::cxstring;
using namespace llvm;

//-----------------------------------------------------------------------------
// C Interface Routines
//-----------------------------------------------------------------------------
extern "C" {

unsigned clang_getNumDiagnostics(CXTranslationUnit Unit) {
  ASTUnit *CXXUnit = static_cast<ASTUnit *>(Unit);
  return CXXUnit? CXXUnit->stored_diag_size() : 0;
}

CXDiagnostic clang_getDiagnostic(CXTranslationUnit Unit, unsigned Index) {
  ASTUnit *CXXUnit = static_cast<ASTUnit *>(Unit);
  if (!CXXUnit || Index >= CXXUnit->stored_diag_size())
    return 0;

  return new CXStoredDiagnostic(CXXUnit->stored_diag_begin()[Index],
                                CXXUnit->getASTContext().getLangOptions());
}

void clang_disposeDiagnostic(CXDiagnostic Diagnostic) {
  CXStoredDiagnostic *Stored = static_cast<CXStoredDiagnostic *>(Diagnostic);
  delete Stored;
}

CXString clang_formatDiagnostic(CXDiagnostic Diagnostic, unsigned Options) {
  if (!Diagnostic)
    return createCXString("");

  CXDiagnosticSeverity Severity = clang_getDiagnosticSeverity(Diagnostic);

  llvm::SmallString<256> Str;
  llvm::raw_svector_ostream Out(Str);
  
  if (Options & CXDiagnostic_DisplaySourceLocation) {
    // Print source location (file:line), along with optional column
    // and source ranges.
    CXFile File;
    unsigned Line, Column;
    clang_getInstantiationLocation(clang_getDiagnosticLocation(Diagnostic),
                                   &File, &Line, &Column, 0);
    if (File) {
      CXString FName = clang_getFileName(File);
      Out << clang_getCString(FName) << ":" << Line << ":";
      clang_disposeString(FName);
      if (Options & CXDiagnostic_DisplayColumn)
        Out << Column << ":";

      if (Options & CXDiagnostic_DisplaySourceRanges) {
        unsigned N = clang_getDiagnosticNumRanges(Diagnostic);
        bool PrintedRange = false;
        for (unsigned I = 0; I != N; ++I) {
          CXFile StartFile, EndFile;
          CXSourceRange Range = clang_getDiagnosticRange(Diagnostic, I);
          
          unsigned StartLine, StartColumn, EndLine, EndColumn;
          clang_getInstantiationLocation(clang_getRangeStart(Range),
                                         &StartFile, &StartLine, &StartColumn,
                                         0);
          clang_getInstantiationLocation(clang_getRangeEnd(Range),
                                         &EndFile, &EndLine, &EndColumn, 0);
          
          if (StartFile != EndFile || StartFile != File)
            continue;
          
          Out << "{" << StartLine << ":" << StartColumn << "-"
              << EndLine << ":" << EndColumn << "}";
          PrintedRange = true;
        }
        if (PrintedRange)
          Out << ":";
      }
      
      Out << " ";
    }
  }

  /* Print warning/error/etc. */
  switch (Severity) {
  case CXDiagnostic_Ignored: assert(0 && "impossible"); break;
  case CXDiagnostic_Note: Out << "note: "; break;
  case CXDiagnostic_Warning: Out << "warning: "; break;
  case CXDiagnostic_Error: Out << "error: "; break;
  case CXDiagnostic_Fatal: Out << "fatal error: "; break;
  }

  CXString Text = clang_getDiagnosticSpelling(Diagnostic);
  if (clang_getCString(Text))
    Out << clang_getCString(Text);
  else
    Out << "<no diagnostic text>";
  clang_disposeString(Text);
  return createCXString(Out.str(), true);
}

unsigned clang_defaultDiagnosticDisplayOptions() {
  return CXDiagnostic_DisplaySourceLocation | CXDiagnostic_DisplayColumn;
}

enum CXDiagnosticSeverity clang_getDiagnosticSeverity(CXDiagnostic Diag) {
  CXStoredDiagnostic *StoredDiag = static_cast<CXStoredDiagnostic *>(Diag);
  if (!StoredDiag)
    return CXDiagnostic_Ignored;

  switch (StoredDiag->Diag.getLevel()) {
  case Diagnostic::Ignored: return CXDiagnostic_Ignored;
  case Diagnostic::Note:    return CXDiagnostic_Note;
  case Diagnostic::Warning: return CXDiagnostic_Warning;
  case Diagnostic::Error:   return CXDiagnostic_Error;
  case Diagnostic::Fatal:   return CXDiagnostic_Fatal;
  }

  llvm_unreachable("Invalid diagnostic level");
  return CXDiagnostic_Ignored;
}

CXSourceLocation clang_getDiagnosticLocation(CXDiagnostic Diag) {
  CXStoredDiagnostic *StoredDiag = static_cast<CXStoredDiagnostic *>(Diag);
  if (!StoredDiag || StoredDiag->Diag.getLocation().isInvalid())
    return clang_getNullLocation();

  return translateSourceLocation(StoredDiag->Diag.getLocation().getManager(),
                                 StoredDiag->LangOpts,
                                 StoredDiag->Diag.getLocation());
}

CXString clang_getDiagnosticSpelling(CXDiagnostic Diag) {
  CXStoredDiagnostic *StoredDiag = static_cast<CXStoredDiagnostic *>(Diag);
  if (!StoredDiag)
    return createCXString("");

  return createCXString(StoredDiag->Diag.getMessage(), false);
}

unsigned clang_getDiagnosticNumRanges(CXDiagnostic Diag) {
  CXStoredDiagnostic *StoredDiag = static_cast<CXStoredDiagnostic *>(Diag);
  if (!StoredDiag || StoredDiag->Diag.getLocation().isInvalid())
    return 0;

  return StoredDiag->Diag.range_size();
}

CXSourceRange clang_getDiagnosticRange(CXDiagnostic Diag, unsigned Range) {
  CXStoredDiagnostic *StoredDiag = static_cast<CXStoredDiagnostic *>(Diag);
  if (!StoredDiag || Range >= StoredDiag->Diag.range_size() ||
      StoredDiag->Diag.getLocation().isInvalid())
    return clang_getNullRange();

  return translateSourceRange(StoredDiag->Diag.getLocation().getManager(),
                              StoredDiag->LangOpts,
                              StoredDiag->Diag.range_begin()[Range]);
}

unsigned clang_getDiagnosticNumFixIts(CXDiagnostic Diag) {
  CXStoredDiagnostic *StoredDiag = static_cast<CXStoredDiagnostic *>(Diag);
  if (!StoredDiag)
    return 0;

  return StoredDiag->Diag.fixit_size();
}

CXString clang_getDiagnosticFixIt(CXDiagnostic Diagnostic, unsigned FixIt,
                                  CXSourceRange *ReplacementRange) {
  CXStoredDiagnostic *StoredDiag
    = static_cast<CXStoredDiagnostic *>(Diagnostic);
  if (!StoredDiag || FixIt >= StoredDiag->Diag.fixit_size() ||
      StoredDiag->Diag.getLocation().isInvalid()) {
    if (ReplacementRange)
      *ReplacementRange = clang_getNullRange();

    return createCXString("");
  }

  const FixItHint &Hint = StoredDiag->Diag.fixit_begin()[FixIt];
  if (ReplacementRange) {
    // Create a range that covers the entire replacement (or
    // removal) range, adjusting the end of the range to point to
    // the end of the token.
    *ReplacementRange
        = translateSourceRange(StoredDiag->Diag.getLocation().getManager(),
                                StoredDiag->LangOpts,
                                Hint.RemoveRange);
  }

  return createCXString(Hint.CodeToInsert);
}

} // end extern "C"
