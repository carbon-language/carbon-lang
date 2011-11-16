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
#include "CXTranslationUnit.h"
#include "CXSourceLocation.h"
#include "CXString.h"

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


CXDiagnosticSetImpl::~CXDiagnosticSetImpl() {
  for (std::vector<CXDiagnosticImpl *>::iterator it = Diagnostics.begin(),
       et = Diagnostics.end();
       it != et; ++it) {
    delete *it;
  }
}

CXDiagnosticImpl::~CXDiagnosticImpl() {}

static CXDiagnosticSetImpl *lazyCreateDiags(CXTranslationUnit TU,
                                            bool checkIfChanged = false) {
  ASTUnit *AU = static_cast<ASTUnit *>(TU->TUData);

  if (TU->Diagnostics && checkIfChanged) {
    // In normal use, ASTUnit's diagnostics should not change unless we reparse.
    // Currently they can only change by using the internal testing flag
    // '-error-on-deserialized-decl' which will error during deserialization of
    // a declaration. What will happen is:
    //
    //  -c-index-test gets a CXTranslationUnit
    //  -checks the diagnostics, the diagnostics set is lazily created,
    //     no errors are reported
    //  -later does an operation, like annotation of tokens, that triggers
    //     -error-on-deserialized-decl, that will emit a diagnostic error,
    //     that ASTUnit will catch and add to its stored diagnostics vector.
    //  -c-index-test wants to check whether an error occurred after performing
    //     the operation but can only query the lazily created set.
    //
    // We check here if a new diagnostic was appended since the last time the
    // diagnostic set was created, in which case we reset it.

    CXDiagnosticSetImpl *
      Set = static_cast<CXDiagnosticSetImpl*>(TU->Diagnostics);
    if (AU->stored_diag_size() != Set->getNumDiagnostics()) {
      // Diagnostics in the ASTUnit were updated, reset the associated
      // diagnostics.
      delete Set;
      TU->Diagnostics = 0;
    }
  }

  if (!TU->Diagnostics) {
    CXDiagnosticSetImpl *Set = new CXDiagnosticSetImpl();
    TU->Diagnostics = Set;
    
    for (ASTUnit::stored_diag_iterator it = AU->stored_diag_begin(),
         ei = AU->stored_diag_end(); it != ei; ++it) {
      CXStoredDiagnostic *D =
        new CXStoredDiagnostic(*it, AU->getASTContext().getLangOptions());
      Set->appendDiagnostic(D);
    }
  }
  return static_cast<CXDiagnosticSetImpl*>(TU->Diagnostics);
}

//-----------------------------------------------------------------------------
// C Interface Routines
//-----------------------------------------------------------------------------
extern "C" {

unsigned clang_getNumDiagnostics(CXTranslationUnit Unit) {
  if (!Unit->TUData)
    return 0;
  return lazyCreateDiags(Unit, /*checkIfChanged=*/true)->getNumDiagnostics();
}

CXDiagnostic clang_getDiagnostic(CXTranslationUnit Unit, unsigned Index) {
  if (!Unit->TUData)
    return 0;

  CXDiagnosticSetImpl *Diags = lazyCreateDiags(Unit);
  if (Index >= Diags->getNumDiagnostics())
    return 0;

  return Diags->getDiagnostic(Index);
}

void clang_disposeDiagnostic(CXDiagnostic Diagnostic) {
  // No-op.  Kept as a legacy API.  CXDiagnostics are now managed
  // by the enclosing CXDiagnosticSet.
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
    clang_getSpellingLocation(clang_getDiagnosticLocation(Diagnostic),
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
          clang_getSpellingLocation(clang_getRangeStart(Range),
                                    &StartFile, &StartLine, &StartColumn,
                                    0);
          clang_getSpellingLocation(clang_getRangeEnd(Range),
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
  case CXDiagnostic_Ignored: llvm_unreachable("impossible");
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
  
  if (Options & (CXDiagnostic_DisplayOption | CXDiagnostic_DisplayCategoryId |
                 CXDiagnostic_DisplayCategoryName)) {
    bool NeedBracket = true;
    bool NeedComma = false;

    if (Options & CXDiagnostic_DisplayOption) {
      CXString OptionName = clang_getDiagnosticOption(Diagnostic, 0);
      if (const char *OptionText = clang_getCString(OptionName)) {
        if (OptionText[0]) {
          Out << " [" << OptionText;
          NeedBracket = false;
          NeedComma = true;
        }
      }
      clang_disposeString(OptionName);
    }
    
    if (Options & (CXDiagnostic_DisplayCategoryId | 
                   CXDiagnostic_DisplayCategoryName)) {
      if (unsigned CategoryID = clang_getDiagnosticCategory(Diagnostic)) {
        if (Options & CXDiagnostic_DisplayCategoryId) {
          if (NeedBracket)
            Out << " [";
          if (NeedComma)
            Out << ", ";
          Out << CategoryID;
          NeedBracket = false;
          NeedComma = true;
        }
        
        if (Options & CXDiagnostic_DisplayCategoryName) {
          CXString CategoryName = clang_getDiagnosticCategoryName(CategoryID);
          if (NeedBracket)
            Out << " [";
          if (NeedComma)
            Out << ", ";
          Out << clang_getCString(CategoryName);
          NeedBracket = false;
          NeedComma = true;
          clang_disposeString(CategoryName);
        }
      }
    }
    
    if (!NeedBracket)
      Out << "]";
  }
  
  return createCXString(Out.str(), true);
}

unsigned clang_defaultDiagnosticDisplayOptions() {
  return CXDiagnostic_DisplaySourceLocation | CXDiagnostic_DisplayColumn |
         CXDiagnostic_DisplayOption;
}

enum CXDiagnosticSeverity clang_getDiagnosticSeverity(CXDiagnostic Diag) {
  if (CXDiagnosticImpl *D = static_cast<CXDiagnosticImpl*>(Diag))
    return D->getSeverity();
  return CXDiagnostic_Ignored;
}

CXSourceLocation clang_getDiagnosticLocation(CXDiagnostic Diag) {
  if (CXDiagnosticImpl *D = static_cast<CXDiagnosticImpl*>(Diag))
    return D->getLocation();
  return clang_getNullLocation();
}

CXString clang_getDiagnosticSpelling(CXDiagnostic Diag) {
  if (CXDiagnosticImpl *D = static_cast<CXDiagnosticImpl *>(Diag))
    return D->getSpelling();
  return createCXString("");
}

CXString clang_getDiagnosticOption(CXDiagnostic Diag, CXString *Disable) {
  if (Disable)
    *Disable = createCXString("");

  if (CXDiagnosticImpl *D = static_cast<CXDiagnosticImpl *>(Diag))
    return D->getDiagnosticOption(Disable);

  return createCXString("");
}

unsigned clang_getDiagnosticCategory(CXDiagnostic Diag) {
  if (CXDiagnosticImpl *D = static_cast<CXDiagnosticImpl *>(Diag))
    return D->getCategory();
  return 0;
}
  
CXString clang_getDiagnosticCategoryName(unsigned Category) {
  return createCXString(DiagnosticIDs::getCategoryNameFromID(Category));
}
  
unsigned clang_getDiagnosticNumRanges(CXDiagnostic Diag) {
  if (CXDiagnosticImpl *D = static_cast<CXDiagnosticImpl *>(Diag))
    return D->getNumRanges();
  return 0;
}

CXSourceRange clang_getDiagnosticRange(CXDiagnostic Diag, unsigned Range) {
  CXDiagnosticImpl *D = static_cast<CXDiagnosticImpl *>(Diag);  
  if (!D || Range >= D->getNumRanges())
    return clang_getNullRange();
  return D->getRange(Range);
}

unsigned clang_getDiagnosticNumFixIts(CXDiagnostic Diag) {
  if (CXDiagnosticImpl *D = static_cast<CXDiagnosticImpl *>(Diag))
    return D->getNumFixIts();
  return 0;
}

CXString clang_getDiagnosticFixIt(CXDiagnostic Diag, unsigned FixIt,
                                  CXSourceRange *ReplacementRange) {
  CXDiagnosticImpl *D = static_cast<CXDiagnosticImpl *>(Diag);
  if (!D || FixIt >= D->getNumFixIts()) {
    if (ReplacementRange)
      *ReplacementRange = clang_getNullRange();
    return createCXString("");
  }
  return D->getFixIt(FixIt, ReplacementRange);
}

void clang_disposeDiagnosticSet(CXDiagnosticSet Diags) {
  CXDiagnosticSetImpl *D = static_cast<CXDiagnosticSetImpl*>(Diags);
  if (D->isExternallyManaged())
    delete D;
}
  
CXDiagnostic clang_getDiagnosticInSet(CXDiagnosticSet Diags,
                                      unsigned Index) {
  if (CXDiagnosticSetImpl *D = static_cast<CXDiagnosticSetImpl*>(Diags))
    if (Index < D->getNumDiagnostics())
      return D->getDiagnostic(Index);
  return 0;  
}
  
CXDiagnosticSet clang_getChildDiagnostics(CXDiagnostic Diag) {
  if (CXDiagnosticImpl *D = static_cast<CXDiagnosticImpl *>(Diag)) {
    CXDiagnosticSetImpl &ChildDiags = D->getChildDiagnostics();
    return ChildDiags.empty() ? 0 : (CXDiagnosticSet) &ChildDiags;
  }
  return 0;
}

unsigned clang_getNumDiagnosticsInSet(CXDiagnosticSet Diags) {
  if (CXDiagnosticSetImpl *D = static_cast<CXDiagnosticSetImpl*>(Diags))
    return D->getNumDiagnostics();
  return 0;
}

} // end extern "C"
