/*===-- CIndexDiagnostics.cpp - Diagnostics C Interface -----------*- C -*-===*\
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

#include "clang/Frontend/FrontendDiagnostic.h"
#include "llvm/Support/MemoryBuffer.h"

using namespace clang;
using namespace clang::cxloc;

//-----------------------------------------------------------------------------
// Opaque data structures                        
//-----------------------------------------------------------------------------
namespace {
  /// \brief The storage behind a CXDiagnostic
  struct CXStoredDiagnostic {
    /// \brief The translation unit this diagnostic came from.
    const LangOptions &LangOpts;
    
    /// \brief The severity level of this diagnostic.
    Diagnostic::Level Level;
    
    /// \brief A reference to the diagnostic information.
    const DiagnosticInfo &Info;
  };
}

//-----------------------------------------------------------------------------
// CIndex Diagnostic Client                        
//-----------------------------------------------------------------------------
CIndexDiagnosticClient::~CIndexDiagnosticClient() { }

void CIndexDiagnosticClient::BeginSourceFile(const LangOptions &LangOpts,
                                             const Preprocessor *PP) {
  this->LangOpts = LangOpts;
}

void CIndexDiagnosticClient::HandleDiagnostic(Diagnostic::Level DiagLevel,
                                              const DiagnosticInfo &Info) {
  if (!Callback)
    return;
  
  CXStoredDiagnostic Stored = { this->LangOpts, DiagLevel, Info };
  Callback(&Stored, ClientData);
}

//-----------------------------------------------------------------------------
// C Interface Routines                        
//-----------------------------------------------------------------------------
extern "C" {
  
enum CXDiagnosticSeverity clang_getDiagnosticSeverity(CXDiagnostic Diag) {
  CXStoredDiagnostic *StoredDiag = static_cast<CXStoredDiagnostic *>(Diag);
  if (!StoredDiag)
    return CXDiagnostic_Ignored;
  
  switch (StoredDiag->Level) {
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
  if (!StoredDiag || StoredDiag->Info.getLocation().isInvalid())
    return clang_getNullLocation();
  
  return translateSourceLocation(StoredDiag->Info.getLocation().getManager(),
                                 StoredDiag->LangOpts,
                                 StoredDiag->Info.getLocation());
}

CXString clang_getDiagnosticSpelling(CXDiagnostic Diag) {
  CXStoredDiagnostic *StoredDiag = static_cast<CXStoredDiagnostic *>(Diag);
  if (!StoredDiag)
    return CIndexer::createCXString("");
  
  llvm::SmallString<64> Spelling;
  StoredDiag->Info.FormatDiagnostic(Spelling);
  return CIndexer::createCXString(Spelling.str(), true);
}

void clang_getDiagnosticRanges(CXDiagnostic Diag, 
                               CXSourceRange **Ranges, 
                               unsigned *NumRanges) {
  if (Ranges) 
    *Ranges = 0;
  if (NumRanges)
    *NumRanges = 0;
  
  CXStoredDiagnostic *StoredDiag = static_cast<CXStoredDiagnostic *>(Diag);
  if (!StoredDiag || !Ranges || !NumRanges || 
      !StoredDiag->Info.getNumRanges() || 
      StoredDiag->Info.getLocation().isInvalid())
    return;
  
  unsigned N = StoredDiag->Info.getNumRanges();
  *Ranges = (CXSourceRange *)malloc(sizeof(CXSourceRange) * N);
  *NumRanges = N;
  for (unsigned I = 0; I != N; ++I)
    (*Ranges)[I] = translateSourceRange(
                                    StoredDiag->Info.getLocation().getManager(),
                                        StoredDiag->LangOpts,
                                        StoredDiag->Info.getRange(I));
}

void clang_disposeDiagnosticRanges(CXSourceRange *Ranges, 
                                   unsigned NumRanges) {
  free(Ranges);
}

unsigned clang_getDiagnosticNumFixIts(CXDiagnostic Diag) {
  CXStoredDiagnostic *StoredDiag = static_cast<CXStoredDiagnostic *>(Diag);
  if (!StoredDiag)
    return 0;
  
  return StoredDiag->Info.getNumCodeModificationHints();
}

enum CXFixItKind clang_getDiagnosticFixItKind(CXDiagnostic Diag, 
                                              unsigned FixIt) {
  CXStoredDiagnostic *StoredDiag = static_cast<CXStoredDiagnostic *>(Diag);
  if (!StoredDiag || FixIt >= StoredDiag->Info.getNumCodeModificationHints())
    return CXFixIt_Insertion;
  
  const CodeModificationHint &Hint
    = StoredDiag->Info.getCodeModificationHint(FixIt);
  if (Hint.RemoveRange.isInvalid())
    return CXFixIt_Insertion;
  if (Hint.InsertionLoc.isInvalid())
    return CXFixIt_Removal;
  
  return CXFixIt_Replacement;  
}

CXString clang_getDiagnosticFixItInsertion(CXDiagnostic Diag, 
                                           unsigned FixIt,
                                           CXSourceLocation *Location) {
  CXStoredDiagnostic *StoredDiag = static_cast<CXStoredDiagnostic *>(Diag);
  if (!StoredDiag || FixIt >= StoredDiag->Info.getNumCodeModificationHints())
    return CIndexer::createCXString("");
  
  const CodeModificationHint &Hint
    = StoredDiag->Info.getCodeModificationHint(FixIt);
  return CIndexer::createCXString(Hint.CodeToInsert);
}

CXSourceRange clang_getDiagnosticFixItRemoval(CXDiagnostic Diag, 
                                              unsigned FixIt) {
  CXStoredDiagnostic *StoredDiag = static_cast<CXStoredDiagnostic *>(Diag);
  if (!StoredDiag || FixIt >= StoredDiag->Info.getNumCodeModificationHints() ||
      StoredDiag->Info.getLocation().isInvalid())
    return clang_getNullRange();
  
  const CodeModificationHint &Hint
    = StoredDiag->Info.getCodeModificationHint(FixIt);
  return translateSourceRange(StoredDiag->Info.getLocation().getManager(),
                              StoredDiag->LangOpts,
                              Hint.RemoveRange);
}

CXString clang_getDiagnosticFixItReplacement(CXDiagnostic Diag, 
                                             unsigned FixIt,
                                             CXSourceRange *Range) {
  CXStoredDiagnostic *StoredDiag = static_cast<CXStoredDiagnostic *>(Diag);
  if (!StoredDiag || FixIt >= StoredDiag->Info.getNumCodeModificationHints() ||
      StoredDiag->Info.getLocation().isInvalid()) {
    if (Range)
      *Range = clang_getNullRange();
    
    return CIndexer::createCXString("");
  }
  
  const CodeModificationHint &Hint
    = StoredDiag->Info.getCodeModificationHint(FixIt);
  if (Range)
    *Range = translateSourceRange(StoredDiag->Info.getLocation().getManager(),
                                  StoredDiag->LangOpts,
                                  Hint.RemoveRange);
  return CIndexer::createCXString(Hint.CodeToInsert);  
}
  
} // end extern "C"

void clang::ReportSerializedDiagnostics(const llvm::sys::Path &DiagnosticsPath,
                                        Diagnostic &Diags,
                                        unsigned num_unsaved_files,
                                        struct CXUnsavedFile *unsaved_files) {
  using llvm::MemoryBuffer;
  using llvm::StringRef;
  MemoryBuffer *F = MemoryBuffer::getFile(DiagnosticsPath.c_str());
  if (!F)
    return;

  // Enter the unsaved files into the file manager.
  SourceManager SourceMgr;
  FileManager FileMgr;
  for (unsigned I = 0; I != num_unsaved_files; ++I) {
    const FileEntry *File = FileMgr.getVirtualFile(unsaved_files[I].Filename,
                                                   unsaved_files[I].Length,
                                                   0);
    if (!File) {
      Diags.Report(diag::err_fe_remap_missing_from_file)
        << unsaved_files[I].Filename;
      return;
    }

    MemoryBuffer *Buffer
      = MemoryBuffer::getMemBuffer(unsaved_files[I].Contents,
                           unsaved_files[I].Contents + unsaved_files[I].Length);
    if (!Buffer)
      return;

    SourceMgr.overrideFileContents(File, Buffer);
  }

  // Parse the diagnostics, emitting them one by one until we've
  // exhausted the data.
  StringRef Buffer = F->getBuffer();
  const char *Memory = Buffer.data(), *MemoryEnd = Memory + Buffer.size();
  while (Memory != MemoryEnd) {
    DiagnosticBuilder DB = Diags.Deserialize(FileMgr, SourceMgr, 
                                             Memory, MemoryEnd);
    if (!DB.isActive())
      return;
  }
}
