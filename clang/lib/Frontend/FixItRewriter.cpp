//===--- FixItRewriter.cpp - Fix-It Rewriter Diagnostic Client --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This is a diagnostic client adaptor that performs rewrites as
// suggested by code modification hints attached to diagnostics. It
// then forwards any diagnostics to the adapted diagnostic client.
//
//===----------------------------------------------------------------------===//

#include "clang/Frontend/FixItRewriter.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Frontend/FrontendDiagnostic.h"
#include "llvm/ADT/OwningPtr.h"
#include "llvm/Support/Streams.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/System/Path.h"
using namespace clang;

FixItRewriter::FixItRewriter(Diagnostic &Diags, SourceManager &SourceMgr,
                             const LangOptions &LangOpts)
  : Diags(Diags), Rewrite(SourceMgr, LangOpts), NumFailures(0) {
  Client = Diags.getClient();
  Diags.setClient(this);
}

FixItRewriter::~FixItRewriter() {
  Diags.setClient(Client);
}

bool FixItRewriter::WriteFixedFile(const std::string &InFileName, 
                                   const std::string &OutFileName) {
  if (NumFailures > 0) {
    Diag(FullSourceLoc(), diag::warn_fixit_no_changes);
    return true;
  }

  llvm::OwningPtr<llvm::raw_ostream> OwnedStream;
  llvm::raw_ostream *OutFile;
  if (!OutFileName.empty()) {
    std::string Err;
    OutFile = new llvm::raw_fd_ostream(OutFileName.c_str(), 
                                       // set binary mode (critical for Windoze)
                                       true, 
                                       Err);
    OwnedStream.reset(OutFile);
  } else if (InFileName == "-") {
    OutFile = &llvm::outs();
  } else {
    llvm::sys::Path Path(InFileName);
    std::string Suffix = Path.getSuffix();
    Path.eraseSuffix();
    Path.appendSuffix("fixit." + Suffix);
    std::string Err;
    OutFile = new llvm::raw_fd_ostream(Path.toString().c_str(), 
                                       // set binary mode (critical for Windoze)
                                       true, 
                                       Err);
    OwnedStream.reset(OutFile);
  }  

  FileID MainFileID = Rewrite.getSourceMgr().getMainFileID();
  if (const RewriteBuffer *RewriteBuf = 
        Rewrite.getRewriteBufferFor(MainFileID)) {
    *OutFile << std::string(RewriteBuf->begin(), RewriteBuf->end());
  } else {
    std::fprintf(stderr, "Main file is unchanged\n");
  }
  OutFile->flush();

  return false;
}

bool FixItRewriter::IncludeInDiagnosticCounts() const {
  return Client? Client->IncludeInDiagnosticCounts() : true;
}

void FixItRewriter::HandleDiagnostic(Diagnostic::Level DiagLevel,
                                     const DiagnosticInfo &Info) {
  Client->HandleDiagnostic(DiagLevel, Info);

  // Skip over any diagnostics that are ignored.
  if (DiagLevel == Diagnostic::Ignored)
    return;

  if (!FixItLocations.empty()) {
    // The user has specified the locations where we should perform
    // the various fix-it modifications.

    // If this diagnostic does not have any code modifications,
    // completely ignore it, even if it's an error: fix-it locations
    // are meant to perform specific fix-ups even in the presence of
    // other errors.
    if (Info.getNumCodeModificationHints() == 0)
      return;

    // See if the location of the error is one that matches what the
    // user requested.
    bool AcceptableLocation = false;
    const FileEntry *File 
      = Rewrite.getSourceMgr().getFileEntryForID(
                                            Info.getLocation().getFileID());
    unsigned Line = Info.getLocation().getSpellingLineNumber();
    unsigned Column = Info.getLocation().getSpellingColumnNumber();
    for (llvm::SmallVector<RequestedSourceLocation, 4>::iterator
           Loc = FixItLocations.begin(), LocEnd = FixItLocations.end();
         Loc != LocEnd; ++Loc) {
      if (Loc->File == File && Loc->Line == Line && Loc->Column == Column) {
        AcceptableLocation = true;
        break;
      }
    }

    if (!AcceptableLocation)
      return;
  }

  // Make sure that we can perform all of the modifications we
  // in this diagnostic.
  bool CanRewrite = Info.getNumCodeModificationHints() > 0;
  for (unsigned Idx = 0, Last = Info.getNumCodeModificationHints();
       Idx < Last; ++Idx) {
    const CodeModificationHint &Hint = Info.getCodeModificationHint(Idx);
    if (Hint.RemoveRange.isValid() &&
        Rewrite.getRangeSize(Hint.RemoveRange) == -1) {
      CanRewrite = false;
      break;
    }

    if (Hint.InsertionLoc.isValid() && 
        !Rewrite.isRewritable(Hint.InsertionLoc)) {
      CanRewrite = false;
      break;
    }
  }

  if (!CanRewrite) { 
    if (Info.getNumCodeModificationHints() > 0)
      Diag(Info.getLocation(), diag::note_fixit_in_macro);

    // If this was an error, refuse to perform any rewriting.
    if (DiagLevel == Diagnostic::Error || DiagLevel == Diagnostic::Fatal) {
      if (++NumFailures == 1)
        Diag(Info.getLocation(), diag::note_fixit_unfixed_error);
    }
    return;
  }

  bool Failed = false;
  for (unsigned Idx = 0, Last = Info.getNumCodeModificationHints(); 
       Idx < Last; ++Idx) {
    const CodeModificationHint &Hint = Info.getCodeModificationHint(Idx);
    if (!Hint.RemoveRange.isValid()) {
      // We're adding code.
      if (Rewrite.InsertStrBefore(Hint.InsertionLoc, Hint.CodeToInsert))
        Failed = true;
      continue;
    }
    
    if (Hint.CodeToInsert.empty()) {
      // We're removing code.
      if (Rewrite.RemoveText(Hint.RemoveRange.getBegin(),
                             Rewrite.getRangeSize(Hint.RemoveRange)))
        Failed = true;
      continue;
    } 
      
    // We're replacing code.
    if (Rewrite.ReplaceText(Hint.RemoveRange.getBegin(),
                            Rewrite.getRangeSize(Hint.RemoveRange),
                            Hint.CodeToInsert.c_str(),
                            Hint.CodeToInsert.size()))
      Failed = true;
  }

  if (Failed) {
    ++NumFailures;
    Diag(Info.getLocation(), diag::note_fixit_failed);
    return;
  }

  Diag(Info.getLocation(), diag::note_fixit_applied);
}

/// \brief Emit a diagnostic via the adapted diagnostic client.
void FixItRewriter::Diag(FullSourceLoc Loc, unsigned DiagID) {
  // When producing this diagnostic, we temporarily bypass ourselves,
  // clear out any current diagnostic, and let the downstream client
  // format the diagnostic.
  Diags.setClient(Client);
  Diags.Clear();
  Diags.Report(Loc, DiagID);
  Diags.setClient(this);  
}
