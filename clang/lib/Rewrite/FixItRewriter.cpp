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

#include "clang/Rewrite/FixItRewriter.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Frontend/FrontendDiagnostic.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Path.h"
#include "llvm/ADT/OwningPtr.h"
#include <cstdio>

using namespace clang;

FixItRewriter::FixItRewriter(DiagnosticsEngine &Diags, SourceManager &SourceMgr,
                             const LangOptions &LangOpts,
                             FixItOptions *FixItOpts)
  : Diags(Diags),
    Rewrite(SourceMgr, LangOpts),
    FixItOpts(FixItOpts),
    NumFailures(0) {
  Client = Diags.takeClient();
  Diags.setClient(this);
}

FixItRewriter::~FixItRewriter() {
  Diags.takeClient();
  Diags.setClient(Client);
}

bool FixItRewriter::WriteFixedFile(FileID ID, raw_ostream &OS) {
  const RewriteBuffer *RewriteBuf = Rewrite.getRewriteBufferFor(ID);
  if (!RewriteBuf) return true;
  RewriteBuf->write(OS);
  OS.flush();
  return false;
}

bool FixItRewriter::WriteFixedFiles() {
  if (NumFailures > 0 && !FixItOpts->FixWhatYouCan) {
    Diag(FullSourceLoc(), diag::warn_fixit_no_changes);
    return true;
  }

  for (iterator I = buffer_begin(), E = buffer_end(); I != E; ++I) {
    const FileEntry *Entry = Rewrite.getSourceMgr().getFileEntryForID(I->first);
    std::string Filename = FixItOpts->RewriteFilename(Entry->getName());
    std::string Err;
    llvm::raw_fd_ostream OS(Filename.c_str(), Err,
                            llvm::raw_fd_ostream::F_Binary);
    if (!Err.empty()) {
      Diags.Report(clang::diag::err_fe_unable_to_open_output)
          << Filename << Err;
      continue;
    }
    RewriteBuffer &RewriteBuf = I->second;
    RewriteBuf.write(OS);
    OS.flush();
  }

  return false;
}

bool FixItRewriter::IncludeInDiagnosticCounts() const {
  return Client ? Client->IncludeInDiagnosticCounts() : true;
}

void FixItRewriter::HandleDiagnostic(DiagnosticsEngine::Level DiagLevel,
                                     const Diagnostic &Info) {
  // Default implementation (Warnings/errors count).
  DiagnosticConsumer::HandleDiagnostic(DiagLevel, Info);

  Client->HandleDiagnostic(DiagLevel, Info);

  // Skip over any diagnostics that are ignored or notes.
  if (DiagLevel <= DiagnosticsEngine::Note)
    return;

  // Make sure that we can perform all of the modifications we
  // in this diagnostic.
  bool CanRewrite = Info.getNumFixItHints() > 0;
  for (unsigned Idx = 0, Last = Info.getNumFixItHints();
       Idx < Last; ++Idx) {
    const FixItHint &Hint = Info.getFixItHint(Idx);
    if (Hint.RemoveRange.isValid() &&
        Rewrite.getRangeSize(Hint.RemoveRange) == -1) {
      CanRewrite = false;
      break;
    }
  }

  if (!CanRewrite) {
    if (Info.getNumFixItHints() > 0)
      Diag(Info.getLocation(), diag::note_fixit_in_macro);

    // If this was an error, refuse to perform any rewriting.
    if (DiagLevel == DiagnosticsEngine::Error ||
          DiagLevel == DiagnosticsEngine::Fatal) {
      if (++NumFailures == 1)
        Diag(Info.getLocation(), diag::note_fixit_unfixed_error);
    }
    return;
  }

  bool Failed = false;
  for (unsigned Idx = 0, Last = Info.getNumFixItHints();
       Idx < Last; ++Idx) {
    const FixItHint &Hint = Info.getFixItHint(Idx);

    if (Hint.CodeToInsert.empty()) {
      // We're removing code.
      if (Rewrite.RemoveText(Hint.RemoveRange))
        Failed = true;
      continue;
    }

    // We're replacing code.
    if (Rewrite.ReplaceText(Hint.RemoveRange.getBegin(),
                            Rewrite.getRangeSize(Hint.RemoveRange),
                            Hint.CodeToInsert))
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
void FixItRewriter::Diag(SourceLocation Loc, unsigned DiagID) {
  // When producing this diagnostic, we temporarily bypass ourselves,
  // clear out any current diagnostic, and let the downstream client
  // format the diagnostic.
  Diags.takeClient();
  Diags.setClient(Client);
  Diags.Clear();
  Diags.Report(Loc, DiagID);
  Diags.takeClient();
  Diags.setClient(this);
}

DiagnosticConsumer *FixItRewriter::clone(DiagnosticsEngine &Diags) const {
  return new FixItRewriter(Diags, Diags.getSourceManager(), 
                           Rewrite.getLangOpts(), FixItOpts);
}

FixItOptions::~FixItOptions() {}
