//===-- Core/Reformatting.cpp - LibFormat integration ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief This file provides the LibFormat integration used to reformat
/// migrated code.
///
//===----------------------------------------------------------------------===//

#include "Core/Reformatting.h"
#include "Core/FileOverrides.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/DiagnosticOptions.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Lex/Lexer.h"

using namespace clang;

void Reformatter::reformatChanges(SourceOverrides &Overrides) {
  llvm::IntrusiveRefCntPtr<clang::DiagnosticOptions> DiagOpts(
      new DiagnosticOptions());
  DiagnosticsEngine Diagnostics(
      llvm::IntrusiveRefCntPtr<DiagnosticIDs>(new DiagnosticIDs()),
      DiagOpts.getPtr());
  FileManager Files((FileSystemOptions()));
  SourceManager SM(Diagnostics, Files);

  reformatChanges(Overrides, SM);
}

void Reformatter::reformatChanges(SourceOverrides &Overrides,
                                  clang::SourceManager &SM) {
  tooling::Replacements Replaces;
  Overrides.applyOverrides(SM);
  if (Overrides.isSourceOverriden())
    Replaces = reformatSingleFile(Overrides.getMainFileName(),
                                  Overrides.getChangedRanges(), SM);

  for (HeaderOverrides::const_iterator I = Overrides.headers_begin(),
                                       E = Overrides.headers_end();
       I != E; ++I) {
    const HeaderOverride &Header = I->getValue();
    const tooling::Replacements &HeaderReplaces =
        reformatSingleFile(Header.getFileName(), Header.getChanges(), SM);
    Replaces.insert(HeaderReplaces.begin(), HeaderReplaces.end());
  }
  Overrides.applyReplacements(Replaces, SM, "reformatter");
}

tooling::Replacements Reformatter::reformatSingleFile(
    llvm::StringRef FileName, const ChangedRanges &Changes, SourceManager &SM) {
  const clang::FileEntry *Entry = SM.getFileManager().getFile(FileName);
  assert(Entry && "expected an existing file");

  FileID ID = SM.translateFile(Entry);
  if (ID.isInvalid())
    ID = SM.createFileID(Entry, SourceLocation(), clang::SrcMgr::C_User);

  std::vector<CharSourceRange> ReformatRanges;
  SourceLocation StartOfFile = SM.getLocForStartOfFile(ID);
  for (ChangedRanges::const_iterator I = Changes.begin(), E = Changes.end();
       I != E; ++I) {
    SourceLocation Start = StartOfFile.getLocWithOffset(I->getOffset());
    SourceLocation End = Start.getLocWithOffset(I->getLength());
    ReformatRanges.push_back(CharSourceRange::getCharRange(Start, End));
  }

  Lexer Lex(ID, SM.getBuffer(ID), SM, getFormattingLangOpts(Style.Standard));
  return format::reformat(Style, Lex, SM, ReformatRanges);
}
