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

void Reformatter::reformatChanges(const FileOverrides &FileStates,
                                  clang::SourceManager &SM,
                                  clang::tooling::ReplacementsVec &Replaces) {
  FileStates.applyOverrides(SM);

  for (FileOverrides::ChangeMap::const_iterator
           I = FileStates.getChangedRanges().begin(),
           E = FileStates.getChangedRanges().end();
       I != E; ++I) {
    reformatSingleFile(I->getKey(), I->getValue(), SM, Replaces);
  }
}

void Reformatter::reformatSingleFile(
    const llvm::StringRef FileName, const ChangedRanges &Changes,
    SourceManager &SM, clang::tooling::ReplacementsVec &FormatReplacements) {

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
  const tooling::Replacements &R =
      format::reformat(Style, Lex, SM, ReformatRanges);
  std::copy(R.begin(), R.end(), std::back_inserter(FormatReplacements));
}
