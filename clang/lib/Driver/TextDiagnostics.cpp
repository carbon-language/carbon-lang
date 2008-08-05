//===--- TextDiagnostics.cpp - Text Diagnostics Parent Class --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This is the parent class for all text diagnostics.
//
//===----------------------------------------------------------------------===//

#include "clang/Driver/TextDiagnostics.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Lex/HeaderSearch.h"
using namespace clang;

TextDiagnostics:: ~TextDiagnostics() {}

std::string TextDiagnostics::FormatDiagnostic(Diagnostic &Diags,
                                              Diagnostic::Level Level,
                                              diag::kind ID,
                                              const std::string *Strs,
                                              unsigned NumStrs) {
  std::string Msg = Diags.getDescription(ID);
  
  // Replace all instances of %0 in Msg with 'Extra'.
  for (unsigned i = 0; i < Msg.size() - 1; ++i) {
    if (Msg[i] == '%' && isdigit(Msg[i + 1])) {
      unsigned StrNo = Msg[i + 1] - '0';
      Msg = std::string(Msg.begin(), Msg.begin() + i) +
            (StrNo < NumStrs ? Strs[StrNo] : "<<<INTERNAL ERROR>>>") +
            std::string(Msg.begin() + i + 2, Msg.end());
    }
  }

  return Msg;
}

bool TextDiagnostics::isInSystemHeader(FullSourceLoc Pos) const {
  if (!Pos.isValid()) return false;
  
  if (const FileEntry *F = Pos.getFileEntryForLoc()) {
    DirectoryLookup::DirType DirInfo = TheHeaderSearch->getFileDirFlavor(F);
    if (DirInfo == DirectoryLookup::SystemHeaderDir ||
        DirInfo == DirectoryLookup::ExternCSystemHeaderDir)
      return true;
  }

  return false;
}
