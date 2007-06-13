//===--- TextDiagnostics.cpp - Text Diagnostics Parent Class --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Bill Wendling and is distributed under the
// University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This is the parent class for all text diagnostics.
//
//===----------------------------------------------------------------------===//

#include "TextDiagnostics.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Lex/HeaderSearch.h"
using namespace llvm;
using namespace clang;

TextDiagnostics:: ~TextDiagnostics() {}

std::string TextDiagnostics::FormatDiagnostic(Diagnostic::Level Level,
                                              diag::kind ID,
                                              const std::string *Strs,
                                              unsigned NumStrs) {
  std::string Msg = Diagnostic::getDescription(ID);
  
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

bool TextDiagnostics::IgnoreDiagnostic(Diagnostic::Level Level,
                                       SourceLocation Pos) {
  if (Pos.isValid()) {
    // If this is a warning or note, and if it a system header, suppress the
    // diagnostic.
    if (Level == Diagnostic::Warning ||
        Level == Diagnostic::Note) {
      SourceLocation PhysLoc = SourceMgr.getPhysicalLoc(Pos);
      const FileEntry *F = SourceMgr.getFileEntryForFileID(PhysLoc.getFileID());
      if (F) {
        DirectoryLookup::DirType DirInfo = TheHeaderSearch->getFileDirFlavor(F);
        if (DirInfo == DirectoryLookup::SystemHeaderDir ||
            DirInfo == DirectoryLookup::ExternCSystemHeaderDir)
          return true;
      }
    }
  }

  return false;
}
