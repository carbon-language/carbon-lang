//===--- HeaderIncludes.cpp - Generate Header Includes --------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "clang/Frontend/Utils.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Lex/Preprocessor.h"
/*
#include "clang/Basic/Diagnostic.h"
#include "clang/Frontend/PreprocessorOutputOptions.h"
#include "clang/Lex/MacroInfo.h"
#include "clang/Lex/PPCallbacks.h"
#include "clang/Lex/Pragma.h"
#include "clang/Lex/TokenConcatenation.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Config/config.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdio>
*/
using namespace clang;

namespace {
class HeaderIncludesCallback : public PPCallbacks {
  SourceManager &SM;
  unsigned CurrentIncludeDepth;
  bool HasProcessedPredefines;

public:
  HeaderIncludesCallback(const Preprocessor *PP)
    : SM(PP->getSourceManager()), CurrentIncludeDepth(0),
      HasProcessedPredefines(false) {}

  virtual void FileChanged(SourceLocation Loc, FileChangeReason Reason,
                           SrcMgr::CharacteristicKind FileType);
};
}

void clang::AttachHeaderIncludeGen(Preprocessor &PP) {
  PP.addPPCallbacks(new HeaderIncludesCallback(&PP));
}

void HeaderIncludesCallback::FileChanged(SourceLocation Loc,
                                         FileChangeReason Reason,
                                       SrcMgr::CharacteristicKind NewFileType) {
  // Unless we are exiting a #include, make sure to skip ahead to the line the
  // #include directive was at.
  PresumedLoc UserLoc = SM.getPresumedLoc(Loc);
  if (UserLoc.isInvalid())
    return;
  
  // Adjust the current include depth.
  if (Reason == PPCallbacks::EnterFile) {
    ++CurrentIncludeDepth;
  } else {
    if (CurrentIncludeDepth)
      --CurrentIncludeDepth;

    // We track when we are done with the predefines by watching for the first
    // place where we drop back to a nesting depth of 0.
    if (CurrentIncludeDepth == 0 && !HasProcessedPredefines)
      HasProcessedPredefines = true;
  }
  
  // Dump the header include information we are past the predefines buffer.
  if (HasProcessedPredefines && Reason == PPCallbacks::EnterFile) {
    // Write to a temporary string to avoid unnecessary flushing on errs().
    llvm::SmallString<512> Filename(UserLoc.getFilename());
    Lexer::Stringify(Filename);

    llvm::SmallString<256> Msg;
    for (unsigned i = 0; i != CurrentIncludeDepth; ++i)
      Msg += '.';
    Msg += ' ';
    Msg += Filename;
    Msg += '\n';

    llvm::errs() << Msg;
  }
}

