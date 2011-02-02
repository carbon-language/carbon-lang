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
#include <cstdio>
using namespace clang;

namespace {
class HeaderIncludesCallback : public PPCallbacks {
  SourceManager &SM;
  FILE *OutputFile;
  unsigned CurrentIncludeDepth;
  bool HasProcessedPredefines;
  bool OwnsOutputFile;
  bool ShowAllHeaders;

public:
  HeaderIncludesCallback(const Preprocessor *PP, bool ShowAllHeaders_,
                         FILE *OutputFile_, bool OwnsOutputFile_)
    : SM(PP->getSourceManager()), OutputFile(OutputFile_),
      CurrentIncludeDepth(0), HasProcessedPredefines(false),
      OwnsOutputFile(OwnsOutputFile_), ShowAllHeaders(ShowAllHeaders_) {}

  ~HeaderIncludesCallback() {
    if (OwnsOutputFile)
      fclose(OutputFile);
  }

  virtual void FileChanged(SourceLocation Loc, FileChangeReason Reason,
                           SrcMgr::CharacteristicKind FileType);
};
}

void clang::AttachHeaderIncludeGen(Preprocessor &PP, bool ShowAllHeaders,
                                   llvm::StringRef OutputPath) {
  FILE *OutputFile;
  bool OwnsOutputFile;

  // Open the output file, if used.
  if (OutputPath.empty()) {
    OutputFile = stderr;
    OwnsOutputFile = false;
  } else {
    OutputFile = fopen(OutputPath.str().c_str(), "a");
    OwnsOutputFile = true;
  }

  PP.addPPCallbacks(new HeaderIncludesCallback(&PP, ShowAllHeaders,
                                               OutputFile, OwnsOutputFile));
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

  // Show the header if we are (a) past the predefines, or (b) showing all
  // headers and in the predefines at a depth past the initial file and command
  // line buffers.
  bool ShowHeader = (HasProcessedPredefines ||
                     (ShowAllHeaders && CurrentIncludeDepth > 2));

  // Dump the header include information we are past the predefines buffer or
  // are showing all headers.
  if (ShowHeader && Reason == PPCallbacks::EnterFile) {
    // Write to a temporary string to avoid unnecessary flushing on errs().
    llvm::SmallString<512> Filename(UserLoc.getFilename());
    Lexer::Stringify(Filename);

    llvm::SmallString<256> Msg;
    for (unsigned i = 0; i != CurrentIncludeDepth; ++i)
      Msg += '.';
    Msg += ' ';
    Msg += Filename;
    Msg += '\n';

    fwrite(Msg.data(), Msg.size(), 1, OutputFile);
  }
}

