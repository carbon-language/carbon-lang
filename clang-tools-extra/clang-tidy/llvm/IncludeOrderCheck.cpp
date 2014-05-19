//===--- IncludeOrderCheck.cpp - clang-tidy -------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "IncludeOrderCheck.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Lex/PPCallbacks.h"
#include "clang/Lex/Preprocessor.h"

namespace clang {
namespace tidy {

namespace {
class IncludeOrderPPCallbacks : public PPCallbacks {
public:
  explicit IncludeOrderPPCallbacks(IncludeOrderCheck &Check) : Check(Check) {}

  void InclusionDirective(SourceLocation HashLoc, const Token &IncludeTok,
                          StringRef FileName, bool IsAngled,
                          CharSourceRange FilenameRange, const FileEntry *File,
                          StringRef SearchPath, StringRef RelativePath,
                          const Module *Imported) override {
    // FIXME: This is a dummy implementation to show how to get at preprocessor
    // information. Implement a real include order check.
    Check.diag(HashLoc, "This is an include");
  }

private:
  IncludeOrderCheck &Check;
};
} // namespace

void IncludeOrderCheck::registerPPCallbacks(CompilerInstance &Compiler) {
  Compiler.getPreprocessor()
      .addPPCallbacks(new IncludeOrderPPCallbacks(*this));
}

} // namespace tidy
} // namespace clang
