//===--- SuspiciousIncludeCheck.cpp - clang-tidy --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SuspiciousIncludeCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/Lex/Preprocessor.h"

namespace clang {
namespace tidy {
namespace bugprone {

namespace {
class SuspiciousIncludePPCallbacks : public PPCallbacks {
public:
  explicit SuspiciousIncludePPCallbacks(SuspiciousIncludeCheck &Check,
                                        const SourceManager &SM,
                                        Preprocessor *PP)
      : Check(Check), PP(PP) {}

  void InclusionDirective(SourceLocation HashLoc, const Token &IncludeTok,
                          StringRef FileName, bool IsAngled,
                          CharSourceRange FilenameRange,
                          Optional<FileEntryRef> File, StringRef SearchPath,
                          StringRef RelativePath, const Module *Imported,
                          SrcMgr::CharacteristicKind FileType) override;

private:
  SuspiciousIncludeCheck &Check;
  Preprocessor *PP;
};
} // namespace

SuspiciousIncludeCheck::SuspiciousIncludeCheck(StringRef Name,
                                               ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      RawStringHeaderFileExtensions(Options.getLocalOrGlobal(
          "HeaderFileExtensions", utils::defaultHeaderFileExtensions())),
      RawStringImplementationFileExtensions(Options.getLocalOrGlobal(
          "ImplementationFileExtensions",
          utils::defaultImplementationFileExtensions())) {
  if (!utils::parseFileExtensions(RawStringImplementationFileExtensions,
                                  ImplementationFileExtensions,
                                  utils::defaultFileExtensionDelimiters())) {
    this->configurationDiag("Invalid implementation file extension: '%0'")
        << RawStringImplementationFileExtensions;
  }

  if (!utils::parseFileExtensions(RawStringHeaderFileExtensions,
                                  HeaderFileExtensions,
                                  utils::defaultFileExtensionDelimiters())) {
    this->configurationDiag("Invalid header file extension: '%0'")
        << RawStringHeaderFileExtensions;
  }
}

void SuspiciousIncludeCheck::storeOptions(ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "ImplementationFileExtensions",
                RawStringImplementationFileExtensions);
  Options.store(Opts, "HeaderFileExtensions", RawStringHeaderFileExtensions);
}

void SuspiciousIncludeCheck::registerPPCallbacks(
    const SourceManager &SM, Preprocessor *PP, Preprocessor *ModuleExpanderPP) {
  PP->addPPCallbacks(
      ::std::make_unique<SuspiciousIncludePPCallbacks>(*this, SM, PP));
}

void SuspiciousIncludePPCallbacks::InclusionDirective(
    SourceLocation HashLoc, const Token &IncludeTok, StringRef FileName,
    bool IsAngled, CharSourceRange FilenameRange, Optional<FileEntryRef> File,
    StringRef SearchPath, StringRef RelativePath, const Module *Imported,
    SrcMgr::CharacteristicKind FileType) {
  if (IncludeTok.getIdentifierInfo()->getPPKeywordID() == tok::pp_import)
    return;

  SourceLocation DiagLoc = FilenameRange.getBegin().getLocWithOffset(1);

  const Optional<StringRef> IFE =
      utils::getFileExtension(FileName, Check.ImplementationFileExtensions);
  if (!IFE)
    return;

  Check.diag(DiagLoc, "suspicious #%0 of file with '%1' extension")
      << IncludeTok.getIdentifierInfo()->getName() << *IFE;

  for (const auto &HFE : Check.HeaderFileExtensions) {
    SmallString<128> GuessedFileName(FileName);
    llvm::sys::path::replace_extension(GuessedFileName,
                                       (HFE.size() ? "." : "") + HFE);

    Optional<FileEntryRef> File =
        PP->LookupFile(DiagLoc, GuessedFileName, IsAngled, nullptr, nullptr,
                       nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
    if (File) {
      Check.diag(DiagLoc, "did you mean to include '%0'?", DiagnosticIDs::Note)
          << GuessedFileName;
    }
  }
}

} // namespace bugprone
} // namespace tidy
} // namespace clang
