//===-- Core/Transform.cpp - Transform Base Class Def'n -------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief This file provides the definition for the base Transform class from
/// which all transforms must subclass.
///
//===----------------------------------------------------------------------===//

#include "Core/Transform.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/ADT/STLExtras.h"

template class llvm::Registry<TransformFactory>;

using namespace clang;

llvm::cl::OptionCategory TransformsOptionsCategory("Transforms' options");

namespace {

using namespace tooling;
using namespace ast_matchers;

/// \brief Custom FrontendActionFactory to produce FrontendActions that simply
/// forward (Begin|End)SourceFileAction calls to a given Transform.
class ActionFactory : public clang::tooling::FrontendActionFactory {
public:
  ActionFactory(MatchFinder &Finder, Transform &Owner)
  : Finder(Finder), Owner(Owner) {}

  FrontendAction *create() override {
    return new FactoryAdaptor(Finder, Owner);
  }

private:
  class FactoryAdaptor : public ASTFrontendAction {
  public:
    FactoryAdaptor(MatchFinder &Finder, Transform &Owner)
        : Finder(Finder), Owner(Owner) {}

    std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &,
                                                   StringRef) override {
      return Finder.newASTConsumer();
    }

    bool BeginSourceFileAction(CompilerInstance &CI,
                               StringRef Filename) override {
      if (!ASTFrontendAction::BeginSourceFileAction(CI, Filename))
        return false;

      return Owner.handleBeginSource(CI, Filename);
    }

    void EndSourceFileAction() override {
      Owner.handleEndSource();
      return ASTFrontendAction::EndSourceFileAction();
    }

  private:
    MatchFinder &Finder;
    Transform &Owner;
  };

  MatchFinder &Finder;
  Transform &Owner;
};
} // namespace

Transform::Transform(llvm::StringRef Name, const TransformOptions &Options)
    : Name(Name), GlobalOptions(Options) {
  Reset();
}

Transform::~Transform() = default;

bool Transform::isFileModifiable(const SourceManager &SM,
                                 SourceLocation Loc) const {
  if (SM.isWrittenInMainFile(Loc))
    return true;

  const FileEntry *FE = SM.getFileEntryForID(SM.getFileID(Loc));
  if (!FE)
    return false;

  return GlobalOptions.ModifiableFiles.isFileIncluded(FE->getName());
}

bool Transform::handleBeginSource(CompilerInstance &CI, StringRef Filename) {
  CurrentSource = Filename;

  if (Options().EnableTiming) {
    Timings.push_back(std::make_pair(Filename.str(), llvm::TimeRecord()));
    Timings.back().second -= llvm::TimeRecord::getCurrentTime(true);
  }
  return true;
}

void Transform::handleEndSource() {
  CurrentSource.clear();
  if (Options().EnableTiming)
    Timings.back().second += llvm::TimeRecord::getCurrentTime(false);
}

void Transform::addTiming(llvm::StringRef Label, llvm::TimeRecord Duration) {
  Timings.push_back(std::make_pair(Label.str(), Duration));
}

bool
Transform::addReplacementForCurrentTU(const clang::tooling::Replacement &R) {
  if (CurrentSource.empty())
    return false;

  TranslationUnitReplacements &TU = Replacements[CurrentSource];
  if (TU.MainSourceFile.empty())
    TU.MainSourceFile = CurrentSource;
  TU.Replacements.push_back(R);

  return true;
}

std::unique_ptr<FrontendActionFactory>
Transform::createActionFactory(MatchFinder &Finder) {
  return llvm::make_unique<ActionFactory>(Finder, /*Owner=*/*this);
}

Version Version::getFromString(llvm::StringRef VersionStr) {
  llvm::StringRef MajorStr, MinorStr;
  Version V;

  std::tie(MajorStr, MinorStr) = VersionStr.split('.');
  if (!MinorStr.empty()) {
    llvm::StringRef Ignore;
    std::tie(MinorStr, Ignore) = MinorStr.split('.');
    if (MinorStr.getAsInteger(10, V.Minor))
      return Version();
  }
  if (MajorStr.getAsInteger(10, V.Major))
    return Version();
  return V;
}

TransformFactory::~TransformFactory() = default;

namespace {
bool versionSupported(Version Required, Version AvailableSince) {
  // null version, means no requirements, means supported
  if (Required.isNull())
    return true;
  return Required >= AvailableSince;
}
} // end anonymous namespace

bool TransformFactory::supportsCompilers(CompilerVersions Required) const {
  return versionSupported(Required.Clang, Since.Clang) &&
         versionSupported(Required.Gcc, Since.Gcc) &&
         versionSupported(Required.Icc, Since.Icc) &&
         versionSupported(Required.Msvc, Since.Msvc);
}
