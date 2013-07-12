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
#include "Core/FileOverrides.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Tooling/Tooling.h"

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

  virtual FrontendAction *create() LLVM_OVERRIDE {
    return new FactoryAdaptor(Finder, Owner);
  }

private:
  class FactoryAdaptor : public ASTFrontendAction {
  public:
    FactoryAdaptor(MatchFinder &Finder, Transform &Owner)
        : Finder(Finder), Owner(Owner) {}

    ASTConsumer *CreateASTConsumer(CompilerInstance &, StringRef) {
      return Finder.newASTConsumer();
    }

    virtual bool BeginSourceFileAction(CompilerInstance &CI,
                                       StringRef Filename) LLVM_OVERRIDE {
      if (!ASTFrontendAction::BeginSourceFileAction(CI, Filename))
        return false;

      return Owner.handleBeginSource(CI, Filename);
    }

    virtual void EndSourceFileAction() LLVM_OVERRIDE {
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
    : Name(Name), GlobalOptions(Options), Overrides(0) {
  Reset();
}

Transform::~Transform() {}

bool Transform::isFileModifiable(const SourceManager &SM,
                                 const SourceLocation &Loc) const {
  if (SM.isFromMainFile(Loc))
    return true;

  if (!GlobalOptions.EnableHeaderModifications)
    return false;

  const FileEntry *FE = SM.getFileEntryForID(SM.getFileID(Loc));
  if (!FE)
    return false;

  return GlobalOptions.ModifiableHeaders.isFileIncluded(FE->getName());
}

bool Transform::handleBeginSource(CompilerInstance &CI, StringRef Filename) {
  assert(Overrides != 0 && "Subclass transform didn't provide InputState");

  CurrentSource = Filename.str();

  FileOverrides::const_iterator I = Overrides->find(CurrentSource);
  if (I != Overrides->end())
    I->second->applyOverrides(CI.getSourceManager());

  Replace.clear();

  if (Options().EnableTiming) {
    Timings.push_back(std::make_pair(Filename.str(), llvm::TimeRecord()));
    Timings.back().second -= llvm::TimeRecord::getCurrentTime(true);
  }
  return true;
}

void Transform::handleEndSource() {
  if (!getReplacements().empty()) {
    SourceOverrides &SO = Overrides->getOrCreate(CurrentSource);
    SO.applyReplacements(getReplacements());
  }

  if (Options().EnableTiming)
    Timings.back().second += llvm::TimeRecord::getCurrentTime(false);
}

void Transform::addTiming(llvm::StringRef Label, llvm::TimeRecord Duration) {
  Timings.push_back(std::make_pair(Label.str(), Duration));
}

FrontendActionFactory *
Transform::createActionFactory(MatchFinder &Finder) {
  return new ActionFactory(Finder, /*Owner=*/ *this);
}
