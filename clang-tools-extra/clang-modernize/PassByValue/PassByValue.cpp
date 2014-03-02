//===-- PassByValue.cpp ---------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief This file provides the implementation of the ReplaceAutoPtrTransform
/// class.
///
//===----------------------------------------------------------------------===//

#include "PassByValue.h"
#include "PassByValueActions.h"
#include "PassByValueMatchers.h"

using namespace clang;
using namespace clang::tooling;
using namespace clang::ast_matchers;

int PassByValueTransform::apply(const tooling::CompilationDatabase &Database,
                                const std::vector<std::string> &SourcePaths) {
  ClangTool Tool(Database, SourcePaths);
  unsigned AcceptedChanges = 0;
  unsigned RejectedChanges = 0;
  MatchFinder Finder;
  ConstructorParamReplacer Replacer(AcceptedChanges, RejectedChanges,
                                    /*Owner=*/ *this);

  Finder.addMatcher(makePassByValueCtorParamMatcher(), &Replacer);

  // make the replacer available to handleBeginSource()
  this->Replacer = &Replacer;

  if (Tool.run(createActionFactory(Finder))) {
    llvm::errs() << "Error encountered during translation.\n";
    return 1;
  }

  setAcceptedChanges(AcceptedChanges);
  setRejectedChanges(RejectedChanges);
  return 0;
}

bool PassByValueTransform::handleBeginSource(CompilerInstance &CI,
                                             llvm::StringRef Filename) {
  assert(Replacer && "Replacer not set");
  IncludeManager.reset(new IncludeDirectives(CI));
  Replacer->setIncludeDirectives(IncludeManager.get());
  return Transform::handleBeginSource(CI, Filename);
}

struct PassByValueFactory : TransformFactory {
  PassByValueFactory() {
    // Based on the Replace Auto-Ptr Transform that is also using std::move().
    Since.Clang = Version(3, 0);
    Since.Gcc = Version(4, 6);
    Since.Icc = Version(13);
    Since.Msvc = Version(11);
  }

  Transform *createTransform(const TransformOptions &Opts) override {
    return new PassByValueTransform(Opts);
  }
};

// Register the factory using this statically initialized variable.
static TransformFactoryRegistry::Add<PassByValueFactory>
X("pass-by-value", "Pass parameters by value where possible");

// This anchor is used to force the linker to link in the generated object file
// and thus register the factory.
volatile int PassByValueTransformAnchorSource = 0;
