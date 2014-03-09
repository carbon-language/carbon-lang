//===-- Core/Transforms.cpp - class Transforms Impl -----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief This file provides the implementation for class Transforms.
///
//===----------------------------------------------------------------------===//

#include "Core/Transforms.h"
#include "Core/Transform.h"

namespace cl = llvm::cl;

cl::OptionCategory TransformCategory("Transforms");

Transforms::~Transforms() {
  for (std::vector<Transform *>::iterator I = ChosenTransforms.begin(),
                                          E = ChosenTransforms.end();
       I != E; ++I)
    delete *I;

  for (OptionMap::iterator I = Options.begin(), E = Options.end(); I != E; ++I)
    delete I->getValue();
}

void Transforms::registerTransforms() {
  for (TransformFactoryRegistry::iterator I = TransformFactoryRegistry::begin(),
                                          E = TransformFactoryRegistry::end();
       I != E; ++I)
    Options[I->getName()] = new cl::opt<bool>(
        I->getName(), cl::desc(I->getDesc()), cl::cat(TransformCategory));
}

bool Transforms::hasAnyExplicitOption() const {
  for (OptionMap::const_iterator I = Options.begin(), E = Options.end(); I != E;
       ++I)
    if (*I->second)
      return true;
  return false;
}

void
Transforms::createSelectedTransforms(const TransformOptions &GlobalOptions,
                                     const CompilerVersions &RequiredVersions) {
  // if at least one transform is set explicitly on the command line, do not
  // enable non-explicit ones
  bool EnableAllTransformsByDefault = !hasAnyExplicitOption();

  for (TransformFactoryRegistry::iterator I = TransformFactoryRegistry::begin(),
                                          E = TransformFactoryRegistry::end();
       I != E; ++I) {
    bool ExplicitlyEnabled = *Options[I->getName()];
    bool OptionEnabled = EnableAllTransformsByDefault || ExplicitlyEnabled;

    if (!OptionEnabled)
      continue;

    std::unique_ptr<TransformFactory> Factory(I->instantiate());
    if (Factory->supportsCompilers(RequiredVersions))
      ChosenTransforms.push_back(Factory->createTransform(GlobalOptions));
    else if (ExplicitlyEnabled)
      llvm::errs() << "note: " << '-' << I->getName()
                   << ": transform not available for specified compilers\n";
  }
}
