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

static cl::OptionCategory TransformCategory("Transforms");

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

void
Transforms::createSelectedTransforms(const TransformOptions &GlobalOptions) {
  for (TransformFactoryRegistry::iterator I = TransformFactoryRegistry::begin(),
                                          E = TransformFactoryRegistry::end();
       I != E; ++I) {
    bool OptionEnabled = *Options[I->getName()];

    if (!OptionEnabled)
      continue;

    llvm::OwningPtr<TransformFactory> Factory(I->instantiate());
    ChosenTransforms.push_back(Factory->createTransform(GlobalOptions));
  }
}
