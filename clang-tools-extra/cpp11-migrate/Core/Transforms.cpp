//===-- cpp11-migrate/Transforms.cpp - class Transforms Impl ----*- C++ -*-===//
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
#include "LoopConvert/LoopConvert.h"
#include "UseNullptr/UseNullptr.h"
#include "UseAuto/UseAuto.h"

namespace cl = llvm::cl;

template <typename T>
Transform *ConstructTransform() {
  return new T();
}

Transforms::~Transforms() {
  for (std::vector<Transform*>::iterator I = ChosenTransforms.begin(),
       E = ChosenTransforms.end(); I != E; ++I) {
    delete *I;
  }
  for (OptionVec::iterator I = Options.begin(),
       E = Options.end(); I != E; ++I) {
    delete I->first;
  }
}

void Transforms::createTransformOpts() {
  Options.push_back(
    OptionVec::value_type(
      new cl::opt<bool>("loop-convert",
        cl::desc("Make use of range-based for loops where possible")),
      &ConstructTransform<LoopConvertTransform>));

  Options.push_back(
    OptionVec::value_type(
      new cl::opt<bool>("use-nullptr",
        cl::desc("Make use of nullptr keyword where possible")),
      &ConstructTransform<UseNullptrTransform>));

  Options.push_back(
    OptionVec::value_type(
      new cl::opt<bool>("use-auto",
        cl::desc("Use of 'auto' type specifier")),
      &ConstructTransform<UseAutoTransform>));

  // Add more transform options here.
}

void Transforms::createSelectedTransforms() {
  for (OptionVec::iterator I = Options.begin(),
       E = Options.end(); I != E; ++I) {
    if (*I->first) {
      ChosenTransforms.push_back(I->second());
    }
  }
}
