//===--- Phases.cpp - Transformations on Driver Types -------------------*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "clang/Driver/Phases.h"

#include <cassert>

using namespace clang::driver;

const char *phases::getPhaseName(ID Id) {
  switch (Id) {
  case Preprocess: return "preprocess";
  case Precompile: return "precompile";
  case Compile: return "compile";
  case Assemble: return "assemble";
  case Link: return "link";
  }

  assert(0 && "Invalid phase id.");
  return 0;
}
