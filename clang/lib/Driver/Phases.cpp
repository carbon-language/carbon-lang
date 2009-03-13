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
  case Preprocess: return "preprocessor";
  case Precompile: return "precompiler";
  case Compile: return "compiler";
  case Assemble: return "assembler";
  case Link: return "linker";
  }

  assert(0 && "Invalid phase id.");
  return 0;
}
