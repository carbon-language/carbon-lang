//===--- Action.cpp - Abstract compilation steps ------------------------*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "clang/Driver/Action.h"

#include <cassert>
using namespace clang::driver;

Action::~Action() {}

const char *Action::getClassName(ActionClass AC) {
  switch (AC) {
  case InputClass: return "input";
  case BindArchClass: return "bind-arch";
  case PreprocessJobClass: return "preprocess";
  case PrecompileJobClass: return "precompile";
  case AnalyzeJobClass: return "analyze";
  case CompileJobClass: return "compile";
  case AssembleJobClass: return "assemble";
  case LinkJobClass: return "link";
  case LipoJobClass: return "lipo";
  }
  
  assert(0 && "invalid class");
  return 0;
}
