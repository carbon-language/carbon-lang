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
  case PreprocessJobClass: return "preprocessor";
  case PrecompileJobClass: return "precompiler";
  case AnalyzeJobClass: return "analyzer";
  case CompileJobClass: return "compiler";
  case AssembleJobClass: return "assembler";
  case LinkJobClass: return "linker";
  case LipoJobClass: return "lipo";
  }
  
  assert(0 && "invalid class");
  return 0;
}
