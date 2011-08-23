//===--- Action.cpp - Abstract compilation steps --------------------------===//
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

Action::~Action() {
  if (OwnsInputs) {
    for (iterator it = begin(), ie = end(); it != ie; ++it)
      delete *it;
  }
}

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
  case DsymutilJobClass: return "dsymutil";
  case VerifyJobClass: return "verify";
  }

  assert(0 && "invalid class");
  return 0;
}

InputAction::InputAction(const Arg &_Input, types::ID _Type)
  : Action(InputClass, _Type), Input(_Input) {
}

BindArchAction::BindArchAction(Action *Input, const char *_ArchName)
  : Action(BindArchClass, Input, Input->getType()), ArchName(_ArchName) {
}

JobAction::JobAction(ActionClass Kind, Action *Input, types::ID Type)
  : Action(Kind, Input, Type) {
}

JobAction::JobAction(ActionClass Kind, const ActionList &Inputs, types::ID Type)
  : Action(Kind, Inputs, Type) {
}

PreprocessJobAction::PreprocessJobAction(Action *Input, types::ID OutputType)
  : JobAction(PreprocessJobClass, Input, OutputType) {
}

PrecompileJobAction::PrecompileJobAction(Action *Input, types::ID OutputType)
  : JobAction(PrecompileJobClass, Input, OutputType) {
}

AnalyzeJobAction::AnalyzeJobAction(Action *Input, types::ID OutputType)
  : JobAction(AnalyzeJobClass, Input, OutputType) {
}

CompileJobAction::CompileJobAction(Action *Input, types::ID OutputType)
  : JobAction(CompileJobClass, Input, OutputType) {
}

AssembleJobAction::AssembleJobAction(Action *Input, types::ID OutputType)
  : JobAction(AssembleJobClass, Input, OutputType) {
}

LinkJobAction::LinkJobAction(ActionList &Inputs, types::ID Type)
  : JobAction(LinkJobClass, Inputs, Type) {
}

LipoJobAction::LipoJobAction(ActionList &Inputs, types::ID Type)
  : JobAction(LipoJobClass, Inputs, Type) {
}

DsymutilJobAction::DsymutilJobAction(ActionList &Inputs, types::ID Type)
  : JobAction(DsymutilJobClass, Inputs, Type) {
}

VerifyJobAction::VerifyJobAction(ActionList &Inputs, types::ID Type)
  : JobAction(VerifyJobClass, Inputs, Type) {
}
