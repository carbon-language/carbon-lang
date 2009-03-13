//===--- Action.h - Abstract compilation steps ------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_DRIVER_ACTION_H_
#define CLANG_DRIVER_ACTION_H_

#include "llvm/ADT/SmallVector.h"

#include "clang/Driver/Types.h"
#include "clang/Driver/Util.h"

namespace clang {
namespace driver {
  class Arg;

/// Action - Represent an abstract compilation step to perform. 
///
/// An action represents an edge in the compilation graph; typically
/// it is a job to transform an input using some tool.
///
/// The current driver is hard wired to expect actions which produce a
/// single primary output, at least in terms of controlling the
/// compilation. Actions can produce auxiliary files, but can only
/// produce a single output to feed into subsequent actions.
class Action {
  /// The output type of this action.
  types::ID Type;
  
  ActionList Inputs;

protected:
  Action(types::ID _Type) : Type(_Type) {}
  Action(Action *Input, types::ID _Type) : Type(_Type),
                                           Inputs(&Input, &Input + 1) {}
  Action(const ActionList &_Inputs, types::ID _Type) : Type(_Type),
                                                       Inputs(_Inputs) {}  
public:
  virtual ~Action();
  
  types::ID getType() { return Type; }
};

class InputAction : public Action {
  const Arg &Input;
public:
  InputAction(const Arg &_Input, types::ID _Type) : Action(_Type),
                                                    Input(_Input) {}
};

class BindArchAction : public Action {
  const char *ArchName;

public:
  BindArchAction(Action *Input, const char *_ArchName) 
    : Action(Input, Input->getType()), ArchName(_ArchName) {
  }
};

class JobAction : public Action {
protected:
  JobAction(Action *Input, types::ID Type) : Action(Input, Type) {}
  JobAction(const ActionList &Inputs, types::ID Type) : Action(Inputs, Type) {}
};

class PreprocessJobAction : public JobAction {
public:
  PreprocessJobAction(Action *Input, types::ID OutputType)
    : JobAction(Input, OutputType) {
  }
};

class PrecompileJobAction : public JobAction {
public:
  PrecompileJobAction(Action *Input, types::ID OutputType)
    : JobAction(Input, OutputType) {
  }
};

class AnalyzeJobAction : public JobAction {
public:
  AnalyzeJobAction(Action *Input, types::ID OutputType)
    : JobAction(Input, OutputType) {
  }
};

class CompileJobAction : public JobAction {
public:
  CompileJobAction(Action *Input, types::ID OutputType)
    : JobAction(Input, OutputType) {
  }
};

class AssembleJobAction : public JobAction {
public:
  AssembleJobAction(Action *Input, types::ID OutputType)
    : JobAction(Input, OutputType) {
  }
};

class LinkJobAction : public JobAction {
public:
  LinkJobAction(ActionList &Inputs, types::ID Type) 
    : JobAction(Inputs, Type) {}
};

class LipoJobAction : public JobAction {
public:
  LipoJobAction(ActionList &Inputs, types::ID Type) 
    : JobAction(Inputs, Type) {}
};

} // end namespace driver
} // end namespace clang

#endif
