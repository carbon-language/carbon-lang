//===- PTXParamManager.cpp - Manager for .param variables -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the PTXParamManager class.
//
//===----------------------------------------------------------------------===//

#include "PTX.h"
#include "PTXParamManager.h"
#include "llvm/ADT/StringExtras.h"

using namespace llvm;

PTXParamManager::PTXParamManager() {
}

unsigned PTXParamManager::addArgumentParam(unsigned Size) {
  PTXParam Param;
  Param.Type = PTX_PARAM_TYPE_ARGUMENT;
  Param.Size = Size;

  std::string Name;
  Name = "__param_";
  Name += utostr(ArgumentParams.size()+1);
  Param.Name = Name;

  unsigned Index = AllParams.size();
  AllParams[Index] = Param;
  ArgumentParams.insert(Index);

  return Index;
}

unsigned PTXParamManager::addReturnParam(unsigned Size) {
  PTXParam Param;
  Param.Type = PTX_PARAM_TYPE_RETURN;
  Param.Size = Size;

  std::string Name;
  Name = "__ret_";
  Name += utostr(ReturnParams.size()+1);
  Param.Name = Name;

  unsigned Index = AllParams.size();
  AllParams[Index] = Param;
  ReturnParams.insert(Index);

  return Index;
}

unsigned PTXParamManager::addLocalParam(unsigned Size) {
  PTXParam Param;
  Param.Type = PTX_PARAM_TYPE_LOCAL;
  Param.Size = Size;

  std::string Name;
  Name = "__localparam_";
  Name += utostr(LocalParams.size()+1);
  Param.Name = Name;

  unsigned Index = AllParams.size();
  AllParams[Index] = Param;
  LocalParams.insert(Index);

  return Index;
}

