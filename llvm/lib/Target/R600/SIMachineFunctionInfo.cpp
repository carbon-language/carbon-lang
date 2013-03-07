//===-- SIMachineFunctionInfo.cpp - SI Machine Function Info -------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
/// \file
//===----------------------------------------------------------------------===//


#include "SIMachineFunctionInfo.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/Function.h"

using namespace llvm;

const char *SIMachineFunctionInfo::ShaderTypeAttribute = "ShaderType";

SIMachineFunctionInfo::SIMachineFunctionInfo(const MachineFunction &MF)
  : MachineFunctionInfo(),
    ShaderType(0),
    PSInputAddr(0) {

  AttributeSet Set = MF.getFunction()->getAttributes();
  Attribute A = Set.getAttribute(AttributeSet::FunctionIndex,
                                 ShaderTypeAttribute);

  if (A.isStringAttribute()) {
    StringRef Str = A.getValueAsString();
    if (Str.getAsInteger(0, ShaderType))
      llvm_unreachable("Can't parse shader type!");
  }
}
