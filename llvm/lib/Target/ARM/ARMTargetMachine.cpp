//===-- ARMTargetMachine.cpp - Define TargetMachine for ARM ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the "Instituto Nokia de Tecnologia" and
// is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//
//===----------------------------------------------------------------------===//

#include "ARMTargetAsmInfo.h"
#include "ARMTargetMachine.h"
#include "ARMFrameInfo.h"
#include "ARM.h"
#include "llvm/Module.h"
#include "llvm/PassManager.h"
#include "llvm/Target/TargetMachineRegistry.h"
using namespace llvm;

namespace {
  // Register the target.
  RegisterTarget<ARMTargetMachine> X("arm", "  ARM");
}


const TargetAsmInfo *ARMTargetMachine::createTargetAsmInfo() const {
  return new ARMTargetAsmInfo(*this);
}


/// TargetMachine ctor - Create an ILP32 architecture model
///
ARMTargetMachine::ARMTargetMachine(const Module &M, const std::string &FS)
  : DataLayout("E-p:32:32") {
}

unsigned ARMTargetMachine::getModuleMatchQuality(const Module &M) {
  std::string TT = M.getTargetTriple();
  if (TT.size() >= 4 && std::string(TT.begin(), TT.begin()+4) == "arm-")
    return 20;

  if (M.getPointerSize() == Module::Pointer32)
    return 1;
  else
    return 0;
}


// Pass Pipeline Configuration
bool ARMTargetMachine::addInstSelector(FunctionPassManager &PM, bool Fast) {
  PM.add(createARMISelDag(*this));
  return false;
}

bool ARMTargetMachine::addPostRegAlloc(FunctionPassManager &PM, bool Fast) {
  PM.add(createARMFixMulPass());
  return true;
}

bool ARMTargetMachine::addAssemblyEmitter(FunctionPassManager &PM, bool Fast, 
                                          std::ostream &Out) {
  // Output assembly language.
  PM.add(createARMCodePrinterPass(Out, *this));
  return false;
}
