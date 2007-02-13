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

#include "ARMTargetMachine.h"
#include "ARMTargetAsmInfo.h"
#include "ARMFrameInfo.h"
#include "ARM.h"
#include "llvm/Module.h"
#include "llvm/PassManager.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Target/TargetMachineRegistry.h"
#include "llvm/Target/TargetOptions.h"
using namespace llvm;

static cl::opt<bool> DisableLdStOpti("disable-arm-loadstore-opti", cl::Hidden,
                              cl::desc("Disable load store optimization pass"));

namespace {
  // Register the target.
  RegisterTarget<ARMTargetMachine> X("arm", "  ARM");
}

/// TargetMachine ctor - Create an ILP32 architecture model
///
ARMTargetMachine::ARMTargetMachine(const Module &M, const std::string &FS)
  : Subtarget(M, FS),
    DataLayout(Subtarget.isAPCS_ABI() ?
               //APCS ABI
          (Subtarget.isThumb() ?
           std::string("e-p:32:32-d:32:32-l:32:32-s:16:32-b:8:32-B:8:32-A:32") :
           std::string("e-p:32:32-d:32:32-l:32:32")) :
               //AAPCS ABI
          (Subtarget.isThumb() ?
           std::string("e-p:32:32-d:64:64-l:64:64-s:16:32-b:8:32-B:8:32-A:32") :
           std::string("e-p:32:32-d:64:64-l:64:64"))),
    InstrInfo(Subtarget),
    FrameInfo(Subtarget) {}

unsigned ARMTargetMachine::getModuleMatchQuality(const Module &M) {
  std::string TT = M.getTargetTriple();
  if (TT.size() >= 4 && std::string(TT.begin(), TT.begin()+4) == "arm-")
    return 20;

  if (M.getPointerSize() == Module::Pointer32)
    return 1;
  else
    return 0;
}


const TargetAsmInfo *ARMTargetMachine::createTargetAsmInfo() const {
  return new ARMTargetAsmInfo(*this);
}


// Pass Pipeline Configuration
bool ARMTargetMachine::addInstSelector(FunctionPassManager &PM, bool Fast) {
  PM.add(createARMISelDag(*this));
  return false;
}

bool ARMTargetMachine::addPreEmitPass(FunctionPassManager &PM, bool Fast) {
  // FIXME: temporarily disabling load / store optimization pass for Thumb mode.
  if (!Fast && !DisableLdStOpti && !Subtarget.isThumb())
    PM.add(createARMLoadStoreOptimizationPass());
  
  PM.add(createARMConstantIslandPass());
  return true;
}

bool ARMTargetMachine::addAssemblyEmitter(FunctionPassManager &PM, bool Fast, 
                                          std::ostream &Out) {
  // Output assembly language.
  PM.add(createARMCodePrinterPass(Out, *this));
  return false;
}
