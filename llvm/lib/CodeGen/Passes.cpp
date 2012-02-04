//===-- Passes.cpp - Target independent code generation passes ------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines interfaces to access the target independent code
// generation passes provided by the LLVM backend.
//
//===---------------------------------------------------------------------===//

#include "llvm/CodeGen/RegAllocRegistry.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/Support/ErrorHandling.h"

using namespace llvm;

//===---------------------------------------------------------------------===//
/// TargetPassConfig
//===---------------------------------------------------------------------===//

INITIALIZE_PASS(TargetPassConfig, "targetpassconfig",
                "Target Pass Configuration", false, false)
char TargetPassConfig::ID = 0;

// Out of line virtual method.
TargetPassConfig::~TargetPassConfig() {}

TargetPassConfig::TargetPassConfig(TargetMachine *tm, PassManagerBase &pm,
                                   bool DisableVerifyFlag)
  : ImmutablePass(ID), TM(tm), PM(pm), DisableVerify(DisableVerifyFlag) {
  // Register all target independent codegen passes to activate their PassIDs,
  // including this pass itself.
  initializeCodeGen(*PassRegistry::getPassRegistry());
}

/// createPassConfig - Create a pass configuration object to be used by
/// addPassToEmitX methods for generating a pipeline of CodeGen passes.
///
/// Targets may override this to extend TargetPassConfig.
TargetPassConfig *LLVMTargetMachine::createPassConfig(PassManagerBase &PM,
                                                      bool DisableVerify) {
  return new TargetPassConfig(this, PM, DisableVerify);
}

TargetPassConfig::TargetPassConfig()
  : ImmutablePass(ID), PM(*(PassManagerBase*)0) {
  llvm_unreachable("TargetPassConfig should not be constructed on-the-fly");
}

//===---------------------------------------------------------------------===//
///
/// RegisterRegAlloc class - Track the registration of register allocators.
///
//===---------------------------------------------------------------------===//
MachinePassRegistry RegisterRegAlloc::Registry;

static FunctionPass *createDefaultRegisterAllocator() { return 0; }
static RegisterRegAlloc
defaultRegAlloc("default",
                "pick register allocator based on -O option",
                createDefaultRegisterAllocator);

//===---------------------------------------------------------------------===//
///
/// RegAlloc command line options.
///
//===---------------------------------------------------------------------===//
static cl::opt<RegisterRegAlloc::FunctionPassCtor, false,
               RegisterPassParser<RegisterRegAlloc> >
RegAlloc("regalloc",
         cl::init(&createDefaultRegisterAllocator),
         cl::desc("Register allocator to use"));


//===---------------------------------------------------------------------===//
///
/// createRegisterAllocator - choose the appropriate register allocator.
///
//===---------------------------------------------------------------------===//
FunctionPass *llvm::createRegisterAllocator(CodeGenOpt::Level OptLevel) {
  RegisterRegAlloc::FunctionPassCtor Ctor = RegisterRegAlloc::getDefault();

  if (!Ctor) {
    Ctor = RegAlloc;
    RegisterRegAlloc::setDefault(RegAlloc);
  }

  if (Ctor != createDefaultRegisterAllocator)
    return Ctor();

  // When the 'default' allocator is requested, pick one based on OptLevel.
  switch (OptLevel) {
  case CodeGenOpt::None:
    return createFastRegisterAllocator();
  default:
    return createGreedyRegisterAllocator();
  }
}
