//===-- ARM64TargetMachine.cpp - Define TargetMachine for ARM64 -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//
//===----------------------------------------------------------------------===//

#include "ARM64.h"
#include "ARM64TargetMachine.h"
#include "llvm/PassManager.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Transforms/Scalar.h"
using namespace llvm;

static cl::opt<bool> EnableCCMP("arm64-ccmp",
                                cl::desc("Enable the CCMP formation pass"),
                                cl::init(true));

static cl::opt<bool> EnableStPairSuppress("arm64-stp-suppress", cl::Hidden,
                                          cl::desc("Suppress STP for ARM64"),
                                          cl::init(true));

static cl::opt<bool>
EnablePromoteConstant("arm64-promote-const", cl::Hidden,
                      cl::desc("Enable the promote constant pass"),
                      cl::init(true));

static cl::opt<bool>
EnableCollectLOH("arm64-collect-loh", cl::Hidden,
                 cl::desc("Enable the pass that emits the linker"
                          " optimization hints (LOH)"),
                 cl::init(true));

static cl::opt<bool>
EnableDeadRegisterElimination("arm64-dead-def-elimination", cl::Hidden,
                              cl::desc("Enable the pass that removes dead"
                                       " definitons and replaces stores to"
                                       " them with stores to the zero"
                                       " register"),
                              cl::init(true));

extern "C" void LLVMInitializeARM64Target() {
  // Register the target.
  RegisterTargetMachine<ARM64leTargetMachine> X(TheARM64leTarget);
  RegisterTargetMachine<ARM64beTargetMachine> Y(TheARM64beTarget);
}

/// TargetMachine ctor - Create an ARM64 architecture model.
///
ARM64TargetMachine::ARM64TargetMachine(const Target &T, StringRef TT,
                                       StringRef CPU, StringRef FS,
                                       const TargetOptions &Options,
                                       Reloc::Model RM, CodeModel::Model CM,
                                       CodeGenOpt::Level OL,
                                       bool LittleEndian)
    : LLVMTargetMachine(T, TT, CPU, FS, Options, RM, CM, OL),
      Subtarget(TT, CPU, FS, LittleEndian),
      // This nested ternary is horrible, but DL needs to be properly initialized
      // before TLInfo is constructed.
      DL(Subtarget.isTargetMachO() ?
         "e-m:o-i64:64-i128:128-n32:64-S128" :
         (LittleEndian ?
          "e-m:e-i64:64-i128:128-n32:64-S128" :
          "E-m:e-i64:64-i128:128-n32:64-S128")),
      InstrInfo(Subtarget), TLInfo(*this), FrameLowering(*this, Subtarget),
      TSInfo(*this) {
  initAsmInfo();
}

void ARM64leTargetMachine::anchor() { }

ARM64leTargetMachine::
ARM64leTargetMachine(const Target &T, StringRef TT,
                       StringRef CPU, StringRef FS, const TargetOptions &Options,
                       Reloc::Model RM, CodeModel::Model CM,
                       CodeGenOpt::Level OL)
  : ARM64TargetMachine(T, TT, CPU, FS, Options, RM, CM, OL, true) {}

void ARM64beTargetMachine::anchor() { }

ARM64beTargetMachine::
ARM64beTargetMachine(const Target &T, StringRef TT,
                       StringRef CPU, StringRef FS, const TargetOptions &Options,
                       Reloc::Model RM, CodeModel::Model CM,
                       CodeGenOpt::Level OL)
  : ARM64TargetMachine(T, TT, CPU, FS, Options, RM, CM, OL, false) {}

namespace {
/// ARM64 Code Generator Pass Configuration Options.
class ARM64PassConfig : public TargetPassConfig {
public:
  ARM64PassConfig(ARM64TargetMachine *TM, PassManagerBase &PM)
      : TargetPassConfig(TM, PM) {}

  ARM64TargetMachine &getARM64TargetMachine() const {
    return getTM<ARM64TargetMachine>();
  }

  virtual bool addPreISel();
  virtual bool addInstSelector();
  virtual bool addILPOpts();
  virtual bool addPreRegAlloc();
  virtual bool addPostRegAlloc();
  virtual bool addPreSched2();
  virtual bool addPreEmitPass();
};
} // namespace

void ARM64TargetMachine::addAnalysisPasses(PassManagerBase &PM) {
  // Add first the target-independent BasicTTI pass, then our ARM64 pass. This
  // allows the ARM64 pass to delegate to the target independent layer when
  // appropriate.
  PM.add(createBasicTargetTransformInfoPass(this));
  PM.add(createARM64TargetTransformInfoPass(this));
}

TargetPassConfig *ARM64TargetMachine::createPassConfig(PassManagerBase &PM) {
  return new ARM64PassConfig(this, PM);
}

// Pass Pipeline Configuration
bool ARM64PassConfig::addPreISel() {
  // Run promote constant before global merge, so that the promoted constants
  // get a chance to be merged
  if (TM->getOptLevel() != CodeGenOpt::None && EnablePromoteConstant)
    addPass(createARM64PromoteConstantPass());
  if (TM->getOptLevel() != CodeGenOpt::None)
    addPass(createGlobalMergePass(TM));
  if (TM->getOptLevel() != CodeGenOpt::None)
    addPass(createARM64AddressTypePromotionPass());

  // Always expand atomic operations, we don't deal with atomicrmw or cmpxchg
  // ourselves.
  addPass(createAtomicExpandLoadLinkedPass(TM));

  return false;
}

bool ARM64PassConfig::addInstSelector() {
  addPass(createARM64ISelDag(getARM64TargetMachine(), getOptLevel()));

  // For ELF, cleanup any local-dynamic TLS accesses (i.e. combine as many
  // references to _TLS_MODULE_BASE_ as possible.
  if (TM->getSubtarget<ARM64Subtarget>().isTargetELF() &&
      getOptLevel() != CodeGenOpt::None)
    addPass(createARM64CleanupLocalDynamicTLSPass());

  return false;
}

bool ARM64PassConfig::addILPOpts() {
  if (EnableCCMP)
    addPass(createARM64ConditionalCompares());
  addPass(&EarlyIfConverterID);
  if (EnableStPairSuppress)
    addPass(createARM64StorePairSuppressPass());
  return true;
}

bool ARM64PassConfig::addPreRegAlloc() {
  // Use AdvSIMD scalar instructions whenever profitable.
  addPass(createARM64AdvSIMDScalar());
  return true;
}

bool ARM64PassConfig::addPostRegAlloc() {
  // Change dead register definitions to refer to the zero register.
  if (EnableDeadRegisterElimination)
    addPass(createARM64DeadRegisterDefinitions());
  return true;
}

bool ARM64PassConfig::addPreSched2() {
  // Expand some pseudo instructions to allow proper scheduling.
  addPass(createARM64ExpandPseudoPass());
  // Use load/store pair instructions when possible.
  addPass(createARM64LoadStoreOptimizationPass());
  return true;
}

bool ARM64PassConfig::addPreEmitPass() {
  // Relax conditional branch instructions if they're otherwise out of
  // range of their destination.
  addPass(createARM64BranchRelaxation());
  if (TM->getOptLevel() != CodeGenOpt::None && EnableCollectLOH &&
      TM->getSubtarget<ARM64Subtarget>().isTargetMachO())
    addPass(createARM64CollectLOHPass());
  return true;
}
