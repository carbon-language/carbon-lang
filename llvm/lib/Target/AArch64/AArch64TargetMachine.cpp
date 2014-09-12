//===-- AArch64TargetMachine.cpp - Define TargetMachine for AArch64 -------===//
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

#include "AArch64.h"
#include "AArch64TargetMachine.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/RegAllocRegistry.h"
#include "llvm/PassManager.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Transforms/Scalar.h"
using namespace llvm;

static cl::opt<bool>
EnableCCMP("aarch64-ccmp", cl::desc("Enable the CCMP formation pass"),
           cl::init(true), cl::Hidden);

static cl::opt<bool> EnableMCR("aarch64-mcr",
                               cl::desc("Enable the machine combiner pass"),
                               cl::init(true), cl::Hidden);

static cl::opt<bool>
EnableStPairSuppress("aarch64-stp-suppress", cl::desc("Suppress STP for AArch64"),
                     cl::init(true), cl::Hidden);

static cl::opt<bool>
EnableAdvSIMDScalar("aarch64-simd-scalar", cl::desc("Enable use of AdvSIMD scalar"
                    " integer instructions"), cl::init(false), cl::Hidden);

static cl::opt<bool>
EnablePromoteConstant("aarch64-promote-const", cl::desc("Enable the promote "
                      "constant pass"), cl::init(true), cl::Hidden);

static cl::opt<bool>
EnableCollectLOH("aarch64-collect-loh", cl::desc("Enable the pass that emits the"
                 " linker optimization hints (LOH)"), cl::init(true),
                 cl::Hidden);

static cl::opt<bool>
EnableDeadRegisterElimination("aarch64-dead-def-elimination", cl::Hidden,
                              cl::desc("Enable the pass that removes dead"
                                       " definitons and replaces stores to"
                                       " them with stores to the zero"
                                       " register"),
                              cl::init(true));

static cl::opt<bool>
EnableLoadStoreOpt("aarch64-load-store-opt", cl::desc("Enable the load/store pair"
                   " optimization pass"), cl::init(true), cl::Hidden);

static cl::opt<bool>
EnableAtomicTidy("aarch64-atomic-cfg-tidy", cl::Hidden,
                 cl::desc("Run SimplifyCFG after expanding atomic operations"
                          " to make use of cmpxchg flow-based information"),
                 cl::init(true));

static cl::opt<bool>
EnableEarlyIfConversion("aarch64-enable-early-ifcvt", cl::Hidden,
                        cl::desc("Run early if-conversion"),
                        cl::init(true));

static cl::opt<bool>
EnableCondOpt("aarch64-condopt",
              cl::desc("Enable the condition optimizer pass"),
              cl::init(true), cl::Hidden);

static cl::opt<bool>
EnablePBQP("aarch64-pbqp", cl::Hidden,
           cl::desc("Use PBQP register allocator (experimental)"),
           cl::init(false));

extern "C" void LLVMInitializeAArch64Target() {
  // Register the target.
  RegisterTargetMachine<AArch64leTargetMachine> X(TheAArch64leTarget);
  RegisterTargetMachine<AArch64beTargetMachine> Y(TheAArch64beTarget);
  RegisterTargetMachine<AArch64leTargetMachine> Z(TheARM64Target);
}

/// TargetMachine ctor - Create an AArch64 architecture model.
///
AArch64TargetMachine::AArch64TargetMachine(const Target &T, StringRef TT,
                                           StringRef CPU, StringRef FS,
                                           const TargetOptions &Options,
                                           Reloc::Model RM, CodeModel::Model CM,
                                           CodeGenOpt::Level OL,
                                           bool LittleEndian)
    : LLVMTargetMachine(T, TT, CPU, FS, Options, RM, CM, OL),
      Subtarget(TT, CPU, FS, *this, LittleEndian),
      usingPBQP(false) {
  initAsmInfo();

  if (EnablePBQP && Subtarget.isCortexA57() && OL != CodeGenOpt::None) {
    usingPBQP = true;
    RegisterRegAlloc::setDefault(createAArch64A57PBQPRegAlloc);
  }
}

void AArch64leTargetMachine::anchor() { }

AArch64leTargetMachine::
AArch64leTargetMachine(const Target &T, StringRef TT,
                       StringRef CPU, StringRef FS, const TargetOptions &Options,
                       Reloc::Model RM, CodeModel::Model CM,
                       CodeGenOpt::Level OL)
  : AArch64TargetMachine(T, TT, CPU, FS, Options, RM, CM, OL, true) {}

void AArch64beTargetMachine::anchor() { }

AArch64beTargetMachine::
AArch64beTargetMachine(const Target &T, StringRef TT,
                       StringRef CPU, StringRef FS, const TargetOptions &Options,
                       Reloc::Model RM, CodeModel::Model CM,
                       CodeGenOpt::Level OL)
  : AArch64TargetMachine(T, TT, CPU, FS, Options, RM, CM, OL, false) {}

namespace {
/// AArch64 Code Generator Pass Configuration Options.
class AArch64PassConfig : public TargetPassConfig {
public:
  AArch64PassConfig(AArch64TargetMachine *TM, PassManagerBase &PM)
      : TargetPassConfig(TM, PM) {
    if (TM->getOptLevel() != CodeGenOpt::None)
      substitutePass(&PostRASchedulerID, &PostMachineSchedulerID);
  }

  AArch64TargetMachine &getAArch64TargetMachine() const {
    return getTM<AArch64TargetMachine>();
  }

  void addIRPasses()  override;
  bool addPreISel() override;
  bool addInstSelector() override;
  bool addILPOpts() override;
  bool addPreRegAlloc() override;
  bool addPostRegAlloc() override;
  bool addPreSched2() override;
  bool addPreEmitPass() override;
};
} // namespace

void AArch64TargetMachine::addAnalysisPasses(PassManagerBase &PM) {
  // Add first the target-independent BasicTTI pass, then our AArch64 pass. This
  // allows the AArch64 pass to delegate to the target independent layer when
  // appropriate.
  PM.add(createBasicTargetTransformInfoPass(this));
  PM.add(createAArch64TargetTransformInfoPass(this));
}

TargetPassConfig *AArch64TargetMachine::createPassConfig(PassManagerBase &PM) {
  return new AArch64PassConfig(this, PM);
}

void AArch64PassConfig::addIRPasses() {
  // Always expand atomic operations, we don't deal with atomicrmw or cmpxchg
  // ourselves.
  addPass(createAtomicExpandPass(TM));

  // Cmpxchg instructions are often used with a subsequent comparison to
  // determine whether it succeeded. We can exploit existing control-flow in
  // ldrex/strex loops to simplify this, but it needs tidying up.
  if (TM->getOptLevel() != CodeGenOpt::None && EnableAtomicTidy)
    addPass(createCFGSimplificationPass());

  TargetPassConfig::addIRPasses();
}

// Pass Pipeline Configuration
bool AArch64PassConfig::addPreISel() {
  // Run promote constant before global merge, so that the promoted constants
  // get a chance to be merged
  if (TM->getOptLevel() != CodeGenOpt::None && EnablePromoteConstant)
    addPass(createAArch64PromoteConstantPass());
  if (TM->getOptLevel() != CodeGenOpt::None)
    addPass(createGlobalMergePass(TM));
  if (TM->getOptLevel() != CodeGenOpt::None)
    addPass(createAArch64AddressTypePromotionPass());

  return false;
}

bool AArch64PassConfig::addInstSelector() {
  addPass(createAArch64ISelDag(getAArch64TargetMachine(), getOptLevel()));

  // For ELF, cleanup any local-dynamic TLS accesses (i.e. combine as many
  // references to _TLS_MODULE_BASE_ as possible.
  if (TM->getSubtarget<AArch64Subtarget>().isTargetELF() &&
      getOptLevel() != CodeGenOpt::None)
    addPass(createAArch64CleanupLocalDynamicTLSPass());

  return false;
}

bool AArch64PassConfig::addILPOpts() {
  if (EnableCondOpt)
    addPass(createAArch64ConditionOptimizerPass());
  if (EnableCCMP)
    addPass(createAArch64ConditionalCompares());
  if (EnableMCR)
    addPass(&MachineCombinerID);
  if (EnableEarlyIfConversion)
    addPass(&EarlyIfConverterID);
  if (EnableStPairSuppress)
    addPass(createAArch64StorePairSuppressPass());
  return true;
}

bool AArch64PassConfig::addPreRegAlloc() {
  // Use AdvSIMD scalar instructions whenever profitable.
  if (TM->getOptLevel() != CodeGenOpt::None && EnableAdvSIMDScalar) {
    addPass(createAArch64AdvSIMDScalar());
    // The AdvSIMD pass may produce copies that can be rewritten to
    // be register coaleascer friendly.
    addPass(&PeepholeOptimizerID);
  }
  return true;
}

bool AArch64PassConfig::addPostRegAlloc() {
  // Change dead register definitions to refer to the zero register.
  if (TM->getOptLevel() != CodeGenOpt::None && EnableDeadRegisterElimination)
    addPass(createAArch64DeadRegisterDefinitions());
  if (TM->getOptLevel() != CodeGenOpt::None &&
      TM->getSubtarget<AArch64Subtarget>().isCortexA57() &&
      !static_cast<const AArch64TargetMachine *>(TM)->isPBQPUsed())
    // Improve performance for some FP/SIMD code for A57.
    addPass(createAArch64A57FPLoadBalancing());
  return true;
}

bool AArch64PassConfig::addPreSched2() {
  // Expand some pseudo instructions to allow proper scheduling.
  addPass(createAArch64ExpandPseudoPass());
  // Use load/store pair instructions when possible.
  if (TM->getOptLevel() != CodeGenOpt::None && EnableLoadStoreOpt)
    addPass(createAArch64LoadStoreOptimizationPass());
  return true;
}

bool AArch64PassConfig::addPreEmitPass() {
  // Relax conditional branch instructions if they're otherwise out of
  // range of their destination.
  addPass(createAArch64BranchRelaxation());
  if (TM->getOptLevel() != CodeGenOpt::None && EnableCollectLOH &&
      TM->getSubtarget<AArch64Subtarget>().isTargetMachO())
    addPass(createAArch64CollectLOHPass());
  return true;
}
