//===-- ARMTargetMachine.cpp - Define TargetMachine for ARM ---------------===//
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

#include "ARM.h"
#include "ARMFrameLowering.h"
#include "ARMTargetMachine.h"
#include "ARMTargetObjectFile.h"
#include "ARMTargetTransformInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/IR/Function.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/PassManager.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Transforms/Scalar.h"
using namespace llvm;

static cl::opt<bool>
DisableA15SDOptimization("disable-a15-sd-optimization", cl::Hidden,
                   cl::desc("Inhibit optimization of S->D register accesses on A15"),
                   cl::init(false));

static cl::opt<bool>
EnableAtomicTidy("arm-atomic-cfg-tidy", cl::Hidden,
                 cl::desc("Run SimplifyCFG after expanding atomic operations"
                          " to make use of cmpxchg flow-based information"),
                 cl::init(true));

extern "C" void LLVMInitializeARMTarget() {
  // Register the target.
  RegisterTargetMachine<ARMLETargetMachine> X(TheARMLETarget);
  RegisterTargetMachine<ARMBETargetMachine> Y(TheARMBETarget);
  RegisterTargetMachine<ThumbLETargetMachine> A(TheThumbLETarget);
  RegisterTargetMachine<ThumbBETargetMachine> B(TheThumbBETarget);
}

static std::unique_ptr<TargetLoweringObjectFile> createTLOF(const Triple &TT) {
  if (TT.isOSBinFormatMachO())
    return make_unique<TargetLoweringObjectFileMachO>();
  if (TT.isOSWindows())
    return make_unique<TargetLoweringObjectFileCOFF>();
  return make_unique<ARMElfTargetObjectFile>();
}

static ARMBaseTargetMachine::ARMABI
computeTargetABI(const Triple &TT, StringRef CPU,
                 const TargetOptions &Options) {
  if (Options.MCOptions.getABIName().startswith("aapcs"))
    return ARMBaseTargetMachine::ARM_ABI_AAPCS;
  else if (Options.MCOptions.getABIName().startswith("apcs"))
    return ARMBaseTargetMachine::ARM_ABI_APCS;

  assert(Options.MCOptions.getABIName().empty() &&
         "Unknown target-abi option!");

  ARMBaseTargetMachine::ARMABI TargetABI =
      ARMBaseTargetMachine::ARM_ABI_UNKNOWN;

  // FIXME: This is duplicated code from the front end and should be unified.
  if (TT.isOSBinFormatMachO()) {
    if (TT.getEnvironment() == llvm::Triple::EABI ||
        (TT.getOS() == llvm::Triple::UnknownOS &&
         TT.getObjectFormat() == llvm::Triple::MachO) ||
        CPU.startswith("cortex-m")) {
      TargetABI = ARMBaseTargetMachine::ARM_ABI_AAPCS;
    } else {
      TargetABI = ARMBaseTargetMachine::ARM_ABI_APCS;
    }
  } else if (TT.isOSWindows()) {
    // FIXME: this is invalid for WindowsCE
    TargetABI = ARMBaseTargetMachine::ARM_ABI_AAPCS;
  } else {
    // Select the default based on the platform.
    switch (TT.getEnvironment()) {
    case llvm::Triple::Android:
    case llvm::Triple::GNUEABI:
    case llvm::Triple::GNUEABIHF:
    case llvm::Triple::EABIHF:
    case llvm::Triple::EABI:
      TargetABI = ARMBaseTargetMachine::ARM_ABI_AAPCS;
      break;
    case llvm::Triple::GNU:
      TargetABI = ARMBaseTargetMachine::ARM_ABI_APCS;
      break;
    default:
      if (TT.getOS() == llvm::Triple::NetBSD)
	TargetABI = ARMBaseTargetMachine::ARM_ABI_APCS;
      else
	TargetABI = ARMBaseTargetMachine::ARM_ABI_AAPCS;
      break;
    }
  }

  return TargetABI;
}

static std::string computeDataLayout(const Triple &TT,
                                     ARMBaseTargetMachine::ARMABI ABI,
                                     bool isLittle) {
  std::string Ret = "";

  if (isLittle)
    // Little endian.
    Ret += "e";
  else
    // Big endian.
    Ret += "E";

  Ret += DataLayout::getManglingComponent(TT);

  // Pointers are 32 bits and aligned to 32 bits.
  Ret += "-p:32:32";

  // ABIs other than APCS have 64 bit integers with natural alignment.
  if (ABI != ARMBaseTargetMachine::ARM_ABI_APCS)
    Ret += "-i64:64";

  // We have 64 bits floats. The APCS ABI requires them to be aligned to 32
  // bits, others to 64 bits. We always try to align to 64 bits.
  if (ABI == ARMBaseTargetMachine::ARM_ABI_APCS)
    Ret += "-f64:32:64";

  // We have 128 and 64 bit vectors. The APCS ABI aligns them to 32 bits, others
  // to 64. We always ty to give them natural alignment.
  if (ABI == ARMBaseTargetMachine::ARM_ABI_APCS)
    Ret += "-v64:32:64-v128:32:128";
  else
    Ret += "-v128:64:128";

  // Try to align aggregates to 32 bits (the default is 64 bits, which has no
  // particular hardware support on 32-bit ARM).
  Ret += "-a:0:32";

  // Integer registers are 32 bits.
  Ret += "-n32";

  // The stack is 128 bit aligned on NaCl, 64 bit aligned on AAPCS and 32 bit
  // aligned everywhere else.
  if (TT.isOSNaCl())
    Ret += "-S128";
  else if (ABI == ARMBaseTargetMachine::ARM_ABI_AAPCS)
    Ret += "-S64";
  else
    Ret += "-S32";

  return Ret;
}

/// TargetMachine ctor - Create an ARM architecture model.
///
ARMBaseTargetMachine::ARMBaseTargetMachine(const Target &T, StringRef TT,
                                           StringRef CPU, StringRef FS,
                                           const TargetOptions &Options,
                                           Reloc::Model RM, CodeModel::Model CM,
                                           CodeGenOpt::Level OL, bool isLittle)
    : LLVMTargetMachine(T, TT, CPU, FS, Options, RM, CM, OL),
      TargetABI(computeTargetABI(Triple(TT), CPU, Options)),
      DL(computeDataLayout(Triple(TT), TargetABI, isLittle)),
      TLOF(createTLOF(Triple(getTargetTriple()))),
      Subtarget(TT, CPU, FS, *this, isLittle), isLittle(isLittle) {

  // Default to triple-appropriate float ABI
  if (Options.FloatABIType == FloatABI::Default)
    this->Options.FloatABIType =
        Subtarget.isTargetHardFloat() ? FloatABI::Hard : FloatABI::Soft;
}

ARMBaseTargetMachine::~ARMBaseTargetMachine() {}

const ARMSubtarget *
ARMBaseTargetMachine::getSubtargetImpl(const Function &F) const {
  AttributeSet FnAttrs = F.getAttributes();
  Attribute CPUAttr =
      FnAttrs.getAttribute(AttributeSet::FunctionIndex, "target-cpu");
  Attribute FSAttr =
      FnAttrs.getAttribute(AttributeSet::FunctionIndex, "target-features");

  std::string CPU = !CPUAttr.hasAttribute(Attribute::None)
                        ? CPUAttr.getValueAsString().str()
                        : TargetCPU;
  std::string FS = !FSAttr.hasAttribute(Attribute::None)
                       ? FSAttr.getValueAsString().str()
                       : TargetFS;

  // FIXME: This is related to the code below to reset the target options,
  // we need to know whether or not the soft float flag is set on the
  // function before we can generate a subtarget. We also need to use
  // it as a key for the subtarget since that can be the only difference
  // between two functions.
  Attribute SFAttr =
      FnAttrs.getAttribute(AttributeSet::FunctionIndex, "use-soft-float");
  bool SoftFloat = !SFAttr.hasAttribute(Attribute::None)
                       ? SFAttr.getValueAsString() == "true"
                       : Options.UseSoftFloat;

  auto &I = SubtargetMap[CPU + FS + (SoftFloat ? "use-soft-float=true"
                                               : "use-soft-float=false")];
  if (!I) {
    // This needs to be done before we create a new subtarget since any
    // creation will depend on the TM and the code generation flags on the
    // function that reside in TargetOptions.
    resetTargetOptions(F);
    I = llvm::make_unique<ARMSubtarget>(TargetTriple, CPU, FS, *this, isLittle);
  }
  return I.get();
}

TargetIRAnalysis ARMBaseTargetMachine::getTargetIRAnalysis() {
  return TargetIRAnalysis(
      [this](Function &F) { return TargetTransformInfo(ARMTTIImpl(this, F)); });
}


void ARMTargetMachine::anchor() { }

ARMTargetMachine::ARMTargetMachine(const Target &T, StringRef TT, StringRef CPU,
                                   StringRef FS, const TargetOptions &Options,
                                   Reloc::Model RM, CodeModel::Model CM,
                                   CodeGenOpt::Level OL, bool isLittle)
    : ARMBaseTargetMachine(T, TT, CPU, FS, Options, RM, CM, OL, isLittle) {
  initAsmInfo();
  if (!Subtarget.hasARMOps())
    report_fatal_error("CPU: '" + Subtarget.getCPUString() + "' does not "
                       "support ARM mode execution!");
}

void ARMLETargetMachine::anchor() { }

ARMLETargetMachine::ARMLETargetMachine(const Target &T, StringRef TT,
                                       StringRef CPU, StringRef FS,
                                       const TargetOptions &Options,
                                       Reloc::Model RM, CodeModel::Model CM,
                                       CodeGenOpt::Level OL)
    : ARMTargetMachine(T, TT, CPU, FS, Options, RM, CM, OL, true) {}

void ARMBETargetMachine::anchor() { }

ARMBETargetMachine::ARMBETargetMachine(const Target &T, StringRef TT,
                                       StringRef CPU, StringRef FS,
                                       const TargetOptions &Options,
                                       Reloc::Model RM, CodeModel::Model CM,
                                       CodeGenOpt::Level OL)
    : ARMTargetMachine(T, TT, CPU, FS, Options, RM, CM, OL, false) {}

void ThumbTargetMachine::anchor() { }

ThumbTargetMachine::ThumbTargetMachine(const Target &T, StringRef TT,
                                       StringRef CPU, StringRef FS,
                                       const TargetOptions &Options,
                                       Reloc::Model RM, CodeModel::Model CM,
                                       CodeGenOpt::Level OL, bool isLittle)
    : ARMBaseTargetMachine(T, TT, CPU, FS, Options, RM, CM, OL,
                           isLittle) {
  initAsmInfo();
}

void ThumbLETargetMachine::anchor() { }

ThumbLETargetMachine::ThumbLETargetMachine(const Target &T, StringRef TT,
                                           StringRef CPU, StringRef FS,
                                           const TargetOptions &Options,
                                           Reloc::Model RM, CodeModel::Model CM,
                                           CodeGenOpt::Level OL)
    : ThumbTargetMachine(T, TT, CPU, FS, Options, RM, CM, OL, true) {}

void ThumbBETargetMachine::anchor() { }

ThumbBETargetMachine::ThumbBETargetMachine(const Target &T, StringRef TT,
                                           StringRef CPU, StringRef FS,
                                           const TargetOptions &Options,
                                           Reloc::Model RM, CodeModel::Model CM,
                                           CodeGenOpt::Level OL)
    : ThumbTargetMachine(T, TT, CPU, FS, Options, RM, CM, OL, false) {}

namespace {
/// ARM Code Generator Pass Configuration Options.
class ARMPassConfig : public TargetPassConfig {
public:
  ARMPassConfig(ARMBaseTargetMachine *TM, PassManagerBase &PM)
    : TargetPassConfig(TM, PM) {}

  ARMBaseTargetMachine &getARMTargetMachine() const {
    return getTM<ARMBaseTargetMachine>();
  }

  const ARMSubtarget &getARMSubtarget() const {
    return *getARMTargetMachine().getSubtargetImpl();
  }

  void addIRPasses() override;
  bool addPreISel() override;
  bool addInstSelector() override;
  void addPreRegAlloc() override;
  void addPreSched2() override;
  void addPreEmitPass() override;
};
} // namespace

TargetPassConfig *ARMBaseTargetMachine::createPassConfig(PassManagerBase &PM) {
  return new ARMPassConfig(this, PM);
}

void ARMPassConfig::addIRPasses() {
  if (TM->Options.ThreadModel == ThreadModel::Single)
    addPass(createLowerAtomicPass());
  else
    addPass(createAtomicExpandPass(TM));

  // Cmpxchg instructions are often used with a subsequent comparison to
  // determine whether it succeeded. We can exploit existing control-flow in
  // ldrex/strex loops to simplify this, but it needs tidying up.
  const ARMSubtarget *Subtarget = &getARMSubtarget();
  if (Subtarget->hasAnyDataBarrier() && !Subtarget->isThumb1Only())
    if (TM->getOptLevel() != CodeGenOpt::None && EnableAtomicTidy)
      addPass(createCFGSimplificationPass());

  TargetPassConfig::addIRPasses();
}

bool ARMPassConfig::addPreISel() {
  if (TM->getOptLevel() != CodeGenOpt::None)
    addPass(createGlobalMergePass(TM));

  return false;
}

bool ARMPassConfig::addInstSelector() {
  addPass(createARMISelDag(getARMTargetMachine(), getOptLevel()));

  const ARMSubtarget *Subtarget = &getARMSubtarget();
  if (Subtarget->isTargetELF() && !Subtarget->isThumb1Only() &&
      TM->Options.EnableFastISel)
    addPass(createARMGlobalBaseRegPass());
  return false;
}

void ARMPassConfig::addPreRegAlloc() {
  if (getOptLevel() != CodeGenOpt::None)
    addPass(createARMLoadStoreOptimizationPass(true));
  if (getOptLevel() != CodeGenOpt::None && getARMSubtarget().isCortexA9())
    addPass(createMLxExpansionPass());
  // Since the A15SDOptimizer pass can insert VDUP instructions, it can only be
  // enabled when NEON is available.
  if (getOptLevel() != CodeGenOpt::None && getARMSubtarget().isCortexA15() &&
    getARMSubtarget().hasNEON() && !DisableA15SDOptimization) {
    addPass(createA15SDOptimizerPass());
  }
}

void ARMPassConfig::addPreSched2() {
  if (getOptLevel() != CodeGenOpt::None) {
    addPass(createARMLoadStoreOptimizationPass());

    if (getARMSubtarget().hasNEON())
      addPass(createExecutionDependencyFixPass(&ARM::DPRRegClass));
  }

  // Expand some pseudo instructions into multiple instructions to allow
  // proper scheduling.
  addPass(createARMExpandPseudoPass());

  if (getOptLevel() != CodeGenOpt::None) {
    if (!getARMSubtarget().isThumb1Only()) {
      // in v8, IfConversion depends on Thumb instruction widths
      if (getARMSubtarget().restrictIT() &&
          !getARMSubtarget().prefers32BitThumb())
        addPass(createThumb2SizeReductionPass());
      addPass(&IfConverterID);
    }
  }
  if (getARMSubtarget().isThumb2())
    addPass(createThumb2ITBlockPass());
}

void ARMPassConfig::addPreEmitPass() {
  if (getARMSubtarget().isThumb2()) {
    if (!getARMSubtarget().prefers32BitThumb())
      addPass(createThumb2SizeReductionPass());

    // Constant island pass work on unbundled instructions.
    addPass(&UnpackMachineBundlesID);
  }

  addPass(createARMOptimizeBarriersPass());
  addPass(createARMConstantIslandPass());
}
