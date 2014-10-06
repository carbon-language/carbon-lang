//===-- PPCTargetMachine.cpp - Define TargetMachine for PowerPC -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Top-level implementation for the PowerPC target.
//
//===----------------------------------------------------------------------===//

#include "PPCTargetMachine.h"
#include "PPC.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/IR/Function.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/PassManager.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Target/TargetOptions.h"
using namespace llvm;

static cl::
opt<bool> DisableCTRLoops("disable-ppc-ctrloops", cl::Hidden,
                        cl::desc("Disable CTR loops for PPC"));

static cl::opt<bool>
VSXFMAMutateEarly("schedule-ppc-vsx-fma-mutation-early",
  cl::Hidden, cl::desc("Schedule VSX FMA instruction mutation early"));

extern "C" void LLVMInitializePowerPCTarget() {
  // Register the targets
  RegisterTargetMachine<PPC32TargetMachine> A(ThePPC32Target);
  RegisterTargetMachine<PPC64TargetMachine> B(ThePPC64Target);
  RegisterTargetMachine<PPC64TargetMachine> C(ThePPC64LETarget);
}

static std::string computeFSAdditions(StringRef FS, CodeGenOpt::Level OL, StringRef TT) {
  std::string FullFS = FS;
  Triple TargetTriple(TT);

  // Make sure 64-bit features are available when CPUname is generic
  if (TargetTriple.getArch() == Triple::ppc64 ||
      TargetTriple.getArch() == Triple::ppc64le) {
    if (!FullFS.empty())
      FullFS = "+64bit," + FullFS;
    else
      FullFS = "+64bit";
  }

  if (OL >= CodeGenOpt::Default) {
    if (!FullFS.empty())
      FullFS = "+crbits," + FullFS;
    else
      FullFS = "+crbits";
  }
  return FullFS;
}

// The FeatureString here is a little subtle. We are modifying the feature string
// with what are (currently) non-function specific overrides as it goes into the
// LLVMTargetMachine constructor and then using the stored value in the
// Subtarget constructor below it.
PPCTargetMachine::PPCTargetMachine(const Target &T, StringRef TT, StringRef CPU,
                                   StringRef FS, const TargetOptions &Options,
                                   Reloc::Model RM, CodeModel::Model CM,
                                   CodeGenOpt::Level OL)
    : LLVMTargetMachine(T, TT, CPU, computeFSAdditions(FS, OL, TT), Options, RM,
                        CM, OL),
      Subtarget(TT, CPU, TargetFS, *this) {
  initAsmInfo();
}

void PPC32TargetMachine::anchor() { }

PPC32TargetMachine::PPC32TargetMachine(const Target &T, StringRef TT,
                                       StringRef CPU, StringRef FS,
                                       const TargetOptions &Options,
                                       Reloc::Model RM, CodeModel::Model CM,
                                       CodeGenOpt::Level OL)
  : PPCTargetMachine(T, TT, CPU, FS, Options, RM, CM, OL) {
}

void PPC64TargetMachine::anchor() { }

PPC64TargetMachine::PPC64TargetMachine(const Target &T, StringRef TT,
                                       StringRef CPU,  StringRef FS,
                                       const TargetOptions &Options,
                                       Reloc::Model RM, CodeModel::Model CM,
                                       CodeGenOpt::Level OL)
  : PPCTargetMachine(T, TT, CPU, FS, Options, RM, CM, OL) {
}

const PPCSubtarget *
PPCTargetMachine::getSubtargetImpl(const Function &F) const {
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

  auto &I = SubtargetMap[CPU + FS];
  if (!I) {
    // This needs to be done before we create a new subtarget since any
    // creation will depend on the TM and the code generation flags on the
    // function that reside in TargetOptions.
    resetTargetOptions(F);
    I = llvm::make_unique<PPCSubtarget>(TargetTriple, CPU, FS, *this);
  }
  return I.get();
}

//===----------------------------------------------------------------------===//
// Pass Pipeline Configuration
//===----------------------------------------------------------------------===//

namespace {
/// PPC Code Generator Pass Configuration Options.
class PPCPassConfig : public TargetPassConfig {
public:
  PPCPassConfig(PPCTargetMachine *TM, PassManagerBase &PM)
    : TargetPassConfig(TM, PM) {}

  PPCTargetMachine &getPPCTargetMachine() const {
    return getTM<PPCTargetMachine>();
  }

  const PPCSubtarget &getPPCSubtarget() const {
    return *getPPCTargetMachine().getSubtargetImpl();
  }

  void addIRPasses() override;
  bool addPreISel() override;
  bool addILPOpts() override;
  bool addInstSelector() override;
  bool addPreRegAlloc() override;
  bool addPreSched2() override;
  bool addPreEmitPass() override;
};
} // namespace

TargetPassConfig *PPCTargetMachine::createPassConfig(PassManagerBase &PM) {
  return new PPCPassConfig(this, PM);
}

void PPCPassConfig::addIRPasses() {
  addPass(createAtomicExpandPass(&getPPCTargetMachine()));
  TargetPassConfig::addIRPasses();
}

bool PPCPassConfig::addPreISel() {
  if (!DisableCTRLoops && getOptLevel() != CodeGenOpt::None)
    addPass(createPPCCTRLoops(getPPCTargetMachine()));

  return false;
}

bool PPCPassConfig::addILPOpts() {
  addPass(&EarlyIfConverterID);
  return true;
}

bool PPCPassConfig::addInstSelector() {
  // Install an instruction selector.
  addPass(createPPCISelDag(getPPCTargetMachine()));

#ifndef NDEBUG
  if (!DisableCTRLoops && getOptLevel() != CodeGenOpt::None)
    addPass(createPPCCTRLoopsVerify());
#endif

  addPass(createPPCVSXCopyPass());
  return false;
}

bool PPCPassConfig::addPreRegAlloc() {
  initializePPCVSXFMAMutatePass(*PassRegistry::getPassRegistry());
  insertPass(VSXFMAMutateEarly ? &RegisterCoalescerID : &MachineSchedulerID,
             &PPCVSXFMAMutateID);
  return false;
}

bool PPCPassConfig::addPreSched2() {
  addPass(createPPCVSXCopyCleanupPass());

  if (getOptLevel() != CodeGenOpt::None)
    addPass(&IfConverterID);

  return true;
}

bool PPCPassConfig::addPreEmitPass() {
  if (getOptLevel() != CodeGenOpt::None)
    addPass(createPPCEarlyReturnPass());
  // Must run branch selection immediately preceding the asm printer.
  addPass(createPPCBranchSelectionPass());
  return false;
}

void PPCTargetMachine::addAnalysisPasses(PassManagerBase &PM) {
  // Add first the target-independent BasicTTI pass, then our PPC pass. This
  // allows the PPC pass to delegate to the target independent layer when
  // appropriate.
  PM.add(createBasicTargetTransformInfoPass(this));
  PM.add(createPPCTargetTransformInfoPass(this));
}

