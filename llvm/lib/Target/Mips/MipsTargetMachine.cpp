//===-- MipsTargetMachine.cpp - Define TargetMachine for Mips -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Implements the info about Mips target spec.
//
//===----------------------------------------------------------------------===//

#include "MipsTargetMachine.h"
#include "Mips.h"
#include "Mips16FrameLowering.h"
#include "Mips16HardFloat.h"
#include "Mips16ISelDAGToDAG.h"
#include "Mips16ISelLowering.h"
#include "Mips16InstrInfo.h"
#include "MipsFrameLowering.h"
#include "MipsInstrInfo.h"
#include "MipsModuleISelDAGToDAG.h"
#include "MipsOs16.h"
#include "MipsSEFrameLowering.h"
#include "MipsSEISelDAGToDAG.h"
#include "MipsSEISelLowering.h"
#include "MipsSEInstrInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/PassManager.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Scalar.h"
using namespace llvm;

#define DEBUG_TYPE "mips"

extern "C" void LLVMInitializeMipsTarget() {
  // Register the target.
  RegisterTargetMachine<MipsebTargetMachine> X(TheMipsTarget);
  RegisterTargetMachine<MipselTargetMachine> Y(TheMipselTarget);
  RegisterTargetMachine<MipsebTargetMachine> A(TheMips64Target);
  RegisterTargetMachine<MipselTargetMachine> B(TheMips64elTarget);
}

// On function prologue, the stack is created by decrementing
// its pointer. Once decremented, all references are done with positive
// offset from the stack/frame pointer, using StackGrowsUp enables
// an easier handling.
// Using CodeModel::Large enables different CALL behavior.
MipsTargetMachine::MipsTargetMachine(const Target &T, StringRef TT,
                                     StringRef CPU, StringRef FS,
                                     const TargetOptions &Options,
                                     Reloc::Model RM, CodeModel::Model CM,
                                     CodeGenOpt::Level OL, bool isLittle)
    : LLVMTargetMachine(T, TT, CPU, FS, Options, RM, CM, OL),
      isLittle(isLittle), Subtarget(nullptr),
      DefaultSubtarget(TT, CPU, FS, isLittle, this),
      NoMips16Subtarget(TT, CPU, FS.empty() ? "-mips16" : FS.str() + ",-mips16",
                        isLittle, this),
      Mips16Subtarget(TT, CPU, FS.empty() ? "+mips16" : FS.str() + ",+mips16",
                      isLittle, this) {
  Subtarget = &DefaultSubtarget;
  initAsmInfo();
}

void MipsebTargetMachine::anchor() { }

MipsebTargetMachine::
MipsebTargetMachine(const Target &T, StringRef TT,
                    StringRef CPU, StringRef FS, const TargetOptions &Options,
                    Reloc::Model RM, CodeModel::Model CM,
                    CodeGenOpt::Level OL)
  : MipsTargetMachine(T, TT, CPU, FS, Options, RM, CM, OL, false) {}

void MipselTargetMachine::anchor() { }

MipselTargetMachine::
MipselTargetMachine(const Target &T, StringRef TT,
                    StringRef CPU, StringRef FS, const TargetOptions &Options,
                    Reloc::Model RM, CodeModel::Model CM,
                    CodeGenOpt::Level OL)
  : MipsTargetMachine(T, TT, CPU, FS, Options, RM, CM, OL, true) {}

const MipsSubtarget *
MipsTargetMachine::getSubtargetImpl(const Function &F) const {
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
  bool hasMips16Attr =
      !FnAttrs.getAttribute(AttributeSet::FunctionIndex, "mips16")
           .hasAttribute(Attribute::None);
  bool hasNoMips16Attr =
      !FnAttrs.getAttribute(AttributeSet::FunctionIndex, "nomips16")
           .hasAttribute(Attribute::None);

  // FIXME: This is related to the code below to reset the target options,
  // we need to know whether or not the soft float flag is set on the
  // function before we can generate a subtarget. We also need to use
  // it as a key for the subtarget since that can be the only difference
  // between two functions.
  Attribute SFAttr =
      FnAttrs.getAttribute(AttributeSet::FunctionIndex, "use-soft-float");
  bool softFloat = !SFAttr.hasAttribute(Attribute::None)
                       ? SFAttr.getValueAsString() == "true"
                       : Options.UseSoftFloat;

  if (hasMips16Attr)
    FS += FS.empty() ? "+mips16" : ",+mips16";
  else if (hasNoMips16Attr)
    FS += FS.empty() ? "-mips16" : ",-mips16";

  auto &I = SubtargetMap[CPU + FS + (softFloat ? "use-soft-float=true"
                                               : "use-soft-float=false")];
  if (!I) {
    // This needs to be done before we create a new subtarget since any
    // creation will depend on the TM and the code generation flags on the
    // function that reside in TargetOptions.
    resetTargetOptions(F);
    I = llvm::make_unique<MipsSubtarget>(TargetTriple, CPU, FS, isLittle, this);
  }
  return I.get();
}

void MipsTargetMachine::resetSubtarget(MachineFunction *MF) {
  DEBUG(dbgs() << "resetSubtarget\n");

  Subtarget = const_cast<MipsSubtarget *>(getSubtargetImpl(*MF->getFunction()));
  MF->setSubtarget(Subtarget);
  return;
}

namespace {
/// Mips Code Generator Pass Configuration Options.
class MipsPassConfig : public TargetPassConfig {
public:
  MipsPassConfig(MipsTargetMachine *TM, PassManagerBase &PM)
    : TargetPassConfig(TM, PM) {
    // The current implementation of long branch pass requires a scratch
    // register ($at) to be available before branch instructions. Tail merging
    // can break this requirement, so disable it when long branch pass is
    // enabled.
    EnableTailMerge = !getMipsSubtarget().enableLongBranchPass();
  }

  MipsTargetMachine &getMipsTargetMachine() const {
    return getTM<MipsTargetMachine>();
  }

  const MipsSubtarget &getMipsSubtarget() const {
    return *getMipsTargetMachine().getSubtargetImpl();
  }

  void addIRPasses() override;
  bool addInstSelector() override;
  void addMachineSSAOptimization() override;
  bool addPreEmitPass() override;

  bool addPreRegAlloc() override;

};
} // namespace

TargetPassConfig *MipsTargetMachine::createPassConfig(PassManagerBase &PM) {
  return new MipsPassConfig(this, PM);
}

void MipsPassConfig::addIRPasses() {
  TargetPassConfig::addIRPasses();
  addPass(createAtomicExpandPass(&getMipsTargetMachine()));
  if (getMipsSubtarget().os16())
    addPass(createMipsOs16(getMipsTargetMachine()));
  if (getMipsSubtarget().inMips16HardFloat())
    addPass(createMips16HardFloat(getMipsTargetMachine()));
}
// Install an instruction selector pass using
// the ISelDag to gen Mips code.
bool MipsPassConfig::addInstSelector() {
  addPass(createMipsModuleISelDag(getMipsTargetMachine()));
  addPass(createMips16ISelDag(getMipsTargetMachine()));
  addPass(createMipsSEISelDag(getMipsTargetMachine()));
  return false;
}

void MipsPassConfig::addMachineSSAOptimization() {
  addPass(createMipsOptimizePICCallPass(getMipsTargetMachine()));
  TargetPassConfig::addMachineSSAOptimization();
}

bool MipsPassConfig::addPreRegAlloc() {
  if (getOptLevel() == CodeGenOpt::None) {
    addPass(createMipsOptimizePICCallPass(getMipsTargetMachine()));
    return true;
  }
  else
    return false;
}

void MipsTargetMachine::addAnalysisPasses(PassManagerBase &PM) {
  if (Subtarget->allowMixed16_32()) {
    DEBUG(errs() << "No ");
    //FIXME: The Basic Target Transform Info
    // pass needs to become a function pass instead of
    // being an immutable pass and then this method as it exists now
    // would be unnecessary.
    PM.add(createNoTargetTransformInfoPass());
  } else
    LLVMTargetMachine::addAnalysisPasses(PM);
  DEBUG(errs() << "Target Transform Info Pass Added\n");
}

// Implemented by targets that want to run passes immediately before
// machine code is emitted. return true if -print-machineinstrs should
// print out the code after the passes.
bool MipsPassConfig::addPreEmitPass() {
  MipsTargetMachine &TM = getMipsTargetMachine();
  addPass(createMipsDelaySlotFillerPass(TM));
  addPass(createMipsLongBranchPass(TM));
  addPass(createMipsConstantIslandPass(TM));
  return true;
}
