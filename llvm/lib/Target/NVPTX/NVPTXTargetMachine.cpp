//===-- NVPTXTargetMachine.cpp - Define TargetMachine for NVPTX -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Top-level implementation for the NVPTX target.
//
//===----------------------------------------------------------------------===//

#include "NVPTXTargetMachine.h"
#include "MCTargetDesc/NVPTXMCAsmInfo.h"
#include "NVPTX.h"
#include "NVPTXAllocaHoisting.h"
#include "NVPTXLowerAggrCopies.h"
#include "llvm/Analysis/Passes.h"
#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/CodeGen/MachineFunctionAnalysis.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/IRPrintingPasses.h"
#include "llvm/IR/Verifier.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/PassManager.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetLowering.h"
#include "llvm/Target/TargetLoweringObjectFile.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/Target/TargetSubtargetInfo.h"
#include "llvm/Transforms/Scalar.h"

using namespace llvm;

namespace llvm {
void initializeNVVMReflectPass(PassRegistry&);
void initializeGenericToNVVMPass(PassRegistry&);
void initializeNVPTXAssignValidGlobalNamesPass(PassRegistry&);
void initializeNVPTXFavorNonGenericAddrSpacesPass(PassRegistry &);
}

extern "C" void LLVMInitializeNVPTXTarget() {
  // Register the target.
  RegisterTargetMachine<NVPTXTargetMachine32> X(TheNVPTXTarget32);
  RegisterTargetMachine<NVPTXTargetMachine64> Y(TheNVPTXTarget64);

  // FIXME: This pass is really intended to be invoked during IR optimization,
  // but it's very NVPTX-specific.
  initializeNVVMReflectPass(*PassRegistry::getPassRegistry());
  initializeGenericToNVVMPass(*PassRegistry::getPassRegistry());
  initializeNVPTXAssignValidGlobalNamesPass(*PassRegistry::getPassRegistry());
  initializeNVPTXFavorNonGenericAddrSpacesPass(
    *PassRegistry::getPassRegistry());
}

static std::string computeDataLayout(const NVPTXSubtarget &ST) {
  std::string Ret = "e";

  if (!ST.is64Bit())
    Ret += "-p:32:32";

  Ret += "-i64:64-v16:16-v32:32-n16:32:64";

  return Ret;
}

NVPTXTargetMachine::NVPTXTargetMachine(
    const Target &T, StringRef TT, StringRef CPU, StringRef FS,
    const TargetOptions &Options, Reloc::Model RM, CodeModel::Model CM,
    CodeGenOpt::Level OL, bool is64bit)
    : LLVMTargetMachine(T, TT, CPU, FS, Options, RM, CM, OL),
      Subtarget(TT, CPU, FS, is64bit), DL(computeDataLayout(Subtarget)),
      InstrInfo(*this), TLInfo(*this), TSInfo(*this),
      FrameLowering(
          *this, is64bit) /*FrameInfo(TargetFrameInfo::StackGrowsUp, 8, 0)*/ {
  initAsmInfo();
}

void NVPTXTargetMachine32::anchor() {}

NVPTXTargetMachine32::NVPTXTargetMachine32(
    const Target &T, StringRef TT, StringRef CPU, StringRef FS,
    const TargetOptions &Options, Reloc::Model RM, CodeModel::Model CM,
    CodeGenOpt::Level OL)
    : NVPTXTargetMachine(T, TT, CPU, FS, Options, RM, CM, OL, false) {}

void NVPTXTargetMachine64::anchor() {}

NVPTXTargetMachine64::NVPTXTargetMachine64(
    const Target &T, StringRef TT, StringRef CPU, StringRef FS,
    const TargetOptions &Options, Reloc::Model RM, CodeModel::Model CM,
    CodeGenOpt::Level OL)
    : NVPTXTargetMachine(T, TT, CPU, FS, Options, RM, CM, OL, true) {}

namespace {
class NVPTXPassConfig : public TargetPassConfig {
public:
  NVPTXPassConfig(NVPTXTargetMachine *TM, PassManagerBase &PM)
      : TargetPassConfig(TM, PM) {}

  NVPTXTargetMachine &getNVPTXTargetMachine() const {
    return getTM<NVPTXTargetMachine>();
  }

  virtual void addIRPasses();
  virtual bool addInstSelector();
  virtual bool addPreRegAlloc();
  virtual bool addPostRegAlloc();

  virtual FunctionPass *createTargetRegisterAllocator(bool) override;
  virtual void addFastRegAlloc(FunctionPass *RegAllocPass);
  virtual void addOptimizedRegAlloc(FunctionPass *RegAllocPass);
};
} // end anonymous namespace

TargetPassConfig *NVPTXTargetMachine::createPassConfig(PassManagerBase &PM) {
  NVPTXPassConfig *PassConfig = new NVPTXPassConfig(this, PM);
  return PassConfig;
}

void NVPTXPassConfig::addIRPasses() {
  // The following passes are known to not play well with virtual regs hanging
  // around after register allocation (which in our case, is *all* registers).
  // We explicitly disable them here.  We do, however, need some functionality
  // of the PrologEpilogCodeInserter pass, so we emulate that behavior in the
  // NVPTXPrologEpilog pass (see NVPTXPrologEpilogPass.cpp).
  disablePass(&PrologEpilogCodeInserterID);
  disablePass(&MachineCopyPropagationID);
  disablePass(&BranchFolderPassID);
  disablePass(&TailDuplicateID);

  addPass(createNVPTXImageOptimizerPass());
  TargetPassConfig::addIRPasses();
  addPass(createNVPTXAssignValidGlobalNamesPass());
  addPass(createGenericToNVVMPass());
  addPass(createNVPTXFavorNonGenericAddrSpacesPass());
  // The FavorNonGenericAddrSpaces pass may remove instructions and leave some
  // values unused. Therefore, we run a DCE pass right afterwards. We could
  // remove unused values in an ad-hoc manner, but it requires manual work and
  // might be error-prone.
  addPass(createDeadCodeEliminationPass());
}

bool NVPTXPassConfig::addInstSelector() {
  const NVPTXSubtarget &ST =
    getTM<NVPTXTargetMachine>().getSubtarget<NVPTXSubtarget>();

  addPass(createLowerAggrCopies());
  addPass(createAllocaHoisting());
  addPass(createNVPTXISelDag(getNVPTXTargetMachine(), getOptLevel()));

  if (!ST.hasImageHandles())
    addPass(createNVPTXReplaceImageHandlesPass());

  return false;
}

bool NVPTXPassConfig::addPreRegAlloc() { return false; }
bool NVPTXPassConfig::addPostRegAlloc() {
  addPass(createNVPTXPrologEpilogPass());
  return false;
}

FunctionPass *NVPTXPassConfig::createTargetRegisterAllocator(bool) {
  return 0; // No reg alloc
}

void NVPTXPassConfig::addFastRegAlloc(FunctionPass *RegAllocPass) {
  assert(!RegAllocPass && "NVPTX uses no regalloc!");
  addPass(&PHIEliminationID);
  addPass(&TwoAddressInstructionPassID);
}

void NVPTXPassConfig::addOptimizedRegAlloc(FunctionPass *RegAllocPass) {
  assert(!RegAllocPass && "NVPTX uses no regalloc!");

  addPass(&ProcessImplicitDefsID);
  addPass(&LiveVariablesID);
  addPass(&MachineLoopInfoID);
  addPass(&PHIEliminationID);

  addPass(&TwoAddressInstructionPassID);
  addPass(&RegisterCoalescerID);

  // PreRA instruction scheduling.
  if (addPass(&MachineSchedulerID))
    printAndVerify("After Machine Scheduling");


  addPass(&StackSlotColoringID);

  // FIXME: Needs physical registers
  //addPass(&PostRAMachineLICMID);

  printAndVerify("After StackSlotColoring");
}
