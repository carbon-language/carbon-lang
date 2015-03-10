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
#include "NVPTXTargetObjectFile.h"
#include "NVPTXTargetTransformInfo.h"
#include "llvm/Analysis/Passes.h"
#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/CodeGen/MachineFunctionAnalysis.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/IRPrintingPasses.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Verifier.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSubtargetInfo.h"
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
void initializeNVPTXAllocaHoistingPass(PassRegistry &);
void initializeNVPTXAssignValidGlobalNamesPass(PassRegistry&);
void initializeNVPTXFavorNonGenericAddrSpacesPass(PassRegistry &);
void initializeNVPTXLowerStructArgsPass(PassRegistry &);
}

extern "C" void LLVMInitializeNVPTXTarget() {
  // Register the target.
  RegisterTargetMachine<NVPTXTargetMachine32> X(TheNVPTXTarget32);
  RegisterTargetMachine<NVPTXTargetMachine64> Y(TheNVPTXTarget64);

  // FIXME: This pass is really intended to be invoked during IR optimization,
  // but it's very NVPTX-specific.
  initializeNVVMReflectPass(*PassRegistry::getPassRegistry());
  initializeGenericToNVVMPass(*PassRegistry::getPassRegistry());
  initializeNVPTXAllocaHoistingPass(*PassRegistry::getPassRegistry());
  initializeNVPTXAssignValidGlobalNamesPass(*PassRegistry::getPassRegistry());
  initializeNVPTXFavorNonGenericAddrSpacesPass(
    *PassRegistry::getPassRegistry());
  initializeNVPTXLowerStructArgsPass(*PassRegistry::getPassRegistry());
}

static std::string computeDataLayout(bool is64Bit) {
  std::string Ret = "e";

  if (!is64Bit)
    Ret += "-p:32:32";

  Ret += "-i64:64-v16:16-v32:32-n16:32:64";

  return Ret;
}

NVPTXTargetMachine::NVPTXTargetMachine(const Target &T, StringRef TT,
                                       StringRef CPU, StringRef FS,
                                       const TargetOptions &Options,
                                       Reloc::Model RM, CodeModel::Model CM,
                                       CodeGenOpt::Level OL, bool is64bit)
    : LLVMTargetMachine(T, TT, CPU, FS, Options, RM, CM, OL), is64bit(is64bit),
      TLOF(make_unique<NVPTXTargetObjectFile>()),
      DL(computeDataLayout(is64bit)), Subtarget(TT, CPU, FS, *this) {
  if (Triple(TT).getOS() == Triple::NVCL)
    drvInterface = NVPTX::NVCL;
  else
    drvInterface = NVPTX::CUDA;
  initAsmInfo();
}

NVPTXTargetMachine::~NVPTXTargetMachine() {}

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

  void addIRPasses() override;
  bool addInstSelector() override;
  void addPostRegAlloc() override;
  void addMachineSSAOptimization() override;

  FunctionPass *createTargetRegisterAllocator(bool) override;
  void addFastRegAlloc(FunctionPass *RegAllocPass) override;
  void addOptimizedRegAlloc(FunctionPass *RegAllocPass) override;
};
} // end anonymous namespace

TargetPassConfig *NVPTXTargetMachine::createPassConfig(PassManagerBase &PM) {
  NVPTXPassConfig *PassConfig = new NVPTXPassConfig(this, PM);
  return PassConfig;
}

TargetIRAnalysis NVPTXTargetMachine::getTargetIRAnalysis() {
  return TargetIRAnalysis(
      [this](Function &) { return TargetTransformInfo(NVPTXTTIImpl(this)); });
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
  addPass(createStraightLineStrengthReducePass());
  addPass(createSeparateConstOffsetFromGEPPass());
  // The SeparateConstOffsetFromGEP pass creates variadic bases that can be used
  // by multiple GEPs. Run GVN or EarlyCSE to really reuse them. GVN generates
  // significantly better code than EarlyCSE for some of our benchmarks.
  if (getOptLevel() == CodeGenOpt::Aggressive)
    addPass(createGVNPass());
  else
    addPass(createEarlyCSEPass());
  // Both FavorNonGenericAddrSpaces and SeparateConstOffsetFromGEP may leave
  // some dead code.  We could remove dead code in an ad-hoc manner, but that
  // requires manual work and might be error-prone.
  //
  // The FavorNonGenericAddrSpaces pass shortcuts unnecessary addrspacecasts,
  // and leave them unused.
  //
  // SeparateConstOffsetFromGEP rebuilds a new index from the old index, and the
  // old index and some of its intermediate results may become unused.
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

void NVPTXPassConfig::addPostRegAlloc() {
  addPass(createNVPTXPrologEpilogPass(), false);
}

FunctionPass *NVPTXPassConfig::createTargetRegisterAllocator(bool) {
  return nullptr; // No reg alloc
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

void NVPTXPassConfig::addMachineSSAOptimization() {
  // Pre-ra tail duplication.
  if (addPass(&EarlyTailDuplicateID))
    printAndVerify("After Pre-RegAlloc TailDuplicate");

  // Optimize PHIs before DCE: removing dead PHI cycles may make more
  // instructions dead.
  addPass(&OptimizePHIsID);

  // This pass merges large allocas. StackSlotColoring is a different pass
  // which merges spill slots.
  addPass(&StackColoringID);

  // If the target requests it, assign local variables to stack slots relative
  // to one another and simplify frame index references where possible.
  addPass(&LocalStackSlotAllocationID);

  // With optimization, dead code should already be eliminated. However
  // there is one known exception: lowered code for arguments that are only
  // used by tail calls, where the tail calls reuse the incoming stack
  // arguments directly (see t11 in test/CodeGen/X86/sibcall.ll).
  addPass(&DeadMachineInstructionElimID);
  printAndVerify("After codegen DCE pass");

  // Allow targets to insert passes that improve instruction level parallelism,
  // like if-conversion. Such passes will typically need dominator trees and
  // loop info, just like LICM and CSE below.
  if (addILPOpts())
    printAndVerify("After ILP optimizations");

  addPass(&MachineLICMID);
  addPass(&MachineCSEID);

  addPass(&MachineSinkingID);
  printAndVerify("After Machine LICM, CSE and Sinking passes");

  addPass(&PeepholeOptimizerID);
  printAndVerify("After codegen peephole optimization pass");
}
