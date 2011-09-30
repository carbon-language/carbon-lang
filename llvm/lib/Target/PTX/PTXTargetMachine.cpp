//===-- PTXTargetMachine.cpp - Define TargetMachine for PTX ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Top-level implementation for the PTX target.
//
//===----------------------------------------------------------------------===//

#include "PTX.h"
#include "PTXTargetMachine.h"
#include "llvm/PassManager.h"
#include "llvm/Analysis/Passes.h"
#include "llvm/Analysis/Verifier.h"
#include "llvm/Assembly/PrintModulePass.h"
#include "llvm/ADT/OwningPtr.h"
#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/CodeGen/MachineFunctionAnalysis.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetLowering.h"
#include "llvm/Target/TargetLoweringObjectFile.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/Target/TargetSubtargetInfo.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/TargetRegistry.h"


using namespace llvm;

namespace llvm {
  MCStreamer *createPTXAsmStreamer(MCContext &Ctx, formatted_raw_ostream &OS,
                                   bool isVerboseAsm, bool useLoc,
                                   bool useCFI,
                                   MCInstPrinter *InstPrint,
                                   MCCodeEmitter *CE,
                                   MCAsmBackend *MAB,
                                   bool ShowInst);
}

extern "C" void LLVMInitializePTXTarget() {

  RegisterTargetMachine<PTX32TargetMachine> X(ThePTX32Target);
  RegisterTargetMachine<PTX64TargetMachine> Y(ThePTX64Target);

  TargetRegistry::RegisterAsmStreamer(ThePTX32Target, createPTXAsmStreamer);
  TargetRegistry::RegisterAsmStreamer(ThePTX64Target, createPTXAsmStreamer);
}

namespace {
  const char* DataLayout32 =
    "e-p:32:32-i64:32:32-f64:32:32-v128:32:128-v64:32:64-n32:64";
  const char* DataLayout64 =
    "e-p:64:64-i64:32:32-f64:32:32-v128:32:128-v64:32:64-n32:64";

  // Copied from LLVMTargetMachine.cpp
  void printNoVerify(PassManagerBase &PM, const char *Banner) {
    if (PrintMachineCode)
      PM.add(createMachineFunctionPrinterPass(dbgs(), Banner));
  }

  void printAndVerify(PassManagerBase &PM,
                      const char *Banner) {
    if (PrintMachineCode)
      PM.add(createMachineFunctionPrinterPass(dbgs(), Banner));

    //if (VerifyMachineCode)
    //  PM.add(createMachineVerifierPass(Banner));
  }
}

// DataLayout and FrameLowering are filled with dummy data
PTXTargetMachine::PTXTargetMachine(const Target &T,
                                   StringRef TT, StringRef CPU, StringRef FS,
                                   Reloc::Model RM, CodeModel::Model CM,
                                   bool is64Bit)
  : LLVMTargetMachine(T, TT, CPU, FS, RM, CM),
    DataLayout(is64Bit ? DataLayout64 : DataLayout32),
    Subtarget(TT, CPU, FS, is64Bit),
    FrameLowering(Subtarget),
    InstrInfo(*this),
    TSInfo(*this),
    TLInfo(*this) {
}

PTX32TargetMachine::PTX32TargetMachine(const Target &T, StringRef TT,
                                       StringRef CPU, StringRef FS,
                                       Reloc::Model RM, CodeModel::Model CM)
  : PTXTargetMachine(T, TT, CPU, FS, RM, CM, false) {
}

PTX64TargetMachine::PTX64TargetMachine(const Target &T, StringRef TT,
                                       StringRef CPU, StringRef FS,
                                       Reloc::Model RM, CodeModel::Model CM)
  : PTXTargetMachine(T, TT, CPU, FS, RM, CM, true) {
}

bool PTXTargetMachine::addInstSelector(PassManagerBase &PM,
                                       CodeGenOpt::Level OptLevel) {
  PM.add(createPTXISelDag(*this, OptLevel));
  return false;
}

bool PTXTargetMachine::addPostRegAlloc(PassManagerBase &PM,
                                       CodeGenOpt::Level OptLevel) {
  // PTXMFInfoExtract must after register allocation!
  //PM.add(createPTXMFInfoExtract(*this, OptLevel));
  return false;
}

bool PTXTargetMachine::addPassesToEmitFile(PassManagerBase &PM,
                                           formatted_raw_ostream &Out,
                                           CodeGenFileType FileType,
                                           CodeGenOpt::Level OptLevel,
                                           bool DisableVerify) {
  // This is mostly based on LLVMTargetMachine::addPassesToEmitFile

  // Add common CodeGen passes.
  MCContext *Context = 0;
  if (addCommonCodeGenPasses(PM, OptLevel, DisableVerify, Context))
    return true;
  assert(Context != 0 && "Failed to get MCContext");

  if (hasMCSaveTempLabels())
    Context->setAllowTemporaryLabels(false);

  const MCAsmInfo &MAI = *getMCAsmInfo();
  const MCSubtargetInfo &STI = getSubtarget<MCSubtargetInfo>();
  OwningPtr<MCStreamer> AsmStreamer;

  switch (FileType) {
  default: return true;
  case CGFT_AssemblyFile: {
    MCInstPrinter *InstPrinter =
      getTarget().createMCInstPrinter(MAI.getAssemblerDialect(), MAI, STI);

    // Create a code emitter if asked to show the encoding.
    MCCodeEmitter *MCE = 0;
    MCAsmBackend *MAB = 0;

    MCStreamer *S = getTarget().createAsmStreamer(*Context, Out,
                                                  true, /* verbose asm */
                                                  hasMCUseLoc(),
                                                  hasMCUseCFI(),
                                                  InstPrinter,
                                                  MCE, MAB,
                                                  false /* show MC encoding */);
    AsmStreamer.reset(S);
    break;
  }
  case CGFT_ObjectFile: {
    llvm_unreachable("Object file emission is not supported with PTX");
  }
  case CGFT_Null:
    // The Null output is intended for use for performance analysis and testing,
    // not real users.
    AsmStreamer.reset(createNullStreamer(*Context));
    break;
  }

  // MC Logging
  //AsmStreamer.reset(createLoggingStreamer(AsmStreamer.take(), errs()));

  // Create the AsmPrinter, which takes ownership of AsmStreamer if successful.
  FunctionPass *Printer = getTarget().createAsmPrinter(*this, *AsmStreamer);
  if (Printer == 0)
    return true;

  // If successful, createAsmPrinter took ownership of AsmStreamer.
  AsmStreamer.take();

  PM.add(Printer);

  PM.add(createGCInfoDeleter());
  return false;
}

bool PTXTargetMachine::addCommonCodeGenPasses(PassManagerBase &PM,
                                              CodeGenOpt::Level OptLevel,
                                              bool DisableVerify,
                                              MCContext *&OutContext) {
  // Add standard LLVM codegen passes.
  // This is derived from LLVMTargetMachine::addCommonCodeGenPasses, with some
  // modifications for the PTX target.

  // Standard LLVM-Level Passes.

  // Basic AliasAnalysis support.
  // Add TypeBasedAliasAnalysis before BasicAliasAnalysis so that
  // BasicAliasAnalysis wins if they disagree. This is intended to help
  // support "obvious" type-punning idioms.
  PM.add(createTypeBasedAliasAnalysisPass());
  PM.add(createBasicAliasAnalysisPass());

  // Before running any passes, run the verifier to determine if the input
  // coming from the front-end and/or optimizer is valid.
  if (!DisableVerify)
    PM.add(createVerifierPass());

  // Run loop strength reduction before anything else.
  if (OptLevel != CodeGenOpt::None) {
    PM.add(createLoopStrengthReducePass(getTargetLowering()));
    //PM.add(createPrintFunctionPass("\n\n*** Code after LSR ***\n", &dbgs()));
  }

  PM.add(createGCLoweringPass());

  // Make sure that no unreachable blocks are instruction selected.
  PM.add(createUnreachableBlockEliminationPass());

  PM.add(createLowerInvokePass(getTargetLowering()));
  // The lower invoke pass may create unreachable code. Remove it.
  PM.add(createUnreachableBlockEliminationPass());

  if (OptLevel != CodeGenOpt::None)
    PM.add(createCodeGenPreparePass(getTargetLowering()));

  PM.add(createStackProtectorPass(getTargetLowering()));

  addPreISel(PM, OptLevel);

  //PM.add(createPrintFunctionPass("\n\n"
  //                               "*** Final LLVM Code input to ISel ***\n",
  //                               &dbgs()));

  // All passes which modify the LLVM IR are now complete; run the verifier
  // to ensure that the IR is valid.
  if (!DisableVerify)
    PM.add(createVerifierPass());

  // Standard Lower-Level Passes.

  // Install a MachineModuleInfo class, which is an immutable pass that holds
  // all the per-module stuff we're generating, including MCContext.
  MachineModuleInfo *MMI = new MachineModuleInfo(*getMCAsmInfo(),
                                                 *getRegisterInfo(),
                                    &getTargetLowering()->getObjFileLowering());
  PM.add(MMI);
  OutContext = &MMI->getContext(); // Return the MCContext specifically by-ref.

  // Set up a MachineFunction for the rest of CodeGen to work on.
  PM.add(new MachineFunctionAnalysis(*this, OptLevel));

  // Ask the target for an isel.
  if (addInstSelector(PM, OptLevel))
    return true;

  // Print the instruction selected machine code...
  printAndVerify(PM, "After Instruction Selection");

  // Expand pseudo-instructions emitted by ISel.
  PM.add(createExpandISelPseudosPass());

  // Pre-ra tail duplication.
  if (OptLevel != CodeGenOpt::None) {
    PM.add(createTailDuplicatePass(true));
    printAndVerify(PM, "After Pre-RegAlloc TailDuplicate");
  }

  // Optimize PHIs before DCE: removing dead PHI cycles may make more
  // instructions dead.
  if (OptLevel != CodeGenOpt::None)
    PM.add(createOptimizePHIsPass());

  // If the target requests it, assign local variables to stack slots relative
  // to one another and simplify frame index references where possible.
  PM.add(createLocalStackSlotAllocationPass());

  if (OptLevel != CodeGenOpt::None) {
    // With optimization, dead code should already be eliminated. However
    // there is one known exception: lowered code for arguments that are only
    // used by tail calls, where the tail calls reuse the incoming stack
    // arguments directly (see t11 in test/CodeGen/X86/sibcall.ll).
    PM.add(createDeadMachineInstructionElimPass());
    printAndVerify(PM, "After codegen DCE pass");

    PM.add(createMachineLICMPass());
    PM.add(createMachineCSEPass());
    PM.add(createMachineSinkingPass());
    printAndVerify(PM, "After Machine LICM, CSE and Sinking passes");

    PM.add(createPeepholeOptimizerPass());
    printAndVerify(PM, "After codegen peephole optimization pass");
  }

  // Run pre-ra passes.
  if (addPreRegAlloc(PM, OptLevel))
    printAndVerify(PM, "After PreRegAlloc passes");

  // Perform register allocation.
  PM.add(createPTXRegisterAllocator());
  printAndVerify(PM, "After Register Allocation");

  // Perform stack slot coloring and post-ra machine LICM.
  if (OptLevel != CodeGenOpt::None) {
    // FIXME: Re-enable coloring with register when it's capable of adding
    // kill markers.
    PM.add(createStackSlotColoringPass(false));

    // FIXME: Post-RA LICM has asserts that fire on virtual registers.
    // Run post-ra machine LICM to hoist reloads / remats.
    //if (!DisablePostRAMachineLICM)
    //  PM.add(createMachineLICMPass(false));

    printAndVerify(PM, "After StackSlotColoring and postra Machine LICM");
  }

  // Run post-ra passes.
  if (addPostRegAlloc(PM, OptLevel))
    printAndVerify(PM, "After PostRegAlloc passes");

  PM.add(createExpandPostRAPseudosPass());
  printAndVerify(PM, "After ExpandPostRAPseudos");

  // Insert prolog/epilog code.  Eliminate abstract frame index references...
  PM.add(createPrologEpilogCodeInserter());
  printAndVerify(PM, "After PrologEpilogCodeInserter");

  // Run pre-sched2 passes.
  if (addPreSched2(PM, OptLevel))
    printAndVerify(PM, "After PreSched2 passes");

  // Second pass scheduler.
  if (OptLevel != CodeGenOpt::None) {
    PM.add(createPostRAScheduler(OptLevel));
    printAndVerify(PM, "After PostRAScheduler");
  }

  // Branch folding must be run after regalloc and prolog/epilog insertion.
  if (OptLevel != CodeGenOpt::None) {
    PM.add(createBranchFoldingPass(getEnableTailMergeDefault()));
    printNoVerify(PM, "After BranchFolding");
  }

  // Tail duplication.
  if (OptLevel != CodeGenOpt::None) {
    PM.add(createTailDuplicatePass(false));
    printNoVerify(PM, "After TailDuplicate");
  }

  PM.add(createGCMachineCodeAnalysisPass());

  //if (PrintGCInfo)
  //  PM.add(createGCInfoPrinter(dbgs()));

  if (OptLevel != CodeGenOpt::None) {
    PM.add(createCodePlacementOptPass());
    printNoVerify(PM, "After CodePlacementOpt");
  }

  if (addPreEmitPass(PM, OptLevel))
    printNoVerify(PM, "After PreEmit passes");

  PM.add(createPTXMFInfoExtract(*this, OptLevel));
  PM.add(createPTXFPRoundingModePass(*this, OptLevel));

  return false;
}
