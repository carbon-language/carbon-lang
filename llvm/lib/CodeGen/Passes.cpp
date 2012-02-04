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

#include "llvm/Analysis/Passes.h"
#include "llvm/Analysis/Verifier.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/PassManager.h"
#include "llvm/CodeGen/GCStrategy.h"
#include "llvm/CodeGen/MachineFunctionAnalysis.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/RegAllocRegistry.h"
#include "llvm/Target/TargetLowering.h"
#include "llvm/Target/TargetLoweringObjectFile.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/Assembly/PrintModulePass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"

using namespace llvm;

static cl::opt<bool> DisablePostRA("disable-post-ra", cl::Hidden,
    cl::desc("Disable Post Regalloc"));
static cl::opt<bool> DisableBranchFold("disable-branch-fold", cl::Hidden,
    cl::desc("Disable branch folding"));
static cl::opt<bool> DisableTailDuplicate("disable-tail-duplicate", cl::Hidden,
    cl::desc("Disable tail duplication"));
static cl::opt<bool> DisableEarlyTailDup("disable-early-taildup", cl::Hidden,
    cl::desc("Disable pre-register allocation tail duplication"));
static cl::opt<bool> EnableBlockPlacement("enable-block-placement",
    cl::Hidden, cl::desc("Enable probability-driven block placement"));
static cl::opt<bool> EnableBlockPlacementStats("enable-block-placement-stats",
    cl::Hidden, cl::desc("Collect probability-driven block placement stats"));
static cl::opt<bool> DisableCodePlace("disable-code-place", cl::Hidden,
    cl::desc("Disable code placement"));
static cl::opt<bool> DisableSSC("disable-ssc", cl::Hidden,
    cl::desc("Disable Stack Slot Coloring"));
static cl::opt<bool> DisableMachineDCE("disable-machine-dce", cl::Hidden,
    cl::desc("Disable Machine Dead Code Elimination"));
static cl::opt<bool> DisableMachineLICM("disable-machine-licm", cl::Hidden,
    cl::desc("Disable Machine LICM"));
static cl::opt<bool> DisableMachineCSE("disable-machine-cse", cl::Hidden,
    cl::desc("Disable Machine Common Subexpression Elimination"));
static cl::opt<bool> DisablePostRAMachineLICM("disable-postra-machine-licm",
    cl::Hidden,
    cl::desc("Disable Machine LICM"));
static cl::opt<bool> DisableMachineSink("disable-machine-sink", cl::Hidden,
    cl::desc("Disable Machine Sinking"));
static cl::opt<bool> DisableLSR("disable-lsr", cl::Hidden,
    cl::desc("Disable Loop Strength Reduction Pass"));
static cl::opt<bool> DisableCGP("disable-cgp", cl::Hidden,
    cl::desc("Disable Codegen Prepare"));
static cl::opt<bool> DisableCopyProp("disable-copyprop", cl::Hidden,
    cl::desc("Disable Copy Propagation pass"));
static cl::opt<bool> PrintLSR("print-lsr-output", cl::Hidden,
    cl::desc("Print LLVM IR produced by the loop-reduce pass"));
static cl::opt<bool> PrintISelInput("print-isel-input", cl::Hidden,
    cl::desc("Print LLVM IR input to isel pass"));
static cl::opt<bool> PrintGCInfo("print-gc", cl::Hidden,
    cl::desc("Dump garbage collector data"));
static cl::opt<bool> VerifyMachineCode("verify-machineinstrs", cl::Hidden,
    cl::desc("Verify generated machine code"),
    cl::init(getenv("LLVM_VERIFY_MACHINEINSTRS")!=NULL));

// Enable or disable FastISel. Both options are needed, because
// FastISel is enabled by default with -fast, and we wish to be
// able to enable or disable fast-isel independently from -O0.
static cl::opt<cl::boolOrDefault>
EnableFastISelOption("fast-isel", cl::Hidden,
  cl::desc("Enable the \"fast\" instruction selector"));

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


void TargetPassConfig::printNoVerify(const char *Banner) const {
  if (TM->shouldPrintMachineCode())
    PM.add(createMachineFunctionPrinterPass(dbgs(), Banner));
}

void TargetPassConfig::printAndVerify(const char *Banner) const {
  if (TM->shouldPrintMachineCode())
    PM.add(createMachineFunctionPrinterPass(dbgs(), Banner));

  if (VerifyMachineCode)
    PM.add(createMachineVerifierPass(Banner));
}

/// addCodeGenPasses - Add standard LLVM codegen passes used for both
/// emitting to assembly files or machine code output.
///
bool TargetPassConfig::addCodeGenPasses(MCContext *&OutContext) {
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
  if (getOptLevel() != CodeGenOpt::None && !DisableLSR) {
    PM.add(createLoopStrengthReducePass(getTargetLowering()));
    if (PrintLSR)
      PM.add(createPrintFunctionPass("\n\n*** Code after LSR ***\n", &dbgs()));
  }

  PM.add(createGCLoweringPass());

  // Make sure that no unreachable blocks are instruction selected.
  PM.add(createUnreachableBlockEliminationPass());

  // Turn exception handling constructs into something the code generators can
  // handle.
  switch (TM->getMCAsmInfo()->getExceptionHandlingType()) {
  case ExceptionHandling::SjLj:
    // SjLj piggy-backs on dwarf for this bit. The cleanups done apply to both
    // Dwarf EH prepare needs to be run after SjLj prepare. Otherwise,
    // catch info can get misplaced when a selector ends up more than one block
    // removed from the parent invoke(s). This could happen when a landing
    // pad is shared by multiple invokes and is also a target of a normal
    // edge from elsewhere.
    PM.add(createSjLjEHPass(getTargetLowering()));
    // FALLTHROUGH
  case ExceptionHandling::DwarfCFI:
  case ExceptionHandling::ARM:
  case ExceptionHandling::Win64:
    PM.add(createDwarfEHPass(TM));
    break;
  case ExceptionHandling::None:
    PM.add(createLowerInvokePass(getTargetLowering()));

    // The lower invoke pass may create unreachable code. Remove it.
    PM.add(createUnreachableBlockEliminationPass());
    break;
  }

  if (getOptLevel() != CodeGenOpt::None && !DisableCGP)
    PM.add(createCodeGenPreparePass(getTargetLowering()));

  PM.add(createStackProtectorPass(getTargetLowering()));

  addPreISel();

  if (PrintISelInput)
    PM.add(createPrintFunctionPass("\n\n"
                                   "*** Final LLVM Code input to ISel ***\n",
                                   &dbgs()));

  // All passes which modify the LLVM IR are now complete; run the verifier
  // to ensure that the IR is valid.
  if (!DisableVerify)
    PM.add(createVerifierPass());

  // Standard Lower-Level Passes.

  // Install a MachineModuleInfo class, which is an immutable pass that holds
  // all the per-module stuff we're generating, including MCContext.
  MachineModuleInfo *MMI =
    new MachineModuleInfo(*TM->getMCAsmInfo(), *TM->getRegisterInfo(),
                          &getTargetLowering()->getObjFileLowering());
  PM.add(MMI);
  OutContext = &MMI->getContext(); // Return the MCContext specifically by-ref.

  // Set up a MachineFunction for the rest of CodeGen to work on.
  PM.add(new MachineFunctionAnalysis(*TM));

  // Enable FastISel with -fast, but allow that to be overridden.
  if (EnableFastISelOption == cl::BOU_TRUE ||
      (getOptLevel() == CodeGenOpt::None &&
       EnableFastISelOption != cl::BOU_FALSE))
    TM->setFastISel(true);

  // Ask the target for an isel.
  if (addInstSelector())
    return true;

  // Print the instruction selected machine code...
  printAndVerify("After Instruction Selection");

  // Expand pseudo-instructions emitted by ISel.
  PM.add(createExpandISelPseudosPass());

  // Pre-ra tail duplication.
  if (getOptLevel() != CodeGenOpt::None && !DisableEarlyTailDup) {
    PM.add(createTailDuplicatePass(true));
    printAndVerify("After Pre-RegAlloc TailDuplicate");
  }

  // Optimize PHIs before DCE: removing dead PHI cycles may make more
  // instructions dead.
  if (getOptLevel() != CodeGenOpt::None)
    PM.add(createOptimizePHIsPass());

  // If the target requests it, assign local variables to stack slots relative
  // to one another and simplify frame index references where possible.
  PM.add(createLocalStackSlotAllocationPass());

  if (getOptLevel() != CodeGenOpt::None) {
    // With optimization, dead code should already be eliminated. However
    // there is one known exception: lowered code for arguments that are only
    // used by tail calls, where the tail calls reuse the incoming stack
    // arguments directly (see t11 in test/CodeGen/X86/sibcall.ll).
    if (!DisableMachineDCE)
      PM.add(createDeadMachineInstructionElimPass());
    printAndVerify("After codegen DCE pass");

    if (!DisableMachineLICM)
      PM.add(createMachineLICMPass());
    if (!DisableMachineCSE)
      PM.add(createMachineCSEPass());
    if (!DisableMachineSink)
      PM.add(createMachineSinkingPass());
    printAndVerify("After Machine LICM, CSE and Sinking passes");

    PM.add(createPeepholeOptimizerPass());
    printAndVerify("After codegen peephole optimization pass");
  }

  // Run pre-ra passes.
  if (addPreRegAlloc())
    printAndVerify("After PreRegAlloc passes");

  // Perform register allocation.
  PM.add(createRegisterAllocator(getOptLevel()));
  printAndVerify("After Register Allocation");

  // Perform stack slot coloring and post-ra machine LICM.
  if (getOptLevel() != CodeGenOpt::None) {
    // FIXME: Re-enable coloring with register when it's capable of adding
    // kill markers.
    if (!DisableSSC)
      PM.add(createStackSlotColoringPass(false));

    // Run post-ra machine LICM to hoist reloads / remats.
    if (!DisablePostRAMachineLICM)
      PM.add(createMachineLICMPass(false));

    printAndVerify("After StackSlotColoring and postra Machine LICM");
  }

  // Run post-ra passes.
  if (addPostRegAlloc())
    printAndVerify("After PostRegAlloc passes");

  // Insert prolog/epilog code.  Eliminate abstract frame index references...
  PM.add(createPrologEpilogCodeInserter());
  printAndVerify("After PrologEpilogCodeInserter");

  // Branch folding must be run after regalloc and prolog/epilog insertion.
  if (getOptLevel() != CodeGenOpt::None && !DisableBranchFold) {
    PM.add(createBranchFoldingPass(getEnableTailMergeDefault()));
    printNoVerify("After BranchFolding");
  }

  // Tail duplication.
  if (getOptLevel() != CodeGenOpt::None && !DisableTailDuplicate) {
    PM.add(createTailDuplicatePass(false));
    printNoVerify("After TailDuplicate");
  }

  // Copy propagation.
  if (getOptLevel() != CodeGenOpt::None && !DisableCopyProp) {
    PM.add(createMachineCopyPropagationPass());
    printNoVerify("After copy propagation pass");
  }

  // Expand pseudo instructions before second scheduling pass.
  PM.add(createExpandPostRAPseudosPass());
  printNoVerify("After ExpandPostRAPseudos");

  // Run pre-sched2 passes.
  if (addPreSched2())
    printNoVerify("After PreSched2 passes");

  // Second pass scheduler.
  if (getOptLevel() != CodeGenOpt::None && !DisablePostRA) {
    PM.add(createPostRAScheduler(getOptLevel()));
    printNoVerify("After PostRAScheduler");
  }

  PM.add(createGCMachineCodeAnalysisPass());

  if (PrintGCInfo)
    PM.add(createGCInfoPrinter(dbgs()));

  if (getOptLevel() != CodeGenOpt::None && !DisableCodePlace) {
    if (EnableBlockPlacement) {
      // MachineBlockPlacement is an experimental pass which is disabled by
      // default currently. Eventually it should subsume CodePlacementOpt, so
      // when enabled, the other is disabled.
      PM.add(createMachineBlockPlacementPass());
      printNoVerify("After MachineBlockPlacement");
    } else {
      PM.add(createCodePlacementOptPass());
      printNoVerify("After CodePlacementOpt");
    }

    // Run a separate pass to collect block placement statistics.
    if (EnableBlockPlacementStats) {
      PM.add(createMachineBlockPlacementStatsPass());
      printNoVerify("After MachineBlockPlacementStats");
    }
  }

  if (addPreEmitPass())
    printNoVerify("After PreEmit passes");

  return false;
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
