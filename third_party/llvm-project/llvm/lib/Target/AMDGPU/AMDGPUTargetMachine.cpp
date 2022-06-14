//===-- AMDGPUTargetMachine.cpp - TargetMachine for hw codegen targets-----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// The AMDGPU target machine contains all of the hardware specific
/// information  needed to emit code for SI+ GPUs.
//
//===----------------------------------------------------------------------===//

#include "AMDGPUTargetMachine.h"
#include "AMDGPU.h"
#include "AMDGPUAliasAnalysis.h"
#include "AMDGPUExportClustering.h"
#include "AMDGPUMFMAIGroupLP.h"
#include "AMDGPUMacroFusion.h"
#include "AMDGPUTargetObjectFile.h"
#include "AMDGPUTargetTransformInfo.h"
#include "GCNIterativeScheduler.h"
#include "GCNSchedStrategy.h"
#include "R600.h"
#include "R600TargetMachine.h"
#include "SIMachineFunctionInfo.h"
#include "SIMachineScheduler.h"
#include "TargetInfo/AMDGPUTargetInfo.h"
#include "llvm/Analysis/CGSCCPassManager.h"
#include "llvm/CodeGen/GlobalISel/CSEInfo.h"
#include "llvm/CodeGen/GlobalISel/IRTranslator.h"
#include "llvm/CodeGen/GlobalISel/InstructionSelect.h"
#include "llvm/CodeGen/GlobalISel/Legalizer.h"
#include "llvm/CodeGen/GlobalISel/Localizer.h"
#include "llvm/CodeGen/GlobalISel/RegBankSelect.h"
#include "llvm/CodeGen/MIRParser/MIParser.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/RegAllocRegistry.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/InitializePasses.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/IPO/AlwaysInliner.h"
#include "llvm/Transforms/IPO/GlobalDCE.h"
#include "llvm/Transforms/IPO/Internalize.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Scalar/GVN.h"
#include "llvm/Transforms/Scalar/InferAddressSpaces.h"
#include "llvm/Transforms/Utils.h"
#include "llvm/Transforms/Utils/SimplifyLibCalls.h"
#include "llvm/Transforms/Vectorize.h"

using namespace llvm;
using namespace llvm::PatternMatch;

namespace {
class SGPRRegisterRegAlloc : public RegisterRegAllocBase<SGPRRegisterRegAlloc> {
public:
  SGPRRegisterRegAlloc(const char *N, const char *D, FunctionPassCtor C)
    : RegisterRegAllocBase(N, D, C) {}
};

class VGPRRegisterRegAlloc : public RegisterRegAllocBase<VGPRRegisterRegAlloc> {
public:
  VGPRRegisterRegAlloc(const char *N, const char *D, FunctionPassCtor C)
    : RegisterRegAllocBase(N, D, C) {}
};

static bool onlyAllocateSGPRs(const TargetRegisterInfo &TRI,
                              const TargetRegisterClass &RC) {
  return static_cast<const SIRegisterInfo &>(TRI).isSGPRClass(&RC);
}

static bool onlyAllocateVGPRs(const TargetRegisterInfo &TRI,
                              const TargetRegisterClass &RC) {
  return !static_cast<const SIRegisterInfo &>(TRI).isSGPRClass(&RC);
}


/// -{sgpr|vgpr}-regalloc=... command line option.
static FunctionPass *useDefaultRegisterAllocator() { return nullptr; }

/// A dummy default pass factory indicates whether the register allocator is
/// overridden on the command line.
static llvm::once_flag InitializeDefaultSGPRRegisterAllocatorFlag;
static llvm::once_flag InitializeDefaultVGPRRegisterAllocatorFlag;

static SGPRRegisterRegAlloc
defaultSGPRRegAlloc("default",
                    "pick SGPR register allocator based on -O option",
                    useDefaultRegisterAllocator);

static cl::opt<SGPRRegisterRegAlloc::FunctionPassCtor, false,
               RegisterPassParser<SGPRRegisterRegAlloc>>
SGPRRegAlloc("sgpr-regalloc", cl::Hidden, cl::init(&useDefaultRegisterAllocator),
             cl::desc("Register allocator to use for SGPRs"));

static cl::opt<VGPRRegisterRegAlloc::FunctionPassCtor, false,
               RegisterPassParser<VGPRRegisterRegAlloc>>
VGPRRegAlloc("vgpr-regalloc", cl::Hidden, cl::init(&useDefaultRegisterAllocator),
             cl::desc("Register allocator to use for VGPRs"));


static void initializeDefaultSGPRRegisterAllocatorOnce() {
  RegisterRegAlloc::FunctionPassCtor Ctor = SGPRRegisterRegAlloc::getDefault();

  if (!Ctor) {
    Ctor = SGPRRegAlloc;
    SGPRRegisterRegAlloc::setDefault(SGPRRegAlloc);
  }
}

static void initializeDefaultVGPRRegisterAllocatorOnce() {
  RegisterRegAlloc::FunctionPassCtor Ctor = VGPRRegisterRegAlloc::getDefault();

  if (!Ctor) {
    Ctor = VGPRRegAlloc;
    VGPRRegisterRegAlloc::setDefault(VGPRRegAlloc);
  }
}

static FunctionPass *createBasicSGPRRegisterAllocator() {
  return createBasicRegisterAllocator(onlyAllocateSGPRs);
}

static FunctionPass *createGreedySGPRRegisterAllocator() {
  return createGreedyRegisterAllocator(onlyAllocateSGPRs);
}

static FunctionPass *createFastSGPRRegisterAllocator() {
  return createFastRegisterAllocator(onlyAllocateSGPRs, false);
}

static FunctionPass *createBasicVGPRRegisterAllocator() {
  return createBasicRegisterAllocator(onlyAllocateVGPRs);
}

static FunctionPass *createGreedyVGPRRegisterAllocator() {
  return createGreedyRegisterAllocator(onlyAllocateVGPRs);
}

static FunctionPass *createFastVGPRRegisterAllocator() {
  return createFastRegisterAllocator(onlyAllocateVGPRs, true);
}

static SGPRRegisterRegAlloc basicRegAllocSGPR(
  "basic", "basic register allocator", createBasicSGPRRegisterAllocator);
static SGPRRegisterRegAlloc greedyRegAllocSGPR(
  "greedy", "greedy register allocator", createGreedySGPRRegisterAllocator);

static SGPRRegisterRegAlloc fastRegAllocSGPR(
  "fast", "fast register allocator", createFastSGPRRegisterAllocator);


static VGPRRegisterRegAlloc basicRegAllocVGPR(
  "basic", "basic register allocator", createBasicVGPRRegisterAllocator);
static VGPRRegisterRegAlloc greedyRegAllocVGPR(
  "greedy", "greedy register allocator", createGreedyVGPRRegisterAllocator);

static VGPRRegisterRegAlloc fastRegAllocVGPR(
  "fast", "fast register allocator", createFastVGPRRegisterAllocator);
}

static cl::opt<bool> EnableSROA(
  "amdgpu-sroa",
  cl::desc("Run SROA after promote alloca pass"),
  cl::ReallyHidden,
  cl::init(true));

static cl::opt<bool>
EnableEarlyIfConversion("amdgpu-early-ifcvt", cl::Hidden,
                        cl::desc("Run early if-conversion"),
                        cl::init(false));

static cl::opt<bool>
OptExecMaskPreRA("amdgpu-opt-exec-mask-pre-ra", cl::Hidden,
            cl::desc("Run pre-RA exec mask optimizations"),
            cl::init(true));

// Option to disable vectorizer for tests.
static cl::opt<bool> EnableLoadStoreVectorizer(
  "amdgpu-load-store-vectorizer",
  cl::desc("Enable load store vectorizer"),
  cl::init(true),
  cl::Hidden);

// Option to control global loads scalarization
static cl::opt<bool> ScalarizeGlobal(
  "amdgpu-scalarize-global-loads",
  cl::desc("Enable global load scalarization"),
  cl::init(true),
  cl::Hidden);

// Option to run internalize pass.
static cl::opt<bool> InternalizeSymbols(
  "amdgpu-internalize-symbols",
  cl::desc("Enable elimination of non-kernel functions and unused globals"),
  cl::init(false),
  cl::Hidden);

// Option to inline all early.
static cl::opt<bool> EarlyInlineAll(
  "amdgpu-early-inline-all",
  cl::desc("Inline all functions early"),
  cl::init(false),
  cl::Hidden);

static cl::opt<bool> EnableSDWAPeephole(
  "amdgpu-sdwa-peephole",
  cl::desc("Enable SDWA peepholer"),
  cl::init(true));

static cl::opt<bool> EnableDPPCombine(
  "amdgpu-dpp-combine",
  cl::desc("Enable DPP combiner"),
  cl::init(true));

// Enable address space based alias analysis
static cl::opt<bool> EnableAMDGPUAliasAnalysis("enable-amdgpu-aa", cl::Hidden,
  cl::desc("Enable AMDGPU Alias Analysis"),
  cl::init(true));

// Option to run late CFG structurizer
static cl::opt<bool, true> LateCFGStructurize(
  "amdgpu-late-structurize",
  cl::desc("Enable late CFG structurization"),
  cl::location(AMDGPUTargetMachine::EnableLateStructurizeCFG),
  cl::Hidden);

// Enable lib calls simplifications
static cl::opt<bool> EnableLibCallSimplify(
  "amdgpu-simplify-libcall",
  cl::desc("Enable amdgpu library simplifications"),
  cl::init(true),
  cl::Hidden);

static cl::opt<bool> EnableLowerKernelArguments(
  "amdgpu-ir-lower-kernel-arguments",
  cl::desc("Lower kernel argument loads in IR pass"),
  cl::init(true),
  cl::Hidden);

static cl::opt<bool> EnableRegReassign(
  "amdgpu-reassign-regs",
  cl::desc("Enable register reassign optimizations on gfx10+"),
  cl::init(true),
  cl::Hidden);

static cl::opt<bool> OptVGPRLiveRange(
    "amdgpu-opt-vgpr-liverange",
    cl::desc("Enable VGPR liverange optimizations for if-else structure"),
    cl::init(true), cl::Hidden);

// Enable atomic optimization
static cl::opt<bool> EnableAtomicOptimizations(
  "amdgpu-atomic-optimizations",
  cl::desc("Enable atomic optimizations"),
  cl::init(false),
  cl::Hidden);

// Enable Mode register optimization
static cl::opt<bool> EnableSIModeRegisterPass(
  "amdgpu-mode-register",
  cl::desc("Enable mode register pass"),
  cl::init(true),
  cl::Hidden);

// Option is used in lit tests to prevent deadcoding of patterns inspected.
static cl::opt<bool>
EnableDCEInRA("amdgpu-dce-in-ra",
    cl::init(true), cl::Hidden,
    cl::desc("Enable machine DCE inside regalloc"));

static cl::opt<bool> EnableSetWavePriority("amdgpu-set-wave-priority",
                                           cl::desc("Adjust wave priority"),
                                           cl::init(false), cl::Hidden);

static cl::opt<bool> EnableScalarIRPasses(
  "amdgpu-scalar-ir-passes",
  cl::desc("Enable scalar IR passes"),
  cl::init(true),
  cl::Hidden);

static cl::opt<bool> EnableStructurizerWorkarounds(
    "amdgpu-enable-structurizer-workarounds",
    cl::desc("Enable workarounds for the StructurizeCFG pass"), cl::init(true),
    cl::Hidden);

static cl::opt<bool> EnableLDSReplaceWithPointer(
    "amdgpu-enable-lds-replace-with-pointer",
    cl::desc("Enable LDS replace with pointer pass"), cl::init(false),
    cl::Hidden);

static cl::opt<bool, true> EnableLowerModuleLDS(
    "amdgpu-enable-lower-module-lds", cl::desc("Enable lower module lds pass"),
    cl::location(AMDGPUTargetMachine::EnableLowerModuleLDS), cl::init(true),
    cl::Hidden);

static cl::opt<bool> EnablePreRAOptimizations(
    "amdgpu-enable-pre-ra-optimizations",
    cl::desc("Enable Pre-RA optimizations pass"), cl::init(true),
    cl::Hidden);

static cl::opt<bool> EnablePromoteKernelArguments(
    "amdgpu-enable-promote-kernel-arguments",
    cl::desc("Enable promotion of flat kernel pointer arguments to global"),
    cl::Hidden, cl::init(true));

extern "C" LLVM_EXTERNAL_VISIBILITY void LLVMInitializeAMDGPUTarget() {
  // Register the target
  RegisterTargetMachine<R600TargetMachine> X(getTheAMDGPUTarget());
  RegisterTargetMachine<GCNTargetMachine> Y(getTheGCNTarget());

  PassRegistry *PR = PassRegistry::getPassRegistry();
  initializeR600ClauseMergePassPass(*PR);
  initializeR600ControlFlowFinalizerPass(*PR);
  initializeR600PacketizerPass(*PR);
  initializeR600ExpandSpecialInstrsPassPass(*PR);
  initializeR600VectorRegMergerPass(*PR);
  initializeGlobalISel(*PR);
  initializeAMDGPUDAGToDAGISelPass(*PR);
  initializeGCNDPPCombinePass(*PR);
  initializeSILowerI1CopiesPass(*PR);
  initializeSILowerSGPRSpillsPass(*PR);
  initializeSIFixSGPRCopiesPass(*PR);
  initializeSIFixVGPRCopiesPass(*PR);
  initializeSIFoldOperandsPass(*PR);
  initializeSIPeepholeSDWAPass(*PR);
  initializeSIShrinkInstructionsPass(*PR);
  initializeSIOptimizeExecMaskingPreRAPass(*PR);
  initializeSIOptimizeVGPRLiveRangePass(*PR);
  initializeSILoadStoreOptimizerPass(*PR);
  initializeAMDGPUCtorDtorLoweringPass(*PR);
  initializeAMDGPUAlwaysInlinePass(*PR);
  initializeAMDGPUAttributorPass(*PR);
  initializeAMDGPUAnnotateKernelFeaturesPass(*PR);
  initializeAMDGPUAnnotateUniformValuesPass(*PR);
  initializeAMDGPUArgumentUsageInfoPass(*PR);
  initializeAMDGPUAtomicOptimizerPass(*PR);
  initializeAMDGPULowerKernelArgumentsPass(*PR);
  initializeAMDGPUPromoteKernelArgumentsPass(*PR);
  initializeAMDGPULowerKernelAttributesPass(*PR);
  initializeAMDGPULowerIntrinsicsPass(*PR);
  initializeAMDGPUOpenCLEnqueuedBlockLoweringPass(*PR);
  initializeAMDGPUPostLegalizerCombinerPass(*PR);
  initializeAMDGPUPreLegalizerCombinerPass(*PR);
  initializeAMDGPURegBankCombinerPass(*PR);
  initializeAMDGPUPromoteAllocaPass(*PR);
  initializeAMDGPUPromoteAllocaToVectorPass(*PR);
  initializeAMDGPUCodeGenPreparePass(*PR);
  initializeAMDGPULateCodeGenPreparePass(*PR);
  initializeAMDGPUPropagateAttributesEarlyPass(*PR);
  initializeAMDGPUPropagateAttributesLatePass(*PR);
  initializeAMDGPUReplaceLDSUseWithPointerPass(*PR);
  initializeAMDGPULowerModuleLDSPass(*PR);
  initializeAMDGPURewriteOutArgumentsPass(*PR);
  initializeAMDGPUUnifyMetadataPass(*PR);
  initializeSIAnnotateControlFlowPass(*PR);
  initializeSIInsertHardClausesPass(*PR);
  initializeSIInsertWaitcntsPass(*PR);
  initializeSIModeRegisterPass(*PR);
  initializeSIWholeQuadModePass(*PR);
  initializeSILowerControlFlowPass(*PR);
  initializeSIPreEmitPeepholePass(*PR);
  initializeSILateBranchLoweringPass(*PR);
  initializeSIMemoryLegalizerPass(*PR);
  initializeSIOptimizeExecMaskingPass(*PR);
  initializeSIPreAllocateWWMRegsPass(*PR);
  initializeSIFormMemoryClausesPass(*PR);
  initializeSIPostRABundlerPass(*PR);
  initializeAMDGPUUnifyDivergentExitNodesPass(*PR);
  initializeAMDGPUAAWrapperPassPass(*PR);
  initializeAMDGPUExternalAAWrapperPass(*PR);
  initializeAMDGPUUseNativeCallsPass(*PR);
  initializeAMDGPUSimplifyLibCallsPass(*PR);
  initializeAMDGPUPrintfRuntimeBindingPass(*PR);
  initializeAMDGPUResourceUsageAnalysisPass(*PR);
  initializeGCNNSAReassignPass(*PR);
  initializeGCNPreRAOptimizationsPass(*PR);
}

static std::unique_ptr<TargetLoweringObjectFile> createTLOF(const Triple &TT) {
  return std::make_unique<AMDGPUTargetObjectFile>();
}

static ScheduleDAGInstrs *createSIMachineScheduler(MachineSchedContext *C) {
  return new SIScheduleDAGMI(C);
}

static ScheduleDAGInstrs *
createGCNMaxOccupancyMachineScheduler(MachineSchedContext *C) {
  ScheduleDAGMILive *DAG =
    new GCNScheduleDAGMILive(C, std::make_unique<GCNMaxOccupancySchedStrategy>(C));
  DAG->addMutation(createLoadClusterDAGMutation(DAG->TII, DAG->TRI));
  DAG->addMutation(createMFMAIGroupLPDAGMutation());
  DAG->addMutation(createAMDGPUMacroFusionDAGMutation());
  DAG->addMutation(createAMDGPUExportClusteringDAGMutation());
  return DAG;
}

static ScheduleDAGInstrs *
createIterativeGCNMaxOccupancyMachineScheduler(MachineSchedContext *C) {
  auto DAG = new GCNIterativeScheduler(C,
    GCNIterativeScheduler::SCHEDULE_LEGACYMAXOCCUPANCY);
  DAG->addMutation(createLoadClusterDAGMutation(DAG->TII, DAG->TRI));
  return DAG;
}

static ScheduleDAGInstrs *createMinRegScheduler(MachineSchedContext *C) {
  return new GCNIterativeScheduler(C,
    GCNIterativeScheduler::SCHEDULE_MINREGFORCED);
}

static ScheduleDAGInstrs *
createIterativeILPMachineScheduler(MachineSchedContext *C) {
  auto DAG = new GCNIterativeScheduler(C,
    GCNIterativeScheduler::SCHEDULE_ILP);
  DAG->addMutation(createLoadClusterDAGMutation(DAG->TII, DAG->TRI));
  DAG->addMutation(createAMDGPUMacroFusionDAGMutation());
  return DAG;
}

static MachineSchedRegistry
SISchedRegistry("si", "Run SI's custom scheduler",
                createSIMachineScheduler);

static MachineSchedRegistry
GCNMaxOccupancySchedRegistry("gcn-max-occupancy",
                             "Run GCN scheduler to maximize occupancy",
                             createGCNMaxOccupancyMachineScheduler);

static MachineSchedRegistry
IterativeGCNMaxOccupancySchedRegistry("gcn-max-occupancy-experimental",
  "Run GCN scheduler to maximize occupancy (experimental)",
  createIterativeGCNMaxOccupancyMachineScheduler);

static MachineSchedRegistry
GCNMinRegSchedRegistry("gcn-minreg",
  "Run GCN iterative scheduler for minimal register usage (experimental)",
  createMinRegScheduler);

static MachineSchedRegistry
GCNILPSchedRegistry("gcn-ilp",
  "Run GCN iterative scheduler for ILP scheduling (experimental)",
  createIterativeILPMachineScheduler);

static StringRef computeDataLayout(const Triple &TT) {
  if (TT.getArch() == Triple::r600) {
    // 32-bit pointers.
    return "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128"
           "-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1";
  }

  // 32-bit private, local, and region pointers. 64-bit global, constant and
  // flat, non-integral buffer fat pointers.
  return "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32"
         "-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128"
         "-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1"
         "-ni:7";
}

LLVM_READNONE
static StringRef getGPUOrDefault(const Triple &TT, StringRef GPU) {
  if (!GPU.empty())
    return GPU;

  // Need to default to a target with flat support for HSA.
  if (TT.getArch() == Triple::amdgcn)
    return TT.getOS() == Triple::AMDHSA ? "generic-hsa" : "generic";

  return "r600";
}

static Reloc::Model getEffectiveRelocModel(Optional<Reloc::Model> RM) {
  // The AMDGPU toolchain only supports generating shared objects, so we
  // must always use PIC.
  return Reloc::PIC_;
}

AMDGPUTargetMachine::AMDGPUTargetMachine(const Target &T, const Triple &TT,
                                         StringRef CPU, StringRef FS,
                                         TargetOptions Options,
                                         Optional<Reloc::Model> RM,
                                         Optional<CodeModel::Model> CM,
                                         CodeGenOpt::Level OptLevel)
    : LLVMTargetMachine(T, computeDataLayout(TT), TT, getGPUOrDefault(TT, CPU),
                        FS, Options, getEffectiveRelocModel(RM),
                        getEffectiveCodeModel(CM, CodeModel::Small), OptLevel),
      TLOF(createTLOF(getTargetTriple())) {
  initAsmInfo();
  if (TT.getArch() == Triple::amdgcn) {
    if (getMCSubtargetInfo()->checkFeatures("+wavefrontsize64"))
      MRI.reset(llvm::createGCNMCRegisterInfo(AMDGPUDwarfFlavour::Wave64));
    else if (getMCSubtargetInfo()->checkFeatures("+wavefrontsize32"))
      MRI.reset(llvm::createGCNMCRegisterInfo(AMDGPUDwarfFlavour::Wave32));
  }
}

bool AMDGPUTargetMachine::EnableLateStructurizeCFG = false;
bool AMDGPUTargetMachine::EnableFunctionCalls = false;
bool AMDGPUTargetMachine::EnableLowerModuleLDS = true;

AMDGPUTargetMachine::~AMDGPUTargetMachine() = default;

StringRef AMDGPUTargetMachine::getGPUName(const Function &F) const {
  Attribute GPUAttr = F.getFnAttribute("target-cpu");
  return GPUAttr.isValid() ? GPUAttr.getValueAsString() : getTargetCPU();
}

StringRef AMDGPUTargetMachine::getFeatureString(const Function &F) const {
  Attribute FSAttr = F.getFnAttribute("target-features");

  return FSAttr.isValid() ? FSAttr.getValueAsString()
                          : getTargetFeatureString();
}

/// Predicate for Internalize pass.
static bool mustPreserveGV(const GlobalValue &GV) {
  if (const Function *F = dyn_cast<Function>(&GV))
    return F->isDeclaration() || F->getName().startswith("__asan_") ||
           F->getName().startswith("__sanitizer_") ||
           AMDGPU::isEntryFunctionCC(F->getCallingConv());

  GV.removeDeadConstantUsers();
  return !GV.use_empty();
}

void AMDGPUTargetMachine::adjustPassManager(PassManagerBuilder &Builder) {
  Builder.DivergentTarget = true;

  bool EnableOpt = getOptLevel() > CodeGenOpt::None;
  bool Internalize = InternalizeSymbols;
  bool EarlyInline = EarlyInlineAll && EnableOpt && !EnableFunctionCalls;
  bool AMDGPUAA = EnableAMDGPUAliasAnalysis && EnableOpt;
  bool LibCallSimplify = EnableLibCallSimplify && EnableOpt;
  bool PromoteKernelArguments =
      EnablePromoteKernelArguments && getOptLevel() > CodeGenOpt::Less;

  if (EnableFunctionCalls) {
    delete Builder.Inliner;
    Builder.Inliner = createFunctionInliningPass();
  }

  Builder.addExtension(
    PassManagerBuilder::EP_ModuleOptimizerEarly,
    [Internalize, EarlyInline, AMDGPUAA, this](const PassManagerBuilder &,
                                               legacy::PassManagerBase &PM) {
      if (AMDGPUAA) {
        PM.add(createAMDGPUAAWrapperPass());
        PM.add(createAMDGPUExternalAAWrapperPass());
      }
      PM.add(createAMDGPUUnifyMetadataPass());
      PM.add(createAMDGPUPrintfRuntimeBinding());
      if (Internalize)
        PM.add(createInternalizePass(mustPreserveGV));
      PM.add(createAMDGPUPropagateAttributesLatePass(this));
      if (Internalize)
        PM.add(createGlobalDCEPass());
      if (EarlyInline)
        PM.add(createAMDGPUAlwaysInlinePass(false));
  });

  Builder.addExtension(
    PassManagerBuilder::EP_EarlyAsPossible,
    [AMDGPUAA, LibCallSimplify, this](const PassManagerBuilder &,
                                      legacy::PassManagerBase &PM) {
      if (AMDGPUAA) {
        PM.add(createAMDGPUAAWrapperPass());
        PM.add(createAMDGPUExternalAAWrapperPass());
      }
      PM.add(llvm::createAMDGPUPropagateAttributesEarlyPass(this));
      PM.add(llvm::createAMDGPUUseNativeCallsPass());
      if (LibCallSimplify)
        PM.add(llvm::createAMDGPUSimplifyLibCallsPass(this));
  });

  Builder.addExtension(
    PassManagerBuilder::EP_CGSCCOptimizerLate,
    [EnableOpt, PromoteKernelArguments](const PassManagerBuilder &,
                                        legacy::PassManagerBase &PM) {
      // Add promote kernel arguments pass to the opt pipeline right before
      // infer address spaces which is needed to do actual address space
      // rewriting.
      if (PromoteKernelArguments)
        PM.add(createAMDGPUPromoteKernelArgumentsPass());

      // Add infer address spaces pass to the opt pipeline after inlining
      // but before SROA to increase SROA opportunities.
      PM.add(createInferAddressSpacesPass());

      // This should run after inlining to have any chance of doing anything,
      // and before other cleanup optimizations.
      PM.add(createAMDGPULowerKernelAttributesPass());

      // Promote alloca to vector before SROA and loop unroll. If we manage
      // to eliminate allocas before unroll we may choose to unroll less.
      if (EnableOpt)
        PM.add(createAMDGPUPromoteAllocaToVector());
  });
}

void AMDGPUTargetMachine::registerDefaultAliasAnalyses(AAManager &AAM) {
  AAM.registerFunctionAnalysis<AMDGPUAA>();
}

void AMDGPUTargetMachine::registerPassBuilderCallbacks(PassBuilder &PB) {
  PB.registerPipelineParsingCallback(
      [this](StringRef PassName, ModulePassManager &PM,
             ArrayRef<PassBuilder::PipelineElement>) {
        if (PassName == "amdgpu-propagate-attributes-late") {
          PM.addPass(AMDGPUPropagateAttributesLatePass(*this));
          return true;
        }
        if (PassName == "amdgpu-unify-metadata") {
          PM.addPass(AMDGPUUnifyMetadataPass());
          return true;
        }
        if (PassName == "amdgpu-printf-runtime-binding") {
          PM.addPass(AMDGPUPrintfRuntimeBindingPass());
          return true;
        }
        if (PassName == "amdgpu-always-inline") {
          PM.addPass(AMDGPUAlwaysInlinePass());
          return true;
        }
        if (PassName == "amdgpu-replace-lds-use-with-pointer") {
          PM.addPass(AMDGPUReplaceLDSUseWithPointerPass());
          return true;
        }
        if (PassName == "amdgpu-lower-module-lds") {
          PM.addPass(AMDGPULowerModuleLDSPass());
          return true;
        }
        return false;
      });
  PB.registerPipelineParsingCallback(
      [this](StringRef PassName, FunctionPassManager &PM,
             ArrayRef<PassBuilder::PipelineElement>) {
        if (PassName == "amdgpu-simplifylib") {
          PM.addPass(AMDGPUSimplifyLibCallsPass(*this));
          return true;
        }
        if (PassName == "amdgpu-usenative") {
          PM.addPass(AMDGPUUseNativeCallsPass());
          return true;
        }
        if (PassName == "amdgpu-promote-alloca") {
          PM.addPass(AMDGPUPromoteAllocaPass(*this));
          return true;
        }
        if (PassName == "amdgpu-promote-alloca-to-vector") {
          PM.addPass(AMDGPUPromoteAllocaToVectorPass(*this));
          return true;
        }
        if (PassName == "amdgpu-lower-kernel-attributes") {
          PM.addPass(AMDGPULowerKernelAttributesPass());
          return true;
        }
        if (PassName == "amdgpu-propagate-attributes-early") {
          PM.addPass(AMDGPUPropagateAttributesEarlyPass(*this));
          return true;
        }
        if (PassName == "amdgpu-promote-kernel-arguments") {
          PM.addPass(AMDGPUPromoteKernelArgumentsPass());
          return true;
        }
        return false;
      });

  PB.registerAnalysisRegistrationCallback([](FunctionAnalysisManager &FAM) {
    FAM.registerPass([&] { return AMDGPUAA(); });
  });

  PB.registerParseAACallback([](StringRef AAName, AAManager &AAM) {
    if (AAName == "amdgpu-aa") {
      AAM.registerFunctionAnalysis<AMDGPUAA>();
      return true;
    }
    return false;
  });

  PB.registerPipelineStartEPCallback(
      [this](ModulePassManager &PM, OptimizationLevel Level) {
        FunctionPassManager FPM;
        FPM.addPass(AMDGPUPropagateAttributesEarlyPass(*this));
        FPM.addPass(AMDGPUUseNativeCallsPass());
        if (EnableLibCallSimplify && Level != OptimizationLevel::O0)
          FPM.addPass(AMDGPUSimplifyLibCallsPass(*this));
        PM.addPass(createModuleToFunctionPassAdaptor(std::move(FPM)));
      });

  PB.registerPipelineEarlySimplificationEPCallback(
      [this](ModulePassManager &PM, OptimizationLevel Level) {
        if (Level == OptimizationLevel::O0)
          return;

        PM.addPass(AMDGPUUnifyMetadataPass());
        PM.addPass(AMDGPUPrintfRuntimeBindingPass());

        if (InternalizeSymbols) {
          PM.addPass(InternalizePass(mustPreserveGV));
        }
        PM.addPass(AMDGPUPropagateAttributesLatePass(*this));
        if (InternalizeSymbols) {
          PM.addPass(GlobalDCEPass());
        }
        if (EarlyInlineAll && !EnableFunctionCalls)
          PM.addPass(AMDGPUAlwaysInlinePass());
      });

  PB.registerCGSCCOptimizerLateEPCallback(
      [this](CGSCCPassManager &PM, OptimizationLevel Level) {
        if (Level == OptimizationLevel::O0)
          return;

        FunctionPassManager FPM;

        // Add promote kernel arguments pass to the opt pipeline right before
        // infer address spaces which is needed to do actual address space
        // rewriting.
        if (Level.getSpeedupLevel() > OptimizationLevel::O1.getSpeedupLevel() &&
            EnablePromoteKernelArguments)
          FPM.addPass(AMDGPUPromoteKernelArgumentsPass());

        // Add infer address spaces pass to the opt pipeline after inlining
        // but before SROA to increase SROA opportunities.
        FPM.addPass(InferAddressSpacesPass());

        // This should run after inlining to have any chance of doing
        // anything, and before other cleanup optimizations.
        FPM.addPass(AMDGPULowerKernelAttributesPass());

        if (Level != OptimizationLevel::O0) {
          // Promote alloca to vector before SROA and loop unroll. If we
          // manage to eliminate allocas before unroll we may choose to unroll
          // less.
          FPM.addPass(AMDGPUPromoteAllocaToVectorPass(*this));
        }

        PM.addPass(createCGSCCToFunctionPassAdaptor(std::move(FPM)));
      });
}

int64_t AMDGPUTargetMachine::getNullPointerValue(unsigned AddrSpace) {
  return (AddrSpace == AMDGPUAS::LOCAL_ADDRESS ||
          AddrSpace == AMDGPUAS::PRIVATE_ADDRESS ||
          AddrSpace == AMDGPUAS::REGION_ADDRESS)
             ? -1
             : 0;
}

bool AMDGPUTargetMachine::isNoopAddrSpaceCast(unsigned SrcAS,
                                              unsigned DestAS) const {
  return AMDGPU::isFlatGlobalAddrSpace(SrcAS) &&
         AMDGPU::isFlatGlobalAddrSpace(DestAS);
}

unsigned AMDGPUTargetMachine::getAssumedAddrSpace(const Value *V) const {
  const auto *LD = dyn_cast<LoadInst>(V);
  if (!LD)
    return AMDGPUAS::UNKNOWN_ADDRESS_SPACE;

  // It must be a generic pointer loaded.
  assert(V->getType()->isPointerTy() &&
         V->getType()->getPointerAddressSpace() == AMDGPUAS::FLAT_ADDRESS);

  const auto *Ptr = LD->getPointerOperand();
  if (Ptr->getType()->getPointerAddressSpace() != AMDGPUAS::CONSTANT_ADDRESS)
    return AMDGPUAS::UNKNOWN_ADDRESS_SPACE;
  // For a generic pointer loaded from the constant memory, it could be assumed
  // as a global pointer since the constant memory is only populated on the
  // host side. As implied by the offload programming model, only global
  // pointers could be referenced on the host side.
  return AMDGPUAS::GLOBAL_ADDRESS;
}

std::pair<const Value *, unsigned>
AMDGPUTargetMachine::getPredicatedAddrSpace(const Value *V) const {
  if (auto *II = dyn_cast<IntrinsicInst>(V)) {
    switch (II->getIntrinsicID()) {
    case Intrinsic::amdgcn_is_shared:
      return std::make_pair(II->getArgOperand(0), AMDGPUAS::LOCAL_ADDRESS);
    case Intrinsic::amdgcn_is_private:
      return std::make_pair(II->getArgOperand(0), AMDGPUAS::PRIVATE_ADDRESS);
    default:
      break;
    }
    return std::make_pair(nullptr, -1);
  }
  // Check the global pointer predication based on
  // (!is_share(p) && !is_private(p)). Note that logic 'and' is commutative and
  // the order of 'is_shared' and 'is_private' is not significant.
  Value *Ptr;
  if (match(
          const_cast<Value *>(V),
          m_c_And(m_Not(m_Intrinsic<Intrinsic::amdgcn_is_shared>(m_Value(Ptr))),
                  m_Not(m_Intrinsic<Intrinsic::amdgcn_is_private>(
                      m_Deferred(Ptr))))))
    return std::make_pair(Ptr, AMDGPUAS::GLOBAL_ADDRESS);

  return std::make_pair(nullptr, -1);
}

unsigned
AMDGPUTargetMachine::getAddressSpaceForPseudoSourceKind(unsigned Kind) const {
  switch (Kind) {
  case PseudoSourceValue::Stack:
  case PseudoSourceValue::FixedStack:
    return AMDGPUAS::PRIVATE_ADDRESS;
  case PseudoSourceValue::ConstantPool:
  case PseudoSourceValue::GOT:
  case PseudoSourceValue::JumpTable:
  case PseudoSourceValue::GlobalValueCallEntry:
  case PseudoSourceValue::ExternalSymbolCallEntry:
  case PseudoSourceValue::TargetCustom:
    return AMDGPUAS::CONSTANT_ADDRESS;
  }
  return AMDGPUAS::FLAT_ADDRESS;
}

//===----------------------------------------------------------------------===//
// GCN Target Machine (SI+)
//===----------------------------------------------------------------------===//

GCNTargetMachine::GCNTargetMachine(const Target &T, const Triple &TT,
                                   StringRef CPU, StringRef FS,
                                   TargetOptions Options,
                                   Optional<Reloc::Model> RM,
                                   Optional<CodeModel::Model> CM,
                                   CodeGenOpt::Level OL, bool JIT)
    : AMDGPUTargetMachine(T, TT, CPU, FS, Options, RM, CM, OL) {}

const TargetSubtargetInfo *
GCNTargetMachine::getSubtargetImpl(const Function &F) const {
  StringRef GPU = getGPUName(F);
  StringRef FS = getFeatureString(F);

  SmallString<128> SubtargetKey(GPU);
  SubtargetKey.append(FS);

  auto &I = SubtargetMap[SubtargetKey];
  if (!I) {
    // This needs to be done before we create a new subtarget since any
    // creation will depend on the TM and the code generation flags on the
    // function that reside in TargetOptions.
    resetTargetOptions(F);
    I = std::make_unique<GCNSubtarget>(TargetTriple, GPU, FS, *this);
  }

  I->setScalarizeGlobalBehavior(ScalarizeGlobal);

  return I.get();
}

TargetTransformInfo
GCNTargetMachine::getTargetTransformInfo(const Function &F) const {
  return TargetTransformInfo(GCNTTIImpl(this, F));
}

//===----------------------------------------------------------------------===//
// AMDGPU Pass Setup
//===----------------------------------------------------------------------===//

std::unique_ptr<CSEConfigBase> llvm::AMDGPUPassConfig::getCSEConfig() const {
  return getStandardCSEConfigForOpt(TM->getOptLevel());
}

namespace {

class GCNPassConfig final : public AMDGPUPassConfig {
public:
  GCNPassConfig(LLVMTargetMachine &TM, PassManagerBase &PM)
    : AMDGPUPassConfig(TM, PM) {
    // It is necessary to know the register usage of the entire call graph.  We
    // allow calls without EnableAMDGPUFunctionCalls if they are marked
    // noinline, so this is always required.
    setRequiresCodeGenSCCOrder(true);
    substitutePass(&PostRASchedulerID, &PostMachineSchedulerID);
  }

  GCNTargetMachine &getGCNTargetMachine() const {
    return getTM<GCNTargetMachine>();
  }

  ScheduleDAGInstrs *
  createMachineScheduler(MachineSchedContext *C) const override;

  ScheduleDAGInstrs *
  createPostMachineScheduler(MachineSchedContext *C) const override {
    ScheduleDAGMI *DAG = createGenericSchedPostRA(C);
    const GCNSubtarget &ST = C->MF->getSubtarget<GCNSubtarget>();
    DAG->addMutation(createLoadClusterDAGMutation(DAG->TII, DAG->TRI));
    DAG->addMutation(ST.createFillMFMAShadowMutation(DAG->TII));
    DAG->addMutation(createMFMAIGroupLPDAGMutation());
    return DAG;
  }

  bool addPreISel() override;
  void addMachineSSAOptimization() override;
  bool addILPOpts() override;
  bool addInstSelector() override;
  bool addIRTranslator() override;
  void addPreLegalizeMachineIR() override;
  bool addLegalizeMachineIR() override;
  void addPreRegBankSelect() override;
  bool addRegBankSelect() override;
  void addPreGlobalInstructionSelect() override;
  bool addGlobalInstructionSelect() override;
  void addFastRegAlloc() override;
  void addOptimizedRegAlloc() override;

  FunctionPass *createSGPRAllocPass(bool Optimized);
  FunctionPass *createVGPRAllocPass(bool Optimized);
  FunctionPass *createRegAllocPass(bool Optimized) override;

  bool addRegAssignAndRewriteFast() override;
  bool addRegAssignAndRewriteOptimized() override;

  void addPreRegAlloc() override;
  bool addPreRewrite() override;
  void addPostRegAlloc() override;
  void addPreSched2() override;
  void addPreEmitPass() override;
};

} // end anonymous namespace

AMDGPUPassConfig::AMDGPUPassConfig(LLVMTargetMachine &TM, PassManagerBase &PM)
    : TargetPassConfig(TM, PM) {
  // Exceptions and StackMaps are not supported, so these passes will never do
  // anything.
  disablePass(&StackMapLivenessID);
  disablePass(&FuncletLayoutID);
  // Garbage collection is not supported.
  disablePass(&GCLoweringID);
  disablePass(&ShadowStackGCLoweringID);
}

void AMDGPUPassConfig::addEarlyCSEOrGVNPass() {
  if (getOptLevel() == CodeGenOpt::Aggressive)
    addPass(createGVNPass());
  else
    addPass(createEarlyCSEPass());
}

void AMDGPUPassConfig::addStraightLineScalarOptimizationPasses() {
  addPass(createLICMPass());
  addPass(createSeparateConstOffsetFromGEPPass());
  addPass(createSpeculativeExecutionPass());
  // ReassociateGEPs exposes more opportunities for SLSR. See
  // the example in reassociate-geps-and-slsr.ll.
  addPass(createStraightLineStrengthReducePass());
  // SeparateConstOffsetFromGEP and SLSR creates common expressions which GVN or
  // EarlyCSE can reuse.
  addEarlyCSEOrGVNPass();
  // Run NaryReassociate after EarlyCSE/GVN to be more effective.
  addPass(createNaryReassociatePass());
  // NaryReassociate on GEPs creates redundant common expressions, so run
  // EarlyCSE after it.
  addPass(createEarlyCSEPass());
}

void AMDGPUPassConfig::addIRPasses() {
  const AMDGPUTargetMachine &TM = getAMDGPUTargetMachine();

  // There is no reason to run these.
  disablePass(&StackMapLivenessID);
  disablePass(&FuncletLayoutID);
  disablePass(&PatchableFunctionID);

  addPass(createAMDGPUPrintfRuntimeBinding());
  addPass(createAMDGPUCtorDtorLoweringPass());

  // A call to propagate attributes pass in the backend in case opt was not run.
  addPass(createAMDGPUPropagateAttributesEarlyPass(&TM));

  addPass(createAMDGPULowerIntrinsicsPass());

  // Function calls are not supported, so make sure we inline everything.
  addPass(createAMDGPUAlwaysInlinePass());
  addPass(createAlwaysInlinerLegacyPass());
  // We need to add the barrier noop pass, otherwise adding the function
  // inlining pass will cause all of the PassConfigs passes to be run
  // one function at a time, which means if we have a module with two
  // functions, then we will generate code for the first function
  // without ever running any passes on the second.
  addPass(createBarrierNoopPass());

  // Handle uses of OpenCL image2d_t, image3d_t and sampler_t arguments.
  if (TM.getTargetTriple().getArch() == Triple::r600)
    addPass(createR600OpenCLImageTypeLoweringPass());

  // Replace OpenCL enqueued block function pointers with global variables.
  addPass(createAMDGPUOpenCLEnqueuedBlockLoweringPass());

  // Can increase LDS used by kernel so runs before PromoteAlloca
  if (EnableLowerModuleLDS) {
    // The pass "amdgpu-replace-lds-use-with-pointer" need to be run before the
    // pass "amdgpu-lower-module-lds", and also it required to be run only if
    // "amdgpu-lower-module-lds" pass is enabled.
    if (EnableLDSReplaceWithPointer)
      addPass(createAMDGPUReplaceLDSUseWithPointerPass());

    addPass(createAMDGPULowerModuleLDSPass());
  }

  if (TM.getOptLevel() > CodeGenOpt::None)
    addPass(createInferAddressSpacesPass());

  addPass(createAtomicExpandPass());

  if (TM.getOptLevel() > CodeGenOpt::None) {
    addPass(createAMDGPUPromoteAlloca());

    if (EnableSROA)
      addPass(createSROAPass());
    if (isPassEnabled(EnableScalarIRPasses))
      addStraightLineScalarOptimizationPasses();

    if (EnableAMDGPUAliasAnalysis) {
      addPass(createAMDGPUAAWrapperPass());
      addPass(createExternalAAWrapperPass([](Pass &P, Function &,
                                             AAResults &AAR) {
        if (auto *WrapperPass = P.getAnalysisIfAvailable<AMDGPUAAWrapperPass>())
          AAR.addAAResult(WrapperPass->getResult());
        }));
    }

    if (TM.getTargetTriple().getArch() == Triple::amdgcn) {
      // TODO: May want to move later or split into an early and late one.
      addPass(createAMDGPUCodeGenPreparePass());
    }
  }

  TargetPassConfig::addIRPasses();

  // EarlyCSE is not always strong enough to clean up what LSR produces. For
  // example, GVN can combine
  //
  //   %0 = add %a, %b
  //   %1 = add %b, %a
  //
  // and
  //
  //   %0 = shl nsw %a, 2
  //   %1 = shl %a, 2
  //
  // but EarlyCSE can do neither of them.
  if (isPassEnabled(EnableScalarIRPasses))
    addEarlyCSEOrGVNPass();
}

void AMDGPUPassConfig::addCodeGenPrepare() {
  if (TM->getTargetTriple().getArch() == Triple::amdgcn) {
    addPass(createAMDGPUAttributorPass());

    // FIXME: This pass adds 2 hacky attributes that can be replaced with an
    // analysis, and should be removed.
    addPass(createAMDGPUAnnotateKernelFeaturesPass());
  }

  if (TM->getTargetTriple().getArch() == Triple::amdgcn &&
      EnableLowerKernelArguments)
    addPass(createAMDGPULowerKernelArgumentsPass());

  TargetPassConfig::addCodeGenPrepare();

  if (isPassEnabled(EnableLoadStoreVectorizer))
    addPass(createLoadStoreVectorizerPass());

  // LowerSwitch pass may introduce unreachable blocks that can
  // cause unexpected behavior for subsequent passes. Placing it
  // here seems better that these blocks would get cleaned up by
  // UnreachableBlockElim inserted next in the pass flow.
  addPass(createLowerSwitchPass());
}

bool AMDGPUPassConfig::addPreISel() {
  if (TM->getOptLevel() > CodeGenOpt::None)
    addPass(createFlattenCFGPass());
  return false;
}

bool AMDGPUPassConfig::addInstSelector() {
  addPass(createAMDGPUISelDag(&getAMDGPUTargetMachine(), getOptLevel()));
  return false;
}

bool AMDGPUPassConfig::addGCPasses() {
  // Do nothing. GC is not supported.
  return false;
}

llvm::ScheduleDAGInstrs *
AMDGPUPassConfig::createMachineScheduler(MachineSchedContext *C) const {
  ScheduleDAGMILive *DAG = createGenericSchedLive(C);
  DAG->addMutation(createLoadClusterDAGMutation(DAG->TII, DAG->TRI));
  return DAG;
}

//===----------------------------------------------------------------------===//
// GCN Pass Setup
//===----------------------------------------------------------------------===//

ScheduleDAGInstrs *GCNPassConfig::createMachineScheduler(
  MachineSchedContext *C) const {
  const GCNSubtarget &ST = C->MF->getSubtarget<GCNSubtarget>();
  if (ST.enableSIScheduler())
    return createSIMachineScheduler(C);
  return createGCNMaxOccupancyMachineScheduler(C);
}

bool GCNPassConfig::addPreISel() {
  AMDGPUPassConfig::addPreISel();

  if (TM->getOptLevel() > CodeGenOpt::None)
    addPass(createAMDGPULateCodeGenPreparePass());

  if (isPassEnabled(EnableAtomicOptimizations, CodeGenOpt::Less)) {
    addPass(createAMDGPUAtomicOptimizerPass());
  }

  if (TM->getOptLevel() > CodeGenOpt::None)
    addPass(createSinkingPass());

  // Merge divergent exit nodes. StructurizeCFG won't recognize the multi-exit
  // regions formed by them.
  addPass(&AMDGPUUnifyDivergentExitNodesID);
  if (!LateCFGStructurize) {
    if (EnableStructurizerWorkarounds) {
      addPass(createFixIrreduciblePass());
      addPass(createUnifyLoopExitsPass());
    }
    addPass(createStructurizeCFGPass(false)); // true -> SkipUniformRegions
  }
  addPass(createAMDGPUAnnotateUniformValues());
  if (!LateCFGStructurize) {
    addPass(createSIAnnotateControlFlowPass());
  }
  addPass(createLCSSAPass());

  if (TM->getOptLevel() > CodeGenOpt::Less)
    addPass(&AMDGPUPerfHintAnalysisID);

  return false;
}

void GCNPassConfig::addMachineSSAOptimization() {
  TargetPassConfig::addMachineSSAOptimization();

  // We want to fold operands after PeepholeOptimizer has run (or as part of
  // it), because it will eliminate extra copies making it easier to fold the
  // real source operand. We want to eliminate dead instructions after, so that
  // we see fewer uses of the copies. We then need to clean up the dead
  // instructions leftover after the operands are folded as well.
  //
  // XXX - Can we get away without running DeadMachineInstructionElim again?
  addPass(&SIFoldOperandsID);
  if (EnableDPPCombine)
    addPass(&GCNDPPCombineID);
  addPass(&SILoadStoreOptimizerID);
  if (isPassEnabled(EnableSDWAPeephole)) {
    addPass(&SIPeepholeSDWAID);
    addPass(&EarlyMachineLICMID);
    addPass(&MachineCSEID);
    addPass(&SIFoldOperandsID);
  }
  addPass(&DeadMachineInstructionElimID);
  addPass(createSIShrinkInstructionsPass());
}

bool GCNPassConfig::addILPOpts() {
  if (EnableEarlyIfConversion)
    addPass(&EarlyIfConverterID);

  TargetPassConfig::addILPOpts();
  return false;
}

bool GCNPassConfig::addInstSelector() {
  AMDGPUPassConfig::addInstSelector();
  addPass(&SIFixSGPRCopiesID);
  addPass(createSILowerI1CopiesPass());
  return false;
}

bool GCNPassConfig::addIRTranslator() {
  addPass(new IRTranslator(getOptLevel()));
  return false;
}

void GCNPassConfig::addPreLegalizeMachineIR() {
  bool IsOptNone = getOptLevel() == CodeGenOpt::None;
  addPass(createAMDGPUPreLegalizeCombiner(IsOptNone));
  addPass(new Localizer());
}

bool GCNPassConfig::addLegalizeMachineIR() {
  addPass(new Legalizer());
  return false;
}

void GCNPassConfig::addPreRegBankSelect() {
  bool IsOptNone = getOptLevel() == CodeGenOpt::None;
  addPass(createAMDGPUPostLegalizeCombiner(IsOptNone));
}

bool GCNPassConfig::addRegBankSelect() {
  addPass(new RegBankSelect());
  return false;
}

void GCNPassConfig::addPreGlobalInstructionSelect() {
  bool IsOptNone = getOptLevel() == CodeGenOpt::None;
  addPass(createAMDGPURegBankCombiner(IsOptNone));
}

bool GCNPassConfig::addGlobalInstructionSelect() {
  addPass(new InstructionSelect(getOptLevel()));
  return false;
}

void GCNPassConfig::addPreRegAlloc() {
  if (LateCFGStructurize) {
    addPass(createAMDGPUMachineCFGStructurizerPass());
  }
}

void GCNPassConfig::addFastRegAlloc() {
  // FIXME: We have to disable the verifier here because of PHIElimination +
  // TwoAddressInstructions disabling it.

  // This must be run immediately after phi elimination and before
  // TwoAddressInstructions, otherwise the processing of the tied operand of
  // SI_ELSE will introduce a copy of the tied operand source after the else.
  insertPass(&PHIEliminationID, &SILowerControlFlowID);

  insertPass(&TwoAddressInstructionPassID, &SIWholeQuadModeID);
  insertPass(&TwoAddressInstructionPassID, &SIPreAllocateWWMRegsID);

  TargetPassConfig::addFastRegAlloc();
}

void GCNPassConfig::addOptimizedRegAlloc() {
  // Allow the scheduler to run before SIWholeQuadMode inserts exec manipulation
  // instructions that cause scheduling barriers.
  insertPass(&MachineSchedulerID, &SIWholeQuadModeID);
  insertPass(&MachineSchedulerID, &SIPreAllocateWWMRegsID);

  if (OptExecMaskPreRA)
    insertPass(&MachineSchedulerID, &SIOptimizeExecMaskingPreRAID);

  if (isPassEnabled(EnablePreRAOptimizations))
    insertPass(&RenameIndependentSubregsID, &GCNPreRAOptimizationsID);

  // This is not an essential optimization and it has a noticeable impact on
  // compilation time, so we only enable it from O2.
  if (TM->getOptLevel() > CodeGenOpt::Less)
    insertPass(&MachineSchedulerID, &SIFormMemoryClausesID);

  // FIXME: when an instruction has a Killed operand, and the instruction is
  // inside a bundle, seems only the BUNDLE instruction appears as the Kills of
  // the register in LiveVariables, this would trigger a failure in verifier,
  // we should fix it and enable the verifier.
  if (OptVGPRLiveRange)
    insertPass(&LiveVariablesID, &SIOptimizeVGPRLiveRangeID);
  // This must be run immediately after phi elimination and before
  // TwoAddressInstructions, otherwise the processing of the tied operand of
  // SI_ELSE will introduce a copy of the tied operand source after the else.
  insertPass(&PHIEliminationID, &SILowerControlFlowID);

  if (EnableDCEInRA)
    insertPass(&DetectDeadLanesID, &DeadMachineInstructionElimID);

  TargetPassConfig::addOptimizedRegAlloc();
}

bool GCNPassConfig::addPreRewrite() {
  if (EnableRegReassign)
    addPass(&GCNNSAReassignID);
  return true;
}

FunctionPass *GCNPassConfig::createSGPRAllocPass(bool Optimized) {
  // Initialize the global default.
  llvm::call_once(InitializeDefaultSGPRRegisterAllocatorFlag,
                  initializeDefaultSGPRRegisterAllocatorOnce);

  RegisterRegAlloc::FunctionPassCtor Ctor = SGPRRegisterRegAlloc::getDefault();
  if (Ctor != useDefaultRegisterAllocator)
    return Ctor();

  if (Optimized)
    return createGreedyRegisterAllocator(onlyAllocateSGPRs);

  return createFastRegisterAllocator(onlyAllocateSGPRs, false);
}

FunctionPass *GCNPassConfig::createVGPRAllocPass(bool Optimized) {
  // Initialize the global default.
  llvm::call_once(InitializeDefaultVGPRRegisterAllocatorFlag,
                  initializeDefaultVGPRRegisterAllocatorOnce);

  RegisterRegAlloc::FunctionPassCtor Ctor = VGPRRegisterRegAlloc::getDefault();
  if (Ctor != useDefaultRegisterAllocator)
    return Ctor();

  if (Optimized)
    return createGreedyVGPRRegisterAllocator();

  return createFastVGPRRegisterAllocator();
}

FunctionPass *GCNPassConfig::createRegAllocPass(bool Optimized) {
  llvm_unreachable("should not be used");
}

static const char RegAllocOptNotSupportedMessage[] =
  "-regalloc not supported with amdgcn. Use -sgpr-regalloc and -vgpr-regalloc";

bool GCNPassConfig::addRegAssignAndRewriteFast() {
  if (!usingDefaultRegAlloc())
    report_fatal_error(RegAllocOptNotSupportedMessage);

  addPass(createSGPRAllocPass(false));

  // Equivalent of PEI for SGPRs.
  addPass(&SILowerSGPRSpillsID);

  addPass(createVGPRAllocPass(false));
  return true;
}

bool GCNPassConfig::addRegAssignAndRewriteOptimized() {
  if (!usingDefaultRegAlloc())
    report_fatal_error(RegAllocOptNotSupportedMessage);

  addPass(createSGPRAllocPass(true));

  // Commit allocated register changes. This is mostly necessary because too
  // many things rely on the use lists of the physical registers, such as the
  // verifier. This is only necessary with allocators which use LiveIntervals,
  // since FastRegAlloc does the replacements itself.
  addPass(createVirtRegRewriter(false));

  // Equivalent of PEI for SGPRs.
  addPass(&SILowerSGPRSpillsID);

  addPass(createVGPRAllocPass(true));

  addPreRewrite();
  addPass(&VirtRegRewriterID);

  return true;
}

void GCNPassConfig::addPostRegAlloc() {
  addPass(&SIFixVGPRCopiesID);
  if (getOptLevel() > CodeGenOpt::None)
    addPass(&SIOptimizeExecMaskingID);
  TargetPassConfig::addPostRegAlloc();
}

void GCNPassConfig::addPreSched2() {
  if (TM->getOptLevel() > CodeGenOpt::None)
    addPass(createSIShrinkInstructionsPass());
  addPass(&SIPostRABundlerID);
}

void GCNPassConfig::addPreEmitPass() {
  addPass(createSIMemoryLegalizerPass());
  addPass(createSIInsertWaitcntsPass());

  addPass(createSIModeRegisterPass());

  if (getOptLevel() > CodeGenOpt::None)
    addPass(&SIInsertHardClausesID);

  addPass(&SILateBranchLoweringPassID);
  if (isPassEnabled(EnableSetWavePriority, CodeGenOpt::Less))
    addPass(createAMDGPUSetWavePriorityPass());
  if (getOptLevel() > CodeGenOpt::None)
    addPass(&SIPreEmitPeepholeID);
  // The hazard recognizer that runs as part of the post-ra scheduler does not
  // guarantee to be able handle all hazards correctly. This is because if there
  // are multiple scheduling regions in a basic block, the regions are scheduled
  // bottom up, so when we begin to schedule a region we don't know what
  // instructions were emitted directly before it.
  //
  // Here we add a stand-alone hazard recognizer pass which can handle all
  // cases.
  addPass(&PostRAHazardRecognizerID);
  addPass(&BranchRelaxationPassID);
}

TargetPassConfig *GCNTargetMachine::createPassConfig(PassManagerBase &PM) {
  return new GCNPassConfig(*this, PM);
}

yaml::MachineFunctionInfo *GCNTargetMachine::createDefaultFuncInfoYAML() const {
  return new yaml::SIMachineFunctionInfo();
}

yaml::MachineFunctionInfo *
GCNTargetMachine::convertFuncInfoToYAML(const MachineFunction &MF) const {
  const SIMachineFunctionInfo *MFI = MF.getInfo<SIMachineFunctionInfo>();
  return new yaml::SIMachineFunctionInfo(
      *MFI, *MF.getSubtarget().getRegisterInfo(), MF);
}

bool GCNTargetMachine::parseMachineFunctionInfo(
    const yaml::MachineFunctionInfo &MFI_, PerFunctionMIParsingState &PFS,
    SMDiagnostic &Error, SMRange &SourceRange) const {
  const yaml::SIMachineFunctionInfo &YamlMFI =
      static_cast<const yaml::SIMachineFunctionInfo &>(MFI_);
  MachineFunction &MF = PFS.MF;
  SIMachineFunctionInfo *MFI = MF.getInfo<SIMachineFunctionInfo>();

  if (MFI->initializeBaseYamlFields(YamlMFI, MF, PFS, Error, SourceRange))
    return true;

  if (MFI->Occupancy == 0) {
    // Fixup the subtarget dependent default value.
    const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();
    MFI->Occupancy = ST.computeOccupancy(MF.getFunction(), MFI->getLDSSize());
  }

  auto parseRegister = [&](const yaml::StringValue &RegName, Register &RegVal) {
    Register TempReg;
    if (parseNamedRegisterReference(PFS, TempReg, RegName.Value, Error)) {
      SourceRange = RegName.SourceRange;
      return true;
    }
    RegVal = TempReg;

    return false;
  };

  auto parseOptionalRegister = [&](const yaml::StringValue &RegName,
                                   Register &RegVal) {
    return !RegName.Value.empty() && parseRegister(RegName, RegVal);
  };

  if (parseOptionalRegister(YamlMFI.VGPRForAGPRCopy, MFI->VGPRForAGPRCopy))
    return true;

  auto diagnoseRegisterClass = [&](const yaml::StringValue &RegName) {
    // Create a diagnostic for a the register string literal.
    const MemoryBuffer &Buffer =
        *PFS.SM->getMemoryBuffer(PFS.SM->getMainFileID());
    Error = SMDiagnostic(*PFS.SM, SMLoc(), Buffer.getBufferIdentifier(), 1,
                         RegName.Value.size(), SourceMgr::DK_Error,
                         "incorrect register class for field", RegName.Value,
                         None, None);
    SourceRange = RegName.SourceRange;
    return true;
  };

  if (parseRegister(YamlMFI.ScratchRSrcReg, MFI->ScratchRSrcReg) ||
      parseRegister(YamlMFI.FrameOffsetReg, MFI->FrameOffsetReg) ||
      parseRegister(YamlMFI.StackPtrOffsetReg, MFI->StackPtrOffsetReg))
    return true;

  if (MFI->ScratchRSrcReg != AMDGPU::PRIVATE_RSRC_REG &&
      !AMDGPU::SGPR_128RegClass.contains(MFI->ScratchRSrcReg)) {
    return diagnoseRegisterClass(YamlMFI.ScratchRSrcReg);
  }

  if (MFI->FrameOffsetReg != AMDGPU::FP_REG &&
      !AMDGPU::SGPR_32RegClass.contains(MFI->FrameOffsetReg)) {
    return diagnoseRegisterClass(YamlMFI.FrameOffsetReg);
  }

  if (MFI->StackPtrOffsetReg != AMDGPU::SP_REG &&
      !AMDGPU::SGPR_32RegClass.contains(MFI->StackPtrOffsetReg)) {
    return diagnoseRegisterClass(YamlMFI.StackPtrOffsetReg);
  }

  for (const auto &YamlReg : YamlMFI.WWMReservedRegs) {
    Register ParsedReg;
    if (parseRegister(YamlReg, ParsedReg))
      return true;

    MFI->reserveWWMRegister(ParsedReg);
  }

  auto parseAndCheckArgument = [&](const Optional<yaml::SIArgument> &A,
                                   const TargetRegisterClass &RC,
                                   ArgDescriptor &Arg, unsigned UserSGPRs,
                                   unsigned SystemSGPRs) {
    // Skip parsing if it's not present.
    if (!A)
      return false;

    if (A->IsRegister) {
      Register Reg;
      if (parseNamedRegisterReference(PFS, Reg, A->RegisterName.Value, Error)) {
        SourceRange = A->RegisterName.SourceRange;
        return true;
      }
      if (!RC.contains(Reg))
        return diagnoseRegisterClass(A->RegisterName);
      Arg = ArgDescriptor::createRegister(Reg);
    } else
      Arg = ArgDescriptor::createStack(A->StackOffset);
    // Check and apply the optional mask.
    if (A->Mask)
      Arg = ArgDescriptor::createArg(Arg, A->Mask.getValue());

    MFI->NumUserSGPRs += UserSGPRs;
    MFI->NumSystemSGPRs += SystemSGPRs;
    return false;
  };

  if (YamlMFI.ArgInfo &&
      (parseAndCheckArgument(YamlMFI.ArgInfo->PrivateSegmentBuffer,
                             AMDGPU::SGPR_128RegClass,
                             MFI->ArgInfo.PrivateSegmentBuffer, 4, 0) ||
       parseAndCheckArgument(YamlMFI.ArgInfo->DispatchPtr,
                             AMDGPU::SReg_64RegClass, MFI->ArgInfo.DispatchPtr,
                             2, 0) ||
       parseAndCheckArgument(YamlMFI.ArgInfo->QueuePtr, AMDGPU::SReg_64RegClass,
                             MFI->ArgInfo.QueuePtr, 2, 0) ||
       parseAndCheckArgument(YamlMFI.ArgInfo->KernargSegmentPtr,
                             AMDGPU::SReg_64RegClass,
                             MFI->ArgInfo.KernargSegmentPtr, 2, 0) ||
       parseAndCheckArgument(YamlMFI.ArgInfo->DispatchID,
                             AMDGPU::SReg_64RegClass, MFI->ArgInfo.DispatchID,
                             2, 0) ||
       parseAndCheckArgument(YamlMFI.ArgInfo->FlatScratchInit,
                             AMDGPU::SReg_64RegClass,
                             MFI->ArgInfo.FlatScratchInit, 2, 0) ||
       parseAndCheckArgument(YamlMFI.ArgInfo->PrivateSegmentSize,
                             AMDGPU::SGPR_32RegClass,
                             MFI->ArgInfo.PrivateSegmentSize, 0, 0) ||
       parseAndCheckArgument(YamlMFI.ArgInfo->WorkGroupIDX,
                             AMDGPU::SGPR_32RegClass, MFI->ArgInfo.WorkGroupIDX,
                             0, 1) ||
       parseAndCheckArgument(YamlMFI.ArgInfo->WorkGroupIDY,
                             AMDGPU::SGPR_32RegClass, MFI->ArgInfo.WorkGroupIDY,
                             0, 1) ||
       parseAndCheckArgument(YamlMFI.ArgInfo->WorkGroupIDZ,
                             AMDGPU::SGPR_32RegClass, MFI->ArgInfo.WorkGroupIDZ,
                             0, 1) ||
       parseAndCheckArgument(YamlMFI.ArgInfo->WorkGroupInfo,
                             AMDGPU::SGPR_32RegClass,
                             MFI->ArgInfo.WorkGroupInfo, 0, 1) ||
       parseAndCheckArgument(YamlMFI.ArgInfo->PrivateSegmentWaveByteOffset,
                             AMDGPU::SGPR_32RegClass,
                             MFI->ArgInfo.PrivateSegmentWaveByteOffset, 0, 1) ||
       parseAndCheckArgument(YamlMFI.ArgInfo->ImplicitArgPtr,
                             AMDGPU::SReg_64RegClass,
                             MFI->ArgInfo.ImplicitArgPtr, 0, 0) ||
       parseAndCheckArgument(YamlMFI.ArgInfo->ImplicitBufferPtr,
                             AMDGPU::SReg_64RegClass,
                             MFI->ArgInfo.ImplicitBufferPtr, 2, 0) ||
       parseAndCheckArgument(YamlMFI.ArgInfo->WorkItemIDX,
                             AMDGPU::VGPR_32RegClass,
                             MFI->ArgInfo.WorkItemIDX, 0, 0) ||
       parseAndCheckArgument(YamlMFI.ArgInfo->WorkItemIDY,
                             AMDGPU::VGPR_32RegClass,
                             MFI->ArgInfo.WorkItemIDY, 0, 0) ||
       parseAndCheckArgument(YamlMFI.ArgInfo->WorkItemIDZ,
                             AMDGPU::VGPR_32RegClass,
                             MFI->ArgInfo.WorkItemIDZ, 0, 0)))
    return true;

  MFI->Mode.IEEE = YamlMFI.Mode.IEEE;
  MFI->Mode.DX10Clamp = YamlMFI.Mode.DX10Clamp;
  MFI->Mode.FP32InputDenormals = YamlMFI.Mode.FP32InputDenormals;
  MFI->Mode.FP32OutputDenormals = YamlMFI.Mode.FP32OutputDenormals;
  MFI->Mode.FP64FP16InputDenormals = YamlMFI.Mode.FP64FP16InputDenormals;
  MFI->Mode.FP64FP16OutputDenormals = YamlMFI.Mode.FP64FP16OutputDenormals;

  return false;
}
