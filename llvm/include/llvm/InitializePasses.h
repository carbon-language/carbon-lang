//===- llvm/InitializePasses.h -------- Initialize All Passes ---*- C++ -*-===//
//
//                      The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the declarations for the pass initialization routines
// for the entire LLVM project.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_INITIALIZEPASSES_H
#define LLVM_INITIALIZEPASSES_H

namespace llvm {

class PassRegistry;

/// initializeCore - Initialize all passes linked into the 
/// TransformUtils library.
void initializeCore(PassRegistry&);

/// initializeTransformUtils - Initialize all passes linked into the 
/// TransformUtils library.
void initializeTransformUtils(PassRegistry&);

/// initializeScalarOpts - Initialize all passes linked into the 
/// ScalarOpts library.
void initializeScalarOpts(PassRegistry&);

/// initializeInstCombine - Initialize all passes linked into the 
/// ScalarOpts library.
void initializeInstCombine(PassRegistry&);

/// initializeIPO - Initialize all passes linked into the IPO library.
void initializeIPO(PassRegistry&);

/// initializeInstrumentation - Initialize all passes linked into the
/// Instrumentation library.
void initializeInstrumentation(PassRegistry&);

/// initializeAnalysis - Initialize all passes linked into the Analysis library.
void initializeAnalysis(PassRegistry&);

/// initializeIPA - Initialize all passes linked into the IPA library.
void initializeIPA(PassRegistry&);

/// initializeCodeGen - Initialize all passes linked into the CodeGen library.
void initializeCodeGen(PassRegistry&);

/// initializeCodeGen - Initialize all passes linked into the CodeGen library.
void initializeTarget(PassRegistry&);

void initializeAAEvalPass(PassRegistry&);
void initializeADCEPass(PassRegistry&);
void initializeAliasAnalysisAnalysisGroup(PassRegistry&);
void initializeAliasAnalysisCounterPass(PassRegistry&);
void initializeAliasDebuggerPass(PassRegistry&);
void initializeAliasSetPrinterPass(PassRegistry&);
void initializeAlwaysInlinerPass(PassRegistry&);
void initializeArgPromotionPass(PassRegistry&);
void initializeBasicAliasAnalysisPass(PassRegistry&);
void initializeBasicCallGraphPass(PassRegistry&);
void initializeBlockExtractorPassPass(PassRegistry&);
void initializeBlockPlacementPass(PassRegistry&);
void initializeBreakCriticalEdgesPass(PassRegistry&);
void initializeCFGOnlyPrinterPass(PassRegistry&);
void initializeCFGOnlyViewerPass(PassRegistry&);
void initializeCFGPrinterPass(PassRegistry&);
void initializeCFGSimplifyPassPass(PassRegistry&);
void initializeCFGViewerPass(PassRegistry&);
void initializeCalculateSpillWeightsPass(PassRegistry&);
void initializeCallGraphAnalysisGroup(PassRegistry&);
void initializeCodeGenPreparePass(PassRegistry&);
void initializeConstantMergePass(PassRegistry&);
void initializeConstantPropagationPass(PassRegistry&);
void initializeCorrelatedValuePropagationPass(PassRegistry&);
void initializeDAEPass(PassRegistry&);
void initializeDAHPass(PassRegistry&);
void initializeDCEPass(PassRegistry&);
void initializeDSEPass(PassRegistry&);
void initializeDTEPass(PassRegistry&);
void initializeDeadInstEliminationPass(PassRegistry&);
void initializeDeadMachineInstructionElimPass(PassRegistry&);
void initializeDomOnlyPrinterPass(PassRegistry&);
void initializeDomOnlyViewerPass(PassRegistry&);
void initializeDomPrinterPass(PassRegistry&);
void initializeDomViewerPass(PassRegistry&);
void initializeDominanceFrontierPass(PassRegistry&);
void initializeDominatorTreePass(PassRegistry&);
void initializeEdgeProfilerPass(PassRegistry&);
void initializeExpandPseudosPass(PassRegistry&);
void initializeFindUsedTypesPass(PassRegistry&);
void initializeFunctionAttrsPass(PassRegistry&);
void initializeGCModuleInfoPass(PassRegistry&);
void initializeGEPSplitterPass(PassRegistry&);
void initializeGVNPass(PassRegistry&);
void initializeGlobalDCEPass(PassRegistry&);
void initializeGlobalOptPass(PassRegistry&);
void initializeGlobalsModRefPass(PassRegistry&);
void initializeIPCPPass(PassRegistry&);
void initializeIPSCCPPass(PassRegistry&);
void initializeIVUsersPass(PassRegistry&);
void initializeIfConverterPass(PassRegistry&);
void initializeIndVarSimplifyPass(PassRegistry&);
void initializeInstCombinerPass(PassRegistry&);
void initializeInstCountPass(PassRegistry&);
void initializeInstNamerPass(PassRegistry&);
void initializeInternalizePassPass(PassRegistry&);
void initializeIntervalPartitionPass(PassRegistry&);
void initializeJumpThreadingPass(PassRegistry&);
void initializeLCSSAPass(PassRegistry&);
void initializeLICMPass(PassRegistry&);
void initializeLazyValueInfoPass(PassRegistry&);
void initializeLibCallAliasAnalysisPass(PassRegistry&);
void initializeLintPass(PassRegistry&);
void initializeLiveIntervalsPass(PassRegistry&);
void initializeLiveStacksPass(PassRegistry&);
void initializeLiveValuesPass(PassRegistry&);
void initializeLiveVariablesPass(PassRegistry&);
void initializeLoaderPassPass(PassRegistry&);
void initializeLoopDeletionPass(PassRegistry&);
void initializeLoopDependenceAnalysisPass(PassRegistry&);
void initializeLoopExtractorPass(PassRegistry&);
void initializeLoopInfoPass(PassRegistry&);
void initializeLoopRotatePass(PassRegistry&);
void initializeLoopSimplifyPass(PassRegistry&);
void initializeLoopSplitterPass(PassRegistry&);
void initializeLoopStrengthReducePass(PassRegistry&);
void initializeLoopUnrollPass(PassRegistry&);
void initializeLoopUnswitchPass(PassRegistry&);
void initializeLowerAtomicPass(PassRegistry&);
void initializeLowerIntrinsicsPass(PassRegistry&);
void initializeLowerInvokePass(PassRegistry&);
void initializeLowerSetJmpPass(PassRegistry&);
void initializeLowerSwitchPass(PassRegistry&);
void initializeMachineCSEPass(PassRegistry&);
void initializeMachineDominatorTreePass(PassRegistry&);
void initializeMachineLICMPass(PassRegistry&);
void initializeMachineLoopInfoPass(PassRegistry&);
void initializeMachineModuleInfoPass(PassRegistry&);
void initializeMachineSinkingPass(PassRegistry&);
void initializeMachineVerifierPassPass(PassRegistry&);
void initializeMemCpyOptPass(PassRegistry&);
void initializeMemDepPrinterPass(PassRegistry&);
void initializeMemoryDependenceAnalysisPass(PassRegistry&);
void initializeMergeFunctionsPass(PassRegistry&);
void initializeModuleDebugInfoPrinterPass(PassRegistry&);
void initializeNoAAPass(PassRegistry&);
void initializeNoProfileInfoPass(PassRegistry&);
void initializeOptimalEdgeProfilerPass(PassRegistry&);
void initializeOptimizePHIsPass(PassRegistry&);
void initializePEIPass(PassRegistry&);
void initializePHIEliminationPass(PassRegistry&);
void initializePartSpecPass(PassRegistry&);
void initializePartialInlinerPass(PassRegistry&);
void initializePeepholeOptimizerPass(PassRegistry&);
void initializePostDomOnlyPrinterPass(PassRegistry&);
void initializePostDomOnlyViewerPass(PassRegistry&);
void initializePostDomPrinterPass(PassRegistry&);
void initializePostDomViewerPass(PassRegistry&);
void initializePostDominanceFrontierPass(PassRegistry&);
void initializePostDominatorTreePass(PassRegistry&);
void initializePreAllocSplittingPass(PassRegistry&);
void initializePreVerifierPass(PassRegistry&);
void initializePrintDbgInfoPass(PassRegistry&);
void initializePrintFunctionPassPass(PassRegistry&);
void initializePrintModulePassPass(PassRegistry&);
void initializeProcessImplicitDefsPass(PassRegistry&);
void initializeProfileEstimatorPassPass(PassRegistry&);
void initializeProfileInfoAnalysisGroup(PassRegistry&);
void initializeProfileVerifierPassPass(PassRegistry&);
void initializePromotePassPass(PassRegistry&);
void initializePruneEHPass(PassRegistry&);
void initializeRALinScanPass(PassRegistry&);
void initializeReassociatePass(PassRegistry&);
void initializeRegToMemPass(PassRegistry&);
void initializeRegionInfoPass(PassRegistry&);
void initializeRegionOnlyPrinterPass(PassRegistry&);
void initializeRegionOnlyViewerPass(PassRegistry&);
void initializeRegionPrinterPass(PassRegistry&);
void initializeRegionViewerPass(PassRegistry&);
void initializeRegisterCoalescerAnalysisGroup(PassRegistry&);
void initializeRenderMachineFunctionPass(PassRegistry&);
void initializeSCCPPass(PassRegistry&);
void initializeSRETPromotionPass(PassRegistry&);
void initializeSROAPass(PassRegistry&);
void initializeScalarEvolutionAliasAnalysisPass(PassRegistry&);
void initializeScalarEvolutionPass(PassRegistry&);
void initializeSimpleInlinerPass(PassRegistry&);
void initializeSimpleRegisterCoalescingPass(PassRegistry&);
void initializeSimplifyHalfPowrLibCallsPass(PassRegistry&);
void initializeSimplifyLibCallsPass(PassRegistry&);
void initializeSingleLoopExtractorPass(PassRegistry&);
void initializeSinkingPass(PassRegistry&);
void initializeSlotIndexesPass(PassRegistry&);
void initializeStackProtectorPass(PassRegistry&);
void initializeStackSlotColoringPass(PassRegistry&);
void initializeStripDeadDebugInfoPass(PassRegistry&);
void initializeStripDeadPrototypesPassPass(PassRegistry&);
void initializeStripDebugDeclarePass(PassRegistry&);
void initializeStripNonDebugSymbolsPass(PassRegistry&);
void initializeStripSymbolsPass(PassRegistry&);
void initializeStrongPHIEliminationPass(PassRegistry&);
void initializeTailCallElimPass(PassRegistry&);
void initializeTailDupPass(PassRegistry&);
void initializeTargetDataPass(PassRegistry&);
void initializeTwoAddressInstructionPassPass(PassRegistry&);
void initializeTypeBasedAliasAnalysisPass(PassRegistry&);
void initializeUnifyFunctionExitNodesPass(PassRegistry&);
void initializeUnreachableBlockElimPass(PassRegistry&);
void initializeUnreachableMachineBlockElimPass(PassRegistry&);
void initializeVerifierPass(PassRegistry&);
void initializeVirtRegMapPass(PassRegistry&);

}

#endif
