//===- Parsing and selection of pass pipelines ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
///
/// This file provides the implementation of the PassBuilder based on our
/// static pass registry as well as related functionality. It also provides
/// helpers to aid in analyzing, debugging, and testing passes and pass
/// pipelines.
///
//===----------------------------------------------------------------------===//

#include "llvm/Passes/PassBuilder.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Analysis/AliasAnalysisEvaluator.h"
#include "llvm/Analysis/AliasSetTracker.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/BasicAliasAnalysis.h"
#include "llvm/Analysis/BlockFrequencyInfo.h"
#include "llvm/Analysis/BranchProbabilityInfo.h"
#include "llvm/Analysis/CFGPrinter.h"
#include "llvm/Analysis/CFLAndersAliasAnalysis.h"
#include "llvm/Analysis/CFLSteensAliasAnalysis.h"
#include "llvm/Analysis/CGSCCPassManager.h"
#include "llvm/Analysis/CallGraph.h"
#include "llvm/Analysis/CostModel.h"
#include "llvm/Analysis/CycleAnalysis.h"
#include "llvm/Analysis/DDG.h"
#include "llvm/Analysis/DDGPrinter.h"
#include "llvm/Analysis/Delinearization.h"
#include "llvm/Analysis/DemandedBits.h"
#include "llvm/Analysis/DependenceAnalysis.h"
#include "llvm/Analysis/DivergenceAnalysis.h"
#include "llvm/Analysis/DomPrinter.h"
#include "llvm/Analysis/DominanceFrontier.h"
#include "llvm/Analysis/FunctionPropertiesAnalysis.h"
#include "llvm/Analysis/GlobalsModRef.h"
#include "llvm/Analysis/IRSimilarityIdentifier.h"
#include "llvm/Analysis/IVUsers.h"
#include "llvm/Analysis/InlineAdvisor.h"
#include "llvm/Analysis/InlineSizeEstimatorAnalysis.h"
#include "llvm/Analysis/InstCount.h"
#include "llvm/Analysis/LazyCallGraph.h"
#include "llvm/Analysis/LazyValueInfo.h"
#include "llvm/Analysis/Lint.h"
#include "llvm/Analysis/LoopAccessAnalysis.h"
#include "llvm/Analysis/LoopCacheAnalysis.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/LoopNestAnalysis.h"
#include "llvm/Analysis/MemDerefPrinter.h"
#include "llvm/Analysis/MemoryDependenceAnalysis.h"
#include "llvm/Analysis/MemorySSA.h"
#include "llvm/Analysis/ModuleDebugInfoPrinter.h"
#include "llvm/Analysis/ModuleSummaryAnalysis.h"
#include "llvm/Analysis/MustExecute.h"
#include "llvm/Analysis/ObjCARCAliasAnalysis.h"
#include "llvm/Analysis/OptimizationRemarkEmitter.h"
#include "llvm/Analysis/PhiValues.h"
#include "llvm/Analysis/PostDominators.h"
#include "llvm/Analysis/ProfileSummaryInfo.h"
#include "llvm/Analysis/RegionInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionAliasAnalysis.h"
#include "llvm/Analysis/ScopedNoAliasAA.h"
#include "llvm/Analysis/StackLifetime.h"
#include "llvm/Analysis/StackSafetyAnalysis.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Analysis/TypeBasedAliasAnalysis.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/IRPrintingPasses.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/PrintPasses.h"
#include "llvm/IR/SafepointIRVerifier.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Regex.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/AggressiveInstCombine/AggressiveInstCombine.h"
#include "llvm/Transforms/Coroutines/CoroCleanup.h"
#include "llvm/Transforms/Coroutines/CoroEarly.h"
#include "llvm/Transforms/Coroutines/CoroElide.h"
#include "llvm/Transforms/Coroutines/CoroSplit.h"
#include "llvm/Transforms/IPO/AlwaysInliner.h"
#include "llvm/Transforms/IPO/Annotation2Metadata.h"
#include "llvm/Transforms/IPO/ArgumentPromotion.h"
#include "llvm/Transforms/IPO/Attributor.h"
#include "llvm/Transforms/IPO/BlockExtractor.h"
#include "llvm/Transforms/IPO/CalledValuePropagation.h"
#include "llvm/Transforms/IPO/ConstantMerge.h"
#include "llvm/Transforms/IPO/CrossDSOCFI.h"
#include "llvm/Transforms/IPO/DeadArgumentElimination.h"
#include "llvm/Transforms/IPO/ElimAvailExtern.h"
#include "llvm/Transforms/IPO/ForceFunctionAttrs.h"
#include "llvm/Transforms/IPO/FunctionAttrs.h"
#include "llvm/Transforms/IPO/FunctionImport.h"
#include "llvm/Transforms/IPO/GlobalDCE.h"
#include "llvm/Transforms/IPO/GlobalOpt.h"
#include "llvm/Transforms/IPO/GlobalSplit.h"
#include "llvm/Transforms/IPO/HotColdSplitting.h"
#include "llvm/Transforms/IPO/IROutliner.h"
#include "llvm/Transforms/IPO/InferFunctionAttrs.h"
#include "llvm/Transforms/IPO/Inliner.h"
#include "llvm/Transforms/IPO/Internalize.h"
#include "llvm/Transforms/IPO/LoopExtractor.h"
#include "llvm/Transforms/IPO/LowerTypeTests.h"
#include "llvm/Transforms/IPO/MergeFunctions.h"
#include "llvm/Transforms/IPO/ModuleInliner.h"
#include "llvm/Transforms/IPO/OpenMPOpt.h"
#include "llvm/Transforms/IPO/PartialInlining.h"
#include "llvm/Transforms/IPO/SCCP.h"
#include "llvm/Transforms/IPO/SampleProfile.h"
#include "llvm/Transforms/IPO/SampleProfileProbe.h"
#include "llvm/Transforms/IPO/StripDeadPrototypes.h"
#include "llvm/Transforms/IPO/StripSymbols.h"
#include "llvm/Transforms/IPO/SyntheticCountsPropagation.h"
#include "llvm/Transforms/IPO/WholeProgramDevirt.h"
#include "llvm/Transforms/InstCombine/InstCombine.h"
#include "llvm/Transforms/Instrumentation.h"
#include "llvm/Transforms/Instrumentation/AddressSanitizer.h"
#include "llvm/Transforms/Instrumentation/BoundsChecking.h"
#include "llvm/Transforms/Instrumentation/CGProfile.h"
#include "llvm/Transforms/Instrumentation/ControlHeightReduction.h"
#include "llvm/Transforms/Instrumentation/DataFlowSanitizer.h"
#include "llvm/Transforms/Instrumentation/GCOVProfiler.h"
#include "llvm/Transforms/Instrumentation/HWAddressSanitizer.h"
#include "llvm/Transforms/Instrumentation/InstrOrderFile.h"
#include "llvm/Transforms/Instrumentation/InstrProfiling.h"
#include "llvm/Transforms/Instrumentation/MemProfiler.h"
#include "llvm/Transforms/Instrumentation/MemorySanitizer.h"
#include "llvm/Transforms/Instrumentation/PGOInstrumentation.h"
#include "llvm/Transforms/Instrumentation/PoisonChecking.h"
#include "llvm/Transforms/Instrumentation/SanitizerCoverage.h"
#include "llvm/Transforms/Instrumentation/ThreadSanitizer.h"
#include "llvm/Transforms/ObjCARC.h"
#include "llvm/Transforms/Scalar/ADCE.h"
#include "llvm/Transforms/Scalar/AlignmentFromAssumptions.h"
#include "llvm/Transforms/Scalar/AnnotationRemarks.h"
#include "llvm/Transforms/Scalar/BDCE.h"
#include "llvm/Transforms/Scalar/CallSiteSplitting.h"
#include "llvm/Transforms/Scalar/ConstantHoisting.h"
#include "llvm/Transforms/Scalar/ConstraintElimination.h"
#include "llvm/Transforms/Scalar/CorrelatedValuePropagation.h"
#include "llvm/Transforms/Scalar/DCE.h"
#include "llvm/Transforms/Scalar/DFAJumpThreading.h"
#include "llvm/Transforms/Scalar/DeadStoreElimination.h"
#include "llvm/Transforms/Scalar/DivRemPairs.h"
#include "llvm/Transforms/Scalar/EarlyCSE.h"
#include "llvm/Transforms/Scalar/FlattenCFG.h"
#include "llvm/Transforms/Scalar/Float2Int.h"
#include "llvm/Transforms/Scalar/GVN.h"
#include "llvm/Transforms/Scalar/GuardWidening.h"
#include "llvm/Transforms/Scalar/IVUsersPrinter.h"
#include "llvm/Transforms/Scalar/IndVarSimplify.h"
#include "llvm/Transforms/Scalar/InductiveRangeCheckElimination.h"
#include "llvm/Transforms/Scalar/InferAddressSpaces.h"
#include "llvm/Transforms/Scalar/InstSimplifyPass.h"
#include "llvm/Transforms/Scalar/JumpThreading.h"
#include "llvm/Transforms/Scalar/LICM.h"
#include "llvm/Transforms/Scalar/LoopAccessAnalysisPrinter.h"
#include "llvm/Transforms/Scalar/LoopBoundSplit.h"
#include "llvm/Transforms/Scalar/LoopDataPrefetch.h"
#include "llvm/Transforms/Scalar/LoopDeletion.h"
#include "llvm/Transforms/Scalar/LoopDistribute.h"
#include "llvm/Transforms/Scalar/LoopFlatten.h"
#include "llvm/Transforms/Scalar/LoopFuse.h"
#include "llvm/Transforms/Scalar/LoopIdiomRecognize.h"
#include "llvm/Transforms/Scalar/LoopInstSimplify.h"
#include "llvm/Transforms/Scalar/LoopInterchange.h"
#include "llvm/Transforms/Scalar/LoopLoadElimination.h"
#include "llvm/Transforms/Scalar/LoopPassManager.h"
#include "llvm/Transforms/Scalar/LoopPredication.h"
#include "llvm/Transforms/Scalar/LoopReroll.h"
#include "llvm/Transforms/Scalar/LoopRotation.h"
#include "llvm/Transforms/Scalar/LoopSimplifyCFG.h"
#include "llvm/Transforms/Scalar/LoopSink.h"
#include "llvm/Transforms/Scalar/LoopStrengthReduce.h"
#include "llvm/Transforms/Scalar/LoopUnrollAndJamPass.h"
#include "llvm/Transforms/Scalar/LoopUnrollPass.h"
#include "llvm/Transforms/Scalar/LoopVersioningLICM.h"
#include "llvm/Transforms/Scalar/LowerAtomic.h"
#include "llvm/Transforms/Scalar/LowerConstantIntrinsics.h"
#include "llvm/Transforms/Scalar/LowerExpectIntrinsic.h"
#include "llvm/Transforms/Scalar/LowerGuardIntrinsic.h"
#include "llvm/Transforms/Scalar/LowerMatrixIntrinsics.h"
#include "llvm/Transforms/Scalar/LowerWidenableCondition.h"
#include "llvm/Transforms/Scalar/MakeGuardsExplicit.h"
#include "llvm/Transforms/Scalar/MemCpyOptimizer.h"
#include "llvm/Transforms/Scalar/MergeICmps.h"
#include "llvm/Transforms/Scalar/MergedLoadStoreMotion.h"
#include "llvm/Transforms/Scalar/NaryReassociate.h"
#include "llvm/Transforms/Scalar/NewGVN.h"
#include "llvm/Transforms/Scalar/PartiallyInlineLibCalls.h"
#include "llvm/Transforms/Scalar/Reassociate.h"
#include "llvm/Transforms/Scalar/Reg2Mem.h"
#include "llvm/Transforms/Scalar/RewriteStatepointsForGC.h"
#include "llvm/Transforms/Scalar/SCCP.h"
#include "llvm/Transforms/Scalar/SROA.h"
#include "llvm/Transforms/Scalar/ScalarizeMaskedMemIntrin.h"
#include "llvm/Transforms/Scalar/Scalarizer.h"
#include "llvm/Transforms/Scalar/SeparateConstOffsetFromGEP.h"
#include "llvm/Transforms/Scalar/SimpleLoopUnswitch.h"
#include "llvm/Transforms/Scalar/SimplifyCFG.h"
#include "llvm/Transforms/Scalar/Sink.h"
#include "llvm/Transforms/Scalar/SpeculativeExecution.h"
#include "llvm/Transforms/Scalar/StraightLineStrengthReduce.h"
#include "llvm/Transforms/Scalar/StructurizeCFG.h"
#include "llvm/Transforms/Scalar/TailRecursionElimination.h"
#include "llvm/Transforms/Scalar/WarnMissedTransforms.h"
#include "llvm/Transforms/Utils/AddDiscriminators.h"
#include "llvm/Transforms/Utils/AssumeBundleBuilder.h"
#include "llvm/Transforms/Utils/BreakCriticalEdges.h"
#include "llvm/Transforms/Utils/CanonicalizeAliases.h"
#include "llvm/Transforms/Utils/CanonicalizeFreezeInLoops.h"
#include "llvm/Transforms/Utils/EntryExitInstrumenter.h"
#include "llvm/Transforms/Utils/FixIrreducible.h"
#include "llvm/Transforms/Utils/HelloWorld.h"
#include "llvm/Transforms/Utils/InjectTLIMappings.h"
#include "llvm/Transforms/Utils/InstructionNamer.h"
#include "llvm/Transforms/Utils/LCSSA.h"
#include "llvm/Transforms/Utils/LibCallsShrinkWrap.h"
#include "llvm/Transforms/Utils/LoopSimplify.h"
#include "llvm/Transforms/Utils/LoopVersioning.h"
#include "llvm/Transforms/Utils/LowerInvoke.h"
#include "llvm/Transforms/Utils/LowerSwitch.h"
#include "llvm/Transforms/Utils/Mem2Reg.h"
#include "llvm/Transforms/Utils/MetaRenamer.h"
#include "llvm/Transforms/Utils/NameAnonGlobals.h"
#include "llvm/Transforms/Utils/RelLookupTableConverter.h"
#include "llvm/Transforms/Utils/StripGCRelocates.h"
#include "llvm/Transforms/Utils/StripNonLineTableDebugInfo.h"
#include "llvm/Transforms/Utils/SymbolRewriter.h"
#include "llvm/Transforms/Utils/UnifyFunctionExitNodes.h"
#include "llvm/Transforms/Utils/UnifyLoopExits.h"
#include "llvm/Transforms/Vectorize/LoadStoreVectorizer.h"
#include "llvm/Transforms/Vectorize/LoopVectorize.h"
#include "llvm/Transforms/Vectorize/SLPVectorizer.h"
#include "llvm/Transforms/Vectorize/VectorCombine.h"

using namespace llvm;

static const Regex DefaultAliasRegex(
    "^(default|thinlto-pre-link|thinlto|lto-pre-link|lto)<(O[0123sz])>$");

namespace llvm {
cl::opt<bool> PrintPipelinePasses(
    "print-pipeline-passes",
    cl::desc("Print a '-passes' compatible string describing the pipeline "
             "(best-effort only)."));
} // namespace llvm

namespace {

// The following passes/analyses have custom names, otherwise their name will
// include `(anonymous namespace)`. These are special since they are only for
// testing purposes and don't live in a header file.

/// No-op module pass which does nothing.
struct NoOpModulePass : PassInfoMixin<NoOpModulePass> {
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &) {
    return PreservedAnalyses::all();
  }

  static StringRef name() { return "NoOpModulePass"; }
};

/// No-op module analysis.
class NoOpModuleAnalysis : public AnalysisInfoMixin<NoOpModuleAnalysis> {
  friend AnalysisInfoMixin<NoOpModuleAnalysis>;
  static AnalysisKey Key;

public:
  struct Result {};
  Result run(Module &, ModuleAnalysisManager &) { return Result(); }
  static StringRef name() { return "NoOpModuleAnalysis"; }
};

/// No-op CGSCC pass which does nothing.
struct NoOpCGSCCPass : PassInfoMixin<NoOpCGSCCPass> {
  PreservedAnalyses run(LazyCallGraph::SCC &C, CGSCCAnalysisManager &,
                        LazyCallGraph &, CGSCCUpdateResult &UR) {
    return PreservedAnalyses::all();
  }
  static StringRef name() { return "NoOpCGSCCPass"; }
};

/// No-op CGSCC analysis.
class NoOpCGSCCAnalysis : public AnalysisInfoMixin<NoOpCGSCCAnalysis> {
  friend AnalysisInfoMixin<NoOpCGSCCAnalysis>;
  static AnalysisKey Key;

public:
  struct Result {};
  Result run(LazyCallGraph::SCC &, CGSCCAnalysisManager &, LazyCallGraph &G) {
    return Result();
  }
  static StringRef name() { return "NoOpCGSCCAnalysis"; }
};

/// No-op function pass which does nothing.
struct NoOpFunctionPass : PassInfoMixin<NoOpFunctionPass> {
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &) {
    return PreservedAnalyses::all();
  }
  static StringRef name() { return "NoOpFunctionPass"; }
};

/// No-op function analysis.
class NoOpFunctionAnalysis : public AnalysisInfoMixin<NoOpFunctionAnalysis> {
  friend AnalysisInfoMixin<NoOpFunctionAnalysis>;
  static AnalysisKey Key;

public:
  struct Result {};
  Result run(Function &, FunctionAnalysisManager &) { return Result(); }
  static StringRef name() { return "NoOpFunctionAnalysis"; }
};

/// No-op loop nest pass which does nothing.
struct NoOpLoopNestPass : PassInfoMixin<NoOpLoopNestPass> {
  PreservedAnalyses run(LoopNest &L, LoopAnalysisManager &,
                        LoopStandardAnalysisResults &, LPMUpdater &) {
    return PreservedAnalyses::all();
  }
  static StringRef name() { return "NoOpLoopNestPass"; }
};

/// No-op loop pass which does nothing.
struct NoOpLoopPass : PassInfoMixin<NoOpLoopPass> {
  PreservedAnalyses run(Loop &L, LoopAnalysisManager &,
                        LoopStandardAnalysisResults &, LPMUpdater &) {
    return PreservedAnalyses::all();
  }
  static StringRef name() { return "NoOpLoopPass"; }
};

/// No-op loop analysis.
class NoOpLoopAnalysis : public AnalysisInfoMixin<NoOpLoopAnalysis> {
  friend AnalysisInfoMixin<NoOpLoopAnalysis>;
  static AnalysisKey Key;

public:
  struct Result {};
  Result run(Loop &, LoopAnalysisManager &, LoopStandardAnalysisResults &) {
    return Result();
  }
  static StringRef name() { return "NoOpLoopAnalysis"; }
};

AnalysisKey NoOpModuleAnalysis::Key;
AnalysisKey NoOpCGSCCAnalysis::Key;
AnalysisKey NoOpFunctionAnalysis::Key;
AnalysisKey NoOpLoopAnalysis::Key;

/// Whether or not we should populate a PassInstrumentationCallbacks's class to
/// pass name map.
///
/// This is for optimization purposes so we don't populate it if we never use
/// it. This should be updated if new pass instrumentation wants to use the map.
/// We currently only use this for --print-before/after.
bool shouldPopulateClassToPassNames() {
  return PrintPipelinePasses || !printBeforePasses().empty() ||
         !printAfterPasses().empty();
}

} // namespace

PassBuilder::PassBuilder(TargetMachine *TM, PipelineTuningOptions PTO,
                         Optional<PGOOptions> PGOOpt,
                         PassInstrumentationCallbacks *PIC)
    : TM(TM), PTO(PTO), PGOOpt(PGOOpt), PIC(PIC) {
  if (TM)
    TM->registerPassBuilderCallbacks(*this);
  if (PIC && shouldPopulateClassToPassNames()) {
#define MODULE_PASS(NAME, CREATE_PASS)                                         \
  PIC->addClassToPassName(decltype(CREATE_PASS)::name(), NAME);
#define MODULE_PASS_WITH_PARAMS(NAME, CLASS, CREATE_PASS, PARSER, PARAMS)      \
  PIC->addClassToPassName(CLASS, NAME);
#define MODULE_ANALYSIS(NAME, CREATE_PASS)                                     \
  PIC->addClassToPassName(decltype(CREATE_PASS)::name(), NAME);
#define FUNCTION_PASS(NAME, CREATE_PASS)                                       \
  PIC->addClassToPassName(decltype(CREATE_PASS)::name(), NAME);
#define FUNCTION_PASS_WITH_PARAMS(NAME, CLASS, CREATE_PASS, PARSER, PARAMS)    \
  PIC->addClassToPassName(CLASS, NAME);
#define FUNCTION_ANALYSIS(NAME, CREATE_PASS)                                   \
  PIC->addClassToPassName(decltype(CREATE_PASS)::name(), NAME);
#define LOOPNEST_PASS(NAME, CREATE_PASS)                                       \
  PIC->addClassToPassName(decltype(CREATE_PASS)::name(), NAME);
#define LOOP_PASS(NAME, CREATE_PASS)                                           \
  PIC->addClassToPassName(decltype(CREATE_PASS)::name(), NAME);
#define LOOP_PASS_WITH_PARAMS(NAME, CLASS, CREATE_PASS, PARSER, PARAMS)        \
  PIC->addClassToPassName(CLASS, NAME);
#define LOOP_ANALYSIS(NAME, CREATE_PASS)                                       \
  PIC->addClassToPassName(decltype(CREATE_PASS)::name(), NAME);
#define CGSCC_PASS(NAME, CREATE_PASS)                                          \
  PIC->addClassToPassName(decltype(CREATE_PASS)::name(), NAME);
#define CGSCC_PASS_WITH_PARAMS(NAME, CLASS, CREATE_PASS, PARSER, PARAMS)       \
  PIC->addClassToPassName(CLASS, NAME);
#define CGSCC_ANALYSIS(NAME, CREATE_PASS)                                      \
  PIC->addClassToPassName(decltype(CREATE_PASS)::name(), NAME);
#include "PassRegistry.def"
  }
}

void PassBuilder::registerModuleAnalyses(ModuleAnalysisManager &MAM) {
#define MODULE_ANALYSIS(NAME, CREATE_PASS)                                     \
  MAM.registerPass([&] { return CREATE_PASS; });
#include "PassRegistry.def"

  for (auto &C : ModuleAnalysisRegistrationCallbacks)
    C(MAM);
}

void PassBuilder::registerCGSCCAnalyses(CGSCCAnalysisManager &CGAM) {
#define CGSCC_ANALYSIS(NAME, CREATE_PASS)                                      \
  CGAM.registerPass([&] { return CREATE_PASS; });
#include "PassRegistry.def"

  for (auto &C : CGSCCAnalysisRegistrationCallbacks)
    C(CGAM);
}

void PassBuilder::registerFunctionAnalyses(FunctionAnalysisManager &FAM) {
  // We almost always want the default alias analysis pipeline.
  // If a user wants a different one, they can register their own before calling
  // registerFunctionAnalyses().
  FAM.registerPass([&] { return buildDefaultAAPipeline(); });

#define FUNCTION_ANALYSIS(NAME, CREATE_PASS)                                   \
  FAM.registerPass([&] { return CREATE_PASS; });
#include "PassRegistry.def"

  for (auto &C : FunctionAnalysisRegistrationCallbacks)
    C(FAM);
}

void PassBuilder::registerLoopAnalyses(LoopAnalysisManager &LAM) {
#define LOOP_ANALYSIS(NAME, CREATE_PASS)                                       \
  LAM.registerPass([&] { return CREATE_PASS; });
#include "PassRegistry.def"

  for (auto &C : LoopAnalysisRegistrationCallbacks)
    C(LAM);
}

static Optional<int> parseRepeatPassName(StringRef Name) {
  if (!Name.consume_front("repeat<") || !Name.consume_back(">"))
    return None;
  int Count;
  if (Name.getAsInteger(0, Count) || Count <= 0)
    return None;
  return Count;
}

static Optional<int> parseDevirtPassName(StringRef Name) {
  if (!Name.consume_front("devirt<") || !Name.consume_back(">"))
    return None;
  int Count;
  if (Name.getAsInteger(0, Count) || Count < 0)
    return None;
  return Count;
}

static bool checkParametrizedPassName(StringRef Name, StringRef PassName) {
  if (!Name.consume_front(PassName))
    return false;
  // normal pass name w/o parameters == default parameters
  if (Name.empty())
    return true;
  return Name.startswith("<") && Name.endswith(">");
}

namespace {

/// This performs customized parsing of pass name with parameters.
///
/// We do not need parametrization of passes in textual pipeline very often,
/// yet on a rare occasion ability to specify parameters right there can be
/// useful.
///
/// \p Name - parameterized specification of a pass from a textual pipeline
/// is a string in a form of :
///      PassName '<' parameter-list '>'
///
/// Parameter list is being parsed by the parser callable argument, \p Parser,
/// It takes a string-ref of parameters and returns either StringError or a
/// parameter list in a form of a custom parameters type, all wrapped into
/// Expected<> template class.
///
template <typename ParametersParseCallableT>
auto parsePassParameters(ParametersParseCallableT &&Parser, StringRef Name,
                         StringRef PassName) -> decltype(Parser(StringRef{})) {
  using ParametersT = typename decltype(Parser(StringRef{}))::value_type;

  StringRef Params = Name;
  if (!Params.consume_front(PassName)) {
    assert(false &&
           "unable to strip pass name from parametrized pass specification");
  }
  if (!Params.empty() &&
      (!Params.consume_front("<") || !Params.consume_back(">"))) {
    assert(false && "invalid format for parametrized pass name");
  }

  Expected<ParametersT> Result = Parser(Params);
  assert((Result || Result.template errorIsA<StringError>()) &&
         "Pass parameter parser can only return StringErrors.");
  return Result;
}

/// Parser of parameters for LoopUnroll pass.
Expected<LoopUnrollOptions> parseLoopUnrollOptions(StringRef Params) {
  LoopUnrollOptions UnrollOpts;
  while (!Params.empty()) {
    StringRef ParamName;
    std::tie(ParamName, Params) = Params.split(';');
    int OptLevel = StringSwitch<int>(ParamName)
                       .Case("O0", 0)
                       .Case("O1", 1)
                       .Case("O2", 2)
                       .Case("O3", 3)
                       .Default(-1);
    if (OptLevel >= 0) {
      UnrollOpts.setOptLevel(OptLevel);
      continue;
    }
    if (ParamName.consume_front("full-unroll-max=")) {
      int Count;
      if (ParamName.getAsInteger(0, Count))
        return make_error<StringError>(
            formatv("invalid LoopUnrollPass parameter '{0}' ", ParamName).str(),
            inconvertibleErrorCode());
      UnrollOpts.setFullUnrollMaxCount(Count);
      continue;
    }

    bool Enable = !ParamName.consume_front("no-");
    if (ParamName == "partial") {
      UnrollOpts.setPartial(Enable);
    } else if (ParamName == "peeling") {
      UnrollOpts.setPeeling(Enable);
    } else if (ParamName == "profile-peeling") {
      UnrollOpts.setProfileBasedPeeling(Enable);
    } else if (ParamName == "runtime") {
      UnrollOpts.setRuntime(Enable);
    } else if (ParamName == "upperbound") {
      UnrollOpts.setUpperBound(Enable);
    } else {
      return make_error<StringError>(
          formatv("invalid LoopUnrollPass parameter '{0}' ", ParamName).str(),
          inconvertibleErrorCode());
    }
  }
  return UnrollOpts;
}

Expected<bool> parseSinglePassOption(StringRef Params, StringRef OptionName,
                                     StringRef PassName) {
  bool Result = false;
  while (!Params.empty()) {
    StringRef ParamName;
    std::tie(ParamName, Params) = Params.split(';');

    if (ParamName == OptionName) {
      Result = true;
    } else {
      return make_error<StringError>(
          formatv("invalid {1} pass parameter '{0}' ", ParamName, PassName)
              .str(),
          inconvertibleErrorCode());
    }
  }
  return Result;
}

Expected<bool> parseInlinerPassOptions(StringRef Params) {
  return parseSinglePassOption(Params, "only-mandatory", "InlinerPass");
}

Expected<bool> parseEarlyCSEPassOptions(StringRef Params) {
  return parseSinglePassOption(Params, "memssa", "EarlyCSE");
}

Expected<bool> parseEntryExitInstrumenterPassOptions(StringRef Params) {
  return parseSinglePassOption(Params, "post-inline", "EntryExitInstrumenter");
}

Expected<bool> parseLoopExtractorPassOptions(StringRef Params) {
  return parseSinglePassOption(Params, "single", "LoopExtractor");
}

Expected<bool> parseLowerMatrixIntrinsicsPassOptions(StringRef Params) {
  return parseSinglePassOption(Params, "minimal", "LowerMatrixIntrinsics");
}

Expected<AddressSanitizerOptions> parseASanPassOptions(StringRef Params) {
  AddressSanitizerOptions Result;
  while (!Params.empty()) {
    StringRef ParamName;
    std::tie(ParamName, Params) = Params.split(';');

    if (ParamName == "kernel") {
      Result.CompileKernel = true;
    } else {
      return make_error<StringError>(
          formatv("invalid AddressSanitizer pass parameter '{0}' ", ParamName)
              .str(),
          inconvertibleErrorCode());
    }
  }
  return Result;
}

Expected<HWAddressSanitizerOptions> parseHWASanPassOptions(StringRef Params) {
  HWAddressSanitizerOptions Result;
  while (!Params.empty()) {
    StringRef ParamName;
    std::tie(ParamName, Params) = Params.split(';');

    if (ParamName == "recover") {
      Result.Recover = true;
    } else if (ParamName == "kernel") {
      Result.CompileKernel = true;
    } else {
      return make_error<StringError>(
          formatv("invalid HWAddressSanitizer pass parameter '{0}' ", ParamName)
              .str(),
          inconvertibleErrorCode());
    }
  }
  return Result;
}

Expected<MemorySanitizerOptions> parseMSanPassOptions(StringRef Params) {
  MemorySanitizerOptions Result;
  while (!Params.empty()) {
    StringRef ParamName;
    std::tie(ParamName, Params) = Params.split(';');

    if (ParamName == "recover") {
      Result.Recover = true;
    } else if (ParamName == "kernel") {
      Result.Kernel = true;
    } else if (ParamName.consume_front("track-origins=")) {
      if (ParamName.getAsInteger(0, Result.TrackOrigins))
        return make_error<StringError>(
            formatv("invalid argument to MemorySanitizer pass track-origins "
                    "parameter: '{0}' ",
                    ParamName)
                .str(),
            inconvertibleErrorCode());
    } else if (ParamName == "eager-checks") {
      Result.EagerChecks = true;
    } else {
      return make_error<StringError>(
          formatv("invalid MemorySanitizer pass parameter '{0}' ", ParamName)
              .str(),
          inconvertibleErrorCode());
    }
  }
  return Result;
}

/// Parser of parameters for SimplifyCFG pass.
Expected<SimplifyCFGOptions> parseSimplifyCFGOptions(StringRef Params) {
  SimplifyCFGOptions Result;
  while (!Params.empty()) {
    StringRef ParamName;
    std::tie(ParamName, Params) = Params.split(';');

    bool Enable = !ParamName.consume_front("no-");
    if (ParamName == "forward-switch-cond") {
      Result.forwardSwitchCondToPhi(Enable);
    } else if (ParamName == "switch-to-lookup") {
      Result.convertSwitchToLookupTable(Enable);
    } else if (ParamName == "keep-loops") {
      Result.needCanonicalLoops(Enable);
    } else if (ParamName == "hoist-common-insts") {
      Result.hoistCommonInsts(Enable);
    } else if (ParamName == "sink-common-insts") {
      Result.sinkCommonInsts(Enable);
    } else if (Enable && ParamName.consume_front("bonus-inst-threshold=")) {
      APInt BonusInstThreshold;
      if (ParamName.getAsInteger(0, BonusInstThreshold))
        return make_error<StringError>(
            formatv("invalid argument to SimplifyCFG pass bonus-threshold "
                    "parameter: '{0}' ",
                    ParamName).str(),
            inconvertibleErrorCode());
      Result.bonusInstThreshold(BonusInstThreshold.getSExtValue());
    } else {
      return make_error<StringError>(
          formatv("invalid SimplifyCFG pass parameter '{0}' ", ParamName).str(),
          inconvertibleErrorCode());
    }
  }
  return Result;
}

/// Parser of parameters for LoopVectorize pass.
Expected<LoopVectorizeOptions> parseLoopVectorizeOptions(StringRef Params) {
  LoopVectorizeOptions Opts;
  while (!Params.empty()) {
    StringRef ParamName;
    std::tie(ParamName, Params) = Params.split(';');

    bool Enable = !ParamName.consume_front("no-");
    if (ParamName == "interleave-forced-only") {
      Opts.setInterleaveOnlyWhenForced(Enable);
    } else if (ParamName == "vectorize-forced-only") {
      Opts.setVectorizeOnlyWhenForced(Enable);
    } else {
      return make_error<StringError>(
          formatv("invalid LoopVectorize parameter '{0}' ", ParamName).str(),
          inconvertibleErrorCode());
    }
  }
  return Opts;
}

Expected<std::pair<bool, bool>> parseLoopUnswitchOptions(StringRef Params) {
  std::pair<bool, bool> Result = {false, true};
  while (!Params.empty()) {
    StringRef ParamName;
    std::tie(ParamName, Params) = Params.split(';');

    bool Enable = !ParamName.consume_front("no-");
    if (ParamName == "nontrivial") {
      Result.first = Enable;
    } else if (ParamName == "trivial") {
      Result.second = Enable;
    } else {
      return make_error<StringError>(
          formatv("invalid LoopUnswitch pass parameter '{0}' ", ParamName)
              .str(),
          inconvertibleErrorCode());
    }
  }
  return Result;
}

Expected<bool> parseMergedLoadStoreMotionOptions(StringRef Params) {
  bool Result = false;
  while (!Params.empty()) {
    StringRef ParamName;
    std::tie(ParamName, Params) = Params.split(';');

    bool Enable = !ParamName.consume_front("no-");
    if (ParamName == "split-footer-bb") {
      Result = Enable;
    } else {
      return make_error<StringError>(
          formatv("invalid MergedLoadStoreMotion pass parameter '{0}' ",
                  ParamName)
              .str(),
          inconvertibleErrorCode());
    }
  }
  return Result;
}

Expected<GVNOptions> parseGVNOptions(StringRef Params) {
  GVNOptions Result;
  while (!Params.empty()) {
    StringRef ParamName;
    std::tie(ParamName, Params) = Params.split(';');

    bool Enable = !ParamName.consume_front("no-");
    if (ParamName == "pre") {
      Result.setPRE(Enable);
    } else if (ParamName == "load-pre") {
      Result.setLoadPRE(Enable);
    } else if (ParamName == "split-backedge-load-pre") {
      Result.setLoadPRESplitBackedge(Enable);
    } else if (ParamName == "memdep") {
      Result.setMemDep(Enable);
    } else {
      return make_error<StringError>(
          formatv("invalid GVN pass parameter '{0}' ", ParamName).str(),
          inconvertibleErrorCode());
    }
  }
  return Result;
}

Expected<StackLifetime::LivenessType>
parseStackLifetimeOptions(StringRef Params) {
  StackLifetime::LivenessType Result = StackLifetime::LivenessType::May;
  while (!Params.empty()) {
    StringRef ParamName;
    std::tie(ParamName, Params) = Params.split(';');

    if (ParamName == "may") {
      Result = StackLifetime::LivenessType::May;
    } else if (ParamName == "must") {
      Result = StackLifetime::LivenessType::Must;
    } else {
      return make_error<StringError>(
          formatv("invalid StackLifetime parameter '{0}' ", ParamName).str(),
          inconvertibleErrorCode());
    }
  }
  return Result;
}

} // namespace

/// Tests whether a pass name starts with a valid prefix for a default pipeline
/// alias.
static bool startsWithDefaultPipelineAliasPrefix(StringRef Name) {
  return Name.startswith("default") || Name.startswith("thinlto") ||
         Name.startswith("lto");
}

/// Tests whether registered callbacks will accept a given pass name.
///
/// When parsing a pipeline text, the type of the outermost pipeline may be
/// omitted, in which case the type is automatically determined from the first
/// pass name in the text. This may be a name that is handled through one of the
/// callbacks. We check this through the oridinary parsing callbacks by setting
/// up a dummy PassManager in order to not force the client to also handle this
/// type of query.
template <typename PassManagerT, typename CallbacksT>
static bool callbacksAcceptPassName(StringRef Name, CallbacksT &Callbacks) {
  if (!Callbacks.empty()) {
    PassManagerT DummyPM;
    for (auto &CB : Callbacks)
      if (CB(Name, DummyPM, {}))
        return true;
  }
  return false;
}

template <typename CallbacksT>
static bool isModulePassName(StringRef Name, CallbacksT &Callbacks) {
  // Manually handle aliases for pre-configured pipeline fragments.
  if (startsWithDefaultPipelineAliasPrefix(Name))
    return DefaultAliasRegex.match(Name);

  // Explicitly handle pass manager names.
  if (Name == "module")
    return true;
  if (Name == "cgscc")
    return true;
  if (Name == "function" || Name == "function<eager-inv>")
    return true;

  // Explicitly handle custom-parsed pass names.
  if (parseRepeatPassName(Name))
    return true;

#define MODULE_PASS(NAME, CREATE_PASS)                                         \
  if (Name == NAME)                                                            \
    return true;
#define MODULE_PASS_WITH_PARAMS(NAME, CLASS, CREATE_PASS, PARSER, PARAMS)      \
  if (checkParametrizedPassName(Name, NAME))                                   \
    return true;
#define MODULE_ANALYSIS(NAME, CREATE_PASS)                                     \
  if (Name == "require<" NAME ">" || Name == "invalidate<" NAME ">")           \
    return true;
#include "PassRegistry.def"

  return callbacksAcceptPassName<ModulePassManager>(Name, Callbacks);
}

template <typename CallbacksT>
static bool isCGSCCPassName(StringRef Name, CallbacksT &Callbacks) {
  // Explicitly handle pass manager names.
  if (Name == "cgscc")
    return true;
  if (Name == "function" || Name == "function<eager-inv>")
    return true;

  // Explicitly handle custom-parsed pass names.
  if (parseRepeatPassName(Name))
    return true;
  if (parseDevirtPassName(Name))
    return true;

#define CGSCC_PASS(NAME, CREATE_PASS)                                          \
  if (Name == NAME)                                                            \
    return true;
#define CGSCC_PASS_WITH_PARAMS(NAME, CLASS, CREATE_PASS, PARSER, PARAMS)       \
  if (checkParametrizedPassName(Name, NAME))                                   \
    return true;
#define CGSCC_ANALYSIS(NAME, CREATE_PASS)                                      \
  if (Name == "require<" NAME ">" || Name == "invalidate<" NAME ">")           \
    return true;
#include "PassRegistry.def"

  return callbacksAcceptPassName<CGSCCPassManager>(Name, Callbacks);
}

template <typename CallbacksT>
static bool isFunctionPassName(StringRef Name, CallbacksT &Callbacks) {
  // Explicitly handle pass manager names.
  if (Name == "function" || Name == "function<eager-inv>")
    return true;
  if (Name == "loop" || Name == "loop-mssa")
    return true;

  // Explicitly handle custom-parsed pass names.
  if (parseRepeatPassName(Name))
    return true;

#define FUNCTION_PASS(NAME, CREATE_PASS)                                       \
  if (Name == NAME)                                                            \
    return true;
#define FUNCTION_PASS_WITH_PARAMS(NAME, CLASS, CREATE_PASS, PARSER, PARAMS)    \
  if (checkParametrizedPassName(Name, NAME))                                   \
    return true;
#define FUNCTION_ANALYSIS(NAME, CREATE_PASS)                                   \
  if (Name == "require<" NAME ">" || Name == "invalidate<" NAME ">")           \
    return true;
#include "PassRegistry.def"

  return callbacksAcceptPassName<FunctionPassManager>(Name, Callbacks);
}

template <typename CallbacksT>
static bool isLoopNestPassName(StringRef Name, CallbacksT &Callbacks,
                               bool &UseMemorySSA) {
  UseMemorySSA = false;

  // Explicitly handle custom-parsed pass names.
  if (parseRepeatPassName(Name))
    return true;

  if (Name == "lnicm") {
    UseMemorySSA = true;
    return true;
  }

#define LOOPNEST_PASS(NAME, CREATE_PASS)                                       \
  if (Name == NAME)                                                            \
    return true;
#include "PassRegistry.def"

  return callbacksAcceptPassName<LoopPassManager>(Name, Callbacks);
}

template <typename CallbacksT>
static bool isLoopPassName(StringRef Name, CallbacksT &Callbacks,
                           bool &UseMemorySSA) {
  UseMemorySSA = false;

  // Explicitly handle custom-parsed pass names.
  if (parseRepeatPassName(Name))
    return true;

  if (Name == "licm") {
    UseMemorySSA = true;
    return true;
  }

#define LOOP_PASS(NAME, CREATE_PASS)                                           \
  if (Name == NAME)                                                            \
    return true;
#define LOOP_PASS_WITH_PARAMS(NAME, CLASS, CREATE_PASS, PARSER, PARAMS)        \
  if (checkParametrizedPassName(Name, NAME))                                   \
    return true;
#define LOOP_ANALYSIS(NAME, CREATE_PASS)                                       \
  if (Name == "require<" NAME ">" || Name == "invalidate<" NAME ">")           \
    return true;
#include "PassRegistry.def"

  return callbacksAcceptPassName<LoopPassManager>(Name, Callbacks);
}

Optional<std::vector<PassBuilder::PipelineElement>>
PassBuilder::parsePipelineText(StringRef Text) {
  std::vector<PipelineElement> ResultPipeline;

  SmallVector<std::vector<PipelineElement> *, 4> PipelineStack = {
      &ResultPipeline};
  for (;;) {
    std::vector<PipelineElement> &Pipeline = *PipelineStack.back();
    size_t Pos = Text.find_first_of(",()");
    Pipeline.push_back({Text.substr(0, Pos), {}});

    // If we have a single terminating name, we're done.
    if (Pos == Text.npos)
      break;

    char Sep = Text[Pos];
    Text = Text.substr(Pos + 1);
    if (Sep == ',')
      // Just a name ending in a comma, continue.
      continue;

    if (Sep == '(') {
      // Push the inner pipeline onto the stack to continue processing.
      PipelineStack.push_back(&Pipeline.back().InnerPipeline);
      continue;
    }

    assert(Sep == ')' && "Bogus separator!");
    // When handling the close parenthesis, we greedily consume them to avoid
    // empty strings in the pipeline.
    do {
      // If we try to pop the outer pipeline we have unbalanced parentheses.
      if (PipelineStack.size() == 1)
        return None;

      PipelineStack.pop_back();
    } while (Text.consume_front(")"));

    // Check if we've finished parsing.
    if (Text.empty())
      break;

    // Otherwise, the end of an inner pipeline always has to be followed by
    // a comma, and then we can continue.
    if (!Text.consume_front(","))
      return None;
  }

  if (PipelineStack.size() > 1)
    // Unbalanced paretheses.
    return None;

  assert(PipelineStack.back() == &ResultPipeline &&
         "Wrong pipeline at the bottom of the stack!");
  return {std::move(ResultPipeline)};
}

Error PassBuilder::parseModulePass(ModulePassManager &MPM,
                                   const PipelineElement &E) {
  auto &Name = E.Name;
  auto &InnerPipeline = E.InnerPipeline;

  // First handle complex passes like the pass managers which carry pipelines.
  if (!InnerPipeline.empty()) {
    if (Name == "module") {
      ModulePassManager NestedMPM;
      if (auto Err = parseModulePassPipeline(NestedMPM, InnerPipeline))
        return Err;
      MPM.addPass(std::move(NestedMPM));
      return Error::success();
    }
    if (Name == "cgscc") {
      CGSCCPassManager CGPM;
      if (auto Err = parseCGSCCPassPipeline(CGPM, InnerPipeline))
        return Err;
      MPM.addPass(createModuleToPostOrderCGSCCPassAdaptor(std::move(CGPM)));
      return Error::success();
    }
    if (Name == "function" || Name == "function<eager-inv>") {
      FunctionPassManager FPM;
      if (auto Err = parseFunctionPassPipeline(FPM, InnerPipeline))
        return Err;
      MPM.addPass(createModuleToFunctionPassAdaptor(std::move(FPM),
                                                    Name != "function"));
      return Error::success();
    }
    if (auto Count = parseRepeatPassName(Name)) {
      ModulePassManager NestedMPM;
      if (auto Err = parseModulePassPipeline(NestedMPM, InnerPipeline))
        return Err;
      MPM.addPass(createRepeatedPass(*Count, std::move(NestedMPM)));
      return Error::success();
    }

    for (auto &C : ModulePipelineParsingCallbacks)
      if (C(Name, MPM, InnerPipeline))
        return Error::success();

    // Normal passes can't have pipelines.
    return make_error<StringError>(
        formatv("invalid use of '{0}' pass as module pipeline", Name).str(),
        inconvertibleErrorCode());
    ;
  }

  // Manually handle aliases for pre-configured pipeline fragments.
  if (startsWithDefaultPipelineAliasPrefix(Name)) {
    SmallVector<StringRef, 3> Matches;
    if (!DefaultAliasRegex.match(Name, &Matches))
      return make_error<StringError>(
          formatv("unknown default pipeline alias '{0}'", Name).str(),
          inconvertibleErrorCode());

    assert(Matches.size() == 3 && "Must capture two matched strings!");

    OptimizationLevel L = StringSwitch<OptimizationLevel>(Matches[2])
                              .Case("O0", OptimizationLevel::O0)
                              .Case("O1", OptimizationLevel::O1)
                              .Case("O2", OptimizationLevel::O2)
                              .Case("O3", OptimizationLevel::O3)
                              .Case("Os", OptimizationLevel::Os)
                              .Case("Oz", OptimizationLevel::Oz);
    if (L == OptimizationLevel::O0 && Matches[1] != "thinlto" &&
        Matches[1] != "lto") {
      MPM.addPass(buildO0DefaultPipeline(L, Matches[1] == "thinlto-pre-link" ||
                                                Matches[1] == "lto-pre-link"));
      return Error::success();
    }

    // This is consistent with old pass manager invoked via opt, but
    // inconsistent with clang. Clang doesn't enable loop vectorization
    // but does enable slp vectorization at Oz.
    PTO.LoopVectorization =
        L.getSpeedupLevel() > 1 && L != OptimizationLevel::Oz;
    PTO.SLPVectorization =
        L.getSpeedupLevel() > 1 && L != OptimizationLevel::Oz;

    if (Matches[1] == "default") {
      MPM.addPass(buildPerModuleDefaultPipeline(L));
    } else if (Matches[1] == "thinlto-pre-link") {
      MPM.addPass(buildThinLTOPreLinkDefaultPipeline(L));
    } else if (Matches[1] == "thinlto") {
      MPM.addPass(buildThinLTODefaultPipeline(L, nullptr));
    } else if (Matches[1] == "lto-pre-link") {
      MPM.addPass(buildLTOPreLinkDefaultPipeline(L));
    } else {
      assert(Matches[1] == "lto" && "Not one of the matched options!");
      MPM.addPass(buildLTODefaultPipeline(L, nullptr));
    }
    return Error::success();
  }

  // Finally expand the basic registered passes from the .inc file.
#define MODULE_PASS(NAME, CREATE_PASS)                                         \
  if (Name == NAME) {                                                          \
    MPM.addPass(CREATE_PASS);                                                  \
    return Error::success();                                                   \
  }
#define MODULE_PASS_WITH_PARAMS(NAME, CLASS, CREATE_PASS, PARSER, PARAMS)      \
  if (checkParametrizedPassName(Name, NAME)) {                                 \
    auto Params = parsePassParameters(PARSER, Name, NAME);                     \
    if (!Params)                                                               \
      return Params.takeError();                                               \
    MPM.addPass(CREATE_PASS(Params.get()));                                    \
    return Error::success();                                                   \
  }
#define MODULE_ANALYSIS(NAME, CREATE_PASS)                                     \
  if (Name == "require<" NAME ">") {                                           \
    MPM.addPass(                                                               \
        RequireAnalysisPass<                                                   \
            std::remove_reference<decltype(CREATE_PASS)>::type, Module>());    \
    return Error::success();                                                   \
  }                                                                            \
  if (Name == "invalidate<" NAME ">") {                                        \
    MPM.addPass(InvalidateAnalysisPass<                                        \
                std::remove_reference<decltype(CREATE_PASS)>::type>());        \
    return Error::success();                                                   \
  }
#define CGSCC_PASS(NAME, CREATE_PASS)                                          \
  if (Name == NAME) {                                                          \
    MPM.addPass(createModuleToPostOrderCGSCCPassAdaptor(CREATE_PASS));         \
    return Error::success();                                                   \
  }
#define CGSCC_PASS_WITH_PARAMS(NAME, CLASS, CREATE_PASS, PARSER, PARAMS)       \
  if (checkParametrizedPassName(Name, NAME)) {                                 \
    auto Params = parsePassParameters(PARSER, Name, NAME);                     \
    if (!Params)                                                               \
      return Params.takeError();                                               \
    MPM.addPass(                                                               \
        createModuleToPostOrderCGSCCPassAdaptor(CREATE_PASS(Params.get())));   \
    return Error::success();                                                   \
  }
#define FUNCTION_PASS(NAME, CREATE_PASS)                                       \
  if (Name == NAME) {                                                          \
    MPM.addPass(createModuleToFunctionPassAdaptor(CREATE_PASS));               \
    return Error::success();                                                   \
  }
#define FUNCTION_PASS_WITH_PARAMS(NAME, CLASS, CREATE_PASS, PARSER, PARAMS)    \
  if (checkParametrizedPassName(Name, NAME)) {                                 \
    auto Params = parsePassParameters(PARSER, Name, NAME);                     \
    if (!Params)                                                               \
      return Params.takeError();                                               \
    MPM.addPass(createModuleToFunctionPassAdaptor(CREATE_PASS(Params.get()))); \
    return Error::success();                                                   \
  }
#define LOOPNEST_PASS(NAME, CREATE_PASS)                                       \
  if (Name == NAME) {                                                          \
    MPM.addPass(createModuleToFunctionPassAdaptor(                             \
        createFunctionToLoopPassAdaptor(CREATE_PASS, false, false)));          \
    return Error::success();                                                   \
  }
#define LOOP_PASS(NAME, CREATE_PASS)                                           \
  if (Name == NAME) {                                                          \
    MPM.addPass(createModuleToFunctionPassAdaptor(                             \
        createFunctionToLoopPassAdaptor(CREATE_PASS, false, false)));          \
    return Error::success();                                                   \
  }
#define LOOP_PASS_WITH_PARAMS(NAME, CLASS, CREATE_PASS, PARSER, PARAMS)        \
  if (checkParametrizedPassName(Name, NAME)) {                                 \
    auto Params = parsePassParameters(PARSER, Name, NAME);                     \
    if (!Params)                                                               \
      return Params.takeError();                                               \
    MPM.addPass(                                                               \
        createModuleToFunctionPassAdaptor(createFunctionToLoopPassAdaptor(     \
            CREATE_PASS(Params.get()), false, false)));                        \
    return Error::success();                                                   \
  }
#include "PassRegistry.def"

  for (auto &C : ModulePipelineParsingCallbacks)
    if (C(Name, MPM, InnerPipeline))
      return Error::success();
  return make_error<StringError>(
      formatv("unknown module pass '{0}'", Name).str(),
      inconvertibleErrorCode());
}

Error PassBuilder::parseCGSCCPass(CGSCCPassManager &CGPM,
                                  const PipelineElement &E) {
  auto &Name = E.Name;
  auto &InnerPipeline = E.InnerPipeline;

  // First handle complex passes like the pass managers which carry pipelines.
  if (!InnerPipeline.empty()) {
    if (Name == "cgscc") {
      CGSCCPassManager NestedCGPM;
      if (auto Err = parseCGSCCPassPipeline(NestedCGPM, InnerPipeline))
        return Err;
      // Add the nested pass manager with the appropriate adaptor.
      CGPM.addPass(std::move(NestedCGPM));
      return Error::success();
    }
    if (Name == "function" || Name == "function<eager-inv>") {
      FunctionPassManager FPM;
      if (auto Err = parseFunctionPassPipeline(FPM, InnerPipeline))
        return Err;
      // Add the nested pass manager with the appropriate adaptor.
      CGPM.addPass(
          createCGSCCToFunctionPassAdaptor(std::move(FPM), Name != "function"));
      return Error::success();
    }
    if (auto Count = parseRepeatPassName(Name)) {
      CGSCCPassManager NestedCGPM;
      if (auto Err = parseCGSCCPassPipeline(NestedCGPM, InnerPipeline))
        return Err;
      CGPM.addPass(createRepeatedPass(*Count, std::move(NestedCGPM)));
      return Error::success();
    }
    if (auto MaxRepetitions = parseDevirtPassName(Name)) {
      CGSCCPassManager NestedCGPM;
      if (auto Err = parseCGSCCPassPipeline(NestedCGPM, InnerPipeline))
        return Err;
      CGPM.addPass(
          createDevirtSCCRepeatedPass(std::move(NestedCGPM), *MaxRepetitions));
      return Error::success();
    }

    for (auto &C : CGSCCPipelineParsingCallbacks)
      if (C(Name, CGPM, InnerPipeline))
        return Error::success();

    // Normal passes can't have pipelines.
    return make_error<StringError>(
        formatv("invalid use of '{0}' pass as cgscc pipeline", Name).str(),
        inconvertibleErrorCode());
  }

// Now expand the basic registered passes from the .inc file.
#define CGSCC_PASS(NAME, CREATE_PASS)                                          \
  if (Name == NAME) {                                                          \
    CGPM.addPass(CREATE_PASS);                                                 \
    return Error::success();                                                   \
  }
#define CGSCC_PASS_WITH_PARAMS(NAME, CLASS, CREATE_PASS, PARSER, PARAMS)       \
  if (checkParametrizedPassName(Name, NAME)) {                                 \
    auto Params = parsePassParameters(PARSER, Name, NAME);                     \
    if (!Params)                                                               \
      return Params.takeError();                                               \
    CGPM.addPass(CREATE_PASS(Params.get()));                                   \
    return Error::success();                                                   \
  }
#define CGSCC_ANALYSIS(NAME, CREATE_PASS)                                      \
  if (Name == "require<" NAME ">") {                                           \
    CGPM.addPass(RequireAnalysisPass<                                          \
                 std::remove_reference<decltype(CREATE_PASS)>::type,           \
                 LazyCallGraph::SCC, CGSCCAnalysisManager, LazyCallGraph &,    \
                 CGSCCUpdateResult &>());                                      \
    return Error::success();                                                   \
  }                                                                            \
  if (Name == "invalidate<" NAME ">") {                                        \
    CGPM.addPass(InvalidateAnalysisPass<                                       \
                 std::remove_reference<decltype(CREATE_PASS)>::type>());       \
    return Error::success();                                                   \
  }
#define FUNCTION_PASS(NAME, CREATE_PASS)                                       \
  if (Name == NAME) {                                                          \
    CGPM.addPass(createCGSCCToFunctionPassAdaptor(CREATE_PASS));               \
    return Error::success();                                                   \
  }
#define FUNCTION_PASS_WITH_PARAMS(NAME, CLASS, CREATE_PASS, PARSER, PARAMS)    \
  if (checkParametrizedPassName(Name, NAME)) {                                 \
    auto Params = parsePassParameters(PARSER, Name, NAME);                     \
    if (!Params)                                                               \
      return Params.takeError();                                               \
    CGPM.addPass(createCGSCCToFunctionPassAdaptor(CREATE_PASS(Params.get()))); \
    return Error::success();                                                   \
  }
#define LOOPNEST_PASS(NAME, CREATE_PASS)                                       \
  if (Name == NAME) {                                                          \
    CGPM.addPass(createCGSCCToFunctionPassAdaptor(                             \
        createFunctionToLoopPassAdaptor(CREATE_PASS, false, false)));          \
    return Error::success();                                                   \
  }
#define LOOP_PASS(NAME, CREATE_PASS)                                           \
  if (Name == NAME) {                                                          \
    CGPM.addPass(createCGSCCToFunctionPassAdaptor(                             \
        createFunctionToLoopPassAdaptor(CREATE_PASS, false, false)));          \
    return Error::success();                                                   \
  }
#define LOOP_PASS_WITH_PARAMS(NAME, CLASS, CREATE_PASS, PARSER, PARAMS)        \
  if (checkParametrizedPassName(Name, NAME)) {                                 \
    auto Params = parsePassParameters(PARSER, Name, NAME);                     \
    if (!Params)                                                               \
      return Params.takeError();                                               \
    CGPM.addPass(                                                              \
        createCGSCCToFunctionPassAdaptor(createFunctionToLoopPassAdaptor(      \
            CREATE_PASS(Params.get()), false, false)));                        \
    return Error::success();                                                   \
  }
#include "PassRegistry.def"

  for (auto &C : CGSCCPipelineParsingCallbacks)
    if (C(Name, CGPM, InnerPipeline))
      return Error::success();
  return make_error<StringError>(
      formatv("unknown cgscc pass '{0}'", Name).str(),
      inconvertibleErrorCode());
}

Error PassBuilder::parseFunctionPass(FunctionPassManager &FPM,
                                     const PipelineElement &E) {
  auto &Name = E.Name;
  auto &InnerPipeline = E.InnerPipeline;

  // First handle complex passes like the pass managers which carry pipelines.
  if (!InnerPipeline.empty()) {
    if (Name == "function") {
      FunctionPassManager NestedFPM;
      if (auto Err = parseFunctionPassPipeline(NestedFPM, InnerPipeline))
        return Err;
      // Add the nested pass manager with the appropriate adaptor.
      FPM.addPass(std::move(NestedFPM));
      return Error::success();
    }
    if (Name == "loop" || Name == "loop-mssa") {
      LoopPassManager LPM;
      if (auto Err = parseLoopPassPipeline(LPM, InnerPipeline))
        return Err;
      // Add the nested pass manager with the appropriate adaptor.
      bool UseMemorySSA = (Name == "loop-mssa");
      bool UseBFI = llvm::any_of(
          InnerPipeline, [](auto Pipeline) { return Pipeline.Name == "licm"; });
      bool UseBPI = llvm::any_of(InnerPipeline, [](auto Pipeline) {
        return Pipeline.Name == "loop-predication";
      });
      FPM.addPass(createFunctionToLoopPassAdaptor(std::move(LPM), UseMemorySSA,
                                                  UseBFI, UseBPI));
      return Error::success();
    }
    if (auto Count = parseRepeatPassName(Name)) {
      FunctionPassManager NestedFPM;
      if (auto Err = parseFunctionPassPipeline(NestedFPM, InnerPipeline))
        return Err;
      FPM.addPass(createRepeatedPass(*Count, std::move(NestedFPM)));
      return Error::success();
    }

    for (auto &C : FunctionPipelineParsingCallbacks)
      if (C(Name, FPM, InnerPipeline))
        return Error::success();

    // Normal passes can't have pipelines.
    return make_error<StringError>(
        formatv("invalid use of '{0}' pass as function pipeline", Name).str(),
        inconvertibleErrorCode());
  }

// Now expand the basic registered passes from the .inc file.
#define FUNCTION_PASS(NAME, CREATE_PASS)                                       \
  if (Name == NAME) {                                                          \
    FPM.addPass(CREATE_PASS);                                                  \
    return Error::success();                                                   \
  }
#define FUNCTION_PASS_WITH_PARAMS(NAME, CLASS, CREATE_PASS, PARSER, PARAMS)    \
  if (checkParametrizedPassName(Name, NAME)) {                                 \
    auto Params = parsePassParameters(PARSER, Name, NAME);                     \
    if (!Params)                                                               \
      return Params.takeError();                                               \
    FPM.addPass(CREATE_PASS(Params.get()));                                    \
    return Error::success();                                                   \
  }
#define FUNCTION_ANALYSIS(NAME, CREATE_PASS)                                   \
  if (Name == "require<" NAME ">") {                                           \
    FPM.addPass(                                                               \
        RequireAnalysisPass<                                                   \
            std::remove_reference<decltype(CREATE_PASS)>::type, Function>());  \
    return Error::success();                                                   \
  }                                                                            \
  if (Name == "invalidate<" NAME ">") {                                        \
    FPM.addPass(InvalidateAnalysisPass<                                        \
                std::remove_reference<decltype(CREATE_PASS)>::type>());        \
    return Error::success();                                                   \
  }
// FIXME: UseMemorySSA is set to false. Maybe we could do things like:
//        bool UseMemorySSA = !("canon-freeze" || "loop-predication" ||
//                              "guard-widening");
//        The risk is that it may become obsolete if we're not careful.
#define LOOPNEST_PASS(NAME, CREATE_PASS)                                       \
  if (Name == NAME) {                                                          \
    FPM.addPass(createFunctionToLoopPassAdaptor(CREATE_PASS, false, false));   \
    return Error::success();                                                   \
  }
#define LOOP_PASS(NAME, CREATE_PASS)                                           \
  if (Name == NAME) {                                                          \
    FPM.addPass(createFunctionToLoopPassAdaptor(CREATE_PASS, false, false));   \
    return Error::success();                                                   \
  }
#define LOOP_PASS_WITH_PARAMS(NAME, CLASS, CREATE_PASS, PARSER, PARAMS)        \
  if (checkParametrizedPassName(Name, NAME)) {                                 \
    auto Params = parsePassParameters(PARSER, Name, NAME);                     \
    if (!Params)                                                               \
      return Params.takeError();                                               \
    FPM.addPass(createFunctionToLoopPassAdaptor(CREATE_PASS(Params.get()),     \
                                                false, false));                \
    return Error::success();                                                   \
  }
#include "PassRegistry.def"

  for (auto &C : FunctionPipelineParsingCallbacks)
    if (C(Name, FPM, InnerPipeline))
      return Error::success();
  return make_error<StringError>(
      formatv("unknown function pass '{0}'", Name).str(),
      inconvertibleErrorCode());
}

Error PassBuilder::parseLoopPass(LoopPassManager &LPM,
                                 const PipelineElement &E) {
  StringRef Name = E.Name;
  auto &InnerPipeline = E.InnerPipeline;

  // First handle complex passes like the pass managers which carry pipelines.
  if (!InnerPipeline.empty()) {
    if (Name == "loop") {
      LoopPassManager NestedLPM;
      if (auto Err = parseLoopPassPipeline(NestedLPM, InnerPipeline))
        return Err;
      // Add the nested pass manager with the appropriate adaptor.
      LPM.addPass(std::move(NestedLPM));
      return Error::success();
    }
    if (auto Count = parseRepeatPassName(Name)) {
      LoopPassManager NestedLPM;
      if (auto Err = parseLoopPassPipeline(NestedLPM, InnerPipeline))
        return Err;
      LPM.addPass(createRepeatedPass(*Count, std::move(NestedLPM)));
      return Error::success();
    }

    for (auto &C : LoopPipelineParsingCallbacks)
      if (C(Name, LPM, InnerPipeline))
        return Error::success();

    // Normal passes can't have pipelines.
    return make_error<StringError>(
        formatv("invalid use of '{0}' pass as loop pipeline", Name).str(),
        inconvertibleErrorCode());
  }

// Now expand the basic registered passes from the .inc file.
#define LOOPNEST_PASS(NAME, CREATE_PASS)                                       \
  if (Name == NAME) {                                                          \
    LPM.addPass(CREATE_PASS);                                                  \
    return Error::success();                                                   \
  }
#define LOOP_PASS(NAME, CREATE_PASS)                                           \
  if (Name == NAME) {                                                          \
    LPM.addPass(CREATE_PASS);                                                  \
    return Error::success();                                                   \
  }
#define LOOP_PASS_WITH_PARAMS(NAME, CLASS, CREATE_PASS, PARSER, PARAMS)        \
  if (checkParametrizedPassName(Name, NAME)) {                                 \
    auto Params = parsePassParameters(PARSER, Name, NAME);                     \
    if (!Params)                                                               \
      return Params.takeError();                                               \
    LPM.addPass(CREATE_PASS(Params.get()));                                    \
    return Error::success();                                                   \
  }
#define LOOP_ANALYSIS(NAME, CREATE_PASS)                                       \
  if (Name == "require<" NAME ">") {                                           \
    LPM.addPass(RequireAnalysisPass<                                           \
                std::remove_reference<decltype(CREATE_PASS)>::type, Loop,      \
                LoopAnalysisManager, LoopStandardAnalysisResults &,            \
                LPMUpdater &>());                                              \
    return Error::success();                                                   \
  }                                                                            \
  if (Name == "invalidate<" NAME ">") {                                        \
    LPM.addPass(InvalidateAnalysisPass<                                        \
                std::remove_reference<decltype(CREATE_PASS)>::type>());        \
    return Error::success();                                                   \
  }
#include "PassRegistry.def"

  for (auto &C : LoopPipelineParsingCallbacks)
    if (C(Name, LPM, InnerPipeline))
      return Error::success();
  return make_error<StringError>(formatv("unknown loop pass '{0}'", Name).str(),
                                 inconvertibleErrorCode());
}

bool PassBuilder::parseAAPassName(AAManager &AA, StringRef Name) {
#define MODULE_ALIAS_ANALYSIS(NAME, CREATE_PASS)                               \
  if (Name == NAME) {                                                          \
    AA.registerModuleAnalysis<                                                 \
        std::remove_reference<decltype(CREATE_PASS)>::type>();                 \
    return true;                                                               \
  }
#define FUNCTION_ALIAS_ANALYSIS(NAME, CREATE_PASS)                             \
  if (Name == NAME) {                                                          \
    AA.registerFunctionAnalysis<                                               \
        std::remove_reference<decltype(CREATE_PASS)>::type>();                 \
    return true;                                                               \
  }
#include "PassRegistry.def"

  for (auto &C : AAParsingCallbacks)
    if (C(Name, AA))
      return true;
  return false;
}

Error PassBuilder::parseLoopPassPipeline(LoopPassManager &LPM,
                                         ArrayRef<PipelineElement> Pipeline) {
  for (const auto &Element : Pipeline) {
    if (auto Err = parseLoopPass(LPM, Element))
      return Err;
  }
  return Error::success();
}

Error PassBuilder::parseFunctionPassPipeline(
    FunctionPassManager &FPM, ArrayRef<PipelineElement> Pipeline) {
  for (const auto &Element : Pipeline) {
    if (auto Err = parseFunctionPass(FPM, Element))
      return Err;
  }
  return Error::success();
}

Error PassBuilder::parseCGSCCPassPipeline(CGSCCPassManager &CGPM,
                                          ArrayRef<PipelineElement> Pipeline) {
  for (const auto &Element : Pipeline) {
    if (auto Err = parseCGSCCPass(CGPM, Element))
      return Err;
  }
  return Error::success();
}

void PassBuilder::crossRegisterProxies(LoopAnalysisManager &LAM,
                                       FunctionAnalysisManager &FAM,
                                       CGSCCAnalysisManager &CGAM,
                                       ModuleAnalysisManager &MAM) {
  MAM.registerPass([&] { return FunctionAnalysisManagerModuleProxy(FAM); });
  MAM.registerPass([&] { return CGSCCAnalysisManagerModuleProxy(CGAM); });
  CGAM.registerPass([&] { return ModuleAnalysisManagerCGSCCProxy(MAM); });
  FAM.registerPass([&] { return CGSCCAnalysisManagerFunctionProxy(CGAM); });
  FAM.registerPass([&] { return ModuleAnalysisManagerFunctionProxy(MAM); });
  FAM.registerPass([&] { return LoopAnalysisManagerFunctionProxy(LAM); });
  LAM.registerPass([&] { return FunctionAnalysisManagerLoopProxy(FAM); });
}

Error PassBuilder::parseModulePassPipeline(ModulePassManager &MPM,
                                           ArrayRef<PipelineElement> Pipeline) {
  for (const auto &Element : Pipeline) {
    if (auto Err = parseModulePass(MPM, Element))
      return Err;
  }
  return Error::success();
}

// Primary pass pipeline description parsing routine for a \c ModulePassManager
// FIXME: Should this routine accept a TargetMachine or require the caller to
// pre-populate the analysis managers with target-specific stuff?
Error PassBuilder::parsePassPipeline(ModulePassManager &MPM,
                                     StringRef PipelineText) {
  auto Pipeline = parsePipelineText(PipelineText);
  if (!Pipeline || Pipeline->empty())
    return make_error<StringError>(
        formatv("invalid pipeline '{0}'", PipelineText).str(),
        inconvertibleErrorCode());

  // If the first name isn't at the module layer, wrap the pipeline up
  // automatically.
  StringRef FirstName = Pipeline->front().Name;

  if (!isModulePassName(FirstName, ModulePipelineParsingCallbacks)) {
    bool UseMemorySSA;
    if (isCGSCCPassName(FirstName, CGSCCPipelineParsingCallbacks)) {
      Pipeline = {{"cgscc", std::move(*Pipeline)}};
    } else if (isFunctionPassName(FirstName,
                                  FunctionPipelineParsingCallbacks)) {
      Pipeline = {{"function", std::move(*Pipeline)}};
    } else if (isLoopNestPassName(FirstName, LoopPipelineParsingCallbacks,
                                  UseMemorySSA)) {
      Pipeline = {{"function", {{UseMemorySSA ? "loop-mssa" : "loop",
                                 std::move(*Pipeline)}}}};
    } else if (isLoopPassName(FirstName, LoopPipelineParsingCallbacks,
                              UseMemorySSA)) {
      Pipeline = {{"function", {{UseMemorySSA ? "loop-mssa" : "loop",
                                 std::move(*Pipeline)}}}};
    } else {
      for (auto &C : TopLevelPipelineParsingCallbacks)
        if (C(MPM, *Pipeline))
          return Error::success();

      // Unknown pass or pipeline name!
      auto &InnerPipeline = Pipeline->front().InnerPipeline;
      return make_error<StringError>(
          formatv("unknown {0} name '{1}'",
                  (InnerPipeline.empty() ? "pass" : "pipeline"), FirstName)
              .str(),
          inconvertibleErrorCode());
    }
  }

  if (auto Err = parseModulePassPipeline(MPM, *Pipeline))
    return Err;
  return Error::success();
}

// Primary pass pipeline description parsing routine for a \c CGSCCPassManager
Error PassBuilder::parsePassPipeline(CGSCCPassManager &CGPM,
                                     StringRef PipelineText) {
  auto Pipeline = parsePipelineText(PipelineText);
  if (!Pipeline || Pipeline->empty())
    return make_error<StringError>(
        formatv("invalid pipeline '{0}'", PipelineText).str(),
        inconvertibleErrorCode());

  StringRef FirstName = Pipeline->front().Name;
  if (!isCGSCCPassName(FirstName, CGSCCPipelineParsingCallbacks))
    return make_error<StringError>(
        formatv("unknown cgscc pass '{0}' in pipeline '{1}'", FirstName,
                PipelineText)
            .str(),
        inconvertibleErrorCode());

  if (auto Err = parseCGSCCPassPipeline(CGPM, *Pipeline))
    return Err;
  return Error::success();
}

// Primary pass pipeline description parsing routine for a \c
// FunctionPassManager
Error PassBuilder::parsePassPipeline(FunctionPassManager &FPM,
                                     StringRef PipelineText) {
  auto Pipeline = parsePipelineText(PipelineText);
  if (!Pipeline || Pipeline->empty())
    return make_error<StringError>(
        formatv("invalid pipeline '{0}'", PipelineText).str(),
        inconvertibleErrorCode());

  StringRef FirstName = Pipeline->front().Name;
  if (!isFunctionPassName(FirstName, FunctionPipelineParsingCallbacks))
    return make_error<StringError>(
        formatv("unknown function pass '{0}' in pipeline '{1}'", FirstName,
                PipelineText)
            .str(),
        inconvertibleErrorCode());

  if (auto Err = parseFunctionPassPipeline(FPM, *Pipeline))
    return Err;
  return Error::success();
}

// Primary pass pipeline description parsing routine for a \c LoopPassManager
Error PassBuilder::parsePassPipeline(LoopPassManager &CGPM,
                                     StringRef PipelineText) {
  auto Pipeline = parsePipelineText(PipelineText);
  if (!Pipeline || Pipeline->empty())
    return make_error<StringError>(
        formatv("invalid pipeline '{0}'", PipelineText).str(),
        inconvertibleErrorCode());

  if (auto Err = parseLoopPassPipeline(CGPM, *Pipeline))
    return Err;

  return Error::success();
}

Error PassBuilder::parseAAPipeline(AAManager &AA, StringRef PipelineText) {
  // If the pipeline just consists of the word 'default' just replace the AA
  // manager with our default one.
  if (PipelineText == "default") {
    AA = buildDefaultAAPipeline();
    return Error::success();
  }

  while (!PipelineText.empty()) {
    StringRef Name;
    std::tie(Name, PipelineText) = PipelineText.split(',');
    if (!parseAAPassName(AA, Name))
      return make_error<StringError>(
          formatv("unknown alias analysis name '{0}'", Name).str(),
          inconvertibleErrorCode());
  }

  return Error::success();
}

bool PassBuilder::isAAPassName(StringRef PassName) {
#define MODULE_ALIAS_ANALYSIS(NAME, CREATE_PASS)                               \
  if (PassName == NAME)                                                        \
    return true;
#define FUNCTION_ALIAS_ANALYSIS(NAME, CREATE_PASS)                             \
  if (PassName == NAME)                                                        \
    return true;
#include "PassRegistry.def"
  return false;
}

bool PassBuilder::isAnalysisPassName(StringRef PassName) {
#define MODULE_ANALYSIS(NAME, CREATE_PASS)                                     \
  if (PassName == NAME)                                                        \
    return true;
#define FUNCTION_ANALYSIS(NAME, CREATE_PASS)                                   \
  if (PassName == NAME)                                                        \
    return true;
#define LOOP_ANALYSIS(NAME, CREATE_PASS)                                       \
  if (PassName == NAME)                                                        \
    return true;
#define CGSCC_ANALYSIS(NAME, CREATE_PASS)                                      \
  if (PassName == NAME)                                                        \
    return true;
#define MODULE_ALIAS_ANALYSIS(NAME, CREATE_PASS)                               \
  if (PassName == NAME)                                                        \
    return true;
#define FUNCTION_ALIAS_ANALYSIS(NAME, CREATE_PASS)                             \
  if (PassName == NAME)                                                        \
    return true;
#include "PassRegistry.def"
  return false;
}

static void printPassName(StringRef PassName, raw_ostream &OS) {
  OS << "  " << PassName << "\n";
}
static void printPassName(StringRef PassName, StringRef Params,
                          raw_ostream &OS) {
  OS << "  " << PassName << "<" << Params << ">\n";
}

void PassBuilder::printPassNames(raw_ostream &OS) {
  // TODO: print pass descriptions when they are available

  OS << "Module passes:\n";
#define MODULE_PASS(NAME, CREATE_PASS) printPassName(NAME, OS);
#include "PassRegistry.def"

  OS << "Module passes with params:\n";
#define MODULE_PASS_WITH_PARAMS(NAME, CLASS, CREATE_PASS, PARSER, PARAMS)      \
  printPassName(NAME, PARAMS, OS);
#include "PassRegistry.def"

  OS << "Module analyses:\n";
#define MODULE_ANALYSIS(NAME, CREATE_PASS) printPassName(NAME, OS);
#include "PassRegistry.def"

  OS << "Module alias analyses:\n";
#define MODULE_ALIAS_ANALYSIS(NAME, CREATE_PASS) printPassName(NAME, OS);
#include "PassRegistry.def"

  OS << "CGSCC passes:\n";
#define CGSCC_PASS(NAME, CREATE_PASS) printPassName(NAME, OS);
#include "PassRegistry.def"

  OS << "CGSCC passes with params:\n";
#define CGSCC_PASS_WITH_PARAMS(NAME, CLASS, CREATE_PASS, PARSER, PARAMS)       \
  printPassName(NAME, PARAMS, OS);
#include "PassRegistry.def"

  OS << "CGSCC analyses:\n";
#define CGSCC_ANALYSIS(NAME, CREATE_PASS) printPassName(NAME, OS);
#include "PassRegistry.def"

  OS << "Function passes:\n";
#define FUNCTION_PASS(NAME, CREATE_PASS) printPassName(NAME, OS);
#include "PassRegistry.def"

  OS << "Function passes with params:\n";
#define FUNCTION_PASS_WITH_PARAMS(NAME, CLASS, CREATE_PASS, PARSER, PARAMS)    \
  printPassName(NAME, PARAMS, OS);
#include "PassRegistry.def"

  OS << "Function analyses:\n";
#define FUNCTION_ANALYSIS(NAME, CREATE_PASS) printPassName(NAME, OS);
#include "PassRegistry.def"

  OS << "Function alias analyses:\n";
#define FUNCTION_ALIAS_ANALYSIS(NAME, CREATE_PASS) printPassName(NAME, OS);
#include "PassRegistry.def"

  OS << "LoopNest passes:\n";
#define LOOPNEST_PASS(NAME, CREATE_PASS) printPassName(NAME, OS);
#include "PassRegistry.def"

  OS << "Loop passes:\n";
#define LOOP_PASS(NAME, CREATE_PASS) printPassName(NAME, OS);
#include "PassRegistry.def"

  OS << "Loop passes with params:\n";
#define LOOP_PASS_WITH_PARAMS(NAME, CLASS, CREATE_PASS, PARSER, PARAMS)        \
  printPassName(NAME, PARAMS, OS);
#include "PassRegistry.def"

  OS << "Loop analyses:\n";
#define LOOP_ANALYSIS(NAME, CREATE_PASS) printPassName(NAME, OS);
#include "PassRegistry.def"
}

void PassBuilder::registerParseTopLevelPipelineCallback(
    const std::function<bool(ModulePassManager &, ArrayRef<PipelineElement>)>
        &C) {
  TopLevelPipelineParsingCallbacks.push_back(C);
}
