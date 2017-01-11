//===- LoopPassManager.h - Loop pass management -----------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file
///
/// This header provides classes for managing a pipeline of passes over loops
/// in LLVM IR.
///
/// The primary loop pass pipeline is managed in a very particular way to
/// provide a set of core guarantees:
/// 1) Loops are, where possible, in simplified form.
/// 2) Loops are *always* in LCSSA form.
/// 3) A collection of Loop-specific analysis results are available:
///    - LoopInfo
///    - DominatorTree
///    - ScalarEvolution
///    - AAManager
/// 4) All loop passes preserve #1 (where possible), #2, and #3.
/// 5) Loop passes run over each loop in the loop nest from the innermost to
///    the outermost. Specifically, all inner loops are processed before
///    passes run over outer loops. When running the pipeline across an inner
///    loop creates new inner loops, those are added and processed in this
///    order as well.
///
/// This process is designed to facilitate transformations which simplify,
/// reduce, and remove loops. For passes which are more oriented towards
/// optimizing loops, especially optimizing loop *nests* instead of single
/// loops in isolation, this framework is less interesting.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_LOOPPASSMANAGER_H
#define LLVM_ANALYSIS_LOOPPASSMANAGER_H

#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/PriorityWorklist.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/BasicAliasAnalysis.h"
#include "llvm/Analysis/GlobalsModRef.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionAliasAnalysis.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/PassManager.h"

namespace llvm {

// Forward declarations of a update tracking and analysis result tracking
// structures used in the API of loop passes that work within this
// infrastructure.
class LPMUpdater;
struct LoopStandardAnalysisResults;

/// Extern template declaration for the analysis set for this IR unit.
extern template class AllAnalysesOn<Loop>;

extern template class AnalysisManager<Loop, LoopStandardAnalysisResults &>;
/// \brief The loop analysis manager.
///
/// See the documentation for the AnalysisManager template for detail
/// documentation. This typedef serves as a convenient way to refer to this
/// construct in the adaptors and proxies used to integrate this into the larger
/// pass manager infrastructure.
typedef AnalysisManager<Loop, LoopStandardAnalysisResults &>
    LoopAnalysisManager;

// Explicit specialization and instantiation declarations for the pass manager.
// See the comments on the definition of the specialization for details on how
// it differs from the primary template.
template <>
PreservedAnalyses
PassManager<Loop, LoopAnalysisManager, LoopStandardAnalysisResults &,
            LPMUpdater &>::run(Loop &InitialL, LoopAnalysisManager &AM,
                               LoopStandardAnalysisResults &AnalysisResults,
                               LPMUpdater &U);
extern template class PassManager<Loop, LoopAnalysisManager,
                                  LoopStandardAnalysisResults &, LPMUpdater &>;

/// \brief The Loop pass manager.
///
/// See the documentation for the PassManager template for details. It runs
/// a sequence of Loop passes over each Loop that the manager is run over. This
/// typedef serves as a convenient way to refer to this construct.
typedef PassManager<Loop, LoopAnalysisManager, LoopStandardAnalysisResults &,
                    LPMUpdater &>
    LoopPassManager;

/// A partial specialization of the require analysis template pass to forward
/// the extra parameters from a transformation's run method to the
/// AnalysisManager's getResult.
template <typename AnalysisT>
struct RequireAnalysisPass<AnalysisT, Loop, LoopAnalysisManager,
                           LoopStandardAnalysisResults &, LPMUpdater &>
    : PassInfoMixin<
          RequireAnalysisPass<AnalysisT, Loop, LoopAnalysisManager,
                              LoopStandardAnalysisResults &, LPMUpdater &>> {
  PreservedAnalyses run(Loop &L, LoopAnalysisManager &AM,
                        LoopStandardAnalysisResults &AR, LPMUpdater &) {
    (void)AM.template getResult<AnalysisT>(L, AR);
    return PreservedAnalyses::all();
  }
};

/// An alias template to easily name a require analysis loop pass.
template <typename AnalysisT>
using RequireAnalysisLoopPass =
    RequireAnalysisPass<AnalysisT, Loop, LoopAnalysisManager,
                        LoopStandardAnalysisResults &, LPMUpdater &>;

/// A proxy from a \c LoopAnalysisManager to a \c Function.
typedef InnerAnalysisManagerProxy<LoopAnalysisManager, Function>
    LoopAnalysisManagerFunctionProxy;

/// A specialized result for the \c LoopAnalysisManagerFunctionProxy which
/// retains a \c LoopInfo reference.
///
/// This allows it to collect loop objects for which analysis results may be
/// cached in the \c LoopAnalysisManager.
template <> class LoopAnalysisManagerFunctionProxy::Result {
public:
  explicit Result(LoopAnalysisManager &InnerAM, LoopInfo &LI)
      : InnerAM(&InnerAM), LI(&LI) {}
  Result(Result &&Arg) : InnerAM(std::move(Arg.InnerAM)), LI(Arg.LI) {
    // We have to null out the analysis manager in the moved-from state
    // because we are taking ownership of the responsibilty to clear the
    // analysis state.
    Arg.InnerAM = nullptr;
  }
  Result &operator=(Result &&RHS) {
    InnerAM = RHS.InnerAM;
    LI = RHS.LI;
    // We have to null out the analysis manager in the moved-from state
    // because we are taking ownership of the responsibilty to clear the
    // analysis state.
    RHS.InnerAM = nullptr;
    return *this;
  }
  ~Result() {
    // InnerAM is cleared in a moved from state where there is nothing to do.
    if (!InnerAM)
      return;

    // Clear out the analysis manager if we're being destroyed -- it means we
    // didn't even see an invalidate call when we got invalidated.
    InnerAM->clear();
  }

  /// Accessor for the analysis manager.
  LoopAnalysisManager &getManager() { return *InnerAM; }

  /// Handler for invalidation of the proxy for a particular function.
  ///
  /// If the proxy, \c LoopInfo, and associated analyses are preserved, this
  /// will merely forward the invalidation event to any cached loop analysis
  /// results for loops within this function.
  ///
  /// If the necessary loop infrastructure is not preserved, this will forcibly
  /// clear all of the cached analysis results that are keyed on the \c
  /// LoopInfo for this function.
  bool invalidate(Function &F, const PreservedAnalyses &PA,
                  FunctionAnalysisManager::Invalidator &Inv);

private:
  LoopAnalysisManager *InnerAM;
  LoopInfo *LI;
};

/// Provide a specialized run method for the \c LoopAnalysisManagerFunctionProxy
/// so it can pass the \c LoopInfo to the result.
template <>
LoopAnalysisManagerFunctionProxy::Result
LoopAnalysisManagerFunctionProxy::run(Function &F, FunctionAnalysisManager &AM);

// Ensure the \c LoopAnalysisManagerFunctionProxy is provided as an extern
// template.
extern template class InnerAnalysisManagerProxy<LoopAnalysisManager, Function>;

extern template class OuterAnalysisManagerProxy<FunctionAnalysisManager, Loop,
                                                LoopStandardAnalysisResults &>;
/// A proxy from a \c FunctionAnalysisManager to a \c Loop.
typedef OuterAnalysisManagerProxy<FunctionAnalysisManager, Loop,
                                  LoopStandardAnalysisResults &>
    FunctionAnalysisManagerLoopProxy;

/// Returns the minimum set of Analyses that all loop passes must preserve.
PreservedAnalyses getLoopPassPreservedAnalyses();

namespace internal {
/// Helper to implement appending of loops onto a worklist.
///
/// We want to process loops in postorder, but the worklist is a LIFO data
/// structure, so we append to it in *reverse* postorder.
///
/// For trees, a preorder traversal is a viable reverse postorder, so we
/// actually append using a preorder walk algorithm.
template <typename RangeT>
inline void appendLoopsToWorklist(RangeT &&Loops,
                                  SmallPriorityWorklist<Loop *, 4> &Worklist) {
  // We use an internal worklist to build up the preorder traversal without
  // recursion.
  SmallVector<Loop *, 4> PreOrderLoops, PreOrderWorklist;

  // We walk the initial sequence of loops in reverse because we generally want
  // to visit defs before uses and the worklist is LIFO.
  for (Loop *RootL : reverse(Loops)) {
    assert(PreOrderLoops.empty() && "Must start with an empty preorder walk.");
    assert(PreOrderWorklist.empty() &&
           "Must start with an empty preorder walk worklist.");
    PreOrderWorklist.push_back(RootL);
    do {
      Loop *L = PreOrderWorklist.pop_back_val();
      PreOrderWorklist.append(L->begin(), L->end());
      PreOrderLoops.push_back(L);
    } while (!PreOrderWorklist.empty());

    Worklist.insert(std::move(PreOrderLoops));
    PreOrderLoops.clear();
  }
}
}

/// The adaptor from a function pass to a loop pass directly computes
/// a standard set of analyses that are especially useful to loop passes and
/// makes them available in the API. Loop passes are also expected to update
/// all of these so that they remain correct across the entire loop pipeline.
struct LoopStandardAnalysisResults {
  AAResults &AA;
  AssumptionCache &AC;
  DominatorTree &DT;
  LoopInfo &LI;
  ScalarEvolution &SE;
  TargetLibraryInfo &TLI;
  TargetTransformInfo &TTI;
};

template <typename LoopPassT> class FunctionToLoopPassAdaptor;

/// This class provides an interface for updating the loop pass manager based
/// on mutations to the loop nest.
///
/// A reference to an instance of this class is passed as an argument to each
/// Loop pass, and Loop passes should use it to update LPM infrastructure if
/// they modify the loop nest structure.
class LPMUpdater {
public:
  /// This can be queried by loop passes which run other loop passes (like pass
  /// managers) to know whether the loop needs to be skipped due to updates to
  /// the loop nest.
  ///
  /// If this returns true, the loop object may have been deleted, so passes
  /// should take care not to touch the object.
  bool skipCurrentLoop() const { return SkipCurrentLoop; }

  /// Loop passes should use this method to indicate they have deleted a loop
  /// from the nest.
  ///
  /// Note that this loop must either be the current loop or a subloop of the
  /// current loop. This routine must be called prior to removing the loop from
  /// the loop nest.
  ///
  /// If this is called for the current loop, in addition to clearing any
  /// state, this routine will mark that the current loop should be skipped by
  /// the rest of the pass management infrastructure.
  void markLoopAsDeleted(Loop &L) {
    LAM.clear(L);
    assert(CurrentL->contains(&L) && "Cannot delete a loop outside of the "
                                     "subloop tree currently being processed.");
    if (&L == CurrentL)
      SkipCurrentLoop = true;
  }

  /// Loop passes should use this method to indicate they have added new child
  /// loops of the current loop.
  ///
  /// \p NewChildLoops must contain only the immediate children. Any nested
  /// loops within them will be visited in postorder as usual for the loop pass
  /// manager.
  void addChildLoops(ArrayRef<Loop *> NewChildLoops) {
    // Insert ourselves back into the worklist first, as this loop should be
    // revisited after all the children have been processed.
    Worklist.insert(CurrentL);

#ifndef NDEBUG
    for (Loop *NewL : NewChildLoops)
      assert(NewL->getParentLoop() == CurrentL && "All of the new loops must "
                                                  "be immediate children of "
                                                  "the current loop!");
#endif

    internal::appendLoopsToWorklist(NewChildLoops, Worklist);

    // Also skip further processing of the current loop--it will be revisited
    // after all of its newly added children are accounted for.
    SkipCurrentLoop = true;
  }

  /// Loop passes should use this method to indicate they have added new
  /// sibling loops to the current loop.
  ///
  /// \p NewSibLoops must only contain the immediate sibling loops. Any nested
  /// loops within them will be visited in postorder as usual for the loop pass
  /// manager.
  void addSiblingLoops(ArrayRef<Loop *> NewSibLoops) {
#ifndef NDEBUG
    for (Loop *NewL : NewSibLoops)
      assert(NewL->getParentLoop() == ParentL &&
             "All of the new loops must be siblings of the current loop!");
#endif

    internal::appendLoopsToWorklist(NewSibLoops, Worklist);

    // No need to skip the current loop or revisit it, as sibling loops
    // shouldn't impact anything.
  }

private:
  template <typename LoopPassT> friend class llvm::FunctionToLoopPassAdaptor;

  /// The \c FunctionToLoopPassAdaptor's worklist of loops to process.
  SmallPriorityWorklist<Loop *, 4> &Worklist;

  /// The analysis manager for use in the current loop nest.
  LoopAnalysisManager &LAM;

  Loop *CurrentL;
  bool SkipCurrentLoop;

#ifndef NDEBUG
  // In debug builds we also track the parent loop to implement asserts even in
  // the face of loop deletion.
  Loop *ParentL;
#endif

  LPMUpdater(SmallPriorityWorklist<Loop *, 4> &Worklist,
             LoopAnalysisManager &LAM)
      : Worklist(Worklist), LAM(LAM) {}
};

/// \brief Adaptor that maps from a function to its loops.
///
/// Designed to allow composition of a LoopPass(Manager) and a
/// FunctionPassManager. Note that if this pass is constructed with a \c
/// FunctionAnalysisManager it will run the \c LoopAnalysisManagerFunctionProxy
/// analysis prior to running the loop passes over the function to enable a \c
/// LoopAnalysisManager to be used within this run safely.
template <typename LoopPassT>
class FunctionToLoopPassAdaptor
    : public PassInfoMixin<FunctionToLoopPassAdaptor<LoopPassT>> {
public:
  explicit FunctionToLoopPassAdaptor(LoopPassT Pass)
      : Pass(std::move(Pass)) {}

  /// \brief Runs the loop passes across every loop in the function.
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM) {
    // Setup the loop analysis manager from its proxy.
    LoopAnalysisManager &LAM =
        AM.getResult<LoopAnalysisManagerFunctionProxy>(F).getManager();
    // Get the loop structure for this function
    LoopInfo &LI = AM.getResult<LoopAnalysis>(F);

    // If there are no loops, there is nothing to do here.
    if (LI.empty())
      return PreservedAnalyses::all();

    // Get the analysis results needed by loop passes.
    LoopStandardAnalysisResults LAR = {AM.getResult<AAManager>(F),
                                       AM.getResult<AssumptionAnalysis>(F),
                                       AM.getResult<DominatorTreeAnalysis>(F),
                                       AM.getResult<LoopAnalysis>(F),
                                       AM.getResult<ScalarEvolutionAnalysis>(F),
                                       AM.getResult<TargetLibraryAnalysis>(F),
                                       AM.getResult<TargetIRAnalysis>(F)};

    PreservedAnalyses PA = PreservedAnalyses::all();

    // A postorder worklist of loops to process.
    SmallPriorityWorklist<Loop *, 4> Worklist;

    // Register the worklist and loop analysis manager so that loop passes can
    // update them when they mutate the loop nest structure.
    LPMUpdater Updater(Worklist, LAM);

    // Add the loop nests in the reverse order of LoopInfo. For some reason,
    // they are stored in RPO w.r.t. the control flow graph in LoopInfo. For
    // the purpose of unrolling, loop deletion, and LICM, we largely want to
    // work forward across the CFG so that we visit defs before uses and can
    // propagate simplifications from one loop nest into the next.
    // FIXME: Consider changing the order in LoopInfo.
    internal::appendLoopsToWorklist(reverse(LI), Worklist);

    do {
      Loop *L = Worklist.pop_back_val();

      // Reset the update structure for this loop.
      Updater.CurrentL = L;
      Updater.SkipCurrentLoop = false;
#ifndef NDEBUG
      Updater.ParentL = L->getParentLoop();
#endif

      PreservedAnalyses PassPA = Pass.run(*L, LAM, LAR, Updater);
      // FIXME: We should verify the set of analyses relevant to Loop passes
      // are preserved.

      // If the loop hasn't been deleted, we need to handle invalidation here.
      if (!Updater.skipCurrentLoop())
        // We know that the loop pass couldn't have invalidated any other
        // loop's analyses (that's the contract of a loop pass), so directly
        // handle the loop analysis manager's invalidation here.
        LAM.invalidate(*L, PassPA);

      // Then intersect the preserved set so that invalidation of module
      // analyses will eventually occur when the module pass completes.
      PA.intersect(std::move(PassPA));
    } while (!Worklist.empty());

    // By definition we preserve the proxy. We also preserve all analyses on
    // Loops. This precludes *any* invalidation of loop analyses by the proxy,
    // but that's OK because we've taken care to invalidate analyses in the
    // loop analysis manager incrementally above.
    PA.preserveSet<AllAnalysesOn<Loop>>();
    PA.preserve<LoopAnalysisManagerFunctionProxy>();
    // We also preserve the set of standard analyses.
    PA.preserve<AssumptionAnalysis>();
    PA.preserve<DominatorTreeAnalysis>();
    PA.preserve<LoopAnalysis>();
    PA.preserve<ScalarEvolutionAnalysis>();
    // FIXME: What we really want to do here is preserve an AA category, but
    // that concept doesn't exist yet.
    PA.preserve<AAManager>();
    PA.preserve<BasicAA>();
    PA.preserve<GlobalsAA>();
    PA.preserve<SCEVAA>();
    return PA;
  }

private:
  LoopPassT Pass;
};

/// \brief A function to deduce a loop pass type and wrap it in the templated
/// adaptor.
template <typename LoopPassT>
FunctionToLoopPassAdaptor<LoopPassT>
createFunctionToLoopPassAdaptor(LoopPassT Pass) {
  return FunctionToLoopPassAdaptor<LoopPassT>(std::move(Pass));
}

/// \brief Pass for printing a loop's contents as textual IR.
class PrintLoopPass : public PassInfoMixin<PrintLoopPass> {
  raw_ostream &OS;
  std::string Banner;

public:
  PrintLoopPass();
  PrintLoopPass(raw_ostream &OS, const std::string &Banner = "");

  PreservedAnalyses run(Loop &L, LoopAnalysisManager &,
                        LoopStandardAnalysisResults &, LPMUpdater &);
};
}

#endif // LLVM_ANALYSIS_LOOPPASSMANAGER_H
