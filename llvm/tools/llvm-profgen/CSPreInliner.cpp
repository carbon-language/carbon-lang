//===-- CSPreInliner.cpp - Profile guided preinliner -------------- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CSPreInliner.h"
#include "ProfiledBinary.h"
#include "llvm/ADT/SCCIterator.h"
#include <cstdint>
#include <queue>

#define DEBUG_TYPE "cs-preinliner"

using namespace llvm;
using namespace sampleprof;

// The switches specify inline thresholds used in SampleProfileLoader inlining.
// TODO: the actual threshold to be tuned here because the size here is based
// on machine code not LLVM IR.
extern cl::opt<int> SampleHotCallSiteThreshold;
extern cl::opt<int> SampleColdCallSiteThreshold;
extern cl::opt<int> ProfileInlineGrowthLimit;
extern cl::opt<int> ProfileInlineLimitMin;
extern cl::opt<int> ProfileInlineLimitMax;

cl::opt<bool> EnableCSPreInliner(
    "csspgo-preinliner", cl::Hidden, cl::init(false),
    cl::desc("Run a global pre-inliner to merge context profile based on "
             "estimated global top-down inline decisions"));

cl::opt<bool> UseContextCostForPreInliner(
    "use-context-cost-for-preinliner", cl::Hidden, cl::init(false),
    cl::desc("Use context-sensitive byte size cost for preinliner decisions"));

static cl::opt<bool> SamplePreInlineReplay(
    "csspgo-replay-preinline", cl::Hidden, cl::init(false),
    cl::desc(
        "Replay previous inlining and adjust context profile accordingly"));

CSPreInliner::CSPreInliner(StringMap<FunctionSamples> &Profiles,
                           ProfiledBinary &Binary, uint64_t HotThreshold,
                           uint64_t ColdThreshold)
    : UseContextCost(UseContextCostForPreInliner), ContextTracker(Profiles),
      ProfileMap(Profiles), Binary(Binary), HotCountThreshold(HotThreshold),
      ColdCountThreshold(ColdThreshold) {}

std::vector<StringRef> CSPreInliner::buildTopDownOrder() {
  std::vector<StringRef> Order;
  ProfiledCallGraph ProfiledCG(ContextTracker);

  // Now that we have a profiled call graph, construct top-down order
  // by building up SCC and reversing SCC order.
  scc_iterator<ProfiledCallGraph *> I = scc_begin(&ProfiledCG);
  while (!I.isAtEnd()) {
    for (ProfiledCallGraphNode *Node : *I) {
      if (Node != ProfiledCG.getEntryNode())
        Order.push_back(Node->Name);
    }
    ++I;
  }
  std::reverse(Order.begin(), Order.end());

  return Order;
}

bool CSPreInliner::getInlineCandidates(ProfiledCandidateQueue &CQueue,
                                       const FunctionSamples *CallerSamples) {
  assert(CallerSamples && "Expect non-null caller samples");

  // Ideally we want to consider everything a function calls, but as far as
  // context profile is concerned, only those frames that are children of
  // current one in the trie is relavent. So we walk the trie instead of call
  // targets from function profile.
  ContextTrieNode *CallerNode =
      ContextTracker.getContextFor(CallerSamples->getContext());

  bool HasNewCandidate = false;
  for (auto &Child : CallerNode->getAllChildContext()) {
    ContextTrieNode *CalleeNode = &Child.second;
    FunctionSamples *CalleeSamples = CalleeNode->getFunctionSamples();
    if (!CalleeSamples)
      continue;

    // Call site count is more reliable, so we look up the corresponding call
    // target profile in caller's context profile to retrieve call site count.
    uint64_t CalleeEntryCount = CalleeSamples->getEntrySamples();
    uint64_t CallsiteCount = 0;
    LineLocation Callsite = CalleeNode->getCallSiteLoc();
    if (auto CallTargets = CallerSamples->findCallTargetMapAt(Callsite)) {
      SampleRecord::CallTargetMap &TargetCounts = CallTargets.get();
      auto It = TargetCounts.find(CalleeSamples->getName());
      if (It != TargetCounts.end())
        CallsiteCount = It->second;
    }

    // TODO: call site and callee entry count should be mostly consistent, add
    // check for that.
    HasNewCandidate = true;
    uint32_t CalleeSize = getFuncSize(*CalleeSamples);
    CQueue.emplace(CalleeSamples, std::max(CallsiteCount, CalleeEntryCount),
                   CalleeSize);
  }

  return HasNewCandidate;
}

uint32_t CSPreInliner::getFuncSize(const FunctionSamples &FSamples) {
  if (UseContextCost) {
    return Binary.getFuncSizeForContext(FSamples.getContext());
  }

  return FSamples.getBodySamples().size();
}

bool CSPreInliner::shouldInline(ProfiledInlineCandidate &Candidate) {
  // If replay inline is requested, simply follow the inline decision of the
  // profiled binary.
  if (SamplePreInlineReplay)
    return Candidate.CalleeSamples->getContext().hasAttribute(
        ContextWasInlined);

  // Adjust threshold based on call site hotness, only do this for callsite
  // prioritized inliner because otherwise cost-benefit check is done earlier.
  unsigned int SampleThreshold = SampleColdCallSiteThreshold;
  if (Candidate.CallsiteCount > HotCountThreshold)
    SampleThreshold = SampleHotCallSiteThreshold;

  // TODO: for small cold functions, we may inlined them and we need to keep
  // context profile accordingly.
  if (Candidate.CallsiteCount < ColdCountThreshold)
    SampleThreshold = SampleColdCallSiteThreshold;

  return (Candidate.SizeCost < SampleThreshold);
}

void CSPreInliner::processFunction(const StringRef Name) {
  FunctionSamples *FSamples = ContextTracker.getBaseSamplesFor(Name);
  if (!FSamples)
    return;

  unsigned FuncSize = getFuncSize(*FSamples);
  unsigned FuncFinalSize = FuncSize;
  unsigned SizeLimit = FuncSize * ProfileInlineGrowthLimit;
  SizeLimit = std::min(SizeLimit, (unsigned)ProfileInlineLimitMax);
  SizeLimit = std::max(SizeLimit, (unsigned)ProfileInlineLimitMin);

  LLVM_DEBUG(dbgs() << "Process " << Name
                    << " for context-sensitive pre-inlining (pre-inline size: "
                    << FuncSize << ", size limit: " << SizeLimit << ")\n");

  ProfiledCandidateQueue CQueue;
  getInlineCandidates(CQueue, FSamples);

  while (!CQueue.empty() && FuncFinalSize < SizeLimit) {
    ProfiledInlineCandidate Candidate = CQueue.top();
    CQueue.pop();
    bool ShouldInline = false;
    if ((ShouldInline = shouldInline(Candidate))) {
      // We mark context as inlined as the corresponding context profile
      // won't be merged into that function's base profile.
      ContextTracker.markContextSamplesInlined(Candidate.CalleeSamples);
      Candidate.CalleeSamples->getContext().setAttribute(
          ContextShouldBeInlined);
      FuncFinalSize += Candidate.SizeCost;
      getInlineCandidates(CQueue, Candidate.CalleeSamples);
    }
    LLVM_DEBUG(dbgs() << (ShouldInline ? "  Inlined" : "  Outlined")
                      << " context profile for: "
                      << Candidate.CalleeSamples->getNameWithContext()
                      << " (callee size: " << Candidate.SizeCost
                      << ", call count:" << Candidate.CallsiteCount << ")\n");
  }

  LLVM_DEBUG({
    if (!CQueue.empty())
      dbgs() << "  Inline candidates ignored due to size limit (inliner "
                "original size: "
             << FuncSize << ", inliner final size: " << FuncFinalSize
             << ", size limit: " << SizeLimit << ")\n";

    while (!CQueue.empty()) {
      ProfiledInlineCandidate Candidate = CQueue.top();
      CQueue.pop();
      bool WasInlined =
          Candidate.CalleeSamples->getContext().hasAttribute(ContextWasInlined);
      dbgs() << "    " << Candidate.CalleeSamples->getNameWithContext()
             << " (candidate size:" << Candidate.SizeCost
             << ", call count: " << Candidate.CallsiteCount << ", previously "
             << (WasInlined ? "inlined)\n" : "not inlined)\n");
    }
  });
}

void CSPreInliner::run() {
#ifndef NDEBUG
  auto printProfileNames = [](StringMap<FunctionSamples> &Profiles,
                              bool IsInput) {
    dbgs() << (IsInput ? "Input" : "Output") << " context-sensitive profiles ("
           << Profiles.size() << " total):\n";
    for (auto &It : Profiles) {
      const FunctionSamples &Samples = It.second;
      dbgs() << "  [" << Samples.getNameWithContext() << "] "
             << Samples.getTotalSamples() << ":" << Samples.getHeadSamples()
             << "\n";
    }
  };
#endif

  LLVM_DEBUG(printProfileNames(ProfileMap, true));

  // Execute global pre-inliner to estimate a global top-down inline
  // decision and merge profiles accordingly. This helps with profile
  // merge for ThinLTO otherwise we won't be able to merge profiles back
  // to base profile across module/thin-backend boundaries.
  // It also helps better compress context profile to control profile
  // size, as we now only need context profile for functions going to
  // be inlined.
  for (StringRef FuncName : buildTopDownOrder()) {
    processFunction(FuncName);
  }

  // Not inlined context profiles are merged into its base, so we can
  // trim out such profiles from the output.
  std::vector<StringRef> ProfilesToBeRemoved;
  for (auto &It : ProfileMap) {
    SampleContext Context = It.second.getContext();
    if (!Context.isBaseContext() && !Context.hasState(InlinedContext)) {
      assert(Context.hasState(MergedContext) &&
             "Not inlined context profile should be merged already");
      ProfilesToBeRemoved.push_back(It.first());
    }
  }

  for (StringRef ContextName : ProfilesToBeRemoved) {
    ProfileMap.erase(ContextName);
  }

  // Make sure ProfileMap's key is consistent with FunctionSamples' name.
  SampleContextTrimmer(ProfileMap).canonicalizeContextProfiles();

  LLVM_DEBUG(printProfileNames(ProfileMap, false));
}
