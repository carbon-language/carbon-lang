//===- CallPrinter.cpp - DOT printer for call graph -----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines '-dot-callgraph', which emit a callgraph.<fnname>.dot
// containing the call graph of a module.
//
// There is also a pass available to directly call dotty ('-view-callgraph').
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/CallPrinter.h"

#include "llvm/Analysis/BlockFrequencyInfo.h"
#include "llvm/Analysis/BranchProbabilityInfo.h"
#include "llvm/Analysis/CallGraph.h"
#include "llvm/Analysis/DOTGraphTraitsPass.h"
#include "llvm/Analysis/HeatUtils.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallSet.h"

using namespace llvm;

static cl::opt<bool> ShowHeatColors("callgraph-heat-colors", cl::init(true),
                                    cl::Hidden,
                                    cl::desc("Show heat colors in call-graph"));

static cl::opt<bool>
    EstimateEdgeWeight("callgraph-weights", cl::init(false), cl::Hidden,
                       cl::desc("Show edges labeled with weights"));

static cl::opt<bool>
    FullCallGraph("callgraph-full", cl::init(false), cl::Hidden,
                  cl::desc("Show full call-graph (including external nodes)"));

static cl::opt<bool> UseCallCounter(
    "callgraph-call-count", cl::init(false), cl::Hidden,
    cl::desc("Use function's call counter as a heat metric. "
             "The default is the function's maximum block frequency."));

namespace llvm {

class CallGraphDOTInfo {
private:
  Module *M;
  CallGraph *CG;
  DenseMap<const Function *, uint64_t> Freq;
  uint64_t MaxFreq;
  uint64_t MaxEdgeCount;

public:
  std::function<BlockFrequencyInfo *(Function &)> LookupBFI;

  CallGraphDOTInfo(Module *M, CallGraph *CG,
                   function_ref<BlockFrequencyInfo *(Function &)> LookupBFI)
      : M(M), CG(CG), LookupBFI(LookupBFI) {
    MaxFreq = 0;
    MaxEdgeCount = 0;

    for (Function &F : *M) {
      Freq[&F] = 0;

      if (FullCallGraph) {
        for (User *U : F.users()) {
          auto CS = CallSite(U);
          if (!CS.getCaller()->isDeclaration()) {
            uint64_t Counter = getNumOfCalls(CS, LookupBFI);
            if (Counter > MaxEdgeCount) {
              MaxEdgeCount = Counter;
            }
          }
        }
      }

      if (F.isDeclaration())
        continue;
      uint64_t localMaxFreq = 0;
      if (UseCallCounter) {
        Function::ProfileCount EntryCount = F.getEntryCount();
        if (EntryCount.hasValue())
          localMaxFreq = EntryCount.getCount();
      } else {
        localMaxFreq = llvm::getMaxFreq(F, LookupBFI(F));
      }
      if (localMaxFreq >= MaxFreq)
        MaxFreq = localMaxFreq;
      Freq[&F] = localMaxFreq;

      if (!FullCallGraph) {
        for (Function &Callee : *M) {
          uint64_t Counter = getNumOfCalls(F, Callee, LookupBFI);
          if (Counter > MaxEdgeCount) {
            MaxEdgeCount = Counter;
          }
        }
      }
    }
    if (!FullCallGraph)
      removeParallelEdges();
  }

  Module *getModule() const { return M; }
  CallGraph *getCallGraph() const { return CG; }

  uint64_t getFreq(const Function *F) { return Freq[F]; }

  uint64_t getMaxFreq() { return MaxFreq; }

  uint64_t getMaxEdgeCount() { return MaxEdgeCount; }

private:
  void removeParallelEdges() {
    for (auto &I : (*CG)) {
      CallGraphNode *Node = I.second.get();

      bool FoundParallelEdge = true;
      while (FoundParallelEdge) {
        SmallSet<Function *, 16> Visited;
        FoundParallelEdge = false;
        for (auto CI = Node->begin(), CE = Node->end(); CI != CE; CI++) {
          if (!Visited.count(CI->second->getFunction()))
            Visited.insert(CI->second->getFunction());
          else {
            FoundParallelEdge = true;
            Node->removeCallEdge(CI);
            break;
          }
        }
      }
    }
  }
};

template <>
struct GraphTraits<CallGraphDOTInfo *>
    : public GraphTraits<const CallGraphNode *> {
  static NodeRef getEntryNode(CallGraphDOTInfo *CGInfo) {
    // Start at the external node!
    return CGInfo->getCallGraph()->getExternalCallingNode();
  }

  typedef std::pair<const Function *const, std::unique_ptr<CallGraphNode>>
      PairTy;
  static const CallGraphNode *CGGetValuePtr(const PairTy &P) {
    return P.second.get();
  }

  // nodes_iterator/begin/end - Allow iteration over all nodes in the graph
  typedef mapped_iterator<CallGraph::const_iterator, decltype(&CGGetValuePtr)>
      nodes_iterator;

  static nodes_iterator nodes_begin(CallGraphDOTInfo *CGInfo) {
    return nodes_iterator(CGInfo->getCallGraph()->begin(), &CGGetValuePtr);
  }
  static nodes_iterator nodes_end(CallGraphDOTInfo *CGInfo) {
    return nodes_iterator(CGInfo->getCallGraph()->end(), &CGGetValuePtr);
  }
};

template <>
struct DOTGraphTraits<CallGraphDOTInfo *> : public DefaultDOTGraphTraits {

  SmallSet<User *, 16> VisitedCallSites;

  DOTGraphTraits(bool isSimple = false) : DefaultDOTGraphTraits(isSimple) {}

  static std::string getGraphName(CallGraphDOTInfo *CGInfo) {
    return "Call graph: " +
           std::string(CGInfo->getModule()->getModuleIdentifier());
  }

  static bool isNodeHidden(const CallGraphNode *Node) {
    if (FullCallGraph)
      return false;

    if (Node->getFunction())
      return false;

    return true;
  }

  std::string getNodeLabel(const CallGraphNode *Node,
                           CallGraphDOTInfo *CGInfo) {
    if (Node == CGInfo->getCallGraph()->getExternalCallingNode())
      return "external caller";

    if (Node == CGInfo->getCallGraph()->getCallsExternalNode())
      return "external callee";

    if (Function *Func = Node->getFunction())
      return Func->getName();

    return "external node";
  }

  static const CallGraphNode *CGGetValuePtr(CallGraphNode::CallRecord P) {
    return P.second;
  }

  // nodes_iterator/begin/end - Allow iteration over all nodes in the graph
  typedef mapped_iterator<CallGraphNode::const_iterator,
                          decltype(&CGGetValuePtr)>
      nodes_iterator;

  std::string getEdgeAttributes(const CallGraphNode *Node, nodes_iterator I,
                                CallGraphDOTInfo *CGInfo) {
    if (!EstimateEdgeWeight)
      return "";

    Function *Caller = Node->getFunction();
    if (Caller == nullptr || Caller->isDeclaration())
      return "";

    Function *Callee = (*I)->getFunction();
    if (Callee == nullptr)
      return "";

    uint64_t Counter = 0;
    if (FullCallGraph) {
      // looks for next call site between Caller and Callee
      for (User *U : Callee->users()) {
        auto CS = CallSite(U);
        if (CS.getCaller() == Caller) {
          if (VisitedCallSites.count(U))
            continue;
          VisitedCallSites.insert(U);
          Counter = getNumOfCalls(CS, CGInfo->LookupBFI);
          break;
        }
      }
    } else {
      Counter = getNumOfCalls(*Caller, *Callee, CGInfo->LookupBFI);
    }

    const unsigned MaxEdgeWidth = 3;

    double Width =
        1 + (MaxEdgeWidth - 1) * (double(Counter) / CGInfo->getMaxEdgeCount());
    std::string Attrs = "label=\"" + std::to_string(Counter) +
                        "\" penwidth=" + std::to_string(Width);

    return Attrs;
  }

  std::string getNodeAttributes(const CallGraphNode *Node,
                                CallGraphDOTInfo *CGInfo) {
    Function *F = Node->getFunction();
    if (F == nullptr || F->isDeclaration())
      return "";

    std::string attrs = "";
    if (ShowHeatColors) {
      uint64_t freq = CGInfo->getFreq(F);
      std::string color = getHeatColor(freq, CGInfo->getMaxFreq());
      std::string edgeColor = (freq <= (CGInfo->getMaxFreq() / 2))
                                  ? getHeatColor(0)
                                  : getHeatColor(1);

      attrs = "color=\"" + edgeColor + "ff\", style=filled, fillcolor=\"" +
              color + "80\"";
    }
    return attrs;
  }
};

} // namespace llvm

namespace {

// Viewer

class CallGraphViewer : public ModulePass {
public:
  static char ID;
  CallGraphViewer() : ModulePass(ID) {}

  void getAnalysisUsage(AnalysisUsage &AU) const override;
  bool runOnModule(Module &M) override;
};

void CallGraphViewer::getAnalysisUsage(AnalysisUsage &AU) const {
  ModulePass::getAnalysisUsage(AU);
  AU.addRequired<BlockFrequencyInfoWrapperPass>();
  AU.setPreservesAll();
}

bool CallGraphViewer::runOnModule(Module &M) {
  auto LookupBFI = [this](Function &F) {
    return &this->getAnalysis<BlockFrequencyInfoWrapperPass>(F).getBFI();
  };

  CallGraph CG(M);
  CallGraphDOTInfo CFGInfo(&M, &CG, LookupBFI);

  std::string Title =
      DOTGraphTraits<CallGraphDOTInfo *>::getGraphName(&CFGInfo);
  ViewGraph(&CFGInfo, "callgraph", true, Title);

  return false;
}

// DOT Printer

class CallGraphDOTPrinter : public ModulePass {
public:
  static char ID;
  CallGraphDOTPrinter() : ModulePass(ID) {}

  void getAnalysisUsage(AnalysisUsage &AU) const override;
  bool runOnModule(Module &M) override;
};

void CallGraphDOTPrinter::getAnalysisUsage(AnalysisUsage &AU) const {
  ModulePass::getAnalysisUsage(AU);
  AU.addRequired<BlockFrequencyInfoWrapperPass>();
  AU.setPreservesAll();
}

bool CallGraphDOTPrinter::runOnModule(Module &M) {
  auto LookupBFI = [this](Function &F) {
    return &this->getAnalysis<BlockFrequencyInfoWrapperPass>(F).getBFI();
  };

  std::string Filename =
      (std::string(M.getModuleIdentifier()) + ".callgraph.dot");
  errs() << "Writing '" << Filename << "'...";

  std::error_code EC;
  raw_fd_ostream File(Filename, EC, sys::fs::F_Text);

  CallGraph CG(M);
  CallGraphDOTInfo CFGInfo(&M, &CG, LookupBFI);

  if (!EC)
    WriteGraph(File, &CFGInfo);
  else
    errs() << "  error opening file for writing!";
  errs() << "\n";

  return false;
}

} // end anonymous namespace

char CallGraphViewer::ID = 0;
INITIALIZE_PASS(CallGraphViewer, "view-callgraph", "View call graph", false,
                false)

char CallGraphDOTPrinter::ID = 0;
INITIALIZE_PASS(CallGraphDOTPrinter, "dot-callgraph",
                "Print call graph to 'dot' file", false, false)

// Create methods available outside of this file, to use them
// "include/llvm/LinkAllPasses.h". Otherwise the pass would be deleted by
// the link time optimization.

ModulePass *llvm::createCallGraphViewerPass() { return new CallGraphViewer(); }

ModulePass *llvm::createCallGraphDOTPrinterPass() {
  return new CallGraphDOTPrinter();
}
