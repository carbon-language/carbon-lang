//===- DSGraphStats.cpp - Various statistics for DS Graphs ----------------===//
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/DataStructure.h"
#include "llvm/Analysis/DSGraph.h"
#include "llvm/Function.h"
#include "llvm/iOther.h"
#include "llvm/iMemory.h"
#include "llvm/Pass.h"
#include "llvm/Support/InstVisitor.h"
#include "Support/Statistic.h"
#include <vector>

namespace {
  Statistic<> TotalNumCallees("totalcallees",
                "Total number of callee functions at all indirect call sites");
  Statistic<> NumIndirectCalls("numindirect",
                "Total number of indirect call sites in the program");
  Statistic<> NumPoolNodes("numpools",
                  "Number of allocation nodes that could be pool allocated");

  // Typed/Untyped memory accesses: If DSA can infer that the types the loads
  // and stores are accessing are correct (ie, the node has not been collapsed),
  // increment the appropriate counter.
  Statistic<> NumTypedMemAccesses("numtypedmemaccesses",
                                "Number of loads/stores which are fully typed");
  Statistic<> NumUntypedMemAccesses("numuntypedmemaccesses",
                                "Number of loads/stores which are untyped");

  class DSGraphStats : public FunctionPass, public InstVisitor<DSGraphStats> {
    void countCallees(const Function &F);
    const DSGraph *TDGraph;

    DSNode *getNodeForValue(Value *V);
    bool isNodeForValueCollapsed(Value *V);
  public:
    /// Driver functions to compute the Load/Store Dep. Graph per function.
    bool runOnFunction(Function& F);

    /// getAnalysisUsage - This modify nothing, and uses the Top-Down Graph.
    void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.setPreservesAll();
      AU.addRequired<TDDataStructures>();
    }

    void visitLoad(LoadInst &LI);
    void visitStore(StoreInst &SI);

    /// Debugging support methods
    void print(std::ostream &O) const { }
  };

  static RegisterAnalysis<DSGraphStats> Z("dsstats", "DS Graph Statistics");
}

static bool isIndirectCallee(Value *V) {
  if (isa<Function>(V)) return false;

  if (CastInst *CI = dyn_cast<CastInst>(V))
    return isIndirectCallee(CI->getOperand(0));
  return true;
}


void DSGraphStats::countCallees(const Function& F) {
  unsigned numIndirectCalls = 0, totalNumCallees = 0;

  const std::vector<DSCallSite> &callSites = TDGraph->getFunctionCalls();
  for (unsigned i = 0, N = callSites.size(); i != N; ++i)
    if (isIndirectCallee(callSites[i].getCallSite().getCalledValue())) {
      // This is an indirect function call
      const std::vector<GlobalValue*> &Callees =
        callSites[i].getCalleeNode()->getGlobals();
      if (Callees.size() > 0) {
        totalNumCallees  += Callees.size();
        ++numIndirectCalls;
      } else
        std::cerr << "WARNING: No callee in Function '" << F.getName()
                  << "' at call: \n"
                  << *callSites[i].getCallSite().getInstruction();
    }
  
  TotalNumCallees  += totalNumCallees;
  NumIndirectCalls += numIndirectCalls;
  
  if (numIndirectCalls)
    std::cout << "  In function " << F.getName() << ":  "
              << (totalNumCallees / (double) numIndirectCalls)
              << " average callees per indirect call\n";
}

DSNode *DSGraphStats::getNodeForValue(Value *V) {
  const DSGraph *G = TDGraph;
  if (isa<GlobalValue>(V) || isa<Constant>(V))
    G = TDGraph->getGlobalsGraph();

  const DSGraph::ScalarMapTy &ScalarMap = G->getScalarMap();
  DSGraph::ScalarMapTy::const_iterator I = ScalarMap.find(V);
  if (I != ScalarMap.end())
    return I->second.getNode();
  return 0;
}

bool DSGraphStats::isNodeForValueCollapsed(Value *V) {
  if (DSNode *N = getNodeForValue(V))
    return N->isNodeCompletelyFolded() || N->isIncomplete();
  return false;
}

void DSGraphStats::visitLoad(LoadInst &LI) {
  if (isNodeForValueCollapsed(LI.getOperand(0))) {
    NumUntypedMemAccesses++;
  } else {
    NumTypedMemAccesses++;
  }
}

void DSGraphStats::visitStore(StoreInst &SI) {
  if (isNodeForValueCollapsed(SI.getOperand(1))) {
    NumUntypedMemAccesses++;
  } else {
    NumTypedMemAccesses++;
  }
}



bool DSGraphStats::runOnFunction(Function& F) {
  TDGraph = &getAnalysis<TDDataStructures>().getDSGraph(F);
  countCallees(F);
  visit(F);
  return true;
}
