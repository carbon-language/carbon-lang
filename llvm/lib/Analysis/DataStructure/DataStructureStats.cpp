//===- DSGraphStats.cpp - Various statistics for DS Graphs -----*- C++ -*--===//
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/DataStructure.h"
#include "llvm/Analysis/DSGraph.h"
#include "llvm/Function.h"
#include "llvm/iOther.h"
#include "llvm/Pass.h"
#include "Support/Statistic.h"
#include <vector>

namespace {
  Statistic<> TotalNumCallees("totalcallees",
                "Total number of callee functions at all indirect call sites");
  Statistic<> NumIndirectCalls("numindirect",
                "Total number of indirect call sites in the program");
  Statistic<> NumPoolNodes("numpools",
                  "Number of allocation nodes that could be pool allocated");

  class DSGraphStats: public FunctionPass {
    void countCallees(const Function& F, const DSGraph& tdGraph);

  public:
    /// Driver functions to compute the Load/Store Dep. Graph per function.
    bool runOnFunction(Function& F);

    /// getAnalysisUsage - This modify nothing, and uses the Top-Down Graph.
    void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.setPreservesAll();
      AU.addRequired<TDDataStructures>();
    }

    /// Debugging support methods
    void print(std::ostream &O) const { }
    void dump() const;
  };

  static RegisterAnalysis<DSGraphStats> Z("dsstats", "DS Graph Statistics");
}

static bool isIndirectCallee(Value *V) {
  if (isa<Function>(V)) return false;

  if (CastInst *CI = dyn_cast<CastInst>(V))
    return isIndirectCallee(CI->getOperand(0));
  return true;
}


void DSGraphStats::countCallees(const Function& F, const DSGraph& tdGraph) {
  unsigned numIndirectCalls = 0, totalNumCallees = 0;

  const std::vector<DSCallSite>& callSites = tdGraph.getFunctionCalls();
  for (unsigned i=0, N = callSites.size(); i < N; ++i)
    if (isIndirectCallee(callSites[i].getCallInst().getCalledValue()))
      { // This is an indirect function call
        std::vector<GlobalValue*> Callees =
          callSites[i].getCalleeNode()->getGlobals();
        if (Callees.size() > 0) {
          totalNumCallees  += Callees.size();
          ++numIndirectCalls;
        }
#ifndef NDEBUG
        else
          std::cerr << "WARNING: No callee in Function " << F.getName()
                      << "at call:\n" << callSites[i].getCallInst();
#endif
      }

  TotalNumCallees  += totalNumCallees;
  NumIndirectCalls += numIndirectCalls;

  if (numIndirectCalls)
    std::cout << "  In function " << F.getName() << ":  "
              << (totalNumCallees / (double) numIndirectCalls)
              << " average callees per indirect call\n";
}


bool DSGraphStats::runOnFunction(Function& F) {
  const DSGraph& tdGraph = getAnalysis<TDDataStructures>().getDSGraph(F);
  countCallees(F, tdGraph);
  return true;
}

void DSGraphStats::dump() const
{
  this->print(std::cerr);
}
